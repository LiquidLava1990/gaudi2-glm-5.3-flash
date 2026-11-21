"""The runner calls the three decode hooks of glm53_mixed_graph where the decode lattice, the decode shard and the decode rank check need them.

Every hook is tested against a fake runner elsewhere (test_decode_lattice.py, test_rank_check_decode.py), so a runner that stops calling one, or calls it and
drops the result, passes all of those. This file reads the runner's own source (the file the image runs) and checks the calls and the order they are in:
  decode_table_width    in the sparse metadata builder, for a decode step only, and its result is the block width the tables are built with;
  decode_key_metadata   in the generic forward, on the trimmed metadata, and its result is what the model is called with;
  observe_decode_hidden in the generic forward, for a decode step only, on the model's own hidden states, after the forward and before the logits.
The mutants below remove or bypass each call in a copy of the source and expect a problem to be named.
"""
from __future__ import annotations

import ast
import inspect
import unittest
from collections.abc import Iterator
from pathlib import Path

from overlay_testlib import OVERLAY_ROOT, load_overlay

RUNNER = OVERLAY_ROOT / "vllm_gaudi" / "v1" / "worker" / "hpu_model_runner.py"
METADATA_FUNCTION, FORWARD_FUNCTION = "_make_glm53_sparse_metadata_cpu", "_execute_model_generic"
HOOK_MODULE = "glm53_mixed_graph"


def _function(tree: ast.AST, name: str) -> ast.FunctionDef | None:
    return next((node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name == name), None)


def _calls(scope: ast.AST, name: str) -> list[ast.Call]:
    return [node for node in ast.walk(scope) if isinstance(node, ast.Call)
            and (getattr(node.func, "id", None) == name or getattr(node.func, "attr", None) == name)]


def _parents(scope: ast.AST) -> dict[ast.AST, ast.AST]:
    return {child: parent for parent in ast.walk(scope) for child in ast.iter_child_nodes(parent)}


def _names(call: ast.Call) -> list[str]:
    return [argument.id for argument in call.args if isinstance(argument, ast.Name)]


def _imports(scope: ast.AST) -> set[str]:
    return {alias.name for node in ast.walk(scope) if isinstance(node, ast.ImportFrom) and node.module == HOOK_MODULE for alias in node.names}


def _guard(call: ast.Call, parents: dict[ast.AST, ast.AST]) -> str:
    """The source of the ``if`` test the call sits under, or ''."""
    node = parents.get(call)
    while node is not None and not isinstance(node, ast.If):
        node = parents.get(node)
    return ast.unparse(node.test) if node is not None else ""


def _only_call(scope: ast.AST, name: str, where: str, problems: list[str]) -> ast.Call | None:
    found = _calls(scope, name)
    if len(found) != 1:
        problems.append(f"{where}: {len(found)} calls to {name}, want exactly 1")
        return None
    return found[0]


def table_width_problems(tree: ast.AST) -> list[str]:
    scope = _function(tree, METADATA_FUNCTION)
    if scope is None:
        return [f"{METADATA_FUNCTION} is gone: find where the sparse block width is built and re-point this file"]
    problems: list[str] = []
    if "decode_table_width" not in _imports(scope):
        problems.append(f"{METADATA_FUNCTION} does not import decode_table_width from {HOOK_MODULE}")
    call = _only_call(scope, "decode_table_width", METADATA_FUNCTION, problems)
    if call is None:
        return problems
    parents = _parents(scope)
    assign = parents.get(call)
    if not (isinstance(assign, ast.Assign) and [target.id for target in assign.targets if isinstance(target, ast.Name)] == ["block_width"]):
        problems.append("decode_table_width's result is not assigned to block_width: the tables would be built at the undecorated width")
    if _names(call) != ["self", "req_ids", "block_width", "positions_cpu"]:
        problems.append(f"decode_table_width is called with {ast.unparse(call)}, want (self, req_ids, block_width, positions_cpu)")
    if _guard(call, parents) != "is_decode":
        problems.append(f"decode_table_width runs under {_guard(call, parents) or 'no condition'}, want `if is_decode:`")
    built = next((node for node in ast.walk(scope) if isinstance(node, ast.Assign) and getattr(node.targets[0], "id", "") == "block_table_cpu"), None)
    if built is not None and built.lineno < call.lineno:
        problems.append("decode_table_width runs after the block table is built with the old width")
    return problems


def _forward_hook_problems(scope: ast.AST, forward: ast.Call, parents: dict[ast.AST, ast.AST]) -> list[str]:
    problems: list[str] = []
    key = _only_call(scope, "decode_key_metadata", FORWARD_FUNCTION, problems)
    observe = _only_call(scope, "observe_decode_hidden", FORWARD_FUNCTION, problems)
    if key is not None:
        assign = parents.get(key)
        target = assign.targets[0].id if isinstance(assign, ast.Assign) and isinstance(assign.targets[0], ast.Name) else None
        given = {keyword.arg: keyword.value for keyword in forward.keywords}.get("attn_metadata")
        if target is None or not (isinstance(given, ast.Name) and given.id == target):
            problems.append("the model is not called with decode_key_metadata's result as attn_metadata: the hook runs and its answer is dropped")
        if [ast.unparse(argument) for argument in key.args] != ["trim_attn_metadata(attn_metadata)", "batch_size", "use_graphs", "warmup_mode"]:
            problems.append(f"decode_key_metadata is called with {ast.unparse(key)}, want (trim_attn_metadata(attn_metadata), batch_size, use_graphs, warmup_mode)")
        if key.lineno > forward.lineno:
            problems.append("decode_key_metadata runs after the model forward")
    if observe is not None:
        logits = _calls(scope, "compute_logits")
        if _guard(observe, parents) != "not attn_metadata.is_prompt":
            problems.append(f"observe_decode_hidden runs under {_guard(observe, parents) or 'no condition'}, want `if not attn_metadata.is_prompt:`")
        if _names(observe) != ["non_flattened_hidden_states"]:
            problems.append(f"observe_decode_hidden is called with {ast.unparse(observe)}, want the model's own hidden states (non_flattened_hidden_states)")
        if observe.lineno < forward.lineno or any(observe.lineno > call.lineno for call in logits):
            problems.append("observe_decode_hidden is not between the model forward and compute_logits")
    return problems


def forward_problems(tree: ast.AST) -> list[str]:
    scope = _function(tree, FORWARD_FUNCTION)
    if scope is None:
        return [f"{FORWARD_FUNCTION} is gone: find the runner's model call and re-point this file"]
    problems: list[str] = []
    missing = {"decode_key_metadata", "observe_decode_hidden"} - _imports(scope)
    if missing:
        problems.append(f"{FORWARD_FUNCTION} does not import {sorted(missing)} from {HOOK_MODULE}")
    forwards = [call for call in _calls(scope, "forward") if ast.unparse(call.func) == "self.model.forward"]
    if len(forwards) != 1:
        return problems + [f"{FORWARD_FUNCTION} has {len(forwards)} self.model.forward calls, want 1"]
    return problems + _forward_hook_problems(scope, forwards[0], _parents(scope))


def wiring_problems(source: str) -> list[str]:
    tree = ast.parse(source)
    return table_width_problems(tree) + forward_problems(tree)


class TheRunnersCalls(unittest.TestCase):
    def test_the_runner_wires_all_three_hooks(self) -> None:
        self.assertEqual(wiring_problems(RUNNER.read_text(encoding="utf-8")), [])

    def test_each_call_fits_the_hooks_signature(self) -> None:
        """A hook that grew or lost a parameter would make the runner's call a TypeError only on a card."""
        graph = load_overlay("glm53_mixed_graph.py")
        tree = ast.parse(RUNNER.read_text(encoding="utf-8"))
        for scope, name in ((_function(tree, METADATA_FUNCTION), "decode_table_width"), (_function(tree, FORWARD_FUNCTION), "decode_key_metadata"),
                            (_function(tree, FORWARD_FUNCTION), "observe_decode_hidden")):
            call = _calls(scope, name)[0]
            with self.subTest(hook=name):
                inspect.signature(getattr(graph, name)).bind(*call.args, **{keyword.arg: keyword.value for keyword in call.keywords})


class TheMutants(unittest.TestCase):
    """Each mutant is the runner with one call removed, bypassed or moved; every one must be named, and the unbroken source must not be."""

    @staticmethod
    def source() -> str:
        return RUNNER.read_text(encoding="utf-8")

    def mutated(self, old: str, new: str, text: str | None = None) -> str:
        text = self.source() if text is None else text
        self.assertEqual(text.count(old), 1, f"the mutant's anchor moved in the runner: {old[:70]!r}")
        return text.replace(old, new)

    def assert_named(self, source: str, mention: str) -> None:
        found = wiring_problems(source)
        self.assertTrue(any(mention in problem for problem in found), (mention, found))

    def test_removing_observe_decode_hidden_is_caught(self) -> None:
        self.assert_named(self.mutated("            observe_decode_hidden(non_flattened_hidden_states)\n", "            pass\n"), "0 calls to observe_decode_hidden")

    def test_bypassing_decode_key_metadata_is_caught(self) -> None:
        old = "        trimmed_attn_metadata = decode_key_metadata(\n            trim_attn_metadata(attn_metadata), batch_size, use_graphs, warmup_mode\n        )\n"
        self.assert_named(self.mutated(old, "        trimmed_attn_metadata = trim_attn_metadata(attn_metadata)\n"), "0 calls to decode_key_metadata")

    def test_calling_decode_key_metadata_and_dropping_its_answer_is_caught(self) -> None:
        old = "        trimmed_attn_metadata = decode_key_metadata(\n            trim_attn_metadata(attn_metadata), batch_size, use_graphs, warmup_mode\n        )\n"
        new = "        decode_key_metadata(trim_attn_metadata(attn_metadata), batch_size, use_graphs, warmup_mode)\n        trimmed_attn_metadata = trim_attn_metadata(attn_metadata)\n"
        self.assert_named(self.mutated(old, new), "answer is dropped")

    def test_calling_the_model_with_the_untouched_metadata_is_caught(self) -> None:
        self.assert_named(self.mutated("attn_metadata=trimmed_attn_metadata,", "attn_metadata=attn_metadata,"), "answer is dropped")

    def test_removing_decode_table_width_is_caught(self) -> None:
        self.assert_named(self.mutated("            block_width = decode_table_width(self, req_ids, block_width, positions_cpu)\n", "            pass\n"),
                          "0 calls to decode_table_width")

    def test_dropping_the_table_width_result_is_caught(self) -> None:
        self.assert_named(self.mutated("            block_width = decode_table_width(self, req_ids, block_width, positions_cpu)\n",
                                       "            decode_table_width(self, req_ids, block_width, positions_cpu)\n"), "not assigned to block_width")

    def test_reading_a_prompt_step_as_a_decode_step_is_caught(self) -> None:
        self.assert_named(self.mutated("        if not attn_metadata.is_prompt:\n            observe_decode_hidden(", "        if attn_metadata.is_prompt:\n            observe_decode_hidden("),
                          "want `if not attn_metadata.is_prompt:`")

    def test_observing_the_indexed_logits_rows_instead_of_the_hidden_states_is_caught(self) -> None:
        self.assert_named(self.mutated("            observe_decode_hidden(non_flattened_hidden_states)\n", "            observe_decode_hidden(hidden_states)\n"),
                          "want the model's own hidden states")

    def test_observing_after_the_logits_is_caught(self) -> None:
        observed = "        if not attn_metadata.is_prompt:\n            observe_decode_hidden(non_flattened_hidden_states)\n"
        logits = "            logits = self.model.compute_logits(hidden_states)\n"
        moved = self.mutated(logits, logits + "            if not attn_metadata.is_prompt:\n                observe_decode_hidden(non_flattened_hidden_states)\n",
                             self.mutated(observed, ""))
        self.assert_named(moved, "not between the model forward and compute_logits")

    def test_the_unbroken_source_names_nothing(self) -> None:
        self.assertEqual(wiring_problems(self.source()), [])


if __name__ == "__main__":
    unittest.main()
