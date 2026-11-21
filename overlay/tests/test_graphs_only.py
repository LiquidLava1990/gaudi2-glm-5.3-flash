"""Step 30: HPU_GLM53_GRAPHS_ONLY=1 on the prompt side. A boot refusal per missing requirement, a prompt that no key can serve raising on
every path that used to run native (including the prompt-only step, whose native route native_no_decode keeps on by default), the
top key against --max-model-len, the witness line, an AST check that no eager path is reachable under the flag, and flag-off parity."""
from __future__ import annotations

import ast
import dataclasses
import os
import sys
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent))

from overlay_testlib import OVERLAY_ROOT  # noqa: E402
from test_prompt_keys import BLOCK, Backend, KeyCase, RowForward  # noqa: E402

ONLY = "HPU_GLM53_GRAPHS_ONLY"
KEYS = "VLLM_GAUDI_MIXED_PROMPT_KEYS"
KEY_SET = "1024:8192,1024:4096,1024:2048,1024:1024"
VALID_ENV = {
    ONLY: "1", KEYS: KEY_SET, "VLLM_GAUDI_MIXED_STEP": "1", "VLLM_GAUDI_MIXED_GRAPHS": "1",
    "VLLM_PREFILL_CONTEXT_CELL_BUDGET": "0", "VLLM_PREFILL_MAX_REQUESTS_PER_STEP": "1",
    "VLLM_ADAPTIVE_LONG_PREFILL_SINGLE_TOKENS": "1024", "VLLM_GAUDI_MIXED_PREFILL_TOKENS": "1024",
    "VLLM_RECURRENT_TAIL_FUSE": "0", "HPU_GLM53_PRIVATE_PROMPT_TOPK": "1",
    "VLLM_GAUDI_MIXED_DECODE_WIDTHS": "1024,4096,8192", "HPU_GLM53_DECODE_WIDTH_PAD": "check", "HPU_GLM53_DECODE_PAD_CHECK_WIDTHS": "2048",
    "HPU_GLM53_MIXED_DECODE_REPLAY": "1", "VLLM_HPU_RESTORED_DECODE_GRAPHS": "1", "HPU_GRAPH_DRY_RUN": "0",
}
NEEDS_NO_ENV = ("VLLM_PREFILL_CONTEXT_CELL_BUDGET", "VLLM_PREFILL_MAX_REQUESTS_PER_STEP", "VLLM_ADAPTIVE_LONG_PREFILL_SINGLE_TOKENS",
                "VLLM_GAUDI_MIXED_PREFILL_TOKENS", "HPU_GLM53_PRIVATE_PROMPT_TOPK", "VLLM_GAUDI_MIXED_NATIVE_NO_DECODE",
                "VLLM_GAUDI_MIXED_DECODE_WIDTHS", "HPU_GLM53_DECODE_WIDTH_PAD", "HPU_GLM53_DECODE_PAD_CHECK_WIDTHS", "HPU_GLM53_MIXED_DECODE_REPLAY",
                "VLLM_HPU_RESTORED_DECODE_GRAPHS", "HPU_GRAPH_DRY_RUN")


class GraphsOnlyCase(KeyCase):
    keys = KEY_SET

    def setUp(self) -> None:
        super().setUp()
        environ = mock.patch.dict(os.environ)
        environ.start()
        self.addCleanup(environ.stop)
        for name in NEEDS_NO_ENV:
            os.environ.pop(name, None)
        os.environ.update(VALID_ENV)
        timer = mock.patch.object(threading, "Timer")
        self.timer = timer.start()
        self.addCleanup(timer.stop)

    def install(self) -> SimpleNamespace:
        adapter = SimpleNamespace()
        self.graph.install_entry(adapter, lambda: None)
        return adapter

    def runner(self, max_len: int = 1_048_576) -> SimpleNamespace:
        return SimpleNamespace(attn_block_size=BLOCK, block_size=BLOCK, max_model_len=max_len,
                               input_batch=SimpleNamespace(max_num_reqs=16))


class BootRefusals(GraphsOnlyCase):
    def test_the_full_environment_boots(self) -> None:
        self.install()

    def test_the_flag_must_be_0_or_1(self) -> None:
        with mock.patch.dict(os.environ, {ONLY: "yes"}), self.assertRaisesRegex(ValueError, ONLY):
            self.install()

    def test_each_missing_requirement_refuses_naming_it_and_its_value(self) -> None:
        cases = (
            ("keys", {KEYS: ""}, f"{KEYS}"),
            ("cell budget unset", {"VLLM_PREFILL_CONTEXT_CELL_BUDGET": None}, "VLLM_PREFILL_CONTEXT_CELL_BUDGET=0"),
            ("cell budget on", {"VLLM_PREFILL_CONTEXT_CELL_BUDGET": "1024"}, "VLLM_PREFILL_CONTEXT_CELL_BUDGET=0"),
            ("one prompt per step unset", {"VLLM_PREFILL_MAX_REQUESTS_PER_STEP": None}, "VLLM_PREFILL_MAX_REQUESTS_PER_STEP=1"),
            ("two prompts per step", {"VLLM_PREFILL_MAX_REQUESTS_PER_STEP": "2"}, "VLLM_PREFILL_MAX_REQUESTS_PER_STEP=1"),
            ("tail fuse", {"VLLM_RECURRENT_TAIL_FUSE": "1"}, "VLLM_RECURRENT_TAIL_FUSE=0"),
            ("private top-k", {"HPU_GLM53_PRIVATE_PROMPT_TOPK": None}, "HPU_GLM53_PRIVATE_PROMPT_TOPK=1"),
            ("single-prompt threshold unset", {"VLLM_ADAPTIVE_LONG_PREFILL_SINGLE_TOKENS": None},
             "VLLM_ADAPTIVE_LONG_PREFILL_SINGLE_TOKENS=1024"),
            ("single-prompt threshold past R", {"VLLM_ADAPTIVE_LONG_PREFILL_SINGLE_TOKENS": "2048"},
             "VLLM_ADAPTIVE_LONG_PREFILL_SINGLE_TOKENS=1024"),
            ("mixed budget past R", {"VLLM_GAUDI_MIXED_PREFILL_TOKENS": "2048"}, "VLLM_GAUDI_MIXED_PREFILL_TOKENS=1024"),
            ("mixed step", {"VLLM_GAUDI_MIXED_STEP": "0"}, "VLLM_GAUDI_MIXED_STEP"),
            ("restored decodes unset", {"VLLM_HPU_RESTORED_DECODE_GRAPHS": None}, "VLLM_HPU_RESTORED_DECODE_GRAPHS=1"),
            ("dry run unset", {"HPU_GRAPH_DRY_RUN": None}, "HPU_GRAPH_DRY_RUN=0"),
            ("dry run on", {"HPU_GRAPH_DRY_RUN": "1"}, "HPU_GRAPH_DRY_RUN=0"),
            ("no decode lattice", {"VLLM_GAUDI_MIXED_DECODE_WIDTHS": "", "HPU_GLM53_DECODE_WIDTH_PAD": "0", "HPU_GLM53_DECODE_PAD_CHECK_WIDTHS": ""},
             "VLLM_GAUDI_MIXED_DECODE_WIDTHS"),
        )
        for label, change, expected in cases:
            with self.subTest(label), mock.patch.dict(os.environ):
                for name, value in change.items():
                    os.environ.pop(name, None) if value is None else os.environ.__setitem__(name, value)
                with self.assertRaises(ValueError) as refused:
                    self.install()
                self.assertIn(expected, str(refused.exception))

    def test_the_mixed_budget_default_fits_under_the_key_rows(self) -> None:
        with mock.patch.dict(os.environ):
            os.environ.pop("VLLM_GAUDI_MIXED_PREFILL_TOKENS")
            self.install()  # the step's own default is 512 rows, inside R = 1024

    def test_flag_off_boots_with_every_requirement_missing(self) -> None:
        with mock.patch.dict(os.environ, {ONLY: "0"}):
            for name in NEEDS_NO_ENV:
                os.environ.pop(name, None)
            self.install()

    def test_the_step_45_env_file_passes_when_the_lane_folder_is_mounted(self) -> None:
        lane = Path(os.environ.get("LANE_DIR", ""))
        env_file = lane / "graphs-dual.env"
        if not env_file.is_file():
            self.skipTest("mount the lane folder at LANE_DIR (run_tests.sh does)")
        lines = [line for line in env_file.read_text(encoding="utf-8").splitlines() if line.strip() and not line.startswith("#")]
        env = dict(line.split("=", 1) for line in lines)
        self.assertEqual(env[KEYS], "1024:8192,1024:1024")
        self.assertEqual(env[ONLY], "1")
        for flag, wanted in (("VLLM_HPU_RESTORED_DECODE_GRAPHS", "1"), ("HPU_GRAPH_DRY_RUN", "0"), ("HPU_GLM53_MIXED_DECODE_REPLAY", "1"),
                             ("VLLM_GAUDI_MIXED_DECODE_WIDTHS", "1024,4096,8192"), ("VLLM_GAUDI_TARGETED_SAMPLER_WARMUP_BATCHES", "1,2,4,8,16"),
                             ("VLLM_GAUDI_BOOTSTRAP_DECODE_BUCKETS", "1,1,128;2,1,256;4,1,512;8,1,1024;16,1,2048")):
            self.assertEqual(env.get(flag), wanted, flag)
        with mock.patch.dict(os.environ, env):
            self.install()


class TopKey(GraphsOnlyCase):
    def test_a_top_key_below_max_model_len_refuses_naming_the_key_and_its_bytes(self) -> None:
        with mock.patch.dict(os.environ, {KEYS: "1024:4096,1024:1024"}), self.assertRaises(ValueError) as refused:
            self.graph.validate_warmup(self.runner())
        message = str(refused.exception)
        self.assertIn("1024:4096", message)
        self.assertIn(str(self.graph.key_estimate_bytes(1024)), message)
        self.assertIn("8192", message)  # the key that would hold it

    def test_the_same_keys_boot_with_the_flag_off(self) -> None:
        with mock.patch.dict(os.environ, {KEYS: "1024:4096,1024:1024", ONLY: "0"}):
            self.assertEqual(len(self.graph.validate_warmup(self.runner())), 2)

    def test_a_top_key_that_holds_max_model_len_logs_the_witness(self) -> None:
        with self.assertLogs(self.graph.LOG, "WARNING") as logs:
            self.graph.validate_warmup(self.runner())
        self.assertIn(f"GLM53_GRAPHS_ONLY keys={KEY_SET} fallback=refused", "\n".join(logs.output))

    def test_flag_off_logs_no_witness(self) -> None:
        with mock.patch.dict(os.environ, {ONLY: "0"}), self.assertNoLogs(self.graph.LOG, "WARNING"):
            self.graph.validate_warmup(self.runner())


class Misses(GraphsOnlyCase):
    def setUp(self) -> None:
        super().setUp()
        self.backend = Backend()
        self.cache = self.graph.GraphCache(RowForward(), BLOCK, self.backend)
        self.boot_capture(self.cache, self.prompt_batch(1024, 0, 1024))

    def refused(self, batch: object) -> str:
        with self.assertRaises(RuntimeError) as refused:
            self.cache.run_key(batch)
        return str(refused.exception)

    def misses(self) -> dict[str, object]:
        decode_like = self.prompt_batch(128, 0, 1024)
        decode_like = dataclasses.replace(decode_like, region=dataclasses.replace(decode_like.region, batch=2, requests=(0, 1),
                                                                                  lengths=(128, 128), starts=(0, 0)))
        tableless = self.prompt_batch(128, 0, 1024)
        tableless.region = dataclasses.replace(tableless.region, metadata=dataclasses.replace(
            tableless.region.metadata, glm53_index_block_table=None))
        return {"width_over": self.prompt_batch(1025, 0, 1024, native_width=1025),
                "ctx_over": self.prompt_batch(1024, 0, 16384),
                "multi_prompt": decode_like, "signature": tableless,
                "not_warmed": self.prompt_batch(128, 0, 4096)}

    def test_every_miss_reason_raises_instead_of_running_native(self) -> None:
        for reason, batch in self.misses().items():
            with self.subTest(reason):
                message = self.refused(batch)
                self.assertIn(ONLY, message)
                self.assertIn(reason, message)
                self.assertIn(KEY_SET, message)

    def test_every_refusal_says_what_to_change(self) -> None:
        fixes = {"width_over": "VLLM_ADAPTIVE_LONG_PREFILL_SINGLE_TOKENS", "ctx_over": KEYS,
                 "multi_prompt": "slot policy", "not_warmed": "boot"}
        misses = self.misses()
        for reason, needle in fixes.items():
            with self.subTest(reason):
                self.assertIn(needle, self.refused(misses[reason]))

    def test_the_multi_prompt_refusal_does_not_recommend_the_flag_that_does_nothing_in_mixed_mode(self) -> None:
        message = self.refused(self.misses()["multi_prompt"])
        self.assertNotIn("VLLM_PREFILL_MAX_REQUESTS_PER_STEP", message)
        self.assertIn("prompt keys", message)

    def test_a_refusal_counts_and_leaves_no_native_trace(self) -> None:
        for batch in self.misses().values():
            self.refused(batch)
        counts = self.cache.counts
        self.assertEqual((counts["graphs_only_refused"], counts["native_fallbacks"], counts["lazy_fallbacks"]), (5, 0, 0))
        self.assertEqual(self.backend.replays, 0)

    def test_a_refusal_arms_the_bounded_exit_so_no_rank_hangs_alone(self) -> None:
        self.refused(self.prompt_batch(1025, 0, 1024, native_width=1025))
        self.timer.assert_called_once()
        self.assertEqual(self.timer.call_args.args[0], 120.0)
        self.timer.return_value.start.assert_called_once()
        self.assertTrue(self.timer.return_value.daemon)

    def test_a_hit_neither_raises_nor_arms_anything(self) -> None:
        self.assertIsNotNone(self.cache.run_key(self.prompt_batch(512, 1024, 1024, seed=3)))
        self.timer.assert_not_called()

    def test_a_real_allocation_failure_in_a_replay_arms_the_exit_and_raises(self) -> None:
        self.backend.replay = mock.Mock(side_effect=RuntimeError("PT_DEVMEM: failed to allocate"))
        with self.assertRaisesRegex(RuntimeError, "PT_DEVMEM"):
            self.cache.run_key(self.prompt_batch(512, 1024, 1024, seed=3))
        self.timer.assert_called_once()

    def test_another_replay_error_or_flag_off_arms_nothing(self) -> None:
        self.backend.replay = mock.Mock(side_effect=RuntimeError("PT_DEVMEM: failed to allocate"))
        with mock.patch.dict(os.environ, {ONLY: "0"}), self.assertRaises(RuntimeError):
            self.cache.run_key(self.prompt_batch(512, 1024, 1024, seed=3))
        self.backend.replay = mock.Mock(side_effect=RuntimeError("shape mismatch"))
        with self.assertRaisesRegex(RuntimeError, "shape mismatch"):
            self.cache.run_key(self.prompt_batch(512, 1024, 1024, seed=3))
        self.timer.assert_not_called()

    def test_flag_off_the_same_miss_still_runs_native(self) -> None:
        with mock.patch.dict(os.environ, {ONLY: "0"}):
            self.assertIsNone(self.cache.run_key(self.prompt_batch(1025, 0, 1024, native_width=1025)))
        self.assertEqual((self.cache.counts["native_fallbacks"], self.cache.counts["graphs_only_refused"]), (1, 0))
        self.timer.assert_not_called()

    def test_the_mixed_decode_plus_prompt_graph_is_not_a_path_under_the_flag(self) -> None:
        with self.assertRaisesRegex(RuntimeError, KEYS):
            self.cache.run([self.prompt_batch(128, 0, 1024)])
        with mock.patch.dict(os.environ, {ONLY: "0"}):
            self.assertIsNone(self.cache.run([self.prompt_batch(128, 0, 1024)]))


class PromptOnlyStep(GraphsOnlyCase):
    """A step with no decode lane: native_no_decode (default on) used to send it to the runner's own forward."""

    def run_step(self, batch: object) -> mock.Mock:
        cache = self.graph.GraphCache(RowForward(), BLOCK, Backend())
        self.boot_capture(cache, self.prompt_batch(1024, 0, 1024))
        model = SimpleNamespace(_glm53_mixed_graph_cache=cache, _glm53_mixed_graph_forward=RowForward(),
                                compute_logits=lambda rows: rows.sum(-1, keepdim=True))
        generic = mock.Mock(return_value="native-result")
        runner = SimpleNamespace(model=model, model_config=SimpleNamespace(enforce_eager=False), attn_block_size=BLOCK,
                                 _execute_model_generic=generic)
        pd = SimpleNamespace(prompt_req_ids=["a"], decode_req_ids=[])
        with mock.patch.object(self.step, "_validate_runner"), mock.patch.object(self.step, "_batches", return_value=[batch]):
            forward = self.step.prepare(runner, None, pd, None, None)
        self.assertIsInstance(forward, self.step.KeyedForward)
        forward(None, None, batch.region.metadata)
        return generic

    def test_a_prompt_only_miss_raises_whatever_native_no_decode_says(self) -> None:
        for native_no_decode in (None, "0", "1"):
            with self.subTest(native_no_decode=native_no_decode), mock.patch.dict(os.environ):
                if native_no_decode is not None:
                    os.environ["VLLM_GAUDI_MIXED_NATIVE_NO_DECODE"] = native_no_decode
                with self.assertRaisesRegex(RuntimeError, ONLY):
                    self.run_step(self.prompt_batch(1025, 0, 1024, native_width=1025))

    def test_a_prompt_only_hit_replays_and_never_touches_the_native_forward(self) -> None:
        generic = self.run_step(self.prompt_batch(700, 0, 1024))
        generic.assert_not_called()

    def test_flag_off_the_prompt_only_miss_still_goes_native(self) -> None:
        with mock.patch.dict(os.environ, {ONLY: "0"}):
            generic = self.run_step(self.prompt_batch(1025, 0, 1024, native_width=1025))
        generic.assert_called_once()


def _functions(tree: ast.AST) -> list[ast.FunctionDef]:
    return [node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)]


def _mentions(node: ast.AST, name: str) -> bool:
    return any(isinstance(child, ast.Name) and child.id == name for child in ast.walk(node))


def _records_a_native_or_lazy_miss(call: ast.Call) -> bool:
    if not (isinstance(call.func, ast.Attribute) and call.func.attr == "record"):
        return False
    return any(_mentions(arg, "miss_event") or (isinstance(arg, ast.Constant) and arg.value in ("native_fallbacks", "lazy_fallbacks"))
               for arg in call.args)


def _gate_line(function: ast.FunctionDef) -> int | None:
    """The first top-level ``if graphs_only_enabled(): ... raise`` of the function, as a line number."""
    for statement in function.body:
        if isinstance(statement, ast.If) and _mentions(statement.test, "graphs_only_enabled") \
                and any(isinstance(child, ast.Raise) for child in ast.walk(statement)):
            return statement.lineno
    return None


class NoEagerPathReachable(unittest.TestCase):
    """Structure, not behavior: a new place that records a native or lazy miss without the flag's raise before it fails here."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.tree = ast.parse((OVERLAY_ROOT / "glm53_mixed_graph.py").read_text(encoding="utf-8"))

    def test_every_function_that_records_a_native_or_lazy_miss_raises_first_under_the_flag(self) -> None:
        sites = {}
        for function in _functions(self.tree):
            lines = [call.lineno for call in ast.walk(function) if isinstance(call, ast.Call) and _records_a_native_or_lazy_miss(call)]
            if lines:
                sites[function.name] = (min(lines), _gate_line(function))
        self.assertEqual(set(sites), {"run", "_key_miss"}, sites)
        for name, (first_miss, gate) in sites.items():
            with self.subTest(name):
                self.assertIsNotNone(gate, f"{name} records a miss with no graphs_only_enabled() raise")
                self.assertLess(gate, first_miss)

    def test_run_key_returns_only_a_hidden_state_or_the_gated_miss(self) -> None:
        run_key = next(function for function in _functions(self.tree) if function.name == "run_key")
        for node in ast.walk(run_key):
            if isinstance(node, ast.Return):
                value = node.value
                gated = isinstance(value, ast.Call) and isinstance(value.func, ast.Attribute) and value.func.attr == "_key_miss"
                self.assertTrue(gated or (isinstance(value, ast.Name) and value.id == "hidden"), ast.dump(node))

    def test_the_runner_fallback_in_keyed_forward_needs_a_none_that_only_the_gated_miss_makes(self) -> None:
        step = ast.parse((OVERLAY_ROOT / "glm53_mixed_step.py").read_text(encoding="utf-8"))
        keyed = next(node for node in ast.walk(step) if isinstance(node, ast.ClassDef) and node.name == "KeyedForward")
        generic = [call for call in ast.walk(keyed) if isinstance(call, ast.Call) and isinstance(call.func, ast.Attribute)
                   and call.func.attr == "_execute_model_generic"]
        self.assertEqual(len(generic), 1)
        guard = next(node for node in ast.walk(keyed) if isinstance(node, ast.If) and _mentions(node.test, "hidden"))
        self.assertIn(generic[0], list(ast.walk(guard)))

    def test_nothing_else_in_the_module_calls_the_runner_forward_on_a_miss(self) -> None:
        source = (OVERLAY_ROOT / "glm53_mixed_graph.py").read_text(encoding="utf-8")
        self.assertNotIn("_execute_model_generic", source)


class FlagReaders(GraphsOnlyCase):
    def test_both_modules_read_the_same_variable(self) -> None:
        guard = __import__("glm53_graphs_only")
        self.assertEqual((guard.ENV, self.graph.GRAPHS_ONLY_FLAG), (ONLY, ONLY))


if __name__ == "__main__":
    unittest.main()
