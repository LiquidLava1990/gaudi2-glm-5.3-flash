"""HPU_GLM53_FINAL_NORM_MARK: the final norm ends a lazy segment, so with the terminal layer's mark the norm is a segment of its own in eager and in a capture.

Card evidence (k1-out/fable-l3 run3 and run4): without it one row per chunk differs after 45 bitwise equal layers. The real modify_model_layers runs here on a
fake conditional root; off, it registers exactly what base registers.
"""
from __future__ import annotations

import ast
import logging
import os
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest import mock

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

import glm53_mixed_graph as graph  # noqa: E402
from overlay_testlib import OVERLAY_ROOT  # noqa: E402

RUNNER_FILE = OVERLAY_ROOT / "vllm_gaudi/v1/worker/hpu_model_runner.py"
FLAG = "HPU_GLM53_FINAL_NORM_MARK"
FUNCTIONS = ("_glm53_final_norm_mark_enabled", "_glm53_final_norm_hook", "modify_model_layers")


def runner_namespace(marks: list[int], terminal: torch.nn.Module) -> dict[str, Any]:
    """The runner's two real functions, with the layer selection and the mark_step faked."""
    tree = ast.parse(RUNNER_FILE.read_text(encoding="utf-8"))
    body = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in FUNCTIONS]
    namespace: dict[str, Any] = {
        "os": os, "torch": torch, "logger": logging.getLogger("glm53.final_norm_test"),
        "htorch": SimpleNamespace(core=SimpleNamespace(mark_step=lambda: marks.append(1))),
        "_select_glm53_terminal_markstep_layers": lambda _module, _suffixes: [terminal],
    }
    exec(compile(ast.Module(body=body, type_ignores=[]), str(RUNNER_FILE), "exec"), namespace)
    return namespace


class Root(torch.nn.Module):
    """language_model.model.norm, as the terminal selection has proved it exists."""

    def __init__(self) -> None:
        super().__init__()
        self.language_model = torch.nn.Module()
        self.language_model.model = torch.nn.Module()
        self.language_model.model.norm = torch.nn.Identity()
        self.language_model.model.layer = torch.nn.Identity()


class FinalNormMark(unittest.TestCase):
    def run_model(self, env: dict[str, str], *, terminal_only: bool = True) -> list[int]:
        """Forward the terminal layer, then the norm; returns the marks after each."""
        marks: list[int] = []
        root = Root()
        model = root.language_model.model
        with mock.patch.dict(os.environ, env):
            runner_namespace(marks, model.layer)["modify_model_layers"](root, ["Nothing"], 1, glm53_terminal_only=terminal_only)
        model.layer(torch.zeros(1))
        after_layer = len(marks)
        model.norm(torch.zeros(1))
        return [after_layer, len(marks)]

    def test_flag_unset_marks_the_terminal_layer_only(self) -> None:
        self.assertEqual(self.run_model({}), [1, 1])

    def test_flag_zero_marks_the_terminal_layer_only(self) -> None:
        self.assertEqual(self.run_model({FLAG: "0"}), [1, 1])

    def test_flag_on_also_marks_after_the_norm(self) -> None:
        self.assertEqual(self.run_model({FLAG: "1"}), [1, 2])

    def test_a_bad_value_stops_the_boot(self) -> None:
        with self.assertRaisesRegex(RuntimeError, FLAG):
            self.run_model({FLAG: "yes"})

    def test_flag_on_without_the_terminal_mark_stops_the_boot(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "HPU_GLM53_TERMINAL_LAYER_MARKSTEP"):
            self.run_model({FLAG: "1"}, terminal_only=False)

    def test_flag_off_without_the_terminal_mark_keeps_the_interval_path(self) -> None:
        marks: list[int] = []
        root = Root()
        with mock.patch.dict(os.environ, {FLAG: "0"}):
            runner_namespace(marks, root.language_model.model.layer)["modify_model_layers"](root, ["Identity"], 1)
        root.language_model.model.norm(torch.zeros(1))
        root.language_model.model.layer(torch.zeros(1))
        self.assertEqual(len(marks), 2)


class Witness(unittest.TestCase):
    """A replay runs no Python, so a live boot can only prove the mark from these lines: registered on each rank, fired inside each capture."""

    def boot(self, env: dict[str, str]) -> tuple[Root, list[str]]:
        root = Root()
        with mock.patch.dict(os.environ, env), self.assertLogs("glm53.final_norm_test", "WARNING") as logs:
            runner_namespace([], root.language_model.model.layer)["modify_model_layers"](root, ["Nothing"], 1, glm53_terminal_only=True)
        return root, logs.output

    def test_flag_on_logs_registered_with_this_process(self) -> None:
        _, lines = self.boot({FLAG: "1"})
        self.assertEqual(len(lines), 1)
        self.assertIn(f"GLM53_FINAL_NORM_MARK registered=1 pid={os.getpid()}", lines[0])

    def test_flag_off_logs_not_registered(self) -> None:
        """A misspelt flag name leaves the flag unset: the line says so, where no line would look like a normal boot."""
        _, lines = self.boot({"HPU_GLM53_FINAL_NORM_MARKK": "1"})
        self.assertIn("GLM53_FINAL_NORM_MARK registered=0", lines[0])

    def test_a_prompt_key_capture_counts_the_mark_on_its_lever(self) -> None:
        root, _ = self.boot({FLAG: "1"})
        graph.reset_witness()
        with graph.key_capture_context():
            root.language_model.model.norm(torch.zeros(1))
        self.assertEqual(graph._WITNESS["final_norm_mark"], 1)

    def test_a_decode_key_capture_logs_the_mark_with_its_key(self) -> None:
        root, _ = self.boot({FLAG: "1"})
        token = graph._DECODE_KEY.set((16, 1024))
        try:
            with self.assertLogs(graph.LOG, "WARNING") as logs:
                root.language_model.model.norm(torch.zeros(1))
        finally:
            graph._DECODE_KEY.reset(token)
        self.assertIn("GLM53_KEY_WITNESS decode_key=(16, 1024) lever=final_norm_mark", logs.output[0])

    def test_outside_a_capture_the_mark_runs_and_counts_nothing(self) -> None:
        root, _ = self.boot({FLAG: "1"})
        graph.reset_witness()
        root.language_model.model.norm(torch.zeros(1))
        self.assertEqual(graph._WITNESS["final_norm_mark"], 0)

    def test_flag_off_captures_show_the_lever_at_zero(self) -> None:
        root, _ = self.boot({FLAG: "0"})
        graph.reset_witness()
        with graph.key_capture_context():
            root.language_model.model.norm(torch.zeros(1))
        self.assertEqual(graph._WITNESS["final_norm_mark"], 0)


if __name__ == "__main__":
    unittest.main()
