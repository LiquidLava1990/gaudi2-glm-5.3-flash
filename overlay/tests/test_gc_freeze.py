"""HPU_GLM53_GC_FREEZE_AT_SEAL: off touches nothing, on collects and freezes once per worker after the seal and logs a witness per rank, a freeze
that did not take refuses the boot naming the flag, and the hook sits in seal_boot_memory after the witness. All on CPU."""
from __future__ import annotations

import ast
import gc
import sys
import unittest
from collections.abc import Callable
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent))

import glm53_gc_freeze as freeze  # noqa: E402
import test_decode_lattice as lattice  # noqa: E402
from overlay_testlib import OVERLAY_ROOT  # noqa: E402

NO_KEYS = {"VLLM_GAUDI_MIXED_PROMPT_KEYS": "", "VLLM_GAUDI_MIXED_DECODE_WIDTHS": "", "HPU_GLM53_DECODE_WIDTH_PAD": "0",
           "HPU_GLM53_DECODE_PAD_CHECK_WIDTHS": ""}
ON = {freeze.FLAG: "1"}


class FakeCollector:
    """The gc calls the freeze makes, recorded. It has no disable(): a call to it raises AttributeError and fails the test."""

    def __init__(self, frozen: int = 962_000, fail: Exception | None = None, on_freeze: Callable[[], None] | None = None) -> None:
        self.calls: list[str] = []
        self.frozen, self.fail, self.on_freeze = frozen, fail, on_freeze

    def collect(self) -> int:
        self.calls.append("collect")
        return 1799

    def freeze(self) -> None:
        self.calls.append("freeze")
        if self.on_freeze is not None:
            self.on_freeze()
        if self.fail is not None:
            raise self.fail

    def get_freeze_count(self) -> int:
        return self.frozen if "freeze" in self.calls else 0

    def isenabled(self) -> bool:
        return True


def stepping_clock() -> Callable[[], float]:
    """Three readings: the collection takes 312 ms, the freeze 0.5 ms."""
    readings = iter((1.0, 1.312, 1.3125))
    return lambda: next(readings)


def said(log: mock.Mock) -> list[str]:
    return [call.args[0] % call.args[1:] for call in log.call_args_list]


class FreezeUnitTest(unittest.TestCase):
    def setUp(self) -> None:
        freeze.reset()
        self.addCleanup(freeze.reset)

    def test_only_zero_one_and_unset_are_values(self) -> None:
        self.assertEqual([freeze.enabled({}), freeze.enabled({freeze.FLAG: "0"}), freeze.enabled(ON)], [False, False, True])
        for bad in ("yes", "true", "2", ""):
            with self.subTest(bad=bad), self.assertRaisesRegex(ValueError, f"{freeze.FLAG}.*must be 0 or 1"):
                freeze.enabled({freeze.FLAG: bad})

    def test_off_touches_nothing(self) -> None:
        log, admit, collector = mock.Mock(), mock.Mock(), FakeCollector()
        self.assertIsNone(freeze.freeze_at_seal(log, admit, environ={}, collector=collector, rank=0))
        self.assertEqual(collector.calls, [])
        log.assert_not_called()
        admit.assert_not_called()

    def test_on_collects_then_freezes_and_logs_one_witness_with_the_collector_left_on(self) -> None:
        log, admit, collector = mock.Mock(), mock.Mock(), FakeCollector()
        result = freeze.freeze_at_seal(log, admit, environ=ON, collector=collector, clock=stepping_clock(), rank=2)
        self.assertEqual(collector.calls, ["collect", "freeze"])
        self.assertEqual(said(log), ["GLM53_GC_FREEZE rank=2 frozen_objects=962000 collected=1799 collect_ms=312.0 freeze_ms=0.5 gc_enabled=True"])
        self.assertEqual((result.frozen, result.error), (962_000, ""))
        admit.assert_called_once_with(("gc_freeze",), 0, 0, "", freeze.ADVICE)

    def test_a_second_call_in_the_same_worker_freezes_nothing_and_says_so(self) -> None:
        log, admit, collector = mock.Mock(), mock.Mock(), FakeCollector()
        first = freeze.freeze_at_seal(log, admit, environ=ON, collector=collector, rank=0)
        again = freeze.freeze_at_seal(log, admit, environ=ON, collector=collector, rank=0)
        self.assertIs(again, first)
        self.assertEqual(collector.calls, ["collect", "freeze"])
        self.assertEqual(said(log)[-1], "GLM53_GC_FREEZE rank=0 already_frozen=1 frozen_objects=962000")

    def test_a_freeze_that_raises_is_logged_and_handed_to_admit_to_refuse_every_rank(self) -> None:
        log, collector = mock.Mock(), FakeCollector(fail=RuntimeError("heap busy"))
        admit = mock.Mock(side_effect=RuntimeError("mixed graph BOOT admission rejected on ALL ranks"))
        with self.assertRaisesRegex(RuntimeError, "rejected on ALL ranks"):
            freeze.freeze_at_seal(log, admit, environ=ON, collector=collector, rank=1)
        error = admit.call_args.args[3]
        self.assertIn(freeze.FLAG, error)
        self.assertIn("heap busy", error)
        self.assertTrue(said(log)[0].startswith("GLM53_GC_FREEZE_FAILED rank=1 error="))
        self.assertEqual(admit.call_args.args[4], freeze.ADVICE)
        self.assertIn("Unset " + freeze.FLAG, freeze.ADVICE)

    def test_a_freeze_that_froze_nothing_is_a_failure_not_a_silent_skip(self) -> None:
        admit = mock.Mock()
        result = freeze.freeze_at_seal(mock.Mock(), admit, environ=ON, collector=FakeCollector(frozen=0), rank=0)
        self.assertIn("left no object in the permanent generation", result.error)
        self.assertEqual(admit.call_args.args[3], result.error)

    def test_a_boot_without_keys_says_nothing_was_frozen_only_when_asked_to(self) -> None:
        log = mock.Mock()
        freeze.skipped_without_keys(log, {})
        log.assert_not_called()
        freeze.skipped_without_keys(log, ON)
        self.assertIn("GLM53_GC_FREEZE_SKIPPED reason=no_keys nothing_frozen=1", said(log)[0])
        with self.assertRaisesRegex(ValueError, freeze.FLAG):
            freeze.skipped_without_keys(log, {freeze.FLAG: "yes"})

    def test_with_the_real_collector_the_heap_is_frozen_and_the_collector_stays_enabled(self) -> None:
        self.addCleanup(gc.unfreeze)
        self.addCleanup(gc.enable)
        log, admit = mock.Mock(), mock.Mock()
        result = freeze.freeze_at_seal(log, admit, environ=ON, rank=0)
        self.assertGreater(gc.get_freeze_count(), 0)
        self.assertEqual(result.frozen, gc.get_freeze_count())
        self.assertTrue(gc.isenabled())
        self.assertIn("gc_enabled=True", said(log)[0])

    def test_garbage_made_after_the_freeze_is_still_collected(self) -> None:
        self.addCleanup(gc.unfreeze)
        self.addCleanup(gc.enable)
        freeze.freeze_at_seal(mock.Mock(), mock.Mock(), environ=ON, rank=0)

        class Node:
            pass

        gc.disable()       # hold the automatic collector off while the cycles are made, so the count is exact
        before = len(gc.get_objects())
        for _ in range(200):
            first, second = Node(), Node()
            first.other, second.other = second, first      # a reference cycle only the collector can free
        self.assertGreaterEqual(len(gc.get_objects()) - before, 400)
        gc.enable()
        gc.collect()
        self.assertLess(len(gc.get_objects()) - before, 50)


class SealHook(lattice.GraphCase):
    """The hook inside seal_boot_memory, run through the same fake keyed boot the lattice tests use."""
    env = {**lattice.LATTICE_ENV, **ON}

    def boot(self, runner: lattice.FakeRunner, collector: FakeCollector, ballast: mock.Mock | None = None) -> list[str]:
        freeze.reset()
        self.addCleanup(freeze.reset)
        with mock.patch.object(freeze, "gc", collector):
            return lattice.LatticeBoot.boot(self, runner, ballast=ballast)

    def test_the_freeze_comes_after_every_key_the_ballast_and_the_witness(self) -> None:
        runner, ballast = lattice.FakeRunner(self.graph), mock.Mock(return_value=4096 * lattice.MIB)
        seen: list[tuple[int, int]] = []
        collector = FakeCollector(on_freeze=lambda: seen.append((len(runner.calls), ballast.call_count)))
        output = self.boot(runner, collector, ballast)
        self.assertEqual(seen, [(len(self.graph.decode_keys()), 1)])       # all decode keys captured, the ballast held, then the freeze
        order = [next(i for i, line in enumerate(output) if tag in line) for tag in ("GLM53_SEAL_BALLAST", "GLM53_BOOT_WITNESS", "GLM53_GC_FREEZE rank=")]
        self.assertEqual(order, sorted(order))
        self.assertEqual(collector.calls, ["collect", "freeze"])

    def test_the_witness_line_carries_the_rank_the_object_count_and_the_milliseconds(self) -> None:
        output = self.boot(lattice.FakeRunner(self.graph), FakeCollector(frozen=962_000))
        (line,) = [text for text in output if "GLM53_GC_FREEZE rank=" in text]
        for part in ("rank=0", "frozen_objects=962000", "collected=1799", "collect_ms=", "freeze_ms=", "gc_enabled=True"):
            self.assertIn(part, line)

    def test_a_failed_freeze_refuses_the_boot_and_names_the_flag(self) -> None:
        with self.assertRaisesRegex(RuntimeError, f"rejected on ALL ranks.*{freeze.FLAG}.*heap busy"):
            self.boot(lattice.FakeRunner(self.graph), FakeCollector(fail=RuntimeError("heap busy")))

    def test_flag_off_the_boot_never_touches_the_collector_and_logs_no_freeze_line(self) -> None:
        self.set_env(**{freeze.FLAG: "0"})
        collector = FakeCollector()
        output = self.boot(lattice.FakeRunner(self.graph), collector)
        self.assertEqual(collector.calls, [])
        self.assertFalse(any("GLM53_GC_FREEZE" in line for line in output))

    def test_a_boot_with_no_keys_freezes_nothing_and_says_so(self) -> None:
        self.set_env(**NO_KEYS)
        collector, runner = FakeCollector(), lattice.FakeRunner(self.graph)
        with mock.patch.object(freeze, "gc", collector), mock.patch.object(self.graph, "LOG") as log:
            self.graph.seal_boot_memory(runner, self.graph.warmup_decode_lattice(runner))
        self.assertEqual(collector.calls, [])
        self.assertIn("GLM53_GC_FREEZE_SKIPPED reason=no_keys", said(log.warning)[0])

    def test_a_bad_value_is_refused_before_any_capture(self) -> None:
        self.set_env(**{freeze.FLAG: "yes"})
        runner = lattice.FakeRunner(self.graph)
        with self.assertRaisesRegex(ValueError, f"{freeze.FLAG}='yes' must be 0 or 1"):
            self.graph.validate_warmup(runner)
        self.assertEqual(runner.calls, [])


class WiringTest(unittest.TestCase):
    def test_seal_boot_memory_freezes_last_and_validate_warmup_checks_the_value_first(self) -> None:
        tree = ast.parse((OVERLAY_ROOT / "glm53_mixed_graph.py").read_text(encoding="utf-8"))
        functions = {node.name: node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)}
        calls = [ast.unparse(node.value.func) for node in functions["seal_boot_memory"].body if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call)]
        self.assertEqual(calls, ["hold_seal_ballast", "_log_boot_witness", "gc_freeze.freeze_at_seal"])
        early = [ast.unparse(node.value.func) for node in functions["validate_warmup"].body[:6] if isinstance(node, ast.Expr)
                 and isinstance(node.value, ast.Call)]
        self.assertIn("gc_freeze.enabled", early)


if __name__ == "__main__":
    unittest.main()
