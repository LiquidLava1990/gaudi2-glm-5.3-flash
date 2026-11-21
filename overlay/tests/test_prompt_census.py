"""Step 20: the eager prompt census is log only: same mark_step sequence, no synchronize, no peak reset, flag off absent."""

from __future__ import annotations

import json
import os
import sys
import unittest
from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent))

from overlay_testlib import GUARD_FILE, load_overlay  # noqa: E402

CENSUS_FILE = "glm53_prompt_census.py"
FLAG = "HPU_GLM53_PREFILL_HBM_CENSUS"
MIB = 1 << 20
TAG = "GLM53_PROMPT_CENSUS "


class FakeRuntime:
    """A fake htorch: each mark_step materializes ``step`` bytes; the peak only moves up."""

    def __init__(self, step: int = 10 * MIB) -> None:
        self.events: list[tuple] = []
        self.in_use = 1000 * MIB
        self.top = self.in_use
        self.step = step
        self.fail_reads = False
        self.reads = 0
        self.core = SimpleNamespace(mark_step=self._mark_step)
        self.hpu = SimpleNamespace(memory_stats=self._stats, synchronize=self._synchronize,
                                   reset_peak_memory_stats=self._reset_peak)

    def _mark_step(self, *args: object, **kwargs: object) -> str:
        self.events.append(("mark_step", args, tuple(sorted(kwargs.items()))))
        self.in_use += self.step
        self.top = max(self.top, self.in_use)
        return "stepped"

    def _stats(self) -> dict[str, int]:
        self.reads += 1
        if self.fail_reads:
            raise RuntimeError("allocator gone")
        return {"Limit": 96_894 * MIB, "InUse": self.in_use, "MaxInUse": self.top}

    def _synchronize(self) -> None:
        self.events.append(("synchronize",))

    def _reset_peak(self) -> None:
        self.events.append(("reset_peak",))


def forward_body(runtime: FakeRuntime) -> Callable[[], str]:
    def body() -> str:
        runtime.core.mark_step()
        runtime.core.mark_step(sync=True)
        runtime.core.mark_step()
        return "hidden"

    return body


class CensusCase(unittest.TestCase):
    def setUp(self) -> None:
        self.module = load_overlay(CENSUS_FILE)
        self.runtime = FakeRuntime()
        patcher = mock.patch.object(self.module, "_runtime", return_value=(self.runtime.core, self.runtime.hpu))
        patcher.start()
        self.addCleanup(patcher.stop)

    def census_line(self, logs: object) -> dict:
        lines = [line for line in logs.output if TAG + "{" in line]
        self.assertEqual(len(lines), 1, logs.output)
        return json.loads(lines[0].split(TAG, 1)[1])

    def run_census(self, body: Callable[[], str] | None = None, rows: int = 1024, context: int = 131072) -> dict:
        with mock.patch.dict(os.environ, {FLAG: "1"}), self.assertLogs("vllm.r589", "WARNING") as logs:
            self.module.census_forward(body or forward_body(self.runtime), rows, context)()
        return self.census_line(logs)


class FlagOff(CensusCase):
    def test_the_forward_is_returned_untouched_and_nothing_is_read(self) -> None:
        body = forward_body(self.runtime)
        self.assertIs(self.module.census_forward(body, 1024, 131072), body)
        with mock.patch.dict(os.environ, {FLAG: "0"}):
            self.assertIs(self.module.census_forward(body, 1024, 131072), body)
        self.assertEqual((self.runtime.reads, self.runtime.events), (0, []))

    def test_a_bad_value_refuses_with_the_fix(self) -> None:
        with mock.patch.dict(os.environ, {FLAG: "on"}):
            with self.assertRaisesRegex(ValueError, f"{FLAG}='on': set 1 .* 0 / unset"):
                self.module.census_enabled()


class LogOnly(CensusCase):
    def test_mark_step_order_and_arguments_are_unchanged(self) -> None:
        forward_body(self.runtime)()
        expected = list(self.runtime.events)
        self.runtime.events.clear()
        self.run_census()
        self.assertEqual(self.runtime.events, expected)

    def test_no_synchronize_and_no_peak_reset(self) -> None:
        self.run_census()
        self.assertEqual({event[0] for event in self.runtime.events}, {"mark_step"})

    def test_the_result_and_mark_step_return_pass_through(self) -> None:
        results: list[object] = []

        def body() -> str:
            results.append(self.runtime.core.mark_step())
            return "hidden"

        with mock.patch.dict(os.environ, {FLAG: "1"}), self.assertLogs("vllm.r589", "WARNING"):
            self.assertEqual(self.module.census_forward(body, 512, 0)(), "hidden")
        self.assertEqual(results, ["stepped"])

    def test_mark_step_is_restored_after_a_forward(self) -> None:
        original = self.runtime.core.mark_step
        self.run_census()
        self.assertIs(self.runtime.core.mark_step, original)

    def test_a_failing_forward_restores_and_reraises_and_still_logs(self) -> None:
        original = self.runtime.core.mark_step

        def body() -> str:
            raise RuntimeError("PT_DEVMEM")

        with mock.patch.dict(os.environ, {FLAG: "1"}), self.assertLogs("vllm.r589", "WARNING") as logs:
            with self.assertRaisesRegex(RuntimeError, "PT_DEVMEM"):
                self.module.census_forward(body, 512, 0)()
        self.assertIs(self.runtime.core.mark_step, original)
        self.assertEqual(self.census_line(logs)["mark_steps"], 0)

    def test_a_failing_read_never_touches_the_forward(self) -> None:
        self.runtime.fail_reads = True
        report = self.run_census()
        self.assertEqual(report["mark_steps"], 3)
        self.assertEqual(report["read_errors"], 5)  # the start, each of the 3 steps, the end
        self.assertEqual(len(self.runtime.events), 3)

    def test_a_read_failing_only_at_the_start_still_attributes_later_steps(self) -> None:
        reads = iter([RuntimeError("late"), None, None, None, None])
        real = self.runtime.hpu.memory_stats

        def flaky() -> dict[str, int]:
            failure = next(reads)
            if failure is not None:
                raise failure
            return real()

        self.runtime.hpu.memory_stats = flaky
        report = self.run_census()
        self.assertEqual(report["read_errors"], 1)
        self.assertEqual(sum(site["growth"] for site in report["sites"]), 20 * MIB)  # the first step only sets the base


class Report(CensusCase):
    def test_rows_context_and_totals(self) -> None:
        report = self.run_census(rows=2048, context=262144)
        self.assertEqual((report["rows"], report["context"], report["mark_steps"]), (2048, 262144, 3))
        self.assertEqual((report["start_in_use"], report["end_in_use"]), (1000 * MIB, 1030 * MIB))
        self.assertEqual((report["max_in_use"], report["limit"]), (1030 * MIB, 96_894 * MIB))
        self.assertEqual(report["pid"], os.getpid())

    def test_growth_max_growth_and_peak_advance_land_on_their_site(self) -> None:
        def body() -> str:
            for _ in range(2):
                self.runtime.core.mark_step()
            return "hidden"

        report = self.run_census(body)
        site = report["sites"][0]
        self.assertEqual((site["calls"], site["growth"], site["max_growth"], site["peak_advance"]),
                         (2, 20 * MIB, 10 * MIB, 20 * MIB))

    def test_bytes_after_the_last_step_are_unattributed(self) -> None:
        def body() -> str:
            self.runtime.core.mark_step()
            self.runtime.in_use += 7 * MIB
            return "hidden"

        self.assertEqual(self.run_census(body)["unattributed"], 7 * MIB)

    def test_the_line_is_one_line_of_json(self) -> None:
        with mock.patch.dict(os.environ, {FLAG: "1"}), self.assertLogs("vllm.r589", "WARNING") as logs:
            self.module.census_forward(forward_body(self.runtime), 512, 0)()
        self.assertNotIn("\n", logs.output[0])


def called_from(filename: str, source: str, core: object, **names: object) -> object:
    """Run ``source`` as if it lived in ``filename``, so the census sees that file in its frames."""
    code = compile(source, filename, "exec")
    scope = {"core": core, **names}
    exec(code, scope)  # noqa: S102  fixed test source
    return scope


class SiteAttribution(CensusCase):
    def sites_of(self, body: Callable[[], str]) -> dict[str, dict]:
        return {site["site"]: site for site in self.run_census(body)["sites"]}

    def test_boundary_sites_split_by_their_label(self) -> None:
        scope = called_from("/pkg/glm53_r589_bounded_prefill.py", (
            "def _boundary(label):\n    core.mark_step(sync=True)\n"), self.runtime.core)

        def body() -> str:
            scope["_boundary"]("moe.tile")
            scope["_boundary"]("kda.tile")
            scope["_boundary"]("moe.tile")
            return "hidden"

        sites = self.sites_of(body)
        self.assertEqual(sorted(sites), ["glm53_r589_bounded_prefill.py:2[kda.tile]", "glm53_r589_bounded_prefill.py:2[moe.tile]"])
        self.assertEqual(sites["glm53_r589_bounded_prefill.py:2[moe.tile]"]["op"], "moe")
        self.assertEqual(sites["glm53_r589_bounded_prefill.py:2[kda.tile]"]["op"], "kda")
        self.assertEqual(sites["glm53_r589_bounded_prefill.py:2[moe.tile]"]["calls"], 2)

    def test_a_shared_helper_is_keyed_by_the_op_that_called_it(self) -> None:
        helper = called_from("/pkg/hpu_communicator.py", "def all_gather():\n    core.mark_step()\n", self.runtime.core)
        for filename, op in (("/pkg/vllm_gaudi/ops/hpu_glm5next_q4kv.py", "sparse_mla"),
                             ("/pkg/vllm_gaudi/ops/hpu_glm5next_kpool.py", "kpool"),
                             ("/pkg/vllm_gaudi/ops/hpu_glm5next_sparse_indexer.py", "indexer"),
                             ("/pkg/vllm_gaudi/ops/hpu_glm5next_kda.py", "kda"),
                             ("/pkg/vllm_gaudi/ops/hpu_fp8.py", "moe")):
            with self.subTest(op=op):
                caller = called_from(filename, "def fetch():\n    gather()\n", self.runtime.core,
                                     gather=helper["all_gather"])

                def body() -> str:
                    caller["fetch"]()
                    return "hidden"

                site = next(iter(self.sites_of(body).values()))
                self.assertEqual(site["op"], op)
                self.assertTrue(site["site"].startswith("hpu_communicator.py:2<"), site["site"])

    def test_an_unknown_caller_is_other(self) -> None:
        self.assertEqual({site["op"] for site in self.run_census()["sites"]}, {"other"})


class GraphForward(CensusCase):
    """Captured graphs: capture and replay readings, with the peak window restarted at the forward."""

    def run_graph(self, body: Callable[[], str] | None = None, captured: bool = False) -> tuple[list[dict], object]:
        self.module.GRAPH_LOG.clear()
        counter = {"graphs": 19}

        def forward() -> str:
            result = (body or forward_body(self.runtime))()
            counter["graphs"] += 1 if captured else 0
            return result

        with self.assertLogs("vllm.r589", "WARNING") as logs:
            result = self.module.census_graph_forward(forward, 1, 131072, lambda: counter["graphs"])()
        lines = [json.loads(line.split(TAG, 1)[1]) for line in logs.output if TAG + "{" in line]
        return lines, result

    def test_a_replay_line_has_its_path_kind_and_result(self) -> None:
        lines, result = self.run_graph()
        self.assertEqual((lines[0]["path"], lines[0]["kind"], result), ("graph", "replay", "hidden"))

    def test_a_grown_graph_count_is_a_capture(self) -> None:
        self.assertEqual(self.run_graph(captured=True)[0][0]["kind"], "capture")

    def test_capture_sites_are_attributed(self) -> None:
        lines, _ = self.run_graph(captured=True)
        self.assertEqual(sum(site["growth"] for site in lines[0]["sites"]), 30 * MIB)

    def test_mark_step_order_is_unchanged_and_nothing_syncs(self) -> None:
        forward_body(self.runtime)()
        expected = [event for event in self.runtime.events if event[0] == "mark_step"]
        self.runtime.events.clear()
        self.run_graph()
        self.assertEqual([event for event in self.runtime.events if event[0] == "mark_step"], expected)
        self.assertNotIn(("synchronize",), self.runtime.events)

    def _reset_to_in_use(self) -> None:
        self.runtime.top = self.runtime.in_use

    def test_the_peak_restarts_at_the_forward_and_the_prior_peak_is_kept_as_between(self) -> None:
        self.runtime.top = 5000 * MIB
        self.runtime.hpu.reset_peak_memory_stats = self._reset_to_in_use
        lines, _ = self.run_graph()
        self.assertEqual(lines[0]["between_peak"], 5000 * MIB)
        self.assertEqual(lines[0]["max_in_use"], 1030 * MIB)

    def test_a_failing_reset_never_touches_the_forward(self) -> None:
        self.runtime.hpu.reset_peak_memory_stats = mock.Mock(side_effect=RuntimeError("gone"))
        lines, result = self.run_graph()
        self.assertEqual((result, lines[0]["peak_reset_failed"]), ("hidden", True))

    def test_a_failing_read_never_touches_the_forward(self) -> None:
        self.runtime.fail_reads = True
        lines, result = self.run_graph()
        self.assertEqual((result, lines[0]["between_peak"]), ("hidden", None))

    def test_replay_lines_are_throttled_but_a_new_peak_and_every_capture_log(self) -> None:
        self.module.GRAPH_LOG.clear()
        logged = 0
        for _ in range(self.module.GRAPH_LOG_FIRST + 10):
            with mock.patch.object(self.module.LOG, "warning") as warning:
                self.module.census_graph_forward(lambda: "hidden", 1, 131072, lambda: 19)()
            logged += warning.call_count
        self.assertEqual(logged, self.module.GRAPH_LOG_FIRST)

        def spike() -> str:
            self.runtime.top += 900 * MIB
            return "hidden"

        with self.assertLogs("vllm.r589", "WARNING") as logs:
            self.module.census_graph_forward(spike, 1, 131072, lambda: 19)()
        self.assertEqual(len(logs.output), 1)
        count = iter((19, 20))
        with self.assertLogs("vllm.r589", "WARNING") as logs:
            self.module.census_graph_forward(lambda: "hidden", 1, 131072, lambda: next(count))()
        self.assertEqual(len(logs.output), 1)


class DualCompatible(CensusCase):
    def test_the_dual_4m_summary_format_is_kept_for_step_5(self) -> None:
        census = self.module.MarkStepCensus(self.runtime.core, self.runtime.hpu)
        with census:
            self.runtime.core.mark_step()
        summary = census.summary()
        self.assertRegex(summary, r"^mark_steps=1 growth_bytes=10485760 unattributed_bytes=0 top_sites=\S+:1:10485760$")


class RealRuntime(unittest.TestCase):
    def test_the_patched_core_is_the_module_callers_import(self) -> None:
        import habana_frameworks.torch.core as htcore

        core, hpu = load_overlay(CENSUS_FILE)._runtime()
        self.assertIs(core, htcore)
        self.assertTrue(callable(hpu.memory_stats))


class GuardHook(unittest.TestCase):
    """The hook in RoutedForward._eager, inside hbm.prompt so the guard's peak window starts first."""

    def setUp(self) -> None:
        self.guard = load_overlay(GUARD_FILE)
        self.calls: list[object] = []
        self.chunk = self.guard.Chunk(1024, 131072)

    def fake_self(self, hbm: object = None) -> SimpleNamespace:
        return SimpleNamespace(route=SimpleNamespace(adaptive=lambda *a, **k: "hidden"), hbm=hbm)

    def eager(self, owner: SimpleNamespace, guarded: bool) -> object:
        return self.guard.RoutedForward._eager(owner, self.guard.EAGER, (), {}, self.chunk, guarded)

    def test_flag_off_runs_the_bare_forward(self) -> None:
        with mock.patch.dict(os.environ, {FLAG: "0"}):
            with self.assertNoLogs("vllm.r589", "WARNING"):
                self.assertEqual(self.eager(self.fake_self(), False), "hidden")

    def test_flag_on_wraps_the_chunk_rows_and_context(self) -> None:
        wrap = mock.Mock(side_effect=lambda run, rows, context: run)
        with mock.patch.object(self.guard, "census_forward", wrap):
            self.assertEqual(self.eager(self.fake_self(), False), "hidden")
        self.assertEqual(wrap.call_args.args[1:], (1024, 131072))

    def graph_owner(self) -> SimpleNamespace:
        return SimpleNamespace(route=SimpleNamespace(adaptive=lambda *a, **k: "hidden"), hbm=None,
                               mode=SimpleNamespace(census=False))

    def test_flag_off_graph_call_is_the_base_call_and_no_census_runs(self) -> None:
        wrap = mock.Mock()
        with mock.patch.dict(os.environ, {FLAG: "0"}), mock.patch.object(self.guard, "census_graph_forward", wrap):
            self.assertEqual(self.guard.RoutedForward._graph(self.graph_owner(), (), {}, False), "hidden")
        wrap.assert_not_called()

    def test_flag_on_graph_call_goes_through_the_census_with_its_shape(self) -> None:
        wrap = mock.Mock(side_effect=lambda run, rows, context, captures, decode_width=None: run)
        with mock.patch.dict(os.environ, {FLAG: "1"}), mock.patch.object(self.guard, "census_graph_forward", wrap):
            self.assertEqual(self.guard.RoutedForward._graph(self.graph_owner(), (), {}, False), "hidden")
        self.assertEqual(wrap.call_args.args[1:3], (0, 0))

    def test_the_census_sits_inside_the_guards_peak_window(self) -> None:
        order: list[str] = []

        def prompt(run: Callable[[], object], suppressed: bool, rows: int, context: int) -> object:
            order.append("guard_restart")
            return run()

        def wrap(run: Callable[[], object], rows: int, context: int) -> Callable[[], object]:
            def measured() -> object:
                order.append("census")
                return run()

            return measured

        with mock.patch.object(self.guard, "census_forward", wrap):
            result = self.eager(self.fake_self(SimpleNamespace(prompt=prompt)), True)
        self.assertEqual((result, order), ("hidden", ["guard_restart", "census"]))


if __name__ == "__main__":
    unittest.main()
