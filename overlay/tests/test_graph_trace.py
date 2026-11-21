"""Stage C1p hook: a device trace of the Nth warmed replay of one named prompt key and of one named decode key. Fake profiler, no card.

Covers the plan grammar and its refusals, which replay is bracketed, the rehearsals that give the unprofiled twin, the free-HBM refusal,
the files the host report reads, and that with the flag unset the module is never imported.
"""

from __future__ import annotations

import functools
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "cells"))

import trace_verdict  # noqa: E402
from overlay_testlib import load_overlay  # noqa: E402

TRACE_FILE = "glm53_graph_trace.py"
CENSUS_FILE = "glm53_prompt_census.py"
PROMPT = "prompt:1024x8192@3"
DECODE = "decode:16x8192@5"


class FakeProfiler:
    def __init__(self, log: list[str]) -> None:
        self.log = log

    def start(self) -> None:
        self.log.append("profiler.start")

    def stop(self) -> None:
        self.log.append("profiler.stop")

    def export_chrome_trace(self, path: str) -> None:
        self.log.append("profiler.export")
        Path(path).write_bytes(b"{}")


class FakeMarker:
    def __init__(self, log: list[str], name: str) -> None:
        self.log, self.name = log, name

    def __enter__(self) -> FakeMarker:
        self.log.append(f"marker.enter {self.name}")
        return self

    def __exit__(self, *exc: object) -> None:
        self.log.append("marker.exit")


class FakeRuntime:
    def __init__(self, rank: int = 0, free_mib: float | None = 4096.0) -> None:
        self.log: list[str] = []
        self.rank_number, self.free = rank, free_mib

    def flush(self) -> None:
        self.log.append("flush")

    def profiler(self) -> FakeProfiler:
        return FakeProfiler(self.log)

    def marker(self, name: str) -> FakeMarker:
        return FakeMarker(self.log, name)

    def rank(self) -> int:
        return self.rank_number

    def free_mib(self) -> float | None:
        return self.free


class FakeClock:
    """Each read is half a second later, so every window has a length and the rehearsal walls are comparable."""

    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        self.now += 0.5
        return self.now


class TraceCase(unittest.TestCase):
    def setUp(self) -> None:
        self.module = load_overlay(TRACE_FILE)
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.directory = Path(tmp.name)
        self.runtime = FakeRuntime()

    def tracer(self, spec: str = f"{PROMPT},{DECODE}", rehearsals: int = 2, runtime: FakeRuntime | None = None):
        plan = self.module.parse_plan(str(self.directory), spec, str(rehearsals))
        tracer = self.module.ReplayTracer(plan, runtime or self.runtime, FakeClock(), FakeClock())
        self.addCleanup(tracer.wait_exports)  # runs before the directory's cleanup: an export thread may still be writing
        return tracer

    def prompt(self, tracer, *, result: str = "hidden", rows: int = 1024, width: int = 8192, key_replay: bool = True, body=None):
        return tracer.wrap(body or (lambda: result), rows=rows, context=width, captures=lambda: 0, key_replay=key_replay, decode_width=None)()

    def decode(self, tracer, *, rows: int = 16, width: int = 8192, counter=None, body=None, result: str = "tokens"):
        return tracer.wrap(body or (lambda: result), rows=rows, context=0, captures=counter or (lambda: 0), key_replay=None, decode_width=width)()

    def record(self, run_id: str, rank: int = 0) -> dict:
        return json.loads((self.directory / self.module.window_name(run_id, rank)).read_text())


class PlanGrammar(TraceCase):
    def test_a_plan_names_each_key_and_the_replay_to_trace(self) -> None:
        plan = self.module.parse_plan("/k1trace", f"{PROMPT}, {DECODE}", "2")
        self.assertEqual([(t.kind, t.rows, t.width, t.nth) for t in plan.targets], [("prompt", 1024, 8192, 3), ("decode", 16, 8192, 5)])
        self.assertEqual([t.run_id for t in plan.targets], ["prompt-1024x8192-n3", "decode-16x8192-n5"])
        self.assertEqual(plan.rehearsals, 2)

    def test_every_malformed_plan_refuses_and_names_the_fix(self) -> None:
        cases = (("", "2", "names no key"), ("prompt:1024@3", "2", "KIND:ROWSxWIDTH@NTH"), ("prompt:1024x8192", "2", "KIND:ROWSxWIDTH@NTH"),
                 ("stall:1x1@3", "2", "KIND:ROWSxWIDTH@NTH"), ("prompt:0x8192@3", "2", "KIND:ROWSxWIDTH@NTH"),
                 (f"{PROMPT},{PROMPT}", "2", "twice"), ("prompt:1024x8192@2", "2", "rehearsals"), ("decode:16x8192@1", "0", "decode"))
        for spec, rehearsals, words in cases:
            with self.subTest(spec=spec), self.assertRaisesRegex(self.module.ReplayTraceError, words):
                self.module.parse_plan("/k1trace", spec, rehearsals)

    def test_the_directory_and_the_rehearsal_count_are_checked(self) -> None:
        for directory, rehearsals, words in (("k1trace", "2", "absolute"), ("", "2", "absolute"), ("/k1trace", "9", "0..8"), ("/k1trace", "x", "0..8")):
            with self.subTest(directory=directory, rehearsals=rehearsals), self.assertRaisesRegex(self.module.ReplayTraceError, words):
                self.module.parse_plan(directory, PROMPT, rehearsals)

    def test_env_unset_is_no_plan_and_a_directory_without_a_plan_refuses(self) -> None:
        self.assertIsNone(self.module.plan_from_env({}))
        self.assertIsNone(self.module.plan_from_env({self.module.DIR_ENV: "  "}))
        with self.assertRaisesRegex(self.module.ReplayTraceError, self.module.PLAN_ENV):
            self.module.plan_from_env({self.module.DIR_ENV: "/k1trace"})
        plan = self.module.plan_from_env({self.module.DIR_ENV: "/k1trace", self.module.PLAN_ENV: PROMPT})
        self.assertEqual(plan.rehearsals, self.module.DEFAULT_REHEARSALS)

    def test_the_dry_run_description_lists_each_traced_replay_its_rehearsals_and_files(self) -> None:
        text = self.module.describe_plan(self.module.parse_plan("/k1trace", f"{PROMPT},{DECODE}", "2"))
        prompt, decode = text["targets"]
        self.assertEqual((prompt["traced_replay"], prompt["rehearsal_replays"]), (3, [1, 2]))
        self.assertEqual((decode["traced_replay"], decode["rehearsal_replays"]), (5, [3, 4]))
        self.assertIn("replaytrace-prompt-1024x8192-n3-rank0.pt.trace.json.gz", prompt["files"])
        self.assertEqual(text["min_free_mib"], 1024)
        self.assertTrue(any("HPU_GLM53_PREFILL_HBM_CENSUS=1" in need for need in text["needs"]))


class FirstFlush(TraceCase):
    def test_a_first_flush_that_raises_is_recorded_and_the_forward_still_runs(self) -> None:
        tracer = self.tracer(PROMPT)
        self.prompt(tracer)
        self.prompt(tracer)
        with mock.patch.object(self.runtime, "flush", side_effect=[RuntimeError("drain failed"), None, None]):
            self.assertEqual(self.prompt(tracer, result="served"), "served")
        tracer.wait_exports()
        self.assertTrue(any("drain failed" in error for error in self.record(tracer.plan.targets[0].run_id)["errors"]))


class PromptReplay(TraceCase):
    def test_the_third_warmed_replay_is_bracketed_with_the_profiler_and_returns_its_result(self) -> None:
        tracer = self.tracer(PROMPT)
        results = [self.prompt(tracer, result=f"hidden{n}") for n in range(1, 5)]
        self.assertEqual(results, ["hidden1", "hidden2", "hidden3", "hidden4"])
        starts = [i for i, line in enumerate(self.runtime.log) if line == "profiler.start"]
        self.assertEqual(len(starts), 1)
        self.assertEqual(self.runtime.log.count("profiler.stop"), 1)
        self.assertEqual(self.runtime.log[starts[0] - 1], "flush", "the device is drained before the profiler starts")
        self.assertEqual(self.runtime.log[self.runtime.log.index("profiler.stop") - 2:self.runtime.log.index("profiler.stop")], ["flush", "marker.exit"])

    def test_the_two_replays_before_it_are_the_unprofiled_twins(self) -> None:
        tracer = self.tracer(PROMPT)
        for _ in range(4):
            self.prompt(tracer)
        tracer.wait_exports()
        record = self.record("prompt-1024x8192-n3")
        self.assertEqual([item["index"] for item in record["rehearsals"]], [1, 2])
        self.assertTrue(all(item["window_ms"] > 0 for item in record["rehearsals"]))
        self.assertEqual(self.runtime.log.count("flush"), 3 * 2, "every bracketed replay drains at both ends")

    def test_a_key_capture_is_not_a_replay_and_never_traced(self) -> None:
        tracer = self.tracer(PROMPT)
        self.prompt(tracer, key_replay=False)
        for _ in range(2):
            self.prompt(tracer)
        self.assertNotIn("profiler.start", self.runtime.log)
        self.prompt(tracer)
        self.assertIn("profiler.start", self.runtime.log)

    def test_other_keys_run_untouched(self) -> None:
        tracer = self.tracer(PROMPT)
        forward = lambda: "hidden"  # noqa: E731
        for rows, width in ((1024, 4096), (512, 8192)):
            wrapped = tracer.wrap(forward, rows=rows, context=width, captures=lambda: 0, key_replay=True, decode_width=None)
            self.assertIs(wrapped, forward)
        self.assertEqual(self.runtime.log, [])

    def test_a_traced_replay_fires_once_per_process(self) -> None:
        tracer = self.tracer(PROMPT)
        for _ in range(9):
            self.prompt(tracer)
        self.assertEqual(self.runtime.log.count("profiler.start"), 1)
        self.assertEqual(tracer.unfired(), [])

    def test_the_window_file_has_the_fields_the_host_report_needs(self) -> None:
        tracer = self.tracer(PROMPT)
        for _ in range(3):
            self.prompt(tracer)
        tracer.wait_exports()
        record = self.record("prompt-1024x8192-n3")
        self.assertEqual((record["kind"], record["rows"], record["width"], record["replay_index"]), ("prompt", 1024, 8192, 3))
        self.assertEqual((record["tokens"], record["trigger"]["min_tokens"], record["rank"]), (1024, 1024, 0))
        self.assertTrue(record["opened"] and record["window_ms"] > 0 and record["opened_unix"])
        self.assertEqual(record["trace"], "replaytrace-prompt-1024x8192-n3-rank0.pt.trace.json.gz")
        self.assertEqual((record["free_mib_before"], record["errors"]), (4096.0, []))
        self.assertTrue((self.directory / record["trace"]).is_file())
        self.assertEqual(record["marker"], trace_verdict.MARKER)

    def test_only_the_export_rank_writes_a_trace_but_every_rank_writes_its_window(self) -> None:
        other = FakeRuntime(rank=1)
        tracer = self.tracer(PROMPT, runtime=other)
        for _ in range(3):
            self.prompt(tracer)
        tracer.wait_exports()
        record = self.record("prompt-1024x8192-n3", rank=1)
        self.assertIsNone(record["trace"])
        self.assertNotIn("profiler.export", other.log)
        self.assertIn("profiler.stop", other.log)


class DecodeReplay(TraceCase):
    def test_a_decode_call_that_grew_the_graph_count_is_a_capture_and_does_not_count(self) -> None:
        graphs = {"count": 0}
        tracer = self.tracer(DECODE)

        def capture_body() -> str:
            graphs["count"] += 1
            return "tokens"

        def call(body) -> str:
            return tracer.wrap(body, rows=16, context=0, captures=lambda: graphs["count"], key_replay=None, decode_width=8192)()

        call(capture_body)
        for _ in range(4):
            call(lambda: "tokens")
        self.assertNotIn("profiler.start", self.runtime.log)
        call(lambda: "tokens")
        self.assertEqual(self.runtime.log.count("profiler.start"), 1)

    def test_the_fifth_decode_replay_is_traced(self) -> None:
        tracer = self.tracer(DECODE)
        for _ in range(6):
            self.assertEqual(self.decode(tracer), "tokens")
        self.assertEqual(self.runtime.log.count("profiler.start"), 1)
        tracer.wait_exports()
        record = self.record("decode-16x8192-n5")
        self.assertEqual([item["index"] for item in record["rehearsals"]], [3, 4])

    def test_a_step_with_no_table_width_is_not_a_decode_key(self) -> None:
        tracer = self.tracer(DECODE)
        forward = lambda: "x"  # noqa: E731
        wrapped = tracer.wrap(forward, rows=16, context=0, captures=lambda: 0, key_replay=None, decode_width=None)
        self.assertIs(wrapped, forward)

    def test_another_width_or_row_count_is_not_this_key(self) -> None:
        tracer = self.tracer(DECODE)
        forward = lambda: "x"  # noqa: E731
        for rows, width in ((8, 8192), (16, 4096)):
            self.assertIs(tracer.wrap(forward, rows=rows, context=0, captures=lambda: 0, key_replay=None, decode_width=width), forward)


class Refusals(TraceCase):
    def test_low_free_hbm_refuses_to_arm_the_profiler_and_says_so(self) -> None:
        low = FakeRuntime(free_mib=900.0)
        tracer = self.tracer(PROMPT, runtime=low)
        for _ in range(3):
            self.assertEqual(self.prompt(tracer), "hidden")
        tracer.wait_exports()
        self.assertNotIn("profiler.start", low.log)
        record = self.record("prompt-1024x8192-n3")
        self.assertFalse(record["opened_profiler"])
        self.assertIn("900", record["errors"][0])
        self.assertIn("1024", record["errors"][0])
        self.assertIsNone(record["trace"])

    def test_an_unreadable_allocator_also_refuses(self) -> None:
        blind = FakeRuntime(free_mib=None)
        tracer = self.tracer(PROMPT, runtime=blind)
        for _ in range(3):
            self.prompt(tracer)
        tracer.wait_exports()
        self.assertNotIn("profiler.start", blind.log)
        self.assertIn("could not be read", self.record("prompt-1024x8192-n3")["errors"][0])

    def test_a_forward_that_raises_in_the_window_stops_the_profiler_and_still_raises(self) -> None:
        tracer = self.tracer(PROMPT)
        for _ in range(2):
            self.prompt(tracer)

        def broken() -> str:
            raise RuntimeError("replay died")

        with self.assertRaisesRegex(RuntimeError, "replay died"):
            self.prompt(tracer, body=broken)
        tracer.wait_exports()
        self.assertEqual(self.runtime.log.count("profiler.stop"), 1)
        self.assertIn("replay died", self.record("prompt-1024x8192-n3")["errors"][0])

    def test_a_traced_call_that_was_a_capture_is_recorded_as_an_error(self) -> None:
        tracer = self.tracer(DECODE, rehearsals=2)
        for _ in range(4):
            self.decode(tracer)
        count = iter([4, 5])  # before, after: the graph count moved
        tracer.wrap(lambda: "tokens", rows=16, context=0, captures=lambda: next(count), key_replay=None, decode_width=8192)()
        tracer.wait_exports()
        self.assertIn("capture", self.record("decode-16x8192-n5")["errors"][0])

    def test_a_profiler_that_fails_to_start_is_recorded_and_never_raised(self) -> None:
        self.runtime.profiler = mock.Mock(side_effect=RuntimeError("no hpu activity"))
        tracer = self.tracer(PROMPT)
        for _ in range(3):
            self.assertEqual(self.prompt(tracer), "hidden")
        tracer.wait_exports()
        self.assertIn("no hpu activity", self.record("prompt-1024x8192-n3")["errors"][0])

    def test_unfired_targets_are_listed_for_the_exit_log(self) -> None:
        tracer = self.tracer(f"{PROMPT},{DECODE}")
        self.prompt(tracer)
        unfired = tracer.unfired()
        self.assertEqual(len(unfired), 2)
        self.assertIn("replays_seen=1", unfired[0])
        self.assertIn("replays_seen=0", unfired[1])


class VerdictCompatibility(TraceCase):
    def test_the_window_file_passes_the_token_and_window_checks_of_the_existing_verdict(self) -> None:
        tracer = self.tracer(PROMPT)
        for _ in range(3):
            self.prompt(tracer)
        tracer.wait_exports()
        record = self.record("prompt-1024x8192-n3")
        self.assertEqual(trace_verdict.window_reasons(record), [])


class FlagOff(unittest.TestCase):
    def test_census_with_the_flag_unset_never_imports_the_hook(self) -> None:
        sys.modules.pop("glm53_graph_trace", None)
        env = {key: value for key, value in os.environ.items() if key != "HPU_GLM53_REPLAY_TRACE_DIR"}
        with mock.patch.dict(os.environ, env, clear=True):
            sys.modules.pop("overlay_probe_census", None)
            import importlib.util
            spec = importlib.util.spec_from_file_location("overlay_probe_census", Path(__file__).resolve().parents[1] / CENSUS_FILE)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
        self.assertIsNone(module.REPLAY_TRACER)
        self.assertNotIn("glm53_graph_trace", sys.modules)

    def test_the_flag_without_the_census_flag_refuses_the_import(self) -> None:
        env = {"HPU_GLM53_REPLAY_TRACE_DIR": "/k1trace", "HPU_GLM53_REPLAY_TRACE": PROMPT, "HPU_GLM53_PREFILL_HBM_CENSUS": "0"}
        import importlib.util
        spec = importlib.util.spec_from_file_location("overlay_probe_census_off", Path(__file__).resolve().parents[1] / CENSUS_FILE)
        module = importlib.util.module_from_spec(spec)
        with mock.patch.dict(os.environ, env), self.assertRaisesRegex(ValueError, "HPU_GLM53_PREFILL_HBM_CENSUS=1"):
            spec.loader.exec_module(module)


class CensusWiring(TraceCase):
    """census_graph_forward hands every key forward to the tracer, and decode forwards their table width."""

    def setUp(self) -> None:
        super().setUp()
        self.census = load_overlay(CENSUS_FILE)
        self.census.GRAPH_LOG.clear()

    def test_the_tracer_wraps_the_census_forward_with_the_key_the_caller_named(self) -> None:
        fake = mock.Mock()
        fake.wrap.side_effect = lambda forward, **keys: forward
        with mock.patch.object(self.census, "REPLAY_TRACER", fake), mock.patch.dict(os.environ, {"HPU_GLM53_PREFILL_HBM_CENSUS": "1"}):
            self.census.census_graph_forward(lambda: "x", 16, 0, lambda: 0, decode_width=8192)
        keys = fake.wrap.call_args.kwargs
        self.assertEqual((keys["rows"], keys["context"], keys["key_replay"], keys["decode_width"]), (16, 0, None, 8192))

    def test_no_tracer_returns_the_census_forward_unchanged(self) -> None:
        with mock.patch.object(self.census, "REPLAY_TRACER", None):
            first = self.census.census_graph_forward(lambda: "x", 16, 0, lambda: 0)
        self.assertIsInstance(first, functools.partial)
        self.assertIs(first.func, self.census._measured_graph_forward)


if __name__ == "__main__":
    unittest.main()
