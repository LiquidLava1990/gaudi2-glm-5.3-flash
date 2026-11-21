"""Stage C1p host report: per-family busy, top op names, launches and traced vs unprofiled wall from the hook's window files and a
synthetic chrome trace; a window that lost its start is reported as the surviving tail, never as a verdict. No card."""

from __future__ import annotations

import gzip
import json
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "cells"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import glm53_graph_trace as hook  # noqa: E402
import replay_trace_report as report  # noqa: E402
import trace_verdict  # noqa: E402

RUN = "prompt-1024x8192-n3"
DECODE_RUN = "decode-16x8192-n5"
LABEL = "id_1_model/model/{layer}/{path} Unique ID 5"


def device(name: str, path: str, start: float, end: float, layer: int = 3) -> dict:
    return {"ph": "X", "name": name, "cat": "kernel", "pid": 0, "tid": 7, "ts": start, "dur": end - start,
            "args": {"EventName": LABEL.format(layer=layer, path=path), "HW event name": "tpc"}}


def launch(at: float) -> dict:
    return {"ph": "X", "name": "ExecuteCachedGraph", "cat": "privateuse1_runtime", "pid": 100, "tid": 1, "ts": at, "dur": 5.0}


def trace(window_us: float, ops: list[dict], launches: tuple[float, ...] = (5.0,)) -> list[dict]:
    meta = {"ph": "M", "name": "process_name", "pid": 0, "args": {"name": "/device:HPU:0"}}
    marker = {"ph": "X", "name": trace_verdict.MARKER, "cat": "user_annotation", "pid": 100, "tid": 1, "ts": 0.0, "dur": window_us}
    return [meta, marker, *[launch(at) for at in launches], *ops]


COMPLETE_OPS = [device("fused_moe_gemm", "mlp/experts", 0, 4000), device("mla_sparse_attn", "self_attn/mla_attn", 4000, 6000),
                device("kda_fused", "self_attn", 6000, 8000, layer=4)]


def window_record(run_id: str, rank: int, window_ms: float, *, rehearsal_ms: tuple[float, ...] = (11.0, 11.0), traced: bool = True,
                  errors: list[str] | None = None) -> dict:
    kind, rest = run_id.split("-", 1)
    rows, width = (int(part) for part in rest.split("-")[0].split("x"))
    return {"schema": hook.SCHEMA, "run_id": run_id, "rank": rank, "marker": hook.MARKER, "kind": kind, "rows": rows, "width": width,
            "replay_index": 3, "tokens": rows, "padded_tokens": rows, "seq_lens": None, "shape": [rows], "request_ids": [],
            "boundaries": 0, "opened": True, "opened_profiler": traced, "opened_unix": 1700000000.0, "window_ms": window_ms,
            "trigger": {"run_id": run_id, "rehearsals": 2, "start": 0, "stop": None, "min_tokens": rows, "export_ranks": [0]},
            "rehearsals": [{"index": index, "window_ms": ms} for index, ms in enumerate(rehearsal_ms, 1)],
            "profiler_start_s": 0.1, "profiler_stop_s": 0.2, "errors": errors or [],
            "trace": hook.trace_name(run_id, rank) if traced and rank == 0 else None}


def write_run(directory: Path, run_id: str, events: list[dict] | None, window_ms: float, *, ranks: int = 4, **kwargs: object) -> None:
    """The files the hook leaves for one run: every rank's window file and, when ``events`` is given, rank 0's trace."""
    for rank in range(ranks):
        (directory / hook.window_name(run_id, rank)).write_text(json.dumps(window_record(run_id, rank, window_ms, **kwargs)))
    if events is not None:
        with gzip.open(directory / hook.trace_name(run_id, 0), "wt") as handle:
            json.dump({"traceEvents": events}, handle)


class ReportCase(unittest.TestCase):
    def setUp(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.directory = Path(tmp.name)

    def write(self, run_id: str, events: list[dict] | None, window_ms: float, *, ranks: int = 4, **kwargs: object) -> None:
        write_run(self.directory, run_id, events, window_ms, ranks=ranks, **kwargs)


class CompleteWindow(ReportCase):
    def setUp(self) -> None:
        super().setUp()
        self.write(RUN, trace(10_000.0, COMPLETE_OPS), 10.0)
        self.result = report.build_report(self.directory, [RUN], 4, [])
        self.run = self.result["runs"][0]

    def test_a_whole_window_is_complete_with_the_device_verdict(self) -> None:
        self.assertEqual((self.run["status"], self.run["verdict"]), ("COMPLETE", "DEVICE_BOUND"))
        self.assertEqual(report.gaps(self.result), [])

    def test_it_reconciles_the_traced_wall_with_the_unprofiled_twin(self) -> None:
        self.assertEqual((self.run["traced_wall_ms"], self.run["unprofiled_wall_ms"]), (10.0, 11.0))
        self.assertAlmostEqual(self.run["device_busy_ms"], 8.0, places=3)
        self.assertAlmostEqual(self.run["busy_fraction"], 0.8, places=3)
        self.assertAlmostEqual(self.run["busy_fraction_unprofiled"], 0.7273, places=3)
        self.assertAlmostEqual(self.run["profiler_overhead"], -0.0909, places=3)

    def test_it_counts_launches_and_names_the_families_and_ops_by_busy_time(self) -> None:
        self.assertEqual((self.run["graph_launches"], self.run["kernels"]), (1, 3))
        families = {item["name"]: item["busy_ms"] for item in self.run["family_top"]}
        self.assertEqual(families, {"moe": 4.0, "mla_attention": 2.0, "kda": 2.0})
        self.assertEqual([item["name"] for item in self.run["kernel_top"]], ["fused_moe_gemm", "kda_fused", "mla_sparse_attn"])
        self.assertEqual(self.run["lead_ms"], 0.0)

    def test_the_rendered_lines_carry_the_totals_and_both_tables(self) -> None:
        text = "\n".join(report.render(self.result))
        self.assertIn(f"REPLAY_TRACE_RUN run={RUN} status=COMPLETE verdict=DEVICE_BOUND", text)
        self.assertIn("unprofiled_wall_ms=11.0", text)
        self.assertIn("graph_launches=1", text)
        self.assertIn(f"REPLAY_TRACE_FAMILY run={RUN}", text)
        self.assertRegex(text, r"0\.5000\s+4\.000\s+1 moe")
        self.assertIn(f"REPLAY_TRACE_KERNEL run={RUN}", text)
        self.assertIn("fused_moe_gemm", text)
        self.assertIn("REPLAY_TRACE_SUMMARY complete=1 partial=0 missing=0", text)


class LostStart(ReportCase):
    def test_a_window_that_lost_its_start_reports_the_surviving_tail_and_no_verdict(self) -> None:
        tail = [device("fused_moe_gemm", "mlp/experts", 80_000, 90_000), device("mla_sparse_attn", "self_attn/mla_attn", 90_000, 95_000)]
        self.write(RUN, trace(100_000.0, tail, launches=(5.0, 6.0, 7.0, 8.0)), 100.0, rehearsal_ms=(110.0, 110.0))
        result = report.build_report(self.directory, [RUN], 4, [])
        run = result["runs"][0]
        self.assertEqual((run["status"], run["verdict"]), ("PARTIAL", "INCOMPLETE"))
        self.assertEqual((run["lead_ms"], run["covered_ms"], run["traced_wall_ms"]), (80.0, 20.0, 100.0))
        self.assertAlmostEqual(run["busy_fraction_of_covered"], 0.75, places=3)
        self.assertTrue(any("lost the window's start" in reason for reason in run["reasons"]))
        self.assertEqual({item["name"] for item in run["family_top"]}, {"moe", "mla_attention"})
        text = "\n".join(report.render(result))
        self.assertIn("status=PARTIAL", text)
        self.assertIn("covered_ms=20.0", text)
        self.assertIn("REPLAY_TRACE_REASON", text)
        self.assertEqual(len(report.gaps(result)), 1)


class MissingWindow(ReportCase):
    def test_a_target_with_no_window_files_is_missing_and_named(self) -> None:
        result = report.build_report(self.directory, [RUN], 4, [])
        run = result["runs"][0]
        self.assertEqual(run["status"], "MISSING")
        self.assertIn(hook.window_name(RUN, 0), " ".join(run["reasons"]))
        self.assertEqual(len(report.gaps(result)), 1)

    def test_a_refused_window_without_a_trace_reports_the_hook_error(self) -> None:
        self.write(RUN, None, 10.0, traced=False, errors=["free HBM 900 MiB is below the 1024 MiB floor (KB C341): profiler not armed"])
        run = report.build_report(self.directory, [RUN], 4, [])["runs"][0]
        self.assertEqual(run["status"], "MISSING")
        self.assertTrue(any("900 MiB" in reason for reason in run["reasons"]), run["reasons"])

    def test_one_missing_rank_file_is_named_even_when_rank_zero_wrote_its_trace(self) -> None:
        self.write(RUN, trace(10_000.0, COMPLETE_OPS), 10.0)
        (self.directory / hook.window_name(RUN, 3)).unlink()
        run = report.build_report(self.directory, [RUN], 4, [])["runs"][0]
        self.assertNotEqual(run["status"], "COMPLETE")
        self.assertIn(hook.window_name(RUN, 3), " ".join(run["reasons"]))


class CensusCrossCheck(ReportCase):
    def line(self, replay_ms: float | None, rows: int = 1024, context: int = 8192, kind: str = "replay") -> str:
        body = {"path": "graph", "kind": kind, "rows": rows, "context": context, **({"replay_ms": replay_ms} if replay_ms else {})}
        return f"2026-11-07 INFO [vllm.r589] GLM53_PROMPT_CENSUS {json.dumps(body)}"

    def test_the_median_timed_replay_of_the_key_is_read_from_the_container_log(self) -> None:
        lines = [self.line(3800.0), self.line(3900.0), self.line(9000.0), self.line(None), self.line(5.0, context=4096), self.line(3000.0, kind="capture")]
        self.assertEqual(report.census_replay_ms(lines, 1024, 8192), (3900.0, 3))

    def test_no_timed_replay_is_none(self) -> None:
        self.assertEqual(report.census_replay_ms([self.line(None)], 1024, 8192), (None, 0))

    def test_a_prompt_run_carries_the_census_median_and_a_decode_run_does_not(self) -> None:
        self.write(RUN, trace(10_000.0, COMPLETE_OPS), 10.0)
        self.write(DECODE_RUN, trace(10_000.0, COMPLETE_OPS), 10.0)
        runs = report.build_report(self.directory, [RUN, DECODE_RUN], 4, [self.line(3800.0)])["runs"]
        self.assertEqual(runs[0]["census_replay_ms"], 3800.0)
        self.assertIsNone(runs[1]["census_replay_ms"])


class Wait(ReportCase):
    def test_wait_returns_when_every_rank_file_is_there(self) -> None:
        self.write(RUN, trace(10_000.0, COMPLETE_OPS), 10.0)
        self.assertEqual(report.wait_for_windows(self.directory, [RUN], 4, timeout_s=5.0, poll_s=1.0, sleep=lambda _: None), [])

    def test_wait_names_what_is_still_missing_after_the_timeout(self) -> None:
        self.write(RUN, trace(10_000.0, COMPLETE_OPS), 10.0)
        (self.directory / hook.window_name(RUN, 2)).unlink()
        ticks = iter(range(100))
        missing = report.wait_for_windows(self.directory, [RUN], 4, timeout_s=3.0, poll_s=1.0, sleep=lambda _: None,
                                          clock=lambda: float(next(ticks)))
        self.assertEqual(missing, [hook.window_name(RUN, 2)])


class CommandLine(ReportCase):
    def run_main(self, *argv: str) -> tuple[int, str]:
        out = StringIO()
        with redirect_stdout(out):
            code = report.main(list(argv))
        return code, out.getvalue()

    def test_report_prints_writes_json_and_exits_zero_only_when_every_run_is_complete(self) -> None:
        self.write(RUN, trace(10_000.0, COMPLETE_OPS), 10.0)
        target = self.directory / "report.json"
        code, text = self.run_main("report", "--dir", str(self.directory), "--runs", RUN, "--out", str(target))
        self.assertEqual(code, 0)
        self.assertIn("REPLAY_TRACE_RUN", text)
        self.assertEqual(json.loads(target.read_text())["runs"][0]["status"], "COMPLETE")
        code, _ = self.run_main("report", "--dir", str(self.directory), "--runs", f"{RUN},{DECODE_RUN}", "--out", str(target))
        self.assertEqual(code, report.EXIT_INCOMPLETE)

    def test_wait_exits_three_when_files_are_missing(self) -> None:
        code, text = self.run_main("wait", "--dir", str(self.directory), "--runs", RUN, "--timeout-s", "0", "--poll-s", "0")
        self.assertEqual(code, report.EXIT_MISSING)
        self.assertIn(hook.window_name(RUN, 0), text)


if __name__ == "__main__":
    unittest.main()
