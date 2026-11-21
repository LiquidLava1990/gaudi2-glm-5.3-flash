"""C1 scorer (plan section 12.3/12.4): one boot's docker log plus the drive rows in, a PASS/FAIL/NOT_MEASURED reading each out."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "cells"))

import c1_score as scorer  # noqa: E402

MIB = 1 << 20
RANKS = (0, 1, 2, 3)
LANE_ENV = Path(__file__).resolve().parents[2] / "graphs-c1.env"
PROMPT_KEYS = "1024:8192,1024:4096,1024:2048,1024:1024"
DECODE_KEYS = ",".join(f"{rows}x{width}" for width in (8192, 4096, 1024) for rows in (16, 8, 4, 2, 1)) + ",1x2048"
WIDTHS = {1024: 1500.0, 2048: 2200.0, 4096: 3000.0, 8192: 5000.0}
BASE_IN_USE_MIB = 1000


def who(rank: int) -> str:
    return f"(Worker_TP{rank}_EP{rank} pid={100 + rank}) WARNING "


def census(rank: int, kind: str, width: int, *, replay_ms: float | None = None, path: str = "graph", rows: int = 1024) -> str:
    start = BASE_IN_USE_MIB * MIB
    body = {"path": path, "kind": kind, "rows": rows, "context": width, "pid": 100 + rank, "start_in_use": start,
            "end_in_use": start + 1500 * MIB, "max_in_use": start + 2600 * MIB, "limit": 90_000 * MIB,
            "between_peak": start + 1500 * MIB, "peak_reset_failed": False, "n": 1, "sites": []}
    if replay_ms is not None:
        body["replay_ms"] = replay_ms
    return f"{who(rank)}GLM53_PROMPT_CENSUS {json.dumps(body, sort_keys=True)}"


def replays(rank: int, width: int, ms: float, count: int = 6, rows: int = 1024) -> list[str]:
    """The first replay awful and untimed, then timed ones (as the census writes every second replay, here every one)."""
    return [census(rank, "replay", width, replay_ms=60_000.0 if index == 0 else ms, rows=rows) for index in range(count)]


def witness(rank: int, *, prompt_keys: str = PROMPT_KEYS, decode_keys: str = DECODE_KEYS, native: int = 0, lazy: int = 0) -> str:
    return (f"{who(rank)}GLM53_BOOT_WITNESS prompt_keys={prompt_keys} decode_keys={decode_keys} decode_pad=check owner=1 "
            f"native_fallbacks={native} fallback_not_warmed=0 lazy_fallbacks={lazy} charged_MiB=7000.0 budget_MiB=8452 free_MiB=900.0")


def counters(rank: int, **counts: int) -> str:
    return f"{who(rank)}GLM53_MIXED_GRAPH counters={counts!r} prompt_tokens_per_step=0.00 decode_rows_per_step=0.00 entries=3 estimated_MiB=5000.0"


def cold_lines(rank: int, *, second_step_native: int = 0) -> list[str]:
    """Boot capture steps, then a 128-token prompt step, then two decode-only steps; every step logs its counters (LOG_EVERY=1)."""
    boot_one = {"mixed_steps": 1, "boot_steps": 1, "captures": 1}
    boot_two = {**boot_one, "mixed_steps": 2, "boot_steps": 2, "captures": 2}
    prompt = {**boot_two, "mixed_steps": 3, "live_steps": 1, "graph_replays": 1, "prompt_tokens": 128}
    first = {**prompt, "mixed_steps": 4, "live_steps": 2, "graph_replays": 2, "decode_rows": 1}
    second = {**first, "mixed_steps": 5, "live_steps": 3, "graph_replays": 3, "decode_rows": 2}
    if second_step_native:
        second = {**second, "graph_replays": 2, "native_fallbacks": second_step_native, "fallback_ctx_over": second_step_native}
    return [counters(rank, **step) for step in (boot_one, boot_two, prompt, first, second)]


def merge_line(rank: int, equal: int, narrow: int = 2048, wide: int = 4096, position: int = 200_000) -> str:
    return f"{who(rank)}GLM53_DECODE_MERGE narrow={narrow} wide={wide} rows=1 position={position} torch_equal={equal}"


def ballast_line(rank: int, total: int = 512, free: int = 900) -> str:
    return f"{who(rank)}GLM53_K1_BALLAST total_mib={total} replays=1 free_bytes={free * MIB}"


def rank_lines(rank: int, *, ms: dict[int, float] | None = None, merge: tuple[int, ...] | None = (1, 1, 1), ballast: int | None = 512,
               free: int = 900, witnessed: bool = True, cold: bool = True, skip_widths: tuple[int, ...] = (), **witness_kwargs) -> list[str]:
    lines = [witness(rank, **witness_kwargs)] if witnessed else []
    for width, value in (ms or WIDTHS).items():
        if width not in skip_widths:
            lines += [census(rank, "capture", width), *replays(rank, width, value)]
    lines += cold_lines(rank) if cold else []
    lines += [merge_line(rank, equal) for equal in merge or ()]
    lines += [ballast_line(rank, ballast, free)] if ballast is not None else []
    return lines


def mock_log(**overrides: object) -> list[str]:
    return [line for rank in RANKS for line in rank_lines(rank, **overrides)]


def with_rank(rank: int, **overrides: object) -> list[str]:
    return [line for other in RANKS for line in rank_lines(other, **(overrides if other == rank else {}))]


def lane_rows(*, skip: tuple[int, int] | None = None, status: dict[tuple[int, int], str] | None = None) -> list[dict]:
    rows = []
    for width in (1024, 4096, 8192):
        for lanes in (1, 2, 4, 8, 16):
            if (lanes, width) == skip:
                continue
            row = {"drive": "lanes", "lanes": lanes, "width": width, "status": (status or {}).get((lanes, width), "ok")}
            rows.append(row | ({"step_ms": 100.0 + lanes + width / 1000, "samples": 60} if row["status"] == "ok" else {"reason": "x"}))
    return rows


def drive_rows(**overrides: object) -> list[dict]:
    rows = [{"drive": "cold", "status": "ok"}, {"drive": "merge", "tokens": 200_000, "outputs_equal": True},
            {"drive": "long", "tokens": 150_000, "ttft_s": 300.0, "warm_ttft_s": 2.0},
            {"drive": "long", "tokens": 530_000, "ttft_s": 900.0, "warm_ttft_s": 3.0}]
    return rows + lane_rows(**overrides)


def score(lines: list[str] | None = None, rows: list[dict] | None = None, boot_seconds: float | None = 900.0, over_budget: bool = False) -> dict:
    return scorer.score(mock_log() if lines is None else lines, drive_rows() if rows is None else rows,
                        scorer.BootFacts(boot_seconds, over_budget),
                        scorer.Expected.default())


def reading(result: dict, name: str) -> dict:
    return next(item for item in result["readings"] if item["name"] == name)


class Healthy(unittest.TestCase):
    def test_a_clean_four_rank_log_passes_every_reading(self) -> None:
        result = score()
        self.assertEqual([(item["name"], item["verdict"]) for item in result["readings"] if item["verdict"] != "PASS"], [])
        self.assertEqual(result["verdict"], "PASS")

    def test_the_report_ends_with_the_verdict_and_names_every_reading(self) -> None:
        report = scorer.render_report(score())
        self.assertEqual(report.splitlines()[-1], "C1_VERDICT: PASS")
        for item in score()["readings"]:
            self.assertIn(item["name"], report)

    def test_default_expectation_is_four_prompt_keys_and_sixteen_decode_keys(self) -> None:
        expected = scorer.Expected.default()
        self.assertEqual(len(expected.prompt_keys), 4)
        self.assertEqual(len(expected.decode_keys), 16)
        self.assertIn("1x2048", expected.decode_keys)

    def test_expectation_follows_the_env_it_is_built_from(self) -> None:
        env = {"VLLM_GAUDI_MIXED_PROMPT_KEYS": "1024:8192,1024:1024", "VLLM_GAUDI_MIXED_DECODE_WIDTHS": "1024,8192",
               "HPU_GLM53_DECODE_PAD_CHECK_WIDTHS": ""}
        expected = scorer.Expected.from_env(env)
        self.assertEqual(expected.prompt_keys, frozenset({"1024:8192", "1024:1024"}))
        self.assertEqual(len(expected.decode_keys), 10)

    @unittest.skipUnless(LANE_ENV.is_file(), "graphs-c1.env sits beside the overlay on the host, not in the test container")
    def test_the_c1_env_file_gives_the_default_expectation(self) -> None:
        env = dict(line.split("=", 1) for line in LANE_ENV.read_text().splitlines() if line.strip() and not line.startswith("#"))
        self.assertEqual(scorer.Expected.from_env(env), scorer.Expected.default())


class BootWitness(unittest.TestCase):
    def test_a_missing_witness_on_one_rank_is_not_measured(self) -> None:
        result = score(with_rank(2, witnessed=False))
        self.assertEqual(reading(result, "boot_witness")["verdict"], "NOT_MEASURED")
        self.assertIn("rank 2", reading(result, "boot_witness")["detail"])

    def test_a_missing_decode_bucket_fails_and_names_it(self) -> None:
        keys = DECODE_KEYS.replace(",1x2048", "")
        result = score(with_rank(1, decode_keys=keys))
        self.assertEqual(reading(result, "boot_witness")["verdict"], "FAIL")
        self.assertIn("1x2048", reading(result, "boot_witness")["detail"])

    def test_a_missing_prompt_key_fails(self) -> None:
        result = score(with_rank(3, prompt_keys="1024:8192,1024:1024"))
        self.assertEqual(reading(result, "boot_witness")["verdict"], "FAIL")
        self.assertIn("1024:4096", reading(result, "boot_witness")["detail"])


class Fallbacks(unittest.TestCase):
    def test_a_boot_fallback_in_the_witness_fails(self) -> None:
        self.assertEqual(reading(score(with_rank(0, native=2)), "fallbacks")["verdict"], "FAIL")

    def test_a_live_fallback_counter_fails(self) -> None:
        lines = mock_log() + [counters(1, mixed_steps=900, fallback_width_over=1, native_fallbacks=1)]
        result = score(lines)
        self.assertEqual(reading(result, "fallbacks")["verdict"], "FAIL")
        self.assertIn("rank 1", reading(result, "fallbacks")["detail"])

    def test_no_witness_and_no_counters_is_not_measured(self) -> None:
        lines = [line for line in mock_log() if "GLM53_BOOT_WITNESS" not in line and "GLM53_MIXED_GRAPH" not in line]
        self.assertEqual(reading(score(lines), "fallbacks")["verdict"], "NOT_MEASURED")

    def test_one_eager_census_line_fails(self) -> None:
        lines = mock_log() + [census(0, "replay", 1024, path="eager")]
        result = score(lines)
        self.assertEqual(reading(result, "eager_census")["verdict"], "FAIL")

    def test_a_log_with_no_census_at_all_is_not_measured(self) -> None:
        lines = [line for line in mock_log() if "GLM53_PROMPT_CENSUS" not in line]
        self.assertEqual(reading(score(lines), "eager_census")["verdict"], "NOT_MEASURED")


class Memory(unittest.TestCase):
    def test_pt_devmem_on_any_rank_fails(self) -> None:
        result = score(mock_log() + [f"{who(2)}RuntimeError PT_DEVMEM out of memory"])
        self.assertEqual(reading(result, "pt_devmem")["verdict"], "FAIL")

    def test_free_hbm_under_512_fails(self) -> None:
        result = score(with_rank(3, free=511))
        self.assertEqual((reading(result, "min_free_mib")["verdict"], reading(result, "min_free_mib")["value"]), ("FAIL", 511))

    def test_exactly_512_free_passes(self) -> None:
        self.assertEqual(reading(score(with_rank(0, free=512)), "min_free_mib")["verdict"], "PASS")

    def test_no_ballast_line_is_not_measured(self) -> None:
        self.assertEqual(reading(score(with_rank(1, ballast=None)), "min_free_mib")["verdict"], "NOT_MEASURED")

    def test_a_smaller_ballast_is_not_the_512_reading(self) -> None:
        self.assertEqual(reading(score(with_rank(1, ballast=256, free=2000)), "min_free_mib")["verdict"], "NOT_MEASURED")


class ReplayTimes(unittest.TestCase):
    def test_width_4096_is_read_and_reported_in_ms(self) -> None:
        result = score()
        self.assertEqual(reading(result, "replay_ms_4096")["value"], 3000.0)
        self.assertEqual(reading(result, "replay_ms_4096")["verdict"], "PASS")

    def test_each_width_has_its_own_bar(self) -> None:
        for width, over in ((1024, 2001.0), (2048, 3001.0), (4096, 4001.0), (8192, 6601.0)):
            result = score(with_rank(2, ms={**WIDTHS, width: over}))
            self.assertEqual(reading(result, f"replay_ms_{width}")["verdict"], "FAIL", width)

    def test_at_the_bar_passes(self) -> None:
        self.assertEqual(reading(score(with_rank(0, ms={**WIDTHS, 4096: 4000.0})), "replay_ms_4096")["verdict"], "PASS")

    def test_a_key_that_never_replayed_is_not_measured(self) -> None:
        result = score(with_rank(0, skip_widths=(4096,)))
        self.assertEqual(reading(result, "replay_ms_4096")["verdict"], "NOT_MEASURED")
        self.assertEqual(reading(result, "replay_ms_8192")["verdict"], "PASS")

    def test_all_four_serving_keys_are_read(self) -> None:
        names = [item["name"] for item in score()["readings"] if item["name"].startswith("replay_ms_")]
        self.assertEqual(names, ["replay_ms_8192", "replay_ms_4096", "replay_ms_2048", "replay_ms_1024"])

    def test_the_worst_rank_decides(self) -> None:
        result = score(with_rank(3, ms={**WIDTHS, 8192: 6000.0}))
        self.assertEqual(reading(result, "replay_ms_8192")["value"], 6000.0)


class DecodeSteps(unittest.TestCase):
    def test_the_report_prints_a_rows_by_width_table_with_dashes_for_gaps(self) -> None:
        report = scorer.render_report(score(rows=drive_rows(status={(16, 8192): "unreachable"})))
        table = report.split("(across):")[1].splitlines()[1:]
        self.assertEqual(table[0].split(), ["rows", "W=1024", "W=4096", "W=8192"])
        self.assertEqual(table[1].split()[:2], ["1", "102.024"])
        self.assertEqual(table[5].split(), ["16", "117.024", "120.096", "-"])

    def test_all_fifteen_cells_pass_and_report_their_ms(self) -> None:
        item = reading(score(), "decode_step_ms")
        self.assertEqual(item["verdict"], "PASS")
        self.assertEqual(len(item["value"]), 15)
        self.assertAlmostEqual(item["value"]["1x1024"], 102.024)

    def test_a_cell_the_drive_never_ran_is_not_measured(self) -> None:
        item = reading(score(rows=drive_rows(skip=(16, 8192))), "decode_step_ms")
        self.assertEqual(item["verdict"], "NOT_MEASURED")
        self.assertIn("16x8192", item["detail"])

    def test_an_unreachable_lane_count_is_not_measured_not_a_failure(self) -> None:
        item = reading(score(rows=drive_rows(status={(16, 1024): "unreachable"})), "decode_step_ms")
        self.assertEqual(item["verdict"], "NOT_MEASURED")
        self.assertEqual(len(item["value"]), 14)

    def test_an_errored_cell_fails(self) -> None:
        item = reading(score(rows=drive_rows(status={(4, 4096): "error"})), "decode_step_ms")
        self.assertEqual(item["verdict"], "FAIL")

    def test_no_lane_rows_at_all_is_not_measured(self) -> None:
        rows = [row for row in drive_rows() if row["drive"] != "lanes"]
        self.assertEqual(reading(score(rows=rows), "decode_step_ms")["verdict"], "NOT_MEASURED")


class PaddingProof(unittest.TestCase):
    def test_all_equal_on_every_rank_passes(self) -> None:
        self.assertEqual(reading(score(), "decode_pad_merge")["verdict"], "PASS")

    def test_one_unequal_line_fails(self) -> None:
        item = reading(score(with_rank(1, merge=(1, 0, 1))), "decode_pad_merge")
        self.assertEqual(item["verdict"], "FAIL")
        self.assertIn("rank 1", item["detail"])

    def test_zero_merge_lines_is_not_measured_never_a_pass(self) -> None:
        lines = [line for line in mock_log() if "GLM53_DECODE_MERGE" not in line]
        self.assertEqual(reading(score(lines), "decode_pad_merge")["verdict"], "NOT_MEASURED")

    def test_one_rank_without_lines_is_not_measured(self) -> None:
        item = reading(score(with_rank(2, merge=None)), "decode_pad_merge")
        self.assertEqual(item["verdict"], "NOT_MEASURED")
        self.assertIn("rank 2", item["detail"])

    def test_repeated_outputs_that_differ_fail_even_with_equal_hidden(self) -> None:
        rows = drive_rows()
        rows[1] = {"drive": "merge", "tokens": 200_000, "outputs_equal": False}
        self.assertEqual(reading(score(rows=rows), "decode_pad_merge")["verdict"], "FAIL")


class ColdDecode(unittest.TestCase):
    def test_the_first_two_decode_steps_replaying_passes(self) -> None:
        self.assertEqual(reading(score(), "cold_decode_replay")["verdict"], "PASS")

    def test_a_native_fallback_on_the_second_step_fails(self) -> None:
        lines = [line for rank in RANKS for line in rank_lines(rank, cold=False)]
        lines += cold_lines(0, second_step_native=1) + cold_lines(1) + cold_lines(2) + cold_lines(3)
        item = reading(score(lines), "cold_decode_replay")
        self.assertEqual(item["verdict"], "FAIL")
        self.assertIn("rank 0", item["detail"])

    def test_no_live_decode_step_is_not_measured(self) -> None:
        self.assertEqual(reading(score(with_rank(1, cold=False)), "cold_decode_replay")["verdict"], "NOT_MEASURED")

    def test_a_gap_in_the_counters_lines_is_not_measured(self) -> None:
        lines = [line for line in mock_log() if "'mixed_steps': 4," not in line]
        item = reading(score(lines), "cold_decode_replay")
        self.assertEqual(item["verdict"], "NOT_MEASURED")
        self.assertIn("LOG_EVERY", item["detail"])


class BootTime(unittest.TestCase):
    def test_under_25_minutes_passes(self) -> None:
        self.assertEqual(reading(score(boot_seconds=1500.0), "boot_time_s")["verdict"], "PASS")

    def test_over_25_minutes_fails(self) -> None:
        self.assertEqual(reading(score(boot_seconds=1501.0), "boot_time_s")["verdict"], "FAIL")

    def test_hitting_the_startup_deadline_fails(self) -> None:
        self.assertEqual(reading(score(boot_seconds=None, over_budget=True), "boot_time_s")["verdict"], "FAIL")

    def test_no_boot_seconds_is_not_measured(self) -> None:
        self.assertEqual(reading(score(boot_seconds=None), "boot_time_s")["verdict"], "NOT_MEASURED")


class LongDrive(unittest.TestCase):
    def test_both_sizes_cold_and_warm_pass(self) -> None:
        self.assertEqual(reading(score(), "long_drive")["verdict"], "PASS")

    def test_an_errored_size_fails(self) -> None:
        rows = drive_rows()
        rows[3] = {"drive": "long", "tokens": 530_000, "error": "TimeoutError: x"}
        self.assertEqual(reading(score(rows=rows), "long_drive")["verdict"], "FAIL")

    def test_a_size_that_never_ran_is_not_measured(self) -> None:
        rows = [row for row in drive_rows() if row.get("tokens") != 530_000]
        self.assertEqual(reading(score(rows=rows), "long_drive")["verdict"], "NOT_MEASURED")

    def test_a_warm_turn_that_differs_from_the_cold_one_fails(self) -> None:
        rows = drive_rows()
        rows[2] = {**rows[2], "error": "warm output differs from cold output on the same prompt"}
        self.assertEqual(reading(score(rows=rows), "long_drive")["verdict"], "FAIL")


class Overall(unittest.TestCase):
    def test_a_fail_beats_not_measured(self) -> None:
        lines = with_rank(0, native=1)
        result = score(lines, rows=drive_rows(skip=(1, 1024)))
        self.assertEqual(result["verdict"], "FAIL")

    def test_any_not_measured_reading_stops_a_pass(self) -> None:
        self.assertEqual(score(rows=drive_rows(skip=(1, 1024)))["verdict"], "NOT_MEASURED")

    def test_an_empty_log_and_no_rows_is_not_measured_everywhere(self) -> None:
        result = scorer.score([], [], scorer.BootFacts(None), scorer.Expected.default())
        self.assertEqual({item["verdict"] for item in result["readings"]}, {"NOT_MEASURED"})
        self.assertEqual(result["verdict"], "NOT_MEASURED")

    def test_the_cli_returns_zero_only_for_a_pass(self) -> None:
        with tempfile.TemporaryDirectory() as tmp, redirect_stdout(StringIO()):
            log, rows, out = Path(tmp, "c.log"), Path(tmp, "d.jsonl"), Path(tmp, "r.json")
            log.write_text("\n".join(mock_log()))
            rows.write_text("\n".join(json.dumps(row) for row in drive_rows()))
            args = ["--log", str(log), "--rows", str(rows), "--boot-seconds", "900", "--json", str(out)]
            self.assertEqual(scorer.main(args), 0)
            self.assertEqual(json.loads(out.read_text())["verdict"], "PASS")
            self.assertEqual(scorer.main(["--log", str(log), "--rows", str(rows)]), 1)  # no boot seconds


if __name__ == "__main__":
    unittest.main()
