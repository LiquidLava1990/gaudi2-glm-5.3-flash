"""K1 kill cell (plan step 28): scorer and runner, on CPU with mock logs. The cell itself waits on a card window."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "cells"))

import k1_cell as cell  # noqa: E402
import k1_score as scorer  # noqa: E402

MIB = 1 << 20
RANKS = (0, 1, 2, 3)
BASE_IN_USE_MIB = 1000
ENV_TEXT = """# comment
VLLM_GAUDI_MIXED_PROMPT_KEYS=1024:8192,1024:1024
VLLM_GAUDI_PROMPT_GRAPH_MAX_TOTAL_TOKENS=1

PT_HPUGRAPH_DISABLE_TENSOR_CACHE=1
HPU_GLM53_PREFILL_HBM_CENSUS=1
HPU_GLM53_R589_CAPTURE_MARKS=0
VLLM_GAUDI_MIXED_PREFILL_TOKENS=1024
HPU_GLM53_Q4_SHARED_GATHER_OWNER=1
PT_HPU_LAZY_COLLECTIVES_HOLD_TENSORS=0
HPU_GLM53_K1_BALLAST_MIB=1538
HPU_GLM53_K1_BALLAST_EVERY=1
HPU_GLM53_PROMPT_KEY_MERGE_CHECK=1
"""


def census(rank: int, kind: str, width: int, *, resident: int = 1500, peak: int = 2600, between: int = 1500,
           replay_ms: float | None = 1500.0, sync_true: int | None = 0, reset_failed: bool = False, prefix: bool = True) -> str:
    """One GLM53_PROMPT_CENSUS graph line of a rank; sizes in MiB above a start of BASE_IN_USE_MIB."""
    start = BASE_IN_USE_MIB * MIB
    body = {"path": "graph", "kind": kind, "rows": 1024, "context": width, "pid": 100 + rank, "start_in_use": start,
            "end_in_use": start + resident * MIB, "max_in_use": start + peak * MIB, "limit": 90_000 * MIB,
            "between_peak": start + between * MIB, "peak_reset_failed": reset_failed, "n": 1, "sites": []}
    if kind == "replay" and replay_ms is not None:
        body["replay_ms"] = replay_ms
    if kind == "capture" and sync_true is not None:
        body["sync_true"] = sync_true
    who = f"(Worker_TP{rank}_EP{rank} pid={100 + rank}) " if prefix else ""
    return f"2026-11-07T10:00:00.000Z {who}WARNING GLM53_PROMPT_CENSUS {json.dumps(body, sort_keys=True)}"


def replay_lines(rank: int, width: int, ms: float, count: int, *, t: int, every_second: bool, prefix: bool) -> list[str]:
    """`count` replays of one key, the first deliberately awful. every_second: only replays #2, #4... carry replay_ms, as the census writes."""
    lines = []
    for index in range(count):
        first = index == 0
        timed = index % 2 == 1 if every_second else True
        lines.append(census(rank, "replay", width, between=9000 if first else t, replay_ms=(60_000.0 if first else ms) if timed else None,
                            prefix=prefix))
    return lines


def rank_lines(rank: int, *, r_wide: int = 1500, r_low: int = 1500, t: int = 1500, ms_wide: float = 5000.0,
               ms_low: float = 1500.0, replays: int = 4, wide_replays: int | None = None, merge: int | None = 1,
               merge_starts: tuple[int, ...] = (0, 4096, 5120),
               sync_true: int | None = 0, prefix: bool = True, every_second: bool = False, ballast: int | None = 1538,
               ballast_replay: int = 1, free: int = 2000, owners: int | None = 1) -> list[str]:
    """Wide then low capture, then `replays` replays of each key (`wide_replays` for the wide key). The first replay is awful."""
    lines = [census(rank, "capture", 8192, resident=r_wide, sync_true=sync_true, prefix=prefix),
             census(rank, "capture", 1024, resident=r_low, sync_true=sync_true, prefix=prefix)]
    counts = {8192: replays if wide_replays is None else wide_replays, 1024: replays}
    for width, ms in ((8192, ms_wide), (1024, ms_low)):
        lines += replay_lines(rank, width, ms, counts[width], t=t, every_second=every_second, prefix=prefix)
    if merge is not None:
        lines += [merge_line(rank, start, merge) for start in merge_starts]
    if owners is not None:
        lines.append(f"(Worker_TP{rank}_EP{rank} pid={100 + rank}) GLM53_Q4_SHARED_OWNER shape=(512, 513, 272) bytes=71442432 owners={owners}")
    if ballast is not None:
        lines.append(f"(Worker_TP{rank}_EP{rank} pid={100 + rank}) GLM53_K1_BALLAST total_mib={ballast} replays={ballast_replay} "
                     f"free_bytes={free * MIB}")
    return lines


def merge_line(rank: int, start: int | None, torch_equal: int) -> str:
    """The GLM53_K1_MERGE line glm53_mixed_graph.KeyMerge.observe writes; start=None is the old line without a start."""
    at = "" if start is None else f" start={start}"
    return f"(Worker_TP{rank}_EP{rank} pid={100 + rank}) GLM53_K1_MERGE narrow=1024 wide=8192 rows=1024{at} torch_equal={torch_equal}"


def mock_log(**overrides: object) -> list[str]:
    return [line for rank in RANKS for line in rank_lines(rank, **overrides)]


def score(lines: list[str], boot_seconds: float | None = 600.0, over_budget: bool = False) -> dict:
    return scorer.score(scorer.parse_log(lines), boot_seconds, over_budget)


class Healthy(unittest.TestCase):
    def test_a_clean_four_rank_log_passes(self) -> None:
        result = score(mock_log())
        self.assertEqual((result["verdict"], result["kills"], result["gaps"]), ("PASS", [], []))

    def test_readings_per_rank(self) -> None:
        row = score(mock_log(r_wide=1700, r_low=1300, t=1234))["ranks"][2]
        self.assertEqual((row["wide"]["resident_mib"], row["low"]["resident_mib"], row["transient_mib"], row["fit_mib"]),
                         (1700, 1300, 1234, 4234))
        self.assertEqual((row["wide"]["capture_peak_mib"], row["wide"]["replay_ms"], row["low"]["replay_ms"]), (2600, 5000.0, 1500.0))

    def test_the_first_replay_is_discarded(self) -> None:
        row = score(mock_log())["ranks"][0]
        self.assertEqual((row["wide"]["replays"], row["wide"]["warmed"], row["transient_mib"], row["wide"]["replay_ms"]),
                         (4, 3, 1500, 5000.0))

    def test_ranks_without_a_worker_prefix_fall_back_to_pid_order(self) -> None:
        lines = [line for rank in RANKS for line in rank_lines(rank, prefix=False, merge=None)]
        parsed = scorer.parse_log(lines)
        self.assertEqual(sorted(parsed.ranks), [0, 1, 2, 3])

    def test_a_failed_peak_reset_does_not_count_as_a_transient(self) -> None:
        lines = mock_log()
        lines.append(census(1, "replay", 8192, between=7000, reset_failed=True))
        self.assertEqual(score(lines)["ranks"][1]["transient_mib"], 1500)

    def test_the_report_ends_with_the_verdict_and_prints_every_rank(self) -> None:
        report = scorer.render_report(score(mock_log()))
        self.assertEqual(report.splitlines()[-1], "K1_VERDICT: PASS")
        self.assertEqual(len([line for line in report.splitlines() if line.split() and line.split()[0] in "0123"]), 4)
        self.assertIn("2r+t", report)


def with_rank(rank: int, **overrides: object) -> list[str]:
    """A healthy log except for one rank."""
    return [line for other in RANKS for line in rank_lines(other, **(overrides if other == rank else {}))]


class KillLines(unittest.TestCase):
    def assert_dead(self, result: dict, *needles: str) -> None:
        self.assertEqual(result["verdict"], "DEAD", result)
        text = "\n".join(result["kills"])
        for needle in needles:
            self.assertIn(needle, text)

    def test_any_pt_devmem_kills(self) -> None:
        lines = mock_log() + ["(Worker_TP2_EP2 pid=102) ERROR PT_DEVMEM allocation failed"]
        self.assert_dead(score(lines), "rank 2", "PT_DEVMEM")

    def test_pt_devmem_on_an_unidentified_process_kills(self) -> None:
        self.assert_dead(score(mock_log() + ["PT_DEVMEM out of memory"]), "PT_DEVMEM")

    def test_any_sync_true_in_capture_kills(self) -> None:
        self.assert_dead(score(with_rank(3, sync_true=2)), "rank 3", "sync=True")

    def test_a_boot_past_25_minutes_kills_and_exactly_25_does_not(self) -> None:
        self.assert_dead(score(mock_log(), boot_seconds=1501.0), "boot", "1501")
        self.assertEqual(score(mock_log(), boot_seconds=1500.0)["verdict"], "PASS")

    def test_hitting_the_startup_deadline_kills(self) -> None:
        self.assert_dead(score([], boot_seconds=None, over_budget=True), "startup deadline")

    def test_a_large_2r_plus_t_is_printed_but_decides_nothing(self) -> None:
        result = score(with_rank(1, r_wide=5000, r_low=5000, t=9000))
        self.assertEqual((result["verdict"], result["kills"]), ("PASS", []))
        self.assertEqual(result["ranks"][1]["fit_mib"], 19000)
        self.assertIn("19000", scorer.render_report(result))

    def test_no_owner_line_is_a_cell_without_the_owner(self) -> None:
        result = score(with_rank(2, owners=None))
        self.assertEqual(result["verdict"], "NOT_MEASURED")
        self.assertIn("GLM53_Q4_SHARED_OWNER", "\n".join(result["gaps"]))

    def test_the_report_shows_the_owner_count_of_each_rank(self) -> None:
        self.assertIn("shared_owners=[2, 1, 1, 1]", scorer.render_report(score(with_rank(0, owners=2))))

    def test_a_capacity_refusal_of_the_second_key_kills(self) -> None:
        for reason in ("measured graph residency exceeded budget", "mixed graph budget exhausted before capture", "free HBM 900.0 MiB < required 2847.0 MiB"):
            line = f"(Worker_TP1_EP1 pid=101) ERROR RuntimeError: mixed graph BOOT admission rejected on ALL ranks: rank 0: {reason}"
            with self.subTest(reason):
                self.assert_dead(score(mock_log() + [line]), "both keys")

    def test_a_configuration_refusal_is_not_a_fit_kill(self) -> None:
        line = "(Worker_TP1_EP1 pid=101) ERROR RuntimeError: mixed graph BOOT admission rejected on ALL ranks: rank 0: free memory query failed: bucket exceeds max_model_len. and set BUDGET_MIB >= calibrated estimate"
        self.assertEqual(score(mock_log() + [line])["kills"], [])

    def test_a_ballast_pt_devmem_kills(self) -> None:
        self.assert_dead(score(with_rank(1) + ["(Worker_TP1_EP1 pid=101) PT_DEVMEM"]), "rank 1", "PT_DEVMEM")

    def test_no_ballast_line_is_not_a_fit_pass(self) -> None:
        result = score(with_rank(2, ballast=None))
        self.assertEqual(result["verdict"], "NOT_MEASURED")
        self.assertIn("rank 2", "\n".join(result["gaps"]))

    def test_a_ballast_under_the_decode_budget_plus_margin_is_not_a_fit_pass(self) -> None:
        self.assertEqual(score(with_rank(0, ballast=1537))["verdict"], "NOT_MEASURED")

    def test_a_ballast_taken_after_the_first_replay_is_not_a_fit_pass(self) -> None:
        self.assertEqual(score(with_rank(0, ballast_replay=13))["verdict"], "NOT_MEASURED")

    def test_the_report_shows_the_lowest_free_of_any_rank(self) -> None:
        result = score(with_rank(3, free=700))
        self.assertEqual(result["ballast_min_free_mib"], 700)
        self.assertIn("ballast_min_free_mib=700", scorer.render_report(result))

    def test_a_replay_that_stops_matching_itself_is_not_a_pass(self) -> None:
        line = "(Worker_TP1_EP1 pid=101) GLM53_K1_SELF rows=1024 width=1024 start=0 torch_equal=0"
        result = score(mock_log() + [line])
        self.assertEqual(result["verdict"], "NOT_MEASURED")
        self.assertIn("GLM53_K1_SELF", "\n".join(result["gaps"]))

    def test_wide_replay_above_6600_ms_kills_and_6600_does_not(self) -> None:
        self.assert_dead(score(with_rank(2, ms_wide=6601.0)), "rank 2", "(1,1024,8192)", "6601")
        self.assertEqual(score(with_rank(2, ms_wide=6600.0))["verdict"], "PASS")

    def test_low_replay_above_2000_ms_kills_and_2000_does_not(self) -> None:
        self.assert_dead(score(with_rank(0, ms_low=2001.0)), "rank 0", "(1,1024,1024)", "2001")
        self.assertEqual(score(with_rank(0, ms_low=2000.0))["verdict"], "PASS")

    def test_a_width_merge_mismatch_kills(self) -> None:
        self.assert_dead(score(with_rank(1, merge=0)), "rank 1", "torch.equal")

    def test_a_mismatch_at_any_one_start_kills_and_names_it(self) -> None:
        for bad in (0, 4096, 5120):
            lines = with_rank(2, merge_starts=tuple(start for start in (0, 4096, 5120) if start != bad))
            lines.append(merge_line(2, bad, 0))
            with self.subTest(bad):
                self.assert_dead(score(lines), "rank 2", f"chunk start {bad}")

    def test_a_mismatch_at_a_start_beats_a_missing_late_line(self) -> None:
        lines = with_rank(1, merge_starts=(0,)) + [merge_line(1, 0, 0)]
        self.assertEqual(score(lines)["verdict"], "DEAD")

    def test_sync_true_counts_only_the_two_k1_keys(self) -> None:
        other_shape = census(1, "capture", 4096, sync_true=5)
        self.assertEqual(score(mock_log() + [other_shape])["verdict"], "PASS")
        self.assertEqual(scorer.summarize_rank(1, scorer.parse_log(mock_log() + [other_shape]).ranks[1]).sync_true, 0)
        self.assert_dead(score(with_rank(1, sync_true=1) + [other_shape]), "rank 1", "2 sync=True")

    def test_the_worst_rank_decides(self) -> None:
        result = score(with_rank(3, ms_low=2500.0))
        self.assertEqual((result["verdict"], len(result["kills"])), ("DEAD", 1))

    def test_a_kill_beats_vacuous(self) -> None:
        lines = [line for line in mock_log() if "Worker_TP0" not in line] + ["(Worker_TP1_EP1 pid=101) PT_DEVMEM"]
        self.assertEqual(score(lines)["verdict"], "DEAD")

    def test_the_first_replays_awful_numbers_never_kill(self) -> None:
        self.assertEqual(score(mock_log(replays=4))["verdict"], "PASS")

    def test_a_median_of_thin_samples_never_kills(self) -> None:
        result = score(mock_log(wide_replays=3, ms_wide=9000.0))
        self.assertEqual((result["verdict"], result["kills"]), ("NOT_MEASURED", []))


class MountAndEnv(unittest.TestCase):
    def test_the_mount_reaches_lab_up(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            args = make_args("B", Path(tmp), "--mount", "/a:/k1trace")
            up = cell.cells.boot_steps(cell.k1_boot(args), args)[0].argv
        self.assertEqual(up[up.index("--mount") + 1], "/a:/k1trace")

    def test_a_stall_dir_without_a_mount_refuses(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            args = make_args("B", Path(tmp))
            args.env_file.write_text(ENV_TEXT + "HPU_GLM53_K1_STALL_DIR=/k1trace\n")
            with self.assertRaisesRegex(SystemExit, "/k1trace"):
                cell.k1_boot(args)

    def test_the_env_must_carry_the_owner_hold_and_ballast(self) -> None:
        for line in ("HPU_GLM53_Q4_SHARED_GATHER_OWNER=1", "PT_HPU_LAZY_COLLECTIVES_HOLD_TENSORS=0", "HPU_GLM53_K1_BALLAST_MIB=1538",
                     "HPU_GLM53_K1_BALLAST_EVERY=1", "HPU_GLM53_PROMPT_KEY_MERGE_CHECK=1"):
            with self.subTest(line), tempfile.TemporaryDirectory() as tmp:
                path = Path(tmp) / "k.env"
                path.write_text(ENV_TEXT.replace(line + "\n", ""))
                with self.assertRaisesRegex(SystemExit, "must set"):
                    cell.load_env(path)


class Vacuous(unittest.TestCase):
    def test_no_replay_at_all_is_vacuous_not_pass(self) -> None:
        self.assertEqual(score(mock_log(replays=0))["verdict"], "VACUOUS")

    def test_one_rank_with_zero_replays_is_vacuous(self) -> None:
        lines = [line for rank in RANKS for line in rank_lines(rank, **({"replays": 0} if rank == 2 else {}))]
        self.assertEqual(score(lines)["verdict"], "VACUOUS")

    def test_a_rank_missing_from_the_log_is_vacuous(self) -> None:
        lines = [line for line in mock_log() if "Worker_TP3" not in line]
        result = score(lines)
        self.assertEqual(result["verdict"], "VACUOUS")
        self.assertEqual(result["ranks"][3]["wide"]["replays"], 0)

    def test_an_empty_log_is_vacuous(self) -> None:
        self.assertEqual(score([])["verdict"], "VACUOUS")

    def test_a_single_replay_per_key_is_not_vacuous_but_has_no_warmed_reading(self) -> None:
        result = score(mock_log(replays=1))
        self.assertEqual(result["verdict"], "NOT_MEASURED")
        self.assertIn("0 warmed", "\n".join(result["gaps"]))


class NotMeasured(unittest.TestCase):
    def test_two_warmed_samples_at_a_width_are_not_measured_and_three_are(self) -> None:
        for kwargs, width in (({"wide_replays": 3}, "(1,1024,8192)"), ({"replays": 3, "wide_replays": 4}, "(1,1024,1024)")):
            with self.subTest(width):
                result = score(mock_log(**kwargs))
                gaps = "\n".join(result["gaps"])
                self.assertEqual(result["verdict"], "NOT_MEASURED")
                self.assertIn(f"{width}: no replay_ms", gaps)
                self.assertIn("2 warmed, 2 timed; a median needs 3 timed", gaps)
        self.assertEqual(score(mock_log(wide_replays=4, replays=4))["verdict"], "PASS")

    def test_every_rank_counts_on_its_own(self) -> None:
        lines = with_rank(3, wide_replays=3)
        result = score(lines)
        self.assertEqual(result["verdict"], "NOT_MEASURED")
        self.assertEqual({gap.split(":")[0] for gap in result["gaps"]}, {"rank 3 (1,1024,8192)"})

    def test_only_timed_replays_are_samples_when_the_census_times_every_second_one(self) -> None:
        """Census writes replay_ms on replays #2, #4...: 6 replays give 3 timed samples, 5 give 2."""
        passing = score(mock_log(replays=6, every_second=True))
        thin = score(mock_log(replays=5, every_second=True))
        self.assertEqual((passing["verdict"], passing["ranks"][0]["wide"]["samples"]), ("PASS", 3))
        self.assertEqual((thin["verdict"], thin["ranks"][0]["wide"]["samples"]), ("NOT_MEASURED", 2))

    def test_the_report_shows_the_timed_samples(self) -> None:
        self.assertIn("4/4(3/3)", scorer.render_report(score(mock_log())))

    def test_a_replay_with_no_timing_field_is_not_a_pass(self) -> None:
        lines = [line.replace('"replay_ms": 5000.0,', "") for line in mock_log()]
        result = score(lines)
        self.assertEqual(result["verdict"], "NOT_MEASURED")
        self.assertIn("replay_ms", "\n".join(result["gaps"]))

    def test_captures_that_never_report_sync_true_are_not_a_pass(self) -> None:
        result = score(mock_log(sync_true=None))
        self.assertEqual(result["verdict"], "NOT_MEASURED")
        self.assertIn("sync_true", "\n".join(result["gaps"]))

    def test_a_rank_without_a_merge_line_from_start_4096_or_later_is_not_a_pass(self) -> None:
        for starts in ((0,), (0, 1024, 2048, 3072), ()):
            with self.subTest(starts):
                result = score(with_rank(3, merge_starts=starts))
                self.assertEqual(result["verdict"], "NOT_MEASURED")
                self.assertIn("rank 3: no GLM53_K1_MERGE line", "\n".join(result["gaps"]))
                if starts:
                    self.assertIn("with start>=4096", "\n".join(result["gaps"]))

    def test_one_late_start_is_enough_and_the_line_without_a_start_is_not_late(self) -> None:
        self.assertEqual(score(with_rank(0, merge_starts=(0, 4096)))["verdict"], "PASS")
        self.assertEqual(score(with_rank(0, merge_starts=(5120,)))["verdict"], "PASS")
        old_format = with_rank(0, merge=None) + [merge_line(0, None, 1)]
        self.assertEqual(score(old_format)["verdict"], "NOT_MEASURED")

    def test_every_rank_needs_its_own_late_merge_line(self) -> None:
        result = score(with_rank(1, merge_starts=(0,)))
        self.assertEqual({gap.split(":")[0] for gap in result["gaps"]}, {"rank 1"})

    def test_no_merge_line_is_not_a_pass(self) -> None:
        result = score(mock_log(merge=None))
        self.assertEqual(result["verdict"], "NOT_MEASURED")
        self.assertIn("GLM53_K1_MERGE", "\n".join(result["gaps"]))

    def test_no_boot_time_is_not_a_pass(self) -> None:
        self.assertEqual(score(mock_log(), boot_seconds=None)["verdict"], "NOT_MEASURED")

    def test_a_garbled_census_line_is_not_a_pass(self) -> None:
        result = score(mock_log() + ["(Worker_TP0_EP0 pid=100) GLM53_PROMPT_CENSUS {not json"])
        self.assertEqual(result["verdict"], "NOT_MEASURED")
        self.assertIn("garbled", "\n".join(result["gaps"]))


class ScorerCommand(unittest.TestCase):
    def run_main(self, lines: list[str], *extra: str) -> tuple[int, str]:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "k1.log"
            path.write_text("\n".join(lines) + "\n")
            out = StringIO()
            with redirect_stdout(out):
                code = scorer.main(["--log", str(path), *extra])
        return code, out.getvalue()

    def test_pass_exits_zero(self) -> None:
        code, out = self.run_main(mock_log(), "--boot-seconds", "700")
        self.assertEqual((code, out.splitlines()[-1]), (0, "K1_VERDICT: PASS"))

    def test_dead_exits_one(self) -> None:
        code, out = self.run_main(mock_log(), "--boot-seconds", "1600")
        self.assertEqual((code, out.splitlines()[-1]), (1, "K1_VERDICT: DEAD"))

    def test_over_budget_flag_kills(self) -> None:
        self.assertEqual(self.run_main(mock_log(), "--boot-over-budget")[1].splitlines()[-1], "K1_VERDICT: DEAD")


def make_args(arm: str, tmp: Path, *extra: str) -> object:
    env_file = tmp / "graphs-k1.env"
    env_file.write_text(ENV_TEXT)
    return cell.parse_args(["run", "--arm", arm, "--image", "claude-flash-dual-graphs:20261106", "--port", "18440",
                            "--out-dir", str(tmp / "out"), "--env-file", str(env_file), "--stamp", "T", *extra])


class EnvFile(unittest.TestCase):
    def test_widest_key_first_and_the_required_settings(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            env = cell.load_env(Path(make_args("B", Path(tmp)).env_file))
        keys = env["VLLM_GAUDI_MIXED_PROMPT_KEYS"].split(",")
        self.assertEqual(keys, ["1024:8192", "1024:1024"])
        self.assertEqual((env["VLLM_GAUDI_PROMPT_GRAPH_MAX_TOTAL_TOKENS"], env["PT_HPUGRAPH_DISABLE_TENSOR_CACHE"],
                          env["HPU_GLM53_PREFILL_HBM_CENSUS"]), ("1", "1", "1"))

    def test_the_arm_switch_is_on_for_b_and_off_for_a(self) -> None:
        env = {"HPU_GLM53_R589_CAPTURE_MARKS": "0", "X": "1"}
        self.assertEqual(cell.arm_env("B", env), {"HPU_GLM53_R589_CAPTURE_MARKS": "1", "X": "1"})
        self.assertEqual(cell.arm_env("A", env)["HPU_GLM53_R589_CAPTURE_MARKS"], "0")

    def test_a_file_without_the_1024_row_chunk_budget_refuses(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bad.env"
            path.write_text(ENV_TEXT.replace("VLLM_GAUDI_MIXED_PREFILL_TOKENS=1024", "VLLM_GAUDI_MIXED_PREFILL_TOKENS=512"))
            with self.assertRaisesRegex(SystemExit, "VLLM_GAUDI_MIXED_PREFILL_TOKENS"):
                cell.load_env(path)
            path.write_text(ENV_TEXT.replace("VLLM_GAUDI_MIXED_PREFILL_TOKENS=1024" + chr(10), ""))
            with self.assertRaisesRegex(SystemExit, "must set"):
                cell.load_env(path)

    def test_a_file_that_lost_a_k1_setting_refuses(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bad.env"
            path.write_text(ENV_TEXT.replace("1024:8192,1024:1024", "1024:1024,1024:8192"))
            with self.assertRaisesRegex(SystemExit, "must set"):
                cell.load_env(path)

    def test_a_malformed_line_refuses_with_its_number(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bad.env"
            path.write_text("OK=1\nnot a pair\n")
            with self.assertRaisesRegex(SystemExit, "bad.env:2"):
                cell.load_env(path)

    @unittest.skipUnless(cell.DEFAULT_ENV_FILE.is_file(), "graphs-k1.env is not mounted next to overlay/ here")
    def test_the_shipped_env_file_loads_with_the_arm_switch_off(self) -> None:
        env = cell.load_env(cell.DEFAULT_ENV_FILE)
        self.assertEqual(env["HPU_GLM53_R589_CAPTURE_MARKS"], "0")
        self.assertEqual(env["VLLM_GAUDI_MIXED_PREFILL_TOKENS"], "1024")


class DryRunDefault(unittest.TestCase):
    def test_no_flag_prints_the_plan_and_runs_nothing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp, mock.patch("subprocess.run", side_effect=AssertionError("touched")) as run:
            out = StringIO()
            with redirect_stdout(out):
                code = cell.main(["run", "--arm", "B", "--image", "x", "--port", "18440", "--out-dir", f"{tmp}/o",
                                  "--env-file", str(make_args("B", Path(tmp)).env_file)])
            plan = json.loads(out.getvalue())
        self.assertEqual((code, run.call_count, plan["dry_run"], plan["touches_cards"]), (0, 0, True, False))
        self.assertEqual([step["label"] for step in plan["steps"]], ["up", "drive", "down"])
        self.assertIn("HPU_GLM53_R589_CAPTURE_MARKS=1", plan["steps"][0]["argv"])
        self.assertFalse(Path(tmp, "o").exists())

    def test_the_up_step_goes_through_lab_py_on_quad_a_with_the_boot_deadline(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            args = make_args("A", Path(tmp))
            up = cell.cells.boot_steps(cell.k1_boot(args), args)[0].argv
        self.assertEqual((up[2], up[up.index("--quad") + 1], up[up.index("--deadline") + 1]), ("up", "A", "1500"))
        self.assertIn("HPU_GLM53_R589_CAPTURE_MARKS=0", up)
        self.assertEqual(up[up.index("--name") + 1], "claude-flash-launch-k1-armA")

    def test_dry_run_and_execute_exclude_each_other(self) -> None:
        with tempfile.TemporaryDirectory() as tmp, redirect_stderr(StringIO()), self.assertRaises(SystemExit):
            make_args("B", Path(tmp), "--dry-run", "--execute")

    def test_a_replica_port_is_refused(self) -> None:
        with self.assertRaisesRegex(SystemExit, "production replica port"):
            cell.parse_args(["run", "--arm", "B", "--image", "x", "--port", "18274", "--out-dir", "o"])

    def test_the_prompt_is_whole_1024_row_chunks_and_reaches_a_chunk_at_4096_or_later(self) -> None:
        starts = cell.chunk_starts(cell.PROMPT_TOKENS)
        self.assertEqual(cell.PROMPT_TOKENS % cell.CHUNK_ROWS, 0)
        self.assertEqual(starts, [0, 1024, 2048, 3072, 4096, 5120])
        self.assertEqual([start for start in starts if start >= cell.LATE_CHUNK_START], [4096, 5120])
        self.assertEqual(cell.CHUNK_ROWS, int(cell.REQUIRED_ENV["VLLM_GAUDI_MIXED_PREFILL_TOKENS"]))

    def test_the_default_repeats_give_the_target_timed_warmed_wide_replays(self) -> None:
        """A pair of repeats forces the 3 merge chunks (starts 0, 4096, 5120) wide; the census times every second wide replay."""
        self.assertEqual(cell.MERGE_CHUNKS_PER_REPEAT, 3)
        for repeats in (cell.MIN_REPEATS, cell.REPEATS):
            wide_replays = repeats // cell.MERGE_STRIDE * cell.MERGE_CHUNKS_PER_REPEAT
            self.assertGreaterEqual(wide_replays // cell.TIMED_REPLAY_STRIDE, cell.TARGET_WARMED_SAMPLES)
            self.assertEqual(repeats % cell.MERGE_STRIDE, 0)
        self.assertGreaterEqual(cell.TARGET_WARMED_SAMPLES, scorer.MIN_WARMED_SAMPLES)
        self.assertEqual((cell.MIN_REPEATS, cell.REPEATS), (8, 12))
        below = (cell.MIN_REPEATS - cell.MERGE_STRIDE) // cell.MERGE_STRIDE * cell.MERGE_CHUNKS_PER_REPEAT
        self.assertLess(below // cell.TIMED_REPLAY_STRIDE, cell.TARGET_WARMED_SAMPLES)

    def test_repeats_below_the_minimum_are_refused_and_the_minimum_is_accepted(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(make_args("B", Path(tmp), "--repeats", str(cell.MIN_REPEATS)).repeats, cell.MIN_REPEATS)
            self.assertEqual(make_args("B", Path(tmp)).repeats, cell.REPEATS)
            with redirect_stderr(StringIO()) as err, self.assertRaises(SystemExit):
                make_args("B", Path(tmp), "--repeats", str(cell.MIN_REPEATS - 1))
        self.assertIn("timed warmed W=8192 replays", err.getvalue())

    def test_the_drive_step_is_bounded_whatever_the_repeats(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            for repeats in (cell.MIN_REPEATS, cell.REPEATS, 500):
                args = make_args("B", Path(tmp), "--repeats", str(repeats))
                drive = cell.cells.boot_steps(cell.k1_boot(args), args)[1]
                self.assertEqual((drive.label, drive.timeout_s <= cell.DRIVE_BUDGET_S + 60), ("drive", True))
                self.assertEqual(drive.argv[drive.argv.index("--repeats") + 1], str(repeats))

    def test_the_fixed_prompt_is_the_same_every_time(self) -> None:
        self.assertEqual(cell.fixed_prompt(64), cell.fixed_prompt(64))
        self.assertEqual(len(cell.fixed_prompt(cell.PROMPT_TOKENS)), 6144)


class Execute(unittest.TestCase):
    """--execute with subprocess.run replaced: the order of steps, the always-down rule and how a failed boot scores."""

    def run_cell(self, tmp: Path, *, up_code: int = 0, up_stderr: str = "", drive_code: object = 0,
                 log: list[str] | None = None) -> tuple[int, list[str], dict]:
        args = make_args("B", tmp, "--execute")
        name = "claude-flash-launch-k1-armB"
        calls: list[str] = []

        def fake_run(argv: list[str], **kwargs: object) -> mock.Mock:
            calls.append(argv[2])
            if argv[2] == "up" and up_code == 0:
                (tmp / f"state-{name}.json").write_text(json.dumps({"startup_ready_s": 640.0}))
            if argv[2] == "down":
                (tmp / f"{name}.log").write_text("\n".join(log or []) + "\n")
            if argv[2] == "drive" and drive_code == "timeout":
                raise cell.subprocess.TimeoutExpired(argv, 1)
            code = {"up": up_code, "drive": drive_code}.get(argv[2], 0)
            return mock.Mock(returncode=code, stderr=up_stderr if argv[2] == "up" else None)

        with mock.patch("subprocess.run", side_effect=fake_run), mock.patch.object(cell, "LANE", tmp),                 redirect_stdout(StringIO()), redirect_stderr(StringIO()):
            code = cell.main(["run", *args_list(args)])
        return code, calls, json.loads((tmp / "out" / "k1-armB-T.json").read_text())

    def test_up_drive_down_then_the_log_is_scored(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            code, calls, result = self.run_cell(Path(tmp), log=mock_log())
        self.assertEqual((code, calls, result["verdict"], result["boot_seconds"]), (0, ["up", "drive", "down"], "PASS", 640.0))
        self.assertEqual([step["label"] for step in result["steps"]], ["up", "drive", "down"])

    def test_down_still_runs_when_the_requests_time_out(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            code, calls, result = self.run_cell(Path(tmp), drive_code="timeout", log=mock_log())
        self.assertEqual((calls, result["steps"][1]["returncode"]), (["up", "drive", "down"], "timeout"))
        self.assertEqual(result["verdict"], "NOT_MEASURED")

    def test_a_drive_that_exits_nonzero_after_a_healthy_looking_log_is_not_a_pass(self) -> None:
        """A worker crash after the first chunks leaves a log that scores PASS; the drive step's code must veto it."""
        for drive_code in (1, 137, "timeout"):
            with self.subTest(drive_code), tempfile.TemporaryDirectory() as tmp:
                code, calls, result = self.run_cell(Path(tmp), drive_code=drive_code, log=mock_log())
                self.assertEqual((code, calls, result["verdict"]), (1, ["up", "drive", "down"], "NOT_MEASURED"))
                self.assertIn("drive step returned", result["gaps"][0])

    def test_a_drive_failure_does_not_hide_a_kill(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            code, _, result = self.run_cell(Path(tmp), drive_code=1, log=mock_log(ms_wide=7000.0))
        self.assertEqual((code, result["verdict"]), (1, "DEAD"))

    def test_a_clean_drive_still_passes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(self.run_cell(Path(tmp), drive_code=0, log=mock_log())[2]["verdict"], "PASS")

    def test_a_deadline_refusal_from_lab_py_is_a_boot_budget_kill(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            code, calls, result = self.run_cell(Path(tmp), up_code=1, up_stderr="startup deadline 1500s exceeded; docker logs x")
        self.assertEqual((code, calls, result["verdict"]), (1, ["up", "down"], "DEAD"))
        self.assertIn("startup deadline", " ".join(result["kills"]))

    def test_any_other_boot_failure_is_not_measured_not_vacuous(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            code, calls, result = self.run_cell(Path(tmp), up_code=1, up_stderr="quad A lock is not held")
        self.assertEqual((code, calls, result["verdict"]), (1, ["up", "down"], "NOT_MEASURED"))
        self.assertIn("boot failed before ready", result["gaps"][0])


def args_list(args: object) -> list[str]:
    """The argv that makes `args` again, for driving cell.main."""
    return ["--arm", args.arm, "--image", args.image, "--port", str(args.port), "--out-dir", str(args.out_dir),
            "--env-file", str(args.env_file), "--stamp", args.stamp, "--execute"]


class LongContextStepTests(unittest.TestCase):
    """Real prompts past the 1024-block key, in the same boot and under the same ballast as the fit readings."""

    @staticmethod
    def _args(*extra: str) -> list[str]:
        return ["run", "--arm", "B", "--image", "leaf", "--port", "18440", "--out-dir", "/tmp/out", *extra]

    def test_no_long_step_unless_asked(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            boot = cell.k1_boot(make_args("B", Path(tmp)))
        self.assertEqual([step.label for step in boot.actions], ["drive"])

    def test_the_long_step_runs_after_the_fit_readings_in_the_same_boot(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            boot = cell.k1_boot(make_args("B", Path(tmp), "--long-tokens", "150000,1048575"))
        self.assertEqual([step.label for step in boot.actions], ["drive", "long"])
        self.assertIn("150000,1048575", boot.actions[1].argv)

    def test_a_prompt_over_the_declared_window_is_refused(self) -> None:
        with self.assertRaises(SystemExit), redirect_stderr(StringIO()):
            cell.parse_args(self._args("--long-tokens", "1048576"))

    def test_the_budget_covers_every_chunk_above_the_wide_limit(self) -> None:
        self.assertGreaterEqual(cell.long_timeout_s(1048575), 1024 * 6.6)
        self.assertGreaterEqual(cell.long_timeout_s(1024), cell.REQUEST_TIMEOUT_S)

    def test_a_failed_fit_drive_skips_the_long_step_and_still_goes_down(self) -> None:
        seen: list[str] = []

        def fake(step: object, *, capture_stderr: bool = False) -> dict[str, object]:
            seen.append(step.label)
            return {"label": step.label, "returncode": 1 if step.label == "drive" else 0, "stderr": ""}

        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(cell, "run_step", fake):
            args = make_args("B", Path(tmp), "--long-tokens", "150000")
            cell.run_k1_boot(cell.k1_boot(args), args)
        self.assertEqual(seen, ["up", "drive", "down"])

    def test_a_long_failure_is_named_and_a_clean_one_is_not(self) -> None:
        self.assertIsNone(cell.long_failure([{"label": "long", "returncode": 0, "stderr": ""}]))
        self.assertIsNone(cell.long_failure([{"label": "drive", "returncode": 0, "stderr": ""}]))
        self.assertIn("timeout", cell.long_failure([{"label": "long", "returncode": "timeout", "stderr": ""}]))

    def test_each_size_is_written_as_it_finishes_and_the_first_failure_stops(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            out = Path(folder) / "long.jsonl"
            args = cell.parse_args(["drive-long", "--port", "18440", "--tokens-list", "150000,400000,1048575", "--out", str(out)])
            finished = {"ttft_s": 200.0, "chunks": [1.0], "end": 1.0, "output_sha256": "same"}
            replies = [finished, finished, RuntimeError("worker died")]
            with mock.patch.object(cell, "_stream_one", side_effect=replies):
                self.assertEqual(cell.drive_long(args), 1)
            rows = [json.loads(line) for line in out.read_text().splitlines()]
        self.assertEqual([row["tokens"] for row in rows], [150000, 400000])
        self.assertEqual(rows[0]["tokens_per_s"], 750.0)
        self.assertIn("worker died", rows[1]["error"])


class LongContextDecodeTests(unittest.TestCase):
    """Decode at long context: a cold prompt, then the same prompt again as a warm turn."""

    @staticmethod
    def _reply(ttft: float, count: int = 5, gap: float = 0.5) -> dict[str, object]:
        chunks = [100.0 + index * gap for index in range(count)]
        return {"ttft_s": ttft, "chunks": chunks, "end": chunks[-1], "output_sha256": "same"}

    def _drive(self, replies: list[object], out: Path) -> tuple[int, list[dict], mock.MagicMock]:
        args = cell.parse_args(["drive-long", "--port", "18440", "--tokens-list", "150000", "--output-tokens", "5", "--out", str(out)])
        with mock.patch.object(cell, "_stream_one", side_effect=replies) as stream, redirect_stdout(StringIO()):
            code = cell.drive_long(args)
        return code, [json.loads(line) for line in out.read_text().splitlines()], stream

    def test_a_row_holds_cold_and_warm_speed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            code, rows, _ = self._drive([self._reply(230.0), self._reply(4.0, gap=0.25)], Path(tmp) / "long.jsonl")
        self.assertEqual(code, 0)
        self.assertEqual(rows[0]["ttft_s"], 230.0)
        self.assertEqual(rows[0]["decode_tok_per_s"], 2.0)
        self.assertEqual(rows[0]["warm_ttft_s"], 4.0)
        self.assertEqual(rows[0]["warm_decode_tok_per_s"], 4.0)

    def test_the_warm_turn_reuses_the_cold_cache_salt(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            _, _, stream = self._drive([self._reply(230.0), self._reply(4.0)], Path(tmp) / "long.jsonl")
        self.assertEqual(stream.call_args_list[0].args[3], stream.call_args_list[1].args[3])
        self.assertEqual(stream.call_args_list[0].args[2], 5)

    def test_a_warm_turn_that_differs_from_the_cold_output_is_recorded_and_fails(self) -> None:
        other = {**self._reply(4.0), "output_sha256": "different"}
        with tempfile.TemporaryDirectory() as tmp:
            code, rows, _ = self._drive([self._reply(230.0), other], Path(tmp) / "long.jsonl")
        self.assertEqual(code, 1)
        self.assertIn("warm output differs", rows[0]["error"])

    def test_one_output_token_has_no_decode_speed(self) -> None:
        self.assertIsNone(cell.decode_rate(self._reply(1.0, count=1)))

    def test_the_step_budget_covers_cold_warm_and_decode(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            args = make_args("B", Path(tmp), "--long-tokens", "150000", "--long-output-tokens", "64")
            step = cell.k1_boot(args).actions[1]
        self.assertIn("--output-tokens", step.argv)
        self.assertGreaterEqual(step.timeout_s, cell.long_timeout_s(150000) + 2 * 64)


if __name__ == "__main__":
    unittest.main()
