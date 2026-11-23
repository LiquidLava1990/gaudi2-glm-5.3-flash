"""C1 cell (plan section 12.4): env file, boot plan, request drives and the run, on CPU with a fake stream. Nothing here touches a card."""

from __future__ import annotations

import itertools
import json
import subprocess
import sys
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "cells"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import c1_cell as cell  # noqa: E402
import c1_score as scorer  # noqa: E402
import k1_cell  # noqa: E402
import memory_cells  # noqa: E402
from test_c1_score import drive_rows, mock_log  # noqa: E402
from test_replay_trace_report import COMPLETE_OPS, trace, write_run  # noqa: E402

LANE_ENV = Path(__file__).resolve().parents[2] / "graphs-c1.env"
NAME = "claude-flash-launch-c1"
ENV_TEXT = "\n".join(f"{key}={value}" for key, value in {**cell.REQUIRED_ENV, "VLLM_GAUDI_MIXED_PROMPT_KEYS": cell.SERVING_KEYS}.items()) + "\n# comment\n\nHPU_GLM53_K1_STALL_DIR=/k1trace\n"


def write_env(tmp: Path, text: str = ENV_TEXT, name: str = "graphs-c1.env") -> Path:
    path = tmp / name
    path.write_text(text)
    return path


TRACE_ENV = ("HPU_GLM53_PREFILL_HBM_CENSUS=1\nHPU_GLM53_REPLAY_TRACE_DIR=/k1trace\n"
             "HPU_GLM53_REPLAY_TRACE=prompt:1024x8192@3,decode:16x8192@20\n")
TRACE_RUNS = ["prompt-1024x8192-n3", "decode-16x8192-n20"]


def traced_args(tmp: Path, *extra: str, text: str = ENV_TEXT + TRACE_ENV, mount: bool = True) -> list[str]:
    """The `run` argv of the p variant; `text` is its env file."""
    env = write_env(tmp, text, "graphs-c1p.env")
    mounts = ["--mount", f"{tmp}/trace:/k1trace"] if mount else []
    return ["run", "--image", "leaf", "--port", "18440", "--out-dir", str(tmp / "out"), "--variant", "c1p", "--env-file", str(env), *mounts,
            "--stamp", "T", *extra]


def run_args(tmp: Path, *extra: str) -> list[str]:
    return ["run", "--image", "leaf", "--port", "18440", "--out-dir", str(tmp / "out"), "--env-file", str(write_env(tmp)),
            "--mount", f"{tmp}/trace:/k1trace", "--stamp", "T", *extra]


def reply(start: float = 0.0, count: int = 60, gap: float = 0.1, ttft: float = 1.0, sha: str = "a") -> dict:
    chunks = [start + index * gap for index in range(count)]
    return {"ttft_s": ttft, "chunks": chunks, "end": chunks[-1], "output_sha256": sha}


def namespace(**overrides: object) -> mock.Mock:
    values = {"port": 18440, "salt": "S", "long_tokens": (150_000, 530_000), "long_output_tokens": 64, "merge_tokens": 200_000,
              "anchors": {1024: 2048, 4096: 262_400, 8192: 530_000}}
    return mock.Mock(**{**values, **overrides})


class EnvFile(unittest.TestCase):
    def test_a_file_with_every_required_setting_loads(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            env = cell.load_env(write_env(Path(tmp)))
        self.assertEqual(env["VLLM_GAUDI_MIXED_PROMPT_KEYS"], "1024:8192,1024:4096,1024:2048,1024:1024")

    def test_a_file_that_lost_a_setting_refuses_and_names_it(self) -> None:
        text = ENV_TEXT.replace("HPU_GLM53_DECODE_WIDTH_PAD=check", "HPU_GLM53_DECODE_WIDTH_PAD=0")
        with tempfile.TemporaryDirectory() as tmp, self.assertRaises(SystemExit) as caught:
            cell.load_env(write_env(Path(tmp), text))
        self.assertIn("HPU_GLM53_DECODE_WIDTH_PAD", str(caught.exception))

    def test_a_malformed_line_refuses_with_its_number(self) -> None:
        with tempfile.TemporaryDirectory() as tmp, self.assertRaises(SystemExit) as caught:
            cell.load_env(write_env(Path(tmp), "NO_EQUALS_SIGN\n"))
        self.assertIn(":1:", str(caught.exception))

    @unittest.skipUnless(LANE_ENV.is_file(), "graphs-c1.env sits beside the overlay on the host, not in the test container")
    def test_the_shipped_env_file_loads_and_keeps_the_k1_settings(self) -> None:
        env = cell.load_env(LANE_ENV)
        for key, value in {"HPU_GLM53_Q4_SHARED_GATHER_OWNER": "1", "VLLM_PREFILL_CONTEXT_CELL_BUDGET": "0",
                           "PT_HPU_LAZY_COLLECTIVES_HOLD_TENSORS": "0", "VLLM_GAUDI_MIXED_GRAPH_BUDGET_MIB": "8452",
                           "HPU_GLM53_K1_BALLAST_MIB": "512", "HPU_GLM53_PROMPT_KEY_MERGE_CHECK": "0",
                           "VLLM_GAUDI_BOOTSTRAP_DECODE_BUCKETS": "1,1,128;2,1,256;4,1,512;8,1,1024;16,1,2048",
                           "VLLM_GAUDI_TARGETED_SAMPLER_WARMUP_BATCHES": "1,2,4,8,16", "VLLM_GAUDI_MIXED_DECODE_WIDTHS": "1024,4096,8192",
                           "HPU_GLM53_DECODE_PAD_CHECK_WIDTHS": "2048", "HPU_GLM53_GRAPHS_ONLY": "1", "HPU_GLM53_PRIVATE_PROMPT_TOPK": "1", "HPU_GRAPH_DRY_RUN": "0"}.items():
            self.assertEqual(env.get(key), value, key)


    @unittest.skipUnless(LANE_ENV.is_file(), "graphs-c1.env sits beside the overlay on the host, not in the test container")
    def test_the_variant_files_differ_from_c1_only_where_the_plan_says(self) -> None:
        def settings(variant: str) -> dict[str, str]:
            return cell.load_env(LANE_ENV.with_name(f"graphs-{variant}.env"), variant)
        base, traced, wide = settings("c1"), settings("c1p"), settings("c1r")
        self.assertEqual({key: value for key, value in traced.items() if base.get(key) != value},
                         {"HPU_GLM53_REPLAY_TRACE_DIR": "/k1trace", "HPU_GLM53_REPLAY_TRACE": "prompt:1024x8192@3,decode:16x8192@20"})
        changed = {key for key in wide if base.get(key) != wide[key]}
        self.assertEqual(changed, {"VLLM_GAUDI_MIXED_PROMPT_KEYS", "VLLM_ADAPTIVE_LONG_PREFILL_SINGLE_TOKENS", "VLLM_GAUDI_MIXED_PREFILL_TOKENS"})
        self.assertTrue(wide["VLLM_GAUDI_MIXED_PROMPT_KEYS"].startswith("2048:8192,1024:8192,"))

    def test_an_env_file_of_another_variant_refuses(self) -> None:
        wide = ENV_TEXT.replace(cell.SERVING_KEYS, cell.VARIANT_KEYS["c1r"])
        with tempfile.TemporaryDirectory() as tmp, self.assertRaises(SystemExit) as caught:
            cell.load_env(write_env(Path(tmp), wide), "c1")
        self.assertIn("graphs-c1.env", str(caught.exception))
        with tempfile.TemporaryDirectory() as tmp:
            self.assertIn("2048:8192", cell.load_env(write_env(Path(tmp), wide), "c1r")["VLLM_GAUDI_MIXED_PROMPT_KEYS"])


class Plan(unittest.TestCase):
    def test_a_variant_names_its_own_boot_and_drive_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            env = write_env(Path(tmp), ENV_TEXT.replace(cell.SERVING_KEYS, cell.VARIANT_KEYS["c1r"]), "graphs-c1r.env")
            args = cell.parse_args(["run", "--image", "leaf", "--port", "18440", "--out-dir", f"{tmp}/o", "--variant", "c1r", "--env-file", str(env),
                                    "--mount", f"{tmp}/trace:/k1trace"])
            boot = cell.c1_boot(args)
        self.assertEqual((boot.name, args.name), ("claude-flash-launch-c1r", "claude-flash-launch-c1r"))
        self.assertTrue(all("claude-flash-launch-c1r.drive.jsonl" in " ".join(step.argv) for step in boot.actions))
    def test_the_steps_run_cold_first_then_merge_long_lanes_and_always_end_down(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            args = cell.parse_args(run_args(Path(tmp)))
            steps = memory_cells.boot_steps(cell.c1_boot(args), args)
        self.assertEqual([step.label for step in steps], ["up", "cold", "merge", "long", "lanes", "down"])
        self.assertTrue(all(step.timeout_s > 0 for step in steps))

    def test_the_up_step_goes_through_lab_py_on_quad_a_with_the_25_minute_deadline(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            args = cell.parse_args(run_args(Path(tmp)))
            up = memory_cells.boot_steps(cell.c1_boot(args), args)[0].argv
        self.assertEqual(up[2:4], ["up", "--quad"])
        self.assertEqual(up[up.index("--quad") + 1], "A")
        self.assertEqual(up[up.index("--deadline") + 1], "1500")

    def test_the_long_and_merge_sizes_reach_their_drive_steps(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            args = cell.parse_args(run_args(Path(tmp)))
            steps = {step.label: step.argv for step in memory_cells.boot_steps(cell.c1_boot(args), args)}
        self.assertEqual(steps["long"][steps["long"].index("--long-tokens") + 1], "150000,530000")
        self.assertEqual(steps["long"][steps["long"].index("--long-output-tokens") + 1], "64")
        self.assertEqual(steps["merge"][steps["merge"].index("--merge-tokens") + 1], "200000")

    def test_the_stall_dir_needs_its_mount(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            args = cell.parse_args(["run", "--image", "leaf", "--port", "18440", "--out-dir", f"{tmp}/o", "--env-file", str(write_env(Path(tmp)))])
            with self.assertRaises(SystemExit) as caught:
                cell.c1_boot(args)
        self.assertIn("--mount", str(caught.exception))

    def test_a_replica_port_is_refused(self) -> None:
        with self.assertRaises(SystemExit), redirect_stderr(StringIO()):
            cell.parse_args(["run", "--image", "x", "--port", "18274", "--out-dir", "o"])

    def test_dry_run_is_the_default_and_touches_nothing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp, mock.patch("subprocess.run", side_effect=AssertionError("touched")) as run:
            out = StringIO()
            with redirect_stdout(out):
                code = cell.main(run_args(Path(tmp)))
            plan = json.loads(out.getvalue())
            self.assertFalse(Path(tmp, "out").exists())
        self.assertEqual((code, run.call_count, plan["dry_run"], plan["touches_cards"], plan["cell"]), (0, 0, True, False, "c1"))
        self.assertEqual([step["label"] for step in plan["steps"]], ["up", "cold", "merge", "long", "lanes", "down"])

    def test_dry_run_and_execute_exclude_each_other(self) -> None:
        with tempfile.TemporaryDirectory() as tmp, self.assertRaises(SystemExit), redirect_stderr(StringIO()):
            cell.parse_args(run_args(Path(tmp), "--dry-run", "--execute"))


class StepTiming(unittest.TestCase):
    def test_one_lane_reads_its_own_gap(self) -> None:
        self.assertEqual(cell.step_ms_from_chunks([reply(count=40, gap=0.125)["chunks"]]), (125.0, 39))

    def test_lanes_are_read_only_where_they_all_decode(self) -> None:
        early = reply(start=0.0, count=100, gap=0.125)["chunks"]
        late = reply(start=3.0, count=100, gap=0.125)["chunks"]
        self.assertEqual(cell.step_ms_from_chunks([early, late]), (125.0, 75))

    def test_lanes_that_never_overlap_have_no_reading(self) -> None:
        self.assertIsNone(cell.step_ms_from_chunks([reply(start=0.0, count=20)["chunks"], reply(start=50.0, count=20)["chunks"]]))

    def test_a_window_with_too_few_tokens_has_no_reading(self) -> None:
        self.assertIsNone(cell.step_ms_from_chunks([reply(start=0.0, count=12)["chunks"], reply(start=0.8, count=12)["chunks"]]))


class Anchors(unittest.TestCase):
    def test_a_prefix_of_a_long_prompt_shares_its_salt(self) -> None:
        self.assertEqual(cell.anchor_salt("S", 262_400, (150_000, 530_000)), "S-long-530000")
        self.assertEqual(cell.anchor_salt("S", 530_000, (150_000, 530_000)), "S-long-530000")
        self.assertEqual(cell.anchor_salt("S", 2048, (150_000, 530_000)), "S-long-150000")

    def test_a_prompt_longer_than_every_long_size_has_its_own_salt(self) -> None:
        self.assertEqual(cell.anchor_salt("S", 700_000, (150_000, 530_000)), "S-anchor-700000")

    def test_the_fixed_prompt_of_a_shorter_size_is_a_prefix_of_a_longer_one(self) -> None:
        self.assertEqual(k1_cell.fixed_prompt(2048), k1_cell.fixed_prompt(4096)[:2048])

    def test_anchor_parsing(self) -> None:
        self.assertEqual(cell.lane_anchors("1024:2048,8192:530000"), {1024: 2048, 8192: 530000})
        with self.assertRaises(Exception):
            cell.lane_anchors("1024")


class Drives(unittest.TestCase):
    def rows(self, path: Path) -> list[dict]:
        return [json.loads(line) for line in path.read_text().splitlines()]

    def test_the_cold_drive_sends_one_short_prompt_with_a_few_tokens(self) -> None:
        send = mock.Mock(return_value=reply(count=4))
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp, "d.jsonl")
            self.assertEqual(cell.drive_cold(namespace(out=out), send), 0)
            rows = self.rows(out)
        self.assertEqual(rows, [{"drive": "cold", "status": "ok", "tokens": cell.COLD_TOKENS, "output_tokens": 4}])
        self.assertEqual(send.call_args.args[1:3], (k1_cell.fixed_prompt(cell.COLD_TOKENS), cell.COLD_OUTPUT_TOKENS))

    def test_the_merge_drive_repeats_one_greedy_request_with_one_salt(self) -> None:
        send = mock.Mock(side_effect=[reply(sha="x"), reply(sha="x")])
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp, "d.jsonl")
            self.assertEqual(cell.drive_merge(namespace(out=out), send), 0)
            rows = self.rows(out)
        self.assertEqual(send.call_count, cell.MERGE_REPEATS)
        self.assertTrue(all(call.args[1] == k1_cell.fixed_prompt(200_000) for call in send.call_args_list))
        self.assertEqual(len({call.kwargs["salt"] for call in send.call_args_list}), 1)
        self.assertEqual((rows[0]["drive"], rows[0]["tokens"], rows[0]["outputs_equal"]), ("merge", 200_000, True))

    def test_the_merge_drive_context_sits_in_the_native_2048_width_band(self) -> None:
        self.assertTrue(131_072 < cell.MERGE_TOKENS <= 262_144)

    def test_repeats_that_differ_are_recorded_unequal_and_fail_the_step(self) -> None:
        send = mock.Mock(side_effect=[reply(sha="x"), reply(sha="y")])
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp, "d.jsonl")
            self.assertEqual(cell.drive_merge(namespace(out=out), send), 1)
            self.assertFalse(self.rows(out)[0]["outputs_equal"])

    def test_long_sizes_run_cold_then_warm_on_one_salt_each(self) -> None:
        send = mock.Mock(side_effect=[reply(ttft=300.0), reply(ttft=2.0), reply(ttft=900.0), reply(ttft=3.0)])
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp, "d.jsonl")
            with redirect_stdout(StringIO()):
                self.assertEqual(cell.drive_long(namespace(out=out), send), 0)
            rows = self.rows(out)
        salts = [call.kwargs["salt"] for call in send.call_args_list]
        self.assertEqual(salts, ["S-long-150000"] * 2 + ["S-long-530000"] * 2)
        self.assertEqual([call.args[2] for call in send.call_args_list], [64] * 4)
        self.assertEqual([(row["tokens"], row["ttft_s"], row["warm_ttft_s"]) for row in rows], [(150_000, 300.0, 2.0), (530_000, 900.0, 3.0)])

    def test_a_long_row_that_errors_stops_the_drive_and_is_kept(self) -> None:
        send = mock.Mock(side_effect=TimeoutError("slow"))
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp, "d.jsonl")
            with redirect_stdout(StringIO()):
                self.assertEqual(cell.drive_long(namespace(out=out), send), 1)
            rows = self.rows(out)
        self.assertEqual(len(rows), 1)
        self.assertIn("TimeoutError", rows[0]["error"])

    def test_a_warm_turn_that_differs_from_the_cold_one_is_an_error_row(self) -> None:
        send = mock.Mock(side_effect=[reply(sha="a"), reply(sha="b")])
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp, "d.jsonl")
            with redirect_stdout(StringIO()):
                self.assertEqual(cell.drive_long(namespace(out=out, long_tokens=(150_000,)), send), 1)
            self.assertIn("differs", self.rows(out)[0]["error"])

    def lane_send(self, apart: bool = False) -> mock.Mock:
        """Every lane decodes 0.1 s per token from t=0; apart: each lane starts 500 s after the one before."""
        calls = itertools.count()

        def send(port: int, prompt: list[int], tokens: int, timeout: int, salt: str) -> dict:
            return reply(start=500.0 * next(calls) if apart else 0.0, count=tokens, gap=0.1)
        return mock.Mock(side_effect=send)

    def test_every_lane_count_at_every_width_gets_a_row_with_its_step_ms(self) -> None:
        send = self.lane_send()
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp, "d.jsonl")
            with redirect_stdout(StringIO()):
                self.assertEqual(cell.drive_lanes(namespace(out=out), send), 0)
            rows = self.rows(out)
        self.assertEqual([(row["lanes"], row["width"]) for row in rows], [(d, w) for w in (1024, 4096, 8192) for d in (1, 2, 4, 8, 16)])
        self.assertTrue(all(row["status"] == "ok" and row["step_ms"] == 100.0 for row in rows))
        self.assertEqual(send.call_count, 3 * (1 + 2 + 4 + 8 + 16))

    def test_lanes_that_do_not_decode_together_are_unreachable_not_errors(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp, "d.jsonl")
            with redirect_stdout(StringIO()):
                self.assertEqual(cell.drive_lanes(namespace(out=out, anchors={1024: 2048}), self.lane_send(apart=True)), 0)
            rows = self.rows(out)
        self.assertEqual(rows[0]["status"], "ok")  # one lane overlaps itself
        self.assertEqual({row["status"] for row in rows[1:]}, {"unreachable"})

    def test_every_lane_sends_the_anchor_prompt_on_the_anchors_warm_salt(self) -> None:
        send = self.lane_send()
        with tempfile.TemporaryDirectory() as tmp, redirect_stdout(StringIO()):
            cell.drive_lanes(namespace(out=Path(tmp, "d.jsonl"), anchors={4096: 262_400}), send)
        self.assertEqual(send.call_count, 1 + 2 + 4 + 8 + 16)
        self.assertEqual({len(call.args[1]) for call in send.call_args_list}, {262_400})
        self.assertEqual({call.kwargs["salt"] for call in send.call_args_list}, {"S-long-530000"})

    def test_a_failed_request_is_an_error_row_and_stops_the_drive(self) -> None:
        send = mock.Mock(side_effect=OSError("connection reset"))
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp, "d.jsonl")
            with redirect_stdout(StringIO()):
                self.assertEqual(cell.drive_lanes(namespace(out=out), send), 1)
            rows = self.rows(out)
        self.assertEqual((len(rows), rows[0]["status"]), (1, "error"))
        self.assertIn("connection reset", rows[0]["reason"])

    def test_the_step_budgets_cover_every_chunk_of_the_largest_prompts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            args = cell.parse_args(run_args(Path(tmp)))
            steps = {step.label: step.timeout_s for step in memory_cells.boot_steps(cell.c1_boot(args), args)}
        self.assertGreater(steps["long"], 2 * k1_cell.long_timeout_s(530_000))
        self.assertGreater(steps["lanes"], k1_cell.long_timeout_s(530_000))


class Execute(unittest.TestCase):
    """--execute with subprocess.run replaced: step order, the always-down rule, and how the log and rows are scored."""

    def run_cell(self, tmp: Path, *, up_code: int = 0, codes: dict[str, object] | None = None, lines: list[str] | None = None,
                 rows: list[dict] | None = None, variant: str = "c1") -> tuple[int, list[str], dict]:
        calls: list[str] = []
        name = f"claude-flash-launch-{variant}"

        def fake_run(argv: list[str], **kwargs: object) -> mock.Mock:
            calls.append(argv[2])
            if argv[2] == "up" and up_code == 0:
                (tmp / f"state-{name}.json").write_text(json.dumps({"startup_ready_s": 700.0}))
            if argv[2] == "down":
                (tmp / f"{name}.log").write_text("\n".join(lines or []) + "\n")
                out = tmp / "out"
                out.mkdir(exist_ok=True)
                (out / f"{name}.drive.jsonl").write_text("\n".join(json.dumps(row) for row in rows or []) + "\n")
            code = {"up": up_code, **(codes or {})}.get(argv[2], 0)
            if code == "timeout":
                raise subprocess.TimeoutExpired(argv, 1)
            return mock.Mock(returncode=code, stderr="")

        with mock.patch("subprocess.run", side_effect=fake_run), mock.patch.object(cell, "LANE", tmp), mock.patch.object(k1_cell, "LANE", tmp), \
                redirect_stdout(StringIO()), redirect_stderr(StringIO()):
            code = cell.main(traced_args(tmp, "--execute") if variant == "c1p" else run_args(tmp, "--execute"))
        return code, calls, json.loads((tmp / "out" / f"{variant}-T.json").read_text())

    def test_a_clean_run_passes_and_exits_zero(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            code, calls, result = self.run_cell(Path(tmp), lines=mock_log(), rows=drive_rows())
        self.assertEqual((code, calls, result["verdict"]), (0, ["up", "cold", "merge", "long", "lanes", "down"], "PASS"))

    def test_down_still_runs_when_a_drive_times_out_and_later_drives_are_skipped(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            code, calls, result = self.run_cell(Path(tmp), codes={"merge": "timeout"}, lines=mock_log(), rows=drive_rows())
        self.assertEqual((calls, code), (["up", "cold", "merge", "down"], 1))
        self.assertEqual(result["verdict"], "NOT_MEASURED")
        self.assertIn("merge", result["gaps"][0])

    def test_a_failed_boot_runs_no_drive_and_is_not_measured(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            code, calls, result = self.run_cell(Path(tmp), up_code=1)
        self.assertEqual((code, calls, result["verdict"]), (1, ["up", "down"], "NOT_MEASURED"))

    def test_a_fail_reading_stands_over_a_failed_drive(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            lines = [*mock_log(), "(Worker_TP1_EP1 pid=101) WARNING PT_DEVMEM out of memory"]
            code, _, result = self.run_cell(Path(tmp), codes={"lanes": 1}, lines=lines, rows=drive_rows())
        self.assertEqual((code, result["verdict"]), (1, "FAIL"))

    def test_the_result_file_holds_the_steps_and_the_readings(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            _, _, result = self.run_cell(Path(tmp), lines=mock_log(), rows=drive_rows())
        self.assertEqual([step["label"] for step in result["steps"]][0], "up")
        self.assertIn("decode_step_ms", [item["name"] for item in result["readings"]])


class TracedVariant(unittest.TestCase):
    """c1p (plan 12.3a): the replay trace hook's settings, its mount, the wait for its files, and the report in the result."""

    def plan(self, tmp: Path, *extra: str) -> tuple[cell.argparse.Namespace, memory_cells.Boot]:
        args = cell.parse_args(traced_args(tmp, *extra))
        return args, cell.c1_boot(args)

    def test_the_p_variant_waits_for_the_trace_files_before_it_stops_the_container(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            args, boot = self.plan(Path(tmp))
            steps = memory_cells.boot_steps(boot, args)
            wait = next(step for step in steps if step.label == "trace-wait")
        self.assertEqual([step.label for step in steps], ["up", "cold", "merge", "long", "lanes", "trace-wait", "down"])
        self.assertIn("replay_trace_report.py", " ".join(wait.argv))
        self.assertEqual(wait.argv[wait.argv.index("--dir") + 1], f"{tmp}/trace")
        self.assertEqual(wait.argv[wait.argv.index("--runs") + 1], ",".join(TRACE_RUNS))

    def test_the_other_variants_have_no_wait_step(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            args = cell.parse_args(run_args(Path(tmp)))
            self.assertNotIn("trace-wait", [step.label for step in memory_cells.boot_steps(cell.c1_boot(args), args)])

    def test_dry_run_prints_the_trace_plan_and_touches_nothing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp, mock.patch("subprocess.run", side_effect=AssertionError("touched")):
            out = StringIO()
            with redirect_stdout(out):
                code = cell.main(traced_args(Path(tmp)))
            plan = json.loads(out.getvalue())
        self.assertEqual((code, plan["cell"], plan["touches_cards"]), (0, "c1p", False))
        traced = plan["trace"]
        self.assertEqual([target["run_id"] for target in traced["targets"]], TRACE_RUNS)
        self.assertEqual((traced["host_dir"], traced["directory"], traced["min_free_mib"]), (f"{tmp}/trace", "/k1trace", 1024))
        self.assertEqual([target["rehearsal_replays"] for target in traced["targets"]], [[1, 2], [18, 19]])

    def test_a_plain_dry_run_has_no_trace_section(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out = StringIO()
            with redirect_stdout(out):
                cell.main(run_args(Path(tmp)))
        self.assertNotIn("trace", json.loads(out.getvalue()))

    def test_the_p_variant_refuses_an_env_file_that_lost_its_trace_settings(self) -> None:
        with tempfile.TemporaryDirectory() as tmp, self.assertRaises(SystemExit) as caught:
            cell.c1_boot(cell.parse_args(traced_args(Path(tmp), text=ENV_TEXT)))
        self.assertIn("HPU_GLM53_REPLAY_TRACE_DIR", str(caught.exception))
        self.assertIn("graphs-c1p.env", str(caught.exception))

    def test_another_variant_refuses_a_trace_setting(self) -> None:
        with tempfile.TemporaryDirectory() as tmp, self.assertRaises(SystemExit) as caught:
            cell.load_env(write_env(Path(tmp), ENV_TEXT + TRACE_ENV), "c1")
        self.assertIn("only c1p", str(caught.exception))

    def test_a_malformed_plan_or_a_missing_census_flag_refuses_before_a_card_window(self) -> None:
        bad_plan = ENV_TEXT + TRACE_ENV.replace("prompt:1024x8192@3", "prompt:1024x8192")
        no_census = ENV_TEXT + TRACE_ENV.replace("HPU_GLM53_PREFILL_HBM_CENSUS=1\n", "")
        for text, words in ((bad_plan, "KIND:ROWSxWIDTH@NTH"), (no_census, "HPU_GLM53_PREFILL_HBM_CENSUS=1")):
            with self.subTest(words=words), tempfile.TemporaryDirectory() as tmp, self.assertRaises(SystemExit) as caught:
                cell.c1_boot(cell.parse_args(traced_args(Path(tmp), text=text)))
            self.assertIn(words, str(caught.exception))

    def test_the_trace_directory_needs_its_mount(self) -> None:
        with tempfile.TemporaryDirectory() as tmp, self.assertRaises(SystemExit) as caught:
            cell.c1_boot(cell.parse_args(traced_args(Path(tmp), mount=False)))
        self.assertIn("--mount <host-dir>:/k1trace", str(caught.exception))

    def test_a_run_whose_trace_files_are_all_there_passes_and_the_report_is_in_the_result(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "trace").mkdir()
            for run_id in TRACE_RUNS:
                write_run(Path(tmp) / "trace", run_id, trace(10_000.0, COMPLETE_OPS), 10.0)
            code, calls, result = Execute.run_cell(self, Path(tmp), variant="c1p", lines=mock_log(), rows=drive_rows())
        self.assertEqual((code, result["verdict"]), (0, "PASS"))
        self.assertEqual(calls[-2:], ["wait", "down"])
        self.assertEqual(result["replay_trace"]["summary"], {"complete": 2, "partial": 0, "missing": 0})

    def test_a_run_with_no_trace_files_is_not_measured_and_names_each_missing_run(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            code, _, result = Execute.run_cell(self, Path(tmp), variant="c1p", lines=mock_log(), rows=drive_rows())
        self.assertEqual((code, result["verdict"]), (1, "NOT_MEASURED"))
        self.assertTrue(all(any(run_id in gap for gap in result["gaps"]) for run_id in TRACE_RUNS), result["gaps"])

    def test_the_wait_still_runs_when_a_drive_failed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            _, calls, _ = Execute.run_cell(self, Path(tmp), variant="c1p", codes={"merge": 1}, lines=mock_log(), rows=drive_rows())
        self.assertEqual(calls, ["up", "cold", "merge", "wait", "down"])


class Entry(unittest.TestCase):
    def test_the_scorer_and_the_cell_agree_on_the_default_long_sizes(self) -> None:
        self.assertEqual(cell.DEFAULT_LONG_TOKENS, scorer.DEFAULT_LONG_SIZES)

    def test_the_default_anchors_cover_the_three_decode_widths(self) -> None:
        self.assertEqual(sorted(cell.DEFAULT_ANCHORS), list(scorer.DECODE_WIDTHS))
        self.assertGreater(cell.DEFAULT_ANCHORS[4096], 262_144)  # native width 4096 starts above 262,144 tokens
        self.assertGreater(cell.DEFAULT_ANCHORS[8192], 524_288)
        self.assertLessEqual(cell.DEFAULT_ANCHORS[1024], 131_072)


if __name__ == "__main__":
    unittest.main()
