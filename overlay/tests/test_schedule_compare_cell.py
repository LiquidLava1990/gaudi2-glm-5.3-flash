"""Bitwise check 3 (plan step 47): one greedy lane replayed at K = 0 and at the slice table, with and without tail-first, must give
identical tokens. Fake runner and fake HTTP only; nothing here touches a card."""

from __future__ import annotations

import json
import re
import sys
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "cells"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import c1_cell  # noqa: E402
import schedule_compare_cell as cell  # noqa: E402
from test_c1_cell import reply, write_env  # noqa: E402

TOKENS = list(range(1000, 1064))


def run_args(tmp: Path, *extra: str) -> list[str]:
    return ["run", "--image", "leaf", "--port", "18440", "--out-dir", str(tmp / "out"), "--env-file", str(write_env(tmp)),
            "--mount", f"{tmp}/trace:/k1trace", "--stamp", "T", *extra]


def tokens_json(tokens: list[int]) -> dict:
    return {"choices": [{"logprobs": {"tokens": [f"token_id:{token}" for token in tokens]}}]}


class Comparing(unittest.TestCase):
    def test_equal_tokens_are_equal(self) -> None:
        self.assertEqual(cell.compare_tokens(TOKENS, list(TOKENS)), (cell.PASS, "64 tokens identical", None))

    def test_the_first_difference_is_found(self) -> None:
        other = TOKENS[:17] + [7] + TOKENS[18:]
        verdict, detail, index = cell.compare_tokens(TOKENS, other)
        self.assertEqual((verdict, index), (cell.FAIL, 17))
        self.assertIn("token 17", detail)

    def test_a_shorter_run_differs_at_its_end(self) -> None:
        verdict, detail, index = cell.compare_tokens(TOKENS, TOKENS[:40])
        self.assertEqual((verdict, index), (cell.FAIL, 40))
        self.assertIn("40 tokens", detail)

    def test_no_tokens_is_not_measured_never_equal(self) -> None:
        self.assertEqual(cell.compare_tokens([], [])[0], cell.NOT_MEASURED)
        self.assertEqual(cell.compare_tokens(TOKENS, [])[0], cell.NOT_MEASURED)
        self.assertEqual(cell.compare_tokens([], TOKENS)[0], cell.NOT_MEASURED)


class Arms(unittest.TestCase):
    def test_four_arms_cover_k0_and_the_slice_table_with_and_without_tail_first(self) -> None:
        self.assertEqual([(arm.name, bool(arm.slice_table), arm.tail_first) for arm in cell.ARMS],
                         [("k0-fifo", False, False), ("k0-tail", False, True), ("slice-fifo", True, False), ("slice-tail", True, True)])

    def test_the_slice_table_is_the_plan_table_in_width_steps_pairs(self) -> None:
        self.assertEqual(cell.SLICE_TABLE, "1024:10,4096:22,8192:36")
        self.assertTrue(re.fullmatch(r"\d+:\d+(,\d+:\d+)*", cell.SLICE_TABLE))

    def test_arm_env_sets_only_what_the_arm_changes(self) -> None:
        base = c1_cell.load_env(self.env_file())
        envs = {arm.name: cell.arm_env(base, arm) for arm in cell.ARMS}
        self.assertNotIn("VLLM_GAUDI_MIXED_SLICE_STEPS", envs["k0-fifo"])
        self.assertEqual(envs["slice-fifo"]["VLLM_GAUDI_MIXED_SLICE_STEPS"], cell.SLICE_TABLE)
        self.assertEqual([envs[name]["VLLM_GAUDI_MIXED_TAIL_FIRST"] for name in ("k0-fifo", "k0-tail", "slice-fifo", "slice-tail")],
                         ["0", "1", "0", "1"])
        self.assertEqual({env["VLLM_COMPLETION_FIRST_LONG_PREFILL"] for env in envs.values()}, {"0"})

    def test_the_ballast_is_not_held_for_a_numerics_run(self) -> None:
        env = cell.arm_env(c1_cell.load_env(self.env_file()), cell.ARMS[0])
        self.assertNotIn("HPU_GLM53_K1_BALLAST_MIB", env)
        self.assertEqual(env["VLLM_GAUDI_MIXED_PROMPT_KEYS"], "1024:8192,1024:1024")

    def env_file(self) -> Path:
        return write_env(Path(self.enterContext(tempfile.TemporaryDirectory())))


class Runner(unittest.TestCase):
    def test_equal_arms_pass(self) -> None:
        result = cell.compare_arms(cell.FakeRunner.from_mode("equal"), cell.ARMS)
        self.assertEqual(result["verdict"], cell.PASS)
        self.assertEqual([item["verdict"] for item in result["arms"]], [cell.PASS] * 3)

    def test_one_unequal_arm_fails_and_is_named(self) -> None:
        result = cell.compare_arms(cell.FakeRunner.from_mode("unequal"), cell.ARMS)
        self.assertEqual(result["verdict"], cell.FAIL)
        bad = [item for item in result["arms"] if item["verdict"] == cell.FAIL]
        self.assertEqual([(item["arm"], item["first_difference"]) for item in bad], [("slice-tail", 17)])

    def test_an_arm_that_produced_nothing_is_not_measured(self) -> None:
        result = cell.compare_arms(cell.FakeRunner.from_mode("empty"), cell.ARMS)
        self.assertEqual(result["verdict"], cell.NOT_MEASURED)

    def test_an_arm_that_errored_is_not_measured_with_its_reason(self) -> None:
        runner = cell.FakeRunner({arm.name: cell.ArmRun(list(TOKENS)) for arm in cell.ARMS} | {"k0-tail": cell.ArmRun([], "boot failed")})
        result = cell.compare_arms(runner, cell.ARMS)
        self.assertEqual(result["verdict"], cell.NOT_MEASURED)
        self.assertIn("boot failed", result["arms"][0]["detail"])

    def test_a_background_error_makes_equal_tokens_not_measured(self) -> None:
        runner = cell.FakeRunner({arm.name: cell.ArmRun(list(TOKENS)) for arm in cell.ARMS} | {"slice-fifo": cell.ArmRun(list(TOKENS), "lane 3 died")})
        self.assertEqual(cell.compare_arms(runner, cell.ARMS)["verdict"], cell.NOT_MEASURED)

    def test_a_reference_without_tokens_leaves_every_arm_not_measured(self) -> None:
        runner = cell.FakeRunner({arm.name: cell.ArmRun(list(TOKENS)) for arm in cell.ARMS} | {"k0-fifo": cell.ArmRun([])})
        result = cell.compare_arms(runner, cell.ARMS)
        self.assertEqual({item["verdict"] for item in result["arms"]}, {cell.NOT_MEASURED})

    def test_a_fail_beats_not_measured(self) -> None:
        runner = cell.FakeRunner({"k0-fifo": cell.ArmRun(list(TOKENS)), "k0-tail": cell.ArmRun([]),
                                  "slice-fifo": cell.ArmRun(TOKENS[:5] + [9]), "slice-tail": cell.ArmRun(list(TOKENS))})
        self.assertEqual(cell.compare_arms(runner, cell.ARMS)["verdict"], cell.FAIL)

    def test_the_fake_command_exits_zero_only_for_equal(self) -> None:
        codes = {}
        for mode in ("equal", "unequal", "empty"):
            with redirect_stdout(StringIO()) as out:
                codes[mode] = cell.main(["fake", "--mode", mode])
            self.assertIn("SCHEDULE_COMPARE", out.getvalue())
        self.assertEqual(codes, {"equal": 0, "unequal": 1, "empty": 1})


class TokenIds(unittest.TestCase):
    def test_token_ids_come_out_of_the_logprobs_tokens(self) -> None:
        self.assertEqual(cell.parse_token_ids(tokens_json([5, 6, 7])), [5, 6, 7])

    def test_a_reply_without_logprobs_names_the_request_option(self) -> None:
        with self.assertRaises(ValueError) as caught:
            cell.parse_token_ids({"choices": [{"text": "hi"}]})
        self.assertIn("return_tokens_as_token_ids", str(caught.exception))

    def test_a_token_that_is_text_not_an_id_is_refused(self) -> None:
        with self.assertRaises(ValueError) as caught:
            cell.parse_token_ids({"choices": [{"logprobs": {"tokens": ["hello"]}}]})
        self.assertIn("token_id:", str(caught.exception))


class Drive(unittest.TestCase):
    def namespace(self, out: Path) -> mock.Mock:
        return mock.Mock(port=18440, arm="slice-tail", out=out, salt="S")

    def test_background_lanes_and_a_long_prompt_run_while_the_subject_is_recorded(self) -> None:
        send = mock.Mock(side_effect=lambda *a, on_first=None, **k: (on_first() if on_first else None) or reply(count=4))
        fetch = mock.Mock(return_value=list(TOKENS))
        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(cell, "LONG_HEAD_START_S", 1):
            out = Path(tmp, "t.jsonl")
            self.assertEqual(cell.drive_arm(self.namespace(out), send, fetch), 0)
            row = json.loads(out.read_text())
        self.assertEqual(send.call_count, cell.BACKGROUND_LANES + 1)  # the lanes and the long prompt
        self.assertEqual(fetch.call_count, 1)
        self.assertEqual((row["arm"], row["tokens"], row["error"]), ("slice-tail", TOKENS, ""))

    def test_the_subject_prompt_and_salt_are_the_same_for_every_arm(self) -> None:
        send = mock.Mock(side_effect=lambda *a, on_first=None, **k: (on_first() if on_first else None) or reply(count=4))
        calls = []
        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(cell, "LONG_HEAD_START_S", 1):
            for arm in ("k0-fifo", "slice-tail"):
                namespace = self.namespace(Path(tmp, f"{arm}.jsonl"))
                namespace.arm = arm
                cell.drive_arm(namespace, send, lambda *a, **k: calls.append((a, k)) or list(TOKENS))
        self.assertEqual(calls[0], calls[1])

    def test_a_failed_background_lane_is_recorded_as_an_error(self) -> None:
        send = mock.Mock(side_effect=OSError("reset"))
        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(cell, "LONG_HEAD_START_S", 1):
            out = Path(tmp, "t.jsonl")
            self.assertEqual(cell.drive_arm(self.namespace(out), send, mock.Mock(return_value=list(TOKENS))), 1)
            self.assertIn("OSError", json.loads(out.read_text())["error"])


class Execute(unittest.TestCase):
    def run_cell(self, tmp: Path, tokens_for: dict[str, list[int]], *extra: str) -> tuple[int, list[list[str]], dict]:
        calls: list[list[str]] = []

        def fake_run(argv: list[str], **kwargs: object) -> mock.Mock:
            calls.append(argv)
            if argv[2] == "drive":
                arm, out = argv[argv.index("--arm") + 1], Path(argv[argv.index("--out") + 1])
                with out.open("a") as handle:
                    handle.write(json.dumps({"arm": arm, "tokens": tokens_for[arm], "error": ""}) + "\n")
            return mock.Mock(returncode=0, stderr="")

        with mock.patch("subprocess.run", side_effect=fake_run), mock.patch.object(cell.k1_cell, "LANE", tmp), \
                redirect_stdout(StringIO()), redirect_stderr(StringIO()):
            Path(tmp, "out").mkdir()
            code = cell.main(run_args(tmp, "--execute", *extra))
        return code, calls, json.loads((tmp / "out" / "schedule-compare-T.json").read_text())

    def test_four_boots_each_up_drive_down_and_equal_tokens_pass(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            code, calls, result = self.run_cell(Path(tmp), {arm.name: list(TOKENS) for arm in cell.ARMS})
        self.assertEqual((code, result["verdict"]), (0, "PASS"))
        self.assertEqual([call[2] for call in calls], ["up", "drive", "down"] * 4)

    def test_only_the_slice_arms_get_the_slice_table_in_their_boot(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            _, calls, _ = self.run_cell(Path(tmp), {arm.name: list(TOKENS) for arm in cell.ARMS})
        ups = [" ".join(call) for call in calls if call[2] == "up"]
        self.assertEqual(["VLLM_GAUDI_MIXED_SLICE_STEPS=" in up for up in ups], [False, False, True, True])
        self.assertEqual(["VLLM_GAUDI_MIXED_TAIL_FIRST=1" in up for up in ups], [False, True, False, True])

    def test_an_unequal_arm_fails_the_run(self) -> None:
        tokens = {arm.name: list(TOKENS) for arm in cell.ARMS} | {"slice-fifo": TOKENS[:3] + [1]}
        with tempfile.TemporaryDirectory() as tmp:
            code, _, result = self.run_cell(Path(tmp), tokens)
        self.assertEqual((code, result["verdict"]), (1, "FAIL"))

    def test_the_arms_option_runs_a_subset_with_the_first_as_reference(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            code, calls, result = self.run_cell(Path(tmp), {arm.name: list(TOKENS) for arm in cell.ARMS}, "--arms", "k0-fifo,slice-tail")
        self.assertEqual((code, len([call for call in calls if call[2] == "up"]), [item["arm"] for item in result["arms"]]), (0, 2, ["slice-tail"]))

    def test_an_unknown_arm_is_refused_with_the_valid_names(self) -> None:
        with tempfile.TemporaryDirectory() as tmp, self.assertRaises(SystemExit), redirect_stderr(StringIO()) as err:
            cell.parse_args(run_args(Path(tmp), "--arms", "nope"))
        self.assertIn("k0-fifo", err.getvalue())

    def test_a_failed_boot_is_not_measured(self) -> None:
        runs = []

        def fake_run(argv: list[str], **kwargs: object) -> mock.Mock:
            runs.append(argv[2])
            return mock.Mock(returncode=1 if argv[2] == "up" else 0, stderr="boom")

        with tempfile.TemporaryDirectory() as tmp, mock.patch("subprocess.run", side_effect=fake_run), \
                redirect_stdout(StringIO()), redirect_stderr(StringIO()):
            Path(tmp, "out").mkdir()
            code = cell.main(run_args(Path(tmp), "--execute"))
            result = json.loads(Path(tmp, "out", "schedule-compare-T.json").read_text())
        self.assertEqual((code, result["verdict"]), (1, "NOT_MEASURED"))
        self.assertNotIn("drive", runs)
        self.assertEqual(runs.count("down"), 4)


class DryRun(unittest.TestCase):
    def test_the_default_prints_every_boot_and_runs_nothing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp, mock.patch("subprocess.run", side_effect=AssertionError("touched")) as run:
            out = StringIO()
            with redirect_stdout(out):
                code = cell.main(run_args(Path(tmp)))
            plan = json.loads(out.getvalue())
            self.assertFalse(Path(tmp, "out").exists())
        self.assertEqual((code, run.call_count, plan["dry_run"], plan["touches_cards"]), (0, 0, True, False))
        self.assertEqual([arm["arm"] for arm in plan["arms"]], [arm.name for arm in cell.ARMS])
        self.assertEqual([step["label"] for step in plan["arms"][0]["steps"]], ["up", "drive", "down"])

    def test_the_drive_step_is_bounded(self) -> None:
        with tempfile.TemporaryDirectory() as tmp, redirect_stdout(StringIO()) as out:
            cell.main(run_args(Path(tmp)))
        drive = json.loads(out.getvalue())["arms"][0]["steps"][1]
        self.assertGreater(drive["timeout_s"], c1_cell.k1_cell.long_timeout_s(cell.LONG_TOKENS))
        self.assertLess(drive["timeout_s"], 4 * 3600)


if __name__ == "__main__":
    unittest.main()
