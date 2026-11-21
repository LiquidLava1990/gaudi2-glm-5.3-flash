"""Bitwise check 2 of DUAL-SUSTAINED-PLAN.md section 12.3 (build step 6): one recorded sequence replayed same-step vs separate-step, and
d, p, d, p vs isolated, compared with torch.equal on hidden or token ids; plus the three excluded C787 controls. Fake runner, fake HTTP and
temp files only; nothing here touches a card."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from pathlib import Path
from unittest import mock

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "cells"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import c1_cell  # noqa: E402
import dual_serial_cell as cell  # noqa: E402
from test_c1_cell import reply, write_env  # noqa: E402

TOKENS = list(range(1000, 1064))


def readings(tokens: list[int]) -> list[cell.Reading]:
    return cell.token_readings(tokens)


def good_runs(**overrides: cell.ArmRun) -> dict[str, cell.ArmRun]:
    runs = {arm.name: cell.ArmRun(readings(TOKENS)) for arm in cell.ARMS}
    return runs | overrides


class FakeControls:
    def __init__(self, *verdicts: str) -> None:
        self.verdicts = verdicts or (cell.RECORDED, cell.RECORDED, cell.PASS)

    def run(self) -> list[dict]:
        return [{"control": name, "verdict": verdict, "detail": "fake"} for name, verdict in zip(cell.CONTROL_NAMES, self.verdicts, strict=True)]


def run_args(tmp: Path, *extra: str) -> list[str]:
    return ["run", "--image", "leaf", "--port", "18440", "--out-dir", str(tmp / "out"), "--env-file", str(write_env(tmp)),
            "--mount", f"{tmp}/trace:/k1trace", "--stamp", "T", *extra]


class Comparing(unittest.TestCase):
    def test_equal_tokens_pass(self) -> None:
        self.assertEqual(cell.compare_readings(readings(TOKENS), readings(TOKENS))[0], cell.PASS)

    def test_the_first_differing_token_and_step_are_named(self) -> None:
        other = TOKENS[:17] + [7] + TOKENS[18:]
        verdict, detail, step = cell.compare_readings(readings(TOKENS), readings(other))
        self.assertEqual((verdict, step), (cell.FAIL, "decode step 17"))
        self.assertIn("decode step 17", detail)
        self.assertIn("token 1017", detail)
        self.assertIn("token 7", detail)

    def test_a_shorter_run_fails_at_the_first_missing_step(self) -> None:
        verdict, detail, step = cell.compare_readings(readings(TOKENS), readings(TOKENS[:40]))
        self.assertEqual((verdict, step), (cell.FAIL, "decode step 40"))
        self.assertIn("missing", detail)

    def test_a_longer_run_fails_at_its_first_extra_step(self) -> None:
        verdict, _, step = cell.compare_readings(readings(TOKENS[:40]), readings(TOKENS))
        self.assertEqual((verdict, step), (cell.FAIL, "decode step 40"))

    def test_zero_compared_readings_is_not_measured_never_pass(self) -> None:
        for reference, other in (([], []), (readings(TOKENS), []), ([], readings(TOKENS))):
            self.assertEqual(cell.compare_readings(reference, other)[0], cell.NOT_MEASURED)

    def test_hidden_tensors_are_compared_with_torch_equal(self) -> None:
        left = [cell.Reading("hidden r0 start 0", torch.arange(6.0))]
        same = [cell.Reading("hidden r0 start 0", torch.arange(6.0))]
        self.assertEqual(cell.compare_readings(left, same)[0], cell.PASS)
        off_by_one_ulp = torch.arange(6.0)
        off_by_one_ulp[3] = torch.nextafter(off_by_one_ulp[3], torch.tensor(9.0))
        verdict, detail, step = cell.compare_readings(left, [cell.Reading("hidden r0 start 0", off_by_one_ulp)])
        self.assertEqual((verdict, step), (cell.FAIL, "hidden r0 start 0"))
        self.assertIn("max abs diff", detail)

    def test_a_tensor_against_a_token_differs_instead_of_raising(self) -> None:
        verdict, _, _ = cell.compare_readings([cell.Reading("s", 5)], [cell.Reading("s", torch.tensor([5]))])
        self.assertEqual(verdict, cell.FAIL)

    def test_saved_hidden_folders_compare_by_rank_and_start(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            first, second = Path(tmp, "a"), Path(tmp, "b")
            for folder in (first, second):
                folder.mkdir()
                torch.save(torch.ones(4), folder / "hidden-r0-1024-8192-0.pt")
                torch.save(torch.zeros(4), folder / "hidden-r1-1024-8192-1024.pt")
            torch.save(torch.ones(4) * 2, second / "hidden-r1-1024-8192-1024.pt")
            steps = [item.step for item in cell.hidden_readings(first)]
            verdict, _, step = cell.compare_readings(cell.hidden_readings(first), cell.hidden_readings(second))
        self.assertEqual(steps, ["hidden r0 start 0", "hidden r1 start 1024"])
        self.assertEqual((verdict, step), (cell.FAIL, "hidden r1 start 1024"))


class HiddenCommand(unittest.TestCase):
    def folder(self, tmp: str, name: str, scale: float) -> Path:
        path = Path(tmp, name)
        path.mkdir()
        torch.save(torch.ones(4) * scale, path / "hidden-r0-1024-8192-0.pt")
        return path

    def test_equal_folders_exit_zero_and_different_ones_name_the_step(self) -> None:
        with tempfile.TemporaryDirectory() as tmp, redirect_stdout(StringIO()) as out:
            first, same, other = (self.folder(tmp, "a", 1.0), self.folder(tmp, "b", 1.0), self.folder(tmp, "c", 2.0))
            codes = (cell.main(["hidden", "--a", str(first), "--b", str(same)]), cell.main(["hidden", "--a", str(first), "--b", str(other)]))
        self.assertEqual(codes, (0, 1))
        self.assertIn("hidden r0 start 0", out.getvalue())

    def test_an_empty_folder_is_not_measured(self) -> None:
        with tempfile.TemporaryDirectory() as tmp, redirect_stdout(StringIO()) as out:
            code = cell.main(["hidden", "--a", str(self.folder(tmp, "a", 1.0)), "--b", str(Path(tmp, "none"))])
        self.assertEqual(code, 1)
        self.assertIn("DUAL_SERIAL_HIDDEN: NOT_MEASURED", out.getvalue())


class Arms(unittest.TestCase):
    def test_four_arms_cover_same_step_separate_step_isolated_and_dpdp(self) -> None:
        self.assertEqual([(arm.name, arm.separate, arm.slice_table, arm.pressure) for arm in cell.ARMS],
                         [("same-step", False, "", True), ("separate-step", True, "", True), ("isolated", False, "", False),
                          ("dpdp", False, cell.DPDP_SLICE, True)])

    def test_the_two_pairs_name_their_reference_first(self) -> None:
        self.assertEqual([(p.name, p.reference, p.other) for p in cell.PAIRS],
                         [("separate-vs-same-step", "same-step", "separate-step"), ("dpdp-vs-isolated", "isolated", "dpdp")])

    def test_arm_env_states_the_seam_flag_and_drops_the_ballast(self) -> None:
        base = {"HPU_GLM53_K1_BALLAST_MIB": "512", "KEEP": "1"}
        envs = {arm.name: cell.arm_env(base, arm) for arm in cell.ARMS}
        self.assertEqual({name: env[cell.SEPARATE_FLAG] for name, env in envs.items()},
                         {"same-step": "0", "separate-step": "1", "isolated": "0", "dpdp": "0"})
        self.assertEqual(envs["dpdp"][cell.SLICE_FLAG], cell.DPDP_SLICE)
        self.assertEqual([cell.SLICE_FLAG in env for env in envs.values()], [False, False, False, True])
        self.assertTrue(all("HPU_GLM53_K1_BALLAST_MIB" not in env and env["KEEP"] == "1" for env in envs.values()))


class Pairs(unittest.TestCase):
    def test_equal_runs_pass_both_pairs(self) -> None:
        result = cell.run_cell(cell.FakeRunner(good_runs()), FakeControls())
        self.assertEqual(result["verdict"], cell.PASS)
        self.assertEqual([item["verdict"] for item in result["pairs"]], [cell.PASS, cell.PASS])

    def test_a_differing_separate_step_fails_and_names_the_pair_step_and_token(self) -> None:
        bad = readings(TOKENS[:5] + [1] + TOKENS[6:])
        result = cell.run_cell(cell.FakeRunner(good_runs(**{"separate-step": cell.ArmRun(bad)})), FakeControls())
        self.assertEqual(result["verdict"], cell.FAIL)
        failed = [item for item in result["pairs"] if item["verdict"] == cell.FAIL]
        self.assertEqual([(item["pair"], item["first_difference"]) for item in failed], [("separate-vs-same-step", "decode step 5")])
        self.assertIn("token 1005", failed[0]["detail"])

    def test_a_dpdp_decode_that_saw_prompt_state_fails_the_second_pair(self) -> None:
        result = cell.run_cell(cell.FakeRunner(good_runs(dpdp=cell.ArmRun(readings(TOKENS[:2] + [0] + TOKENS[3:])))), FakeControls())
        self.assertEqual([item["verdict"] for item in result["pairs"]], [cell.PASS, cell.FAIL])

    def test_an_empty_arm_is_not_measured(self) -> None:
        result = cell.run_cell(cell.FakeRunner(good_runs(isolated=cell.ArmRun([]))), FakeControls())
        self.assertEqual(result["verdict"], cell.NOT_MEASURED)

    def test_an_errored_arm_is_not_measured_with_its_reason_even_when_tokens_match(self) -> None:
        result = cell.run_cell(cell.FakeRunner(good_runs(dpdp=cell.ArmRun(readings(TOKENS), "pressure lane died"))), FakeControls())
        self.assertEqual(result["verdict"], cell.NOT_MEASURED)
        self.assertIn("pressure lane died", result["pairs"][1]["detail"])

    def test_a_fail_beats_not_measured(self) -> None:
        runs = good_runs(isolated=cell.ArmRun([]), **{"separate-step": cell.ArmRun(readings([9] * 64))})
        self.assertEqual(cell.run_cell(cell.FakeRunner(runs), FakeControls())["verdict"], cell.FAIL)

    def test_the_result_only_reports_verdicts_pairs_and_controls(self) -> None:
        result = cell.run_cell(cell.FakeRunner(good_runs()), FakeControls())
        self.assertEqual(sorted(result), ["controls", "pairs", "verdict"])


class Controls(unittest.TestCase):
    def test_the_three_excluded_controls_are_named(self) -> None:
        self.assertEqual(cell.CONTROL_NAMES, ("diff-graph", "diff-step", "flag-off-observables"))

    def test_a_failing_control_fails_an_otherwise_equal_run(self) -> None:
        result = cell.run_cell(cell.FakeRunner(good_runs()), FakeControls(cell.RECORDED, cell.RECORDED, cell.FAIL))
        self.assertEqual(result["verdict"], cell.FAIL)

    def test_an_unmeasured_control_keeps_a_run_from_passing(self) -> None:
        result = cell.run_cell(cell.FakeRunner(good_runs()), FakeControls(cell.NOT_MEASURED, cell.RECORDED, cell.PASS))
        self.assertEqual(result["verdict"], cell.NOT_MEASURED)

    def test_recorded_controls_are_neutral(self) -> None:
        self.assertEqual(cell.run_cell(cell.FakeRunner(good_runs()), FakeControls())["verdict"], cell.PASS)

    def write(self, folder: Path, name: str, text: str) -> Path:
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / name
        path.write_text(text)
        return path

    def test_the_diff_control_records_changed_added_and_removed_definitions(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            old = self.write(Path(tmp, "o"), "m.py", "X = 1\ndef a():\n    return 1\ndef b():\n    return 2\nclass C:\n    def m(self):\n        return 1\n")
            new = self.write(Path(tmp, "n"), "m.py", "X = 1\ndef a():\n    return 5\ndef c():\n    return 3\nclass C:\n    def m(self):\n        return 1\n")
            result = cell.diff_control("diff-graph", old, new)
        self.assertEqual(result["verdict"], cell.RECORDED)
        self.assertEqual((result["changed"], result["added"], result["removed"]), (["a"], ["c"], ["b"]))

    def test_the_diff_control_is_not_measured_when_a_tree_is_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            result = cell.diff_control("diff-step", Path(tmp, "nope.py"), Path(tmp, "nope2.py"))
        self.assertEqual(result["verdict"], cell.NOT_MEASURED)
        self.assertIn("nope.py", result["detail"])

    def test_the_flag_off_control_maps_pass_fail_and_missing_modules(self) -> None:
        def failing() -> None:
            raise AssertionError("tables differ")

        def missing() -> None:
            raise ImportError("no torch")

        self.assertEqual(cell.flag_off_control(lambda: None)["verdict"], cell.PASS)
        failed = cell.flag_off_control(failing)
        self.assertEqual(failed["verdict"], cell.FAIL)
        self.assertIn("tables differ", failed["detail"])
        self.assertEqual(cell.flag_off_control(missing)["verdict"], cell.NOT_MEASURED)

    def test_source_controls_return_the_three_in_order(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            overlay, fixtures = Path(tmp, "overlay"), Path(tmp, "fixtures")
            for name in ("glm53_mixed_graph.py", "glm53_mixed_step.py"):
                self.write(overlay, name, "def f():\n    return 1\n")
                self.write(fixtures / "orig", name, "def f():\n    return 1\n")
            results = cell.SourceControls(overlay, fixtures, lambda: None).run()
        self.assertEqual([item["control"] for item in results], list(cell.CONTROL_NAMES))
        self.assertEqual([item["verdict"] for item in results], [cell.RECORDED, cell.RECORDED, cell.PASS])


class Seam(unittest.TestCase):
    def test_a_missing_seam_flag_is_listed_by_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            Path(tmp, "glm53_mixed_graph.py").write_text("X = 1\n")
            Path(tmp, "glm53_mixed_step.py").write_text(f"FLAG = '{cell.SEPARATE_FLAG}'\n")
            missing = cell.seam_missing(Path(tmp))
        self.assertEqual(missing, ["glm53_mixed_graph.py"])

    def test_a_missing_overlay_file_counts_as_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(cell.seam_missing(Path(tmp)), ["glm53_mixed_graph.py", "glm53_mixed_step.py"])


class Reports(unittest.TestCase):
    def test_a_fail_report_names_the_pair_and_step_and_ends_with_the_verdict_line(self) -> None:
        bad = cell.ArmRun(readings(TOKENS[:5] + [1] + TOKENS[6:]))
        text = cell.render_report(cell.run_cell(cell.FakeRunner(good_runs(**{"separate-step": bad})), FakeControls()))
        self.assertIn("separate-vs-same-step", text)
        self.assertIn("decode step 5", text)
        self.assertEqual(text.splitlines()[-1], "DUAL_SERIAL: FAIL")

    def test_fake_modes_exit_zero_only_for_equal(self) -> None:
        codes = {}
        for mode in ("equal", "unequal", "empty"):
            with redirect_stdout(StringIO()) as out:
                codes[mode] = (cell.main(["fake", "--mode", mode]), out.getvalue().splitlines()[-1])
        self.assertEqual(codes, {"equal": (0, "DUAL_SERIAL: PASS"), "unequal": (1, "DUAL_SERIAL: FAIL"), "empty": (1, "DUAL_SERIAL: NOT_MEASURED")})


class Drive(unittest.TestCase):
    def namespace(self, out: Path, arm: str) -> mock.Mock:
        return mock.Mock(port=18440, arm=arm, out=out, salt="S")

    def run_drive(self, arm: str, fetch: mock.Mock | None = None) -> tuple[int, dict, mock.Mock]:
        send = mock.Mock(return_value=reply(count=1))
        fetch = fetch or mock.Mock(return_value=list(TOKENS))
        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(cell, "PRESSURE_DELAY_S", 0):
            out = Path(tmp, "t.jsonl")
            code = cell.drive_arm(self.namespace(out, arm), send, fetch)
            row = json.loads(out.read_text())
        return code, row, send

    def test_a_pressure_arm_sends_the_prompts_while_the_subject_is_recorded(self) -> None:
        code, row, send = self.run_drive("dpdp")
        self.assertEqual((code, row["tokens"], row["error"], send.call_count), (0, TOKENS, "", cell.PRESSURE_PROMPTS))

    def test_the_isolated_arm_sends_nothing_but_the_subject(self) -> None:
        code, row, send = self.run_drive("isolated")
        self.assertEqual((code, row["tokens"], send.call_count), (0, TOKENS, 0))

    def test_the_subject_request_is_identical_in_every_arm(self) -> None:
        calls = []
        for arm in ("same-step", "isolated"):
            self.run_drive(arm, mock.Mock(side_effect=lambda *a, **k: calls.append((a, k)) or list(TOKENS)))
        self.assertEqual(calls[0], calls[1])

    def test_a_failed_pressure_prompt_is_recorded_as_an_error(self) -> None:
        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(cell, "PRESSURE_DELAY_S", 0):
            out = Path(tmp, "t.jsonl")
            code = cell.drive_arm(self.namespace(out, "dpdp"), mock.Mock(side_effect=OSError("reset")), mock.Mock(return_value=list(TOKENS)))
            row = json.loads(out.read_text())
        self.assertEqual(code, 1)
        self.assertIn("OSError", row["error"])

    def test_a_failed_subject_leaves_no_tokens_and_the_reason(self) -> None:
        code, row, _ = self.run_drive("same-step", mock.Mock(side_effect=TimeoutError("slow")))
        self.assertEqual((code, row["tokens"]), (1, []))
        self.assertIn("subject TimeoutError", row["error"])


class Execute(unittest.TestCase):
    def seamed_overlay(self, tmp: Path) -> Path:
        for name in ("glm53_mixed_graph.py", "glm53_mixed_step.py"):
            Path(tmp, name).write_text(f"FLAG = '{cell.SEPARATE_FLAG}'\n")
        return tmp

    def run_cell(self, tmp: Path, tokens_for: dict[str, list[int]], controls: FakeControls) -> tuple[int, list[list[str]], dict]:
        calls: list[list[str]] = []

        def fake_run(argv: list[str], **kwargs: object) -> mock.Mock:
            calls.append(argv)
            if argv[2] == "drive":
                arm, out = argv[argv.index("--arm") + 1], Path(argv[argv.index("--out") + 1])
                c1_cell.append_row(out, {"arm": arm, "tokens": tokens_for[arm], "error": ""})
            return mock.Mock(returncode=0, stderr="")

        with mock.patch("subprocess.run", side_effect=fake_run), mock.patch.object(cell.k1_cell, "LANE", tmp), \
                mock.patch.object(cell, "OVERLAY", self.seamed_overlay(tmp)), mock.patch.object(cell, "SourceControls", return_value=controls), \
                redirect_stdout(StringIO()), redirect_stderr(StringIO()):
            code = cell.main(run_args(tmp, "--execute"))
        return code, calls, json.loads((tmp / "out" / "dual-serial-T.json").read_text())

    def test_four_boots_each_up_drive_down_and_equal_tokens_pass(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            code, calls, result = self.run_cell(Path(tmp), {arm.name: list(TOKENS) for arm in cell.ARMS}, FakeControls())
        self.assertEqual((code, result["verdict"]), (0, "PASS"))
        self.assertEqual([call[2] for call in calls], ["up", "drive", "down"] * 4)

    def test_only_the_separate_arm_boots_with_the_seam_on(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            _, calls, _ = self.run_cell(Path(tmp), {arm.name: list(TOKENS) for arm in cell.ARMS}, FakeControls())
        ups = [" ".join(call) for call in calls if call[2] == "up"]
        self.assertEqual([f"{cell.SEPARATE_FLAG}=1" in up for up in ups], [False, True, False, False])

    def test_an_unequal_arm_fails_and_the_result_file_names_the_step(self) -> None:
        tokens = {arm.name: list(TOKENS) for arm in cell.ARMS} | {"separate-step": TOKENS[:3] + [1]}
        with tempfile.TemporaryDirectory() as tmp:
            code, _, result = self.run_cell(Path(tmp), tokens, FakeControls())
        self.assertEqual((code, result["verdict"]), (1, "FAIL"))
        self.assertEqual(result["pairs"][0]["first_difference"], "decode step 3")

    def test_a_missing_seam_refuses_before_any_boot_and_names_the_patch(self) -> None:
        with tempfile.TemporaryDirectory() as tmp, mock.patch("subprocess.run", side_effect=AssertionError("booted")), \
                mock.patch.object(cell, "OVERLAY", Path(tmp)), self.assertRaises(SystemExit) as caught:
            cell.main(run_args(Path(tmp), "--execute"))
        self.assertIn(cell.SEPARATE_FLAG, str(caught.exception))
        self.assertIn("glm53_mixed_graph.py", str(caught.exception))

    def test_a_used_output_folder_is_refused(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            Path(tmp, "out").mkdir()
            Path(tmp, "out", "old.json").write_text("{}")
            with mock.patch.object(cell, "OVERLAY", self.seamed_overlay(Path(tmp))), self.assertRaises(SystemExit) as caught:
                cell.main(run_args(Path(tmp), "--execute"))
        self.assertIn("one output folder per run", str(caught.exception))


class DryRun(unittest.TestCase):
    def plan(self, tmp: Path) -> dict:
        with mock.patch("subprocess.run", side_effect=AssertionError("touched")), redirect_stdout(StringIO()) as out:
            self.assertEqual(cell.main(run_args(tmp)), 0)
        return json.loads(out.getvalue())

    def test_the_default_prints_every_boot_and_runs_nothing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            plan = self.plan(Path(tmp))
            self.assertFalse(Path(tmp, "out").exists())
        self.assertEqual((plan["dry_run"], plan["touches_cards"]), (True, False))
        self.assertEqual([arm["arm"] for arm in plan["arms"]], [arm.name for arm in cell.ARMS])
        self.assertEqual([step["label"] for step in plan["arms"][0]["steps"]], ["up", "drive", "down"])
        self.assertEqual(plan["controls"], list(cell.CONTROL_NAMES))

    def test_the_plan_reports_the_seam_state(self) -> None:
        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(cell, "OVERLAY", Path(tmp)):
            plan = self.plan(Path(tmp))
        self.assertEqual(plan["seam_missing"], ["glm53_mixed_graph.py", "glm53_mixed_step.py"])

    def test_the_drive_step_is_bounded(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            drive = self.plan(Path(tmp))["arms"][0]["steps"][1]
        self.assertGreater(drive["timeout_s"], c1_cell.k1_cell.long_timeout_s(cell.PRESSURE_TOKENS))
        self.assertLess(drive["timeout_s"], 4 * 3600)


if __name__ == "__main__":
    unittest.main()
