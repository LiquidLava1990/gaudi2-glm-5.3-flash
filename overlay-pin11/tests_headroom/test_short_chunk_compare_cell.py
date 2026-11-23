"""Short final chunk check (keyed vs native): prompts whose last chunk is shorter than the graph must decode the same tokens in both boots.
Fake runner, fake HTTP and a fake docker log only; nothing here touches a card."""

from __future__ import annotations

import importlib.util
import json
import os
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
import short_chunk_compare_cell as cell  # noqa: E402
from test_c1_cell import write_env  # noqa: E402
from test_c1_score import RANKS, counters, mock_log, who  # noqa: E402
from test_prompt_keys import KeyCase  # noqa: E402

LANE_DIR = Path(os.environ.get("LANE_DIR") or Path(__file__).resolve().parents[2])
ON_HOST = unittest.skipUnless((LANE_DIR / "graphs-c1m.env").is_file(), "graphs-c1*.env sit beside the overlay on the host (LANE_DIR)")
TOKENS = list(range(1000, 1064))
SINGLE = cell.cases_for(False)
SHARED = cell.cases_for(True)


def outputs(cases: list, tokens: list[int] | None = None) -> dict[str, list[int]]:
    return {case.name: list(TOKENS if tokens is None else tokens) for case in cases}


def runner(cases: list = SINGLE, native: dict | None = None, keyed: dict | None = None, error: str = "") -> cell.FakeRunner:
    return cell.FakeRunner({"native": cell.ArmRun(outputs(cases) if native is None else native),
                            "keyed": cell.ArmRun(outputs(cases) if keyed is None else keyed, error)})


def run_args(tmp: Path, *extra: str) -> list[str]:
    return ["run", "--image", "leaf", "--port", "18440", "--out-dir", str(tmp / "out"), "--env-file", str(write_env(tmp)),
            "--mount", f"{tmp}/trace:/k1trace", "--stamp", "T", *extra]


class Cases(unittest.TestCase):
    def test_the_final_chunks_are_1_11_700_and_1024_rows_after_two_full_chunks(self) -> None:
        self.assertEqual([case.final_rows for case in SINGLE], [1, 11, 700, 1024])
        self.assertEqual([case.chunks for case in SINGLE], [[1024, 1024, 1], [1024, 1024, 11], [1024, 1024, 700], [1024, 1024, 1024]])
        self.assertEqual([case.tokens for case in SINGLE], [2049, 2059, 2748, 3072])

    def test_each_prompt_has_its_length_and_no_prefix_in_common_with_another(self) -> None:
        prompts = [cell.case_prompt(case) for case in SHARED]
        self.assertEqual([len(prompt) for prompt in prompts], [case.tokens for case in SHARED])
        self.assertEqual(len({tuple(prompt[:128]) for prompt in prompts}), len(prompts))  # a prefix hit would skip a chunk

    def test_a_prompt_is_the_same_on_every_call(self) -> None:
        self.assertEqual(cell.case_prompt(SINGLE[2]), cell.case_prompt(SINGLE[2]))

    def test_the_shared_list_adds_a_wave_of_four_short_finals_that_fit_the_256_row_key(self) -> None:
        self.assertEqual([case.name for case in SHARED[:4]], [case.name for case in SINGLE])
        wave = [case for case in SHARED if case.wave == 1]
        self.assertEqual(len(wave), 4)
        self.assertLessEqual(max(case.final_rows for case in wave), 256)
        self.assertEqual(len({case.name for case in SHARED}), len(SHARED))

    def test_waves_group_the_cases_for_the_drive(self) -> None:
        self.assertEqual([len(wave) for wave in cell.case_waves(False)], [4])
        self.assertEqual([len(wave) for wave in cell.case_waves(True)], [4, 4])


class Comparing(unittest.TestCase):
    def compare(self, reference: list[int], other: list[int], case: cell.Case = SINGLE[0]) -> tuple[str, str, int | None]:
        return cell.compare_case(case, reference, other)

    def test_equal_tokens_pass(self) -> None:
        self.assertEqual(self.compare(TOKENS, list(TOKENS))[0], cell.PASS)

    def test_a_difference_at_the_last_token_is_a_fail_that_names_prompt_chunk_and_token(self) -> None:
        other = TOKENS[:-1] + [TOKENS[-1] + 1]
        verdict, detail, index = self.compare(TOKENS, other, SINGLE[2])
        self.assertEqual((verdict, index), (cell.FAIL, 63))
        for named in (SINGLE[2].name, "final chunk 700 rows", "token 63", "1063", "1064"):
            self.assertIn(named, detail)

    def test_a_difference_at_the_first_token_is_a_fail(self) -> None:
        self.assertEqual(self.compare(TOKENS, [7] + TOKENS[1:])[::2], (cell.FAIL, 0))

    def test_a_difference_in_a_run_that_stopped_early_is_still_a_fail(self) -> None:
        self.assertEqual(self.compare(TOKENS, TOKENS[:5] + [7])[::2], (cell.FAIL, 5))

    def test_fewer_than_64_identical_tokens_are_not_measured(self) -> None:
        verdict, detail, _ = self.compare(TOKENS[:63], TOKENS[:63])
        self.assertEqual(verdict, cell.NOT_MEASURED)
        self.assertIn("63", detail)
        self.assertEqual(self.compare(TOKENS, TOKENS[:40])[0], cell.NOT_MEASURED)

    def test_no_tokens_is_not_measured_never_equal(self) -> None:
        for reference, other in (([], []), (TOKENS, []), ([], TOKENS)):
            self.assertEqual(self.compare(reference, other)[0], cell.NOT_MEASURED)

    def test_two_runs_past_64_tokens_that_part_in_length_fail(self) -> None:
        self.assertEqual(self.compare(TOKENS, TOKENS + [5])[::2], (cell.FAIL, 64))


class Runner(unittest.TestCase):
    def test_equal_outputs_pass_every_case(self) -> None:
        result = cell.compare_arms(runner(), SINGLE)
        self.assertEqual(result["verdict"], cell.PASS)
        self.assertEqual([item["verdict"] for item in result["cases"]], [cell.PASS] * 4)

    def test_one_case_that_differs_fails_the_run_and_is_named(self) -> None:
        keyed = outputs(SINGLE)
        keyed[SINGLE[1].name] = TOKENS[:-1] + [1]
        result = cell.compare_arms(runner(keyed=keyed), SINGLE)
        bad = [item for item in result["cases"] if item["verdict"] == cell.FAIL]
        self.assertEqual((result["verdict"], [(item["case"], item["final_rows"], item["first_difference"]) for item in bad]),
                         (cell.FAIL, [(SINGLE[1].name, 11, 63)]))

    def test_a_case_with_no_output_is_not_measured(self) -> None:
        keyed = outputs(SINGLE[:3])
        result = cell.compare_arms(runner(keyed=keyed), SINGLE)
        self.assertEqual(result["verdict"], cell.NOT_MEASURED)
        self.assertEqual([item["verdict"] for item in result["cases"]], [cell.PASS] * 3 + [cell.NOT_MEASURED])

    def test_an_arm_error_makes_equal_outputs_not_measured_with_its_reason(self) -> None:
        result = cell.compare_arms(runner(error="rank 2: 1 recapture(s) in the keyed arm"), SINGLE)
        self.assertEqual(result["verdict"], cell.NOT_MEASURED)
        self.assertIn("recapture", result["cases"][0]["detail"])

    def test_a_difference_still_fails_when_the_arm_also_errored(self) -> None:
        keyed = outputs(SINGLE)
        keyed[SINGLE[0].name] = [9] + TOKENS[1:]
        self.assertEqual(cell.compare_arms(runner(keyed=keyed, error="gap"), SINGLE)["verdict"], cell.FAIL)

    def test_the_result_says_it_compared_token_ids_only(self) -> None:
        result = cell.compare_arms(runner(), SINGLE)
        self.assertEqual(result["compared"], "token ids only")
        self.assertIn("hidden", result["hidden"])

    def test_the_fake_command_exits_zero_only_for_equal(self) -> None:
        codes = {}
        for mode in ("equal", "unequal", "last-token", "empty", "recapture"):
            with redirect_stdout(StringIO()) as out:
                codes[mode] = cell.main(["fake", "--mode", mode])
            self.assertIn("SHORT_CHUNK_COMPARE:", out.getvalue())
        self.assertEqual(codes, {"equal": 0, "unequal": 1, "last-token": 1, "empty": 1, "recapture": 1})

    def test_the_fake_last_token_mode_is_a_fail_not_a_pass(self) -> None:
        with redirect_stdout(StringIO()) as out:
            cell.main(["fake", "--mode", "last-token"])
        self.assertIn("SHORT_CHUNK_COMPARE: FAIL", out.getvalue())
        self.assertIn("token 63", out.getvalue())


class Env(unittest.TestCase):
    def base(self, text: str | None = None) -> dict[str, str]:
        with tempfile.TemporaryDirectory() as tmp:
            return c1_cell.load_env(write_env(Path(tmp)) if text is None else write_env(Path(tmp), text))

    def test_the_keyed_arm_keeps_the_keys_and_graphs_only_and_drops_the_ballast(self) -> None:
        env = cell.arm_env(self.base(), cell.ARMS[1])
        self.assertEqual(env["VLLM_GAUDI_MIXED_PROMPT_KEYS"], c1_cell.SERVING_KEYS)
        self.assertEqual(env["HPU_GLM53_GRAPHS_ONLY"], "1")
        self.assertNotIn("HPU_GLM53_K1_BALLAST_MIB", env)

    def test_the_native_arm_has_no_keys_no_graphs_only_and_nothing_that_needs_keys(self) -> None:
        env = cell.arm_env(self.base(), cell.ARMS[0])
        for name in ("VLLM_GAUDI_MIXED_PROMPT_KEYS", "HPU_GLM53_GRAPHS_ONLY", "VLLM_GAUDI_MIXED_DECODE_WIDTHS", "HPU_GLM53_DECODE_WIDTH_PAD",
                     "HPU_GLM53_DECODE_PAD_CHECK_WIDTHS", "HPU_GLM53_R589_CAPTURE_MARKS", "HPU_GLM53_K1_BALLAST_MIB", *c1_cell.SHARED_ONLY_ENV):
            self.assertNotIn(name, env)

    @ON_HOST
    def test_both_arms_cut_the_prompt_into_the_same_1024_row_chunks_and_share_the_rest(self) -> None:
        base = c1_cell.load_env(LANE_DIR / "graphs-c1.env")
        native, keyed = (cell.arm_env(base, arm) for arm in cell.ARMS)
        for env in (native, keyed):
            self.assertEqual((env["VLLM_GAUDI_MIXED_PREFILL_TOKENS"], env["VLLM_ADAPTIVE_LONG_PREFILL_SINGLE_TOKENS"]), ("1024", "1024"))
        self.assertEqual(native["HPU_GLM53_PRIVATE_PROMPT_TOPK"], keyed["HPU_GLM53_PRIVATE_PROMPT_TOPK"])

    def test_the_arms_are_native_first_as_the_reference(self) -> None:
        self.assertEqual([(arm.name, arm.keyed) for arm in cell.ARMS], [("native", False), ("keyed", True)])

    @ON_HOST
    def test_the_native_arm_is_built_from_the_shared_file_without_its_four_prompt_step(self) -> None:
        base = c1_cell.load_env(LANE_DIR / "graphs-c1m.env", c1_cell.SHARED_VARIANT)
        native, keyed = (cell.arm_env(base, arm) for arm in cell.ARMS)
        self.assertEqual(keyed["VLLM_GAUDI_MIXED_PROMPT_BATCH_KEYS"], "4:256:1024,4:1024:1024")
        self.assertEqual(native["VLLM_GAUDI_MIXED_PREFILL_TOKENS"], "1024")
        self.assertNotIn("VLLM_GAUDI_MIXED_PROMPT_BATCH_KEYS", native)


IN_IMAGE = unittest.skipUnless(all(importlib.util.find_spec(name) for name in ("mixed_recurrent", "glm53_mixed_tail_fuse")),
                               "the boot flag checks load the image's kernels: run_tests.sh runs this class, the host skips it")


@ON_HOST
@IN_IMAGE
class BootChecks(KeyCase):
    """The boot's own flag checks accept the arms they should: a native arm that tripped them would never reach a request."""
    keys = ""

    def refuses(self, env: dict[str, str]) -> str:
        with mock.patch.dict(os.environ, env, clear=True):
            try:
                self.graph.check_prompt_key_flags()
            except ValueError as error:
                return str(error)
        return ""

    def test_both_arms_of_the_single_prompt_file_and_the_native_arm_of_the_shared_file_pass(self) -> None:
        for variant, arm in (("c1", cell.ARMS[0]), ("c1", cell.ARMS[1]), ("c1m", cell.ARMS[0])):
            base = c1_cell.load_env(LANE_DIR / f"graphs-{variant}.env", variant)
            self.assertEqual(self.refuses(cell.arm_env(base, arm)), "", f"{variant} {arm.name}")

    def test_the_native_arm_does_not_pass_by_accident_of_an_empty_check(self) -> None:
        keyed = cell.arm_env(c1_cell.load_env(LANE_DIR / "graphs-c1.env"), cell.ARMS[1])
        self.assertNotEqual(self.refuses({**keyed, "VLLM_GAUDI_MIXED_PROMPT_KEYS": ""}), "")


def keyed_log(*, recaptures: int = 0, fallbacks: int = 0, tokens: int | None = None, shared_replays: bool = False) -> list[str]:
    """Boot capture steps, then live prompt steps and decodes; counters are cumulative (LOG_EVERY=1)."""
    total = sum(case.tokens for case in SINGLE) if tokens is None else tokens
    lines = []
    for rank in RANKS:
        boot = {"mixed_steps": 2, "boot_steps": 2, "captures": 2, "prompt_tokens": 0}
        live = {**boot, "mixed_steps": 3, "live_steps": 1, "graph_replays": 1, "prompt_tokens": total, "captures": 2 + recaptures,
                "native_fallbacks": fallbacks}
        lines += [counters(rank, **boot), counters(rank, **live)]
    return lines + (shared_census() if shared_replays else [])


def shared_census() -> list[str]:
    body = {"path": "graph", "kind": "replay", "rows": 1024, "context": 1024, "prompts": 4, "pid": 1, "start_in_use": 1, "end_in_use": 2,
            "max_in_use": 3, "limit": 4, "between_peak": 2, "peak_reset_failed": False, "n": 1, "sites": []}
    return [f"{who(rank)}GLM53_PROMPT_CENSUS {json.dumps(body)}" for rank in RANKS]


class LogGap(unittest.TestCase):
    def gap(self, lines: list[str], shared: bool = False) -> str:
        return cell.keyed_log_gap(lines, SHARED if shared else SINGLE, shared)

    def test_a_clean_keyed_log_has_no_gap(self) -> None:
        self.assertEqual(self.gap(keyed_log()), "")

    def test_a_capture_after_the_boot_is_a_recapture_and_names_the_rank(self) -> None:
        gap = self.gap(keyed_log(recaptures=1))
        self.assertIn("recapture", gap)
        self.assertIn("rank 0", gap)

    def test_a_fallback_after_the_boot_means_a_chunk_did_not_replay(self) -> None:
        self.assertIn("fallback", self.gap(keyed_log(fallbacks=1)))

    def test_fewer_prompt_rows_than_the_prompts_hold_means_a_chunk_never_ran(self) -> None:
        gap = self.gap(keyed_log(tokens=sum(case.tokens for case in SINGLE) - 1))
        self.assertIn("prompt tokens", gap)

    def test_no_counters_lines_is_a_gap_not_a_pass(self) -> None:
        self.assertIn("LOG_EVERY", self.gap(mock_log(cold=False)))

    def test_the_shared_run_needs_the_shared_key_to_have_replayed(self) -> None:
        total = sum(case.tokens for case in SHARED)
        self.assertIn("4:1024:1024", self.gap(keyed_log(tokens=total), shared=True))
        self.assertEqual(self.gap(keyed_log(tokens=total, shared_replays=True), shared=True), "")


class Drive(unittest.TestCase):
    def namespace(self, out: Path, shared: bool = False) -> mock.Mock:
        return mock.Mock(port=18440, arm="keyed", out=out, salt="S", shared=shared)

    def run_drive(self, shared: bool, fetch: mock.Mock) -> tuple[int, dict]:
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp, "t.jsonl")
            code = cell.drive_arm(self.namespace(out, shared), fetch)
            return code, json.loads(out.read_text())

    def test_every_case_is_sent_once_and_its_tokens_recorded_by_name(self) -> None:
        fetch = mock.Mock(return_value=list(TOKENS))
        code, row = self.run_drive(False, fetch)
        self.assertEqual((code, fetch.call_count, row["arm"], row["error"]), (0, 4, "keyed", ""))
        self.assertEqual(row["outputs"], outputs(SINGLE))

    def test_the_request_is_greedy_64_tokens_on_a_salt_of_its_own(self) -> None:
        fetch = mock.Mock(return_value=list(TOKENS))
        self.run_drive(False, fetch)
        salts = {call.kwargs["salt"] for call in fetch.call_args_list}
        self.assertEqual(len(salts), 4)
        self.assertEqual({call.args[2] for call in fetch.call_args_list}, {cell.OUTPUT_TOKENS})
        self.assertEqual([len(call.args[1]) for call in fetch.call_args_list], [case.tokens for case in SINGLE])

    def test_the_prompts_and_salts_are_the_same_for_every_arm(self) -> None:
        calls = []
        for arm in ("native", "keyed"):
            namespace = self.namespace(Path(tempfile.mkdtemp(), f"{arm}.jsonl"))
            namespace.arm = arm
            cell.drive_arm(namespace, lambda *a, **k: calls.append((a, k)) or list(TOKENS))
        self.assertEqual(calls[:4], calls[4:])

    def test_the_shared_drive_sends_both_waves(self) -> None:
        fetch = mock.Mock(return_value=list(TOKENS))
        code, row = self.run_drive(True, fetch)
        self.assertEqual((code, fetch.call_count, sorted(row["outputs"])), (0, 8, sorted(case.name for case in SHARED)))

    def test_a_failed_request_is_recorded_and_the_drive_exits_nonzero(self) -> None:
        fetch = mock.Mock(side_effect=OSError("reset"))
        code, row = self.run_drive(False, fetch)
        self.assertEqual(code, 1)
        self.assertIn("OSError", row["error"])

    def test_the_http_payload_carries_the_fixed_seed_and_temperature_zero(self) -> None:
        sent = {}

        def urlopen(request: object, timeout: int) -> StringIO:
            sent.update(json.loads(request.data))
            return StringIO(json.dumps({"choices": [{"logprobs": {"tokens": ["token_id:5", "token_id:6"]}}]}))

        with mock.patch.object(cell, "urlopen", urlopen), mock.patch.dict(sys.modules, {"probe_flash_c16_starvation": mock.Mock(MODEL="m")}):
            self.assertEqual(cell.request_token_ids(18440, [1, 2], 2, timeout=5, salt="s"), [5, 6])
        self.assertEqual((sent["seed"], sent["temperature"], sent["max_tokens"], sent["cache_salt"]), (cell.SEED, 0, 2, "s"))


class Execute(unittest.TestCase):
    def run_cell(self, tmp: Path, outputs_for: dict[str, dict], *extra: str, log: list[str] | None = None) -> tuple[int, list[list[str]], dict]:
        calls: list[list[str]] = []

        def fake_run(argv: list[str], **kwargs: object) -> mock.Mock:
            calls.append(argv)
            if argv[2] == "drive":
                arm, out = argv[argv.index("--arm") + 1], Path(argv[argv.index("--out") + 1])
                with out.open("a") as handle:
                    handle.write(json.dumps({"arm": arm, "outputs": outputs_for[arm], "error": ""}) + "\n")
            return mock.Mock(returncode=0, stderr="")

        text = "\n".join(keyed_log() if log is None else log) + "\n"
        with mock.patch("subprocess.run", side_effect=fake_run), mock.patch.object(cell.k1_cell, "LANE", tmp), \
                redirect_stdout(StringIO()), redirect_stderr(StringIO()):
            Path(tmp, "out").mkdir()
            Path(tmp, f"{cell.boot_name('keyed', False)}.log").write_text(text)
            code = cell.main(run_args(tmp, "--execute", *extra))
        return code, calls, json.loads((tmp / "out" / "short-chunk-T.json").read_text())

    def good(self) -> dict[str, dict]:
        return {"native": outputs(SINGLE), "keyed": outputs(SINGLE)}

    def test_two_boots_each_up_drive_down_and_equal_tokens_pass(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            code, calls, result = self.run_cell(Path(tmp), self.good())
        self.assertEqual((code, result["verdict"]), (0, "PASS"))
        self.assertEqual([call[2] for call in calls], ["up", "drive", "down"] * 2)

    def test_only_the_native_boot_lacks_keys_and_graphs_only(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            _, calls, _ = self.run_cell(Path(tmp), self.good())
        native, keyed = (" ".join(call) for call in calls if call[2] == "up")
        self.assertNotIn("VLLM_GAUDI_MIXED_PROMPT_KEYS=", native)
        self.assertNotIn("HPU_GLM53_GRAPHS_ONLY=", native)
        self.assertIn("VLLM_GAUDI_MIXED_PROMPT_KEYS=1024:8192", keyed)
        self.assertIn("HPU_GLM53_GRAPHS_ONLY=1", keyed)

    def test_a_token_difference_at_the_last_token_fails_the_run(self) -> None:
        arms = self.good()
        arms["keyed"][SINGLE[3].name] = TOKENS[:-1] + [0]
        with tempfile.TemporaryDirectory() as tmp:
            code, _, result = self.run_cell(Path(tmp), arms)
        self.assertEqual((code, result["verdict"], result["cases"][3]["first_difference"]), (1, "FAIL", 63))

    def test_a_recapture_in_the_keyed_log_is_not_measured(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            code, _, result = self.run_cell(Path(tmp), self.good(), log=keyed_log(recaptures=2))
        self.assertEqual((code, result["verdict"]), (1, "NOT_MEASURED"))
        self.assertIn("recapture", result["cases"][0]["detail"])

    def test_a_missing_keyed_log_is_not_measured(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            code, _, result = self.run_cell(Path(tmp), self.good(), log=[])
        self.assertEqual((code, result["verdict"]), (1, "NOT_MEASURED"))

    def test_a_failed_boot_is_not_measured_and_down_still_runs(self) -> None:
        runs = []

        def fake_run(argv: list[str], **kwargs: object) -> mock.Mock:
            runs.append(argv[2])
            return mock.Mock(returncode=1 if argv[2] == "up" else 0, stderr="boom")

        with tempfile.TemporaryDirectory() as tmp, mock.patch("subprocess.run", side_effect=fake_run), \
                redirect_stdout(StringIO()), redirect_stderr(StringIO()):
            Path(tmp, "out").mkdir()
            code = cell.main(run_args(Path(tmp), "--execute"))
            result = json.loads(Path(tmp, "out", "short-chunk-T.json").read_text())
        self.assertEqual((code, result["verdict"]), (1, "NOT_MEASURED"))
        self.assertNotIn("drive", runs)
        self.assertEqual(runs.count("down"), 2)

    def test_an_output_folder_that_already_holds_rows_is_refused(self) -> None:
        with tempfile.TemporaryDirectory() as tmp, mock.patch("subprocess.run", side_effect=AssertionError("touched")), \
                redirect_stdout(StringIO()), redirect_stderr(StringIO()), self.assertRaises(SystemExit) as caught:
            out = Path(tmp, "out")
            out.mkdir()
            (out / cell.ROWS_NAME).write_text("{}\n")
            cell.main(run_args(Path(tmp), "--execute"))
        self.assertIn("new --out-dir", str(caught.exception))


class DryRun(unittest.TestCase):
    def plan(self, *extra: str) -> tuple[int, dict, mock.Mock]:
        with tempfile.TemporaryDirectory() as tmp, mock.patch("subprocess.run", side_effect=AssertionError("touched")) as run:
            out = StringIO()
            with redirect_stdout(out):
                code = cell.main(run_args(Path(tmp), *extra))
            self.assertFalse(Path(tmp, "out").exists())
        return code, json.loads(out.getvalue()), run

    def test_the_default_prints_both_boots_and_runs_nothing(self) -> None:
        code, plan, run = self.plan()
        self.assertEqual((code, run.call_count, plan["dry_run"], plan["touches_cards"]), (0, 0, True, False))
        self.assertEqual([arm["arm"] for arm in plan["arms"]], ["native", "keyed"])
        self.assertEqual([step["label"] for step in plan["arms"][0]["steps"]], ["up", "drive", "down"])
        self.assertEqual(plan["compared"], "token ids only")

    def test_the_dry_run_flag_is_the_same_as_the_default(self) -> None:
        self.assertEqual(self.plan("--dry-run")[0], 0)

    def test_the_plan_lists_the_cases_with_their_chunks(self) -> None:
        plan = self.plan()[1]
        self.assertEqual([item["chunks"] for item in plan["cases"]], [case.chunks for case in SINGLE])

    def test_the_drive_step_is_bounded(self) -> None:
        drive = self.plan()[1]["arms"][0]["steps"][1]
        self.assertGreater(drive["timeout_s"], c1_cell.k1_cell.REQUEST_TIMEOUT_S)
        self.assertLess(drive["timeout_s"], 4 * 3600)

    def test_the_default_env_file_is_graphs_c1_and_the_shared_option_reads_graphs_c1m(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            single = cell.parse_args(run_args(Path(tmp)))
        self.assertEqual((single.variant, single.shared), ("c1", False))
        with tempfile.TemporaryDirectory() as tmp:
            args = cell.parse_args(["run", "--image", "leaf", "--port", "18440", "--out-dir", str(Path(tmp) / "out"), "--shared-keys"])
        self.assertEqual((args.variant, args.shared, args.env_file.name), ("c1m", True, "graphs-c1m.env"))
        self.assertEqual(cell.parse_args(["run", "--image", "leaf", "--port", "18440", "--out-dir", "o"]).env_file.name, "graphs-c1.env")

    def test_a_replica_port_is_refused(self) -> None:
        with tempfile.TemporaryDirectory() as tmp, self.assertRaises(SystemExit), redirect_stderr(StringIO()):
            cell.parse_args(["run", "--image", "leaf", "--port", "18274", "--out-dir", str(Path(tmp) / "out")])

    def test_the_drive_steps_of_the_shared_option_send_both_waves_concurrently(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            text = "\n".join(f"{key}={value}" for key, value in {**c1_cell.REQUIRED_ENV, **c1_cell.VARIANT_ENV["c1m"],
                                                                     "VLLM_GAUDI_MIXED_PROMPT_KEYS": c1_cell.SHARED_SINGLE_KEYS}.items())
            env_file = write_env(Path(tmp), text + "\nHPU_GLM53_K1_STALL_DIR=/k1trace\n", "graphs-c1m.env")
            out = StringIO()
            with redirect_stdout(out):
                cell.main(["run", "--image", "leaf", "--port", "18440", "--out-dir", str(Path(tmp) / "out"), "--env-file", str(env_file),
                           "--mount", f"{tmp}/t:/k1trace", "--shared-keys"])
        plan = json.loads(out.getvalue())
        self.assertIn("--shared", plan["arms"][1]["steps"][1]["argv"])
        self.assertEqual(len(plan["cases"]), 8)
        up = plan["arms"][0]["steps"][0]["argv"]
        self.assertEqual(up[up.index("--name") + 1], cell.boot_name("native", True))


if __name__ == "__main__":
    unittest.main()
