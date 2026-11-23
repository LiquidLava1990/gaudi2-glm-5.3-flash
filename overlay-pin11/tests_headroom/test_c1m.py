"""C1m (plan 12.3b, step 49 on a card): four cold prompts of about 4,000 tokens in a shared step against the same four one at a time on the
single 1024-row key. The census and witness name a shared key by its prompts, the scorer reads each shared key's warmed replay ms and resident
HBM beside 1:1024:1024 and compares the output tokens, and the cell and env file wire it as C1_VARIANT=c1m. CPU only; nothing here touches a card."""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import threading
import unittest
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "cells"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import c1_cell as cell  # noqa: E402
import c1_score as scorer  # noqa: E402
import k1_cell  # noqa: E402
import k1_score  # noqa: E402
import memory_cells  # noqa: E402
from test_c1_cell import ENV_TEXT, namespace, write_env  # noqa: E402
from test_c1_score import MIB, RANKS, ballast_line, cold_lines, census, DECODE_KEYS, replays, who, witness  # noqa: E402
from test_prompt_batch_keys import SharedCase  # noqa: E402
from test_prompt_keys import FakeCore, FakeHpu  # noqa: E402

PASS, FAIL, NOT_MEASURED = scorer.PASS, scorer.FAIL, scorer.NOT_MEASURED
LANE_DIR = Path(os.environ.get("LANE_DIR", "/nonexistent"))
BATCH = "4:256:1024,4:1024:1024"
M_ENV_TEXT = "\n".join(f"{key}={value}" for key, value in {**cell.REQUIRED_ENV, **cell.VARIANT_ENV["c1m"],
                                                           "VLLM_GAUDI_MIXED_PROMPT_KEYS": cell.VARIANT_KEYS["c1m"]}.items()) + "\nHPU_GLM53_K1_STALL_DIR=/k1trace\n"
SHARED_KEYS = ((4, 256, 1024), (4, 1024, 1024))
WITNESS_KEYS = "1024:8192,1024:1024,256:1024:4,1024:1024:4"


class CensusNamesTheSharedKey(SharedCase):
    """A 4:1024:1024 key and the 1:1024:1024 key share rows and width; only `prompts` tells their lines apart."""

    def setUp(self) -> None:
        super().setUp()
        import glm53_prompt_census as census_module

        self.census_module = census_module
        patches = [mock.patch.dict(os.environ, {"HPU_GLM53_PREFILL_HBM_CENSUS": "1"}),
                   mock.patch.object(census_module, "_runtime", lambda: (FakeCore(), FakeHpu()))]
        for patch in patches:
            patch.start()
            self.addCleanup(patch.stop)
        census_module.GRAPH_LOG.clear()
        self.addCleanup(census_module.GRAPH_LOG.clear)

    def bodies(self, run: object) -> list[dict]:
        with self.assertLogs("vllm.r589", "WARNING") as logs:
            run()
        return [json.loads(line.split("GLM53_PROMPT_CENSUS ", 1)[1]) for line in logs.output if "GLM53_PROMPT_CENSUS {" in line]

    def test_a_shared_key_forward_logs_its_prompts_and_a_single_key_forward_does_not(self) -> None:
        shared = self.bodies(lambda: self.boot_capture(self.cache, self.shared_batch([256, 256], [0, 0])))[0]
        single = self.bodies(lambda: self.boot_capture(self.cache, self.prompt_batch(700, 0, 1024)))[0]
        self.assertEqual((shared["prompts"], shared["rows"], shared["context"], shared["kind"]), (4, 256, 1024, "capture"))
        self.assertNotIn("prompts", single)

    def test_the_replay_of_a_shared_key_is_logged_with_its_prompts_too(self) -> None:
        self.boot_capture(self.cache, self.shared_batch([256, 256], [0, 0]))
        replay = self.bodies(lambda: self.cache.run_key(self.shared_batch([100, 200], [1024, 512])))[0]
        self.assertEqual((replay["kind"], replay["prompts"]), ("replay", 4))

    def test_throttle_and_timing_count_each_shape_by_its_prompts(self) -> None:
        module = self.census_module
        for _ in range(module.GRAPH_LOG_FIRST + 4):
            with mock.patch.object(module.LOG, "warning"):
                module.census_graph_forward(lambda: "h", 1024, 1024, lambda: 1, key_replay=True, prompts=4)()
        with mock.patch.object(module.LOG, "warning") as warning:
            module.census_graph_forward(lambda: "h", 1024, 1024, lambda: 1, key_replay=True)()
        self.assertEqual(warning.call_count, 1, "the single key's first replay still logs after sixteen of the shared key's")
        self.assertEqual(module.GRAPH_LOG[("replay", 1024, 1024)][0], 1)
        self.assertEqual(module.GRAPH_LOG[("replay", 1024, 1024, 4)][0], module.GRAPH_LOG_FIRST + 4)


class WitnessNamesTheSharedKey(SharedCase):
    def test_the_witness_lists_a_shared_key_as_rows_width_prompts(self) -> None:
        entries = {((0, 1024, 8192), "s"): None, ((0, 1024, 1024), "s"): None, ((0, 256, 1024, 4), "s"): None, ((0, 1024, 1024, 4), "s"): None}
        runner = SimpleNamespace(model=SimpleNamespace(_glm53_mixed_graph_cache=SimpleNamespace(entries=entries, counts={})))
        with mock.patch.object(self.graph, "_free_hbm", return_value=0), mock.patch.object(self.graph, "admit_capture"), \
                self.assertLogs(self.graph.LOG, "WARNING") as logs:
            self.graph._log_boot_witness(runner, (), 0, 0)
        self.assertIn("prompt_keys=1024:8192,1024:1024,256:1024:4,1024:1024:4", logs.output[0])


def shared_body(kind: str, prompts: int, rows: int, width: int, *, replay_ms: float | None = None, resident_mib: int = 1000) -> dict:
    start = 1000 * MIB
    body = {"path": "graph", "kind": kind, "rows": rows, "context": width, "pid": 100, "start_in_use": start, "end_in_use": start + resident_mib * MIB,
            "max_in_use": start + (resident_mib + 900) * MIB, "limit": 90_000 * MIB, "between_peak": start + 500 * MIB, "peak_reset_failed": False,
            "n": 1, "sites": []}
    if prompts > 1:
        body["prompts"] = prompts
    if replay_ms is not None:
        body["replay_ms"] = replay_ms
    return body


def shared_lines(rank: int, key: tuple[int, int, int], ms: float | None, resident_mib: int, count: int = 6) -> list[str]:
    """A capture and `count` replays of one shared key on a rank (the first replay awful and untimed, as the single-key helpers write them)."""
    prompts, rows, width = key
    lines = [f"{who(rank)}GLM53_PROMPT_CENSUS " + json.dumps(shared_body("capture", prompts, rows, width, resident_mib=resident_mib), sort_keys=True)]
    for index in range(count if ms is not None else 0):
        body = shared_body("replay", prompts, rows, width, replay_ms=60_000.0 if index == 0 else ms, resident_mib=resident_mib)
        lines.append(f"{who(rank)}GLM53_PROMPT_CENSUS " + json.dumps(body, sort_keys=True))
    return lines


M_MS = {(4, 256, 1024): 1500.0, (4, 1024, 1024): 2600.0}
M_RESIDENT = {(4, 256, 1024): 1700, (4, 1024, 1024): 2100}


def m_rank_lines(rank: int, *, ms: dict[tuple[int, int, int], float | None] | None = None, witnessed: bool = True) -> list[str]:
    ms = M_MS if ms is None else ms
    lines = [witness(rank, prompt_keys=WITNESS_KEYS)] if witnessed else []
    lines += [census(rank, "capture", 1024), *replays(rank, 1024, 1500.0)]
    for key in SHARED_KEYS:
        lines += shared_lines(rank, key, ms.get(key), M_RESIDENT[key])
    return lines + cold_lines(rank) + [ballast_line(rank)]


def m_log(**overrides: object) -> list[str]:
    return [line for rank in RANKS for line in m_rank_lines(rank, **overrides)]


def tokens(seed: int, count: int = 32) -> list[int]:
    return [1000 + seed * 100 + index for index in range(count)]


def m_rows(*, shared_outputs: list[list[int]] | None = None, serial_outputs: list[list[int]] | None = None, rounds: int = 3) -> list[dict]:
    good = [tokens(index) for index in range(4)]
    serial = good if serial_outputs is None else serial_outputs
    rows = [{"drive": "cold", "status": "ok"}]
    rows += [{"drive": "shared", "round": index, "kind": "long", "outputs": good if shared_outputs is None else shared_outputs} for index in range(rounds)]
    rows += [{"drive": "shared", "round": 9, "kind": "short", "status": "ok"}]
    return rows + [{"drive": "serial", "outputs": serial}]


M_ENV = {"VLLM_GAUDI_MIXED_PROMPT_KEYS": cell.VARIANT_KEYS["c1m"], "VLLM_GAUDI_MIXED_PROMPT_BATCH_KEYS": BATCH,
         "VLLM_GAUDI_MIXED_DECODE_WIDTHS": "1024,4096,8192", "HPU_GLM53_DECODE_PAD_CHECK_WIDTHS": "2048"}


def m_score(lines: list[str] | None = None, rows: list[dict] | None = None) -> dict:
    return scorer.score(m_log() if lines is None else lines, m_rows() if rows is None else rows, scorer.BootFacts(900.0),
                        scorer.Expected.from_env(M_ENV), variant="c1m")


def named(result: dict, name: str) -> dict:
    return next(item for item in result["readings"] if item["name"] == name)


class ParseRoutesSharedBodies(unittest.TestCase):
    def test_a_shared_key_never_lands_in_the_single_key_of_the_same_rows_and_width(self) -> None:
        parsed = k1_score.parse_log(m_rank_lines(0))
        data = parsed.ranks[0]
        self.assertEqual(len(data.keys[(1024, 1024)].replays), 6)
        self.assertEqual(sorted(data.shared), [(4, 256, 1024), (4, 1024, 1024)])
        self.assertEqual(len(data.shared[(4, 1024, 1024)].replays), 6)
        self.assertNotIn((256, 1024), data.keys)


class SharedReadings(unittest.TestCase):
    def test_a_clean_log_and_rows_pass_and_name_each_shared_key(self) -> None:
        result = m_score()
        self.assertEqual([(item["name"], item["verdict"]) for item in result["readings"] if item["verdict"] != PASS], [])
        names = [item["name"] for item in result["readings"]]
        for key in ("4x256x1024", "4x1024x1024"):
            self.assertIn(f"replay_ms_{key}", names)
            self.assertIn(f"resident_mib_{key}", names)
        self.assertEqual(result["verdict"], PASS)

    def test_the_replay_reading_sits_beside_the_single_1024_key(self) -> None:
        item = named(m_score(), "replay_ms_4x256x1024")
        self.assertEqual(item["value"], 1500.0)
        self.assertIn("1.00x", item["detail"])
        self.assertIn("1024:1024", item["detail"])
        self.assertIn("rows/s", item["detail"])

    def test_the_resident_reading_is_the_capture_delta_per_key(self) -> None:
        values = {key: named(m_score(), f"resident_mib_{key}")["value"] for key in ("4x256x1024", "4x1024x1024")}
        self.assertEqual(values, {"4x256x1024": 1700, "4x1024x1024": 2100})

    def test_a_shared_key_no_prompt_replayed_is_not_measured_never_a_pass(self) -> None:
        result = m_score(m_log(ms={(4, 256, 1024): None, (4, 1024, 1024): 2600.0}))
        self.assertEqual(named(result, "replay_ms_4x256x1024")["verdict"], NOT_MEASURED)
        self.assertEqual(named(result, "resident_mib_4x256x1024")["verdict"], PASS)
        self.assertEqual(named(result, "replay_ms_4x1024x1024")["verdict"], PASS)
        self.assertEqual(result["verdict"], NOT_MEASURED)

    def test_a_key_with_no_line_at_all_reads_not_measured_for_both_readings(self) -> None:
        lines = [line for line in m_log() if '"prompts": 4' not in line or '"rows": 256' not in line]
        result = m_score(lines)
        self.assertEqual([named(result, name)["verdict"] for name in ("replay_ms_4x256x1024", "resident_mib_4x256x1024")], [NOT_MEASURED] * 2)

    def test_one_rank_short_of_samples_is_not_measured(self) -> None:
        lines = [line for rank in RANKS for line in m_rank_lines(rank, ms=M_MS if rank != 2 else {(4, 256, 1024): 1500.0, (4, 1024, 1024): None})]
        self.assertEqual(named(m_score(lines), "replay_ms_4x1024x1024")["verdict"], NOT_MEASURED)

    def test_a_witness_without_the_shared_keys_fails(self) -> None:
        lines = [line for rank in RANKS for line in [witness(rank, prompt_keys="1024:8192,1024:1024")] + m_rank_lines(rank, witnessed=False)]
        self.assertEqual(named(m_score(lines), "boot_witness")["verdict"], FAIL)

    def test_the_serving_readings_of_the_other_variants_are_not_asked_of_this_one(self) -> None:
        names = {item["name"] for item in m_score()["readings"]}
        self.assertFalse(names & {"decode_step_ms", "decode_pad_merge", "long_drive", "replay_ms_8192"})
        self.assertIn("replay_ms_1024", names)
        self.assertEqual(scorer.Expected.from_env(M_ENV).batch_keys, frozenset({"256:1024:4", "1024:1024:4"}))


class SharedVersusSerial(unittest.TestCase):
    def test_equal_tokens_in_every_round_pass(self) -> None:
        item = named(m_score(), "shared_vs_serial")
        self.assertEqual(item["verdict"], PASS)
        self.assertIn("3 shared rounds", item["detail"])

    def test_one_differing_token_fails_naming_round_prompt_token_and_both_ids(self) -> None:
        bad = [tokens(index) for index in range(4)]
        bad[2] = bad[2][:17] + [5] + bad[2][18:]
        item = named(m_score(rows=m_rows(shared_outputs=bad)), "shared_vs_serial")
        self.assertEqual(item["verdict"], FAIL)
        for word in ("round 0", "prompt 2", "token 17", "shared 5", f"serial {tokens(2)[17]}"):
            self.assertIn(word, item["detail"])

    def test_a_shorter_output_fails_naming_where_it_ended(self) -> None:
        short = [tokens(index) for index in range(4)]
        short[1] = short[1][:20]
        item = named(m_score(rows=m_rows(shared_outputs=short)), "shared_vs_serial")
        self.assertEqual(item["verdict"], FAIL)
        self.assertIn("prompt 1", item["detail"])
        self.assertIn("token 20", item["detail"])

    def test_the_overall_verdict_is_fail(self) -> None:
        bad = [tokens(index) for index in range(4)]
        bad[0] = [9] + bad[0][1:]
        self.assertEqual(m_score(rows=m_rows(shared_outputs=bad))["verdict"], FAIL)

    def test_no_rows_or_empty_outputs_are_not_measured(self) -> None:
        cases = {"no rows": [{"drive": "cold", "status": "ok"}], "no serial": m_rows()[:-1],
                 "empty serial": m_rows(serial_outputs=[[], [], [], []]), "no shared outputs": [row for row in m_rows() if row["drive"] != "shared"],
                 "empty shared": m_rows(shared_outputs=[[], [], [], []])}
        for label, rows in cases.items():
            with self.subTest(label):
                self.assertEqual(named(m_score(rows=rows), "shared_vs_serial")["verdict"], NOT_MEASURED)

    def test_equal_outputs_with_no_replay_of_the_big_shared_key_are_not_measured(self) -> None:
        """Four prompts that never shared a step would match their serial run trivially: that is not a pass."""
        lines = m_log(ms={(4, 256, 1024): 1500.0, (4, 1024, 1024): None})
        item = named(m_score(lines), "shared_vs_serial")
        self.assertEqual(item["verdict"], NOT_MEASURED)
        self.assertIn("4:1024:1024", item["detail"])


class EnvFile(unittest.TestCase):
    def test_the_variant_loads_with_its_own_keys_and_flags(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            env = cell.load_env(write_env(Path(tmp), M_ENV_TEXT, "graphs-c1m.env"), "c1m")
        self.assertEqual(env["VLLM_GAUDI_MIXED_PROMPT_BATCH_KEYS"], BATCH)
        self.assertEqual((env["VLLM_GAUDI_MIXED_PROMPTS_PER_STEP"], env["VLLM_GAUDI_MIXED_TAIL_FIRST"], env["VLLM_PROMPT_BS_BUCKET_MAX"]), ("4", "1", "4"))

    def test_a_c1m_file_that_lost_a_shared_setting_refuses_and_names_it(self) -> None:
        for name in ("VLLM_GAUDI_MIXED_PROMPTS_PER_STEP", "VLLM_GAUDI_MIXED_PROMPT_BATCH_KEYS", "VLLM_GAUDI_MIXED_TAIL_FIRST", "VLLM_PROMPT_BS_BUCKET_MAX"):
            text = "\n".join(line for line in M_ENV_TEXT.splitlines() if not line.startswith(name + "="))
            with self.subTest(name), tempfile.TemporaryDirectory() as tmp, self.assertRaises(SystemExit) as caught:
                cell.load_env(write_env(Path(tmp), text, "graphs-c1m.env"), "c1m")
            self.assertIn(name, str(caught.exception))
            self.assertIn("graphs-c1m.env", str(caught.exception))

    def test_every_other_variant_refuses_a_shared_setting(self) -> None:
        for variant in ("c1", "c1r"):
            text = ENV_TEXT.replace(cell.SERVING_KEYS, cell.VARIANT_KEYS[variant]) + f"VLLM_GAUDI_MIXED_PROMPT_BATCH_KEYS={BATCH}\n"
            with self.subTest(variant), tempfile.TemporaryDirectory() as tmp, self.assertRaises(SystemExit) as caught:
                cell.load_env(write_env(Path(tmp), text), variant)
            self.assertIn("only c1m", str(caught.exception))

    @unittest.skipUnless((LANE_DIR / "graphs-c1m.env").is_file(), "graphs-c1m.env sits beside the overlay on the host (LANE_DIR)")
    def test_the_shipped_file_differs_from_c1_only_where_the_plan_says(self) -> None:
        base, shared = cell.load_env(LANE_DIR / "graphs-c1.env"), cell.load_env(LANE_DIR / "graphs-c1m.env", "c1m")
        changed = {key for key in shared if base.get(key) != shared[key]} | (set(base) - set(shared))
        self.assertEqual(changed, {"VLLM_GAUDI_MIXED_PROMPT_KEYS", "VLLM_GAUDI_MIXED_PROMPTS_PER_STEP", "VLLM_GAUDI_MIXED_PROMPT_BATCH_KEYS",
                                   "VLLM_GAUDI_MIXED_TAIL_FIRST", "VLLM_PROMPT_BS_BUCKET_MAX", "VLLM_GAUDI_MIXED_PREFILL_TOKENS",
                                   "VLLM_COMPLETION_FIRST_LONG_PREFILL"})

    @unittest.skipUnless((LANE_DIR / "graphs-c1m.env").is_file(), "graphs-c1m.env sits beside the overlay on the host (LANE_DIR)")
    def test_the_shipped_file_passes_the_boot_checks_of_the_step_policy(self) -> None:
        import glm53_mixed_step as step_policy

        env = cell.load_env(LANE_DIR / "graphs-c1m.env", "c1m")
        with mock.patch.dict(os.environ, env, clear=False):
            self.assertIsNone(step_policy.check_slot_flags(completion_first=False))
            self.assertIsNone(step_policy.check_shared_prompt_batch(int(env["VLLM_PROMPT_BS_BUCKET_MAX"])))


    @unittest.skipUnless((LANE_DIR / "graphs-c1m.env").is_file(), "graphs-c1m.env sits beside the overlay on the host (LANE_DIR)")
    def test_the_shipped_file_passes_the_graph_checks_graphs_only_included(self) -> None:
        from overlay_testlib import load_overlay

        graph, env = load_overlay("glm53_mixed_graph.py"), cell.load_env(LANE_DIR / "graphs-c1m.env", "c1m")
        runner = SimpleNamespace(attn_block_size=128, block_size=128, max_model_len=1_048_576)
        with mock.patch.dict(os.environ, env, clear=False):
            graph.check_prompt_key_flags()
            buckets = graph.validate_prompt_keys(runner)
        self.assertEqual(buckets, ((0, 1024, 8192), (0, 1024, 1024), (0, 1024, 1024, 4), (0, 256, 1024, 4)))


def m_args(tmp: Path, *extra: str) -> list[str]:
    env = write_env(tmp, M_ENV_TEXT, "graphs-c1m.env")
    return ["run", "--image", "leaf", "--port", "18440", "--out-dir", str(tmp / "out"), "--variant", "c1m", "--env-file", str(env),
            "--mount", f"{tmp}/trace:/k1trace", "--stamp", "T", *extra]


class Plan(unittest.TestCase):
    def test_the_steps_are_cold_shared_serial_between_up_and_down(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            args = cell.parse_args(m_args(Path(tmp)))
            steps = memory_cells.boot_steps(cell.c1_boot(args), args)
        self.assertEqual([step.label for step in steps], ["up", "cold", "shared", "serial", "down"])
        self.assertTrue(all(step.timeout_s > 0 for step in steps))

    def test_the_boot_and_drive_file_are_named_for_the_variant(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            args = cell.parse_args(m_args(Path(tmp)))
            boot = cell.c1_boot(args)
        self.assertEqual(boot.name, "claude-flash-launch-c1m")
        self.assertTrue(all("claude-flash-launch-c1m.drive.jsonl" in " ".join(step.argv) for step in boot.actions))

    def test_the_step_budgets_cover_four_prompts_in_one_step_and_one_at_a_time(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            args = cell.parse_args(m_args(Path(tmp)))
            steps = {step.label: step.timeout_s for step in memory_cells.boot_steps(cell.c1_boot(args), args)}
        self.assertGreater(steps["serial"], cell.SHARED_PROMPTS * k1_cell.long_timeout_s(cell.SHARED_PROMPT_TOKENS))
        self.assertGreater(steps["shared"], cell.SHARED_LONG_ROUNDS * k1_cell.long_timeout_s(cell.SHARED_PROMPT_TOKENS))

    def test_dry_run_names_the_variant_and_touches_nothing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp, mock.patch("subprocess.run", side_effect=AssertionError("touched")):
            out = StringIO()
            with redirect_stdout(out):
                code = cell.main(m_args(Path(tmp)))
            plan = json.loads(out.getvalue())
        self.assertEqual((code, plan["cell"], plan["touches_cards"]), (0, "c1m", False))

    def test_the_final_cell_script_names_the_variant_and_parses(self) -> None:
        script = Path(__file__).resolve().parents[1] / "cells" / "c1_final_cell.sh"
        self.assertIn("c1m", script.read_text())
        self.assertEqual(subprocess.run(["bash", "-n", str(script)], capture_output=True).returncode, 0)


def fake_fetch(outputs: dict[tuple[int, ...], list[int]] | None = None, *, barrier: threading.Barrier | None = None, calls: list | None = None) -> mock.Mock:
    """Each prompt answers with tokens that depend only on its own first token: what a lane that reads only its own rows would return."""
    lock, state = threading.Lock(), {"live": 0, "peak": 0}

    def fetch(port: int, prompt: list[int], count: int, *, timeout: int, salt: str) -> list[int]:
        with lock:
            state["live"] += 1
            state["peak"] = max(state["peak"], state["live"])
            if calls is not None:
                calls.append({"tokens": len(prompt), "first": prompt[0], "salt": salt, "count": count})
        try:
            if barrier is not None and len(prompt) == cell.SHARED_PROMPT_TOKENS:
                barrier.wait(timeout=5)
            return [prompt[0] + index for index in range(count)]
        finally:
            with lock:
                state["live"] -= 1

    result = mock.Mock(side_effect=fetch)
    result.state = state
    return result


class Drives(unittest.TestCase):
    def rows(self, path: Path) -> list[dict]:
        return [json.loads(line) for line in path.read_text().splitlines()]

    def test_the_four_prompts_are_distinct_and_the_size_the_plan_says(self) -> None:
        prompts = [cell.shared_prompt(index) for index in range(cell.SHARED_PROMPTS)]
        self.assertEqual({len(prompt) for prompt in prompts}, {cell.SHARED_PROMPT_TOKENS})
        self.assertEqual(len({tuple(prompt) for prompt in prompts}), cell.SHARED_PROMPTS)
        self.assertEqual(prompts[0], cell.shared_prompt(0))
        self.assertTrue(3500 <= cell.SHARED_PROMPT_TOKENS <= 4500)

    def test_the_shared_rounds_send_all_four_prompts_at_once(self) -> None:
        calls: list[dict] = []
        fetch = fake_fetch(barrier=threading.Barrier(cell.SHARED_PROMPTS), calls=calls)
        with tempfile.TemporaryDirectory() as tmp, redirect_stdout(StringIO()):
            out = Path(tmp, "d.jsonl")
            code = cell.drive_shared(namespace(out=out), fetch)
            rows = self.rows(out)
        self.assertEqual(code, 0)
        longs = [row for row in rows if row["kind"] == "long"]
        self.assertEqual(len(longs), cell.SHARED_LONG_ROUNDS)
        self.assertTrue(all(len(row["outputs"]) == cell.SHARED_PROMPTS and all(row["outputs"]) for row in longs))
        self.assertGreaterEqual(fetch.state["peak"], cell.SHARED_PROMPTS)

    def test_short_rounds_drive_the_small_key_with_prompts_that_fit_it(self) -> None:
        calls: list[dict] = []
        with tempfile.TemporaryDirectory() as tmp, redirect_stdout(StringIO()):
            out = Path(tmp, "d.jsonl")
            cell.drive_shared(namespace(out=out), fake_fetch(calls=calls))
            rows = self.rows(out)
        self.assertEqual(len([row for row in rows if row["kind"] == "short"]), cell.SHORT_ROUNDS)
        shorts = [call for call in calls if call["tokens"] == cell.SHORT_PROMPT_TOKENS_SHARED]
        self.assertEqual(len(shorts), cell.SHORT_ROUNDS * cell.SHARED_PROMPTS)
        self.assertLessEqual(cell.SHORT_PROMPT_TOKENS_SHARED, 256)

    def test_every_request_is_cold_each_shared_round_has_its_own_salts(self) -> None:
        calls: list[dict] = []
        with tempfile.TemporaryDirectory() as tmp, redirect_stdout(StringIO()):
            cell.drive_shared(namespace(out=Path(tmp, "d.jsonl")), fake_fetch(calls=calls))
            cell.drive_serial(namespace(out=Path(tmp, "e.jsonl")), fake_fetch(calls=calls))
        salts = [call["salt"] for call in calls]
        self.assertEqual(len(salts), len(set(salts)), "a repeated salt would be a prefix-cache hit, not a cold prefill")

    def test_the_serial_run_sends_the_same_four_prompts_one_at_a_time(self) -> None:
        calls: list[dict] = []
        fetch = fake_fetch(calls=calls)
        with tempfile.TemporaryDirectory() as tmp, redirect_stdout(StringIO()):
            out = Path(tmp, "d.jsonl")
            code = cell.drive_serial(namespace(out=out), fetch)
            rows = self.rows(out)
        self.assertEqual((code, fetch.state["peak"], len(rows)), (0, 1, 1))
        self.assertEqual([call["first"] for call in calls], [cell.shared_prompt(index)[0] for index in range(cell.SHARED_PROMPTS)])
        self.assertEqual(len(rows[0]["outputs"]), cell.SHARED_PROMPTS)

    def test_what_the_two_drives_write_compares_equal_for_a_lane_that_reads_only_its_own_rows(self) -> None:
        with tempfile.TemporaryDirectory() as tmp, redirect_stdout(StringIO()):
            out = Path(tmp, "d.jsonl")
            cell.drive_shared(namespace(out=out), fake_fetch())
            cell.drive_serial(namespace(out=out), fake_fetch())
            rows = self.rows(out)
        self.assertEqual(named(m_score(rows=rows), "shared_vs_serial")["verdict"], PASS)

    def test_a_failed_request_is_recorded_and_stops_the_drive_nonzero(self) -> None:
        fetch = mock.Mock(side_effect=OSError("connection reset"))
        with tempfile.TemporaryDirectory() as tmp, redirect_stdout(StringIO()):
            out = Path(tmp, "d.jsonl")
            self.assertEqual(cell.drive_shared(namespace(out=out), fetch), 1)
            rows = self.rows(out)
        self.assertIn("connection reset", rows[0]["error"])


class Execute(unittest.TestCase):
    def run_cell(self, tmp: Path, *, lines: list[str] | None = None, rows: list[dict] | None = None, codes: dict[str, int] | None = None) -> tuple[int, list[str], dict]:
        calls: list[str] = []
        name = "claude-flash-launch-c1m"

        def fake_run(argv: list[str], **kwargs: object) -> mock.Mock:
            calls.append(argv[2])
            if argv[2] == "up":
                (tmp / f"state-{name}.json").write_text(json.dumps({"startup_ready_s": 700.0}))
            if argv[2] == "down":
                (tmp / f"{name}.log").write_text("\n".join(lines or []) + "\n")
                (tmp / "out").mkdir(exist_ok=True)
                (tmp / "out" / f"{name}.drive.jsonl").write_text("\n".join(json.dumps(row) for row in rows or []) + "\n")
            return mock.Mock(returncode=(codes or {}).get(argv[2], 0), stderr="")

        with mock.patch("subprocess.run", side_effect=fake_run), mock.patch.object(cell, "LANE", tmp), mock.patch.object(k1_cell, "LANE", tmp), \
                redirect_stdout(StringIO()), redirect_stderr(StringIO()):
            code = cell.main(m_args(tmp, "--execute"))
        return code, calls, json.loads((tmp / "out" / "c1m-T.json").read_text())

    def test_a_clean_run_passes_and_exits_zero(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            code, calls, result = self.run_cell(Path(tmp), lines=m_log(), rows=m_rows())
        self.assertEqual((code, calls, result["verdict"]), (0, ["up", "cold", "shared", "serial", "down"], "PASS"))

    def test_a_differing_token_fails_the_run(self) -> None:
        bad = [tokens(index) for index in range(4)]
        bad[3][5] = 1
        with tempfile.TemporaryDirectory() as tmp:
            code, _, result = self.run_cell(Path(tmp), lines=m_log(), rows=m_rows(shared_outputs=bad))
        self.assertEqual((code, result["verdict"]), (1, "FAIL"))

    def test_a_drive_that_failed_keeps_the_run_from_passing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            code, calls, result = self.run_cell(Path(tmp), lines=m_log(), rows=m_rows(), codes={"serial": 1})
        self.assertEqual((code, calls[-1], result["verdict"]), (1, "down", "NOT_MEASURED"))


if __name__ == "__main__":
    unittest.main()
