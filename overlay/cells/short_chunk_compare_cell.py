"""Short final chunk check: a prompt whose last chunk is shorter than the graph must decode the same tokens whether a captured key served
its chunks (keyed) or the native path did. A keyed prompt pads that last chunk up to the key and saves its conv state from the padded graph;
the first decodes read that state. WAITING ON A CARD WINDOW: nothing here touches a card unless --execute is given; the default is a dry run.

    short_chunk_compare_cell.py run --image <leaf> --port 18440 --out-dir <new dir> [--env-file graphs-c1.env] [--mount SRC:DST]...
                                    [--shared-keys] [--execute]
    short_chunk_compare_cell.py fake --mode equal|unequal|last-token|empty|recapture [--shared-keys]   (the scoring path: no boot, no card)
    short_chunk_compare_cell.py drive --port P --arm NAME --salt S --out <jsonl> [--shared]            (the request step, run by `run`)

Two boots of one image, one through lab.py each (so its quad lock and 25 min deadline refusals apply), because graphs-only refuses native:
  native  the env file minus every key and graphs-only (and what needs keys: the decode lattice, capture marks, the shared-step flags); the
          prompt is cut into the same 1,024-row chunks. The reference.
  keyed   the env file as it is (graphs-c1.env: keys 1024:8192..1024:1024, graphs-only on), minus the fit ballast. A 2,000-3,000 token prompt
          takes the 1024:1024 key for every chunk.
Each boot sends four prompts of two full 1,024-row chunks and a final chunk of 1, 11, 700 and 1,024 rows (2,049 / 2,059 / 2,748 / 3,072
tokens): greedy, temperature 0, a fixed seed, 64 output tokens, every prompt its own tokens and its own cache salt (no prefix hit can skip a
chunk). Pass means the token ids of every prompt are identical across the two boots for at least 64 tokens. Any difference is a FAIL that
names the prompt, its final chunk length and the first differing token. No output, fewer than 64 tokens, or a keyed arm that recaptured, fell
back or ran fewer prompt rows than were sent (read from the keyed container log) is NOT_MEASURED, never PASS.

--shared-keys (plan step 49, graphs-c1m.env) boots the shared key 4:256:1024 as well and sends the prompts four at a time: the four above, then
four whose final chunks are 1, 11, 100 and 256 rows. The native arm is the same file minus the four-prompt step, one prompt per step. The keyed
log must show a replay of that key with 4 real prompts on every rank, or the prompts never shared a step.

Compared: token ids only. Existing cells cannot dump a first-decode hidden: glm53_mixed_graph.KeyMerge._save_first (read by
k1_hidden_compare.py and dual_serial_cell.py hidden) saves a keyed prompt chunk's output, not a decode's, and only inside the keyed path.
The cell ends with one SHORT_CHUNK_COMPARE line; exit 0 only for PASS.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from collections.abc import Callable, Iterable, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Protocol
from urllib.request import Request, urlopen

import c1_cell
import c1_score as scorer
import k1_cell
import k1_score as k1
import memory_cells as cells
import schedule_compare_cell as scheduling

PASS, FAIL, NOT_MEASURED = scorer.PASS, scorer.FAIL, scorer.NOT_MEASURED
QUAD = c1_cell.QUAD
CHUNK_ROWS = k1_cell.CHUNK_ROWS
FULL_CHUNKS = 2
SINGLE_FINAL_ROWS = (1, 11, 700, 1024)
SHORT_WAVE_ROWS = (1, 11, 100, 256)  # every final chunk fits the 4:256 shared key
OUTPUT_TOKENS = MIN_COMPARED_TOKENS = 64
SEED, PROMPT_SEED = 20261109, 20261110
SHARED_FULL_KEY = (4, 256, 1024)  # (prompts, rows, width): the key a step of four prompts takes, graphs-c1m.env's default
DROPPED_ENV = scheduling.DROPPED_ENV  # a numerics run holds no fit ballast
KEY_DEPENDENT_ENV = ("VLLM_GAUDI_MIXED_PROMPT_KEYS", "HPU_GLM53_GRAPHS_ONLY", "VLLM_GAUDI_MIXED_DECODE_WIDTHS", "HPU_GLM53_DECODE_WIDTH_PAD",
                     "HPU_GLM53_DECODE_PAD_CHECK_WIDTHS", "HPU_GLM53_R589_CAPTURE_MARKS")  # the boot refuses each of these without keys
CHUNK_ENV = ("VLLM_GAUDI_MIXED_PREFILL_TOKENS", "VLLM_ADAPTIVE_LONG_PREFILL_SINGLE_TOKENS")
ROWS_NAME = "short-chunk.tokens.jsonl"
COMPARED = "token ids only"
HIDDEN_NOTE = ("first-decode hidden is not compared: the only hidden dump in the cells (KeyMerge._save_first, read by k1_hidden_compare.py) "
               "saves a keyed prompt chunk's output, not a decode, and does not run in the native arm")
Fetch = Callable[..., list[int]]


@dataclass(frozen=True)
class Case:
    index: int
    final_rows: int
    wave: int

    @property
    def tokens(self) -> int:
        return FULL_CHUNKS * CHUNK_ROWS + self.final_rows

    @property
    def name(self) -> str:
        return f"p{self.index}-final-{self.final_rows}"

    @property
    def chunks(self) -> list[int]:
        return [CHUNK_ROWS] * FULL_CHUNKS + [self.final_rows]


@dataclass(frozen=True)
class Arm:
    name: str
    keyed: bool


ARMS = (Arm("native", False), Arm("keyed", True))  # the reference first


@dataclass(frozen=True)
class ArmRun:
    outputs: dict[str, list[int]]
    error: str = ""


class Runner(Protocol):
    def run_arm(self, arm: Arm) -> ArmRun: ...


def cases_for(shared: bool) -> list[Case]:
    """The four single-prompt cases; with shared keys, a second wave of four whose final chunks all fit the 4:256 key."""
    waves = (SINGLE_FINAL_ROWS, SHORT_WAVE_ROWS) if shared else (SINGLE_FINAL_ROWS,)
    placed = [(wave, rows) for wave, group in enumerate(waves) for rows in group]
    return [Case(index, rows, wave) for index, (wave, rows) in enumerate(placed)]


def case_waves(shared: bool) -> list[list[Case]]:
    cases = cases_for(shared)
    return [[case for case in cases if case.wave == wave] for wave in sorted({case.wave for case in cases})]


def case_prompt(case: Case) -> list[int]:
    """Distinct tokens per case, so no two prompts share a prefix a cache hit could skip a chunk on."""
    rng = random.Random(PROMPT_SEED + case.index)
    return [rng.randrange(1000, 16000) for _ in range(case.tokens)]


def compare_case(case: Case, reference: Sequence[int], other: Sequence[int]) -> tuple[str, str, int | None]:
    """(verdict, detail, first differing token). A difference is a FAIL even when a run is short; equal but short is NOT_MEASURED."""
    verdict, detail, index = scheduling.compare_tokens(reference, other)
    shortest = min(len(reference), len(other))
    equal_so_far = verdict == PASS or (verdict == FAIL and index == shortest)  # no token differs; only the lengths may
    if equal_so_far and shortest < MIN_COMPARED_TOKENS:
        return NOT_MEASURED, f"{case.name}: {shortest} tokens compared, at least {MIN_COMPARED_TOKENS} needed", None
    named = f"{case.name} (chunks {case.chunks}), final chunk {case.final_rows} rows: native vs keyed"
    return verdict, f"{named}: {detail}", index


def _case_result(case: Case, runs: Sequence[ArmRun]) -> dict:
    reference, keyed = (run.outputs.get(case.name, []) for run in runs)
    verdict, detail, index = compare_case(case, reference, keyed)
    gap = next((f"{arm.name} arm: {run.error}" for arm, run in zip(ARMS, runs, strict=True) if run.error), "")
    if gap and verdict == PASS:
        verdict, detail = NOT_MEASURED, f"{case.name}: {gap}"
    return {"case": case.name, "final_rows": case.final_rows, "chunks": case.chunks, "verdict": verdict, "detail": detail, "first_difference": index}


def compare_arms(runner: Runner, cases: Sequence[Case]) -> dict:
    """Run both arms, then compare each case. FAIL beats NOT_MEASURED beats PASS."""
    runs = [runner.run_arm(arm) for arm in ARMS]
    results = [_case_result(case, runs) for case in cases]
    return {"verdict": scorer.combine(item["verdict"] for item in results), "compared": COMPARED, "hidden": HIDDEN_NOTE, "cases": results}


def boot_name(arm_name: str, shared: bool) -> str:
    return f"claude-flash-launch-shortchunk-{'shared' if shared else 'single'}-{arm_name}"


def arm_env(base: dict[str, str], arm: Arm) -> dict[str, str]:
    """The base env minus the ballast. Native also loses every key and what needs keys, and cuts the prompt into the same 1,024-row chunks."""
    env = {key: value for key, value in base.items() if key not in DROPPED_ENV}
    if arm.keyed:
        return env
    env = {key: value for key, value in env.items() if key not in KEY_DEPENDENT_ENV and key not in c1_cell.SHARED_ONLY_ENV}
    return {**env, **dict.fromkeys(CHUNK_ENV, str(CHUNK_ROWS))}


class FakeRunner:
    """A runner that boots nothing: each arm's ArmRun is scripted."""

    def __init__(self, scripts: dict[str, ArmRun]) -> None:
        self.scripts = scripts

    def run_arm(self, arm: Arm) -> ArmRun:
        return self.scripts[arm.name]

    @classmethod
    def from_mode(cls, mode: str, cases: Sequence[Case]) -> FakeRunner:
        tokens = list(range(1000, 1000 + OUTPUT_TOKENS))
        native, keyed = ({case.name: list(tokens) for case in cases} for _ in ARMS)
        error = ""
        if mode == "unequal":
            keyed[cases[1].name] = tokens[:17] + [tokens[17] + 1] + tokens[18:]
        if mode == "last-token":
            keyed[cases[-1].name] = tokens[:-1] + [tokens[-1] + 1]
        if mode == "empty":
            keyed[cases[0].name] = []
        if mode == "recapture":
            error = "rank 0: 1 recapture(s)"
        return cls({"native": ArmRun(native), "keyed": ArmRun(keyed, error)})


def _rank_gap(rank: int, counters: list[dict[str, int]], sent_tokens: int) -> str:
    """Why the keyed arm's counters (cumulative, one line per step) do not show every chunk replaying its key; '' when they do."""
    live = next((position for position, line in enumerate(counters) if line.get("live_steps", 0)), None)
    if not live:
        return f"rank {rank}: no counters line before and after the first live step; set VLLM_GAUDI_MIXED_GRAPH_LOG_EVERY=1"
    before, last = counters[live - 1], counters[-1]
    grown = {name: last.get(name, 0) - before.get(name, 0) for name in last}
    fallbacks = sum(count for name, count in grown.items() if "fallback" in name)
    problems = [f"{count} {what}" for count, what in ((grown.get("captures", 0), "recapture(s) after the boot"),
                                                       (fallbacks, "fallback(s): a chunk did not replay its key")) if count]
    if grown.get("prompt_tokens", 0) < sent_tokens:
        problems.append(f"{grown.get('prompt_tokens', 0)} prompt tokens ran, {sent_tokens} sent")
    return f"rank {rank}: " + ", ".join(problems) if problems else ""


def _shared_key_gaps(lines: list[str]) -> list[str]:
    parsed = k1.parse_log(lines)
    name = ":".join(map(str, SHARED_FULL_KEY))
    return [f"rank {rank}: no replay of the {name} shared key carried {SHARED_FULL_KEY[0]} real prompts, so the prompts never shared a step"
            for rank in k1.EXPECTED_RANKS if not scorer.replayed_full(parsed, rank, SHARED_FULL_KEY)]


def keyed_log_gap(lines: Iterable[str], cases: Sequence[Case], shared: bool) -> str:
    """'' when the keyed container log shows every prompt row replaying a key with no capture, fallback or missing chunk, on every rank."""
    lines = list(lines)
    ranks = scorer.scan_log(lines).ranks
    sent = sum(case.tokens for case in cases)
    problems = [gap for rank in scorer.EXPECTED_RANKS if (gap := _rank_gap(rank, ranks[rank].counters, sent))]
    return "; ".join(problems or (_shared_key_gaps(lines) if shared else []))


def request_token_ids(port: int, prompt: list[int], tokens: int, *, timeout: int, salt: str) -> list[int]:
    """One non-streamed greedy completion with the fixed seed; the output token ids."""
    sys.path.insert(0, str(cells.FLASH))
    from probe_flash_c16_starvation import MODEL

    payload = {"model": MODEL, "prompt": prompt, "max_tokens": tokens, "ignore_eos": True, "temperature": 0, "seed": SEED, "logprobs": 0,
               "return_tokens_as_token_ids": True, "cache_salt": salt}
    request = Request(f"http://127.0.0.1:{port}/v1/completions", data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"})
    with urlopen(request, timeout=timeout) as response:
        return scheduling.parse_token_ids(json.load(response))


def _fetch_case(args: argparse.Namespace, fetch: Fetch, case: Case) -> list[int]:
    return fetch(args.port, case_prompt(case), OUTPUT_TOKENS, timeout=k1_cell.long_timeout_s(case.tokens), salt=f"{args.salt}-{case.name}")


def _run_wave(args: argparse.Namespace, fetch: Fetch, wave: list[Case], found: dict[str, list[int]]) -> list[str]:
    """The wave's cases at once with shared keys (so they can share a step), one at a time otherwise; the errors it hit."""
    errors = []
    with ThreadPoolExecutor(max_workers=len(wave) if args.shared else 1) as pool:
        futures = [(case, pool.submit(_fetch_case, args, fetch, case)) for case in wave]
        for case, future in futures:
            try:
                found[case.name] = future.result()
            except Exception as error:  # a dropped connection or a timeout is the arm's result
                errors.append(f"{case.name} {type(error).__name__}: {error}")
    return errors


def drive_arm(args: argparse.Namespace, fetch: Fetch = request_token_ids) -> int:
    """Every case's tokens, one row of outputs and errors. The same prompts, seed and salts in every arm."""
    found: dict[str, list[int]] = {}
    errors = [error for wave in case_waves(args.shared) for error in _run_wave(args, fetch, wave, found)]
    c1_cell.append_row(args.out, {"arm": args.arm, "outputs": found, "error": "; ".join(errors)})
    return 0 if not errors and all(found.values()) else 1


def arm_boot(base_env: dict[str, str], args: argparse.Namespace, arm: Arm) -> cells.Boot:
    drive = ["python3", str(Path(__file__).resolve()), "drive", "--port", str(args.port), "--arm", arm.name, "--salt", args.salt,
             "--out", str(args.out_dir / ROWS_NAME), *(["--shared"] * args.shared)]
    budget = len(cases_for(args.shared)) * k1_cell.REQUEST_TIMEOUT_S + 60
    return cells.Boot(boot_name(arm.name, args.shared), arm_env(base_env, arm), [cells.Step("drive", drive, budget)])


def arm_outputs(path: Path, arm: str) -> ArmRun:
    """The last row the arm's drive wrote; no row is an empty run."""
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()] if path.is_file() else []
    mine = [row for row in rows if row["arm"] == arm]
    return ArmRun(mine[-1]["outputs"], mine[-1]["error"]) if mine else ArmRun({}, "the drive step wrote no row")


class LabRunner:
    """One lab.py boot per arm; down always runs. The drive's row carries the tokens, the keyed container log the recapture check."""

    def __init__(self, base_env: dict[str, str], args: argparse.Namespace) -> None:
        self.base_env, self.args = base_env, args

    def run_arm(self, arm: Arm) -> ArmRun:
        boot = arm_boot(self.base_env, self.args, arm)
        results = k1_cell.run_k1_boot(boot, self.args)
        failed = next((step for step in results if step["returncode"] != 0), None)
        if failed is not None:
            return ArmRun({}, f"{failed['label']} step returned {failed['returncode']} ({failed['stderr']})".strip())
        run = arm_outputs(self.args.out_dir / ROWS_NAME, arm.name)
        gap = self.log_gap(boot.name) if arm.keyed else ""
        return replace(run, error="; ".join(part for part in (run.error, gap) if part))

    def log_gap(self, name: str) -> str:
        log = k1_cell.LANE / f"{name}.log"  # lab.py down writes it
        if not log.is_file():
            return f"no container log {log}, so a recapture cannot be ruled out"
        return keyed_log_gap(log.read_text(errors="replace").splitlines(), cases_for(self.args.shared), self.args.shared)


def render_report(result: dict) -> str:
    lines = [f"{item['verdict']:<13} {item['detail']}" for item in result["cases"]]
    return "\n".join([*lines, f"compared: {result['compared']}; {result['hidden']}", f"SHORT_CHUNK_COMPARE: {result['verdict']}"])


def dry_run_plan(base_env: dict[str, str], args: argparse.Namespace) -> dict[str, object]:
    arms = []
    for arm in ARMS:
        steps = cells.boot_steps(arm_boot(base_env, args, arm), args)
        arms.append({"arm": arm.name, "env": arm_env(base_env, arm), "worst_case_s": sum(step.timeout_s for step in steps),
                     "steps": [{"label": s.label, "timeout_s": s.timeout_s, "argv": s.argv} for s in steps]})
    cases = [{"case": case.name, "tokens": case.tokens, "chunks": case.chunks, "wave": case.wave} for case in cases_for(args.shared)]
    return {"dry_run": True, "touches_cards": False, "cell": "short-chunk-compare", "quad": QUAD, "compared": COMPARED, "hidden": HIDDEN_NOTE,
            "arms": arms, "cases": cases}


def execute(base_env: dict[str, str], args: argparse.Namespace) -> int:
    if (args.out_dir / ROWS_NAME).exists():
        raise SystemExit(f"{args.out_dir / ROWS_NAME} exists: one output folder per run, pass a new --out-dir")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    result = compare_arms(LabRunner(base_env, args), cases_for(args.shared))
    path = args.out_dir / f"short-chunk-{args.stamp}.json"
    path.write_text(json.dumps(result, indent=2) + "\n")
    print(render_report(result))
    print("SHORT_CHUNK_COMPARE_RESULT", json.dumps({"verdict": result["verdict"], "result": str(path)}), flush=True)
    return 0 if result["verdict"] == PASS else 1


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run")
    run.add_argument("--image", required=True, help="the claude-flash-* leaf holding this overlay")
    run.add_argument("--port", required=True, type=k1_cell.lab_port)
    run.add_argument("--out-dir", required=True, type=Path)
    run.add_argument("--env-file", type=Path, default=None, help="default graphs-c1.env beside lab.py (graphs-c1m.env with --shared-keys)")
    run.add_argument("--shared-keys", dest="shared", action="store_true", help="step 49: graphs-c1m.env, prompts four at a time")
    run.add_argument("--mount", action="append", default=[], help="SRC:DST for lab.py up, e.g. the stall trace and Habana logs")
    run.add_argument("--stamp", default=datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"))
    run.add_argument("--salt", default="", help="cache-salt prefix; default claude-short-<stamp>")
    mode = run.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true", help="the default: print the plan, touch nothing")
    mode.add_argument("--execute", action="store_true", help="boot quad A once per arm and compare (needs the card window)")
    fake = sub.add_parser("fake")
    fake.add_argument("--mode", required=True, choices=("equal", "unequal", "last-token", "empty", "recapture"))
    fake.add_argument("--shared-keys", dest="shared", action="store_true")
    driver = sub.add_parser("drive")
    driver.add_argument("--port", required=True, type=k1_cell.lab_port)
    driver.add_argument("--arm", required=True, choices=[arm.name for arm in ARMS])
    driver.add_argument("--salt", required=True)
    driver.add_argument("--out", required=True, type=Path)
    driver.add_argument("--shared", action="store_true")
    args = parser.parse_args(argv)
    if args.command == "run":
        args.quad, args.variant = QUAD, scorer.SHARED_VARIANT if args.shared else scorer.DEFAULT_VARIANT
        args.env_file = args.env_file or k1_cell.LANE / f"graphs-{args.variant}.env"
        args.salt = args.salt or f"claude-short-{args.stamp}"
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.command == "drive":
        return drive_arm(args)
    if args.command == "fake":
        result = compare_arms(FakeRunner.from_mode(args.mode, cases_for(args.shared)), cases_for(args.shared))
        print(render_report(result))
        return 0 if result["verdict"] == PASS else 1
    base_env = c1_cell.load_env(args.env_file, args.variant)
    c1_cell.require_stall_mount(base_env, args)
    if not args.execute:
        print(json.dumps(dry_run_plan(base_env, args), indent=2))
        return 0
    return execute(base_env, args)


if __name__ == "__main__":
    raise SystemExit(main())
