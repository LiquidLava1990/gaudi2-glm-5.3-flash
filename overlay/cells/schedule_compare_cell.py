"""Bitwise check 3 of DUAL-SUSTAINED-PLAN.md section 12.3 (build step 47): one greedy lane replayed at K = 0 and at the slice table,
with and without tail-first reordering, must produce identical tokens. WAITING ON A CARD WINDOW: nothing here touches a card unless
--execute is given; the default is a dry run.

    schedule_compare_cell.py run --image <leaf> --port 18440 --out-dir <dir> [--env-file graphs-c1.env] [--mount SRC:DST]... [--arms A,B]
                             [--execute]
    schedule_compare_cell.py fake --mode equal|unequal|empty        (the scoring path on a fake runner: no boot, no card)
    schedule_compare_cell.py drive --port P --arm NAME --salt S --out <jsonl>       (the request step, run by `run`)

Four arms, one boot each through lab.py (so its quad lock and 25 min deadline refusals apply), on the C1 env minus the ballast:
k0-fifo (the reference), k0-tail, slice-fifo, slice-tail. K = 0 is VLLM_GAUDI_MIXED_SLICE_STEPS unset; the slice arms set the plan table.
Each boot runs the same pressure: 7 short decode lanes, then a cold 150K prompt, then the subject, a greedy 4,096-token prompt with 64
output tokens, whose token ids are compared with the reference arm's. Equal -> PASS, any difference -> FAIL (first differing index named),
no tokens or a lane error -> NOT_MEASURED. The cell ends with one SCHEDULE_COMPARE line; exit 0 only for PASS.
"""
from __future__ import annotations

import argparse
import json
import sys
import threading
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Protocol
from urllib.request import Request, urlopen

import c1_cell
import k1_cell
import memory_cells as cells

PASS, FAIL, NOT_MEASURED = "PASS", "FAIL", "NOT_MEASURED"
QUAD = "A"
SLICE_TABLE = "1024:10,4096:22,8192:36"  # plan 12.2: K = p/d at W = 1024 / 4096 / 8192
SLICE_FLAG, TAIL_FIRST_FLAG = "VLLM_GAUDI_MIXED_SLICE_STEPS", "VLLM_GAUDI_MIXED_TAIL_FIRST"
DROPPED_ENV = ("HPU_GLM53_K1_BALLAST_MIB", "HPU_GLM53_K1_BALLAST_EVERY")  # a numerics run holds no fit ballast
BACKGROUND_LANES, BACKGROUND_OUTPUT_TOKENS = 7, 256
LONG_TOKENS = 150_000
SUBJECT_TOKENS, SUBJECT_OUTPUT_TOKENS = 4096, 64
LONG_HEAD_START_S = 5  # the long prompt is already prefilling when the subject arrives
DRIVE_BUDGET_S = 2 * k1_cell.long_timeout_s(LONG_TOKENS) + k1_cell.REQUEST_TIMEOUT_S
ROWS_NAME = "schedule-compare.tokens.jsonl"
TOKEN_ID = "token_id:"
Send = Callable[..., dict]
Fetch = Callable[..., list[int]]


@dataclass(frozen=True)
class Arm:
    name: str
    slice_table: str
    tail_first: bool


ARMS = (Arm("k0-fifo", "", False), Arm("k0-tail", "", True), Arm("slice-fifo", SLICE_TABLE, False), Arm("slice-tail", SLICE_TABLE, True))


@dataclass(frozen=True)
class ArmRun:
    tokens: list[int]
    error: str = ""


class Runner(Protocol):
    def run_arm(self, arm: Arm) -> ArmRun: ...


def arm_env(base: dict[str, str], arm: Arm) -> dict[str, str]:
    """The base env minus the ballast, with the arm's slice table (absent = K 0), tail-first switch and the FIFO-incompatible flag off."""
    env = {key: value for key, value in base.items() if key not in DROPPED_ENV}
    env[TAIL_FIRST_FLAG] = "1" if arm.tail_first else "0"
    env["VLLM_COMPLETION_FIRST_LONG_PREFILL"] = "0"  # tail-first and completion-first are two prefill orders; every arm states the same one
    if arm.slice_table:
        env[SLICE_FLAG] = arm.slice_table
    return env


def compare_tokens(reference: Sequence[int], other: Sequence[int]) -> tuple[str, str, int | None]:
    """(verdict, detail, first differing index). Nothing to compare is NOT_MEASURED: two empty runs are not 'equal'."""
    if not reference or not other:
        return NOT_MEASURED, f"reference has {len(reference)} tokens, run has {len(other)}", None
    for index, (left, right) in enumerate(zip(reference, other, strict=False)):
        if left != right:
            return FAIL, f"token {index} differs: reference {left}, run {right}", index
    if len(reference) != len(other):
        shorter = min(len(reference), len(other))
        return FAIL, f"run has {len(other)} tokens, reference {len(reference)}: they part at token {shorter}", shorter
    return PASS, f"{len(reference)} tokens identical", None


def _arm_result(arm: Arm, reference: ArmRun, run: ArmRun) -> dict:
    if run.error or reference.error:
        detail = f"{arm.name}: {run.error}" if run.error else f"reference arm failed: {reference.error}"
        return {"arm": arm.name, "verdict": NOT_MEASURED, "detail": detail, "first_difference": None, "tokens": len(run.tokens)}
    verdict, detail, index = compare_tokens(reference.tokens, run.tokens)
    return {"arm": arm.name, "verdict": verdict, "detail": detail, "first_difference": index, "tokens": len(run.tokens)}


def compare_arms(runner: Runner, arms: Sequence[Arm]) -> dict:
    """Run every arm, then compare each with the first. FAIL beats NOT_MEASURED beats PASS."""
    runs = [(arm, runner.run_arm(arm)) for arm in arms]
    reference = runs[0][1]
    results = [_arm_result(arm, reference, run) for arm, run in runs[1:]]
    verdicts = {item["verdict"] for item in results}
    verdict = FAIL if FAIL in verdicts else (NOT_MEASURED if NOT_MEASURED in verdicts or not verdicts else PASS)
    return {"verdict": verdict, "reference": arms[0].name, "reference_tokens": len(reference.tokens), "arms": results}


class FakeRunner:
    """A runner that boots nothing: each arm's ArmRun is scripted."""

    def __init__(self, scripts: dict[str, ArmRun]) -> None:
        self.scripts = scripts

    def run_arm(self, arm: Arm) -> ArmRun:
        return self.scripts[arm.name]

    @classmethod
    def from_mode(cls, mode: str) -> FakeRunner:
        tokens = list(range(1000, 1000 + SUBJECT_OUTPUT_TOKENS))
        scripts = {arm.name: ArmRun(list(tokens)) for arm in ARMS}
        if mode == "unequal":
            scripts["slice-tail"] = ArmRun(tokens[:17] + [tokens[17] + 1] + tokens[18:])
        if mode == "empty":
            scripts["slice-fifo"] = ArmRun([])
        return cls(scripts)


def parse_token_ids(body: dict) -> list[int]:
    """Token ids of a completion asked for with logprobs and return_tokens_as_token_ids ('token_id:123' strings)."""
    try:
        tokens = body["choices"][0]["logprobs"]["tokens"]
    except (KeyError, IndexError, TypeError) as error:
        raise ValueError("the reply has no choices[0].logprobs.tokens; send logprobs=0 and return_tokens_as_token_ids=true") from error
    if not all(isinstance(token, str) and token.startswith(TOKEN_ID) for token in tokens):
        raise ValueError(f"tokens must read 'token_id:<n>' (set return_tokens_as_token_ids=true); got {tokens[:3]}")
    return [int(token[len(TOKEN_ID):]) for token in tokens]


def request_token_ids(port: int, prompt: list[int], tokens: int, *, timeout: int, salt: str) -> list[int]:
    """One non-streamed greedy completion; the output token ids."""
    sys.path.insert(0, str(cells.FLASH))
    from probe_flash_c16_starvation import MODEL

    payload = {"model": MODEL, "prompt": prompt, "max_tokens": tokens, "ignore_eos": True, "temperature": 0, "logprobs": 0,
               "return_tokens_as_token_ids": True, "cache_salt": salt}
    request = Request(f"http://127.0.0.1:{port}/v1/completions", data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"})
    with urlopen(request, timeout=timeout) as response:
        return parse_token_ids(json.load(response))


def _failure(error: BaseException) -> str:
    return f"{type(error).__name__}: {error}"


def _background_lane(args: argparse.Namespace, send: Send, index: int, started: threading.Event) -> None:
    try:
        send(args.port, c1_cell.short_prompt(index), BACKGROUND_OUTPUT_TOKENS, timeout=k1_cell.REQUEST_TIMEOUT_S,
             salt=f"{args.salt}-bg-{index}", on_first=started.set)
    finally:
        started.set()


def _long_prompt(args: argparse.Namespace, send: Send, errors: list[str]) -> None:
    try:
        send(args.port, k1_cell.fixed_prompt(LONG_TOKENS), 1, timeout=k1_cell.long_timeout_s(LONG_TOKENS), salt=f"{args.salt}-long")
    except Exception as error:  # the row records it; the subject's tokens stay readable
        errors.append(f"long prompt {_failure(error)}")


def drive_arm(args: argparse.Namespace, send: Send = c1_cell.stream, fetch: Fetch = request_token_ids) -> int:
    """Pressure, then the subject. The long prompt runs on a daemon thread: the boot goes down when the subject is done, not when it ends."""
    started, errors = threading.Event(), []
    long_thread = threading.Thread(target=_long_prompt, args=(args, send, errors), daemon=True)
    with ThreadPoolExecutor(max_workers=BACKGROUND_LANES) as pool:
        lanes = [pool.submit(_background_lane, args, send, index, started) for index in range(BACKGROUND_LANES)]
        started.wait()
        long_thread.start()
        long_thread.join(timeout=LONG_HEAD_START_S)
        tokens: list[int] = []
        try:
            tokens = fetch(args.port, k1_cell.fixed_prompt(SUBJECT_TOKENS), SUBJECT_OUTPUT_TOKENS,
                           timeout=k1_cell.long_timeout_s(LONG_TOKENS), salt=f"{args.salt}-subject")
        except Exception as error:  # a dropped connection or a timeout is the arm's result
            errors.append(f"subject {_failure(error)}")
        errors += [f"background lane {_failure(lane.exception())}" for lane in lanes if lane.exception() is not None]
    c1_cell.append_row(args.out, {"arm": args.arm, "tokens": tokens, "error": "; ".join(errors)})
    return 0 if tokens and not errors else 1


def arm_boot(base_env: dict[str, str], args: argparse.Namespace, arm: Arm) -> cells.Boot:
    name = f"claude-flash-launch-sched-{arm.name}"
    drive = ["python3", str(Path(__file__).resolve()), "drive", "--port", str(args.port), "--arm", arm.name, "--salt", args.salt,
             "--out", str(args.out_dir / ROWS_NAME)]
    return cells.Boot(name, arm_env(base_env, arm), [cells.Step("drive", drive, DRIVE_BUDGET_S)])


def arm_tokens(path: Path, arm: str) -> ArmRun:
    """The last row the arm's drive wrote; no row is an empty run."""
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()] if path.is_file() else []
    mine = [row for row in rows if row["arm"] == arm]
    return ArmRun(mine[-1]["tokens"], mine[-1]["error"]) if mine else ArmRun([], "the drive step wrote no row")


class LabRunner:
    """One lab.py boot per arm; down always runs. The drive's row carries the tokens."""

    def __init__(self, base_env: dict[str, str], args: argparse.Namespace) -> None:
        self.base_env, self.args = base_env, args

    def run_arm(self, arm: Arm) -> ArmRun:
        results = k1_cell.run_k1_boot(arm_boot(self.base_env, self.args, arm), self.args)
        failed = next((step for step in results if step["returncode"] != 0), None)
        if failed is not None:
            return ArmRun([], f"{failed['label']} step returned {failed['returncode']} ({failed['stderr']})".strip())
        return arm_tokens(self.args.out_dir / ROWS_NAME, arm.name)


def render_report(result: dict) -> str:
    lines = [f"{item['verdict']:<13} {item['arm']:<12} {item['detail']}" for item in result["arms"]]
    return "\n".join([*lines, f"SCHEDULE_COMPARE: {result['verdict']}"])


def dry_run_plan(base_env: dict[str, str], args: argparse.Namespace) -> dict[str, object]:
    arms = []
    for arm in args.arms:
        steps = cells.boot_steps(arm_boot(base_env, args, arm), args)
        arms.append({"arm": arm.name, "env": arm_env(base_env, arm), "worst_case_s": sum(step.timeout_s for step in steps),
                     "steps": [{"label": s.label, "timeout_s": s.timeout_s, "argv": s.argv} for s in steps]})
    return {"dry_run": True, "touches_cards": False, "cell": "schedule-compare", "quad": QUAD, "arms": arms}


def execute(base_env: dict[str, str], args: argparse.Namespace) -> int:
    args.out_dir.mkdir(parents=True, exist_ok=True)
    result = compare_arms(LabRunner(base_env, args), args.arms)
    path = args.out_dir / f"schedule-compare-{args.stamp}.json"
    path.write_text(json.dumps(result, indent=2) + "\n")
    print(render_report(result))
    print("SCHEDULE_COMPARE_RESULT", json.dumps({"verdict": result["verdict"], "result": str(path)}), flush=True)
    return 0 if result["verdict"] == PASS else 1


def arm_list(raw: str) -> list[Arm]:
    by_name = {arm.name: arm for arm in ARMS}
    unknown = [name for name in raw.split(",") if name not in by_name]
    if unknown or len(raw.split(",")) < 2:
        raise argparse.ArgumentTypeError(f"--arms {raw!r}: need the reference first and at least one more of {', '.join(by_name)}; unknown: {unknown}")
    return [by_name[name] for name in raw.split(",")]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run")
    run.add_argument("--image", required=True, help="the claude-flash-* leaf holding this overlay")
    run.add_argument("--port", required=True, type=k1_cell.lab_port)
    run.add_argument("--out-dir", required=True, type=Path)
    run.add_argument("--env-file", type=Path, default=c1_cell.DEFAULT_ENV_FILE)
    run.add_argument("--arms", type=arm_list, default=list(ARMS), help="comma list, reference first (default all four)")
    run.add_argument("--mount", action="append", default=[], help="SRC:DST for lab.py up, e.g. the stall trace and Habana logs")
    run.add_argument("--stamp", default=datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"))
    run.add_argument("--salt", default="", help="cache-salt prefix; default claude-sched-<stamp>")
    mode = run.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true", help="the default: print the plan, touch nothing")
    mode.add_argument("--execute", action="store_true", help="boot quad A once per arm and compare (needs the card window)")
    fake = sub.add_parser("fake")
    fake.add_argument("--mode", required=True, choices=("equal", "unequal", "empty"))
    driver = sub.add_parser("drive")
    driver.add_argument("--port", required=True, type=k1_cell.lab_port)
    driver.add_argument("--arm", required=True, choices=[arm.name for arm in ARMS])
    driver.add_argument("--salt", required=True)
    driver.add_argument("--out", required=True, type=Path)
    args = parser.parse_args(argv)
    if args.command == "run":
        args.quad, args.salt = QUAD, args.salt or f"claude-sched-{args.stamp}"
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.command == "drive":
        return drive_arm(args)
    if args.command == "fake":
        result = compare_arms(FakeRunner.from_mode(args.mode), ARMS)
        print(render_report(result))
        return 0 if result["verdict"] == PASS else 1
    base_env = c1_cell.load_env(args.env_file)
    c1_cell.require_stall_mount(base_env, args)
    if not args.execute:
        print(json.dumps(dry_run_plan(base_env, args), indent=2))
        return 0
    return execute(base_env, args)


if __name__ == "__main__":
    raise SystemExit(main())
