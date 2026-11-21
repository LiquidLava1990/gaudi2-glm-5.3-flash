"""K1 kill cell of plan step 28 (DUAL-SUSTAINED-PLAN.md section 0). WAITING ON A CARD WINDOW: nothing here touches a card
unless --execute is given; the default is a dry run.

    k1_cell.py run --arm B|A --image <leaf> --port 18440 --out-dir <dir> [--env-file graphs-k1.env] [--tokens N] [--repeats N] [--execute]
    k1_cell.py drive --port P --tokens N --repeats N --timeout S --out <file>      (the request step, run by `run`)

Boots quad A once through lab.py (so its quad lock, source-unit, replica-port and 25 min deadline refusals apply), sends the one
fixed prompt `--repeats` times (default 12, at least 8) with a distinct cache salt each, always runs lab.py down, then scores the
container's log with k1_score.py (per rank: resident r, capture peak, replay transient t from census `between_peak`, warmed replay
ms, width merge). All 6,144 tokens sit at W=1024; the W=8192 key replays only on the forced merges: the chunks at start 0, 4096 and
5120, wide on every second repeat, so a pair of repeats gives 3 wide replays per rank. The census times every second replay and
the first is discarded, so 8 repeats give 12 wide replays (6 timed) and 12 give 18 (8 timed of the 16 it logs) per rank. W=1024
gets far more: every other chunk of every repeat. The scorer also needs a merge line from a chunk starting at 4096 or later.
Arm B first (HPU_GLM53_R589_CAPTURE_MARKS=1, tile marks kept as sync=False); arm A (flag 0, `_boundary` no-op) only if B fails.
--dry-run (the default) prints the plan as JSON and runs nothing. The cell ends with one K1_CELL line; exit 0 only for PASS.
"""
from __future__ import annotations

import argparse
import json
import random
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import k1_score as scorer
import memory_cells as cells

LANE = cells.LANE
FLASH = cells.FLASH
QUAD = "A"
ARM_FLAG = "HPU_GLM53_R589_CAPTURE_MARKS"
ARM_VALUES = {"B": "1", "A": "0"}
DEFAULT_ENV_FILE = LANE / "graphs-k1.env"
REQUIRED_ENV = {"VLLM_GAUDI_MIXED_PROMPT_KEYS": "1024:8192,1024:1024", "VLLM_GAUDI_PROMPT_GRAPH_MAX_TOTAL_TOKENS": "1",
                "PT_HPUGRAPH_DISABLE_TENSOR_CACHE": "1", "HPU_GLM53_PREFILL_HBM_CENSUS": "1",
                "VLLM_GAUDI_MIXED_PREFILL_TOKENS": "1024",  # the default 512 caps every chunk (scheduler.py mixed_remaining)
                "HPU_GLM53_Q4_SHARED_GATHER_OWNER": "1", "PT_HPU_LAZY_COLLECTIVES_HOLD_TENSORS": "0",
                "HPU_GLM53_K1_BALLAST_MIB": "1538", "HPU_GLM53_K1_BALLAST_EVERY": "1", "HPU_GLM53_PROMPT_KEY_MERGE_CHECK": "1"}
STALL_DIR_FLAG = "HPU_GLM53_K1_STALL_DIR"
DEADLINE_MESSAGE = "startup deadline"
PROMPT_SEED = 20261106
# Six 1024-row chunks, starts 0, 1024 ... 5120. Every chunk's context stays inside 1024 table blocks (131,072 tokens at 128 per
# block), so all 6,144 tokens sit at W=1024 and the 8192 key never replays on its own. It replays only on a forced merge
# (HPU_GLM53_PROMPT_KEY_MERGE_CHECK=1): the merge check covers the chunk at start 0 and every chunk starting at LATE_CHUNK_START or
# later (4096 and 5120 here), runs each of them narrow on the first repeat and on the widest key on the next. So a pair of repeats
# gives MERGE_CHUNKS_PER_REPEAT (3) W=8192 replays, and the 3 merge lines (start 0, 4096, 5120) the scorer needs.
PROMPT_TOKENS = 6144
CHUNK_ROWS = 1024  # VLLM_GAUDI_MIXED_PREFILL_TOKENS and the scheduler's single-prompt chunk
LATE_CHUNK_START = 4096  # glm53_mixed_graph.MERGE_MIN_PAST_TOP_K_START: from here a chunk is past top-k 2048
MERGE_CHUNKS_PER_REPEAT = sum(1 for start in range(0, PROMPT_TOKENS, CHUNK_ROWS) if start == 0 or start >= LATE_CHUNK_START)
TARGET_WARMED_SAMPLES = 5  # timed warmed replays per width per rank the cell must produce (the scorer refuses below 3)
TIMED_REPLAY_STRIDE = 2  # glm53_prompt_census times every second replay of a key: replays #2, #4, ...; #1 is the discarded first
MERGE_STRIDE = 2  # glm53_mixed_graph.KeyMerge: narrow on one repeat, wide on the next
SPARE_WIDE_REPLAYS = 6  # headroom over the target; the census logs a shape's first 16 replays, so 8 timed ones are the most it can show
MIN_WIDE_REPLAYS = TARGET_WARMED_SAMPLES * TIMED_REPLAY_STRIDE


def repeats_for(wide_replays: int) -> int:
    """Repeats (always whole narrow/wide pairs) whose forced merges give at least `wide_replays` W=8192 replays per rank."""
    return MERGE_STRIDE * -(-wide_replays // MERGE_CHUNKS_PER_REPEAT)


MIN_REPEATS = repeats_for(MIN_WIDE_REPLAYS)  # 8 repeats = 12 wide replays = 6 timed
REPEATS = repeats_for(MIN_WIDE_REPLAYS + SPARE_WIDE_REPLAYS)  # 12 repeats = 18 wide replays, 8 of the 16 logged ones timed
REQUEST_TIMEOUT_S = 900
DRIVE_BUDGET_S = 1800  # all repeats together: 12 healthy requests take minutes, so a stall never holds the card for hours
MAX_MODEL_TOKENS = 1_048_576  # clients' declared window; a prompt leaves room for its one token
LONG_CHUNK_BUDGET_S = 8  # above the k1_score wide limit of 6,600 ms: a stall ends the request, a slow chunk does not
LONG_MARGIN_S = 300
DECODE_TOKEN_BUDGET_S = 10  # far above any decode step at long context: a stall ends the request, a slow token does not


def load_env(path: Path) -> dict[str, str]:
    """KEY=VALUE lines of the env file (blank lines and # comments skipped); refuses a file that lost a K1 setting."""
    env: dict[str, str] = {}
    for number, raw in enumerate(path.read_text().splitlines(), 1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        key, separator, value = line.partition("=")
        if not separator or not key:
            raise SystemExit(f"{path}:{number}: expected KEY=VALUE, got {raw!r}")
        env[key] = value
    wrong = {key: env.get(key) for key, want in REQUIRED_ENV.items() if env.get(key) != want}
    if wrong:
        raise SystemExit(f"{path} must set {REQUIRED_ENV}; found {wrong}. Restore graphs-k1.env before booking a card window.")
    return env


def arm_env(arm: str, file_env: dict[str, str]) -> dict[str, str]:
    """The env file with the arm switch applied last: B marks tile boundaries as sync=False, A leaves the flag off."""
    return {**file_env, ARM_FLAG: ARM_VALUES[arm]}


def k1_boot(args: argparse.Namespace) -> cells.Boot:
    name = f"claude-flash-launch-k1-arm{args.arm}"
    drive = ["python3", str(Path(__file__).resolve()), "drive", "--port", str(args.port),
             "--tokens", str(args.tokens), "--repeats", str(args.repeats), "--timeout", str(REQUEST_TIMEOUT_S),
             "--out", str(args.out_dir / f"{name}.drive.json")]
    steps = [cells.Step("drive", drive, min(args.repeats * REQUEST_TIMEOUT_S, DRIVE_BUDGET_S) + 60)]
    if args.long_tokens:
        steps.append(long_step(args, name))
    env = arm_env(args.arm, load_env(args.env_file))
    stall_dir = env.get(STALL_DIR_FLAG, "")
    if stall_dir and not any(mount.split(":")[1] == stall_dir for mount in args.mount):
        raise SystemExit(f"{args.env_file} sets {STALL_DIR_FLAG}={stall_dir}; add --mount <host-dir>:{stall_dir}")
    return cells.Boot(name, env, steps)


def long_timeout_s(tokens: int) -> int:
    """One long request's limit: every chunk at LONG_CHUNK_BUDGET_S, and never under the fit requests' limit."""
    return max(REQUEST_TIMEOUT_S, -(-tokens // CHUNK_ROWS) * LONG_CHUNK_BUDGET_S + LONG_MARGIN_S)


def long_step(args: argparse.Namespace, name: str) -> cells.Step:
    """Real long prompts in the fit boot, so they replay under the same two keys and ballast."""
    listed = ",".join(str(tokens) for tokens in args.long_tokens)
    output = args.long_output_tokens
    argv = ["python3", str(Path(__file__).resolve()), "drive-long", "--port", str(args.port), "--tokens-list", listed,
            "--output-tokens", str(output), "--out", str(args.out_dir / f"{name}.long.jsonl")]
    decode_budget = 2 * output * DECODE_TOKEN_BUDGET_S  # the cold request's tokens and the warm turn's
    return cells.Step("long", argv, sum(2 * LONG_MARGIN_S + long_timeout_s(tokens) + decode_budget for tokens in args.long_tokens) + 60)


def run_step(step: cells.Step, *, capture_stderr: bool = False) -> dict[str, object]:
    stderr = subprocess.PIPE if capture_stderr else None
    try:
        done = subprocess.run([*step.argv], timeout=step.timeout_s, check=False, stderr=stderr, text=True)
    except subprocess.TimeoutExpired:
        return {"label": step.label, "returncode": "timeout", "stderr": ""}
    if capture_stderr:
        sys.stderr.write(done.stderr)
    return {"label": step.label, "returncode": done.returncode, "stderr": (done.stderr or "")[-500:]}


def run_k1_boot(boot: cells.Boot, args: argparse.Namespace) -> list[dict[str, object]]:
    """up (stderr kept: the deadline refusal is a kill), then the requests while the boot is usable; down always runs."""
    up, *actions, down = cells.boot_steps(boot, args)
    results = [run_step(up, capture_stderr=True)]
    try:
        for action in actions if results[0]["returncode"] == 0 else []:
            results.append(run_step(action))
            if results[-1]["returncode"] != 0:
                break
    finally:
        results.append(run_step(down))
    return results


def boot_outcome(results: list[dict[str, object]], name: str) -> tuple[float | None, bool, bool]:
    """(seconds to ready, hit the 25 min budget, failed before ready) from the up step and lab.py's state file."""
    up = results[0]
    over_budget = up["returncode"] == "timeout" or DEADLINE_MESSAGE in str(up["stderr"])
    if up["returncode"] == 0:
        state = LANE / f"state-{name}.json"
        return (float(json.loads(state.read_text())["startup_ready_s"]) if state.is_file() else None), False, False
    return None, over_budget, not over_budget


def demote_to_not_measured(result: dict, gap: str) -> None:
    """A run that did not complete proves nothing: PASS or VACUOUS becomes NOT_MEASURED with `gap` first. DEAD stands (its kills are real)."""
    if result["verdict"] in ("PASS", "VACUOUS", "NOT_MEASURED"):
        result["verdict"] = "NOT_MEASURED"
        result["gaps"].insert(0, gap)


def drive_failure(results: list[dict[str, object]]) -> str | None:
    """Why the request step did not finish all its repeats (nonzero exit or timeout), None when it did or never ran."""
    drive = next((step for step in results if step["label"] == "drive"), None)
    if drive is None or drive["returncode"] == 0:
        return None
    return (f"drive step returned {drive['returncode']}: not every repeat completed (a worker crash or a stall); "
            f"readings of a partial run are not a verdict. {drive['stderr']}".strip())


def long_failure(results: list[dict[str, object]]) -> str | None:
    """Why the long-context step did not finish every size, None when it did or never ran. Fit readings stand either way."""
    step = next((item for item in results if item["label"] == "long"), None)
    if step is None or step["returncode"] == 0:
        return None
    return f"long-context step returned {step['returncode']}: see the .long.jsonl rows for the size that failed"


def score_boot(results: list[dict[str, object]], name: str) -> dict:
    seconds, over_budget, boot_failed = boot_outcome(results, name)
    log = LANE / f"{name}.log"  # lab.py down writes it
    lines = log.read_text(errors="replace").splitlines() if log.is_file() else []
    result = scorer.score(scorer.parse_log(lines), seconds, over_budget)
    if boot_failed:
        demote_to_not_measured(result, f"boot failed before ready (see docker logs {name} / {log}); no card reading was taken")
    failure = drive_failure(results)
    if failure:
        demote_to_not_measured(result, failure)
    return result


def dry_run_plan(boot: cells.Boot, args: argparse.Namespace) -> dict[str, object]:
    steps = cells.boot_steps(boot, args)
    return {"dry_run": True, "touches_cards": False, "cell": "k1", "arm": args.arm, "quad": QUAD, "waiting_on": f"a card window on quad {QUAD}",
            "env": boot.env, "worst_case_s": sum(step.timeout_s for step in steps),
            "steps": [{"label": s.label, "timeout_s": s.timeout_s, "argv": s.argv} for s in steps]}


def execute(boot: cells.Boot, args: argparse.Namespace) -> int:
    args.out_dir.mkdir(parents=True, exist_ok=True)
    results = run_k1_boot(boot, args)
    result = score_boot(results, boot.name)
    path = args.out_dir / f"k1-arm{args.arm}-{args.stamp}.json"
    path.write_text(json.dumps({"arm": args.arm, "steps": results, **result}, indent=2) + "\n")
    print(scorer.render_report(result))
    print("K1_CELL", json.dumps({"arm": args.arm, "verdict": result["verdict"], "result": str(path)}), flush=True)
    failure = long_failure(results)
    print("K1_LONG", failure or "complete" if args.long_tokens else "not asked", flush=True)
    return 0 if result["verdict"] == "PASS" and failure is None else 1


def chunk_starts(tokens: int, chunk_rows: int = CHUNK_ROWS) -> list[int]:
    """Start token of every chunk a prompt of `tokens` is scheduled in at `chunk_rows` per step."""
    return list(range(0, tokens, chunk_rows))


def fixed_prompt(tokens: int) -> list[int]:
    rng = random.Random(PROMPT_SEED)
    return [rng.randrange(1000, 16000) for _ in range(tokens)]


def drive(args: argparse.Namespace) -> int:
    """The one fixed prompt, `repeats` times, each with its own cache salt so every request prefills cold."""
    sys.path.insert(0, str(FLASH))
    from probe_flash_c16_starvation import stream

    prompt = fixed_prompt(args.tokens)
    begin = datetime.now(timezone.utc).isoformat()
    ttfts = [stream(args.port, prompt, 1, timeout=args.timeout, cache_salt=f"claude-k1-{time.time_ns()}")["ttft_s"]
             for _ in range(args.repeats)]
    args.out.write_text(json.dumps({"begin_utc": begin, "tokens": args.tokens, "ttft_s": ttfts}) + "\n")
    return 0


def _stream_one(port: int, tokens: int, output_tokens: int, salt: str, timeout: int) -> dict:
    sys.path.insert(0, str(FLASH))
    from probe_flash_c16_starvation import stream

    return stream(port, fixed_prompt(tokens), output_tokens, timeout=timeout, cache_salt=salt)


def decode_rate(reply: dict) -> float | None:
    """Tokens per second after the first token; None when one token leaves nothing to time."""
    chunks = reply["chunks"]
    spread = chunks[-1] - chunks[0]
    return round((len(chunks) - 1) / spread, 3) if len(chunks) > 1 and spread > 0 else None


def long_readings(args: argparse.Namespace, tokens: int) -> dict[str, object]:
    """One size cold, then the same prompt again as a warm turn: same salt, so it hits what the cold request cached."""
    salt = f"claude-k1-long-{time.time_ns()}"
    cold = _stream_one(args.port, tokens, args.output_tokens, salt, long_timeout_s(tokens))
    row = {"ttft_s": cold["ttft_s"], "tokens_per_s": round(tokens / cold["ttft_s"], 1), "decode_tok_per_s": decode_rate(cold)}
    warm = _stream_one(args.port, tokens, args.output_tokens, salt, long_timeout_s(tokens))
    row.update(warm_ttft_s=warm["ttft_s"], warm_decode_tok_per_s=decode_rate(warm))
    if warm["output_sha256"] != cold["output_sha256"]:
        row["error"] = "warm output differs from cold output on the same prompt"
    return row


def drive_long(args: argparse.Namespace) -> int:
    """Each size cold then warm, smallest first. A row is written as each ends, so a crash keeps the sizes that finished."""
    with args.out.open("a") as out:
        for tokens in args.tokens_list:
            row: dict[str, object] = {"tokens": tokens, "begin_utc": datetime.now(timezone.utc).isoformat()}
            try:
                row.update(long_readings(args, tokens))
            except Exception as error:  # a dropped connection or a timeout is this size's result, recorded before stopping
                row["error"] = f"{type(error).__name__}: {error}"
            out.write(json.dumps(row) + "\n")
            out.flush()
            print("K1_LONG_ROW", json.dumps(row), flush=True)
            if "error" in row:
                return 1
    return 0


def long_tokens(raw: str) -> tuple[int, ...]:
    sizes = tuple(int(item) for item in raw.split(",") if item)
    if any(size < 1 or size >= MAX_MODEL_TOKENS for size in sizes):
        raise argparse.ArgumentTypeError(f"--long-tokens {raw}: each size must be 1 to {MAX_MODEL_TOKENS - 1}, leaving room for one output token")
    return sizes


def output_count(raw: str) -> int:
    count = int(raw)
    if count < 1:
        raise argparse.ArgumentTypeError(f"output tokens {raw}: must be at least 1")
    return count


def repeat_count(raw: str) -> int:
    repeats = int(raw)
    if repeats < MIN_REPEATS:
        raise argparse.ArgumentTypeError(
            f"--repeats {repeats} gives fewer than {TARGET_WARMED_SAMPLES} timed warmed W=8192 replays per rank: a pair of repeats forces "
            f"{MERGE_CHUNKS_PER_REPEAT} wide chunks and the census times every second replay; use {MIN_REPEATS} or more (default {REPEATS})")
    return repeats


def lab_port(raw: str) -> int:
    port = int(raw)
    if port in cells.REPLICA_PORTS:
        raise SystemExit(f"port {port} is a production replica port ({sorted(cells.REPLICA_PORTS)}); pick a lab port such as 18440")
    return port


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run")
    run.add_argument("--arm", required=True, choices=sorted(ARM_VALUES))
    run.add_argument("--image", required=True, help="the claude-flash-* leaf holding this overlay")
    run.add_argument("--port", required=True, type=lab_port)
    run.add_argument("--out-dir", required=True, type=Path)
    run.add_argument("--env-file", type=Path, default=DEFAULT_ENV_FILE)
    run.add_argument("--tokens", type=int, default=PROMPT_TOKENS)
    run.add_argument("--repeats", type=repeat_count, default=REPEATS)
    run.add_argument("--long-tokens", type=long_tokens, default=(), help="comma list of real prompt sizes, run after the fit readings")
    run.add_argument("--long-output-tokens", type=output_count, default=1, help="tokens each long request decodes, cold and warm")
    run.add_argument("--mount", action="append", default=[], help="SRC:DST for lab.py up, e.g. the stall trace and Habana logs")
    run.add_argument("--stamp", default=datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"))
    mode = run.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true", help="the default: print the plan, touch nothing")
    mode.add_argument("--execute", action="store_true", help="boot quad A and take the readings (needs the card window)")
    driver = sub.add_parser("drive")
    driver.add_argument("--port", required=True, type=lab_port)
    driver.add_argument("--tokens", required=True, type=int)
    driver.add_argument("--repeats", required=True, type=int)
    driver.add_argument("--timeout", required=True, type=int)
    driver.add_argument("--out", required=True, type=Path)
    longer = sub.add_parser("drive-long")
    longer.add_argument("--port", required=True, type=lab_port)
    longer.add_argument("--tokens-list", required=True, type=long_tokens)
    longer.add_argument("--output-tokens", type=output_count, default=1)
    longer.add_argument("--out", required=True, type=Path)
    args = parser.parse_args(argv)
    if args.command == "run":
        args.quad = QUAD
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.command == "drive":
        return drive(args)
    if args.command == "drive-long":
        return drive_long(args)
    boot = k1_boot(args)
    if not args.execute:
        print(json.dumps(dry_run_plan(boot, args), indent=2))
        return 0
    return execute(boot, args)


if __name__ == "__main__":
    raise SystemExit(main())
