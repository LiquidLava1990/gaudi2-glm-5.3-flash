"""C1 cell of the dual-step build (DUAL-SUSTAINED-PLAN.md section 12.4). WAITING ON A CARD WINDOW: nothing here touches a card
unless --execute is given; the default is a dry run.

    c1_cell.py run --image <leaf> --port 18440 --out-dir <dir> [--env-file graphs-c1.env] [--mount SRC:DST]... [--fresh-cache] [--execute]
    c1_cell.py cold|merge|long|lanes --port P --salt S --out <jsonl> ...      (the request steps, run by `run`)

One boot on quad A through lab.py (its quad lock, source-unit, replica-port and 25 min deadline refusals apply) with the four prompt
keys (1024 rows at widths 8192, 4096, 2048, 1024) and the real decode lattice (graphs-c1.env; --variant c1p reads graphs-c1p.env, see that file). Request steps, all appending rows to one `<name>.drive.jsonl`:
  cold   a lone 128-token prompt and 4 greedy tokens, first after the boot: its first two decode steps must replay
  merge  one lone greedy request at 200,000 tokens (native width 2048, padded to 4096) sent twice: GLM53_DECODE_MERGE lines
  long   150K and 530K tokens, 64 output tokens, each cold then warm (one salt per size)
  lanes  1, 2, 4, 8, 16 concurrent lanes, all at the context of widths 1024, 4096, 8192 (warm prefix after the first lane count); each row keeps
         the lanes' output_sha256, which lane_hash_compare.py compares between two runs
  (c1p only) trace-wait  until every rank's replay-trace window file is in the mounted folder (replay_trace_report.py wait); it runs even when a
                         drive failed, so the container is never stopped mid-export
Then lab.py down always runs, and c1_score.py scores the container log and the rows. The cell ends with one C1_CELL line; exit 0 only for PASS.
c1p (graphs-c1p.env) arms glm53_graph_trace: one device trace of the Nth warmed replay of a prompt key and of a decode key (plan 12.3a). Its
report (replay_trace_report.py: per-family busy ms, top ops, launches, traced vs unprofiled wall) is printed as REPLAY_TRACE_* lines and kept in
the result file under replay_trace; a target without a complete trace is a gap, so the cell reads NOT_MEASURED, never PASS.
c1m (graphs-c1m.env, plan 12.3b, step 49) boots the single keys 1024:8192 and 1024:1024 plus the shared key 4:256:1024 (4:1024:1024 is optional) and runs
  cold    as above
  shared  SHARED_LONG_ROUNDS rounds of 4 concurrent cold prompts of about 4,000 tokens (their output tokens kept), then SHORT_ROUNDS rounds of 4
          concurrent 256-token prompts. All of them take 4:256 at the 1,024-row step budget; with the optional key, the 4,000-token rounds take 4:1024
          at a 4,096-row step budget
  serial  the same 4 long prompts one at a time on the 1024:1024 key, output tokens kept
The scorer reads each shared key's warmed replay ms and resident HBM beside 1024:1024 and compares the tokens: any difference is a FAIL.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import statistics
import sys
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import c1_score as scorer
import k1_cell
import memory_cells as cells
import replay_trace_report as tracing  # puts the overlay folder on sys.path: the hook's plan grammar is read from there
import glm53_graph_trace as hook  # noqa: E402  (after tracing)

LANE = cells.LANE
QUAD = os.environ.get("C1_QUAD", "A")  # a lab cell may run on quad B while A is busy
SERVING_KEYS = os.environ.get("C1_SERVING_KEYS", "1024:8192,1024:1024")  # a lab read may boot fewer keys
SHARED_VARIANT = scorer.SHARED_VARIANT
SHARED_SINGLE_KEYS = "1024:8192,1024:1024"  # the widest key a long prompt needs and the key the shared keys are read beside
VARIANT_KEYS = {"c1": SERVING_KEYS, "c1p": SERVING_KEYS,  # c1p: one traced replay
                SHARED_VARIANT: SHARED_SINGLE_KEYS}  # c1m: four prompts in one step
# Step 49: only c1m shares a step. A 1,024-row step budget splits 256 rows over each of 4 prompts, the 4:256:1024 key. A capture at 2048 rows held about
# 7.1 GB, so 4:1024:1024 (4,096 rows, needing a 4,096-row budget) is unlikely to fit and is optional. Tail-first picks the prompts; completion-first is off.
SHARED_ONLY_ENV = ("VLLM_GAUDI_MIXED_PROMPTS_PER_STEP", "VLLM_GAUDI_MIXED_PROMPT_BATCH_KEYS", "VLLM_GAUDI_MIXED_TAIL_FIRST", "VLLM_PROMPT_BS_BUCKET_MAX")
VARIANT_ENV = {SHARED_VARIANT: {"VLLM_GAUDI_MIXED_PROMPTS_PER_STEP": "4", scorer.BATCH_KEYS_ENV: "4:256:1024",
                                "VLLM_GAUDI_MIXED_TAIL_FIRST": "1", "VLLM_PROMPT_BS_BUCKET_MAX": "4", "VLLM_GAUDI_MIXED_PREFILL_TOKENS": "1024",
                                "VLLM_COMPLETION_FIRST_LONG_PREFILL": "0"}}
SHARED_OPTIONAL_ENV = {scorer.BATCH_KEYS_ENV: "4:256:1024,4:1024:1024", "VLLM_GAUDI_MIXED_PREFILL_TOKENS": "4096"}
DEFAULT_VARIANT = "c1"
TRACE_VARIANT = "c1p"
TRACE_WAIT_LABEL = "trace-wait"
TRACE_WAIT_S = 600  # rank 0 exports a trace of millions of events after the replay; the wait ends as soon as every window file is there
TP_RANKS = 4
DEFAULT_ENV_FILE = LANE / "graphs-c1.env"
REQUIRED_ENV = {"VLLM_GAUDI_MIXED_DECODE_WIDTHS": "1024,4096,8192",
                "HPU_GLM53_DECODE_WIDTH_PAD": "check", "HPU_GLM53_DECODE_PAD_CHECK_WIDTHS": "2048",
                "VLLM_GAUDI_BOOTSTRAP_DECODE_BUCKETS": "1,1,128;2,1,256;4,1,512;8,1,1024;16,1,2048",
                "VLLM_GAUDI_TARGETED_SAMPLER_WARMUP_BATCHES": "1,2,4,8,16", "HPU_GLM53_MIXED_DECODE_REPLAY": "1",
                "VLLM_HPU_RESTORED_DECODE_GRAPHS": "1", "HPU_GRAPH_DRY_RUN": "0", "HPU_GLM53_K1_BALLAST_MIB": "512",
                "HPU_GLM53_PROMPT_KEY_MERGE_CHECK": "0", "HPU_GLM53_GRAPHS_ONLY": "1", "HPU_GLM53_PRIVATE_PROMPT_TOPK": "1",
                "HPU_GLM53_Q4_SHARED_GATHER_OWNER": "1", "VLLM_PREFILL_CONTEXT_CELL_BUDGET": "0",
                "PT_HPU_LAZY_COLLECTIVES_HOLD_TENSORS": "0", "VLLM_GAUDI_MIXED_GRAPH_BUDGET_MIB": "8452",
                "VLLM_GAUDI_MIXED_GRAPH_LOG_EVERY": "1"}
DEFAULT_LONG_TOKENS = scorer.DEFAULT_LONG_SIZES
DEFAULT_LONG_OUTPUT_TOKENS = 64
COLD_TOKENS, COLD_OUTPUT_TOKENS = 128, 4
MERGE_TOKENS, MERGE_OUTPUT_TOKENS, MERGE_REPEATS = 200_000, 32, 2  # 131,073-262,144 tokens: native table width 2048, padded to 4096
# Anchor prompt tokens per decode table width (blocks of 128 tokens): inside 1024, 2048..4096 and 4096..8192 blocks.
DEFAULT_ANCHORS = {1024: 2048, 4096: 262_400, 8192: 530_000}
LANE_OUTPUT_TOKENS = 192
SHORT_PROMPT_TOKENS = 128
LANE_SEED = 20261107
MIN_STEP_INTERVALS = 8  # token gaps every lane needs inside the shared window before a step reading counts
STALL_DIR_FLAG = k1_cell.STALL_DIR_FLAG
SHARED_PROMPTS, SHARED_PROMPT_TOKENS, SHARED_OUTPUT_TOKENS, SHARED_LONG_ROUNDS = 4, 4000, 32, 3
SHORT_PROMPT_TOKENS_SHARED, SHORT_OUTPUT_TOKENS, SHORT_ROUNDS, SHORT_ROUND_BUDGET_S = 256, 4, 12, 120
SHARED_SEED = 20261108
Send = Callable[..., dict]
Fetch = Callable[..., list[int]]


def required_env(env: dict[str, str], variant: str) -> dict[str, str]:
    """What a variant's file must set. A c1m file that lists the optional shared key must also set the step budget that key needs."""
    shared = VARIANT_ENV.get(variant, {})
    if variant == SHARED_VARIANT and env.get(scorer.BATCH_KEYS_ENV) == SHARED_OPTIONAL_ENV[scorer.BATCH_KEYS_ENV]:
        shared = {**shared, **SHARED_OPTIONAL_ENV}
    return {**REQUIRED_ENV, **shared, "VLLM_GAUDI_MIXED_PROMPT_KEYS": VARIANT_KEYS[variant]}


def load_env(path: Path, variant: str = DEFAULT_VARIANT) -> dict[str, str]:
    """KEY=VALUE lines of the env file (blank lines and # comments skipped); refuses a file that lost a C1 setting or holds another variant's keys."""
    env: dict[str, str] = {}
    for number, raw in enumerate(path.read_text().splitlines(), 1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        key, separator, value = line.partition("=")
        if not separator or not key:
            raise SystemExit(f"{path}:{number}: expected KEY=VALUE, got {raw!r}")
        env[key] = value
    required = required_env(env, variant)
    wrong = {key: env.get(key) for key, want in required.items() if env.get(key) != want}
    if wrong:
        raise SystemExit(f"{path} must set {required}; found {wrong}. Restore graphs-{variant}.env before booking a card window.")
    check_trace_env(env, path, variant)
    check_shared_env(env, path, variant)
    return env


def check_shared_env(env: dict[str, str], path: Path, variant: str) -> None:
    """Only c1m shares a step: another variant's file with a step-49 flag would measure something else under its own name."""
    stray = [name for name in SHARED_ONLY_ENV if name in env]
    if stray and variant != SHARED_VARIANT:
        raise SystemExit(f"{path} sets {', '.join(stray)}: only {SHARED_VARIANT} shares a step. Restore graphs-{variant}.env before booking a card window.")


def check_trace_env(env: dict[str, str], path: Path, variant: str) -> None:
    """Only the p variant traces, and it needs the hook's directory, a plan that parses, and the census wrapper that calls the hook."""
    traced = bool(env.get(hook.DIR_ENV, "").strip())
    if traced and variant != TRACE_VARIANT:
        raise SystemExit(f"{path} sets {hook.DIR_ENV}: only {TRACE_VARIANT} traces a replay. Restore graphs-{variant}.env before booking a card window.")
    if variant != TRACE_VARIANT:
        return
    if not traced:
        raise SystemExit(f"{path} must set {hook.DIR_ENV} and {hook.PLAN_ENV} (the replays to trace). Restore graphs-{variant}.env before booking a card window.")
    if env.get(hook.CENSUS_ENV) != "1":
        raise SystemExit(f"{path}: {hook.DIR_ENV} needs {hook.CENSUS_ENV}=1 (the census wrapper is the hook's only caller)")
    try:
        hook.plan_from_env(env)
    except hook.ReplayTraceError as error:
        raise SystemExit(f"{path}: {error}") from error


@dataclass(frozen=True)
class TraceSetup:
    """What the p variant traces: the hook's plan, the host folder behind its container directory, and how many ranks write window files."""
    plan: hook.TracePlan
    host_dir: str
    ranks: int = TP_RANKS

    @property
    def run_ids(self) -> list[str]:
        return [target.run_id for target in self.plan.targets]

    def report_argv(self, verb: str, *extra: str) -> list[str]:
        return ["python3", str(Path(tracing.__file__).resolve()), verb, "--dir", self.host_dir, "--runs", ",".join(self.run_ids),
                "--ranks", str(self.ranks), *extra]


def trace_setup(env: dict[str, str], variant: str, mounts: list[str]) -> TraceSetup | None:
    """None unless this is the p variant. The trace folder must be a mount, or the container loses the files at down."""
    plan = hook.plan_from_env(env) if variant == TRACE_VARIANT else None
    if plan is None:
        return None
    for mount in mounts:
        parts = mount.split(":")
        if parts[-1] in ("ro", "rw"):
            parts.pop()
        if len(parts) >= 2 and parts[-1] == plan.directory:
            return TraceSetup(plan, ":".join(parts[:-1]))
    raise SystemExit(f"{hook.DIR_ENV}={plan.directory} holds the replay traces and needs its mount: add --mount <host-dir>:{plan.directory}")


def lane_anchors(raw: str) -> dict[int, int]:
    """`WIDTH:TOKENS,...` -> {width: tokens}."""
    try:
        pairs = [tuple(int(part) for part in item.split(":")) for item in raw.split(",") if item]
        anchors = dict(pairs)
    except ValueError as error:
        raise argparse.ArgumentTypeError(f"--lane-anchors {raw!r}: expected WIDTH:TOKENS pairs such as 1024:2048,8192:530000") from error
    if not anchors or any(len(pair) != 2 for pair in pairs):
        raise argparse.ArgumentTypeError(f"--lane-anchors {raw!r}: expected WIDTH:TOKENS pairs such as 1024:2048,8192:530000")
    return anchors


def drive_step(label: str, args: argparse.Namespace, extra: list[str], timeout_s: int) -> cells.Step:
    argv = ["python3", str(Path(__file__).resolve()), label, "--port", str(args.port), "--salt", args.salt,
            "--out", str(args.out_dir / f"{args.name}.drive.jsonl"), *extra]
    return cells.Step(label, argv, timeout_s)


def long_budget_s(args: argparse.Namespace) -> int:
    decode = 2 * args.long_output_tokens * k1_cell.DECODE_TOKEN_BUDGET_S
    return sum(2 * k1_cell.long_timeout_s(tokens) + decode for tokens in args.long_tokens) + 60


def lanes_budget_s(args: argparse.Namespace) -> int:
    """Per width: the anchor's first (cold) request, then a warm request per other lane count."""
    return sum(k1_cell.long_timeout_s(tokens) + (len(scorer.LANE_COUNTS) - 1) * k1_cell.REQUEST_TIMEOUT_S
               for tokens in args.anchors.values()) + 60


def require_stall_mount(env: dict[str, str], args: argparse.Namespace) -> None:
    """The stall trace the census writes lands in a mounted directory, or the container loses it at down."""
    stall_dir = env.get(STALL_DIR_FLAG, "")
    if stall_dir and not any(stall_dir in mount.split(":")[1:] for mount in args.mount):
        raise SystemExit(f"{args.env_file} sets {STALL_DIR_FLAG}={stall_dir}; add --mount <host-dir>:{stall_dir}")


def serving_steps(args: argparse.Namespace) -> list[cells.Step]:
    listed = ",".join(map(str, args.long_tokens))
    anchors = ",".join(f"{width}:{tokens}" for width, tokens in args.anchors.items())
    return [drive_step("cold", args, [], k1_cell.REQUEST_TIMEOUT_S + 60),
            drive_step("merge", args, ["--merge-tokens", str(args.merge_tokens)],
                       MERGE_REPEATS * k1_cell.long_timeout_s(args.merge_tokens) + 60),
            drive_step("long", args, ["--long-tokens", listed, "--long-output-tokens", str(args.long_output_tokens)], long_budget_s(args)),
            drive_step("lanes", args, ["--lane-anchors", anchors, "--long-tokens", listed], lanes_budget_s(args))]


def shared_steps(args: argparse.Namespace) -> list[cells.Step]:
    """c1m: the cold first step the scorer reads, four prompts sharing steps, then the same four one at a time."""
    long_s = k1_cell.long_timeout_s(SHARED_PROMPT_TOKENS)
    return [drive_step("cold", args, [], k1_cell.REQUEST_TIMEOUT_S + 60),
            drive_step("shared", args, [], SHARED_LONG_ROUNDS * long_s + SHORT_ROUNDS * SHORT_ROUND_BUDGET_S + 60),
            drive_step("serial", args, [], SHARED_PROMPTS * long_s + 60)]


def c1_boot(args: argparse.Namespace) -> cells.Boot:
    env = load_env(args.env_file, args.variant)
    require_stall_mount(env, args)
    args.trace = trace_setup(env, args.variant, args.mount)
    steps = shared_steps(args) if args.variant == SHARED_VARIANT else serving_steps(args)
    if args.trace is not None:
        steps.append(cells.Step(TRACE_WAIT_LABEL, args.trace.report_argv("wait", "--timeout-s", str(TRACE_WAIT_S)), TRACE_WAIT_S + 60))
    return cells.Boot(args.name, env, steps)


def run_c1_boot(boot: cells.Boot, args: argparse.Namespace) -> list[dict[str, object]]:
    """k1_cell.run_k1_boot, except the trace wait runs after a failed drive too: down must not cut a trace export short."""
    up, *actions, down = cells.boot_steps(boot, args)
    waits = [action for action in actions if action.label == TRACE_WAIT_LABEL]
    results = [k1_cell.run_step(up, capture_stderr=True)]
    try:
        for action in [action for action in actions if action not in waits] if results[0]["returncode"] == 0 else []:
            results.append(k1_cell.run_step(action))
            if results[-1]["returncode"] != 0:
                break
        results += [k1_cell.run_step(wait) for wait in waits] if results[0]["returncode"] == 0 else []
    finally:
        results.append(k1_cell.run_step(down))
    return results


def step_failures(results: list[dict[str, object]]) -> list[str]:
    """One line per request step that did not finish: its readings are partial, so a PASS on them is not earned."""
    return [f"{step['label']} step returned {step['returncode']}: its rows are partial ({step['stderr']})".strip()
            for step in results if step["label"] not in ("up", "down") and step["returncode"] != 0]


def score_boot(results: list[dict[str, object]], args: argparse.Namespace, expected: scorer.Expected) -> dict:
    seconds, over_budget, boot_failed = k1_cell.boot_outcome(results, args.name)
    log = LANE / f"{args.name}.log"  # lab.py down writes it
    lines = log.read_text(errors="replace").splitlines() if log.is_file() else []
    rows = scorer.load_rows(args.out_dir / f"{args.name}.drive.jsonl")
    result = scorer.score(lines, rows, scorer.BootFacts(seconds, over_budget), expected, args.long_tokens, variant=args.variant)
    gaps = step_failures(results)
    setup = getattr(args, "trace", None)
    if isinstance(setup, TraceSetup) and not boot_failed:
        result["replay_trace"] = tracing.build_report(Path(setup.host_dir), setup.run_ids, setup.ranks, lines)
        gaps += tracing.gaps(result["replay_trace"])
    if boot_failed:
        gaps.insert(0, f"boot failed before ready (see docker logs {args.name} / {log}); no card reading was taken")
    result["gaps"] = gaps
    if gaps and result["verdict"] == scorer.PASS:
        result["verdict"] = scorer.NOT_MEASURED
    return result


def dry_run_plan(boot: cells.Boot, args: argparse.Namespace) -> dict[str, object]:
    steps = cells.boot_steps(boot, args)
    plan: dict[str, object] = {
        "dry_run": True, "touches_cards": False, "cell": args.variant, "quad": QUAD, "waiting_on": f"a card window on quad {QUAD}",
        "env": boot.env, "worst_case_s": sum(step.timeout_s for step in steps),
        "steps": [{"label": s.label, "timeout_s": s.timeout_s, "argv": s.argv} for s in steps]}
    if isinstance(args.trace, TraceSetup):
        plan["trace"] = {**hook.describe_plan(args.trace.plan, args.trace.ranks), "host_dir": args.trace.host_dir,
                         "report_command": " ".join(args.trace.report_argv("report", "--container-log", "<container.log>"))}
    return plan


def execute(boot: cells.Boot, args: argparse.Namespace) -> int:
    args.out_dir.mkdir(parents=True, exist_ok=True)
    results = run_c1_boot(boot, args)
    result = score_boot(results, args, scorer.Expected.from_env(boot.env))
    path = args.out_dir / f"{args.variant}-{args.stamp}.json"
    path.write_text(json.dumps({"steps": results, **result}, indent=2) + "\n")
    print(scorer.render_report(result))
    if "replay_trace" in result:
        print("\n".join(tracing.render(result["replay_trace"])))
    print("C1_CELL", args.variant, json.dumps({"verdict": result["verdict"], "result": str(path)}), flush=True)
    return 0 if result["verdict"] == scorer.PASS else 1


def stream(port: int, prompt: list[int], tokens: int, *, timeout: int, salt: str, on_first: Callable[[], None] | None = None) -> dict:
    """One streamed greedy completion through the probe helper the K1 cell uses: ttft_s, chunk times, output_sha256."""
    sys.path.insert(0, str(cells.FLASH))
    from probe_flash_c16_starvation import stream as probe_stream

    return probe_stream(port, prompt, tokens, on_first=on_first, timeout=timeout, cache_salt=salt)


def append_row(path: Path, row: dict) -> None:
    with path.open("a") as out:
        out.write(json.dumps(row) + "\n")


def drive_cold(args: argparse.Namespace, send: Send = stream) -> int:
    """The first request after the boot: a lone short prompt, so its decode steps are the cold ones the scorer reads from the log."""
    send(args.port, k1_cell.fixed_prompt(COLD_TOKENS), COLD_OUTPUT_TOKENS, timeout=k1_cell.REQUEST_TIMEOUT_S, salt=f"{args.salt}-cold")
    append_row(args.out, {"drive": "cold", "status": "ok", "tokens": COLD_TOKENS, "output_tokens": COLD_OUTPUT_TOKENS})
    return 0


def drive_merge(args: argparse.Namespace, send: Send = stream) -> int:
    """One greedy request, alone, twice with one salt: the first runs the native width, the repeat the padded one, and the workers compare."""
    prompt = k1_cell.fixed_prompt(args.merge_tokens)
    shas = [send(args.port, prompt, MERGE_OUTPUT_TOKENS, timeout=k1_cell.long_timeout_s(args.merge_tokens),
                 salt=f"{args.salt}-merge")["output_sha256"] for _ in range(MERGE_REPEATS)]
    equal = len(set(shas)) == 1
    append_row(args.out, {"drive": "merge", "tokens": args.merge_tokens, "repeats": MERGE_REPEATS, "outputs_equal": equal})
    return 0 if equal else 1


def long_row(args: argparse.Namespace, send: Send, tokens: int) -> dict:
    """One size cold, then the same prompt as a warm turn on the same salt."""
    salt, prompt, timeout = f"{args.salt}-long-{tokens}", k1_cell.fixed_prompt(tokens), k1_cell.long_timeout_s(tokens)
    cold = send(args.port, prompt, args.long_output_tokens, timeout=timeout, salt=salt)
    warm = send(args.port, prompt, args.long_output_tokens, timeout=timeout, salt=salt)
    row = {"ttft_s": cold["ttft_s"], "tokens_per_s": round(tokens / cold["ttft_s"], 1), "decode_tok_per_s": k1_cell.decode_rate(cold),
           "warm_ttft_s": warm["ttft_s"], "warm_decode_tok_per_s": k1_cell.decode_rate(warm)}
    if warm["output_sha256"] != cold["output_sha256"]:
        row["error"] = "warm output differs from cold output on the same prompt"
    return row


def drive_long(args: argparse.Namespace, send: Send = stream) -> int:
    """Each size cold then warm, smallest first. A row is written as each ends, so a crash keeps the sizes that finished."""
    for tokens in args.long_tokens:
        row: dict[str, object] = {"drive": "long", "tokens": tokens, "begin_utc": datetime.now(timezone.utc).isoformat()}
        try:
            row.update(long_row(args, send, tokens))
        except Exception as error:  # a dropped connection or a timeout is this size's result, recorded before stopping
            row["error"] = f"{type(error).__name__}: {error}"
        append_row(args.out, row)
        print("C1_LONG_ROW", json.dumps(row), flush=True)
        if "error" in row:
            return 1
    return 0


def anchor_salt(salt: str, tokens: int, long_sizes: tuple[int, ...]) -> str:
    """fixed_prompt(n) is a prefix of fixed_prompt(m) for n < m, so an anchor shares the salt of the smallest long size that covers it:
    the long drive's cached prompt may serve it. A longer anchor gets its own salt."""
    covering = next((size for size in sorted(long_sizes) if size >= tokens), None)
    return f"{salt}-long-{covering}" if covering else f"{salt}-anchor-{tokens}"


def short_prompt(index: int) -> list[int]:
    rng = random.Random(LANE_SEED + index)
    return [rng.randrange(1000, 16000) for _ in range(SHORT_PROMPT_TOKENS)]


def step_ms_from_chunks(lanes: list[list[float]]) -> tuple[float, int] | None:
    """(median token gap in ms, intervals per lane) inside the window where every lane is decoding, None when there is none worth reading."""
    start, end = max(chunks[0] for chunks in lanes), min(chunks[-1] for chunks in lanes)
    if end <= start:
        return None
    gaps, intervals = [], []
    for chunks in lanes:
        inside = [moment for moment in chunks if start <= moment <= end]
        if len(inside) - 1 < MIN_STEP_INTERVALS:
            return None
        gaps.append((inside[-1] - inside[0]) / (len(inside) - 1))
        intervals.append(len(inside) - 1)
    return round(statistics.median(gaps) * 1000, 1), min(intervals)


def run_lane_cell(args: argparse.Namespace, send: Send, width: int, lanes: int) -> list[dict]:
    """Every lane sends the anchor prompt on the anchor's salt, so all of them decode at the width's context; the first lane count of a
    width pays the cold prefill, the rest hit its cache. Lanes the KV cannot hold together queue, and the cell reads unreachable."""
    tokens = args.anchors[width]
    prompt, salt, timeout = k1_cell.fixed_prompt(tokens), anchor_salt(args.salt, tokens, args.long_tokens), k1_cell.long_timeout_s(tokens)
    with ThreadPoolExecutor(max_workers=lanes) as pool:
        futures = [pool.submit(send, args.port, prompt, LANE_OUTPUT_TOKENS, timeout=timeout, salt=salt) for _ in range(lanes)]
        return [future.result() for future in futures]


def lane_row(args: argparse.Namespace, send: Send, width: int, lanes: int) -> dict:
    row = {"drive": "lanes", "lanes": lanes, "width": width, "anchor_tokens": args.anchors[width]}
    try:
        replies = run_lane_cell(args, send, width, lanes)
    except Exception as error:  # a failed request is this cell's result; the drive stops after recording it
        return {**row, "status": "error", "reason": f"{type(error).__name__}: {error}"}
    # Each lane's output hash, kept with or without a step reading: cells/lane_hash_compare.py compares two runs' outputs (flag off against flag on) by it.
    row = {**row, "output_sha256": [reply["output_sha256"] for reply in replies]}
    reading = step_ms_from_chunks([reply["chunks"] for reply in replies])
    if reading is None:
        return {**row, "status": "unreachable", "reason": f"the {lanes} lanes never decoded together for {MIN_STEP_INTERVALS} tokens"}
    return {**row, "status": "ok", "step_ms": reading[0], "samples": reading[1]}


def drive_lanes(args: argparse.Namespace, send: Send = stream) -> int:
    """Lane counts ascending within each width, so the anchor's cold prefill is paid once (by one lane) and the rest hit its cache."""
    for width in sorted(args.anchors):
        for lanes in scorer.LANE_COUNTS:
            row = lane_row(args, send, width, lanes)
            append_row(args.out, row)
            print("C1_LANES_ROW", json.dumps(row), flush=True)
            if row["status"] == "error":
                return 1
    return 0


def token_ids(port: int, prompt: list[int], tokens: int, *, timeout: int, salt: str) -> list[int]:
    """One greedy completion's output token ids (the shared-step comparison needs the ids, not a digest)."""
    import schedule_compare_cell as scheduling  # imports this module, so not at the top

    return scheduling.request_token_ids(port, prompt, tokens, timeout=timeout, salt=salt)


def shared_prompt(index: int, tokens: int = SHARED_PROMPT_TOKENS) -> list[int]:
    """Prompt ``index`` of the shared drive: distinct tokens per index, so a step that mixed one prompt's rows into another's shows in the output."""
    rng = random.Random(SHARED_SEED + index)
    return [rng.randrange(1000, 16000) for _ in range(tokens)]


def round_prompts(kind: str, round_index: int) -> tuple[list[list[int]], int]:
    """(the round's prompts, output tokens each). Long rounds repeat the same four prompts, which the serial drive then sends one at a time;
    short rounds send fresh ones, 256 tokens that all fit the 4:256 key."""
    if kind == "long":
        return [shared_prompt(index) for index in range(SHARED_PROMPTS)], SHARED_OUTPUT_TOKENS
    first = SHARED_PROMPTS * (round_index + 1)
    return [shared_prompt(first + index, SHORT_PROMPT_TOKENS_SHARED) for index in range(SHARED_PROMPTS)], SHORT_OUTPUT_TOKENS


def run_round(args: argparse.Namespace, fetch: Fetch, kind: str, round_index: int) -> list[list[int]]:
    """The round's prompts sent at once, each on its own cold salt; the output token ids in prompt order."""
    prompts, count = round_prompts(kind, round_index)
    timeout = k1_cell.long_timeout_s(len(prompts[0]))
    with ThreadPoolExecutor(max_workers=len(prompts)) as pool:
        futures = [pool.submit(fetch, args.port, prompt, count, timeout=timeout, salt=f"{args.salt}-shared-{kind}-{round_index}-{index}")
                   for index, prompt in enumerate(prompts)]
        return [future.result() for future in futures]


def shared_row(args: argparse.Namespace, fetch: Fetch, kind: str, round_index: int) -> dict:
    row = {"drive": "shared", "kind": kind, "round": round_index}
    try:
        outputs = run_round(args, fetch, kind, round_index)
    except Exception as error:  # a dropped connection or a timeout is this round's result, recorded before stopping
        return {**row, "error": f"{type(error).__name__}: {error}"}
    return {**row, "outputs": outputs} if kind == "long" else {**row, "status": "ok"}


def drive_shared(args: argparse.Namespace, fetch: Fetch = token_ids) -> int:
    """Long rounds, then short ones; a row is written as each ends, so a crash keeps the rounds that finished."""
    for kind, rounds in (("long", SHARED_LONG_ROUNDS), ("short", SHORT_ROUNDS)):
        for round_index in range(rounds):
            row = shared_row(args, fetch, kind, round_index)
            append_row(args.out, row)
            print("C1_SHARED_ROW", json.dumps({key: value for key, value in row.items() if key != "outputs"}), flush=True)
            if "error" in row:
                return 1
    return 0


def drive_serial(args: argparse.Namespace, fetch: Fetch = token_ids) -> int:
    """The long rounds' four prompts one at a time, each alone in its step on the 1024:1024 key, on salts no shared round used."""
    row: dict[str, object] = {"drive": "serial"}
    timeout = k1_cell.long_timeout_s(SHARED_PROMPT_TOKENS)
    try:
        row["outputs"] = [fetch(args.port, shared_prompt(index), SHARED_OUTPUT_TOKENS, timeout=timeout, salt=f"{args.salt}-serial-{index}")
                          for index in range(SHARED_PROMPTS)]
    except Exception as error:
        row["error"] = f"{type(error).__name__}: {error}"
    append_row(args.out, row)
    print("C1_SERIAL_ROW", json.dumps({key: value for key, value in row.items() if key != "outputs"}), flush=True)
    return 1 if "error" in row else 0


DRIVES = {"cold": drive_cold, "merge": drive_merge, "long": drive_long, "lanes": drive_lanes, "shared": drive_shared, "serial": drive_serial}


def positive_tokens(raw: str) -> int:
    count = int(raw)
    if count < 1:
        raise argparse.ArgumentTypeError(f"{raw}: must be at least 1")
    return count


def add_drive_options(parser: argparse.ArgumentParser, label: str) -> None:
    """The options every request step takes, plus the ones its own drive reads."""
    parser.add_argument("--port", required=True, type=k1_cell.lab_port)
    parser.add_argument("--salt", required=True, help="cache-salt prefix shared by every request step of the run")
    parser.add_argument("--out", required=True, type=Path, help="the drive jsonl rows are appended to")
    if label == "merge":
        parser.add_argument("--merge-tokens", type=positive_tokens, default=MERGE_TOKENS)
    if label in ("long", "lanes"):
        parser.add_argument("--long-tokens", type=k1_cell.long_tokens, default=DEFAULT_LONG_TOKENS)
    if label == "long":
        parser.add_argument("--long-output-tokens", type=k1_cell.output_count, default=DEFAULT_LONG_OUTPUT_TOKENS)
    if label == "lanes":
        parser.add_argument("--lane-anchors", dest="anchors", type=lane_anchors, default=DEFAULT_ANCHORS)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run")
    run.add_argument("--image", required=True, help="the claude-flash-* leaf holding this overlay")
    run.add_argument("--port", required=True, type=k1_cell.lab_port)
    run.add_argument("--out-dir", required=True, type=Path)
    run.add_argument("--variant", choices=sorted(VARIANT_KEYS), default=DEFAULT_VARIANT,
                     help="c1 the full set; c1p the same set for one traced replay; "
                          "c1m shares a step between four prompts (key 4:256:1024, optionally 4:1024:1024)")
    run.add_argument("--env-file", type=Path, default=None, help="default graphs-<variant>.env beside lab.py")
    run.add_argument("--long-tokens", type=k1_cell.long_tokens, default=DEFAULT_LONG_TOKENS)
    run.add_argument("--long-output-tokens", type=k1_cell.output_count, default=DEFAULT_LONG_OUTPUT_TOKENS)
    run.add_argument("--merge-tokens", type=positive_tokens, default=MERGE_TOKENS)
    run.add_argument("--lane-anchors", dest="anchors", type=lane_anchors, default=DEFAULT_ANCHORS)
    run.add_argument("--mount", action="append", default=[], help="SRC:DST for lab.py up, e.g. the stall trace and Habana logs")
    run.add_argument("--fresh-cache", action="store_true",
                     help="lab.py up --fresh-cache: boot from an empty recipe leaf (changed graphs or kernels: no seeded recipe may replay)")
    run.add_argument("--stamp", default=datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"))
    run.add_argument("--salt", default="", help="cache-salt prefix; default claude-c1-<stamp>")
    mode = run.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true", help="the default: print the plan, touch nothing")
    mode.add_argument("--execute", action="store_true", help="boot quad A and take the readings (needs the card window)")
    for label in DRIVES:
        add_drive_options(sub.add_parser(label), label)
    args = parser.parse_args(argv)
    if args.command == "run":
        args.quad, args.name = QUAD, f"claude-flash-launch-{args.variant}"
        args.env_file, args.salt = args.env_file or LANE / f"graphs-{args.variant}.env", args.salt or f"claude-{args.variant}-{args.stamp}"
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.command != "run":
        return DRIVES[args.command](args)
    boot = c1_boot(args)
    if not args.execute:
        print(json.dumps(dry_run_plan(boot, args), indent=2))
        return 0
    return execute(boot, args)


if __name__ == "__main__":
    raise SystemExit(main())
