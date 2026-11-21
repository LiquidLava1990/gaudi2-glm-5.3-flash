"""Scorer of the C1 cell (DUAL-SUSTAINED-PLAN.md sections 12.3 and 12.4): one boot's docker log and the drive rows in, one
PASS / FAIL / NOT_MEASURED per reading and an overall verdict out. A reading the log or the rows do not carry is NOT_MEASURED, never a pass.

    c1_score.py --log <docker log> --rows <drive jsonl> [--boot-seconds N] [--boot-over-budget] [--env-file graphs-c1.env] [--json <out>]

Readings (all per rank, rank = the `Worker_TP<n>` of the log prefix; all four ranks must carry each):
  boot_witness        GLM53_BOOT_WITNESS lists every prompt key and decode key of the env on every rank
  fallbacks           witness native/not_warmed/lazy counts and every `*fallback*` counter of GLM53_MIXED_GRAPH are 0
  eager_census        no GLM53_PROMPT_CENSUS line with path eager
  pt_devmem           no PT_DEVMEM line
  min_free_mib        free HBM after the 512 MiB ballast (GLM53_K1_BALLAST) is at least 512 MiB on every rank
  replay_ms_<width>   median warmed replay per prompt key: W=1024 <= 2,000, W=2048 <= 3,000 (provisional), W=4096 <= 4,000, W=8192 <= 6,600
  decode_step_ms      printed as a table, rows 1, 2, 4, 8, 16 against widths
  decode_step_ms      lanes 1, 2, 4, 8, 16 at widths 1024, 4096, 8192: the drive's `lanes` rows; unreachable or absent cell = NOT_MEASURED
  decode_pad_merge    every GLM53_DECODE_MERGE line says torch_equal=1 on every rank, and the repeated request's two outputs matched
  (not a reading) GLM53_DECODE_BELOW_KEY lines: a context narrower than the lowest key is served padded up to it and never compared. The
                  result carries their count and widths (decode_below_key) and the report says so in one line; they are not evidence of
                  pad equality, never reach decode_pad_merge, and no width is expected to refuse
  cold_decode_replay  the first two decode-only live steps each add one graph replay and no fallback (needs LOG_EVERY=1 counters lines)
  boot_time_s         seconds to ready <= 1,500
  long_drive          every long size ran cold and warm without an error row

c1m (--variant c1m, plan 12.3b: four prompts in one step) reads instead of the serving readings: replay_ms_1024 for the single key, and per shared
key (VLLM_GAUDI_MIXED_PROMPT_BATCH_KEYS, named P x R x W)
  replay_ms_<P>x<R>x<W>    warmed median beside the 1024:1024 key's (ratio and rows/s), on every rank; no replay on a rank = NOT_MEASURED
  resident_mib_<P>x<R>x<W> the key's capture delta of allocator in-use bytes, worst rank, beside the 1024:1024 key's
  shared_vs_serial         the output tokens of the four prompts in shared steps equal those of the same four one at a time: the first differing
                           token is named (FAIL); no rows, an empty output, no replay of the largest shared key, or none that carried as many
                           real prompts as the key holds on every rank (the prompts never shared a step; pad rows are not prompts) = NOT_MEASURED
"""
from __future__ import annotations

import argparse
import ast
import json
import re
from collections import defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path

import k1_score as k1

PASS, FAIL, NOT_MEASURED = "PASS", "FAIL", "NOT_MEASURED"
EXPECTED_RANKS = k1.EXPECTED_RANKS
BOOT_LIMIT_S = k1.BOOT_LIMIT_S
MIN_FREE_MIB = 512
BALLAST_MIB = 512
REPLAY_LIMITS_MS = {1024: 2000.0, 2048: 3000.0, 4096: 4000.0, 8192: 6600.0}  # plan 12.3; 4096 is set after C1, 2048 is provisional (about 1.7 s [E])
BASE_ROWS = 1024  # the rows of the four serving keys
DECODE_ROWS = (1, 2, 4, 8, 16)  # glm53_mixed_graph.DECODE_ROWS
LANE_COUNTS = DECODE_ROWS
DECODE_WIDTHS = (1024, 4096, 8192)
LANE_FIELDS = ("lanes", "width", "status")
KEY_VALUES = re.compile(r"(\w+)=(\S+)")
COUNTERS_LINE = re.compile(r"GLM53_MIXED_GRAPH counters=(\{.*?\}) prompt_tokens_per_step")
MERGE_LINE = re.compile(r"GLM53_DECODE_MERGE narrow=(\d+) wide=(\d+) rows=1 position=\d+ torch_equal=([01])")
BELOW_KEY_LINE = re.compile(r"GLM53_DECODE_BELOW_KEY native=(\d+) key=\d+")
EAGER_CENSUS = re.compile(r'"path":\s*"eager"')
WITNESS_TAG = "GLM53_BOOT_WITNESS"
FALLBACK_WITNESS_FIELDS = ("native_fallbacks", "fallback_not_warmed", "lazy_fallbacks")
COLD_DECODE_STEPS = 2
DEFAULT_LONG_SIZES = (150_000, 530_000)
DEFAULT_VARIANT, SHARED_VARIANT = "c1", "c1m"
BATCH_KEYS_ENV = "VLLM_GAUDI_MIXED_PROMPT_BATCH_KEYS"
SHARED_BASE_KEY = (BASE_ROWS, 1024)  # the single key (rows, width) every shared key is read beside: one prompt, 1:1024:1024
SharedKey = tuple[int, int, int]  # (prompts, rows per prompt, width)


@dataclass(frozen=True)
class Reading:
    name: str
    verdict: str
    detail: str = ""
    value: object = None

    def as_json(self) -> dict:
        return {"name": self.name, "verdict": self.verdict, "detail": self.detail, "value": self.value}


@dataclass(frozen=True)
class BootFacts:
    """What lab.py says of the boot: seconds to ready (None = unknown) and whether it hit the startup deadline."""
    seconds: float | None
    over_budget: bool = False


@dataclass(frozen=True)
class Expected:
    prompt_keys: frozenset[str]
    decode_keys: frozenset[str]
    batch_keys: frozenset[str] = frozenset()  # shared keys as the witness names them, rows:width:prompts

    @classmethod
    def from_env(cls, env: Mapping[str, str]) -> Expected:
        """The buckets BOOT must hold: the env's prompt keys, every decode row x width, and the pad check's native one-row keys."""
        prompts = frozenset(item for item in env.get("VLLM_GAUDI_MIXED_PROMPT_KEYS", "").split(",") if item)
        widths = [int(item) for item in env.get("VLLM_GAUDI_MIXED_DECODE_WIDTHS", "").split(",") if item]
        checks = [int(item) for item in env.get("HPU_GLM53_DECODE_PAD_CHECK_WIDTHS", "").split(",") if item]
        decodes = {f"{rows}x{width}" for width in widths for rows in DECODE_ROWS} | {f"1x{width}" for width in checks}
        shared = (item.split(":") for item in env.get(BATCH_KEYS_ENV, "").split(",") if item)
        return cls(prompts, frozenset(decodes), frozenset(f"{rows}:{width}:{count}" for count, rows, width in shared))

    @classmethod
    def default(cls) -> Expected:
        return cls.from_env({"VLLM_GAUDI_MIXED_PROMPT_KEYS": "1024:8192,1024:4096,1024:2048,1024:1024",
                             "VLLM_GAUDI_MIXED_DECODE_WIDTHS": "1024,4096,8192", "HPU_GLM53_DECODE_PAD_CHECK_WIDTHS": "2048"})


@dataclass
class RankScan:
    witness: dict[str, str] | None = None
    counters: list[dict[str, int]] = field(default_factory=list)  # in log order
    merges: list[bool] = field(default_factory=list)
    below_key: list[int] = field(default_factory=list)  # native table width of each GLM53_DECODE_BELOW_KEY line: served padded, uncompared
    ballast_free_mib: list[int] = field(default_factory=list)  # free after each ballast of at least BALLAST_MIB


@dataclass
class LogScan:
    ranks: dict[int, RankScan] = field(default_factory=lambda: defaultdict(RankScan))
    eager_lines: int = 0
    census_lines: int = 0
    garbled: int = 0


def scan_log(lines: Iterable[str]) -> LogScan:
    scan = LogScan()
    for line in lines:
        if "GLM53_PROMPT_CENSUS" in line:
            scan.census_lines += 1
            scan.eager_lines += bool(EAGER_CENSUS.search(line))
        match = k1.RANK_PREFIX.search(line)
        if match is not None:
            _scan_rank_line(scan, scan.ranks[int(match.group(1))], line)
    return scan


def _scan_rank_line(scan: LogScan, rank: RankScan, line: str) -> None:
    if WITNESS_TAG in line:
        rank.witness = dict(KEY_VALUES.findall(line[line.index(WITNESS_TAG) + len(WITNESS_TAG):]))
    counters = COUNTERS_LINE.search(line)
    if counters is not None:
        try:
            rank.counters.append(ast.literal_eval(counters.group(1)))
        except (ValueError, SyntaxError):
            scan.garbled += 1
    merge = MERGE_LINE.search(line)
    if merge is not None:
        rank.merges.append(merge.group(3) == "1")
    below_key = BELOW_KEY_LINE.search(line)
    if below_key is not None:
        rank.below_key.append(int(below_key.group(1)))
    ballast = k1.BALLAST_LINE.search(line)
    if ballast is not None and int(ballast.group(1)) >= BALLAST_MIB:
        rank.ballast_free_mib.append(int(ballast.group(3)) // k1.MIB)


def combine(verdicts: Iterable[str]) -> str:
    """FAIL beats NOT_MEASURED beats PASS; nothing to combine is NOT_MEASURED."""
    found = set(verdicts)
    if FAIL in found:
        return FAIL
    return NOT_MEASURED if NOT_MEASURED in found or not found else PASS


def _ranks(scan: LogScan) -> list[tuple[int, RankScan]]:
    return [(rank, scan.ranks[rank] if rank in scan.ranks else RankScan()) for rank in sorted(set(EXPECTED_RANKS) | set(scan.ranks))]


def _csv(raw: str) -> frozenset[str]:
    return frozenset(item for item in raw.split(",") if item and item != "none")


def read_boot_witness(scan: LogScan, expected: Expected) -> Reading:
    missing, short = [], []
    for rank, data in _ranks(scan):
        if data.witness is None:
            missing.append(f"rank {rank}: no GLM53_BOOT_WITNESS line")
            continue
        absent = ((expected.prompt_keys | expected.batch_keys) - _csv(data.witness.get("prompt_keys", ""))) | (expected.decode_keys - _csv(data.witness.get("decode_keys", "")))
        if absent:
            short.append(f"rank {rank}: witness lacks {', '.join(sorted(absent))}")
    if short:
        return Reading("boot_witness", FAIL, "; ".join(short + missing))
    if missing:
        return Reading("boot_witness", NOT_MEASURED, "; ".join(missing))
    shared = f", {len(expected.batch_keys)} shared keys" if expected.batch_keys else ""
    return Reading("boot_witness", PASS, f"{len(expected.prompt_keys)} prompt keys{shared} and {len(expected.decode_keys)} decode keys on every rank")


def _counter_fallbacks(counters: list[dict[str, int]]) -> dict[str, int]:
    """Each fallback counter's highest value over the rank's counters lines (the counters only grow)."""
    highest: dict[str, int] = {}
    for line in counters:
        for name, count in line.items():
            if "fallback" in name:
                highest[name] = max(highest.get(name, 0), count)
    return highest


def read_fallbacks(scan: LogScan) -> Reading:
    bad, unread = [], []
    for rank, data in _ranks(scan):
        if data.witness is None and not data.counters:
            unread.append(f"rank {rank}")
            continue
        found = {name: int(data.witness.get(name, 0)) for name in FALLBACK_WITNESS_FIELDS} if data.witness else {}
        for name, count in _counter_fallbacks(data.counters).items():
            found[name] = max(found.get(name, 0), count)
        bad += [f"rank {rank}: {name}={count}" for name, count in sorted(found.items()) if count]
    if bad:
        return Reading("fallbacks", FAIL, "; ".join(bad))
    if unread:
        return Reading("fallbacks", NOT_MEASURED, f"no witness and no counters line on {', '.join(unread)}")
    return Reading("fallbacks", PASS, "0 fallbacks on every rank")


def read_eager_census(scan: LogScan) -> Reading:
    if scan.eager_lines:
        return Reading("eager_census", FAIL, f"{scan.eager_lines} eager GLM53_PROMPT_CENSUS line(s)", scan.eager_lines)
    if not scan.census_lines:
        return Reading("eager_census", NOT_MEASURED, "no GLM53_PROMPT_CENSUS line: HPU_GLM53_PREFILL_HBM_CENSUS=1 produced nothing to count")
    return Reading("eager_census", PASS, f"0 eager of {scan.census_lines} census lines", 0)


def read_devmem(parsed: k1.ParsedLog, scan: LogScan) -> Reading:
    if parsed.devmem_total:
        return Reading("pt_devmem", FAIL, f"{parsed.devmem_total} PT_DEVMEM line(s)", parsed.devmem_total)
    if not any(data.witness for data in scan.ranks.values()):
        return Reading("pt_devmem", NOT_MEASURED, "no boot witness, so the boot never finished and 0 lines proves nothing")
    return Reading("pt_devmem", PASS, "0 PT_DEVMEM lines", 0)


def read_min_free(scan: LogScan) -> Reading:
    lowest, missing = {}, []
    for rank, data in _ranks(scan):
        if data.ballast_free_mib:
            lowest[rank] = min(data.ballast_free_mib)
        else:
            missing.append(f"rank {rank}: no GLM53_K1_BALLAST line with total_mib >= {BALLAST_MIB}")
    worst = min(lowest.values()) if lowest else None
    if worst is not None and worst < MIN_FREE_MIB:
        return Reading("min_free_mib", FAIL, f"free {worst} MiB under {MIN_FREE_MIB} with the {BALLAST_MIB} MiB ballast held", worst)
    if missing:
        return Reading("min_free_mib", NOT_MEASURED, "; ".join(missing), worst)
    return Reading("min_free_mib", PASS, f"lowest free {worst} MiB, bar {MIN_FREE_MIB}", worst)


def read_replay(parsed: k1.ParsedLog, width: int) -> Reading:
    name, limit, medians, gaps = f"replay_ms_{width}", REPLAY_LIMITS_MS[width], [], []
    for rank in EXPECTED_RANKS:
        summary = k1.summarize_key(parsed.ranks[rank].keys.get((1024, width)) if rank in parsed.ranks else None)
        if summary.replay_ms is None:
            gaps.append(f"rank {rank}: {summary.replays} replay(s), {summary.samples} timed warmed; a median needs {k1.MIN_WARMED_SAMPLES}")
        else:
            medians.append(summary.replay_ms)
    worst = max(medians) if medians else None
    if worst is not None and worst > limit:
        return Reading(name, FAIL, f"warmed median {worst:.0f} ms above {limit:.0f} ms", worst)
    if gaps:
        return Reading(name, NOT_MEASURED, "; ".join(gaps), worst)
    return Reading(name, PASS, f"worst rank median {worst:.0f} ms, bar {limit:.0f}", worst)


def _shared_data(parsed: k1.ParsedLog, rank: int, key: SharedKey | k1.Key) -> k1.KeyData | None:
    data = parsed.ranks[rank] if rank in parsed.ranks else None
    if data is None:
        return None
    return data.shared.get(key) if len(key) == 3 else data.keys.get(key)


def _summaries(parsed: k1.ParsedLog, key: SharedKey | k1.Key) -> list[k1.KeySummary]:
    return [k1.summarize_key(_shared_data(parsed, rank, key)) for rank in EXPECTED_RANKS]


def _key_name(key: SharedKey) -> str:
    return f"{key[0]}x{key[1]}x{key[2]}"


def _beside_single(key: SharedKey, wide_ms: float, base_ms: float | None) -> str:
    """The shared key's warmed ms against the single key's: the ratio, and rows per second, since P x R rows move in one forward."""
    single = f"{SHARED_BASE_KEY[0]}:{SHARED_BASE_KEY[1]}"
    if not base_ms:
        return f", no {single} reading beside it"
    rows = key[0] * key[1]
    return (f", {wide_ms / base_ms:.2f}x the {single} key's {base_ms:.0f} ms; {rows / wide_ms * 1000:.0f} rows/s against "
            f"{SHARED_BASE_KEY[0] / base_ms * 1000:.0f}")


def read_shared_replay(parsed: k1.ParsedLog, key: SharedKey) -> Reading:
    """Report-only: a shared key's warmed median beside the single 1024-row key's. Measured on every rank is the pass."""
    name = f"replay_ms_{_key_name(key)}"
    summaries = _summaries(parsed, key)
    medians = [summary.replay_ms for summary in summaries]
    if None in medians:
        counts = [f"rank {rank}: {summary.replays} replay(s), {summary.samples} timed warmed"
                  for rank, summary in zip(EXPECTED_RANKS, summaries) if summary.replay_ms is None]
        return Reading(name, NOT_MEASURED, f"no rank-complete warmed median (needs {k1.MIN_WARMED_SAMPLES} timed replays per rank): " + "; ".join(counts))
    base = [summary.replay_ms for summary in _summaries(parsed, SHARED_BASE_KEY)]
    worst = max(medians)
    return Reading(name, PASS, f"worst rank median {worst:.0f} ms" + _beside_single(key, worst, None if None in base else max(base)), worst)


def read_shared_resident(parsed: k1.ParsedLog, key: SharedKey) -> Reading:
    """The key's resident HBM: the allocator's in-use delta over its capture, worst rank, beside the single key's."""
    name = f"resident_mib_{_key_name(key)}"
    found = [summary.resident_mib for summary in _summaries(parsed, key)]
    if None in found:
        silent = [str(rank) for rank, mib in zip(EXPECTED_RANKS, found) if mib is None]
        return Reading(name, NOT_MEASURED, f"no capture reading on rank {', '.join(silent)}")
    single = [summary.resident_mib for summary in _summaries(parsed, SHARED_BASE_KEY)]
    beside = "" if None in single else f", {max(single)} MiB for the {SHARED_BASE_KEY[0]}:{SHARED_BASE_KEY[1]} key"
    return Reading(name, PASS, f"worst rank {max(found)} MiB resident{beside}", max(found))


def _first_mismatch(rounds: list[dict], serial: list[list[int]]) -> str:
    """'' when every shared round's outputs equal the serial run's, else the first differing token, round by round and prompt by prompt."""
    for row in rounds:
        for prompt, (shared, alone) in enumerate(zip(row["outputs"], serial)):
            token = next((index for index, (left, right) in enumerate(zip(shared, alone)) if left != right), None)
            if token is not None:
                return f"round {row.get('round')} prompt {prompt} token {token}: shared {shared[token]}, serial {alone[token]}"
            if len(shared) != len(alone):
                return f"round {row.get('round')} prompt {prompt} token {min(len(shared), len(alone))}: shared has {len(shared)} tokens, serial {len(alone)}"
    return ""


def _output_gap(rounds: list[dict], serial: dict | None) -> str:
    """Why the outputs cannot be compared ('' when they can): a missing drive row, an output with no tokens, or other prompt counts."""
    if not rounds:
        return "no shared round row with outputs (the shared drive wrote none)"
    if serial is None or not serial.get("outputs"):
        return "no serial row with outputs (the serial drive wrote none)"
    outputs = [row["outputs"] for row in rounds] + [serial["outputs"]]
    if any(not tokens for group in outputs for tokens in group):
        return "an output has no tokens, so there is nothing to compare"
    if len({len(group) for group in outputs}) != 1:
        return f"the rounds ran {sorted({len(group) for group in outputs})} prompts: they must match the serial run's"
    return ""


def replayed_full(parsed: k1.ParsedLog, rank: int, key: SharedKey) -> bool:
    """This rank replayed the key with as many real prompts as it holds. The `prompts` field is only the key's capacity."""
    data = _shared_data(parsed, rank, key)
    return data is not None and any(int(body.get("real_prompts", 0)) >= key[0] for body in data.replays)


def read_shared_output(rows: list[dict], parsed: k1.ParsedLog, keys: list[SharedKey]) -> Reading:
    """The four prompts' output tokens in shared steps against the same four one at a time on the single key: all equal or FAIL."""
    name = "shared_vs_serial"
    rounds = [row for row in rows if row.get("drive") == "shared" and row.get("kind") == "long" and row.get("outputs")]
    serial = next((row for row in rows if row.get("drive") == "serial"), None)
    gap = _output_gap(rounds, serial)
    if gap:
        return Reading(name, NOT_MEASURED, gap)
    mismatch = _first_mismatch(rounds, serial["outputs"])
    if mismatch:
        return Reading(name, FAIL, mismatch)
    big = max(keys, key=lambda key: key[1]) if keys else None
    if big is None or any(not summary.replays for summary in _summaries(parsed, big)):
        return Reading(name, NOT_MEASURED, f"outputs equal, but the log has no replay of the {':'.join(map(str, big or ()))} key on every rank: "
                       "the prompts never shared a step, so equal proves nothing")
    if not all(replayed_full(parsed, rank, big) for rank in EXPECTED_RANKS):
        return Reading(name, NOT_MEASURED, f"outputs equal, but no replay of the {':'.join(map(str, big))} key carried {big[0]} real prompts on every rank: "
                       "the steps ran pad rows, so equal does not prove the prompts shared a step")
    count, length = len(serial["outputs"]), len(serial["outputs"][0])
    return Reading(name, PASS, f"{len(rounds)} shared rounds x {count} prompts equal the one-at-a-time run ({length} tokens each)")


def _lane_cells(rows: list[dict]) -> dict[tuple[int, int], dict]:
    return {(row["lanes"], row["width"]): row for row in rows if row.get("drive") == "lanes" and all(key in row for key in LANE_FIELDS)}


def read_decode_steps(rows: list[dict]) -> Reading:
    cells = _lane_cells(rows)
    measured = {f"{lanes}x{width}": cells[(lanes, width)]["step_ms"] for lanes in LANE_COUNTS for width in DECODE_WIDTHS
                if (lanes, width) in cells and cells[(lanes, width)]["status"] == "ok"}
    errored = [f"{lanes}x{width}: {row.get('reason', '')}" for (lanes, width), row in sorted(cells.items()) if row["status"] == "error"]
    absent = [f"{lanes}x{width}" for lanes in LANE_COUNTS for width in DECODE_WIDTHS if f"{lanes}x{width}" not in measured]
    if errored:
        return Reading("decode_step_ms", FAIL, "request failed in " + "; ".join(errored), measured)
    if absent:
        return Reading("decode_step_ms", NOT_MEASURED, f"no step reading for {', '.join(absent)} (unreachable or never run)", measured)
    return Reading("decode_step_ms", PASS, f"{len(measured)} (lanes x width) cells measured", measured)


def read_pad_merge(scan: LogScan, rows: list[dict]) -> Reading:
    unequal = [f"rank {rank}: {data.merges.count(False)} of {len(data.merges)} merge lines unequal" for rank, data in _ranks(scan) if False in data.merges]
    unequal += ["the repeated request's native and padded outputs differ"] * any(
        row.get("drive") == "merge" and row.get("outputs_equal") is False for row in rows)
    silent = [f"rank {rank}" for rank, data in _ranks(scan) if not data.merges]
    if unequal:
        return Reading("decode_pad_merge", FAIL, "; ".join(unequal))
    if silent:
        return Reading("decode_pad_merge", NOT_MEASURED, f"no GLM53_DECODE_MERGE line on {', '.join(silent)} (zero lines is not a pass)")
    return Reading("decode_pad_merge", PASS, f"{sum(len(data.merges) for _, data in _ranks(scan))} merge lines, all torch_equal")


def below_key_summary(scan: LogScan) -> dict[str, object]:
    """The decodes served padded to the lowest key and never compared: lines, distinct native widths, and ranks that logged any."""
    per_rank = [data.below_key for _, data in _ranks(scan) if data.below_key]
    return {"lines": sum(len(widths) for widths in per_rank), "widths": sorted({width for widths in per_rank for width in widths}),
            "ranks": len(per_rank)}


def _cold_decode_problem(counters: list[dict[str, int]]) -> str:
    """'' when the first COLD_DECODE_STEPS decode-only live steps replayed; else why not (prefix 'unread:' = a gap, not a failure)."""
    previous: dict[str, int] = {}
    found = 0
    for current in counters:
        live = current.get("live_steps", 0) > previous.get("live_steps", 0)
        if live and current.get("mixed_steps", 0) - previous.get("mixed_steps", 0) != 1:
            return "unread: counters lines are not consecutive; set VLLM_GAUDI_MIXED_GRAPH_LOG_EVERY=1"
        decode_only = (live and current.get("decode_rows", 0) > previous.get("decode_rows", 0)
                       and current.get("prompt_tokens", 0) == previous.get("prompt_tokens", 0))
        if decode_only:
            found += 1
            replayed = current.get("graph_replays", 0) - previous.get("graph_replays", 0)
            fallen = sum(current.get(name, 0) - previous.get(name, 0) for name in current if "fallback" in name)
            if replayed != 1 or fallen:
                return f"decode step {found} added {replayed} graph replay(s) and {fallen} fallback(s)"
        previous = current
        if found == COLD_DECODE_STEPS:
            return ""
    return f"unread: {found} of {COLD_DECODE_STEPS} cold decode steps in the counters lines"


def read_cold_decode(scan: LogScan) -> Reading:
    failed, unread = [], []
    for rank, data in _ranks(scan):
        problem = _cold_decode_problem(data.counters)
        if problem.startswith("unread:"):
            unread.append(f"rank {rank}: {problem[len('unread:'):].strip()}")
        elif problem:
            failed.append(f"rank {rank}: {problem}")
    if failed:
        return Reading("cold_decode_replay", FAIL, "; ".join(failed))
    if unread:
        return Reading("cold_decode_replay", NOT_MEASURED, "; ".join(unread))
    return Reading("cold_decode_replay", PASS, f"first {COLD_DECODE_STEPS} decode steps replayed on every rank")


def read_boot_time(boot: BootFacts) -> Reading:
    boot_seconds = boot.seconds
    if boot.over_budget or (boot_seconds is not None and boot_seconds > BOOT_LIMIT_S):
        took = "hit lab.py's startup deadline" if boot_seconds is None else f"took {boot_seconds:.0f} s"
        return Reading("boot_time_s", FAIL, f"boot {took}, limit {BOOT_LIMIT_S} s", boot_seconds)
    if boot_seconds is None:
        return Reading("boot_time_s", NOT_MEASURED, "boot seconds not given (lab.py state file missing)")
    return Reading("boot_time_s", PASS, f"ready in {boot_seconds:.0f} s, limit {BOOT_LIMIT_S} s", boot_seconds)


def read_long(rows: list[dict], wanted: tuple[int, ...]) -> Reading:
    longs = {row["tokens"]: row for row in rows if row.get("drive") == "long"}
    errors = [f"{size}: {longs[size]['error']}" for size in wanted if size in longs and "error" in longs[size]]
    absent = [str(size) for size in wanted if size not in longs]
    if errors:
        return Reading("long_drive", FAIL, "; ".join(errors))
    if absent or not wanted:
        return Reading("long_drive", NOT_MEASURED, f"no long row for {', '.join(absent) or 'any size'}")
    return Reading("long_drive", PASS, f"{', '.join(map(str, wanted))} tokens cold and warm", {size: longs[size].get("warm_ttft_s") for size in wanted})


def _keys(expected: Expected) -> list[tuple[int, int]]:
    return sorted((tuple(int(part) for part in key.split(":")) for key in expected.prompt_keys), key=lambda key: (-key[1], -key[0]))


def _base_widths(expected: Expected) -> list[int]:
    return [width for rows, width in _keys(expected) if rows == BASE_ROWS]


@dataclass(frozen=True)
class Evidence:
    """What one boot left to read: its log (as scanned and as the K1 census parse), the drive rows, the boot facts and the keys the env asked for."""
    scan: LogScan
    parsed: k1.ParsedLog
    rows: list[dict]
    boot: BootFacts
    expected: Expected


def _common_readings(evidence: Evidence) -> list[Reading]:
    """The readings every variant takes: the boot's witness, its fallbacks, eager census, PT_DEVMEM lines and the free HBM under the ballast."""
    scan = evidence.scan
    return [read_boot_witness(scan, evidence.expected), read_fallbacks(scan), read_eager_census(scan), read_devmem(evidence.parsed, scan),
            read_min_free(scan)]


def _serving_readings(evidence: Evidence, long_sizes: tuple[int, ...]) -> list[Reading]:
    parsed, rows, expected = evidence.parsed, evidence.rows, evidence.expected
    return [*_common_readings(evidence), *(read_replay(parsed, width) for width in _base_widths(expected)),
            read_decode_steps(rows), read_pad_merge(evidence.scan, rows), read_cold_decode(evidence.scan), read_boot_time(evidence.boot),
            read_long(rows, long_sizes)]


def _shared_keys(expected: Expected) -> list[SharedKey]:
    """(prompts, rows, width) of the env's shared keys, fewest rows first."""
    keys = sorted(tuple(int(part) for part in key.split(":")) for key in expected.batch_keys)
    return [(prompts, key_rows, width) for key_rows, width, prompts in keys]


def _shared_readings(evidence: Evidence) -> list[Reading]:
    """c1m: the cell drives the 1024:1024 key and the shared keys only, so the other serving keys, the lattice rows and the long sizes are not asked."""
    parsed, keys = evidence.parsed, _shared_keys(evidence.expected)
    return [*_common_readings(evidence), read_replay(parsed, SHARED_BASE_KEY[1]),
            *(read_shared_replay(parsed, key) for key in keys), *(read_shared_resident(parsed, key) for key in keys),
            read_shared_output(evidence.rows, parsed, keys), read_cold_decode(evidence.scan), read_boot_time(evidence.boot)]


def score(lines: Iterable[str], rows: list[dict], boot: BootFacts, expected: Expected,
          long_sizes: tuple[int, ...] = DEFAULT_LONG_SIZES, *, variant: str = DEFAULT_VARIANT) -> dict:
    lines = list(lines)
    evidence = Evidence(scan_log(lines), k1.parse_log(lines), rows, boot, expected)
    readings = _shared_readings(evidence) if variant == SHARED_VARIANT else _serving_readings(evidence, long_sizes)
    return {"verdict": combine(item.verdict for item in readings), "boot_seconds": boot.seconds,
            "garbled": evidence.scan.garbled + evidence.parsed.garbled, "readings": [item.as_json() for item in readings],
            "decode_below_key": below_key_summary(evidence.scan)}


def render_decode_table(steps: dict[str, float]) -> list[str]:
    """Decode step ms, one line per row count, one column per width; `-` is a cell with no reading (unreachable or never run)."""
    head = f"{'rows':>6}" + "".join(f"{'W=' + str(width):>10}" for width in DECODE_WIDTHS)
    body = [f"{rows:>6}" + "".join(f"{steps.get(f'{rows}x{width}', '-'):>10}" for width in DECODE_WIDTHS) for rows in DECODE_ROWS]
    return ["decode step ms by rows (down) and table width in blocks (across):", head, *body]


def render_below_key(summary: dict[str, object]) -> str:
    widths = ",".join(map(str, summary["widths"])) or "-"
    return (f"{'NOTE':<13} {'decode_below_key':<20} {len(summary['widths'])} width(s) [{widths}] on {summary['ranks']} rank(s): served padded to "
            "the lowest key and uncompared, so not evidence of pad equality")


def render_report(result: dict) -> str:
    lines = [f"{item['verdict']:<13} {item['name']:<20} {item['detail']}" for item in result["readings"]]
    for item in result["readings"]:
        if item["name"] == "decode_step_ms" and isinstance(item["value"], dict):
            lines += render_decode_table(item["value"])
    if "decode_below_key" in result:
        lines.append(render_below_key(result["decode_below_key"]))
    lines.append(f"C1_VERDICT: {result['verdict']}")
    return "\n".join(lines)


def load_rows(path: Path | None) -> list[dict]:
    """The drive jsonl; a missing file is an empty list (every drive reading then says NOT_MEASURED)."""
    if path is None or not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--rows", type=Path, help="the drive jsonl the cell wrote")
    parser.add_argument("--boot-seconds", type=float)
    parser.add_argument("--boot-over-budget", action="store_true", help="the boot hit lab.py's 25 min startup deadline")
    parser.add_argument("--json", type=Path, help="also write the result here")
    args = parser.parse_args(argv)
    result = score(args.log.read_text(errors="replace").splitlines(), load_rows(args.rows), BootFacts(args.boot_seconds, args.boot_over_budget),
                   Expected.default())
    if args.json:
        args.json.write_text(json.dumps(result, indent=2) + "\n")
    print(render_report(result))
    return 0 if result["verdict"] == PASS else 1


if __name__ == "__main__":
    raise SystemExit(main())
