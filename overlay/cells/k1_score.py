"""Scorer of the K1 kill cell (plan DUAL-SUSTAINED-PLAN.md section 0, step 28): one boot's docker log in, a table per rank and
PASS, DEAD, VACUOUS or NOT_MEASURED out. Worst rank decides.

    k1_score.py --log <docker log> [--boot-seconds N] [--boot-over-budget] [--json <out>]

Kill numbers, exactly as section 0 (any one on any rank is DEAD):
  capture   any PT_DEVMEM, any sync=True in capture, boot past 25 min (1500 s)
  fit       a PT_DEVMEM with the 1,538 MiB ballast (decode 1,026 + margin 512) held per rank from the first replay. `t` fills whatever
            is free, so `2r + t` (r = resident of the two keys, t from census `between_peak`) is raw data here and decides nothing.
            A rank without that ballast is NOT_MEASURED, whatever its numbers. A BOOT admission refused on capacity means the two keys
            do not fit, which is the same kill; a refusal on configuration is a boot failure and stays NOT_MEASURED.
  speed     warmed (1,1024,8192) replay above 6,600 ms; warmed (1,1024,1024) replay above 2,000 ms
  merge     (1,1024,1024) vs (1,1024,8192) hidden not torch.equal on the one fixed prompt, at any chunk start. Each rank also needs
            a merge line from a chunk starting at 4096 or later; without one the verdict is NOT_MEASURED

Log lines read (all per rank, rank = the `Worker_TP<n>` of the log prefix, else the order of the census `pid`):
  GLM53_PROMPT_CENSUS {json}   path=graph; kind capture|replay; rows, context (= key width), start_in_use, end_in_use,
                               max_in_use, between_peak, peak_reset_failed (all in use by glm53_prompt_census.py today);
                               `replay_ms` on a replay and `sync_true` on a capture are NOT written by that module yet
  GLM53_K1_MERGE narrow=N wide=N rows=N start=N torch_equal=0|1     (one per chunk start the merge check covers)
  PT_DEVMEM                    anywhere
Warmed = every replay of a key on a rank after the first, which is discarded (its between_peak still holds the capture).
A speed reading is the median of the warmed replays that carry `replay_ms` (the census times every second replay of a key). A key
with fewer than MIN_WARMED_SAMPLES (3) of them has no speed reading, so that width is NOT_MEASURED: one warmed replay is not a median.
A rank with zero graph replays is VACUOUS (a cell that replayed nothing proves nothing). A reading the log does not carry is
NOT_MEASURED, never a pass.
"""
from __future__ import annotations

import argparse
import json
import re
import statistics
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path

MIB = 1 << 20
EXPECTED_RANKS = (0, 1, 2, 3)
WIDE_KEY = (1024, 8192)  # (rows, width); the batch size is always 1 in this cell
LOW_KEY = (1024, 1024)
FIT_BALLAST_MIB = 1538  # 1,026 best-case decode lattice + 512 margin, held per rank during replays
WIDE_REPLAY_LIMIT_MS = 6600.0
LOW_REPLAY_LIMIT_MS = 2000.0
BOOT_LIMIT_S = 25 * 60
MIN_WARMED_SAMPLES = 3  # timed warmed replays a width needs before its median counts
CENSUS_LINE = re.compile(r"GLM53_PROMPT_CENSUS (\{.*\})\s*$")
MERGE_LINE = re.compile(r"GLM53_K1_MERGE\b.*\btorch_equal=([01])")
MERGE_START = re.compile(r"GLM53_K1_MERGE\b.*\bstart=(\d+)\b")
BALLAST_LINE = re.compile(r"GLM53_K1_BALLAST total_mib=(\d+) replays=(\d+) free_bytes=(\d+)")
OWNER_LINE = re.compile(r"GLM53_Q4_SHARED_OWNER\b.*\bowners=(\d+)")
CAPACITY_REFUSAL = re.compile(r"BOOT admission rejected.*(budget|free HBM)")
SELF_DIFF = re.compile(r"GLM53_K1_SELF\b.*\btorch_equal=0")
MERGE_LATE_START = 4096  # a merge line from a chunk starting here or later is past top-k 2048, where selection can differ by width
RANK_PREFIX = re.compile(r"Worker_TP(\d+)")
Key = tuple[int, int]
SharedKey = tuple[int, int, int]  # (prompts, rows per prompt, width): a VLLM_GAUDI_MIXED_PROMPT_BATCH_KEYS key


@dataclass
class KeyData:
    captures: list[dict] = field(default_factory=list)
    replays: list[dict] = field(default_factory=list)  # in log order


@dataclass
class RankData:
    keys: dict[Key, KeyData] = field(default_factory=lambda: defaultdict(KeyData))
    shared: dict[SharedKey, KeyData] = field(default_factory=lambda: defaultdict(KeyData))  # census lines that carry `prompts`
    devmem: int = 0
    merge: list[tuple[int | None, bool]] = field(default_factory=list)  # (chunk start or None for a line without one, torch_equal)
    ballast: tuple[int, int, int] | None = None  # (total MiB, the replay it was taken at, free MiB after)
    self_diffs: int = 0
    owners: int | None = None  # the last GLM53_Q4_SHARED_OWNER count


@dataclass
class ParsedLog:
    capacity_refusal: str = ""
    ranks: dict[int, RankData] = field(default_factory=lambda: defaultdict(RankData))
    devmem_total: int = 0
    garbled: int = 0


@dataclass(frozen=True)
class KeySummary:
    resident_mib: int | None
    capture_peak_mib: int | None
    transient_mib: int | None
    replay_ms: float | None
    replays: int
    warmed: int
    samples: int  # warmed replays that carry replay_ms: what the median is taken over


@dataclass(frozen=True)
class RankRow:
    rank: int
    wide: KeySummary
    low: KeySummary
    devmem: int
    sync_true: int | None  # None: no capture line of this rank carried the field
    merge: bool | None  # True only when every torch_equal line of the rank said 1, at every start
    merge_late: bool = False  # the rank has a merge line from a chunk starting at MERGE_LATE_START or later
    merge_diff_starts: tuple[int | None, ...] = ()  # chunk starts whose line said torch_equal=0
    ballast: tuple[int, int, int] | None = None  # (total MiB, the replay it was taken at, free MiB after)
    self_diffs: int = 0
    owners: int | None = None  # the last GLM53_Q4_SHARED_OWNER count

    @property
    def replays(self) -> int:
        return self.wide.replays + self.low.replays

    @property
    def transient_mib(self) -> int | None:
        found = [key.transient_mib for key in (self.wide, self.low) if key.transient_mib is not None]
        return max(found) if found else None

    @property
    def fit_floor_mib(self) -> int:
        """2r + t with every reading the log lacks counted as zero: a lower bound, so a kill on it is a real kill."""
        return sum(key.resident_mib or 0 for key in (self.wide, self.low)) + (self.transient_mib or 0)

    @property
    def fit_mib(self) -> int | None:
        complete = None not in (self.wide.resident_mib, self.low.resident_mib, self.transient_mib)
        return self.fit_floor_mib if complete else None


def _mib(value: int | None) -> int | None:
    return None if value is None else value // MIB


def _rank_of_prefix(line: str) -> int | None:
    match = RANK_PREFIX.search(line)
    return int(match.group(1)) if match else None


def _census_body(line: str) -> dict | None:
    """The graph-path census body of a line, None for any other line; a body that is not JSON raises ValueError."""
    if "GLM53_PROMPT_CENSUS" not in line or "GLM53_PROMPT_CENSUS read_error" in line:
        return None
    match = CENSUS_LINE.search(line)
    if match is None:
        raise ValueError(line[:200])
    body = json.loads(match.group(1))
    return body if body.get("path") == "graph" else None


def _pid_ranks(pids: Iterable[int]) -> dict[int, int]:
    """Fallback rank by the order of the worker pid, for logs whose prefix names no Worker_TP."""
    return {pid: index for index, pid in enumerate(sorted(set(pids)))}


def _read_ballast(parsed: ParsedLog, rank: int | None, line: str) -> None:
    if rank is None:
        return
    match = BALLAST_LINE.search(line)
    if match:
        total, replay, free = (int(part) for part in match.groups())
        parsed.ranks[rank].ballast = (total, replay, free // MIB)
    parsed.ranks[rank].self_diffs += bool(SELF_DIFF.search(line))
    owner = OWNER_LINE.search(line)
    if owner:
        parsed.ranks[rank].owners = int(owner.group(1))


def _read_counts(parsed: ParsedLog, rank: int | None, line: str) -> None:
    """The line-level readings of one log line: PT_DEVMEM, the ballast and owner lines, a capacity refusal, a merge check."""
    if "PT_DEVMEM" in line:
        parsed.devmem_total += 1
        if rank is not None:
            parsed.ranks[rank].devmem += 1
    _read_ballast(parsed, rank, line)
    refusal = CAPACITY_REFUSAL.search(line)
    if refusal:
        parsed.capacity_refusal = line[refusal.start():][:200]
    merge = MERGE_LINE.search(line)
    if merge and rank is not None:
        start = MERGE_START.search(line)
        parsed.ranks[rank].merge.append((int(start.group(1)) if start else None, merge.group(1) == "1"))


def parse_log(lines: Iterable[str]) -> ParsedLog:
    parsed = ParsedLog()
    census: list[tuple[int | None, dict]] = []
    for line in lines:
        rank = _rank_of_prefix(line)
        _read_counts(parsed, rank, line)
        try:
            body = _census_body(line)
        except ValueError:
            parsed.garbled += 1
            continue
        if body is not None:
            census.append((rank, body))
    _file_census(parsed, census)
    return parsed


def _file_census(parsed: ParsedLog, census: list[tuple[int | None, dict]]) -> None:
    by_pid = _pid_ranks([body["pid"] for rank, body in census if rank is None])
    for rank, body in census:
        owner = rank if rank is not None else by_pid[body["pid"]]
        key = (int(body["rows"]), int(body["context"]))
        prompts = int(body.get("prompts", 1))
        data = parsed.ranks[owner].shared[(prompts, *key)] if prompts > 1 else parsed.ranks[owner].keys[key]
        (data.captures if body.get("kind") == "capture" else data.replays).append(body)


def _largest_above_start_mib(bodies: list[dict], top_field: str) -> int | None:
    """Largest `top_field - start_in_use` over the bodies, in MiB; a body that lacks either reading is skipped (None if all)."""
    sizes = [(body[top_field] - body["start_in_use"]) // MIB for body in bodies
             if body.get(top_field) is not None and body.get("start_in_use") is not None]
    return max(sizes) if sizes else None


def summarize_key(data: KeyData | None) -> KeySummary:
    data = data or KeyData()
    warmed = data.replays[1:]  # the first replay still carries the capture in between_peak and pays first-launch costs
    times = [float(r["replay_ms"]) for r in warmed if r.get("replay_ms") is not None]
    trusted = [r for r in warmed if not r.get("peak_reset_failed")]  # a failed peak reset leaves between_peak cumulative
    median = statistics.median(times) if len(times) >= MIN_WARMED_SAMPLES else None
    return KeySummary(_largest_above_start_mib(data.captures, "end_in_use"), _largest_above_start_mib(data.captures, "max_in_use"),
                      _largest_above_start_mib(trusted, "between_peak"), median, len(data.replays), len(warmed), len(times))


def _sync_true(data: RankData) -> int | None:
    """sync=True marks in the captures of the two K1 keys only: any other graph shape on the rank is not this cell's."""
    reported = [c["sync_true"] for key in (WIDE_KEY, LOW_KEY) if key in data.keys for c in data.keys[key].captures
                if c.get("sync_true") is not None]
    return sum(int(count) for count in reported) if reported else None


def summarize_rank(rank: int, data: RankData) -> RankRow:
    return RankRow(rank, summarize_key(data.keys.get(WIDE_KEY)), summarize_key(data.keys.get(LOW_KEY)), data.devmem,
                   _sync_true(data), all(equal for _, equal in data.merge) if data.merge else None,
                   any(start is not None and start >= MERGE_LATE_START for start, _ in data.merge),
                   tuple(start for start, equal in data.merge if not equal), data.ballast, data.self_diffs, data.owners)


def rank_rows(parsed: ParsedLog) -> list[RankRow]:
    ranks = sorted(set(EXPECTED_RANKS) | set(parsed.ranks))
    return [summarize_rank(rank, parsed.ranks[rank] if rank in parsed.ranks else RankData()) for rank in ranks]


def _speed_kills(row: RankRow) -> list[str]:
    kills = []
    for name, key, limit in (("(1,1024,8192)", row.wide, WIDE_REPLAY_LIMIT_MS), ("(1,1024,1024)", row.low, LOW_REPLAY_LIMIT_MS)):
        if key.replay_ms is not None and key.replay_ms > limit:
            kills.append(f"rank {row.rank}: warmed {name} replay {key.replay_ms:.0f} ms above {limit:.0f} ms")
    return kills


def rank_kills(row: RankRow) -> list[str]:
    kills = []
    if row.devmem:
        kills.append(f"rank {row.rank}: {row.devmem} PT_DEVMEM line(s)")
    if row.sync_true:
        kills.append(f"rank {row.rank}: {row.sync_true} sync=True mark_step(s) in capture")
    if row.merge is False:
        starts = ", ".join("?" if start is None else str(start) for start in row.merge_diff_starts)
        kills.append(f"rank {row.rank}: width merge, hidden of (1,1024,1024) vs (1,1024,8192) is not torch.equal "
                     f"(chunk start {starts})")
    return kills + _speed_kills(row)


def _ballast_gaps(row: RankRow) -> list[str]:
    gaps = [f"rank {row.rank}: {row.self_diffs} GLM53_K1_SELF line(s) where a replay stopped matching its first output"] * bool(row.self_diffs)
    if row.owners is None:
        gaps.append(f"rank {row.rank}: no GLM53_Q4_SHARED_OWNER line, so this cell ran without the shared owner")
    if row.ballast is None:
        return [*gaps, f"rank {row.rank}: no GLM53_K1_BALLAST line, so no fit reading (needs {FIT_BALLAST_MIB} MiB from the first replay)"]
    total, replay, _ = row.ballast
    if total < FIT_BALLAST_MIB or replay != 1:
        gaps.append(f"rank {row.rank}: ballast {total} MiB taken at replay {replay}; fit needs {FIT_BALLAST_MIB} MiB from replay 1")
    return gaps


def _rank_gaps(row: RankRow) -> list[str]:
    gaps = []
    for name, key in (("(1,1024,8192)", row.wide), ("(1,1024,1024)", row.low)):
        missing = [label for label, value in (("resident", key.resident_mib), ("capture peak", key.capture_peak_mib),
                                              ("replay_ms", key.replay_ms), ("between_peak transient", key.transient_mib))
                   if value is None]
        if missing:
            gaps.append(f"rank {row.rank} {name}: no {', '.join(missing)} "
                        f"({key.replays} replay(s), {key.warmed} warmed, {key.samples} timed; a median needs {MIN_WARMED_SAMPLES} timed)")
    if row.sync_true is None:
        gaps.append(f"rank {row.rank}: no capture line carries sync_true")
    gaps += _ballast_gaps(row)
    if row.merge is None:
        gaps.append(f"rank {row.rank}: no GLM53_K1_MERGE line")
    elif not row.merge_late:
        gaps.append(f"rank {row.rank}: no GLM53_K1_MERGE line with start>={MERGE_LATE_START} (a chunk past top-k 2048; "
                    "start=0 alone cannot show a width-dependent selection)")
    return gaps


def score(parsed: ParsedLog, boot_seconds: float | None, boot_over_budget: bool = False) -> dict:
    rows = rank_rows(parsed)
    kills = [kill for row in rows for kill in rank_kills(row)]
    if parsed.devmem_total and not any("PT_DEVMEM" in kill for kill in kills):
        kills.append(f"{parsed.devmem_total} PT_DEVMEM line(s) on no identifiable rank")
    if parsed.capacity_refusal:
        kills.append(f"both keys do not fit: {parsed.capacity_refusal}")
    if boot_over_budget or (boot_seconds is not None and boot_seconds > BOOT_LIMIT_S):
        took = "hit lab.py's startup deadline" if boot_seconds is None else f"took {boot_seconds:.0f} s"
        kills.append(f"boot {took}, limit {BOOT_LIMIT_S} s")
    gaps = [gap for row in rows for gap in _rank_gaps(row)]
    gaps += ["boot seconds not given"] * (boot_seconds is None and not boot_over_budget)
    gaps += [f"{parsed.garbled} garbled census line(s)"] * bool(parsed.garbled)
    free = [row.ballast[2] for row in rows if row.ballast]
    return {"verdict": _verdict(rows, kills, gaps), "kills": kills, "gaps": gaps, "boot_seconds": boot_seconds,
            "ballast_min_free_mib": min(free) if free else None, "shared_owners": [row.owners for row in rows],
            "ranks": [_row_json(row) for row in rows]}


def _verdict(rows: list[RankRow], kills: list[str], gaps: list[str]) -> str:
    if kills:
        return "DEAD"
    if any(row.replays == 0 for row in rows):
        return "VACUOUS"
    return "NOT_MEASURED" if gaps else "PASS"


def _row_json(row: RankRow) -> dict:
    return {"rank": row.rank, "wide": row.wide.__dict__, "low": row.low.__dict__, "transient_mib": row.transient_mib,
            "fit_mib": row.fit_mib, "devmem": row.devmem, "sync_true": row.sync_true, "merge": row.merge}


def _cell(value: object) -> str:
    return "-" if value is None else (f"{value:.0f}" if isinstance(value, float) else str(value))


HEADER = ("rank", "r_wide", "r_low", "cap_wide", "cap_low", "t", "2r+t", "ms_wide", "ms_low", "replays", "devmem", "sync", "merge")


def render_table(result: dict) -> str:
    """Per rank, MiB and ms. `-` is a reading the log did not carry. replays = wide/low, timed warmed samples in brackets."""
    lines = ["  ".join(f"{name:>8}" for name in HEADER)]
    for row in result["ranks"]:
        wide, low = row["wide"], row["low"]
        merge = "-" if row["merge"] is None else ("equal" if row["merge"] else "DIFF")
        cells = [row["rank"], wide["resident_mib"], low["resident_mib"], wide["capture_peak_mib"], low["capture_peak_mib"],
                 row["transient_mib"], row["fit_mib"], wide["replay_ms"], low["replay_ms"],
                 f"{wide['replays']}/{low['replays']}({wide['samples']}/{low['samples']})", row["devmem"], row["sync_true"], merge]
        lines.append("  ".join(f"{_cell(cell):>8}" for cell in cells))
    return "\n".join(lines)


def render_report(result: dict) -> str:
    lines = [render_table(result), "", f"boot_seconds={_cell(result['boot_seconds'])} (limit {BOOT_LIMIT_S})",
             f"ballast_min_free_mib={_cell(result['ballast_min_free_mib'])} (2r+t above is raw data, no verdict)", f"shared_owners={result['shared_owners']}"]
    lines += [f"KILL {kill}" for kill in result["kills"]]
    lines += [f"GAP {gap}" for gap in result["gaps"]]
    lines.append(f"K1_VERDICT: {result['verdict']}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--boot-seconds", type=float)
    parser.add_argument("--boot-over-budget", action="store_true", help="the boot hit lab.py's 25 min startup deadline")
    parser.add_argument("--json", type=Path, help="also write the result here")
    args = parser.parse_args(argv)
    result = score(parse_log(args.log.read_text(errors="replace").splitlines()), args.boot_seconds, args.boot_over_budget)
    if args.json:
        args.json.write_text(json.dumps(result, indent=2) + "\n")
    print(render_report(result))
    return 0 if result["verdict"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
