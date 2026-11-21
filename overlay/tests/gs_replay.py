"""Replay real GLM53_GS_SATURATION + GLM53_STEP_STATS sequences through a real HbmGuard (open loop: runtime graph counts
and timings come from the log, so a replayed decision cannot change the later readings).

The extract is raw `docker logs` lines. GLM53_STEP_STATS only exists on rank 0; every TP rank runs the same forwards in
the same order, so the capture flag of rank 0's k-th reading applies to the k-th reading of every rank.
"""

from __future__ import annotations

import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from types import FunctionType, ModuleType, SimpleNamespace
from typing import Callable, Mapping, NamedTuple

MIB = 1 << 20
LIMIT_MIB = 96_894
READING_RE = re.compile(
    r"GLM53_GS_SATURATION rows=(\d+) context=(\d+) saturated=\d covered=\d measured_mib=(\d+) charged_mib=\d+ "
    r"headroom_mib=(-?\d+) graphs=(\d+) runtime=(\d+) .*?ms=([\d.]+) tripwire_left=\d+ pid=(\d+)")
SEAL_RE = re.compile(r"GLM53_GS_SATURATION event=seal .*pid=(\d+)")
TRIP_RE = re.compile(r"GLM53_GS_SATURATION event=tripwire rows=(\d+) context=(\d+) ms=([\d.]+) .*pid=(\d+)")
NOISE_RE = re.compile(r" (?:pid|tp_min_ms|check_ms_max)=[\d.]+")  # the pid, and the wall time of a check collective: not decisions
STEP_RE = re.compile(r"GLM53_STEP_STATS worker-step (\{.*\})")
R0_ENV: Mapping[str, str] = {
    "HPU_GLM53_GS_SATURATED_PEAK": "1", "HPU_GLM53_GS_REAL_NEED_PEAK_MIB": "2420",
    "HPU_GLM53_GS_REAL_NEED_MAX_ROWS": "2048", "HPU_GLM53_GS_REAL_NEED_MAX_CONTEXT": "131072",
    "HPU_GLM53_GS_RUNAHEAD_CEILING_MIB": "6144", "HPU_GLM53_GS_SATURATION_MARGIN_MIB": "256",
}


class Forward(NamedTuple):
    """One guarded prompt forward as the log recorded it."""

    rows: int
    context: int
    measured_mib: int
    headroom_mib: int
    graphs: int
    runtime: int
    ms: float
    captured: bool = False


@dataclass
class RankLog:
    pid: int
    warmup: list[Forward] = field(default_factory=list)
    serving: list[Forward] = field(default_factory=list)
    trips: list[tuple[int, int, float]] = field(default_factory=list)
    sealed: bool = False


@dataclass
class ReplayResult:
    lines: list[str]
    trips: list[tuple[int, int, float]]
    capture_trips: int = 0
    other_trips: int = 0
    covered: int = 0
    held: int = 0
    releases: int = 0  # drive_checks only: the releases the real route would have run
    trip_headrooms: list[int] = field(default_factory=list)  # headroom MiB of each tripped reading, in trip order
    guard: object | None = None

    @property
    def held_share(self) -> float:
        return self.held / max(1, self.covered)


class FakeAllocator:
    """The habana counters of the reading being replayed."""

    def __init__(self) -> None:
        self.in_use, self.top = 0, 0

    def load(self, forward: Forward) -> None:
        self.top = (LIMIT_MIB - forward.headroom_mib) * MIB
        self.in_use = self.top - forward.measured_mib * MIB

    def stats(self) -> dict[str, int]:
        return {"InUse": self.in_use, "MaxInUse": self.top, "Limit": LIMIT_MIB * MIB}

    def reset_peak(self) -> None:
        return None


def build_guard(module: ModuleType, environ: Mapping[str, str], allocator: FakeAllocator, cache: dict,
                free_bytes: Callable[[], float] | None = None) -> object:
    """The real hbm_guard on a fake adapter: every flag reaches the guard exactly as it does in serving.
    ``free_bytes``: what HabanaMemoryProfiler reads (default 0, which no replay without drive_checks looks at)."""
    hpu = SimpleNamespace(memory_stats=allocator.stats, reset_peak_memory_stats=allocator.reset_peak,
                          synchronize=lambda: None)
    namespace = {"htorch": SimpleNamespace(core=SimpleNamespace(mark_step=lambda sync=False: None), hpu=hpu),
                 "HabanaMemoryProfiler": SimpleNamespace(current_free_device_memory=free_bytes or (lambda: 0.0))}

    def clear_cache() -> None:
        cache.clear()

    forward = FunctionType((lambda *args, **kwargs: None).__code__, namespace)  # its __globals__ are the fake runner
    return module.hbm_guard(SimpleNamespace(forward=forward, clear_cache=clear_cache), environ)


def _parse_forward(match: re.Match[str]) -> Forward:
    groups = match.groups()
    return Forward(int(groups[0]), int(groups[1]), int(groups[2]), int(groups[3]), int(groups[4]), int(groups[5]),
                   float(groups[6]))


def _note_reading(ranks: dict[int, RankLog], pending: dict[int, list[int]], match: re.Match[str]) -> None:
    pid = int(match.group(8))
    rank = ranks.setdefault(pid, RankLog(pid))
    target = rank.serving if rank.sealed else rank.warmup
    target.append(_parse_forward(match))
    if rank.sealed:
        pending.setdefault(pid, []).append(len(target) - 1)


def _note_step(ranks: dict[int, RankLog], pending: dict[int, list[int]], step: dict) -> None:
    rank = ranks.setdefault(step["pid"], RankLog(step["pid"]))
    captured = step["hpu_graph"]["capture"] > 0
    for index in pending.pop(step["pid"], []):
        rank.serving[index] = rank.serving[index]._replace(captured=captured)


def read_extract(path: Path) -> dict[int, RankLog]:
    """Per pid, in log order: warmup and serving readings (rank 0's with the capture flag of their step), and trips."""
    ranks: dict[int, RankLog] = {}
    pending: dict[int, list[int]] = {}
    with path.open(errors="replace") as lines:
        for line in lines:
            if (match := READING_RE.search(line)) is not None:
                _note_reading(ranks, pending, match)
            elif (match := SEAL_RE.search(line)) is not None:
                ranks.setdefault(int(match.group(1)), RankLog(int(match.group(1)))).sealed = True
            elif (match := TRIP_RE.search(line)) is not None:
                ranks.setdefault(int(match.group(4)), RankLog(int(match.group(4)))).trips.append(
                    (int(match.group(1)), int(match.group(2)), float(match.group(3))))
            elif (match := STEP_RE.search(line)) is not None:
                _note_step(ranks, pending, json.loads(match.group(1)))
    return ranks


def share_captures(ranks: dict[int, RankLog]) -> None:
    """Copy rank 0's capture flags to the other ranks, refusing logs whose forwards do not line up."""
    first = next((rank for rank in ranks.values() if any(forward.captured for forward in rank.serving)), None)
    if first is None:
        raise ValueError("no rank has GLM53_STEP_STATS captures in this extract; extract the worker-step lines too")
    shapes = [(forward.rows, forward.context) for forward in first.serving]
    for rank in ranks.values():
        common = min(len(shapes), len(rank.serving))
        if [(forward.rows, forward.context) for forward in rank.serving[:common]] != shapes[:common]:
            raise ValueError(f"pid {rank.pid}: readings do not line up with rank 0; cannot share capture flags")
        rank.serving[:common] = [forward._replace(captured=source.captured)
                                 for forward, source in zip(rank.serving[:common], first.serving[:common])]


def _sync_runtime(cache: dict, boot: int, runtime: int) -> None:
    """Graph count from the log: the replay is open loop."""
    keys = list(cache)
    for key in keys[boot + runtime:]:
        del cache[key]
    for index in range(len(cache), boot + runtime):
        cache[f"runtime-{index}"] = object()


def _graph_forward(guard: object, cache: dict, captured: bool, index: int) -> bool:
    """The decode graph forward that precedes the prompt forward of a mixed step, through the real entry points."""
    shape = ("capture", index) if captured else ("replay",)
    if not captured:
        guard.shapes.add(shape)
    step = guard.before_graph(shape)
    if captured and not getattr(guard, "witness", False):  # a witness guard refuses a serving capture (test_guard_witness.WitnessGrowth), so a log's never grows it
        cache[f"captured-{index}"] = object()
    return guard.after_graph(shape, step, 1)


def _prompt_forward(guard: object, forward: Forward) -> None:
    ticks = iter((0.0, forward.ms / 1000.0))
    guard.clock = lambda: next(ticks)
    guard.prompt(lambda: None, False, forward.rows, forward.context)


def _record(result: ReplayResult, guard: object, forward: Forward, trips_before: int) -> None:
    if forward.rows <= guard.saturation.max_rows and forward.context <= guard.saturation.max_context:
        result.covered += 1
        result.held += int(guard.tripwire_left > 0)
    if guard.saturation_counts["tripwires"] > trips_before:
        result.trips.append((forward.rows, forward.context, forward.ms))
        result.trip_headrooms.append(forward.headroom_mib)
        result.capture_trips += int(forward.captured)
        result.other_trips += int(not forward.captured)


def _checked_graph_forward(guard: object, cache: dict, forward: Forward, index: int, result: ReplayResult) -> None:
    """RoutedForward._start + _graph: a release left pending runs first; a capture below the floor leaves one pending."""
    if guard.pending:
        guard.release()
        result.releases += 1
    module = sys.modules[type(guard).__module__]
    if _graph_forward(guard, cache, forward.captured, index) and not guard.check(module.CAPTURE):
        guard.pending = True


def _checked_prompt_forward(guard: object, forward: Forward, result: ReplayResult) -> None:
    """RoutedForward.__call__ for a guarded eager chunk: check first, release before the chunk when the check says so."""
    module = sys.modules[type(guard).__module__]
    chunk = module.Chunk(forward.rows, forward.context)
    if not guard.check(module.EAGER, chunk):
        guard.release(chunk)
        result.releases += 1
    _prompt_forward(guard, forward)


def replay(module: ModuleType, rank: RankLog, environ: Mapping[str, str], *, drive_checks: bool = False) -> ReplayResult:
    """Drive one rank's log through ``module``'s guard; every line the guard logs is kept, pid and collective wall times removed.

    ``drive_checks``: also take the route's check points (the free HBM of a check is the log reading's Limit - InUse), so releases and
    GLM53_GS_HBM lines are replayed too. Off, the replay is the tripwire's alone.
    """
    result = ReplayResult(lines=[], trips=[])
    module.log_line = lambda line: result.lines.append(NOISE_RE.sub("", line))
    allocator = FakeAllocator()
    boot = rank.serving[0].graphs - rank.serving[0].runtime if rank.serving else 0
    cache: dict = {f"boot-{index}": object() for index in range(boot)}
    guard = build_guard(module, environ, allocator, cache, lambda: float(allocator.stats()["Limit"] - allocator.in_use))
    for forward in rank.warmup:
        allocator.load(forward)
        _prompt_forward(guard, forward)
    guard.seal_boot()
    for index, forward in enumerate(rank.serving):
        allocator.load(forward)
        if drive_checks:
            _checked_graph_forward(guard, cache, forward, index, result)
        else:
            _graph_forward(guard, cache, forward.captured, index)
        _sync_runtime(cache, boot, forward.runtime)
        before = guard.saturation_counts["tripwires"]
        if drive_checks:
            _checked_prompt_forward(guard, forward, result)
        else:
            _prompt_forward(guard, forward)
        _record(result, guard, forward, before)
    result.guard = guard
    return result
