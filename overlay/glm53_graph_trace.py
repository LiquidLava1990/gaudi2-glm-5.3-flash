"""Device trace of one warmed graph replay (DUAL-SUSTAINED-PLAN.md 12.3a, stage C1p). Off by default; unset, nothing imports this file.

HPU_GLM53_REPLAY_TRACE_DIR=<absolute container directory> arms it (the cell's --mount, /k1trace) and needs
HPU_GLM53_PREFILL_HBM_CENSUS=1, because glm53_prompt_census.census_graph_forward is the one caller: it hands every key forward to
``ReplayTracer.wrap``. HPU_GLM53_REPLAY_TRACE names what to trace, comma separated ``KIND:ROWSxWIDTH@NTH``:

    prompt:1024x8192@3     the 3rd warmed replay of the prompt key (rows 1024, table width 8192)
    decode:16x8192@20      the 20th warmed replay of the decode key (16 rows, table width 8192)

A replay is a call that did not capture a graph: a prompt key's own ``key_replay`` flag, or for a decode key (the graph state does not
know) a call after which the graph count has not grown. The ``HPU_GLM53_REPLAY_TRACE_REHEARSALS`` (default 2) replays before the
traced one are bracketed exactly like it, without the profiler: their wall is the unprofiled twin the host report reconciles against.
Every bracketed replay is mark_step + synchronize, the replay, mark_step + synchronize; the traced one opens the torch profiler (CPU +
HPU, no shapes, no stacks: the worker's own recipe, C820) and a MARKER span after the first drain and closes both after the second.

Buffer. The prior tool (glm53_forward_trace, B2 r6) never set a profiler buffer: it cut an eager forward into slices at its
_boundary calls, because a device record of 1.4-1.5 M kernels keeps only the last 0.5-0.7 s and loses the window's start, and a long
window can exhaust HBM outright (KB C341: ~1,000 steps of 346-423 k events each killed the engine). A graph replay is one launch with no
host boundary to cut at, so this hook sets no buffer either. It bounds the damage instead: it refuses to arm below MIN_FREE_MIB of free
HBM (the C341 floor), traces each target once, and the host report (cells/replay_trace_report.py) says how much of the window survived.
Whether the 3.85 s wide replay survives whole is a card reading.

Every rank brackets (symmetric overhead), only rank 0 exports a trace; each rank writes ``replaytrace-<run_id>-rank<R>.json`` last, in the
window-file schema cells/trace_verdict.py reads (a replay's "tokens" are its rows), so its presence means the trace is complete.
Torch is imported only by HpuRuntime.
"""
from __future__ import annotations

import atexit
import functools
import json
import logging
import os
import re
import threading
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Protocol, TypeVar

DIR_ENV = "HPU_GLM53_REPLAY_TRACE_DIR"
PLAN_ENV = "HPU_GLM53_REPLAY_TRACE"
REHEARSALS_ENV = "HPU_GLM53_REPLAY_TRACE_REHEARSALS"
CENSUS_ENV = "HPU_GLM53_PREFILL_HBM_CENSUS"
DEFAULT_REHEARSALS = 2
MAX_REHEARSALS = 8
MIN_FREE_MIB = 1024  # KB C341: a device-trace window that would leave less than 1 GiB free is refused
EXPORT_RANKS = (0,)
FILE_PREFIX = "replaytrace-"
MARKER = "glm53_fwdtrace_window"  # = trace_verdict.MARKER
SCHEMA = "glm53-replaytrace-window/1"
TARGET = re.compile(r"(?P<kind>prompt|decode):(?P<rows>[1-9]\d*)x(?P<width>[1-9]\d*)@(?P<nth>[1-9]\d*)")
GRAMMAR = "KIND:ROWSxWIDTH@NTH, e.g. prompt:1024x8192@3,decode:16x8192@20 (KIND prompt or decode, NTH a positive integer)"
MIB = 1 << 20
ERROR_CHARS = 300
LOG = logging.getLogger("vllm.replaytrace")
T = TypeVar("T")


class ReplayTraceError(ValueError):
    """The flag or the plan is malformed; the message names the setting and the fix."""


# ---- the plan: what to trace -----------------------------------------------------------------------------------------
@dataclass(frozen=True)
class Target:
    kind: str
    rows: int
    width: int
    nth: int

    @property
    def key(self) -> tuple[str, int, int]:
        return self.kind, self.rows, self.width

    @property
    def run_id(self) -> str:
        return f"{self.kind}-{self.rows}x{self.width}-n{self.nth}"

    def rehearsal_replays(self, rehearsals: int) -> list[int]:
        return list(range(self.nth - rehearsals, self.nth))


@dataclass(frozen=True)
class TracePlan:
    directory: str
    targets: tuple[Target, ...]
    rehearsals: int


def window_name(run_id: str, rank: int) -> str:
    return f"{FILE_PREFIX}{run_id}-rank{rank}.json"


def trace_name(run_id: str, rank: int) -> str:
    return f"{FILE_PREFIX}{run_id}-rank{rank}.pt.trace.json.gz"


def _rehearsal_count(text: str) -> int:
    if not text.strip().isdigit() or not 0 <= int(text) <= MAX_REHEARSALS:
        raise ReplayTraceError(f"{REHEARSALS_ENV}={text!r}: an integer 0..{MAX_REHEARSALS} (replays bracketed untraced before the traced one)")
    return int(text)


def _parse_target(item: str, rehearsals: int) -> Target:
    match = TARGET.fullmatch(item)
    if match is None:
        raise ReplayTraceError(f"{PLAN_ENV} entry {item!r}: expected {GRAMMAR}")
    target = Target(match["kind"], int(match["rows"]), int(match["width"]), int(match["nth"]))
    if target.nth <= rehearsals:
        raise ReplayTraceError(f"{PLAN_ENV} entry {item!r}: replay {target.nth} leaves no room for {rehearsals} rehearsals before it; "
                               f"use NTH >= {rehearsals + 1} or lower {REHEARSALS_ENV}")
    if target.kind == "decode" and target.nth < 2:
        raise ReplayTraceError(f"{PLAN_ENV} entry {item!r}: a decode key is warmed after its first call, which may be its capture; use NTH >= 2")
    return target


def parse_plan(directory: str, spec: str, rehearsals_text: str) -> TracePlan:
    """The plan from the three settings, validated; ReplayTraceError names the one to fix."""
    if not directory or not (PurePosixPath(directory).is_absolute() or os.path.isabs(directory)):  # os.path: a Windows dev host
        raise ReplayTraceError(f"{DIR_ENV}={directory!r}: must be an absolute container directory (the cell's --mount, /k1trace); "
                               "unset it to run without the hook")
    rehearsals = _rehearsal_count(rehearsals_text)
    items = [item.strip() for item in spec.split(",") if item.strip()]
    if not items:
        raise ReplayTraceError(f"{PLAN_ENV}={spec!r} names no key: expected {GRAMMAR}")
    targets = tuple(_parse_target(item, rehearsals) for item in items)
    seen: set[tuple[str, int, int]] = set()
    for target in targets:
        if target.key in seen:
            raise ReplayTraceError(f"{PLAN_ENV} names {target.kind}:{target.rows}x{target.width} twice: each key is traced once per process")
        seen.add(target.key)
    return TracePlan(directory, targets, rehearsals)


def plan_from_env(env: Mapping[str, str]) -> TracePlan | None:
    """None while HPU_GLM53_REPLAY_TRACE_DIR is unset or blank; else the validated plan."""
    directory = env.get(DIR_ENV, "").strip()
    if not directory:
        return None
    if not env.get(PLAN_ENV, "").strip():
        raise ReplayTraceError(f"{DIR_ENV} is set but {PLAN_ENV} is not: set it to {GRAMMAR}, or unset {DIR_ENV}")
    return parse_plan(directory, env[PLAN_ENV], env.get(REHEARSALS_ENV, str(DEFAULT_REHEARSALS)))


def describe_plan(plan: TracePlan, ranks: int = 4) -> dict[str, object]:
    """What a run would trace and write, for a dry run (touches nothing)."""
    targets = []
    for target in plan.targets:
        files = [window_name(target.run_id, rank) for rank in range(ranks)] + [trace_name(target.run_id, rank) for rank in EXPORT_RANKS]
        targets.append({"run_id": target.run_id, "kind": target.kind, "rows": target.rows, "width": target.width,
                        "traced_replay": target.nth, "rehearsal_replays": target.rehearsal_replays(plan.rehearsals), "files": files})
    return {"directory": plan.directory, "rehearsals": plan.rehearsals, "min_free_mib": MIN_FREE_MIB, "export_ranks": list(EXPORT_RANKS),
            "targets": targets,
            "needs": [f"{CENSUS_ENV}=1 (the census wrapper is the only caller)", f"{plan.directory} mounted from the host",
                      "decode targets: the graph-state patch that passes decode_width (patch-c1p-decode-width-graph-state.diff)"]}


# ---- the device pieces -----------------------------------------------------------------------------------------------
class Profiler(Protocol):
    def start(self) -> None: ...

    def stop(self) -> None: ...

    def export_chrome_trace(self, path: str) -> object: ...


class Marker(Protocol):
    def __enter__(self) -> object: ...

    def __exit__(self, *exc: object) -> object: ...


class Runtime(Protocol):
    """What the hook touches on the device (HpuRuntime in the container, a fake in the tests)."""

    def flush(self) -> None: ...

    def profiler(self) -> Profiler: ...

    def marker(self, name: str) -> Marker: ...

    def rank(self) -> int: ...

    def free_mib(self) -> float | None: ...


class HpuRuntime:
    """Lazy-mode HPU: flush = mark_step + a device drain; the profiler is glm53_forward_trace's recipe (CPU + HPU, no stack, no shapes)."""

    def flush(self) -> None:
        import habana_frameworks.torch.core as htcore
        import torch
        htcore.mark_step()
        torch.hpu.synchronize()

    def profiler(self) -> Profiler:
        import torch
        activities = [torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.HPU]
        return torch.profiler.profile(activities=activities, record_shapes=False, with_stack=False)

    def marker(self, name: str) -> Marker:
        import torch
        return torch.profiler.record_function(name)

    def rank(self) -> int:
        import torch
        if torch.distributed.is_initialized():
            return int(torch.distributed.get_rank())
        return int(os.environ.get("RANK", "-1"))

    def free_mib(self) -> float | None:
        """Allocator limit minus bytes in use, None when the allocator cannot be read (the caller then refuses to arm)."""
        import habana_frameworks.torch as htorch
        try:
            stats = htorch.hpu.memory_stats()
            return (int(stats["Limit"]) - int(stats["InUse"])) / MIB
        except Exception:  # noqa: BLE001 - unreadable is a refusal reason the window file records, never a failed forward
            return None


# ---- one bracketed replay ----------------------------------------------------------------------------------------------
@dataclass
class Window:
    """One bracketed replay: a rehearsal (no profiler) or the target's traced replay."""
    target: Target
    traced: bool
    index: int
    profiler: Profiler | None = None
    marker: Marker | None = None
    profiler_stopped: bool = False
    opened_at: float | None = None
    opened_unix: float | None = None
    closed_at: float | None = None
    profiler_start_s: float | None = None
    profiler_stop_s: float | None = None
    free_mib_before: float | None = None
    free_mib_after: float | None = None
    was_capture: bool = False
    failed: bool = False
    errors: list[str] = field(default_factory=list)

    def window_ms(self) -> float | None:
        if self.opened_at is None or self.closed_at is None:
            return None
        return round((self.closed_at - self.opened_at) * 1e3, 3)

    def summary(self) -> dict[str, object]:
        return {"index": self.index, "window_ms": self.window_ms(), "opened_unix": self.opened_unix, "errors": list(self.errors)}


@dataclass
class TargetState:
    target: Target
    replays: int = 0   # warmed replays completed so far (captures do not count)
    done: bool = False
    rehearsals: list[dict[str, object]] = field(default_factory=list)


class ReplayTracer:
    """The per-process state machine behind ``wrap`` (see the module docstring)."""

    def __init__(self, plan: TracePlan, runtime: Runtime, clock: Callable[[], float] = time.perf_counter,
                 wall: Callable[[], float] = time.time) -> None:
        self.plan, self.runtime, self.clock, self.wall = plan, runtime, clock, wall
        self.directory = Path(plan.directory)
        self.states = {target.key: TargetState(target) for target in plan.targets}
        self.exporter: threading.Thread | None = None

    # ---- the wrapper -------------------------------------------------------------------------------------------------
    def wrap(self, forward: Callable[[], T], *, rows: int, context: int, captures: Callable[[], int], key_replay: bool | None,
             decode_width: int | None) -> Callable[[], T]:
        """``forward`` itself unless (kind, rows, width) is a target. kind prompt: a key forward (key_replay True or False), width =
        context. kind decode: key_replay None with a table width."""
        if key_replay is not None:
            key = ("prompt", rows, context)
        elif decode_width is not None:
            key = ("decode", rows, decode_width)
        else:
            return forward
        state = self.states.get(key)
        if state is None:
            return forward
        return functools.wraps(forward)(functools.partial(self._call, state, forward, captures, key_replay))

    def _call(self, state: TargetState, forward: Callable[[], T], captures: Callable[[], int], key_replay: bool | None) -> T:
        """One call of a target key's forward: bracketed when it is the traced replay or one of its rehearsals, else just counted."""
        if state.done or key_replay is False:
            return forward()
        index, nth = state.replays + 1, state.target.nth
        if index == nth:
            return self._bracketed(state, True, forward, captures, key_replay)
        if nth - self.plan.rehearsals <= index < nth:
            return self._bracketed(state, False, forward, captures, key_replay)
        return self._counted(state, forward, captures, key_replay)

    def _counted(self, state: TargetState, forward: Callable[[], T], captures: Callable[[], int], key_replay: bool | None) -> T:
        before = captures()
        result = forward()
        if key_replay is True or captures() == before:
            state.replays += 1
        return result

    def _bracketed(self, state: TargetState, traced: bool, forward: Callable[[], T], captures: Callable[[], int],
                   key_replay: bool | None) -> T:
        window = Window(state.target, traced, state.replays + 1)
        self._join_exporter()
        self._guarded(window, "first flush", self.runtime.flush)
        if traced:
            self._arm(window)
        window.opened_at, window.opened_unix = self.clock(), self.wall()
        before = captures()
        try:
            result = forward()
            self.runtime.flush()
            window.closed_at = self.clock()
        except BaseException as error:
            window.errors.append(f"forward raised {type(error).__name__}: {error}"[:ERROR_CHARS])
            window.failed = True
            window.closed_at = self.clock()
            raise
        finally:
            window.was_capture = key_replay is False or captures() != before
            self._guarded(window, "profiler stop", lambda: self._disarm(window))
            self._guarded(window, "window finish", lambda: self._finish(state, window))
        return result

    # ---- the profiler ------------------------------------------------------------------------------------------------
    def _arm(self, window: Window) -> None:
        free = self.runtime.free_mib()
        window.free_mib_before = free
        if free is None:
            window.errors.append(f"free HBM could not be read: profiler not armed (the floor is {MIN_FREE_MIB} MiB, KB C341)")
            return
        if free < MIN_FREE_MIB:
            window.errors.append(f"free HBM {free:.0f} MiB is below the {MIN_FREE_MIB} MiB floor (KB C341): profiler not armed")
            return
        started = self.clock()
        self._guarded(window, "profiler start", lambda: self._start_profiler(window))
        window.profiler_start_s = round(self.clock() - started, 6)

    def _start_profiler(self, window: Window) -> None:
        profiler = self.runtime.profiler()
        profiler.start()
        window.profiler = profiler
        marker = self.runtime.marker(MARKER)
        marker.__enter__()
        window.marker = marker

    def _disarm(self, window: Window) -> None:
        marker, window.marker = window.marker, None
        if marker is not None:
            marker.__exit__(None, None, None)
        if window.profiler is not None and not window.profiler_stopped:
            started = self.clock()
            window.profiler_stopped = True
            window.profiler.stop()
            window.profiler_stop_s = round(self.clock() - started, 6)
        if window.traced:
            window.free_mib_after = self.runtime.free_mib()

    def _guarded(self, window: Window, what: str, action: Callable[[], None]) -> None:
        """Profiler trouble is recorded on the window (the report says so), never raised into serving."""
        try:
            action()
        except Exception as error:  # noqa: BLE001 - recorded on the window and logged, never swallowed
            window.errors.append(f"{what}: {type(error).__name__}: {error}"[:ERROR_CHARS])
            LOG.warning("GLM53_REPLAY_TRACE %s failed: %s", what, error)

    # ---- after the replay --------------------------------------------------------------------------------------------
    def _finish(self, state: TargetState, window: Window) -> None:
        if window.traced and window.was_capture:
            window.errors.append("the traced call grew the graph count: it was a capture, not a warmed replay")
        counts = not window.was_capture and not window.failed
        state.replays += counts
        if not window.traced:
            if counts and not window.errors:
                state.rehearsals.append(window.summary())
                LOG.warning("GLM53_REPLAY_TRACE rehearsal run=%s %s", window.target.run_id, json.dumps(window.summary(), sort_keys=True))
            return
        state.done = True
        record = self._record(state, window)
        exported = window.profiler if self.runtime.rank() in EXPORT_RANKS and window.profiler is not None else None
        self.exporter = threading.Thread(target=self._export_and_write, args=(record, exported), name="replaytrace-export", daemon=True)
        self.exporter.start()

    def _record(self, state: TargetState, window: Window) -> dict[str, object]:
        target = window.target
        return {"schema": SCHEMA, "run_id": target.run_id, "rank": self.runtime.rank(), "pid": os.getpid(), "marker": MARKER,
                "trigger": {"run_id": target.run_id, "rehearsals": self.plan.rehearsals, "start": 0, "stop": None,
                            "min_tokens": target.rows, "export_ranks": list(EXPORT_RANKS)},
                "kind": target.kind, "rows": target.rows, "width": target.width, "replay_index": window.index,
                "tokens": target.rows, "padded_tokens": target.rows, "seq_lens": None, "shape": [target.rows], "request_ids": [],
                "boundaries": 0, "time_unix": self.wall(), "opened": window.opened_at is not None,
                "opened_profiler": window.profiler is not None, "opened_unix": window.opened_unix,
                "profiler_start_s": window.profiler_start_s, "profiler_stop_s": window.profiler_stop_s,
                "window_ms": window.window_ms(), "free_mib_before": window.free_mib_before, "free_mib_after": window.free_mib_after,
                "errors": list(window.errors), "rehearsals": list(state.rehearsals), "trace": None, "export_s": None}

    def _export_and_write(self, record: dict[str, object], profiler: Profiler | None) -> None:
        name = trace_name(str(record["run_id"]), int(record["rank"]))  # type: ignore[call-overload]
        if profiler is not None:
            started = self.clock()
            try:
                profiler.export_chrome_trace(str(self.directory / name))
                record["trace"], record["export_s"] = name, round(self.clock() - started, 3)
            except Exception as error:  # noqa: BLE001 - recorded in the window file, which the report reads
                record["errors"].append(f"trace export: {type(error).__name__}: {error}"[:ERROR_CHARS])  # type: ignore[union-attr]
        path = self.directory / window_name(str(record["run_id"]), int(record["rank"]))  # type: ignore[call-overload]
        try:
            write_json_atomic(path, record)
        except OSError as error:
            LOG.warning("GLM53_REPLAY_TRACE window file %s not written (%s); the host report will time out on it", path, error)
            return
        LOG.warning("GLM53_REPLAY_TRACE done run=%s rank=%s window_ms=%s trace=%s errors=%d", record["run_id"], record["rank"],
                    record["window_ms"], record["trace"], len(record["errors"]))  # type: ignore[arg-type]

    def _join_exporter(self) -> None:
        """Never open a window while the previous export is still running."""
        exporter, self.exporter = self.exporter, None
        if exporter is not None:
            exporter.join()

    def wait_exports(self) -> None:
        self._join_exporter()

    def unfired(self) -> list[str]:
        """One line per target never traced, with the warmed replays seen: what the exit log shows when a key never reached its NTH."""
        return [f"run={state.target.run_id} replays_seen={state.replays} needed={state.target.nth}"
                for state in self.states.values() if not state.done]

    def log_unfired(self) -> None:
        for line in self.unfired():
            LOG.warning("GLM53_REPLAY_TRACE unfired %s: the key never reached its replay (check the key name, the workload, and for decode "
                        "the graph-state patch)", line)


def write_json_atomic(path: Path, record: dict[str, object]) -> None:
    staging = path.with_name(f".{path.name}.part")
    staging.write_text(json.dumps(record, sort_keys=True, indent=1) + "\n", encoding="utf-8")
    os.replace(staging, path)


def install_from_env(env: Mapping[str, str] | None = None, runtime: Runtime | None = None) -> ReplayTracer | None:
    """The process's tracer for the environment, None while the flag is unset. A bad plan or a missing directory fails the import, so a
    bad arm never boots and never traces into a directory the host cannot see."""
    plan = plan_from_env(os.environ if env is None else env)
    if plan is None:
        return None
    if not Path(plan.directory).is_dir():
        raise ReplayTraceError(f"{DIR_ENV}={plan.directory}: not a directory in this container; mount the host folder there")
    tracer = ReplayTracer(plan, runtime if runtime is not None else HpuRuntime())
    atexit.register(tracer.log_unfired)
    LOG.warning("GLM53_REPLAY_TRACE armed dir=%s targets=%s rehearsals=%d", plan.directory, ",".join(t.run_id for t in plan.targets),
                plan.rehearsals)
    return tracer
