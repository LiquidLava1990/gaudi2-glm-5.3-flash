"""Per-op HBM census of eager prompt and captured graph forwards (plan step 20). No extra mark_step or synchronize.

HPU_GLM53_PREFILL_HBM_CENSUS=1 logs GLM53_PROMPT_CENSUS JSON lines, per rank: `path` eager or graph, the latter `kind`
capture or replay, with the bytes each mark_step call site materialized. Allocator reads are host-side.
cells/census_table.py turns the lines into the per-cell tables, need-table.json and the verdicts.

Eager forwards never touch the peak: the guard's window has just started. A graph forward has no window of its own, so
the flag restarts the peak at its start and reports the old peak as `between_peak`, the high water since the previous
forward. That is where bytes a replay allocates after returning show up, so it is the graph path's run-ahead reading.
With the flag on, the guard's next capture peak no longer includes earlier replays' transients; that peak is diagnostic
in these runs. Boot captures of mixed graphs do not pass the routed forward: dual's HbmProbe logs them, using this class.

MarkStepCensus is the shared home of dual-4m's class (glm53_mixed_graph.py there, absent from base 1481bb99). The step 5
rebase must import it from here and drop its copy.
"""
from __future__ import annotations

import faulthandler
import functools
import json
import logging
import os
import sys
import time
from collections import Counter
from collections.abc import Callable
from contextlib import nullcontext
from types import FrameType, TracebackType
from typing import Any, TypeVar

FLAG_ENV = "HPU_GLM53_PREFILL_HBM_CENSUS"
STALL_DIR_ENV = "HPU_GLM53_K1_STALL_DIR"
REPLAY_TRACE_ENV = "HPU_GLM53_REPLAY_TRACE_DIR"
STACK_DUMP_EVERY_S = 60
TRACED_COLLECTIVES = ("all_reduce", "all_gather", "all_gather_into_tensor", "all_gather_object", "broadcast", "barrier",
                      "reduce_scatter_tensor", "all_to_all", "gather", "scatter")
LINE_TAG = "GLM53_PROMPT_CENSUS"
LOG = logging.getLogger("vllm.r589")
T = TypeVar("T")

OTHER_OP = "other"
BOUNDARY_FUNCTION = "_boundary"
GRAPH_LOG_FIRST = 16  # replays log their first lines per shape, then each new peak and every GRAPH_LOG_EVERY-th
GRAPH_LOG_EVERY = 100
GRAPH_LOG: dict[tuple, list[int]] = {}  # (kind, rows, context[, prompts of a shared key]) -> [forwards, largest peak, most real prompts]
FRAME_WALK_LIMIT = 24  # shared helpers sit a few frames under the op; the model's own frames carry no op file
# Nearest frame wins: the indexer calls into the k-pool and both call Q4 fetches.
OP_BY_FILE = {
    "hpu_glm5next_q4kv.py": "sparse_mla",
    "hpu_glm5next_owner_mla.py": "sparse_mla",
    "oot_mla.py": "sparse_mla",
    "hpu_glm5next_kpool.py": "kpool",
    "hpu_glm5next_sparse_indexer.py": "indexer",
    "hpu_glm5next_kda.py": "kda",
    "glm53_kda_batch_prefill.py": "kda",
    "hpu_fp8.py": "moe",
    "glm53_moe_device_plan.py": "moe",
}
OP_BY_BOUNDARY_PREFIX = {"moe": "moe", "kda": "kda"}
OP_BY_FUNCTION = {"_gather_query_shard": "indexer"}  # the query-shard gather runs in Q4's helper but is the indexer's collective: it gets its own site


def census_enabled() -> bool:
    value = os.environ.get(FLAG_ENV, "0").strip()
    if value not in ("", "0", "1"):
        raise ValueError(f"{FLAG_ENV}={value!r}: set 1 to log per-call-site HBM growth of eager prompt forwards, or "
                         "0 / unset for none")
    return value == "1"


census_enabled()  # import-time parse: a bad value fails every worker's boot


def _load_replay_tracer() -> Any | None:
    """The process's replay tracer (glm53_graph_trace, stage C1p), or None without importing it while its flag is unset."""
    if not os.environ.get(REPLAY_TRACE_ENV, "").strip():
        return None
    if not census_enabled():
        raise ValueError(f"{REPLAY_TRACE_ENV} traces the replays this census wraps: set {FLAG_ENV}=1, or unset {REPLAY_TRACE_ENV}")
    from glm53_graph_trace import install_from_env

    return install_from_env()


REPLAY_TRACER = _load_replay_tracer()


def _runtime() -> tuple[Any, Any]:
    import habana_frameworks.torch as htorch

    return htorch.core, htorch.hpu


def _dist() -> Any:
    import torch.distributed

    return torch.distributed


def _boundary_site(frame: FrameType, site: str) -> tuple[str, str]:
    """All R589 boundaries share one mark_step line: their label is what tells the tiles apart."""
    label = str(frame.f_locals.get("label", ""))
    return f"{site}[{label}]", OP_BY_BOUNDARY_PREFIX.get(label.split(".", 1)[0], OTHER_OP)


def _outranking_function(frame: FrameType) -> str | None:
    """The OP_BY_FUNCTION caller of a shared helper's frame, or None."""
    walker: FrameType | None = frame
    for _ in range(FRAME_WALK_LIMIT):
        if walker is None:
            return None
        if walker.f_code.co_name in OP_BY_FUNCTION:
            return walker.f_code.co_name
        walker = walker.f_back
    return None


def site_of(frame: FrameType) -> tuple[str, str]:
    """(call site, op) of a mark_step caller. A shared helper's site names the nearest op file that called it."""
    filename = os.path.basename(frame.f_code.co_filename)
    site = f"{filename}:{frame.f_lineno}"
    if frame.f_code.co_name == BOUNDARY_FUNCTION:
        return _boundary_site(frame, site)
    walker: FrameType | None = frame
    for _ in range(FRAME_WALK_LIMIT):
        if walker is None:
            break
        name = os.path.basename(walker.f_code.co_filename)
        if name in OP_BY_FILE:
            function = _outranking_function(walker)
            if function is not None:
                return f"{site}<{function}", OP_BY_FUNCTION[function]
            return (site if walker is frame else f"{site}<{name}:{walker.f_lineno}"), OP_BY_FILE[name]
        walker = walker.f_back
    return site, OTHER_OP


class StallTrace:
    """HPU_GLM53_K1_STALL_DIR=<dir>: during a key capture each rank writes its marks and Python collectives to its own file.

    Unbuffered, so the last line names where a stalled rank sits; faulthandler dumps every thread into the same file every
    STACK_DUMP_EVERY_S, from a C thread that needs no GIL. Log only. Comparing the files finds a rank-divergent sequence.
    """

    def __init__(self, directory: str, rows: int, context: int) -> None:
        self.directory, self.rows, self.context = directory, rows, context
        self.started = time.perf_counter()
        self.patched: list[tuple[str, Callable[..., Any]]] = []

    def __enter__(self) -> StallTrace:
        path = os.path.join(self.directory, f"k1stall-{os.getpid()}.txt")
        self.file = open(path, "a", buffering=1)
        dist = _dist()
        rank = dist.get_rank() if dist.is_initialized() else -1
        self.file.write(f"# rank={rank} pid={os.getpid()} rows={self.rows} context={self.context}\n")
        self.line("begin")
        faulthandler.dump_traceback_later(STACK_DUMP_EVERY_S, repeat=True, file=self.file)
        self._wrap_collectives(dist)
        return self

    def __exit__(self, kind: type[BaseException] | None, error: BaseException | None, trace: TracebackType | None) -> None:
        dist = _dist()
        for name, original in self.patched:
            setattr(dist, name, original)
        faulthandler.cancel_dump_traceback_later()
        self.line("end", error=repr(error))
        self.file.close()

    def line(self, event: str, **fields: object) -> None:
        details = " ".join(f"{key}={value}" for key, value in fields.items())
        self.file.write(f"{time.perf_counter() - self.started:.3f} {event} {details}\n")

    def _wrap_collectives(self, dist: Any) -> None:
        for name in TRACED_COLLECTIVES:
            original = getattr(dist, name, None)
            if original is not None:
                self.patched.append((name, original))
                setattr(dist, name, self._traced(name, original))

    def _traced(self, name: str, original: Callable[..., Any]) -> Callable[..., Any]:
        def traced(*args: object, **kwargs: object) -> object:
            shape = getattr(args[0], "shape", None) if args else None
            self.line("coll.enter", name=name, shape=None if shape is None else tuple(shape))
            result = original(*args, **kwargs)
            self.line("coll.exit", name=name)
            return result
        return traced


def stall_trace(rows: int, context: int, key_capture: bool) -> StallTrace | None:
    """A trace for a key capture when the directory is set; None otherwise."""
    directory = os.environ.get(STALL_DIR_ENV, "").strip()
    return StallTrace(directory, rows, context) if key_capture and directory else None


class MarkStepCensus:
    """Allocated bytes added between consecutive Python mark_step calls, keyed by call site.

    Tensors materialize at mark_step, so the growth since the previous call lands on the site that ended the segment.
    peak_advance is how far that segment moved MaxInUse; the peak is never reset here. A failed read is counted and
    costs only that sample: the forward is never touched.
    """

    def __init__(self, core: Any, hpu: Any) -> None:
        self.core, self.hpu = core, hpu
        self.growth: Counter[str] = Counter()
        self.calls: Counter[str] = Counter()
        self.max_growth: Counter[str] = Counter()
        self.peak_advance: Counter[str] = Counter()
        self.ops: dict[str, str] = {}
        self.read_errors = 0
        self.sync_true = 0
        self.trace: StallTrace | None = None

    def _read(self) -> tuple[int, int, int] | None:
        """(InUse, MaxInUse, Limit), or None when the allocator cannot be read."""
        try:
            stats = self.hpu.memory_stats()
            return int(stats["InUse"]), int(stats["MaxInUse"]), int(stats["Limit"])
        except Exception as error:  # noqa: BLE001  log only: a diagnostic must never fail a forward
            self.read_errors += 1
            if self.read_errors == 1:
                LOG.warning("%s read_error error=%r: this forward's census is partial; check htorch.hpu.memory_stats() "
                            "in this image", LINE_TAG, error)
            return None

    def __enter__(self) -> MarkStepCensus:
        self.original = self.core.mark_step
        self.first = self.last = self._read()
        self.core.mark_step = self.record
        return self

    def __exit__(self, kind: type[BaseException] | None, error: BaseException | None, trace: TracebackType | None) -> None:
        self.core.mark_step = self.original

    def record(self, *args: object, **kwargs: object) -> object:
        """Stands in for mark_step: the real call first, then the bytes it materialized, keyed by the caller."""
        if self.trace is not None:
            self.trace.line("mark.enter", site=site_of(sys._getframe(1))[0], sync=kwargs.get("sync", args[0] if args else None))
        result = self.original(*args, **kwargs)
        if self.trace is not None:
            self.trace.line("mark.exit")
        self.sync_true += bool(kwargs.get("sync", args[0] if args else False))
        site, op = site_of(sys._getframe(1))
        self.calls[site] += 1
        self.ops[site] = op
        now = self._read()
        if now is not None:
            self._charge(site, now)
        return result

    def _charge(self, site: str, now: tuple[int, int, int]) -> None:
        if self.last is not None:
            grown = now[0] - self.last[0]
            self.growth[site] += grown
            self.max_growth[site] = max(self.max_growth[site], grown)
            self.peak_advance[site] += max(0, now[1] - self.last[1])
        self.first = self.first or now
        self.last = now

    def summary(self, limit: int = 12) -> str:
        """One line for a capture log (dual-4m's HbmProbe format)."""
        end = self._read() or self.last or (0, 0, 0)
        start = self.first or end
        last = self.last or end
        top = sorted(self.growth, key=lambda site: -abs(self.growth[site]))[:limit]
        sites = ",".join(f"{site}:{self.calls[site]}:{self.growth[site]}" for site in top) or "-"
        return (f"mark_steps={sum(self.calls.values())} growth_bytes={end[0] - start[0]} "
                f"unattributed_bytes={end[0] - last[0]} top_sites={sites}")

    def report(self, rows: int, context: int) -> dict[str, object]:
        """The GLM53_PROMPT_CENSUS body. Sizes are bytes; None where the allocator could not be read."""
        end = self._read()
        return {"rows": rows, "context": context, "pid": os.getpid(), "mark_steps": sum(self.calls.values()),
                "read_errors": self.read_errors, "sync_true": self.sync_true,
                "start_in_use": self.first[0] if self.first else None,
                "end_in_use": end[0] if end else None,
                "max_in_use": end[1] if end else None,
                "limit": end[2] if end else None,
                "unattributed": end[0] - self.last[0] if end and self.last else None,
                "sites": [{"site": site, "op": self.ops[site], "calls": self.calls[site], "growth": self.growth[site],
                           "max_growth": self.max_growth[site], "peak_advance": self.peak_advance[site]}
                          for site in sorted(self.calls)]}


def census_forward(run: Callable[[], T], rows: int, context: int) -> Callable[[], T]:
    """``run`` itself when the flag is off; else ``run`` under a census that logs one line when it ends, failed or not."""
    if not census_enabled():
        return run

    def measured() -> T:
        core, hpu = _runtime()
        census = MarkStepCensus(core, hpu)
        try:
            with census:
                return run()
        finally:
            _log_line({**census.report(rows, context), "path": "eager"})

    return measured


def _log_line(body: dict[str, object]) -> None:
    LOG.warning("%s %s", LINE_TAG, json.dumps(body, sort_keys=True))


def _restart_peak(hpu: Any) -> tuple[int | None, bool]:
    """(the old MaxInUse, the restart failed). Failures cost this forward's reading only."""
    between = None
    try:
        between = int(hpu.memory_stats()["MaxInUse"])
    except Exception:  # noqa: BLE001  log only: a diagnostic must never fail a forward
        pass
    try:
        hpu.reset_peak_memory_stats()
    except Exception:  # noqa: BLE001  same; the line carries peak_reset_failed
        return between, True
    return between, False


def _shape(kind: str, rows: int, context: int, prompts: int = 1) -> tuple:
    """GRAPH_LOG's key. A shared key (prompts > 1) has the rows and width of a single key beside it, so its prompts tell them apart."""
    return (kind, rows, context) if prompts == 1 else (kind, rows, context, prompts)


def _log_graph_line(report: dict[str, Any], kind: str, between: int | None, reset_failed: bool) -> None:
    """Every capture; replays throttled per shape, because decode replays run at every step. A new high of real prompts always logs, so the
    throttle cannot hide the first full step of a shared key."""
    top, start = report["max_in_use"], report["start_in_use"]
    peak = top - start if top is not None and start is not None else 0
    entry = GRAPH_LOG.setdefault(_shape(kind, report["rows"], report["context"], report.get("prompts", 1)), [0, 0, 0])
    entry[0] += 1
    new_peak = peak > entry[1]
    entry[1] = max(entry[1], peak)
    new_real = report.get("real_prompts", 0) > entry[2]
    entry[2] = max(entry[2], report.get("real_prompts", 0))
    if kind == "capture" or new_peak or new_real or entry[0] <= GRAPH_LOG_FIRST or entry[0] % GRAPH_LOG_EVERY == 0:
        _log_line({**report, "path": "graph", "kind": kind, "between_peak": between, "peak_reset_failed": reset_failed,
                   "n": entry[0]})


def _next_replay_timed(rows: int, context: int, prompts: int = 1) -> bool:
    """Every second replay of a shape is timed; the others go unfenced and keep the run-ahead reading."""
    return GRAPH_LOG.get(_shape("replay", rows, context, prompts), [0])[0] % 2 == 1


def _drain(core: Any, hpu: Any) -> None:
    core.mark_step()
    hpu.synchronize()


def _measured_graph_forward(run: Callable[[], T], rows: int, context: int, captures: Callable[[], int],
                            key_replay: bool | None, *, prompts: int = 1, real_prompts: int = 1) -> T:
    """``run`` under a census: one GLM53_PROMPT_CENSUS line when it ends (see census_graph_forward)."""
    core, hpu = _runtime()
    timed = key_replay is True and _next_replay_timed(rows, context, prompts)
    if timed:
        _drain(core, hpu)  # the previous chunk's tail, outside this replay's clock and peak
    before = captures()
    between, reset_failed = _restart_peak(hpu)
    census = MarkStepCensus(core, hpu)
    trace = stall_trace(rows, context, key_replay is False)
    census.trace = trace
    started, extra, finished = time.perf_counter(), {}, False
    try:
        with census, trace or nullcontext():
            result = run()
        if timed:
            _drain(core, hpu)
            extra = {"replay_ms": round((time.perf_counter() - started) * 1000.0, 3)}
        finished = True
        return result
    finally:
        kind = "capture" if captures() != before else "replay"
        if finished or key_replay is None or kind == "capture":
            _log_graph_line({**census.report(rows, context), **extra, **({"prompts": prompts, "real_prompts": real_prompts} if prompts > 1 else {})},
                            kind, between, reset_failed)


def census_graph_forward(run: Callable[[], T], rows: int, context: int, captures: Callable[[], int],
                         *, key_replay: bool | None = None, decode_width: int | None = None, prompts: int = 1, real_prompts: int = 1) -> Callable[[], T]:
    """``run`` under a census; ``captures`` is the graph count, whose growth marks this forward as a capture.

    ``key_replay`` True: a prompt key's replay, with `replay_ms` on every second one, device idle at its start and end.
    False: a key's capture, never fenced. Either way a key forward that failed before it had a graph writes no line.
    ``decode_width``: a decode step's block-table width (None for anything else); with ``key_replay`` it names the key the replay
    tracer (HPU_GLM53_REPLAY_TRACE_DIR) may bracket. The census itself ignores it.
    ``prompts``: the prompts a shared key (VLLM_GAUDI_MIXED_PROMPT_BATCH_KEYS) carries; above 1 it is a field of the line and part of the
    shape the throttle and the timing count, so the line is not mistaken for the single key of the same rows and width.
    ``real_prompts``: the prompts this forward really carries; a shared key's others are pad rows, so `prompts` alone proves no sharing.
    """

    measured = functools.partial(_measured_graph_forward, run, rows, context, captures, key_replay, prompts=prompts, real_prompts=real_prompts)
    if REPLAY_TRACER is None:
        return measured
    return REPLAY_TRACER.wrap(measured, rows=rows, context=context, captures=captures, key_replay=key_replay, decode_width=decode_width)
