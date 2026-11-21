"""PROMPT-GRAPH-PLAN change B: eager prompt forwards use a private top-k buffer, so long chunks stop clearing the
whole HPU graph cache. Also the S5 device storage census (R7 / C513).

Why. GLM-5.3 keeps ONE shared mutable ``topk_indices_buffer`` per model (image vllm/models/glm5next/nvidia/model.py
791: (max_num_batched_tokens, round128(index_topk + index_kpool - 1)) int32 = (8192, 2176) = 71,303,168 B/rank).
Each sparse MLA wrapper, its indexer and its ``mla_attn`` hold that same tensor (oot_mla 682/708, indexer 163), every
captured decode graph reads and writes views of it, and eager prompt forwards write it in place (indexer 697/815/
869/965). So the runner's ``adaptive_forward`` (d23b680f 1766-1879) latches after a >=512-token prompt at sparse
width >= HPU_SHARED_STATE_RECAPTURE_MIN_SPARSE_WIDTH and clears the WHOLE graph cache at the next graph forward
(C511: ~80 recaptures, ~78 s per run).

Flag on (HPU_GLM53_PRIVATE_PROMPT_TOPK=1):
  * at wrap time (load_model 6914, before profile_run, so KV sizing sees the bytes) one ``torch.empty_like`` copy of
    the shared buffer is allocated per model root (GLM53_GS_PRIVATE);
  * every eager forward (the large-prefill route and every ``bypass_hpu_graphs`` call) runs with every holder
    attribute swapped to the private tensor, restored in ``finally``; graph forwards always see the shared tensor;
  * a large-prefill bypass at width >= the recapture width no longer sets the dirty latch: it runs the runner's own
    large-prefill route (segmented lazy collectives + the original forward) directly. The restored-long-request
    latch (``refresh_hpu_graphs_after_external_bypass``, KV offload restores) is left to the runner, but in serving
    the whole-cache clear it runs (d23b680f 1865) is a counted no-op (clears_scoped): that request's eager forwards
    ran on the private buffer too, and so did every decode step the runner deferred while the latch was set
    (d23b680f 1838-1849, external_graph_transition_active: eager on the original forward, bound by B,
    deferred_bound on GLM53_GS_STATS), so no graph saw their writes. Without the private bind (DIAG) it stays whole.
  * HPU_GLM53_GS_HBM_GUARD=lru (C536 ship config, the floor below OFF): no floor check points. Each guard (target,
    drafter) seals its own cap N = (TP-min (Limit - InUse) - 512 MiB) // TP-max capture resident bytes (the InUse
    a warmup capture added; 264 MiB when warmup measured none), at least 1 (GLM53_GS_LRU event=cap; source=error
    when the free read failed). Runtime graphs are decode AND prompt graphs (the runner's _use_graphs captures small
    prompts while serving, 7137-7179), one LRU entry per step shape. A serving graph step of a shape without a live
    graph captures on every rank; first ONE collective over [Limit - InUse, -(P + G), -G] (P the measured eager
    chunk spike, G the TP-max graph resident) drops the least recently used shapes until the TP-min free covers
    P + G (all of them when a rank's read failed), and to keep at most N (event=room, then event=evict per shape,
    one fence pair; evictions / released / recaptures on GLM53_GS_STATS). Any cache growth after a serving graph
    forward joins that shape's entry (or a stray set dropped at the next eviction), so GLM53_GS_TICK's
    untracked = graphs - boot - tracked stays 0. The boot set is never dropped. Default (floor): below.
  * HBM floor: that clear was also the only release of runtime-captured decode graphs (3 decode buckets are warmed at
    boot, ~93 restored at runtime, ~130-210 MiB each). B needs HPUModelRunner.warmup_model (refused without it).
    The boot set (every key in habana's graph cache dict at the
    first serving forward, i.e. after warmup_model and pgk's seal: the warmup decode buckets and pgk's replay-only
    prompt keys) is NEVER released: habana's clear_cache (graphs.py 908, cache.clear()) is never called by B, and a
    release deletes only the keys captured after that seal, fenced on both sides (GLM53_GS_RELEASE). Recapturing a
    boot key instead would be unsafe (C15; pgk never recaptures). Check points, once warmup_model returned (a runner
    without warmup_model is refused): every suppression decision, every large eager forward
    (>= the bypass size, any sparse width; in serving these are most chunks) and every decode step whose shape
    (input_ids' shape and every field of the runner's trim_attn_metadata namedtuple by name, tensors by dtype,
    shape and strides, the same on every TP rank; habana's
    input_hash seeds per process, graphs.py 554-560, so cache growth is rank-local) has no live graph since the
    last release. Each is ONE gloo all_reduce MIN over [free, -floor] on every rank: the smallest free HBM and the
    largest floor, the same answer everywhere (runner 1843). A rank whose free read raises still joins with
    FREE_ERROR, so every rank stops together instead of hanging. Plain replays and small eager calls are never
    checked, so the cost is at most one 2-int64 CPU collective per checked forward (tp_min_ms on GLM53_GS_HBM,
    checks / check_ms_max on GLM53_GS_STATS), plus one after each release. Below the floor the release runs before
    the eager forward, or, after a capture whose outputs are still in use, before the next forward (that forward is
    then not checked again). A release that leaves the TP-min free below the floor stops the guard
    (GLM53_GS_FLOOR event=exhausted, STATS stopped=exhausted). RC4 then kept every graph resident for good; C613
    (HPU_GLM53_GS_EXHAUSTED_REARM=1, default) pauses only the capture checks (never a per-decode-step capture /
    release / capture thrash) while every large eager chunk still checks, releases the runtime graphs captured
    since (action=starved when no rank has one) and the first check above the floor re-arms (event=rearm).
    GLM53_GS_FLOOR also logs retained_peak_mib (the InUse a large eager chunk leaves behind; diagnostic, for the
    scheduler's context row cap calibration). RC5.1: a graph-bypassing prompt chunk past
    HPU_GLM53_GS_LONG_CONTEXT_TOKENS (default 65536) of context counts as large (checked, released for, measured)
    whatever its rows; paused, a chunk below the floor keeps the runtime graphs (action=spare) when its predicted
    peak (the largest measured peak of a chunk at least as wide in rows and context, else the prompt peak) plus
    2 x the capture peak fits the TP-min free, or the peak alone fits and the last release was < 30 s ago
    (HPU_GLM53_GS_PAUSED_SPARE=0: RC5). The floor in effect is
        max(HPU_GLM53_GS_HBM_FLOOR_MIB (default 512), prompt peak + 2 x decode-graph capture peak),
    both peaks MEASURED in this process from habana's allocator counters (MaxInUse above InUse, hpu/memory.py), each
    the largest of its last 64 measurements so a one-off peak (the profile run) ages out: the prompt peak around
    every large eager forward (a GLM53_GS_PEAK line around each suppressed chunk), the capture peak over every
    decode capture check point (prompt-graph captures are not counted). Every change of a term prints
    GLM53_GS_FLOOR (event=boot at wrap time, then event=prompt|capture, event=seal with the boot keys as rows); when
    warmup_model returns each guard prints event=inert with free_mib and inert=1 when the floor is above it (B then
    releases at its first check, and stops when the boot set alone leaves free below the floor). Reads are
    host-side allocator counters (no device sync): a lazy segment is counted once launched.
  * refused unless the wrapper was built with HPU_GRAPH_DRY_RUN=0 / HPU_GRAPH_FREE_INPLACE=0 (graphs keep their
    state) and the runner's adaptive_forward source is the pinned d23b680f function (ADAPTIVE_SOURCE_SHA).
Bitwise argument (plan 2.3): the buffer is dead-in (the first sparse layer of each forward runs fill_(-1) + copy_
before any read; checked at wrap time) and the private copy has the same shape/stride/dtype/device, so prompt
recipes and values are the same; decode replays G instead of recapturing the same-key G'.

Hook. The runner's bytes are not changed. ``install()`` (called by usercustomize.py) adds a one-shot import finder
for ``vllm_gaudi.v1.worker.hpu_model_runner`` (the glm53_step_stats pattern) that wraps the module's
``_maybe_wrap_in_hpu_graph``; load_model looks that global up at call time (target 6914, drafter 6961). The adaptive
route's configuration (original_forward, bypass/recapture thresholds) is read from its closure, never re-parsed.
Flags off: usercustomize reads the flag variables and imports nothing. Flags on: any install failure stops the
interpreter (usercustomize exits 3) or fails the runner import (GraphStateError); it never boots a vacuous arm.

Diagnostics (never shipped):
  HPU_GLM53_GS_SKIP_CLEAR_ONLY=1  latch suppression WITHOUT the private bind (TB8: is the private buffer needed?).
  HPU_GLM53_GRAPH_STATE_CENSUS=1  storage census: after every forward that captured an HPU graph (runner decode
      graphs AND the dual arm's raw mixed graphs, both built by habana's wrapped_hpugraph_forward), every eager
      forward and every cache clear, record (data_ptr, storage ptr, storage nbytes) of every tensor a module holds
      as a plain attribute (incl. lists and dicts: KV caches, recurrent state, the top-k holders) for every wrapped
      model, plus the non-module holders: the static_forward_context passed to the wrap (layers outside the root,
      plain entries) and the runner whose load_model called it (its tensor attributes). GLM53_GS_CENSUS per
      capture/clear (with the shared and private top-k storage pointers), GLM53_GS_CENSUS_DIFF on any identity
      change, scope=runner when only runner attributes moved (per-step host buffers), else scope=state. Census-only
      mode never changes routing, so it runs on the dual runner (9eed89c0) too.
      Snapshots run only after a forward returns, never inside a capture, and only after a device fence
      (mark_step(sync=True) + hpu.synchronize, the runner's own clear handoff 1863-1864): ``fenced=1`` on the line.

Log lines (stdout; vLLM prefixes the worker): GLM53_GS_INSTALL, GLM53_GS_PRIVATE, GLM53_GS_BIND (rank 0, every 50th),
GLM53_GS_FLOOR (boot, every term change, event=seal|inert at warmup end, event=exhausted|error when the guard
stops, every rank), GLM53_GS_HBM (every check point, every rank, point=suppress|eager|capture, with max_in_use_mib and
tp_min_ms), GLM53_GS_HBM_ERROR (a failed free read), GLM53_GS_LRU (lru: event=cap|room|evict), GLM53_GS_TICK (every 60 s per
guard: graphs, boot, tracked, untracked, runtime_live, cap, free_mib, stopped, also after the guard stopped), GLM53_GS_RELEASE (every release: dropped, tp_min_free_mib after,
result=ok|exhausted|error), GLM53_GS_PEAK (every suppressed chunk, every rank), GLM53_GS_STATS (every 60 s, on
suppressions, on releases and at exit of a process that loaded the runner; released / captures / recaptures /
clears_scoped / stopped / guard / evictions / deferred_bound), GLM53_GS_CENSUS (roots= names every
census root), GLM53_GS_CENSUS_DIFF, GLM53_GS_CENSUS_HOLDERS (a wrap whose ctx/runner holder was not found).
"""
from __future__ import annotations

import atexit
import contextlib
import hashlib
import importlib.abc
import importlib.machinery
import inspect
import os
import sys
import time
import weakref
from collections import Counter, OrderedDict, deque
from collections.abc import Callable, Iterator, Mapping, MutableMapping, Sequence
from dataclasses import dataclass, field, fields, is_dataclass
from types import FrameType, ModuleType
from typing import TYPE_CHECKING, Any, NamedTuple, TypeVar

from glm53_graphs_only import ENV as GRAPHS_ONLY_ENV, arm_exit, fail_fast_on_allocation_error
from glm53_prompt_census import census_enabled, census_forward, census_graph_forward

if TYPE_CHECKING:
    import torch

ENABLE_ENV = "HPU_GLM53_PRIVATE_PROMPT_TOPK"
MIXED_DECODE_REPLAY_ENV = "HPU_GLM53_MIXED_DECODE_REPLAY"
DECODE_WIDTHS_ENV = "VLLM_GAUDI_MIXED_DECODE_WIDTHS"
SKIP_CLEAR_ONLY_ENV = "HPU_GLM53_GS_SKIP_CLEAR_ONLY"
CENSUS_ENV = "HPU_GLM53_GRAPH_STATE_CENSUS"
ROWS_ENV = "HPU_GLM53_PRIVATE_PROMPT_TOPK_ROWS"
HBM_FLOOR_ENV = "HPU_GLM53_GS_HBM_FLOOR_MIB"
HBM_FLOOR_DEFAULT_MIB = 512  # the minimum floor; measured peaks raise the floor in effect above it
CAPTURE_MARGIN = 2  # decode-graph captures the floor leaves room for between two check points
PEAK_WINDOW = 64  # each floor term is the largest of its last 64 measurements (a one-off peak ages out)
MIB = 1 << 20
FREE_ERROR = -1  # a rank whose free read failed sends this through the check collective: every rank stops
NOT_STOPPED, STOP_EXHAUSTED, STOP_ERROR = "none", "exhausted", "error"
GUARD_ENV = "HPU_GLM53_GS_HBM_GUARD"  # floor (default) | lru (C536 ship config: the floor off, runtime graphs capped)
GUARD_FLOOR, GUARD_LRU = "floor", "lru"
# C613: exhausted pauses capture checks instead of stopping the guard for the life of the process (a boot-time
# profile peak latched it off, so 22 runtime graphs piled up and a long-context chunk ran out of HBM). 0 = RC4 latch.
REARM_ENV = "HPU_GLM53_GS_EXHAUSTED_REARM"
# RC5.1: the scheduler's 256/128-row rungs (context > ~270k) are below the 512-row large-prefill bypass, so they ran
# eager and unchecked. A prompt chunk that bypasses graphs at a context past this many tokens is guarded (checked,
# released for, measured) like a large one. 0 = RC5 (only >= the bypass rows).
LONG_CONTEXT_ENV = "HPU_GLM53_GS_LONG_CONTEXT_TOKENS"
LONG_CONTEXT_DEFAULT = 65536
# RC5.1: paused (exhausted), a chunk below the floor releases only when ITS predicted peak + margin does not fit the
# TP-min free, and a release that only the margin asks for waits PAUSED_RELEASE_MIN_S after the last one (no
# recapture churn under c16 decode). 0 = RC5 (every paused chunk below the floor releases).
PAUSED_SPARE_ENV = "HPU_GLM53_GS_PAUSED_SPARE"
PAUSED_RELEASE_MIN_S = 30.0
# Lane PEAK: with the R616/R606 drains off, the host runs ahead of the device, frees wait for recipe retirement and
# a long chunk's MaxInUse lands within ~64 MiB of the allocator Limit whatever the chunk needs (final-b C16: 96,830 of
# 96,894 MiB on every rank, at every start InUse). Such a reading says nothing about the chunk's need, and floors
# built from it always sit above free. On (default off): a guarded chunk whose MaxInUse is within MARGIN of Limit and
# whose rows / context are within the measured shape is charged the configured real-need peak (measured with every
# drain on) instead. final-g2: so is any covered reading above the need up to HPU_GLM53_GS_RUNAHEAD_CEILING_MIB
# (drains-off readings are 1.5-4x the drains-on need without touching the margin); a non-saturated covered reading
# above the ceiling is genuine, charged as measured, and once serving (after the seal) raises later run-ahead
# charges for the 64-sample window. At the seal, warmup chunk-bucket peaks and the warmup prompt window expire and
# the prompt window restarts at the need. Everything else - minimum reserve, 2 x capture peak, the TP-wide vote, exhaustion - is final-b.
# final-g3: genuine also needs < 60% of the free at the chunk's start; at the seal only COVERED warmup buckets expire
# (uncovered ones seed admission_floor for uncovered chunks); a slow run-ahead chunk with graphs resident trips a
# 64-chunk free-at-start charge. GLM53_GS_SATURATION counters: charged_n = run-ahead readings charged the need;
# sat_uncovered_n = saturated readings of uncovered chunks (charged as measured); tripwire_left = chunks still tripped.
SATURATED_PEAK_ENV = "HPU_GLM53_GS_SATURATED_PEAK"
REAL_NEED_PEAK_ENV = "HPU_GLM53_GS_REAL_NEED_PEAK_MIB"
REAL_NEED_MAX_ROWS_ENV = "HPU_GLM53_GS_REAL_NEED_MAX_ROWS"
REAL_NEED_MAX_CONTEXT_ENV = "HPU_GLM53_GS_REAL_NEED_MAX_CONTEXT"
SATURATION_MARGIN_ENV = "HPU_GLM53_GS_SATURATION_MARGIN_MIB"
SATURATION_MARGIN_DEFAULT_MIB = 256
SATURATION_MARGIN_MAX_MIB = 1024  # a wider margin would call honest readings saturated
SATURATION_LRU_WARNED: set[str] = set()  # the lru guard has no floor to charge: warned once per process
REAL_NEED_MAX_ROWS_DEFAULT = 2048
RUNAHEAD_CEILING_ENV = "HPU_GLM53_GS_RUNAHEAD_CEILING_MIB"
RUNAHEAD, HONEST, GENUINE, UNCOVERED = "runahead", "honest", "genuine", "uncovered"
GENUINE_FREE_FRACTION = 0.6  # final-g3: a reading that used >= 60% of the free it started with is run-ahead fill
TRIPWIRE_SLOWDOWN = 1.5  # a run-ahead chunk this much slower than its bucket's p50, graphs resident, trips
TRIPWIRE_MIN_SAMPLES = 8  # bucket timings needed before a p50 is trusted
TRIPWIRE_CHUNKS = 64  # covered chunks charged free-at-start after a trip
TRIPWIRE_HISTORY = 32  # timings kept per bucket
# Tripwire v2 (default off): a prompt chunk behind a graph capture of its own step is slow because of the capture (the
# decode capture runs first in a mixed step), so it neither samples its bucket nor trips. The 1.5 factor is unchanged.
TRIPWIRE_V2_ENV = "HPU_GLM53_GS_TRIPWIRE_V2"
TRIPWIRE_SKIP_LOG_EVERY = 64  # GLM53_GS_SATURATION event=tripwire_capture_skip prints the 1st, 65th, ... skip
# Step 23, HPU_GLM53_GRAPHS_ONLY=1: every serving forward replays a captured graph, so the guard has nothing to release. It stays as a witness: the
# same check collective reports the TP-min free, no floor decides anything, and a slow run-ahead chunk is only logged, and only when under
# this much of the Limit was left above its peak (the replay of r0 / r1: 99 of 173 trips had more).
WITNESS_HEADROOM_MIB = 1024
NEVER_RELEASED_MS = 1 << 62  # "since the last release" before the first one (an int64 the MIN collective carries)
LRU_RESERVE_MIB = 512  # HBM the cap leaves free under the TP-min free at the seal
LRU_DEFAULT_GRAPH_MIB = 264  # a decode capture's size when warmup measured none (r6 capture_peak, C534)
# sha256 of inspect.getsource(adaptive_forward) in runner d23b680f (lines 1766-1879); tail-fuse keeps these bytes.
ADAPTIVE_SOURCE_SHA = "f22b118132663b7d224b6b863e18aa255357d5fc64647900674d8fb8c058fa8a"
STATE_KEEPING_POLICY = (False, False)  # (dry_run, free_inplace) as runner _record_wrapped_graph_policy stores it
FLAG_ENVS = (ENABLE_ENV, SKIP_CLEAR_ONLY_ENV, CENSUS_ENV)
BYPASS_MIN_ENV = "HPU_GLM53_Q4_LARGE_PREFILL_BYPASS_GRAPH_MIN_TOKENS"
RUNNER_MODULE = "vllm_gaudi.v1.worker.hpu_model_runner"
GRAPHS_MODULE = "habana_frameworks.torch.hpu.graphs"
COMMUNICATOR_MODULE = "vllm_gaudi.distributed.device_communicators.hpu_communicator"
HOLDER_ATTRS = ("topk_indices_buffer", "hpu_sparse_topk_indices_buffer")
LIVE_IN_MARKERS = ("set_skip_topk", "compact_topk_indices")  # image glm5next/nvidia/mtp.py 142-156
ROUTE_VARS = ("original_forward", "bypass_min_tokens", "recapture_min_sparse_width", "bypass_sparse_width_change")
LATCH_VAR = "recapture_shared_state_after_bypass"  # adaptive_forward's dirty latch (a nonlocal: read the live cell)
REFRESH_KWARG = "refresh_hpu_graphs_after_external_bypass"
EXTERNAL_KWARG = "external_graph_transition_active"
BIND_LOG_EVERY = 50
SUPPRESS_LOG_FIRST = 8
STATS_EVERY_S = 60.0
CENSUS_LOGGED_EVENTS = frozenset({"wrap", "capture", "clear"})
CENSUS_DIFF_NAMES = 6
CENSUS_EVERY = 50  # an unlogged event kind still prints its 1st, 51st, ... census line
RUNNER_SCOPE = "runner"
RUNNER_FRAME_DEPTH = 8  # load_model -> other levers' wrap wrappers (dtc, pgk, profilers) -> this wrap
RUNNER_CLASS_MARK = "ModelRunner"  # HPUModelRunner (and subclasses): the object whose load_model wraps
MARK = "_glm53_graph_state"


T = TypeVar("T")


class GraphStateError(RuntimeError):
    """A contract this module relies on does not hold; the message says which and what to change."""


def log_line(line: str) -> None:
    print(line, flush=True)


def _flag(environ: Mapping[str, str], name: str) -> bool:
    value = environ.get(name, "0").strip()
    if value not in ("", "0", "1"):
        raise GraphStateError(f"{name}={value!r}: use 1 (on) or 0/unset (off)")
    return value == "1"


@dataclass(frozen=True)
class Mode:
    private: bool = False
    skip_clear_only: bool = False
    census: bool = False

    @property
    def routes(self) -> bool:
        """Large-prefill latch suppression is on (change B, or the TB8 diagnostic)."""
        return self.private or self.skip_clear_only

    @property
    def engaged(self) -> bool:
        return self.routes or self.census


def read_mode(environ: Mapping[str, str] | None = None) -> Mode:
    environ = os.environ if environ is None else environ
    mode = Mode(_flag(environ, ENABLE_ENV), _flag(environ, SKIP_CLEAR_ONLY_ENV), _flag(environ, CENSUS_ENV))
    if mode.private and mode.skip_clear_only:
        raise GraphStateError(f"{ENABLE_ENV}=1 and {SKIP_CLEAR_ONLY_ENV}=1 contradict (private bind vs no bind); "
                              "set exactly one")
    if graphs_only(environ) and not mode.private:
        raise GraphStateError(f"{GRAPHS_ONLY_ENV}=1 needs {ENABLE_ENV}=1: a prompt key graph writes the private top-k buffer, so the "
                              f"decode graphs' shared one is never touched. Set {ENABLE_ENV}=1 or unset {GRAPHS_ONLY_ENV}")
    return mode


def graphs_only(environ: Mapping[str, str]) -> bool:
    """HPU_GLM53_GRAPHS_ONLY=1: the guard is a witness (report only: no floor, no clear, no release)."""
    return _flag(environ, GRAPHS_ONLY_ENV)


# --- holders -----------------------------------------------------------------------------------------------------


class Holder(NamedTuple):
    name: str
    module: Any
    attr: str

    def get(self) -> torch.Tensor:
        return vars(self.module)[self.attr]

    def put(self, tensor: torch.Tensor) -> None:
        vars(self.module)[self.attr] = tensor  # a plain attribute: no nn.Module __setattr__, no hooks


def holders(root: object) -> tuple[Holder, ...]:
    """Every (module, attr) holding a top-k buffer as a plain attribute, in module order."""
    import torch

    found = []
    for name, module in root.named_modules():
        for attr in HOLDER_ATTRS:
            if isinstance(vars(module).get(attr), torch.Tensor):
                found.append(Holder(f"{name}.{attr}" if name else attr, module, attr))
    return tuple(found)


def shared_base(found: Sequence[Holder]) -> torch.Tensor:
    """The one tensor every holder holds; raises naming the holders otherwise."""
    if not found:
        raise GraphStateError(f"no {'/'.join(HOLDER_ATTRS)} holder in the wrapped model: {ENABLE_ENV} only applies "
                              "to GLM-5.3 sparse-indexer models; unset it for this arm")
    groups: dict[int, list[str]] = {}
    for holder in found:
        groups.setdefault(id(holder.get()), []).append(holder.name)
    if len(groups) != 1:
        detail = "; ".join(f"{len(names)} holders e.g. {names[:2]}" for names in groups.values())
        raise GraphStateError(f"{len(groups)} distinct top-k tensors in one model root ({detail}); the private bind "
                              "swaps exactly one shared base: extend holders() for this layout before enabling")
    return found[0].get()


def check_dead_in(root: object) -> str:
    """Name of the first sparse MLA layer; raises unless every forward writes the buffer before it reads it."""
    modules = list(root.named_modules())
    for name, module in modules:
        live = [marker for marker in LIVE_IN_MARKERS if callable(getattr(module, marker, None))]
        if live:
            raise GraphStateError(f"{name or type(module).__name__} has {live}: an MTP drafter keeps top-k rows live "
                                  f"across iterations, so a private prompt buffer breaks it; unset {ENABLE_ENV} "
                                  "on MTP arms")
    for name, module in modules:
        if not getattr(module, "hpu_glm5next_sparse", False):
            continue
        if getattr(module, "indexer", None) is None or getattr(module, "skip_topk", True):
            raise GraphStateError(f"first sparse MLA layer {name} has skip_topk or no indexer: it reads the top-k "
                                  f"buffer before any layer of the forward wrote it (dead-in contract, plan 2.1); "
                                  f"unset {ENABLE_ENV}")
        return name
    raise GraphStateError(f"no sparse MLA layer (hpu_glm5next_sparse) under {type(root).__name__}; unset {ENABLE_ENV}")


def allocate_private(shared: torch.Tensor, environ: Mapping[str, str]) -> torch.Tensor:
    import torch

    rows = environ.get(ROWS_ENV, "").strip()
    if rows and rows != str(shared.shape[0]):
        raise GraphStateError(f"{ROWS_ENV}={rows}: a private buffer other than the shared {shared.shape[0]} rows "
                              "changes the [:n_buffer] views, which needs the S5 recipe-identity check first; unset it")
    private = torch.empty_like(shared)
    layout = (private.shape, private.stride(), private.dtype, private.device)
    if layout != (shared.shape, shared.stride(), shared.dtype, shared.device):
        raise GraphStateError(f"private top-k layout {layout} differs from the shared buffer; the bitwise argument "
                              "needs an identical layout - report the shared tensor's strides")
    return private


# --- process state -------------------------------------------------------------------------------------------------


def _rank() -> int:
    try:
        from vllm.distributed import get_tensor_model_parallel_rank

        return int(get_tensor_model_parallel_rank())
    except Exception:  # before distributed init or outside vLLM: the env rank is the best label
        return int(os.environ.get("RANK", "0") or 0)


@dataclass
class Stats:
    binds: int = 0
    max_rows: int = 0
    latch_suppressed: int = 0
    latch_external: int = 0
    clears: int = 0
    census_diffs: int = 0
    latch_hbm: int = 0  # suppression decisions taken below the floor (the graph cache is released first)
    clears_hbm: int = 0  # releases (runtime graphs dropped, fenced), each before the forward that needed the room
    checks: int = 0  # check points (one gloo collective each)
    checks_low: int = 0  # eager / capture check points below the floor
    check_ms_max: float = 0.0  # slowest check collective
    released: int = 0  # runtime graph keys dropped (by B's releases and by scoped runner clears)
    captures: int = 0  # serving decode steps of a shape without a live graph
    recaptures: int = 0  # ... of a shape whose graph a release dropped
    clears_scoped: int = 0  # runner clear_cache calls in serving that dropped runtime keys only
    stopped: str = NOT_STOPPED  # exhausted | error: the guard stopped checking (every graph stays resident)
    guard: str = "-"  # floor | lru (HPU_GLM53_GS_HBM_GUARD)
    evictions: int = 0  # lru: least-recently-used runtime shapes dropped to stay under the cap
    deferred_bound: int = 0  # decode steps the runner deferred eagerly during a restore, run on the private buffer
    eager_forwards: int = 0  # every forward this route ran eager (deferred decodes, bypassed or suppressed prompts); the boot witness reads it, the line does not (flag off logs as base)
    last_emit: float = field(default_factory=time.monotonic)

    def line(self) -> str:
        return (f"GLM53_GS_STATS bind={self.binds} latch_suppressed={self.latch_suppressed} "
                f"latch_external={self.latch_external} clears={self.clears} census_diffs={self.census_diffs} "
                f"latch_hbm={self.latch_hbm} clears_hbm={self.clears_hbm} checks={self.checks} "
                f"checks_low={self.checks_low} check_ms_max={self.check_ms_max:.3f} released={self.released} "
                f"captures={self.captures} recaptures={self.recaptures} clears_scoped={self.clears_scoped} "
                f"stopped={self.stopped} guard={self.guard} evictions={self.evictions} "
                f"deferred_bound={self.deferred_bound} "
                f"max_rows={self.max_rows} "
                f"pid={os.getpid()}")

    def note_check(self, point: str, keep: bool, elapsed_ms: float) -> None:
        self.checks += 1
        self.check_ms_max = max(self.check_ms_max, elapsed_ms)
        if not keep:
            if point == "suppress":
                self.latch_hbm += 1
            else:
                self.checks_low += 1

    def tick(self) -> None:
        """Every 60 s: the counters, then each guard's free HBM and graph count (also once it stopped checking)."""
        now = time.monotonic()
        if now - self.last_emit >= STATS_EVERY_S:
            self.last_emit = now
            log_line(self.line())
            for guard in LIFECYCLE.live_guards():
                guard.log_tick()


class GraphState:
    """One wrapped model root: its holders, the shared base and (flag on) the private copy."""

    def __init__(self, root: Any, mode: Mode, environ: Mapping[str, str]) -> None:
        self.found = holders(root)
        self.private: torch.Tensor | None = None
        self.first_sparse = ""
        self._depth = 0
        if not mode.routes:  # census only: observe whatever layout this runner has, never refuse it
            self.shared = self.found[0].get() if len({id(h.get()) for h in self.found}) == 1 else None
            return
        self.shared = shared_base(self.found)
        self.first_sparse = check_dead_in(root)
        if mode.private:
            self.private = allocate_private(self.shared, environ)

    @contextlib.contextmanager
    def bind(self) -> Iterator[None]:
        """Every holder -> private for the duration (no-op without a private copy, or when already bound)."""
        if self.private is None or self._depth:
            self._depth += 1
            try:
                yield
            finally:
                self._depth -= 1
            return
        STATS.binds += 1
        self._depth = 1
        for holder in self.found:
            holder.put(self.private)
        try:
            yield
        finally:
            for holder in self.found:
                holder.put(self.shared)
            self._depth = 0


class Entry(NamedTuple):
    data_ptr: int
    storage_ptr: int
    nbytes: int


UNREADABLE = Entry(-1, -1, -1)


def _entry(tensor: torch.Tensor) -> Entry:
    try:
        storage = tensor.untyped_storage()
        return Entry(int(tensor.data_ptr()), int(storage.data_ptr()), int(storage.nbytes()))
    except (RuntimeError, NotImplementedError, TypeError):
        return UNREADABLE


def _flatten(label: str, value: object, depth: int = 0) -> Iterator[tuple[str, torch.Tensor]]:
    import torch

    if isinstance(value, torch.Tensor):
        yield label, value
    elif depth >= 2:
        return
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            yield from _flatten(f"{label}[{index}]", item, depth + 1)
    elif isinstance(value, Mapping):
        for key, item in value.items():
            yield from _flatten(f"{label}[{key!r}]", item, depth + 1)


def _module_internals() -> frozenset[str]:
    """nn.Module's own bookkeeping attributes (_parameters, _buffers, _modules, hook dicts): never state holders."""
    from torch import nn

    return frozenset(vars(nn.Module()))


def module_tensors(root: object, prefix: str = "", seen: set[int] | None = None) -> Iterator[tuple[str, torch.Tensor]]:
    """Tensors held as plain module attributes, incl. inside lists and dicts (parameters and registered buffers are
    not plain attributes). ``seen`` (module ids) skips modules another holder already walked."""
    internals = _module_internals()
    seen = set() if seen is None else seen
    for name, module in root.named_modules():
        if id(module) in seen:
            continue
        seen.add(id(module))
        label = f"{prefix}{name}." if name else prefix
        for attr, value in vars(module).items():
            if attr not in internals:
                yield from _flatten(label + attr, value)


def holder_tensors(holder: object, seen: set[int]) -> Iterator[tuple[str, torch.Tensor]]:
    """A census root's tensors: a module tree, a name -> layer mapping (static_forward_context) or any object's
    tensor attributes (the runner); modules met inside a mapping are walked once across all roots."""
    from torch import nn

    if isinstance(holder, nn.Module):
        yield from module_tensors(holder, seen=seen)
    elif isinstance(holder, Mapping):
        for key, value in holder.items():
            if isinstance(value, nn.Module):
                yield from module_tensors(value, f"{key}.", seen)
            else:
                yield from _flatten(str(key), value)
    else:
        for attr, value in vars(holder).items():
            yield from _flatten(attr, value)


def device_fence(namespace: Mapping[str, object]) -> Callable[[], None] | None:
    """The runner's clear handoff (d23b680f 1863-1864) from a module namespace holding ``htorch``; None without it."""
    htorch = namespace.get("htorch")
    if htorch is None:
        return None

    def fence() -> None:
        htorch.core.mark_step(sync=True)
        htorch.hpu.synchronize()

    return fence


class Census:
    """Identity snapshots of every registered root, each after a device fence; diffs are logged and counted."""

    def __init__(self) -> None:
        self.roots: list[tuple[str, Callable[[], object], GraphState | None]] = []
        self.holder_ids: set[int] = set()
        self.last: dict[str, Entry] = {}
        self.seq = 0
        self.events: Counter[str] = Counter()
        self.pending_captures = 0
        self.fence: Callable[[], None] | None = None

    def register(self, label: str, root: object, state: GraphState | None,
                 fence: Callable[[], None] | None = None) -> None:
        self.roots.append((label, _reference(root), state))
        self.holder_ids.add(id(root))
        self.fence = self.fence or fence

    def add_holder(self, kind: str, holder: object) -> None:
        """A non-root holder (static_forward_context, the runner), once per object (target and drafter share them)."""
        if id(holder) not in self.holder_ids:
            count = sum(label.startswith(kind) for label, _, _ in self.roots)
            self.register(f"{kind}{count}", holder, None)

    def snapshot(self) -> dict[str, Entry]:
        entries, seen = {}, set()
        for label, ref, _ in self.roots:
            holder = ref()
            if holder is not None:
                entries.update((f"{label}:{name}", _entry(tensor)) for name, tensor in holder_tensors(holder, seen))
        return entries

    def _topk(self) -> tuple[str, str]:
        def storage(tensor: torch.Tensor | None) -> str:
            return "none" if tensor is None else f"{_entry(tensor).storage_ptr:#x}"

        shared = ",".join(f"{label}:{storage(state.shared)}" for label, _, state in self.roots if state is not None)
        private = ",".join(f"{label}:{storage(state.private)}" for label, _, state in self.roots if state is not None)
        return shared or "none", private or "none"

    def record(self, event: str, origin: str = "") -> list[str]:
        """Fence, snapshot; returns the names whose identity changed since the previous snapshot."""
        if self.fence is not None:
            self.fence()  # lazy storages are materialised before their pointers are read
        current = self.snapshot()
        changed = sorted(name for name in current.keys() | self.last.keys()
                         if self.last and current.get(name) != self.last.get(name))
        self.seq += 1
        self.events[event] += 1
        if changed:
            STATS.census_diffs += len(changed)
            detail = " ".join(f"{name}:{_short(self.last.get(name))}->{_short(current.get(name))}"
                              for name in changed[:CENSUS_DIFF_NAMES])
            scope = RUNNER_SCOPE if all(name.startswith(RUNNER_SCOPE) for name in changed) else "state"
            log_line(f"GLM53_GS_CENSUS_DIFF event={event} origin={origin or '-'} seq={self.seq} n={len(changed)} "
                     f"scope={scope} {detail}")
        if event in CENSUS_LOGGED_EVENTS or changed or self.events[event] % CENSUS_EVERY == 1:
            shared, private = self._topk()
            unreadable = sum(entry == UNREADABLE for entry in current.values())
            log_line(f"GLM53_GS_CENSUS event={event} origin={origin or '-'} seq={self.seq} tensors={len(current)} "
                     f"unreadable={unreadable} diffs={len(changed)} fenced={int(self.fence is not None)} "
                     f"roots={','.join(label for label, _, _ in self.roots)} "
                     f"topk={shared} private={private} events={dict(self.events)}")
        self.last = current
        return changed


def _reference(holder: object) -> Callable[[], object]:
    """A weak reference where the type allows one (modules, the runner); a dict (static_forward_context) lives as
    long as its vllm_config, so it is held directly."""
    try:
        return weakref.ref(holder)
    except TypeError:
        return lambda: holder


def _short(entry: Entry | None) -> str:
    return "absent" if entry is None else f"{entry.storage_ptr:#x}+{entry.data_ptr - entry.storage_ptr}/{entry.nbytes}"


STATS = Stats()
CENSUS = Census()
_STATES: weakref.WeakKeyDictionary[Any, GraphState] = weakref.WeakKeyDictionary()
_EXIT_LINE: list[Callable[[], None]] = []  # one exit GLM53_GS_STATS line per interpreter that loaded the runner


def reset_state() -> None:
    """Tests only: forget every root, counter and the warmup phase."""
    global STATS, CENSUS, LIFECYCLE
    STATS, CENSUS, LIFECYCLE = Stats(), Census(), Lifecycle()
    _STATES.clear()


def state_of(adapter: object) -> GraphState:
    return _STATES[adapter]


@contextlib.contextmanager
def private_key_bind() -> Iterator[int]:
    """Every root's holders -> private while a prompt key graph captures; yields how many states have a private copy."""
    states = list(_STATES.values())
    with contextlib.ExitStack() as binds:
        for state in states:
            binds.enter_context(state.bind())
        yield sum(state.private is not None for state in states)


# --- routing -------------------------------------------------------------------------------------------------------


def _closure(function: Callable[..., Any]) -> dict[str, Any]:
    code = getattr(function, "__code__", None)
    cells = getattr(function, "__closure__", None) or ()
    if code is None:
        return {}
    return {name: cell.cell_contents for name, cell in zip(code.co_freevars, cells, strict=True)}


def _never() -> bool:
    return False


def _cell_reader(function: Callable[..., Any], name: str) -> Callable[[], bool]:
    """The current value of one closure variable (a nonlocal the function rebinds), read at call time."""
    cell = dict(zip(function.__code__.co_freevars, function.__closure__ or (), strict=True))[name]

    def read() -> bool:
        return bool(cell.cell_contents)

    return read


class Route(NamedTuple):
    adaptive: Callable[..., Any]
    original: Callable[..., Any]
    bypass_min_tokens: int
    recapture_min_width: int
    should_bypass: Callable[[Any, int], bool]
    latched: Callable[[], bool] = _never  # the runner's dirty latch is set (its next graph steps defer or clear)


def source_sha(function: Callable[..., object]) -> str:
    """sha256 of the function's source text as its code object's file holds it (Python-version independent)."""
    try:
        source = inspect.getsource(function.__code__)
    except (AttributeError, OSError, TypeError) as error:
        raise GraphStateError(f"cannot read the source of {function!r} ({error}): the adaptive-route pin needs the "
                              "runner .py on disk; unset the graph-state flags for this arm") from error
    return hashlib.sha256(source.encode("utf-8")).hexdigest()


def read_route(adaptive: Callable[..., Any]) -> Route:
    """The runner's own adaptive configuration, read from adaptive_forward's closure (d23b680f 1730-1764)."""
    names = _closure(adaptive)
    missing = [name for name in (*ROUTE_VARS, LATCH_VAR) if name not in names]
    should_bypass = getattr(adaptive, "__globals__", {}).get("_should_bypass_hpu_graph_for_large_prefill")
    if missing or should_bypass is None:
        raise GraphStateError(f"runner adaptive_forward lacks {missing or ['_should_bypass_hpu_graph_for_large_prefill']}"
                              "; glm53_graph_state is written against the d23b680f adaptive route: re-check it "
                              "against this runner before enabling")
    sha = source_sha(adaptive)
    if sha != ADAPTIVE_SOURCE_SHA:
        raise GraphStateError(f"runner adaptive_forward source sha256 {sha[:12]} is not the pinned d23b680f "
                              f"{ADAPTIVE_SOURCE_SHA[:12]}: this module replays that function's large-prefill route "
                              "from its closure; diff the function against d23b680f 1766-1879, then re-pin "
                              f"ADAPTIVE_SOURCE_SHA, or unset {ENABLE_ENV}")
    if names["bypass_sparse_width_change"]:
        raise GraphStateError("HPU_GLM53_GRAPH_BYPASS_SPARSE_WIDTH_CHANGE=1 tracks widths inside adaptive_forward, "
                              f"which the suppressed route skips; unset it or unset {ENABLE_ENV}")
    return Route(adaptive, names["original_forward"], int(names["bypass_min_tokens"]),
                 int(names["recapture_min_sparse_width"]), should_bypass, _cell_reader(adaptive, LATCH_VAR))


def _input_ids(args: tuple, kwargs: Mapping[str, Any]) -> Any:
    import torch

    input_ids = kwargs.get("input_ids")
    if input_ids is None and args and isinstance(args[0], torch.Tensor):
        input_ids = args[0]
    return input_ids


def _sparse_width(kwargs: Mapping[str, Any]) -> int | None:
    import torch

    table = getattr(kwargs.get("attn_metadata"), "glm53_index_block_table", None)
    if isinstance(table, torch.Tensor) and table.dim() > 0:
        return int(table.shape[-1])
    return None


GRAPH, EAGER, SUPPRESS = "graph", "eager", "suppress"
CAPTURE = "capture"  # the check point after a decode step of a shape without a live graph


def classify(route: Route, args: tuple, kwargs: Mapping[str, Any]) -> str:
    """graph (replay/capture), eager (bypass or large prefill, runner route) or suppress (would set the dirty latch)."""
    input_ids = _input_ids(args, kwargs)
    large = route.should_bypass(input_ids, route.bypass_min_tokens)
    bypass = bool(kwargs.get("bypass_hpu_graphs", False))
    width = _sparse_width(kwargs)
    latches = bypass and large and width is not None and width >= route.recapture_min_width
    if latches and kwargs.get(REFRESH_KWARG, False):
        STATS.latch_external += 1
        return EAGER
    if latches:
        return SUPPRESS
    return EAGER if (large or bypass) else GRAPH


def _shape_of(value: object) -> object:
    """A value as habana's input_hash keys it (graphs.py 547-560) minus its per-process parts: tensors by dtype,
    shape and strides (never data or the view hash's storage part), plain values as themselves, other objects by
    type."""
    import torch

    if isinstance(value, torch.Tensor):
        return (str(value.dtype), tuple(value.shape), tuple(value.stride()))
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, (tuple, list)):
        return tuple(_shape_of(item) for item in value)
    return type(value).__name__


def step_shape(args: tuple, kwargs: Mapping[str, Any]) -> tuple:
    """The decode graph a step needs, the same on every TP rank: input_ids' shape (the batch bucket) and every
    attn_metadata field's shape or value by field name (block_list's bucket, glm53_index_block_table's width)."""
    meta = kwargs.get("attn_metadata")
    if hasattr(meta, "_asdict"):  # the runner's trim_attn_metadata namedtuple (runner 1947; field order per process)
        items = list(meta._asdict().items())
    elif is_dataclass(meta):
        items = [(item.name, getattr(meta, item.name)) for item in fields(meta)]
    else:
        items = list(getattr(meta, "__dict__", {}).items())
    return (_shape_of(_input_ids(args, kwargs)), tuple(sorted((name, _shape_of(value)) for name, value in items)))


def _rows(args: tuple, kwargs: Mapping[str, Any]) -> int:
    input_ids = _input_ids(args, kwargs)
    return int(input_ids.numel()) if input_ids is not None else 0


def _note_bind(rows: int) -> None:
    STATS.max_rows = max(STATS.max_rows, rows)
    if STATS.binds % BIND_LOG_EVERY == 1 and _rank() == 0:
        log_line(f"GLM53_GS_BIND n={STATS.binds} rows={STATS.max_rows}")


def _suppressed(route: Route, args: tuple, kwargs: dict[str, Any]) -> Any:
    """The runner's large-prefill route (d23b680f 1819-1837) without the latch it sets at 1805-1810."""
    from importlib import import_module

    kwargs.pop(REFRESH_KWARG, None)
    kwargs.pop(EXTERNAL_KWARG, None)
    decisions = STATS.latch_suppressed + STATS.latch_hbm
    if decisions <= SUPPRESS_LOG_FIRST or decisions % BIND_LOG_EVERY == 0:
        log_line(STATS.line())  # queue scripts save docker logs before stop, so the exit line never lands
    with import_module(COMMUNICATOR_MODULE).segmented_lazy_collectives():
        return route.original(*args, **kwargs)


def hbm_floor(environ: Mapping[str, str]) -> tuple[int, str]:
    """(minimum floor bytes, 'env' | 'default')."""
    raw = environ.get(HBM_FLOOR_ENV, "").strip()
    if raw and not raw.isdigit():
        raise GraphStateError(f"{HBM_FLOOR_ENV}={raw!r}: use a whole number of MiB (default {HBM_FLOOR_DEFAULT_MIB}); "
                              "it is the minimum floor, raised by the measured prompt peak + 2 x decode capture peak")
    return (int(raw) if raw else HBM_FLOOR_DEFAULT_MIB) * MIB, "env" if raw else "default"


def exhausted_rearm(environ: Mapping[str, str]) -> bool:
    raw = environ.get(REARM_ENV, "").strip() or "1"
    if raw not in ("0", "1"):
        raise GraphStateError(f"{REARM_ENV}={raw!r}: use 1 (default: an exhausted release pauses capture checks and "
                              "large eager chunks keep releasing runtime graphs) or 0 (RC4: exhausted stops the guard)")
    return raw == "1"


def long_context_tokens(environ: Mapping[str, str]) -> int:
    raw = environ.get(LONG_CONTEXT_ENV, "").strip()
    if raw and not raw.isdigit():
        raise GraphStateError(f"{LONG_CONTEXT_ENV}={raw!r}: use a whole number of context tokens (default "
                              f"{LONG_CONTEXT_DEFAULT}: a graph-bypassing prompt chunk past it is guarded like a large "
                              "one) or 0 (RC5: only chunks of at least the bypass rows are guarded)")
    return int(raw) if raw else LONG_CONTEXT_DEFAULT


def paused_spare(environ: Mapping[str, str]) -> bool:
    raw = environ.get(PAUSED_SPARE_ENV, "").strip() or "1"
    if raw not in ("0", "1"):
        raise GraphStateError(f"{PAUSED_SPARE_ENV}={raw!r}: use 1 (default: paused, release only for a chunk whose "
                              "predicted peak does not fit, rate-limited) or 0 (RC5: every paused chunk below the "
                              "floor releases)")
    return raw == "1"


@dataclass(frozen=True)
class SaturationPolicy:
    """Lane PEAK: what a saturated prompt-peak reading is charged instead (bytes / rows / context tokens)."""

    need: int  # the real-need prompt peak measured with every drain on
    margin: int  # MaxInUse within this of Limit is a saturated (run-ahead) reading
    max_rows: int  # the widest chunk rows the real need was measured for
    max_context: int  # the longest chunk context (Chunk.context units) it was measured for
    ceiling: int  # a covered reading above the need up to this is run-ahead; above it (not saturated) is genuine

    def covers(self, rows: int, context: int) -> bool:
        return rows <= self.max_rows and context <= self.max_context


def tripwire_v2(environ: Mapping[str, str]) -> bool:
    """HPU_GLM53_GS_TRIPWIRE_V2=1: chunks behind a capture of their own step stay out of the tripwire."""
    return _flag(environ, TRIPWIRE_V2_ENV)


def _positive_mib(environ: Mapping[str, str], name: str, default: int | None) -> int:
    raw = environ.get(name, "").strip()
    if not raw and default is not None:
        return default
    if not (raw.isascii() and raw.isdigit()) or int(raw) <= 0:
        raise GraphStateError(f"{name}={raw!r}: {SATURATED_PEAK_ENV}=1 needs a positive whole number here (see the "
                              "lane PEAK cell: real-need peak MiB / rows / context of the drains-on long C16 chunks); "
                              f"fix it or set {SATURATED_PEAK_ENV}=0")
    return int(raw)


def saturation_policy(environ: Mapping[str, str]) -> SaturationPolicy | None:
    """None (final-b) unless HPU_GLM53_GS_SATURATED_PEAK=1; then the real-need peak and its context are required."""
    raw = environ.get(SATURATED_PEAK_ENV, "").strip() or "0"
    if raw not in ("0", "1"):
        raise GraphStateError(f"{SATURATED_PEAK_ENV}={raw!r}: use 0 (default: every prompt peak as measured) or 1 "
                              "(a reading within the margin of the allocator Limit is charged the real-need peak)")
    if raw == "0":
        return None
    need = _positive_mib(environ, REAL_NEED_PEAK_ENV, None)
    margin = _positive_mib(environ, SATURATION_MARGIN_ENV, SATURATION_MARGIN_DEFAULT_MIB)
    if margin > SATURATION_MARGIN_MAX_MIB:
        raise GraphStateError(f"{SATURATION_MARGIN_ENV}={margin}: at most {SATURATION_MARGIN_MAX_MIB} MiB (final-b "
                              "saturated readings sit 18-64 MiB under Limit); a wider margin would replace honest "
                              f"peaks with the configured need. Lower it or set {SATURATED_PEAK_ENV}=0")
    reserve = hbm_floor(environ)[0] // MIB
    if need < reserve:
        raise GraphStateError(f"{REAL_NEED_PEAK_ENV}={need} is below the minimum reserve {reserve} MiB "
                              f"({HBM_FLOOR_ENV}): a need that small cannot be a measured prompt peak. Fill it from "
                              f"the lane PEAK cell's parse or set {SATURATED_PEAK_ENV}=0")
    ceiling = _positive_mib(environ, RUNAHEAD_CEILING_ENV, None)
    if ceiling < need:
        raise GraphStateError(f"{RUNAHEAD_CEILING_ENV}={ceiling} is below {REAL_NEED_PEAK_ENV}={need}: the ceiling "
                              "bounds the run-ahead readings above the need; set it above the largest drains-off "
                              f"covered reading (lane PEAK CELL.md) or set {SATURATED_PEAK_ENV}=0")
    return SaturationPolicy(need=need * MIB, margin=margin * MIB,
                            max_rows=_positive_mib(environ, REAL_NEED_MAX_ROWS_ENV, REAL_NEED_MAX_ROWS_DEFAULT),
                            max_context=_positive_mib(environ, REAL_NEED_MAX_CONTEXT_ENV, None), ceiling=ceiling * MIB)


def reading_class(policy: SaturationPolicy, measured: int, saturated: bool, covered: bool, free_at_start: int) -> str:
    """uncovered (outside the measured shape) | honest (at most the need) | genuine (above the ceiling, not
    saturated, and below GENUINE_FREE_FRACTION of the free it started with) | runahead (everything else: run-ahead
    fills whatever free a release left, so a large reading near the starting free is not evidence of need)."""
    if not covered:
        return UNCOVERED
    if saturated:
        return RUNAHEAD
    if measured <= policy.need:
        return HONEST
    if policy.ceiling < measured < GENUINE_FREE_FRACTION * free_at_start:
        return GENUINE
    return RUNAHEAD


class Reading(NamedTuple):
    """One guarded prompt forward as the saturation policy sees it."""

    measured: int  # MaxInUse - InUse at the start
    top: int  # MaxInUse
    chunk: Chunk
    ms: float  # host wall time of the forward


class Chunk(NamedTuple):
    """One prompt forward's geometry from shapes only (rank-symmetric, no device read): input rows, context tokens."""

    rows: int
    context: int


def prompt_context_tokens(kwargs: Mapping[str, Any]) -> int:
    """The padded context a prompt forward spans: block_list (every sequence's context blocks, padded to the batch's
    bucket, runner 4660/4676) x block_size; 0 for a decode step or a chunk without context."""
    import torch

    meta = kwargs.get("attn_metadata")
    if getattr(meta, "is_prompt", None) is not True:
        return 0
    blocks, block_size = getattr(meta, "block_list", None), getattr(meta, "block_size", None)
    if not isinstance(blocks, torch.Tensor) or not isinstance(block_size, int):
        return 0
    return int(blocks.numel()) * block_size


def prompt_request_context_tokens(args: tuple, kwargs: Mapping[str, Any]) -> int:
    """RC5.2: prompt_context_tokens per sequence. block_list is (batch bucket x context-block bucket) flattened
    (runner 4415), so this is the padded context of the batch's longest request, not bs x it."""
    input_ids = _input_ids(args, kwargs)
    shape = getattr(input_ids, "shape", ())
    batch = int(shape[0]) if len(shape) >= 2 else 1
    return prompt_context_tokens(kwargs) // max(1, batch)


def _chunk_key(chunk: Chunk) -> tuple[int, int]:
    """Power-of-two buckets of rows and context: a measured chunk bounds every chunk in buckets at or below it."""
    return 1 << max(0, chunk.rows - 1).bit_length(), 1 << max(0, chunk.context - 1).bit_length()


class TpRead(NamedTuple):
    """One check collective's answer (TP-min free / since, TP-max of the rest), plus this rank's free."""

    free: int
    tp_free: int
    floor: int
    runtime: int
    peak: int  # the chunk's predicted prompt peak
    chunk_floor: int  # max(minimum floor, predicted peak + CAPTURE_MARGIN x capture peak)
    since_ms: int  # since the last release (NEVER_RELEASED_MS before the first)
    covered: bool  # RC5.2: on EVERY rank a measured chunk bounds this one (else its peak is the prompt peak)
    ms: float


def guard_mode(environ: Mapping[str, str]) -> str:
    raw = environ.get(GUARD_ENV, "").strip() or GUARD_FLOOR
    if raw not in (GUARD_FLOOR, GUARD_LRU):
        raise GraphStateError(f"{GUARD_ENV}={raw!r}: use floor or lru (floor, the default: release runtime graphs "
                              "below the measured HBM floor; lru: the floor off, at most N runtime decode graphs, "
                              "least recently used dropped first)")
    return raw


def _tp_min(values: Sequence[int]) -> list[int]:
    """Elementwise minimum over the TP ranks in ONE gloo CPU all_reduce: every rank must pick the same route
    (runner 1843-1844). Outside an initialized process group, the values themselves."""
    import torch
    import torch.distributed as dist

    if not dist.is_available() or not dist.is_initialized():
        return list(values)
    from vllm.distributed import get_tp_group

    tensor = torch.tensor(list(values), dtype=torch.int64)
    dist.all_reduce(tensor, op=dist.ReduceOp.MIN, group=get_tp_group().cpu_group)
    return [int(value) for value in tensor.tolist()]


class Allocator(NamedTuple):
    """habana's allocator counters (habana_frameworks/torch/hpu/memory.py): InUse now, MaxInUse since the last
    reset_peak_memory_stats. Host-side reads, no device sync."""

    stats: Callable[[], Mapping[str, int]]
    reset_peak: Callable[[], None]

    def max_in_use(self) -> int:
        return int(self.stats()["MaxInUse"])

    def restart(self) -> int:
        """Restart the peak at the current InUse and return it."""
        self.reset_peak()
        return int(self.stats()["InUse"])

    def in_use(self) -> int:
        return int(self.stats()["InUse"])

    def free(self) -> int:
        """Limit - InUse: what the allocator can still hand out."""
        stats = self.stats()
        return int(stats["Limit"]) - int(stats["InUse"])

    def limit(self) -> int:
        """The allocator's Limit (what InUse can reach)."""
        return int(self.stats()["Limit"])


class PeakWindow:
    """The largest of the last PEAK_WINDOW measurements of one floor term, so a one-off peak (the profile run, a
    warmup shape) ages out instead of pinning the floor for the life of the process."""

    def __init__(self) -> None:
        self.samples: deque[int] = deque(maxlen=PEAK_WINDOW)

    @property
    def value(self) -> int:
        return max(self.samples, default=0)

    def add(self, nbytes: int) -> bool:
        """Record one measurement; True when the term's MiB value changed (a rise, or an old peak aged out)."""
        before = self.value // MIB
        self.samples.append(max(0, nbytes))
        return self.value // MIB != before


@dataclass
class Lifecycle:
    """Process phase: checks and releases start once warmup_model returned (warmup graphs are the boot set)."""

    warmup_hooked: bool = False
    serving: bool = False
    guards: list[weakref.ref] = field(default_factory=list)

    def checks_on(self) -> bool:
        return self.serving or not self.warmup_hooked

    def live_guards(self) -> Iterator[HbmGuard]:
        for ref in self.guards:
            guard = ref()
            if guard is not None:
                yield guard

    def start_serving(self) -> None:
        """warmup_model returned: check points start, and each live guard logs its floor next to the free HBM."""
        self.serving = True
        for guard in self.live_guards():
            guard.log_inert()


LIFECYCLE = Lifecycle()


class GraphStep(NamedTuple):
    """What a graph forward saw before it ran: the cache size and InUse (-1: a shape with a live graph)."""

    graphs: int
    in_use: int


class HbmGuard:
    """Suppressing the latch keeps every runtime-captured decode graph resident, because the clear was their only
    release. Each check point (a suppression decision, a large eager forward, a decode step of a new shape) asks one
    collective whether the TP-min free HBM is still above the TP-max floor; below it the graphs captured after the
    boot seal are dropped (fenced) before the next forward runs. The boot set (warmup graphs, pgk's warmed keys) is
    never dropped. A release that cannot lift the TP-min free above the floor stops the guard (event=exhausted)."""

    def __init__(self, free_bytes: Callable[[], float], graph_cache: MutableMapping[object, object],
                 floor: tuple[int, str], allocator: Allocator, fence: Callable[[], None], *,
                 guard: str = GUARD_FLOOR, rearm: bool = True, long_context: int = LONG_CONTEXT_DEFAULT,
                 spare: bool = True, saturation: SaturationPolicy | None = None,
                 skip_capture_steps: bool = False, witness: bool = False) -> None:
        self.witness = witness  # step 23, HPU_GLM53_GRAPHS_ONLY=1: report only, never a floor, a clear or a release
        self.free_bytes, self.graph_cache, self.allocator, self.fence = free_bytes, graph_cache, allocator, fence
        self.saturation = saturation  # lane PEAK (None: every prompt peak as measured, final-b)
        self.saturation_counts: Counter[str] = Counter()  # charged / uncovered / limit_errors (flag on only)
        self.genuine_window = PeakWindow()  # flag on: serving readings of covered chunks above the ceiling
        self.uncovered_seed = 0  # flag on: the uncovered-chunk fallback peak, fixed at the seal
        self.warmup_raw_peak = 0  # flag on: the largest RAW (uncharged) reading before the seal
        self.clock: Callable[[], float] = time.perf_counter  # flag on: run-ahead chunk prefill timing (tripwire)
        self.bucket_ms: dict[tuple[int, int], deque[float]] = {}
        self.tripwire_charge, self.tripwire_left = 0, 0
        self.skip_capture_steps = skip_capture_steps  # tripwire v2: capture steps neither sample nor trip
        self.step_captured = False  # tripwire v2: the latest graph forward captured and no reading has consumed it
        if skip_capture_steps and saturation is None:
            log_line(f"GLM53_GS_SATURATION event=inert {TRIPWIRE_V2_ENV}=1 has no effect without {SATURATED_PEAK_ENV}=1 "
                     f"(no tripwire runs); set both or unset it pid={os.getpid()}")
        if saturation is not None and guard == GUARD_LRU and GUARD_LRU not in SATURATION_LRU_WARNED:
            SATURATION_LRU_WARNED.add(GUARD_LRU)
            log_line(f"GLM53_GS_SATURATION event=inert guard=lru: {SATURATED_PEAK_ENV}=1 has no effect under "
                     f"{GUARD_ENV}=lru (no floor is charged); use the floor guard or set {SATURATED_PEAK_ENV}=0 "
                     f"pid={os.getpid()}")
        self.rearm = rearm  # C613: exhausted pauses capture checks; eager checks keep releasing (False: RC4 stop)
        self.retained_window = PeakWindow()  # C613 diagnostic: InUse a large eager chunk leaves behind (logged)
        self.long_context = long_context  # RC5.1: bypassing prompt chunks past this context are guarded (0: RC5)
        self.spare = spare  # RC5.1: paused releases only for a chunk whose predicted peak does not fit (False: RC5)
        self.chunk_peaks: dict[tuple[int, int], PeakWindow] = {}  # (rows, context) bucket -> measured prompt peaks
        self.released_at: float | None = None  # this rank's monotonic time of the last release
        self.lru = guard == GUARD_LRU  # the floor is off: no check points, the runtime graphs are capped instead
        self.live: OrderedDict[tuple, frozenset[object]] = OrderedDict()  # lru: runtime shape -> its own keys
        self.strays: set[object] = set()  # runtime keys grown on a shape without an LRU entry (dropped at evictions)
        self.tracked: set[object] = set()  # every runtime key in live or strays
        self.cap = 0  # lru: this guard's most runtime graphs (sealed once, rank-symmetric)
        self.resident_peak = 0  # the largest InUse growth over one capture (what a kept graph costs)
        STATS.guard = guard
        self.min_floor_bytes, self.source = floor
        self.prompt_window, self.capture_window = PeakWindow(), PeakWindow()
        self.pending = False  # a capture check fell below the floor: release before the next forward
        self.boot_keys: frozenset[object] | None = None  # the cache keys at the boot seal (never released)
        self.boot_shapes: frozenset[tuple] = frozenset()
        self.shapes: set[tuple] = set()  # decode step shapes with a live graph: the boot set + since the last release
        self.released_shapes: set[tuple] = set()
        self.base = allocator.restart()  # InUse where the current peak window started
        self._log_floor("boot")
        if witness:
            log_line(f"GLM53_GS_WITNESS mode=graphs_only floor=report_only clear=refused release=refused tripwire=log_only "
                     f"headroom_gate_mib={WITNESS_HEADROOM_MIB} pid={os.getpid()}")

    @property
    def prompt_peak(self) -> int:
        return self.prompt_window.value

    @property
    def capture_peak(self) -> int:
        return self.capture_window.value

    @property
    def retained_peak(self) -> int:
        return self.retained_window.value

    def floor_bytes(self) -> int:
        """The retained term is NOT added: the prompt peak already spans the bytes a chunk keeps, and earlier
        chunks' kept bytes are already out of the free a check reads (adding it would count them twice)."""
        return max(self.min_floor_bytes, self.prompt_peak + CAPTURE_MARGIN * self.capture_peak)

    def covering_peak(self, chunk: Chunk) -> int | None:
        """The largest measured prompt peak of a chunk at least as wide in rows AND context (bucketed); None when
        no measurement covers this chunk."""
        rows, context = _chunk_key(chunk)
        policy = self.saturation  # flag on: a covered chunk is bounded by covered buckets only
        only_covered = policy is not None and policy.covers(chunk.rows, chunk.context)
        bounding = [window.value for (key_rows, key_context), window in self.chunk_peaks.items()
                    if key_rows >= rows and key_context >= context and window.samples
                    and (not only_covered or policy.covers(key_rows, key_context))]
        return max(bounding) if bounding else None

    def admission_floor(self, chunk: Chunk | None) -> int:
        """The floor a check votes. Flag off, or a covered chunk / no chunk: floor_bytes() (final-b). Flag on and an
        uncovered chunk: also its predicted peak (its covering bucket, else the largest uncovered warmup peak) +
        CAPTURE_MARGIN x capture peak, so an 8192-row or past-max-context chunk is never admitted at the need."""
        floor, policy = self.floor_bytes(), self.saturation
        if policy is None or chunk is None or policy.covers(chunk.rows, chunk.context):
            return floor
        covering = self.covering_peak(chunk)
        predicted = self.uncovered_seed if covering is None else covering
        return max(floor, self.min_floor_bytes, predicted + CAPTURE_MARGIN * self.capture_peak)

    def predicted_peak(self, chunk: Chunk) -> int:
        """covering_peak, else the floor's prompt peak (the largest of every chunk: RC5's answer)."""
        covering = self.covering_peak(chunk)
        return self.prompt_peak if covering is None else covering

    def chunk_floor(self, chunk: Chunk | None) -> tuple[int, int]:
        """(predicted peak, max(minimum floor, predicted peak + CAPTURE_MARGIN x capture peak)); (0, 0) without one."""
        if chunk is None:
            return 0, 0
        peak = self.predicted_peak(chunk)
        return peak, max(self.min_floor_bytes, peak + CAPTURE_MARGIN * self.capture_peak)

    def is_guarded(self, large: bool, args: tuple, kwargs: Mapping[str, Any]) -> bool:
        """A forward the guard checks and measures: the runner's large prefill, or (RC5.1) any graph-bypassing prompt
        chunk whose per-request context (RC5.2: not bs x it) is past ``long_context`` tokens (the 256/128-row
        rungs of a > ~270k prompt)."""
        if large:
            return True
        return (self.long_context > 0 and bool(kwargs.get("bypass_hpu_graphs", False))
                and prompt_request_context_tokens(args, kwargs) > self.long_context)

    def runtime_graphs(self) -> int:
        """Graph keys captured after the boot seal (what a release can drop); 0 before the seal."""
        if self.boot_keys is None:
            return 0
        return sum(1 for key in self.graph_cache if key not in self.boot_keys)

    def graphs(self) -> int:
        return len(self.graph_cache)

    def seal_boot(self) -> None:
        """The first serving forward (warmup_model and pgk's seal have returned): every key in the cache now is the
        boot set, and every decode shape seen so far has a boot graph."""
        if self.boot_keys is not None:
            return
        self.boot_keys = frozenset(self.graph_cache)
        self.boot_shapes = frozenset(self.shapes)
        self._log_floor("seal", rows=len(self.boot_keys))
        if self.saturation is not None:
            self._expire_warmup_buckets()
        if self.lru:
            self._set_cap()

    def _expire_warmup_buckets(self) -> None:
        """Flag on, at the seal: drop the warmup chunk-bucket peaks of COVERED buckets and restart the global prompt
        window at the need (warmup's ~8 GB samples held the floor for 64 serving chunks, final-g quad A). Uncovered
        buckets survive, and their largest peak seeds admission_floor for an uncovered chunk with no covering bucket,
        so an 8192-row or past-max-context chunk is never admitted at the need. The seed is the larger of those and
        the largest RAW warmup reading, so a warmup of covered chunks only still seeds it."""
        policy = self.saturation
        covered = [key for key in self.chunk_peaks if policy.covers(*key)]
        for key in covered:
            del self.chunk_peaks[key]
        self.uncovered_seed = max(self.warmup_raw_peak,
                                  max((window.value for window in self.chunk_peaks.values()), default=0))
        self.bucket_ms.clear()  # tripwire history starts at the seal (no warmup compile timings)
        samples = len(self.prompt_window.samples)
        self.prompt_window = PeakWindow()
        self.prompt_window.add(policy.need)
        log_line(f"GLM53_GS_SATURATION event=seal expired_buckets={len(covered)} "
                 f"kept_uncovered_buckets={len(self.chunk_peaks)} uncovered_seed_mib={self.uncovered_seed // MIB} "
                 f"expired_prompt_samples={samples} seeded_mib={policy.need // MIB} pid={os.getpid()}")
        self._log_floor("seal-expiry")

    def graph_bytes(self) -> int:
        """What one more kept graph costs: the largest measured capture resident, else the r6 default."""
        return self.resident_peak or LRU_DEFAULT_GRAPH_MIB * MIB

    def _alloc_free(self) -> int:
        try:
            return self.allocator.free()
        except Exception as error:  # every rank must still join the collective (a skipped all_reduce hangs TP)
            log_line(f"GLM53_GS_HBM_ERROR allocator free failed ({error!r}): this rank sends FREE_ERROR "
                     f"pid={os.getpid()}")
            return FREE_ERROR

    def _set_cap(self) -> None:
        """This guard's N = (TP-min (Limit - InUse) - 512 MiB) // TP-max capture resident, in ONE collective at the
        seal (every rank seals at its first serving forward), at least 1. A failed read caps at 1 (source=error:
        the soak-lru grade fails it, a cap of 1 thrashes)."""
        free = self._alloc_free()
        tp_free, tp_neg_graph = _tp_min([free, -self.graph_bytes()])
        graph = -tp_neg_graph
        self.cap = max(1, (tp_free - LRU_RESERVE_MIB * MIB) // graph) if tp_free >= 0 else 1
        source = "error" if tp_free < 0 else ("measured" if self.resident_peak else "default")
        log_line(f"GLM53_GS_LRU event=cap cap={self.cap} tp_min_free_mib={tp_free // MIB} graph_mib={graph // MIB} "
                 f"reserve_mib={LRU_RESERVE_MIB} source={source} boot_graphs={len(self.boot_keys or ())} "
                 f"pid={os.getpid()}")

    def before_graph(self, shape: tuple | None) -> GraphStep:
        """Before a graph forward. A shape with a live graph is a replay (its LRU entry is touched); a new one
        captures on every rank, so under the LRU it first makes room (one collective, fenced evictions)."""
        graphs = self.graphs()
        if shape is None or shape in self.shapes:
            if shape in self.live:
                self.live.move_to_end(shape)
            return GraphStep(graphs, -1)
        if self.lru and self.boot_keys is not None:
            self._make_room()
        return GraphStep(graphs, self.allocator.in_use())

    def _make_room(self) -> None:
        """ONE collective [free, -(P + G), -G]: evict least recently used shapes until the TP-min free covers the
        TP-max eager chunk spike P plus the graph about to be captured G (each eviction frees about G), every
        runtime shape when a rank's read failed, and at least enough to stay under this guard's cap."""
        graph = self.graph_bytes()
        tp_free, tp_neg_need, tp_neg_graph = _tp_min([self._alloc_free(), -(self.prompt_peak + graph), -graph])
        need, graph = -tp_neg_need, -tp_neg_graph
        short = len(self.live) * graph if tp_free < 0 else need - tp_free
        for_room = -(-short // graph) if short > 0 else 0
        count = min(len(self.live), max(for_room, len(self.live) - self.cap + 1))
        log_line(f"GLM53_GS_LRU event=room tp_min_free_mib={tp_free // MIB} need_mib={need // MIB} "
                 f"graph_mib={graph // MIB} evict={count} live={len(self.live)} cap={self.cap} pid={os.getpid()}")
        if count or (short > 0 and self.strays):
            self._evict(count, "room" if for_room >= count else "cap")

    def _evict(self, count: int, reason: str) -> None:
        """Drop the ``count`` least recently used shapes' keys and every stray key, inside one fence pair."""
        victims = [self.live.popitem(last=False) for _ in range(count)]
        strays, self.strays = self.strays, set()
        self.fence()
        for _, keys in victims:
            self._drop(keys)
        self._drop(strays)
        self.fence()
        STATS.released += len(strays)
        for shape, keys in victims:
            self.shapes.discard(shape)
            self.released_shapes.add(shape)
            STATS.evictions += 1
            STATS.released += len(keys)
            log_line(f"GLM53_GS_LRU event=evict dropped={len(keys)} live={len(self.live)} cap={self.cap} "
                     f"evictions={STATS.evictions} graphs={self.graphs()} reason={reason} strays={len(strays)} "
                     f"pid={os.getpid()}")

    def _drop(self, keys: set[object] | frozenset[object]) -> None:
        for key in keys:
            self.tracked.discard(key)
            if key in self.graph_cache:
                del self.graph_cache[key]

    def _runtime_untracked(self) -> list[object]:
        boot = self.boot_keys or frozenset()
        return [key for key in self.graph_cache if key not in boot and key not in self.tracked]

    def _track(self, shape: tuple | None) -> None:
        """Cache growth after a serving graph forward: the new keys join the shape's LRU entry (input_hash is
        rank-local, so a known shape can capture too), else the stray set the next eviction drops."""
        new = frozenset(self._runtime_untracked())
        self.tracked |= new
        if shape in self.live:
            self.live[shape] |= new
        else:
            self.strays |= new
    def tp_free_floor(self, chunk: Chunk | None = None) -> TpRead:
        """This rank's free, then in ONE gloo collective: TP-min free, TP-max floor, TP-max runtime graphs, the
        chunk's TP-max predicted peak and floor, TP-min time since the last release, and (RC5.2) whether a measured
        chunk covers this one on every rank. A rank whose free read fails still joins it, with FREE_ERROR, so no
        rank waits in a collective another rank skipped. The runtime count makes "is there anything to release" a
        TP-wide answer; every rank sends the same seven values."""
        free, floor = self._read_free(), self.admission_floor(chunk)
        peak, chunk_floor = self.chunk_floor(chunk)
        covered = int(chunk is not None and self.covering_peak(chunk) is not None)
        since_ms = (NEVER_RELEASED_MS if self.released_at is None
                    else int((time.monotonic() - self.released_at) * 1000.0))
        started = time.perf_counter()
        tp_free, neg_floor, neg_runtime, neg_peak, neg_chunk_floor, tp_since, tp_covered = _tp_min(
            [free, -floor, -self.runtime_graphs(), -peak, -chunk_floor, since_ms, covered])
        return TpRead(free, tp_free, -neg_floor, -neg_runtime, -neg_peak, -neg_chunk_floor, tp_since,
                      tp_covered == 1, (time.perf_counter() - started) * 1000.0)

    def _read_free(self) -> int:
        try:
            return int(self.free_bytes())
        except Exception as error:  # every rank must still reach the collective (a skipped all_reduce hangs TP)
            log_line(f"GLM53_GS_HBM_ERROR free_bytes failed ({error!r}): this rank sends FREE_ERROR, every rank stops "
                     f"checking (graphs stay resident) pid={os.getpid()}")
            return FREE_ERROR

    def check(self, point: str, chunk: Chunk | None = None) -> bool:
        """One check point: TP-min free HBM >= TP-max floor. Every check point is a forward all TP ranks run with
        the same shapes, so every rank makes the same call. A stopped guard keeps every graph without a collective.
        ``chunk``: the eager / suppress forward about to run (its predicted peak decides paused releases)."""
        if self.witness:
            return self._witness_check(point, chunk)
        if self._checks_off(point):
            return True
        read = self.tp_free_floor(chunk)
        tp_free, floor = read.tp_free, read.floor
        if tp_free < 0:
            self._stop(STOP_ERROR, tp_free, floor)
            return True
        keep = tp_free >= floor
        STATS.note_check(point, keep, read.ms)
        starved = not keep and self.rearm and read.runtime == 0  # below the floor with nothing to release
        spared = not keep and not starved and self._spared(point, read)
        action = ("suppress" if point == SUPPRESS else "keep") if keep else \
            ("starved" if starved else "spare" if spared else "clear")
        log_line(f"GLM53_GS_HBM point={point} graphs={self.graphs()} free_mib={read.free // MIB} "
                 f"tp_min_free_mib={tp_free // MIB} floor_mib={floor // MIB} action={action} "
                 f"latch_hbm={STATS.latch_hbm} max_in_use_mib={self.allocator.max_in_use() // MIB} "
                 f"tp_runtime={read.runtime} rows={chunk.rows if chunk else 0} "
                 f"context={chunk.context if chunk else 0} peak_mib={read.peak // MIB} "
                 f"chunk_floor_mib={read.chunk_floor // MIB} covered={int(read.covered)} tp_min_ms={read.ms:.3f} pid={os.getpid()}")
        if keep and STATS.stopped == STOP_EXHAUSTED:
            STATS.stopped = NOT_STOPPED  # the floor fits again: capture checks resume (same answer on every rank)
            self._log_floor("rearm", extra=f" tp_min_free_mib={tp_free // MIB} tp_max_floor_mib={floor // MIB}")
        return keep or starved or spared

    def _witness_check(self, point: str, chunk: Chunk | None) -> bool:
        """GRAPHS_ONLY: the same collective, so the TP-min free is on the record, but its answer decides nothing: always keep. A rank whose
        free read failed makes the answer negative on every rank at once, and every rank raises (a real allocation failure is not retried)."""
        read = self.tp_free_floor(chunk)
        if read.tp_free < 0:
            raise GraphStateError(f"{GRAPHS_ONLY_ENV}=1: a rank could not read its free HBM at point={point}, so no rank can vouch for the "
                                  "graphs it holds. Every rank stops here; read the GLM53_GS_HBM_ERROR line, fix the device read and "
                                  "restart all workers")
        STATS.note_check(point, True, read.ms)
        log_line(f"GLM53_GS_HBM point={point} graphs={self.graphs()} free_mib={read.free // MIB} tp_min_free_mib={read.tp_free // MIB} "
                 f"min_floor_mib={self.min_floor_bytes // MIB} action=report witness=1 max_in_use_mib={self.allocator.max_in_use() // MIB} "
                 f"rows={chunk.rows if chunk else 0} context={chunk.context if chunk else 0} tp_min_ms={read.ms:.3f} pid={os.getpid()}")
        return True

    def _refuse_witness_growth(self, shape: tuple | None, before: int) -> None:
        """GRAPHS_ONLY: the decode check reads a Python set, so a graph-hash miss would capture here silently. The cache holds the boot set
        and nothing else; a rank that grows it exits in a bounded time so a peer waiting on the capture's collective stops with it."""
        arm_exit("GLM53_GS_GROWTH")
        raise GraphStateError(f"{GRAPHS_ONLY_ENV}=1: the graph cache grew from {before} to {self.graphs()} graphs after the seal, on the step of shape "
                              f"{shape}. Every graph is captured at boot under this flag, so this step needs a key that boot never captured; "
                              "check HPU_GLM53_DECODE_WIDTH_PAD and VLLM_GAUDI_MIXED_DECODE_WIDTHS against the shapes the runner builds, and restart")

    def _refuse_witness_release(self, action: str) -> None:
        if self.witness:
            raise GraphStateError(f"{GRAPHS_ONLY_ENV}=1: the guard never {action} a graph, since a graph dropped here is one the next forward "
                                  "must recapture or run eager. This is a bug in the caller: it must not reach a release under the flag")

    def _spared(self, point: str, read: TpRead) -> bool:
        """RC5.1, paused eager / suppress checks below the floor: keep the runtime graphs when this chunk's predicted
        peak + margin fits the TP-min free, or when the peak alone fits and the last release (TP-min) was under
        PAUSED_RELEASE_MIN_S ago. Only (RC5.2) when a measured chunk covers this one on every rank: a predicted
        peak that fell back to the prompt peak releases as RC5 does. Every input is a collective answer, so every
        rank decides the same."""
        if not (self.spare and self.rearm and STATS.stopped == STOP_EXHAUSTED and point != CAPTURE):
            return False
        if read.chunk_floor <= 0 or not read.covered:  # RC5.2: no covering measurement on some rank: release (RC5)
            return False
        if read.tp_free >= read.chunk_floor:
            return True
        return read.tp_free >= read.peak and read.since_ms < PAUSED_RELEASE_MIN_S * 1000.0

    def _checks_off(self, point: str) -> bool:
        """No check (and no collective) at this point. Every input is rank-symmetric: the point comes from the
        forward's shape and ``stopped`` only changes on a collective's answer."""
        if self.lru or STATS.stopped == STOP_ERROR:
            return True
        if STATS.stopped != STOP_EXHAUSTED:
            return False
        # Exhausted: RC4 stopped here for good. Re-armed, only capture checks pause (no capture / release /
        # recapture thrash per decode step); every large eager chunk still checks and releases first.
        return not self.rearm or point == CAPTURE

    def drop_runtime(self) -> int:
        """Delete every graph captured after the boot seal (habana graphs.py 908's clear_cache is cache.clear(),
        which would also drop the warmup decode buckets and pgk's replay-only keys). Unsealed: nothing is runtime."""
        self._refuse_witness_release("drops")
        boot = self.boot_keys if self.boot_keys is not None else frozenset(self.graph_cache)
        runtime = [key for key in self.graph_cache if key not in boot]
        for key in runtime:
            del self.graph_cache[key]
        self.live.clear()
        self.strays.clear()
        self.tracked.clear()
        self.released_shapes |= self.shapes - self.boot_shapes
        self.shapes = set(self.boot_shapes)
        STATS.released += len(runtime)
        return len(runtime)

    def release(self, chunk: Chunk | None = None) -> None:
        """fence + drop the runtime graphs + fence, before a forward, then one more collective: a release that left
        the TP-min free below the floor cannot be repeated usefully (every later capture would be dropped again)."""
        self._refuse_witness_release("releases")
        self.pending = False
        self.fence()
        dropped = self.drop_runtime()
        self.fence()
        STATS.clears_hbm += 1
        self.released_at = time.monotonic()
        read = self.tp_free_floor(chunk if self.saturation is not None else None)  # flag off: final-b's vote
        tp_free, floor = read.tp_free, read.floor
        result = "ok" if tp_free >= floor else (STOP_ERROR if tp_free < 0 else STOP_EXHAUSTED)
        log_line(f"GLM53_GS_RELEASE dropped={dropped} graphs={self.graphs()} tp_min_free_mib={tp_free // MIB} "
                 f"floor_mib={floor // MIB} result={result} pid={os.getpid()}")
        if result == "ok" and STATS.stopped == STOP_EXHAUSTED:
            STATS.stopped = NOT_STOPPED
            self._log_floor("rearm", extra=f" tp_min_free_mib={tp_free // MIB} tp_max_floor_mib={floor // MIB}")
        elif result == STOP_EXHAUSTED and self.rearm:
            self._pause(tp_free, floor)
        elif result != "ok":
            self._stop(result, tp_free, floor)
        log_line(STATS.line())

    def _pause(self, tp_free: int, floor: int) -> None:
        """C613: a release that could not reach the floor pauses capture checks only; every large eager chunk still
        checks and drops the runtime graphs captured since, and the first check above the floor re-arms."""
        STATS.stopped = STOP_EXHAUSTED
        self._log_floor(STOP_EXHAUSTED, extra=f" tp_min_free_mib={tp_free // MIB} tp_max_floor_mib={floor // MIB} "
                        "checks=eager rearm=1")

    def _stop(self, reason: str, tp_free: int, floor: int) -> None:
        """No more check points (the same decision on every rank: it comes from a collective); every graph stays
        resident for the rest of the process."""
        STATS.stopped = reason
        self._log_floor(reason, extra=f" tp_min_free_mib={tp_free // MIB} tp_max_floor_mib={floor // MIB} checks=off")

    def prompt(self, run: Callable[[], T], suppressed: bool, rows: int, context: int = 0) -> T:
        """Run one large eager forward, measuring its peak above the InUse it started at (also per chunk bucket)."""
        in_use = self.allocator.restart()
        policy = self.saturation
        started = 0.0 if policy is None else self.clock()
        result = run()
        top = self.allocator.max_in_use()
        peak = top - in_use if policy is None else self._charged_peak(
            policy, Reading(top - in_use, top, Chunk(rows, context), (self.clock() - started) * 1000.0))
        self._note("prompt", peak, rows)
        self.chunk_peaks.setdefault(_chunk_key(Chunk(rows, context)), PeakWindow()).add(peak)
        self._note("retained", self.allocator.in_use() - in_use, rows)
        if suppressed:
            log_line(f"GLM53_GS_PEAK kind=suppress rows={rows} in_use_mib={in_use // MIB} max_in_use_mib={top // MIB} "
                     f"delta_mib={max(0, top - in_use) // MIB} floor_mib={self.floor_bytes() // MIB} pid={os.getpid()}")
        self.base = self.allocator.restart()
        return result

    def _charged_peak(self, policy: SaturationPolicy, reading: Reading) -> int:
        """Lane PEAK: a run-ahead reading of a covered chunk (saturated, or above the need without being genuine) is
        charged max(real-need peak, the largest genuine serving reading in the window, the tripwire charge); honest,
        genuine and uncovered readings are charged as measured. A failed or invalid Limit read never lowers a peak
        (the measured one is charged and the line says limit_error)."""
        limit = self._read_limit()
        if limit is None:
            return reading.measured
        chunk, headroom = reading.chunk, limit - reading.top
        free_at_start = limit - (reading.top - reading.measured)
        saturated = headroom <= policy.margin
        covered = policy.covers(chunk.rows, chunk.context)
        kind = reading_class(policy, reading.measured, saturated, covered, free_at_start)
        if self.boot_keys is None:
            self.warmup_raw_peak = max(self.warmup_raw_peak, reading.measured)
        if kind == GENUINE and self.boot_keys is not None:  # warmup readings never calibrate serving charges
            self.genuine_window.add(reading.measured)
        captured, self.step_captured = self.step_captured, False
        if covered:
            self._tripwire(kind, reading, free_at_start, captured)
        tripped = self.tripwire_charge if self.tripwire_left > 0 else 0
        charged = max(policy.need, self.genuine_window.value, tripped) if kind == RUNAHEAD else reading.measured
        self.saturation_counts["charged"] += int(kind == RUNAHEAD)
        self.saturation_counts["sat_uncovered"] += int(saturated and not covered)
        log_line(f"GLM53_GS_SATURATION rows={chunk.rows} context={chunk.context} saturated={int(saturated)} "
                 f"covered={int(covered)} measured_mib={reading.measured // MIB} charged_mib={charged // MIB} "
                 f"headroom_mib={headroom // MIB} graphs={self.graphs()} runtime={self.runtime_graphs()} "
                 f"charged_n={self.saturation_counts['charged']} "
                 f"sat_uncovered_n={self.saturation_counts['sat_uncovered']} class={kind} ms={reading.ms:.1f} "
                 f"tripwire_left={self.tripwire_left} pid={os.getpid()}")
        return charged

    def _tripwire(self, kind: str, reading: Reading, free_at_start: int, captured: bool) -> None:
        """OOM / allocation-retry tripwire, covered chunks only: a run-ahead chunk slower than TRIPWIRE_SLOWDOWN x
        its bucket's rolling p50 while runtime graphs are resident charges its free-at-start for the next
        TRIPWIRE_CHUNKS covered chunks (this one included), so the floor rises above free and the graphs release.
        ``captured`` (tripwire v2 only): this chunk's step captured a graph, so its timing says nothing about the chunk."""
        if self.tripwire_left > 0:
            self.tripwire_left -= 1
            if self.tripwire_left == 0:
                self._purge_tripped()
        if kind != RUNAHEAD or self.boot_keys is None:
            return
        if captured:
            self._note_capture_skip(reading)
            return
        chunk = reading.chunk
        cost = reading.ms / max(1, chunk.rows * (chunk.context + chunk.rows))  # ms per unit of prefill work
        history = self.bucket_ms.setdefault(_chunk_key(chunk), deque(maxlen=TRIPWIRE_HISTORY))
        p50 = sorted(history)[len(history) // 2] if len(history) >= TRIPWIRE_MIN_SAMPLES else None
        history.append(cost)
        runtime = self.runtime_graphs()
        resident = self.graphs() > 0 if self.witness else runtime > 0  # witness: every graph is a boot graph, so runtime is always 0
        if p50 is None or not resident or cost <= TRIPWIRE_SLOWDOWN * p50:
            return
        if self.witness:
            self._witness_trip(reading, cost / p50, free_at_start - reading.measured)
            return
        self.tripwire_charge, self.tripwire_left = free_at_start, TRIPWIRE_CHUNKS
        self.saturation_counts["tripwires"] += 1
        log_line(f"GLM53_GS_SATURATION event=tripwire rows={reading.chunk.rows} context={reading.chunk.context} "
                 f"ms={reading.ms:.1f} cost_vs_p50={cost / p50:.2f} runtime={runtime} charge_mib={free_at_start // MIB} "
                 f"chunks={TRIPWIRE_CHUNKS} tripwires={self.saturation_counts['tripwires']} pid={os.getpid()}")

    def _witness_trip(self, reading: Reading, slowdown: float, headroom: int) -> None:
        """GRAPHS_ONLY: a slow run-ahead chunk is an allocation-retry signal only when under WITNESS_HEADROOM_MIB of the Limit was left above its
        peak. Logged and counted, never charged, never released: nothing the guard could do about it would not drop a graph."""
        if headroom >= WITNESS_HEADROOM_MIB * MIB:
            return
        self.saturation_counts["tripwires"] += 1
        log_line(f"GLM53_GS_SATURATION event=tripwire rows={reading.chunk.rows} context={reading.chunk.context} ms={reading.ms:.1f} "
                 f"cost_vs_p50={slowdown:.2f} graphs={self.graphs()} headroom_mib={headroom // MIB} log_only=1 "
                 f"tripwires={self.saturation_counts['tripwires']} pid={os.getpid()}")

    def _note_capture_skip(self, reading: Reading) -> None:
        self.saturation_counts["tripwire_capture_skips"] += 1
        skips = self.saturation_counts["tripwire_capture_skips"]
        if skips % TRIPWIRE_SKIP_LOG_EVERY == 1:
            log_line(f"GLM53_GS_SATURATION event=tripwire_capture_skip rows={reading.chunk.rows} "
                     f"context={reading.chunk.context} ms={reading.ms:.1f} skips={skips} pid={os.getpid()}")

    def _purge_tripped(self) -> None:
        """The tripwire countdown ended: remove its charged samples from the prompt window and every chunk bucket
        (aging them out would hold the floor for up to 64 more chunks); an emptied prompt window restarts at the
        need."""
        charge = self.tripwire_charge
        for window in (self.prompt_window, *self.chunk_peaks.values()):
            window.samples = deque((value for value in window.samples if value != charge), maxlen=PEAK_WINDOW)
        if not self.prompt_window.samples:
            self.prompt_window.add(self.saturation.need)
        self.tripwire_charge = 0
        log_line(f"GLM53_GS_SATURATION event=tripwire_end prompt_peak_mib={self.prompt_peak // MIB} pid={os.getpid()}")

    def _read_limit(self) -> int | None:
        try:
            limit = self.allocator.limit()
        except Exception as error:  # noqa: BLE001  an unreadable Limit must not lower a floor
            limit, reason = None, repr(error)
        else:
            reason = "" if limit > 0 else f"Limit={limit}"
        if not reason:
            return limit
        self.saturation_counts["limit_errors"] += 1
        log_line(f"GLM53_GS_SATURATION event=limit_error ({reason}): the measured prompt peak is charged; check "
                 f"htorch.hpu.memory_stats()['Limit'] or set {SATURATED_PEAK_ENV}=0 "
                 f"limit_errors={self.saturation_counts['limit_errors']} pid={os.getpid()}")
        return None

    def after_graph(self, shape: tuple | None, step: GraphStep, rows: int) -> bool:
        """True when this was a step of a shape without a live graph: a capture on EVERY rank, decided from the
        step shape (habana's input_hash seeds per process, graphs.py 554-560, so cache growth is rank-local). Its
        peak is MaxInUse over the window start (an over-estimate when a small eager prompt ran in the same window,
        never an under-estimate); its resident cost is the InUse it added. Any serving cache growth is tracked.
        Unshaped (floor-mode prompt) graph growth only restarts the measurement window."""
        grew = self.graphs() > step.graphs
        sealed = self.boot_keys is not None
        if self.witness and sealed and grew:
            self._refuse_witness_growth(shape, step.graphs)
        self.step_captured = self.skip_capture_steps and (grew or (shape is not None and step.in_use >= 0))
        if shape is None or step.in_use < 0:
            if sealed and grew:
                self._track(shape)
            if shape is None and grew:
                self.base = self.allocator.restart()
            return False
        self.shapes.add(shape)
        self.resident_peak = max(self.resident_peak, self.allocator.in_use() - step.in_use)
        if self.lru and sealed:
            self.live[shape] = frozenset()  # every rank adds the entry (the shape decides), its keys are local
        if sealed:
            self._track(shape)
            STATS.captures += 1
            STATS.recaptures += shape in self.released_shapes
        self._note("capture", self.allocator.max_in_use() - self.base, rows)
        self.base = self.allocator.restart()
        return True

    def log_inert(self) -> None:
        """At 'Warmup finished': the floor in effect next to the free HBM serving starts with (inert=1: below it, B
        releases at its first check, and stops there when the boot set alone is above the floor)."""
        self._log_floor("inert", free=self._read_free())

    def log_tick(self) -> None:
        """The STATS tick: free HBM and the graph count, whatever the guard's state."""
        log_line(f"GLM53_GS_TICK graphs={self.graphs()} boot={len(self.boot_keys or ())} tracked={len(self.tracked)} "
                 f"untracked={len(self._runtime_untracked())} runtime_live={len(self.live)} cap={self.cap} "
                 f"free_mib={self._read_free() // MIB} stopped={STATS.stopped} guard={'lru' if self.lru else 'floor'} "
                 f"pid={os.getpid()}")

    def _note(self, kind: str, nbytes: int, rows: int) -> None:
        window = {"prompt": self.prompt_window, "retained": self.retained_window}.get(kind, self.capture_window)
        if window.add(nbytes):
            self._log_floor(kind, rows)

    def _log_floor(self, event: str, rows: int = 0, free: int | None = None, extra: str = "") -> None:
        """rows = input tokens of the forward that moved a term (names a profile-run or warmup-set peak); at the
        seal, the boot graphs kept."""
        tail = "" if free is None else f" free_mib={free // MIB} inert={int(self.floor_bytes() > free)}"
        log_line(f"GLM53_GS_FLOOR event={event} rows={rows} floor_mib={self.floor_bytes() // MIB} "
                 f"min_mib={self.min_floor_bytes // MIB} prompt_peak_mib={self.prompt_peak // MIB} "
                 f"retained_peak_mib={self.retained_peak // MIB} "
                 f"capture_peak_mib={self.capture_peak // MIB} capture_margin={CAPTURE_MARGIN} "
                 f"window={PEAK_WINDOW} source={self.source}{tail}{extra} pid={os.getpid()}")


def _allocator(namespace: Mapping[str, object]) -> Allocator:
    hpu = getattr(namespace.get("htorch"), "hpu", None)
    stats, reset = getattr(hpu, "memory_stats", None), getattr(hpu, "reset_peak_memory_stats", None)
    if stats is None or reset is None:
        raise GraphStateError("runner htorch.hpu lacks memory_stats/reset_peak_memory_stats, so the HBM floor cannot "
                              "be derived from measured MaxInUse peaks; re-check this image's habana_frameworks or "
                              f"unset {ENABLE_ENV}")
    return Allocator(stats, reset)


def hbm_guard(adapter: object, environ: Mapping[str, str]) -> HbmGuard:
    """Reads habana's graph cache dict from clear_cache's closure (graphs.py 885/908) before _counting_clear wraps
    it; a release deletes that dict's runtime keys itself and never calls clear_cache."""
    namespace = getattr(adapter.forward, "__globals__", {})
    free_bytes = getattr(namespace.get("HabanaMemoryProfiler"), "current_free_device_memory", None)
    fence = device_fence(namespace)
    if free_bytes is None or fence is None:
        raise GraphStateError("runner has no HabanaMemoryProfiler.current_free_device_memory or htorch fence, so the "
                              "HBM floor cannot be read and runtime decode graphs would never be released; re-check "
                              f"this runner or unset {ENABLE_ENV}")
    cache = _closure(getattr(adapter, "clear_cache", None)).get("cache")
    if not isinstance(cache, MutableMapping):
        raise GraphStateError(f"{type(adapter).__name__}.clear_cache closes over no graph cache dict (found "
                              f"{type(cache).__name__}): B could only release by clearing the whole cache (the "
                              "warmup decode buckets and pgk's replay-only keys) and its checks would see no graphs; "
                              f"re-check habana graphs.py wrap_in_hpu_graph (885-909) or unset {ENABLE_ENV}")
    witness = graphs_only(environ)
    if witness and guard_mode(environ) == GUARD_LRU:
        raise GraphStateError(f"{GUARD_ENV}=lru evicts graphs to stay under its cap, and {GRAPHS_ONLY_ENV}=1 never drops one; unset "
                              f"{GUARD_ENV} (the guard is then a witness) or unset {GRAPHS_ONLY_ENV}")
    guard = HbmGuard(free_bytes, cache, hbm_floor(environ), _allocator(namespace), fence,
                     guard=guard_mode(environ), rearm=exhausted_rearm(environ),
                     long_context=long_context_tokens(environ), spare=paused_spare(environ),
                     saturation=saturation_policy(environ), skip_capture_steps=tripwire_v2(environ), witness=witness)
    LIFECYCLE.guards.append(weakref.ref(guard))
    return guard


class RoutedForward:
    """The adapter's new forward: graph calls go to the runner untouched; eager calls run bound (and unlatched).
    Check points (serving only, at most one check per forward, none on a plain replay or a small eager call):
    a suppression decision, a large eager forward (released before it runs), a decode step of a new shape
    (released before the next forward, once the graph's outputs were consumed)."""

    def __init__(self, route: Route, state: GraphState, mode: Mode, hbm: HbmGuard | None = None) -> None:
        self.route, self.state, self.mode, self.hbm = route, state, mode, hbm
        self.__wrapped__ = route.adaptive
        self.mixed_decode_graph_forwards = 0

    def _witness(self) -> bool:
        return self.hbm is not None and self.hbm.witness

    def __call__(self, *args: object, **kwargs: object) -> object:
        if not self._witness():
            return self._routed(*args, **kwargs)
        with fail_fast_on_allocation_error("guarded forward"):  # a real allocation failure stops every rank, never retried or run eager
            return self._routed(*args, **kwargs)

    def _routed(self, *args: object, **kwargs: object) -> object:
        route, mode = self.route, self.mode
        kind = classify(route, args, kwargs) if mode.routes else _observe_kind(route, args, kwargs)
        STATS.tick()
        checking, released = self._start(self.hbm is not None and LIFECYCLE.checks_on())
        if kind == GRAPH and kwargs.get(EXTERNAL_KWARG) and mode.routes and route.latched():
            if self._mixed_decode_replay_allowed(args, kwargs):
                # All eager writers used GraphState.bind()'s private top-k.
                # Keep the runner's fenced handoff and the HBM admission path;
                # clear_cache is already scoped by _runner_clear in this mode.
                # Metadata, token storage and prefill dispatch stay untouched.
                kwargs = dict(kwargs)
                kwargs[EXTERNAL_KWARG] = False
                self.mixed_decode_graph_forwards += 1
                if self.mixed_decode_graph_forwards == 1:
                    log_line(f"GLM53_MIXED_DECODE_GRAPH first_rows={_rows(args, kwargs)} private_topk=1")
                return self._graph(args, kwargs, checking)
            if os.environ.get(DECODE_WIDTHS_ENV) or self._witness():
                flag = DECODE_WIDTHS_ENV if os.environ.get(DECODE_WIDTHS_ENV) else GRAPHS_ONLY_ENV
                raise GraphStateError(f"{flag} is set, and this latched decode would run eager; set "
                                      f"{MIXED_DECODE_REPLAY_ENV}=1 with the private top-k, or find what latched the route")
            return self._deferred(args, kwargs)
        if kind == GRAPH:
            return self._graph(args, kwargs, checking)
        chunk = Chunk(_rows(args, kwargs), prompt_context_tokens(kwargs))
        guarded = self._guarded(kind, args, kwargs)
        if kind == SUPPRESS:
            self._decide(checking and not released, chunk)
        elif checking and not released and guarded and not self.hbm.check(EAGER, chunk):
            self.hbm.release(chunk)  # a large eager chunk the runner does not latch: release before it runs
        with self.state.bind():
            if self.state.private is not None:
                _note_bind(chunk.rows)
            result = self._eager(kind, args, kwargs, chunk, guarded)
        if mode.census:
            CENSUS.record("eager")
        return result

    def _mixed_decode_replay_allowed(self, args: tuple, kwargs: Mapping[str, Any]) -> bool:
        """Only an ordinary one-token decode with the installed private bind."""
        if not _flag(os.environ, MIXED_DECODE_REPLAY_ENV):
            return False
        metadata = kwargs.get("attn_metadata")
        tokens = _input_ids(args, kwargs)
        return (
            self.mode.private and self.state.private is not None
            and getattr(metadata, "is_prompt", None) is False
            and not getattr(metadata, "mixed_step", False)
            and not kwargs.get(REFRESH_KWARG, False)
            and tokens is not None and tokens.dim() == 2 and tokens.shape[1] == 1
        )

    def _guarded(self, kind: str, args: tuple, kwargs: Mapping[str, Any]) -> bool:
        """The guard checks and measures this eager forward: a suppressed chunk, a large prefill, or (RC5.1) a
        graph-bypassing prompt chunk at a long context."""
        if self.hbm is None:
            return False
        route = self.route
        large = kind == SUPPRESS or route.should_bypass(_input_ids(args, kwargs), route.bypass_min_tokens)
        return self.hbm.is_guarded(large, args, kwargs)

    def _start(self, checking: bool) -> tuple[bool, bool]:
        """(checking, a pending release ran now): seal the boot set at the first serving forward, then run a
        release a capture check left pending (this forward is not checked again)."""
        if not checking:
            return False, False
        self.hbm.seal_boot()
        if not self.hbm.pending:
            return True, False
        self.hbm.release()
        return True, True

    def _deferred(self, args: tuple, kwargs: dict[str, Any]) -> object:
        """A decode step the runner defers while its dirty latch is set (d23b680f 1838-1849: eager, the original
        forward, no capture): bound like every eager forward, and its shape is never marked as captured."""
        STATS.eager_forwards += 1
        with self.state.bind():
            if self.state.private is not None:
                STATS.deferred_bound += 1
            result = self.route.adaptive(*args, **kwargs)
        if self.mode.census:
            CENSUS.record("eager")
        return result

    def _decide(self, checking: bool, chunk: Chunk) -> None:
        """Keep the graphs, or (below the floor) release the runtime ones now, before the chunk; it never latches."""
        if not checking or self.hbm.check(SUPPRESS, chunk):
            STATS.latch_suppressed += 1
        else:
            self.hbm.release(chunk)

    def _eager(self, kind: str, args: tuple, kwargs: dict[str, Any], chunk: Chunk, guarded: bool) -> object:
        route = self.route
        STATS.eager_forwards += 1

        def run() -> object:
            return _suppressed(route, args, kwargs) if kind == SUPPRESS else route.adaptive(*args, **kwargs)

        measured = census_forward(run, chunk.rows, chunk.context)  # inside hbm.prompt: its peak window starts first
        if not guarded:
            return measured()
        return self.hbm.prompt(measured, kind == SUPPRESS, chunk.rows, chunk.context)

    def _graph(self, args: tuple, kwargs: dict[str, Any], checking: bool) -> object:
        if self.hbm is None:
            return _measured_graph_call(self.route, args, kwargs, self.mode, None)
        decode = getattr(kwargs.get("attn_metadata"), "is_prompt", None) is False
        shape = step_shape(args, kwargs) if decode or self.hbm.lru else None  # lru: runtime prompt graphs too
        step = self.hbm.before_graph(shape)
        result = _measured_graph_call(self.route, args, kwargs, self.mode, self.hbm)
        if self.hbm.after_graph(shape, step, _rows(args, kwargs)) and checking and not self.hbm.check(CAPTURE):
            self.hbm.pending = True  # the captured graph's outputs are still in use: release at the next forward
        return result


def _observe_kind(route: Route, args: tuple, kwargs: Mapping[str, Any]) -> str:
    large = route.should_bypass(_input_ids(args, kwargs), route.bypass_min_tokens)
    return EAGER if (large or kwargs.get("bypass_hpu_graphs", False)) else GRAPH


def _measured_graph_call(route: Route, args: tuple, kwargs: dict[str, Any], mode: Mode, hbm: HbmGuard | None) -> Any:
    """The graph forward, under the step 20 census when its flag is on (flag off: exactly _graph_call)."""
    if not census_enabled():
        return _graph_call(route.adaptive, args, kwargs, mode)
    captures = hbm.graphs if hbm is not None else (lambda: CENSUS.pending_captures)
    run = lambda: _graph_call(route.adaptive, args, kwargs, mode)  # noqa: E731
    decode = getattr(kwargs.get("attn_metadata"), "is_prompt", None) is False
    width = _sparse_width(kwargs) if decode else None  # the decode key's table width, for the replay tracer (HPU_GLM53_REPLAY_TRACE_DIR)
    return census_graph_forward(run, _rows(args, kwargs), prompt_context_tokens(kwargs), captures, decode_width=width)()


def _graph_call(adaptive: Callable[..., Any], args: tuple, kwargs: dict[str, Any], mode: Mode) -> Any:
    before = CENSUS.pending_captures
    result = adaptive(*args, **kwargs)
    if mode.census and CENSUS.pending_captures != before:
        CENSUS.record("capture", "runner")
    return result


def _counting_clear(adapter: Any, mode: Mode, state: GraphState, hbm: HbmGuard | None) -> None:
    """Count the runner's clears. In serving with the private bind, the runner's restored-long-request clear
    (d23b680f 1865, a latch B hands over) drops nothing: that request's eager forwards ran on the private buffer like
    every eager forward, so every graph holds the state it was captured with; the runner fences around the call."""
    clear = getattr(adapter, "clear_cache", None)
    if clear is None or getattr(clear, MARK, False):
        return

    def clear_cache() -> None:
        _runner_clear(clear, mode, state, hbm)

    setattr(clear_cache, MARK, True)
    adapter.clear_cache = clear_cache


def _runner_clear(clear: Callable[[], None], mode: Mode, state: GraphState, hbm: HbmGuard | None) -> None:
    """One runner clear_cache call: a counted no-op once sealed with the private bind (the eager forward it follows
    ran on the private buffer, so no graph, boot or runtime, saw its writes), else the whole clear."""
    if hbm is not None and hbm.boot_keys is not None and state.private is not None:
        STATS.clears_scoped += 1
    else:
        clear()
    STATS.clears += 1
    if mode.census:
        CENSUS.record("clear")


def check_policy(adaptive: Callable[..., object]) -> None:
    """Refuse unless the wrapper was built so captured graphs keep their state (runner 1716-1724, 322-339)."""
    policy = getattr(adaptive, "__globals__", {}).get("_WRAPPED_GRAPH_POLICY")
    if policy != STATE_KEEPING_POLICY:
        raise GraphStateError(f"HPU graph policy (dry_run, free_inplace)={policy}: keeping decode graphs across eager "
                              "prompts is only sound when captured graphs retain their state; set HPU_GRAPH_DRY_RUN=0 "
                              f"and HPU_GRAPH_FREE_INPLACE=0 (the base env), or unset {ENABLE_ENV}")


def _route_for(adapter: object, mode: Mode, environ: Mapping[str, str]) -> tuple[Route, HbmGuard | None]:
    if not mode.routes:
        return _observe_route(adapter.forward, environ), None
    check_policy(adapter.forward)
    return read_route(adapter.forward), hbm_guard(adapter, environ)


def attach(adapter: object, mode: Mode, environ: Mapping[str, str] | None = None, *,
           holders: Sequence[tuple[str, object]] = ()) -> object:
    """Install the routed forward on one wrapped adapter (the object _maybe_wrap_in_hpu_graph returned).
    ``holders``: (kind, object) non-module census holders (census mode only)."""
    environ = os.environ if environ is None else environ
    namespace = getattr(adapter.forward, "__globals__", {})
    if not getattr(adapter, "_supports_external_graph_transition_control", False):
        if mode.routes:
            raise GraphStateError(f"no adaptive route on {type(adapter).__name__} (lazy mode off, or {BYPASS_MIN_ENV}=0)"
                                  f": change B has nothing to change; set {BYPASS_MIN_ENV}=512 or unset {ENABLE_ENV}")
        _register_census(adapter, None, mode, namespace, holders)
        return adapter
    route, hbm = _route_for(adapter, mode, environ)
    state = GraphState(adapter, mode, environ)
    _STATES[adapter] = state
    _counting_clear(adapter, mode, state, hbm)
    adapter.forward = RoutedForward(route, state, mode, hbm)
    if state.private is not None:
        log_line(f"GLM53_GS_PRIVATE bytes={state.private.untyped_storage().nbytes()} holders={len(state.found)} "
                 f"shape={tuple(state.private.shape)} dtype={state.private.dtype} first_sparse={state.first_sparse} "
                 f"hbm_floor_mib={hbm.floor_bytes() // MIB if hbm else '-'} pid={os.getpid()}")
    _register_census(adapter, state, mode, namespace, holders)
    return adapter


def _observe_route(adaptive: Callable[..., Any], environ: Mapping[str, str]) -> Route:
    """Census-only: any runner's adaptive forward, observed (thresholds from the closure when present, else env)."""
    names = _closure(adaptive)
    bypass_min = int(names.get("bypass_min_tokens", environ.get(BYPASS_MIN_ENV, "0") or 0))
    should_bypass = getattr(adaptive, "__globals__", {}).get("_should_bypass_hpu_graph_for_large_prefill", _large)
    return Route(adaptive, adaptive, bypass_min, 0, should_bypass)


def _large(input_ids: torch.Tensor | None, min_tokens: int) -> bool:
    """Same predicate as the runner's _should_bypass_hpu_graph_for_large_prefill (d23b680f 1685-1696)."""
    return min_tokens > 0 and input_ids is not None and input_ids.numel() >= min_tokens


def _register_census(adapter: object, state: GraphState | None, mode: Mode, namespace: Mapping[str, object],
                     holders: Sequence[tuple[str, object]]) -> None:
    if not mode.census:
        return
    CENSUS.register(f"root{sum(label.startswith('root') for label, _, _ in CENSUS.roots)}", adapter, state,
                    device_fence(namespace))
    for kind, holder in holders:
        CENSUS.add_holder(kind, holder)
    missing = sorted({"ctx", RUNNER_SCOPE} - {kind for kind, _ in holders})
    if missing:  # never silent: the census graders refuse lines without ctx0/runner0 roots
        log_line(f"GLM53_GS_CENSUS_HOLDERS missing={','.join(missing)} (no static_forward_context in the wrap call or "
                 f"no *{RUNNER_CLASS_MARK} self within {RUNNER_FRAME_DEPTH} caller frames) pid={os.getpid()}")
    CENSUS.record("wrap", type(getattr(adapter, "model", adapter)).__name__)


def census_holders(vllm_config: object, frame: FrameType | None) -> tuple[tuple[str, object], ...]:
    """The wrap call's non-module holders: its static_forward_context (layer name -> layer, vllm compilation config)
    and the runner whose load_model called it (6914 target / 6961 drafter), found up the caller frames."""
    found: list[tuple[str, object]] = []
    context = getattr(getattr(vllm_config, "compilation_config", None), "static_forward_context", None)
    if isinstance(context, Mapping):
        found.append(("ctx", context))
    for _ in range(RUNNER_FRAME_DEPTH):
        if frame is None:
            break
        owner = frame.f_locals.get("self")
        if owner is not None and RUNNER_CLASS_MARK in type(owner).__name__:
            found.append((RUNNER_SCOPE, owner))
            break
        frame = frame.f_back
    return tuple(found)


# --- graph capture hook (census) -------------------------------------------------------------------------------------


def _capture_origin() -> str:
    frame = sys._getframe(1)
    while frame is not None:
        filename = frame.f_code.co_filename
        if filename.endswith("glm53_mixed_graph.py"):
            return "mixed"
        if filename.endswith("hpu_model_runner.py"):
            return "runner"
        frame = frame.f_back
    return "other"


def patch_graphs(graphs: ModuleType) -> bool:
    """Count captures in HPUGraph.capture_end (no device call) and snapshot after the capturing forward returns."""
    capture_end = graphs.HPUGraph.capture_end
    forward = graphs.wrapped_hpugraph_forward
    if getattr(forward, MARK, False):
        return False

    def counting_capture_end(self: object, *args: object, **kwargs: object) -> object:
        result = capture_end(self, *args, **kwargs)
        CENSUS.pending_captures += 1
        return result

    def wrapped_hpugraph_forward(*args: object, **kwargs: object) -> object:
        before = CENSUS.pending_captures
        result = forward(*args, **kwargs)
        if CENSUS.pending_captures != before and _capture_origin() == "mixed":
            CENSUS.record("capture", "mixed")
        return result

    setattr(wrapped_hpugraph_forward, MARK, True)
    graphs.HPUGraph.capture_end = counting_capture_end
    graphs.wrapped_hpugraph_forward = wrapped_hpugraph_forward
    return True


# --- install -------------------------------------------------------------------------------------------------------


def patch_runner(runner: ModuleType, mode: Mode) -> bool:
    """Wrap runner._maybe_wrap_in_hpu_graph; False (nothing changed) when off or already patched."""
    original = runner._maybe_wrap_in_hpu_graph
    if not mode.engaged or getattr(original, MARK, False):
        return False

    def _maybe_wrap_in_hpu_graph(*args: object, **kwargs: object) -> object:
        holders = census_holders(kwargs.get("vllm_config"), sys._getframe(1)) if mode.census else ()
        return attach(original(*args, **kwargs), mode, holders=holders)

    _maybe_wrap_in_hpu_graph.__wrapped__ = original
    setattr(_maybe_wrap_in_hpu_graph, MARK, True)
    runner._maybe_wrap_in_hpu_graph = _maybe_wrap_in_hpu_graph
    if mode.census:
        from importlib import import_module

        patch_graphs(import_module(GRAPHS_MODULE))
    if mode.routes:
        _hook_warmup(runner)
    if not _EXIT_LINE:  # only an interpreter that loaded the runner prints the exit counters
        _EXIT_LINE.append(atexit.register(_exit_line))
    log_line(f"GLM53_GS_INSTALL patched {runner.__name__}._maybe_wrap_in_hpu_graph mode={mode} pid={os.getpid()}")
    return True


def _exit_line() -> None:
    log_line(STATS.line())


def _hook_warmup(runner: ModuleType) -> None:
    """HPUModelRunner.warmup_model: check points start when it returns (warmup graphs are the boot set, never
    released), and each guard logs its floor next to the free HBM (GLM53_GS_FLOOR event=inert). Refused without it:
    the boot set could not be sealed after warmup."""
    cls = getattr(runner, "HPUModelRunner", None)
    warmup = getattr(cls, "warmup_model", None)
    if warmup is None:
        raise GraphStateError(f"{runner.__name__} has no HPUModelRunner.warmup_model: change B seals the boot graph set "
                              "(warmup decode buckets, pgk's keys) when it returns; mount the d23b680f runner or unset "
                              f"{ENABLE_ENV} and {SKIP_CLEAR_ONLY_ENV}")
    if getattr(warmup, MARK, False):
        return

    def warmup_model(self: object, *args: object, **kwargs: object) -> object:
        result = warmup(self, *args, **kwargs)
        LIFECYCLE.start_serving()
        return result

    warmup_model.__wrapped__ = warmup
    setattr(warmup_model, MARK, True)
    cls.warmup_model = warmup_model
    LIFECYCLE.warmup_hooked = True


class _PatchOnImport(importlib.abc.MetaPathFinder):
    """One-shot finder: lets the normal path finder load ``target``, then calls ``on_loaded(module)``."""

    def __init__(self, target: str, on_loaded: Callable[[ModuleType], None]) -> None:
        self.target, self._on_loaded = target, on_loaded

    def find_spec(self, fullname: str, path: Sequence[str] | None = None,
                  target: ModuleType | None = None) -> importlib.machinery.ModuleSpec | None:
        if fullname != self.target:
            return None
        spec = importlib.machinery.PathFinder.find_spec(fullname, path)
        if spec is None:
            return None  # not on sys.path: the import itself fails loudly
        if not hasattr(spec.loader, "exec_module"):
            raise GraphStateError(f"{fullname}: loader {type(spec.loader).__name__} has no exec_module, so the "
                                  "graph-state hook cannot wrap the runner; the arm would boot without change B. "
                                  "Check the image's import system or unset the graph-state flags")
        load = spec.loader.exec_module

        def exec_module(module: ModuleType) -> None:
            load(module)
            if self in sys.meta_path:
                sys.meta_path.remove(self)
            self._on_loaded(module)

        spec.loader.exec_module = exec_module
        return spec


def install(environ: Mapping[str, str] | None = None) -> bool:
    """Boot entry (usercustomize.py). Off: returns False and changes nothing."""
    mode = read_mode(environ)
    if not mode.engaged:
        return False

    def on_loaded(module: ModuleType) -> None:
        patch_runner(module, mode)

    loaded = sys.modules.get(RUNNER_MODULE)
    if loaded is not None:
        on_loaded(loaded)
    elif not any(isinstance(finder, _PatchOnImport) and finder.target == RUNNER_MODULE for finder in sys.meta_path):
        sys.meta_path.insert(0, _PatchOnImport(RUNNER_MODULE, on_loaded))
    return True
