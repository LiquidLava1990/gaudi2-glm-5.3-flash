"""Device-vs-host verdict of one traced forward window (C286; work/w-forward-trace).  Stdlib only.

Input: one rank's torch-profiler trace (the HPU worker's chrome JSON, .gz or plain), that rank's window file
(glm53_forward_trace) and every rank's window file.  The window is the one MARKER user_annotation span on a host pid
(the worker's main thread); its length is the wall every share below divides by (the trace's own clock, the one the
device events are on: B2's traces put kernel and host events on one timeline).  The device process is the pid the
trace names "/device:..." (process_name metadata); this TP4 layout gives each rank's trace exactly one, pid 0
("/device:HPU:0").  Device busy = the union of that process's events inside the window: cat "kernel" (TPC / MME), cat
"hpu_op" (DMA engines, including the HCCL in-box collective transfers, so a collective counts as device time, not as
idle) and cat "gpu_memcpy".  Every device-process event is hardware-traced (args "HW event name"; B2: all 1.82 M).
The host logs Synapse API calls (synchronizeEvent, getRecipeDebugInfo, memcpyAsync, ...) as cat "hpu_op" /
"gpu_memcpy" too, on its own unnamed pid and never hardware-traced: those are host time, counted as host_api_events,
never device time (B2 trace ...8558252: 9,333 of them, 4.55 s, would read busy 0.968 instead of 0.093).  Device
idle is split, first match wins, by what the worker's main thread was doing meanwhile: waiting in a D2H copy or a
synchronize (the host blocked on the device, which idles anyway: another rank's lag or an untraced transfer -
neither side's own work), inside another lazy-runtime event (cat privateuse1_runtime: LaunchSyncTensorsGraph,
SyncTensorsGraphInternal, ExecuteCachedGraph, *_hpu_lazy = C++ lazy lowering / recipe lookup / launch), inside an
aten op outside those (cat cpu_op: dispatch), or in none (Python between ops).
What fills the device time (read by no verdict rule): kernel_top, the KERNEL_TOP device event names, and family_top,
the op FAMILIES matched on each event's "name @ module path", each with its busy union, its summed per-engine time
(the profiler logs one event per TPC / MME engine), its event count and its share of the device busy union.

INCOMPLETE, never a verdict, when the record cannot be the whole window of the probe's forward:
  the device record lost the window's start: in some B2 r6 windows every graph launch preceded the first surviving
    kernel (they kept 1.4-1.5 M device events, while another B2 window kept 1.83 M whole, so the loss is no fixed
    event count); here more than LAUNCHES_BEFORE_FIRST_DEVICE_MAX launches before the first device op, or a first
    device op later than LEAD_MAX of the window;
  the marker is not this window: not exactly one host user_annotation MARKER span, a span more than MARKER_MATCH_MAX
    off the hook's own window_ms, or no graph launch on the marker's thread;
  the device record is not the one read: the trace names device processes other than exactly DEVICE_PIDS, or a
    kernel / hardware-traced event sits on another pid (device work the busy union would drop);
  the ranks opened their windows more than RANK_SKEW_MAX of the window apart (rank 0's idle then holds another rank's
    lag, not its own host time), or by enough that the verdict class flips with the skew taken out of the window;
  the forward was not the probe's: more than one request, or real (unpadded) tokens other than the trigger's
    min_tokens (the probe's exact prompt length);
  no device events, device time outside the window (the drain did not hold), hook errors, or no trace;
and, for the final report, a busy fraction of the traced window that the unprofiled rehearsal wall does not confirm
(the two must classify alike and agree within FRACTION_AGREEMENT_MAX), no rehearsal to check it against, rehearsal
walls more than REHEARSAL_SPREAD_MAX apart, or a traced window more than OVERHEAD_LIMIT FASTER than its unprofiled
twin (the profiler only adds time: the twin is not like-for-like; OVERHEAD_LIMIT slower is a caveat).
"""
from __future__ import annotations

import bisect
import gzip
import itertools
import json
import os
import re
import statistics
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

MARKER = "glm53_fwdtrace_window"  # = glm53_forward_trace.MARKER
MARKER_CAT = "user_annotation"
DEVICE_PIDS = frozenset({0})  # this TP4 layout: each rank's trace names its one HPU "/device:HPU:0", on pid 0
DEVICE_NAME_PREFIX = "/device:"
HW_EVENT_ARG = "HW event name"  # every device-process event carries it; no host Synapse API call does (B2 traces)
DEVICE_BOUND_MIN = 0.70  # device busy >= 70 % of the window: the forward is device work (lever: the kernels themselves)
HOST_BOUND_MAX = 0.40    # device busy <= 40 %: the host starves the device (lever: compile / capture the prefill tiles)
LAUNCHES_BEFORE_FIRST_DEVICE_MAX = 2  # a drained window's first launch precedes its own first op; > 2 = a lost start
LEAD_MAX = 0.25          # the first device op starts more than 25 % of the window after the window opens
OUTSIDE_MAX = 0.02       # device time outside the window above 2 % of it: the edge drains did not hold
MARKER_MATCH_MAX = 0.05  # the trace's marker span vs the hook's own window_ms: more than 5 % apart = not this window
RANK_SKEW_MAX = 0.05     # the ranks' window opens more than 5 % of the window apart: rank 0 waited on another rank
FRACTION_AGREEMENT_MAX = 0.10  # busy / traced window vs busy / unprofiled rehearsal wall: at most 0.10 apart
OVERHEAD_LIMIT = 0.10    # traced vs unprofiled: > +10 % = inflated host time (caveat); < -10 % = no twin (INCOMPLETE)
REHEARSAL_SPREAD_MAX = 0.10  # a run's rehearsal walls: max / min - 1 above 10 % = no stable unprofiled wall
BUFFER_HINT_EVENTS = 1_400_000  # B2 r6 windows that lost their start kept 1.4-1.5 M device events (a caveat only)
TOP_GAPS = 10
HOST_PER_GAP = 3
DEVICE_CATS = frozenset({"kernel", "hpu_op", "gpu_memcpy"})
KERNEL_CAT = "kernel"
RUNTIME_CAT = "privateuse1_runtime"
OP_CAT = "cpu_op"
LAUNCH_NAMES = frozenset({"ExecuteCachedGraph", "LaunchSyncTensorsGraph"})
WAIT_PREFIXES = ("copy_hpu_lazy_D2H",)  # + any host event whose name holds "synchronize": the host blocks on the device
DEVICE_BOUND, HOST_BOUND, MIXED, INCOMPLETE = "DEVICE_BOUND", "HOST_BOUND", "MIXED", "INCOMPLETE"
THRESHOLDS = {"device_bound_min": DEVICE_BOUND_MIN, "host_bound_max": HOST_BOUND_MAX,
              "launches_before_first_device_max": LAUNCHES_BEFORE_FIRST_DEVICE_MAX, "lead_max": LEAD_MAX,
              "outside_max": OUTSIDE_MAX, "marker_match_max": MARKER_MATCH_MAX, "rank_skew_max": RANK_SKEW_MAX,
              "fraction_agreement_max": FRACTION_AGREEMENT_MAX, "overhead_limit": OVERHEAD_LIMIT,
              "rehearsal_spread_max": REHEARSAL_SPREAD_MAX, "device_pids": sorted(DEVICE_PIDS),
              "buffer_hint_events": BUFFER_HINT_EVENTS}
KERNEL_TOP = 40  # kernel_top: the device event names with the most busy time
FAMILIES = (("moe", ("moe", "expert", "gpt", "mixture")), ("kda", ("kda",)), ("mla_attention", ("mla", "attention")),
            ("gemm", ("gemm", "matmul", "fp8")), ("norm", ("norm",)), ("cast_copy", ("cast", "copy", "memcpy")))
OTHER_FAMILY = "other"  # family_top: substrings of the lowercased "name @ module path" label, first family wins
KDA_MATCH_FLAG = "HPU_GLM53_FWD_TRACE_KDA_MATCH"
_MODULE_OWNER = re.compile(r"(model/model/\d+/(?:self_attn(?:/mla_attn)?|mlp|mhc_[^/ @]+))(?:/|$)")
_ID_PREFIX = re.compile(r"^id_\d+_")
Interval = tuple[float, float]


def classify(busy_fraction: float) -> str:
    if busy_fraction >= DEVICE_BOUND_MIN:
        return DEVICE_BOUND
    if busy_fraction <= HOST_BOUND_MAX:
        return HOST_BOUND
    return MIXED


def load_events(path: Path) -> list[dict]:
    if path.name.endswith(".gz"):
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            return json.load(handle)["traceEvents"]
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)["traceEvents"]


# ---- interval arithmetic (us; every list sorted and disjoint after union) --------------------------------------------
def union(intervals: Iterable[Interval]) -> list[Interval]:
    merged: list[Interval] = []
    for start, end in sorted(intervals):
        if merged and start <= merged[-1][1]:
            if end > merged[-1][1]:
                merged[-1] = (merged[-1][0], end)
        else:
            merged.append((start, end))
    return merged


def intersect(first: Sequence[Interval], second: Sequence[Interval]) -> list[Interval]:
    result, i, j = [], 0, 0
    while i < len(first) and j < len(second):
        start, end = max(first[i][0], second[j][0]), min(first[i][1], second[j][1])
        if start < end:
            result.append((start, end))
        if first[i][1] < second[j][1]:
            i += 1
        else:
            j += 1
    return result


def total(intervals: Iterable[Interval]) -> float:
    return sum(end - start for start, end in intervals)


def complement(busy: Sequence[Interval], low: float, high: float) -> list[Interval]:
    gaps, cursor = [], low
    for start, end in busy:
        if start > cursor:
            gaps.append((cursor, start))
        cursor = max(cursor, end)
    if high > cursor:
        gaps.append((cursor, high))
    return gaps


def _clip(start: float, end: float, low: float, high: float) -> Interval | None:
    start, end = max(start, low), min(end, high)
    return (start, end) if end > start else None


# ---- the trace, split ---------------------------------------------------------------------------------------------
@dataclass
class TraceView:
    """The window and the events the verdict reads (device intervals keep their event index for labels)."""
    start: float
    end: float
    device: list[tuple[float, float, int]]  # clipped to the window, sorted
    outside_us: float
    kernels: int
    device_events: int
    host: list[tuple[float, float, int]]     # the marker thread's cpu_op + runtime events, clipped
    waits: list[Interval]
    runtime: list[Interval]
    ops: list[Interval]
    launches: list[float]
    events: list[dict]
    device_pids: list[int] = field(default_factory=list)  # the pids the trace names "/device:..."
    stray: dict[object, int] = field(default_factory=dict)  # pid -> device events on a pid that is not the device's
    host_api_events: int = 0  # Synapse API calls on a host pid (cat hpu_op / gpu_memcpy, never hardware-traced)

    @property
    def length(self) -> float:
        return self.end - self.start


def device_pids(events: Sequence[dict]) -> list[int]:
    """The pids the trace's process_name metadata names "/device:..." (B2: only pid 0, "/device:HPU:0")."""
    return sorted({event.get("pid") for event in events if event.get("ph") == "M"
                   and event.get("name") == "process_name"
                   and str((event.get("args") or {}).get("name", "")).startswith(DEVICE_NAME_PREFIX)}, key=str)


def host_markers(events: Sequence[dict]) -> list[dict]:
    """The window spans: MARKER complete events of cat user_annotation on a host pid (a device twin never counts)."""
    devices = set(device_pids(events)) | DEVICE_PIDS
    return [event for event in events if event.get("name") == MARKER and event.get("ph") == "X" and "dur" in event
            and event.get("cat") == MARKER_CAT and event.get("pid") not in devices]


def _device_part(view: TraceView, event: dict, index: int) -> None:
    start, end = float(event["ts"]), float(event["ts"]) + float(event["dur"])
    inside = _clip(start, end, view.start, view.end)
    view.outside_us += (end - start) - ((inside[1] - inside[0]) if inside else 0.0)
    if inside is None:
        return
    view.device.append((*inside, index))
    view.device_events += 1
    view.kernels += event.get("cat") == KERNEL_CAT


def _off_device_part(view: TraceView, event: dict) -> None:
    """A device category on another pid: a kernel or a hardware-traced event is stray device work (INCOMPLETE), the
    rest a host Synapse API call (host time, counted)."""
    if event.get("cat") == KERNEL_CAT or HW_EVENT_ARG in (event.get("args") or {}):
        view.stray[event.get("pid")] = view.stray.get(event.get("pid"), 0) + 1
    else:
        view.host_api_events += 1


def _host_part(view: TraceView, event: dict, index: int) -> None:
    inside = _clip(float(event["ts"]), float(event["ts"]) + float(event["dur"]), view.start, view.end)
    if inside is None:
        return
    view.host.append((*inside, index))
    name = str(event.get("name", ""))
    if name.startswith(WAIT_PREFIXES) or "synchronize" in name.lower():
        view.waits.append(inside)
    (view.runtime if event.get("cat") == RUNTIME_CAT else view.ops).append(inside)
    if name in LAUNCH_NAMES:
        view.launches.append(inside[0])


def split_events(events: list[dict], marker: dict) -> TraceView:
    start = float(marker["ts"])
    view = TraceView(start, start + float(marker["dur"]), [], 0.0, 0, 0, [], [], [], [], [], events,
                     device_pids(events))
    devices, thread = set(view.device_pids), (marker.get("pid"), marker.get("tid"))
    for index, event in enumerate(events):
        if event.get("ph") != "X" or "dur" not in event or "ts" not in event:
            continue
        category = event.get("cat")
        if category in DEVICE_CATS and event.get("pid") in devices:
            _device_part(view, event, index)
        elif category in DEVICE_CATS:
            _off_device_part(view, event)
        elif category in (RUNTIME_CAT, OP_CAT) and (event.get("pid"), event.get("tid")) == thread:
            _host_part(view, event, index)
    view.device.sort()
    view.host.sort()
    view.launches.sort()
    return view


# ---- busy, idle, gaps -----------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class Busy:
    start: float
    end: float
    first: int  # the event that opens the busy run
    last: int   # the event that ends last in it


def busy_runs(device: Sequence[tuple[float, float, int]]) -> list[Busy]:
    runs: list[Busy] = []
    for start, end, index in device:
        if runs and start <= runs[-1].end:
            if end > runs[-1].end:
                runs[-1] = Busy(runs[-1].start, end, runs[-1].first, index)
        else:
            runs.append(Busy(start, end, index, index))
    return runs


def device_label(event: dict | None) -> str | None:
    if event is None:
        return None
    path = _ID_PREFIX.sub("", str(event.get("args", {}).get("EventName", "")).split(" Unique ID")[0])
    name = str(event.get("name", "?"))
    return f"{name} @ {path}" if path else name


def host_around(view: TraceView, low: float, high: float) -> list[dict]:
    """The main-thread host events the gap overlaps most (ties: the innermost, the shortest)."""
    found = []
    for start, end, index in view.host[:bisect.bisect_left(view.host, (high,))]:
        overlap = min(end, high) - max(start, low)
        if overlap > 0:
            found.append((-overlap, end - start, index))
    return [{"name": view.events[index].get("name"), "cat": view.events[index].get("cat"),
             "overlap_ms": round(-negative / 1e3, 3)} for negative, _, index in sorted(found)[:HOST_PER_GAP]]


GapEdge = tuple[float, int | None, float, int | None]  # (idle start, event before, idle end, event after); None at a window edge


def gap_edges(view: TraceView, runs: Sequence[Busy]) -> list[GapEdge]:
    """Every device-idle stretch of the window, lead and tail included, in time order, with the events on each side."""
    edges = [(view.start, None, runs[0].start if runs else view.end, runs[0].first if runs else None)]
    edges += [(left.end, left.last, right.start, right.first) for left, right in itertools.pairwise(runs)]
    if runs:
        edges.append((runs[-1].end, runs[-1].last, view.end, None))
    return [edge for edge in edges if edge[2] > edge[0]]


def largest_gaps(view: TraceView, runs: Sequence[Busy]) -> list[dict]:
    """The TOP_GAPS longest device-idle stretches of the window (lead and tail included), longest first."""
    gaps = sorted(gap_edges(view, runs), key=lambda edge: edge[0] - edge[2])[:TOP_GAPS]
    return [{"start_ms": round((low - view.start) / 1e3, 3), "length_ms": round((high - low) / 1e3, 3),
             "before": device_label(None if before is None else view.events[before]),
             "after": device_label(None if after is None else view.events[after]),
             "host": host_around(view, low, high)} for low, before, high, after in gaps]


IDLE_BUCKETS = ("in_d2h_wait_ms", "in_lazy_runtime_ms", "in_aten_ops_ms")  # then uncovered_ms: the rest


def idle_split(view: TraceView, busy: Sequence[Interval]) -> dict[str, float]:
    """Device idle by the main thread's state, first match wins: D2H / synchronize wait, other lazy runtime, aten op."""
    idle = complement(busy, view.start, view.end)
    split = {"total_ms": round(total(idle) / 1e3, 3)}
    for bucket, events in zip(IDLE_BUCKETS, (view.waits, view.runtime, view.ops), strict=True):
        cover = union(events)
        split[bucket] = round(total(intersect(idle, cover)) / 1e3, 3)
        idle = intersect(idle, complement(cover, view.start, view.end))
    split["uncovered_ms"] = round(total(idle) / 1e3, 3)
    return split


# ---- what fills the device time --------------------------------------------------------------------------------
def family_of(label: str) -> str:
    """The op family of a device event's label (device_label: "name @ module path"): the first FAMILIES entry with
    a substring in it, else OTHER_FAMILY."""
    lowered = label.lower()
    return next((family for family, needles in FAMILIES if any(needle in lowered for needle in needles)),
                OTHER_FAMILY)


def kda_family_of(label: str) -> str:
    """GLM-specific self_attn is KDA except its explicit MLA subtree; no hash dictionary."""
    lowered = label.lower()
    if any(word in lowered for word in ("mla", "attention", "moe", "expert", "mixture")):
        return family_of(label)
    if "self_attn/" in lowered or any(word in lowered for word in ("kda", "causal_conv1d", "gated_delta")):
        return "kda"
    return family_of(label)


def recipe_key(event: dict) -> tuple[str, str] | None:
    args = event.get("args", {})
    recipe = args.get("recipeId")
    if recipe is None or str(recipe) == "0":
        return None
    return str(args.get("recipeHandle", "")), str(recipe)


def module_owner(event: dict) -> str | None:
    label = device_label(event) or ""
    match = _MODULE_OWNER.search(label)
    if match:
        return match[1]
    return "unattributable" if "model/" in label else None


def owner_family(owner: str, labels: Sequence[str]) -> str:
    if "/self_attn" in owner:
        return "mla_attention" if "/mla_attn" in owner else "kda"
    if "/mlp" in owner and any("expert" in label.lower() for label in labels):
        return "moe"
    return "other"


def recipe_owners(events: Iterable[dict]) -> dict[tuple[str, str], tuple[str, str]]:
    """Infer only recipes whose ALL module anchors agree; mixed recipes remain opaque."""
    anchors: dict[tuple[str, str], dict[str, set[str]]] = {}
    for event in events:
        if event.get("pid") != 0 or event.get("cat") not in DEVICE_CATS:
            continue
        key, owner = recipe_key(event), module_owner(event)
        if key is not None and owner is not None:
            anchors.setdefault(key, {}).setdefault(owner, set()).add(device_label(event) or "")
    result = {}
    for key, owners in anchors.items():
        if len(owners) == 1 and "unattributable" not in owners:
            owner, labels = next(iter(owners.items()))
            result[key] = owner_family(owner, tuple(labels)), owner
    return result


def event_owner(event: dict, owners: dict[tuple[str, str], tuple[str, str]]) -> tuple[str, str, str]:
    """Module paths are direct evidence; single-owner recipes are inference, never call stacks."""
    owner = module_owner(event)
    if owner is not None:
        return kda_family_of(device_label(event) or ""), owner, "module_path"
    inferred = owners.get(recipe_key(event))
    if inferred is not None:
        return *inferred, "recipe_inference"
    return kda_family_of(device_label(event) or ""), "unattributable", "unresolved"


class _Tally:
    """One key's device events, fed in start order: the union of their spans (busy), their summed span (engine
    time: one event per TPC / MME engine, so parallel engines add up) and their count."""
    __slots__ = ("start", "end", "closed_us", "engine_us", "events")

    def __init__(self, start: float, end: float) -> None:
        self.start, self.end, self.closed_us, self.engine_us, self.events = start, end, 0.0, end - start, 1

    def add(self, start: float, end: float) -> None:
        self.engine_us += end - start
        self.events += 1
        if start <= self.end:
            self.end = max(self.end, end)
            return
        self.closed_us += self.end - self.start
        self.start, self.end = start, end

    @property
    def busy_us(self) -> float:
        return self.closed_us + self.end - self.start


def _tallies(view: TraceView, key: Callable[[dict], str]) -> dict[str, _Tally]:
    tallies: dict[str, _Tally] = {}
    for start, end, index in view.device:  # sorted by start: each key's union closes run by run
        name = key(view.events[index])
        if name in tallies:
            tallies[name].add(start, end)
        else:
            tallies[name] = _Tally(start, end)
    return tallies


def _entry(name: str, busy_us: float, engine_us: float, events: int, device_busy_us: float) -> dict:
    return {"name": name, "busy_ms": round(busy_us / 1e3, 3), "engine_ms": round(engine_us / 1e3, 3),
            "events": events, "share": round(busy_us / device_busy_us, 4) if device_busy_us > 0 else 0.0}


def _ranked(rows: dict[str, tuple[float, float, int]], device_busy_us: float, limit: int | None) -> list[dict]:
    """rows: key -> (busy us, engine us, events); busiest first (ties by name), at most ``limit``."""
    ordered = sorted(rows.items(), key=lambda item: (-item[1][0], item[0]))[:limit]
    return [_entry(name, busy_us, engine_us, events, device_busy_us)
            for name, (busy_us, engine_us, events) in ordered]


def histograms(view: TraceView, device_busy_us: float) -> dict[str, list[dict]]:
    """kernel_top: the KERNEL_TOP device event names (every device category) by busy time; family_top: every op family
    (FAMILIES).  share = the key's busy union / the device busy union; the keys overlap in time (MME and TPC run
    together), so the shares can sum above 1."""
    names = _tallies(view, lambda event: str(event.get("name", "?")))
    if os.environ.get(KDA_MATCH_FLAG) == "1":
        owners = recipe_owners(view.events[index] for _, _, index in view.device)
        families = _tallies(view, lambda event: event_owner(event, owners)[0])
    else:
        families = _tallies(view, lambda event: family_of(device_label(event) or "?"))
    return {key: _ranked({name: (tally.busy_us, tally.engine_us, tally.events) for name, tally in tallies.items()},
                         device_busy_us, limit)
            for key, tallies, limit in (("kernel_top", names, KERNEL_TOP), ("family_top", families, None))}


def summed_histogram(reports: Sequence[dict], key: str, device_busy_ms: float, limit: int | None) -> list[dict]:
    """The slices' histograms added key by key (their windows are disjoint, so busy adds).  A name below a slice's
    own top KERNEL_TOP adds nothing from that slice: a combined kernel_top entry is a lower bound; families are
    exact."""
    rows: dict[str, tuple[float, float, int]] = {}
    for report in reports:
        for item in report[key]:
            busy, engine, events = rows.get(item["name"], (0.0, 0.0, 0))
            rows[item["name"]] = (busy + item["busy_ms"] * 1e3, engine + item["engine_ms"] * 1e3,
                                  events + item["events"])
    return _ranked(rows, device_busy_ms * 1e3, limit)


# ---- completeness -----------------------------------------------------------------------------------------------
def window_reasons(window: dict) -> list[str]:
    reasons = [f"hook: {error}" for error in window.get("errors", [])]
    if not window.get("opened"):
        reasons.append("the window never opened")
    if not window.get("trace"):
        reasons.append("no trace exported (export_ranks must hold this rank)")
    if len(window.get("request_ids", [])) > 1:
        reasons.append(f"the traced forward held {len(window['request_ids'])} requests, not the probe's one")
    return reasons + token_reasons(window)


def token_reasons(window: dict) -> list[str]:
    """The traced forward prefilled exactly the probe's prompt: real tokens (unpadded) = the trigger's min_tokens."""
    expected, real = (window.get("trigger") or {}).get("min_tokens"), window.get("tokens")
    if real is None:
        return [f"the hook recorded no real token count (attn_metadata had no readable seq_lens_tensor; "
                f"{window.get('padded_tokens')} padded token ids): the forward cannot be shown to be the probe's"]
    if real != expected:
        return [f"the traced forward prefilled {real} real tokens (seq lens {window.get('seq_lens')}), not the "
                f"probe's {expected}"]
    return []


def marker_reasons(markers: Sequence[dict], window: dict) -> list[str]:
    if len(markers) != 1:
        return [f"found {len(markers)} {MARKER} {MARKER_CAT} spans on a host pid, not 1: the window was not recorded "
                "on the worker's main thread (or twice)"]
    hook_ms, marker_ms = window.get("window_ms"), float(markers[0]["dur"]) / 1e3
    if not hook_ms:
        return ["the hook recorded no window_ms: the trace's span cannot be matched to the window"]
    if abs(marker_ms - hook_ms) > MARKER_MATCH_MAX * hook_ms:
        return [f"the trace's {MARKER} span is {marker_ms:.1f} ms but the hook timed its window at {hook_ms:.1f} ms "
                f"(more than {MARKER_MATCH_MAX:.0%} apart): the span is not this window"]
    return []


def truncation_reasons(view: TraceView) -> list[str]:
    if not view.device:
        return ["no device events inside the window"]
    reasons, first = [], view.device[0][0]
    before = bisect.bisect_left(view.launches, first)
    if before > LAUNCHES_BEFORE_FIRST_DEVICE_MAX:
        reasons.append(f"device buffer lost the window's start: {before} of {len(view.launches)} graph launches "
                       "precede the first device op (B2 r6); trace a shorter slice")
    lead = (first - view.start) / view.length
    if lead > LEAD_MAX:
        reasons.append(f"the first device op starts {(first - view.start) / 1e3:.1f} ms ({lead:.0%}) into the "
                       "window: its start was lost (device buffer) or the recipes compiled in it; trace a shorter "
                       "slice or add rehearsals")
    if view.outside_us > OUTSIDE_MAX * view.length:
        reasons.append(f"{view.outside_us / 1e3:.1f} ms of device time lies outside the window: the edge drains did "
                       "not hold or the clocks differ")
    return reasons


def device_process_reasons(view: TraceView) -> list[str]:
    """The device record is the one this analysis reads: exactly DEVICE_PIDS named "/device:", and no device work on
    another pid."""
    reasons = []
    if view.device_pids != sorted(DEVICE_PIDS):
        reasons.append(f"the trace names device processes {view.device_pids}, not exactly {sorted(DEVICE_PIDS)} (this "
                       "TP4 layout: one HPU per rank's trace, \"/device:HPU:0\" on pid 0): the device record is not "
                       "the one this analysis reads")
    if view.stray:
        reasons.append(f"{sum(view.stray.values())} device events on non-device pid(s) {sorted(view.stray, key=str)}: "
                       f"kernels or hardware-traced events ({HW_EVENT_ARG!r}) outside the device process, which the "
                       "busy union would drop")
    return reasons


def deskew_reasons(fraction: float, deskewed: float | None, skew_ms: float | None) -> list[str]:
    """Up to RANK_SKEW_MAX of skew passes, but those ms may be rank 0 waiting on a late rank: the class must hold."""
    if deskewed is None or classify(deskewed) == classify(fraction):
        return []
    return [f"busy fraction {fraction:.3f} reads {classify(fraction)} but {deskewed:.3f} reads {classify(deskewed)} "
            f"with the ranks' skew taken out of the window ({skew_ms:.1f} ms): rank 0's idle may be another rank's "
            "lag; rerun the probe"]


def launch_reasons(view: TraceView) -> list[str]:
    if view.launches:
        return []
    return ["no graph launch on the marker's thread inside the window: the marker is not on the worker's main "
            "thread, or its runtime events were not recorded"]


def rank_skew_ms(peers: Sequence[dict]) -> float | None:
    opened = [peer.get("opened_unix") for peer in peers]
    if not opened or None in opened:
        return None
    return round((max(opened) - min(opened)) * 1e3, 3)


def rank_reasons(peers: Sequence[dict], window_ms: float | None) -> list[str]:
    """Every rank opened its window within RANK_SKEW_MAX of the window of the others (one host clock: time.time)."""
    missing = [peer.get("rank") for peer in peers if peer.get("opened_unix") is None]
    if missing:
        return [f"ranks {missing} recorded no opened_unix: their window start is unknown, so rank 0's idle may hold "
                "their lag"]
    skew = rank_skew_ms(peers)
    if window_ms and skew is not None and skew > RANK_SKEW_MAX * window_ms:
        return [f"the ranks opened their windows {skew:.1f} ms apart, more than {RANK_SKEW_MAX:.0%} of the "
                f"{window_ms:.1f} ms window: rank 0's device idle then holds another rank's lag; rerun the probe"]
    return []


# ---- one run ----------------------------------------------------------------------------------------------------
def _run_fields(window: dict, peers: Sequence[dict]) -> dict:
    return {"run_id": window.get("run_id"), "rank": window.get("rank"), "trigger": window.get("trigger"),
            "tokens": window.get("tokens"), "padded_tokens": window.get("padded_tokens"),
            "seq_lens": window.get("seq_lens"), "shape": window.get("shape"), "boundaries": window.get("boundaries"),
            "hook_window_ms": window.get("window_ms"), "rehearsals": window.get("rehearsals", []),
            "rank_open_skew_ms": rank_skew_ms(peers),
            "profiler_start_s": [peer.get("profiler_start_s") for peer in peers],
            "profiler_stop_s": [peer.get("profiler_stop_s") for peer in peers],
            "reasons": window_reasons(window) + rank_reasons(peers, window.get("window_ms")), "caveats": []}


def analyze(events: list[dict], window: dict, peers: Sequence[dict] | None = None) -> dict:
    """The per-run report: the window file's fields, the device metrics and the verdict.  ``peers`` = every rank's
    window file of this run (rank 0's included); None = this rank's alone."""
    report = _run_fields(window, [window] if peers is None else peers)
    markers = host_markers(events)
    report["reasons"] += marker_reasons(markers, window)
    if len(markers) != 1 or float(markers[0]["dur"]) <= 0:
        return report | {"verdict": INCOMPLETE}
    view = split_events(events, markers[0])
    report["reasons"] += device_process_reasons(view) + truncation_reasons(view) + launch_reasons(view)
    runs = busy_runs(view.device)
    busy = [(run.start, run.end) for run in runs]
    kernel_spans = [(start, end) for start, end, index in view.device if events[index].get("cat") == KERNEL_CAT]
    kernel_busy = union(kernel_spans)
    fraction = total(busy) / view.length
    skew_ms = report["rank_open_skew_ms"]
    deskewed = None if skew_ms is None or skew_ms * 1e3 >= view.length else total(busy) / (view.length - skew_ms * 1e3)
    report["reasons"] += deskew_reasons(fraction, deskewed, skew_ms)
    report |= {"window_ms": round(view.length / 1e3, 3), "device_busy_ms": round(total(busy) / 1e3, 3),
               "kernel_busy_ms": round(total(kernel_busy) / 1e3, 3), "busy_fraction": round(fraction, 4),
               "busy_fraction_deskewed": None if deskewed is None else round(deskewed, 4),
               "kernels": view.kernels, "device_events": view.device_events, "graph_launches": len(view.launches),
               "device_pids": view.device_pids, "host_api_events": view.host_api_events,
               "idle": idle_split(view, busy), "largest_idle_gaps": largest_gaps(view, runs)}
    report |= histograms(view, total(busy))
    if view.device_events >= BUFFER_HINT_EVENTS:
        report["caveats"].append(f"{view.device_events} device events: as many as the B2 r6 windows that lost their "
                                 "start kept (1.4-1.5 M; a complete B2 window kept 1.83 M, so the loss is no fixed "
                                 "count): the start checks above decide")
    report["verdict"] = INCOMPLETE if report["reasons"] else classify(fraction)
    return report


# ---- slices and the final report ------------------------------------------------------------------------------------
def tiling_reasons(reports: Sequence[dict]) -> list[str]:
    """The slices must cut one forward into consecutive windows: [0, c1), [c1, c2), ..., [ck, exit)."""
    cuts = [((report.get("trigger") or {}).get("start"), (report.get("trigger") or {}).get("stop"))
            for report in reports]
    if not cuts or cuts[0][0] != 0 or cuts[-1][1] is not None:
        return [f"the slices {cuts} do not start at the forward's entry (0) and end at its exit (null)"]
    if any(left[1] != right[0] for left, right in itertools.pairwise(cuts)):
        return [f"the slices {cuts} are not consecutive"]
    counts = {report.get("boundaries") for report in reports}
    if len(counts) != 1:
        return [f"the sliced forwards made different boundary counts {sorted(counts, key=str)}"]
    return []


def _summed(reports: Sequence[dict], key: str) -> float:
    return sum(report[key] for report in reports)


def combine(reports: Sequence[dict]) -> dict:
    """The forward as the sum of its slices (each slice is its own forward of the same shape)."""
    ordered = sorted(reports, key=lambda report: (report.get("trigger") or {}).get("start", 0))
    caveats = [f"slice {report['run_id']}: {caveat}" for report in ordered for caveat in report.get("caveats", [])]
    reasons = [f"slice {report['run_id']}: {reason}" for report in ordered for reason in report["reasons"]]
    reasons += tiling_reasons(ordered)
    if reasons:
        return {"verdict": INCOMPLETE, "reasons": reasons, "caveats": caveats}
    window = _summed(ordered, "window_ms")
    gaps, offset = [], 0.0
    for report in ordered:
        gaps += [gap | {"start_ms": round(gap["start_ms"] + offset, 3), "run_id": report["run_id"]}
                 for gap in report["largest_idle_gaps"]]
        offset += report["window_ms"]
    idle = {key: round(sum(report["idle"][key] for report in ordered), 3) for key in ordered[0]["idle"]}
    fraction = _summed(ordered, "device_busy_ms") / window
    tops = {key: summed_histogram(ordered, key, _summed(ordered, "device_busy_ms"), limit)
            for key, limit in (("kernel_top", KERNEL_TOP), ("family_top", None))}
    return tops | {"verdict": classify(fraction), "reasons": [], "caveats": caveats, "window_ms": round(window, 3),
            "device_busy_ms": round(_summed(ordered, "device_busy_ms"), 3),
            "kernel_busy_ms": round(_summed(ordered, "kernel_busy_ms"), 3), "busy_fraction": round(fraction, 4),
            "kernels": int(_summed(ordered, "kernels")), "device_events": int(_summed(ordered, "device_events")),
            "graph_launches": int(_summed(ordered, "graph_launches")), "idle": idle,
            "largest_idle_gaps": sorted(gaps, key=lambda gap: -gap["length_ms"])[:TOP_GAPS]}


def overhead(traced_ms: float | None, unprofiled_ms: float | None) -> float | None:
    """Traced vs unprofiled wall - 1; None without both."""
    if traced_ms is None or not unprofiled_ms:
        return None
    return round(traced_ms / unprofiled_ms - 1.0, 4)


def rehearsal_walls(run: dict) -> list[float]:
    return [rehearsal["window_ms"] for rehearsal in run.get("rehearsals", []) if rehearsal.get("window_ms")]


def unprofiled_wall_ms(runs: Sequence[dict]) -> float | None:
    """The summed median rehearsal window of each run: the same cuts, drains and shape without the profiler (None
    when a run has no rehearsal)."""
    medians = [statistics.median(rehearsal_walls(run)) if rehearsal_walls(run) else None for run in runs]
    return None if not medians or None in medians else round(sum(medians), 3)


def rehearsal_reasons(runs: Sequence[dict]) -> list[str]:
    """Two rehearsals' median is their mean: it is a wall only when they agree within REHEARSAL_SPREAD_MAX."""
    reasons = []
    for run in runs:
        walls = rehearsal_walls(run)
        spread = max(walls) / min(walls) - 1.0 if len(walls) >= 2 else 0.0
        if spread > REHEARSAL_SPREAD_MAX:
            reasons.append(f"run {run.get('run_id')}: rehearsal walls {walls} ms spread {spread:.1%} (more than "
                           f"{REHEARSAL_SPREAD_MAX:.0%}): no stable unprofiled wall; rerun the probe")
    return reasons


def overhead_reasons(profiler_overhead: float | None) -> list[str]:
    if profiler_overhead is None or profiler_overhead >= -OVERHEAD_LIMIT:
        return []
    return [f"the traced window is {-profiler_overhead:.1%} faster than its unprofiled twin (more than "
            f"{OVERHEAD_LIMIT:.0%}): the profiler only adds time, so the twin is not like-for-like (a cold or "
            "contended rehearsal); rerun the probe"]


def agreement_reasons(chosen: dict, unprofiled_ms: float | None) -> list[str]:
    """The busy fraction must not depend on whose wall it divides by: the traced window or the unprofiled twin."""
    if not unprofiled_ms:
        return ["no unprofiled rehearsal wall to check the busy fraction against (the profiler's own host time could "
                "carry the reading); run with rehearsals >= 1"]
    traced = chosen["device_busy_ms"] / chosen["window_ms"]
    unprofiled = chosen["device_busy_ms"] / unprofiled_ms
    if classify(traced) == classify(unprofiled) and abs(traced - unprofiled) <= FRACTION_AGREEMENT_MAX:
        return []
    return [f"busy fraction {traced:.3f} of the traced window vs {unprofiled:.3f} of the unprofiled rehearsal wall "
            f"({unprofiled_ms:.1f} ms): {classify(traced)} vs {classify(unprofiled)}, {abs(traced - unprofiled):.3f} "
            f"apart; they must classify alike and agree within {FRACTION_AGREEMENT_MAX}: the profiler moved the "
            "reading"]


FIELDS = ("window_ms", "device_busy_ms", "kernel_busy_ms", "busy_fraction", "kernels", "device_events",
          "graph_launches", "idle", "largest_idle_gaps")
HISTOGRAM_FIELDS = ("kernel_top", "family_top")  # what fills the device time; no verdict rule reads them


def candidate_report(chosen: dict | None, runs: Sequence[dict]) -> dict:
    """One candidate for the final report (the whole run, or the combined slices with their runs): its metrics, the
    profiler checks, and INCOMPLETE with the reasons when it is not complete or the unprofiled wall disagrees."""
    if chosen is None:
        return {"verdict": INCOMPLETE, "reasons": ["not run"], "caveats": []}
    if chosen["verdict"] == INCOMPLETE:
        return {"verdict": INCOMPLETE, "reasons": list(chosen["reasons"]), "caveats": list(chosen.get("caveats", []))}
    unprofiled = unprofiled_wall_ms(runs)
    walls = [run.get("hook_window_ms") for run in runs]
    result = {key: chosen[key] for key in FIELDS + HISTOGRAM_FIELDS} | {
        "forward_wall_ms": chosen["window_ms"], "unprofiled_wall_ms": unprofiled,
        "busy_fraction_unprofiled": round(chosen["device_busy_ms"] / unprofiled, 4) if unprofiled else None,
        "profiler_overhead": overhead(None if None in walls else sum(walls), unprofiled),
        "caveats": list(chosen.get("caveats", []))}
    result["reasons"] = (rehearsal_reasons(runs) + agreement_reasons(chosen, unprofiled)
                         + overhead_reasons(result["profiler_overhead"]))
    result["verdict"] = INCOMPLETE if result["reasons"] else chosen["verdict"]
    if result["profiler_overhead"] is not None and result["profiler_overhead"] > OVERHEAD_LIMIT:
        result["caveats"].append(f"profiler overhead {result['profiler_overhead']:+.1%} vs the unprofiled "
                                 "rehearsals: host time is inflated, a HOST_BOUND or MIXED reading leans host-ward")
    return result


def slice_caveats(whole_reasons: Sequence[str], slices: Sequence[dict]) -> list[str]:
    """What a sliced verdict is not: the whole forward (its INCOMPLETE reasons), and drain-free across the cuts."""
    return [f"the whole run is INCOMPLETE ({'; '.join(whole_reasons)}): this verdict sums {len(slices)} slices",
            f"the slices add {len(slices) - 1} mid-forward drains (mark_step + synchronize at each inner cut) the "
            "whole forward does not make: each slice starts on a drained, idle device, so overlap across a cut is "
            "lost and the summed idle includes every restart"]


def final_report(whole: dict | None, slices: Sequence[dict]) -> dict:
    """The whole-forward run when it is complete and confirmed, else the combined slices when they are, else
    INCOMPLETE with both sides' reasons."""
    result: dict = {"verdict": INCOMPLETE, "source": None, "thresholds": THRESHOLDS, "reasons": [], "caveats": [],
                    "tokens": (whole or {}).get("tokens"), "shape": (whole or {}).get("shape"),
                    "boundaries": (whole or {}).get("boundaries"), "runs": [whole, *slices] if whole else list(slices)}
    first = candidate_report(whole, [whole] if whole else [])
    if first["verdict"] != INCOMPLETE:
        return result | first | {"source": "whole"}
    second = candidate_report(combine(slices), slices) if slices else None
    if second is not None and second["verdict"] != INCOMPLETE:
        caveats = slice_caveats(first["reasons"], slices) + second["caveats"]
        return result | second | {"source": "slices", "caveats": caveats}
    result["reasons"] = [f"whole: {reason}" for reason in first["reasons"]]
    if second is None:
        result["reasons"].append("slices: not run")
    else:  # a slice's own reasons already name it ("slice <run_id>: ...")
        result["reasons"] += [reason if reason.startswith("slice ") else f"slices: {reason}"
                              for reason in second["reasons"]]
    return result
