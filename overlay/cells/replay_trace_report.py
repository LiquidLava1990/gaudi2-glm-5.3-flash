#!/usr/bin/env python3
"""Stage C1p host side (DUAL-SUSTAINED-PLAN.md 12.3a): what filled a traced graph replay. Stdlib only; never touches a card.

    replay_trace_report.py wait   --dir DIR --runs RUN[,RUN...] [--ranks 4] [--timeout-s 600] [--poll-s 2]
    replay_trace_report.py report --dir DIR --runs RUN[,RUN...] [--ranks 4] [--out report.json] [--container-log FILE]

RUN is the hook's run id (glm53_graph_trace.Target.run_id), e.g. prompt-1024x8192-n3 or decode-16x8192-n5. ``wait`` blocks until every
rank's window file is there (the hook writes it last, after its trace) so the cell does not stop the container mid-export. ``report``
reads each run's rank-0 trace with every rank's window file through trace_verdict (the C820 analyzer, unchanged) and prints, per run:
REPLAY_TRACE_RUN (status, verdict, traced vs unprofiled wall, device busy, launches, kernels), REPLAY_TRACE_FAMILY (busy ms per op
family: moe / kda / mla_attention / gemm / norm / cast_copy / other) and REPLAY_TRACE_KERNEL (the busiest op names), then REPLAY_TRACE_REASON
lines and one REPLAY_TRACE_SUMMARY.

After a run's REPLAY_TRACE_REASON lines come the tables of WIDE-PROMPT-KERNEL-PLAN step 1, each under its own prefix: REPLAY_TRACE_SCAN (the
recipes that run the g6_topk_emit_i32 kernel (g6_topk_emit_wordskip_i32 with HPU_GLM53_EMIT_WORDSKIP=1), the indexer's top-k scan: recipes, launches, the union of their device time and the union of all the
rest), REPLAY_TRACE_RECIPE (per recipe: launches, busy union, summed and mean call wall), REPLAY_TRACE_STAGE (one scan call cut at its anchor
kernels: routing, GEMM + score, keys, state, emit, merge) and REPLAY_TRACE_GAP (the device idle: gaps under 30 us as one class, the longer ones by size
band and the kernels before and after). A recipe's call is its events until it is silent for CALL_GAP_US.

status COMPLETE: the whole window survived and the unprofiled twin agrees. PARTIAL: the device record lost the window's start (the profiler
keeps only the last 0.5-0.7 s at 1.4-1.5 M kernels): the tables then cover only the surviving tail, ``lead_ms`` of the window are gone and
``covered_ms`` remain; they are real but they are not the replay. MISSING: no trace (refused for free HBM, profiler error, rank 0 absent).
Exit 0 when every run is COMPLETE, 4 when any is not; ``wait`` exits 3 when files are still missing at the timeout.
"""
from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
import time
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

import trace_verdict

OVERLAY = Path(__file__).resolve().parents[1]
if str(OVERLAY) not in sys.path:
    sys.path.append(str(OVERLAY))  # after site-packages: only the hook's file name and schema come from here
import glm53_graph_trace as hook  # noqa: E402

EXIT_OK, EXIT_MISSING, EXIT_INCOMPLETE = 0, 3, 4
DEFAULT_RANKS = 4  # TP4 on one quad
KERNEL_LINES = 20
RUN_ID = re.compile(r"(?P<kind>prompt|decode)-(?P<rows>\d+)x(?P<width>\d+)-n(?P<nth>\d+)")
CENSUS_TAG = "GLM53_PROMPT_CENSUS "
COMPLETE, PARTIAL, MISSING = "COMPLETE", "PARTIAL", "MISSING"
SCAN_ANCHORS = ("g6_topk_emit_i32", "g6_topk_emit_wordskip_i32")  # a recipe that runs either kernel is a scan recipe: the indexer's top-k scan (WIDE-PROMPT-KERNEL-PLAN section 1); the second runs with HPU_GLM53_EMIT_WORDSKIP=1 (step 6)
CALL_GAP_US = 100.0  # a recipe's events are one call until it is silent this long (in-call gaps are under 12 us; the next call is a layer away)
GAP_BANDS = ((10_000.0, "10ms+"), (1_000.0, "1ms-10ms"), (30.0, "30us-1ms"))  # (floor us, label), largest first
GAP_SPLIT_US = GAP_BANDS[-1][0]  # idle shorter than this is one class ("between nodes"); longer idle is classed by size and the kernels around it
UNDER_BAND = "<30us"
WINDOW_START, WINDOW_END = "<window start>", "<window end>"
STAGE_ANCHORS = (("gemm_score", "GEMM"), ("topkkeys", "mla_topkkeys_i32"), ("topkstate", "mla_topkstate_i32"),
                 *(("emit", guid) for guid in SCAN_ANCHORS))  # one call holds one emit kernel, so one emit stage
RECIPE_LINES, GAP_CLASS_LINES, GAP_CLASS_KEEP = 50, 20, 60  # printed recipe rows; printed and stored long-gap classes
Span = tuple[float, float, int]  # trace_verdict.TraceView.device entries: (start us, end us, event index)


def read_json(path: Path) -> dict | None:
    """The file's JSON, None when it is absent or unreadable (the caller names it)."""
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def missing_windows(directory: Path, run_ids: Sequence[str], ranks: int) -> list[str]:
    return [hook.window_name(run_id, rank) for run_id in run_ids for rank in range(ranks) if not (directory / hook.window_name(run_id, rank)).is_file()]


def wait_for_windows(directory: Path, run_ids: Sequence[str], ranks: int, *, timeout_s: float, poll_s: float,
                     sleep: Callable[[float], None] = time.sleep, clock: Callable[[], float] = time.monotonic) -> list[str]:
    """The window files still missing when every one is there (empty) or the timeout passes."""
    deadline = clock() + timeout_s
    while True:
        missing = missing_windows(directory, run_ids, ranks)
        if not missing or clock() >= deadline:
            return missing
        sleep(poll_s)


def census_replay_ms(lines: Sequence[str], rows: int, width: int) -> tuple[float | None, int]:
    """(median, count) of the timed `replay_ms` the census logged for a prompt key's warmed replays: the production bracket's own wall."""
    values = []
    for line in lines:
        if CENSUS_TAG not in line:
            continue
        try:
            body = json.loads(line.split(CENSUS_TAG, 1)[1])
        except ValueError:
            continue
        if (body.get("path"), body.get("kind"), body.get("rows"), body.get("context")) == ("graph", "replay", rows, width) and body.get("replay_ms"):
            values.append(float(body["replay_ms"]))
    return (statistics.median(values) if values else None), len(values)


def window_view(events: list[dict]) -> trace_verdict.TraceView | None:
    """The marker window's events split for reading; None unless the trace holds exactly one marker span."""
    markers = trace_verdict.host_markers(events)
    return trace_verdict.split_events(events, markers[0]) if len(markers) == 1 else None


def surviving_window(view: trace_verdict.TraceView | None) -> tuple[float, float] | None:
    """(lead_ms, covered_ms): how far into the marker window the first device op starts, and what is left after it."""
    if view is None or not view.device:
        return None
    lead = (view.device[0][0] - view.start) / 1e3
    return round(lead, 3), round(view.length / 1e3 - lead, 3)


def _metrics(source: dict) -> dict:
    keys = ("device_busy_ms", "kernel_busy_ms", "busy_fraction", "graph_launches", "kernels", "device_events", "idle", "family_top",
            "kernel_top", "largest_idle_gaps")
    return {key: source.get(key) for key in keys}


def _run_shell(run_id: str) -> dict:
    match = RUN_ID.fullmatch(run_id)
    parts = ({"kind": match["kind"], "rows": int(match["rows"]), "width": int(match["width"]), "nth": int(match["nth"])}
             if match else {"kind": None, "rows": None, "width": None, "nth": None})
    return {"run_id": run_id, **parts, "status": MISSING, "verdict": None, "reasons": [], "replay_index": None, "traced_wall_ms": None,
            "unprofiled_wall_ms": None, "profiler_overhead": None, "busy_fraction_unprofiled": None, "busy_fraction_of_covered": None,
            "lead_ms": None, "covered_ms": None, "free_mib_before": None, "free_mib_after": None, "census_replay_ms": None,
            "census_replay_n": 0, "recipes": [], "scan": None, "stage_timeline": None, "gap_classes": None, **_metrics({})}


def _analyze_run(run: dict, directory: Path, ranks: int) -> None:
    """Fill ``run`` from the rank windows and rank 0's trace; status stays MISSING when there is nothing to read."""
    missing = missing_windows(directory, [run["run_id"]], ranks)
    run["reasons"] += [f"{name} is not in {directory}: the rank never finished its window" for name in missing]
    peers = [window for rank in range(ranks) if (window := read_json(directory / hook.window_name(run["run_id"], rank))) is not None]
    window = next((peer for peer in peers if peer.get("rank") == 0), None)
    if window is None:
        return
    run |= {"replay_index": window.get("replay_index"), "traced_wall_ms": window.get("window_ms"),
            "free_mib_before": window.get("free_mib_before"), "free_mib_after": window.get("free_mib_after")}
    events = trace_verdict.load_events(directory / window["trace"]) if window.get("trace") and (directory / window["trace"]).is_file() else []
    chosen = trace_verdict.analyze(events, window, peers)
    final = trace_verdict.candidate_report(chosen, [chosen])
    run["verdict"], run["reasons"] = final["verdict"], run["reasons"] + final["reasons"]
    if chosen.get("device_busy_ms") is None:
        return  # no marker span or no trace: the reasons say which
    source = final if final["verdict"] != trace_verdict.INCOMPLETE else chosen
    run |= _metrics(source)
    run["unprofiled_wall_ms"] = trace_verdict.unprofiled_wall_ms([chosen])
    run["profiler_overhead"] = trace_verdict.overhead(window.get("window_ms"), run["unprofiled_wall_ms"])
    run["busy_fraction_unprofiled"] = (round(source["device_busy_ms"] / run["unprofiled_wall_ms"], 4) if run["unprofiled_wall_ms"] else None)
    view = window_view(events)
    if view is not None:
        run |= trace_tables(view)
    survived = surviving_window(view)
    if survived is not None:
        run["lead_ms"], run["covered_ms"] = survived
        run["busy_fraction_of_covered"] = round(source["device_busy_ms"] / survived[1], 4) if survived[1] > 0 else None
    run["status"] = COMPLETE if final["verdict"] != trace_verdict.INCOMPLETE and not missing else PARTIAL


def build_report(directory: Path, run_ids: Sequence[str], ranks: int, container_lines: Sequence[str]) -> dict:
    runs = []
    for run_id in run_ids:
        run = _run_shell(run_id)
        _analyze_run(run, directory, ranks)
        if run["kind"] == "prompt":
            run["census_replay_ms"], run["census_replay_n"] = census_replay_ms(container_lines, run["rows"], run["width"])
        runs.append(run)
    counts = {status.lower(): sum(run["status"] == status for run in runs) for status in (COMPLETE, PARTIAL, MISSING)}
    return {"runs": runs, "summary": counts, "ranks": ranks, "directory": str(directory)}


def gaps(result: dict) -> list[str]:
    """One line per run that is not COMPLETE, for a cell's gap list."""
    return [f"replay trace {run['run_id']} is {run['status']}: " + ("; ".join(run["reasons"]) or "no reason recorded")[:400]
            for run in result["runs"] if run["status"] != COMPLETE]


# ---- tables from the device record (WIDE-PROMPT-KERNEL-PLAN step 1) ----------------------------------------------------
@dataclass(frozen=True)
class Call:
    """One launch of a recipe: spans[first:stop] of the recipe's span list."""
    start: float
    end: float
    first: int
    stop: int

    @property
    def wall(self) -> float:
        return self.end - self.start


@dataclass(frozen=True)
class RecipeIndex:
    """The window's device spans by recipe id; ``unowned`` holds recipe 0 (host copies, DMA descriptors), ``scan`` the recipes that run an emit
    kernel and ``anchors`` the emit kernels (SCAN_ANCHORS) they ran."""
    groups: dict[str, list[Span]]
    unowned: list[Span]
    scan: frozenset[str]
    anchors: frozenset[str]


@dataclass
class GapTally:
    gaps: int = 0
    total_us: float = 0.0
    max_us: float = 0.0

    def add(self, length: float) -> None:
        self.gaps += 1
        self.total_us += length
        self.max_us = max(self.max_us, length)

    def row(self, band: str, before: str, after: str) -> dict:
        return {"band": band, "before": before, "after": after, "gaps": self.gaps, "total_ms": round(self.total_us / 1e3, 3),
                "max_ms": round(self.max_us / 1e3, 3)}


def recipe_id(event: dict) -> str | None:
    """The event's args.recipeId as text; None for recipe 0 and for events without one."""
    key = trace_verdict.recipe_key(event)
    return None if key is None else key[1]


def group_by_recipe(view: trace_verdict.TraceView) -> RecipeIndex:
    groups: dict[str, list[Span]] = {}
    unowned: list[Span] = []
    scan: set[str] = set()
    anchors: set[str] = set()
    for span in view.device:
        event = view.events[span[2]]
        recipe = recipe_id(event)
        if recipe is None:
            unowned.append(span)
            continue
        groups.setdefault(recipe, []).append(span)
        if event.get("name") in SCAN_ANCHORS:
            scan.add(recipe)
            anchors.add(event["name"])
    return RecipeIndex(groups, unowned, frozenset(scan), frozenset(anchors))


def split_calls(spans: Sequence[Span]) -> list[Call]:
    """A recipe's spans (sorted by start) cut into calls: a new one starts when the recipe has been silent for more than CALL_GAP_US."""
    calls, first, reach = [], 0, spans[0][1]
    for position in range(1, len(spans)):
        start, end, _ = spans[position]
        if start - reach > CALL_GAP_US:
            calls.append(Call(spans[first][0], reach, first, position))
            first, reach = position, end
        else:
            reach = max(reach, end)
    calls.append(Call(spans[first][0], reach, first, len(spans)))
    return calls


def _union_ms(spans: Iterable[Span]) -> float:
    return trace_verdict.total(trace_verdict.union((start, end) for start, end, _ in spans)) / 1e3


def recipe_rows(index: RecipeIndex, calls: dict[str, list[Call]]) -> list[dict]:
    """One row per recipe, busiest first: launches (calls), events, busy union, and the summed, mean, shortest and longest call wall."""
    rows = []
    for recipe, spans in index.groups.items():
        walls = [call.wall for call in calls[recipe]]
        rows.append({"recipe": recipe, "scan": recipe in index.scan, "launches": len(walls), "events": len(spans),
                     "busy_ms": round(_union_ms(spans), 3), "wall_ms": round(sum(walls) / 1e3, 3),
                     "call_ms_mean": round(statistics.fmean(walls) / 1e3, 4), "call_ms_min": round(min(walls) / 1e3, 4),
                     "call_ms_max": round(max(walls) / 1e3, 4)})
    return sorted(rows, key=lambda row: (-row["busy_ms"], row["recipe"]))


def scan_summary(index: RecipeIndex, calls: dict[str, list[Call]], busy_ms: float) -> dict:
    """The scan recipes' device time as one union, next to the union of everything else (the two overlap where the engines run together)."""
    scan_spans = [span for recipe in index.scan for span in index.groups[recipe]]
    rest_spans = index.unowned + [span for recipe, spans in index.groups.items() if recipe not in index.scan for span in spans]
    walls = [call.wall for recipe in index.scan for call in calls[recipe]]
    union_ms = _union_ms(scan_spans)
    anchor = "+".join(sorted(index.anchors)) or SCAN_ANCHORS[0]  # the emit kernels the trace ran; a trace with no scan recipe names the stock one
    return {"anchor": anchor, "recipes": len(index.scan), "launches": len(walls), "union_ms": round(union_ms, 3),
            "wall_ms": round(sum(walls) / 1e3, 3), "rest_ms": round(_union_ms(rest_spans), 3),
            "share": round(union_ms / busy_ms, 4) if busy_ms else None}


def representative_call(index: RecipeIndex, calls: dict[str, list[Call]]) -> tuple[str, int] | None:
    """(recipe, call position): the scan recipe with the most launches (ties: the lowest id) at its median wall (the lower one for an even count)."""
    if not index.scan:
        return None
    recipe = min(index.scan, key=lambda candidate: (-len(calls[candidate]), len(candidate), candidate))
    by_wall = sorted(range(len(calls[recipe])), key=lambda position: calls[recipe][position].wall)
    return recipe, by_wall[(len(by_wall) - 1) // 2]


def stage_cuts(call: Call, spans: Sequence[Span], events: Sequence[dict]) -> list[tuple[str, str, float]]:
    """(stage, anchor, start us) of a scan call in time order: routing from the call start, one stage per anchor kernel's first start, merge from the last emit end."""
    first_start: dict[str, float] = {}
    emit_end: float | None = None
    for start, end, index in spans[call.first:call.stop]:
        name = events[index].get("name")
        first_start.setdefault(name, start)
        if name in SCAN_ANCHORS:
            emit_end = end if emit_end is None else max(emit_end, end)
    cuts = [("routing", "call start", call.start)]
    cuts += [(stage, name, first_start[name]) for stage, name in STAGE_ANCHORS if name in first_start]
    if emit_end is not None:
        cuts.append(("merge", "emit end", emit_end))
    return sorted(cuts, key=lambda cut: cut[2])


def stage_timeline(index: RecipeIndex, calls: dict[str, list[Call]], events: Sequence[dict]) -> dict | None:
    """One scan call cut at its anchor kernels; each stage runs to the next one's start and the last to the call's end."""
    picked = representative_call(index, calls)
    if picked is None:
        return None
    recipe, position = picked
    call = calls[recipe][position]
    cuts = stage_cuts(call, index.groups[recipe], events)
    stops = [cut[2] for cut in cuts[1:]] + [call.end]
    stages = [{"stage": stage, "anchor": anchor, "start_us": round(start - call.start, 1), "length_us": round(stop - start, 1)}
              for (stage, anchor, start), stop in zip(cuts, stops, strict=True)]
    return {"recipe": recipe, "call_index": position, "calls": len(calls[recipe]), "wall_us": round(call.wall, 1), "stages": stages}


def band_of(length_us: float) -> str:
    return next((label for floor, label in GAP_BANDS if length_us >= floor), UNDER_BAND)


def gap_label(events: Sequence[dict], index: int | None, window_edge: str) -> str:
    """The kernel beside an idle stretch as ``name`` or ``name@recipe``, the name cut at its first space (``H2D pdma_tx_commands 0 chunk=5`` is ``H2D``)."""
    if index is None:
        return window_edge
    event = events[index]
    name = str(event.get("name", "?")).split(" ", 1)[0]
    recipe = recipe_id(event)
    return name if recipe is None else f"{name}@{recipe}"


def gap_classes(view: trace_verdict.TraceView, runs: Sequence[trace_verdict.Busy]) -> dict:
    """The window's device idle: stretches under GAP_SPLIT_US as one class, longer ones by (size band, kernel before, kernel after)."""
    under, classes = GapTally(), {}
    for low, before, high, after in trace_verdict.gap_edges(view, runs):
        length = high - low
        if length < GAP_SPLIT_US:
            under.add(length)
            continue
        key = (band_of(length), gap_label(view.events, before, WINDOW_START), gap_label(view.events, after, WINDOW_END))
        classes.setdefault(key, GapTally()).add(length)
    ranked = sorted(classes.items(), key=lambda item: (-item[1].total_us, item[0]))
    over_us = sum(tally.total_us for tally in classes.values())
    return {"threshold_us": GAP_SPLIT_US, "idle_ms": round((under.total_us + over_us) / 1e3, 3),
            "over_gaps": sum(tally.gaps for tally in classes.values()), "over_ms": round(over_us / 1e3, 3), "class_count": len(classes),
            "classes": [under.row(UNDER_BAND, "*", "*"), *(tally.row(*key) for key, tally in ranked[:GAP_CLASS_KEEP])]}


def trace_tables(view: trace_verdict.TraceView) -> dict:
    """Recipe rows, the scan union, one scan call's stage timeline and the idle-gap classes of one marker window."""
    runs = trace_verdict.busy_runs(view.device)
    busy_ms = sum(run.end - run.start for run in runs) / 1e3
    index = group_by_recipe(view)
    calls = {recipe: split_calls(spans) for recipe, spans in index.groups.items()}
    return {"recipes": recipe_rows(index, calls), "scan": scan_summary(index, calls, busy_ms),
            "stage_timeline": stage_timeline(index, calls, view.events), "gap_classes": gap_classes(view, runs)}


# ---- text ------------------------------------------------------------------------------------------------------------
def _table(tag: str, run: dict, key: str, limit: int | None) -> list[str]:
    entries = (run.get(key) or [])[:limit]
    if not entries:
        return []
    head = f"{tag} run={run['run_id']} device_busy_ms={run['device_busy_ms']} top={len(entries)} columns=share,busy_ms,events,name"
    return [head] + [f"  {item['share']:.4f} {item['busy_ms']:10.3f} {item['events']:9d} {item['name']}" for item in entries]


def _run_line(run: dict) -> str:
    fields = ("status", "verdict", "replay_index", "traced_wall_ms", "unprofiled_wall_ms", "profiler_overhead", "device_busy_ms", "busy_fraction",
              "busy_fraction_unprofiled", "busy_fraction_of_covered", "graph_launches", "kernels", "device_events", "lead_ms", "covered_ms",
              "census_replay_ms", "free_mib_before", "free_mib_after")
    return f"REPLAY_TRACE_RUN run={run['run_id']} " + " ".join(f"{name}={run[name]}" for name in fields)


def _scan_lines(run: dict) -> list[str]:
    scan = run.get("scan")
    if not scan or not scan["recipes"]:
        return []
    fields = " ".join(f"{name}={scan[name]}" for name in ("anchor", "recipes", "launches", "union_ms", "wall_ms", "rest_ms", "share"))
    return [f"REPLAY_TRACE_SCAN run={run['run_id']} {fields} device_busy_ms={run['device_busy_ms']}"]


def _recipe_lines(run: dict) -> list[str]:
    rows = run.get("recipes") or []
    if not rows:
        return []
    shown = rows[:RECIPE_LINES]
    head = (f"REPLAY_TRACE_RECIPE run={run['run_id']} recipes={len(rows)} shown={len(shown)} "
            "columns=recipe,scan,launches,events,busy_ms,wall_ms,call_ms_mean,call_ms_min,call_ms_max")
    return [head] + [f"  {row['recipe']:>6} {'scan' if row['scan'] else '-':>4} {row['launches']:7d} {row['events']:9d} {row['busy_ms']:10.3f} "
                     f"{row['wall_ms']:10.3f} {row['call_ms_mean']:9.4f} {row['call_ms_min']:9.4f} {row['call_ms_max']:9.4f}" for row in shown]


def _stage_lines(run: dict) -> list[str]:
    timeline = run.get("stage_timeline")
    if not timeline:
        return []
    head = (f"REPLAY_TRACE_STAGE run={run['run_id']} recipe={timeline['recipe']} call_index={timeline['call_index']} calls={timeline['calls']} "
            f"wall_us={timeline['wall_us']} columns=stage,start_us,length_us,anchor")
    return [head] + [f"  {stage['stage']:<10} {stage['start_us']:9.1f} {stage['length_us']:9.1f} {stage['anchor']}" for stage in timeline["stages"]]


def _gap_lines(run: dict) -> list[str]:
    gaps = run.get("gap_classes")
    if not gaps:
        return []
    rows = gaps["classes"][:GAP_CLASS_LINES + 1]  # the under-threshold class, then the largest long-gap classes
    head = (f"REPLAY_TRACE_GAP run={run['run_id']} threshold_us={gaps['threshold_us']} idle_ms={gaps['idle_ms']} over_gaps={gaps['over_gaps']} "
            f"over_ms={gaps['over_ms']} classes={gaps['class_count']} shown={len(rows)} columns=band,gaps,total_ms,max_ms,before,after")
    return [head] + [f"  {row['band']:>8} {row['gaps']:7d} {row['total_ms']:9.3f} {row['max_ms']:8.3f} {row['before']} -> {row['after']}" for row in rows]


def _detail_lines(run: dict) -> list[str]:
    return _scan_lines(run) + _recipe_lines(run) + _stage_lines(run) + _gap_lines(run)


def render(result: dict) -> list[str]:
    lines: list[str] = []
    for run in result["runs"]:
        lines.append(_run_line(run))
        lines += _table("REPLAY_TRACE_FAMILY", run, "family_top", None)
        lines += _table("REPLAY_TRACE_KERNEL", run, "kernel_top", KERNEL_LINES)
        lines += [f"REPLAY_TRACE_REASON run={run['run_id']} {reason}" for reason in run["reasons"]]
        lines += _detail_lines(run)
    summary = result["summary"]
    lines.append(f"REPLAY_TRACE_SUMMARY complete={summary['complete']} partial={summary['partial']} missing={summary['missing']}")
    return lines


# ---- command line ----------------------------------------------------------------------------------------------------
def command_wait(args: argparse.Namespace) -> int:
    missing = wait_for_windows(args.dir, args.runs, args.ranks, timeout_s=args.timeout_s, poll_s=args.poll_s)
    for name in missing:
        print(f"REPLAY_TRACE_WAIT missing {name}")
    print(f"REPLAY_TRACE_WAIT runs={len(args.runs)} missing={len(missing)}", flush=True)
    return EXIT_MISSING if missing else EXIT_OK


def command_report(args: argparse.Namespace) -> int:
    lines = args.container_log.read_text(errors="replace").splitlines() if args.container_log else []
    result = build_report(args.dir, args.runs, args.ranks, lines)
    for line in render(result):
        print(line)
    if args.out:
        args.out.write_text(json.dumps(result, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    return EXIT_OK if not gaps(result) else EXIT_INCOMPLETE


def run_list(text: str) -> list[str]:
    runs = [item.strip() for item in text.split(",") if item.strip()]
    if not runs or any(RUN_ID.fullmatch(run) is None for run in runs):
        raise argparse.ArgumentTypeError(f"--runs {text!r}: comma separated run ids such as prompt-1024x8192-n3,decode-16x8192-n5")
    return runs


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    verbs = parser.add_subparsers(dest="verb", required=True)
    for verb in ("wait", "report"):
        sub = verbs.add_parser(verb)
        sub.add_argument("--dir", required=True, type=Path, help="the host folder behind the hook's directory")
        sub.add_argument("--runs", required=True, type=run_list)
        sub.add_argument("--ranks", type=int, default=DEFAULT_RANKS)
        if verb == "wait":
            sub.add_argument("--timeout-s", type=float, default=600.0)
            sub.add_argument("--poll-s", type=float, default=2.0)
        else:
            sub.add_argument("--out", type=Path, default=None, help="write the full report here as JSON")
            sub.add_argument("--container-log", type=Path, default=None, help="docker logs: adds the census's timed replay_ms per prompt key")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    return command_wait(args) if args.verb == "wait" else command_report(args)


if __name__ == "__main__":
    raise SystemExit(main())
