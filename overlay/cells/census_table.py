"""Step 20's card output from GLM53_PROMPT_CENSUS lines: table per call site per cell, graph-census.tsv of capture and
replay bytes per graph shape, need-table.json, written verdicts.

    census_table.py --log on:<docker log>[:<request begin utc>] --log off:<docker log>[:<begin>] --out-dir <dir>

Each log holds one boot and one cold request. The line's `context` is the padded bucket (block_list x block_size); a true
context would need a device read, which the census may not do, so here it is the rows that request already sent.
The verdicts are a first reading of the numbers for the three questions in plan step 20; they name their evidence.
"""
from __future__ import annotations

import argparse
import json
import re
import statistics
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

MIB = 1 << 20
ROWS = (512, 1024, 1792, 2048)
CONTEXTS = (70_000, 131_072, 262_144, 500_000)
DRAINS = ("on", "off")
CELL_WINDOW_CHUNKS = 2  # a reading belongs to a cell within this many of its own chunks of the target context
SATURATED_WITHIN_MIB = 64  # run-ahead readings sit 18-64 MiB under Limit [M guard SATURATION notes]
QUEUE_SHARE = 0.5  # the jump is the queue when drains remove more than this share of it
SCALE_RATIO = 1.5
SCALE_MIN_DELTA_MIB = 64
LINE = re.compile(r"GLM53_PROMPT_CENSUS (\{.*\})\s*$")
STAMP_CHARS = 19  # YYYY-MM-DDTHH:MM:SS, comparable as text once both sides are UTC
GRAPH_FIELDS = ("kind", "rows", "context", "readings", "peak_mib_max", "between_mib_max")
TSV_FIELDS = ("rows", "context", "drains", "site", "op", "readings", "growth_mib_median", "max_growth_mib",
              "peak_advance_mib_median", "calls_median")


@dataclass(frozen=True)
class Reading:
    pid: int
    rows: int
    bucket: int
    true_context: int
    start: int
    top: int
    limit: int
    drains: str
    sites: tuple[dict, ...]

    @property
    def peak_mib(self) -> int:
        return (self.top - self.start) // MIB


def _body(line: str) -> dict | None:
    if "GLM53_PROMPT_CENSUS" not in line or "GLM53_PROMPT_CENSUS read_error" in line:
        return None
    match = LINE.search(line)
    try:
        return json.loads(match.group(1)) if match else _refuse(line)
    except json.JSONDecodeError:
        return _refuse(line)


def _refuse(line: str) -> dict:
    raise SystemExit(f"garbled census line, re-extract the log from docker logs: {line[:200]}")


def parse(lines: Iterable[str], drains: str, since: str | None = None) -> list[Reading]:
    """Readings of one boot's one cold request, in order; a line older than ``since`` is warmup."""
    sent: dict[int, int] = defaultdict(int)
    readings = []
    for line in lines:
        body = _body(line)
        if body is None or body.get("path") == "graph" or (since is not None and line[:STAMP_CHARS] < since[:STAMP_CHARS]):
            continue
        pid, rows = int(body["pid"]), int(body["rows"])
        if sent[pid] > int(body["context"]):
            raise SystemExit(f"pid {pid} sent {sent[pid]} tokens but the bucket is {body['context']}: this log holds more "
                             "than one request, or warmup; pass the request's begin time as the third --log field")
        readings.append(Reading(pid, rows, int(body["context"]), sent[pid], int(body["start_in_use"] or 0),
                                int(body["max_in_use"] or 0), int(body["limit"] or 0), drains, tuple(body["sites"])))
        sent[pid] += rows
    return readings


def cell_readings(readings: Iterable[Reading], rows: int, context: int, drains: str) -> list[Reading]:
    window = CELL_WINDOW_CHUNKS * rows
    return [r for r in readings
            if r.rows == rows and r.drains == drains and abs(r.true_context - context) <= window]


def site_rows(readings: Iterable[Reading]) -> list[dict]:
    """One row per cell and call site: this is the plan's 'bytes per call site per cell'."""
    grouped: dict[tuple, list[dict]] = defaultdict(list)
    for reading in readings:
        for site in reading.sites:
            grouped[reading.rows, reading.true_context, reading.drains, site["site"], site["op"]].append(site)
    return [_site_row(key, sites) for key, sites in sorted(grouped.items())]


def _site_row(key: tuple, sites: list[dict]) -> dict:
    rows, context, drains, site, op = key
    return {"rows": rows, "context": context, "drains": drains, "site": site, "op": op, "readings": len(sites),
            "growth_mib_median": statistics.median(s["growth"] for s in sites) / MIB,
            "max_growth_mib": max(s["max_growth"] for s in sites) / MIB,
            "peak_advance_mib_median": statistics.median(s["peak_advance"] for s in sites) / MIB,
            "calls_median": float(statistics.median(s["calls"] for s in sites))}


def to_tsv(rows: Iterable[dict], fields: tuple[str, ...] = TSV_FIELDS) -> str:
    lines = ["\t".join(fields)]
    lines += ["\t".join(str(row[field]) for field in fields) for row in rows]
    return "\n".join(lines) + "\n"


def graph_rows(lines: Iterable[str]) -> list[dict]:
    """Largest capture and replay peak per graph shape, and the largest between-forward peak: all above InUse at start."""
    grouped: dict[tuple[str, int, int], list[dict]] = defaultdict(list)
    for line in lines:
        body = _body(line)
        if body is not None and body.get("path") == "graph" and body["start_in_use"] is not None:
            grouped[body["kind"], body["rows"], body["context"]].append(body)
    return [{"kind": kind, "rows": rows, "context": context, "readings": len(bodies),
             "peak_mib_max": max(((b["max_in_use"] or 0) - b["start_in_use"]) // MIB for b in bodies),
             "between_mib_max": max(((b["between_peak"] or 0) - b["start_in_use"]) // MIB for b in bodies)}
            for (kind, rows, context), bodies in sorted(grouped.items())]


def planned_cells(readings: list[Reading]) -> Iterable[Reading]:
    """Each reading relabelled with the plan context it falls on; a reading on no plan cell is dropped."""
    for reading in readings:
        for context in CONTEXTS:
            if abs(reading.true_context - context) <= CELL_WINDOW_CHUNKS * reading.rows:
                yield Reading(reading.pid, reading.rows, reading.bucket, context, reading.start, reading.top, reading.limit,
                              reading.drains, reading.sites)


def _is_saturated(reading: Reading) -> bool:
    return reading.limit - reading.top <= SATURATED_WITHIN_MIB * MIB


def need_table(readings: Iterable[Reading]) -> dict:
    """Drains-on need per (rows, context), for step 23's HPU_GLM53_GS_NEED_TABLE. A saturated reading is flagged: it
    reads the Limit, not the need."""
    grouped: dict[tuple[int, int], list[Reading]] = defaultdict(list)
    for reading in readings:
        if reading.drains == "on":
            grouped[reading.rows, reading.true_context].append(reading)
    entries = [{"rows": rows, "context": context, "need_mib": max(r.peak_mib for r in group), "readings": len(group),
                "saturated": sum(_is_saturated(r) for r in group)}
               for (rows, context), group in sorted(grouped.items())]
    return {"schema": 1, "source": "drains-on prompt census", "entries": entries}


def missing_cells(readings: list[Reading]) -> list[dict]:
    return [{"rows": rows, "context": context, "drains": drains}
            for rows in ROWS for context in CONTEXTS for drains in DRAINS
            if not cell_readings(readings, rows, context, drains)]


def _median_peak(readings: Iterable[Reading], rows: int, context: int, drains: str) -> float | None:
    peaks = [r.peak_mib for r in cell_readings(readings, rows, context, drains)]
    return statistics.median(peaks) if peaks else None


def jump_verdict(readings: list[Reading], context: int = 131_072) -> dict:
    """Is the 512-to-1024 jump the R589 queue? Yes when turning drains on removes most of the drains-off jump."""
    cells = {(rows, drains): _median_peak(readings, rows, context, drains) for rows in (512, 1024) for drains in DRAINS}
    evidence = {f"{rows}_{drains}_mib": peak for (rows, drains), peak in cells.items()}
    if None in cells.values():
        return {"answer": "not_measured", "context": context, **evidence}
    jump = cells[1024, "off"] - cells[512, "off"]
    removed = cells[1024, "off"] - cells[1024, "on"]
    answer = "queue" if jump > 0 and removed > QUEUE_SHARE * jump else "not_queue"
    return {"answer": answer, "context": context, "jump_mib": jump, "removed_by_drains_mib": removed, **evidence}


def _largest_growing_site(readings: list[Reading], low: int, high: int) -> str | None:
    def growth_by_site(context: int) -> dict[str, float]:
        cell = cell_readings(readings, 1024, context, "on")
        return {row["site"]: row["growth_mib_median"] for row in site_rows(cell)}

    before, after = growth_by_site(low), growth_by_site(high)
    deltas = {site: mib - before.get(site, 0.0) for site, mib in after.items()}
    return max(deltas, key=lambda site: deltas[site]) if deltas else None


def width_verdict(readings: list[Reading], low: int = 70_000, high: int = 131_072) -> dict:
    """What explains 1,995 vs 3,817 at 1024 rows? The queue when the gap closes with drains on, else width at a named site."""
    off_low, off_high = _median_peak(readings, 1024, low, "off"), _median_peak(readings, 1024, high, "off")
    on_low, on_high = _median_peak(readings, 1024, low, "on"), _median_peak(readings, 1024, high, "on")
    if None in (off_low, off_high, on_low, on_high):
        return {"answer": "not_measured", "low": low, "high": high}
    gap_off, gap_on = off_high - off_low, on_high - on_low
    answer = "queue" if gap_on <= (1 - QUEUE_SHARE) * max(gap_off, 0) else "width"
    return {"answer": answer, "low": low, "high": high, "gap_off_mib": gap_off, "gap_on_mib": gap_on,
            "largest_site": _largest_growing_site(readings, low, high)}


def scaling_sites(readings: list[Reading]) -> list[dict]:
    """Does any site grow with context? Drains on, so the queue is out of the reading; compared at the cell ends."""
    by_site: dict[tuple[int, str, str], dict[int, float]] = defaultdict(dict)
    for row in site_rows(r for r in readings if r.drains == "on"):
        by_site[row["rows"], row["site"], row["op"]][row["context"]] = row["growth_mib_median"]
    named = []
    for (rows, site, op), growth in sorted(by_site.items()):
        small, large = min(growth), max(growth)
        delta = growth[large] - growth[small]
        if large > small and delta > SCALE_MIN_DELTA_MIB and growth[large] > SCALE_RATIO * max(growth[small], 0.0):
            named.append({"rows": rows, "site": site, "op": op, "smallest_context": small, "largest_context": large,
                          "growth_mib_at_smallest": growth[small], "growth_mib_at_largest": growth[large]})
    return named


def devmem_count(lines: Iterable[str]) -> int:
    """Any PT_DEVMEM is recorded with its cell and is itself a finding (plan step 20)."""
    return sum("PT_DEVMEM" in line for line in lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--log", action="append", required=True, help="on|off:<log path>[:<request begin utc>]")
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    readings: list[Reading] = []
    devmem = 0
    graphs: list[str] = []
    for item in args.log:
        drains, path, *since = item.split(":", 2)
        lines = Path(path).read_text(errors="replace").splitlines()
        devmem += devmem_count(lines)
        graphs += lines
        readings += parse(lines, drains, since[0] if since else None)
    cells = list(planned_cells(readings))
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "census-table.tsv").write_text(to_tsv(site_rows(cells)))
    (args.out_dir / "graph-census.tsv").write_text(to_tsv(graph_rows(graphs), GRAPH_FIELDS))
    (args.out_dir / "need-table.json").write_text(json.dumps(need_table(cells), indent=2) + "\n")
    verdicts = {"jump_512_to_1024": jump_verdict(cells), "gap_1995_vs_3817": width_verdict(cells),
                "context_scaled_sites": scaling_sites(cells), "devmem_lines": devmem, "missing_cells": missing_cells(cells)}
    (args.out_dir / "census-verdicts.json").write_text(json.dumps(verdicts, indent=2) + "\n")
    print("CENSUS_TABLE", json.dumps({k: verdicts[k] for k in ("jump_512_to_1024", "gap_1995_vs_3817", "devmem_lines")}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
