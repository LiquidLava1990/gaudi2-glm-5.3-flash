"""Per-lane speed, per-turn token accounting and the decode step breakdown of one sustain_harness / sustain_long500 run (measure phase only).

usage: turn_breakdown.py RUN_DIR [--turns RUN_DIR/medium/turns.jsonl] [--log RUN_DIR/container.log]

Lanes: decode tok/s = completion_tokens / decode_s per measured turn; per-lane p50, then p50 and min over lanes; lanes_at_10 = lanes whose p50 >= 10.
Turns: growth = prompt_tokens minus the same lane's previous prompt_tokens; recomputed = prompt tokens the engine prefilled in the measure window
(engine step prompt chunks) per measured turn; loaded = growth-independent remainder prompt_tokens - recomputed; re-read fraction = loaded / prompt.
Steps (measure window): wall p50 by kind, the share of non-idle steps and of wall that carry a prefill piece, decode steps per second.
"""
from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
from pathlib import Path

L = Path.home() / ".codex/tmp/flash-spikes-20261104/team-20261106-dual"
sys.path[:0] = [str(L / "diag-cells"), str(L / "diag-loadpath")]

import loadpath_report  # noqa: E402  (FENCE, METRIC, HIT: the same line formats its report reads)
import read_cells  # noqa: E402


def pct(values: list[float], q: float) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(q * (len(ordered) - 1) + 0.5))] if ordered else float("nan")


def measured(turns_path: Path) -> list[dict]:
    rows = [json.loads(line) for line in turns_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return [row for row in rows if row.get("phase") == "measure" and row.get("lane") is not None]


def lane_lines(rows: list[dict]) -> list[str]:
    per_lane: dict[int, list[float]] = {}
    for row in rows:
        if row.get("decode_s") and row.get("completion_tokens", 0) > 1:
            per_lane.setdefault(row["lane"], []).append(row["completion_tokens"] / row["decode_s"])
    lane_p50 = {lane: statistics.median(values) for lane, values in per_lane.items()}
    ttft = [row["ttft_s"] for row in rows if row.get("ttft_s")]
    prefill = [row["prompt_tokens"] / row["ttft_s"] for row in rows if row.get("ttft_s")]
    prompt = [row["prompt_tokens"] for row in rows]
    detail = " ".join(f"{lane}:{value:.2f}" for lane, value in sorted(lane_p50.items()))
    return [f"LANES turns={len(rows)} lanes={len(lane_p50)} decode_tok_s lane-p50: p50={statistics.median(lane_p50.values()):.2f} "
            f"min={min(lane_p50.values()):.2f} max={max(lane_p50.values()):.2f} lanes_at_10={sum(v >= 10 for v in lane_p50.values())}/{len(lane_p50)}",
            f"LANES per-lane decode p50 tok/s: {detail}",
            f"TTFT p50={pct(ttft, 0.5):.2f}s p90={pct(ttft, 0.9):.2f}s  prefill tok/s (prompt/ttft) p50={pct(prefill, 0.5):.0f}  "
            f"prompt tokens p50={pct(prompt, 0.5):.0f} max={max(prompt)}"]


def turn_lines(rows: list[dict], steps: list[dict]) -> list[str]:
    previous: dict[int, int] = {}
    growth = []
    for row in sorted(rows, key=lambda r: r["started_at"]):
        lane = row["lane"]
        if lane in previous:
            growth.append(row["prompt_tokens"] - previous[lane])
        previous[lane] = row["prompt_tokens"]
    recomputed = sum(chunk for step in steps for chunk, _ in (step.get("prompts") or []))
    prompt_total = sum(row["prompt_tokens"] for row in rows)
    per_turn = recomputed / len(rows)
    loaded = max(0.0, prompt_total - recomputed) / len(rows)
    return [f"TURNS per measured turn: growth p50={pct(growth, 0.5):.0f} mean={statistics.mean(growth):.0f} tok; recomputed (prefilled in the window) "
            f"mean={per_turn:.0f} tok; loaded/reused mean={loaded:.0f} tok; re-read fraction={loaded / (prompt_total / len(rows)):.1%} "
            f"(prompt mean {prompt_total / len(rows):.0f})"]


def step_lines(steps: list[dict]) -> list[str]:
    busy = [step for step in steps if step.get("kind") != "idle"]
    wall = sum(step["wall_ms"] for step in busy)
    lines = []
    for kind in ("pure-decode", "mixed", "prompt-only"):
        chosen = [step["wall_ms"] for step in busy if step.get("kind") == kind]
        if chosen:
            lines.append(f"STEPS {kind}: n={len(chosen)} wall_ms p50={statistics.median(chosen):.1f} mean={statistics.mean(chosen):.1f} "
                         f"wall_share={sum(chosen) / wall:.1%}")
    prefill = [step for step in busy if step.get("kind") in ("mixed", "prompt-only")]
    decode = [step for step in busy if step.get("kind") in ("pure-decode", "mixed")]
    seconds = wall / 1000
    lines.append(f"STEPS carrying a prefill piece: {len(prefill)}/{len(busy)} = {len(prefill) / len(busy):.1%} of steps, "
                 f"{sum(s['wall_ms'] for s in prefill) / wall:.1%} of wall; decode-bearing steps/s = {len(decode) / seconds:.2f} over {seconds:.0f} s")
    return lines


METRIC_INTERVAL_S = 10.0  # vLLM logs the KV Transfer metrics every 10 s; a line's bytes are the interval that ends at its stamp


def window_events(log_path: Path, start: float, end: float, local_re: re.Pattern, evict_re: re.Pattern) -> dict:
    """Inside the window, engine-wide: fence waits and load-byte intervals with their times, the last hit-rate line, the session pin's
    local-hit token readings (local_re group 1) and its eviction line count."""
    events: dict = {"fences": [], "loads": [], "hit": None, "local_hits": [], "evictions": 0}
    with log_path.open(encoding="utf-8", errors="replace") as handle:
        for line in handle:
            stamp = line.split(" ", 1)[0]
            if not stamp[:4].isdigit():
                continue
            moment = read_cells.log_epoch(stamp)
            if not start <= moment <= end:
                continue
            fence = loadpath_report.FENCE.search(line)
            if fence:
                events["fences"].append((moment, float(fence.group(1))))
                continue
            if "KV Transfer metrics" in line:
                events["loads"].append((moment, sum(float(v) for name, v in loadpath_report.METRIC.findall(line) if name == "load_bytes")))
                continue
            events["hit"] = loadpath_report.HIT.search(line) or events["hit"]
            local = local_re.search(line)
            if local:
                events["local_hits"].append(int(local.group(1)))
            events["evictions"] += bool(evict_re.search(line))
    return events


def load_lines(events: dict, turns: int) -> list[str]:
    """Totals over the window per measured turn of every lane started in it."""
    fences = [ms for _, ms in events["fences"]]
    load_bytes = sum(value for _, value in events["loads"])
    hit = events["hit"]
    external = f"{float(hit.group(2)):.1f}%" if hit and hit.group(2) else "n/a"
    local = events["local_hits"]
    return [f"LOADS window: fence p50={pct(fences, 0.5):.1f} ms p90={pct(fences, 0.9):.1f} n={len(fences)}; loaded {load_bytes / 1e9:.1f} GB = "
            f"{load_bytes / 1e9 / max(turns, 1):.2f} GB/turn; external hit (last 10 s line) {external}; "
            f"pin local_hit_tokens lines={len(local)} last={local[-1] if local else 'n/a'} sum={sum(local)}; pin eviction lines={events['evictions']}"]


def class_lines(rows: list[dict], events: dict) -> list[str]:
    """Each lane's first measured turn vs its later turns: loads and fences attributed by time to the turns' prefill windows
    [started_at, started_at + ttft]; a load interval shared by several open windows is split evenly between them."""
    first_turn = {}
    for row in sorted(rows, key=lambda r: r["started_at"]):
        first_turn.setdefault(row["lane"], row["turn"])
    windows = [(row["started_at"], row["started_at"] + (row.get("ttft_s") or 0), "first" if row["turn"] == first_turn[row["lane"]] else "later")
               for row in rows]
    gigabytes = {"first": 0.0, "later": 0.0}
    for moment, value in events["loads"]:
        open_windows = [kind for begin, finish, kind in windows if begin < moment and finish > moment - METRIC_INTERVAL_S]
        for kind in open_windows:
            gigabytes[kind] += value / 1e9 / len(open_windows)
    lines = []
    for kind in ("first", "later"):
        mine = [(begin, finish) for begin, finish, k in windows if k == kind]
        fences = [ms for moment, ms in events["fences"] if any(begin <= moment <= finish for begin, finish in mine)]
        ttft = [finish - begin for begin, finish in mine]
        lines.append(f"PINCLASS {kind}-turns n={len(mine)} GB/turn={gigabytes[kind] / max(len(mine), 1):.2f} fence p50={pct(fences, 0.5):.1f} ms "
                     f"n={len(fences)} TTFT p50={pct(ttft, 0.5):.2f}s (time-attributed; first = each lane's first measured turn)")
    return lines


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("run", type=Path)
    parser.add_argument("--turns", type=Path)
    parser.add_argument("--log", type=Path)
    parser.add_argument("--min-prompt", type=int, default=0, help="keep measured turns whose prompt is at least this (a context band)")
    parser.add_argument("--max-prompt", type=int, default=10**9)
    parser.add_argument("--local-hit-re", default=r"local_hit_tokens=([0-9]+)", help="the session pin's local-hit token reading (group 1)")
    parser.add_argument("--evict-re", default=r"REUSE_V2\S*.*evict", help="the session pin's eviction line")
    args = parser.parse_args(argv)
    every = measured(args.turns or args.run / "medium" / "turns.jsonl")
    rows = [row for row in every if args.min_prompt <= row["prompt_tokens"] <= args.max_prompt]
    if not rows:
        print("TURN_BREAKDOWN: no measured turns in the band", file=sys.stderr)
        return 1
    start = min(row["started_at"] for row in rows)
    end = max(row["started_at"] + (row.get("wall_s") or 0) for row in rows)
    log_path = args.log or args.run / "container.log"
    steps = [step for step in read_cells.Log(log_path).engine if start <= step["t"] <= end]
    band = f" band {args.min_prompt}-{args.max_prompt}" if args.min_prompt or args.max_prompt < 10**9 else ""
    events = window_events(log_path, start, end, re.compile(args.local_hit_re), re.compile(args.evict_re))
    turns_in_window = sum(1 for row in every if start <= row["started_at"] <= end)
    print("\n".join([f"WINDOW measure{band} {end - start:.0f} s, {len(steps)} engine steps"] + lane_lines(rows) + turn_lines(rows, steps)
                    + load_lines(events, turns_in_window) + class_lines(rows, events) + step_lines(steps)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
