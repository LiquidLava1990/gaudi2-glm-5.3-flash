"""Step 22 read-cost cell: what an allocator read costs at an R589 boundary, and the sampling interval that follows.

NOT RUN by the build lane: needs a card. Measures, on one card in lazy mode:
  idle        memory_stats() and memory_allocated() with nothing queued;
  under_load  memory_stats() right after mark_step(sync=True) with the queue undrained, as a boundary reads it.
Then pick_interval() gives HPU_GLM53_RUNAHEAD_SAMPLE_EVERY: the smallest N whose reads fit READ_BUDGET of a forward, capped
by how many boundaries of growth the margin can cover between reads. Pass the census's largest per-boundary growth with
--max-boundary-growth-mib once step 20 has run; the growth seen here is only this cell's synthetic tile. The quad cell
then confirms it live: GLM53_RUNAHEAD event=stats carries reads and read_ns_total.

`--device cpu` rehearses the runner logic with a fake allocator; its verdict never counts as the cell (`card_verdict=false`).
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
import time
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import torch

MIB = 1 << 20
BOUNDARIES_PER_FORWARD = 866  # [M census]
FORWARD_MS = 2341.0  # a traced prompt forward [M C820]
READ_BUDGET = 0.003  # a tenth of step 22's 3% hot C16 bar
DRAIN_AT_LIMIT_SHARE = 0.5  # this cell drains its synthetic queue at half the Limit: it measures reads, not the Limit
DEFAULT_MARGIN_MIB = 512


def summarize_ns(samples: list[int]) -> dict[str, float]:
    ordered = sorted(samples)
    p99 = ordered[max(0, math.ceil(0.99 * len(ordered)) - 1)]
    return {"n": len(ordered), "p50_ns": ordered[max(0, math.ceil(0.5 * len(ordered)) - 1)], "p99_ns": p99,
            "max_ns": ordered[-1], "mean_ns": statistics.fmean(ordered)}


def pick_interval(read_ns: float, boundaries: int, forward_ms: float, budget_fraction: float, margin_mib: int,
                  max_boundary_growth_mib: float) -> dict[str, Any]:
    """Smallest N whose reads fit the budget, and the largest N the margin allows. ``fits`` false means raise the margin."""
    budget_ns = budget_fraction * forward_ms * 1e6
    sample_every = max(1, math.ceil(read_ns * boundaries / budget_ns))
    ceiling = boundaries if max_boundary_growth_mib <= 0 else max(1, math.floor(margin_mib / max_boundary_growth_mib))
    fits = sample_every <= ceiling
    action = (f"set HPU_GLM53_RUNAHEAD_SAMPLE_EVERY={sample_every}" if fits else
              f"reads need N={sample_every} but the margin covers N={ceiling}: raise HPU_GLM53_RUNAHEAD_MARGIN_MIB to at "
              f"least {math.ceil(sample_every * max_boundary_growth_mib)}, then re-run the hot C16 bar")
    return {"sample_every": sample_every, "ceiling": ceiling, "fits": fits, "action": action, "read_ns": read_ns,
            "boundaries": boundaries, "forward_ms": forward_ms, "budget_fraction": budget_fraction,
            "margin_mib": margin_mib, "max_boundary_growth_mib": max_boundary_growth_mib}


def cell_passes(device: str, pick: Mapping[str, Any]) -> bool:
    return device == "hpu" and bool(pick["fits"])


def _timed_reads(read: Callable[[], object], count: int) -> list[int]:
    samples = []
    for _ in range(count):
        started = time.perf_counter_ns()
        read()
        samples.append(time.perf_counter_ns() - started)
    return samples


class Runtime:
    """The calls the cell makes: real on a card, stand-ins for a CPU rehearsal."""

    def __init__(self, device: str) -> None:
        self.device = device
        if device == "hpu":
            import habana_frameworks.torch as htorch

            self.memory_stats, self.memory_allocated = htorch.hpu.memory_stats, htorch.hpu.memory_allocated
            self.mark_step, self.synchronize = htorch.core.mark_step, htorch.hpu.synchronize
            return
        self.memory_stats = lambda: {"Limit": 1 << 40, "InUse": 0}
        self.memory_allocated = lambda: 0
        self.mark_step = lambda sync=False: None
        self.synchronize = lambda: None


def under_load(runtime: Runtime, boundaries: int, tile_rows: int) -> dict[str, Any]:
    """Queue a matmul per boundary and leave it undrained, as drains-off serving does, reading right after mark_step."""
    samples, growths = [], []
    tile = torch.ones((tile_rows, tile_rows), dtype=torch.float32, device=runtime.device)
    limit = int(runtime.memory_stats()["Limit"])
    last = int(runtime.memory_stats()["InUse"])
    held: list[torch.Tensor] = []
    for _ in range(boundaries):
        held.append(tile @ tile)
        runtime.mark_step(sync=True)
        started = time.perf_counter_ns()
        stats = runtime.memory_stats()
        samples.append(time.perf_counter_ns() - started)
        in_use = int(stats["InUse"])
        growths.append(in_use - last)
        last = in_use
        if in_use > DRAIN_AT_LIMIT_SHARE * limit:
            runtime.synchronize()
            held.clear()
            last = int(runtime.memory_stats()["InUse"])
    runtime.synchronize()
    return {"read": summarize_ns(samples), "tile_rows": tile_rows, "max_boundary_growth_mib": max(growths) / MIB}


def run(args: argparse.Namespace) -> dict[str, Any]:
    runtime = Runtime(args.device)
    idle = {"memory_stats": summarize_ns(_timed_reads(runtime.memory_stats, args.reads)),
            "memory_allocated": summarize_ns(_timed_reads(runtime.memory_allocated, args.reads))}
    load = under_load(runtime, args.boundaries, args.tile_rows)
    growth = args.max_boundary_growth_mib if args.max_boundary_growth_mib is not None else load["max_boundary_growth_mib"]
    pick = pick_interval(load["read"]["p99_ns"], BOUNDARIES_PER_FORWARD, args.forward_ms, READ_BUDGET, args.margin_mib, growth)
    return {"device": args.device, "card_verdict": args.device == "hpu", "pass": cell_passes(args.device, pick),
            "growth_source": "census" if args.max_boundary_growth_mib is not None else "synthetic tile",
            "idle": idle, "under_load": load, "pick": pick}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--device", choices=("hpu", "cpu"), required=True)
    parser.add_argument("--reads", type=int, default=20_000)
    parser.add_argument("--boundaries", type=int, default=BOUNDARIES_PER_FORWARD)
    parser.add_argument("--tile-rows", type=int, default=2048)
    parser.add_argument("--forward-ms", type=float, default=FORWARD_MS)
    parser.add_argument("--margin-mib", type=int, default=DEFAULT_MARGIN_MIB)
    parser.add_argument("--max-boundary-growth-mib", type=float)
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args(argv)


def main() -> int:
    args = parse_args()
    result = run(args)
    line = json.dumps(result, sort_keys=True)
    print(f"READ_COST_CELL {line}", flush=True)
    args.out.write_text(line + "\n")
    return 0 if result["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
