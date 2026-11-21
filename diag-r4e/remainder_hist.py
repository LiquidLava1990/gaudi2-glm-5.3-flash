"""remainder_hist: the r4e run's prompt-piece remainder census, which picks 256 vs 384 for r4f (MIXED_STEP_R4E_PLAN.md section 8).

usage: remainder_hist.py CONTAINER_LOG [--turns TURNS_JSONL] [--scope LABEL]

Every engine step's natural prompt chunks (GLM53_STEP_STATS step "prompts", read by diag-cells/read_cells.py). A chunk's remainder is
chunk % 1024; a multiple of 1024 is whole 1024-row pieces and has none. Bins: 1-128 / 129-256 / 257-384 / 385-1023. --turns keeps the
steps inside that run's measured window (the medium profile); without it every step in the log counts.
Rule: when more than 35% of the remainders fall in 257-384, the 384 line (mixed-step/key384/key384.env) replaces 256 in r4f; else 256 stays.
Exit: 0 decided, 1 no remainders (UNPROVEN), 2 the log or turns file is missing.
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

L = Path.home() / ".codex/tmp/flash-spikes-20261104/team-20261106-dual"
sys.path[:0] = [str(L / "diag-cells")]

PIECE_ROWS = 1024
SHARE_FOR_384 = 0.35
BINS = (("1-128", 1, 128), ("129-256", 129, 256), ("257-384", 257, 384), ("385-1023", 385, 1023))


@dataclass
class Census:
    whole: int = 0
    bins: dict[str, int] = field(default_factory=lambda: {name: 0 for name, _, _ in BINS})

    @property
    def remainders(self) -> int:
        return sum(self.bins.values())

    def share_384(self) -> float:
        return self.bins["257-384"] / self.remainders if self.remainders else float("nan")

    def decision(self) -> str:
        if not self.remainders:
            return "UNPROVEN"
        return "384" if self.share_384() > SHARE_FOR_384 else "256"

    def line(self, scope: str) -> str:
        counts = " ".join(f"{name}={count}" for name, count in self.bins.items())
        return (f"REMAINDER_HIST scope={scope} remainders={self.remainders} whole={self.whole} {counts} "
                f"share_257_384={self.share_384():.1%} rule=>{SHARE_FOR_384:.0%} r4f_key={self.decision()}")


def census(chunks: list[int]) -> Census:
    result = Census()
    for chunk in chunks:
        remainder = chunk % PIECE_ROWS
        if remainder == 0:
            result.whole += 1
            continue
        name = next(name for name, low, high in BINS if low <= remainder <= high)
        result.bins[name] += 1
    return result


def measure_window(turns: Path) -> tuple[float, float]:
    rows = [json.loads(line) for line in turns.read_text().splitlines() if line.strip()]
    rows = [row for row in rows if row.get("phase") != "seed" and row.get("started_at")]
    if not rows:
        return 0.0, float("inf")
    return min(row["started_at"] for row in rows), max(row["started_at"] + (row.get("wall_s") or 0) for row in rows)


def log_chunks(log: Path, start: float, end: float) -> list[int]:
    import read_cells  # noqa: PLC0415  (host module: imported only when a real log is read)
    steps = [step for step in read_cells.Log(log).engine if start <= step["t"] <= end]
    return [int(chunk) for step in steps for chunk, _ in (step.get("prompts") or [])]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("log", type=Path)
    parser.add_argument("--turns", type=Path)
    parser.add_argument("--scope", default="")
    args = parser.parse_args(argv)
    for path in (args.log, args.turns):
        if path is not None and not path.is_file():
            print(f"REMAINDER_HIST missing {path}: pass the run's container.log (and its medium/turns.jsonl for --turns)")
            return 2
    start, end = measure_window(args.turns) if args.turns else (0.0, float("inf"))
    result = census(log_chunks(args.log, start, end))
    print(result.line(args.scope or ("measure" if args.turns else "log")))
    return 1 if result.decision() == "UNPROVEN" else 0


if __name__ == "__main__":
    sys.exit(main())
