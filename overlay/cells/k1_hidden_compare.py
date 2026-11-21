#!/usr/bin/env python3
"""Cross-boot width merge: compares the first hidden outputs two K1 boots saved, by rank and chunk start. Run in the image (it has torch).

usage: k1_hidden_compare.py <trace-dir-a> <trace-dir-b>
Files are hidden-r<rank>-<rows>-<width>-<start>.pt. Prints one line per rank and start, and a total. Exit 0 only when every pair is equal.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import torch

NAME = re.compile(r"hidden-r(\d+)-(\d+)-(\d+)-(\d+)\.pt$")


def saved(directory: Path) -> dict[tuple[int, int], tuple[int, Path]]:
    """{(rank, start): (width, path)}."""
    found: dict[tuple[int, int], tuple[int, Path]] = {}
    for path in directory.glob("hidden-*.pt"):
        match = NAME.search(path.name)
        if match:
            found[(int(match.group(1)), int(match.group(4)))] = (int(match.group(3)), path)
    return found


def main() -> int:
    first, second = saved(Path(sys.argv[1])), saved(Path(sys.argv[2]))
    pairs = sorted(set(first) & set(second))
    equal = 0
    for rank, start in pairs:
        left, right = torch.load(first[(rank, start)][1], weights_only=True), torch.load(second[(rank, start)][1], weights_only=True)
        same = torch.equal(left, right)
        equal += same
        difference = "-" if same or left.shape != right.shape else f"{(left - right).abs().max().item():.3e}"
        print(f"rank={rank} start={start} widths={first[(rank, start)][0]},{second[(rank, start)][0]} equal={int(same)} max_abs_diff={difference}")
    print(f"TOTAL pairs={len(pairs)} equal={equal} only_a={len(set(first) - set(second))} only_b={len(set(second) - set(first))}")
    return 0 if pairs and equal == len(pairs) else 1


if __name__ == "__main__":
    raise SystemExit(main())
