#!/usr/bin/env python3
"""Per-rank, per-key table of one K1 measurement boot, for any key set and rank count (k1_score.py is fixed to the 1024-row keys).

usage: k1_table.py <container.log> [--decode-mib 1026] [--margin-mib 512]
Prints r, capture peak, t and warmed ms per key, merge and self-compare counts by start, and rank fit:
sum(r) + max(t) + decode + margin against the free pool the first admission line reported.
"""
from __future__ import annotations

import argparse
import re
from collections import Counter
from pathlib import Path

import k1_score as scorer

MIB = 1 << 20
RANK = re.compile(r"Worker_TP(\d+)")
FREE = re.compile(r"GLM53_MIXED_GRAPH admitted .*free_bytes=(\d+)")
COMPARE = re.compile(r"(GLM53_K1_MERGE|GLM53_K1_SELF)\b.*?(?:width=(\d+) )?start=(\d+) torch_equal=([01])")


def first_free(lines: list[str]) -> dict[int, int]:
    """{rank: free MiB at its first key admission}."""
    free: dict[int, int] = {}
    for line in lines:
        rank, match = RANK.search(line), FREE.search(line)
        if rank and match:
            free.setdefault(int(rank.group(1)), int(match.group(1)) // MIB)
    return free


def compare_counts(lines: list[str]) -> Counter[tuple[str, str, int, int]]:
    """{(tag, width, start, equal): lines over all ranks}."""
    counts: Counter[tuple[str, str, int, int]] = Counter()
    for line in lines:
        match = COMPARE.search(line)
        if match:
            counts[(match.group(1), match.group(2) or "-", int(match.group(3)), int(match.group(4)))] += 1
    return counts


def print_ranks(parsed: scorer.ParsedLog, free: dict[int, int], extra_mib: int) -> None:
    print("rank key            r     peak      t       ms replays timed  devmem")
    for rank in sorted(parsed.ranks):
        data = parsed.ranks[rank]
        resident, transient = 0, 0
        for key in sorted(data.keys, key=lambda item: (-item[0], -item[1])):
            row = scorer.summarize_key(data.keys[key])
            print(f"{rank:>4} {key[0]}:{key[1]:<8} {row.resident_mib!s:>6} {row.capture_peak_mib!s:>6} {row.transient_mib!s:>6} "
                  f"{row.replay_ms!s:>8} {row.replays:>7} {row.samples:>5} {data.devmem:>7}")
            resident += row.resident_mib or 0
            transient = max(transient, row.transient_mib or 0)
        need = resident + transient + extra_mib
        pool = free.get(rank)
        verdict = "-" if pool is None else ("FITS" if need <= pool else "OVER")
        print(f"     sum(r)+t+{extra_mib}={need} free={pool} {verdict} margin={None if pool is None else pool - need}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("log", type=Path)
    parser.add_argument("--decode-mib", type=int, default=1026)
    parser.add_argument("--margin-mib", type=int, default=512)
    args = parser.parse_args()
    lines = args.log.read_text(errors="replace").splitlines()
    print_ranks(scorer.parse_log(lines), first_free(lines), args.decode_mib + args.margin_mib)
    for (tag, width, start, equal), count in sorted(compare_counts(lines).items()):
        print(f"{tag} width={width} start={start} torch_equal={equal} lines={count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
