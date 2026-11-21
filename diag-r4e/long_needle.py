"""Ground-truth long needles (coordinator 2026-11-08 22:4x): graded's needle tasks at the long lengths, scored right/wrong against the known
answer. There is no paired reference (the eager anchor has no needle above 8,205 tokens), so the result is a pass rate per length, not a verdict.

usage: long_needle.py select TASKS_JSON OUT_JSON --lengths 33k,150k     write the needle tasks of those lengths (same file shape as TASKS_JSON)
       long_needle.py score LABEL=RESULTS_JSON [LABEL=RESULTS_JSON ...] --tasks OUT_JSON
                                                                         one LONG_NEEDLE line per label and one for all labels together
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

LENGTH = re.compile(r"^needle-(\d+k)-")


def task_list(document: object) -> list[dict]:
    """graded's tasks file is a list of tasks or an object with a "tasks" list."""
    if isinstance(document, dict):
        return document["tasks"]
    return document  # type: ignore[return-value]


def length_of(task_id: str) -> str | None:
    match = LENGTH.match(task_id)
    return match.group(1) if match else None


def select(tasks_path: Path, out_path: Path, lengths: list[str], limit: int = 0) -> int:
    document = json.loads(tasks_path.read_text(encoding="utf-8"))
    chosen = [task for task in task_list(document) if task.get("category") == "needle" and length_of(task["id"]) in lengths]
    chosen = chosen[:limit] if limit > 0 else chosen  # --limit 1: the single-task proof (sweep rule M2)
    if not chosen:
        print(f"LONG_NEEDLE select: no needle task of lengths {lengths} in {tasks_path}", file=sys.stderr)
        return 1
    shaped = {**document, "tasks": chosen} if isinstance(document, dict) else chosen
    out_path.write_text(json.dumps(shaped, indent=1), encoding="utf-8")
    counts = {length: sum(1 for task in chosen if length_of(task["id"]) == length) for length in lengths}
    print(f"LONG_NEEDLE select: {len(chosen)} tasks {counts} -> {out_path}")
    return 0


def tally(results: dict, ids: list[str]) -> dict[str, list[int]]:
    """{length: [passed, finished, total]} over the selected ids; a task counts as finished once its greedy reply is graded."""
    table: dict[str, list[int]] = {}
    for task_id in ids:
        row = table.setdefault(length_of(task_id) or "?", [0, 0, 0])
        row[2] += 1
        greedy = (results.get(task_id) or {}).get("greedy") or {}
        if "passed" in greedy:
            row[1] += 1
            row[0] += bool(greedy["passed"])
    return table


def line(label: str, table: dict[str, list[int]]) -> str:
    cells = " ".join(f"{length}={p}/{f} (of {t})" for length, (p, f, t) in sorted(table.items(), key=lambda kv: int(kv[0][:-1])))
    return f"LONG_NEEDLE {label} greedy passed/finished: {cells}"


def score(pairs: list[str], tasks_path: Path) -> int:
    ids = [task["id"] for task in task_list(json.loads(tasks_path.read_text(encoding="utf-8")))]
    combined: dict[str, list[int]] = {}
    for pair in pairs:
        label, _, path = pair.partition("=")
        results_file = Path(path)
        results = json.loads(results_file.read_text(encoding="utf-8")).get("results", {}) if results_file.is_file() else {}
        table = tally(results, ids)
        print(line(label, table))
        for length, counts in table.items():
            merged = combined.setdefault(length, [0, 0, 0])
            for index in range(3):
                merged[index] += counts[index]
    print(line("all", combined))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    chooser = commands.add_parser("select")
    chooser.add_argument("tasks", type=Path)
    chooser.add_argument("out", type=Path)
    chooser.add_argument("--lengths", default="33k,150k")
    chooser.add_argument("--limit", type=int, default=0, help="keep only the first N selected tasks (0 = all)")
    scorer = commands.add_parser("score")
    scorer.add_argument("pairs", nargs="+", help="LABEL=RESULTS_JSON")
    scorer.add_argument("--tasks", required=True, type=Path)
    args = parser.parse_args(argv)
    if args.command == "select":
        return select(args.tasks, args.out, [length.strip() for length in args.lengths.split(",") if length.strip()], args.limit)
    return score(args.pairs, args.tasks)


if __name__ == "__main__":
    sys.exit(main())
