#!/usr/bin/env python3
"""Do two C1 runs' lanes produce the same outputs? Compares the `output_sha256` list of every lanes x width row (cells/c1_cell.py drive_lanes).

usage: lane_hash_compare.py <run-a> <run-b> [--lanes 4,8,16] [--widths 1024,4096,8192]
A run is its `<name>.drive.jsonl` or the run folder ($L/c1-out/c1-<stamp>/) holding exactly one. The default grid is every lane count at every width.

Per cell: IDENTICAL (the two runs' lane hashes are the same multiset, lane order ignored), DIFFERENT, or NOT_MEASURED (the cell has no row, an error row, no
hashes, a hash count other than the lane count, a second row, or the two runs' anchor_tokens differ). Overall DIFFERENT (exit 1) outranks NOT_MEASURED
(exit 2); IDENTICAL (exit 0) needs every cell of the grid identical. The last line is LANE_HASH_COMPARE <status> <summary>.

Same prompts across runs: every lane of a cell sends k1_cell.fixed_prompt(anchor_tokens), whose random stream is seeded with the constant
k1_cell.PROMPT_SEED, so two runs of one build send the same token ids; each row records anchor_tokens, and a cell whose runs disagree on it is not compared.
What differs between runs is the cache salt (claude-c1-<stamp>-long-N), which only keys the prefix cache and does not enter the model.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import c1_score as scorer

IDENTICAL, DIFFERENT, NOT_MEASURED = 'IDENTICAL', 'DIFFERENT', 'NOT_MEASURED'
EXIT_CODES = {IDENTICAL: 0, DIFFERENT: 1, NOT_MEASURED: 2}
DRIVE_SUFFIX = '.drive.jsonl'
RUN_NAMES = ('first', 'second')
Cell = tuple[int, int]  # (lanes, width)


class RunUnreadable(ValueError):
    """A run path that holds no readable drive rows; the message says which path and what to give instead."""


@dataclass(frozen=True)
class Verdict:
    status: str
    summary: str
    lines: tuple[str, ...]

    @property
    def exit_code(self) -> int:
        return EXIT_CODES[self.status]


def _drive_file(path: Path) -> Path:
    if path.is_file():
        return path
    if not path.is_dir():
        raise RunUnreadable(f'{path} is neither a drive file nor a run folder: give the *{DRIVE_SUFFIX} file or the run folder')
    found = sorted(path.glob(f'*{DRIVE_SUFFIX}'))
    if not found:
        raise RunUnreadable(f'{path} holds no *{DRIVE_SUFFIX} file: the run never reached its drive steps, or this is not the run folder')
    if len(found) > 1:
        raise RunUnreadable(f'{path} holds {len(found)} *{DRIVE_SUFFIX} files ({", ".join(file.name for file in found)}), and a run has one: name the file')
    return found[0]


def _parse_line(file: Path, number: int, line: str) -> dict:
    try:
        row = json.loads(line)
    except json.JSONDecodeError as error:
        raise RunUnreadable(f'{file} line {number}: not JSON ({error}); the drive file is cut off or edited') from error
    if not isinstance(row, dict):
        raise RunUnreadable(f'{file} line {number}: a drive row is a JSON object, got {type(row).__name__}')
    return row


def read_rows(path: Path) -> list[dict]:
    """Every drive row of one run (a drive file, or the run folder holding one)."""
    file = _drive_file(path)
    try:
        lines = file.read_text(encoding='utf-8').splitlines()
    except OSError as error:
        raise RunUnreadable(f'cannot read {file}: {error}') from error
    return [_parse_line(file, number, line) for number, line in enumerate(lines, 1) if line.strip()]


def _lane_cells(rows: Iterable[Mapping]) -> dict[Cell, list[Mapping]]:
    cells: dict[Cell, list[Mapping]] = defaultdict(list)
    for row in rows:
        if row.get('drive') == 'lanes' and 'lanes' in row and 'width' in row:
            cells[(row['lanes'], row['width'])].append(row)
    return cells


def _row_problem(rows: Sequence[Mapping], lanes: int) -> str | None:
    """Why this run's rows for one cell cannot be compared, or None."""
    if not rows:
        return 'no lanes row for this cell (the drive stopped before it, or never ran)'
    if len(rows) > 1:
        return 'logged twice or more, so two runs share this file'
    row = rows[0]
    if row.get('status') == 'error':
        return f'status error ({row.get("reason", "no reason given")})'
    hashes = row.get('output_sha256')
    if hashes is None:
        return 'no output_sha256 on the row (a run from before the cell kept them)'
    if len(hashes) != lanes:
        return f'{len(hashes)} hashes for {lanes} lanes'
    return None


def _problem_text(problems: Sequence[str | None]) -> str:
    if problems[0] == problems[1]:
        return f'first and second run: {problems[0]}'
    return '; '.join(f'{name} run: {problem}' for name, problem in zip(RUN_NAMES, problems, strict=True) if problem)


def _prompt_problem(first: Mapping, second: Mapping) -> str | None:
    tokens = first.get('anchor_tokens'), second.get('anchor_tokens')
    if None in tokens or tokens[0] == tokens[1]:
        return None
    return f'the runs sent different prompts: anchor_tokens {tokens[0]} and {tokens[1]}'


def _judge_cell(cell: Cell, runs: Sequence[Mapping[Cell, list[Mapping]]]) -> tuple[str, str]:
    """(status, detail) of one lanes x width cell across the two runs."""
    found = [run.get(cell, []) for run in runs]
    problems = [_row_problem(rows, cell[0]) for rows in found]
    if any(problems):
        return NOT_MEASURED, _problem_text(problems)
    first, second = found[0][0], found[1][0]
    prompts = _prompt_problem(first, second)
    if prompts:
        return NOT_MEASURED, prompts
    left, right = Counter(first['output_sha256']), Counter(second['output_sha256'])
    detail = f'{sum(left.values())} lanes, {len(left)} and {len(right)} distinct outputs'
    if left == right:
        return IDENTICAL, detail
    return DIFFERENT, f'{detail}, {sum((left - right).values())} lane outputs of the first run have no match in the second'


def judge(first: Sequence[Mapping], second: Sequence[Mapping], lanes: Sequence[int] = scorer.LANE_COUNTS,
          widths: Sequence[int] = scorer.DECODE_WIDTHS) -> Verdict:
    runs = (_lane_cells(first), _lane_cells(second))
    grid = [(count, width) for width in widths for count in lanes]
    results = [(cell, *_judge_cell(cell, runs)) for cell in grid]
    counts = Counter(status for _, status, _ in results)
    status = DIFFERENT if counts[DIFFERENT] else NOT_MEASURED if counts[NOT_MEASURED] else IDENTICAL
    lines = tuple(f'LANE_HASH lanes={cell[0]} width={cell[1]} {cell_status} {detail}' for cell, cell_status, detail in results)
    summary = f'{counts[IDENTICAL]} of {len(grid)} cells identical, {counts[DIFFERENT]} different, {counts[NOT_MEASURED]} not measured'
    return Verdict(status, summary, lines)


def _int_list(text: str) -> tuple[int, ...]:
    try:
        values = tuple(int(part) for part in text.split(',') if part)
    except ValueError:
        values = ()
    if not values:
        raise argparse.ArgumentTypeError(f'{text!r} must be comma separated whole numbers such as 4,8,16')
    return values


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('first', type=Path, help='the first run (flag off): its drive file or run folder')
    parser.add_argument('second', type=Path, help='the second run (flag on): its drive file or run folder')
    parser.add_argument('--lanes', type=_int_list, default=scorer.LANE_COUNTS, help='lane counts to compare (default every one)')
    parser.add_argument('--widths', type=_int_list, default=scorer.DECODE_WIDTHS, help='table widths to compare (default every one)')
    args = parser.parse_args(argv)
    try:
        verdict = judge(read_rows(args.first), read_rows(args.second), args.lanes, args.widths)
    except RunUnreadable as error:
        print(f'LANE_HASH_COMPARE {NOT_MEASURED} {error}')
        return EXIT_CODES[NOT_MEASURED]
    print('\n'.join(verdict.lines))
    print(f'LANE_HASH_COMPARE {verdict.status} {verdict.summary}')
    return verdict.exit_code


if __name__ == '__main__':
    raise SystemExit(main())
