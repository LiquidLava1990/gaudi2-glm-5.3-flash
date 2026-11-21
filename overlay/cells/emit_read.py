#!/usr/bin/env python3
"""The verdict on one HPU_GLM53_EMIT_WORDSKIP run (WIDE-PROMPT-KERNEL-PLAN step 6), read from the container log and, for a traced run, the cell log.

usage: emit_read.py <container.log> --tp-size 4 [--cell-log <cell.log>] [--expect on|off]

--expect on (default), the word-skip run. PASS (exit 0) only when the container log holds exactly --tp-size FLASH_EMIT_WORDSKIP_LIVE lines, one per rank
0..tp-size-1, each naming g6_topk_emit_wordskip_i32, and no 'HPU_GLM53_EMIT_WORDSKIP=1 cannot boot' refusal. With --cell-log, every prompt run's
REPLAY_TRACE_KERNEL table must also name g6_topk_emit_wordskip_i32 and its REPLAY_TRACE_SCAN anchor must include it: a trace that shows only the stock
emit means the flag did nothing, whatever the live lines say. A KERNEL table at the reporter's row limit may leave the emit below the cut, so it fails only when it
is complete or names the stock emit; the anchor is never cut off.
--expect off, the control (flag unset): no live line, no refusal, and with --cell-log the word-skip GUID in no prompt run's KERNEL table or SCAN anchor.
FAIL (exit 1): the log contradicts the run: a refusal, a live line of another GUID, a count other than --tp-size (the flag reached some ranks only, or none),
a repeated or missing rank, or a trace table that lacks (on) or holds (off) the word-skip GUID.
NOT_MEASURED (exit 2): nothing contradicts the run but it proves nothing: an unreadable or empty log, or --cell-log with no prompt run table to read.
"""
from __future__ import annotations

import argparse
import re
from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

PASS, FAIL, NOT_MEASURED = 'PASS', 'FAIL', 'NOT_MEASURED'
EXIT_CODES = {PASS: 0, FAIL: 1, NOT_MEASURED: 2}
ON, OFF = 'on', 'off'
WORDSKIP_GUID = 'g6_topk_emit_wordskip_i32'
STOCK_GUID = 'g6_topk_emit_i32'
PROMPT_RUN_PREFIX = 'prompt'  # run ids look like prompt-1024x8192-n3; decode runs take no word-skip emit and are not read

LIVE = re.compile(r'FLASH_EMIT_WORDSKIP_LIVE rank=(?P<rank>-?\d+) rows=(?P<rows>\d+) width=(?P<width>\d+) guid=(?P<guid>\S+)')
REFUSAL = re.compile(r'HPU_GLM53_EMIT_WORDSKIP=1 cannot boot: (?P<why>.*)')
KERNEL_HEAD = re.compile(r'REPLAY_TRACE_KERNEL run=(?P<run>\S+) ')
KERNEL_ROW = re.compile(r'^\s+\d+\.\d+\s+\d+\.\d+\s+\d+\s+(?P<name>\S+)\s*$')  # share busy_ms events name
KERNEL_TABLE_LIMIT = 20  # replay_trace_report.KERNEL_LINES (its top= field is the rows printed, so it cannot show a cut); a test pins the two together
SCAN = re.compile(r'REPLAY_TRACE_SCAN run=(?P<run>\S+) anchor=(?P<anchor>\S+)')


@dataclass(frozen=True)
class Verdict:
    status: str
    summary: str
    reasons: tuple[str, ...]

    @property
    def exit_code(self) -> int:
        return EXIT_CODES[self.status]


@dataclass
class RunTables:
    """What one traced run's report printed: the op names of its KERNEL table and its SCAN anchor (None when absent)."""

    kernels: set[str] | None = None
    anchor: str | None = None

    def kernels_support_word_skip(self, expect: str) -> bool | None:
        """Whether the KERNEL table agrees the word-skip emit ran. On a run with the flag on, a table cut off at its limit says nothing unless the stock emit made the cut."""
        if self.kernels is None:
            return None
        if WORDSKIP_GUID in self.kernels:
            return True
        return expect == ON and len(self.kernels) >= KERNEL_TABLE_LIMIT and STOCK_GUID not in self.kernels


def parse_trace(lines: Iterable[str]) -> dict[str, RunTables]:
    runs: dict[str, RunTables] = {}
    rows: set[str] | None = None  # the open KERNEL table's names, while its indented rows are being read
    for line in lines:
        head, scan = KERNEL_HEAD.search(line), SCAN.search(line)
        if head:
            tables = runs.setdefault(head['run'], RunTables())
            tables.kernels = rows = tables.kernels or set()
        elif scan:
            runs.setdefault(scan['run'], RunTables()).anchor, rows = scan['anchor'], None
        elif rows is not None and (row := KERNEL_ROW.match(line)):
            rows.add(row['name'])
        else:
            rows = None
    return runs


def _live_findings(lines: Sequence[str], tp_size: int, expect: str) -> list[str]:
    lives = [found for found in map(LIVE.search, lines) if found]
    if expect == OFF:
        return [f'{len(lives)} FLASH_EMIT_WORDSKIP_LIVE lines in a run with the flag unset'] if lives else []
    problems = [f'rank {found["rank"]} logs guid={found["guid"]}, not {WORDSKIP_GUID}' for found in lives if found['guid'] != WORDSKIP_GUID]
    if len(lives) != tp_size:
        problems.append(f'{len(lives)} FLASH_EMIT_WORDSKIP_LIVE lines, expected one per rank: {tp_size} (--tp-size)')
    seen = Counter(int(found['rank']) for found in lives)
    problems += [f'rank {rank} logs {count} live lines' for rank, count in sorted(seen.items()) if count > 1]
    problems += [f'rank {rank} has no live line' for rank in range(tp_size) if lives and rank not in seen]
    problems += [f'rank {rank} is not a worker rank of a TP{tp_size} run' for rank in sorted(seen) if not 0 <= rank < tp_size]
    return problems


def _trace_findings(runs: dict[str, RunTables], expect: str) -> tuple[list[str], list[str]]:
    """(contradictions, gaps) in the prompt runs' KERNEL tables and SCAN anchors."""
    prompts = {run: tables for run, tables in sorted(runs.items()) if run.startswith(PROMPT_RUN_PREFIX)}
    if not prompts:
        return [], ['--cell-log has no prompt run in its REPLAY_TRACE tables: nothing to read the emit GUID from']
    failures: list[str] = []
    gaps: list[str] = []
    for run, tables in prompts.items():
        for label, present in (('REPLAY_TRACE_KERNEL table', tables.kernels_support_word_skip(expect)),
                               ('REPLAY_TRACE_SCAN anchor', None if tables.anchor is None else WORDSKIP_GUID in tables.anchor.split('+'))):
            if present is None:
                gaps.append(f'{run}: no {label} was printed')
            elif present != (expect == ON):
                failures.append(f'{run}: the {label} {"does not name" if expect == ON else "names"} {WORDSKIP_GUID}')
    return failures, gaps


def judge(container: Sequence[str], tp_size: int, expect: str = ON, cell: Sequence[str] | None = None) -> Verdict:
    if not any(line.strip() for line in container):
        return Verdict(NOT_MEASURED, 'empty container log', ('the container log has no lines: the cell never booted, or this is the wrong file',))
    failures = [f'boot refused: {found["why"][:300]}' for found in map(REFUSAL.search, container) if found]
    failures += _live_findings(container, tp_size, expect)
    gaps: list[str] = []
    if cell is not None:
        trace_failures, gaps = _trace_findings(parse_trace(cell), expect)
        failures += trace_failures
    summary = f'expect {expect}, tp_size={tp_size}, {sum(bool(LIVE.search(line)) for line in container)} live lines' + (', trace read' if cell is not None else '')
    if failures:
        return Verdict(FAIL, summary, tuple(failures + gaps))
    if gaps:
        return Verdict(NOT_MEASURED, summary, tuple(gaps))
    return Verdict(PASS, summary, ())


def _read(path: Path) -> list[str] | None:
    try:
        return path.read_text(encoding='utf-8', errors='replace').splitlines()
    except OSError as error:
        print(f'{NOT_MEASURED}: cannot read {path}: {error}')
        return None


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('log', type=Path, help='the cell\'s container.log')
    parser.add_argument('--tp-size', type=int, required=True, help='worker ranks of the run (one boot of TP4 is 4); the live-line count must equal it')
    parser.add_argument('--cell-log', type=Path, default=None, help='the cell.log holding the REPLAY_TRACE_* tables (a c1p run)')
    parser.add_argument('--expect', choices=(ON, OFF), default=ON, help='on: the word-skip run (default); off: the flag-unset control')
    args = parser.parse_args(argv)
    if args.tp_size < 1:
        parser.error('--tp-size must be at least 1')
    container = _read(args.log)
    cell = _read(args.cell_log) if args.cell_log else None
    if container is None or (args.cell_log and cell is None):
        return EXIT_CODES[NOT_MEASURED]
    verdict = judge(container, args.tp_size, args.expect, cell)
    print(f'EMIT_READ {verdict.status}: {verdict.summary}')
    for reason in verdict.reasons:
        print(f'  - {reason}')
    return verdict.exit_code


if __name__ == '__main__':
    raise SystemExit(main())
