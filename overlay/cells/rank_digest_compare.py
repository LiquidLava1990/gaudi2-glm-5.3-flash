#!/usr/bin/env python3
"""Do two runs' indexer rows agree, layer by layer? Compares the rows_digest= tag of every GLM53_RANK_CHECK line (glm53_mixed_graph.RankCheck).

usage: rank_digest_compare.py <run-a> <run-b> [--key decode:16x8192 ...] [--decode-only]
A run is its container log or the run folder holding container.log. Run a is the unsharded arm, run b the sharded one, driven alike (the same lanes,
prompts and seed: cells/c1_cell.py); a run compared with a second run of itself is the null arm, which says whether the drive is repeatable at all.

A sharded run's own lines say equal=1 whatever the gather did (the ranks hold the gathered rows), so the tag is the evidence: one hash of the rows rank 0
read, in order. The k-th read of a key in run a is compared with the k-th read of the same key in run b (a read = the layers logged for one key at one
start and replay), layer by layer. IDENTICAL: every layer of every read has the same tag. DIFFERENT: some layer's tag differs; the first (earliest read,
lowest layer) is printed as RANK_DIGEST_FIRST_DIFFERENCE. NOT_MEASURED: a log without tags or lines, a key with another read count in each run, a read
whose start (the longest lane's position) or live_rows differ, or a layer with no line, two lines, a NOT_RETAINED or a LOG_FAILED in either run.
DIFFERENT (exit 1) outranks NOT_MEASURED (exit 2); IDENTICAL is exit 0. Eager lines (eager=1) are not compared. The last line is
RANK_DIGEST_COMPARE <status> <summary>.
"""
from __future__ import annotations

import argparse
import re
from collections import Counter, defaultdict
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import rank_check_read as reader

IDENTICAL, DIFFERENT, NOT_MEASURED = 'IDENTICAL', 'DIFFERENT', 'NOT_MEASURED'
EXIT_CODES = {IDENTICAL: 0, DIFFERENT: 1, NOT_MEASURED: 2}
LOG_NAME = 'container.log'
RUN_NAMES = ('first', 'second')
MAX_PROBLEMS_PER_READ = 3  # a read with a dozen voids prints the first ones and counts the rest
VERDICT, VOID, LOG_FAILED = 'verdict', 'void', 'log_failed'

TAGGED = re.compile(r'GLM53_RANK_CHECK key=(?P<key>\S+) layer=(?P<layer>\S+) equal=[01] rows=\d+ rows_differ=\d+ live_rows=(?P<live_rows>\d+) '
                    r'ranks=\d+ start=(?P<start>\d+) replay=(?P<replay>\d+)(?: rows_digest=(?P<tag>[0-9a-f]+))?')


class RunUnreadable(ValueError):
    """A run path that holds no readable log; the message says which path and what to give instead."""


@dataclass(frozen=True)
class Verdict:
    status: str
    summary: str
    lines: tuple[str, ...]

    @property
    def exit_code(self) -> int:
        return EXIT_CODES[self.status]


@dataclass(frozen=True)
class Entry:
    """One log line about one layer of a read: a verdict (tag None on a line from before the tag), a void (NOT_RETAINED) or a LOG_FAILED."""

    kind: str
    tag: str | None = None
    live_rows: int | None = None
    reason: str = ''


@dataclass
class Read:
    """The lines logged for one key at one start and replay: every layer once, in a clean run."""

    start: int
    order: int  # the line number of its first line, to put reads of different keys in the order the run took them
    layers: dict[str, list[Entry]] = field(default_factory=lambda: defaultdict(list))


@dataclass
class Log:
    reads: dict[str, list[Read]] = field(default_factory=lambda: defaultdict(list))

    def entries(self) -> Iterator[Entry]:
        return (entry for reads in self.reads.values() for read in reads for found in read.layers.values() for entry in found)


@dataclass(frozen=True)
class ReadResult:
    key: str
    number: int
    start: int
    order: int
    layers: int
    status: str
    differing: tuple[tuple[str, str, str], ...] = ()  # (layer, first run's tag, second run's tag), lowest layer first
    problems: tuple[str, ...] = ()


def _entry(text: str) -> tuple[str, int, int, str, Entry] | None:
    """(key, start, replay, layer, entry) of a line, or None for any other line (eager lines included: they have no start)."""
    tagged = TAGGED.search(text)
    if tagged:
        found = tagged.groupdict()
        return found['key'], int(found['start']), int(found['replay']), found['layer'], Entry(VERDICT, found['tag'], int(found['live_rows']))
    for kind, pattern in ((VOID, reader.VOID), (LOG_FAILED, reader.LOG_FAILED)):
        found = pattern.search(text)
        if found and found['start'] is not None:
            return found['key'], int(found['start']), int(found['replay']), found['layer'], Entry(kind, reason=found['reason'])
    return None


def parse_log(lines: Iterable[str]) -> Log:
    log = Log()
    chunks: dict[tuple[str, int, int], Read] = {}
    for number, text in enumerate(lines):
        parsed = _entry(text)
        if parsed is None:
            continue
        key, start, replay, layer, entry = parsed
        read = chunks.get((key, start, replay))
        if read is None:
            read = chunks[(key, start, replay)] = Read(start, number)
            log.reads[key].append(read)
        read.layers[layer].append(entry)
    return log


def _layer_order(layer: str) -> tuple[int, str]:
    return (int(layer), layer) if layer.isdigit() else (1 << 30, layer)


def _log_problems(log: Log, name: str) -> list[str]:
    entries = list(log.entries())
    if not entries:
        return [f'no GLM53_RANK_CHECK lines in the {name} run: the check never ran, or its lines are in another log']
    verdicts = [entry for entry in entries if entry.kind == VERDICT]
    if verdicts and not any(entry.tag for entry in verdicts):
        return [f'no rows_digest on any GLM53_RANK_CHECK line of the {name} run: a build from before the tag; rerun it with this overlay']
    return []


def _layer_problem(layer: str, found: Sequence[Sequence[Entry]]) -> str:
    """Why this layer of a read cannot be compared, from either run, or ''."""
    problems: list[str] = []
    for name, entries in zip(RUN_NAMES, found, strict=True):
        if not entries:
            problems.append(f'layer {layer} has no line in the {name} run')
        elif len(entries) > 1:
            problems.append(f'layer {layer} logged twice in the {name} run, so two runs share this log')
        elif entries[0].kind == VOID:
            problems.append(f'layer {layer} has no verdict in the {name} run (NOT_RETAINED: {entries[0].reason})')
        elif entries[0].kind == LOG_FAILED:
            problems.append(f'layer {layer} has no line in the {name} run (LOG_FAILED: {entries[0].reason})')
        elif entries[0].tag is None:
            problems.append(f'layer {layer} has no rows_digest in the {name} run')
    return '; '.join(problems)


def _compare_read(key: str, number: int, reads: Sequence[Read], layers: Sequence[str]) -> ReadResult:
    first, second = reads
    head = (key, number, first.start, first.order, len(layers))
    if first.start != second.start:
        return ReadResult(*head, NOT_MEASURED, problems=(f'start={first.start} in the first run, {second.start} in the second: the runs were not driven '
                                                         f'alike, so these reads hold other batches',))
    differing: list[tuple[str, str, str]] = []
    problems: list[str] = []
    for layer in layers:
        found = (first.layers.get(layer, []), second.layers.get(layer, []))
        problem = _layer_problem(layer, found)
        if problem:
            problems.append(problem)
        elif found[0][0].live_rows != found[1][0].live_rows:
            problems.append(f'layer {layer}: live_rows={found[0][0].live_rows} in the first run, {found[1][0].live_rows} in the second, so the batches differ')
        elif found[0][0].tag != found[1][0].tag:
            differing.append((layer, found[0][0].tag or '', found[1][0].tag or ''))
    status = DIFFERENT if differing else NOT_MEASURED if problems else IDENTICAL
    return ReadResult(*head, status, tuple(differing), tuple(problems))


def _selected_keys(logs: Sequence[Log], keys: Sequence[str], decode_only: bool) -> list[str]:
    if keys:
        return list(keys)
    present = dict.fromkeys(key for log in logs for key in log.reads)
    return [key for key in present if not decode_only or key.startswith(reader.DECODE_PREFIX)]


def _key_problem(key: str, counts: Sequence[int]) -> str:
    if counts == [0, 0]:
        return f'key {key} has no reads in either run'
    return f'key {key} has {counts[0]} reads in the first run, {counts[1]} in the second'


def _read_lines(result: ReadResult) -> list[str]:
    lines = [f'RANK_DIGEST_READ key={result.key} read={result.number} start={result.start} status={result.status} layers={result.layers} '
             f'layers_differ={len(result.differing)}']
    shown = result.problems[:MAX_PROBLEMS_PER_READ]
    lines += [f'RANK_DIGEST_PROBLEM key={result.key} read={result.number}: {problem}' for problem in shown]
    if len(result.problems) > len(shown):
        lines.append(f'RANK_DIGEST_PROBLEM key={result.key} read={result.number}: ... and {len(result.problems) - len(shown)} more')
    return lines


def _first_difference(results: Sequence[ReadResult]) -> list[str]:
    for result in results:
        if result.differing:
            layer, tag_a, tag_b = result.differing[0]
            return [f'RANK_DIGEST_FIRST_DIFFERENCE key={result.key} read={result.number} start={result.start} layer={layer} a={tag_a} b={tag_b}']
    return []


def _verdict(results: list[ReadResult], key_problems: list[str], layers: Sequence[str]) -> Verdict:
    counts = Counter(result.status for result in results)
    status = DIFFERENT if counts[DIFFERENT] else NOT_MEASURED if counts[NOT_MEASURED] or key_problems else IDENTICAL
    lines = [line for result in results for line in _read_lines(result)]
    lines += [f'RANK_DIGEST_PROBLEM {problem}' for problem in key_problems]
    lines += _first_difference(results)
    size = f'{len(results)} reads x {len(layers)} layers'
    if status == IDENTICAL:
        return Verdict(status, f'{size}, every rows_digest equal', tuple(lines))
    extra = f', {len(key_problems)} key problems' if key_problems else ''
    summary = f'{size}: {counts[IDENTICAL]} identical, {counts[DIFFERENT]} different, {counts[NOT_MEASURED]} not measured{extra}'
    return Verdict(status, summary, tuple(lines))


def judge(first: Iterable[str], second: Iterable[str], keys: Sequence[str] = (), decode_only: bool = False) -> Verdict:
    """Compare the two logs' rows_digest tags, read by read (see the module docstring); keys limits it to those keys, decode_only to decode keys."""
    logs = (parse_log(first), parse_log(second))
    unusable = [problem for log, name in zip(logs, RUN_NAMES, strict=True) for problem in _log_problems(log, name)]
    if unusable:
        return Verdict(NOT_MEASURED, 'a run has no usable rows_digest lines', tuple(f'RANK_DIGEST_PROBLEM {problem}' for problem in unusable))
    selected = _selected_keys(logs, keys, decode_only)
    if not selected:
        return Verdict(NOT_MEASURED, 'no reads to compare', ('RANK_DIGEST_PROBLEM no keys to compare: neither run has a decode key, or none was named',))
    layers = sorted({layer for log in logs for key in selected for read in log.reads.get(key, ()) for layer in read.layers}, key=_layer_order)
    results: list[ReadResult] = []
    key_problems: list[str] = []
    for key in selected:
        reads = [log.reads.get(key, []) for log in logs]
        if len(reads[0]) != len(reads[1]) or not reads[0]:
            key_problems.append(_key_problem(key, [len(found) for found in reads]))
        results += [_compare_read(key, number, pair, layers) for number, pair in enumerate(zip(*reads, strict=False), 1)]
    results.sort(key=lambda result: result.order)
    return _verdict(results, key_problems, layers)


def _log_file(path: Path) -> Path:
    if path.is_file():
        return path
    if not path.is_dir():
        raise RunUnreadable(f'{path} is neither a log file nor a run folder: give the container log or the run folder holding {LOG_NAME}')
    inside = path / LOG_NAME
    if not inside.is_file():
        raise RunUnreadable(f'{path} holds no {LOG_NAME}: the run never started a container, or give the log file itself')
    return inside


def read_log(path: Path) -> list[str]:
    file = _log_file(path)
    try:
        return file.read_text(encoding='utf-8', errors='replace').splitlines()
    except OSError as error:
        raise RunUnreadable(f'cannot read {file}: {error}') from error


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('first', type=Path, help='the first run (unsharded): its container log or run folder')
    parser.add_argument('second', type=Path, help='the second run (sharded, or the same arm again as the null arm): its container log or run folder')
    parser.add_argument('--key', action='append', default=[], metavar='KEY', help='compare only this key (the key= of a log line, e.g. decode:16x8192); repeatable')
    parser.add_argument('--decode-only', action='store_true', help='compare decode keys only (the prompt keys of a decode run are left out)')
    args = parser.parse_args(argv)
    try:
        verdict = judge(read_log(args.first), read_log(args.second), tuple(args.key), args.decode_only)
    except RunUnreadable as error:
        print(f'RANK_DIGEST_COMPARE {NOT_MEASURED} {error}')
        return EXIT_CODES[NOT_MEASURED]
    print('\n'.join(verdict.lines))
    print(f'RANK_DIGEST_COMPARE {verdict.status} {verdict.summary}')
    return verdict.exit_code


if __name__ == '__main__':
    raise SystemExit(main())
