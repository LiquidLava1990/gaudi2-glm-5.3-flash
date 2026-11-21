#!/usr/bin/env python3
"""The verdict on one card run's HPU_GLM53_INDEXER_RANK_CHECK=1 lines (glm53_mixed_graph.RankCheck), read from the container log.

usage: rank_check_read.py <container.log> [--ranks 4] [--layers N] [--min-start 150000 --min-start 530000] [--decode-key 8x4096 ...] [--decode-only]

PASS (exit 0) only when every chunk the log shows has every layer, each logged once with equal=1 and ranks=<ranks>, there is no NOT_RETAINED, no
LOG_FAILED and no eager=1 line, and some complete chunk starts at or past each --min-start (default 150000 and 530000; one chunk can satisfy several).
Decode keys log as key=decode:ROWSxWIDTH, whose start is the step's longest lane position, not a prompt's start: they are judged like any chunk, but
never count toward --min-start. Each --decode-key ROWSxWIDTH must have a complete equal=1 chunk, and --decode-only (a decode run, whose prompts are
not long) asks for no --min-start at all; prompt lines found in such a log are still judged.
FAIL (exit 1): a layer measured unequal, i.e. the ranks disagree on that chunk's rows. This outranks everything else in the log.
NOT_MEASURED (exit 2): nothing wrong was measured, but the run does not prove the ranks agree: a void (no verdict on a layer), a layer missing from
a chunk, ranks other than expected, an eager line, a long start never read, or no check lines at all. Rerun; the reasons say what to fix.
Layers expected per chunk: --layers, else the GLM53_RANK_CHECK_READY line's count, else the layers seen in the log (a layer missing from every
chunk is then invisible, so pass --layers when the log has no READY line).
"""
from __future__ import annotations

import argparse
import re
from collections import defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

PASS, FAIL, NOT_MEASURED = 'PASS', 'FAIL', 'NOT_MEASURED'
EXIT_CODES = {PASS: 0, FAIL: 1, NOT_MEASURED: 2}
DEFAULT_RANKS = 4
DEFAULT_MIN_STARTS = (150_000, 530_000)
DECODE_PREFIX = 'decode:'  # glm53_mixed_graph.DecodeRankRead's key label: decode:ROWSxWIDTH
DECODE_KEY = re.compile(r'\d+x\d+')
MAX_REASONS = 20  # a run of hundreds of voids prints the first ones and counts the rest

WHERE = r'(?:start=(?P<start>\d+) replay=(?P<replay>\d+)|(?P<eager>eager=1))'
VERDICT = re.compile(r'GLM53_RANK_CHECK key=(?P<key>\S+) layer=(?P<layer>\S+) equal=(?P<equal>[01]) rows=\d+ rows_differ=(?P<differ>\d+) '
                     r'live_rows=\d+ ranks=(?P<ranks>\d+) ' + WHERE)
VOID = re.compile(r'GLM53_RANK_CHECK_NOT_RETAINED key=(?P<key>\S+) layer=(?P<layer>\S+) ' + WHERE + r' reason=(?P<reason>.*?)\. No verdict')
LOG_FAILED = re.compile(r'GLM53_RANK_CHECK_LOG_FAILED key=(?P<key>\S+) layer=(?P<layer>\S+) ' + WHERE + r' error=(?P<reason>.*?)\. No line')
READY = re.compile(r'GLM53_RANK_CHECK_READY layers=(?P<layers>\d+)')


@dataclass(frozen=True)
class Reading:
    """One layer of one chunk as the log says it: kind is equal, unequal, void (NOT_RETAINED) or log_failed."""

    kind: str
    key: str
    layer: str
    start: int | None
    replay: int | None
    eager: bool
    ranks: int | None = None
    differ: int = 0
    reason: str = ''

    @property
    def chunk(self) -> tuple[str, int | None, int | None]:
        return self.key, self.start, self.replay

    @property
    def label(self) -> str:
        where = 'eager=1' if self.eager else f'start={self.start} replay={self.replay}'
        return f'key {self.key} layer {self.layer} {where}'


@dataclass(frozen=True)
class Verdict:
    status: str
    summary: str
    reasons: tuple[str, ...]

    @property
    def exit_code(self) -> int:
        return EXIT_CODES[self.status]


def _reading(kind: str, found: re.Match[str]) -> Reading:
    fields = found.groupdict()
    start, replay = fields.get('start'), fields.get('replay')
    return Reading(kind, fields['key'], fields['layer'], None if start is None else int(start), None if replay is None else int(replay),
                   fields['eager'] is not None, int(fields['ranks']) if fields.get('ranks') else None, int(fields.get('differ') or 0),
                   fields.get('reason') or '')


def parse(lines: Iterable[str]) -> tuple[list[Reading], int | None]:
    """Every layer reading in the log, and the layer count of the first GLM53_RANK_CHECK_READY line, if any."""
    readings: list[Reading] = []
    ready: int | None = None
    for line in lines:
        verdict = VERDICT.search(line)
        if verdict:
            readings.append(_reading('equal' if verdict['equal'] == '1' else 'unequal', verdict))
            continue
        for kind, pattern in (('void', VOID), ('log_failed', LOG_FAILED)):
            found = pattern.search(line)
            if found:
                readings.append(_reading(kind, found))
                break
        else:
            found = READY.search(line)
            if found and ready is None:
                ready = int(found['layers'])
    return readings, ready


def _line_findings(readings: list[Reading], ranks: int) -> tuple[list[str], list[str]]:
    """(layers measured unequal, reasons a line proves nothing), one entry per offending reading."""
    unequal: list[str] = []
    unmeasured: list[str] = []
    for reading in readings:
        if reading.kind == 'unequal':
            unequal.append(f'{reading.label}: equal=0 rows_differ={reading.differ}')
        elif reading.kind == 'void':
            unmeasured.append(f'{reading.label}: NOT_RETAINED, no verdict ({reading.reason})')
        elif reading.kind == 'log_failed':
            unmeasured.append(f'{reading.label}: LOG_FAILED, the line is missing ({reading.reason})')
        if reading.eager and reading.kind in ('equal', 'unequal'):
            unmeasured.append(f'{reading.label}: eager=1, the runner\'s own forward, which proves less than a replay')
        if reading.ranks is not None and reading.ranks != ranks:
            unmeasured.append(f'{reading.label}: ranks={reading.ranks}, expected {ranks}')
    return unequal, unmeasured


def _chunks(readings: list[Reading]) -> dict[tuple, dict[str, list[Reading]]]:
    chunks: dict[tuple, dict[str, list[Reading]]] = defaultdict(lambda: defaultdict(list))
    for reading in readings:
        if not reading.eager:
            chunks[reading.chunk][reading.layer].append(reading)
    return chunks


def _chunk_problems(chunk: tuple, by_layer: dict[str, list[Reading]], layers: int) -> list[str]:
    """Why this chunk is not a complete set of layers, each logged once."""
    key, start, replay = chunk
    name = f'chunk key {key} start={start} replay={replay}'
    problems = [f'{name}: layer {layer} logged twice, so two runs share this log' for layer, found in sorted(by_layer.items()) if len(found) > 1]
    measured = sum(any(r.kind in ('equal', 'unequal') for r in found) for found in by_layer.values())
    if measured != layers:
        problems.append(f'{name} has {measured} of {layers} layers with a verdict')
    return problems


def _is_clean(by_layer: dict[str, list[Reading]], layers: int, ranks: int) -> bool:
    return len(by_layer) == layers and all(len(found) == 1 and found[0].kind == 'equal' and found[0].ranks == ranks for found in by_layer.values())


def _coverage_problems(clean_starts: list[int], min_starts: Sequence[int]) -> list[str]:
    top = max(clean_starts, default=None)
    return [f'no complete equal=1 chunk at start >= {need} (highest complete start: {top})' for need in min_starts if top is None or top < need]


def _decode_coverage_problems(clean_decode: set[str], wanted: Sequence[str]) -> list[str]:
    seen = ', '.join(sorted(clean_decode)) or 'none'
    return [f'no complete equal=1 chunk of decode key {key} (decode keys with one: {seen})' for key in wanted if key not in clean_decode]


def _clean_chunks(chunks: dict[tuple, dict[str, list[Reading]]], layers: int, ranks: int) -> tuple[list[str], list[int], set[str]]:
    """(problems of incomplete chunks, prompt starts of complete equal=1 chunks, decode keys with such a chunk), chunks in start order."""
    problems: list[str] = []
    clean_starts: list[int] = []
    clean_decode: set[str] = set()
    for chunk, by_layer in sorted(chunks.items(), key=lambda item: (item[0][1] or 0, item[0][2] or 0)):
        found = _chunk_problems(chunk, by_layer, layers)
        problems += found
        if found or not _is_clean(by_layer, layers, ranks):
            continue
        if chunk[0].startswith(DECODE_PREFIX):
            clean_decode.add(chunk[0][len(DECODE_PREFIX):])
        else:
            clean_starts.append(chunk[1])
    return problems, clean_starts, clean_decode


def _capped(reasons: list[str]) -> tuple[str, ...]:
    if len(reasons) <= MAX_REASONS:
        return tuple(reasons)
    return (*reasons[:MAX_REASONS], f'... and {len(reasons) - MAX_REASONS} more')


def _passed(chunks: int, layers: int, ranks: int, clean_starts: list[int], clean_decode: set[str]) -> str:
    starts = f', starts {min(clean_starts)}..{max(clean_starts)}' if clean_starts else ''
    decodes = f', decode keys {", ".join(sorted(clean_decode))}' if clean_decode else ''
    return f'{chunks} chunks x {layers} layers, ranks={ranks}, all equal=1{starts}{decodes}'


@dataclass(frozen=True)
class DecodeAsk:
    """What a decode run asks of the log on top of the equality of every chunk: ``keys`` (ROWSxWIDTH) each with a complete equal=1 chunk;
    ``only``: a decode run, whose prompts are not long, so no --min-start is asked."""

    keys: tuple[str, ...] = ()
    only: bool = False


def judge(lines: Iterable[str], ranks: int = DEFAULT_RANKS, layers: int | None = None, min_starts: Sequence[int] = DEFAULT_MIN_STARTS,
          decode: DecodeAsk | None = None) -> Verdict:
    readings, ready = parse(lines)
    if not readings:
        return Verdict(NOT_MEASURED, 'no readings', ('no GLM53_RANK_CHECK lines in this log: the check never ran, or its lines are in another log',))
    decode = decode or DecodeAsk()
    chunks = _chunks(readings)
    expected = layers or ready or len({reading.layer for reading in readings if not reading.eager})
    unequal, reasons = _line_findings(readings, ranks)
    problems, clean_starts, clean_decode = _clean_chunks(chunks, expected, ranks)
    reasons += problems + _coverage_problems(clean_starts, () if decode.only else min_starts) + _decode_coverage_problems(clean_decode, decode.keys)
    summary = f'{len(chunks)} chunks x {expected} layers, ranks={ranks}, {len(unequal)} unequal, {len(reasons)} other problems'
    if unequal:
        return Verdict(FAIL, summary, _capped(unequal + reasons))
    if reasons:
        return Verdict(NOT_MEASURED, summary, _capped(reasons))
    return Verdict(PASS, _passed(len(chunks), expected, ranks, clean_starts, clean_decode), ())


def _decode_key_argument(text: str) -> str:
    if not DECODE_KEY.fullmatch(text):
        raise argparse.ArgumentTypeError(f'{text!r} must be ROWSxWIDTH, e.g. 8x4096 (the key=decode:ROWSxWIDTH of a log line)')
    return text


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('log', type=Path)
    parser.add_argument('--ranks', type=int, default=DEFAULT_RANKS, help='tensor-parallel ranks every reading must show (default 4)')
    parser.add_argument('--layers', type=int, default=None, help='layers per chunk (default: the READY line, else the layers seen)')
    parser.add_argument('--min-start', type=int, action='append', default=None,
                        help='a complete equal=1 chunk must start at or past this; repeatable (default 150000 and 530000)')
    parser.add_argument('--decode-key', type=_decode_key_argument, action='append', default=[], metavar='ROWSxWIDTH',
                        help='a decode key that must have a complete equal=1 chunk; repeatable')
    parser.add_argument('--decode-only', action='store_true', help='a decode run: ask for no --min-start (prompt lines in the log are still judged)')
    args = parser.parse_args(argv)
    try:
        lines = args.log.read_text(encoding='utf-8', errors='replace').splitlines()
    except OSError as error:
        print(f'{NOT_MEASURED}: cannot read {args.log}: {error}')
        return EXIT_CODES[NOT_MEASURED]
    min_starts = tuple(args.min_start) if args.min_start else DEFAULT_MIN_STARTS
    result = judge(lines, args.ranks, args.layers, min_starts, DecodeAsk(tuple(args.decode_key), args.decode_only))
    print(f'{result.status}: {result.summary}')
    for reason in result.reasons:
        print(f'  - {reason}')
    return result.exit_code


if __name__ == '__main__':
    raise SystemExit(main())
