"""diag-r4e/det_texts.py: the self-det drive with its lane texts kept, and the per-lane diff of two such drives.

drive  CELLS_DIR TEXTS -- DRIVE_CELLS_ARGS...
    Runs CELLS_DIR/drive_cells.py main() unchanged (its cells.jsonl rows and exit code are untouched), with diag-tie/det_drive.complete
    wrapped so every set's 16 lane texts are also appended to TEXTS as {"cell", "salt", "texts"}. drive_cells and det_drive keep their
    pinned bytes: the wrap lives here, in this process only.
    DET_LOGPROBS=N (1..20; 0 or unset = off, the default) sends det_drive's request with `logprobs: N` added and keeps each lane's
    tokens, token_logprobs, top_logprobs and text_offset in the row's "logprobs". It is NOT host-side only: the server's sampler then
    runs log_softmax + top-k + a rank count on the device logits (vllm/v1/sample/sampler.py forward/gather_logprobs, called with
    logits_device by hpu_model_runner._run_sampling) and the penalty fold keys that as its own sampler graph, which no warmup covers
    (glm53_penalty_fold.py graph_key extras "logprobs"; its header: such requests compile live and boot_budget.py serve fails them).
    The model graphs and the sampled token (argmax of the same logits) do not depend on it.
diff   REF_DIR REP_DIR [--port P] [--margin-dir DIR]
    Per det set, every lane whose texts differ between REF_DIR/texts.jsonl and REP_DIR/texts.jsonl: the first divergent character, the
    first divergent token (the token count of the shared prefix through the engine's /tokenize when --port answers, else "n/a"), and a
    short window of both sides. Then, per pass that kept logprobs (ref, rep, and DIR = one more drive of the same sets, not judged),
    a DET_MARGIN line for the first differing token: the token chosen, its logprob, the top 5 and the top1-top2 margin in nats.
    Prints DET_DIFF and DET_MARGIN lines and one DET_DIFF_SUMMARY line; exit 0 always (it reports, det_gate.py judges).
"""
from __future__ import annotations

import argparse
import bisect
import json
import os
import sys
import urllib.request
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType

L = Path.home() / '.codex/tmp/flash-spikes-20261104/team-20261106-dual'
MODEL = 'glm53flash-abliterated'
WINDOW = 24
LOGPROBS_ENV = 'DET_LOGPROBS'
MAX_LOGPROBS = 20  # the server's max_logprobs default (vllm/config/model.py); a larger ask is refused with a 400
TOP_SHOWN = 5
LOGPROB_FIELDS = ('tokens', 'token_logprobs', 'top_logprobs', 'text_offset')
MARGIN_HINT = ('no logprobs were recorded for this lane (R4E_DET_LOGPROBS=0 and no margin drive); set R4E_DET_MARGIN_ON_FAIL=5 to '
               'record them in one more drive after a FAIL, leaving the judged passes as they are')


def logprobs_wanted(raw: str | None = None) -> int:
    """The top-N the drive asks for, from DET_LOGPROBS; 0 (or unset) sends det_drive's own request with no logprobs field."""
    raw = os.environ.get(LOGPROBS_ENV, '0') if raw is None else raw
    if not raw.isdigit() or int(raw) > MAX_LOGPROBS:
        raise SystemExit(f'HALT {LOGPROBS_ENV}={raw!r}: use 0 (off, the default) or 1..{MAX_LOGPROBS} top alternatives per token '
                         f'(the server refuses more than max_logprobs={MAX_LOGPROBS})')
    return int(raw)


def kept_logprobs(raw: dict | None) -> dict | None:
    """One lane's logprobs as stored (None when the server sent none: the diff then says n/a for that pass)."""
    return {field: raw.get(field) for field in LOGPROB_FIELDS} if raw else None


@dataclass(frozen=True)
class Recorder:
    """One drive's wrap of det_drive.complete: each set's lane texts (and, for top > 0, their logprobs) appended to texts_path."""
    det_drive: ModuleType
    original: Callable[..., list[str]]
    texts_path: Path
    top: int = 0

    def with_logprobs(self, port: int, prompts: list[list[int]], tokens: int, salt: str, timeout: int) -> tuple[list[str], list[dict | None]]:
        """det_drive.complete's request with `logprobs: top` added; (lane texts, lane logprobs), both in lane order."""
        body = {'model': self.det_drive.MODEL, 'prompt': prompts, 'temperature': 0, 'ignore_eos': True, 'max_tokens': tokens,
                'cache_salt': salt, 'logprobs': self.top}
        request = urllib.request.Request(f'http://127.0.0.1:{port}/v1/completions', data=json.dumps(body).encode(),
                                         headers={'Content-Type': 'application/json'})
        with urllib.request.urlopen(request, timeout=timeout) as response:
            choices = sorted(json.load(response)['choices'], key=lambda choice: choice['index'])
        return [choice['text'] for choice in choices], [kept_logprobs(choice.get('logprobs')) for choice in choices]

    def complete(self, port: int, prompts: list[list[int]], tokens: int, salt: str, timeout: int) -> list[str]:
        """det_drive.complete's signature and return value: the texts handed back are the ones det_drive hashes."""
        width_of = {anchor: width for width, anchor in self.det_drive.ANCHOR_TOKENS.items()}
        row: dict[str, object] = {'cell': f'det-16x{width_of.get(len(prompts[0]), len(prompts[0]))}', 'salt': salt}
        if self.top:
            row['texts'], row['logprobs'] = self.with_logprobs(port, prompts, tokens, salt, timeout)
        else:
            row['texts'] = self.original(port, prompts, tokens, salt, timeout)
        with self.texts_path.open('a', encoding='utf-8') as out:
            out.write(json.dumps(row) + '\n')
        return row['texts']


def record_texts(det_drive: ModuleType, texts_path: Path, top: int = 0) -> None:
    """Wrap det_drive.complete so each set's texts land in texts_path; top > 0 also asks for and keeps the lanes' logprobs."""
    det_drive.complete = Recorder(det_drive, det_drive.complete, texts_path, top).complete


def drive(cells_dir: Path, texts_path: Path, drive_args: list[str]) -> int:
    top = logprobs_wanted()  # before any import or request: a bad value stops the drive here
    sys.path.insert(0, str(L / 'diag-tie'))
    import det_drive  # noqa: PLC0415  (the module drive_cells.drive_det imports; sys.modules hands it the wrapped one)

    record_texts(det_drive, texts_path, top)
    sys.path.insert(0, str(cells_dir))
    import drive_cells  # noqa: PLC0415

    return drive_cells.main(drive_args)


def read_rows(folder: Path | None) -> dict[str, dict]:
    """cell -> the last recorded row for it ({} when there is no folder or the drive kept no texts)."""
    path = folder / 'texts.jsonl' if folder is not None else None
    if path is None or not path.exists():
        return {}
    rows = [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]
    return {row['cell']: row for row in rows}


def token_count(port: int | None, text: str) -> str:
    """Tokens in text through the engine's /tokenize; "n/a" when there is no port or it does not answer."""
    if port is None:
        return 'n/a'
    body = json.dumps({'model': MODEL, 'prompt': text, 'add_special_tokens': False}).encode()
    request = urllib.request.Request(f'http://127.0.0.1:{port}/tokenize', data=body, headers={'Content-Type': 'application/json'})
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            return str(json.load(response)['count'])
    except Exception as error:  # the diff is a report: a dead tokenizer endpoint costs the token index, not the diff
        return f'n/a ({type(error).__name__})'


def first_divergence(left: Sequence, right: Sequence) -> int:
    return next((index for index, (a, b) in enumerate(zip(left, right)) if a != b), min(len(left), len(right)))


def lane_line(cell: str, lane: int, left: str, right: str, port: int | None) -> str:
    at = first_divergence(left, right)
    ref, rep = left[at:at + WINDOW], right[at:at + WINDOW]
    return (f'DET_DIFF {cell} lane {lane}: first divergent char {at}/{len(left)} token {token_count(port, left[:at])} '
            f'ref={json.dumps(ref)} rep={json.dumps(rep)}')


def lane_logprobs(rows: dict[str, dict], cell: str, lane: int) -> dict | None:
    """The lane's stored logprobs in one pass; None when the pass kept none (or has no tokens for the lane)."""
    lanes = rows.get(cell, {}).get('logprobs')
    found = lanes[lane] if lanes and lane < len(lanes) else None
    return found if found and found.get('tokens') else None


def token_at(logprobs: dict, char: int) -> int:
    """The index of the token holding output character `char` (text_offset is each token's first character)."""
    offsets = logprobs.get('text_offset') or [0]
    return min(max(bisect.bisect_right(offsets, offsets[0] + char) - 1, 0), len(logprobs['tokens']) - 1)


def entry(values: list | None, index: int, default: object) -> object:
    """values[index], or default when the list is missing, short or holds None there."""
    return values[index] if values and index < len(values) and values[index] is not None else default


def margin_line(cell: str, lane: int, name: str, logprobs: dict, index: int) -> str:
    """One pass's DET_MARGIN line at token `index`: the token chosen, its logprob, the top alternatives and the top1-top2 margin."""
    index = min(index, len(logprobs['tokens']) - 1)
    top = sorted(entry(logprobs.get('top_logprobs'), index, {}).items(), key=lambda item: -item[1])[:TOP_SHOWN]
    margin = f'{top[0][1] - top[1][1]:.6f}' if len(top) > 1 else 'n/a'
    chosen = entry(logprobs.get('token_logprobs'), index, None)
    return (f'DET_MARGIN {cell} lane {lane} pass={name} token {index}/{len(logprobs["tokens"])}: chose {json.dumps(logprobs["tokens"][index])} '
            f'logprob={"n/a" if chosen is None else format(chosen, ".6f")} margin_nats={margin} (top1-top2) '
            f'top{TOP_SHOWN}={json.dumps([[token, round(value, 6)] for token, value in top])}')


def margin_lines(cell: str, lane: int, char: int, passes: dict[str, dict | None]) -> list[str]:
    """The DET_MARGIN lines of one differing lane, one per pass that kept logprobs. With both judged passes the token is the first
    one their token lists differ at; a lone pass (the margin drive) is read at the token holding the first divergent character."""
    ref, rep = passes.get('ref'), passes.get('rep')
    shared = first_divergence(ref['tokens'], rep['tokens']) if ref and rep else None
    lines = [margin_line(cell, lane, name, logprobs, token_at(logprobs, char) if shared is None or name == 'margin' else shared)
             for name, logprobs in passes.items() if logprobs]
    return lines or [f'DET_MARGIN {cell} lane {lane}: n/a, {MARGIN_HINT}']


def diff(ref_dir: Path, rep_dir: Path, port: int | None, margin_dir: Path | None = None) -> list[str]:
    rows = {'ref': read_rows(ref_dir), 'rep': read_rows(rep_dir), 'margin': read_rows(margin_dir)}
    ref, rep = ({cell: row['texts'] for cell, row in rows[name].items()} for name in ('ref', 'rep'))
    if not ref or not rep:
        return [f'DET_DIFF_SUMMARY texts not captured (ref={len(ref)} sets, rep={len(rep)} sets): lanes from det_gate.py only']
    lines, differing = [], []
    for cell in sorted(set(ref) & set(rep)):
        lanes = [lane for lane, (a, b) in enumerate(zip(ref[cell], rep[cell])) if a != b]
        differing += [f'{cell}:{lane}' for lane in lanes]
        for lane in lanes:
            lines.append(lane_line(cell, lane, ref[cell][lane], rep[cell][lane], port))
            lines += margin_lines(cell, lane, first_divergence(ref[cell][lane], rep[cell][lane]),
                                  {name: lane_logprobs(rows[name], cell, lane) for name in rows})
    unpaired = sorted(set(ref) ^ set(rep))
    tail = f' unpaired_sets={unpaired}' if unpaired else ''
    return lines + [f'DET_DIFF_SUMMARY lanes_differing={len(differing)} sets={len(set(ref) & set(rep))} [{" ".join(differing)}]{tail}']


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if argv[:1] == ['drive'] and len(argv) >= 4 and argv[3] == '--':
        return drive(Path(argv[1]), Path(argv[2]), argv[4:])
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['diff'])
    parser.add_argument('ref_dir', type=Path)
    parser.add_argument('rep_dir', type=Path)
    parser.add_argument('--port', type=int)
    parser.add_argument('--margin-dir', type=Path)
    args = parser.parse_args(argv)
    print('\n'.join(diff(args.ref_dir, args.rep_dir, args.port, args.margin_dir)))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
