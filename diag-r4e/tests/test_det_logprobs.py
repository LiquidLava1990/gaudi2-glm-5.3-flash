"""The self-det drive and grader with a stub engine: what DET_LOGPROBS sends and keeps, that det_gate.py's verdict does not move, and the
margin lines of the diff. The real det_texts.py drive -> diag-cells/drive_cells.py -> diag-tie/det_drive.py run as the launcher runs them
(a child process per pass) against a loopback HTTP stub; det_gate.py is the pinned file, run on the stub's rows. Host python3, no card.

usage: python3 tests/test_det_logprobs.py        (R4E_KIT=<dir> tests a staged copy of the kit; CELLS=<dir> another diag-cells)
"""
from __future__ import annotations

import hashlib
import json
import os
import random
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

KIT = Path(os.environ.get('R4E_KIT', Path(__file__).resolve().parents[1]))
L = Path.home() / '.codex/tmp/flash-spikes-20261104/team-20261106-dual'
CELLS = Path(os.environ.get('CELLS', L / 'diag-cells'))
WIDTH_OF = {2048: 1024, 262_400: 4096, 530_000: 8192}
TOKENS = [' alpha', ' beta', ' gamma', ' delta']
RIVAL = ' omega'
FLIP = (4096, 11)  # tonight's lane: det-16x4096 lane 11
DRIVE_KEYS = {'model', 'prompt', 'temperature', 'ignore_eos', 'max_tokens', 'cache_salt'}  # det_drive.complete's request
METRICS = 'vllm:prefix_cache_hits_total{model="m"} 0.0\nvllm:external_prefix_cache_hits_total{model="m"} 0.0\n'


class Engine:
    """What the stub serves and what it saw: flip_at = the token index lane FLIP answers the rival token at (None = no flip)."""

    def __init__(self) -> None:
        self.flip_at: int | None = None
        self.send_logprobs = True
        self.requests: list[dict] = []
        self.steps: list[str] = []

    def lane_tokens(self, width: int, lane: int) -> list[str]:
        tokens = [f'{token}{lane}' for token in TOKENS]
        if self.flip_at is not None and (width, lane) == FLIP:
            tokens[self.flip_at] = RIVAL
        return tokens

    def lane_logprobs(self, tokens: list[str]) -> dict:
        offsets = [sum(len(token) for token in tokens[:index]) for index in range(len(tokens))]
        tops = [{token: -0.693140, (RIVAL if token != RIVAL else ' alpha11'): -0.693160, ' x': -5.0, ' y': -6.0, ' z': -7.0, ' w': -8.0}
                for token in tokens]
        return {'tokens': tokens, 'token_logprobs': [-0.693140] * len(tokens), 'top_logprobs': tops, 'text_offset': offsets}

    def complete(self, body: dict) -> dict:
        width = WIDTH_OF[len(body['prompt'][0])]
        self.requests.append({key: value for key, value in body.items() if key != 'prompt'} | {'lanes': len(body['prompt'])})
        self.steps.append('(EngineCore pid=1) GLM53_STEP_STATS step ' + json.dumps(
            {'i': len(self.steps), 't': time.time(), 'kind': 'pure-decode', 'wall_ms': 50.0, 'decode_rows': 16, 'prompt_rows': 0, 'prompts': []}))
        choices = []
        for lane in range(len(body['prompt'])):
            tokens = self.lane_tokens(width, lane)
            choice = {'index': lane, 'text': ''.join(tokens)}
            if 'logprobs' in body and self.send_logprobs:
                choice['logprobs'] = self.lane_logprobs(tokens)
            choices.append(choice)
        random.Random(width).shuffle(choices)  # the drive must order by index, not by arrival
        return {'choices': choices}


def handler_for(engine: Engine) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        def reply(self, payload: bytes) -> None:
            self.send_response(200)
            self.send_header('Content-Length', str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def do_GET(self) -> None:  # noqa: N802  (/metrics: the prefix-cache hit counters, never moving)
            self.reply(METRICS.encode())

        def do_POST(self) -> None:  # noqa: N802
            body = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
            if self.path == '/tokenize':
                self.reply(json.dumps({'count': len(body['prompt'].split())}).encode())
            else:
                self.reply(json.dumps(engine.complete(body)).encode())

        def log_message(self, *args: object) -> None:
            return

    return Handler


class Pass:
    """One det drive (a child process, as r4e_run.sh det_pass runs it) and what it left."""

    def __init__(self, root: Path, name: str, port: int, engine: Engine, *, logprobs: str | None = None, flip_at: int | None = None) -> None:
        self.dir = root / name
        self.dir.mkdir()
        engine.flip_at, first = flip_at, len(engine.requests)
        env = {key: value for key, value in os.environ.items() if key != 'DET_LOGPROBS'} | ({} if logprobs is None else {'DET_LOGPROBS': logprobs})
        done = subprocess.run([sys.executable, str(KIT / 'det_texts.py'), 'drive', str(CELLS), str(self.dir / 'texts.jsonl'), '--',
                               '--port', str(port), '--out', str(self.dir / 'cells.jsonl'), '--plan', 'det'],
                              cwd=L, env=env, capture_output=True, text=True, timeout=600)
        self.rc, self.stderr = done.returncode, done.stderr
        self.requests = engine.requests[first:]
        self.texts = self.rows('texts.jsonl')
        self.hashes = [row.get('output_sha256') for row in self.rows('cells.jsonl')]

    def rows(self, name: str) -> list[dict]:
        path = self.dir / name
        return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []


class SelfDetLogprobs(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.engine = Engine()
        cls.server = ThreadingHTTPServer(('127.0.0.1', 0), handler_for(cls.engine))
        cls.port = cls.server.server_address[1]
        threading.Thread(target=cls.server.serve_forever, daemon=True).start()
        cls.folder = tempfile.TemporaryDirectory()
        root = Path(cls.folder.name)
        cls.plain_ref = Pass(root, 'plain-ref', cls.port, cls.engine)
        cls.plain_same = Pass(root, 'plain-same', cls.port, cls.engine)
        cls.plain_rep = Pass(root, 'plain-rep', cls.port, cls.engine, logprobs='0', flip_at=0)
        cls.asked_ref = Pass(root, 'asked-ref', cls.port, cls.engine, logprobs='5')
        cls.asked_rep = Pass(root, 'asked-rep', cls.port, cls.engine, logprobs='5', flip_at=0)
        cls.margin_late = Pass(root, 'margin-late', cls.port, cls.engine, logprobs='5', flip_at=2)
        cls.plain_late = Pass(root, 'plain-late', cls.port, cls.engine, flip_at=2)
        for run in (cls.plain_ref, cls.plain_same, cls.plain_rep, cls.asked_ref, cls.asked_rep, cls.margin_late, cls.plain_late):
            (run.dir / 'container.log').write_text('\n'.join(cls.engine.steps) + '\n')  # r4e_run.sh: one boot log, copied to both passes

    @classmethod
    def tearDownClass(cls) -> None:
        cls.server.shutdown()
        cls.server.server_close()
        cls.folder.cleanup()

    def gate(self, left: Pass, right: Pass) -> tuple[int, str]:
        done = subprocess.run([sys.executable, 'det_gate.py', str(left.dir), str(right.dir)], cwd=CELLS, capture_output=True, text=True, timeout=120)
        return done.returncode, done.stdout

    def diff(self, left: Pass, right: Pass, *extra: str) -> list[str]:
        done = subprocess.run([sys.executable, '-I', str(KIT / 'det_texts.py'), 'diff', str(left.dir), str(right.dir), *extra],
                              capture_output=True, text=True, timeout=120)
        self.assertEqual(done.returncode, 0, done.stderr)
        return done.stdout.splitlines()

    def test_off_by_default_the_request_is_det_drives_own(self) -> None:
        for run in (self.plain_ref, self.plain_rep):  # unset and DET_LOGPROBS=0
            self.assertEqual((run.rc, len(run.requests)), (0, 3), run.stderr)
            self.assertTrue(all(set(request) | {'prompt'} == DRIVE_KEYS | {'lanes'} for request in run.requests), run.requests)
            self.assertTrue(all('logprobs' not in row for row in run.texts))

    def test_asked_the_request_gains_only_logprobs_and_every_lane_keeps_them(self) -> None:
        run = self.asked_ref
        self.assertEqual((run.rc, len(run.requests)), (0, 3), run.stderr)
        for asked, plain in zip(run.requests, self.plain_ref.requests):
            self.assertEqual(asked['logprobs'], 5)
            self.assertEqual({key: value for key, value in asked.items() if key not in ('logprobs', 'cache_salt')},
                             {key: value for key, value in plain.items() if key != 'cache_salt'})
        for row in run.texts:
            self.assertEqual(len(row['logprobs']), 16)
            self.assertTrue(all(set(lane) == {'tokens', 'token_logprobs', 'top_logprobs', 'text_offset'} for lane in row['logprobs']))
            self.assertEqual([''.join(lane['tokens']) for lane in row['logprobs']], row['texts'])

    def test_asking_does_not_move_the_lane_hashes_or_texts(self) -> None:
        self.assertEqual(self.asked_ref.hashes, self.plain_ref.hashes)
        self.assertEqual(self.asked_rep.hashes, self.plain_rep.hashes)
        self.assertEqual([row['texts'] for row in self.asked_rep.texts], [row['texts'] for row in self.plain_rep.texts])
        self.assertEqual(len(self.plain_ref.hashes[0]), 16)

    def test_a_bad_value_halts_before_any_request(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            for value in ('five', '21', '-1', ''):
                run = Pass(Path(folder), f'bad-{value or "empty"}', self.port, self.engine, logprobs=value)
                self.assertNotEqual(run.rc, 0)
                self.assertIn(f"HALT DET_LOGPROBS={value!r}: use 0 (off, the default) or 1..20", run.stderr)
                self.assertEqual((run.requests, run.texts, run.hashes), ([], [], []))

    def test_a_server_that_sends_no_logprobs_still_drives(self) -> None:
        self.engine.send_logprobs = False
        try:
            with tempfile.TemporaryDirectory() as folder:
                run = Pass(Path(folder), 'silent', self.port, self.engine, logprobs='5')
        finally:
            self.engine.send_logprobs = True
        self.assertEqual(run.rc, 0, run.stderr)
        self.assertEqual([row['logprobs'] for row in run.texts], [[None] * 16] * 3)
        self.assertEqual(run.hashes, self.plain_ref.hashes)

    def test_the_grader_is_the_pinned_file_and_its_verdict_does_not_move(self) -> None:
        pinned = dict(reversed(line.split()) for line in (CELLS / 'SHA256SUMS').read_text().splitlines())
        self.assertEqual(hashlib.sha256((CELLS / 'det_gate.py').read_bytes()).hexdigest(), pinned['det_gate.py'])
        self.assertEqual(self.gate(self.plain_ref, self.plain_same)[0], 0)
        self.assertEqual(self.gate(self.asked_ref, self.plain_same)[0], 0)
        for left, right in ((self.plain_ref, self.plain_rep), (self.asked_ref, self.asked_rep)):
            code, out = self.gate(left, right)
            self.assertEqual(code, 1, out)
            self.assertIn('det-16x4096: steps matched 1/1 DIFFER lanes [11]', out)
            self.assertIn('det-16x1024: steps matched 1/1 IDENTICAL', out)
            self.assertTrue(out.rstrip().endswith('DET_GATE FAIL: a lane differs at a matched schedule'), out)

    def test_the_diff_says_both_tokens_logprobs_top5_and_margin(self) -> None:
        lines = self.diff(self.asked_ref, self.asked_rep)
        top_ref = '[[" alpha11", -0.69314], [" omega", -0.69316], [" x", -5.0], [" y", -6.0], [" z", -7.0]]'
        top_rep = '[[" omega", -0.69314], [" alpha11", -0.69316], [" x", -5.0], [" y", -6.0], [" z", -7.0]]'
        self.assertEqual(lines, [
            'DET_DIFF det-16x4096 lane 11: first divergent char 1/31 token n/a ref="alpha11 beta11 gamma11 d" rep="omega beta11 gamma11 del"',
            f'DET_MARGIN det-16x4096 lane 11 pass=ref token 0/4: chose " alpha11" logprob=-0.693140 margin_nats=0.000020 (top1-top2) top5={top_ref}',
            f'DET_MARGIN det-16x4096 lane 11 pass=rep token 0/4: chose " omega" logprob=-0.693140 margin_nats=0.000020 (top1-top2) top5={top_rep}',
            'DET_DIFF_SUMMARY lanes_differing=1 sets=3 [det-16x4096:11]'])

    def test_the_diff_without_logprobs_says_how_to_get_them(self) -> None:
        lines = self.diff(self.plain_ref, self.plain_rep, '--port', str(self.port))
        self.assertEqual(lines[0], 'DET_DIFF det-16x4096 lane 11: first divergent char 1/31 token 0 ref="alpha11 beta11 gamma11 d" '
                                   'rep="omega beta11 gamma11 del"')
        self.assertTrue(lines[1].startswith('DET_MARGIN det-16x4096 lane 11: n/a, no logprobs were recorded for this lane'), lines[1])
        self.assertIn('set R4E_DET_MARGIN_ON_FAIL=5', lines[1])
        self.assertEqual(len(lines), 3)

    def test_a_margin_drive_gives_the_margin_of_passes_judged_without_logprobs(self) -> None:
        lines = self.diff(self.plain_ref, self.plain_late, '--margin-dir', str(self.margin_late.dir))
        self.assertTrue(lines[0].startswith('DET_DIFF det-16x4096 lane 11: first divergent char 16/31 '), lines[0])
        self.assertTrue(lines[1].startswith('DET_MARGIN det-16x4096 lane 11 pass=margin token 2/4: chose " omega" logprob=-0.693140 '
                                            'margin_nats=0.000020 (top1-top2) top5=[[" omega", -0.69314], [" alpha11", -0.69316], '), lines[1])
        self.assertEqual(len(lines), 3)


if __name__ == '__main__':
    unittest.main(verbosity=1)
