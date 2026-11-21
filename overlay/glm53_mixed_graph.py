"""Bounded mixed HPUGraphs. Sampling and metadata preparation stay outside capture.

Each entry owns all input tensor storage and the captured output. A capture
executes the model exactly once: never warm a stateful forward by running it
twice on live requests. Capture failure is fatal, since cache writes may have
already executed; it must never retry the same request through lazy forward.
"""
from __future__ import annotations

from collections import Counter
from contextlib import contextmanager, nullcontext
from contextvars import ContextVar
from dataclasses import fields, is_dataclass, replace
from functools import lru_cache
from itertools import accumulate
from pathlib import Path
from types import SimpleNamespace
import hashlib
import logging
import os
import re
import sys
import time
from typing import Any, Callable, Iterable, Iterator, NamedTuple, Protocol

import glm53_gc_freeze as gc_freeze
import glm53_graphs_only as graphs_only
from glm53_prompt_census import STALL_DIR_ENV, MarkStepCensus, census_enabled, census_graph_forward
import torch

FLAG = 'VLLM_GAUDI_MIXED_GRAPHS'
LOG = logging.getLogger(__name__)
MIB = 1024 * 1024
_CAPTURE = ContextVar('glm53_mixed_capture', default=False)
_BOOT = ContextVar('glm53_mixed_boot', default=False)
CALIBRATION_BYTES = 5_020_906_496
CALIBRATION_PROXY = 256 * MIB + 516 * MIB + 516 * 256 * 264


def capture_active() -> bool:
    return _CAPTURE.get()


@contextmanager
def capture_context() -> Any:
    token = _CAPTURE.set(True)
    try:
        yield
    finally:
        _CAPTURE.reset(token)


def enabled() -> bool:
    return os.getenv(FLAG, '0') == '1'


@contextmanager
def boot_capture_context() -> Any:
    token = _BOOT.set(True)
    try:
        yield
    finally:
        _BOOT.reset(token)


PROMPT_KEYS_FLAG = 'VLLM_GAUDI_MIXED_PROMPT_KEYS'
CAPTURE_MARKS_FLAG = 'HPU_GLM53_R589_CAPTURE_MARKS'
MERGE_CHECK_FLAG = 'HPU_GLM53_PROMPT_KEY_MERGE_CHECK'
TAIL_FUSE_FLAG = 'VLLM_RECURRENT_TAIL_FUSE'
TAIL_FUSE_KEYED_FLAG = 'VLLM_RECURRENT_TAIL_FUSE_KEYED'  # the fused floor128 tail inside a prompt key (glm53_mixed_tail_fuse rows)
KEY_ROWS_QUANTUM = 128
SMALL_KEY_MAX_ROWS = 128  # the indexer's legacy route takes these: the tiled route starts above 128 rows
SMALL_KEY_MERGE_FLAG = 'HPU_GLM53_SMALL_KEY_PARTIAL_MERGE'
PARTIAL_SCAN_TOPK_FLAG = 'HPU_GLM53_PARTIAL_SCAN_TOPK'  # kpool's _USE_PARTIAL_SCAN_TOPK
R590_FLAG = 'HPU_GLM53_R590_ENABLE'  # glm53_r590_long_history.enabled
KEY_BASE_MIB, KEY_MIB_PER_ROW, KEY_RESIDENT_PER_MILLE = 1536, 7, 327  # section 3: resident = (1536 + 7R) MiB x 0.327
KEY_RESERVE_DEFAULT_MIB = 512
KEPT_LEVERS = ('mla_unpack', 'mla_topkfuse', 'k512_topk', 'fwht', 'sinkhorn', 'kda_bitwise_rows', 'kda_block_inverse',
               'private_topk', 'final_norm_mark')
MERGE_MIN_PAST_TOP_K_START = 4096  # top-k 2048: a chunk starting here selects, so a width-dependent selection can show
MERGE_MAX_PENDING = 4  # hidden snapshots held on the host between the two runs of a chunk
BALLAST_ENV = 'HPU_GLM53_K1_BALLAST_MIB'
BALLAST_EVERY_ENV = 'HPU_GLM53_K1_BALLAST_EVERY'
BALLAST_EVERY = 12  # key replays per level by default: 2 repeats of the 6-chunk K1 prompt
BALLAST_CHUNK_MIB = 1024
SELF_MAX_REFERENCES = 16  # first outputs kept per (chunk, key); the K1 prompt has at most 8
_KEY_CAPTURE = ContextVar('glm53_key_capture', default=False)
DECODE_WIDTHS_FLAG = 'VLLM_GAUDI_MIXED_DECODE_WIDTHS'  # table widths in blocks; the lattice is every DECODE_ROWS x every width
DECODE_PAD_FLAG = 'HPU_GLM53_DECODE_WIDTH_PAD'  # 0 native; check pads and compares; 1 pads, naming the run that proved it
DECODE_PAD_PROOF_FLAG = 'HPU_GLM53_DECODE_PAD_PROOF'
DECODE_CHECK_WIDTHS_FLAG = 'HPU_GLM53_DECODE_PAD_CHECK_WIDTHS'  # native one-row keys the check compares against the padded key
DECODE_REPLAY_FLAG = 'HPU_GLM53_MIXED_DECODE_REPLAY'
CELL_BUDGET_FLAG = 'VLLM_PREFILL_CONTEXT_CELL_BUDGET'
SHARED_OWNER_FLAG = 'HPU_GLM53_Q4_SHARED_GATHER_OWNER'
RESTORED_DECODE_FLAG = 'VLLM_HPU_RESTORED_DECODE_GRAPHS'
DRY_RUN_FLAG = 'HPU_GRAPH_DRY_RUN'
GRAPHS_ONLY_FLAG = graphs_only.ENV  # steps 23, 30: every serving forward replays a captured graph; one that cannot is refused on every rank
MAX_REQUESTS_FLAG = 'VLLM_PREFILL_MAX_REQUESTS_PER_STEP'
SINGLE_TOKENS_FLAG = 'VLLM_ADAPTIVE_LONG_PREFILL_SINGLE_TOKENS'  # a single prompt's chunk rows under completion-first long prefill
MIXED_TOKENS_FLAG = 'VLLM_GAUDI_MIXED_PREFILL_TOKENS'  # the mixed step's chunk rows
PRIVATE_TOPK_FLAG = 'HPU_GLM53_PRIVATE_PROMPT_TOPK'
GRAPHS_ONLY_FIXES = {
    'width_over': f'the chunk has more rows than the tallest key; set {SINGLE_TOKENS_FLAG} and {MIXED_TOKENS_FLAG} to the key rows, or add a taller key',
    'ctx_over': f'the chunk table is wider than the widest key; add a wider key to {PROMPT_KEYS_FLAG} (context and KV stay as they are)',
    'multi_prompt': 'a step carried more than one prompt; the prompt pool closes through the slot policy or the prompt keys, so add a key a second prompt can use or restart with the slot policy that admits one prompt per step',
    'signature': 'the chunk has no block table a key can pad; the step was built outside the paired runner, so report it with this message',
    'not_warmed': 'the key was never captured at boot, which boot should have refused; check the boot log for GLM53_BOOT_WITNESS and restart',
}
QUERY_SHARD_FLAG = 'HPU_GLM53_QUERY_SHARD_KEYS'  # ROWS:WIDTH prompt keys whose scan each rank runs on its own rows (glm53_r590_long_history.query_shard)
QUERY_SHARD_DECODE_FLAG = 'HPU_GLM53_QUERY_SHARD_DECODE_KEYS'  # ROWS:WIDTH decode keys, the same split (glm53_r590_long_history.decode_query_shard)
RANK_CHECK_FLAG = 'HPU_GLM53_INDEXER_RANK_CHECK'  # 1: every layer's physical rows compared across ranks after a real replay (prompt and decode keys); eager: on the runner's own forward
DECODE_CHECK_REPLAYS_PER_KEY = 16  # served replays of one decode key the rank check reads: each read copies every layer's buffer to the host and hashes it
HOLD_TENSORS_FLAG = 'PT_HPU_LAZY_COLLECTIVES_HOLD_TENSORS'
SHARDED_KPOOL_FLAG = 'HPU_GLM53_TP_SHARDED_KPOOL'
RANK_CHECK_ATTR = '_glm53_rank_check_rows'  # an indexer's own rows buffer, only with the check on; plain attribute, as glm53_graph_state's holders are
RANK_CHECK_LAYER = re.compile(r'layers\.(\d+)')
_RANK_CHECK_BOOT_DONE = [False]  # warmup ended: an eager forward from here on holds real rows, not PT_COMPILE_ONLY_MODE's
WORKSPACE_RELEASE_FLAG = 'VLLM_GAUDI_MIXED_KEY_WORKSPACE_RELEASE'
RELEASE_OVERSHOOT_MIB = 64  # past free by this much forces the hand-back; fast4r overshot by 49 and released 5.1 GiB on all four ranks
SEAL_BALLAST_FLAG = 'VLLM_GAUDI_MIXED_SEAL_BALLAST_MIB'
SEAL_BALLAST_MIB = 512  # the margin only: the seal runs with the decode lattice captured, so K1's 1,538 (1,026 lattice + 512) would count it twice
SEAL_ADVICE = (f'Drop a key from {PROMPT_KEYS_FLAG}, or raise VLLM_GAUDI_MIXED_GRAPH_BUDGET_MIB after a held-ballast HBM reading, '
               'and restart all workers; context and KV stay as they are.')
KEY_CAPTURE_ADVICE = (f'Drop a key from {PROMPT_KEYS_FLAG}, or lower VLLM_GAUDI_MIXED_GRAPH_RESERVE_MIB against a held-ballast HBM reading, '
                      'and restart all workers; context and KV stay as they are.')
DECODE_ROWS = (1, 2, 4, 8, 16)
# MiB per decode row by table width: max over rows and ranks of each decode key's capture peak / rows in diag-key128/runs/r1 (owner on, the
# default 16,384-pool scan tile), rounded up. r1's peaks (census MaxInUse over the dummy and its forward, decode_peaks.py) equal the residual
# InUse deltas to 0.1 MiB for 19 of 20 keys; the first key's setup window holds the prompt seal, so 16x8192 has its forward peak only.
# The old rows-only table (owner off, width 1024) charged 84 MiB for an 8x8192 key that took 339.
# +1 MiB per row on each r1 figure: r1 16x8192 peaked 750.6 MiB against 47 x 16 = 752, and c1c (flags off) peaked 752.1-752.8 on every rank
# (cells r2), so the table had 1.4 MiB of slack at 16 rows and the image moved it by about 2. 1 MiB per row is 16 MiB at 16 rows.
DECODE_KEY_MIB_PER_ROW = {1024: 20, 2048: 24, 4096: 32, 8192: 48}
DECODE_OWNER_PERCENT = 72  # section 3: the owner at 72% of owner-off, the only owner reading; owner-off is charged the table over it until measured
DECODE_CAPTURE_MARGIN_MIB = 64  # allocator rounding on top of the estimate; the fatal r1 k256 allocation was a 4.3 MB block
HBM_MARGIN_FLAG = 'HPU_GLM53_HBM_MARGIN_MIB'  # diag-hbm's margin seal (glm53_hbm_margin); installed, the witness's charged_MiB is measured bytes
HBM_MARGIN_MODULE = 'glm53_hbm_margin'
HBM_MARGIN_WRAPPERS = frozenset({'seal_boot_memory_with_margin', '_seal_boot_memory_with_margin'})  # unmarked 7bf39655, d54efdc6 wrapper names
DECODE_SCAN_V2_MODULE = 'glm53_decode_scan_v2'  # the decode lane's L2a module; without it HPU_GLM53_DECODE_KPOOL_TILE_CEILING_FOR does nothing
CONTEXT_CHUNK_BUDGET_FLAG = 'HPU_GLM53_CONTEXT_CHUNK_BUDGET_MIB'
CONTEXT_CHUNK_BUDGET_DEFAULT_MIB = 512  # the indexer's default (hpu_glm5next_sparse_indexer)
DECODE_CHECK_MAX_PENDING = 64  # one hidden block per output token of a repeated request, held on the host
SPARSE_TOPK_TOKENS = 2048  # a decode batch whose longest row fits the indexer's top-k window takes the exact-causal branch at its native width
MERGE_LINE = re.compile(r'GLM53_DECODE_MERGE narrow=(\d+) wide=(\d+) rows=(\d+) position=\d+ torch_equal=([01]) short=([01])')
DECODE_ADVICE = (f'Drop a width from {DECODE_WIDTHS_FLAG}, or raise VLLM_GAUDI_MIXED_GRAPH_BUDGET_MIB after a held-ballast HBM reading, '
                 'and restart all workers; context and KV stay as they are.')
DECODE_HEADROOM_ADVICE = (f'Drop a width from {DECODE_WIDTHS_FLAG} or a key from {PROMPT_KEYS_FLAG} and restart all workers; raising '
                          'VLLM_GAUDI_MIXED_GRAPH_BUDGET_MIB cannot help, the card itself is full. Context and KV stay as they are.')
BOOT_ADVICE = ('Reduce WARMUP to one bucket, use --num-gpu-blocks-override 8000, '
               'and set BUDGET_MIB >= calibrated estimate; restart all workers.')
_DECODE_KEY = ContextVar('glm53_decode_key', default=None)
_WITNESS: Counter = Counter()


class RowsRule(NamedTuple):
    """The row counts a key flag accepts: ``allows`` says whether one qualifies, ``text`` is what the refusal says rows must be."""
    allows: Callable[[int], bool]
    text: str


PROMPT_ROWS = RowsRule(lambda rows: rows >= 1 and not rows % KEY_ROWS_QUANTUM, f'a positive multiple of {KEY_ROWS_QUANTUM}')
DECODE_KEY_ROWS = RowsRule(lambda rows: rows in DECODE_ROWS, f'one of the decode lattice rows {list(DECODE_ROWS)}')


def _parse_key(item: str, flag: str, example: str, rows_rule: RowsRule) -> tuple[int, int]:
    """One ROWS:WIDTH item of a key flag: rows by ``rows_rule``, the width a power of two."""
    parts = item.split(':')
    if len(parts) != 2 or not all(part.isdigit() for part in parts):
        raise ValueError(f'{flag} must list ROWS:WIDTH pairs (e.g. {example}); got {item!r}')
    rows, width = int(parts[0]), int(parts[1])
    if not rows_rule.allows(rows) or width < 1 or width & (width - 1):
        raise ValueError(f'{flag} key {item!r}: rows must be {rows_rule.text} and width a power of two, the only table widths the runner builds')
    return rows, width


def _parse_prompt_key(item: str, flag: str = PROMPT_KEYS_FLAG) -> tuple[int, int]:
    return _parse_key(item, flag, '1024:8192,1024:1024', PROMPT_ROWS)


def _parse_decode_key(item: str, flag: str = QUERY_SHARD_DECODE_FLAG) -> tuple[int, int]:
    return _parse_key(item, flag, '8:4096,16:4096,4:8192', DECODE_KEY_ROWS)


def _key_list(spec: str, flag: str, parse: Callable[[str, str], tuple[int, int]]) -> tuple[tuple[int, int], ...]:
    keys: list[tuple[int, int]] = []
    for item in filter(None, spec.split(',')):
        key = parse(item, flag)
        if key in keys:
            raise ValueError(f'duplicate {flag} key {item!r}; list each key once')
        keys.append(key)
    return tuple(keys)


@lru_cache(maxsize=8)
def _parsed_prompt_keys(spec: str, flag: str = PROMPT_KEYS_FLAG) -> tuple[tuple[int, int], ...]:
    return _key_list(spec, flag, _parse_prompt_key)


@lru_cache(maxsize=8)
def _parsed_decode_keys(spec: str) -> tuple[tuple[int, int], ...]:
    return _key_list(spec, QUERY_SHARD_DECODE_FLAG, _parse_decode_key)


def prompt_keys() -> tuple[tuple[int, int], ...]:
    """Form B: (rows, table width in blocks) of one graph per key, BOOT captures in written order; empty = no keys."""
    return _parsed_prompt_keys(os.getenv(PROMPT_KEYS_FLAG, ''))


def query_shard_keys() -> tuple[tuple[int, int], ...]:
    """The prompt keys whose scan is query-sharded; empty = every key replicated, the same ops as without the flag."""
    return _parsed_prompt_keys(os.getenv(QUERY_SHARD_FLAG, ''), QUERY_SHARD_FLAG)


def decode_query_shard_keys() -> tuple[tuple[int, int], ...]:
    """The decode keys (rows, table width in blocks) whose scan is query-sharded; empty = every decode key replicated, the same ops as without the flag."""
    return _parsed_decode_keys(os.getenv(QUERY_SHARD_DECODE_FLAG, ''))


def rank_check_mode() -> str:
    """'' off, '1' read after each real replay, 'eager' the non-serving fallback on the runner's own forward."""
    value = os.getenv(RANK_CHECK_FLAG, '0').strip()
    if value not in ('', '0', '1', 'eager'):
        raise ValueError(f'{RANK_CHECK_FLAG}={value!r} must be 1 (read rows after real replays), eager (non-serving run on the '
                         "runner's own forward), or 0 / unset for off")
    return value if value in ('1', 'eager') else ''


def _switch(name: str) -> bool:
    value = os.getenv(name, '0')
    if value not in ('0', '1'):
        raise ValueError(f'{name}={value!r} must be 0 or 1; unset it (or 0) to keep it off')
    return value == '1'


def tail_fuse_keyed_enabled() -> bool:
    """VLLM_RECURRENT_TAIL_FUSE inside the prompt keys: the keyed step attaches tail fuse's device rows to its prompt region, so one key
    replay runs the chunk past b = floor128(P) and snapshots the KDA state at b into the bank (no separate <128-row tail step)."""
    return _switch(TAIL_FUSE_KEYED_FLAG)


def check_tail_fuse_keyed_flags(keys: tuple[tuple[int, int], ...]) -> None:
    """What the keyed tail fuse needs: the tail-fuse flag whose plan it fuses, a key capture, and one prompt per key."""
    if not tail_fuse_keyed_enabled():
        return
    from glm53_mixed_step import BATCH_KEYS_FLAG, prompt_batch_keys
    if os.getenv(TAIL_FUSE_FLAG, '0') == '0':
        raise ValueError(f'{TAIL_FUSE_KEYED_FLAG}=1 fuses the tail that {TAIL_FUSE_FLAG}=1 plans; set {TAIL_FUSE_FLAG}=1 too, '
                         f'or unset {TAIL_FUSE_KEYED_FLAG}')
    if not keys:
        raise ValueError(f'{TAIL_FUSE_KEYED_FLAG}=1 only applies to a key capture; set {PROMPT_KEYS_FLAG} or unset {TAIL_FUSE_KEYED_FLAG}')
    if prompt_batch_keys():
        raise ValueError(f'{TAIL_FUSE_KEYED_FLAG}=1 fuses one prompt per key and a shared key is unproven with a fused row; '
                         f'unset {BATCH_KEYS_FLAG} or {TAIL_FUSE_KEYED_FLAG}')


def capture_marks_enabled() -> bool:
    """Arm B: R589 tile boundaries and scan fences stay in a key capture as mark_step(sync=False)."""
    return _switch(CAPTURE_MARKS_FLAG)


def merge_check_enabled() -> bool:
    return _switch(MERGE_CHECK_FLAG)


def key_capture_active() -> bool:
    return _KEY_CAPTURE.get()


def any_key_capture_active() -> bool:
    """A prompt key's capture or a decode key's: the two captures of a graph of the lattice's own."""
    return _KEY_CAPTURE.get() or _DECODE_KEY.get() is not None


def capture_tile_marks() -> bool:
    return _KEY_CAPTURE.get() and capture_marks_enabled()


def lazy_fence_sync() -> bool:
    """The sync value of a lazy fence: True everywhere but a key capture, where sync=True is illegal."""
    return not _KEY_CAPTURE.get()


def lazy_fence(core: Any) -> None:
    """An unconditional mark_step(sync=True) site: unchanged outside a key capture; inside, nothing (A) or unsynced (B)."""
    if _KEY_CAPTURE.get() and not capture_marks_enabled():
        return
    core.mark_step(sync=lazy_fence_sync())


def witness_lever(name: str) -> None:
    """Counts a kept lever's branch in a key capture; logged per lever by log_witness. Outside one, nothing."""
    if not _KEY_CAPTURE.get():
        return
    if name not in KEPT_LEVERS:
        raise ValueError(f'witness lever {name!r} is not in KEPT_LEVERS; add it there so its line is logged')
    _WITNESS[name] += 1


def witness_final_norm_mark() -> None:
    """The final norm's mark ran in this forward: counted on a prompt key's lever line, one line for a decode key. Outside a capture, nothing."""
    witness_lever('final_norm_mark')
    decode_key = _DECODE_KEY.get()
    if decode_key is not None:
        LOG.warning('GLM53_KEY_WITNESS decode_key=%s lever=final_norm_mark', decode_key)


def reset_witness() -> None:
    _WITNESS.clear()


_CAPTURE_SCALARS: list[tuple[str, int]] = []


def record_capture_scalar(name: str, value: int) -> int:
    """R6: a host int the sparse indexer read at capture (the dummy's max sequence length, its branch, its scan bucket). Kept in a capture."""
    if any_key_capture_active():
        _CAPTURE_SCALARS.append((name, int(value)))
    return value


def record_capture_branch(max_sequence_length: int, topk_tokens: int, pool_size: int) -> None:
    record_capture_scalar('max_seq', max_sequence_length)
    record_capture_scalar('exact_causal', int(max_sequence_length <= topk_tokens))
    record_capture_scalar('topk_tokens', topk_tokens)
    record_capture_scalar('pool_size', pool_size)


def take_capture_scalars() -> list[tuple[str, int]]:
    taken = list(_CAPTURE_SCALARS)
    _CAPTURE_SCALARS.clear()
    return taken


def _capture_scalar_error(label: str, scalars: list[tuple[str, int]], expected: int) -> str:
    """Why what this capture read is not its dummy's own values; '' when every read of every layer is (or it read nothing).

    A key's route reads a static bound, table width x block size, and scans that whole width: the branch and the scan follow from `expected`.
    """
    values: dict[str, set[int]] = {}
    for name, value in scalars:
        values.setdefault(name, set()).add(value)
    if 'max_seq' not in values:
        return ''
    wanted = {'max_seq': expected}
    if 'topk_tokens' in values:
        wanted['exact_causal'] = int(expected <= max(values['topk_tokens']))
    if 'pool_size' in values:
        wanted['scan_bucket'] = expected // max(values['pool_size'])
    if all(values.get(name, {value}) == {value} for name, value in wanted.items()):
        return ''
    seen = ' '.join(f"{name}={','.join(map(str, sorted(read)))}" for name, read in values.items())
    return (f'{label} capture read {seen} but its dummy is {expected} tokens long, so the host read did not return a real value under '
            'PT_COMPILE_ONLY_MODE and the graph holds the wrong scan and branch')


def check_capture_scalars(label: str, expected_max_sequence: int) -> None:
    """Log what this rank read at the capture and refuse, on every rank, when ranks differ or a read cannot be the dummy's."""
    scalars = take_capture_scalars()
    rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
    LOG.warning('GLM53_CAPTURE_SCALARS key=%s rank=%d %s expected_max_seq=%d', label, rank,
                ' '.join(f'{name}={value}' for name, value in scalars) or 'none', expected_max_sequence)
    reports = rank_reports({'scalars': scalars})
    error = _capture_scalar_error(label, scalars, expected_max_sequence)
    for peer, report in enumerate(reports):
        if report['scalars'] != reports[0]['scalars']:
            error = error or f"{label}: rank {peer} read {report['scalars']} but rank 0 read {reports[0]['scalars']}"
    admit_capture(('capture_scalars', label), 0, 0, error, 'Report this log; a key whose host reads differ per rank cannot share one graph.')


def log_witness(bucket: tuple[int, int, int]) -> None:
    """One line per kept lever, zero included: a lever that did not fire in this capture is a finding."""
    for lever in KEPT_LEVERS:
        LOG.warning('GLM53_KEY_WITNESS key=%d:%d lever=%s captures=%d', bucket[1], bucket[2], lever, _WITNESS[lever])


@contextmanager
def key_capture_context() -> Iterator[None]:
    token = _KEY_CAPTURE.set(True)
    try:
        yield
    finally:
        _KEY_CAPTURE.reset(token)


@contextmanager
def _private_key_bind() -> Iterator[None]:
    """A key graph writes the private top-k like every other prompt forward; decode graphs keep the shared buffer."""
    guard = sys.modules.get('glm53_graph_state')
    if guard is None:
        yield
        return
    with guard.private_key_bind() as bound:
        if bound:
            witness_lever('private_topk')
        yield


@contextmanager
def capture_scope(bucket: tuple[int, int, int]) -> Iterator[None]:
    """What a capture of this bucket runs under on top of capture_context: a key's fences and bind; else nothing."""
    if bucket[0] != 0:
        yield
        return
    reset_witness()
    take_capture_scalars()
    with key_capture_context(), _private_key_bind():
        yield
    log_witness(bucket)


def arm_failed_capture_exit() -> None:
    """After a rank's key capture failed, e.g. PT_DEVMEM. Its peers may be stuck in the graph's collective and never reach the
    gather, so the bounded wait ends in this process's exit, which the executor turns into every worker's stop."""
    if not any_key_capture_active():
        return
    graphs_only.arm_exit('GLM53_KEY_CAPTURE_FAILED')


def check_prompt_key_flags() -> None:
    """Refuse the boot on a key flag that would otherwise do nothing, or fight the graph it asks for."""
    check_decode_lattice_flags()
    keys = prompt_keys()
    from glm53_mixed_step import BATCH_KEYS_FLAG, prompt_batch_keys
    if prompt_batch_keys() and not keys:
        raise ValueError(f'{BATCH_KEYS_FLAG} needs {PROMPT_KEYS_FLAG}: a prompt left alone in its step runs a single-prompt key')
    if prompt_batch_keys() and merge_check_enabled():
        raise ValueError(f'{MERGE_CHECK_FLAG}=1 compares one prompt per chunk; unset it or {BATCH_KEYS_FLAG}')
    for name, on in ((CAPTURE_MARKS_FLAG, capture_marks_enabled()), (MERGE_CHECK_FLAG, merge_check_enabled())):
        if on and not keys:
            raise ValueError(f'{name}=1 only applies to a key capture; set {PROMPT_KEYS_FLAG} or unset {name}')
    check_separate_steps_flags(keys)
    check_small_key_merge(keys)
    check_graphs_only_flags(keys)
    check_query_shard_flags(keys)
    check_query_shard_decode_flags()
    check_rank_check_flags(keys)
    check_tail_fuse_keyed_flags(keys)
    if not keys:
        return
    missing =[name for name in (FLAG, 'VLLM_GAUDI_MIXED_STEP') if os.getenv(name, '0') != '1']
    if missing:
        raise ValueError(f'{PROMPT_KEYS_FLAG} needs {"=1 and ".join(missing)}=1; set them or unset the keys')
    if native_pin_enabled():
        raise ValueError(f'{PROMPT_KEYS_FLAG} and {PIN_FLAG}=1 are two graph shapes; unset {PIN_FLAG}')
    if os.getenv(TAIL_FUSE_FLAG, '0') != '0' and not tail_fuse_keyed_enabled():
        raise ValueError(f'{PROMPT_KEYS_FLAG} has no tail fuse path; set {TAIL_FUSE_FLAG}=0, or {TAIL_FUSE_KEYED_FLAG}=1 to fuse the '
                         'floor128 tail inside the keys')


def _query_shard_errors(keys: tuple[tuple[int, int], ...]) -> list[str]:
    """The settings a query-sharded key needs: it must be captured, replay only, gather into one shared owner, and keep the kpool replicated and the capture-hang fix."""
    errors = [f'{rows}:{width} is not in {PROMPT_KEYS_FLAG}; add it there or drop it here' for rows, width in query_shard_keys() if (rows, width) not in keys]
    return errors + _shard_collective_errors()


def _shard_collective_errors() -> list[str]:
    """What any query shard's all-gather needs of the boot, prompt key or decode key: replay only, one shared owner, the capture-hang fix, a replicated kpool."""
    errors = []
    if not graphs_only_enabled():
        errors.append(f'{GRAPHS_ONLY_FLAG}=1: the gain and the proof are for captured graphs only')
    if os.getenv(SHARED_OWNER_FLAG, '0').strip() != '1':
        errors.append(f'{SHARED_OWNER_FLAG}=1: without one shared owner every layer keeps its own gather output resident in the captured graph')
    if os.getenv(HOLD_TENSORS_FLAG, '') != '0':
        errors.append(f'{HOLD_TENSORS_FLAG}=0: it fixes the width-8192 capture hang, and this adds collectives to the capture')
    if os.getenv(SHARDED_KPOOL_FLAG, '0').strip().lower() in ('1', 'true'):
        errors.append(f'{SHARDED_KPOOL_FLAG}=0: query-shard splits the rows of a replicated kpool, and R590 refuses a sharded one')
    return errors


def check_query_shard_flags(keys: tuple[tuple[int, int], ...]) -> None:
    """Refuse the boot on a listed key that could not shard on every rank; a listed key whose rows do not divide stays replicated and logs that."""
    if not query_shard_keys():
        return
    errors = _query_shard_errors(keys)
    if errors:
        raise ValueError(f'{QUERY_SHARD_FLAG} refuses this boot. Set ' + '; '.join(errors) + f'. Or unset {QUERY_SHARD_FLAG}, and restart all workers.')


def _decode_key_errors() -> list[str]:
    """Why the listed decode keys are not keys of the decode lattice this boot captures: a decode graph exists only for a captured (rows, width)."""
    keys = decode_keys()
    if not keys:
        return [f'{QUERY_SHARD_DECODE_FLAG} shards the decode keys, and no decode lattice is captured: set {DECODE_WIDTHS_FLAG}.']
    widths = sorted({width for _, width in keys}, reverse=True)
    return [f'{rows}:{width} is not a decode key; {DECODE_WIDTHS_FLAG} captures widths {widths} at rows {list(DECODE_ROWS)}.'
            for rows, width in decode_query_shard_keys() if (rows, width) not in keys]


def check_query_shard_decode_flags() -> None:
    """Refuse the boot on a decode shard that could not run: a key the lattice does not capture, or a setting its all-gather needs. The rows
    and the tensor-parallel size are asked at the lattice's admission (decode_shard_tp_error), where the model's TP group exists."""
    if not decode_query_shard_keys():
        return
    problems = _decode_key_errors()
    settings = _shard_collective_errors()
    if settings:
        problems.append('Set ' + '; '.join(settings) + '.')
    if problems:
        raise ValueError(f'{QUERY_SHARD_DECODE_FLAG} refuses this boot. ' + ' '.join(problems) + f' Or unset {QUERY_SHARD_DECODE_FLAG}, and restart all workers.')


def decode_key_tp_error(key: tuple[int, int], tp_size: int) -> str:
    """Why this decode key cannot split its rows over ``tp_size`` ranks, or ''. One and two rows, at TP4, have fewer rows than ranks to give them to."""
    rows, width = key
    if rows % tp_size == 0:
        return ''
    fits = [candidate for candidate in DECODE_ROWS if candidate % tp_size == 0]
    only = f'Only the rows {fits} can' if fits else 'No decode row count can'
    return (f'{rows}:{width} cannot shard: a {rows}-row key does not divide by tensor-parallel size {tp_size}, so a rank would score the rows it scans '
            f'today and add a collective. {only}; drop {rows}:{width} from {QUERY_SHARD_DECODE_FLAG}.')


def decode_shard_tp_error(tp_size: int) -> str:
    """Why the listed decode keys cannot shard on a model of ``tp_size`` ranks, or ''; nothing listed reads ''."""
    keys = decode_query_shard_keys()
    if not keys:
        return ''
    if tp_size < 2:
        return f'{QUERY_SHARD_DECODE_FLAG} shares one scan between ranks and needs tensor-parallel size 2 or more; this model runs on {tp_size}. Unset it.'
    return ' '.join(filter(None, (decode_key_tp_error(key, tp_size) for key in keys)))


def tensor_parallel_size() -> int:
    from vllm.distributed import get_tensor_model_parallel_world_size
    return get_tensor_model_parallel_world_size()


def check_rank_check_flags(keys: tuple[tuple[int, int], ...]) -> None:
    """The check reads replicated rows on a keyed boot; anything else could not give it a verdict."""
    mode = rank_check_mode()
    if not mode:
        return
    if not keys:
        raise ValueError(f'{RANK_CHECK_FLAG}={mode} reads prompt keys; set {PROMPT_KEYS_FLAG} (e.g. 1024:8192,1024:1024) or unset {RANK_CHECK_FLAG}')
    if query_shard_keys():
        raise ValueError(f'{RANK_CHECK_FLAG}={mode} with {QUERY_SHARD_FLAG}: gathered rows match by construction, so equal=1 would prove nothing about the '
                         f'replicated rows the shard relies on. Unset {QUERY_SHARD_FLAG} for the check')
    if mode == 'eager' and graphs_only_enabled():
        raise ValueError(f'{RANK_CHECK_FLAG}=eager runs live chunks on the runner\'s own forward, which {GRAPHS_ONLY_FLAG}=1 refuses. Set {GRAPHS_ONLY_FLAG}=0 for this '
                         'non-serving run, or use the check on replays with 1')
    if mode == '1' and decode_query_shard_keys():
        _note_decode_shard_check()


def _note_decode_shard_check() -> None:
    """The decode shard's lines read equal=1 whatever the ranks computed (they hold the gathered rows), so the boot says where the evidence is instead."""
    keys = ','.join(f'{rows}:{width}' for rows, width in decode_query_shard_keys())
    LOG.warning('GLM53_RANK_CHECK_DECODE_SHARD keys=%s: with %s the decode lines read equal=1 by construction (every rank holds the gathered rows) and prove '
                'nothing about the ranks; the evidence is rows_digest= on each line, compared with an unsharded run of the same fixed batches by '
                'cells/rank_digest_compare.py', keys, QUERY_SHARD_DECODE_FLAG)


def graphs_only_enabled() -> bool:
    return _switch(GRAPHS_ONLY_FLAG)


def _env_int(name: str) -> int:
    raw = os.getenv(name, '')
    return int(raw) if raw.isdigit() else 0


def _shared_key_rows() -> int:
    """The rows of the tallest shared key's whole step: a step's budget may fill it, however short each prompt's share."""
    from glm53_mixed_step import prompt_batch_keys
    return max((prompts * rows for prompts, rows, _ in prompt_batch_keys()), default=0)


def _chunk_row_limits() -> dict[str, int]:
    """The rows each setting lets a prompt chunk carry; 0 = unset, which the scheduler reads as no row limit of its own."""
    from glm53_mixed_step import prefill_budget
    return {SINGLE_TOKENS_FLAG: _env_int(SINGLE_TOKENS_FLAG), MIXED_TOKENS_FLAG: prefill_budget()}


def _graphs_only_errors(key_rows: int) -> list[str]:
    errors = [f'{flag}={wanted} (is {os.getenv(flag)!r})'
              for flag, wanted in ((CELL_BUDGET_FLAG, '0'), (MAX_REQUESTS_FLAG, '1'), (PRIVATE_TOPK_FLAG, '1'),
                                   (RESTORED_DECODE_FLAG, '1'), (DRY_RUN_FLAG, '0'), (DECODE_REPLAY_FLAG, '1'))
              if os.getenv(flag, '') != wanted]
    if not decode_widths():
        errors.append(f'{DECODE_WIDTHS_FLAG} (is empty; with no decode lattice a decode has no key and raises on every rank)')
    errors += [f'{flag}={key_rows} or less (is {rows}; a chunk taller than the tallest key, {key_rows} rows, has no key)'
               for flag, rows in _chunk_row_limits().items()
               if not 0 < rows <= (max(key_rows, _shared_key_rows()) if flag == MIXED_TOKENS_FLAG else key_rows)]
    return errors


def _small_key_route_missing() -> list[str]:
    """What keeps kpool from opening _partial_chunk_merge for a 128-row key: the flag, the partial scan, R590, and an unsharded kpool (tp_size 1)."""
    missing = [f'{flag}=1' for flag in (SMALL_KEY_MERGE_FLAG, PARTIAL_SCAN_TOPK_FLAG, R590_FLAG) if os.getenv(flag, '0') != '1']
    if os.getenv(SHARDED_KPOOL_FLAG, '0').strip().lower() in ('1', 'true'):
        missing.append(f'{SHARDED_KPOOL_FLAG}=0 (a sharded kpool runs tp_size > 1)')
    return missing


def check_small_key_merge(keys: tuple[tuple[int, int], ...]) -> None:
    """A key of 128 rows or fewer takes the legacy indexer route: unless kpool opens the partial merge for it, it selects wrong tokens for any
    row with 1 to 511 complete pools, silently."""
    small = [f'{rows}:{width}' for rows, width in keys if rows <= SMALL_KEY_MAX_ROWS]
    missing = _small_key_route_missing() if small else []
    if missing:
        raise ValueError(f'{PROMPT_KEYS_FLAG} holds {small}, which select wrong tokens unless kpool takes the partial merge for them; '
                         f'it needs {", ".join(missing)}. Set them, or drop those keys')
    if os.getenv(SMALL_KEY_MERGE_FLAG, '0') == '1':
        LOG.warning('GLM53_SMALL_KEY_PARTIAL_MERGE boot keys=%s route=%s', ','.join(small) or 'none',
                    'partial_chunk_merge' if small else 'none (no prompt key of 128 rows or fewer)')


def check_separate_steps_flags(keys: tuple[tuple[int, int], ...]) -> None:
    """VLLM_GAUDI_MIXED_SEPARATE_STEPS=1 (lab, dual_serial_cell.py) with HPU_GLM53_GRAPHS_ONLY=1: the scheduler then sends prompt-only and
    decode-only steps. A prompt-only step replays its key and a decode-only step its own graph, neither kind new to the graphs-only boot,
    but without keys the prompt-only step would run native, so refuse here with the flag that asked for it."""
    from glm53_mixed_step import SEPARATE_STEPS_FLAG, separate_steps_enabled
    if separate_steps_enabled() and graphs_only_enabled() and not keys:
        raise ValueError(f'{SEPARATE_STEPS_FLAG}=1 with {GRAPHS_ONLY_FLAG}=1 needs {PROMPT_KEYS_FLAG} (e.g. 1024:8192,1024:1024): a prompt-only '
                         f'step would run native. Set the keys, or unset {SEPARATE_STEPS_FLAG} or {GRAPHS_ONLY_FLAG}')


def check_graphs_only_flags(keys: tuple[tuple[int, int], ...]) -> None:
    """HPU_GLM53_GRAPHS_ONLY=1: refuse the boot unless every setting that makes a prompt chunk fit a key and a restored decode replay its key
    is in place (the tail fuse and the mixed-step pairing are the key flags' own checks)."""
    if not graphs_only_enabled():
        return
    if not keys:
        raise ValueError(f'{GRAPHS_ONLY_FLAG}=1 needs {PROMPT_KEYS_FLAG} (e.g. 1024:8192,1024:4096,1024:1024): without keys every prompt chunk '
                         f'would run native. Set the keys or unset {GRAPHS_ONLY_FLAG}')
    errors = _graphs_only_errors(max(rows for rows, _ in keys))
    if errors:
        raise ValueError(f'{GRAPHS_ONLY_FLAG}=1 refuses this boot, since a forward with no key would run native or raise. Set '
                         + '; '.join(errors) + f'. Or unset {GRAPHS_ONLY_FLAG}.')


def graphs_only_refusal(reason: str) -> str:
    fix = GRAPHS_ONLY_FIXES.get(reason, 'report this reason with the step that raised it')
    return (f'{GRAPHS_ONLY_FLAG}=1 refused a prompt chunk with no captured key ({reason}): {fix}. Keys: {os.getenv(PROMPT_KEYS_FLAG, "")}. '
            'Nothing runs native under this flag, so every rank raises here; restart all workers after changing the setting.')


def check_graphs_only_keys(runner: Any, buckets: tuple) -> None:
    """GRAPHS_ONLY: the widest key must hold --max-model-len, or a long prompt would have no key and nothing may run native. Else the witness."""
    if not graphs_only_enabled():
        return
    rows, width = max(((rows, width) for _, rows, width in buckets), key=lambda key: (key[1], key[0]))
    tokens = width * runner.attn_block_size
    if tokens < runner.max_model_len:
        needed = 1 << (-(-runner.max_model_len // runner.attn_block_size) - 1).bit_length()
        raise ValueError(f'{GRAPHS_ONLY_FLAG}=1: the widest key {rows}:{width} ({key_estimate_bytes(rows)} bytes estimated) holds {tokens} tokens, '
                         f'short of --max-model-len {runner.max_model_len}, so a longer prompt would have no key. Add {rows}:{needed} to '
                         f'{PROMPT_KEYS_FLAG}; context and KV stay as they are')
    LOG.warning('GLM53_GRAPHS_ONLY keys=%s fallback=refused top_key=%d:%d tokens=%d max_model_len=%d',
                ','.join(f'{rows}:{width}' for _, rows, width in buckets), rows, width, tokens, runner.max_model_len)


def key_estimate_bytes(rows: int) -> int:
    return (KEY_BASE_MIB + KEY_MIB_PER_ROW * rows) * KEY_RESIDENT_PER_MILLE // 1000 * MIB


def keys_holding(keys: Iterable[tuple[int, ...]], needs: tuple[tuple[int, int, str], ...]) -> tuple[list[tuple[int, ...]], str]:
    """The keys whose field `index` is at least `minimum` for every (index, minimum, miss) need, filtered in that order; ([], miss) of the
    first need no key left meets (the selectors' miss reason)."""
    fitting = list(keys)
    for index, minimum, miss in needs:
        fitting = [key for key in fitting if key[index] >= minimum]
        if not fitting:
            return [], miss
    return fitting, ''


def select_shared_prompt_key(prompts: int, rows: int, width: int) -> tuple[tuple[int, int, int] | None, str]:
    """(rows, width, prompts) of the narrowest, then smallest, shared key that holds this chunk; (None, miss reason) else."""
    from glm53_mixed_step import prompt_batch_keys
    fitting, miss = keys_holding(prompt_batch_keys(), ((0, prompts, 'multi_prompt'), (1, rows, 'width_over'), (2, width, 'ctx_over')))
    if miss:
        return None, miss
    prompts_held, rows_held, table = min(fitting, key=lambda key: (key[2], key[0] * key[1]))
    return (rows_held, table, prompts_held), ''


def select_prompt_key(rows: int, width: int) -> tuple[tuple[int, int] | None, str]:
    """(key, '') of the narrowest table width, then the fewest rows, that holds this chunk; (None, miss reason) else."""
    fitting, miss = keys_holding(prompt_keys(), ((0, rows, 'width_over'), (1, width, 'ctx_over')))
    if miss:
        return None, miss
    return min(fitting, key=lambda key: (key[1], key[0])), ''


def rank_reports(report: dict) -> list[dict]:
    """Use the existing Gloo world group; no device allocation or new group."""
    if not torch.distributed.is_initialized():
        return [report]
    from vllm.distributed.parallel_state import get_world_group
    group = get_world_group().cpu_group
    reports = [None] * torch.distributed.get_world_size(group=group)
    torch.distributed.all_gather_object(reports, report, group=group)
    return reports


def admit_capture(bucket: tuple, free: int, required: int, error: str = '', advice: str = BOOT_ADVICE) -> None:
    reports = rank_reports(dict(bucket=bucket, free=free, required=required, error=error))
    errors = []
    for rank, report in enumerate(reports):
        if report['bucket'] != reports[0]['bucket']:
            errors.append(f'rank {rank} bucket disagrees with rank 0')
        if report['error']:
            errors.append(f"rank {rank}: {report['error']}")
        if report['free'] < report['required']:
            errors.append(f"rank {rank}: free HBM {report['free']/MIB:.1f} MiB < "
                          f"required {report['required']/MIB:.1f} MiB")
    if errors:
        raise RuntimeError('mixed graph BOOT admission rejected on ALL ranks: ' + '; '.join(errors) + '. ' + advice)


def capture_requirement(estimate: int) -> int:
    """Calibrated raw ceiling + 50% transient allowance + 2 GiB emergency HBM. A keyed boot checks the reserve only: free HBM is not capacity, since
    capture transient reclaims under pressure. A real PT_DEVMEM decides the capture; the seal's held ballast, a replay under it and InUse budget decide the fit."""
    if prompt_keys():
        return positive_env('VLLM_GAUDI_MIXED_GRAPH_RESERVE_MIB', KEY_RESERVE_DEFAULT_MIB) * MIB
    return (estimate * 3 + 1) // 2 + positive_env('VLLM_GAUDI_MIXED_GRAPH_RESERVE_MIB', 2048) * MIB


def capture_advice() -> str:
    return KEY_CAPTURE_ADVICE if prompt_keys() else BOOT_ADVICE


def _key_setting(bucket: tuple) -> tuple[str, str]:
    """A prompt key as its setting lists it, and that setting: rows:width, a shared key's prompts:rows:width."""
    if len(bucket) > 3:
        from glm53_mixed_step import BATCH_KEYS_FLAG
        return f'{bucket[3]}:{bucket[1]}:{bucket[2]}', BATCH_KEYS_FLAG
    return f'{bucket[1]}:{bucket[2]}', PROMPT_KEYS_FLAG


def _bucket_prompts(bucket: tuple) -> int:
    """1 for a prompt key (0, rows, width); a shared key (0, rows, width, prompts) packs several."""
    return bucket[3] if len(bucket) > 3 else 1


def _bucket_rows(bucket: tuple) -> int:
    """The rows a replay of this key writes: a shared key (prompts:rows:width) gives each of its prompts the key's rows."""
    return bucket[1] * _bucket_prompts(bucket)


def _capture_label(bucket: tuple) -> str:
    return f'prompt key {_key_setting(bucket)[0]}' if bucket[:1] == (0,) else f'bucket {bucket}'


def capture_failure(bucket: tuple, free: int, error: Exception) -> str:
    """Why a capture refuses the boot. A prompt key that ran out of device memory names itself, the free reading and what to drop."""
    detail = f'{type(error).__name__}: {error}'
    if bucket[:1] != (0,) or not graphs_only.is_allocation_failure(error):
        return f'capture failed without retry: {detail}'
    label, setting = _key_setting(bucket)
    return (f'prompt key {label} ran out of device memory in its capture, with {free / MIB:.1f} MiB free HBM before it ({detail}), and there is no retry. '
            f'Drop {label} or another key from {setting}')


def install_entry(adapter: Any, original_forward: Callable) -> None:
    check_prompt_key_flags()
    if enabled():
        traces = ('HPU_GLM53_TRACE_KDA', 'HPU_GLM53_TRACE_MLA_INTERNAL',
                  'HPU_GLM53_TRACE_ACTIVATIONS', 'HPU_GLM53_RUNTIME_TRACE_ACTIVE',
                  'HPU_GLM53_R589_TRACE', 'HPU_GLM53_CAP_LEGACY_WARMUP',
                  'HPU_GLM53_TRUE_HOST_PREFILL_SCAN_BOUND')
        active = [name for name in traces if os.getenv(name, '0').lower() in ('1', 'true', 'yes', 'on')]
        if active or os.getenv('HPU_GLM53_RUNTIME_TRACE_SENTINEL'):
            raise ValueError('mixed graphs require tracing, warmup scan cap and true-host scan bound OFF: '
                             + ', '.join(active))
        adapter._glm53_mixed_graph_forward = original_forward


def positive_env(name: str, default: int) -> int:
    try:
        value = int(os.getenv(name, str(default)))
    except ValueError as exc:
        raise ValueError(f'{name} must be a positive integer') from exc
    if value <= 0:
        raise ValueError(f'{name} must be a positive integer')
    return value


def lattice(block_size: int) -> tuple[tuple[int, int, int], ...]:
    """Context tiers include 40k tokens; units are attention blocks per sequence. Prompt slices 256/512/2048."""
    if block_size < 1:
        raise ValueError('mixed graphs require a positive attention block_size')
    tiers, n = [0], 1
    while n * block_size <= 65536:
        tiers.append(n)
        n *= 2
    decode_rows = (1, 2, 4, 8, 16)
    standard = tuple((d, p, c) for d in decode_rows
                     for p in (256, 512, 2048) for c in tiers if c * block_size >= p)
    if os.getenv('VLLM_GAUDI_MIXED_LONG_1792', '0') == '0':
        return standard
    if os.getenv('VLLM_GAUDI_MIXED_LONG_1792') != '1' or block_size != 128 or not native_pin_enabled():
        raise ValueError('MIXED_LONG_1792 requires value 1, 128-token blocks and native pinning')
    width = 640 if tight_long_context_enabled() else 1024
    return standard + tuple((d, 1792, width) for d in decode_rows)


def tight_long_context_enabled() -> bool:
    return all(os.getenv(name, '0') == '1' for name in (
        'VLLM_GAUDI_MIXED_LONG_1792', 'HPU_GLM53_SPARSE_DECODE_BLOCK_TIER_640',
        'HPU_GLM53_SPARSE_PREFILL_BLOCK_TIER_640',
    ))


ESTIMATE_ENV = 'VLLM_GAUDI_MIXED_GRAPH_ESTIMATE_MIB'


def measured_estimate_bytes() -> int | None:
    """C574 measured graph size for single-bucket lattices; unset keeps the calibrated ceiling."""
    if os.getenv(ESTIMATE_ENV) is None:
        return None
    return positive_env(ESTIMATE_ENV, 1) * MIB


def estimate_bytes(bucket: tuple[int, int, int], block_size: int) -> int:
    """Raw-capture calibrated ceiling, NOT measured cache-disabled residency.

    Anchor: (4,512,256), block_size=128 allocated 5,020,906,496 bytes.
    Keep this ceiling until hardware measures the reduced tensor-cache path;
    VLLM_GAUDI_MIXED_GRAPH_ESTIMATE_MIB replaces it with that measurement.
    """
    measured = measured_estimate_bytes()
    if measured is not None:
        return measured
    d, p, c = bucket[:3]
    if d == 0:
        return key_estimate_bytes(p * _bucket_prompts(bucket))
    proxy = 256 * MIB + (d + p) * MIB + (d + p) * c * (8 + 2 * block_size)
    return (proxy * CALIBRATION_BYTES + CALIBRATION_PROXY - 1) // CALIBRATION_PROXY


def selected_buckets(block_size: int) -> tuple[tuple[int, int, int], ...]:
    spec = os.getenv('VLLM_GAUDI_MIXED_GRAPH_WARMUP', '4:512:256')
    buckets = []
    for item in filter(None, spec.split(',')):
        try:
            bucket = tuple(map(int, item.split(':')))
        except ValueError as exc:
            raise ValueError('mixed warmup must be decode_bs:prompt_tokens:context_blocks,...') from exc
        if bucket not in lattice(block_size):
            raise ValueError(f'mixed warmup bucket {bucket} outside lattice')
        if bucket in buckets:
            raise ValueError(f'duplicate mixed graph warmup bucket {bucket}')
        buckets.append(bucket)
    if not buckets:
        raise ValueError('mixed graph warmup list must contain at least one bucket')
    return tuple(buckets)


def _map(value: Any, tensor: Callable[[torch.Tensor], Any]) -> Any:
    if isinstance(value, torch.Tensor):
        return tensor(value)
    if is_dataclass(value):
        return replace(value, **{f.name: _map(getattr(value, f.name), tensor) for f in fields(value)})
    if isinstance(value, tuple) and hasattr(value, '_fields'):
        return type(value)(*(_map(v, tensor) for v in value))
    if isinstance(value, tuple):
        return tuple(_map(v, tensor) for v in value)
    if isinstance(value, list):
        return [_map(v, tensor) for v in value]
    if isinstance(value, dict):
        return {k: _map(v, tensor) for k, v in value.items()}
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise ValueError(f'unsupported mixed graph metadata {type(value).__name__}; extend explicit schema')


def signature(value: Any) -> tuple:
    tensors = []

    def record(tensor: torch.Tensor) -> tuple:
        tensors.append((tuple(tensor.shape), tensor.dtype, tensor.device))
        return ('tensor', len(tensors) - 1)

    control = _map(value, record)
    return repr(control), tuple(tensors)


def clone_inputs(value: Any) -> Any:
    return _map(value, lambda t: t.detach().clone())


def copy_inputs(target: Any, source: Any) -> None:
    if signature(target) != signature(source):
        raise ValueError('mixed graph input signature changed; choose a different bucket')
    dst, src = [], []
    _map(target, lambda t: dst.append(t))
    _map(source, lambda t: src.append(t))
    for a, b in zip(dst, src):
        a.copy_(b)


def _bucket_prompts(bucket: tuple) -> int:
    """Prompts of a prompt key's bucket: (0, rows, width) holds one, (0, rows, width, prompts) a shared key's."""
    return bucket[3] if len(bucket) > 3 else 1


def _region_shape(region: Any, bucket: tuple | None) -> tuple[int, int]:
    """(padded rows, rows per row of the batch) a region runs at: its own, or its bucket's."""
    if bucket is None:
        return region.batch, region.width
    return (bucket[0], 1) if region.decode else (_bucket_prompts(bucket), bucket[1])


def padded_inputs(batches: list, bucket: tuple | None = None) -> tuple:
    from glm53_mixed_step import MixedMetadata
    regions, tokens, positions, cursor = [], [], [], 0
    for batch in batches:
        r = batch.region
        size, width = _region_shape(r, bucket)
        count = size * width
        metadata = r.metadata if bucket is None else pad_metadata(r, size, width, bucket[2])
        # Host ownership/positions are deliberately absent from the graph key.
        # Tensor metadata retains real physical slots, masks, and sequence lengths.
        regions.append(replace(r, requests=tuple(range(cursor, cursor+size)),
                               lengths=(width,) * size, starts=(0,) * size,
                               batch=size, width=width, metadata=metadata,
                               valid=tuple(range(count)), graph_padded=True))
        tokens.append(pad_tensor(_key_rows(batch.tokens[:size], width), (size, width)))
        positions.append(pad_tensor(_key_rows(batch.positions[:size], width), (size, width)))
        cursor += size
    return (MixedMetadata(tuple(regions)),
            torch.cat([t.reshape(-1) for t in tokens]).unsqueeze(0),
            torch.cat([t.reshape(-1) for t in positions]).unsqueeze(0))


def compact_hidden(hidden: torch.Tensor, batches: list, bucket: tuple | None = None) -> torch.Tensor:
    from glm53_mixed_step import unpack_region
    hidden = hidden.reshape(-1, hidden.shape[-1])
    parts, cursor = [], 0
    for batch in batches:
        r = batch.region
        size, width = _region_shape(r, bucket)
        count = size * width
        local = hidden[cursor:cursor+count].reshape(size, width, -1)
        if size > 1 and not r.decode:  # each prompt's row of the key: the runner's own row width can differ from the key's
            parts.extend(local[row, :length] for row, length in enumerate(r.lengths))
        else:
            parts.append(unpack_region(local[:r.batch, :r.width].reshape(-1, hidden.shape[-1]), r))
        cursor += count
    return torch.cat(parts)


def select_bucket(batches: list, block_size: int) -> tuple[tuple | None, str]:
    if not batches or not batches[0].region.decode:
        return None, 'no_decode'
    if len(batches) != 2 or batches[1].region.decode:
        return None, 'multi_prompt'
    d, p = (b.region for b in batches)
    if p.batch != 1 or len(p.requests) != 1:
        return None, 'multi_prompt'
    tables = [getattr(r.metadata, 'glm53_index_block_table', None) for r in (d, p)]
    if any(t is None or t.ndim != 2 for t in tables):
        return None, 'signature'
    needed = (d.batch, p.width, max(t.shape[1] for t in tables))
    candidates = list(selected_buckets(block_size))
    if exact_shapes_enabled():
        candidates = [b for b in candidates if b[:2] == needed[:2]]
        if not candidates:
            return None, 'shape_mismatch'
    for axis, reason in enumerate(('decode_rows_over', 'width_over', 'ctx_over')):
        candidates = [b for b in candidates if b[axis] >= needed[axis]]
        if not candidates:
            return None, reason
    return min(candidates, key=lambda b: (estimate_bytes(b, block_size), b)), ''


def bucket_for(batches: list, block_size: int) -> tuple | None:
    return select_bucket(batches, block_size)[0]


def _key_rows(value: torch.Tensor, width: int) -> torch.Tensor:
    """A key's rows of a batch the runner padded wider than them (its own width past the chunk's real rows is all padding)."""
    return value[:, :width] if prompt_keys() else value


def pad_tensor(value: torch.Tensor, shape: tuple, fill: int = 0) -> torch.Tensor:
    if value.ndim != len(shape) or any(a > b for a, b in zip(value.shape, shape)):
        raise ValueError(f'mixed metadata shape {tuple(value.shape)} exceeds bucket {shape}')
    result = value.new_full(shape, fill)
    result[tuple(slice(0, n) for n in value.shape)] = value
    return result


def pad_token_field(value: torch.Tensor, region: Any, size: int, width: int,
                    fill: int) -> torch.Tensor:
    valid = len(region.requests)
    shaped = value.reshape(region.batch, region.width)[:valid]
    # Ignore the runner's original padding addresses, which use several conventions.
    mask = torch.arange(region.width, device=value.device)[None, :] < torch.tensor(
        region.lengths, device=value.device)[:, None]
    shaped = torch.where(mask, shaped, shaped.new_full((), fill))
    result = pad_tensor(_key_rows(shaped, width), (size, width), fill)
    return result if value.ndim == 2 else result.reshape(-1)


def prompt_start_locs(value: torch.Tensor, region: Any, size: int, width: int) -> torch.Tensor:
    """Real cumulative lengths of a key's rows, pad rows at length 0. hpu_causal_conv1d_fn takes each row's length from these and saves new
    conv state at that column; a uniform width would save the zero-masked padding. The conv reads them as tensor ops only, so this stays a
    live input of the captured graph (StaticCall marks it) and every replay brings the chunk's own lengths."""
    if any(length > width for length in region.lengths):
        raise ValueError(f'a prompt row of {max(region.lengths)} rows does not fit a key of {width} rows')
    lengths = (*region.lengths, *(0,) * (size - len(region.lengths)))
    return torch.tensor((0, *accumulate(lengths)), dtype=value.dtype, device=value.device)


PAD_TOKEN_FIELDS = ('slot_mapping', 'glm53_index_slot_mapping', 'glm53_tail_slot_mapping', 'glm53_token_mask', 'padding_mask_flat')
PAD_STATE_INDEX_FIELDS = ('load_indices_tensor', 'store_indices_tensor', 'non_spec_state_indices_tensor')
PAD_ROW_FIELDS = ('seq_lens_tensor', 'context_lens_tensor', 'has_initial_states_p')
PAD_START_FIELDS = ('query_start_loc', 'query_start_loc_p', 'non_spec_query_start_loc')


def _pad_block_table(value: torch.Tensor, region: Any, size: int, ctx: int) -> torch.Tensor:
    """The index block table at the key's size rows of ctx columns: -1 pads under prompt keys; a native-pinned table is already its rows."""
    if prompt_keys():
        return pad_tensor(value[:len(region.requests)], (size, ctx), -1)
    if not native_pin_enabled():
        return pad_tensor(value[:len(region.requests)], (size, ctx))
    if value.ndim != 2 or value.shape[0] != size:
        raise ValueError(f'native-pinned block table {tuple(value.shape)} is not {size} native rows')
    return value


def _pad_start_locs(value: torch.Tensor, region: Any, size: int, width: int) -> torch.Tensor:
    if region.decode:  # a decode row is one real token (MixedMetadata refuses any other), so uniform is its real length
        return torch.arange(size + 1, dtype=value.dtype, device=value.device) * width
    return prompt_start_locs(value, region, size, width) if prompt_keys() else value


def pad_field(name: str, value: Any, region: Any, size: int, width: int, *, ctx: int) -> Any:
    """One metadata field of `region` padded to the key's `size` rows of `width` tokens (`ctx`: the key's table columns)."""
    if name == 'num_actual_tokens':
        return size * width
    if name == 'num_decodes':
        return size if region.decode else 0
    if not isinstance(value, torch.Tensor):
        return value
    if name == 'glm53_index_block_table':
        return _pad_block_table(value, region, size, ctx)
    if name in PAD_TOKEN_FIELDS:
        return pad_token_field(value, region, size, width, -1 if 'slot_mapping' in name else 0)
    if name in PAD_STATE_INDEX_FIELDS:
        return pad_tensor(value[..., :len(region.requests)], (*value.shape[:-1], size), -1)
    if name in PAD_ROW_FIELDS:
        return pad_tensor(value[:len(region.requests)], (size, *value.shape[1:]))
    if name in PAD_START_FIELDS:
        return _pad_start_locs(value, region, size, width)
    return value


def pad_metadata(region: Any, size: int, width: int, ctx: int) -> Any:
    """Keep GLM's direct sparse/KDA inputs; discard unused dense-PA scratch.

    -1 state slots map to the allocation-owned final garbage row in _forward_hpu.
    -1 sparse/Q4 slots map to their allocation-owned final garbage rows. No new
    recurrent slots, KV pages or offload request ownership are allocated here.
    """
    metadata = region.metadata
    names = [f.name for f in fields(metadata)] if is_dataclass(metadata) else metadata._fields
    unused = {'attn_bias', 'block_list', 'block_mapping', 'block_usage', 'block_groups',
              'blocks_caching_range', 'mamba_chunks_to_block_mapping', 'seqlens_offsets_for_blocks',
              'last_chunk_indices_p', 'prep_initial_states'}
    updates = {name: (None if name in unused else pad_field(
        name, getattr(metadata, name), region, size, width, ctx=ctx)) for name in names}
    return replace(metadata, **updates) if is_dataclass(metadata) else metadata._replace(**updates)


def write_q4_rows(cache: torch.Tensor, slots: torch.Tensor, packed: torch.Tensor) -> None:
    """Unsharded counterpart of Q4's existing TP allocation-owned dummy row."""
    valid = slots >= 0
    targets = torch.where(valid, slots.long(), torch.full_like(slots, cache.shape[0] - 1).long())
    values = torch.where(valid.unsqueeze(-1), packed, torch.zeros_like(packed))
    cache.index_copy_(0, targets, values)


SPLIT_FLAG = 'VLLM_GAUDI_MIXED_SPLIT_FORWARD'


def split_forward_enabled() -> bool:
    """C574, opt-in: one model forward per region inside the captured mixed graph.

    Token-wise ops (GEMMs, MoE router and experts, mhc, norms) then see each region's own padded rows, 16 decode
    and 256 prompt, the row classes of the native decode graph and the native (1, 256) prompt forward, instead of
    one shared M = 272 call whose recipes match neither (C165 / C293 / C573).
    """
    value = os.getenv(SPLIT_FLAG, '0')
    if value not in ('0', '1'):
        raise ValueError(f'{SPLIT_FLAG}={value!r} must be 0 or 1; unset it (or 0) to keep one shared forward '
                         'per mixed graph, 1 to run decode and prompt rows as separate forwards')
    return value == '1'


EXACT_FLAG = 'VLLM_GAUDI_MIXED_EXACT_SHAPES'


def exact_shapes_enabled() -> bool:
    """C576, opt-in: serve a mixed step only at the padded shapes native runs it at; a miss runs native.

    With the split forward each region then keeps its native row count end to end, MoE algorithm included
    (hpu_fp8 sends > 512 rows to the R589 dense tile, 9-512 rows to the grouped capacity path, <= 8 rows to the
    routed op): bucket d must equal the padded decode batch and bucket p the padded chunk width. The miss path is
    the runner's native per-batch forward (glm53_mixed_step.prepare), never the lazy mixed forward (C570).
    """
    value = os.getenv(EXACT_FLAG, '0')
    if value not in ('0', '1'):
        raise ValueError(f'{EXACT_FLAG}={value!r} must be 0 or 1; unset it (or 0) to pad to the smallest '
                         'fitting bucket, 1 to serve only the native padded shapes')
    if value == '1' and not split_forward_enabled():
        raise ValueError(f'{EXACT_FLAG}=1 needs {SPLIT_FLAG}=1: a shared forward mixes the regions\' row counts '
                         'whatever the bucket; set both, or unset both')
    return value == '1'


PIN_FLAG = 'VLLM_GAUDI_MIXED_NATIVE_PIN'
WIDTHS_FLAG = 'VLLM_GAUDI_MIXED_NATIVE_WIDTHS'


def native_pin_enabled() -> bool:
    """Nativepin, opt-in: each region keeps native's block table and logits rows (C573 follow-up to C576).

    Native builds glm53_index_block_table at a power-of-two width of the batch's own blocks with -1 fill
    (hpu_model_runner _make_glm53_sparse_metadata_cpu) and computes logits per batch on hidden[logits_indices];
    the graph otherwise pads every table to the bucket's c blocks with 0 and joins the logits rows. A captured graph
    has one static width, so BOOT captures one graph per (decode width, prompt width) pair in WIDTHS_FLAG and a live
    step at any other width runs native.
    """
    value = os.getenv(PIN_FLAG, '0')
    if value not in ('0', '1'):
        raise ValueError(f'{PIN_FLAG}={value!r} must be 0 or 1; unset it (or 0) to pad tables to the bucket, '
                         '1 to keep the native block table and logits rows')
    if value == '1' and not exact_shapes_enabled():
        raise ValueError(f'{PIN_FLAG}=1 needs {EXACT_FLAG}=1 (and {SPLIT_FLAG}=1): only native-shaped regions have '
                         'a native block table to keep; set all three, or unset the pin')
    return value == '1'


def native_widths(context_blocks: int) -> tuple[tuple[int, int], ...]:
    """(decode width, prompt width) pairs BOOT captures for a bucket; unset = the bucket's width for both regions."""
    spec = os.getenv(WIDTHS_FLAG)
    if spec is None:
        return ((context_blocks, context_blocks),)
    pairs = []
    for item in filter(None, spec.split(',')):
        parts = item.split(':')
        if len(parts) != 2 or not all(part.isdigit() for part in parts):
            raise ValueError(f'{WIDTHS_FLAG} must list decode_width:prompt_width pairs (e.g. 32:16,32:32); '
                             f'got {item!r}')
        pair = (int(parts[0]), int(parts[1]))
        if pair in pairs:
            raise ValueError(f'duplicate {WIDTHS_FLAG} pair {item!r}; list each width pair once')
        pairs.append(pair)
    if not pairs:
        raise ValueError(f'{WIDTHS_FLAG} is set but lists no width pair; unset it or name at least one')
    return tuple(pairs)


def check_width_pair(bucket: tuple[int, int, int], pair: tuple[int, int], block_size: int, page_size: int) -> None:
    """Refuse a (decode, prompt) width pair native can never build, or warmup cannot dummy, for this bucket."""
    _, p, c = bucket
    for width in pair:
        tight_width = width == c == 640 and p == 1792 and tight_long_context_enabled()
        if width < 1 or (width & (width - 1) and not tight_width):
            raise ValueError(f'{WIDTHS_FLAG} width {width} is not a power of two or an enabled 640-block long key')
        if width > c:
            raise ValueError(f'{WIDTHS_FLAG} width {width} exceeds the bucket\'s {c} context blocks of '
                             f'{bucket}; raise the bucket or drop the pair')
    if pair[1] * block_size < p:
        raise ValueError(f'{WIDTHS_FLAG} prompt width {pair[1]} cannot hold the {p}-token chunk of '
                         f'{bucket}; the smallest native width is {-(-p // block_size)} blocks')
    if (pair[1] * block_size - p) % page_size:
        raise ValueError(f'{WIDTHS_FLAG} prompt width {pair[1]}: its dummy context {pair[1] * block_size - p} tokens '
                         f'must align to runner.block_size {page_size} (the bucket rule in validate_warmup)')


def check_native_widths(buckets: tuple, block_size: int, page_size: int) -> int:
    """Validate every bucket's width pairs; returns the number of graphs BOOT will capture."""
    graphs = 0
    for bucket in buckets:
        for pair in native_widths(bucket[2]):
            check_width_pair(bucket, pair, block_size, page_size)
            graphs += 1
    limit = positive_env('VLLM_GAUDI_MIXED_GRAPH_MAX_GRAPHS', 4)
    if graphs > limit:
        raise ValueError(f'nativepin warmup needs {graphs} graphs (buckets x {WIDTHS_FLAG} pairs) but '
                         f'VLLM_GAUDI_MIXED_GRAPH_MAX_GRAPHS={limit}; raise it after HBM measurement or drop pairs')
    return graphs


def native_width_pair(batches: list) -> tuple[int, ...]:
    """(decode, prompt) block-table widths of a mixed step, as the nativepin graph key and log label carry them."""
    return tuple(int(b.region.metadata.glm53_index_block_table.shape[1]) for b in batches)


def warmup_scenarios(runner: Any, buckets: tuple) -> list[tuple[tuple, tuple]]:
    """(prompt_cfg, decode_cfg) per BOOT capture: decode dummies at wd*block-1 tokens, the chunk ending at wp*block."""
    if prompt_keys():
        return [((_bucket_prompts(bucket), bucket[1], (bucket[2] * runner.attn_block_size - bucket[1]) // runner.block_size), None)
                for bucket in buckets]
    pinned = native_pin_enabled()
    scenarios = []
    for d, p, c in buckets:
        rows = warmup_decode_rows(runner, d)
        for decode_width, prompt_width in (native_widths(c) if pinned else ((c, c),)):
            context = prompt_width * runner.attn_block_size - p
            scenarios.append(((1, p, context // runner.block_size), (rows, 1, decode_width)))
    return scenarios


def miss_event() -> str:
    """Live misses run native under exact shapes; BOOT and flag-off misses keep the lazy mixed forward."""
    return 'native_fallbacks' if exact_shapes_enabled() and not _BOOT.get() else 'lazy_fallbacks'


class RegionPlan(Protocol):
    """glm53_mixed_step.MixedMetadata as region_forwards reads it (a frozen dataclass: replace() rebuilds it)."""

    regions: tuple


def region_forwards(forward: Callable[..., torch.Tensor], plan: RegionPlan, tokens: torch.Tensor,
                    positions: torch.Tensor) -> torch.Tensor:
    """Each region of a padded plan through its own forward; rows come back flat, in plan order."""
    parts, cursor = [], 0
    for region in plan.regions:
        span = slice(cursor, cursor + region.count)
        hidden = forward(input_ids=tokens[:, span], positions=positions[:, span],
                         attn_metadata=replace(plan, regions=(region,)))
        parts.append(hidden.reshape(-1, hidden.shape[-1]))
        cursor += region.count
    return torch.cat(parts)


class StaticCall:
    """Keep all metadata tensors explicit inputs to the bridge's lifetime analysis."""

    def __init__(self, forward: Callable, static: tuple) -> None:
        self.forward, self.static = forward, static
        self.split = split_forward_enabled()
        tensors = []
        _map(static, lambda tensor: tensors.append(tensor))
        self.tensor_inputs = tuple(tensors)

    def __call__(self) -> torch.Tensor:
        plan, tokens, positions = self.static
        if self.split:
            return region_forwards(self.forward, plan, tokens, positions)
        return self.forward(input_ids=tokens, positions=positions, attn_metadata=plan)


class CaptureModule(torch.nn.Module):
    def __init__(self, call: Callable) -> None:
        super().__init__()
        self.call = call

    def forward(self, *tensor_inputs: torch.Tensor) -> torch.Tensor:
        # These are the same static tensors used by call, marked as user inputs.
        return self.call()


def bridge_policy_error() -> str:
    if os.getenv('PT_HPUGRAPH_DISABLE_TENSOR_CACHE', '1') != '1':
        return 'PT_HPUGRAPH_DISABLE_TENSOR_CACHE must be unset or 1'
    if os.getenv('PT_HPU_LAZY_MODE', '1') != '1':
        return 'PT_HPU_LAZY_MODE must be 1'
    return ''


class HpuBackend:
    """Runner's bridge wrapper, with dry capture and retained in-place state."""

    capture_bucket: tuple = ()
    free_before: int = 0

    def prepare_capture(self, bucket: tuple, estimate: int, error: str) -> None:
        """Admit a capture on every rank. The reserve is checked against the allocator's Limit - InUse, where PT_DEVMEM aborts the worker;
        device free HBM (mem_get_info) is logged beside it and names the reading in a capture failure."""
        free, headroom, required = 0, 0, 0
        try:
            import habana_frameworks.torch as htorch
            error = error or bridge_policy_error()
            if not _BOOT.get():
                error = error or 'first-hit capture outside BOOT is forbidden; warm this bucket'
            htorch.core.mark_step()
            htorch.hpu.synchronize()
            free, total = htorch.hpu.mem_get_info()
            if not isinstance(free, int) or not 0 <= free <= total:
                raise ValueError('mem_get_info returned invalid free/total bytes')
            headroom, required = _allocator_headroom(), capture_requirement(estimate)
        except Exception as exc:
            error = error or f'free memory query failed: {type(exc).__name__}: {exc}'
        if not error and headroom < required:
            error = (f'{_capture_label(bucket)} has {headroom / MIB:.1f} MiB under the allocator limit (Limit - InUse, '
                     f'{free / MIB:.1f} MiB device free), under the {required / MIB:.1f} MiB it needs to capture')
        admit_capture(bucket, 0, 0, error, capture_advice())
        self.capture_bucket, self.free_before = bucket, free
        LOG.warning('GLM53_MIXED_GRAPH admitted bucket=%s free_bytes=%s headroom_bytes=%s required_bytes=%s',
                    bucket, free, headroom, required)

    def capture(self, call: Callable[[], torch.Tensor]) -> tuple:
        import habana_frameworks.torch as htorch
        error = bridge_policy_error()
        if error:
            raise RuntimeError(error)
        handle, output, error = None, None, ''
        try:
            htorch.core.mark_step()
            htorch.hpu.synchronize()
            before = htorch.hpu.memory_allocated()
            free_before, _ = htorch.hpu.mem_get_info()
            htorch.hpu.reset_peak_memory_stats()
            module = htorch.hpu.wrap_in_hpu_graph(
                CaptureModule(call), disable_tensor_cache=True, dry_run=True,
                free_inplace=False, max_graphs=1)
            handle = module, call.tensor_inputs
            output = module(*call.tensor_inputs)
            htorch.hpu.synchronize()
            self.last_capture_bytes = max(0, htorch.hpu.memory_allocated() - before)
            self.free_after, _ = htorch.hpu.mem_get_info()
            self.last_capture_hbm_bytes = max(0, free_before - self.free_after)
            self.last_capture_peak_bytes = max(0, htorch.hpu.max_memory_allocated() - before)
        except Exception as exc:
            error = capture_failure(self.capture_bucket, self.free_before, exc)
            arm_failed_capture_exit()
        admit_capture((), 0, 0, error, capture_advice())
        return handle, output

    def replay(self, handle: Any) -> None:
        import habana_frameworks.torch as htorch
        htorch.core.mark_step()
        module, tensor_inputs = handle
        module(*tensor_inputs)


HBM_PROBE = 'VLLM_GAUDI_MIXED_GRAPH_HBM_PROBE'


def hbm_probe_enabled() -> bool:
    """C568 log-only HBM probe switch. Unset or 0 keeps the plain backend; any other value refuses the boot."""
    value = os.getenv(HBM_PROBE, '0').strip()
    if value not in ('0', '1'):
        raise ValueError(f'{HBM_PROBE}={value!r}: set 1 to log mixed-graph HBM readings, or 0 / unset for none')
    return value == '1'


def default_backend() -> HpuBackend | HbmProbe:
    return HbmProbe(HpuBackend()) if hbm_probe_enabled() else HpuBackend()


class HbmProbe:
    """Log-only composition around the bridge backend: the same calls in the same order, plus allocator readings.

    capture logs the MarkStepCensus of the graph; replay logs the allocated bytes before it, its own peak, and the
    peak since the previous replay (native prompt-only and decode steps), resetting the peak after every replay.
    """

    def __init__(self, backend: Any) -> None:
        self.backend = backend
        self.bucket: tuple = ()
        self.buckets: dict[int, tuple] = {}
        self.replays = self.max_transient = self.max_between = 0

    def __getattr__(self, name: str) -> Any:
        """GraphCache's getattr reads of the backend's capture measurements (last_capture_bytes, free_after...)."""
        backend = self.__dict__.get('backend')
        if backend is None:
            raise AttributeError(name)
        return getattr(backend, name)

    def prepare_capture(self, bucket: tuple, estimate: int, error: str) -> None:
        self.bucket = bucket
        self.backend.prepare_capture(bucket, estimate, error)

    def capture(self, call: Callable[[], torch.Tensor]) -> tuple:
        import habana_frameworks.torch as htorch
        with MarkStepCensus(htorch.core, htorch.hpu) as census:
            try:
                handle, output = self.backend.capture(call)
            except Exception:
                LOG.warning('GLM53_MIXED_GRAPH hbm_probe capture_failed bucket=%s %s', self.bucket, census.summary())
                raise
        self.buckets[id(handle)] = self.bucket
        LOG.warning('GLM53_MIXED_GRAPH hbm_probe capture bucket=%s %s', self.bucket, census.summary())
        htorch.hpu.reset_peak_memory_stats()  # the first replay's between-peak must not carry the capture peak
        return handle, output

    def replay(self, handle: Any) -> None:
        import habana_frameworks.torch as htorch
        hpu = htorch.hpu
        between, before = hpu.max_memory_allocated(), hpu.memory_allocated()
        hpu.reset_peak_memory_stats()
        self.backend.replay(handle)
        peak, after = hpu.max_memory_allocated(), hpu.memory_allocated()
        hpu.reset_peak_memory_stats()
        self.log_replay(self.buckets.get(id(handle), ()), before, peak, after, between)

    def log_replay(self, bucket: tuple, before: int, peak: int, after: int, between: int) -> None:
        self.replays += 1
        transient, gap = peak - before, between - before
        new_max = transient > self.max_transient or gap > self.max_between
        self.max_transient, self.max_between = max(self.max_transient, transient), max(self.max_between, gap)
        every = positive_env('VLLM_GAUDI_MIXED_GRAPH_LOG_EVERY', 100)
        if self.replays <= 16 or new_max or self.replays % every == 0:
            LOG.warning('GLM53_MIXED_GRAPH hbm_probe replay n=%d bucket=%s before_bytes=%d replay_peak_bytes=%d '
                        'after_bytes=%d between_peak_bytes=%d max_transient_bytes=%d max_between_bytes=%d',
                        self.replays, bucket, before, peak, after, between, self.max_transient, self.max_between)


class KeyMerge:
    """HPU_GLM53_PROMPT_KEY_MERGE_CHECK=1 (K1): chunks of a repeated prompt run narrow, then widest; compares hidden.

    Live only. The chunk at start 0 and every chunk at start MERGE_MIN_PAST_TOP_K_START or later: below that, top-k keeps every
    token, so only those later chunks can show a width-dependent selection. A repeat of the same tokens at the same start runs
    the widest key of its rows instead of its own, and one GLM53_K1_MERGE line says whether the two hidden tensors are
    torch.equal. The next repeat starts over. Every rank must take the same key: the choice is gathered, and ranks that
    disagree all stop.
    """

    def __init__(self) -> None:
        self.pending: dict[bytes, torch.Tensor] = {}
        self.references: dict[tuple[bytes, tuple[int, int]], torch.Tensor] = {}

    @staticmethod
    def _covered(batch: Any) -> bool:
        start = batch.region.starts[0]
        return merge_check_enabled() and not _BOOT.get() and (start == 0 or start >= MERGE_MIN_PAST_TOP_K_START)

    @classmethod
    def _wide_key(cls, batch: Any, natural: tuple[int, int]) -> tuple[int, int] | None:
        if not cls._covered(batch):
            return None
        widest = max(key[1] for key in prompt_keys() if key[0] == natural[0])
        return (natural[0], widest) if widest > natural[1] else None

    @staticmethod
    def _digest(batch: Any) -> bytes:
        tokens = batch.tokens[0, :batch.region.lengths[0]].cpu().numpy().tobytes()
        return hashlib.sha256(str(batch.region.starts[0]).encode() + b':' + tokens).digest()

    def key_for(self, batch: Any, natural: tuple[int, int]) -> tuple[int, int]:
        wide = self._wide_key(batch, natural)
        if wide is None:
            return natural
        digest = self._digest(batch)
        use_wide = digest in self.pending
        reports = rank_reports({'digest': digest.hex(), 'wide': use_wide})
        for rank, report in enumerate(reports):
            if report != reports[0]:
                raise RuntimeError(f'merge check: rank {rank} chose {report}, rank 0 {reports[0]}; every rank must run one key. '
                                   f'Unset {MERGE_CHECK_FLAG} and report the prompt chunk at start {batch.region.starts[0]}')
        return wide if use_wide else natural

    def _compare_self(self, digest: bytes, batch: Any, chosen: tuple[int, int], snapshot: torch.Tensor) -> None:
        """GLM53_K1_SELF: this replay against the same key's first output for these tokens. Log only."""
        reference = self.references.get((digest, chosen))
        if reference is None:
            if len(self.references) < SELF_MAX_REFERENCES:
                self.references[(digest, chosen)] = snapshot
                self._save_first(batch, chosen, snapshot)
            return
        LOG.warning('GLM53_K1_SELF rows=%d width=%d start=%d torch_equal=%d', chosen[0], chosen[1], batch.region.starts[0],
                    int(torch.equal(reference, snapshot)))

    @staticmethod
    def _save_first(batch: Any, chosen: tuple[int, int], snapshot: torch.Tensor) -> None:
        """A key's first output for a chunk, for a merge across boots when two keys cannot share one. Needs the K1 trace dir."""
        directory = os.environ.get(STALL_DIR_ENV, '').strip()
        if directory:
            rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
            name = f'hidden-r{rank}-{chosen[0]}-{chosen[1]}-{batch.region.starts[0]}.pt'
            torch.save(snapshot, os.path.join(directory, name))

    def observe(self, batch: Any, natural: tuple[int, int], chosen: tuple[int, int], hidden: torch.Tensor) -> None:
        if not self._covered(batch):
            return
        snapshot, digest = hidden.detach().float().cpu(), self._digest(batch)
        self._compare_self(digest, batch, chosen, snapshot)
        if self._wide_key(batch, natural) is None:
            return
        if chosen == natural:
            while len(self.pending) >= MERGE_MAX_PENDING:
                del self.pending[next(iter(self.pending))]
            self.pending[digest] = snapshot
            return
        LOG.warning('GLM53_K1_MERGE narrow=%d wide=%d rows=%d start=%d torch_equal=%d', natural[1], chosen[1], natural[0],
                    batch.region.starts[0], int(torch.equal(self.pending.pop(digest), snapshot)))


def _rank_check_fence() -> None:
    """The replay's last recipe has landed, so a host read sees its writes. Host side: no op joins the graph."""
    import habana_frameworks.torch as htorch
    htorch.core.mark_step()
    htorch.hpu.synchronize()


def _rank_check_is_logger() -> bool:
    return not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0


def _row_digests(rows: torch.Tensor) -> list[bytes]:
    return [hashlib.blake2b(row.tobytes(), digest_size=16).digest() for row in rows.numpy()]


def _layer_entry(host: torch.Tensor, digests: list[bytes], previous: list[bytes] | None, window: int, packed_rows: int | None = None) -> dict:
    """What a replay left in the first ``window`` rows: their digests, how many hold indices (live), how many of those are the previous
    replay's, byte for byte (stale; a dead row is padding, never stale), and how many rows are on the wrong side of ``packed_rows``.
    ``packed_rows``: the chunk's rows when they come first and padding follows (one prompt in the key), else None."""
    window = min(window, len(digests))
    alive = (host[:window] >= 0).any(dim=1).tolist()
    stale = sum(live and digests[row] == previous[row] for row, live in enumerate(alive)) if previous else 0
    misplaced = 0 if packed_rows is None else sum(live != (row < packed_rows) for row, live in enumerate(alive))
    return {'digests': digests[:window], 'live': sum(alive), 'stale': stale, 'misplaced': misplaced}


def _rows_report(rows: torch.Tensor) -> dict:
    """A DMA read to the host: the whole buffer, no slice op on the device."""
    host = rows.to('cpu')
    return _layer_entry(host, _row_digests(host), None, host.shape[0])


def _layer_label(op: Any) -> str:
    prefix = str(getattr(op.k_cache, 'prefix', ''))
    found = RANK_CHECK_LAYER.search(prefix)
    return found.group(1) if found else prefix


def _rank_check_tp_size() -> int:
    """The tensor-parallel size the gathered reports must cover. The indexer's own tp_size is 1 with a replicated kpool, so it cannot say."""
    return tensor_parallel_size()


def _exchange_rows(where: tuple, read: Callable[[], list[dict]]) -> list[dict]:
    """Every rank reaches the gloo gather whatever its own read did: a raise here would leave the peers in it."""
    report = {'where': where, 'error': '', 'tp_size': None, 'layers': []}
    try:
        report['tp_size'] = _rank_check_tp_size()
        report['layers'] = read()
    except Exception as error:  # reported on every rank as NOT_RETAINED; the serving forward already finished
        report['error'] = f'{type(error).__name__}: {error}'
    return rank_reports(report)


def _unwritten_rows(layer: dict, rank: int) -> str:
    """Why one rank's rows are not all this replay's, or ''. A row no replay wrote matches on every rank, so it must not read as equal.
    ``expected`` is the chunk's real rows: the padding below them is dead (-1), so more or fewer live rows means a row kept an older value or never got one;
    ``misplaced`` is a real row dead or a padding row live, which the count alone cannot see."""
    if not layer['live']:
        return f'no_live_rows on rank {rank}'
    expected = layer.get('expected')
    if expected is not None and layer['live'] != expected:
        return f"live_rows={layer['live']} on rank {rank} but the chunk has {expected} real rows: part of the buffer was not written by this replay"
    if layer['misplaced']:
        return f"misplaced_rows={layer['misplaced']} on rank {rank}: a real row is dead or a padding row is live, so these rows are not this chunk's"
    if layer['stale']:
        return f"stale_rows={layer['stale']} of {layer['live']} live rows on rank {rank}: this replay left them as the previous replay wrote them"
    return ''


def _ranks_missing(reports: list[dict]) -> str:
    """Why these reports do not cover the model's tensor-parallel ranks, or ''. A rank that is missing, or only half of them, agrees with itself."""
    size = reports[0].get('tp_size')
    if len(reports) >= 2 and all(report.get('tp_size') == len(reports) for report in reports):
        return ''
    return (f'ranks={len(reports)} of tp_size={size}: the rows of every tensor-parallel rank (2 or more) must meet in one gather, or the missing ones '
            'agree with themselves; run the check with all of them in one process group')


def _rank_check_void(reports: list[dict], index: int) -> str:
    """Why these rows prove nothing, or ''. Ranks that are missing, an unwritten row and a row on the wrong side of the padding all match across ranks."""
    for rank, report in enumerate(reports):
        if report['where'] != reports[0]['where']:
            return f"ranks not in lockstep: rank {rank} at {report['where']}, rank 0 at {reports[0]['where']}"
        if report['error']:
            return f"rank {rank}: {report['error']}"
    missing = _ranks_missing(reports)
    if missing:
        return missing
    for rank, report in enumerate(reports):
        unwritten = _unwritten_rows(report['layers'][index], rank)
        if unwritten:
            return unwritten
    return ''


def _window_tag(row_digests: list[bytes]) -> str:
    """One 64-bit tag of rank 0's window of row digests, in row order: what the rows held, for two runs' lines to be compared (cells/rank_digest_compare.py).
    The only evidence a gathered (sharded) run has of its rows, since its ranks agree by construction."""
    return hashlib.blake2b(b''.join(row_digests), digest_size=8).hexdigest()


def _rows_differ(digests: list[list[bytes]]) -> int:
    """Rows on which any rank differs from rank 0; a row two ranks differ on counts once."""
    return sum(any(peer[row] != digests[0][row] for peer in digests[1:]) for row in range(len(digests[0])))


def _log_rank_check(key: str, layer: str, reports: list[dict], index: int, where: str) -> None:
    """Only rank 0 logs. A report it cannot read must not raise there: its peers would enter the next collective with no exit armed.
    The layer's line is then missing, which the reader (cells/rank_check_read.py) reports."""
    try:
        _write_rank_check_line(key, layer, reports, index, where)
    except Exception as error:  # a diagnostic never takes serving down
        LOG.error('GLM53_RANK_CHECK_LOG_FAILED key=%s layer=%s %s error=%s: %s. No line for this layer; a rank sent a report this one cannot read, '
                  'so rerun the check and send this line with the log', key, layer, where, type(error).__name__, error)


def _write_rank_check_line(key: str, layer: str, reports: list[dict], index: int, where: str) -> None:
    void = _rank_check_void(reports, index)
    if void:
        LOG.warning('GLM53_RANK_CHECK_NOT_RETAINED key=%s layer=%s %s reason=%s. No verdict on these rows; fix the read, or rerun non-serving with %s=eager and %s=0',
                    key, layer, where, void, RANK_CHECK_FLAG, GRAPHS_ONLY_FLAG)
        return
    digests = [report['layers'][index]['digests'] for report in reports]
    differ = _rows_differ(digests)
    LOG.warning('GLM53_RANK_CHECK key=%s layer=%s equal=%d rows=%d rows_differ=%d live_rows=%d ranks=%d %s rows_digest=%s',
                key, layer, int(differ == 0), len(digests[0]), differ, reports[0]['layers'][index]['live'], len(reports), where, _window_tag(digests[0]))


class RankCheck:
    """HPU_GLM53_INDEXER_RANK_CHECK=1: query-shard's kill gate. Is every layer's physical top-k the same on every rank, on real replays?

    One top-k buffer serves all 11 layers, so after a replay it holds the last layer's rows only. With the flag on each indexer writes a buffer of
    its own, allocated before the first capture (a tensor created inside a capture would not be the one a replay writes). The graph runs the same
    ops; only the destination differs, and it is sized to the tallest key's rows and the selected columns, not to the shared buffer. A shorter key
    writes the first rows of it, so only a key's own rows are compared. The rows MLA reads are the returned ones either way. Each row is hashed on
    the host and the hashes meet over gloo, on every rank after every real replay, outside the census's clock.
    """

    def __init__(self) -> None:
        self.layers: list[tuple[str, torch.Tensor]] = []
        self.previous: dict[str, list[bytes]] = {}
        self.baseline_error = ''
        self.replays = 0

    def prepare(self, model: Any, rows: int) -> None:
        for _, module in model.named_modules():
            shared = vars(module).get('topk_indices_buffer') if callable(getattr(module, 'canonical_shared_scan', None)) else None
            if isinstance(shared, torch.Tensor):
                own = torch.full((rows, module.topk_tokens + module.pool_size - 1), -1, dtype=torch.int32, device=shared.device)
                vars(module)[RANK_CHECK_ATTR] = own
                self.layers.append((_layer_label(module), own))
        if not self.layers:
            raise RuntimeError(f'{RANK_CHECK_FLAG}=1 found no sparse indexer in the model, so it has nothing to read. Unset {RANK_CHECK_FLAG} on this model and restart')
        _rank_check_fence()
        LOG.warning('GLM53_RANK_CHECK_READY layers=%d rows=%d bytes_each=%d', len(self.layers), rows, self.layers[0][1].numel() * 4)

    def before_replay(self) -> None:
        """What the buffers hold before the first real replay, so a replay that never writes them shows as stale. A host read on this rank only."""
        if self.previous or self.baseline_error:
            return
        try:
            self.previous = {label: _row_digests(rows.to('cpu')) for label, rows in self.layers}
        except Exception as error:  # after_replay reports it on every rank
            self.baseline_error = f'baseline read: {type(error).__name__}: {error}'

    def forget_baseline(self) -> None:
        """A replay this check did not read has rewritten the buffers, so what the last read left is no baseline: the next before_replay reads a new one."""
        self.previous = {}
        self.baseline_error = ''

    def _read_layers(self, key_rows: int, real_rows: int, prompts: int) -> list[dict]:
        """Each layer's first ``key_rows`` rows, judged against ``real_rows``. The baseline for the next replay changes whole or not at all: a read that
        died halfway leaves none, and before_replay takes one from what the buffers hold then (half new, half old digests would hide a stale layer)."""
        if self.baseline_error:
            raise RuntimeError(self.baseline_error)
        try:
            layers, latest = self._read_all(key_rows, real_rows, prompts)
        except Exception:
            self.previous = {}
            raise
        self.previous = latest
        return layers

    def _read_all(self, key_rows: int, real_rows: int, prompts: int) -> tuple[list[dict], dict[str, list[bytes]]]:
        _rank_check_fence()
        packed_rows = real_rows if prompts == 1 else None
        layers, latest = [], {}
        for label, rows in self.layers:
            host = rows.to('cpu')
            digests = _row_digests(host)
            layers.append({**_layer_entry(host, digests, self.previous.get(label), key_rows, packed_rows), 'expected': real_rows})
            latest[label] = digests
        return layers, latest

    def after_replay(self, key: str, start: int, key_rows: int, real_rows: int, prompts: int = 1) -> None:
        """``key_rows``: the rows the replayed key writes; ``real_rows``: the chunk's own, which the rest of the key's rows pad; ``prompts``: how
        many prompts the key packs (one: the chunk's rows come first and the padding after, so each row's place is checked, not just their count)."""
        self.replays += 1
        reports = _exchange_rows((key, start), lambda: self._read_layers(key_rows, real_rows, prompts))
        if _rank_check_is_logger():
            for index, (label, _) in enumerate(self.layers):
                _log_rank_check(key, label, reports, index, f'start={start} replay={self.replays}')


def rank_check_eager_rows(op: Any, rows: torch.Tensor, key: str) -> None:
    """HPU_GLM53_INDEXER_RANK_CHECK=eager, the plan's fallback when a replay's rows cannot be read: compare the rows a live native forward just selected.

    Non-serving: the host read ends the forward's lazy segment. Native takes the legacy merge, not a key's k512 route, so equal=1 here is weaker than on a replay.
    """
    if rank_check_mode() != 'eager' or not _RANK_CHECK_BOOT_DONE[0] or capture_active():
        return
    reports = _exchange_rows((key,), lambda: [_rows_report(rows)])
    if _rank_check_is_logger():
        _log_rank_check(key, _layer_label(op), reports, 0, 'eager=1')


@contextmanager
def rank_check_target(op: Any, rows: int) -> Iterator[None]:
    """Inside a prompt key's or a decode key's capture with the check on, this indexer's writes go to its own buffer. The same ops; only the destination differs."""
    own = vars(op).get(RANK_CHECK_ATTR)
    if own is None or not any_key_capture_active():
        yield
        return
    if rows > own.shape[0]:
        LOG.warning('GLM53_RANK_CHECK_TARGET_TOO_SHORT layer=%s forward_rows=%d buffer_rows=%d: this capture writes the shared buffer, so the check will read '
                    'nothing for it; size the buffers by the tallest key', _layer_label(op), rows, own.shape[0])
        yield
        return
    shared = vars(op)['topk_indices_buffer']
    vars(op)['topk_indices_buffer'] = own
    try:
        yield
    finally:
        vars(op)['topk_indices_buffer'] = shared


def prepare_rank_check(runner: Any, buckets: tuple) -> None:
    """Before the first capture and the seal's first reading, so the keys are not charged for the diagnostic's buffers.

    A failure on one rank (the allocation, the fence) leaves it out of warmup's next gather while its peers wait there: arm the bounded exit, then raise.
    """
    if rank_check_mode() != '1':
        return
    try:
        check = graph_cache(runner).rank_check
        check.prepare(runner.model, max(_bucket_rows(bucket) for bucket in buckets))
    except Exception:
        graphs_only.arm_exit('GLM53_RANK_CHECK_PREPARE_FAILED')
        raise
    _DECODE_RANK.bind(check)


class DecodeRankRead:
    """HPU_GLM53_INDEXER_RANK_CHECK=1 on decode keys: the runner's three decode hooks hand each served step to the prompt keys' RankCheck.

    decode_table_width notes the step's lanes and its longest position; decode_key_metadata arms the read for a served replay of a captured key
    (a key's capture and a warmup forward are never read); observe_decode_hidden reads the replay's rows. A decode key's rows are the lanes first and
    padding after, so the read is a packed one-prompt read of ``lanes`` rows. Each key is read DECODE_CHECK_REPLAYS_PER_KEY times: a replay the check
    skips has rewritten the buffers, so its baseline is dropped. Every rank decides from host values alone, so every rank enters each gloo gather.
    """

    def __init__(self) -> None:
        self.check: RankCheck | None = None
        self.lanes: tuple[int, int] | None = None  # (lane count, longest lane position) noted for the step being built
        self.pending: tuple[str, int, int, int] | None = None  # (key label, start, key rows, lanes) to read after this forward
        self.reads: Counter = Counter()

    def bind(self, check: RankCheck) -> None:
        self.check = check

    def note_step(self, req_ids: list[str], positions: torch.Tensor) -> None:
        """decode_table_width: the step's lanes are its first len(req_ids) rows; ``positions`` holds each lane's token position in column 0."""
        if self.check is None:
            return
        count = len(req_ids)
        lanes = [int(position) for position in positions[:count, 0]]
        self.lanes = (count, max(lanes)) if count and len(lanes) == count else None

    def before_replay(self, key: tuple[int, int]) -> None:
        """decode_key_metadata, a served replay of a captured key: take the baseline now, or drop it when this replay goes unread."""
        if self.check is None:
            return
        lanes, self.lanes, self.pending = self.lanes, None, None
        if lanes is None or self.reads[key] >= DECODE_CHECK_REPLAYS_PER_KEY:
            self.check.forget_baseline()
            return
        self.reads[key] += 1
        self.check.before_replay()
        self.pending = (f'decode:{key[0]}x{key[1]}', lanes[1], key[0], lanes[0])

    def after_forward(self) -> None:
        """observe_decode_hidden: the replay has been submitted; read every layer's rows of the key, if this forward was armed."""
        pending, self.pending = self.pending, None
        if pending is None or self.check is None:
            return
        label, start, key_rows, lanes = pending
        self.check.after_replay(label, start, key_rows, lanes, 1)


_DECODE_RANK = DecodeRankRead()


class Ballast:
    """HPU_GLM53_K1_BALLAST_MIB=a,b,c (K1 only, unset = nothing): every HPU_GLM53_K1_BALLAST_EVERY key replays the rank holds the next total.

    between_peak fills whatever is free, so it cannot show what a replay needs. The largest total under which replays still run
    bounds the need from above; the first PT_DEVMEM ends the run. Every rank steps at the same replay, so they stay in order.
    """

    def __init__(self) -> None:
        raw = os.getenv(BALLAST_ENV, '').strip()
        self.levels = [int(item) for item in raw.split(',')] if raw else []
        self.every = positive_env(BALLAST_EVERY_ENV, BALLAST_EVERY)
        self.held: list[torch.Tensor] = []
        self.total_mib = self.replays = 0

    def _allocate(self, mib: int) -> None:
        import habana_frameworks.torch as htorch
        for size in [BALLAST_CHUNK_MIB] * (mib // BALLAST_CHUNK_MIB) + [mib % BALLAST_CHUNK_MIB]:
            if size:
                self.held.append(torch.ones(size * MIB, dtype=torch.uint8, device='hpu'))
        htorch.core.mark_step()
        htorch.hpu.synchronize()

    @staticmethod
    def _free_bytes() -> int:
        import habana_frameworks.torch as htorch
        return htorch.hpu.mem_get_info()[0]

    def advance(self) -> None:
        if not self.levels:
            return
        self.replays += 1
        if self.replays % self.every:
            return
        target = self.levels.pop(0)
        self._allocate(target - self.total_mib)
        self.total_mib = target
        LOG.warning('GLM53_K1_BALLAST total_mib=%d replays=%d free_bytes=%d', target, self.replays, self._free_bytes())


class GraphCache:
    def __init__(self, raw_forward: Callable, block_size: int, backend: Any = None) -> None:
        self.raw_forward = raw_forward
        self.block_size = block_size
        self.backend = backend if backend is not None else default_backend()
        self.entries: dict[tuple, tuple] = {}
        self.counts: Counter = Counter(mixed_steps=0, graph_replays=0, captures=0,
                                      lazy_fallbacks=0, native_fallbacks=0, prompt_tokens=0, decode_rows=0)
        self.counts.update(dict.fromkeys(('fallback_' + reason for reason in (
            'decode_rows_over', 'width_over', 'ctx_over', 'no_decode', 'multi_prompt', 'signature',
            'shape_mismatch')), 0))
        self.estimated_bytes = 0
        self.sealed_bytes = 0  # the prompt keys' measured allocator in-use delta, set at the seal (seal_prompt_keys)
        self.merge = KeyMerge()
        self.ballast = Ballast()
        self.rank_check = RankCheck()
        self.key_captures = 0
        self.workspace_released = False

    def record(self, batches: list, event: str, reason: str = '') -> None:
        self.counts['mixed_steps'] += 1
        self.counts['boot_steps' if _BOOT.get() else 'live_steps'] += 1
        self.counts[event] += 1
        self.counts['prompt_tokens'] += sum(b.region.count for b in batches if not b.region.decode)
        self.counts['decode_rows'] += sum(b.region.count for b in batches if b.region.decode)
        if reason:
            self.counts['fallback_' + reason] += 1
        if event in ('graph_replays', 'captures') and native_pin_enabled():
            self.counts[event + '_w' + 'x'.join(map(str, native_width_pair(batches)))] += 1
        every = positive_env('VLLM_GAUDI_MIXED_GRAPH_LOG_EVERY', 100)
        if self.counts['mixed_steps'] == 1 or self.counts['mixed_steps'] % every == 0:
            steps = self.counts['mixed_steps']
            LOG.warning('GLM53_MIXED_GRAPH counters=%s prompt_tokens_per_step=%.2f '
                        'decode_rows_per_step=%.2f entries=%d estimated_MiB=%.1f',
                        dict(self.counts), self.counts['prompt_tokens']/steps,
                        self.counts['decode_rows']/steps, len(self.entries), self.estimated_bytes/MIB)

    def run(self, batches: list) -> torch.Tensor | None:
        if graphs_only_enabled():
            raise RuntimeError(f'{GRAPHS_ONLY_FLAG}=1 has no mixed decode plus prompt graph: prompts replay their {PROMPT_KEYS_FLAG} key and '
                               f'decodes their own graph. Set {PROMPT_KEYS_FLAG} (the keyed step never calls this), or unset {GRAPHS_ONLY_FLAG}')
        bucket, reason = select_bucket(batches, self.block_size)
        if bucket is None:
            self.counts['fallback_outside_lattice'] += 1  # Legacy aggregate, not another fallback.
            self.record(batches, miss_event(), reason)
            return None
        try:
            inputs = padded_inputs(batches, bucket)
        except (ValueError, TypeError, AttributeError):
            self.record(batches, miss_event(), 'signature')
            return None
        key = bucket, signature(inputs)
        if key in self.entries:
            return self._replay(key, inputs, batches)
        # Nativepin: BOOT warms one graph per native width pair of a bucket; a live unwarmed width runs native.
        if any(entry[0] == bucket for entry in self.entries) and not (_BOOT.get() and native_pin_enabled()):
            self.record(batches, miss_event(), 'signature')
            return None
        return self.capture(key, inputs, batches)

    def _replay(self, key: tuple, inputs: tuple, batches: list) -> torch.Tensor:
        static, graph, output = self.entries[key]
        copy_inputs(static, inputs)
        with graphs_only.fail_fast_on_allocation_error('prompt key replay') if graphs_only_enabled() else nullcontext():
            self.backend.replay(graph)
        self.record(batches, 'graph_replays')
        return compact_hidden(output, batches, key[0])

    def _key_miss(self, batches: list, reason: str) -> None:
        """A live chunk with no key runs the runner's own forward; a BOOT dummy with none means the key list is wrong."""
        if _BOOT.get():
            raise RuntimeError(f'prompt key warmup dummy has no graph ({reason}); check {PROMPT_KEYS_FLAG}')
        if graphs_only_enabled():
            raise self._refused(batches, reason)
        self.record(batches, 'native_fallbacks', reason)

    def _refused(self, batches: list, reason: str) -> RuntimeError:
        """The error for a chunk no key can serve. Every rank holds the same host metadata, so every rank refuses the same chunk before any
        collective; the bounded exit covers a peer that did not."""
        self.counts['graphs_only_refused'] += 1
        self.counts['fallback_' + reason] += 1
        tokens = sum(b.region.count for b in batches if not b.region.decode)
        LOG.error('GLM53_GRAPHS_ONLY_REFUSED reason=%s prompt_tokens=%d refused=%d', reason, tokens, self.counts['graphs_only_refused'])
        graphs_only.arm_exit(f'prompt chunk refused ({reason})')
        return RuntimeError(graphs_only_refusal(reason))

    def _natural_key(self, batch: Any) -> tuple[tuple[int, int] | None, str]:
        from glm53_mixed_step import prompt_batch_keys
        region = batch.region
        if region.decode or (not prompt_batch_keys() and (region.batch != 1 or len(region.requests) != 1)):
            return None, 'multi_prompt'
        table = getattr(region.metadata, 'glm53_index_block_table', None)
        if table is None or table.ndim != 2:
            return None, 'signature'
        if len(region.requests) > 1:
            return select_shared_prompt_key(len(region.requests), max(region.lengths), table.shape[1])
        return select_prompt_key(region.lengths[0], table.shape[1])

    def _key_forward(self, key: tuple, inputs: tuple, batches: list) -> torch.Tensor:
        def run() -> torch.Tensor:
            if key in self.entries:
                self.ballast.advance()
                return self._replay(key, inputs, batches)
            return self.capture(key, inputs, batches)

        if not census_enabled():
            return run()
        rows, width = key[0][1:3]
        prompts = key[0][3] if len(key[0]) > 3 else 1
        return census_graph_forward(run, rows, width, lambda: self.key_captures, key_replay=key in self.entries, prompts=prompts,
                                    real_prompts=len(batches[0].region.requests))()

    def release_workspace_once(self, bucket: tuple) -> None:
        """After the first prompt key, flag on. Once only: a second fill has nothing to hand back and aborts the rank (fast4r attempt 2)."""
        if self.workspace_released or bucket[0] != 0 or os.getenv(WORKSPACE_RELEASE_FLAG, '0') != '1':
            return
        self.workspace_released = True
        release_capture_workspace(bucket)

    def run_key(self, batch: Any) -> torch.Tensor | None:
        """Form B: this prompt chunk alone through its key's graph. None = the caller runs the runner's own forward."""
        natural, reason = self._natural_key(batch)
        if natural is None:
            return self._key_miss([batch], reason)
        if rank_check_mode() == 'eager' and not _BOOT.get():
            return self._key_miss([batch], 'rank_check_eager')
        chosen = self.merge.key_for(batch, natural)
        bucket = (0, *chosen)
        try:
            inputs = padded_inputs([batch], bucket)
        except (ValueError, TypeError, AttributeError):
            return self._key_miss([batch], 'signature')
        key = bucket, signature(inputs)
        if key not in self.entries and not _BOOT.get():  # the runner's own warmups come before the keys', and run native
            return self._key_miss([batch], 'not_warmed')
        read_rows = rank_check_mode() == '1' and not _BOOT.get()
        if read_rows:
            self.rank_check.before_replay()
        hidden = self._key_forward(key, inputs, [batch])
        self.merge.observe(batch, natural, chosen, hidden)
        if read_rows:
            self.rank_check.after_replay(_key_setting(bucket)[0], batch.region.starts[0], _bucket_rows(bucket), sum(batch.region.lengths),
                                         _bucket_prompts(bucket))
        return hidden

    def capture(self, key: tuple, inputs: tuple, batches: list) -> torch.Tensor | None:
        estimate = estimate_bytes(key[0], self.block_size)
        limit = positive_env('VLLM_GAUDI_MIXED_GRAPH_MAX_GRAPHS', 4)
        budget = positive_env('VLLM_GAUDI_MIXED_GRAPH_BUDGET_MIB', 6144) * MIB
        judged = not prompt_keys()  # keys are judged by the seal's measured in-use delta, not by summed estimates
        error = ''
        if len(self.entries) >= limit or (judged and self.estimated_bytes + estimate > budget):
            error = 'mixed graph budget exhausted before capture; shrink warmup lattice '
            error += 'or raise MAX_GRAPHS/BUDGET_MIB after per-rank HBM measurement'
        prepare = getattr(self.backend, 'prepare_capture', None)
        if prepare is not None:
            prepare(key[0], estimate, error)
        elif error:
            raise ValueError(error)
        static = clone_inputs(inputs)
        call = StaticCall(self.raw_forward, static)
        # No catch-and-lazy retry: a failed capture may have mutated KDA/KV.
        with capture_context(), capture_scope(key[0]):
            graph, output = self.backend.capture(call)
        if key[0][0] == 0:
            check_capture_scalars(f'{key[0][1]}:{key[0][2]}', key[0][2] * self.block_size)
        self.key_captures += 1
        measured = getattr(self.backend, 'last_capture_bytes', None)
        device_delta = getattr(self.backend, 'last_capture_hbm_bytes', 0)
        inputs_bytes = sum(t.numel() * t.element_size() for t in call.tensor_inputs)
        charge = max(estimate, max(measured or 0, device_delta) + inputs_bytes)
        error = 'measured graph residency exceeded budget' if judged and self.estimated_bytes + charge > budget else ''
        if prepare is not None:
            admit_capture(key[0], 0, 0, error)
        self.entries[key] = static, graph, output
        self.release_workspace_once(key[0])
        self.estimated_bytes += charge
        self.record(batches, 'captures')
        LOG.warning('GLM53_MIXED_GRAPH captured bucket=%s estimate_MiB=%.1f allocated_delta_bytes=%s '
                    'hbm_delta_bytes=%s static_input_bytes=%s peak_delta_bytes=%s '
                    'free_before_bytes=%s free_after_bytes=%s tensor_cache=off split_forward=%d%s',
                    key[0], estimate/MIB, measured, device_delta, inputs_bytes,
                    getattr(self.backend, 'last_capture_peak_bytes', None),
                    getattr(self.backend, 'free_before', None), getattr(self.backend, 'free_after', None),
                    call.split, f' native_widths={native_width_pair(batches)}' if native_pin_enabled() else '')
        return compact_hidden(output, batches, key[0])


def forward(runner: Any, batches: list) -> torch.Tensor | None:
    if not enabled():
        return None
    return graph_cache(runner).run(batches)


RESTORE_LEDGER = '_glm53_restore_copies_unfenced'  # runner attribute: requests whose CPU-restore copy no fence has landed yet


def note_restore_copies(runner: Any, restoring: Iterable[str], finished: Iterable[str]) -> None:
    """Arm one fence per restore job the scheduler output carries; its copy is submitted before this step's first forward."""
    if not graphs_only_enabled():
        return
    unfenced = getattr(runner, RESTORE_LEDGER, None)
    if unfenced is None:
        unfenced = set()
        setattr(runner, RESTORE_LEDGER, unfenced)
    unfenced.difference_update(finished)
    unfenced.update(restoring)


def _land_restore_copies() -> float:
    """mark_step submits the lazy copy, synchronize waits for it: the milliseconds that took. A failure here stops every rank."""
    import habana_frameworks.torch as htorch
    started = time.perf_counter()
    try:
        htorch.core.mark_step()
        htorch.hpu.synchronize()
    except Exception as error:
        graphs_only.arm_exit('restore fence')
        raise RuntimeError(f'restore fence failed ({type(error).__name__}: {error}), so a replay could read a slot before its restore copy '
                           'lands; no replay ran. Check the device, then restart all workers') from error
    return (time.perf_counter() - started) * 1000


def fence_restored_rows(runner: Any, request_ids: Iterable[str], *, native: bool = False) -> None:
    """KB 1984: a freed recurrent slot is the next one handed out (LIFO), so before its restore copy lands a replay reads the previous owner's
    state. Land the copy before the first forward of a restored request; later forwards of its latch window replay with no fence.
    `native`: this forward would run off-graph for a latched request, which no serving forward may do. Host state only, so every rank refuses alike."""
    if not graphs_only_enabled():
        return
    if native:
        graphs_only.arm_exit('latched forward refused')
        raise RuntimeError(f'a CPU-restored request would run native, and {GRAPHS_ONLY_FLAG}=1 replays every forward. A decode needs '
                           f'{RESTORED_DECODE_FLAG}=1 and HPU_GRAPH_DRY_RUN=0; a prompt chunk needs its {PROMPT_KEYS_FLAG} key. Set them and restart')
    unfenced = getattr(runner, RESTORE_LEDGER, None)
    landing = unfenced.intersection(request_ids) if unfenced else None
    if not landing:
        return
    elapsed_ms = _land_restore_copies()
    unfenced.difference_update(landing)
    LOG.warning('GLM53_RESTORE_FENCE ms=%.2f requests=%d', elapsed_ms, len(landing))


def _fence_restored_batch(runner: Any, batch: Any) -> None:
    """A prompt region names its rows by batch index; a runner with nothing armed never reads them."""
    if getattr(runner, RESTORE_LEDGER, None):
        fence_restored_rows(runner, [runner.input_batch.req_ids[index] for index in batch.region.requests])


def forward_key(runner: Any, batch: Any) -> torch.Tensor | None:
    _fence_restored_batch(runner, batch)
    return graph_cache(runner).run_key(batch)


def graph_cache(runner: Any) -> GraphCache:
    raw = getattr(runner.model, '_glm53_mixed_graph_forward', None)
    if raw is None:
        raise RuntimeError('mixed graphs require the paired lazy HpuModelAdapter patch; restart with flag ON')
    if runner.model_config.enforce_eager:
        raise ValueError('mixed graphs require enforce_eager=False and PT_HPU_LAZY_MODE=1')
    cache = getattr(runner.model, '_glm53_mixed_graph_cache', None)
    if cache is None:
        cache = GraphCache(raw, runner.attn_block_size)
        runner.model._glm53_mixed_graph_cache = cache
    return cache


def warmup(runner: Any) -> None:
    """Real execute_model/sample_tokens path, after compile-only contexts exit."""
    if not enabled():
        return
    buckets, error = (), ''
    try:
        buckets = validate_warmup(runner)
    except Exception as exc:
        error = str(exc)
    admit_capture(buckets, 0, 0, error)
    prepare_rank_check(runner, buckets)
    mark_boot_window()
    in_use_before = _in_use_bytes() if prompt_keys() else 0
    with boot_capture_context():
        for prompt_cfg, decode_cfg in warmup_scenarios(runner, buckets):
            runner._prepare_dummy_scenario(prompt_cfg, decode_cfg)
    check_keys_captured(runner, buckets)
    if prompt_keys():
        seal_prompt_keys(runner, in_use_before)
    # Outside capture and BOOT: lazy prefills outside the lattice reuse these
    # widened indexer recipes. No-op unless HPU_GLM53_WARM_ADAPTIVE_TILES=1.
    from glm53_r590_long_history import warm_adaptive_tiles_at_boot
    count_off_graph('adaptive_tile_warm', warm_adaptive_tiles_at_boot(runner.model))
    _RANK_CHECK_BOOT_DONE[0] = True


def check_keys_captured(runner: Any, buckets: tuple) -> None:
    """A key whose dummy took another key's graph is a boot that would serve every chunk of it native: refuse it."""
    if not prompt_keys():
        return
    captured = {key[0] for key in runner.model._glm53_mixed_graph_cache.entries}
    missing = [':'.join(map(str, bucket[1:])) for bucket in buckets if bucket not in captured]
    if missing:
        raise RuntimeError(f'prompt keys {missing} captured no graph at BOOT; check the dummy table width against {PROMPT_KEYS_FLAG}')


def _in_use_bytes() -> int:
    """The allocator's in-use bytes after a fence: what the graphs and tensors hold, whatever the pool grew by."""
    import habana_frameworks.torch as htorch
    htorch.core.mark_step()
    htorch.hpu.synchronize()
    return int(htorch.hpu.memory_stats()['InUse'])


def seal_prompt_keys(runner: Any, in_use_before: int) -> int:
    """Charge the prompt keys what the allocator measured: in-use after the last capture minus in-use before the first, on this rank.

    Summed per-key estimates (about 3,500 MiB each) overcharge: k1-out/final-armB-20261107T011429Z left 4,301 of 8,443 MiB free
    with both keys and a 1,538 MiB ballast. The check is against the budget, on every rank (one rank over refuses them all).
    """
    after = _in_use_bytes()
    delta = max(0, after - in_use_before)
    budget = positive_env('VLLM_GAUDI_MIXED_GRAPH_BUDGET_MIB', 6144) * MIB
    keys = ','.join(f'{rows}:{width}' for rows, width in sorted(prompt_keys(), key=lambda key: (-key[1], -key[0])))
    error = '' if delta <= budget else (f'prompt keys {keys} hold {delta/MIB:.1f} MiB of allocator in-use bytes after capture '
                                        f'(in-use {in_use_before/MIB:.1f} -> {after/MIB:.1f} MiB), past VLLM_GAUDI_MIXED_GRAPH_BUDGET_MIB={budget//MIB}')
    admit_capture(('prompt_seal',), 0, 0, error, SEAL_ADVICE)
    runner.model._glm53_mixed_graph_cache.sealed_bytes = delta
    LOG.warning('GLM53_KEY_SEAL keys=%s in_use_before_MiB=%.1f in_use_after_MiB=%.1f in_use_delta_MiB=%.1f budget_MiB=%d',
                keys, in_use_before / MIB, after / MIB, delta / MIB, budget // MIB)
    return delta


def warmup_decode_rows(runner: Any, padded_rows: int) -> int:
    rows = min(padded_rows, runner.input_batch.max_num_reqs - 1)
    if rows < 1 or 1 << (rows - 1).bit_length() != padded_rows:
        raise ValueError(f'mixed warmup request capacity cannot represent {padded_rows} '
                         'padded decode rows plus one prompt; shrink WARMUP decode tier '
                         'or raise max-num-seqs')
    return rows


def validate_prompt_keys(runner: Any) -> tuple:
    """The (0, rows, width) buckets BOOT captures, widest first so the 8192 capture gets the free HBM: decode 0 marks a prompt-only graph.

    The keys' HBM is judged at the seal (seal_prompt_keys) by the allocator's measured in-use delta, not by summed per-key estimates.
    """
    block = runner.attn_block_size
    from glm53_mixed_step import prompt_batch_keys
    singles = tuple((0, rows, width) for rows, width in sorted(prompt_keys(), key=lambda key: (-key[1], -key[0])))
    shared = tuple((0, rows, width, prompts) for prompts, rows, width in sorted(prompt_batch_keys(), key=lambda key: (-key[2], -key[1])))
    buckets = singles + shared
    if measured_estimate_bytes() is not None and len(buckets) != 1:
        raise ValueError(f'{ESTIMATE_ENV} is one measured graph; list one key (got {len(buckets)}) or unset it')
    limit = positive_env('VLLM_GAUDI_MIXED_GRAPH_MAX_GRAPHS', 4)
    if len(buckets) > limit:
        raise ValueError(f'{len(buckets)} prompt keys exceed VLLM_GAUDI_MIXED_GRAPH_MAX_GRAPHS={limit}; raise it or drop a key')
    for rows, width in (bucket[1:3] for bucket in buckets):
        tokens = width * block
        if tokens > runner.max_model_len or rows > tokens or (tokens - rows) % runner.block_size:
            raise ValueError(f'prompt key {rows}:{width} needs a {tokens}-token dummy aligned to {runner.block_size} tokens '
                             f'with its rows inside; max_model_len is {runner.max_model_len}')
    check_graphs_only_keys(runner, singles)
    return buckets


@lru_cache(maxsize=8)
def _parsed_widths(flag: str, spec: str) -> tuple[int, ...]:
    widths: list[int] = []
    for item in filter(None, spec.split(',')):
        if not item.isdigit() or int(item) < 1 or int(item) & (int(item) - 1):
            raise ValueError(f'{flag} must list power-of-two table widths in blocks (e.g. 1024,4096,8192); got {item!r}')
        if int(item) in widths:
            raise ValueError(f'duplicate {flag} width {item!r}; list each width once')
        widths.append(int(item))
    return tuple(sorted(widths, reverse=True))


def decode_widths() -> tuple[int, ...]:
    """The decode lattice's table widths, widest first; empty = no lattice, and every decode hook below is the identity."""
    return _parsed_widths(DECODE_WIDTHS_FLAG, os.getenv(DECODE_WIDTHS_FLAG, ''))


def decode_check_widths() -> tuple[int, ...]:
    return _parsed_widths(DECODE_CHECK_WIDTHS_FLAG, os.getenv(DECODE_CHECK_WIDTHS_FLAG, ''))


def decode_pad_mode() -> str:
    mode = os.getenv(DECODE_PAD_FLAG, '0').strip()
    if mode not in ('0', 'check', '1'):
        raise ValueError(f'{DECODE_PAD_FLAG}={mode!r}: set 0 for native widths, check to pad and compare, 1 to pad with {DECODE_PAD_PROOF_FLAG}')
    return mode


def decode_keys() -> tuple[tuple[int, int], ...]:
    """(decode rows, table width) of every decode graph BOOT captures, widest first, then the check widths' native keys at every row count."""
    lattice_keys = tuple((rows, width) for width in decode_widths() for rows in reversed(DECODE_ROWS))
    return lattice_keys + tuple((rows, width) for width in decode_check_widths() for rows in reversed(DECODE_ROWS))


def _check_decode_pad_flags(widths: tuple[int, ...], mode: str, check: tuple[int, ...]) -> None:
    top = widths[0]
    if mode == '0' and any(1 << power not in widths for power in range(top.bit_length())):
        raise ValueError(f'{DECODE_PAD_FLAG}=0 serves native widths only, so {DECODE_WIDTHS_FLAG} must list every power of two up to {top}; '
                         f'or set {DECODE_PAD_FLAG}=check to pad and compare')
    if mode == '1':
        _check_decode_pad_proof(os.getenv(DECODE_PAD_PROOF_FLAG, '').strip())
    if (mode == 'check') != bool(check) or any(width in widths or width > top for width in check):
        raise ValueError(f'{DECODE_CHECK_WIDTHS_FLAG} goes with {DECODE_PAD_FLAG}=check and lists native widths that pad up to a width of '
                         f'{DECODE_WIDTHS_FLAG}; got {list(check)} with mode {mode}')


def decode_pad_pairs() -> list[tuple[int, int]]:
    """Every (native, padded) width pair mode 1 would serve: each power of two up to the top width that is not a key itself."""
    return [(native, _padded_decode_width(native)) for native in (1 << power for power in range(decode_widths()[0].bit_length()))
            if native not in decode_widths()]


def _proof_merge_lines(spec: str) -> list[tuple[int, ...]] | None:
    """(narrow, wide, rows, torch_equal, short) of every GLM53_DECODE_MERGE line in the file, or in every file under the directory; None = no such path."""
    path = Path(spec)
    files = sorted(file for file in path.rglob('*') if file.is_file()) if path.is_dir() else [path] if path.is_file() else []
    if not files:
        return None
    return [tuple(int(group) for group in match.groups()) for file in files
            for match in map(MERGE_LINE.search, file.read_text(encoding='utf-8', errors='replace').splitlines()) if match]


def pad_proof_error(spec: str, pairs: list[tuple[int, int]]) -> str:
    """Why this run proves nothing about the pairs; '' when its merge lines cover them all, equal, with a 16-row and an all-short batch."""
    lines = _proof_merge_lines(spec) if spec else None
    if lines is None:
        return (f'{DECODE_PAD_PROOF_FLAG}={spec!r} is not a file or directory on this host; name the saved log of the {DECODE_PAD_FLAG}=check '
                'run (a file, or a directory of them), whose GLM53_DECODE_MERGE lines the boot reads')
    if not lines:
        return f'{DECODE_PAD_PROOF_FLAG}={spec!r} holds no GLM53_DECODE_MERGE line with a short= field; keep the log of a {DECODE_PAD_FLAG}=check run of this build'
    unequal = [line for line in lines if not line[3]]
    if unequal:
        return (f'{DECODE_PAD_PROOF_FLAG}={spec!r}: a GLM53_DECODE_MERGE line reads torch_equal=0 (narrow={unequal[0][0]} wide={unequal[0][1]} '
                f'rows={unequal[0][2]}), so padding changed a result; keep {DECODE_PAD_FLAG}=0')
    missing = [f'{narrow} -> {wide}' for narrow, wide in pairs if not any(line[:2] == (narrow, wide) for line in lines)]
    if missing:
        return (f'{DECODE_PAD_PROOF_FLAG}={spec!r} never compared the widths {missing} that {DECODE_PAD_FLAG}=1 would pad; '
                f'list them in {DECODE_CHECK_WIDTHS_FLAG}, run check, and name that run')
    for found, what in ((any(line[2] == DECODE_ROWS[-1] for line in lines), f'a {DECODE_ROWS[-1]}-row batch'),
                        (any(line[4] for line in lines), 'an all-short batch (every row inside the top-k window)')):
        if not found:
            return f'{DECODE_PAD_PROOF_FLAG}={spec!r} holds no comparison of {what}; serve one under {DECODE_PAD_FLAG}=check and name that run'
    return ''


def _check_decode_pad_proof(spec: str) -> None:
    error = pad_proof_error(spec, decode_pad_pairs())
    if error:
        raise ValueError(f'{DECODE_PAD_FLAG}=1 refused: {error}')


def check_decode_lattice_flags() -> None:
    widths, mode, check = decode_widths(), decode_pad_mode(), decode_check_widths()
    if not widths:
        if mode != '0' or check:
            raise ValueError(f'{DECODE_PAD_FLAG} and {DECODE_CHECK_WIDTHS_FLAG} only apply to a lattice; set {DECODE_WIDTHS_FLAG} or unset them')
        return
    if not prompt_keys():
        raise ValueError(f'{DECODE_WIDTHS_FLAG} captures after the prompt keys; set {PROMPT_KEYS_FLAG}')
    if os.getenv(DECODE_REPLAY_FLAG, '0') != '1':
        raise ValueError(f'{DECODE_WIDTHS_FLAG} needs {DECODE_REPLAY_FLAG}=1: a latched route otherwise runs its decode eager')
    if os.getenv(CELL_BUDGET_FLAG, '') != '0':
        raise ValueError(f'{DECODE_WIDTHS_FLAG} needs {CELL_BUDGET_FLAG}=0: a key always replays its full rows, the cap only wastes them')
    _check_decode_pad_flags(widths, mode, check)


def decode_key_capture_active() -> bool:
    return _DECODE_KEY.get() is not None


@contextmanager
def decode_key_context(key: tuple[int, int]) -> Iterator[None]:
    token = _DECODE_KEY.set(key)
    try:
        yield
    finally:
        _DECODE_KEY.reset(token)


def decode_rows_for(num_decodes: int, bucket_rows: int) -> int:
    """Padded rows of a decode step: the runner's bucket without a lattice, else the smallest key that holds it."""
    if not decode_widths():
        return bucket_rows
    for rows in DECODE_ROWS:
        if rows >= num_decodes:
            return rows
    raise RuntimeError(f'{num_decodes} decodes exceed the top decode key of {DECODE_ROWS[-1]} rows; set --max-num-seqs to {DECODE_ROWS[-1]} or less')


class DecodeStep(NamedTuple):
    digest: bytes
    native: int
    padded: int
    use_padded: bool
    rows: int
    position: int
    short: bool


class DecodeWidthCheck:
    """HPU_GLM53_DECODE_WIDTH_PAD=check: a repeated greedy step runs native, then padded; every real row of its hidden is compared.

    Any lane mix at a check width, since the native key exists at every row count. A decode is stateful, so the two runs are two repeats of
    one step (the same lanes at the same positions), the way KeyMerge compares two widths of a prompt. A step whose longest row fits the
    top-k window is marked short: at its native width it takes the exact-causal branch, and padding may move it to the scored one. Every
    rank must pick one width, and a mismatch on any rank stops them all.
    """

    def __init__(self) -> None:
        self.pending: dict[bytes, torch.Tensor] = {}
        self.prompts: dict[tuple[str, int], bytes] = {}
        self.current: DecodeStep | None = None

    def _prompt_digest(self, runner: Any, req_id: str) -> bytes:
        prompt = runner.requests[req_id].prompt_token_ids
        key = req_id, len(prompt)
        if key not in self.prompts:
            if len(self.prompts) >= DECODE_CHECK_MAX_PENDING:
                self.prompts.clear()
            self.prompts[key] = hashlib.sha256(str(list(prompt)).encode()).digest()
        return self.prompts[key]

    def _digest(self, runner: Any, req_ids: list[str], positions: list[int]) -> bytes:
        lanes = b''.join(self._prompt_digest(runner, req_id) + str(position).encode() for req_id, position in zip(req_ids, positions))
        return hashlib.sha256(lanes).digest()

    def width_for(self, runner: Any, req_ids: list[str], native: int, padded: int, positions: torch.Tensor) -> int:
        rows = len(req_ids)
        lanes = [int(position) for position in positions[:rows, 0]]
        digest = self._digest(runner, req_ids, lanes)
        use_padded = digest in self.pending
        reports = rank_reports({'digest': digest.hex(), 'pad': use_padded})
        for rank, report in enumerate(reports):
            if report != reports[0]:
                raise RuntimeError(f'decode width check: rank {rank} chose {report}, rank 0 {reports[0]}; every rank must run one key. '
                                   f'Set {DECODE_PAD_FLAG}=0 and report the step at position {max(lanes)}')
        self.current = DecodeStep(digest, native, padded, use_padded, rows, max(lanes), max(lanes) + 1 <= SPARSE_TOPK_TOKENS)
        return padded if use_padded else native

    def observe(self, hidden: torch.Tensor) -> None:
        step, self.current = self.current, None
        if step is None:
            return
        snapshot = hidden[:step.rows].detach().float().cpu()
        if step.use_padded:
            self._compare(step, self.pending.pop(step.digest), snapshot)
            return
        while len(self.pending) >= DECODE_CHECK_MAX_PENDING:
            del self.pending[next(iter(self.pending))]
        self.pending[step.digest] = snapshot

    @staticmethod
    def _compare(step: DecodeStep, native: torch.Tensor, padded: torch.Tensor) -> None:
        equal = torch.equal(native, padded)
        LOG.warning('GLM53_DECODE_MERGE narrow=%d wide=%d rows=%d position=%d torch_equal=%d short=%d', step.native, step.padded, step.rows,
                    step.position, int(equal), int(step.short))
        different = [f'rank {rank}' for rank, report in enumerate(rank_reports({'equal': equal})) if not report['equal']]
        if different:
            raise RuntimeError(f'decode width check: padding table width {step.native} to {step.padded} changed the hidden of {step.rows} rows at '
                               f"position {step.position} on {', '.join(different)}; padding is not exact for this lattice. Set {DECODE_PAD_FLAG}=0 "
                               f'and keep every power-of-two width as a key, and report the GLM53_DECODE_MERGE line')


_DECODE_CHECK = DecodeWidthCheck()
_DECODE_CAPTURED: set[tuple[int, int]] = set()
_DECODE_BUILT: list[tuple[int, int]] = []
_DECODE_BELOW_KEY: set[int] = set()


def reset_decode_lattice() -> None:
    """Forget this process's captured keys and pending comparisons."""
    _DECODE_CHECK.__init__()
    _DECODE_CAPTURED.clear()
    _DECODE_BUILT.clear()
    _DECODE_BELOW_KEY.clear()
    _DECODE_RANK.__init__()
    OFF_GRAPH.clear()
    _EAGER_AT_WINDOW_START[0] = 0
    _CAPTURE_SCALARS.clear()


def mark_decode_key_captured(key: tuple[int, int]) -> None:
    _DECODE_CAPTURED.add(key)


def _padded_decode_width(native: int) -> int:
    fitting = [width for width in decode_widths() if width >= native]
    if not fitting:
        raise RuntimeError(f'a decode needs table width {native}, past the top of {DECODE_WIDTHS_FLAG}={decode_widths()}; '
                           'add the width and restart: context stays as it is')
    return min(fitting)


def _padded_below_lowest_key(native: int, padded: int) -> int:
    """R11: a short context has no graph of its own width, and native keys for all ten widths under 1024 would hold about 4,760 MiB (3,430 with
    the shared owner). So it serves on the lowest key, uncompared unless a run lists its width in the check widths; the line says which."""
    if native not in _DECODE_BELOW_KEY:
        _DECODE_BELOW_KEY.add(native)
        LOG.warning('GLM53_DECODE_BELOW_KEY native=%d key=%d: a short context pads up to the lowest key, with no native key to compare against',
                    native, padded)
    return padded


def decode_table_width(runner: Any, req_ids: list[str], native: int, positions: torch.Tensor) -> int:
    """The decode table width: native, or padded with the table's -1 fill up to the next key. A key's own capture keeps its width."""
    _DECODE_RANK.note_step(req_ids, positions)
    if not decode_widths() or decode_key_capture_active():
        return native
    mode = decode_pad_mode()
    if mode == '0':
        if native not in decode_widths():
            raise RuntimeError(f'native table width {native} has no key in {DECODE_WIDTHS_FLAG}={decode_widths()}; list it and restart')
        return native
    padded = _padded_decode_width(native)
    if mode != 'check':
        return padded
    if native in decode_check_widths():
        return _DECODE_CHECK.width_for(runner, req_ids, native, padded, positions)
    # Padding is unproven until compared: below the lowest key there is no other graph; anywhere else native, whose lack of a key refuses at the metadata.
    return _padded_below_lowest_key(native, padded) if native < decode_widths()[-1] else native


def observe_decode_hidden(hidden: torch.Tensor) -> None:
    _DECODE_CHECK.observe(hidden)
    _DECODE_RANK.after_forward()


def decode_key_metadata(trimmed: Any, rows: int, use_graphs: bool, warmup_mode: bool) -> Any:
    """The metadata a decode key's graph hashes, or a refusal when this forward would not replay one.

    block_list, block_usage and block_groups are sized by the step's total blocks, so they would split one (rows, width) into a graph
    per lane mix; the sparse path never reads them, and without block_usage the graph holds no one-hot or bias of that size.
    """
    if not decode_widths() or trimmed.is_prompt:
        return trimmed
    table = trimmed.glm53_index_block_table
    key = rows, table.shape[1]
    if decode_key_capture_active():
        _DECODE_BUILT.append(key)
    elif warmup_mode:
        return trimmed
    elif key not in _DECODE_CAPTURED:
        raise RuntimeError(f'no captured decode key {rows}x{key[1]}; captured {sorted(_DECODE_CAPTURED)}. Add the width to {DECODE_WIDTHS_FLAG} and restart')
    elif not use_graphs:
        raise RuntimeError(f'decode key {rows}x{key[1]} would run eager; set {RESTORED_DECODE_FLAG}=1 and HPU_GRAPH_DRY_RUN=0 and restart')
    else:
        _DECODE_RANK.before_replay(key)
    return trimmed._replace(block_list=None, block_usage=None, block_groups=None)


def decode_config_blocks(metadata: Any, num_blocks: int) -> int:
    """The runner's warmed-config count: a decode key is its table width, not the step's total blocks, which vary per lane mix."""
    if not decode_widths() or metadata.is_prompt:
        return num_blocks
    return metadata.glm53_index_block_table.shape[1]


def decode_dummy_lengths(rows: int, width: int, block: int, max_model_len: int) -> list[int]:
    """One long row sets the table width; short rows fill the padded batch at one block each."""
    return [min(width * block, max_model_len) - 1] + [block - 1] * (rows - 1)


def decode_key_estimate_bytes(rows: int, width: int) -> int:
    """A decode key's capture peak over its pre-capture in-use: the narrowest measured width that holds it; past the widest, that rate scaled
    by width (r1 cost grew less than width from 4096 to 8192, so the scaling is a ceiling, not a reading). The readings are owner-on;
    owner-off is charged them over DECODE_OWNER_PERCENT, since the owner is what shrinks the gather (charge high, never low, until measured)."""
    measured = sorted(DECODE_KEY_MIB_PER_ROW)
    wider = [known for known in measured if known >= width]
    per_row = DECODE_KEY_MIB_PER_ROW[wider[0]] if wider else -(-DECODE_KEY_MIB_PER_ROW[measured[-1]] * width // measured[-1])
    estimate = rows * per_row * MIB
    return estimate if os.getenv(SHARED_OWNER_FLAG, '0') == '1' else -(-estimate * 100 // DECODE_OWNER_PERCENT)


def listed_scan_workspace_bytes(key: tuple[int, int]) -> int:
    """L2a: a key listed in the decode lane's tile ceilings scans chunks wider than the tile r1 measured, under max(context budget, listed
    budget); the indexer refuses a chunk past that budget (budget_shortfall), so the budget bounds the workspace it adds. 0 when unlisted."""
    scan_v2 = sys.modules.get(DECODE_SCAN_V2_MODULE)
    if scan_v2 is None or scan_v2.tile_ceiling_for(key) is None:
        return 0
    context_mib = int(os.getenv(CONTEXT_CHUNK_BUDGET_FLAG, str(CONTEXT_CHUNK_BUDGET_DEFAULT_MIB)))
    return max(context_mib, scan_v2.chunk_budget_mib()) * MIB


def decode_capture_requirement(key: tuple[int, int]) -> int:
    return decode_key_estimate_bytes(*key) + listed_scan_workspace_bytes(key) + DECODE_CAPTURE_MARGIN_MIB * MIB


def _allocator_headroom() -> int:
    """Bytes the allocator can still hand out after a fence: its Limit minus InUse. PT_DEVMEM aborts past it whatever the budget says."""
    import habana_frameworks.torch as htorch
    htorch.core.mark_step()
    htorch.hpu.synchronize()
    stats = htorch.hpu.memory_stats()
    return int(stats['Limit']) - int(stats['InUse'])


def _decode_headroom_error(key: tuple[int, int]) -> tuple[str, int]:
    """(error, headroom) before a decode capture: a capture past the limit kills the worker on PT_DEVMEM, so it never starts."""
    label = f'decode key {key[0]}x{key[1]}'
    try:
        headroom = _allocator_headroom()
    except Exception as exc:
        return f'{label}: allocator headroom could not be read before its capture ({type(exc).__name__}: {exc})', 0
    required = decode_capture_requirement(key)
    if headroom >= required:
        return '', headroom
    workspace = listed_scan_workspace_bytes(key)
    listed = f' + {workspace / MIB:.0f} listed scan workspace' if workspace else ''
    return (f'{label} needs {required / MIB:.1f} MiB under the allocator limit to capture ({decode_key_estimate_bytes(*key) / MIB:.1f} '
            f'measured peak{listed} + {DECODE_CAPTURE_MARGIN_MIB} margin) and {headroom / MIB:.1f} MiB is left'), headroom


def _reset_peak_in_use() -> None:
    import habana_frameworks.torch as htorch
    htorch.hpu.reset_peak_memory_stats()


def _peak_in_use_bytes() -> int:
    """The allocator's MaxInUse after a fence: the highest in-use since the last reset."""
    import habana_frameworks.torch as htorch
    htorch.core.mark_step()
    htorch.hpu.synchronize()
    return int(htorch.hpu.memory_stats()['MaxInUse'])


def _free_hbm() -> int:
    import habana_frameworks.torch as htorch
    htorch.core.mark_step()
    htorch.hpu.synchronize()
    return htorch.hpu.mem_get_info()[0]


def _decode_lattice_error(runner: Any, keys: tuple, charged: int, budget: int) -> str:
    """Why this lattice cannot boot, before its first capture; '' when it can."""
    rows = max(key[0] for key in keys)
    if runner.input_batch.max_num_reqs != rows:
        return (f'the top decode key holds {rows} rows, so --max-num-seqs {rows} is the batch that fits it; the batch holds '
                f'{runner.input_batch.max_num_reqs}, and a step of more decodes than {rows} would raise at serving')
    if decode_widths()[0] * runner.attn_block_size < runner.max_model_len:
        return f'top decode width {decode_widths()[0]} cannot hold max_model_len {runner.max_model_len}; list a wider one'
    if runner._external_latch_applies(SimpleNamespace(is_prompt=False)):
        return f'a restored decode would run eager; set {RESTORED_DECODE_FLAG}=1 and HPU_GRAPH_DRY_RUN=0'
    shard_error = decode_shard_tp_error(tensor_parallel_size()) if decode_query_shard_keys() else ''
    if shard_error:
        return shard_error
    estimate = sum(decode_key_estimate_bytes(*key) for key in keys)
    if charged + estimate > budget:
        return (f'{len(keys)} decode keys need {estimate/MIB:.1f} MiB on top of {charged/MIB:.1f} MiB charged, past '
                f'VLLM_GAUDI_MIXED_GRAPH_BUDGET_MIB={budget//MIB}')
    return ''


def _graph_count() -> int | None:
    """Graphs in the HPU graph cache the decode keys land in, or None when no graph-state guard is loaded to count them."""
    state = sys.modules.get('glm53_graph_state')
    guards = list(state.LIFECYCLE.live_guards()) if state is not None else []
    return sum(guard.graphs() for guard in guards) if guards else None


def _capture_growth_error(key: tuple[int, int], before: int | None, after: int | None) -> str:
    """The dummy built the key's metadata; this says whether it also captured exactly one graph (R3)."""
    label = f'decode key {key[0]}x{key[1]}'
    if before is None or after is None:
        return (f'{label}: no graph count to prove the capture; {PRIVATE_TOPK_FLAG}=1 loads the graph guard that counts them'
                if graphs_only_enabled() else '')
    if after - before == 1:
        return ''
    outcome = f'captured {after - before} graphs' if after > before else 'captured no graph'
    return f'{label} {outcome} (graph count {before} -> {after}); its dummy built the metadata without one clean capture, so serving would capture silently'


def _run_decode_capture(runner: Any, key: tuple[int, int]) -> tuple[str, int, int]:
    """(error, allocator bytes the capture kept, its peak over the in-use before it) of the one dummy forward that captures this key.
    No retry: a failed capture may have left state."""
    rows, width = key
    lengths = decode_dummy_lengths(rows, width, runner.attn_block_size, runner.max_model_len)
    _DECODE_BUILT.clear()
    take_capture_scalars()
    free_before = 0
    try:
        free_before, in_use_before, graphs_before = _free_hbm(), _in_use_bytes(), _graph_count()
        _reset_peak_in_use()
        with decode_key_context(key):
            try:
                runner._prepare_dummy_scenario(None, (rows, 1, width), decode_seq_lengths=lengths)
                taken, graphs_after = max(0, _in_use_bytes() - in_use_before), _graph_count()
                peak = max(0, _peak_in_use_bytes() - in_use_before)
            except Exception:
                arm_failed_capture_exit()
                raise
    except Exception as exc:
        return f'decode key {rows}x{width} capture failed without retry, with {free_before / MIB:.1f} MiB free HBM before it: {type(exc).__name__}: {exc}', 0, 0
    if _DECODE_BUILT != [key]:
        return f'the dummy of decode key {rows}x{width} built {_DECODE_BUILT}; check the bucket file and dummy lengths', taken, peak
    return _capture_growth_error(key, graphs_before, graphs_after), taken, peak


def hbm_margin_seal_on() -> bool:
    """diag-hbm's margin seal is installed: HPU_GLM53_HBM_MARGIN_MIB is a positive MiB AND glm53_hbm_margin.pth put its wrapper over
    seal_boot_memory (the env alone, in an image without the .pth, judges nothing). Its own module validates the value; anything that is
    not a positive integer reads off here."""
    raw = os.getenv(HBM_MARGIN_FLAG, '').strip()
    return raw.isdigit() and int(raw) > 0 and _is_margin_wrapper(seal_boot_memory)


HBM_MARGIN_MARKER = 'glm53_hbm_margin_seal'  # True on the wrapper diag-hbm's install_from_env puts over seal_boot_memory (the contract)


def _is_margin_wrapper(seal: object) -> bool:
    """The installed margin seal: its marker, or (kits before the marker) one of the names those kits gave the wrapper."""
    return getattr(seal, HBM_MARGIN_MARKER, False) is True or getattr(seal, '__name__', '') in HBM_MARGIN_WRAPPERS


def hbm_margin_seals_decode_keys() -> bool:
    """The installed seal also replays every decode key with the lattice resident and refuses a seal with no reading: diag-hbm 7bf39655's
    glm53_hbm_margin.key_replays (prompt + decode). The 7ed2594b seal read prompt keys only, so a decode-only lattice passed unmeasured."""
    module = sys.modules.get(HBM_MARGIN_MODULE)
    return hbm_margin_seal_on() and callable(getattr(module, 'key_replays', None))


def _peak_error(key: tuple[int, int], peak: int) -> str:
    """After a clean capture: a peak past the estimate means the table is wrong for this build and the key fit on the margin only.
    It also keeps the budget whole: the lattice admission charged every key its estimate, and a key's residual never exceeds its peak.
    Under a margin seal that measures the decode keys too (hbm_margin_seals_decode_keys) the seal judges real headroom, so only a peak
    past the whole capture requirement (estimate + DECODE_CAPTURE_MARGIN_MIB) refuses: that capture outgrew its pre-capture check."""
    estimate = decode_key_estimate_bytes(*key) + listed_scan_workspace_bytes(key)
    sealed = hbm_margin_seals_decode_keys()
    if peak <= estimate + (DECODE_CAPTURE_MARGIN_MIB * MIB if sealed else 0):
        return ''
    why = (f' and the {DECODE_CAPTURE_MARGIN_MIB} MiB capture margin, so the pre-capture headroom check under-asks for it' if sealed
           else ', so a later capture could abort on PT_DEVMEM')
    return (f'decode key {key[0]}x{key[1]} peaked at {peak/MIB:.1f} MiB over its pre-capture in-use, past its {estimate/MIB:.1f} MiB estimate{why}; '
            f'raise DECODE_KEY_MIB_PER_ROW[{key[1]}] to {-(-peak // (key[0] * MIB))} or more (glm53_mixed_graph.py) from this reading')


def _charged_decode_bytes(key: tuple[int, int], taken: int) -> int:
    """The bytes a captured decode key adds to the boot witness's charged_MiB (the lattice admission charged estimates before any
    capture): what it kept under the installed HBM margin seal, else its estimate (never under what it kept)."""
    return taken if hbm_margin_seal_on() else max(decode_key_estimate_bytes(*key), taken)


def _capture_decode_key(runner: Any, key: tuple[int, int]) -> int:
    """Capture one decode key behind a headroom check; the bytes it adds to the witness's charged_MiB (_charged_decode_bytes)."""
    headroom_error, headroom = _decode_headroom_error(key)
    admit_capture(key, 0, 0, headroom_error, DECODE_HEADROOM_ADVICE)
    error, taken, peak = _run_decode_capture(runner, key)
    if not error:
        check_capture_scalars(f'{key[0]}x{key[1]}', key[1] * runner.attn_block_size)
        error = _peak_error(key, peak)
    admit_capture(key, 0, 0, error, DECODE_ADVICE)
    mark_decode_key_captured(key)
    LOG.warning('GLM53_DECODE_KEY captured rows=%d width=%d estimate_MiB=%.1f hbm_delta_MiB=%.1f peak_MiB=%.1f headroom_MiB=%.1f',
                key[0], key[1], decode_key_estimate_bytes(*key) / MIB, taken / MIB, peak / MIB, headroom / MIB)
    return _charged_decode_bytes(key, taken)


def _hold_ballast(nbytes: int, while_held: Callable[[], None]) -> int:
    """Allocate `nbytes` on the device, read a byte back from each tensor, run `while_held` under them, release them; the free HBM with them held.
    The byte is the proof they materialized: under PT_COMPILE_ONLY_MODE the ones may not, while a ballast that forces reclaim grows InUse by less than itself."""
    import habana_frameworks.torch as htorch
    mib = nbytes // MIB
    sizes = [BALLAST_CHUNK_MIB] * (mib // BALLAST_CHUNK_MIB) + [mib % BALLAST_CHUNK_MIB]
    held = [torch.ones(size * MIB, dtype=torch.uint8, device='hpu') for size in sizes if size]
    htorch.core.mark_step()
    htorch.hpu.synchronize()
    if any(int(tensor[-1].item()) != 1 for tensor in held):
        raise RuntimeError(f'the {mib} MiB ballast does not read back, so it never reached the device; run the seal outside compile-only mode')
    free = htorch.hpu.mem_get_info()[0]
    while_held()
    del held
    return free


def release_capture_workspace(bucket: tuple) -> None:
    """The widest key's capture leaves workspace the runtime hands back only when a launch does not fit. A fill just past free is that launch.
    Without it the next captures run free down to 1 MiB and a rank aborts on PT_DEVMEM (c1p-20261107T071305Z)."""
    before = _free_hbm()
    LOG.warning('GLM53_BEFORE_WORKSPACE_RELEASE bucket=%s free_MiB=%d overshoot_MiB=%d flag=%s; an abort here means nothing was left to hand back: '
                'unset the flag', bucket, before // MIB, RELEASE_OVERSHOOT_MIB, WORKSPACE_RELEASE_FLAG)
    try:
        held = _hold_ballast(before // MIB * MIB + RELEASE_OVERSHOOT_MIB * MIB, lambda: None)
    except Exception as exc:
        graphs_only.arm_exit('GLM53_WORKSPACE_RELEASE_FAILED')
        raise RuntimeError(f'{WORKSPACE_RELEASE_FLAG}: the fill past {before // MIB} MiB free found nothing to hand back ({type(exc).__name__}: {exc}); '
                           'unset the flag and drop a key') from exc
    # A read here is one fence early (fast3r: 5,052); the next capture's free_before_bytes shows what came back (9,924).
    LOG.warning('GLM53_KEY_WORKSPACE_RELEASE bucket=%s free_MiB before=%d held=%d released_while_held=%d',
                bucket, before // MIB, held // MIB, (held + (before // MIB * MIB + RELEASE_OVERSHOOT_MIB * MIB) - before) // MIB)


def _replay_prompt_keys(cache: Any) -> None:
    """One replay of each prompt key on its capture's own inputs, widest first as they were captured. Dummy prompts write only the null block 0,
    which serving never allocates. A rank that fails may leave its peers in the graph's collective, so the failure arms the bounded exit."""
    import habana_frameworks.torch as htorch
    for (bucket, _), (_, graph, _) in getattr(cache, 'entries', {}).items():
        if bucket[0] != 0:
            continue
        label = _key_setting(bucket)[0]
        LOG.warning('GLM53_BEFORE_SEAL_REPLAY key=%s flag=%s', label, SEAL_BALLAST_FLAG)
        try:
            cache.backend.replay(graph)
            htorch.hpu.synchronize()
        except Exception as exc:
            raise RuntimeError(f'prompt key {label} did not replay under it: {type(exc).__name__}: {exc}') from exc
        LOG.warning('GLM53_SEAL_REPLAY key=%s', label)


def hold_seal_ballast(cache: Any) -> None:
    """R5: the captures compile with no ballast held, so the first replay's workspace shows only at serving. Hold the K1-measured size and replay
    every prompt key under it; a rank that cannot refuses them all, with the size and the setting."""
    ballast = positive_env(SEAL_BALLAST_FLAG, SEAL_BALLAST_MIB) * MIB
    error, free = '', 0
    LOG.warning('GLM53_BEFORE_SEAL_BALLAST ballast_MiB=%d flag=%s; an abort here means the keys leave less than that free: %s',
                ballast // MIB, SEAL_BALLAST_FLAG, SEAL_ADVICE)
    try:
        free = _hold_ballast(ballast, lambda: _replay_prompt_keys(cache))
    except Exception as exc:
        graphs_only.arm_exit('GLM53_SEAL_BALLAST_FAILED')
        error = (f'the seal could not hold a {ballast // MIB} MiB ballast after the captures and replay the prompt keys under it ({type(exc).__name__}: {exc}), '
                 f'so a first replay would find less workspace than that; lower {SEAL_BALLAST_FLAG} only against a held-ballast reading')
    admit_capture(('seal_ballast',), 0, 0, error, SEAL_ADVICE)
    LOG.warning('GLM53_SEAL_BALLAST ballast_MiB=%d free_with_ballast_MiB=%.1f', ballast // MIB, free / MIB)


OFF_GRAPH: Counter = Counter()  # forwards the boot ran outside any graph that GraphCache does not see (count_off_graph)
_EAGER_AT_WINDOW_START = [0]
OFF_GRAPH_FIXES = {'adaptive_tile_warm': 'unset HPU_GLM53_WARM_ADAPTIVE_TILES: its widened indexer shapes are eager forwards no graph replays',
                   'routed_eager': 'find the eager forward the route ran during the keys and the lattice'}


def count_off_graph(kind: str, count: int | None = 1) -> None:
    OFF_GRAPH[kind] += count or 0


def _routed_eager_forwards() -> int:
    state = sys.modules.get('glm53_graph_state')
    return state.STATS.eager_forwards if state is not None else 0


def mark_boot_window() -> None:
    """Start of the keyed boot: eager forwards the guard's route runs from here to the witness are off-graph boot forwards."""
    OFF_GRAPH.clear()
    _EAGER_AT_WINDOW_START[0] = _routed_eager_forwards()


def _witness_counts(cache: Any) -> dict[str, int]:
    counts = getattr(cache, 'counts', {})
    seen = {name: counts.get(name, 0) for name in ('native_fallbacks', 'fallback_not_warmed', 'lazy_fallbacks')}
    return {**seen, 'routed_eager': _routed_eager_forwards() - _EAGER_AT_WINDOW_START[0], 'adaptive_tile_warm': OFF_GRAPH['adaptive_tile_warm']}


def _log_boot_witness(runner: Any, keys: tuple, charged: int, budget: int) -> None:
    """The one line naming every bucket this boot holds. A forward that fell back before serving refuses the boot."""
    cache = getattr(runner.model, '_glm53_mixed_graph_cache', None)
    fallbacks = _witness_counts(cache)
    bad = {name: count for name, count in fallbacks.items() if count}
    fixes = ' '.join(OFF_GRAPH_FIXES[name] + '.' for name in bad if name in OFF_GRAPH_FIXES)
    error = f'boot ran off-graph forwards {bad}; every boot forward must be a captured key. {fixes}' if bad else ''
    admit_capture(('boot_witness',), 0, 0, error, f'Check {PROMPT_KEYS_FLAG} and restart all workers.')
    prompts = ','.join(':'.join(map(str, key[0][1:])) for key in getattr(cache, 'entries', {}))  # rows:width, a shared key rows:width:prompts
    decodes = ','.join(f'{rows}x{width}' for rows, width in keys) or 'none'
    LOG.warning('GLM53_BOOT_WITNESS prompt_keys=%s decode_keys=%s decode_pad=%s owner=%s %s charged_MiB=%.1f budget_MiB=%d free_MiB=%.1f',
                prompts, decodes, decode_pad_mode(), os.getenv(SHARED_OWNER_FLAG, '0'),
                ' '.join(f'{name}={count}' for name, count in fallbacks.items()), charged / MIB, budget // MIB, _free_hbm() / MIB)


def warmup_decode_lattice(runner: Any) -> int:
    """After the prompt keys: one capture per (rows, width); returns the bytes charged. Any failure refuses the boot on every rank."""
    keys = decode_keys()
    if not keys and not prompt_keys():
        return 0
    cache = getattr(runner.model, '_glm53_mixed_graph_cache', None)
    charged = getattr(cache, 'sealed_bytes', 0)
    budget = positive_env('VLLM_GAUDI_MIXED_GRAPH_BUDGET_MIB', 6144) * MIB
    if keys:
        admit_capture(('decode_lattice',), 0, 0, _decode_lattice_error(runner, keys, charged, budget), DECODE_ADVICE)
    for key in keys:
        charged += _capture_decode_key(runner, key)
    return charged


def seal_boot_memory(runner: Any, charged: int) -> None:
    """R5, outside the lattice's compile-only context, where a ballast really allocates: hold it with a replay under it, then the boot witness,
    then (HPU_GLM53_GC_FREEZE_AT_SEAL=1) the one heap freeze, once every key is captured and the witness has accepted the boot."""
    keys = decode_keys()
    if not keys and not prompt_keys():
        gc_freeze.skipped_without_keys(LOG.warning)
        return
    hold_seal_ballast(getattr(runner.model, '_glm53_mixed_graph_cache', None))
    _log_boot_witness(runner, keys, charged, positive_env('VLLM_GAUDI_MIXED_GRAPH_BUDGET_MIB', 6144) * MIB)
    gc_freeze.freeze_at_seal(LOG.warning, admit_capture)


def validate_warmup(runner: Any) -> tuple:
    if os.getenv('VLLM_GAUDI_MIXED_STEP', '0') != '1':
        raise ValueError('VLLM_GAUDI_MIXED_GRAPHS=1 requires VLLM_GAUDI_MIXED_STEP=1')
    exact_shapes_enabled()  # a bad split / exact-shape value or pairing fails the boot before any capture
    pinned = native_pin_enabled()  # so does a bad pin value or a pin without exact shapes
    gc_freeze.enabled()  # and a bad HPU_GLM53_GC_FREEZE_AT_SEAL value: refused here, not after the captures
    if prompt_keys():
        return validate_prompt_keys(runner)
    buckets = selected_buckets(runner.attn_block_size)
    if measured_estimate_bytes() is not None and len(buckets) != 1:
        raise ValueError(f'{ESTIMATE_ENV} is one measured graph; set WARMUP to a single bucket '
                         f'(got {len(buckets)}) or unset it for the calibrated ceiling')
    if pinned:
        check_native_widths(buckets, runner.attn_block_size, runner.block_size)
    # Each nativepin width graph is priced as its bucket's graph: widths <= c, so the bucket price is the ceiling.
    estimate = sum(estimate_bytes(b, runner.attn_block_size) * (len(native_widths(b[2])) if pinned else 1)
                   for b in buckets)
    budget = positive_env('VLLM_GAUDI_MIXED_GRAPH_BUDGET_MIB', 6144) * MIB
    if estimate > budget or len(buckets) > positive_env('VLLM_GAUDI_MIXED_GRAPH_MAX_GRAPHS', 4):
        raise ValueError(f'mixed graph warmup budget needs {estimate/MIB:.1f} MiB; '
                         'reduce WARMUP or raise BUDGET_MIB after HBM measurement')
    for d, p, c in buckets:
        if c * runner.attn_block_size > runner.max_model_len:
            raise ValueError(f'mixed warmup bucket {(d, p, c)} exceeds max_model_len')
        context = c * runner.attn_block_size - p
        if context % runner.block_size:
            raise ValueError('mixed warmup sparse tier must align to runner.block_size')
        warmup_decode_rows(runner, d)
    return buckets
