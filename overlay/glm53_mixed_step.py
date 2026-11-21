"""Opt-in lazy mixed GLM step; regional kernels retain existing metadata.

Install this module beside the overlay's glm53_r589_bounded_prefill.py.
Only real tokens traverse the model's token-wise operations. Padding is local
to each attention/state kernel and is discarded before returning to the layer.
"""
from __future__ import annotations

from contextlib import contextmanager, nullcontext
from collections import Counter, deque
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from functools import lru_cache
import logging
import math
import os
import time
from typing import Any, Callable, ContextManager, Iterator

import torch
from glm53_mixed_tail_fuse import (attach as attach_tail_fuse, bind_region as bind_tail_fuse,
                                   enabled as tail_fuse_enabled, finish as finish_tail_fuse,
                                   mark_written as mark_tail_fuse)
from mixed_recurrent import _check_indices

logger = logging.getLogger(__name__)


def enabled() -> bool:
    return os.getenv('VLLM_GAUDI_MIXED_STEP', '0') == '1'


NATIVE_NO_DECODE = 'VLLM_GAUDI_MIXED_NATIVE_NO_DECODE'


def native_no_decode() -> bool:
    """C1 drift fix (default on): a prompt-only step has nothing to mix, so it runs the native forward."""
    value = os.getenv(NATIVE_NO_DECODE, '1')
    if value not in ('0', '1'):
        raise ValueError(f'{NATIVE_NO_DECODE}={value!r} must be 0 or 1; unset it to keep the default 1')
    return value == '1'


def prefill_budget() -> int:
    name = 'VLLM_GAUDI_MIXED_PREFILL_TOKENS'
    try:
        value = int(os.getenv(name, '512'))
    except ValueError as exc:
        raise ValueError(f'{name} must be a positive integer') from exc
    if value < 1:
        raise ValueError(f'{name} must be a positive integer')
    return value


def single_prompt_remaining(owner: Any, remaining: int, scheduled: dict[str, int],
                            prompts: set[str], *, from_waiting: bool = False) -> int:
    """Close a mixed step's prompt pool after its first committed prompt.

    Use scheduled request IDs, not running membership: parked/stale decodes
    must not reduce prompt throughput. Called only after allocation succeeds.
    """
    if slot_policy_active() or _prompt_keys_set():
        # A single-prompt key holds one prompt, so a second one in the step is a key miss, and under graphs-only a refusal on every rank.
        # Only VLLM_GAUDI_MIXED_PROMPTS_PER_STEP keeps the pool open, and only for the prompts the pick named.
        return remaining if tail_first_enabled() and len(prompts) < prompts_per_step() else 0
    if (os.getenv('VLLM_GAUDI_MIXED_GRAPH_SINGLE_PROMPT', '0') != '1'
            or os.getenv('VLLM_GAUDI_MIXED_GRAPHS', '0') != '1'):
        return remaining
    counts = getattr(owner, '_glm53_single_prompt_counts', None)
    if counts is None:
        counts = Counter(dict.fromkeys(('no_decode', 'budget_full', 'running_tail',
                                        'waiting_tail', 'withheld_tokens', 'events'), 0))
        setattr(owner, '_glm53_single_prompt_counts', counts)
    has_decode = any(req not in prompts for req in scheduled)
    reason = ('no_decode' if not has_decode else 'budget_full' if remaining <= 0
              else 'waiting_tail' if from_waiting else 'running_tail')
    counts[reason] += 1
    counts['events'] += 1
    if has_decode:
        counts['withheld_tokens'] += max(0, remaining)
    if counts['events'] == 1 or counts['events'] % 100 == 0:
        logger.info('mixed_single_prompt counts=%s', dict(counts))
    return 0 if has_decode else remaining


# Steps 43, 44, 49 (DUAL-SUSTAINED-PLAN 12.2, 12.3b): which steps carry a prompt chunk, and whose. K is a step count from a table, so the
# decode slice replays exactly. A prompt's age is wall clock, because 120 s is a promise in real time: which prompt a chunk goes to can
# differ between two runs. Bitwise check 3 stays valid because it compares tokens, and a lane's tokens depend only on its own rows, read
# whole and in order under any schedule (tests: ScheduleIndependence, WallClockAging). All flags empty: nothing here is read or kept.
SLICE_FLAG = 'VLLM_GAUDI_MIXED_SLICE_STEPS'
TAIL_FIRST_FLAG = 'VLLM_GAUDI_MIXED_TAIL_FIRST'
TAIL_AGE_FLAG = 'VLLM_GAUDI_MIXED_TAIL_AGE_S'
STEP_SECONDS_FLAG = 'VLLM_GAUDI_MIXED_STEP_SECONDS'
PROMPTS_PER_STEP_FLAG = 'VLLM_GAUDI_MIXED_PROMPTS_PER_STEP'
BATCH_KEYS_FLAG = 'VLLM_GAUDI_MIXED_PROMPT_BATCH_KEYS'  # P:R:W, as the single-prompt keys' R:W: one graph per key, BOOT captures
BATCH_KEY_ROWS_QUANTUM = 128  # as the single-prompt keys: a chunk's rows end on a block
SEPARATE_STEPS_FLAG = 'VLLM_GAUDI_MIXED_SEPARATE_STEPS'  # lab only (dual_serial_cell.py's separate-step arm): prompt-only and decode-only steps
PHASE_MIXED, PHASE_DECODE, PHASE_PROMPT = 'mixed', 'decode', 'prompt'
RECENT_CYCLES = 16  # slot-to-slot wall times the age margin reads
DEFAULT_TAIL_AGE_S = 120.0
DEFAULT_STEP_SECONDS = 'decode:0.107,1024:1.104,4096:2.4,8192:3.85'  # plan 12.1/12.3 readings: d, then p by table width


def _parse_width(flag: str, item: str, text: str) -> int:
    if not text.isdigit() or int(text) < 1 or int(text) & (int(text) - 1):
        raise ValueError(f'{flag} entry {item!r}: the width must be a power of two, the only table widths the runner builds')
    return int(text)


@lru_cache(maxsize=8)
def _parsed_slice_table(spec: str) -> tuple[tuple[int, int], ...]:
    entries: dict[int, int] = {}
    for item in filter(None, spec.split(',')):
        width, _, steps = item.partition(':')
        if not steps.isdigit():
            raise ValueError(f'{SLICE_FLAG} must list WIDTH:STEPS pairs (e.g. 1024:10,8192:36); got {item!r}')
        if _parse_width(SLICE_FLAG, item, width) in entries:
            raise ValueError(f'duplicate {SLICE_FLAG} width in {item!r}; list each width once')
        entries[int(width)] = int(steps)
    return tuple(sorted(entries.items()))


def slice_table() -> tuple[tuple[int, int], ...]:
    """(table width in blocks, decode-only steps before one chunk), narrowest first; empty = no slice."""
    return _parsed_slice_table(os.getenv(SLICE_FLAG, ''))


def tail_first_enabled() -> bool:
    value = os.getenv(TAIL_FIRST_FLAG, '') or '0'
    if value not in ('0', '1'):
        raise ValueError(f'{TAIL_FIRST_FLAG}={value!r} must be 0 or 1; unset it to keep FIFO')
    return value == '1'


def slot_policy_active() -> bool:
    return bool(slice_table()) or tail_first_enabled()


def separate_steps_enabled() -> bool:
    value = os.getenv(SEPARATE_STEPS_FLAG, '') or '0'
    if value not in ('0', '1'):
        raise ValueError(f'{SEPARATE_STEPS_FLAG}={value!r} must be 0 or 1; unset it to keep one mixed step')
    return value == '1'


def separate_phase(owner: Any, *, decode_ready: bool, prompt_ready: bool) -> str:
    """Which kind of step this one is under SEPARATE_STEPS: PHASE_MIXED = nothing withheld (flag off, or only one kind has work), else the
    kind that runs while the other waits. With both ready the two take turns, the one that did not run last going first (decode, at first).
    ``prompt_ready`` is False for a step the slice table already made decode-only, which counts as the decode turn."""
    if not separate_steps_enabled():
        return PHASE_MIXED
    last = getattr(owner, '_glm53_separate_last', None)
    if decode_ready and prompt_ready:
        phase = ran = PHASE_PROMPT if last == PHASE_DECODE else PHASE_DECODE
    else:
        phase, ran = PHASE_MIXED, PHASE_DECODE if decode_ready else PHASE_PROMPT if prompt_ready else last
    owner._glm53_separate_last = ran
    return phase


def refuse_mixed_step_when_separate(prompt_req_ids: Sequence[str], decode_req_ids: Sequence[str], *, warmup_mode: bool = False) -> None:
    """The scheduler's seam must have split the step; a step with both under the flag means a scheduler without it, which would make the
    separate-step arm a copy of the same-step arm."""
    if warmup_mode or not (prompt_req_ids and decode_req_ids) or not separate_steps_enabled():
        return
    raise ValueError(f'{SEPARATE_STEPS_FLAG}=1 but this step carries {len(prompt_req_ids)} prompt and {len(decode_req_ids)} decode requests: '
                     'install the paired scheduler (glm53 overlay vllm/v1/core/sched/scheduler.py), or unset the flag')


def _prompt_keys_set() -> bool:
    from glm53_mixed_graph import prompt_keys  # not at the top: that module imports this one's types when it runs
    return bool(prompt_keys())


def prompts_per_step() -> int:
    """Step 49: prompts one forward may carry. Default 1, one prompt per step."""
    text = os.getenv(PROMPTS_PER_STEP_FLAG, '') or '1'
    if not text.isdigit() or int(text) < 1:
        raise ValueError(f'{PROMPTS_PER_STEP_FLAG}={text!r} must be a positive integer; unset it to keep one prompt per step')
    return int(text)


@lru_cache(maxsize=8)
def _parsed_batch_keys(spec: str) -> tuple[tuple[int, int, int], ...]:
    keys: list[tuple[int, int, int]] = []
    for item in filter(None, spec.split(',')):
        parts = item.split(':')
        if len(parts) != 3 or not all(part.isdigit() for part in parts):
            raise ValueError(f'{BATCH_KEYS_FLAG} must list PROMPTS:ROWS:WIDTH (e.g. 4:256:1024); got {item!r}')
        prompts, rows, width = int(parts[0]), int(parts[1]), _parse_width(BATCH_KEYS_FLAG, item, parts[2])
        if prompts < 2 or rows < 1 or rows % BATCH_KEY_ROWS_QUANTUM:
            raise ValueError(f'{BATCH_KEYS_FLAG} key {item!r}: at least 2 prompts (one prompt is a VLLM_GAUDI_MIXED_PROMPT_KEYS key), rows a '
                             f'positive multiple of {BATCH_KEY_ROWS_QUANTUM}')
        if (prompts, rows, width) in keys:
            raise ValueError(f'duplicate {BATCH_KEYS_FLAG} key {item!r}; list each key once')
        keys.append((prompts, rows, width))
    return tuple(keys)


def prompt_batch_keys() -> tuple[tuple[int, int, int], ...]:
    """(prompts, rows per prompt, table width in blocks) of one padded multi-prompt graph per key; empty = none."""
    return _parsed_batch_keys(os.getenv(BATCH_KEYS_FLAG, ''))


def rows_per_prompt() -> int:
    """What each prompt of a shared step runs: the step's budget split evenly, so the padded rows of a key are P x this."""
    return prefill_budget() // prompts_per_step()


def shared_prompt_merge(held: int, incoming: int) -> bool | None:
    """The runner's question, may these prompts share a forward? None: the flag is off, its own rules answer. Else the scheduler already
    chose them, so only the count is left."""
    prompts = prompts_per_step()
    return None if prompts == 1 else held + incoming <= prompts


def check_shared_prompt_batch(max_prefill_batch_size: int) -> None:
    """Refuse a runner whose bucket could not hold a shared step: it would split it into a forward per prompt, in silence."""
    prompts = prompts_per_step()
    if prompts > max_prefill_batch_size:
        raise ValueError(f'{PROMPTS_PER_STEP_FLAG}={prompts} needs a prompt bucket that holds {prompts} rows; the runner tops out at '
                         f'{max_prefill_batch_size}. Set VLLM_PROMPT_BS_BUCKET_MAX={prompts} or lower the flag')


def batch_key_width(width: int) -> int | None:
    """The table width of the narrowest key this step's split can use for a chunk ending at ``width`` blocks; None = no such key."""
    prompts, rows = prompts_per_step(), rows_per_prompt()
    return min((key[2] for key in prompt_batch_keys() if key[:2] == (prompts, rows) and key[2] >= width), default=None)


def _for_width(table: Sequence[tuple[int, Any]], width: int) -> Any:
    """The narrowest entry that holds ``width``; past the top entry, the top entry."""
    return next((value for entry, value in table if entry >= width), table[-1][1])


@lru_cache(maxsize=8)
def _parsed_step_seconds(spec: str) -> tuple[float, tuple[tuple[int, float], ...]]:
    decode, chunks = None, {}
    for item in filter(None, spec.split(',')):
        name, _, text = item.partition(':')
        try:
            seconds = float(text)
        except ValueError as exc:
            raise ValueError(f'{STEP_SECONDS_FLAG} entry {item!r} needs seconds (e.g. {DEFAULT_STEP_SECONDS})') from exc
        if not seconds > 0:
            raise ValueError(f'{STEP_SECONDS_FLAG} entry {item!r} needs positive seconds')
        if name == 'decode':
            decode = seconds
        else:
            chunks[_parse_width(STEP_SECONDS_FLAG, item, name)] = seconds
    if decode is None or not chunks:
        raise ValueError(f'{STEP_SECONDS_FLAG} needs decode:SECONDS and at least one WIDTH:SECONDS (e.g. {DEFAULT_STEP_SECONDS})')
    return decode, tuple(sorted(chunks.items()))


def step_seconds() -> tuple[float, tuple[tuple[int, float], ...]]:
    """(nominal decode-only step, nominal chunk step by table width): how aging counts seconds without a clock."""
    return _parsed_step_seconds(os.getenv(STEP_SECONDS_FLAG, '') or DEFAULT_STEP_SECONDS)


def tail_age_limit() -> float:
    text = os.getenv(TAIL_AGE_FLAG, '') or str(DEFAULT_TAIL_AGE_S)
    advice = f'{TAIL_AGE_FLAG}={text!r} must be a positive, finite number of seconds; unset it to keep {DEFAULT_TAIL_AGE_S}'
    try:
        seconds = float(text)
    except ValueError as exc:
        raise ValueError(advice) from exc
    if not math.isfinite(seconds) or seconds <= 0:  # NaN compares false with every age, which would turn aging off in silence
        raise ValueError(advice)
    return seconds


def _widest_cycle_seconds() -> float:
    """The longest a prompt can wait for the slot after this one: a full K of decode-only steps, then one chunk."""
    decode, chunks = step_seconds()
    table = slice_table()
    return max(seconds + (_for_width(table, width) if table else 0) * decode for width, seconds in chunks)


def _check_prompts_per_step_flags() -> None:
    prompts = prompts_per_step()
    if prompts == 1:
        if prompt_batch_keys():
            raise ValueError(f'{BATCH_KEYS_FLAG} only applies to shared steps; set {PROMPTS_PER_STEP_FLAG} above 1 or unset the keys')
        return
    if not tail_first_enabled():
        raise ValueError(f'{PROMPTS_PER_STEP_FLAG}={prompts} picks its prompts by tail-first with aging; set {TAIL_FIRST_FLAG}=1 or unset it')
    budget = prefill_budget()
    if budget % prompts:
        raise ValueError(f'VLLM_GAUDI_MIXED_PREFILL_TOKENS={budget} does not split evenly over {PROMPTS_PER_STEP_FLAG}={prompts}; '
                         'a padded key has the same rows for every prompt')
    if batch_key_width(1) is None:
        raise ValueError(f'{PROMPTS_PER_STEP_FLAG}={prompts} needs a key {prompts}:{budget // prompts}:WIDTH in {BATCH_KEYS_FLAG}: without '
                         'one no step can share. Add it or unset the flag; context and KV stay as they are')


def check_slot_flags(*, completion_first: bool) -> None:
    """Refuse the boot on a slot flag that would do nothing or fight the lane it would share the prefill order with."""
    _check_prompts_per_step_flags()
    if separate_steps_enabled() and not enabled():
        raise ValueError(f'{SEPARATE_STEPS_FLAG}=1 only applies to mixed steps; set VLLM_GAUDI_MIXED_STEP=1 or unset it')
    if not slot_policy_active():
        return
    if not enabled():
        raise ValueError(f'{SLICE_FLAG} and {TAIL_FIRST_FLAG} only apply to mixed steps; set VLLM_GAUDI_MIXED_STEP=1 or unset them')
    if not tail_first_enabled():
        return
    if completion_first:
        raise ValueError(f'{TAIL_FIRST_FLAG}=1 and VLLM_COMPLETION_FIRST_LONG_PREFILL=1 are two prefill orders; set the latter to 0')
    if tail_age_limit() <= _widest_cycle_seconds():
        raise ValueError(f'{TAIL_AGE_FLAG}={tail_age_limit()} cannot outlast one slice cycle ({_widest_cycle_seconds():.2f} s); '
                         f'raise it or shrink {SLICE_FLAG}')


def slice_steps(width: int, prompts: int) -> int:
    """K for a chunk ending at ``width``, as the table says, unless ``prompts`` taking turns would then outlast the age limit: one slot per
    K + 1 steps must get round them all in time. Read from the tables, never the clock, so K stays a replayable step count."""
    steps = _for_width(slice_table(), width)
    if not tail_first_enabled():
        return steps
    decode, chunks = step_seconds()
    room = tail_age_limit() / max(prompts, 1) - _for_width(chunks, width)
    return min(steps, max(0, int(room / decode)))


def decode_slice_allows(owner: Any, width: int | None, prompts: int = 1) -> bool:
    """False for K steps in a row, then True until a chunk runs. ``width`` None: before the chunk is known, whose estimate costs
    cache lookups; withholds only while every width in the table still would. Call only with a decode lane ready: K = 0 otherwise."""
    table = slice_table()
    if not table:
        return True
    withheld = getattr(owner, '_glm53_slice_withheld', 0)
    steps = (min(slice_steps(entry, prompts) for entry, _ in table) if width is None else slice_steps(width, prompts))
    if withheld >= steps:
        return True
    owner._glm53_slice_withheld = withheld + 1
    return False


def note_chunk_ran(owner: Any) -> None:
    if hasattr(owner, '_glm53_slice_withheld'):
        owner._glm53_slice_withheld = 0


def chunk_width(end_tokens: int, block_size: int) -> int:
    return math.ceil(end_tokens / block_size)


@dataclass(frozen=True)
class PromptCandidate:
    """A prompt that could take this step's chunk; ``start`` is computed for a running one, estimated for a waiting one."""

    request_id: str
    start: int
    end: int


@dataclass(frozen=True)
class SlotPick:
    request_id: str
    rows: int
    width: int
    aged: bool
    next_best_chunks: int | None  # fewest chunks left among the others, for the check after the real cache lookup
    companions: tuple[str, ...] = ()  # step 49: prompts sharing this step's forward; ``rows`` is then each one's share

    def admits(self, request_id: str) -> bool:
        return request_id == self.request_id or request_id in self.companions


def pick_rows(pick: SlotPick | None, remaining: int) -> int:
    """The rows a prompt may take now: all that is left, unless it shares the step."""
    return min(remaining, pick.rows) if pick is not None and pick.companions else remaining


def _chunks_left(start: int, end: int, rows: int) -> int:
    return max(1, math.ceil((end - start) / rows))


@dataclass
class SlotLedger:
    """Wall seconds since each live prompt's last chunk started (or it arrived), kept only for prompts still waiting for one."""

    clock: Callable[[], float] = time.monotonic
    now: float = 0.0
    last_chunk: dict[str, float] = field(default_factory=dict)
    looked_up: dict[str, int] = field(default_factory=dict)  # the real hit of a pick the hint oversold; replaces the hint
    mismatches: int = 0
    last_slot: float | None = None
    recent_cycles: deque[float] = field(default_factory=lambda: deque(maxlen=RECENT_CYCLES))
    holding: bool = False  # an aged waiter is holding new waiters back, so the log says when that starts

    def observe(self, request_ids: Iterable[str]) -> None:
        self.now = self.clock()
        live = set(request_ids)
        if not live:
            self.last_slot = None  # idle is not a slow cycle
        self.last_chunk = {request_id: self.last_chunk.get(request_id, self.now) for request_id in live}
        self.looked_up = {request_id: start for request_id, start in self.looked_up.items() if request_id in live}

    def age_margin(self, prompts: int) -> float:
        """How long before the limit a prompt counts as aged: every prompt ahead of it takes a cycle, at least as long as the card's lately."""
        return prompts * max(_widest_cycle_seconds(), max(self.recent_cycles, default=0.0))

    def is_aged(self, request_id: str, prompts: int) -> bool:
        return self.now - self.last_chunk.get(request_id, self.now) >= tail_age_limit() - self.age_margin(prompts)

    def pick(self, candidates: Sequence[PromptCandidate], rows: int, block_size: int) -> SlotPick:
        """Earliest deadline first once a prompt could not survive its turn in the queue; else the fewest chunks left, FIFO on ties."""
        aged = [item for item in candidates if self.is_aged(item.request_id, len(candidates))]
        chosen = (min(aged, key=lambda item: self.last_chunk[item.request_id]) if aged
                  else min(candidates, key=lambda item: _chunks_left(item.start, item.end, rows)))
        companions, shared_rows, width = self._companions(chosen, candidates, aged, rows, block_size)
        taken = {chosen.request_id, *(item.request_id for item in companions)}
        others = [_chunks_left(item.start, item.end, shared_rows) for item in candidates if item.request_id not in taken]
        return SlotPick(chosen.request_id, shared_rows, width, bool(aged), min(others, default=None),
                        tuple(item.request_id for item in companions))

    def _companions(self, chosen: PromptCandidate, candidates: Sequence[PromptCandidate], aged: Sequence[PromptCandidate], rows: int,
                    block_size: int) -> tuple[list[PromptCandidate], int, int]:
        """Step 49: (prompts sharing ``chosen``'s forward, each one's rows, the width that step costs). Oldest past the limit first, then the
        fewest chunks left. Only prompts whose chunks a single key holds: a tail sharing a wider prompt's table would pay for all of it."""
        alone = chunk_width(min(chosen.start + rows, chosen.end), block_size)
        if prompts_per_step() == 1 or len(candidates) == 1:
            return [], rows, alone
        share = rows_per_prompt()
        ends = {item.request_id: chunk_width(min(item.start + share, item.end), block_size) for item in candidates}
        width = batch_key_width(ends[chosen.request_id])
        if width is None:
            return [], rows, alone
        fits = [item for item in candidates if item is not chosen and batch_key_width(ends[item.request_id]) == width]
        fits.sort(key=lambda item: (item not in aged, self.last_chunk.get(item.request_id, self.now) if item in aged else 0.0,
                                    _chunks_left(item.start, item.end, share)))
        companions = fits[:prompts_per_step() - 1]
        return (companions, share, width) if companions else ([], rows, alone)

    def outgrew_key(self, pick: SlotPick, request_id: str, real_start: int, end: int, block_size: int) -> bool:
        """After the real lookup: True when a deeper hit than the hint would end past the shared key's table, which no shared key may hold."""
        if not pick.companions:
            return False
        width = batch_key_width(chunk_width(min(real_start + pick.rows, end), block_size))
        if width is not None and width <= pick.width:
            return False
        self.looked_up[request_id] = real_start
        self.mismatches += 1
        return True

    def oversold(self, pick: SlotPick, request_id: str, real_start: int, end: int) -> bool:
        """After the real lookup: True when the hint was too generous for this pick to still be the shortest."""
        if pick.aged or pick.next_best_chunks is None or _chunks_left(real_start, end, pick.rows) <= pick.next_best_chunks:
            return False
        self.looked_up[request_id] = real_start
        self.mismatches += 1
        return True

    def note_chunks(self, request_ids: Iterable[str]) -> None:
        """This step's prompts got their chunk. The age counts from its start, so a chunk's own run time is inside the limit."""
        for request_id in request_ids:
            self.last_chunk[request_id] = self.now
        if self.last_slot is not None:
            self.recent_cycles.append(self.now - self.last_slot)
        self.last_slot = self.now


def slot_ledger(owner: Any) -> SlotLedger:
    ledger = getattr(owner, '_glm53_slot_ledger', None)
    if ledger is None:
        ledger = owner._glm53_slot_ledger = SlotLedger()
    return ledger


@dataclass(frozen=True)
class Region:
    requests: tuple[int, ...]
    lengths: tuple[int, ...]
    starts: tuple[int, ...]
    width: int
    batch: int
    valid: tuple[int, ...]
    metadata: Any
    decode: bool
    graph_padded: bool = False
    tail_fuse: Any = None   # glm53_mixed_tail_fuse.TailFuseRows on prompt regions, VLLM_RECURRENT_TAIL_FUSE=1 only

    @property
    def count(self) -> int:
        return self.batch * self.width if self.graph_padded else sum(self.lengths)


@dataclass(frozen=True)
class MixedMetadata:
    regions: tuple[Region, ...]
    is_prompt: bool = False
    mixed_step: bool = True

    def __post_init__(self) -> None:
        requests = tuple(req for r in self.regions for req in r.requests)
        if len(set(requests)) != len(requests):
            raise ValueError('mixed step has duplicate request/state ownership')
        prompt_seen = False
        for r in self.regions:
            if r.decode and prompt_seen:
                raise ValueError('mixed layout requires decode rows before prompt rows')
            prompt_seen |= not r.decode
            if not (len(r.requests) == len(r.lengths) == len(r.starts)):
                raise ValueError('mixed region requires lengths/positions per request')
            if r.width < 1 or r.batch < len(r.requests):
                raise ValueError('mixed region bucket cannot hold its requests')
            if any(n < 1 or n > r.width for n in r.lengths):
                raise ValueError('mixed region length outside prepared bucket')
            if r.decode and any(n != 1 for n in r.lengths):
                raise ValueError('mixed v1 needs one decode token per request; disable MTP')
            expected = tuple(i*r.width+j for i,n in enumerate(r.lengths) for j in range(n))
            _check_indices(r.valid, r.batch*r.width, 'mixed kernel row')
            if r.valid != expected:
                raise ValueError('mixed region valid rows must preserve request/token order')

    @property
    def num_decode_tokens(self) -> int:
        return sum(r.count for r in self.regions if r.decode)

    @property
    def prefill_slices(self) -> list[tuple[int, int, int]]:
        return [(req, start, n) for r in self.regions if not r.decode
                for req, start, n in zip(r.requests, r.starts, r.lengths)]

    @property
    def request_indices(self) -> tuple[int, ...]:
        return tuple(req for r in self.regions for req, n in zip(r.requests, r.lengths)
                     for _ in range(n))

    @property
    def positions(self) -> tuple[int, ...]:
        return tuple(start+j for r in self.regions for start,n in zip(r.starts,r.lengths)
                     for j in range(n))


def sample_rows(plan: MixedMetadata, prompt_ends: dict[int, int]) -> tuple[int, ...]:
    rows, cursor = [], 0
    for r in plan.regions:
        for req, start, n in zip(r.requests, r.starts, r.lengths):
            if r.decode or start + n == prompt_ends[req]:
                rows.append(cursor + n - 1)
            cursor += n
    return tuple(rows)


def _indices(rows: tuple[int, ...], tensor: torch.Tensor) -> torch.Tensor:
    return torch.tensor(rows, dtype=torch.long, device=tensor.device)


def pack_region(x: torch.Tensor, r: Region, fill: int = 0) -> torch.Tensor:
    if r.graph_padded:
        return x
    padded = x.new_full((r.batch*r.width,) + x.shape[1:], fill)
    return padded.index_copy(0, _indices(r.valid, x), x)


def unpack_region(x: torch.Tensor, r: Region) -> torch.Tensor:
    if r.graph_padded:
        return x
    return x.index_select(0, _indices(r.valid, x))


def run_regions(x: torch.Tensor, plan: MixedMetadata,
                kernel: Callable[[torch.Tensor, Region], torch.Tensor]) -> torch.Tensor:
    parts, cursor = [], 0
    for r in plan.regions:
        parts.append(unpack_region(kernel(pack_region(x[cursor:cursor+r.count], r), r), r))
        cursor += r.count
    return torch.cat(parts)


@contextmanager
def region_context(metadata: Any) -> Iterator[None]:
    from vllm.forward_context import get_forward_context
    context = get_forward_context()
    previous = context.attn_metadata
    context.attn_metadata = metadata
    try:
        yield
    finally:
        context.attn_metadata = previous


def packed_regions(plan: MixedMetadata, tensors: tuple[torch.Tensor, ...],
                   fills: tuple[int, ...] | None = None) -> Iterator[tuple[Region, list[torch.Tensor]]]:
    """Each region with its packed rows of every tensor (pad ``fills``, default 0), run under its metadata."""
    cursor = 0
    for r in plan.regions:
        span = slice(cursor, cursor+r.count)
        values = [pack_region(t[span], r, fill)
                  for t, fill in zip(tensors, fills or (0,) * len(tensors), strict=True)]
        with region_context(r.metadata):
            yield r, values
        cursor += r.count


def regional_indexer(owner: Any, plan: MixedMetadata, hidden: torch.Tensor,
                     q: torch.Tensor, k: torch.Tensor, weights: torch.Tensor,
                     gate: torch.Tensor, ape: torch.Tensor,
                     positions: torch.Tensor) -> torch.Tensor:
    parts = []
    for r, values in packed_regions(plan, (hidden, q, k, weights, gate, positions)):
        result = owner.forward(*values[:4], gate_score=values[4],
                               compress_ape=ape, positions=values[5])
        # The legacy indexer owns a shared scratch buffer. Keep segment storage
        # independent before the next region overwrites that scratch.
        parts.append(unpack_region(result, r).clone())
    return torch.cat(parts)


def regional_mla(owner: Any, plan: MixedMetadata, q: torch.Tensor,
                 k: torch.Tensor, pe: torch.Tensor, cache: Any,
                 slots: torch.Tensor | None) -> torch.Tensor:
    if slots is None:
        raise RuntimeError('mixed sparse MLA requires explicit indexer slots; enable direct MLA')
    parts = []
    for r, values in packed_regions(plan, (q, k, pe, slots), fills=(0, 0, 0, -1)):
        result = owner.forward_impl(*values[:3], cache, r.metadata,
                                    hpu_sparse_slots=values[3])
        parts.append(unpack_region(result, r))
    return torch.cat(parts)


def regional_recurrent(owner: Any, plan: MixedMetadata, qkv: torch.Tensor,
                       gate: torch.Tensor, beta: torch.Tensor,
                       output: torch.Tensor) -> None:
    cursor = 0
    for r in plan.regions:
        span = slice(cursor, cursor+r.count)
        local_qkv = pack_region(qkv[span], r)
        local_gate = pack_region(gate[0, span], r).unsqueeze(0)
        local_beta = pack_region(beta[0, span], r).unsqueeze(0)
        local_output = output.new_zeros((1, r.batch*r.width) + output.shape[2:])
        # Keep original per-cache-group load/state indices. Never substitute
        # packed row numbers for physical recurrent/conv state slots.
        with region_context(r.metadata), bind_tail_fuse(r.tail_fuse):
            owner._forward_hpu(local_qkv, local_gate, local_beta, local_output)
        if r.graph_padded:
            rows = _indices(tuple(range(cursor, cursor+r.count)), output)
            output.index_copy_(1, rows, local_output)
        else:
            output[0, span].copy_(unpack_region(local_output[0], r))
        cursor += r.count


@dataclass
class PreparedBatch:
    region: Region
    tokens: torch.Tensor
    positions: torch.Tensor
    logits: tuple[int, ...]
    key: int


def _batch(runner: Any, ids: list[str], tokens: torch.Tensor, positions: torch.Tensor,
           metadata: Any, logits: tuple[int, ...], schedule: Any, decode: bool,
           trim: Callable | None = None) -> PreparedBatch:
    indices = tuple(runner.input_batch.req_id_to_index[req] for req in ids)
    lengths = tuple(schedule.num_scheduled_tokens[req] for req in ids)
    starts = tuple(int(runner.input_batch.num_computed_tokens_cpu[i]) for i in indices)
    batch, width = tokens.shape
    valid = tuple(i*width+j for i,n in enumerate(lengths) for j in range(n))
    processed = runner.model.metadata_processor.process_metadata(
        trim(metadata) if trim else metadata, batch, width, tokens.device, runner.dtype, False)
    region = Region(indices, lengths, starts, width, batch, valid, processed, decode)
    return PreparedBatch(region, tokens, positions, logits, id(metadata))


def _host_logits(ids: list[str], requests: list[str], lengths: tuple[int, ...],
                 width: int, padded_count: int) -> tuple[int, ...]:
    """Rebuild ship-v3's tail logits using CPU request labels, never HPU values."""
    counts = Counter(requests)
    offsets = {req: i*width+n-counts[req] for i, (req, n) in enumerate(zip(ids, lengths))}
    if padded_count < len(requests) or any(counts[req] > n for req, n in zip(ids, lengths)):
        raise ValueError('mixed logits exceed prepared rows; inspect prefill logits_requests')
    rows = []
    for req in requests:
        if req not in offsets:
            raise ValueError(f'mixed logits request {req!r} missing from prefill batch')
        rows.append(offsets[req])
        offsets[req] += 1
    return tuple(rows) + (-1,) * (padded_count-len(rows))


def _batches(runner: Any, schedule: Any, pd: Any, prefills: Any, decode: Any,
             trim: Callable | None = None, *, prompts_only: bool = False) -> list[PreparedBatch]:
    batches = []
    if pd.decode_req_ids and not prompts_only:  # a keyed step leaves its decode to the runner's own graph
        indices = tuple(range(decode.token_ids.numel()))
        batches.append(_batch(runner, pd.decode_req_ids, decode.token_ids, decode.position_ids,
                              decode.attn_metadata, indices, schedule, True, trim))
    for ids, tokens, positions, metadata, logits, requests in zip(
            prefills.request_ids, prefills.token_ids, prefills.position_ids,
            prefills.attn_metadata, prefills.logits_indices, prefills.logits_requests):
        lengths = tuple(schedule.num_scheduled_tokens[req] for req in ids)
        host = _host_logits(ids, requests, lengths, tokens.shape[1], logits.numel())
        batches.append(_batch(runner, ids, tokens, positions, metadata,
                              host, schedule, False, trim))
    return batches


def _validate_runner(runner: Any) -> None:
    if runner.vllm_config.speculative_config is not None:
        raise ValueError('VLLM_GAUDI_MIXED_STEP v1 requires MTP/speculation OFF')
    parallel = runner.vllm_config.parallel_config
    if parallel.data_parallel_size != 1 or parallel.pipeline_parallel_size != 1:
        raise ValueError('mixed v1 supports TP only; set DP=PP=1')
    if runner.lora_config or runner.use_aux_hidden_state_outputs or runner.use_merged_prefill:
        raise ValueError('mixed v1 needs LoRA/aux outputs/merged-prefill disabled')
    if runner.model_has_chunked_attention or runner.input_batch.num_prompt_logprobs:
        raise ValueError('mixed v1 needs chunked-attention/prompt-logprobs disabled')
    if not runner.model.flatten_input:
        raise ValueError('mixed v1 requires the flattened GLM model adapter')
    if runner.model.is_mm_optimized:
        raise ValueError('mixed v1 supports text-only GLM inputs')


def _sampling_map(batches: list[PreparedBatch]) -> tuple[tuple[int, ...], list[dict[int, int]]]:
    needed, maps, cursor = [], [], 0
    for batch in batches:
        mapping = {padded: cursor+i for i, padded in enumerate(batch.region.valid)}
        local = {}
        for i, row in enumerate(batch.logits):
            if row < 0:
                continue
            if row in mapping:
                local[i] = len(needed)
                needed.append(mapping[row])
            elif not batch.region.decode:
                raise ValueError('mixed prefill logit outside valid rows; inspect logits_requests')
        maps.append(local)
        cursor += batch.region.count
    return tuple(needed), maps


def _scatter_logits(logits: torch.Tensor, hidden: torch.Tensor, mapping: dict[int, int],
                    count: int) -> tuple[torch.Tensor, torch.Tensor]:
    out = logits.new_zeros((count,) + logits.shape[1:])
    samples = hidden.new_zeros((count,) + hidden.shape[1:])
    if mapping:
        dst = _indices(tuple(mapping), logits)
        src = _indices(tuple(mapping.values()), logits)
        out.index_copy_(0, dst, logits.index_select(0, src))
        samples.index_copy_(0, dst, hidden.index_select(0, src))
    return out, samples


def _results(batches: list[PreparedBatch], hidden: torch.Tensor, logits: torch.Tensor,
             sampled_hidden: torch.Tensor, maps: list[dict[int, int]]) -> dict[int, tuple]:
    results, cursor = {}, 0
    for batch, mapping in zip(batches, maps):
        r = batch.region
        full = pack_region(hidden[cursor:cursor+r.count], r)
        full = full.reshape(r.batch, r.width, -1)
        local_logits, local_samples = _scatter_logits(logits, sampled_hidden, mapping, len(batch.logits))
        results[batch.key] = (full, None, local_samples, local_logits)
        cursor += r.count
    return results


def _prompt_logits_padded(batches: list[PreparedBatch]) -> bool:
    """A -1 prompt logits row is native's hidden[-1] of the padded batch; the graph's compact rows cannot index it."""
    return any(row < 0 for batch in batches if not batch.region.decode for row in batch.logits)


def _native_logit_rows(batch: PreparedBatch) -> tuple[int, ...]:
    """Native's logits_indices for this batch, as indices into the region's compact (valid) hidden rows.

    Decode: logits_indices = zeros(padded batch) with [:num_decodes] = each request's last token
    (hpu_model_runner _create_decode_input_data), so pad rows read row 0. Prompt: its own host rows.
    """
    region = batch.region
    compact = {padded: index for index, padded in enumerate(region.valid)}
    if not region.decode:
        return tuple(compact[row] for row in batch.logits)
    last = [i * region.width + n - 1 for i, n in enumerate(region.lengths)]
    padded = last + [0] * (len(batch.logits) - len(last))
    return tuple(compact[row] for row in padded)


def _native_results(runner: Any, batches: list[PreparedBatch], hidden: torch.Tensor) -> dict[int, tuple]:
    """One compute_logits per batch at native's rows, in native's order (decode first), as _execute_model_generic."""
    results, cursor = {}, 0
    for batch in batches:
        r = batch.region
        local = hidden[cursor:cursor+r.count]
        sampled = local.index_select(0, _indices(_native_logit_rows(batch), local))
        full = pack_region(local, r).reshape(r.batch, r.width, -1)
        results[batch.key] = (full, None, sampled, runner.model.compute_logits(sampled))
        cursor += r.count
    return results


class Replay:
    def __init__(self, results: dict[int, tuple]) -> None:
        self.results = results

    def __call__(self, tokens: torch.Tensor, positions: torch.Tensor, metadata: Any,
                 *args: Any, **kwargs: Any) -> tuple:
        key = id(metadata)
        if key not in self.results:
            raise RuntimeError('mixed result missing/already consumed; inspect runner region dispatch')
        return self.results.pop(key)


def _key_logit_rows(batch: PreparedBatch) -> tuple[int, ...]:
    """Rows of a key's compact hidden. A -1 row (a non-final chunk, whose logits nobody samples) reads the last real row."""
    region = batch.region
    compact = {padded: index for index, padded in enumerate(region.valid)}
    return tuple(compact[row] if row >= 0 else len(region.valid) - 1 for row in batch.logits)


class KeyedForward:
    """Form B: the runner's dispatch, decode first on its own graph; a prompt batch through its key, else the runner's forward.

    ``tail_rows`` (VLLM_RECURRENT_TAIL_FUSE_KEYED): the step's attached tail-fuse rows. A replayed prompt batch is marked written and
    the bank is stamped once every prompt batch of the step has run; a batch routed native leaves its rows to tail fuse's host path.
    """

    def __init__(self, runner: Any, prompts: dict[int, PreparedBatch], tail_rows: Any = None) -> None:
        self.runner, self.prompts, self.tail_rows = runner, prompts, tail_rows
        self.ran: set[int] = set()

    def __call__(self, tokens: torch.Tensor, positions: torch.Tensor, metadata: Any, *args: Any, **kwargs: Any) -> tuple:
        import glm53_mixed_graph
        batch = self.prompts.get(id(metadata))
        hidden = None if batch is None else glm53_mixed_graph.forward_key(self.runner, batch)
        if batch is not None:
            self._note_ran(batch, graphed=hidden is not None)
        if hidden is None:
            return self.runner._execute_model_generic(tokens, positions, metadata, *args, **kwargs)
        region = batch.region
        sampled = hidden.index_select(0, _indices(_key_logit_rows(batch), hidden))
        full = pack_region(hidden, region).reshape(region.batch, region.width, -1)
        return full, None, sampled, self.runner.model.compute_logits(sampled)

    def _note_ran(self, batch: PreparedBatch, *, graphed: bool) -> None:
        """Keyed tail fuse only: mark a key replay written; the last prompt batch of the step stamps the bank rows the graph wrote."""
        if self.tail_rows is None:
            return
        if graphed:
            mark_tail_fuse(self.tail_rows, batch, True)
        self.ran.add(id(batch))
        if len(self.ran) == len(self.prompts):
            finish_tail_fuse(self.tail_rows)


def _keyed_tail_rows(runner: Any, batches: list[PreparedBatch]) -> Any:
    """VLLM_RECURRENT_TAIL_FUSE_KEYED: every prompt region's tail-fuse device rows before its key runs, the boot's dummies included so
    the key's signature carries the rows and a fused step replays the same graph. None with the flag off: the keyed step the image runs."""
    import glm53_mixed_graph
    if not glm53_mixed_graph.tail_fuse_keyed_enabled():
        return None
    return attach_tail_fuse(runner.input_batch.req_ids, batches)


def prepare(runner: Any, schedule: Any, pd: Any, prefills: Any, decode: Any, *,
            trim_metadata: Callable | None = None, warmup_mode: bool = False,
            warmup_context: ContextManager | None = None) -> Callable:
    if not enabled():
        return runner._execute_model_generic
    refuse_mixed_step_when_separate(pd.prompt_req_ids, pd.decode_req_ids, warmup_mode=warmup_mode)
    _validate_runner(runner)
    native = native_no_decode()
    if not pd.prompt_req_ids:
        return runner._execute_model_generic
    import glm53_mixed_graph
    if glm53_mixed_graph.prompt_keys():
        batches = _batches(runner, schedule, pd, prefills, decode, trim_metadata, prompts_only=True)
        return KeyedForward(runner, {batch.key: batch for batch in batches}, _keyed_tail_rows(runner, batches))
    prompt_only = native and not pd.decode_req_ids
    if prompt_only and not tail_fuse_enabled():
        return runner._execute_model_generic
    batches = _batches(runner, schedule, pd, prefills, decode, trim_metadata)
    # Every prompt batch gets its tail-fuse device rows before any routing; only a batch the mixed forward marks
    # written is stamped, so rows of a batch routed native are left to tail fuse's host path.
    tail_rows = attach_tail_fuse(runner.input_batch.req_ids, batches)
    if prompt_only:
        return runner._execute_model_generic
    plan = MixedMetadata(tuple(b.region for b in batches))
    if not warmup_mode and sum(n for _,_,n in plan.prefill_slices) > prefill_budget():
        raise ValueError('mixed prompt budget exceeded; install the paired scheduler patch')
    needed, maps = _sampling_map(batches)
    pinned = glm53_mixed_graph.native_pin_enabled()
    if pinned and _prompt_logits_padded(batches):
        return runner._execute_model_generic  # before any forward: native owns a padded prompt logits row
    hidden = glm53_mixed_graph.forward(runner, batches)
    graphed = hidden is not None
    if hidden is None and glm53_mixed_graph.exact_shapes_enabled() and not warmup_mode:
        # C576: no graph at the native shapes -> the reference's own per-batch path, never the lazy mixed forward.
        return runner._execute_model_generic
    if hidden is None:
        tokens = torch.cat([unpack_region(b.tokens.reshape(-1), b.region) for b in batches])
        positions = torch.cat([unpack_region(b.positions.reshape(-1), b.region) for b in batches])
        with warmup_context if warmup_context is not None else nullcontext():
            hidden = runner.model.forward(input_ids=tokens.unsqueeze(0), positions=positions.unsqueeze(0),
                                          attn_metadata=plan, bypass_hpu_graphs=True)
    for batch in batches:
        mark_tail_fuse(tail_rows, batch, graphed)
    finish_tail_fuse(tail_rows)
    hidden = hidden.reshape(-1, hidden.shape[-1])
    if pinned:
        return Replay(_native_results(runner, batches, hidden))
    sampled_hidden = hidden.index_select(0, _indices(needed, hidden))
    logits = runner.model.compute_logits(sampled_hidden)
    return Replay(_results(batches, hidden, logits, sampled_hidden, maps))
