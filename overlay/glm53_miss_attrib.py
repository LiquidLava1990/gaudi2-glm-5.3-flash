"""Offload miss attribution, log only (VLLM_OFFLOAD_MISS_ATTRIB=1, default off).

One GLM53_MISS line per request, at its first resolved lookup, says where the prompt's missing tokens went:

- ``backtrack``: tokens the attention cache hit but the recurrent snapshot store did not. ``class`` names why: ``evicted``
  (one of the walked snapshot keys is in the evicted-key LRU), ``never_stored`` (none is) or ``none`` (no backtrack).
- ``attn_class``: the same three answers for the first attention chunk past the attention hit.
- ``lane``, ``prev_prompt_end``, ``lcp``: the lane's previous prompt and the prefix shared with it. A large ``lcp`` above
  the attention hit on a ``never_stored`` key points at another replica or a template rewrite, not at this cache.

The evicted-key LRUs are fed by the managers' removal listeners, which also fire for a failed store; those are rare and
their keys are gone either way. reset_cache does not notify, so keys dropped by a reset read as ``never_stored``.
"""

from __future__ import annotations

import os
from collections import OrderedDict
from collections.abc import Callable, Hashable, Iterable, Mapping, Sequence
from typing import NamedTuple

FLAG_ENV = "VLLM_OFFLOAD_MISS_ATTRIB"
LANE_BLOCK_ENV = "VLLM_OFFLOAD_MISS_ATTRIB_LANE_BLOCK"
LANE_BLOCK_DEFAULT = 64
EVICTED_KEYS_KEPT = 4096
LANES_KEPT = 256
EVICTED, NEVER_STORED, NONE = "evicted", "never_stored", "none"
REASON_LOOKUP, REASON_COLD, REASON_SKIP_READ = "lookup", "cold", "skip_read"
NO_PREVIOUS = -1
NO_LANE = "-"
LANE_ID_HEX = 16


class EvictedKeys:
    """The most recent EVICTED_KEYS_KEPT removed keys, oldest out."""

    def __init__(self, capacity: int = EVICTED_KEYS_KEPT) -> None:
        self.capacity = capacity
        self.keys: OrderedDict[Hashable, None] = OrderedDict()

    def add(self, keys: Iterable[Hashable]) -> None:
        for key in keys:
            self.keys[key] = None
            self.keys.move_to_end(key)
        while len(self.keys) > self.capacity:
            self.keys.popitem(last=False)

    def __contains__(self, key: Hashable) -> bool:
        return key in self.keys

    def __len__(self) -> int:
        return len(self.keys)


class MissLookup(NamedTuple):
    """What one request's resolved lookup found. Hits are tokens beyond ``local_tokens``."""

    request_id: str
    block_hashes: Sequence[bytes]
    prompt_tokens: int
    local_tokens: int
    attention_hit: int
    recurrent_hit: int
    walked_recurrent_keys: Sequence[Hashable]  # the snapshot keys the sidecar walk found missing
    missing_attention_key: Hashable | None  # the first chunk past the attention hit, if the manager says it is absent
    store_wait_ms: float
    reason: str


class PreviousTurn(NamedTuple):
    prompt_tokens: int
    block_hashes: tuple[bytes, ...]


def common_prefix_blocks(first: Sequence[bytes], second: Sequence[bytes]) -> int:
    for index, (left, right) in enumerate(zip(first, second)):
        if left != right:
            return index
    return min(len(first), len(second))


class MissAttribution:
    """Holds only its two evicted-key LRUs and each lane's previous prompt hashes; never reads a manager."""

    def __init__(self, tokens_per_hash: int, lane_block: int = LANE_BLOCK_DEFAULT,
                 emit: Callable[[str], None] = print) -> None:
        self.tokens_per_hash, self.lane_block, self.emit = tokens_per_hash, lane_block, emit
        self.evicted_recurrent, self.evicted_attention = EvictedKeys(), EvictedKeys()
        self.previous: OrderedDict[str, PreviousTurn] = OrderedDict()

    def note_recurrent_removed(self, keys: Sequence[Hashable]) -> None:
        self.evicted_recurrent.add(keys)

    def note_attention_removed(self, keys: Sequence[Hashable]) -> None:
        self.evicted_attention.add(keys)

    def _recurrent_class(self, lookup: MissLookup) -> str:
        if lookup.attention_hit <= lookup.recurrent_hit:
            return NONE
        evicted = any(key in self.evicted_recurrent for key in lookup.walked_recurrent_keys)
        return EVICTED if evicted else NEVER_STORED

    def _attention_class(self, lookup: MissLookup) -> str:
        if lookup.missing_attention_key is None:
            return NONE
        return EVICTED if lookup.missing_attention_key in self.evicted_attention else NEVER_STORED

    def _lane(self, block_hashes: Sequence[bytes]) -> str:
        return block_hashes[self.lane_block].hex()[:LANE_ID_HEX] if len(block_hashes) > self.lane_block else NO_LANE

    def _advance_lane(self, lane: str, lookup: MissLookup) -> tuple[int, int]:
        """(previous prompt end, tokens shared with it), and this prompt becomes the lane's previous turn."""
        if lane == NO_LANE:
            return NO_PREVIOUS, 0
        hashes = tuple(lookup.block_hashes[:lookup.prompt_tokens // self.tokens_per_hash])
        previous = self.previous.pop(lane, None)
        self.previous[lane] = PreviousTurn(lookup.prompt_tokens, hashes)
        while len(self.previous) > LANES_KEPT:
            self.previous.popitem(last=False)
        if previous is None:
            return NO_PREVIOUS, 0
        return previous.prompt_tokens, common_prefix_blocks(previous.block_hashes, hashes) * self.tokens_per_hash

    def record(self, lookup: MissLookup) -> None:
        lane = self._lane(lookup.block_hashes)
        previous_end, lcp = self._advance_lane(lane, lookup)
        self.emit(f"GLM53_MISS req={lookup.request_id} lane={lane} prompt={lookup.prompt_tokens} "
                  f"local={lookup.local_tokens} attention_hit={lookup.attention_hit} "
                  f"recurrent_hit={lookup.recurrent_hit} backtrack={max(0, lookup.attention_hit - lookup.recurrent_hit)} "
                  f"class={self._recurrent_class(lookup)} attn_class={self._attention_class(lookup)} "
                  f"prev_prompt_end={previous_end} lcp={lcp} store_wait_ms={lookup.store_wait_ms:.1f} "
                  f"reason={lookup.reason}")


def _lane_block(environ: Mapping[str, str]) -> int:
    raw = environ.get(LANE_BLOCK_ENV, "").strip()
    if not raw:
        return LANE_BLOCK_DEFAULT
    if not (raw.isascii() and raw.isdigit()):
        raise ValueError(f"{LANE_BLOCK_ENV}={raw!r}: use a block index past any prefix the lanes share (a 70K shared "
                         f"prefix at 128 tokens per hash is 547 blocks), or unset it for {LANE_BLOCK_DEFAULT}")
    return int(raw)


def from_env(tokens_per_hash: int, emit: Callable[[str], None],
             environ: Mapping[str, str] | None = None) -> MissAttribution | None:
    """None unless VLLM_OFFLOAD_MISS_ATTRIB=1: flag off, nothing is built and the scheduler reads nothing extra."""
    environ = os.environ if environ is None else environ
    raw = environ.get(FLAG_ENV, "").strip() or "0"
    if raw not in ("0", "1"):
        raise ValueError(f"{FLAG_ENV}={raw!r}: use 1 (log GLM53_MISS per request) or 0/unset (off)")
    if raw == "0":
        return None
    return MissAttribution(tokens_per_hash, _lane_block(environ), emit)
