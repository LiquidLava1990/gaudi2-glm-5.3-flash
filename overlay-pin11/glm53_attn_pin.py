# SPDX-License-Identifier: Apache-2.0
"""HPU_GLM53_ATTN_PIN: attention-only session pin for the graphs-only c1d image (REREAD_C1D_PLAN.md, pin-kit/prebuild.md).

On c1d prefix caching is off, so a finished turn frees its HBM blocks and the next turn of the same session reloads its
whole prefix from host RAM (run4: ~378K tokens, ~1.64 GB per turn, 1,500 restore-fence waits at p50 911 ms). This module
keeps the finished request's full-attention KV blocks in HBM, as the bytes its own graphs wrote (no copy, no latch):
  * finish:  KVCacheManager.free(R1) or KVCacheManager.pop_blocks_for_free(R1) first moves R1's leading full blocks (one
             per block hash, every FullAttention group, never KpoolTail or Mamba) into a PinEntry; the stock call releases
             (or defers) the rest. Serving runs async scheduling with a kv_both connector, so defer_block_free is on
             (scheduler.py:521-527) and a stop-token finish while the next step is in flight frees through
             pop_blocks_for_free (scheduler.py:3743-3752), never free: r4f-A graded (239 stop finishes) pinned nothing.
             The pinned blocks sit below num_computed - num_in_flight, so the in-flight step never writes them;
  * hit:     the stock OffloadingConnectorScheduler.get_num_new_matched_tokens(R2, 0) decides L exactly as today (CPU
             store walk + recurrent sidecar gate). The pin then takes the entry with the longest block-hash prefix match
             among the pins the request continues (it shares at least MIN_PIN_USE of the pin's blocks or all but
             TAIL_SLACK_BLOCKS of them; a request that shares only a common head leaves every pin alone,
             declined_partial) and plans
             k = min(match, L // B - 1) blocks: the last hit block always stays on the connector so the load
             job keeps its attention transfer (offloading/worker.py:405);
  * alloc:   after the stock add_local_computed_blocks (which asserts the request owns no blocks yet) the k pinned blocks
             are appended to req_to_blocks[R2]; allocate_external_computed_blocks then allocates only cdiv(L, B) - k
             fresh blocks; get_num_blocks_to_allocate is discounted by k so admission does not over-count;
  * load:    the stock update_state_after_alloc sees the k leading blocks carrying a block hash, treats them as locally
             computed (offloading/scheduler.py:1913-1960), loads chunks [k, n) plus the recurrent row at L through
             today's connector + LOADPATH_V2 path, and the marker hashes are reset right after the call. Marking is
             all-or-nothing and refuses blocks that already carry a hash: with caching off get_new_blocks never clears
             one (block_pool.py:671-677), so a leaked marker would make a later fresh block skip its load.
  * settle:  allocate_slots clamps a pending plan to the external tokens it really loads; when the scheduler drops the
             hit (hybrid fail-closed, scheduler.py:1999-2010, or a kept local tail) the pin goes back, no discount.
Graph keys are the keys of today's connector-hit turn (same L, same suffix). The adopted blocks are the bytes this
session's own graphs wrote; the load path reads the CPU store's copy, and the store dedups by block hash, so a chunk first
stored by another request with the same hash chain holds that request's bytes. Equality with the load path is therefore
a measured gate (det/oracle, INSTALL-NOTES.md), not a construction.
Pressure: allocate_slots -> None evicts the LRU pin and retries, and once no pin is left drops the oldest pending plan
(its request then loads the stock way); schedule() evicts pins (non-matching first, then pending plans) until every
scanned waiter the stock gate would admit with the pins free passes _glm53_leaves_partials_room;
Scheduler.reset_prefix_cache and the connector's reset_cache release every pin and pending plan before the stock reset
runs (the stock reset_prefix_cache resets the KV manager first, scheduler.py:3849, and the block pool refuses while any
block is in use, block_pool.py:773-780: with a pin held it returned False, or raised with reset_running_requests);
HPU_GLM53_ATTN_PIN_MAX caps the count; a pin that sees HPU_GLM53_ATTN_PIN_IDLE_TURNS later finished requests
without a hit goes back to the pool (r4f 2026-11-09: the det drive's 64 single-turn pins held 10,408 blocks, 31% of KV,
into the graded run with lookups=0), so traffic with no session reuse never holds KV for nothing. A failure inside the pin's own code (the actions run under _hook)
latches it off: pins released, markers reset, stock path continues, errors>0 marks the arm INVALID. The latch frees only
blocks the pin still holds: a PinEntry exists before its blocks leave the finished request, and injection takes each
group's slice before any change and drops the handed blocks from the plan's entry as the request gains them, so a
failure at any point neither frees a block a request owns nor leaves one with no owner. The latch does not cover the
stock methods the wrappers call: their exceptions propagate exactly as they would without the pin.
Flag unset: install_from_env() registers nothing and imports no vllm; every function object stays the image's.

Env (read at scheduler init; a bad value refuses the boot there, since site swallows .pth errors):
  HPU_GLM53_ATTN_PIN=1          enable (anything else: off)
  HPU_GLM53_ATTN_PIN_MAX=N      pinned sessions kept, LRU beyond (default 64; positive integer)
  HPU_GLM53_ATTN_PIN_LOG_S=S    seconds between counter lines (default 30; 0 logs every event)
  HPU_GLM53_ATTN_PIN_IDLE_TURNS=N  later finished requests a pin may wait for its hit (default 32 = 2 x 16 seats)
Boot refusals: prefix caching on; no OffloadingConnector; no FullAttention group at the hash block size; blocks_per_chunk != 1;
VLLM_CONTIGUOUS_PA on (1/true/yes/on): the block defragmenter copies blocks one way and can overwrite pinned ones.
Lines: `GLM53_ATTN_PIN armed ...` once, then (and once more at Scheduler.shutdown)
  `GLM53_ATTN_PIN pinned_sessions= pinned_blocks= held_plans= local_hit_tokens= evicted_adopt_trim= evicted_pressure=
   evicted_cap= evicted_idle= connector_fallbacks= adopted= lookups= pinned= declined_partial= errors=` (one line).
lookups counts connector hits only: a drive whose prompts share no stored prefix (the det cells, every arm's
external_prefix_cache_hits=0.0) runs 0 lookups by design; the live proof needs a multi-turn session.
"""
from __future__ import annotations

import copy
import functools
import importlib.abc
import importlib.machinery
import os
import sys
import time
from dataclasses import dataclass
from types import ModuleType
from typing import Any, Callable, NamedTuple, Sequence

FLAG = "HPU_GLM53_ATTN_PIN"
MAX_ENV = "HPU_GLM53_ATTN_PIN_MAX"
LOG_ENV = "HPU_GLM53_ATTN_PIN_LOG_S"
IDLE_ENV = "HPU_GLM53_ATTN_PIN_IDLE_TURNS"
LINE = "GLM53_ATTN_PIN"
CONTIGUOUS_PA_ENV = "VLLM_CONTIGUOUS_PA"
ENV_ON_VALUES = frozenset({"1", "true", "yes", "on"})  # vllm scheduler.py's env-bool rule, after strip().lower()
ORIGINALS_ATTR = "_glm53_attn_pin_originals"
DEFAULT_MAX_PINS = 64
DEFAULT_LOG_S = 30.0
DEFAULT_IDLE_TURNS = 32  # two finishes per seat at 16 seats: a live session's next turn arrives well inside it
WAITING_QUEUES = ("skipped_waiting", "waiting")  # both queues the mixed slot gate reads
GATE_STATUSES = frozenset({"WAITING", "PREEMPTED"})  # every status Scheduler._is_blocked_waiting_status lets through
# A pin serves exactly one follower: a request may consume it only when it shares at least MIN_PIN_USE_NUM/MIN_PIN_USE_DEN
# of the pin's blocks, i.e. it continues that conversation. A request that shares only a common head (one system prompt:
# 156 of a 700-block pin) would otherwise take the pin and trim the rest, and the owner's next turn would then take a short
# head from someone else: a cascade (live: 628 adoptions trimmed 220,903 blocks and served half of each prompt from pins).
MIN_PIN_USE_NUM, MIN_PIN_USE_DEN = 1, 2
# A short conversation can fail the half rule while genuinely continuing: the previous turn's last blocks (reasoning
# tokens the next prompt strips) never match, e.g. a pin of 8 blocks matched on 3. A request that leaves at most this
# many of the pin's blocks unmatched (and shares at least one) also continues it; a stranger on a long pin leaves hundreds.
TAIL_SLACK_BLOCKS = 6  # covers a stripped reasoning tail of up to ~768 tokens (6 blocks of 128)
# allocate_slots(request, num_new_tokens, num_new_computed_tokens, new_computed_blocks, num_lookahead_tokens,
# num_external_computed_tokens, ...): the position of num_external_computed_tokens after request (kv_cache_manager.py:347)
EXTERNAL_TOKENS_ARG = 4
EVICT_TRIM, EVICT_PRESSURE, EVICT_CAP, EVICT_IDLE = "adopt_trim", "pressure", "cap", "idle"
PINNABLE_CLASS = "FullAttentionManager"
REMEDY = f"or unset {FLAG} to run the stock load path"
SCHEDULER_MODULE = "vllm.v1.core.sched.scheduler"
# module -> ((class, method), ...) wrapped when the module is imported with the flag set
TARGET_MODULES: dict[str, tuple[tuple[str, str], ...]] = {
    SCHEDULER_MODULE: (("Scheduler", "__init__"), ("Scheduler", "schedule"), ("Scheduler", "shutdown"),
                       ("Scheduler", "reset_prefix_cache")),
    "vllm.v1.core.kv_cache_manager": (("KVCacheManager", "free"), ("KVCacheManager", "pop_blocks_for_free"),
                                      ("KVCacheManager", "allocate_slots")),
    "vllm.v1.core.kv_cache_coordinator": (("KVCacheCoordinator", "get_num_blocks_to_allocate"),),
    "vllm.v1.core.single_type_kv_cache_manager": (("SingleTypeKVCacheManager", "add_local_computed_blocks"),),
    "vllm.distributed.kv_transfer.kv_connector.v1.offloading.scheduler": (
        ("OffloadingConnectorScheduler", "get_num_new_matched_tokens"),
        ("OffloadingConnectorScheduler", "update_state_after_alloc"),
        ("OffloadingConnectorScheduler", "reset_cache"),
    ),
}


def enabled() -> bool:
    """Only "1" arms and "0" (or unset) stays off; any other value logs one refusal line and stays off."""
    value = os.environ.get(FLAG, "0")
    if value not in ("0", "1"):
        _print_line(f"{LINE} invalid value for {FLAG}={value}, refused")
        return False
    return value == "1"


def _positive_int(name: str, default: int, what: str) -> int:
    raw = os.environ.get(name, "").strip()
    if raw == "":
        return default
    if not raw.isdigit() or int(raw) < 1:
        raise ValueError(f"{name}={raw!r} refuses the boot: set {what} (a positive integer, default {default}), or unset it")
    return int(raw)


def max_pins() -> int:
    return _positive_int(MAX_ENV, DEFAULT_MAX_PINS, "the number of pinned sessions to keep (LRU beyond)")


def idle_turns() -> int:
    return _positive_int(IDLE_ENV, DEFAULT_IDLE_TURNS, "the later finished requests a pin may wait for its hit")


def log_interval_s() -> float:
    raw = os.environ.get(LOG_ENV, "").strip()
    if raw == "":
        return DEFAULT_LOG_S
    try:
        value = float(raw)
    except ValueError:
        value = -1.0
    if value < 0:
        raise ValueError(f"{LOG_ENV}={raw!r} refuses the boot: set the seconds between counter lines (0 or more, default "
                         f"{DEFAULT_LOG_S:g}), or unset it")
    return value


def cdiv(a: int, b: int) -> int:
    return -(-a // b)


def common_prefix(pinned: tuple[bytes, ...], hashes: Sequence[bytes]) -> int:
    """Length of the common block-hash prefix; a full match is one C-speed compare, else a bisection on prefixes."""
    n = min(len(pinned), len(hashes))
    if n == 0 or pinned[0] != hashes[0]:
        return 0
    if pinned[:n] == tuple(hashes[:n]):
        return n
    low, high = 1, n - 1
    while low < high:
        mid = (low + high + 1) // 2
        if pinned[:mid] == tuple(hashes[:mid]):
            low = mid
        else:
            high = mid - 1
    return low


def continues_pin(shared: int, pinned: int) -> bool:
    """True when a request sharing `shared` of the pin's `pinned` blocks continues that conversation: it shares at least
    MIN_PIN_USE_* of the pin, or leaves at most TAIL_SLACK_BLOCKS of it unmatched. Sharing nothing never continues."""
    if shared < 1:
        return False
    return shared * MIN_PIN_USE_DEN >= pinned * MIN_PIN_USE_NUM or pinned - shared <= TAIL_SLACK_BLOCKS


def pinnable_manager(manager: Any, block_size: int) -> bool:
    """Exactly a FullAttentionManager at the hash block size. Its subclasses recycle or special-case blocks (KpoolTail's one
    circular block per request, RSWA/Sink windows), and Mamba rows live on the host sidecar: none of them pins."""
    return type(manager).__name__ == PINNABLE_CLASS and getattr(manager, "block_size", None) == block_size


def _status_name(request: Any) -> str:
    status = getattr(request, "status", None)
    return str(getattr(status, "name", status))


@dataclass(slots=True)
class PinEntry:
    source: str
    hashes: tuple[bytes, ...]
    blocks: tuple[list[Any], ...]  # one list per pinned group: the blocks the pin holds (all hashes until injection)
    created: float
    turn: int  # the registry's finish ordinal when this entry was pinned

    @property
    def num_blocks(self) -> int:
        return len(self.hashes)


class PinMatch(NamedTuple):
    entry: PinEntry | None  # the pin the request continues: the longest common prefix among the pins passing continues_pin
    k_match: int
    declined_partial: bool  # no pin passes, but some pin shares a head with the request: it leaves them all untouched


@dataclass(slots=True)
class AdoptPlan:
    entry: PinEntry
    k_match: int
    k: int
    injected: bool = False


@dataclass(slots=True)
class PinCounters:
    lookups: int = 0
    adopted: int = 0
    local_hit_tokens: int = 0
    pinned: int = 0
    evicted_adopt_trim: int = 0  # pinned blocks beyond the adopted k, returned at injection
    evicted_pressure: int = 0  # allocation retry, make_room, pending-plan drops, the two resets and the latch release
    evicted_cap: int = 0  # HPU_GLM53_ATTN_PIN_MAX
    evicted_idle: int = 0  # HPU_GLM53_ATTN_PIN_IDLE_TURNS
    connector_fallbacks: int = 0
    # plan attempts (calls, not distinct requests) that shared only a head of the pins and left them alone
    declined_partial: int = 0
    errors: int = 0

    @property
    def evicted_blocks(self) -> int:
        return self.evicted_adopt_trim + self.evicted_pressure + self.evicted_cap + self.evicted_idle

    def add_evicted(self, reason: str, blocks: int) -> None:
        field_name = f"evicted_{reason}"
        setattr(self, field_name, getattr(self, field_name) + blocks)


def _print_line(line: str) -> None:
    sys.stdout.write(line + "\n")
    sys.stdout.flush()


class AttnPinRegistry:
    """Scheduler-process owner of the pinned blocks. Built once by the Scheduler.__init__ hook."""

    def __init__(self, *, kv_manager: Any, inner_connector: Any, group_indices: tuple[int, ...],
                 projected_positions: tuple[int, ...], block_size: int, max_entries: int, log_every_s: float,
                 emit: Callable[[str], None], clock: Callable[[], float] = time.monotonic,
                 idle_turns_limit: int = DEFAULT_IDLE_TURNS) -> None:
        self.kv = kv_manager
        self.inner = inner_connector
        self.group_indices = group_indices
        self.managers: tuple[Any, ...] = tuple(kv_manager.coordinator.single_type_managers[i] for i in group_indices)
        self.projected_positions = projected_positions
        self.block_size = block_size
        self.max_entries = max_entries
        self.idle_turns = idle_turns_limit
        self.turns = 0  # finished requests seen by pin_on_free: the clock a pin ages on
        self.log_every_s = log_every_s
        self.emit = emit
        self.clock = clock
        self.pins: list[PinEntry] = []  # LRU order: index 0 is the oldest
        self.plans: dict[str, AdoptPlan] = {}  # insertion order: the first is the oldest
        self.pending_marks: list[Any] = []  # blocks carrying a marker hash right now (reset by the wrapper or the latch)
        self.counters = PinCounters()
        self.disabled = False
        self._manager_pos = {id(manager): pos for pos, manager in enumerate(self.managers)}
        self._last_log = float("-inf")

    # ----- boot -----
    @classmethod
    def from_scheduler(cls, scheduler: Any, emit: Callable[[str], None]) -> "AttnPinRegistry":
        inner = _refuse_bad_config(scheduler)
        kv = scheduler.kv_cache_manager
        managers = kv.coordinator.single_type_managers
        block_size = _attention_block_size(scheduler, managers)
        group_indices = tuple(i for i, manager in enumerate(managers) if pinnable_manager(manager, block_size))
        offload_groups = tuple(getattr(scheduler.connector, "_offload_group_indices", range(len(managers))))
        positions = tuple(pos for pos, gi in enumerate(offload_groups) if gi in group_indices)
        if len(positions) != len(group_indices):
            raise RuntimeError(f"{FLAG}=1 needs every pinned attention group offloaded by the connector; pinned "
                               f"{group_indices}, offloaded {offload_groups}. Align HPU_GLM53_ATTENTION_ONLY_OFFLOAD {REMEDY}")
        return cls(kv_manager=kv, inner_connector=inner, group_indices=group_indices, projected_positions=positions,
                   block_size=block_size, max_entries=max_pins(), log_every_s=log_interval_s(), emit=emit,
                   idle_turns_limit=idle_turns())

    def emit_armed(self) -> None:
        self.emit(f"{LINE} armed {FLAG}=1 groups={self.group_indices} block_size={self.block_size} max_pins={self.max_entries} "
                  f"idle_turns={self.idle_turns} log_s={self.log_every_s:g} k_rule=hit_blocks-1 recurrent=host_sidecar "
                  f"pid={os.getpid()} contiguous_pa=off")

    # ----- finish: keep the attention blocks -----
    def pin_on_free(self, request: Any) -> int:
        """Move the request's leading full attention blocks into a pin before the stock free (or deferred pop); 0 when
        nothing is kept. A finished request that still holds attention blocks is one turn on the idle clock. The entry
        is built before the blocks leave req_to_blocks, so a failure building it leaves the stock free owning them all."""
        rid = request.request_id
        plan = self.plans.pop(rid, None)
        if plan is not None:
            self._return_plan(plan)
        if self.disabled or not request.is_finished() or rid not in self.managers[0].req_to_blocks:
            return 0
        self.turns += 1
        self._expire_idle()
        n = self._pinnable_blocks(request)
        if n < 1:
            return 0
        lists = [manager.req_to_blocks.get(rid) for manager in self.managers]
        if any(lst is None or len(lst) < n or not _owned(lst[:n]) for lst in lists):
            return 0
        blocks = tuple(list(lst[:n]) for lst in lists)  # type: ignore[union-attr]
        entry = PinEntry(rid, tuple(request.block_hashes[:n]), blocks, self.clock(), self.turns)
        for lst in lists:
            del lst[:n]  # type: ignore[index]
        self.pins.append(entry)
        self.counters.pinned += 1
        self._enforce_cap()
        self.maybe_log()
        return n

    def _pinnable_blocks(self, request: Any) -> int:
        processed = max(0, request.num_computed_tokens - getattr(request, "num_in_flight_tokens", 0))
        return min(len(request.block_hashes), processed // self.block_size)

    # ----- hit: plan the adoption -----
    def match(self, request: Any) -> PinMatch:
        """The pin the request continues (continues_pin). A pin it does not continue is skipped, so a short common head
        never shadows the right pin; declined_partial reports that such a pin existed when no pin passes."""
        best, best_k, partial_seen = None, 0, False
        for entry in reversed(self.pins):  # most recent first
            k = common_prefix(entry.hashes, request.block_hashes)
            if k == 0:
                continue
            if not continues_pin(k, entry.num_blocks):
                partial_seen = True
            elif k > best_k:
                best, best_k = entry, k
        return PinMatch(best, best_k, partial_seen and best is None)

    def plan_adopt(self, request: Any, hit_tokens: int) -> int:
        """Consume the best pin into a plan of k blocks (the last hit block stays on the connector); 0 = no adoption."""
        self.counters.lookups += 1
        self.maybe_log()  # lookups and fallbacks reach the log even when no pin or adoption follows
        if request.request_id not in self.plans and not self._open_plan(request, hit_tokens):
            self.counters.connector_fallbacks += 1
            return 0
        return self.settle_plan(request.request_id, hit_tokens)

    def _open_plan(self, request: Any, hit_tokens: int) -> bool:
        entry, k_match, declined_partial = self.match(request)
        if entry is None:
            if declined_partial:
                self.counters.declined_partial += 1
            return False
        k = min(k_match, hit_tokens // self.block_size - 1)
        if k < 1:
            return False
        self.pins.remove(entry)
        self.plans[request.request_id] = AdoptPlan(entry, k_match, k)
        return True

    def settle_plan(self, request_id: str, external_tokens: int) -> int:
        """Clamp a pending plan to the external tokens the allocation really loads. A hit too short to keep one block
        (shrunk between attempts, or zeroed by the scheduler: hybrid fail-closed scheduler.py:1999-2010, a kept local
        tail) gives the pin back, so no discount ever applies without an external load."""
        plan = self.plans.get(request_id)
        if plan is None or plan.injected:
            return 0
        plan.k = min(plan.k_match, external_tokens // self.block_size - 1)
        if plan.k >= 1:
            return plan.k
        self._return_plan(self.plans.pop(request_id))
        self.counters.connector_fallbacks += 1
        return 0

    @property
    def held_plans(self) -> int:
        """Plans holding a pin's blocks for a request that has not been allocated yet."""
        return sum(1 for plan in self.plans.values() if not plan.injected)

    def held_blocks(self) -> int:
        """HBM blocks the pin holds that the stock path would have free: every pin plus every pending plan."""
        entries = [*self.pins, *(plan.entry for plan in self.plans.values() if not plan.injected)]
        return sum(len(blocks) for entry in entries for blocks in entry.blocks)

    def discount(self, request_id: str) -> int:
        """Blocks a planned request already holds, so get_num_blocks_to_allocate does not over-count before injection."""
        plan = self.plans.get(request_id)
        if plan is None or plan.injected:
            return 0
        return plan.k * len(self.managers)

    # ----- alloc: hand the blocks over -----
    def inject_after_local(self, manager: Any, request_id: str) -> int:
        """Hand this group's first k planned blocks to the request and trim the rest back to the pool.

        The slice and the target list are taken before any change. The plan is booked injected first (from then on it
        never goes back to the pins), and the handed blocks leave the entry in the statement right after the request
        gains them, so the entry always lists exactly the blocks the pin still holds: the latch frees those, never one
        the request owns, and a failure on a later group still frees that group's blocks."""
        plan = self.plans.get(request_id)
        pos = self._manager_pos.get(id(manager))
        if plan is None or pos is None:
            return 0
        held = plan.entry.blocks[pos]
        handed = held[:plan.k]
        owner = manager.req_to_blocks[request_id]
        plan.injected = True
        owner.extend(handed)
        del held[:len(handed)]  # the request owns these now
        trimmed = _release_held(manager, held)
        if trimmed:
            self.counters.add_evicted(EVICT_TRIM, trimmed)
        return plan.k

    def mark_hashes(self, request_id: str, projected_blocks: Any, num_cached_tokens: int) -> list[Any]:
        """Set a marker hash on the adopted leading blocks so the stock update_state_after_alloc treats them as local.

        All or nothing: every block the connector scans must be unhashed first, and on any failure every marker set
        here is reset before the error propagates (to _hook, which latches the pin off and resets pending_marks too)."""
        plan = self.plans.get(request_id)
        if plan is None or not plan.injected:
            return []
        scanned = cdiv(num_cached_tokens, self.block_size)
        plan.k = max(0, min(plan.k, scanned - 1))
        groups = [projected_blocks.blocks[pos] for pos in self.projected_positions]
        _require_unhashed(groups, scanned)
        try:
            for group in groups:
                self._mark_group(group[:plan.k], plan.entry.hashes)
        except BaseException:
            self.clear_marks()
            raise
        return list(self.pending_marks)

    def _mark_group(self, blocks: Sequence[Any], hashes: Sequence[bytes]) -> None:
        if len(hashes) < len(blocks):
            raise RuntimeError(f"{len(blocks)} adopted blocks but only {len(hashes)} pinned hashes; report the hook")
        for block, block_hash in zip(blocks, hashes):
            self.pending_marks.append(block)  # recorded first, so a failure on this very block still resets it
            block._block_hash = block_hash
            block._block_hash_num_tokens = self.block_size

    def clear_marks(self) -> None:
        _reset_hashes(self.pending_marks)
        self.pending_marks.clear()

    def finish_adopt(self, request_id: str) -> None:
        """Book the adoption once the wrapper has reset the markers (it resets them directly, never through _hook). A
        complete injection left the entry empty; anything it still holds goes back to the pool, never to nobody."""
        self.pending_marks.clear()
        plan = self.plans.pop(request_id, None)
        if plan is None:
            return
        self._return_plan(plan)
        self.counters.adopted += 1
        self.counters.local_hit_tokens += plan.k * self.block_size
        self.maybe_log()

    # ----- pressure -----
    def evict_lru(self) -> int:
        """Free the oldest pin; with no pin left, drop the oldest plan not yet injected. Its request then reads no
        discount and no injection, so the stock load path runs for it (plans held across a waiting-loop break or
        continue, scheduler.py:1985/2075/2261-2263/2280, would otherwise keep their blocks out of every allocation)."""
        if self.pins:
            return self._free_entry(self.pins.pop(0), EVICT_PRESSURE)
        request_id = next((rid for rid, plan in self.plans.items() if not plan.injected), None)
        if request_id is None:
            return 0
        return self._free_entry(self.plans.pop(request_id).entry, EVICT_PRESSURE)

    def make_room_for_waiting(self, scheduler: Any) -> int:
        """Before schedule(): every scanned waiter the stock gate would admit must still pass it with the pins held.

        _glm53_leaves_partials_room (scheduler.py:1115) runs for every waiter of the mixed slot pick (scheduler.py:1104):
        it admits one only when free - reserved >= its remaining blocks, before the connector lookup, so it cannot see an
        adoption. While in-flight prefills hold a reservation, pins (non-matching first, then matching, then pending
        plans) are evicted until free - reserved >= the largest remaining(r) over the scanned waiters that stock
        (free + held) would pass. With no reservation the gate is off and allocate_slots' retry frees on demand.
        """
        if not self.pins and not self.held_plans:
            return 0
        reserved = scheduler._inflight_prefill_reserved_blocks()
        if reserved <= 0:
            return 0
        waiters = _scanned_waiting(scheduler)
        freed = 0
        while True:
            need = self._gate_need(scheduler, waiters, reserved)
            if need is None or self.kv.block_pool.get_num_free_blocks() >= need:
                return freed
            evicted = self._evict_for(waiters)
            if evicted == 0:
                return freed
            freed += evicted

    def _gate_need(self, scheduler: Any, waiters: Sequence[Any], reserved: int) -> int | None:
        """Free blocks the gate needs so every waiter stock would admit is admitted; None when stock admits none."""
        stock_room = self.kv.block_pool.get_num_free_blocks() + self.held_blocks() - reserved
        remaining = [scheduler._request_remaining_blocks(waiter) for waiter in waiters]
        admitted = [blocks for blocks in remaining if stock_room >= blocks]
        return reserved + max(admitted) if admitted else None

    def _evict_for(self, waiters: Sequence[Any]) -> int:
        """Evict one pin no fresh waiter matches, else the oldest pin, else the oldest pending plan."""
        matched = {id(self.match(waiter).entry) for waiter in waiters if getattr(waiter, "num_computed_tokens", 0) == 0}
        victim = next((entry for entry in self.pins if id(entry) not in matched), None)
        if victim is None:
            return self.evict_lru()
        self.pins.remove(victim)
        return self._free_entry(victim, EVICT_PRESSURE)

    def release_all(self) -> int:
        for plan in list(self.plans.values()):
            self._return_plan(plan)
        self.plans.clear()
        freed = 0
        while self.pins:
            freed += self._free_entry(self.pins.pop(0), EVICT_PRESSURE)
        return freed

    def fail(self, hook: str, error: BaseException) -> None:
        """Latch the pin off: markers reset, pins released, stock path continues, the arm is INVALID (errors>0)."""
        self.counters.errors += 1
        self.disabled = True
        self.clear_marks()
        released = self.release_all()
        self.emit(f"{LINE} disabled hook={hook} error={type(error).__name__}: {error} released_blocks={released}; the stock "
                  f"load path continues, this arm is INVALID (errors>0); restart with {FLAG} unset or report the hook")
        self.maybe_log(force=True)

    # ----- counters -----
    def maybe_log(self, force: bool = False) -> None:
        now = self.clock()
        if not force and now - self._last_log < self.log_every_s:
            return
        self._last_log = now
        c = self.counters
        self.emit(f"{LINE} pinned_sessions={len(self.pins)} pinned_blocks={sum(e.num_blocks for e in self.pins)} "
                  f"held_plans={self.held_plans} local_hit_tokens={c.local_hit_tokens} "
                  f"evicted_adopt_trim={c.evicted_adopt_trim} evicted_pressure={c.evicted_pressure} "
                  f"evicted_cap={c.evicted_cap} evicted_idle={c.evicted_idle} connector_fallbacks={c.connector_fallbacks} "
                  f"adopted={c.adopted} "
                  f"lookups={c.lookups} pinned={c.pinned} declined_partial={c.declined_partial} errors={c.errors}")

    # ----- internals -----
    def _free_entry(self, entry: PinEntry, reason: str) -> int:
        total = sum(_release_held(manager, blocks) for manager, blocks in zip(self.managers, entry.blocks))
        self.counters.add_evicted(reason, total)
        return total

    def _return_plan(self, plan: AdoptPlan) -> None:
        """A plan nothing was handed from goes back to the pins whole. Once injection started, its entry lists only the
        blocks not handed to the request (none after a complete injection), and those go back to the pool."""
        if plan.injected:
            self._free_entry(plan.entry, EVICT_PRESSURE)
            return
        self.pins.append(plan.entry)

    def _enforce_cap(self) -> None:
        while len(self.pins) > self.max_entries:
            self._free_entry(self.pins.pop(0), EVICT_CAP)

    def _expire_idle(self) -> None:
        """Free every pin that has seen idle_turns later finishes without a hit. A returned plan keeps its pin turn, and
        a pending plan is not aged (its request is mid-admission; pressure and evict_lru reclaim it)."""
        expired = [entry for entry in self.pins if self.turns - entry.turn >= self.idle_turns]
        for entry in expired:
            self.pins.remove(entry)
            self._free_entry(entry, EVICT_IDLE)


def _release_held(manager: Any, held: list[Any]) -> int:
    """Return the blocks the pin holds in one group to the pool (LIFO, as the stock free) and drop them from the list
    only once the pool has them, so a failing free leaves them listed for the latch instead of owned by nobody."""
    if not held:
        return 0
    count = len(held)
    manager.block_pool.free_blocks(reversed(held))
    held.clear()
    return count


def _owned(blocks: Sequence[Any]) -> bool:
    return all(not block.is_null and block.ref_cnt == 1 for block in blocks)


def _reset_hashes(blocks: Sequence[Any]) -> None:
    for block in blocks:
        block.reset_hash()


def _require_unhashed(groups: Sequence[Sequence[Any]], scanned: int) -> None:
    """Every block the connector scans must be unhashed before marking, or its load would be skipped without us."""
    hashed = [block.block_id for group in groups for block in group[:scanned]
              if not block.is_null and block.block_hash is not None]
    if hashed:
        raise RuntimeError(f"blocks {hashed[:8]} carry a block hash before marking (prefix caching must be off and "
                           f"markers reset); the connector would skip their load. Report the hook {REMEDY}")


def _scanned_waiting(scheduler: Any) -> list[Any]:
    """The waiters the mixed slot gate looks at (scheduler.py:1023 _glm53_waiting_prompts).

    The scheduler's own list is used when it has one: both queues, minus the blocked statuses, a paused scheduler, stale
    output and a full recurrent slot table. Otherwise every WAITING or PREEMPTED request of skipped_waiting and waiting.
    A prompt deferred once, for an async load or for room, is parked in skipped_waiting: a pin that only watched `waiting`
    kept its blocks while those prompts never ran again (r1, 2026-11-22 18:03)."""
    gate_list = getattr(scheduler, "_glm53_waiting_prompts", None)
    if callable(gate_list):
        return list(gate_list())
    return [request for name in WAITING_QUEUES for request in _queue_requests(scheduler, name)
            if _status_name(request) in GATE_STATUSES]


def _queue_requests(scheduler: Any, name: str) -> list[Any]:
    try:
        return list(getattr(scheduler, name, ()))
    except TypeError:
        return []


def _attention_block_size(scheduler: Any, managers: Sequence[Any]) -> int:
    """The pin's block size is the FullAttention groups' own, not Scheduler.block_size: on a hybrid model (GLM-5.3F)
    the scheduler block size is the KDA group's (1048576 on r4e) while MLA and the request hashes use 128. Each
    attention block must carry exactly one request hash, so every attention group must sit at the hash block size."""
    names = [type(manager).__name__ for manager in managers]
    sizes = sorted({int(manager.block_size) for manager in managers if type(manager).__name__ == PINNABLE_CLASS})
    if not sizes:
        raise RuntimeError(f"{FLAG}=1 found no FullAttention KV group to pin (groups: {names}); check the model's KV "
                           f"groups {REMEDY}")
    hash_block_size = int(getattr(scheduler, "hash_block_size", None) or sizes[0])
    if sizes != [hash_block_size]:
        raise RuntimeError(f"{FLAG}=1 needs one attention block size equal to the request hash block size; got "
                           f"attention block size {' / '.join(map(str, sizes))} vs hash block size {hash_block_size} "
                           f"(groups: {names}). Set request_hash_block_size to the attention block size {REMEDY}")
    return hash_block_size


def _refuse_contiguous_pa() -> None:
    """Contiguous paged attention runs the block defragmenter, whose one-way copies can overwrite pinned blocks."""
    value = os.environ.get(CONTIGUOUS_PA_ENV, "0").strip()
    if value.lower() in ENV_ON_VALUES:
        raise RuntimeError(f"{FLAG}=1 conflicts with {CONTIGUOUS_PA_ENV}={value}: contiguous paged attention runs the block "
                           f"defragmenter, and defragmentation can overwrite pinned blocks and silently corrupt outputs. "
                           f"Unset {CONTIGUOUS_PA_ENV} {REMEDY}")


def _refuse_bad_config(scheduler: Any) -> Any:
    """The boot refusals that need only the scheduler and the environment; returns the scheduler-side connector."""
    _refuse_contiguous_pa()
    if getattr(getattr(scheduler, "cache_config", None), "enable_prefix_caching", False):
        raise RuntimeError(f"{FLAG}=1 conflicts with prefix caching ON: the pin owns finished requests' attention blocks "
                           f"itself. Run --no-enable-prefix-caching (the c1d contract) {REMEDY}")
    inner = getattr(getattr(scheduler, "connector", None), "connector_scheduler", None)
    if inner is None:
        raise RuntimeError(f"{FLAG}=1 needs the OffloadingConnector (it decides the hit length L and loads the recurrent "
                           f"row); no connector_scheduler is configured. Add the kv-transfer OffloadingConnector {REMEDY}")
    blocks_per_chunk = getattr(getattr(inner, "config", None), "blocks_per_chunk", 1)
    if blocks_per_chunk != 1:
        raise RuntimeError(f"{FLAG}=1 needs one KV block per offload chunk; the connector has blocks_per_chunk="
                           f"{blocks_per_chunk}. Set the offload block size to the KV block size {REMEDY}")
    return inner


# ----- the wrappers (scheduler process; a worker process gets the same hook but never builds these objects) -----
_REGISTRY: AttnPinRegistry | None = None
_EMIT: Callable[[str], None] = _print_line


def registry() -> AttnPinRegistry | None:
    return _REGISTRY


def reset_for_tests() -> None:
    global _REGISTRY, _EMIT
    _REGISTRY = None
    _EMIT = _print_line


def _hook(name: str, default: Any, action: Callable[[AttnPinRegistry], Any]) -> Any:
    """Run one pin action; a failure inside the pin latches it off and the stock call proceeds. Only the action is
    covered: the stock method each wrapper calls runs outside _hook and raises exactly as it would without the pin."""
    reg = _REGISTRY
    if reg is None or reg.disabled:
        return default
    try:
        return action(reg)
    except Exception as error:  # noqa: BLE001 - the pin never takes the engine down
        reg.fail(name, error)
        return default


def _wrap_scheduler_init(original: Callable[..., None]) -> Callable[..., None]:
    @functools.wraps(original)
    def __init__(self: Any, *args: Any, **kwargs: Any) -> None:
        original(self, *args, **kwargs)
        global _REGISTRY
        _REGISTRY = AttnPinRegistry.from_scheduler(self, _EMIT)  # a refusal raises here: the engine boot fails loudly
        _REGISTRY.emit_armed()

    return __init__


def _wrap_schedule(original: Callable[..., Any]) -> Callable[..., Any]:
    @functools.wraps(original)
    def schedule(self: Any, *args: Any, **kwargs: Any) -> Any:
        _hook("make_room", 0, lambda reg: reg.make_room_for_waiting(self))
        return original(self, *args, **kwargs)

    return schedule


def _wrap_shutdown(original: Callable[..., Any]) -> Callable[..., Any]:
    @functools.wraps(original)
    def shutdown(self: Any, *args: Any, **kwargs: Any) -> Any:
        _hook("shutdown_flush", None, lambda reg: reg.maybe_log(force=True))  # the final counter line
        return original(self, *args, **kwargs)

    return shutdown


def _wrap_free(original: Callable[..., Any]) -> Callable[..., Any]:
    """KVCacheManager.free and KVCacheManager.pop_blocks_for_free (the deferred free): pin first, then the stock call
    frees, or returns for a fenced free, only the blocks the pin left."""
    @functools.wraps(original)
    def free(self: Any, request: Any) -> Any:
        _hook("pin_on_free", 0, lambda reg: reg.pin_on_free(request) if reg.kv is self else 0)
        return original(self, request)

    return free


def _allocate_with_pins(original: Callable[..., Any], manager: Any, request: Any, args: tuple[Any, ...],
                        kwargs: dict[str, Any]) -> Any:
    """Settle the request's plan to its real external tokens, allocate, and on None evict one pin (or plan) and retry."""
    external = _external_tokens(args, kwargs)
    _hook("settle_plan", 0, lambda reg: reg.settle_plan(request.request_id, external) if reg.kv is manager else 0)
    result = original(manager, request, *args, **kwargs)
    while result is None and _hook("evict_lru", 0, lambda reg: reg.evict_lru() if reg.kv is manager else 0) > 0:
        result = original(manager, request, *args, **kwargs)
    return result


def _wrap_allocate_slots(original: Callable[..., Any]) -> Callable[..., Any]:
    @functools.wraps(original)
    def allocate_slots(self: Any, request: Any, *args: Any, **kwargs: Any) -> Any:
        return _allocate_with_pins(original, self, request, args, kwargs)

    return allocate_slots


def _external_tokens(args: tuple[Any, ...], kwargs: dict[str, Any]) -> int:
    if "num_external_computed_tokens" in kwargs:
        return int(kwargs["num_external_computed_tokens"])
    return int(args[EXTERNAL_TOKENS_ARG]) if len(args) > EXTERNAL_TOKENS_ARG else 0


def _wrap_get_num_blocks_to_allocate(original: Callable[..., int]) -> Callable[..., int]:
    @functools.wraps(original)
    def get_num_blocks_to_allocate(self: Any, request_id: str, *args: Any, **kwargs: Any) -> int:
        total = original(self, request_id, *args, **kwargs)
        discount = _hook("discount", 0, lambda reg: reg.discount(request_id) if reg.kv.coordinator is self else 0)
        return max(total - discount, 0)

    return get_num_blocks_to_allocate


def _wrap_add_local_computed_blocks(original: Callable[..., None]) -> Callable[..., None]:
    @functools.wraps(original)
    def add_local_computed_blocks(self: Any, request_id: str, *args: Any, **kwargs: Any) -> None:
        original(self, request_id, *args, **kwargs)
        _hook("inject", 0, lambda reg: reg.inject_after_local(self, request_id))

    return add_local_computed_blocks


def _wrap_get_num_new_matched_tokens(original: Callable[..., Any]) -> Callable[..., Any]:
    @functools.wraps(original)
    def get_num_new_matched_tokens(self: Any, request: Any, num_computed_tokens: int) -> Any:
        hit, load_async = original(self, request, num_computed_tokens)
        if hit and load_async:
            _hook("plan_adopt", 0, lambda reg: reg.plan_adopt(request, hit) if reg.inner is self else 0)
        return hit, load_async

    return get_num_new_matched_tokens


def _num_cached_tokens(inner: Any, request_id: str, num_external_tokens: int) -> int:
    status = getattr(inner, "_req_status", {}).get(request_id)
    return int(getattr(status, "num_locally_computed_tokens", 0)) + num_external_tokens


def _marked_for(inner: Any, request: Any, blocks: Any, num_external_tokens: int) -> list[Any]:
    if not num_external_tokens:
        return []
    cached = _num_cached_tokens(inner, request.request_id, num_external_tokens)
    return _hook("mark", [], lambda reg: reg.mark_hashes(request.request_id, blocks, cached) if reg.inner is inner else [])


def _update_with_markers(original: Callable[..., Any], inner: Any, request: Any, blocks: Any,
                         num_external_tokens: int) -> Any:
    marked = _marked_for(inner, request, blocks, num_external_tokens)
    try:
        return original(inner, request, blocks, num_external_tokens)
    finally:
        _reset_hashes(marked)  # directly: _hook skips once the pin has latched off, the markers must still go
        if marked:
            _hook("finish_adopt", None, lambda reg: reg.finish_adopt(request.request_id))


def _wrap_update_state_after_alloc(original: Callable[..., Any]) -> Callable[..., Any]:
    @functools.wraps(original)
    def update_state_after_alloc(self: Any, request: Any, blocks: Any, num_external_tokens: int) -> Any:
        return _update_with_markers(original, self, request, blocks, num_external_tokens)

    return update_state_after_alloc


def _release_before(hook: str, owns: Callable[[AttnPinRegistry, Any], bool],
                    ) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """A reset wrapper: every pin and pending plan goes back to the pool before the stock reset runs."""
    def wrap(original: Callable[..., Any]) -> Callable[..., Any]:
        @functools.wraps(original)
        def reset(self: Any, *args: Any, **kwargs: Any) -> Any:
            _hook(hook, 0, lambda reg: reg.release_all() if owns(reg, self) else 0)
            return original(self, *args, **kwargs)

        return reset

    return wrap


_wrap_reset_prefix_cache = _release_before("reset_prefix_cache",
                                           lambda reg, scheduler: reg.kv is scheduler.kv_cache_manager)
_wrap_reset_cache = _release_before("release_all", lambda reg, inner: reg.inner is inner)


WRAPPERS: dict[tuple[str, str], Callable[[Callable[..., Any]], Callable[..., Any]]] = {
    ("Scheduler", "__init__"): _wrap_scheduler_init,
    ("Scheduler", "schedule"): _wrap_schedule,
    ("Scheduler", "shutdown"): _wrap_shutdown,
    ("Scheduler", "reset_prefix_cache"): _wrap_reset_prefix_cache,
    ("KVCacheManager", "free"): _wrap_free,
    ("KVCacheManager", "pop_blocks_for_free"): _wrap_free,
    ("KVCacheManager", "allocate_slots"): _wrap_allocate_slots,
    ("KVCacheCoordinator", "get_num_blocks_to_allocate"): _wrap_get_num_blocks_to_allocate,
    ("SingleTypeKVCacheManager", "add_local_computed_blocks"): _wrap_add_local_computed_blocks,
    ("OffloadingConnectorScheduler", "get_num_new_matched_tokens"): _wrap_get_num_new_matched_tokens,
    ("OffloadingConnectorScheduler", "update_state_after_alloc"): _wrap_update_state_after_alloc,
    ("OffloadingConnectorScheduler", "reset_cache"): _wrap_reset_cache,
}


def PATCH_POINTS(modules: dict[str, ModuleType]) -> list[tuple[ModuleType, str, str]]:  # noqa: N802 - test helper
    return [(modules[name], cls, method) for name, points in TARGET_MODULES.items() if name in modules
            for cls, method in points]


def install_module(module: ModuleType) -> int:
    """Wrap the module's target methods once (idempotent); returns the number wrapped now."""
    count = 0
    for cls_name, cls, method, originals in _module_points(module):
        if method not in originals:
            originals[method] = getattr(cls, method)
            setattr(cls, method, WRAPPERS[(cls_name, method)](originals[method]))
            count += 1
    return count


def uninstall_module(module: ModuleType) -> int:
    count = 0
    for _, cls, method, originals in _module_points(module, create=False):
        if method in originals:
            setattr(cls, method, originals.pop(method))
            count += 1
    return count


def _module_points(module: ModuleType, create: bool = True) -> list[tuple[str, type, str, dict[str, Any]]]:
    """(class name, class, method, the class's own originals dict) per patch point; create=False skips unwrapped classes."""
    points = []
    for cls_name, method in TARGET_MODULES.get(module.__name__, ()):
        cls = getattr(module, cls_name)
        originals = vars(cls).get(ORIGINALS_ATTR)
        if originals is None and not create:
            continue
        if originals is None:
            originals = {}
            setattr(cls, ORIGINALS_ATTR, originals)
        points.append((cls_name, cls, method, originals))
    return points


def install_all(modules: dict[str, ModuleType], emit: Callable[[str], None] | None = None) -> int:
    global _EMIT
    if emit is not None:
        _EMIT = emit
    return sum(install_module(module) for module in modules.values())


def uninstall_all(modules: dict[str, ModuleType]) -> int:
    return sum(uninstall_module(module) for module in modules.values())


class _PatchingLoader(importlib.abc.Loader):
    def __init__(self, inner: importlib.abc.Loader) -> None:
        self._inner = inner

    def create_module(self, spec: importlib.machinery.ModuleSpec) -> ModuleType | None:
        return self._inner.create_module(spec)

    def exec_module(self, module: ModuleType) -> None:
        self._inner.exec_module(module)
        install_module(module)


class _PatchOnImport(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname: str, path: Any, target: Any = None) -> importlib.machinery.ModuleSpec | None:
        if fullname not in TARGET_MODULES:
            return None
        for finder in list(sys.meta_path):
            find = getattr(finder, "find_spec", None)
            if finder is self or find is None:
                continue
            spec = find(fullname, path, target)
            if spec is None or spec.loader is None:
                continue
            patched = copy.copy(spec)
            patched.loader = _PatchingLoader(spec.loader)
            return patched
        return None


_FINDER: list[_PatchOnImport] = []


def install_from_env() -> bool:
    """Register the import hook when the flag is set; never raises (site swallows .pth errors), config is checked at boot."""
    if not enabled():
        return False
    if _FINDER:
        return True
    finder = _PatchOnImport()
    sys.meta_path.insert(0, finder)
    _FINDER.append(finder)
    for name in TARGET_MODULES:
        already = sys.modules.get(name)
        if already is not None:
            install_module(already)
    return True
