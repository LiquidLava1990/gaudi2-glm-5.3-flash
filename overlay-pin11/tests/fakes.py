"""CPU fakes of the r4d scheduler-side KV stack, shaped like the verified bytes (see ../prebuild.md section 1).

Class names matter: the pin picks pinnable groups by MRO class name (FullAttentionManager, not KpoolTailManager), so the
fakes carry the real names. Every method here mirrors the real contract the pin relies on:
  * SingleTypeKVCacheManager.add_local_computed_blocks asserts the request has no blocks yet (single_type:257);
  * allocate_external_computed_blocks allocates cdiv(total, B) - len(req_blocks) fresh blocks (single_type:323-327);
  * get_num_blocks_to_allocate discounts len(req_to_blocks[rid]) (single_type:193-214);
  * update_state_after_alloc skips leading blocks whose block_hash is not None (offloading/scheduler.py:1913-1960);
  * Scheduler.reset_prefix_cache resets the KV manager before the connector (scheduler.py:3821-3861), and the block pool
    refuses while any block but the null block is in use (block_pool.py:764-780).
"""
from __future__ import annotations

from collections import Counter, defaultdict, deque
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any, Callable, Iterable, Sequence


def cdiv(a: int, b: int) -> int:
    return -(-a // b)


@dataclass(eq=False)
class KVCacheBlock:
    block_id: int
    ref_cnt: int = 0
    _block_hash: bytes | None = None
    _block_hash_num_tokens: int | None = None
    is_null: bool = False

    @property
    def block_hash(self) -> bytes | None:
        return self._block_hash

    def reset_hash(self) -> None:
        self._block_hash = None
        self._block_hash_num_tokens = None


class BlockPool:
    def __init__(self, num_gpu_blocks: int, enable_caching: bool = False) -> None:
        self.enable_caching = enable_caching
        self.blocks = [KVCacheBlock(i) for i in range(num_gpu_blocks)]
        self.null_block = self.blocks[0]
        self.null_block.is_null = True
        self.null_block.ref_cnt = 1
        self.free_block_queue: deque[KVCacheBlock] = deque(self.blocks[1:])

    def get_num_free_blocks(self) -> int:
        return len(self.free_block_queue)

    def get_new_blocks(self, num_blocks: int) -> list[KVCacheBlock]:
        if num_blocks > self.get_num_free_blocks():
            raise ValueError(f"Cannot get {num_blocks} free blocks from the pool")
        ret = [self.free_block_queue.popleft() for _ in range(num_blocks)]
        for block in ret:
            assert block.ref_cnt == 0
            block.ref_cnt += 1
        return ret

    def free_blocks(self, ordered_blocks: Iterable[KVCacheBlock]) -> None:
        first: list[KVCacheBlock] = []
        for block in ordered_blocks:
            block.ref_cnt -= 1
            assert block.ref_cnt >= 0, f"block {block.block_id} freed twice"
            if block.ref_cnt == 0 and not block.is_null:
                first.append(block)
        self.free_block_queue.extendleft(reversed(first))

    def reset_prefix_cache(self) -> bool:
        if len(self.blocks) - self.get_num_free_blocks() != 1:  # the null block is always in use
            return False
        for block in self.blocks:
            block.reset_hash()
        return True


class SingleTypeKVCacheManager:
    def __init__(self, block_pool: BlockPool, block_size: int, kv_cache_group_id: int) -> None:
        self.block_pool = block_pool
        self.block_size = block_size
        self.enable_caching = False
        self.kv_cache_group_id = kv_cache_group_id
        self.req_to_blocks: defaultdict[str, list[KVCacheBlock]] = defaultdict(list)
        self.num_cached_block: dict[str, int] = {}

    def get_num_blocks_to_allocate(self, request_id: str, *, num_tokens: int, new_computed_blocks: Any,
                                   total_computed_tokens: int, num_local_computed_tokens: int,
                                   num_tokens_main_model: int, apply_admission_cap: bool = False) -> int:
        num_required = cdiv(num_tokens, self.block_size)
        return max(num_required - len(self.req_to_blocks.get(request_id, ())), 0)

    def add_local_computed_blocks(self, request_id: str, new_computed_blocks: Any, num_local_computed_tokens: int,
                                  num_external_computed_tokens: int) -> None:
        req_blocks = self.req_to_blocks[request_id]
        assert len(req_blocks) == 0
        assert not any(new_computed_blocks), "Computed blocks should be empty when prefix caching is disabled"
        self.num_cached_block[request_id] = 0

    def allocate_external_computed_blocks(self, request_id: str, num_local_computed_tokens: int,
                                          num_external_computed_tokens: int) -> None:
        if num_external_computed_tokens <= 0:
            return
        req_blocks = self.req_to_blocks[request_id]
        total = num_local_computed_tokens + num_external_computed_tokens
        req_blocks.extend(self.block_pool.get_new_blocks(cdiv(total, self.block_size) - len(req_blocks)))

    def allocate_new_blocks(self, request_id: str, num_tokens: int, num_tokens_main_model: int) -> list[KVCacheBlock]:
        req_blocks = self.req_to_blocks[request_id]
        num_new = cdiv(num_tokens, self.block_size) - len(req_blocks)
        if num_new <= 0:
            return []
        new_blocks = self.block_pool.get_new_blocks(num_new)
        req_blocks.extend(new_blocks)
        return new_blocks

    def pop_blocks_for_free(self, request_id: str) -> list[KVCacheBlock]:
        self.num_cached_block.pop(request_id, None)
        return self.req_to_blocks.pop(request_id, [])

    def free(self, request_id: str) -> None:
        self.block_pool.free_blocks(reversed(self.pop_blocks_for_free(request_id)))


class FullAttentionManager(SingleTypeKVCacheManager):
    pass


class KpoolTailManager(FullAttentionManager):
    """One block per request, circular (single_type:1114-1239)."""

    def get_num_blocks_to_allocate(self, request_id: str, *, num_tokens: int, new_computed_blocks: Any,
                                   total_computed_tokens: int, num_local_computed_tokens: int,
                                   num_tokens_main_model: int, apply_admission_cap: bool = False) -> int:
        return max(1 - len(self.req_to_blocks.get(request_id, ())), 0)

    def add_local_computed_blocks(self, request_id: str, new_computed_blocks: Any, num_local_computed_tokens: int,
                                  num_external_computed_tokens: int) -> None:
        return

    def allocate_external_computed_blocks(self, request_id: str, num_local_computed_tokens: int,
                                          num_external_computed_tokens: int) -> None:
        self.allocate_new_blocks(request_id, 1, 1)

    def allocate_new_blocks(self, request_id: str, num_tokens: int, num_tokens_main_model: int) -> list[KVCacheBlock]:
        req_blocks = self.req_to_blocks[request_id]
        if req_blocks:
            return []
        new_blocks = self.block_pool.get_new_blocks(1)
        req_blocks.extend(new_blocks)
        return new_blocks


class MambaManager(SingleTypeKVCacheManager):
    """A KDA/recurrent group: block size = max_model_len on the release, so one block per request."""


MANAGER_KINDS: dict[str, type[SingleTypeKVCacheManager]] = {
    "full": FullAttentionManager, "tail": KpoolTailManager, "mamba": MambaManager}
# r4e-A boot (graded/r4e/container.log, 2026-11-09 03:03): Scheduler.block_size 1048576, hash block size 128;
# groups MLA (FullAttention) 128, KpoolTail 4, KDA (Mamba) at max_model_len 1048576.
R4E_SCHEDULER_BLOCK_SIZE = 1 << 20
R4E_GROUPS: tuple[tuple[str, int], ...] = (("full", 128), ("tail", 4), ("mamba", R4E_SCHEDULER_BLOCK_SIZE))


class KVCacheCoordinator:
    def __init__(self, managers: list[SingleTypeKVCacheManager], block_pool: BlockPool) -> None:
        self.single_type_managers = managers
        self.block_pool = block_pool

    def get_num_blocks_to_allocate(self, request_id: str, *, num_tokens: int, new_computed_blocks: Any,
                                   num_encoder_tokens: int, total_computed_tokens: int,
                                   num_local_computed_tokens: int, num_tokens_main_model: int,
                                   apply_admission_cap: bool = False) -> int:
        del num_encoder_tokens  # no cross-attention in the fake stack
        return sum(manager.get_num_blocks_to_allocate(
            request_id, num_tokens=num_tokens, new_computed_blocks=new_computed_blocks[i],
            total_computed_tokens=total_computed_tokens, num_local_computed_tokens=num_local_computed_tokens,
            num_tokens_main_model=num_tokens_main_model, apply_admission_cap=apply_admission_cap)
            for i, manager in enumerate(self.single_type_managers))

    def allocate_new_computed_blocks(self, request_id: str, new_computed_blocks: Any,
                                     num_local_computed_tokens: int, num_external_computed_tokens: int) -> None:
        for i, manager in enumerate(self.single_type_managers):
            manager.add_local_computed_blocks(request_id, new_computed_blocks[i], num_local_computed_tokens,
                                              num_external_computed_tokens)
        if num_external_computed_tokens > 0:
            for manager in self.single_type_managers:
                manager.allocate_external_computed_blocks(request_id, num_local_computed_tokens,
                                                          num_external_computed_tokens)

    def allocate_new_blocks(self, request_id: str, num_tokens: int, num_tokens_main_model: int,
                            num_encoder_tokens: int = 0) -> tuple[list[KVCacheBlock], ...]:
        return tuple(m.allocate_new_blocks(request_id, num_tokens, num_tokens_main_model)
                     for m in self.single_type_managers)

    def free(self, request_id: str) -> None:
        for manager in self.single_type_managers:
            manager.free(request_id)

    def get_blocks(self, request_id: str) -> tuple[list[KVCacheBlock], ...]:
        return tuple(m.req_to_blocks.get(request_id) or [] for m in self.single_type_managers)


@dataclass
class KVCacheBlocks:
    blocks: tuple[list[KVCacheBlock], ...]


class KVCacheManager:
    """The stock allocate_slots, reduced to the branches a connector hit on c1d takes (kv_cache_manager.py:443-568)."""

    def __init__(self, coordinator: KVCacheCoordinator, enable_caching: bool = False) -> None:
        self.coordinator = coordinator
        self.block_pool = coordinator.block_pool
        self.enable_caching = enable_caching
        self.empty_kv_cache_blocks = KVCacheBlocks(tuple(() for _ in coordinator.single_type_managers))

    def allocate_slots(self, request: Any, num_new_tokens: int, *, num_new_computed_tokens: int = 0,
                       new_computed_blocks: KVCacheBlocks | None = None, num_lookahead_tokens: int = 0,
                       num_external_computed_tokens: int = 0, delay_cache_blocks: bool = False, **_: Any,
                       ) -> KVCacheBlocks | None:
        if num_new_tokens == 0 and num_external_computed_tokens == 0:
            raise ValueError("num_new_tokens must be greater than 0 when there are no external computed tokens")
        computed_list = self.empty_kv_cache_blocks.blocks if new_computed_blocks is None else new_computed_blocks.blocks
        num_local = request.num_computed_tokens + num_new_computed_tokens
        total = num_local + num_external_computed_tokens
        need_slot = total + num_new_tokens + num_lookahead_tokens
        to_allocate = self.coordinator.get_num_blocks_to_allocate(
            request.request_id, num_tokens=need_slot, new_computed_blocks=computed_list, num_encoder_tokens=0,
            total_computed_tokens=total, num_local_computed_tokens=num_local, num_tokens_main_model=total + num_new_tokens)
        if to_allocate > self.block_pool.get_num_free_blocks():
            return None
        if num_external_computed_tokens > 0:
            self.coordinator.allocate_new_computed_blocks(request.request_id, computed_list, num_local,
                                                          num_external_computed_tokens)
        new_blocks = self.coordinator.allocate_new_blocks(request.request_id, need_slot, total + num_new_tokens)
        return KVCacheBlocks(new_blocks)

    def free(self, request: Any) -> None:
        self.coordinator.free(request.request_id)

    def pop_blocks_for_free(self, request: Any) -> list[KVCacheBlock]:
        """The deferred free of scheduler.py:3750: bookkeeping popped, blocks returned to the caller, pool untouched."""
        return [block for manager in self.coordinator.single_type_managers
                for block in manager.pop_blocks_for_free(request.request_id)]

    def get_blocks(self, request_id: str) -> KVCacheBlocks:
        return KVCacheBlocks(self.coordinator.get_blocks(request_id))

    def reset_prefix_cache(self) -> bool:
        return self.block_pool.reset_prefix_cache()


class RequestStatus:
    WAITING = "WAITING"
    WAITING_FOR_REMOTE_KVS = "WAITING_FOR_REMOTE_KVS"
    RUNNING = "RUNNING"
    PREEMPTED = "PREEMPTED"
    FINISHED_STOPPED = "FINISHED_STOPPED"
    FINISHED_ABORTED = "FINISHED_ABORTED"


@dataclass
class Request:
    request_id: str
    block_hashes: list[bytes]
    num_tokens: int
    num_computed_tokens: int = 0
    num_in_flight_tokens: int = 0
    status: str = RequestStatus.WAITING

    def is_finished(self) -> bool:
        return self.status.startswith("FINISHED")


def hashes_for(prefix: str, count: int) -> list[bytes]:
    """Deterministic 32-byte chain hashes; a shared prefix name gives a shared hash prefix."""
    return [f"{prefix}:{i:08d}".encode().ljust(32, b"\0") for i in range(count)]


@dataclass
class LoadRecord:
    request_id: str
    local_blocks: int
    loaded_blocks: int
    num_cached_tokens: int
    hashed_block_ids: list[int] = field(default_factory=list)


class OffloadingConnectorScheduler:
    """The hit answer is scripted per request; update_state_after_alloc mirrors the hashed-is-local scan."""

    def __init__(self, block_size: int, group_indices: tuple[int, ...]) -> None:
        self.block_size = block_size
        self.hits: dict[str, int] = {}
        self.loads: list[LoadRecord] = []
        self.resets = 0
        self.during_update: Callable[[], None] | None = None  # test hook: runs inside the stock call, after the scan
        self.config = SimpleNamespace(blocks_per_chunk=1, tokens_per_hash=block_size)
        self._offload_group_indices = group_indices

    def get_num_new_matched_tokens(self, request: Request, num_computed_tokens: int) -> tuple[int | None, bool]:
        hit = self.hits.get(request.request_id, 0)
        return hit, bool(hit)

    def update_state_after_alloc(self, request: Request, blocks: KVCacheBlocks, num_external_tokens: int) -> None:
        if num_external_tokens == 0:
            return
        group_blocks = blocks.blocks[0]
        num_gpu_blocks = cdiv(num_external_tokens, self.block_size)
        assert len(group_blocks) >= num_gpu_blocks
        local = num_gpu_blocks
        for i, block in enumerate(group_blocks[:num_gpu_blocks]):
            if not block.is_null and block.block_hash is None:
                local = i
                break
        hashed = [b.block_id for b in group_blocks[:local]]
        self.loads.append(LoadRecord(request.request_id, local, num_gpu_blocks - local, num_external_tokens, hashed))
        if self.during_update is not None:
            self.during_update()

    def reset_cache(self) -> None:
        self.resets += 1


class OffloadingConnector:
    def __init__(self, inner: OffloadingConnectorScheduler) -> None:
        self.connector_scheduler = inner
        self._offload_group_indices = inner._offload_group_indices

    def get_num_new_matched_tokens(self, request: Request, num_computed_tokens: int) -> tuple[int | None, bool]:
        return self.connector_scheduler.get_num_new_matched_tokens(request, num_computed_tokens)

    def update_state_after_alloc(self, request: Request, blocks: KVCacheBlocks, num_external_tokens: int) -> None:
        projected = KVCacheBlocks(tuple(blocks.blocks[i] for i in self._offload_group_indices))
        self.connector_scheduler.update_state_after_alloc(request, projected, num_external_tokens)

    def reset_cache(self) -> bool:
        self.connector_scheduler.reset_cache()
        return True


class Scheduler:
    """Enough of vllm.v1.core.sched.scheduler.Scheduler for the pin: construction, the hit/alloc sequence, finish."""

    def __init__(self, *, num_blocks: int = 64, block_size: int = 128, enable_prefix_caching: bool = False,
                 with_connector: bool = True, with_tail: bool = True, group_indices: tuple[int, ...] | None = None,
                 groups: tuple[tuple[str, int], ...] | None = None, scheduler_block_size: int | None = None,
                 ) -> None:
        """groups: (kind, block size) per KV group, kinds in MANAGER_KINDS (default: full + tail at block_size);
        scheduler_block_size: the engine's Scheduler.block_size when it differs from the hash block size (hybrid)."""
        pool = BlockPool(num_blocks)
        if groups is None:
            groups = (("full", block_size),) + ((("tail", block_size),) if with_tail else ())
        managers = [MANAGER_KINDS[kind](pool, size, index) for index, (kind, size) in enumerate(groups)]
        self.kv_cache_manager = KVCacheManager(KVCacheCoordinator(managers, pool), enable_prefix_caching)
        self.cache_config = SimpleNamespace(enable_prefix_caching=enable_prefix_caching)
        self.block_size = block_size if scheduler_block_size is None else scheduler_block_size
        self.hash_block_size = block_size
        if group_indices is None:
            group_indices = tuple(range(len(managers)))
        inner = OffloadingConnectorScheduler(block_size, group_indices)
        self.connector = OffloadingConnector(inner) if with_connector else None
        self.inner = inner
        self.waiting: deque[Request] = deque()
        self.running: list[Request] = []
        self.requests: dict[str, Request] = {}
        self.max_model_len = 1 << 20
        self._inflight_prefills: list[Request] = []

    def schedule(self) -> None:
        return None

    def shutdown(self) -> None:
        self.shut_down = True

    def reset_prefix_cache(self, reset_running_requests: bool = False, reset_connector: bool = False) -> bool:
        """The stock order: preempt the running requests when asked, reset the KV manager, then the connector."""
        while reset_running_requests and self.running:
            request = self.running.pop()
            self.kv_cache_manager.free(request)  # _preempt_request frees while the request is still RUNNING
            request.status = RequestStatus.PREEMPTED
            self.waiting.appendleft(request)
        reset_successful = self.kv_cache_manager.reset_prefix_cache()
        if reset_running_requests and not reset_successful:
            raise RuntimeError("Failed to reset KV cache even when all the running requests are preempted and moved to "
                               "the waiting queue.")
        if reset_connector:
            reset_successful = self.reset_connector_cache() and reset_successful
        return reset_successful

    def reset_connector_cache(self) -> bool:
        return self.connector is None or self.connector.reset_cache() is not False

    def _inflight_prefill_reserved_blocks(self) -> int:
        return sum(self._request_remaining_blocks(r) for r in self._inflight_prefills)

    def _request_remaining_blocks(self, request: Request) -> int:
        return self.kv_cache_manager.coordinator.get_num_blocks_to_allocate(
            request.request_id, num_tokens=request.num_tokens,
            new_computed_blocks=self.kv_cache_manager.empty_kv_cache_blocks.blocks, num_encoder_tokens=0,
            total_computed_tokens=request.num_computed_tokens, num_local_computed_tokens=request.num_computed_tokens,
            num_tokens_main_model=request.num_tokens, apply_admission_cap=True)

    # The waiting-loop sequence of sched/scheduler.py:1961-2298 for one new request with a connector hit.
    def admit(self, request: Request, num_new_tokens: int = 0) -> KVCacheBlocks | None:
        self.requests[request.request_id] = request
        ext, load_async = self.connector.get_num_new_matched_tokens(request, 0)
        assert ext is not None
        new_blocks = self.kv_cache_manager.allocate_slots(
            request, num_new_tokens if not load_async else 0, num_external_computed_tokens=ext,
            delay_cache_blocks=load_async)
        if new_blocks is None:
            return None
        self.connector.update_state_after_alloc(request, self.kv_cache_manager.get_blocks(request.request_id), ext)
        request.num_computed_tokens = ext
        request.status = RequestStatus.WAITING_FOR_REMOTE_KVS if load_async else RequestStatus.RUNNING
        return new_blocks

    def finish(self, request: Request, status: str = RequestStatus.FINISHED_STOPPED) -> None:
        request.status = status
        self.kv_cache_manager.free(request)
        self.requests.pop(request.request_id, None)


def run_prompt_to_end(scheduler: Scheduler, request: Request, num_tokens: int) -> None:
    """Grow a RUNNING request to num_tokens the stock way (allocate_slots per chunk) and mark it computed."""
    request.status = RequestStatus.RUNNING
    new = num_tokens - request.num_computed_tokens
    result = scheduler.kv_cache_manager.allocate_slots(request, new)
    assert result is not None, "fake pool too small for the fixture"
    request.num_computed_tokens = num_tokens
    request.num_tokens = num_tokens


class ExtendFailsOnce(list):
    """A req_to_blocks list whose first extend raises before it changes anything: a fault at the pin's handover."""

    def __init__(self, *blocks: Any) -> None:
        super().__init__(*blocks)
        self.failed = False

    def extend(self, blocks: Iterable[Any]) -> None:
        if not self.failed:
            self.failed = True
            raise RuntimeError("injected fault: req_to_blocks.extend")
        super().extend(blocks)


def ownership_faults(scheduler: Scheduler, held: Iterable[Sequence[KVCacheBlock]]) -> list[str]:
    """Every non-null block has exactly one owner (a request's req_to_blocks, a pin's held list, or the free queue) and
    the matching ref_cnt; returns one line per block that breaks it (two owners, or none: a leak)."""
    managers = scheduler.kv_cache_manager.coordinator.single_type_managers
    requests = [blocks for manager in managers for blocks in manager.req_to_blocks.values()]
    owners = Counter(id(block) for blocks in (*requests, *held) for block in blocks)
    pool = scheduler.kv_cache_manager.block_pool
    queued = Counter(id(block) for block in pool.free_block_queue)
    faults = []
    for block in pool.blocks[1:]:
        state = (owners[id(block)], queued[id(block)], block.ref_cnt)
        if state not in ((1, 0, 1), (0, 1, 0)):
            faults.append(f"block {block.block_id}: owners={state[0]} queued={state[1]} ref_cnt={state[2]}")
    return faults
