# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import os
import time
from collections.abc import Iterable, MutableMapping, Sequence
from dataclasses import dataclass, field
from itertools import chain, islice
from typing import Any, NamedTuple

from vllm.config import VllmConfig
from vllm.distributed.kv_events import KVCacheEvent
from vllm.distributed.kv_transfer.kv_connector.utils import yield_req_data
from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorMetadata
from vllm.distributed.kv_transfer.kv_connector.v1.offloading.common import (
    OffloadingConnectorMetadata,
    OffloadingWorkerMetadata,
    ReqId,
    RecurrentSidecarTransfer,
    TransferJob,
)
from vllm.distributed.kv_transfer.kv_connector.v1.offloading import (
    recurrent_checkpoint_flags,
)
from vllm.distributed.kv_transfer.kv_connector.v1.offloading.recurrent_checkpoint_coordinator import (  # noqa: E501
    RecurrentCheckpointCoordinator,
)
from vllm.distributed.kv_transfer.kv_connector.v1.offloading.events import (
    OffloadingEventGroupSpec,
    OffloadingEventsTracker,
    get_offloading_event_group_spec,
)
from vllm.distributed.kv_transfer.kv_connector.v1.offloading.metrics import (
    OffloadingConnectorStats,
    _ConnectorMetricName,
    _TransferMetricName,
)
import glm53_tail_fuse as _tail_fuse
import glm53_decode_checkpoint as _decode_ckpt
import glm53_miss_attrib as _miss_attrib
from vllm.logger import init_logger
from vllm.v1.kv_offload.cpu.recurrent_sidecar import validate_budget_split
from vllm.utils.math_utils import cdiv, round_down
from vllm.v1.core.kv_cache_manager import KVCacheBlocks
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.kv_cache_interface import (
    ChunkedLocalAttentionSpec,
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheSpec,
    MambaSpec,
    SlidingWindowSpec,
    UniformTypeKVCacheSpecs,
)
from vllm.v1.kv_offload.base import (
    GPULoadStoreSpec,
    Locality,
    LookupResult,
    Medium,
    OffloadingManager,
    OffloadingSpec,
    OffloadKey,
    OffloadPolicy,
    ReqContext,
    RequestOffloadingContext,
    ScheduleEndContext,
    TierFilter,
    TierMatcher,
    make_offload_key,
) 
from vllm.v1.kv_offload.cpu.common import CPULoadStoreSpec
from vllm.v1.kv_offload.cpu.manager import CPUOffloadingManager
from vllm.v1.outputs import KVConnectorOutput
from vllm.v1.request import Request, RequestStatus

logger = init_logger(__name__)

KV_LOAD_TIERS_KEY = "kv_load_tiers"
MATCHER_MEDIUM_KEY = "medium"
MATCHER_LOCALITY_KEY = "locality"


# R463_EXACT_PENDING_STORE_DEPENDENCIES_V1
# R466_PENDING_STORE_WAIT_DEADLINE_V1
def _parse_pending_store_wait_deadline(value: object) -> float:
    """Parse a finite non-negative exact-dependency wait deadline."""
    try:
        deadline_s = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(
            "pending_store_wait_deadline_s must be a finite non-negative number"
        ) from error
    if not 0.0 <= deadline_s < float("inf"):
        raise ValueError(
            "pending_store_wait_deadline_s must be a finite non-negative number"
        )
    return deadline_s

def _resolve_finished_store_prefix(
    hit_count: int,
    first_dependency_hit_count: int | None,
    has_other_defer: bool,
    tokens_per_chunk: int,
    wait_threshold: int,
    dependency_wait_elapsed_s: float,
    wait_deadline_s: float,
) -> int | None:
    """Choose whether an exact pending-store prefix should wait or recompute."""
    if has_other_defer:
        return None
    if first_dependency_hit_count is None:
        return hit_count
    dependent_tokens = (
        hit_count - first_dependency_hit_count
    ) * tokens_per_chunk
    if (
        dependent_tokens >= wait_threshold
        and dependency_wait_elapsed_s < wait_deadline_s
    ):
        return None
    return first_dependency_hit_count


@dataclass(slots=True)
class TransferJobStatus:
    """Tracks scheduler-side state for a single transfer job."""

    req_id: ReqId
    # Number of workers still pending. Starts at num_workers,
    # decremented as each worker reports completion. Job is done at 0.
    pending_count: int
    # Offload keys this job covers; passed to manager.complete_*().
    keys: set[OffloadKey]
    is_store: bool
    # Store source blocks fenced after the request finishes.
    deferred_fence_block_ids: list[int] | None = None
    # Store source blocks fenced when the transfer is created.
    fenced_block_ids: list[int] | None = None
    # Exact-boundary recurrent state managed by the independent CPU LRU.
    recurrent_sidecar_key: OffloadKey | None = None
    attention_transfer: bool = True


# DECODE_CKPT_V1: the scheduler status and the worker job of a sidecar-only store. The
# recurrent post-forward builder spells the same pair inline (its body is pinned by the
# w-tailfuse suites until r7 merges) and can adopt these then.
def _sidecar_store_status(
    req_id: ReqId, key: OffloadKey, num_workers: int
) -> TransferJobStatus:
    return TransferJobStatus(
        req_id=req_id,
        pending_count=num_workers,
        keys=set(),
        is_store=True,
        recurrent_sidecar_key=key,
        attention_transfer=False,
    )


def _sidecar_only_job(
    req_id: ReqId, transfer: RecurrentSidecarTransfer
) -> TransferJob:
    return TransferJob(
        req_id=req_id,
        src_spec=None,
        dst_spec=None,
        attention_transfer=False,
        recurrent_sidecar=transfer,
    )


class GroupOffloadConfig(NamedTuple):
    group_idx: int
    tokens_per_block: int
    tokens_per_chunk: int
    hashes_per_chunk: int
    # KV cache spec metadata propagated onto emitted BlockStored events so
    # KV-aware consumers can classify and filter the group.
    kv_event_group_spec: OffloadingEventGroupSpec
    # None below means full attention
    sliding_window_size_in_chunks: int | None
    # Partial-tail data for this group comes from the scheduler's CoW hand-off
    # rather than the request block table.
    requires_cow_source: bool = False
    # Number of this group's offloaded chunks per full-attention alignment
    # segment. Used to skip storing SWA chunks that can never serve a load
    # hit (e.g. DeepSeek V4 where SWA groups have much smaller block sizes
    # than the MLA full-attention group).
    # None for full-attention groups or when the optimization doesn't apply.
    alignment_chunk_count: int | None = None
    # True for EAGLE/MTP draft-model attention groups. The trailing chunk
    # of these groups is volatile and lacks a stable hash, so it must
    # be excluded from store and load scheduling.
    is_eagle_group: bool = False


def get_sliding_window_size_in_chunks(
    kv_cache_spec: KVCacheSpec, tokens_per_chunk: int
) -> int | None:
    if isinstance(kv_cache_spec, SlidingWindowSpec):
        assert kv_cache_spec.sliding_window > 0
        return cdiv(kv_cache_spec.sliding_window, tokens_per_chunk)

    if isinstance(kv_cache_spec, ChunkedLocalAttentionSpec):
        # Attention never reaches back past one chunk
        assert kv_cache_spec.attention_chunk_size > 0
        return cdiv(kv_cache_spec.attention_chunk_size, tokens_per_chunk)

    if isinstance(kv_cache_spec, MambaSpec):
        # Mamba depends on a single state
        return 1

    assert isinstance(kv_cache_spec, FullAttentionSpec)
    return None


def is_store_reachable_swa_chunk(
    absolute_chunk_index: int,
    storable_chunk_count: int,
    alignment_chunk_count: int | None,
    sliding_window_chunks: int | None,
    is_eagle_group: bool,
) -> bool:
    """Return whether an SWA chunk can participate in an external-cache hit."""
    if alignment_chunk_count is None:
        return True
    assert sliding_window_chunks is not None
    position_in_segment = absolute_chunk_index % alignment_chunk_count
    segment_start = absolute_chunk_index - position_in_segment
    actual_segment_length = min(
        alignment_chunk_count, storable_chunk_count - segment_start
    )
    reachable_tail = sliding_window_chunks + int(is_eagle_group)
    return position_in_segment >= actual_segment_length - reachable_tail


def resolve_mamba_align_size(
    spec: "OffloadingSpec", kv_cache_config: KVCacheConfig
) -> int | None:
    """Scan all KV cache groups in *spec* and return the single mamba alignment
    size, or None if no group requires mamba alignment.

    For MambaSpec groups in "align" or "all" cache mode the hit window must be
    rounded down to a multiple of the offloaded chunk size. Asserts that all
    such groups agree on the same value.
    """
    mamba_align_size: int | None = None
    for idx, tokens_per_block in enumerate(spec.tokens_per_block):
        kv_spec = kv_cache_config.kv_cache_groups[idx].kv_cache_spec
        if isinstance(kv_spec, MambaSpec) and kv_spec.mamba_cache_mode in (
            "align",
            "all",
        ):
            tokens_per_chunk = tokens_per_block * spec.blocks_per_chunk
            assert mamba_align_size is None or mamba_align_size == tokens_per_chunk
            mamba_align_size = tokens_per_chunk
    return mamba_align_size


class SchedulerOffloadConfig(NamedTuple):
    kv_group_configs: tuple[GroupOffloadConfig, ...]
    blocks_per_chunk: int
    tokens_per_hash: int
    num_workers: int
    offload_prompt_only: bool
    supports_partial_tail: bool

    @classmethod
    def from_spec(
        cls,
        spec: OffloadingSpec,
        vllm_config: VllmConfig,
        kv_cache_config: KVCacheConfig,
    ) -> "SchedulerOffloadConfig":
        # Determine the alignment token count from the full-attention group(s).
        # This is the tokens_per_chunk of the full-attention group; load
        # hits are always aligned to this boundary, so SWA blocks earlier in
        # each segment can never serve a load hit. Relevant for hybrid
        # architectures like DeepSeek V4 (MLA + SWA groups).
        full_attn_tokens_per_chunk: set[int] = set()
        for idx, tokens_per_block in enumerate(spec.tokens_per_block):
            kv_spec = kv_cache_config.kv_cache_groups[idx].kv_cache_spec
            sw = get_sliding_window_size_in_chunks(
                kv_spec, tokens_per_block * spec.blocks_per_chunk
            )
            if sw is None:
                full_attn_tokens_per_chunk.add(tokens_per_block * spec.blocks_per_chunk)

        # Only apply the optimization if there's a single consistent
        # full-attention alignment size.
        alignment_tokens: int | None = None
        if len(full_attn_tokens_per_chunk) == 1:
            alignment_tokens = full_attn_tokens_per_chunk.pop()

        def _alignment_chunk_count(
            tokens_per_chunk: int,
            sliding_window_size_in_chunks: int | None,
        ) -> int | None:
            if alignment_tokens is None or sliding_window_size_in_chunks is None:
                return None
            if alignment_tokens <= tokens_per_chunk:
                return None
            per_segment = alignment_tokens // tokens_per_chunk
            if sliding_window_size_in_chunks >= per_segment:
                return None
            return per_segment

        eagle_groups = {
            idx
            for idx, g in enumerate(kv_cache_config.kv_cache_groups)
            if g.is_eagle_group
        }

        use_eagle = (
            vllm_config.speculative_config is not None
            and vllm_config.speculative_config.use_eagle()
        )
        if use_eagle and not eagle_groups:
            eagle_groups = set(range(len(kv_cache_config.kv_cache_groups)))

        if eagle_groups:
            logger.info(
                "KV offloading: EAGLE/MTP draft attention groups %s "
                "detected. The trailing chunk of these groups will be "
                "excluded from offloading due to volatility.",
                sorted(eagle_groups),
            )

        kv_group_configs_list: list[GroupOffloadConfig] = []
        for idx, tokens_per_block in enumerate(spec.tokens_per_block):
            kv_cache_group = kv_cache_config.kv_cache_groups[idx]
            kv_spec = kv_cache_group.kv_cache_spec
            sw = get_sliding_window_size_in_chunks(
                kv_spec, tokens_per_block * spec.blocks_per_chunk
            )
            kv_group_configs_list.append(
                GroupOffloadConfig(
                    group_idx=idx,
                    tokens_per_block=tokens_per_block,
                    tokens_per_chunk=tokens_per_block * spec.blocks_per_chunk,
                    hashes_per_chunk=(
                        (tokens_per_block * spec.blocks_per_chunk)
                        // spec.tokens_per_hash
                    ),
                    sliding_window_size_in_chunks=sw,
                    alignment_chunk_count=_alignment_chunk_count(
                        tokens_per_block * spec.blocks_per_chunk, sw
                    ),
                    kv_event_group_spec=get_offloading_event_group_spec(kv_cache_group),
                    is_eagle_group=idx in eagle_groups,
                    requires_cow_source=(
                        isinstance(kv_spec, MambaSpec)
                        and kv_spec.mamba_cache_mode == "align"
                    ),
                )
            )
        kv_group_configs = tuple(kv_group_configs_list)
        group_block_sizes = {config.tokens_per_block for config in kv_group_configs}
        has_partial_recurrent_group = any(
            config.requires_cow_source
            and config.tokens_per_block > spec.tokens_per_hash
            for config in kv_group_configs
        )
        # Partial tails currently require one physical block per offload chunk
        # and uniform, non-windowed groups so one boundary identifies every
        # group's source. EAGLE and DCP need additional hand-off semantics.
        supports_partial_tail = (
            spec.blocks_per_chunk == 1
            and len(group_block_sizes) == 1
            and has_partial_recurrent_group
            and all(
                config.sliding_window_size_in_chunks is None
                or config.requires_cow_source
                for config in kv_group_configs
            )
            and not any(config.is_eagle_group for config in kv_group_configs)
            and vllm_config.parallel_config.decode_context_parallel_size == 1
        )

        return cls(
            num_workers=vllm_config.parallel_config.world_size,
            kv_group_configs=kv_group_configs,
            blocks_per_chunk=spec.blocks_per_chunk,
            tokens_per_hash=spec.tokens_per_hash,
            offload_prompt_only=spec.offload_prompt_only,
            supports_partial_tail=supports_partial_tail,
        )


@dataclass
class RequestGroupState:
    offload_keys: list[OffloadKey] = field(default_factory=list)
    block_ids: list[int] = field(default_factory=list)
    # Index of the next chunk to offload.
    next_stored_chunk_idx: int = 0
    # Number of offloaded chunks hit (including GPU prefix cache)
    # when the request first started
    num_hit_chunks: int = 0


@dataclass(slots=True)
class RequestOffloadState:
    config: SchedulerOffloadConfig
    req: Request
    req_context: ReqContext
    offloading_context: RequestOffloadingContext
    group_states: tuple[RequestGroupState, ...] = field(init=False)
    # upper bound on tokens to offload for this request; None means no cap
    max_offload_tokens: int | None = None
    # number of hits in the GPU cache
    num_locally_computed_tokens: int = 0
    # In-flight job IDs. Per the connector's invariant, at any given time
    # this contains either a single load job, or one or more store jobs.
    transfer_jobs: set[int] = field(default_factory=set)
    # time.monotonic() of this request's first deferred offload lookup;
    # None once consumed (observed) or while no lookup is pending.
    deferred_lookup_start_time: float | None = None
    # Start of this request's current large exact-store dependency wait.
    pending_store_wait_start_time: float | None = None
    # Whether this lookup observed an exact dependency that uses the wait policy.
    pending_store_wait_deferred: bool = False
    # Fine-grained token boundary selected beyond the last complete offload
    # chunk. It is consumed when the corresponding load is scheduled.
    partial_tail_boundary: int | None = None
    # True once on_request_finished has been signaled to the manager.
    finished_signaled: bool = False
    # DECODE_CKPT_V1: deepest decode boundary planned for this request (0: none).
    # Raises the prompt-only store limit to it (see _num_storable_tokens).
    decode_checkpoint_boundary: int = 0
    # VLLM_OFFLOAD_MISS_ATTRIB: this request's GLM53_MISS line has been logged.
    miss_recorded: bool = False

    def __post_init__(self) -> None:
        self.group_states = tuple(
            RequestGroupState() for _ in self.config.kv_group_configs
        )
        params = self.req.kv_transfer_params

        # NOTE: This field is experimental and subject to change in the future.
        raw = params.get("max_offload_tokens") if params else None
        if type(raw) is int and raw >= 0:
            self.max_offload_tokens = raw
            logger.debug(
                "Request %s: max_offload_tokens set to %d",
                self.req.request_id,
                raw,
            )
        elif raw is not None:
            logger.warning(
                "max_offload_tokens must be a non-negative int, got %r; ignoring", raw
            )

    def update_offload_keys(self) -> None:
        for group_config, group_state in zip(
            self.config.kv_group_configs, self.group_states
        ):
            for req_block_hash in islice(
                self.req.block_hashes,
                group_config.hashes_per_chunk * len(group_state.offload_keys)
                + group_config.hashes_per_chunk
                - 1,
                None,
                group_config.hashes_per_chunk,
            ):
                group_state.offload_keys.append(
                    make_offload_key(req_block_hash, group_config.group_idx)
                )

    def update_block_id_groups(
        self, new_block_id_groups: tuple[list[int], ...] | None
    ) -> None:
        if new_block_id_groups is None:
            return

        assert len(new_block_id_groups) == len(self.group_states)
        for group_state, new_blocks in zip(self.group_states, new_block_id_groups):
            group_state.block_ids.extend(new_blocks)

    def storable_chunks(
        self,
        group_config: "GroupOffloadConfig",
        group_state: RequestGroupState,
        num_offloadable_tokens: int,
    ) -> int:
        """Number of allocated leading offloaded chunks eligible for store.

        For eagle/MTP groups the volatile trailing chunk of the offloadable
        range is excluded while decoding: the draft-layer KV of the last
        accepted position may be rewritten after spec-token rejection. During
        prefill the trailing chunk is stable (the draft input for a chunk's
        last position is the next prompt token), so it is stored immediately.
        The exclusion must be applied consistently everywhere
        ``next_stored_chunk_idx`` is derived: otherwise the trailing chunk of
        each step is skipped on collection but jumped over by
        ``next_stored_chunk_idx``, so it is never re-considered and a
        permanent hole breaks prefix-reuse lookup.
        """
        num_chunks = num_offloadable_tokens // group_config.tokens_per_chunk
        is_decoding = num_offloadable_tokens > self.req.num_prompt_tokens
        if group_config.is_eagle_group and is_decoding:
            num_chunks = max(0, num_chunks - 1)
        num_allocated_chunks = (
            len(group_state.block_ids) // self.config.blocks_per_chunk
        )
        return min(num_chunks, num_allocated_chunks)

    def advance_stored_idx(self, num_offloadable_tokens: int) -> None:
        # max(): at the prefill->decode transition of a chunk-aligned prompt,
        # storable_chunks drops by one (the eagle exclusion kicks in), and the
        # index must not move backwards past already-stored chunks.
        for group_config, group_state in zip(
            self.config.kv_group_configs, self.group_states
        ):
            group_state.next_stored_chunk_idx = max(
                group_state.next_stored_chunk_idx,
                self.storable_chunks(group_config, group_state, num_offloadable_tokens),
            )

    def update_num_hit_chunks(self, num_cached_tokens: int) -> None:
        for group_config, group_state in zip(
            self.config.kv_group_configs, self.group_states
        ):
            group_state.num_hit_chunks = (
                num_cached_tokens // group_config.tokens_per_chunk
            )


def _parse_tier_filter(raw: Any) -> TierFilter:
    """Parse raw kv_transfer_params tier matchers into a TierFilter."""
    if not isinstance(raw, list):
        logger.warning(
            "_parse_tier_filter: expected list, got %s; ignoring",
            type(raw).__name__,
        )
        return TierFilter.ALL
    matchers: list[TierMatcher] = []
    for entry in raw:
        if not isinstance(entry, dict):
            logger.warning("_parse_tier_filter: entry is not a dict; skipping")
            continue
        medium: Medium | None = None
        locality: Locality | None = None
        raw_medium = entry.get(MATCHER_MEDIUM_KEY)
        if raw_medium is not None:
            try:
                medium = Medium(raw_medium.upper())
            except (ValueError, AttributeError):
                logger.warning(
                    "_parse_tier_filter: unknown medium %r; skipping entry",
                    raw_medium,
                )
                continue
        raw_locality = entry.get(MATCHER_LOCALITY_KEY)
        if raw_locality is not None:
            try:
                locality = Locality(raw_locality.upper())
            except (ValueError, AttributeError):
                logger.warning(
                    "_parse_tier_filter: unknown locality %r; skipping entry",
                    raw_locality,
                )
                continue
        matchers.append(TierMatcher(medium=medium, locality=locality))
    if not matchers:
        if not raw:  # input was [] — user explicitly wants nothing
            return TierFilter(matchers=())
        # all entries were invalid — fall back to ALL
        return TierFilter.ALL
    return TierFilter(matchers=tuple(matchers))


def _create_req_context(req: Request) -> ReqContext:
    params = req.kv_transfer_params
    load_filter = TierFilter.ALL
    if params:
        raw = params.get(KV_LOAD_TIERS_KEY)
        if raw is not None:
            load_filter = _parse_tier_filter(raw)
    return ReqContext(
        req_id=req.request_id,
        kv_transfer_params=params,
        load_tier_filter=load_filter,
    )


def recurrent_state_bytes_per_rank(kv_cache_config: KVCacheConfig) -> int:
    """Native bytes for one recurrent boundary on one worker rank."""
    total = 0
    for group in kv_cache_config.kv_cache_groups:
        group_spec = group.kv_cache_spec
        per_layer = (
            group_spec.kv_cache_specs
            if isinstance(group_spec, UniformTypeKVCacheSpecs)
            else {}
        )
        for layer_name in group.layer_names:
            spec = per_layer.get(layer_name, group_spec)
            if isinstance(spec, MambaSpec):
                total += spec.state_content_size_bytes
    return total


class OffloadingConnectorScheduler:
    """Implementation of Scheduler side methods"""

    def __init__(
        self,
        spec: OffloadingSpec,
        vllm_config: VllmConfig,
        kv_cache_config: KVCacheConfig,
        recurrent_kv_cache_config: KVCacheConfig | None = None,
    ):
        self.config = SchedulerOffloadConfig.from_spec(
            spec, vllm_config, kv_cache_config
        )
        self.manager: OffloadingManager = spec.get_manager()
        recurrent_config = recurrent_kv_cache_config or kv_cache_config
        self._recurrent_state_bytes_per_rank = recurrent_state_bytes_per_rank(
            recurrent_config
        )
        raw_sidecar_budget = spec.extra_config.get(
            "recurrent_state_bytes_to_use", 0
        )
        try:
            sidecar_budget = int(raw_sidecar_budget)
        except (TypeError, ValueError) as error:
            raise ValueError(
                "recurrent_state_bytes_to_use must be an integer byte count"
            ) from error
        if sidecar_budget < 0:
            raise ValueError("recurrent_state_bytes_to_use must be non-negative")
        raw_total_budget = spec.extra_config.get("total_cpu_bytes_to_use")
        if sidecar_budget and raw_total_budget is None:
            raise ValueError(
                "total_cpu_bytes_to_use is required with recurrent state offload"
            )
        if raw_total_budget is not None:
            validate_budget_split(
                int(spec.extra_config["cpu_bytes_to_use"]),
                sidecar_budget,
                int(raw_total_budget),
            )
        global_state_bytes = (
            self._recurrent_state_bytes_per_rank
            * vllm_config.parallel_config.world_size
        )
        self._recurrent_manager: CPUOffloadingManager | None = None
        if sidecar_budget and global_state_bytes:
            num_sidecar_blocks = sidecar_budget // global_state_bytes
            logger.info(
                "Recurrent host cache: budget_bytes=%d boundary_bytes=%d capacity=%d",
                sidecar_budget, global_state_bytes, num_sidecar_blocks,
            )
            if num_sidecar_blocks < 1:
                raise ValueError(
                    "recurrent_state_bytes_to_use cannot hold one all-worker state"
                )
            if vllm_config.cache_config.enable_prefix_caching:
                raise ValueError(
                    "recurrent sidecar does not support GPU prefix caching"
                )
            self._recurrent_manager = CPUOffloadingManager(
                num_blocks=num_sidecar_blocks,
                cache_policy="lru",
                enable_events=False,
                store_threshold=1,
                bytes_per_block=global_state_bytes,
            )
        self._recurrent_sidecar_registrations = 0
        self._recurrent_sidecar_signature: tuple[int, str] | None = None
        # Capability remains false until every rank reports compatible state.
        self.recurrent_sidecar_ready = False
        self._connector_stats = OffloadingConnectorStats()
        # R593_COLD_REUSE_MIN_TOKENS_V1
        raw_cold_reuse_min_tokens = spec.extra_config.get(
            "cold_reuse_min_tokens", 0
        )
        try:
            self._cold_reuse_min_tokens = int(raw_cold_reuse_min_tokens)
        except (TypeError, ValueError) as error:
            raise ValueError("cold_reuse_min_tokens must be an integer") from error
        if self._cold_reuse_min_tokens < 0:
            raise ValueError("cold_reuse_min_tokens must be non-negative")
        self._current_batch_attention_ready_req_ids: set[ReqId] = set()
        # TAIL_FUSE_V1: {req_id: (start, b, snapshot_index)} accepted during this schedule().
        self._tail_fuse_notes: dict[ReqId, tuple[int, int, int]] = {}
        _tail_fuse.log_disabled_once()
        raw_store_batch_keys = spec.extra_config.get("store_batch_keys", 1)
        try:
            self._store_batch_keys = int(raw_store_batch_keys)
        except (TypeError, ValueError) as error:
            raise ValueError("store_batch_keys must be an integer") from error
        if self._store_batch_keys < 1:
            raise ValueError("store_batch_keys must be positive")
        default_wait_threshold = 4 * min(
            group.tokens_per_chunk for group in self.config.kv_group_configs
        )
        raw_wait_threshold = spec.extra_config.get(
            "pending_store_wait_threshold", default_wait_threshold
        )
        try:
            self._pending_store_wait_threshold = int(raw_wait_threshold)
        except (TypeError, ValueError) as error:
            raise ValueError(
                "pending_store_wait_threshold must be an integer"
            ) from error
        if self._pending_store_wait_threshold < 0:
            raise ValueError(
                "pending_store_wait_threshold must be non-negative"
            )
        self._pending_store_wait_deadline_s = (
            _parse_pending_store_wait_deadline(
                spec.extra_config.get("pending_store_wait_deadline_s", 2.0)
            )
        )

        full_attention_groups: list[int] = []
        sliding_window_groups: list[int] = []
        for group_config in self.config.kv_group_configs:
            if group_config.sliding_window_size_in_chunks is None:
                full_attention_groups.append(group_config.group_idx)
            else:
                sliding_window_groups.append(group_config.group_idx)

        # sort sliding window groups by window size in decreasing order
        def _sliding_window_sort_key(i: int) -> int:
            val = self.config.kv_group_configs[i].sliding_window_size_in_chunks
            assert val is not None
            return val

        sliding_window_groups.sort(key=_sliding_window_sort_key, reverse=True)

        # used by _lookup
        self._sliding_window_groups: tuple[int, ...] = tuple(sliding_window_groups)
        self._lookup_groups = tuple(full_attention_groups) + self._sliding_window_groups
        self._mamba_align_size: int | None = resolve_mamba_align_size(
            spec, kv_cache_config
        )
        self._partial_tail_block_size = (
            self.config.kv_group_configs[0].tokens_per_block
            if self.config.supports_partial_tail
            else 0
        )
        self._cow_source_groups = frozenset(
            config.group_idx
            for config in self.config.kv_group_configs
            if config.requires_cow_source
        )

        self._req_status: dict[ReqId, RequestOffloadState] = {}
        # Keys that a finished request can still publish.  The reverse index
        # makes lookup an exact O(1) dependency check; per-request ownership
        # prevents one duplicate producer from releasing another producer's key.
        self._pending_finished_store_keys: dict[ReqId, set[OffloadKey]] = {}
        self._pending_finished_store_key_owners: dict[
            OffloadKey, set[ReqId]
        ] = {}
        self._current_batch_load_jobs: dict[int, TransferJob] = {}
        self._current_batch_jobs_to_flush: set[int] = set()
        # GPU block IDs allocated in the current engine step
        self._current_batch_allocated_block_ids: set[int] = set()
        # if GPU prefix caching is enabled,
        # Track loaded chunks to avoid redundant loads.
        self._chunks_being_loaded: set[OffloadKey] | None = (
            set() if vllm_config.cache_config.enable_prefix_caching else None
        )

        # Job ID counter shared by loads and stores.
        self._job_counter: int = 0
        # Threshold value for stale jobs. All job ids >= _stale_job_threshold are
        # active jobs.
        self._stale_job_threshold: int = 0
        self._jobs: dict[int, TransferJobStatus] = {}

        # block_id -> pending store job_ids. Used to track jobs that needs
        # flushing in case a block is re-allocated by the KV cache manager.
        # Populated only for finished requests (running-request blocks are
        # protected by their ref_cnt) and for sliding window blocks (which can
        # be freed before a request finishes).
        self._block_id_to_pending_jobs: dict[int, set[int]] = {}

        self._events_tracker = OffloadingEventsTracker(spec.kv_events_config)
        self._checkpoints = self._build_checkpoint_coordinator()
        self._wire_checkpoint_removal_listeners()
        self._decode_checkpoints = self._init_decode_checkpoints(vllm_config)

    def _build_checkpoint_coordinator(self) -> RecurrentCheckpointCoordinator:
        """V2 divergence checkpoints; inert unless verbs and a sidecar exist."""
        verbs = (
            recurrent_checkpoint_flags.RECURRENT_CHECKPOINT_V2_FOR
            if self._recurrent_manager is not None
            else frozenset()
        )
        if not recurrent_checkpoint_flags.completion_first_cache_aware():
            # The hint only feeds the cache-aware completion-first gate.
            verbs = verbs - {recurrent_checkpoint_flags.GATE_HINT}
        return RecurrentCheckpointCoordinator.build(
            verbs=verbs,
            tokens_per_hash=self.config.tokens_per_hash,
            tokens_per_chunks=[
                group.tokens_per_chunk for group in self.config.kv_group_configs
            ],
            min_tokens=self._cold_reuse_min_tokens,
            count=self._increase_checkpoint_counter,
            merge_keep=recurrent_checkpoint_flags.JUNCTION_MERGE_KEEP,
        )

    def _wire_checkpoint_removal_listeners(self) -> None:
        """Every manager removal (eviction, failed store) reaches the index and, flag on, the miss attribution."""
        self._wire_miss_attribution()
        if not self._checkpoints.active:
            return
        if self._recurrent_manager is not None:
            self._recurrent_manager.add_removal_listener(self._checkpoints.forget_keys)
        if self.manager.supports_removal_listener:
            self.manager.add_removal_listener(
                self._checkpoints.on_attention_keys_removed
            )
        if self._checkpoints.enabled(recurrent_checkpoint_flags.GATE_HINT) and not (
            self.manager.supports_read_only_lookup
            and self.manager.supports_removal_listener
        ):
            logger.warning(
                "gate_hint is inert: %s lacks supports_read_only_lookup or "
                "supports_removal_listener, so estimate_exact_hit_tokens "
                "returns 0. Use a CPUOffloadingManager or remove gate_hint "
                "from VLLM_RECURRENT_CHECKPOINT_V2_FOR.",
                type(self.manager).__name__,
            )

    def _wire_miss_attribution(self) -> None:
        """VLLM_OFFLOAD_MISS_ATTRIB=1: the log-only GLM53_MISS line; flag off adds no listener and no read."""
        self._miss_attrib = _miss_attrib.from_env(
            self.config.tokens_per_hash, lambda line: logger.warning("%s", line)
        )
        if self._miss_attrib is None:
            return
        if self._recurrent_manager is not None:
            self._recurrent_manager.add_removal_listener(
                self._miss_attrib.note_recurrent_removed
            )
        if self.manager.supports_removal_listener:
            self.manager.add_removal_listener(self._miss_attrib.note_attention_removed)

    def _walked_recurrent_keys(
        self, req_status: RequestOffloadState, attention_hit: int, recurrent_hit: int
    ) -> list[OffloadKey]:
        """The snapshot keys _require_ready_recurrent_sidecar found missing (no manager call)."""
        local_boundary = req_status.num_locally_computed_tokens
        initial_boundary = local_boundary + attention_hit
        boundaries = range(
            initial_boundary, local_boundary + recurrent_hit, -self.config.tokens_per_hash
        )
        keys = (
            self._recurrent_sidecar_key(req_status.req, boundary)
            for boundary in boundaries
            if self._is_sidecar_walk_boundary(boundary, initial_boundary)
        )
        return [key for key in keys if key is not None]

    def _missing_attention_key(
        self, req_status: RequestOffloadState, attention_hit: int
    ) -> OffloadKey | None:
        """The first attention chunk past the hit, when a read-only peek says it is absent."""
        if not self.manager.supports_read_only_lookup:
            return None
        req = req_status.req
        group = self.config.kv_group_configs[0]
        boundary = round_down(
            req_status.num_locally_computed_tokens + attention_hit,
            group.tokens_per_chunk,
        ) + group.tokens_per_chunk
        hash_count = boundary // self.config.tokens_per_hash
        if boundary > req.num_prompt_tokens or hash_count > len(req.block_hashes):
            return None
        key = self._make_boundary_key(req, group.group_idx, boundary)
        return key if self.manager.peek(key) is LookupResult.MISS else None

    def _attribute_miss(
        self,
        req_status: RequestOffloadState,
        attention_hit: int,
        recurrent_hit: int,
        reason: str,
        lookup_time: float = 0.0,
    ) -> None:
        """Log-only: one GLM53_MISS per request, at its first resolved lookup."""
        if self._miss_attrib is None or req_status.miss_recorded:
            return
        req_status.miss_recorded = True
        req = req_status.req
        wait_start = req_status.deferred_lookup_start_time
        looked_up = reason == _miss_attrib.REASON_LOOKUP
        self._miss_attrib.record(_miss_attrib.MissLookup(
            request_id=req.request_id,
            block_hashes=req.block_hashes,
            prompt_tokens=req.num_prompt_tokens,
            local_tokens=req_status.num_locally_computed_tokens,
            attention_hit=attention_hit,
            recurrent_hit=recurrent_hit,
            walked_recurrent_keys=self._walked_recurrent_keys(
                req_status, attention_hit, recurrent_hit
            ),
            missing_attention_key=(
                self._missing_attention_key(req_status, attention_hit)
                if looked_up else None
            ),
            store_wait_ms=(
                0.0 if wait_start is None else (lookup_time - wait_start) * 1000.0
            ),
            reason=reason,
        ))

    def _increase_checkpoint_counter(self, name: str, value: int) -> None:
        self._connector_stats.increase_counter(name, value)

    def _computed_tokens_of(self, req_id: ReqId) -> int | None:
        req_status = self._req_status.get(req_id)
        return None if req_status is None else req_status.req.num_computed_tokens

    def recurrent_checkpoint_positions(
        self, request: Request, start: int, end: int
    ) -> tuple[int, ...]:
        """Planned exact recurrent checkpoints strictly inside (start, end)."""
        return self._checkpoints.positions_in(request.request_id, start, end)

    def estimate_exact_hit_tokens(self, request: Request) -> int:
        """gate_hint: completed exact boundary whose attention key hits.

        Read-only: walks the completed-store index (never the pending pool)
        and probes both managers through ``peek`` only, so it requires the
        explicit ``supports_read_only_lookup`` capability on each, plus
        ``supports_removal_listener`` on the attention manager so a removal
        invalidates the memo (keyed by request, boundary, index version).
        """
        req_status = self._req_status.get(request.request_id)
        if (
            req_status is None
            or self._recurrent_manager is None
            or not self.recurrent_sidecar_ready
            or request.skip_reading_prefix_cache
            or not self._cold_reuse_enabled(req_status)
            or not self.manager.supports_read_only_lookup
            or not self.manager.supports_removal_listener
            or not self._recurrent_manager.supports_read_only_lookup
        ):
            return 0
        recurrent = self._recurrent_manager
        boundary = self._checkpoints.best_completed_boundary(
            request.request_id,
            request.num_prompt_tokens,
            len(request.block_hashes),
            lambda candidate: self._recurrent_sidecar_key(request, candidate),
            lambda key: recurrent.peek(key) is LookupResult.HIT,
        )
        if boundary and self._checkpoints.attention_ready(
            request.request_id,
            boundary,
            lambda: self._attention_hits_at(req_status, boundary),
        ):
            return boundary
        return 0

    def _attention_hits_at(
        self, req_status: RequestOffloadState, boundary: int
    ) -> bool:
        """Whether every group's last attention chunk before boundary is ready."""
        for group_config in self.config.kv_group_configs:
            if boundary % group_config.tokens_per_chunk:
                return False
            key = self._make_boundary_key(
                req_status.req, group_config.group_idx, boundary
            )
            if (
                key in self._pending_finished_store_key_owners
                or self.manager.peek(key) is not LookupResult.HIT
            ):
                return False
        return True

    def _cold_reuse_enabled(
        self, req_status: RequestOffloadState
    ) -> bool:
        """Whether this request has enough reusable prompt span for cold I/O."""
        max_prompt_span = req_status.req.num_prompt_tokens
        if req_status.max_offload_tokens is not None:
            max_prompt_span = min(
                max_prompt_span, req_status.max_offload_tokens
            )
        return (
            self._cold_reuse_min_tokens == 0
            or max_prompt_span >= self._cold_reuse_min_tokens
        )

    # TAIL_FUSE_V1 (TAIL-FUSE-PLAN S6): the core scheduler asks before it skips the floor128
    # split; notes live for one schedule() and are consumed or counted lost in build_connector_meta.
    def tail_fuse_accepts(self, request: Request, start: int, boundary: int) -> bool:
        """May this step run [start, end) past b = boundary and store b from the snapshot bank?"""
        if not _tail_fuse.enabled():
            return False
        req_id = request.request_id
        note = self._tail_fuse_notes.get(req_id)
        if note is not None:
            self._tail_fuse_notes[req_id] = (start, boundary, note[2])
            return True
        if not self.recurrent_sidecar_ready or not self.requires_exact_prefill_boundary_split(request):
            return False
        if len(self._tail_fuse_notes) >= _tail_fuse.max_rows():
            _tail_fuse.count("fb_rows")
            return False
        self._tail_fuse_notes[req_id] = (start, boundary, len(self._tail_fuse_notes))
        _tail_fuse.count("accepted")
        _tail_fuse.log_accept(req_id, start, boundary, request.num_prompt_tokens)
        return True

    def _tail_fuse_snapshot(
        self, req_status: RequestOffloadState, latest_computed: int
    ) -> tuple[int, int] | None:
        """(snapshot_index, b) for a noted request whose step ran strictly past b, b itself not capped.

        _calc_num_offloadable_tokens has no hash floor (it returns latest = P for the fused
        step), so the rule tests b directly: start < b < latest and _calc(req_status, b) == b.
        """
        notes = self._tail_fuse_notes
        note = notes.get(req_status.req.request_id) if notes else None
        if note is None:
            return None
        start, boundary, snapshot_index = note
        if not start < boundary < latest_computed:
            return None
        if self._calc_num_offloadable_tokens(req_status, boundary) != boundary:
            return None
        return snapshot_index, boundary

    def _tail_fuse_job_boundary(
        self, req_status: RequestOffloadState, latest_computed: int
    ) -> tuple[int | None, int | None, int | None]:
        """(boundary, snapshot_index, snapshot_boundary) of this step's recurrent store job.

        A fused step stores b from the snapshot bank: (b, index, b). Every other step keeps
        the original rule, the live row only on an exact uncapped boundary: (boundary, None,
        None); (None, None, None) when this step must not attach a row at all.
        """
        snapshot = self._tail_fuse_snapshot(req_status, latest_computed)
        if snapshot is not None:
            return snapshot[1], snapshot[0], snapshot[1]
        boundary = self._calc_num_offloadable_tokens(req_status, latest_computed)
        # Never attach a current row to an older/capped prefix hash.
        if boundary != latest_computed:
            return None, None, None
        return boundary, None, None

    def _tail_fuse_settle(self, req_id: ReqId, boundary: int) -> bool:
        """Drop the note for this request's b; False when it holds no note for this boundary."""
        note = self._tail_fuse_notes.get(req_id)
        if note is None or note[1] != boundary:
            return False
        del self._tail_fuse_notes[req_id]
        return True

    def _tail_fuse_stored(
        self, req_id: ReqId, boundary: int, snapshot_index: int | None, key: OffloadKey
    ) -> None:
        """A job for b's key was issued: from the snapshot bank, or (index None) by the original
        live-row path because the step ended exactly on b. Either way the note is not lost."""
        if not self._tail_fuse_settle(req_id, boundary):
            return
        _tail_fuse.count("stored")
        if snapshot_index is not None:
            _tail_fuse.log_store(req_id, boundary, snapshot_index, key)

    def _tail_fuse_dup(self, req_id: ReqId, boundary: int) -> None:
        """b's key is already in the CPU store: nothing to snapshot, and not lost."""
        if self._tail_fuse_settle(req_id, boundary):
            _tail_fuse.count("dup")
            _tail_fuse.log_dup(req_id, boundary)

    def _tail_fuse_end_step(self) -> None:
        """Unconsumed notes (not scheduled, b not strictly inside, no CPU block) are lost.

        Every step then offers the STATS line (once per 60 s), so it prints on the success
        path too, where every note was stored and none is left to lose.
        """
        if not _tail_fuse.enabled():
            return
        for req_id, (start, boundary, snapshot_index) in self._tail_fuse_notes.items():
            _tail_fuse.count("lost")
            logger.warning(
                "GLM53_TAIL_FUSE_LOST req=%s start=%d b=%d snap=%d: the fused step stored no "
                "checkpoint at b; the next turn recomputes from the previous checkpoint",
                req_id, start, boundary, snapshot_index,
            )
        self._tail_fuse_notes.clear()
        _tail_fuse.maybe_log_stats()

    def requires_exact_prefill_boundary_split(self, request: Request) -> bool:
        req_status = self._req_status.get(request.request_id)
        return bool(
            self._recurrent_manager is not None
            and req_status is not None
            and self._cold_reuse_enabled(req_status)
        )

    def _maybe_observe_lookup_async_delay(
        self, req_status: RequestOffloadState
    ) -> None:
        start_time = req_status.deferred_lookup_start_time
        if start_time is None:
            return
        req_status.deferred_lookup_start_time = None
        self._connector_stats.observe_histogram(
            _ConnectorMetricName.LOOKUP_ASYNC_DELAY,
            time.monotonic() - start_time,
        )

    def _generate_job_id(self) -> int:
        job_id = self._job_counter
        self._job_counter += 1
        return job_id

    def _recurrent_sidecar_key(
        self, req: Request, boundary: int
    ) -> OffloadKey | None:
        if boundary <= 0 or boundary % self.config.tokens_per_hash:
            return None
        hash_idx = boundary // self.config.tokens_per_hash - 1
        if hash_idx < 0 or hash_idx >= len(req.block_hashes):
            return None
        # 0xffffffff is outside the projected KV group index domain.
        return make_offload_key(req.block_hashes[hash_idx], 0xFFFFFFFF)

    def _is_sidecar_walk_boundary(self, boundary: int, initial_boundary: int) -> bool:
        """The sidecar walk checks the deepest attention hit and chunk-aligned boundaries."""
        return boundary == initial_boundary or all(
            boundary % group_config.tokens_per_chunk == 0
            for group_config in self.config.kv_group_configs
        )

    def _use_sidecar_hit(
        self,
        req_status: RequestOffloadState,
        key: OffloadKey,
        boundary: int,
        initial_boundary: int,
    ) -> None:
        """Touch the snapshot the walk stopped at and account for any tokens it backed off."""
        self._recurrent_manager.touch((key,), req_status.req_context)
        self._checkpoints.on_sidecar_hit(key)
        if boundary == initial_boundary:
            return
        self._checkpoints.observe_backtrack(
            req_status.req.request_id, initial_boundary, boundary
        )
        req_status.partial_tail_boundary = None
        self._connector_stats.increase_counter(
            "vllm:kv_offload_recurrent_boundary_backtrack_tokens",
            initial_boundary - boundary,
        )

    def _require_ready_recurrent_sidecar(
        self,
        req_status: RequestOffloadState,
        num_hit_tokens: int | None,
    ) -> int | None:
        if not num_hit_tokens or self._recurrent_manager is None:
            return num_hit_tokens
        if not self.recurrent_sidecar_ready:
            req_status.partial_tail_boundary = None
            return 0
        local_boundary = req_status.num_locally_computed_tokens
        initial_boundary = local_boundary + num_hit_tokens
        boundary = initial_boundary
        while boundary > local_boundary:
            if self._is_sidecar_walk_boundary(boundary, initial_boundary):
                key = self._recurrent_sidecar_key(req_status.req, boundary)
                if key is None:
                    req_status.partial_tail_boundary = None
                    return 0
                result = self._recurrent_manager.lookup(
                    key, req_status.req_context
                )
                if result is LookupResult.HIT:
                    self._use_sidecar_hit(req_status, key, boundary, initial_boundary)
                    return boundary - local_boundary
                if result in (LookupResult.HIT_PENDING, LookupResult.RETRY):
                    return None
            boundary -= self.config.tokens_per_hash
        self._checkpoints.observe_backtrack(
            req_status.req.request_id, initial_boundary, local_boundary
        )
        req_status.partial_tail_boundary = None
        self._connector_stats.increase_counter(
            "vllm:kv_offload_recurrent_boundary_backtrack_tokens",
            initial_boundary - local_boundary,
        )
        return 0

    def _remove_pending_job(self, job_id: int, block_ids: list[int] | None) -> None:
        for bid in block_ids or ():
            pending = self._block_id_to_pending_jobs[bid]
            pending.remove(job_id)
            if not pending:
                del self._block_id_to_pending_jobs[bid]

    def _calc_num_offloadable_tokens(
        self, req_status: RequestOffloadState, num_computed_tokens: int
    ) -> int:
        num = min(num_computed_tokens, req_status.req.num_tokens)
        max_offload_tokens = req_status.max_offload_tokens
        if max_offload_tokens is not None:
            num = min(num, max_offload_tokens)
        if self.config.offload_prompt_only:
            num = min(num, req_status.req.num_prompt_tokens)
        return num

    def _track_pending_finished_store_keys(
        self, req_id: ReqId, keys: Iterable[OffloadKey]
    ) -> None:
        keys = set(keys)
        if not keys:
            return
        req_keys = self._pending_finished_store_keys.setdefault(req_id, set())
        keys.difference_update(req_keys)
        req_keys.update(keys)
        for key in keys:
            self._pending_finished_store_key_owners.setdefault(key, set()).add(
                req_id
            )

    def _release_pending_finished_store_keys(
        self, req_id: ReqId, keys: Iterable[OffloadKey] | None = None
    ) -> None:
        req_keys = self._pending_finished_store_keys.get(req_id)
        if not req_keys:
            return
        released = set(req_keys if keys is None else req_keys.intersection(keys))
        for key in released:
            owners = self._pending_finished_store_key_owners[key]
            owners.remove(req_id)
            if not owners:
                del self._pending_finished_store_key_owners[key]
        req_keys.difference_update(released)
        if not req_keys:
            del self._pending_finished_store_keys[req_id]

    def _lookup_with_finished_store_dependency(
        self, key: OffloadKey, req_context: ReqContext
    ) -> tuple[LookupResult, bool]:
        if key in self._pending_finished_store_key_owners:
            # Avoid a manager MISS before prepare_store().  Some managers use
            # lookup for admission accounting, so probing here is not neutral.
            return LookupResult.HIT_PENDING, True
        return self.manager.lookup(key, req_context), False

    def _new_store_keys(
        self,
        req_status: RequestOffloadState,
        num_offloadable_tokens: int,
    ) -> list[OffloadKey]:
        blocks_per_chunk = self.config.blocks_per_chunk
        new_offload_keys: list[OffloadKey] = []
        for group_config, group_state in zip(
            self.config.kv_group_configs, req_status.group_states
        ):
            num_chunks = req_status.storable_chunks(
                group_config, group_state, num_offloadable_tokens
            )
            start_chunk_idx = group_state.next_stored_chunk_idx
            if num_chunks <= start_chunk_idx:
                continue
            offload_keys = group_state.offload_keys[start_chunk_idx:num_chunks]
            offload_block_ids = group_state.block_ids[
                start_chunk_idx * blocks_per_chunk
                + blocks_per_chunk
                - 1 : num_chunks * blocks_per_chunk : blocks_per_chunk
            ]
            assert len(offload_keys) == len(offload_block_ids)
            for key_idx, (offload_key, block_id) in enumerate(
                zip(offload_keys, offload_block_ids)
            ):
                if block_id == 0:
                    continue
                abs_chunk_idx = start_chunk_idx + key_idx
                if not is_store_reachable_swa_chunk(
                    abs_chunk_idx,
                    num_chunks,
                    group_config.alignment_chunk_count,
                    group_config.sliding_window_size_in_chunks,
                    group_config.is_eagle_group,
                ):
                    continue
                new_offload_keys.append(offload_key)
        return new_offload_keys

    def _maximal_prefix_lookup(
        self,
        keys: Iterable[OffloadKey],
        req_context: ReqContext,
        req: Request,
        group_config: GroupOffloadConfig,
        start_chunk_idx: int,
    ) -> int | None:
        """Return the number of consecutive offloaded chunks from the start.

        Exact finished-store dependencies may fall back to the ready prefix
        when their dependent reusable span is below the configured threshold.
        Other backend deferrals preserve the existing retry contract.
        """
        hit_count = 0
        first_dependency_hit_count: int | None = None
        has_other_defer = False
        # Managers with lookup-frequency admission (for example
        # ``store_threshold >= 2``) must observe the entire candidate span.
        # Stopping at the first miss only increments the first unseen key, so
        # a repeated long prefix advances by one group/chunk per request and
        # never becomes practically reusable.  Continue the read-only lookup
        # census after the first miss, but never treat suffix hits as a usable
        # prefix and never publish lookup events for them.
        observe_miss_suffix = isinstance(
            getattr(self.manager, "counts", None), MutableMapping
        )
        prefix_open = True
        for local_idx, key in enumerate(keys):
            result, is_finished_store_dependency = (
                self._lookup_with_finished_store_dependency(key, req_context)
            )
            if not prefix_open:
                continue
            match result:
                case LookupResult.HIT:
                    # Hits after a pending dependency are not part of the
                    # fallback prefix, so publish their events only on retry.
                    if first_dependency_hit_count is None:
                        self._events_tracker.record_lookup(
                            req,
                            group_config,
                            start_chunk_idx + local_idx,
                            key,
                        )
                    hit_count += 1
                case LookupResult.HIT_PENDING:
                    if is_finished_store_dependency:
                        if first_dependency_hit_count is None:
                            first_dependency_hit_count = hit_count
                    else:
                        has_other_defer = True
                    hit_count += 1
                case LookupResult.RETRY:
                    # Unknown backend readiness must retain the conservative
                    # retry behavior even if another exact dependency is short.
                    has_other_defer = True
                case LookupResult.MISS:
                    prefix_open = False
                    if not observe_miss_suffix:
                        break
        is_wait_candidate = (
            first_dependency_hit_count is not None
            and (hit_count - first_dependency_hit_count)
            * group_config.tokens_per_chunk
            >= self._pending_store_wait_threshold
        )
        dependency_wait_elapsed_s = 0.0
        if is_wait_candidate:
            req_status = self._req_status[req.request_id]
            req_status.pending_store_wait_deferred = True
            now = time.monotonic()
            wait_start = req_status.pending_store_wait_start_time
            if wait_start is None:
                wait_start = now
                req_status.pending_store_wait_start_time = wait_start
            dependency_wait_elapsed_s = max(0.0, now - wait_start)

        return _resolve_finished_store_prefix(
            hit_count,
            first_dependency_hit_count,
            has_other_defer,
            group_config.tokens_per_chunk,
            self._pending_store_wait_threshold,
            dependency_wait_elapsed_s,
            self._pending_store_wait_deadline_s,
        )

    def _observe_remaining_lookup_groups(
        self,
        req_status: RequestOffloadState,
        group_indices: Iterable[int],
        num_computed_tokens: int,
        max_hit_size_tokens: int,
    ) -> None:
        """Account for unseen groups after an unusable prefix miss.

        This is inert for managers without lookup-frequency admission.  It
        intentionally ignores hit/miss results: the request must still compute
        locally, while a later identical request can satisfy the configured
        reuse threshold across the complete projected attention span.
        """
        if not isinstance(getattr(self.manager, "counts", None), MutableMapping):
            return
        for group_idx in group_indices:
            group_config = self.config.kv_group_configs[group_idx]
            group_state = req_status.group_states[group_idx]
            tokens_per_chunk = group_config.tokens_per_chunk
            start_chunk_idx = num_computed_tokens // tokens_per_chunk
            num_chunks = min(
                cdiv(max_hit_size_tokens, tokens_per_chunk),
                len(group_state.offload_keys),
            )
            for key in group_state.offload_keys[start_chunk_idx:num_chunks]:
                self._lookup_with_finished_store_dependency(
                    key, req_status.req_context
                )

    def _sliding_window_lookup(
        self,
        keys: Sequence[OffloadKey],
        sliding_window_size: int,
        req_context: ReqContext,
    ) -> int | None:
        """Return the end index (in `keys`) of the last run of
        `sliding_window_size` consecutive hits, scanning from the end.
        Returns 0 on miss, None if the backend deferred a lookup."""
        defer_lookup = False
        consecutive_hits = 0
        for idx in range(len(keys) - 1, -1, -1):
            match self._lookup_with_finished_store_dependency(
                keys[idx], req_context
            )[0]:
                case LookupResult.HIT:
                    consecutive_hits += 1
                case LookupResult.HIT_PENDING:
                    # Block is in cache, just not readable yet — counts
                    # as hit for the consecutive streak. Don't break:
                    # keep scanning to let manager kick off async lookups.
                    defer_lookup = True
                    consecutive_hits += 1
                case LookupResult.RETRY:
                    # Block location uncertain — does not count as hit.
                    # Don't break: keep scanning to let manager kick off
                    # async lookups.
                    defer_lookup = True
                    consecutive_hits = 0
                case LookupResult.MISS:
                    consecutive_hits = 0
            if consecutive_hits == sliding_window_size:
                return idx + sliding_window_size if not defer_lookup else None
        return consecutive_hits if not defer_lookup else None

    def _touch(self, req_status: RequestOffloadState):
        for group_config, group_state in zip(
            self.config.kv_group_configs, req_status.group_states
        ):
            if group_config.sliding_window_size_in_chunks is None:
                self.manager.touch(group_state.offload_keys, req_status.req_context)
            else:
                # Keep only chunks needed to hit the original request, plus
                # decoded chunks.
                chunks_to_skip = max(
                    0,
                    group_state.num_hit_chunks
                    - group_config.sliding_window_size_in_chunks,
                )
                self.manager.touch(
                    group_state.offload_keys[chunks_to_skip:],
                    req_status.req_context,
                )
        if req_status.partial_tail_boundary is not None:
            self.manager.touch(
                tuple(
                    self._make_boundary_key(
                        req_status.req,
                        group.group_idx,
                        req_status.partial_tail_boundary,
                    )
                    for group in self.config.kv_group_configs
                ),
                req_status.req_context,
            )

    def _lookup_complete_chunks(self, req_status: RequestOffloadState) -> int | None:
        """
        Find how many tokens beyond num_locally_computed_tokens can be loaded.

        Iterates full-attention groups first (prefix lookup), then sliding-window
        groups (suffix lookup). Each group may tighten max_hit_size_tokens, which
        can invalidate an earlier group's result, so the loop re-runs when that
        happens until num_hit_tokens converges.
        """
        num_computed_tokens = req_status.num_locally_computed_tokens
        max_hit_size_tokens: int = req_status.req.num_tokens
        if self._sliding_window_groups:
            # the last prompt token has to be recomputed to get the logprobs
            # for sliding window attention, we must reduce by 1 to make sure
            # we still have a hit after reduction
            max_hit_size_tokens -= 1
            if self._mamba_align_size is not None:
                # Constrain hit-window to the mamba block size.
                max_hit_size_tokens = round_down(
                    max_hit_size_tokens, self._mamba_align_size
                )

        num_hit_tokens: int = 0
        defer_lookup = False
        lookup_groups = self._lookup_groups

        # Tracks which eagle groups have already popped their volatile trailing chunk
        # in the current convergence iteration. Reset when a non-eagle group
        # tightens the hit boundary, requiring a fresh pop.
        eagle_verified: set[int] = set()
        while lookup_groups:
            looked_up_sliding_window: bool = False
            groups_iter = iter(lookup_groups)
            lookup_groups = ()
            for group_idx in groups_iter:
                group_config: GroupOffloadConfig = self.config.kv_group_configs[
                    group_idx
                ]
                group_state: RequestGroupState = req_status.group_states[group_idx]
                tokens_per_chunk = group_config.tokens_per_chunk
                offload_keys = group_state.offload_keys

                assert (
                    len(offload_keys) >= req_status.req.num_tokens // tokens_per_chunk
                )

                is_eagle_unverified = (
                    group_config.is_eagle_group and group_idx not in eagle_verified
                )

                # Constrain to a chunk-aligned boundary for this group.
                max_hit_size_tokens = min(
                    max_hit_size_tokens, len(offload_keys) * tokens_per_chunk
                )
                if max_hit_size_tokens - num_computed_tokens < tokens_per_chunk:
                    # We can only load less than a chunk, so skip.
                    return 0

                sliding_window_size_in_chunks = (
                    group_config.sliding_window_size_in_chunks
                )

                # For eagle groups, query one extra chunk that will be popped.
                # We only need to increase the query size for sliding window groups.
                query_max = max_hit_size_tokens
                if is_eagle_unverified and sliding_window_size_in_chunks is not None:
                    query_max = min(
                        max_hit_size_tokens + tokens_per_chunk,
                        len(offload_keys) * tokens_per_chunk,
                    )

                num_chunks = min(cdiv(query_max, tokens_per_chunk), len(offload_keys))
                start_chunk_idx = num_computed_tokens // tokens_per_chunk
                offload_keys = offload_keys[start_chunk_idx:num_chunks]

                # end index (in the sliced offload_keys) up to which we
                # have backend-confirmed hits
                num_hit_chunks: int | None
                if sliding_window_size_in_chunks is None:
                    num_hit_chunks = self._maximal_prefix_lookup(
                        offload_keys,
                        req_status.req_context,
                        req_status.req,
                        group_config,
                        start_chunk_idx,
                    )
                else:
                    required_window = sliding_window_size_in_chunks
                    if is_eagle_unverified:
                        required_window += 1
                    num_hit_chunks = self._sliding_window_lookup(
                        offload_keys,
                        required_window,
                        req_status.req_context,
                    )
                if num_hit_chunks == 0:
                    self._observe_remaining_lookup_groups(
                        req_status,
                        groups_iter,
                        num_computed_tokens,
                        max_hit_size_tokens,
                    )
                    return 0

                if num_hit_chunks is None:
                    defer_lookup = True
                else:
                    if is_eagle_unverified:
                        num_hit_chunks -= 1
                        eagle_verified.add(group_idx)

                    max_hit_size_tokens = min(
                        max_hit_size_tokens,
                        tokens_per_chunk * (start_chunk_idx + num_hit_chunks),
                    )

                new_num_hit_tokens = max_hit_size_tokens - num_computed_tokens
                if new_num_hit_tokens < tokens_per_chunk:
                    # We can only load less than a chunk, so skip.
                    return 0

                if new_num_hit_tokens < num_hit_tokens:
                    if not group_config.is_eagle_group:
                        eagle_verified.clear()
                    if defer_lookup:
                        # make another iteration on all groups to check
                        # if we still need to defer lookup
                        defer_lookup = False
                        lookup_groups = self._lookup_groups
                    elif looked_up_sliding_window and not lookup_groups:
                        # we need another iteration to confirm previously looked up
                        # sliding window works with the new_num_hit_tokens
                        lookup_groups = self._sliding_window_groups

                looked_up_sliding_window |= sliding_window_size_in_chunks is not None
                num_hit_tokens = new_num_hit_tokens

        if defer_lookup:
            logger.debug(
                "Offloading manager delayed request %s as backend requested",
                req_status.req.request_id,
            )
            return None

        # Possibly delay the request if any hit chunk is already being loaded.
        if self._chunks_being_loaded:
            for group_config, group_state in zip(
                self.config.kv_group_configs, req_status.group_states
            ):
                tokens_per_chunk = group_config.tokens_per_chunk
                sliding_window_size_in_chunks = (
                    group_config.sliding_window_size_in_chunks
                )
                offload_keys = group_state.offload_keys
                num_chunks = cdiv(
                    num_computed_tokens + num_hit_tokens, tokens_per_chunk
                )
                start_chunk_idx = num_computed_tokens // tokens_per_chunk
                offload_keys = offload_keys[start_chunk_idx:num_chunks]
                if sliding_window_size_in_chunks is not None:
                    offload_keys = offload_keys[-sliding_window_size_in_chunks:]
                if any(key in self._chunks_being_loaded for key in offload_keys):
                    # Hit chunks are being loaded, so delay the request.
                    logger.debug(
                        "Delaying request %s since some of its"
                        " chunks are already being loaded",
                        req_status.req.request_id,
                    )
                    return None

        logger.debug(
            "Request %s hit %s offloaded tokens after %s GPU hit tokens",
            req_status.req.request_id,
            num_hit_tokens,
            num_computed_tokens,
        )

        return num_hit_tokens

    def _make_boundary_key(
        self, request: Request, group_idx: int, boundary_tokens: int
    ) -> OffloadKey:
        hash_idx = boundary_tokens // self.config.tokens_per_hash - 1
        return make_offload_key(request.block_hashes[hash_idx], group_idx)

    def _lookup(self, req_status: RequestOffloadState) -> int | None:
        complete_hit = self._lookup_complete_chunks(req_status)
        req_status.partial_tail_boundary = None
        if complete_hit is None or not self.config.supports_partial_tail:
            return complete_hit

        local_tokens = req_status.num_locally_computed_tokens
        complete_boundary = local_tokens + complete_hit
        tokens_per_hash = self.config.tokens_per_hash
        block_end = complete_boundary + self._partial_tail_block_size
        max_boundary = round_down(
            min(req_status.req.num_prompt_tokens - 1, block_end - 1), tokens_per_hash
        )
        if max_boundary <= complete_boundary:
            return complete_hit

        pending = False
        for boundary in range(max_boundary, complete_boundary, -tokens_per_hash):
            boundary_pending = False
            boundary_missed = False
            boundary_keys = []
            for group_config in self.config.kv_group_configs:
                key = self._make_boundary_key(
                    req_status.req, group_config.group_idx, boundary
                )
                boundary_keys.append(key)
                result = self._lookup_with_finished_store_dependency(
                    key, req_status.req_context
                )[0]
                if result is LookupResult.MISS:
                    boundary_missed = True
                    break
                if result in (LookupResult.HIT_PENDING, LookupResult.RETRY):
                    boundary_pending = True

            pending |= boundary_pending
            if not boundary_missed and not boundary_pending:
                for group_config, key in zip(
                    self.config.kv_group_configs, boundary_keys
                ):
                    self._events_tracker.record_partial_lookup(
                        req_status.req, group_config, boundary, key
                    )
                req_status.partial_tail_boundary = boundary
                return boundary - local_tokens

        if pending and complete_hit == 0:
            return None
        return complete_hit

    def on_new_request(self, request: Request) -> None:
        """Called when a new request is added to the scheduler."""
        req_context = _create_req_context(request)
        offloading_context = self.manager.on_new_request(req_context)
        req_status = RequestOffloadState(
            config=self.config,
            req=request,
            req_context=req_context,
            offloading_context=offloading_context,
        )
        self._req_status[request.request_id] = req_status
        if self._recurrent_manager is not None:
            self._recurrent_manager.on_new_request(req_context)
        self._checkpoints.on_new_request(
            request.request_id,
            request.block_hashes,
            request.num_prompt_tokens,
            self._computed_tokens_of,
        )

    def _lookup_ready_hit_tokens(self, req_status: RequestOffloadState) -> int | None:
        """Attention walk, then the snapshot gate; None defers the request."""
        lookup_start = time.monotonic()
        attention_hit = self._lookup(req_status)
        num_hit_tokens = self._require_ready_recurrent_sidecar(req_status, attention_hit)
        self._connector_stats.observe_histogram(
            _ConnectorMetricName.LOOKUP_SYNC_DELAY,
            time.monotonic() - lookup_start,
        )
        if num_hit_tokens is None:
            if req_status.deferred_lookup_start_time is None:
                req_status.deferred_lookup_start_time = lookup_start
            return None
        self._attribute_miss(
            req_status,
            attention_hit,
            num_hit_tokens,
            _miss_attrib.REASON_LOOKUP,
            lookup_start,
        )
        self._maybe_observe_lookup_async_delay(req_status)
        return num_hit_tokens

    def _match_without_cold_reuse(
        self, req_status: RequestOffloadState, num_computed_tokens: int
    ) -> tuple[int | None, bool]:
        req_status.num_locally_computed_tokens = num_computed_tokens
        req_status.update_num_hit_chunks(num_computed_tokens)
        self._attribute_miss(req_status, 0, 0, _miss_attrib.REASON_COLD)
        return 0, False

    def _finish_match(
        self,
        req_status: RequestOffloadState,
        num_computed_tokens: int,
        num_hit_tokens: int | None,
    ) -> tuple[int | None, bool]:
        if (
            num_hit_tokens is not None
            or not req_status.pending_store_wait_deferred
        ):
            req_status.pending_store_wait_start_time = None
            req_status.pending_store_wait_deferred = False
        req_status.update_num_hit_chunks(num_computed_tokens + (num_hit_tokens or 0))

        self._touch(req_status)

        return num_hit_tokens, bool(num_hit_tokens)

    def get_num_new_matched_tokens(
        self, request: Request, num_computed_tokens: int
    ) -> tuple[int | None, bool]:
        """
        Get number of new tokens that can be loaded beyond the
        num_computed_tokens.

        Args:
            request (Request): the request object.
            num_computed_tokens (int): the number of locally
                computed tokens for this request

        Returns:
            A tuple with the following elements:
                - The number of tokens that can be loaded beyond what is
                  already computed.
                  If None, it means that the connector needs more time to
                  determine the number of matched tokens, and the scheduler
                  should query for this request again later.
                - `True` if tokens will be loaded asynchronously
                  (between scheduler steps).
        """
        req_status = self._req_status[request.request_id]
        for group_state in req_status.group_states:
            group_state.block_ids.clear()

        if not self._cold_reuse_enabled(req_status):
            return self._match_without_cold_reuse(req_status, num_computed_tokens)

        if req_status.transfer_jobs:
            logger.debug(
                "Delaying request %s since it still has in-flight transfers",
                request.request_id,
            )
            return None, False

        req_status.update_offload_keys()
        req_status.num_locally_computed_tokens = num_computed_tokens

        req_status.pending_store_wait_deferred = False
        num_hit_tokens: int | None
        if request.skip_reading_prefix_cache:
            num_hit_tokens = 0
            self._attribute_miss(req_status, 0, 0, _miss_attrib.REASON_SKIP_READ)
        else:
            num_hit_tokens = self._lookup_ready_hit_tokens(req_status)
        return self._finish_match(req_status, num_computed_tokens, num_hit_tokens)

    def update_state_after_alloc(
        self, request: Request, blocks: KVCacheBlocks, num_external_tokens: int
    ):
        if num_external_tokens == 0:
            return

        req_status = self._req_status[request.request_id]

        num_locally_computed_tokens = req_status.num_locally_computed_tokens
        num_cached_tokens = num_locally_computed_tokens + num_external_tokens
        partial_tail_boundary = req_status.partial_tail_boundary
        if partial_tail_boundary is not None:
            assert partial_tail_boundary == num_cached_tokens

        keys_to_load: list[OffloadKey] = []
        dst_block_ids: list[int] = []
        # per group
        group_sizes: list[int] = []
        block_indices: list[int] = []
        for group_config, group_state, group_blocks in zip(
            self.config.kv_group_configs,
            req_status.group_states,
            blocks.blocks,
        ):
            self._current_batch_allocated_block_ids.update(
                block.block_id for block in group_blocks if block.block_id != 0
            )

            tokens_per_block = group_config.tokens_per_block
            tokens_per_chunk = group_config.tokens_per_chunk
            offload_keys = group_state.offload_keys
            num_gpu_blocks = cdiv(num_cached_tokens, tokens_per_block)

            assert len(group_blocks) >= num_gpu_blocks
            num_locally_computed_gpu_blocks = num_gpu_blocks
            # Skip null placeholder blocks (used for sliding window or mamba padding).
            for i, block in enumerate(group_blocks[:num_gpu_blocks]):
                if not block.is_null and block.block_hash is None:
                    num_locally_computed_gpu_blocks = i
                    break

            assert (
                num_locally_computed_tokens
                <= num_locally_computed_gpu_blocks * tokens_per_block
            )
            num_pending_gpu_blocks = num_gpu_blocks - num_locally_computed_gpu_blocks

            if group_config.sliding_window_size_in_chunks is not None:
                assert (
                    num_pending_gpu_blocks
                    <= group_config.sliding_window_size_in_chunks
                    * self.config.blocks_per_chunk
                    + 1
                )

            num_chunks = cdiv(num_cached_tokens, tokens_per_chunk)
            if num_pending_gpu_blocks:
                start_chunk_idx = (
                    num_locally_computed_gpu_blocks // self.config.blocks_per_chunk
                )
                end_chunk_idx = num_chunks - (partial_tail_boundary is not None)
                assert len(offload_keys) >= end_chunk_idx
                keys_to_load.extend(offload_keys[start_chunk_idx:end_chunk_idx])
                if partial_tail_boundary is not None:
                    keys_to_load.append(
                        self._make_boundary_key(
                            request, group_config.group_idx, partial_tail_boundary
                        )
                    )

            dst_block_ids.extend(
                block.block_id
                for block in group_blocks[
                    num_locally_computed_gpu_blocks:num_gpu_blocks
                ]
            )
            group_sizes.append(num_pending_gpu_blocks)
            block_indices.append(num_locally_computed_gpu_blocks)

            # Skip prefix-hit chunks for block-level policy; for
            # request-level, next_stored_chunk_idx stays at 0 so all
            # chunks (including hits) are offloaded.
            if req_status.offloading_context.policy == OffloadPolicy.BLOCK_LEVEL:
                group_state.next_stored_chunk_idx = num_chunks

        src_spec = self.manager.prepare_load(keys_to_load, req_status.req_context)
        recurrent_sidecar = None
        recurrent_sidecar_key = None
        if self._recurrent_manager is not None:
            recurrent_sidecar_key = self._recurrent_sidecar_key(
                request, num_cached_tokens
            )
            if recurrent_sidecar_key is None:
                raise RuntimeError("external recurrent hit is not hash-aligned")
            sidecar_src = self._recurrent_manager.prepare_load(
                (recurrent_sidecar_key,), req_status.req_context
            )
            assert isinstance(sidecar_src, CPULoadStoreSpec)
            assert sidecar_src.block_ids.size == 1
            recurrent_sidecar = RecurrentSidecarTransfer(
                key=recurrent_sidecar_key,
                cpu_block_id=int(sidecar_src.block_ids[0]),
            )
        dst_spec = GPULoadStoreSpec(
            dst_block_ids, group_sizes=group_sizes, block_indices=block_indices
        )

        load_job_id = self._generate_job_id()
        self._current_batch_load_jobs[load_job_id] = TransferJob(
            req_id=request.request_id,
            src_spec=src_spec,
            dst_spec=dst_spec,
            recurrent_sidecar=recurrent_sidecar,
        )
        # a load can only be issued when no other jobs are pending.
        assert not req_status.transfer_jobs
        req_status.transfer_jobs.add(load_job_id)
        self._jobs[load_job_id] = TransferJobStatus(
            req_id=request.request_id,
            pending_count=self.config.num_workers,
            keys=set(keys_to_load),
            is_store=False,
            recurrent_sidecar_key=recurrent_sidecar_key,
        )

        if self._chunks_being_loaded is not None:
            self._chunks_being_loaded.update(keys_to_load)
        req_status.partial_tail_boundary = None

    def _update_req_states(self, scheduler_output: SchedulerOutput) -> None:
        """
        Update request states from the Scheduler's output.
        """

        # new_block_ids_end[req_id][i] = end of pre-existing block_ids for
        # the i-th sliding window group (before this step's extend).
        # Used to detect sliding window blocks that got re-allocated.
        new_block_ids_end: dict[str, tuple[int, ...]] = {}

        for req_id, new_block_id_groups, preempted in yield_req_data(scheduler_output):
            req_status = self._req_status[req_id]
            req_status.update_offload_keys()

            if preempted:
                for group_state in req_status.group_states:
                    group_state.block_ids.clear()

            if new_block_id_groups:
                if self._sliding_window_groups:
                    new_block_ids_end[req_id] = tuple(
                        len(req_status.group_states[grp_idx].block_ids)
                        for grp_idx in self._sliding_window_groups
                    )
                req_status.update_block_id_groups(new_block_id_groups)
                for new_blocks in new_block_id_groups:
                    for bid in new_blocks:
                        if bid != 0:
                            self._current_batch_allocated_block_ids.add(bid)

        # Zero out stale block_ids in sliding window groups' pending-store
        # positions. Only sliding window groups can have stale entries (blocks
        # freed by remove_skipped_blocks then reallocated). Only positions in
        # [next_stored_chunk_idx * bsf, end) need checking where end is the
        # pre-extend length: earlier positions were already offloaded, later
        # ones are fresh allocations from this step.
        if self._sliding_window_groups and self._current_batch_allocated_block_ids:
            blocks_per_chunk = self.config.blocks_per_chunk
            for req_id, req_status in self._req_status.items():
                ends = new_block_ids_end.get(req_id)
                for i, grp_idx in enumerate(self._sliding_window_groups):
                    group_state = req_status.group_states[grp_idx]
                    start = group_state.next_stored_chunk_idx * blocks_per_chunk
                    end = ends[i] if ends is not None else len(group_state.block_ids)
                    for j in range(start, end):
                        if (
                            group_state.block_ids[j]
                            in self._current_batch_allocated_block_ids
                        ):
                            group_state.block_ids[j] = 0
        self._prune_checkpoints(scheduler_output)

    def _prune_checkpoints(self, scheduler_output: SchedulerOutput) -> None:
        for req_id in scheduler_output.num_scheduled_tokens:
            req_status = self._req_status.get(req_id)
            if req_status is not None:
                self._checkpoints.prune(req_id, req_status.req.num_computed_tokens)

    def _build_partial_tail_store_jobs(
        self, scheduler_output: SchedulerOutput
    ) -> dict[int, TransferJob]:
        handoffs = scheduler_output.partial_tail_offloads
        if not self.config.supports_partial_tail or not handoffs:
            return {}

        store_jobs: dict[int, TransferJob] = {}
        for req_id, entries in handoffs.items():
            req_status = self._req_status.get(req_id)
            assert req_status is not None
            if not self._cold_reuse_enabled(req_status):
                continue
            # The connector contract permits either one load or one-or-more
            # stores for a request, never both.  A load created during this
            # scheduler batch has not reached worker completion yet, so defer
            # the transient partial-tail store instead of mixing directions.
            if any(
                not self._jobs[job_id].is_store
                for job_id in req_status.transfer_jobs
            ):
                logger.debug(
                    "Deferring partial-tail store for request %s while a load "
                    "is pending",
                    req_id,
                )
                continue
            assert entries
            boundaries = {boundary for _, _, boundary in entries}
            assert len(boundaries) == 1
            boundary = boundaries.pop()
            req = req_status.req
            max_boundary = min(
                req.num_prompt_tokens,
                req_status.max_offload_tokens or req.num_prompt_tokens,
            )
            assert boundary > 0
            assert boundary % self.config.tokens_per_hash == 0
            assert boundary <= max_boundary

            cow_blocks = {group_idx: block_id for group_idx, block_id, _ in entries}
            assert self._cow_source_groups.issubset(cow_blocks)

            assert boundary % self._partial_tail_block_size != 0
            block_idx = boundary // self._partial_tail_block_size
            if any(
                group.group_idx not in self._cow_source_groups
                and block_idx >= len(req_status.group_states[group.group_idx].block_ids)
                for group in self.config.kv_group_configs
            ):
                continue
            keys = [
                self._make_boundary_key(req, group.group_idx, boundary)
                for group in self.config.kv_group_configs
            ]
            block_ids = [
                cow_blocks[group.group_idx]
                if group.group_idx in self._cow_source_groups
                else req_status.group_states[group.group_idx].block_ids[block_idx]
                for group in self.config.kv_group_configs
            ]
            assert all(block_id != 0 for block_id in block_ids)

            store_output = self.manager.prepare_store(keys, req_status.req_context)
            if store_output is None:
                self._connector_stats.increase_counter(
                    _ConnectorMetricName.ALLOCATION_FAILURE
                )
                continue
            if not store_output.keys_to_store:
                self._current_batch_attention_ready_req_ids.add(req_id)
                continue

            for group_config, key in zip(self.config.kv_group_configs, keys):
                if key in store_output.keys_to_store:
                    self._events_tracker.record_partial_store(
                        req, group_config, boundary, key
                    )

            group_by_key = {key: idx for idx, key in enumerate(keys)}
            accepted_groups = [group_by_key[key] for key in store_output.keys_to_store]
            group_sizes = [0] * len(self.config.kv_group_configs)
            block_indices = [0] * len(self.config.kv_group_configs)
            for group_idx in accepted_groups:
                group_sizes[group_idx] = 1
                block_indices[group_idx] = block_idx
            source_blocks = [block_ids[group_idx] for group_idx in accepted_groups]

            job_id = self._generate_job_id()
            req_status.transfer_jobs.add(job_id)
            for block_id in source_blocks:
                self._block_id_to_pending_jobs.setdefault(block_id, set()).add(job_id)
            self._jobs[job_id] = TransferJobStatus(
                req_id=req_id,
                pending_count=self.config.num_workers,
                keys=set(store_output.keys_to_store),
                is_store=True,
                fenced_block_ids=source_blocks,
            )
            store_jobs[job_id] = TransferJob(
                req_id=req_id,
                src_spec=GPULoadStoreSpec(
                    source_blocks,
                    group_sizes=group_sizes,
                    block_indices=block_indices,
                ),
                dst_spec=store_output.store_spec,
            )

        return store_jobs

    def _build_recurrent_sidecar_store_jobs(
        self,
        scheduler_output: SchedulerOutput,
        attention_store_req_ids: set[ReqId],
    ) -> dict[int, TransferJob]:
        """Store only the state at this step's latest exact hash boundary."""
        if self._recurrent_manager is None or not self.recurrent_sidecar_ready:
            return {}
        jobs: dict[int, TransferJob] = {}
        # Only scheduled requests run this step and retain a post-forward row.
        # finished_req_ids were already removed/freed by runner._update_states.
        for req_id in scheduler_output.num_scheduled_tokens:
            if req_id not in attention_store_req_ids:
                continue
            req_status = self._req_status.get(req_id)
            if req_status is None:
                continue
            if not self._cold_reuse_enabled(req_status):
                continue
            if any(
                not self._jobs[job_id].is_store
                for job_id in req_status.transfer_jobs
            ):
                continue
            req = req_status.req
            if req.is_finished():
                continue
            latest_computed = (
                req.num_computed_tokens
                + scheduler_output.num_scheduled_tokens[req_id]
            )
            # TAIL_FUSE_V1: a fused step stores b from the snapshot bank; every other step
            # keeps the original rule (never attach a current row to an older/capped hash).
            boundary, snapshot_index, snapshot_boundary = self._tail_fuse_job_boundary(
                req_status, latest_computed
            )
            if boundary is None:
                continue
            key = self._recurrent_sidecar_key(req, boundary)
            if key is None:
                continue
            output = self._recurrent_manager.prepare_store(
                (key,), req_status.req_context
            )
            if output is not None and output.evicted_keys:
                self._connector_stats.increase_counter(
                    "vllm:kv_offload_recurrent_cache_evictions",
                    len(output.evicted_keys),
                )
            if output is None:
                self._connector_stats.increase_counter(
                    _ConnectorMetricName.ALLOCATION_FAILURE
                )
                continue
            if not output.keys_to_store:
                self._tail_fuse_dup(req_id, boundary)
                continue
            assert output.keys_to_store == [key]
            assert isinstance(output.store_spec, CPULoadStoreSpec)
            assert output.store_spec.block_ids.size == 1
            transfer = RecurrentSidecarTransfer(
                key=key,
                cpu_block_id=int(output.store_spec.block_ids[0]),
                snapshot_index=snapshot_index,
                snapshot_boundary=snapshot_boundary,
            )
            job_id = self._generate_job_id()
            req_status.transfer_jobs.add(job_id)
            self._jobs[job_id] = TransferJobStatus(
                req_id=req_id,
                pending_count=self.config.num_workers,
                keys=set(),
                is_store=True,
                recurrent_sidecar_key=key,
                attention_transfer=False,
            )
            jobs[job_id] = TransferJob(
                req_id=req_id,
                src_spec=None,
                dst_spec=None,
                attention_transfer=False,
                recurrent_sidecar=transfer,
            )
            self._checkpoints.on_sidecar_store_issued(req_id, boundary, key)
            self._tail_fuse_stored(req_id, boundary, snapshot_index, key)
        return jobs

    def _is_previous_hash_checkpoint(
        self, req_status: RequestOffloadState, num_offloadable_tokens: int,
        latest_computed: int,
    ) -> bool:
        """Flush attention plus the current row only at an exact reusable boundary."""
        if (
            os.environ.get("VLLM_RECURRENT_OFFLOAD_PREVIOUS_HASH_CHECKPOINT", "0") != "1"
            or self._recurrent_manager is None
            or not self.recurrent_sidecar_ready
            or num_offloadable_tokens <= 0
            or num_offloadable_tokens != latest_computed
        ):
            return False
        tokens_per_hash = self.config.tokens_per_hash
        previous = (
            req_status.req.num_prompt_tokens // tokens_per_hash - 1
        ) * tokens_per_hash
        return bool(
            num_offloadable_tokens == previous
            and all(
                previous % group.tokens_per_chunk == 0
                for group in self.config.kv_group_configs
            )
        )

    def _is_forced_checkpoint(
        self, req_status: RequestOffloadState, num_offloadable_tokens: int,
        latest_computed: int,
    ) -> bool:
        """V1 previous-hash checkpoint, or a planned V2 divergence checkpoint."""
        if self._is_previous_hash_checkpoint(
            req_status, num_offloadable_tokens, latest_computed
        ):
            return True
        return bool(
            self._recurrent_manager is not None
            and self.recurrent_sidecar_ready
            and num_offloadable_tokens == latest_computed
            and self._checkpoints.is_forced(
                req_status.req.request_id, num_offloadable_tokens
            )
        )

    def _build_store_jobs(
        self,
        scheduler_output: SchedulerOutput,
    ) -> dict[int, TransferJob]:
        blocks_per_chunk = self.config.blocks_per_chunk
        store_jobs: dict[int, TransferJob] = {}
        for req_id in chain(
            scheduler_output.num_scheduled_tokens,
            scheduler_output.finished_req_ids or (),
        ):
            req_status = self._req_status.get(req_id)
            if req_status is None:
                continue
            if not self._cold_reuse_enabled(req_status):
                continue
            # update_state_after_alloc() can enqueue a load earlier in this
            # same scheduler batch.  Reconsider normal stores on a later step,
            # after completion drains the load job, without advancing the
            # request's stored-chunk cursor or allocating a store destination.
            if any(
                not self._jobs[job_id].is_store
                for job_id in req_status.transfer_jobs
            ):
                logger.debug(
                    "Deferring store for request %s while a load is pending",
                    req_id,
                )
                continue
            req = req_status.req

            if req.status is RequestStatus.FINISHED_ABORTED:
                num_tokens_after_batch = req.num_computed_tokens
            elif req.is_finished():
                num_tokens_after_batch = req.num_tokens
            else:
                num_scheduled_tokens = scheduler_output.num_scheduled_tokens[req_id]
                num_tokens_after_batch = req.num_computed_tokens + num_scheduled_tokens

            num_offloadable_tokens = self._num_storable_tokens(
                req_status, num_tokens_after_batch
            )

            new_offload_keys = self._new_store_keys(
                req_status, num_offloadable_tokens
            )

            if not new_offload_keys:
                # The attention prefix through this cursor was already
                # accepted by the manager.  An exact sidecar may still be
                # missing (for example, after an older unaligned prefill).
                self._current_batch_attention_ready_req_ids.add(req_id)
                req_status.advance_stored_idx(num_offloadable_tokens)
                continue

            final_prompt_limit = self._calc_num_offloadable_tokens(
                req_status, req.num_prompt_tokens
            )
            final_prompt_boundary = min(
                round_down(final_prompt_limit, group_config.tokens_per_chunk)
                for group_config in self.config.kv_group_configs
            )
            if (
                not req.is_finished()
                and len(new_offload_keys) < self._store_batch_keys
                and num_offloadable_tokens < final_prompt_boundary
                and not self._is_forced_checkpoint(
                    req_status, num_offloadable_tokens, num_tokens_after_batch
                )
            ):
                # Keep every group cursor unchanged.  A later scheduler step
                # will include these exact keys in a larger transfer job.
                continue

            store_output = self.manager.prepare_store(
                new_offload_keys, req_status.req_context
            )
            if store_output is None:
                self._release_pending_finished_store_keys(
                    req_id, new_offload_keys
                )
                self._connector_stats.increase_counter(
                    _ConnectorMetricName.ALLOCATION_FAILURE
                )
                logger.warning("Request %s: cannot store chunks", req_id)
                continue

            if not store_output.keys_to_store:
                # All requested attention keys already exist in the manager;
                # still publish the exact recurrent row for this boundary.
                self._current_batch_attention_ready_req_ids.add(req_id)
                self._release_pending_finished_store_keys(
                    req_id, new_offload_keys
                )
                req_status.advance_stored_idx(num_offloadable_tokens)
                continue

            self._touch(req_status)

            keys_to_store = set(store_output.keys_to_store)
            if req.is_finished():
                self._track_pending_finished_store_keys(req_id, keys_to_store)
                self._release_pending_finished_store_keys(
                    req_id, set(new_offload_keys).difference(keys_to_store)
                )

            group_sizes: list[int] = []
            block_indices: list[int] = []
            src_block_ids: list[int] = []
            fenced_block_ids: list[int] = []
            deferred_fence_block_ids: list[int] = []
            for group_config, group_state in zip(
                self.config.kv_group_configs, req_status.group_states
            ):
                is_sliding_window = (
                    group_config.sliding_window_size_in_chunks is not None
                )
                num_chunks = req_status.storable_chunks(
                    group_config, group_state, num_offloadable_tokens
                )
                start_chunk_idx = group_state.next_stored_chunk_idx
                block_ids = group_state.block_ids
                num_group_blocks = 0
                start_gpu_block_idx: int | None = None
                for idx, offload_key in enumerate(
                    group_state.offload_keys[start_chunk_idx:num_chunks]
                ):
                    if offload_key not in keys_to_store:
                        continue

                    chunk_idx = start_chunk_idx + idx

                    self._events_tracker.record_store(
                        req, group_config, chunk_idx, offload_key
                    )

                    gpu_block_idx = chunk_idx * blocks_per_chunk
                    for i in range(blocks_per_chunk):
                        block_id = block_ids[gpu_block_idx + i]
                        if block_id == 0:
                            continue
                        if start_gpu_block_idx is None:
                            start_gpu_block_idx = gpu_block_idx + i
                        src_block_ids.append(block_id)
                        num_group_blocks += 1
                        if is_sliding_window:
                            fenced_block_ids.append(block_id)
                        else:
                            deferred_fence_block_ids.append(block_id)

                group_sizes.append(num_group_blocks)
                block_indices.append(start_gpu_block_idx or 0)
                group_state.next_stored_chunk_idx = max(
                    group_state.next_stored_chunk_idx, num_chunks
                )

            src_spec = GPULoadStoreSpec(
                src_block_ids, group_sizes=group_sizes, block_indices=block_indices
            )
            dst_spec = store_output.store_spec

            job_id = self._generate_job_id()
            # a store can only be issued when no load is pending.
            if req_status.transfer_jobs:
                any_jid = next(iter(req_status.transfer_jobs))
                assert self._jobs[any_jid].is_store
            req_status.transfer_jobs.add(job_id)

            # Watch sliding window blocks as they may get evicted
            # before the request finishes
            for bid in fenced_block_ids:
                self._block_id_to_pending_jobs.setdefault(bid, set()).add(job_id)

            # the non-sliding window blocks will be watched only
            # when the request finishes
            self._jobs[job_id] = TransferJobStatus(
                req_id=req_id,
                pending_count=self.config.num_workers,
                keys=set(keys_to_store),
                is_store=True,
                deferred_fence_block_ids=deferred_fence_block_ids,
                fenced_block_ids=fenced_block_ids or None,
            )

            store_jobs[job_id] = TransferJob(
                req_id=req_id, src_spec=src_spec, dst_spec=dst_spec
            )

            logger.debug(
                "Request %s offloading %s chunks upto %d tokens (job %d)",
                req_id,
                len(keys_to_store),
                num_offloadable_tokens,
                job_id,
            )

            if req.is_finished():
                # Register non-sliding-window blocks for flush detection.
                for bid in deferred_fence_block_ids:
                    self._block_id_to_pending_jobs.setdefault(bid, set()).add(job_id)
                    if bid in self._current_batch_allocated_block_ids:
                        self._current_batch_jobs_to_flush.add(job_id)

        return store_jobs

    # DECODE_CKPT_V1 (verb ``decode``): a recurrent checkpoint at every 128-token
    # boundary b a request crosses while decoding (glm53_decode_checkpoint), so the
    # next turn of the conversation restores at b, not at floor128(previous prompt).
    def _init_decode_checkpoints(self, vllm_config: VllmConfig) -> bool:
        """Boot switch: on with a recurrent sidecar, refused with spec decoding."""
        flags = recurrent_checkpoint_flags
        if flags.DECODE not in flags.RECURRENT_CHECKPOINT_V2_FOR:
            return False
        if self._recurrent_manager is None:
            logger.warning(
                "decode is inert: the connector has no recurrent sidecar "
                "(recurrent_state_bytes_to_use is 0 or the model has no recurrent "
                "state). Set recurrent_state_bytes_to_use or remove decode from %s.",
                flags.ENV_NAME,
            )
            return False
        if vllm_config.speculative_config is not None:
            raise ValueError(
                f"{flags.ENV_NAME} has decode, which stores the live recurrent row "
                "at each decode boundary; speculative decoding rewrites that row "
                "after rejections. Remove decode from the verb list or run without "
                "--speculative-config."
            )
        _decode_ckpt.log_enabled(
            self.config.tokens_per_hash, self._decode_chunk_sizes()
        )
        return True

    def _decode_chunk_sizes(self) -> tuple[int, ...]:
        return tuple(
            group.tokens_per_chunk for group in self.config.kv_group_configs
        )

    def _has_pending_load(self, req_status: RequestOffloadState) -> bool:
        """A load is in flight: never mixed with stores of the same request."""
        return any(
            not self._jobs[job_id].is_store for job_id in req_status.transfer_jobs
        )

    def _num_storable_tokens(
        self, req_status: RequestOffloadState, num_tokens: int
    ) -> int:
        """_calc_num_offloadable_tokens, extended through the planned decode boundary.

        The prompt-only cap stops the attention store at the prompt, but the row
        at a decode boundary b is restorable only with the chunks under b. b was
        planned only once hashed and within max_offload_tokens.
        """
        num = self._calc_num_offloadable_tokens(req_status, num_tokens)
        boundary = req_status.decode_checkpoint_boundary
        if boundary <= num:
            return num
        return min(num_tokens, req_status.req.num_tokens, boundary)

    def _decode_checkpoint_boundary(
        self, req_status: RequestOffloadState
    ) -> int | None:
        """The new boundary this request's dispatched steps end on, if storable now."""
        req = req_status.req
        if (
            req.is_finished()
            or not self._cold_reuse_enabled(req_status)
            or self._has_pending_load(req_status)
        ):
            return None
        position = _decode_ckpt.RequestPosition(
            num_computed=req.num_computed_tokens,
            num_prompt=req.num_prompt_tokens,
            num_hashed=len(req.block_hashes) * self.config.tokens_per_hash,
            last_boundary=req_status.decode_checkpoint_boundary,
            max_offload=req_status.max_offload_tokens,
        )
        return _decode_ckpt.decode_boundary(
            position, self.config.tokens_per_hash, self._decode_chunk_sizes()
        )

    def _plan_decode_checkpoints(
        self, scheduler_output: SchedulerOutput
    ) -> dict[ReqId, int]:
        """{req_id: b} for scheduled requests whose steps end on a new boundary b.

        Runs before _build_store_jobs, so this step's attention store covers the
        chunks under b. One attempt per boundary: b is kept even if not stored.
        """
        if not self._decode_checkpoints or not self.recurrent_sidecar_ready:
            return {}
        planned: dict[ReqId, int] = {}
        for req_id in scheduler_output.num_scheduled_tokens:
            req_status = self._req_status.get(req_id)
            boundary = (
                None
                if req_status is None
                else self._decode_checkpoint_boundary(req_status)
            )
            if boundary is None:
                continue
            req_status.decode_checkpoint_boundary = boundary
            planned[req_id] = boundary
        if planned:
            _decode_ckpt.count("planned", len(planned))
        return planned

    def _build_decode_checkpoint_jobs(
        self, planned: dict[ReqId, int], attention_store_req_ids: set[ReqId]
    ) -> dict[int, TransferJob]:
        """One sidecar job per planned boundary, captured before this step's forward.

        A row without the attention chunks under it can never be restored, so it
        is not stored either.
        """
        jobs: dict[int, TransferJob] = {}
        for req_id, boundary in planned.items():
            if req_id not in attention_store_req_ids:
                _decode_ckpt.count("no_attention")
                continue
            issued = self._issue_decode_checkpoint(self._req_status[req_id], boundary)
            if issued is not None:
                jobs[issued[0]] = issued[1]
        if self._decode_checkpoints:
            _decode_ckpt.maybe_log_stats()
        return jobs

    def _issue_decode_checkpoint(
        self, req_status: RequestOffloadState, boundary: int
    ) -> tuple[int, TransferJob] | None:
        """Book the row's CPU slot and its job; None if the key is held or no room."""
        req = req_status.req
        key = self._recurrent_sidecar_key(req, boundary)
        if key is None or self._recurrent_manager is None:
            raise _decode_ckpt.DecodeCheckpointError(
                f"{req.request_id}: decode boundary {boundary} has no recurrent key "
                f"or sidecar ({len(req.block_hashes)} block hashes); only hashed "
                "boundaries may be planned"
            )
        output = self._recurrent_manager.prepare_store((key,), req_status.req_context)
        if output is None:
            _decode_ckpt.count("failed")
            self._connector_stats.increase_counter(
                _ConnectorMetricName.ALLOCATION_FAILURE
            )
            return None
        if output.evicted_keys:
            self._connector_stats.increase_counter(
                "vllm:kv_offload_recurrent_cache_evictions", len(output.evicted_keys)
            )
        if not output.keys_to_store:
            _decode_ckpt.count("dup")
            return None
        _decode_ckpt.count("issued")
        _decode_ckpt.log_issue(req.request_id, boundary, req.num_prompt_tokens, key)
        self._checkpoints.on_sidecar_store_issued(req.request_id, boundary, key)
        return self._book_decode_job(
            req_status, key, int(output.store_spec.block_ids[0])
        )

    def _book_decode_job(
        self, req_status: RequestOffloadState, key: OffloadKey, cpu_block_id: int
    ) -> tuple[int, TransferJob]:
        """A sidecar-only store job; every worker captures it once, pre-forward."""
        req_id = req_status.req.request_id
        job_id = self._generate_job_id()
        req_status.transfer_jobs.add(job_id)
        self._jobs[job_id] = _sidecar_store_status(
            req_id, key, self.config.num_workers
        )
        transfer = RecurrentSidecarTransfer(
            key=key, cpu_block_id=cpu_block_id, pre_forward=True
        )
        return job_id, _sidecar_only_job(req_id, transfer)

    def build_connector_meta(
        self, scheduler_output: SchedulerOutput
    ) -> KVConnectorMetadata:
        self._update_req_states(scheduler_output)
        schedule_end_context = ScheduleEndContext(
            new_req_ids=[req.req_id for req in scheduler_output.scheduled_new_reqs],
            preempted_req_ids=scheduler_output.preempted_req_ids or (),
        )
        self.manager.on_schedule_end(schedule_end_context)
        if self._recurrent_manager is not None:
            self._recurrent_manager.on_schedule_end(schedule_end_context)

        # Flush jobs for preempted requests.
        for req_id in scheduler_output.preempted_req_ids or ():
            req_status = self._req_status.get(req_id)
            if req_status is None or not req_status.transfer_jobs:
                continue
            any_jid = next(iter(req_status.transfer_jobs))
            assert self._jobs[any_jid].is_store
            self._current_batch_jobs_to_flush.update(req_status.transfer_jobs)

        # Flush jobs that contain re-allocated blocks.
        if (
            self._block_id_to_pending_jobs
            and not self._block_id_to_pending_jobs.keys().isdisjoint(
                self._current_batch_allocated_block_ids
            )
        ):
            self._current_batch_jobs_to_flush.update(
                jid
                for bid in self._current_batch_allocated_block_ids
                if bid in self._block_id_to_pending_jobs
                for jid in self._block_id_to_pending_jobs[bid]
            )

        partial_store_jobs = self._build_partial_tail_store_jobs(scheduler_output)
        # DECODE_CKPT_V1: planned first, so the attention store covers the chunks under b.
        decode_boundaries = self._plan_decode_checkpoints(scheduler_output)
        normal_store_jobs = self._build_store_jobs(scheduler_output)
        attention_store_req_ids = (
            self._current_batch_attention_ready_req_ids
            | {
                entry.req_id
                for entry in (partial_store_jobs | normal_store_jobs).values()
            }
        )
        recurrent_store_jobs = self._build_recurrent_sidecar_store_jobs(
            scheduler_output, attention_store_req_ids
        )
        decode_store_jobs = self._build_decode_checkpoint_jobs(
            decode_boundaries, attention_store_req_ids
        )
        meta = OffloadingConnectorMetadata(
            load_jobs=self._current_batch_load_jobs,
            store_jobs=(
                partial_store_jobs
                | normal_store_jobs
                | recurrent_store_jobs
                | decode_store_jobs
            ),
            jobs_to_flush=self._current_batch_jobs_to_flush,
        )

        # All prepare_store calls for finished requests have been issued.
        # Signal on_request_finished and clean up state where possible.
        for req_id in scheduler_output.finished_req_ids or ():
            req_status = self._req_status.get(req_id)
            if req_status is None:
                continue
            req_status.finished_signaled = True
            self.manager.on_request_finished(req_status.req_context)
            if self._recurrent_manager is not None:
                self._recurrent_manager.on_request_finished(req_status.req_context)
            if not req_status.transfer_jobs:
                self._release_pending_finished_store_keys(req_id)
                del self._req_status[req_id]
        self._current_batch_load_jobs = {}
        self._current_batch_jobs_to_flush = set()
        self._current_batch_allocated_block_ids = set()
        self._current_batch_attention_ready_req_ids = set()
        self._tail_fuse_end_step()
        return meta

    def has_pending_push_work(self) -> bool:
        """Whether the engine must keep stepping.

        While True, build_connector_meta() and update_connector_output()
        continue to be called even when no requests are scheduled.
        """
        return (
            bool(self._jobs)
            or self.manager.has_pending_work()
            or (
                self._recurrent_manager is not None
                and self._recurrent_manager.has_pending_work()
            )
        )

    def update_connector_output(self, connector_output: KVConnectorOutput):
        """
        Update KVConnector state from worker-side connectors output.

        Args:
            connector_output (KVConnectorOutput): the worker-side
                connectors output.
        """
        meta = connector_output.kv_connector_worker_meta
        if not isinstance(meta, OffloadingWorkerMetadata):
            assert meta is None
            meta = OffloadingWorkerMetadata()
        if not meta.transfer_stats.is_empty():
            transfer_stats = OffloadingConnectorStats()
            if not meta.transfer_stats.load.is_empty():
                transfer_stats.increase_counter(
                    _TransferMetricName.LOAD_BYTES,
                    meta.transfer_stats.load.bytes,
                )
                transfer_stats.increase_counter(
                    _TransferMetricName.LOAD_TIME,
                    meta.transfer_stats.load.time,
                )
                for size in meta.transfer_stats.load.sizes:
                    transfer_stats.observe_histogram(
                        _TransferMetricName.LOAD_SIZE, size
                    )
            if not meta.transfer_stats.store.is_empty():
                transfer_stats.increase_counter(
                    _TransferMetricName.STORE_BYTES,
                    meta.transfer_stats.store.bytes,
                )
                transfer_stats.increase_counter(
                    _TransferMetricName.STORE_TIME,
                    meta.transfer_stats.store.time,
                )
                for size in meta.transfer_stats.store.sizes:
                    transfer_stats.observe_histogram(
                        _TransferMetricName.STORE_SIZE, size
                    )
            self._connector_stats.aggregate(transfer_stats)

        if meta.recurrent_sidecar_registrations:
            if self._recurrent_manager is None:
                raise RuntimeError(
                    "worker reported recurrent registration without sidecar budget"
                )
            if meta.recurrent_sidecar_signature is None:
                raise RuntimeError("recurrent registration omitted its signature")
            registration = (
                meta.recurrent_sidecar_bytes_per_rank,
                meta.recurrent_sidecar_signature,
            )
            if self._recurrent_sidecar_signature is None:
                self._recurrent_sidecar_signature = registration
            elif self._recurrent_sidecar_signature != registration:
                raise RuntimeError("worker recurrent registrations are incompatible")
            self._recurrent_sidecar_registrations += (
                meta.recurrent_sidecar_registrations
            )
            if self._recurrent_sidecar_registrations > self.config.num_workers:
                raise RuntimeError("duplicate recurrent worker registration")
            self.recurrent_sidecar_ready = (
                self._recurrent_sidecar_registrations == self.config.num_workers
                and meta.recurrent_sidecar_bytes_per_rank
                == self._recurrent_state_bytes_per_rank
            )

        for job_id, count in meta.completed_jobs.items():
            assert count > 0
            if job_id < self._stale_job_threshold:
                logger.debug(
                    "Skipping stale completed job %d (pre-reset counter: %d)",
                    job_id,
                    self._stale_job_threshold,
                )
                continue
            job_status = self._jobs[job_id]
            job_status.pending_count -= count
            if job_status.pending_count > 0:
                continue
            assert job_status.pending_count == 0

            req_status = self._req_status[job_status.req_id]
            if job_status.is_store:
                if job_status.attention_transfer:
                    self.manager.complete_store(
                        job_status.keys, req_status.req_context
                    )
                    self._checkpoints.on_attention_store_completed()
                    self._release_pending_finished_store_keys(
                        job_status.req_id, job_status.keys
                    )
                if job_status.recurrent_sidecar_key is not None:
                    assert self._recurrent_manager is not None
                    self._recurrent_manager.complete_store(
                        (job_status.recurrent_sidecar_key,),
                        req_status.req_context,
                    )
                    self._checkpoints.on_sidecar_store_completed(
                        job_status.recurrent_sidecar_key
                    )
            else:
                if job_status.attention_transfer:
                    self.manager.complete_load(
                        job_status.keys, req_status.req_context
                    )
                if job_status.recurrent_sidecar_key is not None:
                    assert self._recurrent_manager is not None
                    self._recurrent_manager.complete_load(
                        (job_status.recurrent_sidecar_key,),
                        req_status.req_context,
                    )
                if self._chunks_being_loaded:
                    self._chunks_being_loaded.difference_update(job_status.keys)
            if self._block_id_to_pending_jobs:
                # Sliding window blocks are tracked from store creation
                # and must be cleaned up unconditionally.
                self._remove_pending_job(job_id, job_status.fenced_block_ids)
                # Non-sliding-window blocks are only tracked after
                # request_finished, so only clean up for finished requests.
                if req_status.req.is_finished():
                    self._remove_pending_job(
                        job_id, job_status.deferred_fence_block_ids
                    )

            del self._jobs[job_id]
            req_status.transfer_jobs.remove(job_id)
            if req_status.finished_signaled and not req_status.transfer_jobs:
                self._release_pending_finished_store_keys(job_status.req_id)
                del self._req_status[job_status.req_id]

    def get_stats(self) -> OffloadingConnectorStats | None:
        stats: OffloadingConnectorStats | None = None
        if not self._connector_stats.is_empty():
            stats = self._connector_stats
            self._connector_stats = OffloadingConnectorStats()

        manager_stats = self.manager.get_stats()
        if manager_stats is not None:
            if stats is None:
                stats = manager_stats
            else:
                stats.aggregate(manager_stats)

        if self._recurrent_manager is not None:
            if stats is None:
                stats = OffloadingConnectorStats()
            recurrent_stats = self._recurrent_manager.get_stats()
            if recurrent_stats is not None:
                for name, value in recurrent_stats.reduce().items():
                    if name.startswith("vllm:kv_offload_cpu_cache_"):
                        stats.set_gauge(
                            name.replace("_cpu_cache_", "_recurrent_cache_", 1), value
                        )

        return stats

    def request_finished(
        self,
        request: Request,
    ) -> tuple[bool, dict[str, Any] | None]:
        """
        Called when a request has finished, before its blocks are freed.

        Returns:
            True if the request is being saved/sent asynchronously and blocks
            should not be freed until the request_id is returned from
            get_finished().
            Optional KVTransferParams to be included in the request outputs
            returned by the engine.
        """
        req_status = self._req_status.get(request.request_id)
        self._checkpoints.drop(request.request_id)

        if req_status is None:
            # Untracked request (offloading never started): no in-flight jobs,
            # nothing was deferred, so finalize immediately.
            req_context = _create_req_context(request)
            self.manager.on_new_request(req_context)
            self.manager.on_request_finished(req_context)
            if self._recurrent_manager is not None:
                self._recurrent_manager.on_new_request(req_context)
                self._recurrent_manager.on_request_finished(req_context)
            return False, None

        if not self._cold_reuse_enabled(req_status):
            # build_connector_meta() owns lifecycle cleanup for tracked
            # finished requests; publish no pending keys or cold stores.
            return False, None

        self._maybe_observe_lookup_async_delay(req_status)
        req_status.pending_store_wait_start_time = None
        req_status.pending_store_wait_deferred = False

        # Capture exactly the not-yet-prepared keys before blocks can be
        # freed.  Existing store jobs are already manager-visible but remain
        # dependencies until their individual completions arrive.
        req_status.update_offload_keys()
        num_tokens_after_batch = (
            request.num_computed_tokens
            if request.status is RequestStatus.FINISHED_ABORTED
            else request.num_tokens
        )
        num_offloadable_tokens = self._num_storable_tokens(
            req_status, num_tokens_after_batch
        )
        pending_keys = set(
            self._new_store_keys(req_status, num_offloadable_tokens)
        )
        pending_keys.update(
            key
            for job_id in req_status.transfer_jobs
            if self._jobs[job_id].is_store
            for key in self._jobs[job_id].keys
        )
        self._track_pending_finished_store_keys(
            request.request_id, pending_keys
        )

        # Keep req_status alive: _build_store_jobs will process finished_req_ids
        # on the next step and handle cleanup after creating store jobs.
        # Register deferred fences so future block reuse triggers a flush via
        # _block_id_to_pending_jobs.
        for job_id in req_status.transfer_jobs:
            job_status = self._jobs[job_id]
            for bid in job_status.deferred_fence_block_ids or ():
                self._block_id_to_pending_jobs.setdefault(bid, set()).add(job_id)

        return False, None

    def take_events(self) -> Iterable[KVCacheEvent]:
        """Drain pending KV cache events.

        Complete metadata is available only when self-describing KV events
        are enabled, and only for full-attention groups. Other shapes retain
        the previous placeholder payload so consumers can ignore them.

        Yields:
            ``BlockStored`` or ``BlockRemoved`` events corresponding to
            the underlying :class:`OffloadingEvent` stream.
        """
        yield from self._events_tracker.take_events(self.manager.take_events())

    def reset_cache(self) -> None:
        """Reset the offloading manager cache, evicting all stored chunks."""

        # reset_cache cannot be called in the middle of a schedule step
        assert not self._current_batch_load_jobs
        assert not self._current_batch_jobs_to_flush
        assert not self._current_batch_allocated_block_ids

        # Flush all in-flight jobs
        self._current_batch_jobs_to_flush.update(self._jobs.keys())

        for req_id, status in list(self._req_status.items()):
            if status.req.is_finished():
                if not status.finished_signaled:
                    self.manager.on_request_finished(status.req_context)
                del self._req_status[req_id]

        # Reset offloading manager caches
        self.manager.reset_cache()
        if self._recurrent_manager is not None:
            self._recurrent_manager.reset_cache()
        self._checkpoints.reset_stores()

        # Reset store progress so active requests re-offload from chunk 0.
        for status in self._req_status.values():
            for group_state in status.group_states:
                group_state.next_stored_chunk_idx = 0
            status.transfer_jobs.clear()
            status.partial_tail_boundary = None
            status.pending_store_wait_start_time = None
            status.pending_store_wait_deferred = False

        # Discard jobs and save job_counter to be able to discard worker responses
        self._stale_job_threshold = self._job_counter
        self._jobs.clear()
        self._block_id_to_pending_jobs.clear()
        self._pending_finished_store_keys.clear()
        self._pending_finished_store_key_owners.clear()

        # The manager pool is empty; pending event payloads and announced
        # reference counts are stale.
        self._events_tracker.reset()

        # Note: _current_batch_jobs_to_flush is intentionally NOT cleared.
        # The load flush IDs collected above must be delivered to workers.
        if self._chunks_being_loaded is not None:
            self._chunks_being_loaded.clear()

    def shutdown(self) -> None:
        self.manager.shutdown()
        if self._recurrent_manager is not None:
            self._recurrent_manager.shutdown()
