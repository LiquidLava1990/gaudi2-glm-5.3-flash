# SPDX-License-Identifier: Apache-2.0
"""HPU sparse kpool indexer for GLM-5.3 Flash.

This is the correctness-first HPU implementation.  It keeps the persistent
index cache at 33 bytes/source-token/layer, uses a fixed four-token raw tail,
streams long-context scoring in bounded chunks, and writes *physical* latent
cache rows into the shared top-k buffer for the HPU MLA consumer.
"""

from __future__ import annotations
from glm53_r589_bounded_prefill import stage as _r589_stage

import os
from glm53_mixed_graph import (capture_active as mixed_graph_capture_active, capture_tile_marks, key_capture_active,
                               lazy_fence_sync, record_capture_branch, record_capture_scalar)

import torch
import torch.nn.functional as F
from torch import nn

from vllm.compilation.breakable_cudagraph import eager_break_during_capture
from vllm.forward_context import get_forward_context
from vllm.logger import init_logger
from vllm.models.glm5next.nvidia import attention as upstream_attention
from vllm.v1.attention.backends.mla.indexer import DeepseekV32IndexerBackend

from vllm_gaudi.ops.hpu_glm5next_kpool import (
    INDEX_DIM,
    compress_complete_kpools,
    finalize_pooled_topk,
    fwht128,
    logical_to_physical_slots,
    pack_pooled_index_rows,
    pooled_slots_to_tp_local,
    quantize_index_fp8,
    streaming_pooled_topk,
    streaming_pooled_topk_candidates,
)


_FIRST_DYNAMIC_KPOOL_DRAINED = False
_COMPILED_STREAMING_POOLED_TOPK = {}
logger = init_logger(__name__)
MULTI_BATCH_TILE_CEILING_ENV = "HPU_GLM53_MULTI_BATCH_KPOOL_TILE_CEILING"


def _announce_multi_batch_tile_ceiling() -> None:
    """One boot line when the ceiling is set in the env (unset keeps the 4096 default and prints nothing): a scan of 2 or more rows
    chunks at most this many pools, unless HPU_GLM53_DECODE_KPOOL_TILE_CEILING_FOR lists the decode key being captured."""
    if MULTI_BATCH_TILE_CEILING_ENV in os.environ:
        logger.warning("GLM53_MULTI_BATCH_KPOOL_TILE_CEILING armed pools=%s context_chunk_pools=%s budget_MiB=%s",
                       os.environ[MULTI_BATCH_TILE_CEILING_ENV], os.environ.get("HPU_GLM53_CONTEXT_CHUNK_POOLS", "default"),
                       os.environ.get("HPU_GLM53_CONTEXT_CHUNK_BUDGET_MIB", "512"))


_announce_multi_batch_tile_ceiling()


def _compiled_streaming_pooled_topk(
    query_values: torch.Tensor,
    head_weights: torch.Tensor,
    packed_cache: torch.Tensor,
    block_table: torch.Tensor,
    sequence_lengths: torch.Tensor,
    source_block_size: int,
    pool_size: int,
    topk_tokens: int,
    context_chunk_pools: int,
    max_complete_pools_bucket: int,
):
    return streaming_pooled_topk(
        query_values,
        head_weights,
        packed_cache,
        block_table,
        sequence_lengths,
        source_block_size=source_block_size,
        pool_size=pool_size,
        topk_tokens=topk_tokens,
        context_chunk_pools=context_chunk_pools,
        max_complete_pools_bucket=max_complete_pools_bucket,
    )


def _get_compiled_streaming_pooled_topk(cache_key=None):
    """Return one compiler wrapper per static HPU scoring shape.

    A single dynamic-false wrapper falls back to eager after Dynamo's eighth
    specialization.  The engine legitimately uses more shapes (prompt plus
    C1..C64), and this scorer's eager graph cannot compile on Gaudi2.  Separate
    wrappers keep every recipe below the specialization limit.
    """

    key = ("default",) if cache_key is None else tuple(cache_key)
    compiled = _COMPILED_STREAMING_POOLED_TOPK.get(key)
    if compiled is None:
        # Dynamo counts specializations by Python code object even when each
        # static shape owns a separate torch.compile wrapper.  The bounded HPU
        # grid has more than eight legitimate shapes, so retain compilation
        # instead of silently falling back to the known-broken eager graph.
        torch._dynamo.config.recompile_limit = max(
            int(torch._dynamo.config.recompile_limit), 256
        )
        if hasattr(torch._dynamo.config, "accumulated_recompile_limit"):
            torch._dynamo.config.accumulated_recompile_limit = max(
                int(torch._dynamo.config.accumulated_recompile_limit), 512
            )
        compiled = torch.compile(
            _compiled_streaming_pooled_topk,
            backend="hpu_backend",
            dynamic=False,
        )
        _COMPILED_STREAMING_POOLED_TOPK[key] = compiled
    return compiled


def bind_hpu_index_cache(cache, kv_cache) -> None:
    """Bind the HPU runner's tuple allocation to an index/tail cache object."""

    if isinstance(kv_cache, (tuple, list)):
        kv_cache = kv_cache[0]
    # CUDA uses [B,H=1,N,C]; HPU MLA allocates a flat [rows,C] tensor.
    if kv_cache.ndim >= 3 and kv_cache.shape[1] == 1:
        kv_cache = kv_cache.squeeze(1)
    cache.kv_cache = kv_cache


class HPUGlm5NextIndexerBackend(DeepseekV32IndexerBackend):
    """HPU pooled-index metadata keyed to the 128-token source page."""

    @staticmethod
    def get_name() -> str:
        return "HPU_GLM53_INDEXER"

    @staticmethod
    def get_supported_kernel_block_sizes() -> list[int]:
        # The HPU implementation consumes pooled rows itself. Its block table
        # addresses the shared 128-token source page, not CUDA's 64-row
        # DeepGEMM kernel page inherited by the upstream metadata backend.
        return [128]


class HPUGlm5NextIndexerCache(upstream_attention.Glm5NextIndexerCache):
    def bind_kv_cache(self, kv_cache: torch.Tensor) -> None:
        bind_hpu_index_cache(self, kv_cache)

    def get_attn_backend(self):
        return HPUGlm5NextIndexerBackend


class HPUGlm5NextSparseIndexerOp(nn.Module):
    """Paged HPU kpool cache update plus bounded streaming top-k."""

    r590_prompt_bias_contract = "indexer_causal_positions_v1"


    is_hpu_glm5next = True

    def __init__(
        self,
        *,
        k_cache,
        tail_cache,
        topk_indices_buffer: torch.Tensor,
        source_block_size: int,
        pool_size: int,
        topk_tokens: int,
        context_chunk_pools: int = 4096,
    ) -> None:
        super().__init__()
        if topk_indices_buffer is None:
            raise ValueError("GLM-5.3 HPU sparse indexer requires a top-k buffer")
        self.k_cache = k_cache
        self.tail_cache = tail_cache
        # These exact objects are storage, not dense attention consumers.
        self.k_cache.r590_prompt_bias_contract = "index_storage_no_prompt_bias_v1"
        self.tail_cache.r590_prompt_bias_contract = "index_storage_no_prompt_bias_v1"
        self.topk_indices_buffer = topk_indices_buffer
        self.source_block_size = int(source_block_size)
        self.pool_size = int(pool_size)
        self.topk_tokens = int(topk_tokens)
        # This is a ceiling, not a mandatory shape.  The scorer selects a
        # smaller graph-static tier when batch workspace would exceed the HBM
        # budget below.
        self.context_chunk_pools = int(
            os.environ.get("HPU_GLM53_CONTEXT_CHUNK_POOLS", context_chunk_pools)
        )
        self.context_chunk_budget_mib = int(
            os.environ.get("HPU_GLM53_CONTEXT_CHUNK_BUDGET_MIB", "512")
        )
        self.shared_kpool_wide_chunk = os.environ.get(
            "HPU_GLM53_SHARED_KPOOL_WIDE_CHUNK", "0"
        ).strip().lower() in ("1", "true", "yes", "on")
        self.shared_kpool_budget_mib = int(os.environ.get(
            "HPU_GLM53_SHARED_KPOOL_CHUNK_BUDGET_MIB",
            str(self.context_chunk_budget_mib),
        ))
        if self.shared_kpool_budget_mib <= 0:
            raise ValueError("GLM-5.3 shared kpool chunk budget must be positive")
        # HPU_GLM53_MULTI_BATCH_KPOOL_TILE_V1
        # Rank-3 HPU scorer kernels have a sharp throughput cliff at very wide
        # context dimensions. Keep B1 unmodified and tile only multi-request
        # scoring. This is a performance ceiling, not a context limit.
        self.multi_batch_kpool_tile_ceiling = int(
            os.environ.get("HPU_GLM53_MULTI_BATCH_KPOOL_TILE_CEILING", "4096")
        )
        if self.context_chunk_pools <= 0:
            raise ValueError("GLM-5.3 kpool chunk ceiling must be positive")
        if self.context_chunk_budget_mib <= 0:
            raise ValueError("GLM-5.3 kpool chunk budget must be positive")
        if self.multi_batch_kpool_tile_ceiling <= 0:
            raise ValueError("GLM-5.3 multi-batch kpool tile ceiling must be positive")
        self.dynamic_pool_bucket = os.environ.get(
            "HPU_GLM53_DYNAMIC_KPOOL_BUCKET", "0"
        ).strip().lower() in ("1", "true")
        self.drain_first_dynamic_pool = os.environ.get(
            "HPU_GLM53_DRAIN_FIRST_KPOOL", "1"
        ).strip().lower() in ("1", "true")
        self.compiled_kpool = os.environ.get(
            "HPU_GLM53_COMPILED_KPOOL", "0"
        ).strip().lower() in ("1", "true")
        self.tp_sharded_kpool = os.environ.get(
            "HPU_GLM53_TP_SHARDED_KPOOL", "0"
        ).strip().lower() in ("1", "true")
        self.tp_rank = 0
        self.tp_size = 1
        if self.tp_sharded_kpool:
            if not torch.distributed.is_available() or not torch.distributed.is_initialized():
                raise RuntimeError(
                    "TP-sharded GLM-5.3 kpool requires initialized torch.distributed"
                )
            from vllm.distributed import (
                get_tensor_model_parallel_rank,
                get_tensor_model_parallel_world_size,
            )

            self.tp_rank = get_tensor_model_parallel_rank()
            self.tp_size = get_tensor_model_parallel_world_size()
            if self.tp_size <= 1:
                raise RuntimeError("TP-sharded GLM-5.3 kpool requires TP > 1")
        self.warmup_max_kpools = int(
            os.environ.get("HPU_GLM53_WARMUP_MAX_KPOOLS", "64")
        )
        self.trace_topk_internal = os.environ.get(
            "HPU_GLM53_TRACE_MLA_INTERNAL", "0"
        ).lower() in ("1", "true", "yes", "on")
        self.trace_topk_internal_calls = 0
        self.trace_topk_min_sequence = int(
            os.environ.get("HPU_GLM53_TRACE_TOPK_MIN_SEQUENCE", "0")
        )
        if self.source_block_size % self.pool_size:
            raise ValueError("source block size must be divisible by kpool")

    def _trace_physical_topk(
        self,
        logical: torch.Tensor,
        block_table: torch.Tensor,
        sequence_lengths: torch.Tensor,
        max_sequence_length: int | None = None,
    ) -> None:
        prefix = str(getattr(self.k_cache, "prefix", "unknown"))
        if (
            not self.trace_topk_internal
            or "layers.3." not in prefix
            or self.trace_topk_internal_calls >= 1
        ):
            return
        max_sequence = (
            int(max_sequence_length)
            if max_sequence_length is not None
            else (
                int(sequence_lengths.long().max().item())
                if sequence_lengths.numel()
                else 0
            )
        )
        if max_sequence < self.trace_topk_min_sequence:
            return
        call_id = self.trace_topk_internal_calls
        self.trace_topk_internal_calls += 1
        # The indexer is already outside the whole-model graph. Synchronize and
        # copy only one 2051-int row after selection; never inspect activations.
        row = logical[-1].detach().cpu()
        valid = row[row >= 0]
        logger.warning(
            "GLM53_TRACE_TOPK_LOGICAL layer=%s call=%d max_sequence=%d "
            "valid=%d/%d block_table_first=%s logical=%s",
            prefix, call_id, max_sequence, valid.numel(), row.numel(),
            block_table[-1].detach().cpu().reshape(-1)[:16].tolist(),
            valid.tolist())

    @staticmethod
    def _active_pool_bucket_for_length(
        sequence_length: int,
        pool_size: int,
        chunk_pools: int,
        capacity_pools: int,
    ) -> int:
        """Round active context to a stable bounded kpool scoring bucket."""

        active = int(sequence_length) // pool_size
        if active <= 0:
            return 0
        # Keep stable useful tiers while allowing the scan chunk itself to be
        # chosen independently from batch workspace.  In particular, a B1
        # 65K scan must not force a 2K-pool request to scan 65K rows.
        tier_ceiling = max(4096, chunk_pools)
        # Preserve static power-of-two recipe shapes without making medium
        # contexts scan up to four times their live pooled rows.  The missing
        # 2K/8K/32K tiers were especially expensive during decode because all
        # eleven MLA indexers repeat this scan for every generated token.
        for tier in (64, 256, 1024, 2048, 4096, 8192, 16384, 32768, 65536):
            if tier > tier_ceiling:
                continue
            if active <= tier:
                return min(tier, capacity_pools)
        rounded = ((active + chunk_pools - 1) // chunk_pools) * chunk_pools
        return min(rounded, capacity_pools)

    @classmethod
    def _active_pool_bucket(
        cls,
        sequence_lengths: torch.Tensor,
        pool_size: int,
        chunk_pools: int,
        capacity_pools: int,
    ) -> int:
        if not sequence_lengths.numel():
            return 0
        return cls._active_pool_bucket_for_length(
            int(sequence_lengths.long().max().item()),
            pool_size,
            chunk_pools,
            capacity_pools,
        )

    # GENERIC_KPOOL_SCORING_BUDGET_V1
    @staticmethod
    def _scoring_bytes_per_pool(
        *,
        index_heads: int,
        index_dim: int,
        packed_row_bytes: int,
    ) -> int:
        """Conservatively bound chunk-local live bytes per batch/pool."""

        index_heads = max(1, int(index_heads))
        index_dim = max(1, int(index_dim))
        packed_row_bytes = max(1, int(packed_row_bytes))

        # streaming_pooled_topk_candidates gathers the packed cache row, then
        # materializes contiguous FP8 values and their FP32 scale.  The native
        # FP8 scorer feeds those values directly to fp8_gemm_v2; only the
        # fallback scorer expands the key vectors before matmul.  Charging a
        # nonexistent FP32 key copy forced 128-row prompt tiles down to 2K-pool
        # scans even though the real native-FP8 working set fits a 4K scan.
        gathered_packed_row_bytes = packed_row_bytes
        unpacked_fp8_row_bytes = index_dim
        unpacked_fp32_scale_bytes = 4
        native_fp8_score = os.environ.get(
            "HPU_GLM53_NATIVE_FP8_KPOOL_SCORE", "0"
        ).strip().lower() in ("1", "true", "yes", "on")
        native_fp8_budget = os.environ.get(
            "HPU_GLM53_NATIVE_FP8_KPOOL_BUDGET", "0"
        ).strip().lower() in ("1", "true", "yes", "on")
        fp32_dequantized_values_bytes = (
            0 if native_fp8_score and native_fp8_budget else index_dim * 4
        )

        # Matmul output can overlap the ReLU/weighted head-logit tensor.  The
        # reduced, scaled, masked, and merged score tensors can also overlap.
        fp32_head_logits_bytes = index_heads * 4 * 2
        fp32_score_staging_bytes = 4 * 4

        # Charge eight int64 routing/ID tensors, eight boolean masks, and both
        # integer and FP32 top-k merge/select bookkeeping.  Some are views or
        # have bounded widths in eager execution; counting all of them per pool
        # deliberately overestimates compiler and allocator liveness.
        id_and_mapping_bytes = 8 * 8
        mask_bytes = 8
        topk_bookkeeping_bytes = 8 * 4 + 4 * 4
        allocator_slack_bytes = 64
        return (
            gathered_packed_row_bytes
            + unpacked_fp8_row_bytes
            + unpacked_fp32_scale_bytes
            + fp32_dequantized_values_bytes
            + fp32_head_logits_bytes
            + fp32_score_staging_bytes
            + id_and_mapping_bytes
            + mask_bytes
            + topk_bookkeeping_bytes
            + allocator_slack_bytes
        )

    def _scoring_chunk(
        self,
        *,
        batch: int,
        index_heads: int,
        index_dim: int,
        packed_row_bytes: int,
        active_pool_bucket: int,
        shared_block_table: bool = False,
    ) -> int:
        """Bound scorer workspace without perturbing an in-budget scan."""

        if active_pool_bucket <= 0:
            return self.context_chunk_pools
        bytes_per_pool = self._scoring_bytes_per_pool(
            index_heads=index_heads,
            index_dim=index_dim,
            packed_row_bytes=packed_row_bytes,
        )
        if (
            shared_block_table
            and self.shared_kpool_wide_chunk
            and not self.tp_sharded_kpool
            and not self.compiled_kpool
            and all(os.environ.get(key, "0").strip().lower()
                    in ("1", "true", "yes", "on") for key in (
                        "HPU_GLM53_NATIVE_FP8_KPOOL_SCORE",
                        "HPU_GLM53_NATIVE_FP8_KPOOL_BUDGET",
                    ))
        ):
            # A certified shared tile gathers and unpacks ONE set of keys.
            # Keep all query-dependent score/routing/sort/slack charges per
            # query; remove only duplicate packed/FP8/scale key storage.
            key_bytes = max(1, packed_row_bytes) + max(1, index_dim) + 4
            workspace_per_pool = (
                max(1, batch) * bytes_per_pool
                - (max(1, batch) - 1) * key_bytes
            )
            budget_bytes = self.shared_kpool_budget_mib * 2**20
            memory_cap = max(1, budget_bytes // max(1, workspace_per_pool))
            requested = min(self.context_chunk_pools, active_pool_bucket)
            if requested <= memory_cap:
                return requested
            ceiling = min(requested, memory_cap)
            return 1 << (max(1, ceiling).bit_length() - 1)
        budget_bytes = self.context_chunk_budget_mib * 2**20
        memory_cap = max(1, budget_bytes // max(1, batch * bytes_per_pool))
        performance_cap = self.context_chunk_pools
        if batch > 1:
            performance_cap = min(
                performance_cap,
                self.multi_batch_kpool_tile_ceiling,
            )
        requested_chunk_pools = min(
            performance_cap,
            active_pool_bucket,
        )
        if requested_chunk_pools <= memory_cap:
            return performance_cap
        ceiling = min(requested_chunk_pools, memory_cap)
        return 1 << (max(1, ceiling).bit_length() - 1)

    @staticmethod
    def _flatten_cache(cache: torch.Tensor, row_width: int) -> torch.Tensor:
        if cache.numel() % row_width:
            raise RuntimeError(
                f"cache with {cache.numel()} elements cannot form {row_width}-wide rows"
            )
        return cache.reshape(-1, row_width)

    @_r589_stage("indexer.cache")
    def _update_pooled_cache(
        self,
        k: torch.Tensor,
        gate_score: torch.Tensor,
        positions: torch.Tensor,
        compressed_slots: torch.Tensor,
        tail_slots: torch.Tensor,
        compress_ape: torch.Tensor,
    ) -> None:
        """Compress every newly completed pool, then retain only the raw tail."""

        n = k.shape[0]
        tail_rows = self._flatten_cache(
            self.tail_cache.kv_cache, 2 * k.shape[-1]
        )
        pooled_rows = self._flatten_cache(self.k_cache.kv_cache, k.shape[-1] + 4)
        compressed_slots = compressed_slots[:n].long()
        tail_slots = tail_slots[:n].long()
        positions = positions[:n].long()

        # Keep every intermediate shape bucket-static.  nonzero() creates a
        # value-dependent output shape, which makes an HPU graph captured on
        # one pool-boundary pattern invalid for the next request.  Compute all
        # N candidate pools, mask non-completions to one identical dummy write,
        # and retain fixed launch shapes across the champion's dynamic buckets.
        complete_mask = compressed_slots >= 0
        if self.tp_sharded_kpool:
            pools_per_block = self.source_block_size // self.pool_size
            compressed_slots, owned = pooled_slots_to_tp_local(
                compressed_slots,
                pools_per_block=pools_per_block,
                tp_rank=self.tp_rank,
                tp_size=self.tp_size,
            )
            complete_mask = complete_mask & owned
        offsets = torch.arange(
            1 - self.pool_size, 1, device=k.device, dtype=torch.long
        )
        all_rows = torch.arange(n, device=k.device, dtype=torch.long)
        candidate_rows = all_rows[:, None] + offsets[None]
        safe_rows = candidate_rows.clamp(0, max(n - 1, 0))
        completion_tail_block = torch.div(
            tail_slots,
            self.pool_size,
            rounding_mode="floor",
        )
        expected_positions = positions[:, None] + offsets[None]
        candidate_tail_block = torch.div(
            tail_slots.index_select(0, safe_rows.reshape(-1)).reshape_as(safe_rows),
            self.pool_size,
            rounding_mode="floor",
        )
        candidate_positions = positions.index_select(
            0, safe_rows.reshape(-1)
        ).reshape_as(safe_rows)
        from_current = (
            (candidate_rows >= 0)
            & (candidate_tail_block == completion_tail_block[:, None])
            & (candidate_positions == expected_positions)
        )

        current_k = k.index_select(0, safe_rows.reshape(-1)).reshape(
            n, self.pool_size, k.shape[-1]
        )
        current_gate = gate_score.index_select(
            0, safe_rows.reshape(-1)
        ).reshape_as(current_k)
        old_tail_slots = (
            completion_tail_block[:, None] * self.pool_size
            + torch.remainder(expected_positions, self.pool_size)
        ).clamp(0, max(tail_rows.shape[0] - 1, 0))
        old_tail = tail_rows.index_select(
            0, old_tail_slots.reshape(-1)
        ).reshape(n, self.pool_size, 2 * k.shape[-1])
        old_k, old_gate = old_tail.split(k.shape[-1], dim=-1)
        pool_k = torch.where(from_current[..., None], current_k, old_k)
        pool_gate = torch.where(from_current[..., None], current_gate, old_gate)
        values, scales = compress_complete_kpools(
            pool_k, pool_gate, compress_ape, pool_size=self.pool_size
        )
        packed = pack_pooled_index_rows(values[:, 0], scales[:, 0])
        pooled_pad = pooled_rows.shape[0] - 1
        pooled_targets = torch.where(
            complete_mask,
            compressed_slots,
            torch.full_like(compressed_slots, pooled_pad),
        )
        packed = torch.where(complete_mask[:, None], packed, torch.zeros_like(packed))
        pooled_rows.index_copy_(0, pooled_targets, packed)

        # Only the last occurrence of each circular slot may survive.  Comparing
        # with row i+kpool removes older repetitions without scatter reductions
        # or duplicate index_copy writes (both problematic in HPU graphs).
        keep = tail_slots >= 0
        if n > self.pool_size:
            later_same = (
                tail_slots[:-self.pool_size] == tail_slots[self.pool_size:]
            ) & (
                positions[:-self.pool_size] + self.pool_size
                == positions[self.pool_size:]
            )
            keep[:-self.pool_size] &= ~later_same
        raw = torch.cat((k, gate_score), dim=-1).to(tail_rows.dtype)
        tail_pad = tail_rows.shape[0] - 1
        tail_targets = torch.where(
            keep,
            tail_slots,
            torch.full_like(tail_slots, tail_pad),
        )
        raw = torch.where(keep[:, None], raw, torch.zeros_like(raw))
        tail_rows.index_copy_(0, tail_targets, raw)

    def _exact_causal_physical(
        self,
        block_table: torch.Tensor,
        sequence_lengths: torch.Tensor,
        source_block_size: int,
    ) -> torch.Tensor:
        """Return every causal token directly when the context fits top-k."""

        width = self.topk_tokens + self.pool_size - 1
        logical = torch.arange(
            width, device=sequence_lengths.device, dtype=torch.long
        ).unsqueeze(0).expand(sequence_lengths.shape[0], -1)
        logical = torch.where(
            logical < sequence_lengths.long().unsqueeze(1),
            logical,
            torch.full_like(logical, -1),
        )
        return logical_to_physical_slots(
            logical, block_table, source_block_size
        ).to(torch.int32)

    def _scan_pool_bucket(
        self,
        block_columns: int,
        source_block_size: int,
        max_sequence_length: int,
        max_pool_cap: int | None,
    ) -> int:
        """Host-side kpool scan width for one scorer call."""
        max_pool_bucket = block_columns * source_block_size // self.pool_size
        if self.dynamic_pool_bucket:
            # The block table is allocated for the 1M logical ceiling, but
            # scoring every capacity row for a short request wastes dozens of
            # chunks and retains their asynchronous temporaries.  MLA itself is
            # eager on Gaudi2.  Use the scheduler's host-side context bound so
            # MTP graph capture never synchronizes a device scalar to Python.
            max_pool_bucket = self._active_pool_bucket_for_length(
                max_sequence_length,
                self.pool_size,
                self.context_chunk_pools,
                max_pool_bucket,
            )
        if max_pool_cap is not None:
            max_pool_bucket = min(max_pool_bucket, max(0, int(max_pool_cap)))
        return max_pool_bucket

    def canonical_shared_scan(
        self,
        max_sequence_length: int,
        max_pool_cap: int | None,
    ) -> tuple[int, int] | None:
        """``(bucket, block_columns)`` independent of the table's width.

        The bucket is the dynamic tier capped by ``max_pool_cap``; the
        columns cover every scanned pool and every token below
        ``max_sequence_length``, so slicing a wider table to them leaves
        ``_scan_pool_bucket`` and every referenced block unchanged.  Live
        prefill has ``bucket >= complete pools``, so columns depend on the
        bucket alone.  ``None`` without dynamic bucketing.
        """
        if not self.dynamic_pool_bucket or self.tp_sharded_kpool or self.compiled_kpool:
            return None
        source_block_size = int(
            getattr(self.k_cache, "hpu_kernel_block_size", self.source_block_size)
        )
        bucket = self._scan_pool_bucket(
            2**31, source_block_size, max_sequence_length, max_pool_cap
        )
        covered_tokens = max(self.pool_size * (bucket + 1), int(max_sequence_length))
        columns = -(-covered_tokens // source_block_size)
        return bucket, columns

    def shared_scan_fits_one_chunk(
        self,
        *,
        rows: int,
        bucket: int,
        index_heads: int,
    ) -> bool:
        """True when a shared ``rows``-query tile scores ``bucket`` in ONE chunk.

        One chunk means no intra-tile ``mark_step(sync=True)`` and no
        pre-scan clone fence in ``_write_topk``; the HBM bound is the same
        ``HPU_GLM53_SHARED_KPOOL_CHUNK_BUDGET_MIB`` workspace model.
        """
        chunk = self._scoring_chunk(
            batch=rows, index_heads=index_heads, index_dim=INDEX_DIM,
            packed_row_bytes=int(self.k_cache.kv_cache.shape[-1]),
            active_pool_bucket=bucket, shared_block_table=True,
        )
        return 0 < bucket <= chunk

    @_r589_stage("indexer.score")
    def _write_topk(
        self,
        row_slice: slice,
        q_values: torch.Tensor,
        weights: torch.Tensor,
        block_table: torch.Tensor,
        sequence_lengths: torch.Tensor,
        max_pool_cap: int | None = None,
        max_sequence_length: int | None = None,
        shared_block_table: bool = False,
    ) -> torch.Tensor:
        if shared_block_table and (self.tp_sharded_kpool or self.compiled_kpool):
            raise ValueError("shared kpool requires replicated, uncompiled scoring")
        count = row_slice.stop - row_slice.start
        if count <= 0:
            return torch.empty(
                (0, self.topk_tokens + self.pool_size - 1),
                device=q_values.device,
                dtype=torch.int32,
            )
        source_block_size = int(
            getattr(self.k_cache, "hpu_kernel_block_size", self.source_block_size)
        )
        if max_sequence_length is None:
            max_sequence_length = (
                int(sequence_lengths.long().max().item())
                if sequence_lengths.numel()
                else 0
            )
        record_capture_branch(max_sequence_length, self.topk_tokens, self.pool_size)
        if max_sequence_length <= self.topk_tokens:
            # The newest upstream GLM path uses the same exact-causal shortcut:
            # no scorer is needed when sparse top-k would retain every token.
            # Returning this tensor directly also avoids hidden mutable-buffer
            # communication across HPU graph boundaries.
            physical = self._exact_causal_physical(
                block_table,
                sequence_lengths[:count],
                source_block_size,
            )
            width = physical.shape[1]
            self.topk_indices_buffer[row_slice, :width].copy_(physical)
            return physical
        max_pool_bucket = record_capture_scalar("scan_bucket", self._scan_pool_bucket(
            int(block_table.shape[1]), source_block_size,
            max_sequence_length, max_pool_cap,
        ))
        scoring_chunk_pools = self._scoring_chunk(
            batch=count,
            index_heads=int(q_values.shape[1]),
            index_dim=int(q_values.shape[-1]),
            packed_row_bytes=int(self.k_cache.kv_cache.shape[-1]),
            active_pool_bucket=max_pool_bucket,
            shared_block_table=shared_block_table,
        )
        score_q = q_values[:count]
        score_weights = weights[:count]
        score_block_table = block_table
        score_lengths = sequence_lengths[:count]
        if (
            max_pool_bucket > scoring_chunk_pools
            and score_q.device.type == "hpu"
            and os.getenv("PT_HPU_LAZY_MODE", "0") == "1"
            and (not mixed_graph_capture_active() or capture_tile_marks())
        ):
            # These small inputs are reused after every segmented scorer step.
            # First materialize their producer graph.  clone() invokes Habana's
            # collective-consumption hook, which is unsafe when a graph view's
            # HbLazyTensorImpl has already expired; that was the native
            # TryGetHbLazyTensor/__dynamic_cast crash seen under long-context
            # concurrency.  Then clone and commit independent storage once.
            # The multi-GiB persistent packed cache intentionally remains shared.
            import habana_frameworks.torch.core as htcore

            htcore.mark_step(sync=lazy_fence_sync())
            score_q = score_q.clone()
            score_weights = score_weights.clone()
            score_block_table = score_block_table.clone()
            score_lengths = score_lengths.clone()
            htcore.mark_step(sync=lazy_fence_sync())
        if self.tp_sharded_kpool:
            if max_pool_bucket >= 2**24:
                raise RuntimeError(
                    "TP-sharded kpool candidate IDs exceed exact float32 collective range"
                )
            candidate_scores, candidate_ids = streaming_pooled_topk_candidates(
                score_q,
                score_weights,
                self.k_cache.kv_cache,
                score_block_table,
                score_lengths,
                source_block_size=source_block_size,
                pool_size=self.pool_size,
                topk_tokens=self.topk_tokens,
                context_chunk_pools=scoring_chunk_pools,
                max_complete_pools_bucket=max_pool_bucket,
                tp_rank=self.tp_rank,
                tp_size=self.tp_size,
            )
            # One compact collective per sparse layer. Pool IDs remain exact in
            # float32 through 67M source tokens, well above the 10M target.
            from vllm.distributed import tensor_model_parallel_all_gather

            payload = torch.stack(
                (candidate_scores, candidate_ids.float()), dim=1
            )
            payload = tensor_model_parallel_all_gather(payload, dim=-1)
            logical, physical = finalize_pooled_topk(
                payload[:, 0],
                payload[:, 1].round().long(),
                score_block_table,
                score_lengths,
                source_block_size=source_block_size,
                pool_size=self.pool_size,
                topk_tokens=self.topk_tokens,
            )
        elif self.compiled_kpool and not mixed_graph_capture_active():
            compile_key = (
                *q_values[:count].shape,
                *block_table.shape,
                str(q_values.dtype),
                source_block_size,
                self.pool_size,
                self.topk_tokens,
                scoring_chunk_pools,
                max_pool_bucket,
            )
            logical, physical = _get_compiled_streaming_pooled_topk(compile_key)(
                score_q,
                score_weights,
                self.k_cache.kv_cache,
                score_block_table,
                score_lengths,
                source_block_size,
                self.pool_size,
                self.topk_tokens,
                scoring_chunk_pools,
                max_pool_bucket,
            )
        else:
            logical, physical = streaming_pooled_topk(
                score_q,
                score_weights,
                self.k_cache.kv_cache,
                score_block_table,
                score_lengths,
                source_block_size=source_block_size,
                pool_size=self.pool_size,
                topk_tokens=self.topk_tokens,
                context_chunk_pools=scoring_chunk_pools,
                max_complete_pools_bucket=max_pool_bucket,
                shared_block_table=shared_block_table,
            )
        width = physical.shape[1]
        self._trace_physical_topk(
            logical,
            block_table,
            sequence_lengths,
            max_sequence_length=max_sequence_length,
        )
        self.topk_indices_buffer[row_slice, :width].copy_(physical)
        return physical

    def _drain_first_dynamic_pool(self) -> None:
        """Retire the first scorer graph before compiling the following mHC op.

        In HPU eager mode the initial kpool scorer can still own asynchronous
        workspace when the first mHC Sinkhorn recipe starts compiling.  That
        caused a generic Synapse compile failure despite the same Sinkhorn
        graph compiling in isolation.  Drain once per worker process and
        release only cached allocator blocks; steady-state decode remains
        fully asynchronous.
        """

        global _FIRST_DYNAMIC_KPOOL_DRAINED
        if (
            not self.dynamic_pool_bucket
            or not self.drain_first_dynamic_pool
            or _FIRST_DYNAMIC_KPOOL_DRAINED
        ):
            return
        if mixed_graph_capture_active():
            return
        torch.hpu.synchronize()
        # Gaudi software 1.23 exposes allocator statistics but no CUDA-style
        # ``empty_cache`` API. Synchronization is the correctness boundary we
        # need here; release cached blocks only on bridge versions that expose
        # the optional method.
        empty_cache = getattr(torch.hpu, "empty_cache", None)
        if empty_cache is not None:
            empty_cache()
        _FIRST_DYNAMIC_KPOOL_DRAINED = True

    @eager_break_during_capture
    @_r589_stage("indexer")
    def forward(
        self,
        hidden_states: torch.Tensor,
        q_values: torch.Tensor,
        k: torch.Tensor,
        weights: torch.Tensor,
        *,
        gate_score: torch.Tensor,
        compress_ape: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        from glm53_mixed_step import enabled as mixed_enabled
        if mixed_enabled():
            metadata_all = get_forward_context().attn_metadata
            if getattr(metadata_all, "mixed_step", False):
                from glm53_mixed_step import regional_indexer
                return regional_indexer(self, metadata_all, hidden_states, q_values, k,
                                        weights, gate_score, compress_ape, positions)
        from glm53_r590_long_history import LegacyRows, enabled as _r590_enabled, indexer_forward, legacy_route_scan
        if _r590_enabled() and min(q_values.shape[0], self.topk_indices_buffer.shape[0]) > 128:
            return indexer_forward(self, q_values, k, weights, gate_score=gate_score,
                compress_ape=compress_ape, positions=positions,
                metadata_all=get_forward_context().attn_metadata)
        del hidden_states
        metadata_all = get_forward_context().attn_metadata
        if not isinstance(metadata_all, dict):
            metadata = metadata_all
            n_buffer = min(q_values.shape[0], self.topk_indices_buffer.shape[0])
            self.topk_indices_buffer[:n_buffer].fill_(-1)
            if n_buffer <= 0:
                return torch.empty(
                    (0, self.topk_tokens + self.pool_size - 1),
                    device=q_values.device,
                    dtype=torch.int32,
                )
            compressed_slots = getattr(metadata, "glm53_index_slot_mapping", None)
            tail_slots = getattr(metadata, "glm53_tail_slot_mapping", None)
            block_table = getattr(metadata, "glm53_index_block_table", None)
            token_mask = getattr(metadata, "glm53_token_mask", None)
            if any(
                value is None
                for value in (compressed_slots, tail_slots, block_table, token_mask)
            ):
                raise RuntimeError("GLM-5.3 sparse metadata is incomplete")
            self._update_pooled_cache(
                k[:n_buffer],
                gate_score[:n_buffer],
                positions[:n_buffer],
                compressed_slots[:n_buffer],
                tail_slots[:n_buffer],
                compress_ape,
            )
            num_reqs = block_table.shape[0]
            if num_reqs <= 0 or n_buffer % num_reqs:
                raise RuntimeError("GLM-5.3 sparse token bucket is not rectangular")
            tokens_per_req = n_buffer // num_reqs
            query_block_table = torch.repeat_interleave(
                block_table,
                tokens_per_req,
                dim=0,
                output_size=n_buffer,
            )
            valid = token_mask[:n_buffer].reshape(-1).bool()
            sequence_lengths = torch.where(
                valid,
                positions[:n_buffer].long() + 1,
                torch.zeros_like(positions[:n_buffer], dtype=torch.long),
            )
            # HPU_GLM53_QUERY_SHARD_DECODE_KEYS: a listed decode key's capture scores
            # its own rows of this scan and all-gathers the rest; anything else is
            # exactly self._write_topk (glm53_r590_long_history.legacy_route_scan).
            physical = legacy_route_scan(
                self,
                num_reqs,
                LegacyRows(
                    slice(0, n_buffer),
                    q_values[:n_buffer],
                    weights[:n_buffer],
                    query_block_table,
                    sequence_lengths,
                ),
                # This legacy metadata route is used by the HPU bucket-grid
                # warmup.  It exercises 16K through 1M dummy context shapes;
                # actually scoring every dummy pool wastes HBM and compiles an
                # enormous eager graph even though MLA itself is kept eager.
                # The fixed-width top-k output still warms every downstream
                # attention shape.  Real hybrid serving uses the dict route
                # below and remains uncapped.
                # This HPU metadata route also carries every real request.
                # Limiting it to the 64-pool profile budget silently reduced
                # the learned 2,048-token window to logical tokens 0..255.
                # Dynamic bucketing already bounds real scoring to active context.
                max_pool_cap=(
                    self.warmup_max_kpools
                    if os.getenv("HPU_GLM53_CAP_LEGACY_WARMUP", "0") == "1" and not key_capture_active()
                    else None
                ),
                # This route is also captured inside the built-in MTP graph.
                # Never synchronize an HPU sequence-length tensor back to
                # Python there.  The runner constructs the block table as a
                # power-of-two tier around the active context, so its static
                # width is a safe upper bound and the intended graph key.
                max_sequence_length=(
                    query_block_table.shape[1]
                    * int(
                        getattr(
                            self.k_cache,
                            "hpu_kernel_block_size",
                            self.source_block_size,
                        )
                    )
                ),
            )
            # Keep prompt index selection out of the following MLA/o_proj
            # lazy segment. At max hot allocation the combined segment needs
            # more workspace than remains in HBM. Decode stays unsplit.
            if (
                bool(getattr(metadata, "is_prompt", False))
                and q_values.device.type == "hpu"
                and os.getenv("PT_HPU_LAZY_MODE", "0") == "1"
            ):
                import habana_frameworks.torch.core as htcore

                htcore.mark_step()
            self._drain_first_dynamic_pool()
            return physical

        metadata = metadata_all[self.k_cache.prefix]
        tail_metadata = metadata_all[self.tail_cache.prefix]
        n_actual = int(metadata.num_decode_tokens + metadata.num_prefill_tokens)
        n_buffer = min(q_values.shape[0], self.topk_indices_buffer.shape[0])
        self.topk_indices_buffer[:n_buffer].fill_(-1)
        if n_actual <= 0:
            return torch.full(
                (n_buffer, self.topk_tokens + self.pool_size - 1),
                -1,
                device=q_values.device,
                dtype=torch.int32,
            )

        self._update_pooled_cache(
            k[:n_actual],
            gate_score[:n_actual],
            positions[:n_actual],
            metadata.slot_mapping,
            tail_metadata.slot_mapping,
            compress_ape,
        )

        # Decode rows are first in the flattened scheduler token order.  Native
        # built-in MTP can retain one block-table row per request; expand it to
        # one row per verification token without changing physical addressing.
        # HPU_GLM53_HOST_MAX_SEQ_LEN_V1
        # The metadata builder computes this bound on the scheduler/CPU side.
        # Keep it as a host scalar so sparse selection never synchronizes an
        # HPU sequence-length tensor merely to choose a stable scoring tier.
        host_max_seq_len = getattr(metadata, "max_seq_len", None)
        if host_max_seq_len is not None and not isinstance(host_max_seq_len, int):
            raise TypeError("sparse-index max_seq_len must be a host integer")

        result_segments: list[tuple[int, int, torch.Tensor]] = []
        if metadata.num_decode_tokens:
            decode = metadata.decode
            if decode is None:
                raise RuntimeError("decode tokens present without indexer metadata")
            decode_count = int(metadata.num_decode_tokens)
            decode_bt = decode.block_table
            if decode_bt.shape[0] != decode_count:
                decode_bt = torch.repeat_interleave(
                    decode_bt,
                    decode.decode_lens.long(),
                    dim=0,
                    output_size=decode_count,
                )
            decode_physical = self._write_topk(
                slice(0, decode_count),
                q_values[:decode_count],
                weights[:decode_count],
                decode_bt,
                positions[:decode_count].long() + 1,
                max_sequence_length=host_max_seq_len,
            )
            result_segments.append((0, decode_count, decode_physical))

        # Prefill chunks already bound peak work-space.  The HPU extension adds
        # query_to_seq so each query receives the correct request block-table
        # row even when preceding requests have zero complete pools.
        if metadata.num_prefill_tokens:
            if metadata.prefill is None:
                raise RuntimeError("prefill tokens present without indexer metadata")
            for chunk in metadata.prefill.chunks:
                start, end = int(chunk.token_start), int(chunk.token_end)
                if end <= start:
                    continue
                query_to_seq = getattr(chunk, "query_to_seq", None)
                if query_to_seq is None:
                    raise RuntimeError(
                        "HPU GLM-5.3 prefill metadata is missing query_to_seq"
                    )
                query_bt = chunk.block_table.index_select(
                    0, query_to_seq.long()
                )
                prefill_physical = self._write_topk(
                    slice(start, end),
                    q_values[start:end],
                    weights[start:end],
                    query_bt,
                    positions[start:end].long() + 1,
                    max_sequence_length=host_max_seq_len,
                )
                result_segments.append((start, end, prefill_physical))
        if (
            metadata.num_prefill_tokens
            and q_values.device.type == "hpu"
            and os.getenv("PT_HPU_LAZY_MODE", "0") == "1"
        ):
            import habana_frameworks.torch.core as htcore

            htcore.mark_step()
        self._drain_first_dynamic_pool()
        # Build the returned dependency out-of-place.  Intel documents in-place
        # view updates as unsupported inside HPU graphs; returning this tensor
        # makes selected-row ownership explicit while the legacy shared buffer
        # remains populated for current MTP reuse code.
        width = self.topk_tokens + self.pool_size - 1
        pieces: list[torch.Tensor] = []
        cursor = 0
        for start, end, physical in sorted(result_segments, key=lambda item: item[0]):
            if start < cursor:
                raise RuntimeError("overlapping GLM sparse indexer result segments")
            if start > cursor:
                pieces.append(
                    torch.full(
                        (start - cursor, width),
                        -1,
                        device=q_values.device,
                        dtype=torch.int32,
                    )
                )
            pieces.append(physical)
            cursor = end
        if cursor < n_buffer:
            pieces.append(
                torch.full(
                    (n_buffer - cursor, width),
                    -1,
                    device=q_values.device,
                    dtype=torch.int32,
                )
            )
        if not pieces:
            return torch.full(
                (n_buffer, width),
                -1,
                device=q_values.device,
                dtype=torch.int32,
            )
        return torch.cat(pieces, dim=0)


def _flatten_indexer_inputs(
    hidden_states: torch.Tensor,
    qr: torch.Tensor,
    positions: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Normalize prompt/decode graph layouts to the indexer's token-major ABI."""
    hidden_states = hidden_states.reshape(-1, hidden_states.shape[-1])
    qr = qr.reshape(-1, qr.shape[-1])
    positions = positions.reshape(-1)
    token_count = hidden_states.shape[0]
    if qr.shape[0] != token_count or positions.shape[0] != token_count:
        raise ValueError(
            "GLM-5.3 HPU indexer input token counts differ: "
            f"hidden={token_count}, qr={qr.shape[0]}, positions={positions.shape[0]}"
        )
    return hidden_states, qr, positions


class HPUGlm5NextIndexer(upstream_attention.Indexer):
    """Checkpoint-compatible GLM indexer using the HPU sparse implementation."""

    r590_prompt_bias_contract = "indexer_causal_positions_v1"


    is_hpu_glm5next = True

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        cache_config = kwargs.get("cache_config")
        if cache_config is None and len(args) >= 6:
            cache_config = args[5]
        if cache_config is None:
            raise ValueError("GLM-5.3 HPU sparse indexer requires cache_config")
        self.indexer_op = HPUGlm5NextSparseIndexerOp(
            k_cache=self.k_cache,
            tail_cache=self.tail_cache,
            topk_indices_buffer=self.topk_indices_buffer,
            source_block_size=cache_config.block_size,
            pool_size=self.index_kpool,
            topk_tokens=self.topk_tokens,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        qr: torch.Tensor,
        positions: torch.Tensor,
        rotary_emb,
    ) -> torch.Tensor:
        hidden_states, qr, positions = _flatten_indexer_inputs(
            hidden_states, qr, positions
        )
        q, _ = self.wq_b(qr)
        q = q.view(-1, self.n_head, self.head_dim)
        kw, _ = self.wk_weights_proj(hidden_states)
        k = kw[:, : self.head_dim]
        if getattr(self, "_wp_fp32", None) is None:
            self._wp_fp32 = (
                self.wk_weights_proj.weight.data[self.head_dim :, :]
                .t()
                .contiguous()
                .float()
            )
        weights = torch.mm(hidden_states.float(), self._wp_fp32)
        k = upstream_attention._fused_indexer_k_norm(
            k, self.k_norm.weight, self.k_norm.bias, self.head_dim, self.k_norm.eps
        )

        if self.rope_dim > 0:
            q_pe, q_nope = torch.split(
                q, [self.rope_dim, self.head_dim - self.rope_dim], dim=-1
            )
            k_pe, k_nope = torch.split(
                k, [self.rope_dim, self.head_dim - self.rope_dim], dim=-1
            )
            q_pe, k_pe = rotary_emb(positions, q_pe, k_pe.unsqueeze(1))
            q = torch.cat(
                (q_pe.reshape(-1, self.n_head, self.rope_dim), q_nope), dim=-1
            )
            k = torch.cat(
                (k_pe.reshape(-1, 1, self.rope_dim).squeeze(1), k_nope), dim=-1
            )

        rotated = fwht128(q)
        q_values, q_scale = quantize_index_fp8(rotated)
        weights = upstream_attention._fused_indexer_weight_scale(
            weights,
            q_scale,
            self.softmax_scale * self.n_head**-0.5,
        )
        gate_score = F.linear(hidden_states, self.index_kpool_compress_gate)
        return self.indexer_op(
            hidden_states,
            q_values,
            k,
            weights,
            gate_score=gate_score,
            compress_ape=self.index_kpool_compress_ape,
            positions=positions,
        )


__all__ = [
    "HPUGlm5NextIndexer",
    "HPUGlm5NextIndexerBackend",
    "HPUGlm5NextIndexerCache",
    "HPUGlm5NextSparseIndexerOp",
    "bind_hpu_index_cache",
]
