from glm53_r589_bounded_prefill import stage as _r589_stage
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch
import torch.nn as nn
import os

from vllm.config import get_current_vllm_config
from vllm.model_executor.custom_op import PluggableLayer
from vllm.model_executor.layers.attention import MLAAttention
from vllm.model_executor.layers.mla import MultiHeadLatentAttentionWrapper
from vllm.model_executor.utils import replace_parameter
from vllm.models.common.ops import fused_q_kv_rmsnorm
from vllm_gaudi.extension.utils import VLLMKVCache
from vllm_gaudi.extension.utils import (FP8Matmul, Matmul, B2BMatmul, ModuleFusedSDPA, Softmax, VLLMFP8KVCache)
from vllm_gaudi.attention.backends.hpu_attn import HPUMLAMetadata
import vllm_gaudi.extension.kernels as kernels
from vllm.forward_context import ForwardContext, get_forward_context
from vllm.logger import init_logger
from vllm_gaudi.ops.hpu_glm5next_kpool import sparse_mla_from_selected
from vllm_gaudi.ops.hpu_glm5next_q4kv import (
    HPUQ4MLAKVCache, grouped_q4_sparse_mla, q4_gather_group_rows)
from vllm_gaudi.ops.hpu_glm5next_owner_mla import q4_sparse_mla_from_slots


logger = init_logger(__name__)


def _trace_glm53_mla_tensor(layer_name: str, call_id: int, name: str,
                            value: torch.Tensor) -> None:
    """Synchronizing statistics for the opt-in one-layer MLA diagnostic."""
    data = value.detach().cpu().reshape(-1).float()
    if data.numel() == 0:
        logger.warning(
            "GLM53_TRACE_MLA_INTERNAL layer=%s call=%d name=%s shape=%s empty=1",
            layer_name, call_id, name, tuple(value.shape))
        return
    finite = torch.isfinite(data)
    safe = torch.where(finite, data, torch.zeros_like(data))
    logger.warning(
        "GLM53_TRACE_MLA_INTERNAL layer=%s call=%d name=%s shape=%s "
        "dtype=%s finite=%d/%d rms=%.9g absmax=%.9g first=%s",
        layer_name, call_id, name, tuple(value.shape), value.dtype,
        int(finite.sum().item()), data.numel(),
        float(safe.square().mean().sqrt().item()),
        float(safe.abs().max().item()), safe[:8].tolist())


class _DummyPrefillBackend:
    """No-op MLA prefill backend for HPU (which has its own attention impl)."""

    def __init__(self, **kwargs):
        pass


class HPUMLAAttention(MLAAttention):

    def r590_prompt_bias_contract(self):
        from glm53_r590_long_history import selected_mla_capability
        return selected_mla_capability(self)


    scale: float

    def __init__(self, *args, **kwargs):
        import vllm.model_executor.layers.attention.mla_attention as mla_mod
        original_get_prefill = mla_mod.get_mla_prefill_backend
        mla_mod.get_mla_prefill_backend = lambda vllm_config: _DummyPrefillBackend
        try:
            super().__init__(*args, **kwargs)
        finally:
            mla_mod.get_mla_prefill_backend = original_get_prefill
        self.enable_fp8_attn = self.kv_cache_dtype == 'fp8_inc' and os.environ.get('QUANT_CONFIG', None) is None
        self.enable_q4_kv = self.kv_cache_dtype == 'int4_per_token_head'
        self.scale = float(self.scale)
        self.matmul_qk = Matmul() if not self.enable_fp8_attn \
            else FP8Matmul()
        self.softmax = Softmax()
        self.matmul_av = Matmul() if not self.enable_fp8_attn \
            else FP8Matmul()
        self.batch2block_matmul = B2BMatmul() if not self.enable_fp8_attn \
            else FP8Matmul()
        self.block2batch_matmul = B2BMatmul() if not self.enable_fp8_attn \
            else FP8Matmul()
        self.k_cache = HPUQ4MLAKVCache() if self.enable_q4_kv else (
            VLLMKVCache() if not self.enable_fp8_attn else VLLMFP8KVCache())
        self.v_cache = VLLMKVCache(is_v_cache=True) if not self.enable_fp8_attn \
            else VLLMFP8KVCache()
        HPUFusedSDPA = kernels.fsdpa()
        self.fused_scaled_dot_product_attention = None if HPUFusedSDPA is None \
            else ModuleFusedSDPA(HPUFusedSDPA)
        self._glm53_trace_mla_internal = os.environ.get(
            "HPU_GLM53_TRACE_MLA_INTERNAL", "0").lower() in (
                "1", "true", "yes", "on")
        self._glm53_trace_mla_internal_calls = 0
        self._glm53_bounded_sparse_width = os.environ.get(
            "HPU_GLM53_BOUNDED_SPARSE_WIDTH", "0"
        ).lower() in ("1", "true", "yes", "on")
        self._glm53_bounded_sparse_width_logged = False
        sparse_mla_width_quantum = os.environ.get(
            "HPU_SPARSE_MLA_WIDTH_QUANTUM", "128"
        )
        try:
            self._sparse_mla_width_quantum = int(sparse_mla_width_quantum)
        except ValueError as exc:
            raise ValueError(
                "HPU_SPARSE_MLA_WIDTH_QUANTUM must be 64 or 128, got "
                f"{sparse_mla_width_quantum!r}"
            ) from exc
        if self._sparse_mla_width_quantum not in (64, 128):
            raise ValueError(
                "HPU_SPARSE_MLA_WIDTH_QUANTUM must be 64 or 128, got "
                f"{self._sparse_mla_width_quantum}"
            )

    def bind_kv_cache(self, kv_cache) -> None:
        """Store the HPU per-layer KV-cache tuple unchanged.

        vllm#51718 changed ``vllm.v1.worker.utils.bind_kv_cache`` to delegate
        binding to each layer's own ``bind_kv_cache`` and added
        ``MLAAttention.bind_kv_cache`` whose body is
        ``self.kv_cache = kv_cache.squeeze(1)`` — it assumes a single packed
        ``[B, H=1, N, C]`` tensor. On HPU the model runner allocates a per-layer
        ``(key_cache, value_cache, key_scales, value_scales)`` tuple for MLA
        layers and ``forward_impl`` consumes it directly (``kv_cache[0]``), so
        the upstream ``.squeeze`` raises ``AttributeError: 'tuple' object has no
        attribute 'squeeze'``. Keep the tuple as-is for the HPU path.
        """
        self.kv_cache = kv_cache

    def forward(
        self,
        q: torch.Tensor,
        kv_c_normed: torch.Tensor,
        k_pe: torch.Tensor,
        output_shape: torch.Size | None = None,
        q_dcp_replicated: torch.Tensor | None = None,
        hpu_sparse_slots: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # `q_dcp_replicated` was added to the base MLAAttention.forward signature
        # by vllm#45964 (DCP query replication for MLA decode). Decode-context
        # parallelism is not enabled on HPU, so the inherited wrapper forward
        # always passes this as None; accept and ignore it to keep the HPU path
        # behaviourally identical to pre-#45964.
        del q_dcp_replicated
        # NOTE: vllm#49389 removed the deprecated runtime KV-scale calculation
        # path (the `calculate_kv_scales` attribute and the
        # `maybe_calc_kv_scales` custom op). Neither exists at the pinned vLLM
        # SHA, so the OOT MLA forward no longer attempts runtime scale
        # calculation.

        if self.use_direct_call:
            forward_context: ForwardContext = get_forward_context()
            attn_metadata = forward_context.attn_metadata
            if isinstance(attn_metadata, dict):
                attn_metadata = attn_metadata[self.layer_name]
            self_kv_cache = self.kv_cache
            #slot_mapping = forward_context.slot_mapping

            #assert isinstance(slot_mapping, dict), (
            #    f"Expected slot_mapping to be a dict, got {type(slot_mapping)}. "
            #)
            #self.impl.do_kv_cache_update(
            #    kv_c_normed,
            #    k_pe,
            #    self_kv_cache,
            #    slot_mapping.get(self.layer_name),
            #    self.kv_cache_dtype,
            #    self._k_scale,
            #)
            return self.forward_impl(
                q,
                kv_c_normed,
                k_pe,
                self_kv_cache,
                attn_metadata,
                hpu_sparse_slots=hpu_sparse_slots,
            )
        else:
            if hpu_sparse_slots is not None:
                raise RuntimeError(
                    "HPU GLM sparse slots require the direct MLA call path"
                )
            kv_cache_dummy_dep = torch.ops.vllm.unified_mla_kv_cache_update(
                kv_c_normed,
                k_pe,
                self.layer_name,
                self.kv_cache_dtype,
                self._k_scale,
            )
            output = torch.empty(output_shape, dtype=q.dtype, device=q.device)
            torch.ops.vllm.unified_mla_attention_with_output(
                q,
                kv_c_normed,
                k_pe,
                output,
                self.layer_name,
                kv_cache_dummy_dep=kv_cache_dummy_dep,
            )
            return output

    @_r589_stage("mla.selected")
    def forward_impl(
        self,
        q: torch.Tensor,
        k_c_normed: torch.Tensor,
        k_pe: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: "HPUMLAMetadata",
        output: torch.Tensor | None = None,
        output_scale: torch.Tensor | None = None,
        output_block_scale: torch.Tensor | None = None,
        hpu_sparse_slots: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if getattr(attn_metadata, "mixed_step", False):
            from glm53_mixed_step import regional_mla
            return regional_mla(self, attn_metadata, q, k_c_normed, k_pe,
                                kv_cache, hpu_sparse_slots)
        from glm53_r590_long_history import guard_mla_fallback
        _r590_slots = hpu_sparse_slots
        if _r590_slots is None:
            _r590_slots = getattr(self, "hpu_sparse_topk_indices_buffer", None)
        guard_mla_fallback(attn_metadata, _r590_slots, q.shape[0])
        is_prefill = attn_metadata.is_prompt

        slot_mapping = attn_metadata.slot_mapping.flatten() if attn_metadata.slot_mapping is not None else None

        latent_vec_k = torch.concat((k_c_normed, k_pe.view(*k_c_normed.shape[:-1], self.qk_rope_head_dim)), dim=-1)
        latent_vec_k = latent_vec_k.view(-1, self.qk_rope_head_dim + self.kv_lora_rank)

        # Write the latent and rope to KV cache, and retain the returned tensor
        # as an explicit dependency for the selected-row gather below.  Merely
        # mutating kv_cache[0] and discarding the result is not reliably ordered
        # across HPU lazy/graph boundaries.
        latent_cache = kv_cache[0] if kv_cache is not None else None
        if kv_cache is not None and len(kv_cache) >= 2:
            # Always use impl-owned cache op so INC quantization maps to one canonical module path.
            latent_cache = self.impl.latent_cache_k(
                latent_vec_k, kv_cache[0], slot_mapping
            )

        # Consume the indexer's returned tensor directly.  The former path only
        # shared a mutable module buffer: the generic wrapper discarded the
        # indexer return value, so HPU lazy execution had no data dependency
        # connecting the buffer copy to this read.  Decode consequently saw
        # stale -1 or uninitialized padded rows and every sparse MLA layer could
        # collapse to zero.  Keep the attribute fallback for non-GLM callers.
        physical_slots = hpu_sparse_slots
        if physical_slots is None:
            physical_slots = getattr(
                self, "hpu_sparse_topk_indices_buffer", None
            )
        if self.enable_q4_kv and not is_prefill and physical_slots is None:
            raise RuntimeError(
                "Q4 MLA decode requires GLM sparse selected-row metadata; "
                "refusing an incompatible dense decode against packed 272-byte rows"
            )
        if physical_slots is not None:
            # GLM-5.3's indexer writes physical source rows. Gather only those
            # bounded rows, then dequantize the selected FP8 tensor; the full
            # 128K-1M latent cache is never expanded or fetched.
            q_nope, q_pe = q.split(
                [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1
            )
            latent_q = torch.bmm(
                q_nope.transpose(0, 1), self.W_UK_T
            ).transpose(0, 1)
            slots = physical_slots[:q.shape[0]]
            if self._glm53_bounded_sparse_width:
                sparse_block_table = getattr(
                    attn_metadata, "glm53_index_block_table", None
                )
                if (
                    isinstance(sparse_block_table, torch.Tensor)
                    and sparse_block_table.dim() >= 2
                ):
                    # The indexer returns a fixed 2,176-column buffer so the
                    # long-context top-k has one graph shape.  Bound selected
                    # row gather/unpack/attention by this bucket's live
                    # context plus its per-request query width.  In particular,
                    # a fresh prompt has a zero-column block table but still
                    # owns only one prompt chunk; treating zero as "unbounded"
                    # expanded all 2,176 padded rows and forced a multi-GiB
                    # Synapse workspace on every TP rank.
                    source_block_size = int(
                        getattr(attn_metadata, "block_size", 128) or 128
                    )
                    tile_width = self._sparse_mla_width_quantum
                    bucket_batch = max(1, int(sparse_block_table.shape[0]))
                    query_capacity = (
                        (int(q.shape[0]) + bucket_batch - 1) // bucket_batch
                        if is_prefill else 1
                    )
                    live_capacity = (
                        int(sparse_block_table.shape[-1]) * source_block_size
                        + query_capacity
                    )
                    bounded_width = min(
                        int(slots.shape[1]),
                        max(tile_width, (
                            (live_capacity + tile_width - 1) // tile_width
                        ) * tile_width),
                    )
                    if bounded_width < slots.shape[1]:
                        slots = slots[:, :bounded_width]
                        if not self._glm53_bounded_sparse_width_logged:
                            logger.warning(
                                "GLM53_BOUNDED_SPARSE_WIDTH layer=%s full=%d bounded=%d block_columns=%d",
                                self.layer_name, physical_slots.shape[1], bounded_width,
                                sparse_block_table.shape[-1])
                            self._glm53_bounded_sparse_width_logged = True
            valid = slots >= 0
            trace_this = (
                self._glm53_trace_mla_internal
                and "layers.3." in str(self.layer_name)
                and self._glm53_trace_mla_internal_calls < 2
            )
            trace_call = self._glm53_trace_mla_internal_calls
            if trace_this:
                self._glm53_trace_mla_internal_calls += 1
                logger.warning(
                    "GLM53_TRACE_MLA_INTERNAL layer=%s call=%d phase=%s "
                    "valid=%d/%d slots_first=%s",
                    self.layer_name, trace_call,
                    "prefill" if is_prefill else "decode",
                    int(valid.detach().cpu().sum().item()), valid.numel(),
                    slots.detach().cpu().reshape(-1)[:16].tolist())
                _trace_glm53_mla_tensor(
                    self.layer_name, trace_call, "q", q)
                _trace_glm53_mla_tensor(
                    self.layer_name, trace_call, "latent_q", latent_q)
            if latent_cache is None:
                raise RuntimeError("sparse MLA requires a bound latent cache")
            def fetch_rows(row_slots):
                # Recreate this view from the persistent cache owner inside
                # each segmented recipe.  Carrying flat_cache across a prior
                # synchronous tile boundary can leave a view whose lazy
                # tensor_data has already been released.
                flat_cache = latent_cache.reshape(
                    -1, latent_cache.shape[-1]
                )
                if self.enable_q4_kv:
                    fetched = self.impl.latent_cache_k.fetch_selected(
                        flat_cache, row_slots, dtype=q.dtype
                    )
                else:
                    fetched = flat_cache.index_select(
                        0, row_slots.long().clamp_min(0).reshape(-1)
                    ).reshape(
                        row_slots.shape[0], row_slots.shape[1],
                        flat_cache.shape[-1]
                    )
                if self.enable_fp8_attn:
                    fetched = self.impl.latent_cache_k.dequant_output(fetched)
                return fetched

            def attend_rows(query, rope_query, row_slots, row_valid):
                if self.enable_q4_kv:
                    return q4_sparse_mla_from_slots(
                        self.impl.latent_cache_k,
                        query,
                        rope_query,
                        latent_cache.reshape(-1, latent_cache.shape[-1]),
                        row_slots,
                        row_valid,
                        kv_lora_rank=self.kv_lora_rank,
                        scale=self.scale,
                    )
                selected_rows = fetch_rows(row_slots)
                if trace_this:
                    _trace_glm53_mla_tensor(
                        self.layer_name,
                        trace_call,
                        "selected",
                        selected_rows,
                    )
                return sparse_mla_from_selected(
                    query,
                    rope_query,
                    selected_rows,
                    row_valid,
                    kv_lora_rank=self.kv_lora_rank,
                    scale=self.scale,
                )

            # Keep the selected-cache fetch and Q4/FP8 unpack inside the same
            # static query tile as sparse attention.  Tiling only the score
            # BMM is too late: the full [query, selected, latent] expansion has
            # already entered the Synapse recipe.  This exact row partition is
            # numerically invariant because each query row has an independent
            # softmax over its unchanged selected-key axis.
            fetch_tile = int(os.getenv("HPU_SPARSE_MLA_FETCH_TILE", "0"))
            if (is_prefill and fetch_tile > 0
                    and slots.shape[1] == physical_slots.shape[1]):
                # HPU_GLM53_PREFILL_TILE_ROWS (default 128) widens this exact
                # row partition within HPU_GLM53_MLA_TILE_BUDGET_MIB; fewer
                # tiles means fewer syncs and fewer Q4 stripe all-gathers.
                # Full selected width only: the recipe then depends on the
                # query bucket alone, which the runner's bucket-grid warmup
                # (16K+ dummy contexts) already executes.
                from glm53_r590_long_history import (
                    mla_bytes_per_query_row, mla_prefill_tile_rows)
                fetch_tile = mla_prefill_tile_rows(
                    fetch_tile, int(slots.shape[0]), mla_bytes_per_query_row(
                        int(slots.shape[1]), int(latent_q.shape[1]), self.kv_lora_rank,
                        int(latent_cache.shape[-1]) * latent_cache.element_size()),
                )
            if is_prefill and fetch_tile > 0 and slots.shape[0] > fetch_tile:
                import habana_frameworks.torch.core as htcore
                from glm53_mixed_graph import lazy_fence

                query_rows = latent_q.shape[0]
                if not (
                    slots.shape[0] == query_rows
                    and valid.shape[0] == query_rows
                    and q_pe.shape[0] == query_rows
                ):
                    raise RuntimeError(
                        "sparse MLA query metadata row mismatch: "
                        f"q={query_rows} q_pe={q_pe.shape[0]} "
                        f"slots={slots.shape[0]} valid={valid.shape[0]}"
                    )
                # The selector/indexer and query projections can be views into
                # lazy graph outputs.  Materialize their producers *before*
                # cloning: clone() itself calls Habana handle_collective(), and
                # asking it to discover a collective through a stale graph view
                # can dereference an expired HbLazyTensorImpl.  This manifested
                # as a native __dynamic_cast/TryGetHbLazyTensor SIGSEGV under
                # concurrent long-context traffic.  The second marker commits
                # the clones to independent storage for reuse by every tile.
                # The persistent latent cache intentionally remains shared.
                lazy_fence(htcore)
                slots = slots.clone()
                valid = valid.clone()
                latent_q = latent_q.clone()
                q_pe = q_pe.clone()
                lazy_fence(htcore)
                # HPU_GLM53_Q4_GATHER_GROUP_ROWS hoists the per-tile Q4 stripe
                # all-gather to one per group (host-int plan, TP lockstep;
                # HPU_GLM53_Q4_GATHER_BUDGET_MIB caps it by selected width);
                # tiles, syncs and every tile's bytes are unchanged.
                group_rows = (
                    q4_gather_group_rows(self.impl.latent_cache_k, query_rows, fetch_tile,
                                         selected_width=int(slots.shape[1]))
                    if self.enable_q4_kv and not trace_this else None
                )
                output_tiles = []
                if group_rows is not None:
                    output_tiles.append(grouped_q4_sparse_mla(
                        self.impl.latent_cache_k,
                        lambda: latent_cache.reshape(-1, latent_cache.shape[-1]),
                        latent_q, q_pe, selection=(slots, valid),
                        rows=(fetch_tile, group_rows),
                        kv_lora_rank=self.kv_lora_rank, scale=self.scale,
                        tile_boundary=lambda: lazy_fence(htcore),
                    ))
                for start in range(0, query_rows if group_rows is None else 0, fetch_tile):
                    end = min(start + fetch_tile, query_rows)
                    output_tiles.append(attend_rows(
                        latent_q[start:end],
                        q_pe[start:end],
                        slots[start:end],
                        valid[start:end],
                    ))
                    # A Python loop alone remains one lazy Synapse recipe.  A
                    # synchronous step between tiles materializes only the
                    # small output tile and releases the selected-cache/Q4
                    # unpack workspace before the next tile is constructed.
                    if end < query_rows:
                        lazy_fence(htcore)
                output = torch.cat(output_tiles, dim=0)
            else:
                output = attend_rows(latent_q, q_pe, slots, valid)
            if trace_this:
                _trace_glm53_mla_tensor(
                    self.layer_name, trace_call, "latent_output", output)
            projected = self._v_up_proj(output)
            if trace_this:
                _trace_glm53_mla_tensor(
                    self.layer_name, trace_call, "projected_output", projected)
            return projected

        guard_mla_fallback(attn_metadata, None, q.shape[0])
        if is_prefill:
            output = self.impl.forward_mha(q, latent_vec_k, kv_cache, attn_metadata)
            return output
        else:
            # Dense MLA still needs the latent query projection.  Sparse MLA
            # computes its own projection above, so defer this BMM until the
            # dense fallback is actually selected instead of doing it twice.
            q_nope, q_pe = q.split(
                [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1
            )
            decode_ql_nope = torch.bmm(
                q_nope.transpose(0, 1), self.W_UK_T
            ).transpose(0, 1)
            output = self.impl.forward_mqa(
                decode_ql_nope, q_pe, kv_cache, attn_metadata
            )
            output = self._v_up_proj(output)
            return output
            # NOTE(Xinyu): Make the loaded weight contiguous to avoid the transpose

    # during each graph execution
    def process_weights_after_loading(self, act_dtype: torch.dtype):
        # HPU-specific: when VLLM_HPU_FORCE_CHANNEL_FP8=True (default), block-quantized
        # FP8 weights (e.g. kv_b_proj in DeepSeek-R1) are converted to channel-wise FP8.
        # After this conversion, weight_scale_inv becomes 1D [N_out] (per-channel) but
        # weight_block_size is not cleared. The upstream MLAAttention.process_weights_after_loading
        # then calls scaled_dequantize with group_shape=weight_block_size, which fails
        # because a 1D scale is incompatible with a 2D block group_shape.
        # We handle this by directly dequantizing kv_b_proj for the HPU path.
        kv_b_proj = self.kv_b_proj
        weight = kv_b_proj.weight
        weight_scale_inv = getattr(kv_b_proj, 'weight_scale_inv', None)

        if weight.dtype == torch.float8_e4m3fn and weight_scale_inv is not None:
            if weight_scale_inv.dim() == 1:
                # Channel-wise FP8 (produced by VLLM_HPU_FORCE_CHANNEL_FP8=True):
                # one scale per output channel; dequant via simple broadcast multiply.
                ws = weight_scale_inv.view(-1, 1).to(act_dtype)  # [N_out, 1]
                kv_b_proj_weight = (weight.to(act_dtype) * ws).T.contiguous()
            else:
                # Block FP8 (force_channel_fp8=False): use HPU block dequant.
                from vllm_gaudi.extension.ops import dequant_block_fp8_weight_naive
                orig_M = kv_b_proj.orig_M.item() if hasattr(kv_b_proj, 'orig_M') else None
                orig_N = kv_b_proj.orig_N.item() if hasattr(kv_b_proj, 'orig_N') else None
                kv_b_proj_weight = dequant_block_fp8_weight_naive(
                    weight.contiguous(),
                    weight_scale_inv,
                    kv_b_proj.weight_block_size,
                    dtype=act_dtype,
                    original_M=orig_M,
                    original_N=orig_N,
                    do_unpad=(orig_M is not None),
                ).T

            assert kv_b_proj_weight.shape == (
                self.kv_lora_rank,
                self.num_heads * (self.qk_nope_head_dim + self.v_head_dim),
            ), (f"{kv_b_proj_weight.shape=}, "
                f"{self.kv_lora_rank=}, {self.num_heads=}, "
                f"{self.qk_nope_head_dim=}, {self.v_head_dim=}")
            kv_b_proj_weight = kv_b_proj_weight.view(
                self.kv_lora_rank,
                self.num_heads,
                self.qk_nope_head_dim + self.v_head_dim,
            )
            W_UK, W_UV = kv_b_proj_weight.split([self.qk_nope_head_dim, self.v_head_dim], dim=-1)
            self.W_UV = W_UV.transpose(0, 1).contiguous()
            self.W_UK_T = W_UK.permute(1, 2, 0).contiguous()

            from vllm.model_executor.layers.attention.attention import (set_default_quant_scales,
                                                                        should_load_quant_weights)
            quant_method = (self.quant_config.get_quant_method(self, prefix=self.layer_name)
                            if self.quant_config else None)
            if not should_load_quant_weights(quant_method):
                set_default_quant_scales(self, register_buffer=False)
        else:
            # Non-FP8 kv_b_proj: use upstream logic as before.
            MLAAttention.process_weights_after_loading(self, act_dtype)
            # Since vllm#48251 base registers W_UV/W_UK_T as nn.Parameter (via
            # replace_parameter), so a plain tensor reassignment raises
            # TypeError. Route the contiguous() update through replace_parameter
            # to preserve the registration.
            replace_parameter(self, "W_UV", self.W_UV.contiguous())
            replace_parameter(self, "W_UK_T", self.W_UK_T.contiguous())

    # NOTE(Chendi): PR25184 using output buffer as default, which can't be used in HPU Graph,
    # so we override and always return a new tensor
    def _v_up_proj(self, x):
        # Convert from (B, N, L) to (N, B, L)
        x = x.view(-1, self.num_heads, self.kv_lora_rank).transpose(0, 1)
        # Multiply (N, B, L) x (N, L, V) -> (N, B, V)
        x = torch.bmm(x, self.W_UV)
        # Convert from (N, B, V) to (B, N * V)
        x = x.transpose(0, 1).reshape(-1, self.num_heads * self.v_head_dim)
        return x


@PluggableLayer.register_oot(name="MultiHeadLatentAttentionWrapper")
class HPUMultiHeadLatentAttentionWrapper(MultiHeadLatentAttentionWrapper):

    def r590_prompt_bias_contract(self):
        from glm53_r590_long_history import selected_wrapper_capability
        return selected_wrapper_capability(self)


    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        scale: float,
        qk_nope_head_dim: int,
        qk_rope_head_dim: int,
        v_head_dim: int,
        q_lora_rank: int | None,
        kv_lora_rank: int,
        mla_modules,
        cache_config=None,
        quant_config=None,
        prefix: str = "",
        skip_topk: bool = False,
        # Added upstream by vllm#50000 (Kimi K3). Gates non-causal multi-token
        # decode; the base MultiHeadLatentAttentionWrapper forwards it to
        # MLAAttention, which the HPU decode path honours via attn metadata.
        # Keep it in the constructor signature and pass it through so the HPU
        # wrapper stays in sync with the base / deepseek_v2 call site.
        non_causal_multi_token_decode: bool = False,
        # Added upstream by vllm#48407 (short-prefill sparse-indexer scoring
        # skip). The optimization it gates is guarded by
        # current_platform.is_cuda() upstream, so it never applies on HPU. We
        # accept-and-ignore the kwarg to keep the constructor signature in sync
        # with the base MultiHeadLatentAttentionWrapper / deepseek_v2 call site.
        allow_short_prefill_indexer_scoring_skip: bool = False,
        # The upstream implementation is a CUDA Triton kernel. Accept the new
        # ABI while keeping HPU on its two native RMSNorm operations.
        fuse_qkv_rmsnorm: bool = False,
    ) -> None:
        # Skip MultiHeadLatentAttentionWrapper.__init__() because it creates
        # MLAAttention → FlashAttnPrefillBackend which crashes on HPU.
        # Instead, inline the field assignments and create HPUMLAAttention.
        nn.Module.__init__(self)
        self.hidden_size = hidden_size
        self.qk_nope_head_dim = qk_nope_head_dim
        self.qk_rope_head_dim = qk_rope_head_dim
        self.qk_head_dim = qk_nope_head_dim + qk_rope_head_dim
        self.v_head_dim = v_head_dim
        self.q_lora_rank = q_lora_rank
        self.kv_lora_rank = kv_lora_rank
        self.num_heads = num_heads
        self.fused_qkv_a_proj = mla_modules.fused_qkv_a_proj
        self.kv_a_proj_with_mqa = mla_modules.kv_a_proj_with_mqa
        self.q_a_layernorm = mla_modules.q_a_layernorm
        self.q_b_proj = mla_modules.q_b_proj
        self.q_proj = mla_modules.q_proj
        self.kv_a_layernorm = mla_modules.kv_a_layernorm
        self.kv_b_proj = mla_modules.kv_b_proj
        self.rotary_emb = mla_modules.rotary_emb
        self.o_proj = mla_modules.o_proj
        self.indexer = mla_modules.indexer
        self.indexer_rope_emb = mla_modules.indexer_rotary_emb
        self.is_sparse = mla_modules.is_sparse
        self.fuse_qkv_rmsnorm = False
        # vllm#50000 (Kimi K3) added `self.g_proj = mla_modules.g_proj`, which
        # the base MultiHeadLatentAttentionWrapper.forward (inherited here, since
        # we do not override forward) reads to gate the attention output. Because
        # we bypass super().__init__(), replicate the assignment. It defaults to
        # None for DeepSeek-V2/R1 (no gate proj), leaving the HPU path unchanged.
        self.g_proj = mla_modules.g_proj

        # DSA sparse attention is not implemented on HPU: sparse layers run as
        # dense MLA and the indexer must never be invoked (its kernels and the
        # DeepseekV32IndexerBackend are CUDA-only).
        # Diagnostic/correctness escape hatch for GLM-5.3.  It deliberately
        # bypasses both the sparse indexer and selected-row gather while leaving
        # the checkpoint, KDA path, mHC, projections, and cache dtype unchanged.
        # This makes a dense run a clean oracle for the custom HPU sparse path.
        self.hpu_glm53_force_dense_mla = (
            os.environ.get("HPU_GLM53_FORCE_DENSE_MLA", "0") == "1"
        )
        self.hpu_glm5next_sparse = (
            not self.hpu_glm53_force_dense_mla
            and self.is_sparse
            and getattr(self.indexer, "is_hpu_glm5next", False)
        )
        self.skip_topk = (
            True
            if self.hpu_glm53_force_dense_mla
            else (
                skip_topk
                if self.hpu_glm5next_sparse
                else (skip_topk or self.is_sparse)
            )
        )
        # vllm#45964 (DCP query replication) added `self.dcp_q_replicate`, which
        # the base MultiHeadLatentAttentionWrapper.forward (inherited here, since
        # we do not override forward) reads and forwards to mla_attn. Because we
        # bypass super().__init__(), replicate the upstream assignment. DCP is
        # not enabled on HPU, so `qrep_active` is absent and this stays False.
        q_proj_layer = self.q_b_proj if self.q_lora_rank is not None else self.q_proj
        self.dcp_q_replicate = getattr(q_proj_layer, "qrep_active", False)
        if self.indexer is not None:
            assert hasattr(self.indexer, "topk_tokens")
            self.topk_tokens = self.indexer.topk_tokens
            self.topk_indices_buffer = mla_modules.topk_indices_buffer

        self.prefix = prefix

        layer_name = f"{prefix}.attn"
        static_ctx = get_current_vllm_config().compilation_config.static_forward_context
        static_ctx.pop(layer_name, None)
        self.mla_attn = HPUMLAAttention(
            num_heads=self.num_heads,
            scale=scale,
            qk_nope_head_dim=self.qk_nope_head_dim,
            qk_rope_head_dim=self.qk_rope_head_dim,
            v_head_dim=self.v_head_dim,
            q_lora_rank=self.q_lora_rank,
            kv_lora_rank=self.kv_lora_rank,
            cache_config=cache_config,
            quant_config=quant_config,
            prefix=layer_name,
            kv_b_proj=self.kv_b_proj,
            # Dense-MLA fallback: never request a sparse backend on HPU.
            use_sparse=False,
            indexer=self.indexer,
            non_causal_multi_token_decode=non_causal_multi_token_decode,
        )
        if self.hpu_glm5next_sparse:
            self.mla_attn.r590_selected_row_consumer = True
            self.mla_attn.hpu_sparse_topk_indices_buffer = (
                self.topk_indices_buffer)

    @_r589_stage("mla.model")
    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        llama_4_scaling: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Run MLA with an explicit indexer-to-attention data dependency.

        Upstream's generic wrapper invokes the sparse indexer only for its
        mutation of ``topk_indices_buffer`` and discards the returned tensor.
        That side-channel is not ordered reliably across HPU lazy/eager graph
        breaks.  Capturing and passing the exact-width result makes both the
        dependency and the valid selected-row extent explicit.
        """
        q_c = None
        if self.q_lora_rank is not None:
            assert self.fused_qkv_a_proj is not None
            assert self.q_a_layernorm is not None
            assert self.q_b_proj is not None
            qkv_lora = self.fused_qkv_a_proj(hidden_states)[0]
            q_c, kv_lora = qkv_lora.split(
                [self.q_lora_rank, self.kv_lora_rank + self.qk_rope_head_dim],
                dim=-1,
            )
            q_proj_layer = self.q_b_proj
            if not self.fuse_qkv_rmsnorm:
                q_c = self.q_a_layernorm(q_c)
            q_proj_input = q_c
        else:
            assert self.kv_a_proj_with_mqa is not None
            assert self.q_proj is not None
            kv_lora = self.kv_a_proj_with_mqa(hidden_states)[0]
            q_proj_layer = self.q_proj
            q_proj_input = hidden_states

        kv_c, k_pe = kv_lora.split(
            [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1
        )
        if self.fuse_qkv_rmsnorm and q_c is not None:
            assert self.q_a_layernorm is not None
            q_proj_input, kv_c_normed = fused_q_kv_rmsnorm(
                q_c,
                kv_c,
                self.q_a_layernorm.weight.data,
                self.kv_a_layernorm.weight.data,
                self.q_a_layernorm.variance_epsilon,
            )
            q_c = q_proj_input
        else:
            kv_c_normed = self.kv_a_layernorm(kv_c)
        k_pe = k_pe.unsqueeze(1)

        q = q_proj_layer(q_proj_input)[0]
        heads = self.num_heads
        if self.dcp_q_replicate:
            heads *= q_proj_layer.group_size
        q = q.view(-1, heads, self.qk_head_dim)
        if self.rotary_emb is not None:
            q[..., self.qk_nope_head_dim :], k_pe = self.rotary_emb(
                positions, q[..., self.qk_nope_head_dim :], k_pe
            )

        hpu_sparse_slots = None
        if self.indexer and self.is_sparse and not self.skip_topk:
            hpu_sparse_slots = self.indexer(
                hidden_states, q_c, positions, self.indexer_rope_emb
            )
        elif self.hpu_glm5next_sparse:
            hpu_sparse_slots = self.topk_indices_buffer
        if llama_4_scaling is not None:
            q *= llama_4_scaling

        q_dcp_replicated = None
        if self.dcp_q_replicate:
            q_dcp_replicated, q = q, q_proj_layer._local_view(q)

        attn_out = self.mla_attn(
            q,
            kv_c_normed,
            k_pe,
            output_shape=(
                hidden_states.shape[0], self.num_heads * self.v_head_dim
            ),
            q_dcp_replicated=q_dcp_replicated,
            hpu_sparse_slots=hpu_sparse_slots,
        )
        if self.g_proj is not None:
            attn_out = attn_out * self.g_proj(hidden_states)[0].sigmoid()
        return self.o_proj(attn_out)[0]
