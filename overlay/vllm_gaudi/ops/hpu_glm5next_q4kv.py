# SPDX-License-Identifier: Apache-2.0
"""Packed Q4 MLA KV-cache primitives for GLM-5.3 on HPU.

The 512-dimensional NoPE content latent is stored as signed Q4 in groups of
32 with one compact 1/16-octave log scale per group.  GLM-5.3 Flash sets
``mla_use_nope=true`` and ``qk_rope_head_dim=0``; persisting a fabricated
64-dimensional rotary tail would waste memory and reject the model's real
512-wide cache writes.  The complete row therefore occupies 272 bytes:

* 256 bytes of packed Q4 latent values;
* 16 bytes of latent scale exponents.

Only selected sparse rows are unpacked.  A full long-context BF16 or FP8 copy
is never materialized.  Large prefill writes may also end the current lazy
execution segment after each sparse MLA layer.  Decode writes remain in the
champion whole-model graph; the split exists only to keep the first long
prefill recipe from accumulating every layer's quantization workspace.
"""

from __future__ import annotations
from glm53_r589_bounded_prefill import stage as _r589_stage

import logging
import os
from collections.abc import Callable

import torch

KV_LORA_RANK = 512
LATENT_GROUP_SIZE = 32
LATENT_GROUPS = KV_LORA_RANK // LATENT_GROUP_SIZE
PACKED_LATENT_BYTES = KV_LORA_RANK // 2
Q4_MLA_ROW_BYTES = PACKED_LATENT_BYTES + LATENT_GROUPS
LOG_SCALE_BIAS = 128
LOG_SCALE_STEPS_PER_OCTAVE = 16

_COMPILED_Q4_PACK: dict[tuple, object] = {}
_COMPILED_Q4_UNPACK: dict[tuple, object] = {}
GATHER_GROUP_ENV = "HPU_GLM53_Q4_GATHER_GROUP_ROWS"
GATHER_BUDGET_ENV = "HPU_GLM53_Q4_GATHER_BUDGET_MIB"
# Mirrors hpu_glm5next_owner_mla.OWNER_MLA_ENV (that module imports this one).
OWNER_COMPUTE_ENV = "HPU_GLM53_OWNER_COMPUTE_Q4KV"
SHARED_OWNER_ENV = "HPU_GLM53_Q4_SHARED_GATHER_OWNER"
LOG = logging.getLogger("vllm.glm53.q4_gather")
_SHARED_OWNERS: dict[tuple, torch.Tensor] = {}


def shared_owner_enabled() -> bool:
    """C574 switch. Unset or 0 keeps a fresh owner per Q4 all-gather; any other value refuses the boot."""
    value = os.getenv(SHARED_OWNER_ENV, "0").strip()
    if value not in ("0", "1"):
        raise ValueError(
            f"{SHARED_OWNER_ENV}={value!r}: set 1 to reuse one Q4 gather owner per stripe shape inside "
            "mixed HPU graphs, or 0 / unset for a fresh owner per gather"
        )
    return value == "1"


def shared_owner_bytes(shape: tuple[int, ...], dtype: torch.dtype = torch.uint8) -> int:
    """HBM one rank-major owner of ``shape`` and ``dtype`` holds (the Q4 stripes are uint8, the decode shard's selected ids int32)."""
    count = 1
    for size in shape:
        count *= size
    return count * torch.empty((), dtype=dtype).element_size()


def shared_gather_owner(local_rows: torch.Tensor, tp_size: int) -> torch.Tensor | None:
    """The one owner every captured all-gather of this stripe shape writes, or None for a fresh owner.

    A captured HPU graph keeps each collective output resident until the next replay, so the
    (16,512,512) mixed graph's 44 tile gathers hold 44 owners (~3 GiB/rank). The owner is scratch:
    each gather overwrites it whole before the reorder reads it, and the next gather of the shape
    waits for the next tile's stripe, which the in-order compute stream produces after this tile's
    reorder. Native lazy steps keep their fresh owners. A decode graph is captured by the adapter's
    own wrapper, not by a mixed capture, so its stream's capture state counts too (step 21b): only
    the tensor the gather writes changes, every mark_step stays where it is.
    """
    if not shared_owner_enabled():
        return None
    from glm53_mixed_graph import capture_active

    if not capture_active() and not _stream_capturing(local_rows):
        return None
    shape = (tp_size * local_rows.shape[0], *local_rows.shape[1:])
    key = (shape, local_rows.dtype, str(local_rows.device))
    owner = _SHARED_OWNERS.get(key)
    if owner is None:
        owner = torch.empty(shape, dtype=local_rows.dtype, device=local_rows.device)
        _SHARED_OWNERS[key] = owner
        LOG.warning("GLM53_Q4_SHARED_OWNER shape=%s bytes=%d owners=%d", shape,
                    shared_owner_bytes(shape, local_rows.dtype), len(_SHARED_OWNERS))
    return owner


def _announce_shared_owner() -> None:
    """One boot line with the flag on; each owner it then allocates in a capture prints GLM53_Q4_SHARED_OWNER with its shape (the live proof)."""
    if os.getenv(SHARED_OWNER_ENV, "0").strip() == "1":
        LOG.warning("GLM53_Q4_SHARED_GATHER_OWNER armed scope=captured_graphs live_line=GLM53_Q4_SHARED_OWNER")


_announce_shared_owner()


def _stream_capturing(local_rows: torch.Tensor) -> bool:
    """Whether the current HPU stream is capturing a graph. Only an HPU tensor can be inside one."""
    if local_rows.device.type != "hpu":
        return False
    from habana_frameworks.torch.hpu.graphs import is_current_stream_capturing

    return bool(is_current_stream_capturing())


def canonical_stripe_rows(owner: torch.Tensor, tp_size: int, start: int, end: int) -> torch.Tensor:
    """Query rows ``[start, end)`` of a rank-major owner in canonical order: column ``c * TP + r``
    is rank ``r``'s stripe column ``c``."""
    compact_width, row_bytes = owner.shape[1], owner.shape[2]
    tile = owner.reshape(tp_size, -1, compact_width, row_bytes)[:, start:end]
    return tile.permute(1, 2, 0, 3).contiguous().reshape(
        end - start, compact_width * tp_size, row_bytes
    )


def _is_positive_int(value: str) -> bool:
    return value.isascii() and value.isdigit() and int(value) > 0


def _parse_gather_group_rows(value: str | None) -> int | None:
    """Rows per grouped Q4 selected-row all-gather; unset keeps one per tile."""
    if value is None:
        return None
    if not _is_positive_int(value) or int(value) % 128:
        raise ValueError(
            f"{GATHER_GROUP_ENV}={value!r} must be a positive multiple of 128 "
            "(query rows per Q4 all-gather); unset it for one gather per tile"
        )
    return int(value)


def _parse_gather_budget_bytes(value: str | None) -> int | None:
    """Per-rank HBM budget (bytes) for one grouped gather; unset means no cap."""
    if value is None:
        return None
    if not _is_positive_int(value):
        raise ValueError(
            f"{GATHER_BUDGET_ENV}={value!r} must be a positive integer MiB "
            "(per-rank HBM for one grouped Q4 all-gather); unset it for no cap"
        )
    return int(value) * 2**20


# Parsed once at import so a malformed value fails at boot, never mid-prefill.
Q4_GATHER_GROUP_ROWS = _parse_gather_group_rows(os.environ.get(GATHER_GROUP_ENV))
Q4_GATHER_BUDGET_BYTES = _parse_gather_budget_bytes(os.environ.get(GATHER_BUDGET_ENV))
_LOGGED_FALLBACKS: set[tuple[str, int, int]] = set()  # (setting, rows, tile) warned once
_USE_BF16_UNPACK = os.getenv(
    "HPU_GLM53_Q4_BF16_UNPACK", "0"
).lower() in ("1", "true", "yes", "on")


def _use_isolated_hpu_recipe() -> bool:
    """Use a nested torch.compile recipe only outside lazy HPU graphs.

    vLLM-Gaudi serves this model with ``PT_HPU_LAZY_MODE=1`` and wraps the
    complete model in an HPU graph.  Nesting a separately compiled Q4 leaf in
    that graph made the first 1,024-token replay request a 12.16 GiB workspace,
    while the same pack and cache writes captured directly by the outer graph
    need only about 48 MiB.  Eager-pipeline deployments without an outer lazy
    graph may still opt into the isolated recipe explicitly.
    """

    override = os.getenv("HPU_GLM53_Q4_ISOLATED_COMPILE")
    if override is not None:
        return override.lower() in ("1", "true", "yes", "on")
    return os.getenv("PT_HPU_LAZY_MODE", "0") != "1"


def prewarm_q4_mla_pack(device: torch.device) -> None:
    """Compile fixed Q4 row recipes before model weights consume HBM.

    The HPU compiler needs temporary workspace the first time each static row
    shape is seen.  Compiling those leaves before the checkpoint is resident
    prevents a valid Q4 recipe from failing because model weights have already
    consumed the compiler's workspace.
    """

    if not _use_isolated_hpu_recipe():
        return

    rows_text = os.getenv(
        "HPU_GLM53_Q4_PACK_WARM_ROWS",
        "1,2,4,8,16,32,64,128,256,512,1024,2048",
    )
    rows = sorted({int(value) for value in rows_text.split(",") if value})
    for row_count in rows:
        if row_count <= 0:
            raise ValueError("Q4 pack warmup rows must be positive")
        source = torch.zeros(
            (row_count, KV_LORA_RANK),
            dtype=torch.bfloat16,
            device=device,
        )
        packed = pack_q4_mla_rows(source)
        del source, packed
        torch.hpu.synchronize()


def tp_local_q4_slots(
    global_slots: torch.Tensor,
    *,
    block_size: int,
    tp_rank: int,
    tp_size: int,
    token_cyclic: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Map striped global cache rows to this TP rank's compact row space.

    Block-cyclic is the production layout. Token-cyclic is the owner-compute
    layout: every four-token sparse-index pool is balanced across TP4, so each
    rank consumes a fixed one-quarter stripe without moving KV rows. ``owned``
    is false for negative padding rows in either layout.
    """

    if block_size <= 0:
        raise ValueError("block_size must be positive")
    if tp_size <= 0 or not 0 <= tp_rank < tp_size:
        raise ValueError("tp_rank must be in [0, tp_size)")
    slots = global_slots.long()
    valid = slots >= 0
    safe = slots.clamp_min(0)
    if token_cyclic:
        owned = valid & (torch.remainder(safe, tp_size) == tp_rank)
        local = torch.div(safe, tp_size, rounding_mode="floor")
        return local, owned
    global_block = torch.div(safe, block_size, rounding_mode="floor")
    owned = valid & (torch.remainder(global_block, tp_size) == tp_rank)
    local_block = torch.div(global_block, tp_size, rounding_mode="floor")
    local = local_block * block_size + torch.remainder(safe, block_size)
    return local, owned


def _compact_log_scale(
    x: torch.Tensor,
    quant_max: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return a 1/16-octave scale and its compact uint8 log code."""

    absmax = x.float().abs().amax(dim=-1, keepdim=True)
    log_scale = torch.log2(absmax.clamp_min(2.0**-8) / quant_max)
    encoded = torch.round(log_scale * LOG_SCALE_STEPS_PER_OCTAVE)
    encoded = (encoded + LOG_SCALE_BIAS).clamp(0, 255).to(torch.uint8)
    decoded_log = (
        encoded.float() - LOG_SCALE_BIAS
    ) / LOG_SCALE_STEPS_PER_OCTAVE
    scale = torch.exp2(decoded_log)
    return scale, encoded


def _pack_q4_mla_rows_impl(x: torch.Tensor) -> torch.Tensor:
    """Pack ``[..., 512]`` BF16/FP16 MLA rows into ``[..., 272]`` uint8."""

    if x.shape[-1] != KV_LORA_RANK:
        raise ValueError(
            f"Q4 MLA expects {KV_LORA_RANK} values, got {x.shape[-1]}"
        )

    latent = x[..., :KV_LORA_RANK].reshape(
        *x.shape[:-1], LATENT_GROUPS, LATENT_GROUP_SIZE
    )
    latent_scale, latent_exp = _compact_log_scale(latent, 7)
    q4 = torch.round(latent.float() / latent_scale).clamp(-7, 7)
    q4 = (q4.to(torch.int16) + 8).to(torch.uint8)
    packed = q4[..., 0::2] | (q4[..., 1::2] << 4)
    packed = packed.reshape(*x.shape[:-1], PACKED_LATENT_BYTES)

    return torch.cat((packed, latent_exp.squeeze(-1)), dim=-1)


def _unpack_q4_mla_rows_impl(
    packed: torch.Tensor,
    dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """Unpack only the supplied ``[..., 272]`` rows to ``[..., 512]``."""

    if packed.dtype != torch.uint8 or packed.shape[-1] != Q4_MLA_ROW_BYTES:
        raise ValueError(
            f"packed Q4 MLA rows must be uint8 [...,{Q4_MLA_ROW_BYTES}]"
        )

    from glm53_mla_unpack import ENABLED as native_unpack_enabled, unpack as native_unpack
    if (native_unpack_enabled and _USE_BF16_UNPACK
            and dtype == torch.bfloat16 and packed.device.type == "hpu"):
        from glm53_mixed_graph import witness_lever
        witness_lever("mla_unpack")
        return native_unpack(packed)

    q4_bytes = packed[..., :PACKED_LATENT_BYTES]
    lo = q4_bytes & 0x0F
    hi = q4_bytes >> 4
    q4 = torch.stack((lo, hi), dim=-1).reshape(
        *packed.shape[:-1], LATENT_GROUPS, LATENT_GROUP_SIZE
    )
    exp_start = PACKED_LATENT_BYTES
    exp_end = exp_start + LATENT_GROUPS
    if _USE_BF16_UNPACK and dtype == torch.bfloat16:
        # Q4 is already the dominant approximation. Keeping the temporary
        # nibble and scale expansion in BF16 removes the persistent FP32
        # selected-row workspace before Gaudi's BF16 MLA GEMM. The numerical
        # delta from the original FP32-then-cast path is bounded by one BF16
        # ULP and is separately guarded by the live coherence matrix.
        q4 = q4.to(torch.bfloat16) - 8
        latent_exp = (
            packed[..., exp_start:exp_end].to(torch.bfloat16)
            - LOG_SCALE_BIAS
        ) / LOG_SCALE_STEPS_PER_OCTAVE
        latent_scale = torch.exp2(latent_exp).unsqueeze(-1)
        return (q4 * latent_scale).reshape(
            *packed.shape[:-1], KV_LORA_RANK
        )

    q4 = q4.to(torch.int16).float() - 8.0
    latent_exp = (
        packed[..., exp_start:exp_end].float() - LOG_SCALE_BIAS
    ) / LOG_SCALE_STEPS_PER_OCTAVE
    latent_scale = torch.exp2(latent_exp).unsqueeze(-1)
    latent = (q4 * latent_scale).reshape(*packed.shape[:-1], KV_LORA_RANK)
    return latent.to(dtype)


def _compiled_q4_leaf(table: dict, key: tuple, function):
    """Return a shape-keyed HPU recipe isolated from the full MLA graph."""

    compiled = table.get(key)
    if compiled is None:
        torch._dynamo.config.recompile_limit = max(
            int(torch._dynamo.config.recompile_limit), 256
        )
        if hasattr(torch._dynamo.config, "accumulated_recompile_limit"):
            torch._dynamo.config.accumulated_recompile_limit = max(
                int(torch._dynamo.config.accumulated_recompile_limit), 512
            )
        compiled = torch.compile(function, backend="hpu_backend", dynamic=False)
        table[key] = compiled
    return compiled


def pack_q4_mla_rows(x: torch.Tensor) -> torch.Tensor:
    """Pack an MLA row behind an independent fixed-shape HPU recipe."""

    pack = _pack_q4_mla_rows_impl
    if x.device.type == "hpu" and _use_isolated_hpu_recipe():
        key = (*x.shape, str(x.dtype))
        pack = _compiled_q4_leaf(_COMPILED_Q4_PACK, key, _pack_q4_mla_rows_impl)
    return pack(x)


def unpack_q4_mla_rows(
    packed: torch.Tensor,
    dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """Unpack selected rows behind an independent fixed-shape HPU recipe."""

    unpack = _unpack_q4_mla_rows_impl
    if packed.device.type == "hpu" and _use_isolated_hpu_recipe():
        key = (*packed.shape, str(packed.dtype), str(dtype))
        unpack = _compiled_q4_leaf(
            _COMPILED_Q4_UNPACK, key, _unpack_q4_mla_rows_impl
        )
    return unpack(packed, dtype=dtype)


class HPUQ4MLAKVCache(torch.nn.Module):
    """Graph-friendly packed store and bounded selected-row unpack."""

    def __init__(self) -> None:
        super().__init__()
        # Populated only by the model runner at the authoritative allocation
        # boundary. Owner-compute refuses to infer cache identity or layout
        # from a request-time tensor.
        self.owner_mla_capability = None
        self.owner_mla_max_reduce = None
        self.owner_mla_sum_reduce_scatter = None
        self.tp_sharded = os.getenv(
            "HPU_GLM53_TP_SHARDED_Q4KV", "0"
        ).lower() in ("1", "true", "yes", "on")
        self.token_cyclic = os.getenv(
            "HPU_GLM53_Q4_TOKEN_CYCLIC", "0"
        ).lower() in ("1", "true", "yes", "on")
        self.owner_layout = (
            "TOKEN_CYCLIC_OWNER" if self.token_cyclic else "BLOCK_CYCLIC_OWNER"
        )
        self.block_size = int(os.getenv("HPU_GLM53_Q4_BLOCK_SIZE", "128"))
        self.prefill_segment_min_rows = int(
            os.getenv("HPU_GLM53_Q4_PREFILL_SEGMENT_MIN_ROWS", "0")
        )
        if self.prefill_segment_min_rows < 0:
            raise ValueError(
                "HPU_GLM53_Q4_PREFILL_SEGMENT_MIN_ROWS must be non-negative"
            )

    def bind_owner_mla_capability(
        self,
        capability,
        *,
        max_reduce=None,
        sum_reduce_scatter=None,
    ) -> None:
        """Bind allocator proof and optional collective test doubles."""

        self.owner_mla_capability = capability
        self.owner_mla_max_reduce = max_reduce
        self.owner_mla_sum_reduce_scatter = sum_reduce_scatter

    @staticmethod
    def _tp_coordinates() -> tuple[int, int]:
        from vllm.distributed import (
            get_tensor_model_parallel_rank,
            get_tensor_model_parallel_world_size,
        )

        return (
            get_tensor_model_parallel_rank(),
            get_tensor_model_parallel_world_size(),
        )

    def forward(
        self,
        input: torch.Tensor | None,
        cache: torch.Tensor,
        slot_mapping: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        del kwargs
        if input is not None:
            packed = pack_q4_mla_rows(input)
            if self.tp_sharded:
                tp_rank, tp_size = self._tp_coordinates()
                local_slots, owned = tp_local_q4_slots(
                    slot_mapping,
                    block_size=self.block_size,
                    tp_rank=tp_rank,
                    tp_size=tp_size,
                    token_cyclic=self.token_cyclic,
                )
                # The final cache row is an allocation-owned dummy.  Unowned
                # writes are redirected there and zeroed; no rank persists a
                # remote block and the live cache remains exactly 1/TP sized.
                dummy = cache.shape[0] - 1
                write_slots = torch.where(
                    owned, local_slots, torch.full_like(local_slots, dummy)
                )
                packed = torch.where(
                    owned.unsqueeze(-1), packed, torch.zeros_like(packed)
                )
                cache.index_copy_(0, write_slots, packed)
            else:
                from glm53_mixed_graph import enabled as mixed_graphs_enabled, write_q4_rows
                if mixed_graphs_enabled():
                    write_q4_rows(cache, slot_mapping, packed)
                else:
                    cache.index_copy_(0, slot_mapping, packed)

            # A combined 11-layer Q4 prefill recipe requested 12.16 GiB/rank.
            # End large-prefill write segments at the configured threshold;
            # decode and small writes retain their whole-model graph.
            if (
                self.prefill_segment_min_rows > 0
                and input.device.type == "hpu"
                and os.getenv("PT_HPU_LAZY_MODE", "0") == "1"
                and input.shape[0] >= self.prefill_segment_min_rows
            ):
                import habana_frameworks.torch.core as htcore

                htcore.mark_step()
        return cache

    @_r589_stage("q4.fetch")
    def fetch_selected(
        self,
        cache: torch.Tensor,
        global_slots: torch.Tensor,
        *,
        dtype: torch.dtype = torch.bfloat16,
    ) -> torch.Tensor:
        """Fetch selected global rows and unpack them.

        Block-cyclic storage retains the exact BF16 all-reduce path. Token-cyclic
        storage exploits the selector four-token pool contract: rank r owns
        columns r::TP. Each rank gathers only its packed uint8 stripe, HCCL
        all-gathers those stripes once, and the result is interleaved back into
        canonical selected-token order before the unchanged dequant/attention
        path. The growing cache is never replicated.
        """

        if not self.tp_sharded:
            rows = cache.index_select(
                0, global_slots.long().clamp_min(0).reshape(-1)
            ).reshape(*global_slots.shape, cache.shape[-1])
            return self.dequant_output(rows, dtype=dtype)

        tp_rank, tp_size = self._tp_coordinates()
        if self.token_cyclic:
            selected_width = global_slots.shape[1]
            local_rows = self._owned_stripes(cache, global_slots, tp_rank, tp_size)
            owner = shared_gather_owner(local_rows, tp_size) if tp_size > 1 else None
            if owner is not None:
                _all_gather_into(owner, local_rows.contiguous(), commit=False)
                rows = canonical_stripe_rows(owner, tp_size, 0, global_slots.shape[0])
            elif tp_size > 1:
                from vllm.distributed import get_tp_group

                gathered = get_tp_group().all_gather(
                    local_rows.contiguous(), dim=1
                )
                compact_width = local_rows.shape[1]
                rows = gathered.reshape(
                    global_slots.shape[0],
                    tp_size,
                    compact_width,
                    cache.shape[-1],
                ).permute(0, 2, 1, 3).contiguous().reshape(
                    global_slots.shape[0],
                    compact_width * tp_size,
                    cache.shape[-1],
                )
            else:
                rows = local_rows
            return self.dequant_output(
                rows[:, :selected_width], dtype=dtype
            )

        local_slots, owned = tp_local_q4_slots(
            global_slots,
            block_size=self.block_size,
            tp_rank=tp_rank,
            tp_size=tp_size,
        )
        dummy = cache.shape[0] - 1
        local_slots = torch.where(
            owned,
            local_slots.clamp(0, dummy),
            torch.full_like(local_slots, dummy),
        )
        rows = cache.index_select(0, local_slots.reshape(-1)).reshape(
            *global_slots.shape, cache.shape[-1]
        )
        if tp_size > 1:
            from vllm.distributed import tensor_model_parallel_all_reduce

            rows = tensor_model_parallel_all_reduce(rows.to(torch.bfloat16))
            rows = rows.to(torch.uint8)
        return self.dequant_output(rows, dtype=dtype)

    def _owned_stripes(
        self,
        cache: torch.Tensor,
        global_slots: torch.Tensor,
        tp_rank: int,
        tp_size: int,
    ) -> torch.Tensor:
        """This rank's packed token-cyclic stripe ``[B, ceil(K/TP), 272]``.

        Rank ``r`` owns selected columns ``r::TP`` (four-token pool
        contract); padding and unowned rows are zero.
        """
        selected_width = global_slots.shape[1]
        remainder = selected_width % tp_size
        if remainder:
            padded_slots = torch.cat(
                (
                    global_slots,
                    torch.full(
                        (global_slots.shape[0], tp_size - remainder),
                        -1,
                        dtype=global_slots.dtype,
                        device=global_slots.device,
                    ),
                ),
                dim=1,
            )
        else:
            padded_slots = global_slots
        striped_slots = padded_slots[:, tp_rank::tp_size]
        local_slots, owned = tp_local_q4_slots(
            striped_slots,
            block_size=self.block_size,
            tp_rank=tp_rank,
            tp_size=tp_size,
            token_cyclic=True,
        )
        if (
            global_slots.device.type == "cpu"
            and not torch.equal(owned, striped_slots >= 0)
        ):
            raise RuntimeError(
                "token-cyclic selector columns do not match TP ownership"
            )
        dummy = cache.shape[0] - 1
        gather_slots = torch.where(
            owned,
            local_slots.clamp(0, dummy),
            torch.full_like(local_slots, dummy),
        )
        local_rows = cache.index_select(
            0, gather_slots.reshape(-1)
        ).reshape(*striped_slots.shape, cache.shape[-1])
        return torch.where(
            owned.unsqueeze(-1), local_rows, torch.zeros_like(local_rows)
        )

    @_r589_stage("q4.fetch")
    def gather_selected_stripes(
        self,
        cache: torch.Tensor,
        global_slots: torch.Tensor,
    ) -> torch.Tensor:
        """ONE all-gather of every rank's stripe for a group of query rows.

        Returns an owned ``[TP * B, ceil(K/TP), 272]`` buffer (rank-major,
        exactly ``all_gather_into_tensor``'s output) that is materialized
        before any tile reads it; tiles build their views from this owner.
        Carries the per-tile path's ``q4.fetch`` stage label.
        """
        tp_rank, tp_size = self._tp_coordinates()
        local_rows = self._owned_stripes(cache, global_slots, tp_rank, tp_size)
        if tp_size == 1:
            return local_rows
        owner = shared_gather_owner(local_rows, tp_size)
        if owner is not None:
            return _all_gather_into(owner, local_rows.contiguous(), commit=True)
        return _all_gather_owned(local_rows.contiguous(), tp_size)

    def selected_from_stripes(
        self,
        owner: torch.Tensor,
        rows: tuple[int, int],
        selected_width: int,
        dtype: torch.dtype = torch.bfloat16,
    ) -> torch.Tensor:
        """Canonical selected rows ``[start, end)`` of a gathered group.

        Same bytes as the per-tile path: column ``c * TP + r`` is rank
        ``r``'s stripe column ``c``.
        """
        start, end = rows
        packed = canonical_stripe_rows(owner, self._tp_coordinates()[1], start, end)
        return self.dequant_output(packed[:, :selected_width], dtype=dtype)

    def dequant_output(
        self,
        output: torch.Tensor,
        dtype: torch.dtype = torch.bfloat16,
    ) -> torch.Tensor:
        return unpack_q4_mla_rows(output, dtype=dtype)

    def fetch_from_cache(
        self,
        cache: torch.Tensor,
        blocks: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        del kwargs
        return self.fetch_selected(cache, blocks)


def _all_gather_owned(local_rows: torch.Tensor, tp_size: int) -> torch.Tensor:
    """Rank-major all-gather into a fresh owned tensor (no view returned)."""
    output = torch.empty(
        (tp_size * local_rows.shape[0], *local_rows.shape[1:]),
        dtype=local_rows.dtype,
        device=local_rows.device,
    )
    return _all_gather_into(output, local_rows, commit=True)


@_r589_stage("tp.all_gather")
def _all_gather_into(output: torch.Tensor, local_rows: torch.Tensor, *, commit: bool) -> torch.Tensor:
    """Rank-major all-gather of ``local_rows`` into ``output``.

    Same submission contract as HpuCommunicator.all_gather: commit the
    producer graph, run HCCL, finish the segmented collective; ``commit``
    adds the synchronous commit grouped owners need before later tile recipes
    read a materialized input.  Bypasses HpuCommunicator, so it carries that
    method's ``tp.all_gather`` stage label.
    """
    import torch.distributed as dist
    from vllm.distributed import get_tp_group

    on_hpu = local_rows.device.type == "hpu"
    if on_hpu:
        import habana_frameworks.torch.core as htcore

        htcore.mark_step()
    dist.all_gather_into_tensor(output, local_rows, group=get_tp_group().device_group)
    if on_hpu:
        from vllm_gaudi.distributed.device_communicators.hpu_communicator import (
            _finish_segmented_lazy_collective,
        )

        _finish_segmented_lazy_collective()
        if commit:
            from glm53_mixed_graph import lazy_fence
            lazy_fence(htcore)
    return output


def q4_gather_bytes_per_row(selected_width: int, tp_size: int) -> int:
    """Per-rank HBM one query row costs a grouped gather.

    The owned ``[TP, ceil(K/TP), 272]`` buffer plus two transient local
    stripes.  Pure arithmetic on host integers.
    """
    stripe_bytes = -(-selected_width // tp_size) * Q4_MLA_ROW_BYTES
    return (tp_size + 2) * stripe_bytes


def _gather_rows_limit(selected_width: int | None) -> tuple[int, str]:
    """Group-row ceiling and the env setting behind it (the flag, or the tighter HBM budget).

    The budget is converted with host integers only (selected width and TP
    world size).  It must never probe free HBM: that differs per rank, and a
    per-rank group size breaks TP collective lockstep.
    """
    flag_setting = f"{GATHER_GROUP_ENV}={Q4_GATHER_GROUP_ROWS}"
    if Q4_GATHER_BUDGET_BYTES is None:
        return Q4_GATHER_GROUP_ROWS, flag_setting
    if selected_width is None:
        raise ValueError(
            f"{GATHER_BUDGET_ENV} needs selected_width (the selected-row count, "
            "slots.shape[1]) to size a group; pass it to q4_gather_group_rows"
        )
    from vllm.distributed import get_tensor_model_parallel_world_size

    tp_size = get_tensor_model_parallel_world_size()
    budget_rows = Q4_GATHER_BUDGET_BYTES // q4_gather_bytes_per_row(selected_width, tp_size)
    if budget_rows < Q4_GATHER_GROUP_ROWS:
        return budget_rows, f"{GATHER_BUDGET_ENV}={Q4_GATHER_BUDGET_BYTES // 2**20}"
    return Q4_GATHER_GROUP_ROWS, flag_setting


def _log_grouping_disabled_once(setting: str, limit: int, tile_rows: int) -> None:
    key = (setting, limit, tile_rows)
    if key in _LOGGED_FALLBACKS:
        return
    _LOGGED_FALLBACKS.add(key)
    LOG.warning(
        "Q4 grouped all-gather disabled: %s allows %d rows per group, not above the %d-row "
        "MLA tile, so every tile keeps its own all-gather. Raise it (512 rows is recommended "
        "for 128-row tiles) or unset it.",
        setting, limit, tile_rows,
    )


def q4_gather_group_rows(
    cache_op: HPUQ4MLAKVCache,
    query_rows: int,
    tile_rows: int,
    selected_width: int | None = None,
) -> int | None:
    """Rows per grouped all-gather for this prefill, or ``None`` for legacy.

    Only the token-cyclic TP-sharded fetch (owner compute off) qualifies.
    The group is the largest multiple of ``tile_rows`` (and above one tile)
    within the ceiling that divides ``query_rows``; ``None`` otherwise.  The
    ceiling is HPU_GLM53_Q4_GATHER_GROUP_ROWS, tightened by
    HPU_GLM53_Q4_GATHER_BUDGET_MIB when set (which needs ``selected_width``).
    Host integers only, so every TP rank derives the same plan and issues
    identical collectives.
    """
    if (Q4_GATHER_GROUP_ROWS is None or not cache_op.tp_sharded
            or not cache_op.token_cyclic or os.getenv(OWNER_COMPUTE_ENV) == "1"):
        return None
    limit, setting = _gather_rows_limit(selected_width)
    if limit <= tile_rows:
        _log_grouping_disabled_once(setting, limit, tile_rows)
        return None
    ceiling = min(limit, query_rows)
    for rows in range(ceiling // tile_rows * tile_rows, tile_rows, -tile_rows):
        if query_rows % rows == 0:
            return rows
    return None


def grouped_q4_sparse_mla(
    cache_op: HPUQ4MLAKVCache,
    local_cache: Callable[[], torch.Tensor],
    latent_q: torch.Tensor,
    q_pe: torch.Tensor,
    *,
    selection: tuple[torch.Tensor, torch.Tensor],
    rows: tuple[int, int],
    kv_lora_rank: int,
    scale: float,
    tile_boundary: Callable[[], None],
) -> torch.Tensor:
    """Per-tile sparse MLA with one Q4 all-gather per ``group_rows``.

    ``selection`` is ``(slots, valid)``; ``rows`` is ``(tile_rows,
    group_rows)``.  ``local_cache`` re-creates the cache view per group.
    Tiles, their order, ``tile_boundary`` placement and every tile's bytes
    match the per-tile path; only the gather is hoisted.  Peak HBM holds one
    owner buffer: ``selected`` is dropped before each tile boundary and
    ``owner`` after each group, so the next gather never allocates beside it.
    """
    from vllm_gaudi.ops.hpu_glm5next_kpool import sparse_mla_from_selected

    slots, valid = selection
    tile_rows, group_rows = rows
    total, width = latent_q.shape[0], slots.shape[1]
    if total % group_rows or group_rows % tile_rows:
        raise ValueError("grouped Q4 gather needs tile | group | query rows; use q4_gather_group_rows")
    tiles = []
    for group in range(0, total, group_rows):
        owner = cache_op.gather_selected_stripes(local_cache(), slots[group:group + group_rows])
        for start in range(group, group + group_rows, tile_rows):
            end = start + tile_rows
            selected = cache_op.selected_from_stripes(
                owner, (start - group, end - group), width, dtype=latent_q.dtype)
            tiles.append(sparse_mla_from_selected(
                latent_q[start:end], q_pe[start:end], selected, valid[start:end],
                kv_lora_rank=kv_lora_rank, scale=scale))
            del selected
            if end < total:
                tile_boundary()
        del owner
    return torch.cat(tiles, dim=0)
