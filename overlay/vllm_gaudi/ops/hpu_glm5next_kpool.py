# SPDX-License-Identifier: Apache-2.0
"""Portable HPU kpool primitives for GLM-5.3 sparse MLA indexing.

The persistent index cache contains one 128-byte FP8 key plus one FP32 scale
per four source tokens.  Raw per-token index keys are never retained.  These
functions are graph-friendly PyTorch references for the HPU implementation;
the same interfaces can later be replaced by fused HPU custom ops.
"""

from __future__ import annotations
from glm53_r589_bounded_prefill import stage as _r589_stage

import logging
import math
from glm53_r590_long_history import canonical_topk as _r590_topk
from glm53_r590_long_history import enabled as _r590_enabled
import os
from glm53_mixed_graph import (capture_active as mixed_graph_capture_active, capture_tile_marks, decode_key_capture_active, key_capture_active,
                               lazy_fence_sync, witness_lever)

import torch

_tpc_topk = None
if os.getenv("HPU_GLM53_INDEXER_TPC_TOPK", "0") != "0":
    import glm53_rc6_flags as _rc6_flags
    import glm53_tpc_topk as _tpc_topk
if os.getenv("HPU_GLM53_EMIT_WORDSKIP", "0") != "0":
    import glm53_mla_topkfuse  # noqa: F401  (a word-skip boot that cannot run stops here, at worker start, not at the first top-k)

INDEX_DIM = 128
# Gaudi2 executes torch.float8_e4m3fn with the FNUZ-compatible finite range.
# Values above 240 can become NaN on HPU even though CUDA E4M3FN reaches 448.
FP8_MAX = 240.0
_USE_BF16_SPARSE_MLA_SCORE = os.getenv(
    "HPU_GLM53_BF16_SPARSE_MLA_SCORE", "0"
).lower() in ("1", "true", "yes", "on")
_USE_BF16_KPOOL_SCORE = os.getenv(
    "HPU_GLM53_BF16_KPOOL_SCORE", "0"
).lower() in ("1", "true", "yes", "on")
_USE_NATIVE_FP8_KPOOL_SCORE = os.getenv(
    "HPU_GLM53_NATIVE_FP8_KPOOL_SCORE", "0"
).lower() in ("1", "true", "yes", "on")

# A blue/green launcher may inherit the previous image's BF16 flag through its
# captured environment. The explicit native flag is the higher-capability
# mode and takes precedence without requiring a model-name or image-specific
# environment scrub.


_USE_SMALL_MERGE_TOPK = os.getenv("HPU_GLM53_SMALL_MERGE_TOPK", "0") == "1"

_USE_PARTIAL_SCAN_TOPK = os.getenv("HPU_GLM53_PARTIAL_SCAN_TOPK", "0") == "1"

# Strangler-fig flag (default off).  A single-chunk replicated scan already
# yields its exact canonical top-k *set* from the threshold select; every later
# canonical re-sort (two stable argsorts per merge/finalize) and the final
# 2,051-wide token sort only reorder that set.  The fused route emits the same
# ascending token rows directly.  Unset/0 is byte-identical legacy behavior.
_USE_FUSED_PREFILL_SELECT = os.getenv(
    "HPU_GLM53_FUSED_PREFILL_SELECT", "0"
) == "1"
# The torch.compile scorer wrapper is a separate certified route; never trace
# the fused selection into it.
_COMPILED_KPOOL = os.getenv(
    "HPU_GLM53_COMPILED_KPOOL", "0"
).strip().lower() in ("1", "true")
# Strangler-fig flag (default off), lane DECODEGAP.  A replicated, unshared
# decode scan wider than one chunk (16 rows at 74K: two 16,384-pool chunks)
# otherwise merges every chunk through a canonical sort of top_pools + chunk
# columns.  A pool in the global canonical top-k is in its own chunk's top-k, so
# merging only each chunk's exact members (native k512 at proven widths, the V1
# pair elsewhere) yields the same candidates bitwise.  Unset/0 is legacy.
_USE_DECODE_SELECT = os.getenv("HPU_GLM53_MLA_DECODESEL", "0") == "1"
DECODE_SELECT_MAX_ROWS = 16
_DECODE_SELECT_LOG = logging.getLogger("glm53.mla_decodesel")
_decode_select_live_logged = False
_decode_select_live_keys: set[tuple[int, int]] = set()
# Strangler-fig flag (default off), plan 12.3c #2.  A prompt key wider than one
# scan chunk leaves the fused k512 route for the V1 pair on every chunk.  Same
# member argument as the decode selector above, and the same kernel and shape as
# the fused route's 128-row tile.  Unset/0 is the legacy per-chunk merge.
_USE_KEY_K512_MEMBER_MERGE = os.getenv("HPU_GLM53_KEY_K512_MEMBER_MERGE", "0") == "1"
KEY_MEMBER_MERGE_ROWS = 128
_KEY_MEMBER_MERGE_LOG = logging.getLogger("glm53.key_member_merge")
# Strangler-fig flag (default off).  A key of 128 rows or fewer scans unshared
# over several chunks, and _ordered_streaming_merge then sorts top_pools + chunk
# columns.  On the card a stable descending sort at such widths returns index -1
# past a row's finite scores, repeating a pool in any row with 1 to 511 of them.
# Each chunk's exact members merge at top_pools and 2 * top_pools columns instead,
# the same member argument as the routes above.  Unset/0 is the legacy merge.
_USE_SMALL_KEY_PARTIAL_MERGE = os.getenv("HPU_GLM53_SMALL_KEY_PARTIAL_MERGE", "0") == "1"
SMALL_KEY_MAX_ROWS = 128  # glm53_mixed_graph.SMALL_KEY_MAX_ROWS: keys this short take the unshared route the flag merges
_PARTIAL_MERGE_LOG = logging.getLogger("glm53.partial_merge")
_partial_merge_live_scans: set[tuple[str, int, int]] = set()
_key_member_merge_live_scans: set[int] = set()


def _exact_topk_membership(
    scores: torch.Tensor, logical_ids: torch.Tensor, count: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Mask of the exact top-``count`` finite scores, ties by lower column.

    Columns must hold unique ascending logical IDs, so cumulative tie rank is
    the canonical lower-ID tie order.  Returns ``(chosen, finite_scores)``.
    """
    valid = torch.isfinite(scores) & (logical_ids >= 0)
    finite_scores = scores.masked_fill(~valid, -torch.inf)
    threshold = torch.topk(finite_scores, count, dim=1, sorted=False).values.amin(dim=1, keepdim=True)
    strict = valid & (finite_scores > threshold)
    ties = valid & (finite_scores == threshold)
    remaining = count - strict.to(torch.int32).sum(dim=1, keepdim=True)
    tie_rank = torch.cumsum(ties.to(torch.int32), dim=1, dtype=torch.int32)
    return strict | (ties & (tie_rank <= remaining)), finite_scores


def _ascending_member_columns(
    chosen: torch.Tensor, logical_ids: torch.Tensor, count: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Columns of ``chosen`` in ascending logical-ID order, chosen first."""
    compact_keys = torch.where(chosen, 1_048_576 - logical_ids, -1)
    compact_indices = torch.topk(compact_keys, count, dim=1, sorted=True).indices
    return compact_indices, torch.gather(chosen, 1, compact_indices)


def _partial_scan_topk(scores, logical_ids, count):
    """Exact finite top-k with ID tie order, without sorting every pool twice.

    This route is restricted to one scan of unique ascending logical IDs.
    Native correctness and performance are required before it can be enabled.
    """
    chosen, finite_scores = _exact_topk_membership(scores, logical_ids, count)
    compact_indices, compact_valid = _ascending_member_columns(chosen, logical_ids, count)
    compact_scores = torch.gather(finite_scores, 1, compact_indices).masked_fill(~compact_valid, -torch.inf)
    compact_ids = torch.where(compact_valid, torch.gather(logical_ids, 1, compact_indices), -1)
    return _r590_topk(compact_scores, compact_ids, count)



def _partial_chunk_merge(running_scores, running_ids, scores, ids, count, *, first):
    """An item's global top-k membership requires its chunk top-k membership."""
    chunk_scores, chunk_ids = _partial_scan_topk(scores, ids, count)
    if first:
        return chunk_scores, chunk_ids
    return _r590_topk(torch.cat((running_scores, chunk_scores), dim=1),
                      torch.cat((running_ids, chunk_ids), dim=1), count)


def _ordered_streaming_merge(running_scores, running_ids, scores, ids, count):
    """Use the canonical stable merge on the certified lazy HPU path.

    The narrowed-sort optimization reproduced invalid finite scores and IDs
    on a second native invocation with all-invalid input rows. Keep canonical
    score/ID ordering until a replacement passes repeated native execution.
    """
    merged_scores = torch.cat((running_scores, scores), dim=1)
    merged_ids = torch.cat((running_ids, ids), dim=1)
    return _r590_topk(merged_scores, merged_ids, count)


def _decode_select_applies(*, batch: int, width: int, top_pools: int, tp_size: int,
                           shared_block_table: bool) -> bool:
    """Host-only gate of the decode member merge; every input is a Python int/bool (no device sync)."""
    return (_USE_DECODE_SELECT and not _COMPILED_KPOOL and _r590_enabled()
            and tp_size == 1 and not shared_block_table
            and batch <= DECODE_SELECT_MAX_ROWS and width >= top_pools
            and not mixed_graph_capture_active())


def _key_member_merge_applies(*, batch: int, width: int, top_pools: int, tp_size: int,
                              shared_block_table: bool) -> bool:
    """Host-only gate of the k512 member merge in a prompt key capture; every input is a Python int/bool (no device sync)."""
    return (_USE_KEY_K512_MEMBER_MERGE and _tpc_topk is not None and not _COMPILED_KPOOL and _r590_enabled()
            and key_capture_active() and shared_block_table and tp_size == 1
            and batch == KEY_MEMBER_MERGE_ROWS and width >= top_pools)


def _decode_select_static_blockers() -> list[str]:
    """Process-wide settings that keep the decode gate from ever opening (host env only)."""
    blockers = []
    if _COMPILED_KPOOL:
        blockers.append("HPU_GLM53_COMPILED_KPOOL=1")
    if os.getenv("HPU_GLM53_TP_SHARDED_KPOOL", "0").strip().lower() in ("1", "true"):
        blockers.append("HPU_GLM53_TP_SHARDED_KPOOL=1 (tp_size>1)")
    if not _r590_enabled():
        blockers.append("HPU_GLM53_R590_ENABLE!=1")
    return blockers


def _announce_decode_select() -> None:
    """One boot line when the flag is on: whether the gate is reachable and whether native k512 is loaded."""
    if not _USE_DECODE_SELECT:
        return
    blockers = _decode_select_static_blockers()
    _DECODE_SELECT_LOG.warning(
        "GLM53_MLA_DECODESEL armed reachable=%d native_k512=%d max_rows=%d blockers=%s",
        int(not blockers), int(_tpc_topk is not None), DECODE_SELECT_MAX_ROWS, ",".join(blockers) or "none")


def _announce_key_member_merge() -> None:
    """One boot line when the flag is on: without the native module the gate never opens."""
    if _USE_KEY_K512_MEMBER_MERGE:
        _KEY_MEMBER_MERGE_LOG.warning("GLM53_KEY_MEMBER_MERGE armed native_k512=%d", int(_tpc_topk is not None))


def _log_key_member_merge_live(batch: int, width: int, scan_width: int) -> None:
    """One line per prompt key width at capture (host only); replay runs no Python, so a line is a graph that holds the merge."""
    if scan_width in _key_member_merge_live_scans:
        return
    _key_member_merge_live_scans.add(scan_width)
    _KEY_MEMBER_MERGE_LOG.warning("GLM53_KEY_MEMBER_MERGE_LIVE rows=%d chunk=%d scan=%d", batch, width, scan_width)


def _log_decode_select_live(batch: int, width: int, scan_width: int) -> None:
    """One line per decode key's capture, else one at the first decode chunk that takes the member merge (warmup time, host only)."""
    global _decode_select_live_logged
    if decode_key_capture_active():
        if (batch, scan_width) in _decode_select_live_keys:
            return
        _decode_select_live_keys.add((batch, scan_width))
        _DECODE_SELECT_LOG.warning("GLM53_MLA_DECODESEL_LIVE rows=%d chunk=%d scan=%d native_k512=%d capture=decode_key",
                                   batch, width, scan_width, int(_tpc_topk is not None))
        return
    if _decode_select_live_logged:
        return
    _decode_select_live_logged = True
    _DECODE_SELECT_LOG.warning("GLM53_MLA_DECODESEL_LIVE rows=%d chunk=%d scan=%d native_k512=%d",
                               batch, width, scan_width, int(_tpc_topk is not None))


def _partial_merge_route(*, width: int, top_pools: int, tp_size: int, shared_block_table: bool, start: int,
                         scan_width: int) -> str | None:
    """Host-only gate of _partial_chunk_merge, and why it opened: shared (a shared table), single (one chunk is the whole scan) or
    small_key (HPU_GLM53_SMALL_KEY_PARTIAL_MERGE, an unshared scan of several chunks). None keeps the legacy merges."""
    if not (_USE_PARTIAL_SCAN_TOPK and _r590_enabled() and tp_size == 1 and width >= top_pools):
        return None
    if shared_block_table:
        return "shared"
    if start == 0 and width == scan_width:
        return "single"
    return "small_key" if _USE_SMALL_KEY_PARTIAL_MERGE else None


def _partial_merge_static_blockers() -> list[str]:
    """Process-wide settings that keep _partial_chunk_merge shut for every scan (host env only), each as the setting that opens it."""
    blockers = []
    if not _USE_PARTIAL_SCAN_TOPK:
        blockers.append("HPU_GLM53_PARTIAL_SCAN_TOPK=1")
    if not _r590_enabled():
        blockers.append("HPU_GLM53_R590_ENABLE=1")
    if os.getenv("HPU_GLM53_TP_SHARDED_KPOOL", "0").strip().lower() in ("1", "true"):
        blockers.append("HPU_GLM53_TP_SHARDED_KPOOL=0 (a sharded kpool runs tp_size>1)")
    return blockers


def _announce_partial_merge() -> None:
    """One boot line per flag that is on: the partial-scan top-k, and the small-key merge that rides on it. reachable=0 names the fixes."""
    if not (_USE_PARTIAL_SCAN_TOPK or _USE_SMALL_KEY_PARTIAL_MERGE):
        return
    blockers = ",".join(_partial_merge_static_blockers()) or "none"
    if _USE_PARTIAL_SCAN_TOPK:
        _PARTIAL_MERGE_LOG.warning("GLM53_PARTIAL_SCAN_TOPK armed reachable=%d blockers=%s", int(blockers == "none"), blockers)
    if _USE_SMALL_KEY_PARTIAL_MERGE:
        _PARTIAL_MERGE_LOG.warning("GLM53_SMALL_KEY_PARTIAL_MERGE armed reachable=%d blockers=%s", int(blockers == "none"), blockers)


def _log_partial_merge_live(route: str, batch: int, width: int, scan_width: int) -> None:
    """One line per route, rows and scan width (host only); a key's replay runs no Python, so a line at capture is a graph that holds it."""
    marker = (route, batch, scan_width)
    if marker in _partial_merge_live_scans or torch.compiler.is_compiling():
        return  # a traced call (HPU_GLM53_COMPILED_KPOOL) stays free of host logging
    _partial_merge_live_scans.add(marker)
    name = "GLM53_SMALL_KEY_PARTIAL_MERGE_LIVE" if route == "small_key" else "GLM53_PARTIAL_SCAN_TOPK_LIVE"
    _PARTIAL_MERGE_LOG.warning("%s route=%s rows=%d chunk=%d scan=%d capture=%d", name, route, batch, width, scan_width,
                               int(key_capture_active()))


def _refuse_narrow_small_key_chunks(*, chunk: int, top_pools: int, tp_size: int, shared_block_table: bool, scan_width: int) -> None:
    """In a key capture (boot), a chunk narrower than top_pools of an unshared scan of several chunks would leave the small-key merge
    for the legacy one, whose sort the card gets wrong: refuse the capture, and with it the boot, rather than select wrong tokens."""
    if not (_USE_SMALL_KEY_PARTIAL_MERGE and key_capture_active() and not shared_block_table and scan_width > chunk):
        return
    if _partial_merge_static_blockers() or tp_size != 1:
        return  # the boot check (glm53_mixed_graph.check_small_key_merge) names these
    tail = scan_width % chunk
    if chunk >= top_pools and not 0 < tail < top_pools:
        return
    narrow = chunk if chunk < top_pools else tail
    raise ValueError(
        f"HPU_GLM53_SMALL_KEY_PARTIAL_MERGE: a key capture scans {scan_width} pools in {chunk}-pool chunks, and a {narrow}-pool chunk is "
        f"narrower than the {top_pools} pools the partial merge needs, so it would take the legacy merge that selects wrong tokens. "
        "Raise HPU_GLM53_CONTEXT_CHUNK_POOLS / HPU_GLM53_CONTEXT_CHUNK_BUDGET_MIB so the chunk is at least "
        f"{top_pools} pools and divides the scan, or drop the key of {SMALL_KEY_MAX_ROWS} rows or fewer, and restart all workers.")


_announce_decode_select()
_announce_key_member_merge()
_announce_partial_merge()


def _decode_member_merge(
    running_scores: torch.Tensor,
    running_ids: torch.Tensor,
    scores: torch.Tensor,
    logical_ids: torch.Tensor,
    count: int,
    *,
    first: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Canonical running top-``count`` after one chunk, sorting only that chunk's exact members.

    ``logical_ids`` must be unique and ascending per row (replicated tiles), so
    lower column = lower ID and the members equal the canonical chunk top-k.
    """
    if not (_tpc_topk is not None and _tpc_topk.applies(scores, logical_ids, count)):
        return _partial_chunk_merge(running_scores, running_ids, scores, logical_ids, count, first=first)
    witness_lever("k512_topk")
    columns, member, finite_scores = _tpc_topk.member_columns(scores, logical_ids)
    chunk_scores = torch.gather(finite_scores, 1, columns).masked_fill(~member, -torch.inf)
    chunk_ids = torch.where(member, torch.gather(logical_ids, 1, columns), -1)
    if first:
        return _r590_topk(chunk_scores, chunk_ids, count)
    return _r590_topk(torch.cat((running_scores, chunk_scores), dim=1),
                      torch.cat((running_ids, chunk_ids), dim=1), count)


def fwht128(x: torch.Tensor) -> torch.Tensor:
    """Orthonormal Walsh-Hadamard transform over the last dimension."""

    if x.shape[-1] != INDEX_DIM:
        raise ValueError(f"FWHT requires dimension {INDEX_DIM}, got {x.shape[-1]}")
    from glm53_mla_fwht import applies as fwht_applies, transform as native_fwht
    if fwht_applies(x):
        witness_lever("fwht")
        return native_fwht(x)
    y = x.float()
    width = 1
    while width < INDEX_DIM:
        shape = (*y.shape[:-1], -1, 2, width)
        grouped = y.reshape(shape)
        left, right = grouped.unbind(dim=-2)
        y = torch.stack((left + right, left - right), dim=-2).reshape_as(y)
        width *= 2
    return y * (INDEX_DIM**-0.5)


def quantize_index_fp8(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-vector E4M3 quantization with an E8M0-compatible power-of-two scale."""

    absmax = x.float().abs().amax(dim=-1, keepdim=True).clamp_min(1e-4)
    scale = torch.pow(2.0, torch.ceil(torch.log2(absmax / FP8_MAX)))
    quant = (x.float() / scale).clamp(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn)
    return quant, scale.float()


def dequantize_index_fp8(
    values: torch.Tensor, scale: torch.Tensor, dtype: torch.dtype = torch.float32
) -> torch.Tensor:
    return values.to(dtype) * scale.to(dtype)


def compress_complete_kpools(
    keys: torch.Tensor,
    gate_scores: torch.Tensor,
    ape: torch.Tensor,
    *,
    pool_size: int = 4,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compress complete token pools and discard all raw index keys.

    Args:
        keys, gate_scores: ``[batch, tokens, 128]``.
        ape: learned ``[pool_size, 128]`` positional compression bias.

    The incomplete tail is deliberately excluded; it lives in the fixed
    one-block-per-request tail cache until enough tokens arrive to complete it.
    """

    if keys.shape != gate_scores.shape or keys.shape[-1] != INDEX_DIM:
        raise ValueError("keys and gate_scores must share [B,T,128]")
    if ape.shape != (pool_size, INDEX_DIM):
        raise ValueError(f"ape must be [{pool_size},{INDEX_DIM}]")
    complete_tokens = (keys.shape[1] // pool_size) * pool_size
    if complete_tokens == 0:
        empty_shape = (keys.shape[0], 0, INDEX_DIM)
        return (
            torch.empty(empty_shape, dtype=torch.float8_e4m3fn, device=keys.device),
            torch.empty((*empty_shape[:-1], 1), dtype=torch.float32, device=keys.device),
        )
    k = keys[:, :complete_tokens].reshape(
        keys.shape[0], -1, pool_size, INDEX_DIM
    )
    scores = gate_scores[:, :complete_tokens].reshape_as(k)
    probability = torch.softmax(scores.float() + ape.float()[None, None], dim=2)
    pooled = (probability * k.float()).sum(dim=2).to(keys.dtype)
    return quantize_index_fp8(fwht128(pooled).to(keys.dtype))


def pooled_index_logits(
    queries: torch.Tensor,
    head_weights: torch.Tensor,
    pooled_values: torch.Tensor,
    pooled_scales: torch.Tensor,
) -> torch.Tensor:
    """Compute per-pool index scores without materializing dequantized cache.

    ``queries`` is ``[batch, index_heads, 128]`` and ``head_weights`` is
    ``[batch, index_heads]``.  Query rotation is applied here; production may
    fuse rotation, FP8 quantization and the matrix product.
    """

    if queries.ndim != 3 or queries.shape[-1] != INDEX_DIM:
        raise ValueError("queries must be [B,H,128]")
    if head_weights.shape != queries.shape[:2]:
        raise ValueError("head_weights must be [B,H]")
    rotated_q = fwht128(queries)
    # Multiplication by the scale is folded into the pooled logits.  This keeps
    # the persistent key cache compressed throughout the operation.
    raw = torch.einsum("bhd,bpd->bhp", rotated_q, pooled_values.float())
    # GLM-5.3's learned sparse scorer applies its nonlinearity per head,
    # before the signed learned head weights are reduced.  Moving this ReLU
    # after the reduction (or omitting it) changes pool ordering.
    raw = torch.relu(raw)
    raw = raw * pooled_scales.squeeze(-1).float().unsqueeze(1)
    return (raw * head_weights.float().unsqueeze(-1)).sum(dim=1)


def select_sparse_token_indices(
    pool_logits: torch.Tensor,
    sequence_lengths: torch.Tensor,
    *,
    topk_tokens: int = 2048,
    pool_size: int = 4,
) -> torch.Tensor:
    """Select top pools, expand to tokens, and append the incomplete tail.

    Returns sorted, padded token indices ``[batch, width]``.  Invalid padding is
    ``-1``.  Sorting provides monotonically increasing physical gathers for the
    HPU paged-latent cache.
    """

    batch, max_pools = pool_logits.shape
    if sequence_lengths.shape != (batch,):
        raise ValueError("sequence_lengths must be [batch]")
    top_pools = math.ceil(topk_tokens / pool_size)
    max_tail = pool_size - 1
    width = top_pools * pool_size + max_tail
    output = torch.full(
        (batch, width), -1, dtype=torch.int32, device=pool_logits.device
    )
    pool_ids = torch.arange(max_pools, device=pool_logits.device)
    offsets = torch.arange(pool_size, device=pool_logits.device)
    for batch_idx in range(batch):
        seq_len = int(sequence_lengths[batch_idx])
        complete_pools = min(seq_len // pool_size, max_pools)
        take = min(top_pools, complete_pools)
        selected = torch.empty(0, dtype=torch.long, device=pool_logits.device)
        if take:
            scores = pool_logits[batch_idx].masked_fill(
                pool_ids >= complete_pools, -torch.inf
            )
            chosen_pools = torch.topk(scores, k=take, sorted=False).indices
            selected = (
                chosen_pools[:, None] * pool_size + offsets[None]
            ).reshape(-1)
        tail_start = complete_pools * pool_size
        tail = torch.arange(tail_start, seq_len, device=pool_logits.device)
        merged = torch.cat((selected, tail)).unique(sorted=True)
        output[batch_idx, : merged.numel()] = merged.to(torch.int32)
    return output


def kpool_bytes_per_source_token(
    *, index_dim: int = INDEX_DIM, pool_size: int = 4, scale_bytes: int = 4
) -> float:
    """Persistent pooled-index bytes per original source token per MLA layer."""

    return (index_dim + scale_bytes) / pool_size


def _batch_index_select(
    values: torch.Tensor,
    indices: torch.Tensor,
) -> torch.Tensor:
    """Select independently from every row without emitting ``aten::gather``.

    Synapse 1.24 can reject the large dynamic gather graph produced by the
    sparse-index path. Flattening row-local indices into one ``index_select``
    is equivalent, stays on HPU, and uses an already-supported primitive.
    """

    if values.ndim != 2 or indices.ndim != 2:
        raise ValueError("values and indices must both be rank two")
    if values.shape[0] != indices.shape[0]:
        raise ValueError("values and indices batch sizes must match")
    columns = values.shape[1]
    row_offsets = (
        torch.arange(values.shape[0], device=indices.device, dtype=torch.long)
        .unsqueeze(1)
        .mul(columns)
    )
    flat_indices = (indices.long() + row_offsets).reshape(-1)
    return values.reshape(-1).index_select(0, flat_indices).reshape(indices.shape)


def logical_to_physical_slots(
    logical_indices: torch.Tensor,
    block_table: torch.Tensor,
    block_size: int,
) -> torch.Tensor:
    """Translate per-request logical indices into flattened physical rows.

    ``logical_indices`` is ``[B,K]`` with ``-1`` padding and ``block_table`` is
    ``[B,max_blocks]``.  The operation is fully tensorized so it can live in an
    HPU graph; invalid entries remain ``-1`` and are never dereferenced.
    """

    if logical_indices.ndim != 2 or block_table.ndim != 2:
        raise ValueError("logical_indices and block_table must both be rank two")
    if logical_indices.shape[0] != block_table.shape[0]:
        raise ValueError("logical_indices and block_table batch sizes must match")
    if block_size <= 0:
        raise ValueError("block_size must be positive")

    logical = logical_indices.long()
    valid = logical >= 0
    safe = logical.clamp_min(0)
    logical_blocks = torch.div(safe, block_size, rounding_mode="floor")
    in_range = logical_blocks < block_table.shape[1]
    safe_blocks = logical_blocks.clamp_max(max(block_table.shape[1] - 1, 0))
    physical_blocks = _batch_index_select(block_table.long(), safe_blocks)
    physical = physical_blocks * block_size + torch.remainder(safe, block_size)
    return torch.where(valid & in_range & (physical_blocks >= 0), physical, -1)


def pooled_slots_to_tp_local(
    global_slots: torch.Tensor,
    *,
    pools_per_block: int,
    tp_rank: int,
    tp_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Map global pooled-index rows to striped TP-local storage.

    Whole 128-token source blocks are striped across TP ranks.  Keeping every
    pool belonging to one source block on the same owner makes the mapping
    identical to the Q4 MLA latent-cache mapping and preserves monotonic local
    rows.  The returned ownership mask is false for ``-1`` padding.
    """

    if pools_per_block <= 0:
        raise ValueError("pools_per_block must be positive")
    if tp_size <= 0 or not 0 <= tp_rank < tp_size:
        raise ValueError("tp_rank must be in [0, tp_size)")
    slots = global_slots.long()
    valid = slots >= 0
    safe = slots.clamp_min(0)
    global_block = torch.div(safe, pools_per_block, rounding_mode="floor")
    owned = valid & (torch.remainder(global_block, tp_size) == tp_rank)
    local_block = torch.div(global_block, tp_size, rounding_mode="floor")
    local = local_block * pools_per_block + torch.remainder(safe, pools_per_block)
    return local, owned


def unpack_pooled_index_rows(
    packed_rows: torch.Tensor,
    *,
    index_dim: int = INDEX_DIM,
) -> tuple[torch.Tensor, torch.Tensor]:
    """View packed ``[FP8 values | FP32 scale]`` rows without a full copy."""

    if packed_rows.dtype != torch.uint8 or packed_rows.shape[-1] != index_dim + 4:
        raise ValueError(f"packed rows must be uint8 [...,{index_dim + 4}]")
    values = packed_rows[..., :index_dim].contiguous().view(torch.float8_e4m3fn)
    scale = packed_rows[..., index_dim:].contiguous().view(torch.float32)
    return values, scale


def pack_pooled_index_rows(
    values: torch.Tensor,
    scales: torch.Tensor,
) -> torch.Tensor:
    """Pack FP8 values and their FP32 per-vector scale into 132-byte rows."""

    if values.dtype != torch.float8_e4m3fn or values.shape[-1] != INDEX_DIM:
        raise ValueError("values must be E4M3 [...,128]")
    if scales.shape != (*values.shape[:-1], 1):
        raise ValueError("scales must be [...,1]")
    return torch.cat(
        (values.contiguous().view(torch.uint8), scales.float().contiguous().view(torch.uint8)),
        dim=-1,
    )


def _validate_scan_inputs(
    query_values: torch.Tensor,
    head_weights: torch.Tensor,
    block_table: torch.Tensor,
    sequence_lengths: torch.Tensor,
    *,
    source_block_size: int,
    pool_size: int,
    context_chunk_pools: int,
    tp_rank: int,
    tp_size: int,
    shared_block_table: bool,
) -> None:
    """Reject malformed scorer inputs before any HPU tensor is built."""
    if query_values.ndim != 3 or query_values.shape[-1] != INDEX_DIM:
        raise ValueError("query_values must be [B,H,128]")
    if head_weights.shape != query_values.shape[:2]:
        raise ValueError("head_weights must be [B,H]")
    if source_block_size % pool_size:
        raise ValueError("source_block_size must be divisible by pool_size")
    if context_chunk_pools <= 0:
        raise ValueError("context_chunk_pools must be positive")
    if tp_size <= 0 or not 0 <= tp_rank < tp_size:
        raise ValueError("tp_rank must be in [0, tp_size)")
    batch = query_values.shape[0]
    routing_batch = 1 if shared_block_table else batch
    if block_table.shape[0] != routing_batch or sequence_lengths.shape != (batch,):
        raise ValueError("block table and sequence lengths must match query batch")


def _score_operands(
    query_values: torch.Tensor,
    head_weights: torch.Tensor,
    shared_block_table: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the scorer's GEMM query operand and FP32 head weights."""
    # The index vectors are already E4M3-quantized.  Expanding the entire
    # context scan to FP32 makes this long-context GEMM run on Gaudi's slow
    # FP32 path.  BF16 keeps the matrix multiply on the MME; immediately widen
    # its bounded output so ReLU, signed head reduction, scaling, and top-k
    # retain the reference FP32 semantics.
    score_matmul_dtype = torch.bfloat16 if _USE_BF16_KPOOL_SCORE else torch.float32
    q = (
        query_values.contiguous()
        if _USE_NATIVE_FP8_KPOOL_SCORE
        else query_values.to(score_matmul_dtype)
    )
    if shared_block_table:
        # One request owns these keys. Fold query and head rows into the
        # M dimension rather than gathering its keys for every query.
        q = q.reshape(1, query_values.shape[0] * query_values.shape[1], INDEX_DIM)
    return q, head_weights.float()


def _native_fp8_logits(q: torch.Tensor, values: torch.Tensor) -> torch.Tensor:
    """FP32-accumulated E4M3 GEMM on the Gaudi MME."""
    if q.device.type != "hpu":
        raise RuntimeError(
            "native FP8 kpool scoring requires an HPU tensor"
        )
    fp8_gemm = getattr(getattr(torch.ops, "hpu", None), "fp8_gemm_v2", None)
    if fp8_gemm is None:
        raise RuntimeError("torch.ops.hpu.fp8_gemm_v2 is unavailable")
    # Both operands are already E4M3 values. Their positive per-vector
    # scales stay outside the GEMM exactly as in the reference path:
    # q_scale is folded into head_weights and key scale is applied
    # after the per-head ReLU. Request FP32 output/accumulation so only
    # the storage-to-MME expansion disappears.
    return fp8_gemm(
        q, False, values.contiguous(), True, None, torch.float32,
        None, None, None, False, None,
    )


def _score_pool_chunk(
    q: torch.Tensor,
    weights: torch.Tensor,
    cache_rows: torch.Tensor,
    physical: torch.Tensor,
    valid: torch.Tensor,
    *,
    query_shape: tuple[int, ...],
    shared_block_table: bool,
) -> torch.Tensor:
    """Masked FP32 pool scores ``[batch, width]`` for one context chunk."""
    routing_batch, width = physical.shape
    rows = cache_rows.index_select(0, physical.clamp_min(0).reshape(-1)).reshape(
        routing_batch, width, -1
    )
    values, scales = unpack_pooled_index_rows(rows)
    if _USE_NATIVE_FP8_KPOOL_SCORE:
        logits = _native_fp8_logits(q, values)
    else:
        score_matmul_dtype = torch.bfloat16 if _USE_BF16_KPOOL_SCORE else torch.float32
        logits = torch.matmul(
            q, values.to(score_matmul_dtype).transpose(1, 2)
        )
    if shared_block_table:
        logits = logits.reshape(query_shape[0], query_shape[1], width)
    logits = logits.float()
    # Query scale and head_dim**-0.5 are folded into ``weights`` and the
    # positive key scale is applied below, so ReLU at this point preserves
    # the reference ordering: relu(per-head score), then signed head sum.
    logits = torch.relu(logits)
    logits = (logits * weights.unsqueeze(-1)).sum(dim=1)
    logits = logits * scales.squeeze(-1).float()
    return logits.masked_fill(~valid, -torch.inf)


def _replicated_chunk_routing(
    start: int,
    width: int,
    block_table: torch.Tensor,
    complete_pools: torch.Tensor,
    pools_per_block: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Logical IDs, physical pooled rows and validity for one replicated tile.

    Derived per tile: views of full-context routing parents lose their lazy
    backing tensor_data across the scorer's per-tile mark_step.
    """
    pool_ids = torch.arange(start, start + width, device=block_table.device)
    logical = pool_ids.unsqueeze(0).expand(block_table.shape[0], -1)
    physical = logical_to_physical_slots(logical, block_table, pools_per_block)
    valid = (logical < complete_pools.unsqueeze(1)) & (physical >= 0)
    return logical, physical, valid


def _fused_select_applies(
    *,
    max_complete_pools_bucket: int | None,
    context_chunk_pools: int,
    top_pools: int,
) -> bool:
    """True only where the legacy route is one exact partial-scan chunk.

    Requires a host-side bucket (no device sync), one replicated chunk wide
    enough for top-k, and the certified R590 canonical tie order.
    """
    if _COMPILED_KPOOL or not (
        _USE_FUSED_PREFILL_SELECT and _USE_PARTIAL_SCAN_TOPK and _r590_enabled()
    ):
        return False
    if max_complete_pools_bucket is None:
        return False
    scan_width = int(max_complete_pools_bucket)
    return top_pools <= scan_width <= context_chunk_pools


def _compose_sorted_tokens(
    pool_ids: torch.Tensor,
    selected_count: torch.Tensor,
    sequence_lengths: torch.Tensor,
    pool_size: int,
) -> torch.Tensor:
    """Ascending, ``-1``-padded source tokens without a sort.

    ``pool_ids`` holds each row's chosen pool IDs ascending in its first
    ``selected_count`` columns.  Every chosen pool is complete, so its tokens
    precede the incomplete tail; the sorted row is therefore the expanded
    pools, then the live tail, then padding.
    """
    batch, top_pools = pool_ids.shape
    width = top_pools * pool_size + pool_size - 1
    column = torch.arange(width, device=pool_ids.device, dtype=torch.long).unsqueeze(0)
    pool_column = torch.div(column, pool_size, rounding_mode="floor").clamp_max(top_pools - 1)
    pool_tokens = (
        _batch_index_select(pool_ids.long(), pool_column.expand(batch, -1)) * pool_size
        + torch.remainder(column, pool_size)
    )
    lengths = sequence_lengths.long().unsqueeze(1)
    tail_start = torch.div(lengths, pool_size, rounding_mode="floor") * pool_size
    pool_end = selected_count.long().unsqueeze(1) * pool_size
    tokens = torch.where(column < pool_end, pool_tokens, tail_start + (column - pool_end))
    return tokens.masked_fill(column >= pool_end + (lengths - tail_start), -1)


@_r589_stage("indexer.tiles")
def _fused_single_chunk_pooled_topk(
    scorer_inputs: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    block_table: torch.Tensor,
    sequence_lengths: torch.Tensor,
    geometry: tuple[int, int, int, int],
    shared_block_table: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Exact equivalent of candidates + finalize for one replicated chunk.

    ``scorer_inputs`` is ``(query_values, head_weights, packed_cache)``;
    ``geometry`` is ``(source_block_size, pool_size, top_pools, scan_width)``.
    """
    query_values, head_weights, packed_cache = scorer_inputs
    source_block_size, pool_size, top_pools, scan_width = geometry
    routing_table = block_table[:1] if shared_block_table else block_table
    logical, physical, valid = _replicated_chunk_routing(
        0, scan_width, routing_table,
        torch.div(sequence_lengths.long(), pool_size, rounding_mode="floor"),
        source_block_size // pool_size,
    )
    operands = _score_operands(query_values, head_weights, shared_block_table)
    logits = _score_pool_chunk(
        *operands, packed_cache.reshape(-1, packed_cache.shape[-1]), physical, valid,
        query_shape=tuple(query_values.shape), shared_block_table=shared_block_table,
    )
    logical_ids = logical.expand(query_values.shape[0], -1)
    if (_tpc_topk is not None
            and shared_block_table and logits.shape[0] == 128
            and _tpc_topk.applies(logits, logical_ids, top_pools)
            and (not mixed_graph_capture_active() or key_capture_active())):
        witness_lever("k512_topk")
        columns, member, _ = _tpc_topk.member_columns(logits, logical_ids)
        if _rc6_flags.witness("tpc_topk", "call") == 1:
            _rc6_flags.LOG.warning("FLASH_K512_LIVE rank=%d rows=%d width=%d",
                                   _rc6_flags.process_rank(), logits.shape[0], scan_width)
        selected_count = member.sum(dim=1)
    else:
        chosen, _ = _exact_topk_membership(logits, logical_ids, top_pools)
        columns, _ = _ascending_member_columns(chosen, logical_ids, top_pools)
        selected_count = chosen.sum(dim=1)
    source_tokens = _compose_sorted_tokens(
        torch.gather(logical_ids, 1, columns), selected_count,
        sequence_lengths, pool_size,
    )
    physical_tokens = logical_to_physical_slots(source_tokens, block_table, source_block_size)
    return source_tokens.to(torch.int32), physical_tokens.to(torch.int32)


@_r589_stage("indexer.tiles")
def streaming_pooled_topk_candidates(
    query_values: torch.Tensor,
    head_weights: torch.Tensor,
    packed_cache: torch.Tensor,
    block_table: torch.Tensor,
    sequence_lengths: torch.Tensor,
    *,
    source_block_size: int,
    pool_size: int = 4,
    topk_tokens: int = 2048,
    context_chunk_pools: int = 4096,
    max_complete_pools_bucket: int | None = None,
    tp_rank: int = 0,
    tp_size: int = 1,
    shared_block_table: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return this rank's bounded top-pool scores and logical pool IDs.

    With ``tp_size > 1`` the persistent cache contains only source blocks
    owned by this rank.  The block table is compacted to those owned blocks
    *before* pooled rows are gathered or scored.  The small fixed-width
    candidate tensors can then be all-gathered and merged, avoiding both
    full-cache replication and the old global-width/late-mask compute waste.
    """

    _validate_scan_inputs(
        query_values, head_weights, block_table, sequence_lengths,
        source_block_size=source_block_size, pool_size=pool_size,
        context_chunk_pools=context_chunk_pools, tp_rank=tp_rank,
        tp_size=tp_size, shared_block_table=shared_block_table,
    )
    batch = query_values.shape[0]
    routing_batch = 1 if shared_block_table else batch
    pools_per_block = source_block_size // pool_size
    top_pools = math.ceil(topk_tokens / pool_size)
    complete_pools = torch.div(sequence_lengths.long(), pool_size, rounding_mode="floor")
    max_complete = (
        int(max_complete_pools_bucket)
        if max_complete_pools_bucket is not None
        else (int(complete_pools.max().item()) if complete_pools.numel() else 0)
    )
    if max_complete < 0:
        raise ValueError("max_complete_pools_bucket must be non-negative")
    cache_rows = packed_cache.reshape(-1, packed_cache.shape[-1])

    if tp_size > 1:
        # Physical block allocation can be skewed across ranks for one request.
        # Compact to the observed maximum instead of assuming an even split and
        # silently dropping owned routes.
        max_logical_blocks = min(
            int(block_table.shape[1]),
            math.ceil(max_complete / pools_per_block),
        )
        if max_logical_blocks:
            logical_blocks_1d = torch.arange(
                max_logical_blocks, device=query_values.device
            )
            logical_blocks = logical_blocks_1d.unsqueeze(0).expand(routing_batch, -1)
            global_blocks = block_table[:, :max_logical_blocks].long()
            owned_blocks = (global_blocks >= 0) & (
                torch.remainder(global_blocks, tp_size) == tp_rank
            )
            local_block_width = (
                max_logical_blocks if mixed_graph_capture_active() or decode_key_capture_active()
                else int(owned_blocks.sum(dim=1).max().item())
            )
        else:
            logical_blocks = torch.empty(
                (routing_batch, 0), dtype=torch.long, device=query_values.device
            )
            global_blocks = logical_blocks
            owned_blocks = logical_blocks.bool()
            local_block_width = 0

        if local_block_width:
            sentinel = max_logical_blocks
            compact_key = torch.where(
                owned_blocks,
                logical_blocks,
                torch.full_like(logical_blocks, sentinel),
            )
            compact_logical_blocks = torch.sort(compact_key, dim=1).values[
                :, :local_block_width
            ]
            compact_valid_blocks = compact_logical_blocks < max_logical_blocks
            safe_logical_blocks = compact_logical_blocks.clamp_max(
                max_logical_blocks - 1
            )
            compact_global_blocks = torch.gather(
                global_blocks, 1, safe_logical_blocks
            )
            compact_local_blocks = torch.div(
                compact_global_blocks.clamp_min(0),
                tp_size,
                rounding_mode="floor",
            )
            pool_offsets = torch.arange(
                pools_per_block, device=query_values.device
            )
            logical_scan = (
                compact_logical_blocks.unsqueeze(-1) * pools_per_block
                + pool_offsets
            ).reshape(routing_batch, -1)
            physical_scan = (
                compact_local_blocks.unsqueeze(-1) * pools_per_block
                + pool_offsets
            ).reshape(routing_batch, -1)
            valid_scan = compact_valid_blocks.unsqueeze(-1).expand(
                -1, -1, pools_per_block
            ).reshape(routing_batch, -1)
            valid_scan = valid_scan & (
                logical_scan < complete_pools.unsqueeze(1)
            )
        else:
            logical_scan = torch.empty(
                (routing_batch, 0), dtype=torch.long, device=query_values.device
            )
            physical_scan = logical_scan
            valid_scan = logical_scan.bool()
    else:
        # Do not build full [query, max-context] routing tensors.  The kpool
        # scorer crosses a lazy step after every context tile; views of those
        # giant parents lose their backing tensor_data at a later tile and
        # also waste several GiB for mostly masked prompt rows.  Replicated
        # kpool routing is cheap and exact to derive independently per tile.
        scan_width = max_complete

    if tp_size > 1:
        scan_width = int(logical_scan.shape[1])

    running_scores = torch.full(
        (batch, top_pools), -torch.inf, device=query_values.device, dtype=torch.float32
    )
    running_ids = torch.full(
        (batch, top_pools), -1, device=query_values.device, dtype=torch.long
    )
    q, weights = _score_operands(query_values, head_weights, shared_block_table)
    _refuse_narrow_small_key_chunks(chunk=context_chunk_pools, top_pools=top_pools, tp_size=tp_size,
                                    shared_block_table=shared_block_table, scan_width=scan_width)

    for start in range(0, scan_width, context_chunk_pools):
        width = min(context_chunk_pools, scan_width - start)
        if tp_size > 1:
            logical = logical_scan[:, start : start + width]
            physical = physical_scan[:, start : start + width]
            valid = valid_scan[:, start : start + width]
        else:
            logical, physical, valid = _replicated_chunk_routing(
                start, width, block_table, complete_pools, pools_per_block
            )
        logits = _score_pool_chunk(
            q, weights, cache_rows, physical, valid,
            query_shape=tuple(query_values.shape),
            shared_block_table=shared_block_table,
        )

        if _decode_select_applies(batch=batch, width=width, top_pools=top_pools, tp_size=tp_size,
                                  shared_block_table=shared_block_table):
            _log_decode_select_live(batch, width, scan_width)
            running_scores, running_ids = _decode_member_merge(
                running_scores, running_ids, logits,
                logical.expand(batch, -1), top_pools, first=(start == 0)
            )
        elif _key_member_merge_applies(batch=batch, width=width, top_pools=top_pools, tp_size=tp_size,
                                       shared_block_table=shared_block_table):
            _log_key_member_merge_live(batch, width, scan_width)
            running_scores, running_ids = _decode_member_merge(
                running_scores, running_ids, logits,
                logical.expand(batch, -1), top_pools, first=(start == 0)
            )
        elif (route := _partial_merge_route(width=width, top_pools=top_pools, tp_size=tp_size,
                                            shared_block_table=shared_block_table, start=start, scan_width=scan_width)):
            # Each replicated tile has unique ascending logical IDs.
            # Keep its exact top-k, then merge only compact candidates.
            _log_partial_merge_live(route, batch, width, scan_width)
            running_scores, running_ids = _partial_chunk_merge(
                running_scores, running_ids, logits,
                logical.expand(batch, -1), top_pools, first=(start == 0)
            )
        elif _USE_SMALL_MERGE_TOPK and _r590_enabled() and tp_size == 1:
            running_scores, running_ids = _ordered_streaming_merge(
                running_scores, running_ids, logits, logical.expand(batch, -1), top_pools
            )
        else:
            merged_scores = torch.cat((running_scores, logits), dim=1)
            merged_ids = torch.cat((running_ids, logical.expand(batch, -1)), dim=1)
            running_scores, running_ids = _r590_topk(merged_scores, merged_ids, top_pools)
        running_ids = torch.where(
            torch.isfinite(running_scores),
            running_ids,
            torch.full_like(running_ids, -1),
        )
        # A Python loop does not by itself bound Synapse lazy workspace: all
        # context tiles otherwise accumulate into one giant prompt recipe.
        # Commit each non-final tile so the fixed-size scorer workspace is
        # reused while the small running top-k tensors carry the dependency.
        if (
            start + width < scan_width
            and query_values.device.type == "hpu"
            and os.getenv("PT_HPU_LAZY_MODE", "0") == "1"
            and (not mixed_graph_capture_active() or capture_tile_marks())
        ):
            import habana_frameworks.torch.core as htcore

            htcore.mark_step(sync=lazy_fence_sync())

    return running_scores, running_ids


def finalize_pooled_topk(
    candidate_scores: torch.Tensor,
    candidate_ids: torch.Tensor,
    block_table: torch.Tensor,
    sequence_lengths: torch.Tensor,
    *,
    source_block_size: int,
    pool_size: int = 4,
    topk_tokens: int = 2048,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Merge rank candidates and return sorted logical/physical token rows."""

    if candidate_scores.shape != candidate_ids.shape or candidate_scores.ndim != 2:
        raise ValueError("candidate scores and ids must share [B,K]")
    if block_table.shape[0] != candidate_scores.shape[0]:
        raise ValueError("block table and candidate batch sizes must match")
    if sequence_lengths.shape != (candidate_scores.shape[0],):
        raise ValueError("sequence_lengths must match candidate batch")
    top_pools = math.ceil(topk_tokens / pool_size)
    if candidate_scores.shape[1] < top_pools:
        raise ValueError("not enough candidates for requested sparse top-k")
    running_scores, running_ids = _r590_topk(candidate_scores, candidate_ids.long(), top_pools)
    running_ids = torch.where(
        torch.isfinite(running_scores),
        running_ids,
        torch.full_like(running_ids, -1),
    )

    batch = candidate_scores.shape[0]
    complete_pools = torch.div(sequence_lengths.long(), pool_size, rounding_mode="floor")
    pool_offsets = torch.arange(pool_size, device=candidate_scores.device)
    source_tokens = (
        running_ids.unsqueeze(-1) * pool_size + pool_offsets
    ).reshape(batch, -1)
    source_tokens = source_tokens.masked_fill(
        running_ids.unsqueeze(-1).expand(-1, -1, pool_size).reshape(batch, -1) < 0,
        -1,
    )

    tail_width = pool_size - 1
    tail_offsets = torch.arange(tail_width, device=candidate_scores.device)
    tail_start = complete_pools * pool_size
    tail = tail_start.unsqueeze(1) + tail_offsets.unsqueeze(0)
    tail = tail.masked_fill(tail >= sequence_lengths.long().unsqueeze(1), -1)
    source_tokens = torch.cat((source_tokens, tail), dim=1)
    sentinel = torch.iinfo(torch.int32).max
    sort_key = torch.where(source_tokens >= 0, source_tokens, sentinel)
    source_tokens, _ = torch.sort(sort_key, dim=1)
    source_tokens = torch.where(source_tokens == sentinel, -1, source_tokens)
    physical_tokens = logical_to_physical_slots(
        source_tokens, block_table, source_block_size
    )
    return source_tokens.to(torch.int32), physical_tokens.to(torch.int32)


def streaming_pooled_topk(
    query_values: torch.Tensor,
    head_weights: torch.Tensor,
    packed_cache: torch.Tensor,
    block_table: torch.Tensor,
    sequence_lengths: torch.Tensor,
    *,
    source_block_size: int,
    pool_size: int = 4,
    topk_tokens: int = 2048,
    context_chunk_pools: int = 4096,
    max_complete_pools_bucket: int | None = None,
    shared_block_table: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Select sparse source tokens while bounding temporary HBM usage.

    Queries are already FWHT-rotated/FP8-quantized and their dequant scale plus
    the model's head weighting is folded into ``head_weights``.  The persistent
    cache remains packed; only one bounded pool chunk is gathered/dequantized at
    a time.  Returns logical and physical source-token rows, both sorted and
    ``-1`` padded.  The incomplete raw tail is always appended.
    """

    if shared_block_table and block_table.shape[0] != query_values.shape[0]:
        raise ValueError("shared scorer needs the full query table for finalization")
    top_pools = math.ceil(topk_tokens / pool_size)
    if _fused_select_applies(
        max_complete_pools_bucket=max_complete_pools_bucket,
        context_chunk_pools=context_chunk_pools,
        top_pools=top_pools,
    ):
        _validate_scan_inputs(
            query_values, head_weights,
            block_table[:1] if shared_block_table else block_table,
            sequence_lengths, source_block_size=source_block_size,
            pool_size=pool_size, context_chunk_pools=context_chunk_pools,
            tp_rank=0, tp_size=1, shared_block_table=shared_block_table,
        )
        return _fused_single_chunk_pooled_topk(
            (query_values, head_weights, packed_cache), block_table,
            sequence_lengths,
            (source_block_size, pool_size, top_pools, int(max_complete_pools_bucket)),
            shared_block_table,
        )
    candidate_scores, candidate_ids = streaming_pooled_topk_candidates(
        query_values,
        head_weights,
        packed_cache,
        block_table[:1] if shared_block_table else block_table,
        sequence_lengths,
        source_block_size=source_block_size,
        pool_size=pool_size,
        topk_tokens=topk_tokens,
        context_chunk_pools=context_chunk_pools,
        max_complete_pools_bucket=max_complete_pools_bucket,
        shared_block_table=shared_block_table,
    )
    return finalize_pooled_topk(
        candidate_scores,
        candidate_ids,
        block_table,
        sequence_lengths,
        source_block_size=source_block_size,
        pool_size=pool_size,
        topk_tokens=topk_tokens,
    )


def sparse_mla_decode_from_slots(
    q_nope: torch.Tensor,
    q_pe: torch.Tensor,
    latent_cache: torch.Tensor,
    physical_slots: torch.Tensor,
    *,
    kv_lora_rank: int,
    scale: float,
) -> torch.Tensor:
    """Exact selected-token MLA decode without fetching the full context."""

    if q_nope.ndim != 3 or q_pe.ndim != 3 or physical_slots.ndim != 2:
        raise ValueError("q_nope/q_pe must be [B,H,D] and slots [B,K]")
    if q_nope.shape[:2] != q_pe.shape[:2] or q_nope.shape[0] != physical_slots.shape[0]:
        raise ValueError("query and slot batches must match")
    valid = physical_slots >= 0
    flat_cache = latent_cache.reshape(-1, latent_cache.shape[-1])
    selected = flat_cache.index_select(
        0, physical_slots.long().clamp_min(0).reshape(-1)
    ).reshape(*physical_slots.shape, flat_cache.shape[-1])
    return sparse_mla_from_selected(
        q_nope,
        q_pe,
        selected,
        valid,
        kv_lora_rank=kv_lora_rank,
        scale=scale,
    )


@_r589_stage("mla.tile")
def sparse_mla_from_selected(
    q_nope: torch.Tensor,
    q_pe: torch.Tensor,
    selected: torch.Tensor,
    valid: torch.Tensor,
    *,
    kv_lora_rank: int,
    scale: float,
) -> torch.Tensor:
    """Selected-token MLA once the bounded gather/dequant has completed.

    Keeping this separate lets the HPU runtime gather only the selected FP8
    latent rows, dequantize that bounded tensor, and never expand the full
    long-context cache.
    """

    if selected.ndim != 3 or valid.shape != selected.shape[:2]:
        raise ValueError("selected must be [B,K,D] and valid must be [B,K]")
    if q_nope.shape[0] != selected.shape[0]:
        raise ValueError("query and selected cache batches must match")
    latent, rope = selected.split(
        [kv_lora_rank, selected.shape[-1] - kv_lora_rank], dim=-1
    )
    if _USE_BF16_SPARSE_MLA_SCORE and latent.dtype == torch.bfloat16:
        # Gaudi MME accumulates BF16 GEMM products without first materializing
        # the complete selected cache in FP32. Convert only the bounded logits
        # to FP32 for scaling and softmax. Capability-check the rotary width so
        # NoPE models do not launch an empty second GEMM.
        logits = torch.matmul(q_nope.to(latent.dtype), latent.transpose(1, 2))
        if rope.shape[-1] > 0:
            logits = logits + torch.matmul(
                q_pe.to(rope.dtype), rope.transpose(1, 2)
            )
        logits = logits.float()
    else:
        logits = torch.matmul(q_nope.float(), latent.float().transpose(1, 2))
        if rope.shape[-1] > 0:
            logits = logits + torch.matmul(
                q_pe.float(), rope.float().transpose(1, 2)
            )
    logits = (logits * float(scale)).masked_fill(~valid.unsqueeze(1), -torch.inf)
    probs = torch.softmax(logits, dim=-1)
    probs = torch.nan_to_num(probs, nan=0.0)
    return torch.matmul(probs.to(latent.dtype), latent)
