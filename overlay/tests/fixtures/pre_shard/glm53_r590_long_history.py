"""Isolated R590: explicit mask capabilities and <=128-row sparse indexing.

KDA, selected MLA, cache updates, scores, and collectives remain R589.
Equal-score top-k ties use logical ID ascending so query partition is neutral.
This module does not authorize an image build, deployment, or promotion.
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, replace
import inspect
import logging
import os

import torch

import glm53_r589_bounded_prefill as r589
from glm53_mixed_graph import MIB, capture_active as mixed_graph_capture_active
from glm53_mixed_graph import key_capture_active, positive_env, rank_reports

QUERY_TILE = 128
KDA = "kda_causal_padding_v1"
SELECTED = "selected_rows_causal_padding_v1"
CACHE = "index_storage_no_prompt_bias_v1"
INDEXER = "indexer_causal_positions_v1"
CONTRACTS = frozenset((KDA, SELECTED, CACHE, INDEXER))
LOG = logging.getLogger("vllm.r590")


class R590ContractError(RuntimeError):
    """Missing capability or an unreviewed bounded-execution mode."""


def enabled() -> bool:
    value = os.environ.get("HPU_GLM53_R590_ENABLE", "0")
    if value not in ("0", "1"):
        raise R590ContractError("HPU_GLM53_R590_ENABLE must be 0 or 1")
    if value == "1" and not r589.enabled():
        raise R590ContractError("R590 requires HPU_GLM53_R589_ENABLE=1")
    return value == "1"


def _capability(module):
    value = getattr(module, "r590_prompt_bias_contract", None)
    return value() if callable(value) else value


def _is_prompt_bias_consumer(module, attention_base) -> bool:
    """Registry membership is not an attention capability (MoE also registers).

    Attention/cache interfaces and explicit declarations are authoritative.
    A custom layer exposing mask/attention metadata in its forward ABI also
    enters the fail-closed census, even without AttentionLayerBase inheritance.
    No model, class-name, or registry-path allowlist is used.
    """
    if (isinstance(module, attention_base)
            or callable(getattr(module, "get_attn_backend", None))
            or callable(getattr(module, "get_kv_cache_spec", None))
            or hasattr(module, "r590_prompt_bias_contract")
            or getattr(module, "consumes_attention_mask", False) is True
            or getattr(module, "consumes_prompt_bias", False) is True):
        return True
    forward = getattr(module, "forward", None)
    if not callable(forward):
        return False
    try:
        parameters = inspect.signature(forward).parameters
    except (TypeError, ValueError):
        return False
    return not {"attention_mask", "attn_bias", "attention_bias", "prompt_bias",
                "attn_metadata", "attention_metadata"}.isdisjoint(parameters)


def selected_mla_capability(module):
    if (getattr(module, "r590_selected_row_consumer", False) is True
            and isinstance(getattr(module, "hpu_sparse_topk_indices_buffer", None), torch.Tensor)):
        return SELECTED
    return None


def selected_wrapper_capability(module):
    if (getattr(module, "hpu_glm5next_sparse", False) is True
            and not getattr(module, "hpu_glm53_force_dense_mla", False)
            and _capability(getattr(module, "indexer", None)) == INDEXER
            and _capability(getattr(module, "mla_attn", None)) == SELECTED
            and isinstance(getattr(module, "topk_indices_buffer", None), torch.Tensor)):
        return SELECTED
    return None


@dataclass(frozen=True)
class MaskCertificate:
    runner: object
    roots: tuple
    registry: object
    registered: tuple
    consumers: tuple
    bindings: tuple
    attention_base: object
    non_consumers: tuple
    errors: tuple

    def valid(self) -> bool:
        if self.errors or not self.consumers:
            return False
        current_roots = [("target", getattr(self.runner, "model", None))]
        if hasattr(self.runner, "drafter"):
            current_roots.append(("drafter", getattr(self.runner.drafter, "model", None)))
        if len(current_roots) != len(self.roots):
            return False
        if any(a != b or model is not prior
               for (a, model), (b, prior) in zip(current_roots, self.roots)):
            return False
        registry = self.runner.vllm_config.compilation_config.static_forward_context
        if registry is not self.registry or tuple((name, id(module)) for name, module in registry.items()) != self.registered:
            return False
        if any(getattr(owner, "self_attn", None) is not module for owner, module in self.bindings):
            return False
        if any(_is_prompt_bias_consumer(module, self.attention_base) for module in self.non_consumers):
            return False
        return all(_capability(module) in CONTRACTS for _, module in self.consumers)


def certify_runner(runner, attention_base=None) -> MaskCertificate | None:
    """Census attention consumers in target, drafter, and the static registry.

    Unknown consumers, an unloaded/opaque drafter, empty attention trees, or
    later registry/model replacement disable omission. No model-name test is
    involved. Storage-only attention/cache entries must declare their role too;
    registered MoE/projection modules are not attention consumers by default.
    """
    if not enabled():
        return None
    if attention_base is None:
        from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase
        attention_base = AttentionLayerBase
    roots = [("target", getattr(runner, "model", None))]
    if hasattr(runner, "drafter"):
        roots.append(("drafter", getattr(runner.drafter, "model", None)))
    errors, consumers, processors, bindings = [], {}, [], []
    if getattr(runner, "metadata_processor", None) is not None:
        processors.append(runner.metadata_processor)
    if getattr(runner.vllm_config, "speculative_config", None) is not None and len(roots) == 1:
        errors.append("configured drafter is not loaded")
    for label, root in roots:
        if root is None or not callable(getattr(root, "named_modules", None)):
            errors.append(label + ": no inspectable loaded model")
            continue
        attention_count = 0
        for name, module in root.named_modules():
            processor = getattr(module, "metadata_processor", None)
            if processor is not None:
                processors.append(processor)
            self_attn = getattr(module, "self_attn", None)
            if self_attn is not None:
                bindings.append((module, self_attn))
                consumers[id(self_attn)] = (label + ":" + name + ".self_attn", self_attn)
                if _capability(self_attn) not in (KDA, SELECTED):
                    errors.append(label + ":" + name + ".self_attn: uncertified consumer")
            if _is_prompt_bias_consumer(module, attention_base):
                consumers[id(module)] = (label + ":" + name, module)
                capability = _capability(module)
                if capability not in CONTRACTS:
                    errors.append(label + ":" + name + ": uncertified attention consumer")
                if capability in (KDA, SELECTED):
                    attention_count += 1
        if not attention_count:
            errors.append(label + ": no certified attention consumers")
    registry = runner.vllm_config.compilation_config.static_forward_context
    non_consumers = []
    for name, module in registry.items():
        if not _is_prompt_bias_consumer(module, attention_base):
            non_consumers.append(module)
            continue
        consumers[id(module)] = ("registered:" + name, module)
        if _capability(module) not in CONTRACTS:
            errors.append("registered:" + name + ": uncertified consumer")
    if not registry:
        errors.append("empty attention registry")
    certificate = MaskCertificate(runner, tuple(roots), registry,
        tuple((name, id(module)) for name, module in registry.items()),
        tuple(consumers.values()), tuple(bindings), attention_base,
        tuple(non_consumers), tuple(errors))
    runner._r590_mask_certificate = certificate
    for processor in processors:
        processor._r590_mask_certificate = certificate
    LOG.warning(
        "R590 mask certificate valid=%s consumers=%d non_consumers=%d roots=%s errors=%s",
        certificate.valid(), len(consumers),
        len(non_consumers), [label for label, _ in roots], errors,
    )
    r589._emit("r590.mask.census", consumers=len(consumers), non_consumers=len(non_consumers),
               roots=[label for label, _ in roots], errors=errors)
    return certificate


def omit_dense_prompt_bias(processor, metadata) -> bool:
    """Return before generic allocation only with a live, all-consumer proof.

    Supplied masks are never removed. Unknown/mixed models retain R589's
    original allocator guard. Sliding/chunked/merged attention stays excluded.
    """
    if (not enabled() or metadata is None
            or not getattr(metadata, "is_prompt", False)
            or getattr(metadata, "attn_bias", None) is not None):
        return False
    certificate = getattr(processor, "_r590_mask_certificate", None)
    if not isinstance(certificate, MaskCertificate) or not certificate.valid():
        return False
    if (getattr(processor, "sliding_window", None)
            or getattr(certificate.runner, "use_merged_prefill", False)
            or getattr(certificate.runner, "model_has_chunked_attention", False)):
        return False
    r589._emit("r590.mask.omitted", consumers=len(certificate.consumers))
    return True


def guard_mla_fallback(metadata, physical_slots, query_rows) -> None:
    """Never enter dense MLA without bias after a sparse-only omission.

    Applied before cache mutation and again immediately before dense fallback.
    R590 intentionally fails closed on an unmasked dense prompt, even if a
    different generic FSDPA route could synthesize its own causal mask.
    """
    if not enabled() or not getattr(metadata, "is_prompt", False):
        return
    if getattr(metadata, "attn_bias", None) is not None:
        return
    if (not isinstance(physical_slots, torch.Tensor)
            or physical_slots.ndim != 2 or physical_slots.shape[0] < query_rows
            or physical_slots.shape[1] < 1
            or physical_slots.dtype not in (torch.int32, torch.int64)):
        raise R590ContractError("R590 unmasked prompt requires selected rows; dense MLA fallback refused")


def canonical_topk(scores, logical_ids, count):
    """Exact score ordering, then lower logical ID, without score perturbation.

    Two stable sorts give a lexicographic comparator at both streaming merge
    and finalization. Work is bounded by query-tile x (context-tile + top-k),
    never by full history. Unsupported HPU stable sort fails, never falls back
    to partition-dependent ties. Disabled R590 retains the original topk.
    """
    if not enabled():
        values, selected = torch.topk(scores, k=count, dim=1, sorted=False)
        return values, torch.gather(logical_ids, 1, selected)
    valid = (logical_ids >= 0) & torch.isfinite(scores)
    scores = scores.masked_fill(~valid, -torch.inf)
    # The HPU bridge lowers the scalar branch of ``where`` through a signed
    # 32-bit immediate even when finalization has promoted logical IDs to
    # int64.  INT64_MAX therefore aborts the lazy graph at mark_step.  Model
    # logical IDs are bounded below max_model_len (1,048,576 here), so that
    # exact, int32-safe ceiling is an invalid sentinel for both tensor dtypes.
    if logical_ids.dtype not in (torch.int32, torch.int64):
        raise R590ContractError("R590 canonical top-k requires int32/int64 logical IDs")
    keys = torch.where(valid, logical_ids, 1_048_576)
    by_id = torch.argsort(keys, dim=1, stable=True)
    ordered_scores = torch.gather(scores, 1, by_id)
    by_score = torch.argsort(ordered_scores, dim=1, descending=True, stable=True)[:, :count]
    selected = torch.gather(by_id, 1, by_score)
    values = torch.gather(scores, 1, selected)
    ids = torch.gather(logical_ids, 1, selected)
    return values, torch.where(torch.isfinite(values), ids, -1)


@dataclass(frozen=True)
class QuerySegment:
    start: int
    end: int
    block_table: torch.Tensor
    row_map: torch.Tensor | None = None
    repeat: int | None = None
    cumulative_lens: torch.Tensor | None = None
    max_sequence_length: int | None = None
    max_pool_cap: int | None = None


def _owned_segment(start, end, table, **kwargs):
    # Copy only request-level tables. Query-level tables are built INSIDE the
    # tile. Full query-row maps are compact 1-D integers, never [T, history].
    row_map = kwargs.pop("row_map", None)
    cumulative = kwargs.pop("cumulative_lens", None)
    if table.ndim != 2 or table.shape[0] > QUERY_TILE:
        raise R590ContractError("R590 requires <=128 request-level block-table rows before expansion")
    return QuerySegment(start, end, table.clone(),
        row_map=row_map.reshape(-1).clone() if row_map is not None else None,
        cumulative_lens=cumulative.clone() if cumulative is not None else None,
        **kwargs)


def _query_table(segment, local_start, local_end):
    if segment.row_map is not None:
        rows = segment.row_map[local_start:local_end].long()
    else:
        rows = torch.arange(local_start, local_end, device=segment.block_table.device, dtype=torch.long)
        if segment.repeat is not None:
            rows = torch.div(rows, segment.repeat, rounding_mode="floor")
        elif segment.cumulative_lens is not None:
            rows = torch.searchsorted(segment.cumulative_lens, rows, right=True)
    return segment.block_table.index_select(0, rows)


def _query_tile_has_shared_request(segment: QuerySegment, local_start: int, local_end: int,
                                   tile_rows: int = QUERY_TILE) -> bool:
    """Certify rectangular request ownership using host metadata only.

    Shape alone never establishes ownership: row-map and cumulative-length
    routes retain the existing scorer until they supply a host certificate.
    """
    repeat = segment.repeat
    if (segment.row_map is not None or segment.cumulative_lens is not None
            or type(repeat) is not int or repeat <= 0
            or segment.block_table.ndim != 2
            or segment.end - segment.start != repeat * segment.block_table.shape[0]
            or not 0 <= local_start < local_end <= segment.end - segment.start
            or local_end - local_start != tile_rows):
        return False
    return local_start // repeat == (local_end - 1) // repeat


def _static_long_prefill_pool_cap(table, query_rows, source_block_size, pool_size, cap):
    """Keep adaptive shared scoring inside the fixed long-prompt graph geometry."""
    if (os.getenv("HPU_GLM53_SPARSE_PREFILL_BLOCK_TIER_640", "0") == "1"
            and tuple(table.shape) == (1, 640) and query_rows == 1792
            and source_block_size == 128 and pool_size == 4):
        capacity = table.shape[1] * source_block_size // pool_size
        return capacity if cap is None else min(cap, capacity)
    return cap


def _true_host_large_prefill_bound(metadata, n_buffer, pool_size, fallback_length, fallback_cap):
    """Default-off tightening only for certified outer-graph-bypassed prefill."""
    if os.environ.get("HPU_GLM53_TRUE_HOST_PREFILL_SCAN_BOUND", "0") != "1":
        return fallback_length, fallback_cap
    try:
        bypass_min = int(os.environ.get("HPU_GLM53_Q4_LARGE_PREFILL_BYPASS_GRAPH_MIN_TOKENS", "0"))
    except (TypeError, ValueError):
        return fallback_length, fallback_cap
    token_mask = getattr(metadata, "glm53_token_mask", None)
    true_max = getattr(metadata, "glm53_true_host_max_seq_len", None)
    if (getattr(metadata, "is_prompt", False) is not True
            or type(n_buffer) is not int or bypass_min < 512 or n_buffer < bypass_min
            or pool_size != 4 or not isinstance(token_mask, torch.Tensor)
            or token_mask.numel() != n_buffer
            or type(true_max) is not int or not 0 < true_max <= fallback_length):
        return fallback_length, fallback_cap
    # Keep physical block-table buckets intact. max_pool_cap prevents the
    # existing 64K-ceiling tier selector from rounding this back to32K pools.
    complete_pools = true_max // pool_size
    if complete_pools == 0:
        return fallback_length, fallback_cap
    rounded_pools = ((complete_pools + 4095) // 4096) * 4096
    cap = rounded_pools if fallback_cap is None else min(fallback_cap, rounded_pools)
    return true_max, cap


ADAPTIVE_TILE_ROWS = (512, 256, QUERY_TILE)
TILE_ROWS_ENV = "HPU_GLM53_PREFILL_TILE_ROWS"
MLA_TILE_BUDGET_ENV = "HPU_GLM53_MLA_TILE_BUDGET_MIB"
WARMUP_ENV = "HPU_GLM53_CAP_LEGACY_WARMUP"
BOOT_WARM_ENV = "HPU_GLM53_WARM_ADAPTIVE_TILES"


def _parse_tile_ceiling(value: str | None) -> int:
    """Strangler-fig ceiling for prefill query tiles; unset keeps 128 rows."""
    if value is None:
        return QUERY_TILE
    if value not in tuple(str(rows) for rows in ADAPTIVE_TILE_ROWS):
        raise R590ContractError(
            f"{TILE_ROWS_ENV}={value!r} must be one of 128, 256 or 512; unset it to keep 128")
    return int(value)


def _parse_mla_budget(value: str | None) -> int | None:
    """MLA tile workspace budget in bytes; unset keeps the base fetch tile."""
    if value is None:
        return None
    if not value.isdigit() or int(value) <= 0:
        raise R590ContractError(f"{MLA_TILE_BUDGET_ENV}={value!r} must be a positive MiB integer")
    return int(value) * 2**20


def _parse_boot_warm(value: str | None) -> bool:
    """Opt-in post-capture warm of widened indexer recipes; unset keeps live compiles."""
    if value is None or value == "0":
        return False
    if value != "1":
        raise R590ContractError(f"{BOOT_WARM_ENV}={value!r} must be 0 or 1; unset it to keep the default")
    return True


# Parsed once at import so a malformed value fails at boot, never mid-prefill.
PREFILL_TILE_CEILING = _parse_tile_ceiling(os.environ.get(TILE_ROWS_ENV))
MLA_TILE_BUDGET_BYTES = _parse_mla_budget(os.environ.get(MLA_TILE_BUDGET_ENV))
BOOT_WARM = _parse_boot_warm(os.environ.get(BOOT_WARM_ENV))
_WARMED_TILES: set[tuple] = set()


@dataclass(frozen=True)
class AdaptiveTile:
    """One widened shared indexer tile: query rows, kpool bucket, table columns."""
    rows: int
    bucket: int
    columns: int


def adaptive_tile(op: object, *, max_length: int, max_pool_cap: int | None,
                  block_columns: int, repeat: int, index_heads: int) -> AdaptiveTile | None:
    """Largest request-shared tile that scores in ONE chunk, or ``None``.

    The tile's table is sliced or ``-1``-padded to ``columns`` (covering the
    canonical bucket), so its recipe depends only on ``(rows, bucket)``;
    :func:`adaptive_warm_plan` enumerates exactly those.  ``block_columns``
    must already cover ``max_length`` tokens (all real rows), else ``None``.
    """
    if (PREFILL_TILE_CEILING <= QUERY_TILE
            or os.getenv("HPU_GLM53_SHARED_REQUEST_KPOOL", "0") != "1"
            or not callable(getattr(op, "canonical_shared_scan", None))
            or max_length <= op.topk_tokens):
        return None
    scan = op.canonical_shared_scan(max_length, max_pool_cap)
    if scan is None or block_columns * op.source_block_size < max_length:
        return None
    for rows in ADAPTIVE_TILE_ROWS:
        if (QUERY_TILE < rows <= PREFILL_TILE_CEILING and repeat % rows == 0
                and op.shared_scan_fits_one_chunk(rows=rows, bucket=scan[0], index_heads=index_heads)):
            return AdaptiveTile(rows, *scan)
    return None


def adaptive_warm_plan(op: object, index_heads: int) -> tuple[AdaptiveTile, ...]:
    """Every tile :func:`adaptive_tile` can return, for any context length.

    Buckets are a step function of complete pools ``p``; jumping to
    ``bucket(p) + 1`` visits each distinct bucket once, with and without the
    true-host 4096-pool cap. The scan stops once no widened tile fits.
    """
    plan: dict[tuple[int, int], AdaptiveTile] = {}
    for capped in (True, False):
        pools = op.topk_tokens // op.pool_size + 1
        while True:
            cap = ((pools + 4095) // 4096) * 4096 if capped else None
            tiles = [adaptive_tile(op, max_length=pools * op.pool_size, max_pool_cap=cap,
                                   block_columns=2**31, repeat=rows, index_heads=index_heads)
                     for rows in ADAPTIVE_TILE_ROWS[:-1]]
            tiles = [tile for tile in tiles if tile is not None]
            if not tiles:
                break
            plan.update({(tile.rows, tile.bucket): tile for tile in tiles})
            pools = tiles[0].bucket + 1
    return tuple(plan[key] for key in sorted(plan))


@dataclass(frozen=True)
class TileInputs:
    """Owned per-forward indexer inputs shared by every tile of every segment."""
    q_values: torch.Tensor
    weights: torch.Tensor
    positions: torch.Tensor
    token_mask: torch.Tensor | None
    width: int


def _tile_plan(op: object, segment: QuerySegment, max_length: int, tile_rows: int,
               q_values: torch.Tensor) -> tuple[QuerySegment, int]:
    """Segment (table sliced and committed when widened) and its tile rows.

    The request-level table is fitted to the tile's columns, committed once,
    so every widened tile recipe is independent of the runner's table width.
    Exact: columns past the real table only add pools at or beyond every
    row's complete-pool count (masked to -inf) and padded ``-1`` blocks no
    selected token references.  Unwidened segments are returned untouched.
    """
    wide = None
    if segment.row_map is None and segment.cumulative_lens is None and type(segment.repeat) is int:
        wide = adaptive_tile(op, max_length=max_length, max_pool_cap=segment.max_pool_cap,
                             block_columns=int(segment.block_table.shape[1]),
                             repeat=segment.repeat, index_heads=int(q_values.shape[1]))
    if wide is None:
        return segment, tile_rows
    table = segment.block_table
    if table.shape[1] >= wide.columns:
        table = table[:, :wide.columns].clone()
    else:
        table = torch.cat((table, table.new_full((table.shape[0], wide.columns - table.shape[1]), -1)), dim=1)
    segment = replace(segment, block_table=table)
    r589._boundary("r590.indexer.wide_table", q_values, drain=False, columns=wide.columns)
    return segment, wide.rows


def _score_tile(op: object, segment: QuerySegment, span: tuple[int, int, int],
                inputs: TileInputs, max_length: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Selected physical rows for one query tile ``span = (start, end, rows)``."""
    start, end, segment_rows = span
    query_table = _query_table(segment, start - segment.start, end - segment.start)
    lengths = inputs.positions[start:end].long() + 1
    if inputs.token_mask is not None:
        lengths = torch.where(inputs.token_mask[start:end].bool(), lengths, torch.zeros_like(lengths))
    shared_keys = (os.getenv("HPU_GLM53_SHARED_REQUEST_KPOOL", "0") == "1"
        and not op.tp_sharded_kpool and not op.compiled_kpool
        and _query_tile_has_shared_request(
            segment, start - segment.start, end - segment.start, segment_rows))
    physical = op._write_topk(slice(start, end), inputs.q_values[start:end], inputs.weights[start:end],
        query_table, lengths, max_pool_cap=segment.max_pool_cap,
        max_sequence_length=max_length,
        **({"shared_block_table": True} if shared_keys else {})).clone()
    if physical.shape != (end - start, inputs.width) or physical.dtype != torch.int32:
        raise R590ContractError("R590 scorer returned an incompatible selected-row tile")
    return physical, query_table, lengths


def warm_adaptive_tiles(op: object, inputs: TileInputs, table: torch.Tensor) -> int:
    """Compile every widened indexer recipe during runner warmup; returns count.

    Only inside the runner's warmup forwards (HPU_GLM53_CAP_LEGACY_WARMUP=1),
    once per shape per process, and never inside a mixed HPUGraph capture:
    there the warm tiles would be recorded into the graph and replayed on
    every step.  Each plan tile runs the live tile body over this forward's
    own owned inputs as one single-request segment, so the recipes match a
    live single-request prefill of the same token bucket.
    """
    if (PREFILL_TILE_CEILING <= QUERY_TILE or os.getenv(WARMUP_ENV, "0") != "1"
            or mixed_graph_capture_active()):
        return 0
    return _warm_plan_tiles(op, inputs, table)


def _warm_key(op: object, inputs: TileInputs, table: torch.Tensor, bucket: int) -> tuple:
    """Everything a widened tile recipe depends on besides the plan tile."""
    cache = op.k_cache.kv_cache
    mask = None if inputs.token_mask is None else str(inputs.token_mask.dtype)
    return ("indexer", bucket, int(inputs.q_values.shape[0]), str(inputs.q_values.dtype),
            str(inputs.weights.dtype), str(inputs.positions.dtype), mask, str(table.dtype),
            tuple(cache.shape), str(cache.dtype))


def _warm_plan_tiles(op: object, inputs: TileInputs, table: torch.Tensor) -> int:
    """Run the live tile body once per unwarmed plan bucket; returns count."""
    rows_total = int(inputs.q_values.shape[0])
    warmed = 0
    plan = {tile.bucket: tile.columns for tile in adaptive_warm_plan(op, int(inputs.q_values.shape[1]))}
    for bucket, columns in sorted(plan.items()):
        # BOOT_WARM unset keeps the base's coarse key, so flag-off warms exactly what the base warms.
        key = (_warm_key(op, inputs, table, bucket) if BOOT_WARM
               else ("indexer", bucket, rows_total, str(inputs.q_values.dtype), str(table.dtype)))
        if key in _WARMED_TILES:
            continue
        _WARMED_TILES.add(key)
        max_length = bucket * op.pool_size
        segment = QuerySegment(0, rows_total, table.new_zeros((1, columns)), repeat=rows_total,
                               max_sequence_length=max_length, max_pool_cap=bucket)
        r589._boundary("r590.adaptive.warm_table", inputs.q_values, drain=False)
        segment, rows = _tile_plan(op, segment, max_length, QUERY_TILE, inputs.q_values)
        if rows == QUERY_TILE:
            continue
        for start in range(0, rows_total, rows):
            _score_tile(op, segment, (start, start + rows, rows), inputs, max_length)
            r589._boundary("r590.adaptive.warm", inputs.q_values, start=start, rows=rows)
        warmed += 1
    return warmed


@dataclass(frozen=True)
class TilePrototype:
    """Host-only dtypes and trailing shapes of a live single-request R590 prefill."""
    q_dtype: torch.dtype
    q_tail: tuple[int, ...]
    weights_dtype: torch.dtype
    weights_tail: tuple[int, ...]
    positions_dtype: torch.dtype
    mask_dtype: torch.dtype | None
    table_dtype: torch.dtype
    device: torch.device


_TILE_PROTOTYPES: set[TilePrototype] = set()


def record_tile_prototype(inputs: TileInputs, segment: QuerySegment) -> None:
    """Remember a widenable forward's input dtypes for the boot warm.

    Reads tensor metadata only (no device op), so it is safe inside capture;
    the mixed boot capture of a prefill bucket is what records it.
    """
    if not BOOT_WARM or segment.row_map is not None or segment.cumulative_lens is not None:
        return
    mask = inputs.token_mask
    _TILE_PROTOTYPES.add(TilePrototype(
        inputs.q_values.dtype, tuple(inputs.q_values.shape[1:]), inputs.weights.dtype,
        tuple(inputs.weights.shape[1:]), inputs.positions.dtype,
        None if mask is None else mask.dtype, segment.block_table.dtype, inputs.q_values.device))


def _prototype_inputs(prototype: TilePrototype, rows: int, width: int) -> tuple[TileInputs, torch.Tensor]:
    """Owned, committed ``rows``-token inputs with the live dtypes, plus a table dtype carrier."""
    device = prototype.device
    q_values = torch.zeros((rows, *prototype.q_tail), dtype=prototype.q_dtype, device=device)
    weights = torch.zeros((rows, *prototype.weights_tail), dtype=prototype.weights_dtype, device=device)
    positions = torch.arange(rows, dtype=prototype.positions_dtype, device=device)
    mask = (None if prototype.mask_dtype is None
            else torch.ones(rows, dtype=prototype.mask_dtype, device=device))
    r589._boundary("r590.adaptive.boot_inputs", q_values, drain=False, rows=rows)
    table = torch.zeros((1, 1), dtype=prototype.table_dtype, device=device)
    return TileInputs(q_values, weights, positions, mask, width), table


def _boot_warm_ops(model: object) -> list:
    """One sparse indexer op per distinct allocated kpool cache signature."""
    named_modules = getattr(model, "named_modules", None)
    ops: dict[tuple, object] = {}
    for _, module in (named_modules() if callable(named_modules) else ()):
        cache = getattr(getattr(module, "k_cache", None), "kv_cache", None)
        if callable(getattr(module, "canonical_shared_scan", None)) and isinstance(cache, torch.Tensor):
            ops.setdefault((tuple(cache.shape), cache.dtype, cache.device), module)
    return list(ops.values())


def _boot_warm_rows() -> tuple[int, ...]:
    """Single-request mixed prefill widths that can widen: multiples of 256 up to the budget."""
    from glm53_mixed_step import prefill_budget
    step = min(rows for rows in ADAPTIVE_TILE_ROWS if rows > QUERY_TILE)
    return tuple(range(step, prefill_budget() + 1, step))


def _device_free_hbm() -> int:
    """Commit pending lazy work, then read free HBM bytes from the bridge."""
    import habana_frameworks.torch as htorch
    htorch.core.mark_step()
    htorch.hpu.synchronize()
    free, total = htorch.hpu.mem_get_info()
    if not isinstance(free, int) or not 0 <= free <= total:
        raise ValueError("mem_get_info returned invalid free/total bytes")
    return free


def _boot_warm_error(ops: list) -> str:
    if not ops:
        return "no sparse indexer op has an allocated kpool cache; warm only after KV allocation"
    if not _TILE_PROTOTYPES:
        return ("no single-request R590 prefill was traced before the warm; keep a prefill "
                "bucket in VLLM_GAUDI_MIXED_GRAPH_WARMUP")
    return ""


def _admit_boot_warm(ops: list, free_hbm: Callable[[], int]) -> tuple[int, int]:
    """Rank-consistent HBM admission: every rank warms, or every rank raises.

    Required = the largest one-chunk shared scorer budget (the bound every
    widened tile satisfies) + VLLM_GAUDI_MIXED_GRAPH_RESERVE_MIB emergency HBM.
    """
    reserve = positive_env("VLLM_GAUDI_MIXED_GRAPH_RESERVE_MIB", 2048) * MIB
    required = max((int(op.shared_kpool_budget_mib) * MIB for op in ops), default=0) + reserve
    error, free = _boot_warm_error(ops), 0
    if not error:
        try:
            free = int(free_hbm())
        except Exception as exc:  # an unreadable allocator must refuse, never warm blind
            error = f"free HBM query failed: {type(exc).__name__}: {exc}"
    problems = []
    for rank, report in enumerate(rank_reports(dict(free=free, required=required, error=error))):
        if report["error"]:
            problems.append(f"rank {rank}: {report['error']}")
        elif report["free"] < report["required"]:
            problems.append(f"rank {rank}: free HBM {report['free'] / MIB:.1f} MiB < "
                            f"required {report['required'] / MIB:.1f} MiB")
    if problems:
        raise R590ContractError(
            f"{BOOT_WARM_ENV}=1 refused on ALL ranks: " + "; ".join(problems)
            + ". Lower HPU_GLM53_SHARED_KPOOL_CHUNK_BUDGET_MIB or HPU_GLM53_PREFILL_TILE_ROWS, "
            f"leave more HBM free (--num-gpu-blocks-override), or unset {BOOT_WARM_ENV}; restart all workers.")
    return free, required


def _boot_warm_op(op: object, rows_totals: tuple[int, ...]) -> int:
    width = op.topk_tokens + op.pool_size - 1
    warmed = 0
    for prototype in sorted(_TILE_PROTOTYPES, key=repr):
        for rows in (rows for rows in rows_totals if rows <= op.topk_indices_buffer.shape[0]):
            inputs, table = _prototype_inputs(prototype, rows, width)
            warmed += _warm_plan_tiles(op, inputs, table)
    return warmed


def warm_adaptive_tiles_at_boot(model: object, *, free_hbm: Callable[[], int] | None = None) -> int:
    """Compile widened indexer recipes after KV allocation, outside capture.

    Opt-in (HPU_GLM53_WARM_ADAPTIVE_TILES=1); unset is a no-op with no
    collective.  ``glm53_mixed_graph.warmup`` calls it after the boot
    captures, whose prefill bucket records the live input dtypes.  For every
    distinct kpool cache it runs the live tile body for each
    :func:`adaptive_warm_plan` bucket and each single-request prefill width
    that can widen, so lazy mixed prefills reuse the recipes instead of
    compiling on first hit.  Refuses inside capture; fails closed on ALL
    ranks when free HBM is below the one-chunk budget plus reserve.  Writes
    only the scratch top-k buffer, never the KV/kpool cache.
    """
    if not BOOT_WARM or PREFILL_TILE_CEILING <= QUERY_TILE:
        return 0
    if mixed_graph_capture_active():
        raise R590ContractError(f"{BOOT_WARM_ENV} warm must run after mixed graph capture, never inside it; "
                                "call it only from glm53_mixed_graph.warmup")
    ops = _boot_warm_ops(model)
    free, required = _admit_boot_warm(ops, free_hbm or _device_free_hbm)
    rows_totals = _boot_warm_rows()
    warmed = sum(_boot_warm_op(op, rows_totals) for op in ops)
    LOG.warning("R590 adaptive boot warm compiled %d widened indexer shapes rows=%s ops=%d prototypes=%d "
                "free_MiB=%.1f required_MiB=%.1f", warmed, list(rows_totals), len(ops),
                len(_TILE_PROTOTYPES), free / MIB, required / MIB)
    return warmed


def mla_bytes_per_query_row(width: int, heads: int, latent_dim: int, packed_row_bytes: int) -> int:
    """Conservative selected-row MLA workspace per query row.

    Packed rows are charged three times (owned stripe, gathered, reordered),
    dequantized latent twice (unpack staging + BF16 output), and FP32
    logits three times (scores, masked scores, probabilities).
    """
    return (3 * packed_row_bytes * width + 2 * 2 * latent_dim * width
            + 3 * 4 * heads * width)


def mla_tile_choices(base_rows: int, per_row_bytes: int) -> tuple[int, ...]:
    """Widened MLA fetch tiles allowed by the ceiling and the byte budget."""
    if MLA_TILE_BUDGET_BYTES is None:
        return ()
    return tuple(rows for rows in ADAPTIVE_TILE_ROWS
                 if base_rows < rows <= PREFILL_TILE_CEILING
                 and rows * per_row_bytes <= MLA_TILE_BUDGET_BYTES)


def mla_prefill_tile_rows(base_rows: int, query_rows: int, per_row_bytes: int) -> int:
    """Largest MLA fetch tile within HPU_GLM53_MLA_TILE_BUDGET_MIB.

    Each query row's softmax is independent, so the row partition is exact.
    ``base_rows`` (HPU_SPARSE_MLA_FETCH_TILE) is returned when the ceiling is
    unset, the budget is absent, or no larger tile divides ``query_rows``.
    ``per_row_bytes`` comes from :func:`mla_bytes_per_query_row`.
    """
    return next((rows for rows in mla_tile_choices(base_rows, per_row_bytes)
                 if query_rows % rows == 0), base_rows)


def indexer_forward(op, q_values, k, weights, *, gate_score, compress_ape,
                    positions, metadata_all, tile_rows=QUERY_TILE):
    """Tile before repeat/index_select, scorer clone, and table int64 casts.

    Preserve absolute output slices and the untiled call's max-length hint.
    Size context chunks for the tile count; canonical_topk makes equal-score
    choices independent of partitions. No warmup cap is introduced/increased.
    """
    if type(tile_rows) is not int or not 1 <= tile_rows <= QUERY_TILE:
        raise R590ContractError("R590 query tile must be an integer in [1,128]")
    if getattr(op, "tp_sharded_kpool", False):
        raise R590ContractError("R590 tiled indexer requires replicated kpool; sharded compaction is unproved")
    n_buffer = min(q_values.shape[0], op.topk_indices_buffer.shape[0])
    width = op.topk_tokens + op.pool_size - 1
    if n_buffer <= 0:
        return torch.empty((0, width), dtype=torch.int32, device=q_values.device)
    positions = positions.reshape(-1)
    if min(positions.numel(), weights.shape[0], k.shape[0], gate_score.shape[0]) < n_buffer:
        raise R590ContractError("R590 sparse inputs do not cover the token buffer")
    segments, token_mask = [], None
    if not isinstance(metadata_all, dict):
        metadata = metadata_all
        names = ("glm53_index_slot_mapping", "glm53_tail_slot_mapping",
                 "glm53_index_block_table", "glm53_token_mask")
        compressed, tail, table, token_mask = (getattr(metadata, name, None) for name in names)
        if any(value is None for value in (compressed, tail, table, token_mask)):
            raise R590ContractError("R590 sparse metadata is incomplete")
        requests = table.shape[0]
        if requests <= 0 or n_buffer % requests:
            raise R590ContractError("R590 sparse token bucket is not rectangular")
        if token_mask.numel() < n_buffer:
            raise R590ContractError("R590 sparse token mask is too short")
        source_block_size = int(getattr(op.k_cache, "hpu_kernel_block_size", op.source_block_size))
        bucket_length = table.shape[1] * source_block_size
        pool_cap = op.warmup_max_kpools if os.getenv("HPU_GLM53_CAP_LEGACY_WARMUP", "0") == "1" else None
        pool_cap = _static_long_prefill_pool_cap(table, n_buffer, source_block_size, op.pool_size, pool_cap)
        host_length, pool_cap = _true_host_large_prefill_bound(
            metadata, n_buffer, op.pool_size, bucket_length, pool_cap)
        if key_capture_active():  # a key's graph scans its whole width: no warmup cap, no host bound of one request
            host_length, pool_cap = bucket_length, bucket_length // op.pool_size
        segments.append(_owned_segment(0, n_buffer, table,
            repeat=n_buffer // requests,
            max_sequence_length=host_length, max_pool_cap=pool_cap))
        n_actual = n_buffer
        compressed, tail = compressed[:n_actual], tail[:n_actual]
        token_mask = token_mask[:n_buffer].reshape(-1).clone()
    else:
        metadata = metadata_all[op.k_cache.prefix]
        tail_metadata = metadata_all[op.tail_cache.prefix]
        n_actual = int(metadata.num_decode_tokens + metadata.num_prefill_tokens)
        if not 0 <= n_actual <= n_buffer:
            raise R590ContractError("R590 actual sparse token count exceeds buffer")
        host_max = getattr(metadata, "max_seq_len", None)
        if host_max is not None and type(host_max) is not int:
            raise R590ContractError("R590 max_seq_len must be a host integer")
        compressed, tail = metadata.slot_mapping, tail_metadata.slot_mapping
        count = int(metadata.num_decode_tokens)
        if count:
            decode = metadata.decode
            if decode is None:
                raise R590ContractError("R590 decode tokens lack metadata")
            table = decode.block_table
            cumulative = None
            if table.shape[0] != count:
                lens = decode.decode_lens.reshape(-1)
                if lens.numel() != table.shape[0]:
                    raise R590ContractError("R590 decode lengths differ from request rows")
                cumulative = lens.long().cumsum(0)
            segments.append(_owned_segment(0, count, table,
                cumulative_lens=cumulative, max_sequence_length=host_max))
        if metadata.num_prefill_tokens:
            if metadata.prefill is None:
                raise R590ContractError("R590 prefill tokens lack metadata")
            for chunk in metadata.prefill.chunks:
                start, end = int(chunk.token_start), int(chunk.token_end)
                if end <= start:
                    continue
                row_map = getattr(chunk, "query_to_seq", None)
                if row_map is None or row_map.numel() != end - start:
                    raise R590ContractError("R590 prefill chunk needs one query_to_seq entry per row")
                segments.append(_owned_segment(start, end, chunk.block_table,
                    row_map=row_map, max_sequence_length=host_max))
    segments.sort(key=lambda segment: segment.start)
    cursor = 0
    for segment in segments:
        if segment.start < cursor or segment.start < 0 or segment.end > n_actual:
            raise R590ContractError("R590 overlapping/out-of-range sparse segments")
        cursor = segment.end
    # Owned q/weight/position and per-request metadata survive every scorer
    # mark_step. Cache writes are submitted together with these owned inputs.
    q_values, weights, positions = (value[:n_buffer].clone() for value in (q_values, weights, positions))
    op.topk_indices_buffer[:n_buffer].fill_(-1)
    if n_actual:
        op._update_pooled_cache(k[:n_actual], gate_score[:n_actual], positions[:n_actual],
                                compressed, tail, compress_ape)
    r589._boundary("r590.indexer.inputs", q_values, rows=n_buffer, tile_rows=tile_rows)
    inputs = TileInputs(q_values, weights, positions, token_mask, width)
    if segments:
        warm_adaptive_tiles(op, inputs, segments[0].block_table)
        record_tile_prototype(inputs, segments[0])
    relax_tile_drains = r589.r606_relax_tile_drains()
    retained_tile = None
    pieces, cursor = [], 0
    for segment in segments:
        if segment.start > cursor:
            gap = torch.full((segment.start - cursor, width), -1, dtype=torch.int32, device=q_values.device)
            pieces.append(gap)
        original_count = segment.end - segment.start
        max_length = segment.max_sequence_length
        if max_length is None:
            max_length = int((positions[segment.start:segment.end].long() + 1).max().item())
        segment, segment_rows = _tile_plan(op, segment, max_length, tile_rows, q_values)
        for start in range(segment.start, segment.end, segment_rows):
            end = min(start + segment_rows, segment.end)
            physical, query_table, lengths = _score_tile(
                op, segment, (start, end, segment_rows), inputs, max_length)
            r589._boundary("r590.indexer.tile", q_values,
                           drain=not relax_tile_drains, start=start, end=end,
                           original_rows=original_count, table_shape=list(query_table.shape))
            pieces.append(physical)
            if relax_tile_drains:
                previous_tile = retained_tile
                retained_tile = (query_table, lengths)
                del query_table, lengths, physical
                if previous_tile is not None:
                    del previous_tile
            else:
                del query_table, lengths, physical
        cursor = segment.end
    if cursor < n_buffer:
        pieces.append(torch.full((n_buffer - cursor, width), -1, dtype=torch.int32, device=q_values.device))
    if relax_tile_drains and retained_tile is not None:
        r589._boundary("r606.indexer.return", q_values, drain=True, rows=n_buffer)
        retained_tile = None
    op._drain_first_dynamic_pool()
    return torch.cat(pieces, dim=0) if pieces else torch.full(
        (n_buffer, width), -1, dtype=torch.int32, device=q_values.device)
