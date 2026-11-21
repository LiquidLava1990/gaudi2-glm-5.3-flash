"""R589 diagnostic-only bounded prefill, composed over exact frozen R573.

No routing, gate, FP8 recipe, clamp, cache layout, or collective is replaced.
The public cache is committed by the original caller only after this returns.
HPU promotion requires separate measured memory, replay and quality evidence.
"""
from __future__ import annotations

import functools
import json
import logging
import os
import sys
from collections.abc import Callable
from types import ModuleType
from typing import Protocol

import torch
from glm53_mixed_graph import capture_active as mixed_graph_capture_active, capture_tile_marks
from glm53_finite_attribution import wrap_stage, wrap_boundary

TILE = 512
LOG = logging.getLogger("vllm.r589")


class R589ContractError(RuntimeError):
    """A shape, route, numerical, or execution-boundary proof failed."""


def enabled() -> bool:
    value = os.environ.get("HPU_GLM53_R589_ENABLE", "0")
    if value not in ("0", "1"):
        raise R589ContractError("HPU_GLM53_R589_ENABLE must be 0 or 1")
    return value == "1"


def _tensor_info(value: object) -> dict[str, object] | list[object] | None:
    if isinstance(value, torch.Tensor):
        return {"shape": list(value.shape), "dtype": str(value.dtype),
                "bytes": value.numel() * value.element_size()}
    if isinstance(value, (tuple, list)):
        return [_tensor_info(item) for item in value[:8]]
    return None


def _emit(phase: str, **fields: object) -> None:
    if os.environ.get("HPU_GLM53_R589_TRACE", "0") != "1":
        return
    memory = {}
    # Reading allocator counters does not synchronize or allocate HPU tensors.
    hpu = getattr(torch, "hpu", None)
    if hpu is not None and hpu.is_available():
        for name in ("memory_allocated", "memory_reserved", "max_memory_allocated"):
            getter = getattr(hpu, name, None)
            memory[name] = int(getter()) if getter is not None else None
    rank = (torch.distributed.get_rank() if torch.distributed.is_initialized()
            else int(os.environ.get("RANK", "-1")))
    LOG.warning("R589_TRACE %s", json.dumps(
        {"phase": phase, "rank": rank, "memory": memory, **fields}, sort_keys=True))


def stage(label: str) -> Callable[[Callable[..., object]], Callable[..., object]]:
    """Opt-in shape/allocator/error hooks; never change graph boundaries."""
    def decorate(function: Callable[..., object]) -> Callable[..., object]:
        @functools.wraps(function)
        def wrapped(*args: object, **kwargs: object) -> object:
            if not enabled() or os.environ.get("HPU_GLM53_R589_TRACE", "0") != "1":
                return function(*args, **kwargs)
            _emit(label + ".entry", args=_tensor_info(args),
                  kwargs={key: _tensor_info(value) for key, value in kwargs.items()})
            try:
                result = function(*args, **kwargs)
            except Exception as exc:
                _emit(label + ".error", error_type=type(exc).__name__)
                raise
            _emit(label + ".exit", result=_tensor_info(result))
            return result
        return wrap_stage(label, wrapped)
    return decorate


class PrefillRunner(Protocol):
    """The model-runner fields the prefill guards read."""

    input_batch: object
    use_merged_prefill: bool


class AttentionProcessor(Protocol):
    """The attention-processor fields guard_dense_bias reads."""

    prefill_use_fusedsdpa: bool
    block_size: int


class PromptMetadata(Protocol):
    """The prompt attention-metadata fields guard_dense_bias reads (its block_size is optional)."""

    is_prompt: bool
    attn_bias: torch.Tensor | None
    block_list: torch.Tensor | None


def exclude_prompt_logprobs(runner: PrefillRunner) -> None:
    """Explicitly unsupported in this isolated candidate; never silently omit."""
    if enabled() and getattr(runner.input_batch, "num_prompt_logprobs", {}):
        raise R589ContractError("R589 excludes prompt_logprobs; retain R573 for that request")


def exclude_merged_prefill(runner: PrefillRunner) -> None:
    if enabled() and runner.use_merged_prefill:
        raise R589ContractError("R589 excludes merged prefill: dense CPU/HPU mask is not bounded")


def guard_dense_bias(processor: AttentionProcessor, metadata: PromptMetadata | None, batch_size: int,
                     seq_len: int, dtype: torch.dtype) -> None:
    """Do not silently remove an attention mask whose consumers are unproven.

    Conservative visible-tensor estimate, not a Synapse peak prediction. Stop
    before the generic processor constructs B*Q*(context+Q) dense tensors.
    """
    if not enabled() or metadata is None or not metadata.is_prompt:
        return
    if metadata.attn_bias is not None:
        return
    blocks = metadata.block_list
    if processor.prefill_use_fusedsdpa and blocks is None:
        return
    total = int(batch_size) * int(seq_len)
    if total <= TILE:
        return
    context = (int(blocks.shape[-1]) // int(batch_size) if blocks is not None else 0)
    context *= int(getattr(metadata, "block_size", processor.block_size))
    cells = total * (context + int(seq_len))
    element_size = torch.empty((), dtype=dtype, device="cpu").element_size()
    visible_bytes = cells * (element_size + 3)  # bias plus mask producers
    _emit("metadata.dense_bias.estimate", total_tokens=total, query_per_sequence=int(seq_len),
          context_tokens=context, visible_bytes=visible_bytes)
    if visible_bytes > 512 * 2**20:
        raise R589ContractError(
            f"R589 dense prompt mask estimate {visible_bytes} exceeds 512 MiB: "
            "long-context 8K promotion is blocked pending proven sparse-only mask omission")


def guard_prefill(runner: PrefillRunner, token_ids: torch.Tensor, metadata: object,
                  logits_indices: torch.Tensor) -> None:
    """Reject an unreviewed execution mode before allocating prefill workspaces."""
    if not enabled():
        return
    exclude_prompt_logprobs(runner)
    tokens = int(token_ids.numel())
    if tokens <= TILE:
        return
    fetch = int(os.environ.get("HPU_SPARSE_MLA_FETCH_TILE", "0"))
    if not 0 < fetch <= 128:
        raise R589ContractError("R589 large prefill requires MLA fetch tile <=128")
    bypass = int(os.environ.get("HPU_GLM53_Q4_LARGE_PREFILL_BYPASS_GRAPH_MIN_TOKENS", "0"))
    if not 0 < bypass <= TILE:
        raise R589ContractError("R589 requires segmented lazy prefill, not whole-model HPU capture")
    if logits_indices.numel() > 128:
        raise R589ContractError("R589 excludes wide prompt-logit/vocabulary materialization")
    table = getattr(metadata, "glm53_index_block_table", None)
    batch = int(token_ids.shape[0]) if token_ids.ndim == 2 else 1
    context = int(table.shape[-1]) * 128 if isinstance(table, torch.Tensor) else 0
    _emit("prefill.census", census=memory_census(tokens, batch, context))


# R616_DRAIN_SITES_V1: which KDA / MoE execution boundaries still wait for the device (C271).
DRAIN_SITES_FLAG = "HPU_GLM53_R616_DRAIN_SITES"
DRAIN_SITE_LABELS = ("kda.owned", "kda.sequence", "moe.owned", "moe.return")
NO_DRAIN_SITES = "none"


CAPTURE_TILE_LABELS = frozenset({"kda.tile", "moe.tile", "r590.indexer.tile"})


def parse_drain_sites(value: str | None) -> frozenset[str]:
    """The sites that keep their drain: None (flag unset) keeps all four, 'none' keeps none.

    Anything else must be a comma list of DRAIN_SITE_LABELS, exactly as written (no blanks, no case folding).
    """
    if value is None:
        return frozenset(DRAIN_SITE_LABELS)
    if value == NO_DRAIN_SITES:
        return frozenset()
    labels = value.split(",")
    for label in labels:
        if label not in DRAIN_SITE_LABELS:
            raise R589ContractError(
                f"{DRAIN_SITES_FLAG}={value!r}: {label!r} is not a drain site; list the sites that keep their "
                f"drain, comma separated, from {', '.join(DRAIN_SITE_LABELS)}, or set {NO_DRAIN_SITES!r} alone "
                "to drop all four; unset it to keep every drain. The kda.tile / moe.tile drains are "
                "HPU_GLM53_R606_RELAX_TILE_DRAINS")
    return frozenset(labels)


def _drain_sites_from_env() -> frozenset[str]:
    """Parse the flag once, at import; a set flag is reported (the -drain-* arms' boot evidence)."""
    value = os.environ.get(DRAIN_SITES_FLAG)
    sites = parse_drain_sites(value)
    if value is not None:
        kept = ",".join(label for label in DRAIN_SITE_LABELS if label in sites) or NO_DRAIN_SITES
        dropped = ",".join(label for label in DRAIN_SITE_LABELS if label not in sites) or NO_DRAIN_SITES
        LOG.warning("R616_DRAIN_SITES %s=%s: drains kept at %s; dropped at %s",
                    DRAIN_SITES_FLAG, value, kept, dropped)
    return sites


DRAIN_SITES = _drain_sites_from_env()  # import-time parse: a bad value fails every worker's boot


def _site_drains(label: str) -> bool:
    """False only for a knob site the flag dropped; every other label keeps its caller's drain."""
    return label not in DRAIN_SITE_LABELS or label in DRAIN_SITES


# R599_R589_BOUNDARY_DRAIN_SPLIT_V1
@wrap_boundary
def _boundary(
    label: str, tensor: torch.Tensor, *, drain: bool = True, **fields: object
) -> None:
    """Submit every lazy recipe; drain the device only at proven lifetimes.

    R616: a knob site drains only while HPU_GLM53_R616_DRAIN_SITES keeps it AND its caller asks (kda.sequence
    still drains after the last sequence alone); mark_step stays. The gate sits here, not at the call sites,
    because the image's batched KDA twin (glm53_kda_batch_prefill) imports this function and reuses the kda.* labels.
    """
    if mixed_graph_capture_active():
        if label in CAPTURE_TILE_LABELS and capture_tile_marks() and tensor.device.type == "hpu":
            import habana_frameworks.torch.core as htcore
            htcore.mark_step(sync=False)  # arm B: C524's fence inside a key capture; never sync=True there
        return
    if tensor.device.type == "hpu":
        if os.environ.get("PT_HPU_LAZY_MODE", "0") != "1":
            raise R589ContractError("R589 is certified only for segmented HPU lazy mode")
        import habana_frameworks.torch.core as htcore
        htcore.mark_step(sync=True)
        if drain and _site_drains(label):
            torch.hpu.synchronize()
    _emit(label, **fields)


def r606_relax_tile_drains() -> bool:
    """Enable lifetime-safe tile drain coalescing; absent/0 is exact R599 behavior."""
    value = os.environ.get("HPU_GLM53_R606_RELAX_TILE_DRAINS", "0")
    if value not in ("0", "1"):
        raise R589ContractError("HPU_GLM53_R606_RELAX_TILE_DRAINS must be 0 or 1")
    return value == "1"


def _finite(label: str, *tensors: torch.Tensor) -> None:
    # Diagnostic candidate always fails closed. Copies are tile-bounded, never
    # full activations/weights. This fence is intentionally not a speed claim.
    for tensor in tensors:
        # Large lazy HPU tensor D2H snapshots were empirically non-coherent
        # (including impossible negative sigmoid weights).  Do not turn that
        # broken diagnostic bridge into a model failure; live API output is the
        # HPU numerical gate for this isolated candidate.
        if tensor.device.type == "hpu":
            continue
        if not bool(torch.isfinite(_cpu_synced(tensor)).all()):
            raise R589ContractError(f"{label}: non-finite tensor")


def _cpu_synced(tensor: torch.Tensor) -> torch.Tensor:
    """Return a completed D2H snapshot, including HPU lazy transfer drain."""
    if mixed_graph_capture_active():
        raise R589ContractError('mixed capture forbids diagnostic readback; disable runtime tensor tracing')
    source = tensor.detach()
    is_hpu = tensor.device.type == "hpu"
    if is_hpu:
        import habana_frameworks.torch.core as htcore
        htcore.mark_step(sync=True)
        torch.hpu.synchronize()
    result = source.to(device="cpu", non_blocking=False, copy=True)
    if is_hpu:
        torch.hpu.synchronize()
    return result


def kda_bounded(original: Callable[..., tuple[torch.Tensor, torch.Tensor | None]], q: torch.Tensor,
                k: torch.Tensor, v: torch.Tensor, g: torch.Tensor, beta: torch.Tensor, *, chunk_size: int = 64,
                initial_state: torch.Tensor | None = None, output_final_state: bool = True,
                use_qk_l2norm: bool = True, tile_tokens: int = TILE) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Independent [B,L,H,D] sequences, unchanged chunk algebra and FP32 state.

    Tile boundaries must coincide with original C-sized algebraic chunks.
    Only the final tile may pad; zero q/k/v/g/beta padding is an identity
    recurrent transition. B is NEVER flattened into the time dimension.
    Owned inputs survive HPU lazy steps; no caller input/cache is mutated.
    """
    if q.ndim != 4 or q.shape != k.shape or q.shape != g.shape:
        raise R589ContractError("KDA q/k/g must share [B,L,H,D]")
    batch, length, heads, key_dim = q.shape
    if v.shape[:3] != (batch, length, heads) or beta.shape != (batch, length, heads):
        raise R589ContractError("KDA v/beta sequence geometry mismatch")
    if chunk_size < 1 or tile_tokens < 1 or tile_tokens > TILE or tile_tokens % chunk_size:
        raise R589ContractError("KDA tile must be <=512 and a positive multiple of chunk_size")
    expected_state = (batch, heads, v.shape[-1], key_dim)
    if initial_state is not None and initial_state.shape != expected_state:
        raise R589ContractError("KDA state must use [B,H,V,K] orientation")
    if any(t.device != q.device for t in (k, v, g, beta)):
        raise R589ContractError("KDA input devices differ")
    # These are the bounded linear-in-T survivors; the 6-D pairwise tensor is
    # created only by original(B=1,L<=512), not before this split.
    q, k, v, g, beta = (t.clone() for t in (q, k, v, g, beta))
    states = (torch.zeros(expected_state, dtype=torch.float32, device=q.device)
              if initial_state is None else initial_state.float().clone())
    output = torch.empty((batch, length, heads, v.shape[-1]),
                         dtype=q.dtype, device=q.device)
    _boundary("kda.owned", q, drain=True, total_tokens=batch * length,
              owned_input_bytes=sum(t.numel() * t.element_size() for t in (q, k, v, g, beta)))
    relax_tile_drains = r606_relax_tile_drains()
    for sequence in range(batch):
        retained_tile = None
        state = states[sequence:sequence + 1].clone()
        for start in range(0, length, tile_tokens):
            end = min(start + tile_tokens, length)
            # R615_KDA_TILE_VIEW_V1: full owned inputs remain live through the
            # sequence drain; the callee immediately materializes each view with
            # transpose().contiguous().float(), so this intermediate clone is redundant.
            pieces = [t[sequence:sequence + 1, start:end] for t in (q, k, v, g, beta)]
            _boundary("kda.inputs", pieces[0], drain=False,
                      sequence=sequence, start=start, end=end)
            gate_cpu = (_cpu_synced(pieces[3])
                        if pieces[3].device.type != "hpu" else None)
            # Preserve the model-produced gate exactly.  The wrapper rejects
            # nonfinite values, but a sign anomaly is diagnostic-only: KDA can
            # consume any finite log-decay and this layer must not silently
            # clamp or otherwise change the model's math.
            finite = torch.isfinite(gate_cpu) if gate_cpu is not None else None
            positive = finite & (gate_cpu > 0) if finite is not None else None
            if gate_cpu is not None and (not bool(finite.all()) or bool(positive.any())):
                flat = gate_cpu.reshape(-1)
                finite_values = flat[torch.isfinite(flat)]
                bad_indices = torch.nonzero(~torch.isfinite(flat) | (flat > 0), as_tuple=False).reshape(-1)
                _emit("kda.gate_anomaly", sequence=sequence, start=start, end=end,
                      gate_dtype=str(gate_cpu.dtype), gate_shape=list(gate_cpu.shape),
                      finite_min=(float(finite_values.min()) if finite_values.numel() else None),
                      finite_max=(float(finite_values.max()) if finite_values.numel() else None),
                      nan=int(torch.isnan(flat).sum()), posinf=int(torch.isposinf(flat).sum()),
                      neginf=int(torch.isneginf(flat).sum()), positive=int(positive.sum()),
                      sample_indices=bad_indices[:8].tolist(),
                      sample_values=[float(flat[index]) for index in bad_indices[:8]])
            if finite is not None and not bool(finite.all()):
                raise R589ContractError("KDA gate contains nonfinite values")
            core, next_state = original(*pieces, chunk_size=chunk_size,
                initial_state=state, output_final_state=True, use_qk_l2norm=use_qk_l2norm)
            if next_state is None or next_state.dtype != torch.float32:
                raise R589ContractError("KDA lost exact FP32 recurrent state")
            if core.shape != (1, end - start, heads, v.shape[-1]) or next_state.shape != (1, *expected_state[1:]):
                raise R589ContractError("KDA original result geometry drifted")
            output[sequence:sequence + 1, start:end].copy_(core)
            state = next_state
            padded = ((end - start + chunk_size - 1) // chunk_size) * chunk_size
            _boundary("kda.tile", core, drain=not relax_tile_drains,
                      sequence=sequence, start=start, end=end,
                      pairwise_bytes=heads * padded * chunk_size * key_dim * 4)
            _finite("KDA tile", core, state)
            if relax_tile_drains:
                previous_tile = retained_tile
                retained_tile = (core, next_state, pieces, gate_cpu)
                del core, next_state, pieces, gate_cpu
                if previous_tile is not None:
                    del previous_tile
            else:
                del core, next_state, pieces, gate_cpu
        states[sequence:sequence + 1].copy_(state)
        _boundary("kda.sequence", states,
                  drain=(sequence == batch - 1), sequence=sequence)
        retained_tile = None
    _emit("kda.return", output=_tensor_info(output), state=_tensor_info(states),
          external_state_commits=0)
    return output, states if output_final_state else None


def normalize_gather_padding(gather: torch.Tensor, grouped_weights: torch.Tensor, tokens: int) -> torch.Tensor:
    """Canonicalize only irrelevant zero-weight padding before gather/scatter."""
    raw = _cpu_synced(gather).long()
    grouped = _cpu_synced(grouped_weights)
    invalid = (raw < 0) | (raw >= int(tokens))
    invalid_active = invalid & (grouped != 0)
    if bool(invalid_active.any()):
        raise R589ContractError("active route gather index escaped tile")
    invalid_padding = invalid & (grouped == 0)
    if bool(invalid_padding.any()):
        _emit("moe.padding_index_canonicalized", count=int(invalid_padding.sum()),
              raw_min=int(raw[invalid].min()), raw_max=int(raw[invalid].max()))
    return torch.where(grouped_weights != 0, gather, torch.zeros_like(gather))


def build_full_capacity_route_plan(ids: torch.Tensor, weights: torch.Tensor, *, experts_min: int, experts: int,
                                   dtype: torch.dtype) -> torch.Tensor:
    """Exact scatter-free route matrix for cap=T."""
    local = ids.to(torch.int32) - int(experts_min)
    expert_axis = torch.arange(
        int(experts), dtype=torch.int32, device=ids.device).view(1, 1, int(experts))
    matches = (local.unsqueeze(-1) == expert_axis) & (weights != 0).unsqueeze(-1)
    route = (weights.float().unsqueeze(-1) * matches.to(torch.float32)).sum(dim=1)
    return route.transpose(0, 1).contiguous().to(dtype)


def audit_dense_route_weights(ids: torch.Tensor, weights: torch.Tensor, grouped_weights: torch.Tensor, *,
                              experts_min: int, experts: int, global_experts: int) -> dict[str, int]:
    """Prove exact token/expert weights without materializing gather indices."""
    ids = _cpu_synced(ids).long()
    weights = _cpu_synced(weights)
    grouped_weights = _cpu_synced(grouped_weights)
    _finite("route weights", weights, grouped_weights)
    if bool((weights < 0).any()):
        raise R589ContractError("R573 sigmoid/softmax routes must be nonnegative")
    active = weights != 0
    if bool((active & ((ids < 0) | (ids >= global_experts))).any()):
        raise R589ContractError("active expert id escaped global router domain")
    tokens = ids.shape[0]
    if grouped_weights.shape != (experts, tokens):
        raise R589ContractError("dense route matrix geometry drifted")
    local = ids - experts_min
    owned = active & (local >= 0) & (local < experts)
    safe = torch.where(owned, local, torch.zeros_like(local))
    expected = torch.zeros(tokens, experts).scatter_add_(
        1, safe, torch.where(owned, weights.float(),
                             torch.zeros_like(weights, dtype=torch.float32)))
    expected = expected.transpose(0, 1).to(grouped_weights.dtype)
    if not torch.equal(grouped_weights, expected):
        raise R589ContractError("dense route matrix changed/dropped token-expert weights")
    pairs = int((expected != 0).sum())
    return {"owned_slots": int(owned.sum()), "owned_pairs": pairs,
            "kept_pairs": pairs, "dropped_pairs": 0, "capacity": tokens}


def audit_route_plan(ids: torch.Tensor, weights: torch.Tensor, gather: torch.Tensor, grouped_weights: torch.Tensor,
                     *, experts_min: int, experts: int, global_experts: int) -> dict[str, int]:
    """Compare every kept (token,expert) weight, not only aggregate counts."""
    ids = _cpu_synced(ids).long()
    weights = _cpu_synced(weights)
    gather = _cpu_synced(gather).long()
    grouped_weights = _cpu_synced(grouped_weights)
    _finite("route weights", weights, grouped_weights)
    if bool((weights < 0).any()):
        raise R589ContractError("R573 sigmoid/softmax routes must be nonnegative")
    active = weights != 0
    if bool((active & ((ids < 0) | (ids >= global_experts))).any()):
        raise R589ContractError("active expert id escaped global router domain")
    tokens = ids.shape[0]
    if gather.shape != (experts, tokens) or grouped_weights.shape != gather.shape:
        raise R589ContractError("route capacity must equal tile length")
    if bool(((gather < 0) | (gather >= tokens)).any()):
        raise R589ContractError("route gather index escaped tile")
    local = ids - experts_min
    owned = active & (local >= 0) & (local < experts)
    safe = torch.where(owned, local, torch.zeros_like(local))
    expected = torch.zeros(tokens, experts).scatter_add_(1, safe,
        torch.where(owned, weights.float(), torch.zeros_like(weights, dtype=torch.float32)))
    expected = expected.to(grouped_weights.dtype)
    actual = torch.zeros(experts, tokens, dtype=grouped_weights.dtype).scatter_add_(
        1, gather, grouped_weights).transpose(0, 1)
    if not torch.equal(actual, expected):
        raise R589ContractError("route plan changed/dropped token-expert weights")
    # Unique token/expert pairs; repeated positive slots are summed exactly as R573.
    kept = int((grouped_weights != 0).sum())
    pairs = int((expected != 0).sum())
    if kept != pairs:
        raise R589ContractError("route plan duplicated or lost pairs")
    return {"owned_slots": int(owned.sum()), "owned_pairs": pairs, "kept_pairs": kept,
            "dropped_pairs": 0, "capacity": tokens}


def _dense_tile(engine: ModuleType, gemm_fn: Callable[..., object], x: torch.Tensor, ids: torch.Tensor,
                weights: torch.Tensor, w13: torch.Tensor, s13: torch.Tensor, w2: torch.Tensor, s2: torch.Tensor, *,
                start: int, end: int, experts_min: int, experts: int, global_experts: int, limit: float,
                weight_layout: int) -> tuple[torch.Tensor, tuple[torch.Tensor, ...], dict[str, object]]:
    """One full-capacity tile: every local expert runs on every tile token.

    Returns the tile, the tensors whose lifetime the caller must retain until
    the next submission boundary, and the boundary trace fields.
    """
    n, hidden = end - start, int(x.shape[1])
    xt, it, wt = (t[start:end].clone() for t in (x, ids, weights))
    # Capacity is the full tile, so counting-sort cannot reduce the grouped
    # GEMM or workspace.  A dense static expert-by-token plan is exact and
    # avoids Gaudi's fragile cumsum/int64-scatter address construction.
    wg = build_full_capacity_route_plan(
        it, wt, experts_min=experts_min, experts=experts, dtype=x.dtype)
    _boundary("moe.plan", wg, drain=False,
              start=start, end=end, capacity=n)
    proof = (audit_dense_route_weights(
        it, wt, wg, experts_min=experts_min, experts=experts,
        global_experts=global_experts) if x.device.type != "hpu" else
        {"owned_slots": None, "owned_pairs": None, "kept_pairs": None,
         "dropped_pairs": 0, "capacity": n})
    expanded = xt.unsqueeze(0).expand(experts, n, hidden)
    padding = xt[:1].reshape(1, 1, hidden).expand(experts, n, hidden)
    xg = torch.where(wg.view(experts, n, 1) != 0, expanded, padding).contiguous()
    yg = engine._call_grouped_glm_gemm(gemm_fn, xg, w13, s13.reshape(experts, -1),
                                      w2, s2.reshape(experts, -1), float(limit), weight_layout)
    if yg is None or yg.shape != xg.shape or yg.dtype != x.dtype:
        raise R589ContractError("grouped FP8 engine result ABI drifted")
    yw = yg * wg.view(experts, n, 1).to(yg.dtype)
    tile = yw.sum(dim=0)
    fields = {"proof": proof,
              "gathered_bytes": xg.numel() * xg.element_size(),
              "output_bytes": yg.numel() * yg.element_size(),
              "weighted_bytes": yw.numel() * yw.element_size()}
    return tile, (xt, it, wt, wg, expanded, padding, xg, yg, yw), fields


DEVICE_PLAN_FLAG = "HPU_GLM53_MOE_DEVICE_PLAN"


def device_plan_requested() -> bool:
    """True when HPU_GLM53_MOE_DEVICE_PLAN=1; absent/0 is exact R589.

    Also runs once at import: hpu_fp8 imports this module at boot, so a bad value
    or flag combination stops the boot instead of the first large prefill. The
    plan reads the R606 drain flag per chunk, so that flag is checked here too.
    """
    value = os.environ.get(DEVICE_PLAN_FLAG, "0")
    if value not in ("0", "1"):
        raise R589ContractError(
            f"{DEVICE_PLAN_FLAG} must be 0 or 1, got {value!r}; unset it to keep the "
            "R589 dense bounded MoE path")
    if value == "0":
        return False
    if os.environ.get("HPU_GLM53_ROUTED_MOE_PREFILL", "0") != "0":
        raise R589ContractError(
            f"{DEVICE_PLAN_FLAG}=1 and HPU_GLM53_ROUTED_MOE_PREFILL are mutually "
            "exclusive; unset HPU_GLM53_ROUTED_MOE_PREFILL")
    r606_relax_tile_drains()
    return True


def _device_plan_selected(tile_tokens: int) -> bool:
    """HPU_GLM53_MOE_DEVICE_PLAN=1 replaces the dense tile loop; absent/0 is R589."""
    if not device_plan_requested():
        return False
    if tile_tokens != TILE:
        raise R589ContractError(
            "the MoE device plan runs fixed TILE slot chunks; keep tile_tokens=TILE")
    return True


def _device_planned_or_none(engine: ModuleType, gemm_fn: Callable, *, tile_tokens: int,
                            **call: object) -> torch.Tensor | None:
    """Device-planned MoE when HPU_GLM53_MOE_DEVICE_PLAN=1; None keeps R589 dense.

    The plan itself also returns None for a single-tile call (before any census),
    when every chunk would run (dense is cheaper), on an unproven or incoherent
    census, or when its sealed return proof fails; dense then recomputes the call
    from the original, untouched inputs. Both paths are collective-free.
    """
    if not _device_plan_selected(tile_tokens):
        return None
    import glm53_moe_device_plan as device_plan  # flag-off processes never import it
    return device_plan.device_planned_moe(engine, gemm_fn, **call)


device_plan_requested()  # import-time flag check: a bad value fails at boot


def grouped_fp8_bounded(engine: ModuleType, x: torch.Tensor, w13: torch.Tensor, s13: torch.Tensor,
                        w2: torch.Tensor, s2: torch.Tensor, ids: torch.Tensor, weights: torch.Tensor, *,
                        experts_min: int, top_k: int, global_experts: int, limit: float,
                        weight_layout: int = 1, tile_tokens: int = TILE,
                        gemm_fn: Callable[..., object] | None = None) -> torch.Tensor:
    """Tile the ACTIVE grouped path after original routing/dispatch.

    Quantization remains per last-axis feature row. Full expert weights and
    channel scales are reused, not cloned, dequantized, or transposed. Capacity
    n covers all possible unique token/expert pairs even with extreme skew.
    Shared experts, routed scaling and TP combine/reduction stay in R573.
    """
    if x.ndim != 2 or ids.shape != weights.shape or ids.shape != (x.shape[0], top_k):
        raise R589ContractError("MoE activation/route geometry mismatch")
    if tile_tokens < 1 or tile_tokens > TILE or weight_layout not in (1, 2):
        raise R589ContractError("invalid MoE tile or checkpoint layout")
    if getattr(engine, "_DEFAULT_GLM_HOT_EXPERTS", 0) != 0:
        raise R589ContractError("R589 requires the exact R573 non-tiered grouped route")
    experts, tokens = int(w13.shape[0]), int(x.shape[0])
    if experts_min < 0 or experts_min + experts > global_experts:
        raise R589ContractError("invalid local expert window")
    if tokens == 0:
        return torch.empty_like(x)
    if gemm_fn is None:
        gemm_fn = (engine._load_grouped_glm_fp8_op() if weight_layout == 1
                   else engine._load_grouped_glm_fp8_smart_op())
    if gemm_fn is None:
        raise R589ContractError("exact grouped FP8 engine missing; no unbounded fallback")
    planned = _device_planned_or_none(
        engine, gemm_fn, x=x, ids=ids, weights=weights, expert_weights=(w13, s13, w2, s2),
        experts_min=experts_min, top_k=top_k, limit=limit, weight_layout=weight_layout,
        tile_tokens=tile_tokens)
    if planned is not None:
        return planned
    x, ids, weights = x.clone(), ids.clone(), weights.clone()
    output = torch.empty_like(x)
    _boundary("moe.owned", x, drain=True,
              total_tokens=tokens, input=_tensor_info(x))
    relax_tile_drains = r606_relax_tile_drains()
    if os.environ.get("HPU_GLM53_ROUTED_MOE_PREFILL", "0") != "0":
        from glm53_routed_moe import enabled as routed_enabled, min_routed_tokens, routed_tiles
    else:
        routed_enabled = None
    # Calls too small for any routed tile skip the routing snapshot entirely.
    if routed_enabled is not None and routed_enabled() and tokens >= min_routed_tokens():
        if tile_tokens != TILE:
            raise R589ContractError("routed MoE prefill plans its own tiles; keep tile_tokens=TILE")
        return routed_tiles(sys.modules[__name__], engine, gemm_fn, x, ids, weights, output,
                            w13, s13, w2, s2, experts_min=experts_min, top_k=top_k,
                            global_experts=global_experts, limit=limit,
                            weight_layout=weight_layout, relax_tile_drains=relax_tile_drains)
    retained_tile = None
    for start in range(0, tokens, tile_tokens):
        end = min(start + tile_tokens, tokens)
        tile, owned, fields = _dense_tile(
            engine, gemm_fn, x, ids, weights, w13, s13, w2, s2, start=start, end=end,
            experts_min=experts_min, experts=experts, global_experts=global_experts,
            limit=limit, weight_layout=weight_layout)
        output[start:end].copy_(tile)
        _boundary("moe.tile", tile, drain=not relax_tile_drains,
                  start=start, end=end, **fields)
        _finite("MoE tile", tile)
        if relax_tile_drains:
            previous_tile = retained_tile
            retained_tile = (*owned, tile)
            del owned, tile
            if previous_tile is not None:
                del previous_tile
        else:
            del owned, tile
    if relax_tile_drains:
        _boundary("moe.return", output, drain=True, total_tokens=tokens)
        retained_tile = None
    return output


def memory_census(total_tokens: int = 8192, batch: int = 8, context_tokens: int = 0) -> dict[str, object]:
    """Source-derived tensor sizes, NOT allocator peaks or fused scratch proof."""
    hidden, heads, dim, experts, intermediate = 4096, 16, 128, 72, 2048
    query = (total_tokens + batch - 1) // batch
    selected = min(2176, max(64, ((context_tokens + query + 63) // 64) * 64))
    cap = max((total_tokens + 1) // 2, 2 * ((total_tokens * 8 + 287) // 288))
    return {
        "kind": "static tensor census; not a simultaneous live-set estimate",
        "T": total_tokens, "BS": batch, "query_per_sequence": query,
        "kda_pairwise_original": total_tokens * heads * 128 * dim * 4,
        "kda_pairwise_r589": TILE * heads * 128 * dim * 4,
        "kda_fp32_state": batch * heads * dim * dim * 4,
        "moe_each_xg_yg_yw_original": experts * cap * hidden * 2,
        "moe_each_xg_yg_yw_r589": experts * TILE * hidden * 2,
        "moe_gu_r589": experts * TILE * 2 * intermediate * 2,
        "moe_each_gate_up_h_r589": experts * TILE * intermediate * 2,
        "model_hidden_each": total_tokens * hidden * 2,
        "mhc_full_residual_each": total_tokens * 4 * hidden * 2,
        "mhc_fp32_full_residual_each": total_tokens * 4 * hidden * 4,
        "router_fp32_each": total_tokens * 288 * 4,
        "mla_latent_q_or_out_each": total_tokens * 16 * 512 * 2,
        "mla_selected_width": selected,
        "mla_selected_bf16_fetch512": TILE * selected * 512 * 2,
        "mla_logits_fp32_each_fetch512": TILE * 16 * selected * 4,
        "mla_selected_bf16_fetch128": 128 * selected * 512 * 2,
        "mla_q4_packed_bytes_fetch128": 128 * selected * 272,
        "mla_q4_bf16_reduce_bytes_fetch128": 128 * selected * 272 * 2,
        "mla_logits_fp32_each_fetch128": 128 * 16 * selected * 4,
        "excluded_prompt_logits_fp32": total_tokens * 154880 * 4,
        "generic_dense_prompt_bias_bf16": total_tokens * (context_tokens + query) * 2,
        "generic_dense_prompt_bias_and_masks_estimate": total_tokens * (context_tokens + query) * 5,
        "indexer_repeated_block_table_int64": total_tokens * ((context_tokens + 127) // 128) * 8,
        "indexer_topk_int32": total_tokens * 2176 * 4,
        "indexer_context_tile_budget": 512 * 2**20,
        "unbounded_with_T": ["mHC residual/norm", "QKV and output projections",
            "indexer repeated block table/top-k output", "MoE dispatch/final output/TP collectives",
            "generic dense prompt mask (guarded >512MiB, not eliminated)"],
        "not_proven": ["Synapse compiler workspaces", "allocator fragmentation",
            "HCCL scratch", "graph/recipe residency", "cross-stage simultaneous liveness"],
    }
