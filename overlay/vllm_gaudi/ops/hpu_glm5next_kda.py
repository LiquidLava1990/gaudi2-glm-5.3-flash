# SPDX-License-Identifier: Apache-2.0
"""HPU-friendly GLM-5.3 KDA reference and decode kernels.

The hot decode path is vectorized over requests and heads.  The chunked
prefill path intentionally mirrors the official Transformers fallback so it
can serve as a correctness oracle while HPU graph buckets and fused kernels
are tuned.  vLLM stores recurrent state as ``[slot, head, value_dim, key_dim]``;
the official reference uses the transpose of the last two dimensions.
"""

from __future__ import annotations

import logging
import os
import threading
from collections.abc import Callable
from typing import NamedTuple

import torch
import torch.nn.functional as F
from torch import nn

_LOG = logging.getLogger(__name__)

_COMPILED_KDA_SPEC_STEPS = {}
_COMPILED_KDA_SPEC_LADDERS = {}
_COMPILED_KDA_TXN_LADDERS = {}
_COMPILED_KDA_SPEC_CONVS = {}
_COMPILED_KDA_DECODE_STEPS = {}
_COMPILED_KDA_FUSED_DECODE_STEPS = {}
_ENABLE_COMPILED_KDA_DECODE = os.getenv(
    "HPU_GLM53_COMPILED_KDA_DECODE", "0"
).lower() in ("1", "true", "yes", "on")
_ENABLE_FUSED_KDA_DECODE = os.getenv(
    "HPU_GLM53_FUSED_KDA_DECODE", "0"
).lower() in ("1", "true", "yes", "on")
_KDA_MATMUL_MIN_BATCH = int(os.getenv(
    "HPU_GLM53_KDA_MATMUL_MIN_BATCH", "0"
))
_ENABLE_KDA_BLOCK_INVERSE_PREFILL = os.getenv(
    "HPU_GLM53_KDA_BLOCK_INVERSE_PREFILL", "0"
).lower() in ("1", "true", "yes", "on")

# KDA V2 (FORK-PORT-PLAN S6): per-verb opt-in to the fork's fast chunk-KDA
# parts.  Unset or empty keeps every call on the untouched V1 bodies.
# w-kdapf adds parallel_scan (the fork's VLLM_KDA_SCAN=parallel) and whole_call
# (no R589 512-token tiles; capture-legal inside prompt / mixed graphs), and exact (V1's math bitwise on
# R589's tile plan, with graphs instead of host-synced tiles; exact_chunks runs that plan one chunk per call).
_KDA_V2_FLAG = "HPU_GLM53_KDA_V2_FOR"
_KDA_V2_R3_VERBS = frozenset({"decay_dot", "batched_recurrence", "parallel_scan", "whole_call"})
_KDA_V2_VERBS = _KDA_V2_R3_VERBS | {"exact", "exact_chunks"}


def _kda_v2_verbs(value: str) -> frozenset[str]:
    """Parse the comma list in ``HPU_GLM53_KDA_V2_FOR``; refuse unknown verbs and unsafe combinations."""

    verbs = frozenset(part.strip() for part in value.split(",") if part.strip())
    unknown = verbs - _KDA_V2_VERBS
    if unknown:
        raise ValueError(
            f"{_KDA_V2_FLAG}={value!r} names unknown verbs {sorted(unknown)}; "
            f"use a comma list of {sorted(_KDA_V2_VERBS)}, or unset it for V1"
        )
    if "whole_call" in verbs and "decay_dot" not in verbs:
        raise ValueError(
            f"{_KDA_V2_FLAG}={value!r}: whole_call needs decay_dot (without it the pair decay is a "
            f"[tokens, 128, 128] fp32 tensor, 2 GiB per 2048-token call); add decay_dot or drop whole_call"
        )
    if "exact" in verbs and verbs & _KDA_V2_R3_VERBS:
        raise ValueError(
            f"{_KDA_V2_FLAG}={value!r}: exact is V1's math and cannot take r3 verbs "
            f"{sorted(verbs & _KDA_V2_R3_VERBS)}; use exact[,exact_chunks] alone"
        )
    if "exact_chunks" in verbs and "exact" not in verbs:
        raise ValueError(f"{_KDA_V2_FLAG}={value!r}: exact_chunks refines exact; use exact,exact_chunks")
    return verbs


_KDA_V2_FOR = _kda_v2_verbs(os.getenv(_KDA_V2_FLAG, ""))
# Recorded per thread by the V2 gate path: ``lower_bound`` is the forget-gate
# bound of this thread's last hpu_kda_gate call (absent/None = unbounded gate,
# or no gate seen yet).  The layer's gate and its chunk call share a thread.
_KDA_V2_GATE = threading.local()


def _l2norm(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    x32 = x.float()
    return x32 / torch.sqrt((x32 * x32).sum(dim=-1, keepdim=True) + eps)


def hpu_kda_safe_gate(
    raw_gate: torch.Tensor,
    a_log: torch.Tensor,
    gate_bias: torch.Tensor,
    lower_bound: float,
) -> torch.Tensor:
    """Exact GLM-5.3 bounded forget gate, computed in FP32.

    ``raw_gate`` is ``[batch, tokens, heads, key_dim]``.  ``gate_bias`` may be
    flat ``[heads * key_dim]`` and ``a_log`` may be any checkpoint-compatible
    shape containing one value per head.
    """

    heads, key_dim = raw_gate.shape[-2:]
    gate = raw_gate.float() + gate_bias.float().reshape(1, 1, heads, key_dim)
    decay_rate = torch.exp(a_log.float().reshape(1, 1, heads, 1))
    return float(lower_bound) * torch.sigmoid(decay_rate * gate)


def hpu_kda_unbounded_gate(
    raw_gate: torch.Tensor,
    a_log: torch.Tensor,
    gate_bias: torch.Tensor,
) -> torch.Tensor:
    """Official non-safe fallback, retained for checkpoint compatibility."""

    heads, key_dim = raw_gate.shape[-2:]
    gate = raw_gate.float() + gate_bias.float().reshape(1, 1, heads, key_dim)
    decay_rate = torch.exp(a_log.float().reshape(1, 1, heads, 1))
    softplus = torch.where(gate > 20.0, gate, torch.log1p(torch.exp(gate)))
    return -decay_rate * softplus


def hpu_kda_gate(
    raw_gate: torch.Tensor,
    a_log: torch.Tensor,
    gate_bias: torch.Tensor,
    lower_bound: float | None,
) -> torch.Tensor:
    if _KDA_V2_FOR:  # HPU_GLM53_KDA_V2_FOR; unset = the untouched V1 body below
        return _kda_v2_gate(raw_gate, a_log, gate_bias, lower_bound)
    if lower_bound is None:
        return hpu_kda_unbounded_gate(raw_gate, a_log, gate_bias)
    return hpu_kda_safe_gate(raw_gate, a_log, gate_bias, lower_bound)


def _kda_v2_gate(
    raw_gate: torch.Tensor,
    a_log: torch.Tensor,
    gate_bias: torch.Tensor,
    lower_bound: float | None,
) -> torch.Tensor:
    """V1 gate math, plus recording the bound that ``_kda_v2_chunk`` requires."""

    _KDA_V2_GATE.lower_bound = None if lower_bound is None else float(lower_bound)
    if lower_bound is None:
        return hpu_kda_unbounded_gate(raw_gate, a_log, gate_bias)
    return hpu_kda_safe_gate(raw_gate, a_log, gate_bias, lower_bound)


def hpu_recurrent_kda_decode(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    recurrent_state: torch.Tensor,
    state_indices: torch.Tensor,
    *,
    beta_is_raw: bool = False,
    use_qk_l2norm: bool = True,
    state_compute_dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """One-token-per-request KDA decode, vectorized across the entire batch.

    Args use ``[batch, heads, dim]`` except the state cache, which is
    ``[slots, heads, value_dim, key_dim]``.  Only selected active slots are
    gathered and written back; inactive capacity never enters the graph.
    """

    if q.ndim != 3 or q.shape != k.shape:
        raise ValueError(f"q/k must share [B,H,K], got {q.shape=} {k.shape=}")
    if v.shape[:2] != q.shape[:2] or g.shape != q.shape:
        raise ValueError("v must share B/H and g must match q")
    if state_indices.ndim != 1 or state_indices.numel() != q.shape[0]:
        raise ValueError("state_indices must contain one slot per request")

    slots = state_indices.long()
    state = recurrent_state.index_select(0, slots).to(state_compute_dtype)
    if (
        _ENABLE_COMPILED_KDA_DECODE
        and q.device.type == "hpu"
        and state_compute_dtype == torch.float32
    ):
        step = _get_compiled_kda_decode_step(
            (
                tuple(q.shape),
                tuple(k.shape),
                tuple(v.shape),
                tuple(g.shape),
                tuple(beta.shape),
                tuple(state.shape),
                str(q.dtype),
                str(k.dtype),
                str(v.dtype),
                str(g.dtype),
                str(beta.dtype),
                str(state.dtype),
            ),
            beta_is_raw=beta_is_raw,
            use_qk_l2norm=use_qk_l2norm,
        )
        output, state = step(q, k, v, g, beta, state)
        recurrent_state.index_copy_(0, slots, state.to(recurrent_state.dtype))
        return output

    q32 = _l2norm(q) if use_qk_l2norm else q.float()
    k32 = _l2norm(k) if use_qk_l2norm else k.float()
    q32 = q32 * (q.shape[-1] ** -0.5)
    v32 = v.to(state_compute_dtype)
    g32 = g.float()
    beta32 = beta.float().sigmoid() if beta_is_raw else beta.float()
    if beta32.ndim == 2:
        beta32 = beta32.unsqueeze(-1)

    key_compute = k32.to(state_compute_dtype)
    query_compute = q32.to(state_compute_dtype)
    state = state * torch.exp(g32).to(state_compute_dtype).unsqueeze(-2)
    use_matmul = (
        _KDA_MATMUL_MIN_BATCH > 0
        and q.shape[0] >= _KDA_MATMUL_MIN_BATCH
    )
    if use_matmul:
        projection = torch.matmul(state, key_compute.unsqueeze(-1)).squeeze(-1)
    else:
        projection = (state * key_compute.unsqueeze(-2)).sum(dim=-1)
    delta = (v32 - projection) * beta32.to(state_compute_dtype)
    state = state + delta.unsqueeze(-1) * key_compute.unsqueeze(-2)
    if use_matmul:
        output = torch.matmul(state, query_compute.unsqueeze(-1)).squeeze(-1)
    else:
        output = (state * query_compute.unsqueeze(-2)).sum(dim=-1)

    # index_copy_ keeps the fixed-size state pool compact and avoids a full
    # cache clone.  The caller may wrap this write in a Dynamo-disabled helper
    # on runtime versions that drop aliasing writes during HPU compilation.
    recurrent_state.index_copy_(0, slots, state.to(recurrent_state.dtype))
    return output.to(v.dtype)


def _hpu_kda_spec_step(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    state: torch.Tensor,
    beta_is_raw: bool,
    use_qk_l2norm: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """One pure KDA state transition for a fixed MTP request bucket."""

    q32 = _l2norm(q) if use_qk_l2norm else q.float()
    k32 = _l2norm(k) if use_qk_l2norm else k.float()
    q32 = q32 * (q.shape[-1] ** -0.5)
    beta32 = beta.float().sigmoid() if beta_is_raw else beta.float()
    if beta32.ndim == 2:
        beta32 = beta32.unsqueeze(-1)
    next_state = state.float() * torch.exp(g.float()).unsqueeze(-2)
    projection = (next_state * k32.unsqueeze(-2)).sum(-1)
    delta = (v.float() - projection) * beta32
    next_state = next_state + delta.unsqueeze(-1) * k32.unsqueeze(-2)
    output = (next_state * q32.unsqueeze(-2)).sum(-1)
    return output.to(v.dtype), next_state


def hpu_kda_apply_transition_factors(
    state: torch.Tensor,
    factors: torch.Tensor,
) -> torch.Tensor:
    """Apply one exact KDA transition stored as O(D), not O(D^2), state.

    ``factors`` concatenates the log decay, normalized key, and value delta
    for each head.  This is the transactional rollback representation used to
    avoid a second full recurrent matrix for one-token native MTP.
    """

    dim = state.shape[-1]
    if factors.shape[-1] != 3 * dim:
        raise ValueError("KDA transition factors must contain [g, k, delta]")
    gate, key, delta = factors.float().split(dim, dim=-1)
    return (
        state.float() * torch.exp(gate).unsqueeze(-2)
        + delta.unsqueeze(-1) * key.unsqueeze(-2)
    )


def _hpu_kda_spec_step_with_factors(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    state: torch.Tensor,
    beta_is_raw: bool,
    use_qk_l2norm: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """One KDA step plus its exact compact rollback transaction."""

    q32 = _l2norm(q) if use_qk_l2norm else q.float()
    k32 = _l2norm(k) if use_qk_l2norm else k.float()
    q32 = q32 * (q.shape[-1] ** -0.5)
    gate = g.float()
    beta32 = beta.float().sigmoid() if beta_is_raw else beta.float()
    if beta32.ndim == 2:
        beta32 = beta32.unsqueeze(-1)
    decayed = state.float() * torch.exp(gate).unsqueeze(-2)
    projection = (decayed * k32.unsqueeze(-2)).sum(-1)
    delta = (v.float() - projection) * beta32
    next_state = decayed + delta.unsqueeze(-1) * k32.unsqueeze(-2)
    output = (next_state * q32.unsqueeze(-2)).sum(-1)
    factors = torch.cat((gate, k32, delta), dim=-1)
    return output.to(v.dtype), next_state, factors


def _get_compiled_kda_decode_step(
    cache_key,
    *,
    beta_is_raw: bool,
    use_qk_l2norm: bool,
):
    """Return one pure recurrent-transition recipe per live decode bucket.

    State gathering and ``index_copy_`` remain outside this leaf.  This keeps
    persistent state aliasing explicit while collapsing the normalization,
    decay, projection, delta, outer-product update, and readout into one HPU
    recipe.  The causal convolution is deliberately not included: compiling
    that path changed GLM-5.3's layer-0 activation oracle on Gaudi2.
    """

    key = (tuple(cache_key), bool(beta_is_raw), bool(use_qk_l2norm))
    compiled = _COMPILED_KDA_DECODE_STEPS.get(key)
    if compiled is None:
        torch._dynamo.config.recompile_limit = max(
            int(torch._dynamo.config.recompile_limit), 256
        )
        if hasattr(torch._dynamo.config, "accumulated_recompile_limit"):
            torch._dynamo.config.accumulated_recompile_limit = max(
                int(torch._dynamo.config.accumulated_recompile_limit), 512
            )

        def owned_leaf(q, k, v, g, beta, state):
            return _hpu_kda_spec_step(
                q,
                k,
                v,
                g,
                beta,
                state,
                beta_is_raw,
                use_qk_l2norm,
            )

        compiled = torch.compile(
            owned_leaf,
            backend="hpu_backend",
            dynamic=False,
        )
        _COMPILED_KDA_DECODE_STEPS[key] = compiled
    return compiled


def _hpu_fused_kda_conv_step(
    qkv: torch.Tensor,
    raw_gate: torch.Tensor,
    raw_beta: torch.Tensor,
    conv_rows: torch.Tensor,
    recurrent_rows: torch.Tensor,
    conv_weight: torch.Tensor,
    conv_bias: torch.Tensor | None,
    a_log: torch.Tensor,
    gate_bias: torch.Tensor,
    token_mask: torch.Tensor,
    lower_bound: float | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pure one-token GLM KDA transition, including depthwise convolution.

    Persistent state gather/scatter is deliberately outside this leaf.  All
    34 KDA layers share the same shapes, so one compiled recipe can consume
    each layer's live weights instead of compiling one recipe per layer.
    """

    batch = qkv.shape[0]
    heads = raw_gate.shape[-2]
    key_dim = raw_gate.shape[-1]
    projection = heads * key_dim
    state_width = conv_rows.shape[1]
    kernel_width = conv_weight.shape[1]
    if kernel_width != state_width + 1:
        raise ValueError("KDA conv state width must be kernel width minus one")

    # conv_rows is [B, state, channels].  Appending the one live token creates
    # the exact depthwise window used by hpu_causal_conv1d_update.
    conv_dtype = conv_rows.dtype
    history = conv_rows.transpose(1, 2)
    window = torch.cat((history, qkv.to(conv_dtype).unsqueeze(-1)), dim=-1)
    weight = conv_weight.to(conv_dtype)
    conv = window[:, :, 0] * weight[:, 0].unsqueeze(0)
    for tap in range(1, kernel_width):
        conv = conv + window[:, :, tap] * weight[:, tap].unsqueeze(0)
    if conv_bias is not None:
        conv = conv + conv_bias.to(conv_dtype).unsqueeze(0)
    conv = F.silu(conv)
    conv = conv * token_mask.to(conv.dtype)
    new_conv_rows = window[:, :, 1:].transpose(1, 2).to(conv_rows.dtype)

    q, k, v = conv.split(projection, dim=-1)
    q = q.reshape(batch, heads, key_dim)
    k = k.reshape(batch, heads, key_dim)
    v = v.reshape(batch, heads, key_dim)

    gate_arg = raw_gate.squeeze(0).float()
    gate_arg = gate_arg + gate_bias.float().reshape(1, heads, key_dim)
    decay_rate = torch.exp(a_log.float().reshape(1, heads, 1))
    if lower_bound is None:
        softplus = torch.where(
            gate_arg > 20.0,
            gate_arg,
            torch.log1p(torch.exp(gate_arg)),
        )
        gate = -decay_rate * softplus
    else:
        gate = float(lower_bound) * torch.sigmoid(decay_rate * gate_arg)

    q32 = _l2norm(q) * (key_dim**-0.5)
    k32 = _l2norm(k)
    beta32 = raw_beta.squeeze(0).float().sigmoid().unsqueeze(-1)
    state = recurrent_rows.float()
    state = state * torch.exp(gate).unsqueeze(-2)
    projection_value = (state * k32.unsqueeze(-2)).sum(dim=-1)
    delta = (v.float() - projection_value) * beta32
    state = state + delta.unsqueeze(-1) * k32.unsqueeze(-2)
    output = (state * q32.unsqueeze(-2)).sum(dim=-1)
    return output.to(v.dtype), new_conv_rows, state


def _get_compiled_fused_kda_decode_step(
    cache_key,
    *,
    lower_bound: float | None,
):
    """Return one reusable conv+gate+KDA decode recipe per batch shape."""

    key = (tuple(cache_key), lower_bound)
    compiled = _COMPILED_KDA_FUSED_DECODE_STEPS.get(key)
    if compiled is None:
        torch._dynamo.config.recompile_limit = max(
            int(torch._dynamo.config.recompile_limit), 256
        )
        if hasattr(torch._dynamo.config, "accumulated_recompile_limit"):
            torch._dynamo.config.accumulated_recompile_limit = max(
                int(torch._dynamo.config.accumulated_recompile_limit), 512
            )

        def owned_leaf(
            qkv,
            raw_gate,
            raw_beta,
            conv_rows,
            recurrent_rows,
            conv_weight,
            conv_bias,
            a_log,
            gate_bias,
            token_mask,
        ):
            return _hpu_fused_kda_conv_step(
                qkv,
                raw_gate,
                raw_beta,
                conv_rows,
                recurrent_rows,
                conv_weight,
                conv_bias,
                a_log,
                gate_bias,
                token_mask,
                lower_bound,
            )

        compiled = torch.compile(
            owned_leaf,
            backend="hpu_backend",
            dynamic=False,
        )
        _COMPILED_KDA_FUSED_DECODE_STEPS[key] = compiled
    return compiled


def hpu_fused_kda_decode(
    qkv: torch.Tensor,
    raw_gate: torch.Tensor,
    raw_beta: torch.Tensor,
    conv_state: torch.Tensor,
    recurrent_state: torch.Tensor,
    state_indices: torch.Tensor,
    conv_weight: torch.Tensor,
    conv_bias: torch.Tensor | None,
    a_log: torch.Tensor,
    gate_bias: torch.Tensor,
    token_mask: torch.Tensor,
    lower_bound: float | None,
) -> torch.Tensor:
    """Gather live state, run the fused leaf, then persist both state pools."""

    slots = state_indices.long()
    conv_rows = conv_state.index_select(0, slots)
    recurrent_rows = recurrent_state.index_select(0, slots)
    cache_key = (
        tuple(qkv.shape),
        tuple(raw_gate.shape),
        tuple(raw_beta.shape),
        tuple(conv_rows.shape),
        tuple(recurrent_rows.shape),
        tuple(conv_weight.shape),
        None if conv_bias is None else tuple(conv_bias.shape),
        tuple(token_mask.shape),
        str(qkv.dtype),
        str(raw_gate.dtype),
        str(raw_beta.dtype),
        str(conv_rows.dtype),
        str(recurrent_rows.dtype),
        str(conv_weight.dtype),
        "none" if conv_bias is None else str(conv_bias.dtype),
        str(token_mask.dtype),
    )
    if _ENABLE_FUSED_KDA_DECODE and qkv.device.type == "hpu":
        step = _get_compiled_fused_kda_decode_step(
            cache_key,
            lower_bound=lower_bound,
        )
        output, new_conv, new_recurrent = step(
            qkv,
            raw_gate,
            raw_beta,
            conv_rows,
            recurrent_rows,
            conv_weight,
            conv_bias,
            a_log,
            gate_bias,
            token_mask,
        )
    else:
        output, new_conv, new_recurrent = _hpu_fused_kda_conv_step(
            qkv,
            raw_gate,
            raw_beta,
            conv_rows,
            recurrent_rows,
            conv_weight,
            conv_bias,
            a_log,
            gate_bias,
            token_mask,
            lower_bound,
        )
    conv_state.index_copy_(0, slots, new_conv.to(conv_state.dtype))
    recurrent_state.index_copy_(
        0, slots, new_recurrent.to(recurrent_state.dtype)
    )
    return output


def _get_compiled_kda_spec_step(cache_key):
    """Return a dedicated small HPU recipe per MTP state-transition shape."""

    key = tuple(cache_key)
    compiled = _COMPILED_KDA_SPEC_STEPS.get(key)
    if compiled is None:
        # C1..C64 plus the MTP buckets legitimately exceed Dynamo's default
        # eight specializations.  Separate wrappers prevent a later shape from
        # falling back to the 413-node eager supergraph that Synapse rejects.
        torch._dynamo.config.recompile_limit = max(
            int(torch._dynamo.config.recompile_limit), 256
        )
        if hasattr(torch._dynamo.config, "accumulated_recompile_limit"):
            torch._dynamo.config.accumulated_recompile_limit = max(
                int(torch._dynamo.config.accumulated_recompile_limit), 512
            )
        compiled = torch.compile(
            _hpu_kda_spec_step,
            backend="hpu_backend",
            dynamic=False,
        )
        _COMPILED_KDA_SPEC_STEPS[key] = compiled
    return compiled


def _hpu_kda_spec_uniform_ladder(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    recurrent_state: torch.Tensor,
    state_indices: torch.Tensor,
    num_accepted_tokens: torch.Tensor,
    token_mask: torch.Tensor,
    beta_is_raw: bool,
    use_qk_l2norm: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pure fixed-width MTP ladder compiled as one bounded HPU recipe."""

    batch, width = state_indices.shape
    q = q.reshape(batch, width, *q.shape[1:])
    k = k.reshape(batch, width, *k.shape[1:])
    v = v.reshape(batch, width, *v.shape[1:])
    g = g.reshape(batch, width, *g.shape[1:])
    beta = beta.reshape(batch, width, *beta.shape[1:])
    active_mask = token_mask.reshape(batch, width).bool()
    accepted_col = (num_accepted_tokens.long() - 1).clamp(0, width - 1)
    initial_slots = torch.gather(
        state_indices.long(), 1, accepted_col.unsqueeze(1)
    ).squeeze(1)
    initial_slots = torch.remainder(initial_slots, recurrent_state.shape[0])
    state = recurrent_state.index_select(0, initial_slots).float()
    outputs = []
    states = []
    for token in range(width):
        token_out, next_state = _hpu_kda_spec_step(
            q[:, token],
            k[:, token],
            v[:, token],
            g[:, token],
            beta[:, token],
            state,
            beta_is_raw,
            use_qk_l2norm,
        )
        active = active_mask[:, token]
        state = torch.where(active[:, None, None, None], next_state, state)
        outputs.append(torch.where(active[:, None, None], token_out, 0.0))
        destination = torch.remainder(
            state_indices[:, token].long(), recurrent_state.shape[0]
        )
        old_destination = recurrent_state.index_select(0, destination)
        states.append(
            torch.where(
                active[:, None, None, None],
                state.to(old_destination.dtype),
                old_destination,
            )
        )
    return torch.stack(outputs, dim=1), torch.stack(states, dim=1)


def _get_compiled_kda_spec_ladder(cache_key):
    key = tuple(cache_key)
    compiled = _COMPILED_KDA_SPEC_LADDERS.get(key)
    if compiled is None:
        torch._dynamo.config.recompile_limit = max(
            int(torch._dynamo.config.recompile_limit), 256
        )
        if hasattr(torch._dynamo.config, "accumulated_recompile_limit"):
            torch._dynamo.config.accumulated_recompile_limit = max(
                int(torch._dynamo.config.accumulated_recompile_limit), 512
            )
        compiled = torch.compile(
            _hpu_kda_spec_uniform_ladder,
            backend="hpu_backend",
            dynamic=False,
        )
        _COMPILED_KDA_SPEC_LADDERS[key] = compiled
    return compiled


def _hpu_kda_spec_uniform_transaction(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    recurrent_state: torch.Tensor,
    transition_factors: torch.Tensor,
    logical_state_indices: torch.Tensor,
    num_accepted_tokens: torch.Tensor,
    token_mask: torch.Tensor,
    beta_is_raw: bool,
    use_qk_l2norm: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Two-token KDA ladder with one full matrix and one O(D) delta."""

    batch, width = logical_state_indices.shape
    if width != 2:
        raise ValueError("transactional KDA currently supports MTP width two")
    q = q.reshape(batch, width, *q.shape[1:])
    k = k.reshape(batch, width, *k.shape[1:])
    v = v.reshape(batch, width, *v.shape[1:])
    g = g.reshape(batch, width, *g.shape[1:])
    beta = beta.reshape(batch, width, *beta.shape[1:])
    active = token_mask.reshape(batch, width).bool()

    # Logical slots retain the two-column convolution ladder.  Recurrent rows
    # collapse each adjacent pair to one canonical full-state slot.
    logical_base = logical_state_indices[:, 0].long()
    base_slots = torch.where(
        logical_base > 0,
        torch.div(logical_base - 1, width, rounding_mode="floor") + 1,
        torch.zeros_like(logical_base),
    )
    base_slots = torch.remainder(base_slots, recurrent_state.shape[0])
    base = recurrent_state.index_select(0, base_slots).float()
    prior_factors = transition_factors.index_select(0, base_slots)
    accepted_second = num_accepted_tokens.long() > 1
    reconstructed = hpu_kda_apply_transition_factors(base, prior_factors)
    state = torch.where(
        accepted_second[:, None, None, None], reconstructed, base
    )

    out0, next0, _ = _hpu_kda_spec_step_with_factors(
        q[:, 0], k[:, 0], v[:, 0], g[:, 0], beta[:, 0], state,
        beta_is_raw, use_qk_l2norm,
    )
    base_write = torch.where(active[:, 0, None, None, None], next0, state)
    out0 = torch.where(active[:, 0, None, None], out0, 0.0)

    out1, _, factor1 = _hpu_kda_spec_step_with_factors(
        q[:, 1], k[:, 1], v[:, 1], g[:, 1], beta[:, 1], base_write,
        beta_is_raw, use_qk_l2norm,
    )
    factor_write = torch.where(
        active[:, 1, None, None], factor1, torch.zeros_like(factor1)
    )
    out1 = torch.where(active[:, 1, None, None], out1, 0.0)
    return torch.stack((out0, out1), dim=1), base_write, factor_write


def _get_compiled_kda_txn_ladder(cache_key):
    key = tuple(cache_key)
    compiled = _COMPILED_KDA_TXN_LADDERS.get(key)
    if compiled is None:
        torch._dynamo.config.recompile_limit = max(
            int(torch._dynamo.config.recompile_limit), 256
        )
        if hasattr(torch._dynamo.config, "accumulated_recompile_limit"):
            torch._dynamo.config.accumulated_recompile_limit = max(
                int(torch._dynamo.config.accumulated_recompile_limit), 512
            )
        compiled = torch.compile(
            _hpu_kda_spec_uniform_transaction,
            backend="hpu_backend",
            dynamic=False,
        )
        _COMPILED_KDA_TXN_LADDERS[key] = compiled
    return compiled


def _pack_variable_sequences(
    x: torch.Tensor,
    query_start_loc: torch.Tensor,
    width: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pack flat token rows into a fixed ``[B,width,...]`` graph bucket."""

    batch = query_start_loc.numel() - 1
    token = torch.arange(x.shape[0], device=x.device, dtype=torch.long)
    req = torch.searchsorted(query_start_loc[1:], token, right=True)
    start = query_start_loc.index_select(0, req).long()
    offset = token - start
    packed = torch.zeros((batch, width, *x.shape[1:]), dtype=x.dtype, device=x.device)
    packed[req, offset] = x
    return packed, req, offset


def _hpu_causal_conv1d_spec_update_impl(
    x: torch.Tensor,
    conv_state: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    state_indices: torch.Tensor,
    num_accepted_tokens: torch.Tensor,
    query_start_loc: torch.Tensor,
    *,
    activation: str | None = "silu",
    token_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Rollback-correct short convolution for built-in MTP verification."""

    dim, kernel = weight.shape
    history = kernel - 1
    state_width = (
        conv_state.shape[-1] if conv_state.shape[1] == dim else conv_state.shape[1]
    )
    max_query_len = state_width - history + 1
    if max_query_len <= 0 or state_indices.shape[1] < max_query_len:
        raise ValueError("conv state/spec columns are inconsistent")

    packed, req, offset = _pack_variable_sequences(x, query_start_loc, max_query_len)
    packed = packed.transpose(1, 2).float()  # [B,D,M]
    slots = torch.remainder(state_indices[:, 0].long(), conv_state.shape[0])
    if conv_state.shape[1] == dim:
        state_dim_first = conv_state
    elif conv_state.shape[2] == dim:
        state_dim_first = conv_state.transpose(1, 2)
    else:
        raise ValueError("conv_state must contain the projection dimension")

    extended = state_dim_first.index_select(0, slots).float()
    accepted_offset = (num_accepted_tokens.long() - 1).clamp(
        min=0, max=state_width - history
    )
    history_index = accepted_offset[:, None, None] + torch.arange(
        history, device=x.device
    )[None, None, :]
    history_index = history_index.expand(-1, dim, -1)
    initial = torch.gather(extended, 2, history_index)
    if token_mask is None:
        sequence = torch.cat((initial, packed), dim=-1)
        windows = sequence.unfold(-1, kernel, 1)
        out = (windows * weight.float()[None, :, None, :]).sum(-1)
        if bias is not None:
            out = out + bias.float()[None, :, None]
        active_tokens = (
            torch.arange(max_query_len, device=x.device)[None, :]
            < torch.diff(query_start_loc).long()[:, None]
        )
    else:
        mask_packed, _, _ = _pack_variable_sequences(
            token_mask.reshape(-1, 1), query_start_loc, max_query_len
        )
        active_tokens = mask_packed[..., 0].bool()
        history_state = initial
        outputs = []
        appended = []
        for token in range(max_query_len):
            x_t = packed[:, :, token]
            active = active_tokens[:, token]
            window = torch.cat((history_state, x_t.unsqueeze(-1)), dim=-1)
            out_t = (window * weight.float()[None, :, :]).sum(-1)
            if bias is not None:
                out_t = out_t + bias.float()[None, :]
            outputs.append(torch.where(active[:, None], out_t, 0.0))
            appended_t = torch.where(active[:, None], x_t, history_state[:, :, -1])
            appended.append(appended_t)
            next_history = torch.cat(
                (history_state[:, :, 1:], appended_t.unsqueeze(-1)), dim=-1
            )
            history_state = torch.where(
                active[:, None, None], next_history, history_state
            )
        out = torch.stack(outputs, dim=-1)
        sequence = torch.cat((initial, torch.stack(appended, dim=-1)), dim=-1)

    if activation in ("silu", "swish"):
        out = F.silu(out)
    elif activation is not None:
        raise ValueError(f"unsupported activation {activation}")
    out = torch.where(active_tokens[:, None, :], out, 0.0)

    # After dropping the first history value, columns [0..M-1] encode the
    # histories after accepting 1..M new tokens for the next rollback choice.
    new_extended = sequence[:, :, 1 : 1 + state_width]
    active = active_tokens.any(dim=1)
    old = state_dim_first.index_select(0, slots)
    new_extended = torch.where(active[:, None, None], new_extended.to(old.dtype), old)
    state_dim_first.index_copy_(0, slots, new_extended)
    return out.transpose(1, 2)[req, offset].to(x.dtype)


def _get_compiled_kda_spec_conv(cache_key):
    """Return one bounded HPU short-convolution recipe per MTP bucket."""

    key = tuple(cache_key)
    compiled = _COMPILED_KDA_SPEC_CONVS.get(key)
    if compiled is None:
        torch._dynamo.config.recompile_limit = max(
            int(torch._dynamo.config.recompile_limit), 256
        )
        if hasattr(torch._dynamo.config, "accumulated_recompile_limit"):
            torch._dynamo.config.accumulated_recompile_limit = max(
                int(torch._dynamo.config.accumulated_recompile_limit), 512
            )
        compiled = torch.compile(
            _hpu_causal_conv1d_spec_update_impl,
            backend="hpu_backend",
            dynamic=False,
        )
        _COMPILED_KDA_SPEC_CONVS[key] = compiled
    return compiled


def hpu_causal_conv1d_spec_update(
    x: torch.Tensor,
    conv_state: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    state_indices: torch.Tensor,
    num_accepted_tokens: torch.Tensor,
    query_start_loc: torch.Tensor,
    *,
    activation: str | None = "silu",
    token_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Dispatch fixed-width native-MTP convolution as its own HPU recipe.

    The pure Python implementation is correct in isolation, but when left in
    the eager KDA wrapper Synapse fuses it with the surrounding projection and
    recurrent paths. The first live verification step then exceeds the graph
    compiler boundary and fails with status 26. Native MTP always uses a fixed
    rectangular width, so each C1..C64 shape can safely own one recipe.
    """

    dim = weight.shape[0]
    history = weight.shape[1] - 1
    state_width = (
        conv_state.shape[-1] if conv_state.shape[1] == dim else conv_state.shape[1]
    )
    max_query_len = state_width - history + 1
    batch = state_indices.shape[0]
    uniform = (
        max_query_len > 0
        and state_indices.ndim == 2
        and state_indices.shape[1] >= max_query_len
        and x.shape[0] == batch * max_query_len
    )
    update = _hpu_causal_conv1d_spec_update_impl
    if uniform and x.device.type == "hpu":
        update = _get_compiled_kda_spec_conv(
            (
                *x.shape,
                *conv_state.shape,
                *weight.shape,
                *state_indices.shape,
                str(x.dtype),
                str(conv_state.dtype),
                str(weight.dtype),
                None if bias is None else str(bias.dtype),
                str(state_indices.dtype),
                str(num_accepted_tokens.dtype),
                str(query_start_loc.dtype),
                None if token_mask is None else str(token_mask.dtype),
                activation,
            )
        )
    return update(
        x,
        conv_state,
        weight,
        bias,
        state_indices,
        num_accepted_tokens,
        query_start_loc,
        activation=activation,
        token_mask=token_mask,
    )


def hpu_recurrent_kda_spec_decode(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    recurrent_state: torch.Tensor,
    state_indices: torch.Tensor,
    num_accepted_tokens: torch.Tensor,
    query_start_loc: torch.Tensor,
    *,
    transition_factors: torch.Tensor | None = None,
    beta_is_raw: bool = True,
    use_qk_l2norm: bool = True,
    token_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Vectorized, rollback-correct KDA ladder for built-in MTP."""

    max_query_len = state_indices.shape[1]
    batch = state_indices.shape[0]
    uniform = q.shape[0] == batch * max_query_len
    if transition_factors is not None:
        if not uniform or max_query_len != 2:
            raise RuntimeError(
                "transactional KDA requires a uniform two-token MTP bucket"
            )
        if token_mask is None:
            uniform_mask = torch.ones(
                (batch, max_query_len), dtype=torch.bool, device=q.device
            )
        else:
            uniform_mask = token_mask.reshape(batch, max_query_len).bool()
        ladder = _hpu_kda_spec_uniform_transaction
        if q.device.type == "hpu":
            ladder = _get_compiled_kda_txn_ladder(
                (
                    *q.shape,
                    *recurrent_state.shape,
                    *transition_factors.shape,
                    *state_indices.shape,
                    str(q.dtype),
                    str(g.dtype),
                    str(beta.dtype),
                    str(recurrent_state.dtype),
                    beta_is_raw,
                    use_qk_l2norm,
                )
            )
        outputs, base_write, factor_write = ladder(
            q,
            k,
            v,
            g,
            beta,
            recurrent_state,
            transition_factors,
            state_indices,
            num_accepted_tokens,
            uniform_mask,
            beta_is_raw,
            use_qk_l2norm,
        )
        logical_base = state_indices[:, 0].long()
        base_slots = torch.where(
            logical_base > 0,
            torch.div(
                logical_base - 1,
                max_query_len,
                rounding_mode="floor",
            ) + 1,
            torch.zeros_like(logical_base),
        )
        base_slots = torch.remainder(base_slots, recurrent_state.shape[0])
        recurrent_state.index_copy_(
            0, base_slots, base_write.to(recurrent_state.dtype)
        )
        transition_factors.index_copy_(
            0, base_slots, factor_write.to(transition_factors.dtype)
        )
        return outputs.reshape(-1, *outputs.shape[2:]).to(v.dtype)
    if uniform and q.device.type == "hpu":
        # Keep all fixed-width packing, rollback selection and two MTP state
        # transitions inside one explicitly bounded recipe.  Letting the HPU
        # eager pipeline discover this boundary produced status 26 at C64.
        if token_mask is None:
            uniform_mask = torch.ones(
                (batch, max_query_len), dtype=torch.bool, device=q.device
            )
        else:
            uniform_mask = token_mask.reshape(batch, max_query_len).bool()
        compiled_ladder = _get_compiled_kda_spec_ladder(
            (
                *q.shape,
                *recurrent_state.shape,
                *state_indices.shape,
                str(q.dtype),
                str(g.dtype),
                str(beta.dtype),
                str(recurrent_state.dtype),
                beta_is_raw,
                use_qk_l2norm,
            )
        )
        uniform_output, uniform_states = compiled_ladder(
            q,
            k,
            v,
            g,
            beta,
            recurrent_state,
            state_indices,
            num_accepted_tokens,
            uniform_mask,
            beta_is_raw,
            use_qk_l2norm,
        )
        destinations = torch.remainder(
            state_indices.long(), recurrent_state.shape[0]
        ).reshape(-1)
        recurrent_state.index_copy_(
            0,
            destinations,
            uniform_states.reshape(
                -1,
                recurrent_state.shape[1],
                recurrent_state.shape[2],
                recurrent_state.shape[3],
            ).to(recurrent_state.dtype),
        )
        return uniform_output.reshape(-1, *uniform_output.shape[2:]).to(v.dtype)
    if uniform:
        # Native MTP uses a fixed verification width.  A direct view avoids
        # five searchsorted/scatter packing graphs; at C64 those graphs alone
        # hit Synapse status 26 before the compiled state-transition leaf ran.
        q_p = q.reshape(batch, max_query_len, *q.shape[1:])
        k_p = k.reshape(batch, max_query_len, *k.shape[1:])
        v_p = v.reshape(batch, max_query_len, *v.shape[1:])
        g_p = g.reshape(batch, max_query_len, *g.shape[1:])
        b_p = beta.reshape(batch, max_query_len, *beta.shape[1:])
        lengths = torch.full(
            (batch,), max_query_len, dtype=torch.long, device=q.device
        )
        req = offset = None
    else:
        q_p, req, offset = _pack_variable_sequences(
            q, query_start_loc, max_query_len
        )
        k_p, _, _ = _pack_variable_sequences(k, query_start_loc, max_query_len)
        v_p, _, _ = _pack_variable_sequences(v, query_start_loc, max_query_len)
        g_p, _, _ = _pack_variable_sequences(g, query_start_loc, max_query_len)
        b_p, _, _ = _pack_variable_sequences(beta, query_start_loc, max_query_len)
        lengths = torch.diff(query_start_loc).long()
    mask_p = None
    if token_mask is not None:
        if uniform:
            mask_p = token_mask.reshape(batch, max_query_len).bool()
        else:
            mask_p, _, _ = _pack_variable_sequences(
                token_mask.reshape(-1, 1), query_start_loc, max_query_len
            )
            mask_p = mask_p[..., 0].bool()

    accepted_col = (num_accepted_tokens.long() - 1).clamp(
        min=0, max=max_query_len - 1
    )
    initial_slots = torch.gather(
        state_indices.long(), 1, accepted_col.unsqueeze(1)
    ).squeeze(1)
    initial_slots = torch.remainder(initial_slots, recurrent_state.shape[0])
    state = recurrent_state.index_select(0, initial_slots).float()
    output = torch.zeros_like(v_p)

    compiled_step = None
    if q.device.type == "hpu":
        compiled_step = _get_compiled_kda_spec_step(
            (
                *q_p[:, 0].shape,
                *state.shape,
                str(q.dtype),
                str(g.dtype),
                str(beta.dtype),
                str(state.dtype),
                beta_is_raw,
                use_qk_l2norm,
            )
        )

    for token in range(max_query_len):
        q_t, k_t, v_t = q_p[:, token], k_p[:, token], v_p[:, token]
        g_t, beta_t = g_p[:, token], b_p[:, token]
        step = compiled_step or _hpu_kda_spec_step
        token_out, next_state = step(
            q_t,
            k_t,
            v_t,
            g_t,
            beta_t,
            state,
            beta_is_raw,
            use_qk_l2norm,
        )
        active = token < lengths
        if mask_p is not None:
            active = active & mask_p[:, token]
        state = torch.where(active[:, None, None, None], next_state, state)
        output[:, token] = torch.where(
            active[:, None, None], token_out.to(output.dtype), output[:, token]
        )

        destination = torch.remainder(
            state_indices[:, token].long(), recurrent_state.shape[0]
        )
        old = recurrent_state.index_select(0, destination)
        write = torch.where(active[:, None, None, None], state.to(old.dtype), old)
        recurrent_state.index_copy_(0, destination, write)

    if uniform:
        return output.reshape(-1, *output.shape[2:]).to(v.dtype)
    return output[req, offset].to(v.dtype)


def hpu_recurrent_kda_reference(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    initial_state: torch.Tensor | None = None,
    *,
    use_qk_l2norm: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Packed-sequence correctness reference with vLLM state orientation."""

    input_dtype = q.dtype
    q32 = _l2norm(q) if use_qk_l2norm else q.float()
    k32 = _l2norm(k) if use_qk_l2norm else k.float()
    v32, g32, beta32 = v.float(), g.float(), beta.float()
    q32 = q32 * (q.shape[-1] ** -0.5)
    batch, seq_len, heads, key_dim = q.shape
    value_dim = v.shape[-1]
    if initial_state is None:
        state = torch.zeros(
            batch,
            heads,
            value_dim,
            key_dim,
            dtype=torch.float32,
            device=q.device,
        )
    else:
        state = initial_state.float().clone()
    out = torch.empty(
        batch, seq_len, heads, value_dim, dtype=torch.float32, device=q.device
    )
    for token in range(seq_len):
        q_t = q32[:, token]
        k_t = k32[:, token]
        v_t = v32[:, token]
        g_t = g32[:, token]
        b_t = beta32[:, token]
        state = state * torch.exp(g_t).unsqueeze(-2)
        projection = (state * k_t.unsqueeze(-2)).sum(dim=-1)
        delta = (v_t - projection) * b_t.unsqueeze(-1)
        state = state + delta.unsqueeze(-1) * k_t.unsqueeze(-2)
        out[:, token] = (state * q_t.unsqueeze(-2)).sum(dim=-1)
    return out.to(input_dtype), state


def _r589_original_chunk_kda(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    *,
    chunk_size: int = 64,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = True,
    use_qk_l2norm: bool = True,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Official chunk-KDA math expressed as portable PyTorch tensor ops.

    This is the initial HPU prefill implementation and an oracle for later
    fused recipes.  It performs all recurrent-state-sensitive arithmetic in
    FP32, matching the official Transformers fallback.
    """

    input_dtype = q.dtype
    q, k, v, beta, g = [
        x.transpose(1, 2).contiguous().float() for x in (q, k, v, beta, g)
    ]
    if use_qk_l2norm:
        q, k = _l2norm(q), _l2norm(k)

    batch, heads, seq_len, key_dim = k.shape
    value_dim = v.shape[-1]
    scale = q.shape[-1] ** -0.5
    pad = (chunk_size - seq_len % chunk_size) % chunk_size
    total_len = seq_len + pad
    q = F.pad(q, (0, 0, 0, pad)) * scale
    k = F.pad(k, (0, 0, 0, pad))
    v = F.pad(v, (0, 0, 0, pad))
    g = F.pad(g, (0, 0, 0, pad))
    beta = F.pad(beta, (0, pad))
    v_beta = v * beta.unsqueeze(-1)
    k_beta = k * beta.unsqueeze(-1)

    def chunks(x: torch.Tensor) -> torch.Tensor:
        return x.reshape(x.shape[0], x.shape[1], -1, chunk_size, x.shape[-1])

    q, k, v, g, k_beta, v_beta = [chunks(x) for x in (q, k, v, g, k_beta, v_beta)]
    beta = beta.reshape(beta.shape[0], beta.shape[1], -1, chunk_size)
    del beta, v

    g = g.cumsum(dim=-2)
    upper_including_diag = torch.triu(
        torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=q.device)
    )
    row_decays = None
    if (os.getenv("HPU_GLM53_KDA_BITWISE_ROWS_PREFILL", "0") == "1"
            and (batch, heads, key_dim, value_dim, chunk_size) == (1, 16, 128, 128, 128)
            and seq_len == total_len and total_len in (128, 256, 512)):
        from glm53_mixed_graph import witness_lever
        witness_lever("kda_bitwise_rows")
        row_decays = []
        key_rows = []
        for start in range(0, chunk_size, 16):
            end = start + 16
            decay_row = (g[..., start:end, :].unsqueeze(-2)
                         - g[..., :end, :].unsqueeze(-3)).exp()
            key_row = (k_beta[..., start:end, :].unsqueeze(-2)
                       * k[..., :end, :].unsqueeze(-3) * decay_row).sum(-1)
            row_decays.append(decay_row)
            key_rows.append(F.pad(key_row, (0, chunk_size - end)))
        attn = -torch.cat(key_rows, dim=-2).masked_fill(upper_including_diag, 0)
    else:
        decay = (g.unsqueeze(-2) - g.unsqueeze(-3)).exp()
        attn = -(
            k_beta.unsqueeze(-2) * k.unsqueeze(-3) * decay
        ).sum(dim=-1).masked_fill(upper_including_diag, 0)
    if _ENABLE_KDA_BLOCK_INVERSE_PREFILL:
        from glm53_mixed_graph import witness_lever
        witness_lever("kda_block_inverse")
        # Exact inverse of L = I - A for strictly-lower A.  Invert all 16x16
        # diagonal blocks together, then merge adjacent blocks using
        # inv([[A,0],[C,D]]) = [[A^-1,0],[-D^-1 C A^-1,D^-1]].
        # This retains MME-friendly batched GEMMs while reducing the solve from
        # 127 serial variable-size row operations to 15 base steps plus three
        # two-GEMM merges for the production 128-token KDA chunk.
        flat_attn = attn.reshape(-1, chunk_size, chunk_size)
        if chunk_size < 16 or chunk_size % 16 != 0 or (chunk_size & (chunk_size - 1)):
            raise RuntimeError(
                "block KDA inverse requires a power-of-two chunk divisible by 16"
            )
        lmat = -flat_attn
        lmat.diagonal(dim1=-2, dim2=-1).fill_(1)
        matrix_count = lmat.shape[0]
        block_size = 16
        block_count = chunk_size // block_size
        diagonal_blocks = torch.stack([
            lmat[:, block_idx * block_size:(block_idx + 1) * block_size,
                 block_idx * block_size:(block_idx + 1) * block_size]
            for block_idx in range(block_count)
        ], dim=1)
        base = diagonal_blocks.reshape(-1, block_size, block_size)
        inverse = torch.zeros_like(base)
        inverse[:, 0, 0] = 1
        for row_idx in range(1, block_size):
            correction = torch.bmm(
                base[:, row_idx:row_idx + 1, :row_idx],
                inverse[:, :row_idx, :],
            )
            inverse[:, row_idx:row_idx + 1, :] = -correction
            inverse[:, row_idx, row_idx] = 1
        inverse_blocks = inverse.reshape(
            matrix_count, block_count, block_size, block_size
        )
        while block_count > 1:
            pair_count = block_count // 2
            a_inv = inverse_blocks[:, 0::2]
            d_inv = inverse_blocks[:, 1::2]
            c_blocks = torch.stack([
                lmat[:, (2 * pair_idx + 1) * block_size:
                         (2 * pair_idx + 2) * block_size,
                         (2 * pair_idx) * block_size:
                         (2 * pair_idx + 1) * block_size]
                for pair_idx in range(pair_count)
            ], dim=1)
            d_flat = d_inv.reshape(-1, block_size, block_size)
            c_flat = c_blocks.reshape(-1, block_size, block_size)
            a_flat = a_inv.reshape(-1, block_size, block_size)
            lower_left = -torch.bmm(
                torch.bmm(d_flat, c_flat), a_flat
            ).reshape(matrix_count, pair_count, block_size, block_size)
            zeros = torch.zeros_like(a_inv)
            top = torch.cat((a_inv, zeros), dim=-1)
            bottom = torch.cat((lower_left, d_inv), dim=-1)
            inverse_blocks = torch.cat((top, bottom), dim=-2)
            block_count = pair_count
            block_size *= 2
        attn = inverse_blocks[:, 0].reshape_as(attn)
    else:
        for row_idx in range(1, chunk_size):
            row = attn[..., row_idx, :row_idx].clone()
            sub = attn[..., :row_idx, :row_idx].clone()
            attn[..., row_idx, :row_idx] = row + (row.unsqueeze(-1) * sub).sum(-2)

        # masked_fill returned fresh storage with a zero diagonal. Inject I
        # directly; cloning the full attention tensor here can race lazy collectives.
        attn.diagonal(dim1=-2, dim2=-1).fill_(1)
    value = attn @ v_beta
    k_cumdecay = attn @ (k_beta * g.exp())

    if initial_state is None:
        # Official orientation [B,H,K,V] for the chunk algebra.
        state = torch.zeros(
            batch,
            heads,
            key_dim,
            value_dim,
            dtype=torch.float32,
            device=q.device,
        )
    else:
        state = initial_state.float().transpose(-1, -2).contiguous()
    out = torch.zeros_like(value)
    upper_strict = torch.triu(
        torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=q.device),
        diagonal=1,
    )
    for chunk_idx in range(total_len // chunk_size):
        q_i, k_i = q[:, :, chunk_idx], k[:, :, chunk_idx]
        v_i, g_i = value[:, :, chunk_idx], g[:, :, chunk_idx]
        inter = (q_i * g_i.exp()) @ state
        if row_decays is not None:
            query_rows = []
            for row_index, start in enumerate(range(0, chunk_size, 16)):
                end = start + 16
                query_row = (q_i[..., start:end, :].unsqueeze(-2)
                             * k_i[..., :end, :].unsqueeze(-3)
                             * row_decays[row_index][:, :, chunk_idx]).sum(-1)
                query_rows.append(F.pad(query_row, (0, chunk_size - end)))
            intra = torch.cat(query_rows, dim=-2).masked_fill(upper_strict, 0)
        else:
            intra = (
                q_i.unsqueeze(-2) * k_i.unsqueeze(-3) * decay[:, :, chunk_idx]
            ).sum(dim=-1).masked_fill(upper_strict, 0)
        v_prime = k_cumdecay[:, :, chunk_idx] @ state
        v_new = v_i - v_prime
        out[:, :, chunk_idx] = inter + intra @ v_new
        state = (
            state * g_i[:, :, -1].exp().unsqueeze(-1)
            + (k_i * (g_i[:, :, -1:] - g_i).exp()).transpose(-1, -2) @ v_new
        )

    out = out.reshape(batch, heads, total_len, value_dim)[:, :, :seq_len]
    out = out.transpose(1, 2).contiguous().to(input_dtype)
    final = state.transpose(-1, -2).contiguous() if output_final_state else None
    return out, final


# ---------------------------------------------------------------------------
# KDA V2 (FORK-PORT-PLAN S6, HPU_GLM53_KDA_V2_FOR).
# _DECAY_BLOCK and _decay_dot are adapted from, and the Phase-B precompute in
# _kda_v2_batched_recurrence is adapted from,
# github.com/abotsis/vllm-gaudi branch carve/5-tpc,
# vllm_gaudi/ops/hpu_kda_pytorch.py :100-139 and :229-275, which carries:
#   SPDX-License-Identifier: Apache-2.0
#   SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# The fork's Neumann/Newton inverse is NOT taken: V2 keeps our exact block
# inverse (the HPU_GLM53_KDA_BLOCK_INVERSE_PREFILL algebra).
# ---------------------------------------------------------------------------

# Rows per exponent reference in _decay_dot (the fork's block). Inside a block
# the same split recurses with blocks a quarter the size, down to
# _DECAY_EXACT_ROWS, where pairs take their decay as an exact difference.
_DECAY_BLOCK = 16
_DECAY_EXACT_ROWS = 4
# The gate floor V2 is validated for (GLM-5.3 gate_lower_bound); the guard enforces it.
_KDA_V2_GATE_FLOOR = -5.0


def _decay_dot(
    a: torch.Tensor,
    b: torch.Tensor,
    ga: torch.Tensor,
    gb: torch.Tensor,
    block: int = _DECAY_BLOCK,
) -> torch.Tensor:
    """sum_d a[..., i, d] * b[..., j, d] * exp(ga[..., i, d] - gb[..., j, d]) for j <= i.

    All inputs [B, tc, D] with tc a multiple of ``block``; returns [B, tc, tc].
    Entries j > i are not meaningful (the callers apply tril). ga/gb are
    per-chunk cumulative log decays, non-increasing along the row axis (gate <= 0).

    Every exp argument is <= 0, so no two exps have to cancel (r3: the fork's
    form let e^{-75} * e^{+75} cancel inside a block, and the HPU exp error grows
    with |x|; the -5 floor gate lost 1e-5 relative on the quad-A probe):
      - j in an earlier block than i: with c = ga at the first row of i's block,
        exp(ga_i - gb_j) = exp(ga_i - c) * exp(c - gb_j), both arguments <= 0;
        one batched MME matmul covers every block, and the j >= block-start
        columns it also produces are masked out.
      - j in i's own block: the same split one level down (blocks of block/4),
        and below _DECAY_EXACT_ROWS rows exp(ga_i - gb_j) directly (clamped at 0
        for j > i).
    """
    B, tc, D = a.shape
    if tc <= _DECAY_EXACT_ROWS:
        pair_decay = (ga.unsqueeze(-2) - gb.unsqueeze(-3)).clamp(max=0.0).exp()
        return (a.unsqueeze(-2) * b.unsqueeze(-3) * pair_decay).sum(dim=-1)
    block_count = tc // block
    rows = (B, block_count, block, D)
    c = ga.reshape(rows)[:, :, :1, :]  # [B, nb, 1, D] reference per block and channel
    left = a.reshape(rows) * (ga.reshape(rows) - c).exp()  # [B, nb, blk, D], factors in (0, 1]
    right = b.unsqueeze(1) * (c - gb.unsqueeze(1)).clamp(max=0.0).exp()  # [B, nb, tc, D]
    earlier = torch.matmul(left, right.transpose(-1, -2))  # [B, nb, blk, tc]
    column_block = torch.arange(tc, device=a.device) // block
    row_block = torch.arange(block_count, device=a.device)
    earlier = earlier.masked_fill(
        (column_block.unsqueeze(0) >= row_block.unsqueeze(1)).reshape(1, block_count, 1, tc), 0.0
    )
    flat = (B * block_count, block, D)
    own = _decay_dot(a.reshape(flat), b.reshape(flat), ga.reshape(flat), gb.reshape(flat),
                     max(block // 4, 1)).reshape(B, block_count, block, block)
    eye_blocks = torch.eye(block_count, dtype=own.dtype, device=a.device)
    placed = own.unsqueeze(-2) * eye_blocks.reshape(1, block_count, 1, block_count, 1)
    return (earlier + placed.reshape(B, block_count, block, tc)).reshape(B, tc, tc)


def _require_kda_v2_contract(chunk_size: int) -> None:
    """Refuse inputs the 16-row decay_dot and the exact block inverse are not sized for."""

    bound = getattr(_KDA_V2_GATE, "lower_bound", None)
    if bound is None or not (_KDA_V2_GATE_FLOOR <= bound <= 0.0):
        raise RuntimeError(
            f"{_KDA_V2_FLAG}={','.join(sorted(_KDA_V2_FOR))} needs the kda_safe_gate "
            f"forget gate with {_KDA_V2_GATE_FLOOR} <= lower_bound <= 0 (its decay_dot needs "
            f"a non-positive gate and is validated to that floor), but the last hpu_kda_gate call had "
            f"lower_bound={bound!r} (None = unbounded gate or no gate yet); unset "
            f"{_KDA_V2_FLAG} for this checkpoint"
        )
    if chunk_size < _DECAY_BLOCK or chunk_size % _DECAY_BLOCK or chunk_size & (chunk_size - 1):
        raise RuntimeError(
            f"{_KDA_V2_FLAG} needs chunk_size to be a power of two divisible by "
            f"{_DECAY_BLOCK} (decay blocks and the exact block inverse), got "
            f"{chunk_size}; call with chunk_size=128 or unset {_KDA_V2_FLAG}"
        )


def _kda_v2_chunks(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    *,
    chunk_size: int,
    use_qk_l2norm: bool,
) -> tuple[torch.Tensor, ...]:
    """The V1 prologue: FP32 [B,H,C,chunk,*] q (scaled), k, in-chunk g cumsum, k*beta, v*beta."""

    q, k, v, beta, g = [
        x.transpose(1, 2).contiguous().float() for x in (q, k, v, beta, g)
    ]
    if use_qk_l2norm:
        q, k = _l2norm(q), _l2norm(k)
    pad = (chunk_size - k.shape[2] % chunk_size) % chunk_size
    q = F.pad(q, (0, 0, 0, pad)) * (q.shape[-1] ** -0.5)
    k, v, g = (F.pad(x, (0, 0, 0, pad)) for x in (k, v, g))
    beta = F.pad(beta, (0, pad))
    v_beta = v * beta.unsqueeze(-1)
    k_beta = k * beta.unsqueeze(-1)

    def chunks(x: torch.Tensor) -> torch.Tensor:
        return x.reshape(x.shape[0], x.shape[1], -1, chunk_size, x.shape[-1])

    q, k, g, k_beta, v_beta = [chunks(x) for x in (q, k, g, k_beta, v_beta)]
    return q, k, g.cumsum(dim=-2), k_beta, v_beta


def _kda_v2_pair_matrices(
    q: torch.Tensor, k: torch.Tensor, k_beta: torch.Tensor, g: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """attn = -strict_lower(k_beta_i.k_j decay_ij), intra = lower(q_i.k_j decay_ij); [B,H,C,c,c]."""

    chunk_size = q.shape[-2]
    if "decay_dot" in _KDA_V2_FOR:
        def flat(x: torch.Tensor) -> torch.Tensor:
            return x.reshape(-1, chunk_size, x.shape[-1])

        pair_shape = (*q.shape[:-1], chunk_size)
        key_dots = _decay_dot(flat(k_beta), flat(k), flat(g), flat(g)).reshape(pair_shape)
        query_dots = _decay_dot(flat(q), flat(k), flat(g), flat(g)).reshape(pair_shape)
    else:
        decay = (g.unsqueeze(-2) - g.unsqueeze(-3)).exp()
        key_dots = (k_beta.unsqueeze(-2) * k.unsqueeze(-3) * decay).sum(dim=-1)
        query_dots = (q.unsqueeze(-2) * k.unsqueeze(-3) * decay).sum(dim=-1)
    ones = torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=q.device)
    attn = -key_dots.masked_fill(torch.triu(ones), 0)
    intra = query_dots.masked_fill(torch.triu(ones, diagonal=1), 0)
    return attn, intra


def _kda_v2_diagonal_inverses(lmat: torch.Tensor, block_size: int) -> torch.Tensor:
    """Forward-substitution inverses of every 16x16 diagonal block of unit-lower ``lmat``."""

    block_count = lmat.shape[-1] // block_size
    diagonal_blocks = torch.stack([
        lmat[:, block_idx * block_size:(block_idx + 1) * block_size,
             block_idx * block_size:(block_idx + 1) * block_size]
        for block_idx in range(block_count)
    ], dim=1)
    base = diagonal_blocks.reshape(-1, block_size, block_size)
    inverse = torch.zeros_like(base)
    inverse[:, 0, 0] = 1
    for row_idx in range(1, block_size):
        correction = torch.bmm(
            base[:, row_idx:row_idx + 1, :row_idx],
            inverse[:, :row_idx, :],
        )
        inverse[:, row_idx:row_idx + 1, :] = -correction
        inverse[:, row_idx, row_idx] = 1
    return inverse.reshape(lmat.shape[0], block_count, block_size, block_size)


def _kda_v2_block_inverse(attn: torch.Tensor) -> torch.Tensor:
    """Our exact block inverse of L = I - attn (V1's HPU_GLM53_KDA_BLOCK_INVERSE_PREFILL algebra).

    inv([[A,0],[C,D]]) = [[A^-1,0],[-D^-1 C A^-1,D^-1]], merged pairwise from 16x16 blocks.
    """

    chunk_size = attn.shape[-1]
    lmat = -attn.reshape(-1, chunk_size, chunk_size)
    lmat.diagonal(dim1=-2, dim2=-1).fill_(1)
    matrix_count, block_size = lmat.shape[0], _DECAY_BLOCK
    inverse_blocks = _kda_v2_diagonal_inverses(lmat, block_size)
    block_count = chunk_size // block_size
    while block_count > 1:
        pair_count = block_count // 2
        a_inv = inverse_blocks[:, 0::2]
        d_inv = inverse_blocks[:, 1::2]
        c_blocks = torch.stack([
            lmat[:, (2 * pair_idx + 1) * block_size:(2 * pair_idx + 2) * block_size,
                 (2 * pair_idx) * block_size:(2 * pair_idx + 1) * block_size]
            for pair_idx in range(pair_count)
        ], dim=1)
        d_flat = d_inv.reshape(-1, block_size, block_size)
        c_flat = c_blocks.reshape(-1, block_size, block_size)
        a_flat = a_inv.reshape(-1, block_size, block_size)
        lower_left = -torch.bmm(torch.bmm(d_flat, c_flat), a_flat).reshape(
            matrix_count, pair_count, block_size, block_size
        )
        top = torch.cat((a_inv, torch.zeros_like(a_inv)), dim=-1)
        bottom = torch.cat((lower_left, d_inv), dim=-1)
        inverse_blocks = torch.cat((top, bottom), dim=-2)
        block_count = pair_count
        block_size *= 2
    return inverse_blocks[:, 0].reshape_as(attn)


class _KdaV2Chunks(NamedTuple):
    """FP32 [B,H,C,chunk,*] chunk algebra both V2 recurrences consume."""

    q: torch.Tensor  # l2-normed, scaled queries
    k: torch.Tensor  # l2-normed keys
    g: torch.Tensor  # in-chunk cumulative log decay
    value: torch.Tensor  # u = T @ (v * beta)
    k_cumdecay: torch.Tensor  # w = T @ (k * beta * e^g)
    intra: torch.Tensor  # lower(q_i . k_j decay_ij), diagonal included


def _kda_v2_decay_factors(
    chunks: _KdaV2Chunks,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """qe = q e^g, kk^T = (k e^{g_last - g})^T and alpha = e^{g_last}, for every chunk at once.

    state' = state*e^{g_last} + (k*delta)^T @ (u - w @ state): delta rides the
    OUTER k factor, so it is folded once into kk = k * delta.
    """

    gs_last = chunks.g[..., -1:, :]  # [B,H,C,1,K]
    qe = chunks.q * chunks.g.exp()  # [B,H,C,c,K]
    kk_t = (chunks.k * (gs_last - chunks.g).exp()).transpose(-1, -2)  # [B,H,C,K,c]
    return qe, kk_t, gs_last.exp().squeeze(-2)  # alpha [B,H,C,K]


def _kda_v2_serial_recurrence(
    chunks: _KdaV2Chunks, state: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """V1's per-chunk order over precomputed intra and decay factors ([B,H,K,V] official state)."""

    qe, kk_t, alpha = _kda_v2_decay_factors(chunks)
    out = torch.zeros_like(chunks.value)
    for chunk_idx in range(out.shape[2]):
        v_new = chunks.value[:, :, chunk_idx] - chunks.k_cumdecay[:, :, chunk_idx] @ state
        inter = qe[:, :, chunk_idx] @ state
        out[:, :, chunk_idx] = inter + chunks.intra[:, :, chunk_idx] @ v_new
        state = state * alpha[:, :, chunk_idx].unsqueeze(-1) + kk_t[:, :, chunk_idx] @ v_new
    return out, state


def _kda_v2_phase_b(
    chunks: _KdaV2Chunks,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Fork Phase B precompute for every chunk in batched matmuls: core, c_h, n_t = kk^T u, r = kk^T w, alpha."""

    qe, kk_t, alpha = _kda_v2_decay_factors(chunks)
    core = torch.matmul(chunks.intra, chunks.value)  # [B,H,C,c,V]
    c_h = qe - torch.matmul(chunks.intra, chunks.k_cumdecay)  # [B,H,C,c,K]
    n_t = torch.matmul(kk_t, chunks.value)  # [B,H,C,K,V]
    r = torch.matmul(kk_t, chunks.k_cumdecay)  # [B,H,C,K,K]
    return core, c_h, n_t, r, alpha


def _kda_v2_batched_recurrence(
    chunks: _KdaV2Chunks, state: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fork Phase B: the batched precompute, then only [K,K]@[K,V] steps stay serial.

    The step is state*alpha + (n_t - r @ state), the fork's m_full = diag(alpha) - r
    applied without the identity matmul: an all-zero pad chunk (alpha 1, r 0, n_t 0)
    leaves the state bit-exact, as V1's update does (tail-fuse padding invariance).
    """

    core, c_h, n_t, r, alpha = _kda_v2_phase_b(chunks)
    out = torch.empty_like(core)
    for ci in range(out.shape[2]):
        out[:, :, ci] = core[:, :, ci] + torch.matmul(c_h[:, :, ci], state)
        state = state * alpha[:, :, ci].unsqueeze(-1) + (n_t[:, :, ci] - torch.matmul(r[:, :, ci], state))
    return out, state


# parallel_scan is adapted from the same fork file, _kda_prefix_states :34-81 and the
# VLLM_KDA_SCAN=parallel branch :265-268 (Apache-2.0, header above).  Changes: chunk axis 2
# ([B,H,C,...]), no identity prefix and no stride-0 operand (the fork notes that expand
# feeding a batched matmul miscompiles on the lazy path): the entry states are one
# [C*K, K] @ [K, V] matmul per head, with chunk 0's entry state the input state itself.

def _kda_v2_compose_level(
    maps: torch.Tensor, adds: torch.Tensor, span: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """One Hillis-Steele level: chunk i (i >= span) absorbs the operator ending at i - span."""

    later = maps[:, :, span:]
    adds = torch.cat((adds[:, :, :span], torch.matmul(later, adds[:, :, :-span]) + adds[:, :, span:]), dim=2)
    maps = torch.cat((maps[:, :, :span], torch.matmul(later, maps[:, :, :-span])), dim=2)
    return maps, adds


def _kda_v2_prefix_states(
    maps: torch.Tensor, adds: torch.Tensor, state: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Entry state of every chunk and the final state of state_i = maps_i @ state_{i-1} + adds_i.

    maps [B,H,C,K,K], adds [B,H,C,K,V], state [B,H,K,V]; ceil(log2 C) batched levels replace C
    dependent steps.  Chunk i's prefix is the same tree for every C > i, so entry states (and the
    outputs) ignore trailing zero chunks bit for bit; the final state does not (tree order).
    """

    batch, heads, chunk_count, key_dim, _ = maps.shape
    span = 1
    while span < chunk_count:
        maps, adds = _kda_v2_compose_level(maps, adds, span)
        span *= 2
    stacked = maps.reshape(batch, heads, chunk_count * key_dim, key_dim)
    after = torch.matmul(stacked, state).reshape(adds.shape) + adds  # [B,H,C,K,V]
    entry = torch.cat((state.unsqueeze(2), after[:, :, :-1]), dim=2)
    return entry, after[:, :, -1]


def _kda_v2_scan_recurrence(
    chunks: _KdaV2Chunks, state: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """parallel_scan: Phase B with the serial chunk loop replaced by the prefix scan (fp32, tree order)."""

    core, c_h, n_t, r, alpha = _kda_v2_phase_b(chunks)
    eye = torch.eye(r.shape[-1], dtype=r.dtype, device=r.device)
    entry, final = _kda_v2_prefix_states(alpha.unsqueeze(-1) * eye - r, n_t, state)
    return core + torch.matmul(c_h, entry), final


def _kda_v2_chunk(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    *,
    chunk_size: int = 64,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = True,
    use_qk_l2norm: bool = True,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Drop-in for ``_r589_original_chunk_kda`` (same layouts, FP32 state) with the S6 verbs."""

    _require_kda_v2_contract(chunk_size)
    input_dtype, seq_len = q.dtype, q.shape[1]
    q, k, g, k_beta, v_beta = _kda_v2_chunks(
        q, k, v, g, beta, chunk_size=chunk_size, use_qk_l2norm=use_qk_l2norm
    )
    attn, intra = _kda_v2_pair_matrices(q, k, k_beta, g)
    inverse = _kda_v2_block_inverse(attn)
    chunks = _KdaV2Chunks(q, k, g, inverse @ v_beta, inverse @ (k_beta * g.exp()), intra)
    batch, heads, chunk_count, _, key_dim = k.shape
    value_dim = chunks.value.shape[-1]
    if initial_state is None:
        state = torch.zeros(batch, heads, key_dim, value_dim, dtype=torch.float32, device=q.device)
    else:
        state = initial_state.float().transpose(-1, -2).contiguous()
    if "parallel_scan" in _KDA_V2_FOR:
        out, state = _kda_v2_scan_recurrence(chunks, state)
    elif "batched_recurrence" in _KDA_V2_FOR:
        out, state = _kda_v2_batched_recurrence(chunks, state)
    else:
        out, state = _kda_v2_serial_recurrence(chunks, state)
    out = out.reshape(batch, heads, chunk_count * chunk_size, value_dim)[:, :, :seq_len]
    out = out.transpose(1, 2).contiguous().to(input_dtype)
    final = state.transpose(-1, -2).contiguous() if output_final_state else None
    return out, final


def _kda_v2_original() -> Callable[..., tuple[torch.Tensor, torch.Tensor | None]]:
    """The chunk body the verbs run: V1's own for exact, else the r3 ``_kda_v2_chunk``."""

    return _r589_original_chunk_kda if "exact" in _KDA_V2_FOR else _kda_v2_chunk


def _kda_v2_tiles(tensors: tuple[torch.Tensor, ...], options: dict) -> tuple[torch.Tensor, torch.Tensor | None]:
    """R589's tile route (kda_bounded / kda_batch_bounded, tile graphs when on) around ``_kda_v2_original()``."""

    from functools import partial
    from glm53_kda_batch_prefill import enabled as batch_enabled, kda_batch_bounded
    from glm53_kda_tile_graph import enabled as tile_enabled, tile
    from glm53_r589_bounded_prefill import kda_bounded
    original = _kda_v2_original()
    kernel = partial(tile, original) if tile_enabled() else original
    bounded = kda_batch_bounded if tensors[0].shape[0] > 1 and batch_enabled() else kda_bounded
    return bounded(kernel, *tensors, **options)


def _kda_v2_bounded(tensors: tuple[torch.Tensor, ...], options: dict) -> tuple[torch.Tensor, torch.Tensor | None]:
    """V1's dispatch with ``_kda_v2_original()`` under kda_bounded / tile (r3's route; V1's own for exact).

    ``tensors`` is (q, k, v, g, beta); ``options`` is hpu_chunk_kda's keyword set.
    """

    from glm53_r589_bounded_prefill import enabled
    q = tensors[0]
    if not (enabled() and q.shape[0] * q.shape[1] > 512):
        return _kda_v2_original()(*tensors, **options)
    return _kda_v2_tiles(tensors, options)


def _kda_v2_remainder(rest: tuple[torch.Tensor, ...], options: dict) -> tuple[torch.Tensor, torch.Tensor | None]:
    """The tail after the windows as production runs it: on R589's tiles at any length (the tail of a tiled
    call), or as one plain call when R589 is off."""

    from glm53_r589_bounded_prefill import enabled
    if enabled():
        return _kda_v2_tiles(rest, options)
    return _kda_v2_original()(*rest, **options)


# exact (C534).  r3's reassociations keep V1's accuracy against the fp64 oracle but flip ~0.01% of the bf16
# KDA outputs per layer, which fails the strict gate end to end.  exact runs V1's own body on R589's own tile
# plan (the same piece shapes, the same state hand-off), so its math IS production's; what changes is the host
# side: inside a capture the plan runs as plain tensor ops (no mark_step / synchronize / read-back), and eager
# 2048-token windows replay one cached graph instead of four tile graphs with a drain each.  exact_chunks runs
# the plan one 128-token chunk per call: V1's pair decay at 128 MiB instead of 512 MiB per call (16 heads).


def _kda_exact_layout(batch: int) -> tuple[int, bool]:
    """(tokens per V1 call, grouped) as production would tile ``batch`` rows: 512 (one chunk for exact_chunks),
    grouped exactly when kda_batch_bounded would take the call."""

    from glm53_kda_batch_prefill import enabled as batch_enabled
    from glm53_r589_bounded_prefill import TILE
    return (0 if "exact_chunks" in _KDA_V2_FOR else TILE), batch > 1 and batch_enabled()


def _kda_exact_plan(batch: int, length: int, chunk_size: int, layout: tuple[int, bool]) -> list[tuple[int, ...]]:
    """kda_bounded / kda_batch_bounded's pieces, in their order: (row_start, row_end, start, end).

    ``layout`` is (tile tokens, grouped); 0 tile tokens means one chunk.  Grouped rows share a tile of
    tile // rows // chunk chunks (kda_batch_bounded); otherwise each row runs alone (kda_bounded).
    """

    tile_tokens, grouped = layout[0] or chunk_size, layout[1]
    group_cap = max(1, tile_tokens // chunk_size) if grouped else 1
    plan = []
    for row_start in range(0, batch, group_cap):
        row_end = min(row_start + group_cap, batch)
        group_tile = (tile_tokens // (row_end - row_start) // chunk_size) * chunk_size
        plan += [(row_start, row_end, start, min(start + group_tile, length)) for start in range(0, length, group_tile)]
    return plan


def _kda_v2_entry_state(tensors: tuple[torch.Tensor, ...], initial: torch.Tensor | None) -> torch.Tensor:
    """The FP32 [B, H, V, K] state a call starts from: ``initial`` as FP32, or zeros (kda_bounded's rule)."""

    q, v = tensors[0], tensors[2]
    if initial is not None:
        return initial.float()
    return torch.zeros(q.shape[0], q.shape[2], v.shape[-1], q.shape[-1], dtype=torch.float32, device=q.device)


def _kda_exact_tiles(
    tensors: tuple[torch.Tensor, ...], options: dict, layout: tuple[int, bool]
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """V1's body over ``_kda_exact_plan`` as plain tensor ops: capture-legal, production's values bit for bit.

    ``tensors`` is (q, k, v, g, beta) [B, L, H, *]; ``options`` is hpu_chunk_kda's keyword set.
    """

    q, v = tensors[0], tensors[2]
    batch, length, heads = q.shape[:3]
    entry = _kda_v2_entry_state(tensors, options["initial_state"])
    output = torch.empty((batch, length, heads, v.shape[-1]), dtype=q.dtype, device=q.device)
    states: dict[int, torch.Tensor] = {}
    for row_start, row_end, start, end in _kda_exact_plan(batch, length, options["chunk_size"], layout):
        state = states.get(row_start, entry[row_start:row_end])
        pieces = [t[row_start:row_end, start:end] for t in tensors]
        core, states[row_start] = _r589_original_chunk_kda(
            *pieces, chunk_size=options["chunk_size"], initial_state=state, output_final_state=True,
            use_qk_l2norm=options["use_qk_l2norm"])
        output[row_start:row_end, start:end].copy_(core)
    final = torch.cat([states[row] for row in sorted(states)]) if options["output_final_state"] else None
    return output, final


def _kda_exact(tensors: tuple[torch.Tensor, ...], options: dict) -> tuple[torch.Tensor, torch.Tensor | None]:
    """exact's routes: production's V1 math everywhere, graphs where production has host-synced tiles."""

    from glm53_r589_bounded_prefill import enabled
    q = tensors[0]
    if not (enabled() and q.shape[0] * q.shape[1] > 512):
        return _r589_original_chunk_kda(*tensors, **options)
    if _kda_v2_capturing(q.device):
        _kda_v2_route("exact_raw")
        return _kda_exact_tiles(tensors, options, _kda_exact_layout(q.shape[0]))
    if q.shape[0] == 1 and q.shape[1] >= _KDA_V2_WINDOW:
        return _kda_v2_windows(tensors, options)
    return _kda_v2_tiles(tensors, options)


# whole_call (w-kdapf).  With decay_dot the pair terms grow linearly in tokens (r3's _decay_dot
# keeps [rows, 8, 128, 128]-class blocks, ~32 MiB per 512 tokens at 16 heads, where V1 builds a
# 512 MiB [tile, 128, 128, 128] decay), so V2 no longer needs R589's 512-token tiles:
#   - inside a graph capture: one raw _kda_v2_chunk over the whole call; nothing on that route
#     marks a step, syncs, reads a value back or nests a graph, so prompt / mixed captures stay legal;
#   - eager, B == 1: full 2048-token windows, each a replay of its own cached HPU graph, and the
#     remainder through the r3 route (its proven R589 tiles and tail shapes);
#   - every other call: the r3 route, unchanged.
# exact shares the windows (each window graph holds V1's four 512-token tiles) and the remainder rule.
_KDA_V2_WINDOW = 2048  # the production long-prefill chunk (VLLM_ADAPTIVE_LONG_PREFILL_SINGLE_TOKENS)
_KDA_V2_WINDOW_GRAPHS = 2  # cached window graphs (q/k/v dtype variants); a new key past it runs eager
_KDA_V2_WINDOW_CACHE: dict[tuple, tuple] = {}
_KDA_V2_STATS = {"whole_raw": 0, "exact_raw": 0, "window_captures": 0, "window_replays": 0, "window_eager": 0, "remainders": 0}
_KDA_V2_ROUTES_LOGGED: set[str] = set()


def _kda_v2_route(name: str) -> None:
    """Count one whole_call route; log its first use per process (the arm's evidence that the verb engaged)."""

    _KDA_V2_STATS[name] += 1
    if name not in _KDA_V2_ROUTES_LOGGED:
        _KDA_V2_ROUTES_LOGGED.add(name)
        _LOG.warning("KDA_V2_WHOLE_CALL first route=%s pid=%d stats=%s", name, os.getpid(), _KDA_V2_STATS)


_KDA_V2_CAPTURE_DEPTH = threading.local()  # .value: this thread's open HPUGraph captures


def _kda_v2_count_captures(graph_class: type) -> None:
    """Wrap ``graph_class``'s capture_begin / capture_end to keep this thread's capture depth (once per class).

    wrap_in_hpu_graph (the model's prompt graphs) calls HPUGraph.capture_begin on the DEFAULT stream and sets no
    stream flag, so the depth is the only trace it leaves; torch.hpu.graph goes through the same two methods.
    """

    if getattr(graph_class, "_kda_v2_counted", False):
        return
    begin, end = graph_class.capture_begin, graph_class.capture_end

    def counted_begin(self: object, *args: object, **kwargs: object) -> object:
        result = begin(self, *args, **kwargs)
        _KDA_V2_CAPTURE_DEPTH.value = getattr(_KDA_V2_CAPTURE_DEPTH, "value", 0) + 1
        return result

    def counted_end(self: object, *args: object, **kwargs: object) -> object:
        _KDA_V2_CAPTURE_DEPTH.value = max(0, getattr(_KDA_V2_CAPTURE_DEPTH, "value", 0) - 1)
        return end(self, *args, **kwargs)

    graph_class.capture_begin, graph_class.capture_end = counted_begin, counted_end
    graph_class._kda_v2_counted = True


def _kda_v2_install_capture_count() -> None:
    """Count HPUGraph captures when a verb has a capture route (exact, whole_call); a no-op off Habana."""

    if not _KDA_V2_FOR & {"exact", "whole_call"}:
        return
    try:
        from habana_frameworks.torch.hpu.graphs import HPUGraph
    except ImportError:
        return
    _kda_v2_count_captures(HPUGraph)


def _kda_v2_capturing(device: torch.device) -> bool:
    """True only inside a real HPU graph capture (this call must not mark_step, sync or nest a graph).

    torch.hpu.graph sets the capture stream's is_capture; wrap_in_hpu_graph sets nothing, and is seen through the
    HPUGraph capture depth.  A non-default stream alone is NOT a capture: eager serving keeps the eager routes.
    """

    if device.type != "hpu":
        return False
    from habana_frameworks.torch import hpu
    return bool(hpu.is_current_stream_capturing()) or getattr(_KDA_V2_CAPTURE_DEPTH, "value", 0) > 0


_kda_v2_install_capture_count()


def _kda_v2_window_kernel(
    pieces: tuple[torch.Tensor, ...], state: torch.Tensor, chunk_size: int, use_qk_l2norm: bool
) -> tuple[torch.Tensor, torch.Tensor]:
    """One window's math: V1 on R589's tile plan for exact, else one _kda_v2_chunk over the window."""

    options = dict(chunk_size=chunk_size, initial_state=state, output_final_state=True, use_qk_l2norm=use_qk_l2norm)
    if "exact" in _KDA_V2_FOR:
        return _kda_exact_tiles(pieces, options, _kda_exact_layout(1))
    return _kda_v2_chunk(*pieces, **options)


def _kda_v2_window_eager(
    pieces: tuple[torch.Tensor, ...], state: torch.Tensor, chunk_size: int, use_qk_l2norm: bool
) -> tuple[torch.Tensor, torch.Tensor]:
    """One window as a plain call (CPU, and the HPU past the window-graph budget)."""

    return _kda_v2_window_kernel(pieces, state, chunk_size, use_qk_l2norm)


def _kda_v2_capture_window(sources: tuple[torch.Tensor, ...], chunk_size: int, use_qk_l2norm: bool) -> tuple:
    """(static buffers, graph, graph outputs) for one window shape; the capture also computes this call."""

    import habana_frameworks.torch.core as htcore

    buffers = tuple(t.contiguous().clone() for t in sources)
    htcore.mark_step(sync=True)
    torch.hpu.synchronize()
    graph = torch.hpu.HPUGraph()
    with torch.hpu.graph(graph, dry_run=False):
        outputs = _kda_v2_window_kernel(buffers[:5], buffers[5], chunk_size, use_qk_l2norm)
    htcore.mark_step(sync=True)
    torch.hpu.synchronize()
    return buffers, graph, outputs


def _kda_v2_window_graph(
    pieces: tuple[torch.Tensor, ...], state: torch.Tensor, chunk_size: int, use_qk_l2norm: bool
) -> tuple[torch.Tensor, torch.Tensor]:
    """One window as a replay of its cached HPU graph, with glm53_kda_tile_graph.tile's protocol: inputs copied
    into static buffers, mark_step before the replay, caller-owned clones of the results."""

    import habana_frameworks.torch.core as htcore

    sources = (*pieces, state)
    key = (_kda_v2_original().__name__, _kda_exact_layout(1), chunk_size, use_qk_l2norm, tuple((tuple(t.shape), t.dtype) for t in sources))
    entry = _KDA_V2_WINDOW_CACHE.get(key)
    if entry is None and len(_KDA_V2_WINDOW_CACHE) >= _KDA_V2_WINDOW_GRAPHS:
        _kda_v2_route("window_eager")
        return _kda_v2_window_eager(pieces, state, chunk_size, use_qk_l2norm)
    if entry is None:
        entry = _KDA_V2_WINDOW_CACHE[key] = _kda_v2_capture_window(sources, chunk_size, use_qk_l2norm)
        _kda_v2_route("window_captures")
    else:
        buffers, graph, _ = entry
        for target, source in zip(buffers, sources):
            target.copy_(source)
        htcore.mark_step(sync=True)
        graph.replay()
        _kda_v2_route("window_replays")
    output, final = entry[2]
    return output.clone(), final.clone()


def _kda_v2_window(
    pieces: tuple[torch.Tensor, ...], state: torch.Tensor, chunk_size: int, use_qk_l2norm: bool
) -> tuple[torch.Tensor, torch.Tensor]:
    """One full window: its cached graph on the HPU, a plain call elsewhere."""

    if state.device.type == "hpu":
        return _kda_v2_window_graph(pieces, state, chunk_size, use_qk_l2norm)
    return _kda_v2_window_eager(pieces, state, chunk_size, use_qk_l2norm)


def _kda_v2_windows(tensors: tuple[torch.Tensor, ...], options: dict) -> tuple[torch.Tensor, torch.Tensor | None]:
    """B == 1 and at least one window: full windows (graphs on the HPU), then the remainder as production tiles it.

    ``tensors`` is (q, k, v, g, beta) [1, L, H, *]; ``options`` is hpu_chunk_kda's keyword set.
    """

    length = tensors[0].shape[1]
    state = _kda_v2_entry_state(tensors, options["initial_state"])
    full = length - length % _KDA_V2_WINDOW
    outputs = []
    for start in range(0, full, _KDA_V2_WINDOW):
        pieces = tuple(t[:, start:start + _KDA_V2_WINDOW] for t in tensors)
        core, state = _kda_v2_window(pieces, state, options["chunk_size"], options["use_qk_l2norm"])
        outputs.append(core)
    if full < length:
        _kda_v2_route("remainders")
        rest = tuple(t[:, full:] for t in tensors)
        core, state = _kda_v2_remainder(rest, {**options, "initial_state": state, "output_final_state": True})
        outputs.append(core)
    out = outputs[0] if len(outputs) == 1 else torch.cat(outputs, dim=1)
    return out, state if options["output_final_state"] else None


def _hpu_chunk_kda_v2(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    *,
    chunk_size: int,
    initial_state: torch.Tensor | None,
    output_final_state: bool,
    use_qk_l2norm: bool,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """The V2 entry: exact's routes; whole_call's for the calls R589 would tile (> 512 tokens); else r3's."""

    kwargs = dict(chunk_size=chunk_size, initial_state=initial_state,
                  output_final_state=output_final_state, use_qk_l2norm=use_qk_l2norm)
    if "exact" in _KDA_V2_FOR:
        return _kda_exact((q, k, v, g, beta), kwargs)
    if "whole_call" in _KDA_V2_FOR and q.shape[0] * q.shape[1] > 512:
        if _kda_v2_capturing(q.device):
            _kda_v2_route("whole_raw")
            return _kda_v2_chunk(q, k, v, g, beta, **kwargs)
        if q.shape[0] == 1 and q.shape[1] >= _KDA_V2_WINDOW:
            return _kda_v2_windows((q, k, v, g, beta), kwargs)
    return _kda_v2_bounded((q, k, v, g, beta), kwargs)


def hpu_chunk_kda(q, k, v, g, beta, *, chunk_size=64, initial_state=None,
                  output_final_state=True, use_qk_l2norm=True):
    if _KDA_V2_FOR:  # HPU_GLM53_KDA_V2_FOR; unset = the untouched V1 body below
        return _hpu_chunk_kda_v2(q, k, v, g, beta, chunk_size=chunk_size,
            initial_state=initial_state, output_final_state=output_final_state,
            use_qk_l2norm=use_qk_l2norm)
    # R589_BOUNDED_PREFILL_V1: decode and <=512-token prefills stay untouched.
    from glm53_r589_bounded_prefill import enabled, kda_bounded
    from glm53_kda_batch_prefill import enabled as batch_enabled, kda_batch_bounded
    if enabled() and q.shape[0] * q.shape[1] > 512:
        from functools import partial
        from glm53_kda_tile_graph import enabled as tile_enabled, tile
        kernel = partial(tile, _r589_original_chunk_kda) if tile_enabled() else _r589_original_chunk_kda
        bounded = kda_batch_bounded if q.shape[0] > 1 and batch_enabled() else kda_bounded
        return bounded(kernel, q, k, v, g, beta,
            chunk_size=chunk_size, initial_state=initial_state,
            output_final_state=output_final_state, use_qk_l2norm=use_qk_l2norm)
    return _r589_original_chunk_kda(q, k, v, g, beta,
        chunk_size=chunk_size, initial_state=initial_state,
        output_final_state=output_final_state, use_qk_l2norm=use_qk_l2norm)


class HPUGlm5NextRMSNormGated(nn.Module):
    """Portable FP32-accumulating RMSNorm + sigmoid gate for HPU graphs."""

    def __init__(self, hidden_size: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
        dtype = hidden_states.dtype
        x = hidden_states.float()
        variance = x.square().mean(dim=-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.variance_epsilon)
        x = x * self.weight.float()
        x = x * torch.sigmoid(gate.float())
        return x.to(dtype)
