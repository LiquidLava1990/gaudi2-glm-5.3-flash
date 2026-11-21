"""Opt-in lazy mixed GLM step; regional kernels retain existing metadata.

Install this module beside the overlay's glm53_r589_bounded_prefill.py.
Only real tokens traverse the model's token-wise operations. Padding is local
to each attention/state kernel and is discarded before returning to the layer.
"""
from __future__ import annotations

from contextlib import contextmanager, nullcontext
from collections import Counter
from dataclasses import dataclass
import os
from typing import Any, Callable, ContextManager, Iterator

import torch
from mixed_recurrent import _check_indices


def enabled() -> bool:
    return os.getenv('VLLM_GAUDI_MIXED_STEP', '0') == '1'


NATIVE_NO_DECODE = 'VLLM_GAUDI_MIXED_NATIVE_NO_DECODE'


def native_no_decode() -> bool:
    """C527 fix, opt-in: a live prompt-only step has nothing to mix, so it runs the native forward."""
    value = os.getenv(NATIVE_NO_DECODE, '0')
    if value not in ('0', '1'):
        raise ValueError(f'{NATIVE_NO_DECODE}={value!r} must be 0 or 1; unset it (or 0) to keep the lazy '
                         'mixed fallback for prompt-only steps')
    return value == '1'


def count_native_step(runner: object) -> None:
    """Count native prompt-only steps on the 'GLM53_MIXED_STEP ' line the gate kit keeps (rowpad-c1.log)."""
    steps = getattr(runner, '_glm53_native_no_decode_steps', 0) + 1
    runner._glm53_native_no_decode_steps = steps
    from glm53_mixed_graph import positive_env
    if steps == 1 or steps % positive_env('VLLM_GAUDI_MIXED_GRAPH_LOG_EVERY', 100) == 0:
        import logging
        logging.getLogger(__name__).warning('GLM53_MIXED_STEP native_no_decode steps=%d', steps)


def prefill_budget() -> int:
    name = 'VLLM_GAUDI_MIXED_PREFILL_TOKENS'
    try:
        value = int(os.getenv(name, '512'))
    except ValueError as exc:
        raise ValueError(f'{name} must be a positive integer') from exc
    if value < 1:
        raise ValueError(f'{name} must be a positive integer')
    return value


def single_prompt_remaining(owner: Any, remaining: int, scheduled: dict[str, int],
                            prompts: set[str], *, from_waiting: bool = False) -> int:
    """Close a mixed step's prompt pool after its first committed prompt.

    Use scheduled request IDs, not running membership: parked/stale decodes
    must not reduce prompt throughput. Called only after allocation succeeds.
    """
    if (os.getenv('VLLM_GAUDI_MIXED_GRAPH_SINGLE_PROMPT', '0') != '1'
            or os.getenv('VLLM_GAUDI_MIXED_GRAPHS', '0') != '1'):
        return remaining
    counts = getattr(owner, '_glm53_single_prompt_counts', None)
    if counts is None:
        counts = Counter(dict.fromkeys(('no_decode', 'budget_full', 'running_tail',
                                        'waiting_tail', 'withheld_tokens', 'events'), 0))
        setattr(owner, '_glm53_single_prompt_counts', counts)
    has_decode = any(req not in prompts for req in scheduled)
    reason = ('no_decode' if not has_decode else 'budget_full' if remaining <= 0
              else 'waiting_tail' if from_waiting else 'running_tail')
    counts[reason] += 1
    counts['events'] += 1
    if has_decode:
        counts['withheld_tokens'] += max(0, remaining)
    if counts['events'] == 1 or counts['events'] % 100 == 0:
        import logging
        logging.getLogger(__name__).info('mixed_single_prompt counts=%s', dict(counts))
    return 0 if has_decode else remaining


@dataclass(frozen=True)
class Region:
    requests: tuple[int, ...]
    lengths: tuple[int, ...]
    starts: tuple[int, ...]
    width: int
    batch: int
    valid: tuple[int, ...]
    metadata: Any
    decode: bool
    graph_padded: bool = False

    @property
    def count(self) -> int:
        return self.batch * self.width if self.graph_padded else sum(self.lengths)


@dataclass(frozen=True)
class MixedMetadata:
    regions: tuple[Region, ...]
    is_prompt: bool = False
    mixed_step: bool = True

    def __post_init__(self) -> None:
        requests = tuple(req for r in self.regions for req in r.requests)
        if len(set(requests)) != len(requests):
            raise ValueError('mixed step has duplicate request/state ownership')
        prompt_seen = False
        for r in self.regions:
            if r.decode and prompt_seen:
                raise ValueError('mixed layout requires decode rows before prompt rows')
            prompt_seen |= not r.decode
            if not (len(r.requests) == len(r.lengths) == len(r.starts)):
                raise ValueError('mixed region requires lengths/positions per request')
            if r.width < 1 or r.batch < len(r.requests):
                raise ValueError('mixed region bucket cannot hold its requests')
            if any(n < 1 or n > r.width for n in r.lengths):
                raise ValueError('mixed region length outside prepared bucket')
            if r.decode and any(n != 1 for n in r.lengths):
                raise ValueError('mixed v1 needs one decode token per request; disable MTP')
            expected = tuple(i*r.width+j for i,n in enumerate(r.lengths) for j in range(n))
            _check_indices(r.valid, r.batch*r.width, 'mixed kernel row')
            if r.valid != expected:
                raise ValueError('mixed region valid rows must preserve request/token order')

    @property
    def num_decode_tokens(self) -> int:
        return sum(r.count for r in self.regions if r.decode)

    @property
    def prefill_slices(self) -> list[tuple[int, int, int]]:
        return [(req, start, n) for r in self.regions if not r.decode
                for req, start, n in zip(r.requests, r.starts, r.lengths)]

    @property
    def request_indices(self) -> tuple[int, ...]:
        return tuple(req for r in self.regions for req, n in zip(r.requests, r.lengths)
                     for _ in range(n))

    @property
    def positions(self) -> tuple[int, ...]:
        return tuple(start+j for r in self.regions for start,n in zip(r.starts,r.lengths)
                     for j in range(n))


def sample_rows(plan: MixedMetadata, prompt_ends: dict[int, int]) -> tuple[int, ...]:
    rows, cursor = [], 0
    for r in plan.regions:
        for req, start, n in zip(r.requests, r.starts, r.lengths):
            if r.decode or start + n == prompt_ends[req]:
                rows.append(cursor + n - 1)
            cursor += n
    return tuple(rows)


def _indices(rows: tuple[int, ...], tensor: torch.Tensor) -> torch.Tensor:
    return torch.tensor(rows, dtype=torch.long, device=tensor.device)


def pack_region(x: torch.Tensor, r: Region, fill: int = 0) -> torch.Tensor:
    if r.graph_padded:
        return x
    padded = x.new_full((r.batch*r.width,) + x.shape[1:], fill)
    return padded.index_copy(0, _indices(r.valid, x), x)


def unpack_region(x: torch.Tensor, r: Region) -> torch.Tensor:
    if r.graph_padded:
        return x
    return x.index_select(0, _indices(r.valid, x))


def run_regions(x: torch.Tensor, plan: MixedMetadata,
                kernel: Callable[[torch.Tensor, Region], torch.Tensor]) -> torch.Tensor:
    parts, cursor = [], 0
    for r in plan.regions:
        parts.append(unpack_region(kernel(pack_region(x[cursor:cursor+r.count], r), r), r))
        cursor += r.count
    return torch.cat(parts)


@contextmanager
def region_context(metadata: Any) -> Iterator[None]:
    from vllm.forward_context import get_forward_context
    context = get_forward_context()
    previous = context.attn_metadata
    context.attn_metadata = metadata
    try:
        yield
    finally:
        context.attn_metadata = previous


def regional_indexer(owner: Any, plan: MixedMetadata, hidden: torch.Tensor,
                     q: torch.Tensor, k: torch.Tensor, weights: torch.Tensor,
                     gate: torch.Tensor, ape: torch.Tensor,
                     positions: torch.Tensor) -> torch.Tensor:
    parts, cursor = [], 0
    for r in plan.regions:
        span = slice(cursor, cursor+r.count)
        values = [pack_region(t[span], r) for t in (hidden, q, k, weights, gate, positions)]
        with region_context(r.metadata):
            result = owner.forward(*values[:4], gate_score=values[4],
                                   compress_ape=ape, positions=values[5])
        # The legacy indexer owns a shared scratch buffer. Keep segment storage
        # independent before the next region overwrites that scratch.
        parts.append(unpack_region(result, r).clone())
        cursor += r.count
    return torch.cat(parts)


def regional_mla(owner: Any, plan: MixedMetadata, q: torch.Tensor,
                 k: torch.Tensor, pe: torch.Tensor, cache: Any,
                 slots: torch.Tensor | None) -> torch.Tensor:
    if slots is None:
        raise RuntimeError('mixed sparse MLA requires explicit indexer slots; enable direct MLA')
    parts, cursor = [], 0
    for r in plan.regions:
        span = slice(cursor, cursor+r.count)
        values = [pack_region(t[span], r) for t in (q, k, pe)]
        region_slots = pack_region(slots[span], r, fill=-1)
        with region_context(r.metadata):
            result = owner.forward_impl(*values, cache, r.metadata,
                                        hpu_sparse_slots=region_slots)
        parts.append(unpack_region(result, r))
        cursor += r.count
    return torch.cat(parts)


def regional_recurrent(owner: Any, plan: MixedMetadata, qkv: torch.Tensor,
                       gate: torch.Tensor, beta: torch.Tensor,
                       output: torch.Tensor) -> None:
    cursor = 0
    for r in plan.regions:
        span = slice(cursor, cursor+r.count)
        local_qkv = pack_region(qkv[span], r)
        local_gate = pack_region(gate[0, span], r).unsqueeze(0)
        local_beta = pack_region(beta[0, span], r).unsqueeze(0)
        local_output = output.new_zeros((1, r.batch*r.width) + output.shape[2:])
        # Keep original per-cache-group load/state indices. Never substitute
        # packed row numbers for physical recurrent/conv state slots.
        with region_context(r.metadata):
            owner._forward_hpu(local_qkv, local_gate, local_beta, local_output)
        if r.graph_padded:
            rows = _indices(tuple(range(cursor, cursor+r.count)), output)
            output.index_copy_(1, rows, local_output)
        else:
            output[0, span].copy_(unpack_region(local_output[0], r))
        cursor += r.count


@dataclass
class PreparedBatch:
    region: Region
    tokens: torch.Tensor
    positions: torch.Tensor
    logits: tuple[int, ...]
    key: int


def _batch(runner: Any, ids: list[str], tokens: torch.Tensor, positions: torch.Tensor,
           metadata: Any, logits: tuple[int, ...], schedule: Any, decode: bool,
           trim: Callable | None = None) -> PreparedBatch:
    indices = tuple(runner.input_batch.req_id_to_index[req] for req in ids)
    lengths = tuple(schedule.num_scheduled_tokens[req] for req in ids)
    starts = tuple(int(runner.input_batch.num_computed_tokens_cpu[i]) for i in indices)
    batch, width = tokens.shape
    valid = tuple(i*width+j for i,n in enumerate(lengths) for j in range(n))
    processed = runner.model.metadata_processor.process_metadata(
        trim(metadata) if trim else metadata, batch, width, tokens.device, runner.dtype, False)
    region = Region(indices, lengths, starts, width, batch, valid, processed, decode)
    return PreparedBatch(region, tokens, positions, logits, id(metadata))


def _host_logits(ids: list[str], requests: list[str], lengths: tuple[int, ...],
                 width: int, padded_count: int) -> tuple[int, ...]:
    """Rebuild ship-v3's tail logits using CPU request labels, never HPU values."""
    counts = Counter(requests)
    offsets = {req: i*width+n-counts[req] for i, (req, n) in enumerate(zip(ids, lengths))}
    if padded_count < len(requests) or any(counts[req] > n for req, n in zip(ids, lengths)):
        raise ValueError('mixed logits exceed prepared rows; inspect prefill logits_requests')
    rows = []
    for req in requests:
        if req not in offsets:
            raise ValueError(f'mixed logits request {req!r} missing from prefill batch')
        rows.append(offsets[req])
        offsets[req] += 1
    return tuple(rows) + (-1,) * (padded_count-len(rows))


def _batches(runner: Any, schedule: Any, pd: Any, prefills: Any, decode: Any,
             trim: Callable | None = None) -> list[PreparedBatch]:
    batches = []
    if pd.decode_req_ids:
        indices = tuple(range(decode.token_ids.numel()))
        batches.append(_batch(runner, pd.decode_req_ids, decode.token_ids, decode.position_ids,
                              decode.attn_metadata, indices, schedule, True, trim))
    for ids, tokens, positions, metadata, logits, requests in zip(
            prefills.request_ids, prefills.token_ids, prefills.position_ids,
            prefills.attn_metadata, prefills.logits_indices, prefills.logits_requests):
        lengths = tuple(schedule.num_scheduled_tokens[req] for req in ids)
        host = _host_logits(ids, requests, lengths, tokens.shape[1], logits.numel())
        batches.append(_batch(runner, ids, tokens, positions, metadata,
                              host, schedule, False, trim))
    return batches


def _validate_runner(runner: Any) -> None:
    if runner.vllm_config.speculative_config is not None:
        raise ValueError('VLLM_GAUDI_MIXED_STEP v1 requires MTP/speculation OFF')
    parallel = runner.vllm_config.parallel_config
    if parallel.data_parallel_size != 1 or parallel.pipeline_parallel_size != 1:
        raise ValueError('mixed v1 supports TP only; set DP=PP=1')
    if runner.lora_config or runner.use_aux_hidden_state_outputs or runner.use_merged_prefill:
        raise ValueError('mixed v1 needs LoRA/aux outputs/merged-prefill disabled')
    if runner.model_has_chunked_attention or runner.input_batch.num_prompt_logprobs:
        raise ValueError('mixed v1 needs chunked-attention/prompt-logprobs disabled')
    if not runner.model.flatten_input:
        raise ValueError('mixed v1 requires the flattened GLM model adapter')
    if runner.model.is_mm_optimized:
        raise ValueError('mixed v1 supports text-only GLM inputs')


def _sampling_map(batches: list[PreparedBatch]) -> tuple[tuple[int, ...], list[dict[int, int]]]:
    needed, maps, cursor = [], [], 0
    for batch in batches:
        mapping = {padded: cursor+i for i, padded in enumerate(batch.region.valid)}
        local = {}
        for i, row in enumerate(batch.logits):
            if row < 0:
                continue
            if row in mapping:
                local[i] = len(needed)
                needed.append(mapping[row])
            elif not batch.region.decode:
                raise ValueError('mixed prefill logit outside valid rows; inspect logits_requests')
        maps.append(local)
        cursor += batch.region.count
    return tuple(needed), maps


def _scatter_logits(logits: torch.Tensor, hidden: torch.Tensor, mapping: dict[int, int],
                    count: int) -> tuple[torch.Tensor, torch.Tensor]:
    out = logits.new_zeros((count,) + logits.shape[1:])
    samples = hidden.new_zeros((count,) + hidden.shape[1:])
    if mapping:
        dst = _indices(tuple(mapping), logits)
        src = _indices(tuple(mapping.values()), logits)
        out.index_copy_(0, dst, logits.index_select(0, src))
        samples.index_copy_(0, dst, hidden.index_select(0, src))
    return out, samples


def _results(batches: list[PreparedBatch], hidden: torch.Tensor, logits: torch.Tensor,
             sampled_hidden: torch.Tensor, maps: list[dict[int, int]]) -> dict[int, tuple]:
    results, cursor = {}, 0
    for batch, mapping in zip(batches, maps):
        r = batch.region
        full = pack_region(hidden[cursor:cursor+r.count], r)
        full = full.reshape(r.batch, r.width, -1)
        local_logits, local_samples = _scatter_logits(logits, sampled_hidden, mapping, len(batch.logits))
        results[batch.key] = (full, None, local_samples, local_logits)
        cursor += r.count
    return results


class Replay:
    def __init__(self, results: dict[int, tuple]) -> None:
        self.results = results

    def __call__(self, tokens: torch.Tensor, positions: torch.Tensor, metadata: Any,
                 *args: Any, **kwargs: Any) -> tuple:
        key = id(metadata)
        if key not in self.results:
            raise RuntimeError('mixed result missing/already consumed; inspect runner region dispatch')
        return self.results.pop(key)


def prepare(runner: Any, schedule: Any, pd: Any, prefills: Any, decode: Any,
            trim_metadata: Callable | None = None, *, warmup_mode: bool = False,
            warmup_context: ContextManager | None = None) -> Callable:
    if not enabled():
        return runner._execute_model_generic
    _validate_runner(runner)
    if not pd.prompt_req_ids:
        return runner._execute_model_generic
    # Warmup keeps the V1 route: boot recipes and the HBM left for the mixed-graph admission stay unchanged.
    if native_no_decode() and not pd.decode_req_ids and not warmup_mode:
        count_native_step(runner)
        return runner._execute_model_generic
    batches = _batches(runner, schedule, pd, prefills, decode, trim_metadata)
    plan = MixedMetadata(tuple(b.region for b in batches))
    if not warmup_mode and sum(n for _,_,n in plan.prefill_slices) > prefill_budget():
        raise ValueError('mixed prompt budget exceeded; install the paired scheduler patch')
    needed, maps = _sampling_map(batches)
    import glm53_mixed_graph
    hidden = glm53_mixed_graph.forward(runner, batches)
    if hidden is None and glm53_mixed_graph.exact_shapes_enabled() and not warmup_mode:
        # C576: no graph at the native shapes -> the reference's own per-batch path, never the lazy mixed forward.
        return runner._execute_model_generic
    if hidden is None:
        tokens = torch.cat([unpack_region(b.tokens.reshape(-1), b.region) for b in batches])
        positions = torch.cat([unpack_region(b.positions.reshape(-1), b.region) for b in batches])
        with warmup_context if warmup_context is not None else nullcontext():
            hidden = runner.model.forward(input_ids=tokens.unsqueeze(0), positions=positions.unsqueeze(0),
                                          attn_metadata=plan, bypass_hpu_graphs=True)
    hidden = hidden.reshape(-1, hidden.shape[-1])
    sampled_hidden = hidden.index_select(0, _indices(needed, hidden))
    logits = runner.model.compute_logits(sampled_hidden)
    return Replay(_results(batches, hidden, logits, sampled_hidden, maps))
