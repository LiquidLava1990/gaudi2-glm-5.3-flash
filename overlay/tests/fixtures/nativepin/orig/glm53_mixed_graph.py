"""Bounded mixed HPUGraphs. Sampling and metadata preparation stay outside capture.

Each entry owns all input tensor storage and the captured output. A capture
executes the model exactly once: never warm a stateful forward by running it
twice on live requests. Capture failure is fatal, since cache writes may have
already executed; it must never retry the same request through lazy forward.
"""
from __future__ import annotations

from collections import Counter
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import fields, is_dataclass, replace
import logging
import os
from typing import Any, Callable, Protocol

import torch

FLAG = 'VLLM_GAUDI_MIXED_GRAPHS'
LOG = logging.getLogger(__name__)
MIB = 1024 * 1024
_CAPTURE = ContextVar('glm53_mixed_capture', default=False)
_BOOT = ContextVar('glm53_mixed_boot', default=False)
CALIBRATION_BYTES = 5_020_906_496
CALIBRATION_PROXY = 256 * MIB + 516 * MIB + 516 * 256 * 264


def capture_active() -> bool:
    return _CAPTURE.get()


@contextmanager
def capture_context() -> Any:
    token = _CAPTURE.set(True)
    try:
        yield
    finally:
        _CAPTURE.reset(token)


def enabled() -> bool:
    return os.getenv(FLAG, '0') == '1'


@contextmanager
def boot_capture_context() -> Any:
    token = _BOOT.set(True)
    try:
        yield
    finally:
        _BOOT.reset(token)


def rank_reports(report: dict) -> list[dict]:
    """Use the existing Gloo world group; no device allocation or new group."""
    if not torch.distributed.is_initialized():
        return [report]
    from vllm.distributed.parallel_state import get_world_group
    group = get_world_group().cpu_group
    reports = [None] * torch.distributed.get_world_size(group=group)
    torch.distributed.all_gather_object(reports, report, group=group)
    return reports


def admit_capture(bucket: tuple, free: int, required: int, error: str = '') -> None:
    reports = rank_reports(dict(bucket=bucket, free=free, required=required, error=error))
    errors = []
    for rank, report in enumerate(reports):
        if report['bucket'] != reports[0]['bucket']:
            errors.append(f'rank {rank} bucket disagrees with rank 0')
        if report['error']:
            errors.append(f"rank {rank}: {report['error']}")
        if report['free'] < report['required']:
            errors.append(f"rank {rank}: free HBM {report['free']/MIB:.1f} MiB < "
                          f"required {report['required']/MIB:.1f} MiB")
    if errors:
        raise RuntimeError('mixed graph BOOT admission rejected on ALL ranks: ' + '; '.join(errors)
                           + '. Reduce WARMUP to one bucket, use --num-gpu-blocks-override 8000, '
                           'and set BUDGET_MIB >= calibrated estimate; restart all workers.')


def capture_requirement(estimate: int) -> int:
    """Calibrated raw ceiling + 50% transient allowance + 2 GiB emergency HBM."""
    return (estimate * 3 + 1) // 2 + positive_env('VLLM_GAUDI_MIXED_GRAPH_RESERVE_MIB', 2048) * MIB


def install_entry(adapter: Any, original_forward: Callable) -> None:
    if enabled():
        traces = ('HPU_GLM53_TRACE_KDA', 'HPU_GLM53_TRACE_MLA_INTERNAL',
                  'HPU_GLM53_TRACE_ACTIVATIONS', 'HPU_GLM53_RUNTIME_TRACE_ACTIVE',
                  'HPU_GLM53_R589_TRACE', 'HPU_GLM53_CAP_LEGACY_WARMUP',
                  'HPU_GLM53_TRUE_HOST_PREFILL_SCAN_BOUND')
        active = [name for name in traces if os.getenv(name, '0').lower() in ('1', 'true', 'yes', 'on')]
        if active or os.getenv('HPU_GLM53_RUNTIME_TRACE_SENTINEL'):
            raise ValueError('mixed graphs require tracing, warmup scan cap and true-host scan bound OFF: '
                             + ', '.join(active))
        adapter._glm53_mixed_graph_forward = original_forward


def positive_env(name: str, default: int) -> int:
    try:
        value = int(os.getenv(name, str(default)))
    except ValueError as exc:
        raise ValueError(f'{name} must be a positive integer') from exc
    if value <= 0:
        raise ValueError(f'{name} must be a positive integer')
    return value


def lattice(block_size: int) -> tuple[tuple[int, int, int], ...]:
    """Context tiers include 40k tokens; units are attention blocks per sequence. Prompt slices 256/512/2048."""
    if block_size < 1:
        raise ValueError('mixed graphs require a positive attention block_size')
    tiers, n = [0], 1
    while n * block_size <= 65536:
        tiers.append(n)
        n *= 2
    return tuple((d, p, c) for d in (1, 2, 4, 8, 16)
                 for p in (256, 512, 2048) for c in tiers if c * block_size >= p)


ESTIMATE_ENV = 'VLLM_GAUDI_MIXED_GRAPH_ESTIMATE_MIB'


def measured_estimate_bytes() -> int | None:
    """C574 measured graph size for single-bucket lattices; unset keeps the calibrated ceiling."""
    if os.getenv(ESTIMATE_ENV) is None:
        return None
    return positive_env(ESTIMATE_ENV, 1) * MIB


def estimate_bytes(bucket: tuple[int, int, int], block_size: int) -> int:
    """Raw-capture calibrated ceiling, NOT measured cache-disabled residency.

    Anchor: (4,512,256), block_size=128 allocated 5,020,906,496 bytes.
    Keep this ceiling until hardware measures the reduced tensor-cache path;
    VLLM_GAUDI_MIXED_GRAPH_ESTIMATE_MIB replaces it with that measurement.
    """
    measured = measured_estimate_bytes()
    if measured is not None:
        return measured
    d, p, c = bucket
    proxy = 256 * MIB + (d + p) * MIB + (d + p) * c * (8 + 2 * block_size)
    return (proxy * CALIBRATION_BYTES + CALIBRATION_PROXY - 1) // CALIBRATION_PROXY


def selected_buckets(block_size: int) -> tuple[tuple[int, int, int], ...]:
    spec = os.getenv('VLLM_GAUDI_MIXED_GRAPH_WARMUP', '4:512:256')
    buckets = []
    for item in filter(None, spec.split(',')):
        try:
            bucket = tuple(map(int, item.split(':')))
        except ValueError as exc:
            raise ValueError('mixed warmup must be decode_bs:prompt_tokens:context_blocks,...') from exc
        if bucket not in lattice(block_size):
            raise ValueError(f'mixed warmup bucket {bucket} outside lattice')
        if bucket in buckets:
            raise ValueError(f'duplicate mixed graph warmup bucket {bucket}')
        buckets.append(bucket)
    if not buckets:
        raise ValueError('mixed graph warmup list must contain at least one bucket')
    return tuple(buckets)


def _map(value: Any, tensor: Callable[[torch.Tensor], Any]) -> Any:
    if isinstance(value, torch.Tensor):
        return tensor(value)
    if is_dataclass(value):
        return replace(value, **{f.name: _map(getattr(value, f.name), tensor) for f in fields(value)})
    if isinstance(value, tuple) and hasattr(value, '_fields'):
        return type(value)(*(_map(v, tensor) for v in value))
    if isinstance(value, tuple):
        return tuple(_map(v, tensor) for v in value)
    if isinstance(value, list):
        return [_map(v, tensor) for v in value]
    if isinstance(value, dict):
        return {k: _map(v, tensor) for k, v in value.items()}
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise ValueError(f'unsupported mixed graph metadata {type(value).__name__}; extend explicit schema')


def signature(value: Any) -> tuple:
    tensors = []

    def record(tensor: torch.Tensor) -> tuple:
        tensors.append((tuple(tensor.shape), tensor.dtype, tensor.device))
        return ('tensor', len(tensors) - 1)

    control = _map(value, record)
    return repr(control), tuple(tensors)


def clone_inputs(value: Any) -> Any:
    return _map(value, lambda t: t.detach().clone())


def copy_inputs(target: Any, source: Any) -> None:
    if signature(target) != signature(source):
        raise ValueError('mixed graph input signature changed; choose a different bucket')
    dst, src = [], []
    _map(target, lambda t: dst.append(t))
    _map(source, lambda t: src.append(t))
    for a, b in zip(dst, src):
        a.copy_(b)


def padded_inputs(batches: list, bucket: tuple | None = None) -> tuple:
    from glm53_mixed_step import MixedMetadata
    regions, tokens, positions, cursor = [], [], [], 0
    for batch in batches:
        r = batch.region
        size, width = (r.batch, r.width) if bucket is None else (
            (bucket[0], 1) if r.decode else (1, bucket[1]))
        count = size * width
        metadata = r.metadata if bucket is None else pad_metadata(r, size, width, bucket[2])
        # Host ownership/positions are deliberately absent from the graph key.
        # Tensor metadata retains real physical slots, masks, and sequence lengths.
        regions.append(replace(r, requests=tuple(range(cursor, cursor+size)),
                               lengths=(width,) * size, starts=(0,) * size,
                               batch=size, width=width, metadata=metadata,
                               valid=tuple(range(count)), graph_padded=True))
        tokens.append(pad_tensor(batch.tokens, (size, width)))
        positions.append(pad_tensor(batch.positions, (size, width)))
        cursor += size
    return (MixedMetadata(tuple(regions)),
            torch.cat([t.reshape(-1) for t in tokens]).unsqueeze(0),
            torch.cat([t.reshape(-1) for t in positions]).unsqueeze(0))


def compact_hidden(hidden: torch.Tensor, batches: list, bucket: tuple | None = None) -> torch.Tensor:
    from glm53_mixed_step import unpack_region
    hidden = hidden.reshape(-1, hidden.shape[-1])
    parts, cursor = [], 0
    for batch in batches:
        r = batch.region
        size, width = (r.batch, r.width) if bucket is None else (
            (bucket[0], 1) if r.decode else (1, bucket[1]))
        count = size * width
        local = hidden[cursor:cursor+count].reshape(size, width, -1)
        parts.append(unpack_region(local[:r.batch, :r.width].reshape(-1, hidden.shape[-1]), r))
        cursor += count
    return torch.cat(parts)


def select_bucket(batches: list, block_size: int) -> tuple[tuple | None, str]:
    if not batches or not batches[0].region.decode:
        return None, 'no_decode'
    if len(batches) != 2 or batches[1].region.decode:
        return None, 'multi_prompt'
    d, p = (b.region for b in batches)
    if p.batch != 1 or len(p.requests) != 1:
        return None, 'multi_prompt'
    tables = [getattr(r.metadata, 'glm53_index_block_table', None) for r in (d, p)]
    if any(t is None or t.ndim != 2 for t in tables):
        return None, 'signature'
    needed = (d.batch, p.width, max(t.shape[1] for t in tables))
    candidates = list(selected_buckets(block_size))
    if exact_shapes_enabled():
        candidates = [b for b in candidates if b[:2] == needed[:2]]
        if not candidates:
            return None, 'shape_mismatch'
    for axis, reason in enumerate(('decode_rows_over', 'width_over', 'ctx_over')):
        candidates = [b for b in candidates if b[axis] >= needed[axis]]
        if not candidates:
            return None, reason
    return min(candidates, key=lambda b: (estimate_bytes(b, block_size), b)), ''


def bucket_for(batches: list, block_size: int) -> tuple | None:
    return select_bucket(batches, block_size)[0]


def pad_tensor(value: torch.Tensor, shape: tuple, fill: int = 0) -> torch.Tensor:
    if value.ndim != len(shape) or any(a > b for a, b in zip(value.shape, shape)):
        raise ValueError(f'mixed metadata shape {tuple(value.shape)} exceeds bucket {shape}')
    result = value.new_full(shape, fill)
    result[tuple(slice(0, n) for n in value.shape)] = value
    return result


def pad_token_field(value: torch.Tensor, region: Any, size: int, width: int,
                    fill: int) -> torch.Tensor:
    valid = len(region.requests)
    shaped = value.reshape(region.batch, region.width)[:valid]
    # Ignore the runner's original padding addresses, which use several conventions.
    mask = torch.arange(region.width, device=value.device)[None, :] < torch.tensor(
        region.lengths, device=value.device)[:, None]
    shaped = torch.where(mask, shaped, shaped.new_full((), fill))
    result = pad_tensor(shaped, (size, width), fill)
    return result if value.ndim == 2 else result.reshape(-1)


def pad_field(name: str, value: Any, region: Any, size: int, width: int, ctx: int) -> Any:
    if name in ('num_actual_tokens', 'num_decodes'):
        return size * width if name == 'num_actual_tokens' else (size if region.decode else 0)
    if not isinstance(value, torch.Tensor):
        return value
    if name == 'glm53_index_block_table':
        return pad_tensor(value[:len(region.requests)], (size, ctx))
    if name in ('slot_mapping', 'glm53_index_slot_mapping', 'glm53_tail_slot_mapping',
                'glm53_token_mask', 'padding_mask_flat'):
        fill = -1 if 'slot_mapping' in name else 0
        return pad_token_field(value, region, size, width, fill)
    if name in ('load_indices_tensor', 'store_indices_tensor', 'non_spec_state_indices_tensor'):
        shape = (*value.shape[:-1], size)
        return pad_tensor(value[..., :len(region.requests)], shape, -1)
    if name in ('seq_lens_tensor', 'context_lens_tensor', 'has_initial_states_p'):
        return pad_tensor(value[:len(region.requests)], (size, *value.shape[1:]))
    if region.decode and name in ('query_start_loc', 'query_start_loc_p', 'non_spec_query_start_loc'):
        return torch.arange(size + 1, dtype=value.dtype, device=value.device)
    return value


def pad_metadata(region: Any, size: int, width: int, ctx: int) -> Any:
    """Keep GLM's direct sparse/KDA inputs; discard unused dense-PA scratch.

    -1 state slots map to the allocation-owned final garbage row in _forward_hpu.
    -1 sparse/Q4 slots map to their allocation-owned final garbage rows. No new
    recurrent slots, KV pages or offload request ownership are allocated here.
    """
    metadata = region.metadata
    names = [f.name for f in fields(metadata)] if is_dataclass(metadata) else metadata._fields
    unused = {'attn_bias', 'block_list', 'block_mapping', 'block_usage', 'block_groups',
              'blocks_caching_range', 'mamba_chunks_to_block_mapping', 'seqlens_offsets_for_blocks',
              'last_chunk_indices_p', 'prep_initial_states'}
    updates = {name: (None if name in unused else pad_field(
        name, getattr(metadata, name), region, size, width, ctx)) for name in names}
    return replace(metadata, **updates) if is_dataclass(metadata) else metadata._replace(**updates)


def write_q4_rows(cache: torch.Tensor, slots: torch.Tensor, packed: torch.Tensor) -> None:
    """Unsharded counterpart of Q4's existing TP allocation-owned dummy row."""
    valid = slots >= 0
    targets = torch.where(valid, slots.long(), torch.full_like(slots, cache.shape[0] - 1).long())
    values = torch.where(valid.unsqueeze(-1), packed, torch.zeros_like(packed))
    cache.index_copy_(0, targets, values)


SPLIT_FLAG = 'VLLM_GAUDI_MIXED_SPLIT_FORWARD'


def split_forward_enabled() -> bool:
    """C574, opt-in: one model forward per region inside the captured mixed graph.

    Token-wise ops (GEMMs, MoE router and experts, mhc, norms) then see each region's own padded rows, 16 decode
    and 256 prompt, the row classes of the native decode graph and the native (1, 256) prompt forward, instead of
    one shared M = 272 call whose recipes match neither (C165 / C293 / C573).
    """
    value = os.getenv(SPLIT_FLAG, '0')
    if value not in ('0', '1'):
        raise ValueError(f'{SPLIT_FLAG}={value!r} must be 0 or 1; unset it (or 0) to keep one shared forward '
                         'per mixed graph, 1 to run decode and prompt rows as separate forwards')
    return value == '1'


EXACT_FLAG = 'VLLM_GAUDI_MIXED_EXACT_SHAPES'


def exact_shapes_enabled() -> bool:
    """C576, opt-in: serve a mixed step only at the padded shapes native runs it at; a miss runs native.

    With the split forward each region then keeps its native row count end to end, MoE algorithm included
    (hpu_fp8 sends > 512 rows to the R589 dense tile, 9-512 rows to the grouped capacity path, <= 8 rows to the
    routed op): bucket d must equal the padded decode batch and bucket p the padded chunk width. The miss path is
    the runner's native per-batch forward (glm53_mixed_step.prepare), never the lazy mixed forward (C570).
    """
    value = os.getenv(EXACT_FLAG, '0')
    if value not in ('0', '1'):
        raise ValueError(f'{EXACT_FLAG}={value!r} must be 0 or 1; unset it (or 0) to pad to the smallest '
                         'fitting bucket, 1 to serve only the native padded shapes')
    if value == '1' and not split_forward_enabled():
        raise ValueError(f'{EXACT_FLAG}=1 needs {SPLIT_FLAG}=1: a shared forward mixes the regions\' row counts '
                         'whatever the bucket; set both, or unset both')
    return value == '1'


def miss_event() -> str:
    """Live misses run native under exact shapes; BOOT and flag-off misses keep the lazy mixed forward."""
    return 'native_fallbacks' if exact_shapes_enabled() and not _BOOT.get() else 'lazy_fallbacks'


class RegionPlan(Protocol):
    """glm53_mixed_step.MixedMetadata as region_forwards reads it (a frozen dataclass: replace() rebuilds it)."""

    regions: tuple


def region_forwards(forward: Callable[..., torch.Tensor], plan: RegionPlan, tokens: torch.Tensor,
                    positions: torch.Tensor) -> torch.Tensor:
    """Each region of a padded plan through its own forward; rows come back flat, in plan order."""
    parts, cursor = [], 0
    for region in plan.regions:
        span = slice(cursor, cursor + region.count)
        hidden = forward(input_ids=tokens[:, span], positions=positions[:, span],
                         attn_metadata=replace(plan, regions=(region,)))
        parts.append(hidden.reshape(-1, hidden.shape[-1]))
        cursor += region.count
    return torch.cat(parts)


class StaticCall:
    """Keep all metadata tensors explicit inputs to the bridge's lifetime analysis."""

    def __init__(self, forward: Callable, static: tuple) -> None:
        self.forward, self.static = forward, static
        self.split = split_forward_enabled()
        tensors = []
        _map(static, lambda tensor: tensors.append(tensor))
        self.tensor_inputs = tuple(tensors)

    def __call__(self) -> torch.Tensor:
        plan, tokens, positions = self.static
        if self.split:
            return region_forwards(self.forward, plan, tokens, positions)
        return self.forward(input_ids=tokens, positions=positions, attn_metadata=plan)


class CaptureModule(torch.nn.Module):
    def __init__(self, call: Callable) -> None:
        super().__init__()
        self.call = call

    def forward(self, *tensor_inputs: torch.Tensor) -> torch.Tensor:
        # These are the same static tensors used by call, marked as user inputs.
        return self.call()


def bridge_policy_error() -> str:
    if os.getenv('PT_HPUGRAPH_DISABLE_TENSOR_CACHE', '1') != '1':
        return 'PT_HPUGRAPH_DISABLE_TENSOR_CACHE must be unset or 1'
    if os.getenv('PT_HPU_LAZY_MODE', '1') != '1':
        return 'PT_HPU_LAZY_MODE must be 1'
    return ''


class HpuBackend:
    """Runner's bridge wrapper, with dry capture and retained in-place state."""

    def prepare_capture(self, bucket: tuple, estimate: int, error: str) -> None:
        free, required = 0, 0
        try:
            import habana_frameworks.torch as htorch
            error = error or bridge_policy_error()
            if not _BOOT.get():
                error = error or 'first-hit capture outside BOOT is forbidden; warm this bucket'
            htorch.core.mark_step()
            htorch.hpu.synchronize()
            free, total = htorch.hpu.mem_get_info()
            if not isinstance(free, int) or not 0 <= free <= total:
                raise ValueError('mem_get_info returned invalid free/total bytes')
            required = capture_requirement(estimate)
        except Exception as exc:
            error = error or f'free memory query failed: {type(exc).__name__}: {exc}'
        admit_capture(bucket, free, required, error)
        self.free_before = free
        LOG.warning('GLM53_MIXED_GRAPH admitted bucket=%s free_bytes=%s required_bytes=%s',
                    bucket, free, required)

    def capture(self, call: Callable[[], torch.Tensor]) -> tuple:
        import habana_frameworks.torch as htorch
        error = bridge_policy_error()
        if error:
            raise RuntimeError(error)
        handle, output, error = None, None, ''
        try:
            htorch.core.mark_step()
            htorch.hpu.synchronize()
            before = htorch.hpu.memory_allocated()
            free_before, _ = htorch.hpu.mem_get_info()
            htorch.hpu.reset_peak_memory_stats()
            module = htorch.hpu.wrap_in_hpu_graph(
                CaptureModule(call), disable_tensor_cache=True, dry_run=True,
                free_inplace=False, max_graphs=1)
            handle = module, call.tensor_inputs
            output = module(*call.tensor_inputs)
            htorch.hpu.synchronize()
            self.last_capture_bytes = max(0, htorch.hpu.memory_allocated() - before)
            self.free_after, _ = htorch.hpu.mem_get_info()
            self.last_capture_hbm_bytes = max(0, free_before - self.free_after)
            self.last_capture_peak_bytes = max(0, htorch.hpu.max_memory_allocated() - before)
        except Exception as exc:
            error = f'capture failed without retry: {type(exc).__name__}: {exc}'
        admit_capture((), 0, 0, error)
        return handle, output

    def replay(self, handle: Any) -> None:
        import habana_frameworks.torch as htorch
        htorch.core.mark_step()
        module, tensor_inputs = handle
        module(*tensor_inputs)


HBM_PROBE = 'VLLM_GAUDI_MIXED_GRAPH_HBM_PROBE'


def hbm_probe_enabled() -> bool:
    """C568 log-only HBM probe switch. Unset or 0 keeps the plain backend; any other value refuses the boot."""
    value = os.getenv(HBM_PROBE, '0').strip()
    if value not in ('0', '1'):
        raise ValueError(f'{HBM_PROBE}={value!r}: set 1 to log mixed-graph HBM readings, or 0 / unset for none')
    return value == '1'


def default_backend() -> HpuBackend | HbmProbe:
    return HbmProbe(HpuBackend()) if hbm_probe_enabled() else HpuBackend()


class MarkStepCensus:
    """Allocated bytes added between consecutive Python mark_step calls of one capture, keyed by call site.

    Tensors materialize at mark_step, so the growth since the previous call lands on the site that ended the
    segment. Bytes added after the last call (the bridge's own replay of the captured graph) are unattributed.
    """

    def __init__(self, core: Any, hpu: Any) -> None:
        self.core, self.hpu = core, hpu
        self.growth: Counter = Counter()
        self.calls: Counter = Counter()

    def __enter__(self) -> 'MarkStepCensus':
        self.original = self.core.mark_step
        self.start = self.last = self.hpu.memory_allocated()
        self.core.mark_step = self.record
        return self

    def __exit__(self, *exc_info: Any) -> None:
        self.core.mark_step = self.original

    def record(self, *args: object, **kwargs: object) -> object:
        """Stands in for mark_step: the real call first, then the bytes it materialized, keyed by the caller."""
        import sys
        result = self.original(*args, **kwargs)
        caller = sys._getframe(1)
        site = f'{os.path.basename(caller.f_code.co_filename)}:{caller.f_lineno}'
        now = self.hpu.memory_allocated()
        self.calls[site] += 1
        self.growth[site] += now - self.last
        self.last = now
        return result

    def summary(self, limit: int = 12) -> str:
        end = self.hpu.memory_allocated()
        top = sorted(self.growth, key=lambda site: -abs(self.growth[site]))[:limit]
        sites = ','.join(f'{site}:{self.calls[site]}:{self.growth[site]}' for site in top) or '-'
        return (f'mark_steps={sum(self.calls.values())} growth_bytes={end - self.start} '
                f'unattributed_bytes={end - self.last} top_sites={sites}')


class HbmProbe:
    """Log-only composition around the bridge backend: the same calls in the same order, plus allocator readings.

    capture logs the MarkStepCensus of the graph; replay logs the allocated bytes before it, its own peak, and the
    peak since the previous replay (native prompt-only and decode steps), resetting the peak after every replay.
    """

    def __init__(self, backend: Any) -> None:
        self.backend = backend
        self.bucket: tuple = ()
        self.buckets: dict[int, tuple] = {}
        self.replays = self.max_transient = self.max_between = 0

    def __getattr__(self, name: str) -> Any:
        """GraphCache's getattr reads of the backend's capture measurements (last_capture_bytes, free_after...)."""
        backend = self.__dict__.get('backend')
        if backend is None:
            raise AttributeError(name)
        return getattr(backend, name)

    def prepare_capture(self, bucket: tuple, estimate: int, error: str) -> None:
        self.bucket = bucket
        self.backend.prepare_capture(bucket, estimate, error)

    def capture(self, call: Callable[[], torch.Tensor]) -> tuple:
        import habana_frameworks.torch as htorch
        with MarkStepCensus(htorch.core, htorch.hpu) as census:
            try:
                handle, output = self.backend.capture(call)
            except Exception:
                LOG.warning('GLM53_MIXED_GRAPH hbm_probe capture_failed bucket=%s %s', self.bucket, census.summary())
                raise
        self.buckets[id(handle)] = self.bucket
        LOG.warning('GLM53_MIXED_GRAPH hbm_probe capture bucket=%s %s', self.bucket, census.summary())
        htorch.hpu.reset_peak_memory_stats()  # the first replay's between-peak must not carry the capture peak
        return handle, output

    def replay(self, handle: Any) -> None:
        import habana_frameworks.torch as htorch
        hpu = htorch.hpu
        between, before = hpu.max_memory_allocated(), hpu.memory_allocated()
        hpu.reset_peak_memory_stats()
        self.backend.replay(handle)
        peak, after = hpu.max_memory_allocated(), hpu.memory_allocated()
        hpu.reset_peak_memory_stats()
        self.log_replay(self.buckets.get(id(handle), ()), before, peak, after, between)

    def log_replay(self, bucket: tuple, before: int, peak: int, after: int, between: int) -> None:
        self.replays += 1
        transient, gap = peak - before, between - before
        new_max = transient > self.max_transient or gap > self.max_between
        self.max_transient, self.max_between = max(self.max_transient, transient), max(self.max_between, gap)
        every = positive_env('VLLM_GAUDI_MIXED_GRAPH_LOG_EVERY', 100)
        if self.replays <= 16 or new_max or self.replays % every == 0:
            LOG.warning('GLM53_MIXED_GRAPH hbm_probe replay n=%d bucket=%s before_bytes=%d replay_peak_bytes=%d '
                        'after_bytes=%d between_peak_bytes=%d max_transient_bytes=%d max_between_bytes=%d',
                        self.replays, bucket, before, peak, after, between, self.max_transient, self.max_between)


class GraphCache:
    def __init__(self, raw_forward: Callable, block_size: int, backend: Any = None) -> None:
        self.raw_forward = raw_forward
        self.block_size = block_size
        self.backend = backend if backend is not None else default_backend()
        self.entries: dict[tuple, tuple] = {}
        self.counts: Counter = Counter(mixed_steps=0, graph_replays=0, captures=0,
                                      lazy_fallbacks=0, native_fallbacks=0, prompt_tokens=0, decode_rows=0)
        self.counts.update(dict.fromkeys(('fallback_' + reason for reason in (
            'decode_rows_over', 'width_over', 'ctx_over', 'no_decode', 'multi_prompt', 'signature',
            'shape_mismatch')), 0))
        self.estimated_bytes = 0

    def record(self, batches: list, event: str, reason: str = '') -> None:
        self.counts['mixed_steps'] += 1
        self.counts['boot_steps' if _BOOT.get() else 'live_steps'] += 1
        self.counts[event] += 1
        self.counts['prompt_tokens'] += sum(b.region.count for b in batches if not b.region.decode)
        self.counts['decode_rows'] += sum(b.region.count for b in batches if b.region.decode)
        if reason:
            self.counts['fallback_' + reason] += 1
        every = positive_env('VLLM_GAUDI_MIXED_GRAPH_LOG_EVERY', 100)
        if self.counts['mixed_steps'] == 1 or self.counts['mixed_steps'] % every == 0:
            steps = self.counts['mixed_steps']
            LOG.warning('GLM53_MIXED_GRAPH counters=%s prompt_tokens_per_step=%.2f '
                        'decode_rows_per_step=%.2f entries=%d estimated_MiB=%.1f',
                        dict(self.counts), self.counts['prompt_tokens']/steps,
                        self.counts['decode_rows']/steps, len(self.entries), self.estimated_bytes/MIB)

    def run(self, batches: list) -> torch.Tensor | None:
        bucket, reason = select_bucket(batches, self.block_size)
        if bucket is None:
            self.counts['fallback_outside_lattice'] += 1  # Legacy aggregate, not another fallback.
            self.record(batches, miss_event(), reason)
            return None
        try:
            inputs = padded_inputs(batches, bucket)
        except (ValueError, TypeError, AttributeError):
            self.record(batches, miss_event(), 'signature')
            return None
        key = bucket, signature(inputs)
        if key in self.entries:
            static, graph, output = self.entries[key]
            copy_inputs(static, inputs)
            self.backend.replay(graph)
            self.record(batches, 'graph_replays')
            return compact_hidden(output, batches, bucket)
        if any(entry[0] == bucket for entry in self.entries):
            self.record(batches, miss_event(), 'signature')
            return None
        return self.capture(key, inputs, batches)

    def capture(self, key: tuple, inputs: tuple, batches: list) -> torch.Tensor | None:
        estimate = estimate_bytes(key[0], self.block_size)
        limit = positive_env('VLLM_GAUDI_MIXED_GRAPH_MAX_GRAPHS', 4)
        budget = positive_env('VLLM_GAUDI_MIXED_GRAPH_BUDGET_MIB', 6144) * MIB
        error = ''
        if len(self.entries) >= limit or self.estimated_bytes + estimate > budget:
            error = 'mixed graph budget exhausted before capture; shrink warmup lattice '
            error += 'or raise MAX_GRAPHS/BUDGET_MIB after per-rank HBM measurement'
        prepare = getattr(self.backend, 'prepare_capture', None)
        if prepare is not None:
            prepare(key[0], estimate, error)
        elif error:
            raise ValueError(error)
        static = clone_inputs(inputs)
        call = StaticCall(self.raw_forward, static)
        # No catch-and-lazy retry: a failed capture may have mutated KDA/KV.
        with capture_context():
            graph, output = self.backend.capture(call)
        measured = getattr(self.backend, 'last_capture_bytes', None)
        device_delta = getattr(self.backend, 'last_capture_hbm_bytes', 0)
        inputs_bytes = sum(t.numel() * t.element_size() for t in call.tensor_inputs)
        charge = max(estimate, max(measured or 0, device_delta) + inputs_bytes)
        error = 'measured graph residency exceeded budget' if self.estimated_bytes + charge > budget else ''
        if prepare is not None:
            admit_capture(key[0], 0, 0, error)
        self.entries[key] = static, graph, output
        self.estimated_bytes += charge
        self.record(batches, 'captures')
        LOG.warning('GLM53_MIXED_GRAPH captured bucket=%s estimate_MiB=%.1f allocated_delta_bytes=%s '
                    'hbm_delta_bytes=%s static_input_bytes=%s peak_delta_bytes=%s '
                    'free_before_bytes=%s free_after_bytes=%s tensor_cache=off split_forward=%d',
                    key[0], estimate/MIB, measured, device_delta, inputs_bytes,
                    getattr(self.backend, 'last_capture_peak_bytes', None),
                    getattr(self.backend, 'free_before', None), getattr(self.backend, 'free_after', None),
                    call.split)
        return compact_hidden(output, batches, key[0])


def forward(runner: Any, batches: list) -> torch.Tensor | None:
    if not enabled():
        return None
    raw = getattr(runner.model, '_glm53_mixed_graph_forward', None)
    if raw is None:
        raise RuntimeError('mixed graphs require the paired lazy HpuModelAdapter patch; restart with flag ON')
    if runner.model_config.enforce_eager:
        raise ValueError('mixed graphs require enforce_eager=False and PT_HPU_LAZY_MODE=1')
    cache = getattr(runner.model, '_glm53_mixed_graph_cache', None)
    if cache is None:
        cache = GraphCache(raw, runner.attn_block_size)
        runner.model._glm53_mixed_graph_cache = cache
    return cache.run(batches)


def warmup(runner: Any) -> None:
    """Real execute_model/sample_tokens path, after compile-only contexts exit."""
    if not enabled():
        return
    buckets, error = (), ''
    try:
        buckets = validate_warmup(runner)
    except Exception as exc:
        error = str(exc)
    admit_capture(buckets, 0, 0, error)
    with boot_capture_context():
        for d, p, c in buckets:
            context = c * runner.attn_block_size - p
            rows = warmup_decode_rows(runner, d)
            runner._prepare_dummy_scenario((1, p, context // runner.block_size), (rows, 1, c))
    # Outside capture and BOOT: lazy prefills outside the lattice reuse these
    # widened indexer recipes. No-op unless HPU_GLM53_WARM_ADAPTIVE_TILES=1.
    from glm53_r590_long_history import warm_adaptive_tiles_at_boot
    warm_adaptive_tiles_at_boot(runner.model)


def warmup_decode_rows(runner: Any, padded_rows: int) -> int:
    rows = min(padded_rows, runner.input_batch.max_num_reqs - 1)
    if rows < 1 or 1 << (rows - 1).bit_length() != padded_rows:
        raise ValueError(f'mixed warmup request capacity cannot represent {padded_rows} '
                         'padded decode rows plus one prompt; shrink WARMUP decode tier '
                         'or raise max-num-seqs')
    return rows


def validate_warmup(runner: Any) -> tuple:
    if os.getenv('VLLM_GAUDI_MIXED_STEP', '0') != '1':
        raise ValueError('VLLM_GAUDI_MIXED_GRAPHS=1 requires VLLM_GAUDI_MIXED_STEP=1')
    exact_shapes_enabled()  # a bad split / exact-shape value or pairing fails the boot before any capture
    buckets = selected_buckets(runner.attn_block_size)
    if measured_estimate_bytes() is not None and len(buckets) != 1:
        raise ValueError(f'{ESTIMATE_ENV} is one measured graph; set WARMUP to a single bucket '
                         f'(got {len(buckets)}) or unset it for the calibrated ceiling')
    estimate = sum(estimate_bytes(b, runner.attn_block_size) for b in buckets)
    budget = positive_env('VLLM_GAUDI_MIXED_GRAPH_BUDGET_MIB', 6144) * MIB
    if estimate > budget or len(buckets) > positive_env('VLLM_GAUDI_MIXED_GRAPH_MAX_GRAPHS', 4):
        raise ValueError(f'mixed graph warmup budget needs {estimate/MIB:.1f} MiB; '
                         'reduce WARMUP or raise BUDGET_MIB after HBM measurement')
    for d, p, c in buckets:
        if c * runner.attn_block_size > runner.max_model_len:
            raise ValueError(f'mixed warmup bucket {(d, p, c)} exceeds max_model_len')
        context = c * runner.attn_block_size - p
        if context % runner.block_size:
            raise ValueError('mixed warmup sparse tier must align to runner.block_size')
        warmup_decode_rows(runner, d)
    return buckets
