# SPDX-License-Identifier: Apache-2.0
"""Native-dtype recurrent-state sidecar storage for hybrid KV offloading."""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from typing import Callable, NamedTuple

import torch

import glm53_tail_fuse as _tail_fuse


# VLLM_RECURRENT_SIDECAR_CONTIG=1: one pinned byte buffer per block, typed views per state tensor. Each pinned tensor
# maps at least 2 MiB, so 1 MiB state tensors cost 2x host memory one by one (C603); one buffer costs about 1x.
CONTIG_ENV = "VLLM_RECURRENT_SIDECAR_CONTIG"
CONTIG_ALIGN_BYTES = 64


class ContigField(NamedTuple):
    """One state tensor's place in a block's byte buffer."""

    group_idx: int
    offset: int
    nbytes: int
    shape: tuple[int, ...]
    dtype: torch.dtype


def contig_enabled() -> bool:
    raw = os.environ.get(CONTIG_ENV, "").strip() or "0"
    if raw not in ("0", "1"):
        raise ValueError(f"{CONTIG_ENV}={raw!r}: use 1 (one pinned buffer per block) or 0/unset (one per tensor)")
    return raw == "1"


def contig_layout(groups: dict[int, tuple[torch.Tensor, ...]]) -> tuple[list[ContigField], int]:
    """(each tensor's aligned place in registry order, the block's total bytes)."""
    fields: list[ContigField] = []
    offset = 0
    for group_idx, tensors in groups.items():
        for tensor in tensors:
            shape = tuple(tensor.shape[1:])
            nbytes = math.prod(shape) * torch.empty(0, dtype=tensor.dtype).element_size()
            fields.append(ContigField(group_idx, offset, nbytes, shape, tensor.dtype))
            offset += (nbytes + CONTIG_ALIGN_BYTES - 1) // CONTIG_ALIGN_BYTES * CONTIG_ALIGN_BYTES
    return fields, offset


def _contig_buffer(nbytes: int, pinned: bool) -> torch.Tensor:
    return torch.empty(nbytes, dtype=torch.uint8, device="cpu", pin_memory=pinned)


def contig_slot(
    fields: list[ContigField], total_bytes: int, pinned: bool
) -> dict[int, tuple[torch.Tensor, ...]]:
    """One block's typed views, same groups, order, shapes and dtypes as the per-tensor slot."""
    buffer = _contig_buffer(total_bytes, pinned)
    if buffer.data_ptr() % CONTIG_ALIGN_BYTES:
        raise RuntimeError(
            f"recurrent sidecar buffer is not {CONTIG_ALIGN_BYTES}-byte aligned, so typed views cannot be trusted; "
            f"unset {CONTIG_ENV} and use the 45/20 store split"
        )
    views: dict[int, list[torch.Tensor]] = {}
    for field in fields:
        raw = buffer[field.offset:field.offset + field.nbytes]
        views.setdefault(field.group_idx, []).append(raw.view(field.dtype).reshape(field.shape))
    return {group_idx: tuple(tensors) for group_idx, tensors in views.items()}


def _require_pinned(slot: dict[int, tuple[torch.Tensor, ...]]) -> None:
    if not all(tensor.is_pinned() for tensors in slot.values() for tensor in tensors):
        raise RuntimeError(
            f"{CONTIG_ENV}=1: a typed view of the pinned buffer does not report pinned, so HPU copies would not use "
            f"pinned host memory; unset {CONTIG_ENV} and use the 45/20 store split (304 slots)"
        )


def recurrent_state_bytes_per_rank(kv_cache_config) -> int:
    """Return native recurrent bytes for one exact boundary on one rank."""
    from vllm.v1.kv_cache_interface import MambaSpec, UniformTypeKVCacheSpecs

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


def validate_budget_split(
    attention_bytes: int,
    recurrent_bytes: int,
    total_bytes: int,
) -> None:
    """Validate the two independent LRUs against one explicit host budget."""
    if min(attention_bytes, recurrent_bytes, total_bytes) < 0:
        raise ValueError("CPU offload budgets must be non-negative")
    if attention_bytes + recurrent_bytes > total_bytes:
        raise ValueError(
            "cpu_bytes_to_use + recurrent_state_bytes_to_use exceeds "
            "total_cpu_bytes_to_use"
        )


def local_transfer_parts(
    *, attention_transfer: bool, has_sidecar: bool, is_store: bool,
    is_store_writer: bool,
) -> int:
    """Number of independently fenced parts performed by this rank."""
    attention_parts = int(
        attention_transfer and (not is_store or is_store_writer)
    )
    return attention_parts + int(has_sidecar)


def compact_sidecar_row(
    base_slot: int,
    num_groups: int,
    group_offset: int,
    spec_width: int,
    active_column: int,
) -> int:
    """Resolve the exact live compact row; row zero remains reserved."""
    if min(base_slot, group_offset, active_column) < 0:
        raise ValueError("compact recurrent coordinates must be non-negative")
    if num_groups <= 0 or spec_width <= 0:
        raise ValueError("compact recurrent geometry must be positive")
    if group_offset >= num_groups or active_column >= spec_width:
        raise ValueError("compact recurrent coordinate is out of range")
    return (
        (base_slot * num_groups + group_offset) * spec_width
        + active_column
        + 1
    )


def resolve_compact_sidecar_row(
    req_id: str,
    group_idx: int,
    *,
    base_slots: dict[str, int],
    active_columns: dict[str, int],
    group_offsets: dict[int, int],
    num_groups: int,
    spec_width: int,
) -> int:
    """Resolve an allocated live row without a default or clamp."""
    if req_id not in base_slots:
        raise RuntimeError(f"no allocated compact slot for {req_id}")
    if req_id not in active_columns:
        raise RuntimeError(f"no active compact column for {req_id}")
    active_column = int(active_columns[req_id])
    if active_column < 0 or active_column >= spec_width:
        raise RuntimeError(
            f"active compact column {active_column} is outside [0, {spec_width})"
        )
    return compact_sidecar_row(
        int(base_slots[req_id]),
        num_groups,
        group_offsets[group_idx],
        spec_width,
        active_column,
    )


@dataclass(frozen=True)
class RecurrentStateRegistry:
    groups: dict[int, tuple[torch.Tensor, ...]]
    resolve_row: Callable[[str, int], int]
    bytes_per_rank: int
    signature: str
    ensure_slot: Callable[[str], int] | None = None


_registry: RecurrentStateRegistry | None = None
_worker: "RecurrentSidecarWorker | None" = None
# TAIL_FUSE_V1: snapshot bank tensors per compact group, in the registry's flat ref order.
# Outside the registry, so signature / bytes_per_rank / the CPU slot layout are unchanged.
_snapshot_sources: dict[int, tuple[torch.Tensor, ...]] | None = None


def _register_snapshot_bank(
    runner, kv_caches: dict[str, object], groups: dict[int, tuple[torch.Tensor, ...]]
) -> None:
    """Allocate max_rows() device rows per KDA layer (before warmup/capture) and log BANK.

    Checks the flag itself, so register_recurrent_state calls it unconditionally: flag off allocates nothing.
    """
    global _snapshot_sources
    if not _tail_fuse.enabled():
        return
    bank = _tail_fuse.replace_bank()
    rows = _tail_fuse.max_rows()
    sources: dict[int, tuple[torch.Tensor, ...]] = {}
    nbytes = 0
    for group_idx, refs in groups.items():
        flat: list[torch.Tensor] = []
        for layer_name in runner.kv_cache_config.kv_cache_groups[group_idx].layer_names:
            raw = kv_caches[layer_name]
            nbytes += bank.allocate(layer_name, raw if isinstance(raw, (tuple, list)) else (raw,), rows)
            flat.extend(bank.rows(layer_name))
        if len(flat) != len(refs):
            raise RuntimeError(
                f"tail-fuse bank for group {group_idx} has {len(flat)} tensors, registry {len(refs)}; "
                "the layer order changed, unset VLLM_RECURRENT_TAIL_FUSE and report"
            )
        sources[group_idx] = tuple(flat)
    _snapshot_sources = sources
    _tail_fuse.log_bank(rows, nbytes, len(bank.layer_names()))


def register_recurrent_state(runner, kv_caches: dict[str, object]) -> None:
    """Register compact Mamba tensors and their host-only row resolver."""
    from vllm.v1.kv_cache_interface import MambaSpec, UniformTypeKVCacheSpecs

    global _registry, _worker
    compact_group_ids = set(getattr(runner, "_compact_gdn_group_ids", ()))
    if not compact_group_ids:
        _registry = None
        _worker = None
        return
    if bool(getattr(runner, "_gdn_transactional_mtp", False)):
        raise RuntimeError(
            "recurrent sidecar requires one compact row geometry for every state tensor"
        )

    group_offsets = dict(getattr(runner, "_compact_gdn_group_offset", {}))
    num_groups = int(getattr(runner, "_num_gdn_groups", 0))
    spec_width = int(getattr(runner, "_gdn_spec_width", 0))
    if set(group_offsets) != compact_group_ids or num_groups != len(compact_group_ids):
        raise RuntimeError("compact recurrent group registry is incomplete")

    groups: dict[int, tuple[torch.Tensor, ...]] = {}
    bytes_per_rank = 0
    for group_idx, group in enumerate(runner.kv_cache_config.kv_cache_groups):
        if group_idx not in compact_group_ids:
            continue
        group_spec = group.kv_cache_spec
        per_layer = (
            group_spec.kv_cache_specs
            if isinstance(group_spec, UniformTypeKVCacheSpecs)
            else {}
        )
        refs: list[torch.Tensor] = []
        for layer_name in group.layer_names:
            spec = per_layer.get(layer_name, group_spec)
            if not isinstance(spec, MambaSpec):
                raise RuntimeError("compact recurrent group contains a non-Mamba layer")
            raw = kv_caches[layer_name]
            tensors = raw if isinstance(raw, (tuple, list)) else (raw,)
            if len(tensors) != len(spec.shapes):
                raise RuntimeError("Mamba state tensor count disagrees with MambaSpec")
            for tensor, shape, dtype in zip(tensors, spec.shapes, spec.dtypes):
                if not isinstance(tensor, torch.Tensor):
                    raise RuntimeError("Mamba state reference is not a tensor")
                if tuple(tensor.shape[1:]) != tuple(shape) or tensor.dtype != dtype:
                    raise RuntimeError("Mamba state tensor disagrees with native spec")
                refs.append(tensor)
            bytes_per_rank += spec.state_content_size_bytes
        groups[group_idx] = tuple(refs)
    _register_snapshot_bank(runner, kv_caches, groups)

    def resolve(req_id: str, group_idx: int) -> int:
        return resolve_compact_sidecar_row(
            req_id,
            group_idx,
            base_slots=runner._gdn_req_to_base_slot,
            active_columns=runner._gdn_active_column,
            group_offsets=group_offsets,
            num_groups=num_groups,
            spec_width=spec_width,
        )

    ensure_slot = getattr(runner, "ensure_recurrent_sidecar_slot", None)
    if not callable(ensure_slot):
        raise RuntimeError(
            "recurrent sidecar requires an idempotent compact-slot allocator"
        )

    signature = repr(
        tuple(
            (
                group_idx,
                tuple(
                    (tuple(tensor.shape[1:]), str(tensor.dtype))
                    for tensor in tensors
                ),
            )
            for group_idx, tensors in sorted(groups.items())
        )
    )
    _registry = RecurrentStateRegistry(
        groups, resolve, bytes_per_rank, signature, ensure_slot
    )
    _worker = RecurrentSidecarWorker(_registry)


@dataclass
class _Pending:
    job_id: int
    event: object | None


class RecurrentSidecarWorker:
    """Per-rank native-dtype CPU state pool with stream/event fencing."""

    def __init__(self, registry: RecurrentStateRegistry):
        self.registry = registry
        self._resident: dict[int, dict[int, tuple[torch.Tensor, ...]]] = {}
        self._pending: dict[int, _Pending] = {}
        self._stream = None
        self._uses_hpu = any(
            tensor.device.type == "hpu"
            for tensors in self.registry.groups.values()
            for tensor in tensors
        )
        self._contig_layout = contig_layout(registry.groups) if contig_enabled() else None
        if self._contig_layout is not None and self._uses_hpu:
            _require_pinned(contig_slot(*self._contig_layout, pinned=True))  # boot refusal; the probe is freed here

    def _new_slot(self) -> dict[int, tuple[torch.Tensor, ...]]:
        if self._contig_layout is not None:
            return contig_slot(*self._contig_layout, pinned=self._uses_hpu)
        return {
            group_idx: tuple(
                torch.empty(
                    tuple(tensor.shape[1:]),
                    dtype=tensor.dtype,
                    device="cpu",
                    pin_memory=self._uses_hpu,
                )
                for tensor in tensors
            )
            for group_idx, tensors in self.registry.groups.items()
        }

    def _slot(self, block_id: int) -> dict[int, tuple[torch.Tensor, ...]]:
        slot = self._resident.get(block_id)
        if slot is None:
            slot = self._new_slot()
            self._resident[block_id] = slot
            if self._uses_hpu and not all(
                tensor.is_pinned()
                for tensors in slot.values()
                for tensor in tensors
            ):
                raise RuntimeError("HPU recurrent sidecar storage must be pinned")
        return slot

    @staticmethod
    def _new_hpu_event():
        # R436's HPU workers use torch.Event(), with record(stream), query(),
        # and synchronize().  Verify that exact API instead of assuming CUDA.
        event = torch.Event()
        for method in ("record", "query", "synchronize"):
            if not callable(getattr(event, method, None)):
                raise RuntimeError(f"torch.Event lacks required {method}() API")
        return event

    def _run(self, job_id: int, copy_fn: Callable[[], None], uses_hpu: bool) -> None:
        if job_id in self._pending:
            raise RuntimeError(f"duplicate recurrent sidecar job {job_id}")
        if not uses_hpu:
            copy_fn()
            self._pending[job_id] = _Pending(job_id, None)
            return

        if self._stream is None:
            self._stream = torch.hpu.Stream()
        current = torch.hpu.current_stream()
        self._stream.wait_stream(current)
        event = self._new_hpu_event()
        with torch.hpu.stream(self._stream):
            copy_fn()
            import habana_frameworks.torch.core as htcore

            htcore.mark_step()
            event.record(self._stream)
        # The next forward may recycle a compact row or consume a restored row.
        # Order it after this sidecar copy without a device-wide synchronize.
        current.wait_stream(self._stream)
        self._pending[job_id] = _Pending(job_id, event)

    def submit_store(self, job_id: int, req_id: str, block_id: int) -> None:
        slot = self._slot(block_id)
        rows = {
            group_idx: self.registry.resolve_row(req_id, group_idx)
            for group_idx in self.registry.groups
        }
        uses_hpu = self._uses_hpu

        def copy() -> None:
            for group_idx, tensors in self.registry.groups.items():
                row = rows[group_idx]
                for source, destination in zip(tensors, slot[group_idx]):
                    destination.copy_(source[row], non_blocking=uses_hpu)

        self._run(job_id, copy, uses_hpu)

    def submit_store_snapshot(
        self, job_id: int, req_id: str, block_id: int, snapshot_index: int, boundary: int
    ) -> int:
        """TAIL_FUSE_V1: store the bank row the fused forward wrote for (req_id, boundary).

        Validated stamps first (raises, never skips); the copy then rides the same post-forward
        fence as submit_store, so the next forward cannot overwrite the bank row before it lands.
        Returns the number of validated layers (for GLM53_TAIL_FUSE_CAPTURE).
        """
        layers = _tail_fuse.bank().validate(snapshot_index, req_id, boundary)
        sources = _snapshot_sources
        if sources is None or set(sources) != set(self.registry.groups):
            raise RuntimeError(
                "tail-fuse snapshot store without a registered bank for every compact group; "
                "VLLM_RECURRENT_TAIL_FUSE must be set on every rank at registration"
            )
        slot = self._slot(block_id)
        uses_hpu = self._uses_hpu

        def copy() -> None:
            for group_idx, bank_rows in sources.items():
                for source, destination in zip(bank_rows, slot[group_idx], strict=True):
                    destination.copy_(source[snapshot_index], non_blocking=uses_hpu)

        self._run(job_id, copy, uses_hpu)
        return layers

    def submit_load(self, job_id: int, req_id: str, block_id: int) -> None:
        # Transfer-only cache-hit steps precede normal request admission.  The
        # shared allocator reserves the row that _update_states later reuses.
        # Validate the source first so an invalid job cannot leak a row.
        if block_id not in self._resident:
            raise RuntimeError(f"missing recurrent sidecar CPU slot {block_id}")
        if self.registry.ensure_slot is None:
            raise RuntimeError("recurrent sidecar has no compact-slot allocator")
        self.registry.ensure_slot(req_id)
        rows = {
            group_idx: self.registry.resolve_row(req_id, group_idx)
            for group_idx in self.registry.groups
        }
        slot = self._resident[block_id]
        uses_hpu = self._uses_hpu

        def copy() -> None:
            for group_idx, tensors in self.registry.groups.items():
                row = rows[group_idx]
                for destination, source in zip(tensors, slot[group_idx]):
                    destination[row].copy_(source, non_blocking=uses_hpu)

        self._run(job_id, copy, uses_hpu)

    def get_finished(self) -> list[int]:
        ready: list[int] = []
        for job_id, pending in tuple(self._pending.items()):
            if pending.event is None or pending.event.query():
                ready.append(job_id)
                del self._pending[job_id]
        return ready

    def wait(self, job_ids: set[int]) -> None:
        for job_id in job_ids:
            pending = self._pending.get(job_id)
            if pending is not None and pending.event is not None:
                pending.event.synchronize()

    def shutdown(self) -> None:
        self.wait(set(self._pending))
        self._pending.clear()
        self._resident.clear()


def get_recurrent_sidecar_worker(required: bool = False) -> RecurrentSidecarWorker | None:
    if required and _worker is None:
        raise RuntimeError("recurrent sidecar transfer requested before state registration")
    return _worker


def get_recurrent_sidecar_registration(
    expected_bytes_per_rank: int,
) -> tuple[int, str] | None:
    """Return a worker-ready token only for compatible registered state."""
    if _registry is None or _worker is None:
        return None
    if _registry.bytes_per_rank != expected_bytes_per_rank:
        raise RuntimeError(
            "registered recurrent bytes disagree with MambaSpec: "
            f"{_registry.bytes_per_rank} != {expected_bytes_per_rank}"
        )
    if not _registry.groups:
        raise RuntimeError("registered recurrent state has no compact groups")
    return _registry.bytes_per_rank, _registry.signature
