"""CPU stand-in for one GLM-5.3 sparse layer's prompt step, from the image's bytes or the overlay's (the w-pgkfence ops_stack pattern).

Every `.device.type == "hpu"` gate is rewritten to "cpu" at load, so the lazy fences run here exactly where they run on device;
`!= "hpu"` dispatch tests are left alone so scoring takes the portable fallback, except in R589, whose only one reads the gate back
to the host. `Tally` replaces habana's mark_step and records each caller's file:line and sync flag.
"""
from __future__ import annotations

import contextlib
import logging
import os
import sys
from collections import Counter
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest import mock

import torch

from overlay_testlib import BASE_ROOT, OVERLAY_ROOT

ISOLATED_PREFIXES = ("vllm", "habana_frameworks", "glm53_r589", "glm53_r590", "glm53_finite_attribution",
                     "glm53_mla_topkfuse")
INDEXER = "vllm_gaudi/ops/hpu_glm5next_sparse_indexer.py"
KPOOL = "vllm_gaudi/ops/hpu_glm5next_kpool.py"
Q4KV = "vllm_gaudi/ops/hpu_glm5next_q4kv.py"
COMMUNICATOR = "vllm_gaudi/distributed/device_communicators/hpu_communicator.py"
R589 = "glm53_r589_bounded_prefill.py"
R590 = "glm53_r590_long_history.py"
# r0's Config.Env for every value that picks a scorer tile, a bucket, a fence or an MLA tile, with the K1 file's scan-bound override.
LAYER_ENV = {
    "PT_HPU_LAZY_MODE": "1", "HPU_GLM53_R589_ENABLE": "1", "HPU_GLM53_R589_TRACE": "0", "HPU_GLM53_R590_ENABLE": "1",
    "HPU_GLM53_R606_RELAX_TILE_DRAINS": "1", "HPU_GLM53_R616_DRAIN_SITES": "none", "HPU_GLM53_FINITE_ATTRIBUTION": "0",
    "HPU_GLM53_CONTEXT_CHUNK_POOLS": "65536", "HPU_GLM53_CONTEXT_CHUNK_BUDGET_MIB": "512",
    "HPU_GLM53_SHARED_KPOOL_CHUNK_BUDGET_MIB": "2048", "HPU_GLM53_SHARED_KPOOL_WIDE_CHUNK": "1",
    "HPU_GLM53_SHARED_REQUEST_KPOOL": "1", "HPU_GLM53_MULTI_BATCH_KPOOL_TILE_CEILING": "16384",
    "HPU_GLM53_NATIVE_FP8_KPOOL_SCORE": "1", "HPU_GLM53_NATIVE_FP8_KPOOL_BUDGET": "1", "HPU_GLM53_BF16_KPOOL_SCORE": "1",
    "HPU_GLM53_DYNAMIC_KPOOL_BUCKET": "1", "HPU_GLM53_DRAIN_FIRST_KPOOL": "1", "HPU_GLM53_COMPILED_KPOOL": "0",
    "HPU_GLM53_TP_SHARDED_KPOOL": "0", "HPU_GLM53_CAP_LEGACY_WARMUP": "0", "HPU_GLM53_WARMUP_MAX_KPOOLS": "32768",
    "HPU_GLM53_PARTIAL_SCAN_TOPK": "1", "HPU_GLM53_SMALL_MERGE_TOPK": "1", "HPU_GLM53_FUSED_PREFILL_SELECT": "1",
    "HPU_GLM53_TRUE_HOST_PREFILL_SCAN_BOUND": "0", "HPU_GLM53_TP_SHARDED_Q4KV": "1", "HPU_GLM53_Q4_TOKEN_CYCLIC": "1",
    "HPU_GLM53_Q4_BLOCK_SIZE": "128", "HPU_GLM53_Q4_PREFILL_SEGMENT_MIN_ROWS": "0",
    "HPU_GLM53_COLLECTIVE_MARKSTEP_V2_FOR": "all_reduce", "HPU_GLM53_PREFILL_TILE_ROWS": "512",
    "HPU_SPARSE_MLA_FETCH_TILE": "128", "HPU_GLM53_INDEXER_TPC_TOPK": "0", "HPU_GLM53_Q4_LARGE_PREFILL_BYPASS_GRAPH_MIN_TOKENS": "512",
    # The image's native kernels need a card to load.
    "HPU_GLM53_MLA_FWHT": "0", "HPU_GLM53_MLA_TOPKFUSE": "0", "HPU_GLM53_MLA_NATIVE_UNPACK": "0",
}
TP, BLOCK, POOL, TOPK, HEADS, INDEX_DIM, Q4_ROW_BYTES = 4, 128, 4, 2048, 32, 128, 272
WIDTH = TOPK + POOL - 1


@dataclass
class Tally:
    """Replacement mark_step: one (file:line, sync) per call, in call order."""

    calls: list[tuple[str, bool]] = field(default_factory=list)

    def mark_step(self, sync: bool = False) -> None:
        caller = sys._getframe(1)
        self.calls.append((f"{Path(caller.f_code.co_filename).name}:{caller.f_lineno}", bool(sync)))

    def syncs(self) -> int:
        return sum(sync for _, sync in self.calls)

    def files(self) -> Counter:
        return Counter((site.split(":")[0], sync) for site, sync in self.calls)


def _stub(name: str, **attributes: object) -> ModuleType:
    module = ModuleType(name)
    vars(module).update(attributes)
    sys.modules[name] = module
    return module


class FakeTP:
    """Rank-0 view of a TP4 group: rank-major copies, the real HpuCommunicator.all_gather around them."""

    def __init__(self, communicator: ModuleType) -> None:
        self.communicator = communicator
        self.device_group = object()
        self.world_size = TP

    def all_gather(self, tensor: torch.Tensor, dim: int = -1) -> torch.Tensor:
        return self.communicator.HpuCommunicator.all_gather(self, tensor, dim)


def _all_gather_into_tensor(output: torch.Tensor, tensor: torch.Tensor, group: object = None) -> None:
    output.copy_(torch.cat([tensor] * TP, dim=0))


def _install_stubs(tally: Tally, context: SimpleNamespace) -> None:
    for package in ("vllm", "vllm.compilation", "vllm.models", "vllm.models.glm5next", "vllm.models.glm5next.nvidia",
                    "vllm.v1", "vllm.v1.attention", "vllm.v1.attention.backends", "vllm.v1.attention.backends.mla",
                    "vllm.distributed.device_communicators", "vllm_gaudi", "vllm_gaudi.ops", "vllm_gaudi.v1",
                    "vllm_gaudi.v1.worker", "vllm_gaudi.distributed", "vllm_gaudi.distributed.device_communicators",
                    "habana_frameworks"):
        _stub(package, __path__=[])
    core = _stub("habana_frameworks.torch.core", mark_step=tally.mark_step, is_lazy=lambda: True)
    _stub("habana_frameworks.torch", core=core)
    _stub("glm53_finite_attribution", wrap_stage=lambda label, function: function, wrap_boundary=lambda function: function)
    _stub("vllm.compilation.breakable_cudagraph", eager_break_during_capture=lambda function: function)
    _stub("vllm.forward_context", get_forward_context=lambda: context)
    _stub("vllm.logger", init_logger=logging.getLogger)
    _stub("vllm.models.glm5next.nvidia.attention", Glm5NextIndexerCache=type("Glm5NextIndexerCache", (), {}),
          Indexer=type("Indexer", (), {}))
    _stub("vllm.v1.attention.backends.mla.indexer", DeepseekV32IndexerBackend=type("DeepseekV32IndexerBackend", (), {}))
    _stub("vllm.distributed.device_communicators.base_device_communicator",
          DeviceCommunicatorBase=type("DeviceCommunicatorBase", (), {}))
    _stub("vllm.distributed.parallel_state", GroupCoordinator=object, get_dp_group=None, get_ep_group=None,
          get_tp_group=lambda: context.tp)
    _stub("vllm.distributed", get_tp_group=lambda: context.tp, get_tensor_model_parallel_rank=lambda: 0,
          get_tensor_model_parallel_world_size=lambda: TP, __path__=[])
    _stub("vllm_gaudi.v1.worker.hpu_dp_utils", get_hpu_dp_metadata=lambda: None)


def load(name: str, path: Path, *, gather_gate: bool = False) -> ModuleType:
    source = path.read_text(encoding="utf-8").replace('.device.type == "hpu"', '.device.type == "cpu"')
    if gather_gate:
        source = source.replace('.device.type != "hpu"', '.device.type != "cpu"')
    module = ModuleType(name)
    module.__file__ = str(path)
    sys.modules[name] = module
    exec(compile(source, str(path), "exec"), vars(module))
    return module


def fast_scores(q: torch.Tensor, weights: torch.Tensor, rows: torch.Tensor, physical: torch.Tensor, valid: torch.Tensor,
                *, query_shape: tuple, shared_block_table: bool) -> torch.Tensor:
    """Scores are not what this suite counts; the portable matmul at width 8192 would take minutes."""
    return torch.ones(query_shape[0], physical.shape[-1]).masked_fill(~valid, -torch.inf)


@contextlib.contextmanager
def layer_stack(root: Path, env: dict[str, str] | None = None) -> Iterator[SimpleNamespace]:
    """The modules of one sparse layer from `root`, in dependency order under their install names; all of it undone on exit."""
    no_card = SimpleNamespace(synchronize=lambda: None)
    with mock.patch.dict(sys.modules), mock.patch.dict(os.environ, {**LAYER_ENV, **(env or {})}),             mock.patch.object(torch, "hpu", no_card, create=True):
        for name in [name for name in sys.modules if name.startswith(ISOLATED_PREFIXES)]:
            del sys.modules[name]
        tally, context = Tally(), SimpleNamespace(attn_metadata=None, tp=None)
        _install_stubs(tally, context)
        r589 = load("glm53_r589_bounded_prefill", root / R589, gather_gate=True)
        r590 = load("glm53_r590_long_history", root / R590)
        kpool = load("vllm_gaudi.ops.hpu_glm5next_kpool", root / KPOOL)
        kpool._USE_NATIVE_FP8_KPOOL_SCORE = False
        kpool._USE_BF16_KPOOL_SCORE = False
        indexer = load("vllm_gaudi.ops.hpu_glm5next_sparse_indexer", root / INDEXER)
        q4kv = load("vllm_gaudi.ops.hpu_glm5next_q4kv", root / Q4KV)
        communicator = load("communicator_base", BASE_ROOT / COMMUNICATOR)
        communicator.dist = SimpleNamespace(all_gather_into_tensor=_all_gather_into_tensor)
        context.tp = FakeTP(communicator)
        yield SimpleNamespace(tally=tally, context=context, r589=r589, r590=r590, kpool=kpool, indexer=indexer, q4kv=q4kv,
                              communicator=communicator)


def prompt_case(rows: int, columns: int, start: int = 0, seed: int = 0) -> dict[str, torch.Tensor]:
    """Prompt inputs: `rows` new tokens at token `start`, every table column allocated (the widest scan the key can have)."""
    generator = torch.Generator().manual_seed(seed)
    pool_rows = (columns + 1) * (BLOCK // POOL)
    return {"values": torch.zeros(pool_rows, INDEX_DIM), "scales": torch.ones(pool_rows, 1),
            "table": torch.randperm(columns, generator=generator).unsqueeze(0).to(torch.int32),
            "positions": torch.arange(start, start + rows, dtype=torch.int32),
            "q_values": torch.randint(-2, 3, (rows, HEADS, INDEX_DIM), generator=generator).float(),
            "weights": torch.randint(-1, 3, (rows, HEADS), generator=generator).float(),
            "k": torch.randn(rows, INDEX_DIM, generator=generator).to(torch.bfloat16),
            "gate": torch.randn(rows, INDEX_DIM, generator=generator).to(torch.bfloat16),
            "ape": torch.randn(POOL, INDEX_DIM, generator=generator).to(torch.bfloat16)}


def indexer_op(stack: SimpleNamespace, case: dict[str, torch.Tensor]) -> object:
    packed = stack.kpool.pack_pooled_index_rows(case["values"].to(torch.float8_e4m3fn), case["scales"])
    rows = case["positions"].shape[0]
    op = stack.indexer.HPUGlm5NextSparseIndexerOp(
        k_cache=SimpleNamespace(kv_cache=packed, prefix="layers.3.k", hpu_kernel_block_size=BLOCK),
        tail_cache=SimpleNamespace(kv_cache=torch.zeros(64, 256, dtype=torch.bfloat16), prefix="layers.3.tail"),
        topk_indices_buffer=torch.full((rows, WIDTH), -7, dtype=torch.int32),
        source_block_size=BLOCK, pool_size=POOL, topk_tokens=TOPK)
    op.drain_first_dynamic_pool = False
    return op


def indexer_step(stack: SimpleNamespace, case: dict[str, torch.Tensor], op: object | None = None,
                 token_mask: torch.Tensor | None = None) -> torch.Tensor:
    """The real HPUGlm5NextSparseIndexerOp.forward, legacy metadata route; returns physical top-k.

    `token_mask`: 1 for a real row, 0 for a key's padding row (default: every row real).
    """
    rows = case["positions"].shape[0]
    op = op or indexer_op(stack, case)
    stack.context.attn_metadata = SimpleNamespace(
        is_prompt=True, attn_bias=None, glm53_index_slot_mapping=torch.full((rows,), -1, dtype=torch.int64),
        glm53_tail_slot_mapping=torch.full((rows,), -1, dtype=torch.int64), glm53_index_block_table=case["table"],
        glm53_token_mask=torch.ones(rows) if token_mask is None else token_mask)
    return op.forward(torch.zeros(rows, 8), case["q_values"].to(torch.float8_e4m3fn), case["k"], case["weights"],
                      gate_score=case["gate"], compress_ape=case["ape"], positions=case["positions"])


def q4_fetch_step(stack: SimpleNamespace, physical: torch.Tensor, columns: int) -> torch.Tensor:
    """The real token-cyclic HPUQ4MLAKVCache.fetch_selected on the indexer's slots."""
    cache = torch.zeros(columns * BLOCK // TP + 1, Q4_ROW_BYTES, dtype=torch.uint8)
    return stack.q4kv.HPUQ4MLAKVCache().fetch_selected(cache, physical)


def kda_step(stack: SimpleNamespace, rows: int) -> None:
    """The real R589 kda_bounded around a kernel with the original's result shapes."""
    heads, dim = 16, 128

    def original(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, g: torch.Tensor, beta: torch.Tensor, **kwargs: object) -> tuple:
        return v.clone(), kwargs["initial_state"].clone()

    q, v = torch.zeros(1, rows, heads, dim), torch.zeros(1, rows, heads, dim)
    stack.r589.kda_bounded(original, q, q.clone(), v, q.clone(), torch.zeros(1, rows, heads), chunk_size=128,
                           initial_state=torch.zeros(1, heads, dim, dim), output_final_state=True, use_qk_l2norm=True)
