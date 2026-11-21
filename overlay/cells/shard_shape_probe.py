#!/usr/bin/env python3
"""One-card proof that a decode key's row split leaves the scorer's floats and the selected ids as the unsplit scan had them.

HPU_GLM53_QUERY_SHARD_DECODE_KEYS hands each of 4 ranks 1/4 of a decode key's rows, so the scorer's batch goes 16 -> 4, 8 -> 2, 4 -> 1 (and a
4-row key's scan chunk goes 16,384 -> 65,536 pools, which also changes the selection route). Prompt shards keep 128-row tiles; decode ones do
not, so nothing proven for the prompt shard transfers. For each (rows, table width, lane lengths, kind of input) this runs, on one card:
  scorer     kpool._score_pool_chunk over the whole scan: all rows at once, against each rank's rows alone, at the chunk the indexer picks for
             that batch. Reports the float32 bits (columns that differ, the largest ULP gap) apart from the ids.
  selection  op._write_topk (the indexer's own scan, which the legacy route calls): all rows at once, against each rank's rows alone, joined
             rank-major. Reports int32 equality, the only thing that crosses the gather.
The whole key is scored twice first (null): a card that disagrees with itself run to run proves nothing, and the case reads NOT_MEASURED.
Inputs, kind=ladder: random E4M3 keys and queries from the real quantizer, with each key's scale chosen so ~640 pools score within a few ULPs of
one value and the 512th best falls among them, so a 1-ULP change in one column changes the selection. kind=random: the quantizer's own scales
(continuous scores, no near-ties). Each row owns its own blocks of the cache, as each request does in decode.

usage: shard_shape_probe.py [--rows 16,8,4] [--blocks 8192,4096,1024] [--profiles full,low] [--kinds ladder,random] [--seed N] [--out FILE.jsonl]
                            [--device hpu] [--env-check on|off] [--smoke]
Run it with the production scorer environment (shape-proof/run_card.sh does); --env-check refuses a run whose flags are not production's.
Exit: 0 EXACT everywhere; 1 some case selected other ids; 2 NOT_MEASURED (an error, a null that differs, a flag mismatch, or no case);
3 ids equal everywhere but some float bits differ.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace

import torch

TP = 4
POOL_SIZE, BLOCK_SIZE, HEADS, INDEX_DIM, TOPK_TOKENS = 4, 128, 32, 128, 2048
POOLS_PER_BLOCK = BLOCK_SIZE // POOL_SIZE
TOP_POOLS = TOPK_TOKENS // POOL_SIZE
SELECTED_COLUMNS = TOPK_TOKENS + POOL_SIZE - 1
PACKED_BYTES = INDEX_DIM + 4  # 128 E4M3 value bytes and a float32 scale
ROW_LENGTH_STEP = 13  # lane r holds 13 r fewer tokens than lane 0: no two lanes share a length, and none ends on a pool edge
LATTICE_ROWS, LATTICE_BLOCKS = (16, 8, 4), (8192, 4096, 1024)
PROFILES = {"full": 1.0, "low": 0.55}  # share of the table the longest lane holds (the rest of the scan is masked)
KINDS = ("ladder", "random")
ABOVE_POOLS, CLUSTER_POOLS = 128, 640  # the ladder: 128 pools far above, 640 a few ULPs apart (the 512th best is among them), the rest far below
ABOVE_FACTOR, BELOW_FACTOR, LADDER_TARGET = 4.0, 0.25, 1.0
EXACT, FLOATS_DIFFER, IDS_DIFFER, NOT_MEASURED = "EXACT", "FLOATS_DIFFER", "IDS_DIFFER", "NOT_MEASURED"
EXIT_CODES = {EXACT: 0, IDS_DIFFER: 1, NOT_MEASURED: 2, FLOATS_DIFFER: 3}
SEVERITY = (IDS_DIFFER, NOT_MEASURED, FLOATS_DIFFER, EXACT)
# What the production units run (docker inspect glm53flash-r0, 2026-11-07), as the kpool module reads it at import.
PRODUCTION_KPOOL = (
    ("_USE_NATIVE_FP8_KPOOL_SCORE", "native FP8 kpool score is off (HPU_GLM53_NATIVE_FP8_KPOOL_SCORE=1 in production)"),
    ("_USE_BF16_KPOOL_SCORE", "BF16 kpool score is off (HPU_GLM53_BF16_KPOOL_SCORE=1 in production)"),
    ("_USE_DECODE_SELECT", "decode member merge is off (HPU_GLM53_MLA_DECODESEL=1 in production)"),
    ("_USE_PARTIAL_SCAN_TOPK", "partial scan top-k is off (HPU_GLM53_PARTIAL_SCAN_TOPK=1 in production)"),
    ("_USE_SMALL_MERGE_TOPK", "small merge top-k is off (HPU_GLM53_SMALL_MERGE_TOPK=1 in production)"),
    ("_USE_FUSED_PREFILL_SELECT", "fused single-chunk select is off (HPU_GLM53_FUSED_PREFILL_SELECT=1 in production)"),
)
PRODUCTION_ENV = (
    ("HPU_GLM53_CONTEXT_CHUNK_POOLS", "65536", "context chunk pools"),
    ("HPU_GLM53_MULTI_BATCH_KPOOL_TILE_CEILING", "16384", "multi-batch tile ceiling"),
    ("HPU_GLM53_DYNAMIC_KPOOL_BUCKET", "1", "dynamic kpool bucket"),
    ("HPU_GLM53_R590_ENABLE", "1", "R590 canonical top-k"),
    ("PT_HPU_LAZY_MODE", "1", "lazy mode"),
)


@dataclass(frozen=True)
class ProbeCase:
    """``rows`` lanes in a ``blocks``-wide table, the longest holding ``profile`` of it, scored as ``kind`` inputs."""

    rows: int
    blocks: int
    profile: str
    kind: str

    @property
    def shard_rows(self) -> int:
        return self.rows // TP

    @property
    def tokens(self) -> int:
        return int(self.blocks * BLOCK_SIZE * PROFILES[self.profile])

    @property
    def label(self) -> str:
        return f"{self.rows}x{self.blocks}:{self.profile}:{self.kind}"

    def lengths(self) -> list[int]:
        return [self.tokens - ROW_LENGTH_STEP * row for row in range(self.rows)]


@dataclass(frozen=True)
class ProbeInputs:
    """What the legacy route hands _write_topk: queries, head weights, per-request tables and lengths, and the shared packed pool cache."""

    q_values: torch.Tensor
    weights: torch.Tensor
    table: torch.Tensor
    lengths: torch.Tensor
    cache: torch.Tensor

    def head(self, rows: int) -> ProbeInputs:
        return ProbeInputs(self.q_values[:rows], self.weights[:rows], self.table[:rows], self.lengths[:rows], self.cache)


@dataclass(frozen=True)
class Scan:
    """One scan as the indexer plans it for ``rows`` queries."""

    rows: int
    chunk: int
    bucket: int
    route: str

    @property
    def chunks(self) -> int:
        return -(-self.bucket // self.chunk) if self.chunk else 0


@dataclass(frozen=True)
class Comparison:
    cells: int
    differ: int
    max_ulp: int

    @classmethod
    def of(cls, first: torch.Tensor, second: torch.Tensor) -> Comparison:
        """Bit for bit: float32 scores by their int32 bits (-inf included), ids by value."""
        left = first.contiguous().view(torch.int32) if first.dtype == torch.float32 else first
        right = second.contiguous().view(torch.int32) if second.dtype == torch.float32 else second
        differ = left != right
        finite = torch.isfinite(first) & torch.isfinite(second) if first.dtype == torch.float32 else torch.ones_like(differ)
        gap = (left.long() - right.long()).abs()[differ & finite]
        return cls(differ.numel(), int(differ.sum()), int(gap.max()) if gap.numel() else 0)


@dataclass(frozen=True)
class ShapeResult:
    case: ProbeCase
    whole: Scan = field(default_factory=lambda: Scan(0, 0, 0, ""))
    split: Scan = field(default_factory=lambda: Scan(0, 0, 0, ""))
    scores: Comparison = Comparison(0, 0, 0)
    ids_rows: int = 0
    ids_rows_differ: int = 0
    null_scores_differ: int = 0
    null_ids_rows_differ: int = 0
    seconds: float = 0.0
    error: str = ""

    @property
    def verdict(self) -> str:
        if self.error or self.null_scores_differ or self.null_ids_rows_differ:
            return NOT_MEASURED
        if self.ids_rows_differ:
            return IDS_DIFFER
        return FLOATS_DIFFER if self.scores.differ else EXACT


def _row_generator(seed: int, blocks: int, row: int) -> torch.Generator:
    """One stream per (seed, table width, row): lane r's data is the same whatever the key's row count, so the lattice's keys share their lanes."""
    return torch.Generator().manual_seed((seed * 1_000_003 + blocks) * 7919 + row)


def _pack(raw_values: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    """kpool.pack_pooled_index_rows' layout from the value bytes as the device holds them: [128 E4M3 bytes | float32 scale]."""
    return torch.cat((raw_values, scales.float().reshape(-1, 1).contiguous().view(torch.uint8)), dim=1)


def _reference_sums(key_values: torch.Tensor, query: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    """Per pool, sum over heads of relu(q . k) x head weight, exactly (float64): what the scorer approximates before the key's scale."""
    dots = key_values.double() @ query.double().T
    return (dots.relu() * weights.double()).sum(dim=1)


def _ladder_scales(sums: torch.Tensor, valid: int, generator: torch.Generator) -> torch.Tensor:
    """Scales that put ~CLUSTER_POOLS pools at LADDER_TARGET (give or take the scorer's own rounding), ABOVE_POOLS at 4x and the rest at 1/4x."""
    if valid < ABOVE_POOLS + CLUSTER_POOLS + 32:
        raise ValueError(f"{valid} complete pools are too few for the ladder, which needs {ABOVE_POOLS + CLUSTER_POOLS + 32}; use profile full or a wider table")
    factor = torch.full((sums.numel(),), BELOW_FACTOR, dtype=torch.float64)
    order = torch.randperm(valid, generator=generator)
    factor[order[:ABOVE_POOLS]] = ABOVE_FACTOR
    factor[order[ABOVE_POOLS:ABOVE_POOLS + CLUSTER_POOLS]] = 1.0
    live = sums > 1e-9
    return torch.where(live, LADDER_TARGET * factor / sums.clamp_min(1e-9), torch.ones_like(sums)).float()


def _queries(stack: SimpleNamespace, generators: list[torch.Generator], device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """(E4M3 queries [rows, heads, 128] on the device, their float32 values on the host, head weights [rows, heads] with the query scale folded in)."""
    raw = torch.stack([torch.randn(HEADS, INDEX_DIM, generator=generator) for generator in generators])
    heads = torch.stack([torch.rand(HEADS, generator=generator) + 0.25 for generator in generators])
    values, scales = stack.kpool.quantize_index_fp8(raw.to(device))
    return values, values.float().cpu(), heads * scales.squeeze(-1).float().cpu()


def _key_row(stack: SimpleNamespace, generator: torch.Generator, capacity: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """(value bytes [capacity, 128] uint8, the quantizer's scales [capacity], the values as floats): the real quantizer, on the device."""
    values, scales = stack.kpool.quantize_index_fp8(torch.randn(capacity, INDEX_DIM, generator=generator).to(device))
    return values.view(torch.uint8).cpu(), scales.squeeze(-1).float().cpu(), values.float().cpu()


def build_inputs(stack: SimpleNamespace, case: ProbeCase, device: torch.device, seed: int = 0) -> ProbeInputs:
    """``case.rows`` lanes of a ``case.blocks``-wide table. Row r owns blocks [r x blocks, (r + 1) x blocks) of the cache, in a shuffled order."""
    rows, blocks = case.rows, case.blocks
    capacity = blocks * POOLS_PER_BLOCK
    generators = [_row_generator(seed, blocks, row) for row in range(rows)]
    q_values, q_floats, weights = _queries(stack, generators, device)
    lengths = torch.tensor(case.lengths(), dtype=torch.long)
    cache = torch.zeros((rows * capacity, PACKED_BYTES), dtype=torch.uint8)
    table = torch.empty((rows, blocks), dtype=torch.int32)
    for row, generator in enumerate(generators):
        table[row] = row * blocks + torch.randperm(blocks, generator=generator)
        raw, scales, floats = _key_row(stack, generator, capacity, device)
        if case.kind == "ladder":
            scales = _ladder_scales(_reference_sums(floats, q_floats[row], weights[row]), int(lengths[row]) // POOL_SIZE, generator)
        slots = (table[row].long().unsqueeze(1) * POOLS_PER_BLOCK + torch.arange(POOLS_PER_BLOCK)).reshape(-1)
        cache[slots] = _pack(raw, scales)
    return ProbeInputs(q_values, weights.to(device), table.to(device), lengths.to(device), cache.to(device))


def make_op(stack: SimpleNamespace, cache: torch.Tensor, rows: int) -> object:
    """The real HPUGlm5NextSparseIndexerOp over ``cache``, with a top-k buffer of ``rows`` rows (all the legacy route's scan needs of it)."""
    op = stack.indexer.HPUGlm5NextSparseIndexerOp(
        k_cache=SimpleNamespace(kv_cache=cache, prefix="probe.k", hpu_kernel_block_size=BLOCK_SIZE),
        tail_cache=SimpleNamespace(kv_cache=torch.zeros(64, 256, dtype=torch.bfloat16, device=cache.device), prefix="probe.tail"),
        topk_indices_buffer=torch.full((rows, SELECTED_COLUMNS), -1, dtype=torch.int32, device=cache.device),
        source_block_size=BLOCK_SIZE, pool_size=POOL_SIZE, topk_tokens=TOPK_TOKENS)
    op.drain_first_dynamic_pool = False
    return op


def _route_name(kpool: object, rows: int, chunk: int, bucket: int) -> str:
    """Which selection the scan takes, from the kpool module's own gates (host values only): fused, decode member merge (with native k512), or merge."""
    if kpool._fused_select_applies(max_complete_pools_bucket=bucket, context_chunk_pools=chunk, top_pools=TOP_POOLS):
        return "fused"
    width = min(chunk, bucket)
    if kpool._decode_select_applies(batch=rows, width=width, top_pools=TOP_POOLS, tp_size=1, shared_block_table=False):
        native = kpool._tpc_topk is not None and width in kpool._tpc_topk.PROVEN_WIDTHS
        return "decodesel+k512" if native else "decodesel"
    return "merge"


def plan_scan(stack: SimpleNamespace, op: object, blocks: int, rows: int) -> Scan:
    """The bucket and chunk _write_topk will use for ``rows`` queries over a ``blocks``-wide table. The legacy route passes the table's width x 128 as the
    longest sequence, so the scan covers the whole table whatever the lanes hold."""
    bucket = op._scan_pool_bucket(blocks, BLOCK_SIZE, blocks * BLOCK_SIZE, None)
    chunk = op._scoring_chunk(batch=rows, index_heads=HEADS, index_dim=INDEX_DIM, packed_row_bytes=PACKED_BYTES, active_pool_bucket=bucket)
    return Scan(rows, chunk, bucket, _route_name(stack.kpool, rows, chunk, bucket))


def scan_scores(stack: SimpleNamespace, inputs: ProbeInputs, rows: slice, chunk: int, bucket: int) -> torch.Tensor:
    """kpool._score_pool_chunk over the whole scan for ``rows``, ``chunk`` pools at a time as streaming_pooled_topk_candidates does: [rows, bucket] on the host."""
    kpool = stack.kpool
    queries, weights = inputs.q_values[rows], inputs.weights[rows]
    table, complete = inputs.table[rows], torch.div(inputs.lengths[rows], POOL_SIZE, rounding_mode="floor")
    operands = kpool._score_operands(queries, weights, False)
    parts = []
    for start in range(0, bucket, chunk):
        _, physical, valid = kpool._replicated_chunk_routing(start, min(chunk, bucket - start), table, complete, POOLS_PER_BLOCK)
        scores = kpool._score_pool_chunk(*operands, inputs.cache, physical, valid, query_shape=tuple(queries.shape), shared_block_table=False)
        parts.append(scores.cpu())
    return torch.cat(parts, dim=1)


def select_rows(op: object, inputs: ProbeInputs, rows: slice, scan_tokens: int) -> torch.Tensor:
    """_write_topk for ``rows`` as one scan filling the buffer from row 0, which is what a rank's share of the legacy route is."""
    count = rows.stop - rows.start
    physical = op._write_topk(slice(0, count), inputs.q_values[rows], inputs.weights[rows], inputs.table[rows], inputs.lengths[rows],
                              max_sequence_length=scan_tokens)
    # diag-tie: read what a consumer gets, never physical.cpu(). On the first run of a graph the first half of the host read of the returned tensor holds every second int32 of the right rows
    # (card, diag-tie/tie_return.py: 4102/4102 cells), while a consumer and the buffer hold the right rows. A shape is not a warm graph: all four 1-row 4x1024:full:ladder spans misread, 1025/1025
    # each (diag-tie/replay_proof.py). Both are read, and a disagreement is an error, so NOT_MEASURED.
    consumer = (physical.clone() + 0).cpu()
    buffered = op.topk_indices_buffer[:count, :physical.shape[1]].cpu().clone()
    if not torch.equal(consumer, buffered):
        raise RuntimeError(f"the returned rows and the indexer's buffer disagree on {int((consumer != buffered).any(dim=1).sum())} of {count} rows")
    return consumer


def _spans(rows: int) -> list[slice]:
    share = rows // TP
    return [slice(rank * share, (rank + 1) * share) for rank in range(TP)]


def run_case(stack: SimpleNamespace, case: ProbeCase, inputs: ProbeInputs) -> ShapeResult:
    started = time.monotonic()
    scan_tokens = case.blocks * BLOCK_SIZE
    whole_op, split_op = make_op(stack, inputs.cache, case.rows), make_op(stack, inputs.cache, case.shard_rows)
    whole, split = plan_scan(stack, whole_op, case.blocks, case.rows), plan_scan(stack, split_op, case.blocks, case.shard_rows)
    everyone = slice(0, case.rows)
    scores = scan_scores(stack, inputs, everyone, whole.chunk, whole.bucket)
    ids = select_rows(whole_op, inputs, everyone, scan_tokens)
    null_scores = Comparison.of(scores, scan_scores(stack, inputs, everyone, whole.chunk, whole.bucket))
    null_ids = (ids != select_rows(whole_op, inputs, everyone, scan_tokens)).any(dim=1)
    split_scores = torch.cat([scan_scores(stack, inputs, span, split.chunk, split.bucket) for span in _spans(case.rows)])
    split_ids = torch.cat([select_rows(split_op, inputs, span, scan_tokens) for span in _spans(case.rows)])
    return ShapeResult(case, whole, split, Comparison.of(scores, split_scores), case.rows, int((ids != split_ids).any(dim=1).sum()),
                       null_scores.differ, int(null_ids.sum()), time.monotonic() - started)


def failed(case: ProbeCase, error: Exception) -> ShapeResult:
    return ShapeResult(case, error=f"{type(error).__name__}: {error}")


def _input_groups(cases: Sequence[ProbeCase]) -> dict[tuple[int, str, str], list[ProbeCase]]:
    groups: dict[tuple[int, str, str], list[ProbeCase]] = {}
    for case in cases:
        groups.setdefault((case.blocks, case.profile, case.kind), []).append(case)
    return groups


def run_proof(stack: SimpleNamespace, cases: Sequence[ProbeCase], device: torch.device, seed: int = 0,
              emit: Callable[[ShapeResult], None] | None = None) -> list[ShapeResult]:
    """Every case, in order; a case that raises is a NOT_MEASURED result of its own and the others still run. Inputs are built once per
    (table width, profile, kind) for the tallest key asked, and each key takes its first rows."""
    results: dict[ProbeCase, ShapeResult] = {}
    for (blocks, profile, kind), group in _input_groups(cases).items():
        try:
            inputs = build_inputs(stack, ProbeCase(max(case.rows for case in group), blocks, profile, kind), device, seed)
        except Exception as error:  # noqa: BLE001  every case of the group reports it
            built = {case: failed(case, error) for case in group}
        else:
            built = {case: _run_or_fail(stack, case, inputs.head(case.rows)) for case in group}
        for case, result in built.items():
            results[case] = result
            if emit:
                emit(result)
    return [results[case] for case in cases]


def _run_or_fail(stack: SimpleNamespace, case: ProbeCase, inputs: ProbeInputs) -> ShapeResult:
    try:
        return run_case(stack, case, inputs)
    except Exception as error:  # noqa: BLE001  the case reports it; one failed shape must not lose the others
        return failed(case, error)


def _equal_word(differ: int) -> str:
    return "DIFFER" if differ else "EQUAL"


def row_line(result: ShapeResult) -> str:
    if result.error:
        return f"SHARD_SHAPE_ROW case={result.case.label} verdict={NOT_MEASURED} error={result.error!r}"
    whole, split = result.whole, result.split
    return (f"SHARD_SHAPE_ROW case={result.case.label} rows={whole.rows}->{split.rows} chunk={whole.chunk}->{split.chunk} bucket={whole.bucket} "
            f"chunks={whole.chunks}->{split.chunks} route={whole.route}->{split.route} "
            f"null={_equal_word(result.null_scores_differ + result.null_ids_rows_differ)} score_bits={_equal_word(result.scores.differ)} "
            f"cols_differ={result.scores.differ}/{result.scores.cells} max_ulp={result.scores.max_ulp} ids={_equal_word(result.ids_rows_differ)} "
            f"id_rows_differ={result.ids_rows_differ}/{result.ids_rows} verdict={result.verdict} seconds={result.seconds:.1f}")


def exit_code(results: Iterable[object]) -> int:
    verdicts = {result.verdict for result in results}
    worst = next((verdict for verdict in SEVERITY if verdict in verdicts), NOT_MEASURED)
    return EXIT_CODES[worst]


def summary_line(results: Sequence[object]) -> str:
    verdicts = [result.verdict for result in results]
    worst = next((verdict for verdict in SEVERITY if verdict in verdicts), NOT_MEASURED)
    counts = " ".join(f"{name.lower()}={verdicts.count(name)}" for name in (EXACT, FLOATS_DIFFER, IDS_DIFFER, NOT_MEASURED))
    return f"SHARD_SHAPE_PROOF status={worst} cases={len(verdicts)} {counts}"


def env_mismatches(stack: SimpleNamespace) -> list[str]:
    """What differs from the production scorer, one line each; a probe on another scorer proves nothing about the shard."""
    found = [message for name, message in PRODUCTION_KPOOL if not getattr(stack.kpool, name, False)]
    if getattr(stack.kpool, "_tpc_topk", None) is None:
        found.append("native k512 top-k module is not loaded (HPU_GLM53_INDEXER_TPC_TOPK=1 in production)")
    for name, want, label in PRODUCTION_ENV:
        got = os.environ.get(name)
        if got != want:
            found.append(f"{label} is {got!r} ({name}), production runs {want!r}")
    return found


def load_stack() -> SimpleNamespace:
    """The image's own kpool and indexer (a card container, the production environment)."""
    from vllm_gaudi.ops import hpu_glm5next_kpool, hpu_glm5next_sparse_indexer
    return SimpleNamespace(kpool=hpu_glm5next_kpool, indexer=hpu_glm5next_sparse_indexer)


def _numbers(text: str) -> list[int]:
    return [int(part) for part in text.split(",") if part]


def _words(text: str, allowed: Iterable[str], what: str) -> list[str]:
    found = [part for part in text.split(",") if part]
    unknown = [part for part in found if part not in allowed]
    if unknown:
        raise SystemExit(f"unknown {what} {unknown}: choose from {sorted(allowed)}")
    return found


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--rows", default=",".join(map(str, LATTICE_ROWS)), help="decode keys' row counts; each must divide by 4 (default 16,8,4)")
    parser.add_argument("--blocks", default=",".join(map(str, LATTICE_BLOCKS)), help="table widths in blocks (default 8192,4096,1024)")
    parser.add_argument("--profiles", default=",".join(PROFILES), help="lane lengths as a share of the table: full, low (default both)")
    parser.add_argument("--kinds", default=",".join(KINDS), help="ladder (near-ties at the selection boundary), random (default both)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=None, help="append one JSON line per case")
    parser.add_argument("--device", default="hpu")
    parser.add_argument("--env-check", choices=("on", "off"), default="on", help="refuse a scorer that is not the production one (default on)")
    parser.add_argument("--smoke", action="store_true", help="the smallest run that still changes a chunk: rows 16,4; width 1024; full; both kinds")
    return parser


def selected_cases(args: argparse.Namespace) -> list[ProbeCase]:
    rows, blocks = ([16, 4], [1024]) if args.smoke else (_numbers(args.rows), _numbers(args.blocks))
    profiles = ["full"] if args.smoke else _words(args.profiles, PROFILES, "profile")
    for count in rows:
        if count < TP or count % TP:
            raise SystemExit(f"rows {count} does not divide by {TP} ranks: the decode shard only splits 4, 8 and 16")
    return [ProbeCase(count, width, profile, kind) for width in blocks for profile in profiles for kind in _words(args.kinds, KINDS, "kind") for count in rows]


def _record(path: Path | None, result: ShapeResult) -> None:
    if path is None:
        return
    row = {"case": result.case.label, "verdict": result.verdict, "error": result.error, "whole": result.whole.__dict__, "split": result.split.__dict__,
           "score_cells": result.scores.cells, "score_cols_differ": result.scores.differ, "score_max_ulp": result.scores.max_ulp,
           "id_rows": result.ids_rows, "id_rows_differ": result.ids_rows_differ, "null_scores_differ": result.null_scores_differ,
           "null_id_rows_differ": result.null_ids_rows_differ, "seconds": round(result.seconds, 3)}
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row) + "\n")


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    cases = selected_cases(args)
    device = torch.device(args.device)
    if device.type == "hpu":
        import habana_frameworks.torch.core  # noqa: F401  registers the device
    stack = load_stack()
    mismatches = env_mismatches(stack) if args.env_check == "on" else []
    for line in mismatches:
        print(f"SHARD_SHAPE_ENV_MISMATCH {line}", flush=True)
    if mismatches:
        print(f"{summary_line([])} reason=env_mismatch: run with the production environment, or --env-check off for a dry run", flush=True)
        return EXIT_CODES[NOT_MEASURED]

    def report(result: ShapeResult) -> None:
        print(row_line(result), flush=True)
        _record(args.out, result)

    results = run_proof(stack, cases, device, args.seed, report)
    print(summary_line(results), flush=True)
    return exit_code(results)


if __name__ == "__main__":
    sys.exit(main())
