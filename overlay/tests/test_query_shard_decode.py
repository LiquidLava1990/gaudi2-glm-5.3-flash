"""HPU_GLM53_QUERY_SHARD_DECODE_KEYS: each rank scores its own rows of a decode key and all-gathers them (the decode twin of test_query_shard.py).

Four ranks run on four threads against the real indexer, kpool, R589, R590 and Q4 gather, in a decode capture (decode_key_context, which is not a
key capture and not a mixed capture). The fake communicator blocks like HCCL, so a rank that skips a collective fails the test instead of passing
it. The mutants at the end break the split and the gather on purpose and expect the same checks to fail. Whether a card's replays are faster is the
cell's job, not this file's.
"""
from __future__ import annotations

import ast
import contextlib
import logging
import os
import sys
import unittest
from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "cells"))

import glm53_mixed_graph as graph  # noqa: E402
import key_stack  # noqa: E402
import scorer_mutants  # noqa: E402
import shard_shape_probe as probe  # noqa: E402
from overlay_testlib import OVERLAY_ROOT  # noqa: E402
from test_decode_lattice import LATTICE_ENV, FakeRunner, GraphCase  # noqa: E402
from test_query_shard import HcclFake, OpRecorder, ShardWorld, row_scores  # noqa: E402

FLAG = "HPU_GLM53_QUERY_SHARD_DECODE_KEYS"
RANK_FLAG = "HPU_GLM53_INDEXER_RANK_CHECK"
TP = key_stack.TP
TABLE = 32  # blocks: 4,096 tokens, so the scan really selects (a 2,048-token window) and every lane position below fits
POSITIONS = (2100, 3500, 2500, 300, 4000, 3000, 2200, 1000, 3900, 2049, 2048, 90, 3333, 2600, 1500, 3100)
PRE_CHANGE_INDEXER = Path(__file__).resolve().parent / "fixtures" / "pre_shard_decode" / "hpu_glm5next_sparse_indexer.py"
REAL_BUFFER_COLUMNS = 2176  # GLM53_GS_PRIVATE shape=(8192, 2176) in the C1p card log


def decode_case(rows: int, lanes: int, columns: int = TABLE, seed: int = 0) -> dict[str, torch.Tensor]:
    """One decode step of a ``rows``-row key: the first ``lanes`` rows are requests (a table and a position each), the rest padding.

    Padding is what the runner builds (_make_glm53_sparse_metadata_cpu): a table of -1, a masked row, nothing written.
    """
    case = key_stack.prompt_case(rows, columns, 0, seed)
    generator = torch.Generator().manual_seed(seed + 1)
    table = torch.full((rows, columns), -1, dtype=torch.int32)
    positions = torch.zeros(rows, dtype=torch.int32)
    for lane in range(lanes):
        table[lane] = torch.randperm(columns, generator=generator).to(torch.int32)
        positions[lane] = POSITIONS[lane % len(POSITIONS)]
    return {**case, "table": table, "positions": positions, "mask": (torch.arange(rows) < lanes).float()}


def decode_step(world: SimpleNamespace, case: dict[str, torch.Tensor], op: object | None = None) -> torch.Tensor:
    """The real HPUGlm5NextSparseIndexerOp.forward on a decode step's metadata (legacy route, one token per request); returns the physical top-k."""
    rows = case["positions"].shape[0]
    op = op or key_stack.indexer_op(world, case)
    world.context.attn_metadata = SimpleNamespace(
        is_prompt=False, attn_bias=None, glm53_index_slot_mapping=torch.full((rows,), -1, dtype=torch.int64),
        glm53_tail_slot_mapping=torch.full((rows,), -1, dtype=torch.int64), glm53_index_block_table=case["table"],
        glm53_token_mask=case["mask"])
    return op.forward(torch.zeros(rows, 8), case["q_values"].to(torch.float8_e4m3fn), case["k"], case["weights"],
                      gate_score=case["gate"], compress_ape=case["ape"], positions=case["positions"])


def decode_capture(rows: int, columns: int = TABLE) -> Callable[[], contextlib.AbstractContextManager]:
    """What the lattice wraps a decode key's dummy forward in."""
    return lambda: graph.decode_key_context((rows, columns))


def standin_gather(output: torch.Tensor, local_rows: torch.Tensor, group: object = None) -> None:
    """A rank's all-gather against peers that sent the same rows: enough for tests that count ops, not for ones that check rows."""
    output.copy_(torch.cat([local_rows] * TP, dim=0))


class DecodeWorld(ShardWorld):
    """ShardWorld whose rank threads run a decode step."""

    step = staticmethod(decode_step)


def problems(results: list[torch.Tensor], expected: torch.Tensor, counts: list[int]) -> list[str]:
    """Everything the exactness tests demand of a sharded step, as a list of what is wrong; empty means every demand holds."""
    found = [f"rank {rank}: rows differ from the replicated scan" for rank, rows in enumerate(results) if not torch.equal(rows, expected)]
    if counts != [1] * len(results):
        found.append(f"collectives per rank {counts}, want exactly one each")
    return found


def near_tie_case(world: SimpleNamespace, rows: int, lanes: int, kind: str = "ladder", seed: int = 0) -> dict[str, torch.Tensor]:
    """decode_case for scores that sit a few ULPs apart where the selection stops (shard_shape_probe.build_inputs, kind ladder), in decode_step's form.

    decode_case's cache is all zeros, so every pool ties with every other and a scorer that moves a score by a ULP or swaps two columns changes nothing
    it can see. Each lane here has a cache of its own: ~640 of its pools score within a few ULPs of one value and the 512th best is among them.
    """
    inputs = probe.build_inputs(world, probe.ProbeCase(rows, TABLE, "full", kind), torch.device("cpu"), seed)
    table = inputs.table.clone()
    table[lanes:] = -1
    positions = (inputs.lengths - 1).to(torch.int32)
    positions[lanes:] = 0
    filler = key_stack.prompt_case(rows, TABLE, 0, seed)  # k, gate and ape feed the pooled-cache update, whose slots are all -1 in decode_step
    return {**filler, "table": table, "positions": positions, "mask": (torch.arange(rows) < lanes).float(), "q_values": inputs.q_values.float(),
            "weights": inputs.weights, "cache": inputs.cache}


def near_tie_op(world: SimpleNamespace, case: dict[str, torch.Tensor]) -> object:
    return probe.make_op(world, case["cache"], case["positions"].shape[0])


class DecodeCase(unittest.TestCase):
    shard = "4:32,8:32,16:32"

    def stack(self, shard: str | None = None, **env: str) -> contextlib.AbstractContextManager:
        return key_stack.layer_stack(OVERLAY_ROOT, {FLAG: self.shard if shard is None else shard, **env})

    def replicated(self, rows: int, lanes: int) -> torch.Tensor:
        with self.stack(shard="") as world:
            shards = DecodeWorld(world, ranks=1)
            with mock.patch("torch.distributed.all_gather_into_tensor", shards.hccl), decode_capture(rows)():
                return decode_step(world, decode_case(rows, lanes))

    def sharded(self, rows: int, lanes: int, **env: str) -> tuple[list[torch.Tensor], DecodeWorld]:
        with self.stack(**env) as world:
            shards = DecodeWorld(world)
            return shards.run(decode_case(rows, lanes), scope=decode_capture(rows)), shards


class Parsing(unittest.TestCase):
    def keys(self, spec: str | None) -> tuple:
        with mock.patch.dict(os.environ):
            os.environ.pop(FLAG, None)
            if spec is not None:
                os.environ[FLAG] = spec
            return graph.decode_query_shard_keys()

    def test_the_list_form_keeps_its_order(self) -> None:
        self.assertEqual(self.keys("8:4096,16:4096,4:8192"), ((8, 4096), (16, 4096), (4, 8192)))

    def test_unset_and_empty_are_no_keys(self) -> None:
        self.assertEqual((self.keys(None), self.keys("")), ((), ()))

    def test_every_decode_row_count_of_the_lattice_parses(self) -> None:
        self.assertEqual(self.keys("1:1024,2:1024,4:1024,8:1024,16:1024"), tuple((rows, 1024) for rows in graph.DECODE_ROWS))

    def test_a_bad_key_refuses_and_names_the_flag(self) -> None:
        for spec in ("8", "a:b", "3:4096", "0:4096", "32:4096", "1024:8192", "8:3000", "8:0", "8:4096,8:4096", "8:4096:1"):
            with self.subTest(spec), self.assertRaisesRegex(ValueError, FLAG):
                self.keys(spec)

    def test_a_row_count_outside_the_lattice_says_which_rows_exist(self) -> None:
        with self.assertRaisesRegex(ValueError, r"\[1, 2, 4, 8, 16\]"):
            self.keys("3:4096")


class GateTable(DecodeCase):
    """(query rows, width, requests, in a decode capture, tp) -> rows per rank, None for replicated, or a refusal."""

    def gate(self, rows: int, width: int = TABLE, *, requests: int | None = None, tp: int = TP, capture: str | None = "decode",
             shard: str | None = None) -> int | None:
        requests = rows if requests is None else requests
        with self.stack(shard=shard) as world:
            DecodeWorld(world, ranks=tp)
            scope = {"decode": lambda: graph.decode_key_context((requests, width)), "key": graph.key_capture_context, None: contextlib.nullcontext}[capture]
            with scope():
                found = world.r590.decode_query_shard(rows, width, requests)
            return None if found is None else found.shard_rows

    def test_the_rows_that_divide_by_tp_shard_one_to_four_rows_per_rank(self) -> None:
        for rows, per_rank in ((4, 1), (8, 2), (16, 4)):
            with self.subTest(rows=rows):
                self.assertEqual(self.gate(rows), per_rank)

    def test_a_key_not_listed_stays_replicated(self) -> None:
        self.assertIsNone(self.gate(8, 64))
        self.assertIsNone(self.gate(8, shard="4:32,16:32"))

    def test_outside_a_decode_capture_stays_replicated(self) -> None:
        self.assertIsNone(self.gate(8, capture=None))

    def test_a_prompt_key_capture_is_not_a_decode_key(self) -> None:
        self.assertIsNone(self.gate(8, capture="key"))

    def test_one_rank_has_nothing_to_share(self) -> None:
        self.assertIsNone(self.gate(8, tp=1))

    def test_the_gate_runs_no_device_op_and_reads_no_tp_coordinate_while_unlisted(self) -> None:
        with self.stack(shard="") as world, graph.decode_key_context((8, TABLE)):
            sys.modules["vllm.distributed"].get_tensor_model_parallel_rank = mock.Mock(side_effect=AssertionError("read the rank"))
            sys.modules["vllm.distributed"].get_tensor_model_parallel_world_size = mock.Mock(side_effect=AssertionError("read the tp size"))
            self.assertIsNone(world.r590.decode_query_shard(8, TABLE, 8))

    def test_rows_that_cannot_split_refuse_with_the_fix_instead_of_running_replicated(self) -> None:
        """One and two rows do not divide by four ranks. A padded route would hand a rank the same single row it scans without the split, plus a collective."""
        for rows in (1, 2):
            with self.subTest(rows=rows), self.assertRaisesRegex(Exception, rf"{rows}:{TABLE}.*divide.*{TP}.*{FLAG}"):
                self.gate(rows, shard=f"{rows}:{TABLE}")

    def test_more_query_rows_than_requests_refuse(self) -> None:
        """A speculative step holds several tokens per request; the split is per request row and would cut a request's tokens across ranks."""
        with self.assertRaisesRegex(Exception, rf"16 query rows.*8 requests.*{FLAG}"):
            self.gate(16, requests=8, shard="8:32")


class ShardRanges(DecodeCase):
    def test_every_rank_owns_consecutive_rows_in_rank_order(self) -> None:
        with self.stack() as world:
            spans = {rows: [(shard.start, shard.end) for shard in (world.r590.QueryShard(rank, TP, rows) for rank in range(TP))] for rows in (4, 8, 16)}
        self.assertEqual(spans[4], [(0, 1), (1, 2), (2, 3), (3, 4)])
        self.assertEqual(spans[8], [(0, 2), (2, 4), (4, 6), (6, 8)])
        self.assertEqual(spans[16], [(0, 4), (4, 8), (8, 12), (12, 16)])


class Exactness(DecodeCase):
    def test_every_rank_gets_the_replicated_rows_for_every_sharding_key_and_lane_count(self) -> None:
        """Full keys, half-full keys and one lane: lanes are the first rows, so padding falls on whichever ranks own the last rows."""
        for rows in (4, 8, 16):
            for lanes in sorted({rows, rows // 2, 1}):
                with self.subTest(rows=rows, lanes=lanes):
                    expected = self.replicated(rows, lanes)
                    results, shards = self.sharded(rows, lanes)
                    self.assertEqual(expected.shape, (rows, key_stack.WIDTH))
                    self.assertEqual(expected.dtype, torch.int32)
                    self.assertEqual(problems(results, expected, shards.hccl.counts), [])

    def test_padding_rows_are_dead_on_every_rank(self) -> None:
        results, _ = self.sharded(8, 3)
        for result in results:
            self.assertTrue(bool((result[3:] == -1).all()))
            self.assertTrue(bool((result[:3] >= 0).any(dim=1).all()))

    def test_the_test_can_see_a_row_that_moved(self) -> None:
        """Every lane scores differently, so the equality above is not comparing identical rows."""
        expected = self.replicated(8, 8)
        self.assertEqual(len({tuple(row.tolist()) for row in expected}), 8)

    def test_the_buffer_holds_the_gathered_rows_on_every_rank(self) -> None:
        """The card's buffer is 2,176 columns by 8,192 rows: wider and taller than the gathered rows."""
        case = decode_case(8, 8)
        with self.stack() as world:
            shards = DecodeWorld(world)
            ops = [key_stack.indexer_op(world, case) for _ in range(TP)]
            for op in ops:
                op.topk_indices_buffer = torch.full((8192, REAL_BUFFER_COLUMNS), -7, dtype=torch.int32)
            results = shards.run(case, scope=decode_capture(8), op_for=lambda rank: ops[rank])
        for op, result in zip(ops, results):
            self.assertTrue(torch.equal(op.topk_indices_buffer[:8, :key_stack.WIDTH], result))
            self.assertTrue(bool((op.topk_indices_buffer[:8, key_stack.WIDTH:] == -1).all()))
            self.assertTrue(bool((op.topk_indices_buffer[8:] == -7).all()))

    def test_every_rank_updates_the_pooled_cache_for_all_rows(self) -> None:
        case, seen = decode_case(8, 8), []
        with self.stack() as world:
            shards = DecodeWorld(world)
            ops = [key_stack.indexer_op(world, case) for _ in range(TP)]
            for op in ops:
                real = op._update_pooled_cache
                op._update_pooled_cache = lambda k, *rest, real=real: (seen.append(k.shape[0]), real(k, *rest))[1]
            shards.run(case, scope=decode_capture(8), op_for=lambda rank: ops[rank])
        self.assertEqual(seen, [8] * TP)

    def test_each_rank_scores_only_its_own_rows(self) -> None:
        sizes: list[int] = []
        case = decode_case(16, 16)
        with self.stack() as world:
            shards = DecodeWorld(world)
            ops = [key_stack.indexer_op(world, case) for _ in range(TP)]
            for op in ops:
                real = op._write_topk
                op._write_topk = lambda row_slice, q, *rest, real=real, **kw: (sizes.append(q.shape[0]), real(row_slice, q, *rest, **kw))[1]
            shards.run(case, scope=decode_capture(16), op_for=lambda rank: ops[rank])
        self.assertEqual(sizes, [4] * TP)

    def test_the_gathered_output_is_the_shared_owner_when_the_stream_reports_a_capture(self) -> None:
        """Given a stream that reports capturing (hpu_glm5next_q4kv._stream_capturing, patched to True here), the shard's gather writes the one owner of
        its shape, (rows, 2051) int32. Whether a captured decode graph reports capturing is NOT shown by this test and has not been seen on a card: Q4's own
        decode owners in the boot logs are all prompt-shaped (512, 513, 272). The card's boot log decides: GLM53_Q4_SHARED_OWNER shape=(R, 2051) (RUN.txt)."""
        case = decode_case(8, 8)
        with self.stack(HPU_GLM53_Q4_SHARED_GATHER_OWNER="1") as world:
            DecodeWorld(world)
            world.q4kv._stream_capturing = lambda local_rows: True
            with mock.patch("torch.distributed.all_gather_into_tensor", standin_gather), decode_capture(8)():
                first, second = decode_step(world, case), decode_step(world, case)
            self.assertIs(first, second)
            self.assertEqual(list(world.q4kv._SHARED_OWNERS.values())[0].shape, (8, key_stack.WIDTH))
            self.assertEqual(list(world.q4kv._SHARED_OWNERS.values())[0].dtype, torch.int32)

    def test_a_stream_that_does_not_report_a_capture_gets_a_fresh_output_each_step_and_no_owner(self) -> None:
        """The owner depends on the stream's capture state and nothing else (a decode capture is not a mixed capture), so a card whose decode capture
        reports no capture keeps one resident gather output per layer and step: the boot log then shows no GLM53_Q4_SHARED_OWNER shape=(R, 2051)."""
        case = decode_case(8, 8)
        with self.stack(HPU_GLM53_Q4_SHARED_GATHER_OWNER="1") as world:
            DecodeWorld(world)
            with mock.patch("torch.distributed.all_gather_into_tensor", standin_gather), decode_capture(8)():
                first, second = decode_step(world, case), decode_step(world, case)
            self.assertIsNot(first, second)
            self.assertEqual(world.q4kv._SHARED_OWNERS, {})


class TheScanLeavesTheMultiBatchChunkAtFourRows(DecodeCase):
    """Why 4 rows is the key that gains most: a rank with one row scans in the B1 chunk (65,536 pools) instead of the multi-batch ceiling (16,384)."""

    def chunk(self, batch: int) -> int:
        with self.stack(shard="") as world:
            op = key_stack.indexer_op(world, decode_case(4, 4))
            return op._scoring_chunk(batch=batch, index_heads=key_stack.HEADS, index_dim=key_stack.INDEX_DIM,
                                     packed_row_bytes=int(op.k_cache.kv_cache.shape[-1]), active_pool_bucket=8192 * 32)

    def test_one_row_per_rank_scans_in_the_b1_chunk_and_two_or_four_do_not(self) -> None:
        self.assertEqual([self.chunk(batch) for batch in (1, 2, 4, 8, 16)], [65536, 16384, 16384, 16384, 16384])


class Hooks(DecodeCase):
    def test_the_decode_forward_never_enters_the_tile_loop(self) -> None:
        """The legacy route scores in one _write_topk; the prompt shard's tile plan and adaptive-tile hooks are not part of it."""
        seen: list[str] = []
        with self.stack() as world:
            shards = DecodeWorld(world)
            world.r590.warm_adaptive_tiles = lambda *args: seen.append("warm")
            world.r590._select_local = lambda *args: seen.append("select_local")
            shards.run(decode_case(8, 8), scope=decode_capture(8))
        self.assertEqual(seen, [])


class CaptureFailure(DecodeCase):
    def fail_gather(self, *args: object, **kwargs: object) -> None:
        raise RuntimeError("synStatus 26 PT_DEVMEM in hcclAllGather")

    def test_a_failed_gather_arms_the_bounded_exit_and_still_raises(self) -> None:
        with self.stack() as world, mock.patch.object(graph.graphs_only, "arm_exit") as arm:
            DecodeWorld(world)
            with mock.patch("torch.distributed.all_gather_into_tensor", self.fail_gather), decode_capture(8)(), self.assertRaisesRegex(RuntimeError, "PT_DEVMEM"):
                decode_step(world, decode_case(8, 8))
        arm.assert_called_once_with("GLM53_KEY_CAPTURE_FAILED")

    def test_a_failed_scan_arms_the_bounded_exit_and_still_raises(self) -> None:
        with self.stack() as world, mock.patch.object(graph.graphs_only, "arm_exit") as arm:
            DecodeWorld(world)
            case = decode_case(8, 8)
            op = key_stack.indexer_op(world, case)
            op._write_topk = mock.Mock(side_effect=RuntimeError("PT_DEVMEM in the scan"))
            with decode_capture(8)(), self.assertRaisesRegex(RuntimeError, "PT_DEVMEM"):
                decode_step(world, case, op)
        arm.assert_called_once_with("GLM53_KEY_CAPTURE_FAILED")


class LiveLine(DecodeCase):
    def test_one_line_per_rank_and_key_not_one_per_layer(self) -> None:
        with self.stack() as world, self.assertLogs("vllm.r590", logging.WARNING) as logs:
            shards = DecodeWorld(world)
            case = decode_case(8, 8)
            shards.run(case, scope=decode_capture(8))
            shards.run(case, scope=decode_capture(8))
        lines = [line for line in logs.output if "GLM53_QUERY_SHARD_DECODE_LIVE" in line]
        self.assertEqual(len(lines), TP)
        for rank, span in enumerate(("0:2", "2:4", "4:6", "6:8")):
            self.assertTrue(any(f"GLM53_QUERY_SHARD_DECODE_LIVE rows=8 rank={rank} shard={span} width={TABLE} tp={TP}" in line for line in lines), lines)


class FlagOffOpSequence(unittest.TestCase):
    """With the flag unset, empty, or naming another key, the decode forward is the file before this step: same ops, shapes, fences and rows."""

    ROWS, LANES = 8, 5

    def sequence(self, indexer: Path | None, *, capture: bool, shard: str | None) -> tuple:
        env = {} if shard is None else {FLAG: shard}
        source = mock.patch.object(key_stack, "INDEXER", str(indexer)) if indexer else contextlib.nullcontext()
        with source, key_stack.layer_stack(OVERLAY_ROOT, env) as world, OpRecorder() as recorder, \
                mock.patch("torch.distributed.all_gather_into_tensor", standin_gather):
            DecodeWorld(world)  # the tensor-parallel stubs a sharded step needs; they change nothing for a replicated one
            with decode_capture(self.ROWS)() if capture else contextlib.nullcontext():
                physical = decode_step(world, decode_case(self.ROWS, self.LANES))
            fences = [(site.split(":")[0], sync) for site, sync in world.tally.calls]
        return recorder.ops, fences, physical

    def assert_same_as_before(self, **kwargs: object) -> None:
        before, after = self.sequence(PRE_CHANGE_INDEXER, **kwargs), self.sequence(None, **kwargs)
        self.assertEqual(before[0], after[0])
        self.assertEqual(before[1], after[1])
        self.assertTrue(torch.equal(before[2], after[2]))
        self.assertGreater(len(after[0]), 100, "a vacuous comparison")

    def test_the_fixture_is_the_file_before_this_step(self) -> None:
        self.assertNotIn("legacy_route_scan", PRE_CHANGE_INDEXER.read_text(encoding="utf-8"))

    def test_in_a_decode_capture_with_the_flag_unset(self) -> None:
        self.assert_same_as_before(capture=True, shard=None)

    def test_in_a_decode_capture_with_the_flag_empty(self) -> None:
        self.assert_same_as_before(capture=True, shard="")

    def test_in_a_decode_capture_naming_another_key(self) -> None:
        self.assert_same_as_before(capture=True, shard="16:4096")

    def test_outside_a_capture_even_with_the_key_listed(self) -> None:
        self.assert_same_as_before(capture=False, shard=f"{self.ROWS}:{TABLE}")

    def test_the_flag_makes_a_difference_when_it_applies(self) -> None:
        """Else the comparisons above would pass whatever the flag did."""
        with self.assertRaises(AssertionError):
            self.assertEqual(self.sequence(PRE_CHANGE_INDEXER, capture=True, shard=None)[0], self.sequence(None, capture=True, shard=f"{self.ROWS}:{TABLE}")[0])


class TheForwardChangedByOneCall(unittest.TestCase):
    """forward is pinned in test_flags_off_base.py by a digest; this says what that digest's change is: the scan call goes through one helper."""

    class Unhook(ast.NodeTransformer):
        """legacy_route_scan(self, requests, LegacyRows(a, b, c, d, e), **kw) -> self._write_topk(a, b, c, d, e, **kw), and the two imports it needed gone."""

        def visit_Call(self, node: ast.Call) -> ast.AST:
            self.generic_visit(node)
            if isinstance(node.func, ast.Name) and node.func.id == "legacy_route_scan":
                op, requests, rows = node.args
                assert isinstance(rows, ast.Call) and rows.func.id == "LegacyRows", "the scan call must hand over LegacyRows(...) as written"
                return ast.Call(func=ast.Attribute(value=op, attr="_write_topk", ctx=ast.Load()), args=rows.args, keywords=node.keywords)
            return node

        def visit_ImportFrom(self, node: ast.ImportFrom) -> ast.AST:
            node.names = [alias for alias in node.names if alias.name not in ("legacy_route_scan", "LegacyRows")]
            return node

    @staticmethod
    def forward(path: Path) -> ast.FunctionDef:
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if isinstance(node, ast.FunctionDef) and node.name == "forward":
                return node
        raise AssertionError(f"no forward in {path}")

    def test_forward_is_the_old_forward_with_the_scan_call_unwrapped(self) -> None:
        now = self.forward(OVERLAY_ROOT / key_stack.INDEXER)
        unhooked = ast.fix_missing_locations(self.Unhook().visit(now))
        self.assertEqual(ast.dump(unhooked), ast.dump(self.forward(PRE_CHANGE_INDEXER)))


class SameSequenceAcrossRanks(DecodeCase):
    def test_all_ranks_run_the_same_ops_and_fences(self) -> None:
        """A rank-dependent mark_step or collective hangs TP. Threads share one tally, so each rank runs alone against a stand-in peer."""
        sequences = []
        for rank in range(TP):
            with self.stack() as world:
                shards = DecodeWorld(world)
                shards.local.rank = rank
                with mock.patch("torch.distributed.all_gather_into_tensor", standin_gather), decode_capture(8)(), OpRecorder() as recorder:
                    decode_step(world, decode_case(8, 5))
                sequences.append((recorder.ops, [(site.split(":")[0], sync) for site, sync in world.tally.calls]))
        for rank, sequence in enumerate(sequences):
            with self.subTest(rank=rank):
                self.assertEqual(sequence, sequences[0])


class WhatTheGatherAddsToTheCapture(DecodeCase):
    """The collective's fences, named: the shard adds the two below to a decode capture and nothing else. The unsynced one is the flush every gather
    makes (Q4's decode fetch, fetch_selected, hpu_glm5next_q4kv.py:495-502, makes it with commit=False). The synchronous one is the commit the shard's
    gather asks for (commit=True); Q4's decode fetch does NOT ask for it, only its prompt stripes do (gather_selected_stripes, :620). So a synchronous
    mark inside a decode capture is new here, and a card has not shown that a decode graph accepts one: RUN.txt (b) is that test."""

    def marks(self, shard: str) -> list[tuple[str, bool]]:
        with self.stack(shard=shard) as world:
            DecodeWorld(world)
            with mock.patch("torch.distributed.all_gather_into_tensor", standin_gather), decode_capture(8)():
                decode_step(world, decode_case(8, 8))
            return [(site.split(":")[0], sync) for site, sync in world.tally.calls]

    def test_one_collective_costs_one_flush_and_one_commit_and_nothing_elsewhere(self) -> None:
        off, on = self.marks(""), self.marks("8:32")
        added = list(on)
        for mark in off:
            added.remove(mark)
        self.assertEqual(sorted(added), sorted([("hpu_glm5next_q4kv.py", False), ("glm53_mixed_graph.py", True)]))


class BootChecks(GraphCase):
    """The static settings (check_query_shard_decode_flags) and the tensor-parallel size (decode_shard_tp_error, asked at the lattice's admission)."""

    OWNER = "HPU_GLM53_Q4_SHARED_GATHER_OWNER"
    env = {**LATTICE_ENV, FLAG: "8:4096,16:4096,4:8192", "HPU_GLM53_GRAPHS_ONLY": "1", "PT_HPU_LAZY_COLLECTIVES_HOLD_TENSORS": "0", OWNER: "1"}

    def check(self, **env: str) -> None:
        self.set_env(**env)
        self.graph.check_query_shard_decode_flags()

    def test_a_good_boot_passes(self) -> None:
        self.check()

    def test_unset_and_empty_do_nothing_whatever_else_is_set(self) -> None:
        for spec in ("", None):
            with self.subTest(spec), mock.patch.dict(os.environ, {"HPU_GLM53_GRAPHS_ONLY": "0", self.OWNER: "0"}):
                os.environ.pop(FLAG, None)
                if spec is not None:
                    os.environ[FLAG] = spec
                self.graph.check_query_shard_decode_flags()
                self.assertEqual(self.graph.decode_query_shard_keys(), ())

    def test_a_key_outside_the_lattice_refuses_and_says_which_widths_exist(self) -> None:
        with self.assertRaisesRegex(ValueError, r"8:512 is not a decode key.*\[8192, 4096, 2048, 1024\]"):
            self.check(**{FLAG: "8:512"})

    def test_a_check_width_is_a_decode_key_at_every_row_count(self) -> None:
        self.check(**{FLAG: "8:2048,16:2048"})

    def test_a_flag_with_no_lattice_refuses(self) -> None:
        with self.assertRaisesRegex(ValueError, "VLLM_GAUDI_MIXED_DECODE_WIDTHS"):
            self.check(VLLM_GAUDI_MIXED_DECODE_WIDTHS="", HPU_GLM53_DECODE_WIDTH_PAD="0", HPU_GLM53_DECODE_PAD_CHECK_WIDTHS="")

    def test_each_setting_the_gather_needs_is_required_and_named(self) -> None:
        for name, bad in (("HPU_GLM53_GRAPHS_ONLY", "0"), ("PT_HPU_LAZY_COLLECTIVES_HOLD_TENSORS", "1"), (self.OWNER, "0"),
                          ("HPU_GLM53_TP_SHARDED_KPOOL", "1")):
            with self.subTest(name), self.assertRaisesRegex(ValueError, name):
                self.check(**{name: bad})

    def test_every_missing_setting_is_named_in_one_refusal(self) -> None:
        with self.assertRaises(ValueError) as refused:
            self.check(**{self.OWNER: "0", "HPU_GLM53_GRAPHS_ONLY": "0"})
        self.assertIn(self.OWNER, str(refused.exception))
        self.assertIn("HPU_GLM53_GRAPHS_ONLY", str(refused.exception))
        self.assertIn(f"Or unset {FLAG}", str(refused.exception))

    def test_the_prompt_flags_check_is_unchanged_in_what_it_names(self) -> None:
        """The refactor that shares the four collective settings keeps the prompt flag's refusal."""
        with mock.patch.dict(os.environ, {"VLLM_GAUDI_MIXED_PROMPT_KEYS": "1024:8192", "HPU_GLM53_QUERY_SHARD_KEYS": "1024:8192", self.OWNER: "0",
                                          "HPU_GLM53_GRAPHS_ONLY": "0", "PT_HPU_LAZY_COLLECTIVES_HOLD_TENSORS": "1", "HPU_GLM53_TP_SHARDED_KPOOL": "1"}):
            with self.assertRaises(ValueError) as refused:
                self.graph.check_query_shard_flags(self.graph.prompt_keys())
        for name in ("HPU_GLM53_GRAPHS_ONLY", self.OWNER, "PT_HPU_LAZY_COLLECTIVES_HOLD_TENSORS", "HPU_GLM53_TP_SHARDED_KPOOL"):
            self.assertIn(name, str(refused.exception))

    def tp_error(self, spec: str, tp: int) -> str:
        self.set_env(**{FLAG: spec})
        return self.graph.decode_shard_tp_error(tp)

    def test_rows_that_divide_by_the_tp_size_pass(self) -> None:
        self.assertEqual(self.tp_error("4:4096,8:4096,16:8192", 4), "")
        self.assertEqual(self.tp_error("2:4096", 2), "")

    def test_one_and_two_rows_refuse_at_tp4_and_say_which_rows_can_shard(self) -> None:
        for spec in ("1:4096", "2:1024", "8:4096,2:1024"):
            error = self.tp_error(spec, 4)
            with self.subTest(spec):
                self.assertIn(spec.split(",")[-1], error)
                self.assertIn("[4, 8, 16]", error)
                self.assertIn(FLAG, error)

    def test_a_single_rank_refuses(self) -> None:
        self.assertIn("tensor-parallel size 2 or more", self.tp_error("8:4096", 1))

    def test_the_lattice_admission_refuses_before_any_capture(self) -> None:
        runner = FakeRunner(self.graph)
        self.set_env(**{FLAG: "2:1024"})
        with mock.patch.object(self.graph, "tensor_parallel_size", return_value=4):
            error = self.graph._decode_lattice_error(runner, self.graph.decode_keys(), 0, 8452 * 1024 * 1024)
        self.assertIn("2:1024", error)
        self.assertEqual(runner.calls, [])

    def test_the_lattice_admission_is_unchanged_with_the_flag_unset(self) -> None:
        runner = FakeRunner(self.graph)
        self.set_env(**{FLAG: ""})
        with mock.patch.object(self.graph, "tensor_parallel_size", side_effect=AssertionError("asked")):
            self.assertEqual(self.graph._decode_lattice_error(runner, self.graph.decode_keys(), 0, 8452 * 1024 * 1024), "")

    def test_the_boot_check_calls_it(self) -> None:
        with mock.patch.dict(os.environ, {"VLLM_GAUDI_MIXED_PROMPT_KEYS": "", FLAG: "", "HPU_GLM53_DECODE_WIDTH_PAD": "0", "VLLM_GAUDI_MIXED_DECODE_WIDTHS": "",
                                          "HPU_GLM53_DECODE_PAD_CHECK_WIDTHS": "", "HPU_GLM53_GRAPHS_ONLY": "0", "VLLM_GAUDI_MIXED_STEP": "0"}), \
                mock.patch.object(self.graph, "check_query_shard_decode_flags") as check:
            self.graph.check_prompt_key_flags()
        check.assert_called_once_with()

    def test_the_rank_check_and_the_decode_shard_boot_together_with_one_note(self) -> None:
        """equal=1 after a gather proves nothing about the ranks, so the lines carry rows_digest= for an unsharded run to be compared with (RUN.txt)."""
        self.set_env(**{RANK_FLAG: "1"})
        with self.assertLogs(self.graph.LOG, logging.WARNING) as logs:
            self.graph.check_rank_check_flags(self.graph.prompt_keys())
        self.assertEqual(sum("GLM53_RANK_CHECK_DECODE_SHARD" in line for line in logs.output), 1, logs.output)

    def test_the_rank_check_alone_still_passes_on_a_lattice_boot(self) -> None:
        self.set_env(**{RANK_FLAG: "1", FLAG: ""})
        self.graph.check_rank_check_flags(self.graph.prompt_keys())


class Mutants(DecodeCase):
    """The checks above must fail when the split is wrong or the gather is skipped: each mutant breaks the real code and runs `problems` on it."""

    ROWS, LANES = 8, 5

    def run_mutant(self, mutate: Callable[[SimpleNamespace, DecodeWorld], None]) -> list[str]:
        expected = self.replicated(self.ROWS, self.LANES)
        with self.stack() as world:
            shards = DecodeWorld(world)
            mutate(world, shards)
            results = shards.run(decode_case(self.ROWS, self.LANES), scope=decode_capture(self.ROWS))
        return problems(results, expected, shards.hccl.counts)

    def test_the_unbroken_build_has_no_problems(self) -> None:
        self.assertEqual(self.run_mutant(lambda world, shards: None), [])

    def test_a_wrong_shard_split_is_caught(self) -> None:
        """Each rank scores the next rank's rows."""
        def shifted(world: SimpleNamespace, shards: DecodeWorld) -> None:
            patched = property(lambda self: ((self.rank + 1) % self.tp_size) * self.shard_rows)
            patcher = mock.patch.object(world.r590.QueryShard, "start", patched)
            patcher.start()
            self.addCleanup(patcher.stop)

        found = self.run_mutant(shifted)
        self.assertTrue(any("rows differ" in line for line in found), found)

    def test_a_split_that_overlaps_is_caught(self) -> None:
        """Every rank scores the first rank's rows."""
        def overlapping(world: SimpleNamespace, shards: DecodeWorld) -> None:
            patcher = mock.patch.object(world.r590.QueryShard, "start", property(lambda self: 0))
            patcher.start()
            self.addCleanup(patcher.stop)

        self.assertTrue(any("rows differ" in line for line in self.run_mutant(overlapping)))

    def test_a_skipped_gather_is_caught(self) -> None:
        """Each rank keeps its own rows and never meets its peers: the buffer holds its rows and -1 elsewhere."""
        def skipped(world: SimpleNamespace, shards: DecodeWorld) -> None:
            def no_gather(op: object, local: torch.Tensor, shard: object) -> torch.Tensor:
                rows = local.new_full((shard.rows, local.shape[1]), -1)
                rows[shard.start:shard.end] = local
                op.topk_indices_buffer[:shard.rows, :rows.shape[1]].copy_(rows)
                return rows

            world.r590._gather_query_shard = no_gather

        found = self.run_mutant(skipped)
        self.assertTrue(any("rows differ" in line for line in found), found)
        self.assertTrue(any("collectives per rank [0, 0, 0, 0]" in line for line in found), found)

    def test_a_gather_in_the_wrong_rank_order_is_caught(self) -> None:
        class Reversed(HcclFake):
            def __call__(self, output: torch.Tensor, local_rows: torch.Tensor, group: object = None) -> None:
                rank = self.local.rank
                self.counts[rank] += 1
                self.slots[rank] = local_rows.clone()
                self.barrier.wait()
                output.copy_(torch.cat([self.slots[peer] for peer in reversed(range(self.ranks))], dim=0))
                self.barrier.wait()

        def reversed_order(world: SimpleNamespace, shards: DecodeWorld) -> None:
            shards.hccl = Reversed(TP, shards.local)

        self.assertTrue(any("rows differ" in line for line in self.run_mutant(reversed_order)))

    def test_a_gather_that_runs_twice_is_caught(self) -> None:
        def twice(world: SimpleNamespace, shards: DecodeWorld) -> None:
            real = world.r590._gather_query_shard
            world.r590._gather_query_shard = lambda op, local, shard: (real(op, local, shard), real(op, local, shard))[1]

        self.assertTrue(any("collectives per rank [2, 2, 2, 2]" in line for line in self.run_mutant(twice)))


class NearTies(DecodeCase):
    """The row split through the REAL scorer on scores a few ULPs apart at the selection boundary, with a scorer that is wrong at some batch sizes.

    ShardWorld swaps kpool._score_pool_chunk for a cheap stand-in, so every suite above runs with a scorer a fault in the real one cannot reach, on zeros
    that tie. Here the real scorer is put back first. Batch-dependent 1-ULP faults and a column flip then change a rank's rows against the replicated ones.
    Whether a card's scorer is exact between batch sizes is what shape-proof/run_card.sh asks; this file shows the shard's tests can tell.
    """

    ENV = {"HPU_GLM53_CONTEXT_CHUNK_POOLS": "1024", "HPU_GLM53_MULTI_BATCH_KPOOL_TILE_CEILING": "512", "HPU_GLM53_MLA_DECODESEL": "1"}

    def world_with_the_real_scorer(self, world: SimpleNamespace, ranks: int, fault: scorer_mutants.Fault | None) -> DecodeWorld:
        real = world.kpool._score_pool_chunk
        shards = DecodeWorld(world, ranks=ranks)
        world.kpool._score_pool_chunk = real
        if fault is not None:
            patcher = scorer_mutants.installed(world.kpool, fault)
            patcher.__enter__()
            self.addCleanup(patcher.__exit__, None, None, None)
        return shards

    def near_tie_problems(self, rows: int, lanes: int, fault: scorer_mutants.Fault | None = None, kind: str = "ladder") -> list[str]:
        with self.stack(shard="", **self.ENV) as world:
            shards = self.world_with_the_real_scorer(world, 1, fault)
            case = near_tie_case(world, rows, lanes, kind)
            with mock.patch("torch.distributed.all_gather_into_tensor", shards.hccl), decode_capture(rows)():
                expected = decode_step(world, case, near_tie_op(world, case))
        with self.stack(**self.ENV) as world:
            shards = self.world_with_the_real_scorer(world, TP, fault)
            case = near_tie_case(world, rows, lanes, kind)
            ops = [near_tie_op(world, case) for _ in range(TP)]
            results = shards.run(case, scope=decode_capture(rows), op_for=lambda rank: ops[rank])
        return problems(results, expected, shards.hccl.counts)

    def test_the_real_scorer_gives_every_rank_the_replicated_rows_at_every_key_and_lane_count(self) -> None:
        for rows in (4, 8, 16):
            for lanes in sorted({rows, rows // 2, 1}):
                with self.subTest(rows=rows, lanes=lanes):
                    self.assertEqual(self.near_tie_problems(rows, lanes), [])

    def test_the_rows_are_not_all_ties(self) -> None:
        """Each lane selects among scores that differ, so a moved row or a changed score shows."""
        with self.stack(shard="", **self.ENV) as world:
            shards = self.world_with_the_real_scorer(world, 1, None)
            case = near_tie_case(world, 8, 8)
            with mock.patch("torch.distributed.all_gather_into_tensor", shards.hccl), decode_capture(8)():
                rows = decode_step(world, case, near_tie_op(world, case))
        self.assertEqual(len({tuple(row.tolist()) for row in rows}), 8)

    def test_a_scorer_that_is_wrong_at_some_batch_sizes_is_caught_for_every_key(self) -> None:
        for fault in scorer_mutants.FAULTS:
            for rows in (4, 8, 16):
                with self.subTest(fault=fault.name, rows=rows):
                    found = self.near_tie_problems(rows, rows, fault)
                    self.assertTrue(any("rows differ" in line for line in found), found)

    def test_a_fault_that_ignores_the_batch_is_not_caught_because_it_is_not_a_split_effect(self) -> None:
        for rows in (4, 8, 16):
            with self.subTest(rows=rows):
                self.assertEqual(self.near_tie_problems(rows, rows, scorer_mutants.CONTROL), [])

    def test_a_one_ulp_fault_needs_the_near_ties_to_be_caught(self) -> None:
        """Continuous scores sit far apart: the same drift moves floats the selection never reads. This is why the fixture above has a ladder."""
        for fault in (scorer_mutants.UP_EVERY_NTH_COLUMN, scorer_mutants.DOWN_EVERY_NTH_COLUMN):
            with self.subTest(fault=fault.name):
                self.assertEqual(self.near_tie_problems(8, 8, fault, kind="random"), [])


class TestFile(unittest.TestCase):
    def test_this_file_runs_against_the_overlay(self) -> None:
        self.assertEqual(Path(graph.__file__).resolve().parent, OVERLAY_ROOT, "set PYTHONPATH to the overlay, as run_tests.sh does")


if __name__ == "__main__":
    unittest.main()
