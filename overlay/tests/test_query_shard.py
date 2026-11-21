"""WIDE-PROMPT-KERNEL-PLAN step 4: HPU_GLM53_QUERY_SHARD_KEYS. Each rank scores its own rows of a prompt key and all-gathers them.

Four ranks run on four threads against the real indexer, kpool, R589, R590 and Q4 gather. The fake communicator blocks like HCCL, so a rank
that skips a collective fails the test instead of passing it. Whether the card's replays agree is steps 3 and 7, not this file.
"""
from __future__ import annotations

import contextlib
import logging
import os
import sys
import threading
import unittest
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import torch
from torch.utils._python_dispatch import TorchDispatchMode

sys.path.insert(0, str(Path(__file__).resolve().parent))

import glm53_mixed_graph as graph  # noqa: E402
import glm53_prompt_census as census  # noqa: E402
import key_stack  # noqa: E402
from overlay_testlib import OVERLAY_ROOT  # noqa: E402

FLAG = "HPU_GLM53_QUERY_SHARD_KEYS"
KEYS = "VLLM_GAUDI_MIXED_PROMPT_KEYS"
TP = key_stack.TP
ROWS, TABLE = 1024, 32
START = 2000  # past the top-k window, inside the table: the scan really selects
PRE_CHANGE_R590 = Path(__file__).resolve().parent / "fixtures" / "pre_shard" / "glm53_r590_long_history.py"
# The indexer from before the decode shard: it does not import legacy_route_scan, which the r590 fixture above predates too.
PRE_DECODE_INDEXER = Path(__file__).resolve().parent / "fixtures" / "pre_shard_decode" / "hpu_glm5next_sparse_indexer.py"
COLLECTIVE_TIMEOUT_S = 120.0
REAL_BUFFER_COLUMNS = 2176  # GLM53_GS_PRIVATE shape=(8192, 2176) in the C1p card log: wider than the 2051 columns the indexer returns


@contextlib.contextmanager
def key_capture() -> Iterator[None]:
    with graph.capture_context(), graph.key_capture_context():
        yield


def row_scores(q: torch.Tensor, weights: torch.Tensor, rows: torch.Tensor, physical: torch.Tensor, valid: torch.Tensor,
               *, query_shape: tuple, shared_block_table: bool) -> torch.Tensor:
    """Every row scores differently, so a row that moved or crossed ranks shows."""
    seed = q.float().reshape(query_shape[0], -1).sum(dim=1, keepdim=True)
    return torch.sin((seed + 1.0) * physical.float()).masked_fill(~valid, -torch.inf)


class HcclFake:
    """all_gather_into_tensor for TP threads: every rank must arrive, or the barrier times out and the test fails naming it."""

    def __init__(self, ranks: int, local: threading.local) -> None:
        self.ranks, self.local = ranks, local
        self.barrier = threading.Barrier(ranks, timeout=COLLECTIVE_TIMEOUT_S)
        self.slots: dict[int, torch.Tensor] = {}
        self.counts = [0] * ranks

    def __call__(self, output: torch.Tensor, local_rows: torch.Tensor, group: object = None) -> None:
        rank = self.local.rank
        self.counts[rank] += 1
        self.slots[rank] = local_rows.clone()
        self.barrier.wait()
        output.copy_(torch.cat([self.slots[peer] for peer in range(self.ranks)], dim=0))
        self.barrier.wait()


@dataclass
class RankJob:
    """What every rank thread of one ShardWorld.run shares: the case, the scope each runs in, each rank's op, and where results and failures land."""

    case: dict
    scope: Callable
    op_for: Callable | None
    results: list
    failures: list


class ShardWorld:
    """One layer's real stack with `ranks` TP ranks on threads; `run` returns each rank's result."""

    step = staticmethod(key_stack.indexer_step)  # what a rank thread runs: (stack, case, op) -> physical rows

    def __init__(self, world: SimpleNamespace, ranks: int = TP) -> None:
        self.world, self.ranks, self.local = world, ranks, threading.local()
        self.local.rank = 0
        distributed = sys.modules["vllm.distributed"]
        distributed.get_tensor_model_parallel_rank = lambda: self.local.rank
        distributed.get_tensor_model_parallel_world_size = lambda: self.ranks
        key_stack._stub("vllm_gaudi.distributed.device_communicators.hpu_communicator", _finish_segmented_lazy_collective=lambda: None)
        world.kpool._score_pool_chunk = row_scores
        self.hccl = HcclFake(ranks, self.local)

    def rank_body(self, rank: int, job: RankJob) -> None:
        self.local.rank = rank
        try:
            with job.scope():
                job.results[rank] = self.step(self.world, job.case, job.op_for(rank) if job.op_for else None)
        except BaseException as error:  # noqa: BLE001  reported on the main thread
            job.failures.append((rank, error))

    def run(self, case: dict, scope: Callable = key_capture, op_for: Callable | None = None) -> list[torch.Tensor]:
        job = RankJob(case, scope, op_for, [None] * self.ranks, [])
        threads = [threading.Thread(target=self.rank_body, args=(rank, job)) for rank in range(self.ranks)]
        with mock.patch("torch.distributed.all_gather_into_tensor", self.hccl):
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join()
        if job.failures:
            raise RuntimeError(f"ranks failed: {job.failures}") from job.failures[0][1]
        return job.results


class ShardCase(unittest.TestCase):
    shard = f"{ROWS}:{TABLE}"

    def stack(self, shard: str | None = None, keys: str | None = None) -> contextlib.AbstractContextManager:
        return key_stack.layer_stack(OVERLAY_ROOT, {KEYS: keys or f"{ROWS}:{TABLE}", FLAG: self.shard if shard is None else shard,
                                                    "HPU_GLM53_R589_CAPTURE_MARKS": "1"})

    def replicated(self, rows: int = ROWS) -> torch.Tensor:
        with self.stack(shard="") as world:
            shards = ShardWorld(world, ranks=1)
            with mock.patch("torch.distributed.all_gather_into_tensor", shards.hccl), key_capture():
                return key_stack.indexer_step(world, key_stack.prompt_case(rows, TABLE, START))


class GateTable(ShardCase):
    """(rows, width, requests, segments, tp, in a key capture) -> rows per rank, or None for replicated."""

    def gate(self, rows: int, width: int, *, requests: int = 1, segments: int = 1, tp: int = TP, capture: bool = True) -> int | None:
        with self.stack(shard="1024:8192,512:8192,256:8192,128:8192,640:8192,1024:1024,2048:8192") as world:
            ShardWorld(world, ranks=tp)
            with key_capture() if capture else contextlib.nullcontext():
                shard = world.r590.query_shard(rows, width, requests, segments)
            return None if shard is None else shard.shard_rows

    def test_divisible_listed_keys_shard(self) -> None:
        for rows, width, per_rank in ((1024, 8192, 256), (1024, 1024, 256), (512, 8192, 128), (2048, 8192, 512)):
            with self.subTest(rows=rows, width=width):
                self.assertEqual(self.gate(rows, width), per_rank)

    def test_under_128_rows_per_rank_stays_replicated(self) -> None:
        """Below 128 per rank a tile leaves the k512 route (logits.shape[0] == 128, KEY_MEMBER_MERGE_ROWS)."""
        for rows in (128, 256, 640):
            with self.subTest(rows=rows):
                self.assertIsNone(self.gate(rows, 8192))

    def test_a_key_not_listed_stays_replicated(self) -> None:
        self.assertIsNone(self.gate(1024, 4096))

    def test_outside_a_key_capture_stays_replicated(self) -> None:
        self.assertIsNone(self.gate(1024, 8192, capture=False))

    def test_several_segments_or_requests_stay_replicated(self) -> None:
        self.assertIsNone(self.gate(1024, 8192, segments=2))
        self.assertIsNone(self.gate(1024, 8192, requests=4))

    def test_one_rank_has_nothing_to_share(self) -> None:
        self.assertIsNone(self.gate(1024, 8192, tp=1))

    def test_the_gate_runs_no_device_op_and_reads_no_tp_coordinate_while_unlisted(self) -> None:
        with self.stack(shard="") as world, key_capture():
            sys.modules["vllm.distributed"].get_tensor_model_parallel_rank = mock.Mock(side_effect=AssertionError("read the rank"))
            self.assertIsNone(world.r590.query_shard(1024, 8192, 1, 1))


class ShardRanges(ShardCase):
    def test_every_rank_owns_consecutive_rows_in_rank_order(self) -> None:
        with self.stack() as world:
            spans = [(shard.start, shard.end) for shard in (world.r590.QueryShard(rank, TP, ROWS) for rank in range(TP))]
        self.assertEqual(spans, [(0, 256), (256, 512), (512, 768), (768, 1024)])


class Exactness(ShardCase):
    def run_shard(self) -> tuple[list[torch.Tensor], ShardWorld]:
        with self.stack() as world:
            shards = ShardWorld(world)
            return shards.run(key_stack.prompt_case(ROWS, TABLE, START)), shards

    def test_rank_order_concat_equals_the_replicated_rows_on_every_rank(self) -> None:
        expected = self.replicated()
        results, _ = self.run_shard()
        self.assertEqual(expected.shape, (ROWS, key_stack.WIDTH))
        self.assertEqual(expected.dtype, torch.int32)
        for rank, result in enumerate(results):
            with self.subTest(rank=rank):
                self.assertEqual(result.dtype, torch.int32)
                self.assertTrue(torch.equal(result, expected))

    def test_the_test_can_see_a_row_that_moved(self) -> None:
        """The scores really differ per row, so the equality above is not comparing identical rows."""
        expected = self.replicated()
        self.assertGreater(len({tuple(row.tolist()) for row in expected[::128]}), 4)

    def test_every_rank_runs_exactly_one_collective_per_layer(self) -> None:
        _, shards = self.run_shard()
        self.assertEqual(shards.hccl.counts, [1] * TP)

    def test_the_buffer_holds_the_gathered_rows_on_every_rank(self) -> None:
        """The card's buffer is 2,176 columns and 8,192 rows, wider and taller than the gathered rows: a whole-row copy would raise."""
        case = key_stack.prompt_case(ROWS, TABLE, START)
        with self.stack() as world:
            shards = ShardWorld(world)
            ops = [key_stack.indexer_op(world, case) for _ in range(TP)]
            for op in ops:
                op.topk_indices_buffer = torch.full((8192, REAL_BUFFER_COLUMNS), -7, dtype=torch.int32)
            results = shards.run(case, op_for=lambda rank: ops[rank])
        for op, result in zip(ops, results):
            self.assertTrue(torch.equal(op.topk_indices_buffer[:ROWS, :key_stack.WIDTH], result))
            self.assertTrue(torch.equal(op.topk_indices_buffer[:ROWS, key_stack.WIDTH:], torch.full((ROWS, REAL_BUFFER_COLUMNS - key_stack.WIDTH), -1, dtype=torch.int32)))

    def test_every_rank_updates_the_replicated_cache_for_all_rows(self) -> None:
        case = key_stack.prompt_case(ROWS, TABLE, START)
        seen: list[int] = []
        with self.stack() as world:
            shards = ShardWorld(world)
            ops = [key_stack.indexer_op(world, case) for _ in range(TP)]
            for op in ops:
                real = op._update_pooled_cache
                op._update_pooled_cache = lambda k, *rest, real=real: (seen.append(k.shape[0]), real(k, *rest))[1]
            shards.run(case, op_for=lambda rank: ops[rank])
        self.assertEqual(seen, [ROWS] * TP)

    def test_the_gathered_output_is_owned_by_the_q4_shared_owner_when_that_flag_is_on(self) -> None:
        """One owner, not one per layer: a captured graph keeps every collective output resident (hpu_glm5next_q4kv.shared_gather_owner)."""
        with key_stack.layer_stack(OVERLAY_ROOT, {KEYS: f"{ROWS}:{TABLE}", FLAG: self.shard, "HPU_GLM53_R589_CAPTURE_MARKS": "1",
                                                  "HPU_GLM53_Q4_SHARED_GATHER_OWNER": "1"}) as world:
            ShardWorld(world)
            case = key_stack.prompt_case(ROWS, TABLE, START)
            gather = lambda output, local, group=None: output.copy_(torch.cat([local] * TP, dim=0))  # noqa: E731
            with mock.patch("torch.distributed.all_gather_into_tensor", gather), key_capture():
                first, second = key_stack.indexer_step(world, case), key_stack.indexer_step(world, case)
            self.assertIs(first, second)
            self.assertEqual(list(world.q4kv._SHARED_OWNERS.values())[0].shape, (ROWS, key_stack.WIDTH))


class Replicated(ShardCase):
    def test_rows_not_divisible_stay_replicated_with_no_collective(self) -> None:
        rows = 640
        with self.stack(shard=f"{rows}:{TABLE}", keys=f"{rows}:{TABLE}") as world, self.assertLogs("vllm.r590", logging.WARNING) as logs:
            shards = ShardWorld(world, ranks=1)
            sys.modules["vllm.distributed"].get_tensor_model_parallel_world_size = lambda: TP
            with mock.patch("torch.distributed.all_gather_into_tensor", shards.hccl), key_capture():
                result = key_stack.indexer_step(world, key_stack.prompt_case(rows, TABLE, START))
        self.assertEqual(shards.hccl.counts, [0])
        self.assertEqual(result.shape[0], rows)
        self.assertEqual(sum("GLM53_QUERY_SHARD_REPLICATED" in line for line in logs.output), 1)
        self.assertFalse([line for line in logs.output if "GLM53_QUERY_SHARD_LIVE" in line])

    def test_decode_is_untouched(self) -> None:
        with self.stack() as world:
            shards = ShardWorld(world, ranks=1)
            with mock.patch("torch.distributed.all_gather_into_tensor", shards.hccl), graph.decode_key_context((16, TABLE)):
                result = key_stack.indexer_step(world, key_stack.prompt_case(ROWS, TABLE, START))
        self.assertEqual(shards.hccl.counts, [0])
        self.assertEqual(result.shape[0], ROWS)

    def test_a_live_forward_outside_a_capture_is_untouched(self) -> None:
        with self.stack() as world:
            shards = ShardWorld(world, ranks=1)
            with mock.patch("torch.distributed.all_gather_into_tensor", shards.hccl):
                key_stack.indexer_step(world, key_stack.prompt_case(ROWS, TABLE, START))
        self.assertEqual(shards.hccl.counts, [0])


class Hooks(ShardCase):
    def test_the_tile_hooks_see_a_shard_sized_tile(self) -> None:
        seen: list[tuple] = []
        with self.stack() as world:
            shards = ShardWorld(world)
            world.r590.warm_adaptive_tiles = lambda op, inputs, table: seen.append(("warm", inputs.q_values.shape[0], inputs.positions.shape[0]))
            world.r590.record_tile_prototype = lambda inputs, segment: seen.append(("record", inputs.q_values.shape[0], segment.end - segment.start))
            shards.run(key_stack.prompt_case(ROWS, TABLE, START))
        self.assertEqual(sorted(seen), sorted([("warm", 256, 256), ("record", 256, 256)] * TP))

    def test_no_tile_is_taller_than_a_shard_with_the_widened_tiles_on(self) -> None:
        """HPU_GLM53_PREFILL_TILE_ROWS=512 is on the key env: the 1024-row plan may pick 512, which a 256-row shard cannot hold."""
        tiles: list[int] = []
        with self.stack() as world:
            shards = ShardWorld(world)
            world.r590.adaptive_tile = lambda op, **kw: world.r590.AdaptiveTile(512, 1024, TABLE) if kw["repeat"] % 512 == 0 else None
            real = world.r590._score_tile
            world.r590._score_tile = lambda op, segment, span, inputs, max_length: (tiles.append(span[1] - span[0]), real(op, segment, span, inputs, max_length))[1]
            shards.run(key_stack.prompt_case(ROWS, TABLE, START))
        self.assertEqual(set(tiles), {128})
        self.assertEqual(len(tiles), TP * 2)


class CaptureFailure(ShardCase):
    def fail_gather(self, *args: object, **kwargs: object) -> None:
        raise RuntimeError("synStatus 26 PT_DEVMEM in hcclAllGather")

    def test_a_failed_gather_arms_the_bounded_exit_and_still_raises(self) -> None:
        with self.stack() as world, mock.patch.object(graph.graphs_only, "arm_exit") as arm:
            ShardWorld(world)
            with mock.patch("torch.distributed.all_gather_into_tensor", self.fail_gather), key_capture(), self.assertRaisesRegex(RuntimeError, "PT_DEVMEM"):
                key_stack.indexer_step(world, key_stack.prompt_case(ROWS, TABLE, START))
        arm.assert_called_once_with("GLM53_KEY_CAPTURE_FAILED")

    def test_a_failed_tile_arms_the_bounded_exit_and_still_raises(self) -> None:
        with self.stack() as world, mock.patch.object(graph.graphs_only, "arm_exit") as arm:
            ShardWorld(world)
            world.r590._score_tile = mock.Mock(side_effect=RuntimeError("PT_DEVMEM in a tile"))
            with key_capture(), self.assertRaisesRegex(RuntimeError, "PT_DEVMEM"):
                key_stack.indexer_step(world, key_stack.prompt_case(ROWS, TABLE, START))
        arm.assert_called_once_with("GLM53_KEY_CAPTURE_FAILED")


class LiveLine(ShardCase):
    def test_one_line_per_rank_and_key_not_one_per_layer(self) -> None:
        with self.stack() as world, self.assertLogs("vllm.r590", logging.WARNING) as logs:
            shards = ShardWorld(world)
            case = key_stack.prompt_case(ROWS, TABLE, START)
            shards.run(case)
            shards.run(case)
        lines = [line for line in logs.output if "GLM53_QUERY_SHARD_LIVE" in line]
        self.assertEqual(len(lines), TP)
        for rank, span in enumerate(("0:256", "256:512", "512:768", "768:1024")):
            self.assertTrue(any(f"GLM53_QUERY_SHARD_LIVE rows=1024 rank={rank} shard={span}" in line for line in lines), lines)


class OpRecorder(TorchDispatchMode):
    """Every torch op a forward dispatches, with its shapes: the op sequence a graph would capture."""

    def __init__(self) -> None:
        super().__init__()
        self.ops: list[tuple] = []

    def __torch_dispatch__(self, func: Callable, types: tuple, args: tuple = (), kwargs: dict | None = None) -> object:
        shapes = tuple(tuple(arg.shape) for arg in args if isinstance(arg, torch.Tensor))
        self.ops.append((func.__name__, shapes))
        return func(*args, **(kwargs or {}))


@contextlib.contextmanager
def files_before_this_step(r590_path: Path | None) -> Iterator[None]:
    """The stack with the r590 fixture of before this step (None: the overlay's own) and, with it, the indexer that does not need what that r590 lacks."""
    if r590_path is None:
        yield
        return
    with mock.patch.object(key_stack, "R590", str(r590_path)), mock.patch.object(key_stack, "INDEXER", str(PRE_DECODE_INDEXER)):
        yield


class FlagOffOpSequence(unittest.TestCase):
    """The refactor of indexer_forward and the flag's reads add nothing: same ops, same shapes, same fences, same rows as the file before this step."""

    def sequence(self, r590_path: Path | None, *, capture: bool, shard: str | None) -> tuple:
        env = {KEYS: f"{ROWS}:{TABLE}", "HPU_GLM53_R589_CAPTURE_MARKS": "1"}
        if shard is not None:
            env[FLAG] = shard
        with files_before_this_step(r590_path), key_stack.layer_stack(OVERLAY_ROOT, env) as world, OpRecorder() as recorder:
            world.kpool._score_pool_chunk = row_scores
            case = key_stack.prompt_case(ROWS, TABLE, START)
            with key_capture() if capture else contextlib.nullcontext():
                physical = key_stack.indexer_step(world, case)
            fences = [(site.split(":")[0], sync) for site, sync in world.tally.calls]
        return recorder.ops, fences, physical

    def assert_same(self, before: tuple, after: tuple) -> None:
        self.assertEqual(before[0], after[0])
        self.assertEqual(before[1], after[1])
        self.assertTrue(torch.equal(before[2], after[2]))

    def test_the_fixture_is_the_file_before_this_step(self) -> None:
        self.assertNotIn("QueryShard", PRE_CHANGE_R590.read_text(encoding="utf-8"))

    def test_in_a_key_capture_with_the_flag_unset(self) -> None:
        self.assert_same(self.sequence(PRE_CHANGE_R590, capture=True, shard=None), self.sequence(None, capture=True, shard=None))

    def test_in_a_key_capture_with_the_flag_empty(self) -> None:
        self.assert_same(self.sequence(PRE_CHANGE_R590, capture=True, shard=None), self.sequence(None, capture=True, shard=""))

    def test_outside_a_capture_even_with_the_key_listed(self) -> None:
        self.assert_same(self.sequence(PRE_CHANGE_R590, capture=False, shard=None), self.sequence(None, capture=False, shard=f"{ROWS}:{TABLE}"))


class SameSequenceAcrossRanks(ShardCase):
    def test_all_ranks_run_the_same_fences(self) -> None:
        """A rank-dependent mark_step or collective hangs TP. Threads share one tally, so each rank runs alone against a stand-in peer."""
        sequences = []
        for rank in range(TP):
            with self.stack() as world:
                shards = ShardWorld(world)
                shards.local.rank = rank
                gather = lambda output, local, group=None: output.copy_(torch.cat([local] * TP, dim=0))  # noqa: E731
                with mock.patch("torch.distributed.all_gather_into_tensor", gather), key_capture(), OpRecorder() as recorder:
                    key_stack.indexer_step(world, key_stack.prompt_case(ROWS, TABLE, START))
                sequences.append((recorder.ops, [(site.split(":")[0], sync) for site, sync in world.tally.calls]))
        for rank, sequence in enumerate(sequences):
            with self.subTest(rank=rank):
                self.assertEqual(sequence, sequences[0])


class Census(ShardCase):
    """The census has to accept the new collective: no sync=True in a capture, and its growth named for the indexer rather than Q4's fetch."""

    def report(self) -> dict:
        stats = {"InUse": 0, "MaxInUse": 0, "Limit": 8443}
        with self.stack() as world:
            ShardWorld(world)
            core = sys.modules["habana_frameworks.torch.core"]
            gather = lambda output, local, group=None: output.copy_(torch.cat([local] * TP, dim=0))  # noqa: E731
            with mock.patch("torch.distributed.all_gather_into_tensor", gather), census.MarkStepCensus(core, SimpleNamespace(memory_stats=lambda: stats)) as seen, key_capture():
                key_stack.indexer_step(world, key_stack.prompt_case(ROWS, TABLE, START))
            return seen.report(ROWS, TABLE)

    def test_no_sync_true_in_the_capture(self) -> None:
        self.assertEqual(self.report()["sync_true"], 0)

    def test_the_gather_marks_are_named_for_the_indexer(self) -> None:
        sites = self.report()["sites"]
        gathers = [site for site in sites if "hpu_glm5next_q4kv.py" in site["site"]]
        self.assertTrue(gathers, sites)
        self.assertEqual({site["op"] for site in gathers}, {"indexer"})

    def test_a_q4_gather_is_still_named_for_mla(self) -> None:
        frame_globals = {"__name__": "q4", "census": census, "sys": sys}
        code = compile("def _all_gather_into():\n    return census.site_of(sys._getframe(0))\n", "hpu_glm5next_q4kv.py", "exec")
        exec(code, frame_globals)
        self.assertEqual(frame_globals["_all_gather_into"]()[1], "sparse_mla")


class BootChecks(unittest.TestCase):
    OWNER = "HPU_GLM53_Q4_SHARED_GATHER_OWNER"
    ENV = {KEYS: "1024:8192,1024:1024", FLAG: "1024:8192", "HPU_GLM53_GRAPHS_ONLY": "1", "PT_HPU_LAZY_COLLECTIVES_HOLD_TENSORS": "0", OWNER: "1"}
    KEYED = ((1024, 8192), (1024, 1024))

    def check(self, **env: str) -> None:
        with mock.patch.dict(os.environ, {**self.ENV, **env}):
            graph.check_query_shard_flags(graph.prompt_keys())

    def test_a_good_boot_passes(self) -> None:
        self.check()

    def test_unset_and_empty_do_nothing_whatever_else_is_set(self) -> None:
        for spec in ("", None):
            environ = {KEYS: "", "HPU_GLM53_GRAPHS_ONLY": "0"}
            if spec is not None:
                environ[FLAG] = spec
            with self.subTest(spec), mock.patch.dict(os.environ, environ):
                if spec is None:
                    os.environ.pop(FLAG, None)
                graph.check_query_shard_flags(())
                self.assertEqual(graph.query_shard_keys(), ())

    def test_malformed_keys_refuse_and_name_this_flag(self) -> None:
        for spec in ("1024", "a:b", "1000:1024", "1024:1000", "1024:8192,1024:8192"):
            with self.subTest(spec), mock.patch.dict(os.environ, {FLAG: spec}), self.assertRaisesRegex(ValueError, FLAG):
                graph.query_shard_keys()

    def test_a_key_that_is_not_captured_refuses(self) -> None:
        with self.assertRaisesRegex(ValueError, "1024:4096"):
            self.check(**{FLAG: "1024:4096"})

    def test_graphs_only_is_required(self) -> None:
        with self.assertRaisesRegex(ValueError, "HPU_GLM53_GRAPHS_ONLY"):
            self.check(HPU_GLM53_GRAPHS_ONLY="0")

    def test_the_capture_hang_fix_is_required(self) -> None:
        with self.assertRaisesRegex(ValueError, "PT_HPU_LAZY_COLLECTIVES_HOLD_TENSORS"):
            self.check(PT_HPU_LAZY_COLLECTIVES_HOLD_TENSORS="1")

    def test_the_shared_gather_owner_is_required(self) -> None:
        """Without it each layer's gather output stays resident in the captured graph (glm53_r590_long_history._gather_query_shard)."""
        with self.assertRaisesRegex(ValueError, self.OWNER + "=1"):
            self.check(**{self.OWNER: "0"})
        with mock.patch.dict(os.environ, self.ENV), self.assertRaisesRegex(ValueError, self.OWNER + "=1"):
            os.environ.pop(self.OWNER)
            graph.check_query_shard_flags(graph.prompt_keys())

    def test_every_missing_setting_is_named_in_one_refusal(self) -> None:
        with self.assertRaises(ValueError) as refused:
            self.check(**{self.OWNER: "0", "HPU_GLM53_GRAPHS_ONLY": "0"})
        self.assertIn(self.OWNER, str(refused.exception))
        self.assertIn("HPU_GLM53_GRAPHS_ONLY", str(refused.exception))

    def test_the_sharded_kpool_is_refused(self) -> None:
        with self.assertRaisesRegex(ValueError, "HPU_GLM53_TP_SHARDED_KPOOL"):
            self.check(HPU_GLM53_TP_SHARDED_KPOOL="1")

    def test_the_boot_check_calls_it(self) -> None:
        with mock.patch.dict(os.environ, {KEYS: "", FLAG: "", "HPU_GLM53_DECODE_WIDTH_PAD": "0", "VLLM_GAUDI_MIXED_DECODE_WIDTHS": "", "HPU_GLM53_DECODE_PAD_CHECK_WIDTHS": ""}), \
                mock.patch.object(graph, "check_query_shard_flags") as check:
            graph.check_prompt_key_flags()
        check.assert_called_once_with(())


class TestFile(unittest.TestCase):
    def test_this_file_runs_against_the_overlay(self) -> None:
        self.assertEqual(Path(graph.__file__).resolve().parent, OVERLAY_ROOT, "set PYTHONPATH to the overlay, as run_tests.sh does")


if __name__ == "__main__":
    unittest.main()
