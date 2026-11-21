"""HPU_GLM53_INDEXER_RANK_CHECK=1 on decode keys (the decode twin of test_rank_check.py).

A decode key's rows are the step's lanes first and padding after, and its lane count changes every step. The check must read equal=1 for
every lane count, and never for rows no replay wrote, rows on the wrong side of the padding, or a replay that wrote nothing.
"""
from __future__ import annotations

import contextlib
import io
import logging
import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "cells"))

import glm53_mixed_graph as graph  # noqa: E402
import key_stack  # noqa: E402
import rank_check_read as reader  # noqa: E402
from overlay_testlib import OVERLAY_ROOT  # noqa: E402
from test_decode_lattice import LATTICE_ENV, GraphCase, Trimmed  # noqa: E402
from test_prompt_keys import RowForward  # noqa: E402
from test_query_shard import OpRecorder, row_scores  # noqa: E402
from test_query_shard_decode import DecodeWorld, decode_capture, decode_case, decode_step, standin_gather  # noqa: E402
from test_rank_check import COLUMNS, LAYERS, RANKS, ROWS, CheckCase, FakeModel  # noqa: E402

FLAG = "HPU_GLM53_INDEXER_RANK_CHECK"
SHARD_FLAG = "HPU_GLM53_QUERY_SHARD_DECODE_KEYS"
WIDTH = 4096
LABEL = f"decode:{ROWS}x{WIDTH}"
LONGEST = 3500


class DecodeReads(CheckCase):
    """RankCheck.after_replay on a decode key: key_rows = the key's rows, real_rows = the step's lanes."""

    def replay_lanes(self, lanes: int, seed: int = 1, start: int = LONGEST) -> list[str]:
        self.before_all()
        self.write_chunk(lanes, seed)
        return self.on_every_rank("after_replay", LABEL, start, ROWS, lanes, 1)

    def test_every_lane_count_reads_equal_and_names_the_decode_key(self) -> None:
        """A short batch is the common case: voiding it would leave the check proving only full steps."""
        for lanes in range(1, ROWS + 1):
            with self.subTest(lanes=lanes):
                lines = self.replay_lanes(lanes, seed=10 * lanes)
                self.assertEqual(len(lines), LAYERS)
                for line in lines:
                    for field in (f"key={LABEL} ", "equal=1", f"rows={ROWS} ", f"live_rows={lanes} ", f"start={LONGEST} ", f"ranks={RANKS}"):
                        self.assertIn(field, line)

    def test_a_lane_count_that_grows_and_shrinks_between_steps_is_never_stale(self) -> None:
        for step, lanes in enumerate((3, 8, 1, 5, 5, 2)):
            with self.subTest(step=step, lanes=lanes):
                self.assertTrue(all("equal=1" in line for line in self.replay_lanes(lanes, seed=100 * (step + 1))))

    def misplace(self, lanes: int, rank: int) -> list[str]:
        """A real lane's row dead and a padding row alive: the live count is right and the rows are not."""
        self.before_all()
        self.write_chunk(lanes, 20 * lanes + rank)
        self.own(rank, 2)[lanes - 1] = -1
        self.own(rank, 2)[lanes] = torch.arange(COLUMNS, dtype=torch.int32)
        return self.on_every_rank("after_replay", LABEL, LONGEST, ROWS, lanes, 1)

    def test_a_lane_row_that_is_dead_and_a_padding_row_that_is_live_are_misplaced_for_every_lane_count(self) -> None:
        for lanes in (1, 3, 5, 7):
            for rank in (0, 3):
                with self.subTest(lanes=lanes, rank=rank):
                    lines = self.misplace(lanes, rank)
                    layer_five = [line for line in lines if "layer=5 " in line]
                    self.assertEqual(len(layer_five), 1, lines)
                    self.assertIn(f"misplaced_rows=2 on rank {rank}", layer_five[0])
                    self.assertEqual(sum("equal=1" in line for line in lines), LAYERS - 1)

    def test_fewer_live_rows_than_lanes_is_voided_by_the_count(self) -> None:
        self.before_all()
        self.write_chunk(5, 3)
        self.own(1, 4)[3:] = -1
        lines = self.on_every_rank("after_replay", LABEL, LONGEST, ROWS, 5, 1)
        layer_seven = next(line for line in lines if "layer=7 " in line)
        self.assertIn("live_rows=3 on rank 1 but the chunk has 5 real rows", layer_seven)

    def test_a_replay_that_wrote_nothing_is_stale_whatever_the_lane_count(self) -> None:
        self.assertTrue(all("equal=1" in line for line in self.replay_lanes(5, seed=1)))
        self.before_all()
        lines = self.on_every_rank("after_replay", LABEL, LONGEST + 1, ROWS, 5, 1)
        self.assertEqual(len(lines), LAYERS)
        for line in lines:
            self.assertIn("NOT_RETAINED", line)
            self.assertIn("stale_rows=5 of 5 live rows", line)
            self.assertNotIn("equal=", line)

    def test_a_smaller_key_judges_only_its_own_rows(self) -> None:
        """A 4-row key writes four rows of buffers sized for the tallest key; the 8-row key's leftovers below them differ by rank and are nobody's verdict."""
        self.assertTrue(all("equal=1" in line for line in self.replay_lanes(ROWS, seed=1)))
        self.before_all()
        for rank in range(RANKS):
            for layer in range(LAYERS):
                rows = torch.arange(4 * COLUMNS, dtype=torch.int32).reshape(4, COLUMNS) + 500 + layer
                rows[3] = -1
                self.own(rank, layer)[:4] = rows
        for layer in range(LAYERS):
            self.own(2, layer)[4:] += 50
        lines = self.on_every_rank("after_replay", "decode:4x4096", 900, 4, 3, 1)
        self.assertEqual(len(lines), LAYERS)
        for line in lines:
            for field in ("equal=1", "rows=4 ", "live_rows=3 ", "rows_differ=0"):
                self.assertIn(field, line)

    def test_a_rank_that_counts_other_lanes_than_its_peers_cannot_pass(self) -> None:
        targets = [SimpleNamespace(after_replay=lambda key, start, rows, lanes, prompts, check=check, rank=rank:
                                   check.after_replay(key, start, rows, 4 if rank == 2 else lanes, prompts))
                   for rank, check in enumerate(self.checks)]
        self.before_all()
        self.write_chunk(5, 1)
        lines = self.on_every_rank("after_replay", LABEL, LONGEST, ROWS, 5, 1, targets=targets)
        self.assertTrue(all("NOT_RETAINED" in line and "live_rows=5 on rank 2 but the chunk has 4 real rows" in line for line in lines), lines)

    def test_one_rank_that_differs_in_one_row_of_a_short_batch_fails_that_layer(self) -> None:
        self.before_all()
        self.write_chunk(3, 1)
        self.own(3, 6)[1, 0] += 1
        lines = self.on_every_rank("after_replay", LABEL, LONGEST, ROWS, 3, 1)
        bad = [line for line in lines if "equal=0" in line]
        self.assertEqual(len(bad), 1)
        self.assertIn("layer=9 equal=0", bad[0])
        self.assertIn("rows_differ=1", bad[0])


class Hooks(CheckCase):
    """DecodeRankRead, one per rank over that rank's RankCheck: what the runner's three decode hooks hand it, in the runner's order."""

    KEY = (ROWS, WIDTH)

    def setUp(self) -> None:
        super().setUp()
        self.reads = [graph.DecodeRankRead() for _ in range(RANKS)]
        for read, check in zip(self.reads, self.checks):
            read.bind(check)

    @staticmethod
    def positions(lanes: int, rows: int, longest: int) -> torch.Tensor:
        found = torch.zeros((rows, 1), dtype=torch.int32)
        found[:lanes, 0] = torch.tensor([longest] + [100 + lane for lane in range(1, lanes)], dtype=torch.int32)[:lanes]
        return found

    def start_step(self, lanes: int, key: tuple[int, int] = KEY, longest: int = LONGEST, note: bool = True) -> None:
        """decode_table_width, then decode_key_metadata, on every rank (host side only: no collective yet)."""
        for rank, read in enumerate(self.reads):
            self.exchange.local.rank = rank
            if note:
                read.note_step([f"req{lane}" for lane in range(lanes)], self.positions(lanes, key[0], longest))
            read.before_replay(key)

    def serve(self, lanes: int, seed: int = 1, *, key: tuple[int, int] = KEY, longest: int = LONGEST, write: bool = True, note: bool = True) -> list[str]:
        """One serving decode step on every rank: the hooks before the forward, the replay's writes, then observe_decode_hidden's read."""
        self.start_step(lanes, key, longest, note)
        if write:
            self.write_chunk(lanes, seed)
        return self.on_every_rank("after_forward", targets=self.reads)

    def test_a_served_step_logs_one_line_per_layer_with_the_key_the_lanes_and_the_longest_position(self) -> None:
        lines = self.serve(3, longest=2500)
        self.assertEqual(len(lines), LAYERS)
        for line in lines:
            for field in (f"key={LABEL} ", "equal=1", "live_rows=3 ", "start=2500 "):
                self.assertIn(field, line)

    def test_the_longest_lane_is_the_start_whichever_lane_it_is(self) -> None:
        positions = torch.zeros((ROWS, 1), dtype=torch.int32)
        positions[:4, 0] = torch.tensor([10, 2500, 300, 40], dtype=torch.int32)
        for read in self.reads:
            read.note_step(["a", "b", "c", "d"], positions)
            read.before_replay(self.KEY)
        self.write_chunk(4, 1)
        lines = self.on_every_rank("after_forward", targets=self.reads)
        self.assertTrue(all("start=2500 " in line for line in lines), lines)

    def test_one_collective_and_one_fence_per_rank_per_read_and_none_for_the_baseline(self) -> None:
        self.serve(5, seed=1)
        self.serve(2, seed=2)
        self.assertEqual(self.exchange.calls, [2] * RANKS)
        self.assertEqual(self.fences, [2] * RANKS)

    def test_the_baseline_is_taken_before_the_replay_so_a_replay_that_wrote_nothing_is_stale(self) -> None:
        self.assertTrue(all("equal=1" in line for line in self.serve(5, seed=1)))
        lines = self.serve(5, write=False)
        self.assertTrue(all("NOT_RETAINED" in line and "stale_rows=5 of 5" in line for line in lines), lines)

    def test_a_read_clears_itself_so_a_forward_with_no_replay_before_it_reads_nothing(self) -> None:
        self.serve(5)
        lines = self.on_every_rank("after_forward", targets=self.reads)
        self.assertEqual((lines, self.exchange.calls), ([], [1] * RANKS))

    def test_a_step_whose_lanes_were_never_noted_is_not_read_and_enters_no_collective(self) -> None:
        self.assertEqual(self.serve(5, note=False), [])
        self.assertEqual(self.exchange.calls, [0] * RANKS)

    def test_lanes_noted_for_one_step_are_not_reused_for_the_next(self) -> None:
        self.serve(5)
        self.assertEqual(self.serve(5, note=False), [])
        self.assertEqual(self.exchange.calls, [1] * RANKS)

    def test_an_unbound_read_touches_nothing(self) -> None:
        read = graph.DecodeRankRead()
        with mock.patch.object(graph, "rank_reports") as reports, mock.patch.object(graph, "_rank_check_fence") as fence:
            read.note_step(["a"], self.positions(1, ROWS, 5))
            read.before_replay(self.KEY)
            read.after_forward()
        reports.assert_not_called()
        fence.assert_not_called()

    def test_each_key_is_read_a_bounded_number_of_times_and_the_others_still_are(self) -> None:
        cap = graph.DECODE_CHECK_REPLAYS_PER_KEY
        for step in range(cap + 3):
            self.serve(5, seed=step + 1)
        self.assertEqual(self.exchange.calls, [cap] * RANKS)
        lines = self.serve(2, seed=999, key=(4, WIDTH))
        self.assertEqual(len(lines), LAYERS)
        self.assertTrue(all("key=decode:4x4096 " in line for line in lines), lines)

    def test_a_replay_past_the_cap_drops_the_baseline_so_the_next_read_is_judged_against_what_the_buffers_held(self) -> None:
        """An unread replay rewrote the rows; a baseline kept from the last read would let a later replay that wrote nothing look new."""
        for step in range(graph.DECODE_CHECK_REPLAYS_PER_KEY):
            self.serve(5, seed=step + 1)
        self.assertEqual(self.serve(5, seed=500), [])
        lines = self.serve(4, write=False, key=(4, WIDTH))
        self.assertEqual(len(lines), LAYERS)
        self.assertTrue(all("NOT_RETAINED" in line and "stale_rows=4 of 4" in line for line in lines), lines)

    def test_an_armed_step_whose_forward_never_finished_is_replaced_by_the_next_one(self) -> None:
        self.start_step(5)
        lines = self.serve(3, seed=4)
        self.assertEqual(len(lines), LAYERS)
        self.assertTrue(all("equal=1" in line and "live_rows=3 " in line for line in lines), lines)


class WhatTheRunnerHooksHand(GraphCase):
    """glm53_mixed_graph.decode_table_width, decode_key_metadata and observe_decode_hidden, in the order the runner calls them."""

    env = {**LATTICE_ENV, "HPU_GLM53_DECODE_WIDTH_PAD": "0", "HPU_GLM53_DECODE_PAD_CHECK_WIDTHS": ""}
    KEY = (4, 1024)

    def setUp(self) -> None:
        super().setUp()
        self.calls: list[tuple] = []
        check = SimpleNamespace(before_replay=lambda: self.calls.append(("before",)), forget_baseline=lambda: self.calls.append(("forget",)),
                                after_replay=lambda *args: self.calls.append(("after", *args)))
        self.graph._DECODE_RANK.bind(check)

    def trimmed(self, rows: int = 4, width: int = 1024) -> Trimmed:
        return Trimmed(False, torch.zeros((rows, width)), torch.zeros(3), torch.zeros(3), torch.zeros(3), None)

    def runner_step(self, lanes: int, *, capture: bool = False, warmup: bool = False) -> None:
        positions = torch.zeros((self.KEY[0], 1), dtype=torch.int32)
        positions[:lanes, 0] = torch.tensor([500, 2500, 40, 7], dtype=torch.int32)[:lanes]
        self.graph.decode_table_width(None, ["a", "b", "c", "d"][:lanes], 1024, positions)
        scope = self.graph.decode_key_context(self.KEY) if capture else contextlib.nullcontext()
        with scope:
            self.graph.decode_key_metadata(self.trimmed(), self.KEY[0], True, warmup)
        self.graph.observe_decode_hidden(torch.zeros(self.KEY[0], 2))

    def test_a_served_step_of_a_captured_key_is_read_after_the_forward(self) -> None:
        self.graph.mark_decode_key_captured(self.KEY)
        self.runner_step(3)
        self.assertEqual(self.calls, [("before",), ("after", "decode:4x1024", 2500, 4, 3, 1)])

    def test_a_key_capture_and_a_warmup_forward_are_never_read(self) -> None:
        self.graph.mark_decode_key_captured(self.KEY)
        self.runner_step(3, capture=True)
        self.runner_step(3, warmup=True)
        self.assertEqual(self.calls, [])

    def test_a_forward_with_no_rank_check_bound_reads_nothing_and_keeps_the_metadata_identity(self) -> None:
        self.graph._DECODE_RANK.__init__()
        self.graph.mark_decode_key_captured(self.KEY)
        self.runner_step(3)
        self.assertEqual(self.calls, [])

    def test_a_table_width_with_no_lattice_is_still_the_identity(self) -> None:
        self.set_env(VLLM_GAUDI_MIXED_DECODE_WIDTHS="", HPU_GLM53_DECODE_WIDTH_PAD="0", HPU_GLM53_DECODE_PAD_CHECK_WIDTHS="")
        self.assertEqual(self.graph.decode_table_width(None, ["a"], 37, torch.zeros((1, 1))), 37)
        self.assertEqual(self.calls, [])

    def test_reset_forgets_the_bound_check_and_the_counts(self) -> None:
        self.graph.mark_decode_key_captured(self.KEY)
        self.runner_step(3)
        self.graph.reset_decode_lattice()
        self.graph.mark_decode_key_captured(self.KEY)
        self.calls.clear()
        self.runner_step(3)
        self.assertEqual(self.calls, [])


class Prepare(unittest.TestCase):
    def runner(self) -> SimpleNamespace:
        model = FakeModel()
        model._glm53_mixed_graph_forward = RowForward()
        return SimpleNamespace(model=model, model_config=SimpleNamespace(enforce_eager=False), attn_block_size=128)

    def bound_check(self, mode: str) -> object:
        runner = self.runner()
        with mock.patch.dict(os.environ, {FLAG: mode}), mock.patch.object(graph, "_rank_check_fence"), mock.patch.object(graph, "_DECODE_RANK", graph.DecodeRankRead()) as read:
            graph.prepare_rank_check(runner, ((0, 1024, 32),))
            return read.check, graph.graph_cache(runner).rank_check

    def test_the_replay_mode_binds_the_decode_read_to_the_caches_check(self) -> None:
        bound, cache_check = self.bound_check("1")
        self.assertIsNotNone(bound)
        self.assertIs(bound, cache_check)

    def test_off_and_eager_bind_nothing(self) -> None:
        for mode in ("0", "eager"):
            with self.subTest(mode):
                self.assertIsNone(self.bound_check(mode)[0])


class CaptureTargetInDecode(unittest.TestCase):
    """With the flag on, a decode capture's legacy-route indexer writes its own buffer; the captured sequence is the flag-off one."""

    def layer(self, flag: str, rows: int = ROWS, lanes: int | None = None) -> SimpleNamespace:
        lanes = rows if lanes is None else lanes
        with key_stack.layer_stack(OVERLAY_ROOT, {FLAG: flag}) as world, OpRecorder() as recorder:
            world.kpool._score_pool_chunk = row_scores
            case = decode_case(rows, lanes)
            op = key_stack.indexer_op(world, case)
            op.topk_indices_buffer = torch.full((16, key_stack.WIDTH), -7, dtype=torch.int32)  # as tall as the check's own, so both slice alike
            shared = op.topk_indices_buffer
            own = torch.full((16, key_stack.WIDTH), -3, dtype=torch.int32)
            if flag == "1":
                vars(op)[graph.RANK_CHECK_ATTR] = own
            before = len(recorder.ops)
            with decode_capture(rows)():
                physical = decode_step(world, case, op)
            fences = [(site.split(":")[0], sync) for site, sync in world.tally.calls]
            return SimpleNamespace(ops=recorder.ops[before:], fences=fences, physical=physical, own=own, shared=shared, op=op)

    def test_the_indexer_writes_its_own_rows_and_gives_the_shared_buffer_back(self) -> None:
        on = self.layer("1")
        self.assertTrue(torch.equal(on.own[:ROWS], on.physical))
        self.assertTrue(bool((on.own[ROWS:] == -3).all()), "rows below the key are the check's leftovers, untouched")
        self.assertIs(on.op.topk_indices_buffer, on.shared)
        self.assertTrue(bool((on.shared[:ROWS] == -1).all()), "the scan must not write the shared buffer")
        self.assertTrue(bool((on.shared[ROWS:] == -7).all()))

    def test_exactly_the_lane_rows_are_live_and_first_for_every_lane_count(self) -> None:
        """The check calls a row live when it holds an index >= 0 and expects the lanes first, padding after: that is the indexer's doing."""
        for lanes in (1, 3, ROWS):
            with self.subTest(lanes=lanes):
                step = self.layer("1", lanes=lanes)
                for name, rows in (("returned rows", step.physical), ("the check's own rows", step.own[:ROWS])):
                    live = (rows >= 0).any(dim=1)
                    self.assertTrue(bool(live[:lanes].all()), f"{name}: a lane's row is dead")
                    self.assertTrue(bool((rows[lanes:] == -1).all()), f"{name}: a padding row holds an index")
                    self.assertEqual(int(live.sum()), lanes)

    def test_the_op_sequence_and_rows_equal_the_flag_off_capture(self) -> None:
        off, on = self.layer("0", lanes=5), self.layer("1", lanes=5)
        self.assertEqual(off.fences, on.fences)
        self.assertTrue(torch.equal(off.physical, on.physical))
        self.assertEqual(off.ops, on.ops)

    def test_a_forward_taller_than_the_buffer_keeps_the_shared_buffer_and_says_so(self) -> None:
        op = SimpleNamespace(k_cache=SimpleNamespace(prefix="layers.3.k"), topk_indices_buffer=torch.zeros((2, 2)))
        vars(op)[graph.RANK_CHECK_ATTR] = torch.zeros((4, 2), dtype=torch.int32)
        shared = op.topk_indices_buffer
        with self.assertLogs(graph.LOG, logging.WARNING) as logs, graph.decode_key_context((8, 32)), graph.rank_check_target(op, 8):
            self.assertIs(op.topk_indices_buffer, shared)
        self.assertIn("GLM53_RANK_CHECK_TARGET_TOO_SHORT layer=3 forward_rows=8 buffer_rows=4", logs.output[0])

    def test_a_failure_gives_the_shared_buffer_back(self) -> None:
        op = SimpleNamespace(k_cache=SimpleNamespace(prefix="layers.3.k"), topk_indices_buffer=torch.zeros((8, 2)))
        vars(op)[graph.RANK_CHECK_ATTR] = torch.zeros((8, 2), dtype=torch.int32)
        shared = op.topk_indices_buffer
        with self.assertRaises(ZeroDivisionError), graph.decode_key_context((8, 32)), graph.rank_check_target(op, 8):
            self.assertIsNot(op.topk_indices_buffer, shared)
            raise ZeroDivisionError
        self.assertIs(op.topk_indices_buffer, shared)

    def test_outside_any_capture_the_shared_buffer_stays(self) -> None:
        op = SimpleNamespace(k_cache=SimpleNamespace(prefix="layers.3.k"), topk_indices_buffer=torch.zeros((8, 2)))
        vars(op)[graph.RANK_CHECK_ATTR] = torch.zeros((8, 2), dtype=torch.int32)
        shared = op.topk_indices_buffer
        with graph.rank_check_target(op, 8):
            self.assertIs(op.topk_indices_buffer, shared)


class CaptureTargetWithTheDecodeShard(unittest.TestCase):
    """Both flags on: each rank scores its own rows and gathers; its own check buffer then holds the gathered rows, so the digest logged from it is the
    digest of the rows the model reads."""

    ROWS, LANES = 8, 5

    def sharded(self, check: str) -> tuple[list[torch.Tensor], list[SimpleNamespace]]:
        case = decode_case(self.ROWS, self.LANES)
        with key_stack.layer_stack(OVERLAY_ROOT, {FLAG: check, SHARD_FLAG: f"{self.ROWS}:{key_stack.TP * 8}"}) as world:
            shards = DecodeWorld(world)
            ops = [key_stack.indexer_op(world, case) for _ in range(key_stack.TP)]
            held = []
            for op in ops:
                op.topk_indices_buffer = torch.full((16, key_stack.WIDTH), -7, dtype=torch.int32)
                own = torch.full((16, key_stack.WIDTH), -3, dtype=torch.int32)
                if check == "1":
                    vars(op)[graph.RANK_CHECK_ATTR] = own
                held.append(SimpleNamespace(own=own, shared=op.topk_indices_buffer, op=op))
            results = shards.run(case, scope=decode_capture(self.ROWS), op_for=lambda rank: ops[rank])
        return results, held

    def test_every_ranks_own_buffer_holds_the_gathered_rows_and_the_shared_buffer_is_given_back(self) -> None:
        results, held = self.sharded("1")
        for rank, (result, rank_held) in enumerate(zip(results, held)):
            with self.subTest(rank=rank):
                self.assertTrue(torch.equal(rank_held.own[:self.ROWS], result), "the check reads what the gather produced")
                self.assertTrue(bool((rank_held.own[self.ROWS:] == -3).all()), "rows below the key are the check's leftovers")
                self.assertIs(rank_held.op.topk_indices_buffer, rank_held.shared)
                self.assertTrue(bool((rank_held.shared[:self.ROWS] == -1).all()) and bool((rank_held.shared[self.ROWS:] == -7).all()))
        self.assertTrue(all(torch.equal(held[0].own, rank_held.own) for rank_held in held))

    def test_the_rows_are_the_check_off_rows(self) -> None:
        on, _ = self.sharded("1")
        off, _ = self.sharded("0")
        for rank in range(key_stack.TP):
            self.assertTrue(torch.equal(on[rank], off[rank]), rank)

    def test_the_check_adds_no_op_and_no_fence_to_the_shard(self) -> None:
        def sequence(check: str) -> tuple[list, list]:
            case = decode_case(self.ROWS, self.LANES)
            with key_stack.layer_stack(OVERLAY_ROOT, {FLAG: check, SHARD_FLAG: f"{self.ROWS}:{key_stack.TP * 8}"}) as world:
                DecodeWorld(world)
                op = key_stack.indexer_op(world, case)
                op.topk_indices_buffer = torch.full((16, key_stack.WIDTH), -7, dtype=torch.int32)  # as tall as the check's own, so both slice alike
                if check == "1":
                    vars(op)[graph.RANK_CHECK_ATTR] = torch.full((16, key_stack.WIDTH), -3, dtype=torch.int32)
                with mock.patch("torch.distributed.all_gather_into_tensor", standin_gather), decode_capture(self.ROWS)(), OpRecorder() as recorder:
                    decode_step(world, case, op)
                return recorder.ops, [(site.split(":")[0], sync) for site, sync in world.tally.calls]

        self.assertEqual(sequence("1"), sequence("0"))


PREFIX = "(Worker_TP0 pid=412) WARNING 11-07 05:41:07 [glm53_mixed_graph.py:1380]"


def decode_verdict(layer: int, key: str, start: int, replay: int, *, equal: int = 1, live: int = 5, ranks: int = 4, differ: int = 0) -> str:
    rows = int(key.split(":")[1].split("x")[0])
    return (f"{PREFIX} GLM53_RANK_CHECK key={key} layer={layer} equal={equal} rows={rows} rows_differ={differ} live_rows={live} ranks={ranks} "
            f"start={start} replay={replay}")


def decode_chunk(key: str, start: int, replay: int, skip: tuple[int, ...] = (), **kwargs: int) -> list[str]:
    return [decode_verdict(layer, key, start, replay, **kwargs) for layer in range(3, 3 + LAYERS) if layer not in skip]


def prompt_chunk(start: int, replay: int) -> list[str]:
    return [f"{PREFIX} GLM53_RANK_CHECK key=1024:8192 layer={layer} equal=1 rows=1024 rows_differ=0 live_rows=1024 ranks=4 start={start} replay={replay}"
            for layer in range(3, 3 + LAYERS)]


class ReaderOnDecodeKeys(unittest.TestCase):
    """cells/rank_check_read.py: decode chunks are judged like prompt chunks, but their start is a lane's position, not a prompt start."""

    def judge(self, lines: list[str], decode_keys: tuple[str, ...] = (), decode_only: bool = False) -> object:
        return reader.judge(lines, ranks=4, layers=LAYERS, decode=reader.DecodeAsk(decode_keys, decode_only))

    def assert_status(self, lines: list[str], status: str, mention: str = "", **kwargs: object) -> object:
        result = self.judge(lines, **kwargs)
        self.assertEqual(result.status, status, result)
        if mention:
            self.assertTrue(any(mention in reason for reason in result.reasons), (mention, result.reasons))
        return result

    def test_a_decode_only_log_passes_when_every_asked_key_has_a_complete_equal_chunk(self) -> None:
        lines = [*decode_chunk("decode:8x4096", 3500, 1), *decode_chunk("decode:16x4096", 90, 2, live=1), *decode_chunk("decode:4x8192", 700000, 3)]
        result = self.assert_status(lines, "PASS", decode_keys=("8x4096", "16x4096", "4x8192"), decode_only=True)
        self.assertIn("3 chunks", result.summary)

    def test_a_short_lane_count_is_a_reading_not_a_void(self) -> None:
        self.assert_status(decode_chunk("decode:8x4096", 90, 1, live=1), "PASS", decode_keys=("8x4096",), decode_only=True)

    def test_a_key_asked_for_and_never_read_is_not_measured(self) -> None:
        self.assert_status(decode_chunk("decode:8x4096", 3500, 1), "NOT_MEASURED", "decode key 16x4096", decode_keys=("8x4096", "16x4096"), decode_only=True)

    def test_a_key_asked_for_whose_only_chunk_is_voided_is_not_measured(self) -> None:
        lines = [*decode_chunk("decode:8x4096", 3500, 1, skip=(5,)), f"{PREFIX} GLM53_RANK_CHECK_NOT_RETAINED key=decode:8x4096 layer=5 start=3500 replay=1 "
                 "reason=misplaced_rows=2 on rank 2: a real row is dead or a padding row is live, so these rows are not this chunk's. No verdict on these rows"]
        result = self.assert_status(lines, "NOT_MEASURED", "NOT_RETAINED", decode_keys=("8x4096",), decode_only=True)
        self.assertTrue(any("decode key 8x4096" in reason for reason in result.reasons), result.reasons)

    def test_a_layer_missing_from_a_decode_chunk(self) -> None:
        self.assert_status(decode_chunk("decode:8x4096", 3500, 1, skip=(7,)), "NOT_MEASURED", f"{LAYERS - 1} of {LAYERS} layers", decode_only=True)

    def test_an_unequal_decode_layer_fails_and_names_the_key(self) -> None:
        lines = [*decode_chunk("decode:8x4096", 3500, 1, skip=(4,)), decode_verdict(4, "decode:8x4096", 3500, 1, equal=0, differ=2)]
        result = self.assert_status(lines, "FAIL", "key decode:8x4096 layer 4", decode_only=True)
        self.assertTrue(any("rows_differ=2" in reason for reason in result.reasons), result.reasons)

    def test_fewer_ranks_than_expected(self) -> None:
        self.assert_status(decode_chunk("decode:8x4096", 3500, 1, ranks=2), "NOT_MEASURED", "ranks=2", decode_only=True)

    def test_a_decode_position_is_not_a_prompt_start(self) -> None:
        """The long-context readings the default asks for are prompt chunks: a decode lane at position 600000 must not satisfy them."""
        self.assert_status(decode_chunk("decode:8x4096", 600000, 1), "NOT_MEASURED", "start >= 150000")

    def test_decode_only_asks_no_prompt_start_and_still_judges_prompt_lines_it_finds(self) -> None:
        lines = [*decode_chunk("decode:8x4096", 3500, 1), *prompt_chunk(140000, 2)]
        self.assert_status(lines, "PASS", decode_only=True)
        unequal = [*decode_chunk("decode:8x4096", 3500, 1), *prompt_chunk(140000, 2)[:-1],
                   prompt_chunk(140000, 2)[-1].replace("equal=1", "equal=0").replace("rows_differ=0", "rows_differ=3")]
        self.assert_status(unequal, "FAIL", "layer 13", decode_only=True)

    def test_a_prompt_and_decode_log_passes_with_both_asked(self) -> None:
        lines = [*prompt_chunk(160000, 1), *prompt_chunk(540000, 2), *decode_chunk("decode:8x4096", 3500, 3)]
        self.assert_status(lines, "PASS", decode_keys=("8x4096",))

    def test_a_decode_key_is_not_a_prompt_key_when_chunks_are_grouped(self) -> None:
        """Same start and replay on a prompt and a decode key are two chunks: ten layers of each must not add up to a complete one."""
        lines = [*prompt_chunk(1000, 1)[:5], *decode_chunk("decode:8x4096", 1000, 1)[5:]]
        self.assert_status(lines, "NOT_MEASURED", "layers with a verdict", decode_only=True)

    def test_the_command_line_takes_decode_keys_and_decode_only(self) -> None:
        lines = [*decode_chunk("decode:8x4096", 3500, 1), *decode_chunk("decode:16x4096", 90, 2, live=1)]
        with tempfile.TemporaryDirectory() as folder:
            log = Path(folder) / "container.log"
            log.write_text("\n".join(lines), encoding="utf-8")
            for arguments, expected in ((["--decode-only", "--decode-key", "8x4096", "--decode-key", "16x4096"], 0),
                                        (["--decode-only", "--decode-key", "4x8192"], 2), ([], 2)):
                with self.subTest(arguments), contextlib.redirect_stdout(io.StringIO()) as printed:
                    self.assertEqual(reader.main([str(log), "--ranks", "4", "--layers", str(LAYERS), *arguments]), expected)
                if expected:
                    self.assertIn("NOT_MEASURED", printed.getvalue())

    def test_a_malformed_decode_key_argument_is_refused_with_the_form(self) -> None:
        with self.assertRaises(SystemExit), contextlib.redirect_stderr(io.StringIO()) as printed:
            reader.main(["x.log", "--decode-key", "8:4096"])
        self.assertIn("ROWSxWIDTH", printed.getvalue())


class TheReaderReadsWhatTheDecodeCheckLogs(CheckCase):
    """The reader against lines this module's RankCheck really writes for decode keys, so a format change in one breaks the other's test."""

    def read(self, lanes: int, seed: int, key_rows: int = ROWS, start: int = LONGEST) -> list[str]:
        self.before_all()
        self.write_chunk(lanes, seed)
        return self.on_every_rank("after_replay", f"decode:{key_rows}x{WIDTH}", start, key_rows, lanes, 1)

    def judge(self, lines: list[str], keys: tuple[str, ...]) -> object:
        return reader.judge(lines, ranks=RANKS, layers=LAYERS, decode=reader.DecodeAsk(keys, True))

    def test_two_good_replays_of_different_lane_counts_pass(self) -> None:
        lines = [*self.read(8, 1), *self.read(3, 2, start=900)]
        self.assertEqual(self.judge(lines, (f"{ROWS}x{WIDTH}",)).status, "PASS")

    def test_a_void_among_good_replays_is_not_a_pass(self) -> None:
        good = self.read(8, 1)
        self.before_all()
        stale = self.on_every_rank("after_replay", LABEL, 900, ROWS, 8, 1)
        result = self.judge([*good, *stale], (f"{ROWS}x{WIDTH}",))
        self.assertEqual(result.status, "NOT_MEASURED", result)
        self.assertTrue(any("stale_rows=" in reason for reason in result.reasons), result.reasons)

    def test_a_layer_that_differs_on_a_short_batch_fails(self) -> None:
        self.before_all()
        self.write_chunk(3, 1)
        self.own(1, 4)[0, 0] += 1
        lines = self.on_every_rank("after_replay", LABEL, LONGEST, ROWS, 3, 1)
        result = self.judge(lines, (f"{ROWS}x{WIDTH}",))
        self.assertEqual(result.status, "FAIL", result)
        self.assertTrue(any("layer 7" in reason and "rows_differ=1" in reason for reason in result.reasons), result.reasons)


if __name__ == "__main__":
    unittest.main()
