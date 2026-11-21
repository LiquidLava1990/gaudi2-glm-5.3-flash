"""WIDE-PROMPT-KERNEL-PLAN step 3: HPU_GLM53_INDEXER_RANK_CHECK, the kill gate for query-shard.

Every layer's physical rows, compared across ranks with gloo. All layers write one shared top-k buffer, so with the flag on each indexer writes
its own buffer instead: same ops, another destination. A stale, partly written or dead buffer, or a lone rank, must never read as equal=1 or equal=0.
"""
from __future__ import annotations

import contextlib
import hashlib
import logging
import os
import sys
import threading
import unittest
from collections.abc import Callable, Iterator
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
from test_prompt_batch_keys import SharedCase  # noqa: E402
from test_prompt_keys import Backend, KeyCase, RowForward  # noqa: E402
from test_query_shard import OpRecorder, row_scores  # noqa: E402

FLAG = "HPU_GLM53_INDEXER_RANK_CHECK"
SHARD_FLAG = "HPU_GLM53_QUERY_SHARD_KEYS"
DECODE_SHARD_FLAG = "HPU_GLM53_QUERY_SHARD_DECODE_KEYS"
KEYS = "VLLM_GAUDI_MIXED_PROMPT_KEYS"
RANKS, ROWS, COLUMNS, LAYERS = 4, 8, 6, 11
TABLE = 32


class FakeOp:
    """What RankCheck reads of an indexer: its scan method marks it, its prefix names the layer."""

    def __init__(self, layer: int, rows: int = 16) -> None:
        self.k_cache = SimpleNamespace(prefix=f"model.language_model.model.layers.{layer}.self_attn.indexer.k_cache")
        self.topk_indices_buffer = torch.full((rows, COLUMNS + 2), -7, dtype=torch.int32)
        self.topk_tokens, self.pool_size = COLUMNS - 3, 4

    def canonical_shared_scan(self) -> None:
        return None


class FakeModel:
    def __init__(self, layers: int = LAYERS) -> None:
        self.ops = [FakeOp(layer + 3) for layer in range(layers)]

    def named_modules(self) -> Iterator[tuple[str, object]]:
        yield "", self
        for index, op in enumerate(self.ops):
            yield f"layers.{index + 3}.indexer", op


class Exchange:
    """rank_reports for rank threads: every rank must arrive, or the barrier times out; the call count is the collective's sequence."""

    def __init__(self) -> None:
        self.barrier = threading.Barrier(RANKS, timeout=60)
        self.local = threading.local()
        self.reports: dict[int, dict] = {}
        self.calls = [0] * RANKS

    def __call__(self, report: dict) -> list[dict]:
        rank = self.local.rank
        self.calls[rank] += 1
        self.reports[rank] = report
        self.barrier.wait()
        gathered = [self.reports[peer] for peer in range(RANKS)]
        self.barrier.wait()
        return gathered


class CheckCase(unittest.TestCase):
    """Four ranks' RankCheck objects over four fake models; a test writes what each rank's replay left in the buffers."""

    def setUp(self) -> None:
        environ = mock.patch.dict(os.environ, {FLAG: "1", KEYS: f"{ROWS}:1024"})
        environ.start()
        self.addCleanup(environ.stop)
        self.exchange = Exchange()
        self.models = [FakeModel() for _ in range(RANKS)]
        fence = mock.patch.object(graph, "_rank_check_fence", self.fence)
        fence.start()
        self.addCleanup(fence.stop)
        self.fences = [0] * RANKS
        report = mock.patch.object(graph, "rank_reports", self.exchange)
        report.start()
        self.addCleanup(report.stop)
        rank = mock.patch.object(graph, "_rank_check_is_logger", lambda: self.exchange.local.rank == 0)
        rank.start()
        self.addCleanup(rank.stop)
        tensor_parallel = mock.patch.object(graph, "_rank_check_tp_size", return_value=RANKS)
        tensor_parallel.start()
        self.addCleanup(tensor_parallel.stop)
        self.exchange.local.rank = 0
        self.checks = [graph.RankCheck() for _ in range(RANKS)]
        for check, model in zip(self.checks, self.models):
            check.prepare(model, ROWS)
        self.fences = [0] * RANKS

    def fence(self) -> None:
        self.fences[self.exchange.local.rank] += 1

    def own(self, rank: int, layer: int) -> torch.Tensor:
        return vars(self.models[rank].ops[layer])[graph.RANK_CHECK_ATTR]

    def write_all(self, seed: int) -> None:
        for rank in range(RANKS):
            for layer in range(LAYERS):
                self.own(rank, layer).copy_(torch.arange(ROWS * COLUMNS, dtype=torch.int32).reshape(ROWS, COLUMNS) + seed + layer)

    def on_every_rank(self, method: str, *args: object, targets: list | None = None) -> list[str]:
        """``method`` of each rank's RankCheck (or of each of ``targets``, one per rank), on rank threads; returns the rank-check lines logged."""
        failures: list = []
        targets = self.checks if targets is None else targets

        def body(rank: int) -> None:
            self.exchange.local.rank = rank
            try:
                getattr(targets[rank], method)(*args)
            except BaseException as error:  # noqa: BLE001  reported below
                failures.append((rank, error))

        with self.assertLogs(graph.LOG, logging.WARNING) as logs:
            graph.LOG.warning("sentinel so assertLogs passes when nothing is logged")
            threads = [threading.Thread(target=body, args=(rank,)) for rank in range(RANKS)]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join()
        self.assertEqual(failures, [])
        return [line for line in logs.output if "GLM53_RANK_CHECK" in line]

    def before_all(self) -> None:
        for rank, check in enumerate(self.checks):
            self.exchange.local.rank = rank
            check.before_replay()

    def replay(self, seed: int, start: int = 131072) -> list[str]:
        self.before_all()
        self.write_all(seed)
        return self.after(start)

    def after(self, start: int = 0, real_rows: int = ROWS, key_rows: int = ROWS, prompts: int = 1) -> list[str]:
        """The read after a replay of a chunk of ``real_rows`` rows through a key of ``key_rows`` rows for ``prompts`` prompts."""
        return self.on_every_rank("after_replay", f"{ROWS}:1024", start, key_rows, real_rows, prompts)

    def write_chunk(self, real_rows: int, seed: int = 1) -> None:
        """What a replay of a ``real_rows``-row chunk leaves: its rows written, the key's padding below them dead."""
        self.write_all(seed)
        for rank in range(RANKS):
            for layer in range(LAYERS):
                self.own(rank, layer)[real_rows:] = -1

    def rewrite_first(self, rows: int) -> None:
        """What a replay that persisted only its first ``rows`` rows leaves: those change, the rest keep the previous chunk's values."""
        for rank in range(RANKS):
            for layer in range(LAYERS):
                self.own(rank, layer)[:rows] += 1000


class Readings(CheckCase):
    def test_the_verdict_says_how_many_ranks_agreed(self) -> None:
        self.assertTrue(all(f"ranks={RANKS}" in line for line in self.replay(1)))

    def test_a_key_shorter_than_the_buffer_compares_and_counts_only_its_rows(self) -> None:
        """The tail below a narrower key holds the taller key's leftovers, which differ by rank and are nobody's verdict."""
        self.assertTrue(all("equal=1" in line for line in self.replay(1)))
        self.before_all()
        self.rewrite_first(4)
        for layer in range(LAYERS):
            self.own(2, layer)[4:] += 50
        lines = self.after(real_rows=4, key_rows=4)
        self.assertEqual(len(lines), LAYERS)
        for line in lines:
            for field in ("equal=1", "rows=4 ", "rows_differ=0", "live_rows=4"):
                self.assertIn(field, line)

    def test_every_layer_gets_one_equal_line_from_one_rank(self) -> None:
        lines = self.replay(1)
        self.assertEqual(len(lines), LAYERS)
        for layer, line in zip(range(3, LAYERS + 3), lines):
            self.assertIn(f"GLM53_RANK_CHECK key={ROWS}:1024 layer={layer} equal=1", line)
            self.assertIn("rows_differ=0", line)
            self.assertIn(f"rows={ROWS}", line)
            self.assertIn("start=131072", line)

    def test_a_mismatch_says_how_many_rows_differ(self) -> None:
        self.before_all()
        self.write_all(1)
        self.own(2, 7)[[0, 2, 4, 5, 7], 3] += 100
        lines = self.after(131072)
        bad = [line for line in lines if "equal=0" in line]
        self.assertEqual(len(bad), 1)
        self.assertIn("layer=10 equal=0", bad[0])
        self.assertIn("rows_differ=5", bad[0])
        self.assertEqual(sum("equal=1" in line for line in lines), LAYERS - 1)

    def test_a_row_two_ranks_differ_on_counts_once(self) -> None:
        self.before_all()
        self.write_all(1)
        self.own(1, 0)[3, 0] += 1
        self.own(3, 0)[3, 1] += 1
        lines = self.after()
        self.assertIn("rows_differ=1", next(line for line in lines if "equal=0" in line))

    def test_a_dead_row_is_counted_live_rows_tells_them_apart(self) -> None:
        self.before_all()
        self.write_all(1)
        for rank in range(RANKS):
            for layer in range(LAYERS):
                self.own(rank, layer)[ROWS - 2:] = -1
        lines = self.after(real_rows=ROWS - 2)
        self.assertIn(f"live_rows={ROWS - 2}", lines[0])
        self.assertIn("equal=1", lines[0])


class RowsDigest(CheckCase):
    """Every verdict line carries rows_digest=, one tag of the rows rank 0 read: the evidence a gathered (sharded) run has instead of equal=, to compare
    with an unsharded run's lines by cells/rank_digest_compare.py. Its definition is part of the log format, so it is spelled out here independently."""

    @staticmethod
    def tag(rows: torch.Tensor) -> str:
        per_row = b"".join(hashlib.blake2b(row.numpy().tobytes(), digest_size=16).digest() for row in rows)
        return hashlib.blake2b(per_row, digest_size=8).hexdigest()

    @staticmethod
    def digest_of(line: str) -> str:
        return line.split("rows_digest=")[1].split()[0]

    def test_every_equal_line_ends_with_the_tag_of_rank_zeros_rows_of_that_layer(self) -> None:
        lines = self.replay(1)
        self.assertEqual(len(lines), LAYERS)
        for layer, line in enumerate(lines):
            self.assertTrue(line.endswith(f" rows_digest={self.tag(self.own(0, layer)[:ROWS])}"), line)

    def read(self, seed: int, mutate: Callable[[], None] | None = None, layer: int = 0) -> str:
        """The tag of one layer after a replay that wrote ``seed`` (then ``mutate``). A replay that wrote what the last one wrote is stale and logs no tag,
        so a test reads a different seed in between."""
        self.before_all()
        self.write_all(seed)
        if mutate:
            mutate()
        return self.digest_of(self.after()[layer])

    def test_the_same_rows_give_the_same_tag_and_other_rows_another(self) -> None:
        first, other, again = self.read(1), self.read(2), self.read(1)
        self.assertEqual(first, again)
        self.assertNotEqual(first, other)

    def test_every_layer_has_a_tag_of_its_own(self) -> None:
        self.assertEqual(len({self.digest_of(line) for line in self.replay(1)}), LAYERS)

    def test_one_changed_index_changes_the_tag(self) -> None:
        def bump() -> None:
            for rank in range(RANKS):
                self.own(rank, 4)[3, 2] += 1

        untouched = self.read(1, layer=4)
        self.read(2, layer=4)
        self.assertNotEqual(self.read(1, bump, layer=4), untouched)

    def test_two_swapped_rows_change_the_tag(self) -> None:
        """Rows are in order: a gather that returned the right rows in the wrong rank order must not read as the same rows."""
        def swap() -> None:
            for rank in range(RANKS):
                rows = self.own(rank, 0)
                rows[[1, 5]] = rows[[5, 1]].clone()

        untouched = self.read(1)
        self.read(2)
        self.assertNotEqual(self.read(1, swap), untouched)

    def test_only_the_keys_own_rows_are_in_the_tag(self) -> None:
        """The buffer below a narrower key holds a taller key's leftovers: they are nobody's rows."""
        def chunk(seed: int, tail: int | None = None) -> str:
            self.before_all()
            self.write_chunk(4, seed)
            if tail is not None:
                for rank in range(RANKS):
                    self.own(rank, 0)[4:] = tail
            return self.digest_of(self.after(real_rows=4, key_rows=4)[0])

        first = chunk(1)
        chunk(2)
        self.assertEqual(chunk(1, tail=9), first)

    def test_a_line_that_says_unequal_still_carries_rank_zeros_tag(self) -> None:
        self.before_all()
        self.write_all(1)
        self.own(2, 0)[1, 1] += 100
        line = self.after()[0]
        self.assertIn("equal=0", line)
        self.assertEqual(self.digest_of(line), self.tag(self.own(0, 0)[:ROWS]))

    def test_a_line_with_no_verdict_has_no_tag(self) -> None:
        self.write_all(1)
        self.before_all()
        lines = self.after()
        self.assertTrue(all("NOT_RETAINED" in line and "rows_digest" not in line for line in lines), lines)


class NeverAFalseVerdict(CheckCase):
    """No replay wrote the buffer, or wrote only -1: nothing is known, so there is no equal= at all."""

    def assert_not_retained(self, lines: list[str], reason: str) -> None:
        self.assertEqual(len(lines), LAYERS)
        for line in lines:
            self.assertIn("GLM53_RANK_CHECK_NOT_RETAINED", line)
            self.assertNotIn("equal=", line)
            self.assertIn(reason, line)
            self.assertIn(f"{FLAG}=eager", line)

    def test_a_replay_that_did_not_write_the_buffer_is_stale(self) -> None:
        self.write_all(1)
        self.before_all()
        self.assert_not_retained(self.after(), "stale")

    def test_a_second_replay_that_wrote_nothing_is_stale(self) -> None:
        self.assertTrue(all("equal=1" in line for line in self.replay(1)))
        self.before_all()
        self.assert_not_retained(self.after(), "stale")

    def test_only_padding_proves_nothing(self) -> None:
        self.before_all()
        self.write_all(1)
        for rank in range(RANKS):
            for layer in range(LAYERS):
                self.own(rank, layer).fill_(-1)
        self.assert_not_retained(self.after(), "no_live_rows")

    def layer_five(self, lines: list[str]) -> str:
        """Layer index 2 is layer label 5; the other ten layers' lines are checked by the caller."""
        found = [line for line in lines if "layer=5 " in line]
        self.assertEqual(len(found), 1, lines)
        return found[0]

    def test_one_dead_rank_voids_its_layer(self) -> None:
        self.before_all()
        self.write_all(1)
        self.own(3, 2).fill_(-1)
        lines = self.after()
        self.assertIn("no_live_rows on rank 3", self.layer_five(lines))
        self.assertEqual(sum("equal=1" in line for line in lines), LAYERS - 1)

    def test_one_stale_rank_voids_its_layer_and_is_named(self) -> None:
        """Rank 3 alone left layer 5 as the last replay wrote it, with live rows: a stale check that reads rank 0 only would let it through."""
        self.assertTrue(all("equal=1" in line for line in self.replay(1)))
        self.before_all()
        for rank in range(RANKS):
            for layer in range(LAYERS):
                if (rank, layer) != (3, 2):
                    self.own(rank, layer).add_(1000)
        lines = self.after()
        self.assertIn(f"NOT_RETAINED key={ROWS}:1024 layer=5", self.layer_five(lines))
        self.assertIn(f"stale_rows={ROWS} of {ROWS} live rows on rank 3", self.layer_five(lines))
        self.assertEqual(sum("equal=1" in line for line in lines), LAYERS - 1)

    def test_one_short_rank_voids_its_layer_and_is_named(self) -> None:
        """Rank 3 alone has fewer live rows than the chunk has: a count check that reads rank 0 only would let it through."""
        self.before_all()
        self.write_all(1)
        self.own(3, 2)[5:] = -1
        lines = self.after()
        self.assertIn(f"live_rows=5 on rank 3 but the chunk has {ROWS} real rows", self.layer_five(lines))
        self.assertEqual(sum("equal=1" in line for line in lines), LAYERS - 1)

    def misplace_on(self, rank: int) -> list[str]:
        """Five live rows on a five-row chunk, but a real row dead and a padding row alive: the count is right and the rows are not."""
        self.before_all()
        self.write_chunk(5)
        self.own(rank, 2)[2] = -1
        self.own(rank, 2)[6] = torch.arange(COLUMNS, dtype=torch.int32)
        return self.after(real_rows=5)

    def test_a_dead_real_row_and_a_live_padding_row_are_misplaced_on_rank_0(self) -> None:
        lines = self.misplace_on(0)
        self.assertIn("misplaced_rows=2 on rank 0", self.layer_five(lines))
        self.assertEqual(sum("equal=1" in line for line in lines), LAYERS - 1)

    def test_a_dead_real_row_and_a_live_padding_row_are_misplaced_on_a_later_rank(self) -> None:
        lines = self.misplace_on(3)
        self.assertIn("misplaced_rows=2 on rank 3", self.layer_five(lines))
        self.assertEqual(sum("equal=1" in line for line in lines), LAYERS - 1)

    def test_a_chunk_whose_padding_is_dead_reads_equal(self) -> None:
        self.before_all()
        self.write_chunk(5)
        lines = self.after(real_rows=5)
        self.assertTrue(all("equal=1" in line and "live_rows=5" in line for line in lines), lines)

    def test_a_shared_keys_rows_are_not_contiguous_so_only_their_count_is_judged(self) -> None:
        """Each prompt of a shared key pads its own rows, so no row is 'the' padding: five live rows of five is all that can be said."""
        self.before_all()
        self.write_chunk(5)
        self.own(0, 2)[2] = -1
        self.own(0, 2)[6] = torch.arange(COLUMNS, dtype=torch.int32)
        lines = self.after(real_rows=5, prompts=2)
        self.assertTrue(all("misplaced" not in line for line in lines), lines)
        self.assertIn("equal=0", self.layer_five(lines))

    def test_a_replay_that_rewrote_only_some_rows_is_voided_and_counts_the_rest(self) -> None:
        """Every rank keeps the previous chunk's rows where nothing was written, so they compare equal: a changed row elsewhere must not vouch for them."""
        self.assertTrue(all("equal=1" in line for line in self.replay(1)))
        self.before_all()
        self.rewrite_first(3)
        self.assert_not_retained(self.after(), f"stale_rows={ROWS - 3}")

    def test_a_first_replay_that_wrote_only_some_rows_is_voided_by_the_live_count(self) -> None:
        self.before_all()
        for rank in range(RANKS):
            for layer in range(LAYERS):
                self.own(rank, layer)[:5] = torch.arange(5 * COLUMNS, dtype=torch.int32).reshape(5, COLUMNS) + layer
        self.assert_not_retained(self.after(), f"live_rows=5 on rank 0 but the chunk has {ROWS} real rows")

    def test_the_previous_chunks_rows_under_a_shorter_chunk_are_voided(self) -> None:
        """A 4-row chunk writes 4 rows; the 8-row chunk before it left live rows below them, equal on every rank and not this replay's."""
        self.assertTrue(all("equal=1" in line for line in self.replay(1)))
        self.before_all()
        self.rewrite_first(4)
        self.assert_not_retained(self.after(real_rows=4), f"live_rows={ROWS} on rank 0 but the chunk has 4 real rows")


class TheReaderReadsWhatTheCheckLogs(CheckCase):
    """cells/rank_check_read.py against the lines this module's RankCheck really writes, so a format change in one breaks the other's test."""

    def judge(self, lines: list[str]) -> object:
        return reader.judge(lines, ranks=RANKS, layers=LAYERS, min_starts=(131072,))

    def test_two_good_replays_pass(self) -> None:
        self.assertEqual(self.judge([*self.replay(1), *self.replay(2, start=140000)]).status, "PASS")

    def test_a_stale_replay_among_good_ones_is_not_measured(self) -> None:
        good = self.replay(1)
        self.before_all()
        result = self.judge([*good, *self.after(start=140000)])
        self.assertEqual(result.status, "NOT_MEASURED", result)
        self.assertTrue(any("stale_rows=" in reason for reason in result.reasons), result.reasons)

    def test_a_replay_one_layer_of_which_differs_fails(self) -> None:
        self.before_all()
        self.write_all(1)
        self.own(1, 4)[0, 0] += 1
        result = self.judge(self.after(start=140000))
        self.assertEqual(result.status, "FAIL", result)
        self.assertTrue(any("layer 7" in reason and "rows_differ=1" in reason for reason in result.reasons), result.reasons)

    def test_an_unreadable_report_costs_a_layer_and_the_reader_sees_it_missing(self) -> None:
        gather = self.exchange

        def rank_1_sent_ten_layers(report: dict) -> list[dict]:
            gathered = gather(report)
            return [gathered[0], {**gathered[1], "layers": gathered[1]["layers"][:LAYERS - 1]}, *gathered[2:]]

        self.before_all()
        self.write_all(1)
        with mock.patch.object(graph, "rank_reports", rank_1_sent_ten_layers):
            result = self.judge(self.after(start=140000))
        self.assertEqual(result.status, "NOT_MEASURED", result)
        self.assertTrue(any("LOG_FAILED" in reason for reason in result.reasons), result.reasons)


class EveryTensorParallelRank(unittest.TestCase):
    """Reports from fewer ranks than the model runs on (a lone worker, or a group of 2 or 3 of the 4) agree with whoever is missing: no equal=1."""

    def lines(self, gathered: int, tp_size: object = RANKS) -> list[str]:
        """The lines rank 0 logs when ``gathered`` identical reports come back from a model whose tensor-parallel size is ``tp_size``."""
        model, check = FakeModel(), graph.RankCheck()
        size = {"side_effect": tp_size} if isinstance(tp_size, Exception) else {"return_value": tp_size}
        with mock.patch.dict(os.environ, {FLAG: "1", KEYS: f"{ROWS}:1024"}), mock.patch.object(graph, "_rank_check_fence"), \
                mock.patch.object(graph, "rank_reports", lambda report: [report] * gathered), \
                mock.patch.object(graph, "_rank_check_is_logger", return_value=True), mock.patch.object(graph, "_rank_check_tp_size", **size):
            check.prepare(model, ROWS)
            check.before_replay()
            for layer, op in enumerate(model.ops):
                vars(op)[graph.RANK_CHECK_ATTR].copy_(torch.arange(ROWS * COLUMNS, dtype=torch.int32).reshape(ROWS, COLUMNS) + layer)
            with self.assertLogs(graph.LOG, logging.WARNING) as logs:
                check.after_replay(f"{ROWS}:1024", 0, ROWS, ROWS)
        return [line for line in logs.output if "GLM53_RANK_CHECK" in line]

    def test_one_two_or_three_ranks_of_four_give_no_verdict(self) -> None:
        for gathered in (1, 2, 3):
            lines = self.lines(gathered)
            with self.subTest(gathered):
                self.assertEqual(len(lines), LAYERS)
                for line in lines:
                    self.assertIn("GLM53_RANK_CHECK_NOT_RETAINED", line)
                    self.assertIn(f"ranks={gathered} of tp_size={RANKS}", line)
                    self.assertNotIn("equal=", line)

    def test_more_reports_than_tensor_parallel_ranks_give_no_verdict(self) -> None:
        """A world group larger than the model's TP group holds ranks of another group: they are not this layer's peers."""
        self.assertTrue(all(f"ranks=8 of tp_size={RANKS}" in line and "equal=" not in line for line in self.lines(8)))

    def test_every_tensor_parallel_rank_present_reads_equal_whatever_its_size(self) -> None:
        for size in (2, 4):
            with self.subTest(size):
                lines = self.lines(size, tp_size=size)
                self.assertTrue(all("equal=1" in line and f"ranks={size}" in line for line in lines), lines)

    def test_a_tensor_parallel_size_that_cannot_be_read_gives_no_verdict_and_says_why(self) -> None:
        lines = self.lines(RANKS, tp_size=AssertionError("tensor model parallel group is not initialized"))
        self.assertEqual(len(lines), LAYERS)
        self.assertTrue(all("NOT_RETAINED" in line and "tensor model parallel group is not initialized" in line for line in lines), lines)


class LockstepOnEveryRank(CheckCase):
    def test_a_read_failure_on_one_rank_still_reaches_the_collective_on_all(self) -> None:
        self.before_all()
        self.write_all(1)
        real = graph._row_digests

        def fails_on_rank_2(rows: torch.Tensor) -> list[bytes]:
            if self.exchange.local.rank == 2:
                raise RuntimeError("synStatus 8 reading a tensor")
            return real(rows)

        with mock.patch.object(graph, "_row_digests", fails_on_rank_2):
            lines = self.after()
        self.assertEqual(self.exchange.calls, [1] * RANKS)
        self.assertTrue(lines)
        self.assertTrue(all("NOT_RETAINED" in line and "rank 2" in line for line in lines), lines)

    def test_a_failed_fence_on_one_rank_still_reaches_the_collective_on_all(self) -> None:
        self.before_all()
        self.write_all(1)

        def fence_fails() -> None:
            if self.exchange.local.rank == 1:
                raise RuntimeError("fence")

        with mock.patch.object(graph, "_rank_check_fence", fence_fails):
            lines = self.after()
        self.assertEqual(self.exchange.calls, [1] * RANKS)
        self.assertTrue(all("NOT_RETAINED" in line for line in lines))

    def test_after_a_read_failed_midway_the_next_replay_is_judged_against_what_the_buffers_held(self) -> None:
        """Rank 2's read dies on its sixth layer. Its baseline must not be half new and half old: layer 10 (index 7) is left alone by
        the next replay, and only a baseline taken after the failed one can see that it is stale."""
        self.assertTrue(all("equal=1" in line for line in self.replay(1)))
        self.before_all()
        self.write_all(2)
        real, reads = graph._row_digests, []

        def dies_on_the_sixth_layer_of_rank_2(rows: torch.Tensor) -> list[bytes]:
            if self.exchange.local.rank == 2:
                reads.append(rows)
                if len(reads) == 6:
                    raise RuntimeError("synStatus 8 reading layer 8")
            return real(rows)

        with mock.patch.object(graph, "_row_digests", dies_on_the_sixth_layer_of_rank_2):
            failed = self.after()
        self.assertTrue(all("NOT_RETAINED" in line and "rank 2" in line for line in failed), failed)
        self.before_all()
        self.write_all(3)
        self.own(2, 7).copy_(torch.arange(ROWS * COLUMNS, dtype=torch.int32).reshape(ROWS, COLUMNS) + 2 + 7)
        lines = self.after()
        void = [line for line in lines if "NOT_RETAINED" in line]
        self.assertEqual(len(void), 1, lines)
        self.assertIn("layer=10 ", void[0])
        self.assertIn(f"stale_rows={ROWS} of {ROWS} live rows on rank 2", void[0])
        self.assertEqual(sum("equal=1" in line for line in lines), LAYERS - 1)

    def test_a_report_rank_0_cannot_read_costs_its_lines_not_its_place_in_the_collective(self) -> None:
        """Only rank 0 logs. A raise there would leave it out of the next gather while its peers enter it with no exit armed."""
        gather = self.exchange

        def rank_1_sent_three_layers(report: dict) -> list[dict]:
            gathered = gather(report)
            return [gathered[0], {**gathered[1], "layers": gathered[1]["layers"][:3]}, *gathered[2:]]

        self.before_all()
        self.write_all(1)
        with mock.patch.object(graph, "rank_reports", rank_1_sent_three_layers):
            lines = self.after()
        self.assertEqual(gather.calls, [1] * RANKS)
        self.assertEqual(sum("equal=1" in line for line in lines), 3)
        failed = [line for line in lines if "GLM53_RANK_CHECK_LOG_FAILED" in line]
        self.assertEqual(len(failed), LAYERS - 3, lines)
        self.assertIn("IndexError", failed[0])

    def test_every_rank_fences_once_and_gathers_once_per_replay(self) -> None:
        self.replay(1)
        self.replay(2)
        self.assertEqual(self.fences, [2] * RANKS)
        self.assertEqual(self.exchange.calls, [2] * RANKS)

    def test_ranks_on_different_chunks_are_reported_not_compared(self) -> None:
        self.before_all()
        self.write_all(1)
        starts = {0: 0, 1: 0, 2: 1024, 3: 0}

        def body(rank: int) -> None:
            self.exchange.local.rank = rank
            self.checks[rank].after_replay(f"{ROWS}:1024", starts[rank], ROWS, ROWS)

        with self.assertLogs(graph.LOG, logging.WARNING) as logs:
            threads = [threading.Thread(target=body, args=(rank,)) for rank in range(RANKS)]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join()
        lines = [line for line in logs.output if "GLM53_RANK_CHECK" in line]
        self.assertTrue(lines and all("NOT_RETAINED" in line and "lockstep" in line for line in lines), lines)


class PrepareTheBuffers(unittest.TestCase):
    def test_every_indexer_gets_one_buffer_of_the_key_rows_not_the_full_8192(self) -> None:
        model = FakeModel()
        with mock.patch.object(graph, "_rank_check_fence"):
            graph.RankCheck().prepare(model, ROWS)
        buffers = [vars(op)[graph.RANK_CHECK_ATTR] for op in model.ops]
        self.assertEqual(len({id(buffer) for buffer in buffers}), LAYERS)
        for buffer in buffers:
            self.assertEqual(buffer.shape, (ROWS, COLUMNS))
            self.assertEqual(buffer.dtype, torch.int32)

    def test_a_model_with_no_indexer_refuses_the_boot(self) -> None:
        with mock.patch.object(graph, "_rank_check_fence"), self.assertRaisesRegex(RuntimeError, FLAG):
            graph.RankCheck().prepare(FakeModel(layers=0), ROWS)


class CaptureTarget(unittest.TestCase):
    """With the flag on the real indexer writes its own buffer, and the sequence the graph captures is the same."""

    def layer(self, flag: str, rows: int = 1024, real_rows: int | None = None) -> tuple:
        """One captured layer step; ``real_rows`` fewer than ``rows``: the chunk's rows are masked in, the key's padding is masked out."""
        env = {KEYS: f"{rows}:{TABLE}", "HPU_GLM53_R589_CAPTURE_MARKS": "1", FLAG: flag}
        mask = None if real_rows is None else (torch.arange(rows) < real_rows).float()
        with key_stack.layer_stack(OVERLAY_ROOT, env) as world, OpRecorder() as recorder:
            world.kpool._score_pool_chunk = row_scores
            case = key_stack.prompt_case(rows, TABLE, 2000)
            op = key_stack.indexer_op(world, case)
            shared = op.topk_indices_buffer
            own = torch.full((rows, key_stack.WIDTH), -3, dtype=torch.int32)
            if flag == "1":
                vars(op)[graph.RANK_CHECK_ATTR] = own
            before = len(recorder.ops)
            with graph.capture_context(), graph.key_capture_context():
                physical = key_stack.indexer_step(world, case, op, token_mask=mask)
            ops = recorder.ops[before:]
            fences = [(site.split(":")[0], sync) for site, sync in world.tally.calls]
            return SimpleNamespace(ops=ops, fences=fences, physical=physical, own=own, shared=shared, op=op)

    def test_each_indexer_writes_its_own_rows_and_gives_the_shared_buffer_back(self) -> None:
        on = self.layer("1")
        self.assertTrue(torch.equal(on.own, on.physical))
        self.assertIs(on.op.topk_indices_buffer, on.shared)
        self.assertTrue(bool((on.shared == -7).all()), "the shared buffer must not be written")

    def test_the_real_indexer_leaves_exactly_the_masked_rows_dead(self) -> None:
        """The check calls a row live when it holds an index >= 0 and expects the chunk's rows first, padding after: that is the indexer's doing."""
        real = 700
        for flag in ("0", "1"):
            with self.subTest(flag):
                step = self.layer(flag, real_rows=real)
                tensors = {"returned rows": step.physical, **({"the check's own rows": step.own} if flag == "1" else {})}
                for name, rows in tensors.items():
                    live = (rows >= 0).any(dim=1)
                    self.assertTrue(bool(live[:real].all()), f"{name}: a chunk row is dead")
                    self.assertTrue(bool((rows[real:] == -1).all()), f"{name}: a padding row holds an index")
                    self.assertEqual(int(live.sum()), real)

    def test_a_fully_masked_chunk_is_all_dead(self) -> None:
        step = self.layer("1", real_rows=0)
        self.assertTrue(bool((step.own == -1).all()))

    def test_the_op_sequence_and_rows_equal_the_flag_off_capture(self) -> None:
        off, on = self.layer("0"), self.layer("1")
        self.assertEqual(off.fences, on.fences)
        self.assertTrue(torch.equal(off.physical, on.physical))
        self.assertEqual(off.ops, on.ops)

    def test_outside_a_key_capture_the_shared_buffer_stays(self) -> None:
        op = FakeOp(3)
        vars(op)[graph.RANK_CHECK_ATTR] = torch.zeros((ROWS, COLUMNS), dtype=torch.int32)
        shared = op.topk_indices_buffer
        with graph.rank_check_target(op, ROWS):
            self.assertIs(op.topk_indices_buffer, shared)

    def test_a_forward_taller_than_the_buffer_keeps_the_shared_buffer(self) -> None:
        op = FakeOp(3)
        vars(op)[graph.RANK_CHECK_ATTR] = torch.zeros((ROWS, COLUMNS), dtype=torch.int32)
        shared = op.topk_indices_buffer
        with self.assertLogs(graph.LOG, logging.WARNING) as logs, graph.key_capture_context(), graph.rank_check_target(op, ROWS + 1):
            self.assertIs(op.topk_indices_buffer, shared)
        self.assertIn(f"GLM53_RANK_CHECK_TARGET_TOO_SHORT layer=3 forward_rows={ROWS + 1} buffer_rows={ROWS}", logs.output[0])

    def test_a_failure_gives_the_shared_buffer_back(self) -> None:
        op = FakeOp(3)
        vars(op)[graph.RANK_CHECK_ATTR] = torch.zeros((ROWS, COLUMNS), dtype=torch.int32)
        shared = op.topk_indices_buffer
        with self.assertRaises(ZeroDivisionError), graph.key_capture_context(), graph.rank_check_target(op, ROWS):
            self.assertIsNot(op.topk_indices_buffer, shared)
            raise ZeroDivisionError
        self.assertIs(op.topk_indices_buffer, shared)


class FlagOff(unittest.TestCase):
    def test_unset_and_zero_read_off(self) -> None:
        with mock.patch.dict(os.environ):
            os.environ.pop(FLAG, None)
            self.assertEqual(graph.rank_check_mode(), "")
            os.environ[FLAG] = "0"
            self.assertEqual(graph.rank_check_mode(), "")

    def test_a_bad_value_refuses_and_names_the_values(self) -> None:
        with mock.patch.dict(os.environ, {FLAG: "2"}), self.assertRaisesRegex(ValueError, "eager"):
            graph.rank_check_mode()

    def test_the_target_is_no_bind_at_all(self) -> None:
        op = FakeOp(3)
        shared = op.topk_indices_buffer
        with graph.key_capture_context(), graph.rank_check_target(op, ROWS):
            self.assertIs(op.topk_indices_buffer, shared)


class Wiring(KeyCase):
    """The hooks in the keyed path: read after a real replay, outside the census's clock, never at boot, and before the seal's first reading."""

    keys = f"1024:{TABLE}"

    def cached(self) -> object:
        self.backend = Backend()
        cache = self.graph.GraphCache(RowForward(), 128, self.backend)
        self.boot_capture(cache, self.prompt_batch(1024, 0, TABLE))
        return cache

    def test_flag_off_touches_nothing(self) -> None:
        cache = self.cached()
        with mock.patch.object(self.graph.RankCheck, "after_replay") as after, mock.patch.object(self.graph.RankCheck, "before_replay") as before:
            cache.run_key(self.prompt_batch(1024, 1024, TABLE))
        after.assert_not_called()
        before.assert_not_called()

    def test_a_live_replay_is_read_after_it_runs_with_its_key_and_start(self) -> None:
        with mock.patch.dict(os.environ, {FLAG: "1"}):
            cache = self.cached()
            order: list[object] = []
            real = self.backend.replay
            self.backend.replay = lambda handle: (order.append("replay"), real(handle))[1]
            cache.rank_check = SimpleNamespace(before_replay=lambda: order.append("before"),
                                               after_replay=lambda *args: order.append(args))
            cache.run_key(self.prompt_batch(1024, 2048, TABLE))
        self.assertEqual(order, ["before", "replay", (f"1024:{TABLE}", 2048, 1024, 1024, 1)])

    def test_the_read_gets_the_keys_rows_and_the_chunks_real_rows(self) -> None:
        with mock.patch.dict(os.environ, {FLAG: "1"}):
            cache = self.cached()
            seen: list[tuple] = []
            cache.rank_check = SimpleNamespace(before_replay=lambda: None, after_replay=lambda *args: seen.append(args))
            cache.run_key(self.prompt_batch(600, 2048, TABLE))
        self.assertEqual(seen, [(f"1024:{TABLE}", 2048, 1024, 600, 1)])

    def test_the_boot_capture_is_not_read(self) -> None:
        with mock.patch.dict(os.environ, {FLAG: "1"}), mock.patch.object(self.graph.RankCheck, "after_replay") as after:
            self.cached()
        after.assert_not_called()

    def test_the_read_is_outside_the_census_clock(self) -> None:
        with mock.patch.dict(os.environ, {FLAG: "1"}):
            cache = self.cached()
            order: list[str] = []
            real = cache._key_forward
            cache._key_forward = lambda *args: (order.append("forward"), real(*args), order.append("forward_done"))[1]
            cache.rank_check = SimpleNamespace(before_replay=lambda: order.append("before"), after_replay=lambda *args: order.append("after"))
            cache.run_key(self.prompt_batch(1024, 2048, TABLE))
        self.assertEqual(order, ["before", "forward", "forward_done", "after"])

    def test_warmup_allocates_the_buffers_before_the_seal_reads_in_use(self) -> None:
        order: list[str] = []
        runner = SimpleNamespace(model=SimpleNamespace(_glm53_mixed_graph_forward=RowForward(), named_modules=lambda: ()),
                                 model_config=SimpleNamespace(enforce_eager=False), attn_block_size=128,
                                 _prepare_dummy_scenario=lambda prompt, decode: order.append("capture"))
        tiles = SimpleNamespace(warm_adaptive_tiles_at_boot=lambda model: 0)
        with mock.patch.dict(os.environ, {FLAG: "1"}), mock.patch.dict(sys.modules, {"glm53_r590_long_history": tiles}), \
                mock.patch.object(self.graph, "validate_warmup", return_value=((0, 1024, TABLE),)), \
                mock.patch.object(self.graph, "warmup_scenarios", return_value=[(None, None)]), mock.patch.object(self.graph, "check_keys_captured"), \
                mock.patch.object(self.graph, "seal_prompt_keys"), mock.patch.object(self.graph, "mark_boot_window"), \
                mock.patch.object(self.graph, "_in_use_bytes", side_effect=lambda: order.append("in_use") or 0), \
                mock.patch.object(self.graph.RankCheck, "prepare", side_effect=lambda model, rows: order.append(f"prepare{rows}")):
            self.graph.warmup(runner)
        self.assertEqual(order, ["prepare1024", "in_use", "capture"])

    def prepare_runner(self) -> SimpleNamespace:
        model = FakeModel()
        model._glm53_mixed_graph_forward = RowForward()
        return SimpleNamespace(model=model, model_config=SimpleNamespace(enforce_eager=False), attn_block_size=128)

    def test_a_prepare_that_fails_on_this_rank_arms_the_exit_and_reraises(self) -> None:
        """Its peers are already in warmup's next gather, which this rank will never reach: only the bounded exit ends them."""
        failures = (("the fence", mock.patch.object(self.graph, "_rank_check_fence", side_effect=RuntimeError("synStatus 8 in the fence"))),
                    ("torch.full", mock.patch.object(self.graph.torch, "full", side_effect=RuntimeError("PT_DEVMEM in torch.full"))))
        for name, failure in failures:
            runner = self.prepare_runner()
            with self.subTest(name), mock.patch.dict(os.environ, {FLAG: "1"}), mock.patch.object(self.graph.graphs_only, "arm_exit") as arm, failure, \
                    self.assertRaisesRegex(RuntimeError, name):
                self.graph.prepare_rank_check(runner, ((0, ROWS, TABLE),))
            arm.assert_called_once()
            self.assertIn("RANK_CHECK", arm.call_args.args[0])

    def test_a_shared_key_sizes_the_buffers_by_all_its_prompts_rows(self) -> None:
        """A 4-prompt shared key of 512 rows writes 2048: buffers of the tallest single-prompt key's 1024 would leave it on the shared buffer, unread."""
        runner = self.prepare_runner()
        with mock.patch.dict(os.environ, {FLAG: "1"}), mock.patch.object(self.graph, "_rank_check_fence"):
            self.graph.prepare_rank_check(runner, ((0, 1024, TABLE), (0, 512, TABLE, 4)))
        for op in runner.model.ops:
            own, shared = vars(op)[graph.RANK_CHECK_ATTR], op.topk_indices_buffer
            self.assertEqual(own.shape[0], 2048)
            with self.graph.key_capture_context(), self.graph.rank_check_target(op, 2048):
                self.assertIs(op.topk_indices_buffer, own)
            self.assertIs(op.topk_indices_buffer, shared)

    def test_a_buckets_rows_are_its_prompts_times_the_keys_rows(self) -> None:
        self.assertEqual((self.graph._bucket_rows((0, 1024, TABLE)), self.graph._bucket_prompts((0, 1024, TABLE))), (1024, 1))
        self.assertEqual((self.graph._bucket_rows((0, 512, TABLE, 4)), self.graph._bucket_prompts((0, 512, TABLE, 4))), (2048, 4))

    def test_a_prepare_that_succeeds_arms_nothing_and_gives_every_indexer_its_buffer(self) -> None:
        runner = self.prepare_runner()
        with mock.patch.dict(os.environ, {FLAG: "1"}), mock.patch.object(self.graph.graphs_only, "arm_exit") as arm, \
                mock.patch.object(self.graph, "_rank_check_fence"):
            self.graph.prepare_rank_check(runner, ((0, ROWS, TABLE),))
        arm.assert_not_called()
        self.assertEqual(sum(graph.RANK_CHECK_ATTR in vars(op) for op in runner.model.ops), LAYERS)

    def test_flag_off_warmup_allocates_nothing(self) -> None:
        runner = SimpleNamespace(model=SimpleNamespace(_glm53_mixed_graph_forward=RowForward(), named_modules=lambda: ()),
                                 model_config=SimpleNamespace(enforce_eager=False), attn_block_size=128, _prepare_dummy_scenario=lambda prompt, decode: None)
        tiles = SimpleNamespace(warm_adaptive_tiles_at_boot=lambda model: 0)
        with mock.patch.dict(sys.modules, {"glm53_r590_long_history": tiles}), mock.patch.object(self.graph, "validate_warmup", return_value=((0, 1024, TABLE),)), \
                mock.patch.object(self.graph, "warmup_scenarios", return_value=[]), mock.patch.object(self.graph, "check_keys_captured"), \
                mock.patch.object(self.graph, "seal_prompt_keys"), mock.patch.object(self.graph, "mark_boot_window"), mock.patch.object(self.graph, "_in_use_bytes", return_value=0), \
                mock.patch.object(self.graph.RankCheck, "prepare") as prepare:
            self.graph.warmup(runner)
        prepare.assert_not_called()


class SharedKeyWiring(SharedCase):
    """A shared key (4 prompts x 256 rows) writes 1024 rows for however many prompts the chunk holds."""

    def test_the_key_rows_the_real_rows_and_the_prompts_reach_the_read(self) -> None:
        with mock.patch.dict(os.environ, {FLAG: "1"}):
            with self.graph.boot_capture_context():
                self.cache.run_key(self.shared_batch([256, 40, 256], [0, 1024, 2048]))
            seen: list[tuple] = []
            self.cache.rank_check = SimpleNamespace(before_replay=lambda: None, after_replay=lambda *args: seen.append(args))
            self.cache.run_key(self.shared_batch([100, 200], [1024, 512]))
        self.assertEqual(seen, [("4:256:1024", 1024, 1024, 300, 4)])


class EagerFallback(KeyCase):
    """The plan's fallback: the same real prompts run on the runner's own forward, so Python sees every layer's rows as it selects them."""

    keys = f"1024:{TABLE}"

    def test_a_live_chunk_runs_native_and_the_boot_capture_still_captures(self) -> None:
        with mock.patch.dict(os.environ, {FLAG: "eager"}):
            backend = Backend()
            cache = self.graph.GraphCache(RowForward(), 128, backend)
            self.assertIsNotNone(self.boot_capture(cache, self.prompt_batch(1024, 0, TABLE)))
            self.assertIsNone(cache.run_key(self.prompt_batch(1024, 1024, TABLE)))
        self.assertEqual(backend.replays, 0)
        self.assertEqual(cache.counts["fallback_rank_check_eager"], 1)

    def live(self, reports: list[dict] | None = None) -> tuple[list[str], torch.Tensor]:
        rows = torch.arange(ROWS * COLUMNS, dtype=torch.int32).reshape(ROWS, COLUMNS)
        exchange = lambda report: reports or [report] * RANKS  # noqa: E731
        with self.assertLogs(self.graph.LOG, logging.WARNING) as logs, mock.patch.dict(os.environ, {FLAG: "eager"}), \
                mock.patch.object(self.graph, "rank_reports", exchange), mock.patch.object(self.graph, "_rank_check_is_logger", return_value=True), \
                mock.patch.object(self.graph, "_RANK_CHECK_BOOT_DONE", [True]), mock.patch.object(self.graph, "_rank_check_tp_size", return_value=RANKS):
            self.graph.LOG.warning("sentinel")
            self.graph.rank_check_eager_rows(FakeOp(3), rows, f"{ROWS}:{TABLE}")
        return [line for line in logs.output if "GLM53_RANK_CHECK" in line], rows

    def test_equal_ranks_log_one_line_with_key_and_layer(self) -> None:
        lines, _ = self.live()
        self.assertEqual(len(lines), 1)
        self.assertIn(f"GLM53_RANK_CHECK key={ROWS}:{TABLE} layer=3 equal=1", lines[0])
        self.assertIn("eager=1", lines[0])

    def test_the_eager_line_carries_the_tag_of_its_rows_too(self) -> None:
        lines, rows = self.live()
        per_row = b"".join(hashlib.blake2b(row.numpy().tobytes(), digest_size=16).digest() for row in rows)
        self.assertTrue(lines[0].endswith(f"eager=1 rows_digest={hashlib.blake2b(per_row, digest_size=8).hexdigest()}"), lines[0])

    def test_one_rank_alone_has_no_verdict(self) -> None:
        rows = torch.arange(ROWS * COLUMNS, dtype=torch.int32).reshape(ROWS, COLUMNS)
        alone = [{"where": (f"{ROWS}:{TABLE}",), "error": "", "tp_size": RANKS, "layers": [self.graph._rows_report(rows)]}]
        lines, _ = self.live(alone)
        self.assertEqual(len(lines), 1)
        self.assertIn("NOT_RETAINED", lines[0])
        self.assertIn(f"ranks=1 of tp_size={RANKS}", lines[0])
        self.assertNotIn("equal=", lines[0])

    def test_nothing_before_the_boot_ends_or_in_a_capture(self) -> None:
        """The runner's own warmups run under PT_COMPILE_ONLY_MODE, whose host reads hold no real rows."""
        rows = torch.zeros((ROWS, COLUMNS), dtype=torch.int32)
        with mock.patch.dict(os.environ, {FLAG: "eager"}), mock.patch.object(self.graph, "rank_reports") as reports:
            with mock.patch.object(self.graph, "_RANK_CHECK_BOOT_DONE", [False]):
                self.graph.rank_check_eager_rows(FakeOp(3), rows, "k")
            with mock.patch.object(self.graph, "_RANK_CHECK_BOOT_DONE", [True]), self.graph.capture_context():
                self.graph.rank_check_eager_rows(FakeOp(3), rows, "k")
        reports.assert_not_called()

    def test_flag_off_is_nothing(self) -> None:
        with mock.patch.object(self.graph, "rank_reports") as reports, mock.patch.object(self.graph, "_RANK_CHECK_BOOT_DONE", [True]):
            self.graph.rank_check_eager_rows(FakeOp(3), torch.zeros((ROWS, COLUMNS)), "k")
        reports.assert_not_called()


class EagerInTheIndexer(unittest.TestCase):
    def test_the_real_indexer_hands_its_rows_to_the_eager_check_on_the_real_key(self) -> None:
        seen: list[tuple] = []
        with key_stack.layer_stack(OVERLAY_ROOT, {KEYS: f"1024:{TABLE}", FLAG: "eager"}) as world:
            world.kpool._score_pool_chunk = row_scores
            world.r590.rank_check_eager_rows = lambda op, rows, key: seen.append((rows.shape, key))
            key_stack.indexer_step(world, key_stack.prompt_case(1024, TABLE, 2000))
        self.assertEqual(seen, [((1024, key_stack.WIDTH), f"1024:{TABLE}")])


class BootChecks(unittest.TestCase):
    GOOD = {KEYS: f"1024:{TABLE}", "HPU_GLM53_GRAPHS_ONLY": "1"}

    def check(self, flag: str, **env: str) -> None:
        with mock.patch.dict(os.environ, {**self.GOOD, FLAG: flag, **env}):
            graph.check_rank_check_flags(graph.prompt_keys())

    def test_replay_mode_with_keys_passes(self) -> None:
        self.check("1")

    def test_replay_mode_without_keys_refuses(self) -> None:
        with mock.patch.dict(os.environ, {FLAG: "1", KEYS: ""}), self.assertRaisesRegex(ValueError, KEYS):
            graph.check_rank_check_flags(())

    def test_eager_needs_graphs_only_off(self) -> None:
        with self.assertRaisesRegex(ValueError, "HPU_GLM53_GRAPHS_ONLY"):
            self.check("eager")
        self.check("eager", HPU_GLM53_GRAPHS_ONLY="0")

    def test_query_shard_makes_the_check_meaningless_so_the_pair_refuses(self) -> None:
        """Gathered rows match by construction: equal=1 would prove nothing about the replicated rows the shard depends on."""
        with self.assertRaisesRegex(ValueError, SHARD_FLAG):
            self.check("1", **{SHARD_FLAG: f"1024:{TABLE}"})

    def test_the_prompt_shard_refusal_does_not_name_the_decode_flag(self) -> None:
        with self.assertRaises(ValueError) as refused:
            self.check("1", **{SHARD_FLAG: f"1024:{TABLE}", DECODE_SHARD_FLAG: "8:1024"})
        self.assertIn(SHARD_FLAG, str(refused.exception))
        self.assertNotIn(DECODE_SHARD_FLAG, str(refused.exception))

    def test_the_decode_shard_boots_with_the_check_and_the_boot_says_what_the_lines_then_prove(self) -> None:
        """Decode lines of a sharded run read equal=1 by construction. The check is how the rows get logged at all (rows_digest=) for an unsharded run to be
        compared with, so the pair boots; the one note says where the evidence is."""
        with self.assertLogs(graph.LOG, logging.WARNING) as logs:
            self.check("1", **{DECODE_SHARD_FLAG: "8:1024,16:4096"})
        note = [line for line in logs.output if "GLM53_RANK_CHECK_DECODE_SHARD" in line]
        self.assertEqual(len(note), 1, logs.output)
        for text in ("8:1024,16:4096", "equal=1", "by construction", "rows_digest", "rank_digest_compare.py"):
            self.assertIn(text, note[0])

    def test_there_is_no_decode_shard_note_without_the_decode_shard_or_in_eager_mode(self) -> None:
        with self.assertNoLogs(graph.LOG, logging.WARNING):
            self.check("1")
            self.check("1", **{DECODE_SHARD_FLAG: ""})
            self.check("eager", HPU_GLM53_GRAPHS_ONLY="0", **{DECODE_SHARD_FLAG: "8:1024"})

    def test_off_never_refuses(self) -> None:
        with mock.patch.dict(os.environ, {FLAG: "0", KEYS: "", SHARD_FLAG: f"1024:{TABLE}"}):
            graph.check_rank_check_flags(())

    def test_the_boot_check_calls_it(self) -> None:
        with mock.patch.dict(os.environ, {KEYS: "", "HPU_GLM53_DECODE_WIDTH_PAD": "0", "VLLM_GAUDI_MIXED_DECODE_WIDTHS": "", "HPU_GLM53_DECODE_PAD_CHECK_WIDTHS": ""}), \
                mock.patch.object(graph, "check_rank_check_flags") as check:
            graph.check_prompt_key_flags()
        check.assert_called_once_with(())


if __name__ == "__main__":
    unittest.main()
