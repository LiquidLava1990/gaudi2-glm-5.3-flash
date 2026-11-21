"""cells/shard_shape_probe.py, the on-card proof that a decode key's row split leaves the scorer's floats and the selected ids as they were.

CPU only: the real kpool and indexer from key_stack, the probe's own inputs, and scorers that are wrong on purpose (scorer_mutants). A card
decides the real question; these tests decide that the probe would see an answer other than "exact", and would not cry wolf.
"""
from __future__ import annotations

import contextlib
import io
import itertools
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "cells"))

import key_stack  # noqa: E402
import scorer_mutants  # noqa: E402
import shard_shape_probe as probe  # noqa: E402
from overlay_testlib import OVERLAY_ROOT  # noqa: E402

CPU = torch.device("cpu")
# 4,096 tokens (32 blocks) is 1,024 pools, twice the 512 a selection keeps. Chunks: 512 pools at batch > 1, 1,024 at batch 1, as 16,384 and 65,536 are on the card.
CHUNK_ENV = {"HPU_GLM53_CONTEXT_CHUNK_POOLS": "1024", "HPU_GLM53_MULTI_BATCH_KPOOL_TILE_CEILING": "512", "HPU_GLM53_MLA_DECODESEL": "1"}
BLOCKS = 32
ONE_ULP_FAULTS = (scorer_mutants.UP_EVERY_NTH_COLUMN, scorer_mutants.DOWN_EVERY_NTH_COLUMN)


def case(rows: int, kind: str = "ladder", profile: str = "full") -> probe.ProbeCase:
    return probe.ProbeCase(rows, BLOCKS, profile, kind)


class ProbeTestCase(unittest.TestCase):
    def stack(self, **env: str) -> contextlib.AbstractContextManager:
        return key_stack.layer_stack(OVERLAY_ROOT, {**CHUNK_ENV, **env})

    def prove(self, rows: int, kind: str = "ladder", fault: scorer_mutants.Fault | None = None, **env: str) -> probe.ShapeResult:
        with self.stack(**env) as world:
            with scorer_mutants.installed(world.kpool, fault) if fault else contextlib.nullcontext():
                return probe.run_proof(world, [case(rows, kind)], CPU)[0]


class TheRealScorer(ProbeTestCase):
    def test_every_shape_pair_and_both_kinds_of_input_are_exact(self) -> None:
        for rows in (4, 8, 16):
            for kind in probe.KINDS:
                with self.subTest(rows=rows, kind=kind):
                    result = self.prove(rows, kind)
                    self.assertEqual(result.verdict, probe.EXACT, result)
                    self.assertEqual((result.scores.differ, result.ids_rows_differ, result.ids_rows), (0, 0, rows))
                    self.assertGreater(result.scores.cells, 0)

    def test_the_two_runs_of_a_pair_are_the_whole_key_and_one_rank_of_it(self) -> None:
        for rows in (4, 8, 16):
            with self.subTest(rows=rows):
                result = self.prove(rows)
                self.assertEqual((result.whole.rows, result.split.rows), (rows, rows // probe.TP))


class TheScanIsTheRealOne(ProbeTestCase):
    """Chunk, bucket and route come from the indexer's own _scoring_chunk and _scan_pool_bucket, not from numbers copied into the probe."""

    def test_a_rank_with_one_row_scans_in_the_wide_chunk_and_changes_route(self) -> None:
        result = self.prove(4)
        self.assertEqual((result.whole.chunk, result.split.chunk), (512, 1024))
        self.assertEqual((result.whole.bucket, result.split.bucket), (1024, 1024))
        self.assertEqual((result.whole.route, result.split.route), ("decodesel", "fused"))

    def test_two_and_four_rows_per_rank_keep_the_chunk_and_the_route(self) -> None:
        for rows in (8, 16):
            with self.subTest(rows=rows):
                result = self.prove(rows)
                self.assertEqual((result.whole.chunk, result.split.chunk), (512, 512))
                self.assertEqual((result.whole.route, result.split.route), ("decodesel", "decodesel"))

    def test_the_bucket_is_the_whole_table_whatever_the_lanes_hold(self) -> None:
        """Real decode hands _write_topk max_sequence_length = table blocks x 128, so the scan covers the table and masks the rest."""
        with self.stack() as world:
            result = probe.run_proof(world, [case(8, "random", "low")], CPU)[0]
        self.assertEqual(result.whole.bucket, BLOCKS * probe.POOLS_PER_BLOCK)


class TheInputs(ProbeTestCase):
    def build(self, rows: int, kind: str = "ladder", seed: int = 0) -> probe.ProbeInputs:
        with self.stack() as world:
            return probe.build_inputs(world, probe.ProbeCase(rows, BLOCKS, "full", kind), CPU, seed)

    def boundary_cluster(self, kind: str) -> tuple[int, int]:
        """(scores within 8 ULP of the 512th best, distinct values among them) for row 0."""
        with self.stack() as world:
            inputs = probe.build_inputs(world, probe.ProbeCase(4, BLOCKS, "full", kind), CPU, 0)
            scores = probe.scan_scores(world, inputs, slice(0, 1), 1024, 1024)[0]
        ranked = scores.sort(descending=True).values
        boundary = ranked[probe.TOP_POOLS - 1]
        ulp = torch.nextafter(boundary, torch.tensor(torch.inf)) - boundary
        near = ranked[(ranked - boundary).abs() <= 8 * ulp]
        return near.numel(), near.unique().numel()

    def test_the_ladder_puts_the_selection_boundary_inside_scores_a_few_ULPs_apart(self) -> None:
        count, distinct = self.boundary_cluster("ladder")
        self.assertGreaterEqual(count, probe.CLUSTER_POOLS // 2)
        self.assertGreaterEqual(distinct, 4, "a cluster of one value would be a tie, not a near-tie")

    def test_random_inputs_have_no_such_cluster(self) -> None:
        count, _ = self.boundary_cluster("random")
        self.assertLess(count, 10)

    @staticmethod
    def comparable(inputs: probe.ProbeInputs) -> tuple[torch.Tensor, ...]:
        return inputs.q_values.view(torch.uint8), inputs.weights, inputs.table, inputs.lengths, inputs.cache

    def test_a_seed_gives_the_same_inputs_and_another_seed_others(self) -> None:
        first, again, other = self.build(4), self.build(4), self.build(4, seed=1)
        for left, right in zip(self.comparable(first), self.comparable(again)):
            self.assertTrue(torch.equal(left, right))
        self.assertFalse(torch.equal(first.cache, other.cache))

    def test_rows_are_nested_so_every_key_of_a_lattice_scores_the_same_lanes(self) -> None:
        sixteen, eight = self.build(16), self.build(8)
        head = sixteen.head(8)
        self.assertTrue(torch.equal(head.q_values.view(torch.uint8), eight.q_values.view(torch.uint8)))
        for name in ("weights", "table", "lengths"):
            self.assertTrue(torch.equal(getattr(head, name), getattr(eight, name)), name)
        rows = 8 * BLOCKS * probe.POOLS_PER_BLOCK
        self.assertTrue(torch.equal(sixteen.cache[:rows], eight.cache[:rows]))

    def test_each_row_owns_blocks_of_its_own_and_has_its_own_length(self) -> None:
        inputs = self.build(8)
        blocks = [set(row.tolist()) for row in inputs.table]
        self.assertTrue(all(first.isdisjoint(second) for first, second in itertools.combinations(blocks, 2)))
        self.assertEqual(len(set(inputs.lengths.tolist())), 8)
        self.assertEqual(int(inputs.lengths.max()), BLOCKS * probe.BLOCK_SIZE)


class TheFaults(ProbeTestCase):
    """A scorer that is wrong only at some batch sizes: killed on the floats everywhere, and on the ids when the inputs have near-ties."""

    def test_every_batch_dependent_fault_is_seen_on_every_pair_in_floats_and_in_ids(self) -> None:
        for fault in scorer_mutants.FAULTS:
            for rows in (4, 8, 16):
                with self.subTest(fault=fault.name, rows=rows):
                    result = self.prove(rows, "ladder", fault)
                    self.assertGreater(result.scores.differ, 0)
                    self.assertGreater(result.ids_rows_differ, 0)
                    self.assertEqual(result.verdict, probe.IDS_DIFFER)

    def test_a_one_ulp_drift_of_random_inputs_moves_the_floats_but_not_one_id(self) -> None:
        """Why the near-tie ladder is there: continuous scores sit far apart, so the same fault leaves the ids alone."""
        for fault in ONE_ULP_FAULTS:
            for rows in (4, 8, 16):
                with self.subTest(fault=fault.name, rows=rows):
                    result = self.prove(rows, "random", fault)
                    self.assertGreater(result.scores.differ, 0)
                    self.assertEqual(result.ids_rows_differ, 0)
                    self.assertEqual(result.verdict, probe.FLOATS_DIFFER)

    def test_a_one_ulp_drift_is_one_ulp(self) -> None:
        result = self.prove(8, "random", scorer_mutants.UP_EVERY_NTH_COLUMN)
        self.assertEqual(result.scores.max_ulp, 1)

    def test_a_fault_that_ignores_the_batch_is_not_a_difference_between_shapes(self) -> None:
        for rows in (4, 8, 16):
            with self.subTest(rows=rows):
                self.assertEqual(self.prove(rows, "ladder", scorer_mutants.CONTROL).verdict, probe.EXACT)

    def test_a_scorer_that_differs_from_itself_proves_nothing(self) -> None:
        """Run to run noise would read as a shape effect: the probe repeats the whole key first and refuses to judge when it disagrees with itself."""
        calls = itertools.count()
        every_third_call = scorer_mutants.Fault("1 ULP on every column of every third call", lambda scores: scorer_mutants.one_ulp(
            scores, torch.inf, scorer_mutants.columns_at(scores, 1, 0)) if next(calls) % 3 == 0 else scores)
        result = self.prove(8, "ladder", every_third_call)
        self.assertEqual(result.verdict, probe.NOT_MEASURED)
        self.assertTrue(result.null_scores_differ > 0 or result.null_ids_rows_differ > 0)


class AFailedCase(ProbeTestCase):
    def test_an_error_is_the_cases_own_not_measured_and_the_others_still_run(self) -> None:
        with self.stack() as world:
            real = world.indexer.HPUGlm5NextSparseIndexerOp._write_topk
            seen = itertools.count()

            def fail_once(op: object, *args: object, **kwargs: object) -> torch.Tensor:
                if next(seen) == 0:
                    raise RuntimeError("synStatus 26 PT_DEVMEM in the scan")
                return real(op, *args, **kwargs)

            with mock.patch.object(world.indexer.HPUGlm5NextSparseIndexerOp, "_write_topk", fail_once):
                results = probe.run_proof(world, [case(8), case(4)], CPU)
        self.assertEqual([result.verdict for result in results], [probe.NOT_MEASURED, probe.EXACT])
        self.assertIn("PT_DEVMEM", results[0].error)
        self.assertIn(case(8).label, probe.row_line(results[0]))


class TheReport(ProbeTestCase):
    def test_a_row_line_carries_floats_and_ids_separately(self) -> None:
        line = probe.row_line(self.prove(4))
        for field in ("SHARD_SHAPE_ROW ", f"case={case(4).label} ", "rows=4->1 ", "chunk=512->1024 ", "route=decodesel->fused ", "null=EQUAL ",
                      "score_bits=EQUAL ", "cols_differ=0/", "max_ulp=0 ", "ids=EQUAL ", "id_rows_differ=0/4", "verdict=EXACT"):
            self.assertIn(field, line)

    def test_a_faulty_pair_reads_different_in_both(self) -> None:
        line = probe.row_line(self.prove(8, "ladder", scorer_mutants.UP_EVERY_NTH_COLUMN))
        for field in ("score_bits=DIFFER ", "ids=DIFFER ", "verdict=IDS_DIFFER"):
            self.assertIn(field, line)
        self.assertNotIn("cols_differ=0/", line)

    def test_the_exit_code_ranks_the_verdicts(self) -> None:
        def result(verdict: str) -> SimpleNamespace:
            return SimpleNamespace(verdict=verdict)

        every = {probe.EXACT: 0, probe.IDS_DIFFER: 1, probe.NOT_MEASURED: 2, probe.FLOATS_DIFFER: 3}
        for verdict, code in every.items():
            self.assertEqual(probe.exit_code([result(probe.EXACT), result(verdict)]), code, verdict)
        self.assertEqual(probe.exit_code([result(probe.FLOATS_DIFFER), result(probe.NOT_MEASURED), result(probe.IDS_DIFFER)]), 1)
        self.assertEqual(probe.exit_code([result(probe.FLOATS_DIFFER), result(probe.NOT_MEASURED)]), 2)
        self.assertEqual(probe.exit_code([]), 2, "no case ran")

    def test_the_summary_counts_each_verdict(self) -> None:
        results = [SimpleNamespace(verdict=verdict) for verdict in (probe.EXACT, probe.EXACT, probe.FLOATS_DIFFER, probe.IDS_DIFFER, probe.NOT_MEASURED)]
        self.assertEqual(probe.summary_line(results), "SHARD_SHAPE_PROOF status=IDS_DIFFER cases=5 exact=2 floats_differ=1 ids_differ=1 not_measured=1")


class TheEnvironment(ProbeTestCase):
    PRODUCTION = {"_USE_NATIVE_FP8_KPOOL_SCORE": True, "_USE_BF16_KPOOL_SCORE": True, "_USE_DECODE_SELECT": True, "_USE_PARTIAL_SCAN_TOPK": True,
                  "_USE_SMALL_MERGE_TOPK": True, "_USE_FUSED_PREFILL_SELECT": True, "_tpc_topk": object()}

    def test_a_stack_that_is_not_the_production_scorer_says_what_differs(self) -> None:
        with self.stack() as world:
            found = probe.env_mismatches(world)
        for name in ("native FP8", "BF16", "k512", "context chunk"):
            self.assertTrue(any(name in line for line in found), (name, found))

    def test_the_production_scorer_has_no_mismatch(self) -> None:
        with self.stack(HPU_GLM53_CONTEXT_CHUNK_POOLS="65536", HPU_GLM53_MULTI_BATCH_KPOOL_TILE_CEILING="16384") as world:
            for name, value in self.PRODUCTION.items():
                setattr(world.kpool, name, value)
            self.assertEqual(probe.env_mismatches(world), [])


class Main(ProbeTestCase):
    def run_main(self, *argv: str) -> tuple[int, list[str]]:
        out = io.StringIO()
        with self.stack() as world, mock.patch.object(probe, "load_stack", return_value=world), contextlib.redirect_stdout(out):
            code = probe.main(["--device", "cpu", "--env-check", "off", "--blocks", str(BLOCKS), "--profiles", "full", *argv])
        return code, out.getvalue().splitlines()

    def test_a_run_prints_a_row_per_case_and_kind_then_the_summary_and_exits_zero_when_exact(self) -> None:
        code, lines = self.run_main("--rows", "4,8")
        self.assertEqual(code, 0)
        rows = [line for line in lines if line.startswith("SHARD_SHAPE_ROW ")]
        self.assertEqual(len(rows), 4)
        self.assertEqual(lines[-1], "SHARD_SHAPE_PROOF status=EXACT cases=4 exact=4 floats_differ=0 ids_differ=0 not_measured=0")

    def test_a_faulty_scorer_exits_one(self) -> None:
        with self.stack() as world, scorer_mutants.installed(world.kpool, scorer_mutants.ADJACENT_FLIP), \
                mock.patch.object(probe, "load_stack", return_value=world), contextlib.redirect_stdout(io.StringIO()):
            code = probe.main(["--device", "cpu", "--env-check", "off", "--blocks", str(BLOCKS), "--profiles", "full", "--rows", "8", "--kinds", "ladder"])
        self.assertEqual(code, 1)

    def test_out_gets_one_json_line_per_case(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            target = Path(folder) / "proof.jsonl"
            code, _ = self.run_main("--rows", "8", "--kinds", "ladder", "--out", str(target))
            written = [json.loads(line) for line in target.read_text(encoding="utf-8").splitlines()]
        self.assertEqual(code, 0)
        self.assertEqual(len(written), 1)
        self.assertEqual((written[0]["verdict"], written[0]["case"]), (probe.EXACT, case(8).label))

    def test_a_production_mismatch_stops_before_any_case_unless_the_check_is_off(self) -> None:
        out = io.StringIO()
        with self.stack() as world, mock.patch.object(probe, "load_stack", return_value=world), contextlib.redirect_stdout(out):
            code = probe.main(["--device", "cpu", "--blocks", str(BLOCKS), "--profiles", "full", "--rows", "8"])
        self.assertEqual(code, 2)
        self.assertIn("SHARD_SHAPE_ENV_MISMATCH", out.getvalue())
        self.assertNotIn("SHARD_SHAPE_ROW ", out.getvalue())

    def test_rows_that_do_not_divide_by_the_ranks_refuse(self) -> None:
        with self.assertRaisesRegex(SystemExit, "divide"):
            probe.main(["--device", "cpu", "--rows", "6"])


class TestFile(unittest.TestCase):
    def test_this_file_runs_against_the_overlay(self) -> None:
        self.assertEqual(Path(probe.__file__).resolve().parents[1], OVERLAY_ROOT, "set PYTHONPATH to the overlay, as run_tests.sh does")


if __name__ == "__main__":
    unittest.main()
