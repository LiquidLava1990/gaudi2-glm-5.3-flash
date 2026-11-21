"""R6: the sparse indexer reads max_sequence_length (and, from it, the scan bucket and the exact-causal branch) as Python ints at capture.
Under PT_COMPILE_ONLY_MODE nothing proves a device scalar read there is a real value, so a key's capture logs what each rank read, and a
rank that read another value, or a branch that does not match the dummy's own length, refuses the boot on every rank."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent))

import key_stack  # noqa: E402
from overlay_testlib import OVERLAY_ROOT  # noqa: E402
from test_decode_lattice import MIB, FakeRunner, GraphCase  # noqa: E402
from test_prompt_keys import Backend, KeyCase, RowForward  # noqa: E402

REAL = [("max_seq", 131_072), ("exact_causal", 0), ("topk_tokens", 2048), ("pool_size", 4), ("scan_bucket", 32_768)]


class Recording(GraphCase):
    def test_outside_a_capture_nothing_is_kept(self) -> None:
        self.assertEqual(self.graph.record_capture_scalar("max_seq", 9), 9)
        self.assertEqual(self.graph.take_capture_scalars(), [])

    def test_a_key_capture_and_a_decode_key_capture_keep_what_they_read(self) -> None:
        with self.graph.key_capture_context():
            self.graph.record_capture_scalar("max_seq", 9)
        with self.graph.decode_key_context((1, 1024)):
            self.graph.record_capture_scalar("scan_bucket", 4)
        self.assertEqual(self.graph.take_capture_scalars(), [("max_seq", 9), ("scan_bucket", 4)])
        self.assertEqual(self.graph.take_capture_scalars(), [])

    def test_the_branch_helper_records_the_four_numbers_in_a_capture_only(self) -> None:
        with self.graph.key_capture_context():
            self.graph.record_capture_branch(5000, 2048, 4)
        self.assertEqual(self.graph.take_capture_scalars(), [("max_seq", 5000), ("exact_causal", 0), ("topk_tokens", 2048), ("pool_size", 4)])
        self.graph.record_capture_branch(5, 2048, 4)
        self.assertEqual(self.graph.take_capture_scalars(), [])


class Checking(GraphCase):
    def check(self, scalars: list[tuple[str, int]], expected: int = 131_072, label: str = "1024:8192") -> list[str]:
        with self.graph.key_capture_context():
            for name, value in scalars:
                self.graph.record_capture_scalar(name, value)
        with self.assertLogs(self.graph.LOG, "WARNING") as logs:
            self.graph.check_capture_scalars(label, expected)
        return logs.output

    def test_real_values_log_one_line_per_rank_and_pass(self) -> None:
        output = self.check(REAL)
        line = next(line for line in output if "GLM53_CAPTURE_SCALARS" in line)
        for needle in ("key=1024:8192", "rank=0", "max_seq=131072", "exact_causal=0", "scan_bucket=32768", "expected_max_seq=131072"):
            self.assertIn(needle, line)

    def test_the_scalars_are_cleared_for_the_next_capture(self) -> None:
        self.check(REAL)
        self.assertEqual(self.graph.take_capture_scalars(), [])

    def test_the_exact_causal_branch_on_a_long_dummy_refuses_with_both_values(self) -> None:
        with self.assertRaises(RuntimeError) as refused:
            self.check([("max_seq", 5), ("exact_causal", 1), ("topk_tokens", 2048)])
        for needle in ("1024:8192", "exact_causal=1", "131072", "max_seq=5", "PT_COMPILE_ONLY_MODE"):
            self.assertIn(needle, str(refused.exception))

    def test_a_zero_read_for_a_long_dummy_refuses(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "max_seq=0"):
            self.check([("max_seq", 0), ("exact_causal", 0), ("topk_tokens", 2048)])

    def test_a_wrong_nonzero_value_on_the_right_branch_refuses(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "max_seq=65536"):
            self.check([("max_seq", 65_536), ("exact_causal", 0), ("topk_tokens", 2048)])

    def test_a_scan_bucket_that_is_not_the_keys_whole_width_refuses(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "scan_bucket=4096"):
            self.check([("max_seq", 131_072), ("exact_causal", 0), ("topk_tokens", 2048), ("pool_size", 4), ("scan_bucket", 4096)])

    def test_one_wrong_read_among_a_captures_many_layers_refuses(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "scan_bucket=4096,32768"):
            self.check(REAL + [("scan_bucket", 4096)])

    def test_a_short_dummy_must_take_the_exact_causal_branch(self) -> None:
        self.check([("max_seq", 100), ("exact_causal", 1), ("topk_tokens", 2048)], expected=100)
        with self.assertRaisesRegex(RuntimeError, "exact_causal=0"):
            self.check([("max_seq", 100), ("exact_causal", 0), ("topk_tokens", 2048)], expected=100)

    def test_a_capture_that_read_none_passes(self) -> None:
        """A capture whose layers never reach the host read (static shapes) has nothing to prove."""
        self.check([])

    def test_ranks_that_read_different_values_all_refuse(self) -> None:
        other = [("max_seq", 0), ("exact_causal", 1), ("topk_tokens", 2048), ("scan_bucket", 0)]
        peers = [{"scalars": REAL}, {"scalars": other}]
        gather = lambda report: peers if "scalars" in report else [report, report]  # noqa: E731
        with mock.patch.object(self.graph, "rank_reports", side_effect=gather), self.assertRaisesRegex(RuntimeError, "rank 1 read"):
            self.check(REAL)


class WiredIn(KeyCase):
    def test_a_prompt_key_capture_checks_against_its_own_table_tokens(self) -> None:
        cache = self.graph.GraphCache(RowForward(), 128, Backend())
        with mock.patch.object(self.graph, "check_capture_scalars") as check:
            self.boot_capture(cache, self.prompt_batch(1024, 0, 8192))
        check.assert_called_once_with("1024:8192", 8192 * 128)

    def test_a_refusal_from_the_check_fails_the_capture(self) -> None:
        cache = self.graph.GraphCache(RowForward(), 128, Backend())
        with mock.patch.object(self.graph, "check_capture_scalars", side_effect=RuntimeError("read 0")), self.assertRaisesRegex(RuntimeError, "read 0"):
            self.boot_capture(cache, self.prompt_batch(1024, 0, 8192))


class DecodeKeys(GraphCase):
    def test_a_decode_key_capture_checks_against_its_long_row(self) -> None:
        runner = FakeRunner(self.graph)
        seen: list[tuple] = []
        with mock.patch.object(self.graph, "check_capture_scalars", side_effect=lambda *args: seen.append(args)), \
                mock.patch.object(self.graph, "_in_use_bytes", return_value=0), mock.patch.object(self.graph, "_free_hbm", return_value=0), \
                mock.patch.object(self.graph, "_allocator_headroom", return_value=100_000 * MIB), \
                mock.patch.object(self.graph, "_reset_peak_in_use"), mock.patch.object(self.graph, "_peak_in_use_bytes", return_value=0), \
                mock.patch.object(self.graph, "_graph_count", side_effect=lambda: len(runner.calls)), \
                mock.patch.object(self.graph, "_hold_ballast", return_value=MIB), self.assertLogs(self.graph.LOG, "WARNING"):
            self.graph.warmup_decode_lattice(runner)
        self.assertEqual(seen[0], ("16x8192", 8192 * 128), "the sparse route's static bound: table width x block size, on prompt and decode keys alike")
        self.assertEqual(len(seen), len(self.graph.decode_keys()))


class IndexerSites(unittest.TestCase):
    """The real indexer in a key capture: the branch and the scan bucket it takes are recorded."""

    def read(self, rows: int, columns: int) -> list[tuple[str, int]]:
        with key_stack.layer_stack(OVERLAY_ROOT) as world:
            world.kpool._score_pool_chunk = key_stack.fast_scores
            graph = sys.modules["glm53_mixed_graph"]
            graph.take_capture_scalars()
            with graph.key_capture_context():
                key_stack.indexer_step(world, key_stack.prompt_case(rows, columns))
            return graph.take_capture_scalars()

    def test_a_short_chunk_records_the_exact_causal_branch(self) -> None:
        scalars = dict(self.read(16, 8))
        self.assertEqual(scalars["exact_causal"], 1)
        self.assertLessEqual(scalars["max_seq"], scalars["topk_tokens"])

    def test_a_long_chunk_records_the_scorer_branch_and_its_scan_bucket(self) -> None:
        scalars = dict(self.read(16, 1024))
        self.assertEqual(scalars["exact_causal"], int(scalars["max_seq"] <= scalars["topk_tokens"]))
        if not scalars["exact_causal"]:
            self.assertIn("scan_bucket", scalars)

    def real_reads(self, scope: str) -> dict[str, int]:
        with key_stack.layer_stack(OVERLAY_ROOT) as world:
            world.kpool._score_pool_chunk = key_stack.fast_scores
            graph = sys.modules["glm53_mixed_graph"]
            graph.take_capture_scalars()
            context = graph.key_capture_context() if scope == "prompt" else graph.decode_key_context((16, 1024))
            with context:
                key_stack.indexer_step(world, key_stack.prompt_case(16, 1024))
            reads = graph.take_capture_scalars()
            self.assertEqual(graph._capture_scalar_error("1024", reads, 1024 * key_stack.BLOCK), "", "the real indexer must pass its own check")
            return dict(reads)

    def test_a_key_of_table_width_w_reads_w_blocks_and_scans_them_all(self) -> None:
        for scope in ("prompt", "decode"):
            with self.subTest(scope=scope):
                reads = self.real_reads(scope)
                self.assertEqual(reads["max_seq"], 1024 * key_stack.BLOCK)
                self.assertEqual(reads["scan_bucket"], 1024 * key_stack.BLOCK // key_stack.POOL)

    def test_outside_a_capture_the_indexer_records_nothing(self) -> None:
        with key_stack.layer_stack(OVERLAY_ROOT) as world:
            world.kpool._score_pool_chunk = key_stack.fast_scores
            graph = sys.modules["glm53_mixed_graph"]
            graph.take_capture_scalars()
            key_stack.indexer_step(world, key_stack.prompt_case(16, 8))
            self.assertEqual(graph.take_capture_scalars(), [])


if __name__ == "__main__":
    unittest.main()
