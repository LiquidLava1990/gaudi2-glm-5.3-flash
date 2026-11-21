"""cells/rank_check_read.py: the verdict on a card run's GLM53_RANK_CHECK lines. PASS only on a complete, voidless, equal reading; a void hiding
among equal=1 lines, a missing layer or a missing long-context reading must never read as PASS."""
from __future__ import annotations

import contextlib
import io
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "cells"))

import rank_check_read as reader  # noqa: E402

LAYERS, KEY = 11, "1024:8192"
PREFIX = "(Worker_TP0 pid=412) WARNING 11-07 05:41:07 [glm53_mixed_graph.py:1380]"


def verdict(layer: int, start: int, replay: int, *, equal: int = 1, ranks: int = 4, differ: int = 0, where: str | None = None) -> str:
    where = where or f"start={start} replay={replay}"
    return (f"{PREFIX} GLM53_RANK_CHECK key={KEY} layer={layer} equal={equal} rows=1024 rows_differ={differ} live_rows=1024 ranks={ranks} {where}")


def void(layer: int, start: int, replay: int, reason: str = "stale_rows=1024 of 1024 live rows on rank 3: this replay left them as the previous replay wrote them") -> str:
    return (f"{PREFIX} GLM53_RANK_CHECK_NOT_RETAINED key={KEY} layer={layer} start={start} replay={replay} reason={reason}. "
            "No verdict on these rows; fix the read, or rerun non-serving with HPU_GLM53_INDEXER_RANK_CHECK=eager and HPU_GLM53_GRAPHS_ONLY=0")


def chunk(start: int, replay: int, skip: tuple[int, ...] = (), **kwargs: int) -> list[str]:
    return [verdict(layer, start, replay, **kwargs) for layer in range(3, 3 + LAYERS) if layer not in skip]


def good_log() -> list[str]:
    noise = [f"{PREFIX} GLM53_MIXED_GRAPH captured bucket=(0, 1024, 8192)", f"{PREFIX} GLM53_RANK_CHECK_READY layers={LAYERS} rows=1024 bytes_each=8192"]
    return [*noise, *chunk(131072, 1), *noise[:1], *chunk(160000, 2), *chunk(540000, 3)]


class Passing(unittest.TestCase):
    def test_a_complete_equal_log_with_both_long_readings_passes(self) -> None:
        result = reader.judge(good_log())
        self.assertEqual((result.status, result.reasons), ("PASS", ()))
        self.assertEqual(result.exit_code, 0)
        self.assertIn("3 chunks", result.summary)
        self.assertIn(f"{LAYERS} layers", result.summary)

    def test_the_min_start_arguments_set_the_long_readings_required(self) -> None:
        self.assertEqual(reader.judge(chunk(131072, 1), min_starts=(100_000,)).status, "PASS")
        self.assertEqual(reader.judge(chunk(131072, 1), min_starts=(150_000,)).status, "NOT_MEASURED")

    def test_one_chunk_past_every_minimum_satisfies_them_all(self) -> None:
        self.assertEqual(reader.judge(chunk(540000, 1)).status, "PASS")

    def test_a_start_exactly_at_the_minimum_counts(self) -> None:
        self.assertEqual(reader.judge(chunk(150_000, 1), min_starts=(150_000,)).status, "PASS")

    def test_the_ready_line_and_other_rank_check_lines_are_not_readings(self) -> None:
        self.assertEqual(len(reader.parse(good_log())[0]), 3 * LAYERS)

    def test_lines_that_end_with_the_row_tag_read_as_they_did_without_it(self) -> None:
        """glm53_mixed_graph appends rows_digest=<16 hex> to every verdict line; the verdict does not use it (cells/rank_digest_compare.py does)."""
        tagged = [f"{line} rows_digest=0123456789abcdef" if "GLM53_RANK_CHECK key=" in line else line for line in good_log()]
        self.assertNotEqual(tagged, good_log())
        self.assertEqual(reader.judge(tagged), reader.judge(good_log()))


class NotMeasured(unittest.TestCase):
    def assert_not_measured(self, lines: list[str], mention: str, **kwargs: object) -> reader.Verdict:
        result = reader.judge(lines, **kwargs)
        self.assertEqual(result.status, "NOT_MEASURED", result)
        self.assertEqual(result.exit_code, 2)
        self.assertTrue(any(mention in reason for reason in result.reasons), (mention, result.reasons))
        return result

    def test_a_void_hiding_among_equal_lines_is_not_a_pass(self) -> None:
        """Eleven layer lines per chunk, ten of them equal=1: counting lines per chunk would call that complete."""
        lines = [*chunk(131072, 1), *chunk(160000, 2, skip=(10,)), void(10, 160000, 2), *chunk(540000, 3)]
        result = self.assert_not_measured(lines, "NOT_RETAINED")
        self.assertTrue(any("layer 10" in reason and "stale_rows=1024" in reason and "start=160000" in reason for reason in result.reasons), result.reasons)

    def test_a_void_after_an_otherwise_complete_log_still_fails_the_pass(self) -> None:
        lines = [*good_log(), void(5, 540000, 4)]
        self.assert_not_measured(lines, "NOT_RETAINED")

    def test_a_layer_missing_from_one_chunk(self) -> None:
        lines = [*chunk(131072, 1), *chunk(160000, 2, skip=(7,)), *chunk(540000, 3)]
        self.assert_not_measured(lines, f"{LAYERS - 1} of {LAYERS} layers")

    def test_a_layer_missing_from_every_chunk_when_the_ready_line_says_how_many(self) -> None:
        lines = [f"{PREFIX} GLM53_RANK_CHECK_READY layers={LAYERS} rows=1024 bytes_each=8192",
                 *chunk(160000, 1, skip=(7,)), *chunk(540000, 2, skip=(7,))]
        self.assert_not_measured(lines, f"{LAYERS - 1} of {LAYERS} layers")

    def test_the_layers_argument_overrides_what_the_log_shows(self) -> None:
        self.assert_not_measured(good_log(), f"{LAYERS} of {LAYERS + 1} layers", layers=LAYERS + 1)

    def test_a_reading_from_fewer_ranks_than_expected(self) -> None:
        lines = [*chunk(160000, 1), *chunk(540000, 2, ranks=2)]
        self.assert_not_measured(lines, "ranks=2")

    def test_an_eager_line_is_not_a_replay_reading(self) -> None:
        lines = [*good_log(), verdict(3, 0, 0, where="eager=1")]
        self.assert_not_measured(lines, "eager=1")

    def test_a_layer_logged_twice_in_a_chunk_means_two_runs_in_one_log(self) -> None:
        lines = [*good_log(), verdict(3, 540000, 3)]
        self.assert_not_measured(lines, "twice")

    def test_a_log_failed_line_is_a_missing_layer(self) -> None:
        failed = f"{PREFIX} GLM53_RANK_CHECK_LOG_FAILED key={KEY} layer=7 start=160000 replay=2 error=IndexError: list index out of range. No line for this layer"
        lines = [*chunk(131072, 1), *chunk(160000, 2, skip=(7,)), failed, *chunk(540000, 3)]
        self.assert_not_measured(lines, "LOG_FAILED")

    def test_no_reading_past_the_first_long_start(self) -> None:
        self.assert_not_measured([*chunk(131072, 1), *chunk(160000, 2)], "start >= 530000")

    def test_a_chunk_below_both_long_starts_reads_but_covers_neither(self) -> None:
        result = self.assert_not_measured(chunk(131072, 1), "start >= 150000")
        self.assertTrue(any("start >= 530000" in reason for reason in result.reasons), result.reasons)

    def test_a_long_start_that_is_voided_does_not_count_as_a_reading(self) -> None:
        lines = [*chunk(160000, 1), *chunk(540000, 2, skip=(5,)), void(5, 540000, 2)]
        self.assert_not_measured(lines, "start >= 530000")

    def test_no_check_lines_at_all(self) -> None:
        self.assert_not_measured([f"{PREFIX} GLM53_MIXED_GRAPH captured bucket=(0, 1024, 8192)"], "no GLM53_RANK_CHECK")
        self.assert_not_measured([], "no GLM53_RANK_CHECK")

    def test_the_reasons_are_bounded_but_counted(self) -> None:
        lines = [void(layer, 131072 + replay, replay) for replay in range(1, 40) for layer in range(3, 14)]
        result = reader.judge(lines)
        self.assertLessEqual(len(result.reasons), reader.MAX_REASONS + 1)
        self.assertIn("more", result.reasons[-1])


class Failing(unittest.TestCase):
    def test_one_unequal_layer_among_many_equal_ones_fails(self) -> None:
        lines = [*chunk(131072, 1), *chunk(160000, 2, skip=(9,)), verdict(9, 160000, 2, equal=0, differ=37), *chunk(540000, 3)]
        result = reader.judge(lines)
        self.assertEqual((result.status, result.exit_code), ("FAIL", 1))
        self.assertTrue(any("layer 9" in reason and "rows_differ=37" in reason and "start=160000" in reason for reason in result.reasons), result.reasons)

    def test_a_measured_disagreement_outranks_a_void_elsewhere(self) -> None:
        lines = [*chunk(160000, 1, skip=(4,)), verdict(4, 160000, 1, equal=0, differ=3), void(8, 540000, 2)]
        result = reader.judge(lines)
        self.assertEqual(result.status, "FAIL")
        self.assertTrue(any("NOT_RETAINED" in reason for reason in result.reasons), "the void is still said")

    def test_an_unequal_eager_line_fails_too(self) -> None:
        result = reader.judge([*good_log(), verdict(3, 0, 0, equal=0, differ=2, where="eager=1")])
        self.assertEqual(result.status, "FAIL")


class Command(unittest.TestCase):
    def run_main(self, lines: list[str], *args: str) -> tuple[int, str]:
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "container.log"
            path.write_text("\n".join(lines) + "\n", encoding="utf-8")
            out = io.StringIO()
            with contextlib.redirect_stdout(out):
                code = reader.main([str(path), *args])
        return code, out.getvalue()

    def test_pass_prints_pass_and_exits_0(self) -> None:
        code, text = self.run_main(good_log())
        self.assertEqual(code, 0)
        self.assertTrue(text.startswith("PASS"), text)

    def test_a_void_prints_why_and_exits_2(self) -> None:
        code, text = self.run_main([*chunk(160000, 1, skip=(5,)), void(5, 160000, 1), *chunk(540000, 2)])
        self.assertEqual(code, 2)
        self.assertTrue(text.startswith("NOT_MEASURED"), text)
        self.assertIn("stale_rows=1024", text)

    def test_unequal_exits_1(self) -> None:
        code, text = self.run_main([*chunk(160000, 1, skip=(5,)), verdict(5, 160000, 1, equal=0, differ=9), *chunk(540000, 2)])
        self.assertEqual(code, 1)
        self.assertTrue(text.startswith("FAIL"), text)

    def test_arguments_reach_the_judge(self) -> None:
        code, _ = self.run_main(chunk(131072, 1), "--min-start", "100000")
        self.assertEqual(code, 0)
        code, _ = self.run_main(good_log(), "--ranks", "8")
        self.assertEqual(code, 2)
        code, _ = self.run_main(good_log(), "--layers", "10")
        self.assertEqual(code, 2)

    def test_a_log_that_cannot_be_read_is_not_measured_and_says_where(self) -> None:
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            code = reader.main(["/nonexistent/container.log"])
        self.assertEqual(code, 2)
        self.assertIn("/nonexistent/container.log", out.getvalue())


if __name__ == "__main__":
    unittest.main()
