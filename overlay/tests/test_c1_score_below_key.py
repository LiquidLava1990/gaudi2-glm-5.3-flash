"""C1 scorer, GLM53_DECODE_BELOW_KEY: a decode narrower than the lowest key is served padded and never compared. The scorer counts those
lines and says so; they are not pad-equality evidence and never reach decode_pad_merge."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "cells"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import c1_score as scorer  # noqa: E402
from test_c1_score import RANKS, drive_rows, mock_log, reading, score, who  # noqa: E402


def below_key_line(rank: int, native: int, key: int = 1024) -> str:
    return (f"{who(rank)}GLM53_DECODE_BELOW_KEY native={native} key={key}: a short context pads up to the lowest key, "
            "with no native key to compare against")


def log_with_below_key(*widths: int, ranks: tuple[int, ...] = RANKS) -> list[str]:
    return mock_log() + [below_key_line(rank, width) for rank in ranks for width in widths]


class Counting(unittest.TestCase):
    def test_lines_and_distinct_widths_are_counted_over_ranks(self) -> None:
        result = score(log_with_below_key(32, 64))
        self.assertEqual(result["decode_below_key"], {"lines": 8, "widths": [32, 64], "ranks": 4})

    def test_a_log_without_the_line_reports_none(self) -> None:
        self.assertEqual(score()["decode_below_key"], {"lines": 0, "widths": [], "ranks": 0})

    def test_one_rank_alone_is_counted_as_one_rank(self) -> None:
        result = score(log_with_below_key(8, ranks=(2,)))
        self.assertEqual(result["decode_below_key"], {"lines": 1, "widths": [8], "ranks": 1})

    def test_widths_are_sorted_numerically(self) -> None:
        self.assertEqual(score(log_with_below_key(512, 8, 64))["decode_below_key"]["widths"], [8, 64, 512])


class NeverProof(unittest.TestCase):
    def test_below_key_lines_do_not_change_any_reading_or_the_verdict(self) -> None:
        plain, with_lines = score(), score(log_with_below_key(32, 64))
        self.assertEqual(with_lines["readings"], plain["readings"])
        self.assertEqual(with_lines["verdict"], "PASS")

    def test_below_key_lines_do_not_stand_in_for_merge_lines(self) -> None:
        lines = [line for line in log_with_below_key(32, 64) if "GLM53_DECODE_MERGE" not in line]
        merge = reading(score(lines), "decode_pad_merge")
        self.assertEqual(merge["verdict"], "NOT_MEASURED")
        self.assertIn("no GLM53_DECODE_MERGE line", merge["detail"])

    def test_a_narrow_width_does_not_refuse_or_fail_anything(self) -> None:
        result = score(log_with_below_key(1, 2, 4, 8, 16, 32, 64, 128, 256, 512))
        self.assertEqual([item["name"] for item in result["readings"] if item["verdict"] != "PASS"], [])


class Reporting(unittest.TestCase):
    def note(self, report: str) -> list[str]:
        return [line for line in report.splitlines() if "decode_below_key" in line]

    def test_one_line_says_served_padded_uncompared_and_not_evidence(self) -> None:
        lines = self.note(scorer.render_report(score(log_with_below_key(32, 64))))
        self.assertEqual(len(lines), 1)
        for phrase in ("2 width(s)", "32,64", "4 rank(s)", "served padded", "uncompared", "not evidence of pad equality"):
            self.assertIn(phrase, lines[0])

    def test_the_line_is_there_when_no_narrow_width_ran(self) -> None:
        lines = self.note(scorer.render_report(score()))
        self.assertEqual(len(lines), 1)
        self.assertIn("0 width(s)", lines[0])
        self.assertIn("not evidence of pad equality", lines[0])

    def test_the_verdict_stays_the_last_line(self) -> None:
        self.assertEqual(scorer.render_report(score(log_with_below_key(32))).splitlines()[-1], "C1_VERDICT: PASS")

    def test_a_result_written_before_the_field_existed_still_renders(self) -> None:
        old = score()
        del old["decode_below_key"]
        self.assertEqual(scorer.render_report(old).splitlines()[-1], "C1_VERDICT: PASS")
        self.assertEqual(self.note(scorer.render_report(old)), [])

    def test_the_shared_variant_reports_it_too(self) -> None:
        result = scorer.score(log_with_below_key(32), drive_rows(), scorer.BootFacts(900.0),
                              scorer.Expected.default(), variant=scorer.SHARED_VARIANT)
        self.assertEqual(result["decode_below_key"]["widths"], [32])


if __name__ == "__main__":
    unittest.main()
