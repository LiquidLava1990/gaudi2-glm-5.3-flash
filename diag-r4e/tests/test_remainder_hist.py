"""remainder_hist.py: bins, the 35% rule and the exit codes (CPU only, synthetic chunks)."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import remainder_hist  # noqa: E402


class CensusTest(unittest.TestCase):
    def test_bins_take_chunk_mod_1024_and_skip_whole_pieces(self) -> None:
        census = remainder_hist.census([1024, 2048, 100, 1124, 200, 300, 2348, 500])
        self.assertEqual(census.whole, 2)
        self.assertEqual(census.bins, {"1-128": 2, "129-256": 1, "257-384": 2, "385-1023": 1})

    def test_bin_edges(self) -> None:
        census = remainder_hist.census([128, 129, 256, 257, 384, 385, 1023])
        self.assertEqual(census.bins, {"1-128": 1, "129-256": 2, "257-384": 2, "385-1023": 2})

    def test_rule_picks_384_above_35_percent(self) -> None:
        census = remainder_hist.census([300] * 4 + [100] * 6)
        self.assertEqual(census.decision(), "384")

    def test_rule_keeps_256_at_or_below_35_percent(self) -> None:
        census = remainder_hist.census([300] * 35 + [100] * 65)
        self.assertEqual(census.decision(), "256")

    def test_no_remainders_is_unproven(self) -> None:
        self.assertEqual(remainder_hist.census([1024, 2048]).decision(), "UNPROVEN")

    def test_line_names_the_counts_share_and_decision(self) -> None:
        line = remainder_hist.census([300, 100]).line("medium")
        self.assertTrue(line.startswith("REMAINDER_HIST scope=medium remainders=2 whole=0 "), line)
        self.assertIn("257-384=1", line)
        self.assertIn("share_257_384=50.0%", line)
        self.assertIn("r4f_key=384", line)

    def test_main_exit_codes(self) -> None:
        missing = Path(__file__).with_name("no-such-container.log")
        self.assertEqual(remainder_hist.main([str(missing)]), 2)


if __name__ == "__main__":
    unittest.main()
