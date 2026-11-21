"""The one-card cell's runner logic, on CPU. The cell itself waits on a card; a CPU rehearsal can never pass it."""

from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "cells"))

import contig_byte_cell as cell  # noqa: E402

MAPS = """\
7f0000000000-7f0000200000 rw-s 00000000 00:05 12  /dev/zero (deleted)
7f0000200000-7f0000400000 rw-p 00000000 00:00 0
7f0000400000-7f0000401000 r--p 00000000 08:01 44  /usr/lib/x.so
"""


class Maps(unittest.TestCase):
    def test_parse_splits_by_backing(self) -> None:
        self.assertEqual(cell.parse_maps(MAPS), {"/dev/zero": 2 << 20, "anon": 2 << 20, "other": 4096})

    def test_growth_is_per_backing(self) -> None:
        before = {"/dev/zero": 1, "anon": 2, "other": 3}
        self.assertEqual(cell.maps_growth(before, {"/dev/zero": 5, "anon": 2, "other": 4}),
                         {"/dev/zero": 4, "anon": 0, "other": 1})


class Verdict(unittest.TestCase):
    STATE = 1000 << 20

    def growth(self, mapped: float) -> dict[str, int]:
        return {"/dev/zero": int(mapped * self.STATE), "anon": 0, "other": 123}

    def test_contig_passes_near_one(self) -> None:
        self.assertTrue(cell.verdict("contig", "hpu", True, self.growth(1.03), self.STATE)["pass"])

    def test_contig_fails_at_two(self) -> None:
        self.assertFalse(cell.verdict("contig", "hpu", True, self.growth(2.0), self.STATE)["pass"])

    def test_contig_fails_when_nothing_was_mapped(self) -> None:
        self.assertFalse(cell.verdict("contig", "hpu", True, self.growth(0.0), self.STATE)["pass"])

    def test_bytes_differ_fails(self) -> None:
        self.assertFalse(cell.verdict("contig", "hpu", False, self.growth(1.03), self.STATE)["pass"])

    def test_native_control_reports_its_ratio(self) -> None:
        result = cell.verdict("native", "hpu", True, self.growth(2.0), self.STATE)
        self.assertEqual((result["pass"], result["mapped_ratio"]), (True, 2.0))

    def test_cpu_never_passes(self) -> None:
        result = cell.verdict("contig", "cpu", True, self.growth(1.03), self.STATE)
        self.assertEqual((result["pass"], result["card_verdict"]), (False, False))


class Rehearsal(unittest.TestCase):
    def rehearse(self, mode: str) -> dict:
        with tempfile.TemporaryDirectory() as folder:
            args = cell.parse_args(["--mode", mode, "--device", "cpu", "--slots", "2", "--state-bytes", "65536",
                                    "--tensors", "5", "--out", f"{folder}/out.json"])
            with mock.patch.dict(os.environ):  # the runner sets the flag for its own process
                return cell.run(args)

    def test_contig_runner_round_trips_bytes_on_cpu(self) -> None:
        result = self.rehearse("contig")
        self.assertEqual((result["byte_equal"], result["card_verdict"], result["pass"]), (True, False, False))

    def test_native_runner_round_trips_bytes_on_cpu(self) -> None:
        self.assertTrue(self.rehearse("native")["byte_equal"])


if __name__ == "__main__":
    unittest.main()
