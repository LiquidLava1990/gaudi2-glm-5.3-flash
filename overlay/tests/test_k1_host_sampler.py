"""The K1 host watcher's pure parts: new dmesg lines, host line. It reads only."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "cells"))

import k1_host_sampler as sampler  # noqa: E402


class Dmesg(unittest.TestCase):
    def test_only_new_matching_lines_are_returned(self) -> None:
        lines = ["[1] habanalabs 0000:b3:00.0: ctx_init failed", "[2] wls93: associated", "[3] mce: [Hardware Error]: x"]
        self.assertEqual(sampler.new_dmesg(lines, 0), (["[1] habanalabs 0000:b3:00.0: ctx_init failed", "[3] mce: [Hardware Error]: x"], 3))
        self.assertEqual(sampler.new_dmesg(lines, 3), ([], 3))
        self.assertEqual(sampler.new_dmesg([*lines, "[4] Out of memory: killed"], 3), (["[4] Out of memory: killed"], 4))

    def test_a_shorter_ring_starts_over(self) -> None:
        self.assertEqual(sampler.new_dmesg(["[9] habanalabs: x"], 5), (["[9] habanalabs: x"], 1))


class HostLine(unittest.TestCase):
    def test_free_and_load_are_named(self) -> None:
        free = "              total used free shared buff/cache available\nMem:   499 92 23 48 435 406\nSwap: 0 0 0\n"
        line = sampler.host_line(free, "12.5 3 2 1/2 3")
        self.assertIn("mem_free=23", line)
        self.assertIn("mem_available=406", line)
        self.assertIn("load=12.5", line)


if __name__ == "__main__":
    unittest.main()
