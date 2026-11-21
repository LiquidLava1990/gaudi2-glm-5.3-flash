"""lab.py up refuses to start a boot the driver would fail: gaudi2_ctx_init needs order-9 (2 MiB) host pages, and without them every worker dies with
synStatus=8 Device acquire failed / Failed to create context -12 (c1p-20261107T073249Z, b2-20261107T082750Z). The check compacts, reads /proc/buddyinfo,
drops the page cache only when still short, and refuses with the numbers. Host only: no card, no root; the steps are recorded."""
from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

LAB = Path(__file__).resolve().parents[2] / "lab.py"
if not LAB.is_file():
    raise unittest.SkipTest("lab.py sits beside the overlay on the host; not in an image")
spec = importlib.util.spec_from_file_location("lab_under_test", LAB)
lab = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = lab
spec.loader.exec_module(lab)

SHORT = """Node 0, zone      DMA      0      0      0      0      0      0      0      0      1      1      2
Node 0, zone    DMA32   1676   1818    895    751    575    320    197     82     44     13    191
Node 0, zone   Normal  25240 122428 139709  96239  38465  16375   4244   1491    810    728    140
Node 1, zone   Normal    484  35312  47978 104729  57237  29167  14101   6205      8      0      0
"""
ENOUGH = SHORT.replace("6205      8      0      0", "6205      8     20      3")


class Blocks(unittest.TestCase):
    def test_blocks_count_order_nine_units_in_normal_zones_per_node(self) -> None:
        self.assertEqual(lab.contiguous_blocks(SHORT), {0: 728 + 2 * 140, 1: 0})
        self.assertEqual(lab.contiguous_blocks(ENOUGH), {0: 728 + 2 * 140, 1: 20 + 2 * 3})

    def test_a_dma_zone_does_not_count(self) -> None:
        only_dma = "Node 0, zone      DMA      0      0      0      0      0      0      0      0      1      1      2\n"
        self.assertEqual(lab.contiguous_blocks(only_dma), {})


class Prepare(unittest.TestCase):
    def run_prepare(self, readings: list[str]) -> list[str]:
        steps: list[str] = []
        remaining = list(readings)
        lab.prepare_host_memory(lambda command: steps.append(command), lambda: remaining.pop(0))
        return steps

    def test_enough_after_compaction_keeps_the_page_cache(self) -> None:
        steps = self.run_prepare([ENOUGH])
        self.assertEqual(steps, [lab.COMPACT_COMMAND])
        self.assertNotIn("drop_caches", " ".join(steps))

    def test_short_after_compaction_drops_the_cache_and_compacts_again(self) -> None:
        steps = self.run_prepare([SHORT, ENOUGH])
        self.assertEqual(steps, [lab.COMPACT_COMMAND, lab.DROP_COMMAND])
        self.assertIn("drop_caches", steps[1])
        self.assertIn("compact_memory", steps[1])

    def test_still_short_refuses_with_the_node_the_numbers_and_what_to_do(self) -> None:
        with self.assertRaises(SystemExit) as refused:
            self.run_prepare([SHORT, SHORT])
        message = str(refused.exception)
        for needle in ("node 1", "0 of", "order-9", "Failed to create context -12", "stop"):
            self.assertIn(needle, message)


if __name__ == "__main__":
    unittest.main()
