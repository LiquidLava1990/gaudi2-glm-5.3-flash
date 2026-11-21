"""HPU_GLM53_SMALL_KEY_PARTIAL_MERGE: a key of 128 rows or fewer takes the legacy indexer route, unshared, over 4096-pool chunks.

Card evidence (k1-out/fable-l3 run6, l3_sort_card.py): a stable descending argsort returns index -1 past the finite scores at widths such as 4608, so
_ordered_streaming_merge's 512 + 4096 columns give a row with 1 to 511 finite scores a repeated pool. The flag sends these chunks through
_partial_chunk_merge, whose merges are 512 and 1024 wide, the widths every serving key already sorts. The CPU sorts right at any width, so the
widths are what these tests read; the wrong sort itself is faked to show the route never reaches it.
"""
from __future__ import annotations

import contextlib
import logging
import os
import sys
import unittest
from collections.abc import Iterator
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

import glm53_mixed_graph as graph  # noqa: E402
import key_stack  # noqa: E402
from overlay_testlib import OVERLAY_ROOT  # noqa: E402
from test_key_select_exactness import key_capture  # noqa: E402

FLAG = "HPU_GLM53_SMALL_KEY_PARTIAL_MERGE"
KEYS = "VLLM_GAUDI_MIXED_PROMPT_KEYS"
COLUMNS = 1024
SAFE_SORT_WIDTH = 1024
COUNT = 512
POOLS_PER_CHUNK = 4096
FINITE_PER_ROW = (0, 1, 2, 3, 100, 511, 512, 513, 1000)


class KeySelect:
    """Mixin: the real indexer in a key capture (no tests of its own, so a subclass reruns none)."""

    def select(self, env: dict[str, str], rows: int = 128, real: int = 11, start: int = 0) -> tuple[torch.Tensor, set[int]]:
        """The real indexer in a capture of a key of `rows` rows: physical rows, and every width canonical_topk sorted."""
        with key_stack.layer_stack(OVERLAY_ROOT, {KEYS: f"{rows}:{COLUMNS}", **env}) as world:
            world.kpool._score_pool_chunk = key_stack.fast_scores
            widths: set[int] = set()
            canonical = world.kpool._r590_topk

            def recorded(scores: torch.Tensor, ids: torch.Tensor, count: int) -> tuple[torch.Tensor, torch.Tensor]:
                widths.add(scores.shape[1])
                return canonical(scores, ids, count)

            world.kpool._r590_topk = recorded
            case = key_stack.prompt_case(rows, COLUMNS, start)
            mask = (torch.arange(rows) < real).float()
            case["positions"] = torch.where(torch.arange(rows) < real, torch.arange(start, start + rows), torch.zeros(rows, dtype=torch.long)).to(torch.int32)
            with key_capture():
                physical = key_stack.indexer_step(world, case, token_mask=mask)
            return physical, widths


class Route(KeySelect, unittest.TestCase):
    def test_flag_unset_keeps_the_legacy_merge_and_its_wide_sort(self) -> None:
        _, widths = self.select({})
        self.assertIn(COUNT + POOLS_PER_CHUNK, widths)

    def test_flag_zero_keeps_the_legacy_merge(self) -> None:
        _, widths = self.select({FLAG: "0"})
        self.assertIn(COUNT + POOLS_PER_CHUNK, widths)

    def test_flag_on_never_sorts_wider_than_the_serving_keys_do(self) -> None:
        _, widths = self.select({FLAG: "1"})
        self.assertTrue(widths)
        self.assertLessEqual(max(widths), SAFE_SORT_WIDTH)

    def test_flag_on_selects_what_the_legacy_merge_selects(self) -> None:
        for real, start in ((11, 0), (128, 0), (128, 2048), (128, 4096)):
            with self.subTest(real=real, start=start):
                legacy, _ = self.select({}, real=real, start=start)
                merged, _ = self.select({FLAG: "1"}, real=real, start=start)
                self.assertTrue(torch.equal(legacy, merged))

    def test_flag_on_leaves_a_key_above_128_rows_alone(self) -> None:
        _, off = self.select({}, rows=256, real=256)
        _, on = self.select({FLAG: "1"}, rows=256, real=256)
        self.assertEqual(off, on)


class Merge(unittest.TestCase):
    """_partial_chunk_merge at the shapes the flag gives it, rows of 0 to 1000 finite scores against the host's exact answer."""

    @staticmethod
    def chunks(rows: int, width: int, count: int) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        generator = torch.Generator().manual_seed(rows + width)
        scores, ids = [], []
        for index in range(count):
            chunk = torch.full((rows, width), -torch.inf)
            for row in range(rows):
                live = max(0, min(width, FINITE_PER_ROW[row % len(FINITE_PER_ROW)] - index * width))
                chunk[row, :live] = torch.randint(0, 50, (live,), generator=generator).float()
            scores.append(chunk)
            ids.append(torch.arange(index * width, (index + 1) * width).expand(rows, -1))
        return scores, ids

    def test_many_chunks_give_the_exact_top_512(self) -> None:
        with key_stack.layer_stack(OVERLAY_ROOT, {FLAG: "1"}) as world:
            kpool = world.kpool
            for width, count in ((4096, 8), (512, 3), (1024, 2)):
                with self.subTest(width=width, count=count):
                    scores, ids = self.chunks(128, width, count)
                    running = (torch.full((128, COUNT), -torch.inf), torch.full((128, COUNT), -1, dtype=torch.long))
                    for index, (chunk, chunk_ids) in enumerate(zip(scores, ids)):
                        running = kpool._partial_chunk_merge(*running, chunk, chunk_ids, COUNT, first=index == 0)
                    self.assert_exact(running, torch.cat(scores, dim=1), torch.cat(ids, dim=1))

    def assert_exact(self, running: tuple[torch.Tensor, torch.Tensor], scores: torch.Tensor, ids: torch.Tensor) -> None:
        for row in range(scores.shape[0]):
            keep = torch.isfinite(scores[row])
            pairs = sorted(zip((-scores[row][keep]).tolist(), ids[row][keep].tolist()))[:COUNT]
            live = running[1][row][running[1][row] >= 0].tolist()
            self.assertEqual(live, [identity for _, identity in pairs], f"row {row}")
            self.assertEqual(len(set(live)), len(live), f"row {row} repeats a pool")


ROUTE_ENV = {FLAG: "1", "HPU_GLM53_PARTIAL_SCAN_TOPK": "1", "HPU_GLM53_R590_ENABLE": "1", "HPU_GLM53_TP_SHARDED_KPOOL": "0"}
# kpool opens _partial_chunk_merge only with the partial scan on, R590 on and an unsharded kpool (tp_size 1); each one missing sends a
# 128-row key back to the legacy merge's wrong-width sort, so each refuses with its own flag.
MISSING = (("HPU_GLM53_PARTIAL_SCAN_TOPK", "0"), ("HPU_GLM53_R590_ENABLE", "0"), ("HPU_GLM53_TP_SHARDED_KPOOL", "1"),
           ("HPU_GLM53_TP_SHARDED_KPOOL", "true"), (FLAG, "0"))


def without(name: str) -> dict:
    return {key: value for key, value in {**os.environ, **ROUTE_ENV}.items() if key != name}


class BootRefusal(unittest.TestCase):
    def test_a_small_key_without_the_merge_is_refused(self) -> None:
        for key in ((128, 1024), (128, 8192)):
            with self.subTest(key), mock.patch.dict(os.environ, {**ROUTE_ENV, FLAG: "0"}), self.assertRaisesRegex(ValueError, FLAG):
                graph.check_small_key_merge(((1024, 1024), key))

    def test_each_missing_route_condition_is_refused_by_name(self) -> None:
        for name, value in MISSING:
            with self.subTest(name=name, value=value), mock.patch.dict(os.environ, {**ROUTE_ENV, name: value}), \
                    self.assertRaisesRegex(ValueError, f"{name}"):
                graph.check_small_key_merge(((128, 1024),))

    def test_an_unset_route_flag_is_refused_the_same(self) -> None:
        for name in (FLAG, "HPU_GLM53_PARTIAL_SCAN_TOPK", "HPU_GLM53_R590_ENABLE"):
            with self.subTest(name=name), mock.patch.dict(os.environ, without(name), clear=True), self.assertRaisesRegex(ValueError, name):
                graph.check_small_key_merge(((128, 1024),))

    def test_every_missing_condition_is_named_at_once(self) -> None:
        with mock.patch.dict(os.environ, {FLAG: "0", "HPU_GLM53_PARTIAL_SCAN_TOPK": "0", "HPU_GLM53_R590_ENABLE": "0",
                                          "HPU_GLM53_TP_SHARDED_KPOOL": "1"}), self.assertRaises(ValueError) as refused:
            graph.check_small_key_merge(((128, 1024),))
        for name in ROUTE_ENV:
            self.assertIn(name, str(refused.exception))

    def test_an_unset_sharded_kpool_is_unsharded(self) -> None:
        with mock.patch.dict(os.environ, without("HPU_GLM53_TP_SHARDED_KPOOL"), clear=True):
            graph.check_small_key_merge(((128, 1024),))

    def test_a_small_key_with_the_route_open_boots(self) -> None:
        with mock.patch.dict(os.environ, ROUTE_ENV):
            graph.check_small_key_merge(((1024, 1024), (128, 1024), (128, 8192)))

    def test_keys_above_128_rows_never_need_it(self) -> None:
        with mock.patch.dict(os.environ, {FLAG: "0", "HPU_GLM53_PARTIAL_SCAN_TOPK": "0", "HPU_GLM53_R590_ENABLE": "0",
                                          "HPU_GLM53_TP_SHARDED_KPOOL": "1"}):
            graph.check_small_key_merge(((1024, 8192), (1024, 1024), (256, 1024), (512, 1024)))


MERGE_LOG = "glm53.partial_merge"  # kpool's armed and live lines


@contextlib.contextmanager
def records(name: str) -> Iterator[list[str]]:
    """Every WARNING line logger `name` emits inside the block, none required (assertLogs fails on none)."""
    lines: list[str] = []
    handler = logging.Handler(logging.WARNING)
    handler.emit = lambda record: lines.append(record.getMessage())
    logger = logging.getLogger(name)
    logger.addHandler(handler)
    try:
        yield lines
    finally:
        logger.removeHandler(handler)


class BootLine(unittest.TestCase):
    """run2 and run3 set the flag and their boot logs had no line naming it: activation was unproven. The boot check now says which keys it covers."""

    def test_an_open_route_logs_the_small_keys_it_merges(self) -> None:
        with mock.patch.dict(os.environ, ROUTE_ENV), self.assertLogs(graph.LOG, logging.WARNING) as logs:
            graph.check_small_key_merge(((1024, 1024), (128, 8192), (128, 1024)))
        self.assertIn("GLM53_SMALL_KEY_PARTIAL_MERGE boot keys=128:8192,128:1024 route=partial_chunk_merge", "\n".join(logs.output))

    def test_the_flag_without_a_small_key_says_it_covers_none(self) -> None:
        with mock.patch.dict(os.environ, ROUTE_ENV), self.assertLogs(graph.LOG, logging.WARNING) as logs:
            graph.check_small_key_merge(((1024, 1024),))
        self.assertIn("GLM53_SMALL_KEY_PARTIAL_MERGE boot keys=none route=none", "\n".join(logs.output))

    def test_flag_off_logs_nothing(self) -> None:
        with mock.patch.dict(os.environ, {**ROUTE_ENV, FLAG: "0"}), self.assertNoLogs(graph.LOG, logging.WARNING):
            graph.check_small_key_merge(((1024, 1024), (256, 1024)))


class KpoolLines(KeySelect, unittest.TestCase):
    """kpool's own lines: armed at import (with what shuts the route), live at the capture that takes the merge."""

    def armed(self, env: dict[str, str]) -> list[str]:
        with records(MERGE_LOG) as lines, key_stack.layer_stack(OVERLAY_ROOT, env):
            pass
        return lines

    def test_both_flags_on_arm_both_lines_reachable(self) -> None:
        lines = self.armed({FLAG: "1"})
        self.assertIn("GLM53_PARTIAL_SCAN_TOPK armed reachable=1 blockers=none", lines)
        self.assertIn("GLM53_SMALL_KEY_PARTIAL_MERGE armed reachable=1 blockers=none", lines)

    def test_each_shut_condition_is_named_on_the_armed_line(self) -> None:
        for name, value, fix in (("HPU_GLM53_R590_ENABLE", "0", "HPU_GLM53_R590_ENABLE=1"), ("HPU_GLM53_PARTIAL_SCAN_TOPK", "0", "HPU_GLM53_PARTIAL_SCAN_TOPK=1"),
                                 ("HPU_GLM53_TP_SHARDED_KPOOL", "1", "HPU_GLM53_TP_SHARDED_KPOOL=0")):
            with self.subTest(name):
                lines = [line for line in self.armed({FLAG: "1", name: value}) if line.startswith("GLM53_SMALL_KEY_PARTIAL_MERGE armed")]
                self.assertEqual(len(lines), 1)
                self.assertIn("reachable=0", lines[0])
                self.assertIn(fix, lines[0])

    def test_flags_off_arm_nothing(self) -> None:
        self.assertEqual(self.armed({FLAG: "0", "HPU_GLM53_PARTIAL_SCAN_TOPK": "0"}), [])

    def test_a_small_key_capture_logs_the_live_merge_once(self) -> None:
        with records(MERGE_LOG) as lines:
            self.select({FLAG: "1"})
        live = [line for line in lines if line.startswith("GLM53_SMALL_KEY_PARTIAL_MERGE_LIVE")]
        self.assertEqual(len(live), 1, lines)
        self.assertRegex(live[0], r"^GLM53_SMALL_KEY_PARTIAL_MERGE_LIVE route=small_key rows=128 chunk=\d+ scan=\d+ capture=1$")

    def test_flag_off_capture_logs_no_small_key_line(self) -> None:
        with records(MERGE_LOG) as lines:
            self.select({})
        self.assertFalse([line for line in lines if "SMALL_KEY" in line], lines)


class NarrowChunk(KeySelect, unittest.TestCase):
    """The merge opens only for chunks of top_pools (512) or more; a key capture whose chunks are narrower would sort the legacy way."""

    NARROW = {"HPU_GLM53_CONTEXT_CHUNK_POOLS": "256"}

    def test_a_chunk_narrower_than_top_pools_refuses_the_capture_with_the_fix(self) -> None:
        with self.assertRaisesRegex(ValueError, r"HPU_GLM53_SMALL_KEY_PARTIAL_MERGE.*256-pool chunk.*512.*HPU_GLM53_CONTEXT_CHUNK_POOLS"):
            self.select({FLAG: "1", **self.NARROW})

    def test_flag_off_keeps_the_narrow_chunk_running(self) -> None:
        _, widths = self.select(dict(self.NARROW))
        self.assertTrue(widths)

    def test_the_serving_chunk_still_captures(self) -> None:
        self.select({FLAG: "1"})


class NarrowTail(unittest.TestCase):
    """The guard's other branches: a chunk of top_pools or more whose last chunk (the scan's remainder) is narrower, and one chunk that is
    the whole scan (the single route's ground, not a multi-chunk scan: the guard leaves it alone)."""

    def guard(self, *, chunk: int, scan_width: int, top_pools: int = COUNT) -> None:
        with key_stack.layer_stack(OVERLAY_ROOT, {FLAG: "1"}) as world, key_capture():
            world.kpool._refuse_narrow_small_key_chunks(chunk=chunk, top_pools=top_pools, tp_size=1, shared_block_table=False,
                                                        scan_width=scan_width)

    def test_a_remainder_narrower_than_top_pools_refuses_with_its_width(self) -> None:
        for chunk in (512, 1024, POOLS_PER_CHUNK):
            with self.subTest(chunk=chunk), self.assertRaisesRegex(ValueError, rf"scans {3 * chunk + 100} pools in {chunk}-pool chunks, "
                                                                              r"and a 100-pool chunk is narrower than the 512 pools"):
                self.guard(chunk=chunk, scan_width=3 * chunk + 100)

    def test_a_remainder_of_top_pools_or_more_captures(self) -> None:
        for tail in (0, COUNT, COUNT + 100):
            with self.subTest(tail=tail):
                self.guard(chunk=1024, scan_width=2048 + tail)

    def test_one_chunk_narrower_than_top_pools_is_the_whole_scan_and_captures(self) -> None:
        for scan_width in (100, COUNT - 1, COUNT):
            with self.subTest(scan_width=scan_width):
                self.guard(chunk=COUNT, scan_width=scan_width)
                self.guard(chunk=256, scan_width=min(scan_width, 256))


class KeyChoice(unittest.TestCase):
    """Which chunk replays which key once the 128-row keys are in the env."""

    ENV = {KEYS: "1024:8192,1024:1024,128:8192,128:1024"}

    def choice(self, rows: int, width: int) -> tuple[int, int]:
        with mock.patch.dict(os.environ, self.ENV):
            return graph.select_prompt_key(rows, width)[0]

    def test_a_chunk_of_128_rows_or_fewer_on_a_narrow_table_takes_128_by_1024(self) -> None:
        for rows in (1, 11, 52, 128):
            self.assertEqual(self.choice(rows, 1024), (128, 1024))

    def test_a_table_wider_than_1024_blocks_takes_128_by_8192(self) -> None:
        for width in (1025, 2048, 4096, 8192):
            self.assertEqual(self.choice(11, width), (128, 8192))

    def test_a_taller_chunk_keeps_the_1024_row_keys(self) -> None:
        self.assertEqual(self.choice(129, 1024), (1024, 1024))
        self.assertEqual(self.choice(129, 8192), (1024, 8192))

    def test_nothing_changes_without_the_small_keys(self) -> None:
        with mock.patch.dict(os.environ, {KEYS: "1024:8192,1024:1024"}):
            self.assertEqual(graph.select_prompt_key(11, 1024)[0], (1024, 1024))


if __name__ == "__main__":
    unittest.main()
