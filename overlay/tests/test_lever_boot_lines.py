"""One boot line per lever that had none in run3's all-levers boot log (/mnt/data1/claude-long/run3/all/boot.log, 2026-11-08).

HPU_GLM53_Q4_SHARED_GATHER_OWNER: its owners printed GLM53_Q4_SHARED_OWNER, which a lever audit grepping the flag's name never finds;
an armed line names the flag. HPU_GLM53_MULTI_BATCH_KPOOL_TILE_CEILING: read per layer with no line at all. Flags off (unset) print nothing.
"""
from __future__ import annotations

import contextlib
import logging
import os
import sys
import unittest
from collections.abc import Iterator
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent))

import key_stack  # noqa: E402
from overlay_testlib import OVERLAY_ROOT  # noqa: E402

SHARED_OWNER = "HPU_GLM53_Q4_SHARED_GATHER_OWNER"
TILE_CEILING = "HPU_GLM53_MULTI_BATCH_KPOOL_TILE_CEILING"
Q4_LOG = "vllm.glm53.q4_gather"
INDEXER_LOG = "vllm_gaudi.ops.hpu_glm5next_sparse_indexer"


@contextlib.contextmanager
def records(name: str) -> Iterator[list[str]]:
    """Every WARNING line logger `name` emits inside the block, none required."""
    lines: list[str] = []
    handler = logging.Handler(logging.WARNING)
    handler.emit = lambda record: lines.append(record.getMessage())
    logger = logging.getLogger(name)
    logger.addHandler(handler)
    try:
        yield lines
    finally:
        logger.removeHandler(handler)


def boot_lines(logger: str, env: dict[str, str], unset: tuple[str, ...] = ()) -> list[str]:
    """The lines `logger` prints while one sparse layer's modules import under `env`, with `unset` absent from the env."""
    with mock.patch.dict(key_stack.LAYER_ENV), mock.patch.dict(os.environ):
        for name in unset:
            key_stack.LAYER_ENV.pop(name, None)
            os.environ.pop(name, None)
        with records(logger) as lines, key_stack.layer_stack(OVERLAY_ROOT, env):
            pass
    return lines


class SharedGatherOwner(unittest.TestCase):
    def test_the_flag_on_prints_the_armed_line_naming_it(self) -> None:
        lines = boot_lines(Q4_LOG, {SHARED_OWNER: "1"})
        self.assertEqual([line for line in lines if SHARED_OWNER.removeprefix("HPU_") in line],
                         ["GLM53_Q4_SHARED_GATHER_OWNER armed scope=captured_graphs live_line=GLM53_Q4_SHARED_OWNER"])

    def test_off_or_unset_prints_nothing(self) -> None:
        self.assertEqual(boot_lines(Q4_LOG, {SHARED_OWNER: "0"}), [])
        self.assertEqual(boot_lines(Q4_LOG, {}, unset=(SHARED_OWNER,)), [])


class MultiBatchTileCeiling(unittest.TestCase):
    def test_a_set_ceiling_prints_its_pools_and_the_chunk_settings(self) -> None:
        lines = boot_lines(INDEXER_LOG, {TILE_CEILING: "16384"})
        self.assertIn("GLM53_MULTI_BATCH_KPOOL_TILE_CEILING armed pools=16384 context_chunk_pools=65536 budget_MiB=512", lines)

    def test_unset_prints_nothing(self) -> None:
        lines = boot_lines(INDEXER_LOG, {}, unset=(TILE_CEILING,))
        self.assertFalse([line for line in lines if "TILE_CEILING" in line], lines)


if __name__ == "__main__":
    unittest.main()
