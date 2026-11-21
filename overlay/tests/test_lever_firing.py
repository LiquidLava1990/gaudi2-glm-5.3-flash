"""Step 27, the four kept levers test_capture_fences.py named but did not fire: mla_topkfuse, sinkhorn, kda_bitwise_rows and
kda_block_inverse. Each runs its real branch inside a key capture and reads the witness; none fires outside one. The last class records why
a lever that is legal only up to 512 rows still fires in a 1024-row key."""
from __future__ import annotations

import contextlib
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
from overlay_testlib import load_overlay  # noqa: E402

KDA_FILE = "vllm_gaudi/ops/hpu_glm5next_kda.py"
MHC_FILE = "vllm/model_executor/kernels/mhc/torch.py"
BITWISE = "HPU_GLM53_KDA_BITWISE_ROWS_PREFILL"
HEADS, DIM, CHUNK = 16, 128, 128


@contextlib.contextmanager
def key_capture() -> Iterator[None]:
    with graph.capture_context(), graph.key_capture_context():
        yield


class HpuTensor(torch.Tensor):
    """A CPU tensor that reports device hpu, so a branch that asks where the tensor lives takes its device path."""

    @property
    def device(self) -> SimpleNamespace:  # type: ignore[override]
        return SimpleNamespace(type="hpu")


def fired(run: object, lever: str) -> int:
    graph.reset_witness()
    with key_capture():
        run()
    return graph._WITNESS[lever]


def kda_inputs(rows: int) -> tuple[torch.Tensor, ...]:
    generator = torch.Generator().manual_seed(3)
    q, k, v = (torch.randn(1, rows, HEADS, DIM, generator=generator) for _ in range(3))
    g = -torch.rand(1, rows, HEADS, DIM, generator=generator) * 0.1
    beta = torch.rand(1, rows, HEADS, generator=generator)
    return q, k, v, g, beta


class MlaTopkFuse(unittest.TestCase):
    def select(self, rows: int) -> int:
        with mock.patch.dict(os.environ, {"HPU_GLM53_MLA_TOPKFUSE": "0"}):
            topkfuse = load_overlay("glm53_mla_topkfuse.py")
        select = mock.Mock(side_effect=lambda tile: tile)
        core = SimpleNamespace(mark_step=lambda: None)
        with mock.patch.dict(sys.modules, {"habana_frameworks": SimpleNamespace(), "habana_frameworks.torch": SimpleNamespace(),
                                           "habana_frameworks.torch.core": core}), \
                mock.patch.object(torch.ops, "mla_fuse", SimpleNamespace(select=select), create=True):
            return fired(lambda: topkfuse.select(torch.zeros(rows, 4)), "mla_topkfuse")

    def test_a_tall_scorer_output_fires_once_whatever_the_tile_count(self) -> None:
        self.assertEqual(self.select(256), 1)

    def test_a_single_tile_fires(self) -> None:
        self.assertEqual(self.select(128), 1)

    def test_outside_a_key_capture_it_does_not(self) -> None:
        graph.reset_witness()
        with mock.patch.dict(os.environ, {"HPU_GLM53_MLA_TOPKFUSE": "0"}):
            topkfuse = load_overlay("glm53_mla_topkfuse.py")
        with mock.patch.object(torch.ops, "mla_fuse", SimpleNamespace(select=lambda tile: tile), create=True):
            topkfuse.select(torch.zeros(128, 4))
        self.assertEqual(graph._WITNESS["mla_topkfuse"], 0)


class Sinkhorn(unittest.TestCase):
    TOKENS, HIDDEN = 3, 8

    def call(self, *, tpc: bool, device_hpu: bool = True, repeat: int = 20, eps: float = 1e-6) -> int:
        module = load_overlay(MHC_FILE)
        residual = torch.zeros(self.TOKENS, 4, self.HIDDEN, dtype=torch.bfloat16)
        residual = residual.as_subclass(HpuTensor) if device_hpu else residual
        args = (residual, torch.zeros(24, 4 * self.HIDDEN), torch.ones(3), torch.zeros(24), 1e-6, 1e-6, eps, 1.0, repeat)
        tpc_op = SimpleNamespace(mhc_sinkhorn=lambda flat: flat)
        with mock.patch.object(module, "_MHC_SINKHORN_TPC", tpc), mock.patch.object(torch.ops, "g6_tpc", tpc_op, create=True):
            return fired(lambda: module.mhc_pre_torch(*args), "sinkhorn")

    def test_the_tpc_kernel_branch_fires_on_hpu_with_the_production_settings(self) -> None:
        self.assertEqual(self.call(tpc=True), 1)

    def test_every_other_setting_runs_the_torch_loop_and_does_not_fire(self) -> None:
        for name, kwargs in (("flag off", {"tpc": False}), ("cpu", {"tpc": True, "device_hpu": False}),
                             ("repeat", {"tpc": True, "repeat": 10}), ("eps", {"tpc": True, "eps": 1e-5})):
            with self.subTest(name):
                self.assertEqual(self.call(**kwargs), 0)


class KdaBranches(unittest.TestCase):
    def run_kda(self, rows: int, *, bitwise: bool, block_inverse: bool, lever: str) -> int:
        module = load_overlay(KDA_FILE)
        with mock.patch.dict(os.environ, {BITWISE: "1" if bitwise else "0"}), \
                mock.patch.object(module, "_ENABLE_KDA_BLOCK_INVERSE_PREFILL", block_inverse):
            return fired(lambda: module._r589_original_chunk_kda(*kda_inputs(rows), chunk_size=CHUNK), lever)

    def test_the_bitwise_rows_branch_fires_at_each_legal_tile_size(self) -> None:
        for rows in (128, 256, 512):
            with self.subTest(rows=rows):
                self.assertEqual(self.run_kda(rows, bitwise=True, block_inverse=False, lever="kda_bitwise_rows"), 1)

    def test_the_bitwise_rows_branch_stays_off_with_its_flag_off(self) -> None:
        self.assertEqual(self.run_kda(128, bitwise=False, block_inverse=False, lever="kda_bitwise_rows"), 0)

    def test_the_block_inverse_branch_fires_with_its_flag_on_and_not_off(self) -> None:
        self.assertEqual(self.run_kda(128, bitwise=False, block_inverse=True, lever="kda_block_inverse"), 1)
        self.assertEqual(self.run_kda(128, bitwise=False, block_inverse=False, lever="kda_block_inverse"), 0)

    def test_both_branches_give_the_row_order_result_the_kernel_would(self) -> None:
        """The levers are bitwise-exact against the plain loop on a 128-row tile; a lever that fires must not move the result."""
        module = load_overlay(KDA_FILE)
        outputs = []
        for bitwise, block_inverse in ((False, False), (True, False), (False, True)):
            with mock.patch.dict(os.environ, {BITWISE: "1" if bitwise else "0"}), \
                    mock.patch.object(module, "_ENABLE_KDA_BLOCK_INVERSE_PREFILL", block_inverse):
                outputs.append(module._r589_original_chunk_kda(*kda_inputs(128), chunk_size=CHUNK)[0])
        for other in outputs[1:]:
            torch.testing.assert_close(other, outputs[0], rtol=1e-3, atol=1e-4)


class Above512Rows(unittest.TestCase):
    """Recorded reason (plan 12.3a, step 27): kda_bitwise_rows is legal only for a whole call of 128, 256 or 512 rows
    (`total_len in (128, 256, 512)`), because its row blocks hold the decay for `end` columns of one chunk set. A 1024-row key therefore
    does not fire it on one call; R589's bounded prefill hands the kernel 512-row tiles, and it fires once per tile."""

    def test_a_1024_row_call_does_not_take_the_branch(self) -> None:
        module = load_overlay(KDA_FILE)
        with mock.patch.dict(os.environ, {BITWISE: "1"}), mock.patch.object(module, "_ENABLE_KDA_BLOCK_INVERSE_PREFILL", False):
            self.assertEqual(fired(lambda: module._r589_original_chunk_kda(*kda_inputs(1024), chunk_size=CHUNK), "kda_bitwise_rows"), 0)

    def test_two_512_row_tiles_of_a_1024_row_key_fire_it_twice(self) -> None:
        module = load_overlay(KDA_FILE)

        def key_rows() -> None:
            for _ in range(1024 // 512):
                module._r589_original_chunk_kda(*kda_inputs(512), chunk_size=CHUNK)

        with mock.patch.dict(os.environ, {BITWISE: "1"}), mock.patch.object(module, "_ENABLE_KDA_BLOCK_INVERSE_PREFILL", False):
            self.assertEqual(fired(key_rows, "kda_bitwise_rows"), 2)


if __name__ == "__main__":
    unittest.main()
