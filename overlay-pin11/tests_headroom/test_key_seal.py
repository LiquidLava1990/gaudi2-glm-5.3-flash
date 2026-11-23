"""The boot seal of the dual build (plan 12.3a and the review findings R3 to R5): the prompt keys are judged by the allocator's measured in-use
delta from before the first capture to after the last, never by summed per-key estimates; every decode key must add exactly one graph; the
lattice refuses a batch wider than its top key; the seal holds a ballast the size of the first replay's workspace. CPU only: the readings are
patched, and what the real allocator returns is a card question."""
from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent))

import test_decode_lattice as lattice  # noqa: E402
from test_decode_lattice import LATTICE_ENV, MIB, FakeRunner, GraphCase  # noqa: E402
from test_prompt_keys import Backend, KeyCase, RowForward  # noqa: E402

FOUR_KEYS = "1024:8192,1024:4096,1024:2048,1024:1024"
BUDGET_MIB = 7940


class SealCase(GraphCase):
    env = {**LATTICE_ENV, "VLLM_GAUDI_MIXED_PROMPT_KEYS": FOUR_KEYS, "VLLM_GAUDI_MIXED_GRAPH_BUDGET_MIB": str(BUDGET_MIB)}

    def runner(self) -> SimpleNamespace:
        cache = self.graph.GraphCache(RowForward(), 128, Backend())
        return SimpleNamespace(model=SimpleNamespace(_glm53_mixed_graph_cache=cache))

    def seal(self, runner: SimpleNamespace, before_mib: int, after_mib: int) -> list[str]:
        with mock.patch.object(self.graph, "_in_use_bytes", return_value=after_mib * MIB), self.assertLogs(self.graph.LOG, "WARNING") as logs:
            self.graph.seal_prompt_keys(runner, before_mib * MIB)
        return logs.output


class PromptKeySeal(SealCase):
    def test_four_keys_that_hold_2604_mib_pass_though_their_estimates_sum_past_the_budget(self) -> None:
        """k1-out/final-armB: two keys left 4,301 of 8,443 MiB free with a 1,538 MiB ballast, so they hold at most 2,604 MiB."""
        runner = self.runner()
        output = self.seal(runner, 1500, 1500 + 2604)
        self.assertEqual(runner.model._glm53_mixed_graph_cache.sealed_bytes, 2604 * MIB)
        line = next(line for line in output if "GLM53_KEY_SEAL" in line)
        self.assertIn("in_use_delta_MiB=2604.0", line)
        self.assertIn(f"budget_MiB={BUDGET_MIB}", line)
        self.assertGreater(4 * self.graph.key_estimate_bytes(1024), BUDGET_MIB * MIB, "the estimates would have refused these four keys")

    def test_a_real_shortfall_refuses_with_the_numbers_and_the_fix(self) -> None:
        with self.assertRaises(RuntimeError) as refused:
            self.seal(self.runner(), 1000, 1000 + 9000)
        message = str(refused.exception)
        for needle in ("9000.0", str(BUDGET_MIB), FOUR_KEYS, "VLLM_GAUDI_MIXED_GRAPH_BUDGET_MIB", "1024:8192"):
            self.assertIn(needle, message)
        self.assertNotIn("num-gpu-blocks-override", message)

    def test_a_delta_exactly_at_the_budget_passes_and_one_mib_over_refuses(self) -> None:
        self.seal(self.runner(), 0, BUDGET_MIB)
        with self.assertRaisesRegex(RuntimeError, "past"):
            self.seal(self.runner(), 0, BUDGET_MIB + 1)

    def test_one_rank_over_refuses_every_rank(self) -> None:
        peers = [dict(bucket=("prompt_seal",), free=0, required=0, error=""),
                 dict(bucket=("prompt_seal",), free=0, required=0, error="rank 1 holds 9000 MiB")]
        with mock.patch.object(self.graph, "rank_reports", return_value=peers), self.assertRaisesRegex(RuntimeError, "rank 1 holds 9000"):
            self.seal(self.runner(), 0, 100)

    def test_a_negative_delta_is_zero(self) -> None:
        runner = self.runner()
        self.seal(runner, 500, 400)
        self.assertEqual(runner.model._glm53_mixed_graph_cache.sealed_bytes, 0)


class WarmupSealsTheKeys(SealCase):
    def test_in_use_is_read_before_the_first_capture_and_after_the_last(self) -> None:
        order: list[str] = []
        readings = iter((100 * MIB, 2704 * MIB))
        runner = self.runner()
        runner._prepare_dummy_scenario = lambda prompt_cfg, decode_cfg: order.append("capture")
        buckets = tuple((0, 1024, width) for width in (8192, 4096, 2048, 1024))
        tiles = SimpleNamespace(warm_adaptive_tiles_at_boot=lambda model: order.append("tiles"))

        def in_use() -> int:
            order.append("in_use")
            return next(readings)

        with mock.patch.dict(sys.modules, {"glm53_r590_long_history": tiles}),                 mock.patch.object(self.graph, "validate_warmup", return_value=buckets),                 mock.patch.object(self.graph, "warmup_scenarios", return_value=[(None, None)] * 4),                 mock.patch.object(self.graph, "check_keys_captured"), mock.patch.object(self.graph, "_in_use_bytes", side_effect=in_use):
            self.graph.warmup(runner)
        self.assertEqual(order, ["in_use"] + ["capture"] * 4 + ["in_use", "tiles"])
        self.assertEqual(runner.model._glm53_mixed_graph_cache.sealed_bytes, 2604 * MIB)

    def test_no_keys_reads_no_allocator(self) -> None:
        self.set_env(VLLM_GAUDI_MIXED_PROMPT_KEYS="", VLLM_GAUDI_MIXED_DECODE_WIDTHS="", HPU_GLM53_DECODE_WIDTH_PAD="0",
                     HPU_GLM53_DECODE_PAD_CHECK_WIDTHS="")
        runner = self.runner()
        tiles = SimpleNamespace(warm_adaptive_tiles_at_boot=lambda model: 0)
        with mock.patch.dict(sys.modules, {"glm53_r590_long_history": tiles}),                 mock.patch.object(self.graph, "validate_warmup", return_value=()),                 mock.patch.object(self.graph, "warmup_scenarios", return_value=[]),                 mock.patch.object(self.graph, "check_keys_captured"), mock.patch.object(self.graph, "_in_use_bytes") as in_use:
            self.graph.warmup(runner)
        in_use.assert_not_called()


class WitnessSeesEveryBootForward(lattice.LatticeBoot):
    """R8: the witness read GraphCache counters only, so warm_adaptive_tiles_at_boot and the guard's routed eager forwards did not show."""

    def test_the_witness_line_lists_both(self) -> None:
        output = self.boot(FakeRunner(self.graph))
        witness = next(line for line in output if "GLM53_BOOT_WITNESS" in line)
        self.assertIn("routed_eager=0", witness)
        self.assertIn("adaptive_tile_warm=0", witness)

    def test_a_routed_eager_forward_in_the_boot_window_refuses(self) -> None:
        readings = iter((5, 7))
        with mock.patch.object(self.graph, "_routed_eager_forwards", side_effect=lambda: next(readings)):
            self.graph.mark_boot_window()
            with self.assertRaisesRegex(RuntimeError, "routed_eager"):
                self.boot(FakeRunner(self.graph))

    def test_an_eager_forward_before_the_window_is_not_this_boots(self) -> None:
        with mock.patch.object(self.graph, "_routed_eager_forwards", return_value=9):
            self.graph.mark_boot_window()
            self.boot(FakeRunner(self.graph))

    def test_a_tile_warm_that_compiled_shapes_refuses_and_names_the_flag(self) -> None:
        self.graph.count_off_graph("adaptive_tile_warm", 3)
        with self.assertRaises(RuntimeError) as refused:
            self.boot(FakeRunner(self.graph))
        for needle in ("adaptive_tile_warm", "HPU_GLM53_WARM_ADAPTIVE_TILES"):
            self.assertIn(needle, str(refused.exception))

    def test_the_window_starts_clean(self) -> None:
        self.graph.count_off_graph("adaptive_tile_warm", 3)
        self.graph.mark_boot_window()
        self.boot(FakeRunner(self.graph))


class WarmupCountsTheTileWarm(SealCase):
    def warm(self, compiled: int) -> None:
        runner = self.runner()
        runner._prepare_dummy_scenario = lambda prompt_cfg, decode_cfg: None
        tiles = SimpleNamespace(warm_adaptive_tiles_at_boot=lambda model: compiled)
        with mock.patch.dict(sys.modules, {"glm53_r590_long_history": tiles}),                 mock.patch.object(self.graph, "validate_warmup", return_value=()), mock.patch.object(self.graph, "warmup_scenarios", return_value=[]),                 mock.patch.object(self.graph, "check_keys_captured"), mock.patch.object(self.graph, "_in_use_bytes", return_value=0),                 mock.patch.object(self.graph, "_routed_eager_forwards", return_value=0), self.assertLogs(self.graph.LOG, "WARNING"):
            self.graph.warmup(runner)

    def test_compiled_shapes_are_counted(self) -> None:
        self.warm(4)
        self.assertEqual(self.graph.OFF_GRAPH["adaptive_tile_warm"], 4)

    def test_a_warm_that_is_off_counts_nothing(self) -> None:
        self.warm(0)
        self.assertEqual(self.graph.OFF_GRAPH["adaptive_tile_warm"], 0)


class KeyCaptureIsNotJudgedByEstimate(KeyCase):
    keys = FOUR_KEYS

    def test_four_captures_under_a_budget_below_their_estimates(self) -> None:
        with mock.patch.dict(os.environ, {"VLLM_GAUDI_MIXED_GRAPH_BUDGET_MIB": "2000"}):
            cache = self.graph.GraphCache(RowForward(), 128, Backend())
            for width in (8192, 4096, 2048, 1024):
                self.boot_capture(cache, self.prompt_batch(1024, 0, width))
            self.assertEqual(cache.counts["captures"], 4)

    def test_max_graphs_still_bounds_the_keys(self) -> None:
        with mock.patch.dict(os.environ, {"VLLM_GAUDI_MIXED_GRAPH_MAX_GRAPHS": "2"}):
            cache = self.graph.GraphCache(RowForward(), 128, Backend())
            self.boot_capture(cache, self.prompt_batch(1024, 0, 8192))
            self.boot_capture(cache, self.prompt_batch(1024, 0, 4096))
            with self.assertRaisesRegex(RuntimeError, "MAX_GRAPHS"):
                self.boot_capture(cache, self.prompt_batch(1024, 0, 2048))


class LatticeAfterSeal(lattice.LatticeBoot):
    """The lattice charges its keys on top of the prompt keys' measured delta, and reads in-use bytes, not free."""

    def test_pool_growth_is_not_demand(self) -> None:
        """R5: a key whose dummy grew the pool (free fell 4000 MiB) but whose allocator in-use rose 10 MiB is charged 10 MiB."""
        runner = FakeRunner(self.graph, charged_mib=2604)
        with mock.patch.object(self.graph, "_free_hbm", side_effect=[4000 * MIB, 0] * 100):
            self.boot(runner, deltas=10)
        self.assertEqual(len(runner.calls), len(self.graph.decode_keys()))

    def test_a_charge_comes_from_the_sealed_delta_not_the_summed_estimates(self) -> None:
        runner = FakeRunner(self.graph, charged_mib=2604)
        output = self.boot(runner)
        witness = next(line for line in output if "GLM53_BOOT_WITNESS" in line)
        self.assertIn("charged_MiB=", witness)
        charged = float(witness.split("charged_MiB=")[1].split()[0])
        self.assertGreaterEqual(charged, 2604.0)
        self.assertLess(charged, 2604.0 + 2000)

    def test_a_dummy_that_built_metadata_but_captured_no_graph_refuses(self) -> None:
        """R3: _DECODE_BUILT proves metadata, not a capture. The graph count must grow by one per key."""
        with self.assertRaisesRegex(RuntimeError, "captured no graph"):
            self.boot(FakeRunner(self.graph), graph_count=lambda: 7)

    def test_a_dummy_that_captured_two_graphs_refuses(self) -> None:
        counts = iter(range(0, 1000, 2))
        with self.assertRaisesRegex(RuntimeError, "captured 2 graphs"):
            self.boot(FakeRunner(self.graph), graph_count=lambda: next(counts))

    def test_no_graph_cache_to_count_refuses_under_graphs_only(self) -> None:
        self.set_env(HPU_GLM53_GRAPHS_ONLY="1")
        with self.assertRaisesRegex(RuntimeError, "no graph count"):
            self.boot(FakeRunner(self.graph), graph_count=lambda: None)

    def test_the_message_names_the_flag_that_loads_the_guard(self) -> None:
        self.set_env(HPU_GLM53_GRAPHS_ONLY="1")
        message = self.graph._capture_growth_error((1, 1024), None, None)
        self.assertIn("HPU_GLM53_PRIVATE_PROMPT_TOPK=1", message)
        self.assertNotIn("HPU_GLM53_GRAPH_STATE=1", message)

    def test_a_failed_in_use_read_after_the_capture_arms_the_bounded_exit(self) -> None:
        reads = iter([0])

        def read() -> int:
            value = next(reads, None)
            if value is None:
                raise RuntimeError("device gone")
            return value

        runner = FakeRunner(self.graph)
        with mock.patch.object(self.graph, "_in_use_bytes", side_effect=read), mock.patch.object(self.graph.graphs_only, "arm_exit") as arm:
            error, _ = self.graph._run_decode_capture(runner, (1, 1024))
        self.assertIn("device gone", error)
        arm.assert_called_once()

    def test_no_graph_cache_to_count_passes_with_the_flag_off(self) -> None:
        runner = FakeRunner(self.graph)
        self.boot(runner, graph_count=lambda: None)
        self.assertTrue(runner.calls)

    def test_a_batch_wider_than_the_top_key_refuses_at_boot(self) -> None:
        """R4: with 32 sequences the 17th decode used to raise at serving."""
        runner = FakeRunner(self.graph)
        runner.input_batch.max_num_reqs = 32
        with self.assertRaisesRegex(RuntimeError, "max-num-seqs 16"):
            self.boot(runner)
        self.assertEqual(runner.calls, [])

    def test_exactly_the_top_key_rows_boots(self) -> None:
        runner = FakeRunner(self.graph)
        self.assertEqual(runner.input_batch.max_num_reqs, 16)
        self.boot(runner)
        self.assertTrue(runner.calls)


class SealBallast(lattice.LatticeBoot):
    """R5: the lattice compiles with no ballast held, so the first replay's workspace is only seen at serving. The seal holds one."""

    def test_the_ballast_is_held_at_the_default_size_and_logged(self) -> None:
        hold = mock.Mock(return_value=3000 * MIB)
        output = self.boot(FakeRunner(self.graph), ballast=hold)
        hold.assert_called_once_with(self.graph.SEAL_BALLAST_MIB * MIB)
        line = next(line for line in output if "GLM53_SEAL_BALLAST" in line)
        self.assertIn(f"ballast_MiB={self.graph.SEAL_BALLAST_MIB}", line)
        self.assertIn("free_with_ballast_MiB=3000.0", line)

    def test_a_ballast_that_does_not_fit_refuses_with_the_numbers(self) -> None:
        hold = mock.Mock(side_effect=MemoryError("PT_DEVMEM"))
        with self.assertRaises(RuntimeError) as refused:
            self.boot(FakeRunner(self.graph), ballast=hold)
        for needle in (str(self.graph.SEAL_BALLAST_MIB), "PT_DEVMEM", "VLLM_GAUDI_MIXED_SEAL_BALLAST_MIB"):
            self.assertIn(needle, str(refused.exception))

    def test_the_ballast_size_is_a_setting(self) -> None:
        self.set_env(VLLM_GAUDI_MIXED_SEAL_BALLAST_MIB="2048")
        hold = mock.Mock(return_value=MIB)
        self.boot(FakeRunner(self.graph), ballast=hold)
        hold.assert_called_once_with(2048 * MIB)

    def fake_device(self, in_use: list[int]) -> tuple[dict, mock.Mock]:
        stats = iter(in_use)
        fake = SimpleNamespace(core=SimpleNamespace(mark_step=lambda: None),
                               hpu=SimpleNamespace(synchronize=lambda: None, mem_get_info=lambda: (77 * MIB, 8452 * MIB),
                                                   memory_stats=lambda: {"InUse": next(stats)}))
        return {"habana_frameworks": SimpleNamespace(torch=fake), "habana_frameworks.torch": fake}, mock.Mock()

    def test_a_ballast_that_really_allocated_reads_the_free_memory_it_left(self) -> None:
        modules, ones = self.fake_device([100 * MIB, 2148 * MIB])
        with mock.patch.dict(sys.modules, modules), mock.patch.object(self.graph.torch, "ones", ones):
            self.assertEqual(self.graph._hold_ballast(2048 * MIB), 77 * MIB)
        self.assertEqual(ones.call_count, 2)

    def test_a_ballast_the_allocator_never_saw_refuses_instead_of_passing_the_seal(self) -> None:
        """Under PT_COMPILE_ONLY_MODE torch.ones may not materialize: a seal that reads no growth has proved nothing."""
        modules, ones = self.fake_device([100 * MIB, 100 * MIB])
        with mock.patch.dict(sys.modules, modules), mock.patch.object(self.graph.torch, "ones", ones), \
                self.assertRaisesRegex(RuntimeError, "2048 MiB.*InUse"):
            self.graph._hold_ballast(2048 * MIB)

    def test_a_refused_ballast_stops_every_rank(self) -> None:
        modules, ones = self.fake_device([100 * MIB, 100 * MIB])
        with mock.patch.dict(sys.modules, modules), mock.patch.object(self.graph.torch, "ones", ones), \
                self.assertRaisesRegex(RuntimeError, "seal could not hold"):
            self.graph.hold_seal_ballast()

    def test_keys_without_a_lattice_hold_the_ballast_too(self) -> None:
        self.set_env(VLLM_GAUDI_MIXED_DECODE_WIDTHS="", HPU_GLM53_DECODE_WIDTH_PAD="0", HPU_GLM53_DECODE_PAD_CHECK_WIDTHS="")
        hold = mock.Mock(return_value=MIB)
        self.boot(FakeRunner(self.graph), ballast=hold)
        hold.assert_called_once()


if __name__ == "__main__":
    unittest.main()
