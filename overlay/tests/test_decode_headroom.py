"""A decode key that cannot fit must refuse the boot on every rank before its capture, never kill a worker (diag-key128 r1, 2026-11-07).

At VLLM_GAUDI_MIXED_GRAPH_BUDGET_MIB=12000 two boots died instead of refusing. k256: rank 0 had 63 MiB under the allocator limit before decode
key 8x2048, whose capture takes 171 MiB, and the PT_DEVMEM abort is not an exception the capture can catch. k512: the prompt keys left 236 MiB
and the first decode capture failed inside the guarded forward. Both passed the lattice admission because the per-row estimates (owner-off,
width-blind) charged 84 MiB for keys that took 339. CPU only: the allocator readings are patched; the measured costs are r1's own log lines."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent))

import test_decode_lattice as lattice  # noqa: E402
from test_decode_lattice import MIB, FakeRunner  # noqa: E402

# GLM53_DECODE_KEY hbm_delta_MiB, max over ranks and the base, k128, k256 and k512 boots of diag-key128/runs/r1 (shared gather owner on).
MEASURED_MIB = {
    (1, 1024): 15.7, (2, 1024): 36.3, (4, 1024): 71.9, (8, 1024): 143.1, (16, 1024): 292.2,
    (1, 2048): 15.7, (2, 2048): 44.3, (4, 2048): 86.5, (8, 2048): 171.1, (16, 2048): 346.4,
    (1, 4096): 25.5, (2, 4096): 60.3, (4, 4096): 115.8, (8, 4096): 226.8, (16, 4096): 456.1,
    (1, 8192): 43.9, (2, 8192): 92.4, (4, 8192): 174.9, (8, 8192): 338.9, (16, 8192): 750.6,
}
ROOMY_MIB = 100_000


class Estimate(lattice.GraphCase):
    def test_no_measured_key_is_undercharged(self) -> None:
        """r1 k256: the 8x8192 estimate was 84.0 MiB and the capture took 338.8."""
        for (rows, width), taken in MEASURED_MIB.items():
            with self.subTest(key=f"{rows}x{width}"):
                self.assertGreaterEqual(self.graph.decode_key_estimate_bytes(rows, width), taken * MIB)

    def test_the_owner_off_is_charged_the_owner_ratio_more(self) -> None:
        """The r1 readings were taken with the owner on; section 3's only owner reading has the owner at 72% of owner-off, so owner-off is
        the reading over 0.72 until it is measured."""
        for rows, width in MEASURED_MIB:
            on = self.graph.decode_key_estimate_bytes(rows, width)
            for value in ("0", ""):
                with self.subTest(key=f"{rows}x{width}", owner=value), mock.patch.dict("os.environ", {"HPU_GLM53_Q4_SHARED_GATHER_OWNER": value}):
                    off = self.graph.decode_key_estimate_bytes(rows, width)
                self.assertGreaterEqual(off * self.graph.DECODE_OWNER_PERCENT, on * 100)
                self.assertGreater(off, on)

    def test_a_width_wider_than_any_reading_scales_up_from_the_widest(self) -> None:
        self.assertGreaterEqual(self.graph.decode_key_estimate_bytes(16, 16384), 2 * self.graph.decode_key_estimate_bytes(16, 8192))

    def test_a_width_between_readings_takes_the_wider_one(self) -> None:
        self.assertEqual(self.graph.decode_key_estimate_bytes(4, 3000), self.graph.decode_key_estimate_bytes(4, 4096))
        self.assertEqual(self.graph.decode_key_estimate_bytes(4, 512), self.graph.decode_key_estimate_bytes(4, 1024))


class Admission(lattice.LatticeBoot):
    def boot_with_headroom(self, runner: FakeRunner, headroom: list[int]) -> list[str]:
        return self.boot(runner, headroom=headroom)

    def test_k512_at_12000_refuses_before_any_capture(self) -> None:
        """r1 k512: 8,282 MiB of prompt keys plus a lattice that really takes about 3,500 passed a 12,000 MiB budget on 1,360 of estimates."""
        self.set_env(VLLM_GAUDI_MIXED_GRAPH_BUDGET_MIB="12000")
        runner = FakeRunner(self.graph, charged_mib=8282)
        with self.assertRaisesRegex(RuntimeError, "rejected on ALL ranks: rank 0: 20 decode keys need"):
            self.graph.warmup_decode_lattice(runner)
        self.assertEqual(runner.calls, [])

    def test_a_key_past_the_allocator_limit_refuses_before_its_capture(self) -> None:
        """r1 k256: 63 MiB left under the limit before 8x2048, the 17th key; its capture aborted the worker on PT_DEVMEM."""
        self.set_env(VLLM_GAUDI_MIXED_GRAPH_BUDGET_MIB="12000")
        runner = FakeRunner(self.graph, charged_mib=2560)
        keys = self.graph.decode_keys()
        index = keys.index((8, 2048))
        with self.assertRaises(RuntimeError) as refused:
            self.boot_with_headroom(runner, [ROOMY_MIB * MIB] * index + [63 * MIB])
        message = str(refused.exception)
        for needle in ("rejected on ALL ranks", "decode key 8x2048", "63.0 MiB", "VLLM_GAUDI_MIXED_DECODE_WIDTHS",
                       "VLLM_GAUDI_MIXED_PROMPT_KEYS", "cannot help"):
            self.assertIn(needle, message)
        self.assertEqual(len(runner.calls), index)

    def test_the_requirement_is_the_estimate_and_a_margin(self) -> None:
        estimate = self.graph.decode_key_estimate_bytes(8, 2048)
        self.assertEqual(self.graph.decode_capture_requirement((8, 2048)), estimate + self.graph.DECODE_CAPTURE_MARGIN_MIB * MIB)
        self.assertGreater(self.graph.DECODE_CAPTURE_MARGIN_MIB, 0)

    def test_exactly_the_requirement_captures_every_key(self) -> None:
        runner = FakeRunner(self.graph, charged_mib=2560)
        headroom = [self.graph.decode_capture_requirement(key) for key in self.graph.decode_keys()]
        output = self.boot_with_headroom(runner, headroom)
        self.assertEqual(len(runner.calls), len(self.graph.decode_keys()))
        self.assertTrue(any("headroom_MiB=" in line for line in output if "GLM53_DECODE_KEY captured" in line))

    def test_one_byte_short_refuses(self) -> None:
        runner = FakeRunner(self.graph, charged_mib=2560)
        with self.assertRaisesRegex(RuntimeError, "decode key 16x8192"):
            self.boot_with_headroom(runner, [self.graph.decode_capture_requirement((16, 8192)) - 1])
        self.assertEqual(runner.calls, [])

    def test_a_failed_headroom_read_refuses_instead_of_capturing_blind(self) -> None:
        runner = FakeRunner(self.graph, charged_mib=2560)
        with self.assertRaisesRegex(RuntimeError, "headroom could not be read.*device gone"):
            self.boot_with_headroom(runner, RuntimeError("device gone"))
        self.assertEqual(runner.calls, [])

    def test_a_capture_that_peaks_past_its_estimate_refuses_and_names_the_table(self) -> None:
        """The allocation fails on the capture's transient peak, not on what it keeps; r1's peaks equal its residuals, so a peak past the
        estimate means the table is wrong for this build, and a boot that fit only on the margin must not pass as measured."""
        runner = FakeRunner(self.graph, charged_mib=2560)
        estimate = self.graph.decode_key_estimate_bytes(16, 8192)
        with self.assertRaises(RuntimeError) as refused:
            self.boot(runner, peak=lambda: estimate + MIB)
        for needle in ("decode key 16x8192", "peak", "DECODE_KEY_MIB_PER_ROW"):
            self.assertIn(needle, str(refused.exception))
        self.assertEqual(len(runner.calls), 1)

    def test_the_peak_is_logged_and_its_window_starts_at_the_capture(self) -> None:
        runner = FakeRunner(self.graph, charged_mib=2560)
        order: list[str] = []
        output = self.boot(runner, deltas=10, peak=lambda: order.append("peak") or 10 * MIB,
                           reset_peak=lambda: order.append("reset"))
        self.assertEqual(order[:2], ["reset", "peak"])
        self.assertEqual(order.count("reset"), len(self.graph.decode_keys()))
        lines = [line for line in output if "GLM53_DECODE_KEY captured" in line]
        self.assertTrue(lines and all("peak_MiB=10.0" in line for line in lines))

    def test_the_r1_base_and_k128_lattices_still_admit(self) -> None:
        """Their prompt keys sealed 2,560 and 3,617 MiB (GLM53_KEY_SEAL); with the lattice at its estimates both stay under 8,452."""
        for charged in (2560, 3618):
            with self.subTest(charged=charged):
                runner = FakeRunner(self.graph, charged_mib=charged)
                self.assertEqual(self.graph._decode_lattice_error(runner, self.graph.decode_keys(), charged * MIB, 8452 * MIB), "")


class ListedKeys(lattice.LatticeBoot):
    """L2a (decode lane, glm53_decode_scan_v2): a key in HPU_GLM53_DECODE_KPOOL_TILE_CEILING_FOR scans 32,768- or 65,536-pool chunks under
    its own budget, while r1 measured the default 16,384-pool tile. The indexer refuses a listed chunk past that budget (budget_shortfall),
    so the budget bounds the extra workspace; the requirement adds it in full."""

    def scan_v2(self, listed: dict, budget_mib: int) -> SimpleNamespace:
        return SimpleNamespace(tile_ceiling_for=lambda key: listed.get(key), chunk_budget_mib=lambda: budget_mib)

    def requirement(self, key: tuple[int, int], module: object | None, context_mib: str = "512") -> int:
        modules = {"glm53_decode_scan_v2": module} if module is not None else {}
        with mock.patch.dict(sys.modules, modules), mock.patch.dict("os.environ", {"HPU_GLM53_CONTEXT_CHUNK_BUDGET_MIB": context_mib}):
            if module is None:
                sys.modules.pop("glm53_decode_scan_v2", None)
            return self.graph.decode_capture_requirement(key)

    def test_a_listed_key_adds_its_chunk_budget(self) -> None:
        base = self.graph.decode_key_estimate_bytes(16, 8192) + self.graph.DECODE_CAPTURE_MARGIN_MIB * MIB
        module = self.scan_v2({(16, 8192): 32768}, 1024)
        self.assertEqual(self.requirement((16, 8192), module), base + 1024 * MIB)

    def test_the_budget_is_never_below_the_context_budget(self) -> None:
        """As the indexer bounds it: max(context budget, listed keys' budget); an unset listed budget reads 0."""
        base = self.graph.decode_key_estimate_bytes(16, 8192) + self.graph.DECODE_CAPTURE_MARGIN_MIB * MIB
        module = self.scan_v2({(16, 8192): 32768}, 0)
        self.assertEqual(self.requirement((16, 8192), module, context_mib="512"), base + 512 * MIB)

    def test_an_unlisted_key_and_no_lane_module_add_nothing(self) -> None:
        base = self.graph.decode_key_estimate_bytes(8, 4096) + self.graph.DECODE_CAPTURE_MARGIN_MIB * MIB
        self.assertEqual(self.requirement((8, 4096), self.scan_v2({(16, 8192): 32768}, 1024)), base)
        self.assertEqual(self.requirement((8, 4096), None), base)

    def test_a_listed_key_short_of_its_workspace_refuses_before_its_capture(self) -> None:
        module = self.scan_v2({(16, 8192): 32768}, 1024)
        runner = FakeRunner(self.graph, charged_mib=2560)
        plain = self.graph.decode_key_estimate_bytes(16, 8192) + self.graph.DECODE_CAPTURE_MARGIN_MIB * MIB
        with mock.patch.dict(sys.modules, {"glm53_decode_scan_v2": module}), self.assertRaises(RuntimeError) as refused:
            self.boot(runner, headroom=lambda: plain + 1023 * MIB)
        self.assertIn("decode key 16x8192", str(refused.exception))
        self.assertIn("scan workspace", str(refused.exception))
        self.assertEqual(runner.calls, [])


class MeasuredCharge(lattice.LatticeBoot):
    """With the HBM margin seal installed (HPU_GLM53_HBM_MARGIN_MIB and diag-hbm's wrapper over seal_boot_memory), a decode key's charge is
    the bytes its capture kept, not the table. That charge reaches the boot witness's charged_MiB only: the lattice admission before the
    captures still charges every key its estimate.

    Fable 11-08: run3's lattice was charged 3,720 MiB of estimate against 2,444 measured, and c1c's 16x8192 peak of 752.8 MiB against a
    752.0 estimate refused a boot that had 64 MiB of capture margin left. The strict peak <= estimate check relaxes to estimate + capture
    margin only when the seal also fenced-replays the decode keys and refuses on zero readings (glm53_hbm_margin.key_replays, Fable 2's
    7bf39655, whose gate_error refuses an empty reading list): a seal that reads prompt keys only judges no decode key's headroom."""

    MARGIN_ON = {"HPU_GLM53_HBM_MARGIN_MIB": "512"}

    def install_seal(self, *, decode_contract: bool = True, wrapper_name: str = "seal_boot_memory_with_margin", marked: bool = False) -> None:
        """diag-hbm's install_from_env, faked: the wrapper over seal_boot_memory (named as that seal version names it, with the
        glm53_hbm_margin_seal marker when ``marked``) and the module's decode-key contract."""
        original = self.graph.seal_boot_memory

        def seal_boot_memory_with_margin(runner: object, charged: int) -> None:
            original(runner, charged)

        seal_boot_memory_with_margin.__name__ = wrapper_name
        if marked:
            seal_boot_memory_with_margin.glm53_hbm_margin_seal = True
        contract = {"key_replays": lambda mixed_graph, runner: []} if decode_contract else {}
        for patcher in (mock.patch.object(self.graph, "seal_boot_memory", seal_boot_memory_with_margin),
                        mock.patch.dict(sys.modules, {"glm53_hbm_margin": SimpleNamespace(**contract)})):
            patcher.start()
            self.addCleanup(patcher.stop)

    def charged(self, deltas: int, **env: str) -> int:
        self.set_env(**env)
        runner = FakeRunner(self.graph, charged_mib=2560)
        in_use = iter(0 if i % 2 == 0 else deltas * MIB for i in range(400))
        with mock.patch.object(self.graph, "_in_use_bytes", side_effect=lambda: next(in_use)), \
                mock.patch.object(self.graph, "_free_hbm", return_value=0), \
                mock.patch.object(self.graph, "_allocator_headroom", return_value=ROOMY_MIB * MIB), \
                mock.patch.object(self.graph, "_reset_peak_in_use"), \
                mock.patch.object(self.graph, "_peak_in_use_bytes", return_value=deltas * MIB), \
                mock.patch.object(self.graph, "_graph_count", side_effect=lambda: len(runner.calls)), \
                self.assertLogs(self.graph.LOG, "WARNING"):
            return self.graph.warmup_decode_lattice(runner)

    def estimates(self) -> int:
        return sum(self.graph.decode_key_estimate_bytes(*key) for key in self.graph.decode_keys())

    def peak_past_estimate(self, runner: FakeRunner) -> list[str]:
        """Each key peaks 1 MiB past its own estimate; the peak is read after its dummy, so the last call is the key being captured."""
        keys = self.graph.decode_keys()
        return self.boot(runner, peak=lambda: self.graph.decode_key_estimate_bytes(*keys[len(runner.calls) - 1]) + MIB)

    def test_the_margin_seal_charges_each_key_what_its_capture_kept(self) -> None:
        self.install_seal()
        keys = self.graph.decode_keys()
        self.assertEqual(self.charged(10, **self.MARGIN_ON), (2560 + 10 * len(keys)) * MIB)

    def test_the_ballast_v2_wrapper_name_is_the_seal_too(self) -> None:
        """diag-hbm d54efdc6 (ballast v2 + StateGuard) names its wrapper _seal_boot_memory_with_margin; it must still count as the seal."""
        self.install_seal(wrapper_name="_seal_boot_memory_with_margin")
        keys = self.graph.decode_keys()
        self.assertEqual(self.charged(10, **self.MARGIN_ON), (2560 + 10 * len(keys)) * MIB)

    def test_the_marker_makes_any_wrapper_the_seal(self) -> None:
        """The marker is the contract, whatever the wrapper is named: with the margin set, the seal-on check reads True."""
        self.set_env(**self.MARGIN_ON)
        self.install_seal(wrapper_name="renamed_again", marked=True)
        self.assertTrue(self.graph.hbm_margin_seal_on())
        self.assertTrue(self.graph.hbm_margin_seals_decode_keys())
        keys = self.graph.decode_keys()
        self.assertEqual(self.charged(10, **self.MARGIN_ON), (2560 + 10 * len(keys)) * MIB)

    def test_the_marker_without_the_margin_is_not_the_seal(self) -> None:
        self.set_env(HPU_GLM53_HBM_MARGIN_MIB="")
        self.install_seal(wrapper_name="renamed_again", marked=True)
        self.assertFalse(self.graph.hbm_margin_seal_on())

    def test_a_wrapper_of_another_name_is_not_the_seal(self) -> None:
        self.install_seal(wrapper_name="some_other_wrapper")
        self.assertEqual(self.charged(10, **self.MARGIN_ON), 2560 * MIB + self.estimates())

    def test_without_the_seal_each_key_is_charged_its_estimate_as_before(self) -> None:
        for value in ("", "0"):
            with self.subTest(flag=value):
                self.assertEqual(self.charged(10, HPU_GLM53_HBM_MARGIN_MIB=value), 2560 * MIB + self.estimates())

    def test_the_flag_without_the_installed_wrapper_charges_the_estimate(self) -> None:
        """The env alone is not the seal: no glm53_hbm_margin.pth in the image means nothing judges the headroom."""
        self.assertEqual(self.charged(10, **self.MARGIN_ON), 2560 * MIB + self.estimates())

    def test_a_peak_inside_the_capture_margin_boots_under_the_seal(self) -> None:
        """c1c r2: 16x8192 peaked at 752.8 MiB against 47 x 16 = 752.0; the refusal was the table's, not the device's."""
        self.set_env(**self.MARGIN_ON)
        self.install_seal()
        runner = FakeRunner(self.graph, charged_mib=2560)
        captured = [line for line in self.peak_past_estimate(runner) if "GLM53_DECODE_KEY captured" in line]
        self.assertEqual(len(captured), len(self.graph.decode_keys()))

    def test_a_seal_that_reads_prompt_keys_only_keeps_the_strict_peak_check(self) -> None:
        """diag-hbm 7ed2594b replayed bucket[0] == 0 only: a decode-only lattice gave no reading and gate_error([]) passed."""
        self.set_env(**self.MARGIN_ON)
        self.install_seal(decode_contract=False)
        with self.assertRaisesRegex(RuntimeError, "past its .* estimate"):
            self.peak_past_estimate(FakeRunner(self.graph, charged_mib=2560))

    def test_the_flag_without_the_installed_wrapper_keeps_the_strict_peak_check(self) -> None:
        self.set_env(**self.MARGIN_ON)
        with self.assertRaisesRegex(RuntimeError, "past its .* estimate"):
            self.peak_past_estimate(FakeRunner(self.graph, charged_mib=2560))

    def test_a_peak_past_the_estimate_and_its_margin_still_refuses_under_the_seal(self) -> None:
        self.set_env(**self.MARGIN_ON)
        self.install_seal()
        requirement = self.graph.decode_capture_requirement((16, 8192))
        with self.assertRaises(RuntimeError) as refused:
            self.boot(FakeRunner(self.graph, charged_mib=2560), peak=lambda: requirement + MIB)
        for needle in ("decode key 16x8192", "capture margin", "DECODE_KEY_MIB_PER_ROW"):
            self.assertIn(needle, str(refused.exception))

    def test_a_bad_margin_value_is_not_read_as_on(self) -> None:
        estimate = self.graph.decode_key_estimate_bytes(16, 8192)
        for value in ("abc", "-5"):
            with self.subTest(flag=value), self.assertRaisesRegex(RuntimeError, "past its .* estimate"):
                self.set_env(HPU_GLM53_HBM_MARGIN_MIB=value)
                self.boot(FakeRunner(self.graph, charged_mib=2560), peak=lambda: estimate + MIB)


if __name__ == "__main__":
    unittest.main()
