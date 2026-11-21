"""Step 23: the HBM guard under HPU_GLM53_GRAPHS_ONLY=1 is a witness. No floor decides anything, nothing is cleared or released, the
tripwire logs (only below 1 GiB of headroom) and never charges, and a real allocation failure stops every rank. Flag off, the guard
decides byte for byte as the base image's does (the same sequences replayed through both modules)."""

from __future__ import annotations

import contextlib
import os
import sys
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent))

import gs_replay  # noqa: E402
from gs_replay import R0_ENV, MIB, FakeAllocator, Forward, RankLog  # noqa: E402
from overlay_testlib import GUARD_FILE, load_base, load_overlay  # noqa: E402

ONLY = "HPU_GLM53_GRAPHS_ONLY"
V2_ENV = {**R0_ENV, "HPU_GLM53_GS_TRIPWIRE_V2": "1"}
WITNESS_ENV = {**V2_ENV, ONLY: "1", "HPU_GLM53_PRIVATE_PROMPT_TOPK": "1"}
EXTRACT_ENV = "GS_REPLAY_EXTRACT"
NORMAL_MS = 1300.0
CHUNK = (1024, 131072)


def forward(ms: float = NORMAL_MS, headroom_mib: int = 900, runtime: int = 1, measured_mib: int = 3800, captured: bool = False) -> Forward:
    return Forward(rows=1024, context=131072, measured_mib=measured_mib, headroom_mib=headroom_mib, graphs=19 + runtime,
                   runtime=runtime, ms=ms, captured=captured)


def rank_log(*serving: Forward, normal: Forward | None = None) -> RankLog:
    steady = normal or forward()
    return RankLog(pid=1, serving=[steady, *([steady] * 8), *serving], sealed=True)


class GuardCase(unittest.TestCase):
    def setUp(self) -> None:
        self.overlay = load_overlay(GUARD_FILE)
        self.overlay.reset_state()
        self.lines: list[str] = []
        self.overlay.log_line = self.lines.append

    def guard(self, environ: dict[str, str], free_mib: float = 100.0, cache: dict | None = None) -> object:
        self.cache = {"boot": object(), "runtime": object()} if cache is None else cache
        guard = gs_replay.build_guard(self.overlay, environ, FakeAllocator(), self.cache, lambda: free_mib * MIB)
        guard.seal_boot()
        self.cache["late"] = object()  # a graph captured after the seal: what a release would drop
        return guard


class WitnessChecks(GuardCase):
    def test_flag_off_below_the_floor_the_check_asks_for_a_release(self) -> None:
        guard = self.guard(V2_ENV)
        self.assertFalse(guard.check(self.overlay.EAGER, self.overlay.Chunk(*CHUNK)))

    def test_witness_below_the_floor_keeps_everything_and_reports(self) -> None:
        guard = self.guard(WITNESS_ENV)
        self.assertTrue(guard.check(self.overlay.EAGER, self.overlay.Chunk(*CHUNK)))
        self.assertTrue(guard.check(self.overlay.CAPTURE))
        reports = [line for line in self.lines if "GLM53_GS_HBM " in line]
        self.assertEqual(len(reports), 2)
        self.assertTrue(all("action=report" in line and "witness=1" in line for line in reports), reports)
        self.assertFalse([line for line in self.lines if "action=clear" in line or "GLM53_GS_RELEASE" in line])
        self.assertEqual(set(self.cache), {"boot", "runtime", "late"})
        self.assertEqual((self.overlay.STATS.clears_hbm, self.overlay.STATS.stopped), (0, self.overlay.NOT_STOPPED))

    def test_witness_never_pauses_or_stops_however_low_the_free_gets(self) -> None:
        guard = self.guard(WITNESS_ENV, free_mib=1.0)
        for _ in range(20):
            self.assertTrue(guard.check(self.overlay.EAGER, self.overlay.Chunk(*CHUNK)))
        self.assertEqual(self.overlay.STATS.stopped, self.overlay.NOT_STOPPED)

    def test_a_release_or_a_drop_refuses_under_the_witness(self) -> None:
        guard = self.guard(WITNESS_ENV)
        for action in (guard.release, guard.drop_runtime):
            with self.subTest(action=action.__name__), self.assertRaisesRegex(self.overlay.GraphStateError, ONLY):
                action()
        self.assertEqual(set(self.cache), {"boot", "runtime", "late"})

    def test_a_failed_free_read_stops_every_rank_with_the_fix(self) -> None:
        def unreadable() -> float:
            raise RuntimeError("allocator gone")

        guard = self.guard(WITNESS_ENV)
        guard.free_bytes = unreadable
        with self.assertRaisesRegex(self.overlay.GraphStateError, "free HBM"):
            guard.check(self.overlay.EAGER, self.overlay.Chunk(*CHUNK))

    def test_flag_off_a_failed_free_read_still_just_stops_checking(self) -> None:
        guard = self.guard(V2_ENV)
        guard.free_bytes = mock.Mock(side_effect=RuntimeError("allocator gone"))
        self.assertTrue(guard.check(self.overlay.EAGER, self.overlay.Chunk(*CHUNK)))
        self.assertEqual(self.overlay.STATS.stopped, self.overlay.STOP_ERROR)

    def test_the_lru_guard_evicts_so_the_witness_refuses_it(self) -> None:
        with self.assertRaisesRegex(self.overlay.GraphStateError, "HPU_GLM53_GS_HBM_GUARD"):
            self.guard({**WITNESS_ENV, "HPU_GLM53_GS_HBM_GUARD": "lru"})

    def test_the_witness_needs_the_private_top_k_and_a_clean_value(self) -> None:
        with self.assertRaisesRegex(self.overlay.GraphStateError, "HPU_GLM53_PRIVATE_PROMPT_TOPK"):
            self.overlay.read_mode({ONLY: "1"})
        with self.assertRaisesRegex(self.overlay.GraphStateError, ONLY):
            self.overlay.read_mode({ONLY: "yes", "HPU_GLM53_PRIVATE_PROMPT_TOPK": "1"})
        self.assertTrue(self.overlay.read_mode({ONLY: "1", "HPU_GLM53_PRIVATE_PROMPT_TOPK": "1"}).private)
        self.assertFalse(self.overlay.read_mode({}).private)

    def test_the_boot_line_says_the_guard_is_a_witness(self) -> None:
        self.guard(WITNESS_ENV)
        self.assertTrue([line for line in self.lines if "GLM53_GS_WITNESS" in line and "release=refused" in line], self.lines)
        self.lines.clear()
        self.guard(V2_ENV)
        self.assertFalse([line for line in self.lines if "GLM53_GS_WITNESS" in line])


class WitnessTripwire(GuardCase):
    def replay(self, environ: dict[str, str], *serving: Forward, **kwargs: object) -> gs_replay.ReplayResult:
        return gs_replay.replay(self.overlay, rank_log(*serving, **kwargs), environ)

    def test_a_slow_chunk_under_one_gib_of_headroom_trips_and_only_logs(self) -> None:
        result = self.replay(WITNESS_ENV, forward(ms=2100.0, headroom_mib=900))
        self.assertEqual(result.other_trips, 1)
        trip = next(line for line in result.lines if "event=tripwire " in line)
        self.assertIn("log_only=1", trip)
        self.assertIn("headroom_mib=900", trip)
        self.assertEqual((result.guard.tripwire_left, result.guard.tripwire_charge), (0, 0))
        self.assertFalse([line for line in result.lines if "tripwire_left=63" in line])

    def test_flag_off_the_same_slow_chunk_charges_the_next_64(self) -> None:
        result = self.replay(V2_ENV, forward(ms=2100.0, headroom_mib=900))
        self.assertEqual(result.other_trips, 1)
        self.assertEqual(result.guard.tripwire_left, 64)
        self.assertGreater(result.guard.tripwire_charge, 0)

    def test_the_gate_is_one_gib_exactly(self) -> None:
        for headroom, trips in ((1023, 1), (1024, 0), (1100, 0)):
            with self.subTest(headroom=headroom):
                result = self.replay(WITNESS_ENV, forward(ms=2100.0, headroom_mib=headroom), normal=forward(headroom_mib=headroom))
                self.assertEqual(result.other_trips, trips)

    def test_graphs_all_boot_still_count_as_resident(self) -> None:
        """Under the flag no graph is captured at runtime, so the base's `runtime > 0` condition would never be true."""
        log = rank_log(forward(ms=2100.0, runtime=0), normal=forward(runtime=0))
        self.assertEqual(gs_replay.replay(self.overlay, log, WITNESS_ENV).other_trips, 1)
        self.assertEqual(gs_replay.replay(self.overlay, log, V2_ENV).other_trips, 0)

    def test_a_normal_chunk_and_a_chunk_behind_a_capture_do_not_trip(self) -> None:
        result = self.replay(WITNESS_ENV, forward(), forward(ms=2100.0, captured=True))
        self.assertEqual((result.other_trips, result.capture_trips), (0, 0))


class WitnessGrowth(GuardCase):
    """R3: serving checks a Python set of decode keys, so a graph-hash miss would capture silently. Under the witness a cache that grows
    after the seal refuses, and the bounded exit backs it up."""

    def serve(self, guard: object, shape: tuple | None, grow: bool) -> bool:
        step = guard.before_graph(shape)
        if grow:
            self.cache[f"serving-{len(self.cache)}"] = object()
        return guard.after_graph(shape, step, 16)

    def test_a_graph_captured_after_the_seal_refuses_and_arms_the_exit(self) -> None:
        guard = self.guard(WITNESS_ENV)
        with mock.patch.object(threading, "Timer") as timer, self.assertRaises(self.overlay.GraphStateError) as refused:
            self.serve(guard, (16, 4096), grow=True)
        for needle in (ONLY, "after the seal", "(16, 4096)", "HPU_GLM53_DECODE_WIDTH_PAD"):
            self.assertIn(needle, str(refused.exception))
        timer.assert_called_once()

    def test_an_unshaped_graph_growth_refuses_too(self) -> None:
        guard = self.guard(WITNESS_ENV)
        with mock.patch.object(threading, "Timer"), self.assertRaisesRegex(self.overlay.GraphStateError, "after the seal"):
            self.serve(guard, None, grow=True)

    def test_a_replay_does_not_refuse(self) -> None:
        guard = self.guard(WITNESS_ENV)
        self.assertFalse(self.serve(guard, None, grow=False))

    def test_flag_off_a_serving_capture_is_tracked_as_before(self) -> None:
        guard = self.guard(V2_ENV)
        self.assertTrue(self.serve(guard, (16, 4096), grow=True))

    def test_growth_before_the_seal_is_the_boot_set(self) -> None:
        guard = gs_replay.build_guard(self.overlay, WITNESS_ENV, FakeAllocator(), {"boot": object()}, lambda: 100.0 * MIB)
        step = guard.before_graph((16, 4096))
        guard.graph_cache["second"] = object()
        self.assertTrue(guard.after_graph((16, 4096), step, 16))


class WitnessReplay(GuardCase):
    """The route's check points over a recorded sequence: no release, no clear, whatever the free HBM does."""

    def log(self) -> RankLog:
        low = forward(measured_mib=300, headroom_mib=100)
        return rank_log(*[low] * 6, forward(measured_mib=300, headroom_mib=100, captured=True), *[low] * 6, normal=low)

    def test_the_base_releases_on_this_sequence(self) -> None:
        base = load_base(GUARD_FILE)
        base.reset_state()
        self.assertGreater(gs_replay.replay(base, self.log(), R0_ENV, drive_checks=True).releases, 0)

    def test_the_witness_releases_nothing_and_clears_nothing(self) -> None:
        result = gs_replay.replay(self.overlay, self.log(), WITNESS_ENV, drive_checks=True)
        self.assertEqual(result.releases, 0)
        self.assertEqual((self.overlay.STATS.clears_hbm, self.overlay.STATS.released), (0, 0))
        self.assertFalse([line for line in result.lines if "GLM53_GS_RELEASE" in line or "action=clear" in line])
        self.assertTrue([line for line in result.lines if "GLM53_GS_HBM " in line])

    def test_flag_off_decides_exactly_as_the_base_does(self) -> None:
        base_module = load_base(GUARD_FILE)
        base_module.reset_state()
        base = gs_replay.replay(base_module, self.log(), R0_ENV, drive_checks=True)
        self.overlay.reset_state()
        overlay = gs_replay.replay(self.overlay, self.log(), R0_ENV, drive_checks=True)
        self.assertGreater(base.releases, 0)
        self.assertEqual((overlay.lines, overlay.trips, overlay.releases), (base.lines, base.trips, base.releases))


class WitnessRoute(unittest.TestCase):
    def setUp(self) -> None:
        self.guard = load_overlay(GUARD_FILE)
        timer = mock.patch.object(threading, "Timer")
        self.timer = timer.start()
        self.addCleanup(timer.stop)

    def routed(self, witness: bool) -> object:
        forward = self.guard.RoutedForward.__new__(self.guard.RoutedForward)
        forward.hbm = SimpleNamespace(witness=witness, seal_boot=lambda: None, pending=False)
        forward.route = SimpleNamespace(latched=lambda: True, should_bypass=lambda *_: False, bypass_min_tokens=512,
                                        recapture_min_width=2, adaptive=mock.Mock(return_value="eager-result"))
        forward.mode = SimpleNamespace(routes=True, private=True, census=False)
        forward.state = SimpleNamespace(private=object(), bind=contextlib.nullcontext)
        forward.mixed_decode_graph_forwards = 0
        return forward

    @staticmethod
    def latched_decode() -> dict[str, object]:
        import torch
        return {"input_ids": torch.zeros((1, 1), dtype=torch.long), "attn_metadata": SimpleNamespace(is_prompt=False),
                "refresh_hpu_graphs_after_external_bypass": False, "external_graph_transition_active": True}

    def test_a_latched_decode_would_run_eager_so_the_witness_refuses_it(self) -> None:
        forward = self.routed(witness=True)
        with mock.patch.dict(os.environ, {"HPU_GLM53_MIXED_DECODE_REPLAY": "0"}), self.assertRaisesRegex(self.guard.GraphStateError, ONLY):
            forward(**self.latched_decode())
        forward.route.adaptive.assert_not_called()

    def test_flag_off_the_latched_decode_still_runs_deferred(self) -> None:
        forward = self.routed(witness=False)
        with mock.patch.dict(os.environ, {"HPU_GLM53_MIXED_DECODE_REPLAY": "0"}):
            self.assertEqual(forward(**self.latched_decode()), "eager-result")

    def test_every_eager_forward_is_counted_for_the_boot_witness(self) -> None:
        self.guard.reset_state()
        forward = self.routed(witness=False)
        with mock.patch.dict(os.environ, {"HPU_GLM53_MIXED_DECODE_REPLAY": "0"}):
            forward(**self.latched_decode())
        self.assertEqual(self.guard.STATS.eager_forwards, 1)
        self.assertNotIn("eager_forwards", self.guard.STATS.line(), "the STATS line is the base's, byte for byte")

    def test_a_graph_forward_is_not_counted(self) -> None:
        self.guard.reset_state()
        forward = self.routed(witness=False)
        forward._graph = mock.Mock(return_value="replayed")
        with mock.patch.object(self.guard, "classify", return_value=self.guard.GRAPH):
            self.assertEqual(forward(), "replayed")
        self.assertEqual(self.guard.STATS.eager_forwards, 0)

    def test_a_real_allocation_failure_re_raises_and_arms_the_bounded_exit(self) -> None:
        forward = self.routed(witness=True)
        forward._routed = mock.Mock(side_effect=RuntimeError("PT_DEVMEM: Failed to allocate"))
        with self.assertRaisesRegex(RuntimeError, "PT_DEVMEM"):
            forward()
        self.timer.assert_called_once()

    def test_another_error_or_the_flag_off_arms_nothing(self) -> None:
        for witness, message in ((True, "shape mismatch"), (False, "PT_DEVMEM: Failed to allocate")):
            with self.subTest(witness=witness):
                forward = self.routed(witness=witness)
                forward._routed = mock.Mock(side_effect=RuntimeError(message))
                with self.assertRaisesRegex(RuntimeError, message):
                    forward()
        self.timer.assert_not_called()

    def test_the_witness_has_no_hbm_guard_to_ask_when_there_is_none(self) -> None:
        forward = self.routed(witness=False)
        forward.hbm = None
        forward._routed = mock.Mock(return_value="plain")
        self.assertEqual(forward(), "plain")


@unittest.skipUnless(os.environ.get(EXTRACT_ENV), f"set {EXTRACT_ENV} to a docker logs extract of a replica")
class WitnessExtractReplay(unittest.TestCase):
    """Real r0/r1 sequences: 0 releases, and the trips left are exactly those under 1 GiB of headroom."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.ranks = gs_replay.read_extract(Path(os.environ[EXTRACT_ENV]))
        gs_replay.share_captures(cls.ranks)
        cls.overlay = load_overlay(GUARD_FILE)

    def test_no_release_and_every_trip_is_under_one_gib(self) -> None:
        for rank in self.ranks.values():
            self.overlay.reset_state()
            result = gs_replay.replay(self.overlay, rank, WITNESS_ENV, drive_checks=True)
            self.assertEqual(result.releases, 0, f"pid {rank.pid}")
            self.assertFalse([line for line in result.lines if "GLM53_GS_RELEASE" in line], f"pid {rank.pid}")
            self.assertTrue(all(headroom < 1024 for headroom in result.trip_headrooms), f"pid {rank.pid}")

    def test_the_trips_left_contain_every_base_trip_under_one_gib(self) -> None:
        for rank in self.ranks.values():
            self.overlay.reset_state()
            base = gs_replay.replay(self.overlay, rank, V2_ENV)
            kept = sorted(trip for trip, headroom in zip(base.trips, base.trip_headrooms) if headroom < 1024)
            witness = gs_replay.replay(self.overlay, rank, WITNESS_ENV, drive_checks=True)
            self.assertLess(len(kept), len(base.trips), f"pid {rank.pid}: the gate must drop some of the {len(base.trips)}")
            self.assertTrue(set(kept) <= set(witness.trips), f"pid {rank.pid}: {len(kept)} of {len(base.trips)} kept")

    def test_flag_off_decisions_are_byte_identical_to_base(self) -> None:
        base_module = load_base(GUARD_FILE)
        for rank in self.ranks.values():
            base_module.reset_state()
            self.overlay.reset_state()
            base = gs_replay.replay(base_module, rank, R0_ENV, drive_checks=True)
            overlay = gs_replay.replay(self.overlay, rank, R0_ENV, drive_checks=True)
            self.assertEqual((overlay.lines, overlay.trips, overlay.releases), (base.lines, base.trips, base.releases), f"pid {rank.pid}")


if __name__ == "__main__":
    unittest.main()
