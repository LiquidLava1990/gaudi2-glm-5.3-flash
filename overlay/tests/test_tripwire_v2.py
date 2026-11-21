"""Step 4: guard tripwire v2. A prompt chunk that follows a graph capture in its own step neither samples nor trips."""

from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import gs_replay  # noqa: E402
from gs_replay import R0_ENV, FakeAllocator, Forward, RankLog  # noqa: E402
from overlay_testlib import GUARD_FILE, load_base, load_overlay  # noqa: E402

V2_ENV = {**R0_ENV, "HPU_GLM53_GS_TRIPWIRE_V2": "1"}
EXTRACT_ENV = "GS_REPLAY_EXTRACT"
NORMAL_MS = 1300.0


def forward(ms: float = NORMAL_MS, captured: bool = False, runtime: int = 1) -> Forward:
    return Forward(rows=1024, context=131072, measured_mib=3800, headroom_mib=900, graphs=19 + runtime,
                   runtime=runtime, ms=ms, captured=captured)


def rank_log(*serving: Forward) -> RankLog:
    return RankLog(pid=1, serving=[forward(), *([forward()] * 8), *serving], sealed=True)


class TripwireV2Synthetic(unittest.TestCase):
    def setUp(self) -> None:
        self.overlay = load_overlay(GUARD_FILE)
        self.base = load_base(GUARD_FILE)

    def test_base_trips_on_a_slow_capture_step(self) -> None:
        result = gs_replay.replay(self.base, rank_log(forward(ms=2100.0, captured=True)), R0_ENV)
        self.assertEqual(result.capture_trips, 1)

    def test_flag_off_trips_on_a_slow_capture_step(self) -> None:
        result = gs_replay.replay(self.overlay, rank_log(forward(ms=2100.0, captured=True)), R0_ENV)
        self.assertEqual(result.capture_trips, 1)

    def test_flag_on_does_not_trip_on_a_slow_capture_step(self) -> None:
        result = gs_replay.replay(self.overlay, rank_log(forward(ms=2100.0, captured=True)), V2_ENV)
        self.assertEqual(result.capture_trips, 0)
        self.assertEqual(result.trips, [])

    def test_flag_on_still_trips_on_a_slow_non_capture_chunk(self) -> None:
        result = gs_replay.replay(self.overlay, rank_log(forward(ms=2100.0)), V2_ENV)
        self.assertEqual(result.other_trips, 1)

    def test_flag_on_keeps_the_slowdown_factor(self) -> None:
        under = gs_replay.replay(self.overlay, rank_log(forward(ms=NORMAL_MS * 1.49)), V2_ENV)
        over = gs_replay.replay(self.overlay, rank_log(forward(ms=NORMAL_MS * 1.51)), V2_ENV)
        self.assertEqual((under.other_trips, over.other_trips), (0, 1))

    def test_capture_sample_stays_out_of_the_bucket_history(self) -> None:
        """Eight slow capture samples would lift the p50 and hide a later slow chunk."""
        slow_captures = [forward(ms=2100.0, captured=True)] * 40
        result = gs_replay.replay(self.overlay, rank_log(*slow_captures, forward(ms=2100.0)), V2_ENV)
        self.assertEqual((result.capture_trips, result.other_trips), (0, 1))

    def test_capture_flag_does_not_outlive_its_step(self) -> None:
        """A replay forward between the capture and the slow chunk means the slow chunk is not a capture step."""
        self.overlay.log_line = lambda line: None
        allocator, cache = FakeAllocator(), {"boot": object()}
        guard = gs_replay.build_guard(self.overlay, V2_ENV, allocator, cache)
        guard.seal_boot()
        gs_replay._graph_forward(guard, cache, True, 1)
        self.assertTrue(guard.step_captured)
        gs_replay._graph_forward(guard, cache, False, 2)
        self.assertFalse(guard.step_captured)

    def test_a_reading_consumes_the_capture_flag(self) -> None:
        self.overlay.log_line = lambda line: None
        allocator, cache = FakeAllocator(), {"boot": object()}
        guard = gs_replay.build_guard(self.overlay, V2_ENV, allocator, cache)
        guard.seal_boot()
        gs_replay._graph_forward(guard, cache, True, 1)
        sample = forward(captured=True)
        allocator.load(sample)
        gs_replay._prompt_forward(guard, sample)
        self.assertFalse(guard.step_captured)

    def test_uncovered_reading_consumes_the_capture_flag(self) -> None:
        self.overlay.log_line = lambda line: None
        allocator, cache = FakeAllocator(), {"boot": object()}
        guard = gs_replay.build_guard(self.overlay, V2_ENV, allocator, cache)
        guard.seal_boot()
        gs_replay._graph_forward(guard, cache, True, 1)
        sample = forward()._replace(context=262144)
        allocator.load(sample)
        gs_replay._prompt_forward(guard, sample)
        self.assertFalse(guard.step_captured)

    def test_capture_step_still_counts_down_a_running_trip(self) -> None:
        result = gs_replay.replay(self.overlay, rank_log(forward(ms=2100.0), forward(ms=2100.0, captured=True)), V2_ENV)
        self.assertEqual(result.other_trips, 1)
        self.assertEqual(sum(line.endswith(" tripwire_left=63") for line in result.lines), 1)

    def test_skip_count_is_logged(self) -> None:
        result = gs_replay.replay(self.overlay, rank_log(*[forward(captured=True)] * 65), V2_ENV)
        skips = [line for line in result.lines if "event=tripwire_capture_skip" in line]
        self.assertEqual([line.split("skips=")[1] for line in skips], ["1", "65"])

    def test_flag_off_logs_no_skip_and_samples_captures(self) -> None:
        result = gs_replay.replay(self.overlay, rank_log(*[forward(captured=True)] * 65), R0_ENV)
        self.assertFalse([line for line in result.lines if "event=tripwire_capture_skip" in line])

    def test_bad_flag_value_is_refused_with_the_fix(self) -> None:
        with self.assertRaises(self.overlay.GraphStateError) as refused:
            gs_replay.build_guard(self.overlay, {**V2_ENV, "HPU_GLM53_GS_TRIPWIRE_V2": "yes"}, FakeAllocator(), {})
        self.assertIn("HPU_GLM53_GS_TRIPWIRE_V2", str(refused.exception))

    def test_flag_without_saturated_peak_says_it_is_inert(self) -> None:
        lines: list[str] = []
        self.overlay.log_line = lines.append
        gs_replay.build_guard(self.overlay, {"HPU_GLM53_GS_TRIPWIRE_V2": "1"}, FakeAllocator(), {})
        self.assertTrue([line for line in lines if "event=inert" in line and "TRIPWIRE_V2" in line])


@unittest.skipUnless(os.environ.get(EXTRACT_ENV), f"set {EXTRACT_ENV} to a docker logs extract of a replica")
class TripwireV2Replay(unittest.TestCase):
    """Real r0 sequences: 4 ranks of SATURATION readings, rank 0's STEP_STATS for the capture flags."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.ranks = gs_replay.read_extract(Path(os.environ[EXTRACT_ENV]))
        gs_replay.share_captures(cls.ranks)
        cls.base = load_base(GUARD_FILE)
        cls.overlay = load_overlay(GUARD_FILE)

    def test_replay_is_faithful_base_reproduces_the_logged_tripwires(self) -> None:
        for rank in self.ranks.values():
            replayed = gs_replay.replay(self.base, rank, R0_ENV)
            matched = len(set(replayed.trips) & set(rank.trips))
            self.assertGreaterEqual(matched, 0.9 * len(rank.trips), f"pid {rank.pid}: {matched}/{len(rank.trips)}")

    def test_red_base_trips_on_capture_steps(self) -> None:
        totals = sum(gs_replay.replay(self.base, rank, R0_ENV).capture_trips for rank in self.ranks.values())
        self.assertGreater(totals, 0)

    def test_green_flag_on_has_no_trip_on_a_capture_step(self) -> None:
        totals = sum(gs_replay.replay(self.overlay, rank, V2_ENV).capture_trips for rank in self.ranks.values())
        self.assertEqual(totals, 0)

    def test_green_genuine_slow_non_capture_chunks_still_trip(self) -> None:
        for rank in self.ranks.values():
            self.assertGreater(gs_replay.replay(self.overlay, rank, V2_ENV).other_trips, 0, f"pid {rank.pid}")

    def test_flag_off_decisions_are_byte_identical_to_base(self) -> None:
        for rank in self.ranks.values():
            base = gs_replay.replay(self.base, rank, R0_ENV)
            overlay = gs_replay.replay(self.overlay, rank, R0_ENV)
            self.assertEqual(base.lines, overlay.lines, f"pid {rank.pid}")
            self.assertEqual(base.trips, overlay.trips, f"pid {rank.pid}")


if __name__ == "__main__":
    unittest.main()
