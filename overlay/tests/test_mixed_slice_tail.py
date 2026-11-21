"""Steps 43 and 44 on the real scheduler and KV manager: decode slice, tail-first with aging, flag-off equal to base.

Why the output cannot depend on the schedule: mixed_sched_harness.Lane reads each request's rows through the schedule output the
worker gets, raises on any row out of order, and samples from a hash of what that request alone has read. Equal tokens across every
schedule therefore mean each prompt was read whole and in order. Whether the real forward agrees is bitwise check 3 on a card.
"""
from __future__ import annotations

import ast
import json
import math
import os
import sys
import unittest
from collections.abc import Iterable
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent))

from mixed_sched_harness import BLOCK_SIZE, ROWS, SCHEDULER_PATH, Drive, StepTrace, card_seconds, prompt_tokens  # noqa: E402
from overlay_testlib import BASE_ROOT, OVERLAY_ROOT  # noqa: E402

import glm53_mixed_step as step_policy  # noqa: E402

TABLE = "1024:10,4096:22,8192:36"
SLICE = "VLLM_GAUDI_MIXED_SLICE_STEPS"
TAIL = "VLLM_GAUDI_MIXED_TAIL_FIRST"
AGE = "VLLM_GAUDI_MIXED_TAIL_AGE_S"
SECONDS = "VLLM_GAUDI_MIXED_STEP_SECONDS"
# Plan 12.2's env: the context row cap off, so every chunk keeps its full rows however deep the prompt is.
MIXED = {"VLLM_GAUDI_MIXED_STEP": "1", "VLLM_GAUDI_MIXED_PREFILL_TOKENS": str(ROWS), "VLLM_PREFILL_CONTEXT_CELL_BUDGET": "0"}
SLICE_ENV = {**MIXED, SLICE: TABLE}
TAIL_ENV = {**MIXED, TAIL: "1"}
BOTH_ENV = {**MIXED, SLICE: TABLE, TAIL: "1"}
RECORDED = Path(__file__).resolve().parent / "fixtures" / "recorded_metadata_r0.json"
AGE_LIMIT = 120.0
KEYS = "VLLM_GAUDI_MIXED_PROMPT_KEYS"
FLAGS = (SLICE, TAIL, AGE, SECONDS, KEYS)


def env(values: dict[str, str]) -> mock._patch_dict:
    cleared = {name: "" for name in FLAGS if name not in values}
    patch = mock.patch.dict(os.environ, {**cleared, **values})
    return patch


def width_of(end_tokens: int) -> int:
    return math.ceil(end_tokens / BLOCK_SIZE)


def table_value(table: Iterable[tuple[int, float]], width: int) -> float:
    rows = list(table)
    return next((value for entry, value in rows if entry >= width), rows[-1][1])


def expected_k(end_tokens: int) -> int:
    return int(table_value(((1024, 10), (4096, 22), (8192, 36)), width_of(end_tokens)))


def wall_gaps(drive: Drive, request_id: str, arrived_step: int = 0) -> list[float]:
    """Wall seconds from the arrival, or each chunk's start, to this request's next chunk's start: the run's own clock, no table."""
    last, gaps = drive.started_at[arrived_step], []
    for index, step in enumerate(drive.trace[arrived_step:], arrived_step):
        if any(chunk[0] == request_id for chunk in step.chunks):
            gaps.append(drive.started_at[index] - last)
            last = drive.started_at[index]
    return gaps


def plan_seconds(step: StepTrace) -> float:
    """A step at the plan's own readings: where the step-count arithmetic is exact, so what is left is the policy's."""
    decode, chunks = step_policy.step_seconds()
    if step.chunks:
        return max(table_value(chunks, width_of(start + rows)) for _, start, rows in step.chunks)
    return decode if step.decodes else 0.01


def chunks_of(trace: list[StepTrace], request_id: str) -> list[tuple[int, int]]:
    return [(start, rows) for step in trace for owner, start, rows in step.chunks if owner == request_id]


def assert_contiguous(case: unittest.TestCase, drive: Drive, request_id: str, first: int = 0) -> None:
    chunks = chunks_of(drive.trace, request_id)
    case.assertTrue(chunks, f"{request_id} never ran")
    case.assertEqual(chunks[0][0], first)
    for (start, rows), (next_start, _) in zip(chunks, chunks[1:]):
        case.assertEqual(start + rows, next_start, f"{request_id} left a hole or re-read rows")
    final_start, final_rows = chunks[-1]
    prompt_length = len(drive.lanes[request_id].tokens) - len(drive.lanes[request_id].sampled)
    case.assertEqual(final_start + final_rows, prompt_length, f"{request_id} did not reach its last prompt row")


def single_chunk_steps(case: unittest.TestCase, trace: list[StepTrace]) -> None:
    case.assertLessEqual(max(len(step.chunks) for step in trace), 1, "a step carried two prompts")


class Flags(unittest.TestCase):
    def test_empty_means_off(self) -> None:
        with env(MIXED):
            self.assertEqual(step_policy.slice_table(), ())
            self.assertFalse(step_policy.tail_first_enabled())
            self.assertFalse(step_policy.slot_policy_active())

    def test_table_parses_sorted_and_turns_the_policy_on(self) -> None:
        with env({SLICE: "8192:36,1024:10,4096:22"}):
            self.assertEqual(step_policy.slice_table(), ((1024, 10), (4096, 22), (8192, 36)))
            self.assertTrue(step_policy.slot_policy_active())

    def test_bad_tables_refuse_and_name_the_flag(self) -> None:
        for spec in ("1024", "1024:x", "1000:10", "1024:-1", "1024:1,1024:2", "0:3"):
            with env({SLICE: spec}), self.subTest(spec=spec):
                with self.assertRaisesRegex(ValueError, SLICE):
                    step_policy.slice_table()

    def test_bad_switch_refuses(self) -> None:
        with env({TAIL: "yes"}), self.assertRaisesRegex(ValueError, TAIL):
            step_policy.tail_first_enabled()

    def test_an_age_that_never_ages_refuses_at_boot(self) -> None:
        """NaN compares false with everything, so it used to pass the boot check and silently turn aging off."""
        for age in ("nan", "NaN", "inf", "-inf", "0", "-5", "soon"):
            with env({**BOTH_ENV, AGE: age}), self.subTest(age=age):
                with self.assertRaisesRegex(ValueError, AGE):
                    step_policy.check_slot_flags(completion_first=False)

    def test_policy_without_the_mixed_step_refuses_at_boot(self) -> None:
        for values in ({SLICE: TABLE}, {TAIL: "1"}):
            with env({"VLLM_GAUDI_MIXED_STEP": "0", **values}), self.subTest(values=values):
                with self.assertRaisesRegex(ValueError, "VLLM_GAUDI_MIXED_STEP"):
                    step_policy.check_slot_flags(completion_first=False)

    def test_tail_first_refuses_the_completion_first_lane(self) -> None:
        with env(TAIL_ENV), self.assertRaisesRegex(ValueError, "VLLM_COMPLETION_FIRST_LONG_PREFILL"):
            step_policy.check_slot_flags(completion_first=True)

    def test_age_limit_must_outlast_the_widest_cycle(self) -> None:
        with env({**BOTH_ENV, AGE: "5"}), self.assertRaisesRegex(ValueError, AGE):
            step_policy.check_slot_flags(completion_first=False)

    def test_scheduler_runs_the_boot_check(self) -> None:
        with env({"VLLM_GAUDI_MIXED_STEP": "0", SLICE: TABLE}), self.assertRaisesRegex(ValueError, "VLLM_GAUDI_MIXED_STEP"):
            Drive(num_blocks=100)


class SlicePolicy(unittest.TestCase):
    def test_k_withheld_then_allowed_and_a_chunk_rearms_it(self) -> None:
        owner = SimpleNamespace()
        with env({SLICE: TABLE}):
            self.assertEqual([step_policy.decode_slice_allows(owner, 1024) for _ in range(12)], [False] * 10 + [True] * 2)
            step_policy.note_chunk_ran(owner)
            self.assertFalse(step_policy.decode_slice_allows(owner, 1024))

    def test_width_picks_the_narrowest_entry_and_the_top_entry_past_it(self) -> None:
        for width, steps in ((1, 10), (1024, 10), (1025, 22), (8192, 36), (70_000, 36)):
            with env({SLICE: TABLE}), self.subTest(width=width):
                owner = SimpleNamespace()
                allowed = [step_policy.decode_slice_allows(owner, width) for _ in range(steps + 1)]
                self.assertEqual(allowed, [False] * steps + [True])

    def test_unknown_width_withholds_only_below_every_entry(self) -> None:
        owner = SimpleNamespace()
        with env({SLICE: "1024:2,8192:5"}):
            self.assertEqual([step_policy.decode_slice_allows(owner, None) for _ in range(3)], [False, False, True])
            self.assertFalse(step_policy.decode_slice_allows(owner, 8192))

    def test_off_always_allows_and_keeps_no_state(self) -> None:
        owner = SimpleNamespace()
        with env({SLICE: ""}):
            self.assertTrue(step_policy.decode_slice_allows(owner, 8192))
        self.assertEqual(vars(owner), {})


class DecodeSliceOnScheduler(unittest.TestCase):
    def lane_decoding(self, drive: Drive, count: int = 1) -> None:
        for index in range(count):
            drive.add(f"lane{index}", prompt_tokens(100 + index, 24), max_tokens=100_000)
        drive.run(2)
        while len(drive.trace[-1].decodes) < count:  # one prompt a step, K steps apart once a lane decodes
            drive.step()
        drive.step()
        self.assertEqual(drive.trace[-1].kind, "decode")

    def test_recorded_metadata_replays_k_decode_steps_then_one_mixed(self) -> None:
        rows = json.loads(RECORDED.read_text(encoding="utf-8"))
        prompts = [row for row in rows if row["kind"] == "prompt" and row["start"] >= ROWS and row["start"] % BLOCK_SIZE == 0
                   and 0 < row["rows"] <= ROWS]
        self.assertGreater(len(prompts), 400, "the fixture is the r0 recording; a short one proves nothing")
        tokens = prompt_tokens(7, max(row["start"] for row in prompts) + 1)
        seen_widths = set()
        with env(SLICE_ENV):
            drive = Drive(num_blocks=4000)
            drive.add("seed", tokens)
            drive.run_until_idle(400)  # no decode lane: every step a chunk, and every recorded start is now a prefix hit
            self.lane_decoding(drive)
            for index, row in enumerate(prompts[::8]):
                start, count = row["start"], row["rows"]
                name = f"replay{index}"
                drive.add(name, tokens[:start] + prompt_tokens(1000 + index, count))  # a hit to the recorded start, cold after it
                steps = [drive.step() for _ in range(expected_k(start + count) + 1)]
                self.assertEqual([step.kind for step in steps], ["decode"] * expected_k(start + count) + ["mixed"], row)
                self.assertEqual(steps[-1].chunks, ((name, start, count),), row)
                seen_widths.add(expected_k(start + count))
        self.assertEqual(seen_widths, {10, 22}, "the recording covers W=1024 and W=4096")

    def test_a_long_prompt_cycles_k_then_one_for_its_whole_life(self) -> None:
        with env(SLICE_ENV):
            drive = Drive()
            self.lane_decoding(drive)
            drive.add("long", prompt_tokens(1, 6 * ROWS))
            mark = len(drive.trace)
            drive.run(6 * 11)
        kinds = [step.kind for step in drive.trace[mark:]]
        self.assertEqual(kinds, (["decode"] * 10 + ["mixed"]) * 6)
        assert_contiguous(self, drive, "long")

    def test_k_is_zero_without_a_decode_lane(self) -> None:
        with env(SLICE_ENV):
            drive = Drive()
            drive.add("long", prompt_tokens(1, 6 * ROWS))
            drive.run(6)
        self.assertEqual([step.kind for step in drive.trace], ["prompt"] * 6)

    def test_a_parked_decode_is_not_a_decode_lane(self) -> None:
        """A step of only parked decodes would be empty: it must not hold a prompt back."""
        with env(SLICE_ENV):
            drive = Drive()
            self.lane_decoding(drive)
            drive.scheduler.requests["lane0"].next_decode_eligible_step = 10 ** 9
            drive.add("long", prompt_tokens(1, 4 * ROWS))
            drive.run(4)
        self.assertEqual([step.kind for step in drive.trace[-4:]], ["prompt"] * 4)

    def test_k_drops_to_zero_the_step_the_last_lane_finishes(self) -> None:
        with env(SLICE_ENV):
            drive = Drive()
            drive.add("lane0", prompt_tokens(100, 24), max_tokens=5)
            drive.run(2)
            drive.add("long", prompt_tokens(1, 4 * ROWS))
            drive.run(8)
        self.assertEqual([step.kind for step in drive.trace[:9]],
                         ["prompt"] + ["decode"] * 4 + ["prompt"] * 4)
        self.assertEqual(drive.finished_at["lane0"], 4)

    def test_withheld_steps_keep_every_decode_and_the_fifo_order(self) -> None:
        with env(SLICE_ENV):
            drive = Drive()
            self.lane_decoding(drive, count=3)
            for name, seed in (("first", 1), ("second", 2)):
                drive.add(name, prompt_tokens(seed, 2 * ROWS))
            mark = len(drive.trace)
            drive.run(33)
        for step in drive.trace[mark:]:
            self.assertEqual(len(step.decodes), 3)
        self.assertEqual([step.chunks[0][:2] for step in drive.trace[mark:] if step.chunks],
                         [("first", 0), ("first", ROWS), ("second", 0)])


class TailFirstOnScheduler(unittest.TestCase):
    def test_tails_pass_a_running_long_prompt(self) -> None:
        with env(TAIL_ENV):
            drive = Drive()
            drive.add("long", prompt_tokens(1, 40 * ROWS))
            drive.run(3)
            drive.add("tail2", prompt_tokens(2, 2 * ROWS))
            drive.add("tail1", prompt_tokens(3, ROWS - 300))
            drive.run_until_idle(100)
        order = sorted(drive.finished_at, key=drive.finished_at.get)
        self.assertEqual(order, ["tail1", "tail2", "long"])
        self.assertEqual(drive.finished_at["tail1"], 3)
        for name in ("long", "tail1", "tail2"):
            assert_contiguous(self, drive, name)
        single_chunk_steps(self, drive.trace)

    def test_a_local_cache_hit_counts_as_a_tail(self) -> None:
        """A returning lane with 100K cached and a short new tail goes before a cold prompt half its size."""
        cached = prompt_tokens(5, 20 * ROWS)
        with env(TAIL_ENV):
            drive = Drive()
            drive.add("seed", cached)
            drive.run_until_idle(40)
            drive.add("cold", prompt_tokens(6, 8 * ROWS))
            drive.add("returning", cached + prompt_tokens(7, ROWS - 200))
            mark = len(drive.trace)
            drive.run(2)
        owner, start, rows = drive.trace[mark].chunks[0]
        self.assertEqual(owner, "returning")
        self.assertEqual(start, 20 * ROWS)
        self.assertEqual(rows, ROWS - 200)

    def test_ties_stay_fifo(self) -> None:
        with env(TAIL_ENV):
            drive = Drive()
            for name, seed in (("a", 1), ("b", 2), ("c", 3)):
                drive.add(name, prompt_tokens(seed, ROWS))
            drive.run_until_idle(6)
        self.assertEqual([step.chunks[0][0] for step in drive.trace], ["a", "b", "c"])

    def test_one_8192_chunk_prompt_plus_tails(self) -> None:
        """The accept case, with tails arriving for long enough that only aging can serve the long prompt."""
        with env(BOTH_ENV):
            drive = Drive(num_blocks=70_000)
            drive.add("lane", prompt_tokens(100, 24), max_tokens=2500)
            drive.run(2)
            arrived = len(drive.trace)
            drive.add("long", prompt_tokens(1, 8192 * ROWS))
            tails = self.feed_tails(drive, count=40)
            drive.run_until_idle(20_000)
        long_chunks = chunks_of(drive.trace, "long")
        self.assertEqual(len(long_chunks), 8192)
        first_tail_done = max(drive.finished_at[name] for name in tails)
        self.assertLess(first_tail_done, drive.finished_at["long"])
        before_tails_end = [step for step in drive.trace[:first_tail_done] for chunk in step.chunks if chunk[0] == "long"]
        self.assertGreater(len(before_tails_end), 0, "tails alone must not starve the long prompt")
        self.assertLess(len(before_tails_end), 40, "the long prompt gets the aged slot, not the tail's")
        self.assertLessEqual(max(wall_gaps(drive, "long", arrived)), AGE_LIMIT)
        assert_contiguous(self, drive, "long")
        for name in tails:
            assert_contiguous(self, drive, name)
            self.assertLessEqual(max(wall_gaps(drive, name, self.arrivals[name])), AGE_LIMIT)
        single_chunk_steps(self, drive.trace)

    def feed_tails(self, drive: Drive, count: int) -> list[str]:
        """Four-chunk tails, one new one whenever the last has finished, until ``count`` have run."""
        self.arrivals: dict[str, int] = {}
        names: list[str] = []
        for index in range(count):
            name = f"tail{index}"
            self.arrivals[name] = len(drive.trace)
            names.append(name)
            drive.add(name, prompt_tokens(200 + index, 4 * ROWS))
            while name not in drive.finished_at:
                drive.step()
        return names

    def test_two_long_prompts_never_starve_each_other(self) -> None:
        with env(BOTH_ENV):
            drive = Drive(num_blocks=8000)
            drive.add("lane", prompt_tokens(100, 24), max_tokens=100_000)
            drive.run(2)
            arrived = len(drive.trace)
            drive.add("big", prompt_tokens(1, 300 * ROWS))
            drive.add("bigger", prompt_tokens(2, 400 * ROWS))
            drive.run(6000)
        self.assertLessEqual(max(wall_gaps(drive, "big", arrived)), AGE_LIMIT)
        bigger = wall_gaps(drive, "bigger", arrived)
        self.assertGreater(len(bigger), 5, "aging must reach the longer prompt")
        self.assertLessEqual(max(bigger), AGE_LIMIT)

    def test_age_limit_is_an_env_choice_and_replays_exactly(self) -> None:
        traces = []
        for _ in range(2):
            with env({**BOTH_ENV, AGE: "60"}):
                drive = Drive(num_blocks=8000)
                drive.add("lane", prompt_tokens(100, 24), max_tokens=100_000)
                drive.run(2)
                drive.add("big", prompt_tokens(1, 300 * ROWS))
                drive.add("bigger", prompt_tokens(2, 400 * ROWS))
                drive.run(2000)
            traces.append(drive)
        self.assertEqual(traces[0].trace, traces[1].trace, "the same clock readings must give the same schedule")
        self.assertLessEqual(max(wall_gaps(traces[0], "bigger", 2)), 60.0)

    def test_a_hint_mismatch_requeues_fifo(self) -> None:
        """The hint said tail, the real lookup says cold: it stays in its FIFO place and nothing passes on the stale hint this step."""
        with env(TAIL_ENV):
            drive = Drive()
            drive.add("first", prompt_tokens(1, 6 * ROWS))
            drive.add("liar", prompt_tokens(2, 20 * ROWS))
            drive.add("third", prompt_tokens(3, 8 * ROWS))
            real = drive.scheduler._glm53_estimated_start
            hinted = lambda request: 19 * ROWS if request.request_id == "liar" else real(request)  # noqa: E731
            with mock.patch.object(drive.scheduler, "_glm53_estimated_start", side_effect=hinted):
                queue = [request.request_id for request in drive.scheduler.waiting]
                self.assertEqual(queue, ["first", "liar", "third"])
                first = drive.step()
                waiting = [request.request_id for queue in (drive.scheduler.skipped_waiting, drive.scheduler.waiting)
                           for request in queue]
                second = drive.step()
        self.assertEqual(first.kind, "idle", "no running chunk yielded, so this step simply carries no prompt")
        self.assertEqual(waiting, ["first", "liar", "third"], "FIFO order survives the requeue")
        self.assertEqual(second.chunks[0][:2], ("first", 0), "the real lookup replaced the hint: first has the fewest left")
        self.assertEqual(drive.scheduler._glm53_slot_ledger.mismatches, 1)

    def test_a_hint_mismatch_that_is_still_the_shortest_runs(self) -> None:
        with env(TAIL_ENV):
            drive = Drive()
            drive.add("liar", prompt_tokens(2, 3 * ROWS))
            drive.add("long", prompt_tokens(3, 8 * ROWS))
            real = drive.scheduler._glm53_estimated_start
            hinted = lambda request: 2 * ROWS if request.request_id == "liar" else real(request)  # noqa: E731
            with mock.patch.object(drive.scheduler, "_glm53_estimated_start", side_effect=hinted):
                step = drive.step()
        self.assertEqual(step.chunks[0][:2], ("liar", 0))
        self.assertEqual(drive.scheduler._glm53_slot_ledger.mismatches, 0)

    def test_estimate_reads_the_cache_and_the_connector_hint(self) -> None:
        cached = prompt_tokens(5, 4 * ROWS)
        with env(TAIL_ENV):
            drive = Drive()
            drive.add("seed", cached)
            drive.run_until_idle(10)
            drive.add("returning", cached + prompt_tokens(6, 2 * ROWS))
            request = drive.scheduler.waiting.peek_request()
            before = drive.scheduler.kv_cache_manager.block_pool.get_num_free_blocks()
            self.assertEqual(drive.scheduler._glm53_estimated_start(request), 4 * ROWS)
            drive.scheduler.connector = SimpleNamespace(estimate_exact_hit_tokens=lambda _: 5 * ROWS)
            self.assertEqual(drive.scheduler._glm53_estimated_start(request), 5 * ROWS)
            drive.scheduler.connector = None
        self.assertEqual(drive.scheduler.kv_cache_manager.block_pool.get_num_free_blocks(), before, "the estimate is read-only")
        self.assertEqual(request.num_computed_tokens, 0)

    def test_a_picked_waiter_that_cannot_run_gives_the_slot_back(self) -> None:
        """RC5.3's retry: the running chunk that yielded runs in the same step, so no slot is lost."""
        with env(TAIL_ENV):
            drive = Drive()
            drive.add("long", prompt_tokens(1, 10 * ROWS))
            drive.run(1)
            drive.add("tail", prompt_tokens(2, ROWS))
            manager = drive.scheduler.kv_cache_manager
            real = manager.allocate_slots

            def refuse_tail(request: object, *args: object, **kwargs: object) -> object:
                return None if request.request_id == "tail" else real(request, *args, **kwargs)

            with mock.patch.object(manager, "allocate_slots", side_effect=refuse_tail):
                step = drive.step()
        self.assertEqual(step.chunks, (("long", ROWS, ROWS),))

    def test_prompts_that_would_overrun_kv_finish_one_at_a_time(self) -> None:
        """Mixed prefill never preempts, so two partial prompts that cannot both fit would never progress again.
        A short age limit alternates them, the worst case for that: without the reservation this ends in "no progress"."""
        with env({**TAIL_ENV, AGE: "4"}):
            drive = Drive(num_blocks=500)
            drive.add("long", prompt_tokens(1, 40 * ROWS))
            drive.run(2)
            drive.add("shorter", prompt_tokens(2, 39 * ROWS))
            mark = len(drive.trace)
            drive.run_until_idle(200)
        self.assertEqual(sorted(drive.finished_at, key=drive.finished_at.get), ["long", "shorter"])
        self.assertTrue(all(step.chunks[0][0] == "long" for step in drive.trace[mark:mark + 38]))
        assert_contiguous(self, drive, "long")
        assert_contiguous(self, drive, "shorter")

    def test_an_aged_waiter_that_does_not_fit_stops_new_waiters_until_the_partials_drain(self) -> None:
        """R1. It must get the pool the partials free, so a tail that would fit today waits: only then does the age limit mean anything."""
        with env({**TAIL_ENV, AGE: "4"}):
            drive = Drive(num_blocks=500)
            drive.add("long", prompt_tokens(1, 40 * ROWS))
            drive.run(2)
            drive.add("too_big", prompt_tokens(2, 39 * ROWS))
            drive.run(5)  # aged by now, and still no room beside the partial prompt
            drive.add("tail", prompt_tokens(3, 2 * ROWS))
            step = drive.step()
            self.assertEqual(step.chunks[0][0], "long", "no new waiter passes the aged one")
            drive.run_until_idle(200)
        self.assertLess(drive.finished_at["long"], drive.first_token_at["too_big"])
        self.assertLess(drive.finished_at["long"], drive.first_token_at["tail"])
        chunk_owners = [chunk[0] for step in drive.trace for chunk in step.chunks]
        self.assertLess(chunk_owners.index("too_big"), chunk_owners.index("tail"), "the aged prompt goes first once there is room")
        for name in ("long", "too_big", "tail"):
            assert_contiguous(self, drive, name)

    def test_a_stream_of_fitting_tails_cannot_starve_an_aged_waiter_past_the_limit(self) -> None:
        """R1's reproduction: 500 blocks, 30 s limit, a long partial that short tails keep passing, so no room for ``big`` opens for many
        multiples of the limit, and nothing logged. The partials cannot be preempted, so draining them is the shortest wait there is."""
        with env({**TAIL_ENV, AGE: "30"}):
            drive = Drive(num_blocks=500)
            drive.add("partial", prompt_tokens(50, 40 * ROWS))
            drive.run(1)
            arrived = len(drive.trace)
            drive.add("big", prompt_tokens(1, 300 * BLOCK_SIZE))
            next_tail = 0
            with self.assertLogs(sys.modules[type(drive.scheduler).__module__].logger, level="INFO") as logged:
                while drive.clock.now < 1000.0 + 600.0:
                    if sum(name.startswith("t") for name in drive.scheduler.requests) == 0:
                        drive.add(f"t{next_tail}", prompt_tokens(100 + next_tail, 2 * ROWS))
                        next_tail += 1
                    drive.step()
        self.assertGreater(next_tail, 1, "the stream must run past the hold for this to prove anything")
        self.assertTrue(chunks_of(drive.trace, "big"), "big starved past the age limit")
        self.assertTrue(any("mixed_tail_first_aged_hold" in line for line in logged.output), "a hold must say so")
        drain = 40 * 1.5 + 2 * 1.5  # at most the partial's whole length and one tail in flight, at the harness card's W=1024 chunk
        self.assertLessEqual(wall_gaps(drive, "big", arrived)[0], 30.0 + drain)
        self.assertEqual(drive.restarted, [], "the hold never needs a preemption")

    def test_an_aged_waiter_with_no_room_only_holds_while_a_partial_can_drain(self) -> None:
        """No partial left to free blocks: nothing to wait for, so it is picked at once, as any aged prompt is."""
        with env({**TAIL_ENV, AGE: "4"}):
            drive = Drive(num_blocks=500)
            drive.add("big", prompt_tokens(1, 58 * ROWS))
            drive.add("tail", prompt_tokens(2, 2 * ROWS))
            drive.run_until_idle(200)
        self.assertEqual(set(drive.finished_at), {"big", "tail"})
        self.assertEqual(drive.restarted, [])

    def test_a_waiter_that_fits_beside_the_reservation_still_passes(self) -> None:
        with env(TAIL_ENV):
            drive = Drive(num_blocks=500)
            drive.add("long", prompt_tokens(1, 40 * ROWS))
            drive.run(2)
            drive.add("tail", prompt_tokens(2, 3 * ROWS))
            step = drive.step()
        self.assertEqual(step.chunks[0][:2], ("tail", 0))


class OnePromptPerStep(unittest.TestCase):
    """B1: prompts that fit the budget together still get one forward each only if the pool closes; only tail-first closed it."""

    def three_short_prompts(self, values: dict[str, str]) -> Drive:
        with env(values):
            drive = Drive()
            drive.add("lane", prompt_tokens(100, 24), max_tokens=500)
            drive.run(2)
            for name, seed in (("a", 1), ("b", 2), ("c", 3)):
                drive.add(name, prompt_tokens(seed, 300))
            drive.run(120)
        return drive

    def test_a_slice_only_step_carries_one_prompt(self) -> None:
        drive = self.three_short_prompts(SLICE_ENV)
        single_chunk_steps(self, drive.trace)
        self.assertEqual({"a", "b", "c"} & set(drive.finished_at), {"a", "b", "c"})

    def test_a_keys_only_step_carries_one_prompt(self) -> None:
        drive = self.three_short_prompts({**MIXED, KEYS: "1024:1024"})
        single_chunk_steps(self, drive.trace)
        self.assertEqual({"a", "b", "c"} & set(drive.finished_at), {"a", "b", "c"})

    def test_no_policy_and_no_keys_keep_bases_several_prompts(self) -> None:
        drive = self.three_short_prompts(MIXED)
        self.assertEqual(max(len(step.chunks) for step in drive.trace), 3, "flag off equals base")

    def test_the_budget_is_one_chunks_when_the_pool_closes(self) -> None:
        with env({**MIXED, KEYS: "1024:1024"}):
            owner = SimpleNamespace()
            self.assertEqual(step_policy.single_prompt_remaining(owner, 700, {"a": 324}, {"a"}), 0)
        with env(MIXED):
            self.assertEqual(step_policy.single_prompt_remaining(SimpleNamespace(), 700, {"a": 324}, {"a"}), 700)


class WallClockAging(unittest.TestCase):
    """R2: 120 s is a promise in real time, and K stays a step count."""

    @staticmethod
    def contention(values: dict[str, str], seconds: object = card_seconds) -> Drive:
        with env(values):
            drive = Drive(num_blocks=8000, seconds=seconds)
            drive.add("lane", prompt_tokens(100, 24), max_tokens=100_000)
            drive.run(2)
            drive.add("big", prompt_tokens(1, 100 * ROWS), max_tokens=3)
            drive.add("bigger", prompt_tokens(2, 140 * ROWS), max_tokens=3)
            drive.run(3500)
        return drive

    def test_a_card_five_times_slower_than_the_plan_still_ages_in_real_seconds(self) -> None:
        drive = self.contention(BOTH_ENV, lambda step: 5 * card_seconds(step))
        self.assertGreater(len(chunks_of(drive.trace, "bigger")), 3)
        self.assertLessEqual(max(wall_gaps(drive, "bigger", 2)), AGE_LIMIT)
        self.assertLessEqual(max(wall_gaps(drive, "big", 2)), AGE_LIMIT)

    def test_k_stays_a_step_count_at_any_card_speed(self) -> None:
        kinds = []
        for scale in (1, 7):
            with env(SLICE_ENV):
                drive = Drive(seconds=lambda step, factor=scale: factor * card_seconds(step))
                drive.add("lane", prompt_tokens(100, 24), max_tokens=100_000)
                drive.run(2)
                drive.add("long", prompt_tokens(1, 4 * ROWS))
                drive.run(44)
            kinds.append([step.kind for step in drive.trace])
        self.assertEqual(kinds[0], kinds[1])

    def test_a_clock_that_changes_the_schedule_cannot_change_a_token(self) -> None:
        """Why bitwise check 3 stays valid: wall clock aging moves chunks between steps, but each lane reads the same rows in the same
        order, and its output is a function of those rows alone. Check 3 compares tokens, never the schedule's order."""
        fast, slow = (self.contention(BOTH_ENV, lambda step, factor=scale: factor * card_seconds(step)) for scale in (1, 6))
        self.assertNotEqual(fast.trace, slow.trace, "the two clocks must really give two schedules for this to prove anything")
        self.assertEqual(fast.lanes["big"].sampled[:3], slow.lanes["big"].sampled[:3])
        for name in ("big", "bigger"):
            shared = min(fast.lanes[name].computed, slow.lanes[name].computed)
            if shared == len(prompt_tokens(1, 100 * ROWS)):
                self.assertEqual(fast.lanes[name].state, slow.lanes[name].state)
        self.assertEqual(fast.lanes["lane"].tokens[:24], slow.lanes["lane"].tokens[:24])

    def test_the_same_clock_replays_the_same_schedule(self) -> None:
        self.assertEqual(self.contention(BOTH_ENV).trace, self.contention(BOTH_ENV).trace)

    def test_the_ledger_reads_only_its_clock_and_a_chunk_resets_the_age(self) -> None:
        now = [10.0]
        with env(TAIL_ENV):
            ledger = step_policy.SlotLedger(clock=lambda: now[0])
            ledger.observe(["a", "b"])
            now[0] += 200.0
            ledger.observe(["a", "b"])
            pick = ledger.pick([step_policy.PromptCandidate("a", 0, 9000), step_policy.PromptCandidate("b", 0, 900)], ROWS, BLOCK_SIZE)
            self.assertEqual((pick.request_id, pick.aged), ("a", True), "both aged: the earliest deadline, FIFO on the tie")
            ledger.note_chunks({"a"})
            self.assertEqual(ledger.last_chunk["a"], 210.0)
            self.assertEqual(ledger.last_chunk["b"], 10.0)

    def test_a_recent_slow_cycle_widens_the_margin_the_table_would_not(self) -> None:
        """A card slower than the table must not aim for the limit with a table's margin."""
        now = [0.0]
        with env(TAIL_ENV):
            ledger = step_policy.SlotLedger(clock=lambda: now[0])
            ledger.observe(["a"])
            for _ in range(3):
                now[0] += 50.0
                ledger.observe(["a", "b"])
                ledger.note_chunks({"a"})
            self.assertGreaterEqual(ledger.age_margin(1), 50.0)


class AgedQueueFitsTheLimit(unittest.TestCase):
    """The NIT: at 16 aged prompts, one slot per K + 1 steps went round in about 123 s, past the limit."""

    def test_sixteen_prompts_at_the_widest_key_stay_inside_the_limit(self) -> None:
        wide = prompt_tokens(5, 4100 * BLOCK_SIZE)  # table width 4,100: the 8192 entry, K = 36
        with env(BOTH_ENV):
            drive = Drive(num_blocks=70_000, seconds=plan_seconds)
            drive.add("seed", wide)
            drive.run_until_idle(600)
            drive.add("lane", prompt_tokens(100, 24), max_tokens=100_000)
            drive.run(2)
            arrived = len(drive.trace)
            names = [f"p{index}" for index in range(16)]
            for index, name in enumerate(names):
                drive.add(name, wide + prompt_tokens(200 + index, 80 * ROWS))
            drive.run(9000)
        for name in names:
            with self.subTest(prompt=name):
                self.assertGreater(len(chunks_of(drive.trace, name)), 3)
                self.assertLessEqual(max(wall_gaps(drive, name, arrived)), AGE_LIMIT)

    def test_k_shrinks_only_as_far_as_the_queue_needs(self) -> None:
        with env(BOTH_ENV):
            self.assertEqual(step_policy.slice_steps(8192, 1), 36)
            self.assertEqual(step_policy.slice_steps(8192, 15), 36)
            self.assertEqual(step_policy.slice_steps(8192, 16), 34)
            self.assertEqual(step_policy.slice_steps(8192, 64), 0)
        with env(SLICE_ENV):
            self.assertEqual(step_policy.slice_steps(8192, 64), 36, "no aging, no promise to keep")


class PreemptedPartials(unittest.TestCase):
    """R10: a decode that needs a block preempts the last running request, which may be a partial prompt."""

    def preempting(self, values: dict[str, str], *, preempt: bool) -> Drive:
        with env(values):
            drive = Drive(num_blocks=4000)
            drive.add("lane", prompt_tokens(100, 24), max_tokens=900)
            drive.run(2)
            drive.add("partial", prompt_tokens(1, 30 * ROWS), max_tokens=4)
            if preempt:
                self.preempt_the_partial_once(drive)
            drive.run_until_idle(6000)
        return drive

    @staticmethod
    def preempt_the_partial_once(drive: Drive) -> None:
        """The lane's next block fails once, on any schedule: the real preemption path then frees the partial prompt, the last one running."""
        while len(chunks_of(drive.trace, "partial")) < 5:
            drive.step()
        manager = drive.scheduler.kv_cache_manager
        real, refused = manager.allocate_slots, []

        def refuse_the_lane_once(request: object, *args: object, **kwargs: object) -> object:
            if request.request_id == "lane" and not refused:
                refused.append(request.request_id)
                return None
            return real(request, *args, **kwargs)

        with mock.patch.object(manager, "allocate_slots", side_effect=refuse_the_lane_once):
            drive.step()
        # The freed rows leave the prefix cache too, as under KV pressure: the resume must read from row 0, not hit the rows it just computed.
        manager.evict_blocks({block.block_id for block in manager.block_pool.blocks if block.ref_cnt == 0})

    def test_a_restart_is_legitimate_and_any_other_reread_still_raises(self) -> None:
        with env(MIXED):
            drive = Drive(num_blocks=100)
            drive.add("lane", prompt_tokens(1, 100))
        lane = drive.lanes["lane"]
        lane.read(0, 20)
        with self.assertRaisesRegex(AssertionError, "did not prefill in order"):
            lane.read(10, 5)
        lane.preempted()
        lane.read(0, 20)
        self.assertEqual(lane.restarts, 1)

    def test_a_preempted_partial_reads_every_row_again_and_samples_the_same_tokens(self) -> None:
        for name, values in (("mixed", MIXED), ("tail", TAIL_ENV), ("slice", SLICE_ENV), ("both", BOTH_ENV)):
            with self.subTest(schedule=name):
                preempted, undisturbed = self.preempting(values, preempt=True), self.preempting(values, preempt=False)
                self.assertEqual(preempted.restarted, ["partial"], "the case needs a real preemption of the partial prompt")
                self.assertEqual(undisturbed.restarted, [])
                starts = [start for start, _ in preempted.lanes["partial"].chunks]
                self.assertEqual(starts.count(0), 2, f"the partial must read from row 0 again: {starts[:12]}")
                self.assertEqual(preempted.lanes["partial"].sampled, undisturbed.lanes["partial"].sampled)
                self.assertEqual(preempted.lanes["partial"].state, undisturbed.lanes["partial"].state)
                self.assertEqual(preempted.lanes["lane"].sampled, undisturbed.lanes["lane"].sampled)
                self.assertEqual(sorted(preempted.finished_at), ["lane", "partial"])

    def test_a_preempted_partial_keeps_its_age(self) -> None:
        drive = self.preempting(TAIL_ENV, preempt=True)
        self.assertEqual(drive.restarted, ["partial"])
        self.assertEqual([start for start, _ in drive.lanes["partial"].chunks].count(0), 2, "the partial must read from row 0 again")
        self.assertLessEqual(max(wall_gaps(drive, "partial", 2)), AGE_LIMIT)


class ScheduleIndependence(unittest.TestCase):
    def workload(self, values: dict[str, str]) -> Drive:
        cached = prompt_tokens(5, 6 * ROWS)
        with env(values):
            drive = Drive(num_blocks=3000)
            drive.add("seed", cached)
            drive.run_until_idle(20)
            drive.add("greedy", prompt_tokens(1, 2 * ROWS + 37), max_tokens=60)
            drive.run(3)
            drive.add("long", prompt_tokens(2, 30 * ROWS + 5), max_tokens=5)
            drive.add("tail", prompt_tokens(3, 300), max_tokens=5)
            drive.add("returning", cached + prompt_tokens(4, 700), max_tokens=5)
            drive.run_until_idle(4000)
        return drive

    def test_every_schedule_reads_the_same_rows_and_samples_the_same_tokens(self) -> None:
        schedules = {"k0_fifo": MIXED, "slice_fifo": SLICE_ENV, "k0_tail": TAIL_ENV, "slice_tail": BOTH_ENV}
        runs = {name: self.workload(values) for name, values in schedules.items()}
        reference = runs["k0_fifo"]
        for name, drive in runs.items():
            for lane, state in drive.lanes.items():
                with self.subTest(schedule=name, lane=lane):
                    self.assertEqual(state.sampled, reference.lanes[lane].sampled)
                    self.assertEqual(state.state, reference.lanes[lane].state)
        orders = {name: tuple(sorted(drive.finished_at, key=drive.finished_at.get)) for name, drive in runs.items()}
        self.assertGreater(len(set(orders.values())), 1, "the schedules must really differ for this to prove anything")
        for name, drive in runs.items():
            assert_contiguous(self, drive, "long")
            assert_contiguous(self, drive, "returning", first=6 * ROWS)

    def test_the_fake_model_catches_a_row_out_of_order(self) -> None:
        with env(MIXED):
            drive = Drive(num_blocks=100)
            drive.add("lane", prompt_tokens(1, 100))
        with self.assertRaisesRegex(AssertionError, "did not prefill in order"):
            drive.lanes["lane"].computed = 50
            drive.lanes["lane"].read(10, 5)


def _definitions(path: Path) -> dict[str, str]:
    found: dict[str, str] = {}

    def walk(node: ast.AST, prefix: str) -> None:
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                name = f"{prefix}{child.name}"
                if not isinstance(child, ast.ClassDef):
                    found[name] = ast.dump(child)
                walk(child, name + ".")
            else:
                walk(child, prefix)

    walk(ast.parse(path.read_text(encoding="utf-8")), "")
    return found


class FlagOffEqualsBase(unittest.TestCase):
    def workload(self, *, base: bool, values: dict[str, str]) -> Drive:
        cached = prompt_tokens(5, 4 * ROWS)
        with env(values):
            drive = Drive(base=base, num_blocks=1500)
            drive.add("seed", cached)
            drive.run_until_idle(30)
            drive.add("lane", prompt_tokens(1, 1500), max_tokens=40)
            drive.run(4)
            drive.add("long", prompt_tokens(2, 12 * ROWS + 11), max_tokens=3)
            drive.add("tail", prompt_tokens(3, 700), max_tokens=3)
            drive.run(3)
            drive.add("returning", cached + prompt_tokens(4, 500), max_tokens=3)
            drive.add("overrun", prompt_tokens(6, 140 * ROWS), max_tokens=3)
            drive.run_until_idle(1000)
        return drive

    def test_same_decisions_as_base_on_every_mode(self) -> None:
        modes = {
            "mixed": MIXED,
            "mixed_context_row_cap": {"VLLM_GAUDI_MIXED_STEP": "1", "VLLM_GAUDI_MIXED_PREFILL_TOKENS": str(ROWS)},
            "mixed_single_prompt": {**MIXED, "VLLM_GAUDI_MIXED_GRAPHS": "1", "VLLM_GAUDI_MIXED_GRAPH_SINGLE_PROMPT": "1"},
            "not_mixed": {"VLLM_GAUDI_MIXED_STEP": "0"},
        }
        for name, values in modes.items():
            with self.subTest(mode=name):
                base, overlay = self.workload(base=True, values=values), self.workload(base=False, values=values)
                self.assertEqual(overlay.trace, base.trace)
                self.assertEqual({lane: state.sampled for lane, state in overlay.lanes.items()},
                                 {lane: state.sampled for lane, state in base.lanes.items()})
                self.assertGreater(len(base.trace), 30)

    def test_policy_off_keeps_no_state_on_the_scheduler(self) -> None:
        drive = self.workload(base=False, values=MIXED)
        self.assertFalse(hasattr(drive.scheduler, "_glm53_slot_ledger"))
        self.assertFalse(hasattr(drive.scheduler, "_glm53_slice_withheld"))

    def test_only_schedule_init_and_new_helpers_differ_from_base(self) -> None:
        base, overlay = _definitions(BASE_ROOT / SCHEDULER_PATH), _definitions(OVERLAY_ROOT / SCHEDULER_PATH)
        changed = {name for name in base.keys() | overlay.keys() if base.get(name) != overlay.get(name)}
        allowed = {"Scheduler.__init__", "Scheduler.schedule", "Scheduler.schedule.schedule_running"}
        self.assertEqual({name for name in changed if "_glm53_" not in name} - allowed, set())
        self.assertTrue(allowed <= changed, "steps 43 and 44 are not wired in")


if __name__ == "__main__":
    unittest.main()
