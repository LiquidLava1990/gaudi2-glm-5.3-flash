"""Lab seam for dual_serial_cell.py: VLLM_GAUDI_MIXED_SEPARATE_STEPS=1 makes the scheduler emit prompt-only and decode-only steps where it
would emit one mixed step, and the mixed step refuses a mixed one. Flag off is today's schedule. The real forward's equality across the two
arms is the dual-serial cell on a card; here every lane reads its own rows whole and in order, so its tokens cannot depend on the grouping."""
from __future__ import annotations

import os
import sys
import unittest
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "cells"))

from mixed_sched_harness import ROWS, Drive, prompt_tokens  # noqa: E402
from overlay_testlib import load_overlay  # noqa: E402

import dual_serial_cell as cell  # noqa: E402
import glm53_mixed_step as step_policy  # noqa: E402

from vllm.v1.request import RequestStatus  # noqa: E402

SEPARATE = "VLLM_GAUDI_MIXED_SEPARATE_STEPS"
KEYS = "VLLM_GAUDI_MIXED_PROMPT_KEYS"
GRAPHS_ONLY = "HPU_GLM53_GRAPHS_ONLY"
FLAGS = (SEPARATE, KEYS, GRAPHS_ONLY, "VLLM_GAUDI_MIXED_STEP", "VLLM_GAUDI_MIXED_SLICE_STEPS", "VLLM_GAUDI_MIXED_TAIL_FIRST")
MIXED = {"VLLM_GAUDI_MIXED_STEP": "1", "VLLM_GAUDI_MIXED_PREFILL_TOKENS": str(ROWS), "VLLM_PREFILL_CONTEXT_CELL_BUDGET": "0"}
KEYED = {**MIXED, KEYS: "1024:1024,1024:8192"}


@contextmanager
def env(values: dict[str, str]) -> Iterator[None]:
    """Exactly these flags set, every other flag of the seam unset."""
    with mock.patch.dict(os.environ, values):
        for name in FLAGS:
            if name not in values:
                os.environ.pop(name, None)
        yield


def staggered_run(values: dict[str, str]) -> Drive:
    """One lane decoding from the start while two long prompts arrive behind it: flag off, that is mixed steps."""
    with env(values):
        drive = Drive()
        drive.add("lane", prompt_tokens(1, 300), max_tokens=12)
        drive.run(2)
        drive.add("long-a", prompt_tokens(2, 3 * ROWS), max_tokens=4)
        drive.add("long-b", prompt_tokens(3, 2 * ROWS), max_tokens=4)
        drive.run_until_idle(60)
    return drive


class Flag(unittest.TestCase):
    def test_default_is_off_and_zero_or_one_only(self) -> None:
        with env({}):
            self.assertFalse(step_policy.separate_steps_enabled())
        for value, expected in (("0", False), ("1", True)):
            with env({SEPARATE: value}), self.subTest(value=value):
                self.assertEqual(step_policy.separate_steps_enabled(), expected)
        for value in ("2", "true", "-1"):
            with env({SEPARATE: value}), self.subTest(value=value), self.assertRaisesRegex(ValueError, SEPARATE):
                step_policy.separate_steps_enabled()

    def test_the_boot_refuses_it_without_the_mixed_step(self) -> None:
        with env({SEPARATE: "1"}), self.assertRaisesRegex(ValueError, "VLLM_GAUDI_MIXED_STEP=1"):
            step_policy.check_slot_flags(completion_first=False)

    def test_a_mixed_step_boot_accepts_it(self) -> None:
        with env({**MIXED, SEPARATE: "1"}):
            self.assertIsNone(step_policy.check_slot_flags(completion_first=False))

    def test_the_scheduler_runs_the_boot_check(self) -> None:
        with env({SEPARATE: "1"}), self.assertRaisesRegex(ValueError, SEPARATE):
            Drive(num_blocks=100)

    def test_both_seam_files_name_the_flag_so_the_cell_will_run(self) -> None:
        self.assertEqual(cell.seam_missing(Path(__file__).resolve().parents[1]), [])


class Schedule(unittest.TestCase):
    def test_flag_off_still_mixes_and_zero_is_flag_off(self) -> None:
        reference = staggered_run(MIXED)
        self.assertIn("mixed", [step.kind for step in reference.trace])
        self.assertEqual(staggered_run({**MIXED, SEPARATE: "0"}).trace, reference.trace)

    def test_on_no_step_mixes_and_both_kinds_run(self) -> None:
        kinds = [step.kind for step in staggered_run({**MIXED, SEPARATE: "1"}).trace]
        self.assertNotIn("mixed", kinds)
        self.assertIn("prompt", kinds)
        self.assertIn("decode", kinds)

    def test_while_both_wait_the_steps_alternate(self) -> None:
        """Steps 0 and 1 are the first lane's prompt and decode; the long prompts arrive at step 2, which continues the alternation."""
        kinds = [step.kind for step in staggered_run({**MIXED, SEPARATE: "1"}).trace]
        self.assertEqual(kinds[:8], ["prompt", "decode"] * 4)

    def test_every_lane_ends_with_the_same_tokens_as_the_mixed_schedule(self) -> None:
        mixed, separate = staggered_run(MIXED), staggered_run({**MIXED, SEPARATE: "1"})
        for name in ("lane", "long-a", "long-b"):
            with self.subTest(lane=name):
                self.assertEqual(separate.lanes[name].sampled, mixed.lanes[name].sampled)
                self.assertGreater(len(separate.lanes[name].sampled), 0)

    def test_no_decode_lane_is_held_when_no_prompt_waits(self) -> None:
        with env({**MIXED, SEPARATE: "1"}):
            drive = Drive()
            drive.add("lane", prompt_tokens(1, 300), max_tokens=6)
            drive.run_until_idle(20)
        self.assertEqual([step.kind for step in drive.trace[1:]], ["decode"] * 5)

    def test_keyed_steps_stay_inside_the_captured_keys(self) -> None:
        """Graphs-only serves a prompt chunk through a key and a decode through its own graph: neither kind is new, only separated."""
        drive = staggered_run({**KEYED, SEPARATE: "1", GRAPHS_ONLY: "1"})
        self.assertNotIn("mixed", [step.kind for step in drive.trace])
        for step in drive.trace:
            for _, start, rows in step.chunks:
                self.assertLessEqual(rows, ROWS)
                self.assertLessEqual(-(-(start + rows) // 128), 8192)


def block_waiter(drive: Drive, name: str, tokens: list[int]) -> None:
    """A prompt whose remote KV load has just finished: WAITING_FOR_REMOTE_KVS at the step's start, promoted when the WAITING pass reaches it."""
    scheduler = drive.scheduler
    drive.add(name, tokens, max_tokens=4)
    request = scheduler.requests[name]
    scheduler.waiting.remove_request(request)
    request.status = RequestStatus.WAITING_FOR_REMOTE_KVS
    scheduler.skipped_waiting.add_request(request)
    scheduler.finished_recving_kv_req_ids.add(name)


def promoted_mid_step(values: dict[str, str]) -> Drive:
    """A lane decoding alone, the step's only ready work, while a blocked request is promoted by the step's own WAITING pass."""
    with env(values):
        drive = Drive()
        drive.add("lane", prompt_tokens(1, 300), max_tokens=12)
        drive.run(2)
        with mock.patch.object(drive.scheduler, "_update_waiting_for_remote_kv"):
            block_waiter(drive, "promoted", prompt_tokens(2, ROWS))
            drive.run_until_idle(60)
    return drive


class PhaseBudget(unittest.TestCase):
    """A decode-only step has no prompt budget, whatever the WAITING pass promotes, or mixed_step.prepare stops the engine."""

    def test_a_request_promoted_mid_step_waits_for_the_prompt_turn(self) -> None:
        values = {**MIXED, SEPARATE: "1"}
        drive = promoted_mid_step(values)
        with env(values):
            for step in drive.trace:  # what mixed_step.prepare runs on every step: a mixed one raises, which stops the engine
                step_policy.refuse_mixed_step_when_separate([chunk[0] for chunk in step.chunks], step.decodes)
        self.assertEqual(drive.trace[2].kind, "decode", "the promoted request stays queued in the step that promoted it")
        self.assertEqual(drive.trace[3].kind, "prompt")
        self.assertGreater(len(drive.lanes["promoted"].sampled), 0)

    def test_the_same_promotion_still_mixes_with_the_flag_off(self) -> None:
        self.assertEqual(promoted_mid_step(MIXED).trace[2].kind, "mixed")

    def test_a_prompt_with_no_decode_ready_keeps_its_budget(self) -> None:
        with env({**MIXED, SEPARATE: "1"}):
            drive = Drive()
            drive.add("alone", prompt_tokens(1, ROWS), max_tokens=1)
            self.assertEqual(drive.step().kind, "prompt")


class NoRoom(unittest.TestCase):
    """A prompt that cannot allocate burns no step: the waiter is tried once, then again only when a block frees or another waiter arrives."""

    def run_no_room(self) -> Drive:
        with env({**MIXED, SEPARATE: "1"}):
            drive = Drive(num_blocks=10)
            drive.add("lane", prompt_tokens(1, 300), max_tokens=80)
            drive.run(2)
            drive.add("no-room", prompt_tokens(2, 8 * 128), max_tokens=2)
            drive.run_until_idle(300)
        return drive

    def test_a_waiter_with_no_room_does_not_alternate_with_idle_steps(self) -> None:
        drive = self.run_no_room()
        idle = [step.kind for step in drive.trace].count("idle")
        self.assertLessEqual(idle, 2, [step.kind for step in drive.trace])

    def test_the_lane_and_the_waiter_both_finish(self) -> None:
        drive = self.run_no_room()
        self.assertEqual(len(drive.lanes["lane"].sampled), 80)
        self.assertEqual(len(drive.lanes["no-room"].sampled), 2)

    def test_another_waiter_earns_one_more_try(self) -> None:
        with env({**MIXED, SEPARATE: "1"}):
            drive = Drive(num_blocks=10)
            drive.add("lane", prompt_tokens(1, 300), max_tokens=40)
            drive.run(2)
            drive.add("no-room", prompt_tokens(2, 8 * 128), max_tokens=2)
            settled = [step.kind for step in drive.run(6)[2:]]
            drive.add("small", prompt_tokens(3, 64), max_tokens=2)
            after = [step.kind for step in drive.run(4)]
        self.assertEqual(settled, ["decode"] * 4)
        self.assertEqual(after.count("idle"), 1, after)


class MixedStepRefusal(unittest.TestCase):
    def prepare(self, values: dict[str, str], *, prompts: list[str], decodes: list[str], warmup: bool = False) -> object:
        step = load_overlay("glm53_mixed_step.py")
        parallel = SimpleNamespace(data_parallel_size=1, pipeline_parallel_size=1)
        runner = SimpleNamespace(_execute_model_generic="generic", vllm_config=SimpleNamespace(speculative_config=None, parallel_config=parallel),
                                 lora_config=None, use_aux_hidden_state_outputs=False, use_merged_prefill=False, model_has_chunked_attention=False,
                                 input_batch=SimpleNamespace(num_prompt_logprobs=0), model=SimpleNamespace(flatten_input=True, is_mm_optimized=False))
        with env(values):
            return step.prepare(runner, None, SimpleNamespace(prompt_req_ids=prompts, decode_req_ids=decodes), None, None, warmup_mode=warmup)

    def test_a_step_with_both_is_refused_naming_the_flag_and_the_fix(self) -> None:
        with self.assertRaisesRegex(ValueError, f"{SEPARATE}.*scheduler"):
            self.prepare({**MIXED, SEPARATE: "1"}, prompts=["p"], decodes=["d"])

    def test_a_decode_only_step_passes_through_to_the_runner(self) -> None:
        self.assertEqual(self.prepare({**MIXED, SEPARATE: "1"}, prompts=[], decodes=["d"]), "generic")

    def test_single_kind_steps_flag_off_and_warmup_are_not_refused(self) -> None:
        step = load_overlay("glm53_mixed_step.py")
        cases = (({**MIXED, SEPARATE: "1"}, ["p"], [], False), ({**MIXED, SEPARATE: "1"}, [], ["d"], False),
                 ({**MIXED, SEPARATE: "0"}, ["p"], ["d"], False), ({**MIXED, SEPARATE: "1"}, ["p"], ["d"], True))
        for values, prompts, decodes, warmup in cases:
            with env(values), self.subTest(values=values, prompts=prompts, decodes=decodes, warmup=warmup):
                step.refuse_mixed_step_when_separate(prompts, decodes, warmup_mode=warmup)


class GraphSide(unittest.TestCase):
    def setUp(self) -> None:
        self.graph = load_overlay("glm53_mixed_graph.py")

    def test_graphs_only_without_keys_refuses_naming_the_seam_flag(self) -> None:
        with env({**MIXED, SEPARATE: "1", GRAPHS_ONLY: "1"}), self.assertRaisesRegex(ValueError, f"{SEPARATE}.*{KEYS}"):
            self.graph.check_separate_steps_flags(())

    def test_graphs_only_with_keys_and_flag_off_or_graphs_only_off_pass(self) -> None:
        for values, keys in (({**KEYED, SEPARATE: "1", GRAPHS_ONLY: "1"}, ((1024, 1024),)), ({**MIXED, SEPARATE: "1"}, ()),
                             ({**MIXED, GRAPHS_ONLY: "1"}, ()), ({}, ())):
            with env(values), self.subTest(values=values):
                self.assertIsNone(self.graph.check_separate_steps_flags(keys))

    def test_the_key_flag_check_runs_it(self) -> None:
        with env({**MIXED, SEPARATE: "1", GRAPHS_ONLY: "1"}), self.assertRaisesRegex(ValueError, SEPARATE):
            self.graph.check_prompt_key_flags()


if __name__ == "__main__":
    unittest.main()
