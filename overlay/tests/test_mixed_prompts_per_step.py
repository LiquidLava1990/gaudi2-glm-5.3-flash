"""Step 49 (DUAL-SUSTAINED-PLAN 12.3b): several agents' prefill in one step, the scheduler and the mixed-step side, on the real scheduler and KV
manager. Output equal to one prompt per step here means every lane read its own rows whole and in order; the real forward's equality is C1m."""
from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from mixed_sched_harness import BLOCK_SIZE, ROWS, Drive, StepTrace, prompt_tokens  # noqa: E402

import glm53_mixed_step as step_policy  # noqa: E402

PROMPTS = "VLLM_GAUDI_MIXED_PROMPTS_PER_STEP"
BATCH_KEYS = "VLLM_GAUDI_MIXED_PROMPT_BATCH_KEYS"
TAIL = "VLLM_GAUDI_MIXED_TAIL_FIRST"
AGE = "VLLM_GAUDI_MIXED_TAIL_AGE_S"
SLICE = "VLLM_GAUDI_MIXED_SLICE_STEPS"
KEYS = "VLLM_GAUDI_MIXED_PROMPT_KEYS"
FLAGS = (PROMPTS, BATCH_KEYS, TAIL, AGE, SLICE, KEYS, "VLLM_GAUDI_MIXED_STEP_SECONDS")
MIXED = {"VLLM_GAUDI_MIXED_STEP": "1", "VLLM_GAUDI_MIXED_PREFILL_TOKENS": str(ROWS), "VLLM_PREFILL_CONTEXT_CELL_BUDGET": "0"}
ONE_PROMPT = {**MIXED, TAIL: "1", KEYS: "1024:1024,1024:8192"}
SHARED = {**ONE_PROMPT, PROMPTS: "4", BATCH_KEYS: "4:256:1024"}
SHARE = ROWS // 4


def env(values: dict[str, str]) -> mock._patch_dict:
    return mock.patch.dict(os.environ, {**{name: "" for name in FLAGS if name not in values}, **values})


def prompts_in(step: StepTrace) -> tuple[str, ...]:
    return tuple(chunk[0] for chunk in step.chunks)


def contiguous(case: unittest.TestCase, drive: Drive, name: str) -> None:
    rows = [(start, count) for step in drive.trace for owner, start, count in step.chunks if owner == name]
    for (start, count), (next_start, _) in zip(rows, rows[1:]):
        case.assertEqual(start + count, next_start, f"{name} left a hole")


class Flags(unittest.TestCase):
    def test_default_is_one_prompt(self) -> None:
        with env(MIXED):
            self.assertEqual(step_policy.prompts_per_step(), 1)
            self.assertEqual(step_policy.prompt_batch_keys(), ())

    def test_bad_values_refuse_naming_the_flag(self) -> None:
        for value in ("0", "-1", "two", "1.5"):
            with env({PROMPTS: value}), self.subTest(value=value), self.assertRaisesRegex(ValueError, PROMPTS):
                step_policy.prompts_per_step()

    def test_bad_keys_refuse_naming_the_flag(self) -> None:
        for spec in ("256:1024", "4:256:1000", "4:100:1024", "1:256:1024", "4:256:1024,4:256:1024", "a:b:c"):
            with env({BATCH_KEYS: spec}), self.subTest(spec=spec), self.assertRaisesRegex(ValueError, BATCH_KEYS):
                step_policy.prompt_batch_keys()

    def test_boot_refuses_what_could_not_share(self) -> None:
        cases = (("no tail-first", {**MIXED, PROMPTS: "4", BATCH_KEYS: "4:256:1024"}, TAIL),
                 ("uneven split", {**ONE_PROMPT, PROMPTS: "3", BATCH_KEYS: "3:256:1024"}, "split evenly"),
                 ("no key at this split", {**ONE_PROMPT, PROMPTS: "4", BATCH_KEYS: "4:512:1024"}, "4:256:WIDTH"),
                 ("no keys", {**ONE_PROMPT, PROMPTS: "4"}, BATCH_KEYS),
                 ("keys with no sharing", {**ONE_PROMPT, BATCH_KEYS: "4:256:1024"}, PROMPTS))
        for name, values, needle in cases:
            with env(values), self.subTest(name), self.assertRaisesRegex(ValueError, needle):
                step_policy.check_slot_flags(completion_first=False)

    def test_the_runner_merges_only_up_to_p_and_keeps_its_own_rules_when_off(self) -> None:
        with env(ONE_PROMPT):
            self.assertIsNone(step_policy.shared_prompt_merge(0, 1))
            step_policy.check_shared_prompt_batch(1)
        with env(SHARED):
            self.assertTrue(step_policy.shared_prompt_merge(0, 1))
            self.assertTrue(step_policy.shared_prompt_merge(3, 1))
            self.assertFalse(step_policy.shared_prompt_merge(4, 1))
            step_policy.check_shared_prompt_batch(4)
            with self.assertRaisesRegex(ValueError, "VLLM_PROMPT_BS_BUCKET_MAX=4"):
                step_policy.check_shared_prompt_batch(2)

    def test_scheduler_runs_the_boot_check(self) -> None:
        with env({**ONE_PROMPT, PROMPTS: "4"}), self.assertRaisesRegex(ValueError, BATCH_KEYS):
            Drive(num_blocks=100)

    def test_a_good_config_boots(self) -> None:
        with env(SHARED):
            step_policy.check_slot_flags(completion_first=False)


class SharedSteps(unittest.TestCase):
    def test_flag_off_and_p_1_are_the_same_schedule(self) -> None:
        traces = []
        for values in (ONE_PROMPT, {**ONE_PROMPT, PROMPTS: "1"}):
            with env(values):
                drive = Drive()
                for name, seed, count in (("a", 1, 3 * ROWS), ("b", 2, 2 * ROWS), ("c", 3, 300)):
                    drive.add(name, prompt_tokens(seed, count))
                drive.run_until_idle(20)
            traces.append(drive.trace)
        self.assertEqual(traces[0], traces[1])
        self.assertEqual(max(len(step.chunks) for step in traces[0]), 1)

    def test_four_tails_share_one_step_at_a_quarter_of_the_budget_each(self) -> None:
        with env(SHARED):
            drive = Drive()
            for name, seed in (("a", 1), ("b", 2), ("c", 3), ("d", 4)):
                drive.add(name, prompt_tokens(seed, 2 * SHARE))
            first = drive.step()
        self.assertEqual(sorted(prompts_in(first)), ["a", "b", "c", "d"])
        self.assertEqual({count for _, _, count in first.chunks}, {SHARE})

    def test_short_tails_all_finish_together_in_fewer_forwards(self) -> None:
        names = ("a", "b", "c", "d")
        counts = {}
        for label, values in (("one", ONE_PROMPT), ("shared", SHARED)):
            with env(values):
                drive = Drive()
                for index, name in enumerate(names):
                    drive.add(name, prompt_tokens(index + 1, SHARE))
                drive.run_until_idle(10)
            counts[label] = sum(1 for step in drive.trace if step.chunks)
            for name in names:
                contiguous(self, drive, name)
        self.assertEqual(counts["one"], 4)
        self.assertEqual(counts["shared"], 1)

    def test_a_chunk_shorter_than_the_share_runs_its_real_rows(self) -> None:
        with env(SHARED):
            drive = Drive()
            drive.add("short", prompt_tokens(1, 40))
            drive.add("long", prompt_tokens(2, 4 * ROWS))
            first = drive.step()
        self.assertEqual(dict((owner, count) for owner, _, count in first.chunks), {"short": 40, "long": SHARE})

    def test_a_lone_prompt_takes_the_whole_budget(self) -> None:
        with env(SHARED):
            drive = Drive()
            drive.add("alone", prompt_tokens(1, 4 * ROWS))
            first = drive.step()
        self.assertEqual(first.chunks, (("alone", 0, ROWS),))

    def test_the_fewest_chunks_left_go_first_and_the_fifth_waits(self) -> None:
        with env(SHARED):
            drive = Drive()
            sizes = (("longest", 40 * SHARE), ("c", 4 * SHARE), ("a", 2 * SHARE), ("d", 5 * SHARE), ("b", 3 * SHARE))
            for index, (name, count) in enumerate(sizes):
                drive.add(name, prompt_tokens(index + 1, count))
            first = drive.step()
        self.assertEqual(sorted(prompts_in(first)), ["a", "b", "c", "d"])

    def test_the_step_never_passes_p_or_the_budget(self) -> None:
        with env(SHARED):
            drive = Drive()
            for index in range(9):
                drive.add(f"p{index}", prompt_tokens(index + 1, (index + 1) * SHARE))
            drive.run_until_idle(60)
        for step in drive.trace:
            self.assertLessEqual(len(step.chunks), 4)
            self.assertLessEqual(sum(count for _, _, count in step.chunks), ROWS)

    def test_decodes_keep_their_slot_in_a_shared_step(self) -> None:
        with env(SHARED):
            drive = Drive()
            drive.add("lane", prompt_tokens(100, 24), max_tokens=500)
            drive.run(2)
            for name, seed in (("a", 1), ("b", 2)):
                drive.add(name, prompt_tokens(seed, 2 * SHARE))
            step = drive.step()
        self.assertEqual(step.decodes, ("lane",))
        self.assertEqual(sorted(prompts_in(step)), ["a", "b"])


class AgingAndWidth(unittest.TestCase):
    def test_a_long_prompt_ages_into_a_shared_step_and_never_past_the_limit(self) -> None:
        with env({**SHARED, AGE: "30"}):
            drive = Drive(num_blocks=3000)
            drive.add("long", prompt_tokens(1, 60 * ROWS))
            for _ in range(120):
                while sum(name.startswith("t") for name in drive.scheduler.requests) < 4:
                    drive.add(f"t{len(drive.lanes)}", prompt_tokens(10_000 + len(drive.lanes), 8 * SHARE))
                drive.step()
        starts = [drive.started_at[index] for index, step in enumerate(drive.trace) if "long" in prompts_in(step)]
        self.assertGreater(len(starts), 2)
        self.assertLessEqual(max(later - earlier for earlier, later in zip([drive.started_at[0], *starts], starts)), 30.0)

    def test_a_tail_does_not_join_a_wider_prompts_table(self) -> None:
        """A chunk at 8,192 blocks would make the tail pay for that whole table. Each goes alone instead, whole budget and all."""
        wide = prompt_tokens(5, 2000 * BLOCK_SIZE)
        with env(SHARED):
            drive = Drive(num_blocks=8000)
            drive.add("seed", wide)
            drive.run_until_idle(300)
            drive.add("deep", wide + prompt_tokens(6, 4 * ROWS))
            drive.add("tail", prompt_tokens(7, 3 * ROWS))
            mark = len(drive.trace)
            drive.run_until_idle(30)
        for step in drive.trace[mark:]:
            self.assertLessEqual(len(step.chunks), 1)
        self.assertTrue(all(count == ROWS for step in drive.trace[mark:mark + 1] for _, _, count in step.chunks))

    def test_prompts_of_a_key_width_share_past_the_first_rows(self) -> None:
        shared = prompt_tokens(5, 900 * BLOCK_SIZE)
        with env(SHARED):
            drive = Drive(num_blocks=8000)
            drive.add("seed", shared)
            drive.run_until_idle(120)
            drive.add("x", shared + prompt_tokens(6, 2 * SHARE))
            drive.add("y", prompt_tokens(7, 2 * SHARE))
            first = drive.step()
        self.assertEqual(sorted(prompts_in(first)), ["x", "y"])

    def test_a_deeper_hit_than_the_hint_leaves_the_shared_step(self) -> None:
        """The real lookup put it past the key's table: no key would hold the step, so it is requeued before anything is allocated."""
        shared = prompt_tokens(5, 2000 * BLOCK_SIZE)
        with env(SHARED):
            drive = Drive(num_blocks=8000)
            drive.add("seed", shared)
            drive.run_until_idle(300)
            drive.add("liar", shared + prompt_tokens(6, 2 * SHARE))
            drive.add("honest", prompt_tokens(7, 2 * SHARE))
            real = drive.scheduler._glm53_estimated_start
            hinted = lambda request: 0 if request.request_id == "liar" else real(request)  # noqa: E731
            with mock.patch.object(drive.scheduler, "_glm53_estimated_start", side_effect=hinted):
                first = drive.step()
                rest = drive.run(4)
        self.assertEqual(prompts_in(first), ("honest",))
        self.assertEqual(drive.scheduler._glm53_slot_ledger.mismatches, 1)
        liar = next(step for step in rest if "liar" in prompts_in(step))
        self.assertEqual(prompts_in(liar), ("liar",), "the real hit replaced the hint, so it goes alone")
        self.assertEqual(liar.chunks[0][1], 2000 * BLOCK_SIZE)

    def test_two_waiters_that_fit_alone_but_not_together_take_turns(self) -> None:
        with env(SHARED):
            drive = Drive(num_blocks=500)
            drive.add("first", prompt_tokens(1, 300 * BLOCK_SIZE))
            drive.add("second", prompt_tokens(2, 300 * BLOCK_SIZE))
            drive.run_until_idle(400)
        order = sorted(drive.finished_at, key=drive.finished_at.get)
        self.assertEqual(order, ["first", "second"])
        self.assertEqual(drive.restarted, [])
        contiguous(self, drive, "first")
        contiguous(self, drive, "second")


class OutputUnchanged(unittest.TestCase):
    def workload(self, values: dict[str, str], seed_blocks: int = 40) -> Drive:
        cached = prompt_tokens(5, seed_blocks * BLOCK_SIZE)
        with env(values):
            drive = Drive(num_blocks=3000)
            drive.add("seed", cached)
            drive.run_until_idle(30)
            drive.add("greedy", prompt_tokens(1, 2 * ROWS + 37), max_tokens=40)
            drive.run(2)
            drive.add("long", prompt_tokens(2, 12 * ROWS + 5), max_tokens=5)
            drive.add("tail", prompt_tokens(3, 300), max_tokens=5)
            drive.add("returning", cached + prompt_tokens(4, 700), max_tokens=5)
            drive.add("cold", prompt_tokens(6, 3 * ROWS + 11), max_tokens=5)
            drive.run_until_idle(800)
        return drive

    def test_every_lane_reads_the_same_rows_and_samples_the_same_tokens(self) -> None:
        one, shared = self.workload(ONE_PROMPT), self.workload(SHARED)
        self.assertGreater(max(len(step.chunks) for step in shared.trace), 1, "the case must really share steps")
        self.assertNotEqual(one.trace, shared.trace)
        for lane, state in one.lanes.items():
            with self.subTest(lane=lane):
                self.assertEqual(shared.lanes[lane].sampled, state.sampled)
                self.assertEqual(shared.lanes[lane].state, state.state)

    def test_shared_steps_survive_a_preemption(self) -> None:
        with env(SHARED):
            drive = Drive(num_blocks=3000)
            drive.add("lane", prompt_tokens(100, 24), max_tokens=300)
            drive.run(2)
            for name, seed in (("a", 1), ("b", 2)):
                drive.add(name, prompt_tokens(seed, 8 * ROWS), max_tokens=3)
            drive.run(4)
            manager, refused = drive.scheduler.kv_cache_manager, []
            real = manager.allocate_slots

            def refuse_once(request: object, *args: object, **kwargs: object) -> object:
                if request.request_id == "lane" and not refused:
                    refused.append(True)
                    return None
                return real(request, *args, **kwargs)

            with mock.patch.object(manager, "allocate_slots", side_effect=refuse_once):
                drive.step()
            drive.run_until_idle(600)
        self.assertTrue(drive.restarted)
        self.assertEqual(sorted(drive.finished_at), ["a", "b", "lane"])


def _fake_runner(lengths: list[int], starts: list[int]) -> SimpleNamespace:
    ids = [f"r{index}" for index in range(len(lengths))]
    computed = {index: start for index, start in enumerate(starts)}
    processor = SimpleNamespace(process_metadata=lambda metadata, *_: metadata)
    return SimpleNamespace(input_batch=SimpleNamespace(req_id_to_index={name: index for index, name in enumerate(ids)},
                                                       num_computed_tokens_cpu=computed),
                           model=SimpleNamespace(metadata_processor=processor), dtype=torch.float32)


class WorkerRegion(unittest.TestCase):
    """What the runner's merged batch becomes on the mixed-step side: one region, one padded row per prompt, and each prompt's logits."""

    LENGTHS, STARTS = [SHARE, 40, SHARE], [1024, 0, 2048]

    def prepared(self, requests: list[str], rows: int) -> step_policy.PreparedBatch:
        ids = ["r0", "r1", "r2"]
        runner = _fake_runner(self.LENGTHS, self.STARTS)
        schedule = SimpleNamespace(num_scheduled_tokens=dict(zip(ids, self.LENGTHS)))
        tokens = torch.zeros((4, SHARE), dtype=torch.long)
        logits = step_policy._host_logits(ids, requests, tuple(self.LENGTHS), SHARE, rows)
        return step_policy._batch(runner, ids, tokens, tokens, SimpleNamespace(), logits, schedule, False)

    def test_one_region_holds_every_prompt_with_its_rows_and_start(self) -> None:
        region = self.prepared(["r1"], 1).region
        self.assertEqual((region.batch, region.width), (4, SHARE))
        self.assertEqual(region.requests, (0, 1, 2))
        self.assertEqual(region.lengths, tuple(self.LENGTHS))
        self.assertEqual(region.starts, tuple(self.STARTS))
        self.assertEqual(region.valid[SHARE], SHARE, "the short prompt starts its own padded row")
        self.assertEqual(region.valid[SHARE + 40], 2 * SHARE, "and the third starts its own, past the padding")
        plan = step_policy.MixedMetadata((region,))
        self.assertEqual(plan.prefill_slices, [(0, 1024, SHARE), (1, 0, 40), (2, 2048, SHARE)])

    def test_each_prompts_final_row_is_the_one_sampled(self) -> None:
        batch = self.prepared(["r1", "r2"], 2)
        self.assertEqual(batch.logits, (SHARE + 39, 2 * SHARE + SHARE - 1))
        compact = step_policy._key_logit_rows(batch)
        self.assertEqual(compact, (SHARE + 39, SHARE + 40 + SHARE - 1), "rows of the compact hidden the padded key's rows collapse to")

    def test_a_shared_chunk_with_no_final_row_reads_a_real_row(self) -> None:
        batch = self.prepared([], 1)
        self.assertEqual(batch.logits, (-1,))
        self.assertEqual(step_policy._key_logit_rows(batch), (sum(self.LENGTHS) - 1,))


if __name__ == "__main__":
    unittest.main()
