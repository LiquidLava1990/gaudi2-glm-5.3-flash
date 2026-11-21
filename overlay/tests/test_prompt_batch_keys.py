"""Step 49, graph side: one region of several prompts through a shared key P:R:W. Comes with the patch to glm53_mixed_graph.py.

Hidden on the fake forward depends only on a row's own token and position, so a prompt's compact rows equal its rows run alone only if
padding, row order and each prompt's own table row all survive. The real forward's equality with bs 1 is card stage C1m."""
from __future__ import annotations

import os
import sys
import unittest
from functools import lru_cache
from pathlib import Path
from unittest import mock

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from overlay_testlib import load_overlay  # noqa: E402
from test_prompt_keys import BLOCK, Backend, FakeMetadata, KeyCase, RowForward  # noqa: E402

BATCH_KEYS = "VLLM_GAUDI_MIXED_PROMPT_BATCH_KEYS"
PROMPTS = "VLLM_GAUDI_MIXED_PROMPTS_PER_STEP"
SHARE = 256


class SharedCase(KeyCase):
    keys = "1024:8192,1024:1024"

    def setUp(self) -> None:
        environ = mock.patch.dict(os.environ, {BATCH_KEYS: "4:256:1024", PROMPTS: "4"})
        environ.start()
        self.addCleanup(environ.stop)
        super().setUp()
        self.forward, self.backend = RowForward(), Backend()
        self.cache = self.graph.GraphCache(self.forward, BLOCK, self.backend)

    def shared_batch(self, rows: list[int], starts: list[int], width: int = 1024, padded: int = SHARE) -> object:
        count = len(rows)
        table = torch.full((count, width), -1, dtype=torch.int32)
        for index, (length, start) in enumerate(zip(rows, starts)):
            used = max(1, (start + length + BLOCK - 1) // BLOCK)
            table[index, :used] = torch.arange(used, dtype=torch.int32) + 1000 * (index + 1)
        size = count * padded
        loc = torch.arange(count + 1) * padded
        meta = FakeMetadata(table, torch.ones(size), torch.arange(size), loc, size, 0)
        valid = tuple(index * padded + j for index, length in enumerate(rows) for j in range(length))
        region = self.step.Region(tuple(range(count)), tuple(rows), tuple(starts), padded, count, valid, meta, False)
        ids = torch.arange(size).reshape(count, padded) + 100
        return self.step.PreparedBatch(region, ids, ids + 7, (), id(meta))


class SharedKeys(SharedCase):
    def test_select_takes_the_narrowest_then_smallest_that_holds_every_prompt(self) -> None:
        with mock.patch.dict(os.environ, {BATCH_KEYS: "4:256:1024,2:256:1024,4:256:8192"}):
            self.assertEqual(self.graph.select_shared_prompt_key(2, 256, 1024), ((256, 1024, 2), ""))
            self.assertEqual(self.graph.select_shared_prompt_key(3, 256, 1024), ((256, 1024, 4), ""))
            self.assertEqual(self.graph.select_shared_prompt_key(3, 256, 2048), ((256, 8192, 4), ""))
            self.assertEqual(self.graph.select_shared_prompt_key(5, 256, 1024), (None, "multi_prompt"))
            self.assertEqual(self.graph.select_shared_prompt_key(3, 512, 1024), (None, "width_over"))
            self.assertEqual(self.graph.select_shared_prompt_key(3, 256, 16384), (None, "ctx_over"))

    def test_capture_runs_one_region_of_p_padded_rows(self) -> None:
        batch = self.shared_batch([256, 40, 256], [0, 1024, 2048])
        with self.graph.boot_capture_context():
            hidden = self.cache.run_key(batch)
        self.assertEqual(self.forward.shapes, [(1, 4 * SHARE)], "one forward, four padded rows")
        self.assertEqual(self.forward.tables[0].shape, (4, 1024))
        self.assertTrue(torch.equal(self.forward.tables[0][3], torch.full((1024,), -1, dtype=torch.int32)), "an empty row's table reads nothing")
        self.assertTrue(torch.equal(self.forward.tables[0][1][:2], batch.region.metadata.glm53_index_block_table[1][:2]))
        self.assertEqual(hidden.shape[0], 256 + 40 + 256)

    def test_each_prompts_rows_come_back_in_order_past_the_padding(self) -> None:
        batch = self.shared_batch([256, 40, 256], [0, 1024, 2048])
        with self.graph.boot_capture_context():
            hidden = self.cache.run_key(batch)
        rows = batch.tokens.reshape(-1)[torch.tensor(batch.region.valid)].float()
        self.assertTrue(torch.equal(hidden[:, 0], rows))

    def test_a_runner_row_wider_than_the_key_keeps_each_prompts_rows(self) -> None:
        """The runner's bucket can pad 256-row chunks to 1,024; the key's stride is its own, and only real rows come back."""
        batch = self.shared_batch([256, 40], [0, 0], padded=1024)
        with self.graph.boot_capture_context():
            hidden = self.cache.run_key(batch)
        rows = batch.tokens.reshape(-1)[torch.tensor(batch.region.valid)].float()
        self.assertTrue(torch.equal(hidden[:, 0], rows))
        self.assertEqual(self.forward.shapes, [(1, 4 * SHARE)])

    def test_replay_hits_the_key_with_no_capture(self) -> None:
        with self.graph.boot_capture_context():
            self.cache.run_key(self.shared_batch([256, 256], [0, 0]))
        hidden = self.cache.run_key(self.shared_batch([100, 200], [1024, 512]))
        self.assertEqual(self.backend.replays, 1)
        self.assertEqual(self.cache.key_captures, 1)
        self.assertEqual(hidden.shape[0], 300)

    def test_a_live_region_past_every_shared_key_refuses_on_every_rank_under_graphs_only(self) -> None:
        with mock.patch.dict(os.environ, {"HPU_GLM53_GRAPHS_ONLY": "1"}), mock.patch.object(self.graph.graphs_only, "arm_exit") as armed:
            for rows, width, reason in (([256] * 5, 1024, "multi_prompt"), ([512, 512], 1024, "width_over"), ([256, 256], 8192, "ctx_over")):
                with self.subTest(reason), self.assertRaisesRegex(RuntimeError, reason):
                    self.cache.run_key(self.shared_batch(rows, [0] * len(rows), width=width, padded=max(rows)))
        self.assertEqual(armed.call_count, 3)

    def test_a_lone_prompt_still_takes_a_single_prompt_key(self) -> None:
        batch = self.prompt_batch(700, 0, 1024)
        hidden = self.boot_capture(self.cache, batch)
        self.assertEqual(self.forward.shapes, [(1, 1024)])
        self.assertEqual(hidden.shape[0], 700)

    def test_with_no_shared_keys_a_region_of_two_is_still_a_miss(self) -> None:
        with mock.patch.dict(os.environ, {BATCH_KEYS: "", PROMPTS: ""}):
            cache = self.graph.GraphCache(RowForward(), BLOCK, Backend())
            self.assertEqual(cache._natural_key(self.shared_batch([256, 256], [0, 0])), (None, "multi_prompt"))

    def test_the_estimate_counts_every_padded_row(self) -> None:
        self.assertEqual(self.graph.estimate_bytes((0, 256, 1024, 4), BLOCK), self.graph.key_estimate_bytes(1024))
        self.assertEqual(self.graph.estimate_bytes((0, 1024, 1024), BLOCK), self.graph.key_estimate_bytes(1024))


class SharedBoot(SharedCase):
    def runner(self) -> object:
        from types import SimpleNamespace
        return SimpleNamespace(attn_block_size=BLOCK, block_size=BLOCK, max_model_len=1_048_576)

    def test_boot_lists_single_keys_then_shared_keys_and_a_dummy_of_p_requests_each(self) -> None:
        buckets = self.graph.validate_prompt_keys(self.runner())
        self.assertEqual(buckets, ((0, 1024, 8192), (0, 1024, 1024), (0, 256, 1024, 4)))
        scenarios = self.graph.warmup_scenarios(self.runner(), buckets)
        self.assertEqual(scenarios[-1], ((4, 256, (1024 * BLOCK - 256) // BLOCK), None))
        self.assertEqual(scenarios[0][0][0], 1)

    def test_shared_keys_with_no_single_keys_refuse(self) -> None:
        with mock.patch.dict(os.environ, {"VLLM_GAUDI_MIXED_PROMPT_KEYS": ""}), self.assertRaisesRegex(ValueError, BATCH_KEYS):
            self.graph.check_prompt_key_flags()

    def test_the_merge_check_refuses_shared_keys(self) -> None:
        with mock.patch.dict(os.environ, {self.graph.MERGE_CHECK_FLAG: "1"}), self.assertRaisesRegex(ValueError, BATCH_KEYS):
            self.graph.check_prompt_key_flags()

    def test_a_budget_filling_the_shared_key_passes_the_row_check(self) -> None:
        with mock.patch.dict(os.environ, {BATCH_KEYS: "4:1024:1024", "VLLM_GAUDI_MIXED_PREFILL_TOKENS": "4096",
                                          self.graph.SINGLE_TOKENS_FLAG: "1024"}):
            self.assertFalse([error for error in self.graph._graphs_only_errors(1024) if "PREFILL_TOKENS" in error])
        with mock.patch.dict(os.environ, {BATCH_KEYS: "4:256:1024", "VLLM_GAUDI_MIXED_PREFILL_TOKENS": "4096"}):
            self.assertTrue([error for error in self.graph._graphs_only_errors(1024) if "PREFILL_TOKENS" in error])

    def test_a_captured_set_missing_a_shared_key_refuses_naming_it(self) -> None:
        from types import SimpleNamespace
        cache = SimpleNamespace(entries={((0, 1024, 1024), "signature"): None})
        runner = SimpleNamespace(model=SimpleNamespace(_glm53_mixed_graph_cache=cache))
        with self.assertRaisesRegex(RuntimeError, "256:1024:4"):
            self.graph.check_keys_captured(runner, ((0, 1024, 1024), (0, 256, 1024, 4)))


@lru_cache(maxsize=1)
def runner_module() -> object:
    """Once only: the runner's imports register torch opaque types, which refuses a second load in one process."""
    return load_overlay("vllm_gaudi/v1/worker/hpu_model_runner.py")


class RunnerMerge(SharedCase):
    """The runner's real rule, on a stand-in with GLM's layers: prefix caching on, which alone would keep every prompt its own forward."""

    def stand_in(self) -> object:
        from types import SimpleNamespace
        return SimpleNamespace(num_mamba_like_layers=2, use_prefix_caching=True, _supports_hpu_batched_prefill=True,
                               _resolve_prefill_bucket=lambda *groups: (4, 256, 1024), _adaptive_long_prefill_batch_size=0,
                               _adaptive_long_prefill_context_tokens=0, max_prefill_batch_size=4, max_num_tokens=8192)

    @staticmethod
    def contents(*contexts: int) -> object:
        from types import SimpleNamespace
        return SimpleNamespace(req_ids=[f"r{index}" for index in range(len(contexts))], context_lens=list(contexts), prompt_lens=[9] * len(contexts),
                               token_ids=[[1] * 256 for _ in contexts])

    def merge(self) -> object:
        return runner_module().HPUModelRunner._can_merge_prefill_contents

    def test_a_shared_step_merges_prefix_cached_prompts_with_history_up_to_p(self) -> None:
        merge = self.merge()
        self.assertTrue(merge(self.stand_in(), self.contents(), self.contents(1024)))
        self.assertTrue(merge(self.stand_in(), self.contents(1024, 0, 2048), self.contents(4096)))
        self.assertFalse(merge(self.stand_in(), self.contents(1024, 0, 2048, 512), self.contents(4096)))

    def test_a_bucket_that_cannot_hold_the_step_still_says_no(self) -> None:
        stand_in = self.stand_in()

        def no_bucket(*groups: object) -> tuple[int, int, int]:
            raise runner_module().BucketingFailedException("no bucket")

        stand_in._resolve_prefill_bucket = no_bucket
        self.assertFalse(self.merge()(stand_in, self.contents(0), self.contents(0)))

    def test_flag_off_keeps_every_recurrent_prompt_its_own_forward(self) -> None:
        merge = self.merge()
        with mock.patch.dict(os.environ, {BATCH_KEYS: "", PROMPTS: ""}):
            self.assertFalse(merge(self.stand_in(), self.contents(), self.contents(0)))
            self.assertFalse(merge(self.stand_in(), self.contents(0), self.contents(0)))


if __name__ == "__main__":
    unittest.main()
