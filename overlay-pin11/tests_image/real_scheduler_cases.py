"""R3: the pin on the real r4d Scheduler + OffloadingConnector, two turns through the real waiting loop (CPU, no model).

test_image_attn_pin runs this file in a child interpreter with HPU_GLM53_ATTN_PIN=1 and HPU_GLM53_ATTENTION_ONLY_OFFLOAD=1,
so the dist-packages .pth wraps the real classes at import. Turn 1 runs a prompt to its stop token (the instant fake worker
completes every store job in the step that carries it) and its attention blocks are pinned; turn 2 is the same history plus
a new user message, so the connector answers a CPU-store hit and the pin plans an adoption. The cases cover:
  * adoption on the stock waiting loop (scheduler.py:1961-2298);
  * a plan held across each waiting-loop exit before allocation: continue at 1985 and 2075, break at 2261-2263 and 2280
    (the held plan survives the exit, is reused on the next pass, and evict_lru reclaims it once no pin is left);
  * the hybrid fail-closed path (1999-2010): ext_tokens drops to 0 after plan_adopt, so the plan is released, no discount;
  * Scheduler.reset_prefix_cache (3821-3861) with a pin and a held plan: both go back before the KV reset, which succeeds.
Shape: FullAttention KV only, block 16, prefix caching off, sync Scheduler, kv_both OffloadingConnector (cold reuse 512).
"""
from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path
from typing import Any, Callable

sys.path.insert(0, os.environ.get("PINKIT", "/pinkit"))
import glm53_attn_pin as pin  # noqa: E402

BLOCK = 16
NUM_BLOCKS = 512
MAX_LEN = 4096
BATCH_TOKENS = 1024
STOP = 7
OUTPUT = (301, 302, 303, STOP)
TURN1_TOKENS = 600  # above cold_reuse_min_tokens (512), so turn 2 reads the CPU store
MAX_STEPS = 200
TINY = str(Path(__file__).resolve().parent / "tiny-llama")


def kv_spec() -> Any:
    import torch
    from vllm.v1.kv_cache_interface import FullAttentionSpec

    return FullAttentionSpec(block_size=BLOCK, num_kv_heads=1, head_size=8, dtype=torch.float32)


def make_vllm_config() -> Any:
    from vllm.config import CacheConfig, KVTransferConfig, ModelConfig, SchedulerConfig, VllmConfig

    extra = {"cpu_bytes_to_use": NUM_BLOCKS * kv_spec().page_size_bytes, "eviction_policy": "lru", "store_threshold": 1,
             "request_hash_block_size": BLOCK, "store_batch_keys": 32, "pending_store_wait_threshold": 8192,
             "pending_store_wait_deadline_s": 2.0, "cold_reuse_min_tokens": 512}
    model = ModelConfig(model=TINY, skip_tokenizer_init=True, dtype="float32", seed=0, max_model_len=MAX_LEN)
    sched = SchedulerConfig(max_num_seqs=8, max_num_batched_tokens=BATCH_TOKENS, max_model_len=MAX_LEN,
                            is_encoder_decoder=False, enable_chunked_prefill=True, async_scheduling=False)
    cache = CacheConfig(block_size=BLOCK, gpu_memory_utilization=0.9, cache_dtype="auto", enable_prefix_caching=False)
    transfer = KVTransferConfig(kv_connector="OffloadingConnector", kv_role="kv_both", kv_connector_extra_config=extra)
    config = VllmConfig(model_config=model, scheduler_config=sched, cache_config=cache, kv_transfer_config=transfer)
    cache.num_gpu_blocks = NUM_BLOCKS
    return config


def make_scheduler() -> Any:
    from vllm.v1.core.sched.scheduler import Scheduler
    from vllm.v1.kv_cache_interface import KVCacheConfig, KVCacheGroupSpec, KVCacheTensor
    from vllm.v1.structured_output import StructuredOutputManager

    spec = kv_spec()
    size = NUM_BLOCKS * spec.page_size_bytes
    tensor = KVCacheTensor(size=size, layers=["layer.0"], layer_stride=size, block_stride=spec.page_size_bytes)
    kv_config = KVCacheConfig(num_blocks=NUM_BLOCKS, kv_cache_tensors=[tensor],
                              kv_cache_groups=[KVCacheGroupSpec(["layer.0"], spec)])
    config = make_vllm_config()
    return Scheduler(vllm_config=config, kv_cache_config=kv_config,
                     structured_output_manager=StructuredOutputManager(config), block_size=BLOCK,
                     hash_block_size=BLOCK, log_stats=False)


def make_request(request_id: str, prompt: list[int]) -> Any:
    from vllm.sampling_params import SamplingParams
    from vllm.utils.hashing import get_hash_fn_by_name
    from vllm.v1.core.kv_cache_utils import get_request_block_hasher, init_none_hash
    from vllm.v1.request import Request

    hash_fn = get_hash_fn_by_name("sha256")
    init_none_hash(hash_fn)
    params = SamplingParams(max_tokens=len(OUTPUT) + 4, stop_token_ids=[STOP], ignore_eos=True)
    return Request(request_id=request_id, prompt_token_ids=prompt, sampling_params=params, pooling_params=None,
                   arrival_time=0.0, block_hasher=get_request_block_hasher(BLOCK, hash_fn))


def connector_output(step: Any) -> Any:
    """The instant worker: every load and store job the step carries has completed."""
    metadata = step.kv_connector_metadata
    if metadata is None or not (metadata.load_jobs or metadata.store_jobs):
        return None
    from vllm.distributed.kv_transfer.kv_connector.v1.offloading.common import OffloadingWorkerMetadata
    from vllm.v1.outputs import KVConnectorOutput

    completed = dict.fromkeys([*metadata.load_jobs, *metadata.store_jobs], 1)
    loaded = {job.req_id for job in metadata.load_jobs.values()}
    return KVConnectorOutput(finished_sending=set(), finished_recving=loaded,
                             kv_connector_worker_meta=OffloadingWorkerMetadata(completed_jobs=completed))


class Driver:
    """One real scheduler stepped synchronously; a request samples OUTPUT once its prefill is complete."""

    def __init__(self) -> None:
        self.scheduler = make_scheduler()
        self.registry = pin.registry()
        assert self.registry is not None and self.registry.kv is self.scheduler.kv_cache_manager, "pin not armed"
        self.emitted: dict[str, int] = {}

    def step(self) -> Any:
        from vllm.v1.outputs import ModelRunnerOutput

        out = self.scheduler.schedule()
        req_ids = list(out.num_scheduled_tokens)
        sampled = [self._sample(rid) for rid in req_ids]
        runner = ModelRunnerOutput(req_ids=req_ids, req_id_to_index={r: i for i, r in enumerate(req_ids)},
                                   sampled_token_ids=sampled, kv_connector_output=connector_output(out))
        self.scheduler.update_from_output(out, runner)
        return out

    def _sample(self, request_id: str) -> list[int]:
        request = self.scheduler.requests[request_id]
        if request.is_prefill_chunk:
            return []
        index = min(self.emitted.get(request_id, 0), len(OUTPUT) - 1)
        self.emitted[request_id] = index + 1
        return [OUTPUT[index]]

    def run_until(self, done: Callable[[], bool], what: str) -> None:
        for _ in range(MAX_STEPS):
            if done():
                return
            self.step()
        raise AssertionError(f"{what} not reached in {MAX_STEPS} steps; pins={len(self.registry.pins)} "
                             f"plans={list(self.registry.plans)} counters={self.registry.counters}")

    def run_until_freed(self, request: Any, what: str) -> None:
        """Until the scheduler dropped the request: KVCacheManager.free (the pin's finish hook) ran, delayed or not."""
        self.run_until(lambda: request.request_id not in self.scheduler.requests, what)

    def stop_on_deferred_free(self, request: Any) -> list[str]:
        """Stop-finish a decoding request the way async serving does: defer_block_free on (async + kv_both,
        scheduler.py:521-527) and the next step already carrying it, so _free_request_blocks takes pop_blocks_for_free.
        Returns the request ids that went through pop_blocks_for_free."""
        from vllm.v1.request import RequestStatus

        kv, calls = self.scheduler.kv_cache_manager, []
        wrapped = kv.pop_blocks_for_free

        def spy(req: Any) -> Any:
            calls.append(req.request_id)
            return wrapped(req)

        kv.pop_blocks_for_free = spy
        self.scheduler.defer_block_free = True
        request.last_sched_seq = self.scheduler.processed_step_seq + 1
        try:
            self.scheduler.finish_requests(request.request_id, RequestStatus.FINISHED_STOPPED)
            self.run_until_freed(request, f"{request.request_id} freed on the deferred path")
        finally:
            del kv.pop_blocks_for_free
        return calls

    def turn1(self) -> Any:
        r1 = make_request("t1", [11 + (i % 50) for i in range(TURN1_TOKENS)])
        self.scheduler.add_request(r1)
        self.run_until(lambda: r1.is_finished() and bool(self.registry.pins), "turn 1 finished and pinned")
        return r1

    def add_turn2(self, history: Any, request_id: str = "t2") -> Any:
        r2 = make_request(request_id, [*history.all_token_ids, *([5000] * 40)])
        self.scheduler.add_request(r2)
        return r2

    def block_ids(self, request_id: str) -> list[int]:
        return [b.block_id for b in self.scheduler.kv_cache_manager.get_blocks(request_id).blocks[0]]

    def hashed_blocks(self) -> list[int]:
        return [b.block_id for b in self.scheduler.kv_cache_manager.block_pool.blocks if b.block_hash is not None]


class RealSchedulerCases(unittest.TestCase):
    def setUp(self) -> None:
        self.assertEqual(os.environ.get(pin.FLAG), "1", "run in the flag-on child interpreter")
        self.assertEqual(os.environ.get("HPU_GLM53_ATTENTION_ONLY_OFFLOAD"), "1")
        self.d = Driver()
        self.r1 = self.d.turn1()
        self.pinned_ids = [b.block_id for b in self.d.registry.pins[0].blocks[0]]

    def assert_adopted(self, r2: Any) -> None:
        reg, k = self.d.registry, self.d.registry.counters.local_hit_tokens // BLOCK
        self.assertEqual(reg.counters.adopted, 1, reg.counters)
        self.assertGreaterEqual(k, 1)
        self.assertEqual(self.d.block_ids(r2.request_id)[:k], self.pinned_ids[:k], "the pinned blocks were adopted")
        self.assertEqual(self.d.hashed_blocks(), [], "every marker hash was reset")
        self.d.run_until(r2.is_finished, "turn 2 finished")
        self.assertEqual(reg.counters.errors, 0)

    def hold_one_step(self, r2: Any, patches: dict[str, Any], owner: Any) -> None:
        """One schedule() with the patched exits: the plan must outlive the exit while r2 stays waiting, unallocated."""
        for name, value in patches.items():
            setattr(owner, name, value)
        try:
            self.d.step()
        finally:
            for name in patches:
                delattr(owner, name)
        self.assertEqual(self.d.registry.held_plans, 1, self.d.registry.counters)
        self.assertEqual(self.d.registry.pins, [])
        self.assertEqual(self.d.block_ids(r2.request_id), [], "no allocation ran for the held request")

    def test_reset_prefix_cache_releases_the_pin_and_a_held_plan(self) -> None:
        """The stock reset resets the KV manager before the connector (scheduler.py:3849) and the block pool refuses
        while any block is in use (block_pool.py:773-780): with the pin held it returned False, and with
        reset_running_requests=True it raised. The pin and the held plan must go back first."""
        sched, reg = self.d.scheduler, self.d.registry
        r2 = self.d.add_turn2(self.r1)
        self.hold_one_step(r2, {"allocate_slots": lambda *a, **k: None}, sched.kv_cache_manager)
        self.assertTrue(sched.reset_prefix_cache(reset_running_requests=True), reg.counters)
        self.assertEqual((reg.pins, reg.plans, reg.counters.errors), ([], {}, 0))
        pool = sched.kv_cache_manager.block_pool
        self.assertEqual(pool.get_num_free_blocks(), pool.num_gpu_blocks - 1, "only the null block is in use")
        self.d.run_until(r2.is_finished, "turn 2 finished on the stock load after the reset")
        self.assertEqual((reg.counters.adopted, reg.counters.errors, self.d.hashed_blocks()), (0, 0, []))

    def test_turn2_adopts_the_pinned_blocks(self) -> None:
        r2 = self.d.add_turn2(self.r1)
        self.d.step()
        self.assert_adopted(r2)

    def test_plan_held_across_continue_2075_then_1985_is_reused(self) -> None:
        r2 = self.d.add_turn2(self.r1)
        self.hold_one_step(r2, {"_defer_completion_first_hint_mismatch": lambda *a, **k: True}, self.d.scheduler)
        self.hold_one_step(r2, {"get_num_new_matched_tokens": lambda *a, **k: (None, False)}, self.d.scheduler.connector)
        self.d.step()
        self.assert_adopted(r2)

    def test_plan_held_across_break_2261_is_reused(self) -> None:
        x = make_request("x", [900 + (i % 7) for i in range(64)])
        self.d.scheduler.add_request(x)  # ahead of r2: the held head
        r2 = self.d.add_turn2(self.r1)
        patches = {"_held_by_step_cells": lambda request, *a, **k: request.request_id == "x",
                   "_load_spares_slot_for": lambda *a, **k: False}
        self.hold_one_step(r2, patches, self.d.scheduler)
        self.d.step()
        self.assert_adopted(r2)

    def test_plan_held_across_break_2280_is_reclaimable(self) -> None:
        r2 = self.d.add_turn2(self.r1)
        self.hold_one_step(r2, {"allocate_slots": lambda *a, **k: None}, self.d.scheduler.kv_cache_manager)
        free_before = self.d.scheduler.kv_cache_manager.block_pool.get_num_free_blocks()
        self.assertEqual(self.d.registry.evict_lru(), len(self.pinned_ids), "the held plan yields its blocks")
        self.assertEqual(self.d.scheduler.kv_cache_manager.block_pool.get_num_free_blocks(),
                         free_before + len(self.pinned_ids))
        self.d.step()
        self.assertEqual(self.d.registry.counters.adopted, 0, "the dropped plan's request loads the stock way")
        self.d.run_until(r2.is_finished, "turn 2 finished on the stock load")
        self.assertEqual(self.d.hashed_blocks(), [])

    def test_a_stop_finish_on_the_deferred_free_path_pins_and_turn2_adopts(self) -> None:
        """Serving runs async scheduling with a kv_both connector, so defer_block_free is on (scheduler.py:521-527). A
        stop-token finish while the next step is in flight frees through KVCacheManager.pop_blocks_for_free
        (scheduler.py:3743-3752), never KVCacheManager.free: on r4f-A graded 239 stop finishes pinned nothing while the det
        drive's length finishes pinned 81. The pin must take the leading full blocks on that path too."""
        sched, reg = self.d.scheduler, self.d.registry
        s1 = make_request("s1", [13 + (i % 47) for i in range(TURN1_TOKENS)])
        sched.add_request(s1)
        self.d.run_until(lambda: len(s1.output_token_ids) >= 1 and not s1.is_finished(), "s1 decoding")
        self.assertEqual(self.d.stop_on_deferred_free(s1), ["s1"], "the free took the deferred branch")
        entry = next((e for e in reg.pins if e.source == "s1"), None)
        self.assertIsNotNone(entry, f"no pin for a deferred free: {reg.counters}")
        held = {b.block_id for b in entry.blocks[0]}
        deferred = {b.block_id for _, blocks in sched.deferred_frees for b in blocks}
        self.assertFalse(held & deferred, "a pinned block is on the deferred free list")
        self.assertTrue(all(b.ref_cnt == 1 for b in entry.blocks[0]), "the pin owns its blocks")
        s2 = make_request("s2", [*s1.all_token_ids, *([5000] * 40)])
        sched.add_request(s2)
        self.d.step()
        k = reg.counters.local_hit_tokens // BLOCK
        self.assertEqual(reg.counters.adopted, 1, reg.counters)
        self.assertTrue(k >= 1 and set(self.d.block_ids("s2")[:k]) <= held, "turn 2 found the pinned prefix")
        self.d.run_until(s2.is_finished, "s2 finished")
        self.assertEqual((reg.counters.errors, self.d.hashed_blocks()), (0, []))

    def test_fresh_single_turn_prompts_age_the_pin_out(self) -> None:
        """r4f det drive: prompts with no stored prefix give the connector hit 0, so lookups stay 0 and a pin only
        costs HBM. After idle_turns later finishes the turn-1 pin is back in the pool and turn 2 loads the stock way."""
        reg = self.d.registry
        reg.idle_turns = 2
        for i in range(2):
            fresh = make_request(f"fresh{i}", [2000 + 100 * i + (j % 13) for j in range(64)])
            self.d.scheduler.add_request(fresh)
            self.d.run_until_freed(fresh, f"fresh prompt {i} finished and freed")
        self.assertEqual(reg.counters.lookups, 0, "no stored prefix: the connector never answers a hit")
        self.assertEqual(reg.counters.evicted_idle, len(self.pinned_ids), "the turn-1 pin aged out whole")
        self.assertNotIn("t1", [entry.source for entry in reg.pins])
        r2 = self.d.add_turn2(self.r1)
        self.d.run_until(r2.is_finished, "turn 2 finished on the stock load")
        self.assertEqual((reg.counters.adopted, reg.counters.errors), (0, 0))
        self.assertEqual(self.d.hashed_blocks(), [])

    def test_hybrid_fail_closed_releases_the_plan(self) -> None:
        connector = self.d.scheduler.connector
        self.assertFalse(connector.supports_divergent_local_hybrid_hits, "no recurrent sidecar on this shape")
        self.d.scheduler.has_mamba_layers = True  # scheduler.py:1999-2010 now zeroes the external hit
        r2 = self.d.add_turn2(self.r1)
        self.d.step()
        reg = self.d.registry
        self.assertGreaterEqual(reg.counters.lookups, 1, "the connector answered a hit and the pin planned it")
        self.assertEqual((reg.plans, reg.counters.adopted), ({}, 0))
        self.assertEqual([b.block_id for b in reg.pins[0].blocks[0]], self.pinned_ids, "the pin went back, whole")
        self.assertFalse(set(self.d.block_ids(r2.request_id)) & set(self.pinned_ids), "no pinned block without a load")
        self.assertEqual(self.d.hashed_blocks(), [])


if __name__ == "__main__":
    unittest.main(verbosity=2)
