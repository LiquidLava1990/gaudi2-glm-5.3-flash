"""ATTN-PIN host tests (CPU, no vllm): the registry on the fake stack of fakes.py, flag-off identity, refusals, counters."""
from __future__ import annotations

import io
import os
import re
import sys
import unittest
from collections import deque
from collections import defaultdict
from pathlib import Path
from types import ModuleType
from unittest import mock

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))

import fakes  # noqa: E402
import glm53_attn_pin as pin  # noqa: E402

B = 128
COUNTER_LINE = re.compile(
    r"^GLM53_ATTN_PIN pinned_sessions=\d+ pinned_blocks=\d+ held_plans=\d+ local_hit_tokens=\d+ "
    r"evicted_adopt_trim=\d+ evicted_pressure=\d+ evicted_cap=\d+ evicted_idle=\d+ "
    r"connector_fallbacks=\d+ adopted=\d+ lookups=\d+ pinned=\d+ declined_partial=\d+ errors=\d+$")


def fake_modules() -> dict[str, ModuleType]:
    """The five target modules, as fakes.py classes under the real module names."""
    named = {name: ModuleType(name) for name in pin.TARGET_MODULES}
    sched, kvm, coord, single, offl = (named[name] for name in pin.TARGET_MODULES)
    sched.Scheduler = fakes.Scheduler
    sched.RequestStatus = fakes.RequestStatus
    kvm.KVCacheManager = fakes.KVCacheManager
    coord.KVCacheCoordinator = fakes.KVCacheCoordinator
    single.SingleTypeKVCacheManager = fakes.SingleTypeKVCacheManager
    offl.OffloadingConnectorScheduler = fakes.OffloadingConnectorScheduler
    return named


class PinnedStack(unittest.TestCase):
    """Patches the fake classes once per test and restores them after, so flag-off identity can be checked too."""

    def setUp(self) -> None:
        self.lines: list[str] = []
        self.modules = fake_modules()
        pin.reset_for_tests()
        pin.install_all(self.modules, emit=self.lines.append)
        self.addCleanup(pin.uninstall_all, self.modules)
        self.addCleanup(pin.reset_for_tests)

    def new_scheduler(self, **kwargs) -> fakes.Scheduler:
        scheduler = fakes.Scheduler(**kwargs)
        self.registry = pin.registry()
        self.assertIsNotNone(self.registry)
        return scheduler

    def finished_turn(self, scheduler: fakes.Scheduler, rid: str, prefix: str, full_blocks: int, extra: int = 37,
                      ) -> fakes.Request:
        """A request that ran to full_blocks*B + extra tokens and finished; block_hashes cover the full blocks."""
        request = fakes.Request(rid, fakes.hashes_for(prefix, full_blocks), full_blocks * B + extra)
        fakes.run_prompt_to_end(scheduler, request, full_blocks * B + extra)
        scheduler.finish(request)
        return request

    def assert_every_block_has_one_owner(self, scheduler: fakes.Scheduler) -> None:
        held = [blocks for entry in self.registry.pins for blocks in entry.blocks]
        held += [blocks for plan in self.registry.plans.values() for blocks in plan.entry.blocks]
        self.assertEqual(fakes.ownership_faults(scheduler, held), [])


class TestFlagOff(unittest.TestCase):
    def test_unset_flag_installs_nothing(self) -> None:
        with mock.patch.dict(os.environ, {}, clear=True):
            self.assertFalse(pin.enabled())
            self.assertFalse(pin.install_from_env())
        self.assertFalse(any(isinstance(f, pin._PatchOnImport) for f in sys.meta_path))

    def test_contiguous_pa_with_the_pin_off_is_not_involved(self) -> None:
        with mock.patch.dict(os.environ, {"VLLM_CONTIGUOUS_PA": "1"}, clear=True), \
                mock.patch("sys.stdout", new=io.StringIO()) as out:
            self.assertFalse(pin.install_from_env())
            fakes.Scheduler()  # the stock scheduler: nothing wrapped, nothing refuses
        self.assertEqual(out.getvalue(), "")
        self.assertFalse(any(isinstance(f, pin._PatchOnImport) for f in sys.meta_path))

    def test_only_0_and_1_are_valid_flag_values(self) -> None:
        for value, armed in (("0", False), ("1", True)):
            with mock.patch.dict(os.environ, {pin.FLAG: value}), mock.patch("sys.stdout", new=io.StringIO()) as out:
                self.assertIs(pin.enabled(), armed)
            self.assertEqual(out.getvalue(), "", f"{value!r} is valid: no refusal line")

    def test_an_invalid_flag_value_logs_one_refusal_and_stays_off(self) -> None:
        for value in ("true", "2", " 1", "yes", ""):
            with mock.patch.dict(os.environ, {pin.FLAG: value}), mock.patch("sys.stdout", new=io.StringIO()) as out:
                self.assertFalse(pin.enabled())
                self.assertFalse(pin.install_from_env())
            lines = out.getvalue().splitlines()
            self.assertEqual(lines, [f"GLM53_ATTN_PIN invalid value for {pin.FLAG}={value}, refused"] * 2)

    def test_uninstall_restores_identity(self) -> None:
        modules = fake_modules()
        originals = {(m, n): getattr(getattr(m, c), n) for m, c, n in pin.PATCH_POINTS(modules)}
        pin.install_all(modules, emit=lambda _: None)
        self.assertTrue(all(getattr(getattr(m, c), n) is not originals[(m, n)] for m, c, n in pin.PATCH_POINTS(modules)))
        pin.uninstall_all(modules)
        pin.reset_for_tests()
        self.assertTrue(all(getattr(getattr(m, c), n) is originals[(m, n)] for m, c, n in pin.PATCH_POINTS(modules)))

    def test_only_a_plain_full_attention_manager_pins(self) -> None:
        pool = fakes.BlockPool(8)
        self.assertTrue(pin.pinnable_manager(fakes.FullAttentionManager(pool, B, 0), B))
        self.assertFalse(pin.pinnable_manager(fakes.KpoolTailManager(pool, B, 1), B))
        self.assertFalse(pin.pinnable_manager(fakes.FullAttentionManager(pool, 2 * B, 0), B))
        rswa = type("RSWAManager", (fakes.FullAttentionManager,), {})(pool, B, 0)
        self.assertFalse(pin.pinnable_manager(rswa, B), "a windowed subclass recycles blocks and never pins")

    def test_bad_env_values_refuse(self) -> None:
        with mock.patch.dict(os.environ, {pin.MAX_ENV: "0"}):
            with self.assertRaisesRegex(ValueError, pin.MAX_ENV):
                pin.max_pins()
        with mock.patch.dict(os.environ, {pin.LOG_ENV: "soon"}):
            with self.assertRaisesRegex(ValueError, pin.LOG_ENV):
                pin.log_interval_s()


class TestBootRefusals(PinnedStack):
    def test_prefix_caching_on_refuses(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "prefix caching"):
            fakes.Scheduler(enable_prefix_caching=True)

    def test_no_connector_refuses(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "OffloadingConnector"):
            fakes.Scheduler(with_connector=False)

    def test_armed_line(self) -> None:
        self.new_scheduler()
        self.assertTrue(any(line.startswith("GLM53_ATTN_PIN armed ") for line in self.lines), self.lines)
        armed = next(line for line in self.lines if line.startswith("GLM53_ATTN_PIN armed "))
        self.assertIn("groups=(0,)", armed)
        self.assertIn(f" {pin.FLAG}=1 ", armed)
        self.assertIn(f"block_size={B}", armed)

    def test_armed_line_reports_contiguous_pa_off_at_its_end(self) -> None:
        for env in ({}, {"VLLM_CONTIGUOUS_PA": "0"}, {"VLLM_CONTIGUOUS_PA": "false"}):
            with self.subTest(env=env), mock.patch.dict(os.environ, env, clear=True):
                self.lines.clear()
                self.new_scheduler()
                armed = next(line for line in self.lines if line.startswith("GLM53_ATTN_PIN armed "))
                self.assertTrue(armed.endswith(f" pid={os.getpid()} contiguous_pa=off"), armed)

    def contiguous_pa_refusal(self, value: str) -> str:
        with mock.patch.dict(os.environ, {"VLLM_CONTIGUOUS_PA": value}), self.assertRaises(RuntimeError) as caught:
            fakes.Scheduler()
        return str(caught.exception)

    def test_contiguous_pa_on_refuses_naming_both_keys(self) -> None:
        for value in ("1", "true", "Yes", " ON "):
            with self.subTest(value=value):
                message = self.contiguous_pa_refusal(value)
                for needle in (pin.FLAG, "VLLM_CONTIGUOUS_PA", "defragmentation", "overwrite", "Unset VLLM_CONTIGUOUS_PA"):
                    self.assertIn(needle, message)
        self.assertEqual(self.lines, [], "a refused boot logs no armed line")


class TestHybridR4eGroups(PinnedStack):
    """The r4e-A boot refusal (03:03): GLM-5.3F is hybrid, so Scheduler.block_size is the KDA group's 1048576 while the
    attention (MLA) group and the request hashes use 128. The pin must read its block size from the attention group."""

    def r4e_scheduler(self, groups: tuple[tuple[str, int], ...] = fakes.R4E_GROUPS) -> fakes.Scheduler:
        return self.new_scheduler(num_blocks=64, groups=groups, group_indices=(0,),
                                  scheduler_block_size=fakes.R4E_SCHEDULER_BLOCK_SIZE)

    def test_r4e_groups_arm_on_the_attention_block_size(self) -> None:
        self.r4e_scheduler()
        armed = next(line for line in self.lines if line.startswith("GLM53_ATTN_PIN armed "))
        self.assertIn("groups=(0,) block_size=128 ", armed)
        self.assertEqual(self.registry.block_size, B)

    def test_r4e_groups_pin_and_adopt_the_attention_blocks_only(self) -> None:
        scheduler = self.r4e_scheduler()
        self.finished_turn(scheduler, "r1", "s", full_blocks=10)
        self.assertEqual([len(blocks) for blocks in self.registry.pins[0].blocks], [10], "attention group only")
        r2 = fakes.Request("r2", fakes.hashes_for("s", 12), 12 * B + 9)
        scheduler.inner.hits["r2"] = 10 * B
        self.assertIsNotNone(scheduler.admit(r2))
        self.assertEqual(scheduler.inner.loads[-1].local_blocks, 9)
        self.assertEqual(self.registry.counters.adopted, 1)

    def test_no_attention_group_refuses_by_name(self) -> None:
        with self.assertRaisesRegex(RuntimeError, r"no FullAttention KV group.*KpoolTailManager.*MambaManager"):
            fakes.Scheduler(groups=(("tail", 4), ("mamba", fakes.R4E_SCHEDULER_BLOCK_SIZE)), group_indices=(0,),
                            scheduler_block_size=fakes.R4E_SCHEDULER_BLOCK_SIZE)

    def test_attention_block_size_must_match_the_hash_block_size(self) -> None:
        with self.assertRaisesRegex(RuntimeError, r"attention block size 256 .*hash block size 128"):
            fakes.Scheduler(groups=(("full", 256), ("tail", 4)), group_indices=(0,))


class TestPinOnFinish(PinnedStack):
    def test_normal_finish_pins_full_attention_blocks_only(self) -> None:
        scheduler = self.new_scheduler(num_blocks=64)
        free_before = scheduler.kv_cache_manager.block_pool.get_num_free_blocks()
        self.finished_turn(scheduler, "r1", "s", full_blocks=10)
        pool = scheduler.kv_cache_manager.block_pool
        # 10 full attention blocks held; the partial 11th and the KpoolTail block went back.
        self.assertEqual(free_before - pool.get_num_free_blocks(), 10)
        self.assertEqual(self.registry.counters.pinned, 1)
        entry = self.registry.pins[0]
        self.assertEqual(len(entry.hashes), 10)
        self.assertTrue(all(b.ref_cnt == 1 and b.block_hash is None for b in entry.blocks[0]))
        self.assertNotIn("r1", scheduler.kv_cache_manager.coordinator.single_type_managers[0].req_to_blocks)

    def test_abort_pins_processed_blocks_only(self) -> None:
        scheduler = self.new_scheduler(num_blocks=64)
        request = fakes.Request("r1", fakes.hashes_for("s", 10), 10 * B + 5)
        fakes.run_prompt_to_end(scheduler, request, 10 * B + 5)
        request.num_computed_tokens = 4 * B + 7
        request.num_in_flight_tokens = B
        scheduler.finish(request, fakes.RequestStatus.FINISHED_ABORTED)
        self.assertEqual(len(self.registry.pins[0].hashes), 3)

    def test_short_request_is_not_pinned(self) -> None:
        scheduler = self.new_scheduler(num_blocks=64)
        free_before = scheduler.kv_cache_manager.block_pool.get_num_free_blocks()
        self.finished_turn(scheduler, "r1", "s", full_blocks=0, extra=50)
        self.assertEqual(self.registry.pins, [])
        self.assertEqual(scheduler.kv_cache_manager.block_pool.get_num_free_blocks(), free_before)

    def test_a_deferred_free_pins_and_returns_only_the_rest(self) -> None:
        """Async serving + kv_both: a stop finish with the next step in flight frees via pop_blocks_for_free
        (scheduler.py:3743-3752). r4f-A graded: 239 stop finishes, 0 pins, since only free was wrapped."""
        scheduler = self.new_scheduler(num_blocks=64)
        request = fakes.Request("r1", fakes.hashes_for("s", 10), 10 * B + 37)
        fakes.run_prompt_to_end(scheduler, request, 10 * B + 37)
        request.num_computed_tokens += 1  # the in-flight step's token
        request.num_in_flight_tokens = 1
        request.status = fakes.RequestStatus.FINISHED_STOPPED
        deferred = scheduler.kv_cache_manager.pop_blocks_for_free(request)
        entry = self.registry.pins[0]
        self.assertEqual((entry.source, len(entry.hashes)), ("r1", 10))
        self.assertFalse({id(b) for b in entry.blocks[0]} & {id(b) for b in deferred}, "a pinned block is deferred")
        self.assertEqual(len(deferred), 2, "the partial 11th attention block and the tail block are deferred")
        self.assertEqual(self.registry.turns, 1)

    def test_a_second_free_of_the_same_request_is_not_a_turn(self) -> None:
        scheduler = self.new_scheduler(num_blocks=64)
        request = self.finished_turn(scheduler, "r1", "s", full_blocks=4)
        scheduler.kv_cache_manager.free(request)  # bookkeeping already popped: nothing to pin, no turn
        self.assertEqual((self.registry.turns, self.registry.counters.pinned), (1, 1))

    def test_cap_evicts_lru(self) -> None:
        with mock.patch.dict(os.environ, {pin.MAX_ENV: "2"}):
            scheduler = self.new_scheduler(num_blocks=64)
        for i in range(3):
            self.finished_turn(scheduler, f"r{i}", f"s{i}", full_blocks=4)
        self.assertEqual([e.source for e in self.registry.pins], ["r1", "r2"])
        self.assertEqual(self.registry.counters.evicted_blocks, 4)

    def test_a_failure_before_the_pin_exists_leaves_every_block_to_the_stock_free(self) -> None:
        """pin_on_free took the blocks out of req_to_blocks before the PinEntry existed: an error in between leaked them."""
        scheduler = self.new_scheduler(num_blocks=64)
        pool = scheduler.kv_cache_manager.block_pool
        free_before = pool.get_num_free_blocks()
        clock, calls = self.registry.clock, []

        def clock_fails_once() -> float:  # the first clock read is the PinEntry's created stamp
            calls.append(1)
            if len(calls) == 1:
                raise RuntimeError("injected fault: PinEntry")
            return clock()

        self.registry.clock = clock_fails_once
        self.finished_turn(scheduler, "r1", "s", full_blocks=10)
        self.assertTrue(self.registry.disabled)
        self.assertEqual(pool.get_num_free_blocks(), free_before, "every block goes back through the stock free")
        self.assert_every_block_has_one_owner(scheduler)


class TestAdopt(PinnedStack):
    def test_exact_prefix_adopts_all_but_last_hit_block(self) -> None:
        scheduler = self.new_scheduler(num_blocks=64)
        r1 = self.finished_turn(scheduler, "r1", "s", full_blocks=10)
        pinned_ids = [b.block_id for b in self.registry.pins[0].blocks[0]]
        r2 = fakes.Request("r2", fakes.hashes_for("s", 12), 12 * B + 9)
        scheduler.inner.hits["r2"] = 10 * B
        free_before = scheduler.kv_cache_manager.block_pool.get_num_free_blocks()
        self.assertIsNotNone(scheduler.admit(r2))
        # 9 pinned blocks reused, 1 fresh block for the last hit chunk (worker.py:405 keeps the attention transfer),
        # 1 KpoolTail block, and the 10th pinned block (beyond k) went back to the pool: net 1 block more than the pin held.
        self.assertEqual(free_before - scheduler.kv_cache_manager.block_pool.get_num_free_blocks(), 1 + 1 - 1)
        self.assertEqual(self.registry.counters.evicted_blocks, 1)
        r2_blocks = scheduler.kv_cache_manager.coordinator.single_type_managers[0].req_to_blocks["r2"]
        self.assertEqual([b.block_id for b in r2_blocks[:9]], pinned_ids[:9])
        self.assertEqual(len(r2_blocks), 10)
        load = scheduler.inner.loads[-1]
        self.assertEqual((load.local_blocks, load.loaded_blocks, load.num_cached_tokens), (9, 1, 10 * B))
        self.assertEqual(load.hashed_block_ids, pinned_ids[:9])
        self.assertTrue(all(b.block_hash is None for b in r2_blocks), "marker hashes must be reset after the call")
        self.assertEqual(self.registry.pins, [])
        self.assertEqual(self.registry.counters.adopted, 1)
        self.assertEqual(self.registry.counters.local_hit_tokens, 9 * B)
        self.assertEqual(self.registry.counters.connector_fallbacks, 0)
        self.assertEqual(r2.num_computed_tokens, 10 * B)
        # the 10th pinned block (beyond k) was returned to the pool, once.
        self.assertTrue(all(b.ref_cnt == 1 for b in r2_blocks))
        self.assertEqual(r1.request_id, "r1")

    def test_hash_mismatch_falls_back_to_connector(self) -> None:
        scheduler = self.new_scheduler(num_blocks=64)
        self.finished_turn(scheduler, "r1", "other", full_blocks=10)
        r2 = fakes.Request("r2", fakes.hashes_for("s", 12), 12 * B)
        scheduler.inner.hits["r2"] = 10 * B
        self.assertIsNotNone(scheduler.admit(r2))
        load = scheduler.inner.loads[-1]
        self.assertEqual((load.local_blocks, load.loaded_blocks), (0, 10))
        self.assertEqual(self.registry.counters.connector_fallbacks, 1)
        self.assertEqual(len(self.registry.pins), 1, "a non-matching pin stays pinned")

    def test_partial_prefix_match_adopts_the_common_blocks(self) -> None:
        scheduler = self.new_scheduler(num_blocks=64)
        self.finished_turn(scheduler, "r1", "s", full_blocks=10)
        hashes = fakes.hashes_for("s", 12)
        hashes[6:] = fakes.hashes_for("diverged", 6)
        r2 = fakes.Request("r2", hashes, 12 * B)
        scheduler.inner.hits["r2"] = 6 * B  # the CPU store also only matches 6 chunks
        self.assertIsNotNone(scheduler.admit(r2))
        load = scheduler.inner.loads[-1]
        self.assertEqual((load.local_blocks, load.loaded_blocks), (5, 1))
        self.assertEqual(self.registry.counters.evicted_blocks, 5, "pinned blocks beyond k are returned")

    def test_no_hit_means_no_adopt(self) -> None:
        scheduler = self.new_scheduler(num_blocks=64)
        self.finished_turn(scheduler, "r1", "s", full_blocks=10)
        r2 = fakes.Request("r2", fakes.hashes_for("s", 12), 12 * B)
        self.assertIsNotNone(scheduler.admit(r2, num_new_tokens=12 * B))
        self.assertEqual(scheduler.inner.loads, [])
        self.assertEqual(len(self.registry.pins), 1)
        self.assertEqual(self.registry.counters.lookups, 0)

    def test_single_block_hit_is_not_adopted(self) -> None:
        scheduler = self.new_scheduler(num_blocks=64)
        self.finished_turn(scheduler, "r1", "s", full_blocks=10)
        r2 = fakes.Request("r2", fakes.hashes_for("s", 12), 12 * B)
        scheduler.inner.hits["r2"] = B
        self.assertIsNotNone(scheduler.admit(r2))
        self.assertEqual(scheduler.inner.loads[-1].local_blocks, 0)
        self.assertEqual(len(self.registry.pins), 1)

    def test_chain_of_turns_repins(self) -> None:
        scheduler = self.new_scheduler(num_blocks=128)
        self.finished_turn(scheduler, "r1", "s", full_blocks=10)
        r2 = fakes.Request("r2", fakes.hashes_for("s", 14), 14 * B + 3)
        scheduler.inner.hits["r2"] = 10 * B
        self.assertIsNotNone(scheduler.admit(r2))
        fakes.run_prompt_to_end(scheduler, r2, 14 * B + 3)
        scheduler.finish(r2)
        self.assertEqual([e.source for e in self.registry.pins], ["r2"])
        self.assertEqual(len(self.registry.pins[0].hashes), 14)
        r3 = fakes.Request("r3", fakes.hashes_for("s", 16), 16 * B)
        scheduler.inner.hits["r3"] = 14 * B
        self.assertIsNotNone(scheduler.admit(r3))
        self.assertEqual(scheduler.inner.loads[-1].local_blocks, 13)


SYSTEM_BLOCKS = 156  # the ~20k-token system prompt every conversation starts with
LONG_PIN_BLOCKS, SHORT_PIN_BLOCKS = 700, 300
BIG_POOL = 2048


class TestOwnerOnlyAdoption(PinnedStack):
    """A pin serves exactly one follower, so only the request that continues the pinned conversation may consume it: a
    request sharing less than half of the pin (the common system prompt) leaves it for its owner."""

    def conversation(self, tag: str, blocks: int, head: int = SYSTEM_BLOCKS) -> list[bytes]:
        """Block hashes of a conversation: `head` blocks of the shared system prompt, then blocks only this one has."""
        return fakes.hashes_for("system", head) + fakes.hashes_for(tag, blocks - head)

    def pin_conversation(self, scheduler: fakes.Scheduler, tag: str, blocks: int, head: int = SYSTEM_BLOCKS) -> None:
        tokens = blocks * B + 37
        request = fakes.Request(f"{tag}1", self.conversation(tag, blocks, head), tokens)
        fakes.run_prompt_to_end(scheduler, request, tokens)
        scheduler.finish(request)

    def follow_up(self, tag: str, blocks: int) -> fakes.Request:
        return fakes.Request(f"{tag}2", self.conversation(tag, blocks), blocks * B + 9)

    def partial_share(self, rid: str, shared: int) -> fakes.Request:
        """A 12-block request sharing its first `shared` blocks with conversation "s" and diverging after."""
        return fakes.Request(rid, fakes.hashes_for("s", shared) + fakes.hashes_for("other", 12 - shared), 12 * B)

    def test_a_request_sharing_only_the_system_prompt_leaves_the_pin(self) -> None:
        scheduler = self.new_scheduler(num_blocks=BIG_POOL)
        self.pin_conversation(scheduler, "a", LONG_PIN_BLOCKS)
        owner_pin = self.registry.pins[0]
        stranger = self.follow_up("c", 200)  # the 156-block system prompt, then 44 blocks of its own
        self.assertEqual(self.registry.plan_adopt(stranger, 200 * B), 0)
        self.assertEqual(len(self.registry.pins), 1)
        self.assertIs(self.registry.pins[0], owner_pin)
        self.assertEqual(owner_pin.num_blocks, LONG_PIN_BLOCKS)
        self.assertEqual(self.registry.plans, {})
        c = self.registry.counters
        self.assertEqual((c.connector_fallbacks, c.declined_partial, c.evicted_adopt_trim), (1, 1, 0))

    def test_the_owner_continuing_the_conversation_still_adopts(self) -> None:
        scheduler = self.new_scheduler(num_blocks=BIG_POOL)
        self.pin_conversation(scheduler, "a", LONG_PIN_BLOCKS)
        scheduler.inner.hits["a2"] = LONG_PIN_BLOCKS * B
        self.assertIsNotNone(scheduler.admit(self.follow_up("a", LONG_PIN_BLOCKS + 20)))
        self.assertEqual(scheduler.inner.loads[-1].local_blocks, LONG_PIN_BLOCKS - 1)
        c = self.registry.counters
        self.assertEqual((c.adopted, c.declined_partial, c.connector_fallbacks, c.evicted_adopt_trim), (1, 0, 0, 1))
        self.assertEqual(self.registry.pins, [])

    def test_two_pins_sharing_the_system_prompt_each_go_to_their_own_conversation(self) -> None:
        scheduler = self.new_scheduler(num_blocks=BIG_POOL)
        self.pin_conversation(scheduler, "a", LONG_PIN_BLOCKS)
        self.pin_conversation(scheduler, "b", SHORT_PIN_BLOCKS)
        long_pin = self.registry.pins[0]
        scheduler.inner.hits["b2"] = SHORT_PIN_BLOCKS * B
        self.assertIsNotNone(scheduler.admit(self.follow_up("b", SHORT_PIN_BLOCKS + 20)))
        self.assertEqual(scheduler.inner.loads[-1].local_blocks, SHORT_PIN_BLOCKS - 1)
        self.assertEqual(len(self.registry.pins), 1)
        self.assertIs(self.registry.pins[0], long_pin)
        self.assertEqual((long_pin.num_blocks, self.registry.counters.evicted_adopt_trim), (LONG_PIN_BLOCKS, 1))
        scheduler.inner.hits["a2"] = LONG_PIN_BLOCKS * B
        self.assertIsNotNone(scheduler.admit(self.follow_up("a", LONG_PIN_BLOCKS + 20)))
        self.assertEqual(scheduler.inner.loads[-1].local_blocks, LONG_PIN_BLOCKS - 1)
        self.assertEqual(self.registry.pins, [])
        self.assertEqual(self.registry.counters.declined_partial, 0)

    def test_a_short_prefix_thief_pin_never_shadows_the_pin_the_request_continues(self) -> None:
        scheduler = self.new_scheduler(num_blocks=BIG_POOL)
        self.pin_conversation(scheduler, "a", LONG_PIN_BLOCKS)
        self.pin_conversation(scheduler, "y", 100, head=100)  # a 100-block conversation: all inside the system prompt
        scheduler.inner.hits["y2"] = 100 * B
        # the request shares 156 blocks with A (the longest match, but under half of 700) and all 100 of Y
        self.assertIsNotNone(scheduler.admit(self.follow_up("y", 200)))
        self.assertEqual(scheduler.inner.loads[-1].local_blocks, 99)
        self.assertEqual([entry.source for entry in self.registry.pins], ["a1"])
        c = self.registry.counters
        self.assertEqual((c.declined_partial, c.connector_fallbacks, c.evicted_adopt_trim), (0, 0, 1))

    def test_exactly_half_of_the_pin_adopts(self) -> None:
        scheduler = self.new_scheduler(num_blocks=64)
        self.finished_turn(scheduler, "r1", "s", full_blocks=10)
        self.assertEqual(self.registry.plan_adopt(self.partial_share("r2", 5), 6 * B), 5)  # 5 * 2 == 10
        self.assertEqual(self.registry.counters.declined_partial, 0)

    def test_one_block_under_half_of_the_pin_declines(self) -> None:
        scheduler = self.new_scheduler(num_blocks=64)
        self.finished_turn(scheduler, "r1", "s", full_blocks=13)
        self.assertEqual(self.registry.plan_adopt(self.partial_share("r2", 6), 6 * B), 0)  # 6 * 2 < 13, 7 left over
        self.assertEqual(len(self.registry.pins), 1)
        self.assertEqual(self.registry.counters.declined_partial, 1)

    def test_a_short_pin_missing_only_its_stripped_reasoning_tail_adopts(self) -> None:
        scheduler = self.new_scheduler(num_blocks=64)
        self.finished_turn(scheduler, "r1", "s", full_blocks=8)
        # the next prompt drops the last 5 blocks of r1 (reasoning tokens): 3 of 8 shared, 5 left over
        self.assertEqual(self.registry.plan_adopt(self.partial_share("r2", 3), 6 * B), 3)
        self.assertEqual((self.registry.pins, self.registry.counters.declined_partial), ([], 0))

    def test_a_short_pin_sharing_one_block_still_declines(self) -> None:
        scheduler = self.new_scheduler(num_blocks=64)
        self.finished_turn(scheduler, "r1", "s", full_blocks=8)
        self.assertEqual(self.registry.plan_adopt(self.partial_share("r2", 1), 6 * B), 0)  # 1 * 2 < 8, 7 left over
        self.assertEqual(len(self.registry.pins), 1)
        self.assertEqual(self.registry.counters.declined_partial, 1)

    def test_a_long_pin_missing_only_a_few_tail_blocks_adopts(self) -> None:
        scheduler = self.new_scheduler(num_blocks=BIG_POOL)
        self.pin_conversation(scheduler, "a", LONG_PIN_BLOCKS)
        self.assertEqual(self.registry.plan_adopt(self.follow_up("a", 694), 694 * B), 693)  # 694 of 700 shared
        self.assertEqual((self.registry.pins, self.registry.counters.declined_partial), ([], 0))

    def test_continues_pin_boundaries(self) -> None:
        cases = {
            (8, 3): True,  # a short conversation that lost its reasoning tail: 5 left over
            (11, 5): True,  # exactly TAIL_SLACK_BLOCKS (6) left over, under half
            (12, 5): False,  # 7 left over, under half
            (8, 1): False,
            (700, 156): False,  # the shared system prompt of a stranger
            (700, 694): True,  # both rules pass
            (10, 5): True,  # exactly half
            (3, 0): False,  # nothing shared: never a continuation, however small the pin
        }
        for (pinned, shared), expected in cases.items():
            with self.subTest(pinned=pinned, shared=shared):
                self.assertIs(pin.continues_pin(shared, pinned), expected)

    def test_counter_line_carries_declined_partial_just_before_errors(self) -> None:
        with mock.patch.dict(os.environ, {pin.LOG_ENV: "0"}):
            scheduler = self.new_scheduler(num_blocks=64)
        self.finished_turn(scheduler, "r1", "s", full_blocks=13)
        self.registry.plan_adopt(self.partial_share("r2", 6), 6 * B)
        self.registry.maybe_log(force=True)
        line = [line for line in self.lines if COUNTER_LINE.match(line)][-1]
        self.assertIn("declined_partial=", line)
        self.assertTrue(line.endswith(" lookups=1 pinned=1 declined_partial=1 errors=0"), line)


class TestPressure(PinnedStack):
    def test_alloc_failure_evicts_lru_pin_and_retries(self) -> None:
        scheduler = self.new_scheduler(num_blocks=24)  # 23 usable
        self.finished_turn(scheduler, "r1", "a", full_blocks=8)
        self.finished_turn(scheduler, "r2", "b", full_blocks=8)
        self.assertEqual(scheduler.kv_cache_manager.block_pool.get_num_free_blocks(), 7)
        r3 = fakes.Request("r3", fakes.hashes_for("c", 12), 12 * B)
        self.assertIsNotNone(scheduler.admit(r3, num_new_tokens=12 * B))  # needs 12 + 1 tail > 7 free
        self.assertEqual([e.source for e in self.registry.pins], ["r2"])
        self.assertEqual(self.registry.counters.evicted_blocks, 8)

    def test_alloc_failure_with_no_pins_stays_none(self) -> None:
        scheduler = self.new_scheduler(num_blocks=8)
        r1 = fakes.Request("r1", fakes.hashes_for("a", 12), 12 * B)
        self.assertIsNone(scheduler.admit(r1, num_new_tokens=12 * B))

    def test_pending_plan_is_evictable_once_pins_are_empty(self) -> None:
        scheduler = self.new_scheduler(num_blocks=64)
        self.finished_turn(scheduler, "r1", "s", full_blocks=10)
        r2 = fakes.Request("r2", fakes.hashes_for("s", 12), 12 * B)
        self.assertEqual(self.registry.plan_adopt(r2, 10 * B), 9)
        self.assertEqual(self.registry.pins, [])
        self.assertEqual(self.registry.held_plans, 1)
        self.assertEqual(self.registry.evict_lru(), 10, "the held plan's blocks go back to the pool")
        self.assertEqual(self.registry.discount("r2"), 0, "a dropped plan discounts nothing")
        self.assertEqual(self.registry.evict_lru(), 0)
        scheduler.requests["r2"] = r2
        scheduler.finish(r2, fakes.RequestStatus.FINISHED_ABORTED)
        self.assertEqual(self.registry.pins, [], "a dropped plan never comes back")

    def test_unadmitted_plan_returns_on_free(self) -> None:
        scheduler = self.new_scheduler(num_blocks=64)
        self.finished_turn(scheduler, "r1", "s", full_blocks=10)
        r2 = fakes.Request("r2", fakes.hashes_for("s", 12), 12 * B)
        self.assertEqual(self.registry.plan_adopt(r2, 10 * B), 9)
        scheduler.requests["r2"] = r2
        scheduler.finish(r2, fakes.RequestStatus.FINISHED_ABORTED)
        self.assertEqual([e.source for e in self.registry.pins], ["r1"])

    def test_held_plan_is_dropped_oldest_first_and_its_request_loads_stock(self) -> None:
        scheduler = self.new_scheduler(num_blocks=24)  # 23 usable
        self.finished_turn(scheduler, "r1", "a", full_blocks=8)
        self.finished_turn(scheduler, "r2", "b", full_blocks=8)
        w1 = fakes.Request("w1", fakes.hashes_for("a", 10), 10 * B)
        w2 = fakes.Request("w2", fakes.hashes_for("b", 10), 10 * B)
        self.registry.plan_adopt(w1, 8 * B)  # both waiters planned, then the waiting loop broke (scheduler.py:2280)
        self.registry.plan_adopt(w2, 8 * B)
        r3 = fakes.Request("r3", fakes.hashes_for("c", 12), 12 * B)
        self.assertIsNotNone(scheduler.admit(r3, num_new_tokens=12 * B), "13 blocks > 7 free: plans must yield")
        self.assertEqual(list(self.registry.plans), ["w2"], "the oldest plan (w1) went first")
        scheduler.inner.hits["w1"] = 8 * B
        self.assertIsNotNone(scheduler.admit(w1))
        self.assertEqual(scheduler.inner.loads[-1].local_blocks, 0, "the dropped plan's request loads the stock way")

    def test_zeroed_external_hit_releases_the_plan(self) -> None:
        """scheduler.py:1999-2010: the connector answered a hit (the plan exists), then the scheduler zeroed it."""
        scheduler = self.new_scheduler(num_blocks=64)
        self.finished_turn(scheduler, "r1", "s", full_blocks=10)
        r2 = fakes.Request("r2", fakes.hashes_for("s", 12), 12 * B)
        self.assertEqual(self.registry.plan_adopt(r2, 10 * B), 9)
        free_before = scheduler.kv_cache_manager.block_pool.get_num_free_blocks()
        blocks = scheduler.kv_cache_manager.allocate_slots(r2, 12 * B, num_external_computed_tokens=0)
        self.assertIsNotNone(blocks)
        self.assertEqual(self.registry.plans, {})
        self.assertEqual([e.source for e in self.registry.pins], ["r1"], "the pin went back, untouched")
        self.assertEqual(free_before - scheduler.kv_cache_manager.block_pool.get_num_free_blocks(), 12 + 1,
                         "no discount: the request allocated every block it computes")
        r2_blocks = scheduler.kv_cache_manager.coordinator.single_type_managers[0].req_to_blocks["r2"]
        self.assertFalse({b.block_id for b in r2_blocks} & {b.block_id for b in self.registry.pins[0].blocks[0]})

    def test_settle_clamps_the_plan_to_a_shorter_external_hit(self) -> None:
        scheduler = self.new_scheduler(num_blocks=64)
        self.finished_turn(scheduler, "r1", "s", full_blocks=10)
        r2 = fakes.Request("r2", fakes.hashes_for("s", 12), 12 * B)
        self.assertEqual(self.registry.plan_adopt(r2, 10 * B), 9)
        self.assertEqual(self.registry.settle_plan("r2", 4 * B), 3)
        self.assertEqual(self.registry.discount("r2"), 3)

    def pinned_pair_with_partial_prefill(self) -> fakes.Scheduler:
        """31 usable blocks: pins r1 (8) + r2 (8), an in-flight partial prefill r0 holding 3 and reserving 2 more."""
        scheduler = self.new_scheduler(num_blocks=32)
        self.finished_turn(scheduler, "r1", "a", full_blocks=8)
        self.finished_turn(scheduler, "r2", "b", full_blocks=8)
        partial = fakes.Request("r0", fakes.hashes_for("z", 4), 4 * B)
        fakes.run_prompt_to_end(scheduler, partial, 2 * B)
        partial.num_tokens = 4 * B
        scheduler._inflight_prefills.append(partial)
        self.assertEqual(scheduler.kv_cache_manager.block_pool.get_num_free_blocks(), 12)
        self.assertEqual(scheduler._inflight_prefill_reserved_blocks(), 2)
        return scheduler

    def test_make_room_for_waiting_evicts_the_non_matching_pin_first(self) -> None:
        scheduler = self.pinned_pair_with_partial_prefill()
        waiter = fakes.Request("r3", fakes.hashes_for("b", 12), 12 * B)  # needs 13 > 12 - 2 reserved
        scheduler.waiting.append(waiter)
        scheduler.schedule()
        self.assertEqual([e.source for e in self.registry.pins], ["r2"], "the pin r3 would adopt survives")
        self.assertEqual(scheduler.kv_cache_manager.block_pool.get_num_free_blocks(), 20)

    def test_make_room_for_waiting_yields_the_matching_pin_last_rather_than_starve(self) -> None:
        scheduler = self.pinned_pair_with_partial_prefill()
        waiter = fakes.Request("r3", fakes.hashes_for("b", 25), 25 * B)  # needs 26: both pins must go
        scheduler.waiting.append(waiter)
        scheduler.schedule()
        self.assertEqual(self.registry.pins, [])
        self.assertEqual(self.registry.counters.errors, 0)

    def test_make_room_drops_a_held_plan_once_pins_are_gone(self) -> None:
        scheduler = self.pinned_pair_with_partial_prefill()
        held = fakes.Request("w0", fakes.hashes_for("b", 10), 10 * B)
        self.assertEqual(self.registry.plan_adopt(held, 8 * B), 7)  # holds r2's 8 blocks, request still waiting
        big = fakes.Request("w1", fakes.hashes_for("q", 25), 25 * B)  # needs 26: stock (12 + 16 - 2 = 26) admits it
        scheduler.waiting.extend([held, big])
        scheduler.schedule()
        self.assertEqual((self.registry.pins, self.registry.plans), ([], {}))
        self.assertEqual(scheduler.kv_cache_manager.block_pool.get_num_free_blocks() - 2, 26)

    def test_make_room_sees_a_waiter_parked_in_the_skipped_queue(self) -> None:
        """Overflow stall 2026-11-22 18:03: five deferred prompts sat in skipped_waiting, the pins kept 10,573 blocks, nothing ran."""
        scheduler = self.pinned_pair_with_partial_prefill()
        scheduler.skipped_waiting = deque([fakes.Request("r3", fakes.hashes_for("q", 12), 12 * B)])  # needs 13 > 12 - 2
        scheduler.schedule()
        self.assertGreaterEqual(scheduler.kv_cache_manager.block_pool.get_num_free_blocks() - 2, 13)
        self.assertGreater(self.registry.counters.evicted_pressure, 0)

    def test_make_room_sees_a_preempted_waiter(self) -> None:
        scheduler = self.pinned_pair_with_partial_prefill()
        waiter = fakes.Request("r3", fakes.hashes_for("q", 12), 12 * B, status=fakes.RequestStatus.PREEMPTED)
        scheduler.skipped_waiting = deque([waiter])
        scheduler.schedule()
        self.assertGreaterEqual(scheduler.kv_cache_manager.block_pool.get_num_free_blocks() - 2, 13)

    def test_make_room_sees_a_waiter_behind_eight_blocked_loads(self) -> None:
        scheduler = self.pinned_pair_with_partial_prefill()
        loading = [fakes.Request(f"l{i}", fakes.hashes_for(f"l{i}", 2), 2 * B, status=fakes.RequestStatus.WAITING_FOR_REMOTE_KVS)
                   for i in range(8)]
        scheduler.skipped_waiting = deque([*loading, fakes.Request("r3", fakes.hashes_for("q", 12), 12 * B)])
        scheduler.schedule()
        self.assertGreaterEqual(scheduler.kv_cache_manager.block_pool.get_num_free_blocks() - 2, 13)

    def test_make_room_ignores_a_blocked_load_on_its_own(self) -> None:
        scheduler = self.pinned_pair_with_partial_prefill()
        scheduler.skipped_waiting = deque([fakes.Request("l0", fakes.hashes_for("q", 12), 12 * B,
                                                         status=fakes.RequestStatus.WAITING_FOR_REMOTE_KVS)])
        scheduler.schedule()
        self.assertEqual([e.source for e in self.registry.pins], ["r1", "r2"])

    def test_make_room_uses_the_gate_list_of_the_scheduler_when_it_has_one(self) -> None:
        scheduler = self.pinned_pair_with_partial_prefill()
        scheduler.waiting.append(fakes.Request("r3", fakes.hashes_for("q", 12), 12 * B))
        scheduler._glm53_waiting_prompts = lambda: []  # paused, stale output or no recurrent slot: the gate scans nobody
        scheduler.schedule()
        self.assertEqual([e.source for e in self.registry.pins], ["r1", "r2"])

    def test_make_room_leaves_pins_when_stock_admits_no_waiter(self) -> None:
        scheduler = self.pinned_pair_with_partial_prefill()
        scheduler.waiting.append(fakes.Request("w1", fakes.hashes_for("q", 40), 40 * B))  # 41 > 26: stock refuses too
        scheduler.schedule()
        self.assertEqual([e.source for e in self.registry.pins], ["r1", "r2"])

    def test_make_room_is_idle_without_a_reservation(self) -> None:
        scheduler = self.new_scheduler(num_blocks=24)
        self.finished_turn(scheduler, "r1", "a", full_blocks=8)
        self.finished_turn(scheduler, "r2", "b", full_blocks=8)
        scheduler.waiting.append(fakes.Request("r3", fakes.hashes_for("c", 12), 12 * B))
        scheduler.schedule()
        self.assertEqual([e.source for e in self.registry.pins], ["r1", "r2"], "allocate_slots' retry frees on demand")

    def test_reset_cache_releases_every_pin(self) -> None:
        scheduler = self.new_scheduler(num_blocks=64)
        self.finished_turn(scheduler, "r1", "a", full_blocks=8)
        free_before = scheduler.kv_cache_manager.block_pool.get_num_free_blocks()
        scheduler.inner.reset_cache()
        self.assertEqual(self.registry.pins, [])
        self.assertEqual(scheduler.kv_cache_manager.block_pool.get_num_free_blocks(), free_before + 8)
        self.assertEqual(scheduler.inner.resets, 1)

    def held_pin_and_plan(self) -> fakes.Scheduler:
        """64 blocks: pins r1 (8) and r2 (8), r2's consumed into a pending plan for a waiter not allocated yet."""
        scheduler = self.new_scheduler(num_blocks=64)
        self.finished_turn(scheduler, "r1", "a", full_blocks=8)
        self.finished_turn(scheduler, "r2", "b", full_blocks=8)
        self.assertEqual(self.registry.plan_adopt(fakes.Request("w1", fakes.hashes_for("b", 10), 10 * B), 8 * B), 7)
        return scheduler

    def test_reset_prefix_cache_releases_pins_and_plans_before_the_kv_reset(self) -> None:
        """scheduler.py:3849 resets the KV manager before the connector and block_pool.py:773-780 refuses while a block
        is in use: with pins held the reset returned False, and with reset_connector=False the pins were never freed."""
        scheduler = self.held_pin_and_plan()
        self.assertTrue(scheduler.reset_prefix_cache(), "the KV reset refused: pinned blocks still in use")
        self.assertEqual((self.registry.pins, self.registry.plans), ([], {}))
        self.assertEqual(scheduler.kv_cache_manager.block_pool.get_num_free_blocks(), 63)
        self.assertEqual((scheduler.inner.resets, self.registry.counters.errors), (0, 0))

    def test_reset_prefix_cache_with_running_requests_preempts_and_succeeds(self) -> None:
        """reset_running_requests=True raised RuntimeError (scheduler.py:3850-3856) whenever a pin was held."""
        scheduler = self.held_pin_and_plan()
        running = fakes.Request("r3", fakes.hashes_for("c", 4), 4 * B)
        fakes.run_prompt_to_end(scheduler, running, 4 * B)
        scheduler.running.append(running)
        self.assertTrue(scheduler.reset_prefix_cache(reset_running_requests=True, reset_connector=True))
        self.assertEqual(list(scheduler.waiting), [running], "the running request was preempted back to waiting")
        self.assertEqual(self.registry.pins, [], "a preempted request is not pinned")
        self.assertEqual(scheduler.kv_cache_manager.block_pool.get_num_free_blocks(), 63)
        self.assertEqual(scheduler.inner.resets, 1)


class TestIdleBound(PinnedStack):
    """r4f-A/B 2026-11-09: the det drive (fresh prompts, connector hit 0 by design) left 64 pins = 10,408 blocks (31% of
    KV) held into the graded run with lookups=0. A pin that sees HPU_GLM53_ATTN_PIN_IDLE_TURNS later finishes without a
    hit goes back to the pool."""

    def idle_scheduler(self, idle_turns: int, num_blocks: int = 128) -> fakes.Scheduler:
        with mock.patch.dict(os.environ, {pin.IDLE_ENV: str(idle_turns)}):
            return self.new_scheduler(num_blocks=num_blocks)

    def test_single_turn_traffic_never_holds_more_than_idle_turns_pins(self) -> None:
        scheduler = self.idle_scheduler(2)
        free_at_boot = scheduler.kv_cache_manager.block_pool.get_num_free_blocks()
        for i in range(6):
            self.finished_turn(scheduler, f"r{i}", f"fresh{i}", full_blocks=4)
        self.assertEqual([e.source for e in self.registry.pins], ["r4", "r5"])
        self.assertEqual(self.registry.counters.evicted_idle, 4 * 4)
        self.assertEqual(self.registry.counters.lookups, 0)
        self.assertEqual(scheduler.kv_cache_manager.block_pool.get_num_free_blocks(), free_at_boot - 2 * 4)

    def test_a_pin_adopted_within_the_window_is_found_on_the_next_turn(self) -> None:
        scheduler = self.idle_scheduler(2)
        self.finished_turn(scheduler, "r1", "s", full_blocks=10)
        self.finished_turn(scheduler, "other", "o", full_blocks=3)  # one later finish: r1 is 1 turn old, kept
        r2 = fakes.Request("r2", fakes.hashes_for("s", 12), 12 * B + 9)
        scheduler.inner.hits["r2"] = 10 * B
        self.assertIsNotNone(scheduler.admit(r2))
        self.assertEqual(scheduler.inner.loads[-1].local_blocks, 9)
        self.assertEqual((self.registry.counters.adopted, self.registry.counters.evicted_idle), (1, 0))

    def test_an_expired_pin_falls_back_to_the_connector(self) -> None:
        scheduler = self.idle_scheduler(1)
        self.finished_turn(scheduler, "r1", "s", full_blocks=10)
        self.finished_turn(scheduler, "other", "o", full_blocks=3)  # r1 saw 1 later finish with no hit: released
        r2 = fakes.Request("r2", fakes.hashes_for("s", 12), 12 * B + 9)
        scheduler.inner.hits["r2"] = 10 * B
        self.assertIsNotNone(scheduler.admit(r2))
        self.assertEqual(scheduler.inner.loads[-1].local_blocks, 0)
        self.assertEqual(self.registry.counters.evicted_idle, 10)
        self.assertEqual(self.registry.counters.connector_fallbacks, 1)

    def test_a_returned_plan_keeps_its_age(self) -> None:
        scheduler = self.idle_scheduler(2)
        self.finished_turn(scheduler, "r1", "s", full_blocks=10)
        r2 = fakes.Request("r2", fakes.hashes_for("s", 12), 12 * B)
        scheduler.inner.hits["r2"] = 10 * B
        self.registry.plan_adopt(r2, 10 * B)  # planned, never allocated
        scheduler.finish(r2, fakes.RequestStatus.FINISHED_ABORTED)  # the plan goes back; r2 held no KV: no turn
        self.assertEqual(([e.source for e in self.registry.pins], self.registry.turns), (["r1"], 1))
        self.finished_turn(scheduler, "o1", "o1", full_blocks=3)  # turn 2: r1 is 1 turn old, kept
        self.finished_turn(scheduler, "o2", "o2", full_blocks=3)  # turn 3: r1 is 2 turns old, released
        self.assertEqual([e.source for e in self.registry.pins], ["o1", "o2"])
        self.assertEqual(self.registry.counters.evicted_idle, 10)

    def test_a_declined_pin_is_still_freed_by_idle_eviction(self) -> None:
        scheduler = self.idle_scheduler(2)
        free_at_boot = scheduler.kv_cache_manager.block_pool.get_num_free_blocks()
        self.finished_turn(scheduler, "r1", "s", full_blocks=13)
        stranger = fakes.Request("r2", fakes.hashes_for("s", 6) + fakes.hashes_for("other", 6), 12 * B)
        self.assertEqual(self.registry.plan_adopt(stranger, 6 * B), 0)  # 6 * 2 < 13, 7 left over: declined, pin kept
        self.assertEqual(([e.source for e in self.registry.pins], self.registry.counters.declined_partial), (["r1"], 1))
        self.finished_turn(scheduler, "o1", "o1", full_blocks=3)  # turn 2: r1 is 1 turn old, kept
        self.finished_turn(scheduler, "o2", "o2", full_blocks=3)  # turn 3: r1 is 2 turns old, released
        self.assertEqual([e.source for e in self.registry.pins], ["o1", "o2"])
        self.assertEqual(self.registry.counters.evicted_idle, 13)
        self.assertEqual(scheduler.kv_cache_manager.block_pool.get_num_free_blocks(), free_at_boot - 2 * 3)

    def test_idle_turns_is_on_the_armed_line_and_a_bad_value_refuses(self) -> None:
        self.idle_scheduler(5)
        armed = next(line for line in self.lines if line.startswith("GLM53_ATTN_PIN armed "))
        self.assertIn(" idle_turns=5 ", armed)
        with mock.patch.dict(os.environ, {pin.IDLE_ENV: "0"}):
            with self.assertRaisesRegex(ValueError, pin.IDLE_ENV):
                pin.idle_turns()
        with mock.patch.dict(os.environ, {}, clear=True):
            self.assertEqual(pin.idle_turns(), pin.DEFAULT_IDLE_TURNS)


class TestCountersAndFailure(PinnedStack):
    def test_counter_line_format_and_cadence(self) -> None:
        with mock.patch.dict(os.environ, {pin.LOG_ENV: "0"}):
            scheduler = self.new_scheduler(num_blocks=64)
        self.finished_turn(scheduler, "r1", "s", full_blocks=10)
        counter_lines = [line for line in self.lines if COUNTER_LINE.match(line)]
        self.assertTrue(counter_lines, self.lines)
        self.assertIn("pinned_sessions=1 pinned_blocks=10 held_plans=0 local_hit_tokens=0 evicted_adopt_trim=0 "
                      "evicted_pressure=0 evicted_cap=0 evicted_idle=0 connector_fallbacks=0", counter_lines[-1])

    def test_evictions_are_split_by_reason(self) -> None:
        with mock.patch.dict(os.environ, {pin.MAX_ENV: "1"}):
            scheduler = self.new_scheduler(num_blocks=64)
        self.finished_turn(scheduler, "r0", "x", full_blocks=3)
        self.finished_turn(scheduler, "r1", "s", full_blocks=10)  # the cap evicts r0 (3)
        r2 = fakes.Request("r2", fakes.hashes_for("s", 12), 12 * B)
        scheduler.inner.hits["r2"] = 6 * B
        scheduler.admit(r2)  # k = 5: 5 pinned blocks beyond k are trimmed
        self.registry.release_all()
        c = self.registry.counters
        self.assertEqual((c.evicted_cap, c.evicted_adopt_trim, c.evicted_pressure), (3, 5, 0))

    def test_shutdown_flushes_a_final_counter_line(self) -> None:
        scheduler = self.new_scheduler(num_blocks=64)
        self.finished_turn(scheduler, "r1", "s", full_blocks=10)
        before = len([line for line in self.lines if COUNTER_LINE.match(line)])
        scheduler.shutdown()
        lines = [line for line in self.lines if COUNTER_LINE.match(line)]
        self.assertEqual(len(lines), before + 1)
        self.assertIn("pinned_sessions=1 pinned_blocks=10 ", lines[-1])
        self.assertTrue(scheduler.shut_down, "the stock shutdown still runs")

    def test_pin_failure_latches_off_and_releases(self) -> None:
        scheduler = self.new_scheduler(num_blocks=64)
        self.finished_turn(scheduler, "r1", "s", full_blocks=10)
        free_before = scheduler.kv_cache_manager.block_pool.get_num_free_blocks()
        r2 = fakes.Request("r2", fakes.hashes_for("s", 12), 12 * B)
        scheduler.inner.hits["r2"] = 10 * B
        with mock.patch.object(pin.AttnPinRegistry, "match", side_effect=KeyError("boom")):
            self.assertIsNotNone(scheduler.admit(r2))
        self.assertTrue(self.registry.disabled)
        self.assertEqual(self.registry.counters.errors, 1)
        self.assertEqual(scheduler.kv_cache_manager.block_pool.get_num_free_blocks() - free_before, 10 - 10 - 1,
                         "pins released (+10), r2 allocated 10 + tail (-11)")
        self.assertEqual(scheduler.inner.loads[-1].local_blocks, 0, "stock load path after the latch")
        self.assertTrue(any("GLM53_ATTN_PIN disabled" in line for line in self.lines), self.lines)
        fakes.run_prompt_to_end(scheduler, r2, 12 * B)
        scheduler.finish(r2)
        self.assertEqual(self.registry.pins, [], "nothing is pinned once latched off")

    def adopt_setup(self) -> tuple[fakes.Scheduler, fakes.Request]:
        scheduler = self.new_scheduler(num_blocks=64)
        self.finished_turn(scheduler, "r1", "s", full_blocks=10)
        r2 = fakes.Request("r2", fakes.hashes_for("s", 12), 12 * B + 9)
        scheduler.inner.hits["r2"] = 10 * B
        return scheduler, r2

    def hashed_blocks(self, scheduler: fakes.Scheduler) -> list[int]:
        return [b.block_id for b in scheduler.kv_cache_manager.block_pool.blocks if b.block_hash is not None]

    def test_markers_are_reset_when_the_pin_latches_during_the_stock_call(self) -> None:
        scheduler, r2 = self.adopt_setup()
        scheduler.inner.during_update = lambda: setattr(self.registry, "disabled", True)  # _hook now skips
        self.assertIsNotNone(scheduler.admit(r2))
        self.assertEqual(scheduler.inner.loads[-1].local_blocks, 9, "the stock call saw the markers")
        self.assertEqual(self.hashed_blocks(scheduler), [], "the wrapper's finally resets the markers itself")

    def test_a_failure_mid_mark_resets_every_marker_it_set(self) -> None:
        scheduler, r2 = self.adopt_setup()
        real_mark_group = pin.AttnPinRegistry._mark_group

        def mark_then_fail(reg: pin.AttnPinRegistry, blocks: list, hashes: tuple) -> None:
            real_mark_group(reg, blocks, hashes)
            raise IndexError("group mismatch")

        with mock.patch.object(pin.AttnPinRegistry, "_mark_group", mark_then_fail):
            self.assertIsNotNone(scheduler.admit(r2))
        self.assertTrue(self.registry.disabled)
        self.assertEqual(self.hashed_blocks(scheduler), [])
        self.assertEqual(scheduler.inner.loads[-1].local_blocks, 0, "the stock call loads every block")

    def test_a_block_hashed_before_marking_latches_off(self) -> None:
        scheduler, r2 = self.adopt_setup()
        stale = self.registry.pins[0].blocks[0][4]  # an injected block that already carries a hash
        stale._block_hash = b"stale"
        self.assertIsNotNone(scheduler.admit(r2))
        self.assertTrue(self.registry.disabled)
        self.assertTrue(any("carry a block hash before marking" in line for line in self.lines), self.lines)
        self.assertEqual(self.hashed_blocks(scheduler), [stale.block_id], "no marker added beside the stale hash")

    def test_a_failure_after_the_handover_never_frees_the_requests_blocks(self) -> None:
        """inject_after_local extended req_to_blocks before it booked the plan injected: an error in between made the
        latch return the request's own blocks to the pool (two owners) and free the trimmed block twice."""
        scheduler, r2 = self.adopt_setup()
        real_add_evicted = pin.PinCounters.add_evicted

        def trim_booking_fails(counters: pin.PinCounters, reason: str, blocks: int) -> None:
            if reason == pin.EVICT_TRIM:
                raise RuntimeError("injected fault: trim booking")
            real_add_evicted(counters, reason, blocks)

        with mock.patch.object(pin.PinCounters, "add_evicted", trim_booking_fails):
            self.assertIsNotNone(scheduler.admit(r2))
        self.assertTrue(self.registry.disabled)
        self.assertEqual((self.registry.pins, self.registry.plans), ([], {}))
        self.assert_every_block_has_one_owner(scheduler)

    def test_a_failure_on_a_later_pinned_group_frees_its_blocks(self) -> None:
        """Two pinned groups: the plan was booked injected after the first, so a failure handing over the second left
        that group's pinned blocks owned by nobody."""
        scheduler = self.new_scheduler(num_blocks=64, groups=(("full", B), ("full", B), ("tail", B)))
        self.finished_turn(scheduler, "r1", "s", full_blocks=8)
        self.assertEqual([len(blocks) for blocks in self.registry.pins[0].blocks], [8, 8])
        r2 = fakes.Request("r2", fakes.hashes_for("s", 10), 10 * B)
        scheduler.inner.hits["r2"] = 8 * B
        scheduler.kv_cache_manager.coordinator.single_type_managers[1].req_to_blocks = defaultdict(fakes.ExtendFailsOnce)
        self.assertIsNotNone(scheduler.admit(r2))
        self.assertTrue(self.registry.disabled)
        self.assertEqual((self.registry.pins, self.registry.plans), ([], {}))
        self.assert_every_block_has_one_owner(scheduler)
        self.assertEqual(scheduler.inner.loads[-1].local_blocks, 0, "the stock load path after the latch")


if __name__ == "__main__":
    unittest.main()
