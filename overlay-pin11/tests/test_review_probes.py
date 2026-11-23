"""Review probes (Opus, first review of pin-kit eefd6499): each test asserts the SAFE behaviour and is expected to FAIL on eefd6499."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest import mock

KIT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(KIT))
sys.path.insert(0, str(KIT / "tests"))

import fakes  # noqa: E402
import glm53_attn_pin as pin  # noqa: E402
from test_attn_pin import B, PinnedStack  # noqa: E402


class Probes(PinnedStack):
    def test_mark_failure_leaves_no_marker_hash(self) -> None:
        """BLOCKER probe: a failure inside mark_hashes after some markers are set must not leave block_hash on any block."""
        scheduler = self.new_scheduler(num_blocks=64)
        self.finished_turn(scheduler, "r1", "s", full_blocks=10)
        r2 = fakes.Request("r2", fakes.hashes_for("s", 12), 12 * B + 9)
        scheduler.inner.hits["r2"] = 10 * B
        real_mark = pin.AttnPinRegistry.mark_hashes

        def mark_then_fail(reg: pin.AttnPinRegistry, rid: str, blocks: object, cached: int) -> list:
            real_mark(reg, rid, blocks, cached)
            raise IndexError("projection mismatch")

        with mock.patch.object(pin.AttnPinRegistry, "mark_hashes", mark_then_fail):
            scheduler.admit(r2)
        pool = scheduler.kv_cache_manager.block_pool
        leaked = [b.block_id for b in pool.blocks if b.block_hash is not None]
        self.assertEqual(leaked, [], "marker hashes leaked: a later fresh block would skip its load")

    def test_held_plan_blocks_yield_before_preemption(self) -> None:
        """REQUIRED probe: blocks held by a not-yet-injected plan must be reclaimable when an allocation fails."""
        scheduler = self.new_scheduler(num_blocks=64)
        self.finished_turn(scheduler, "r1", "s", full_blocks=10)
        r2 = fakes.Request("r2", fakes.hashes_for("s", 12), 12 * B)
        self.assertEqual(self.registry.plan_adopt(r2, 10 * B), 9)  # planned, allocation not yet run (request still waiting)
        self.assertGreater(self.registry.evict_lru(), 0, "a waiting request's plan pins 10 blocks a running decode cannot reclaim")


class GateProbe(PinnedStack):
    def test_second_waiter_sees_stock_room(self) -> None:
        """REQUIRED probe: pins must not fail the partials-room gate for a waiter that stock (pins free) would pass."""
        from test_attn_pin import TestPressure
        scheduler = TestPressure.pinned_pair_with_partial_prefill(self)  # type: ignore[arg-type]
        small = fakes.Request("w1", fakes.hashes_for("q", 4), 4 * B)
        big = fakes.Request("w2", fakes.hashes_for("p", 12), 12 * B)
        scheduler.waiting.extend([small, big])
        pinned = sum(entry.num_blocks for entry in self.registry.pins)
        reserved = scheduler._inflight_prefill_reserved_blocks()
        stock_free = scheduler.kv_cache_manager.block_pool.get_num_free_blocks() + pinned
        self.assertGreaterEqual(stock_free - reserved, scheduler._request_remaining_blocks(big), "stock admits w2")
        self.registry.make_room_for_waiting(scheduler)
        free = scheduler.kv_cache_manager.block_pool.get_num_free_blocks()
        self.assertGreaterEqual(free - reserved, scheduler._request_remaining_blocks(big), "pins block w2 at the gate")


if __name__ == "__main__":
    unittest.main()
