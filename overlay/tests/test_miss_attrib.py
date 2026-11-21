"""Step 2: miss attribution, log only. Fake managers record every call, so "no behaviour change" is a call-log equality."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent))

from overlay_testlib import OVERLAY_ROOT, SCHEDULER_FILE, load_base, load_overlay  # noqa: E402

sys.path.append(str(OVERLAY_ROOT))

from vllm.v1.kv_offload.base import ReqContext, make_offload_key  # noqa: E402
from vllm.v1.kv_offload.cpu.manager import CPUOffloadingManager  # noqa: E402

TOKENS_PER_HASH = 128
RECURRENT_GROUP = 0xFFFFFFFF
FLAG = "VLLM_OFFLOAD_MISS_ATTRIB"
LANE_BLOCK = "VLLM_OFFLOAD_MISS_ATTRIB_LANE_BLOCK"
CONTEXT = ReqContext(req_id="seed", kv_transfer_params=None)


def block_hashes(lane: str, blocks: int, start: int = 0) -> list[bytes]:
    return [f"{lane}:{index}".encode().ljust(32, b"\0") for index in range(start, blocks)]


class RecordingManager(CPUOffloadingManager):
    """The real manager; every call that could change or read its state lands in ``calls``."""

    def __init__(self, num_blocks: int) -> None:
        super().__init__(num_blocks=num_blocks, cache_policy="lru", enable_events=False, store_threshold=1,
                         bytes_per_block=1)
        self.calls: list[tuple] = []

    def lookup(self, key, req_context):
        self.calls.append(("lookup", key))
        return super().peek(key)

    def touch(self, keys, req_context):
        self.calls.append(("touch", tuple(keys)))
        super().touch(keys, req_context)

    def peek(self, key):
        self.calls.append(("peek", key))
        return super().peek(key)

    def seed(self, key) -> None:
        output = self.prepare_store((key,), CONTEXT)
        self.complete_store(output.keys_to_store, CONTEXT)

    def order(self) -> list:
        return list(self._policy.evictable_blocks)


class Rig:
    """One OffloadingConnectorScheduler without __init__: real managers and request state, a scripted attention walk."""

    def __init__(self, module: ModuleType, attention_hits: list[int | None], flag: str | None = None,
                 min_tokens: int = 0, recurrent_blocks: int = 2, attention_blocks: int = 2,
                 lane_block: str | None = None) -> None:
        self.module, self.lines = module, []
        self.attention = RecordingManager(attention_blocks)
        self.recurrent = RecordingManager(recurrent_blocks)
        self.scheduler = scheduler = object.__new__(module.OffloadingConnectorScheduler)
        group = module.GroupOffloadConfig(0, TOKENS_PER_HASH, TOKENS_PER_HASH, 1, None, None)
        scheduler.config = module.SchedulerOffloadConfig((group,), 1, TOKENS_PER_HASH, 4, True, False)
        scheduler.manager, scheduler._recurrent_manager = self.attention, self.recurrent
        scheduler.recurrent_sidecar_ready = True
        scheduler._req_status = {}
        scheduler._cold_reuse_min_tokens = min_tokens
        scheduler._connector_stats = module.OffloadingConnectorStats()
        scheduler._checkpoints = mock.MagicMock(active=False)
        self.hits = iter(attention_hits)
        scheduler._lookup = lambda req_status: next(self.hits)
        scheduler._touch = lambda req_status: self.attention.calls.append(("sched_touch",))
        self.environ = {key: value for key, value in ((FLAG, flag), (LANE_BLOCK, lane_block)) if value is not None}
        with mock.patch.dict("os.environ", self.environ, clear=False):
            scheduler._wire_checkpoint_removal_listeners()
        attrib = getattr(scheduler, "_miss_attrib", None)
        if attrib is not None:
            attrib.emit = self.lines.append

    def request(self, request_id: str, hashes: list[bytes], prompt_tokens: int, skip: bool = False) -> SimpleNamespace:
        request = SimpleNamespace(request_id=request_id, block_hashes=hashes, num_prompt_tokens=prompt_tokens,
                                  num_tokens=prompt_tokens, kv_transfer_params=None, skip_reading_prefix_cache=skip,
                                  is_finished=lambda: False)
        context = self.module._create_req_context(request)
        self.scheduler._req_status[request_id] = self.module.RequestOffloadState(
            config=self.scheduler.config, req=request, req_context=context,
            offloading_context=self.attention.on_new_request(context))
        return request

    def match(self, request: SimpleNamespace, computed: int = 0) -> tuple[int | None, bool]:
        return self.scheduler.get_num_new_matched_tokens(request, computed)

    def recurrent_key(self, hashes: list[bytes], boundary: int) -> bytes:
        return make_offload_key(hashes[boundary // TOKENS_PER_HASH - 1], RECURRENT_GROUP)

    def attention_key(self, hashes: list[bytes], boundary: int) -> bytes:
        return make_offload_key(hashes[boundary // TOKENS_PER_HASH - 1], 0)


def field(line: str, name: str) -> str:
    return next(part.split("=", 1)[1] for part in line.split() if part.startswith(name + "="))


class MissClasses(unittest.TestCase):
    def setUp(self) -> None:
        self.module = load_overlay(SCHEDULER_FILE)
        self.hashes = block_hashes("lane-a", 100)

    def rig(self, *hits: int | None, **kwargs) -> Rig:
        return Rig(self.module, list(hits), flag="1", **kwargs)

    def test_full_hit_is_class_none(self) -> None:
        rig = self.rig(1280)
        rig.recurrent.seed(rig.recurrent_key(self.hashes, 1280))
        self.assertEqual(rig.match(rig.request("r", self.hashes, 12800)), (1280, True))
        line = rig.lines[0]
        self.assertTrue(line.startswith("GLM53_MISS "))
        self.assertEqual((field(line, "class"), field(line, "backtrack"), field(line, "recurrent_hit")), ("none", "0", "1280"))
        self.assertEqual((field(line, "attention_hit"), field(line, "prompt")), ("1280", "12800"))

    def test_backtrack_over_an_evicted_key_is_class_evicted(self) -> None:
        rig = self.rig(1280, recurrent_blocks=2)
        rig.recurrent.seed(rig.recurrent_key(self.hashes, 1280))
        rig.recurrent.seed(rig.recurrent_key(self.hashes, 640))
        rig.recurrent.seed(rig.recurrent_key(self.hashes, 256))  # the LRU victim is the 1280 boundary
        self.assertEqual(rig.match(rig.request("r", self.hashes, 12800)), (640, True))
        self.assertEqual((field(rig.lines[0], "class"), field(rig.lines[0], "backtrack")), ("evicted", "640"))

    def test_backtrack_past_keys_never_stored_is_class_never_stored(self) -> None:
        rig = self.rig(1280)
        rig.recurrent.seed(rig.recurrent_key(self.hashes, 640))
        self.assertEqual(rig.match(rig.request("r", self.hashes, 12800)), (640, True))
        self.assertEqual((field(rig.lines[0], "class"), field(rig.lines[0], "backtrack")), ("never_stored", "640"))

    def test_no_recurrent_snapshot_backtracks_all_of_the_hit(self) -> None:
        rig = self.rig(1280)
        self.assertEqual(rig.match(rig.request("r", self.hashes, 12800)), (0, False))
        self.assertEqual((field(rig.lines[0], "class"), field(rig.lines[0], "backtrack"), field(rig.lines[0], "recurrent_hit")),
                         ("never_stored", "1280", "0"))

    def test_evicted_at_a_deeper_key_wins_over_never_stored_keys(self) -> None:
        rig = self.rig(1280, recurrent_blocks=1)
        rig.recurrent.seed(rig.recurrent_key(self.hashes, 1152))
        rig.recurrent.seed(rig.recurrent_key(self.hashes, 640))  # 1152 is evicted; 1280 and the rest were never stored
        rig.match(rig.request("r", self.hashes, 12800))
        self.assertEqual(field(rig.lines[0], "class"), "evicted")

    def test_attention_shortfall_is_classified_by_its_first_missing_chunk(self) -> None:
        rig = self.rig(0, 0, attention_blocks=1)
        hashes_b = block_hashes("lane-b", 100)
        rig.match(rig.request("never", hashes_b, 12800))
        self.assertEqual(field(rig.lines[0], "attn_class"), "never_stored")
        rig.attention.seed(rig.attention_key(self.hashes, 128))
        rig.attention.seed(rig.attention_key(hashes_b, 128))  # evicts lane-a's first chunk
        rig.match(rig.request("evicted", self.hashes, 12800))
        self.assertEqual(field(rig.lines[1], "attn_class"), "evicted")

    def test_attention_hit_to_the_prompt_end_has_no_attention_class(self) -> None:
        rig = self.rig(1280)
        rig.recurrent.seed(rig.recurrent_key(self.hashes, 1280))
        rig.match(rig.request("r", self.hashes, 1300))
        self.assertEqual(field(rig.lines[0], "attn_class"), "none")

    def test_cold_prompt_is_logged_with_its_reason(self) -> None:
        rig = self.rig(min_tokens=8192)
        self.assertEqual(rig.match(rig.request("r", self.hashes[:10], 1280)), (0, False))
        self.assertEqual((field(rig.lines[0], "reason"), field(rig.lines[0], "class")), ("cold", "none"))

    def test_skip_read_is_logged_with_its_reason(self) -> None:
        rig = self.rig()
        self.assertEqual(rig.match(rig.request("r", self.hashes, 12800, skip=True)), (0, False))
        self.assertEqual(field(rig.lines[0], "reason"), "skip_read")

    def test_deferred_lookup_logs_once_when_resolved_with_the_store_wait(self) -> None:
        rig = self.rig(None, 1280, 1280)
        rig.recurrent.seed(rig.recurrent_key(self.hashes, 1280))
        request = rig.request("r", self.hashes, 12800)
        clock = iter((100.0, 100.0, 100.25))  # start and sync delay of the deferred lookup, start of the resolved one
        with mock.patch.object(self.module.time, "monotonic", side_effect=lambda: next(clock, 101.0)):
            self.assertEqual(rig.match(request), (None, False))
            self.assertEqual(rig.lines, [])
            rig.match(request)
        self.assertEqual(float(field(rig.lines[0], "store_wait_ms")), 250.0)
        rig.match(request)
        self.assertEqual(len(rig.lines), 1)


class Lanes(unittest.TestCase):
    def setUp(self) -> None:
        self.module = load_overlay(SCHEDULER_FILE)

    def test_lane_is_the_hash_of_block_64_and_lcp_is_to_the_previous_turn(self) -> None:
        rig = Rig(self.module, [0, 0], flag="1")
        first = block_hashes("lane-a", 100)
        second = first[:90] + block_hashes("turn-2", 120, start=90)
        rig.match(rig.request("t1", first, 100 * TOKENS_PER_HASH))
        rig.match(rig.request("t2", second, 120 * TOKENS_PER_HASH))
        self.assertEqual(field(rig.lines[0], "lane"), first[64].hex()[:16])
        self.assertEqual(field(rig.lines[1], "lane"), first[64].hex()[:16])
        self.assertEqual((field(rig.lines[0], "prev_prompt_end"), field(rig.lines[0], "lcp")), ("-1", "0"))
        self.assertEqual((field(rig.lines[1], "prev_prompt_end"), field(rig.lines[1], "lcp")),
                         (str(100 * TOKENS_PER_HASH), str(90 * TOKENS_PER_HASH)))

    def test_prompt_shorter_than_the_lane_block_has_no_lane(self) -> None:
        rig = Rig(self.module, [0], flag="1")
        rig.match(rig.request("t1", block_hashes("short", 10), 1280))
        self.assertEqual((field(rig.lines[0], "lane"), field(rig.lines[0], "lcp")), ("-", "0"))

    def test_lane_block_moves_past_a_shared_prefix(self) -> None:
        """A 70K shared prefix is 547 blocks: at block 64 every harness lane would be one lane."""
        shared = block_hashes("shared", 600)
        lane_a = shared + block_hashes("a", 700, start=600)
        lane_b = shared + block_hashes("b", 700, start=600)
        default = Rig(self.module, [0, 0], flag="1")
        moved = Rig(self.module, [0, 0], flag="1", lane_block="650")
        for rig in (default, moved):
            rig.match(rig.request("a", lane_a, 700 * TOKENS_PER_HASH))
            rig.match(rig.request("b", lane_b, 700 * TOKENS_PER_HASH))
        self.assertEqual(field(default.lines[0], "lane"), field(default.lines[1], "lane"))
        self.assertNotEqual(field(moved.lines[0], "lane"), field(moved.lines[1], "lane"))
        self.assertEqual(field(moved.lines[1], "prev_prompt_end"), "-1")

    def test_bad_lane_block_is_refused_with_the_fix(self) -> None:
        with self.assertRaises(ValueError) as refused:
            Rig(self.module, [], flag="1", lane_block="many")
        self.assertIn(LANE_BLOCK, str(refused.exception))

    def test_bad_flag_is_refused_with_the_fix(self) -> None:
        with self.assertRaises(ValueError) as refused:
            Rig(self.module, [], flag="yes")
        self.assertIn(FLAG, str(refused.exception))


class Bounds(unittest.TestCase):
    def test_evicted_key_lru_is_bounded_at_4096_oldest_out(self) -> None:
        import glm53_miss_attrib as attrib

        keys = attrib.EvictedKeys()
        for index in range(5000):
            keys.add((index,))
        self.assertEqual(len(keys), 4096)
        self.assertNotIn(0, keys)
        self.assertIn(4999, keys)

    def test_previous_turn_table_is_bounded(self) -> None:
        import glm53_miss_attrib as attrib

        miss = attrib.MissAttribution(TOKENS_PER_HASH)
        miss.emit = lambda line: None
        for index in range(attrib.LANES_KEPT + 10):
            hashes = block_hashes(f"lane-{index}", 70)
            miss.record(attrib.MissLookup(f"r{index}", hashes, 70 * TOKENS_PER_HASH, 0, 0, 0, (), None, 0.0, "lookup"))
        self.assertEqual(len(miss.previous), attrib.LANES_KEPT)


class FlagOff(unittest.TestCase):
    """Base scheduler vs overlay scheduler on the same scripted scenario: same answers, same manager call log."""

    @staticmethod
    def drive(module: ModuleType, flag: str | None) -> tuple[list, list, list, list, list, dict]:
        hashes = block_hashes("lane-a", 100)
        rig = Rig(module, [1280, None, 1280, 640, 0], flag=flag, recurrent_blocks=2)
        rig.recurrent.seed(rig.recurrent_key(hashes, 1280))
        rig.recurrent.seed(rig.recurrent_key(hashes, 640))
        answers = [rig.match(rig.request("a", hashes, 12800))]
        deferred = rig.request("b", hashes, 12800)
        answers += [rig.match(deferred), rig.match(deferred)]
        answers.append(rig.match(rig.request("c", block_hashes("lane-c", 100), 12800, skip=True)))
        answers.append(rig.match(rig.request("d", hashes, 12800)))
        return (answers, rig.attention.calls, rig.recurrent.calls, rig.recurrent.order(), rig.lines,
                {"listeners": (len(rig.attention._removal_listeners), len(rig.recurrent._removal_listeners))})

    def test_flag_off_matches_base_answers_and_manager_calls(self) -> None:
        base = self.drive(load_base(SCHEDULER_FILE), None)
        overlay = self.drive(load_overlay(SCHEDULER_FILE), None)
        self.assertEqual(base, overlay)

    def test_flag_off_emits_nothing_and_peeks_nothing(self) -> None:
        answers, attention, recurrent, order, lines, listeners = self.drive(load_overlay(SCHEDULER_FILE), "0")
        self.assertEqual(lines, [])
        self.assertFalse([call for call in attention + recurrent if call[0] == "peek"])
        self.assertEqual(listeners["listeners"], (0, 0))

    def test_flag_on_adds_only_peeks_and_keeps_answers_lookups_touches_and_lru_order(self) -> None:
        off = self.drive(load_overlay(SCHEDULER_FILE), None)
        on = self.drive(load_overlay(SCHEDULER_FILE), "1")
        self.assertEqual(on[0], off[0])
        for index in (1, 2):
            self.assertEqual([call for call in on[index] if call[0] != "peek"], off[index])
        self.assertEqual(on[3], off[3])
        self.assertTrue(on[4])
        self.assertFalse([call for call in on[2] if call[0] == "peek"], "the recurrent class reads the evicted LRU only")


if __name__ == "__main__":
    unittest.main()
