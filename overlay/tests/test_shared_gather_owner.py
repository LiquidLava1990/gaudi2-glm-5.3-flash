"""Step 21: HPU_GLM53_Q4_SHARED_GATHER_OWNER on the kept tree (d53a529e 3-way onto a1535efe). TP4 is simulated on CPU."""

from __future__ import annotations

import importlib.util
import logging
import os
import sys
import unittest
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest import mock

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from glm53_mixed_graph import capture_context  # noqa: E402
from overlay_testlib import OVERLAY_ROOT, load_base, load_overlay  # noqa: E402

Q4KV_FILE = "vllm_gaudi/ops/hpu_glm5next_q4kv.py"
IMAGE_TESTS = Path("/opt/glm53/test_hpu_glm5next_q4kv.py")
ENV = {"HPU_GLM53_TP_SHARDED_Q4KV": "1", "HPU_GLM53_Q4_TOKEN_CYCLIC": "1"}
FLAG = "HPU_GLM53_Q4_SHARED_GATHER_OWNER"
TP = 4
ROW_BYTES = 272
CACHE_SLOTS = 256


def slots_for(rows: int = 3, width: int = 8, base: int = 16) -> torch.Tensor:
    """The selector's four-token pool contract: column c holds a slot congruent to c mod TP."""
    return torch.tensor([[row * base + column for column in range(width)] for row in range(rows)], dtype=torch.long)


def row_bytes(slot: int) -> torch.Tensor:
    return torch.tensor([(slot * 7 + index) % 251 for index in range(ROW_BYTES)], dtype=torch.uint8)


def rank_cache(rank: int, tp_size: int = TP) -> torch.Tensor:
    cache = torch.zeros((CACHE_SLOTS // tp_size + 1, ROW_BYTES), dtype=torch.uint8)
    for slot in range(rank, CACHE_SLOTS, tp_size):
        cache[slot // tp_size] = row_bytes(slot)
    return cache


def expected_rows(slots: torch.Tensor) -> torch.Tensor:
    return torch.stack([torch.stack([row_bytes(int(slot)) for slot in row]) for row in slots])


class World:
    """Rank 0's view of a TP4 group: the collectives return what the real rank-major gather would."""

    def __init__(self, module: ModuleType) -> None:
        self.q4 = module.HPUQ4MLAKVCache()
        self.q4._tp_coordinates = lambda: (0, TP)
        self.q4.dequant_output = lambda packed, dtype=torch.bfloat16: packed
        self.slots = slots_for()
        self.gathers = 0

    def stripes(self) -> list[torch.Tensor]:
        return [self.q4._owned_stripes(rank_cache(rank), self.slots, rank, TP) for rank in range(TP)]

    def gather_into(self, output: torch.Tensor, local_rows: torch.Tensor, group: object = None) -> None:
        self.gathers += 1
        output.copy_(torch.cat(self.stripes(), dim=0))

    def all_gather(self, local_rows: torch.Tensor, dim: int = 0) -> torch.Tensor:
        self.gathers += 1
        return torch.cat(self.stripes(), dim=dim)

    def patches(self) -> list:
        group = SimpleNamespace(device_group=object(), all_gather=self.all_gather)
        return [mock.patch("vllm.distributed.get_tp_group", return_value=group),
                mock.patch("torch.distributed.all_gather_into_tensor", side_effect=self.gather_into)]


class GatherCase(unittest.TestCase):
    flag = "0"

    def setUp(self) -> None:
        environ = mock.patch.dict(os.environ, {**ENV, FLAG: self.flag})
        environ.start()
        self.addCleanup(environ.stop)
        self.module = load_overlay(Q4KV_FILE)
        self.base = load_base(Q4KV_FILE)
        self.module._SHARED_OWNERS.clear()
        self.world = World(self.module)
        for patcher in self.world.patches():
            patcher.start()
            self.addCleanup(patcher.stop)

    def cache(self) -> torch.Tensor:
        return rank_cache(0)


class FlagOff(GatherCase):
    def test_fetch_selected_bytes_equal_base(self) -> None:
        base = World(self.base)
        with mock.patch("torch.distributed.all_gather_into_tensor", side_effect=base.gather_into):
            expected = base.q4.fetch_selected(self.cache(), base.slots)
        self.assertTrue(torch.equal(self.world.q4.fetch_selected(self.cache(), self.world.slots), expected))
        self.assertTrue(torch.equal(expected, expected_rows(self.world.slots)))

    def test_gather_selected_stripes_bytes_equal_base(self) -> None:
        base = World(self.base)
        with mock.patch("torch.distributed.all_gather_into_tensor", side_effect=base.gather_into):
            expected = base.q4.gather_selected_stripes(self.cache(), base.slots)
        self.assertTrue(torch.equal(self.world.q4.gather_selected_stripes(self.cache(), self.world.slots), expected))

    def test_selected_from_stripes_bytes_equal_base(self) -> None:
        owner = self.world.q4.gather_selected_stripes(self.cache(), self.world.slots)
        base_q4 = World(self.base).q4
        for rows in ((0, 1), (1, 3), (0, 3)):
            self.assertTrue(torch.equal(self.world.q4.selected_from_stripes(owner, rows, 7),
                                        base_q4.selected_from_stripes(owner, rows, 7)), rows)

    def test_a_fresh_owner_per_gather_even_inside_a_capture(self) -> None:
        with capture_context():
            first = self.world.q4.gather_selected_stripes(self.cache(), self.world.slots)
            second = self.world.q4.gather_selected_stripes(self.cache(), self.world.slots)
        self.assertIsNot(first, second)
        self.assertEqual(self.module._SHARED_OWNERS, {})

    def test_unset_flag_is_off(self) -> None:
        with mock.patch.dict(os.environ):
            del os.environ[FLAG]
            self.assertFalse(self.module.shared_owner_enabled())

    def test_collective_call_order_equals_base(self) -> None:
        expected = hpu_gather_events(self.base, self.base._all_gather_owned, 4)
        self.assertEqual(hpu_gather_events(self.module, self.module._all_gather_owned, 4), expected)
        self.assertEqual(expected, [("mark_step", (), {}), ("gather",), ("finish",), ("mark_step", (), {"sync": True})])

    def test_stage_label_fires_once_per_gather(self) -> None:
        with mock.patch.dict(os.environ, {"HPU_GLM53_R589_ENABLE": "1", "HPU_GLM53_R589_TRACE": "1"}), \
                mock.patch("glm53_r589_bounded_prefill._emit") as emit:
            self.world.q4.gather_selected_stripes(self.cache(), self.world.slots)
        labels = [call.args[0] for call in emit.call_args_list]
        self.assertEqual(labels.count("tp.all_gather.entry"), 1)


class FlagOn(GatherCase):
    flag = "1"

    def test_native_lazy_steps_keep_fresh_owners_outside_a_mixed_capture(self) -> None:
        """Non-dual serving never enters capture_context, so the flag is inert there (reported to the plan owner)."""
        first = self.world.q4.gather_selected_stripes(self.cache(), self.world.slots)
        second = self.world.q4.gather_selected_stripes(self.cache(), self.world.slots)
        self.assertIsNot(first, second)
        self.assertEqual(self.module._SHARED_OWNERS, {})

    def test_44_tile_gathers_of_one_shape_write_one_owner(self) -> None:
        with capture_context():
            owners = {id(self.world.q4.gather_selected_stripes(self.cache(), self.world.slots)) for _ in range(44)}
        self.assertEqual(len(owners), 1)
        self.assertEqual(self.world.gathers, 44)
        self.assertEqual(len(self.module._SHARED_OWNERS), 1)

    def test_two_key_captures_of_one_stripe_shape_share_one_owner(self) -> None:
        with capture_context():
            first = self.world.q4.gather_selected_stripes(self.cache(), self.world.slots)
        with capture_context():
            second = self.world.q4.gather_selected_stripes(self.cache(), self.world.slots)
        self.assertIs(first, second)
        self.assertEqual(len(self.module._SHARED_OWNERS), 1)

    def test_each_stripe_shape_has_its_own_owner(self) -> None:
        with capture_context():
            first = self.world.q4.gather_selected_stripes(self.cache(), self.world.slots)
            self.world.slots = slots_for(width=16)
            second = self.world.q4.gather_selected_stripes(self.cache(), self.world.slots)
        self.assertIsNot(first, second)
        self.assertEqual(sorted(tuple(key[0]) for key in self.module._SHARED_OWNERS),
                         [(12, 2, ROW_BYTES), (12, 4, ROW_BYTES)])

    def test_owner_bytes_are_the_shape_product(self) -> None:
        self.assertEqual(self.module.shared_owner_bytes((64, 2048, ROW_BYTES)), 64 * 2048 * ROW_BYTES)

    def test_owner_bytes_count_the_dtype_so_an_int32_owner_is_four_bytes_an_element(self) -> None:
        """The decode shard's gather owner holds int32 selected ids (R x 2051); the log line must not report its elements as bytes."""
        self.assertEqual(self.module.shared_owner_bytes((16, 2051), torch.int32), 16 * 2051 * 4)
        self.assertEqual(self.module.shared_owner_bytes((4, 8), torch.bfloat16), 4 * 8 * 2)
        self.assertEqual(self.module.shared_owner_bytes((4, 8), torch.uint8), 4 * 8)

    def test_the_new_owner_line_reports_the_bytes_of_the_owners_dtype(self) -> None:
        local_rows = torch.zeros((4, 2051), dtype=torch.int32)
        with capture_context(), self.assertLogs(self.module.LOG, logging.WARNING) as logs:
            owner = self.module.shared_gather_owner(local_rows, TP)
        self.assertEqual((tuple(owner.shape), owner.dtype), ((16, 2051), torch.int32))
        self.assertEqual(len(logs.output), 1)
        self.assertIn(f"shape=(16, 2051) bytes={16 * 2051 * 4} owners=1", logs.output[0])

    def test_fetch_selected_bytes_equal_the_native_path(self) -> None:
        native = self.world.q4.fetch_selected(self.cache(), self.world.slots)
        with capture_context():
            shared = self.world.q4.fetch_selected(self.cache(), self.world.slots)
        self.assertTrue(torch.equal(shared, native))
        self.assertTrue(torch.equal(shared, expected_rows(self.world.slots)))

    def test_grouped_tiles_bytes_equal_the_native_path(self) -> None:
        native = self.world.q4.gather_selected_stripes(self.cache(), self.world.slots)
        with capture_context():
            owner = self.world.q4.gather_selected_stripes(self.cache(), self.world.slots)
        self.assertIn(id(owner), {id(value) for value in self.module._SHARED_OWNERS.values()})
        for rows in ((0, 1), (1, 3), (0, 3)):
            self.assertTrue(torch.equal(self.world.q4.selected_from_stripes(owner, rows, 8),
                                        self.world.q4.selected_from_stripes(native, rows, 8)), rows)

    def test_a_read_tile_survives_the_next_gather_of_the_shape(self) -> None:
        with capture_context():
            owner = self.world.q4.gather_selected_stripes(self.cache(), self.world.slots)
            tile = self.world.q4.selected_from_stripes(owner, (0, 3), 8)
            expected = expected_rows(self.world.slots)
            self.world.slots = slots_for(base=32)
            self.world.q4.gather_selected_stripes(self.cache(), self.world.slots)
        self.assertTrue(torch.equal(tile, expected))
        self.assertTrue(torch.equal(self.world.q4.selected_from_stripes(owner, (0, 3), 8),
                                    expected_rows(self.world.slots)))

    def test_owner_gather_commits_only_when_asked(self) -> None:
        def gather(commit: bool) -> list[tuple]:
            return hpu_gather_events(self.module, lambda rows, tp: self.module._all_gather_into(
                mock.sentinel.output, rows, commit=commit), 4)

        self.assertEqual(gather(False), [("mark_step", (), {}), ("gather",), ("finish",)])
        self.assertEqual(gather(True), [("mark_step", (), {}), ("gather",), ("finish",), ("mark_step", (), {"sync": True})])

    def test_bad_value_refuses_with_the_fix(self) -> None:
        with mock.patch.dict(os.environ, {FLAG: "yes"}):
            with self.assertRaisesRegex(ValueError, f"{FLAG}='yes': set 1 .* or 0 / unset"):
                self.module.shared_owner_enabled()


def hpu_gather_events(module: ModuleType, call: object, tp_size: int) -> list[tuple]:
    """The lazy-mode calls of one gather. A stand-in tensor reports device hpu, so only the order is exercised."""
    events: list[tuple] = []
    local_rows = SimpleNamespace(shape=(2, 3, ROW_BYTES), dtype=torch.uint8, device=SimpleNamespace(type="hpu"))
    group = SimpleNamespace(device_group=object())
    mark_step = lambda *args, **kwargs: events.append(("mark_step", args, kwargs))  # noqa: E731
    finish = "vllm_gaudi.distributed.device_communicators.hpu_communicator._finish_segmented_lazy_collective"
    with mock.patch("vllm.distributed.get_tp_group", return_value=group), \
            mock.patch("torch.empty", return_value=mock.sentinel.output), \
            mock.patch("habana_frameworks.torch.core.mark_step", side_effect=mark_step), \
            mock.patch(finish, side_effect=lambda: events.append(("finish",))), \
            mock.patch("torch.distributed.all_gather_into_tensor", side_effect=lambda *a, **k: events.append(("gather",))):
        call(local_rows, tp_size)
    return events


class CanonicalOrder(unittest.TestCase):
    def test_column_c_times_tp_plus_r_is_rank_r_column_c(self) -> None:
        module = load_overlay(Q4KV_FILE)
        for tp_size in (2, 4, 8):
            owner = torch.tensor([[[rank * 100 + row * 10 + column] for column in range(3)]
                                  for rank in range(tp_size) for row in range(2)], dtype=torch.int64)
            rows = module.canonical_stripe_rows(owner, tp_size, 1, 2)
            expected = [[[rank * 100 + 10 + column] for column in range(3) for rank in range(tp_size)]]
            self.assertEqual(rows.tolist(), expected, tp_size)


class MergeKeepsBothSides(unittest.TestCase):
    def test_a1535efe_native_unpack_branch_and_the_owner_patch_are_both_present(self) -> None:
        source = (OVERLAY_ROOT / Q4KV_FILE).read_text()
        self.assertIn("from glm53_mla_unpack import ENABLED as native_unpack_enabled", source)
        self.assertIn('SHARED_OWNER_ENV = "HPU_GLM53_Q4_SHARED_GATHER_OWNER"', source)


class ImageSuite(unittest.TestCase):
    """The image's own Q4 suite, run against the overlay module. libmla_lazy.so loads only with a card (the suite fails
    on the unmodified base here too), so the native unpack module is stubbed off; a card run covers that branch."""

    def test_image_q4_suite_passes_on_the_overlay(self) -> None:
        spec = importlib.util.spec_from_file_location("image_q4_suite", IMAGE_TESTS)
        self.assertIsNotNone(spec, f"{IMAGE_TESTS} is missing: run inside the 1481bb99 container")
        suite = importlib.util.module_from_spec(spec)
        no_native_unpack = SimpleNamespace(ENABLED=False, unpack=None)
        with mock.patch.dict(sys.modules, {"vllm_gaudi.ops.hpu_glm5next_q4kv": load_overlay(Q4KV_FILE),
                                           "glm53_mla_unpack": no_native_unpack}):
            spec.loader.exec_module(suite)
            for name in sorted(vars(suite)):
                if name.startswith("test_"):
                    with self.subTest(name):
                        getattr(suite, name)()


if __name__ == "__main__":
    unittest.main()
