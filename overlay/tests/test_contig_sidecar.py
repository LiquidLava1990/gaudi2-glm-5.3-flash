"""Step 3: contiguous recurrent sidecar buffer. CPU tests of the views; the pinned path uses a seam, the card is step 14."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest import mock

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from overlay_testlib import SIDECAR_FILE, load_base, load_overlay  # noqa: E402

FLAG = "VLLM_RECURRENT_SIDECAR_CONTIG"
ROWS = 4
DTYPES = (torch.float32, torch.bfloat16, torch.uint8, torch.int64, torch.float16, torch.bool)


def sources(seed: int = 7) -> dict[int, tuple[torch.Tensor, ...]]:
    generator = torch.Generator().manual_seed(seed)
    groups: dict[int, tuple[torch.Tensor, ...]] = {}
    for group_idx, shapes in enumerate((((3, 5), (7,), (2, 2, 2)), ((11,), (1, 9), (13,)))):
        tensors = []
        for dtype, shape in zip(DTYPES[group_idx * 3:], shapes):
            element = torch.empty(0, dtype=dtype).element_size()
            count = int(torch.tensor((ROWS, *shape)).prod().item())
            raw = torch.randint(0, 2, (count * element,), generator=generator, dtype=torch.uint8)
            tensors.append(raw.view(dtype).reshape(ROWS, *shape))
        groups[group_idx] = tuple(tensors)
    return groups


def worker(module: ModuleType, flag: str | None, groups: dict | None = None) -> object:
    groups = sources() if groups is None else groups
    registry = module.RecurrentStateRegistry(
        groups, lambda req_id, group_idx: int(req_id), sum(t[0].nbytes for ts in groups.values() for t in ts), "sig",
        lambda req_id: 0)
    environ = {} if flag is None else {FLAG: flag}
    with mock.patch.dict("os.environ", environ, clear=False):
        return module.RecurrentSidecarWorker(registry)


def raw_bytes(tensor: torch.Tensor) -> bytes:
    return tensor.reshape(-1).view(torch.uint8).numpy().tobytes()


def slot_tensors(slot: dict) -> list[torch.Tensor]:
    return [tensor for group in sorted(slot) for tensor in slot[group]]


class Views(unittest.TestCase):
    def setUp(self) -> None:
        self.module = load_overlay(SIDECAR_FILE)
        self.worker = worker(self.module, "1")
        self.slot = self.worker._slot(3)

    def test_views_have_the_shapes_and_dtypes_of_the_native_slot(self) -> None:
        native = worker(self.module, None)._slot(3)
        self.assertEqual([(t.shape, t.dtype) for t in slot_tensors(self.slot)],
                         [(t.shape, t.dtype) for t in slot_tensors(native)])
        self.assertEqual(list(self.slot), list(native))

    def test_views_share_one_buffer(self) -> None:
        pointers = {tensor.untyped_storage().data_ptr() for tensor in slot_tensors(self.slot)}
        self.assertEqual(len(pointers), 1)

    def test_views_are_aligned(self) -> None:
        for tensor in slot_tensors(self.slot):
            self.assertEqual(tensor.data_ptr() % self.module.CONTIG_ALIGN_BYTES, 0, tensor.dtype)

    def test_views_do_not_overlap(self) -> None:
        tensors = slot_tensors(self.slot)
        for tensor in tensors:
            tensor.reshape(-1).view(torch.uint8).fill_(0)
        for index, tensor in enumerate(tensors):
            tensor.reshape(-1).view(torch.uint8).fill_(1)
            for other_index, other in enumerate(tensors):
                expected = 1 if other_index <= index else 0
                self.assertEqual(set(raw_bytes(other)), {expected}, (index, other_index))

    def test_buffer_is_barely_larger_than_the_state(self) -> None:
        state = sum(tensor.nbytes for tensor in slot_tensors(self.slot))
        buffer = self.slot[0][0].untyped_storage().nbytes()
        self.assertLessEqual(buffer, state + len(slot_tensors(self.slot)) * self.module.CONTIG_ALIGN_BYTES)

    def test_one_buffer_per_block_and_block_reuse(self) -> None:
        other = self.worker._slot(4)
        self.assertNotEqual(self.slot[0][0].untyped_storage().data_ptr(), other[0][0].untyped_storage().data_ptr())
        self.assertIs(self.worker._slot(3), self.slot)


class RoundTrip(unittest.TestCase):
    def setUp(self) -> None:
        self.module = load_overlay(SIDECAR_FILE)

    def test_store_then_load_restores_exact_bytes_through_the_views(self) -> None:
        groups = sources()
        sidecar = worker(self.module, "1", groups)
        expected = [raw_bytes(tensor[1].clone()) for group in sorted(groups) for tensor in groups[group]]
        sidecar.submit_store(1, "1", 9)
        self.assertEqual([raw_bytes(tensor) for tensor in slot_tensors(sidecar._slot(9))], expected)
        for tensors in groups.values():
            for tensor in tensors:
                tensor[2].reshape(-1).view(torch.uint8).fill_(0)
        sidecar.submit_load(2, "2", 9)
        self.assertEqual([raw_bytes(tensor[2]) for group in sorted(groups) for tensor in groups[group]], expected)
        self.assertEqual(sorted(sidecar.get_finished()), [1, 2])

    def test_contig_and_native_store_the_same_bytes(self) -> None:
        native, contig = worker(self.module, None), worker(self.module, "1")
        for sidecar in (native, contig):
            sidecar.submit_store(1, "0", 5)
        self.assertEqual([raw_bytes(t) for t in slot_tensors(native._slot(5))],
                         [raw_bytes(t) for t in slot_tensors(contig._slot(5))])


class FlagOff(unittest.TestCase):
    def test_flag_off_slot_matches_base_one_allocation_per_tensor(self) -> None:
        base, overlay = load_base(SIDECAR_FILE), load_overlay(SIDECAR_FILE)
        for flag in (None, "0"):
            base_slot, overlay_slot = worker(base, None)._slot(1), worker(overlay, flag)._slot(1)
            self.assertEqual([(t.shape, t.dtype, t.storage_offset()) for t in slot_tensors(base_slot)],
                             [(t.shape, t.dtype, t.storage_offset()) for t in slot_tensors(overlay_slot)])
            self.assertEqual(len({t.untyped_storage().data_ptr() for t in slot_tensors(overlay_slot)}),
                             len(slot_tensors(overlay_slot)))

    def test_flag_off_allocates_with_the_same_torch_empty_calls_as_base(self) -> None:
        base, overlay = load_base(SIDECAR_FILE), load_overlay(SIDECAR_FILE)
        calls: list[list] = []
        for module in (base, overlay):
            sidecar = worker(module, None)
            with mock.patch.object(module.torch, "empty", wraps=torch.empty) as empty:
                sidecar._slot(1)
            calls.append(empty.call_args_list)
        self.assertEqual(calls[0], calls[1])


def hpu_like() -> dict[int, tuple[SimpleNamespace, ...]]:
    device = SimpleNamespace(type="hpu")
    return {0: (SimpleNamespace(device=device, shape=(4, 8), dtype=torch.float32),
                SimpleNamespace(device=device, shape=(4, 3), dtype=torch.bfloat16))}


class PinRefusal(unittest.TestCase):
    """An HPU worker refuses at construction (boot), before the first store."""

    def setUp(self) -> None:
        self.module = load_overlay(SIDECAR_FILE)
        self.registry = self.module.RecurrentStateRegistry(hpu_like(), lambda req_id, group_idx: 0, 44, "sig", lambda req_id: 0)
        self.plain_buffer = mock.patch.object(self.module, "_contig_buffer",
                                              lambda nbytes, pinned: torch.empty(nbytes, dtype=torch.uint8))

    def build(self) -> object:
        with mock.patch.dict("os.environ", {FLAG: "1"}, clear=False):
            return self.module.RecurrentSidecarWorker(self.registry)

    def test_unpinned_views_refuse_at_boot_with_the_fix(self) -> None:
        with self.plain_buffer, self.assertRaises(RuntimeError) as refused:
            self.build()
        self.assertIn(FLAG, str(refused.exception))
        self.assertIn("45/20", str(refused.exception))

    def test_pinned_views_boot(self) -> None:
        with self.plain_buffer, mock.patch.object(torch.Tensor, "is_pinned", return_value=True):
            sidecar = self.build()
        self.assertEqual(sidecar._resident, {}, "the boot probe holds no slot")

    def test_flag_off_does_not_probe_at_boot(self) -> None:
        with mock.patch.object(self.module, "_contig_buffer") as buffer:
            self.module.RecurrentSidecarWorker(self.registry)
        buffer.assert_not_called()

    def test_misaligned_buffer_refuses_at_boot(self) -> None:
        with self.plain_buffer, mock.patch.object(torch.Tensor, "is_pinned", return_value=True), \
                mock.patch.object(torch.Tensor, "data_ptr", return_value=63), self.assertRaises(RuntimeError) as refused:
            self.build()
        self.assertIn("aligned", str(refused.exception))

    def test_bad_flag_value_is_refused_with_the_fix(self) -> None:
        with mock.patch.dict("os.environ", {FLAG: "yes"}, clear=False), self.assertRaises(ValueError) as refused:
            self.module.RecurrentSidecarWorker(self.registry)
        self.assertIn(FLAG, str(refused.exception))


if __name__ == "__main__":
    unittest.main()
