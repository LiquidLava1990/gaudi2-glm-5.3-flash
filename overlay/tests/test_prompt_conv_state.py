"""B2: a short keyed chunk must save its real last inputs as conv state. The real base-image conv reads each row's length from query_start_loc
and takes new state at that column, past the masked padding. These run the real function on what padded_inputs hands a key's graph. What a
card's Synapse does with the captured input is the gate named in the plan; the trace below proves only that the Python path reads no value."""
from __future__ import annotations

import os
import sys
import unittest
from dataclasses import dataclass
from pathlib import Path
from unittest import mock

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from overlay_testlib import load_base  # noqa: E402
from test_prompt_keys import KeyCase  # noqa: E402

conv = load_base("vllm_gaudi/ops/causal_conv1d_pytorch.py")

DIM, STATE_LEN, KEY_ROWS = 2, 3, 16
SLOTS = 4  # the last slot is the allocation's garbage row
HISTORY = 500.0


@dataclass(frozen=True)
class ChunkMetadata:
    """The per-row fields pad_field handles, shaped as the runner builds them for a prompt batch."""
    glm53_index_block_table: torch.Tensor
    glm53_token_mask: torch.Tensor
    query_start_loc: torch.Tensor
    load_indices_tensor: torch.Tensor
    seq_lens_tensor: torch.Tensor
    context_lens_tensor: torch.Tensor
    has_initial_states_p: torch.Tensor
    num_actual_tokens: int
    num_decodes: int


class ConvCase(KeyCase):
    keys = "128:1024"  # the flag; these call padded_inputs with a 16-row bucket directly, small enough to read

    def batch(self, lengths: tuple[int, ...], has_initial: tuple[bool, ...] | None = None, width: int | None = None) -> object:
        """One runner batch of these prompts, each row padded to the runner's own width, as _batch prepares it."""
        count, width = len(lengths), width or max(lengths)
        query_start_loc = torch.tensor([0, *(width * (row + 1) for row in range(count))], dtype=torch.int32)
        metadata = ChunkMetadata(
            torch.zeros((count, 8), dtype=torch.int32), torch.ones(count * width), query_start_loc,
            torch.arange(count, dtype=torch.int64), torch.tensor(lengths), torch.zeros(count, dtype=torch.int32),
            torch.tensor(has_initial or (False,) * count), count * width, 0)
        valid = tuple(row * width + column for row, length in enumerate(lengths) for column in range(length))
        region = self.step.Region(tuple(range(count)), lengths, (0,) * count, width, count, valid, metadata, False)
        tokens = torch.zeros((count, width), dtype=torch.int64)
        return self.step.PreparedBatch(region, tokens, tokens.clone(), (lengths[0] - 1,), id(metadata))

    def padded(self, batch: object, prompts: int = 1) -> object:
        return self.graph.padded_inputs([batch], (0, KEY_ROWS, 1024, prompts))[0].regions[0].metadata

    @staticmethod
    def inputs(metadata: ChunkMetadata) -> torch.Tensor:
        """(DIM, rows x KEY_ROWS) with the values 1, 2, 3... in each row, masked by the padded token mask as the model masks qkv first."""
        rows = metadata.glm53_token_mask.numel() // KEY_ROWS
        values = torch.arange(1.0, KEY_ROWS + 1).repeat(rows)
        return (values * metadata.glm53_token_mask).unsqueeze(0).repeat(DIM, 1)

    @staticmethod
    def states() -> torch.Tensor:
        """Every slot holds its own earlier history."""
        return HISTORY * torch.arange(1.0, SLOTS + 1)[:, None, None].repeat(1, STATE_LEN, DIM)

    def run_conv(self, metadata: ChunkMetadata, slots: torch.Tensor, states: torch.Tensor) -> torch.Tensor:
        weight = torch.ones(DIM, STATE_LEN + 1)
        return conv.hpu_causal_conv1d_fn(self.inputs(metadata), weight, None, states, metadata.query_start_loc, cache_indices=slots,
                                         has_initial_state=metadata.has_initial_states_p, activation=None)


class KeyedChunkState(ConvCase):
    def saved(self, length: int, has_initial: bool = False) -> list[float]:
        states = self.states()
        self.run_conv(self.padded(self.batch((length,), (has_initial,))), torch.tensor([0]), states)
        return states[0, :, 0].tolist()

    def test_a_short_chunk_saves_its_last_real_inputs_not_zeros(self) -> None:
        self.assertEqual(self.saved(11), [9.0, 10.0, 11.0])

    def test_a_full_chunk_saves_the_end_of_the_key(self) -> None:
        self.assertEqual(self.saved(KEY_ROWS), [14.0, 15.0, 16.0])

    def test_a_chunk_shorter_than_the_state_keeps_the_earlier_history(self) -> None:
        self.assertEqual(self.saved(1, has_initial=True), [HISTORY, HISTORY, 1.0])

    def test_a_chunk_from_a_wider_runner_row_saves_the_same_state(self) -> None:
        states = self.states()
        self.run_conv(self.padded(self.batch((11,), width=64)), torch.tensor([0]), states)
        self.assertEqual(states[0, :, 0].tolist(), [9.0, 10.0, 11.0])

    def test_the_key_saves_what_the_runner_batch_saves_natively(self) -> None:
        batch = self.batch((11,))
        native_states, keyed_states = self.states(), self.states()
        region = batch.region.metadata
        values = torch.arange(1.0, 12).unsqueeze(0).repeat(DIM, 1)
        conv.hpu_causal_conv1d_fn(values, torch.ones(DIM, STATE_LEN + 1), None, native_states, torch.tensor([0, 11]), cache_indices=torch.tensor([0]),
                                  has_initial_state=region.has_initial_states_p, activation=None)
        self.run_conv(self.padded(batch), torch.tensor([0]), keyed_states)
        self.assertTrue(torch.equal(keyed_states, native_states))

    def test_the_input_keeps_its_dtype_and_key_shape(self) -> None:
        metadata = self.padded(self.batch((11,)))
        self.assertEqual(metadata.query_start_loc.dtype, torch.int32)
        self.assertEqual(metadata.query_start_loc.tolist(), [0, 11])

    def test_a_row_longer_than_the_key_refuses_as_no_fit(self) -> None:
        with self.assertRaisesRegex(ValueError, "17 rows"):
            self.graph.pad_field("query_start_loc", torch.tensor([0, 17]), self.batch((17,)).region, 1, KEY_ROWS, ctx= 1024)


class SharedRows(ConvCase):
    """Step 49: several prompts in one key, with pad rows after them."""

    def test_each_prompt_saves_at_its_own_length_and_the_pad_rows_save_nothing(self) -> None:
        metadata = self.padded(self.batch((5, 11)), prompts=3)
        self.assertEqual(metadata.query_start_loc.tolist(), [0, 5, 16, 16])
        states, slots = self.states(), torch.tensor([0, 1, -1])
        garbage = states[-1].clone()
        self.run_conv(metadata, slots, states)
        self.assertEqual(states[0, :, 0].tolist(), [3.0, 4.0, 5.0])
        self.assertEqual(states[1, :, 0].tolist(), [9.0, 10.0, 11.0])
        self.assertTrue(torch.equal(states[-1], garbage))

    def test_every_other_per_row_field_keeps_its_real_rows_and_zeroes_the_pad_rows(self) -> None:
        batch = self.batch((5, 11), (True, False))
        metadata = self.padded(batch, prompts=3)
        self.assertEqual(metadata.seq_lens_tensor.tolist(), [5, 11, 0])
        self.assertEqual(metadata.has_initial_states_p.tolist(), [True, False, False])
        self.assertEqual(metadata.load_indices_tensor.tolist(), [0, 1, -1])
        self.assertEqual(metadata.glm53_token_mask.reshape(3, KEY_ROWS).sum(1).tolist(), [5, 11, 0])


class CapturedInput(ConvCase):
    """A capture bakes any Python read of a tensor's value. A trace does the same, so a value read in the conv would fix the dummy's length here."""

    @staticmethod
    def captured() -> torch.jit.ScriptFunction:
        def forward(x: torch.Tensor, states: torch.Tensor, starts: torch.Tensor, slots: torch.Tensor, initial: torch.Tensor) -> torch.Tensor:
            return conv.hpu_causal_conv1d_fn(x, torch.ones(DIM, STATE_LEN + 1), None, states, starts, cache_indices=slots,
                                             has_initial_state=initial, activation=None)

        dummy = torch.ones(DIM, KEY_ROWS), torch.zeros(SLOTS, STATE_LEN, DIM), torch.tensor([0, KEY_ROWS], dtype=torch.int32), torch.tensor([0]), torch.tensor([False])
        return torch.jit.trace(forward, dummy, check_trace=False)

    def test_a_graph_captured_at_the_full_dummy_replays_a_short_chunk_at_its_own_length(self) -> None:
        graph = self.captured()
        for length, expected in ((11, [9.0, 10.0, 11.0]), (4, [2.0, 3.0, 4.0]), (KEY_ROWS, [14.0, 15.0, 16.0])):
            with self.subTest(length=length):
                metadata, states = self.padded(self.batch((length,))), self.states()
                graph(self.inputs(metadata), states, metadata.query_start_loc, torch.tensor([0]), metadata.has_initial_states_p)
                self.assertEqual(states[0, :, 0].tolist(), expected)

    def test_the_captured_call_marks_the_input_a_graph_input(self) -> None:
        """StaticCall hands the graph every metadata tensor as a user input, the query_start_loc of a key included."""
        inputs = self.graph.clone_inputs(self.graph.padded_inputs([self.batch((11,))], (0, KEY_ROWS, 1024, 1)))
        call = self.graph.StaticCall(lambda **_: None, inputs)
        starts = inputs[0].regions[0].metadata.query_start_loc
        self.assertTrue(any(tensor is starts for tensor in call.tensor_inputs))


class DecodeRows(ConvCase):
    """A decode row really is one token: a lattice key pads the table's width, never a row's tokens, so the padded key's length of 1 is the truth."""

    def test_a_decode_region_stays_one_token_a_row_and_a_row_that_is_not_refuses(self) -> None:
        region = self.step.Region((0, 1), (1, 1), (5, 9), 1, 2, (0, 1), None, True)
        starts = self.graph.pad_field("query_start_loc", torch.tensor([0, 1, 2]), region, 4, 1, ctx=1024)
        self.assertEqual(starts.tolist(), [0, 1, 2, 3, 4])
        with self.assertRaisesRegex(ValueError, "one decode token"):
            self.step.MixedMetadata((self.step.Region((0,), (2,), (5,), 2, 1, (0, 1), None, True),))

    def test_real_rows_shift_by_one_and_pad_rows_touch_only_the_garbage_slot(self) -> None:
        states, slots = self.states(), torch.tensor([0, 1, -1, -1])
        before = states.clone()
        x = torch.tensor([[1.0, 1.0], [2.0, 2.0], [7.0, 7.0], [7.0, 7.0]])
        with mock.patch.dict(os.environ, {"VLLM_HPU_CONV1D_DISABLE_TPC": "1"}):
            conv.hpu_causal_conv1d_update(x, states, torch.ones(DIM, STATE_LEN + 1), None, activation=None, conv_state_indices=slots,
                                          query_start_loc=torch.arange(5))
        self.assertEqual(states[0, :, 0].tolist(), [HISTORY, HISTORY, 1.0])
        self.assertEqual(states[1, :, 0].tolist(), [2 * HISTORY, 2 * HISTORY, 2.0])
        self.assertTrue(torch.equal(states[2], before[2]))


if __name__ == "__main__":
    unittest.main()
