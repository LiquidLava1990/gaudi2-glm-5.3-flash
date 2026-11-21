"""Refactor pins (c1d-r3b, Clean Code on contact): pad_field and the two prompt-key selectors answer what they answered before the
refactor, for every field, mode and key grid below. refactor_pins.json holds the outcomes the pre-refactor code (r3, fcae2603) gave,
generated with PIN_WRITE=1 on that tree; any output or refusal that moves fails here, naming the case."""
from __future__ import annotations

import itertools
import json
import os
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

import glm53_mixed_graph as graph  # noqa: E402

PINS = Path(__file__).resolve().parent / 'refactor_pins.json'
FIELDS = ('num_actual_tokens', 'num_decodes', 'glm53_index_block_table', 'slot_mapping', 'glm53_index_slot_mapping', 'glm53_tail_slot_mapping',
          'glm53_token_mask', 'padding_mask_flat', 'load_indices_tensor', 'store_indices_tensor', 'non_spec_state_indices_tensor',
          'seq_lens_tensor', 'context_lens_tensor', 'has_initial_states_p', 'query_start_loc', 'query_start_loc_p', 'non_spec_query_start_loc',
          'other_field')
SIZE, WIDTH, CTX = 4, 4, 8
SHARED_KEYS = ((2, 256, 1024), (4, 256, 1024), (4, 256, 8192), (2, 1024, 8192), (8, 128, 2048))
KEYS = ((1024, 8192), (1024, 1024), (128, 8192), (128, 1024), (512, 4096))


def field_value(name: str, native_rows: int) -> object:
    """The value a two-request region of widths (3, 2) in a 4-token batch carries in `name`."""
    values = {
        'num_actual_tokens': 8, 'num_decodes': 0, 'other_field': torch.tensor([7, 7]),
        'glm53_index_block_table': torch.arange(native_rows * 6, dtype=torch.int32).reshape(native_rows, 6),
        'load_indices_tensor': torch.tensor([[5, 6]]), 'store_indices_tensor': torch.tensor([5, 6]),
        'non_spec_state_indices_tensor': torch.tensor([[5, 6], [7, 8]]),
        'seq_lens_tensor': torch.tensor([3, 2]), 'context_lens_tensor': torch.tensor([[1], [2]]), 'has_initial_states_p': torch.tensor([True, False]),
    }
    if 'query_start_loc' in name:
        return torch.tensor([0, 3, 5], dtype=torch.int32)
    return values.get(name, torch.arange(8, dtype=torch.int64))


def outcome(call: object) -> str:
    try:
        result = call()
    except ValueError as error:
        return f'ValueError: {error}'
    return repr((str(result.dtype), tuple(result.shape), result.tolist())) if isinstance(result, torch.Tensor) else repr(result)


def pad_table() -> dict[str, str]:
    table = {}
    for keyed, pinned, decode, native_rows in itertools.product((True, False), (True, False), (True, False), (2, SIZE)):
        region = SimpleNamespace(requests=(0, 1), lengths=(3, 2), batch=2, width=4, decode=decode)
        with mock.patch.object(graph, 'prompt_keys', return_value=((128, 1024),) if keyed else ()),                 mock.patch.object(graph, 'native_pin_enabled', return_value=pinned):
            for name in FIELDS:
                value = field_value(name, native_rows)
                table[f'{name}|keyed={keyed}|pinned={pinned}|decode={decode}|rows={native_rows}'] = outcome(
                    lambda: graph.pad_field(name, value, region, SIZE, WIDTH, ctx=CTX))
    return table


def selector_table() -> dict[str, str]:
    table = {}
    for keys in (SHARED_KEYS, SHARED_KEYS[:1], ()):
        for prompts, rows, width in itertools.product((1, 2, 3, 5, 9), (1, 128, 256, 512, 2048), (1, 1024, 2048, 8192, 16384)):
            with mock.patch('glm53_mixed_step.prompt_batch_keys', return_value=keys):
                table[f'shared|{len(keys)}|{prompts}|{rows}|{width}'] = repr(graph.select_shared_prompt_key(prompts, rows, width))
    for keys in (KEYS, KEYS[:1], ()):
        for rows, width in itertools.product((1, 11, 128, 129, 512, 1024, 4096), (1, 1024, 1025, 4096, 8192, 16384)):
            with mock.patch.object(graph, 'prompt_keys', return_value=keys):
                table[f'single|{len(keys)}|{rows}|{width}'] = repr(graph.select_prompt_key(rows, width))
    return table


class Pins(unittest.TestCase):
    def check(self, part: str, table: dict[str, str]) -> None:
        pinned = json.loads(PINS.read_text(encoding='utf-8'))[part]
        self.assertEqual(sorted(table), sorted(pinned))
        for case, answer in table.items():
            with self.subTest(case=case):
                self.assertEqual(answer, pinned[case])

    def test_pad_field_answers_every_field_and_mode_as_before(self) -> None:
        self.check('pad_field', pad_table())

    def test_the_selectors_answer_the_grid_as_before(self) -> None:
        self.check('selectors', selector_table())


if __name__ == '__main__':
    if os.environ.get('PIN_WRITE') == '1':
        PINS.write_text(json.dumps({'pad_field': pad_table(), 'selectors': selector_table()}, indent=1, sort_keys=True) + '\n', encoding='utf-8')
        print('PIN_WRITE', PINS)
    else:
        unittest.main()
