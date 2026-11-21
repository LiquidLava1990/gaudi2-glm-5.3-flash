"""R7: width padding runs only on a real proof. `check` pads and compares every real row of any lane mix at a check width, runs native
everywhere else, and refuses on a mismatch on every rank; `1` needs a proof that names a run whose GLM53_DECODE_MERGE lines the code reads:
all equal, a full 16-row batch, an all-short batch, and every (narrow, wide) pair the lattice would pad."""
from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from test_decode_lattice import LATTICE_ENV, GraphCase  # noqa: E402

PROOF = "HPU_GLM53_DECODE_PAD_PROOF"
MODE = "HPU_GLM53_DECODE_WIDTH_PAD"


def request(prompt: list[int]) -> object:
    from types import SimpleNamespace
    return SimpleNamespace(prompt_token_ids=prompt)


class Lanes(GraphCase):
    """A decode step of `count` lanes at native width 2048 (a check width); positions below the top-k window count as short."""

    def runner(self, count: int) -> object:
        from types import SimpleNamespace
        return SimpleNamespace(requests={str(lane): request([lane, 2, 3]) for lane in range(count)})

    def step(self, count: int, positions: list[int] | None = None, native: int = 2048) -> int:
        positions = positions or [500 + lane for lane in range(count)]
        table = torch.tensor([[position] for position in positions])
        return self.graph.decode_table_width(self.runner(count), [str(lane) for lane in range(count)], native, table)

    def hidden(self, count: int, bump_row: int | None = None) -> torch.Tensor:
        rows = torch.arange(float(count * 4)).reshape(count, 1, 4)
        if bump_row is not None:
            rows[bump_row] += 1
        return rows


class CheckEveryLaneMix(Lanes):
    def test_a_sixteen_lane_step_is_compared_across_all_sixteen_rows(self) -> None:
        self.assertEqual(self.step(16), 2048)
        self.graph.observe_decode_hidden(self.hidden(16))
        self.assertEqual(self.step(16), 4096)
        with self.assertLogs(self.graph.LOG, "WARNING") as logs:
            self.graph.observe_decode_hidden(self.hidden(16))
        self.assertIn("narrow=2048 wide=4096 rows=16 ", logs.output[0])
        self.assertIn("torch_equal=1", logs.output[0])
        stored = next(iter(self.graph._DECODE_CHECK.pending.values()), None)
        self.assertIsNone(stored, "the comparison consumed the native run")

    def test_the_native_run_keeps_every_real_row_not_the_padded_batch(self) -> None:
        self.step(3)
        self.graph.observe_decode_hidden(torch.zeros((16, 1, 4)))
        stored = next(iter(self.graph._DECODE_CHECK.pending.values()))
        self.assertEqual(stored.shape[0], 3)

    def test_a_difference_in_any_one_row_refuses(self) -> None:
        for bump in (0, 7, 15):
            with self.subTest(row=bump):
                self.graph._DECODE_CHECK.__init__()
                self.step(16)
                self.graph.observe_decode_hidden(self.hidden(16))
                self.step(16)
                with self.assertLogs(self.graph.LOG, "WARNING") as logs, self.assertRaisesRegex(RuntimeError, "HPU_GLM53_DECODE_WIDTH_PAD"):
                    self.graph.observe_decode_hidden(self.hidden(16, bump_row=bump))
                self.assertIn("torch_equal=0", logs.output[0])

    def test_one_rank_seeing_a_difference_refuses_every_rank(self) -> None:
        self.step(2)
        self.graph.observe_decode_hidden(self.hidden(2))
        self.step(2)
        peers = [{"equal": True}, {"equal": False}]
        with mock.patch.object(self.graph, "rank_reports", return_value=peers), self.assertLogs(self.graph.LOG, "WARNING"), \
                self.assertRaisesRegex(RuntimeError, "rank 1"):
            self.graph.observe_decode_hidden(self.hidden(2))

    def test_different_lanes_or_positions_are_a_different_comparison(self) -> None:
        self.step(2)
        self.graph.observe_decode_hidden(self.hidden(2))
        self.assertEqual(self.step(2, positions=[900, 901]), 2048, "other positions are another first run, not the repeat")
        self.assertEqual(self.step(3), 2048)

    def test_short_batches_are_marked(self) -> None:
        for positions, short in (([10, 20], 1), ([10, 5000], 0)):
            with self.subTest(positions=positions):
                self.graph._DECODE_CHECK.__init__()
                self.step(2, positions)
                self.graph.observe_decode_hidden(self.hidden(2))
                self.step(2, positions)
                with self.assertLogs(self.graph.LOG, "WARNING") as logs:
                    self.graph.observe_decode_hidden(self.hidden(2))
                self.assertIn(f"short={short}", logs.output[0])


class CheckRunsNativeWhereNotProven(Lanes):
    def test_a_width_between_keys_outside_the_check_widths_runs_native(self) -> None:
        self.set_env(HPU_GLM53_DECODE_PAD_CHECK_WIDTHS="512")
        self.assertEqual(self.step(1, native=2048), 2048)
        self.assertEqual(self.graph._DECODE_CHECK.pending, {})

    def test_a_native_width_with_no_key_then_refuses_at_the_metadata(self) -> None:
        self.set_env(HPU_GLM53_DECODE_PAD_CHECK_WIDTHS="512")
        width = self.step(1, native=2048)
        trimmed = mock.Mock(is_prompt=False, glm53_index_block_table=torch.zeros((1, width)))
        with self.assertRaisesRegex(RuntimeError, "no captured decode key 1x2048"):
            self.graph.decode_key_metadata(trimmed, 1, True, False)

    def test_a_table_past_the_top_still_refuses(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "VLLM_GAUDI_MIXED_DECODE_WIDTHS"):
            self.step(1, native=16384)

    def test_a_listed_width_is_its_own_key(self) -> None:
        self.assertEqual(self.step(4, native=4096), 4096)

    def test_the_check_widths_have_a_native_key_at_every_row_count(self) -> None:
        keys = self.graph.decode_keys()
        for rows in self.graph.DECODE_ROWS:
            self.assertIn((rows, 2048), keys)
        self.assertEqual(len(keys), 20)


class ShortContextsBelowTheLowestKey(Lanes):
    """R11: nothing narrower than the lowest key has a graph, so a short context pads up to it; every width at or past the lowest stays proven."""

    def test_every_width_below_the_lowest_key_pads_up_with_no_comparison(self) -> None:
        for power in range(10):
            with self.subTest(native=1 << power):
                self.assertEqual(self.step(1, native=1 << power), 1024)
        self.assertEqual(self.graph._DECODE_CHECK.pending, {})

    def test_a_4000_token_decode_reaches_a_captured_key_at_every_row_count(self) -> None:
        for rows in self.graph.DECODE_ROWS:
            with self.subTest(rows=rows):
                self.assertIn((rows, self.step(rows, native=32)), self.graph.decode_keys())

    def test_every_decode_width_check_mode_can_meet_has_a_key(self) -> None:
        for power in range(14):
            with self.subTest(native=1 << power):
                self.graph._DECODE_CHECK.__init__()
                self.assertIn((1, self.step(1, native=1 << power)), self.graph.decode_keys())

    def test_the_lowest_key_itself_stays_native(self) -> None:
        self.assertEqual(self.step(1, native=1024), 1024)

    def test_listed_as_a_check_width_it_is_compared_at_a_price_instead(self) -> None:
        self.set_env(HPU_GLM53_DECODE_PAD_CHECK_WIDTHS="16,2048")
        self.assertEqual(self.step(1, native=16), 16)
        self.assertEqual(len(self.graph._DECODE_CHECK.current), 7)
        self.graph.observe_decode_hidden(self.hidden(1))
        self.assertEqual(self.step(1, native=16), 1024)
        with self.assertLogs(self.graph.LOG, "WARNING") as logs:
            self.graph.observe_decode_hidden(self.hidden(1))
        self.assertIn("narrow=16 wide=1024 rows=1", logs.output[0])
        self.assertIn("short=1", logs.output[0])

    def test_the_serving_line_names_each_width_once(self) -> None:
        with self.assertLogs(self.graph.LOG, "WARNING") as logs:
            self.step(1, native=32)
            self.step(1, native=32)
            self.step(1, native=64)
        lines = [line for line in logs.output if "GLM53_DECODE_BELOW_KEY" in line]
        self.assertEqual(len(lines), 2)
        self.assertIn("native=32 key=1024", lines[0])

    def test_proven_mode_still_asks_a_proof_of_every_pair(self) -> None:
        self.assertEqual(self.graph.decode_pad_pairs()[0], (1, 1024))


class ProofCase(GraphCase):
    env = {**LATTICE_ENV, MODE: "1", "HPU_GLM53_DECODE_PAD_CHECK_WIDTHS": ""}

    def setUp(self) -> None:
        super().setUp()
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.directory = Path(directory.name)

    @staticmethod
    def line(narrow: int, wide: int, rows: int = 1, equal: int = 1, short: int = 0) -> str:
        return f"WARNING 11-07 04:00:00 [glm53_mixed_graph.py:1] GLM53_DECODE_MERGE narrow={narrow} wide={wide} rows={rows} position=500 torch_equal={equal} short={short}"

    def pairs(self) -> list[str]:
        return [self.line(narrow, 1024) for narrow in (1, 2, 4, 8, 16, 32, 64, 128, 256, 512)] + [self.line(2048, 4096)]

    def full(self) -> list[str]:
        return self.pairs() + [self.line(2048, 4096, rows=16), self.line(512, 1024, rows=3, short=1)]

    def write(self, lines: list[str], name: str = "run.log") -> Path:
        path = self.directory / name
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return path

    def refuses(self, pattern: str, proof: Path | str) -> None:
        self.set_env(**{PROOF: str(proof)})
        with self.assertRaisesRegex(ValueError, pattern):
            self.graph.check_prompt_key_flags()


class PadProof(ProofCase):
    def test_a_complete_proof_passes_as_a_file_or_a_directory(self) -> None:
        path = self.write(self.full())
        for proof in (path, self.directory):
            with self.subTest(proof=proof):
                self.set_env(**{PROOF: str(proof)})
                self.graph.check_prompt_key_flags()

    def test_a_name_that_is_not_a_run_refuses(self) -> None:
        self.refuses("not a file or directory", "k1-out/run")

    def test_any_non_empty_text_no_longer_passes(self) -> None:
        self.refuses(PROOF, "yes")

    def test_a_proof_with_no_merge_lines_refuses(self) -> None:
        self.refuses("no GLM53_DECODE_MERGE", self.write(["nothing here"]))

    def test_one_unequal_line_refuses_and_names_it(self) -> None:
        self.refuses("torch_equal=0.*narrow=2048", self.write(self.full() + [self.line(2048, 4096, equal=0)]))

    def test_a_proof_without_a_sixteen_row_batch_refuses(self) -> None:
        self.refuses("16-row", self.write(self.pairs() + [self.line(512, 1024, rows=3, short=1)]))

    def test_a_proof_without_an_all_short_batch_refuses(self) -> None:
        self.refuses("short", self.write(self.pairs() + [self.line(2048, 4096, rows=16)]))

    def test_a_pair_the_lattice_would_pad_but_the_run_never_compared_refuses_and_names_it(self) -> None:
        lines = [line for line in self.full() if "narrow=256 " not in line]
        self.refuses("256 -> 1024", self.write(lines))

    def test_a_line_from_before_the_short_field_does_not_count(self) -> None:
        old = [line.rsplit(" short=", 1)[0] for line in self.full()]
        self.refuses("no GLM53_DECODE_MERGE", self.write(old))

    def test_the_directory_proof_reads_every_file(self) -> None:
        self.write(self.pairs(), "a.log")
        self.write([self.line(2048, 4096, rows=16), self.line(512, 1024, rows=3, short=1)], "b.log")
        self.set_env(**{PROOF: str(self.directory)})
        self.graph.check_prompt_key_flags()


if __name__ == "__main__":
    unittest.main()
