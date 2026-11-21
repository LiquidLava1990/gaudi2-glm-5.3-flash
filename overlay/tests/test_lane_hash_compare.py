"""cells/lane_hash_compare.py: two C1 runs' lane output hashes, cell by cell (lanes x width), on CPU from drive rows. Nothing here touches a card."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from collections.abc import Callable
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "cells"))

import c1_score as scorer  # noqa: E402
import lane_hash_compare as compare  # noqa: E402

ANCHORS = {1024: 2048, 4096: 262_400, 8192: 530_000}
Hashes = Callable[[int, int], list[str]]


def same_for_every_lane(lanes: int, width: int) -> list[str]:
    return [f"out-{width}"] * lanes


def lane_row(lanes: int, width: int, hashes: Hashes = same_for_every_lane, **overrides: object) -> dict:
    row = {"drive": "lanes", "lanes": lanes, "width": width, "anchor_tokens": ANCHORS[width], "status": "ok", "step_ms": 100.0,
           "samples": 190, "output_sha256": hashes(lanes, width)}
    return {**row, **overrides}


def run(hashes: Hashes = same_for_every_lane, skip: tuple[int, int] | None = None) -> list[dict]:
    """The drive rows of a whole run: the other steps' rows, then every lanes x width cell but ``skip``."""
    cells = [lane_row(lanes, width, hashes) for width in scorer.DECODE_WIDTHS for lanes in scorer.LANE_COUNTS if (lanes, width) != skip]
    return [{"drive": "cold", "status": "ok"}, {"drive": "merge", "outputs_equal": True}, *cells]


def with_cell(rows: list[dict], lanes: int, width: int, **overrides: object) -> list[dict]:
    return [{**row, **overrides} if (row.get("lanes"), row.get("width")) == (lanes, width) else row for row in rows]


def differing_at(lanes: int, width: int) -> Hashes:
    return lambda count, wide: ["other"] + same_for_every_lane(count, wide)[1:] if (count, wide) == (lanes, width) else same_for_every_lane(count, wide)


class Judging(unittest.TestCase):
    def test_two_runs_with_the_same_hashes_are_identical_in_every_cell(self) -> None:
        verdict = compare.judge(run(), run())
        self.assertEqual((verdict.status, verdict.exit_code), (compare.IDENTICAL, 0))
        self.assertIn("15 of 15 cells", verdict.summary)

    def test_the_order_of_a_cells_lanes_does_not_matter(self) -> None:
        mixed: Hashes = lambda lanes, width: [f"o{index}" for index in range(lanes)]
        backwards: Hashes = lambda lanes, width: list(reversed(mixed(lanes, width)))
        self.assertEqual(compare.judge(run(mixed), run(backwards)).status, compare.IDENTICAL)

    def test_one_lane_that_differs_makes_its_cell_different_and_names_it(self) -> None:
        verdict = compare.judge(run(), run(differing_at(8, 4096)))
        self.assertEqual((verdict.status, verdict.exit_code), (compare.DIFFERENT, 1))
        different = [line for line in verdict.lines if compare.DIFFERENT in line]
        self.assertEqual(len(different), 1)
        self.assertIn("lanes=8 width=4096", different[0])

    def test_the_same_hashes_in_a_different_number_of_copies_differ(self) -> None:
        once: Hashes = lambda lanes, width: ["a", "b"] + ["a"] * (lanes - 2) if lanes > 1 else ["a"]
        twice: Hashes = lambda lanes, width: ["a", "b"] + ["b"] * (lanes - 2) if lanes > 1 else ["a"]
        self.assertEqual(compare.judge(run(once), run(twice), lanes=(4,)).status, compare.DIFFERENT)

    def test_a_cell_absent_from_one_run_is_not_measured_and_says_which_run(self) -> None:
        verdict = compare.judge(run(), run(skip=(16, 8192)))
        self.assertEqual((verdict.status, verdict.exit_code), (compare.NOT_MEASURED, 2))
        missing = [line for line in verdict.lines if "lanes=16 width=8192" in line]
        self.assertEqual(len(missing), 1)
        self.assertIn("second run", missing[0])

    def test_a_cell_absent_from_both_runs_is_not_measured(self) -> None:
        verdict = compare.judge(run(skip=(4, 1024)), run(skip=(4, 1024)))
        self.assertEqual(verdict.status, compare.NOT_MEASURED)
        self.assertIn("first and second run", next(line for line in verdict.lines if "lanes=4 width=1024" in line))

    def test_a_run_from_before_the_hashes_were_kept_is_not_measured(self) -> None:
        old = [{key: value for key, value in row.items() if key != "output_sha256"} for row in run()]
        verdict = compare.judge(old, run())
        self.assertEqual(verdict.status, compare.NOT_MEASURED)
        self.assertIn("output_sha256", verdict.lines[0])

    def test_fewer_hashes_than_lanes_is_not_measured(self) -> None:
        verdict = compare.judge(with_cell(run(), 4, 4096, output_sha256=["a", "a"]), run())
        self.assertEqual(verdict.status, compare.NOT_MEASURED)
        self.assertIn("2 hashes for 4 lanes", next(line for line in verdict.lines if "lanes=4 width=4096" in line))

    def test_a_cell_that_errored_is_not_measured(self) -> None:
        errored = with_cell(run(), 2, 1024, status="error", reason="OSError: connection reset")
        errored = [{key: value for key, value in row.items() if key != "output_sha256"} if row.get("status") == "error" else row for row in errored]
        verdict = compare.judge(errored, run())
        self.assertEqual(verdict.status, compare.NOT_MEASURED)
        self.assertIn("error", next(line for line in verdict.lines if "lanes=2 width=1024" in line))

    def test_lanes_that_never_overlapped_still_compare_their_outputs(self) -> None:
        unreachable = with_cell(run(), 8, 1024, status="unreachable", reason="never decoded together")
        unreachable = [{key: value for key, value in row.items() if key != "step_ms"} if row.get("status") == "unreachable" else row for row in unreachable]
        self.assertEqual(compare.judge(unreachable, run()).status, compare.IDENTICAL)

    def test_runs_that_sent_different_prompts_are_not_comparable(self) -> None:
        verdict = compare.judge(with_cell(run(), 4, 4096, anchor_tokens=300_000), run())
        self.assertEqual(verdict.status, compare.NOT_MEASURED)
        self.assertIn("anchor_tokens 300000 and 262400", next(line for line in verdict.lines if "lanes=4 width=4096" in line))

    def test_a_cell_logged_twice_in_one_run_is_not_measured(self) -> None:
        verdict = compare.judge([*run(), lane_row(4, 1024)], run())
        self.assertEqual(verdict.status, compare.NOT_MEASURED)
        self.assertIn("twice", next(line for line in verdict.lines if "lanes=4 width=1024" in line))

    def test_a_difference_outranks_a_cell_that_was_not_measured(self) -> None:
        verdict = compare.judge(run(differing_at(16, 1024)), run(skip=(1, 8192)))
        self.assertEqual((verdict.status, verdict.exit_code), (compare.DIFFERENT, 1))

    def test_a_chosen_grid_leaves_the_other_cells_out(self) -> None:
        verdict = compare.judge(run(differing_at(1, 1024)), run(), lanes=(4, 8, 16), widths=(4096,))
        self.assertEqual(verdict.status, compare.IDENTICAL)
        self.assertIn("3 of 3 cells", verdict.summary)

    def test_every_cell_gets_one_line(self) -> None:
        verdict = compare.judge(run(), run())
        self.assertEqual(len(verdict.lines), 15)
        self.assertTrue(all(line.startswith("LANE_HASH lanes=") for line in verdict.lines))


class Reading(unittest.TestCase):
    @staticmethod
    def write(path: Path, rows: list[dict]) -> Path:
        path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
        return path

    def test_a_drive_file_is_read_row_by_row_and_blank_lines_are_skipped(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp, "x.drive.jsonl")
            path.write_text(json.dumps({"drive": "cold"}) + "\n\n" + json.dumps({"drive": "merge"}) + "\n", encoding="utf-8")
            self.assertEqual(compare.read_rows(path), [{"drive": "cold"}, {"drive": "merge"}])

    def test_a_run_folder_is_read_through_its_one_drive_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            self.write(Path(tmp, "claude-flash-launch-c1.drive.jsonl"), run())
            Path(tmp, "container.log").write_text("noise\n", encoding="utf-8")
            self.assertEqual(compare.read_rows(Path(tmp)), run())

    def test_a_folder_without_exactly_one_drive_file_refuses_and_says_what_it_holds(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(compare.RunUnreadable, "no .*drive.jsonl"):
                compare.read_rows(Path(tmp))
            self.write(Path(tmp, "a.drive.jsonl"), run())
            self.write(Path(tmp, "b.drive.jsonl"), run())
            with self.assertRaisesRegex(compare.RunUnreadable, "2 .*drive.jsonl.*a.drive.jsonl.*b.drive.jsonl"):
                compare.read_rows(Path(tmp))

    def test_a_missing_path_and_a_broken_line_refuse_with_the_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(compare.RunUnreadable, "nowhere"):
                compare.read_rows(Path(tmp, "nowhere"))
            broken = Path(tmp, "x.drive.jsonl")
            broken.write_text('{"drive": "cold"}\n{not json\n', encoding="utf-8")
            with self.assertRaisesRegex(compare.RunUnreadable, r"x\.drive\.jsonl line 2"):
                compare.read_rows(broken)


class Main(unittest.TestCase):
    def main(self, first: list[dict], second: list[dict], *extra: str) -> tuple[int, str]:
        with tempfile.TemporaryDirectory() as tmp:
            paths = [Reading.write(Path(tmp, name), rows) for name, rows in (("a.drive.jsonl", first), ("b.drive.jsonl", second))]
            out = StringIO()
            with redirect_stdout(out):
                code = compare.main([str(paths[0]), str(paths[1]), *extra])
        return code, out.getvalue()

    def test_identical_runs_print_the_verdict_last_and_exit_zero(self) -> None:
        code, text = self.main(run(), run())
        self.assertEqual(code, 0)
        self.assertTrue(text.strip().splitlines()[-1].startswith("LANE_HASH_COMPARE IDENTICAL"))

    def test_different_runs_exit_one_and_not_measured_exits_two(self) -> None:
        self.assertEqual(self.main(run(), run(differing_at(4, 1024)))[0], 1)
        self.assertEqual(self.main(run(), run(skip=(4, 1024)))[0], 2)

    def test_the_grid_options_choose_the_cells(self) -> None:
        code, text = self.main(run(differing_at(1, 1024)), run(), "--lanes", "4,8,16", "--widths", "4096,8192")
        self.assertEqual(code, 0)
        self.assertIn("6 of 6 cells", text)

    def test_an_unreadable_run_is_not_measured_not_a_traceback(self) -> None:
        out = StringIO()
        with redirect_stdout(out):
            code = compare.main(["/nowhere/a", "/nowhere/b"])
        self.assertEqual(code, 2)
        self.assertIn("NOT_MEASURED", out.getvalue())

    def test_a_bad_grid_option_is_refused_by_the_parser(self) -> None:
        with tempfile.TemporaryDirectory() as tmp, self.assertRaises(SystemExit):
            Reading.write(Path(tmp, "a.drive.jsonl"), run())
            compare.main([str(Path(tmp, "a.drive.jsonl")), str(Path(tmp, "a.drive.jsonl")), "--lanes", "4,x"])


if __name__ == "__main__":
    unittest.main()
