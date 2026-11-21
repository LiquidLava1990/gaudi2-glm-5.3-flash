"""cells/rank_digest_compare.py: two runs' GLM53_RANK_CHECK lines, read by read and layer by layer, on their rows_digest tags. Nothing here touches a card."""
from __future__ import annotations

import contextlib
import io
import sys
import tempfile
import unittest
from collections.abc import Callable
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "cells"))

import rank_digest_compare as compare  # noqa: E402

PREFIX = "(Worker_TP0 pid=412) WARNING 11-07 05:41:07 [glm53_mixed_graph.py:1380]"
LAYERS = range(3, 14)
Tag = Callable[[str, int, int], str]


def tag_of(key: str, read: int, layer: int) -> str:
    return f"{abs(hash((key, read, layer))) % 16**16:016x}"


def line(key: str, layer: int, start: int, replay: int, tag: str, *, equal: int = 1, live: int = 16) -> str:
    rows = int(key.split(":")[1].split("x")[0]) if key.startswith("decode:") else 1024
    return (f"{PREFIX} GLM53_RANK_CHECK key={key} layer={layer} equal={equal} rows={rows} rows_differ=0 live_rows={live} ranks=4 "
            f"start={start} replay={replay} rows_digest={tag}")


def run(starts: dict[str, list[int]] | None = None, tag: Tag = tag_of, skip: tuple[str, int, int] | None = None) -> list[str]:
    """A log: per key, one complete read (every layer) per start, replays numbered in order. ``skip`` drops (key, read, layer)."""
    starts = starts or {"decode:16x8192": [3500, 3510, 3520], "decode:4x1024": [900, 910]}
    lines, replay = ["GLM53_RANK_CHECK_READY layers=11 rows=16 bytes_each=8192"], 0
    for key, positions in starts.items():
        for read, start in enumerate(positions, 1):
            replay += 1
            lines += [line(key, layer, start, replay, tag(key, read, layer)) for layer in LAYERS if (key, read, layer) != skip]
    return lines


def differing_at(key: str, read: int, layers: tuple[int, ...]) -> Tag:
    return lambda found_key, found_read, layer: "f" * 16 if (found_key, found_read, layer) in {(key, read, each) for each in layers} else tag_of(found_key, found_read, layer)


class Identical(unittest.TestCase):
    def test_two_runs_with_the_same_tags_are_identical(self) -> None:
        verdict = compare.judge(run(), run())
        self.assertEqual((verdict.status, verdict.exit_code), (compare.IDENTICAL, 0))
        self.assertIn("5 reads x 11 layers", verdict.summary)

    def test_the_replay_numbers_and_the_other_lines_between_reads_do_not_matter(self) -> None:
        shifted = [text.replace("replay=", "replay=9") if "GLM53_RANK_CHECK key=" in text else text for text in run()]
        noisy = [*shifted[:3], "(Worker_TP1) INFO something else", *shifted[3:]]
        self.assertEqual(compare.judge(run(), noisy).status, compare.IDENTICAL)

    def test_prompt_keys_are_compared_too(self) -> None:
        both = {"1024:8192": [131072, 160000], "decode:8x4096": [3000]}
        self.assertEqual(compare.judge(run(both), run(both)).status, compare.IDENTICAL)
        self.assertEqual(compare.judge(run(both), run(both, differing_at("1024:8192", 2, (5,)))).status, compare.DIFFERENT)


class Different(unittest.TestCase):
    def test_the_first_differing_layer_of_the_first_differing_read_is_named(self) -> None:
        second = run(tag=differing_at("decode:16x8192", 2, (9, 6, 11)))
        verdict = compare.judge(run(), second)
        self.assertEqual((verdict.status, verdict.exit_code), (compare.DIFFERENT, 1))
        first = next(text for text in verdict.lines if text.startswith("RANK_DIGEST_FIRST_DIFFERENCE"))
        for field in ("key=decode:16x8192", "read=2", "start=3510", "layer=6", f"a={tag_of('decode:16x8192', 2, 6)}", f"b={'f' * 16}"):
            self.assertIn(field, first)
        self.assertEqual(sum(text.startswith("RANK_DIGEST_FIRST_DIFFERENCE") for text in verdict.lines), 1)

    def test_layers_are_ordered_as_numbers_not_as_text(self) -> None:
        verdict = compare.judge(run(), run(tag=differing_at("decode:16x8192", 1, (13, 4))))
        self.assertIn("layer=4 ", next(text for text in verdict.lines if text.startswith("RANK_DIGEST_FIRST_DIFFERENCE")))

    def test_every_read_gets_a_line_saying_identical_or_how_many_layers_differ(self) -> None:
        verdict = compare.judge(run(), run(tag=differing_at("decode:4x1024", 1, (3, 4))))
        reads = [text for text in verdict.lines if text.startswith("RANK_DIGEST_READ")]
        self.assertEqual(len(reads), 5)
        broken = next(text for text in reads if "key=decode:4x1024 read=1 " in text)
        self.assertIn("status=DIFFERENT", broken)
        self.assertIn("layers_differ=2", broken)
        self.assertEqual(sum("status=IDENTICAL" in text for text in reads), 4)

    def test_different_outranks_a_read_that_could_not_be_compared(self) -> None:
        first = run(skip=("decode:16x8192", 1, 7))
        verdict = compare.judge(first, run(tag=differing_at("decode:4x1024", 2, (5,))))
        self.assertEqual(verdict.status, compare.DIFFERENT)
        self.assertTrue(any("layer 7" in text for text in verdict.lines))

    def test_a_tag_that_differs_is_a_difference_even_where_both_runs_said_equal_1(self) -> None:
        """The point of the tool: a sharded run's equal=1 is by construction, its tag is the evidence."""
        self.assertTrue(all("equal=1" in text for text in run() if "GLM53_RANK_CHECK key=" in text))
        self.assertEqual(compare.judge(run(), run(tag=differing_at("decode:16x8192", 3, (3,)))).status, compare.DIFFERENT)


class NotMeasured(unittest.TestCase):
    def assert_not_measured(self, first: list[str], second: list[str], mention: str) -> compare.Verdict:
        verdict = compare.judge(first, second)
        self.assertEqual((verdict.status, verdict.exit_code), (compare.NOT_MEASURED, 2), verdict)
        self.assertTrue(any(mention in text for text in verdict.lines), (mention, verdict.lines))
        return verdict

    def test_a_log_with_no_tags_is_not_measured(self) -> None:
        untagged = [text.split(" rows_digest=")[0] for text in run()]
        self.assert_not_measured(untagged, run(), "no rows_digest")
        self.assert_not_measured(run(), untagged, "no rows_digest")

    def test_a_log_with_no_check_lines_is_not_measured(self) -> None:
        self.assert_not_measured(["nothing"], run(), "no GLM53_RANK_CHECK")

    def test_a_key_only_one_run_read_is_not_measured(self) -> None:
        self.assert_not_measured(run({"decode:16x8192": [3500]}), run({"decode:16x8192": [3500], "decode:8x4096": [100]}), "decode:8x4096")

    def test_a_different_number_of_reads_of_a_key_is_not_measured(self) -> None:
        self.assert_not_measured(run({"decode:16x8192": [3500, 3510]}), run({"decode:16x8192": [3500]}), "2 reads in the first run, 1 in the second")

    def test_reads_of_another_batch_are_not_compared(self) -> None:
        """Fixed batches: the longest lane's position is the same at the same read, or the runs were not driven alike."""
        self.assert_not_measured(run({"decode:16x8192": [3500]}), run({"decode:16x8192": [3501]}), "start=3500 in the first run, 3501 in the second")

    def test_another_live_row_count_is_not_compared(self) -> None:
        other = [text.replace("live_rows=16", "live_rows=15") for text in run({"decode:16x8192": [3500]})]
        self.assert_not_measured(run({"decode:16x8192": [3500]}), other, "live_rows")

    def test_a_layer_missing_from_a_read_is_not_measured_and_named(self) -> None:
        verdict = self.assert_not_measured(run(skip=("decode:4x1024", 2, 8)), run(), "layer 8")
        self.assertTrue(any("decode:4x1024 read=2" in text for text in verdict.lines))

    def test_a_layer_with_no_verdict_is_not_measured(self) -> None:
        void = f"{PREFIX} GLM53_RANK_CHECK_NOT_RETAINED key=decode:16x8192 layer=7 start=3500 replay=1 reason=stale_rows=16 of 16. No verdict on these rows"
        voided = [void if "layer=7 " in text and "decode:16x8192" in text and "start=3500" in text else text for text in run()]
        self.assert_not_measured(voided, run(), "no verdict")

    def test_a_layer_logged_twice_means_two_runs_share_a_log(self) -> None:
        self.assert_not_measured([*run(), *run()], run(), "logged twice")


class Selecting(unittest.TestCase):
    def test_key_limits_the_comparison_to_that_key(self) -> None:
        one_key = run({"decode:16x8192": [3500, 3510, 3520]})
        self.assertEqual(compare.judge(run(), one_key).status, compare.NOT_MEASURED, "the second run has no 4x1024 reads")
        self.assertEqual(compare.judge(run(), one_key, keys=("decode:16x8192",)).status, compare.IDENTICAL)
        differs = run(tag=differing_at("decode:4x1024", 1, (3,)))
        self.assertEqual(compare.judge(run(), differs, keys=("decode:16x8192",)).status, compare.IDENTICAL)
        self.assertEqual(compare.judge(run(), differs, keys=("decode:4x1024",)).status, compare.DIFFERENT)

    def test_a_key_neither_log_has_is_not_measured(self) -> None:
        self.assertEqual(compare.judge(run(), run(), keys=("decode:2x1024",)).status, compare.NOT_MEASURED)

    def test_decode_only_leaves_the_prompt_keys_out(self) -> None:
        both = {"1024:8192": [131072], "decode:8x4096": [3000]}
        prompt_differs = run(both, differing_at("1024:8192", 1, (5,)))
        self.assertEqual(compare.judge(run(both), prompt_differs).status, compare.DIFFERENT)
        self.assertEqual(compare.judge(run(both), prompt_differs, decode_only=True).status, compare.IDENTICAL)

    def test_decode_only_with_no_decode_key_compares_nothing_and_says_so(self) -> None:
        prompts = {"1024:8192": [131072]}
        verdict = compare.judge(run(prompts), run(prompts), decode_only=True)
        self.assertEqual(verdict.status, compare.NOT_MEASURED)
        self.assertIn("no keys to compare", "\n".join(verdict.lines))


class Main(unittest.TestCase):
    def write(self, folder: Path, name: str, lines: list[str]) -> Path:
        path = folder / name
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return path

    def run_main(self, *argv: str) -> tuple[int, list[str]]:
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            code = compare.main(list(argv))
        return code, out.getvalue().splitlines()

    def test_two_log_files_print_the_reads_then_one_last_line_and_exit_with_the_verdict(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            first, second = self.write(Path(folder), "a.log", run()), self.write(Path(folder), "b.log", run())
            code, lines = self.run_main(str(first), str(second))
        self.assertEqual(code, 0)
        self.assertTrue(lines[-1].startswith("RANK_DIGEST_COMPARE IDENTICAL "), lines[-1])

    def test_a_run_folder_means_its_container_log(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            runs = []
            for name in ("c1-a", "c1-b"):
                (Path(folder) / name).mkdir()
                self.write(Path(folder) / name, "container.log", run())
                runs.append(str(Path(folder) / name))
            code, lines = self.run_main(*runs)
        self.assertEqual((code, lines[-1].split()[1]), (0, "IDENTICAL"))

    def test_a_missing_log_is_not_measured_and_says_which(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            code, lines = self.run_main(str(Path(folder) / "nope.log"), str(Path(folder) / "also.log"))
        self.assertEqual(code, 2)
        self.assertIn("nope.log", "\n".join(lines))
        self.assertTrue(lines[-1].startswith("RANK_DIGEST_COMPARE NOT_MEASURED "), lines[-1])

    def test_a_folder_without_a_container_log_says_what_to_give_instead(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            code, lines = self.run_main(folder, folder)
        self.assertEqual(code, 2)
        self.assertIn("container.log", "\n".join(lines))

    def test_the_arguments_select_keys(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            first = self.write(Path(folder), "a.log", run())
            second = self.write(Path(folder), "b.log", run(tag=differing_at("decode:4x1024", 1, (3,))))
            code, _ = self.run_main(str(first), str(second), "--key", "decode:16x8192")
            self.assertEqual(code, 0)
            code, _ = self.run_main(str(first), str(second), "--decode-only")
            self.assertEqual(code, 1)


if __name__ == "__main__":
    unittest.main()
