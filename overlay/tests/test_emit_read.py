"""emit_read.py: the verdict on a HPU_GLM53_EMIT_WORDSKIP run from its container log (one FLASH_EMIT_WORDSKIP_LIVE line per rank) and, for a traced run,
its cell log (the REPLAY_TRACE_KERNEL table and the REPLAY_TRACE_SCAN anchor name the new emit GUID). The trace text is the real reporter's output
(replay_trace_report.render), so a change to its columns breaks these tests, not a card run. CPU only."""
from __future__ import annotations

import sys
import unittest
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "cells"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import emit_read as reader  # noqa: E402
import replay_trace_report as report  # noqa: E402
from test_replay_trace_report import COMPLETE_OPS, DECODE_RUN, RUN, ReportCase, trace  # noqa: E402
from test_replay_trace_tables import scan_call  # noqa: E402

NEW, STOCK = reader.WORDSKIP_GUID, reader.STOCK_GUID
TP = 4


def live_line(rank: int, guid: str = NEW) -> str:
    return f"(Worker_TP{rank} pid=41{rank}) WARNING 11-07 08:00:00 [vllm.rc6] FLASH_EMIT_WORDSKIP_LIVE rank={rank} rows=128 width=32768 guid={guid}"


def container_log(ranks: tuple[int, ...] = tuple(range(TP)), guid: str = NEW, extra: tuple[str, ...] = ()) -> list[str]:
    return ["INFO engine ready", *map(live_line, ranks, [guid] * len(ranks)), *extra]


class Reports(ReportCase):
    def cell_text(self, *emits: str, decode: bool = True) -> list[str]:
        """The reporter's own lines for a prompt run whose scan recipes end in each of `emits` (one recipe per emit), plus a decode run."""
        ops = [event for recipe, emit in enumerate(emits, start=100) for at in (1000, 3000) for event in scan_call(recipe, at, emit=emit)]
        runs = []
        if ops:
            self.write(RUN, trace(6000.0, ops), 6.0)
            runs.append(RUN)
        if decode:
            self.write(DECODE_RUN, trace(10_000.0, COMPLETE_OPS), 10.0)
            runs.append(DECODE_RUN)
        return report.render(report.build_report(self.directory, runs, 4, []))

    def verdict(self, container: list[str], cell: list[str] | None = None, expect: str = reader.ON, tp_size: int = TP) -> reader.Verdict:
        return reader.judge(container, tp_size, expect, cell)


class WordSkipRun(Reports):
    def test_four_live_lines_and_a_new_guid_trace_pass(self) -> None:
        verdict = self.verdict(container_log(), self.cell_text(NEW))
        self.assertEqual((verdict.status, verdict.reasons, verdict.exit_code), (reader.PASS, (), 0))

    def test_the_reader_finds_the_kernel_names_and_the_anchor_in_the_reporter_s_real_text(self) -> None:
        tables = reader.parse_trace(self.cell_text(NEW))
        self.assertIn(NEW, tables[RUN].kernels)
        self.assertIn("mla_topkkeys_i32", tables[RUN].kernels)
        self.assertEqual(tables[RUN].anchor, NEW)
        self.assertNotIn(NEW, tables[DECODE_RUN].kernels)

    def test_an_untraced_cell_is_judged_on_its_container_log_alone(self) -> None:
        self.assertEqual(self.verdict(container_log()).status, reader.PASS)

    def test_a_trace_with_both_emits_passes_on_the_new_guid_in_its_anchor(self) -> None:
        text = self.cell_text(STOCK, NEW)
        self.assertEqual(reader.parse_trace(text)[RUN].anchor, f"{STOCK}+{NEW}")
        self.assertEqual(self.verdict(container_log(), text).status, reader.PASS)


class LiveLineCount(Reports):
    def assertFails(self, verdict: reader.Verdict, *expected: str) -> None:
        self.assertEqual((verdict.status, verdict.exit_code), (reader.FAIL, 1))
        for text in expected:
            self.assertIn(text, "\n".join(verdict.reasons))

    def test_no_live_line_fails_because_the_flag_reached_no_rank(self) -> None:
        self.assertFails(self.verdict(container_log(ranks=())), "0 FLASH_EMIT_WORDSKIP_LIVE lines, expected one per rank: 4")

    def test_three_lines_fail_and_name_the_rank_without_one(self) -> None:
        self.assertFails(self.verdict(container_log(ranks=(0, 1, 3))), "3 FLASH_EMIT_WORDSKIP_LIVE lines, expected one per rank: 4", "rank 2 has no live line")

    def test_a_count_equal_to_tp_size_still_fails_when_a_rank_logs_twice(self) -> None:
        self.assertFails(self.verdict(container_log(ranks=(0, 1, 2, 2))), "rank 2 logs 2 live lines", "rank 3 has no live line")

    def test_five_lines_fail(self) -> None:
        self.assertFails(self.verdict(container_log(ranks=(0, 1, 2, 3, 3))), "5 FLASH_EMIT_WORDSKIP_LIVE lines, expected one per rank: 4")

    def test_the_count_must_equal_the_tp_size_given_not_a_fixed_four(self) -> None:
        self.assertFails(self.verdict(container_log(), tp_size=8), "4 FLASH_EMIT_WORDSKIP_LIVE lines, expected one per rank: 8")
        self.assertEqual(self.verdict(container_log(ranks=tuple(range(8))), tp_size=8).status, reader.PASS)

    def test_a_rank_outside_the_group_fails(self) -> None:
        self.assertFails(self.verdict(container_log(ranks=(0, 1, 2, -1))), "rank -1 is not a worker rank of a TP4 run")

    def test_a_live_line_naming_the_stock_guid_fails_and_names_its_rank(self) -> None:
        log = [*container_log(ranks=(0, 1, 2)), live_line(3, STOCK)]
        self.assertFails(self.verdict(log), f"rank 3 logs guid={STOCK}, not {NEW}")

    def test_a_refusal_fails_and_carries_the_message(self) -> None:
        refusal = "HPU_GLM53_EMIT_WORDSKIP=1 cannot boot: /opt/glm53/emit/libg6_ws_lazy.so is not a readable file"
        self.assertFails(self.verdict(container_log(ranks=(), extra=(f"RuntimeError: {refusal}",))), "boot refused: /opt/glm53/emit/libg6_ws_lazy.so is not a readable file")


class TraceEvidence(Reports):
    def test_a_stock_only_trace_fails_even_with_four_live_lines(self) -> None:
        verdict = self.verdict(container_log(), self.cell_text(STOCK))
        self.assertEqual(verdict.status, reader.FAIL)
        self.assertIn(f"{RUN}: the REPLAY_TRACE_KERNEL table does not name {NEW}", verdict.reasons)
        self.assertIn(f"{RUN}: the REPLAY_TRACE_SCAN anchor does not name {NEW}", verdict.reasons)

    def test_each_table_is_judged_on_its_own(self) -> None:
        good = self.cell_text(NEW)
        anchor_only = [line.replace(f"anchor={NEW}", f"anchor={STOCK}") for line in good]
        kernel_only = [line.replace(NEW, "g6_topk_emit_other_i32") if line.startswith("  ") and NEW in line else line for line in good]
        self.assertEqual(self.verdict(container_log(), anchor_only).reasons, (f"{RUN}: the REPLAY_TRACE_SCAN anchor does not name {NEW}",))
        self.assertEqual(self.verdict(container_log(), kernel_only).reasons, (f"{RUN}: the REPLAY_TRACE_KERNEL table does not name {NEW}",))

    def kernel_table(self, *names: str, anchor: str = NEW, fill: int = reader.KERNEL_TABLE_LIMIT) -> list[str]:
        """A prompt run's KERNEL table of `names` padded with other ops to `fill` rows: at the limit it is cut off, as a real W=1024 run prints it when the emit ranks below the cut."""
        all_names = [*names, *(f"other_op_{index}" for index in range(fill - len(names)))]
        rows = [f"  0.1000     10.000       100 {name}" for name in all_names]
        return [f"REPLAY_TRACE_KERNEL run={RUN} device_busy_ms=817.596 top={len(rows)} columns=share,busy_ms,events,name", *rows,
                f"REPLAY_TRACE_SCAN run={RUN} anchor={anchor} recipes=3 launches=22 union_ms=50.531"]

    def test_the_limit_is_the_reporter_s_own(self) -> None:
        self.assertEqual(reader.KERNEL_TABLE_LIMIT, report.KERNEL_LINES)

    def test_a_table_at_its_limit_may_leave_the_new_emit_below_the_cut(self) -> None:
        verdict = self.verdict(container_log(), self.kernel_table("BatchGemm"))
        self.assertEqual((verdict.status, verdict.reasons), (reader.PASS, ()))

    def test_a_table_at_its_limit_still_needs_the_anchor(self) -> None:
        verdict = self.verdict(container_log(), self.kernel_table("BatchGemm", anchor=STOCK))
        self.assertEqual(verdict.reasons, (f"{RUN}: the REPLAY_TRACE_SCAN anchor does not name {NEW}",))

    def test_a_stock_emit_above_the_cut_fails_with_the_new_one_absent(self) -> None:
        verdict = self.verdict(container_log(), self.kernel_table("BatchGemm", STOCK))
        self.assertEqual(verdict.reasons, (f"{RUN}: the REPLAY_TRACE_KERNEL table does not name {NEW}",))

    def test_a_table_under_its_limit_is_complete_and_must_name_the_new_emit(self) -> None:
        verdict = self.verdict(container_log(), self.kernel_table("BatchGemm", fill=reader.KERNEL_TABLE_LIMIT - 1))
        self.assertEqual(verdict.reasons, (f"{RUN}: the REPLAY_TRACE_KERNEL table does not name {NEW}",))

    def test_a_cut_off_table_naming_the_new_emit_still_fails_a_control_run(self) -> None:
        verdict = self.verdict(container_log(ranks=()), self.kernel_table("BatchGemm", NEW, anchor=STOCK), expect=reader.OFF)
        self.assertEqual(verdict.reasons, (f"{RUN}: the REPLAY_TRACE_KERNEL table names {NEW}",))

    def test_a_trace_with_no_prompt_run_is_not_measured(self) -> None:
        for name, text in (("empty", []), ("decode only", self.cell_text(decode=True))):
            with self.subTest(cell=name):
                verdict = self.verdict(container_log(), text)
                self.assertEqual((verdict.status, verdict.exit_code), (reader.NOT_MEASURED, 2))
                self.assertIn("no prompt run", verdict.reasons[0])

    def test_a_prompt_run_with_no_scan_line_is_not_measured_for_the_anchor(self) -> None:
        text = [line for line in self.cell_text(NEW) if not line.startswith("REPLAY_TRACE_SCAN")]
        verdict = self.verdict(container_log(), text)
        self.assertEqual((verdict.status, verdict.reasons), (reader.NOT_MEASURED, (f"{RUN}: no REPLAY_TRACE_SCAN anchor was printed",)))

    def test_a_contradiction_outranks_a_gap(self) -> None:
        verdict = self.verdict(container_log(ranks=(0, 1)), [])
        self.assertEqual(verdict.status, reader.FAIL)
        self.assertTrue(any("no prompt run" in reason for reason in verdict.reasons))

    def test_an_empty_container_log_is_not_measured(self) -> None:
        for lines in ([], ["", "  "]):
            with self.subTest(lines=lines):
                self.assertEqual(self.verdict(lines).status, reader.NOT_MEASURED)


class Control(Reports):
    """--expect off: the same cell with the flag unset must show none of it."""

    def test_no_live_line_and_a_stock_trace_pass(self) -> None:
        self.assertEqual(self.verdict(container_log(ranks=()), self.cell_text(STOCK), reader.OFF).status, reader.PASS)

    def test_a_live_line_fails(self) -> None:
        verdict = self.verdict(container_log(ranks=(0,)), None, reader.OFF)
        self.assertEqual(verdict.status, reader.FAIL)
        self.assertIn("1 FLASH_EMIT_WORDSKIP_LIVE lines in a run with the flag unset", verdict.reasons)

    def test_the_new_guid_in_the_trace_fails(self) -> None:
        verdict = self.verdict(container_log(ranks=()), self.cell_text(NEW), reader.OFF)
        self.assertEqual(verdict.status, reader.FAIL)
        self.assertIn(f"{RUN}: the REPLAY_TRACE_SCAN anchor names {NEW}", verdict.reasons)
        self.assertIn(f"{RUN}: the REPLAY_TRACE_KERNEL table names {NEW}", verdict.reasons)


class CommandLine(Reports):
    def run_main(self, *args: str) -> tuple[int, str]:
        out = StringIO()
        with redirect_stdout(out):
            code = reader.main(list(args))
        return code, out.getvalue()

    def put(self, name: str, lines: list[str]) -> str:
        path = self.directory / name
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return str(path)

    def test_a_good_run_exits_zero_and_prints_the_verdict_line(self) -> None:
        container, cell = self.put("container.log", container_log()), self.put("cell.log", self.cell_text(NEW))
        code, out = self.run_main(container, "--tp-size", "4", "--cell-log", cell)
        self.assertEqual(code, 0)
        self.assertTrue(out.startswith("EMIT_READ PASS: expect on, tp_size=4, 4 live lines, trace read"), out)

    def test_a_short_run_exits_one_and_lists_the_reasons(self) -> None:
        code, out = self.run_main(self.put("container.log", container_log(ranks=(0, 1))), "--tp-size", "4")
        self.assertEqual(code, 1)
        self.assertIn("EMIT_READ FAIL", out)
        self.assertIn("  - 2 FLASH_EMIT_WORDSKIP_LIVE lines, expected one per rank: 4", out)

    def test_the_control_mode_is_chosen_by_expect_off(self) -> None:
        code, out = self.run_main(self.put("container.log", container_log(ranks=())), "--tp-size", "4", "--expect", "off")
        self.assertEqual((code, out.split(":")[0]), (0, "EMIT_READ PASS"))

    def test_an_unreadable_log_exits_two(self) -> None:
        code, out = self.run_main(str(self.directory / "absent.log"), "--tp-size", "4")
        self.assertEqual((code, out.startswith("NOT_MEASURED: cannot read")), (2, True))

    def test_an_unreadable_cell_log_exits_two(self) -> None:
        code, _ = self.run_main(self.put("container.log", container_log()), "--tp-size", "4", "--cell-log", str(self.directory / "absent.log"))
        self.assertEqual(code, 2)

    def test_the_tp_size_is_required_and_positive(self) -> None:
        log = self.put("container.log", container_log())
        for args in ((log,), (log, "--tp-size", "0")):
            with self.subTest(args=args), self.assertRaises(SystemExit) as raised, redirect_stderr(StringIO()):
                reader.main(list(args))
            self.assertEqual(raised.exception.code, 2)


if __name__ == "__main__":
    unittest.main()
