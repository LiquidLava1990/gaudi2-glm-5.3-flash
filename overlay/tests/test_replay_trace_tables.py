"""WIDE-PROMPT-KERNEL-PLAN step 1: the trace reader's recipe tables, scan union, one call's stage timeline and idle-gap classes, on small
synthetic device records (every number is worked out in the comments) and, when the C1p trace is mounted, on the real one. No card."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "cells"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import replay_trace_report as report  # noqa: E402
import trace_verdict  # noqa: E402
from test_replay_trace_report import RUN, ReportCase, trace  # noqa: E402

C1P_TRACE = Path("/c1-out/c1p-20261107T080120Z/trace/replaytrace-prompt-1024x8192-n3-rank0.pt.trace.json.gz")


def op(name: str, recipe: int, start: float, end: float, *, cat: str = "kernel") -> dict:
    """One device event of a recipe (recipe 0 = a DMA descriptor or host copy that belongs to no recipe)."""
    args = {"recipeId": recipe, "recipeHandle": "9", "EventName": f"id_1_{name} Unique ID:1/{name}/1", "HW event name": "tpc"}
    return {"ph": "X", "name": name, "cat": cat, "pid": 0, "tid": 7, "ts": start, "dur": end - start, "args": args}


STOCK_EMIT, WORDSKIP_EMIT = "g6_topk_emit_i32", "g6_topk_emit_wordskip_i32"


def scan_call(recipe: int, at: float, tail_end: float = 350.0, emit: str = STOCK_EMIT) -> list[dict]:
    """A scan call: routing 20 us, GEMM + score 80, keys 30, state 70, two emit events (last ends at +320) and a merge kernel to ``tail_end``."""
    return [op("range_i32", recipe, at, at + 10), op("GEMM", recipe, at + 20, at + 60), op("mla_topkkeys_i32", recipe, at + 100, at + 110),
            op("mla_topkstate_i32", recipe, at + 130, at + 190), op(emit, recipe, at + 200, at + 300),
            op(emit, recipe, at + 210, at + 320), op("gather_elements_fwd_i32", recipe, at + 330, at + tail_end)]


def view_of(ops: list[dict], window_us: float) -> trace_verdict.TraceView:
    view = report.window_view(trace(window_us, ops))
    assert view is not None
    return view


# Recipe 100: three scan calls (walls 350, 360, 340 us; busy 260, 270, 250). Recipe 200: two scan calls (260 busy each). Recipe 300: a
# kernel overlapping recipe 100's first call by 70 us, then a second call. Recipe 0: one host copy. Scan union = 260+270+250+520 = 1300 us.
SCAN_OPS = [*scan_call(100, 1000), *scan_call(100, 3000, 360.0), *scan_call(100, 5000, 340.0), *scan_call(200, 7000), *scan_call(200, 9000),
            op("moe_gemm", 300, 1100, 1200), op("moe_gemm", 300, 11000, 11050), op("moe_gemm", 300, 11060, 11100),
            op("H2D pdma_tx_commands 0 apiId=0 chunk=3", 0, 12000, 12100, cat="hpu_op")]
WINDOW_US = 14_000.0


class RecipeTables(unittest.TestCase):
    def setUp(self) -> None:
        self.tables = report.trace_tables(view_of(SCAN_OPS, WINDOW_US))
        self.rows = {row["recipe"]: row for row in self.tables["recipes"]}

    def test_every_recipe_has_its_launches_events_busy_and_call_walls_and_recipe_zero_has_no_row(self) -> None:
        self.assertEqual(set(self.rows), {"100", "200", "300"})
        row = self.rows["100"]
        self.assertEqual((row["scan"], row["launches"], row["events"]), (True, 3, 21))
        self.assertAlmostEqual(row["busy_ms"], 0.780, places=3)
        self.assertAlmostEqual(row["wall_ms"], 1.050, places=3)
        self.assertAlmostEqual(row["call_ms_mean"], 0.350, places=4)
        self.assertEqual((row["call_ms_min"], row["call_ms_max"]), (0.340, 0.360))

    def test_a_recipe_with_no_emit_is_not_scan_and_its_calls_split_on_the_silence(self) -> None:
        row = self.rows["300"]
        self.assertEqual((row["scan"], row["launches"], row["events"]), (False, 2, 3))
        self.assertAlmostEqual(row["busy_ms"], 0.190, places=3)

    def test_the_rows_are_busiest_first(self) -> None:
        self.assertEqual([row["recipe"] for row in self.tables["recipes"]], ["100", "200", "300"])


class CallSplit(unittest.TestCase):
    def test_a_recipe_is_silent_for_more_than_the_call_gap_between_two_calls(self) -> None:
        spans = [(0.0, 10.0, 0), (60.0, 70.0, 1), (170.0, 180.0, 2), (290.0, 300.0, 3), (700.0, 710.0, 4)]
        calls = report.split_calls(spans)  # 60 - 10 = 50 and 170 - 70 = 100 stay one call, 290 - 180 = 110 and 700 - 300 split it
        self.assertEqual([(call.start, call.end, call.first, call.stop) for call in calls],
                         [(0.0, 180.0, 0, 3), (290.0, 300.0, 3, 4), (700.0, 710.0, 4, 5)])

    def test_a_long_event_holds_the_call_open_for_the_events_that_start_inside_it(self) -> None:
        spans = [(0.0, 500.0, 0), (200.0, 210.0, 1), (600.0, 610.0, 2)]
        self.assertEqual([(call.first, call.stop) for call in report.split_calls(spans)], [(0, 3)])


class ScanUnion(unittest.TestCase):
    def setUp(self) -> None:
        self.scan = report.trace_tables(view_of(SCAN_OPS, WINDOW_US))["scan"]

    def test_the_scan_is_the_union_of_the_recipes_that_hold_the_emit_kernel(self) -> None:
        self.assertEqual((self.scan["anchor"], self.scan["recipes"], self.scan["launches"]), ("g6_topk_emit_i32", 2, 5))
        self.assertAlmostEqual(self.scan["union_ms"], 1.300, places=3)
        self.assertAlmostEqual(self.scan["wall_ms"], 1.050 + 0.700, places=3)

    def test_the_rest_is_everything_else_including_host_copies_and_the_union_counts_an_overlap_once(self) -> None:
        self.assertAlmostEqual(self.scan["rest_ms"], 0.290, places=3)  # recipe 300: 100 + 50 + 40, recipe 0: 100
        self.assertAlmostEqual(self.scan["share"], 1.300 / 1.520, places=4)  # busy = 1300 + 290 - 70 of overlap

    def test_a_trace_with_no_scan_recipe_has_an_empty_scan_and_no_timeline(self) -> None:
        tables = report.trace_tables(view_of([op("moe_gemm", 300, 0, 100)], 1000.0))
        self.assertEqual((tables["scan"]["recipes"], tables["scan"]["launches"], tables["scan"]["union_ms"]), (0, 0, 0.0))
        self.assertIsNone(tables["stage_timeline"])


class WordSkipEmitIsAScanToo(unittest.TestCase):
    """Step 6, HPU_GLM53_EMIT_WORDSKIP=1: the scan recipes end in g6_topk_emit_wordskip_i32, and every table must find them as it finds the stock emit."""

    @staticmethod
    def tables(emit_for_recipe: dict[int, str]) -> dict:
        ops = [event for recipe, emit in emit_for_recipe.items() for at in (1000, 3000) for event in scan_call(recipe, at, emit=emit)]
        return report.trace_tables(view_of(ops, 6000.0))

    def test_a_word_skip_trace_reads_as_the_same_scan_with_the_new_anchor(self) -> None:
        scan = self.tables({100: WORDSKIP_EMIT})["scan"]
        self.assertEqual((scan["anchor"], scan["recipes"], scan["launches"]), (WORDSKIP_EMIT, 1, 2))
        self.assertAlmostEqual(scan["union_ms"], 0.520, places=3)

    def test_the_numbers_do_not_depend_on_which_emit_ran(self) -> None:
        stock, wordskip = self.tables({100: STOCK_EMIT}), self.tables({100: WORDSKIP_EMIT})
        self.assertEqual({key: value for key, value in wordskip["scan"].items() if key != "anchor"},
                         {key: value for key, value in stock["scan"].items() if key != "anchor"})
        self.assertEqual(wordskip["recipes"], stock["recipes"])
        self.assertEqual(wordskip["stage_timeline"]["wall_us"], stock["stage_timeline"]["wall_us"])

    def test_the_stage_timeline_names_the_word_skip_kernel_as_the_emit_anchor(self) -> None:
        stages = {stage["stage"]: stage for stage in self.tables({100: WORDSKIP_EMIT})["stage_timeline"]["stages"]}
        self.assertEqual((stages["emit"]["anchor"], stages["emit"]["start_us"], stages["emit"]["length_us"]), (WORDSKIP_EMIT, 200.0, 120.0))
        self.assertEqual((stages["merge"]["start_us"], stages["merge"]["length_us"]), (320.0, 30.0))
        self.assertEqual(list(stages), ["routing", "gemm_score", "topkkeys", "topkstate", "emit", "merge"])

    def test_a_trace_with_both_emits_names_both_anchors(self) -> None:
        scan = self.tables({100: STOCK_EMIT, 200: WORDSKIP_EMIT})["scan"]
        self.assertEqual((scan["anchor"], scan["recipes"], scan["launches"]), (f"{STOCK_EMIT}+{WORDSKIP_EMIT}", 2, 4))


class StageTimeline(unittest.TestCase):
    def setUp(self) -> None:
        self.timeline = report.trace_tables(view_of(SCAN_OPS, WINDOW_US))["stage_timeline"]
        self.stages = {stage["stage"]: stage for stage in self.timeline["stages"]}

    def test_the_call_is_the_median_wall_launch_of_the_scan_recipe_with_the_most_launches(self) -> None:
        self.assertEqual((self.timeline["recipe"], self.timeline["calls"], self.timeline["call_index"]), ("100", 3, 0))
        self.assertEqual(self.timeline["wall_us"], 350.0)

    def test_each_stage_runs_from_its_anchor_kernel_to_the_next_and_merge_starts_at_the_last_emit_end(self) -> None:
        got = [(stage["stage"], stage["start_us"], stage["length_us"]) for stage in self.timeline["stages"]]
        self.assertEqual(got, [("routing", 0.0, 20.0), ("gemm_score", 20.0, 80.0), ("topkkeys", 100.0, 30.0), ("topkstate", 130.0, 70.0),
                               ("emit", 200.0, 120.0), ("merge", 320.0, 30.0)])
        self.assertEqual(sum(stage["length_us"] for stage in self.timeline["stages"]), self.timeline["wall_us"])

    def test_the_stages_name_their_anchor(self) -> None:
        self.assertEqual((self.stages["routing"]["anchor"], self.stages["gemm_score"]["anchor"], self.stages["merge"]["anchor"]),
                         ("call start", "GEMM", "emit end"))

    def test_a_tie_in_launches_goes_to_the_lowest_recipe_id(self) -> None:
        tied = [*scan_call(200, 1000), *scan_call(100, 3000), *scan_call(1000, 5000)]
        timeline = report.trace_tables(view_of(tied, 7000.0))["stage_timeline"]
        self.assertEqual(timeline["recipe"], "100")

    def test_an_anchor_the_call_never_ran_leaves_its_stage_out(self) -> None:
        call = [event for event in scan_call(100, 1000) if event["name"] != "GEMM"]
        stages = report.trace_tables(view_of(call, 3000.0))["stage_timeline"]["stages"]
        self.assertEqual([stage["stage"] for stage in stages], ["routing", "topkkeys", "topkstate", "emit", "merge"])
        self.assertEqual(stages[0]["length_us"], 100.0)


# One window of 40,000 us. Gaps (us): lead 100 | k1 100-200 | 10 | k2 210-300 | 20 | k3 320-400 | 100 | H2D 500-520 | 5000 | sibo 5520-5600 |
# 400 | H2D 6000-6020 | 6000 | sibo 12020-12100 | 20000 | DmaMemcpy of recipe 61548 32100-32200 | tail 7800.
GAP_OPS = [op("k1", 7, 100, 200), op("k2", 7, 210, 300), op("k3", 7, 320, 400),
           op("H2D pdma_tx_commands 0 apiId=0 chunk=5", 0, 500, 520, cat="hpu_op"), op("sibo stream_index 1 chunk_id 0", 0, 5520, 5600, cat="hpu_op"),
           op("H2D pdma_tx_commands 0 apiId=0 chunk=8", 0, 6000, 6020, cat="hpu_op"), op("sibo stream_index 1 chunk_id 0", 0, 12020, 12100, cat="hpu_op"),
           op("DmaMemcpy", 61548, 32100, 32200, cat="hpu_op")]


class GapClasses(unittest.TestCase):
    def setUp(self) -> None:
        self.gaps = report.trace_tables(view_of(GAP_OPS, 40_000.0))["gap_classes"]
        self.classes = {(row["band"], row["before"], row["after"]): row for row in self.gaps["classes"]}

    def test_gaps_under_the_threshold_are_one_class_with_their_count_and_time(self) -> None:
        under = self.classes[("<30us", "*", "*")]
        self.assertEqual((under["gaps"], under["total_ms"], under["max_ms"]), (2, 0.030, 0.020))
        self.assertEqual(self.gaps["threshold_us"], 30.0)

    def test_longer_gaps_are_classed_by_size_band_and_the_kernels_around_them_with_chunk_numbers_dropped(self) -> None:
        self.assertEqual(set(self.classes) - {("<30us", "*", "*")}, {
            ("10ms+", "sibo", "DmaMemcpy@61548"), ("1ms-10ms", "H2D", "sibo"), ("1ms-10ms", "DmaMemcpy@61548", "<window end>"),
            ("30us-1ms", "sibo", "H2D"), ("30us-1ms", "<window start>", "k1@7"), ("30us-1ms", "k3@7", "H2D")})
        pair = self.classes[("1ms-10ms", "H2D", "sibo")]
        self.assertEqual((pair["gaps"], pair["total_ms"], pair["max_ms"]), (2, 11.0, 6.0))
        self.assertEqual(self.classes[("10ms+", "sibo", "DmaMemcpy@61548")]["total_ms"], 20.0)

    def test_the_classes_are_largest_total_first_under_class_leading(self) -> None:
        rows = self.gaps["classes"]
        self.assertEqual(rows[0]["band"], "<30us")
        totals = [row["total_ms"] for row in rows[1:]]
        self.assertEqual(totals, sorted(totals, reverse=True))
        self.assertEqual((rows[-2]["before"], rows[-1]["before"]), ("<window start>", "k3@7"))  # a tie in time goes by label

    def test_the_totals_reconcile_with_the_window_idle(self) -> None:
        self.assertEqual((self.gaps["over_gaps"], self.gaps["class_count"]), (7, 6))
        self.assertAlmostEqual(self.gaps["over_ms"], 39.4, places=3)
        self.assertAlmostEqual(self.gaps["idle_ms"], 39.43, places=3)
        self.assertEqual(sum(row["gaps"] for row in self.gaps["classes"]), 9)

    def test_gap_edges_lists_every_idle_stretch_with_the_events_on_each_side_and_none_at_a_window_edge(self) -> None:
        view = view_of(GAP_OPS, 40_000.0)
        edges = trace_verdict.gap_edges(view, trace_verdict.busy_runs(view.device))
        self.assertEqual(len(edges), 9)
        self.assertEqual((edges[0][1], edges[-1][3]), (None, None))
        self.assertEqual([round(high - low) for low, _, high, _ in edges], [100, 10, 20, 100, 5000, 400, 6000, 20000, 7800])
        self.assertEqual(trace_verdict.largest_gaps(view, trace_verdict.busy_runs(view.device))[0]["length_ms"], 20.0)

    def test_the_kept_classes_are_capped_but_the_count_is_not(self) -> None:
        many = [op(f"k{n}", 7, n * 1000.0, n * 1000.0 + 10) for n in range(report.GAP_CLASS_KEEP + 20)]
        gaps = report.trace_tables(view_of(many, 100_000.0))["gap_classes"]
        self.assertEqual(len(gaps["classes"]), report.GAP_CLASS_KEEP + 1)  # the under class, then the kept ones
        self.assertGreater(gaps["class_count"], report.GAP_CLASS_KEEP)


class RenderedLines(ReportCase):
    def lines(self) -> list[str]:
        self.write(RUN, trace(WINDOW_US, SCAN_OPS), 14.0)
        return report.render(report.build_report(self.directory, [RUN], 4, []))

    def test_the_new_tables_carry_their_own_prefixes_and_the_old_lines_keep_theirs(self) -> None:
        lines = self.lines()
        prefixes = [line.split(" ", 1)[0] for line in lines if not line.startswith(" ")]
        self.assertEqual(prefixes[0], "REPLAY_TRACE_RUN")
        self.assertEqual(prefixes[-1], "REPLAY_TRACE_SUMMARY")
        for prefix in ("REPLAY_TRACE_SCAN", "REPLAY_TRACE_RECIPE", "REPLAY_TRACE_STAGE", "REPLAY_TRACE_GAP"):
            self.assertEqual(prefixes.count(prefix), 1, prefix)
        self.assertLess(prefixes.index("REPLAY_TRACE_FAMILY"), prefixes.index("REPLAY_TRACE_SCAN"))

    def test_the_scan_line_gives_recipes_launches_union_and_share(self) -> None:
        scan = next(line for line in self.lines() if line.startswith("REPLAY_TRACE_SCAN"))
        for field in (f"run={RUN}", "anchor=g6_topk_emit_i32", "recipes=2", "launches=5", "union_ms=1.3", "rest_ms=0.29", "wall_ms=1.75", "share=0.8553"):
            self.assertIn(field, scan)

    def test_the_recipe_stage_and_gap_rows_are_indented_under_their_header(self) -> None:
        text = "\n".join(self.lines())
        self.assertRegex(text, r"REPLAY_TRACE_RECIPE run=\S+ recipes=3 shown=3 columns=\S+\n  +100 +scan +3 +21 +0\.780 +1\.050 +0\.3500 +0\.3400 +0\.3600\n")
        self.assertRegex(text, r"REPLAY_TRACE_STAGE run=\S+ recipe=100 call_index=0 calls=3 wall_us=350\.0 columns=\S+\n  +routing +0\.0 +20\.0 +call start\n")
        self.assertRegex(text, r"REPLAY_TRACE_GAP run=\S+ threshold_us=30\.0 idle_ms=\S+ over_gaps=\d+ over_ms=\S+ classes=\d+ shown=\d+ columns=\S+\n  +<30us +\d+ +\S+ +\S+ \* -> \*\n")

    def test_a_run_with_no_recipes_prints_only_the_gap_table(self) -> None:
        self.write(RUN, trace(10_000.0, [op("fused", 0, 0, 8000)]), 10.0)
        lines = report.render(report.build_report(self.directory, [RUN], 4, []))
        prefixes = {line.split(" ", 1)[0] for line in lines if not line.startswith(" ")}
        self.assertIn("REPLAY_TRACE_GAP", prefixes)
        self.assertFalse(prefixes & {"REPLAY_TRACE_SCAN", "REPLAY_TRACE_RECIPE", "REPLAY_TRACE_STAGE"})

    def test_the_tables_are_in_the_json_report_and_a_missing_run_has_empty_ones(self) -> None:
        self.write(RUN, trace(WINDOW_US, SCAN_OPS), 14.0)
        result = report.build_report(self.directory, [RUN, "decode-16x8192-n5"], 4, [])
        found, missing = result["runs"]
        self.assertEqual((found["scan"]["recipes"], len(found["recipes"]), found["stage_timeline"]["recipe"]), (2, 3, "100"))
        self.assertEqual((missing["recipes"], missing["scan"], missing["stage_timeline"], missing["gap_classes"]), ([], None, None, None))


@unittest.skipUnless(C1P_TRACE.is_file(), "the C1p trace is not mounted (docker run -v $L/c1-out:/c1-out:ro)")
class C1pTrace(unittest.TestCase):
    """The acceptance read of WIDE-PROMPT-KERNEL-PLAN step 1 on the 8192-wide prompt replay."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.tables = report.trace_tables(report.window_view(trace_verdict.load_events(C1P_TRACE)))
        cls.rows = {row["recipe"]: row for row in cls.tables["recipes"]}
        cls.gaps = {(row["band"], row["before"], row["after"]): row for row in cls.tables["gap_classes"]["classes"]}

    def test_the_scan_is_15_recipes_704_launches_and_1843_ms(self) -> None:
        scan = self.tables["scan"]
        self.assertEqual((scan["recipes"], scan["launches"]), (15, 704))
        self.assertAlmostEqual(scan["union_ms"], 1843.0, delta=5.0)
        self.assertAlmostEqual(scan["rest_ms"], 787.0, delta=787.0 * 0.05)

    def test_recipe_60324_runs_88_calls_of_about_2_574_ms(self) -> None:
        row = self.rows["60324"]
        self.assertEqual(row["launches"], 88)
        self.assertAlmostEqual(row["call_ms_mean"], 2.574, delta=0.002)

    def test_the_stage_timeline_is_one_scan_call_of_recipe_60324(self) -> None:
        timeline = self.tables["stage_timeline"]
        self.assertEqual((timeline["recipe"], timeline["calls"]), ("60324", 88))
        stages = {stage["stage"]: stage["length_us"] for stage in timeline["stages"]}
        for stage, plan_us in (("routing", 103), ("gemm_score", 787), ("topkkeys", 33), ("topkstate", 527), ("emit", 818), ("merge", 302)):
            self.assertAlmostEqual(stages[stage], plan_us, delta=4.0, msg=stage)

    def test_the_idle_gap_classes_are_the_ones_the_plan_names(self) -> None:
        under = self.gaps[("<30us", "*", "*")]
        self.assertAlmostEqual(under["total_ms"], 59.7, delta=0.1)
        bursts = [row for key, row in self.gaps.items() if key[0] == "10ms+" and key[1] != "<window start>"]
        self.assertEqual(sum(row["gaps"] for row in bursts), 5)
        self.assertAlmostEqual(sum(row["total_ms"] for row in bursts), 113.0, delta=1.0)
        stall = [row for key, row in self.gaps.items() if key[2] == "DmaMemcpy@61548"]
        self.assertEqual(sum(row["gaps"] for row in stall), 11)
        self.assertAlmostEqual(sum(row["total_ms"] for row in stall), 42.4, delta=0.2)
        self.assertAlmostEqual(self.gaps[("10ms+", "<window start>", "H2D")]["total_ms"], 10.9, delta=0.05)


if __name__ == "__main__":
    unittest.main()
