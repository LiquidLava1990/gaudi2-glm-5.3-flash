"""Runners and scoring of the step 20, 21 and 22 card cells, on CPU. The cells themselves wait on a card."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "cells"))

import census_table as table  # noqa: E402
import memory_cells as cells  # noqa: E402
import read_cost_cell as read_cost  # noqa: E402

MIB = 1 << 20


def census_line(rows: int, start_mib: int, top_mib: int, *sites: tuple[str, str, int], pid: int = 7,
                bucket: int = 524288, stamp: str = "2026-11-07T10:00:00.000Z", limit_mib: int = 96_894) -> str:
    body = {"rows": rows, "context": bucket, "pid": pid, "start_in_use": start_mib * MIB, "max_in_use": top_mib * MIB,
            "end_in_use": start_mib * MIB, "limit": limit_mib * MIB, "mark_steps": 1, "read_errors": 0, "unattributed": 0,
            "sites": [{"site": site, "op": op, "calls": 1, "growth": mib * MIB, "max_growth": mib * MIB,
                       "peak_advance": mib * MIB} for site, op, mib in sites]}
    return f"{stamp} (Worker_TP0 pid={pid}) GLM53_PROMPT_CENSUS {json.dumps(body)}"


def readings_at(rows: int, contexts: dict[int, int], drains: str, site: str = "s:1", op: str = "moe", *,
                sites: dict[int, int] | None = None) -> list[table.Reading]:
    made = []
    for context, peak in contexts.items():
        growth = (sites or contexts)[context]
        made.append(table.Reading(pid=1, rows=rows, bucket=524288, true_context=context, start=0, top=peak * MIB,
                                  limit=96_894 * MIB, drains=drains,
                                  sites=({"site": site, "op": op, "calls": 1, "growth": growth * MIB,
                                          "max_growth": growth * MIB, "peak_advance": growth * MIB},)))
    return made


class TrueContext(unittest.TestCase):
    def test_true_context_is_the_rows_already_sent_in_the_one_cold_request(self) -> None:
        lines = [census_line(1024, 100, 200, ("s:1", "moe", 5)) for _ in range(3)]
        self.assertEqual([r.true_context for r in table.parse(lines, "on")], [0, 1024, 2048])

    def test_each_rank_counts_separately(self) -> None:
        lines = [census_line(1024, 1, 2, pid=pid) for pid in (1, 2, 1, 2)]
        readings = table.parse(lines, "on")
        self.assertEqual([(r.pid, r.true_context) for r in readings], [(1, 0), (2, 0), (1, 1024), (2, 1024)])

    def test_lines_before_the_request_began_are_warmup(self) -> None:
        lines = [census_line(2048, 1, 2, stamp="2026-11-07T09:59:59.000Z"), census_line(2048, 1, 2)]
        readings = table.parse(lines, "on", since="2026-11-07T10:00:00+00:00")
        self.assertEqual([r.true_context for r in readings], [0])

    def test_a_context_above_the_padded_bucket_refuses(self) -> None:
        lines = [census_line(1024, 1, 2, bucket=1024) for _ in range(3)]
        with self.assertRaisesRegex(SystemExit, "more than one request"):
            table.parse(lines, "on")

    def test_a_garbled_line_refuses_with_the_line_named(self) -> None:
        with self.assertRaisesRegex(SystemExit, "GLM53_PROMPT_CENSUS"):
            table.parse(["x GLM53_PROMPT_CENSUS {broken"], "on")

    def test_unrelated_lines_are_skipped(self) -> None:
        self.assertEqual(table.parse(["GLM53_GS_FLOOR event=boot"], "on"), [])


class GraphLines(unittest.TestCase):
    def line(self, kind: str, rows: int = 1, peak_mib: int = 300, between_mib: int = 1100) -> str:
        body = {"path": "graph", "kind": kind, "rows": rows, "context": 131072, "pid": 7, "start_in_use": 1000 * MIB,
                "max_in_use": (1000 + peak_mib) * MIB, "between_peak": between_mib * MIB, "n": 1, "limit": 96_894 * MIB,
                "sites": []}
        return f"GLM53_PROMPT_CENSUS {json.dumps(body)}"

    def test_graph_lines_do_not_count_as_prompt_tokens_sent(self) -> None:
        lines = [self.line("replay"), census_line(1024, 100, 200), self.line("capture"), census_line(1024, 100, 200)]
        self.assertEqual([r.true_context for r in table.parse(lines, "on")], [0, 1024])

    def test_graph_rows_give_capture_and_replay_bytes_per_shape(self) -> None:
        rows = table.graph_rows([self.line("replay", peak_mib=300), self.line("replay", peak_mib=500), self.line("capture", peak_mib=900)])
        self.assertEqual(rows, [
            {"kind": "capture", "rows": 1, "context": 131072, "readings": 1, "peak_mib_max": 900, "between_mib_max": 100},
            {"kind": "replay", "rows": 1, "context": 131072, "readings": 2, "peak_mib_max": 500, "between_mib_max": 100}])

    def test_main_writes_the_graph_table(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            log = Path(folder) / "log.txt"
            log.write_text(self.line("replay") + "\n")
            table.main(["--log", f"on:{log}", "--out-dir", folder])
            self.assertEqual(len((Path(folder) / "graph-census.tsv").read_text().splitlines()), 2)


class Tables(unittest.TestCase):
    def test_a_reading_lands_on_its_cell_within_two_chunks(self) -> None:
        readings = readings_at(1024, {69_000: 10, 70_500: 20, 90_000: 30}, "on")
        self.assertEqual(table.cell_readings(readings, 1024, 70_000, "on"), readings[:1] + readings[1:2])

    def test_site_rows_give_bytes_per_call_site_per_cell(self) -> None:
        rows = table.site_rows(readings_at(1024, {131_072: 900}, "off"))
        self.assertEqual(rows, [{"rows": 1024, "context": 131_072, "drains": "off", "site": "s:1", "op": "moe", "readings": 1,
                                 "growth_mib_median": 900.0, "max_growth_mib": 900.0, "peak_advance_mib_median": 900.0,
                                 "calls_median": 1.0}])

    def test_tsv_has_a_header_and_one_line_per_row(self) -> None:
        text = table.to_tsv(table.site_rows(readings_at(1024, {131_072: 900}, "off")))
        self.assertEqual(len(text.splitlines()), 2)
        self.assertTrue(text.startswith("rows\tcontext\tdrains\tsite\top\t"))

    def test_need_table_takes_the_largest_drains_on_peak_of_a_cell(self) -> None:
        readings = readings_at(2048, {262_144: 1100}, "on") + readings_at(2048, {262_144: 1400}, "on")
        readings += readings_at(2048, {262_144: 5000}, "off")
        need = table.need_table(readings)
        self.assertEqual(need["entries"], [{"rows": 2048, "context": 262_144, "need_mib": 1400, "readings": 2, "saturated": 0}])

    def test_a_saturated_drains_on_reading_is_flagged(self) -> None:
        readings = [table.Reading(pid=1, rows=2048, bucket=524288, true_context=262_144, start=0, top=96_880 * MIB,
                                  limit=96_894 * MIB, drains="on", sites=())]
        self.assertEqual(table.need_table(readings)["entries"][0]["saturated"], 1)

    def test_missing_cells_are_named(self) -> None:
        missing = table.missing_cells(readings_at(512, {70_000: 1}, "on"))
        self.assertIn({"rows": 2048, "context": 500_000, "drains": "off"}, missing)
        self.assertNotIn({"rows": 512, "context": 70_000, "drains": "on"}, missing)
        self.assertEqual(len(missing), 31)


class Verdicts(unittest.TestCase):
    def matrix(self, on_1024: int, off_1024: int) -> list[table.Reading]:
        return (readings_at(512, {131_072: 400}, "off") + readings_at(512, {131_072: 380}, "on")
                + readings_at(1024, {131_072: off_1024}, "off") + readings_at(1024, {131_072: on_1024}, "on"))

    def test_the_jump_is_the_queue_when_drains_remove_most_of_it(self) -> None:
        self.assertEqual(table.jump_verdict(self.matrix(600, 3800))["answer"], "queue")

    def test_the_jump_is_not_the_queue_when_it_survives_drains(self) -> None:
        self.assertEqual(table.jump_verdict(self.matrix(3500, 3800))["answer"], "not_queue")

    def test_the_jump_is_not_measured_without_all_four_cells(self) -> None:
        self.assertEqual(table.jump_verdict(readings_at(512, {131_072: 400}, "off"))["answer"], "not_measured")

    def test_the_1995_vs_3817_gap_is_the_queue_when_it_vanishes_with_drains(self) -> None:
        readings = (readings_at(1024, {70_000: 1995, 131_072: 3817}, "off")
                    + readings_at(1024, {70_000: 700, 131_072: 760}, "on"))
        self.assertEqual(table.width_verdict(readings)["answer"], "queue")

    def test_the_gap_names_the_site_when_it_survives_drains(self) -> None:
        readings = (readings_at(1024, {70_000: 1995, 131_072: 3817}, "off")
                    + readings_at(1024, {70_000: 700, 131_072: 2400}, "on", site="indexer:9", op="indexer"))
        answer = table.width_verdict(readings)
        self.assertEqual((answer["answer"], answer["largest_site"]), ("width", "indexer:9"))

    def test_a_site_that_grows_with_context_is_named(self) -> None:
        readings = readings_at(2048, {70_000: 100, 500_000: 600}, "on", site="indexer:9", op="indexer")
        self.assertEqual(table.scaling_sites(readings), [{"rows": 2048, "site": "indexer:9", "op": "indexer",
                                                          "smallest_context": 70_000, "largest_context": 500_000,
                                                          "growth_mib_at_smallest": 100.0, "growth_mib_at_largest": 600.0}])

    def test_a_flat_site_is_not_named(self) -> None:
        self.assertEqual(table.scaling_sites(readings_at(2048, {70_000: 100, 500_000: 120}, "on")), [])

    def test_devmem_lines_are_counted_as_a_finding(self) -> None:
        self.assertEqual(table.devmem_count(["x PT_DEVMEM allocation failed", "ok"]), 1)


class CensusMain(unittest.TestCase):
    def test_main_writes_the_table_need_table_and_verdicts(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            log = Path(folder) / "log.txt"
            log.write_text("\n".join(census_line(1024, 100, 200, ("s:1", "moe", 5)) for _ in range(2)) + "\n")
            code = table.main(["--log", f"on:{log}", "--out-dir", folder])
            self.assertEqual(code, 0)
            names = {path.name for path in Path(folder).iterdir()}
            self.assertLessEqual({"census-table.tsv", "need-table.json", "census-verdicts.json"}, names)
            verdicts = json.loads((Path(folder) / "census-verdicts.json").read_text())
            self.assertEqual(verdicts["jump_512_to_1024"]["answer"], "not_measured")
            self.assertEqual(verdicts["devmem_lines"], 0)
            self.assertEqual(len(verdicts["missing_cells"]), 32)  # context 0 and 1024 are in no plan cell


class Plans(unittest.TestCase):
    def args(self, cell: str, *extra: str) -> object:
        return cells.parse_args([cell, "--quad", "A", "--image", "img:tag", "--port", "18430", "--out-dir", "/x", *extra])

    def flat(self, boots: list[cells.Boot]) -> list[str]:
        return [" ".join(step.argv) for boot in boots for step in cells.boot_steps(boot, self.args("census"))]

    def test_census_has_16_cells_in_8_boots_of_rows_and_drains(self) -> None:
        boots = cells.census_boots(self.args("census"))
        self.assertEqual(len(boots), 8)
        self.assertEqual(sorted({(b.env["VLLM_ADAPTIVE_LONG_PREFILL_SINGLE_TOKENS"],
                                  b.env["HPU_GLM53_R616_DRAIN_SITES"]) for b in boots})[0],
                         ("1024", "kda.owned,kda.sequence,moe.owned,moe.return"))

    def test_census_env_turns_the_cap_off_the_census_on_and_sets_the_drains(self) -> None:
        for boot in cells.census_boots(self.args("census")):
            self.assertEqual(boot.env["VLLM_PREFILL_CONTEXT_CELL_BUDGET"], "0")
            self.assertEqual(boot.env["HPU_GLM53_STEP_CELL_BUDGET"], "0")
            self.assertEqual(boot.env["HPU_GLM53_PREFILL_HBM_CENSUS"], "1")
            on = boot.env["HPU_GLM53_R606_RELAX_TILE_DRAINS"] == "0"
            self.assertEqual(boot.env["HPU_GLM53_R616_DRAIN_SITES"] != "none", on)

    def test_census_prompt_reaches_the_largest_context_plus_two_chunks(self) -> None:
        boot = cells.census_boots(self.args("census", "--rows", "2048", "--drains", "on"))[0]
        self.assertIn(f"--tokens {500_000 + 2 * 2048}", " ".join(boot.actions[0].argv))

    def test_rows_and_drains_filters_let_a_window_run_part_of_the_matrix(self) -> None:
        self.assertEqual(len(cells.census_boots(self.args("census", "--rows", "512,1024", "--drains", "off"))), 2)

    def test_every_boot_is_bounded(self) -> None:
        for boots in (cells.census_boots(self.args("census")), cells.owner_boots(self.args("owner")),
                      cells.runahead_boots(self.args("runahead"))):
            for boot in boots:
                for step in cells.boot_steps(boot, self.args("census")):
                    self.assertGreater(step.timeout_s, 0, step.label)
                    self.assertLessEqual(step.timeout_s, cells.MAX_STEP_S, step.label)

    def test_owner_arms_differ_only_by_the_flag(self) -> None:
        off, on = cells.owner_boots(self.args("owner"))
        self.assertEqual((off.env.pop("HPU_GLM53_Q4_SHARED_GATHER_OWNER"), on.env.pop("HPU_GLM53_Q4_SHARED_GATHER_OWNER")),
                         ("0", "1"))
        self.assertEqual(off.env, on.env)

    def test_runahead_is_flag_off_c16_then_two_fresh_flag_on_boots(self) -> None:
        boots = cells.runahead_boots(self.args("runahead", "--margin-mib", "768", "--sample-every", "2"))
        self.assertEqual([b.env.get("HPU_GLM53_RUNAHEAD_CREDIT") for b in boots], ["0", "1", "1"])
        self.assertEqual([[a.label for a in b.actions] for b in boots], [["c16"], ["c16", "gates"], ["gates"]])
        self.assertEqual(boots[1].env["HPU_GLM53_RUNAHEAD_MARGIN_MIB"], "768")
        self.assertEqual(boots[1].env["HPU_GLM53_RUNAHEAD_SAMPLE_EVERY"], "2")

    def test_every_boot_goes_through_lab_with_its_quad_image_and_port(self) -> None:
        boot = cells.owner_boots(self.args("owner", "--env", "A=1"))[0]
        steps = cells.boot_steps(boot, self.args("owner"))
        self.assertEqual(steps[0].label, "up")
        self.assertIn("lab.py", steps[0].argv[1])
        self.assertTrue({"--quad", "A", "--image", "img:tag", "--port", "18430"} <= set(steps[0].argv))
        self.assertEqual(steps[-1].label, "down")

    def test_replica_ports_refuse(self) -> None:
        for port in ("18274", "18275"):
            with self.assertRaisesRegex(SystemExit, "replica"):
                cells.parse_args(["census", "--quad", "A", "--image", "i", "--port", port, "--out-dir", "/x"])

    def test_an_env_file_is_loaded_and_env_overrides_win(self) -> None:
        with tempfile.NamedTemporaryFile("w", suffix=".env") as handle:
            handle.write("A=1\nB=2\n")
            handle.flush()
            args = self.args("owner", "--env-file", handle.name, "--env", "B=3")
            self.assertEqual(cells.extra_env(args), {"A": "1", "B": "3"})

    def test_a_cell_flag_wins_over_the_extra_env(self) -> None:
        args = self.args("owner", "--env", "HPU_GLM53_Q4_SHARED_GATHER_OWNER=7")
        self.assertEqual(cells.owner_boots(args)[0].env["HPU_GLM53_Q4_SHARED_GATHER_OWNER"], "0")

    def test_dry_run_runs_nothing_and_prints_every_step(self) -> None:
        with mock.patch("subprocess.run") as run, mock.patch("builtins.print") as printed:
            self.assertEqual(cells.main(["owner", "--quad", "A", "--image", "i", "--port", "18430", "--out-dir", "/x", "--dry-run"]), 0)
        run.assert_not_called()
        plan = json.loads(printed.call_args.args[0])
        self.assertTrue(plan["dry_run"])
        self.assertEqual(plan["waiting_on"], "a card window on quad A")
        self.assertEqual([step["label"] for step in plan["steps"]][:3], ["up", "gates", "down"])

    def test_down_runs_after_a_failed_action(self) -> None:
        boot = cells.owner_boots(self.args("owner"))[0]

        def fake_run(argv: list[str], **kwargs: object) -> object:
            return mock.Mock(returncode=1 if "launch_gates.py" in " ".join(argv) else 0)

        with mock.patch("subprocess.run", side_effect=fake_run) as run:
            results = cells.run_boot(boot, self.args("owner"))
        self.assertIn("down", run.call_args_list[-1].args[0])
        self.assertEqual([(r["label"], r["returncode"]) for r in results], [("up", 0), ("gates", 1), ("down", 0)])

    def test_an_action_timeout_is_recorded_and_down_still_runs(self) -> None:
        boot = cells.owner_boots(self.args("owner"))[0]

        def fake_run(argv: list[str], **kwargs: object) -> object:
            if "launch_gates.py" in " ".join(argv):
                raise cells.subprocess.TimeoutExpired(argv, 1)
            return mock.Mock(returncode=0)

        with mock.patch("subprocess.run", side_effect=fake_run):
            results = cells.run_boot(boot, self.args("owner"))
        self.assertEqual([(r["label"], r["returncode"]) for r in results], [("up", 0), ("gates", "timeout"), ("down", 0)])

    def test_a_failed_boot_skips_its_actions_but_still_goes_down(self) -> None:
        boot = cells.owner_boots(self.args("owner"))[0]
        with mock.patch("subprocess.run", return_value=mock.Mock(returncode=1)):
            results = cells.run_boot(boot, self.args("owner"))
        self.assertEqual([r["label"] for r in results], ["up", "down"])


FLOOR = "(W pid={pid}) GLM53_GS_FLOOR event={event} rows={rows} floor_mib=1 retained_peak_mib={peak} pid={pid}"


def owner_log(graphs: int, peak: int, owners: int) -> list[str]:
    lines = [FLOOR.format(pid=pid, event="seal", rows=graphs, peak=1) for pid in (1, 2)]
    lines += [FLOOR.format(pid=pid, event="prompt", rows=1024, peak=peak) for pid in (1, 2)]
    lines += ["GLM53_Q4_SHARED_OWNER shape=(1,) bytes=1 owners=1"] * owners
    return lines


class OwnerScore(unittest.TestCase):
    def score(self, off: list[str], on: list[str], passed: bool = True) -> dict:
        return cells.owner_verdict(off, on, gates_passed=passed)

    def test_pass_when_the_owner_fired_and_the_peak_fell(self) -> None:
        result = self.score(owner_log(19, 5000, 0), owner_log(19, 3500, 2))
        self.assertEqual((result["verdict"], result["retained_peak_mib"]), ("PASS", {"off": 5000, "on": 3500}))

    def test_no_owner_line_is_vacuous_not_a_pass(self) -> None:
        self.assertEqual(self.score(owner_log(19, 5000, 0), owner_log(19, 5000, 0))["verdict"], "VACUOUS")

    def test_a_different_graph_count_is_not_comparable(self) -> None:
        self.assertEqual(self.score(owner_log(19, 5000, 0), owner_log(20, 3000, 2))["verdict"], "NOT_COMPARABLE")

    def test_a_higher_peak_fails(self) -> None:
        self.assertEqual(self.score(owner_log(19, 5000, 0), owner_log(19, 5200, 2))["verdict"], "FAIL")

    def test_failed_gates_fail(self) -> None:
        self.assertEqual(self.score(owner_log(19, 5000, 0), owner_log(19, 3500, 2), False)["verdict"], "FAIL")


SAT = "GLM53_GS_SATURATION rows={rows} context=65536 saturated={saturated} measured_mib=1"


class RunaheadScore(unittest.TestCase):
    def lines(self, saturated: int, total: int = 2000, rows: int = 2048) -> list[str]:
        return [SAT.format(rows=rows, saturated=1 if index < saturated else 0) for index in range(total)]

    def verdict(self, lines: list[str], on_s: float = 100.0, off_s: float = 100.0, passes: tuple[bool, bool] = (True, True)) -> dict:
        return cells.runahead_verdict(lines, hot_on_s=on_s, hot_off_s=off_s, gates_passed=passes)

    def test_pass(self) -> None:
        self.assertEqual(self.verdict(self.lines(1), 102.9)["verdict"], "PASS")

    def test_saturation_at_exactly_the_bar_fails(self) -> None:
        self.assertEqual(self.verdict(self.lines(2))["verdict"], "FAIL")

    def test_other_rows_are_not_scored(self) -> None:
        result = self.verdict(self.lines(500, rows=1024) + self.lines(0, 1000))
        self.assertEqual((result["verdict"], result["saturated_readings"]), ("PASS", 0))

    def test_no_2048_reading_is_not_a_pass(self) -> None:
        self.assertEqual(self.verdict(self.lines(0, rows=1024))["verdict"], "NOT_MEASURED")

    def test_hot_c16_above_three_percent_fails(self) -> None:
        self.assertEqual(self.verdict(self.lines(0), 103.1)["verdict"], "FAIL")

    def test_either_boot_failing_its_gates_fails(self) -> None:
        self.assertEqual(self.verdict(self.lines(0), passes=(True, False))["verdict"], "FAIL")


class ReadCostPick(unittest.TestCase):
    def pick(self, read_ns: float, growth_mib: float = 40.0, margin_mib: int = 512) -> dict:
        return read_cost.pick_interval(read_ns, boundaries=866, forward_ms=2341.0, budget_fraction=0.003,
                                       margin_mib=margin_mib, max_boundary_growth_mib=growth_mib)

    def test_a_cheap_read_samples_every_boundary(self) -> None:
        self.assertEqual(self.pick(5_000)["sample_every"], 1)

    def test_a_slow_read_samples_sparsely_enough_to_fit_the_budget(self) -> None:
        # 866 reads x 100 us = 86.6 ms; the budget is 7.0 ms of a 2,341 ms forward: every 13th boundary.
        result = self.pick(100_000, growth_mib=30.0)
        self.assertEqual((result["sample_every"], result["fits"]), (13, True))

    def test_the_margin_bounds_how_sparse_sampling_may_be(self) -> None:
        # 13 boundaries x 200 MiB each would overrun a 512 MiB margin between reads.
        result = self.pick(100_000, growth_mib=200.0)
        self.assertEqual((result["sample_every"], result["ceiling"], result["fits"]), (13, 2, False))
        self.assertIn("HPU_GLM53_RUNAHEAD_MARGIN_MIB", result["action"])

    def test_no_growth_measured_leaves_the_ceiling_at_the_forward(self) -> None:
        self.assertEqual(self.pick(100_000, growth_mib=0.0)["ceiling"], 866)

    def test_percentiles(self) -> None:
        summary = read_cost.summarize_ns(list(range(1, 101)))
        self.assertEqual((summary["p50_ns"], summary["p99_ns"], summary["max_ns"], summary["n"]), (50, 99, 100, 100))


class ReadCostRehearsal(unittest.TestCase):
    def test_cpu_rehearsal_runs_and_never_counts_as_the_cell(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            args = read_cost.parse_args(["--device", "cpu", "--reads", "50", "--boundaries", "20", "--tile-rows", "8",
                                         "--out", f"{folder}/out.json"])
            result = read_cost.run(args)
        self.assertEqual((result["card_verdict"], result["pass"]), (False, False))
        self.assertEqual(result["idle"]["memory_stats"]["n"], 50)
        self.assertEqual(result["under_load"]["read"]["n"], 20)
        self.assertGreaterEqual(result["pick"]["sample_every"], 1)

    def test_a_card_run_with_a_pick_that_fits_passes(self) -> None:
        self.assertTrue(read_cost.cell_passes("hpu", {"fits": True}))
        self.assertFalse(read_cost.cell_passes("hpu", {"fits": False}))
        self.assertFalse(read_cost.cell_passes("cpu", {"fits": True}))


if __name__ == "__main__":
    unittest.main()
