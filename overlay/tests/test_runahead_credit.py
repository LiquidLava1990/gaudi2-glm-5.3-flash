"""Step 22: credit-bounded run-ahead. Only synchronize moves; the mark_step sequence is the base's, flag on or off."""

from __future__ import annotations

import importlib.util
import logging
import os
import random
import sys
import unittest
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest import mock

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from glm53_mixed_graph import capture_context  # noqa: E402
from overlay_testlib import OVERLAY_ROOT, load_base, load_overlay  # noqa: E402

CREDIT_FILE = "glm53_runahead_credit.py"
R589_FILE = "glm53_r589_bounded_prefill.py"
MIB = 1 << 20
LIMIT = 96_894 * MIB
HPU_TENSOR = SimpleNamespace(device=SimpleNamespace(type="hpu"))
# The labels and drain arguments of the real call sites, with relax on: tile sites ask for no drain.
BOUNDARY_TRACE = (("kda.owned", True), ("kda.inputs", False), ("kda.tile", False), ("kda.tile", False),
                  ("kda.sequence", True), ("moe.plan", False), ("moe.owned", True), ("moe.tile", False),
                  ("moe.tile", False), ("moe.return", True))


class FakeAllocator:
    """Scripted InUse readings; the last repeats. A reading that is an exception is raised."""

    def __init__(self, *free_mib: object) -> None:
        self.script = list(free_mib)
        self.reads = 0

    def __call__(self) -> dict[str, int]:
        value = self.script[min(self.reads, len(self.script) - 1)]
        self.reads += 1
        if isinstance(value, Exception):
            raise value
        return {"Limit": LIMIT, "InUse": LIMIT - int(value) * MIB, "MaxInUse": LIMIT}


def credit(allocator: Callable[[], dict[str, int]], margin_mib: int = 512, every: int = 1) -> object:
    module = load_overlay(CREDIT_FILE)
    return module.RunaheadCredit(allocator, margin_mib * MIB, every)


class Decisions(unittest.TestCase):
    def test_no_drain_with_room(self) -> None:
        self.assertFalse(credit(FakeAllocator(5_000)).should_drain("moe.tile"))

    def test_drain_near_the_limit(self) -> None:
        self.assertTrue(credit(FakeAllocator(511)).should_drain("moe.tile"))

    def test_exactly_at_the_margin_does_not_drain(self) -> None:
        self.assertFalse(credit(FakeAllocator(512)).should_drain("moe.tile"))

    def test_read_error_drains_and_logs(self) -> None:
        subject = credit(FakeAllocator(RuntimeError("allocator gone")))
        with self.assertLogs("vllm.r589", "WARNING") as logs:
            self.assertTrue(subject.should_drain("moe.tile"))
        self.assertIn("read_error", logs.output[0])
        self.assertIn("allocator gone", logs.output[0])
        self.assertEqual(subject.read_errors, 1)

    def test_a_reading_without_a_limit_counts_as_a_read_error(self) -> None:
        subject = credit(lambda: {"InUse": 5})
        with self.assertLogs("vllm.r589", "WARNING"):
            self.assertTrue(subject.should_drain("moe.tile"))
        self.assertEqual(subject.read_errors, 1)

    def test_a_zero_limit_counts_as_a_read_error(self) -> None:
        subject = credit(lambda: {"Limit": 0, "InUse": 0})
        with self.assertLogs("vllm.r589", "WARNING"):
            self.assertTrue(subject.should_drain("moe.tile"))
        self.assertEqual(subject.read_errors, 1)

    def test_read_errors_do_not_flood_the_log(self) -> None:
        subject = credit(FakeAllocator(RuntimeError("gone")))
        with self.assertLogs("vllm.r589", "WARNING") as logs:
            for _ in range(300):
                subject.should_drain("moe.tile")
        self.assertEqual(len(logs.output), 2)  # the 1st and the 256th
        self.assertEqual(subject.drains, 300)

    def test_decision_equals_the_plan_formula_over_a_random_forward(self) -> None:
        """Plan: drain when growth since forward start > free at start - margin. Limit - InUse < margin is the same."""
        logging.disable(logging.WARNING)
        self.addCleanup(logging.disable, logging.NOTSET)
        rng = random.Random(1106)
        margin_mib = 512
        free_at_start = 6_000
        in_use_growth = 0
        for _ in range(2_000):
            in_use_growth = max(0, in_use_growth + rng.randrange(-900, 1_000))
            free = max(0, free_at_start - in_use_growth)
            growth = free_at_start - free
            planned = growth > free_at_start - margin_mib
            self.assertEqual(credit(FakeAllocator(free)).should_drain("moe.tile"), planned, (free, growth))


class Sampling(unittest.TestCase):
    def test_every_boundary_reads_by_default(self) -> None:
        allocator = FakeAllocator(5_000)
        subject = credit(allocator)
        for _ in range(7):
            subject.should_drain("moe.tile")
        self.assertEqual((allocator.reads, subject.boundaries), (7, 7))

    def test_sampling_reads_every_nth_boundary(self) -> None:
        allocator = FakeAllocator(5_000)
        subject = credit(allocator, every=4)
        decisions = [subject.should_drain("moe.tile") for _ in range(8)]
        self.assertEqual(allocator.reads, 2)
        self.assertEqual(decisions, [False] * 8)

    def test_sampling_projects_the_unread_boundaries_from_the_largest_growth_seen(self) -> None:
        # 4 boundaries per read. The last read grew 1,800 MiB: 450 per boundary, 1,350 over the 3 unread, so 1,200 free projects negative.
        subject = credit(FakeAllocator(5_000, 4_000, 3_000, 1_200), every=4)
        results = [subject.should_drain("moe.tile") for _ in range(16)]
        self.assertEqual([index for index, drained in enumerate(results) if drained], [15])


class Parsing(unittest.TestCase):
    def setUp(self) -> None:
        self.module = load_overlay(CREDIT_FILE)

    def parse(self, **environ: str) -> object:
        return self.module.runahead_credit_from_env(environ, FakeAllocator(5_000))

    def test_off_when_unset_or_zero(self) -> None:
        self.assertIsNone(self.parse())
        self.assertIsNone(self.parse(HPU_GLM53_RUNAHEAD_CREDIT="0"))

    def test_margin_is_not_parsed_when_off(self) -> None:
        self.assertIsNone(self.parse(HPU_GLM53_RUNAHEAD_MARGIN_MIB="junk"))

    def test_defaults_when_on(self) -> None:
        subject = self.parse(HPU_GLM53_RUNAHEAD_CREDIT="1")
        self.assertEqual((subject.margin_bytes, subject.sample_every), (512 * MIB, 1))

    def test_margin_below_256_refuses_boot(self) -> None:
        with self.assertRaisesRegex(ValueError, "HPU_GLM53_RUNAHEAD_MARGIN_MIB=255: .*at least 256"):
            self.parse(HPU_GLM53_RUNAHEAD_CREDIT="1", HPU_GLM53_RUNAHEAD_MARGIN_MIB="255")

    def test_margin_of_256_is_accepted(self) -> None:
        subject = self.parse(HPU_GLM53_RUNAHEAD_CREDIT="1", HPU_GLM53_RUNAHEAD_MARGIN_MIB="256")
        self.assertEqual(subject.margin_bytes, 256 * MIB)

    def test_bad_values_refuse_boot(self) -> None:
        for environ in ({"HPU_GLM53_RUNAHEAD_CREDIT": "yes"},
                        {"HPU_GLM53_RUNAHEAD_CREDIT": "1", "HPU_GLM53_RUNAHEAD_MARGIN_MIB": "abc"},
                        {"HPU_GLM53_RUNAHEAD_CREDIT": "1", "HPU_GLM53_RUNAHEAD_SAMPLE_EVERY": "0"}):
            with self.subTest(environ), self.assertRaisesRegex(ValueError, "HPU_GLM53_RUNAHEAD_"):
                self.parse(**environ)


def r589_with_env(alias: str, environ: dict[str, str]) -> ModuleType:
    """A fresh import of the overlay's R589 under ``environ``, proving the import-time parse."""
    spec = importlib.util.spec_from_file_location(alias, OVERLAY_ROOT / R589_FILE)
    module = importlib.util.module_from_spec(spec)
    with mock.patch.dict(os.environ, environ), mock.patch.dict(sys.modules, {alias: module}):
        spec.loader.exec_module(module)
    return module


@contextmanager
def lazy_mode(events: list[tuple]) -> Iterator[None]:
    with mock.patch.dict(os.environ, {"PT_HPU_LAZY_MODE": "1"}), \
            mock.patch("habana_frameworks.torch.core.mark_step",
                       side_effect=lambda *a, **k: events.append(("mark_step", a, tuple(sorted(k.items()))))), \
            mock.patch.object(torch.hpu, "synchronize", side_effect=lambda: events.append(("synchronize",)), create=True):
        yield


def boundary_events(module: ModuleType, sites: frozenset[str], runahead: object = None, *,
                    trace: tuple[tuple[str, bool], ...] = BOUNDARY_TRACE) -> list[tuple]:
    events: list[tuple] = []
    attributes = {"DRAIN_SITES": sites}
    if hasattr(module, "RUNAHEAD"):
        attributes["RUNAHEAD"] = runahead
    with lazy_mode(events), mock.patch.multiple(module, **attributes):
        for label, drain in trace:
            module._boundary(label, HPU_TENSOR, drain=drain, total_tokens=1)
    return events


def only_mark_steps(events: list[tuple]) -> list[tuple]:
    return [event for event in events if event[0] == "mark_step"]


HOOK_IN_R589 = "RUNAHEAD" in (OVERLAY_ROOT / R589_FILE).read_text(encoding="utf-8")
needs_hook = unittest.skipUnless(HOOK_IN_R589, "plan step 22 is held: the shipped R589 carries no credit hook (see glm53_runahead_credit.py)")


class Boundary(unittest.TestCase):
    NONE: frozenset[str] = frozenset()

    def setUp(self) -> None:
        self.base = load_base(R589_FILE)
        self.overlay = load_overlay(R589_FILE)
        self.all_sites = frozenset(self.base.DRAIN_SITE_LABELS)

    @needs_hook
    def test_flag_off_is_the_default_import(self) -> None:
        self.assertIsNone(r589_with_env("r589_flag_off", {}).RUNAHEAD)

    def test_flag_off_equals_base_exactly(self) -> None:
        for sites in (self.NONE, self.all_sites):
            with self.subTest(sites=sorted(sites)):
                self.assertEqual(boundary_events(self.overlay, sites), boundary_events(self.base, sites))

    @needs_hook
    def test_mark_step_sequence_is_identical_flag_on_and_off(self) -> None:
        logging.disable(logging.WARNING)
        self.addCleanup(logging.disable, logging.NOTSET)
        rng = random.Random(22)
        expected = only_mark_steps(boundary_events(self.base, self.NONE))
        self.assertEqual(len(expected), len(BOUNDARY_TRACE))
        for trial in range(200):
            frees = [rng.randrange(0, 6_000) for _ in BOUNDARY_TRACE]
            with self.subTest(trial=trial):
                events = boundary_events(self.overlay, self.NONE, credit(FakeAllocator(*frees)))
                self.assertEqual(only_mark_steps(events), expected)

    @needs_hook
    def test_no_drain_with_room_matches_drains_off(self) -> None:
        events = boundary_events(self.overlay, self.NONE, credit(FakeAllocator(5_000)))
        self.assertEqual(events, boundary_events(self.base, self.NONE))
        self.assertNotIn(("synchronize",), events)

    @needs_hook
    def test_near_the_limit_drains_after_the_mark_step(self) -> None:
        events = boundary_events(self.overlay, self.NONE, credit(FakeAllocator(100)),
                                 trace=(("moe.tile", False),))
        self.assertEqual(events, [("mark_step", (), (("sync", True),)), ("synchronize",)])

    @needs_hook
    def test_read_error_drains(self) -> None:
        with self.assertLogs("vllm.r589", "WARNING"):
            events = boundary_events(self.overlay, self.NONE, credit(FakeAllocator(RuntimeError("gone"))),
                                     trace=(("moe.tile", False),))
        self.assertIn(("synchronize",), events)

    @needs_hook
    def test_a_site_that_already_drains_does_not_read(self) -> None:
        allocator = FakeAllocator(5_000)
        boundary_events(self.overlay, self.all_sites, credit(allocator), trace=(("moe.owned", True),))
        self.assertEqual(allocator.reads, 0)

    @needs_hook
    def test_a_mixed_capture_neither_reads_nor_steps(self) -> None:
        allocator = FakeAllocator(100)
        with capture_context():
            events = boundary_events(self.overlay, self.NONE, credit(allocator))
        self.assertEqual((events, allocator.reads), ([], 0))

    @needs_hook
    def test_a_host_tensor_neither_reads_nor_steps(self) -> None:
        allocator = FakeAllocator(100)
        events: list[tuple] = []
        cpu_tensor = torch.zeros(1)
        with lazy_mode(events), mock.patch.object(self.overlay, "RUNAHEAD", credit(allocator)):
            self.overlay._boundary("moe.tile", cpu_tensor, drain=False)
        self.assertEqual((events, allocator.reads), ([], 0))

    @needs_hook
    def test_bad_margin_fails_the_import_so_every_worker_refuses_boot(self) -> None:
        environ = {"HPU_GLM53_RUNAHEAD_CREDIT": "1", "HPU_GLM53_RUNAHEAD_MARGIN_MIB": "100"}
        with self.assertRaisesRegex(ValueError, "HPU_GLM53_RUNAHEAD_MARGIN_MIB=100"):
            r589_with_env("r589_bad_margin", environ)

    def test_the_shipped_r589_carries_no_credit_hook(self) -> None:
        """Plan step 22 is held out of the image: R589 neither imports the credit module nor reads its flag."""
        source = (OVERLAY_ROOT / R589_FILE).read_text(encoding="utf-8").lower()
        self.assertNotIn("runahead", source)
        self.assertFalse(hasattr(self.overlay, "RUNAHEAD"))


class Stats(unittest.TestCase):
    def test_periodic_line_reports_the_counters_a_card_cell_scores(self) -> None:
        module = load_overlay(CREDIT_FILE)
        subject = module.RunaheadCredit(FakeAllocator(5_000), 512 * MIB, 1)
        with self.assertLogs("vllm.r589", "WARNING") as logs:
            for _ in range(module.STATS_EVERY):
                subject.should_drain("moe.tile")
        line = logs.output[-1]
        self.assertIn("GLM53_RUNAHEAD event=stats", line)
        for field in (f"boundaries={module.STATS_EVERY}", f"reads={module.STATS_EVERY}", "drains=0", "read_errors=0",
                      "read_ns_total=", "sample_every=1", "margin_mib=512"):
            self.assertIn(field, line)

    def test_first_drain_logs_its_site(self) -> None:
        subject = credit(FakeAllocator(100))
        with self.assertLogs("vllm.r589", "WARNING") as logs:
            subject.should_drain("kda.tile")
        self.assertIn("GLM53_RUNAHEAD event=drain site=kda.tile", logs.output[0])
        self.assertIn("free_mib=100", logs.output[0])


if __name__ == "__main__":
    unittest.main()
