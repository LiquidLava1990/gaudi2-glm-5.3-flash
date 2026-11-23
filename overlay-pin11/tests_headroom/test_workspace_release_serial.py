"""Crash plan 2026-11-22: the first key's workspace release over-asks the card on all four ranks at the same moment, and the host died with a
PCIe machine check in that window. VLLM_GAUDI_MIXED_KEY_WORKSPACE_RELEASE_MODE=serial runs the same over-ask one rank at a time in rank order,
every rank meeting in one gather per turn; unset keeps the all-ranks-at-once path call for call. HPU_GLM53_CAPTURE_BALLAST_MIB holds a ballast
through the first prompt key's capture so its workspace cannot grow into that slack. CPU only: what the card hands back is a card question."""
from __future__ import annotations

import os
import sys
import tempfile
import threading
import time
import unittest
import weakref
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent))

from test_key_workspace_release import FLAG, FREE_MIB, LIMIT_MIB, MIB, ReleaseCase  # noqa: E402
from test_prompt_keys import BLOCK, Backend, RowForward  # noqa: E402

MODE_FLAG = "VLLM_GAUDI_MIXED_KEY_WORKSPACE_RELEASE_MODE"
BALLAST_FLAG = "HPU_GLM53_CAPTURE_BALLAST_MIB"
CRUMB_FLAG = "HPU_GLM53_BREADCRUMB_FILE"
BUCKET = (0, 1536, 4096)
HELD_MIB = 5052
RANKS = 4


def crumb_path(case: unittest.TestCase) -> Path:
    """A breadcrumb file in a directory the case removes when it ends."""
    directory = tempfile.TemporaryDirectory()
    case.addCleanup(directory.cleanup)
    return Path(directory.name) / "crumbs.log"


def crumb_steps(path: Path) -> list[str]:
    return [line.split(" step=")[1].split(" ")[0] for line in path.read_text().splitlines()]


class FakeWorld:
    """Four ranks in threads: gather is an all_gather_object (every rank blocks until all four report), the over-ask counts who is inside it."""

    def __init__(self, fail_rank: int | None = None) -> None:
        self.barrier = threading.Barrier(RANKS, timeout=10)
        self.slots: list[object] = [None] * RANKS
        self.local = threading.local()
        self.lock = threading.Lock()
        self.inside = 0
        self.most_inside = 0
        self.order: list[int] = []
        self.gathers = [0] * RANKS
        self.fail_rank = fail_rank

    def position(self) -> tuple[int, int]:
        return self.local.rank, RANKS

    def gather(self, report: dict) -> list[dict]:
        rank = self.local.rank
        self.gathers[rank] += 1
        self.slots[rank] = report
        self.barrier.wait()
        reports = list(self.slots)
        self.barrier.wait()
        return reports

    def hold(self, nbytes: int, while_held: object) -> int:
        with self.lock:
            self.inside += 1
            self.most_inside = max(self.most_inside, self.inside)
            self.order.append(self.local.rank)
        time.sleep(0.02)  # long enough for an unserialized peer to enter too
        with self.lock:
            self.inside -= 1
        if self.local.rank == self.fail_rank:
            raise RuntimeError("PT_DEVMEM")
        return HELD_MIB * MIB


class SerialRelease(ReleaseCase):
    def run_ranks(self, world: FakeWorld) -> list[BaseException | None]:
        outcomes: list[BaseException | None] = [None] * RANKS

        def rank_main(rank: int) -> None:
            world.local.rank = rank
            cache = self.graph.GraphCache(RowForward(), BLOCK, Backend())
            try:
                cache.release_workspace_once(BUCKET)
            except BaseException as exc:  # the test reads every rank's outcome
                outcomes[rank] = exc

        with mock.patch.dict(os.environ, {FLAG: "1", MODE_FLAG: "serial"}), \
                mock.patch.object(self.graph, "_world_position", world.position), \
                mock.patch.object(self.graph, "rank_reports", world.gather), \
                mock.patch.object(self.graph, "_hold_ballast", world.hold), \
                mock.patch.object(self.graph, "_free_hbm", return_value=FREE_MIB * MIB), \
                mock.patch.object(self.graph.graphs_only, "arm_exit") as self.arm:
            threads = [threading.Thread(target=rank_main, args=(rank,)) for rank in range(RANKS)]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join(30)
        self.assertFalse(any(thread.is_alive() for thread in threads), "a rank hung in the gather")
        return outcomes

    def test_one_rank_at_a_time_in_rank_order_and_every_rank_meets_every_turn(self) -> None:
        world = FakeWorld()
        outcomes = self.run_ranks(world)
        self.assertEqual(outcomes, [None] * RANKS)
        self.assertEqual(world.most_inside, 1, "two ranks were inside the over-ask at once")
        self.assertEqual(world.order, list(range(RANKS)))
        self.assertEqual(world.gathers, [RANKS] * RANKS)

    def test_a_failed_turn_stops_every_rank_together_naming_the_rank(self) -> None:
        world = FakeWorld(fail_rank=2)
        outcomes = self.run_ranks(world)
        for rank, outcome in enumerate(outcomes):
            with self.subTest(rank=rank):
                self.assertIsInstance(outcome, RuntimeError)
                for needle in ("ALL ranks", f"rank 2: RuntimeError: {FLAG}", "PT_DEVMEM"):
                    self.assertIn(needle, str(outcome))
        self.assertEqual(world.gathers, [3] * RANKS, "every rank leaves at the failed turn, none waits for a later one")
        self.arm.assert_called_once_with("GLM53_WORKSPACE_RELEASE_FAILED")
        self.assertEqual(world.order, [0, 1, 2])

    def test_each_rank_logs_what_it_released(self) -> None:
        world = FakeWorld()
        with self.assertLogs(self.graph.LOG, "WARNING") as logs:
            self.run_ranks(world)
        after = [line for line in logs.output if "GLM53_AFTER_WORKSPACE_RELEASE" in line]
        self.assertEqual(len(after), RANKS)
        released = HELD_MIB + FREE_MIB + self.graph.RELEASE_OVERSHOOT_MIB - FREE_MIB
        for rank in range(RANKS):
            line = next(line for line in after if f" rank={rank} " in line)
            for needle in ("mode=serial", f"free_MiB_before={FREE_MIB}", f"held_MiB={HELD_MIB}", f"released_MiB={released}"):
                self.assertIn(needle, line)

    def test_a_mode_it_does_not_know_refuses_naming_the_setting(self) -> None:
        cache = self.graph.GraphCache(RowForward(), BLOCK, Backend())
        with mock.patch.dict(os.environ, {FLAG: "1", MODE_FLAG: "staggered"}), mock.patch.object(self.graph, "_hold_ballast") as hold, \
                self.assertRaisesRegex(ValueError, f"{MODE_FLAG}='staggered'.*serial"):
            cache.release_workspace_once(BUCKET)
        hold.assert_not_called()


class SettingsAgreement(ReleaseCase):
    """A key set on some ranks only: the first prompt key's existing admission gather carries both settings, so every rank refuses there,
    before any rank takes a ballast or a release turn the others would not meet."""

    def admit_ranks(self, settings_of: dict[int, dict[str, str]]) -> tuple[FakeWorld, list[BaseException | None]]:
        world = FakeWorld()
        outcomes: list[BaseException | None] = [None] * RANKS
        fake = mock.MagicMock()
        fake.hpu.mem_get_info.return_value = (FREE_MIB * MIB, LIMIT_MIB * MIB)

        def rank_main(rank: int) -> None:
            world.local.rank = rank
            try:
                with self.graph.boot_capture_context():
                    self.graph.HpuBackend().prepare_capture(BUCKET, 0, "")
            except BaseException as exc:  # the test reads every rank's outcome
                outcomes[rank] = exc

        with mock.patch.dict(sys.modules, {"habana_frameworks": mock.MagicMock(torch=fake), "habana_frameworks.torch": fake}), \
                mock.patch.object(self.graph, "rank_reports", world.gather), \
                mock.patch.object(self.graph, "boot_key_settings", lambda: settings_of[world.local.rank]):
            threads = [threading.Thread(target=rank_main, args=(rank,)) for rank in range(RANKS)]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join(30)
        self.assertFalse(any(thread.is_alive() for thread in threads), "a rank hung in the gather")
        return world, outcomes

    def assert_all_refuse(self, settings_of: dict[int, dict[str, str]], value: str) -> None:
        world, outcomes = self.admit_ranks(settings_of)
        self.assertEqual(world.gathers, [1] * RANKS, "the refusal rides the existing admission gather, no extra one")
        for rank, outcome in enumerate(outcomes):
            with self.subTest(rank=rank):
                self.assertIsInstance(outcome, RuntimeError)
                for needle in ("ALL ranks", MODE_FLAG, BALLAST_FLAG, "rank 2", value, "set them container-wide in the levers file"):
                    self.assertIn(needle, str(outcome))

    def test_a_ballast_on_one_rank_refuses_on_all(self) -> None:
        self.assert_all_refuse({rank: {MODE_FLAG: "", BALLAST_FLAG: "2048" if rank == 2 else ""} for rank in range(RANKS)}, "'2048'")

    def test_serial_mode_on_one_rank_refuses_on_all(self) -> None:
        self.assert_all_refuse({rank: {MODE_FLAG: "serial" if rank == 2 else "", BALLAST_FLAG: ""} for rank in range(RANKS)}, "'serial'")

    def test_agreeing_ranks_are_admitted(self) -> None:
        world, outcomes = self.admit_ranks({rank: {MODE_FLAG: "serial", BALLAST_FLAG: "2048"} for rank in range(RANKS)})
        self.assertEqual(outcomes, [None] * RANKS)
        self.assertEqual(world.gathers, [1] * RANKS)

    def test_the_settings_are_the_raw_env_values(self) -> None:
        with mock.patch.dict(os.environ, {MODE_FLAG: "serial", BALLAST_FLAG: " 2048 "}):
            self.assertEqual(self.graph.boot_key_settings(), {MODE_FLAG: "serial", BALLAST_FLAG: "2048"})

    def test_a_bad_value_fails_warmup_validation_before_any_capture(self) -> None:
        for flag, value in ((MODE_FLAG, "staggered"), (BALLAST_FLAG, "2g")):
            with self.subTest(flag=flag), mock.patch.dict(os.environ, {flag: value}), \
                    mock.patch.object(self.graph, "validate_prompt_keys", return_value=()), self.assertRaisesRegex(ValueError, flag):
                self.graph.validate_warmup(mock.Mock())


class UnsetModeIsTheOldPath(ReleaseCase):
    """Mode unset: the same reads, the same fill, the same two lines, no gather and no breadcrumb file touched."""

    def test_call_sequence_and_lines_are_the_old_ones(self) -> None:
        calls: list[tuple] = []
        free = mock.Mock(side_effect=lambda: calls.append(("free",)) or FREE_MIB * MIB)
        hold = mock.Mock(side_effect=lambda nbytes, fn: calls.append(("hold", nbytes, fn())) or HELD_MIB * MIB)
        gather = mock.Mock(side_effect=AssertionError("the old path never gathers"))
        cache = self.graph.GraphCache(RowForward(), BLOCK, Backend())
        with mock.patch.dict(os.environ, {FLAG: "1"}), \
                mock.patch.object(self.graph, "_free_hbm", free), mock.patch.object(self.graph, "_hold_ballast", hold), \
                mock.patch.object(self.graph, "rank_reports", gather), mock.patch("os.open", side_effect=AssertionError("breadcrumb wrote")), \
                self.assertLogs(self.graph.LOG, "WARNING") as logs:
            os.environ.pop(MODE_FLAG, None)
            os.environ.pop(CRUMB_FLAG, None)
            cache.release_workspace_once(BUCKET)
        fill = (FREE_MIB + self.graph.RELEASE_OVERSHOOT_MIB) * MIB
        self.assertEqual(calls, [("free",), ("hold", fill, None)])
        released = (HELD_MIB * MIB + fill - FREE_MIB * MIB) // MIB
        self.assertEqual([line.split(":", 2)[2] for line in logs.output], [
            f"GLM53_BEFORE_WORKSPACE_RELEASE bucket={BUCKET} free_MiB={FREE_MIB} limit_MiB={LIMIT_MIB} in_use_MiB={LIMIT_MIB - FREE_MIB} "
            f"overshoot_MiB=64 flag={FLAG}; an abort here means nothing was left to hand back: unset the flag",
            f"GLM53_KEY_WORKSPACE_RELEASE bucket={BUCKET} free_MiB before={FREE_MIB} held={HELD_MIB} released_while_held={released}",
        ])

    def test_the_old_path_drops_breadcrumbs_around_the_over_ask_when_asked(self) -> None:
        path = crumb_path(self)
        with mock.patch.dict(os.environ, {FLAG: "1", CRUMB_FLAG: str(path)}), \
                mock.patch.object(self.graph, "_free_hbm", return_value=FREE_MIB * MIB), \
                mock.patch.object(self.graph, "_hold_ballast", return_value=HELD_MIB * MIB):
            self.graph.release_capture_workspace(BUCKET)
        self.assertEqual(crumb_steps(path), ["workspace_over_ask", "workspace_over_ask_returned"])
        self.assertIn(f"fill_MiB={FREE_MIB + 64}", path.read_text())


class HoldBallastOrder(ReleaseCase):
    """_hold_ballast's device calls in their old order: allocate, fence, read back, read free, run, release."""

    def test_device_call_order_is_unchanged(self) -> None:
        events: list[str] = []
        fake = mock.MagicMock()
        fake.core.mark_step.side_effect = lambda: events.append("mark_step")
        fake.hpu.synchronize.side_effect = lambda: events.append("synchronize")
        fake.hpu.mem_get_info.side_effect = lambda: events.append("mem_get_info") or (7 * MIB, 0)
        real_ones = self.graph.torch.ones

        def ones(size: int, dtype: object, device: str) -> object:
            events.append(f"ones:{size // MIB}")
            return real_ones(4, dtype=dtype)

        package = mock.MagicMock(torch=fake)
        with mock.patch.dict(sys.modules, {"habana_frameworks": package, "habana_frameworks.torch": fake}), \
                mock.patch.object(self.graph.torch, "ones", ones):
            free = self.graph._hold_ballast(1030 * MIB, lambda: events.append("held"))
        self.assertEqual(free, 7 * MIB)
        self.assertEqual(events, ["ones:1024", "ones:6", "mark_step", "synchronize", "mem_get_info", "held"])


class Sentinel:
    """A stand-in ballast tensor the test can watch die."""


class CaptureBallast(ReleaseCase):
    def capture_two_keys(self, backend: Backend, refs: list[list[weakref.ref]]) -> tuple[mock.Mock, list[str]]:
        """Both keys with a 2048 MiB ballast: refs gets one list of ballast references per allocation."""
        def materialize(nbytes: int) -> list[Sentinel]:
            chunks = [Sentinel() for _ in range(nbytes // (1024 * MIB))]
            refs.append([weakref.ref(chunk) for chunk in chunks])
            return chunks

        take = mock.Mock(side_effect=materialize)
        cache = self.graph.GraphCache(RowForward(), BLOCK, backend)
        with mock.patch.dict(os.environ, {BALLAST_FLAG: "2048"}), mock.patch.object(self.graph, "_materialize_ballast", take),                 mock.patch.object(self.graph, "_allocator_headroom", return_value=8192 * MIB),                 mock.patch.object(self.graph, "_free_hbm", side_effect=[1000 * MIB, 3048 * MIB]),                 self.assertLogs(self.graph.LOG, "WARNING") as logs:
            self.boot_capture(cache, self.prompt_batch(1024, 0, 8192))
            self.boot_capture(cache, self.prompt_batch(1024, 0, 1024))
        return take, logs.output

    def test_unset_holds_nothing(self) -> None:
        cache = self.graph.GraphCache(RowForward(), BLOCK, Backend())
        with mock.patch.object(self.graph, "_materialize_ballast") as take, mock.patch.object(self.graph, "_allocator_headroom") as headroom:
            os.environ.pop(BALLAST_FLAG, None)
            self.boot_capture(cache, self.prompt_batch(1024, 0, 8192))
        take.assert_not_called()
        headroom.assert_not_called()

    def test_the_first_key_captures_under_the_ballast_and_it_is_freed_after(self) -> None:
        refs: list[list[weakref.ref]] = []
        alive_during: list[bool] = []

        class Watching(Backend):
            def capture(self, call: object) -> tuple:
                alive_during.append(bool(refs) and all(ref() is not None for ref in refs[-1]))
                return super().capture(call)

        take, output = self.capture_two_keys(Watching(), refs)
        take.assert_called_once_with(2048 * MIB)
        self.assertEqual(alive_during, [True, False], "the first key captures with the ballast held, the second without")
        self.assertEqual(len(refs), 1, "only the first prompt key takes a ballast")
        self.assertTrue(all(ref() is None for ref in refs[0]), "the ballast outlived the capture")
        line = next(line for line in output if "GLM53_CAPTURE_BALLAST" in line)
        for needle in ("bucket=(0, 1024, 8192)", "held_MiB=2048", "freed_MiB=2048"):
            self.assertIn(needle, line)

    def test_a_failed_capture_drops_the_ballast_without_touching_the_device(self) -> None:
        """The device may be wedged after a failed capture: a fence there could hang, so the ballast is only dereferenced."""
        refs: list[weakref.ref] = []
        died = RuntimeError("capture died")

        class Failing(Backend):
            def capture(self, call: object) -> tuple:
                raise died

        def materialize(nbytes: int) -> list[Sentinel]:
            chunks = [Sentinel(), Sentinel()]
            refs.extend(weakref.ref(chunk) for chunk in chunks)
            return chunks

        cache = self.graph.GraphCache(RowForward(), BLOCK, Failing())
        path = crumb_path(self)
        with mock.patch.dict(os.environ, {BALLAST_FLAG: "2048", CRUMB_FLAG: str(path)}), \
                mock.patch.object(self.graph, "_materialize_ballast", side_effect=materialize), \
                mock.patch.object(self.graph, "_allocator_headroom", return_value=8192 * MIB), \
                mock.patch.object(self.graph, "_free_hbm", side_effect=AssertionError("fenced after a failed capture")) as fence, \
                self.assertLogs(self.graph.LOG, "WARNING") as logs, self.assertRaises(RuntimeError) as raised:
            self.boot_capture(cache, self.prompt_batch(1024, 0, 8192))
        self.assertIs(raised.exception, died)
        fence.assert_not_called()
        self.assertEqual(len(refs), 2)
        self.assertTrue(all(ref() is None for ref in refs))
        line = next(line for line in logs.output if "GLM53_CAPTURE_BALLAST" in line)
        self.assertIn("freed_MiB=skipped reason=capture_failed", line)
        self.assertIn(" step=capture_ballast_freed bucket=(0,1024,8192) held_MiB=2048 freed_MiB=skipped reason=capture_failed",
                      path.read_text())

    def test_too_little_headroom_refuses_on_all_ranks_without_allocating(self) -> None:
        cache = self.graph.GraphCache(RowForward(), BLOCK, Backend())
        with mock.patch.dict(os.environ, {BALLAST_FLAG: "2048"}), mock.patch.object(self.graph, "_materialize_ballast") as take, \
                mock.patch.object(self.graph, "_allocator_headroom", return_value=2300 * MIB), \
                self.assertRaisesRegex(RuntimeError, f"ALL ranks.*{BALLAST_FLAG}=2048.*2300"):
            self.boot_capture(cache, self.prompt_batch(1024, 0, 8192))
        take.assert_not_called()

    def test_a_bad_value_refuses_naming_the_setting(self) -> None:
        cache = self.graph.GraphCache(RowForward(), BLOCK, Backend())
        with mock.patch.dict(os.environ, {BALLAST_FLAG: "2g"}), self.assertRaisesRegex(ValueError, f"{BALLAST_FLAG}='2g'"):
            self.boot_capture(cache, self.prompt_batch(1024, 0, 8192))


class CaptureBreadcrumbs(ReleaseCase):
    def test_each_prompt_key_leaves_admit_admitted_and_captured(self) -> None:
        path = crumb_path(self)
        cache = self.graph.GraphCache(RowForward(), BLOCK, Backend())
        with mock.patch.dict(os.environ, {CRUMB_FLAG: str(path)}):
            self.boot_capture(cache, self.prompt_batch(1024, 0, 8192))
        self.assertEqual(crumb_steps(path), ["capture_admit", "capture_admitted", "captured"])
        self.assertTrue(all("bucket=(0,1024,8192)" in line for line in path.read_text().splitlines()))

    def test_a_ballasted_key_leaves_held_and_freed_around_its_capture(self) -> None:
        path = crumb_path(self)
        cache = self.graph.GraphCache(RowForward(), BLOCK, Backend())
        with mock.patch.dict(os.environ, {CRUMB_FLAG: str(path), BALLAST_FLAG: "1024"}), \
                mock.patch.object(self.graph, "_materialize_ballast", return_value=[Sentinel()]), \
                mock.patch.object(self.graph, "_allocator_headroom", return_value=8192 * MIB), \
                mock.patch.object(self.graph, "_free_hbm", side_effect=[1000 * MIB, 2024 * MIB]):
            self.boot_capture(cache, self.prompt_batch(1024, 0, 8192))
        self.assertEqual(crumb_steps(path), ["capture_admit", "capture_admitted", "capture_ballast_take", "capture_ballast_held", "captured",
                                             "capture_ballast_freed"])

    def test_warmup_end_leaves_a_breadcrumb(self) -> None:
        path = crumb_path(self)
        graph = self.graph
        long_history = mock.Mock(warm_adaptive_tiles_at_boot=mock.Mock(return_value=0))
        self.addCleanup(graph._RANK_CHECK_BOOT_DONE.__setitem__, 0, graph._RANK_CHECK_BOOT_DONE[0])
        with mock.patch.dict(os.environ, {CRUMB_FLAG: str(path)}), mock.patch.dict(sys.modules, {"glm53_r590_long_history": long_history}), \
                mock.patch.object(graph, "validate_warmup", return_value=(), create=True), mock.patch.object(graph, "prepare_rank_check"), \
                mock.patch.object(graph, "warmup_scenarios", return_value=[]), mock.patch.object(graph, "check_keys_captured"), \
                mock.patch.object(graph, "prompt_keys", return_value=()), mock.patch.object(graph, "count_off_graph", create=True), \
                mock.patch.object(graph, "mark_boot_window", create=True):
            graph.warmup(mock.Mock())
        self.assertEqual(crumb_steps(path), ["mixed_warmup_done"])


if __name__ == "__main__":
    unittest.main()
