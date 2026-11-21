"""One workspace release after the first prompt key (k1-out/fastread-fast4r-20261107T071910Z): the capture of the widest key leaves about 5.1 GiB
of workspace the runtime hands back only when a launch does not fit. A fill just past free forces that once. A second fill finds nothing to hand
back and aborts the rank with an uncatchable PT_DEVMEM, so it must run once only. CPU only: the readings are patched, what the runtime releases
is a card question."""
from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent))

from test_prompt_keys import BLOCK, Backend, KeyCase, RowForward  # noqa: E402

MIB = 1024 * 1024
FLAG = "VLLM_GAUDI_MIXED_KEY_WORKSPACE_RELEASE"


class Release(KeyCase):
    def run_keys(self, enabled: bool) -> tuple[mock.Mock, list[str]]:
        cache = self.graph.GraphCache(RowForward(), BLOCK, Backend())
        hold = mock.Mock(return_value=5057 * MIB)
        environ = {FLAG: "1"} if enabled else {}
        with mock.patch.dict(os.environ, environ), mock.patch.object(self.graph, "_hold_ballast", hold), \
                mock.patch.object(self.graph, "_free_hbm", return_value=4815 * MIB), \
                self.assertLogs(self.graph.LOG, "WARNING") as logs:
            self.boot_capture(cache, self.prompt_batch(1024, 0, 8192))
            self.boot_capture(cache, self.prompt_batch(1024, 0, 1024))
        return hold, logs.output

    def test_off_by_default_no_fill(self) -> None:
        hold, output = self.run_keys(False)
        hold.assert_not_called()
        self.assertFalse([line for line in output if "GLM53_KEY_WORKSPACE_RELEASE" in line])

    def test_one_fill_just_past_free_after_the_first_key_only(self) -> None:
        hold, _ = self.run_keys(True)
        hold.assert_called_once()
        self.assertEqual(hold.call_args.args[0], (4815 + self.graph.RELEASE_OVERSHOOT_MIB) * MIB)

    def test_the_overshoot_stays_small_and_the_fill_runs_nothing_under_it(self) -> None:
        hold, _ = self.run_keys(True)
        self.assertLessEqual(self.graph.RELEASE_OVERSHOOT_MIB, 128, "fast4q key 2 released only 682 MiB; a big overshoot risks the abort")
        self.assertIsNone(hold.call_args.args[1]())

    def test_the_line_states_free_before_held_after_and_released(self) -> None:
        _, output = self.run_keys(True)
        line = next(line for line in output if "GLM53_KEY_WORKSPACE_RELEASE" in line)
        for needle in ("bucket=(0, 1024, 8192)", "before=4815", "held=5057", "released_while_held=5121"):
            self.assertIn(needle, line)

    def test_a_decode_capture_does_not_fill(self) -> None:
        cache = self.graph.GraphCache(RowForward(), BLOCK, Backend())
        hold = mock.Mock(return_value=0)
        with mock.patch.dict(os.environ, {FLAG: "1"}), mock.patch.object(self.graph, "_hold_ballast", hold):
            cache.release_workspace_once((16, 2048, 128))
        hold.assert_not_called()


class ReleaseFailure(KeyCase):
    def test_a_failed_fill_stops_every_rank_and_names_the_flag(self) -> None:
        with mock.patch.object(self.graph, "_free_hbm", return_value=4815 * MIB), \
                mock.patch.object(self.graph, "_hold_ballast", side_effect=RuntimeError("PT_DEVMEM")), \
                mock.patch.object(self.graph.graphs_only, "arm_exit") as arm, \
                self.assertRaisesRegex(RuntimeError, f"{FLAG}.*PT_DEVMEM"):
            self.graph.release_capture_workspace((0, 1024, 8192))
        arm.assert_called_once_with("GLM53_WORKSPACE_RELEASE_FAILED")

    def test_the_fill_is_logged_before_it_can_abort(self) -> None:
        with mock.patch.object(self.graph, "_free_hbm", return_value=4815 * MIB), \
                mock.patch.object(self.graph, "_hold_ballast", side_effect=RuntimeError("PT_DEVMEM")), \
                mock.patch.object(self.graph.graphs_only, "arm_exit"), self.assertLogs(self.graph.LOG, "WARNING") as logs, \
                self.assertRaises(RuntimeError):
            self.graph.release_capture_workspace((0, 1024, 8192))
        line = next(line for line in logs.output if "GLM53_BEFORE_WORKSPACE_RELEASE" in line)
        for needle in ("free_MiB=4815", FLAG):
            self.assertIn(needle, line)


if __name__ == "__main__":
    unittest.main()
