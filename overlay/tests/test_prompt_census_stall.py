"""K1 stall trace: per-rank mark and collective lines of a key capture, stack dumps while it runs, nothing with the flag off."""

from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent))

from overlay_testlib import load_overlay  # noqa: E402
from test_prompt_census import FLAG, FakeRuntime  # noqa: E402

CENSUS_FILE = "glm53_prompt_census.py"
STALL_DIR = "HPU_GLM53_K1_STALL_DIR"


class StallTraceCase(unittest.TestCase):
    def setUp(self) -> None:
        self.module = load_overlay(CENSUS_FILE)
        self.runtime = FakeRuntime()
        patcher = mock.patch.object(self.module, "_runtime", return_value=(self.runtime.core, self.runtime.hpu))
        patcher.start()
        self.addCleanup(patcher.stop)
        self.module.GRAPH_LOG.clear()

    def body(self) -> str:
        self.runtime.core.mark_step(sync=False)
        self.module._dist().all_reduce("tensor")
        return "hidden"

    def capture(self, directory: str | None, key_replay: bool | None = False) -> list[str]:
        env = {FLAG: "1", **({STALL_DIR: directory} if directory else {})}
        count = iter((0, 1, 1))
        fake_dist = mock.Mock(spec=["all_reduce", "get_rank", "is_initialized"])
        fake_dist.get_rank.return_value = 2
        fake_dist.is_initialized.return_value = True
        with mock.patch.dict(os.environ, env), mock.patch.object(self.module, "_dist", return_value=fake_dist), \
                mock.patch.object(self.module.faulthandler, "dump_traceback_later") as self.dump, \
                mock.patch.object(self.module.faulthandler, "cancel_dump_traceback_later") as self.cancel, \
                self.assertLogs("vllm.r589", "WARNING"):
            self.fake_all_reduce = fake_dist.all_reduce
            self.module.census_graph_forward(self.body, 1024, 1024, lambda: next(count), key_replay=key_replay)()
        if not directory:
            return []
        files = list(Path(directory).iterdir())
        self.assertEqual(len(files), 1)
        return files[0].read_text().splitlines()


class CaptureTrace(StallTraceCase):
    def test_writes_rank_marks_and_collectives_in_order(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            lines = self.capture(tmp)
        events = [line.split()[1] for line in lines if not line.startswith("#")]
        self.assertTrue(lines[0].startswith("# rank=2 "), lines[0])
        self.assertEqual(events, ["begin", "mark.enter", "mark.exit", "coll.enter", "coll.exit", "end"])

    def test_mark_line_names_site_and_sync(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            lines = self.capture(tmp)
        mark = next(line for line in lines if " mark.enter " in line)
        self.assertIn("sync=False", mark)
        self.assertIn("test_prompt_census_stall.py", mark)

    def test_stack_dumps_run_only_while_capturing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            self.capture(tmp)
        self.dump.assert_called_once()
        self.assertEqual(self.dump.call_args.kwargs["repeat"], True)
        self.cancel.assert_called_once()

    def test_collective_is_still_called_and_restored(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            self.capture(tmp)
        self.fake_all_reduce.assert_called_once_with("tensor")


class NoTrace(StallTraceCase):
    def test_flag_off_opens_no_file_and_arms_no_dump(self) -> None:
        self.capture(None)
        self.dump.assert_not_called()

    def test_a_replay_is_not_traced(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            env = {FLAG: "1", STALL_DIR: tmp}
            with mock.patch.dict(os.environ, env), self.assertLogs("vllm.r589", "WARNING"):
                self.module.census_graph_forward(lambda: "hidden", 1024, 1024, lambda: 1, key_replay=True)()
            self.assertEqual(list(Path(tmp).iterdir()), [])


if __name__ == "__main__":
    unittest.main()
