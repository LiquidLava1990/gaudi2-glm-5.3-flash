"""--fresh-cache on the C1 cell: a boot whose graphs or kernels no earlier boot compiled for (diag-emit/RUN.txt, the word-skip emit) starts from an empty
recipe leaf, so no seeded recipe can replay. lab.py up has the option; c1_cell.py run takes it and memory_cells.boot_steps passes it on. CPU only."""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "cells"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import c1_cell as cell  # noqa: E402
import memory_cells  # noqa: E402
from test_c1_cell import run_args  # noqa: E402

OPTION = "--fresh-cache"
FINAL_CELL = Path(__file__).resolve().parents[1] / "cells" / "c1_final_cell.sh"


class FreshCache(unittest.TestCase):
    def setUp(self) -> None:
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.directory = Path(directory.name)

    def boot_argvs(self, *extra: str) -> dict[str, list[str]]:
        args = cell.parse_args(run_args(self.directory, *extra))
        return {step.label: step.argv for step in memory_cells.boot_steps(cell.c1_boot(args), args)}

    def test_without_the_option_no_step_carries_it(self) -> None:
        for label, argv in self.boot_argvs().items():
            with self.subTest(step=label):
                self.assertNotIn(OPTION, argv)

    def test_with_the_option_the_up_step_carries_it_once(self) -> None:
        up = self.boot_argvs(OPTION)["up"]
        self.assertEqual(up.count(OPTION), 1)
        self.assertEqual(up[2], "up")

    def test_only_the_up_step_carries_it(self) -> None:
        for label, argv in self.boot_argvs(OPTION).items():
            with self.subTest(step=label):
                self.assertEqual(OPTION in argv, label == "up")

    def test_the_rest_of_the_up_step_is_unchanged(self) -> None:
        plain, fresh = self.boot_argvs()["up"], self.boot_argvs(OPTION)["up"]
        self.assertEqual([word for word in fresh if word != OPTION], plain)

    def test_the_dry_run_plan_shows_it_and_still_touches_no_card(self) -> None:
        with tempfile.TemporaryDirectory() as tmp, mock.patch("subprocess.run", side_effect=AssertionError("touched")):
            out = StringIO()
            with redirect_stdout(out):
                code = cell.main(run_args(Path(tmp), OPTION))
        plan = json.loads(out.getvalue())
        self.assertEqual((code, plan["touches_cards"]), (0, False))
        self.assertIn(OPTION, plan["steps"][0]["argv"])

    def test_a_cell_that_has_no_such_argument_boots_as_before(self) -> None:
        args = SimpleNamespace(quad="A", image="leaf", port=18440, mount=[])
        up = memory_cells.boot_steps(memory_cells.Boot("claude-flash-launch-x", {}, []), args)[0].argv
        self.assertNotIn(OPTION, up)

    def test_the_final_cell_script_passes_c1_mounts_unquoted_so_the_option_rides_in_it(self) -> None:
        script = FINAL_CELL.read_text(encoding="utf-8")
        self.assertIn("${C1_MOUNTS:-}", script)
        self.assertNotIn('"${C1_MOUNTS:-}"', script)


if __name__ == "__main__":
    unittest.main()
