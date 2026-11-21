"""check_flags_off.py: four base-image ENV flags pass only at their base values, and anything else that is on fails the gate.

The gate's stage was permanently red on those four, so a real ON flag looked like the usual failure.
"""
from __future__ import annotations

import importlib.util
import os
import sys
import unittest
from pathlib import Path
from unittest import mock

LANE_DIR = Path(os.environ.get("LANE_DIR", Path(__file__).resolve().parents[2]))
SCRIPT = LANE_DIR / "check_flags_off.py"
BASE_VALUES = {"HPU_GLM53_MIXED_DECODE_REPLAY": "1", "HPU_GLM53_PRIVATE_PROMPT_TOPK": "1", "HPU_GRAPH_DRY_RUN": "0",
               "VLLM_PREFILL_MAX_REQUESTS_PER_STEP": "0"}
UNEXPECTED = "HPU_GLM53_NOT_IN_THE_BASE_ENV"


def load_gate() -> object:
    spec = importlib.util.spec_from_file_location("check_flags_off_under_test", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@unittest.skipUnless(SCRIPT.is_file(), "check_flags_off.py sits beside the overlay on the lab host")
class BaseEnvAllowlist(unittest.TestCase):
    def setUp(self) -> None:
        self.gate = load_gate()

    def exit_code(self, env: dict[str, str], probes: dict | None = None, flags: tuple[str, ...] = tuple(BASE_VALUES)) -> int:
        introduced = {name: ["file.py"] for name in flags}
        with mock.patch.object(self.gate, "new_flags", return_value=introduced), mock.patch.dict(self.gate.PROBES, probes or {}), \
                mock.patch.dict(os.environ, env, clear=True), mock.patch("builtins.print"):
            return self.gate.main()

    def test_the_four_are_the_base_images_values(self) -> None:
        self.assertEqual(self.gate.BASE_IMAGE_ENV, BASE_VALUES)

    def test_the_four_at_their_base_values_pass(self) -> None:
        self.assertEqual(self.exit_code(BASE_VALUES), 0)

    def test_each_of_the_four_at_another_value_fails(self) -> None:
        for name, value in BASE_VALUES.items():
            with self.subTest(name):
                changed = "0" if value == "1" else "1"
                self.assertEqual(self.exit_code({**BASE_VALUES, name: changed}), 1)

    def test_an_unexpected_flag_in_the_image_env_fails(self) -> None:
        self.assertEqual(self.exit_code({**BASE_VALUES, UNEXPECTED: "1"}, flags=(*BASE_VALUES, UNEXPECTED)), 1)

    def test_a_probe_that_reads_on_fails(self) -> None:
        self.assertEqual(self.exit_code(BASE_VALUES, {UNEXPECTED: lambda: False}, (*BASE_VALUES, UNEXPECTED)), 1)

    def test_a_probe_that_reads_off_passes(self) -> None:
        self.assertEqual(self.exit_code(BASE_VALUES, {UNEXPECTED: lambda: True}, (*BASE_VALUES, UNEXPECTED)), 0)

    def test_an_unprobed_flag_stays_listed_without_failing(self) -> None:
        self.assertEqual(self.exit_code(BASE_VALUES, flags=(*BASE_VALUES, UNEXPECTED)), 0)

    def test_an_allowlisted_verdict_is_its_own_word(self) -> None:
        with mock.patch.dict(os.environ, BASE_VALUES, clear=True):
            self.assertEqual(self.gate.judge("HPU_GRAPH_DRY_RUN"), "base-image-env")
        with mock.patch.dict(os.environ, {"HPU_GRAPH_DRY_RUN": "1"}, clear=True):
            self.assertEqual(self.gate.judge("HPU_GRAPH_DRY_RUN"), "SET-IN-IMAGE-ENV")


if __name__ == "__main__":
    unittest.main()
