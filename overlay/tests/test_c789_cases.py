"""dual-4m's C789 CPU checks on the overlay: the native-pin cases and the 11 real 1792-row prompt metadata cases, each against the
overlay's files. The 16 shared-owner checks are test_shared_gather_owner.py."""
from __future__ import annotations

import ast
import importlib.util
import inspect
import os
import sys
import unittest
from collections.abc import Callable
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest import mock

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from overlay_testlib import BASE_ROOT, OVERLAY_ROOT  # noqa: E402
import pytest_shim  # noqa: E402

RUNNER_FILE = "vllm_gaudi/v1/worker/hpu_model_runner.py"
PIN_CASES = 31  # C789 reported 30; dual-4m's file holds one case more than it did then
DESELECTED = ("test_only_the_listed_definitions_change",)  # compares that lane's own two trees; deselected in C789 too
RUNNER_METHODS = {"_glm53_sparse_block_bucket", "_make_glm53_sparse_metadata_cpu"}
DECODE_TIER = "HPU_GLM53_SPARSE_DECODE_BLOCK_TIER_640"
PREFILL_TIER = "HPU_GLM53_SPARSE_PREFILL_BLOCK_TIER_640"
METADATA_CASES = (
    (512, 1792, 1, True, 128, 4, 512), (513, 1792, 1, True, 128, 4, 640), (565, 1792, 1, True, 128, 4, 640),
    (579, 1792, 1, True, 128, 4, 640), (640, 1792, 1, True, 128, 4, 640), (641, 1792, 1, True, 128, 4, 1024),
    (580, 1792, 1, False, 128, 4, 1024), (580, 1792, 2, True, 128, 4, 1024), (580, 1791, 1, True, 128, 4, 1024),
    (580, 1792, 1, True, 256, 4, 1024), (580, 1792, 1, True, 128, 8, 1024))


def load_cases() -> ModuleType:
    spec = importlib.util.spec_from_file_location("pin_cases_c789", Path(__file__).resolve().parent / "pin_cases_c789.py")
    module = importlib.util.module_from_spec(spec)
    with mock.patch.dict(sys.modules, {"pytest": pytest_shim, "pin_cases_c789": module}):
        spec.loader.exec_module(module)
    return module


def bind(function: Callable, cases: ModuleType, patch: pytest_shim.Patcher, log: pytest_shim.CapLog,
         given: dict[str, object]) -> dict[str, object]:
    """The fixtures a case names, built in order."""
    known = {"monkeypatch": patch, "caplog": log}
    values = dict(given)
    for name in inspect.signature(function).parameters:
        if name not in values:
            fixture = getattr(cases, name, None)
            values[name] = known[name] if name in known else fixture(**bind(fixture, cases, patch, log, {}))
    return values


def run_case(cases: ModuleType, function: Callable, given: dict[str, object]) -> None:
    patch, log = pytest_shim.Patcher(), pytest_shim.CapLog()
    try:
        function(**bind(function, cases, patch, log, given))
    finally:
        log.close_all()
        patch.undo()


def expanded_cases(cases: ModuleType) -> list[tuple[str, Callable, dict[str, object]]]:
    found = []
    for name, function in vars(cases).items():
        if not name.startswith("test_") or name in DESELECTED or not callable(function):
            continue
        names, values = getattr(function, "parameters", ([], [()]))
        for index, value in enumerate(values):
            row = value if len(names) > 1 else (value,)
            found.append((f"{name}[{index}]", function, dict(zip(names, row))))
    return found


class NativePinCases(unittest.TestCase):
    def test_all_the_native_pin_cases(self) -> None:
        cases = load_cases()
        expanded = expanded_cases(cases)
        self.assertEqual(len(expanded), PIN_CASES, [name for name, _, _ in expanded])
        for name, function, given in expanded:
            with self.subTest(name):
                run_case(cases, function, given)


def runner_class(path: Path) -> type:
    source = ast.parse(path.read_text(encoding="utf-8"))
    methods = [item for cls in source.body if isinstance(cls, ast.ClassDef) for item in cls.body
               if isinstance(item, ast.FunctionDef) and item.name in RUNNER_METHODS]
    cls = ast.ClassDef(name="Runner", bases=[], keywords=[], body=methods, decorator_list=[])
    namespace = {"os": os, "torch": torch, "logger": SimpleNamespace(info_once=lambda *_: None)}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[cls], type_ignores=[])), str(path), "exec"), namespace)
    return namespace["Runner"]


class Table:
    def __init__(self, data: torch.Tensor, block_size: int) -> None:
        self.data, self.block_size = data, block_size

    def get_cpu_tensor(self) -> torch.Tensor:
        return self.data


def configured(cls: type, source: int, pool: int) -> object:
    runner = cls()
    runner._glm53_index_group_id, runner._glm53_tail_group_id = 0, 1
    runner.warmup_mode, runner.dtype = False, torch.bfloat16
    runner._resolve_block = lambda block: block + 9
    runner.input_batch = SimpleNamespace(
        block_table=[Table(torch.arange(4 * 2048).reshape(4, 2048), source), Table(torch.arange(4).reshape(4, 1), pool)],
        req_id_to_index={str(i): i for i in range(4)})
    return runner


class PromptMetadataCases(unittest.TestCase):
    def test_eleven_real_1792_row_cases(self) -> None:
        base, overlay = runner_class(BASE_ROOT / RUNNER_FILE), runner_class(OVERLAY_ROOT / RUNNER_FILE)
        for blocks, rows, batch, on, source, pool, expected in METADATA_CASES:
            with self.subTest(blocks=blocks, rows=rows, batch=batch, on=on, source=source, pool=pool), \
                    mock.patch.dict(os.environ, {PREFILL_TIER: str(int(on)), DECODE_TIER: "0"}):
                end = blocks * source
                positions = [list(range(end - rows, end))] + [[0] * rows] * (batch - 1)
                inputs = (["0"], [rows], positions, batch, rows)
                old = configured(base, source, pool)._make_glm53_sparse_metadata_cpu(*inputs)
                new = configured(overlay, source, pool)._make_glm53_sparse_metadata_cpu(*inputs)
                self.assertEqual(new[2].shape, (batch, expected))
                self.assertEqual((old[4], new[4]), (end, end))
                for index in (0, 1, 3):
                    self.assertTrue(torch.equal(old[index], new[index]))
                self.assertTrue(torch.equal(old[2][:, :expected], new[2]))
                self.assertTrue(torch.all(old[2][:, expected:] == -1))


if __name__ == "__main__":
    unittest.main()
