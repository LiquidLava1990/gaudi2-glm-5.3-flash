"""glm53_hbm_margin pool floor (plan gaudi-card-headroom-20261122 step 4): with HPU_GLM53_POOL_LIMIT_MAX_MIB set every rank reads the PT pool
Limit before the KV cache is allocated and refuses the boot when the pool ignored PT_HPU_POOL_MEM_ACQUIRE_PERC; unset, nothing is read."""
from __future__ import annotations

import ast
import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import glm53_hbm_margin as margin  # noqa: E402

MIB = 1 << 20
FLAG = "HPU_GLM53_POOL_LIMIT_MAX_MIB"
PERC = "PT_HPU_POOL_MEM_ACQUIRE_PERC"
WORKER = ROOT / "vllm_gaudi" / "v1" / "worker" / "hpu_worker.py"


def _stats(limit_mib: int, in_use_mib: int = 1_200) -> dict[str, int]:
    return {"Limit": limit_mib * MIB, "InUse": in_use_mib * MIB, "MaxInUse": in_use_mib * MIB}


class FakeRuntime:
    def __init__(self, stats: dict[str, int]) -> None:
        self.reads = 0
        self._stats = stats

    def stats(self) -> dict[str, int]:
        self.reads += 1
        return self._stats


@pytest.fixture
def runtime(monkeypatch: pytest.MonkeyPatch) -> FakeRuntime:
    fake = FakeRuntime(_stats(95_926))
    monkeypatch.setattr(margin, "_runtime", lambda: fake)
    monkeypatch.delenv(FLAG, raising=False)
    return fake


def test_unset_reads_nothing_and_prints_nothing(runtime: FakeRuntime, capsys: pytest.CaptureFixture) -> None:
    margin.enforce_pool_floor()
    assert runtime.reads == 0
    assert capsys.readouterr().out == ""


def test_a_capped_pool_passes_and_prints_one_floor_line(runtime: FakeRuntime, monkeypatch: pytest.MonkeyPatch,
                                                         capsys: pytest.CaptureFixture) -> None:
    monkeypatch.setenv(FLAG, "96400")
    margin.enforce_pool_floor()
    lines = capsys.readouterr().out.splitlines()
    assert lines == [f"GLM53_POOL_FLOOR limit_MiB=95926 max_MiB=96400 in_use_MiB=1200 pid={os.getpid()}"]


def test_a_pool_that_ignored_the_cap_refuses_naming_both_values_and_the_fix(runtime: FakeRuntime,
                                                                            monkeypatch: pytest.MonkeyPatch) -> None:
    runtime._stats = _stats(96_895)
    monkeypatch.setenv(FLAG, "96400")
    monkeypatch.setenv(PERC, "99")
    with pytest.raises(RuntimeError) as refused:
        margin.enforce_pool_floor()
    for needle in ("ALL ranks", "96895", "96400", f"{PERC}=99", FLAG, "hl-smi"):
        assert needle in str(refused.value)


@pytest.mark.parametrize("value", ["95k", "-1", "0", "9.5", "\u00b2\u00b3"])  # superscripts: isdigit() passes them, int() does not
def test_a_bad_limit_refuses_with_the_setting_named(runtime: FakeRuntime, monkeypatch: pytest.MonkeyPatch, value: str) -> None:
    monkeypatch.setenv(FLAG, value)
    with pytest.raises(RuntimeError, match=f"{FLAG}='{value}'.*positive integer"):
        margin.enforce_pool_floor()


def test_an_unreadable_pool_refuses_rather_than_skipping_the_check(monkeypatch: pytest.MonkeyPatch) -> None:
    def broken() -> FakeRuntime:
        raise ModuleNotFoundError("No module named 'habana_frameworks'")

    monkeypatch.setattr(margin, "_runtime", broken)
    monkeypatch.setenv(FLAG, "96400")
    with pytest.raises(RuntimeError, match="pool floor check could not run.*habana_frameworks"):
        margin.enforce_pool_floor()


def _method(name: str) -> ast.FunctionDef:
    tree = ast.parse(WORKER.read_text())
    worker = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "HPUWorker")
    return next(node for node in worker.body if isinstance(node, ast.FunctionDef) and node.name == name)


def _call_lines(function: ast.FunctionDef) -> dict[str, int]:
    """Each called name and the first source line that calls it."""
    lines: dict[str, int] = {}
    for node in ast.walk(function):
        if isinstance(node, ast.Call):
            name = ast.unparse(node.func)
            lines[name] = min(node.lineno, lines.get(name, node.lineno))
    return lines


def test_the_worker_checks_the_floor_before_it_allocates_the_kv_cache() -> None:
    lines = _call_lines(_method("initialize_from_config"))
    assert "glm53_hbm_margin.enforce_pool_floor" in lines
    first_kv_step = min(lines["ensure_kv_transfer_initialized"], lines["self.model_runner.initialize_kv_cache"])
    assert lines["glm53_hbm_margin.enforce_pool_floor"] < first_kv_step
