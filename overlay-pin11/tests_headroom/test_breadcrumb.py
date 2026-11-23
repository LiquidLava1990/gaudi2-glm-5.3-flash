"""glm53_hbm_margin.breadcrumb (crash plan 2026-11-22): a host crash loses the docker log tail still in page cache, so each boot step that can
take the host down appends one line to HPU_GLM53_BREADCRUMB_FILE that is on disk (O_DSYNC) when the call returns. Unset: nothing is opened.
A file that cannot be written warns once per process on stderr and never stops the caller."""
from __future__ import annotations

import os
import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import glm53_hbm_margin as margin  # noqa: E402

FLAG = "HPU_GLM53_BREADCRUMB_FILE"
LINE = re.compile(r"^\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d\.\d{6}\+00:00 pid=(\d+) rank=(\S+) step=(\S+)( \S+=\S+)*$")


@pytest.fixture(autouse=True)
def fresh_warning(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(margin, "_BREADCRUMB_WARNED", [])
    monkeypatch.delenv(FLAG, raising=False)
    monkeypatch.delenv("RANK", raising=False)


def test_unset_opens_nothing(monkeypatch: pytest.MonkeyPatch) -> None:
    def refuse(*args: object) -> int:
        raise AssertionError("breadcrumb opened a file with the flag unset")

    monkeypatch.setattr(os, "open", refuse)
    margin.breadcrumb("capture_admit", bucket=(0, 1536, 4096))


def test_one_durable_line_per_call_through_one_write(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    path = tmp_path / "crumbs.log"
    monkeypatch.setenv(FLAG, str(path))
    opened: list[int] = []
    writes: list[bytes] = []
    real_open, real_write = os.open, os.write
    monkeypatch.setattr(os, "open", lambda name, flags, mode=0o777: opened.append(flags) or real_open(name, flags, mode))
    monkeypatch.setattr(os, "write", lambda fd, data: writes.append(data) or real_write(fd, data))
    margin.breadcrumb("workspace_over_ask", bucket=(0, 1536, 4096), fill_MiB=2387)
    margin.breadcrumb("workspace_over_ask_returned", held_MiB=5052)
    lines = path.read_text().splitlines()
    assert len(lines) == 2 and len(writes) == 2
    wanted = os.O_WRONLY | os.O_APPEND | os.O_CREAT | os.O_DSYNC | os.O_NONBLOCK
    assert all(flags & wanted == wanted for flags in opened)
    match = LINE.match(lines[0])
    assert match, lines[0]
    assert match.group(1) == str(os.getpid()) and match.group(3) == "workspace_over_ask"
    assert lines[0].endswith(" bucket=(0,1536,4096) fill_MiB=2387")
    assert lines[1].endswith(" step=workspace_over_ask_returned held_MiB=5052")


def test_the_rank_comes_from_the_launch_env_when_torch_distributed_is_not_up(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    path = tmp_path / "crumbs.log"
    monkeypatch.setenv(FLAG, str(path))
    monkeypatch.setenv("RANK", "3")
    margin.breadcrumb("pool_floor")
    assert " rank=3 step=pool_floor" in path.read_text()


def test_an_unwritable_file_warns_once_and_never_raises(monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
                                                         capsys: pytest.CaptureFixture) -> None:
    monkeypatch.setenv(FLAG, str(tmp_path / "missing-dir" / "crumbs.log"))
    margin.breadcrumb("capture_admit")
    margin.breadcrumb("captured")
    err = capsys.readouterr().err
    assert err.count("GLM53_BREADCRUMB_UNWRITABLE") == 1
    assert FLAG in err and "bind-mounted" in err


def test_the_pool_floor_check_leaves_a_breadcrumb(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    path = tmp_path / "crumbs.log"
    monkeypatch.setenv(FLAG, str(path))
    monkeypatch.setenv("HPU_GLM53_POOL_LIMIT_MAX_MIB", "96400")

    class Runtime:
        def stats(self) -> dict[str, int]:
            return {"Limit": 95_926 << 20, "InUse": 1_200 << 20, "MaxInUse": 1_200 << 20}

    monkeypatch.setattr(margin, "_runtime", Runtime)
    margin.enforce_pool_floor()
    assert " step=pool_floor limit_MiB=95926 refused=0" in path.read_text()
