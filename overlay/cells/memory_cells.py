"""Quad card cells of plan steps 20, 21 and 22. WAITING ON A CARD: run only in a card window the owner scheduled.

    memory_cells.py census   --quad A --image <leaf> --port 18430 --out-dir <dir> [--rows 512,1024] [--drains on,off] [--dry-run]
    memory_cells.py owner    --quad A --image <leaf> --port 18430 --out-dir <dir> [--env K=V]... [--env-file F] [--dry-run]
    memory_cells.py runahead --quad A --image <leaf> --port 18430 --out-dir <dir> [--margin-mib 512] [--sample-every 1] [--dry-run]

Every boot goes through the lane's lab.py, so its quad lock, source-unit, replica-port and boot-deadline refusals all
apply; every step has a timeout and `down` always runs. --dry-run prints the plan as JSON and runs nothing. Each cell ends
with one MEMORY_CELL line. A cell that did not exercise its flag says VACUOUS or NOT_MEASURED; neither is a pass.
"""
from __future__ import annotations

import argparse
import json
import random
import re
import subprocess
import sys
import time
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

CELLS_DIR = Path(__file__).resolve().parent
LANE = CELLS_DIR.parents[1]
FLASH = LANE.parent
REPLICA_PORTS = frozenset({18274, 18275})
STOP_DEADLINE_S = 300
BOOT_DEADLINE_S = 1500  # lab.py's own; the boot budget refuses anything longer
UP_TIMEOUT_S = BOOT_DEADLINE_S + 300  # includes the recipe leaf copy
DRIVE_TIMEOUT_S = 3600  # 500K tokens at the measured 4.2 ms/token at 512 rows is 35 min
GATES_TIMEOUT_S = 3600
C16_TIMEOUT_S = 1800
MAX_STEP_S = DRIVE_TIMEOUT_S + 60
ROWS = (512, 1024, 1792, 2048)
LARGEST_CONTEXT = 500_000
DRAINS_ON = {"HPU_GLM53_R616_DRAIN_SITES": "kda.owned,kda.sequence,moe.owned,moe.return", "HPU_GLM53_R606_RELAX_TILE_DRAINS": "0"}
DRAINS_OFF = {"HPU_GLM53_R616_DRAIN_SITES": "none", "HPU_GLM53_R606_RELAX_TILE_DRAINS": "1"}
CAP_OFF = {"VLLM_PREFILL_CONTEXT_CELL_BUDGET": "0", "HPU_GLM53_STEP_CELL_BUDGET": "0"}  # this diagnostic only (plan step 20)
HOT_C16_MAX_RATIO = 1.03
SATURATED_BAR = 0.001
SCORED_ROWS = 2048
PROMPT_SEED = 500_000
OWNER_LINE = "GLM53_Q4_SHARED_OWNER"
FLOOR_LINE = re.compile(r"GLM53_GS_FLOOR event=(\w+) rows=(\d+) .*retained_peak_mib=(\d+)")
SATURATION_LINE = re.compile(r"GLM53_GS_SATURATION rows=(\d+) .*saturated=([01])")


@dataclass(frozen=True)
class Step:
    label: str
    argv: list[str]
    timeout_s: int


@dataclass
class Boot:
    name: str
    env: dict[str, str]
    actions: list[Step] = field(default_factory=list)


def _python(script: Path, *args: str) -> list[str]:
    return ["python3", str(script), *args]


def _gates(name: str, port: int, tag: str) -> Step:
    return Step("gates", _python(LANE / "launch_gates.py", "--name", name, "--port", str(port), "--tag", tag), GATES_TIMEOUT_S)


def _c16(name: str, port: int, tag: str) -> Step:
    return Step("c16", _python(LANE / "run_c16.py", "--name", name, "--port", str(port), "--tag", tag), C16_TIMEOUT_S)


def extra_env(args: argparse.Namespace) -> dict[str, str]:
    env: dict[str, str] = {}
    if args.env_file is not None:
        env.update(line.split("=", 1) for line in args.env_file.read_text().splitlines() if line.strip())
    env.update(item.split("=", 1) for item in args.env)
    return env


def census_boots(args: argparse.Namespace) -> list[Boot]:
    """rows x drains: one boot and one cold request each. The request passes every context of the matrix on its way."""
    boots = []
    for rows in args.rows:
        for drains in args.drains:
            name = f"claude-flash-launch-census-r{rows}-d{drains}"
            env = {**extra_env(args), **CAP_OFF, **(DRAINS_ON if drains == "on" else DRAINS_OFF),
                   "VLLM_ADAPTIVE_LONG_PREFILL_SINGLE_TOKENS": str(rows), "HPU_GLM53_PREFILL_HBM_CENSUS": "1"}
            drive = _python(Path(__file__), "drive", "--port", str(args.port), "--tokens", str(LARGEST_CONTEXT + 2 * rows),
                            "--timeout", str(DRIVE_TIMEOUT_S), "--out", str(args.out_dir / f"{name}.drive.json"))
            boots.append(Boot(name, env, [Step("drive", drive, DRIVE_TIMEOUT_S + 60)]))
    return boots


def owner_boots(args: argparse.Namespace) -> list[Boot]:
    boots = []
    for value, arm in (("0", "off"), ("1", "on")):
        name = f"claude-flash-launch-owner-{arm}"
        env = {**extra_env(args), "HPU_GLM53_Q4_SHARED_GATHER_OWNER": value}
        boots.append(Boot(name, env, [_gates(name, args.port, f"owner-{arm}-{args.stamp}")]))
    return boots


def runahead_boots(args: argparse.Namespace) -> list[Boot]:
    """Flag off for the hot C16 reference, then two fresh flag-on boots: the 70K repeat is bimodal per boot [M KB 31105]."""
    on_env = {**extra_env(args), "HPU_GLM53_RUNAHEAD_CREDIT": "1", "HPU_GLM53_RUNAHEAD_MARGIN_MIB": str(args.margin_mib),
              "HPU_GLM53_RUNAHEAD_SAMPLE_EVERY": str(args.sample_every)}
    off, first, second = (f"claude-flash-launch-runahead-{arm}" for arm in ("off", "on1", "on2"))
    return [Boot(off, {**extra_env(args), "HPU_GLM53_RUNAHEAD_CREDIT": "0"}, [_c16(off, args.port, f"runahead-off-{args.stamp}")]),
            Boot(first, dict(on_env), [_c16(first, args.port, f"runahead-on-{args.stamp}"),
                                       _gates(first, args.port, f"runahead-on1-{args.stamp}")]),
            Boot(second, dict(on_env), [_gates(second, args.port, f"runahead-on2-{args.stamp}")])]


def boot_steps(boot: Boot, args: argparse.Namespace) -> list[Step]:
    up = _python(LANE / "lab.py", "up", "--quad", args.quad, "--name", boot.name, "--image", args.image,
                 "--port", str(args.port), "--deadline", str(BOOT_DEADLINE_S))
    for key, value in boot.env.items():
        up += ["--env", f"{key}={value}"]
    for mount in getattr(args, "mount", []):
        up += ["--mount", mount]
    if getattr(args, "fresh_cache", False):
        up.append("--fresh-cache")  # lab.py: an empty recipe leaf, for graphs no earlier recipe was compiled for
    down = _python(LANE / "lab.py", "down", "--name", boot.name)
    return [Step("up", up, UP_TIMEOUT_S), *boot.actions, Step("down", down, STOP_DEADLINE_S)]


def _run_step(step: Step) -> dict[str, object]:
    try:
        code: object = subprocess.run(step.argv, timeout=step.timeout_s, check=False).returncode
    except subprocess.TimeoutExpired:
        code = "timeout"
    return {"label": step.label, "returncode": code}


def run_boot(boot: Boot, args: argparse.Namespace) -> list[dict[str, object]]:
    """up, then the actions while the boot is usable, and always down."""
    up, *actions, down = boot_steps(boot, args)
    results = [_run_step(up)]
    if results[0]["returncode"] == 0:
        for action in actions:
            results.append(_run_step(action))
            if results[-1]["returncode"] == "timeout":
                break
    results.append(_run_step(down))
    return results


def _lines(name: str) -> list[str]:
    path = LANE / f"{name}.log"  # lab.py down writes it
    return path.read_text(errors="replace").splitlines() if path.is_file() else []


def _gates_passed(tag: str) -> bool:
    path = LANE / f"gates-{tag}" / "summary.json"
    return path.is_file() and bool(json.loads(path.read_text()).get("passed"))


def _hot_makespan(tag: str) -> float | None:
    path = LANE / f"c16-{tag}" / "summary.json"
    return float(json.loads(path.read_text())["hot_makespan_s"]) if path.is_file() else None


def _floor_summary(lines: Iterable[str]) -> tuple[int | None, int]:
    """(boot graphs at seal, the largest retained peak of any rank)."""
    graphs, peak = None, 0
    for line in lines:
        match = FLOOR_LINE.search(line)
        if match is None:
            continue
        if match.group(1) == "seal":
            graphs = int(match.group(2))
        peak = max(peak, int(match.group(3)))
    return graphs, peak


def owner_verdict(off: Sequence[str], on: Sequence[str], gates_passed: bool) -> dict[str, object]:
    """Step 21. The same graphs, a retained peak no higher, gates exact, and the owner actually ran."""
    (graphs_off, peak_off), (graphs_on, peak_on) = _floor_summary(off), _floor_summary(on)
    owners = sum(OWNER_LINE in line for line in on)
    if not owners:
        verdict = "VACUOUS"  # the flag acts only inside capture_context: the arm's env must run mixed graphs
    elif graphs_off is None or graphs_off != graphs_on:
        verdict = "NOT_COMPARABLE"
    elif not gates_passed or peak_on > peak_off:
        verdict = "FAIL"
    else:
        verdict = "PASS"
    return {"cell": "owner", "verdict": verdict, "owner_lines": owners, "graphs": {"off": graphs_off, "on": graphs_on},
            "retained_peak_mib": {"off": peak_off, "on": peak_on}, "gates_passed": gates_passed}


def runahead_verdict(lines: Iterable[str], hot_on_s: float | None, hot_off_s: float | None,
                     gates_passed: tuple[bool, bool]) -> dict[str, object]:
    """Step 22: saturated=1 under 0.1% at 2048 rows, hot C16 within 3% of flag off, gates exact on two fresh boots."""
    flags = [int(match.group(2)) for match in map(SATURATION_LINE.search, lines)
             if match is not None and int(match.group(1)) == SCORED_ROWS]
    saturated = sum(flags)
    ratio = saturated / len(flags) if flags else None
    hot = hot_on_s / hot_off_s if hot_on_s and hot_off_s else None
    if ratio is None or hot is None:
        verdict = "NOT_MEASURED"
    elif ratio < SATURATED_BAR and hot <= HOT_C16_MAX_RATIO and all(gates_passed):
        verdict = "PASS"
    else:
        verdict = "FAIL"
    return {"cell": "runahead", "verdict": verdict, "saturated_readings": saturated, "readings_2048": len(flags),
            "saturated_ratio": ratio, "hot_ratio": hot, "gates_passed": list(gates_passed)}


def score_census(args: argparse.Namespace, boots: list[Boot]) -> dict[str, object]:
    logs = []
    for boot in boots:
        begin_path = args.out_dir / f"{boot.name}.drive.json"
        if begin_path.is_file():
            drains = "on" if boot.env["HPU_GLM53_R606_RELAX_TILE_DRAINS"] == "0" else "off"
            logs += ["--log", f"{drains}:{LANE / (boot.name + '.log')}:{json.loads(begin_path.read_text())['begin_utc']}"]
    if not logs:
        return {"cell": "census", "verdict": "NOT_MEASURED", "reason": "no request began"}
    code = subprocess.run(_python(CELLS_DIR / "census_table.py", *logs, "--out-dir", str(args.out_dir)), check=False).returncode
    return {"cell": "census", "verdict": "SEE_VERDICTS" if code == 0 else "FAIL", "out_dir": str(args.out_dir)}


def score(args: argparse.Namespace, boots: list[Boot]) -> dict[str, object]:
    if args.cell == "census":
        return score_census(args, boots)
    if args.cell == "owner":
        return owner_verdict(_lines(boots[0].name), _lines(boots[1].name), _gates_passed(f"owner-on-{args.stamp}"))
    on_lines = _lines(boots[1].name) + _lines(boots[2].name)
    return runahead_verdict(on_lines, _hot_makespan(f"runahead-on-{args.stamp}"), _hot_makespan(f"runahead-off-{args.stamp}"),
                            (_gates_passed(f"runahead-on1-{args.stamp}"), _gates_passed(f"runahead-on2-{args.stamp}")))


def drive(args: argparse.Namespace) -> int:
    """One salted cold request that passes every context of the matrix. Records when it began, so warmup lines drop out."""
    sys.path.insert(0, str(FLASH))
    from probe_flash_c16_starvation import stream

    rng = random.Random(PROMPT_SEED)
    prompt = [rng.randrange(1000, 16000) for _ in range(args.tokens)]
    begin = datetime.now(timezone.utc).isoformat()
    args.out.write_text(json.dumps({"begin_utc": begin, "tokens": args.tokens}) + "\n")
    result = stream(args.port, prompt, 1, timeout=args.timeout, cache_salt=f"claude-census-{time.time_ns()}")
    args.out.write_text(json.dumps({"begin_utc": begin, "tokens": args.tokens, "ttft_s": result["ttft_s"]}) + "\n")
    return 0


BOOTS = {"census": census_boots, "owner": owner_boots, "runahead": runahead_boots}


def _lab_port(raw: str) -> int:
    port = int(raw)
    if port in REPLICA_PORTS:
        raise SystemExit(f"port {port} is a production replica port ({sorted(REPLICA_PORTS)}); pick a lab port such as 18430")
    return port


def _comma_rows(raw: str) -> tuple[int, ...]:
    return tuple(int(item) for item in raw.split(","))


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cell", required=True)
    for cell in BOOTS:
        item = sub.add_parser(cell)
        item.add_argument("--quad", required=True, choices=("A", "B"))
        item.add_argument("--image", required=True, help="the claude-flash-* leaf holding this overlay")
        item.add_argument("--port", required=True, type=_lab_port)
        item.add_argument("--out-dir", required=True, type=Path)
        item.add_argument("--env", action="append", default=[])
        item.add_argument("--env-file", type=Path)
        item.add_argument("--stamp", default=datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"))
        item.add_argument("--dry-run", action="store_true")
        if cell == "census":
            item.add_argument("--rows", type=_comma_rows, default=ROWS)
            item.add_argument("--drains", type=lambda raw: tuple(raw.split(",")), default=("on", "off"))
        if cell == "runahead":
            item.add_argument("--margin-mib", type=int, default=512)
            item.add_argument("--sample-every", type=int, default=1)
    driver = sub.add_parser("drive")
    driver.add_argument("--port", required=True, type=_lab_port)
    driver.add_argument("--tokens", required=True, type=int)
    driver.add_argument("--timeout", required=True, type=int)
    driver.add_argument("--out", required=True, type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.cell == "drive":
        return drive(args)
    boots = BOOTS[args.cell](args)
    steps = [step for boot in boots for step in boot_steps(boot, args)]
    if args.dry_run:
        print(json.dumps({"dry_run": True, "cell": args.cell, "waiting_on": f"a card window on quad {args.quad}",
                          "boots": len(boots), "worst_case_s": sum(step.timeout_s for step in steps),
                          "steps": [{"label": s.label, "timeout_s": s.timeout_s, "argv": s.argv} for s in steps]}, indent=2))
        return 0
    args.out_dir.mkdir(parents=True, exist_ok=True)
    results = {boot.name: run_boot(boot, args) for boot in boots}
    verdict = score(args, boots)
    (args.out_dir / f"{args.cell}-{args.stamp}.json").write_text(json.dumps({"results": results, **verdict}, indent=2) + "\n")
    print("MEMORY_CELL", json.dumps(verdict), flush=True)
    return 0 if verdict["verdict"] in ("PASS", "SEE_VERDICTS") else 1


if __name__ == "__main__":
    raise SystemExit(main())
