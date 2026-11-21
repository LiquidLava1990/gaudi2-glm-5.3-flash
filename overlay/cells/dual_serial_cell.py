"""Bitwise check 2 of DUAL-SUSTAINED-PLAN.md section 12.3 (build step 6): the same recorded sequence replayed same-step vs separate-step, and
order d, p, d, p vs isolated, so decode cannot see prompt state; compared with torch.equal on hidden or token ids. It also records the three
excluded C787 controls on the current source. WAITING ON A CARD WINDOW: nothing here touches a card unless --execute is given; the default
is a dry run. This cell only reports: it carries no threshold and changes none.

    dual_serial_cell.py run --image <leaf> --port 18440 --out-dir <new dir> [--env-file graphs-c1.env] [--mount SRC:DST]... [--execute]
    dual_serial_cell.py fake --mode equal|unequal|empty     (the scoring path on a fake runner: no boot, no card)
    dual_serial_cell.py hidden --a <trace-dir> --b <trace-dir>   (saved hidden-r<rank>-<rows>-<width>-<start>.pt files, torch.equal per step)
    dual_serial_cell.py drive --port P --arm NAME --salt S --out <jsonl>   (the request step, run by `run`)

Four arms, one boot each through lab.py (so its quad lock and 25 min deadline refusals apply), on the C1 env minus the ballast:
same-step (the reference: prompt and decode in one mixed step), separate-step (VLLM_GAUDI_MIXED_SEPARATE_STEPS=1: one region per step),
isolated (the subject alone) and dpdp (the subject while prompts arrive, slice table 1024:1 so steps alternate d, p, d, p). The subject is
a greedy 4,096-token prompt with 64 output tokens. Pairs: separate-step vs same-step, dpdp vs isolated. Equal -> PASS, any difference -> FAIL
(the first differing step and token named), nothing compared or an arm error -> NOT_MEASURED, never PASS. The cell ends with one DUAL_SERIAL
line; exit 0 only for PASS. A FAIL or NOT_MEASURED control keeps the run from passing; a RECORDED control is neutral.

The separate-step arm needs a seam in glm53_mixed_graph.py / glm53_mixed_step.py (the flag above); `run --execute` refuses until it exists.
"""
from __future__ import annotations

import argparse
import ast
import json
import sys
import threading
import time
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Protocol

import c1_cell
import k1_cell
import memory_cells as cells
import schedule_compare_cell as scheduling

PASS, FAIL, NOT_MEASURED, RECORDED = "PASS", "FAIL", "NOT_MEASURED", "RECORDED"
QUAD = "A"
OVERLAY = Path(__file__).resolve().parents[1]
SEPARATE_FLAG, SLICE_FLAG = "VLLM_GAUDI_MIXED_SEPARATE_STEPS", scheduling.SLICE_FLAG
DPDP_SLICE = "1024:1"  # one decode-only step before each chunk at width 1024: the steps alternate d, p, d, p
DROPPED_ENV = scheduling.DROPPED_ENV
SEAM_FILES = ("glm53_mixed_graph.py", "glm53_mixed_step.py")
CONTROL_NAMES = ("diff-graph", "diff-step", "flag-off-observables")
SUBJECT_TOKENS, SUBJECT_OUTPUT_TOKENS = 4096, 64
PRESSURE_PROMPTS, PRESSURE_TOKENS, PRESSURE_DELAY_S = 4, 6144, 2
SUBJECT_TIMEOUT_S = k1_cell.long_timeout_s(PRESSURE_TOKENS) * PRESSURE_PROMPTS
DRIVE_BUDGET_S = SUBJECT_TIMEOUT_S + k1_cell.REQUEST_TIMEOUT_S
ROWS_NAME = "dual-serial.tokens.jsonl"
Send = Callable[..., dict]
Fetch = Callable[..., list[int]]


@dataclass(frozen=True)
class Arm:
    name: str
    separate: bool
    slice_table: str
    pressure: bool


@dataclass(frozen=True)
class Pair:
    name: str
    reference: str
    other: str


ARMS = (Arm("same-step", False, "", True), Arm("separate-step", True, "", True), Arm("isolated", False, "", False),
        Arm("dpdp", False, DPDP_SLICE, True))
PAIRS = (Pair("separate-vs-same-step", "same-step", "separate-step"), Pair("dpdp-vs-isolated", "isolated", "dpdp"))
ARM_BY_NAME = {arm.name: arm for arm in ARMS}


@dataclass(frozen=True)
class Reading:
    """One compared item: a decode step's token id (int) or one saved hidden tensor."""
    step: str
    value: object


@dataclass(frozen=True)
class ArmRun:
    readings: list[Reading]
    error: str = ""


class Runner(Protocol):
    def run_arm(self, arm: Arm) -> ArmRun: ...


class Controls(Protocol):
    def run(self) -> list[dict]: ...


def token_readings(tokens: Sequence[int]) -> list[Reading]:
    return [Reading(f"decode step {index}", token) for index, token in enumerate(tokens)]


def hidden_readings(directory: Path) -> list[Reading]:
    """The first hidden outputs a K1 trace dir holds, by rank and chunk start (the files k1_hidden_compare reads)."""
    import k1_hidden_compare
    import torch

    return [Reading(f"hidden r{rank} start {start}", torch.load(path, weights_only=True))
            for (rank, start), (_, path) in sorted(k1_hidden_compare.saved(directory).items())]


def values_equal(left: object, right: object) -> bool:
    if isinstance(left, int) and isinstance(right, int):
        return left == right
    if isinstance(left, int) or isinstance(right, int):
        return False
    import torch

    return bool(torch.equal(left, right))


def _describe(reference: Reading, value: object) -> str:
    if isinstance(reference.value, int) and isinstance(value, int):
        return f"{reference.step}: reference token {reference.value}, run token {value}"
    if hasattr(value, "shape") and hasattr(reference.value, "shape") and value.shape == reference.value.shape:
        return f"{reference.step}: hidden tensors differ (max abs diff {(value - reference.value).abs().max().item():.3e})"
    return f"{reference.step}: reference {type(reference.value).__name__}, run {type(value).__name__}, not comparable"


def compare_readings(reference: Sequence[Reading], other: Sequence[Reading]) -> tuple[str, str, str | None]:
    """(verdict, detail, first differing step). Nothing to compare is NOT_MEASURED: two empty runs are not 'equal'."""
    if not reference or not other:
        return NOT_MEASURED, f"reference has {len(reference)} readings, run has {len(other)}", None
    theirs = {item.step: item.value for item in other}
    for item in reference:
        if item.step not in theirs:
            return FAIL, f"{item.step} is missing from the run ({len(other)} readings, reference {len(reference)})", item.step
        if not values_equal(item.value, theirs[item.step]):
            return FAIL, _describe(item, theirs[item.step]), item.step
    known = {item.step for item in reference}
    extra = next((item.step for item in other if item.step not in known), None)
    if extra is not None:
        return FAIL, f"{extra} is not in the reference ({len(reference)} readings)", extra
    return PASS, f"{len(reference)} readings identical (torch.equal)", None


def _arm_error(pair: Pair, runs: dict[str, ArmRun]) -> str:
    return next((f"{name}: {runs[name].error}" for name in (pair.other, pair.reference) if runs[name].error), "")


def compare_pair(pair: Pair, runs: dict[str, ArmRun]) -> dict:
    error = _arm_error(pair, runs)
    verdict, detail, step = (NOT_MEASURED, error, None) if error else compare_readings(runs[pair.reference].readings, runs[pair.other].readings)
    return {"pair": pair.name, "reference": pair.reference, "other": pair.other, "verdict": verdict, "detail": detail,
            "first_difference": step, "compared": len(runs[pair.reference].readings)}


def overall_verdict(pair_verdicts: Iterable[str], control_verdicts: Iterable[str]) -> str:
    """FAIL beats NOT_MEASURED beats PASS; RECORDED controls are neutral; no pair at all is NOT_MEASURED."""
    pairs, controls = list(pair_verdicts), list(control_verdicts)
    seen = {*pairs, *controls}
    if FAIL in seen:
        return FAIL
    return PASS if pairs and NOT_MEASURED not in seen else NOT_MEASURED


def run_cell(runner: Runner, controls: Controls) -> dict:
    runs = {arm.name: runner.run_arm(arm) for arm in ARMS}
    pairs = [compare_pair(pair, runs) for pair in PAIRS]
    checked = controls.run()
    return {"verdict": overall_verdict((item["verdict"] for item in pairs), (item["verdict"] for item in checked)),
            "pairs": pairs, "controls": checked}


def arm_env(base: dict[str, str], arm: Arm) -> dict[str, str]:
    """The base env minus the ballast; the seam flag is stated for every arm, the slice table only for the arm that has one."""
    env = {key: value for key, value in base.items() if key not in DROPPED_ENV}
    env[SEPARATE_FLAG] = "1" if arm.separate else "0"
    if arm.slice_table:
        env[SLICE_FLAG] = arm.slice_table
    return env


def seam_missing(overlay: Path) -> list[str]:
    """The overlay files that do not yet name the separate-step flag."""
    present = [(overlay / name).is_file() and SEPARATE_FLAG in (overlay / name).read_text(encoding="utf-8") for name in SEAM_FILES]
    return [name for name, found in zip(SEAM_FILES, present, strict=True) if not found]


def _class_digests(node: ast.ClassDef) -> dict[str, str]:
    found = {node.name: ast.dump(ast.ClassDef(node.name, node.bases, node.keywords, [], node.decorator_list))}
    for item in node.body:
        if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
            found[f"{node.name}.{item.name}"] = ast.dump(item)
    return found


def definition_digests(path: Path) -> dict[str, str]:
    """{definition name: ast dump} for a file's functions, classes and methods, and its module-level assignments."""
    found: dict[str, str] = {}
    for node in ast.parse(path.read_text(encoding="utf-8")).body:
        if isinstance(node, ast.ClassDef):
            found.update(_class_digests(node))
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            found[node.name] = ast.dump(node)
        elif isinstance(node, (ast.Assign, ast.AnnAssign)):
            found[f"assign:{ast.unparse(node).split('=')[0].strip()}"] = ast.dump(node)
    return found


def diff_control(name: str, original: Path, current: Path) -> dict:
    """Differential on the current source: which definitions the overlay's copy changed, added or removed against the combined original."""
    missing = [str(path) for path in (original, current) if not path.is_file()]
    if missing:
        return {"control": name, "verdict": NOT_MEASURED, "detail": f"cannot read {', '.join(missing)}: mount the overlay and tests/fixtures/nativepin"}
    old, new = definition_digests(original), definition_digests(current)
    changed, added, removed = (sorted(key for key in old if key in new and old[key] != new[key]), sorted(set(new) - set(old)), sorted(set(old) - set(new)))
    return {"control": name, "verdict": RECORDED, "detail": f"{len(changed)} changed, {len(added)} added, {len(removed)} removed",
            "changed": changed, "added": added, "removed": removed}


def pin_flag_off_case() -> None:
    """C787's flag-off control as dual-4m wrote it, run on the overlay's files through the CPU shim (needs torch: run in the image)."""
    sys.path.insert(0, str(OVERLAY / "tests"))
    import test_c789_cases as c789

    cases = c789.load_cases()
    c789.run_case(cases, cases.test_flag_off_observables_equal_combined, {})


def flag_off_control(case: Callable[[], None] = pin_flag_off_case) -> dict:
    name = CONTROL_NAMES[2]
    try:
        case()
    except AssertionError as error:
        return {"control": name, "verdict": FAIL, "detail": f"flag-off observables differ from the combined original: {error}"}
    except (ImportError, OSError) as error:
        return {"control": name, "verdict": NOT_MEASURED, "detail": f"could not run the case ({type(error).__name__}: {error}); run it inside the image"}
    return {"control": name, "verdict": PASS, "detail": "flag-off observables equal the combined original"}


class SourceControls:
    """The three excluded C787 controls, run on the overlay's current source against the combined original."""

    def __init__(self, overlay: Path, fixtures: Path, flag_off: Callable[[], None] = pin_flag_off_case) -> None:
        self.overlay, self.fixtures, self.flag_off = overlay, fixtures, flag_off

    def run(self) -> list[dict]:
        diffs = [diff_control(name, self.fixtures / "orig" / file, self.overlay / file) for name, file in zip(CONTROL_NAMES[:2], SEAM_FILES, strict=True)]
        return [*diffs, flag_off_control(self.flag_off)]


class FixedControls:
    def __init__(self, *verdicts: str) -> None:
        self.verdicts = verdicts

    def run(self) -> list[dict]:
        return [{"control": name, "verdict": verdict, "detail": "fake"} for name, verdict in zip(CONTROL_NAMES, self.verdicts, strict=True)]


class FakeRunner:
    """A runner that boots nothing: each arm's ArmRun is scripted."""

    def __init__(self, scripts: dict[str, ArmRun]) -> None:
        self.scripts = scripts

    def run_arm(self, arm: Arm) -> ArmRun:
        return self.scripts[arm.name]

    @classmethod
    def from_mode(cls, mode: str) -> FakeRunner:
        tokens = list(range(1000, 1000 + SUBJECT_OUTPUT_TOKENS))
        scripts = {arm.name: ArmRun(token_readings(tokens)) for arm in ARMS}
        if mode == "unequal":
            scripts["separate-step"] = ArmRun(token_readings(tokens[:17] + [tokens[17] + 1] + tokens[18:]))
        if mode == "empty":
            scripts["isolated"] = ArmRun([])
        return cls(scripts)


def _failure(error: BaseException) -> str:
    return f"{type(error).__name__}: {error}"


def _pressure(args: argparse.Namespace, send: Send, errors: list[str]) -> None:
    """Cold prompts arriving one after another while the subject decodes."""
    time.sleep(PRESSURE_DELAY_S)
    for index in range(PRESSURE_PROMPTS):
        try:
            send(args.port, k1_cell.fixed_prompt(PRESSURE_TOKENS), 1, timeout=k1_cell.long_timeout_s(PRESSURE_TOKENS), salt=f"{args.salt}-pressure-{index}")
        except Exception as error:  # the row records it; the subject's tokens stay readable
            errors.append(f"pressure prompt {_failure(error)}")
            return


def _subject_tokens(args: argparse.Namespace, fetch: Fetch, errors: list[str]) -> list[int]:
    try:
        return fetch(args.port, k1_cell.fixed_prompt(SUBJECT_TOKENS), SUBJECT_OUTPUT_TOKENS, timeout=SUBJECT_TIMEOUT_S, salt=f"{args.salt}-subject")
    except Exception as error:  # a dropped connection or a timeout is the arm's result
        errors.append(f"subject {_failure(error)}")
        return []


def drive_arm(args: argparse.Namespace, send: Send = c1_cell.stream, fetch: Fetch = scheduling.request_token_ids) -> int:
    """The subject, with the pressure prompts alongside when the arm has them; one row of token ids and errors."""
    errors: list[str] = []
    pressure = threading.Thread(target=_pressure, args=(args, send, errors), daemon=True)
    has_pressure = ARM_BY_NAME[args.arm].pressure
    if has_pressure:
        pressure.start()
    tokens = _subject_tokens(args, fetch, errors)
    if has_pressure:
        pressure.join(timeout=SUBJECT_TIMEOUT_S)
    c1_cell.append_row(args.out, {"arm": args.arm, "tokens": tokens, "error": "; ".join(errors)})
    return 0 if tokens and not errors else 1


def arm_boot(base_env: dict[str, str], args: argparse.Namespace, arm: Arm) -> cells.Boot:
    drive = ["python3", str(Path(__file__).resolve()), "drive", "--port", str(args.port), "--arm", arm.name, "--salt", args.salt,
             "--out", str(args.out_dir / ROWS_NAME)]
    return cells.Boot(f"claude-flash-launch-dual-{arm.name}", arm_env(base_env, arm), [cells.Step("drive", drive, DRIVE_BUDGET_S)])


class LabRunner:
    """One lab.py boot per arm; down always runs. The drive's row carries the tokens."""

    def __init__(self, base_env: dict[str, str], args: argparse.Namespace) -> None:
        self.base_env, self.args = base_env, args

    def run_arm(self, arm: Arm) -> ArmRun:
        results = k1_cell.run_k1_boot(arm_boot(self.base_env, self.args, arm), self.args)
        failed = next((step for step in results if step["returncode"] != 0), None)
        if failed is not None:
            return ArmRun([], f"{failed['label']} step returned {failed['returncode']} ({failed['stderr']})".strip())
        row = scheduling.arm_tokens(self.args.out_dir / ROWS_NAME, arm.name)
        return ArmRun(token_readings(row.tokens), row.error)


def render_report(result: dict) -> str:
    pairs = [f"{item['verdict']:<13} {item['pair']:<22} {item['detail']}" for item in result["pairs"]]
    controls = [f"{item['verdict']:<13} control {item['control']:<22} {item['detail']}" for item in result["controls"]]
    return "\n".join([*pairs, *controls, f"DUAL_SERIAL: {result['verdict']}"])


def dry_run_plan(base_env: dict[str, str], args: argparse.Namespace) -> dict[str, object]:
    arms = []
    for arm in ARMS:
        steps = cells.boot_steps(arm_boot(base_env, args, arm), args)
        arms.append({"arm": arm.name, "env": arm_env(base_env, arm), "worst_case_s": sum(step.timeout_s for step in steps),
                     "steps": [{"label": s.label, "timeout_s": s.timeout_s, "argv": s.argv} for s in steps]})
    return {"dry_run": True, "touches_cards": False, "cell": "dual-serial", "quad": QUAD, "arms": arms,
            "pairs": [pair.name for pair in PAIRS], "controls": list(CONTROL_NAMES), "seam_missing": seam_missing(OVERLAY)}


def require_ready(args: argparse.Namespace) -> None:
    """Refuse before any boot: the seam must exist and the output folder must be new."""
    missing = seam_missing(OVERLAY)
    if missing:
        raise SystemExit(f"{', '.join(missing)} lack {SEPARATE_FLAG}, so the separate-step arm cannot run. Apply the seam patch "
                         f"(a flag that makes the scheduler emit prompt and decode regions in separate steps), then rerun")
    if args.out_dir.exists() and any(args.out_dir.iterdir()):
        raise SystemExit(f"{args.out_dir} already holds files; one output folder per run: pass a new --out-dir")


def execute(base_env: dict[str, str], args: argparse.Namespace) -> int:
    require_ready(args)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    result = run_cell(LabRunner(base_env, args), SourceControls(OVERLAY, OVERLAY / "tests" / "fixtures" / "nativepin"))
    path = args.out_dir / f"dual-serial-{args.stamp}.json"
    path.write_text(json.dumps(result, indent=2, default=str) + "\n")
    print(render_report(result))
    print("DUAL_SERIAL_RESULT", json.dumps({"verdict": result["verdict"], "result": str(path)}), flush=True)
    return 0 if result["verdict"] == PASS else 1


def compare_hidden(first: Path, second: Path) -> int:
    verdict, detail, _ = compare_readings(hidden_readings(first), hidden_readings(second))
    print(f"{verdict:<13} {detail}\nDUAL_SERIAL_HIDDEN: {verdict}")
    return 0 if verdict == PASS else 1


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run")
    run.add_argument("--image", required=True, help="the claude-flash-* leaf holding this overlay")
    run.add_argument("--port", required=True, type=k1_cell.lab_port)
    run.add_argument("--out-dir", required=True, type=Path, help="a new folder: one output folder per run")
    run.add_argument("--env-file", type=Path, default=c1_cell.DEFAULT_ENV_FILE)
    run.add_argument("--mount", action="append", default=[], help="SRC:DST for lab.py up, e.g. the stall trace and Habana logs")
    run.add_argument("--stamp", default=datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"))
    run.add_argument("--salt", default="", help="cache-salt prefix; default claude-dual-<stamp>")
    mode = run.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true", help="the default: print the plan, touch nothing")
    mode.add_argument("--execute", action="store_true", help="boot quad A once per arm and compare (needs the card window)")
    fake = sub.add_parser("fake")
    fake.add_argument("--mode", required=True, choices=("equal", "unequal", "empty"))
    hidden = sub.add_parser("hidden")
    hidden.add_argument("--a", required=True, type=Path)
    hidden.add_argument("--b", required=True, type=Path)
    driver = sub.add_parser("drive")
    driver.add_argument("--port", required=True, type=k1_cell.lab_port)
    driver.add_argument("--arm", required=True, choices=list(ARM_BY_NAME))
    driver.add_argument("--salt", required=True)
    driver.add_argument("--out", required=True, type=Path)
    args = parser.parse_args(argv)
    if args.command == "run":
        args.quad, args.salt = QUAD, args.salt or f"claude-dual-{args.stamp}"
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.command == "drive":
        return drive_arm(args)
    if args.command == "hidden":
        return compare_hidden(args.a, args.b)
    if args.command == "fake":
        result = run_cell(FakeRunner.from_mode(args.mode), FixedControls(RECORDED, RECORDED, PASS))
        print(render_report(result))
        return 0 if result["verdict"] == PASS else 1
    base_env = c1_cell.load_env(args.env_file)
    c1_cell.require_stall_mount(base_env, args)
    if not args.execute:
        print(json.dumps(dry_run_plan(base_env, args), indent=2))
        return 0
    return execute(base_env, args)


if __name__ == "__main__":
    raise SystemExit(main())
