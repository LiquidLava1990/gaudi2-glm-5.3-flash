#!/usr/bin/env python3
"""Host-side stall sampler for one K1 lab container: every period, per worker, top threads by CPU, main-thread wchan and kernel stack.

usage: k1_host_sampler.py <container> <out-dir> [--period 30] [--gdb-after 240]
Each worker is named by its /dev/accel fd (hl index). When a key capture stays open --gdb-after seconds, one
`gdb thread apply all bt` per rank is taken (once). Reads only; exits when the container is gone.
"""
from __future__ import annotations

import argparse
import os
import re
import subprocess
import time
from pathlib import Path

CLOCK_TICKS = os.sysconf("SC_CLK_TCK")
TOP_THREADS = 6
DMESG_MATCH = re.compile(r"habanalabs|Hardware Error|Out of memory|oom-kill|ctx_init|Failed to create context|mce:", re.IGNORECASE)
GDB_TIMEOUT_S = 200


def new_dmesg(lines: list[str], seen: int) -> tuple[list[str], int]:
    """(matching lines after the first `seen`, the new count). A ring shorter than `seen` has wrapped, so it starts over."""
    start = seen if seen <= len(lines) else 0
    return [line for line in lines[start:] if DMESG_MATCH.search(line)], len(lines)


def host_line(free_text: str, loadavg: str) -> str:
    """`free -g` and /proc/loadavg as one line."""
    fields = next(line for line in free_text.splitlines() if line.startswith("Mem:")).split()
    return f"host mem_free={fields[3]} mem_available={fields[6]} buff_cache={fields[5]} load={loadavg.split()[0]}"


def sample_host(seen: int, out: object) -> int:
    free_text = subprocess.run(["free", "-g"], capture_output=True, text=True, check=False).stdout
    print(f"{time.strftime('%T')} {host_line(free_text, Path('/proc/loadavg').read_text())}", file=out, flush=True)
    ring = subprocess.run(["sudo", "-n", "dmesg", "-T"], capture_output=True, text=True, check=False).stdout.splitlines()
    lines, seen = new_dmesg(ring, seen)
    for line in lines:
        print(f"{time.strftime('%T')} dmesg {line[:200]}", file=out, flush=True)
    return seen


def worker_pids(container: str) -> dict[int, str]:
    """{host pid: comm} of the container's worker processes."""
    done = subprocess.run(["docker", "top", container, "-eo", "pid,comm"], capture_output=True, text=True, check=False)
    if done.returncode != 0:
        return {}
    found: dict[int, str] = {}
    for line in done.stdout.splitlines()[1:]:
        fields = line.split(None, 1)
        if len(fields) == 2 and fields[0].isdigit() and "Worker" in fields[1]:
            found[int(fields[0])] = fields[1]
    return found


def sudo_read(path: str) -> str:
    done = subprocess.run(["sudo", "-n", "cat", path], capture_output=True, text=True, check=False)
    return done.stdout


def accel_of(pid: int) -> str:
    """The /dev/accel node this pid holds, from a root readlink of its fds."""
    done = subprocess.run(["sudo", "-n", "find", f"/proc/{pid}/fd", "-lname", "/dev/accel/accel*", "-printf", "%l\n"],
                          capture_output=True, text=True, check=False)
    nodes = sorted({line for line in done.stdout.splitlines() if re.fullmatch(r"/dev/accel/accel\d+", line)})
    return ",".join(nodes) or "?"


def thread_ticks(pid: int) -> dict[int, tuple[str, int, str]]:
    """{tid: (comm, utime+stime ticks, state)}."""
    table: dict[int, tuple[str, int, str]] = {}
    for tid_dir in Path(f"/proc/{pid}/task").iterdir():
        try:
            stat = (tid_dir / "stat").read_text()
        except OSError:
            continue
        comm = stat[stat.index("(") + 1:stat.rindex(")")]
        rest = stat[stat.rindex(")") + 2:].split()
        table[int(tid_dir.name)] = (comm, int(rest[11]) + int(rest[12]), rest[0])
    return table


def sample_worker(pid: int, previous: dict[int, tuple[str, int, str]], period: float, out: object) -> dict[int, tuple[str, int, str]]:
    now = thread_ticks(pid)
    deltas = sorted(((now[tid][1] - previous.get(tid, ("", 0, ""))[1], tid) for tid in now), reverse=True)[:TOP_THREADS]
    top = " ".join(f"{now[tid][0]}/{tid}:{delta / CLOCK_TICKS / period:.2f}{now[tid][2]}" for delta, tid in deltas if delta > 0)
    total = sum(now[tid][1] - previous.get(tid, ("", 0, ""))[1] for tid in now) / CLOCK_TICKS / period
    wchan = sudo_read(f"/proc/{pid}/wchan").strip()
    stack = "<".join(line.split()[-1].split("+")[0] for line in sudo_read(f"/proc/{pid}/stack").splitlines()[:6])
    print(f"{time.strftime('%T')} pid={pid} cpu={total:.2f} threads={len(now)} main_state={now[pid][2]} wchan={wchan} "
          f"kstack={stack} top={top}", file=out, flush=True)
    return now


def capture_open(out_dir: Path) -> bool:
    """A trace file exists whose last event line is not `end`: some rank is inside a key capture."""
    for path in out_dir.glob("k1stall-*.txt"):
        events = [line for line in path.read_text(errors="replace").splitlines() if line[:1].isdigit() and " " in line]
        if events and events[-1].split()[1] != "end":
            return True
    return False


def gdb_all(pids: dict[int, str], out_dir: Path) -> None:
    for pid in pids:
        with open(out_dir / f"gdb-{pid}.txt", "w") as out:
            subprocess.run(["sudo", "-n", "timeout", str(GDB_TIMEOUT_S), "gdb", "-p", str(pid), "--batch", "-nx",
                            "-ex", "set pagination off", "-ex", "thread apply all bt 24"],
                           stdout=out, stderr=subprocess.STDOUT, check=False)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("container")
    parser.add_argument("out_dir", type=Path)
    parser.add_argument("--period", type=float, default=30.0)
    parser.add_argument("--gdb-after", type=float, default=240.0)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    previous: dict[int, dict[int, tuple[str, int, str]]] = {}
    took_gdb = False
    open_since: float | None = None
    misses = 0
    seen = len(subprocess.run(["sudo", "-n", "dmesg"], capture_output=True, text=True, check=False).stdout.splitlines())
    while misses < 40:
        pids = worker_pids(args.container)
        misses = 0 if pids else misses + 1
        with open(args.out_dir / "sampler.txt", "a") as out:
            seen = sample_host(seen, out)
            for pid in pids:
                if pid not in previous:
                    print(f"{time.strftime('%T')} WORKER pid={pid} comm={pids[pid]} accel={accel_of(pid)}", file=out, flush=True)
                try:
                    previous[pid] = sample_worker(pid, previous.get(pid, {}), args.period, out)
                except OSError as error:
                    print(f"{time.strftime('%T')} pid={pid} read failed: {error}", file=out, flush=True)
            open_since = (open_since or time.time()) if capture_open(args.out_dir) else None
            print(f"{time.strftime('%T')} load={Path('/proc/loadavg').read_text().split()[0]} capture_open_since={open_since}", file=out, flush=True)
        if pids and not took_gdb and open_since is not None and time.time() - open_since > args.gdb_after:
            gdb_all(pids, args.out_dir)
            took_gdb = True
        time.sleep(args.period)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
