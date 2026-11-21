"""Tests for champion_boot.sh: starting a champion replica again after it stopped.

Everything outside the script is a stand-in: docker (its state and event history live in files), systemctl (the docker
service), serve_champion.sh (records what it found, then marks the replica serving) and a quad holder that behaves like quad_hold.sh (it ends when its release file
appears, and removes that file).

Run on the lab host:  python3 test_champion_boot.py
Test another copy:    CHAMPION_BOOT=/path/to/champion_boot.sh.new python3 test_champion_boot.py
"""
from __future__ import annotations

import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import time
import unittest

KIT_DIR = Path(__file__).resolve().parent
SCRIPT_UNDER_TEST = Path(os.environ.get("CHAMPION_BOOT", str(KIT_DIR / "champion_boot.sh")))
CUT = "tcut"
ROLE = "r0"
RUN_FOLDER = f"serve-{CUT}-{ROLE}"
HOLDER_START_TIMEOUT_S = 5.0

FAKE_DOCKER = """#!/bin/bash
# docker stand-in. $FAKE_STATE/docker-state holds "absent", "stopped" or "running"; docker-events holds the actions
# docker recorded for the container (one per line); the file oom-killed marks a container the kernel killed for memory.
echo "docker $*" >> "$FAKE_STATE/calls"
state="$(cat "$FAKE_STATE/docker-state")"
case "$1" in
    inspect)
        [ "$state" = absent ] && exit 1
        case "$*" in
            *State.Running*) if [ "$state" = running ]; then echo true; else echo false; fi ;;
            *State.OOMKilled*) if [ -e "$FAKE_STATE/oom-killed" ]; then echo true; else echo false; fi ;;
            *State.FinishedAt*) date -u +%FT%T.000000000Z ;;
            *.Id*) echo stand-in-container-id ;;
        esac ;;
    events) cat "$FAKE_STATE/docker-events" 2> /dev/null ;;
    rename) echo absent > "$FAKE_STATE/docker-state" ;;
    cp) mkdir -p "$3" && echo "synapse log" > "$3/dfa_log.txt" ;;
    stop)  # the file slow-stop holds how long the container takes to end; it counts as running until then
        [ -e "$FAKE_STATE/slow-stop" ] && sleep "$(cat "$FAKE_STATE/slow-stop")"
        echo stopped > "$FAKE_STATE/docker-state" ;;
esac
exit 0
"""

FAKE_SYSTEMCTL = """#!/bin/bash
# systemctl stand-in for the docker service: "is-active" and the time the service started ("show", unix form).
case "$1" in
    is-active) if [ -e "$FAKE_STATE/docker-going-down" ]; then echo deactivating; else echo active; fi ;;
    show) if [ -e "$FAKE_STATE/docker-restarted" ]; then echo "@$(( $(date +%s) + 3600 ))"; else echo "@1000000000"; fi ;;
esac
"""

FAKE_SERVE = f"""#!/bin/bash
# serve_champion.sh stand-in: "--check" approves; a start records whether a quad holder was still alive, then serves.
if [ "${{2:-}}" = --check ]; then echo "CHAMPION_CHECK_OK role=$1"; exit 0; fi
run="$QUEUE_DIR/serve-{CUT}-$1"
if [ -e "$FAKE_STATE/holder-alive" ]; then echo holder=alive; else echo holder=gone; fi >> "$FAKE_STATE/serve-saw"
mkdir -p "$run"
if [ -e "$FAKE_STATE/serve-fails" ]; then echo "2026-01-01 00:00:00 EST BOOT_FAILED stand-in" >> "$run/status"; exit 1; fi
echo "2026-01-01 00:00:00 EST BOOTED serve role=$1 up rc=0 boot_s=1 running=true" >> "$run/status"
echo "2026-01-01 00:00:00 EST SERVING name=stand-in" >> "$run/status"
echo running > "$FAKE_STATE/docker-state"
# Like the real one, it can leave a holder process running after it returns.
if [ -e "$FAKE_STATE/serve-leaves-process" ]; then setsid nohup sleep 20 > /dev/null 2>&1 & fi
"""

FAKE_JOURNALCTL = """#!/bin/bash
# journalctl stand-in: two kernel lines, one of them from the card driver.
echo "Jan 01 00:00:00 host kernel: habanalabs 0000:cd:00.0: Going to reset device"
echo "Jan 01 00:00:00 host kernel: unrelated line"
"""

FAKE_HOLDER = """#!/bin/bash
# quad_hold.sh stand-in: alive until the release file appears, then removes it and ends.
release="$1"; alive="$2"
touch "$alive"
while [ ! -f "$release" ]; do sleep 0.2; done
rm -f "$alive" "$release"
"""


def write_script(path: Path, text: str) -> Path:
    path.write_text(text)
    path.chmod(0o755)
    return path


class ChampionBoot(unittest.TestCase):
    def setUp(self) -> None:
        self.root = Path(tempfile.mkdtemp(prefix="champion-boot-test."))
        self.addCleanup(shutil.rmtree, self.root, ignore_errors=True)
        self.kit = self.root / "kit"
        self.queue = self.root / "queue"
        self.state = self.root / "state"
        self.bin = self.root / "bin"
        for folder in (self.kit, self.queue, self.state, self.bin):
            folder.mkdir()
        shutil.copy(SCRIPT_UNDER_TEST, self.kit / "champion_boot.sh")
        write_script(self.kit / "serve_champion.sh", FAKE_SERVE)
        (self.kit / "champion.staged").write_text(f"CHAMPION_CUT={CUT}\nCHAMPION_FINAL_ENV={self.root}/final.env\n")
        write_script(self.bin / "docker", FAKE_DOCKER)
        write_script(self.bin / "journalctl", FAKE_JOURNALCTL)
        write_script(self.bin / "systemctl", FAKE_SYSTEMCTL)
        self.wanted = self.queue / f"champion-{ROLE}.wanted"
        self.holder_script = write_script(self.bin / "holder.sh", FAKE_HOLDER)
        self.run_folder = self.queue / RUN_FOLDER
        self.holder_alive = self.state / "holder-alive"

    def docker_state(self, state: str) -> None:
        (self.state / "docker-state").write_text(state + "\n")

    def old_run(self) -> None:
        """A run folder left by the replica that served before."""
        self.run_folder.mkdir()
        (self.run_folder / "status").write_text("2026-01-01 00:00:00 EST SERVING name=earlier\n")

    def start_holder(self) -> subprocess.Popen:
        """A quad holder that waits on the old run folder's release file, as the serving replica's holder does."""
        holder = subprocess.Popen(["bash", str(self.holder_script), str(self.run_folder / "release"), str(self.holder_alive)])
        self.addCleanup(holder.wait, 5)
        self.addCleanup(holder.kill)
        deadline = time.monotonic() + HOLDER_START_TIMEOUT_S
        while not self.holder_alive.exists():
            self.assertLess(time.monotonic(), deadline, "the stand-in holder did not start")
            time.sleep(0.05)
        return holder

    def boot_env(self, holder_wait_s: int = 10, max_tries: int = 3) -> dict[str, str]:
        return dict(os.environ, PATH=f"{self.bin}:{os.environ['PATH']}", QUEUE_DIR=str(self.queue),
                    FAKE_STATE=str(self.state), HOLDER_WAIT_S=str(holder_wait_s), RECOVER_MAX_TRIES=str(max_tries))

    def boot(self, *args: str, holder_wait_s: int = 10, max_tries: int = 3) -> subprocess.CompletedProcess:
        return subprocess.run(["bash", str(self.kit / "champion_boot.sh"), ROLE, *args], env=self.boot_env(holder_wait_s, max_tries),
                              capture_output=True, text=True, timeout=60)

    def boot_in_background(self, *args: str) -> subprocess.Popen:
        return subprocess.Popen(["bash", str(self.kit / "champion_boot.sh"), ROLE, *args], env=self.boot_env(),
                                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    def serve_saw(self) -> str:
        seen = self.state / "serve-saw"
        return seen.read_text().strip() if seen.exists() else "never started"

    def moved_aside(self) -> list[Path]:
        return sorted(self.queue.glob(f"{RUN_FOLDER}.run-*"))

    def test_replica_that_died_has_its_holder_released_before_the_boot(self) -> None:
        self.docker_state("stopped")
        self.old_run()
        holder = self.start_holder()

        run = self.boot()

        self.assertEqual(run.returncode, 0, run.stdout + run.stderr)
        self.assertEqual(self.serve_saw(), "holder=gone", "the boot started while the dead replica's holder still held the quad")
        self.assertEqual(holder.wait(timeout=5), 0)
        self.assertIn("RELEASED", run.stdout)
        self.assertIn(" SERVING ", run.stdout)
        self.assertEqual(len(self.moved_aside()), 1)
        self.assertFalse((self.moved_aside()[0] / "release").exists(), "a release file was left in the old run folder")

    def test_start_with_no_holder_waits_briefly_then_boots(self) -> None:
        self.docker_state("stopped")
        self.old_run()

        started = time.monotonic()
        run = self.boot(holder_wait_s=1)

        self.assertEqual(run.returncode, 0, run.stdout + run.stderr)
        self.assertLess(time.monotonic() - started, 20)
        self.assertEqual(self.serve_saw(), "holder=gone")
        self.assertNotIn("RELEASED", run.stdout)
        self.assertFalse((self.moved_aside()[0] / "release").exists(), "the unused release file was left behind")

    def test_running_replica_keeps_its_holder(self) -> None:
        self.docker_state("running")
        self.old_run()
        holder = self.start_holder()

        run = self.boot()

        self.assertEqual(run.returncode, 0, run.stdout + run.stderr)
        self.assertIn("ALREADY_SERVING", run.stdout)
        self.assertIsNone(holder.poll(), "the holder of a running replica was released")
        self.assertFalse((self.run_folder / "release").exists())
        self.assertEqual(self.serve_saw(), "never started")
        (self.run_folder / "release").touch()  # let the stand-in holder end

    def test_first_start_with_no_earlier_run(self) -> None:
        self.docker_state("absent")

        run = self.boot()

        self.assertEqual(run.returncode, 0, run.stdout + run.stderr)
        self.assertEqual(self.serve_saw(), "holder=gone")
        self.assertEqual(self.moved_aside(), [])

    def test_plan_names_the_holder_release_and_changes_nothing(self) -> None:
        self.docker_state("stopped")
        self.old_run()
        holder = self.start_holder()

        run = self.boot("--plan")

        self.assertEqual(run.returncode, 0, run.stdout + run.stderr)
        self.assertIn(f"PLAN touch {self.run_folder}/release", run.stdout)
        self.assertIsNone(holder.poll(), "--plan released the holder")
        self.assertEqual(self.moved_aside(), [])
        self.assertEqual(self.serve_saw(), "never started")
        (self.run_folder / "release").touch()  # let the stand-in holder end


class Recover(ChampionBoot):
    """champion_boot.sh ROLE --recover: what the restart timer runs every minute."""

    def docker_recorded(self, *actions: str) -> None:
        """What docker's event history holds for the container."""
        (self.state / "docker-events").write_text("".join(action + "\n" for action in actions))

    def died_on_its_own(self) -> None:
        self.docker_state("stopped")
        self.docker_recorded("die")

    def served_then_stopped(self) -> None:
        self.died_on_its_own()
        self.old_run()
        self.wanted.touch()

    def served_then_stopped_through_docker(self) -> None:
        """What "docker stop" leaves: docker recorded the signal it sent, then the exit."""
        self.docker_state("stopped")
        self.docker_recorded("kill", "die", "stop")
        self.old_run()
        self.wanted.touch()

    def serve_starts(self) -> int:
        seen = self.state / "serve-saw"
        return len(seen.read_text().splitlines()) if seen.exists() else 0

    def test_replica_stopped_through_docker_stays_down(self) -> None:
        self.served_then_stopped_through_docker()

        first = self.boot("--recover", holder_wait_s=1)
        later = self.boot("--recover", holder_wait_s=1)

        self.assertEqual(first.returncode, 0, first.stdout + first.stderr)
        self.assertIn("STOPPED_BY_HAND", first.stdout)
        self.assertEqual(self.serve_starts(), 0, "a replica stopped with docker stop was started again")
        self.assertFalse(self.wanted.exists())
        self.assertEqual(later.stdout, "")

    def test_stop_through_docker_frees_the_quad_for_another_container(self) -> None:
        self.served_then_stopped_through_docker()
        holder = self.start_holder()

        self.boot("--recover", holder_wait_s=1)

        self.assertEqual(holder.wait(timeout=5), 0, "the stopped replica's holder still holds the quad")

    def test_replica_removed_through_docker_stays_down(self) -> None:
        self.docker_state("absent")
        self.old_run()
        self.wanted.touch()

        run = self.boot("--recover", holder_wait_s=1)

        self.assertIn("STOPPED_BY_HAND", run.stdout)
        self.assertEqual(self.serve_starts(), 0)

    def test_replica_taken_down_by_its_memory_guard_is_restarted(self) -> None:
        self.docker_state("absent")
        self.docker_recorded("kill", "die", "stop", "destroy")
        self.old_run()
        self.wanted.touch()
        guard_folder = self.run_folder / "graded" / f"champion-{ROLE}"
        guard_folder.mkdir(parents=True)
        (guard_folder / "ABORT").write_text("node under the floor\n")

        run = self.boot("--recover", holder_wait_s=1)

        self.assertIn("RECOVER ", run.stdout)
        self.assertIn(" SERVING ", run.stdout)

    def test_replica_killed_for_memory_is_restarted(self) -> None:
        self.docker_state("stopped")
        (self.state / "oom-killed").touch()
        self.old_run()
        self.wanted.touch()

        run = self.boot("--recover", holder_wait_s=1)

        self.assertIn("RECOVER ", run.stdout)
        self.assertEqual(self.serve_starts(), 1)

    def test_stop_docker_has_no_record_of_is_left_down_and_said_once(self) -> None:
        self.docker_state("stopped")
        self.old_run()
        self.wanted.touch()

        first = self.boot("--recover", holder_wait_s=1)
        later = self.boot("--recover", holder_wait_s=1)

        self.assertIn("RECOVER_UNSURE", first.stdout)
        self.assertEqual(self.serve_starts(), 0)
        self.assertEqual(later.stdout, "")

    def test_replica_stopped_by_a_docker_restart_is_restarted(self) -> None:
        self.docker_state("stopped")
        (self.state / "docker-restarted").touch()
        self.old_run()
        self.wanted.touch()

        run = self.boot("--recover", holder_wait_s=1)

        self.assertIn("RECOVER ", run.stdout)
        self.assertEqual(self.serve_starts(), 1)

    def test_nothing_is_decided_while_docker_is_going_down(self) -> None:
        self.served_then_stopped_through_docker()
        (self.state / "docker-going-down").touch()

        run = self.boot("--recover", holder_wait_s=1)

        self.assertEqual(run.stdout, "")
        self.assertTrue(self.wanted.exists(), "a stop made by docker shutting down was taken for a stop by hand")
        self.assertEqual(self.serve_starts(), 0)

    def test_failed_restart_is_tried_again_even_though_the_container_is_gone(self) -> None:
        self.served_then_stopped()
        (self.state / "serve-fails").touch()

        self.boot("--recover", holder_wait_s=1)
        second = self.boot("--recover", holder_wait_s=1)

        self.assertIn("RECOVER ", second.stdout)
        self.assertEqual(self.serve_starts(), 2)

    def test_replica_that_stopped_on_its_own_is_restarted(self) -> None:
        self.served_then_stopped()

        run = self.boot("--recover", holder_wait_s=1)

        self.assertEqual(run.returncode, 0, run.stdout + run.stderr)
        self.assertIn("RECOVER ", run.stdout)
        self.assertIn(" SERVING ", run.stdout)
        self.assertEqual(self.serve_starts(), 1)
        self.assertTrue(self.wanted.exists())

    def test_restart_keeps_what_the_stopped_replica_left(self) -> None:
        self.served_then_stopped()

        self.boot("--recover", holder_wait_s=1)

        evidence = self.moved_aside()[0] / "death-evidence"
        self.assertTrue((evidence / "habana_logs" / "dfa_log.txt").exists(), "the Synapse logs were not copied out")
        kernel_lines = (evidence / "kernel-habanalabs.txt").read_text()
        self.assertIn("Going to reset device", kernel_lines)
        self.assertNotIn("unrelated line", kernel_lines)
        self.assertTrue((evidence / "container-state.txt").exists())

    def test_running_replica_is_left_alone_without_a_word(self) -> None:
        self.docker_state("running")
        self.old_run()

        run = self.boot("--recover")

        self.assertEqual(run.returncode, 0, run.stdout + run.stderr)
        self.assertEqual(run.stdout, "")
        self.assertEqual(self.serve_starts(), 0)
        self.assertTrue(self.wanted.exists(), "a replica found serving was not marked as wanted")

    def test_replica_stopped_on_purpose_stays_down(self) -> None:
        self.docker_state("running")
        self.old_run()
        self.wanted.touch()

        stop = self.boot("--stop")
        run = self.boot("--recover")

        self.assertEqual(stop.returncode, 0, stop.stdout + stop.stderr)
        self.assertFalse(self.wanted.exists(), "--stop left the replica marked as wanted")
        self.assertEqual(run.stdout, "")
        self.assertEqual(self.serve_starts(), 0)

    def test_timer_run_during_a_slow_stop_on_purpose_does_not_bring_the_replica_back(self) -> None:
        """2026-11-13 23:45 ET: --stop took 32 s to end the replica. The timer ran meanwhile, found it still running and
        marked it wanted again; the next run found only "die" (docker keeps 256 events, about 100 s on this host, so
        the "kill" record was gone) and restarted it under the gate that had stopped it."""
        self.docker_state("running")
        self.old_run()
        self.wanted.touch()
        (self.state / "slow-stop").write_text("2\n")

        stop = self.boot_in_background("--stop")
        time.sleep(0.7)
        during = self.boot("--recover")
        self.assertEqual(stop.wait(timeout=30), 0)
        self.docker_recorded("die")
        after = self.boot("--recover", holder_wait_s=1)

        self.assertEqual(during.stdout, "")
        self.assertEqual(after.stdout, "")
        self.assertEqual(self.serve_starts(), 0, "a replica stopped on purpose was started again")
        self.assertFalse(self.wanted.exists(), "the timer marked a replica that was being stopped on purpose as wanted")

    def test_replica_started_by_hand_after_a_stop_on_purpose_is_watched_again(self) -> None:
        self.docker_state("running")
        self.old_run()
        self.wanted.touch()

        self.boot("--stop")
        by_hand = self.boot(holder_wait_s=1)
        self.died_on_its_own()
        again = self.boot("--recover", holder_wait_s=1)

        self.assertIn(" SERVING ", by_hand.stdout)
        self.assertIn("RECOVER ", again.stdout, "a replica started by hand after --stop was not restarted when it died")
        self.assertEqual(self.serve_starts(), 2)

    def test_replica_that_never_served_is_not_started(self) -> None:
        self.docker_state("absent")

        run = self.boot("--recover")

        self.assertEqual(run.returncode, 0, run.stdout + run.stderr)
        self.assertEqual(run.stdout, "")
        self.assertEqual(self.serve_starts(), 0)

    def test_start_by_hand_marks_the_replica_wanted(self) -> None:
        self.docker_state("absent")

        run = self.boot()

        self.assertEqual(run.returncode, 0, run.stdout + run.stderr)
        self.assertTrue(self.wanted.exists())

    def test_restart_gives_up_after_repeated_failures_and_says_so_once(self) -> None:
        self.served_then_stopped()
        (self.state / "serve-fails").touch()

        attempts = [self.boot("--recover", holder_wait_s=1, max_tries=2) for _ in range(2)]
        halted = self.boot("--recover", holder_wait_s=1, max_tries=2)
        quiet = self.boot("--recover", holder_wait_s=1, max_tries=2)

        for attempt in attempts:
            self.assertEqual(attempt.returncode, 0, attempt.stdout + attempt.stderr)
            self.assertIn("RECOVER ", attempt.stdout)
        self.assertEqual(self.serve_starts(), 2)
        self.assertIn("RECOVER_HALT", halted.stdout)
        self.assertEqual(halted.returncode, 0)
        self.assertEqual(quiet.stdout, "")

    def test_start_by_hand_clears_the_give_up_state(self) -> None:
        self.served_then_stopped()
        (self.state / "serve-fails").touch()
        for _ in range(3):
            self.boot("--recover", holder_wait_s=1, max_tries=2)
        (self.state / "serve-fails").unlink()

        by_hand = self.boot(holder_wait_s=1)
        self.died_on_its_own()
        again = self.boot("--recover", holder_wait_s=1, max_tries=2)

        self.assertIn(" SERVING ", by_hand.stdout)
        self.assertIn("RECOVER ", again.stdout)
        self.assertIn(" SERVING ", again.stdout)

    def test_processes_a_boot_leaves_behind_do_not_keep_the_start_lock(self) -> None:
        self.docker_state("absent")
        (self.state / "serve-leaves-process").touch()

        first = self.boot()
        self.died_on_its_own()
        restart = self.boot("--recover", holder_wait_s=1)

        self.assertIn(" SERVING ", first.stdout)
        self.assertIn("RECOVER ", restart.stdout, "the first boot's left-behind process still held the start lock")
        self.assertEqual(self.serve_starts(), 2)

    def test_second_start_while_one_is_running_is_left_alone(self) -> None:
        self.served_then_stopped()
        lock = self.queue / f"champion-boot-{ROLE}.lock"
        other_start = subprocess.Popen(["flock", str(lock), "sleep", "30"])
        self.addCleanup(other_start.wait, 5)
        self.addCleanup(other_start.kill)
        time.sleep(0.5)

        by_hand = self.boot(holder_wait_s=1)
        by_timer = self.boot("--recover", holder_wait_s=1)

        self.assertEqual(by_hand.returncode, 0, by_hand.stdout + by_hand.stderr)
        self.assertIn("BOOT_IN_PROGRESS", by_hand.stdout)
        self.assertEqual(by_timer.stdout, "")
        self.assertEqual(self.serve_starts(), 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
