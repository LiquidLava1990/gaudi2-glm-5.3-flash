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
IMAGE = "sha256:" + "ab12" * 16           # the staged image
OTHER_IMAGE = "sha256:" + "cd34" * 16     # another image's record must never seed this one

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
            *Config.Env*) cat "$FAKE_STATE/container-env" 2> /dev/null ;;
            *'{{.Image}}'*) cat "$FAKE_STATE/container-image" 2> /dev/null ;;
        esac ;;
    top) echo PID; cat "$FAKE_STATE/container-pids" 2> /dev/null ;;
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
echo "${{LAB_RECIPE_SEED_LEAF:-none}}" >> "$FAKE_STATE/serve-seed"
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

FAKE_TASKSET = """#!/bin/bash
# taskset stand-in: records each pin; the file taskset-fails makes every pin fail.
echo "taskset $*" >> "$FAKE_STATE/calls"
[ ! -e "$FAKE_STATE/taskset-fails" ]
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
        shutil.copy(SCRIPT_UNDER_TEST.parent / "common.sh", self.kit / "common.sh")
        write_script(self.kit / "serve_champion.sh", FAKE_SERVE)
        (self.kit / "champion.staged").write_text(f"CHAMPION_CUT={CUT}\nCHAMPION_FINAL_ENV={self.root}/final.env\n"
                                                  f"CHAMPION_IMAGE_SHA={IMAGE}\n")
        self.recipes = self.root / "recipe_cache"
        self.proc = self.root / "proc"
        write_script(self.bin / "docker", FAKE_DOCKER)
        write_script(self.bin / "taskset", FAKE_TASKSET)
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
        base = {key: value for key, value in os.environ.items() if key != "LAB_RECIPE_SEED_LEAF"}
        return dict(base, PATH=f"{self.bin}:{os.environ['PATH']}", QUEUE_DIR=str(self.queue),
                    FAKE_STATE=str(self.state), HOLDER_WAIT_S=str(holder_wait_s), RECOVER_MAX_TRIES=str(max_tries),
                    RECIPE_ROOT=str(self.recipes), GLM53_PROMOTE_TEST="1", PROC_ROOT=str(self.proc), SUDO="")

    def boot(self, *args: str, holder_wait_s: int = 10, max_tries: int = 3,
             extra_env: dict[str, str] | None = None) -> subprocess.CompletedProcess:
        env = dict(self.boot_env(holder_wait_s, max_tries), **(extra_env or {}))
        return subprocess.run(["bash", str(self.kit / "champion_boot.sh"), ROLE, *args], env=env,
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


class SeededCache(ChampionBoot):
    """The recipe leaf a serving boot used seeds the next boot of the same image (recipe-seed-<sha12>.leaf in the queue)."""

    def leaf(self, name: str, files: int = 1) -> Path:
        leaf = self.recipes / name
        leaf.mkdir(parents=True)
        for index in range(files):
            (leaf / f"graph{index}.recipe").write_text("recipe\n")
        return leaf

    def record(self, image: str, leaf: Path) -> Path:
        record = self.queue / f"recipe-seed-{image.removeprefix('sha256:')[:12]}.leaf"
        record.write_text(f"{leaf}\n")
        return record

    def booted_container_uses(self, leaf_name: str, image: str = IMAGE) -> None:
        """What docker inspect shows for the container the stand-in serve started."""
        (self.state / "container-env").write_text(f"PATH=/usr/bin\nPT_HPU_RECIPE_CACHE_CONFIG=/recipe_cache/{leaf_name},false,8192\n")
        (self.state / "container-image").write_text(image + "\n")

    def seed_seen(self) -> str:
        return (self.state / "serve-seed").read_text().strip()

    def test_recorded_leaf_of_the_staged_image_seeds_the_boot(self) -> None:
        self.docker_state("absent")
        leaf = self.leaf("earlier-r1-leaf")
        self.record(IMAGE, leaf)

        run = self.boot()

        self.assertEqual(run.returncode, 0, run.stdout + run.stderr)
        self.assertEqual(self.seed_seen(), str(leaf))
        self.assertIn(f"SEED {leaf}", run.stdout)

    def test_recorded_leaf_that_is_gone_boots_unseeded(self) -> None:
        self.docker_state("absent")
        self.record(IMAGE, self.recipes / "removed-leaf")

        run = self.boot()

        self.assertEqual(self.seed_seen(), "none")
        self.assertIn("SEED none (the leaf", run.stdout)

    def test_recorded_leaf_that_is_empty_boots_unseeded(self) -> None:
        self.docker_state("absent")
        self.record(IMAGE, self.leaf("empty-leaf", files=0))

        run = self.boot()

        self.assertEqual(self.seed_seen(), "none")
        self.assertIn("is gone or empty", run.stdout)

    def test_leaf_recorded_for_another_image_never_seeds(self) -> None:
        self.docker_state("absent")
        self.record(OTHER_IMAGE, self.leaf("other-image-leaf"))

        run = self.boot()

        self.assertEqual(self.seed_seen(), "none")
        self.assertIn(f"SEED none (no recorded leaf for image {IMAGE.removeprefix('sha256:')[:12]})", run.stdout)

    def test_seed_leaf_from_the_environment_wins_over_the_record(self) -> None:
        self.docker_state("absent")
        self.record(IMAGE, self.leaf("recorded-leaf"))
        chosen = str(self.recipes / "chosen-by-hand")

        run = self.boot(extra_env={"LAB_RECIPE_SEED_LEAF": chosen})

        self.assertEqual(self.seed_seen(), chosen)
        self.assertIn(f"SEED {chosen} (LAB_RECIPE_SEED_LEAF from the environment)", run.stdout)

    def test_boot_that_serves_records_its_leaf_for_its_image(self) -> None:
        self.docker_state("absent")
        self.record(IMAGE, self.leaf("older-leaf"))
        new_leaf = self.leaf("claude-flash-launch-glm53flash-r0-tcut-20261122T200000Z")
        self.booted_container_uses(new_leaf.name)

        run = self.boot()

        self.assertEqual(run.returncode, 0, run.stdout + run.stderr)
        record = self.queue / f"recipe-seed-{IMAGE.removeprefix('sha256:')[:12]}.leaf"
        self.assertEqual(record.read_text(), f"{new_leaf}\n")
        self.assertEqual(sorted(path.name for path in self.queue.glob("recipe-seed-*.leaf*")), [record.name], "a temp file was left")
        self.assertIn(f"SEED_RECORDED {new_leaf}", run.stdout)

    def test_boot_that_fails_records_nothing(self) -> None:
        self.docker_state("absent")
        self.booted_container_uses(self.leaf("failed-boot-leaf").name)
        (self.state / "serve-fails").touch()

        run = self.boot()

        self.assertEqual(run.returncode, 1, run.stdout + run.stderr)
        self.assertEqual(list(self.queue.glob("recipe-seed-*")), [])
        self.assertNotIn("SEED_RECORDED", run.stdout)

    def test_boot_whose_leaf_is_empty_keeps_the_earlier_record(self) -> None:
        self.docker_state("absent")
        older = self.leaf("older-leaf")
        record = self.record(IMAGE, older)
        self.booted_container_uses(self.leaf("empty-new-leaf", files=0).name)

        run = self.boot()

        self.assertEqual(record.read_text(), f"{older}\n")
        self.assertIn("SEED_RECORD_SKIPPED", run.stdout)

    def test_plan_names_the_seed_and_records_nothing(self) -> None:
        self.docker_state("absent")
        leaf = self.leaf("recorded-leaf")
        self.record(IMAGE, leaf)

        run = self.boot("--plan")

        self.assertIn(f"SEED {leaf}", run.stdout)
        self.assertEqual(self.serve_saw(), "never started")

    # Durable seed: a serving boot keeps one protected copy of its leaf per image (recipe_cache/seed-<sha12>, marked in the
    # queue by recipe-seed-<sha12>.copy). lab.py down of the serving name removes the leaf; the copy still seeds the next boot.

    @staticmethod
    def sha12(image: str) -> str:
        return image.removeprefix("sha256:")[:12]

    def seed_dir(self, image: str = IMAGE) -> Path:
        return self.recipes / f"seed-{self.sha12(image)}"

    def mark(self, image: str = IMAGE) -> Path:
        return self.queue / f"recipe-seed-{self.sha12(image)}.copy"

    def our_seed(self, image: str, files: dict[str, str], age_s: int = 0) -> Path:
        """A copy this kit made earlier: the seed folder and its mark (line 1 the image, line 2 the folder)."""
        seed = self.seed_dir(image)
        seed.mkdir(parents=True)
        for name, text in files.items():
            (seed / name).write_text(text)
        self.mark(image).write_text(f"{image}\n{seed}\n")
        stamp = time.time() - age_s
        os.utime(self.mark(image), (stamp, stamp))
        return seed

    def leftovers(self) -> list[str]:
        return sorted(path.name for path in self.recipes.glob("seed-*.*"))

    def test_boot_that_serves_keeps_a_protected_copy_of_its_leaf(self) -> None:
        self.docker_state("absent")
        leaf = self.leaf("claude-flash-launch-glm53flash-r0-tcut-20261122T210000Z", files=3)
        self.booted_container_uses(leaf.name)

        run = self.boot()

        self.assertEqual(run.returncode, 0, run.stdout + run.stderr)
        seed = self.seed_dir()
        self.assertEqual(sorted(path.name for path in seed.iterdir()), sorted(path.name for path in leaf.iterdir()))
        self.assertEqual(self.mark().read_text(), f"{IMAGE}\n{seed}\n")
        self.assertEqual(self.leftovers(), [], "a temp copy was left")
        self.assertIn(f"SEED_COPIED {leaf} -> {seed}", run.stdout)

    def test_the_protected_copy_seeds_the_boot_once_lab_down_removed_the_recorded_leaf(self) -> None:
        self.docker_state("absent")
        self.record(IMAGE, self.recipes / "claude-flash-launch-glm53flash-r0-tcut-removed-by-lab-down")
        seed = self.our_seed(IMAGE, {"graph0.recipe": "recipe\n"})

        run = self.boot()

        self.assertEqual(run.returncode, 0, run.stdout + run.stderr)
        self.assertEqual(self.seed_seen(), str(seed))
        self.assertIn(f"SEED {seed} (the protected copy", run.stdout)

    def test_the_recorded_leaf_still_wins_over_the_copy_while_it_exists(self) -> None:
        self.docker_state("absent")
        leaf = self.leaf("live-leaf")
        self.record(IMAGE, leaf)
        self.our_seed(IMAGE, {"graph0.recipe": "recipe\n"})

        self.boot()

        self.assertEqual(self.seed_seen(), str(leaf))

    def test_a_copy_for_another_image_never_seeds(self) -> None:
        self.docker_state("absent")
        self.our_seed(OTHER_IMAGE, {"graph0.recipe": "recipe\n"})

        run = self.boot()

        self.assertEqual(self.seed_seen(), "none")
        self.assertIn("SEED none (no recorded leaf", run.stdout)

    def test_an_unmarked_seed_folder_is_never_used_replaced_or_pruned(self) -> None:
        self.docker_state("absent")
        foreign = self.seed_dir()
        foreign.mkdir(parents=True)
        (foreign / "foreign.recipe").write_text("not ours\n")
        self.booted_container_uses(self.leaf("new-leaf").name)

        run = self.boot()

        self.assertEqual(self.seed_seen(), "none")
        self.assertEqual([path.name for path in foreign.iterdir()], ["foreign.recipe"])
        self.assertFalse(self.mark().exists())
        self.assertIn("SEED_COPY_SKIPPED", run.stdout)

    def test_a_refresh_swaps_the_whole_copy_and_leaves_no_temp_folder(self) -> None:
        self.docker_state("absent")
        seed = self.our_seed(IMAGE, {"stale.recipe": "old\n"}, age_s=3600)
        self.booted_container_uses(self.leaf("new-leaf", files=2).name)

        run = self.boot()

        self.assertEqual(run.returncode, 0, run.stdout + run.stderr)
        self.assertEqual(sorted(path.name for path in seed.iterdir()), ["graph0.recipe", "graph1.recipe"])
        self.assertEqual(self.leftovers(), [])
        self.assertGreater(self.mark().stat().st_mtime, time.time() - 600, "the mark was not refreshed")

    def test_only_the_two_newest_copies_stay_and_nothing_unmarked_is_touched(self) -> None:
        self.docker_state("absent")
        images = ["sha256:" + digit * 64 for digit in "123"]
        for age, image in zip((300, 200, 100), images):
            self.our_seed(image, {"graph0.recipe": "recipe\n"}, age_s=age)
        unmarked = self.recipes / f"seed-{'4' * 12}"
        unmarked.mkdir()
        self.booted_container_uses(self.leaf("claude-flash-launch-glm53flash-r0-tcut-20261122T220000Z").name)

        run = self.boot()

        self.assertEqual(run.returncode, 0, run.stdout + run.stderr)
        kept = sorted(path.name for path in self.recipes.glob("seed-*"))
        self.assertEqual(kept, sorted([self.seed_dir().name, self.seed_dir(images[2]).name, unmarked.name]))
        self.assertEqual(sorted(path.name for path in self.queue.glob("recipe-seed-*.copy")),
                         sorted([self.mark().name, self.mark(images[2]).name]))
        self.assertTrue((self.recipes / "claude-flash-launch-glm53flash-r0-tcut-20261122T220000Z").is_dir())
        self.assertIn(f"SEED_PRUNED {self.seed_dir(images[0])}", run.stdout)

    def test_a_mark_that_names_another_folder_is_left_alone(self) -> None:
        self.docker_state("absent")
        elsewhere = self.recipes / "claude-flash-launch-glm53flash-r1-tcut-20261101T000000Z"
        elsewhere.mkdir(parents=True)
        for age, digit in ((300, "1"), (200, "2")):
            self.our_seed("sha256:" + digit * 64, {"graph0.recipe": "recipe\n"}, age_s=age)
        self.mark("sha256:" + "1" * 64).write_text(f"sha256:{'1' * 64}\n{elsewhere}\n")
        self.booted_container_uses(self.leaf("new-leaf").name)

        run = self.boot()

        self.assertTrue(elsewhere.is_dir())
        self.assertIn("SEED_PRUNE_SKIPPED", run.stdout)

    def test_a_copy_that_fails_leaves_no_partial_copy_and_the_boot_still_serves(self) -> None:
        self.docker_state("absent")
        leaf = self.leaf("new-leaf", files=2)
        (leaf / "graph1.recipe").chmod(0)
        self.booted_container_uses(leaf.name)

        run = self.boot()

        self.assertEqual(run.returncode, 0, run.stdout + run.stderr)
        self.assertFalse(self.seed_dir().exists())
        self.assertFalse(self.mark().exists())
        self.assertEqual(self.leftovers(), [])
        self.assertIn("SEED_COPY_FAILED", run.stdout)
        self.assertTrue(self.wanted.exists())

    def test_a_temp_copy_left_by_a_killed_boot_is_removed_and_a_live_ones_kept(self) -> None:
        self.docker_state("absent")
        dead = self.recipes / f"seed-{self.sha12(IMAGE)}.new-999999999"   # above pid_max: no such process
        live = self.recipes / f"seed-{self.sha12(IMAGE)}.new-{os.getpid()}"
        for temp in (dead, live):
            temp.mkdir(parents=True)
            (temp / "graph0.recipe").write_text("half a copy\n")
        self.booted_container_uses(self.leaf("new-leaf").name)

        run = self.boot()

        self.assertEqual(run.returncode, 0, run.stdout + run.stderr)
        self.assertFalse(dead.exists())
        self.assertTrue(live.is_dir())
        self.assertTrue(self.seed_dir().is_dir())

    def test_keep_seed_mode_copies_a_running_replicas_leaf_without_starting_anything(self) -> None:
        self.docker_state("running")
        leaf = self.leaf("adopted-leaf", files=2)
        self.booted_container_uses(leaf.name)

        run = self.boot("--keep-seed")

        self.assertEqual(run.returncode, 0, run.stdout + run.stderr)
        self.assertEqual(sorted(path.name for path in self.seed_dir().iterdir()), ["graph0.recipe", "graph1.recipe"])
        self.assertIn(f"SEED_COPIED {leaf}", run.stdout)
        self.assertEqual(self.serve_saw(), "never started")

    def test_keep_seed_mode_on_a_stopped_replica_says_so_and_fails(self) -> None:
        self.docker_state("stopped")
        self.booted_container_uses(self.leaf("old-leaf").name)

        run = self.boot("--keep-seed")

        self.assertEqual(run.returncode, 1)
        self.assertIn("is not running; no leaf to keep", run.stdout)
        self.assertFalse(self.seed_dir().exists())

    def test_boot_that_fails_copies_nothing(self) -> None:
        self.docker_state("absent")
        self.booted_container_uses(self.leaf("failed-boot-leaf").name)
        (self.state / "serve-fails").touch()

        self.boot()

        self.assertFalse(self.seed_dir().exists())


class AccThreadPins(ChampionBoot):
    """After a boot that serves, the 4 AccThreadPool threads get one cpu each (r0: 50-53, ascending tid)."""

    def threads(self, accs: dict[int, list[int]], others: dict[int, list[int]] | None = None) -> None:
        """The container's processes (docker top) and their threads: accs are AccThreadPool, others are not."""
        pids = sorted(set(accs) | set(others or {}))
        (self.state / "container-pids").write_text("".join(f"{pid}\n" for pid in pids))
        for kind, table in (("AccThreadPool", accs), ("python3", others or {})):
            for pid, tids in table.items():
                for tid in tids:
                    task = self.proc / str(pid) / "task" / str(tid)
                    task.mkdir(parents=True)
                    (task / "comm").write_text(kind + "\n")

    def taskset_calls(self) -> list[str]:
        calls = self.state / "calls"
        return [line for line in calls.read_text().splitlines() if line.startswith("taskset")] if calls.exists() else []

    def test_boot_that_serves_pins_the_four_threads_in_tid_order(self) -> None:
        self.docker_state("absent")
        self.threads({300: [3005], 100: [1009], 200: [2001], 400: [900]}, others={100: [100, 1001]})

        run = self.boot()

        self.assertEqual(run.returncode, 0, run.stdout + run.stderr)
        self.assertEqual(self.taskset_calls(), ["taskset -pc 50 900", "taskset -pc 51 1009", "taskset -pc 52 2001", "taskset -pc 53 3005"])
        self.assertIn("ACC_PIN role=r0 tid=900 cpu=50", run.stdout)
        self.assertIn("ACC_PIN_DONE role=r0 container=claude-flash-launch-glm53flash-r0-tcut pinned=4/4 cpus=50-53", run.stdout)

    def test_three_threads_are_left_unpinned_and_the_boot_still_succeeds(self) -> None:
        self.docker_state("absent")
        self.threads({100: [101], 200: [201], 300: [301]})

        run = self.boot()

        self.assertEqual(run.returncode, 0, run.stdout + run.stderr)
        self.assertEqual(self.taskset_calls(), [])
        self.assertIn("ACC_PIN_WARN role=r0 found 3 AccThreadPool threads", run.stdout)
        self.assertIn("ACC_PIN_INCOMPLETE", run.stdout)
        self.assertTrue(self.wanted.exists())

    def test_failed_pins_never_fail_the_boot(self) -> None:
        self.docker_state("absent")
        self.threads({100: [101], 200: [201], 300: [301], 400: [401]})
        (self.state / "taskset-fails").touch()

        run = self.boot()

        self.assertEqual(run.returncode, 0, run.stdout + run.stderr)
        self.assertIn("pinned=0/4", run.stdout)
        self.assertIn("ACC_PIN_INCOMPLETE", run.stdout)

    def test_boot_that_fails_pins_nothing(self) -> None:
        self.docker_state("absent")
        self.threads({100: [101], 200: [201], 300: [301], 400: [401]})
        (self.state / "serve-fails").touch()

        self.boot()

        self.assertEqual(self.taskset_calls(), [])

    def test_pin_acc_mode_pins_a_running_replica_without_starting_anything(self) -> None:
        self.docker_state("running")
        self.threads({100: [101], 200: [201], 300: [301], 400: [401]})

        run = self.boot("--pin-acc")

        self.assertEqual(run.returncode, 0, run.stdout + run.stderr)
        self.assertEqual(len(self.taskset_calls()), 4)
        self.assertEqual(self.serve_saw(), "never started")

    def test_pin_acc_mode_on_a_stopped_replica_says_so_and_fails(self) -> None:
        self.docker_state("stopped")

        run = self.boot("--pin-acc")

        self.assertEqual(run.returncode, 1)
        self.assertIn("is not running; nothing to pin", run.stdout)
        self.assertEqual(self.taskset_calls(), [])


if __name__ == "__main__":
    unittest.main(verbosity=2)
