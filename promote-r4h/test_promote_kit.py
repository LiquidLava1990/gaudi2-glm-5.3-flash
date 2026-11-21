#!/usr/bin/env python3
"""CPU tests of promote_r4h.sh and rollback_r4f.sh (no card, no unit, no real container): docker, systemctl, curl, both launchers
and the smoke are fakes that log each call; RC56's drop-ins are copies of the live bytes. Run on the lab host (the rollback reads
serve-levers.env):  python3 -I test_promote_kit.py
"""
from __future__ import annotations

import hashlib
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

HERE = Path(__file__).resolve().parent
LIVE_UNITS = Path("/etc/systemd/system")
RC56_DROPIN = "zzzzzzz-claude-RC56.conf"
R4H_DROPIN = "zzzzzzzzzz-claude-c1d-r4h.conf"
R4F = "sha256:aabd010a1851eb65dee18e1bb486c3a24eaa2c57b593cdafe6470784609a81c8"
RC56 = "sha256:1481bb99fce076776c9e88604219726df33a903514015aa13fb0efc2350c8bbc"
CHAMPION = "sha256:" + "c" * 64
CUT = "r4h5"   # the champion is a parameter (promote_r4h.sh stage --image <tag>); r4h5 here
CHAMPION_TAG = f"claude-flash-dual-graphs-c1d:20261109-{CUT}"
LEVERS = "HPU_GLM53_CAPTURE_EXECUTE_KEYS=128:1024,128:8192\nHPU_GLM53_ATTN_PIN=0\n"
GATES = "R4E_REQUIRED_KEYS='VLLM_DECODE_PRIORITY HPU_GLM53_ATTN_PIN'\nR4E_PIN_GATE=on\n"   # the pin arm: LEVERS sets the pin
QUERY_R4H4 = "HPU_GLM53_QUERY_SHARD_DECODE_KEYS=4:1024"   # r4h4 serves the query shard on 4:1024 only (W4's decode keys refuse overlaps)
QUERY_R4H3 = "HPU_GLM53_QUERY_SHARD_DECODE_KEYS=4:1024,16:1024,4:4096,16:4096,4:8192,16:8192"

# docker: container <name> = $STATE/c/<name> "<image> <running>", its env $STATE/env-<name>, its log $STATE/log-<name>.
FAKE_DOCKER = r'''#!/bin/bash
echo "docker $*" >> "$STATE/calls"
c="$STATE/c"; last="${@: -1}"
case "$1" in
  image) case "$*" in *"{{.Id}}"*) [ "$last" = "$CHAMPION_TAG" ] && echo "$CHAMPION" ;; *) [ "$last" = "$R4F" ] ;; esac ;;
  ps) for f in "$c"/*; do read -r image running < "$f"; [ "$running" = true ] && basename "$f"; done; exit 0 ;;
  inspect)
    [ -f "$c/$last" ] || exit 1
    read -r image running < "$c/$last"
    case "$3" in *Image*) echo "$image" ;; *Running*) echo "$running" ;; *Env*) cat "$STATE/env-$last" 2>/dev/null ;;
                 *Args*) cat "$STATE/args-$last" 2>/dev/null; echo " null" ;; esac ;;
  rename) mv "$c/$2" "$c/$3" ;;
  stop) read -r image running < "$c/$last"; echo "$image false" > "$c/$last" ;;
  logs) cat "$STATE/log-$last" 2>/dev/null ;;
  *) exit 1 ;;
esac
'''
FAKE_SYSTEMCTL = r'''#!/bin/bash
echo "systemctl $*" >> "$STATE/calls"
[ "$1" = is-active ] && { [ -f "$STATE/active-${@: -1}" ]; exit; }
[ "$1" = stop ] && rm -f "$STATE"/active-*
exit 0
'''
# A launcher: $1 role. Fails (DONE) when $STATE/fail-<cut>-<role> exists; else creates its container (env = the levers file unless
# $STATE/drift-<role>), a worker log with the warmup line, and BOOTED + SERVING lines naming a release file.
# --check: the launcher's pre-boot config check only ("check <cut> <role>"); fails when $STATE/check-fail-<role> exists.
FAKE_SERVE = r'''#!/bin/bash
cut=__CUT__; image=__IMAGE__; role="$1"; name="claude-flash-launch-glm53flash-$role-$cut"; run="$QUEUE_DIR/serve-$cut-$role"
if [ "${2:-}" = --check ]; then
    echo "check $cut $role" >> "$STATE/calls"
    [ ! -f "$STATE/check-fail-$role" ] || { echo "HALT arm=champion-$role: VLLM_DECODE_PRIORITY is not set"; exit 1; }
    echo "CHAMPION_CHECK_OK role=$role"; exit 0
fi
case "$role" in r0) echo "--port 18274" > "$STATE/args-$name" ;; *) echo "--port 18275" > "$STATE/args-$name" ;; esac
echo "serve $cut $role" >> "$STATE/calls"
mkdir -p "$run"
if [ -f "$STATE/fail-$cut-$role" ]; then echo "x DONE verdict=BOOT_FAILED" >> "$run/status"; exit 1; fi
echo "$image true" > "$STATE/c/$name"
if [ -f "$STATE/drift-$role" ]; then echo "HPU_GLM53_ATTN_PIN=1" > "$STATE/env-$name"; else cp "$STATE/levers.env" "$STATE/env-$name"; fi
[ ! -f "$STATE/live-env-$role" ] || cp "$STATE/live-env-$role" "$STATE/env-$name"
echo "INFO Warmup finished in $(cat "$STATE/warmup" 2>/dev/null || echo 250) secs, allocated 5 GiB" > "$STATE/log-$name"
echo "x BOOTED serve role=$role up rc=0 boot_s=$(cat "$STATE/boot" 2>/dev/null || echo 1100) running=true" >> "$run/status"
echo "x SERVING name=$name port=1 holder=1; rollback: docker stop $name, then touch $run/release" >> "$run/status"
'''
# ss -H -ltn "sport = :<port>": the listener lines in $STATE/ss-<port>, if any
FAKE_SS = '#!/bin/bash\nport="${@: -1}"; port="${port##*:}"; cat "$STATE/ss-$port" 2>/dev/null; exit 0\n'
FAKE_CURL = '#!/bin/bash\necho "curl $*" >> "$STATE/calls"\n'
FAKE_SMOKE = 'print("SMOKE_OK")\n'


class Kit(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name)
        self.kit = root / "kit"
        shutil.copytree(HERE, self.kit, ignore=shutil.ignore_patterns("logs", "champion.staged", "__pycache__"))
        self.state, self.units, self.queue, bin_dir = root / "state", root / "units", root / "queue", root / "bin"
        for folder in (self.state / "c", self.queue, bin_dir, root / "holders"):
            folder.mkdir(parents=True)
        self.holders = root / "holders"
        (self.state / "levers.env").write_text(LEVERS)
        (self.state / "gates.env").write_text(GATES)
        for role in ("r0", "r1"):
            unit = self.units / f"glm53flash-{role}.service.d"
            unit.mkdir(parents=True)
            shutil.copy(LIVE_UNITS / f"glm53flash-{role}.service.d" / RC56_DROPIN, unit / RC56_DROPIN)
            self.container(f"claude-flash-launch-glm53flash-{role}-r4f", R4F, "true")
            (self.state / f"args-claude-flash-launch-glm53flash-{role}-r4f").write_text(f"--port {18274 if role == 'r0' else 18275}\n")
            self.container(f"glm53flash-{role}", RC56, "false")
            run = self.queue / f"serve-r4f-{role}"
            run.mkdir()
            (run / "status").write_text(f"x SERVING name=claude-flash-launch-glm53flash-{role}-r4f port=1 holder=1; rollback: "
                                        f"docker stop x, then touch {run}/release\n")
        self.fakes = {}
        fakes = (("docker", FAKE_DOCKER), ("systemctl", FAKE_SYSTEMCTL), ("curl", FAKE_CURL), ("smoke.py", FAKE_SMOKE), ("ss", FAKE_SS),
                 ("serve_r4f", FAKE_SERVE.replace("__CUT__", "r4f").replace("__IMAGE__", R4F)),
                 ("serve_champion", FAKE_SERVE.replace("__CUT__", CUT).replace("__IMAGE__", CHAMPION)))
        for name, text in fakes:
            path = bin_dir / name
            path.write_text(text)
            path.chmod(0o755)
            self.fakes[name] = str(path)

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def container(self, name: str, image: str, running: str) -> None:
        (self.state / "c" / name).write_text(f"{image} {running}\n")

    def image(self, name: str) -> str:
        return (self.state / "c" / name).read_text().split()

    def run_kit(self, script: str, *args: str) -> tuple[subprocess.CompletedProcess, list[str]]:
        env = dict(os.environ, GLM53_PROMOTE_TEST="1", STATE=str(self.state), R4F=R4F, CHAMPION=CHAMPION, CHAMPION_TAG=CHAMPION_TAG,
                   UNIT_DIR=str(self.units), SS=self.fakes["ss"],
                   SERVE_SCRIPT=self.fakes["serve_r4f"], CHAMPION_SERVE=self.fakes["serve_champion"],
                   SMOKE_SCRIPT=self.fakes["smoke.py"], QUEUE_DIR=str(self.queue), DOCKER=self.fakes["docker"], SUDO="",
                   SYSTEMCTL=self.fakes["systemctl"], CURL=self.fakes["curl"], POLL_S="0", HOLDER_DIR=str(self.holders),
                   STAGED=str(self.kit / "champion.staged"))
        run = subprocess.run(["bash", str(self.kit / script), *args], env=env, capture_output=True, text=True, timeout=120)
        log = sorted((self.kit / "logs").glob(f"{script.removesuffix('.sh')}-2*.log"))
        output = run.stdout + run.stderr + (log[-1].read_text() if log else "")
        calls = (self.state / "calls").read_text().splitlines() if (self.state / "calls").exists() else []
        return subprocess.CompletedProcess(run.args, run.returncode, output, ""), calls

    def stage(self) -> subprocess.CompletedProcess:
        run, _ = self.run_kit("promote_r4h.sh", "stage", "--image", CHAMPION_TAG, "--image-sha", CHAMPION,
                              "--levers", str(self.state / "levers.env"), "--gates", str(self.state / "gates.env"))
        return run


class Promote(Kit):
    def test_stage_pins_the_image_and_the_levers_file(self) -> None:
        run = self.stage()
        self.assertEqual(run.returncode, 0, run.stdout)
        staged = (self.kit / "champion.staged").read_text()
        self.assertIn(f"CHAMPION_IMAGE_SHA={CHAMPION}\n", staged)
        self.assertIn(f"CHAMPION_CUT={CUT}\n", staged)
        self.assertIn(f"CHAMPION_IMAGE={CHAMPION_TAG}\n", staged)
        self.assertIn("CHAMPION_LEVERS_SHA=", staged)
        self.assertIn(f"CHAMPION_GATES={self.state / 'gates.env'}\n", staged)
        self.assertIn("CHAMPION_GATES_SHA=", staged)

    def test_stage_refuses_without_a_gates_file(self) -> None:
        run, _ = self.run_kit("promote_r4h.sh", "stage", "--image", CHAMPION_TAG, "--image-sha", CHAMPION,
                              "--levers", str(self.state / "levers.env"))
        self.assertEqual(run.returncode, 2, run.stdout)
        self.assertIn("--gates", run.stdout)
        self.assertFalse((self.kit / "champion.staged").exists())

    def test_gates_and_levers_that_disagree_on_the_pin_refuse_dry_run_and_promote_before_any_change(self) -> None:
        cases = (("R4E_PIN_GATE=on\n", "HPU_GLM53_CAPTURE_EXECUTE_KEYS=128:1024\n", "R4E_PIN_GATE=on but the levers leave HPU_GLM53_ATTN_PIN unset"),
                 ("R4E_PIN_GATE=off\n", LEVERS, "R4E_PIN_GATE=off but the levers set HPU_GLM53_ATTN_PIN"),
                 ("R4E_ROUTED_GATE=off\n", LEVERS, "names no R4E_PIN_GATE"))
        for gates, levers, refusal in cases:
            with self.subTest(refusal=refusal):
                (self.state / "gates.env").write_text(gates)
                (self.state / "levers.env").write_text(levers)
                self.stage()
                for args in (("--dry-run",), ()):
                    run, calls = self.run_kit("promote_r4h.sh", *args)
                    self.assertEqual(run.returncode, 2, run.stdout)
                    self.assertIn(refusal, run.stdout)
                    self.assertFalse([call for call in calls if call.startswith(("docker stop", "serve", "check"))])

    def test_dry_run_runs_the_launchers_own_pre_boot_check_for_both_roles(self) -> None:
        self.stage()
        run, calls = self.run_kit("promote_r4h.sh", "--dry-run")
        self.assertEqual(run.returncode, 0, run.stdout)
        self.assertEqual([call for call in calls if call.startswith("check")], [f"check {CUT} r0", f"check {CUT} r1"])
        self.assertIn("CHAMPION_CHECK_OK role=r1", run.stdout)

    def test_a_failing_pre_boot_check_refuses_dry_run_and_promote_before_any_change(self) -> None:
        self.stage()
        (self.state / "check-fail-r1").touch()
        for args in (("--dry-run",), ()):
            run, calls = self.run_kit("promote_r4h.sh", *args)
            self.assertEqual(run.returncode, 2, run.stdout)
            self.assertIn("r1's pre-boot check refused: HALT arm=champion-r1: VLLM_DECODE_PRIORITY is not set", run.stdout)
            self.assertFalse([call for call in calls if call.startswith(("docker stop", "serve"))])

    def test_a_moved_gates_file_refuses(self) -> None:
        self.stage()
        (self.state / "gates.env").write_text(GATES + "R4E_ROUTED_GATE=on\n")
        run, calls = self.run_kit("promote_r4h.sh", "--dry-run")
        self.assertEqual(run.returncode, 2, run.stdout)
        self.assertIn("is not the staged", run.stdout)

    def test_serve_champion_reads_only_the_staged_gates_file(self) -> None:
        text = (self.kit / "serve_champion.sh").read_text()
        self.assertNotIn("r4f-B2", text)
        self.assertNotIn("r4f-gates.override.env", text)
        self.stage()
        staged = self.kit / "champion.staged"
        staged.write_text("".join(line + "\n" for line in staged.read_text().splitlines() if not line.startswith("CHAMPION_GATES")))
        run = subprocess.run(["bash", str(self.kit / "serve_champion.sh"), "r0", "--check"], capture_output=True, text=True, timeout=60)
        self.assertEqual(run.returncode, 2, run.stdout + run.stderr)
        self.assertIn("names no CHAMPION_GATES", run.stderr)

    def test_stage_refuses_a_tag_outside_the_c1d_repo_or_the_rollback_cut(self) -> None:
        for tag in ("other-repo:20261109-r4h5", "claude-flash-dual-graphs-c1d:20261109-r4f"):
            run, _ = self.run_kit("promote_r4h.sh", "stage", "--image", tag, "--image-sha", CHAMPION,
                                  "--levers", str(self.state / "levers.env"))
            self.assertEqual(run.returncode, 2, run.stdout)
            self.assertIn("REFUSED", run.stdout)
            self.assertFalse((self.kit / "champion.staged").exists())

    def test_a_port_held_by_an_unknown_container_refuses_naming_it_before_any_stop(self) -> None:
        self.stage()
        self.container("someone-elses-server", R4F, "true")
        (self.state / "args-someone-elses-server").write_text("--port 18275\n")
        run, calls = self.run_kit("promote_r4h.sh")
        self.assertEqual(run.returncode, 2, run.stdout)
        self.assertIn("port :18275 (r1) is held by someone-elses-server", run.stdout)
        self.assertFalse([call for call in calls if call.startswith(("docker stop", "serve"))])

    def test_a_port_held_outside_docker_refuses_before_any_stop(self) -> None:
        self.stage()
        self.container("claude-flash-launch-glm53flash-r0-r4f", R4F, "false")
        (self.state / "ss-18274").write_text("LISTEN 0 4096 0.0.0.0:18274 0.0.0.0:*\n")
        run, calls = self.run_kit("promote_r4h.sh")
        self.assertEqual(run.returncode, 2, run.stdout)
        self.assertIn("port :18274 (r0) has a listener no running container names", run.stdout)
        self.assertFalse([call for call in calls if call.startswith(("docker stop", "serve"))])

    def test_dry_run_changes_nothing(self) -> None:
        self.stage()
        run, calls = self.run_kit("promote_r4h.sh", "--dry-run")
        self.assertEqual(run.returncode, 0, run.stdout)
        self.assertIn("PROMOTE_DRYRUN_OK", run.stdout)
        self.assertFalse([call for call in calls if call.startswith(("docker stop", "docker rename", "serve", "systemctl stop"))])

    def test_r0_then_r1_each_through_every_gate_r4f_stopped_and_released_first(self) -> None:
        self.stage()
        run, calls = self.run_kit("promote_r4h.sh")
        self.assertEqual(run.returncode, 0, run.stdout)
        self.assertIn(f"C1D_CHAMPION_PROMOTED cut={CUT}", run.stdout)
        order = [call for call in calls if call.startswith(("docker stop", "serve"))]
        self.assertEqual(order, ["docker stop -t 60 claude-flash-launch-glm53flash-r0-r4f", f"serve {CUT} r0",
                                 "docker stop -t 60 claude-flash-launch-glm53flash-r1-r4f", f"serve {CUT} r1"])
        self.assertTrue((self.queue / "serve-r4f-r0" / "release").exists())
        self.assertTrue((self.queue / "serve-r4f-r1" / "release").exists())
        self.assertEqual(run.stdout.count("GATES_PASS"), 2)
        self.assertFalse([call for call in calls if call.startswith(("docker rm", "systemctl start", "docker run"))])

    def assert_rolled_back(self, run: subprocess.CompletedProcess, calls: list[str], why: str) -> None:
        self.assertEqual(run.returncode, 1, run.stdout)
        self.assertIn(why, run.stdout)
        self.assertIn("PROMOTE FAILED", run.stdout)
        self.assertIn("R4F_SERVE_RESTORED", run.stdout)
        self.assertNotIn(f"serve {CUT} r1", calls)
        self.assertEqual(self.image(f"claude-flash-launch-glm53flash-r0-{CUT}"), [CHAMPION, "false"])

    def test_a_boot_over_budget_rolls_back_before_r1(self) -> None:
        self.stage()
        (self.state / "boot").write_text("1501")
        self.assert_rolled_back(*self.run_kit("promote_r4h.sh"), "GATE_FAIL r0: boot 1501s (budget 1500s)")

    def test_a_warmup_over_budget_rolls_back(self) -> None:
        self.stage()
        (self.state / "warmup").write_text("301")
        self.assert_rolled_back(*self.run_kit("promote_r4h.sh"), "GATE_FAIL r0: warmup 301s (budget 300s)")

    def test_value_drift_rolls_back_naming_the_key(self) -> None:
        self.stage()
        (self.state / "drift-r0").touch()
        run, calls = self.run_kit("promote_r4h.sh")
        self.assert_rolled_back(run, calls, f"VALUE-DRIFT claude-flash-launch-glm53flash-r0-{CUT} HPU_GLM53_CAPTURE_EXECUTE_KEYS")
        self.assertIn("HPU_GLM53_ATTN_PIN intended=0 live=1", run.stdout)

    def test_the_r4h4_query_shard_at_4_1024_is_accepted_when_served(self) -> None:
        (self.state / "levers.env").write_text(LEVERS + QUERY_R4H4 + "\n")
        self.stage()
        run, _ = self.run_kit("promote_r4h.sh")
        self.assertEqual(run.returncode, 0, run.stdout)
        self.assertEqual(run.stdout.count("GATES_PASS"), 2)

    def test_the_query_shard_served_at_r4h3s_list_is_drift(self) -> None:
        (self.state / "levers.env").write_text(LEVERS + QUERY_R4H4 + "\n")
        (self.state / "live-env-r0").write_text(LEVERS + QUERY_R4H3 + "\n")
        self.stage()
        run, calls = self.run_kit("promote_r4h.sh")
        self.assert_rolled_back(run, calls, f"VALUE-DRIFT claude-flash-launch-glm53flash-r0-{CUT} HPU_GLM53_QUERY_SHARD_DECODE_KEYS "
                                            "intended=4:1024 live=4:1024,16:1024")

    def test_a_key_set_twice_live_is_drift_even_when_one_value_matches(self) -> None:
        """docker serves the last of a key's entries: an intended value present beside another one is not proof it is served."""
        (self.state / "levers.env").write_text(LEVERS + QUERY_R4H4 + "\n")
        (self.state / "live-env-r0").write_text(LEVERS + QUERY_R4H4 + "\n" + QUERY_R4H3 + "\n")
        self.stage()
        run, calls = self.run_kit("promote_r4h.sh")
        self.assert_rolled_back(run, calls, f"VALUE-DRIFT claude-flash-launch-glm53flash-r0-{CUT} HPU_GLM53_QUERY_SHARD_DECODE_KEYS "
                                            "set 2 times live")

    def test_a_failed_r1_boot_rolls_both_back_to_r4f(self) -> None:
        self.stage()
        (self.state / f"fail-{CUT}-r1").touch()
        run, calls = self.run_kit("promote_r4h.sh")
        self.assertEqual(run.returncode, 1, run.stdout)
        self.assertIn("R4F_SERVE_RESTORED", run.stdout)
        self.assertEqual(self.image(f"claude-flash-launch-glm53flash-r0-{CUT}"), [CHAMPION, "false"])
        self.assertIn("serve r4f r0", calls)
        self.assertIn("serve r4f r1", calls)

    def test_an_unstaged_or_moved_levers_file_refuses_before_any_change(self) -> None:
        run, calls = self.run_kit("promote_r4h.sh")
        self.assertEqual(run.returncode, 2)
        self.assertIn("champion.staged is missing", run.stdout)
        self.stage()
        (self.state / "levers.env").write_text(LEVERS + "X=1\n")
        run, calls = self.run_kit("promote_r4h.sh")
        self.assertEqual(run.returncode, 2)
        self.assertIn("is not the staged", run.stdout)
        self.assertFalse([call for call in calls if call.startswith(("docker stop", "serve"))])


# docker for champion_check.sh: create/cp/rm against $STATE/img (the image's files); run is logged and refused (it would start one).
FAKE_IMAGE_DOCKER = r'''#!/bin/bash
echo "docker $*" >> "$STATE/calls"
case "$1" in
  create) echo cid123 ;;
  cp) src="${2#*:}"; [ -f "$STATE/img$src" ] || exit 1; tar -C "$STATE/img$(dirname "$src")" -cf - "$(basename "$src")" ;;
  rm) exit 0 ;;
  *) exit 1 ;;
esac
'''


class ChampionCheck(unittest.TestCase):
    """champion_check.sh's baked_sha: the image's file hashed without starting a container (create + cp + rm)."""

    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.state = Path(self.tmp.name)
        (self.state / "img/site").mkdir(parents=True)
        (self.state / "img/site/glm53_loadpath.py").write_text("print('baked')\n")
        self.docker = self.state / "docker"
        self.docker.write_text(FAKE_IMAGE_DOCKER)
        self.docker.chmod(0o755)

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def baked(self, name: str) -> tuple[str, list[str]]:
        script = f'source "{HERE / "champion_check.sh"}"; IMAGE=img:tag; SITE=/site; baked_sha {name}'
        run = subprocess.run(["bash", "-c", script], capture_output=True, text=True, timeout=60,
                             env=dict(os.environ, STATE=str(self.state), DOCKER=str(self.docker)))
        return run.stdout.strip(), (self.state / "calls").read_text().splitlines()

    def test_the_baked_file_is_hashed_from_a_created_container_that_never_starts_and_is_removed(self) -> None:
        sha, calls = self.baked("glm53_loadpath.py")
        self.assertEqual(sha, hashlib.sha256(b"print('baked')\n").hexdigest()[:8])
        self.assertTrue(calls[0].startswith("docker create ") and "--network none" in calls[0]
                        and "HABANA_VISIBLE_DEVICES=none" in calls[0] and calls[0].endswith(" img:tag"), calls)
        self.assertEqual(calls[1:], ["docker cp cid123:/site/glm53_loadpath.py -", "docker rm cid123"])
        self.assertFalse([call for call in calls if call.startswith(("docker run", "docker start"))])

    def test_a_file_the_image_lacks_prints_nothing_and_still_removes_the_created_container(self) -> None:
        sha, calls = self.baked("absent.py")
        self.assertEqual(sha, "")
        self.assertEqual(calls[-1], "docker rm cid123")

    def test_serve_champions_check_mode_binds_baked_sha_to_the_kit_helper_after_the_launcher(self) -> None:
        text = (HERE / "serve_champion.sh").read_text()
        launcher, helper = text.find('source "$L/diag-r4e/r4e_run.sh"'), text.find('source "$KIT_DIR/champion_check.sh"')
        self.assertGreater(launcher, 0)
        self.assertGreater(helper, launcher)   # sourced after r4e_run.sh, so its baked_sha replaces the docker-run one


class Rollback(Kit):
    def setUp(self) -> None:
        super().setUp()
        for role in ("r0", "r1"):   # the champion serves; r4f's containers are stopped; a stray r4h unit drop-in is left
            self.container(f"claude-flash-launch-glm53flash-{role}-r4f", R4F, "false")
            self.container(f"claude-flash-launch-glm53flash-{role}-{CUT}", CHAMPION, "true")
            (self.units / f"glm53flash-{role}.service.d" / R4H_DROPIN).write_text("[Service]\n")
        for name in ("glm53flash-production.target", "glm53flash-r0.service", "glm53flash-r1.service"):
            (self.state / f"active-{name}").touch()
        self.stage()   # the rollback stops the staged champion's containers

    def test_dry_run_changes_nothing(self) -> None:
        run, calls = self.run_kit("rollback_r4f.sh", "--dry-run")
        self.assertEqual(run.returncode, 0, run.stdout)
        self.assertIn("ROLLBACK_R4F_DRYRUN_OK", run.stdout)
        self.assertFalse([call for call in calls if call.startswith(("docker stop", "docker rename", "serve", "systemctl stop"))])

    def test_the_champion_stops_and_r4f_boots_r0_then_r1_nothing_started_or_removed(self) -> None:
        run, calls = self.run_kit("rollback_r4f.sh")
        self.assertEqual(run.returncode, 0, run.stdout)
        self.assertIn("R4F_SERVE_RESTORED", run.stdout)
        self.assertLess(calls.index(f"docker stop -t 60 claude-flash-launch-glm53flash-r0-{CUT}"), calls.index("serve r4f r0"))
        self.assertLess(calls.index("serve r4f r0"), calls.index("serve r4f r1"))
        self.assertFalse([call for call in calls if call.startswith(("docker rm", "docker start", "docker run", "systemctl start"))])
        for role in ("r0", "r1"):
            unit = self.units / f"glm53flash-{role}.service.d"
            self.assertFalse((unit / R4H_DROPIN).exists())
            self.assertEqual(len(list(unit.glob(f"{R4H_DROPIN}.disabled-*"))), 1)
            self.assertEqual(self.image(f"glm53flash-{role}")[0], RC56)

    def test_a_role_r4f_still_serves_is_kept(self) -> None:
        self.container("claude-flash-launch-glm53flash-r1-r4f", R4F, "true")
        run, calls = self.run_kit("rollback_r4f.sh")
        self.assertEqual(run.returncode, 0, run.stdout)
        self.assertIn("r4f still serves r1 healthy; kept", run.stdout)
        self.assertNotIn("serve r4f r1", calls)

    def test_a_changed_rc56_dropin_refuses_before_any_change(self) -> None:
        (self.units / "glm53flash-r1.service.d" / RC56_DROPIN).write_text("edited\n")
        run, calls = self.run_kit("rollback_r4f.sh")
        self.assertEqual(run.returncode, 2)
        self.assertIn("RC56's drop-in must stay untouched", run.stdout)
        self.assertFalse([call for call in calls if not call.startswith("docker image")])

    def test_a_unit_still_active_after_the_stop_refuses_before_any_boot(self) -> None:
        Path(self.fakes["systemctl"]).write_text(FAKE_SYSTEMCTL.replace('rm -f "$STATE"/active-*', ":"))
        run, calls = self.run_kit("rollback_r4f.sh")
        self.assertEqual(run.returncode, 2)
        self.assertIn("still active after the stop", run.stdout)
        self.assertFalse([call for call in calls if call.startswith("serve")])

    def test_a_failed_r0_boot_alarms_and_still_brings_r1_back(self) -> None:
        (self.state / "fail-r4f-r0").touch()
        run, calls = self.run_kit("rollback_r4f.sh")
        self.assertEqual(run.returncode, 1)
        self.assertIn("ALARM: R4F ROLLBACK INCOMPLETE: replicas down: r0", run.stdout)
        self.assertIn("VERIFIED r1", run.stdout)

    def test_a_test_hook_without_test_mode_refuses(self) -> None:
        env = {key: value for key, value in os.environ.items() if key != "GLM53_PROMOTE_TEST"}
        run = subprocess.run(["bash", str(self.kit / "rollback_r4f.sh"), "--dry-run"], env=dict(env, DOCKER="/bin/true"),
                             capture_output=True, text=True, timeout=60)
        self.assertEqual(run.returncode, 2)
        self.assertIn("test hook DOCKER is set without GLM53_PROMOTE_TEST=1", run.stderr)


if __name__ == "__main__":
    unittest.main()
