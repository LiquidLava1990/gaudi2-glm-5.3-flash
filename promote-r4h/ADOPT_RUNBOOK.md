# Promote the pin-trial champion: no-restart adopt (prepared 2026-11-22, nothing here has been run on production)

Kit to install: `$HOME/tmp/pin-trial-20261122/promote-final/` (= production kit + the trial's seeding / AccThreadPool pins /
C1D_KV_BLOCKS=30880 + lab_serve_seeded.py, + a durable seed copy, + `adopt_check.sh`, + `champion_boot.sh --keep-seed`).
Its `champion.staged` is a PREVIEW only (cand5 values, promote-final paths, alias tag). Step 3 re-stages it with the final inputs.

## Why there is no `promote_r4h.sh` run (iv)

`promote_r4h.sh` (no args) is an **r4f -> champion** installer. Today it is wrong in four ways:

1. `promote_preflight` -> `check_port_owner` lets only `claude-flash-launch-glm53flash-rN-r4f` hold :18274/:18275. Both ports are held
   by `...-rN-r4h8`, so `--dry-run` and the real run both stop with
   `REFUSED: port :18274 (r0) is held by claude-flash-launch-glm53flash-r0-r4h8, not claude-flash-launch-glm53flash-r0-r4f; stop it by hand first`
   (seen 17:18:52 from promote-final). Following that advice means restarting both replicas for nothing.
2. If someone stopped the replicas by hand to get past it, `promote_role` would stop an r4f container that does not exist, then
   boot both replicas again (about 15 min of full-quad compile each).
3. On any failed gate, `roll_back` stops **both** champion containers and runs `rollback_r4f.sh`. That stops
   `glm53flash-production.target` and the units, and boots **r4f aabd010a (2026-11-09)**, not the r4h8 champion.
4. `stage` takes the cut from the tag's last `-` suffix. Every candidate tag ends in something other than `r4h8`
   (`20261122-r4h8-pin9hr2s1` gives cut `pin9hr2s1`; seen 17:18:28). The container name and run folder would then become
   `...-rN-pin9hr2s1`. The recover timer runs every minute, finds `champion-rN.wanted` with no running
   `...-pin9hr2s1` container and no `serve-pin9hr2s1-rN/status`, and calls `restart_replica "the last start failed"`, which
   **boots a second replica on a quad that is already in use.** So stage only with a tag ending in `-r4h8` (step 2) and check
   the cut before installing.

How `champion_boot.sh` decides that a running container matches: **by container name only**
(`claude-flash-launch-glm53flash-$ROLE-$CHAMPION_CUT`, `is_running` = `docker inspect .State.Running`). It never compares the
image sha, the levers or the KV blocks. `--plan` says `nothing to do` for a container of the right name running ANY image (seen:
r1 runs b8f9ae96 while cand5 stages 56fdfa0b). The recover timer leaves any running container of that name alone. Only once that
container stops on its own does it boot `champion.staged`. **A mismatch is therefore silent until the next crash or reboot, and
then the replica changes image.** `adopt_check.sh` is the gate that closes this gap (image id, VALUE-DRIFT 0 against the staged
levers, `--num-gpu-blocks-override` = the kit's 30880, /health; optional `--smoke`, `--launcher-check`).

## Inputs

```bash
IMAGE=claude-flash-dual-graphs-c1d:<final tag>          # e.g. ...:20261122-r4h8-pin9hr2s1
IMAGE_SHA=sha256:<64 hex>                                # docker image inspect -f '{{.Id}}' "$IMAGE"
LEVERS=$HOME/tmp/pin-trial-20261122/inputs/<final levers>.env
GATES_SRC=$HOME/tmp/pin-trial-20261122/promote-final/inputs/gates.override.env   # production gates with R4E_PIN_GATE=on
#   use /mnt/data1/graded-out/queue/serve-r4h8-row8-tail-gap8-high-inputs/gates.override.env (off) only if LEVERS sets no HPU_GLM53_ATTN_PIN
FINAL_ENV=/mnt/data1/graded-out/queue/serve-r4h6e-inputs/final-config.env           # unchanged (5a465cc2)
P=$HOME/.codex/tmp/flash-spikes-20261104/team-20261106-dual/promote-r4h
F=$HOME/tmp/pin-trial-20261122/promote-final
DEST=/mnt/data1/graded-out/queue/serve-r4h8-pinfinal-inputs   # durable home beside the current champion's serve-r4h8-row8-tail-gap8-high-inputs/
```

## 0. Precondition: both replicas already serve IMAGE (read-only)

```bash
for r in r0 r1; do docker inspect -f '{{.Name}} {{.Image}}' claude-flash-launch-glm53flash-$r-r4h8; done
```
Expected: both lines end in `$IMAGE_SHA`. If one role differs, see "One role on another image" below.

## 1. Durable inputs

```bash
mkdir "$DEST"                                   # must not exist yet; an error here = STOP
install -m 444 "$LEVERS" "$DEST/levers.env"
install -m 444 "$GATES_SRC" "$DEST/gates.override.env"
sha256sum "$DEST"/*                             # note both sha8s
```

## 2. A stage tag whose cut is r4h8 (adds a tag only; no container is touched)

```bash
case "$IMAGE" in *-r4h8) STAGE_TAG="$IMAGE" ;;
  *) STAGE_TAG="claude-flash-dual-graphs-c1d:$(date +%Y%m%d)-${IMAGE##*-}-r4h8"; docker tag "$IMAGE" "$STAGE_TAG" ;; esac
[ "$(docker image inspect -f '{{.Id}}' "$STAGE_TAG")" = "$IMAGE_SHA" ] && echo TAG_OK "$STAGE_TAG"
```
Expected: `TAG_OK claude-flash-dual-graphs-c1d:2026112x-<variant>-r4h8`.

## 3. Stage (ii) into promote-final, not production

```bash
cd "$F" && bash promote_r4h.sh stage --image "$STAGE_TAG" --image-sha "$IMAGE_SHA" \
    --levers "$DEST/levers.env" --gates "$DEST/gates.override.env" --final-env "$FINAL_ENV"
grep -x 'CHAMPION_CUT=r4h8' "$F/champion.staged" || echo STOP_WRONG_CUT
```
Expected: `[... EST] STAGED CHAMPION_CUT=r4h8 CHAMPION_IMAGE=<STAGE_TAG> CHAMPION_IMAGE_SHA=<IMAGE_SHA> ...`, then `CHAMPION_CUT=r4h8`.
`STOP_WRONG_CUT` = do not install.

## 4. Gates before installing (read-only; iii)

```bash
cd "$F" && bash adopt_check.sh --launcher-check          # --smoke adds one small request per replica
bash champion_boot.sh r0 --plan; bash champion_boot.sh r1 --plan
bash promote_r4h.sh --dry-run; tail -n 2 logs/promote_r4h-latest.log
```
Expected:
- `ADOPT_ROLE_OK r0 claude-flash-launch-glm53flash-r0-r4h8 image <sha12> VALUE-DRIFT 0 kv_blocks 30880 :18274`, the same line for r1 on `:18275`, `CHAMPION_CHECK_OK role=r0|r1 cut=r4h8 ...`, and last `ADOPT_OK cut=r4h8 ... nothing to restart`.
- `PLAN nothing to do: claude-flash-launch-glm53flash-r0-r4h8 is running`, and the same for r1.
- `ROLLBACK_PREFLIGHT_OK ...`, then `REFUSED: port :18274 (r0) is held by claude-flash-launch-glm53flash-r0-r4h8 ...`. This is expected (see iv). It proves the dry-run is not the gate.

`ADOPT_REFUSED` = STOP (nothing has changed yet).

## 5. Install the kit into production (i)

This follows the 11-09 convention: `<file>.prev-<sha8 of the old bytes>`, `SHA256SUMS.<sha8 of the old SHA256SUMS>`,
`prior-<same sha8>/` holding the whole previous kit, and `champion.staged.prev-<previous champion label>`. Each file goes in by
rename, so a running `champion_boot.sh` (the recover timer) keeps reading the bytes it opened.

```bash
cd "$P"
OLD=$(sha256sum < SHA256SUMS | cut -c1-8)                     # b5019956 today (stale: it predates champion_boot.sh)
mkdir "prior-$OLD" && cp -p SHA256SUMS champion.staged champion_boot.sh common.sh serve_champion.sh promote_r4h.sh \
    rollback_r4f.sh champion_check.sh test_champion_boot.py test_promote_kit.py "prior-$OLD/"
(cd "prior-$OLD" && sha256sum *.sh *.py champion.staged > SHA256SUMS.files)   # the real bytes of the prior kit
cp -p SHA256SUMS "SHA256SUMS.$OLD"
cp -p champion.staged champion.staged.prev-r4h8-row8-tail-gap8-high
KIT="champion_boot.sh common.sh serve_champion.sh promote_r4h.sh adopt_check.sh test_champion_boot.py test_promote_kit.py"
for f in $KIT; do
  if [ -f "$f" ] && ! cmp -s "$f" "$F/$f"; then b="$f.prev-$(sha256sum < "$f" | cut -c1-8)"; [ -e "$b" ] || cp -p "$f" "$b"; fi
done
for f in $KIT champion.staged; do cp -p "$F/$f" ".$f.new" && mv -f ".$f.new" "$f"; done
sha256sum common.sh promote_r4h.sh rollback_r4f.sh serve_champion.sh champion_check.sh test_promote_kit.py \
    champion_boot.sh test_champion_boot.py adopt_check.sh > SHA256SUMS.new && mv -f SHA256SUMS.new SHA256SUMS
sha256sum -c SHA256SUMS
```
Expected new `.prev` files: `champion_boot.sh.prev-73672b76`, `common.sh.prev-39e06c74`, `serve_champion.sh.prev-b91d1242`,
`promote_r4h.sh.prev-09fa8ac6`, `test_champion_boot.py.prev-946f75d6`, `test_promote_kit.py.prev-22ab1e56`.
`sha256sum -c` prints 9 `: OK` lines. The installed sha8s are champion_boot 7d79beb4, common 4b5bdef9, serve_champion 6db9a975,
promote 5f1b219c, adopt_check fb3761f0, test_champion_boot 276c924f, test_promote_kit b1cb3c66; rollback b01d90bb and
champion_check dab6ae40 are unchanged. If you change a file in promote-final after 2026-11-22 17:16, re-hash.

## 6. Verify in production (read-only), then protect the seed

```bash
cd "$P" && bash adopt_check.sh && bash champion_boot.sh r0 --plan && bash champion_boot.sh r1 --plan
nice -n19 python3 -I test_champion_boot.py 2>&1 | tail -n 1; nice -n19 python3 -I test_promote_kit.py 2>&1 | tail -n 1
sleep 150; grep -E 'RECOVER|BOOT ' /mnt/data1/graded-out/queue/champion-boot-r[01].log | tail -n 4   # no new line after the install
bash champion_boot.sh r0 --keep-seed        # one copy per image; r1 serves the same image, so one run is enough
```
Expected: `ADOPT_OK cut=r4h8 ...`, two `PLAN nothing to do`, `OK` for each test file (69 and 41 tests), no RECOVER/BOOT line
after the install time, and `SEED_COPIED /mnt/data1/recipe_cache/claude-flash-launch-glm53flash-r0-r4h8-<stamp> -> /mnt/data1/recipe_cache/seed-<sha12> ...`
(about 350 MB at idle IO priority).

## One role on another image (r1 is on b8f9ae96 right now)

Adopt refuses. If you still promote: complete steps 1-3, then steps 5-6 with `adopt_check.sh` expected to fail for that role only.
Then restart **only that role** from the installed kit: `bash $P/champion_boot.sh r1 --stop`, wait for its `STOPPED` line, then
`bash $P/champion_boot.sh r1`. Expected: `SEED /mnt/data1/recipe_cache/claude-flash-launch-glm53flash-r0-r4h8-<stamp>` (r0's
recorded leaf for the same image), `SERVING ...`, `SEED_RECORDED`, `ACC_PIN_DONE ... pinned=4/4`, `SEED_COPIED`. Re-run
`adopt_check.sh` and expect `ADOPT_OK`.

## (v) Rollback to the 20261112-r4h8 champion

Files first. This changes nothing that is running:
```bash
cd "$P"; OLD=b5019956
for f in champion_boot.sh common.sh serve_champion.sh promote_r4h.sh test_champion_boot.py test_promote_kit.py champion.staged SHA256SUMS; do
  cp -p "prior-$OLD/$f" ".$f.rb" && mv -f ".$f.rb" "$f"; done
mv adopt_check.sh "adopt_check.sh.prev-$(sha256sum < adopt_check.sh | cut -c1-8)"
(cd "prior-$OLD" && sha256sum -c SHA256SUMS.files)
grep -E '^CHAMPION_(CUT|IMAGE)=' champion.staged   # CHAMPION_CUT=r4h8 / CHAMPION_IMAGE=claude-flash-dual-graphs-c1d:20261112-r4h8
```
The replicas keep serving the new image under the same names, and the timer leaves them alone. To actually serve 8bf1872d
again, do one role at a time and start r0 only after r1 shows SERVING:
`bash $P/champion_boot.sh r1 --stop`, then `bash $P/champion_boot.sh r1` (expect `SERVING BOOTED serve role=r1 ... boot_s=<=1500`), then the same for r0.
The restored kit boots unseeded (stock lab_serve.py, KV 33280), which means a full compile on all 4 cards per replica, exactly
as r4h8 booted before 2026-11-22. The `seed-<sha12>` copies and `recipe-seed-*.copy` marks are additive and can stay.
