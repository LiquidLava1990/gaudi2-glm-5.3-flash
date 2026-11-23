#!/bin/bash
# promote-r4h/champion_boot.sh: keep one champion serving replica up (owner 2026-11-11: the champions "stay up";
# owner 2026-11-13: a replica that stops on its own restarts itself).
# Run by glm53-champion-r0|r1.service at host boot and by glm53-champion-recover@r0|r1.timer every minute; safe by hand.
# It serves whatever champion.staged names, through serve_champion.sh, exactly as a manual launch does.
# usage: champion_boot.sh r0|r1             start the replica if it is not running (a running one is left alone)
#        champion_boot.sh r0|r1 --plan      print what a start would do, change nothing
#        champion_boot.sh r0|r1 --stop      take the replica down on purpose: docker stop, free its quad; it stays down
#        champion_boot.sh r0|r1 --recover   restart the replica if it went down by itself (silent when there is nothing to do)
#        champion_boot.sh r0|r1 --pin-acc   re-apply the AccThreadPool cpu pins of the running replica only (common.sh pin_acc_threads)
#        champion_boot.sh r0|r1 --keep-seed copy the running replica's recipe leaf to its protected seed copy now (idle IO; no boot)
# A boot is seeded from the recipe leaf the last serving boot of the same staged image recorded (either role; recipe-seed-<sha12>.leaf
# in the queue folder); that leaf gone (lab.py down removes it), the boot is seeded from the protected copy the serving boot kept
# (recipe_cache/seed-<sha12>); neither, a fresh empty leaf. LAB_RECIPE_SEED_LEAF in the environment wins.
# To take a quad away from a serving replica, use --stop or a plain "docker stop" (or "docker rm -f"): docker records
# the signal it was asked to send, so --recover sees the stop was made by hand, leaves the replica down and frees its
# quad within a minute. Only a replica that went down by itself (card error, memory guard, killed for memory, docker
# restarted under it) is restarted. Bring a replica stopped by hand back with: champion_boot.sh r0|r1
# A replica that stopped on its own (card error, memory guard) leaves its quad holder alive; a start frees that holder
# first, the same way --stop does, so the boot does not wait on a quad nobody is using.
# Deletes nothing: a stopped container of the same name and an old run folder are renamed with a time stamp.
set -u
KIT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROLE="${1:?usage: champion_boot.sh r0|r1 [--plan|--stop|--recover|--pin-acc|--keep-seed]}"
MODE="${2:-start}"
case "$ROLE" in r0|r1) ;; *) echo "HALT champion_boot.sh: role $ROLE is not r0 or r1" >&2; exit 2 ;; esac
case "$MODE" in start|--plan|--stop|--recover|--pin-acc|--keep-seed) ;;
    *) echo "HALT champion_boot.sh: mode $MODE is not --plan, --stop, --recover, --pin-acc or --keep-seed" >&2; exit 2 ;; esac
Q="${QUEUE_DIR:-/mnt/data1/graded-out/queue}"
STAGED="$KIT_DIR/champion.staged"
CUT="$(sed -n 's/^CHAMPION_CUT=//p' "$STAGED" 2> /dev/null)"
FINAL_ENV_FILE="$(sed -n 's/^CHAMPION_FINAL_ENV=//p' "$STAGED" 2> /dev/null)"
[ -n "$CUT" ] && [ -n "$FINAL_ENV_FILE" ] || {
    echo "HALT champion_boot.sh: $STAGED names no CHAMPION_CUT or CHAMPION_FINAL_ENV; stage a champion with promote_r4h.sh stage --final-env" >&2; exit 2; }
IMAGE_SHA="$(sed -n 's/^CHAMPION_IMAGE_SHA=//p' "$STAGED" 2> /dev/null)"
RECIPE_ROOT="${RECIPE_ROOT:-/mnt/data1/recipe_cache}"   # where the launcher makes its leaves (lab_serve_seeded.py RECIPE_ROOT)
NAME="claude-flash-launch-glm53flash-$ROLE-$CUT"
RUN="$Q/serve-$CUT-$ROLE"
STAMP="$(date +%Y%m%d-%H%M%S)"
LOG="$Q/champion-boot-$ROLE.log"
RELEASE="$RUN/release"                        # the file the serving replica's quad holder waits for (quad_hold.sh)
HOLDER_WAIT_S="${HOLDER_WAIT_S:-15}"          # quad_hold.sh looks for its release file every 5 s
WANTED="$Q/champion-$ROLE.wanted"             # present while this replica is meant to be serving
STOPPED_ON_PURPOSE="$Q/champion-$ROLE.stopped-on-purpose"   # written by --stop, removed by a start by hand: --recover stays out
TRIES="$Q/champion-recover-$ROLE.tries"       # one line (epoch seconds) per automatic restart
HALTED="$Q/champion-recover-$ROLE.halted"     # present once the give-up line has been written
BOOT_LOCK="$Q/champion-boot-$ROLE.lock"       # one start per replica at a time
LOCK_BUSY=75                                  # flock's exit code when another start holds the lock
RECOVER_MAX_TRIES="${RECOVER_MAX_TRIES:-3}"   # automatic restarts allowed within the window
RECOVER_WINDOW_S="${RECOVER_WINDOW_S:-10800}"
EVENT_LOOKBACK_S=300                          # docker's event history is read from this long before the container stopped
say() { echo "$(TZ=America/New_York date '+%F %T %Z') [$ROLE] $*" | tee -a "$LOG"; }
is_running() { [ "$(docker inspect -f '{{.State.Running}}' "$NAME" 2> /dev/null)" = true ]; }
exists() { docker inspect "$NAME" > /dev/null 2>&1; }
was_serving() { grep -q ' SERVING ' "$RUN/status" 2> /dev/null; }
sha12() { local sha="${1#sha256:}"; echo "${sha:0:12}"; }
seed_record() { echo "$Q/recipe-seed-$(sha12 "$1").leaf"; }   # $1 image id: the leaf its last serving boot used
leaf_has_files() { [ -d "$1" ] && [ -n "$(find "$1" -mindepth 1 -maxdepth 1 -print -quit 2> /dev/null)" ]; }

# Seeded recipe cache (2026-11-22): an empty leaf recompiles every graph on all four cards at once, which stresses the host.
container_leaf() {   # the host path of the recipe leaf the container uses (its PT_HPU_RECIPE_CACHE_CONFIG=/recipe_cache/<leaf>,...)
    local config
    config="$(docker inspect -f '{{range .Config.Env}}{{println .}}{{end}}' "$NAME" 2> /dev/null | sed -n 's/^PT_HPU_RECIPE_CACHE_CONFIG=//p')"
    config="${config%%,*}"
    [ -n "$config" ] && echo "$RECIPE_ROOT/${config##*/}"
}

record_seed_leaf() {   # after SERVING: this boot's leaf seeds the next boot of the same image (temp file + mv)
    local image leaf record
    image="$(docker inspect -f '{{.Image}}' "$NAME" 2> /dev/null)"
    leaf="$(container_leaf)"
    if [ -z "$image" ] || ! leaf_has_files "$leaf"; then
        say "SEED_RECORD_SKIPPED $NAME: image '$image' leaf '$leaf' (no image id, or no leaf with files); the earlier record stays"; return 0
    fi
    record="$(seed_record "$image")"
    if ! { printf '%s\n' "$leaf" > "$record.tmp.$$" && mv -f "$record.tmp.$$" "$record"; }; then
        rm -f "$record.tmp.$$"; say "SEED_RECORD_FAILED could not write $record; make $Q writable"; return 0
    fi
    say "SEED_RECORDED $leaf for image $(sha12 "$image") ($record)"
}

choose_seed() {   # before a boot: the caller's LAB_RECIPE_SEED_LEAF, else the recorded leaf, else the protected copy, else none
    if [ -n "${LAB_RECIPE_SEED_LEAF:-}" ]; then say "SEED $LAB_RECIPE_SEED_LEAF (LAB_RECIPE_SEED_LEAF from the environment)"; return 0; fi
    unset LAB_RECIPE_SEED_LEAF
    if [ -z "$IMAGE_SHA" ]; then say "SEED none ($STAGED names no CHAMPION_IMAGE_SHA)"; return 0; fi
    local record leaf="" seed
    record="$(seed_record "$IMAGE_SHA")"
    [ -f "$record" ] && leaf="$(head -n 1 "$record")"
    if [ -n "$leaf" ] && leaf_has_files "$leaf"; then export LAB_RECIPE_SEED_LEAF="$leaf"; say "SEED $leaf"; return 0; fi
    seed="$(seed_dir "$IMAGE_SHA")"
    if is_kit_seed "$IMAGE_SHA" && leaf_has_files "$seed"; then
        export LAB_RECIPE_SEED_LEAF="$seed"
        say "SEED $seed (the protected copy; the recorded leaf '$leaf' is gone or empty)"; return 0
    fi
    if [ ! -f "$record" ]; then say "SEED none (no recorded leaf for image $(sha12 "$IMAGE_SHA"))"; return 0; fi
    say "SEED none (the leaf $record names, '$leaf', is gone or empty, and no protected copy; the next boot that serves records a new one)"
}

# Durable seed (2026-11-22): lab.py down of the serving name removes its leaf (lab_serve_seeded.py remove_leaf), which leaves the
# record above naming nothing. After a serving boot one copy per image is kept as recipe_cache/seed-<sha12>: never a claude-flash-*
# leaf, so lab.py never removes it. A real copy (ext4 has no reflink; protected_hardlinks refuses links to the container's
# root-owned recipes) at idle IO priority, swapped in whole (mv --exchange). Its mark in the queue (recipe-seed-<sha12>.copy:
# line 1 the image id, line 2 the copy) proves the kit made it: only a marked copy is used, replaced or pruned.
SEEDS_KEPT=2                                  # copies kept: the newest images' (the one just copied always stays)
seed_dir() { echo "$RECIPE_ROOT/seed-$(sha12 "$1")"; }
seed_mark() { echo "$Q/recipe-seed-$(sha12 "$1").copy"; }
is_kit_seed() {   # $1 image id: seed-<sha12> is there and its mark names it
    [ -d "$(seed_dir "$1")" ] && [ -f "$(seed_mark "$1")" ] && [ "$(sed -n 2p "$(seed_mark "$1")")" = "$(seed_dir "$1")" ]
}

write_mark() {   # $1 image id (temp file + mv)
    local mark
    mark="$(seed_mark "$1")"
    printf '%s\n%s\n' "$1" "$(seed_dir "$1")" > "$mark.tmp.$$" && mv -f "$mark.tmp.$$" "$mark" || { rm -f "$mark.tmp.$$"; return 1; }
}

IDLE=(ionice -c3 nice -n19)                   # the copy and its clean-up yield to every other disk and cpu user
remove_dead_temp_copies() {   # $1 seed folder: <seed>.new-<pid> copies whose champion_boot.sh is gone (killed mid-copy)
    local temp pid
    for temp in "$1".new-*; do
        pid="${temp##*.new-}"
        if [ -d "$temp" ] && [[ "$pid" =~ ^[0-9]+$ ]] && ! kill -0 "$pid" 2> /dev/null; then "${IDLE[@]}" rm -rf -- "$temp"; fi
    done
}

copy_seed() {   # $1 image id, $2 leaf: a fresh full copy beside the old one, then one rename swaps it in
    local seed new
    seed="$(seed_dir "$1")"; new="$seed.new-$$"
    if [ -e "$seed" ] && ! is_kit_seed "$1"; then say "SEED_COPY_SKIPPED $seed has no mark $(seed_mark "$1"): not a copy this kit made; left alone"; return 1; fi
    remove_dead_temp_copies "$seed"
    if ! { mkdir -- "$new" && "${IDLE[@]}" cp -a -- "$2/." "$new/"; }; then
        "${IDLE[@]}" rm -rf -- "$new"; say "SEED_COPY_FAILED could not copy $2 to $new; the earlier copy (if any) stays"; return 1
    fi
    if [ -e "$seed" ]; then   # the old copy comes out under the temp name, so a reader never finds seed-<sha12> missing
        mv --exchange -T -- "$new" "$seed" || { "${IDLE[@]}" rm -rf -- "$new"; say "SEED_COPY_FAILED could not swap $new with $seed; the earlier copy stays"; return 1; }
        "${IDLE[@]}" rm -rf -- "$new"; touch -- "$(seed_mark "$1")"; return 0
    fi
    write_mark "$1" && mv -T -- "$new" "$seed" && return 0
    "${IDLE[@]}" rm -rf -- "$new"; say "SEED_COPY_FAILED could not move $new to $seed"; return 1
}

prune_seeds() {   # $1 the image just copied: marked copies of other images beyond the SEEDS_KEPT newest marks go
    local mark image kept=0
    while IFS= read -r mark; do
        image="$(sed -n 1p "$mark")"
        if [ "$mark" != "$(seed_mark "$image")" ] || [ "$(sed -n 2p "$mark")" != "$(seed_dir "$image")" ] || ! [[ "$(sha12 "$image")" =~ ^[0-9a-f]{12}$ ]]; then
            say "SEED_PRUNE_SKIPPED $mark does not name its own seed-<sha12> copy; left alone"; continue
        fi
        kept=$((kept + 1))
        if [ "$(sha12 "$image")" = "$(sha12 "$1")" ] || [ "$kept" -le "$SEEDS_KEPT" ]; then continue; fi
        "${IDLE[@]}" rm -rf -- "$(seed_dir "$image")" && rm -f -- "$mark" && say "SEED_PRUNED $(seed_dir "$image") (image $(sha12 "$image"); the $SEEDS_KEPT newest copies stay)"
    done < <(ls -1t -- "$Q"/recipe-seed-*.copy 2> /dev/null)
}

keep_seed_copy() {   # after SERVING and the pins (or --keep-seed): the protected copy of the replica's leaf, then the prune
    local image leaf
    image="$(docker inspect -f '{{.Image}}' "$NAME" 2> /dev/null)"
    leaf="$(container_leaf)"
    if [ -z "$image" ] || ! leaf_has_files "$leaf"; then say "SEED_COPY_SKIPPED $NAME: image '$image' leaf '$leaf' (no image id, or no leaf with files)"; return 1; fi
    copy_seed "$image" "$leaf" || return 1
    say "SEED_COPIED $leaf -> $(seed_dir "$image") (protected from lab.py down; mark $(seed_mark "$image"))"
    prune_seeds "$image"
}

keep_seed_running() {   # --keep-seed: a replica adopted without a boot by this kit gets its protected copy now
    is_running || { say "HALT $NAME is not running; no leaf to keep (a boot that serves keeps one: champion_boot.sh $ROLE)"; return 1; }
    keep_seed_copy
}

# common.sh runs in a clean shell: its test-hook names (STAGED, ...) are plain variables here.
pin_acc() {
    local line
    KIT_DIR="$KIT_DIR" bash -c 'source "$KIT_DIR/common.sh" && pin_acc_threads "$1"' pin-acc "$ROLE" 2>&1 \
        | while IFS= read -r line; do say "$line"; done
    return "${PIPESTATUS[0]}"
}

pin_running() {   # --pin-acc
    is_running || { say "HALT $NAME is not running; nothing to pin (start it: champion_boot.sh $ROLE)"; return 1; }
    pin_acc
}

# The stop keeps its own record, written before docker is asked: the container can take a minute to end, the timer runs
# meanwhile, and docker's event history (256 events, about 100 s on this host) may no longer hold the "kill" afterwards.
stop_replica() {
    date > "$STOPPED_ON_PURPOSE"
    rm -f "$WANTED"
    is_running && docker stop -t 60 "$NAME" > /dev/null
    [ -d "$RUN" ] && touch "$RELEASE"
    say "STOPPED $NAME (release file touched; the quad holder frees itself; it stays down until a start by hand)"
}

# Free the quad holder of a replica that is no longer running. The holder removes the release file when it ends; a file
# still there after the wait means no holder was watching it.
release_dead_holder() {
    [ -d "$RUN" ] || return 0
    touch "$RELEASE"
    local waited=0
    while [ -e "$RELEASE" ] && [ "$waited" -lt "$HOLDER_WAIT_S" ]; do sleep 1; waited=$((waited + 1)); done
    if [ -e "$RELEASE" ]; then rm -f "$RELEASE"; return 0; fi
    say "RELEASED the quad holder of the stopped $NAME"
}

plan() {
    if is_running; then say "PLAN nothing to do: $NAME is running"; return 0; fi
    exists && say "PLAN docker rename $NAME $NAME-stopped-$STAMP"
    [ -d "$RUN" ] && say "PLAN touch $RELEASE (frees the stopped replica's quad holder), wait up to ${HOLDER_WAIT_S}s"
    [ -e "$RUN" ] && say "PLAN mv $RUN $RUN.run-$STAMP"
    say "PLAN serve_champion.sh $ROLE --check, then serve_champion.sh $ROLE (cut $CUT, final env $FINAL_ENV_FILE)"
    choose_seed
}

start_replica() {
    if is_running; then say "ALREADY_SERVING $NAME: left alone"; return 0; fi
    if exists; then docker rename "$NAME" "$NAME-stopped-$STAMP" || { say "HALT could not rename the stopped $NAME; rename it by hand"; return 1; }; fi
    release_dead_holder
    if [ -e "$RUN" ]; then mv "$RUN" "$RUN.run-$STAMP" || { say "HALT could not move $RUN aside; move it by hand"; return 1; }; fi
    cd "$KIT_DIR" || return 1
    local check
    check="$(env -u LAB_MAX_MODEL_LEN FINAL_ENV="$FINAL_ENV_FILE" bash serve_champion.sh "$ROLE" --check 2>&1 | tail -n 1)"
    case "$check" in CHAMPION_CHECK_OK*) say "$check" ;; *) say "HALT pre-boot check refused: $check (fix champion.staged or its files, then rerun)"; return 1 ;; esac
    choose_seed
    say "BOOT $NAME (about 15 min)"
    env -u LAB_MAX_MODEL_LEN FINAL_ENV="$FINAL_ENV_FILE" bash serve_champion.sh "$ROLE" < /dev/null > "$Q/champion-boot-$ROLE.console" 2>&1
    if was_serving && is_running; then
        date > "$WANTED"
        say "SERVING $(grep ' BOOTED ' "$RUN/status" | tail -n 1 | cut -c25-120)"
        record_seed_leaf
        pin_acc || say "ACC_PIN_INCOMPLETE $NAME serves with its AccThreadPool threads unpinned; once fixed: champion_boot.sh $ROLE --pin-acc"
        keep_seed_copy
        return 0
    fi
    say "HALT $NAME did not come up: $(tail -n 2 "$RUN/status" 2> /dev/null | tr '\n' ' ' | cut -c1-400) (console: $Q/champion-boot-$ROLE.console)"
    return 1
}

# A person (or the boot unit) asked for the replica: forget earlier automatic tries.
start_by_hand() {
    rm -f "$TRIES" "$HALTED" "$STOPPED_ON_PURPOSE"
    start_replica
}

recent_tries() {
    [ -f "$TRIES" ] || { echo 0; return 0; }
    awk -v since="$(( $(date +%s) - RECOVER_WINDOW_S ))" '$1 >= since' "$TRIES" | wc -l
}

# Keep what the stopped replica left, so the cause can be read later: its exit state, the Synapse logs inside the
# container, and the card driver's kernel lines. The folder moves aside with the run folder.
save_death_evidence() {
    local out="$RUN/death-evidence"
    mkdir -p "$out" || return 0
    docker inspect -f 'exit={{.State.ExitCode}} oom={{.State.OOMKilled}} finished={{.State.FinishedAt}}' "$NAME" > "$out/container-state.txt" 2>&1
    docker cp "$NAME:/var/log/habana_logs" "$out/habana_logs" > /dev/null 2>&1
    journalctl -k --since "-30min" --no-pager 2> /dev/null | grep -i habanalabs | tail -n 300 > "$out/kernel-habanalabs.txt"
    return 0
}

give_up_once() {
    [ -e "$HALTED" ] && return 0
    date > "$HALTED"
    say "RECOVER_HALT $NAME: $RECOVER_MAX_TRIES automatic restarts in the last $((RECOVER_WINDOW_S / 60)) min did not keep it serving; read $LOG, fix the cause, then start it by hand: champion_boot.sh $ROLE"
}

# What the restart timer runs every minute. Says nothing unless it acts; always succeeds, the log carries the outcome.
docker_is_up() { [ "$(systemctl is-active docker 2> /dev/null)" = active ]; }

stopped_at() {   # epoch seconds at which the container stopped
    date -d "$(docker inspect -f '{{.State.FinishedAt}}' "$NAME" 2> /dev/null)" +%s 2> /dev/null || date +%s
}

stop_events() {   # the actions docker recorded for the container around its stop, one per line
    local id
    id="$(docker inspect -f '{{.Id}}' "$NAME" 2> /dev/null)" || return 0
    docker events --since "$(( $(stopped_at) - EVENT_LOOKBACK_S ))" --until "$(date +%s)" \
        --filter "container=$id" --format '{{.Action}}' 2> /dev/null
}

stopped_before_docker_started() {
    local docker_started
    docker_started="$(systemctl show -p ActiveEnterTimestamp --value --timestamp=unix docker 2> /dev/null | tr -d '@')"
    [ -n "$docker_started" ] && [ "$(stopped_at)" -le "$docker_started" ]
}

# Why a replica that was serving is not running. Docker records a "kill" action only when it is asked to send a signal
# (docker stop, docker kill, docker rm -f); a container that ends by itself leaves only "die".
#   guard    its own memory watcher took it down (the watcher leaves an ABORT file; it uses docker stop itself)
#   own      it ended by itself, was killed for memory, or docker restarted under it
#   hand     someone stopped or removed it through docker
#   unknown  docker no longer has the record
stop_reason() {
    if compgen -G "$RUN/graded/*/ABORT" > /dev/null; then echo guard; return 0; fi
    exists || { echo hand; return 0; }
    if [ "$(docker inspect -f '{{.State.OOMKilled}}' "$NAME" 2> /dev/null)" = true ]; then echo own; return 0; fi
    local events
    events="$(stop_events)"
    if grep -qx kill <<< "$events"; then echo hand; return 0; fi
    if grep -qx die <<< "$events"; then echo own; return 0; fi
    if stopped_before_docker_started; then echo own; return 0; fi
    echo unknown
}

stopped_by_hand() {
    rm -f "$WANTED"
    [ -d "$RUN" ] && touch "$RELEASE"
    say "STOPPED_BY_HAND $NAME was stopped through docker; it stays down and its quad is free for another container (release file touched). Start it again by hand: champion_boot.sh $ROLE"
}

left_down_unsure() {
    rm -f "$WANTED"
    say "RECOVER_UNSURE $NAME is not running and docker no longer has the record of how it stopped; left down, its quad holder untouched. If it should be serving, start it by hand: champion_boot.sh $ROLE"
}

restart_replica() {
    local reason="$1"
    if [ "$(recent_tries)" -ge "$RECOVER_MAX_TRIES" ]; then give_up_once; return 0; fi
    rm -f "$HALTED"
    date +%s >> "$TRIES"
    say "RECOVER $NAME is meant to be serving and is not running ($reason); restarting it (try $(recent_tries) of $RECOVER_MAX_TRIES per $((RECOVER_WINDOW_S / 60)) min; what it left: $RUN.run-$STAMP/death-evidence)"
    save_death_evidence
    start_replica
    return 0
}

recover_replica() {
    docker_is_up || return 0
    [ -e "$STOPPED_ON_PURPOSE" ] && return 0   # --stop took it down (or is still doing so): nothing to judge, nothing to mark
    if is_running; then
        if was_serving && [ ! -e "$WANTED" ]; then date > "$WANTED"; fi
        return 0
    fi
    [ -e "$WANTED" ] || return 0
    # A run folder without SERVING is a restart that failed: there is no stop to judge, only another try to make.
    was_serving || { restart_replica "the last start failed"; return 0; }
    local reason
    reason="$(stop_reason)"
    case "$reason" in
        hand) stopped_by_hand ;;
        unknown) left_down_unsure ;;
        guard) restart_replica "its memory guard took it down" ;;
        *) restart_replica "it went down by itself" ;;
    esac
    return 0
}

# One start per replica at a time: the timer, the boot unit and a person may all ask at once. flock closes the lock
# file before running the script again, so the holder and watcher processes a boot leaves behind do not keep the lock.
run_locked() {
    CHAMPION_BOOT_LOCKED=1 flock -n -o -E "$LOCK_BUSY" "$BOOT_LOCK" bash "${BASH_SOURCE[0]}" "$ROLE" "$MODE"
}

case "$MODE" in
    --plan) plan ;;
    --stop) stop_replica ;;
    --pin-acc) pin_running; exit $? ;;
    --keep-seed) keep_seed_running; exit $? ;;
    start|--recover)
        if [ -z "${CHAMPION_BOOT_LOCKED:-}" ]; then
            run_locked; status=$?
            [ "$status" -eq "$LOCK_BUSY" ] || exit "$status"
            [ "$MODE" = start ] && say "BOOT_IN_PROGRESS another start of $ROLE is running; left alone"
            exit 0
        fi
        if [ "$MODE" = start ]; then start_by_hand; else recover_replica; fi ;;
esac
