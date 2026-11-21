#!/bin/bash
# promote-r4h/champion_boot.sh: keep one champion serving replica up (owner 2026-11-11: the champions "stay up";
# owner 2026-11-13: a replica that stops on its own restarts itself).
# Run by glm53-champion-r0|r1.service at host boot and by glm53-champion-recover@r0|r1.timer every minute; safe by hand.
# It serves whatever champion.staged names, through serve_champion.sh, exactly as a manual launch does.
# usage: champion_boot.sh r0|r1             start the replica if it is not running (a running one is left alone)
#        champion_boot.sh r0|r1 --plan      print what a start would do, change nothing
#        champion_boot.sh r0|r1 --stop      take the replica down on purpose: docker stop, free its quad; it stays down
#        champion_boot.sh r0|r1 --recover   restart the replica if it went down by itself (silent when there is nothing to do)
# To take a quad away from a serving replica, use --stop or a plain "docker stop" (or "docker rm -f"): docker records
# the signal it was asked to send, so --recover sees the stop was made by hand, leaves the replica down and frees its
# quad within a minute. Only a replica that went down by itself (card error, memory guard, killed for memory, docker
# restarted under it) is restarted. Bring a replica stopped by hand back with: champion_boot.sh r0|r1
# A replica that stopped on its own (card error, memory guard) leaves its quad holder alive; a start frees that holder
# first, the same way --stop does, so the boot does not wait on a quad nobody is using.
# Deletes nothing: a stopped container of the same name and an old run folder are renamed with a time stamp.
set -u
KIT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROLE="${1:?usage: champion_boot.sh r0|r1 [--plan|--stop|--recover]}"
MODE="${2:-start}"
case "$ROLE" in r0|r1) ;; *) echo "HALT champion_boot.sh: role $ROLE is not r0 or r1" >&2; exit 2 ;; esac
case "$MODE" in start|--plan|--stop|--recover) ;;
    *) echo "HALT champion_boot.sh: mode $MODE is not --plan, --stop or --recover" >&2; exit 2 ;; esac
Q="${QUEUE_DIR:-/mnt/data1/graded-out/queue}"
STAGED="$KIT_DIR/champion.staged"
CUT="$(sed -n 's/^CHAMPION_CUT=//p' "$STAGED" 2> /dev/null)"
FINAL_ENV_FILE="$(sed -n 's/^CHAMPION_FINAL_ENV=//p' "$STAGED" 2> /dev/null)"
[ -n "$CUT" ] && [ -n "$FINAL_ENV_FILE" ] || {
    echo "HALT champion_boot.sh: $STAGED names no CHAMPION_CUT or CHAMPION_FINAL_ENV; stage a champion with promote_r4h.sh stage --final-env" >&2; exit 2; }
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
    say "BOOT $NAME (about 15 min)"
    env -u LAB_MAX_MODEL_LEN FINAL_ENV="$FINAL_ENV_FILE" bash serve_champion.sh "$ROLE" < /dev/null > "$Q/champion-boot-$ROLE.console" 2>&1
    if was_serving && is_running; then
        date > "$WANTED"
        say "SERVING $(grep ' BOOTED ' "$RUN/status" | tail -n 1 | cut -c25-120)"; return 0
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
    start|--recover)
        if [ -z "${CHAMPION_BOOT_LOCKED:-}" ]; then
            run_locked; status=$?
            [ "$status" -eq "$LOCK_BUSY" ] || exit "$status"
            [ "$MODE" = start ] && say "BOOT_IN_PROGRESS another start of $ROLE is running; left alone"
            exit 0
        fi
        if [ "$MODE" = start ]; then start_by_hand; else recover_replica; fi ;;
esac
