#!/bin/bash
# promote-r4h/common.sh: shared by promote_r4h.sh and rollback_r4f.sh (sourced, never run). Both replicas run through the lab-launcher
# path the 09:00 r4f serving install used (diag-r4e/r4e_run.sh prepare + boot, lab_serve.py, a quad holder per replica): r0 on quad A
# (modules 4-7, :18274), r1 on quad B (modules 0-3, :18275). RC56 is never run again (owner): its drop-ins stay the pinned bytes and
# nothing here starts a unit. Nothing is deleted: containers are stopped (never rm) or renamed aside.

SERVE_DIR=/mnt/data1/declane-out/c1d-r4f-serve
R4F_IMAGE=sha256:aabd010a1851eb65dee18e1bb486c3a24eaa2c57b593cdafe6470784609a81c8
RC56_IMAGE=sha256:1481bb99fce076776c9e88604219726df33a903514015aa13fb0efc2350c8bbc
R4F_LEVERS_SHA=9d1724ad5013fc5b6b7e241f3a194811691eafd32b0303b1a780fade23e3b0db
R4H_DROPIN=zzzzzzzzzz-claude-c1d-r4h.conf   # a unit drop-in an r4h install might have left; rollback renames it aside
RC56_DROPIN=zzzzzzz-claude-RC56.conf
declare -A RC56_DROPIN_SHA=([r0]=2e2b6b11e115a48116ae795162e879c24300ddd9973d217133725333a6e21e36
                            [r1]=cd6bb666d333c0312adf69f06963d92bd7336b6f0c23e40eab4d6f59366a889a)
declare -A PORT=([r0]=18274 [r1]=18275)
declare -A QUAD=([r0]=A [r1]=B)
ROLES=(r0 r1)
TARGET=glm53flash-production.target
BOOT_BUDGET_S=1500     # per replica (owner serving gates): boot <= 1500 s, warmup <= 300 s
WARMUP_BUDGET_S=300
WAIT_LIMIT_S=2400      # how long a launcher run may take to print SERVING or DONE (serve_*.sh's own deadline still applies)
HOLDER_WAIT_S=120      # a released quad holder exits within its 5 s poll

# Test hooks: only with GLM53_PROMOTE_TEST=1 (dry runs and CPU tests); a leftover hook in a real run refuses.
TEST_HOOKS=(UNIT_DIR SERVE_SCRIPT CHAMPION_SERVE SMOKE_SCRIPT QUEUE_DIR DOCKER SUDO SYSTEMCTL CURL SS POLL_S HOLDER_DIR STAGED)
for hook in "${TEST_HOOKS[@]}"; do
    [ -z "${!hook+x}" ] || [ "${GLM53_PROMOTE_TEST:-0}" = 1 ] \
        || { echo "REFUSED: test hook $hook is set without GLM53_PROMOTE_TEST=1; unset it for a real run" >&2; exit 2; }
done
UNIT_DIR=${UNIT_DIR:-/etc/systemd/system}
SERVE_SCRIPT=${SERVE_SCRIPT:-$SERVE_DIR/serve_r4f.sh}
CHAMPION_SERVE=${CHAMPION_SERVE:-$KIT_DIR/serve_champion.sh}
SMOKE_SCRIPT=${SMOKE_SCRIPT:-$SERVE_DIR/serve_smoke.py}
QUEUE_DIR=${QUEUE_DIR:-/mnt/data1/graded-out/queue}
DOCKER=${DOCKER:-docker}
SUDO=${SUDO-sudo -n}   # unset = sudo -n; tests set it empty
SYSTEMCTL=${SYSTEMCTL:-systemctl}
CURL=${CURL:-curl}
SS=${SS:-ss}
POLL_S=${POLL_S:-15}
HOLDER_DIR=${HOLDER_DIR:-/tmp}
STAGED=${STAGED:-$KIT_DIR/champion.staged}
# the champion is a parameter: promote_r4h.sh stage --image <repo:tag-CUT> writes CHAMPION_CUT; unstaged = empty (no champion
# container to stop on rollback, and promote refuses)
CHAMPION_CUT=$(sed -n 's/^CHAMPION_CUT=//p' "$STAGED" 2>/dev/null || true)

stamp() { TZ=America/New_York date '+%Y%m%dT%H%M%S'; }
log() { echo "[$(TZ=America/New_York date '+%Y-%m-%d %H:%M:%S %Z')] $*"; }
refuse() { log "REFUSED: $*"; exit 2; }
alarm() { log "ALARM: $*"; logger -p user.crit -t glm53-promote-r4h "ALARM: $*" 2>/dev/null || true; }
STAMP=$(stamp)

container_image() { $DOCKER inspect -f '{{.Image}}' "$1" 2>/dev/null || true; }
container_running() { [ "$($DOCKER inspect -f '{{.State.Running}}' "$1" 2>/dev/null || true)" = true ]; }
lab_name() { echo "claude-flash-launch-glm53flash-$1-$2"; }   # $1 role, $2 cut (r4f or the champion's)

check_rc56_untouched() {  # RC56's drop-ins are the bytes pinned here (never edited, never started)
    local role path
    for role in "${ROLES[@]}"; do
        path="$UNIT_DIR/glm53flash-$role.service.d/$RC56_DROPIN"
        [ -f "$path" ] || refuse "$path is missing: RC56's drop-in must stay untouched"
        [ "$(sha256sum "$path" | cut -d' ' -f1)" = "${RC56_DROPIN_SHA[$role]}" ] || refuse "$path changed: RC56's drop-in must stay untouched"
    done
}

release_file() {  # $1 container name: the release file its launcher's SERVING line names ("then touch <file>"), if any
    grep -a -h -F "SERVING name=$1 " "$QUEUE_DIR"/*/status 2>/dev/null | tail -n 1 | sed -n 's/.*then touch \([^ ]*\).*/\1/p'
}

holder_alive() {  # $1 quad: its holder file names a live process
    local pid
    pid=$(cat "$HOLDER_DIR/claude-flash-quad$1.holder" 2>/dev/null || true)
    [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null
}

stop_and_release() {  # $1 container, $2 role: docker stop (never rm), then its launcher's release file, then the quad holder gone
    local name="$1" role="$2" release start=$SECONDS
    container_running "$name" || return 0
    $DOCKER stop -t 60 "$name" > /dev/null
    release=$(release_file "$name")
    if [ -n "$release" ]; then touch "$release"; log "stopped $name; released its quad ${QUAD[$role]} holder ($release)"
    else log "stopped $name; its launcher named no release file"; fi
    while holder_alive "${QUAD[$role]}"; do
        [ $((SECONDS - start)) -lt "$HOLDER_WAIT_S" ] || { log "HOLDER_STILL_ALIVE quad ${QUAD[$role]} after ${HOLDER_WAIT_S}s"; return 1; }
        sleep "$POLL_S"
    done
}

port_holders() {  # $1 role: every running container whose argv or port bindings name its port
    local name ports   # captured first: grep -q on a pipe would SIGPIPE docker and pipefail would drop the match
    while IFS= read -r name; do
        [ -n "$name" ] || continue
        ports=$($DOCKER inspect -f '{{join .Args " "}} {{json .HostConfig.PortBindings}}' "$name" 2>/dev/null || true)
        grep -q -w -- "${PORT[$1]}" <<< "$ports" && echo "$name"
    done < <($DOCKER ps --format '{{.Names}}' 2>/dev/null || true)
    return 0
}

check_port_owner() {  # $1 role: its port is free or held only by its r4f container; anything else refuses, named, before any stop
    local role="$1" holder expected
    expected=$(lab_name "$role" r4f)
    holder=$(port_holders "$role" | grep -v -x -F -- "$expected" | tr '\n' ' ' || true)
    [ -z "$holder" ] || refuse "port :${PORT[$role]} ($role) is held by ${holder% }, not $expected; stop it by hand first"
    port_holders "$role" | grep -q -x -F -- "$expected" && return 0
    [ -z "$($SS -H -ltn "sport = :${PORT[$role]}" 2>/dev/null || true)" ] \
        || refuse "port :${PORT[$role]} ($role) has a listener no running container names; find it (sudo ss -ltnp) and stop it by hand first"
}

status_count() { cat "$QUEUE_DIR/$1/status" 2>/dev/null | wc -l; }

launch_and_wait() {  # $1 script, $2 role, $3 run dir name: run it in its own session; print its new BOOTED line; rc 1 on DONE/timeout
    local before line booted="" start=$SECONDS
    before=$(status_count "$3")
    setsid nohup bash "$1" "$2" < /dev/null > "$QUEUE_DIR/$3-$STAMP.console" 2>&1 &
    log "LAUNCH $2 via $1 (console $QUEUE_DIR/$3-$STAMP.console)" >&2
    while [ $((SECONDS - start)) -lt "$WAIT_LIMIT_S" ]; do
        while IFS= read -r line; do
            case "$line" in *" BOOTED "*) booted="$line" ;; *" SERVING "*) log "$line" >&2; echo "$booted"; return 0 ;;
                             *" DONE verdict="*) log "$line" >&2; return 1 ;; esac
        done < <(tail -n +"$((before + 1))" "$QUEUE_DIR/$3/status" 2>/dev/null)
        sleep "$POLL_S"
    done
    log "WAIT_TIMEOUT $2 after ${WAIT_LIMIT_S}s" >&2; return 1
}

health_and_smoke() {  # $1 role
    $CURL -fsS -m 10 "http://127.0.0.1:${PORT[$1]}/health" > /dev/null || { log "GATE_FAIL $1: /health on ${PORT[$1]}"; return 1; }
    python3 -I "$SMOKE_SCRIPT" "${PORT[$1]}" | tail -n 1 | grep -q -x SMOKE_OK || { log "GATE_FAIL $1: smoke"; return 1; }
}
