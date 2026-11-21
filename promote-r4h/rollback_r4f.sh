#!/bin/bash
# promote-r4h/rollback_r4f.sh: the champion's rollback (coordinator 10:2x): back to the r4f two-replica serving install, image
# aabd010a + serve-levers 9d1724ad, r0 on :18274 (quad A) then r1 on :18275 (quad B), through c1d-r4f-serve/serve_r4f.sh, the
# 09:00 install's launcher. RC56 is never run again: its drop-ins stay pinned and no unit is started (common.sh).
# Steps: stop the production units (stop only) and rename any r4h unit drop-in aside; per role, stop the champion's lab container and
# release its quad holder; a role whose r4f container still serves healthy is kept; otherwise the exited r4f container is renamed
# aside, glm53flash-<role> is RC56's stopped container (the lab source, as at 09:00), r4f boots, and image, /health and SMOKE_OK are
# checked. Ends R4F_SERVE_RESTORED; a role that does not come back raises an ALARM and the other role still runs.
# usage: setsid nohup ./rollback_r4f.sh [--dry-run] </dev/null >/dev/null 2>&1 &   log: logs/rollback_r4f-latest.log
set -euo pipefail
KIT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "$KIT_DIR/common.sh"

rollback_preflight() {
    check_rc56_untouched
    $DOCKER image inspect "$R4F_IMAGE" > /dev/null 2>&1 || refuse "r4f image $R4F_IMAGE is missing"
    echo "$R4F_LEVERS_SHA  $SERVE_DIR/serve-levers.env" | sha256sum --quiet -c - 2> /dev/null \
        || refuse "$SERVE_DIR/serve-levers.env is not ${R4F_LEVERS_SHA:0:8}"
    [ -f "$SERVE_SCRIPT" ] || refuse "$SERVE_SCRIPT is missing"
    log "ROLLBACK_PREFLIGHT_OK rc56 drop-ins untouched, r4f ${R4F_IMAGE:7:12}, serve-levers ${R4F_LEVERS_SHA:0:8}"
}

stop_units() {  # stop only (never start); r4h unit drop-ins renamed aside, so the units stay stopped on RC56's pinned drop-in
    local role path unit
    $SUDO $SYSTEMCTL stop "$TARGET" glm53flash-r0.service glm53flash-r1.service
    for role in "${ROLES[@]}"; do
        path="$UNIT_DIR/glm53flash-$role.service.d/$R4H_DROPIN"
        [ ! -e "$path" ] || { $SUDO mv -- "$path" "$path.disabled-$STAMP"; log "r4h drop-in aside: $path.disabled-$STAMP"; }
    done
    $SUDO $SYSTEMCTL daemon-reload
    for unit in "$TARGET" glm53flash-r0.service glm53flash-r1.service; do
        ! $SYSTEMCTL is-active --quiet "$unit" || refuse "$unit is still active after the stop; nothing was booted"
    done
    log "UNITS_STOPPED $TARGET r0 r1 inactive (RC56 not started)"
}

r4f_serving() {  # $1 role: its r4f container runs and answers /health
    container_running "$(lab_name "$1" r4f)" && $CURL -fsS -m 10 "http://127.0.0.1:${PORT[$1]}/health" > /dev/null
}

prepare_role() {  # $1 role: the exited r4f container aside; glm53flash-<role> = RC56's stopped container (the lab source)
    local role="$1" lab unit="glm53flash-$1" image
    lab=$(lab_name "$role" r4f)
    if [ -n "$(container_image "$lab")" ]; then
        ! container_running "$lab" || stop_and_release "$lab" "$role" || return 1
        $DOCKER rename "$lab" "$lab-before-rollback-$STAMP"
        log "parked $lab -> $lab-before-rollback-$STAMP"
    fi
    image=$(container_image "$unit")
    if [ -n "$image" ] && [ "$image" != "$RC56_IMAGE" ]; then
        $DOCKER rename "$unit" "$unit-r4h-before-rollback-$STAMP"; log "parked $unit (image $image) -> $unit-r4h-before-rollback-$STAMP"
        image=""
    fi
    if [ -z "$image" ]; then
        [ -n "$(container_image "$unit-RC56-before-c1d-r4h")" ] || { log "no RC56 $unit container to boot $role from (lab source)"; return 1; }
        $DOCKER rename "$unit-RC56-before-c1d-r4h" "$unit"; log "restored the lab source $unit (stopped; never started)"
    fi
    ! container_running "$unit" || { log "$unit is running; it must stay stopped"; return 1; }
}

restore_role() {  # $1 role: rc 0 when r4f serves it verified
    local role="$1"
    [ -z "$CHAMPION_CUT" ] || stop_and_release "$(lab_name "$role" "$CHAMPION_CUT")" "$role" || return 1   # none staged: none to stop
    if r4f_serving "$role"; then log "r4f still serves $role healthy; kept"; return 0; fi
    prepare_role "$role" || return 1
    launch_and_wait "$SERVE_SCRIPT" "$role" "serve-r4f-$role" > /dev/null || return 1
    [ "$(container_image "$(lab_name "$role" r4f)")" = "$R4F_IMAGE" ] || { log "GATE_FAIL $role: not the r4f image"; return 1; }
    health_and_smoke "$role" || return 1
    log "VERIFIED $role r4f ${R4F_IMAGE:7:12} :${PORT[$role]} SMOKE_OK"
}

rollback_main() {
    case "$#:${1:-}" in 0:) DRY_RUN=0 ;; 1:--dry-run) DRY_RUN=1 ;; *) refuse "usage: rollback_r4f.sh [--dry-run]" ;; esac
    rollback_preflight
    if [ "$DRY_RUN" = 1 ]; then
        log "ROLLBACK_R4F_DRYRUN_OK (would stop $TARGET r0 r1, stop+release the ${CHAMPION_CUT:-<none staged>} champion containers, boot r4f r0 then r1)"; return 0
    fi
    stop_units
    local role failed=()
    for role in "${ROLES[@]}"; do
        restore_role "$role" && continue
        failed+=("$role")
        alarm "r4f $role did not come back (the rollback continues with the other replica); RC56 stays stopped"
    done
    [ "${#failed[@]}" = 0 ] || { alarm "R4F ROLLBACK INCOMPLETE: replicas down: ${failed[*]}"; return 1; }
    log "R4F_SERVE_RESTORED r0 :18274 r1 :18275 (r4f ${R4F_IMAGE:7:12}, serve-levers ${R4F_LEVERS_SHA:0:8}; RC56 untouched)"
}

[ "${BASH_SOURCE[0]}" != "$0" ] || {
    mkdir -p "$KIT_DIR/logs"
    LOG="$KIT_DIR/logs/rollback_r4f-$STAMP.log"
    ln -sfn "$LOG" "$KIT_DIR/logs/rollback_r4f-latest.log"
    rollback_main "$@" > "$LOG" 2>&1
}
