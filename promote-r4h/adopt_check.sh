#!/bin/bash
# promote-r4h/adopt_check.sh: read-only proof that the replicas already serving ARE the staged champion, so a promotion can adopt
# them with no restart (2026-11-22: both replicas were booted with the final image by a trial kit, under the cut's own names).
# promote_r4h.sh cannot do this: it accepts only r4f on :18274/:18275 (check_port_owner) and would refuse. champion_boot.sh --plan
# only asks docker whether a container of the staged name runs; this checks what a restart by the recover timer would change:
# per role, the container of the staged cut runs, its image id is CHAMPION_IMAGE_SHA, every staged lever is live
# (promote_r4h.sh value_drift), its KV block count is the one serve_champion.sh boots, and /health answers.
# Starts, stops, renames and writes nothing (--launcher-check: the launcher's pre-boot check, a created container never started).
# usage: adopt_check.sh [--smoke] [--launcher-check]    ends ADOPT_OK (rc 0) or ADOPT_REFUSED (rc 1); a bad staged file refuses (rc 2)
set -euo pipefail
KIT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=promote_r4h.sh
source "$KIT_DIR/promote_r4h.sh"   # common.sh, the rollback's and the promote's functions (their mains run only when executed)

SMOKE=0
LAUNCHER_CHECK=0
for option in "$@"; do
    case "$option" in --smoke) SMOKE=1 ;; --launcher-check) LAUNCHER_CHECK=1 ;;
        *) refuse "usage: adopt_check.sh [--smoke] [--launcher-check] (unknown option $option)" ;; esac
done
KIT_KV_BLOCKS=$(sed -n 's/^export C1D_KV_BLOCKS=\([0-9]*\) .*/\1/p' "$KIT_DIR/serve_champion.sh")
[ -n "$KIT_KV_BLOCKS" ] || refuse "$KIT_DIR/serve_champion.sh exports no C1D_KV_BLOCKS: this is not the seeded kit"

final_env_ok() {  # the base config a restart applies is the staged bytes (serve_champion.sh refuses a boot otherwise)
    local file sha
    file=$(staged_value CHAMPION_FINAL_ENV); sha=$(staged_value CHAMPION_FINAL_ENV_SHA)
    [ -n "$file" ] && [ -n "$sha" ] || refuse "$STAGED names no CHAMPION_FINAL_ENV: re-run promote_r4h.sh stage ... --final-env <file>"
    echo "$sha  $file" | sha256sum --quiet -c - 2> /dev/null || refuse "$file is not the staged ${sha:0:8}"
}

live_kv_blocks() {  # $1 container: its --num-gpu-blocks-override (captured first: pipefail would end the script on a missing container)
    local args
    args=$($DOCKER inspect -f '{{join .Args " "}}' "$1" 2> /dev/null || true)
    sed -n 's/.*--num-gpu-blocks-override[= ]\([0-9]*\).*/\1/p' <<< "$args"
}

adopt_role() {  # $1 role: rc 1 when its serving container is not the staged champion a restart would boot
    local role="$1" name image kv failed=0
    name=$(lab_name "$role" "$CHAMPION_CUT")
    container_running "$name" || { log "ADOPT_FAIL $role: $name is not running (adoption needs it serving; boot it with champion_boot.sh $role)"; return 1; }
    image=$(container_image "$name")
    [ "$image" = "$(staged_value CHAMPION_IMAGE_SHA)" ] \
        || { log "ADOPT_FAIL $role: $name runs ${image:7:12}, not the staged $(staged_value CHAMPION_IMAGE_SHA | cut -c8-19)"; failed=1; }
    value_drift "$name" "$(staged_value CHAMPION_LEVERS)" || { log "ADOPT_FAIL $role: live parity"; failed=1; }
    kv=$(live_kv_blocks "$name")
    [ "$kv" = "$KIT_KV_BLOCKS" ] || { log "ADOPT_FAIL $role: $name serves ${kv:-<none>} KV blocks; a restart from this kit boots $KIT_KV_BLOCKS"; failed=1; }
    if [ "$SMOKE" = 1 ]; then health_and_smoke "$role" || failed=1
    else $CURL -fsS -m 10 "http://127.0.0.1:${PORT[$role]}/health" > /dev/null || { log "ADOPT_FAIL $role: /health on ${PORT[$role]}"; failed=1; }; fi
    [ "$LAUNCHER_CHECK" = 0 ] || launcher_check "$role"
    [ "$failed" = 0 ] || return 1
    log "ADOPT_ROLE_OK $role $name image ${image:7:12} VALUE-DRIFT 0 kv_blocks $kv :${PORT[$role]}"
}

adopt_main() {
    staged_pins_ok
    final_env_ok
    local role failed=()
    for role in "${ROLES[@]}"; do adopt_role "$role" || failed+=("$role"); done
    if [ "${#failed[@]}" != 0 ]; then
        log "ADOPT_REFUSED ${failed[*]}: not the staged champion; leave champion.staged as it was (RUNBOOK.md rollback (v)) or boot the role from this kit"
        return 1
    fi
    log "ADOPT_OK cut=$CHAMPION_CUT image $(staged_value CHAMPION_IMAGE_SHA | cut -c8-19) levers $(staged_value CHAMPION_LEVERS_SHA | cut -c1-8): both replicas already serve it; nothing to restart"
}

adopt_main
