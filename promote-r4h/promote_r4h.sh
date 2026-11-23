#!/bin/bash
# promote-r4h/promote_r4h.sh: install the staged champion (stage --image names it; r4h5 expected) on both serving replicas (coordinator 11:0x), through the lab-launcher path the
# 09:00 r4f install used (serve_champion.sh). Staggered: r1 starts only after r0 is fully BOOTED, SERVING and through every gate.
# Per replica: the r4f container on its port is stopped (docker stop, never rm) and its quad holder released; the champion boots; gates
# = boot <= 1500 s (the launcher's BOOTED boot_s), warmup <= 300 s (the workers' "Warmup finished in N secs"), live parity against the
# staged levers file (live_parity --intended's VALUE-DRIFT rule: every intended key at its value in docker inspect .Config.Env), /health
# and SMOKE_OK. Any failure: the champion container is stopped and released, then rollback_r4f.sh's rollback runs (r4f aabd010a +
# serve-levers 9d1724ad on both replicas; RC56 never). Ends C1D_CHAMPION_PROMOTED cut=<CUT>.
# usage: promote_r4h.sh stage --image $CHAMPION_REPO:<date>-<CUT> --image-sha <sha256:...> --levers <file> --gates <file>
#        (writes champion.staged: the image, the levers file and the launcher's gate settings, each pinned by sha)
#        setsid nohup ./promote_r4h.sh [--dry-run] </dev/null >/dev/null 2>&1 &   log: logs/promote_r4h-latest.log
# Only for an r4f -> champion promotion: with a champion already serving under its cut, check_port_owner refuses (nothing is
# stopped). To take over replicas that already serve the staged image, adopt them instead: adopt_check.sh (RUNBOOK.md).
set -euo pipefail
KIT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=rollback_r4f.sh
source "$KIT_DIR/rollback_r4f.sh"   # common.sh, and the rollback's functions (its main runs only when executed)
CHAMPION_REPO=claude-flash-dual-graphs-c1d

staged_value() { sed -n "s/^$1=//p" "$STAGED"; }

stage() {  # --image TAG --image-sha SHA --levers FILE --gates FILE: pin all four into champion.staged; the cut is the tag's -suffix
    local usage="usage: promote_r4h.sh stage --image $CHAMPION_REPO:<date>-<cut> --image-sha <sha256:...> --levers <file> --gates <file> [--final-env <file>]"
    [ "$#" = 8 ] || { [ "$#" = 10 ] && [ "$9" = --final-env ]; } || refuse "$usage"
    [ "$1" = --image ] && [ "$3" = --image-sha ] && [ "$5" = --levers ] && [ "$7" = --gates ] || refuse "$usage"
    [ "$#" = 8 ] || [ -f "${10}" ] || refuse "final env file ${10} is missing"
    local image="$2" cut="${2##*-}"
    [[ "$image" == "$CHAMPION_REPO":*-* ]] || refuse "$image is not a $CHAMPION_REPO:<date>-<cut> tag"
    [[ "$cut" =~ ^[a-z0-9]+$ ]] && [ "$cut" != r4f ] || refuse "cut '$cut' of $image is not a champion (r4f is the rollback)"
    [ "$($DOCKER image inspect -f '{{.Id}}' "$image" 2>/dev/null || true)" = "$4" ] || refuse "$image is not $4"
    [ -f "$6" ] || refuse "levers file $6 is missing"
    [ -f "$8" ] || refuse "gates file $8 is missing"
    printf 'CHAMPION_CUT=%s\nCHAMPION_IMAGE=%s\nCHAMPION_IMAGE_SHA=%s\nCHAMPION_LEVERS=%s\nCHAMPION_LEVERS_SHA=%s\nCHAMPION_GATES=%s\nCHAMPION_GATES_SHA=%s\n' \
        "$cut" "$image" "$4" "$(readlink -f "$6")" "$(sha256sum "$6" | cut -d' ' -f1)" \
        "$(readlink -f "$8")" "$(sha256sum "$8" | cut -d' ' -f1)" > "$STAGED.new"
    # the base config the launcher applies before the levers; serve_champion.sh refuses a staged file without it
    [ "$#" = 8 ] || printf 'CHAMPION_FINAL_ENV=%s\nCHAMPION_FINAL_ENV_SHA=%s\n' "$(readlink -f "${10}")" "$(sha256sum "${10}" | cut -d' ' -f1)" >> "$STAGED.new"
    mv -- "$STAGED.new" "$STAGED"
    log "STAGED $(tr '\n' ' ' < "$STAGED")"
}

staged_pins_ok() {  # champion.staged names a cut, its tag is still the pinned image, levers and gates are the pinned bytes and agree
    [ -f "$STAGED" ] || refuse "$STAGED is missing: run promote_r4h.sh stage first"
    [ -n "$CHAMPION_CUT" ] || refuse "$STAGED names no CHAMPION_CUT: re-run promote_r4h.sh stage --image ..."
    [ "$($DOCKER image inspect -f '{{.Id}}' "$(staged_value CHAMPION_IMAGE)" 2>/dev/null || true)" = "$(staged_value CHAMPION_IMAGE_SHA)" ] \
        || refuse "$(staged_value CHAMPION_IMAGE) no longer names the staged $(staged_value CHAMPION_IMAGE_SHA)"
    local pinned
    for pinned in LEVERS GATES; do
        [ -n "$(staged_value "CHAMPION_$pinned")" ] || refuse "$STAGED names no CHAMPION_$pinned: re-run promote_r4h.sh stage ... --gates <file>"
        echo "$(staged_value "CHAMPION_${pinned}_SHA")  $(staged_value "CHAMPION_$pinned")" | sha256sum --quiet -c - 2> /dev/null \
            || refuse "$(staged_value "CHAMPION_$pinned") is not the staged $(staged_value "CHAMPION_${pinned}_SHA" | cut -c1-8)"
    done
    check_pin_agreement
}

promote_preflight() {
    staged_pins_ok
    [ -f "$CHAMPION_SERVE" ] || refuse "$CHAMPION_SERVE is missing"
    rollback_preflight   # the rollback must be able to run before anything changes
    local role
    for role in "${ROLES[@]}"; do check_port_owner "$role"; done   # only r4f may hold :18274/:18275; anything else refuses, named
    for role in "${ROLES[@]}"; do launcher_check "$role"; done     # the launcher's own pre-boot refusals, so a dry-run OK boots
    log "PROMOTE_PREFLIGHT_OK $(staged_value CHAMPION_IMAGE_SHA | cut -c8-19) levers $(staged_value CHAMPION_LEVERS_SHA | cut -c1-8)"
}

check_pin_agreement() {  # the gates' R4E_PIN_GATE and the levers' HPU_GLM53_ATTN_PIN agree (the launcher refuses either mismatch)
    local gate pin_set=no
    gate=$(sed -n "s/^R4E_PIN_GATE=[\"']\{0,1\}\([a-z]*\).*/\1/p" "$(staged_value CHAMPION_GATES)" | tail -n 1)
    [ -n "$gate" ] || refuse "$(staged_value CHAMPION_GATES) names no R4E_PIN_GATE (on or off): set it beside the levers' pin"
    intended_lines "$(staged_value CHAMPION_LEVERS)" | grep -q '^HPU_GLM53_ATTN_PIN=' && pin_set=yes
    [ "$gate:$pin_set" != on:no ] || refuse "R4E_PIN_GATE=on but the levers leave HPU_GLM53_ATTN_PIN unset: stage the pin arm's levers or gates off"
    [ "$gate:$pin_set" != off:yes ] || refuse "R4E_PIN_GATE=off but the levers set HPU_GLM53_ATTN_PIN: the launcher refuses the key with the gate off"
}

launcher_check() {  # $1 role: serve_champion.sh --check runs the launcher's pre-boot refusals (config, levers, gates, image) only
    local out
    out=$(FINAL_ENV="$(staged_value CHAMPION_FINAL_ENV)" bash "$CHAMPION_SERVE" "$1" --check 2>&1) || refuse "$1's pre-boot check refused: $(grep -m 1 'HALT' <<< "$out" || tail -n 1 <<< "$out")"
    log "$(tail -n 1 <<< "$out")"
}

intended_lines() { grep -v -E '^[[:space:]]*(#|$)' "$1" | sed -e 's/^export //' -e "s/=[\"']\(.*\)[\"']$/=\1/"; }

value_drift() {  # $1 container, $2 levers file: VALUE-DRIFT lines (an intended key unset live, at another value, or set more than
    local live line key drift=0 value count   # once: docker serves the last entry, so a matching one beside it proves nothing); rc 1 on any
    live=$($DOCKER inspect -f '{{range .Config.Env}}{{println .}}{{end}}' "$1")
    while IFS= read -r line; do
        key="${line%%=*}"
        count=$(grep -c -E "^$key=" <<< "$live" || true)
        if [ "$count" -gt 1 ]; then
            log "VALUE-DRIFT $1 $key set $count times live (intended=${line#*=})"; drift=$((drift + 1)); continue
        fi
        value=$(grep -m 1 -E "^$key=" <<< "$live" | cut -d= -f2- || true)
        grep -q -x -F -- "$line" <<< "$live" && continue
        log "VALUE-DRIFT $1 $key intended=${line#*=} live=${value:-<unset>}"; drift=$((drift + 1))
    done < <(intended_lines "$2")
    log "LIVE_PARITY $1 VALUE-DRIFT $drift"
    [ "$drift" = 0 ]
}

replica_gates() {  # $1 role, $2 its BOOTED line
    local role="$1" name boot warmup
    name=$(lab_name "$role" "$CHAMPION_CUT")
    boot=$(sed -n 's/.*boot_s=\([0-9]*\).*/\1/p' <<< "$2")
    [ -n "$boot" ] && [ "$boot" -le "$BOOT_BUDGET_S" ] || { log "GATE_FAIL $role: boot ${boot:-<none>}s (budget ${BOOT_BUDGET_S}s)"; return 1; }
    warmup=$($DOCKER logs "$name" 2>&1 | sed -n 's/.*Warmup finished in \([0-9]*\) secs.*/\1/p' | sort -n | tail -n 1)
    [ -n "$warmup" ] && [ "$warmup" -le "$WARMUP_BUDGET_S" ] || { log "GATE_FAIL $role: warmup ${warmup:-<none>}s (budget ${WARMUP_BUDGET_S}s)"; return 1; }
    [ "$(container_image "$name")" = "$(staged_value CHAMPION_IMAGE_SHA)" ] || { log "GATE_FAIL $role: $name is not the staged image"; return 1; }
    value_drift "$name" "$(staged_value CHAMPION_LEVERS)" || { log "GATE_FAIL $role: live parity"; return 1; }
    health_and_smoke "$role" || return 1
    log "GATES_PASS $role boot ${boot}s warmup ${warmup}s VALUE-DRIFT 0 SMOKE_OK :${PORT[$role]}"
}

promote_role() {  # $1 role: r4f off its port (holder released), the champion booted and through its gates
    local role="$1" booted
    stop_and_release "$(lab_name "$role" r4f)" "$role" || return 1
    # the staged base config, for this launch only (the r4f rollback launcher keeps its own)
    booted=$(FINAL_ENV="$(staged_value CHAMPION_FINAL_ENV)" launch_and_wait "$CHAMPION_SERVE" "$role" "serve-$CHAMPION_CUT-$role") || return 1
    replica_gates "$role" "$booted"
}

roll_back() {  # $1 why: every champion container stopped and released, then the r4f rollback (both replicas)
    local role
    alarm "PROMOTE FAILED: $1; rolling back to r4f"
    for role in "${ROLES[@]}"; do stop_and_release "$(lab_name "$role" "$CHAMPION_CUT")" "$role" || true; done
    rollback_main || { alarm "ROLLBACK ALSO FAILED: see the log above"; exit 1; }
    exit 1
}

promote_main() {
    case "$#:${1:-}" in 0:) DRY_RUN=0 ;; 1:--dry-run) DRY_RUN=1 ;; *) refuse "usage: promote_r4h.sh [--dry-run] | stage ..." ;; esac
    promote_preflight
    [ "$DRY_RUN" = 0 ] || { log "PROMOTE_DRYRUN_OK (would promote r0 then r1 to $(staged_value CHAMPION_IMAGE))"; return 0; }
    local role
    for role in "${ROLES[@]}"; do
        promote_role "$role" || roll_back "$role did not pass (boot, warmup, parity or smoke)"
    done
    log "C1D_CHAMPION_PROMOTED cut=$CHAMPION_CUT r0 :18274 r1 :18275 image $(staged_value CHAMPION_IMAGE_SHA | cut -c8-19) levers $(staged_value CHAMPION_LEVERS_SHA | cut -c1-8)"
}

[ "${BASH_SOURCE[0]}" != "$0" ] || {
    [ "${1:-}" != stage ] || { shift; stage "$@"; exit 0; }
    mkdir -p "$KIT_DIR/logs"
    LOG="$KIT_DIR/logs/promote_r4h-$STAMP.log"
    ln -sfn "$LOG" "$KIT_DIR/logs/promote_r4h-latest.log"
    promote_main "$@" > "$LOG" 2>&1
}
