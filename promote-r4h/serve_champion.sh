#!/bin/bash
# promote-r4h/serve_champion.sh: one champion SERVING replica through the 09:00 install's launcher (diag-r4e/r4e_run.sh prepare + boot),
# as c1d-r4f-serve/serve_r4f.sh, with the image and levers file promote_r4h.sh staged (champion.staged, each pinned by sha). Production
# port, host 0.0.0.0, the source unit's argv (1M context, 16 seqs, production KV). No measure, no down: the replica keeps serving.
# Stop it with promote-r4h's stop_and_release (docker stop, then the SERVING line's release file frees the quad holder).
# The gate settings are the staged CHAMPION_GATES file (pinned by sha), never a hard-coded arm's.
# usage (promote_r4h.sh runs it): setsid nohup bash serve_champion.sh r0|r1 < /dev/null > <console> 2>&1 &
#        serve_champion.sh r0|r1 --check   the launcher's pre-boot refusals only (r4e_prepare's config chain), no quad, no boot:
#                                           CHAMPION_CHECK_OK or its HALT line; output under a fresh temp folder
set -u
KIT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROLE="${1:?usage: serve_champion.sh r0|r1 [--check]}"
MODE="${2:-serve}"
case "$MODE" in serve|--check) ;; *) echo "HALT serve_champion.sh: mode $MODE is not --check" >&2; exit 2 ;; esac
case "$ROLE" in r0) SERVE_QUAD=A ;; r1) SERVE_QUAD=B ;; *) echo "HALT serve_champion.sh: role $ROLE is not r0 or r1" >&2; exit 2 ;; esac
L=$HOME/.codex/tmp/flash-spikes-20261104/team-20261106-dual
STAGED_FILE="$KIT_DIR/champion.staged"
[ -f "$STAGED_FILE" ] || { echo "HALT serve_champion.sh: $STAGED_FILE is missing (promote_r4h.sh stage)" >&2; exit 2; }
CHAMPION_CUT="$(sed -n 's/^CHAMPION_CUT=//p' "$STAGED_FILE")"
[ -n "$CHAMPION_CUT" ] || { echo "HALT serve_champion.sh: $STAGED_FILE names no CHAMPION_CUT (promote_r4h.sh stage --image)" >&2; exit 2; }
CHAMPION_IMAGE="$(sed -n 's/^CHAMPION_IMAGE=//p' "$STAGED_FILE")"
CHAMPION_IMAGE_SHA="$(sed -n 's/^CHAMPION_IMAGE_SHA=//p' "$STAGED_FILE")"
CHAMPION_LEVERS="$(sed -n 's/^CHAMPION_LEVERS=//p' "$STAGED_FILE")"
CHAMPION_LEVERS_SHA="$(sed -n 's/^CHAMPION_LEVERS_SHA=//p' "$STAGED_FILE")"
CHAMPION_GATES="$(sed -n 's/^CHAMPION_GATES=//p' "$STAGED_FILE")"
CHAMPION_GATES_SHA="$(sed -n 's/^CHAMPION_GATES_SHA=//p' "$STAGED_FILE")"
[ -n "$CHAMPION_GATES" ] || { echo "HALT serve_champion.sh: $STAGED_FILE names no CHAMPION_GATES (promote_r4h.sh stage --gates)" >&2; exit 2; }
echo "$CHAMPION_LEVERS_SHA  $CHAMPION_LEVERS" | sha256sum --quiet -c - 2> /dev/null \
    || { echo "HALT serve_champion.sh: $CHAMPION_LEVERS is not the staged ${CHAMPION_LEVERS_SHA:0:8}" >&2; exit 2; }
echo "$CHAMPION_GATES_SHA  $CHAMPION_GATES" | sha256sum --quiet -c - 2> /dev/null \
    || { echo "HALT serve_champion.sh: $CHAMPION_GATES is not the staged ${CHAMPION_GATES_SHA:0:8}" >&2; exit 2; }
# The base config is staged and pinned like the levers and gates (the 2026-11-11 r1 launch ran with FINAL_ENV unset and booted
# final_run.sh's default base, another configuration): no staged pin, an unset FINAL_ENV or other bytes all refuse before any boot.
CHAMPION_FINAL_ENV="$(sed -n 's/^CHAMPION_FINAL_ENV=//p' "$STAGED_FILE")"
CHAMPION_FINAL_ENV_SHA="$(sed -n 's/^CHAMPION_FINAL_ENV_SHA=//p' "$STAGED_FILE")"
[ -n "$CHAMPION_FINAL_ENV" ] && [ -n "$CHAMPION_FINAL_ENV_SHA" ] \
    || { echo "HALT serve_champion.sh: $STAGED_FILE names no CHAMPION_FINAL_ENV: the launcher would boot its default base config, not the champion's (promote_r4h.sh stage ... --final-env <file>)" >&2; exit 2; }
[ -n "${FINAL_ENV:-}" ] \
    || { echo "HALT serve_champion.sh: FINAL_ENV is unset: export FINAL_ENV=$CHAMPION_FINAL_ENV (the staged base config ${CHAMPION_FINAL_ENV_SHA:0:8}); without it the replica boots another configuration" >&2; exit 2; }
echo "$CHAMPION_FINAL_ENV_SHA  $FINAL_ENV" | sha256sum --quiet -c - 2> /dev/null \
    || { echo "HALT serve_champion.sh: FINAL_ENV=$FINAL_ENV is not the staged ${CHAMPION_FINAL_ENV_SHA:0:8}: export FINAL_ENV=$CHAMPION_FINAL_ENV" >&2; exit 2; }
export FINAL_ENV
export C1D_KV_BLOCKS=30880 PORT_A=18274 PORT_B=18275 LAB_HOST=0.0.0.0 LAB_SERVE_PROD=1 LABPY="$L/lab_serve_seeded.py"
# LAB_RECIPE_SEED_LEAF is inherited, never set here: champion_boot.sh exports the leaf the last serving boot of the staged image
# recorded (or keeps the caller's own); unset, the launcher boots a fresh empty leaf exactly as the stock chain does.
export R4E_IMAGE="$CHAMPION_IMAGE" R4E_IMAGE_SHA="$CHAMPION_IMAGE_SHA"
export R4E_GATES_OVERRIDE="$CHAMPION_GATES" R4E_LEVERS_ENV="$CHAMPION_LEVERS"
export R4E_ARM="champion-$ROLE"
export FREE_FOR_S="${FREE_FOR_S:-30}" GRADED_LIMIT_S="${HOLD_S:-2592000}"   # the quad hold lasts while the replica serves (30 days)
# shellcheck source=/dev/null
RUN_NAME="serve-$CHAMPION_CUT-$ROLE"
if [ "$MODE" = --check ]; then OUT_ROOT="$(mktemp -d /tmp/champion-check.XXXXXX)"; export OUT_ROOT; RUN_NAME="check-$CHAMPION_CUT-$ROLE"
    trap 'rm -rf -- "$OUT_ROOT"' EXIT; fi   # the check's own fresh temp folder only
R4E_RUN_NO_MAIN=1 source "$L/diag-r4e/r4e_run.sh" "$RUN_NAME" "$SERVE_QUAD"
# shellcheck source=champion_check.sh
[ "$MODE" != --check ] || source "$KIT_DIR/champion_check.sh"   # its baked_sha starts no container (create + cp + rm)
NAME="claude-flash-launch-glm53flash-$ROLE-$CHAMPION_CUT"   # lab.py admits claude-flash-launch-* names only

serve_main() {
    r4e_prepare || { say "DONE verdict=HALT"; return 1; }
    start_watch "$NAME" "$QUAD" "$RUN"
    say "BOOT serve role=$ROLE quad=$QUAD port=${PORT[$QUAD]} image=$IMAGE name=$NAME"
    if ! boot "$NAME" "$QUAD" "$IMAGE" "$RUN" "$DEADLINE"; then
        say "BOOT_FAILED serve role=$ROLE $(tail -n 1 "$RUN/rc") $(grep -a -m1 'refuses\|Error' "$RUN/up.log" | cut -c1-160)"
        finish; say "DONE verdict=BOOT_FAILED"; return 1
    fi
    say "BOOTED serve role=$ROLE $(tail -n 1 "$RUN/rc")"
    say "SERVING name=$NAME port=${PORT[$QUAD]} holder=$HOLD; rollback: docker stop $NAME, then touch $RELEASE"
}

check_main() {  # r4e_prepare's refusal chain before any card time, without its run folders, quad wait or boot
    mkdir -p "$(dirname "$STATUS")"   # say() appends the HALT line to $STATUS (inside the check's temp folder)
    if r4e_args && forbid && require_levers && image_bytes_ok; then
        echo "CHAMPION_CHECK_OK role=$ROLE cut=$CHAMPION_CUT gates=${CHAMPION_GATES_SHA:0:8} levers=${CHAMPION_LEVERS_SHA:0:8} final_env=${CHAMPION_FINAL_ENV_SHA:0:8} pin_gate=$R4E_PIN_GATE"
        return 0
    fi
    echo "CHAMPION_CHECK_HALT role=$ROLE $(grep -a 'HALT' "$STATUS" 2>/dev/null | tail -n 1 | cut -d' ' -f4-)"
    return 1
}

if [ "$MODE" = --check ]; then check_main; else serve_main; fi
