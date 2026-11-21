#!/bin/bash
# One lab boot on one quad, then cold and warm long prompts: reads memory fit and speed for a key set on an existing image.
# Speed and fit only: this image predates the conv state fix, so output is not judged here.
# usage: fastread_cell.sh <A|B> <port> <env-file> <tag> <tokens,list> <output-tokens> <image>
# Teardown always: lab down, quad lock released, watcher stopped, cards checked free. It never stops r0, r1 or the target, and no reset.
set -u
QUAD="$1"; PORT="$2"; ENV_FILE="$3"; TAG="$4"; TOKENS="$5"; OUTPUT="$6"; IMAGE="$7"
L="$(cd "$(dirname "$0")/../.." && pwd)"
NAME="claude-flash-launch-fastread-$TAG"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
OUT="$L/k1-out/fastread-$TAG-$STAMP"
RELEASE="$OUT/release"
RC_FILE="$L/k1-out/fastread-$TAG.rc"
CARD_FREE_MIB=1024
CARD_WAIT_S=300
BOOT_DEADLINE_S=2400
DRIVE_LIMIT_S=7200
case "$QUAD" in
    A) MODULES="4 5 6 7" ;;
    B) MODULES="0 1 2 3" ;;
    *) echo "quad must be A or B, got $QUAD"; exit 2 ;;
esac

WATCHER=""
RESULT=1
cards_free() {
    local module used
    for module in $MODULES; do
        used="$(hl-smi -Q module_id,memory.used -f csv,noheader,nounits | awk -F', *' -v m="$module" '$1 == m {print $2}')"
        [ -n "$used" ] && [ "$used" -le "$CARD_FREE_MIB" ] || return 1
    done
}
teardown() {
    trap - EXIT INT TERM
    if docker inspect "$NAME" > /dev/null 2>&1; then
        python3 "$L/lab.py" down --name "$NAME"
    fi
    cp "$L/$NAME.log" "$OUT/container.log" 2> /dev/null
    [ -n "$WATCHER" ] && kill "$WATCHER" 2> /dev/null
    touch "$RELEASE"
    local waited=0
    until cards_free; do
        if [ "$waited" -ge "$CARD_WAIT_S" ]; then
            echo "CARDS_STUCK after ${CARD_WAIT_S}s: stop and report; no reset"
            RESULT=3
            break
        fi
        sleep 10
        waited=$((waited + 10))
    done
    [ -f "$OUT/container.log" ] && summarize
    echo "$RESULT $OUT" > "$RC_FILE"
}
summarize() {
    python3 "$L/overlay/cells/k1_table.py" "$OUT/container.log" > "$OUT/table.txt" 2>&1
    {
        echo "FAST_EAGER_LINES $(grep -ac '"path": "eager"' "$OUT/container.log")"
        echo "FAST_PT_DEVMEM $(grep -ac 'PT_DEVMEM' "$OUT/container.log")"
        echo "FAST_NO_PROGRESS $(grep -ac 'No progress' "$OUT/container.log")"
        echo "FAST_TRACEBACK $(grep -ac 'Traceback' "$OUT/container.log")"
        echo "FAST_KEY_MISS $(grep -aoE "'fallback_(ctx|width)_over': [1-9][0-9]*" "$OUT/container.log" | sort -u | tr '\n' ' ')"
        cat "$OUT/$NAME.long.jsonl" 2> /dev/null
    } >> "$OUT/table.txt"
}
trap teardown EXIT
trap 'exit 130' INT TERM

rm -f "$RC_FILE"
mkdir -p "$OUT/trace" "$OUT/hl"
chmod 777 "$OUT/trace" "$OUT/hl"
cp "$ENV_FILE" "$OUT/env"
setsid nohup bash "$L/quad_hold.sh" "$QUAD" "$RELEASE" 14400 > /dev/null 2>&1 &
HOLD_WAITED=0
until [ -s "/tmp/claude-flash-quad$QUAD.holder" ]; do
    sleep 1
    HOLD_WAITED=$((HOLD_WAITED + 1))
    if [ "$HOLD_WAITED" -ge 30 ]; then
        echo "quad $QUAD lock not taken in 30 s: another holder is alive; nothing booted"
        exit 1
    fi
done

ENV_ARGS=()
while IFS= read -r line; do
    case "$line" in ''|'#'*) continue ;; esac
    ENV_ARGS+=(--env "$line")
done < "$ENV_FILE"

sudo -n sh -c 'echo 1 > /proc/sys/vm/compact_memory'  # compaction only: dropping the file cache makes weights load from disk
python3 "$L/overlay/cells/k1_host_sampler.py" "$NAME" "$OUT/trace" > "$OUT/watcher.err" 2>&1 &
WATCHER=$!
python3 "$L/lab.py" up --quad "$QUAD" --name "$NAME" --image "$IMAGE" --port "$PORT" --deadline "$BOOT_DEADLINE_S" \
    "${ENV_ARGS[@]}" ${FAST_MOUNTS:-} --mount "$OUT/trace:/k1trace" --mount "$OUT/hl:/var/log/habana_logs" 2>&1 | tee "$OUT/up.log"
[ "${PIPESTATUS[0]}" -eq 0 ] || { echo "FAST_BOOT_FAILED"; exit 1; }
cp "$L/state-$NAME.json" "$OUT/state.json" 2> /dev/null
timeout "$DRIVE_LIMIT_S" python3 "$L/overlay/cells/k1_cell.py" drive-long --port "$PORT" --tokens-list "$TOKENS" \
    --output-tokens "$OUTPUT" --out "$OUT/$NAME.long.jsonl" 2>&1 | tee "$OUT/drive.log"
RESULT="${PIPESTATUS[0]}"
exit "$RESULT"
