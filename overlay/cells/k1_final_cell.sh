#!/bin/bash
# The corrected TP4 K1 cell: both 1024 keys, owner on, hold 0, 1,538 MiB ballast from the first replay, at KV 31,264.
# usage: k1_final_cell.sh <A|B arm> <image> [--dry-run]
#   --dry-run prints the k1_cell plan and touches no card. Without it the caller needs quad A held and r0 stopped: lab.py refuses otherwise.
# Runs the host watcher (free, load, dmesg, per-worker threads). Teardown always: k1_cell's own down, or this trap if the cell was killed;
# then the watcher stops and cards must read at most CARD_FREE_MIB within CARD_WAIT_S. It never stops r0, r1 or the target, and no reset.
set -u
ARM="$1"; IMAGE="$2"; MODE="${3:---execute}"
L="$(cd "$(dirname "$0")/../.." && pwd)"
NAME="claude-flash-launch-k1-arm$ARM"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
OUT="$L/k1-out/final-arm$ARM-$STAMP"
MODULES="4 5 6 7"
CARD_FREE_MIB=1024
CARD_WAIT_S=300
RUN=(python3 "$L/overlay/cells/k1_cell.py" run --arm "$ARM" --image "$IMAGE" --port 18440 --out-dir "$OUT" --stamp "$STAMP"
     --env-file "$L/graphs-k1.env" --mount "$OUT/trace:/k1trace" --mount "$OUT/hl:/var/log/habana_logs"
     --long-tokens "${K1_LONG_TOKENS:-150000,400000,1048575}" --long-output-tokens "${K1_LONG_OUTPUT:-1}")
if [ "$MODE" = "--dry-run" ]; then
    "${RUN[@]}" --dry-run
    exit $?
fi

WATCHER=""
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
    [ -n "$WATCHER" ] && kill "$WATCHER" 2> /dev/null
    local waited=0
    until cards_free; do
        if [ "$waited" -ge "$CARD_WAIT_S" ]; then
            echo "CARDS_STUCK after ${CARD_WAIT_S}s: stop and report; no reset"
            exit 3
        fi
        sleep 10
        waited=$((waited + 10))
    done
    echo "CARDS_FREE after ${waited}s"
}
trap teardown EXIT
trap 'teardown; exit 130' INT TERM

mkdir -p "$OUT/trace" "$OUT/hl"
chmod 777 "$OUT/trace" "$OUT/hl"
sudo -n sh -c 'sync; echo 1 > /proc/sys/vm/drop_caches; echo 1 > /proc/sys/vm/compact_memory'  # C189: ctx_init -12 on a fragmented node
python3 "$L/overlay/cells/k1_host_sampler.py" "$NAME" "$OUT/trace" > "$OUT/watcher.err" 2>&1 &
WATCHER=$!
"${RUN[@]}" --execute 2>&1 | tee "$OUT/cell.log"
RC=${PIPESTATUS[0]}
cp "$L/$NAME.log" "$OUT/container.log" 2> /dev/null
python3 "$L/overlay/cells/k1_table.py" "$OUT/container.log" 2>&1 | tee "$OUT/table.txt"
# Graphs only: any eager chunk or key miss during the long prompts is a failure of the graph path, whatever the fit verdict says.
echo "K1_EAGER_LINES $(grep -ac '"path": "eager"' "$OUT/container.log")" | tee -a "$OUT/table.txt"
echo "K1_KEY_MISS_COUNTERS $(grep -aoE "'fallback_(ctx|width)_over': [1-9][0-9]*" "$OUT/container.log" | sort -u | tr '\n' ' ')" | tee -a "$OUT/table.txt"
cat "$OUT/$NAME.long.jsonl" 2> /dev/null | tee -a "$OUT/table.txt"
echo "K1_FINAL_RC $RC $OUT"
exit "$RC"
