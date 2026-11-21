#!/bin/bash
# The C1 cell (DUAL-SUSTAINED-PLAN.md 12.4): one boot on quad A, four prompt keys + the real decode lattice, 512 MiB ballast, KV 31,264.
# usage: [C1_VARIANT=c1|c1p|c1m] c1_final_cell.sh <image> [--dry-run]   (variant files: graphs-c1.env, graphs-c1p.env, graphs-c1m.env)
#   --dry-run prints the c1_cell plan (for c1p, with the trace plan) and touches no card. Without it the caller needs quad A held and r0 stopped.
# C1_VARIANT=c1p: the same boot with glm53_graph_trace armed (graphs-c1p.env): $OUT/trace holds the replay traces (replaytrace-*), the cell waits for
#   every rank's window file, prints REPLAY_TRACE_* lines (per-family busy ms, top ops, launches, traced vs unprofiled wall) and keeps them in the result.
#   Re-read later: python3 overlay/cells/replay_trace_report.py report --dir $OUT/trace --runs <ids in cell.log> --container-log $OUT/container.log
# C1_VARIANT=c1m (plan 12.3b, step 49): the same boot with the shared key 4:256:1024, 4:1024:1024 optional (graphs-c1m.env). Drives cold, then 4 concurrent cold prompts of
#   about 4,000 tokens (and 4 short ones), then the same 4 one at a time; prints each shared key's warmed replay ms and resident HBM beside 1:1024:1024 and the
#   token comparison (shared_vs_serial: any differing token FAILs the variant, naming it). It drives no long, merge or lane step, so the long-context readings are absent.
# Runs the host watcher (free, load, dmesg, per-worker threads). Teardown always: c1_cell's own down, or this trap if the cell was killed;
# then the watcher stops and cards must read at most CARD_FREE_MIB within CARD_WAIT_S. It never stops r0, r1 or the target, and no reset.
set -u
IMAGE="$1"; MODE="${2:---execute}"
L="$(cd "$(dirname "$0")/../.." && pwd)"
VARIANT="${C1_VARIANT:-c1}"
NAME="claude-flash-launch-$VARIANT"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
OUT="$L/c1-out/$VARIANT-$STAMP"
MODULES="4 5 6 7"
CARD_FREE_MIB=1024
CARD_WAIT_S=300
RUN=(python3 "$L/overlay/cells/c1_cell.py" run --variant "$VARIANT" --image "$IMAGE" --port 18440 --out-dir "$OUT" --stamp "$STAMP"
     --env-file "${C1_ENV_FILE:-$L/graphs-$VARIANT.env}" --mount "$OUT/trace:/k1trace" --mount "$OUT/hl:/var/log/habana_logs" ${C1_MOUNTS:-}
     --long-tokens "${C1_LONG_TOKENS:-150000,530000}" --long-output-tokens "${C1_LONG_OUTPUT:-64}")
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
sudo -n sh -c 'echo 1 > /proc/sys/vm/compact_memory'  # C189: ctx_init -12 on a fragmented node
python3 "$L/overlay/cells/k1_host_sampler.py" "$NAME" "$OUT/trace" > "$OUT/watcher.err" 2>&1 &
WATCHER=$!
touch "$OUT/started"  # the lab log survives from earlier runs under this name; only one written after this counts
"${RUN[@]}" --execute 2>&1 | tee "$OUT/cell.log"
RC=${PIPESTATUS[0]}
if [ "$L/$NAME.log" -nt "$OUT/started" ]; then cp "$L/$NAME.log" "$OUT/container.log"; else : > "$OUT/container.log"; echo "C1_NO_CONTAINER_LOG this cell never booted; read cell.log" | tee -a "$OUT/cell.log"; fi
# Graphs only: any eager chunk or key miss is a failure of the graph path, whatever the readings say.
echo "C1_EAGER_LINES $(grep -ac '"path": "eager"' "$OUT/container.log")" | tee "$OUT/summary.txt"
echo "C1_KEY_MISS_COUNTERS $(grep -aoE "'fallback_[a-z_]+': [1-9][0-9]*" "$OUT/container.log" | sort -u | tr '\n' ' ')" | tee -a "$OUT/summary.txt"
if [ "$VARIANT" = "c1p" ]; then
    echo "C1_TRACE_FILES $(find "$OUT/trace" -maxdepth 1 -name 'replaytrace-*' | wc -l) $(du -sh "$OUT/trace" | cut -f1)" | tee -a "$OUT/summary.txt"
    grep -a "GLM53_REPLAY_TRACE" "$OUT/container.log" | cut -c1-300 | tail -n 24 | tee -a "$OUT/summary.txt"
fi
if [ "$VARIANT" = "c1m" ]; then
    echo "C1_SHARED_CENSUS_LINES $(grep -ac '"prompts": [0-9]' "$OUT/container.log")" | tee -a "$OUT/summary.txt"
    grep -a -E "^(replay_ms_|resident_mib_|shared_vs_serial)" "$OUT/cell.log" | cut -c1-300 | tee -a "$OUT/summary.txt"
fi
echo "C1_FINAL_RC $RC $OUT" | tee -a "$OUT/summary.txt"
exit "$RC"
