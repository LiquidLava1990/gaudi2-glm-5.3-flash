#!/bin/bash
# Ground-truth long-needle probe (coordinator 2026-11-08 22:4x): graded's 33K and 150K needles, greedy, scored right/wrong against the known
# answer, beside a running load (the 01:30 soak on the promoted r4d pair, or the r4e arm after its graded pass). PROBE_LANES per node (default 2,
# so 2 nodes add 4 lanes to the 16+16 profile): with --long-lanes 1, one 150K and one 33K request at a time per node, biggest first.
# Every finished task counts; the probe stops at PROBE_MAX_S. No paired reference exists: the result is a pass rate per length.
# usage: long_needle_probe.sh OUT_DIR LABEL=URL [LABEL=URL ...]
#        e.g. long_needle_probe.sh /mnt/data1/claude-flash-cells/soak-r4d/long-needle r0=http://127.0.0.1:18274 r1=http://127.0.0.1:18275
#   env: PROBE_LENGTHS=33k,150k  PROBE_LANES=2  PROBE_MAX_S=3300  TASKS=run2 tasks-cal/tasks.json
# Output: OUT_DIR/{tasks-long.json,LABEL/results.json,LABEL.log,summary.txt}; the LONG_NEEDLE lines are printed and kept in summary.txt.
set -u
OUT_DIR="${1:?usage: long_needle_probe.sh OUT_DIR LABEL=URL [LABEL=URL ...]}"
shift
[ "$#" -gt 0 ] || { echo "usage: long_needle_probe.sh OUT_DIR LABEL=URL [LABEL=URL ...]" >&2; exit 2; }
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
L="${L:-$HOME/.codex/tmp/flash-spikes-20261104/team-20261106-dual}"
GRADED_PY="${GRADED_PY:-$L/diag-graded/graded.py}"
TASKS="${TASKS:-/mnt/data1/graded-out/queue/run2/tasks-cal/tasks.json}"
PROBE_LENGTHS="${PROBE_LENGTHS:-33k,150k}"
PROBE_LANES="${PROBE_LANES:-2}"
PROBE_MAX_S="${PROBE_MAX_S:-3300}"
PROBE_LIMIT="${PROBE_LIMIT:-0}"   # 1 = the single-task proof against a live endpoint (sweep rule M2)
mkdir -p "$OUT_DIR"
python3 -I "$HERE/long_needle.py" select "$TASKS" "$OUT_DIR/tasks-long.json" --lengths "$PROBE_LENGTHS" --limit "$PROBE_LIMIT" || exit 2

pids=()
pairs=()
for target in "$@"; do
    label="${target%%=*}"; url="${target#*=}"
    [ -n "$label" ] && [ "$url" != "$target" ] || { echo "LONG_NEEDLE bad target '$target': want LABEL=URL" >&2; exit 2; }
    timeout "$PROBE_MAX_S" python3 -I "$GRADED_PY" run --base-url "$url" --tasks "$OUT_DIR/tasks-long.json" --label "long-needle-$label" \
        --out-dir "$OUT_DIR/$label" --lanes "$PROBE_LANES" --long-lanes 1 --samples 0 > "$OUT_DIR/$label.log" 2>&1 &
    pids+=("$!")
    pairs+=("$label=$OUT_DIR/$label/results.json")
done
for pid in "${pids[@]}"; do wait "$pid"; done
python3 -I "$HERE/long_needle.py" score "${pairs[@]}" --tasks "$OUT_DIR/tasks-long.json" | tee "$OUT_DIR/summary.txt"
