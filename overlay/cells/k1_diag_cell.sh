#!/bin/bash
# One K1 diagnostic boot: usage k1_diag_cell.sh <label> <env-file> <image> [repeats] [tokens]
# Boots quad A through lab.py with the stall-trace dir and Habana logs bind-mounted, runs the host sampler, drives the fixed prompt
# when the boot gets ready, always takes lab.py down, scores the log, and copies the container log into k1-out/<label>/.
set -u
LABEL="$1"; ENV_FILE="$2"; IMAGE="$3"; REPEATS="${4:-12}"; TOKENS="${5:-6144}"
L=~/.codex/tmp/flash-spikes-20261104/team-20261106-dual
NAME="claude-flash-launch-k1-$LABEL"
OUT="$L/k1-out/$LABEL"
PORT=18440
mkdir -p "$OUT/trace" "$OUT/hl"
chmod 777 "$OUT/trace" "$OUT/hl"
ENVS=()
while IFS= read -r line; do
    case "$line" in ''|\#*) continue ;; esac
    ENVS+=(--env "$line")
done < "$ENV_FILE"
ENVS+=(--env HPU_GLM53_K1_STALL_DIR=/k1trace)

python3 "$L/overlay/cells/k1_host_sampler.py" "$NAME" "$OUT/trace" > "$OUT/sampler.err" 2>&1 &
SAMPLER=$!

python3 "$L/lab.py" up --quad "${QUAD:-A}" --name "$NAME" --image "$IMAGE" --port "$PORT" --deadline 1500 ${LAB_KV_BLOCKS:+--lab-kv-blocks "$LAB_KV_BLOCKS"} "${ENVS[@]}" \
    --mount "$OUT/trace:/k1trace" --mount "$OUT/hl:/var/log/habana_logs" > "$OUT/up.log" 2>&1
UP=$?
echo "UP_RC $UP $(date +%T)" | tee -a "$OUT/result.txt"
if [ "$UP" -eq 0 ]; then
    cd "$L/overlay/cells" && timeout 1860 python3 k1_cell.py drive --port "$PORT" --tokens "$TOKENS" --repeats "$REPEATS" \
        --timeout 900 --out "$OUT/drive.json" >> "$OUT/drive.log" 2>&1
    echo "DRIVE_RC $? $(date +%T)" | tee -a "$OUT/result.txt"
fi
python3 "$L/lab.py" down --name "$NAME" >> "$OUT/up.log" 2>&1
kill "$SAMPLER" 2>/dev/null
cp "$L/state-$NAME.json" "$OUT/state.json" 2>/dev/null
mv "$L/$NAME.log" "$OUT/container.log" 2>/dev/null
echo "DONE $(date +%T)" | tee -a "$OUT/result.txt"
