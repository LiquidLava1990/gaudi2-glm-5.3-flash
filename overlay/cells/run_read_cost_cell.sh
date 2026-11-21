#!/bin/bash
# One-card allocator read-cost cell (plan step 22). WAITING ON A CARD: run only in a card window the owner scheduled.
# usage: run_read_cost_cell.sh --module <HABANA module id> --accel <N> --uverbs <N> --out-dir <host dir> [--margin-mib 512] [--max-boundary-growth-mib X] [--dry-run]
# Refuses if any running container holds /dev/accel/accel<N>. Prints the docker command before it runs.
set -euo pipefail
OVERLAY="$(cd "$(dirname "$0")/.." && pwd)"
IMAGE=1481bb99fce0
DRY=0 MODULE="" ACCEL="" UVERBS="" OUT="" MARGIN=512 GROWTH=()
while [ "$#" -gt 0 ]; do
    case "$1" in
        --module) MODULE="$2"; shift 2 ;;
        --accel) ACCEL="$2"; shift 2 ;;
        --uverbs) UVERBS="$2"; shift 2 ;;
        --out-dir) OUT="$2"; shift 2 ;;
        --margin-mib) MARGIN="$2"; shift 2 ;;
        --max-boundary-growth-mib) GROWTH=(--max-boundary-growth-mib "$2"); shift 2 ;;
        --dry-run) DRY=1; shift ;;
        *) echo "unknown argument $1: see the usage line at the top of this file" >&2; exit 2 ;;
    esac
done
if [ -z "$MODULE" ] || [ -z "$ACCEL" ] || [ -z "$UVERBS" ] || [ -z "$OUT" ]; then
    echo "--module, --accel, --uverbs and --out-dir are required: take them from the quad lab.py dry-run reports" >&2
    exit 2
fi
for id in $(docker ps -q); do
    if docker inspect "$id" --format '{{range .HostConfig.Devices}}{{.PathOnHost}} {{end}}' | tr ' ' '\n' | grep -qx "/dev/accel/accel$ACCEL"; then
        echo "accel$ACCEL is held by running container $id: stop that quad in a scheduled card window first" >&2
        [ "$DRY" = 1 ] || exit 3
    fi
done
NAME="claude-flash-read-cost-cell-$$"
RUN=(docker run --rm --name "$NAME" --network none --ipc host --ulimit memlock=-1:-1 --security-opt label=disable --cap-add CAP_SYS_NICE --cap-add CAP_SYS_PTRACE
     --device "/dev/accel/accel$ACCEL" --device "/dev/accel/accel_controlD$ACCEL" --device "/dev/infiniband/uverbs$UVERBS"
     -e "HABANA_VISIBLE_MODULES=$MODULE" -e PYTHONDONTWRITEBYTECODE=1 -e PT_HPU_LAZY_MODE=1
     -v "$OVERLAY:/overlay:ro" -v "$OUT:/out" --entrypoint python3 "$IMAGE"
     /overlay/cells/read_cost_cell.py --device hpu --margin-mib "$MARGIN" "${GROWTH[@]}" --out /out/read-cost.json)
printf '%q ' "${RUN[@]}"; echo
if [ "$DRY" = 1 ]; then
    exit 0
fi
mkdir -p "$OUT"
timeout 900 "${RUN[@]}"
