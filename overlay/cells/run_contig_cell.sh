#!/bin/bash
# One-card contig byte cell (plan steps 3 and 14). WAITING ON A CARD: run only in a card window the owner scheduled.
# usage: run_contig_cell.sh --module <HABANA module id> --accel <N> --uverbs <N> --out-dir <host dir> [--mode contig|native] [--slots 50] [--dry-run]
# Refuses if any running container holds /dev/accel/accel<N>. Prints the docker command before it runs.
set -euo pipefail
OVERLAY="$(cd "$(dirname "$0")/.." && pwd)"
IMAGE=1481bb99fce0
MODE=contig SLOTS=50 DRY=0 MODULE="" ACCEL="" UVERBS="" OUT=""
while [ "$#" -gt 0 ]; do
    case "$1" in
        --module) MODULE="$2"; shift 2 ;;
        --accel) ACCEL="$2"; shift 2 ;;
        --uverbs) UVERBS="$2"; shift 2 ;;
        --out-dir) OUT="$2"; shift 2 ;;
        --mode) MODE="$2"; shift 2 ;;
        --slots) SLOTS="$2"; shift 2 ;;
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
NAME="claude-flash-contig-cell-$$"
RUN=(docker run --rm --name "$NAME" --network none --ipc host --ulimit memlock=-1:-1 --security-opt label=disable --cap-add CAP_SYS_NICE --cap-add CAP_SYS_PTRACE
     --device "/dev/accel/accel$ACCEL" --device "/dev/accel/accel_controlD$ACCEL" --device "/dev/infiniband/uverbs$UVERBS"
     -e "HABANA_VISIBLE_MODULES=$MODULE" -e PYTHONDONTWRITEBYTECODE=1 -e PT_HPU_LAZY_MODE=1
     -v "$OVERLAY:/overlay:ro" -v "$OUT:/out" --entrypoint python3 "$IMAGE"
     /overlay/cells/contig_byte_cell.py --mode "$MODE" --device hpu --slots "$SLOTS" --overlay /overlay --out "/out/contig-$MODE.json")
printf '%q ' "${RUN[@]}"; echo
if [ "$DRY" = 1 ]; then
    exit 0
fi
mkdir -p "$OUT"
timeout 900 "${RUN[@]}"
