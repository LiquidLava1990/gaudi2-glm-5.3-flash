#!/bin/bash
# Run the overlay's CPU tests in a throwaway container of the base image: no network, no card devices, removed after.
# Replay extract: docker logs <replica> 2>&1 | grep -a -E "GLM53_GS_SATURATION|GLM53_STEP_STATS worker-step" > /tmp/<file>; remove it after the run.
# The lane folder (the overlay's parent: env files, harness) is mounted read-only at /lane for the tests that read an env file.
# usage: run_tests.sh [unittest args]     env: GS_EXTRACT=<host file> mounts a docker logs extract for the replay tests
set -euo pipefail
OVERLAY="$(cd "$(dirname "$0")" && pwd)"
LANE="$(dirname "$OVERLAY")"
IMAGE=1481bb99fce0
ARGS=(--rm --network none --entrypoint python3 -e PYTHONDONTWRITEBYTECODE=1 -e HABANA_VISIBLE_DEVICES=none
      -v "$OVERLAY:/overlay:ro" -v "$LANE:/lane:ro" -e LANE_DIR=/lane -e PYTHONPATH=/overlay -w /overlay/tests)
if [ -n "${GS_EXTRACT:-}" ]; then
    ARGS+=(-v "$GS_EXTRACT:/extract.log:ro" -e GS_REPLAY_EXTRACT=/extract.log)
fi
if [ "$#" -eq 0 ]; then
    set -- discover -s /overlay/tests -t /overlay/tests
fi
exec docker run "${ARGS[@]}" "$IMAGE" -m unittest "$@"
