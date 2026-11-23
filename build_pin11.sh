#!/bin/bash
# Rerunnable build of the r4h8-pin11 champion image: overlay-pin11/ copied onto claude-flash-dual-graphs-c1d:20261112-r4h8.
# usage: build_pin11.sh [tag]          default tag claude-flash-dual-graphs-c1d:20261122-pin11-r4h8 (a rerun moves only that tag)
# Starts no card and no card device. Never removes, retags or prunes an image.
# Stages: base id check -> tag check -> docker build -> installed bytes = overlay-pin11/SHA256SUMS -> pin suites (run_tests.sh).
# Output is also written to build_pin11.log in this folder (overwritten each run).
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1

LANE="$(cd "$(dirname "$0")" && pwd)"
TAG="${1:-claude-flash-dual-graphs-c1d:20261122-pin11-r4h8}"
BASE_ID="sha256:8bf1872d93c199fef9da428ed473c74be8736a695406f6d9666ae572e0d29e8b"
PROTECTED=(8bf1872d 0e5df6b7)   # the 2026-11-12 r4h8 champion and the production pin11 image
SITE=/usr/local/lib/python3.12/dist-packages

die() {
    echo "build_pin11: $*" >&2
    exit 1
}

check_tag_is_free() {
    local prefix current
    current="$(docker image inspect --format '{{.Id}}' "$TAG" 2>/dev/null || true)"
    [ -z "$current" ] && return 0
    for prefix in "${PROTECTED[@]}"; do
        case "$current" in sha256:"$prefix"*) die "tag $TAG points at protected image $prefix; choose another tag" ;; esac
    done
}

build() {
    docker image inspect "$BASE_ID" > /dev/null \
        || die "base image $BASE_ID (claude-flash-dual-graphs-c1d:20261112-r4h8) is not on this host; build or load it first"
    docker build -f "$LANE/Dockerfile.pin11" --build-arg "BASE_IMAGE=$BASE_ID" -t "$TAG" "$LANE"
}

verify_installed_bytes() {
    docker run --rm --network none --entrypoint bash -e HABANA_VISIBLE_DEVICES=none \
        -v "$LANE/overlay-pin11/SHA256SUMS:/check/SHA256SUMS:ro" "$TAG" -c "cd $SITE && sha256sum --strict -c /check/SHA256SUMS"
}

main() {
    exec > >(tee "$LANE/build_pin11.log") 2>&1
    check_tag_is_free
    build
    echo "== installed bytes =="
    verify_installed_bytes || die "a file in $TAG differs from overlay-pin11/SHA256SUMS"
    echo "== pin suites (host + image, CPU only) =="
    bash "$LANE/overlay-pin11/run_tests.sh" "$TAG" || die "overlay-pin11/run_tests.sh failed against $TAG"
    echo "IMAGE $TAG $(docker image inspect --format '{{.Id}}' "$TAG")"
    echo "BUILD_RESULT: PASS"
}

main
