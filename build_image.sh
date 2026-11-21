#!/bin/bash
# Rerunnable build of claude-flash-dual-sustain from the CURRENT overlay/ (new files in overlay/ are picked up with no edit).
# usage: build_image.sh [tag]          default tag claude-flash-dual-sustain:20261106 (a rerun moves only that tag)
# Starts no card and no card device. Never removes, retags or prunes an image. Writes nothing on /mnt/data1 (it only checks it).
# Stages: disk check -> protected images recorded -> docker build -> protected images unchanged -> tests (no network)
#         -> new flags default off -> held step 22 module absent from the image. Output is also written to build_image.log in this folder (overwritten each run).
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1

LANE="$(cd "$(dirname "$0")" && pwd)"
TAG="${1:-claude-flash-dual-sustain:20261106}"
BASE_ID="sha256:1481bb99fce076776c9e88604219726df33a903514015aa13fb0efc2350c8bb"
PROTECTED=(1481bb99 73fc9013 db7841c6 bc2899df)
MIN_FREE_GB=100
# Plan step 22 (credit-bounded run-ahead) is held: the module stays in overlay/ on disk, .dockerignore keeps it out of the image,
# and the overlay's glm53_r589_bounded_prefill.py carries no hook for it.
HELD_OUT_MODULE="glm53_runahead_credit.py"
DOCKER_ROOT="/var/lib/docker"
DATA_DISK="/mnt/data1"
SNAPSHOT="$(mktemp)"
trap 'rm -f "$SNAPSHOT"' EXIT

die() {
    echo "build_image: $*" >&2
    exit 1
}

free_gb() {
    df --output=avail -BG "$1" | tail -1 | tr -dc '0-9'
}

check_disk() {
    local path free
    for path in "$DOCKER_ROOT" "$DATA_DISK"; do
        free="$(free_gb "$path")"
        echo "disk: $path free ${free} GB (need ${MIN_FREE_GB})"
        [ "$free" -ge "$MIN_FREE_GB" ] || die "$path has ${free} GB free, below ${MIN_FREE_GB}; free space (not by deleting images 1481bb99, 73fc9013, db7841c6, bc2899df) and rerun"
    done
}

protected_ids() {
    local prefix
    for prefix in "${PROTECTED[@]}"; do
        echo "$prefix $(docker image inspect --format '{{.Id}}' "$prefix")"
    done
}

check_tag_is_free() {
    local prefix
    local current
    current="$(docker image inspect --format '{{.Id}}' "$TAG" 2>/dev/null || true)"
    [ -z "$current" ] && return 0
    for prefix in "${PROTECTED[@]}"; do
        case "$current" in sha256:"$prefix"*) die "tag $TAG points at protected image $prefix; choose another tag" ;; esac
    done
}

overlay_digest() {
    (cd "$LANE" && find overlay -type f ! -path 'overlay/tests/*' ! -path 'overlay/cells/*' ! -name run_tests.sh \
        ! -name "$HELD_OUT_MODULE" ! -path '*/__pycache__/*' | sort | xargs -d '\n' sha256sum | sha256sum | cut -c1-64)
}

build() {
    docker image inspect "$BASE_ID" > /dev/null || die "base image $BASE_ID (1481bb99) is not on this host"
    docker build --build-arg "BASE_IMAGE=$BASE_ID" --build-arg "OVERLAY_SHA256=$(overlay_digest)" -t "$TAG" "$LANE"
}

verify_protected_unchanged() {
    local after
    after="$(protected_ids)"
    [ "$after" == "$(cat "$SNAPSHOT")" ] || die "a protected image changed during the build: before/after differ"
    echo "protected images unchanged:"
    echo "$after"
}

# run_tests.sh pins IMAGE=1481bb99fce0; the same command line is reused here against the new tag.
run_tests_in_new_image() {
    local script
    script="$(mktemp)"
    sed -e "s|^IMAGE=.*|IMAGE=\"$TAG\"|" -e "s|^OVERLAY=.*|OVERLAY=\"$LANE/overlay\"|" "$LANE/overlay/run_tests.sh" > "$script"
    grep -q "^IMAGE=\"$TAG\"" "$script" || die "overlay/run_tests.sh no longer has an IMAGE= line to retarget"
    bash "$script"
    rm -f "$script"
}

check_flags_default_off() {
    docker run --rm --network none --entrypoint python3 -e PYTHONDONTWRITEBYTECODE=1 -e HABANA_VISIBLE_DEVICES=none \
        -v "$LANE/check_flags_off.py:/check/check_flags_off.py:ro" "$TAG" -B /check/check_flags_off.py
}

# The held step 22 module is absent from the image and R589 does not mention it (R589 would fail to import without it).
check_held_out_of_image() {
    docker run --rm --network none --entrypoint bash -e HABANA_VISIBLE_DEVICES=none "$TAG" -c \
        "cd /usr/local/lib/python3.12/dist-packages && test ! -e $HELD_OUT_MODULE && ! grep -qi runahead glm53_r589_bounded_prefill.py \
         && python3 -B -c 'import glm53_r589_bounded_prefill' > /dev/null 2>&1 \
         && echo 'held out: $HELD_OUT_MODULE is not in the image and R589 has no credit hook'"
}

FAILED_STAGES=()

# Runs one verification stage; a failing stage is recorded and the later stages still run, so one report shows everything.
stage() {
    local title="$1"
    shift
    echo "== $title =="
    "$@" || FAILED_STAGES+=("$title")
}

main() {
    exec > >(tee "$LANE/build_image.log") 2>&1
    check_disk
    check_tag_is_free
    protected_ids | tee "$SNAPSHOT"
    build
    verify_protected_unchanged
    stage "differential tests: base image 1481bb99fce0 vs overlay" bash "$LANE/overlay/run_tests.sh"
    stage "same tests inside $TAG (--network none, no card devices)" run_tests_in_new_image
    stage "new flags default off in $TAG" check_flags_default_off
    stage "held step 22 credit module absent from $TAG" check_held_out_of_image
    echo "IMAGE $TAG $(docker image inspect --format '{{.Id}}' "$TAG")"
    if [ "${#FAILED_STAGES[@]}" -gt 0 ]; then
        echo "BUILD_RESULT: FAIL (image built; failed stages: ${FAILED_STAGES[*]})"
        exit 1
    fi
    echo "BUILD_RESULT: PASS"
}

main
