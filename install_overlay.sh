#!/bin/bash
# Runs inside `docker build`. Copies every file of the overlay onto site-packages after proving the base files are the ones
# BASE_SHA256.txt names. usage: install_overlay.sh <overlay-dir> <site-packages> <state-dir> [base-name]
#   base-name                      the base image named in error messages (default 1481bb99; overlay-pin11 passes 20261112-r4h8)
#   state-dir/base/<path>          the verified base file each overlay file replaced (provenance, used by check_flags_off.py)
#   state-dir/INSTALLED_SHA256.txt sha256 of every installed file
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1

SRC="$1"
PKG="$2"
STATE="$3"
BASE_NAME="${4:-1481bb99}"

die() {
    echo "install_overlay: $*" >&2
    exit 1
}

check_context() {
    [ -f "$SRC/BASE_SHA256.txt" ] || die "$SRC/BASE_SHA256.txt is missing; every overlay file that replaces a base file needs its base hash listed there"
    for name in tests cells run_tests.sh tests_image tests_headroom logs INSTALL-NOTES.md prebuild.md SHA256SUMS; do
        [ ! -e "$SRC/$name" ] || die "overlay/$name is in the build context; build from the lane folder so .dockerignore applies (./build_image.sh)"
    done
}

verify_base() {
    local listed
    listed="$(grep -v -E '^[[:space:]]*(#|$)' "$SRC/BASE_SHA256.txt")" || die "BASE_SHA256.txt has no hash lines"
    (cd "$PKG" && printf '%s\n' "$listed" | sha256sum --strict -c -) \
        || die "a base file in the image differs from BASE_SHA256.txt (see FAILED above); the base image is not $BASE_NAME or the list is stale"
}

has_base_hash() {
    awk -v wanted="$1" '$2 == wanted { found = 1 } END { exit !found }' "$SRC/BASE_SHA256.txt"
}

install_one() {
    local rel="$1"
    if [ -e "$PKG/$rel" ]; then
        has_base_hash "$rel" || die "overlay/$rel replaces a base file but BASE_SHA256.txt has no hash for it; add '<sha256>  $rel' from image $BASE_NAME"
        mkdir -p "$(dirname "$STATE/base/$rel")"
        cp "$PKG/$rel" "$STATE/base/$rel"
        cp "$SRC/$rel" "$PKG/$rel"
        return
    fi
    mkdir -p "$(dirname "$PKG/$rel")"
    cp "$SRC/$rel" "$PKG/$rel"
    chmod 0644 "$PKG/$rel"
}

main() {
    check_context
    verify_base
    local files
    files="$(cd "$SRC" && find . -type f ! -name BASE_SHA256.txt -printf '%P\n' | sort)"
    [ -n "$files" ] || die "overlay has no files to install"
    while IFS= read -r rel; do
        install_one "$rel"
    done <<< "$files"
    (cd "$PKG" && printf '%s\n' "$files" | xargs -d '\n' sha256sum) > "$STATE/INSTALLED_SHA256.txt"
    (cd "$PKG" && printf '%s\n' "$files" | grep -E '\.py$' | xargs -d '\n' python3 -B -c \
        'import sys; [compile(open(p, encoding="utf-8").read(), p, "exec") for p in sys.argv[1:]]') \
        || die "a Python file in the overlay does not compile; fix the syntax error above"
    echo "install_overlay: installed $(printf '%s\n' "$files" | wc -l) files, base hashes verified, all .py compile"
}

main
