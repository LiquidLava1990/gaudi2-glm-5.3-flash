#!/bin/bash
# promote-r4h/champion_check.sh: sourced by serve_champion.sh --check AFTER diag-r4e/r4e_run.sh (never run). Its baked_sha replaces
# r4e_run.sh's (docker run --rm --entrypoint sha256sum, which starts a container): the check hashes the image's file from a created
# container that never starts (no network, no Habana devices), then removes that container. r4e_run.sh itself is not edited.

baked_sha() {  # FILE: its sha8 inside $IMAGE ($SITE/FILE); prints nothing when the image lacks it
    local id
    id="$(${DOCKER:-docker} create --network none -e HABANA_VISIBLE_DEVICES=none --entrypoint true "$IMAGE" 2> /dev/null)" || return 0
    ${DOCKER:-docker} cp "$id:$SITE/$1" - 2> /dev/null | tar -xOf - 2> /dev/null | sha256sum | cut -c1-8 | grep -v -x e3b0c442
    ${DOCKER:-docker} rm "$id" > /dev/null 2>&1 || true
}
