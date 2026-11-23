#!/usr/bin/env bash
# ATTN-PIN test runner: the host suite (fake KV stack, any python3 >= 3.10) and the real-import suite inside IMAGE
# (CPU only: --network none, HABANA_VISIBLE_DEVICES=none, no card). The image on disk is never modified: the overlay is
# copied into the throw-away container's dist-packages so the .pth runs the way it will in the bake.
# usage: bash run_tests.sh IMAGE   (required: the image the pin is baked on, e.g. the r4g tag; no default)
set -euo pipefail
KIT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
IMG="${1:?usage: run_tests.sh IMAGE (the image the pin is baked on, e.g. claude-flash-dual-graphs-c1d:<r4g tag>); there is no default}"
docker image inspect "$IMG" > /dev/null || { echo "run_tests.sh: image $IMG not found; pass a local tag" >&2; exit 2; }
SITE=/usr/local/lib/python3.12/dist-packages
mkdir -p "$KIT/logs"

echo "== host suite =="
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$KIT" python3 -m unittest discover -s "$KIT/tests" -t "$KIT/tests" 2>&1 \
  | tee "$KIT/logs/host.log"

echo "== craft =="
python3 "$KIT/tests/check_craft.py" "$KIT/glm53_attn_pin.py" "$KIT"/tests/*.py "$KIT"/tests_image/*.py \
  | tee -a "$KIT/logs/host.log"

# The real-Scheduler cases run in a flag-on child with HPU_GLM53_ATTENTION_ONLY_OFFLOAD=1 (the r4e offload shape).
echo "== image suite ($IMG) =="
docker run --rm --network none --entrypoint bash \
  -e PYTHONDONTWRITEBYTECODE=1 -e HABANA_VISIBLE_DEVICES=none -e PINKIT=/pinkit -e HPU_GLM53_ATTENTION_ONLY_OFFLOAD=1 \
  -v "$KIT:/pinkit:ro" "$IMG" -c "
    set -euo pipefail
    cp /pinkit/glm53_attn_pin.py /pinkit/glm53_attn_pin.pth $SITE/
    cd /pinkit/tests_image
    python3 -m unittest -v test_image_attn_pin 2>&1 | grep -Ev '^(INFO|WARNING|DEBUG)'
  " | tee "$KIT/logs/image.log"
