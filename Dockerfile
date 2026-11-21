# claude-flash-dual-sustain: the whole current overlay/ copied onto image 1481bb99 (RC56 tree), base files hash-verified.
# Build with ./build_image.sh (it checks disk, protects the base images and runs the tests). To build by hand:
#   docker build -t claude-flash-dual-sustain:20261106 .      (from this folder: .dockerignore keeps overlay/tests and cells out)
# overlay/ is read at build time, so files other lanes add to it are picked up with no edit here.
ARG BASE_IMAGE=sha256:1481bb99fce076776c9e88604219726df33a903514015aa13fb0efc2350c8bb
FROM ${BASE_IMAGE}
ARG OVERLAY_SHA256=unset
COPY install_overlay.sh /opt/sustain-overlay/install_overlay.sh
COPY overlay/ /opt/sustain-overlay/src/
RUN bash /opt/sustain-overlay/install_overlay.sh /opt/sustain-overlay/src /usr/local/lib/python3.12/dist-packages /opt/sustain-overlay
LABEL claude.flash.sustain="1481bb99 + overlay (miss attribution, tripwire v2, contiguous sidecar, ...); every new flag defaults off" \
      claude.flash.sustain.base="1481bb99fce0 (claude-flash-final-h4:20261105)" \
      claude.flash.sustain.overlay-sha256="${OVERLAY_SHA256}"
