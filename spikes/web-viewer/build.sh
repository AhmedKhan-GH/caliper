#!/usr/bin/env bash
# Build the web viewer spike (docs/plans/web-viewer-spike.md).
#
#   EMSDK=<path to an activated emsdk> spikes/web-viewer/build.sh <build dir> [Release|Debug]
#
# Needs cmake and ninja on PATH and an emsdk checkout with a version activated
# (the spike was measured with Emscripten 6.0.13). The first build downloads
# the emdawnwebgpu port into the emsdk cache.
set -euo pipefail

here="$(cd "$(dirname "$0")" && pwd)"
build="${1:?usage: build.sh <build dir> [Release|Debug]}"
config="${2:-Release}"
: "${EMSDK:?set EMSDK to an emsdk checkout with an activated version}"

# shellcheck disable=SC1091
source "$EMSDK/emsdk_env.sh" >/dev/null 2>&1
emcmake cmake -S "$here" -B "$build" -G Ninja -DCMAKE_BUILD_TYPE="$config"
cmake --build "$build"
ls -l "$build"/caliper_web_spike.js "$build"/caliper_web_spike.wasm "$build"/index.html
