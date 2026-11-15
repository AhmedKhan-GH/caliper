#!/usr/bin/env bash
# Build Caliper's view layer for the browser (web/wasm/README.md).
#
#   EMSDK=<path to an activated emsdk> web/wasm/build.sh <build dir> [Release|Debug]
#
# Needs cmake and ninja on PATH and an emsdk checkout with a version activated
# (measured with Emscripten 6.0.13). The first build downloads the emdawnwebgpu
# port into the emsdk cache. The output is <build dir>/caliper-views.js and
# <build dir>/caliper-views.wasm.
set -euo pipefail

here="$(cd "$(dirname "$0")" && pwd)"
build="${1:?usage: build.sh <build dir> [Release|Debug]}"
config="${2:-Release}"
: "${EMSDK:?set EMSDK to an emsdk checkout with an activated version}"

# shellcheck disable=SC1091
source "$EMSDK/emsdk_env.sh" >/dev/null 2>&1
emcmake cmake -S "$here" -B "$build" -G Ninja -DCMAKE_BUILD_TYPE="$config" >/dev/null
cmake --build "$build"
ls -l "$build"/caliper-views.js "$build"/caliper-views.wasm
