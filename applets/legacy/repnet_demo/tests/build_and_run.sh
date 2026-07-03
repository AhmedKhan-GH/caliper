#!/usr/bin/env bash
# Build and run a single Training Lab unit test against libtorch directly
# (no full caliper build). Usage: tests/build_and_run.sh <test.cpp> [extra .cpp ...]
# Run from the caliper repo root. Golden dir resolved via GOLDEN_DIR or default.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"  # caliper repo root
cd "$ROOT"
LT="third_party/libtorch"
NLOHMANN="third_party/llama.cpp/vendor"   # provides nlohmann/json.hpp

test_cpp="$1"; shift
out="/tmp/$(basename "${test_cpp%.cpp}")"

/usr/bin/c++ -std=gnu++17 -arch arm64 -O1 -g \
  "$test_cpp" "$@" \
  -I applets/repnet_demo -I applets/repnet_demo/train \
  -I applets/repnet_demo/tests \
  -isystem "$NLOHMANN" \
  -isystem "$LT/include" -isystem "$LT/include/torch/csrc/api/include" \
  -L "$LT/lib" -ltorch -ltorch_cpu -lc10 \
  -Wl,-rpath,"$ROOT/$LT/lib" -Wno-unknown-pragmas \
  -o "$out"

GOLDEN_DIR="${GOLDEN_DIR:-$ROOT/applets/repnet_demo/tests/golden}" "$out"
