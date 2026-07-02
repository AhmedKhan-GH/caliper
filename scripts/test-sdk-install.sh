#!/usr/bin/env bash
# Phase 0 exit proof (PLATFORM.md §17): the SDK installs to a prefix and a
# standalone consumer builds against it via find_package — no monorepo paths.
set -euo pipefail
BUILD_DIR="${1:-build}"
PREFIX="$(mktemp -d)"
trap 'rm -rf "$PREFIX"' EXIT
cmake --install "$BUILD_DIR" --component sdk --prefix "$PREFIX" >/dev/null
cmake -S tests/sdk_install_probe -B "$PREFIX/probe-build" \
      -DCMAKE_PREFIX_PATH="$PREFIX" >/dev/null
cmake --build "$PREFIX/probe-build" >/dev/null
echo "sdk-install-probe: OK (prefix consumable via find_package(caliper-sdk))"
