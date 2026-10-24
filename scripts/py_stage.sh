#!/bin/sh
# Stage an importable `caliper` package from a CMake build tree without
# installing a wheel: <build-dir>/py/caliper = symlinks to python/caliper/*.py
# plus the built extension modules and the viewer binary.
#   scripts/py_stage.sh cmake-build-X   then   PYTHONPATH=cmake-build-X/py python ...
set -e
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BUILD="$(cd "$1" && pwd)"
PKG="$BUILD/py/caliper"
rm -rf "$PKG" && mkdir -p "$PKG/bin"
for f in "$ROOT"/python/caliper/*.py; do ln -sf "$f" "$PKG/$(basename "$f")"; done
for so in "$BUILD"/python/_native*.so "$BUILD"/python/_live*.so; do
    [ -e "$so" ] && ln -sf "$so" "$PKG/$(basename "$so")"
done
for v in "$BUILD/caliper-viewer" "$BUILD/caliper_viewer"; do [ -e "$v" ] && ln -sf "$v" "$PKG/bin/caliper-viewer"; done
echo "staged $PKG:"; ls "$PKG" "$PKG/bin"
