#!/usr/bin/env bash
# Installs the SDK from a configured and built core build folder and builds
# applets against it, each in its own build folder, the way a developer
# outside this repository would: the only link to the core is
# -DCMAKE_PREFIX_PATH=<prefix>.
#
#   scripts/build_applets_against_sdk.sh <core build folder> <work folder> [applet folder...]
#
# Without applet folders it builds examples/hello, examples/sine_scope and
# every applets/* folder that has a CMakeLists.txt. The SDK goes to
# <work folder>/caliper-sdk and the builds to <work folder>/build/<name>.
# CMAKE_GENERATOR and CMAKE_BUILD_TYPE from the environment are honoured
# (the build type defaults to Release).
set -euo pipefail

if [ $# -lt 2 ]; then
    echo "usage: $0 <core build folder> <work folder> [applet folder...]" >&2
    exit 2
fi
core_build="$1"
work="$2"
shift 2
# Under Git Bash on Windows, pwd -W gives the folder as D:/a/b, which native
# programs such as cmake read without MSYS path conversion; elsewhere it fails
# and pwd gives the usual path.
root="$(cd "$(dirname "$0")/.." && { pwd -W 2>/dev/null || pwd; })"

if [ $# -eq 0 ]; then
    set -- "$root/examples/hello" "$root/examples/sine_scope"
    for dir in "$root"/applets/*/; do
        [ -f "$dir/CMakeLists.txt" ] && set -- "$@" "${dir%/}"
    done
fi

prefix="$work/caliper-sdk"
rm -rf "$prefix"
cmake --install "$core_build" --prefix "$prefix" --component sdk > /dev/null

for dir in "$@"; do
    name="$(basename "$dir")"
    echo "== $name"
    cmake -S "$dir" -B "$work/build/$name" \
        -DCMAKE_PREFIX_PATH="$prefix" \
        -DCMAKE_BUILD_TYPE="${CMAKE_BUILD_TYPE:-Release}"
    cmake --build "$work/build/$name" --parallel
    ls "$work/build/$name/applets"
done
