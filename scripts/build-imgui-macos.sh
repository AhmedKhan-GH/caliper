#!/usr/bin/env bash

set -e
set -o pipefail

############################
# Resolve paths
############################

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PARENT_DIR="$(cd "${PROJECT_ROOT}/.." && pwd)"

IMGUI_SRC="${PARENT_DIR}/external/imgui"
THIRD_PARTY_DIR="${PROJECT_ROOT}/third-party"
BUILD_DIR="${THIRD_PARTY_DIR}/imgui-build"
INSTALL_DIR="${THIRD_PARTY_DIR}/imgui"

JOBS=$(sysctl -n hw.ncpu)

############################
# Sanity checks
############################

if [ ! -d "${IMGUI_SRC}" ]; then
    echo "Error: ImGui repo not found at:"
    echo "  ${IMGUI_SRC}"
    exit 1
fi

if [ ! -f "${IMGUI_SRC}/imgui.h" ]; then
    echo "Error: Invalid ImGui source directory."
    exit 1
fi

############################
# Environment (macOS)
############################

export MACOSX_DEPLOYMENT_TARGET=11.0
export CC=clang
export CXX=clang++

############################
# Prepare directories
############################

mkdir -p "${BUILD_DIR}"
mkdir -p "${INSTALL_DIR}/include"
mkdir -p "${INSTALL_DIR}/lib"

############################
# Copy headers
############################

echo "Copying ImGui headers..."

cp "${IMGUI_SRC}"/*.h "${INSTALL_DIR}/include/"
cp "${IMGUI_SRC}"/*.cpp "${INSTALL_DIR}/include/"
cp -r "${IMGUI_SRC}/backends" "${INSTALL_DIR}/include/"

############################
# Build static library
############################

echo "Building ImGui static library..."

cd "${BUILD_DIR}"

clang++ -c -std=c++17 -O3 \
    "${IMGUI_SRC}/imgui.cpp" \
    "${IMGUI_SRC}/imgui_demo.cpp" \
    "${IMGUI_SRC}/imgui_draw.cpp" \
    "${IMGUI_SRC}/imgui_tables.cpp" \
    "${IMGUI_SRC}/imgui_widgets.cpp"

ar rcs libimgui.a imgui.o imgui_demo.o imgui_draw.o imgui_tables.o imgui_widgets.o

cp libimgui.a "${INSTALL_DIR}/lib/"

############################
# Cleanup
############################

echo "Cleaning up build directory..."
rm -rf "${BUILD_DIR}"

############################
# Done
############################

echo ""
echo "ImGui successfully built."
echo "Installed to:"
echo "  ${INSTALL_DIR}"
echo ""
echo "Use in CMake with:"
echo "  include_directories(\"\${CMAKE_CURRENT_SOURCE_DIR}/third-party/imgui/include\")"
echo "  link_directories(\"\${CMAKE_CURRENT_SOURCE_DIR}/third-party/imgui/lib\")"
echo "  target_link_libraries(your_target imgui)"
echo ""
