#!/usr/bin/env bash

set -e
set -o pipefail

############################
# Resolve paths
############################

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PARENT_DIR="$(cd "${PROJECT_ROOT}/.." && pwd)"

IMPLOT_SRC="${PARENT_DIR}/external/implot"
THIRD_PARTY_DIR="${PROJECT_ROOT}/third-party"
BUILD_DIR="${THIRD_PARTY_DIR}/implot-build"
INSTALL_DIR="${THIRD_PARTY_DIR}/implot"

JOBS=$(sysctl -n hw.ncpu)

############################
# Sanity checks
############################

if [ ! -d "${IMPLOT_SRC}" ]; then
    echo "Error: ImPlot repo not found at:"
    echo "  ${IMPLOT_SRC}"
    exit 1
fi

if [ ! -f "${IMPLOT_SRC}/implot.h" ]; then
    echo "Error: Invalid ImPlot source directory."
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

echo "Copying ImPlot headers..."

cp "${IMPLOT_SRC}"/*.h "${INSTALL_DIR}/include/"
cp "${IMPLOT_SRC}"/*.cpp "${INSTALL_DIR}/include/"

############################
# Build static library
############################

echo "Building ImPlot static library..."

cd "${BUILD_DIR}"

# ImPlot requires ImGui headers
IMGUI_INCLUDE="${THIRD_PARTY_DIR}/imgui/include"

if [ ! -d "${IMGUI_INCLUDE}" ]; then
    echo "Error: ImGui headers not found at ${IMGUI_INCLUDE}"
    echo "Please run build-imgui-macos.sh first."
    exit 1
fi

clang++ -c -std=c++17 -O3 \
    -I"${IMGUI_INCLUDE}" \
    "${IMPLOT_SRC}/implot.cpp" \
    "${IMPLOT_SRC}/implot_items.cpp" \
    "${IMPLOT_SRC}/implot_demo.cpp"

ar rcs libimplot.a implot.o implot_items.o implot_demo.o

cp libimplot.a "${INSTALL_DIR}/lib/"

############################
# Cleanup
############################

echo "Cleaning up build directory..."
rm -rf "${BUILD_DIR}"

############################
# Done
############################

echo ""
echo "ImPlot successfully built."
echo "Installed to:"
echo "  ${INSTALL_DIR}"
echo ""
echo "Use in CMake with:"
echo "  include_directories(\"\${CMAKE_CURRENT_SOURCE_DIR}/third-party/implot/include\")"
echo "  link_directories(\"\${CMAKE_CURRENT_SOURCE_DIR}/third-party/implot/lib\")"
echo "  target_link_libraries(your_target implot)"
echo ""
