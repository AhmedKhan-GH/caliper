#!/usr/bin/env bash

set -e
set -o pipefail

############################
# Resolve paths
############################

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PARENT_DIR="$(cd "${PROJECT_ROOT}/.." && pwd)"

GLFW_SRC="${PARENT_DIR}/external/glfw"
THIRD_PARTY_DIR="${PROJECT_ROOT}/third-party"
BUILD_DIR="${THIRD_PARTY_DIR}/glfw-build"
INSTALL_DIR="${THIRD_PARTY_DIR}/glfw"

JOBS=$(sysctl -n hw.ncpu)

############################
# Sanity checks
############################

if [ ! -d "${GLFW_SRC}" ]; then
    echo "Error: GLFW repo not found at:"
    echo "  ${GLFW_SRC}"
    exit 1
fi

if [ ! -f "${GLFW_SRC}/CMakeLists.txt" ]; then
    echo "Error: Invalid GLFW source directory."
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
mkdir -p "${INSTALL_DIR}"

cd "${BUILD_DIR}"

############################
# Configure CMake
############################

cmake \
  -DCMAKE_BUILD_TYPE=Release \
  -DBUILD_SHARED_LIBS=OFF \
  -DGLFW_BUILD_EXAMPLES=OFF \
  -DGLFW_BUILD_TESTS=OFF \
  -DGLFW_BUILD_DOCS=OFF \
  -DCMAKE_INSTALL_PREFIX="${INSTALL_DIR}" \
  "${GLFW_SRC}"

############################
# Build and install
############################

cmake --build . --target install -- -j${JOBS}

############################
# Cleanup
############################

echo "Cleaning up build directory..."
rm -rf "${BUILD_DIR}"

############################
# Done
############################

echo ""
echo "GLFW successfully built."
echo "Installed to:"
echo "  ${INSTALL_DIR}"
echo ""
echo "Use in CMake with:"
echo "  set(CMAKE_PREFIX_PATH \"\${CMAKE_CURRENT_SOURCE_DIR}/third-party/glfw\")"
echo "  find_package(glfw3 REQUIRED)"
echo "  target_link_libraries(your_target glfw)"
echo ""
