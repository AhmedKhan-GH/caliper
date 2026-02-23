#!/usr/bin/env bash

set -e
set -o pipefail

############################
# Resolve paths
############################

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PARENT_DIR="$(cd "${PROJECT_ROOT}/.." && pwd)"

GLM_SRC="${PARENT_DIR}/external/glm"
THIRD_PARTY_DIR="${PROJECT_ROOT}/third-party"
BUILD_DIR="${THIRD_PARTY_DIR}/glm-build"
INSTALL_DIR="${THIRD_PARTY_DIR}/glm"

JOBS=$(sysctl -n hw.ncpu)

############################
# Sanity checks
############################

if [ ! -d "${GLM_SRC}" ]; then
    echo "Error: GLM repo not found at:"
    echo "  ${GLM_SRC}"
    exit 1
fi

if [ ! -f "${GLM_SRC}/CMakeLists.txt" ]; then
    echo "Error: Invalid GLM source directory."
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
  -DGLM_BUILD_TESTS=OFF \
  -DCMAKE_INSTALL_PREFIX="${INSTALL_DIR}" \
  "${GLM_SRC}"

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
echo "GLM successfully built."
echo "Installed to:"
echo "  ${INSTALL_DIR}"
echo ""
echo "Use in CMake with:"
echo "  set(CMAKE_PREFIX_PATH \"\${CMAKE_CURRENT_SOURCE_DIR}/third-party/glm\")"
echo "  find_package(glm REQUIRED)"
echo "  target_link_libraries(your_target glm::glm)"
echo ""
