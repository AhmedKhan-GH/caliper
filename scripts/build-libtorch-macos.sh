#!/usr/bin/env bash

set -e
set -o pipefail

############################
# Resolve paths
############################

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PARENT_DIR="$(cd "${PROJECT_ROOT}/.." && pwd)"

PYTORCH_SRC="${PARENT_DIR}/external/pytorch"
THIRD_PARTY_DIR="${PROJECT_ROOT}/third-party"
BUILD_DIR="${THIRD_PARTY_DIR}/libtorch-build"
INSTALL_DIR="${THIRD_PARTY_DIR}/libtorch"

JOBS=$(sysctl -n hw.ncpu)

############################
# Sanity checks
############################

if [ ! -d "${PYTORCH_SRC}" ]; then
    echo "Error: PyTorch repo not found at:"
    echo "  ${PYTORCH_SRC}"
    exit 1
fi

if [ ! -f "${PYTORCH_SRC}/CMakeLists.txt" ]; then
    echo "Error: Invalid PyTorch source directory."
    exit 1
fi

############################
# Initialize submodules
############################

echo "Initializing PyTorch submodules..."

cd "${PYTORCH_SRC}"
git submodule sync
git submodule update --init --recursive

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
  -DBUILD_SHARED_LIBS=ON \
  -DBUILD_PYTHON=OFF \
  -DBUILD_TEST=OFF \
  -DUSE_CUDA=OFF \
  -DUSE_MPS=ON \
  -DUSE_DISTRIBUTED=OFF \
  -DUSE_MKLDNN=ON \
  -DUSE_NNPACK=OFF \
  -DUSE_QNNPACK=OFF \
  -DUSE_XNNPACK=OFF \
  -DCMAKE_INSTALL_PREFIX="${INSTALL_DIR}" \
  "${PYTORCH_SRC}"

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
echo "LibTorch successfully built."
echo "Installed to:"
echo "  ${INSTALL_DIR}"
echo ""
echo "Use in CMake with:"
echo "  set(CMAKE_PREFIX_PATH \"${INSTALL_DIR}\")"
echo ""
