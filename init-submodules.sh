#!/usr/bin/env bash

# ============================================================================
# Caliper Submodule Initialization Script
# ============================================================================
# This script initializes all git submodules for the Caliper project.
# Run this after cloning the repository for the first time.
#
# Usage:
#   ./init-submodules.sh
# ============================================================================

set -e
set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${SCRIPT_DIR}"

echo "============================================"
echo "Initializing Caliper Submodules"
echo "============================================"
echo ""

cd "${PROJECT_ROOT}"

# Step 1: Sync submodule URLs
echo "Step 1: Syncing submodule URLs..."
git submodule sync
echo "✓ Submodule URLs synced"
echo ""

# Step 2: Initialize and update top-level submodules
echo "Step 2: Initializing top-level submodules..."
echo "  - glfw (window management)"
echo "  - glm (math library)"
echo "  - imgui (UI library)"
echo "  - implot (plotting library)"
echo "  - pytorch (deep learning framework)"
git submodule update --init
echo "✓ Top-level submodules initialized"
echo ""

# Step 3: Initialize PyTorch's nested submodules
echo "Step 3: Initializing PyTorch submodules (this may take a while)..."
cd "${PROJECT_ROOT}/third_party/pytorch"
git submodule sync
git submodule update --init --recursive
echo "✓ PyTorch submodules initialized"
echo ""

cd "${PROJECT_ROOT}"

echo "============================================"
echo "Submodule initialization complete!"
echo "============================================"
echo ""
echo "You can now build the project:"
echo "  mkdir build && cd build"
echo "  cmake .."
echo "  cmake --build ."
echo ""
