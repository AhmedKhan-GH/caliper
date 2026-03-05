#!/bin/bash

# Script to properly initialize git submodules
# This removes build artifacts and existing submodule directories before reinitializing

echo "Starting submodule initialization process..."

# Remove build directories
echo "Removing build directories..."
rm -rf cmake-build-debug
rm -rf cmake-build-release
rm -rf build

# Deinitialize and clean submodules first
echo "Cleaning git submodules..."
git submodule deinit -f --all 2>/dev/null || true

# Remove submodule entries from .git
echo "Removing .git/modules..."
rm -rf .git/modules

# Remove all third_party directories forcefully
echo "Removing all third_party directories..."
find third_party -mindepth 1 -delete 2>/dev/null || true
rm -rf third_party
mkdir -p third_party

# Remove pytorch from .gitmodules permanently (it's not a real submodule)
echo "Removing pytorch from .gitmodules..."
if grep -q "third_party/pytorch" .gitmodules 2>/dev/null; then
    git config -f .gitmodules --remove-section submodule.third_party/pytorch
    git add .gitmodules
    git commit -m "Remove pytorch submodule (using libtorch binaries instead)" --no-verify || true
    echo "  ✓ PyTorch submodule entry removed"
fi

# Remove pytorch from .git/config as well
git config --remove-section submodule.third_party/pytorch 2>/dev/null || true

# Sync submodule URLs
echo "Syncing submodule URLs..."
git submodule sync --recursive

# Initialize and update submodules (excluding pytorch)
echo "Initializing submodules..."
git submodule update --init --recursive --force

# Generate GLEW sources (only needed on macOS/Linux, Windows uses pre-built binaries)
if [[ "$OSTYPE" != "msys" && "$OSTYPE" != "win32" && "$OSTYPE" != "cygwin" ]]; then
    echo "Generating GLEW sources..."
    if [ -d "third_party/glew" ]; then
        cd third_party/glew
        make extensions > /dev/null 2>&1
        if [ -f "src/glew.c" ]; then
            echo "  ✓ GLEW sources generated successfully"
        else
            echo "  ✗ GLEW source generation failed"
        fi
        cd ../..
    else
        echo "  ✗ GLEW submodule not found"
    fi
else
    echo "Skipping GLEW source generation on Windows (using pre-built binaries)"
fi

echo ""
echo "Submodule initialization complete!"
echo "Note: PyTorch libtorch binaries will be downloaded by CMake during build."
