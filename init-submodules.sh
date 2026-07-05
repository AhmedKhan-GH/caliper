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
#
# We update each submodule individually instead of a single
# `git submodule update --init --recursive --force`. That single command aborts
# at the first submodule whose pinned commit is missing from its remote, which
# leaves every submodule processed after it un-initialized. Some pins in this
# repo reference commits that were never pushed upstream (e.g. locally-generated
# GLEW sources, or a llama.cpp fork commit), so we tolerate those and fall back
# to the remote's default branch rather than failing the whole setup.
echo "Initializing submodules..."
git config -f .gitmodules --get-regexp 'path$' | awk '{print $2}' | while read -r sm_path; do
    sm_name=$(basename "$sm_path")
    if git submodule update --init --recursive --force "$sm_path" >/dev/null 2>&1; then
        echo "  ✓ $sm_name @ $(git -C "$sm_path" rev-parse --short HEAD)"
        continue
    fi

    # Pinned commit could not be checked out. This is almost always because the
    # commit recorded in the superproject is not fetchable from the submodule's
    # remote ("upload-pack: not our ref").
    sm_url=$(git config -f .gitmodules --get "submodule.$sm_path.url")
    echo "  ! $sm_name: pinned commit unavailable from $sm_url"

    if git -C "$sm_path" rev-parse HEAD >/dev/null 2>&1; then
        # `git submodule update` already cloned the repo at its default branch
        # before the checkout to the (missing) pinned commit failed. Keep it.
        echo "    -> using default branch instead: $(git -C "$sm_path" rev-parse --short HEAD)"
    else
        # Nothing usable on disk; clone the default branch ourselves.
        echo "    -> cloning default branch from $sm_url"
        rm -rf "$sm_path"
        if git clone --recursive "$sm_url" "$sm_path" >/dev/null 2>&1; then
            echo "    ✓ cloned $sm_name @ $(git -C "$sm_path" rev-parse --short HEAD)"
        else
            echo "    ✗ failed to clone $sm_name — build may not work"
        fi
    fi
done

# Generate GLEW sources (only needed on macOS/Linux, Windows uses pre-built binaries)
# Detect Windows more reliably
if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "win32" || "$OSTYPE" == "cygwin" || -n "$WINDIR" ]]; then
    echo "Skipping GLEW source generation on Windows (using pre-built binaries)"
else
    echo "Generating GLEW sources..."
    if [ -d "third_party/glew" ]; then
        cd third_party/glew
        # GLEW's make scripts invoke "python" — create a temporary shim
        # so it works whether the system has "python", "python3", or both.
        if ! command -v python &>/dev/null && command -v python3 &>/dev/null; then
            _glew_shim_dir=$(mktemp -d)
            ln -s "$(command -v python3)" "$_glew_shim_dir/python"
            export PATH="$_glew_shim_dir:$PATH"
        fi
        make extensions >/dev/null 2>&1
        # Clean up shim if we created one
        if [ -n "$_glew_shim_dir" ]; then
            rm -rf "$_glew_shim_dir"
            unset _glew_shim_dir
        fi
        if [ -f "src/glew.c" ]; then
            echo "  ✓ GLEW sources generated successfully"
        else
            echo "  ✗ GLEW source generation failed"
        fi
        cd ../..
    else
        echo "  ✗ GLEW submodule not found"
    fi
fi

echo ""
echo "Submodule initialization complete!"
echo "Note: PyTorch libtorch binaries will be downloaded by CMake during build."
