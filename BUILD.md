# Caliper Build Instructions

Caliper uses a PyTorch-style build system with git submodules for dependency management.

## Prerequisites

- CMake 3.18 or higher
- C++17 compatible compiler (clang, gcc, or MSVC)
- Git
- Platform-specific requirements:
  - **macOS**: Xcode Command Line Tools
  - **Linux**: OpenGL development libraries
  - **Windows**: Visual Studio 2019 or later

## Quick Start

### 1. Clone the Repository

```bash
git clone <your-repo-url>
cd caliper
```

### 2. Initialize Submodules

Run the initialization script to download all dependencies:

```bash
./init-submodules.sh
```

This will:
- Initialize all top-level submodules (GLFW, GLM, ImGui, ImPlot, PyTorch)
- Initialize PyTorch's nested submodules (this may take a while)

**Note:** PyTorch has 35+ nested submodules, so this step may take 5-10 minutes depending on your internet connection.

### 3. Build the Project

```bash
mkdir build
cd build
cmake ..
cmake --build . -- -j$(nproc)
```

On macOS with MPS (Metal Performance Shaders) support:
```bash
cmake -DUSE_MPS=ON ..
cmake --build . -- -j$(sysctl -n hw.ncpu)
```

On Linux/Windows with CUDA support:
```bash
cmake -DUSE_CUDA=ON ..
cmake --build . -- -j$(nproc)
```

## Build Options

You can customize the build using CMake options:

| Option | Default | Description |
|--------|---------|-------------|
| `USE_CUDA` | ON (Linux/Windows), OFF (macOS) | Enable CUDA support |
| `USE_MPS` | ON (macOS with Metal), OFF (others) | Enable Metal Performance Shaders (Apple Silicon) |
| `BUILD_TESTS` | OFF | Build test suite |
| `CMAKE_BUILD_TYPE` | Release | Build type (Release, Debug, RelWithDebInfo) |

Example:
```bash
cmake -DUSE_CUDA=ON -DUSE_MPS=OFF -DCMAKE_BUILD_TYPE=Debug ..
```

## Build System Architecture

Caliper follows PyTorch's dependency management pattern:

### 1. System Libraries
- OpenGL (found on system)
- Platform frameworks (Cocoa, IOKit, CoreVideo on macOS)

### 2. Header-Only Libraries
- GLM (Mathematics)

### 3. CMake-Based Libraries
- GLFW (Window management)
- ImGui (UI library)
- ImPlot (Plotting)

### 4. Complex External Projects
- PyTorch (Built using `ExternalProject_Add`)

### Directory Structure

```
caliper/
├── .gitmodules              # Submodule definitions
├── CMakeLists.txt          # Root CMake configuration
├── cmake/
│   └── Dependencies.cmake  # Dependency orchestration
├── scripts/
│   └── init-submodules.sh  # Submodule initialization
├── third_party/            # All dependencies as submodules
│   ├── glfw/
│   ├── glm/
│   ├── imgui/
│   ├── implot/
│   └── pytorch/           # With its own nested submodules
└── build/                  # Build directory (created by user)
    ├── pytorch_build/      # PyTorch build artifacts
    └── pytorch_install/    # PyTorch installation

```

## How It Works

1. **Submodule Initialization**: `init-submodules.sh` downloads all dependencies
2. **CMake Configuration**: `cmake/Dependencies.cmake` orchestrates the build:
   - System libraries: Found via `find_package()`
   - Simple libraries: Built via `add_subdirectory()`
   - PyTorch: Built via `ExternalProject_Add()`
3. **Build**: All dependencies are built in dependency order
4. **Link**: Final executable links against all built libraries

## Troubleshooting

### Submodules not initialized
```
Error: Git submodules not initialized
```
**Solution**: Run `./init-submodules.sh`

### PyTorch submodules missing
```
Error: PyTorch submodules not initialized
```
**Solution**:
```bash
cd third_party/pytorch
git submodule update --init --recursive
```

### Build fails with "ExternalProject_Add not found"
**Solution**: Update CMake to 3.18 or higher

### Out of disk space during PyTorch build
PyTorch build requires ~10GB of disk space. The build directory can be cleaned after:
```bash
rm -rf build/pytorch_build/
```

The installed libraries in `build/pytorch_install/` are still needed (~2GB).

## Manual Submodule Initialization

If you prefer not to use the script:

```bash
# Initialize top-level submodules
git submodule sync
git submodule update --init

# Initialize PyTorch's nested submodules
cd third_party/pytorch
git submodule sync
git submodule update --init --recursive
cd ../..
```

## Environment Variables

You can control the build with environment variables:

```bash
# Limit parallel build jobs (useful for low-memory systems)
MAX_JOBS=4 cmake --build .

# Force specific compiler
CC=clang CXX=clang++ cmake ..
```

## Clean Build

To start from scratch:

```bash
rm -rf build/
mkdir build && cd build
cmake ..
cmake --build .
```

## Development Tips

- Use `EXCLUDE_FROM_ALL` in `add_subdirectory()` to skip building unused targets
- PyTorch is built once and cached in `build/pytorch_install/`
- Incremental builds of your own code are fast after initial PyTorch build
- Use `ccache` to speed up rebuilds

## Platform-Specific Notes

### macOS
- MPS (Metal) support is auto-detected if Metal compiler is available
- Uses Cocoa, IOKit, and CoreVideo frameworks

### Linux
- CUDA support is enabled by default (if CUDA toolkit is found)
- May need to install OpenGL development packages: `sudo apt install libgl1-mesa-dev`

### Windows
- Use Visual Studio developer command prompt
- CUDA support is enabled by default (if CUDA toolkit is found)
