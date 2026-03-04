# Caliper

Cross-platform machine learning application with ImGui interface supporting CUDA (Windows) and MPS (macOS).

---

## Quick Start

```bash
# 1. Initialize submodules
git submodule update --init --recursive

# 2. Configure (Windows CUDA)
cmake -B build -DUSE_CUDA=ON -DUSE_MPS=OFF

# 2. Configure (macOS MPS)
cmake -B build -DUSE_CUDA=OFF -DUSE_MPS=ON

# 3. Build
cmake --build build --config Release -j8

# 4. Run
.\build\caliper.exe          # Windows
./build/caliper              # macOS
```

---

## Prerequisites

### All Platforms
- **Git** (for submodule management)
- **CMake 3.18+**
- **C++17 compatible compiler**
- **OpenGL** (system library)

### Windows-Specific (CUDA)
⚠️ **Important**: These exact versions must match your device configuration:
- **Visual Studio 2022** (MSVC v143)
- **CUDA Toolkit 12.8** at `C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.8`
- **LibTorch 2.5.1+cu121** (auto-downloaded, ~2GB)
- **NVIDIA GPU** with compute capability 3.5+

💡 **To use different versions**: Update paths in `CMakeLists.txt:27-30` and LibTorch URL in `cmake/Dependencies.cmake:147-162`

### macOS-Specific (MPS)
- **Xcode** with command line tools
- **Apple Silicon** (M1/M2/M3)
- **Metal compiler** (xcrun --find metal)

---

## Build Options

| Option | Default | Description |
|--------|---------|-------------|
| `USE_CUDA` | ON (Windows), OFF (macOS) | Enable NVIDIA CUDA support |
| `USE_MPS` | ON (macOS), OFF (Windows) | Enable Apple Silicon GPU |
| `BUILD_TESTS` | OFF | Build test suite |

**Examples:**
```bash
cmake -B build -DUSE_CUDA=ON -DCMAKE_BUILD_TYPE=Debug
cmake -B build -DBUILD_TESTS=ON
```

---

## Project Structure

```
caliper/
├── CMakeLists.txt              # Main build config
├── cmake/
│   ├── Dependencies.cmake      # Dependency management
│   └── wrappers/               # ImGui/ImPlot CMake wrappers
├── third_party/                # Git submodules (glfw, glm, imgui, implot)
└── main.cpp                    # Application entry point
```

---

## Dependencies

**System:**
- OpenGL (rendering)

**Submodules (built from source):**
- [GLFW](https://github.com/glfw/glfw) - Window management
- [GLM](https://github.com/g-truc/glm) - Math library (header-only)
- [ImGui](https://github.com/ocornut/imgui) - UI framework
- [ImPlot](https://github.com/epezent/implot) - Plotting

**Downloaded:**
- [LibTorch](https://pytorch.org) - Windows: pre-built with CUDA | macOS: built from source with MPS

**No manual downloads required!** Everything is handled by git submodules and CMake.

---

## Platform-Specific Notes

### Windows (CUDA)
- **First build**: 5-10 minutes (downloads pre-built PyTorch ~2GB)
- **Subsequent builds**: ~10 seconds (PyTorch cached)
- CUDA DLLs automatically copied to executable directory
- To force re-download: `rm -rf third_party/libtorch/`

### macOS (MPS)
- **First build**: 10-30 minutes (builds PyTorch from source)
- **Subsequent builds**: ~10 seconds (PyTorch cached)
- MPS support requires Metal compiler
- To force rebuild: `rm -rf build/pytorch_build build/pytorch_install`

### Performance (GPU vs CPU)
- Matrix operations: **15x faster**
- Neural network training: **7-10x faster**

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| **CUDA not available** | Verify `USE_CUDA=ON`, check DLLs in `build/Release/`, ensure CUDA 12.8 installed |
| **Submodules missing** | Run `git submodule update --init --recursive` |
| **ImGui/ImPlot errors** | Delete `third_party/imgui/CMakeLists.txt` and `third_party/implot/CMakeLists.txt`, reconfigure |
| **OpenGL not found** | Update graphics drivers |
| **LibTorch download fails** | Check internet; manually download from PyTorch website to `third_party/libtorch/` |
| **MPS not available** | Ensure Apple Silicon Mac, Metal compiler available (`xcrun --find metal`) |
| **Out of memory** | Reduce parallel jobs: `cmake --build build -j2` |
| **Clean rebuild** | `rm -rf build/ && cmake -B build && cmake --build build -j8` |

---

## GPU Usage Examples

### Apple Silicon (MPS)
```cpp
torch::Device device(torch::kMPS);
auto tensor = torch::randn({1000, 1000}, device);
model.to(device);
```

### NVIDIA (CUDA)
```cpp
torch::Device device(torch::kCUDA);
auto tensor = torch::randn({1000, 1000}, device);
model.to(device);
```

---

## Resources

- [GLFW](https://www.glfw.org/documentation.html) | [GLM](https://github.com/g-truc/glm) | [ImGui](https://github.com/ocornut/imgui/wiki) | [ImPlot](https://github.com/epezent/implot) | [PyTorch C++](https://pytorch.org/cppdocs/)

---

## License

[Your License Here]

---

## Acknowledgments

Build system inspired by [PyTorch](https://github.com/pytorch/pytorch).

**Dependencies:**
- PyTorch by Meta AI
- ImGui by Omar Cornut
- ImPlot by Evan Pezent
- GLFW by Marcus Geelnard and Camilla Löwy
- GLM by Christophe Riccio
