# Caliper

EEG/ECG waveform analysis toolkit with deep learning capabilities, built on PyTorch with GPU acceleration for NVIDIA (CUDA) and Apple Silicon (MPS).

---

## Quick Start

### Default Build (3 commands)

```bash
git submodule update --init --recursive   # Download dependencies (~2GB)
mkdir build && cd build                    # Create build directory
cmake .. && cmake --build . -- -j8        # Build everything
```

**First build**: 10-30 minutes (PyTorch compilation)
**Subsequent builds**: ~10 seconds (PyTorch cached)

### Run

```bash
./caliper
```

That's it! 🎉

---

## Build Options

All options are set during the `cmake ..` step.

### Platform-Specific Defaults

**macOS:**
```bash
cmake ..  # MPS (Metal) enabled by default
```

**Linux/Windows:**
```bash
cmake ..  # CUDA enabled by default if toolkit detected
```

### Custom Options

| Option | Default | Description |
|--------|---------|-------------|
| `USE_CUDA` | ON (Linux/Win), OFF (macOS) | Enable NVIDIA CUDA support |
| `USE_MPS` | ON (macOS), OFF (others) | Enable Apple Silicon GPU |
| `BUILD_TESTS` | OFF | Build test suite |
| `CMAKE_BUILD_TYPE` | Release | Build type (Release/Debug) |

**Examples:**
```bash
cmake .. -DUSE_CUDA=ON -DCMAKE_BUILD_TYPE=Debug
cmake .. -DBUILD_TESTS=ON
```

---

## Dataset

The project uses the **Nightingale ECG Dataset**:
- **12-lead ECG** (I, II, III, aVR, aVL, aVF, V1-V6)
- **3,750 patients**
- **~5,500 samples** per patient per lead (11 seconds at 500 Hz)
- **Binary classification**: RWMA (Regional Wall Motion Abnormality)
  - 341 positive cases (9.1%)
  - 3,409 negative cases (90.9%)

Place dataset in `data/Nightingale Dataset/` (gitignored by default).

---

## Features

- 📊 **Waveform Analysis** - Real-time ECG/EEG visualization and statistics
- 🧠 **Deep Learning Ready** - PyTorch backend for classification models
- 🎯 **GPU Acceleration** - CUDA (NVIDIA) and MPS (Apple Silicon)
- 📈 **Dataset Support** - Nightingale ECG dataset (12-lead, 3750 patients, RWMA classification)
- 🚀 **PyTorch-style Build** - All dependencies from source via git submodules
- 🖥️ **Interactive UI** - ImGui/ImPlot for visualization and analysis

---

## Dependencies (All Auto-Built)

### From Git Submodules
- **[PyTorch](https://github.com/pytorch/pytorch)** (~2GB) - Built from source, 10-30 min first time
- **[ImGui](https://github.com/ocornut/imgui)** (8MB) - Immediate mode GUI
- **[ImPlot](https://github.com/epezent/implot)** (748KB) - Plotting library
- **[GLFW](https://github.com/glfw/glfw)** (5MB) - Window/input management
- **[GLM](https://github.com/g-truc/glm)** (24MB) - OpenGL mathematics

### System Libraries (Auto-Detected)
- OpenGL - Graphics API
- Metal/MetalPerformanceShaders (macOS) - GPU acceleration

**No manual downloads required!** Everything is handled by git submodules and CMake.

---

## Platform Support

### macOS
✅ **Apple Silicon (M1/M2/M3)** - MPS GPU acceleration (default ON)
✅ **Intel** - CPU optimizations
**Requires**: Xcode Command Line Tools (`xcode-select --install`)

### Linux
✅ **x86_64** - CUDA support (auto-enabled if toolkit detected)
**Requires**:
- `sudo apt-get install libx11-dev libxrandr-dev libxinerama-dev libxcursor-dev libxi-dev libgl1-mesa-dev`
- CUDA Toolkit 11.8+ (optional, for GPU acceleration)

### Windows
✅ **x86_64** - CUDA support (auto-enabled if toolkit detected)
**Requires**:
- Visual Studio 2019 or later (with C++ desktop development)
- CUDA Toolkit 11.8+ (optional, for GPU acceleration)

---

## Performance

### Build Times (Apple Silicon M1 Pro)
- **First build**: 10-30 minutes (PyTorch compilation)
- **Subsequent builds**: ~10 seconds (only your code)
- **CMake configure**: ~1 second

### Runtime (MPS vs CPU on Apple Silicon)
- Matrix multiplication: **15x faster**
- Convolution: **15x faster**
- Neural network training: **7-10x faster**

---

## Project Structure

```
caliper/
├── CMakeLists.txt              # Main build configuration
├── README.md                   # This file
├── main.cpp                    # Application source
├── cmake/
│   └── Dependencies.cmake      # Dependency orchestration
├── data/                       # Dataset directory (gitignored)
│   └── Nightingale Dataset/
│       ├── MDC_ECG_LEAD_I.csv
│       ├── MDC_ECG_LEAD_II.csv
│       ├── ... (12 leads total)
│       └── rwma-outcomes.csv
├── third_party/                # Git submodules (tracked as references only)
│   ├── pytorch/
│   ├── imgui/
│   ├── implot/
│   ├── glfw/
│   └── glm/
└── build/                      # Generated (don't commit)
    ├── pytorch_build/          # PyTorch build artifacts
    ├── pytorch_install/        # Built PyTorch (cached)
    └── caliper                 # Executable
```

---

---

## Advanced Topics

### Building PyTorch from Source

PyTorch is built automatically from the `third_party/pytorch` git submodule. The build is cached after the first run.

**First build**: 10-30 minutes (compiles PyTorch)
**Subsequent builds**: Cached (0 seconds)

To force rebuild:
```bash
rm -rf build/libtorch build/pytorch-build
```

To pin a specific PyTorch version:
```bash
cd third_party/pytorch
git checkout v2.3.1  # or any tag/commit
cd ../..
git add third_party/pytorch
```

### Using GPU Acceleration

**Apple Silicon (MPS):**
```cpp
torch::Device device(torch::kMPS);
auto tensor = torch::randn({1000, 1000}, device);
model.to(device);
```

**NVIDIA (CUDA):**
```cpp
torch::Device device(torch::kCUDA);
auto tensor = torch::randn({1000, 1000}, device);
model.to(device);
```

**Performance**: 7-15x speedup over CPU for neural network training.

---

## Troubleshooting

### "PyTorch source not found"
```bash
git submodule update --init --recursive
```

### "CMake configuration failed"
Make sure you have build tools:
```bash
# macOS
xcode-select --install

# Linux
sudo apt-get install build-essential cmake
```

### Out of memory during build
Reduce parallel jobs:
```bash
cmake --build . -- -j2   # Use 2 cores instead of 8
```

### Clean rebuild
```bash
rm -rf build/
mkdir build && cd build
cmake .. && cmake --build . -- -j8
```

---

## How It Works

1. **`git submodule update`** downloads all source code (~2GB)
2. **`cmake ..`** configures build (detects platform, sets options)
3. **`cmake --build`** compiles:
   - PyTorch from source (10-30 min, cached after)
   - ImGui, ImPlot, GLFW, GLM (~30 seconds)
   - Your application (~10 seconds)

**Result**: Single executable with everything built from source, optimized for your platform!

---

## License

[Your License Here]

---

## Acknowledgments

Build system inspired by [PyTorch](https://github.com/pytorch/pytorch).

**Dependencies**:
- PyTorch by Meta AI
- ImGui by Omar Cornut
- ImPlot by Evan Pezent
- GLFW by Marcus Geelnard and Camilla Löwy
- GLM by Christophe Riccio
