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

## LibTorch Configuration

Caliper depends on **libtorch 2.5.1**. There are several flavors (CPU, CUDA
11.8, CUDA 12.1, CUDA 12.4, macOS arm64). The build picks one at *configure
time*. This section explains how to let it pick automatically vs. how to
override it.

### TL;DR

| You are on... | Run this | Result |
|---|---|---|
| Apple Silicon Mac | `cmake -B build .` | macOS arm64 (CPU + MPS, no flags needed) |
| Windows / Linux, no GPU | `cmake -B build .` | CPU |
| Windows / Linux, NVIDIA GPU | `cmake -B build -DLIBTORCH_VARIANT=cu121 .` | CUDA 12.1 |
| Want a specific CUDA | `cmake -B build -DLIBTORCH_VARIANT=cu124 .` | CUDA 12.4 |

If `LIBTORCH_VARIANT` is unset, the build picks a sensible default for your
platform. If you set it, the build does what you say.

### The variants

| `LIBTORCH_VARIANT` | What you get | Min NVIDIA driver |
|---|---|---|
| `cpu` | CPU-only build | (none) |
| `cu118` | CUDA 11.8 | 520.61 (Linux) / 522.06 (Windows) |
| `cu121` | CUDA 12.1 | 530.30 / 531.14 |
| `cu124` | CUDA 12.4 | 550.54 / 551.78 |
| `macos-arm64` | macOS arm64 (CPU + MPS) | (n/a — Apple Silicon, macOS 12.3+) |

Driver-version table comes from
[NVIDIA's CUDA compatibility matrix](https://docs.nvidia.com/deploy/cuda-compatibility/).

### Auto mode (default)

If you do **not** set `LIBTORCH_VARIANT`, the build chooses for you based on
the host platform:

```
host                         → variant
─────────────────────────────────────────
APPLE                        → macos-arm64
WIN32 / Linux + USE_CUDA=ON  → cu121
WIN32 / Linux + USE_CUDA=OFF → cpu
```

Auto mode is the right pick if you don't know or don't care. It will:

- Always work (CPU is the fallback).
- Use MPS on Apple Silicon at runtime, automatically.
- Use CUDA 12.1 on Linux/Windows when `USE_CUDA=ON`, which has the broadest
  driver support among current options.

### Custom mode (explicit override)

Pass `-DLIBTORCH_VARIANT=<flavor>` when configuring. Examples:

```bash
# Latest CUDA, requires driver ≥ 550.54
cmake -B build -DLIBTORCH_VARIANT=cu124 .

# Older CUDA for a machine with an older driver
cmake -B build -DLIBTORCH_VARIANT=cu118 .

# CPU on a CUDA-capable machine (debugging, perf comparison)
cmake -B build -DLIBTORCH_VARIANT=cpu .
```

Custom mode is the right pick when:

- You know your driver version doesn't support `cu121` (use `cu118`).
- You want bleeding-edge CUDA support (use `cu124`).
- You're on a Linux box with no NVIDIA GPU but want to verify CPU paths.
- You're shipping per-variant binaries from CI (each job hard-pins a variant).

### Switching variants after the fact

The download is cached at `third_party/libtorch/`. Switching variants
requires deleting it so the next configure pulls the right archive:

```bash
rm -rf third_party/libtorch
cmake -B build -DLIBTORCH_VARIANT=cu124 .
```

If you forget this step, you'll keep using the previously-downloaded variant
silently. (We may add a stamp-file check in the future to detect this.)

### What about MPS?

There is **no `--mps` variant**. MPS support is compiled into the
`macos-arm64` libtorch you already get on Apple Silicon. At runtime:

```cpp
torch::Device pick_device() {
    if (torch::cuda::is_available()) return torch::kCUDA;
    if (torch::mps::is_available())  return torch::kMPS;
    return torch::kCPU;
}
```

On an M1/M2/M3 Mac running macOS 12.3+, `torch::mps::is_available()` returns
true and your tensors run on the GPU via Apple's Metal Performance Shaders
backend. You don't ship anything different. You don't pick anything
different. It just works.

If you're on an Intel Mac, MPS isn't supported by hardware; libtorch falls
back to CPU automatically. Same binary.

### CI / multi-variant builds

To produce per-variant releases (e.g., `caliper-cu121`, `caliper-cu124`,
`caliper-cpu`), run a build matrix:

```yaml
strategy:
  matrix:
    variant: [cpu, cu118, cu121, cu124]
steps:
  - run: cmake -B build -DLIBTORCH_VARIANT=${{ matrix.variant }} .
  - run: cmake --build build
  - run: tar czf caliper-${{ matrix.variant }}.tar.gz build/caliper third_party/libtorch
```

Macs get their own job that uses auto mode (`macos-arm64`).

### How the runtime resolves which backend to actually use

Variant selection only decides *what gets shipped*. The decision of CPU vs
GPU at run time is separate, and made by libtorch:

| Variant shipped | At runtime, does the user's hardware support a GPU backend? | What runs on |
|---|---|---|
| `cu121` | Yes (`cuda::is_available()` true) | CUDA |
| `cu121` | No (no GPU / driver too old) | CPU (silent fallback) |
| `macos-arm64` | Apple Silicon + macOS 12.3+ | MPS |
| `macos-arm64` | Intel Mac or older macOS | CPU |
| `cpu` | (irrelevant) | CPU |

So a single CUDA build "works" on a machine without a GPU — it just runs on
CPU. The cost is: the binary carries CUDA libraries it'll never use. If
that's acceptable, ship one CUDA variant and call it done. If it isn't,
ship a CPU variant in parallel.

### LibTorch troubleshooting

**"CUDA out of memory" on first run** — your GPU's VRAM is smaller than
what the workload allocates. Lower batch size or fall back to CPU.

**"undefined symbol" linking against libtorch** — you've mixed libtorch
versions or variants. Delete `third_party/libtorch/` and reconfigure.

**MPS works on Python PyTorch but not from C++ libtorch** — confirm
`torch::mps::is_available()` at runtime; ensure your `.to()` calls go
through the helper that picks `kMPS`. The pre-built libtorch supports MPS
out of the box on Apple Silicon — there's no separate flag.

**Driver version mismatch on launch** (`forward compatibility was attempted
on non supported HW`) — your shipped variant requires a newer NVIDIA driver
than the user has. Either tell them to upgrade the driver, or rebuild with
an older `LIBTORCH_VARIANT` (e.g., `cu118`).

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
