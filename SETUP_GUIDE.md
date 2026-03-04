# Caliper Setup Guide: Adding Full Submodules with CUDA Support

This guide explains how to set up the Caliper project with all required submodules (imgui, implot, glm, glfw) while **preserving existing CUDA functionality**.

## Prerequisites

- **Git** (for submodule management)
- **CMake 3.18+**
- **C++17 compatible compiler**
- **CUDA 12.8** (already configured at `C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.8`)
- **OpenGL** (system library)

## Current Project Status

Your repository already has:
- ✅ CUDA 12.8 configured and working
- ✅ PyTorch/libtorch integration (with CUDA support)
- ✅ Submodules initialized: glfw, glm, imgui, implot
- ✅ CMake wrappers for imgui/implot
- ✅ Working build system

## Step 1: Verify Submodules

First, check that all submodules are properly initialized:

```bash
git submodule status
```

Expected output:
```
 3d89365fdb7449487993515156cf81011dc6121c third_party/glfw (3.4-82-g3d89365f)
 e7970a8b26732f1b0df9690f7180546f8c30e48e third_party/glm (0.9.5.3-2971-ge7970a8b)
 ba84d2d37253c89b17b2cb7b357e130093b79ef5 third_party/imgui (v1.62-5234-gba84d2d37)
 93c801b4bb801c5c11031d880b6af1d1f70bd79d third_party/implot (v0.17-31-g93c801b4b)
```

If any submodules show a `-` prefix (uninitialized), run:

```bash
git submodule update --init --recursive
```

## Step 2: Verify CMake Wrappers Exist

The project uses CMake wrappers for imgui and implot. Verify these files exist:

```
cmake/wrappers/imgui_CMakeLists.txt
cmake/wrappers/implot_CMakeLists.txt
```

These are automatically copied to the submodule directories during CMake configuration.

## Step 3: Configure the Project with CUDA

Run CMake configuration with CUDA enabled:

```bash
cmake -B build -DUSE_CUDA=ON
```

### Key Configuration Options:

- **`USE_CUDA=ON`** - Enable CUDA support (default: ON) ⚠️ **CRITICAL: Keep this ON**
- **`USE_MPS=OFF`** - Disable MPS (for Windows) (default: OFF)
- **`BUILD_TESTS=OFF`** - Skip test builds (default: OFF)

### What Happens During Configuration:

1. **CUDA Detection**: CMake finds CUDA 12.8 at the configured path
2. **Submodule Check**: Auto-initializes any missing submodules
3. **OpenGL Detection**: Finds system OpenGL library
4. **GLM Setup**: Configured as header-only library
5. **GLFW Build**: Compiled from submodule (window management)
6. **ImGui Build**: Compiled with OpenGL3 + GLFW backends
7. **ImPlot Build**: Compiled and linked with ImGui
8. **PyTorch Download**: Downloads pre-built libtorch with CUDA 12.1 support (~2GB)

## Step 4: Build the Project

```bash
cmake --build build --config Release
```

Or for parallel builds (faster):

```bash
cmake --build build --config Release -j 8
```

### Build Output:

The build will create:
- `build/Release/caliper.exe` - Main application
- `build/Release/*.dll` - PyTorch CUDA DLLs (auto-copied)

## Step 5: Verify CUDA Functionality

### Option A: Run the Main Application

```bash
.\build\Release\caliper.exe
```

### Option B: Build and Run CUDA Test (Optional)

You have a `cuda_test.cpp` file. To test CUDA independently:

1. Add to `CMakeLists.txt` (line 75, after main executable):

```cmake
# CUDA test executable
add_executable(cuda_test cuda_test.cpp)
target_link_libraries(cuda_test PRIVATE torch torch_cpu c10)
if(USE_CUDA)
    target_link_libraries(cuda_test PRIVATE torch_cuda c10_cuda)
    target_compile_definitions(cuda_test PRIVATE USE_CUDA)
endif()
```

2. Rebuild and run:

```bash
cmake --build build --config Release
.\build\Release\cuda_test.exe
```

Expected output:
```
LibTorch CUDA Test
==================

CUDA available: YES
CUDA device count: 1
Successfully created CUDA tensor:
...
```

## Step 6: Using the Libraries in Your Code

### Example: Basic ImGui + CUDA Application

```cpp
#include <torch/torch.h>
#include <imgui.h>
#include <imgui_impl_glfw.h>
#include <imgui_impl_opengl3.h>
#include <implot.h>
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>

int main() {
    // Initialize GLFW
    glfwInit();
    GLFWwindow* window = glfwCreateWindow(1280, 720, "Caliper", NULL, NULL);
    glfwMakeContextCurrent(window);

    // Initialize ImGui
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImPlot::CreateContext();
    ImGui_ImplGlfw_InitForOpenGL(window, true);
    ImGui_ImplOpenGL3_Init("#version 150");

    // CUDA functionality
    std::cout << "CUDA available: "
              << (torch::cuda::is_available() ? "YES" : "NO") << "\n";

    // Main loop
    while (!glfwWindowShouldClose(window)) {
        glfwPollEvents();

        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();

        // Your ImGui UI here
        ImGui::Begin("CUDA Status");
        ImGui::Text("CUDA Available: %s",
                    torch::cuda::is_available() ? "Yes" : "No");
        ImGui::End();

        ImGui::Render();
        glClear(GL_COLOR_BUFFER_BIT);
        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
        glfwSwapBuffers(window);
    }

    // Cleanup
    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImPlot::DestroyContext();
    ImGui::DestroyContext();
    glfwDestroyWindow(window);
    glfwTerminate();

    return 0;
}
```

## Project Architecture

```
caliper/
├── CMakeLists.txt              # Main build configuration (CUDA enabled)
├── cmake/
│   ├── Dependencies.cmake       # All dependency management
│   └── wrappers/
│       ├── imgui_CMakeLists.txt  # ImGui build wrapper
│       └── implot_CMakeLists.txt # ImPlot build wrapper
├── third_party/
│   ├── glfw/                    # Submodule: Window management
│   ├── glm/                     # Submodule: Math library (header-only)
│   ├── imgui/                   # Submodule: UI framework
│   ├── implot/                  # Submodule: Plotting library
│   └── libtorch/                # Downloaded: PyTorch with CUDA
├── main.cpp                     # Your main application
└── cuda_test.cpp                # CUDA verification test

```

## Dependency Tree

```
caliper (main application)
├── OpenGL (system)
├── glfw (submodule → built from source)
├── glm (submodule → header-only)
├── imgui (submodule → built from source)
│   ├── depends on: glfw
│   └── depends on: OpenGL
├── implot (submodule → built from source)
│   └── depends on: imgui
└── torch/libtorch (downloaded → pre-built with CUDA)
    ├── torch, torch_cpu, c10 (always)
    └── torch_cuda, c10_cuda (when USE_CUDA=ON) ⚠️ **CRITICAL**
```

## CUDA-Specific Configuration

### CUDA Paths (in CMakeLists.txt:26-30)

```cmake
if(USE_CUDA)
    set(ENV{CUDA_PATH} "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.8")
    set(CUDA_TOOLKIT_ROOT_DIR "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.8")
    list(APPEND CMAKE_PREFIX_PATH "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.8")
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -allow-unsupported-compiler")
```

⚠️ **DO NOT MODIFY** these paths unless your CUDA installation location changes.

### CUDA Libraries (in Dependencies.cmake:383-401)

When `USE_CUDA=ON`, the following libraries are automatically linked:
- `torch_cuda.lib` / `torch_cuda.dll`
- `c10_cuda.lib` / `c10_cuda.dll`

These are included in the pre-built PyTorch download.

## Troubleshooting

### Problem: CUDA Not Available in Application

**Solution:**
1. Verify `USE_CUDA=ON` during CMake configuration
2. Check that CUDA DLLs are copied to executable directory:
   ```bash
   dir build\Release\*.dll
   ```
3. Ensure CUDA 12.8 is installed at the correct path

### Problem: Submodules Not Found

**Solution:**
```bash
git submodule update --init --recursive
```

### Problem: ImGui/ImPlot Build Errors

**Solution:**
1. Verify wrapper files exist in `cmake/wrappers/`
2. Delete `third_party/imgui/CMakeLists.txt` and `third_party/implot/CMakeLists.txt`
3. Reconfigure CMake (wrappers will be auto-copied)

### Problem: OpenGL Not Found

**Solution:**
- Windows: OpenGL should be available with graphics drivers
- Check that graphics drivers are up to date

### Problem: PyTorch Download Fails

**Solution:**
1. Check internet connection
2. Manually download from: `https://download.pytorch.org/libtorch/cu121/libtorch-win-shared-with-deps-2.5.1%2Bcu121.zip`
3. Extract to `third_party/libtorch/`

## Important Notes

⚠️ **CRITICAL: Preserving CUDA Functionality**

1. **Always use `USE_CUDA=ON`** when configuring CMake
2. **Never modify** the CUDA paths in CMakeLists.txt lines 26-30
3. **Keep the CUDA libraries** in the dependency list (Dependencies.cmake:383-401)
4. **Ensure PyTorch CUDA variant** is downloaded (not CPU-only)
5. **Test CUDA** after any configuration changes using `cuda_test.cpp`

## Additional Resources

- **GLFW Documentation**: https://www.glfw.org/documentation.html
- **GLM Documentation**: https://github.com/g-truc/glm/blob/master/manual.md
- **ImGui Documentation**: https://github.com/ocornut/imgui/wiki
- **ImPlot Documentation**: https://github.com/epezent/implot
- **PyTorch C++ API**: https://pytorch.org/cppdocs/

## Summary

Your repository is **already configured** with all submodules and CUDA support. To get started:

```bash
# 1. Ensure submodules are initialized
git submodule update --init --recursive

# 2. Configure with CUDA enabled
cmake -B build -DUSE_CUDA=ON

# 3. Build the project
cmake --build build --config Release -j 8

# 4. Run the application
.\build\Release\caliper.exe
```

**The CUDA functionality is preserved** through the `USE_CUDA` option and PyTorch CUDA libraries. All UI libraries (imgui, implot, glfw, glm) are integrated and will build automatically without affecting CUDA support.
