# ============================================================================
# Caliper Dependencies Configuration (PyTorch-style)
# ============================================================================
#
# This file orchestrates all third-party dependencies following PyTorch's
# architecture:
#   1. System libraries (find on system)
#   2. Header-only libraries (include paths only)
#   3. CMake-based libraries (add_subdirectory)
#   4. Complex libraries (ExternalProject_Add for PyTorch)
#
# ============================================================================

message(STATUS "Configuring Caliper dependencies...")

# Set third-party root directory
set(THIRD_PARTY_DIR "${CMAKE_CURRENT_SOURCE_DIR}/third_party")

# Check if submodules are initialized, and auto-initialize if needed
if(NOT EXISTS "${THIRD_PARTY_DIR}/pytorch/CMakeLists.txt")
    message(STATUS
        "========================================\n"
        "Git submodules not initialized!\n"
        "Automatically initializing submodules...\n"
        "This will download ~2GB and may take 5-10 minutes.\n"
        "========================================"
    )
    execute_process(
        COMMAND git submodule update --init --recursive
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
        RESULT_VARIABLE SUBMODULE_INIT_RESULT
        OUTPUT_VARIABLE SUBMODULE_INIT_OUTPUT
        ERROR_VARIABLE SUBMODULE_INIT_ERROR
    )
    if(NOT SUBMODULE_INIT_RESULT EQUAL 0)
        message(FATAL_ERROR
            "Failed to initialize git submodules!\n"
            "Error: ${SUBMODULE_INIT_ERROR}\n"
            "Please manually run: git submodule update --init --recursive"
        )
    endif()
    message(STATUS "✓ Submodules initialized successfully")
endif()

# ============================================================================
# Category 1: System Libraries (Find on system)
# ============================================================================

message(STATUS "Finding system libraries...")

# OpenGL (required for rendering)
find_package(OpenGL REQUIRED)
if(OpenGL_FOUND)
    message(STATUS "  ✓ OpenGL found")
    list(APPEND CALIPER_DEPENDENCY_LIBS OpenGL::GL)
else()
    message(FATAL_ERROR "OpenGL not found")
endif()

# Platform-specific frameworks (macOS)
if(APPLE)
    message(STATUS "  ✓ Adding macOS frameworks")
    list(APPEND CALIPER_DEPENDENCY_LIBS
        "-framework Cocoa"
        "-framework IOKit"
        "-framework CoreVideo"
    )
endif()

# ============================================================================
# Category 2: Header-Only Libraries
# ============================================================================

# GLM (Mathematics library - can be used header-only)
message(STATUS "Configuring GLM (header-only mode)...")
set(GLM_INCLUDE_DIR "${THIRD_PARTY_DIR}/glm")
include_directories(SYSTEM ${GLM_INCLUDE_DIR})
message(STATUS "  ✓ GLM configured")

# ============================================================================
# Category 3: CMake-Based Libraries (build from submodules)
# ============================================================================

message(STATUS "Configuring CMake-based dependencies...")

# --- GLFW (Window management) ---
message(STATUS "  Configuring GLFW...")
set(GLFW_BUILD_DOCS OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(GLFW_INSTALL OFF CACHE BOOL "" FORCE)
add_subdirectory(${THIRD_PARTY_DIR}/glfw EXCLUDE_FROM_ALL)
list(APPEND CALIPER_DEPENDENCY_LIBS glfw)
message(STATUS "    ✓ GLFW configured")

# --- ImGui (UI library) ---
message(STATUS "  Configuring ImGui...")
# Copy our CMakeLists.txt wrapper if it doesn't exist
if(NOT EXISTS "${THIRD_PARTY_DIR}/imgui/CMakeLists.txt")
    configure_file(
        "${CMAKE_CURRENT_SOURCE_DIR}/cmake/wrappers/imgui_CMakeLists.txt"
        "${THIRD_PARTY_DIR}/imgui/CMakeLists.txt"
        COPYONLY
    )
endif()
add_subdirectory(${THIRD_PARTY_DIR}/imgui EXCLUDE_FROM_ALL)
list(APPEND CALIPER_DEPENDENCY_LIBS imgui)
message(STATUS "    ✓ ImGui configured")

# --- ImPlot (Plotting library) ---
message(STATUS "  Configuring ImPlot...")
# Copy our CMakeLists.txt wrapper if it doesn't exist
if(NOT EXISTS "${THIRD_PARTY_DIR}/implot/CMakeLists.txt")
    configure_file(
        "${CMAKE_CURRENT_SOURCE_DIR}/cmake/wrappers/implot_CMakeLists.txt"
        "${THIRD_PARTY_DIR}/implot/CMakeLists.txt"
        COPYONLY
    )
endif()
add_subdirectory(${THIRD_PARTY_DIR}/implot EXCLUDE_FROM_ALL)
list(APPEND CALIPER_DEPENDENCY_LIBS implot)
message(STATUS "    ✓ ImPlot configured")

# ============================================================================
# Category 4: PyTorch (Large dependency)
# ============================================================================

message(STATUS "Configuring PyTorch (libtorch)...")

# Include ExternalProject module
include(ExternalProject)

# Set PyTorch install directory
set(PYTORCH_INSTALL_DIR "${CMAKE_CURRENT_BINARY_DIR}/pytorch_install")

# ============================================================================
# Windows: Download pre-built libtorch (CUDA or CPU)
# macOS/Linux: Build from source (for MPS support on macOS)
# ============================================================================

if(WIN32)
    message(STATUS "  Using pre-built PyTorch for Windows...")

    # Determine which version to download
    if(USE_CUDA AND DEFINED CUDAToolkit_VERSION)
        # Extract major version (e.g., 13.1 -> 13, 12.4 -> 12)
        string(REGEX MATCH "^([0-9]+)" CUDA_MAJOR_VERSION "${CUDAToolkit_VERSION}")

        # Map CUDA version to PyTorch supported version
        # PyTorch 2.5.1 supports: CUDA 11.8, 12.1, 12.4
        if(CUDA_MAJOR_VERSION EQUAL 13 OR CUDA_MAJOR_VERSION EQUAL 12)
            # CUDA 12.x or 13.x -> use PyTorch built for CUDA 12.4
            set(PYTORCH_CUDA_VERSION "124")
            set(PYTORCH_VARIANT "cu124")
            message(STATUS "  Detected CUDA ${CUDAToolkit_VERSION}, using PyTorch built for CUDA 12.4")
        elseif(CUDA_MAJOR_VERSION EQUAL 11)
            # CUDA 11.x -> use PyTorch built for CUDA 11.8
            set(PYTORCH_CUDA_VERSION "118")
            set(PYTORCH_VARIANT "cu118")
            message(STATUS "  Detected CUDA ${CUDAToolkit_VERSION}, using PyTorch built for CUDA 11.8")
        else()
            # Unsupported CUDA version, fall back to CPU
            set(PYTORCH_VARIANT "cpu")
            message(WARNING "CUDA ${CUDAToolkit_VERSION} not directly supported by PyTorch. Using CPU version.")
            message(WARNING "Supported CUDA versions: 11.8, 12.1, 12.4")
        endif()

        if(NOT PYTORCH_VARIANT STREQUAL "cpu")
            set(PYTORCH_URL "https://download.pytorch.org/libtorch/${PYTORCH_VARIANT}/libtorch-win-shared-with-deps-2.5.1%2B${PYTORCH_VARIANT}.zip")
            message(STATUS "  Downloading PyTorch with ${PYTORCH_VARIANT} support...")
        else()
            set(PYTORCH_URL "https://download.pytorch.org/libtorch/cpu/libtorch-win-shared-with-deps-2.5.1%2Bcpu.zip")
            message(STATUS "  Downloading PyTorch CPU-only version...")
        endif()
    else()
        set(PYTORCH_VARIANT "cpu")
        set(PYTORCH_URL "https://download.pytorch.org/libtorch/cpu/libtorch-win-shared-with-deps-2.5.1%2Bcpu.zip")
        message(STATUS "  Downloading PyTorch CPU-only version...")
    endif()

    # Download and extract pre-built libtorch
    ExternalProject_Add(pytorch_external
        URL ${PYTORCH_URL}
        DOWNLOAD_NO_PROGRESS FALSE
        DOWNLOAD_DIR "${CMAKE_CURRENT_BINARY_DIR}/downloads"
        SOURCE_DIR "${CMAKE_CURRENT_BINARY_DIR}/libtorch_download"

        # Copy files to the expected location after extraction
        CONFIGURE_COMMAND ${CMAKE_COMMAND} -E copy_directory
            "${CMAKE_CURRENT_BINARY_DIR}/libtorch_download"
            "${PYTORCH_INSTALL_DIR}"

        BUILD_COMMAND ""
        INSTALL_COMMAND ""

        # Declare the output files (paths defined below)
        BUILD_BYPRODUCTS
            "${PYTORCH_INSTALL_DIR}/lib/torch.lib"
            "${PYTORCH_INSTALL_DIR}/lib/torch_cpu.lib"
            "${PYTORCH_INSTALL_DIR}/lib/c10.lib"
            "${PYTORCH_INSTALL_DIR}/lib/asmjit.lib"
            "${PYTORCH_INSTALL_DIR}/lib/fbgemm.lib"

        LOG_DOWNLOAD TRUE
    )

    message(STATUS "  PyTorch will be downloaded on first build (~2GB download)")

else()
    # macOS and Linux: Build from source
    message(STATUS "  Building PyTorch from source...")

    set(PYTORCH_SOURCE_DIR "${THIRD_PARTY_DIR}/pytorch")
    set(PYTORCH_BUILD_DIR "${CMAKE_CURRENT_BINARY_DIR}/pytorch_build")

    # Check if PyTorch's own submodules are initialized
    if(NOT EXISTS "${PYTORCH_SOURCE_DIR}/third_party/pybind11/CMakeLists.txt")
        message(STATUS "  Initializing PyTorch submodules (this may take a while)...")
        execute_process(
            COMMAND git submodule update --init --recursive
            WORKING_DIRECTORY ${PYTORCH_SOURCE_DIR}
            RESULT_VARIABLE PYTORCH_SUBMODULE_RESULT
            OUTPUT_QUIET
        )
        if(NOT PYTORCH_SUBMODULE_RESULT EQUAL 0)
            message(FATAL_ERROR "Failed to initialize PyTorch submodules")
        endif()
        message(STATUS "    ✓ PyTorch submodules initialized")
    endif()
    # Configure PyTorch CMake arguments
    set(PYTORCH_CMAKE_ARGS
        -DCMAKE_BUILD_TYPE=Release
        -DCMAKE_INSTALL_PREFIX=${PYTORCH_INSTALL_DIR}
        -DCMAKE_PREFIX_PATH=${PYTORCH_INSTALL_DIR}

        # Build configuration
        -DBUILD_SHARED_LIBS=ON
        -DBUILD_PYTHON=OFF
        -DBUILD_TEST=OFF
        -DUSE_NUMPY=OFF

        # Disable unnecessary components
        -DUSE_GLOG=OFF
        -DUSE_GFLAGS=OFF

        # Distributed training (usually not needed for inference)
        -DUSE_DISTRIBUTED=OFF
        -DUSE_MPI=OFF
        -DUSE_GLOO=OFF
        -DUSE_NCCL=OFF

        # Disable some accelerators to speed up build
        -DUSE_NNPACK=OFF
        -DUSE_PYTORCH_QNNPACK=ON
        -DUSE_XNNPACK=OFF
    )

    # CUDA configuration
    if(USE_CUDA)
        message(STATUS "  CUDA support: ENABLED")

        # Pass CUDA paths to PyTorch if available
        if(DEFINED CUDAToolkit_ROOT)
            list(APPEND PYTORCH_CMAKE_ARGS
                -DCUDA_TOOLKIT_ROOT_DIR=${CUDAToolkit_ROOT}
                -DCMAKE_CUDA_COMPILER=${CUDAToolkit_ROOT}/bin/nvcc
            )
            message(STATUS "  Passing CUDA path to PyTorch: ${CUDAToolkit_ROOT}")
        endif()

        list(APPEND PYTORCH_CMAKE_ARGS
            -DUSE_CUDA=ON
            -DUSE_CUDNN=ON
        )
    else()
        message(STATUS "  CUDA support: DISABLED")
        list(APPEND PYTORCH_CMAKE_ARGS
            -DUSE_CUDA=OFF
            -DUSE_CUDNN=OFF
        )
    endif()

    # MPS configuration (Apple Silicon)
    if(USE_MPS)
        message(STATUS "  MPS support: ENABLED")
        list(APPEND PYTORCH_CMAKE_ARGS -DUSE_MPS=ON)
    else()
        message(STATUS "  MPS support: DISABLED")
        list(APPEND PYTORCH_CMAKE_ARGS -DUSE_MPS=OFF)
    endif()

    # Determine number of build jobs
    if(DEFINED ENV{MAX_JOBS})
        set(PYTORCH_BUILD_JOBS "$ENV{MAX_JOBS}")
    else()
        include(ProcessorCount)
        ProcessorCount(NUM_CORES)
        if(NUM_CORES EQUAL 0)
            set(PYTORCH_BUILD_JOBS 4)
        else()
            math(EXPR PYTORCH_BUILD_JOBS "${NUM_CORES}")
        endif()
    endif()

    message(STATUS "  Building PyTorch with ${PYTORCH_BUILD_JOBS} parallel jobs...")

    # Build PyTorch as an external project
    ExternalProject_Add(pytorch_external
        SOURCE_DIR ${PYTORCH_SOURCE_DIR}
        BINARY_DIR ${PYTORCH_BUILD_DIR}

        CMAKE_ARGS ${PYTORCH_CMAKE_ARGS}

        BUILD_COMMAND ${CMAKE_COMMAND} --build . --target install -- -j${PYTORCH_BUILD_JOBS}

        INSTALL_COMMAND ""

        # Build in source to avoid path issues
        BUILD_IN_SOURCE 0

        # Show build progress in terminal
        USES_TERMINAL_BUILD TRUE
        USES_TERMINAL_CONFIGURE TRUE
    )

endif()  # WIN32 vs build from source

# ============================================================================
# Setup PyTorch libraries (common for both download and build)
# ============================================================================

# Create placeholder directories for CMake validation
file(MAKE_DIRECTORY ${PYTORCH_INSTALL_DIR}/include)
file(MAKE_DIRECTORY ${PYTORCH_INSTALL_DIR}/include/torch/csrc/api/include)
file(MAKE_DIRECTORY ${PYTORCH_INSTALL_DIR}/lib)

# Platform-specific library names
if(WIN32)
    set(TORCH_LIB "${PYTORCH_INSTALL_DIR}/lib/torch.lib")
    set(TORCH_CPU_LIB "${PYTORCH_INSTALL_DIR}/lib/torch_cpu.lib")
    set(C10_LIB "${PYTORCH_INSTALL_DIR}/lib/c10.lib")
else()
    set(TORCH_LIB "${PYTORCH_INSTALL_DIR}/lib/libtorch${CMAKE_SHARED_LIBRARY_SUFFIX}")
    set(TORCH_CPU_LIB "${PYTORCH_INSTALL_DIR}/lib/libtorch_cpu${CMAKE_SHARED_LIBRARY_SUFFIX}")
    set(C10_LIB "${PYTORCH_INSTALL_DIR}/lib/libc10${CMAKE_SHARED_LIBRARY_SUFFIX}")
endif()

# Create interface libraries for PyTorch components
add_library(torch SHARED IMPORTED GLOBAL)
set_target_properties(torch PROPERTIES
    IMPORTED_LOCATION ${TORCH_LIB}
    INTERFACE_INCLUDE_DIRECTORIES "${PYTORCH_INSTALL_DIR}/include;${PYTORCH_INSTALL_DIR}/include/torch/csrc/api/include"
)
if(WIN32)
    set_target_properties(torch PROPERTIES
        IMPORTED_IMPLIB ${TORCH_LIB}
    )
endif()
add_dependencies(torch pytorch_external)

add_library(torch_cpu SHARED IMPORTED GLOBAL)
set_target_properties(torch_cpu PROPERTIES
    IMPORTED_LOCATION ${TORCH_CPU_LIB}
)
if(WIN32)
    set_target_properties(torch_cpu PROPERTIES
        IMPORTED_IMPLIB ${TORCH_CPU_LIB}
    )
endif()
add_dependencies(torch_cpu pytorch_external)

add_library(c10 SHARED IMPORTED GLOBAL)
set_target_properties(c10 PROPERTIES
    IMPORTED_LOCATION ${C10_LIB}
)
if(WIN32)
    set_target_properties(c10 PROPERTIES
        IMPORTED_IMPLIB ${C10_LIB}
    )
endif()
add_dependencies(c10 pytorch_external)

# Add PyTorch libraries to dependency list
list(APPEND CALIPER_DEPENDENCY_LIBS torch torch_cpu c10)

message(STATUS "  ✓ PyTorch configured")

# ============================================================================
# Export dependency list
# ============================================================================

# CALIPER_DEPENDENCY_LIBS is already in the correct scope
# (include() doesn't create a new scope, so no PARENT_SCOPE needed)

message(STATUS "")
message(STATUS "Dependencies configured successfully:")
message(STATUS "  Total libraries to link: ${CALIPER_DEPENDENCY_LIBS}")
message(STATUS "")
