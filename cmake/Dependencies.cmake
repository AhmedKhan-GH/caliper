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
# Copy our othercmake.txt wrapper if it doesn't exist
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
# Copy our othercmake.txt wrapper if it doesn't exist
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
    message(STATUS "  Using pre-built PyTorch 2.5.1 for Windows...")

    set(PYTORCH_VERSION "2.5.1")

    # Select CUDA or CPU variant based on USE_CUDA option
    if(USE_CUDA)
        set(PYTORCH_VARIANT "cu121")
        set(PYTORCH_VARIANT_NAME "CUDA 12.1")
    else()
        set(PYTORCH_VARIANT "cpu")
        set(PYTORCH_VARIANT_NAME "CPU-only")
    endif()

    # Select Debug or Release version based on build type
    if(CMAKE_BUILD_TYPE MATCHES "Debug")
        set(PYTORCH_URL "https://download.pytorch.org/libtorch/${PYTORCH_VARIANT}/libtorch-win-shared-with-deps-debug-${PYTORCH_VERSION}%2B${PYTORCH_VARIANT}.zip")
        set(PYTORCH_BUILD_TYPE "Debug")
    else()
        set(PYTORCH_URL "https://download.pytorch.org/libtorch/${PYTORCH_VARIANT}/libtorch-win-shared-with-deps-${PYTORCH_VERSION}%2B${PYTORCH_VARIANT}.zip")
        set(PYTORCH_BUILD_TYPE "Release")
    endif()

    set(LIBTORCH_DIR "${THIRD_PARTY_DIR}/libtorch")

    # Download and extract if not already present
    if(NOT EXISTS "${LIBTORCH_DIR}")
        message(STATUS "  Downloading PyTorch ${PYTORCH_BUILD_TYPE} (${PYTORCH_VARIANT_NAME})...")
        message(STATUS "  This is a ~2GB download and may take several minutes...")

        file(DOWNLOAD
            ${PYTORCH_URL}
            "${CMAKE_BINARY_DIR}/libtorch.zip"
            SHOW_PROGRESS
            STATUS DOWNLOAD_STATUS
        )

        list(GET DOWNLOAD_STATUS 0 DOWNLOAD_ERROR)
        if(DOWNLOAD_ERROR)
            list(GET DOWNLOAD_STATUS 1 DOWNLOAD_ERROR_MSG)
            message(FATAL_ERROR "Failed to download PyTorch: ${DOWNLOAD_ERROR_MSG}")
        endif()

        message(STATUS "  Extracting PyTorch to ${THIRD_PARTY_DIR}...")
        execute_process(
            COMMAND ${CMAKE_COMMAND} -E tar xzf "${CMAKE_BINARY_DIR}/libtorch.zip"
            WORKING_DIRECTORY ${THIRD_PARTY_DIR}
        )

        # Clean up zip file
        file(REMOVE "${CMAKE_BINARY_DIR}/libtorch.zip")
        message(STATUS "  ✓ PyTorch extracted successfully")
    else()
        message(STATUS "  ✓ PyTorch already downloaded (${PYTORCH_VARIANT_NAME})")
    endif()

    # Add LibTorch to CMAKE_PREFIX_PATH
    list(APPEND CMAKE_PREFIX_PATH ${LIBTORCH_DIR})

    # Workaround for missing CUDA::nvToolsExt in CUDA 12+
    if(USE_CUDA AND NOT TARGET CUDA::nvToolsExt)
        add_library(CUDA::nvToolsExt INTERFACE IMPORTED)
    endif()

    # Find Torch package (standard approach)
    find_package(Torch REQUIRED)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

    # Use Torch's provided libraries
    list(APPEND CALIPER_DEPENDENCY_LIBS "${TORCH_LIBRARIES}")

    message(STATUS "  ✓ PyTorch configured via find_package(Torch)")

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
# Setup PyTorch libraries for macOS/Linux (build from source)
# ============================================================================

if(NOT WIN32)
    # Set PyTorch directory for non-Windows platforms
    set(PYTORCH_INSTALL_DIR "${CMAKE_CURRENT_BINARY_DIR}/pytorch_install")

    # Platform-specific library names
    set(TORCH_LIB "${PYTORCH_INSTALL_DIR}/lib/libtorch${CMAKE_SHARED_LIBRARY_SUFFIX}")
    set(TORCH_CPU_LIB "${PYTORCH_INSTALL_DIR}/lib/libtorch_cpu${CMAKE_SHARED_LIBRARY_SUFFIX}")
    set(C10_LIB "${PYTORCH_INSTALL_DIR}/lib/libc10${CMAKE_SHARED_LIBRARY_SUFFIX}")

    # Create interface libraries for PyTorch components
    add_library(torch SHARED IMPORTED GLOBAL)
    set_target_properties(torch PROPERTIES
        IMPORTED_LOCATION ${TORCH_LIB}
        INTERFACE_INCLUDE_DIRECTORIES "${PYTORCH_INSTALL_DIR}/include;${PYTORCH_INSTALL_DIR}/include/torch/csrc/api/include"
    )

    add_library(torch_cpu SHARED IMPORTED GLOBAL)
    set_target_properties(torch_cpu PROPERTIES
        IMPORTED_LOCATION ${TORCH_CPU_LIB}
    )

    add_library(c10 SHARED IMPORTED GLOBAL)
    set_target_properties(c10 PROPERTIES
        IMPORTED_LOCATION ${C10_LIB}
    )

    # Add PyTorch libraries to dependency list
    list(APPEND CALIPER_DEPENDENCY_LIBS torch torch_cpu c10)

    message(STATUS "  ✓ PyTorch configured (built from source)")
endif()

# ============================================================================
# Export dependency list
# ============================================================================

# CALIPER_DEPENDENCY_LIBS is already in the correct scope
# (include() doesn't create a new scope, so no PARENT_SCOPE needed)

message(STATUS "")
message(STATUS "Dependencies configured successfully:")
message(STATUS "  Total libraries to link: ${CALIPER_DEPENDENCY_LIBS}")
message(STATUS "")
