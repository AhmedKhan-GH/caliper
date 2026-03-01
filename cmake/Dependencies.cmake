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

# Check if submodules are initialized, auto-init if not
if(NOT EXISTS "${THIRD_PARTY_DIR}/pytorch/CMakeLists.txt")
    message(STATUS "========================================")
    message(STATUS "Git submodules not initialized")
    message(STATUS "Initializing now (this may take 5-10 minutes)")
    message(STATUS "Downloading ~2GB from GitHub...")
    message(STATUS "========================================")

    execute_process(
        COMMAND git submodule update --init --recursive --progress
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
        RESULT_VARIABLE SUBMODULE_RESULT
    )

    if(NOT SUBMODULE_RESULT EQUAL 0)
        message(FATAL_ERROR
            "\n========================================\n"
            "Failed to initialize git submodules.\n"
            "Please run manually:\n"
            "  ./init-submodules.sh\n"
            "Or:\n"
            "  git submodule update --init --recursive\n"
            "========================================"
        )
    endif()

    message(STATUS "========================================")
    message(STATUS "Submodules initialized successfully!")
    message(STATUS "========================================")
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
add_subdirectory(${THIRD_PARTY_DIR}/imgui EXCLUDE_FROM_ALL)
list(APPEND CALIPER_DEPENDENCY_LIBS imgui)
message(STATUS "    ✓ ImGui configured")

# --- ImPlot (Plotting library) ---
message(STATUS "  Configuring ImPlot...")
add_subdirectory(${THIRD_PARTY_DIR}/implot EXCLUDE_FROM_ALL)
list(APPEND CALIPER_DEPENDENCY_LIBS implot)
message(STATUS "    ✓ ImPlot configured")

# ============================================================================
# Category 4: PyTorch (Large dependency - ExternalProject)
# ============================================================================

message(STATUS "Configuring PyTorch (libtorch)...")

# Include ExternalProject module
include(ExternalProject)

# Set PyTorch build directory
set(PYTORCH_SOURCE_DIR "${THIRD_PARTY_DIR}/pytorch")
set(PYTORCH_BUILD_DIR "${CMAKE_CURRENT_BINARY_DIR}/pytorch_build")
set(PYTORCH_INSTALL_DIR "${CMAKE_CURRENT_BINARY_DIR}/pytorch_install")

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

# Create placeholder directories for CMake validation
# (ExternalProject will populate these during build)
file(MAKE_DIRECTORY ${PYTORCH_INSTALL_DIR}/include)
file(MAKE_DIRECTORY ${PYTORCH_INSTALL_DIR}/include/torch/csrc/api/include)
file(MAKE_DIRECTORY ${PYTORCH_INSTALL_DIR}/lib)

# Build PyTorch as an external project
ExternalProject_Add(pytorch_external
    SOURCE_DIR ${PYTORCH_SOURCE_DIR}
    BINARY_DIR ${PYTORCH_BUILD_DIR}

    CMAKE_ARGS ${PYTORCH_CMAKE_ARGS}

    BUILD_COMMAND ${CMAKE_COMMAND} --build . --target install -- -j${PYTORCH_BUILD_JOBS}

    INSTALL_COMMAND ""

    # Output libraries that will be created
    BUILD_BYPRODUCTS
        ${PYTORCH_INSTALL_DIR}/lib/libtorch${CMAKE_SHARED_LIBRARY_SUFFIX}
        ${PYTORCH_INSTALL_DIR}/lib/libtorch_cpu${CMAKE_SHARED_LIBRARY_SUFFIX}
        ${PYTORCH_INSTALL_DIR}/lib/libc10${CMAKE_SHARED_LIBRARY_SUFFIX}

    # Build in source to avoid path issues
    BUILD_IN_SOURCE 0

    # Show build progress in terminal
    USES_TERMINAL_BUILD TRUE
    USES_TERMINAL_CONFIGURE TRUE
)

# Create interface libraries for PyTorch components
add_library(torch SHARED IMPORTED GLOBAL)
set_target_properties(torch PROPERTIES
    IMPORTED_LOCATION ${PYTORCH_INSTALL_DIR}/lib/libtorch${CMAKE_SHARED_LIBRARY_SUFFIX}
    INTERFACE_INCLUDE_DIRECTORIES "${PYTORCH_INSTALL_DIR}/include;${PYTORCH_INSTALL_DIR}/include/torch/csrc/api/include"
)
add_dependencies(torch pytorch_external)

add_library(torch_cpu SHARED IMPORTED GLOBAL)
set_target_properties(torch_cpu PROPERTIES
    IMPORTED_LOCATION ${PYTORCH_INSTALL_DIR}/lib/libtorch_cpu${CMAKE_SHARED_LIBRARY_SUFFIX}
)
add_dependencies(torch_cpu pytorch_external)

add_library(c10 SHARED IMPORTED GLOBAL)
set_target_properties(c10 PROPERTIES
    IMPORTED_LOCATION ${PYTORCH_INSTALL_DIR}/lib/libc10${CMAKE_SHARED_LIBRARY_SUFFIX}
)
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
