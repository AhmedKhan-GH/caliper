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
if(NOT EXISTS "${THIRD_PARTY_DIR}/glfw/CMakeLists.txt")
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

    # Always show output for debugging
    if(SUBMODULE_INIT_OUTPUT)
        message(STATUS "Git output: ${SUBMODULE_INIT_OUTPUT}")
    endif()
    if(SUBMODULE_INIT_ERROR)
        message(STATUS "Git stderr: ${SUBMODULE_INIT_ERROR}")
    endif()

    if(NOT SUBMODULE_INIT_RESULT EQUAL 0)
        message(FATAL_ERROR
            "Failed to initialize git submodules!\n"
            "Error: ${SUBMODULE_INIT_ERROR}\n"
            "Please manually run: git submodule update --init --recursive"
        )
    endif()

    # Verify submodules actually exist after init
    if(NOT EXISTS "${THIRD_PARTY_DIR}/glfw/CMakeLists.txt")
        message(FATAL_ERROR
            "Git submodule command succeeded but glfw is still missing!\n"
            "This likely means submodules are not properly registered.\n"
            "Please manually run:\n"
            "  git submodule sync\n"
            "  git submodule update --init --recursive"
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

# --- GLEW (OpenGL Extension Wrangler) ---
message(STATUS "  Configuring GLEW...")

if(WIN32)
    # Windows: Download pre-built GLEW binaries
    set(GLEW_VERSION "2.3.1")
    set(GLEW_PREBUILT_DIR "${THIRD_PARTY_DIR}/glew-prebuilt")

    # Download and extract if not already present
    if(NOT EXISTS "${GLEW_PREBUILT_DIR}")
        message(STATUS "    Downloading pre-built GLEW ${GLEW_VERSION} for Windows...")

        file(DOWNLOAD
            "https://github.com/nigels-com/glew/releases/download/glew-${GLEW_VERSION}/glew-${GLEW_VERSION}-win32.zip"
            "${CMAKE_BINARY_DIR}/glew-win32.zip"
            SHOW_PROGRESS
            STATUS DOWNLOAD_STATUS
        )

        list(GET DOWNLOAD_STATUS 0 DOWNLOAD_ERROR)
        if(DOWNLOAD_ERROR)
            list(GET DOWNLOAD_STATUS 1 DOWNLOAD_ERROR_MSG)
            message(FATAL_ERROR "Failed to download GLEW: ${DOWNLOAD_ERROR_MSG}")
        endif()

        message(STATUS "    Extracting GLEW to ${THIRD_PARTY_DIR}...")
        execute_process(
            COMMAND ${CMAKE_COMMAND} -E tar xzf "${CMAKE_BINARY_DIR}/glew-win32.zip"
            WORKING_DIRECTORY ${THIRD_PARTY_DIR}
        )

        # Rename to glew-prebuilt for consistency
        file(RENAME "${THIRD_PARTY_DIR}/glew-${GLEW_VERSION}" "${GLEW_PREBUILT_DIR}")

        # Clean up zip file
        file(REMOVE "${CMAKE_BINARY_DIR}/glew-win32.zip")
        message(STATUS "    ✓ GLEW extracted successfully")
    else()
        message(STATUS "    ✓ GLEW already downloaded")
    endif()

    # Create imported target for GLEW
    add_library(libglew_static STATIC IMPORTED)

    # Determine architecture-specific paths
    if(CMAKE_SIZEOF_VOID_P EQUAL 8)
        set(GLEW_LIB_DIR "${GLEW_PREBUILT_DIR}/lib/Release/x64")
    else()
        set(GLEW_LIB_DIR "${GLEW_PREBUILT_DIR}/lib/Release/Win32")
    endif()

    set_target_properties(libglew_static PROPERTIES
        IMPORTED_LOCATION "${GLEW_LIB_DIR}/glew32s.lib"
        INTERFACE_INCLUDE_DIRECTORIES "${GLEW_PREBUILT_DIR}/include"
        INTERFACE_COMPILE_DEFINITIONS "GLEW_STATIC"
    )

else()
    # macOS/Linux: Build from source using CMake wrapper
    # Copy our CMake wrapper if it doesn't exist
    if(NOT EXISTS "${THIRD_PARTY_DIR}/glew/CMakeLists.txt")
        configure_file(
            "${CMAKE_CURRENT_SOURCE_DIR}/cmake/wrappers/glew_CMakeLists.txt"
            "${THIRD_PARTY_DIR}/glew/CMakeLists.txt"
            COPYONLY
        )
    endif()
    add_subdirectory(${THIRD_PARTY_DIR}/glew EXCLUDE_FROM_ALL)
endif()

list(APPEND CALIPER_DEPENDENCY_LIBS libglew_static)
message(STATUS "    ✓ GLEW configured")

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
    # macOS and Linux: Download pre-built libtorch
    message(STATUS "  Using pre-built PyTorch (libtorch) for macOS/Linux...")

    set(PYTORCH_VERSION "2.5.1")
    set(LIBTORCH_DIR "${THIRD_PARTY_DIR}/libtorch")

    # Determine platform and architecture
    if(APPLE)
        # macOS - CPU only (MPS support included)
        set(PYTORCH_URL "https://download.pytorch.org/libtorch/cpu/libtorch-macos-arm64-${PYTORCH_VERSION}.zip")
        set(PYTORCH_PLATFORM "macOS ARM64")
    else()
        # Linux
        if(USE_CUDA)
            set(PYTORCH_URL "https://download.pytorch.org/libtorch/cu121/libtorch-cxx11-abi-shared-with-deps-${PYTORCH_VERSION}%2Bcu121.zip")
            set(PYTORCH_PLATFORM "Linux CUDA 12.1")
        else()
            set(PYTORCH_URL "https://download.pytorch.org/libtorch/cpu/libtorch-cxx11-abi-shared-with-deps-${PYTORCH_VERSION}%2Bcpu.zip")
            set(PYTORCH_PLATFORM "Linux CPU")
        endif()
    endif()

    # Download and extract if not already present
    if(NOT EXISTS "${LIBTORCH_DIR}")
        message(STATUS "  Downloading PyTorch for ${PYTORCH_PLATFORM}...")
        message(STATUS "  This is a ~200MB download and may take several minutes...")

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
        message(STATUS "  ✓ PyTorch already downloaded (${PYTORCH_PLATFORM})")
    endif()

    # Add LibTorch to CMAKE_PREFIX_PATH
    list(APPEND CMAKE_PREFIX_PATH ${LIBTORCH_DIR})

    # Find Torch package (standard approach)
    find_package(Torch REQUIRED)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

    # Use Torch's provided libraries
    list(APPEND CALIPER_DEPENDENCY_LIBS "${TORCH_LIBRARIES}")

    message(STATUS "  ✓ PyTorch configured via find_package(Torch)")

endif()  # WIN32 vs macOS/Linux

# ============================================================================
# Export dependency list
# ============================================================================

# CALIPER_DEPENDENCY_LIBS is already in the correct scope
# (include() doesn't create a new scope, so no PARENT_SCOPE needed)

message(STATUS "")
message(STATUS "Dependencies configured successfully:")
message(STATUS "  Total libraries to link: ${CALIPER_DEPENDENCY_LIBS}")
message(STATUS "")
