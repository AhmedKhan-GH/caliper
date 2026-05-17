cmake_minimum_required(VERSION 3.18...4.2)
project(caliper CXX C OBJC)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Suppress CLion compiler info warnings
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Use all available cores for building
include(ProcessorCount)
ProcessorCount(N)
if(NOT N EQUAL 0)
    set(CMAKE_BUILD_PARALLEL_LEVEL ${N})
    message(STATUS "Building with ${N} parallel jobs")
endif()

# ============================================================================
# Build Options (PyTorch-style)
# ============================================================================

# Force static runtime library for all targets to match DuckDB
set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")
set(BUILD_SHARED_LIBS OFF)

# CUDA enabled
option(USE_CUDA "Build with CUDA support" ON)

# Set CUDA paths and configuration
if(USE_CUDA)
    set(ENV{CUDA_PATH} "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.8")
    set(CUDA_TOOLKIT_ROOT_DIR "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.8")
    list(APPEND CMAKE_PREFIX_PATH "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.8")
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -allow-unsupported-compiler")

    # Workaround for missing CUDA::nvToolsExt in CUDA 12+
    if(NOT TARGET CUDA::nvToolsExt)
        add_library(CUDA::nvToolsExt INTERFACE IMPORTED)
    endif()

    message(STATUS "CUDA support: ENABLED (CUDA 12.8)")
else()
    message(STATUS "CUDA support: DISABLED (CPU/MPS only)")
endif()

# Auto-enable MPS on macOS if Metal compiler is available
if(APPLE)
    execute_process(
        COMMAND xcrun --find metal
        RESULT_VARIABLE METAL_FOUND
        OUTPUT_QUIET ERROR_QUIET
    )
    if(METAL_FOUND EQUAL 0)
        option(USE_MPS "Build with MPS (Metal Performance Shaders) support for Apple Silicon" ON)
    else()
        option(USE_MPS "Build with MPS (Metal Performance Shaders) support for Apple Silicon" OFF)
        message(STATUS "Metal compiler not found. MPS support disabled.")
    endif()
else()
    option(USE_MPS "Build with MPS (Metal Performance Shaders) support for Apple Silicon" OFF)
endif()

option(BUILD_TESTS "Build tests" OFF)

# ============================================================================
# Include Dependencies (PyTorch-style architecture)
# ============================================================================

# This file handles all third-party dependencies
include(cmake/Dependencies.cmake)

# ============================================================================
# Applet SDK (interface target for shared library applets)
# ============================================================================

add_library(caliper_applet_sdk INTERFACE)
target_include_directories(caliper_applet_sdk INTERFACE
    ${CMAKE_SOURCE_DIR}/src
    ${THIRD_PARTY_DIR}/imgui
    ${THIRD_PARTY_DIR}/imgui/backends
    ${THIRD_PARTY_DIR}/implot
    ${THIRD_PARTY_DIR}/implot3d
    ${THIRD_PARTY_DIR}/ImGuiFileDialog
    ${THIRD_PARTY_DIR}/duckdb/src/include
)
target_link_libraries(caliper_applet_sdk INTERFACE
    imgui implot implot3d ImGuiFileDialog
)
target_compile_definitions(caliper_applet_sdk INTERFACE
    DUCKDB_BUILD_LIBRARY
    IMGUI_IMPL_OPENGL_LOADER_GLEW
)

# ============================================================================
# Main Application (host)
# ============================================================================

add_executable(caliper
    src/main.cpp
    src/intro_screen.cpp
    src/applet_host.cpp
    src/app_paths.cpp
)

# Host links UI libs + platform libs (no DuckDB, no LibTorch)
target_link_libraries(caliper PRIVATE ${CALIPER_DEPENDENCY_LIBS})

if(USE_CUDA)
    target_compile_definitions(caliper PRIVATE USE_CUDA)
endif()

set_target_properties(caliper PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
)

if(MSVC)
    target_compile_options(caliper PRIVATE /W3 /MP /wd4251 /wd4275)
    file(GLOB TORCH_DLLS "${CMAKE_SOURCE_DIR}/third_party/libtorch/lib/*.dll")
    add_custom_command(TARGET caliper POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            ${TORCH_DLLS}
            $<TARGET_FILE_DIR:caliper>
        COMMENT "Copying PyTorch DLLs to executable directory"
    )
endif()

if(APPLE)
    set_target_properties(caliper PROPERTIES
        INSTALL_RPATH "@executable_path;@executable_path/../third_party/libtorch/lib"
        BUILD_WITH_INSTALL_RPATH TRUE
        BUILD_RPATH "${CMAKE_SOURCE_DIR}/third_party/libtorch/lib"
    )
elseif(UNIX)
    set_target_properties(caliper PROPERTIES
        INSTALL_RPATH "$ORIGIN:$ORIGIN/../third_party/libtorch/lib"
        BUILD_WITH_INSTALL_RPATH TRUE
        BUILD_RPATH "${CMAKE_SOURCE_DIR}/third_party/libtorch/lib"
    )
endif()

# ============================================================================
# Applets (each builds a shared library)
# ============================================================================

add_subdirectory(applets/repnet_demo)
add_subdirectory(applets/opengllama)

# ============================================================================
# Summary
# ============================================================================

message(STATUS "")
message(STATUS "========================================")
message(STATUS "Caliper Build Configuration")
message(STATUS "========================================")
message(STATUS "  C++ Standard: ${CMAKE_CXX_STANDARD}")
message(STATUS "  Build Type: ${CMAKE_BUILD_TYPE}")
message(STATUS "  CUDA Support: ${USE_CUDA}")
message(STATUS "  MPS Support: ${USE_MPS}")
message(STATUS "  Parallel Jobs: ${N}")
message(STATUS "========================================")
message(STATUS "")
