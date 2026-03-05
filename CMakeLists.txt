cmake_minimum_required(VERSION 3.18)
project(caliper CXX C)

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
# Main Application
# ============================================================================

# Application sources
add_executable(caliper main.cpp)

# No CUDA test executables - CPU/MPS only build

# Link all dependencies
target_link_libraries(caliper PRIVATE ${CALIPER_DEPENDENCY_LIBS})

# Define USE_CUDA preprocessor macro if CUDA is enabled
if(USE_CUDA)
    target_compile_definitions(caliper PRIVATE USE_CUDA)
endif()

# Set C++ standard
set_target_properties(caliper PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
)

# MSVC-specific settings
if(MSVC)
    # Set warning level and enable parallel compilation
    target_compile_options(caliper PRIVATE /W3 /MP)

    # Suppress specific warnings from PyTorch headers
    target_compile_options(caliper PRIVATE
        /wd4251  # class needs to have dll-interface
        /wd4275  # non dll-interface class used as base
    )

    # Set runtime library to match PyTorch (Multi-threaded DLL)
    set_property(TARGET caliper PROPERTY MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>DLL")

    # Copy PyTorch DLLs to executable directory after build
    file(GLOB TORCH_DLLS "${CMAKE_SOURCE_DIR}/third_party/libtorch/lib/*.dll")
    add_custom_command(TARGET caliper POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            ${TORCH_DLLS}
            $<TARGET_FILE_DIR:caliper>
        COMMENT "Copying PyTorch DLLs to executable directory"
    )
endif()

# Set RPATH for dynamic libraries
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
