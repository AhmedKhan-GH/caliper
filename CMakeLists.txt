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

# Check for CUDA and auto-enable only if found
if(WIN32 OR (UNIX AND NOT APPLE))
    # Try to find CUDA toolkit with explicit search on Windows
    if(WIN32 AND NOT DEFINED CUDAToolkit_ROOT)
        # Search for CUDA in common Windows locations
        file(GLOB CUDA_VERSIONS "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v*")
        if(CUDA_VERSIONS)
            list(GET CUDA_VERSIONS -1 CUDA_PATH_GUESS)  # Get the latest version
            set(CUDAToolkit_ROOT "${CUDA_PATH_GUESS}" CACHE PATH "CUDA Toolkit root directory")
            message(STATUS "Found CUDA installation at: ${CUDA_PATH_GUESS}")
        endif()
    endif()

    # Try modern CMake CUDA detection first
    find_package(CUDAToolkit QUIET)
    if(CUDAToolkit_FOUND)
        option(USE_CUDA "Build with CUDA support" ON)
        message(STATUS "CUDA Toolkit found: version ${CUDAToolkit_VERSION}")
        if(WIN32)
            message(STATUS "Windows will download pre-built PyTorch with CUDA ${CUDAToolkit_VERSION} support")
        endif()
    else()
        # Fallback to older CUDA detection method
        find_package(CUDA QUIET)
        if(CUDA_FOUND)
            option(USE_CUDA "Build with CUDA support" ON)
            message(STATUS "CUDA found at: ${CUDA_TOOLKIT_ROOT_DIR}")
            if(WIN32)
                message(STATUS "Windows will download pre-built PyTorch with CUDA support")
            endif()
        else()
            option(USE_CUDA "Build with CUDA support" OFF)
            message(STATUS "CUDA not found. Building without CUDA support.")
            if(WIN32)
                message(STATUS "Windows will download pre-built PyTorch CPU version")
            endif()
        endif()
    endif()
else()
    option(USE_CUDA "Build with CUDA support" OFF)
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

# Link all dependencies
target_link_libraries(caliper PRIVATE ${CALIPER_DEPENDENCY_LIBS})

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
    add_custom_command(TARGET caliper POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_directory
            "${CMAKE_SOURCE_DIR}/third_party/libtorch/lib"
            "$<TARGET_FILE_DIR:caliper>"
        COMMENT "Copying PyTorch DLLs to executable directory"
    )
endif()

# Set RPATH for dynamic libraries
if(APPLE)
    set_target_properties(caliper PROPERTIES
        INSTALL_RPATH "@executable_path;@executable_path/pytorch_install/lib"
        BUILD_WITH_INSTALL_RPATH TRUE
    )
elseif(UNIX)
    set_target_properties(caliper PROPERTIES
        INSTALL_RPATH "$ORIGIN:$ORIGIN/pytorch_install/lib"
        BUILD_WITH_INSTALL_RPATH TRUE
    )
endif()

# Also set runtime search path for build directory
set_target_properties(caliper PROPERTIES
    BUILD_RPATH "${CMAKE_BINARY_DIR}/pytorch_install/lib"
)

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
