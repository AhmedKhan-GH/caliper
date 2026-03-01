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
endif()

# Add CMake module path
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake/Modules")

# Build configuration
# Auto-enable CUDA on Windows and Linux by default
if(WIN32 OR (UNIX AND NOT APPLE))
    option(USE_CUDA "Build with CUDA support" ON)
else()
    option(USE_CUDA "Build with CUDA support" OFF)
endif()
# Auto-enable MPS on macOS if Metal compiler is available
if(APPLE)
    # Check if Metal compiler is available
    execute_process(
        COMMAND xcrun --find metal
        RESULT_VARIABLE METAL_FOUND
        OUTPUT_QUIET ERROR_QUIET
    )
    if(METAL_FOUND EQUAL 0)
        option(USE_MPS "Build with MPS (Metal Performance Shaders) support for Apple Silicon" ON)
    else()
        option(USE_MPS "Build with MPS (Metal Performance Shaders) support for Apple Silicon" OFF)
        message(STATUS "Metal compiler not found. MPS support disabled. Install Xcode to enable MPS.")
    endif()
else()
    option(USE_MPS "Build with MPS (Metal Performance Shaders) support for Apple Silicon" OFF)
endif()
option(BUILD_TESTS "Build tests" OFF)

# Include main dependency configuration
include(cmake/Dependencies.cmake)

# ImGui backend sources - check both new and legacy paths
if(EXISTS "${IMGUI_DIR}/backends/imgui_impl_glfw.cpp")
    set(IMGUI_BACKENDS
        "${IMGUI_DIR}/backends/imgui_impl_glfw.cpp"
        "${IMGUI_DIR}/backends/imgui_impl_opengl3.cpp"
    )
elseif(EXISTS "${IMGUI_DIR}/include/backends/imgui_impl_glfw.cpp")
    set(IMGUI_BACKENDS
        "${IMGUI_DIR}/include/backends/imgui_impl_glfw.cpp"
        "${IMGUI_DIR}/include/backends/imgui_impl_opengl3.cpp"
    )
else()
    message(WARNING "ImGui backends not found, using header-only mode")
    set(IMGUI_BACKENDS "")
endif()

# Main executable
add_executable(caliper

    main.cpp
    ${IMGUI_BACKENDS}
)

# Link all dependencies
if(PYTORCH_BUILDING_FROM_SOURCE)
    # Use interface library that depends on PyTorch build
    target_link_libraries(caliper PRIVATE
        ${TORCH_INTERFACE_LIB}
        glfw
        glm::glm
        imgui
        implot
        OpenGL::GL
        ${PLATFORM_LIBS}
    )
else()
    # Use pre-built PyTorch
    target_link_libraries(caliper PRIVATE
        ${TORCH_LIBRARIES}
        glfw
        glm::glm
        imgui
        implot
        OpenGL::GL
        ${PLATFORM_LIBS}
    )
    set_target_properties(caliper PROPERTIES
        CXX_STANDARD 17
        CXX_STANDARD_REQUIRED ON
    )
endif()

# Set RPATH for dynamic libraries (macOS)
set_target_properties(caliper PROPERTIES
    INSTALL_RPATH "@executable_path"
)

# Copy dynamic libraries to build directory if needed
if(APPLE)
    target_link_libraries(caliper PRIVATE
        "-framework Cocoa"
        "-framework IOKit"
        "-framework CoreVideo"
    )
endif()
