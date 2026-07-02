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

if(USE_CUDA)
    find_package(CUDAToolkit QUIET)
    if(CUDAToolkit_FOUND)
        set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -allow-unsupported-compiler")

        if(NOT TARGET CUDA::nvToolsExt)
            add_library(CUDA::nvToolsExt INTERFACE IMPORTED)
        endif()

        message(STATUS "CUDA support: ENABLED (CUDA ${CUDAToolkit_VERSION})")
    else()
        message(WARNING "USE_CUDA is ON but no CUDA toolkit found — falling back to CPU")
        set(USE_CUDA OFF)
    endif()
endif()
if(NOT USE_CUDA)
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
# Caliper SDK (PLATFORM.md §17 Phase 0: installable package, in-tree source)
# ============================================================================

set(CALIPER_SDK_VERSION 0.1.0)

add_library(caliper_sdk INTERFACE)
add_library(caliper::sdk ALIAS caliper_sdk)
target_include_directories(caliper_sdk INTERFACE
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/sdk/include>
    $<INSTALL_INTERFACE:include>
)

# UI stack: pinned imgui/implot/implot3d/ImGuiFileDialog from the monorepo.
# Moves into the SDK repo at Phase 3; a separate target so the seam is visible.
add_library(caliper_ui_stack INTERFACE)
add_library(caliper::ui_stack ALIAS caliper_ui_stack)
target_include_directories(caliper_ui_stack INTERFACE
    ${THIRD_PARTY_DIR}/imgui
    ${THIRD_PARTY_DIR}/imgui/backends
    ${THIRD_PARTY_DIR}/implot
    ${THIRD_PARTY_DIR}/implot3d
    ${THIRD_PARTY_DIR}/ImGuiFileDialog
)
target_link_libraries(caliper_ui_stack INTERFACE
    imgui implot implot3d ImGuiFileDialog
)
target_compile_definitions(caliper_ui_stack INTERFACE
    IMGUI_IMPL_OPENGL_LOADER_GLEW
)

# Host-owned app-paths utility. Applets that still need the shared data dir
# link this TARGET instead of compiling src/app_paths.cpp by path.
# Dies when applets move to CaliperHost.applet_data_dir (Phase 1+).
add_library(caliper_app_paths STATIC src/app_paths.cpp)
target_include_directories(caliper_app_paths PUBLIC ${CMAKE_SOURCE_DIR}/src)
set_target_properties(caliper_app_paths PROPERTIES POSITION_INDEPENDENT_CODE ON)

# ---- SDK install rules (component: sdk) ----
include(CMakePackageConfigHelpers)
install(DIRECTORY sdk/include/ DESTINATION include COMPONENT sdk)
set_target_properties(caliper_sdk PROPERTIES EXPORT_NAME sdk)
install(TARGETS caliper_sdk EXPORT CaliperSDKTargets COMPONENT sdk)
install(EXPORT CaliperSDKTargets NAMESPACE caliper::
        DESTINATION lib/cmake/caliper-sdk COMPONENT sdk)
write_basic_package_version_file(
    "${CMAKE_BINARY_DIR}/caliper-sdk-config-version.cmake"
    VERSION ${CALIPER_SDK_VERSION} COMPATIBILITY SameMajorVersion)
install(FILES
    "${CMAKE_SOURCE_DIR}/sdk/cmake/caliper-sdk-config.cmake"
    "${CMAKE_BINARY_DIR}/caliper-sdk-config-version.cmake"
    DESTINATION lib/cmake/caliper-sdk COMPONENT sdk)

# Host library: loader/negotiation logic, unit-testable without OpenGL.
include(FetchContent)
FetchContent_Declare(tomlplusplus
    GIT_REPOSITORY https://github.com/marzer/tomlplusplus.git
    GIT_TAG v3.4.0 GIT_SHALLOW TRUE)
FetchContent_MakeAvailable(tomlplusplus)

add_library(caliper_host_lib STATIC
    src/host/applet_manifest.cpp
    src/host/negotiation.cpp
)
target_include_directories(caliper_host_lib PUBLIC ${CMAKE_SOURCE_DIR}/src/host)
target_link_libraries(caliper_host_lib
    PUBLIC caliper::sdk
    PRIVATE tomlplusplus::tomlplusplus)
set_target_properties(caliper_host_lib PROPERTIES
    CXX_STANDARD 20 CXX_STANDARD_REQUIRED ON POSITION_INDEPENDENT_CODE ON)

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

file(GLOB applet_dirs LIST_DIRECTORIES true CONFIGURE_DEPENDS applets/*)

set(_active_applet_libs "")
foreach(dir ${applet_dirs})
    if(IS_DIRECTORY "${dir}" AND EXISTS "${dir}/CMakeLists.txt")
        get_filename_component(applet_name "${dir}" NAME)
        message(STATUS "  Found applet: ${applet_name}")
        add_subdirectory("${dir}")
        add_dependencies(caliper "${applet_name}")
        list(APPEND _active_applet_libs
            "${CMAKE_BINARY_DIR}/applets/${CMAKE_SHARED_LIBRARY_PREFIX}${applet_name}${CMAKE_SHARED_LIBRARY_SUFFIX}")
    endif()
endforeach()

# Remove stale applet libraries left from previously deleted applets
file(GLOB _existing_applet_libs "${CMAKE_BINARY_DIR}/applets/${CMAKE_SHARED_LIBRARY_PREFIX}*${CMAKE_SHARED_LIBRARY_SUFFIX}")
foreach(lib ${_existing_applet_libs})
    if(NOT lib IN_LIST _active_applet_libs)
        get_filename_component(_stale_name "${lib}" NAME)
        message(STATUS "  Removing stale applet: ${_stale_name}")
        file(REMOVE "${lib}")
    endif()
endforeach()

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

if(BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()
