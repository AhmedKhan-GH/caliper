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
    # macOS/Linux: Build from source via our out-of-tree wrapper so the
    # submodule stays pristine (no CMakeLists gets copied inside it).
    set(GLEW_SRC_DIR "${THIRD_PARTY_DIR}/glew")
    add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/cmake/wrappers/glew EXCLUDE_FROM_ALL)
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
set(IMGUI_SRC_DIR "${THIRD_PARTY_DIR}/imgui")
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/cmake/wrappers/imgui EXCLUDE_FROM_ALL)
list(APPEND CALIPER_DEPENDENCY_LIBS imgui)
message(STATUS "    ✓ ImGui configured")

# --- ImPlot (Plotting library) ---
message(STATUS "  Configuring ImPlot...")
set(IMPLOT_SRC_DIR "${THIRD_PARTY_DIR}/implot")
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/cmake/wrappers/implot EXCLUDE_FROM_ALL)
list(APPEND CALIPER_DEPENDENCY_LIBS implot)
message(STATUS "    ✓ ImPlot configured")

# --- ImPlot3D (3D Plotting library) ---
message(STATUS "  Configuring ImPlot3D...")
set(IMPLOT3D_SRC_DIR "${THIRD_PARTY_DIR}/implot3d")
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/cmake/wrappers/implot3d EXCLUDE_FROM_ALL)
list(APPEND CALIPER_DEPENDENCY_LIBS implot3d)
message(STATUS "    ✓ ImPlot3D configured")

# --- imgui-node-editor (Blueprints-style node graph editor) ---
message(STATUS "  Configuring imgui-node-editor...")
set(IMGUI_NODE_EDITOR_SRC_DIR "${THIRD_PARTY_DIR}/imgui-node-editor")
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/cmake/wrappers/imgui-node-editor EXCLUDE_FROM_ALL)
list(APPEND CALIPER_DEPENDENCY_LIBS imgui-node-editor)
message(STATUS "    ✓ imgui-node-editor configured")

# --- DuckDB (Embedded analytical database) ---
# Heavy dep. We disable the shell, unit tests, benchmarks, and python module so
# only the core static library gets built. First configure takes a few minutes;
# subsequent incremental builds are fine.
message(STATUS "  Configuring DuckDB...")
set(BUILD_UNITTESTS  OFF CACHE BOOL "" FORCE)
set(BUILD_SHELL      OFF CACHE BOOL "" FORCE)
set(BUILD_BENCHMARKS OFF CACHE BOOL "" FORCE)
set(BUILD_PYTHON     OFF CACHE BOOL "" FORCE)
# DuckDB enables sanitizers by default in debug builds, which requires the
# UBSan/ASan runtime at final link. We don't want that for our shipping binary.
set(ENABLE_SANITIZER         OFF CACHE BOOL "" FORCE)
set(ENABLE_UBSAN             OFF CACHE BOOL "" FORCE)
set(ENABLE_THREAD_SANITIZER  OFF CACHE BOOL "" FORCE)
set(DUCKDB_EXPLICIT_VERSION "v1.5.2" CACHE STRING "" FORCE)
# Set explicit platform to avoid running duckdb_platform_binary which can create NUL files on Windows
if(WIN32)
    if(CMAKE_SIZEOF_VOID_P EQUAL 8)
        set(DUCKDB_EXPLICIT_PLATFORM "windows_amd64" CACHE STRING "" FORCE)
    else()
        set(DUCKDB_EXPLICIT_PLATFORM "windows_x86" CACHE STRING "" FORCE)
    endif()
endif()
add_subdirectory(${THIRD_PARTY_DIR}/duckdb EXCLUDE_FROM_ALL)
# DuckDB's own CMakeLists hard-sets CMAKE_MSVC_RUNTIME_LIBRARY to the STATIC
# CRT for its whole subtree, which conflicts with our /MD policy (PLATFORM.md
# D7: host + applet DLLs must share one heap) and fails the final link with
# LNK2005 (msvcprtd vs duckdb_static). Re-apply our runtime choice to every
# target DuckDB created, recursively.
if(WIN32)
    function(caliper_force_msvc_runtime dir)
        get_property(_tgts DIRECTORY "${dir}" PROPERTY BUILDSYSTEM_TARGETS)
        foreach(_t ${_tgts})
            get_target_property(_type ${_t} TYPE)
            if(NOT _type STREQUAL "INTERFACE_LIBRARY")
                set_property(TARGET ${_t} PROPERTY
                    MSVC_RUNTIME_LIBRARY "${CMAKE_MSVC_RUNTIME_LIBRARY}")
            endif()
        endforeach()
        get_property(_subs DIRECTORY "${dir}" PROPERTY SUBDIRECTORIES)
        foreach(_s ${_subs})
            caliper_force_msvc_runtime("${_s}")
        endforeach()
    endfunction()
    caliper_force_msvc_runtime("${THIRD_PARTY_DIR}/duckdb")
endif()
# DuckDB doesn't set target_include_directories on duckdb_static — it relies
# on global include_directories() inside its own subdirectory. Re-attach the
# include path so our targets can `#include <duckdb.hpp>`.
target_include_directories(duckdb_static INTERFACE
    $<BUILD_INTERFACE:${THIRD_PARTY_DIR}/duckdb/src/include>
)
# Define DUCKDB_BUILD_LIBRARY to tell DuckDB headers we're linking statically
# This prevents __declspec(dllimport) decorations on Windows
target_compile_definitions(duckdb_static INTERFACE DUCKDB_BUILD_LIBRARY)
# DuckDB's extension/ subdirectory tries to append its loader objects to
# ALL_OBJECT_FILES *after* src/ has already created duckdb_static, so the
# extension-loader symbols (ExtensionHelper::LoadAllExtensions etc.) end up
# in a sibling target instead of inside duckdb_static. Link that sibling
# explicitly here.
list(APPEND CALIPER_DEPENDENCY_LIBS
    duckdb_static
    duckdb_generated_extension_loader
    parquet_extension
    core_functions_extension
)
message(STATUS "    ✓ DuckDB configured (target: duckdb_static)")

# --- slang (SystemVerilog compiler library) ---
message(STATUS "  Configuring slang...")
set(SLANG_INCLUDE_TOOLS OFF CACHE BOOL "" FORCE)
set(SLANG_INCLUDE_TESTS OFF CACHE BOOL "" FORCE)
set(SLANG_INCLUDE_DOCS OFF CACHE BOOL "" FORCE)
set(SLANG_INCLUDE_PYLIB OFF CACHE BOOL "" FORCE)
set(SLANG_INCLUDE_INSTALL OFF CACHE BOOL "" FORCE)
add_subdirectory(${THIRD_PARTY_DIR}/slang EXCLUDE_FROM_ALL)
message(STATUS "    ✓ slang configured (target: slang::slang)")

# --- ImGuiColorTextEdit (Syntax-highlighted text editor widget) ---
message(STATUS "  Configuring ImGuiColorTextEdit...")
set(IMGUI_COLOR_TEXT_EDIT_SRC_DIR "${THIRD_PARTY_DIR}/ImGuiColorTextEdit")
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/cmake/wrappers/imgui-color-text-edit EXCLUDE_FROM_ALL)
list(APPEND CALIPER_DEPENDENCY_LIBS imgui-color-text-edit)
message(STATUS "    ✓ ImGuiColorTextEdit configured")

# --- ImTerm (Header-only ImGui terminal/console widget) ---
message(STATUS "  Configuring ImTerm...")
set(IMTERM_SRC_DIR "${THIRD_PARTY_DIR}/ImTerm")
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/cmake/wrappers/imterm EXCLUDE_FROM_ALL)
list(APPEND CALIPER_DEPENDENCY_LIBS imterm)
message(STATUS "    ✓ ImTerm configured")

# --- llama.cpp (LLM inference) ---
message(STATUS "  Configuring llama.cpp...")
set(LLAMA_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_SERVER OFF CACHE BOOL "" FORCE)
set(BUILD_SHARED_LIBS_SAVED ${BUILD_SHARED_LIBS})
set(BUILD_SHARED_LIBS OFF)
if(APPLE)
    set(GGML_METAL ON CACHE BOOL "" FORCE)
    set(GGML_METAL_EMBED_LIBRARY ON CACHE BOOL "" FORCE)
    message(STATUS "    Metal backend enabled for llama.cpp")
elseif(USE_CUDA AND NOT WIN32)
    set(GGML_CUDA ON CACHE BOOL "" FORCE)
    message(STATUS "    CUDA backend enabled for llama.cpp")
elseif(WIN32 AND USE_CUDA AND CUDAToolkit_VERSION VERSION_GREATER_EQUAL 13)
    # Windows needs CUDA 13+: a 12.x ggml build would load a cudart64_12.dll /
    # cublas64_12.dll that collide with the copies libtorch cu12x ships, and
    # CUDA 12.x's host_config.h hard-rejects MSVC >= 19.50 (VS2026). CUDA 13's
    # runtime DLLs carry different names (cudart64_13.dll), so both runtimes
    # coexist in one process, and nvcc 13's MSVC gate is lifted by the
    # -allow-unsupported-compiler flag set at the top level.
    # Pin ggml's nvcc to the same toolkit find_package(CUDAToolkit) resolved,
    # so the version gate above and the compiler actually agree.
    set(CMAKE_CUDA_COMPILER "${CUDAToolkit_BIN_DIR}/nvcc.exe")
    # Both ggml and torch's Caffe2 config enable the CUDA language and need
    # CMAKE_CUDA_ARCHITECTURES defined at this (root) scope — Caffe2's
    # try_compile hard-fails on an empty value. "native" targets the local
    # GPU (requires one at configure time, which USE_CUDA already implies).
    if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
        set(CMAKE_CUDA_ARCHITECTURES native)
    endif()
    set(GGML_CUDA ON CACHE BOOL "" FORCE)
    message(STATUS "    CUDA backend enabled for llama.cpp (CUDA ${CUDAToolkit_VERSION})")
elseif(WIN32)
    set(GGML_CUDA OFF CACHE BOOL "" FORCE)
    message(STATUS "    llama.cpp: CPU backend on Windows (CUDA 13+ toolkit not found)")
endif()

# Caliper's local llama.cpp deltas live as patch files in-tree (NOT a fork):
# the submodule pin stays on fetchable upstream, and these apply at configure
# time. They adapt the loader to ollama's GGUF encoding (ssm_dt tensor name,
# 3-section rope) and add GPT-OSS arch support — see
# third_party/patches/llama.cpp/README.md. The apply is idempotent AND atomic:
# the whole set is reverse-checked first (already-applied dev checkout or a
# re-configure -> no-op), else forward-checked and applied together (both
# patches touch llama-model.cpp, so they must land as one transaction).
file(GLOB _llama_patches "${CMAKE_CURRENT_SOURCE_DIR}/third_party/patches/llama.cpp/*.patch")
list(SORT _llama_patches)
if(_llama_patches)
    set(_llama_patches_rev ${_llama_patches})
    list(REVERSE _llama_patches_rev)
    execute_process(
        COMMAND git -C ${THIRD_PARTY_DIR}/llama.cpp apply --reverse --check ${_llama_patches_rev}
        RESULT_VARIABLE _llama_patched OUTPUT_QUIET ERROR_QUIET)
    if(_llama_patched EQUAL 0)
        message(STATUS "    llama.cpp patches already applied — skipping")
    else()
        execute_process(
            COMMAND git -C ${THIRD_PARTY_DIR}/llama.cpp apply --check ${_llama_patches}
            RESULT_VARIABLE _llama_clean OUTPUT_QUIET ERROR_QUIET)
        if(NOT _llama_clean EQUAL 0)
            message(FATAL_ERROR
                "  llama.cpp patches neither apply cleanly nor are already applied.\n"
                "  The submodule is likely not at the expected upstream base "
                "(a4107133).\n"
                "  Reset it (git -C third_party/llama.cpp checkout a4107133) or "
                "regenerate the patches — see third_party/patches/llama.cpp/README.md.")
        endif()
        execute_process(
            COMMAND git -C ${THIRD_PARTY_DIR}/llama.cpp apply ${_llama_patches}
            RESULT_VARIABLE _llama_apply)
        if(NOT _llama_apply EQUAL 0)
            message(FATAL_ERROR "  Failed to apply llama.cpp patches.")
        endif()
        list(LENGTH _llama_patches _llama_npatch)
        message(STATUS "    applied ${_llama_npatch} local llama.cpp patch(es)")
    endif()
endif()

add_subdirectory(${THIRD_PARTY_DIR}/llama.cpp EXCLUDE_FROM_ALL)
set(BUILD_SHARED_LIBS ${BUILD_SHARED_LIBS_SAVED})
message(STATUS "    ✓ llama.cpp configured")

# --- ImGuiFileDialog (File open/save dialog for ImGui) ---
message(STATUS "  Configuring ImGuiFileDialog...")
# Upstream ships a CMakeLists.txt that tries find_package(imgui) QUIET and skips
# linking when not found. We use our own imgui submodule target instead.
set(IGFD_INSTALL OFF CACHE BOOL "" FORCE)
add_subdirectory(${THIRD_PARTY_DIR}/ImGuiFileDialog EXCLUDE_FROM_ALL)
target_link_libraries(ImGuiFileDialog PUBLIC imgui)
list(APPEND CALIPER_DEPENDENCY_LIBS ImGuiFileDialog)
message(STATUS "    ✓ ImGuiFileDialog configured")

# --- CURL + ZLIB (Windows only) ---
# On macOS these are system libraries; the applets find them with plain
# find_package. Windows ships neither, so build them in-tree and let the
# applets' `if(NOT TARGET ...)` guards pick these targets up instead.
# curl uses Schannel (Windows-native TLS) so no OpenSSL is involved.
if(WIN32)
    message(STATUS "  Configuring ZLIB + CURL (Windows in-tree)...")
    include(FetchContent)

    # zlib's ancient cmake_minimum_required needs this shim under CMake 4.x
    # (same trick tests/ uses for doctest).
    set(CMAKE_POLICY_VERSION_MINIMUM 3.5)

    set(ZLIB_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
    FetchContent_Declare(zlib
        URL https://github.com/madler/zlib/releases/download/v1.3.1/zlib-1.3.1.tar.gz
        DOWNLOAD_EXTRACT_TIMESTAMP TRUE)
    FetchContent_MakeAvailable(zlib)
    # zlib's CMake predates usage requirements: attach include dirs (zconf.h is
    # generated into the build dir) and provide the standard imported name.
    target_include_directories(zlibstatic INTERFACE
        $<BUILD_INTERFACE:${zlib_SOURCE_DIR}>
        $<BUILD_INTERFACE:${zlib_BINARY_DIR}>)
    add_library(ZLIB::ZLIB ALIAS zlibstatic)

    set(BUILD_CURL_EXE OFF CACHE BOOL "" FORCE)
    set(BUILD_STATIC_LIBS ON CACHE BOOL "" FORCE)
    set(BUILD_LIBCURL_DOCS OFF CACHE BOOL "" FORCE)
    set(BUILD_MISC_DOCS OFF CACHE BOOL "" FORCE)
    set(ENABLE_CURL_MANUAL OFF CACHE BOOL "" FORCE)
    set(HTTP_ONLY ON CACHE BOOL "" FORCE)
    set(CURL_USE_SCHANNEL ON CACHE BOOL "" FORCE)
    set(CURL_USE_LIBPSL OFF CACHE BOOL "" FORCE)
    set(CURL_USE_LIBSSH2 OFF CACHE BOOL "" FORCE)
    set(CURL_ZLIB OFF CACHE STRING "" FORCE)
    set(CURL_BROTLI OFF CACHE BOOL "" FORCE)
    set(CURL_ZSTD OFF CACHE BOOL "" FORCE)
    set(USE_NGHTTP2 OFF CACHE BOOL "" FORCE)
    set(USE_LIBIDN2 OFF CACHE BOOL "" FORCE)
    set(CURL_DISABLE_INSTALL ON CACHE BOOL "" FORCE)
    FetchContent_Declare(curl
        URL https://github.com/curl/curl/releases/download/curl-8_11_1/curl-8.11.1.tar.gz
        DOWNLOAD_EXTRACT_TIMESTAMP TRUE)
    FetchContent_MakeAvailable(curl)

    unset(CMAKE_POLICY_VERSION_MINIMUM)
    message(STATUS "    ✓ ZLIB + CURL configured (static, Schannel TLS)")

    # --- Vulkan backend toolchain (Windows, Phase 4) ---
    # No Vulkan SDK required on the machine: headers + the volk loader come in
    # as FetchContent (the runtime vulkan-1.dll ships with the GPU driver),
    # and glslang builds from source to compile colormap.comp to a SPIR-V
    # header at build time. One consistent vulkan-sdk tag for all three.
    message(STATUS "  Configuring Vulkan headers + volk + glslang...")
    set(_VK_TAG "vulkan-sdk-1.3.290.0")
    FetchContent_Declare(vulkan-headers
        URL https://github.com/KhronosGroup/Vulkan-Headers/archive/refs/tags/${_VK_TAG}.tar.gz
        DOWNLOAD_EXTRACT_TIMESTAMP TRUE)
    set(VOLK_STATIC_DEFINES VK_USE_PLATFORM_WIN32_KHR)   # win32 entry points
    FetchContent_Declare(volk
        URL https://github.com/zeux/volk/archive/refs/tags/${_VK_TAG}.tar.gz
        DOWNLOAD_EXTRACT_TIMESTAMP TRUE)
    set(ENABLE_OPT OFF CACHE BOOL "" FORCE)              # no SPIRV-Tools dep
    set(ENABLE_HLSL OFF CACHE BOOL "" FORCE)
    set(ENABLE_SPVREMAPPER OFF CACHE BOOL "" FORCE)
    set(ENABLE_GLSLANG_BINARIES ON CACHE BOOL "" FORCE)  # the compiler exe
    set(GLSLANG_TESTS OFF CACHE BOOL "" FORCE)
    set(GLSLANG_ENABLE_INSTALL OFF CACHE BOOL "" FORCE)
    FetchContent_Declare(glslang
        URL https://github.com/KhronosGroup/glslang/archive/refs/tags/${_VK_TAG}.tar.gz
        DOWNLOAD_EXTRACT_TIMESTAMP TRUE)
    # volk at this tag locates headers via VULKAN_HEADERS_INSTALL_DIR (a path
    # it appends /include to), not the Vulkan::Headers target — populate the
    # headers first and point volk at their source dir.
    FetchContent_MakeAvailable(vulkan-headers)
    set(VULKAN_HEADERS_INSTALL_DIR "${vulkan-headers_SOURCE_DIR}" CACHE PATH "" FORCE)
    FetchContent_MakeAvailable(volk glslang)
    message(STATUS "    ✓ Vulkan toolchain configured (headers + volk + glslang)")
endif()

# ============================================================================
# Category 4: PyTorch (Large dependency)
# ============================================================================

message(STATUS "Configuring PyTorch (libtorch)...")

# Include ExternalProject module
include(ExternalProject)

# Set PyTorch install directory
set(PYTORCH_INSTALL_DIR "${CMAKE_CURRENT_BINARY_DIR}/pytorch_install")

# ── Auto-select CUDA variant from detected toolkit version ──
# PyTorch publishes pre-built binaries for a fixed set of CUDA versions.
# Pick the highest variant whose CUDA version <= the installed toolkit.
# CUDA is forward-compatible within a major version, so cu121 works on 12.8.
if(USE_CUDA)
    set(_CUDA_VER "${CUDAToolkit_VERSION_MAJOR}.${CUDAToolkit_VERSION_MINOR}")

    # Available variants for PyTorch 2.5.1 (highest first). Plain if/elseif —
    # the previous pair-list foreach flattened its "ver;tag" entries into
    # single tokens, so list(GET _entry 1 ...) errored whenever CUDA was found.
    if(NOT _CUDA_VER VERSION_LESS 12.4)
        set(PYTORCH_VARIANT "cu124")
    elseif(NOT _CUDA_VER VERSION_LESS 12.1)
        set(PYTORCH_VARIANT "cu121")
    elseif(NOT _CUDA_VER VERSION_LESS 11.8)
        set(PYTORCH_VARIANT "cu118")
    else()
        set(PYTORCH_VARIANT "")
    endif()

    if(PYTORCH_VARIANT STREQUAL "")
        message(WARNING "CUDA ${_CUDA_VER} is too old for any PyTorch 2.5.1 CUDA build — falling back to CPU")
        set(PYTORCH_VARIANT "cpu")
        set(USE_CUDA OFF)
    endif()
else()
    set(PYTORCH_VARIANT "cpu")
endif()

# ============================================================================
# Windows: Download pre-built libtorch (CUDA or CPU)
# macOS/Linux: Build from source (for MPS support on macOS)
# ============================================================================

if(WIN32)
    message(STATUS "  Using pre-built PyTorch 2.5.1 for Windows...")

    set(PYTORCH_VERSION "2.5.1")

    if(USE_CUDA)
        set(PYTORCH_VARIANT_NAME "CUDA (${PYTORCH_VARIANT})")
    else()
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

    # Debug and Release libtorch are DIFFERENT zips (debug torch is 10-50x
    # slower at runtime and uses the debug CRT). The download is guarded by
    # directory existence, so the two variants must not share a directory —
    # a stale variant would silently keep linking into the wrong config.
    # Variant-specific dirs let cmake-build-debug and cmake-build-release
    # coexist, each against its matching libtorch.
    string(TOLOWER "${PYTORCH_BUILD_TYPE}" _TORCH_FLAVOR)
    set(LIBTORCH_DIR "${THIRD_PARTY_DIR}/libtorch-${_TORCH_FLAVOR}")

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
        # The zip's root folder is always "libtorch" — move it to the
        # variant-specific location.
        file(RENAME "${THIRD_PARTY_DIR}/libtorch" "${LIBTORCH_DIR}")

        # Clean up zip file
        file(REMOVE "${CMAKE_BINARY_DIR}/libtorch.zip")
        message(STATUS "  ✓ PyTorch extracted successfully")
    else()
        message(STATUS "  ✓ PyTorch already downloaded (${PYTORCH_BUILD_TYPE}, ${PYTORCH_VARIANT_NAME})")
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

    # torch headers pull in windows.h; without NOMINMAX its min/max macros
    # break tensor.max() / std::max in every consumer. Attach to the imported
    # target so all torch-linking applets inherit it.
    set_property(TARGET torch APPEND PROPERTY INTERFACE_COMPILE_DEFINITIONS NOMINMAX)

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
            set(PYTORCH_URL "https://download.pytorch.org/libtorch/${PYTORCH_VARIANT}/libtorch-cxx11-abi-shared-with-deps-${PYTORCH_VERSION}%2B${PYTORCH_VARIANT}.zip")
            set(PYTORCH_PLATFORM "Linux CUDA (${PYTORCH_VARIANT})")
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
