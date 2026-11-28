# ============================================================================
# Caliper Dependencies Configuration (PyTorch-style)
# ============================================================================
#
# This file orchestrates all third-party dependencies following PyTorch's
# architecture:
#   0. FetchContent declarations (single source of truth for the whole repo)
#   1. System libraries (find on system)
#   2. Header-only libraries (include paths only)
#   3. CMake-based libraries (add_subdirectory)
#   4. Complex libraries (ExternalProject_Add for PyTorch)
#
# Standalone sub-builds (src/builder, python/netio, web/wasm) and
# cmake/SternbildTokens.cmake include this file with
# CALIPER_DEPENDENCIES_DECLARATIONS_ONLY=ON and call the caliper_fetch_*()
# functions they need.
#
# ============================================================================

# Single-source-of-truth FetchContent pins.
set(CALIPER_TOMLPLUSPLUS_TAG v3.4.0)
set(CALIPER_NLOHMANN_JSON_VERSION 3.11.3)
set(CALIPER_NLOHMANN_JSON_SHA256 d6c65aca6b1ed68e7a182f4757257b107ae403032760ed6ef121c9d55e81757d)
set(CALIPER_DOCTEST_TAG v2.4.11)
set(STERNBILD_TOKENS_REPOSITORY https://github.com/AhmedKhan-GH/organization.git)
set(STERNBILD_TOKENS_TAG design-v2.0.0)
set(STERNBILD_TOKENS_COMMIT f88dc88a2223cdeb029625639d23be1ab5755f95)

function(caliper_fetch_tomlplusplus)
    if(TARGET tomlplusplus::tomlplusplus)
        return()
    endif()
    include(FetchContent)
    FetchContent_Declare(tomlplusplus
        GIT_REPOSITORY https://github.com/marzer/tomlplusplus.git
        GIT_TAG ${CALIPER_TOMLPLUSPLUS_TAG}
        GIT_SHALLOW TRUE)
    FetchContent_MakeAvailable(tomlplusplus)
endfunction()

function(caliper_fetch_nlohmann_json)
    if(TARGET nlohmann_json::nlohmann_json)
        return()
    endif()
    include(FetchContent)
    set(JSON_BuildTests OFF CACHE INTERNAL "")
    FetchContent_Declare(nlohmann_json
        URL https://github.com/nlohmann/json/releases/download/v${CALIPER_NLOHMANN_JSON_VERSION}/json.tar.xz
        URL_HASH SHA256=${CALIPER_NLOHMANN_JSON_SHA256})
    FetchContent_MakeAvailable(nlohmann_json)
endfunction()

function(caliper_fetch_doctest)
    include(FetchContent)
    if(NOT TARGET doctest::doctest)
        # doctest v2.4.11 declares cmake_minimum_required 2.8, which CMake >= 4.0 rejects.
        set(CMAKE_POLICY_VERSION_MINIMUM 3.5)
        FetchContent_Declare(doctest
            GIT_REPOSITORY https://github.com/doctest/doctest.git
            GIT_TAG ${CALIPER_DOCTEST_TAG}
            GIT_SHALLOW TRUE)
        FetchContent_MakeAvailable(doctest)
    endif()
    FetchContent_GetProperties(doctest SOURCE_DIR _doctest_src BINARY_DIR _doctest_bin)
    if(_doctest_src)
        set(doctest_SOURCE_DIR "${_doctest_src}" PARENT_SCOPE)
    endif()
    if(_doctest_bin)
        set(doctest_BINARY_DIR "${_doctest_bin}" PARENT_SCOPE)
    endif()
endfunction()

# When the download cannot happen, FetchContent stops in a sub-build with git's
# retries and a call stack that do not name the design tokens. Thus, before a
# download, the repository is asked for the tag, and a failure stops the
# configure with a message that says what is missing and how to go on. The same
# answer gives the tag's commit, and a commit other than the pinned one also
# stops the configure. The check runs only when the build folder does not hold
# the pinned commit yet.
function(_caliper_sternbild_tokens_check_reachable)
    if(FETCHCONTENT_SOURCE_DIR_STERNBILD_TOKENS OR FETCHCONTENT_FULLY_DISCONNECTED)
        return()
    endif()
    find_package(Git QUIET)
    set(_what "The app's theme reads the Sternbild design tokens, release ${STERNBILD_TOKENS_TAG} (commit ${STERNBILD_TOKENS_COMMIT}) of ${STERNBILD_TOKENS_REPOSITORY}, pinned in cmake/Dependencies.cmake and cmake/SternbildTokens.cmake")
    set(_way_out "or configure with -DFETCHCONTENT_SOURCE_DIR_STERNBILD_TOKENS=<path> set to a local checkout of that repository at commit ${STERNBILD_TOKENS_COMMIT}")
    if(NOT GIT_FOUND)
        message(FATAL_ERROR "${_what}. The first configure of a build folder downloads them with git, and git was not found. Install git and configure again, ${_way_out}.")
    endif()
    set(_base "${FETCHCONTENT_BASE_DIR}")
    if(NOT _base)
        set(_base "${CMAKE_BINARY_DIR}/_deps")
    endif()
    set(_src "${_base}/sternbild_tokens-src")
    if(EXISTS "${_src}/.git")
        execute_process(COMMAND "${GIT_EXECUTABLE}" -C "${_src}" rev-parse -q --verify HEAD
                        OUTPUT_VARIABLE _head OUTPUT_STRIP_TRAILING_WHITESPACE RESULT_VARIABLE _rc_head ERROR_QUIET)
        if(_rc_head EQUAL 0 AND _head STREQUAL STERNBILD_TOKENS_COMMIT)
            return()   # the build folder holds the pinned commit: FetchContent needs no network
        endif()
    endif()
    execute_process(
        COMMAND "${CMAKE_COMMAND}" -E env GIT_TERMINAL_PROMPT=0
                "${GIT_EXECUTABLE}" ls-remote --exit-code --tags "${STERNBILD_TOKENS_REPOSITORY}"
                "refs/tags/${STERNBILD_TOKENS_TAG}" "refs/tags/${STERNBILD_TOKENS_TAG}^{}"
        RESULT_VARIABLE _rc OUTPUT_VARIABLE _out ERROR_VARIABLE _err ERROR_STRIP_TRAILING_WHITESPACE
        TIMEOUT 120)
    if(_rc EQUAL 2)
        message(FATAL_ERROR "${_what}. The repository has no tag ${STERNBILD_TOKENS_TAG}. Correct STERNBILD_TOKENS_TAG in cmake/Dependencies.cmake and cmake/SternbildTokens.cmake.")
    elseif(NOT _rc EQUAL 0)
        message(FATAL_ERROR "${_what}. The first configure of a build folder downloads them, and git could not reach the repository:\n  ${_err}\nConnect to the network and configure again, ${_way_out}.")
    endif()
    # The tag's commit: the peeled line of an annotated tag, else the tag's own line.
    set(_remote "")
    string(REGEX MATCH "([0-9a-f]+)[ \t]+refs/tags/${STERNBILD_TOKENS_TAG}\\^{}" _peeled "${_out}")
    if(_peeled)
        set(_remote "${CMAKE_MATCH_1}")
    elseif(_out MATCHES "([0-9a-f]+)[ \t]+refs/tags/${STERNBILD_TOKENS_TAG}")
        set(_remote "${CMAKE_MATCH_1}")
    endif()
    if(NOT _remote STREQUAL STERNBILD_TOKENS_COMMIT)
        message(FATAL_ERROR "${_what}. The tag ${STERNBILD_TOKENS_TAG} on the repository points to commit ${_remote}, which is not the pinned commit ${STERNBILD_TOKENS_COMMIT}. A tag that moved can carry other code. Check the release, then set STERNBILD_TOKENS_TAG and STERNBILD_TOKENS_COMMIT together: a bump changes both lines.")
    endif()
endfunction()

function(caliper_fetch_sternbild_tokens)
    if(TARGET sternbild::tokens)
        return()
    endif()
    include(FetchContent)
    # By the commit, thus without GIT_SHALLOW, which CMake allows only for a branch or a tag.
    FetchContent_Declare(sternbild_tokens
        GIT_REPOSITORY ${STERNBILD_TOKENS_REPOSITORY}
        GIT_TAG ${STERNBILD_TOKENS_COMMIT}
        SOURCE_SUBDIR packages/sternbild-tokens-c)
    _caliper_sternbild_tokens_check_reachable()
    FetchContent_MakeAvailable(sternbild_tokens)
endfunction()

if(CALIPER_DEPENDENCIES_DECLARATIONS_ONLY)
    return()
endif()

message(STATUS "Configuring Caliper dependencies...")

# Two link lists come out of this file:
#   CALIPER_DEPENDENCY_LIBS — everything the caliper exe / gfx + export tests
#                             link (UI chrome widgets, DuckDB, ...).
#   CALIPER_CORE_LIBS       — ONLY what libcaliper's own sources + the Metal and
#                             Vulkan backends need (the Apple frameworks, GLFW
#                             and ImGui's GLFW backend). libcaliper
#                             PRIVATE-links this, so an
#                             embedder that links just libcaliper (embed_host,
#                             the Python extension) never pulls in the chrome
#                             widgets or slang. DuckDB reaches
#                             it through caliper_host_lib; imgui/implot/implot3d/
#                             ImGuiFileDialog through caliper::ui_stack.
set(CALIPER_CORE_LIBS "")

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

# Platform-specific frameworks (macOS)
if(APPLE)
    message(STATUS "  ✓ Adding macOS frameworks")
    set(_caliper_apple_frameworks
        "-framework Cocoa"
        "-framework IOKit"
        "-framework CoreVideo"
    )
    list(APPEND CALIPER_DEPENDENCY_LIBS ${_caliper_apple_frameworks})
    list(APPEND CALIPER_CORE_LIBS ${_caliper_apple_frameworks})
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
list(APPEND CALIPER_CORE_LIBS glfw)
message(STATUS "    ✓ GLFW configured")

# --- Sternbild design tokens (theme values and house fonts) ---
message(STATUS "  Configuring Sternbild design tokens...")
caliper_fetch_sternbild_tokens()
message(STATUS "    ✓ Sternbild design tokens configured")

# --- FetchContent libraries (toml++, doctest, and nlohmann/json for builder) ---
message(STATUS "  Configuring toml++ and doctest...")
caliper_fetch_tomlplusplus()
caliper_fetch_doctest()
if(CALIPER_WITH_BUILDER)
    message(STATUS "  Configuring nlohmann/json (CALIPER_WITH_BUILDER=ON)...")
    caliper_fetch_nlohmann_json()
endif()

# --- ImGui (UI library) ---
message(STATUS "  Configuring ImGui...")
set(IMGUI_SRC_DIR "${THIRD_PARTY_DIR}/imgui")
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/cmake/wrappers/imgui EXCLUDE_FROM_ALL)
# imgui is the core library (part of caliper::ui_stack); imgui_backends holds the
# GLFW platform backend, which only the host links. The renderer backends
# (imgui_impl_metal, imgui_impl_vulkan) compile with the Metal and Vulkan backends.
list(APPEND CALIPER_DEPENDENCY_LIBS imgui imgui_backends)
list(APPEND CALIPER_CORE_LIBS imgui_backends)
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
set(DUCKDB_EXPLICIT_VERSION "v1.5.6" CACHE STRING "" FORCE)
# Set explicit platform to avoid running duckdb_platform_binary which can create NUL files on Windows
if(WIN32)
    if(CMAKE_SIZEOF_VOID_P EQUAL 8)
        set(DUCKDB_EXPLICIT_PLATFORM "windows_amd64" CACHE STRING "" FORCE)
    else()
        set(DUCKDB_EXPLICIT_PLATFORM "windows_x86" CACHE STRING "" FORCE)
    endif()
endif()
# DuckDB links its parquet and core_functions extensions statically in every
# build (third_party/duckdb/extension/extension_config.cmake), and links more
# when they are named in BUILD_EXTENSIONS (extension_build_tools.cmake). The
# builder's loaders read JSON and JSONL files with the json extension, so a
# build with the builder names json there.
if(CALIPER_WITH_BUILDER)
    set(BUILD_EXTENSIONS json)
endif()
add_subdirectory(${THIRD_PARTY_DIR}/duckdb EXCLUDE_FROM_ALL)
# DuckDB's own CMakeLists hard-sets CMAKE_MSVC_RUNTIME_LIBRARY to the STATIC
# CRT for its whole subtree, which conflicts with our /MD policy (host +
# applet DLLs must share one heap) and fails the final link with
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
# explicitly here. The generated loader names every statically linked
# extension and loads each one when a database opens, so every target that
# links it links all of them: CALIPER_DUCKDB_LIBS is that one list, used here,
# by caliper_host_lib and by the builder's caliper_builder_duckdb.
set(CALIPER_DUCKDB_LIBS
    duckdb_static
    duckdb_generated_extension_loader
    parquet_extension
    core_functions_extension
)
if(CALIPER_WITH_BUILDER)
    list(APPEND CALIPER_DUCKDB_LIBS json_extension)
endif()
list(APPEND CALIPER_DEPENDENCY_LIBS ${CALIPER_DUCKDB_LIBS})
message(STATUS "    ✓ DuckDB configured (target: duckdb_static)")

# --- slang (SystemVerilog compiler library) ---
# No in-tree target links slang::slang today; its configure is ~half of a fresh
# configure and FetchContent-clones boost_regex + mimalloc. CALIPER_WITH_SLANG=OFF
# skips it entirely (torch-free / Python-extension builds).
if(NOT CALIPER_WITH_SLANG)
    message(STATUS "  Skipping slang: CALIPER_WITH_SLANG=OFF")
else()
message(STATUS "  Configuring slang...")
set(SLANG_INCLUDE_TOOLS OFF CACHE BOOL "" FORCE)
set(SLANG_INCLUDE_TESTS OFF CACHE BOOL "" FORCE)
set(SLANG_INCLUDE_DOCS OFF CACHE BOOL "" FORCE)
set(SLANG_INCLUDE_PYLIB OFF CACHE BOOL "" FORCE)
set(SLANG_INCLUDE_INSTALL OFF CACHE BOOL "" FORCE)
add_subdirectory(${THIRD_PARTY_DIR}/slang EXCLUDE_FROM_ALL)
message(STATUS "    ✓ slang configured (target: slang::slang)")
endif()  # CALIPER_WITH_SLANG

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

# --- CUDA architectures on Windows ---
# torch's Caffe2 config enables the CUDA language on Windows with CUDA 13+ and
# needs CMAKE_CUDA_ARCHITECTURES defined at this (root) scope: its try_compile
# fails on an empty value. "native" targets the local GPU, which USE_CUDA implies.
if(WIN32 AND USE_CUDA AND CUDAToolkit_VERSION VERSION_GREATER_EQUAL 13
   AND NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES native)
endif()

# --- ImGuiFileDialog (File open/save dialog for ImGui) ---
message(STATUS "  Configuring ImGuiFileDialog...")
# Upstream ships a CMakeLists.txt that tries find_package(imgui) QUIET and skips
# linking when not found. We use our own imgui submodule target instead.
set(IGFD_INSTALL OFF CACHE BOOL "" FORCE)
add_subdirectory(${THIRD_PARTY_DIR}/ImGuiFileDialog EXCLUDE_FROM_ALL)
target_link_libraries(ImGuiFileDialog PUBLIC imgui)
set_target_properties(ImGuiFileDialog PROPERTIES POSITION_INDEPENDENT_CODE ON)
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
# APPLET-ONLY. libcaliper / the caliper exe / embed_host never link libtorch:
# TORCH_LIBRARIES is consumed by examples/mnist/applet and the
# torch-labelled tests, never appended to CALIPER_DEPENDENCY_LIBS. With
# CALIPER_WITH_TORCH=OFF the whole category is skipped (no download, no
# find_package(Torch)) and TORCH_LIBRARIES stays empty.
if(NOT CALIPER_WITH_TORCH)
    message(STATUS "Skipping PyTorch (libtorch): CALIPER_WITH_TORCH=OFF (torch-free libcaliper)")
    set(TORCH_LIBRARIES "")
else()

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

    # TORCH_LIBRARIES is deliberately NOT appended to CALIPER_DEPENDENCY_LIBS:
    # only applets / torch tests link it (torch-free libcaliper).

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

    # TORCH_LIBRARIES is deliberately NOT appended to CALIPER_DEPENDENCY_LIBS:
    # only applets / torch tests link it (torch-free libcaliper).

    message(STATUS "  ✓ PyTorch configured via find_package(Torch)")

endif()  # WIN32 vs macOS/Linux

endif()  # CALIPER_WITH_TORCH

# ============================================================================
# Export dependency list
# ============================================================================

# CALIPER_DEPENDENCY_LIBS is already in the correct scope
# (include() doesn't create a new scope, so no PARENT_SCOPE needed)

message(STATUS "")
message(STATUS "Dependencies configured successfully:")
message(STATUS "  Total libraries to link: ${CALIPER_DEPENDENCY_LIBS}")
message(STATUS "  libcaliper core libs:    ${CALIPER_CORE_LIBS}")
message(STATUS "")
