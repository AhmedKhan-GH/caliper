# The runtime WGSL compiler (tools/shaders/caliper-wgsl), as the imported target caliper::wgsl.
#
# The library is Naga behind the C interface of include/caliper/wgsl.h. It is a Rust static
# library, prebuilt for each platform by CI (.github/workflows/wgsl.yml) and published as a
# release asset, so that a build of Caliper needs no Rust. The repository holds no binary: it
# pins each platform's archive by SHA-256 in tools/shaders/caliper-wgsl/prebuilt.cmake.
#
# The target exists only when the library is there. Without it nothing changes: no target, no
# definition, CALIPER_HAS_WGSL is OFF and the tests that need it report a skip. Three ways give
# the library, in this order:
#
#   CALIPER_WGSL_LIBRARY=<file>    a library that a script or another build placed;
#   CALIPER_WGSL_DOWNLOAD=ON       the pinned release archive for this platform, downloaded at
#                                  configure time into the build folder and checked against
#                                  its SHA-256 (CALIPER_WGSL_URL names the release folder);
#   CALIPER_WGSL_FROM_SOURCE=ON    a build from source with cargo, through build_library.py, at
#                                  build time (needs Rust and Python).
#
# A library already in <build>/caliper-wgsl/lib/ from an earlier download is used as well. Each
# explicit request fails the configure with the reason when it cannot give the library.
#
# The Python wheel configures this same tree (pyproject.toml), thus it gets the target in the
# same way: a define in [tool.scikit-build.cmake.define], such as CALIPER_WGSL_DOWNLOAD = "ON"
# once the archives are pinned, or SKBUILD_CMAKE_DEFINE=CALIPER_WGSL_LIBRARY=<file> for one
# build. The library is static, thus the native module that uses it (the producer runtime of M2)
# links caliper::wgsl under if(TARGET caliper::wgsl) and the wheel carries no extra file.

set(CALIPER_WGSL_LIBRARY "" CACHE FILEPATH "Prebuilt caliper_wgsl static library (empty: none)")
option(CALIPER_WGSL_DOWNLOAD "Download the pinned prebuilt caliper_wgsl library for this platform" OFF)
option(CALIPER_WGSL_FROM_SOURCE "Build the caliper_wgsl library from source with cargo" OFF)
set(CALIPER_WGSL_URL "" CACHE STRING
    "Release folder of the prebuilt caliper_wgsl archives (https://github.com/<owner>/<repository>/releases/download/<tag>)")

set(_wgsl_crate "${CMAKE_CURRENT_LIST_DIR}/../tools/shaders/caliper-wgsl")
get_filename_component(_wgsl_crate "${_wgsl_crate}" ABSOLUTE)
include("${_wgsl_crate}/prebuilt.cmake")

if(WIN32)
    set(_wgsl_file "caliper_wgsl.lib")
else()
    set(_wgsl_file "libcaliper_wgsl.a")
endif()
if(CMAKE_SYSTEM_PROCESSOR MATCHES "^(arm64|aarch64|ARM64)$")
    set(_wgsl_arch "arm64")
else()
    set(_wgsl_arch "x64")
endif()
if(APPLE)
    set(_wgsl_platform "macos-${_wgsl_arch}")
elseif(WIN32)
    set(_wgsl_platform "windows-${_wgsl_arch}")
else()
    set(_wgsl_platform "linux-${_wgsl_arch}")
endif()

set(_wgsl_dir "${CMAKE_BINARY_DIR}/caliper-wgsl")
set(_wgsl_lib "")
set(_wgsl_build_target "")

if(CALIPER_WGSL_LIBRARY)
    if(NOT EXISTS "${CALIPER_WGSL_LIBRARY}")
        message(FATAL_ERROR "CALIPER_WGSL_LIBRARY names ${CALIPER_WGSL_LIBRARY}, which does not exist. "
            "Build it with tools/shaders/caliper-wgsl/build_library.py, or clear the variable.")
    endif()
    set(_wgsl_lib "${CALIPER_WGSL_LIBRARY}")
elseif(CALIPER_WGSL_FROM_SOURCE)
    find_program(CALIPER_CARGO cargo)
    find_package(Python3 COMPONENTS Interpreter)
    if(NOT CALIPER_CARGO OR NOT Python3_Interpreter_FOUND)
        message(FATAL_ERROR "CALIPER_WGSL_FROM_SOURCE needs cargo and Python 3 on PATH "
            "(cargo: '${CALIPER_CARGO}'). Install Rust, or use CALIPER_WGSL_DOWNLOAD=ON.")
    endif()
    set(_wgsl_lib "${_wgsl_dir}/source/lib/${_wgsl_file}")
    add_custom_command(
        OUTPUT "${_wgsl_lib}"
        COMMAND "${Python3_EXECUTABLE}" "${_wgsl_crate}/build_library.py" --out "${_wgsl_dir}/source"
        DEPENDS "${_wgsl_crate}/Cargo.toml" "${_wgsl_crate}/Cargo.lock" "${_wgsl_crate}/src/lib.rs"
                "${_wgsl_crate}/build_library.py"
        COMMENT "Building caliper_wgsl (Naga ${CALIPER_WGSL_NAGA_VERSION}) with cargo"
        USES_TERMINAL VERBATIM)
    add_custom_target(caliper_wgsl_build DEPENDS "${_wgsl_lib}")
    set(_wgsl_build_target caliper_wgsl_build)
else()
    set(_wgsl_archive "caliper_wgsl-${CALIPER_WGSL_PREBUILT_VERSION}-${_wgsl_platform}.tar.gz")
    set(_wgsl_hash "${CALIPER_WGSL_SHA256_${_wgsl_platform}}")
    set(_wgsl_stamp "${_wgsl_dir}/lib/${_wgsl_file}.sha256-of-archive")
    set(_wgsl_have "")
    if(EXISTS "${_wgsl_dir}/lib/${_wgsl_file}" AND EXISTS "${_wgsl_stamp}")
        file(READ "${_wgsl_stamp}" _wgsl_have)
        string(STRIP "${_wgsl_have}" _wgsl_have)
    endif()
    if(_wgsl_hash AND _wgsl_have STREQUAL _wgsl_hash)
        set(_wgsl_lib "${_wgsl_dir}/lib/${_wgsl_file}")
    elseif(CALIPER_WGSL_DOWNLOAD)
        if(NOT _wgsl_hash)
            message(FATAL_ERROR "CALIPER_WGSL_DOWNLOAD: no prebuilt caliper_wgsl ${CALIPER_WGSL_PREBUILT_VERSION} "
                "is pinned for ${_wgsl_platform} in tools/shaders/caliper-wgsl/prebuilt.cmake. "
                "Use CALIPER_WGSL_FROM_SOURCE=ON with Rust, or CALIPER_WGSL_LIBRARY=<file>.")
        endif()
        if(NOT CALIPER_WGSL_URL)
            message(FATAL_ERROR "CALIPER_WGSL_DOWNLOAD needs CALIPER_WGSL_URL, the release folder that holds "
                "${_wgsl_archive}.")
        endif()
        file(DOWNLOAD "${CALIPER_WGSL_URL}/${_wgsl_archive}" "${_wgsl_dir}/${_wgsl_archive}"
            EXPECTED_HASH SHA256=${_wgsl_hash} STATUS _wgsl_status TLS_VERIFY ON)
        list(GET _wgsl_status 0 _wgsl_code)
        if(NOT _wgsl_code EQUAL 0)
            file(REMOVE "${_wgsl_dir}/${_wgsl_archive}")
            message(FATAL_ERROR "CALIPER_WGSL_DOWNLOAD: ${CALIPER_WGSL_URL}/${_wgsl_archive} failed: ${_wgsl_status}")
        endif()
        file(REMOVE_RECURSE "${_wgsl_dir}/lib")
        file(ARCHIVE_EXTRACT INPUT "${_wgsl_dir}/${_wgsl_archive}" DESTINATION "${_wgsl_dir}")
        if(NOT EXISTS "${_wgsl_dir}/lib/${_wgsl_file}")
            message(FATAL_ERROR "CALIPER_WGSL_DOWNLOAD: ${_wgsl_archive} holds no lib/${_wgsl_file}")
        endif()
        file(WRITE "${_wgsl_stamp}" "${_wgsl_hash}\n")
        set(_wgsl_lib "${_wgsl_dir}/lib/${_wgsl_file}")
    endif()
endif()

if(_wgsl_lib)
    add_library(caliper_wgsl STATIC IMPORTED GLOBAL)
    set_target_properties(caliper_wgsl PROPERTIES
        IMPORTED_LOCATION "${_wgsl_lib}"
        INTERFACE_INCLUDE_DIRECTORIES "${_wgsl_crate}/include"
        INTERFACE_COMPILE_DEFINITIONS "CALIPER_HAS_WGSL=1")
    # The system libraries that Rust's standard library needs in a static link.
    if(WIN32)
        set_property(TARGET caliper_wgsl PROPERTY INTERFACE_LINK_LIBRARIES
            kernel32 advapi32 ntdll userenv ws2_32 dbghelp bcrypt)
    elseif(NOT APPLE)
        find_package(Threads REQUIRED)
        set_property(TARGET caliper_wgsl PROPERTY INTERFACE_LINK_LIBRARIES
            Threads::Threads ${CMAKE_DL_LIBS} m)
    endif()
    if(_wgsl_build_target)
        add_dependencies(caliper_wgsl ${_wgsl_build_target})
    endif()
    add_library(caliper::wgsl ALIAS caliper_wgsl)
    set(CALIPER_HAS_WGSL ON)
    message(STATUS "caliper_wgsl: ${_wgsl_lib} (Naga ${CALIPER_WGSL_NAGA_VERSION})")
else()
    set(CALIPER_HAS_WGSL OFF)
    message(STATUS "caliper_wgsl: absent (CALIPER_WGSL_LIBRARY, CALIPER_WGSL_DOWNLOAD or "
        "CALIPER_WGSL_FROM_SOURCE gives it); the runtime WGSL compiler is off")
endif()
