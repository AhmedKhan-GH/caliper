# The embed install proof, run by ctest as caliper_embed_install:
#   1. install the embed component of the core build, alone, to
#      <WORK_DIR>/prefix and check the installed tree: the headers under
#      include/caliper-embed, nothing under the SDK's include/caliper, the
#      config and version files, and no path into the source or build tree in
#      the installed CMake files;
#   2. with NM and CORE_LIBRARY (macOS), check that the embedding library
#      exports exactly the C functions that libcaliper.a defines.
#
#   cmake -DCALIPER_BUILD_DIR=<build> -DCALIPER_SOURCE_DIR=<source> -DWORK_DIR=<dir>
#         -DEXPECT_VERSION=<major.minor.patch> [-DNM=<nm>] [-DCORE_LIBRARY=<libcaliper.a>]
#         [-DLIB_PREFIX=lib] [-DLIB_SUFFIX=.dylib] -P run.cmake
cmake_minimum_required(VERSION 3.18)

foreach(var CALIPER_BUILD_DIR CALIPER_SOURCE_DIR WORK_DIR EXPECT_VERSION)
    if(NOT ${var})
        message(FATAL_ERROR "run.cmake needs -D${var}=...")
    endif()
endforeach()

function(run what)
    execute_process(COMMAND ${ARGN} RESULT_VARIABLE rc)
    if(NOT rc EQUAL 0)
        message(FATAL_ERROR "${what} failed (exit ${rc})")
    endif()
endfunction()

file(REMOVE_RECURSE "${WORK_DIR}")
file(MAKE_DIRECTORY "${WORK_DIR}")
set(prefix "${WORK_DIR}/prefix")
run("Installing the embed package"
    "${CMAKE_COMMAND}" --install "${CALIPER_BUILD_DIR}" --prefix "${prefix}" --component embed)

# ---- 1. The installed tree ----
set(package "${prefix}/lib/cmake/caliper-embed")
foreach(path
        include/caliper-embed/caliper/embed.h
        include/caliper-embed/caliper/live.h
        include/caliper-embed/caliper/live_views.h
        include/caliper-embed/caliper/tensor.h
        include/caliper-embed/caliper/services/metrics_v1.h
        lib/cmake/caliper-embed/caliper-embed-config.cmake
        lib/cmake/caliper-embed/caliper-embed-config-version.cmake
        lib/cmake/caliper-embed/CaliperEmbedTargets.cmake)
    if(NOT EXISTS "${prefix}/${path}")
        message(FATAL_ERROR "the embed package lacks ${path}")
    endif()
endforeach()
if(EXISTS "${prefix}/include/caliper")
    message(FATAL_ERROR "the embed package installed into the SDK include root include/caliper")
endif()
file(READ "${package}/caliper-embed-config-version.cmake" version_file)
if(NOT version_file MATCHES "set\\(PACKAGE_VERSION \"${EXPECT_VERSION}\"\\)")
    message(FATAL_ERROR "the package version is not ${EXPECT_VERSION}")
endif()
get_filename_component(src_real "${CALIPER_SOURCE_DIR}" REALPATH)
get_filename_component(build_real "${CALIPER_BUILD_DIR}" REALPATH)
file(GLOB cmake_files "${package}/*.cmake")
foreach(file IN LISTS cmake_files)
    file(READ "${file}" text)
    foreach(root "${CALIPER_SOURCE_DIR}" "${src_real}" "${CALIPER_BUILD_DIR}" "${build_real}")
        string(FIND "${text}" "${root}" at)
        if(NOT at EQUAL -1)
            message(FATAL_ERROR "${file} names ${root}")
        endif()
    endforeach()
endforeach()

# ---- 2. The exports (macOS) ----
# cmake/CaliperEmbed.cmake derives the export list from the headers. Every C
# function libcaliper.a defines is part of the interface, so the two sets must
# be equal; a declaration the header scan misses shows up here.
if(NM AND CORE_LIBRARY)
    set(library "${prefix}/lib/${LIB_PREFIX}caliper_embed${LIB_SUFFIX}")
    execute_process(COMMAND "${NM}" -gU "${library}"
        RESULT_VARIABLE rc OUTPUT_VARIABLE exported)
    if(NOT rc EQUAL 0)
        message(FATAL_ERROR "nm failed on ${library}")
    endif()
    execute_process(COMMAND "${NM}" -g "${CORE_LIBRARY}"
        RESULT_VARIABLE rc OUTPUT_VARIABLE defined ERROR_QUIET)
    if(NOT rc EQUAL 0)
        message(FATAL_ERROR "nm failed on ${CORE_LIBRARY}")
    endif()
    string(REGEX MATCHALL " T _caliper_[a-z0-9_]+" defined "${defined}")
    string(REGEX MATCHALL " [A-Za-z] _[A-Za-z0-9_]+" exported "${exported}")
    list(TRANSFORM defined REPLACE "^ T _" "")
    list(TRANSFORM exported REPLACE "^ [A-Za-z] _" "")
    list(REMOVE_DUPLICATES defined)
    list(SORT defined)
    list(SORT exported)
    if(NOT defined STREQUAL exported)
        message(FATAL_ERROR "the embedding library exports\n  ${exported}\n"
            "but libcaliper.a defines the C functions\n  ${defined}")
    endif()
    list(LENGTH exported count)
    message(STATUS "the embedding library exports the ${count} C functions of libcaliper.a and nothing else")
endif()

message(STATUS "embed install proof: OK (${prefix})")
