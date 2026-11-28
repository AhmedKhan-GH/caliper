# The embed install proof, run by ctest as caliper_embed_install:
#   1. install the embed component of the core build, alone, to
#      <WORK_DIR>/prefix and check the installed tree: the headers under
#      include/caliper-embed, nothing under the SDK's include/caliper, the
#      config and version files, and no path into the source or build tree in
#      the installed CMake files;
#   2. with NM and CORE_LIBRARY (macOS), check that the embedding library
#      exports exactly the C functions that libcaliper.a defines;
#   3. copy examples/embed_host to <WORK_DIR>, build the copy with only
#      -DCMAKE_PREFIX_PATH=<WORK_DIR>/prefix, and check that the build names
#      nothing in the source tree;
#   4. run the copy in its offscreen mode, which creates no window, on a copy
#      of the hello_world fixture from INSTRUMENTS_DIR, and check that hello_world loaded and
#      drew. When the canvas cannot attach (no GPU) the run is reported as
#      skipped, unless CALIPER_REQUIRE_GPU=1 makes that a failure.
#
#   cmake -DCALIPER_BUILD_DIR=<build> -DCALIPER_SOURCE_DIR=<source> -DWORK_DIR=<dir>
#         -DEXPECT_VERSION=<major.minor.patch> -DINSTRUMENTS_DIR=<dir with hello_world>
#         [-DGENERATOR=<generator>] [-DMAKE_PROGRAM=<path>] [-DCXX_COMPILER=<path>]
#         [-DOBJCXX_COMPILER=<path>] [-DBUILD_TYPE=<type>] [-DNM=<nm>]
#         [-DCORE_LIBRARY=<libcaliper.a>] [-DLIB_PREFIX=lib] [-DLIB_SUFFIX=.dylib]
#         [-DEXE_SUFFIX=.exe] -P run.cmake
cmake_minimum_required(VERSION 3.18)

foreach(var CALIPER_BUILD_DIR CALIPER_SOURCE_DIR WORK_DIR EXPECT_VERSION INSTRUMENTS_DIR)
    if(NOT ${var})
        message(FATAL_ERROR "run.cmake needs -D${var}=...")
    endif()
endforeach()
if(NOT BUILD_TYPE)
    set(BUILD_TYPE Release)
endif()

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

# ---- 3. examples/embed_host outside the source tree ----
set(configure_args "-DCMAKE_PREFIX_PATH=${prefix}" "-DCMAKE_BUILD_TYPE=${BUILD_TYPE}")
if(GENERATOR)
    list(APPEND configure_args -G "${GENERATOR}")
endif()
if(MAKE_PROGRAM)
    list(APPEND configure_args "-DCMAKE_MAKE_PROGRAM=${MAKE_PROGRAM}")
endif()
if(CXX_COMPILER)
    list(APPEND configure_args "-DCMAKE_CXX_COMPILER=${CXX_COMPILER}")
endif()
if(OBJCXX_COMPILER)
    list(APPEND configure_args "-DCMAKE_OBJCXX_COMPILER=${OBJCXX_COMPILER}")
endif()
set(host_src "${WORK_DIR}/embed_host")
set(host_build "${WORK_DIR}/embed_host-build")
file(COPY "${CALIPER_SOURCE_DIR}/examples/embed_host/" DESTINATION "${host_src}")
run("Configuring embed_host" "${CMAKE_COMMAND}" -S "${host_src}" -B "${host_build}" ${configure_args})
run("Building embed_host" "${CMAKE_COMMAND}" --build "${host_build}")

file(READ "${host_build}/CMakeCache.txt" cache)
if(NOT cache MATCHES "\ncaliper-embed_DIR:PATH=${prefix}/lib/cmake/caliper-embed\n")
    message(FATAL_ERROR "embed_host did not find caliper-embed in ${prefix}")
endif()
foreach(file CMakeCache.txt build.ninja)
    if(NOT EXISTS "${host_build}/${file}")
        continue()
    endif()
    file(READ "${host_build}/${file}" text)
    foreach(root "${CALIPER_SOURCE_DIR}" "${src_real}")
        foreach(sub sdk src include examples third_party instruments cmake)
            string(FIND "${text}" "${root}/${sub}/" at)
            if(NOT at EQUAL -1)
                message(FATAL_ERROR "embed_host: ${file} refers to the source tree (${root}/${sub}/)")
            endif()
        endforeach()
    endforeach()
endforeach()
set(host_exe "${host_build}/embed_host${EXE_SUFFIX}")
if(NOT EXISTS "${host_exe}")
    message(FATAL_ERROR "embed_host: ${host_exe} was not built")
endif()

# ---- 4. The offscreen run ----
# HOME and CALIPER_USER_INSTRUMENTS_DIR keep the core's data folder and user
# instruments inside the work folder (macOS), so the run reads and writes nothing
# of the user's own Caliper data and finds only the hello_world copy.
set(instruments "${WORK_DIR}/instruments")
file(MAKE_DIRECTORY "${instruments}")
foreach(name "${LIB_PREFIX}hello_world${LIB_SUFFIX}" hello_world.caliper.toml)
    if(NOT EXISTS "${INSTRUMENTS_DIR}/${name}")
        message(FATAL_ERROR "the hello_world fixture is not built: ${INSTRUMENTS_DIR}/${name}")
    endif()
    file(COPY "${INSTRUMENTS_DIR}/${name}" DESTINATION "${instruments}")
endforeach()
execute_process(
    COMMAND "${CMAKE_COMMAND}" -E env --unset=CALIPER_INSTRUMENT_PATH
        CALIPER_EMBED_OFFSCREEN=1 CALIPER_EMBED_EXIT_AFTER=1
        "CALIPER_EMBED_INSTRUMENTS=${instruments}"
        "CALIPER_USER_INSTRUMENTS_DIR=${WORK_DIR}/user-instruments"
        "HOME=${WORK_DIR}/home"
        "${host_exe}" ai.caliperapp.hello-world
    RESULT_VARIABLE rc OUTPUT_VARIABLE out ERROR_VARIABLE out TIMEOUT 120)
message(STATUS "embed_host output:\n${out}")
if(rc EQUAL 0)
    foreach(expect
            "offscreen mode, embed ${EXPECT_VERSION}"
            "[instrument] hello_world.on_init"
            "[instrument] hello_world.on_cleanup")
        string(FIND "${out}" "${expect}" at)
        if(at EQUAL -1)
            message(FATAL_ERROR "embed_host: the output lacks \"${expect}\"")
        endif()
    endforeach()
    if(NOT out MATCHES "offscreen: [1-9][0-9]* frames of dev\\.caliper\\.hello-world at 1280x800 in [0-9.]+ s, [1-9][0-9]* drawn pixels")
        message(FATAL_ERROR "embed_host: the offscreen run did not report drawn frames")
    endif()
elseif(out MATCHES "attach_canvas failed" AND NOT "$ENV{CALIPER_REQUIRE_GPU}" STREQUAL "1")
    message(STATUS "embed install proof: OK up to the build; headless run skipped: no GPU")
    return()
else()
    message(FATAL_ERROR "embed_host: the offscreen run failed (exit ${rc})")
endif()

message(STATUS "embed install proof: OK (${host_exe} ran offscreen against ${prefix})")
