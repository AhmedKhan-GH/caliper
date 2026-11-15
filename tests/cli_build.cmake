# `caliper build <folder>` as an applet author runs it, run by ctest as
# caliper_cli_build:
#   1. install the SDK component of this build to <WORK_DIR>/prefix;
#   2. run `caliper new` into <WORK_DIR>/built;
#   3. run `caliper build` on it with CALIPER_SDK_DIR naming the prefix: it
#      must configure, build and pass `caliper check`, and leave the library
#      and its manifest in build/applets;
#   4. the clear errors: CALIPER_SDK_DIR without an SDK, CALIPER_CMAKE that is
#      not a program, a project that finds no SDK anywhere, and a compiler that
#      does not exist (each on a fresh project, so no cache hides the cause);
#   5. check that no step created the run's app-data folder.
#
#   cmake -DEXE=<caliper> -DCALIPER_BUILD_DIR=<build> -DWORK_DIR=<dir>
#         [-DGENERATOR=<generator>] [-DMAKE_PROGRAM=<path>] [-DCXX_COMPILER=<path>]
#         -P cli_build.cmake
cmake_minimum_required(VERSION 3.18)
foreach(var EXE CALIPER_BUILD_DIR WORK_DIR)
    if(NOT ${var})
        message(FATAL_ERROR "cli_build.cmake needs -D${var}=...")
    endif()
endforeach()

file(REMOVE_RECURSE "${WORK_DIR}")
file(MAKE_DIRECTORY "${WORK_DIR}/home")
set(prefix "${WORK_DIR}/prefix")
set(data "${WORK_DIR}/app-data")

# The environment of every caliper run: no app-data or user applets folder of
# the user's, a home folder without caliper-sdk, no CMAKE_PREFIX_PATH, and the
# generator, build tool and compiler of this build.
set(env --unset=CALIPER_APPLET_PATH --unset=CALIPER_SDK_DIR --unset=CMAKE_PREFIX_PATH
    "CALIPER_DATA_DIR=${data}" "CALIPER_USER_APPLETS_DIR=${WORK_DIR}/user-applets"
    "HOME=${WORK_DIR}/home" "CALIPER_CMAKE=${CMAKE_COMMAND}")
if(GENERATOR)
    list(APPEND env "CMAKE_GENERATOR=${GENERATOR}")
endif()
if(MAKE_PROGRAM)
    get_filename_component(tool_dir "${MAKE_PROGRAM}" DIRECTORY)
    list(APPEND env "PATH=${tool_dir}:$ENV{PATH}")
endif()

# caliper with `env` plus the extra assignments in ARGN, then its arguments
# after "--": sets rc and out in the caller.
function(caliper)
    cmake_parse_arguments(C "" "" "ENV;ARGS" ${ARGN})
    execute_process(COMMAND "${CMAKE_COMMAND}" -E env ${env} ${C_ENV} "${EXE}" ${C_ARGS}
        RESULT_VARIABLE r OUTPUT_VARIABLE o ERROR_VARIABLE o)
    set(rc "${r}" PARENT_SCOPE)
    set(out "${o}" PARENT_SCOPE)
endfunction()

function(expect_fail what pattern)
    if(rc EQUAL 0 OR NOT out MATCHES "${pattern}")
        message(FATAL_ERROR "${what}: expected exit 1 and \"${pattern}\", got exit ${rc}:\n${out}")
    endif()
    message(STATUS "caliper build: ${what}: refused as expected")
endfunction()

execute_process(COMMAND "${CMAKE_COMMAND}" --install "${CALIPER_BUILD_DIR}" --prefix "${prefix}"
    --component sdk RESULT_VARIABLE rc OUTPUT_VARIABLE out ERROR_VARIABLE out)
if(NOT rc EQUAL 0)
    message(FATAL_ERROR "Installing the SDK failed:\n${out}")
endif()
foreach(name built no-sdk no-compiler)
    caliper(ARGS new com.example.${name} "${WORK_DIR}/${name}")
    if(NOT rc EQUAL 0)
        message(FATAL_ERROR "caliper new ${name} failed:\n${out}")
    endif()
endforeach()

# The applet, built and checked.
set(compiler_env)
if(CXX_COMPILER)
    set(compiler_env "CXX=${CXX_COMPILER}")
endif()
caliper(ENV "CALIPER_SDK_DIR=${prefix}" ${compiler_env} ARGS build "${WORK_DIR}/built")
if(NOT rc EQUAL 0)
    message(FATAL_ERROR "caliper build failed (exit ${rc}):\n${out}")
endif()
if(NOT out MATCHES "using the Caliper SDK in [^\n]*prefix\n")
    message(FATAL_ERROR "caliper build did not name the SDK it used:\n${out}")
endif()
if(NOT out MATCHES "READY +com\\.example\\.built +0\\.1\\.0 .*1 of 1 applets pass the checks")
    message(FATAL_ERROR "caliper build did not pass the checks:\n${out}")
endif()
if(NOT EXISTS "${WORK_DIR}/built/build/applets/built.caliper.toml")
    message(FATAL_ERROR "caliper build left no manifest in built/build/applets")
endif()
message(STATUS "caliper build: configured, built and checked ${WORK_DIR}/built")

# A second run builds nothing new and passes again.
caliper(ENV "CALIPER_SDK_DIR=${prefix}" ${compiler_env} ARGS build "${WORK_DIR}/built")
if(NOT rc EQUAL 0 OR NOT out MATCHES "1 of 1 applets pass the checks")
    message(FATAL_ERROR "a second caliper build failed (exit ${rc}):\n${out}")
endif()

# The clear errors.
caliper(ENV "CALIPER_SDK_DIR=${WORK_DIR}/home" ARGS build "${WORK_DIR}/no-sdk")
expect_fail("CALIPER_SDK_DIR without an SDK" "CALIPER_SDK_DIR is [^\n]*, which holds no Caliper SDK")
caliper(ENV "CALIPER_CMAKE=${WORK_DIR}/no-cmake" ARGS build "${WORK_DIR}/no-sdk")
expect_fail("CALIPER_CMAKE that is not a program" "CALIPER_CMAKE is [^\n]*no-cmake, which is not a program")
caliper(ENV ${compiler_env} ARGS build "${WORK_DIR}/no-sdk")
expect_fail("no SDK anywhere" "no Caliper SDK in .*caliper: no Caliper SDK was found")
caliper(ENV "CALIPER_SDK_DIR=${prefix}" "CXX=${WORK_DIR}/no-such-compiler" ARGS build "${WORK_DIR}/no-compiler")
expect_fail("a compiler that does not exist" "caliper: CMake found no working C\\+\\+ compiler")

if(EXISTS "${data}")
    message(FATAL_ERROR "a step created the app-data folder ${data}")
endif()
message(STATUS "caliper build: built, checked and refused as expected (${WORK_DIR})")
