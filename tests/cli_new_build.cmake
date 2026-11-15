# The template of `caliper new`, built as an applet author builds it, run by
# ctest as caliper_cli_new_builds:
#   1. install the SDK component of this build to <WORK_DIR>/prefix;
#   2. run `caliper new` into <WORK_DIR>/template-check;
#   3. configure and build that project with only -DCMAKE_PREFIX_PATH set,
#      and run its own test;
#   4. run `caliper check` on its build folder, which must pass;
#   5. check that no step created the run's app-data folder.
#
#   cmake -DEXE=<caliper> -DCALIPER_BUILD_DIR=<build> -DWORK_DIR=<dir>
#         [-DGENERATOR=<generator>] [-DMAKE_PROGRAM=<path>] [-DCXX_COMPILER=<path>]
#         [-DBUILD_TYPE=<type>] -P cli_new_build.cmake
cmake_minimum_required(VERSION 3.18)
foreach(var EXE CALIPER_BUILD_DIR WORK_DIR)
    if(NOT ${var})
        message(FATAL_ERROR "cli_new_build.cmake needs -D${var}=...")
    endif()
endforeach()
if(NOT BUILD_TYPE)
    set(BUILD_TYPE Release)
endif()

function(run what)
    execute_process(COMMAND ${ARGN} RESULT_VARIABLE rc OUTPUT_VARIABLE out ERROR_VARIABLE out)
    if(NOT rc EQUAL 0)
        message(FATAL_ERROR "${what} failed (exit ${rc}):\n${out}")
    endif()
    set(run_output "${out}" PARENT_SCOPE)
endfunction()

file(REMOVE_RECURSE "${WORK_DIR}")
file(MAKE_DIRECTORY "${WORK_DIR}")
set(prefix "${WORK_DIR}/prefix")
set(project "${WORK_DIR}/template-check")
set(data "${WORK_DIR}/app-data")
set(caliper "${CMAKE_COMMAND}" -E env --unset=CALIPER_APPLET_PATH
    "CALIPER_DATA_DIR=${data}" "CALIPER_USER_APPLETS_DIR=${WORK_DIR}/user-applets" "${EXE}")

run("Installing the SDK" "${CMAKE_COMMAND}" --install "${CALIPER_BUILD_DIR}"
    --prefix "${prefix}" --component sdk)
run("caliper new" ${caliper} new com.example.template-check "${project}")

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
run("Configuring the template" "${CMAKE_COMMAND}" -S "${project}" -B "${project}/build"
    ${configure_args})
run("Building the template" "${CMAKE_COMMAND}" --build "${project}/build" --config "${BUILD_TYPE}")
run("Running the template's test" "${CMAKE_CTEST_COMMAND}" --test-dir "${project}/build"
    -C "${BUILD_TYPE}" --output-on-failure)

run("caliper check" ${caliper} check "${project}/build")
if(NOT run_output MATCHES "READY +com\\.example\\.template-check +0\\.1\\.0 .*1 of 1 applets pass the checks")
    message(FATAL_ERROR "caliper check did not pass the template:\n${run_output}")
endif()
if(EXISTS "${data}")
    message(FATAL_ERROR "a step created the app-data folder ${data}")
endif()
message(STATUS "caliper new template: configured, built, tested and checked (${project})")
