# The SDK install proof, run by ctest as caliper_sdk_install:
#   1. install the SDK component of the core build to <WORK_DIR>/prefix;
#   2. copy this folder and examples/hello to <WORK_DIR> and build each copy
#      with only -DCMAKE_PREFIX_PATH=<WORK_DIR>/prefix;
#   3. run the probe's doctest binary, check that hello's library and stamped
#      manifest exist, and check that neither build refers to the source tree.
#
#   cmake -DCALIPER_BUILD_DIR=<build> -DCALIPER_SOURCE_DIR=<source> -DWORK_DIR=<dir>
#         [-DGENERATOR=<generator>] [-DMAKE_PROGRAM=<path>] [-DCXX_COMPILER=<path>]
#         [-DBUILD_TYPE=<type>] [-DEXPECT_SDK_VERSION=<version>]
#         [-DEXPECT_UI_STACK_HASH=<hash>] [-DLIB_PREFIX=lib] [-DLIB_SUFFIX=.dylib]
#         [-DEXE_SUFFIX=.exe] -P run.cmake
cmake_minimum_required(VERSION 3.18)

foreach(var CALIPER_BUILD_DIR CALIPER_SOURCE_DIR WORK_DIR)
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
run("Installing the SDK"
    "${CMAKE_COMMAND}" --install "${CALIPER_BUILD_DIR}" --prefix "${prefix}" --component sdk)

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

# Builds a copy of <source>/<folder> in <WORK_DIR>/<name>.
function(build_copy name folder)
    file(COPY "${CALIPER_SOURCE_DIR}/${folder}/" DESTINATION "${WORK_DIR}/${name}")
    run("Configuring ${name}" "${CMAKE_COMMAND}" -S "${WORK_DIR}/${name}"
        -B "${WORK_DIR}/${name}-build" ${configure_args})
    run("Building ${name}" "${CMAKE_COMMAND}" --build "${WORK_DIR}/${name}-build")
endfunction()

# A build that names a file in the source tree did not use the package alone.
function(check_no_source_refs name)
    get_filename_component(src_real "${CALIPER_SOURCE_DIR}" REALPATH)
    foreach(file CMakeCache.txt build.ninja)
        set(path "${WORK_DIR}/${name}-build/${file}")
        if(NOT EXISTS "${path}")
            continue()
        endif()
        file(READ "${path}" text)
        foreach(root "${CALIPER_SOURCE_DIR}" "${src_real}")
            foreach(sub sdk src include examples third_party applets cmake)
                string(FIND "${text}" "${root}/${sub}/" at)
                if(NOT at EQUAL -1)
                    message(FATAL_ERROR "${name}: ${file} refers to the source tree (${root}/${sub}/)")
                endif()
            endforeach()
        endforeach()
    endforeach()
endfunction()

build_copy(probe tests/sdk_install_probe)
check_no_source_refs(probe)
run("Running the probe" "${WORK_DIR}/probe-build/probe${EXE_SUFFIX}")

build_copy(hello examples/hello)
check_no_source_refs(hello)
set(applets "${WORK_DIR}/hello-build/applets")
set(library "${applets}/${LIB_PREFIX}hello${LIB_SUFFIX}")
set(manifest "${applets}/hello.caliper.toml")
foreach(path "${library}" "${manifest}")
    if(NOT EXISTS "${path}")
        message(FATAL_ERROR "hello: ${path} was not built")
    endif()
endforeach()
file(READ "${manifest}" stamped)
foreach(line "[build]" "sdk = \"${EXPECT_SDK_VERSION}\"" "ui_stack = \"${EXPECT_UI_STACK_HASH}\"")
    string(FIND "${stamped}" "${line}" at)
    if(at EQUAL -1)
        message(FATAL_ERROR "hello: the stamped manifest lacks the line: ${line}")
    endif()
endforeach()
message(STATUS "sdk install proof: OK (${library})")
