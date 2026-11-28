# caliper._netio on its own: the native sender of caliper.serve() built from
# python/src/_netio.cpp and src/host/net (caliper_net) with nothing else of the
# project, so it builds in seconds. scripts/build_netio.py runs it and puts the
# module in python/caliper/, where PYTHONPATH=python finds it. The wheel builds
# the same module from python/CMakeLists.txt.
#
#   cmake -S python/netio -B build/netio -G Ninja -DPython_EXECUTABLE=.venv/bin/python
#   cmake --build build/netio
#
# -DNETIO_ASAN=ON builds it and the caliper_net tests (netio_net_tests) with
# AddressSanitizer.
cmake_minimum_required(VERSION 3.20)
# For macOS 14.0 (Sonoma) or later, as the core build (before project()).
include(${CMAKE_CURRENT_SOURCE_DIR}/../../cmake/CaliperMacOSTarget.cmake)
project(caliper_netio LANGUAGES CXX)

set(REPO ${CMAKE_CURRENT_SOURCE_DIR}/../..)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()
option(NETIO_ASAN "Build with AddressSanitizer" OFF)
if(NETIO_ASAN)
    add_compile_options(-fsanitize=address -fno-omit-frame-pointer -g)
    add_link_options(-fsanitize=address)
endif()

find_package(Python 3.12 COMPONENTS Interpreter Development.Module REQUIRED)
execute_process(
    COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
    OUTPUT_STRIP_TRAILING_WHITESPACE
    OUTPUT_VARIABLE nanobind_ROOT)
find_package(nanobind CONFIG REQUIRED)

# caliper_net's sources (as the root CMakeLists.txt lists them; upgrade.cpp holds
# the upgrade checks in later trees).
set(NET_SOURCES sha1.cpp ws.cpp json.cpp wire.cpp wire_controls.cpp wire_scene.cpp wire_timelines.cpp wire_programs.cpp upgrade.cpp)
list(TRANSFORM NET_SOURCES PREPEND ${REPO}/src/host/net/)
foreach(f ${NET_SOURCES})
    if(NOT EXISTS ${f})
        list(REMOVE_ITEM NET_SOURCES ${f})
    endif()
endforeach()
add_library(netio_net STATIC ${NET_SOURCES})
target_include_directories(netio_net PUBLIC ${REPO}/src/host ${REPO}/include)
if(WIN32)
    target_link_libraries(netio_net PUBLIC ws2_32)
endif()

nanobind_add_module(_netio NB_STATIC ${REPO}/python/src/_netio.cpp)
target_link_libraries(_netio PRIVATE netio_net)
if(APPLE)
    target_link_options(_netio PRIVATE "LINKER:-exported_symbol,_PyInit__netio")
endif()
set_target_properties(_netio PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/out)

# The caliper_net tests (doctest, declared in cmake/Dependencies.cmake) for an
# AddressSanitizer run.
if(NETIO_ASAN AND EXISTS ${REPO}/tests/test_net_wire.cpp)
    if(NOT COMMAND caliper_fetch_doctest)
        set(CALIPER_DEPENDENCIES_DECLARATIONS_ONLY ON)
        include(${REPO}/cmake/Dependencies.cmake)
        unset(CALIPER_DEPENDENCIES_DECLARATIONS_ONLY)
    endif()
    caliper_fetch_doctest()
    add_executable(netio_net_tests ${REPO}/tests/test_main.cpp ${REPO}/tests/test_net_wire.cpp
                   ${REPO}/tests/test_net_ws.cpp ${REPO}/tests/test_net_wire_32.cpp)
    target_compile_definitions(netio_net_tests PRIVATE CALIPER_TEST_SOURCE_ROOT="${REPO}")
    target_link_libraries(netio_net_tests PRIVATE netio_net doctest::doctest)
endif()
