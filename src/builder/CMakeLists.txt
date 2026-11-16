# The builder libraries of sprint 3: the spec format, the catalog, the loaders,
# the command layer, the session that binds them, caliper-mcp, the ACP client
# and the caliper-agent sidecar (design/builder-overview.md).
#
# This folder configures in two ways:
#   - on its own, as a small project that needs no host build:
#       cmake -S src/builder -B <build> -G Ninja -DCALIPER_BUILDER_DUCKDB_DIR=<dir>
#     where <dir> holds duckdb.hpp and the libduckdb shared library of a DuckDB
#     release (the version the host pins, v1.5.6);
#   - from the root CMakeLists.txt with add_subdirectory(src/builder), where it
#     uses the host's toml++ and its statically built DuckDB.
# Each module is a subfolder with its own CMakeLists.txt, added when present.
cmake_minimum_required(VERSION 3.18...4.2)

if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    project(caliper_builder CXX C)
    set(CALIPER_BUILDER_STANDALONE ON)
    set(CMAKE_CXX_STANDARD 17)
    set(CMAKE_CXX_STANDARD_REQUIRED ON)
    set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
    option(BUILD_TESTS "Build tests" ON)
    if(BUILD_TESTS)
        enable_testing()
    endif()
else()
    set(CALIPER_BUILDER_STANDALONE OFF)
endif()

# doctest v2.4.11 declares cmake_minimum_required 2.8, which CMake >= 4.0 rejects.
set(CMAKE_POLICY_VERSION_MINIMUM 3.5)
include(FetchContent)

if(NOT TARGET tomlplusplus::tomlplusplus)
    FetchContent_Declare(tomlplusplus
        GIT_REPOSITORY https://github.com/marzer/tomlplusplus.git
        GIT_TAG v3.4.0 GIT_SHALLOW TRUE)
    FetchContent_MakeAvailable(tomlplusplus)
endif()

# nlohmann/json (MIT) carries every JSON value of the builder: the JSON form of
# a spec, command arguments, the journal, MCP and ACP messages.
if(NOT TARGET nlohmann_json::nlohmann_json)
    set(JSON_BuildTests OFF CACHE INTERNAL "")
    FetchContent_Declare(nlohmann_json
        URL https://github.com/nlohmann/json/releases/download/v3.11.3/json.tar.xz
        URL_HASH SHA256=d6c65aca6b1ed68e7a182f4757257b107ae403032760ed6ef121c9d55e81757d)
    FetchContent_MakeAvailable(nlohmann_json)
endif()

# DuckDB, as one target the modules link: caliper_builder_duckdb.
add_library(caliper_builder_duckdb INTERFACE)
if(TARGET duckdb_static)
    target_link_libraries(caliper_builder_duckdb INTERFACE
        duckdb_static duckdb_generated_extension_loader parquet_extension core_functions_extension)
    set(CALIPER_BUILDER_HAVE_DUCKDB ON)
elseif(CALIPER_BUILDER_DUCKDB_DIR)
    find_library(CALIPER_BUILDER_DUCKDB_LIB duckdb
        PATHS "${CALIPER_BUILDER_DUCKDB_DIR}" NO_DEFAULT_PATH REQUIRED)
    target_include_directories(caliper_builder_duckdb SYSTEM INTERFACE "${CALIPER_BUILDER_DUCKDB_DIR}")
    target_link_libraries(caliper_builder_duckdb INTERFACE "${CALIPER_BUILDER_DUCKDB_LIB}")
    set(CMAKE_BUILD_RPATH "${CALIPER_BUILDER_DUCKDB_DIR}")
    set(CALIPER_BUILDER_HAVE_DUCKDB ON)
else()
    message(STATUS "caliper_builder: no DuckDB given (CALIPER_BUILDER_DUCKDB_DIR); the modules that need it are skipped")
    set(CALIPER_BUILDER_HAVE_DUCKDB OFF)
endif()

# One doctest main shared by every module's test executable. A module adds
#   caliper_builder_add_tests(<module> SOURCES <files> LIBS <targets>)
# and gets the ctest test builder_<module>.
if(BUILD_TESTS)
    if(NOT TARGET doctest::doctest)
        FetchContent_Declare(doctest
            GIT_REPOSITORY https://github.com/doctest/doctest.git
            GIT_TAG v2.4.11 GIT_SHALLOW TRUE)
        FetchContent_MakeAvailable(doctest)
    endif()
    add_library(caliper_builder_test_main OBJECT tests/test_main.cpp)
    target_link_libraries(caliper_builder_test_main PUBLIC doctest::doctest)
endif()

function(caliper_builder_add_tests module)
    if(NOT BUILD_TESTS)
        return()
    endif()
    cmake_parse_arguments(ARG "" "" "SOURCES;LIBS" ${ARGN})
    add_executable(caliper_builder_${module}_tests ${ARG_SOURCES})
    target_link_libraries(caliper_builder_${module}_tests PRIVATE
        caliper_builder_test_main doctest::doctest ${ARG_LIBS})
    target_compile_definitions(caliper_builder_${module}_tests PRIVATE
        CALIPER_BUILDER_SOURCE_DIR="${CMAKE_CURRENT_FUNCTION_LIST_DIR}"
        CALIPER_BUILDER_TEST_TMP="${CMAKE_CURRENT_BINARY_DIR}/test-tmp")
    add_test(NAME builder_${module} COMMAND caliper_builder_${module}_tests)
endfunction()

# Public headers are included as <caliper_builder/<module>/<header>.h>.
set(CALIPER_BUILDER_INCLUDE_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/include")

foreach(_module spec catalog commands loaders acp session mcp agent)
    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/${_module}/CMakeLists.txt")
        add_subdirectory(${_module})
    endif()
endforeach()
