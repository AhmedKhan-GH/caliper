# The builder libraries of sprint 3: the spec format, the catalog, the loaders,
# the command layer, the session that binds them, caliper-mcp, the ACP client
# and the caliper-agent sidecar (design/builder-overview.md).
#
# This folder configures in two ways:
#   - on its own, as a small project that needs no host build:
#       cmake -S src/builder -B <build> -G Ninja -DCALIPER_BUILDER_DUCKDB_DIR=<dir>
#     where <dir> holds duckdb.hpp and the libduckdb shared library of a DuckDB
#     release (the version the host pins, v1.5.6);
#   - from the root CMakeLists.txt, which adds it with add_subdirectory(src/builder)
#     when CALIPER_WITH_BUILDER is ON (the default, except on Windows); there it
#     uses the host's toml++, doctest, BUILD_TESTS and statically built DuckDB,
#     and its ctest entries sit beside the host's.
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
    # The host's DuckDB with every extension it links statically, json among
    # them (CALIPER_DUCKDB_LIBS in cmake/Dependencies.cmake).
    target_link_libraries(caliper_builder_duckdb INTERFACE ${CALIPER_DUCKDB_LIBS})
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

# The sidecar's offline switch for every test of this tree, and the entry
# that checks it (cmake/CaliperOfflineTests.cmake).
include(${CMAKE_CURRENT_LIST_DIR}/cmake/CaliperOfflineTests.cmake)
if(BUILD_TESTS)
    caliper_offline_tests()
    caliper_offline_check(builder_offline_switch)
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

# One Python interpreter for the Python loader, session scope-parity and
# caliper-agent tests across src/builder/ and src/host/builder/. When empty,
# auto-select a local venv (or CALIPER_ACCEPTANCE_PYTHON) that already has the
# builder and agent test dependencies. When set explicitly, verify those
# dependencies at configure time so test_context_bundle and builder_agent_py
# cannot fail later with a missing pydantic_ai import.
set(CALIPER_BUILDER_TEST_PYTHON "" CACHE FILEPATH
    "A Python interpreter with numpy, pyarrow, and caliper-agent[test] dependencies for the builder tests")
if(BUILD_TESTS)
    function(_caliper_probe_builder_python py_exe out_ok out_err)
        execute_process(
            COMMAND "${py_exe}" -I -c "import numpy, pydantic_ai, anthropic, openai, pytest, acp.schema"
            RESULT_VARIABLE _rc
            OUTPUT_QUIET
            ERROR_VARIABLE _err
            ERROR_STRIP_TRAILING_WHITESPACE)
        if(_rc EQUAL 0)
            set(${out_ok} TRUE PARENT_SCOPE)
            set(${out_err} "" PARENT_SCOPE)
        else()
            set(${out_ok} FALSE PARENT_SCOPE)
            set(${out_err} "${_err}" PARENT_SCOPE)
        endif()
    endfunction()

    if(NOT CALIPER_BUILDER_TEST_PYTHON)
        if(WIN32)
            set(_venv_py "Scripts/python.exe")
        else()
            set(_venv_py "bin/python")
        endif()
        set(_builder_py_candidates
            "${CMAKE_CURRENT_SOURCE_DIR}/agent/.venv/${_venv_py}"
            "${CMAKE_SOURCE_DIR}/.venv/${_venv_py}")
        if(CALIPER_ACCEPTANCE_PYTHON)
            list(APPEND _builder_py_candidates "${CALIPER_ACCEPTANCE_PYTHON}")
        endif()
        foreach(_cand IN LISTS _builder_py_candidates)
            if(EXISTS "${_cand}")
                _caliper_probe_builder_python("${_cand}" _cand_ok _cand_err)
                if(_cand_ok)
                    set(CALIPER_BUILDER_TEST_PYTHON "${_cand}" CACHE FILEPATH
                        "A Python interpreter with numpy, pyarrow, and caliper-agent[test] dependencies for the builder tests"
                        FORCE)
                    message(STATUS "caliper_builder: auto-selected CALIPER_BUILDER_TEST_PYTHON=${CALIPER_BUILDER_TEST_PYTHON}")
                    break()
                endif()
            endif()
        endforeach()
        if(NOT CALIPER_BUILDER_TEST_PYTHON)
            message(STATUS
                "caliper_builder: CALIPER_BUILDER_TEST_PYTHON is not set and no local venv with "
                "caliper-agent[test] dependencies was found; Python loader and agent tests will skip")
        endif()
    else()
        if(NOT IS_ABSOLUTE "${CALIPER_BUILDER_TEST_PYTHON}")
            get_filename_component(_builder_py_abs "${CALIPER_BUILDER_TEST_PYTHON}" ABSOLUTE BASE_DIR "${CMAKE_SOURCE_DIR}")
            if(EXISTS "${_builder_py_abs}")
                set(CALIPER_BUILDER_TEST_PYTHON "${_builder_py_abs}" CACHE FILEPATH
                    "A Python interpreter with numpy, pyarrow, and caliper-agent[test] dependencies for the builder tests"
                    FORCE)
            endif()
        endif()
        if(NOT EXISTS "${CALIPER_BUILDER_TEST_PYTHON}")
            message(FATAL_ERROR
                "caliper_builder: CALIPER_BUILDER_TEST_PYTHON does not exist: ${CALIPER_BUILDER_TEST_PYTHON}")
        endif()
        _caliper_probe_builder_python("${CALIPER_BUILDER_TEST_PYTHON}" _builder_py_ok _builder_py_err)
        if(NOT _builder_py_ok)
            message(FATAL_ERROR
                "caliper_builder: CALIPER_BUILDER_TEST_PYTHON (${CALIPER_BUILDER_TEST_PYTHON}) "
                "lacks required builder/agent test dependencies "
                "(numpy, pydantic_ai, anthropic, openai, pytest, acp.schema): ${_builder_py_err}\n"
                "Install them into that interpreter:\n"
                "  uv pip install --python \"${CALIPER_BUILDER_TEST_PYTHON}\" "
                "-e \"${CMAKE_CURRENT_SOURCE_DIR}/agent[test]\" numpy pyarrow")
        endif()
    endif()
    if(CALIPER_BUILDER_TEST_PYTHON)
        execute_process(
            COMMAND "${CALIPER_BUILDER_TEST_PYTHON}" -I -c "import pyarrow.parquet"
            RESULT_VARIABLE _pyarrow_rc
            OUTPUT_QUIET ERROR_QUIET)
        if(NOT _pyarrow_rc EQUAL 0)
            message(STATUS
                "caliper_builder: ${CALIPER_BUILDER_TEST_PYTHON} lacks pyarrow; "
                "Parquet cases in the Python loader tests will skip")
        endif()
    endif()
endif()

foreach(_module spec programs text operators catalog commands loaders acp session mcp agent)
    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/${_module}/CMakeLists.txt")
        add_subdirectory(${_module})
    endif()
endforeach()
