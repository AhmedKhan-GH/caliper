# caliper_builder_loaders: the tensor type and transforms, the tensor and
# tabular readers, the loader recipe and the workspace's loader store
# (design/loaders.md).
if(NOT CALIPER_BUILDER_HAVE_DUCKDB)
    message(STATUS "caliper_builder: the loaders module needs DuckDB and is skipped")
    return()
endif()

add_library(caliper_builder_loaders STATIC
    tensor.cpp
    tensor_ops.cpp
    file_io.cpp
    npy.cpp
    zip_reader.cpp
    npz.cpp
    safetensors.cpp
    image.cpp
    wav.cpp
    hdf5.cpp
    readers.cpp
    pickle.cpp
    db.cpp
    tabular.cpp
    loader.cpp
    paths.cpp
    inference.cpp
    python_loader.cpp
    store.cpp)
target_include_directories(caliper_builder_loaders PUBLIC "${CALIPER_BUILDER_INCLUDE_ROOT}")

# The Python loader's host script is embedded in the library, so that the app
# can write it next to a run and start it by path without Caliper LIB.
set(CALIPER_BUILDER_LOADER_HOST_PY "${CMAKE_CURRENT_SOURCE_DIR}/../../../python/caliper/loader_host.py")
set(_loader_host_inc "${CMAKE_CURRENT_BINARY_DIR}/generated/loader_host_py.inc")
add_custom_command(
    OUTPUT "${_loader_host_inc}"
    COMMAND "${CMAKE_COMMAND}" "-DINPUT=${CALIPER_BUILDER_LOADER_HOST_PY}" "-DOUTPUT=${_loader_host_inc}"
            -P "${CMAKE_CURRENT_SOURCE_DIR}/embed_file.cmake"
    DEPENDS "${CALIPER_BUILDER_LOADER_HOST_PY}" "${CMAKE_CURRENT_SOURCE_DIR}/embed_file.cmake"
    COMMENT "Embedding loader_host.py"
    VERBATIM)
target_sources(caliper_builder_loaders PRIVATE "${_loader_host_inc}")
target_include_directories(caliper_builder_loaders PRIVATE "${CMAKE_CURRENT_BINARY_DIR}/generated")

# stb_image is vendored in the tree; image.cpp compiles it with static linkage.
target_include_directories(caliper_builder_loaders SYSTEM PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/../../../third_party/stb")
# The Python loader starts its process with the acp module's subprocess helper.
target_link_libraries(caliper_builder_loaders
    PUBLIC nlohmann_json::nlohmann_json
    PRIVATE caliper_builder_duckdb tomlplusplus::tomlplusplus caliper_builder_acp)
target_compile_features(caliper_builder_loaders PUBLIC cxx_std_17)

# zlib inflates deflated npz members. Without it, stored members still load and
# deflated ones are refused with a clear error.
find_package(ZLIB)
if(ZLIB_FOUND)
    target_link_libraries(caliper_builder_loaders PRIVATE ZLIB::ZLIB)
    target_compile_definitions(caliper_builder_loaders PRIVATE CALIPER_BUILDER_HAVE_ZLIB=1)
else()
    message(STATUS "caliper_builder: no zlib; the npz reader refuses deflated members")
endif()

set(_loaders_test_libs caliper_builder_loaders)
if(ZLIB_FOUND)
    list(APPEND _loaders_test_libs ZLIB::ZLIB)
endif()
caliper_builder_add_tests(loaders
    SOURCES
        tests/test_tensor_ops.cpp
        tests/test_npy_npz.cpp
        tests/test_safetensors.cpp
        tests/test_image_wav.cpp
        tests/test_pickle.cpp
        tests/test_tabular.cpp
        tests/test_loader.cpp
        tests/test_store.cpp
    LIBS ${_loaders_test_libs})
if(TARGET caliper_builder_loaders_tests)
    target_include_directories(caliper_builder_loaders_tests SYSTEM PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/../../../third_party/stb")
    if(ZLIB_FOUND)
        target_compile_definitions(caliper_builder_loaders_tests PRIVATE CALIPER_BUILDER_HAVE_ZLIB=1)
    endif()
endif()

# The Python loader. CALIPER_BUILDER_TEST_PYTHON (defined and validated in
# src/builder/CMakeLists.txt) names an interpreter with numpy, pyarrow and the
# caliper-agent test dependencies. When it is empty, the tests that run Python
# print that they are skipped and the rest still run; loader_host.py's own tests
# are reported as skipped.
if(BUILD_TESTS AND NOT WIN32)
    caliper_builder_add_tests(loaders_python
        SOURCES tests/test_python_loader.cpp
        LIBS caliper_builder_loaders caliper_builder_acp)
    target_compile_definitions(caliper_builder_loaders_python_tests PRIVATE
        CALIPER_BUILDER_TEST_PYTHON="${CALIPER_BUILDER_TEST_PYTHON}"
        CALIPER_BUILDER_LOADER_HOST_PY="${CALIPER_BUILDER_LOADER_HOST_PY}"
        CALIPER_BUILDER_PYTHON_FIXTURES="${CMAKE_CURRENT_SOURCE_DIR}/tests/fixtures/python")
    if(CALIPER_BUILDER_TEST_PYTHON)
        add_test(NAME builder_loaders_python_host
            COMMAND "${CALIPER_BUILDER_TEST_PYTHON}" -I "${CMAKE_CURRENT_SOURCE_DIR}/tests/test_loader_host.py"
                    "${CALIPER_BUILDER_LOADER_HOST_PY}" "${CMAKE_CURRENT_BINARY_DIR}/test-tmp/loader_host")
    else()
        message(STATUS "caliper_builder: CALIPER_BUILDER_TEST_PYTHON is not set; the Python loader tests that run Python are skipped")
        add_test(NAME builder_loaders_python_host
            COMMAND "${CMAKE_COMMAND}" -E echo
                    "Skipped: CALIPER_BUILDER_TEST_PYTHON is not set, so loader_host.py is not tested.")
        set_tests_properties(builder_loaders_python_host PROPERTIES
            SKIP_REGULAR_EXPRESSION "Skipped: CALIPER_BUILDER_TEST_PYTHON")
    endif()
endif()
