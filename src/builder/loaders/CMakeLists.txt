# caliper_builder_loaders: the tensor type and transforms, the tensor and
# tabular readers, the loader recipe and the workspace's loader store
# (docs/plans/loaders.md).
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
    store.cpp)
target_include_directories(caliper_builder_loaders PUBLIC "${CALIPER_BUILDER_INCLUDE_ROOT}")
# stb_image is vendored in the tree; image.cpp compiles it with static linkage.
target_include_directories(caliper_builder_loaders SYSTEM PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/../../../third_party/stb")
target_link_libraries(caliper_builder_loaders
    PUBLIC nlohmann_json::nlohmann_json
    PRIVATE caliper_builder_duckdb tomlplusplus::tomlplusplus)
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
