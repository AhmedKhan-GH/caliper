# caliper_builder_loaders: the tensor type and transforms, the tensor and
# tabular readers, the loader recipe and the workspace's loader store
# (docs/plans/loaders.md).
if(NOT CALIPER_BUILDER_HAVE_DUCKDB)
    message(STATUS "caliper_builder: the loaders module needs DuckDB and is skipped")
    return()
endif()

add_library(caliper_builder_loaders STATIC
    tensor.cpp
    tensor_ops.cpp)
target_include_directories(caliper_builder_loaders PUBLIC "${CALIPER_BUILDER_INCLUDE_ROOT}")
target_link_libraries(caliper_builder_loaders PUBLIC nlohmann_json::nlohmann_json)
target_compile_features(caliper_builder_loaders PUBLIC cxx_std_17)

caliper_builder_add_tests(loaders
    SOURCES
        tests/test_tensor_ops.cpp
    LIBS caliper_builder_loaders)
