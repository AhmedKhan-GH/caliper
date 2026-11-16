# caliper_builder_catalog: the catalog entry and its JSON form, the provider
# interface, the merging Catalog, an in-memory provider and the tensor sampler
# (docs/design/catalog.md).
add_library(caliper_builder_catalog STATIC
    catalog_entry.cpp
    catalog.cpp
    memory_provider.cpp
    tensor_sample.cpp
    text_util.cpp)
target_include_directories(caliper_builder_catalog PUBLIC "${CALIPER_BUILDER_INCLUDE_ROOT}")
target_link_libraries(caliper_builder_catalog PUBLIC nlohmann_json::nlohmann_json)
target_compile_features(caliper_builder_catalog PUBLIC cxx_std_17)

caliper_builder_add_tests(catalog
    SOURCES
        tests/test_catalog_entry.cpp
        tests/test_tensor_sample.cpp
        tests/test_catalog.cpp
    LIBS caliper_builder_catalog)
