# The host's catalog providers for the builder (design/catalog.md,
# design/builder-session.md): the live views of the attached session or stream,
# the channels of caliper.feed.v1 and the metrics of caliper.metrics.v1 as
# catalog sources, and their data for the session's resolve. Added by the root
# CMakeLists.txt only with CALIPER_WITH_BUILDER, after src/builder.
if(NOT TARGET caliper_builder_session)
    message(STATUS "catalog providers: the builder's session module is not built; skipped")
    return()
endif()

add_library(caliper_catalog_providers STATIC
    catalog_providers.cpp
    catalog_util.cpp
    feed_catalog.cpp
    live_catalog.cpp
    metrics_catalog.cpp)
target_link_libraries(caliper_catalog_providers PUBLIC libcaliper caliper_builder_session)
set_target_properties(caliper_catalog_providers PROPERTIES
    CXX_STANDARD 17 CXX_STANDARD_REQUIRED ON POSITION_INDEPENDENT_CODE ON)
if(NOT MSVC)
    target_compile_options(caliper_catalog_providers PRIVATE -Wall -Wextra)
endif()

if(BUILD_TESTS)
    # Headless; the case that reads a real core's live views needs an offscreen
    # canvas and is counted as skipped without one (gpu_skip.h, exit 77).
    add_executable(caliper_catalog_providers_tests tests/test_catalog_providers.cpp tests/test_live_catalog_steps.cpp)
    target_include_directories(caliper_catalog_providers_tests PRIVATE ${CMAKE_SOURCE_DIR}/tests)
    target_link_libraries(caliper_catalog_providers_tests PRIVATE caliper_catalog_providers doctest::doctest
        caliper_test_run_marker)
    set_target_properties(caliper_catalog_providers_tests PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED ON)
    add_test(NAME caliper_catalog_providers_tests COMMAND caliper_catalog_providers_tests)
    # The embed core's stores go to a work folder, never the user's app-data.
    set_tests_properties(caliper_catalog_providers_tests PROPERTIES SKIP_RETURN_CODE 77 ENVIRONMENT
        "CALIPER_DATA_DIR=${CMAKE_CURRENT_BINARY_DIR}/app-data/caliper_catalog_providers_tests")
endif()

# What a live panel costs per snapshot and per frame (bench/bench_live_entropy.cpp):
# built on demand, never run by ctest.
add_executable(caliper_catalog_providers_bench EXCLUDE_FROM_ALL bench/bench_live_entropy.cpp)
target_link_libraries(caliper_catalog_providers_bench PRIVATE caliper_catalog_providers)
set_target_properties(caliper_catalog_providers_bench PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED ON)
