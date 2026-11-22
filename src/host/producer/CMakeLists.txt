# caliper_producer: the producer runtime's core (caliper_producer.h,
# design/gpu-programs.md and design/producer-runtime.md), which the native
# module caliper._producer links: the graphics layer without a window, the
# runtime WGSL compiler and the sandbox of caliper_builder_programs.
add_library(caliper_producer STATIC producer.cpp)
target_include_directories(caliper_producer PUBLIC ${CMAKE_SOURCE_DIR}/src/host)
set_target_properties(caliper_producer PROPERTIES
    CXX_STANDARD 17 CXX_STANDARD_REQUIRED ON POSITION_INDEPENDENT_CODE ON)
if(NOT MSVC)
    target_compile_options(caliper_producer PRIVATE -Wall -Wextra)
endif()

if(BUILD_TESTS AND TARGET doctest::doctest)
    add_executable(caliper_producer_tests tests/test_producer.cpp)
    target_link_libraries(caliper_producer_tests PRIVATE caliper_producer doctest::doctest)
    set_target_properties(caliper_producer_tests PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED ON)
    add_test(NAME caliper_producer_tests COMMAND caliper_producer_tests)
endif()
