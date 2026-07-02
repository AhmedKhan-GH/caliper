# doctest v2.4.11 declares cmake_minimum_required 2.8, which CMake >= 4.0
# rejects; this floor lets FetchContent configure it without a manual flag.
set(CMAKE_POLICY_VERSION_MINIMUM 3.5)

include(FetchContent)
FetchContent_Declare(doctest
    GIT_REPOSITORY https://github.com/doctest/doctest.git
    GIT_TAG v2.4.11 GIT_SHALLOW TRUE)
FetchContent_MakeAvailable(doctest)

add_executable(caliper_tests
    test_main.cpp
    test_abi.cpp
    abi_c_check.c
    test_manifest.cpp
    test_negotiation.cpp
    test_crash_guard.cpp
    test_watchdog.cpp
    test_sugar.cpp
    test_jobs.cpp
    test_device.cpp
    test_sugar_services.cpp
    test_mnist_idx.cpp
)
find_package(ZLIB REQUIRED)
target_link_libraries(caliper_tests PRIVATE doctest::doctest caliper::sdk caliper_host_lib
    caliper::sdk_testing caliper::ui_stack ZLIB::ZLIB)
target_include_directories(caliper_tests PRIVATE ${CMAKE_SOURCE_DIR}/examples/ml_scope)
set_target_properties(caliper_tests PROPERTIES CXX_STANDARD 20 CXX_STANDARD_REQUIRED ON)

target_sources(caliper_tests PRIVATE test_loader.cpp)
add_dependencies(caliper_tests hello_applet)
target_compile_definitions(caliper_tests PRIVATE
    CALIPER_TEST_APPLETS_DIR="${CMAKE_BINARY_DIR}/applets"
    CALIPER_TEST_DATA_ROOT="${CMAKE_BINARY_DIR}/test-data")

add_test(NAME caliper_tests COMMAND caliper_tests)
