include(FetchContent)
FetchContent_Declare(doctest
    GIT_REPOSITORY https://github.com/doctest/doctest.git
    GIT_TAG v2.4.11 GIT_SHALLOW TRUE)
FetchContent_MakeAvailable(doctest)

add_executable(caliper_tests
    test_main.cpp
    test_abi.cpp
    abi_c_check.c
)
target_link_libraries(caliper_tests PRIVATE doctest::doctest caliper::sdk)
set_target_properties(caliper_tests PROPERTIES CXX_STANDARD 20 CXX_STANDARD_REQUIRED ON)
add_test(NAME caliper_tests COMMAND caliper_tests)
