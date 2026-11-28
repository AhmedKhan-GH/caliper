# A consumer of the installed caliper-sdk package that uses nothing but
# find_package: it compiles the sugar and the UI stack and runs a doctest
# binary on the fixture host. run.cmake builds it from a copy outside the
# source tree.
cmake_minimum_required(VERSION 3.18)
project(sdk_install_probe LANGUAGES CXX)
find_package(caliper-sdk 0.3 CONFIG REQUIRED)

foreach(var CALIPER_SDK_VERSION CALIPER_SDK_ABI_EPOCH CALIPER_UI_STACK_HASH)
    if(NOT ${var})
        message(FATAL_ERROR "The installed caliper-sdk package does not set ${var}.")
    endif()
endforeach()
if(NOT CALIPER_UI_STACK_HASH MATCHES "^[0-9a-f]+$")
    message(FATAL_ERROR "CALIPER_UI_STACK_HASH is not a hex string: ${CALIPER_UI_STACK_HASH}")
endif()

add_executable(probe probe.cpp)
target_link_libraries(probe PRIVATE caliper::sdk caliper::ui_stack caliper::sdk_testing)
target_compile_definitions(probe PRIVATE
    PROBE_ABI_EPOCH=${CALIPER_SDK_ABI_EPOCH})
set_target_properties(probe PROPERTIES CXX_STANDARD 20 CXX_STANDARD_REQUIRED ON)
