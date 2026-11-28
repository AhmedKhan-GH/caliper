# The hello_world applet. On its own it builds against an installed SDK:
#   cmake -S . -B build -DCMAKE_PREFIX_PATH=<sdk prefix>
#   cmake --build build
# Inside the core build it is the loader-test fixture.
if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    cmake_minimum_required(VERSION 3.18)
    project(hello_world LANGUAGES CXX)
    find_package(caliper-sdk 0.2 CONFIG REQUIRED)
endif()

caliper_add_applet(hello_world_applet
    SOURCES hello_world.cpp
    MANIFEST hello_world.caliper.toml
    OUTPUT_NAME hello_world)
