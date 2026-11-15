# SineScope, the ML tutorial applet. On its own it builds against an installed
# SDK that was built with libtorch:
#   cmake -S . -B build -DCMAKE_PREFIX_PATH=<sdk prefix>
#   cmake --build build
if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    cmake_minimum_required(VERSION 3.18)
    project(sine_scope LANGUAGES CXX)
    find_package(caliper-sdk 0.2 CONFIG REQUIRED)
endif()

caliper_add_applet(sine_scope
    SOURCES sine_scope.cpp
    MANIFEST sine_scope.caliper.toml
    TORCH)
