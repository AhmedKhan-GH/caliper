# PulseScope — the caliper.feed.v1 dashboard exemplar. Deliberately CPU and
# ImPlot only: no libtorch, no bridge or geometry. It links just the SDK sugar
# and the UI stack (the hello shape), so it builds and loads host-neutrally — on
# a host with no telemetry provider it degrades to one honest line.
# On its own it builds against an installed SDK:
#   cmake -S . -B build -DCMAKE_PREFIX_PATH=<sdk prefix>
#   cmake --build build
if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    cmake_minimum_required(VERSION 3.18)
    project(pulse_scope LANGUAGES CXX)
    find_package(caliper-sdk 0.2 CONFIG REQUIRED)
endif()

caliper_add_applet(pulse_scope
    SOURCES plugin.cpp pulse_scope.cpp
    MANIFEST pulse_scope.caliper.toml)
target_include_directories(pulse_scope PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
