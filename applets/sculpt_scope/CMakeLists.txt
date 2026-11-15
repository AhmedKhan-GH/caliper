# SculptScope — a net sculpts a point cloud, rendered zero-copy from its own
# forward output (id dev.caliper.sculpt-scope). The camera math is hand-rolled
# and the fallback is ImPlot3D — no dependencies beyond the SDK and libtorch.
# On its own it builds against an installed SDK:
#   cmake -S . -B build -DCMAKE_PREFIX_PATH=<sdk prefix>
#   cmake --build build
if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    cmake_minimum_required(VERSION 3.18)
    project(sculpt_scope LANGUAGES CXX)
    find_package(caliper-sdk 0.2 CONFIG REQUIRED)
endif()

caliper_add_applet(sculpt_scope
    SOURCES plugin.cpp sculpt_scope.cpp
    MANIFEST sculpt_scope.caliper.toml
    TORCH)
target_include_directories(sculpt_scope PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
