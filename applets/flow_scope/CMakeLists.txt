# FlowScope — the caliper.geometry.v1 exemplar (id dev.caliper.flow-scope).
# No dependencies beyond the SDK and libtorch — the camera math is hand-rolled
# and the fallback is ImPlot3D.
# On its own it builds against an installed SDK:
#   cmake -S . -B build -DCMAKE_PREFIX_PATH=<sdk prefix>
#   cmake --build build
if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    cmake_minimum_required(VERSION 3.18)
    project(flow_scope LANGUAGES CXX)
    find_package(caliper-sdk 0.2 CONFIG REQUIRED)
endif()

caliper_add_applet(flow_scope
    SOURCES plugin.cpp flow_scope.cpp
    MANIFEST flow_scope.caliper.toml
    TORCH)
target_include_directories(flow_scope PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
