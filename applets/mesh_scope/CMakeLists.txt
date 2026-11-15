# MeshScope (id dev.caliper.mesh-scope). It shares export_affordance.h with
# TwinScope through the sibling folder ../common.
# On its own it builds against an installed SDK:
#   cmake -S . -B build -DCMAKE_PREFIX_PATH=<sdk prefix>
#   cmake --build build
if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    cmake_minimum_required(VERSION 3.18)
    project(mesh_scope LANGUAGES CXX)
    find_package(caliper-sdk 0.2 CONFIG REQUIRED)
endif()

caliper_add_applet(mesh_scope
    SOURCES plugin.cpp mesh_scope.cpp
    MANIFEST mesh_scope.caliper.toml
    TORCH)
target_include_directories(mesh_scope PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/../common)
