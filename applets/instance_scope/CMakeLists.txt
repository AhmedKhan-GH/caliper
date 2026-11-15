# InstanceScope (id dev.caliper.instance-scope).
# On its own it builds against an installed SDK:
#   cmake -S . -B build -DCMAKE_PREFIX_PATH=<sdk prefix>
#   cmake --build build
if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    cmake_minimum_required(VERSION 3.18)
    project(instance_scope LANGUAGES CXX)
    find_package(caliper-sdk 0.2 CONFIG REQUIRED)
endif()

caliper_add_applet(instance_scope
    SOURCES plugin.cpp instance_scope.cpp
    MANIFEST instance_scope.caliper.toml
    TORCH)
target_include_directories(instance_scope PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
