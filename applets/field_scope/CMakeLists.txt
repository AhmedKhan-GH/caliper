# FieldScope — charged particles in an EM field (id dev.caliper.field-scope).
# No dependencies beyond the SDK and libtorch — the camera math is hand-rolled
# and the fallback is ImPlot3D.
# On its own it builds against an installed SDK:
#   cmake -S . -B build -DCMAKE_PREFIX_PATH=<sdk prefix>
#   cmake --build build
if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    cmake_minimum_required(VERSION 3.18)
    project(field_scope LANGUAGES CXX)
    find_package(caliper-sdk 0.2 CONFIG REQUIRED)
endif()

caliper_add_applet(field_scope
    SOURCES plugin.cpp field_scope.cpp
    MANIFEST field_scope.caliper.toml
    TORCH)
target_include_directories(field_scope PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
