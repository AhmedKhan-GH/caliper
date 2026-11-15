# TwinScope (id dev.caliper.twin-scope). It shares export_affordance.h with
# MeshScope through the sibling folder ../common.
# On its own it builds against an installed SDK:
#   cmake -S . -B build -DCMAKE_PREFIX_PATH=<sdk prefix>
#   cmake --build build
if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    cmake_minimum_required(VERSION 3.18)
    project(twin_scope LANGUAGES CXX)
    find_package(caliper-sdk 0.2 CONFIG REQUIRED)
endif()

caliper_add_applet(twin_scope
    SOURCES plugin.cpp twin_scope.cpp
    MANIFEST twin_scope.caliper.toml
    TORCH)
target_include_directories(twin_scope PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/../common)
# TWIN_SCOPE_ASSET_PATH is the source-tree development fallback: the applet
# reads the copy staged beside its library first
# (<library dir>/assets/twin_scope/housing.obj, staged below) and falls back to
# this path only when the staged copy is missing. The staging destination must
# match what the code reads.
target_compile_definitions(twin_scope PRIVATE
    TWIN_SCOPE_ASSET_PATH="${CMAKE_CURRENT_SOURCE_DIR}/assets/housing.obj")
add_custom_command(TARGET twin_scope POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E make_directory
        "$<TARGET_FILE_DIR:twin_scope>/assets/twin_scope"
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${CMAKE_CURRENT_SOURCE_DIR}/assets/housing.obj"
        "$<TARGET_FILE_DIR:twin_scope>/assets/twin_scope/housing.obj"
    VERBATIM)
