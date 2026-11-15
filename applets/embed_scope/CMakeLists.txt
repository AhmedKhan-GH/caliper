# EmbedScope — the 3D embedding projector exemplar. It exercises the whole
# platform surface: jobs/device/metrics/tensor_bridge plus artifacts.v1 and
# data.v1. curl + zlib are macOS system libs covering the cancellable MNIST
# download + gunzip inside the job; the MNIST IDX reader is its own mnist_idx.h
# (self-contained — the exemplar doubles as a template). The host links none of
# these: data acquisition is applet business.
# On its own it builds against an installed SDK:
#   cmake -S . -B build -DCMAKE_PREFIX_PATH=<sdk prefix>
#   cmake --build build
if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    cmake_minimum_required(VERSION 3.18)
    project(embed_scope LANGUAGES CXX)
    find_package(caliper-sdk 0.2 CONFIG REQUIRED)
endif()

# Inside the core build on Windows, the host build provides both targets
# in-tree (cmake/Dependencies.cmake).
if(NOT TARGET CURL::libcurl)
    find_package(CURL REQUIRED)
endif()
if(NOT TARGET ZLIB::ZLIB)
    find_package(ZLIB REQUIRED)
endif()

caliper_add_applet(embed_scope
    SOURCES plugin.cpp embed_scope.cpp
    MANIFEST embed_scope.caliper.toml
    TORCH)
target_include_directories(embed_scope PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR})   # mnist_idx.h is local — the exemplar is self-contained
target_link_libraries(embed_scope PRIVATE CURL::libcurl ZLIB::ZLIB)
