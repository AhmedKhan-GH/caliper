# GPTScope 2 — mechanistic-interpretability rebuild (id dev.caliper.gpt-scope).
# curl is a macOS system lib covering the cancellable TinyShakespeare download
# inside the job — no zlib: the corpus is plain text. The host links neither:
# data acquisition is applet business.
# On its own it builds against an installed SDK:
#   cmake -S . -B build -DCMAKE_PREFIX_PATH=<sdk prefix>
#   cmake --build build
if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    cmake_minimum_required(VERSION 3.18)
    project(gpt_scope LANGUAGES CXX)
    find_package(caliper-sdk 0.2 CONFIG REQUIRED)
endif()

# Inside the core build on Windows, the host build provides CURL::libcurl
# in-tree (cmake/Dependencies.cmake).
if(NOT TARGET CURL::libcurl)
    find_package(CURL REQUIRED)
endif()

caliper_add_applet(gpt_scope
    SOURCES plugin.cpp gpt_scope.cpp
    MANIFEST gpt_scope.caliper.toml
    TORCH)
target_include_directories(gpt_scope PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(gpt_scope PRIVATE CURL::libcurl)
