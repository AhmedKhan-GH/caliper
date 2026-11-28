# Canonical ML applet: the PyTorch C++ MNIST CNN trained live with LibTorch.
# On its own it builds against an installed SDK that was built with libtorch:
#   cmake -S . -B build -DCMAKE_PREFIX_PATH=<sdk prefix>
#   cmake --build build
if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    cmake_minimum_required(VERSION 3.18)
    project(mnist_applet LANGUAGES CXX)
    find_package(caliper-sdk 0.2 CONFIG REQUIRED)
endif()

caliper_add_applet(mnist_applet
    SOURCES mnist_applet.cpp
    MANIFEST mnist.caliper.toml
    OUTPUT_NAME mnist
    TORCH)
