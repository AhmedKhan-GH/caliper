# An instrument source that includes caliper/embed.h, built against every target of
# the installed caliper-sdk package. It must configure and then fail to compile:
# the embed headers live on the caliper-embed package's own include root, which
# no SDK target exposes, even when both packages share a prefix. run.cmake
# checks that the build fails for that reason.
cmake_minimum_required(VERSION 3.18)
project(sdk_no_embed_probe LANGUAGES CXX)
find_package(caliper-sdk 0.3 CONFIG REQUIRED)

add_library(no_embed OBJECT no_embed.cpp)
target_link_libraries(no_embed PRIVATE caliper::sdk caliper::ui_stack caliper::sdk_testing)
set_target_properties(no_embed PROPERTIES CXX_STANDARD 20 CXX_STANDARD_REQUIRED ON)
