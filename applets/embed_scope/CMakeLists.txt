# EmbedScope — the 3D embedding projector exemplar (Phase 2F′, Task F5). It
# exercises the whole platform surface: jobs/device/metrics/tensor_bridge plus
# the two new demand-driven services artifacts.v1 + data.v1. Links the in-tree
# libtorch the same way ml_scope/gpt_scope do (runtime packs replace this at
# Phase 4). curl + zlib are macOS system libs covering the cancellable MNIST
# download + gunzip inside the job; the MNIST IDX reader is reused verbatim from
# its own mnist_idx.h (self-contained — the exemplar doubles as a template). The host
# links none of these: data acquisition is applet business.
find_package(CURL REQUIRED)
find_package(ZLIB REQUIRED)

add_library(embed_scope SHARED
    plugin.cpp
    embed_scope.cpp
)

target_include_directories(embed_scope PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR})   # mnist_idx.h is local — the exemplar is self-contained

target_link_libraries(embed_scope PRIVATE
    caliper::sdk
    caliper::ui_stack
    "${TORCH_LIBRARIES}"
    CURL::libcurl
    ZLIB::ZLIB)

target_compile_definitions(embed_scope PRIVATE CALIPER_APPLET_EXPORT)

set_target_properties(embed_scope PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/applets"
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/applets"
    CXX_STANDARD 20
    CXX_STANDARD_REQUIRED ON)

if(APPLE)
    set_target_properties(embed_scope PROPERTIES
        BUILD_RPATH "${CMAKE_SOURCE_DIR}/third_party/libtorch/lib")
endif()

add_custom_command(TARGET embed_scope POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        ${CMAKE_CURRENT_SOURCE_DIR}/embed_scope.caliper.toml
        ${CMAKE_BINARY_DIR}/applets/embed_scope.caliper.toml)
