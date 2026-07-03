# GPTScope — the flagship applet (Task E1). Links the in-tree libtorch the same
# way repnet_demo/ml_scope do (runtime packs replace this at Phase 4). curl is a
# macOS system lib covering the cancellable TinyShakespeare download inside the
# job — no zlib: the corpus is plain text. The host links neither: data
# acquisition is applet business.
find_package(CURL REQUIRED)

add_library(gpt_scope SHARED
    plugin.cpp
    gpt_scope.cpp
)

target_include_directories(gpt_scope PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

target_link_libraries(gpt_scope PRIVATE
    caliper::sdk
    caliper::ui_stack
    "${TORCH_LIBRARIES}"
    CURL::libcurl
)

target_compile_definitions(gpt_scope PRIVATE CALIPER_APPLET_EXPORT)

set_target_properties(gpt_scope PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/applets"
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/applets"
    CXX_STANDARD 20
    CXX_STANDARD_REQUIRED ON
)

if(APPLE)
    set_target_properties(gpt_scope PROPERTIES
        BUILD_RPATH "${CMAKE_SOURCE_DIR}/third_party/libtorch/lib")
endif()

add_custom_command(TARGET gpt_scope POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        ${CMAKE_CURRENT_SOURCE_DIR}/gpt_scope.caliper.toml
        ${CMAKE_BINARY_DIR}/applets/gpt_scope.caliper.toml)
