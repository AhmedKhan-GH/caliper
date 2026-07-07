# FlowScope — the caliper.geometry.v1 exemplar (id dev.caliper.flow-scope).
# Links the in-tree libtorch the same way embed_scope/gpt_scope do; no other
# dependencies — the camera math is hand-rolled and the fallback is ImPlot3D.
add_library(flow_scope SHARED
    plugin.cpp
    flow_scope.cpp
)

target_include_directories(flow_scope PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

target_link_libraries(flow_scope PRIVATE
    caliper::sdk
    caliper::ui_stack
    "${TORCH_LIBRARIES}"
)

target_compile_definitions(flow_scope PRIVATE CALIPER_APPLET_EXPORT)

set_target_properties(flow_scope PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/applets"
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/applets"
    CXX_STANDARD 20
    CXX_STANDARD_REQUIRED ON
)

if(APPLE)
    set_target_properties(flow_scope PROPERTIES
        BUILD_RPATH "${CMAKE_SOURCE_DIR}/third_party/libtorch/lib")
endif()

add_custom_command(TARGET flow_scope POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        ${CMAKE_CURRENT_SOURCE_DIR}/flow_scope.caliper.toml
        ${CMAKE_BINARY_DIR}/applets/flow_scope.caliper.toml)
