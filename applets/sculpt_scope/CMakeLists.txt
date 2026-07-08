# SculptScope — a net sculpts a point cloud, rendered zero-copy from its own
# forward output (id dev.caliper.sculpt-scope). Links the in-tree libtorch the
# same way field_scope/embed_scope do; camera math is hand-rolled, fallback is
# ImPlot3D — no dependencies beyond the SDK, ui_stack, and torch.
add_library(sculpt_scope SHARED
    plugin.cpp
    sculpt_scope.cpp
)

target_include_directories(sculpt_scope PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

target_link_libraries(sculpt_scope PRIVATE
    caliper::sdk
    caliper::ui_stack
    "${TORCH_LIBRARIES}"
)

target_compile_definitions(sculpt_scope PRIVATE CALIPER_APPLET_EXPORT)

set_target_properties(sculpt_scope PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/applets"
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/applets"
    CXX_STANDARD 20
    CXX_STANDARD_REQUIRED ON
)

if(APPLE)
    set_target_properties(sculpt_scope PROPERTIES
        BUILD_RPATH "${CMAKE_SOURCE_DIR}/third_party/libtorch/lib")
endif()

add_custom_command(TARGET sculpt_scope POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        ${CMAKE_CURRENT_SOURCE_DIR}/sculpt_scope.caliper.toml
        ${CMAKE_BINARY_DIR}/applets/sculpt_scope.caliper.toml)
