# FieldScope — charged particles in an EM field (id dev.caliper.field-scope).
# Links the in-tree libtorch the same way flow_scope/embed_scope/gpt_scope do;
# no other dependencies — the camera math is hand-rolled and the fallback is
# ImPlot3D.
add_library(field_scope SHARED
    plugin.cpp
    field_scope.cpp
)

target_include_directories(field_scope PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

target_link_libraries(field_scope PRIVATE
    caliper::sdk
    caliper::ui_stack
    "${TORCH_LIBRARIES}"
)

target_compile_definitions(field_scope PRIVATE CALIPER_APPLET_EXPORT)

set_target_properties(field_scope PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/applets"
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/applets"
    CXX_STANDARD 20
    CXX_STANDARD_REQUIRED ON
)

if(APPLE)
    set_target_properties(field_scope PROPERTIES
        BUILD_RPATH "${CMAKE_SOURCE_DIR}/third_party/libtorch/lib")
endif()

add_custom_command(TARGET field_scope POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        ${CMAKE_CURRENT_SOURCE_DIR}/field_scope.caliper.toml
        ${CMAKE_BINARY_DIR}/applets/field_scope.caliper.toml)
