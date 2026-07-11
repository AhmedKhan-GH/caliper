add_library(instance_scope SHARED
    plugin.cpp
    instance_scope.cpp
)

target_include_directories(instance_scope PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

target_link_libraries(instance_scope PRIVATE
    caliper::sdk
    caliper::ui_stack
    "${TORCH_LIBRARIES}"
)

target_compile_definitions(instance_scope PRIVATE CALIPER_APPLET_EXPORT)

set_target_properties(instance_scope PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/applets"
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/applets"
    CXX_STANDARD 20
    CXX_STANDARD_REQUIRED ON
)

if(APPLE)
    set_target_properties(instance_scope PROPERTIES
        BUILD_RPATH "${CMAKE_SOURCE_DIR}/third_party/libtorch/lib")
endif()

add_custom_command(TARGET instance_scope POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        ${CMAKE_CURRENT_SOURCE_DIR}/instance_scope.caliper.toml
        ${CMAKE_BINARY_DIR}/applets/instance_scope.caliper.toml)
