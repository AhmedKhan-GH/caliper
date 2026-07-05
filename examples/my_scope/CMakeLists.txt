add_library(my_scope SHARED my_scope.cpp)
target_link_libraries(my_scope PRIVATE caliper::sdk caliper::ui_stack)
target_compile_definitions(my_scope PRIVATE CALIPER_APPLET_EXPORT)
set_target_properties(my_scope PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/applets"
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/applets"   # .dll on Windows
    CXX_STANDARD 20 CXX_STANDARD_REQUIRED ON)
add_custom_command(TARGET my_scope POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        ${CMAKE_CURRENT_SOURCE_DIR}/my_scope.caliper.toml
        ${CMAKE_BINARY_DIR}/applets/my_scope.caliper.toml)
