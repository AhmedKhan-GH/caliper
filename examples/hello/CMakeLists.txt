add_library(hello_applet SHARED hello.cpp)
target_link_libraries(hello_applet PRIVATE caliper::sdk caliper::ui_stack)
target_compile_definitions(hello_applet PRIVATE CALIPER_APPLET_EXPORT)
set_target_properties(hello_applet PROPERTIES
    OUTPUT_NAME hello
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/applets"
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/applets"
    CXX_STANDARD 20 CXX_STANDARD_REQUIRED ON)
add_custom_command(TARGET hello_applet POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        ${CMAKE_CURRENT_SOURCE_DIR}/hello.caliper.toml
        ${CMAKE_BINARY_DIR}/applets/hello.caliper.toml)
