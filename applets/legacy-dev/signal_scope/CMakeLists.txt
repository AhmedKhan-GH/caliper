# The complete build definition of a Caliper applet. In-tree it uses the
# monorepo's caliper::sdk / caliper::ui_stack targets; an out-of-tree applet
# replaces those two lines with a find_package/CPM fetch of caliper-sdk
# (identical target names — that symmetry is the point).
add_library(signal_scope_applet SHARED signal_scope.cpp)
target_link_libraries(signal_scope_applet PRIVATE caliper::sdk caliper::ui_stack)
target_compile_definitions(signal_scope_applet PRIVATE CALIPER_APPLET_EXPORT)
set_target_properties(signal_scope_applet PROPERTIES
    OUTPUT_NAME signal_scope
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/applets"
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/applets"
    CXX_STANDARD 20 CXX_STANDARD_REQUIRED ON)
add_custom_command(TARGET signal_scope_applet POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        ${CMAKE_CURRENT_SOURCE_DIR}/signal_scope.caliper.toml
        ${CMAKE_BINARY_DIR}/applets/signal_scope.caliper.toml)
