# PulseScope — the caliper.feed.v1 dashboard exemplar (feed spec §5).
# Deliberately CPU + ImPlot only: NO torch, NO bridge/geometry. It links just the
# SDK sugar and the ImGui/ImPlot UI stack (the hello_applet shape), so it builds
# and loads host-neutrally — on a host with no telemetry provider it degrades to
# one honest line.
add_library(pulse_scope SHARED
    plugin.cpp
    pulse_scope.cpp
)

target_include_directories(pulse_scope PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

target_link_libraries(pulse_scope PRIVATE
    caliper::sdk
    caliper::ui_stack
)

target_compile_definitions(pulse_scope PRIVATE CALIPER_APPLET_EXPORT)

set_target_properties(pulse_scope PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/applets"
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/applets"
    CXX_STANDARD 20
    CXX_STANDARD_REQUIRED ON
)

add_custom_command(TARGET pulse_scope POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        ${CMAKE_CURRENT_SOURCE_DIR}/pulse_scope.caliper.toml
        ${CMAKE_BINARY_DIR}/applets/pulse_scope.caliper.toml)
