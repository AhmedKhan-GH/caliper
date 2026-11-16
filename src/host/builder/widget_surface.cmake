# The widget agent's inline surface (src/host/builder/widget_surface.h,
# design/widget-agent.md), included from the top-level CMakeLists.txt after
# the spec runtime. Built only with CALIPER_WITH_BUILDER.
#
#   caliper_widget_surface   the surface's state machine, its layout and its
#                            ImGui side, the WidgetAgent interface with the
#                            scripted agent, and the agent over AcpClient
#   caliper-widget-shot      a development tool, not installed: the surface
#                            driven through a scripted scenario on one panel of
#                            a spec, a PNG at each state
#   caliper_widget_surface_tests   ctest caliper_widget_surface (headless) and
#                            caliper_widget_surface_gpu (the offscreen canvas
#                            and caliper-widget-shot as a process)
if(NOT TARGET caliper_spec_runtime)
    return()
endif()

set(_widget_dir ${CMAKE_SOURCE_DIR}/src/host/builder)
add_library(caliper_widget_surface STATIC
    ${_widget_dir}/widget_agent.cpp
    ${_widget_dir}/widget_surface.cpp)
target_include_directories(caliper_widget_surface PUBLIC ${CMAKE_SOURCE_DIR}/src/host)
target_link_libraries(caliper_widget_surface PUBLIC caliper_spec_runtime caliper_builder_session caliper_builder_acp
    caliper_builder_agent)
set_target_properties(caliper_widget_surface PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED ON)
if(NOT MSVC)
    target_compile_options(caliper_widget_surface PRIVATE -Wall -Wextra)
endif()

add_executable(caliper-widget-shot ${_widget_dir}/widget_shot_main.cpp)
target_link_libraries(caliper-widget-shot PRIVATE caliper_widget_surface)
set_target_properties(caliper-widget-shot PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED ON)
caliper_spec_runtime_fonts(caliper-widget-shot)
