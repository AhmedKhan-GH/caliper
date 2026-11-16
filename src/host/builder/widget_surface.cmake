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

if(BUILD_TESTS AND TARGET doctest::doctest)
    add_executable(caliper_widget_surface_tests
        ${_widget_dir}/tests/test_main.cpp
        ${_widget_dir}/tests/test_widget_surface.cpp
        ${_widget_dir}/tests/test_widget_layout.cpp
        ${_widget_dir}/tests/test_widget_shot.cpp)
    target_link_libraries(caliper_widget_surface_tests PRIVATE caliper_widget_surface doctest::doctest
        caliper_builder_duckdb)
    target_include_directories(caliper_widget_surface_tests PRIVATE ${THIRD_PARTY_DIR}/stb
        ${CMAKE_SOURCE_DIR}/src/host/spec_runtime/tests)
    target_compile_definitions(caliper_widget_surface_tests PRIVATE
        CALIPER_SPEC_FIXTURES="${CMAKE_SOURCE_DIR}/src/builder/spec/tests/fixtures"
        CALIPER_SPEC_TEST_TMP="${CMAKE_CURRENT_BINARY_DIR}/widget-test-tmp"
        CALIPER_WIDGET_SHOT_EXE="$<TARGET_FILE:caliper-widget-shot>")
    set_target_properties(caliper_widget_surface_tests PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED ON)
    add_dependencies(caliper_widget_surface_tests caliper-widget-shot)
    # Headless: the state machine over a real session, the preview's life,
    # the compact diff and the layout.
    add_test(NAME caliper_widget_surface COMMAND caliper_widget_surface_tests --test-suite-exclude=gpu)
    # Offscreen: caliper-widget-shot as a process, checked by image properties.
    add_test(NAME caliper_widget_surface_gpu COMMAND caliper_widget_surface_tests --test-suite=gpu)
    set_tests_properties(caliper_widget_surface caliper_widget_surface_gpu PROPERTIES LABELS "builder;widget")
    set_tests_properties(caliper_widget_surface_gpu PROPERTIES TIMEOUT 300)
endif()
