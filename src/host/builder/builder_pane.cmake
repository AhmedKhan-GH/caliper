# The WYSIWYG builder pane (src/host/builder/builder_pane.h,
# design/builder-pane.md), included from the top-level CMakeLists.txt after
# the spec runtime. Built only with CALIPER_WITH_BUILDER.
#
#   caliper_builder_pane       the pane's state and gestures over a session, its
#                              regions and drop targets as plain rectangles,
#                              its pure helpers and its ImGui side
#   caliper-builder-shot       a development tool, not installed: the pane
#                              driven through its own input on an offscreen
#                              canvas, building the three-panel dashboard by
#                              hand, with a PNG at each named step
#   caliper_builder_pane_tests ctest caliper_builder_pane (headless) and
#                              caliper_builder_pane_gpu (caliper-builder-shot
#                              as a process)
if(NOT TARGET caliper_spec_runtime)
    return()
endif()

set(_pane_dir ${CMAKE_SOURCE_DIR}/src/host/builder)
add_library(caliper_builder_pane STATIC
    ${_pane_dir}/builder_pane.cpp
    ${_pane_dir}/builder_pane_draw.cpp
    ${_pane_dir}/builder_pane_edits.cpp
    ${_pane_dir}/builder_pane_layout.cpp)
target_include_directories(caliper_builder_pane PUBLIC ${CMAKE_SOURCE_DIR}/src/host)
target_link_libraries(caliper_builder_pane PUBLIC caliper_spec_runtime caliper_builder_session)
set_target_properties(caliper_builder_pane PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED ON)
if(NOT MSVC)
    target_compile_options(caliper_builder_pane PRIVATE -Wall -Wextra)
endif()

add_executable(caliper-builder-shot ${_pane_dir}/builder_shot_main.cpp)
target_link_libraries(caliper-builder-shot PRIVATE caliper_builder_pane)
set_target_properties(caliper-builder-shot PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED ON)
caliper_spec_runtime_fonts(caliper-builder-shot)

if(BUILD_TESTS AND TARGET doctest::doctest)
    add_executable(caliper_builder_pane_tests
        ${_pane_dir}/tests/test_main.cpp
        ${_pane_dir}/tests/test_builder_pane.cpp
        ${_pane_dir}/tests/test_builder_pane_layout.cpp
        ${_pane_dir}/tests/test_builder_shot.cpp)
    target_link_libraries(caliper_builder_pane_tests PRIVATE caliper_builder_pane doctest::doctest caliper_builder_duckdb
        caliper_builder_acp)
    target_include_directories(caliper_builder_pane_tests PRIVATE ${THIRD_PARTY_DIR}/stb
        ${CMAKE_SOURCE_DIR}/src/host/spec_runtime/tests)
    target_compile_definitions(caliper_builder_pane_tests PRIVATE
        CALIPER_SPEC_FIXTURES="${CMAKE_SOURCE_DIR}/src/builder/spec/tests/fixtures"
        CALIPER_SPEC_TEST_TMP="${CMAKE_CURRENT_BINARY_DIR}/builder-pane-test-tmp"
        CALIPER_BUILDER_SHOT_EXE="$<TARGET_FILE:caliper-builder-shot>")
    set_target_properties(caliper_builder_pane_tests PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED ON)
    add_dependencies(caliper_builder_pane_tests caliper-builder-shot)
    # Headless: the pane's state and every gesture over a real session, the
    # drop targets, the spec text both ways, the regions and the file drop.
    add_test(NAME caliper_builder_pane COMMAND caliper_builder_pane_tests --test-suite-exclude=gpu)
    # Offscreen: caliper-builder-shot builds the dashboard by hand as a process.
    add_test(NAME caliper_builder_pane_gpu COMMAND caliper_builder_pane_tests --test-suite=gpu)
    set_tests_properties(caliper_builder_pane caliper_builder_pane_gpu PROPERTIES LABELS "builder;builder_pane")
    set_tests_properties(caliper_builder_pane_gpu PROPERTIES TIMEOUT 300)
endif()
