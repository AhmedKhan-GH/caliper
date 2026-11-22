# The runs panel of Hosted Runs (src/host/builder/runs_panel.h,
# design/hosted-runs.md), included from the top-level CMakeLists.txt after the
# model runner. Built only with CALIPER_WITH_BUILDER.
#
#   caliper_runs_panel         the panel's state, its layout and its ImGui side
#   caliper-runs-panel-shot    a development tool, not installed: the panel
#                              drawn offscreen over a fake model runner, a PNG
#   caliper_runs_panel_tests   ctest caliper_runs_panel (the fields, the card,
#                              the layout and the clicks over a session with a
#                              fake runner; then a run and its fork over the
#                              model runner and the stand-in run, skipped
#                              without a Python that imports Caliper LIB) and
#                              caliper_runs_panel_shot (the shot tool as a
#                              process; skipped without a GPU)
if(NOT TARGET caliper_spec_runtime OR NOT TARGET caliper_runner)
    return()
endif()

set(_runs_dir ${CMAKE_SOURCE_DIR}/src/host/builder)
add_library(caliper_runs_panel STATIC ${_runs_dir}/runs_panel.cpp)
target_include_directories(caliper_runs_panel PUBLIC ${CMAKE_SOURCE_DIR}/src/host)
target_link_libraries(caliper_runs_panel PUBLIC caliper_spec_runtime caliper_builder_session)
set_target_properties(caliper_runs_panel PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED ON)
if(NOT MSVC)
    target_compile_options(caliper_runs_panel PRIVATE -Wall -Wextra)
endif()

if(NOT WIN32)
    add_executable(caliper-runs-panel-shot ${_runs_dir}/runs_panel_shot_main.cpp)
    target_link_libraries(caliper-runs-panel-shot PRIVATE caliper_runs_panel)
    set_target_properties(caliper-runs-panel-shot PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED ON)
    caliper_spec_runtime_fonts(caliper-runs-panel-shot)
endif()

if(BUILD_TESTS AND TARGET doctest::doctest AND NOT WIN32)
    # A Python that imports Caliper LIB from this checkout, as the model
    # runner's tests take it; without one the run of the stand-in skips.
    set(_runs_python "${CALIPER_M1_PYTHON}")
    if(NOT _runs_python)
        set(_runs_python "${CALIPER_BUILDER_TEST_PYTHON}")
    endif()
    add_executable(caliper_runs_panel_tests ${_runs_dir}/tests/test_runs_panel.cpp)
    target_link_libraries(caliper_runs_panel_tests PRIVATE caliper_runs_panel caliper_runner caliper_builder_test_main
        doctest::doctest)
    target_compile_definitions(caliper_runs_panel_tests PRIVATE
        CALIPER_RUNS_PANEL_TEST_PYTHON="${_runs_python}"
        CALIPER_RUNS_PANEL_PYTHONPATH="${CMAKE_SOURCE_DIR}/python"
        CALIPER_RUNS_PANEL_STAND_IN="${CMAKE_SOURCE_DIR}/src/host/runner/tests/stand_in_run.py"
        CALIPER_RUNS_PANEL_TEST_TMP="${CMAKE_CURRENT_BINARY_DIR}/runs-panel-test-tmp")
    set_target_properties(caliper_runs_panel_tests PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED ON)
    add_test(NAME caliper_runs_panel COMMAND caliper_runs_panel_tests)
    set_tests_properties(caliper_runs_panel PROPERTIES LABELS "builder" TIMEOUT 300
        ENVIRONMENT "CALIPER_AGENT_OFFLINE=1")
    add_test(NAME caliper_runs_panel_shot COMMAND caliper-runs-panel-shot --scenario fork-card
        --workspace ${CMAKE_CURRENT_BINARY_DIR}/runs-panel-shot-workspace
        --out ${CMAKE_CURRENT_BINARY_DIR}/runs-panel-shot.png)
    set_tests_properties(caliper_runs_panel_shot PROPERTIES LABELS "builder" TIMEOUT 120
        ENVIRONMENT "CALIPER_AGENT_OFFLINE=1" SKIP_REGULAR_EXPRESSION "caliper_runs_panel_shot skipped"
        PASS_REGULAR_EXPRESSION "\"card\":\"Fork the run train-1")
endif()

# The app shows the panel (src/main.cpp).
if(TARGET caliper)
    target_link_libraries(caliper PRIVATE caliper_runs_panel)
endif()
