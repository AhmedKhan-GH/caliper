# The model runner of Hosted Runs (src/host/runner/model_runner.h,
# design/hosted-runs.md), included from the top-level CMakeLists.txt with the
# builder session. The caliper executable links it; its tests start real child
# processes and, with a Python that imports Caliper LIB, the stand-in run
# src/host/runner/tests/stand_in_run.py.
#
#   caliper_runner         the child processes, the runner's connection and
#                          the model runner
#   caliper_runner_tests   ctest caliper_runner (POSIX only: the Windows
#                          process code compiles and is not tested)
set(_runner_dir ${CMAKE_SOURCE_DIR}/src/host/runner)
add_library(caliper_runner STATIC ${_runner_dir}/model_runner.cpp ${_runner_dir}/run_connection.cpp)
if(WIN32)
    target_sources(caliper_runner PRIVATE ${_runner_dir}/child_process_win.cpp)
else()
    target_sources(caliper_runner PRIVATE ${_runner_dir}/child_process_posix.cpp)
endif()
target_link_libraries(caliper_runner PUBLIC caliper_builder_session caliper_net)
target_include_directories(caliper_runner PUBLIC ${CMAKE_SOURCE_DIR}/src/host)
set_target_properties(caliper_runner PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED ON)
if(NOT MSVC)
    target_compile_options(caliper_runner PRIVATE -Wall -Wextra)
endif()
target_link_libraries(caliper PRIVATE caliper_runner)

if(BUILD_TESTS AND TARGET doctest::doctest AND NOT WIN32)
    # A Python that imports Caliper LIB from this checkout: the M1 Python, else
    # the builder's test Python; without one the runs of the stand-in skip.
    set(_runner_python "${CALIPER_M1_PYTHON}")
    if(NOT _runner_python)
        set(_runner_python "${CALIPER_BUILDER_TEST_PYTHON}")
    endif()
    add_executable(caliper_runner_tests ${_runner_dir}/tests/test_model_runner.cpp)
    target_link_libraries(caliper_runner_tests PRIVATE caliper_runner caliper_builder_test_main doctest::doctest)
    target_compile_definitions(caliper_runner_tests PRIVATE
        CALIPER_RUNNER_TEST_PYTHON="${_runner_python}"
        CALIPER_RUNNER_PYTHONPATH="${CMAKE_SOURCE_DIR}/python"
        CALIPER_RUNNER_STAND_IN="${_runner_dir}/tests/stand_in_run.py"
        CALIPER_RUNNER_TEST_TMP="${CMAKE_CURRENT_BINARY_DIR}/runner-test-tmp")
    set_target_properties(caliper_runner_tests PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED ON)
    add_test(NAME caliper_runner COMMAND caliper_runner_tests)
    set_tests_properties(caliper_runner PROPERTIES LABELS "builder" TIMEOUT 300
        ENVIRONMENT "CALIPER_AGENT_OFFLINE=1")
endif()
