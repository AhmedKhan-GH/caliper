# The widget agent's settings (src/host/builder/agent_settings.h,
# design/widget-agent.md, Presets), included from the top-level CMakeLists.txt
# after the widget surface. Built only with CALIPER_WITH_BUILDER.
#
#   caliper_agent_settings        the settings' state, widget-agent.json, the
#                                 sidecar's start-up options, the probe, the
#                                 layout and the ImGui side; the widget
#                                 surface's library links it, since the app's
#                                 agent is made from it (widget_app.cpp)
#   caliper-agent-settings-shot   a development tool, not installed: the
#                                 settings drawn offscreen in a named state,
#                                 with an in-memory key store and a fake model
#                                 server on loopback, a PNG
#   caliper_agent_settings_tests  ctest caliper_agent_settings (headless),
#                                 caliper_agent_settings_agent (List and Probe
#                                 through caliper-agent against the fake
#                                 server; skipped without
#                                 CALIPER_BUILDER_TEST_PYTHON) and
#                                 caliper_agent_settings_gpu (the shot tool as
#                                 a process); caliper_agent_settings_app (the
#                                 hidden app: ⌘K's Setup opens the settings, and
#                                 the Shared GPU note while the producer's
#                                 stream is attached)
if(NOT TARGET caliper_widget_surface)
    return()
endif()

set(_settings_dir ${CMAKE_SOURCE_DIR}/src/host/builder)
add_library(caliper_agent_settings STATIC ${_settings_dir}/agent_settings.cpp)
target_include_directories(caliper_agent_settings PUBLIC ${CMAKE_SOURCE_DIR}/src/host)
target_link_libraries(caliper_agent_settings PUBLIC caliper_spec_runtime caliper_builder_session caliper_builder_acp
    caliper_builder_agent)
set_target_properties(caliper_agent_settings PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED ON)
if(NOT MSVC)
    target_compile_options(caliper_agent_settings PRIVATE -Wall -Wextra)
endif()
target_link_libraries(caliper_widget_surface PUBLIC caliper_agent_settings)

if(NOT WIN32)
    add_executable(caliper-agent-settings-shot ${_settings_dir}/agent_settings_shot_main.cpp)
    target_link_libraries(caliper-agent-settings-shot PRIVATE caliper_widget_surface)
    set_target_properties(caliper-agent-settings-shot PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED ON)
    caliper_spec_runtime_fonts(caliper-agent-settings-shot)
endif()

if(BUILD_TESTS AND TARGET doctest::doctest AND NOT WIN32)
    add_executable(caliper_agent_settings_tests
        ${_settings_dir}/tests/test_main.cpp
        ${_settings_dir}/tests/test_agent_settings.cpp
        ${_settings_dir}/tests/test_agent_settings_app.cpp)
    target_link_libraries(caliper_agent_settings_tests PRIVATE caliper_widget_surface doctest::doctest
        caliper_builder_duckdb)
    target_include_directories(caliper_agent_settings_tests PRIVATE ${THIRD_PARTY_DIR}/stb
        ${CMAKE_SOURCE_DIR}/src/host/spec_runtime/tests)
    target_compile_definitions(caliper_agent_settings_tests PRIVATE
        CALIPER_SPEC_TEST_TMP="${CMAKE_CURRENT_BINARY_DIR}/agent-settings-test-tmp"
        CALIPER_SPEC_FIXTURES="${CMAKE_SOURCE_DIR}/src/builder/spec/tests/fixtures"
        CALIPER_BUILDER_SOURCE_DIR="${CMAKE_SOURCE_DIR}/src/builder"
        CALIPER_BUILDER_TEST_PYTHON="${CALIPER_BUILDER_TEST_PYTHON}"
        CALIPER_AGENT_SETTINGS_SHOT_EXE="$<TARGET_FILE:caliper-agent-settings-shot>")
    set_target_properties(caliper_agent_settings_tests PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED ON)
    add_dependencies(caliper_agent_settings_tests caliper-agent-settings-shot)
    add_test(NAME caliper_agent_settings COMMAND caliper_agent_settings_tests --test-suite=settings)
    add_test(NAME caliper_agent_settings_agent COMMAND caliper_agent_settings_tests --test-suite=agent)
    add_test(NAME caliper_agent_settings_gpu COMMAND caliper_agent_settings_tests --test-suite=gpu)
    set_tests_properties(caliper_agent_settings caliper_agent_settings_agent caliper_agent_settings_gpu
        PROPERTIES LABELS "builder;widget")
    set_tests_properties(caliper_agent_settings_agent PROPERTIES TIMEOUT 300
        SKIP_REGULAR_EXPRESSION "caliper_agent_settings_agent skipped")
    set_tests_properties(caliper_agent_settings_gpu PROPERTIES TIMEOUT 300
        SKIP_REGULAR_EXPRESSION "caliper_agent_settings_gpu skipped")
    # No test of the settings may reach a model off this machine or touch the
    # system's key store: the switch keeps caliper-agent offline.
    set_tests_properties(caliper_agent_settings caliper_agent_settings_agent caliper_agent_settings_gpu
        PROPERTIES ENVIRONMENT "CALIPER_AGENT_OFFLINE=1")
    if(TARGET caliper)
        add_dependencies(caliper_agent_settings_tests caliper)
        set(_settings_app_env "CALIPER_AGENT_OFFLINE=1;CALIPER_TEST_CALIPER_EXE=$<TARGET_FILE:caliper>")
        if(APPLE AND TARGET caliper_net)
            # caliper_attention_producer is defined later, under the same condition.
            add_dependencies(caliper_agent_settings_tests caliper_attention_producer)
            list(APPEND _settings_app_env "CALIPER_TEST_PRODUCER_EXE=$<TARGET_FILE:caliper_attention_producer>"
                "CALIPER_TEST_INSTRUMENTS=${CMAKE_BINARY_DIR}/instruments")
        endif()
        add_test(NAME caliper_agent_settings_app COMMAND caliper_agent_settings_tests --test-suite=app)
        set_tests_properties(caliper_agent_settings_app PROPERTIES LABELS "builder;widget;cli" TIMEOUT 400
            ENVIRONMENT "${_settings_app_env}" SKIP_REGULAR_EXPRESSION "caliper_agent_settings_app skipped")
    endif()
endif()
