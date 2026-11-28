# The agent module (design/byok-agent.md):
#   caliper_builder_agent  the key store of the widget agent, one key per
#                          provider preset: an interface, an in-memory store for
#                          tests and the macOS Keychain store (compiled on macOS,
#                          never called by a test); other systems get a stub
#   caliper_agent/         the Python sidecar caliper-agent (caliper[agent] in root pyproject.toml)
#
# The sidecar's pytest suite runs as the ctest builder_agent_py when
# CALIPER_BUILDER_TEST_PYTHON names a Python interpreter that has the sidecar's
# dependencies and its test extras (pydantic-ai-slim, anthropic, openai, pytest,
# agent-client-protocol; the loaders module's tests of the same variable also
# need numpy and pyarrow), for example a local venv installed with the agent
# and test packages from the root pyproject.toml:
#   uv venv --python 3.12 src/builder/agent/.venv
#   uv pip install --python src/builder/agent/.venv/bin/python \
#     "pydantic-ai-slim[anthropic]>=2.68,<3" "anthropic>=2.8,<3" "openai>=3.51,<4" \
#     "pytest>=8" "agent-client-protocol>=1.0,<2" numpy pyarrow
#   cmake ... -DCALIPER_BUILDER_TEST_PYTHON=src/builder/agent/.venv/bin/python
# Without it the test reports that it was skipped.
set(_agent_sources key_store.cpp)
if(APPLE)
    list(APPEND _agent_sources keychain_store_macos.cpp)
endif()

add_library(caliper_builder_agent STATIC ${_agent_sources})
target_include_directories(caliper_builder_agent PUBLIC "${CALIPER_BUILDER_INCLUDE_ROOT}")
target_link_libraries(caliper_builder_agent PUBLIC caliper_builder_acp)
target_compile_features(caliper_builder_agent PUBLIC cxx_std_17)
# The provider presets, which caliper-agent reads at start, embedded in the key
# store (key_store.cpp) so that both know the same ids and key variables; the
# SHA-256 of a Custom entry's base URL comes from the commands library.
set(_presets "${CMAKE_CURRENT_SOURCE_DIR}/caliper_agent/presets.json")
file(READ "${_presets}" CALIPER_PRESETS_JSON)
configure_file(presets_json.h.in "${CMAKE_CURRENT_BINARY_DIR}/generated/caliper_builder/presets_json.h" @ONLY)
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS "${_presets}")
target_include_directories(caliper_builder_agent PRIVATE "${CMAKE_CURRENT_BINARY_DIR}/generated")
target_link_libraries(caliper_builder_agent PRIVATE caliper_builder_commands nlohmann_json::nlohmann_json)
if(APPLE)
    target_link_libraries(caliper_builder_agent PRIVATE "-framework Security" "-framework CoreFoundation")
endif()

caliper_builder_add_tests(agent SOURCES tests/test_key_store.cpp LIBS caliper_builder_agent)

# The sidecar end to end on the real command layer: caliper-acp-harness drives
# caliper-agent, whose tools call caliper-builder-host over a fresh workspace
# (tests/test_dashboard_e2e.cpp). POSIX only, like the host; it needs the
# session module and CALIPER_BUILDER_TEST_PYTHON, and says it is skipped
# without the interpreter.
if(BUILD_TESTS AND NOT WIN32 AND TARGET caliper_builder_session_host AND TARGET caliper-builder-host)
    caliper_builder_add_tests(agent_e2e SOURCES tests/test_dashboard_e2e.cpp
        LIBS caliper_builder_session caliper_builder_acp caliper_builder_duckdb)
    add_dependencies(caliper_builder_agent_e2e_tests caliper-builder-host caliper-acp-harness)
    target_compile_definitions(caliper_builder_agent_e2e_tests PRIVATE
        CALIPER_BUILDER_HOST_EXE="$<TARGET_FILE:caliper-builder-host>"
        CALIPER_ACP_HARNESS="$<TARGET_FILE:caliper-acp-harness>"
        CALIPER_BUILDER_TEST_PYTHON="${CALIPER_BUILDER_TEST_PYTHON}")
    # CALIPER_AGENT_OFFLINE: no test reaches a model server off loopback (caliper_agent/offline.py).
    set_tests_properties(builder_agent_e2e PROPERTIES TIMEOUT 300 SKIP_REGULAR_EXPRESSION "builder_agent_e2e skipped"
        ENVIRONMENT "CALIPER_AGENT_OFFLINE=1")

    # A widget session end to end: the sidecar, driven by caliper-acp-harness
    # and by a small client, edits a panel on a stacked preview of a session
    # that runs in the test behind the stand-in's server, and the test commits
    # it from the builder origin (tests/test_widget_e2e.cpp).
    caliper_builder_add_tests(agent_widget SOURCES tests/test_widget_e2e.cpp
        LIBS caliper_builder_session_host caliper_builder_session caliper_builder_acp caliper_builder_duckdb)
    add_dependencies(caliper_builder_agent_widget_tests caliper-acp-harness)
    target_compile_definitions(caliper_builder_agent_widget_tests PRIVATE
        CALIPER_ACP_HARNESS="$<TARGET_FILE:caliper-acp-harness>"
        CALIPER_BUILDER_TEST_PYTHON="${CALIPER_BUILDER_TEST_PYTHON}")
    set_tests_properties(builder_agent_widget PROPERTIES TIMEOUT 300
        SKIP_REGULAR_EXPRESSION "builder_agent_widget skipped" ENVIRONMENT "CALIPER_AGENT_OFFLINE=1")
endif()

# The provider conformance suite (conformance/conformance_main.cpp): the widget
# agent's tasks on a fixture workspace, through the real session, context bundle
# and ACP client. A tool for tests and for on-demand runs, so it is built with
# the tests; POSIX only, like the stand-in's server. tests/test_conformance.py
# runs it against the scripted model and the fake OpenAI-compatible server.
if(BUILD_TESTS AND NOT WIN32 AND TARGET caliper_builder_session_host AND TARGET caliper_builder_duckdb)
    set(_tasks "${CMAKE_CURRENT_SOURCE_DIR}/conformance/tasks.json")
    file(READ "${_tasks}" CALIPER_CONFORMANCE_TASKS_JSON)
    configure_file(conformance/tasks_json.h.in
        "${CMAKE_CURRENT_BINARY_DIR}/generated/caliper_builder/conformance_tasks_json.h" @ONLY)
    set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS "${_tasks}")
    add_executable(caliper-agent-conformance conformance/conformance_main.cpp)
    target_include_directories(caliper-agent-conformance PRIVATE "${CMAKE_CURRENT_BINARY_DIR}/generated")
    target_link_libraries(caliper-agent-conformance PRIVATE caliper_builder_session_host caliper_builder_session
        caliper_builder_acp caliper_builder_agent caliper_builder_duckdb)
    target_compile_definitions(caliper-agent-conformance PRIVATE
        CALIPER_BUILDER_SOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}/..")
endif()

if(BUILD_TESTS)
    # CALIPER_BUILDER_TEST_PYTHON is defined and validated in src/builder/CMakeLists.txt.
    if(CALIPER_BUILDER_TEST_PYTHON)
        add_test(NAME builder_agent_py
            COMMAND "${CALIPER_BUILDER_TEST_PYTHON}" -m pytest -q -p no:cacheprovider
                    --basetemp=${CMAKE_CURRENT_BINARY_DIR}/pytest-tmp
                    ${CMAKE_CURRENT_SOURCE_DIR}/tests
            WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
        set(_agent_py_env "PYTHONDONTWRITEBYTECODE=1" "CALIPER_AGENT_OFFLINE=1" "PYTHONPATH=${CMAKE_CURRENT_SOURCE_DIR}")
        if(TARGET caliper-agent-conformance)
            # tests/test_conformance.py runs the suite in its default modes.
            list(APPEND _agent_py_env "CALIPER_AGENT_CONFORMANCE=$<TARGET_FILE:caliper-agent-conformance>")
        endif()
        if(NOT WIN32)
            # The end-to-end test drives the sidecar through the real C++ client.
            list(APPEND _agent_py_env "CALIPER_ACP_HARNESS=$<TARGET_FILE:caliper-acp-harness>")
        endif()
        set_tests_properties(builder_agent_py PROPERTIES ENVIRONMENT "${_agent_py_env}" TIMEOUT 900)
    else()
        add_test(NAME builder_agent_py
            COMMAND ${CMAKE_COMMAND} -E echo
                "builder_agent_py skipped: set CALIPER_BUILDER_TEST_PYTHON to a Python with pydantic-ai and pytest")
        set_tests_properties(builder_agent_py PROPERTIES SKIP_REGULAR_EXPRESSION "builder_agent_py skipped")
    endif()
endif()
