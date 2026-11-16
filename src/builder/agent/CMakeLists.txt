# The agent module (docs/plans/byok-agent.md):
#   caliper_builder_agent  the key store of the agent pane's BYOK backend: an
#                          interface, an in-memory store for tests and the macOS
#                          Keychain store (compiled on macOS, never called by a
#                          test); other systems get a stub
#   caliper_agent/         the Python sidecar caliper-agent (its own pyproject.toml)
#
# The sidecar's pytest suite runs as the ctest builder_agent_py when
# CALIPER_BUILDER_TEST_PYTHON names a Python interpreter that has the sidecar's
# dependencies (anthropic, pytest), for example the .venv of this folder:
#   uv venv --python 3.12 src/builder/agent/.venv
#   uv pip install --python src/builder/agent/.venv/bin/python anthropic pytest
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
if(APPLE)
    target_link_libraries(caliper_builder_agent PRIVATE "-framework Security" "-framework CoreFoundation")
endif()

caliper_builder_add_tests(agent SOURCES tests/test_key_store.cpp LIBS caliper_builder_agent)

if(BUILD_TESTS)
    set(CALIPER_BUILDER_TEST_PYTHON "" CACHE FILEPATH
        "A Python interpreter with anthropic and pytest, to run the caliper-agent tests")
    if(CALIPER_BUILDER_TEST_PYTHON)
        add_test(NAME builder_agent_py
            COMMAND "${CALIPER_BUILDER_TEST_PYTHON}" -m pytest -q -p no:cacheprovider
                    --basetemp=${CMAKE_CURRENT_BINARY_DIR}/pytest-tmp
                    ${CMAKE_CURRENT_SOURCE_DIR}/tests
            WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
        set(_agent_py_env "PYTHONDONTWRITEBYTECODE=1")
        if(NOT WIN32)
            # The end-to-end test drives the sidecar through the real C++ client.
            list(APPEND _agent_py_env "CALIPER_ACP_HARNESS=$<TARGET_FILE:caliper-acp-harness>")
        endif()
        set_tests_properties(builder_agent_py PROPERTIES ENVIRONMENT "${_agent_py_env}" TIMEOUT 600)
    else()
        add_test(NAME builder_agent_py
            COMMAND ${CMAKE_COMMAND} -E echo
                "builder_agent_py skipped: set CALIPER_BUILDER_TEST_PYTHON to a Python with anthropic and pytest")
        set_tests_properties(builder_agent_py PROPERTIES SKIP_REGULAR_EXPRESSION "builder_agent_py skipped")
    endif()
endif()
