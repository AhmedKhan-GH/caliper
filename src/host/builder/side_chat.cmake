# The side chat (src/host/builder/side_chat.h, design/acp-integration.md),
# included from the top-level CMakeLists.txt after the widget surface. Built
# only with CALIPER_WITH_BUILDER.
#
#   caliper_side_chat        the agents and their launch, the handoff draft,
#                            the state machine over AcpClient, the layout and
#                            the ImGui side
#   caliper-chat-shot        a development tool, not installed: the dashboard
#                            drawn by the spec runtime beside the side chat,
#                            driven by a scripted agent through a session, the
#                            app's API server and caliper-mcp, a PNG at each
#                            state
#   caliper_chat_stub_agent  a test program: an ACP agent with scripted turns
#                            that drive Caliper through the caliper-mcp it is
#                            handed, and exits on cue
#   caliper_side_chat_tests  ctest caliper_side_chat (headless: the state
#                            machine against the stub agents, the launch, the
#                            handoff and the layout) and caliper_side_chat_gpu
#                            (caliper-chat-shot as a process)
if(NOT TARGET caliper_widget_surface OR NOT TARGET caliper_api_session OR NOT TARGET caliper-mcp)
    return()
endif()

set(_chat_dir ${CMAKE_SOURCE_DIR}/src/host/builder)
add_library(caliper_side_chat STATIC ${_chat_dir}/side_chat.cpp)
target_include_directories(caliper_side_chat PUBLIC ${CMAKE_SOURCE_DIR}/src/host)
target_link_libraries(caliper_side_chat PUBLIC caliper_widget_surface caliper_builder_acp caliper_builder_commands)
# A development build finds caliper-mcp in its build tree when it is not
# beside the executable (find_caliper_mcp()); packaging puts it beside.
target_compile_definitions(caliper_side_chat PRIVATE CALIPER_MCP_BUILD_PATH="$<TARGET_FILE:caliper-mcp>")
add_dependencies(caliper_side_chat caliper-mcp)
set_target_properties(caliper_side_chat PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED ON)
if(NOT MSVC)
    target_compile_options(caliper_side_chat PRIVATE -Wall -Wextra)
endif()

# The app's pane (chat_pane.cpp, compiled by the caliper target itself as
# spec_page.cpp is) draws this component.
if(TARGET caliper)
    target_link_libraries(caliper PRIVATE caliper_side_chat)
endif()

if(NOT WIN32)
    add_executable(caliper-chat-shot ${_chat_dir}/chat_shot_main.cpp)
    target_link_libraries(caliper-chat-shot PRIVATE caliper_side_chat caliper_api_session)
    set_target_properties(caliper-chat-shot PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED ON)
    caliper_spec_runtime_fonts(caliper-chat-shot)
    target_compile_definitions(caliper-chat-shot PRIVATE CALIPER_MCP_EXE="$<TARGET_FILE:caliper-mcp>")
endif()

if(BUILD_TESTS AND TARGET doctest::doctest AND NOT WIN32)
    add_executable(caliper_chat_stub_agent ${_chat_dir}/tests/chat_stub_agent.cpp)
    target_link_libraries(caliper_chat_stub_agent PRIVATE caliper_builder_acp nlohmann_json::nlohmann_json)
    set_target_properties(caliper_chat_stub_agent PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED ON)

    add_executable(caliper_side_chat_tests
        ${_chat_dir}/tests/test_main.cpp
        ${_chat_dir}/tests/test_side_chat.cpp
        ${_chat_dir}/tests/test_side_chat_launch.cpp
        ${_chat_dir}/tests/test_side_chat_layout.cpp
        ${_chat_dir}/tests/test_chat_shot.cpp)
    target_link_libraries(caliper_side_chat_tests PRIVATE caliper_side_chat caliper_api_session doctest::doctest
        caliper_builder_duckdb)
    target_include_directories(caliper_side_chat_tests PRIVATE ${THIRD_PARTY_DIR}/stb
        ${CMAKE_SOURCE_DIR}/src/host/spec_runtime/tests)
    target_compile_definitions(caliper_side_chat_tests PRIVATE
        CALIPER_SPEC_FIXTURES="${CMAKE_SOURCE_DIR}/src/builder/spec/tests/fixtures"
        CALIPER_SPEC_TEST_TMP="${CMAKE_CURRENT_BINARY_DIR}/side-chat-test-tmp"
        CALIPER_CHAT_SHOT_EXE="$<TARGET_FILE:caliper-chat-shot>"
        CALIPER_MCP_EXE="$<TARGET_FILE:caliper-mcp>"
        CALIPER_ACP_STUB_AGENT="$<TARGET_FILE:caliper_acp_stub_agent>"
        CALIPER_CHAT_STUB_AGENT="$<TARGET_FILE:caliper_chat_stub_agent>")
    set_target_properties(caliper_side_chat_tests PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED ON)
    add_dependencies(caliper_side_chat_tests caliper-chat-shot caliper-mcp caliper_acp_stub_agent caliper_chat_stub_agent)
    # Headless: the state machine against the stub agents, the launch, the
    # handoff draft and the layout.
    add_test(NAME caliper_side_chat COMMAND caliper_side_chat_tests --test-suite-exclude=gpu)
    # Offscreen: caliper-chat-shot as a process, the scenario's conversation
    # and the panel the agent added through the command layer.
    add_test(NAME caliper_side_chat_gpu COMMAND caliper_side_chat_tests --test-suite=gpu)
    set_tests_properties(caliper_side_chat caliper_side_chat_gpu PROPERTIES LABELS "builder;chat")
    set_tests_properties(caliper_side_chat PROPERTIES TIMEOUT 240)
    set_tests_properties(caliper_side_chat_gpu PROPERTIES TIMEOUT 300)
    # No test of the chat may reach a model or an agent off this machine; the
    # agents here are the stubs, and the switch keeps caliper-agent offline.
    set_tests_properties(caliper_side_chat caliper_side_chat_gpu PROPERTIES ENVIRONMENT "CALIPER_AGENT_OFFLINE=1")
endif()
