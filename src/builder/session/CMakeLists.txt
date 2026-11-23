# The session module (design/builder-session.md):
#   caliper_builder_session        Session, which binds the spec, catalog,
#                                  commands and loaders libraries for one
#                                  workspace, with the spec edits, the line
#                                  diff, the loader catalog provider, the
#                                  resolved data of a panel and the widget
#                                  agent's context bundle
#   caliper_builder_session_host   the strict loopback HTTP server of the
#                                  command endpoints (POSIX; a stub on Windows)
#   caliper-builder-host           a headless development tool that stands in
#                                  for the app's command endpoints until the
#                                  wiring
if(NOT TARGET caliper_builder_loaders)
    message(STATUS "caliper_builder: the session module needs the loaders module and is skipped")
    return()
endif()
find_package(Threads REQUIRED)

add_library(caliper_builder_session STATIC
    applet_files.cpp
    context_bundle.cpp
    control_edit.cpp
    event_ring.cpp
    intervene_commands.cpp
    line_diff.cpp
    loader_catalog.cpp
    panel_data.cpp
    panel_data_viewer.cpp
    program_commands.cpp
    piece_commands.cpp
    resource_commands.cpp
    program_edit.cpp
    resolve.cpp
    run_commands.cpp
    runner_commands.cpp
    session.cpp
    spec_edit.cpp
    util.cpp
    instrument_commands.cpp
    state_commands.cpp
    selection_commands.cpp
    cursor_commands.cpp
    entity_commands.cpp
    camera_commands.cpp
    annotation_commands.cpp
    values.cpp
    history.cpp
    references.cpp
    steps_window.cpp
    operator_steps.cpp
    panel_data_scene.cpp)
target_include_directories(caliper_builder_session PUBLIC "${CALIPER_BUILDER_INCLUDE_ROOT}")
target_link_libraries(caliper_builder_session
    PUBLIC caliper_builder_spec caliper_builder_programs caliper_builder_catalog caliper_builder_commands
           caliper_builder_loaders caliper_builder_operators caliper_builder_text nlohmann_json::nlohmann_json
    PRIVATE tomlplusplus::tomlplusplus)
target_compile_features(caliper_builder_session PUBLIC cxx_std_17)
if(NOT MSVC)
    target_compile_options(caliper_builder_session PRIVATE -Wall -Wextra)
endif()
# The widget agent's scope rule, which caliper-agent reads at start, embedded
# in the context bundle (context_bundle.cpp) so that both apply one rule.
set(_widget_scopes "${CMAKE_CURRENT_SOURCE_DIR}/../commands/widget_scopes.json")
file(READ "${_widget_scopes}" CALIPER_WIDGET_SCOPES_JSON)
configure_file(context_bundle_scopes.h.in
    "${CMAKE_CURRENT_BINARY_DIR}/generated/caliper_builder/widget_scopes_json.h" @ONLY)
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS "${_widget_scopes}")
target_include_directories(caliper_builder_session PRIVATE "${CMAKE_CURRENT_BINARY_DIR}/generated")

add_library(caliper_builder_session_host STATIC command_server.cpp)
target_link_libraries(caliper_builder_session_host PUBLIC caliper_builder_session Threads::Threads)
if(NOT MSVC)
    target_compile_options(caliper_builder_session_host PRIVATE -Wall -Wextra)
endif()

add_executable(caliper-builder-host host_main.cpp)
target_link_libraries(caliper-builder-host PRIVATE caliper_builder_session_host)

caliper_builder_add_tests(session
    SOURCES
        tests/test_control_edit.cpp
        tests/test_events.cpp
        tests/test_run_commands.cpp
        tests/test_runner_commands.cpp
        tests/test_program_commands.cpp
        tests/test_piece_commands.cpp
        tests/test_resource_commands.cpp
        tests/test_intervene_commands.cpp
        tests/test_line_diff.cpp
        tests/test_table_conformance.cpp
        tests/test_spec_edit.cpp
        tests/test_session.cpp
        tests/test_preview.cpp
        tests/test_context_bundle.cpp
        tests/test_review.cpp
        tests/test_content_review.cpp
        tests/test_history_refusal.cpp
        tests/test_resolve.cpp
        tests/test_panel_data.cpp
        tests/test_panel_data_viewer.cpp
        tests/test_served_sources.cpp
        tests/test_command_server.cpp
        tests/test_acceptance.cpp
        tests/test_agent_modes.cpp
        tests/test_instrument_commands.cpp
        tests/test_entity_commands.cpp
        tests/test_references.cpp
        tests/test_state.cpp
        tests/test_step_history.cpp
        tests/test_cursor.cpp
        tests/test_annotation_commands.cpp
    LIBS caliper_builder_session caliper_builder_session_host caliper_builder_acp caliper_builder_duckdb)
if(TARGET caliper_builder_session_tests)
    # The HTTP acceptance test runs the stand-in host as a child process.
    add_dependencies(caliper_builder_session_tests caliper-builder-host)
    # CALIPER_BUILDER_TEST_PYTHON (the loaders module's cache variable) names an
    # interpreter with numpy; when it is empty, the test that runs a Python
    # loader's script says it is skipped and checks only the approval.
    target_compile_definitions(caliper_builder_session_tests PRIVATE
        CALIPER_BUILDER_HOST_EXE="$<TARGET_FILE:caliper-builder-host>"
        CALIPER_BUILDER_TEST_PYTHON="${CALIPER_BUILDER_TEST_PYTHON}")
endif()
