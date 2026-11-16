# The session module (docs/design/builder-session.md):
#   caliper_builder_session        Session, which binds the spec, catalog,
#                                  commands and loaders libraries for one
#                                  workspace, with the spec edits, the line
#                                  diff, the loader catalog provider and the
#                                  resolved data of a panel
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
    line_diff.cpp
    loader_catalog.cpp
    resolve.cpp
    session.cpp
    spec_edit.cpp
    util.cpp)
target_include_directories(caliper_builder_session PUBLIC "${CALIPER_BUILDER_INCLUDE_ROOT}")
target_link_libraries(caliper_builder_session
    PUBLIC caliper_builder_spec caliper_builder_catalog caliper_builder_commands caliper_builder_loaders
           nlohmann_json::nlohmann_json
    PRIVATE tomlplusplus::tomlplusplus)
target_compile_features(caliper_builder_session PUBLIC cxx_std_17)
if(NOT MSVC)
    target_compile_options(caliper_builder_session PRIVATE -Wall -Wextra)
endif()

add_library(caliper_builder_session_host STATIC command_server.cpp)
target_link_libraries(caliper_builder_session_host PUBLIC caliper_builder_session Threads::Threads)
if(NOT MSVC)
    target_compile_options(caliper_builder_session_host PRIVATE -Wall -Wextra)
endif()

add_executable(caliper-builder-host host_main.cpp)
target_link_libraries(caliper-builder-host PRIVATE caliper_builder_session_host)

caliper_builder_add_tests(session
    SOURCES
        tests/test_line_diff.cpp
        tests/test_table_conformance.cpp
        tests/test_spec_edit.cpp
        tests/test_session.cpp
        tests/test_resolve.cpp
        tests/test_command_server.cpp
        tests/test_acceptance.cpp
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
