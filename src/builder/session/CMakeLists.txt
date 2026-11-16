# The session module (docs/plans/builder-session.md):
#   caliper_builder_session        Session, which binds the spec, catalog,
#                                  commands and loaders libraries for one
#                                  workspace, with the spec edits, the line
#                                  diff, the loader catalog provider and the
#                                  resolved data of a panel
if(NOT TARGET caliper_builder_loaders)
    message(STATUS "caliper_builder: the session module needs the loaders module and is skipped")
    return()
endif()
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

caliper_builder_add_tests(session
    SOURCES
        tests/test_line_diff.cpp
        tests/test_table_conformance.cpp
        tests/test_spec_edit.cpp
        tests/test_session.cpp
        tests/test_resolve.cpp
    LIBS caliper_builder_session)
if(TARGET caliper_builder_session_tests)
    # CALIPER_BUILDER_TEST_PYTHON (the loaders module's cache variable) names an
    # interpreter with numpy; when it is empty, the test that runs a Python
    # loader's script says it is skipped and checks only the approval.
    target_compile_definitions(caliper_builder_session_tests PRIVATE
        CALIPER_BUILDER_TEST_PYTHON="${CALIPER_BUILDER_TEST_PYTHON}")
endif()
