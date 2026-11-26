# The approval cards (src/host/builder/approval_cards.h,
# design/acp-integration.md), included from the top-level CMakeLists.txt after
# the caliper target's builder block. The caliper executable compiles
# approval_cards.cpp itself, as it does spec_page.cpp, and with the builder
# links the inline surface's library for the diff card the two share.
#
#   caliper_approval_cards_tests   ctest caliper_approval_cards (headless: the
#                                  cards over a real session with review on,
#                                  and their layout) and
#                                  caliper_approval_cards_app (the hidden app
#                                  and caliper-mcp: an outside agent's
#                                  panel.create waits, is approved and lands)
if(NOT TARGET caliper_widget_surface)
    return()
endif()

target_link_libraries(caliper PRIVATE caliper_widget_surface)

set(_cards_dir ${CMAKE_SOURCE_DIR}/src/host/builder)
if(BUILD_TESTS AND TARGET doctest::doctest)
    add_executable(caliper_approval_cards_tests
        ${_cards_dir}/approval_cards.cpp
        ${_cards_dir}/tests/test_approval_cards.cpp
        ${_cards_dir}/tests/test_approval_cards_app.cpp)
    target_link_libraries(caliper_approval_cards_tests PRIVATE caliper_builder_test_main doctest::doctest
        caliper_widget_surface caliper_builder_duckdb caliper_builder_acp)
    target_include_directories(caliper_approval_cards_tests PRIVATE ${CMAKE_SOURCE_DIR}/src/host
        ${CMAKE_SOURCE_DIR}/src/host/spec_runtime/tests ${CMAKE_SOURCE_DIR}/src/builder/mcp/tests
        ${CMAKE_SOURCE_DIR}/tests)
    target_compile_definitions(caliper_approval_cards_tests PRIVATE
        CALIPER_SPEC_FIXTURES="${CMAKE_SOURCE_DIR}/src/builder/spec/tests/fixtures"
        CALIPER_SPEC_TEST_TMP="${CMAKE_CURRENT_BINARY_DIR}/approval-test-tmp")
    if(TARGET caliper-mcp)
        target_compile_definitions(caliper_approval_cards_tests PRIVATE CALIPER_MCP_EXE="$<TARGET_FILE:caliper-mcp>")
        add_dependencies(caliper_approval_cards_tests caliper caliper-mcp)
    endif()
    set_target_properties(caliper_approval_cards_tests PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED ON)
    add_test(NAME caliper_approval_cards COMMAND caliper_approval_cards_tests --test-suite-exclude=app)
    set_tests_properties(caliper_approval_cards PROPERTIES LABELS "builder" TIMEOUT 120
        ENVIRONMENT "CALIPER_AGENT_OFFLINE=1")
    # The hidden app and the real caliper-mcp, POSIX only as the other
    # process tests of the hidden app; skipped without a Metal device.
    if(NOT WIN32 AND TARGET caliper-mcp)
        add_test(NAME caliper_approval_cards_app COMMAND caliper_approval_cards_tests --test-suite=app)
        set_tests_properties(caliper_approval_cards_app PROPERTIES LABELS "cli;builder" TIMEOUT 240
            ENVIRONMENT "CALIPER_TEST_CALIPER_EXE=$<TARGET_FILE:caliper>;CALIPER_AGENT_OFFLINE=1"
            SKIP_REGULAR_EXPRESSION "run skipped: no GPU")
    endif()
endif()
