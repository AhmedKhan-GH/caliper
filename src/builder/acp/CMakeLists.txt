# The acp module: an Agent Client Protocol client that runs the user's own
# agent as a child process (docs/plans/acp-integration.md).
#   caliper_builder_acp   the subprocess helper and the JSON-RPC peer
#   caliper_acp_test_child   a test program
find_package(Threads REQUIRED)

add_library(caliper_builder_acp STATIC
    subprocess.cpp
    fd_io.cpp
    jsonrpc.cpp)
target_include_directories(caliper_builder_acp PUBLIC "${CALIPER_BUILDER_INCLUDE_ROOT}")
target_link_libraries(caliper_builder_acp PUBLIC nlohmann_json::nlohmann_json Threads::Threads)
target_compile_features(caliper_builder_acp PUBLIC cxx_std_17)

if(BUILD_TESTS AND NOT WIN32)
    # Spawning and in-process pipes are POSIX only for now; on Windows the
    # library returns process.unsupported.
    add_executable(caliper_acp_test_child tests/test_child.cpp)
    caliper_builder_add_tests(acp SOURCES tests/test_subprocess.cpp tests/test_jsonrpc.cpp LIBS caliper_builder_acp)
    add_dependencies(caliper_builder_acp_tests caliper_acp_test_child)
    target_compile_definitions(caliper_builder_acp_tests PRIVATE
        CALIPER_ACP_TEST_CHILD="$<TARGET_FILE:caliper_acp_test_child>")
endif()
