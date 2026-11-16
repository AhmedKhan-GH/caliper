# The mcp module: caliper-mcp, the stdio MCP server that an outside agent
# launches to drive the running Caliper app (docs/plans/mcp-server.md).
#   caliper_builder_mcp   the loopback HTTP client and the codecs it shares
find_package(Threads REQUIRED)

add_library(caliper_builder_mcp STATIC
    codec.cpp
    loopback_http.cpp)
target_include_directories(caliper_builder_mcp PUBLIC "${CALIPER_BUILDER_INCLUDE_ROOT}")
target_link_libraries(caliper_builder_mcp PUBLIC caliper_builder_commands nlohmann_json::nlohmann_json Threads::Threads)
target_compile_features(caliper_builder_mcp PUBLIC cxx_std_17)

if(BUILD_TESTS)
    set(_mcp_test_sources tests/test_codec.cpp)
    if(NOT WIN32)
        # The loopback client and the fake server are POSIX only for now.
        list(APPEND _mcp_test_sources tests/fake_http.cpp tests/test_http.cpp)
    endif()
    caliper_builder_add_tests(mcp SOURCES ${_mcp_test_sources} LIBS caliper_builder_mcp)
endif()
