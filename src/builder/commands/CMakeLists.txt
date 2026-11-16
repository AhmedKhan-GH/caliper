# caliper_builder_commands: the command layer of docs/plans/command-layer.md.
add_library(caliper_builder_commands STATIC
    command.cpp
    history.cpp
    journal.cpp
    queue.cpp
    registry.cpp
    schema.cpp
    sha256.cpp)
target_include_directories(caliper_builder_commands PUBLIC "${CALIPER_BUILDER_INCLUDE_ROOT}")
target_link_libraries(caliper_builder_commands PUBLIC nlohmann_json::nlohmann_json)
target_compile_features(caliper_builder_commands PUBLIC cxx_std_17)

caliper_builder_add_tests(commands
    SOURCES
        tests/test_schema.cpp
        tests/test_command.cpp
        tests/test_history.cpp
        tests/test_journal.cpp
        tests/test_registry.cpp
        tests/test_queue.cpp
    LIBS caliper_builder_commands)
