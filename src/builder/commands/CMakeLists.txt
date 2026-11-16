# caliper_builder_commands: the command layer of docs/plans/command-layer.md:
# the command model, the schema validator, the undo history, the journal, the
# registry with its approval policy, the queue and the descriptor table.
add_library(caliper_builder_commands STATIC
    command.cpp
    command_table.cpp
    history.cpp
    journal.cpp
    queue.cpp
    registry.cpp
    schema.cpp
    sha256.cpp)
target_include_directories(caliper_builder_commands PUBLIC "${CALIPER_BUILDER_INCLUDE_ROOT}")
target_link_libraries(caliper_builder_commands PUBLIC nlohmann_json::nlohmann_json)
target_compile_features(caliper_builder_commands PUBLIC cxx_std_17)

# Prints the descriptor table, or regenerates command_table.json:
#   caliper_builder_command_table --write src/builder/commands/command_table.json
add_executable(caliper_builder_command_table tools/dump_command_table.cpp)
target_link_libraries(caliper_builder_command_table PRIVATE caliper_builder_commands)

caliper_builder_add_tests(commands
    SOURCES
        tests/test_schema.cpp
        tests/test_command.cpp
        tests/test_history.cpp
        tests/test_journal.cpp
        tests/test_registry.cpp
        tests/test_queue.cpp
        tests/test_command_table.cpp
    LIBS caliper_builder_commands caliper_builder_catalog)
