// caliper_builder_command_table: prints the descriptor table as the JSON of
// GET /v0/commands, or writes it to a file. It regenerates
// src/builder/commands/command_table.json after the table changes:
//   caliper_builder_command_table --write src/builder/commands/command_table.json
#include <caliper_builder/commands/command_table.h>

#include <cstdio>
#include <cstring>
#include <fstream>
#include <iostream>
#include <string>

int main(int argc, char** argv) {
    const auto& errors = caliper_builder::command_table_build_errors();
    for (const auto& e : errors) std::cerr << "command table: " << e << "\n";
    if (!errors.empty()) return 1;
    const std::string text = caliper_builder::command_table_json_text();
    if (argc == 1) {
        std::cout << text;
        return 0;
    }
    if (argc == 3 && std::strcmp(argv[1], "--write") == 0) {
        std::ofstream out(argv[2], std::ios::binary | std::ios::trunc);
        out << text;
        out.close();
        if (!out) {
            std::cerr << "command table: the file could not be written.\n";
            return 1;
        }
        return 0;
    }
    std::cerr << "usage: caliper_builder_command_table [--write <path>]\n";
    return 2;
}
