// caliper-mcp: the stdio MCP server that an outside agent launches to drive
// the running Caliper app (docs/plans/mcp-server.md). Standard output carries
// MCP messages only; diagnostics go to standard error.
#include <caliper_builder/mcp/server.h>

#include <cstdio>
#include <cstring>
#include <string>

namespace {

const char kUsage[] =
    "usage: caliper-mcp [--api-file <path>] [--docs <dir>] [--sdk <dir>] [--origin mcp|agent]\n"
    "                   [--command-timeout <seconds>]\n"
    "       caliper-mcp --version | --help\n"
    "\n"
    "Serves MCP on standard input and output and drives the running Caliper app through\n"
    "Caliper API. The API file is api.json in the app-data folder (CALIPER_DATA_DIR moves it)\n"
    "unless --api-file names another. --docs and --sdk expose the documentation pages and the\n"
    "SDK headers as resources. A command call waits up to --command-timeout seconds (130 by\n"
    "default), since a command may wait for the person's approval in the app.\n";

int usage_error(const std::string& message) {
    std::fprintf(stderr, "caliper-mcp: %s\n%s", message.c_str(), kUsage);
    return 2;
}

}  // namespace

int main(int argc, char** argv) {
    caliper_builder::mcp::ServerOptions options;
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        std::string value;
        bool has_value = false;
        const auto eq = arg.find('=');
        if (arg.rfind("--", 0) == 0 && eq != std::string::npos) {
            value = arg.substr(eq + 1);
            arg = arg.substr(0, eq);
            has_value = true;
        }
        auto take = [&](const char* name) -> bool {
            if (has_value) return true;
            if (i + 1 >= argc) return false;
            value = argv[++i];
            (void)name;
            return true;
        };
        if (arg == "--help" || arg == "-h") {
            std::fputs(kUsage, stdout);
            return 0;
        }
        if (arg == "--version") {
            std::printf("caliper-mcp %s\n", caliper_builder::mcp::server_version());
            return 0;
        }
        if (arg == "--api-file" || arg == "--docs" || arg == "--sdk" || arg == "--origin" ||
            arg == "--command-timeout") {
            if (!take(arg.c_str()) || value.empty()) return usage_error(arg + " needs a value");
            if (arg == "--api-file") options.api_file = value;
            else if (arg == "--docs") options.docs_dir = value;
            else if (arg == "--sdk") options.sdk_dir = value;
            else if (arg == "--origin") {
                if (value != "mcp" && value != "agent") return usage_error("--origin is mcp or agent");
                options.origin = value;
            } else {
                int seconds = 0;
                for (char c : value) {
                    if (c < '0' || c > '9' || seconds > 100000) return usage_error("--command-timeout takes whole seconds");
                    seconds = seconds * 10 + (c - '0');
                }
                if (seconds < 1 || seconds > 3600) return usage_error("--command-timeout is 1 to 3600 seconds");
                options.command_timeout_ms = seconds * 1000;
            }
            continue;
        }
        return usage_error("unknown argument: " + arg);
    }
    return caliper_builder::mcp::run_stdio_server(options);
}
