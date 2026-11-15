#pragma once
// ============================================================================
// The caliper executable's command line. Parsed before any window, AppKit,
// GLFW or renderer exists, so a bad argument never flashes a window.
//
//   caliper [--applets <dir>]...      start the app
//   caliper --help                    print the usage and exit
//
// `--applets <dir>` (or `--applets=<dir>`) adds a development folder to the
// applet search, before CALIPER_APPLET_PATH and the installed folders; it may
// be given more than once, and the folders are searched in the order given.
//
// Header-only and free of any graphics dependency, so the tests include it.
// ============================================================================
#include <string>
#include <vector>

namespace caliper_host {

struct CommandLine {
    enum class Command { Run, Help };
    Command command = Command::Run;
    std::vector<std::string> applet_dirs;   // --applets, in the order given
    std::string error;                      // set: print it and the usage, exit 2
};

inline const char* usage_text() {
    return "usage: caliper [--applets <dir>]...\n"
           "       caliper --help\n";
}

inline CommandLine parse_command_line(int argc, const char* const* argv) {
    CommandLine cl;
    const std::string kApplets = "--applets";
    for (int i = 1; i < argc; i++) {
        const std::string arg = argv[i] ? argv[i] : "";
        if (arg == "--help" || arg == "-h") {
            cl.command = CommandLine::Command::Help;
        } else if (arg == kApplets) {
            if (i + 1 >= argc || !argv[i + 1] || !*argv[i + 1]) {
                cl.error = "--applets needs a folder";
                return cl;
            }
            cl.applet_dirs.push_back(argv[++i]);
        } else if (arg.rfind(kApplets + "=", 0) == 0) {
            const std::string dir = arg.substr(kApplets.size() + 1);
            if (dir.empty()) {
                cl.error = "--applets needs a folder";
                return cl;
            }
            cl.applet_dirs.push_back(dir);
        } else {
            cl.error = "unknown argument: " + arg;
            return cl;
        }
    }
    return cl;
}

} // namespace caliper_host
