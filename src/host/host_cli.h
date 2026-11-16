#pragma once
// ============================================================================
// The caliper executable's command line. It is parsed before any window,
// AppKit, GLFW or renderer exists. `caliper` with no subcommand and
// `caliper dev` start the app; every other subcommand runs and exits without
// starting it, so it never shows a window. `shot` draws the app's applet page
// into an offscreen canvas (shot.h) and opens no window either.
//
//   caliper [--applets <dir>]... [--url <link>] [--api [--api-port <n>] [--allow-control]] [--hidden]
//           [--workspace <dir>]               start the app
//   caliper dev <folder> [--applets <dir>]... [--url <link>] [--api [--api-port <n>] [--allow-control]]
//               [--hidden] [--workspace <dir>] start the app, <folder> searched first
//   caliper list [--applets <dir>]...         list the applets in search order
//   caliper check <folder>                    run the pre-load checks on <folder>
//   caliper install <folder>                  check, then copy into the user folder
//   caliper new <name> [<dir>]                create an applet from a template
//   caliper build <folder>                    configure and build it, then check it
//   caliper info                              print versions and served services
//   caliper shot [--applets <dir>]... --launch <id> [--size <W>x<H>]
//                [--frames <N>] --out <file.png> [--api [--api-port <n>] [--allow-control]]
//                [--workspace <dir>]          write the applet page as a PNG
//   caliper serve [--port <n>] [--token-file <file>] [--assets <dir>]
//                                             run the hub of wire protocol v3
//   caliper --help                            print the usage
//
// `--applets <dir>` (or `--applets=<dir>`) adds a development folder to the
// search, before CALIPER_APPLET_PATH and the installed folders; it may be
// given more than once, and the folders are searched in the order given.
// A development folder without manifests whose applets/ subfolder has some,
// such as an applet's build folder, stands for that subfolder
// (resolve_applet_folder); dev, check and install resolve their folder the
// same way.
//
// `caliper serve` runs the hub (net/hub.h) on 127.0.0.1 until SIGINT or
// SIGTERM; it needs no window, renderer or GPU either. --port, --token-file
// and --assets apply only to it. `--allow-control` lets the hub forward
// viewers' controls (wire protocol 3.1) to their session's script: with serve,
// and with --api for the stream of Caliper API. Off without it.
//
// `--api` serves Caliper API v0 (api/api_server.h) on 127.0.0.1 while the app
// runs: with no subcommand, with dev, and with shot, whose offscreen canvas
// is the app's windowless mode. `--api-port <n>` chooses its port (7341
// without it, 0 for any free port) and needs --api.
//
// `--url <link>` (or `--url=<link>`) attaches the app to a stream of wire
// protocol v3 as soon as it starts, as the attach pane's stream link does: the
// ws:// link of a viewer or the http:// page link, holding the session token.
// It applies to starting the app and to dev; a link that does not attach
// leaves the app on its landing page and says why in Activity.
//
// `--hidden` (or CALIPER_HIDDEN=1), a testing aid, runs the real app with
// its window never shown: no Dock icon, no menu bar, no focus (main.cpp).
// It applies to starting the app and to dev.
//
// `--workspace <dir>` (or `--workspace=<dir>`) names the builder's workspace
// folder (src/host/builder/app_session.h) for the app, dev and shot, in place
// of CALIPER_WORKSPACE and of workspace/ under the app-data folder
// (workspace_choice below). A workspace that cannot be opened leaves the app
// running without the builder and says why.
//
// On Windows, `serve`, `build`, `--api` and `--url` are refused before
// anything starts, with one message (windows_unavailable below).
//
// Exit codes: 0 success; 1 a check failed, nothing was found, or a file
// operation failed; 2 the command line is wrong (the usage goes to stderr).
//
// Header-only but for serve's signal handler (host_cli.cpp) and `build`
// (cli_build.cpp), and free of any graphics dependency: main() dispatches here
// and the tests call the same functions with their own CliContext. `shot`
// needs the renderer, so only its parsing is here and main() runs it
// (run_shot in shot.h).
// ============================================================================
#include "applet_loader.h"
#include "negotiation.h"
#include "net/hub.h"
#include "net/json.h"
#include <algorithm>
#include <atomic>
#include <iterator>
#include <cctype>
#include <chrono>
#include <csignal>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <map>
#include <ostream>
#include <string>
#include <system_error>
#include <vector>
#ifdef _WIN32
#include <process.h>   // _getpid
#else
#include <unistd.h>    // getpid
#endif

namespace caliper_host {

struct CommandLine {
    enum class Command { Run, Help, Dev, List, Check, Install, New, Build, Info, Shot, Serve };
    Command command = Command::Run;
    std::vector<std::string> applet_dirs;   // dev's folder, then --applets, in order
    std::vector<std::string> operands;      // the subcommand's operands
    // serve
    int port = -1;                          // --port; -1: kDefaultHubPort
    std::string token_file;                 // --token-file
    std::string assets_dir;                 // --assets
    // Caliper API (the app, dev and shot)
    bool api = false;                       // --api
    int api_port = -1;                      // --api-port; -1: caliper_api::kDefaultPort
    bool hidden = false;                    // --hidden: the app's window is never shown
    std::string url;                        // --url: a stream link to attach to at start
    bool allow_control = false;             // --allow-control: the hub forwards controls
    std::string workspace;                  // --workspace: the builder's workspace folder
    std::string error;                      // set: print it and the usage, exit 2

    // shot's options, with their defaults.
    std::string launch_id;                  // --launch <id>
    int width = 1600;                       // --size <W>x<H>, in pixels
    int height = 1000;
    int frames = 60;                        // --frames <N>
    std::string out;                        // --out <file.png>
};

inline const char* usage_text() {
    return "usage: caliper [--applets <dir>]... [--url <link>] [--api [--api-port <n>] [--allow-control]] [--hidden]\n"
           "               [--workspace <dir>]\n"
           "       caliper dev <folder> [--applets <dir>]... [--url <link>] [--api [--api-port <n>] [--allow-control]]\n"
           "                   [--hidden] [--workspace <dir>]\n"
           "       caliper list [--applets <dir>]...\n"
           "       caliper check <folder>\n"
           "       caliper install <folder>\n"
           "       caliper new <name> [<dir>]\n"
           "       caliper build <folder>\n"
           "       caliper info\n"
           "       caliper shot [--applets <dir>]... --launch <id> [--size <W>x<H>]\n"
           "                    [--frames <N>] --out <file.png> [--api [--api-port <n>] [--allow-control]]\n"
           "                    [--workspace <dir>]\n"
           "       caliper serve [--port <n>] [--token-file <file>] [--assets <dir>] [--allow-control]\n"
           "       caliper --help\n";
}

namespace cli_detail {

// A whole number from lo to hi written in decimal digits only, or -1.
inline int parse_count(const std::string& s, int lo, int hi) {
    if (s.empty() || s.size() > 9) return -1;
    int n = 0;
    for (char c : s) {
        if (c < '0' || c > '9') return -1;
        n = n * 10 + (c - '0');
    }
    return n >= lo && n <= hi ? n : -1;
}

} // namespace cli_detail

// The largest side `shot --size` accepts: the largest texture side that Metal
// supports on Apple GPUs, and that desktop Vulkan drivers commonly support.
constexpr int kShotMaxSide = 16384;

// The port `caliper serve` listens on without --port.
constexpr int kDefaultHubPort = 7340;

// ---- what the Windows build refuses for now ------------------------------------
// The network stream, the hub, Caliper API and `caliper build` compile on
// Windows, and the in-process tests of the WebSocket layer, the hub and the
// API run there, but none of them has run on Windows as a process yet, and
// the network source that --url attaches through is macOS-only, as the live
// attach is. So the Windows build refuses them before anything starts:
// main() prints windows_unavailable_message() and exits with 1. The function
// is pure and platform-neutral, so the tests check it everywhere.
inline const char* windows_unavailable(const CommandLine& cl) {
    using C = CommandLine::Command;
    if (cl.command == C::Serve) return "caliper serve";
    if (cl.command == C::Build) return "caliper build";
    if (cl.command != C::Run && cl.command != C::Dev && cl.command != C::Shot) return nullptr;
    if (cl.api) return "Caliper API (--api)";
    if (!cl.url.empty()) return "attaching to a stream (--url)";
    return nullptr;
}

inline std::string windows_unavailable_message(const char* what) {
    return std::string(what) + " is not available on Windows yet";
}

inline CommandLine parse_command_line(int argc, const char* const* argv) {
    using C = CommandLine::Command;
    CommandLine cl;
    const std::string kApplets = "--applets";
    bool have_command = false;
    // shot's options, in the order of their first appearance.
    std::vector<std::string> shot_options;
    std::vector<std::string> serve_options;   // given, to refuse them elsewhere
    bool api_port_given = false;
    for (int i = 1; i < argc; i++) {
        const std::string arg = argv[i] ? argv[i] : "";
        // macOS passes -psn_<number> to an app that Finder launches on older
        // systems; it is not ours to read.
        if (arg.rfind("-psn_", 0) == 0) continue;
        if (arg == "--help" || arg == "-h") {
            cl.command = C::Help;
            cl.operands.clear();
            cl.applet_dirs.clear();
            return cl;
        }
        if (arg == kApplets || arg.rfind(kApplets + "=", 0) == 0) {
            std::string dir;
            if (arg == kApplets) {
                if (i + 1 < argc && argv[i + 1]) dir = argv[++i];
            } else {
                dir = arg.substr(kApplets.size() + 1);
            }
            // "--applets --help" names no folder: an option is never one.
            if (dir.empty() || dir.rfind("--", 0) == 0) {
                cl.error = "--applets needs a folder";
                return cl;
            }
            cl.applet_dirs.push_back(dir);
            continue;
        }
        // shot's options: --launch, --size, --frames and --out, each with its
        // value as the next argument or after '='.
        bool shot_option = false;
        for (const std::string opt : {"--launch", "--size", "--frames", "--out"}) {
            if (arg != opt && arg.rfind(opt + "=", 0) != 0) continue;
            shot_option = true;
            std::string value;
            if (arg == opt) {
                if (i + 1 < argc && argv[i + 1]) value = argv[++i];
            } else {
                value = arg.substr(opt.size() + 1);
            }
            if (std::find(shot_options.begin(), shot_options.end(), opt) !=
                shot_options.end()) {
                cl.error = opt + " is given twice";
                return cl;
            }
            shot_options.push_back(opt);
            if (value.empty() || value.rfind("--", 0) == 0) {
                cl.error = opt + (opt == "--launch"   ? " needs an applet id"
                                  : opt == "--size"   ? " needs a size such as 1600x1000"
                                  : opt == "--frames" ? " needs a number of frames"
                                                      : " needs a file");
                return cl;
            }
            if (opt == "--launch") {
                cl.launch_id = value;
            } else if (opt == "--out") {
                cl.out = value;
            } else if (opt == "--frames") {
                cl.frames = cli_detail::parse_count(value, 1, 100000);
                if (cl.frames < 0) {
                    cl.error = "--frames takes a whole number from 1 to 100000: " + value;
                    return cl;
                }
            } else {
                const size_t x = value.find('x');
                const int w = x == std::string::npos
                                  ? -1
                                  : cli_detail::parse_count(value.substr(0, x), 1, kShotMaxSide);
                const int h = x == std::string::npos
                                  ? -1
                                  : cli_detail::parse_count(value.substr(x + 1), 1, kShotMaxSide);
                if (w < 0 || h < 0) {
                    cl.error = "--size takes <W>x<H>, each from 1 to " +
                               std::to_string(kShotMaxSide) + " pixels, such as 1600x1000: " +
                               value;
                    return cl;
                }
                cl.width = w;
                cl.height = h;
            }
            break;
        }
        if (shot_option) continue;
        // Controls through the hub (serve, or the stream of --api): a flag.
        if (arg == "--allow-control") {
            cl.allow_control = true;
            continue;
        }
        if (arg.rfind("--allow-control=", 0) == 0) {
            cl.error = "--allow-control takes no value";
            return cl;
        }
        // The hidden window (a testing aid): a flag.
        if (arg == "--hidden") {
            cl.hidden = true;
            continue;
        }
        if (arg.rfind("--hidden=", 0) == 0) {
            cl.error = "--hidden takes no value";
            return cl;
        }
        // A stream to attach to at start: --url <link>.
        if (arg == "--url" || arg.rfind("--url=", 0) == 0) {
            std::string value;
            if (arg == "--url") {
                if (i + 1 < argc && argv[i + 1]) value = argv[++i];
            } else {
                value = arg.substr(std::string("--url=").size());
            }
            if (value.empty() || value.rfind("--", 0) == 0) {
                cl.error = "--url needs a stream link";
                return cl;
            }
            if (!cl.url.empty()) {
                cl.error = "--url is given twice";
                return cl;
            }
            cl.url = value;
            continue;
        }
        // The builder's workspace: --workspace <dir>.
        if (arg == "--workspace" || arg.rfind("--workspace=", 0) == 0) {
            std::string value;
            if (arg == "--workspace") {
                if (i + 1 < argc && argv[i + 1]) value = argv[++i];
            } else {
                value = arg.substr(std::string("--workspace=").size());
            }
            if (value.empty() || value.rfind("--", 0) == 0) {
                cl.error = "--workspace needs a folder";
                return cl;
            }
            if (!cl.workspace.empty()) {
                cl.error = "--workspace is given twice";
                return cl;
            }
            cl.workspace = value;
            continue;
        }
        // Caliper API: --api, a flag, and --api-port <n>.
        if (arg == "--api") {
            cl.api = true;
            continue;
        }
        if (arg.rfind("--api=", 0) == 0) {
            cl.error = "--api takes no value; choose the port with --api-port <n>";
            return cl;
        }
        if (arg == "--api-port" || arg.rfind("--api-port=", 0) == 0) {
            std::string value;
            if (arg == "--api-port") {
                if (i + 1 < argc && argv[i + 1]) value = argv[++i];
            } else {
                value = arg.substr(std::string("--api-port=").size());
            }
            if (value.empty() || value.rfind("--", 0) == 0) {
                cl.error = "--api-port needs a number";
                return cl;
            }
            if (api_port_given) {
                cl.error = "--api-port is given twice";
                return cl;
            }
            api_port_given = true;
            cl.api_port = cli_detail::parse_count(value, 0, 65535);
            if (cl.api_port < 0) {
                cl.error = "--api-port needs a number from 0 to 65535, not " + value;
                return cl;
            }
            continue;
        }
        // The options of serve: --port, --token-file, --assets.
        bool serve_option = false;
        for (const char* opt : {"--port", "--token-file", "--assets"}) {
            const std::string o = opt;
            if (arg != o && arg.rfind(o + "=", 0) != 0) continue;
            serve_option = true;
            std::string value;
            if (arg == o) {
                if (i + 1 < argc && argv[i + 1]) value = argv[++i];
            } else {
                value = arg.substr(o.size() + 1);
            }
            if (value.empty()) {
                cl.error = o + (o == "--port" ? " needs a number" : o == "--assets" ? " needs a folder" : " needs a file");
                return cl;
            }
            if (o == "--port") {
                int port = 0;
                bool ok = value.size() <= 5;
                for (char c : value) ok = ok && c >= '0' && c <= '9';
                if (ok) port = std::stoi(value);
                if (!ok || port > 65535) {
                    cl.error = "--port needs a number from 0 to 65535, not " + value;
                    return cl;
                }
                cl.port = port;
            } else if (o == "--token-file") {
                cl.token_file = value;
            } else {
                cl.assets_dir = value;
            }
            serve_options.push_back(o);
        }
        if (serve_option) continue;
        if (arg.size() > 1 && arg[0] == '-') {
            cl.error = "unknown option: " + arg;
            return cl;
        }
        if (!have_command) {
            have_command = true;
            if (arg == "dev") cl.command = C::Dev;
            else if (arg == "list") cl.command = C::List;
            else if (arg == "check") cl.command = C::Check;
            else if (arg == "install") cl.command = C::Install;
            else if (arg == "new") cl.command = C::New;
            else if (arg == "build") cl.command = C::Build;
            else if (arg == "info") cl.command = C::Info;
            else if (arg == "shot") cl.command = C::Shot;
            else if (arg == "serve") cl.command = C::Serve;
            else if (arg == "help") cl.command = C::Help;
            else {
                cl.error = "unknown command: " + arg;
                return cl;
            }
            continue;
        }
        cl.operands.push_back(arg);
    }

    auto operands = [&](size_t lo, size_t hi, const char* name) {
        if (cl.operands.size() < lo || cl.operands.size() > hi)
            cl.error = std::string(name) +
                       (lo == hi ? (lo == 0 ? " takes no operands"
                                            : " takes one folder")
                                 : " takes a name and an optional folder");
    };
    switch (cl.command) {
        case C::Run:
        case C::Help:
            break;
        case C::Dev:
            operands(1, 1, "dev");
            if (cl.error.empty())   // the dev folder is searched first
                cl.applet_dirs.insert(cl.applet_dirs.begin(), cl.operands[0]);
            break;
        case C::List: operands(0, 0, "list"); break;
        case C::Info: operands(0, 0, "info"); break;
        case C::Serve: operands(0, 0, "serve"); break;
        case C::Check: operands(1, 1, "check"); break;
        case C::Install: operands(1, 1, "install"); break;
        case C::New: operands(1, 2, "new"); break;
        case C::Build: operands(1, 1, "build"); break;
        case C::Shot:
            operands(0, 0, "shot");
            if (cl.error.empty() && cl.launch_id.empty())
                cl.error = "shot needs --launch <id>";
            else if (cl.error.empty() && cl.out.empty())
                cl.error = "shot needs --out <file.png>";
            break;
    }
    if (cl.error.empty() && !cl.applet_dirs.empty() &&
        (cl.command == C::Check || cl.command == C::Install || cl.command == C::Build ||
         cl.command == C::New || cl.command == C::Info || cl.command == C::Serve))
        cl.error = "--applets applies only to starting the app, dev, list and shot";
    if (cl.error.empty() && !shot_options.empty() && cl.command != C::Shot)
        cl.error = shot_options.front() + " applies only to shot";
    if (cl.error.empty() && !serve_options.empty() && cl.command != C::Serve)
        cl.error = serve_options.front() + " applies only to serve";
    if (cl.error.empty() && (cl.api || api_port_given) && cl.command != C::Run &&
        cl.command != C::Dev && cl.command != C::Shot && cl.command != C::Help)
        cl.error = std::string(cl.api ? "--api" : "--api-port") +
                   " applies only to starting the app, dev and shot";
    if (cl.error.empty() && api_port_given && !cl.api) cl.error = "--api-port needs --api";
    if (cl.error.empty() && cl.hidden && cl.command != C::Run && cl.command != C::Dev &&
        cl.command != C::Help)
        cl.error = "--hidden applies only to starting the app and dev";
    if (cl.error.empty() && !cl.url.empty() && cl.command != C::Run && cl.command != C::Dev &&
        cl.command != C::Help)
        cl.error = "--url applies only to starting the app and dev";
    if (cl.error.empty() && !cl.workspace.empty() && cl.command != C::Run && cl.command != C::Dev &&
        cl.command != C::Shot && cl.command != C::Help)
        cl.error = "--workspace applies only to starting the app, dev and shot";
    if (cl.error.empty() && cl.allow_control && cl.command != C::Serve && cl.command != C::Help) {
        if (cl.command != C::Run && cl.command != C::Dev && cl.command != C::Shot)
            cl.error = "--allow-control applies only to serve, and to the app, dev and shot with --api";
        else if (!cl.api)
            cl.error = "--allow-control needs --api when starting the app, dev or shot";
    }
    return cl;
}

// The builder's workspace folder (src/host/builder/app_session.h):
// --workspace, else CALIPER_WORKSPACE when it is set and not empty, else
// `default_dir`, workspace/ under the app-data folder
// (caliper::default_workspace_dir()). `chosen` is false for the default,
// which the app opens only when something first needs it; a chosen folder is
// opened as soon as the app has drawn its first frame. A relative path is
// taken from the working directory.
struct WorkspaceChoice {
    std::string dir;
    bool chosen = false;
    const char* from = "default";   // "--workspace", "CALIPER_WORKSPACE" or "default"
};

inline WorkspaceChoice workspace_choice(const CommandLine& cl, const char* env,
                                        const std::string& default_dir) {
    WorkspaceChoice w;
    if (!cl.workspace.empty()) {
        w.dir = cl.workspace;
        w.chosen = true;
        w.from = "--workspace";
    } else if (env && *env) {
        w.dir = env;
        w.chosen = true;
        w.from = "CALIPER_WORKSPACE";
    } else {
        w.dir = default_dir;
    }
    std::error_code ec;
    std::filesystem::path abs = std::filesystem::absolute(std::filesystem::path(w.dir), ec);
    if (ec) return w;
    abs = abs.lexically_normal();
    // "work/" is the folder "work": a trailing separator leaves an empty file name.
    if (!abs.has_filename() && abs.has_relative_path()) abs = abs.parent_path();
    w.dir = abs.string();
    return w;
}

// What the subcommands read from the host. main() fills it with the
// executable's own values; a test supplies its own folders.
struct CliContext {
    HostCaps caps;                      // what the host advertises (host_caps())
    std::string sdk_version;            // host_sdk_version(); "" when unknown
    std::string user_dir;               // the user applets folder
    std::string shipped_dir;            // the applets shipped with the app
    std::string data_root;              // <app data>/data, for data-schema stamps
    const char* applet_path_env = nullptr;   // CALIPER_APPLET_PATH
    std::string hub_file;               // serve: where producers find the hub ("" none)
    std::string exe_dir;                // build: the executable's folder, for the SDK search
};

namespace cli_detail {

namespace fs = std::filesystem;

inline bool passes(const AppletEntry& a) {
    return a.status == AppletStatus::Ready || a.status == AppletStatus::Shadowed;
}

// The order list prints the applets in: search order, then id, then the
// highest version first.
inline std::vector<int> list_order(const AppletLoader& L) {
    std::vector<int> order;
    for (int i = 0; i < L.count(); i++) order.push_back(i);
    std::stable_sort(order.begin(), order.end(), [&](int x, int y) {
        const AppletEntry& a = L.at(x);
        const AppletEntry& b = L.at(y);
        if (a.search_rank != b.search_rank) return a.search_rank < b.search_rank;
        const std::string& ia = a.manifest.id.empty() ? a.manifest.name : a.manifest.id;
        const std::string& ib = b.manifest.id.empty() ? b.manifest.name : b.manifest.id;
        if (ia != ib) return ia < ib;
        return semver_cmp(a.manifest.version, b.manifest.version) > 0;
    });
    return order;
}

inline std::string list_id(const AppletEntry& a) {
    return a.manifest.id.empty() ? a.manifest.name : a.manifest.id;
}

// One line per applet: status, id, version, kind (library or spec), folder,
// in columns; a refused, failed or shadowed applet's reason on the next line,
// under the id. `notes`, when given, holds lines to print under an entry
// after its reason, or in its place for a failed entry: check's diagnostics
// of a spec applet, whose first error is the reason of a failed one.
inline void print_entries(const AppletLoader& L, std::ostream& out,
                          const std::map<int, std::vector<std::string>>* notes = nullptr) {
    const std::vector<int> order = list_order(L);
    auto id_of = [](const AppletEntry& a) { return list_id(a); };
    auto version_of = [](const AppletEntry& a) {
        return a.manifest.version.empty() ? std::string("-") : a.manifest.version;
    };
    auto kind_of = [](const AppletEntry& a) {
        return std::string(applet_kind_word(a.manifest.kind));
    };
    size_t ws = 0, wi = 0, wv = 0, wk = 0;
    for (int i : order) {
        ws = std::max(ws, std::string(applet_status_word(L.at(i).status)).size());
        wi = std::max(wi, id_of(L.at(i)).size());
        wv = std::max(wv, version_of(L.at(i)).size());
        wk = std::max(wk, kind_of(L.at(i)).size());
    }
    auto pad = [](const std::string& s, size_t w) {
        return s + std::string(w > s.size() ? w - s.size() : 0, ' ');
    };
    for (int i : order) {
        const AppletEntry& a = L.at(i);
        out << pad(applet_status_word(a.status), ws) << "  " << pad(id_of(a), wi)
            << "  " << pad(version_of(a), wv) << "  " << pad(kind_of(a), wk) << "  "
            << a.folder << "\n";
        const std::vector<std::string>* lines = nullptr;
        if (notes)
            if (auto it = notes->find(i); it != notes->end()) lines = &it->second;
        // A failed spec applet's reason is its first error, which its
        // diagnostics hold too.
        const bool reason = a.status != AppletStatus::Ready && !a.status_text.empty() &&
                            !(lines && a.status == AppletStatus::Failed);
        if (reason) out << std::string(ws + 2, ' ') << a.status_text << "\n";
        if (lines)
            for (const std::string& line : *lines) out << std::string(ws + 2, ' ') << line << "\n";
    }
}

// Scans one folder and prints it; the number of applets that do not pass.
// A spec applet that passes the listing checks is then validated in full
// (AppletLoader::check_spec), and its diagnostics are printed under it with
// their lines and columns; one with an error fails.
inline int check_folder(const std::string& folder, AppletLoader& L,
                        std::ostream& out, std::ostream& err) {
    std::error_code ec;
    if (!fs::is_directory(folder, ec)) {
        err << "caliper: not a folder: " << folder << "\n";
        return -1;
    }
    // Absolute, as list prints folders, and resolved: a build folder whose
    // manifests are in its applets/ subfolder stands for that subfolder.
    const auto dirs = applet_search_dirs({folder}, nullptr, "", "");
    L.scan(dirs);
    if (L.count() == 0) {
        err << "caliper: no applets in " << dirs.front() << "\n";
        return -1;
    }
    if (resolve_applet_folder(folder) != folder)   // its applets/ subfolder
        out << "using the applets in " << dirs.front() << "\n";
    std::map<int, std::vector<std::string>> notes;
    for (int i = 0; i < L.count(); i++) {
        const AppletEntry& a = L.at(i);
        if (a.manifest.kind != AppletKind::Spec || !passes(a)) continue;
        std::vector<SpecDiagnostic> found;
        L.check_spec(i, &found);
        const std::string file = fs::path(a.spec_path).filename().string();
        for (const SpecDiagnostic& d : found)
            notes[i].push_back(format_spec_diagnostic(file, d, true));
    }
    print_entries(L, out, &notes);
    int failing = 0;
    for (int i = 0; i < L.count(); i++) failing += passes(L.at(i)) ? 0 : 1;
    return failing;
}

// Writes `text` to `path` through a temporary file renamed over it.
inline bool write_file(const fs::path& path, const std::string& text) {
    const fs::path tmp = path.string() + ".tmp";
    {
        std::ofstream f(tmp, std::ios::binary | std::ios::trunc);
        if (!f) return false;
        f << text;
        f.flush();
        if (!f) return false;
    }
    std::error_code ec;
    fs::rename(tmp, path, ec);
    if (ec) fs::remove(tmp, ec);
    return !ec;
}

// One file or folder that install places, from the checked folder into the
// user applets folder.
struct InstallCopy {
    fs::path from;
    fs::path to;
};

// Places every copy, all or nothing. Each source is first copied to a
// temporary name beside its destination; only when every copy is staged are
// they renamed into place, in the order given, a destination that exists
// being moved aside first. If a step fails, everything this call placed is
// removed, everything it moved aside is put back, and every temporary is
// removed. `fail_at` makes the rename of that copy fail, for the tests.
inline bool install_all(const std::vector<InstallCopy>& copies, std::string& error,
                        int fail_at = -1) {
    const std::string suffix =
        ".install-" + std::to_string(std::chrono::steady_clock::now()
                                         .time_since_epoch().count());
    struct Step {
        fs::path tmp, to, aside;
        bool moved_aside = false, placed = false;
    };
    std::vector<Step> steps;
    std::error_code ec;
    auto cleanup = [&] {
        for (auto it = steps.rbegin(); it != steps.rend(); ++it) {
            std::error_code e;
            if (it->placed) fs::remove_all(it->to, e);
            if (it->moved_aside) fs::rename(it->aside, it->to, e);
            fs::remove_all(it->tmp, e);
        }
    };
    // Hidden temporary names: the loader never takes them for applets.
    for (const InstallCopy& c : copies) {
        Step st;
        st.to = c.to;
        st.tmp = c.to.parent_path() / ("." + c.to.filename().string() + suffix);
        st.aside = c.to.parent_path() / ("." + c.to.filename().string() + suffix + ".old");
        steps.push_back(st);
        fs::create_directories(c.to.parent_path(), ec);
        if (fs::is_directory(c.from, ec))
            fs::copy(c.from, st.tmp, fs::copy_options::recursive, ec);
        else
            fs::copy_file(c.from, st.tmp, ec);
        if (ec) {
            error = "could not copy " + c.from.string() + ": " + ec.message();
            cleanup();
            return false;
        }
    }
    for (size_t i = 0; i < steps.size(); i++) {
        Step& st = steps[i];
        if (fs::exists(fs::symlink_status(st.to, ec))) {
            fs::rename(st.to, st.aside, ec);
            if (ec) {
                error = "could not replace " + st.to.string() + ": " + ec.message();
                cleanup();
                return false;
            }
            st.moved_aside = true;
        }
        if ((int)i == fail_at)
            ec = std::make_error_code(std::errc::io_error);
        else
            fs::rename(st.tmp, st.to, ec);
        if (ec) {
            error = "could not place " + st.to.string() + ": " + ec.message();
            cleanup();
            return false;
        }
        st.placed = true;
    }
    for (const Step& st : steps)
        if (st.moved_aside) fs::remove_all(st.aside, ec);
    return true;
}

inline std::string replace_all(std::string s, const std::string& from,
                               const std::string& to) {
    for (size_t at = s.find(from); at != std::string::npos;
         at = s.find(from, at + to.size()))
        s.replace(at, from.size(), to);
    return s;
}

} // namespace cli_detail

// ---- caliper list ----------------------------------------------------------
inline int run_list(const CommandLine& cl, const CliContext& ctx, std::ostream& out,
                    std::ostream& /*err*/) {
    AppletLoader L(ctx.caps, ctx.data_root);
    L.scan(applet_search_dirs(cl.applet_dirs, ctx.applet_path_env, ctx.user_dir,
                              ctx.shipped_dir));
    if (L.count() == 0) {
        out << "no applets found\n";
        return 0;
    }
    cli_detail::print_entries(L, out);
    return 0;
}

// ---- caliper check <folder> ------------------------------------------------
inline int run_check(const CommandLine& cl, const CliContext& ctx, std::ostream& out,
                     std::ostream& err) {
    AppletLoader L(ctx.caps, ctx.data_root);
    const int failing = cli_detail::check_folder(cl.operands[0], L, out, err);
    if (failing < 0) return 1;
    out << (L.count() - failing) << " of " << L.count() << " applets pass the checks\n";
    return failing == 0 ? 0 : 1;
}

// ---- caliper install <folder> ----------------------------------------------
inline int run_install(const CommandLine& cl, const CliContext& ctx, std::ostream& out,
                       std::ostream& err) {
    namespace fs = std::filesystem;
    const std::string& folder = cl.operands[0];
    if (folder.find("://") != std::string::npos) {
        err << "caliper: install takes a local folder, not " << folder << "\n";
        return 2;
    }
    if (ctx.user_dir.empty()) {
        err << "caliper: the user applets folder is unknown\n";
        return 1;
    }
    std::error_code ec;
    if (fs::is_directory(folder, ec) && fs::exists(ctx.user_dir, ec) &&
        fs::equivalent(folder, ctx.user_dir, ec)) {
        err << "caliper: " << folder << " is the user applets folder already\n";
        return 1;
    }
    AppletLoader L(ctx.caps, ctx.data_root);
    const int failing = cli_detail::check_folder(folder, L, out, err);
    if (failing < 0) return 1;
    if (failing > 0) {
        err << "caliper: nothing installed; " << failing << " of " << L.count()
            << " applets do not pass the checks\n";
        return 1;
    }
    // A spec applet has no library to place; it lives in a workspace.
    for (int i = 0; i < L.count(); i++)
        if (L.at(i).manifest.kind == AppletKind::Spec) {
            err << "caliper: nothing installed; " << L.at(i).manifest.id
                << " is a spec applet, which install does not take\n";
            return 1;
        }
    fs::create_directories(ctx.user_dir, ec);
    if (!fs::is_directory(ctx.user_dir, ec)) {
        err << "caliper: cannot create " << ctx.user_dir << "\n";
        return 1;
    }
    // Per applet: its library, then its asset folder assets/<stem>/ beside
    // the library when there is one (where an applet that reads files from
    // beside its library keeps them), then its manifest last, so a
    // manifest never names a library or assets not yet in place. One
    // transaction for the whole folder: a failure installs nothing.
    const fs::path dest(ctx.user_dir);
    std::vector<cli_detail::InstallCopy> copies;
    for (int i = 0; i < L.count(); i++) {
        const AppletEntry& a = L.at(i);
        const fs::path lib(a.dylib_path), manifest(a.manifest_path);
        std::string stem = manifest.filename().string();
        stem.resize(stem.size() - std::string(".caliper.toml").size());
        copies.push_back({lib, dest / lib.filename()});
        const fs::path assets = lib.parent_path() / "assets" / stem;
        if (fs::is_directory(assets, ec))
            copies.push_back({assets, dest / "assets" / stem});
        copies.push_back({manifest, dest / manifest.filename()});
    }
    std::string why;
    if (!cli_detail::install_all(copies, why)) {
        err << "caliper: nothing installed; " << why << "\n";
        return 1;
    }
    for (int i = 0; i < L.count(); i++)
        out << "installed " << L.at(i).manifest.id << " " << L.at(i).manifest.version
            << " in " << dest.string() << "\n";
    return 0;
}

// ---- caliper new <name> [<dir>] --------------------------------------------

// The names an applet made by `caliper new` gets from one argument: either a
// short name (`my-viewer`, id `dev.example.my-viewer`) or a full id
// (`com.example.my-viewer`, whose last part is the short name).
struct NewAppletNames {
    std::string id;            // com.example.my-viewer
    std::string short_name;    // my-viewer: the default folder
    std::string stem;          // my_viewer: target, library, manifest, source
    std::string display;       // My Viewer: manifest name and window title
    std::string class_name;    // MyViewerApplet
    std::string error;
};

inline NewAppletNames new_applet_names(const std::string& arg) {
    NewAppletNames n;
    const size_t dot = arg.rfind('.');
    n.short_name = dot == std::string::npos ? arg : arg.substr(dot + 1);
    n.id = dot == std::string::npos ? "dev.example." + arg : arg;
    bool ok = !n.short_name.empty() && n.short_name.size() <= 64 &&
              n.short_name[0] >= 'a' && n.short_name[0] <= 'z' &&
              n.short_name.back() != '-' && is_valid_applet_id(n.id);
    for (char c : n.short_name)
        ok = ok && ((c >= 'a' && c <= 'z') || (c >= '0' && c <= '9') || c == '-');
    if (!ok) {
        n.error = "not a valid applet name: " + arg +
                  " (use lowercase letters, digits and '-', starting with a letter, "
                  "or a full id such as com.example.my-viewer)";
        return n;
    }
    // The stem names the CMake targets <stem> and <stem>_tests, so it must
    // not be a name CMake reserves (some generators compare them without
    // case) or the name of the SDK's own target.
    std::string stem = n.short_name;
    std::replace(stem.begin(), stem.end(), '-', '_');
    static const char* const kReserved[] = {
        "all", "all_build", "clean", "depend", "edit_cache", "help", "install",
        "list_install_components", "package", "package_source", "preinstall",
        "rebuild_cache", "run_tests", "test", "zero_check"};
    for (const std::string& target : {stem, stem + "_tests"}) {
        const bool reserved =
            std::find_if(std::begin(kReserved), std::end(kReserved),
                         [&](const char* r) { return target == r; }) != std::end(kReserved) ||
            target.rfind("caliper_", 0) == 0;
        if (reserved) {
            n.error = "not a usable applet name: " + arg + " (the CMake target \"" +
                      target + "\" is reserved; choose another name)";
            return n;
        }
    }
    bool word_start = true;
    for (char c : n.short_name) {
        if (c == '-') {
            n.stem += '_';
            n.display += ' ';
            word_start = true;
            continue;
        }
        n.stem += c;
        const char up = word_start ? (char)std::toupper((unsigned char)c) : c;
        n.display += up;
        n.class_name += up;
        word_start = false;
    }
    n.class_name += "Applet";
    return n;
}

namespace cli_detail {

inline const char* kNewCMakeLists = R"(# @NAME@: a Caliper applet, built against an installed Caliper SDK.
#
#   cmake -S . -B build -DCMAKE_PREFIX_PATH=<sdk prefix>
#   cmake --build build
#   ctest --test-dir build
#   caliper dev build/applets
cmake_minimum_required(VERSION 3.18)
project(@STEM@ LANGUAGES CXX)

find_package(caliper-sdk @SDK_VERSION@CONFIG REQUIRED)

# The library lib@STEM@ (or @STEM@.dll) and its manifest, in build/applets.
# Add TORCH to link the libtorch the host was built with.
caliper_add_applet(@STEM@
    SOURCES @STEM@.cpp
    MANIFEST @STEM@.caliper.toml)

# The applet's own test, on the SDK's fixture host: no window, no host.
enable_testing()
add_executable(@STEM@_tests tests/test_@STEM@.cpp)
target_link_libraries(@STEM@_tests PRIVATE caliper::sdk_testing caliper::ui_stack)
set_target_properties(@STEM@_tests PROPERTIES CXX_STANDARD 20 CXX_STANDARD_REQUIRED ON)
add_test(NAME @STEM@_tests COMMAND @STEM@_tests)
)";

inline const char* kNewManifest = R"([applet]
id      = "@ID@"
name    = "@NAME@"
version = "0.1.0"
summary = "A new Caliper applet."
tag     = "Applet"

[compat]
abi_epoch = @EPOCH@
min_host  = "@HOST_VERSION@"

[services]
required = ["caliper.ui.v1", "caliper.log.v1"]
)";

inline const char* kNewSource = R"(// @NAME@: a Caliper applet. One window with a button and a plot.
#include <caliper/caliper.hpp>
#include <cmath>
#include <vector>

class @CLASS@ final : public caliper::Applet {
public:
    bool on_init(caliper::Host& host) override {
        host_ = &host;
        host.log_info("@STEM@: on_init");
        return true;
    }

    void on_frame(const caliper::Frame& f) override {
        // The window docks in the centre of the host's layout the first time
        // it appears; a layout the user arranged wins.
        caliper::ui::dock_hint(host_, "@NAME@", CALIPER_DOCK_CENTRAL);
        ImGui::Begin("@NAME@");
        ImGui::Text("framebuffer: %d x %d px", f.fb_width, f.fb_height);
        if (ImGui::Button(playing_ ? "Pause" : "Play")) playing_ = !playing_;
        if (playing_) phase_ += (float)f.delta_sec;
        if (ImPlot::BeginPlot("sine", {-1, 220})) {
            std::vector<float> xs(256), ys(256);
            for (int i = 0; i < 256; i++) {
                xs[i] = i / 255.0f * 6.28318f;
                ys[i] = std::sin(xs[i] + phase_);
            }
            ImPlot::PlotLine("sin", xs.data(), ys.data(), 256);
            ImPlot::EndPlot();
        }
        ImGui::End();
    }

    void on_cleanup() override {
        if (host_) host_->log_info("@STEM@: on_cleanup");
    }

private:
    caliper::Host* host_ = nullptr;
    float phase_ = 0.0f;
    bool playing_ = true;
};

// The id, version and services must match @STEM@.caliper.toml.
CALIPER_APPLET(@CLASS@,
    .id       = "@ID@",
    .version  = "0.1.0",
    .name     = "@NAME@",
    .summary  = "A new Caliper applet.",
    .tag      = "Applet",
    .services = {CALIPER_UI_V1, CALIPER_LOG_V1})
)";

inline const char* kNewTest = R"(// @NAME@ on the SDK's fixture host: the descriptor agrees with the manifest,
// and the applet initializes and cleans up without a window.
#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>
#include <caliper/fixture_host.h>
#include <string>

// The applet's source, compiled into the test so the test can drive its
// descriptor directly.
#include "../@STEM@.cpp"

TEST_CASE("@STEM@: descriptor and lifecycle") {
    const CaliperAppletDescriptor* d = caliper_applet_descriptor();
    REQUIRE(d != nullptr);
    CHECK(std::string(d->id) == "@ID@");
    CHECK(std::string(d->version) == "0.1.0");

    caliper::testing::FixtureHost fx;
    void* applet = d->api.create();
    REQUIRE(applet != nullptr);
    CHECK(d->api.initialize(applet, fx.host()));
    CHECK(fx.log_contains("@STEM@: on_init"));
    d->api.cleanup(applet);
    d->api.destroy(applet);
    CHECK(fx.log_contains("@STEM@: on_cleanup"));
}
)";

} // namespace cli_detail

inline int run_new(const CommandLine& cl, const CliContext& ctx, std::ostream& out,
                   std::ostream& err) {
    namespace fs = std::filesystem;
    using cli_detail::replace_all;
    const NewAppletNames n = new_applet_names(cl.operands[0]);
    if (!n.error.empty()) {
        err << "caliper: " << n.error << "\n";
        return 2;
    }
    std::error_code ec;
    const fs::path dir = fs::absolute(
        cl.operands.size() > 1 ? fs::path(cl.operands[1]) : fs::path(n.short_name), ec)
        .lexically_normal();
    if (fs::exists(dir, ec) || fs::is_symlink(dir, ec)) {
        err << "caliper: " << dir.string() << " already exists; nothing written\n";
        return 1;
    }
    fs::create_directories(dir / "tests", ec);
    if (ec) {
        err << "caliper: cannot create " << dir.string() << ": " << ec.message() << "\n";
        return 1;
    }
    // A known SDK version becomes the find_package() floor; "0.2.0 " keeps
    // the template's spacing when it is there and when it is not.
    const std::string sdk = ctx.sdk_version.empty() ? "" : ctx.sdk_version + " ";
    auto fill = [&](std::string t) {
        t = replace_all(t, "@ID@", n.id);
        t = replace_all(t, "@NAME@", n.display);
        t = replace_all(t, "@STEM@", n.stem);
        t = replace_all(t, "@CLASS@", n.class_name);
        t = replace_all(t, "@EPOCH@", std::to_string(ctx.caps.abi_epoch));
        t = replace_all(t, "@HOST_VERSION@", ctx.caps.version);
        t = replace_all(t, "@SDK_VERSION@", sdk);
        return t;
    };
    const std::pair<std::string, const char*> files[] = {
        {"CMakeLists.txt", cli_detail::kNewCMakeLists},
        {n.stem + ".caliper.toml", cli_detail::kNewManifest},
        {n.stem + ".cpp", cli_detail::kNewSource},
        {"tests/test_" + n.stem + ".cpp", cli_detail::kNewTest},
    };
    for (const auto& [name, text] : files) {
        if (!cli_detail::write_file(dir / name, fill(text))) {
            err << "caliper: cannot write " << (dir / name).string() << "\n";
            return 1;
        }
    }
    out << "created " << dir.string() << " for applet " << n.id << "\n";
    for (const auto& [name, text] : files) out << "  " << name << "\n";
    return 0;
}

// ---- caliper info ----------------------------------------------------------
inline int run_info(const CommandLine&, const CliContext& ctx, std::ostream& out,
                    std::ostream&) {
    out << "host " << ctx.caps.version << "\n";
    out << "abi_epoch " << ctx.caps.abi_epoch << "\n";
    out << "sdk " << (ctx.sdk_version.empty() ? "unknown" : ctx.sdk_version) << "\n";
    out << "libtorch "
        << (ctx.caps.torch_version.empty() ? "none" : ctx.caps.torch_version) << "\n";
    out << "ui_stack "
        << (ctx.caps.ui_stack_hash.empty() ? "none" : ctx.caps.ui_stack_hash) << "\n";
    for (const std::string& id : ctx.caps.services) out << "service " << id << "\n";
    return 0;
}

// ---- the same data as JSON (Caliper API v0) ----------------------------------

// What `caliper info` prints, as a JSON object: {"host", "abi_epoch", "sdk",
// "libtorch", "ui_stack", "services"}, with null for what is unknown or absent.
inline std::string info_json(const CliContext& ctx) {
    using caliper_net::Json;
    auto or_null = [](const std::string& s) { return s.empty() ? Json() : Json(s); };
    Json j = Json::object();
    j.set("host", ctx.caps.version);
    j.set("abi_epoch", (int64_t)ctx.caps.abi_epoch);
    j.set("sdk", or_null(ctx.sdk_version));
    j.set("libtorch", or_null(ctx.caps.torch_version));
    j.set("ui_stack", or_null(ctx.caps.ui_stack_hash));
    Json services = Json::array();
    for (const std::string& id : ctx.caps.services) services.push(id);
    j.set("services", std::move(services));
    return j.dump();
}

// What `caliper list` prints for the applets of `L`, in its order, as a JSON
// object {"applets": [{"status", "id", "version", "kind", "folder", "reason"}]}:
// the status word, the version ("" when the manifest has none), the kind
// ("library" or "spec") and the reason that list prints under the id ("" for
// a READY applet).
inline std::string applets_json(const AppletLoader& L) {
    using caliper_net::Json;
    Json list = Json::array();
    for (int i : cli_detail::list_order(L)) {
        const AppletEntry& a = L.at(i);
        Json e = Json::object();
        e.set("status", applet_status_word(a.status));
        e.set("id", cli_detail::list_id(a));
        e.set("version", a.manifest.version);
        e.set("kind", applet_kind_word(a.manifest.kind));
        e.set("folder", a.folder);
        e.set("reason", a.status == AppletStatus::Ready ? std::string() : a.status_text);
        list.push(std::move(e));
    }
    Json j = Json::object();
    j.set("applets", std::move(list));
    return j.dump();
}

// ---- caliper serve -----------------------------------------------------------
namespace cli_detail {
// The hub that SIGINT and SIGTERM stop. Hub::stop() is async-signal-safe.
inline std::atomic<caliper_net::Hub*>& serving_hub() {
    static std::atomic<caliper_net::Hub*> hub{nullptr};
    return hub;
}
// SIGINT and SIGTERM's handler while serve runs: stops serving_hub(). Defined
// in host_cli.cpp, since a function with C linkage is not made inline.
extern "C" void caliper_serve_on_signal(int);
} // namespace cli_detail

// Runs the hub until SIGINT or SIGTERM. Prints the viewer link and the
// producer address, writes the hub file (0600) for producers on this machine
// and removes it on the way out.
inline int run_serve(const CommandLine& cl, const CliContext& ctx, std::ostream& out,
                     std::ostream& err) {
    caliper_net::HubOptions o;
    o.port = (uint16_t)(cl.port >= 0 ? cl.port : kDefaultHubPort);
    o.allow_control = cl.allow_control;
    o.assets_dir = cl.assets_dir;
    o.software_version = ctx.caps.version;
    if (!cl.token_file.empty()) {
        std::string why;
        if (!caliper_net::read_token_file(cl.token_file, &o.token, &why)) {
            err << "caliper: " << why << "\n";
            return 1;
        }
    } else {
        o.token = caliper_net::random_token();
    }
    o.log = [&out](const std::string& line) { out << line << std::endl; };
    caliper_net::Hub hub(o);
    std::string why;
    if (!hub.listen(&why)) {
        err << "caliper: " << why << "\n";
        if (cl.port < 0)
            err << "caliper: is another hub running? Choose a port with --port, or --port 0 "
                   "for any free port\n";
        return 1;
    }
    std::string hub_text;
    if (!ctx.hub_file.empty()) {
#ifdef _WIN32
        const long pid = (long)_getpid();
#else
        const long pid = (long)getpid();
#endif
        hub_text = caliper_net::hub_file_text(hub.port(), hub.token(), pid);
        if (!caliper_net::write_hub_file(ctx.hub_file, hub_text, &why)) {
            err << "caliper: " << why << "; producers must be given the hub's address\n";
            hub_text.clear();
        }
    }
    out << "viewers   " << hub.viewer_url() << "\n";
    out << "producers " << hub.stream_url();
    if (!hub_text.empty()) out << " (address and token in " << ctx.hub_file << ")";
    out << "\n";
    if (!cl.assets_dir.empty()) out << "serving the viewer page from " << cl.assets_dir << "\n";
    out << (cl.allow_control ? "controls are forwarded to the scripts (--allow-control)\n"
                             : "controls are not forwarded; start with --allow-control to allow them\n");
    out << "stop with Ctrl-C" << std::endl;

    cli_detail::serving_hub().store(&hub);
    auto old_int = std::signal(SIGINT, cli_detail::caliper_serve_on_signal);
    auto old_term = std::signal(SIGTERM, cli_detail::caliper_serve_on_signal);
    hub.run();
    std::signal(SIGINT, old_int);
    std::signal(SIGTERM, old_term);
    cli_detail::serving_hub().store(nullptr);
    if (!hub_text.empty()) caliper_net::remove_hub_file(ctx.hub_file, hub_text);
    out << "hub stopped" << std::endl;
    return 0;
}

// ---- caliper build <folder> ----------------------------------------------------
// CMake against the installed SDK, then check (cli_build.h, cli_build.cpp).
int run_build(const CommandLine& cl, const CliContext& ctx, std::ostream& out, std::ostream& err);

// Runs a subcommand that needs no window. Not for Run, Dev or Help, which
// main() handles.
inline int run_subcommand(const CommandLine& cl, const CliContext& ctx,
                          std::ostream& out, std::ostream& err) {
    using C = CommandLine::Command;
    switch (cl.command) {
        case C::List: return run_list(cl, ctx, out, err);
        case C::Check: return run_check(cl, ctx, out, err);
        case C::Install: return run_install(cl, ctx, out, err);
        case C::New: return run_new(cl, ctx, out, err);
        case C::Build: return run_build(cl, ctx, out, err);
        case C::Info: return run_info(cl, ctx, out, err);
        case C::Serve: return run_serve(cl, ctx, out, err);
        default: return 2;
    }
}

} // namespace caliper_host
