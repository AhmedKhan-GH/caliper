#pragma once
// ============================================================================
// The caliper executable's command line. It is parsed before any window,
// AppKit, GLFW or renderer exists, and every subcommand runs and exits
// without starting the UI, so none of them ever shows a window.
//
//   caliper [--applets <dir>]...              start the app
//   caliper dev <folder> [--applets <dir>]... start the app, <folder> searched first
//   caliper list [--applets <dir>]...         list the applets in search order
//   caliper check <folder>                    run the pre-load checks on <folder>
//   caliper install <folder>                  check, then copy into the user folder
//   caliper new <name> [<dir>]                create an applet from a template
//   caliper info                              print versions and served services
//   caliper --help                            print the usage
//
// `--applets <dir>` (or `--applets=<dir>`) adds a development folder to the
// search, before CALIPER_APPLET_PATH and the installed folders; it may be
// given more than once, and the folders are searched in the order given.
//
// Exit codes: 0 success; 1 a check failed, nothing was found, or a file
// operation failed; 2 the command line is wrong (the usage goes to stderr).
//
// Header-only and free of any graphics dependency: main() dispatches here
// and the tests call the same functions with their own CliContext.
// ============================================================================
#include "applet_loader.h"
#include "negotiation.h"
#include <algorithm>
#include <cctype>
#include <filesystem>
#include <fstream>
#include <ostream>
#include <string>
#include <system_error>
#include <vector>

namespace caliper_host {

struct CommandLine {
    enum class Command { Run, Help, Dev, List, Check, Install, New, Info };
    Command command = Command::Run;
    std::vector<std::string> applet_dirs;   // dev's folder, then --applets, in order
    std::vector<std::string> operands;      // the subcommand's operands
    std::string error;                      // set: print it and the usage, exit 2
};

inline const char* usage_text() {
    return "usage: caliper [--applets <dir>]...\n"
           "       caliper dev <folder> [--applets <dir>]...\n"
           "       caliper list [--applets <dir>]...\n"
           "       caliper check <folder>\n"
           "       caliper install <folder>\n"
           "       caliper new <name> [<dir>]\n"
           "       caliper info\n"
           "       caliper --help\n";
}

inline CommandLine parse_command_line(int argc, const char* const* argv) {
    using C = CommandLine::Command;
    CommandLine cl;
    const std::string kApplets = "--applets";
    bool have_command = false;
    for (int i = 1; i < argc; i++) {
        const std::string arg = argv[i] ? argv[i] : "";
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
            if (dir.empty()) {
                cl.error = "--applets needs a folder";
                return cl;
            }
            cl.applet_dirs.push_back(dir);
            continue;
        }
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
            else if (arg == "info") cl.command = C::Info;
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
        case C::Check: operands(1, 1, "check"); break;
        case C::Install: operands(1, 1, "install"); break;
        case C::New: operands(1, 2, "new"); break;
    }
    if (cl.error.empty() && !cl.applet_dirs.empty() &&
        (cl.command == C::Check || cl.command == C::Install ||
         cl.command == C::New || cl.command == C::Info))
        cl.error = "--applets applies only to starting the app, dev and list";
    return cl;
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
};

namespace cli_detail {

namespace fs = std::filesystem;

inline bool passes(const AppletEntry& a) {
    return a.status == AppletStatus::Ready || a.status == AppletStatus::Shadowed;
}

// One line per applet: status, id, version, folder, in columns; a refused,
// failed or shadowed applet's reason on the next line, under the id.
inline void print_entries(const AppletLoader& L, std::ostream& out) {
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
    auto id_of = [](const AppletEntry& a) {
        return a.manifest.id.empty() ? a.manifest.name : a.manifest.id;
    };
    auto version_of = [](const AppletEntry& a) {
        return a.manifest.version.empty() ? std::string("-") : a.manifest.version;
    };
    size_t ws = 0, wi = 0, wv = 0;
    for (int i : order) {
        ws = std::max(ws, std::string(applet_status_word(L.at(i).status)).size());
        wi = std::max(wi, id_of(L.at(i)).size());
        wv = std::max(wv, version_of(L.at(i)).size());
    }
    auto pad = [](const std::string& s, size_t w) {
        return s + std::string(w > s.size() ? w - s.size() : 0, ' ');
    };
    for (int i : order) {
        const AppletEntry& a = L.at(i);
        out << pad(applet_status_word(a.status), ws) << "  " << pad(id_of(a), wi)
            << "  " << pad(version_of(a), wv) << "  " << a.folder << "\n";
        if (a.status != AppletStatus::Ready && !a.status_text.empty())
            out << std::string(ws + 2, ' ') << a.status_text << "\n";
    }
}

// Scans one folder and prints it; the number of applets that do not pass.
inline int check_folder(const std::string& folder, AppletLoader& L,
                        std::ostream& out, std::ostream& err) {
    std::error_code ec;
    if (!fs::is_directory(folder, ec)) {
        err << "caliper: not a folder: " << folder << "\n";
        return -1;
    }
    // Absolute, as list prints folders.
    const auto dirs = applet_search_dirs({folder}, nullptr, "", "");
    L.scan(dirs);
    if (L.count() == 0) {
        err << "caliper: no applets in " << dirs.front() << "\n";
        return -1;
    }
    print_entries(L, out);
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

// Copies a file through a temporary copy renamed over the destination, so
// a library is never seen half-written.
inline bool copy_file_atomic(const fs::path& from, const fs::path& to) {
    const fs::path tmp = to.string() + ".tmp";
    std::error_code ec;
    fs::copy_file(from, tmp, fs::copy_options::overwrite_existing, ec);
    if (ec) return false;
    fs::rename(tmp, to, ec);
    if (ec) fs::remove(tmp, ec);
    return !ec;
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
    fs::create_directories(ctx.user_dir, ec);
    if (!fs::is_directory(ctx.user_dir, ec)) {
        err << "caliper: cannot create " << ctx.user_dir << "\n";
        return 1;
    }
    const fs::path dest(ctx.user_dir);
    for (int i = 0; i < L.count(); i++) {
        const AppletEntry& a = L.at(i);
        const fs::path lib(a.dylib_path), manifest(a.manifest_path);
        // The library first: a manifest never names a library not yet there.
        if (!cli_detail::copy_file_atomic(lib, dest / lib.filename()) ||
            !cli_detail::copy_file_atomic(manifest, dest / manifest.filename())) {
            err << "caliper: could not copy " << a.manifest.id << " into "
                << dest.string() << "\n";
            return 1;
        }
        out << "installed " << a.manifest.id << " " << a.manifest.version << " in "
            << dest.string() << "\n";
    }
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
        case C::Info: return run_info(cl, ctx, out, err);
        default: return 2;
    }
}

} // namespace caliper_host
