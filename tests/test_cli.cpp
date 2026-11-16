// The caliper command line's subcommands (src/host/host_cli.h), called as
// main() calls them but with a CliContext of temporary folders, so nothing
// touches the user's applets and no window ever opens. `dev` starts the app,
// so only its parsing is tested.
#include <doctest/doctest.h>
#include "cli_build.h"
#include "host_cli.h"
#include "net/wire.h"
#include "net/ws.h"
#include "test_proc.h"
#include "test_temp.h"
#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <functional>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#ifndef _WIN32
#include <arpa/inet.h>
#include <fcntl.h>
#include <netinet/in.h>
#include <poll.h>
#include <signal.h>
#include <spawn.h>
#include <sys/socket.h>
#include <sys/wait.h>
#include <unistd.h>
extern char** environ;
#endif
namespace fs = std::filesystem;
using namespace caliper_host;
using Command = CommandLine::Command;

namespace {

std::string lib_name(const std::string& stem) {
#ifdef _WIN32
    return stem + ".dll";
#elif defined(__APPLE__)
    return "lib" + stem + ".dylib";
#else
    return "lib" + stem + ".so";
#endif
}

// A new folder of this run's own (test_temp.h), removed when the case ends.
struct TempDir {
    fs::path path;
    explicit TempDir(const std::string& name)
        : path(caliper_test::unique_temp_path("caliper-cli-" + name)) {
        fs::create_directories(path);
    }
    ~TempDir() {
        std::error_code ec;
        fs::remove_all(path, ec);
    }
    std::string str(const std::string& sub) const { return (path / sub).string(); }
};

void fake_applet(const fs::path& dir, const std::string& stem, const std::string& id,
                 const std::string& version, const std::string& extra = "") {
    fs::create_directories(dir);
    std::ofstream(dir / (stem + ".caliper.toml")) <<
        "[applet]\nid=\"" << id << "\"\nname=\"Example\"\nversion=\"" << version <<
        "\"\n[compat]\nabi_epoch=2\n" << extra;
    std::ofstream(dir / lib_name(stem)) << "library bytes";
}

CommandLine parse(std::vector<const char*> args) {
    args.insert(args.begin(), "caliper");
    return parse_command_line((int)args.size(), args.data());
}

struct Run {
    int code = -1;
    std::string out, err;
};

Run run(const std::vector<const char*>& args, const CliContext& ctx) {
    const CommandLine cl = parse(args);
    REQUIRE_MESSAGE(cl.error.empty(), cl.error);
    std::ostringstream out, err;
    Run r;
    r.code = run_subcommand(cl, ctx, out, err);
    r.out = out.str();
    r.err = err.str();
    return r;
}

CliContext context(const TempDir& t) {
    CliContext c;
    c.caps = HostCaps{2, "0.6.0", {"caliper.ui.v1", "caliper.log.v1"}, "", ""};
    c.sdk_version = "0.2.0";
    c.user_dir = t.str("user");
    c.shipped_dir = t.str("shipped");
    c.data_root = t.str("data");
    return c;
}

std::string slurp(const fs::path& p) {
    std::ifstream f(p);
    std::stringstream ss;
    ss << f.rdbuf();
    return ss.str();
}

} // namespace

TEST_CASE("cli: --allow-control applies to serve, and to the app, dev and shot with --api") {
    CHECK(parse({"serve", "--allow-control"}).allow_control);
    CHECK_FALSE(parse({"serve"}).allow_control);
    CHECK(parse({"--api", "--allow-control"}).allow_control);
    CHECK(parse({"dev", "/a", "--api", "--allow-control"}).error.empty());
    CHECK(parse({"--allow-control"}).error == "--allow-control needs --api when starting the app, dev or shot");
    CHECK(parse({"list", "--allow-control"}).error ==
          "--allow-control applies only to serve, and to the app, dev and shot with --api");
    CHECK(parse({"serve", "--allow-control=yes"}).error == "--allow-control takes no value");
}

TEST_CASE("cli: --hidden starts the app or dev with the window never shown") {
    CHECK(parse({"--hidden"}).hidden);
    CHECK(parse({"--hidden"}).command == Command::Run);
    CHECK_FALSE(parse({}).hidden);
    auto dev = parse({"dev", "/a", "--hidden", "--api"});
    CHECK(dev.error.empty());
    CHECK(dev.hidden);
    CHECK(dev.api);
    CHECK(parse({"list", "--hidden"}).error == "--hidden applies only to starting the app and dev");
    CHECK(parse({"--hidden=1"}).error == "--hidden takes no value");
}

TEST_CASE("cli: subcommands and their operands parse") {
    CHECK(parse({}).command == Command::Run);
    CHECK(parse({"--help"}).command == Command::Help);
    CHECK(parse({"help"}).command == Command::Help);
    CHECK(parse({"list", "--help"}).command == Command::Help);

    auto dev = parse({"--applets", "/b", "dev", "/a", "--applets=/c"});
    CHECK(dev.error.empty());
    CHECK(dev.command == Command::Dev);
    CHECK(dev.applet_dirs == std::vector<std::string>{"/a", "/b", "/c"});   // dev's first

    auto list = parse({"list", "--applets", "/x"});
    CHECK(list.command == Command::List);
    CHECK(list.applet_dirs == std::vector<std::string>{"/x"});

    CHECK(parse({"check", "/f"}).operands == std::vector<std::string>{"/f"});
    CHECK(parse({"install", "/f"}).command == Command::Install);
    CHECK(parse({"info"}).command == Command::Info);
    auto made = parse({"new", "my-viewer", "/tmp/x"});
    CHECK(made.command == Command::New);
    CHECK(made.operands == std::vector<std::string>{"my-viewer", "/tmp/x"});

    auto serve = parse({"serve"});
    CHECK(serve.error.empty());
    CHECK(serve.command == Command::Serve);
    CHECK(serve.port == -1);
    auto served = parse({"serve", "--port", "0", "--token-file=/t", "--assets", "/web"});
    CHECK(served.error.empty());
    CHECK(served.port == 0);
    CHECK(served.token_file == "/t");
    CHECK(served.assets_dir == "/web");
    CHECK(parse({"--port=8080", "serve"}).port == 8080);
}

TEST_CASE("cli: wrong command lines are usage errors") {
    CHECK(parse({"frobnicate"}).error == "unknown command: frobnicate");
    CHECK(parse({"-x"}).error == "unknown option: -x");
    CHECK(parse({"dev"}).error == "dev takes one folder");
    CHECK(parse({"dev", "a", "b"}).error == "dev takes one folder");
    CHECK(parse({"check"}).error == "check takes one folder");
    CHECK(parse({"install", "a", "b"}).error == "install takes one folder");
    CHECK(parse({"new"}).error == "new takes a name and an optional folder");
    CHECK(parse({"new", "a", "b", "c"}).error == "new takes a name and an optional folder");
    CHECK(parse({"info", "x"}).error == "info takes no operands");
    CHECK(parse({"list", "x"}).error == "list takes no operands");
    CHECK(parse({"check", "/f", "--applets", "/x"}).error ==
          "--applets applies only to starting the app, dev, list and shot");
    CHECK(parse({"--applets"}).error == "--applets needs a folder");
    CHECK(parse({"--applets", "--help"}).error == "--applets needs a folder");
    CHECK(parse({"list", "--applets=--x"}).error == "--applets needs a folder");
    CHECK(parse({"serve", "x"}).error == "serve takes no operands");
    CHECK(parse({"serve", "--port"}).error == "--port needs a number");
    CHECK(parse({"serve", "--port", "65536"}).error == "--port needs a number from 0 to 65535, not 65536");
    CHECK(parse({"serve", "--port", "-1"}).error == "--port needs a number from 0 to 65535, not -1");
    CHECK(parse({"serve", "--token-file"}).error == "--token-file needs a file");
    CHECK(parse({"serve", "--assets="}).error == "--assets needs a folder");
    CHECK(parse({"list", "--port", "1"}).error == "--port applies only to serve");
    CHECK(parse({"serve", "--applets", "/x"}).error ==
          "--applets applies only to starting the app, dev, list and shot");
}

TEST_CASE("cli: shot takes its options in either spelling, with defaults") {
    auto shot = parse({"shot", "--applets", "/a", "--launch", "com.example.viewer",
                       "--out", "/tmp/v.png"});
    CHECK(shot.error.empty());
    CHECK(shot.command == Command::Shot);
    CHECK(shot.applet_dirs == std::vector<std::string>{"/a"});
    CHECK(shot.launch_id == "com.example.viewer");
    CHECK(shot.out == "/tmp/v.png");
    CHECK(shot.width == 1600);
    CHECK(shot.height == 1000);
    CHECK(shot.frames == 60);

    auto sized = parse({"--size=800x600", "shot", "--launch=x.y.z", "--frames=5",
                        "--out=o.png", "--applets=/b"});
    CHECK(sized.error.empty());
    CHECK(sized.command == Command::Shot);
    CHECK(sized.width == 800);
    CHECK(sized.height == 600);
    CHECK(sized.frames == 5);
    CHECK(sized.launch_id == "x.y.z");
    CHECK(sized.out == "o.png");
    CHECK(sized.applet_dirs == std::vector<std::string>{"/b"});

    auto largest = parse({"shot", "--launch", "x", "--size", "16384x1", "--out", "o.png"});
    CHECK(largest.error.empty());
    CHECK(largest.width == kShotMaxSide);
    CHECK(largest.height == 1);
    CHECK(parse({"shot", "--help"}).command == Command::Help);
}

TEST_CASE("cli: wrong shot command lines are usage errors") {
    CHECK(parse({"shot", "--out", "o.png"}).error == "shot needs --launch <id>");
    CHECK(parse({"shot", "--launch", "x"}).error == "shot needs --out <file.png>");
    CHECK(parse({"shot", "x", "--launch", "x", "--out", "o.png"}).error ==
          "shot takes no operands");
    CHECK(parse({"shot", "--launch"}).error == "--launch needs an applet id");
    CHECK(parse({"shot", "--launch", "--out", "o.png"}).error == "--launch needs an applet id");
    CHECK(parse({"shot", "--launch", "x", "--out"}).error == "--out needs a file");
    CHECK(parse({"shot", "--frames="}).error == "--frames needs a number of frames");
    CHECK(parse({"shot", "--size"}).error == "--size needs a size such as 1600x1000");
    CHECK(parse({"shot", "--launch", "x", "--launch", "y"}).error == "--launch is given twice");
    for (const char* bad : {"0x10", "10x0", "10", "x10", "10x", "10x10x10", "-5x10",
                            "10 x10", "16385x10", "abcxdef", "1e3x10"}) {
        CAPTURE(bad);
        CHECK(parse({"shot", "--size", bad}).error.rfind("--size takes <W>x<H>", 0) == 0);
    }
    for (const char* bad : {"0", "-1", "100001", "ten", "1.5", "+3"}) {
        CAPTURE(bad);
        CHECK(parse({"shot", "--frames", bad}).error.rfind("--frames takes a whole number", 0) == 0);
    }
    CHECK(parse({"list", "--frames", "3"}).error == "--frames applies only to shot");
    CHECK(parse({"--launch", "x"}).error == "--launch applies only to shot");
    CHECK(parse({"info", "--out", "o.png", "--size", "1x1"}).error == "--out applies only to shot");
}

TEST_CASE("cli: the usage names shot") {
    CHECK(std::string(usage_text()).find("caliper shot [--applets <dir>]... --launch <id>") !=
          std::string::npos);
}

TEST_CASE("cli: the -psn_ argument Finder passes is ignored") {
    auto cl = parse({"-psn_0_12345"});
    CHECK(cl.error.empty());
    CHECK(cl.command == Command::Run);
    auto dev = parse({"dev", "-psn_0_1", "/a"});
    CHECK(dev.error.empty());
    CHECK(dev.applet_dirs == std::vector<std::string>{"/a"});
}

// Not on Windows, which refuses serve before it runs: a Hub opens a socket
// even when it fails before listening (caliper_net_tests in tests/CMakeLists.txt).
#ifndef _WIN32
TEST_CASE("cli: serve refuses a token file it cannot use, before it listens") {
    TempDir t("serve-refuse");
    CommandLine cl = parse({"serve", "--port", "0"});
    cl.token_file = t.str("missing");
    CliContext ctx;
    std::ostringstream out, err;
    CHECK(run_serve(cl, ctx, out, err) == 1);
    CHECK(err.str().find("cannot read the token file") != std::string::npos);
    std::ofstream(t.str("short")) << "too-short\n";
    fs::permissions(t.str("short"), fs::perms::owner_read | fs::perms::owner_write);
    cl.token_file = t.str("short");
    std::ostringstream out2, err2;
    CHECK(run_serve(cl, ctx, out2, err2) == 1);
    CHECK(err2.str().find("does not start with a session token") != std::string::npos);
    cl.token_file.clear();
    cl.assets_dir = t.str("no-such-folder");
    std::ostringstream out3, err3;
    CHECK(run_serve(cl, ctx, out3, err3) == 1);
    CHECK(err3.str().find("not a folder") != std::string::npos);
    CHECK(out.str().empty());
}
#endif

TEST_CASE("cli: info prints the versions and every service") {
    TempDir t("info");
    CliContext ctx = context(t);
    ctx.caps.torch_version = "2.5.1";
    ctx.caps.ui_stack_hash = "0123456789abcdef";
    const Run r = run({"info"}, ctx);
    CHECK(r.code == 0);
    CHECK(r.out == "host 0.6.0\nabi_epoch 2\nsdk 0.2.0\nlibtorch 2.5.1\n"
                   "ui_stack 0123456789abcdef\n"
                   "service caliper.log.v1\nservice caliper.ui.v1\n");
    ctx.sdk_version.clear();
    ctx.caps.torch_version.clear();
    ctx.caps.ui_stack_hash.clear();
    const Run bare = run({"info"}, ctx);
    CHECK(bare.out.find("sdk unknown\nlibtorch none\nui_stack none\n") != std::string::npos);
}

TEST_CASE("cli: list shows every copy in search order, with reasons") {
    TempDir t("list");
    const CliContext ctx = context(t);
    CHECK(run({"list"}, ctx).out == "no applets found\n");

    fake_applet(t.path / "dev", "x", "x.app", "2.0.0", "torch=\"9.9.9\"\n");
    fake_applet(t.path / "user", "x", "x.app", "1.0.0");
    fake_applet(t.path / "shipped", "x", "x.app", "1.0.0");
    const std::string dev = t.str("dev");
    const Run r = run({"list", "--applets", dev.c_str()}, ctx);
    CHECK(r.code == 0);
    const std::string expected =
        "REFUSED   x.app  2.0.0  " + dev + "\n"
        "          Built against libtorch 9.9.9; this host provides no libtorch.\n"
        "READY     x.app  1.0.0  " + t.str("user") + "\n"
        "SHADOWED  x.app  1.0.0  " + t.str("shipped") + "\n"
        "          Version 1.0.0 in " + t.str("user") + " is used instead.\n";
    CHECK(r.out == expected);
}

TEST_CASE("cli: --api and --api-port start the app, dev and shot with Caliper API") {
    CHECK_FALSE(parse({}).api);
    auto app = parse({"--api"});
    CHECK(app.error.empty());
    CHECK(app.command == Command::Run);
    CHECK(app.api);
    CHECK(app.api_port == -1);
    auto dev = parse({"dev", "/a", "--api", "--api-port", "0"});
    CHECK(dev.error.empty());
    CHECK(dev.api);
    CHECK(dev.api_port == 0);
    auto shot = parse({"shot", "--launch", "x", "--out", "o.png", "--api-port=7400", "--api"});
    CHECK(shot.error.empty());
    CHECK(shot.api);
    CHECK(shot.api_port == 7400);
    CHECK(parse({"--api", "--help"}).command == Command::Help);
}

TEST_CASE("cli: wrong --api command lines are usage errors") {
    CHECK(parse({"list", "--api"}).error == "--api applies only to starting the app, dev and shot");
    CHECK(parse({"serve", "--api"}).error == "--api applies only to starting the app, dev and shot");
    CHECK(parse({"info", "--api-port", "1"}).error ==
          "--api-port applies only to starting the app, dev and shot");
    CHECK(parse({"--api-port", "7400"}).error == "--api-port needs --api");
    CHECK(parse({"--api", "--api-port"}).error == "--api-port needs a number");
    CHECK(parse({"--api", "--api-port", "--help"}).error == "--api-port needs a number");
    CHECK(parse({"--api", "--api-port", "65536"}).error ==
          "--api-port needs a number from 0 to 65535, not 65536");
    CHECK(parse({"--api", "--api-port", "-1"}).error ==
          "--api-port needs a number from 0 to 65535, not -1");
    CHECK(parse({"--api", "--api-port", "1", "--api-port", "2"}).error == "--api-port is given twice");
    CHECK(parse({"--api=7400"}).error == "--api takes no value; choose the port with --api-port <n>");
}

TEST_CASE("cli: --url attaches the app or dev to a stream at start") {
    auto app = parse({"--url", "ws://127.0.0.1:7340/v3/stream?token=abc"});
    CHECK(app.error.empty());
    CHECK(app.command == Command::Run);
    CHECK(app.url == "ws://127.0.0.1:7340/v3/stream?token=abc");
    auto dev = parse({"dev", "/a", "--url=http://127.0.0.1:8000/?token=abc", "--hidden"});
    CHECK(dev.error.empty());
    CHECK(dev.url == "http://127.0.0.1:8000/?token=abc");
    CHECK(parse({"--url"}).error == "--url needs a stream link");
    CHECK(parse({"--url", "--hidden"}).error == "--url needs a stream link");
    CHECK(parse({"--url", "a", "--url", "b"}).error == "--url is given twice");
    CHECK(parse({"list", "--url", "a"}).error == "--url applies only to starting the app and dev");
    CHECK(parse({"shot", "--launch", "x", "--out", "o.png", "--url", "a"}).error ==
          "--url applies only to starting the app and dev");
}

TEST_CASE("cli: the usage names --url, --api, --allow-control and --hidden") {
    const std::string u = usage_text();
    CHECK(u.find("usage: caliper [--applets <dir>]... [--url <link>] [--api [--api-port <n>] [--allow-control]] "
                 "[--hidden]") != std::string::npos);
    CHECK(u.find("caliper dev <folder> [--applets <dir>]... [--url <link>] [--api [--api-port <n>] "
                 "[--allow-control]]\n                   [--hidden]") != std::string::npos);
    CHECK(u.find("--out <file.png> [--api [--api-port <n>] [--allow-control]]") != std::string::npos);
    CHECK(u.find("caliper serve [--port <n>] [--token-file <file>] [--assets <dir>] [--allow-control]") !=
          std::string::npos);
}

TEST_CASE("cli: info and list as JSON hold what the text forms print") {
    TempDir t("json");
    CliContext ctx = context(t);
    ctx.caps.torch_version = "2.5.1";
    CHECK(info_json(ctx) ==
          "{\"abi_epoch\":2,\"host\":\"0.6.0\",\"libtorch\":\"2.5.1\",\"sdk\":\"0.2.0\","
          "\"services\":[\"caliper.log.v1\",\"caliper.ui.v1\"],\"ui_stack\":null}");

    fake_applet(t.path / "dev", "x", "x.app", "2.0.0", "torch=\"9.9.9\"\n");
    fake_applet(t.path / "user", "x", "x.app", "1.0.0");
    ctx = context(t);   // a host without libtorch, as in the list test
    AppletLoader L(ctx.caps, ctx.data_root);
    L.scan(applet_search_dirs({t.str("dev")}, nullptr, ctx.user_dir, ctx.shipped_dir));
    caliper_net::Json j;
    std::string err;
    const std::string text = applets_json(L);
    REQUIRE_MESSAGE(caliper_net::Json::parse(text.data(), text.size(), &j, &err), err);
    const caliper_net::Json* list = j.find("applets");
    REQUIRE(list);
    REQUIRE(list->size() == 2);
    const caliper_net::Json& refused = list->at(0);
    CHECK(refused.find("status")->as_string() == "REFUSED");
    CHECK(refused.find("id")->as_string() == "x.app");
    CHECK(refused.find("version")->as_string() == "2.0.0");
    CHECK(refused.find("folder")->as_string() == t.str("dev"));
    CHECK(refused.find("reason")->as_string() ==
          "Built against libtorch 9.9.9; this host provides no libtorch.");
    CHECK(list->at(1).find("status")->as_string() == "READY");
    CHECK(list->at(1).find("reason")->as_string().empty());
}

TEST_CASE("cli: Windows refuses serve, build, --api and --url with one message, and nothing else") {
    auto refused = [](std::vector<const char*> args) {
        const CommandLine cl = parse(args);
        REQUIRE_MESSAGE(cl.error.empty(), cl.error);
        const char* what = windows_unavailable(cl);
        return what ? std::string(what) : std::string();
    };
    CHECK(refused({"serve"}) == "caliper serve");
    CHECK(refused({"serve", "--port", "0", "--allow-control"}) == "caliper serve");
    CHECK(refused({"build", "/a"}) == "caliper build");
    CHECK(refused({"--api"}) == "Caliper API (--api)");
    CHECK(refused({"dev", "/a", "--api", "--api-port", "0"}) == "Caliper API (--api)");
    CHECK(refused({"shot", "--launch", "x", "--out", "o.png", "--api"}) == "Caliper API (--api)");
    CHECK(refused({"--url", "ws://127.0.0.1:1/v3/stream?token=t"}) == "attaching to a stream (--url)");
    for (std::vector<const char*> fine : {std::vector<const char*>{}, {"dev", "/a"}, {"--hidden"},
                                          {"list"}, {"check", "/a"}, {"install", "/a"}, {"new", "x"},
                                          {"info"}, {"shot", "--launch", "x", "--out", "o.png"},
                                          {"--help"}}) {
        CAPTURE(fine.size());
        CHECK(refused(fine).empty());
    }
    CHECK(windows_unavailable_message("caliper serve") == "caliper serve is not available on Windows yet");
}

TEST_CASE("cli: build takes one folder and no --applets") {
    auto b = parse({"build", "/a"});
    CHECK(b.error.empty());
    CHECK(b.command == Command::Build);
    CHECK(b.operands == std::vector<std::string>{"/a"});
    CHECK(parse({"build"}).error == "build takes one folder");
    CHECK(parse({"build", "/a", "/b"}).error == "build takes one folder");
    CHECK(parse({"build", "/a", "--applets", "/x"}).error ==
          "--applets applies only to starting the app, dev, list and shot");
    CHECK(parse({"build", "/a", "--api"}).error == "--api applies only to starting the app, dev and shot");
    CHECK(std::string(usage_text()).find("caliper build <folder>\n") != std::string::npos);
}

TEST_CASE("cli: build finds the SDK prefix in its order") {
    TempDir t("build-sdk");
    auto make_sdk = [&](const fs::path& prefix) {
        fs::create_directories(prefix / "lib/cmake/caliper-sdk");
        std::ofstream(prefix / sdk_config_file()) << "# caliper-sdk\n";
    };
    const fs::path exe_dir = t.path / "prefix-a" / "bin";
    fs::create_directories(exe_dir);
    const std::string home = t.str("home");
    // Paths compare as paths: the same file when both exist, else equal once
    // normalised, so the separator and its spelling do not matter.
    auto same = [](const std::string& got, const fs::path& want) {
        std::error_code ec;
        if (fs::exists(got, ec) && fs::exists(want, ec)) return fs::equivalent(got, want, ec);
        return fs::path(got).lexically_normal() == want.lexically_normal();
    };

    // Nothing anywhere: every candidate is tried, none is chosen.
    SdkSearch none = find_sdk_prefix(nullptr, exe_dir.string(), home.c_str());
    CHECK(none.prefix.empty());
    CHECK(none.error.empty());
    REQUIRE(none.tried.size() == 3);
    CHECK(same(none.tried[0], t.path / "prefix-a"));
    CHECK(same(none.tried[1], exe_dir));
    CHECK(same(none.tried[2], fs::path(home) / "caliper-sdk"));
    // The home folder's caliper-sdk, then the executable's prefix before it.
    make_sdk(fs::path(home) / "caliper-sdk");
    CHECK(same(find_sdk_prefix(nullptr, exe_dir.string(), home.c_str()).prefix, fs::path(home) / "caliper-sdk"));
    make_sdk(t.path / "prefix-a");
    CHECK(same(find_sdk_prefix("", exe_dir.string(), home.c_str()).prefix, t.path / "prefix-a"));
    // CALIPER_SDK_DIR wins, and must hold the SDK.
    make_sdk(t.path / "named");
    CHECK(same(find_sdk_prefix(t.str("named").c_str(), exe_dir.string(), home.c_str()).prefix, t.path / "named"));
    SdkSearch wrong = find_sdk_prefix(t.str("empty").c_str(), exe_dir.string(), home.c_str());
    CHECK(wrong.prefix.empty());
    CHECK(wrong.error.find("CALIPER_SDK_DIR is " + t.str("empty") + ", which holds no Caliper SDK") == 0);
    // No executable folder and no home: nothing to try.
    CHECK(find_sdk_prefix(nullptr, "", nullptr).tried.empty());
}

TEST_CASE("cli: build finds cmake by CALIPER_CMAKE, then PATH, then the fallbacks") {
    TempDir t("build-cmake");
    auto make_program = [&](const fs::path& p, bool executable) {
        fs::create_directories(p.parent_path());
        std::ofstream(p) << "#!/bin/sh\n";
        fs::permissions(p, executable ? fs::perms::owner_all : fs::perms::owner_read | fs::perms::owner_write);
    };
#ifdef _WIN32
    const char* name = "cmake.exe";
    const std::string sep = ";";
#else
    const char* name = "cmake";
    const std::string sep = ":";
#endif
    make_program(t.path / "one" / name, false);   // not executable: skipped on POSIX
    make_program(t.path / "two" / name, true);
    make_program(t.path / "app" / "cmake", true);
    std::string err;
    const std::string path = t.str("missing") + sep + sep + t.str("one") + sep + t.str("two");
#ifndef _WIN32
    CHECK(find_cmake(nullptr, path.c_str(), {}, &err) == (t.path / "two" / name).string());
#endif
    CHECK(find_cmake("", t.str("missing").c_str(), {t.str("app/cmake")}, &err) == t.str("app/cmake"));
    CHECK(find_cmake(t.str("app/cmake").c_str(), path.c_str(), {}, &err) == t.str("app/cmake"));
    CHECK(find_cmake(t.str("nope").c_str(), path.c_str(), {}, &err).empty());
    CHECK(err == "CALIPER_CMAKE is " + t.str("nope") + ", which is not a program");
    CHECK(find_cmake(nullptr, t.str("missing").c_str(), {t.str("nope")}, &err).empty());
    CHECK(err.rfind("cmake was not found", 0) == 0);
    CHECK(find_cmake(nullptr, nullptr, {}, &err).empty());
}

TEST_CASE("cli: build reads what a failed configure lacked from CMake's output") {
    CHECK(configure_lack("-- The CXX compiler identification is unknown\nCMake Error ...") ==
          BuildLack::Compiler);
    CHECK(configure_lack("CMake Error at CMakeLists.txt:8 (project):\n  No CMAKE_CXX_COMPILER could be "
                         "found.") == BuildLack::Compiler);
    CHECK(configure_lack("The CMAKE_CXX_COMPILER:\n\n    /x/c++\n\n  is not a full path to an existing "
                         "compiler tool.") == BuildLack::Compiler);
    CHECK(configure_lack("CMake Error: CMake was unable to find a build program corresponding to \"Ninja\".") ==
          BuildLack::BuildTool);
    CHECK(configure_lack("  Could not find a package configuration file provided by \"caliper-sdk\"\n"
                         "  (requested version 0.2.0) with any of the following names:") == BuildLack::Sdk);
    CHECK(configure_lack("  Could not find a configuration file for package \"caliper-sdk\" that is "
                         "compatible with requested version \"0.3.0\".") == BuildLack::SdkVersion);
    CHECK(configure_lack("CMake Error at CMakeLists.txt:12 (caliper_add_applet): abi_epoch") ==
          BuildLack::Unknown);
}

TEST_CASE("cli: build refuses what is not an applet folder before it runs anything") {
    TempDir t("build-refuse");
    const CliContext ctx = context(t);
    const std::string missing = t.str("missing");
    CommandLine cl = parse({"build", missing.c_str()});
    std::ostringstream out, err;
    CHECK(run_subcommand(cl, ctx, out, err) == 1);
    CHECK(err.str() == "caliper: not a folder: " + missing + "\n");
    fs::create_directories(t.path / "plain");
    const std::string plain = t.str("plain");
    cl = parse({"build", plain.c_str()});
    std::ostringstream out2, err2;
    CHECK(run_subcommand(cl, ctx, out2, err2) == 1);
    CHECK(err2.str().find("has no CMakeLists.txt") != std::string::npos);
    CHECK(out.str().empty());
    CHECK(out2.str().empty());
}

TEST_CASE("cli: check passes a good folder and fails a bad one") {
    TempDir t("check");
    const CliContext ctx = context(t);
    fake_applet(t.path / "good", "a", "x.a", "1.0.0");
    fake_applet(t.path / "good", "b", "x.b", "0.3.0");
    const std::string good = t.str("good");
    const Run ok = run({"check", good.c_str()}, ctx);
    CHECK(ok.code == 0);
    CHECK(ok.out == "READY  x.a  1.0.0  " + good + "\n"
                    "READY  x.b  0.3.0  " + good + "\n"
                    "2 of 2 applets pass the checks\n");

    fake_applet(t.path / "bad", "a", "x.a", "1.0.0", "min_host=\"9.0.0\"\n");
    fake_applet(t.path / "bad", "b", "x.b", "1.0.0", "[build]\nui_stack=\"ffff\"\n");
    std::ofstream(t.path / "bad" / "c.caliper.toml") << "[applet]\nid=\"x.c\"\n";
    CliContext pinned = ctx;
    pinned.caps.ui_stack_hash = "0000";
    const std::string bad = t.str("bad");
    const Run no = run({"check", bad.c_str()}, pinned);
    CHECK(no.code == 1);
    CHECK(no.out.find("REFUSED  x.a") != std::string::npos);
    CHECK(no.out.find("Requires host 9.0.0 or newer; this host is 0.6.0.") != std::string::npos);
    CHECK(no.out.find("REFUSED  x.b") != std::string::npos);
    CHECK(no.out.find("Built against UI stack ffff; this host uses 0000") != std::string::npos);
    CHECK(no.out.find("FAILED   c.caliper.toml") != std::string::npos);
    CHECK(no.out.find("manifest missing applet.name") != std::string::npos);
    CHECK(no.out.find("0 of 3 applets pass the checks") != std::string::npos);

    const Run missing = run({"check", t.str("nowhere").c_str()}, ctx);
    CHECK(missing.code == 1);
    CHECK(missing.err == "caliper: not a folder: " + t.str("nowhere") + "\n");
    fs::create_directories(t.path / "empty");
    const Run empty = run({"check", t.str("empty").c_str()}, ctx);
    CHECK(empty.code == 1);
    CHECK(empty.err == "caliper: no applets in " + t.str("empty") + "\n");
}

TEST_CASE("cli: install copies passing applets into the user folder, or nothing") {
    TempDir t("install");
    const CliContext ctx = context(t);
    fake_applet(t.path / "build", "a", "x.a", "1.0.0");
    const std::string build = t.str("build");
    const Run r = run({"install", build.c_str()}, ctx);
    CHECK(r.code == 0);
    CHECK(r.out.find("installed x.a 1.0.0 in " + t.str("user") + "\n") != std::string::npos);
    CHECK(slurp(t.path / "user" / "a.caliper.toml").find("x.a") != std::string::npos);
    CHECK(slurp(t.path / "user" / lib_name("a")) == "library bytes");
    CHECK_FALSE(fs::exists(t.path / "user" / (lib_name("a") + ".tmp")));

    // An update replaces the installed copy.
    fake_applet(t.path / "build", "a", "x.a", "1.1.0");
    CHECK(run({"install", build.c_str()}, ctx).code == 0);
    CHECK(slurp(t.path / "user" / "a.caliper.toml").find("1.1.0") != std::string::npos);

    // One applet that does not pass: nothing is copied.
    fake_applet(t.path / "mixed", "b", "x.b", "1.0.0");
    fake_applet(t.path / "mixed", "c", "x.c", "1.0.0", "torch=\"9.9.9\"\n");
    const Run no = run({"install", t.str("mixed").c_str()}, ctx);
    CHECK(no.code == 1);
    CHECK(no.err == "caliper: nothing installed; 1 of 2 applets do not pass the checks\n");
    CHECK_FALSE(fs::exists(t.path / "user" / "b.caliper.toml"));

    CHECK(run({"install", "https://example.com/x.zip"}, ctx).code == 2);
    CHECK(run({"install", t.str("user").c_str()}, ctx).code == 1);   // onto itself
}

TEST_CASE("cli: new derives an id, a stem and names from one argument") {
    auto n = new_applet_names("my-viewer");
    CHECK(n.error.empty());
    CHECK(n.id == "dev.example.my-viewer");
    CHECK(n.short_name == "my-viewer");
    CHECK(n.stem == "my_viewer");
    CHECK(n.display == "My Viewer");
    CHECK(n.class_name == "MyViewerApplet");
    auto full = new_applet_names("com.example.scope2");
    CHECK(full.id == "com.example.scope2");
    CHECK(full.stem == "scope2");
    CHECK(full.display == "Scope2");
    for (const char* bad : {"", "My-Viewer", "9lives", "my_viewer", "a..b", "trail-",
                            "com.example.", "../x", "a/b"}) {
        CAPTURE(bad);
        CHECK_FALSE(new_applet_names(bad).error.empty());
    }
}

TEST_CASE("cli: new refuses names whose CMake targets are reserved") {
    // CMake (or one of its generators) reserves these target names; the stem
    // is the applet's target and <stem>_tests its test's.
    for (const char* name : {"all", "clean", "help", "install", "package", "package-source",
                             "preinstall", "rebuild-cache", "edit-cache", "test",
                             "zero-check", "all-build", "run", "depend",
                             "com.example.test", "caliper-sdk-ui-stack"}) {
        CAPTURE(name);
        const auto n = new_applet_names(name);
        CHECK(n.error.find("is reserved; choose another name") != std::string::npos);
    }
    CHECK(new_applet_names("run").error.find("\"run_tests\"") != std::string::npos);
    for (const char* name : {"tests", "testing", "installer", "my-test", "allot"}) {
        CAPTURE(name);
        CHECK(new_applet_names(name).error.empty());
    }
    TempDir t("reserved");
    const CliContext ctx = context(t);
    const Run r = run({"new", "install", t.str("x").c_str()}, ctx);
    CHECK(r.code == 2);
    CHECK(r.err == "caliper: not a usable applet name: install (the CMake target "
                   "\"install\" is reserved; choose another name)\n");
    CHECK_FALSE(fs::exists(t.path / "x"));
}

TEST_CASE("cli: new writes a project from the template and never overwrites") {
    TempDir t("new");
    const CliContext ctx = context(t);
    const std::string dir = t.str("viewer");
    const Run r = run({"new", "com.example.my-viewer", dir.c_str()}, ctx);
    CHECK(r.code == 0);
    CHECK(r.out == "created " + dir + " for applet com.example.my-viewer\n"
                   "  CMakeLists.txt\n  my_viewer.caliper.toml\n  my_viewer.cpp\n"
                   "  tests/test_my_viewer.cpp\n");
    const std::string cmake = slurp(t.path / "viewer" / "CMakeLists.txt");
    CHECK(cmake.find("find_package(caliper-sdk 0.2.0 CONFIG REQUIRED)") != std::string::npos);
    CHECK(cmake.find("caliper_add_applet(my_viewer\n    SOURCES my_viewer.cpp\n"
                     "    MANIFEST my_viewer.caliper.toml)") != std::string::npos);
    const std::string manifest = slurp(t.path / "viewer" / "my_viewer.caliper.toml");
    auto parsed = parse_manifest_text(manifest);
    REQUIRE(parsed.ok);
    CHECK(parsed.manifest.id == "com.example.my-viewer");
    CHECK(parsed.manifest.name == "My Viewer");
    CHECK(parsed.manifest.abi_epoch == 2);
    CHECK(parsed.manifest.min_host == "0.6.0");
    const std::string source = slurp(t.path / "viewer" / "my_viewer.cpp");
    CHECK(source.find("class MyViewerApplet final") != std::string::npos);
    CHECK(source.find(".id       = \"com.example.my-viewer\"") != std::string::npos);
    CHECK(source.find("caliper::ui::dock_hint(host_, \"My Viewer\", CALIPER_DOCK_CENTRAL)") !=
          std::string::npos);
    CHECK(source.find('@') == std::string::npos);   // every placeholder filled
    CHECK(slurp(t.path / "viewer" / "tests" / "test_my_viewer.cpp")
              .find("#include \"../my_viewer.cpp\"") != std::string::npos);

    const Run again = run({"new", "com.example.my-viewer", dir.c_str()}, ctx);
    CHECK(again.code == 1);
    CHECK(again.err == "caliper: " + dir + " already exists; nothing written\n");
    CHECK(run({"new", "Bad Name"}, ctx).code == 2);

    CliContext unknown_sdk = ctx;
    unknown_sdk.sdk_version.clear();
    REQUIRE(run({"new", "other", t.str("other").c_str()}, unknown_sdk).code == 0);
    CHECK(slurp(t.path / "other" / "CMakeLists.txt")
              .find("find_package(caliper-sdk CONFIG REQUIRED)") != std::string::npos);
}

TEST_CASE("cli: check and install take a build folder whose applets are in applets/") {
    TempDir t("build-folder");
    const CliContext ctx = context(t);
    fake_applet(t.path / "build" / "applets", "a", "x.a", "1.0.0");
    const std::string build = t.str("build");
    // Joined part by part: the host prints folders with the platform's
    // separator, '\' on Windows.
    const std::string applets = (t.path / "build" / "applets").string();
    const Run r = run({"check", build.c_str()}, ctx);
    CHECK(r.code == 0);
    CHECK(r.out == "using the applets in " + applets + "\n"
                   "READY  x.a  1.0.0  " + applets + "\n"
                   "1 of 1 applets pass the checks\n");
    // Named directly, the folder is used as given and nothing more is said.
    const Run direct = run({"check", applets.c_str()}, ctx);
    CHECK(direct.out.rfind("READY  x.a", 0) == 0);

    const Run in = run({"install", build.c_str()}, ctx);
    CHECK(in.code == 0);
    CHECK(in.out.rfind("using the applets in " + applets + "\n", 0) == 0);
    CHECK(fs::exists(t.path / "user" / "a.caliper.toml"));

    // A folder with no manifests and none in applets/ names itself.
    fs::create_directories(t.path / "empty" / "applets");
    const Run none = run({"check", t.str("empty").c_str()}, ctx);
    CHECK(none.code == 1);
    CHECK(none.err == "caliper: no applets in " + t.str("empty") + "\n");
}

namespace {
// Names in `dir` that start with ".": the temporary and set-aside copies an
// install may leave behind if it fails to clean up.
std::vector<std::string> hidden_in(const fs::path& dir) {
    std::vector<std::string> out;
    std::error_code ec;
    for (const auto& e : fs::recursive_directory_iterator(dir, ec))
        if (e.path().filename().string().rfind('.', 0) == 0)
            out.push_back(e.path().string());
    return out;
}
void write(const fs::path& p, const std::string& text) {
    fs::create_directories(p.parent_path());
    std::ofstream(p) << text;
}
} // namespace

TEST_CASE("cli: install copies an applet's asset folder and replaces it on update") {
    TempDir t("assets");
    const CliContext ctx = context(t);
    fake_applet(t.path / "build", "a", "x.a", "1.0.0");
    write(t.path / "build" / "assets" / "a" / "old.txt", "v1");
    write(t.path / "build" / "assets" / "other" / "x.txt", "not a's");
    const std::string build = t.str("build");
    REQUIRE(run({"install", build.c_str()}, ctx).code == 0);
    CHECK(slurp(t.path / "user" / "assets" / "a" / "old.txt") == "v1");
    CHECK_FALSE(fs::exists(t.path / "user" / "assets" / "other"));

    fs::remove(t.path / "build" / "assets" / "a" / "old.txt");
    write(t.path / "build" / "assets" / "a" / "new.txt", "v2");
    fake_applet(t.path / "build", "a", "x.a", "1.1.0");
    REQUIRE(run({"install", build.c_str()}, ctx).code == 0);
    CHECK(slurp(t.path / "user" / "assets" / "a" / "new.txt") == "v2");
    CHECK_FALSE(fs::exists(t.path / "user" / "assets" / "a" / "old.txt"));
    CHECK(hidden_in(t.path / "user").empty());
}

TEST_CASE("cli: an install that fails part-way puts everything back") {
    TempDir t("rollback");
    const fs::path src = t.path / "src", dst = t.path / "dst";
    write(src / "liba.dylib", "new library");
    write(src / "assets" / "a" / "new.txt", "new asset");
    write(src / "a.caliper.toml", "new manifest");
    write(src / "libb.dylib", "b library");
    write(src / "b.caliper.toml", "b manifest");
    write(dst / "liba.dylib", "old library");
    write(dst / "assets" / "a" / "old.txt", "old asset");
    write(dst / "a.caliper.toml", "old manifest");
    const std::vector<cli_detail::InstallCopy> copies = {
        {src / "liba.dylib", dst / "liba.dylib"},
        {src / "assets" / "a", dst / "assets" / "a"},
        {src / "a.caliper.toml", dst / "a.caliper.toml"},
        {src / "libb.dylib", dst / "libb.dylib"},
        {src / "b.caliper.toml", dst / "b.caliper.toml"},
    };
    for (int fail_at = 0; fail_at < (int)copies.size(); fail_at++) {
        CAPTURE(fail_at);
        std::string why;
        CHECK_FALSE(cli_detail::install_all(copies, why, fail_at));
        CHECK_FALSE(why.empty());
        CHECK(slurp(dst / "liba.dylib") == "old library");
        CHECK(slurp(dst / "assets" / "a" / "old.txt") == "old asset");
        CHECK_FALSE(fs::exists(dst / "assets" / "a" / "new.txt"));
        CHECK(slurp(dst / "a.caliper.toml") == "old manifest");
        CHECK_FALSE(fs::exists(dst / "libb.dylib"));
        CHECK_FALSE(fs::exists(dst / "b.caliper.toml"));
        CHECK(hidden_in(dst).empty());
    }
    std::string why;
    REQUIRE(cli_detail::install_all(copies, why));
    CHECK(slurp(dst / "liba.dylib") == "new library");
    CHECK(slurp(dst / "assets" / "a" / "new.txt") == "new asset");
    CHECK_FALSE(fs::exists(dst / "assets" / "a" / "old.txt"));
    CHECK(slurp(dst / "b.caliper.toml") == "b manifest");
    CHECK(hidden_in(dst).empty());
}

TEST_CASE("cli: an install that cannot stage a file installs nothing") {
    TempDir t("stage-fail");
    const CliContext ctx = context(t);
    fake_applet(t.path / "build", "a", "x.a", "1.0.0");
    fake_applet(t.path / "build", "b", "x.b", "1.0.0");
    write(t.path / "build" / "assets" / "b" / "data.txt", "b's");
    write(t.path / "user" / "assets", "a file where a folder must go");
    const Run r = run({"install", t.str("build").c_str()}, ctx);
    CHECK(r.code == 1);
    CHECK(r.err.rfind("caliper: nothing installed; ", 0) == 0);
    CHECK_FALSE(fs::exists(t.path / "user" / lib_name("a")));
    CHECK_FALSE(fs::exists(t.path / "user" / "a.caliper.toml"));
    CHECK_FALSE(fs::exists(t.path / "user" / "b.caliper.toml"));
    CHECK(hidden_in(t.path / "user").empty());
}

// ---- caliper serve as a process --------------------------------------------
// Runs the caliper executable named by CALIPER_TEST_CALIPER_EXE (the ctest
// entry caliper_cli_serve sets it) as a user runs it: it must print its link,
// write the hub file, serve a viewer and a producer, stop on SIGTERM with BYE
// to both, exit with 0, and leave neither the hub file nor a listening socket.
// POSIX only (it sends a signal).
#ifndef _WIN32
namespace {

// A client that says HELLO and collects what the hub sends.
struct HubClient {
    caliper_net::Client c;
    std::vector<caliper_net::wire::Message> got;
    bool open = false, closed = false;
    uint16_t code = 0;
    void poll() {
        for (auto& e : c.poll(5)) {
            using T = caliper_net::Event::Type;
            if (e.type == T::Open) open = true;
            if (e.type == T::Closed) {
                closed = true;
                code = e.code;
            }
            if (e.type == T::Message) {
                caliper_net::wire::Message m;
                if (caliper_net::wire::decode(e.data.data(), e.data.size(), &m).ok()) got.push_back(m);
            }
        }
    }
    template <typename T>
    const T* find() const {
        for (const auto& m : got)
            if (auto* p = std::get_if<T>(&m)) return p;
        return nullptr;
    }
    bool until(const std::function<bool()>& f, int ms = 5000) {
        const auto end = std::chrono::steady_clock::now() + std::chrono::milliseconds(ms);
        while (!f() && std::chrono::steady_clock::now() < end) poll();
        return f();
    }
    void send(const caliper_net::wire::Message& m) {
        std::vector<uint8_t> b;
        REQUIRE(caliper_net::wire::encode(m, &b));
        c.send(b.data(), b.size());
    }
};

}  // namespace

TEST_CASE("cli: caliper serve runs as a process, serves, stops on SIGTERM and leaves nothing behind") {
    namespace wire = caliper_net::wire;
    const char* exe = std::getenv("CALIPER_TEST_CALIPER_EXE");
    if (!exe || !*exe) {
        MESSAGE("CALIPER_TEST_CALIPER_EXE is not set: run the ctest entry caliper_cli_serve");
        return;
    }
    TempDir t("serve-process");
    const std::string token = "cli_serve_token-0123456789abcdef";
    std::ofstream(t.str("token")) << token << "\n";
    fs::permissions(t.str("token"), fs::perms::owner_read | fs::perms::owner_write);
    const std::string hub_file = t.str("hub.json");

    Proc p;
    REQUIRE(p.start(exe, {"serve", "--port", "0", "--token-file", t.str("token")}, {"CALIPER_HUB_FILE=" + hub_file}));
    REQUIRE_MESSAGE(p.read_until("stop with Ctrl-C", 10000), p.text);
    // viewers   http://127.0.0.1:<port>/?token=<token>
    const std::string tag = "viewers   http://127.0.0.1:";
    const size_t at = p.text.find(tag);
    REQUIRE(at != std::string::npos);
    const int port = std::atoi(p.text.c_str() + at + tag.size());
    REQUIRE(port > 0);
    CHECK(p.text.find("http://127.0.0.1:" + std::to_string(port) + "/?token=" + token + "\n") != std::string::npos);
    CHECK(p.text.find("producers ws://127.0.0.1:" + std::to_string(port) + "/v3/stream") != std::string::npos);
    // The hub file: owner only, with the address and the token.
    REQUIRE(fs::exists(hub_file));
    CHECK((fs::status(hub_file).permissions() & (fs::perms::group_all | fs::perms::others_all)) == fs::perms::none);
    std::ifstream hf(hub_file);
    const std::string hub_text((std::istreambuf_iterator<char>(hf)), std::istreambuf_iterator<char>());
    CHECK(hub_text.find("\"token\":\"" + token + "\"") != std::string::npos);
    CHECK(hub_text.find("\"port\":" + std::to_string(port)) != std::string::npos);

    const std::string url = "ws://127.0.0.1:" + std::to_string(port) + "/v3/stream?token=" + token;
    std::string err;
    HubClient producer, viewer;
    REQUIRE(producer.c.connect(url, &err));
    REQUIRE(producer.until([&] { return producer.open; }));
    wire::Hello ph;
    ph.role = CALIPER_NET_ROLE_PRODUCER;
    ph.session = wire::Identity{"cli-session", "train.py", 77, 1}.to_json();
    producer.send(ph);
    REQUIRE(producer.until([&] { return producer.find<wire::Welcome>() != nullptr; }));
    CHECK(producer.find<wire::Welcome>()->role == CALIPER_NET_ROLE_HUB);
    REQUIRE(viewer.c.connect(url, &err));
    REQUIRE(viewer.until([&] { return viewer.open; }));
    wire::Hello vh;
    vh.role = CALIPER_NET_ROLE_VIEWER;
    viewer.send(vh);
    REQUIRE(viewer.until([&] {
        auto* s = viewer.find<wire::Sessions>();
        return s && s->list.size() == 1 && viewer.find<wire::Views>();
    }));
    CHECK(viewer.find<wire::Sessions>()->list[0].identity.id == "cli-session");
    REQUIRE(p.read_until("session cli-session started", 5000));

    ::kill(p.pid, SIGTERM);
    REQUIRE(viewer.until([&] { return viewer.closed; }));
    REQUIRE(producer.until([&] { return producer.closed; }));
    REQUIRE(viewer.find<wire::Bye>());
    CHECK(viewer.find<wire::Bye>()->reason == CALIPER_NET_BYE_SHUTDOWN);
    CHECK(viewer.code == CALIPER_NET_CLOSE_GOING_AWAY);
    REQUIRE(producer.find<wire::Bye>());
    CHECK(producer.code == CALIPER_NET_CLOSE_GOING_AWAY);
    CHECK(p.wait(5000) == 0);
    CHECK(p.text.find("hub stopped") != std::string::npos);
    CHECK(p.text.find(token, p.text.find("producers")) == std::string::npos);   // the log never repeats it

    // Nothing left: no hub file, nothing listening on the port.
    CHECK_FALSE(fs::exists(hub_file));
    for (const auto& e : fs::directory_iterator(t.path))
        CHECK_MESSAGE(e.path().filename() == "token", e.path().string());
    const int fd = ::socket(AF_INET, SOCK_STREAM, 0);
    sockaddr_in a{};
    a.sin_family = AF_INET;
    a.sin_port = htons((uint16_t)port);
    inet_pton(AF_INET, "127.0.0.1", &a.sin_addr);
    CHECK(::connect(fd, (sockaddr*)&a, sizeof a) != 0);
    ::close(fd);

    // A token file other users can read is refused, and nothing is written.
    fs::permissions(t.str("token"), fs::perms::owner_read | fs::perms::owner_write | fs::perms::others_read);
    Proc q;
    REQUIRE(q.start(exe, {"serve", "--port", "0", "--token-file", t.str("token")}, {"CALIPER_HUB_FILE=" + hub_file}));
    CHECK(q.wait(5000) == 1);
    CHECK(q.text.find("chmod 600") != std::string::npos);
    CHECK_FALSE(fs::exists(hub_file));
}
#endif

