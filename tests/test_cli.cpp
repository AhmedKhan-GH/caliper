// The caliper command line's subcommands (src/host/host_cli.h), called as
// main() calls them but with a CliContext of temporary folders, so nothing
// touches the user's applets and no window ever opens. `dev` starts the app,
// so only its parsing is tested.
#include <doctest/doctest.h>
#include "host_cli.h"
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
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

struct TempDir {
    fs::path path;
    explicit TempDir(const std::string& name)
        : path(fs::temp_directory_path() / ("caliper-cli-" + name)) {
        std::error_code ec;
        fs::remove_all(path, ec);
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
          "--applets applies only to starting the app, dev and list");
    CHECK(parse({"--applets"}).error == "--applets needs a folder");
    CHECK(parse({"--applets", "--help"}).error == "--applets needs a folder");
    CHECK(parse({"list", "--applets=--x"}).error == "--applets needs a folder");
}

TEST_CASE("cli: the -psn_ argument Finder passes is ignored") {
    auto cl = parse({"-psn_0_12345"});
    CHECK(cl.error.empty());
    CHECK(cl.command == Command::Run);
    auto dev = parse({"dev", "-psn_0_1", "/a"});
    CHECK(dev.error.empty());
    CHECK(dev.applet_dirs == std::vector<std::string>{"/a"});
}

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
    const std::string applets = t.str("build/applets");
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
