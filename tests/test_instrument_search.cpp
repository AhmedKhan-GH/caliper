// Loading from outside the core (docs/development/compatibility-and-epochs.md, "Several
// copies of one instrument"): the search order, CALIPER_INSTRUMENT_PATH, the
// --instruments command line, and the choice between copies of one id.
#include <doctest/doctest.h>
#include "instrument_loader.h"
#include "host_cli.h"
#include "test_temp.h"
#include <caliper/fixture_host.h>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>
namespace fs = std::filesystem;
using namespace caliper_host;

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

HostCaps caps() { return HostCaps{4, "0.6.0", {"caliper.ui.v1", "caliper.log.v1"}}; }

struct TempDir {
    fs::path path;
    explicit TempDir(const std::string& name)
        : path(caliper_test::unique_temp_path("caliper-search-" + name)) {
        fs::create_directories(path);
    }
    ~TempDir() {
        std::error_code ec;
        fs::remove_all(path, ec);
    }
    std::string str(const std::string& sub) const { return (path / sub).string(); }
};

// A manifest for `id` at `version` beside a stand-in library that is never
// opened: enough for every check made before dlopen.
void fake_instrument(const fs::path& dir, const std::string& stem, const std::string& id,
                     const std::string& version, const std::string& extra = "") {
    fs::create_directories(dir);
    std::ofstream(dir / (stem + ".caliper.toml")) <<
        "[instrument]\nid=\"" << id << "\"\nname=\"Example\"\nversion=\"" << version <<
        "\"\n[compat]\nabi_epoch=4\n" << extra;
    std::ofstream(dir / lib_name(stem)) << "never opened";
}

// The hello_world fixture's library under `stem`, with its own manifest.
void hello_world_instrument(const fs::path& dir, const std::string& stem) {
    fs::create_directories(dir);
    fs::copy_file(fs::path(CALIPER_TEST_INSTRUMENTS_DIR) / lib_name("hello_world"),
                  dir / lib_name(stem), fs::copy_options::overwrite_existing);
    std::ofstream(dir / (stem + ".caliper.toml")) <<
        "[instrument]\nid=\"dev.caliper.hello-world\"\nname=\"Hello World\"\nversion=\"0.1.0\"\n"
        "[compat]\nabi_epoch=4\n[services]\nrequired=[\"caliper.ui.v1\"]\n";
}

// The entry of `id` found in `folder`, or -1.
int entry_in(const InstrumentLoader& L, const std::string& id, const fs::path& folder) {
    for (int i = 0; i < L.count(); i++)
        if (L.at(i).manifest.id == id && fs::path(L.at(i).folder) == folder) return i;
    return -1;
}

std::string join(const std::vector<std::string>& parts) {
    std::string out;
    for (size_t i = 0; i < parts.size(); i++) {
        if (i) out += kInstrumentPathSeparator;
        out += parts[i];
    }
    return out;
}

} // namespace

TEST_CASE("search: CALIPER_INSTRUMENT_PATH splits on the platform separator") {
    CHECK(split_instrument_path("").empty());
    CHECK(split_instrument_path(join({"a"})) == std::vector<std::string>{"a"});
    CHECK(split_instrument_path(join({"a", "b", "", "c"})) ==
          std::vector<std::string>{"a", "b", "c"});
    const std::string sep(1, kInstrumentPathSeparator);
    CHECK(split_instrument_path(sep + "a" + sep) == std::vector<std::string>{"a"});
#ifdef _WIN32
    CHECK(kInstrumentPathSeparator == ';');
    // A drive letter's colon is part of the folder on Windows.
    CHECK(split_instrument_path("C:\\a;D:\\b") == std::vector<std::string>{"C:\\a", "D:\\b"});
#else
    CHECK(kInstrumentPathSeparator == ':');
#endif
}

TEST_CASE("search: development paths, then the user folder, then the shipped one") {
    TempDir t("order");
    const std::string env = join({t.str("env1"), t.str("env2")});
    const auto dirs = instrument_search_dirs({t.str("cli1"), t.str("cli2")}, env.c_str(),
                                             t.str("user"), t.str("shipped"));
    CHECK(dirs == std::vector<std::string>{t.str("cli1"), t.str("cli2"), t.str("env1"),
                                           t.str("env2"), t.str("user"),
                                           t.str("shipped")});
}

TEST_CASE("search: folders are absolute, listed once, and never empty") {
    TempDir t("dedupe");
    const std::string user = t.str("user");
    const std::string env = join({user + "/", t.str("dev"), t.str("dev")});
    const auto dirs = instrument_search_dirs({t.str("dev")}, env.c_str(), user, "");
    CHECK(dirs == std::vector<std::string>{t.str("dev"), user});

    const auto rel = instrument_search_dirs({"relative-folder"}, nullptr, "", "");
    REQUIRE(rel.size() == 1);
    CHECK(fs::path(rel[0]).is_absolute());
    CHECK(fs::path(rel[0]).filename() == "relative-folder");
    CHECK(instrument_search_dirs({}, nullptr, "", "").empty());
    CHECK(instrument_search_dirs({}, "", user, "") == std::vector<std::string>{user});
}

TEST_CASE("command line: --instruments in both spellings, --help, and errors") {
    {
        const char* argv[] = {"caliper", "--instruments", "/a", "--instruments=/b", "--instruments", "c"};
        auto cl = parse_command_line(6, argv);
        CHECK(cl.error.empty());
        CHECK(cl.command == CommandLine::Command::Run);
        CHECK(cl.instrument_dirs == std::vector<std::string>{"/a", "/b", "c"});
    }
    {
        const char* argv[] = {"caliper"};
        auto cl = parse_command_line(1, argv);
        CHECK(cl.error.empty());
        CHECK(cl.command == CommandLine::Command::Run);
        CHECK(cl.instrument_dirs.empty());
    }
    {
        const char* argv[] = {"caliper", "--help"};
        CHECK(parse_command_line(2, argv).command == CommandLine::Command::Help);
    }
    {
        const char* argv[] = {"caliper", "--instruments"};
        CHECK(parse_command_line(2, argv).error == "--instruments needs a folder");
    }
    {
        const char* argv[] = {"caliper", "--instruments="};
        CHECK(parse_command_line(2, argv).error == "--instruments needs a folder");
    }
    {
        const char* argv[] = {"caliper", "--frobnicate"};
        CHECK(parse_command_line(2, argv).error == "unknown option: --frobnicate");
    }
    CHECK(std::string(usage_text()).rfind("usage: caliper", 0) == 0);
}

TEST_CASE("copies: the highest version is chosen, the others are Shadowed") {
    TempDir t("highest");
    fake_instrument(t.path / "dev", "x", "x.app", "1.0.0");
    fake_instrument(t.path / "user", "x", "x.app", "1.10.0");   // numeric, not lexical
    fake_instrument(t.path / "shipped", "x", "x.app", "1.9.0");
    InstrumentLoader L(caps(), t.str("data"));
    L.scan(std::vector<std::string>{t.str("dev"), t.str("user"), t.str("shipped")});
    REQUIRE(L.count() == 3);
    const int dev = entry_in(L, "x.app", t.path / "dev");
    const int user = entry_in(L, "x.app", t.path / "user");
    const int shipped = entry_in(L, "x.app", t.path / "shipped");
    CHECK(L.at(user).status == InstrumentStatus::Ready);
    CHECK(L.at(dev).status == InstrumentStatus::Shadowed);
    CHECK(L.at(shipped).status == InstrumentStatus::Shadowed);
    CHECK(L.at(dev).status_text ==
          "Version 1.10.0 in " + t.str("user") + " is used instead.");
    CHECK(L.find("x.app") == user);
    CHECK(L.at(dev).search_rank < L.at(user).search_rank);
}

TEST_CASE("copies: a refused newest copy stays visible and the next one is chosen") {
    TempDir t("fallback");
    fake_instrument(t.path / "dev", "x", "x.app", "3.0.0", "torch=\"9.9.9\"\n");  // refused
    fake_instrument(t.path / "user", "x", "x.app", "2.0.0");
    std::ofstream(t.path / "user" / "y.caliper.toml") <<                       // failed
        "[instrument]\nid=\"x.app\"\nname=\"Example\"\nversion=\"2.5.0\"\n[compat]\nabi_epoch=4\n";
    fake_instrument(t.path / "shipped", "x", "x.app", "1.0.0");
    InstrumentLoader L(caps(), t.str("data"));
    L.scan(instrument_search_dirs({t.str("dev")}, nullptr, t.str("user"), t.str("shipped")));
    REQUIRE(L.count() == 4);
    int refused = -1, failed = -1, ready = -1, shadowed = -1;
    for (int i = 0; i < L.count(); i++) {
        switch (L.at(i).status) {
            case InstrumentStatus::Refused: refused = i; break;
            case InstrumentStatus::Failed: failed = i; break;
            case InstrumentStatus::Ready: ready = i; break;
            case InstrumentStatus::Shadowed: shadowed = i; break;
            default: break;
        }
    }
    REQUIRE(refused >= 0);
    REQUIRE(failed >= 0);
    REQUIRE(ready >= 0);
    REQUIRE(shadowed >= 0);
    CHECK(L.at(refused).manifest.version == "3.0.0");
    CHECK(L.at(refused).status_text.find("libtorch 9.9.9") != std::string::npos);
    CHECK(L.at(failed).manifest.version == "2.5.0");
    CHECK(L.at(failed).status_text.find("binary not found") != std::string::npos);
    CHECK(L.at(ready).manifest.version == "2.0.0");
    CHECK(L.at(shadowed).manifest.version == "1.0.0");
    CHECK(L.find("x.app") == ready);
}

TEST_CASE("copies: equal versions go to the copy found first") {
    TempDir t("tie");
    fake_instrument(t.path / "a", "x", "x.app", "1.0.0");
    fake_instrument(t.path / "b", "x", "x.app", "1.0.0");
    InstrumentLoader L(caps(), t.str("data"));
    L.scan(std::vector<std::string>{t.str("b"), t.str("a")});   // b is searched first
    const int a = entry_in(L, "x.app", t.path / "a");
    const int b = entry_in(L, "x.app", t.path / "b");
    CHECK(L.at(b).status == InstrumentStatus::Ready);
    CHECK(L.at(a).status == InstrumentStatus::Shadowed);
    CHECK(L.at(a).status_text == "Version 1.0.0 in " + t.str("b") + " is used instead.");
}

TEST_CASE("copies: the data schema decides with the other checks") {
    TempDir t("schema");
    const std::string data = t.str("data");
    REQUIRE(raise_data_schema_stamp(data + "/x.app", 3));

    SUBCASE("the newest reads the stored data: the older copy is refused, not shadowed") {
        fake_instrument(t.path / "dev", "x", "x.app", "2.0.0", "[data]\nschema=3\n");
        fake_instrument(t.path / "user", "x", "x.app", "1.0.0", "[data]\nschema=2\n");
        InstrumentLoader L(caps(), data);
        L.scan(std::vector<std::string>{t.str("dev"), t.str("user")});
        const int dev = entry_in(L, "x.app", t.path / "dev");
        const int user = entry_in(L, "x.app", t.path / "user");
        CHECK(L.at(dev).status == InstrumentStatus::Ready);
        CHECK(L.at(user).status == InstrumentStatus::Refused);
        CHECK(L.at(user).status_text ==
              "Stored data is schema 3; this instrument reads schema 2.");
    }
    SUBCASE("the newest is refused: the fallback cannot read the stored data either") {
        fake_instrument(t.path / "dev", "x", "x.app", "2.0.0",
                        "torch=\"9.9.9\"\n[data]\nschema=3\n");
        fake_instrument(t.path / "user", "x", "x.app", "1.0.0", "[data]\nschema=2\n");
        fake_instrument(t.path / "shipped", "x", "x.app", "0.9.0");   // no schema: no check
        InstrumentLoader L(caps(), data);
        L.scan(std::vector<std::string>{t.str("dev"), t.str("user"), t.str("shipped")});
        const int dev = entry_in(L, "x.app", t.path / "dev");
        const int user = entry_in(L, "x.app", t.path / "user");
        const int shipped = entry_in(L, "x.app", t.path / "shipped");
        CHECK(L.at(dev).status == InstrumentStatus::Refused);
        CHECK(L.at(user).status == InstrumentStatus::Refused);
        CHECK(L.at(user).status_text.find("Stored data is schema 3") != std::string::npos);
        CHECK(L.at(shipped).status == InstrumentStatus::Ready);
        CHECK(L.find("x.app") == shipped);
    }
}

TEST_CASE("copies: a later scan can take over, and only the chosen copy launches") {
    caliper::testing::FixtureHost fx;
    TempDir t("launch");
    hello_world_instrument(t.path / "first", "hello_world");
    hello_world_instrument(t.path / "second", "hello_world_copy");
    InstrumentLoader L(caps(), t.str("data"));
    L.scan(t.str("first"));
    CHECK(L.at(L.find("dev.caliper.hello-world")).status == InstrumentStatus::Ready);
    L.scan(t.str("second"));   // equal version, searched later: shadowed
    REQUIRE(L.count() == 2);
    const int first = entry_in(L, "dev.caliper.hello-world", t.path / "first");
    const int second = entry_in(L, "dev.caliper.hello-world", t.path / "second");
    CHECK(L.find("dev.caliper.hello-world") == first);
    CHECK(L.at(second).status == InstrumentStatus::Shadowed);
    CHECK_FALSE(L.launch(second, *fx.host()));      // a shadowed copy never runs
    CHECK(L.at(second).handle == nullptr);
    REQUIRE(L.launch(first, *fx.host()));
    CHECK(L.at(first).status == InstrumentStatus::Active);
    CHECK(L.find("dev.caliper.hello-world") == first);    // a running copy is the one
    L.close_all();
}

TEST_CASE("copies: probe opens, checks and creates without initializing") {
    caliper::testing::FixtureHost fx;
    TempDir t("probe");
    hello_world_instrument(t.path / "a", "hello_world");
    std::ofstream(t.path / "a" / "liar.caliper.toml") <<   // descriptor disagrees
        "[instrument]\nid=\"dev.caliper.liar\"\nname=\"Liar\"\nversion=\"0.1.0\"\n"
        "[compat]\nabi_epoch=4\n";
    fs::copy_file(t.path / "a" / lib_name("hello_world"), t.path / "a" / lib_name("liar"));
    InstrumentLoader L(caps(), t.str("data"));
    L.scan(t.str("a"));
    const int hello = L.find("dev.caliper.hello-world");
    const int liar = L.find("dev.caliper.liar");
    CHECK(L.probe(hello));
    CHECK(L.at(hello).status == InstrumentStatus::Ready);
    CHECK_FALSE(fx.log_contains("hello_world.on_init"));
    CHECK_FALSE(fs::exists(t.path / "data" / "dev.caliper.hello-world"));   // no data folder
    CHECK_FALSE(L.probe(liar));
    CHECK(L.at(liar).status == InstrumentStatus::Failed);
    CHECK(L.at(liar).status_text == "descriptor id disagrees with manifest");
    // A probed instrument still launches.
    REQUIRE(L.launch(hello, *fx.host()));
    L.close_all();
}

TEST_CASE("search: a build folder stands for its instruments/ subfolder") {
    TempDir t("resolve");
    // Manifests only in build/instruments, where caliper_add_instrument() puts them.
    fake_instrument(t.path / "build" / "instruments", "x", "x.app", "1.0.0");
    CHECK(resolve_instrument_folder(t.str("build")) == (t.path / "build" / "instruments").string());
    // A folder with manifests of its own is used as given, instruments/ or not.
    fake_instrument(t.path / "both", "y", "y.app", "1.0.0");
    fake_instrument(t.path / "both" / "instruments", "z", "z.app", "1.0.0");
    CHECK(resolve_instrument_folder(t.str("both")) == t.str("both"));
    CHECK(resolve_instrument_folder(t.str("build/instruments")) == t.str("build/instruments"));
    // Nothing to resolve to: the folder as given.
    fs::create_directories(t.path / "bare" / "instruments");
    CHECK(resolve_instrument_folder(t.str("bare")) == t.str("bare"));
    CHECK(resolve_instrument_folder(t.str("nowhere")) == t.str("nowhere"));
    CHECK(resolve_instrument_folder("").empty());
}

TEST_CASE("search: development folders resolve, installed folders do not") {
    TempDir t("resolve-order");
    fake_instrument(t.path / "cli" / "instruments", "a", "a.app", "1.0.0");
    fake_instrument(t.path / "env" / "instruments", "b", "b.app", "1.0.0");
    fake_instrument(t.path / "user" / "instruments", "c", "c.app", "1.0.0");
    const std::string env = t.str("env");
    const auto dirs = instrument_search_dirs({t.str("cli"), t.str("cli/instruments")}, env.c_str(),
                                             t.str("user"), "");
    // "cli" and "cli/instruments" name one folder, which is searched once. The
    // folders come back with the platform's separator, '\' on Windows.
    CHECK(dirs == std::vector<std::string>{(t.path / "cli" / "instruments").string(),
                                           (t.path / "env" / "instruments").string(),
                                           t.str("user")});
    InstrumentLoader L(caps(), t.str("data"));
    L.scan(dirs);
    CHECK(L.find("a.app") >= 0);
    CHECK(L.find("b.app") >= 0);
    CHECK(L.find("c.app") < 0);   // the user folder is never resolved
}
