#include <doctest/doctest.h>
#include "applet_loader.h"
#include "test_temp.h"
#include <caliper/fixture_host.h>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <thread>
#include <vector>
#ifdef _WIN32
// POSIX setenv/unsetenv don't exist on MSVC; _putenv_s with "" removes.
#define setenv(k, v, overwrite) _putenv_s(k, v)
#define unsetenv(k) _putenv_s(k, "")
#endif
namespace fs = std::filesystem;
using namespace caliper_host;

// Platform shared-library naming (matches applet_loader.cpp's constants):
// Windows = "<name>.dll", macOS = "lib<name>.dylib", Linux = "lib<name>.so".
static std::string lib_name(const std::string& stem) {
#ifdef _WIN32
    return stem + ".dll";
#elif defined(__APPLE__)
    return "lib" + stem + ".dylib";
#else
    return "lib" + stem + ".so";
#endif
}

// CALIPER_TEST_APPLETS_DIR + CALIPER_TEST_DATA_ROOT are compile definitions.
static HostCaps caps() {
    return HostCaps{3, "0.6.0", {"caliper.ui.v1", "caliper.log.v1"}};
}
static int find_by_id(AppletLoader& L, const std::string& id) {
    for (int i = 0; i < L.count(); i++)
        if (L.at(i).manifest.id == id) return i;
    return -1;
}
static int count_log(const caliper::testing::FixtureHost& fx, const std::string& s) {
    int n = 0;
    for (auto& l : fx.log_lines()) if (l.find(s) != std::string::npos) n++;
    return n;
}

TEST_CASE("loader: scan finds hello_world via manifest, status Ready") {
    AppletLoader L(caps(), CALIPER_TEST_DATA_ROOT);
    L.scan(CALIPER_TEST_APPLETS_DIR);
    int i = find_by_id(L, "dev.caliper.hello-world");
    REQUIRE(i >= 0);
    CHECK(L.at(i).status == AppletStatus::Ready);
    CHECK_FALSE(L.at(i).dylib_path.empty());
}

TEST_CASE("loader: full lifecycle, hooks called exactly once") {
    caliper::testing::FixtureHost fx;
    AppletLoader L(caps(), CALIPER_TEST_DATA_ROOT);
    L.scan(CALIPER_TEST_APPLETS_DIR);
    int i = find_by_id(L, "dev.caliper.hello-world");
    REQUIRE(i >= 0);
    REQUIRE(L.launch(i, *fx.host()));
    CHECK(L.at(i).status == AppletStatus::Active);
    CHECK(count_log(fx, "hello_world.on_init") == 1);
    L.teardown(i);
    CHECK(L.at(i).status == AppletStatus::Ready);
    CHECK(count_log(fx, "hello_world.on_cleanup") == 1);
    L.close_all();
}

TEST_CASE("loader: relaunch tears down the old instance first") {
    caliper::testing::FixtureHost fx;
    AppletLoader L(caps(), CALIPER_TEST_DATA_ROOT);
    L.scan(CALIPER_TEST_APPLETS_DIR);
    int i = find_by_id(L, "dev.caliper.hello-world");
    REQUIRE(L.launch(i, *fx.host()));
    REQUIRE(L.launch(i, *fx.host()));
    CHECK(count_log(fx, "hello_world.on_init") == 2);
    CHECK(count_log(fx, "hello_world.on_cleanup") == 1);
    L.close_all();
}

TEST_CASE("loader: descriptor/manifest agreement is enforced") {
    // Manifest lies about the version -> launch must fail with a reason.
    caliper::testing::FixtureHost fx;
    fs::path dir = caliper_test::unique_temp_path("caliper-liar");
    fs::create_directories(dir);
    fs::copy_file(fs::path(CALIPER_TEST_APPLETS_DIR) / lib_name("hello_world"),
                  dir / lib_name("hello_world"), fs::copy_options::overwrite_existing);
    std::ofstream(dir / "hello_world.caliper.toml") <<
        "[applet]\nid=\"dev.caliper.hello-world\"\nname=\"Hello World\"\nversion=\"9.9.9\"\n"
        "[compat]\nabi_epoch=3\n[services]\nrequired=[\"caliper.ui.v1\"]\n";
    AppletLoader L(caps(), CALIPER_TEST_DATA_ROOT);
    L.scan(dir.string());
    REQUIRE(L.count() == 1);
    CHECK(L.at(0).status == AppletStatus::Ready);      // pre-dlopen checks pass
    CHECK_FALSE(L.launch(0, *fx.host()));              // descriptor sanity fails
    CHECK(L.at(0).status == AppletStatus::Failed);
    CHECK(L.at(0).status_text.find("descriptor") != std::string::npos);
    // Windows locks loaded DLL files: drop the loader's handle before cleanup,
    // and tolerate a straggling lock rather than failing the case on cleanup.
    L.close_all();
    std::error_code ec;
    fs::remove_all(dir, ec);
}

TEST_CASE("loader: epoch mismatch refused before any dlopen") {
    fs::path dir = caliper_test::unique_temp_path("caliper-epoch99");
    fs::create_directories(dir);
    std::ofstream(dir / "fake.caliper.toml") <<
        "[applet]\nid=\"x.fake\"\nname=\"Fake\"\nversion=\"1.0.0\"\n"
        "[compat]\nabi_epoch=99\n";
    std::ofstream(dir / lib_name("fake")) << "not a real library";  // never opened
    AppletLoader L(caps(), CALIPER_TEST_DATA_ROOT);
    L.scan(dir.string());
    REQUIRE(L.count() == 1);
    CHECK(L.at(0).status == AppletStatus::Refused);
    CHECK(L.at(0).status_text.find("epoch 99") != std::string::npos);
    fs::remove_all(dir);
}

TEST_CASE("loader: missing binary is a Failed card, not a crash") {
    fs::path dir = caliper_test::unique_temp_path("caliper-nobin");
    fs::create_directories(dir);
    std::ofstream(dir / "ghost.caliper.toml") <<
        "[applet]\nid=\"x.ghost\"\nname=\"Ghost\"\nversion=\"1.0.0\"\n"
        "[compat]\nabi_epoch=3\n";
    AppletLoader L(caps(), CALIPER_TEST_DATA_ROOT);
    L.scan(dir.string());
    REQUIRE(L.count() == 1);
    CHECK(L.at(0).status == AppletStatus::Failed);
    CHECK(L.at(0).status_text.find("not found") != std::string::npos);
    fs::remove_all(dir);
}

TEST_CASE("loader: fault in frame() quarantines, host survives") {
    caliper::testing::FixtureHost fx;
    setenv("CALIPER_HELLO_WORLD_CRASH", "1", 1);
    AppletLoader L(caps(), CALIPER_TEST_DATA_ROOT);
    L.scan(CALIPER_TEST_APPLETS_DIR);
    int i = find_by_id(L, "dev.caliper.hello-world");
    REQUIRE(L.launch(i, *fx.host()));
    CaliperFrameInfo fi{}; fi.struct_size = sizeof fi;
    CHECK_FALSE(L.frame(i, fi));                       // fault -> quarantined
    CHECK(L.at(i).status == AppletStatus::Quarantined);
    // POSIX guard reports "SIGSEGV (...)", the Windows guard "SEH exception 0x...".
    CHECK((L.at(i).status_text.find("SIG") != std::string::npos ||
           L.at(i).status_text.find("SEH") != std::string::npos));
    unsetenv("CALIPER_HELLO_WORLD_CRASH");
    // The host (this test process) is alive to assert all of the above.
}

TEST_CASE("loader: a path-escaping manifest id is a Failed card, never a data dir") {
    // The id becomes <data_root>/<id>: "../x" would land the applet's sandbox
    // (and create_directories at launch) outside the data root. Such ids are
    // refused before any path is built; the card still has a title (fname).
    caliper::testing::FixtureHost fx;
    fs::path dir = caliper_test::unique_temp_path("caliper-badid");
    fs::create_directories(dir);
    const char* ids[] = {"../../caliper-escaped", "a..b", "Dev.Caliper.Hello",
                         "dev.under_score", "a/b", ".lead", "trail."};
    int n = 0;
    for (const char* id : ids) {
        std::ofstream(dir / ("bad" + std::to_string(n++) + ".caliper.toml")) <<
            "[applet]\nid=\"" << id << "\"\nname=\"Bad\"\nversion=\"1.0.0\"\n"
            "[compat]\nabi_epoch=3\n";
    }
    AppletLoader L(caps(), CALIPER_TEST_DATA_ROOT);
    L.scan(dir.string());
    REQUIRE(L.count() == n);
    for (int i = 0; i < L.count(); i++) {
        CAPTURE(L.at(i).manifest.name);
        CHECK(L.at(i).status == AppletStatus::Failed);
        CHECK(L.at(i).status_text.find("id") != std::string::npos);
        CHECK(L.at(i).data_dir.empty());
        CHECK_FALSE(L.launch(i, *fx.host()));          // Failed is not Ready
    }
    // Nothing escaped: the sibling the first id pointed at does not exist.
    const fs::path escaped =
        fs::path(CALIPER_TEST_DATA_ROOT).parent_path().parent_path() /
        "caliper-escaped";
    CHECK_FALSE(fs::exists(escaped));
    fs::remove_all(dir);
}

TEST_CASE("loader: descriptor-required service missing refuses with the friendly card") {
    // The manifest only admits to ui.v1, but hello_world's DESCRIPTOR declares
    // {ui.v1, log.v1} — the binary's own list is the one that must hold. A
    // host without log.v1 passes the pre-dlopen manifest check (Ready), then
    // refuses at launch with the same card text negotiate() would have used.
    caliper::testing::FixtureHost fx;
    fs::path dir = caliper_test::unique_temp_path("caliper-underdeclared");
    fs::create_directories(dir);
    fs::copy_file(fs::path(CALIPER_TEST_APPLETS_DIR) / lib_name("hello_world"),
                  dir / lib_name("hello_world"), fs::copy_options::overwrite_existing);
    std::ofstream(dir / "hello_world.caliper.toml") <<
        "[applet]\nid=\"dev.caliper.hello-world\"\nname=\"Hello World\"\nversion=\"0.1.0\"\n"
        "[compat]\nabi_epoch=3\n[services]\nrequired=[\"caliper.ui.v1\"]\n";
    AppletLoader L(HostCaps{3, "0.6.0", {"caliper.ui.v1"}}, CALIPER_TEST_DATA_ROOT);
    L.scan(dir.string());
    REQUIRE(L.count() == 1);
    CHECK(L.at(0).status == AppletStatus::Ready);      // manifest list passes
    CHECK_FALSE(L.launch(0, *fx.host()));              // descriptor list fails
    CHECK(L.at(0).status == AppletStatus::Refused);
    CHECK(L.at(0).status_text.find("caliper.log.v1") != std::string::npos);
    CHECK(L.at(0).status_text.find("Requires a capability") != std::string::npos);
    CHECK(count_log(fx, "hello_world.on_init") == 0);  // never initialized
    L.close_all();
    std::error_code ec;
    fs::remove_all(dir, ec);
}

// ---------------------------------------------------------------------------
// Data-schema stamp (docs/development/compatibility-and-epochs.md, "The version of each
// applet's stored data").
// ---------------------------------------------------------------------------

// A new folder of this run's own under the system temp dir (test_temp.h),
// removed when the case ends.
struct TempDir {
    fs::path path;
    explicit TempDir(const std::string& name)
        : path(caliper_test::unique_temp_path("caliper-test-" + name)) {
        fs::create_directories(path);
    }
    ~TempDir() {
        std::error_code ec;
        fs::remove_all(path, ec);
    }
};

// Copies the hello_world fixture's library into `dir` as <stem> beside a manifest
// for dev.caliper.hello-world 0.1.0 (what its descriptor says) plus `extra` TOML.
static void hello_world_copy(const fs::path& dir, const std::string& stem,
                             const std::string& extra = "") {
    fs::create_directories(dir);
    fs::copy_file(fs::path(CALIPER_TEST_APPLETS_DIR) / lib_name("hello_world"),
                  dir / lib_name(stem), fs::copy_options::overwrite_existing);
    std::ofstream(dir / (stem + ".caliper.toml")) <<
        "[applet]\nid=\"dev.caliper.hello-world\"\nname=\"Hello World\"\nversion=\"0.1.0\"\n"
        "[compat]\nabi_epoch=3\n[services]\nrequired=[\"caliper.ui.v1\"]\n" << extra;
}

TEST_CASE("data stamp: read, raise, never lowered") {
    TempDir t("stamp");
    const std::string d = (t.path / "data").string();
    CHECK(read_data_schema_stamp(d) == 0);
    CHECK(raise_data_schema_stamp(d, 2));
    CHECK(read_data_schema_stamp(d) == 2);
    CHECK(raise_data_schema_stamp(d, 1));      // lower: kept at 2
    CHECK(read_data_schema_stamp(d) == 2);
    CHECK(raise_data_schema_stamp(d, 5));
    CHECK(read_data_schema_stamp(d) == 5);
    CHECK_FALSE(fs::exists(fs::path(d) / (std::string(kDataStampName) + ".tmp")));
    std::ofstream(fs::path(d) / kDataStampName) << "schema = {{{";
    CHECK(read_data_schema_stamp(d) == 0);     // damaged: ignored
}

TEST_CASE("data stamp: concurrent raises keep the highest schema") {
    // Writers race on one folder, as two processes opening the same applet
    // data would; each has its own temporary file, and the stamp ends at the
    // highest schema any of them recorded.
    for (int round = 0; round < 20; round++) {
        TempDir t("stamp-race");
        const std::string d = (t.path / "data").string();
        std::vector<std::thread> writers;
        bool ok[8] = {};
        for (int i = 0; i < 8; i++)
            writers.emplace_back([&, i] { ok[i] = raise_data_schema_stamp(d, (uint32_t)(i + 1)); });
        for (auto& w : writers) w.join();
        for (bool b : ok) CHECK(b);
        CHECK(read_data_schema_stamp(d) == 8);
        int leftovers = 0;   // only the stamp and its lock file remain
        for (const auto& e : fs::directory_iterator(d))
            leftovers += e.path().filename() != kDataStampName &&
                         e.path().filename() != ".caliper-data.lock";
        CHECK(leftovers == 0);
    }
}

TEST_CASE("data stamp: raised before initialize(), even when it fails") {
    caliper::testing::FixtureHost fx;
    TempDir t("stamp-init");
    hello_world_copy(t.path / "applets", "hello_world", "[data]\nschema=2\n");
    const std::string data_root = (t.path / "data").string();
    setenv("CALIPER_HELLO_WORLD_INIT_FAIL", "1", 1);
    AppletLoader L(caps(), data_root);
    L.scan((t.path / "applets").string());
    REQUIRE(L.count() == 1);
    REQUIRE(L.at(0).status == AppletStatus::Ready);
    CHECK_FALSE(L.launch(0, *fx.host()));
    unsetenv("CALIPER_HELLO_WORLD_INIT_FAIL");
    CHECK(L.at(0).status_text == "initialize() returned false");
    CHECK(read_data_schema_stamp(data_root + "/dev.caliper.hello-world") == 2);
    L.close_all();
}

TEST_CASE("data stamp: an applet without a schema writes no stamp") {
    caliper::testing::FixtureHost fx;
    TempDir t("stamp-none");
    hello_world_copy(t.path / "applets", "hello_world");
    const std::string data_root = (t.path / "data").string();
    AppletLoader L(caps(), data_root);
    L.scan((t.path / "applets").string());
    REQUIRE(L.launch(0, *fx.host()));
    CHECK(fs::is_directory(data_root + "/dev.caliper.hello-world"));
    CHECK_FALSE(fs::exists(fs::path(data_root) / "dev.caliper.hello-world" / kDataStampName));
    L.close_all();
}

TEST_CASE("data stamp: a lower schema is refused before dlopen") {
    TempDir t("stamp-refuse");
    fs::create_directories(t.path / "applets");
    std::ofstream(t.path / "applets" / "old.caliper.toml") <<
        "[applet]\nid=\"x.old\"\nname=\"Old\"\nversion=\"1.0.0\"\n"
        "[compat]\nabi_epoch=3\n[data]\nschema=2\n";
    std::ofstream(t.path / "applets" / lib_name("old")) << "never opened";
    const std::string data_root = (t.path / "data").string();
    REQUIRE(raise_data_schema_stamp(data_root + "/x.old", 3));
    AppletLoader L(caps(), data_root);
    L.scan((t.path / "applets").string());
    REQUIRE(L.count() == 1);
    CHECK(L.at(0).status == AppletStatus::Refused);
    CHECK(L.at(0).status_text == "Stored data is schema 3; this applet reads schema 2.");
}

TEST_CASE("data stamp: checked again at launch, before dlopen") {
    caliper::testing::FixtureHost fx;
    TempDir t("stamp-relaunch");
    hello_world_copy(t.path / "applets", "hello_world", "[data]\nschema=2\n");
    const std::string data_root = (t.path / "data").string();
    AppletLoader L(caps(), data_root);
    L.scan((t.path / "applets").string());
    REQUIRE(L.at(0).status == AppletStatus::Ready);
    // Another copy (or process) raised the stamp after the scan.
    REQUIRE(raise_data_schema_stamp(data_root + "/dev.caliper.hello-world", 4));
    CHECK_FALSE(L.launch(0, *fx.host()));
    CHECK(L.at(0).status == AppletStatus::Refused);
    CHECK(L.at(0).status_text == "Stored data is schema 4; this applet reads schema 2.");
    CHECK(L.at(0).handle == nullptr);          // never opened
    CHECK(count_log(fx, "hello_world.on_init") == 0);
}

TEST_CASE("loader: libtorch and UI-stack refusals come before dlopen") {
    TempDir t("pins");
    fs::create_directories(t.path);
    std::ofstream(t.path / "t.caliper.toml") <<
        "[applet]\nid=\"x.torch\"\nname=\"T\"\nversion=\"1.0.0\"\n"
        "[compat]\nabi_epoch=3\ntorch=\"2.5.1\"\n";
    std::ofstream(t.path / "u.caliper.toml") <<
        "[applet]\nid=\"x.ui\"\nname=\"U\"\nversion=\"1.0.0\"\n"
        "[compat]\nabi_epoch=3\n[build]\nui_stack=\"ffff0000ffff0000\"\n";
    std::ofstream(t.path / lib_name("t")) << "never opened";
    std::ofstream(t.path / lib_name("u")) << "never opened";
    HostCaps c = caps();
    c.ui_stack_hash = "0000ffff0000ffff";
    AppletLoader L(c, (t.path / "data").string());
    L.scan(t.path.string());
    REQUIRE(L.count() == 2);
    const int ti = find_by_id(L, "x.torch"), ui = find_by_id(L, "x.ui");
    CHECK(L.at(ti).status == AppletStatus::Refused);
    CHECK(L.at(ti).status_text == "Built against libtorch 2.5.1; this host provides no libtorch.");
    CHECK(L.at(ui).status == AppletStatus::Refused);
    CHECK(L.at(ui).status_text.find("UI stack ffff0000ffff") != std::string::npos);
}
