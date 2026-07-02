#include <doctest/doctest.h>
#include "applet_loader.h"
#include <caliper/fixture_host.h>
#include <cstdlib>
#include <filesystem>
#include <fstream>
namespace fs = std::filesystem;
using namespace caliper_host;

// CALIPER_TEST_APPLETS_DIR + CALIPER_TEST_DATA_ROOT are compile definitions.
static HostCaps caps() {
    return HostCaps{2, "0.6.0", {"caliper.ui.v1", "caliper.log.v1"}};
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

TEST_CASE("loader: scan finds hello via manifest, status Ready") {
    AppletLoader L(caps(), CALIPER_TEST_DATA_ROOT);
    L.scan(CALIPER_TEST_APPLETS_DIR);
    int i = find_by_id(L, "dev.caliper.hello");
    REQUIRE(i >= 0);
    CHECK(L.at(i).status == AppletStatus::Ready);
    CHECK_FALSE(L.at(i).dylib_path.empty());
}

TEST_CASE("loader: full lifecycle, hooks called exactly once") {
    caliper::testing::FixtureHost fx;
    AppletLoader L(caps(), CALIPER_TEST_DATA_ROOT);
    L.scan(CALIPER_TEST_APPLETS_DIR);
    int i = find_by_id(L, "dev.caliper.hello");
    REQUIRE(i >= 0);
    REQUIRE(L.launch(i, *fx.host()));
    CHECK(L.at(i).status == AppletStatus::Active);
    CHECK(count_log(fx, "hello.on_init") == 1);
    L.teardown(i);
    CHECK(L.at(i).status == AppletStatus::Ready);
    CHECK(count_log(fx, "hello.on_cleanup") == 1);
    L.close_all();
}

TEST_CASE("loader: relaunch tears down the old instance first") {
    caliper::testing::FixtureHost fx;
    AppletLoader L(caps(), CALIPER_TEST_DATA_ROOT);
    L.scan(CALIPER_TEST_APPLETS_DIR);
    int i = find_by_id(L, "dev.caliper.hello");
    REQUIRE(L.launch(i, *fx.host()));
    REQUIRE(L.launch(i, *fx.host()));
    CHECK(count_log(fx, "hello.on_init") == 2);
    CHECK(count_log(fx, "hello.on_cleanup") == 1);
    L.close_all();
}

TEST_CASE("loader: descriptor/manifest agreement is enforced") {
    // Manifest lies about the version -> launch must fail with a reason.
    caliper::testing::FixtureHost fx;
    fs::path dir = fs::temp_directory_path() / "caliper-liar";
    fs::create_directories(dir);
    fs::copy_file(fs::path(CALIPER_TEST_APPLETS_DIR) / "libhello.dylib",
                  dir / "libhello.dylib", fs::copy_options::overwrite_existing);
    std::ofstream(dir / "hello.caliper.toml") <<
        "[applet]\nid=\"dev.caliper.hello\"\nname=\"Hello\"\nversion=\"9.9.9\"\n"
        "[compat]\nabi_epoch=2\n[services]\nrequired=[\"caliper.ui.v1\"]\n";
    AppletLoader L(caps(), CALIPER_TEST_DATA_ROOT);
    L.scan(dir.string());
    REQUIRE(L.count() == 1);
    CHECK(L.at(0).status == AppletStatus::Ready);      // pre-dlopen checks pass
    CHECK_FALSE(L.launch(0, *fx.host()));              // descriptor sanity fails
    CHECK(L.at(0).status == AppletStatus::Failed);
    CHECK(L.at(0).status_text.find("descriptor") != std::string::npos);
    fs::remove_all(dir);
}

TEST_CASE("loader: epoch mismatch refused before any dlopen") {
    fs::path dir = fs::temp_directory_path() / "caliper-epoch99";
    fs::create_directories(dir);
    std::ofstream(dir / "fake.caliper.toml") <<
        "[applet]\nid=\"x.fake\"\nname=\"Fake\"\nversion=\"1.0.0\"\n"
        "[compat]\nabi_epoch=99\n";
    std::ofstream(dir / "libfake.dylib") << "not a real dylib";  // never opened
    AppletLoader L(caps(), CALIPER_TEST_DATA_ROOT);
    L.scan(dir.string());
    REQUIRE(L.count() == 1);
    CHECK(L.at(0).status == AppletStatus::Refused);
    CHECK(L.at(0).status_text.find("epoch 99") != std::string::npos);
    fs::remove_all(dir);
}

TEST_CASE("loader: missing binary is a Failed card, not a crash") {
    fs::path dir = fs::temp_directory_path() / "caliper-nobin";
    fs::create_directories(dir);
    std::ofstream(dir / "ghost.caliper.toml") <<
        "[applet]\nid=\"x.ghost\"\nname=\"Ghost\"\nversion=\"1.0.0\"\n"
        "[compat]\nabi_epoch=2\n";
    AppletLoader L(caps(), CALIPER_TEST_DATA_ROOT);
    L.scan(dir.string());
    REQUIRE(L.count() == 1);
    CHECK(L.at(0).status == AppletStatus::Failed);
    CHECK(L.at(0).status_text.find("not found") != std::string::npos);
    fs::remove_all(dir);
}

TEST_CASE("loader: fault in frame() quarantines, host survives") {
    caliper::testing::FixtureHost fx;
    setenv("CALIPER_HELLO_CRASH", "1", 1);
    AppletLoader L(caps(), CALIPER_TEST_DATA_ROOT);
    L.scan(CALIPER_TEST_APPLETS_DIR);
    int i = find_by_id(L, "dev.caliper.hello");
    REQUIRE(L.launch(i, *fx.host()));
    CaliperFrameInfo fi{}; fi.struct_size = sizeof fi;
    CHECK_FALSE(L.frame(i, fi));                       // fault -> quarantined
    CHECK(L.at(i).status == AppletStatus::Quarantined);
    CHECK(L.at(i).status_text.find("SIG") != std::string::npos);
    unsetenv("CALIPER_HELLO_CRASH");
    // The host (this test process) is alive to assert all of the above.
}
