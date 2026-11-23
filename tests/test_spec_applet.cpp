// Spec applets in the loader (docs/applets/manifest.md, "Spec applets"):
// a folder with a manifest of kind "spec" and a spec, and no library. The
// loader finds them in the search folders, their subfolders and the open
// workspace's applets/ folder, chooses one copy per id by the usual rules,
// checks what is cheap when it lists them, validates the spec in full with
// the spec library at launch and in check_spec(), and never opens or calls
// anything for them. A build without the builder lists them as refused; that
// path runs in caliper_loader_nospec_tests (test_spec_applet_nospec.cpp).
#include <doctest/doctest.h>
#include "applet_loader.h"
#include "spec_applet_fixture.h"
#include "test_temp.h"
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
namespace fs = std::filesystem;
using namespace caliper_host;
using caliper_test::kInvalidSpec;
using caliper_test::kValidSpec;
using caliper_test::kWarningSpec;
using caliper_test::write_spec_applet;

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

HostCaps caps() { return HostCaps{3, "0.6.0", {"caliper.ui.v1", "caliper.log.v1"}}; }

struct TempDir {
    fs::path path;
    explicit TempDir(const std::string& name)
        : path(caliper_test::unique_temp_path("caliper-spec-" + name)) {
        fs::create_directories(path);
    }
    ~TempDir() {
        std::error_code ec;
        fs::remove_all(path, ec);
    }
    std::string str(const std::string& sub) const { return (path / sub).string(); }
};

// A compiled applet's manifest beside a stand-in library that is never opened.
void fake_library(const fs::path& dir, const std::string& stem, const std::string& id,
                  const std::string& version) {
    fs::create_directories(dir);
    std::ofstream(dir / (stem + ".caliper.toml")) <<
        "[applet]\nid=\"" << id << "\"\nname=\"Example\"\nversion=\"" << version <<
        "\"\n[compat]\nabi_epoch=3\n";
    std::ofstream(dir / lib_name(stem)) << "never opened";
}

// The entry whose manifest is `manifest`, or -1.
int entry_at(const AppletLoader& L, const fs::path& manifest) {
    for (int i = 0; i < L.count(); i++)
        if (fs::path(L.at(i).manifest_path) == manifest) return i;
    return -1;
}

// The single entry of a loader that scanned one folder.
const AppletEntry& only(const AppletLoader& L) {
    REQUIRE(L.count() == 1);
    return L.at(0);
}

} // namespace

TEST_CASE("spec applets: listed Ready with their kind, and nothing is opened or called") {
    if (!spec_runtime_available()) return;   // caliper_loader_nospec_tests covers this build
    TempDir t("ready");
    write_spec_applet(t.path / "user" / "demo", "demo", "local.demo", "0.1.0");
    AppletLoader L(caps(), t.str("data"));
    CHECK(L.scan(t.str("user")) == 1);
    const AppletEntry& a = only(L);
    CHECK(a.manifest.kind == AppletKind::Spec);
    CHECK(a.manifest.id == "local.demo");
    CHECK(a.status == AppletStatus::Ready);
    CHECK(a.status_text.empty());
    CHECK(fs::path(a.folder) == t.path / "user" / "demo");
    CHECK(fs::path(a.spec_path) == t.path / "user" / "demo" / "demo.spec.toml");
    CHECK(a.dylib_path.empty());
    CHECK(a.data_dir.empty());
    CHECK(L.find("local.demo") == 0);
    CHECK(spec_format_version() == 1);

    std::vector<SpecDiagnostic> found;
    CHECK(L.check_spec(0, &found));
    CHECK(found.empty());
    CHECK(L.probe(0));
    CHECK(L.at(0).status == AppletStatus::Ready);

    // Launching does not load anything: it stays Ready and says why no page
    // opened, every time, and no data folder appears.
    CHECK_FALSE(L.launch(0, CaliperHost{}));
    CHECK(L.at(0).status == AppletStatus::Ready);
    CHECK(L.at(0).status_text == kSpecPageUnavailable);
    CHECK_FALSE(L.launch(0, CaliperHost{}));
    CHECK(L.at(0).status == AppletStatus::Ready);
    CHECK(L.at(0).handle == nullptr);
    CHECK(L.at(0).instance == nullptr);
    CHECK_FALSE(L.frame(0, CaliperFrameInfo{}));
    L.teardown(0);
    CHECK_FALSE(L.restartable(0));
    CHECK_FALSE(fs::exists(t.path / "data"));
}

TEST_CASE("spec applets: the workspace's folder in the search order, and one copy per id") {
    if (!spec_runtime_available()) return;
    TempDir t("order");
    const fs::path dev = t.path / "dev", ws = t.path / "ws", user = t.path / "user";
    const fs::path ws_applets = ws / "applets";
    // Dev folders first, then the workspace's applets/, then the user and
    // shipped folders; the workspace's folder is not resolved like a
    // development folder.
    CHECK(workspace_applets_dir(ws.string()) == ws_applets.string());
    CHECK(workspace_applets_dir("").empty());
    const auto dirs = applet_search_dirs({dev.string()}, nullptr, user.string(),
                                         t.str("shipped"), ws.string());
    CHECK(dirs == std::vector<std::string>{dev.string(), ws_applets.string(), user.string(),
                                           t.str("shipped")});
    // Without a workspace the list is as before.
    CHECK(applet_search_dirs({dev.string()}, nullptr, user.string(), t.str("shipped")) ==
          std::vector<std::string>{dev.string(), user.string(), t.str("shipped")});

    // One id in the workspace and the user folder at one version: the
    // workspace's copy is found first. A second id is newer in the user
    // folder, so that copy wins whatever the order. A compiled applet and a
    // spec applet share the ids, and the copies are compared alike.
    write_spec_applet(ws_applets / "demo", "demo", "local.demo", "0.1.0");
    write_spec_applet(user / "demo", "demo", "local.demo", "0.1.0");
    write_spec_applet(ws_applets / "chart", "chart", "local.chart", "0.1.0");
    fake_library(user, "chart", "local.chart", "0.2.0");
    // A compiled applet in the workspace is refused: a workspace is data.
    fake_library(ws_applets, "intruder", "local.intruder", "1.0.0");
    fake_library(ws_applets / "nested", "nested", "local.nested", "1.0.0");

    AppletLoader L(caps(), t.str("data"));
    L.set_workspace(ws.string());
    CHECK(L.workspace_dir() == ws.string());
    L.scan(dirs);
    REQUIRE(L.count() == 6);

    const int ws_demo = entry_at(L, ws_applets / "demo" / "demo.caliper.toml");
    const int user_demo = entry_at(L, user / "demo" / "demo.caliper.toml");
    REQUIRE(ws_demo >= 0);
    REQUIRE(user_demo >= 0);
    CHECK(L.at(ws_demo).status == AppletStatus::Ready);
    CHECK(L.at(user_demo).status == AppletStatus::Shadowed);
    CHECK(L.at(user_demo).status_text ==
          shadowed_reason("0.1.0", (ws_applets / "demo").string()));
    CHECK(L.find("local.demo") == ws_demo);

    const int ws_chart = entry_at(L, ws_applets / "chart" / "chart.caliper.toml");
    const int user_chart = entry_at(L, user / "chart.caliper.toml");
    REQUIRE(ws_chart >= 0);
    REQUIRE(user_chart >= 0);
    CHECK(L.at(user_chart).status == AppletStatus::Ready);
    CHECK(L.at(user_chart).manifest.kind == AppletKind::Library);
    CHECK(L.at(ws_chart).status == AppletStatus::Shadowed);

    for (const fs::path& m : {ws_applets / "intruder.caliper.toml",
                              ws_applets / "nested" / "nested.caliper.toml"}) {
        const int i = entry_at(L, m);
        REQUIRE(i >= 0);
        CHECK(L.at(i).status == AppletStatus::Refused);
        CHECK(L.at(i).status_text == workspace_library_reason());
        CHECK(L.at(i).dylib_path.empty());
        CHECK_FALSE(L.launch(i, CaliperHost{}));
        CHECK(L.at(i).handle == nullptr);
    }

    // The same folder scanned without the workspace set is a plain search
    // folder: its compiled applet is listed as usual, and a subfolder holds
    // spec applets only.
    AppletLoader plain(caps(), t.str("data"));
    plain.scan(ws_applets.string());
    const int intruder = entry_at(plain, ws_applets / "intruder.caliper.toml");
    REQUIRE(intruder >= 0);
    CHECK(plain.at(intruder).status == AppletStatus::Ready);
    CHECK(entry_at(plain, ws_applets / "nested" / "nested.caliper.toml") < 0);
}

TEST_CASE("spec applets: what a subfolder adds, and what it does not") {
    if (!spec_runtime_available()) return;
    TempDir t("subfolders");
    const fs::path user = t.path / "user";
    write_spec_applet(user / "good", "good", "local.good", "0.1.0");
    // A compiled applet's manifest in a subfolder is not looked at, as before.
    fake_library(user / "compiled", "compiled", "local.compiled", "0.1.0");
    // A broken TOML file says no kind and is skipped; a broken spec applet
    // that says its kind is listed as failed.
    fs::create_directories(user / "noise");
    std::ofstream(user / "noise" / "noise.caliper.toml") << "not toml {{{";
    fs::create_directories(user / "half");
    std::ofstream(user / "half" / "half.caliper.toml")
        << "[applet]\nid=\"local.half\"\nname=\"Half\"\nversion=\"0.1.0\"\nkind=\"spec\"\n";
    // Hidden folders, such as an install's temporaries, are skipped.
    write_spec_applet(user / ".staging", "hidden", "local.hidden", "0.1.0");
#ifndef _WIN32
    // A folder that is a link is not followed.
    write_spec_applet(t.path / "elsewhere", "linked", "local.linked", "0.1.0");
    fs::create_directory_symlink(t.path / "elsewhere", user / "linked");
#endif
    AppletLoader L(caps(), t.str("data"));
    L.scan(user.string());
    REQUIRE(L.count() == 2);
    const int good = entry_at(L, user / "good" / "good.caliper.toml");
    const int half = entry_at(L, user / "half" / "half.caliper.toml");
    REQUIRE(good >= 0);
    REQUIRE(half >= 0);
    CHECK(L.at(good).status == AppletStatus::Ready);
    CHECK(L.at(half).status == AppletStatus::Failed);
    CHECK(L.at(half).manifest.kind == AppletKind::Spec);
    CHECK(L.at(half).manifest.name == "half.caliper.toml");
    CHECK(L.at(half).status_text.find("manifest missing applet.spec") == 0);

    // In the workspace every manifest is listed, a broken one as failed.
    TempDir w("workspace-noise");
    fs::create_directories(w.path / "applets" / "noise");
    std::ofstream(w.path / "applets" / "noise" / "noise.caliper.toml") << "not toml {{{";
    AppletLoader W(caps(), w.str("data"));
    W.set_workspace(w.path.string());
    W.scan(workspace_applets_dir(w.path.string()));
    const AppletEntry& noise = only(W);
    CHECK(noise.status == AppletStatus::Failed);
    CHECK(noise.status_text.find("manifest parse error") == 0);
}

TEST_CASE("spec applets: the spec never leaves its folder") {
    if (!spec_runtime_available()) return;
    TempDir t("paths");
    const fs::path dir = t.path / "applets";
    std::ofstream(t.path / "outside.spec.toml") << kValidSpec;

    // A name with a folder part is refused by the manifest parser.
    for (const char* name : {"../outside.spec.toml", "sub/x.spec.toml", "/etc/hosts"}) {
        CAPTURE(name);
        TempDir one("name");
        fs::create_directories(one.path / "a");
        std::ofstream(one.path / "a" / "a.caliper.toml")
            << "[applet]\nid=\"local.a\"\nname=\"A\"\nversion=\"0.1.0\"\nkind=\"spec\"\n"
               "spec=\"" << name << "\"\n";
        AppletLoader L(caps(), one.str("data"));
        L.scan(one.path.string());
        const AppletEntry& a = only(L);
        CHECK(a.status == AppletStatus::Failed);
        CHECK(a.status_text.find("applet.spec is not a plain file name") == 0);
        CHECK(a.spec_path.empty());
    }

    // Missing, a folder, or a link wherever it points: failed.
    write_spec_applet(dir / "missing", "missing", "local.missing", "0.1.0", nullptr);
    write_spec_applet(dir / "folder", "folder", "local.folder", "0.1.0", nullptr);
    fs::create_directories(dir / "folder" / "folder.spec.toml");
#ifndef _WIN32
    write_spec_applet(dir / "out", "out", "local.out", "0.1.0", nullptr);
    fs::create_symlink(t.path / "outside.spec.toml", dir / "out" / "out.spec.toml");
    write_spec_applet(dir / "in", "in", "local.in", "0.1.0", nullptr);
    std::ofstream(dir / "in" / "real.toml") << kValidSpec;
    fs::create_symlink("real.toml", dir / "in" / "in.spec.toml");
#endif
    AppletLoader L(caps(), t.str("data"));
    L.scan(dir.string());
    auto text_of = [&](const std::string& stem) {
        const int i = entry_at(L, dir / stem / (stem + ".caliper.toml"));
        REQUIRE(i >= 0);
        CHECK(L.at(i).status == AppletStatus::Failed);
        CHECK(L.at(i).spec_path.empty());
        return L.at(i).status_text;
    };
    CHECK(text_of("missing") == "spec file missing.spec.toml not found next to missing.caliper.toml");
    CHECK(text_of("folder") == "spec file folder.spec.toml is not a regular file");
#ifndef _WIN32
    const std::string link_text =
        " is a symbolic link; a spec applet's spec is a regular file in its folder";
    CHECK(text_of("out") == "spec file out.spec.toml" + link_text);
    CHECK(text_of("in") == "spec file in.spec.toml" + link_text);
#endif
}

TEST_CASE("spec applets: an oversized spec fails when listed, a spec at the limit passes") {
    if (!spec_runtime_available()) return;
    TempDir t("size");
    std::string at_limit = kValidSpec;
    while (at_limit.size() + 80 <= kMaxSpecFileBytes) at_limit += "# " + std::string(77, 'x') + "\n";
    at_limit += "#" + std::string(kMaxSpecFileBytes - at_limit.size() - 2, 'x') + "\n";
    REQUIRE(at_limit.size() == kMaxSpecFileBytes);
    write_spec_applet(t.path / "a" / "limit", "limit", "local.limit", "0.1.0", at_limit.c_str());
    const std::string over = at_limit + "\n";
    write_spec_applet(t.path / "a" / "over", "over", "local.over", "0.1.0", over.c_str());
    AppletLoader L(caps(), t.str("data"));
    L.scan(t.str("a"));
    const int limit = entry_at(L, t.path / "a" / "limit" / "limit.caliper.toml");
    const int big = entry_at(L, t.path / "a" / "over" / "over.caliper.toml");
    REQUIRE(limit >= 0);
    REQUIRE(big >= 0);
    CHECK(L.at(limit).status == AppletStatus::Ready);
    CHECK(L.check_spec(limit));
    CHECK(L.at(big).status == AppletStatus::Failed);
    CHECK(L.at(big).status_text == "spec file over.spec.toml is larger than 512 KiB");
}

TEST_CASE("spec applets: the format version line") {
    if (!spec_runtime_available()) return;
    TempDir t("version");
    const fs::path dir = t.path / "a";
    write_spec_applet(dir / "later", "later", "local.later", "0.1.0",
                      "caliper_spec = 2\n[applet]\nid = \"later\"\n");
    write_spec_applet(dir / "older", "older", "local.older", "0.1.0",
                      "caliper_spec = -1\n[applet]\nid = \"older\"\n");
    write_spec_applet(dir / "text", "text", "local.text", "0.1.0",
                      "caliper_spec = \"0\"\n[applet]\nid = \"text\"\n");
    write_spec_applet(dir / "none", "none", "local.none", "0.1.0", "[applet]\nid = \"none\"\n");
    write_spec_applet(dir / "late", "late", "local.late", "0.1.0",
                      "[applet]\nid = \"late\"\ncaliper_spec = 0\n");
    AppletLoader L(caps(), t.str("data"));
    L.scan(dir.string());
    auto entry = [&](const std::string& stem) -> const AppletEntry& {
        const int i = entry_at(L, dir / stem / (stem + ".caliper.toml"));
        REQUIRE(i >= 0);
        return L.at(i);
    };
    CHECK(entry("later").status == AppletStatus::Refused);
    CHECK(entry("later").status_text ==
          "Written in spec format 2; this host reads spec format 1 — update Caliper to open it.");
    CHECK(entry("later").status_text == spec_version_reason(2, 1));
    CHECK(entry("older").status == AppletStatus::Refused);
    CHECK(entry("older").status_text == "Written in spec format -1; this host reads spec format 1.");
    CHECK(entry("text").status == AppletStatus::Failed);
    CHECK(entry("text").status_text ==
          "spec file text.spec.toml has a caliper_spec that is not a decimal integer");
    for (const char* stem : {"none", "late"}) {
        CAPTURE(stem);
        CHECK(entry(stem).status == AppletStatus::Failed);
        CHECK(entry(stem).status_text == std::string("spec file ") + stem +
                                             ".spec.toml has no line caliper_spec = <version> "
                                             "before its first table");
    }

    // The forms of the line that TOML allows.
    for (const char* text : {"caliper_spec = 0\n", "\xEF\xBB\xBF# c\n\ncaliper_spec=0 # v0\n",
                             "\"caliper_spec\" = +0\r\n", "'caliper_spec'\t=\t0_0\n",
                             "layout = \"p\"\ncaliper_spec = 0\n[applet]\n"}) {
        CAPTURE(text);
        std::istringstream in(text);
        int64_t v = 7;
        std::string why;
        CHECK(read_spec_version_line(in, v, why));
        CHECK(v == 0);
    }
    for (const char* text : {"caliper_spec = 0x0\n", "caliper_spec = 1.0\n", "caliper_spec = 0 x\n",
                             "caliper_spec = 1__0\n", "caliper_spec = 9999999999999999999\n"}) {
        CAPTURE(text);
        std::istringstream in(text);
        int64_t v = 0;
        std::string why;
        CHECK_FALSE(read_spec_version_line(in, v, why));
        CHECK(why == "has a caliper_spec that is not a decimal integer");
    }
}

TEST_CASE("spec applets: a spec that parses but fails validation, with the first error as reason") {
    if (!spec_runtime_available()) return;
    TempDir t("invalid");
    write_spec_applet(t.path / "a" / "broken", "broken", "local.broken", "0.1.0", kInvalidSpec);
    AppletLoader L(caps(), t.str("data"));
    L.scan(t.str("a"));
    // Listing is cheap: the spec is not parsed, so it is Ready.
    CHECK(only(L).status == AppletStatus::Ready);

    std::vector<SpecDiagnostic> found;
    CHECK_FALSE(L.check_spec(0, &found));
    REQUIRE_FALSE(found.empty());
    CHECK(found[0].error);
    CHECK(found[0].code == "spec.unknown_source");
    CHECK(found[0].line == 9);
    CHECK(found[0].column > 0);
    const std::string reason = format_spec_diagnostic("broken.spec.toml", found[0], false);
    CHECK(reason.rfind("broken.spec.toml:9:", 0) == 0);
    CHECK(reason.find(": spec.unknown_source: ") != std::string::npos);
    CHECK(L.at(0).status == AppletStatus::Failed);
    CHECK(L.at(0).status_text == reason);
    // Once failed, neither check_spec nor launch tries again until a rescan.
    CHECK_FALSE(L.check_spec(0, &found));
    CHECK(found.empty());

    // launch() validates too, and fails with the same reason.
    AppletLoader again(caps(), t.str("data"));
    again.scan(t.str("a"));
    CHECK_FALSE(again.launch(0, CaliperHost{}));
    CHECK(again.at(0).status == AppletStatus::Failed);
    CHECK(again.at(0).status_text == reason);

    // A spec that fails to parse fails with the parser's first error.
    write_spec_applet(t.path / "b" / "garbled", "garbled", "local.garbled", "0.1.0",
                      "caliper_spec = 0\n[applet]\nid = \"garbled\"\nwhat = 1\n");
    AppletLoader parse(caps(), t.str("data"));
    parse.scan(t.str("b"));
    CHECK_FALSE(parse.check_spec(0, &found));
    CHECK(parse.at(0).status_text.rfind("garbled.spec.toml:4:", 0) == 0);
    CHECK(parse.at(0).status_text.find("spec.unknown_key") != std::string::npos);

    // Warnings do not fail a spec.
    write_spec_applet(t.path / "c" / "warned", "warned", "local.warned", "0.1.0", kWarningSpec);
    AppletLoader warned(caps(), t.str("data"));
    warned.scan(t.str("c"));
    CHECK(warned.check_spec(0, &found));
    REQUIRE(found.size() == 1);
    CHECK_FALSE(found[0].error);
    CHECK(found[0].code == "spec.unused_source");
    CHECK(warned.at(0).status == AppletStatus::Ready);
}

TEST_CASE("spec applets: min_host is checked, the keys of a library are not") {
    if (!spec_runtime_available()) return;
    TempDir t("min-host");
    write_spec_applet(t.path / "a" / "future", "future", "local.future", "0.1.0", kValidSpec,
                      "[compat]\nmin_host = \"9.0.0\"\n");
    write_spec_applet(t.path / "a" / "epoch", "epoch", "local.epoch", "0.1.0", kValidSpec,
                      "[compat]\nabi_epoch = 3\n");
    AppletLoader L(caps(), t.str("data"));
    L.scan(t.str("a"));
    const int future = entry_at(L, t.path / "a" / "future" / "future.caliper.toml");
    const int epoch = entry_at(L, t.path / "a" / "epoch" / "epoch.caliper.toml");
    REQUIRE(future >= 0);
    REQUIRE(epoch >= 0);
    CHECK(L.at(future).status == AppletStatus::Refused);
    CHECK(L.at(future).status_text == "Requires host 9.0.0 or newer; this host is 0.6.0.");
    CHECK(L.at(epoch).status == AppletStatus::Failed);
    CHECK(L.at(epoch).manifest.kind == AppletKind::Spec);
    CHECK(L.at(epoch).status_text ==
          "compat.abi_epoch does not apply to a spec applet, which has no library");
}

TEST_CASE("spec applets: a manifest that may be anyone's is read with bounds") {
    if (!spec_runtime_available()) return;
    TempDir t("bounds");
    const fs::path applets = t.path / "applets";
    fs::create_directories(applets / "big");
    {
        std::ofstream big(applets / "big" / "big.caliper.toml");
        big << "[applet]\nid=\"local.big\"\nname=\"Big\"\nversion=\"0.1.0\"\nkind=\"spec\"\n"
               "spec=\"big.spec.toml\"\n";
        big << "# " << std::string(kMaxSpecManifestBytes, 'x') << "\n";
    }
    fs::create_directories(applets / "deep");
    {
        std::string key = "a";
        for (int i = 0; i < 40; i++) key += ".a";
        std::ofstream(applets / "deep" / "deep.caliper.toml") << key << " = 1\n";
    }
    AppletLoader L(caps(), t.str("data"));
    L.set_workspace(t.path.string());
    L.scan(applets.string());
    const int big = entry_at(L, applets / "big" / "big.caliper.toml");
    const int deep = entry_at(L, applets / "deep" / "deep.caliper.toml");
    REQUIRE(big >= 0);
    REQUIRE(deep >= 0);
    CHECK(L.at(big).status_text == "manifest is larger than 64 KiB");
    CHECK(L.at(deep).status_text == "manifest has a dotted key of more than 32 parts");

    // Thirty-two parts, and a float, are fine.
    std::string key = "x";
    for (int i = 0; i < 31; i++) key += ".x";
    TempDir ok("bounds-ok");
    write_spec_applet(ok.path / "applets" / "fine", "fine", "local.fine", "0.1.0", kValidSpec,
                      key + " = 1.5\n");
    AppletLoader F(caps(), ok.str("data"));
    F.set_workspace(ok.path.string());
    F.scan(workspace_applets_dir(ok.path.string()));
    CHECK(only(F).status == AppletStatus::Ready);
}

TEST_CASE("spec applets: diagnostics are written with their place") {
    SpecDiagnostic d;
    d.code = "spec.unknown_source";
    d.message = "A binding names a source the spec does not have.";
    d.path = "panel[0].bind.y";
    d.line = 9;
    d.column = 10;
    CHECK(format_spec_diagnostic("x.spec.toml", d, false) ==
          "x.spec.toml:9:10: spec.unknown_source: A binding names a source the spec does not have.");
    CHECK(format_spec_diagnostic("x.spec.toml", d, true) ==
          "x.spec.toml:9:10: error spec.unknown_source: A binding names a source the spec does not have.");
    d.error = false;
    d.line = 0;
    d.column = 0;
    CHECK(format_spec_diagnostic("x.spec.toml", d, true) ==
          "x.spec.toml: warning spec.unknown_source: A binding names a source the spec does not "
          "have. (at panel[0].bind.y)");
}
