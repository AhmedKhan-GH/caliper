// What applet.save writes is what the loader accepts: an applet is built and
// saved through a builder Session over a temporary workspace
// (design/builder-session.md), and the loader, told the workspace, lists it
// as a Ready spec applet, validates it in full and reads back the same spec
// (src/host/applet_spec.h). Built with CALIPER_WITH_BUILDER only.
#include <doctest/doctest.h>
#include "applet_loader.h"
#include "applet_spec.h"
#include "test_temp.h"
#include <caliper_builder/commands/command.h>
#include <caliper_builder/session/session.h>
#include <nlohmann/json.hpp>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>
namespace fs = std::filesystem;
using nlohmann::json;
using namespace caliper_host;

namespace {

struct TempDir {
    fs::path path;
    explicit TempDir(const std::string& name)
        : path(caliper_test::unique_temp_path("caliper-spec-session-" + name)) {
        fs::create_directories(path);
    }
    ~TempDir() {
        std::error_code ec;
        fs::remove_all(path, ec);
    }
};

json run(caliper_builder::Session& s, const std::string& name, json args = json::object()) {
    caliper_builder::CommandInvocation inv;
    inv.name = name;
    inv.args = std::move(args);
    inv.origin = caliper_builder::CommandOrigin::ui;
    auto r = s.invoke(std::move(inv));
    INFO(name << ": " << r.error.code << " at " << r.error.where << ": " << r.error.message);
    REQUIRE(r.ok);
    return r.result;
}

int entry_of(const AppletLoader& L, const std::string& id) {
    for (int i = 0; i < L.count(); i++)
        if (L.at(i).manifest.id == id) return i;
    return -1;
}

} // namespace

TEST_CASE("applet.save writes what the loader lists, validates and reads back") {
    TempDir t("round-trip");
    const fs::path ws = t.path / "workspace";
    fs::create_directories(ws / "data");
    std::ofstream(ws / "data" / "metrics.csv") << "step,loss\n0,1.0\n1,0.5\n2,0.25\n";

    auto opened = caliper_builder::Session::open(ws.string());
    REQUIRE_MESSAGE(opened.error.ok(), opened.error.code << ": " << opened.error.message);
    caliper_builder::Session& s = *opened.session;
    run(s, "data.import", {{"paths", {"data/metrics.csv"}}});
    run(s, "spec.new", {{"id", "loss_view"}, {"title", "Loss view"}, {"description", "Loss over steps."}});
    run(s, "source.add", {{"id", "m"}, {"catalog", "table:metrics"}});
    run(s, "panel.create", {{"id", "p"}, {"primitive", "series"}, {"bind", {{"x", "m.step"}, {"y", "m.loss"}}}});
    run(s, "applet.save");
    // A copy under a second id, and a compiled applet's files dropped into
    // the workspace's folder beside them, which the loader must refuse.
    run(s, "applet.save", {{"id", "copy"}});
    std::ofstream(ws / "applets" / "intruder.caliper.toml")
        << "[applet]\nid=\"local.intruder\"\nname=\"I\"\nversion=\"0.1.0\"\n[compat]\nabi_epoch=2\n";
    const caliper_builder::spec::Spec saved = s.open_spec()->spec;
    opened.session.reset();

    AppletLoader L(host_caps(), (t.path / "data").string());
    L.set_workspace(ws.string());
    L.scan(applet_search_dirs({}, nullptr, "", "", ws.string()));
    REQUIRE(L.count() == 3);

    // The workspace id loss_view is the manifest id local.loss-view.
    const int i = entry_of(L, "local.loss-view");
    REQUIRE(i >= 0);
    const AppletEntry& a = L.at(i);
    CHECK(a.manifest.kind == AppletKind::Spec);
    CHECK(a.status == AppletStatus::Ready);
    CHECK(a.status_text.empty());
    CHECK(a.manifest.name == "Loss view");
    CHECK(a.manifest.version == "0.1.0");
    CHECK(a.manifest.summary == "Loss over steps.");
    CHECK(fs::path(a.folder) == ws / "applets" / "loss_view");
    CHECK(fs::path(a.spec_path) == ws / "applets" / "loss_view" / "loss_view.spec.toml");

    std::vector<SpecDiagnostic> found;
    CHECK(L.check_spec(i, &found));
    for (const SpecDiagnostic& d : found) CHECK_FALSE(d.error);
    const SpecFile file = read_spec_file(a.spec_path);
    REQUIRE(file.ok);
    CHECK(file.spec == saved);
    CHECK(file.reason.empty());

    const int copy = entry_of(L, "local.copy");
    REQUIRE(copy >= 0);
    CHECK(L.at(copy).status == AppletStatus::Ready);
    CHECK(L.check_spec(copy));
    CHECK(read_spec_file(L.at(copy).spec_path).spec.applet.id == "copy");

    const int intruder = entry_of(L, "local.intruder");
    REQUIRE(intruder >= 0);
    CHECK(L.at(intruder).status == AppletStatus::Refused);
    CHECK(L.at(intruder).status_text == workspace_library_reason());

    // Launching opens nothing: the page belongs to the app.
    CHECK_FALSE(L.launch(i, CaliperHost{}));
    CHECK(L.at(i).status == AppletStatus::Ready);
    CHECK(L.at(i).status_text == kSpecPageUnavailable);
}

TEST_CASE("rescan_workspace lists an applet saved after the scan, at the workspace's place, and drops a deleted one") {
    TempDir t("rescan");
    const fs::path ws = t.path / "workspace";
    const fs::path user = t.path / "user";
    fs::create_directories(user);
    auto opened = caliper_builder::Session::open(ws.string());
    REQUIRE_MESSAGE(opened.error.ok(), opened.error.message);
    caliper_builder::Session& s = *opened.session;
    run(s, "spec.new", {{"id", "first"}, {"title", "First"}});
    run(s, "panel.create", {{"id", "t"}, {"primitive", "text"}, {"options", {{"text", "one"}}}});
    run(s, "applet.save");

    AppletLoader L(host_caps(), (t.path / "data").string());
    // Without a workspace there is nothing to list again.
    CHECK(L.rescan_workspace() == 0);
    L.set_workspace(ws.string());
    L.scan(applet_search_dirs({}, nullptr, user.string(), "", ws.string()));
    REQUIRE(L.count() == 1);
    // A spec applet the app marked Active is listed again as Ready.
    L.at(0).status = AppletStatus::Active;

    // A second applet saved during the session appears without a new scan;
    // the first is not listed twice.
    run(s, "spec.new", {{"id", "second"}, {"title", "Second"}});
    run(s, "panel.create", {{"id", "t"}, {"primitive", "text"}, {"options", {{"text", "two"}}}});
    run(s, "applet.save");
    CHECK(L.rescan_workspace() == 2);
    REQUIRE(L.count() == 2);
    REQUIRE(entry_of(L, "local.first") >= 0);
    REQUIRE(entry_of(L, "local.second") >= 0);
    CHECK(L.at(entry_of(L, "local.first")).status == AppletStatus::Ready);
    CHECK(L.at(entry_of(L, "local.second")).status == AppletStatus::Ready);

    // A copy of the same id in the user folder, which comes after the
    // workspace in the search order, stays shadowed after a rescan.
    fs::create_directories(user / "second");
    fs::copy_file(ws / "applets" / "second" / "second.spec.toml", user / "second" / "second.spec.toml");
    fs::copy_file(ws / "applets" / "second" / "second.caliper.toml", user / "second" / "second.caliper.toml");
    AppletLoader M(host_caps(), (t.path / "data").string());
    M.set_workspace(ws.string());
    M.scan(applet_search_dirs({}, nullptr, user.string(), "", ws.string()));
    REQUIRE(M.count() == 3);
    CHECK(M.rescan_workspace() == 2);
    REQUIRE(M.count() == 3);
    const int chosen = M.find("local.second");
    REQUIRE(chosen >= 0);
    CHECK(fs::path(M.at(chosen).folder).parent_path() == ws / "applets");
    int shadowed = 0;
    for (int i = 0; i < M.count(); i++) shadowed += M.at(i).status == AppletStatus::Shadowed;
    CHECK(shadowed == 1);

    // A deleted applet goes.
    fs::remove_all(ws / "applets" / "first");
    CHECK(L.rescan_workspace() == 1);
    CHECK(L.count() == 1);
    CHECK(entry_of(L, "local.first") < 0);
}
