// Tests of Session's handlers through the command registry: the catalog and
// data commands, the loader commands with their undo and the path rules for
// each origin, the open spec with dry runs, diffs, undo and merging, the
// applet files and the approval policy, the SessionHost binding, and the
// failure paths of each.
#include <caliper_builder/session/session.h>

#include <caliper_builder/commands/command_table.h>
#include <caliper_builder/session/spec_edit.h>
#include <caliper_builder/spec/format.h>

#include <doctest/doctest.h>

#include "helpers.h"

using namespace caliper_builder;
using namespace session_test;
using nlohmann::json;

namespace {

const CommandOrigin kUi = CommandOrigin::ui;
const CommandOrigin kMcp = CommandOrigin::mcp;
const CommandOrigin kAgent = CommandOrigin::agent;
CommandOrigin kApi() { return CommandOrigin::api; }

// A workspace with a CSV file inside it and one outside it.
struct Fixture {
    std::string root;
    std::string ws;
    std::string outside;
    std::unique_ptr<Session> s;

    explicit Fixture(const std::string& name) {
        root = fresh_dir(name);
        ws = root + "/workspace";
        outside = root + "/outside";
        write_text(ws + "/data/metrics.csv", "step,loss\n0,1.0\n1,0.5\n2,0.25\n3,nan\n");
        write_text(outside + "/other.csv", "a,b\n1,2\n3,4\n");
        write_safetensors_f32(ws + "/data/heat.safetensors", "w", {2, 3}, {1, 2, 3, 4, 5, 6});
        s = open_session(ws);
    }

    void reopen() {
        s.reset();
        s = open_session(ws);
    }
};

class FakeHost : public SessionHost {
public:
    int renders = 0;
    PanelData last;
    CommandOutput app_info() override {
        return CommandOutput::ok({{"abi_epoch", 2},
                                  {"host", "0.0.0-test"},
                                  {"libtorch", nullptr},
                                  {"sdk", nullptr},
                                  {"services", json::array()},
                                  {"ui_stack", nullptr}});
    }
    CommandOutput app_applets() override { return CommandOutput::ok({{"applets", json::array()}}); }
    CommandOutput app_launch(const std::string& id) override {
        if (id != "com.example.viewer") return CommandOutput::fail(Error("app.applet_not_found", "No such applet."));
        return CommandOutput::ok({{"already_open", false}, {"id", id}, {"status", "ACTIVE"}, {"version", "0.1.0"}});
    }
    CommandOutput render_panel(const spec::Spec&, const PanelData& data, int width, int height) override {
        ++renders;
        last = data;
        return CommandOutput::ok({{"png_base64", "iVBORw0KGgo="}, {"width", width}, {"height", height}});
    }
    CommandOutput capture_frame() override {
        return CommandOutput::fail(Error("capture.unavailable", "The fake host draws no frame."));
    }
};

}  // namespace

TEST_CASE("opening a workspace makes its folders, binds every windowless command and leaves the app's unbound") {
    Fixture f("open");
    for (const char* dir : {"loaders", "applets", "journal", "tensors"}) CHECK(fs::is_directory(f.ws + "/" + dir));
    CHECK(fs::exists(f.ws + "/workspace.duckdb"));
    CHECK(f.s->registry().descriptors().size() == command_table().size());
    for (const auto& d : command_table()) {
        CAPTURE(d.name);
        const bool app = std::find(session_host_commands().begin(), session_host_commands().end(), d.name) !=
                         session_host_commands().end();
        CHECK(f.s->registry().is_bound(d.name) == !app);
    }
    CHECK(err(*f.s, "app.info") == "command.not_bound");
    CHECK(err(*f.s, "spec.get") == "spec.no_open_spec");
    CHECK(err(*f.s, "source.add", {{"id", "a"}, {"sql", "SELECT 1"}}) == "spec.no_open_spec");
    CHECK(err(*f.s, "applet.save") == "spec.no_open_spec");
    CHECK(!f.s->open_spec());
}

TEST_CASE("data.import refuses an outside path from an agent and grants it from the person") {
    Fixture f("import");
    const json from_agent = ok(*f.s, "data.import", {{"paths", {f.outside + "/other.csv", "data/metrics.csv"}}}, kMcp);
    REQUIRE(from_agent["refused"].size() == 1);
    CHECK(from_agent["refused"][0]["code"] == "loader.path_refused");
    REQUIRE(from_agent["loaders"].size() == 1);
    CHECK(from_agent["loaders"][0]["id"] == "metrics");
    CHECK(from_agent["loaders"][0]["reader"] == "csv");
    CHECK(from_agent["loaders"][0]["ran"] == true);
    CHECK(from_agent["loaders"][0]["outputs"] == json{"table:metrics"});
    CHECK(f.s->store().granted_files().empty());

    const json from_ui = ok(*f.s, "data.import", {{"paths", {f.outside + "/other.csv", "data/missing.csv"}}, {"run", false}}, kUi);
    REQUIRE(from_ui["loaders"].size() == 1);
    CHECK(from_ui["loaders"][0]["ran"] == false);
    CHECK(from_ui["refused"][0]["code"] == "loader.io");
    CHECK(f.s->store().granted_files().size() == 1);

    // Pickles are refused one by one, and the rest goes on.
    write_text(f.ws + "/data/model.pkl", "not really a pickle");
    const json mixed = ok(*f.s, "data.import", {{"paths", {"data/model.pkl", "data/heat.safetensors"}}}, kAgent);
    CHECK(mixed["refused"][0]["code"] == "loader.pickle_refused");
    CHECK(mixed["loaders"][0]["outputs"] == json{"tensor:heat_w"});

    // Undo deletes the loaders of the last import; redo makes them again.
    ok(*f.s, "history.undo");
    CHECK(err(*f.s, "loader.describe", {{"id", "heat"}}) == "loader.not_found");
    CHECK(err(*f.s, "catalog.describe", {{"id", "tensor:heat_w"}}) == "catalog.not_found");
    ok(*f.s, "history.redo");
    CHECK(ok(*f.s, "loader.describe", {{"id", "heat"}})["status"] == "ok");
}

TEST_CASE("the catalog and data commands read the loader outputs") {
    Fixture f("catalog");
    ok(*f.s, "data.import", {{"paths", {"data/metrics.csv", "data/heat.safetensors"}}});
    const json all = ok(*f.s, "catalog.list");
    REQUIRE(all["entries"].size() == 2);
    CHECK(all["entries"][0]["id"] == "table:metrics");
    CHECK(all["entries"][1]["id"] == "tensor:heat_w");
    CHECK(ok(*f.s, "catalog.list", {{"kind", "tensor"}})["entries"].size() == 1);
    CHECK(ok(*f.s, "catalog.list", {{"kind", "live"}})["entries"].empty());
    const json limited = ok(*f.s, "catalog.list", {{"limit", 1}});
    CHECK(limited["entries"].size() == 1);
    CHECK(limited["truncated"] == true);
    CHECK(ok(*f.s, "catalog.list", {{"prefix", "heat"}})["entries"].size() == 1);

    const json table = ok(*f.s, "catalog.describe", {{"id", "table:metrics"}})["entry"];
    CHECK(table["rows"] == 4);
    CHECK(table["origin"]["loader"] == "metrics");
    CHECK(table["columns"][1]["name"] == "loss");
    CHECK(err(*f.s, "catalog.describe", {{"id", "table:none"}}) == "catalog.not_found");
    CHECK(err(*f.s, "catalog.describe", {{"id", "nokind:x"}}) == "command.invalid_args");

    const json rows = ok(*f.s, "catalog.sample", {{"id", "table:metrics"}, {"limit", 2}});
    CHECK(rows["rows"].size() == 2);
    CHECK(rows["truncated"] == true);
    CHECK(rows["total_rows"] == 4);
    const json tensor = ok(*f.s, "catalog.sample", {{"id", "tensor:heat_w"}});
    CHECK(tensor["shape"] == json{2, 3});
    CHECK(tensor["stats"]["max"] == 6.0);
    CHECK(tensor["slice"]["values"] == json{1.0, 2.0, 3.0, 4.0, 5.0, 6.0});
    CHECK(ok(*f.s, "catalog.sample", {{"id", "tensor:heat_w"}, {"start", {1, 1}}, {"count", {1, 2}}})["slice"]["values"] ==
          json{5.0, 6.0});
    CHECK(err(*f.s, "catalog.sample", {{"id", "tensor:heat_w"}, {"start", {2, 0}}, {"count", {1, 1}}}) ==
          "catalog.bad_request");

    const json q = ok(*f.s, "data.query", {{"sql", "SELECT step, loss FROM metrics ORDER BY step"}, {"limit", 3}});
    CHECK(q["rows"].size() == 3);
    CHECK(q["truncated"] == true);
    CHECK(q["columns"][0]["name"] == "step");
    const json nan = ok(*f.s, "data.query", {{"sql", "SELECT loss FROM metrics WHERE step = 3"}});
    CHECK(nan["rows"][0][0] == "nan");
    CHECK(err(*f.s, "data.query", {{"sql", "DELETE FROM metrics"}}) == "data.not_select");
    CHECK(err(*f.s, "data.query", {{"sql", "SELECT * FROM nowhere"}}) == "data.sql_error");
    CHECK(err(*f.s, "data.query", {{"sql", "SELECT 1; SELECT 2"}}) == "data.not_select");
    CHECK(err(*f.s, "data.query", {{"sql", "SELECT * FROM read_csv('" + f.outside + "/other.csv')"}}) == "data.sql_error");
    CHECK(err(*f.s, "data.query", {{"sql", "SELECT 1"}, {"limit", 0}}) == "command.invalid_args");
}

TEST_CASE("the loader commands save, run, update and delete loaders with exact undo") {
    Fixture f("loaders");
    const std::string recipe =
        "caliper_loader = 0\n[loader]\nid = \"m\"\n[input]\npath = \"data/metrics.csv\"\n[[output]]\nname = \"m\"\n";
    CHECK(err(*f.s, "loader.create", {{"text", "caliper_loader = 0\n[loader]\nid = \"Bad\"\n"}}) == "loader.invalid");
    const json created = ok(*f.s, "loader.create", {{"text", recipe}, {"run", true}});
    CHECK(created == json{{"id", "m"}, {"outputs", {"table:m"}}});
    CHECK(err(*f.s, "loader.create", {{"text", recipe}}) == "loader.exists");

    const json list = ok(*f.s, "loader.list");
    REQUIRE(list["loaders"].size() == 1);
    CHECK(list["loaders"][0]["status"] == "ok");
    CHECK(list["loaders"][0]["path"] == "data/metrics.csv");
    CHECK(list["loaders"][0]["outputs"] == json{"table:m"});
    const json described = ok(*f.s, "loader.describe", {{"id", "m"}});
    CHECK(described["last_run"]["ok"] == true);
    CHECK(described["last_run"]["elapsed_ms"].is_number());
    CHECK(described["text"].get<std::string>().find("data/metrics.csv") != std::string::npos);

    const std::string updated =
        "caliper_loader = 0\n[loader]\nid = \"m\"\n[input]\npath = \"data/metrics.csv\"\n[[output]]\nname = \"m\"\n"
        "sql = \"SELECT step FROM input\"\n";
    CHECK(err(*f.s, "loader.update", {{"id", "zz"}, {"text", updated}}) == "loader.not_found");
    CHECK(err(*f.s, "loader.update", {{"id", "m"}, {"text", "caliper_loader = 0\n[loader]\nid = \"n\"\n[input]\npath = "
                                                            "\"data/metrics.csv\"\n[[output]]\nname = \"m\"\n"}}) ==
          "loader.id_mismatch");
    ok(*f.s, "loader.update", {{"id", "m"}, {"text", updated}, {"run", true}});
    CHECK(ok(*f.s, "catalog.describe", {{"id", "table:m"}})["entry"]["columns"].size() == 1);
    ok(*f.s, "history.undo");
    CHECK(ok(*f.s, "loader.describe", {{"id", "m"}})["text"].get<std::string>().find("sql") == std::string::npos);
    // The tables stay as the last run made them until the loader runs again.
    CHECK(ok(*f.s, "catalog.describe", {{"id", "table:m"}})["entry"]["columns"].size() == 1);
    const json ran = ok(*f.s, "loader.run", {{"id", "m"}});
    CHECK(ran["outputs"][0] == json{{"id", "table:m"}, {"rows", 4}, {"shape", nullptr}});
    CHECK(ok(*f.s, "catalog.describe", {{"id", "table:m"}})["entry"]["columns"].size() == 2);
    CHECK(err(*f.s, "loader.run", {{"id", "zz"}}) == "loader.not_found");

    CHECK(err(*f.s, "loader.delete", {{"id", "m"}, {"keep_outputs", true}}) == "loader.unsupported");
    CHECK(ok(*f.s, "loader.delete", {{"id", "m"}})["removed"] == json{"table:m"});
    CHECK(ok(*f.s, "loader.list")["loaders"].empty());
    ok(*f.s, "history.undo");
    CHECK(ok(*f.s, "loader.describe", {{"id", "m"}})["status"] == "never_run");
    CHECK(err(*f.s, "loader.delete", {{"id", "zz"}}) == "loader.not_found");

    // A loader whose first run fails is not kept.
    const std::string broken =
        "caliper_loader = 0\n[loader]\nid = \"b\"\n[input]\npath = \"data/none.csv\"\n[[output]]\nname = \"b\"\n";
    CHECK(err(*f.s, "loader.create", {{"text", broken}, {"run", true}}) == "loader.no_match");
    CHECK(err(*f.s, "loader.describe", {{"id", "b"}}) == "loader.not_found");
}

TEST_CASE("a loader's input outside the workspace is the person's to grant") {
    Fixture f("loader-paths");
    const std::string outside_recipe = "caliper_loader = 0\n[loader]\nid = \"o\"\n[input]\npath = \"" + f.outside +
                                       "/other.csv\"\n[[output]]\nname = \"o\"\n";
    CHECK(err(*f.s, "loader.create", {{"text", outside_recipe}}, kMcp) == "loader.path_refused");
    CHECK(err(*f.s, "loader.create", {{"text", outside_recipe}}, kAgent) == "loader.path_refused");
    const std::string glob = "caliper_loader = 0\n[loader]\nid = \"g\"\n[input]\npath = \"" + f.outside +
                             "/*.csv\"\n[[output]]\nname = \"g\"\n";
    CHECK(err(*f.s, "loader.create", {{"text", glob}}, CommandOrigin::api) == "loader.path_refused");
    const std::string escape = "caliper_loader = 0\n[loader]\nid = \"e\"\n[input]\npath = \"../outside/other.csv\"\n"
                               "[[output]]\nname = \"e\"\n";
    CHECK(err(*f.s, "loader.create", {{"text", escape}}, kUi) == "loader.path_refused");
    CHECK(f.s->store().granted_files().empty());

    ok(*f.s, "loader.create", {{"text", outside_recipe}, {"run", true}}, kUi);
    CHECK(f.s->store().granted_files().size() == 1);
    // Once granted, an agent may make another loader over the same file.
    const std::string again = "caliper_loader = 0\n[loader]\nid = \"o2\"\n[input]\npath = \"" + f.outside +
                              "/other.csv\"\n[[output]]\nname = \"o2\"\n";
    ok(*f.s, "loader.create", {{"text", again}}, kMcp);
    // A glob from the person grants its folder.
    ok(*f.s, "loader.create", {{"text", glob}}, kUi);
    CHECK(f.s->store().granted_folders().size() == 1);
}

TEST_CASE("the destructive loader commands ask an agent for approval") {
    Fixture f("loader-approval");
    ok(*f.s, "data.import", {{"paths", {"data/metrics.csv"}}, {"run", false}}, kMcp);
    const auto asked = invoke(*f.s, "loader.run", {{"id", "metrics"}}, kMcp);
    CHECK(asked.error.code == "command.needs_approval");
    REQUIRE(!asked.approval.empty());
    const auto approved = f.s->registry().approve(asked.approval);
    CHECK(approved.ok);
    const auto denied = invoke(*f.s, "loader.delete", {{"id", "metrics"}}, kAgent);
    CHECK(denied.error.code == "command.needs_approval");
    CHECK(f.s->registry().deny(denied.approval).error.code == "command.denied");
    CHECK(ok(*f.s, "loader.list")["loaders"].size() == 1);
}

TEST_CASE("the open spec is edited with dry runs, diffs, warnings and exact undo") {
    Fixture f("spec");
    ok(*f.s, "data.import", {{"paths", {"data/metrics.csv", "data/heat.safetensors"}}});
    CHECK(err(*f.s, "spec.new", {{"id", "Bad id"}}) == "command.invalid_args");
    const json made = ok(*f.s, "spec.new", {{"id", "s1"}, {"title", "One"}});
    CHECK(made["text"] == "caliper_spec = 0\n\n[applet]\nid = \"s1\"\ntitle = \"One\"\n");
    json got = ok(*f.s, "spec.get");
    CHECK(got["path"].is_null());
    CHECK(got["dirty"] == true);
    CHECK(got["json"]["applet"]["id"] == "s1");

    // A dry run says what would change and changes nothing.
    const json dry = ok(*f.s, "source.add", {{"id", "m"}, {"catalog", "table:metrics"}, {"dry_run", true}});
    CHECK(dry["applied"] == false);
    CHECK(dry["diff"].get<std::string>().find("+catalog = \"table:metrics\"") != std::string::npos);
    CHECK(ok(*f.s, "spec.get")["json"].count("source") == 0);

    const json added = ok(*f.s, "source.add", {{"id", "m"}, {"catalog", "table:metrics"}});
    CHECK(added["applied"] == true);
    CHECK(added["diff"] == dry["diff"]);
    CHECK(added["warnings"][0]["code"] == "spec.unused_source");
    // A catalog id that nothing knows is a warning, so a loader can run later.
    const json unknown = ok(*f.s, "source.add", {{"id", "later"}, {"catalog", "tensor:not_yet"}});
    bool unresolved = false;
    for (const auto& w : unknown["warnings"]) unresolved = unresolved || w["code"] == "spec.unresolved_source";
    CHECK(unresolved);
    ok(*f.s, "source.remove", {{"id", "later"}});

    // The resolver knows the columns, so a wrong one is refused.
    CHECK(err(*f.s, "panel.create", {{"id", "p"}, {"primitive", "series"}, {"bind", {{"y", "m.nope"}}}}) ==
          "spec.unknown_column");
    ok(*f.s, "panel.create", {{"id", "p"}, {"primitive", "series"}, {"bind", {{"x", "m.step"}, {"y", "m.loss"}}}});
    ok(*f.s, "source.add", {{"id", "h"}, {"catalog", "tensor:heat_w"}});
    CHECK(err(*f.s, "panel.create", {{"id", "q"}, {"primitive", "heatmap"}, {"bind", {{"values", "m"}}}}) ==
          "spec.slot_mismatch");
    ok(*f.s, "panel.create",
       {{"id", "q"}, {"primitive", "heatmap"}, {"bind", {{"values", "h"}}}, {"place", {{"target", "p"}, {"where", "below"}}}});
    const json placed = ok(*f.s, "layout.place", {{"panel", "q"}, {"target", "p"}, {"where", "beside"}, {"ratio", 0.25}});
    CHECK(placed["layout"] == json{{"root", "n1"},
                                   {"nodes", {{{"id", "n1"}, {"split", "horizontal"}, {"ratio", 0.75}, {"children", {"p", "q"}}}}}});
    CHECK(err(*f.s, "source.remove", {{"id", "m"}}) == "source.in_use");

    // Undo walks back one step at a time; redo comes forward again.
    const std::string before_undo = f.s->open_spec()->text;
    ok(*f.s, "history.undo");
    CHECK(f.s->open_spec()->text != before_undo);
    ok(*f.s, "history.redo");
    CHECK(f.s->open_spec()->text == before_undo);

    // A run of edits with one merge key undoes as one step.
    for (const char* t : {"A", "AB", "ABC"}) {
        const auto r = invoke(*f.s, "panel.patch", {{"id", "p"}, {"patch", {{"title", t}}}}, kUi, {}, "title-drag");
        REQUIRE(r.ok);
    }
    CHECK(f.s->open_spec()->spec.find_panel("p")->title == std::optional<std::string>("ABC"));
    ok(*f.s, "history.undo");
    CHECK(!f.s->open_spec()->spec.find_panel("p")->title);
    ok(*f.s, "history.redo");
    CHECK(f.s->open_spec()->spec.find_panel("p")->title == std::optional<std::string>("ABC"));

    // A group of an agent's edits, each reviewed and approved in turn, undoes
    // as one step.
    for (const json& args : {json{{"id", "p"}, {"patch", {{"options", {{"x_label", "Step"}}}}}},
                             json{{"id", "q"}, {"patch", {{"options", {{"colormap", "magma"}}}}}}}) {
        const auto parked = invoke(*f.s, "panel.patch", args, kAgent, "turn-1");
        REQUIRE(parked.error.code == "command.needs_approval");
        REQUIRE(f.s->registry().approve(parked.approval).ok);
    }
    ok(*f.s, "history.undo", json::object(), kAgent);
    CHECK(f.s->open_spec()->spec.find_panel("p")->options.x_label == std::nullopt);
    CHECK(f.s->open_spec()->spec.find_panel("q")->options.colormap == std::nullopt);

    // spec.set_text replaces the whole spec and refuses an invalid one.
    CHECK(err(*f.s, "spec.set_text", {{"text", "caliper_spec = 1\n[applet]\nid = \"x\"\n"}}) == "spec.unsupported_version");
    const json replaced = ok(*f.s, "spec.set_text", {{"text", read_text(fixture_path("minimal.spec.toml"))}});
    CHECK(replaced["id"] == "empty");
    ok(*f.s, "history.undo");
    CHECK(f.s->open_spec()->spec.applet.id == "s1");

    const json v = ok(*f.s, "spec.validate", {{"text", "caliper_spec = 0\n[applet]\nid = \"x\"\n[[panel]]\nid = \"p\"\nprimitive = \"series\"\n"}});
    CHECK(v["valid"] == false);
    CHECK(v["errors"][0]["code"] == "spec.missing_slot");
    CHECK(ok(*f.s, "spec.validate")["valid"] == true);

    // Opening another spec, then undoing, returns to this one with its changes.
    const std::string mine = f.s->open_spec()->text;
    ok(*f.s, "spec.new", {{"id", "s2"}});
    CHECK(f.s->open_spec()->spec.applet.id == "s2");
    ok(*f.s, "history.undo");
    CHECK(f.s->open_spec()->text == mine);
    ok(*f.s, "history.redo");
    CHECK(f.s->open_spec()->spec.applet.id == "s2");
    ok(*f.s, "history.undo");
}

TEST_CASE("applets are saved atomically, listed, opened, and overwritten only with approval for an agent") {
    Fixture f("applets");
    ok(*f.s, "data.import", {{"paths", {"data/metrics.csv"}}});
    ok(*f.s, "spec.new", {{"id", "my_view"}, {"title", "My view"}, {"description", "Loss over steps."}});
    ok(*f.s, "source.add", {{"id", "m"}, {"catalog", "table:metrics"}});
    ok(*f.s, "panel.create", {{"id", "p"}, {"primitive", "series"}, {"bind", {{"y", "m.loss"}}}});

    // A new applet from an agent needs no approval: nothing is overwritten.
    const json saved = ok(*f.s, "applet.save", json::object(), kMcp);
    CHECK(saved == json{{"id", "my_view"},
                        {"manifest", "applets/my_view/my_view.caliper.toml"},
                        {"spec", "applets/my_view/my_view.spec.toml"}});
    CHECK(read_text(f.ws + "/applets/my_view/my_view.caliper.toml") ==
          "[applet]\nid = \"local.my-view\"\nname = \"My view\"\nversion = \"0.1.0\"\nsummary = \"Loss over "
          "steps.\"\nkind = \"spec\"\nspec = \"my_view.spec.toml\"\n");
    CHECK(read_text(f.ws + "/applets/my_view/my_view.spec.toml") == f.s->open_spec()->text);
    auto state = f.s->open_spec();
    CHECK(state->dirty == false);
    CHECK(state->path == std::optional<std::string>("applets/my_view/my_view.spec.toml"));
    for (const auto& entry : fs::directory_iterator(f.ws + "/applets/my_view"))
        CHECK(entry.path().filename().string().find(".tmp") == std::string::npos);

    CHECK(err(*f.s, "applet.save", json::object(), kMcp) == "applet.exists");
    ok(*f.s, "panel.patch", {{"id", "p"}, {"patch", {{"title", "Loss"}}}});
    CHECK(f.s->open_spec()->dirty == true);
    const auto asked = invoke(*f.s, "applet.save", {{"overwrite", true}}, kMcp);
    CHECK(asked.error.code == "command.needs_approval");
    CHECK(f.s->registry().approve(asked.approval).ok);
    CHECK(f.s->open_spec()->dirty == false);
    ok(*f.s, "applet.save", {{"overwrite", true}}, kUi);

    // Saving under another id writes a copy and leaves the open spec's home.
    ok(*f.s, "applet.save", {{"id", "copy"}}, kUi);
    CHECK(f.s->open_spec()->path == std::optional<std::string>("applets/my_view/my_view.spec.toml"));
    CHECK(read_text(f.ws + "/applets/copy/copy.spec.toml").find("id = \"copy\"") != std::string::npos);
    // An id that maps to a taken manifest id is refused.
    ok(*f.s, "spec.new", {{"id", "my-view"}});
    CHECK(err(*f.s, "applet.save") == "applet.exists");

    const json listed = ok(*f.s, "applet.list");
    REQUIRE(listed["applets"].size() == 2);
    CHECK(listed["applets"][0]["id"] == "copy");
    CHECK(listed["applets"][1]["title"] == "My view");
    CHECK(listed["applets"][1]["updated_at"].is_string());

    const json opened = ok(*f.s, "applet.open", {{"id", "my_view"}});
    CHECK(opened["title"] == "My view");
    CHECK(f.s->open_spec()->spec.find_panel("p")->title == std::optional<std::string>("Loss"));
    CHECK(err(*f.s, "applet.open", {{"id", "none"}}) == "applet.not_found");
    write_text(f.ws + "/applets/lib/lib.caliper.toml", "[applet]\nid = \"local.lib\"\nname = \"Lib\"\nversion = \"1.0.0\"\n");
    CHECK(err(*f.s, "applet.open", {{"id", "lib"}}) == "applet.not_spec");
    write_text(f.ws + "/applets/bad/bad.caliper.toml", "[applet]\nkind = \"spec\"\nspec = \"../../x.spec.toml\"\n");
    CHECK(err(*f.s, "applet.open", {{"id", "bad"}}) == "applet.not_spec");
    write_text(f.ws + "/applets/broken/broken.caliper.toml", "[applet]\nkind = \"spec\"\nspec = \"broken.spec.toml\"\n");
    write_text(f.ws + "/applets/broken/broken.spec.toml", "caliper_spec = 0\n[applet]\nid = \"broken\"\nnope = 1\n");
    CHECK(err(*f.s, "applet.open", {{"id", "broken"}}) == "spec.unknown_key");
    CHECK(ok(*f.s, "applet.list")["applets"].size() == 3);  // the broken one still lists; lib is not a spec

    CHECK(ok(*f.s, "spec.open", {{"path", "applets/copy/copy.spec.toml"}})["id"] == "copy");
    CHECK(err(*f.s, "spec.open", {{"path", "../outside/x.spec.toml"}}) == "spec.path_refused");
    CHECK(err(*f.s, "spec.open", {{"path", f.ws + "/applets/copy/copy.spec.toml"}}) == "spec.path_refused");
    CHECK(err(*f.s, "spec.open", {{"path", "applets/none.spec.toml"}}) == "spec.not_found");
    CHECK(err(*f.s, "spec.open", {{"path", "applets/copy/copy.caliper.toml"}}) == "command.invalid_args");
#ifndef _WIN32
    fs::create_symlink(f.outside, f.ws + "/link");
    write_text(f.outside + "/evil.spec.toml", "caliper_spec = 0\n[applet]\nid = \"evil\"\n");
    CHECK(err(*f.s, "spec.open", {{"path", "link/evil.spec.toml"}}) == "spec.path_refused");
#endif
    write_text(f.ws + "/big.spec.toml", std::string(600 * 1024, '#'));
    CHECK(err(*f.s, "spec.open", {{"path", "big.spec.toml"}}) == "spec.too_large");
}

TEST_CASE("a SessionHost serves the app's commands, and panel.render draws resolved data") {
    Fixture f("host");
    auto host = std::make_shared<FakeHost>();
    f.s->set_host(host);
    for (const auto& name : session_host_commands()) CHECK(f.s->registry().is_bound(name));
    CHECK(ok(*f.s, "app.info")["host"] == "0.0.0-test");
    CHECK(ok(*f.s, "app.applets")["applets"].empty());
    CHECK(ok(*f.s, "app.launch", {{"id", "com.example.viewer"}})["status"] == "ACTIVE");
    CHECK(err(*f.s, "app.launch", {{"id", "com.example.other"}}) == "app.applet_not_found");
    CHECK(err(*f.s, "capture.frame") == "capture.unavailable");
    CHECK(err(*f.s, "panel.render", {{"id", "p"}}) == "spec.no_open_spec");

    ok(*f.s, "data.import", {{"paths", {"data/heat.safetensors"}}});
    ok(*f.s, "spec.new", {{"id", "r"}});
    ok(*f.s, "source.add", {{"id", "h"}, {"catalog", "tensor:heat_w"}});
    ok(*f.s, "panel.create", {{"id", "p"}, {"primitive", "heatmap"}, {"bind", {{"values", "h"}}}});
    CHECK(err(*f.s, "panel.render", {{"id", "zz"}}) == "panel.not_found");
    const json png = ok(*f.s, "panel.render", {{"id", "p"}, {"width", 320}, {"height", 200}});
    CHECK(png["width"] == 320);
    CHECK(host->renders == 1);
    REQUIRE(host->last.slots.size() == 1);
    CHECK(host->last.slots[0].tensor.shape() == std::vector<int64_t>{2, 3});
    // A panel whose source is offline reaches the host with its resolve
    // error, so the host draws it in that state.
    ok(*f.s, "source.add", {{"id", "live"}, {"catalog", "live:attn"}});
    ok(*f.s, "panel.create", {{"id", "q"}, {"primitive", "heatmap"}, {"bind", {{"values", "live"}}}});
    CHECK(ok(*f.s, "panel.render", {{"id", "q"}})["width"] == 800);
    CHECK(host->renders == 2);
    CHECK(host->last.error.code == "resolve.unavailable");

    f.s->set_host(nullptr);
    CHECK(err(*f.s, "app.info") == "command.not_bound");
}

TEST_CASE("the journal records the session, and a replay rebuilds the open spec") {
    Fixture f("journal");
    ok(*f.s, "data.import", {{"paths", {"data/metrics.csv"}}});
    ok(*f.s, "spec.new", {{"id", "j"}});
    ok(*f.s, "source.add", {{"id", "m"}, {"catalog", "table:metrics"}});
    ok(*f.s, "source.add", {{"id", "x"}, {"sql", "SELECT 1 AS one"}, {"dry_run", true}});
    ok(*f.s, "panel.create", {{"id", "p"}, {"primitive", "table"}, {"bind", {{"rows", "m"}}}});
    ok(*f.s, "panel.patch", {{"id", "p"}, {"patch", {{"title", "Rows"}}}});
    ok(*f.s, "history.undo");
    err(*f.s, "panel.create", {{"id", "p"}, {"primitive", "series"}, {"bind", {{"y", "m.loss"}}}});
    const spec::Spec expected = f.s->open_spec()->spec;
    const JournalRead read = read_journal(f.ws + "/journal/commands.jsonl");
    REQUIRE(read.error.ok());
    CHECK(read.problems.empty());
    CHECK(read.events.size() == 7);  // five do events, one failure, one undo; the dry run left no line
    f.s->close();
    CHECK(invoke(*f.s, "spec.get").error.code == "command.shutdown");

    const SpecReplay replay = replay_spec_journal(f.ws);
    REQUIRE_MESSAGE(replay.error.ok(), replay.error.message);
    REQUIRE(replay.spec);
    CHECK(*replay.spec == expected);
}

TEST_CASE("running a Python loader asks every origin but the person's own, per invocation") {
    Fixture f("python");
    const std::string python = CALIPER_BUILDER_TEST_PYTHON;
    write_text(f.ws + "/scripts/s.py", "def load(path, out):\n    out.tensor('w', [1.0, 2.0, 3.0])\n");
    write_text(f.ws + "/data/in.bin", "anything");
    auto recipe = [&](const std::string& id) {
        return "caliper_loader = 0\n[loader]\nid = \"" + id +
               "\"\n[input]\npath = \"data/in.bin\"\nreader = \"python\"\n[reader]\nscript = \"scripts/s.py\"\n" +
               (python.empty() ? std::string() : "python = \"" + python + "\"\n") + "[[output]]\nname = \"" + id +
               "_w\"\ntensor = \"w\"\n";
    };
    // Saving a Python loader runs nothing, so it needs nothing.
    ok(*f.s, "loader.create", {{"text", recipe("py")}}, kMcp);

    // loader.run from mcp asks; a denial runs nothing and records no status.
    const auto asked = invoke(*f.s, "loader.run", {{"id", "py"}}, kMcp);
    CHECK(asked.error.code == "command.needs_approval");
    CHECK(f.s->registry().deny(asked.approval).error.code == "command.denied");
    CHECK(ok(*f.s, "loader.describe", {{"id", "py"}})["status"] == "never_run");
    // loader.create with run asks too, though loader.create is not destructive.
    const auto create_asked = invoke(*f.s, "loader.create", {{"text", recipe("py2")}, {"run", true}}, kAgent);
    CHECK(create_asked.error.code == "command.needs_approval");
    CHECK(f.s->registry().deny(create_asked.approval).error.code == "command.denied");
    CHECK(err(*f.s, "loader.describe", {{"id", "py2"}}) == "loader.not_found");
    // loader.update with run of a Python loader asks, as every loader.update does.
    CHECK(invoke(*f.s, "loader.update", {{"id", "py"}, {"text", recipe("py")}, {"run", true}}, kApi()).error.code ==
          "command.needs_approval");

    // A policy that lets everything through does not give the permission: only
    // an approval of the session's own question does, so the store refuses.
    f.s->registry().set_approval_policy([](const CommandDescriptor&, const CommandInvocation&) {
        return ApprovalDecision::allow;
    });
    CHECK(err(*f.s, "loader.run", {{"id", "py"}}, kMcp) == "loader.python_needs_approval");
    f.reopen();

    if (python.empty()) {
        MESSAGE("Skipped: CALIPER_BUILDER_TEST_PYTHON is not set, so the approved runs do not run Python.");
        return;
    }
    // Approved, the script runs once.
    const auto again = invoke(*f.s, "loader.run", {{"id", "py"}}, kMcp);
    REQUIRE(again.error.code == "command.needs_approval");
    const auto approved = f.s->registry().approve(again.approval);
    INFO(approved.error.code << ": " << approved.error.message);
    REQUIRE(approved.ok);
    CHECK(approved.result["outputs"][0]["id"] == "tensor:py_w");
    CHECK(ok(*f.s, "catalog.sample", {{"id", "tensor:py_w"}})["slice"]["values"] == json{1.0, 2.0, 3.0});
    // The same invocation sent again asks again.
    CHECK(invoke(*f.s, "loader.run", {{"id", "py"}}, kMcp).error.code == "command.needs_approval");
    // The person's own run needs no question.
    CHECK(ok(*f.s, "loader.run", {{"id", "py"}}, kUi)["outputs"][0]["id"] == "tensor:py_w");
    const auto created = invoke(*f.s, "loader.create", {{"text", recipe("py3")}, {"run", true}}, kAgent);
    REQUIRE(created.error.code == "command.needs_approval");
    REQUIRE(f.s->registry().approve(created.approval).ok);
    CHECK(ok(*f.s, "loader.describe", {{"id", "py3"}})["status"] == "ok");
    // A redo puts the recipe back but never runs the script.
    ok(*f.s, "history.undo");
    CHECK(err(*f.s, "loader.describe", {{"id", "py3"}}) == "loader.not_found");
    ok(*f.s, "history.redo");
    CHECK(ok(*f.s, "loader.describe", {{"id", "py3"}})["status"] == "never_run");
}

TEST_CASE("a dry run and an edit that changes nothing leave no trace in the history or the journal") {
    Fixture f("no-effect");
    ok(*f.s, "data.import", {{"paths", {"data/metrics.csv"}}});
    ok(*f.s, "spec.new", {{"id", "n"}});
    ok(*f.s, "source.add", {{"id", "m"}, {"catalog", "table:metrics"}});
    ok(*f.s, "panel.create", {{"id", "p"}, {"primitive", "series"}, {"bind", {{"y", "m.loss"}}}});
    ok(*f.s, "panel.patch", {{"id", "p"}, {"patch", {{"title", "A"}}}});
    ok(*f.s, "history.undo");
    const std::string journal = f.ws + "/journal/commands.jsonl";
    const std::size_t lines = read_journal(journal).events.size();
    const std::size_t steps = f.s->registry().history_state().undo.size();

    // An agent previews two changes, one of them refused; the person's redo survives.
    const json preview = ok(*f.s, "panel.patch", {{"id", "p"}, {"patch", {{"title", "B"}}}, {"dry_run", true}}, kAgent);
    CHECK(preview["applied"] == false);
    CHECK(preview["diff"].get<std::string>().find("+title = \"B\"") != std::string::npos);
    CHECK(err(*f.s, "panel.remove", {{"id", "zz"}, {"dry_run", true}}, kAgent) == "panel.not_found");
    CHECK(err(*f.s, "source.add", {{"id", "m"}, {"sql", "SELECT 1"}, {"dry_run", true}}, kMcp) == "source.exists");
    CHECK(read_journal(journal).events.size() == lines);
    CHECK(f.s->registry().history_state().can_redo);
    CHECK(f.s->registry().history_state().undo.size() == steps);
    ok(*f.s, "history.redo");
    CHECK(f.s->open_spec()->spec.find_panel("p")->title == std::optional<std::string>("A"));

    // An edit whose diff is empty files no step and writes no line either.
    const std::size_t lines_after_redo = read_journal(journal).events.size();
    const std::size_t steps_after_redo = f.s->registry().history_state().undo.size();
    const json same = ok(*f.s, "panel.patch", {{"id", "p"}, {"patch", {{"title", "A"}}}});
    CHECK(same["applied"] == true);
    CHECK(same["diff"] == "");
    CHECK(f.s->registry().history_state().undo.size() == steps_after_redo);
    CHECK(read_journal(journal).events.size() == lines_after_redo);
    // A real edit after them is journaled and undone as usual.
    ok(*f.s, "panel.patch", {{"id", "p"}, {"patch", {{"title", "C"}}}});
    CHECK(read_journal(journal).events.size() == lines_after_redo + 1);
    ok(*f.s, "history.undo");
    CHECK(f.s->open_spec()->spec.find_panel("p")->title == std::optional<std::string>("A"));

    // Replay still gives the open spec.
    const spec::Spec expected = f.s->open_spec()->spec;
    f.s->close();
    const SpecReplay replay = replay_spec_journal(f.ws);
    REQUIRE(replay.error.ok());
    REQUIRE(replay.spec);
    CHECK(*replay.spec == expected);
}
