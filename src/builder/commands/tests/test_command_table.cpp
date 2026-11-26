// Tests of the descriptor table: every command of v0 is there once with a
// well-formed name and its flags; every schema is within the subset, closed
// and described, as strict tools need; each example passes its schema; no
// property is named like a secret; the catalog's own JSON forms pass the
// catalog commands' result schemas; and the checked-in command_table.json is
// exactly command_table_to_json().
#include <caliper_builder/catalog/catalog.h>
#include <caliper_builder/catalog/catalog_entry.h>
#include <caliper_builder/catalog/tensor_sample.h>
#include <caliper_builder/commands/command_table.h>
#include <caliper_builder/commands/registry.h>
#include <caliper_builder/commands/schema.h>

#include <doctest/doctest.h>

#include <algorithm>
#include <cstring>
#include <fstream>
#include <set>
#include <sstream>
#include <string>

using caliper_builder::CommandDescriptor;
using nlohmann::json;
namespace cb = caliper_builder;

namespace {

const char* const kV0[] = {
    "app.info",        "app.applets",     "app.launch",     "catalog.list",  "catalog.describe", "catalog.sample",
    "data.query",      "data.import",     "loader.list",    "loader.describe", "loader.create",  "loader.update",
    "loader.run",      "loader.delete",   "spec.new",       "spec.open",     "spec.get",         "spec.set_text",
    "spec.validate",   "source.add",      "source.patch",   "source.remove", "panel.create",     "panel.patch",
    "panel.remove",    "panel.render",    "panel.data",     "layout.place",  "layout.set",       "layout.lock",
    "preview.begin",
    "preview.diff",    "preview.commit",  "preview.discard", "applet.save",  "applet.open",      "applet.list",
    "review.status",   "capture.frame",   "control.create", "control.patch", "control.remove",
    // The eighteen commands of design/instrument-interface.md, Commands.
    "state.create",    "state.patch",     "state.remove",   "state.set",     "state.get",        "control.set",
    "selection.set",   "selection.get",   "entity.set",     "entity.remove", "annotation.set",   "annotation.remove",
    "cursor.get",      "cursor.set",      "cursor.step",    "camera.get",    "camera.set",       "camera.save",
    "run.state",
    "run.set_param",   "run.action",      "run.pause",      "run.list",      "run.start",        "run.fork",
    "run.stop",        "run.logs",        "events.wait",
    "program.create",  "program.patch",   "program.remove", "program.status", "piece.check",    "piece.list",
    "piece.show",      "piece.save",      "piece.copy",     "piece.update",  "piece.remove",     "program.read",
    "feature.dictionary", "feature.show", "feature.examples", "trace.run", "trace.status", "trace.read", "patch.sweep", "trace.validate",
    "intervene.points", "intervene.set",
    "intervene.clear", "intervene.list",  "history.undo",   "history.redo",  "history.list",
};

std::set<std::string> names_where(bool (*pred)(const CommandDescriptor&)) {
    std::set<std::string> out;
    for (const auto& d : cb::command_table())
        if (pred(d)) out.insert(d.name);
    return out;
}

// Every property name in a schema, at any depth.
void property_names(const json& schema, std::set<std::string>& out) {
    if (!schema.is_object()) return;
    if (schema.contains("properties"))
        for (const auto& item : schema["properties"].items()) {
            out.insert(item.key());
            property_names(item.value(), out);
        }
    if (schema.contains("items")) property_names(schema["items"], out);
}

std::vector<std::string> open_paths_of(const std::string& name) {
    std::vector<std::string> paths;
    for (const auto& p : cb::command_table_open_paths())
        if (p.first == name) paths.push_back(p.second);
    return paths;
}

}  // namespace

TEST_CASE("the table builds without faults and holds every command of v0 once") {
    for (const auto& e : cb::command_table_build_errors()) FAIL(e);
    const auto& table = cb::command_table();
    REQUIRE(table.size() == sizeof kV0 / sizeof kV0[0]);
    std::set<std::string> seen;
    for (std::size_t i = 0; i < table.size(); ++i) {
        CHECK(table[i].name == kV0[i]);
        CHECK(seen.insert(table[i].name).second);
        CHECK(cb::check_command_name(table[i].name).ok());
        CHECK(cb::command_name_from_tool(cb::command_tool_name(table[i].name)) == table[i].name);
        CHECK(cb::find_command_descriptor(table[i].name) == &table[i]);
    }
    CHECK(cb::find_command_descriptor("panel.explode") == nullptr);
}

TEST_CASE("every descriptor is sound and its example passes its schema") {
    for (const auto& d : cb::command_table()) {
        CAPTURE(d.name);
        const auto err = cb::check_command_descriptor(d);
        CHECK_MESSAGE(err.ok(), (err.where + ": " + err.message));
        REQUIRE(!d.examples.empty());
        for (const auto& example : d.examples) CHECK(cb::validate_command_value(d.args, example).ok());
    }
}

TEST_CASE("every schema is closed, lists its required properties and describes every property") {
    for (const auto& d : cb::command_table()) {
        CAPTURE(d.name);
        CHECK(cb::check_command_schema(d.args).ok());
        CHECK(cb::check_command_schema(d.result).ok());
        const auto args = cb::check_closed_command_schema(d.args, "args");
        CHECK_MESSAGE(args.ok(), (args.where + ": " + args.message));
        const auto result = cb::check_closed_command_schema(d.result, "result", open_paths_of(d.name));
        CHECK_MESSAGE(result.ok(), (result.where + ": " + result.message));
        CHECK(d.summary.size() > 40);
        CHECK(d.summary.size() < 1200);
        CHECK(d.summary.back() == '.');
    }
    // The spec payload is open only at its declared result path.
    const auto* get = cb::find_command_descriptor("spec.get");
    REQUIRE(get);
    CHECK(get->result["properties"]["json"]["additionalProperties"] == true);
    CHECK(cb::check_closed_command_schema(get->result, "result").where == "result.json");
}

TEST_CASE("no command takes or returns a value named like a secret") {
    std::set<std::string> names;
    for (const auto& d : cb::command_table()) {
        property_names(d.args, names);
        property_names(d.result, names);
    }
    for (const auto& n : names) {
        CAPTURE(n);
        // The event key `key` and the control option `keys` name keyboard shortcuts, not secrets.
        if (n == "key" || n == "keys" || n == "positive_token" || n == "negative_token") continue;
        for (const char* bad : {"token", "secret", "password", "api_key", "apikey", "authorization", "key"})
            CHECK(n.find(bad) == std::string::npos);
    }
}

TEST_CASE("the flags follow the design page") {
    CHECK(names_where([](const CommandDescriptor& d) { return d.destructive; }) ==
          std::set<std::string>{"loader.update", "loader.run", "loader.delete", "applet.save", "piece.remove"});
    CHECK(names_where([](const CommandDescriptor& d) { return d.mutates; }) ==
          std::set<std::string>{"data.import", "loader.create", "loader.update", "loader.run", "loader.delete",
                                "spec.new", "spec.open", "spec.set_text", "source.add", "source.patch",
                                "source.remove", "panel.create", "panel.patch", "panel.remove", "layout.place",
                                "layout.set", "layout.lock", "preview.commit", "applet.save", "applet.open", "control.create",
                                "control.patch", "control.remove", "program.create",
                                "program.patch", "program.remove", "history.undo", "history.redo", "state.create",
                                "state.patch", "state.remove", "entity.set", "entity.remove", "annotation.set",
                                "annotation.remove", "camera.save", "piece.save", "piece.copy", "piece.update",
                                "piece.remove"});
    CHECK(names_where([](const CommandDescriptor& d) { return d.frame_thread; }) ==
          std::set<std::string>{"app.applets", "app.launch", "spec.new", "spec.open", "spec.get", "spec.set_text",
                                "spec.validate", "source.add", "source.patch", "source.remove", "panel.create",
                                "panel.patch", "panel.remove", "panel.render", "panel.data", "layout.place", "layout.set", "layout.lock",
                                "preview.begin", "preview.diff", "preview.commit", "preview.discard",
                                "applet.save", "applet.open", "capture.frame", "control.create", "control.patch",
                                "control.remove", "state.create", "state.patch", "state.remove", "state.set",
                                "state.get", "control.set", "selection.set", "selection.get", "entity.set",
                                "entity.remove", "annotation.set", "annotation.remove", "cursor.get", "cursor.set",
                                "cursor.step", "camera.get", "camera.set", "camera.save", "run.state", "run.set_param", "run.action", "run.pause",
                                "run.list", "run.start", "run.fork", "run.stop", "run.logs",
                                "program.create", "program.patch", "program.remove", "program.status",
                                "feature.dictionary", "feature.show", "feature.examples", "trace.run", "trace.status", "trace.read", "patch.sweep", "trace.validate",
                                "piece.list", "piece.show", "piece.save", "program.read", "intervene.points", "intervene.set", "intervene.clear", "intervene.list",
                                "history.undo", "history.redo", "history.list"});
    // Every destructive command mutates.
    for (const auto& d : cb::command_table())
        if (d.destructive) CHECK(d.mutates);
}

TEST_CASE("a registry of the table refuses unbound commands but checks their arguments first") {
    cb::CommandRegistry registry;
    REQUIRE(registry.add_all(cb::command_table()).ok());
    CHECK(registry.descriptors().size() == cb::command_table().size());
    for (const auto& d : cb::command_table()) {
        CAPTURE(d.name);
        cb::CommandInvocation inv;
        inv.name = d.name;
        inv.args = d.examples.at(0);
        const auto r = registry.invoke(inv);
        if (d.name.rfind("history.", 0) == 0) {
            CHECK((r.ok || r.error.code == "history.nothing_to_undo" || r.error.code == "history.nothing_to_redo"));
        } else {
            CHECK(r.error.code == "command.not_bound");
        }
    }
    cb::CommandInvocation bad;
    bad.name = "catalog.list";
    bad.args = {{"limit", 5000}};
    CHECK(registry.invoke(bad).error.code == "command.invalid_args");
    bad.args = {{"kind", "recording"}};
    CHECK(registry.invoke(bad).error.where == "args.kind");
    bad.name = "panel.patch";
    bad.args = {{"id", "heat"}, {"patch", {{"id", "renamed"}}}};
    CHECK(registry.invoke(bad).error.where == "args.patch.id");
}

TEST_CASE("the catalog's JSON forms pass the catalog commands' result schemas") {
    const auto* describe = cb::find_command_descriptor("catalog.describe");
    const auto* sample = cb::find_command_descriptor("catalog.sample");
    const auto* list = cb::find_command_descriptor("catalog.list");
    REQUIRE((describe && sample && list));

    cb::CatalogEntry e;
    e.kind = cb::CatalogKind::tensor;
    e.name = "attn_weights";
    e.id = "tensor:attn_weights";
    e.title = "attn_weights";
    e.origin.loader = "run42-weights";
    e.origin.file = "model.safetensors";
    e.origin.run = 3;
    e.dtype = cb::CatalogDtype::f32;
    e.shape = std::vector<std::int64_t>{2, 3};
    e.bytes = 24;
    e.wire = true;
    e.updated_at = "2026-11-15T12:00:00Z";
    REQUIRE(cb::validate_catalog_entry(e).ok());
    const json entry = cb::catalog_entry_to_json(e);
    CHECK(cb::validate_command_value(describe->result, json{{"entry", entry}}, "result").ok());
    CHECK(cb::validate_command_value(list->result,
                                     json{{"entries", {entry}}, {"truncated", false}, {"warnings", json::array()}},
                                     "result")
              .ok());

    const float values[6] = {1, 2, 3, 4, 5, 6};
    const auto t = cb::sample_tensor(cb::CatalogDtype::f32, {2, 3}, values, sizeof values);
    REQUIRE(t.error.ok());
    cb::CatalogSample s;
    s.id = e.id;
    s.kind = e.kind;
    s.dtype = e.dtype;
    s.shape = e.shape;
    s.stats = t.stats;
    s.slice = t.slice;
    const auto tensor_check = cb::validate_command_value(sample->result, cb::catalog_sample_to_json(s), "result");
    CHECK_MESSAGE(tensor_check.ok(), (tensor_check.where + ": " + tensor_check.message));

    cb::CatalogSample rows;
    rows.id = "table:x";
    rows.kind = cb::CatalogKind::table;
    rows.columns = std::vector<cb::CatalogColumn>{{"step", "BIGINT"}, {"loss", "DOUBLE"}};
    rows.rows = json::array({json::array({1, 0.5}), json::array({2, "nan"})});
    rows.total_rows = 2;
    const auto table_check = cb::validate_command_value(sample->result, cb::catalog_sample_to_json(rows), "result");
    CHECK_MESSAGE(table_check.ok(), (table_check.where + ": " + table_check.message));
}

TEST_CASE("command_table.json is exactly the table's JSON, so the sidecar sees the same tools") {
    const std::string path = std::string(CALIPER_BUILDER_SOURCE_DIR) + "/commands/command_table.json";
    std::ifstream in(path, std::ios::binary);
    REQUIRE_MESSAGE(in.good(), "command_table.json is missing; regenerate it with caliper_builder_command_table --write");
    std::stringstream ss;
    ss << in.rdbuf();
    const std::string file = ss.str();
    const std::string expected = cb::command_table_json_text();
    CHECK_MESSAGE(file == expected,
                  "command_table.json differs from the C++ table; regenerate it with "
                  "caliper_builder_command_table --write src/builder/commands/command_table.json");
    CHECK(file.back() == '\n');

    const json body = cb::command_table_to_json();
    CHECK(body["version"] == 0);
    CHECK(body["commands"].size() == cb::command_table().size());
    CHECK(body["commands"][0]["name"] == "app.info");
    for (const auto& c : body["commands"]) {
        CHECK(c.size() == 10);
        for (const char* k : {"name", "summary", "args", "result", "examples", "mutates", "destructive", "frame_thread",
                              "agents", "effect"})
            CHECK(c.contains(k));
        CHECK(c["agents"] == (c["name"] != "history.undo" && c["name"] != "history.redo"));
    }
}

TEST_CASE("M8 command schemas retain inference resources and structured intervention fields") {
    const auto* start = cb::find_command_descriptor("run.start");
    REQUIRE(start);
    CHECK_FALSE(cb::validate_command_value(start->args,{{"kind","inference"},{"model","fixture"},{"dictionary","dict"},{"prompt","AB"}},"args"));
    CHECK(cb::validate_command_value(start->args,{{"kind","inference"},{"model",true}},"args"));
    CHECK(cb::validate_command_value(start->args,{{"kind","inference"},{"prompt",std::string(4097,'x')}},"args"));
    for (const auto* name : {"patch.sweep","trace.validate"}) {
        const auto* descriptor = cb::find_command_descriptor(name);
        REQUIRE(descriptor);
        CHECK(descriptor->effect == cb::CommandEffect::run);
        CHECK_FALSE(cb::validate_command_value(descriptor->args,descriptor->examples.front(),"args"));
    }
}
