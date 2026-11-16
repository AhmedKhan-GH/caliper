// Keeps the command table in line with the spec format: every source, panel
// and layout of every valid spec fixture, turned into the arguments of
// source.add, panel.create, panel.patch and layout.set, passes the table's
// schemas, and the result forms pass the result schemas. Values that the spec
// library refuses and that the schema subset can express are refused by the
// table too.
#include <caliper_builder/commands/command_table.h>
#include <caliper_builder/commands/schema.h>
#include <caliper_builder/session/spec_edit.h>
#include <caliper_builder/spec/format.h>
#include <caliper_builder/spec/validate.h>

#include <doctest/doctest.h>

#include "helpers.h"

using namespace caliper_builder;
using nlohmann::json;
using session_test::parse_fixture;

namespace {

const char* const kFixtures[] = {"attention-entropy.spec.toml", "dashboard.spec.toml", "every-primitive.spec.toml",
                                 "minimal.spec.toml"};

const json& args_schema(const char* name) {
    const CommandDescriptor* d = find_command_descriptor(name);
    REQUIRE(d);
    return d->args;
}

const json& result_schema(const char* name) {
    const CommandDescriptor* d = find_command_descriptor(name);
    REQUIRE(d);
    return d->result;
}

void check_passes(const char* command_name, const json& args) {
    const std::string command = command_name;
    const Error e = validate_command_value(args_schema(command_name), args);
    CHECK_MESSAGE(e.ok(), command << " refuses " << args.dump() << ": " << e.where << ": " << e.message);
}

void check_refused(const char* command_name, const json& args, const std::string& where) {
    const std::string command = command_name;
    const Error e = validate_command_value(args_schema(command_name), args);
    CHECK_MESSAGE(e.code == "command.invalid_args", command << " accepts " << args.dump());
    CHECK(e.where == where);
}

json panel_args(json bind, json options, const char* primitive = "series") {
    json a{{"id", "p"}, {"primitive", primitive}, {"bind", std::move(bind)}};
    if (!options.is_null()) a["options"] = std::move(options);
    return a;
}

}  // namespace

TEST_CASE("every source, panel and layout of every valid fixture passes the table's schemas") {
    for (const char* fixture : kFixtures) {
        const std::string name = fixture;
        CAPTURE(name);
        const spec::Spec s = parse_fixture(name);
        REQUIRE(spec::validate(s).ok());
        for (const auto& src : s.sources) {
            CAPTURE(src.id);
            const json args = source_command_args(src);
            check_passes("source.add", args);
            json with_dry = args;
            with_dry["dry_run"] = true;
            check_passes("source.add", with_dry);
            // The same keys as a merge patch of the source.
            json patch = args;
            patch.erase("id");
            check_passes("source.patch", json{{"id", src.id}, {"patch", patch}});
            CHECK(validate_command_value(result_schema("source.patch")["properties"]["source"], source_result_json(src),
                                         "result.source")
                      .ok());
            // The arguments decode back to the same source, apart from `x`.
            auto back = spec::source_from_json(args);
            REQUIRE(back.ok());
            spec::Source plain = src;
            plain.x = nullptr;
            for (auto& t : plain.transforms) t.x = nullptr;
            CHECK(back.value == plain);
        }
        for (const auto& panel : s.panels) {
            CAPTURE(panel.id);
            const json args = panel_command_args(panel);
            check_passes("panel.create", args);
            json patch = args;
            patch.erase("id");
            check_passes("panel.patch", json{{"id", panel.id}, {"patch", patch}});
            const Error r = validate_command_value(result_schema("panel.patch")["properties"]["panel"],
                                                   panel_result_json(panel), "result.panel");
            CHECK_MESSAGE(r.ok(), r.where << ": " << r.message);
            auto back = spec::panel_from_json(args);
            REQUIRE(back.ok());
            spec::Panel plain = panel;
            plain.x = nullptr;
            plain.options.x = nullptr;
            CHECK(back.value == plain);
        }
        if (s.layout) {
            const json flat = layout_to_flat(*s.layout, s);
            check_passes("layout.set", json{{"layout", flat}});
            CHECK(validate_command_value(result_schema("layout.set")["properties"]["layout"], flat, "result.layout").ok());
            // The flat form gives back the same tree, apart from `x`.
            FlatLayoutParse back = layout_from_flat(flat, s);
            REQUIRE_MESSAGE(back.error.ok(), back.error.message);
            spec::Spec a = s;
            spec::Spec b = s;
            b.layout = back.root;
            // Layout extension tables are not carried by the flat form.
            std::function<void(spec::LayoutNode&)> strip = [&](spec::LayoutNode& n) {
                n.x = nullptr;
                for (auto& c : n.children) strip(c);
            };
            strip(*a.layout);
            CHECK(a.layout == b.layout);
        }
    }
}

TEST_CASE("spec.new and spec.set_text take what a spec's applet and text may hold") {
    const spec::Spec s = parse_fixture("dashboard.spec.toml");
    json a{{"id", s.applet.id}, {"title", *s.applet.title}, {"description", *s.applet.description}};
    check_passes("spec.new", a);
    check_passes("spec.set_text", json{{"text", spec::to_toml(s)}, {"dry_run", true}});
    check_refused("spec.new", json{{"id", "Bad"}}, "args.id");
    check_refused("spec.new", json{{"id", "ok"}, {"title", ""}}, "args.title");
    check_refused("spec.new", json{{"id", "ok"}, {"description", std::string(4097, 'a')}}, "args.description");
}

TEST_CASE("the panel schemas take every slot and option of the format") {
    check_passes("panel.create", panel_args({{"x", "m.step"}, {"y", json::array({"m.loss", "m.acc"})}},
                                            {{"x_label", "Step"}, {"y_label", "Loss"}, {"x_range", {0, 10}},
                                             {"y_range", {0.0, 1.5}}, {"x_log", false}, {"y_log", true}}));
    check_passes("panel.create", panel_args({{"positions", "cloud"}, {"color", "labels"}, {"size", "sizes"}},
                                            {{"x_label", "x"}, {"y_label", "y"}, {"z_label", "z"}, {"colormap", "rdbu"}},
                                            "points"));
    check_passes("panel.create", panel_args({{"values", "acts"}},
                                            {{"bins", 65536}, {"range", {-1, 1}}, {"y_log", true}}, "histogram"));
    check_passes("panel.create", panel_args({{"values", "img"}}, {{"colormap", "greys"}, {"range", {0, 255}}}, "image"));
    check_passes("panel.create", panel_args({{"rows", "q"}}, {{"columns", {"step", "value"}}}, "table"));
    check_passes("panel.create",
                 panel_args({{"value", "metrics.value"}}, {{"text", std::string(16384, 't')}}, "text"));
    check_passes("panel.create", panel_args(json::object(), {{"text", "Notes."}}, "text"));
    // A column name may hold dots and spaces.
    check_passes("panel.create", panel_args({{"y", "m.loss per step.mean"}}, nullptr));
    for (const char* cmap : {"viridis", "magma", "rdbu", "greys"})
        check_passes("panel.create", panel_args({{"values", "h"}}, {{"colormap", cmap}}, "heatmap"));
}

TEST_CASE("the table refuses what the spec refuses where the schema subset can say so") {
    check_refused("panel.create", panel_args({{"values", "h"}}, {{"colormap", "jet"}}, "heatmap"), "args.options.colormap");
    check_refused("panel.create", panel_args({{"values", "h"}}, {{"bins", 0}}, "histogram"), "args.options.bins");
    check_refused("panel.create", panel_args({{"values", "h"}}, {{"bins", 65537}}, "histogram"), "args.options.bins");
    check_refused("panel.create", panel_args({{"values", "h"}}, {{"range", {0, 1, 2}}}, "heatmap"), "args.options.range");
    check_refused("panel.create", panel_args({{"values", "h"}}, {{"range", {0, "1"}}}, "heatmap"),
                  "args.options.range[1]");
    check_refused("panel.create", panel_args(json::object(), {{"text", std::string(16385, 't')}}, "text"),
                  "args.options.text");
    check_refused("panel.create", panel_args({{"rows", "q"}}, {{"columns", json::array()}}, "table"),
                  "args.options.columns");
    check_refused("panel.create", panel_args({{"rows", "q"}}, {{"columns", {""}}}, "table"), "args.options.columns[0]");
    check_refused("panel.create", panel_args({{"y", "m.loss"}}, {{"x_label", std::string(257, 'l')}}),
                  "args.options.x_label");
    check_refused("panel.create", panel_args({{"y", "m.loss"}}, {{"legend", true}}), "args.options.legend");
    check_refused("panel.create", panel_args({{"y", "Metrics.loss"}}, nullptr), "args.bind.y");
    check_refused("panel.create", panel_args({{"y", "m."}}, nullptr), "args.bind.y");
    check_refused("panel.create", panel_args({{"y", json::array()}}, nullptr), "args.bind.y");
    json many = json::array();
    for (int i = 0; i < 17; ++i) many.push_back("m.c" + std::to_string(i));
    check_refused("panel.create", panel_args({{"y", many}}, nullptr), "args.bind.y");
    check_refused("panel.create", panel_args({{"rows", "q.step"}}, nullptr, "table"), "args.bind.rows");
    check_refused("panel.create", panel_args({{"x", std::string(321, 'a')}, {"y", "m.loss"}}, nullptr), "args.bind.x");
    check_refused("panel.create", panel_args({{"wheel", "m.loss"}}, nullptr), "args.bind.wheel");
    check_refused("panel.create", panel_args({{"values", "h"}}, nullptr, "scene"), "args.primitive");
    json titled = panel_args({{"y", "m.loss"}}, nullptr);
    titled["title"] = "";
    check_refused("panel.create", titled, "args.title");

    check_refused("source.add", json{{"id", "a"}, {"catalog", "recording:x"}}, "args.catalog");
    check_refused("source.add", json{{"id", "a"}, {"catalog", "table:" + std::string(251, 'n')}}, "args.catalog");
    check_refused("source.add", json{{"id", "a"}, {"sql", ""}}, "args.sql");
    check_refused("source.add",
                  json{{"id", "a"}, {"from", "b"}, {"transform", {{{"op", "slice"}, {"axis", 8}, {"index", 0}}}}},
                  "args.transform[0].axis");
    check_refused("source.add",
                  json{{"id", "a"}, {"from", "b"}, {"transform", {{{"op", "reduce"}, {"axis", {0, -9}}, {"fn", "sum"}}}}},
                  "args.transform[0].axis[1]");
    check_refused("source.add",
                  json{{"id", "a"}, {"from", "b"}, {"transform", {{{"op", "slice"}, {"axis", 0}, {"step", 0}}}}},
                  "args.transform[0].step");
    check_refused("source.add",
                  json{{"id", "a"}, {"from", "b"}, {"transform", {{{"op", "slice"}, {"axis", 0}, {"index", 9007199254740992.0}}}}},
                  "args.transform[0].index");
    check_refused("source.add",
                  json{{"id", "a"}, {"from", "b"}, {"transform", {{{"op", "reduce"}, {"axis", 0}, {"fn", "median"}}}}},
                  "args.transform[0].fn");
    check_refused("source.add", json{{"id", "a"}, {"from", "b"}, {"transform", {{{"op", "normalise"}, {"mode", "max"}}}}},
                  "args.transform[0].mode");
    json transforms = json::array();
    for (int i = 0; i < 17; ++i) transforms.push_back({{"op", "normalise"}, {"mode", "l2"}});
    check_refused("source.add", json{{"id", "a"}, {"from", "b"}, {"transform", transforms}}, "args.transform");

    json node{{"id", "n1"}, {"split", "tabs"}, {"ratio", nullptr}, {"title", ""}, {"children", {"a"}}};
    check_refused("layout.set", json{{"layout", {{"root", "n1"}, {"nodes", {node}}}}}, "args.layout.nodes[0].title");
}

TEST_CASE("command_table.json carries the aligned schemas") {
    const json file = json::parse(session_test::read_text(std::string(CALIPER_BUILDER_SOURCE_DIR) +
                                                          "/commands/command_table.json"));
    const json* create = nullptr;
    for (const auto& c : file["commands"])
        if (c["name"] == "panel.create") create = &c;
    REQUIRE(create);
    const json& options = (*create)["args"]["properties"]["options"]["properties"];
    CHECK(options["colormap"]["enum"] == json({"viridis", "magma", "rdbu", "greys"}));
    CHECK(options["bins"]["maximum"] == 65536);
    CHECK(options["text"]["maxLength"] == 16384);
    for (const char* key : {"z_label", "x_range", "y_range", "range", "x_log", "y_log"}) CHECK(options.contains(key));
    CHECK((*create)["args"]["properties"].contains("dry_run"));
    CHECK((*create)["result"]["required"] == json({"id", "warnings", "diff", "applied"}));
}
