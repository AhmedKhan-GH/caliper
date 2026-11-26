// The checked-in JSON Schema against the library: every valid spec's canonical
// JSON passes the schema, and the documents the schema refuses are invalid for
// the library too. The schema is read through a test-only checker
// (schema_check.h) that implements exactly the keywords the schema uses. Each
// document it checks is also written to schema-docs/ in the test's build
// folder (".bad.json" for the refused ones), so that a reference validator can
// check the same files.
#include <filesystem>
#include <fstream>
#include <ostream>
#include <string>

#include <caliper_builder/spec/patch.h>

#include <doctest/doctest.h>

#include "helpers.h"
#include "schema_check.h"

using namespace spec_test;
using nlohmann::json;

namespace {

const json& schema() {
    static const json s = json::parse(read_file(CALIPER_SPEC_SCHEMA));
    return s;
}

const SchemaChecker& checker() {
    static const SchemaChecker c(schema());
    return c;
}

std::string join_errors(const std::vector<std::string>& errors) {
    std::string s;
    for (const auto& e : errors) s += e + "\n";
    return s;
}

void keep(const json& doc, bool valid) {
    static int n = 0;
    const std::filesystem::path dir = std::filesystem::path(CALIPER_BUILDER_TEST_TMP) / "schema-docs";
    std::error_code ec;
    std::filesystem::create_directories(dir, ec);
    std::ofstream(dir / (std::to_string(n++) + (valid ? ".json" : ".bad.json"))) << doc.dump(1);
}

// The library's verdict on a JSON document: it parses and validates without error.
bool library_accepts(const json& doc) {
    auto parsed = cs::from_json(doc);
    if (!parsed.ok()) return false;
    return cs::validate(parsed.spec).ok();
}

}  // namespace

TEST_CASE("the schema is draft 2020-12 and uses only keywords the checker implements") {
    CHECK(schema().at("$schema") == "https://json-schema.org/draft/2020-12/schema");
    CHECK_NOTHROW(SchemaChecker{schema()});
    // The schema relies on these; if it stops using one, this list is updated on purpose.
    for (const char* k : {"$ref", "type", "const", "enum", "properties", "additionalProperties", "required", "items",
                          "minItems", "maxItems", "uniqueItems", "minLength", "maxLength", "pattern", "minimum",
                          "maximum", "anyOf", "oneOf", "if", "then"}) {
        CAPTURE(k);
        CHECK(checker().keywords().count(k) == 1);
    }
    CHECK_THROWS(SchemaChecker{json::parse(R"({"type": "object", "patternProperties": {}})")});
}

TEST_CASE("the checker follows draft 2020-12 for the keywords it implements") {
    SchemaChecker c(json::parse(R"({
        "$defs": {"n": {"type": "integer", "minimum": 0, "maximum": 3}},
        "type": "object",
        "required": ["a"],
        "additionalProperties": false,
        "properties": {
            "a": {"$ref": "#/$defs/n"},
            "b": {"oneOf": [{"type": "number"}, {"type": "integer"}]},
            "c": {"anyOf": [{"const": "x"}, {"enum": [1, 2]}]},
            "d": {"type": "array", "minItems": 1, "maxItems": 2, "uniqueItems": true, "items": {"type": "string",
                  "minLength": 1, "maxLength": 2}},
            "g": {"type": "string", "pattern": "^[a-z]+$"},
            "e": {"type": "object", "if": {"required": ["k"]}, "then": {"properties": {"m": false}},
                  "else": {"required": ["m"]}}
        }
    })"));
    CHECK(c.passes(json::parse(R"({"a": 3})")));
    CHECK(c.passes(json::parse(R"({"a": 2.0})")));  // an integer may be written 2.0
    CHECK_FALSE(c.passes(json::parse(R"({"a": 2.5})")));
    CHECK_FALSE(c.passes(json::parse(R"({"a": 4})")));
    CHECK_FALSE(c.passes(json::parse(R"({})")));
    CHECK_FALSE(c.passes(json::parse(R"({"a": 1, "z": 1})")));
    CHECK(c.passes(json::parse(R"({"a": 1, "b": 1.5})")));
    CHECK_FALSE(c.passes(json::parse(R"({"a": 1, "b": 1})")));  // both branches of oneOf match
    CHECK(c.passes(json::parse(R"({"a": 1, "c": 2.0})")));     // enum compares numbers by value
    CHECK_FALSE(c.passes(json::parse(R"({"a": 1, "c": "y"})")));
    CHECK(c.passes(json::parse(R"({"a": 1, "d": ["ab", "é"]})")));  // lengths in code points
    CHECK_FALSE(c.passes(json::parse(R"({"a": 1, "d": ["ab", "ab"]})")));
    CHECK_FALSE(c.passes(json::parse(R"({"a": 1, "d": ["abc"]})")));
    CHECK(c.passes(json::parse(R"({"a": 1, "g": "abc"})")));
    CHECK_FALSE(c.passes(json::parse(R"({"a": 1, "g": "aBc"})")));
    CHECK_FALSE(c.passes(json::parse(R"({"a": 1, "d": []})")));
    CHECK(c.passes(json::parse(R"({"a": 1, "e": {"k": 1}})")));
    CHECK_FALSE(c.passes(json::parse(R"({"a": 1, "e": {"k": 1, "m": 1}})")));
    CHECK_FALSE(c.passes(json::parse(R"({"a": 1, "e": {}})")));
    CHECK(c.passes(json::parse(R"({"a": 1, "e": {"m": 1}})")));
}

TEST_CASE("the canonical JSON of every valid fixture passes the schema") {
    for (const std::string& name : valid_fixtures()) {
        CAPTURE(name);
        auto parsed = cs::parse_toml(fixture(name));
        REQUIRE(parsed.ok());
        REQUIRE(cs::validate(parsed.spec, fixture_resolver()).ok());
        const json doc = cs::to_json(parsed.spec);
        keep(doc, true);
        const auto errors = checker().errors(doc);
        CHECK_MESSAGE(errors.empty(), join_errors(errors));
    }
}

TEST_CASE("more valid specs pass the schema, canonical or not") {
    const std::vector<std::string> documents = {
        R"({"caliper_spec": 0, "applet": {"id": "a"}, "source": [], "panel": [], "x": {}})",
        R"({"caliper_spec": 0.0, "applet": {"id": "a", "title": "A", "description": ""}})",
        R"({"caliper_spec": 0, "applet": {"id": "a"},
            "source": [{"id": "t", "catalog": "tensor:t", "transform": [
                {"op": "slice", "axis": 0, "index": -1},
                {"op": "reduce", "axis": [2.0], "fn": "argmax", "keepdims": true},
                {"op": "normalise", "mode": "l2"}]}],
            "panel": [{"id": "v", "primitive": "text", "bind": {"value": "t"}},
                      {"id": "h", "primitive": "histogram", "bind": {"values": "t"}}],
            "layout": {"split": "horizontal", "children": ["v", "h"], "title": "Both"}})",
        R"({"caliper_spec": 0, "applet": {"id": "a"},
            "source": [{"id": "m", "catalog": "metric:run/loss"}],
            "panel": [{"id": "p", "primitive": "points",
                       "bind": {"x": "m.step", "y": "m.value", "z": "m.value", "color": "m.step", "size": "m.value"},
                       "options": {"colormap": "greys", "z_label": "", "x": {"k": [1, "two", true, {"n": null}]}}}],
            "layout": "p"})",
        R"({"caliper_spec": 0, "applet": {"id": "a"},
            "source": [{"id": "m", "sql": "SELECT 1 AS v"}],
            "panel": [{"id": "a1", "primitive": "table", "bind": {"rows": "m"}},
                      {"id": "a2", "primitive": "series", "bind": {"y": "m.v"}},
                      {"id": "a3", "primitive": "series", "bind": {"y": ["m.v"]}, "options": {"x_range": [0, 1]}}],
            "layout": {"split": "vertical", "children": ["a1", "a2", {"tabs": ["a3"]}]}})",
    };
    for (const std::string& text : documents) {
        CAPTURE(text);
        const json doc = json::parse(text);
        const bool extension_null = text.find("null") != std::string::npos;
        if (extension_null) {
            // A null inside x is refused by the library, and the schema refuses it too.
            CHECK_FALSE(library_accepts(doc));
            CHECK_FALSE(checker().passes(doc));
            keep(doc, false);
            continue;
        }
        REQUIRE(library_accepts(doc));
        keep(doc, true);
        const auto raw = checker().errors(doc);
        CHECK_MESSAGE(raw.empty(), join_errors(raw));
        const auto canonical = checker().errors(cs::to_json(cs::from_json(doc).spec));
        CHECK_MESSAGE(canonical.empty(), join_errors(canonical));
    }
}

TEST_CASE("documents the schema refuses are refused by the library too") {
    const std::string head = R"("caliper_spec": 0, "applet": {"id": "a"})";
    auto doc = [&](const std::string& rest) { return json::parse("{" + head + (rest.empty() ? "" : ", " + rest) + "}"); };
    const std::string tensor = R"("source": [{"id": "t", "catalog": "tensor:t"}])";
    const std::string table = R"("source": [{"id": "m", "catalog": "table:m"}])";
    const std::vector<json> documents = {
        json::parse(R"({"caliper_spec": 2, "applet": {"id": "a"}})"),
        json::parse(R"({"caliper_spec": 0})"),
        json::parse(R"({"caliper_spec": 0, "applet": {"id": "Bad"}})"),
        json::parse(R"({"caliper_spec": 0, "applet": {"id": "a", "title": ""}})"),
        doc(R"("panels": [])"),
        doc(R"("x": 3)"),
        doc(R"("source": [{"id": "s", "catalog": "tensor:t", "sql": "SELECT 1"}])"),
        doc(R"("source": [{"id": "s"}])"),
        doc(R"("source": [{"id": "s", "catalog": "gpu:util"}])"),
        doc(R"("source": [{"id": "s", "sql": ""}])"),
        doc(R"("source": [{"id": "s", "catalog": "tensor:t", "transform": [{"op": "slice", "axis": 0, "index": 1, "start": 0}]}])"),
        doc(R"("source": [{"id": "s", "catalog": "tensor:t", "transform": [{"op": "reduce", "axis": [0, 1], "fn": "argmax"}]}])"),
        doc(R"("source": [{"id": "s", "catalog": "tensor:t", "transform": [{"op": "reduce", "axis": [1, 1], "fn": "sum"}]}])"),
        doc(R"("source": [{"id": "s", "catalog": "tensor:t", "transform": [{"op": "transpose"}]}])"),
        doc(R"("source": [{"id": "s", "catalog": "tensor:t", "transform": [{"op": "slice", "axis": 8}]}])"),
        doc(R"("source": [{"id": "s", "catalog": "tensor:t", "transform": [{"op": "slice", "axis": 0, "step": 0}]}])"),
        doc(R"("source": [{"id": "s", "catalog": "tensor:t", "transform": [{"op": "normalise", "mode": "l2", "fn": "sum"}]}])"),
        doc(tensor + R"(, "panel": [{"id": "p", "primitive": "heatmap", "bind": {"values": "t"}, "options": {"bins": 4}}])"),
        doc(tensor + R"(, "panel": [{"id": "p", "primitive": "heatmap", "bind": {"values": "t"}, "options": {"colormap": "jet"}}])"),
        doc(tensor + R"(, "panel": [{"id": "p", "primitive": "pie", "bind": {"values": "t"}}])"),
        doc(tensor + R"(, "panel": [{"id": "p", "primitive": "series"}])"),
        doc(tensor + R"(, "panel": [{"id": "p", "primitive": "series", "bind": {"y": []}}])"),
        doc(tensor + R"(, "panel": [{"id": "p", "primitive": "series", "bind": {"y": "t", "z": "t"}}])"),
        doc(tensor + R"(, "panel": [{"id": "p", "primitive": "points", "bind": {"positions": "t", "x": "t"}}])"),
        doc(tensor + R"(, "panel": [{"id": "p", "primitive": "points", "bind": {"x": "t"}}])"),
        doc(tensor + R"(, "panel": [{"id": "p", "primitive": "text", "title": "No value"}])"),
        doc(tensor + R"(, "panel": [{"id": "p", "primitive": "histogram", "bind": {"values": "t"}, "options": {"bins": 0}}])"),
        doc(tensor + R"(, "panel": [{"id": "p", "primitive": "histogram", "bind": {"values": "t"}, "options": {"range": [1]}}])"),
        doc(tensor + R"(, "panel": [{"id": "p", "primitive": "histogram", "bind": {"values": "t"}, "options": {"y_log": 1}}])"),
        doc(tensor + R"(, "panel": [{"id": "p", "primitive": "histogram", "bind": {"values": "Bad binding"}}])"),
        doc(table + R"(, "panel": [{"id": "p", "primitive": "table", "bind": {"rows": "m.step"}}])"),
        doc(table + R"(, "panel": [{"id": "p", "primitive": "table", "bind": {"rows": "m"}, "options": {"columns": ["a", "a"]}}])"),
        doc(table + R"(, "panel": [{"id": "p", "primitive": "table", "bind": {"rows": "m"}, "options": {"columns": []}}])"),
        doc(tensor + R"(, "panel": [{"id": "p", "primitive": "histogram", "bind": {"values": "t"}}], "layout": {"split": "vertical", "ratio": 0.5, "children": ["p", "p", "p"]})"),
        doc(tensor + R"(, "panel": [{"id": "p", "primitive": "histogram", "bind": {"values": "t"}}], "layout": {"split": "vertical", "ratio": 0.99, "children": ["p", "p"]})"),
        doc(tensor + R"(, "panel": [{"id": "p", "primitive": "histogram", "bind": {"values": "t"}}], "layout": {"split": "vertical", "children": ["p"]})"),
        doc(tensor + R"(, "panel": [{"id": "p", "primitive": "histogram", "bind": {"values": "t"}}], "layout": {"tabs": []})"),
        doc(tensor + R"(, "panel": [{"id": "p", "primitive": "histogram", "bind": {"values": "t"}}], "layout": {"tabs": ["p"], "split": "vertical"})"),
        doc(tensor + R"(, "panel": [{"id": "p", "primitive": "histogram", "bind": {"values": "t"}}], "layout": 3)"),
        doc(R"("applet": {"id": "a", "title": null})"),
    };
    for (const json& d : documents) {
        CAPTURE(d.dump());
        CHECK_FALSE(checker().passes(d));
        CHECK_FALSE(library_accepts(d));
        keep(d, false);
    }
}

// ---- the JSON Schema of version 1 ---------------------------------------------

namespace {

// The schema of version 1 sits beside the fixtures' parent folder in the source tree.
const SchemaChecker& checker_v1() {
    static const SchemaChecker c(
        json::parse(read_file(std::string(CALIPER_SPEC_FIXTURES) + "/../../schema/caliper-spec-v1.schema.json")));
    return c;
}

}  // namespace

TEST_CASE("the text control schema accepts parameter bindings at the byte limit") {
    const auto document = [](const char* bind, int limit) {
        return json{{"caliper_spec", 1}, {"applet", {{"id", "text-control"}}},
                    {"control", json::array({{{"id", "prompt"}, {"kind", "text"}, {"bind", bind},
                        {"options", {{"max_length", limit}}}}})}};
    };
    auto prompt = document("param:prompt", 4096);
    std::string value;
    for (int i = 0; i < 2048; ++i) value += "\xc3\xa9";
    REQUIRE(value.size() == 4096);
    prompt["control"][0]["options"]["value"] = value;
    REQUIRE(library_accepts(prompt));
    const auto parsed = cs::from_json(prompt);
    REQUIRE(parsed.ok());
    for (const auto& doc : {prompt, cs::to_json(parsed.spec), document("value", 1024)}) {
        CHECK(library_accepts(doc));
        const auto errors = checker_v1().errors(doc);
        CHECK_MESSAGE(errors.empty(), join_errors(errors));
        keep(doc, true);
    }
    // The runtime validator enforces byte limits for each binding target.
    CHECK_FALSE(library_accepts(document("param:prompt", 4097)));
    CHECK_FALSE(library_accepts(document("value", 1025)));
    prompt["control"][0]["options"]["value"] = value + "\xc3\xa9";
    CHECK_FALSE(library_accepts(prompt));
    for (const char* bind : {"param:", "action:save", "intervene:head"}) {
        CHECK_FALSE(checker_v1().passes(document(bind, 4096)));
        CHECK_FALSE(library_accepts(document(bind, 4096)));
    }
}

TEST_CASE("the selected label schema accepts integer references and refuses malformed declarations") {
    const auto root = json::parse(read_file(std::string(CALIPER_SPEC_FIXTURES) + "/../../schema/caliper-spec-v1.schema.json"));
    const SchemaChecker selected(root.at("$defs").at("piece_draw").at("properties").at("text").at("properties").at("selected"));
    json value = {{"anchor", 32}, {"param", "selected_node"}, {"string_offset", 32}};
    REQUIRE(selected.passes(value));
    SUBCASE("required anchor") { value.erase("anchor"); }
    SUBCASE("required parameter") { value.erase("param"); }
    SUBCASE("negative anchor") { value["anchor"] = -1; }
    SUBCASE("fractional anchor") { value["anchor"] = 1.5; }
    SUBCASE("anchor limit") { value["anchor"] = 512; }
    SUBCASE("string limit") { value["string_offset"] = 4096; }
    SUBCASE("invalid parameter name") { value["param"] = "unknown.field"; }
    SUBCASE("unknown key") { value["other"] = 1; }
    SUBCASE("non-table selected") { value = false; }
    SUBCASE("optional offset") {
        value.erase("string_offset");
        CHECK(selected.passes(value));
        return;
    }
    CHECK_FALSE(selected.passes(value));
}

TEST_CASE("the schema of version 1 accepts the version 1 fixture and every version 0 fixture raised to 1") {
    auto v1 = cs::parse_toml(fixture("instruments.spec.toml"));
    REQUIRE(v1.ok());
    std::vector<std::string> errors = checker_v1().errors(cs::to_json(v1.spec));
    CHECK_MESSAGE(errors.empty(), join_errors(errors));
    CHECK_FALSE(checker().passes(cs::to_json(v1.spec)));   // version 0's schema refuses it
    for (const std::string& name : valid_fixtures()) {
        CAPTURE(name);
        auto parsed = cs::parse_toml(fixture(name));
        REQUIRE(parsed.ok());
        json doc = cs::to_json(parsed.spec);
        doc["caliper_spec"] = 1;
        errors = checker_v1().errors(doc);
        CHECK_MESSAGE(errors.empty(), join_errors(errors));
        CHECK(library_accepts(doc));
    }
}

TEST_CASE("documents the schema of version 1 refuses are refused by the library too") {
    const std::string head = R"("caliper_spec": 1, "applet": {"id": "a"})";
    auto doc = [&](const std::string& rest) { return json::parse("{" + head + ", " + rest + "}"); };
    const std::string scene = R"("panel": [{"id": "w", "primitive": "scene", "bind": {"scene": "live:model"}}])";
    const std::vector<json> documents = {
        doc(R"("control": [{"id": "k", "kind": "knob", "bind": "value"}])"),
        doc(R"("control": [{"id": "k", "kind": "slider"}])"),
        doc(R"("control": [{"id": "k", "kind": "slider", "bind": "action:save"}])"),
        doc(R"("control": [{"id": "k", "kind": "button", "bind": "param:lr"}])"),
        doc(R"("control": [{"id": "k", "kind": "button", "bind": "run:pause", "options": {"min": 0}}])"),
        doc(R"("control": [{"id": "k", "kind": "slider", "bind": "param:"}])"),
        doc(R"("control": [{"id": "k", "kind": "slider", "bind": "intervene:x"}])"),
        doc(R"("control": [{"id": "k", "kind": "slider", "bind": "value", "options": {"colormap": "magma"}}])"),
        doc(R"("control": {"id": "k"})"),
        doc(R"("panel": [{"id": "w", "primitive": "scene", "bind": {"scene": "model"}}])"),
        doc(R"("panel": [{"id": "w", "primitive": "scene"}])"),
        doc(R"("panel": [{"id": "w", "primitive": "scene", "bind": {"scene": "live:model", "x": "t"}}])"),
        doc(R"("panel": [{"id": "w", "primitive": "scene", "bind": {"scene": "live:model"}, "options": {"up": "+w"}}])"),
        doc(R"("panel": [{"id": "w", "primitive": "scene", "bind": {"scene": "live:model"}, "options": {"include": []}}])"),
        doc(R"("panel": [{"id": "w", "primitive": "scene", "bind": {"scene": "live:model"}, "options": {"view_camera": {"elevation": 90}}}])"),
        doc(R"("panel": [{"id": "w", "primitive": "scene", "bind": {"scene": "live:model"}, "options": {"colormap": "magma"}}])"),
        json::parse(R"({"caliper_spec": 0, "applet": {"id": "a"}, "control": [{"id": "k", "kind": "button", "bind": "run:pause"}]})"),
    };
    for (const json& d : documents) {
        CAPTURE(d.dump());
        if (d.at("caliper_spec") == 1) CHECK_FALSE(checker_v1().passes(d));
        CHECK_FALSE(library_accepts(d));
    }
    // And what both accept.
    for (const json& d : {doc(scene), doc(R"("control": [{"id": "k", "kind": "slider", "bind": "param:lr", "options": {"min": 0.001, "max": 1, "log": true}}])")}) {
        CAPTURE(d.dump());
        const std::vector<std::string> errors = checker_v1().errors(d);
        CHECK_MESSAGE(errors.empty(), join_errors(errors));
        CHECK(library_accepts(d));
    }
}

TEST_CASE("the schema of version 1 takes programs and provenance, and refuses what the library refuses") {
    auto parsed = cs::parse_toml(fixture("programs.spec.toml"));
    REQUIRE(parsed.ok());
    const json canonical = cs::to_json(parsed.spec);
    std::vector<std::string> errors = checker_v1().errors(canonical);
    CHECK_MESSAGE(errors.empty(), join_errors(errors));

    const std::string head = R"("caliper_spec": 1, "applet": {"id": "a"}, "panel": [{"id": "w", "primitive": "scene", "bind": {"scene": "live:model"}}])";
    auto doc = [&](const std::string& rest) { return json::parse("{" + head + ", " + rest + "}"); };
    const std::string geometry = R"({"kind": "geometry", "panel": "w", "max_count": 8, "arrays": ["positions"]})";
    auto program = [&](const std::string& output, const std::string& more = "") {
        return R"("program": [{"id": "p", "wgsl": "x", "inputs": [{"var": "v", "source": "live:v"}], "output": )" + output +
               more + "}]";
    };
    const std::vector<json> refused = {
        doc(program(R"({"kind": "geometry", "panel": "w", "max_count": 0, "arrays": ["positions"]})")),
        doc(program(R"({"kind": "geometry", "panel": "w", "max_count": 8, "arrays": ["normals"]})")),
        doc(program(R"({"kind": "geometry", "panel": "w", "max_count": 8})")),
        doc(program(R"({"kind": "pixels", "width": 4, "height": 4, "panel": "w"})")),
        doc(program(R"({"kind": "pixels", "width": 4096, "height": 4})")),
        doc(program(R"({"kind": "geometry", "panel": "w", "max_count": 8, "arrays": ["positions"], "width": 4})")),
        doc(program(geometry, R"(, "limits": {"every": 0})")),
        doc(program(geometry, R"(, "limits": {"time_ms": 60})")),
        doc(program(geometry, R"(, "workgroups": [1, 2])")),
        doc(program(geometry, R"(, "runs": true)")),
        doc(R"("program": [{"id": "p", "wgsl": "x", "inputs": [{"var": "v", "source": "w"}], "output": )" + geometry + "}]"),
        doc(R"("provenance": {"from": "x"})"),
        doc(R"("provenance": {"foreign": true, "reviewed": [{"target": "program:p", "sha256": "abc"}]})"),
        doc(R"("control": [{"id": "k", "kind": "toggle", "bind": "action:save"}])"),
        doc(R"("control": [{"id": "k", "kind": "toggle", "bind": "intervene:blocks.0", "options": {"min": 0}}])"),
    };
    for (const json& d : refused) {
        CAPTURE(d.dump());
        CHECK_FALSE(checker_v1().passes(d));
        CHECK_FALSE(library_accepts(d));
    }
    const std::vector<json> accepted = {
        doc(program(geometry)),
        doc(program(R"({"kind": "pixels", "width": 4, "height": 4})", R"(, "limits": {"every": 5, "memory_mb": 8})")),
        doc(R"("provenance": {"foreign": true, "from": "a download", "reviewed": [{"target": "program:p", "sha256": ")" +
            std::string(64, 'f') + R"("}]})"),
        doc(R"("control": [{"id": "k", "kind": "toggle", "bind": "intervene:blocks.0.attn#3"}])"),
    };
    for (const json& d : accepted) {
        CAPTURE(d.dump());
        errors = checker_v1().errors(d);
        CHECK_MESSAGE(errors.empty(), join_errors(errors));
        CHECK(library_accepts(d));
    }
}

TEST_CASE("closable and layout_locked validate, round-trip, and stay gated to version 1") {
    const json v1_doc = {
        {"caliper_spec", 1},
        {"applet", {{"id", "locked-instrument"}, {"layout_locked", true}}},
        {"panel", json::array({
                      {{"id", "notes"}, {"primitive", "text"}, {"closable", false}, {"options", {{"text", "Fixed"}}}},
                      {{"id", "scene"}, {"primitive", "scene"}, {"closable", true}, {"bind", {{"scene", "live:model"}}}},
                  })},
        {"control", json::array({
                        {{"id", "gain"}, {"kind", "slider"}, {"closable", false}, {"bind", "value"}},
                    })},
        {"layout", {{"split", "horizontal"}, {"children", json::array({"notes", "scene", "gain"})}}},
    };
    const auto schema_errors = checker_v1().errors(v1_doc);
    CHECK_MESSAGE(schema_errors.empty(), join_errors(schema_errors));
    CHECK(library_accepts(v1_doc));

    const auto parsed = cs::from_json(v1_doc);
    REQUIRE(parsed.ok());
    CHECK(parsed.spec.applet.layout_locked == std::optional<bool>(true));
    REQUIRE(parsed.spec.panels.size() == 2);
    CHECK(parsed.spec.panels[0].closable == std::optional<bool>(false));
    CHECK(parsed.spec.panels[1].closable == std::optional<bool>(true));
    REQUIRE(parsed.spec.controls.size() == 1);
    CHECK(parsed.spec.controls[0].closable == std::optional<bool>(false));
    CHECK(cs::required_version(parsed.spec) == cs::kInstrumentVersion);

    const std::string toml = cs::to_toml(parsed.spec);
    CHECK(toml.find("layout_locked = true") != std::string::npos);
    CHECK(toml.find("closable = false") != std::string::npos);
    const auto reparsed = cs::parse_toml(toml);
    REQUIRE(reparsed.ok());
    CHECK(reparsed.spec == parsed.spec);
    CHECK_MESSAGE(checker_v1().errors(cs::to_json(reparsed.spec)).empty(), "round-tripped JSON fails v1 schema");

    // Patching closable on a version 0 spec raises it to version 1 and round-trips through merge patch.
    const auto v0_base = cs::parse_toml(R"(
caliper_spec = 0
[applet]
id = "base"
[[panel]]
id = "note"
primitive = "text"
options = { text = "hi" }
)");
    REQUIRE(v0_base.ok());
    CHECK(cs::required_version(v0_base.spec) == 0);
    const auto patched = cs::patch_panel(v0_base.spec, "note", json{{"closable", false}});
    REQUIRE(patched.ok());
    CHECK(patched.spec.version == cs::kInstrumentVersion);
    CHECK(patched.spec.panels[0].closable == std::optional<bool>(false));
    const auto cleared = cs::patch_panel(patched.spec, "note", json{{"closable", nullptr}});
    REQUIRE(cleared.ok());
    CHECK_FALSE(cleared.spec.panels[0].closable.has_value());

    // Version 0 rejects closable and layout_locked; non-boolean values fail both v1 schema and library.
    const std::vector<json> refused = {
        json::parse(R"({"caliper_spec": 0, "applet": {"id": "a", "layout_locked": true}})"),
        json::parse(R"({"caliper_spec": 0, "applet": {"id": "a"},
                        "panel": [{"id": "n", "primitive": "text", "closable": false, "options": {"text": "x"}}]})"),
        json::parse(R"({"caliper_spec": 1, "applet": {"id": "a", "layout_locked": "yes"}})"),
        json::parse(R"({"caliper_spec": 1, "applet": {"id": "a"},
                        "panel": [{"id": "n", "primitive": "text", "closable": 0, "options": {"text": "x"}}]})"),
        json::parse(R"({"caliper_spec": 1, "applet": {"id": "a"},
                        "control": [{"id": "c", "kind": "slider", "closable": "no", "bind": "value"}]})"),
    };
    for (const json& d : refused) {
        CAPTURE(d.dump());
        if (d.at("caliper_spec") == 1) CHECK_FALSE(checker_v1().passes(d));
        CHECK_FALSE(library_accepts(d));
    }
}
