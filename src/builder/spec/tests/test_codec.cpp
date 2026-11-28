// Parsing and writing a spec: the fixtures, round trips through canonical TOML
// and JSON, positions, the value rules and hostile input.
#include <ostream>
#include <string>

#include <doctest/doctest.h>

#include <cmath>
#include <limits>

#include "helpers.h"

using namespace spec_test;
using nlohmann::json;

namespace {

const char* kCanonicalExample = R"(caliper_spec = 2

[instrument]
id = "attention-entropy"
title = "Attention entropy"
description = "Entropy of each attention head while the model trains."

[[source]]
id = "attn"
catalog = "live:attn.weights"

[[source]]
id = "entropy"
from = "attn"
  [[source.transform]]
  op = "reduce"
  axis = 3
  fn = "entropy"
  [[source.transform]]
  op = "reduce"
  axis = 2
  fn = "mean"

[[source]]
id = "loss"
sql = "SELECT step, loss FROM run42_metrics ORDER BY step"

[[panel]]
id = "heat"
primitive = "heatmap"
title = "Entropy per layer and head"
bind = { values = "entropy" }
options = { colormap = "viridis", x_label = "Head", y_label = "Layer" }

[[panel]]
id = "loss"
primitive = "series"
bind = { x = "loss.step", y = ["loss.loss"] }

[layout]
split = "horizontal"
ratio = 0.6
children = ["heat", "loss"]
)";

cs::ParseResult parse(const std::string& text) { return cs::parse_toml(text); }

std::string with_instrument(const std::string& body) { return "caliper_spec = 2\n[instrument]\nid = \"a\"\n" + body; }

}  // namespace

TEST_CASE("the contracts' example parses into the expected spec") {
    auto r = parse(fixture("attention-entropy.spec.toml"));
    REQUIRE_MESSAGE(r.ok(), dump(r.diagnostics));
    const cs::Spec& s = r.spec;
    CHECK(s.version == 2);
    CHECK(s.instrument.id == "attention-entropy");
    CHECK(s.instrument.title == std::optional<std::string>("Attention entropy"));
    REQUIRE(s.sources.size() == 3);
    CHECK(s.sources[0].catalog == std::optional<std::string>("live:attn.weights"));
    CHECK(s.sources[1].from == std::optional<std::string>("attn"));
    REQUIRE(s.sources[1].transforms.size() == 2);
    CHECK(s.sources[1].transforms[0].op == cs::TransformOp::reduce);
    CHECK(s.sources[1].transforms[0].axis == std::vector<std::int64_t>{3});
    CHECK_FALSE(s.sources[1].transforms[0].axis_list);
    CHECK(s.sources[1].transforms[0].fn == std::optional<std::string>("entropy"));
    CHECK(s.sources[2].sql.has_value());
    REQUIRE(s.panels.size() == 2);
    CHECK(s.panels[0].primitive == cs::Primitive::heatmap);
    CHECK(s.panels[0].bind.at("values").bindings == std::vector<std::string>{"entropy"});
    CHECK(s.panels[0].options.colormap == std::optional<std::string>("viridis"));
    CHECK(s.panels[1].bind.at("y").list);
    REQUIRE(s.layout.has_value());
    CHECK(s.layout->kind == cs::LayoutNode::Kind::split);
    CHECK(s.layout->split == cs::SplitDirection::horizontal);
    CHECK(s.layout->ratio == std::optional<double>(0.6));
    REQUIRE(s.layout->children.size() == 2);
    CHECK(s.layout->children[1].panel == "loss");
}

TEST_CASE("the canonical TOML of the contracts' example is the example without its comments") {
    auto r = parse(fixture("attention-entropy.spec.toml"));
    REQUIRE(r.ok());
    CHECK(cs::to_toml(r.spec) == kCanonicalExample);
}

TEST_CASE("every valid fixture round trips through canonical TOML") {
    for (const std::string& name : valid_fixtures()) {
        CAPTURE(name);
        auto first = parse(fixture(name));
        REQUIRE_MESSAGE(first.ok(), dump(first.diagnostics));
        const std::string text = cs::to_toml(first.spec);
        auto second = parse(text);
        REQUIRE_MESSAGE(second.ok(), (dump(second.diagnostics) + text));
        CHECK(second.spec == first.spec);
        CHECK(cs::to_toml(second.spec) == text);  // the writer reaches a fixed point at once
    }
}

TEST_CASE("every valid fixture round trips through canonical JSON") {
    for (const std::string& name : valid_fixtures()) {
        CAPTURE(name);
        auto parsed = parse(fixture(name));
        REQUIRE(parsed.ok());
        const json j = cs::to_json(parsed.spec);
        auto from_value = cs::from_json(j);
        REQUIRE_MESSAGE(from_value.ok(), dump(from_value.diagnostics));
        CHECK(from_value.spec == parsed.spec);

        const std::string text = cs::to_json_text(parsed.spec);
        auto from_text = cs::from_json_text(text);
        REQUIRE_MESSAGE(from_text.ok(), dump(from_text.diagnostics));
        CHECK(from_text.spec == parsed.spec);
        CHECK(cs::to_json_text(from_text.spec) == text);
        CHECK(json::parse(text) == j);
    }
}

TEST_CASE("the canonical JSON text is compact, with keys in byte order") {
    auto r = parse(fixture("minimal.spec.toml"));
    REQUIRE(r.ok());
    CHECK(cs::to_json_text(r.spec) == R"({"caliper_spec":2,"instrument":{"id":"empty"}})");

    auto e = parse(fixture("attention-entropy.spec.toml"));
    REQUIRE(e.ok());
    const std::string text = cs::to_json_text(e.spec);
    CHECK(text.find(R"("layout":{"children":["heat","loss"],"ratio":0.6,"split":"horizontal"})") !=
          std::string::npos);
    CHECK(text.find(R"("bind":{"x":"loss.step","y":["loss.loss"]})") != std::string::npos);
}

TEST_CASE("strings with line breaks, quotes and control characters survive both writers") {
    cs::Spec s;
    s.instrument.id = "texts";
    s.instrument.description = std::string("first line\nsecond \"quoted\" line\twith a tab\\ and \x01 a control\r\n");
    s.instrument.title = "Unicode \xC3\xA9t\xC3\xA9 \xE2\x9C\x93";
    cs::Panel p;
    p.id = "note";
    p.primitive = cs::Primitive::text;
    p.options.text = std::string("\"\"\"three quotes\"\"\" and a trailing quote\"");
    s.panels.push_back(p);
    const std::string toml = cs::to_toml(s);
    auto back = parse(toml);
    REQUIRE_MESSAGE(back.ok(), (dump(back.diagnostics) + toml));
    CHECK(back.spec == s);
    auto json_back = cs::from_json_text(cs::to_json_text(s));
    REQUIRE(json_back.ok());
    CHECK(json_back.spec == s);
}

TEST_CASE("an extension table is kept as written, and an empty one is dropped") {
    auto r = parse(fixture("every-primitive.spec.toml"));
    REQUIRE(r.ok());
    CHECK(r.spec.x == json::parse(R"({"editor": {"zoom": 1.5, "open": ["a", "b"]}})"));
    CHECK(r.spec.panels[4].x == json::parse(R"({"pinned": true})"));
    CHECK(r.spec.layout->x == json::parse(R"({"active": 0})"));

    auto e = parse(with_instrument("x = {}\n"));
    REQUIRE(e.ok());
    CHECK(e.spec.instrument.x.is_null());
}

TEST_CASE("an integer key takes a number with no fractional part, as a browser cannot tell them apart") {
    auto r = parse(with_instrument(R"(
[[source]]
id = "t"
catalog = "tensor:t"
  [[source.transform]]
  op = "slice"
  axis = 1.0
  index = -2.0
)"));
    REQUIRE_MESSAGE(r.ok(), dump(r.diagnostics));
    CHECK(r.spec.sources[0].transforms[0].axis == std::vector<std::int64_t>{1});
    CHECK(r.spec.sources[0].transforms[0].index == std::optional<std::int64_t>(-2));

    auto bad = parse(with_instrument(R"(
[[source]]
id = "t"
catalog = "tensor:t"
  [[source.transform]]
  op = "slice"
  axis = 1.5
)"));
    CHECK(has(bad.diagnostics, "spec.wrong_type", "source[0].transform[0].axis"));
}

TEST_CASE("parse errors carry the line and column of the value") {
    auto r = parse("caliper_spec = 2\n[instrument]\nid = \"a\"\n\n[[panel]]\nid = \"p\"\nprimitive = \"heatmap\"\n"
                   "options = { colourmap = \"viridis\" }\n");
    REQUIRE_FALSE(r.ok());
    const cs::Diagnostic* d = find(r.diagnostics, "spec.unknown_option");
    REQUIRE(d);
    CHECK(d->path == "panel[0].options.colourmap");
    CHECK(d->line == 8);
    CHECK(d->column > 1);
    CHECK(d->to_error().where == "panel[0].options.colourmap");

    // A missing key points at the table that lacks it.
    auto m = parse("caliper_spec = 2\n[instrument]\nid = \"a\"\n\n[[panel]]\nprimitive = \"text\"\n");
    const cs::Diagnostic* md = find(m.diagnostics, "spec.missing_key");
    REQUIRE(md);
    CHECK(md->path == "panel[0].id");
    CHECK(md->line == 5);
}

TEST_CASE("syntax errors carry a line and a column") {
    auto t = parse("caliper_spec = 2\n[instrument]\nid = \"a\"\ntitle = = 3\n");
    REQUIRE(t.diagnostics.size() == 1);
    CHECK(t.diagnostics[0].code == "spec.syntax");
    CHECK(t.diagnostics[0].line == 4);
    CHECK(t.diagnostics[0].to_error().where == "4:" + std::to_string(t.diagnostics[0].column));

    auto j = cs::from_json_text("{\n  \"caliper_spec\": 2,\n  \"instrument\": {\"id\": \"a\",}\n}");
    REQUIRE(j.diagnostics.size() == 1);
    CHECK(j.diagnostics[0].code == "spec.syntax");
    CHECK(j.diagnostics[0].line == 3);

    auto dup = cs::from_json_text(R"({"caliper_spec": 2, "instrument": {"id": "a", "id": "b"}})");
    REQUIRE(dup.diagnostics.size() == 1);
    CHECK(dup.diagnostics[0].code == "spec.syntax");
    CHECK(dup.diagnostics[0].message.find("\"id\"") != std::string::npos);
}

TEST_CASE("values that the JSON form cannot carry are refused") {
    SUBCASE("a date") {
        auto r = parse(with_instrument("x = { when = 2026-11-15T12:00:00Z }\n"));
        CHECK(has(r.diagnostics, "spec.wrong_type", "instrument.x.when"));
        CHECK(r.diagnostics[0].line == 4);
    }
    SUBCASE("a local date and a time") {
        CHECK(has(parse(with_instrument("title = 2026-11-15\n")).diagnostics, "spec.wrong_type", "instrument.title"));
        CHECK(has(parse(with_instrument("title = 12:00:00\n")).diagnostics, "spec.wrong_type", "instrument.title"));
    }
    SUBCASE("infinity and NaN") {
        CHECK(has(parse(with_instrument("x = { a = inf }\n")).diagnostics, "spec.out_of_range", "instrument.x.a"));
        CHECK(has(parse(with_instrument("x = { a = nan }\n")).diagnostics, "spec.out_of_range", "instrument.x.a"));
    }
    SUBCASE("an integer that JavaScript cannot read exactly") {
        CHECK(has(parse(with_instrument("x = { a = 9007199254740992 }\n")).diagnostics, "spec.out_of_range"));
        CHECK(parse(with_instrument("x = { a = 9007199254740991 }\n")).ok());
        CHECK(has(cs::from_json(json::parse(R"({"caliper_spec":2,"instrument":{"id":"a","x":{"n":18446744073709551615}}})"))
                      .diagnostics,
                  "spec.out_of_range"));
    }
    SUBCASE("null") {
        auto r = cs::from_json(json::parse(R"({"caliper_spec":2,"instrument":{"id":"a","title":null}})"));
        CHECK(has(r.diagnostics, "spec.wrong_type", "instrument.title"));
    }
    SUBCASE("text that is not UTF-8") {
        json doc = {{"caliper_spec", 2}, {"instrument", {{"id", "a"}, {"title", std::string("bad \xFF byte")}}}};
        CHECK(has(cs::from_json(doc).diagnostics, "spec.bad_value", "instrument.title"));
    }
}

TEST_CASE("strict parsing refuses unknown keys outside x") {
    auto r = parse(with_instrument("titel = \"typo\"\n[[panel]]\nid = \"p\"\nprimitive = \"text\"\noptions = { text = \"t\" }\n"
                                   "colour = \"red\"\n"));
    CHECK(has(r.diagnostics, "spec.unknown_key", "instrument.titel"));
    CHECK(has(r.diagnostics, "spec.unknown_key", "panel[0].colour"));
    CHECK(r.spec == cs::Spec{});  // a failed parse returns an empty spec

    auto legacy = parse("caliper_spec = 2\n[applet]\nid = \"a\"\n");
    CHECK(has(legacy.diagnostics, "spec.unknown_key", "applet"));
    CHECK(has(legacy.diagnostics, "spec.missing_key", "instrument"));
}

TEST_CASE("the item decoders that the command layer uses") {
    auto p = cs::panel_from_json(json::parse(R"({"id": "h", "primitive": "histogram", "bind": {"values": "a"},
                                                  "options": {"bins": 32}})"),
                                 "panel[4]");
    REQUIRE(p.ok());
    CHECK(p.value.options.bins == std::optional<std::int64_t>(32));
    CHECK(cs::to_json(p.value) == json::parse(R"({"id":"h","primitive":"histogram","bind":{"values":"a"},
                                                  "options":{"bins":32}})"));

    auto bad = cs::panel_from_json(json::parse(R"({"id": "h", "primitive": "histogram", "bind": {"y": "a"}})"),
                                   "panel[4]");
    CHECK(has(bad.diagnostics, "spec.unknown_slot", "panel[4].bind.y"));

    auto s = cs::source_from_json(json::parse(R"({"id": "s", "sql": "SELECT 1", "x": {"k": 1}})"), "source[0]");
    REQUIRE(s.ok());
    CHECK(s.value.sql == std::optional<std::string>("SELECT 1"));

    auto l = cs::layout_from_json(json::parse(R"({"tabs": ["a", {"split": "vertical", "children": ["b", "c"]}]})"));
    REQUIRE(l.ok());
    CHECK(l.value.kind == cs::LayoutNode::Kind::tabs);
    CHECK(cs::to_json(l.value) == json::parse(R"({"tabs":["a",{"split":"vertical","children":["b","c"]}]})"));

    auto both = cs::layout_from_json(json::parse(R"({"tabs": ["a"], "split": "vertical"})"));
    CHECK(has(both.diagnostics, "spec.bad_layout", "layout"));
}

TEST_CASE("hostile input is refused within bounded work") {
    SUBCASE("a text over the size limit") {
        std::string big = with_instrument("title = \"") + std::string(1024 * 1024, 'a') + "\"\n";
        auto r = parse(big);
        REQUIRE(r.diagnostics.size() == 1);
        CHECK(r.diagnostics[0].code == "spec.too_large");
        CHECK(cs::from_json_text(std::string(cs::limits::kTextBytes + 1, ' ')).diagnostics[0].code ==
              "spec.too_large");
    }
    SUBCASE("a 1 MiB string in a JSON value") {
        json doc = {{"caliper_spec", 2}, {"instrument", {{"id", "a"}, {"title", std::string(1024 * 1024, 'a')}}}};
        auto r = cs::from_json(doc);
        CHECK(has(r.diagnostics, "spec.string_too_long", "instrument.title"));
        // The same string in a Spec built in code is caught by validation.
        cs::Spec s;
        s.instrument.id = "a";
        s.instrument.description = std::string(1024 * 1024, 'd');
        CHECK(has(cs::validate(s).diagnostics, "spec.string_too_long", "instrument.description"));
    }
    SUBCASE("deeply nested inline arrays") {
        std::string open(40, '['), close(40, ']');
        auto r = parse(with_instrument("x = { a = " + open + "1" + close + " }\n"));
        CHECK(has(r.diagnostics, "spec.too_deep"));
        std::string open2(400, '['), close2(400, ']');
        auto r2 = parse(with_instrument("x = { a = " + open2 + "1" + close2 + " }\n"));
        CHECK(r2.diagnostics.size() == 1);
        CHECK((r2.diagnostics[0].code == "spec.syntax" || r2.diagnostics[0].code == "spec.too_deep"));
    }
    SUBCASE("a dotted key of 100,000 parts") {
        std::string key = "a";
        for (int i = 0; i < 100000; ++i) key += ".a";
        auto r = parse("caliper_spec = 2\n[x." + key + "]\n");
        REQUIRE(r.diagnostics.size() == 1);
        CHECK(r.diagnostics[0].code == "spec.too_deep");
        CHECK(r.diagnostics[0].line == 2);
        // Quoted parts count too.
        std::string quoted = "\"a\"";
        for (int i = 0; i < 100; ++i) quoted += ".\"a\"";
        CHECK(parse("caliper_spec = 2\nx." + quoted + " = 1\n").diagnostics[0].code == "spec.too_deep");
        // Floats and dates do not count as dotted keys.
        CHECK(parse(with_instrument("x = { a = [1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, "
                                    "7.5, 8.5, 9.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5, 1.5, 2.5, 3.5, 4.5, 5.5, "
                                    "6.5, 7.5] }\n"))
                  .ok());
    }
    SUBCASE("deep nesting in JSON") {
        std::string open(100000, '['), close(100000, ']');
        auto r = cs::from_json_text("{\"caliper_spec\":2,\"instrument\":{\"id\":\"a\",\"x\":{\"a\":" + open + close +
                                    "}}}");
        REQUIRE(r.diagnostics.size() == 1);
        CHECK(r.diagnostics[0].code == "spec.too_deep");
        json deep = json::array();
        for (int i = 0; i < 40; ++i) deep = json::array({deep});
        json doc = {{"caliper_spec", 2}, {"instrument", {{"id", "a"}, {"x", {{"a", deep}}}}}};
        CHECK(has(cs::from_json(doc).diagnostics, "spec.too_deep"));
    }
    SUBCASE("a huge count of values") {
        std::string many = "x = { a = [";
        for (int i = 0; i < 70000; ++i) many += "1,";
        many += "] }\n";
        auto r = parse(with_instrument(many));
        REQUIRE(r.diagnostics.size() == 1);
        CHECK(r.diagnostics[0].code == "spec.too_many");
    }
    SUBCASE("many errors produce a bounded report") {
        std::string many;
        for (int i = 0; i < 2000; ++i) many += "k" + std::to_string(i) + " = 1\n";
        auto r = parse("caliper_spec = 2\n" + many + "[instrument]\nid = \"a\"\n");
        CHECK(r.diagnostics.size() == cs::limits::kDiagnostics);
    }
}

TEST_CASE("a retired, later or invalid version is refused with one error") {
    auto later = parse("caliper_spec = 3\n[instrument]\nid = \"a\"\nfuture_key = 1\n[[gadget]]\n");
    REQUIRE(later.diagnostics.size() == 1);
    CHECK(later.diagnostics[0].code == "spec.unsupported_version");
    CHECK(later.diagnostics[0].message.find("newer Caliper") != std::string::npos);
    CHECK(later.diagnostics[0].line == 1);

    for (int old_version : {0, 1}) {
        CAPTURE(old_version);
        auto old = parse("caliper_spec = " + std::to_string(old_version) + "\n[instrument]\nid = \"a\"\n");
        REQUIRE(old.diagnostics.size() == 1);
        CHECK(old.diagnostics[0].code == "spec.unsupported_version");
        CHECK(old.diagnostics[0].message.find("newer Caliper") == std::string::npos);
    }

    CHECK(has(parse("caliper_spec = \"2\"\n[instrument]\nid = \"a\"\n").diagnostics, "spec.wrong_type", "caliper_spec"));
    CHECK(has(parse("[instrument]\nid = \"a\"\n").diagnostics, "spec.missing_key", "caliper_spec"));
}

TEST_CASE("a spec built in code round trips through both forms") {
    cs::Spec s;
    s.instrument.id = "built";
    s.instrument.description = std::string("ends with three quotes \"\"\" and a backslash \\");
    s.x = json::parse(R"({"k": [1, 2.5, "s", true, {"n": {}}], "weird key": {"é": -0.0}})");
    cs::Source src;
    src.id = "t";
    src.catalog = "tensor:t";
    cs::Transform slice;
    slice.op = cs::TransformOp::slice;
    slice.axis = {-1};
    slice.index = 3;
    cs::Transform reduce;
    reduce.op = cs::TransformOp::reduce;
    reduce.axis = {2};
    reduce.axis_list = true;  // written as [2]
    reduce.fn = "var";
    cs::Transform norm;
    norm.op = cs::TransformOp::normalise;
    norm.mode = "minmax";
    src.transforms = {slice, reduce, norm};
    s.sources.push_back(src);
    cs::Source q;
    q.id = "q";
    q.sql = "SELECT \"step\", 'v'\nFROM m\n";
    s.sources.push_back(q);
    cs::Panel p;
    p.id = "p";
    p.primitive = cs::Primitive::series;
    p.bind["y"] = cs::SlotValue{{"t"}, false};
    p.options.x_range = std::array<double, 2>{-1e300, 2.5e-7};
    p.options.x_log = true;
    p.x = json::parse(R"({"a.b": 1})");
    s.panels.push_back(p);
    s.layout = cs::LayoutNode::of_panel("p");

    const std::string toml = cs::to_toml(s);
    CHECK(toml.rfind("caliper_spec = 2\nlayout = \"p\"\nx = ", 0) == 0);
    CHECK(toml.find("axis = [2]") != std::string::npos);
    auto back = parse(toml);
    REQUIRE_MESSAGE(back.ok(), (dump(back.diagnostics) + toml));
    CHECK(back.spec == s);
    CHECK(cs::to_toml(back.spec) == toml);
    auto from_json = cs::from_json(cs::to_json(s));
    REQUIRE(from_json.ok());
    CHECK(from_json.spec == s);
    CHECK(cs::validate(s).ok());
}

// ---- scene panels and controls (design/instruments.md) ----------------------

TEST_CASE("a version 2 spec with a scene panel and controls parses into the expected spec") {
    auto r = parse(fixture("instruments.spec.toml"));
    REQUIRE_MESSAGE(r.ok(), dump(r.diagnostics));
    const cs::Spec& s = r.spec;
    CHECK(s.version == 2);
    REQUIRE(s.panels.size() == 1);
    const cs::Panel& world = s.panels[0];
    CHECK(world.primitive == cs::Primitive::scene);
    CHECK(world.bind.at("scene").bindings == std::vector<std::string>{"live:model"});
    CHECK(cs::live_scene_name(world.bind.at("scene").bindings[0]) == std::optional<std::string>("model"));
    CHECK(world.options.scene.up == std::optional<std::string>("+z"));
    CHECK(world.options.scene.axes == std::optional<bool>(true));
    CHECK_FALSE(world.options.scene.labels.has_value());
    CHECK(world.options.scene.view_camera.at("target") == json::array({0, 1, 0}));
    CHECK(world.options.set_names() == std::vector<std::string>{"axes", "up", "view_camera"});
    REQUIRE(s.controls.size() == 2);
    CHECK(s.controls[0].id == "lr");
    CHECK(s.controls[0].kind == cs::ControlKind::slider);
    CHECK(s.controls[0].title == std::optional<std::string>("Learning rate"));
    CHECK(s.controls[0].bind == "param:lr");
    CHECK(s.controls[0].options.log == std::optional<bool>(true));
    CHECK(s.controls[0].options.min == std::optional<double>(0.00001));
    CHECK(s.controls[1].kind == cs::ControlKind::button);
    CHECK(s.controls[1].bind == "run:pause");
    CHECK(s.controls[1].options.empty());
    CHECK(cs::required_version(s) == 2);
}

TEST_CASE("a version 2 spec with controls round trips through canonical TOML and canonical JSON") {
    auto first = parse(fixture("instruments.spec.toml"));
    REQUIRE(first.ok());
    const std::string toml = cs::to_toml(first.spec);
    CHECK(toml.find("caliper_spec = 2\n") == 0);
    CHECK(toml.find("\n[[control]]\nid = \"lr\"\n") != std::string::npos);
    CHECK(toml.find("[[control]]") < toml.find("[layout]"));
    auto second = parse(toml);
    REQUIRE_MESSAGE(second.ok(), dump(second.diagnostics));
    CHECK(second.spec == first.spec);
    CHECK(cs::to_toml(second.spec) == toml);

    const std::string text = cs::to_json_text(first.spec);
    auto back = cs::from_json_text(text);
    REQUIRE_MESSAGE(back.ok(), dump(back.diagnostics));
    CHECK(back.spec == first.spec);
    CHECK(cs::to_json_text(back.spec) == text);
    const json j = json::parse(text);
    CHECK(j.at("control").size() == 2);
    CHECK(j.at("control")[1] == json::parse(R"({"id": "pause", "kind": "button", "bind": "run:pause"})"));
}

TEST_CASE("the fixtures read and write as version 2") {
    for (const std::string& name : valid_fixtures()) {
        CAPTURE(name);
        auto r = parse(fixture(name));
        REQUIRE(r.ok());
        CHECK(r.spec.version == 2);
        CHECK(r.spec.controls.empty());
        CHECK(cs::required_version(r.spec) == 2);
        CHECK(cs::to_toml(r.spec).find("caliper_spec = 2\n") == 0);
    }
}

TEST_CASE("the parser checks the types and keys of controls and scene options") {
    const std::string head = "caliper_spec = 2\n[instrument]\nid = \"a\"\n";
    auto control = [&](const std::string& body) { return parse(head + "[[control]]\n" + body).diagnostics; };
    CHECK(has(control("id = \"k\"\nkind = \"knob\"\nbind = \"value\"\n"), "spec.unknown_control_kind",
              "control[0].kind"));
    CHECK(has(control("id = \"k\"\nbind = \"value\"\n"), "spec.missing_key", "control[0].kind"));
    CHECK(has(control("id = \"k\"\nkind = \"slider\"\n"), "spec.missing_key", "control[0].bind"));
    CHECK(has(control("kind = \"slider\"\nbind = \"value\"\n"), "spec.missing_key", "control[0].id"));
    CHECK(has(control("id = \"k\"\nkind = \"slider\"\nbind = \"value\"\nlabel = \"x\"\n"), "spec.unknown_key",
              "control[0].label"));
    CHECK(has(control("id = \"k\"\nkind = \"slider\"\nbind = \"value\"\noptions = { colormap = \"magma\" }\n"),
              "spec.unknown_option", "control[0].options.colormap"));
    CHECK(has(control("id = \"k\"\nkind = \"slider\"\nbind = \"value\"\noptions = { min = \"0\" }\n"),
              "spec.wrong_type", "control[0].options.min"));
    CHECK(has(parse("caliper_spec = 2\ncontrol = 3\n[instrument]\nid = \"a\"\n").diagnostics, "spec.wrong_type", "control"));

    auto scene = [&](const std::string& options) {
        return parse(head + "[[panel]]\nid = \"w\"\nprimitive = \"scene\"\nbind = { scene = \"live:model\" }\n"
                            "options = { " +
                     options + " }\n")
            .diagnostics;
    };
    CHECK(has(scene("up = \"+w\""), "spec.bad_value", "panel[0].options.up"));
    CHECK(has(scene("include = []"), "spec.bad_value", "panel[0].options.include"));
    CHECK(has(scene("exclude = [1]"), "spec.wrong_type", "panel[0].options.exclude[0]"));
    CHECK(has(scene("view_camera = 3"), "spec.wrong_type", "panel[0].options.view_camera"));
    CHECK(has(scene("view_camera = { target = [1, 2] }"), "spec.wrong_type", "panel[0].options.view_camera.target"));
    CHECK(has(scene("view_camera = { zoom = 2 }"), "spec.unknown_key", "panel[0].options.view_camera.zoom"));
    CHECK(has(scene("colormap = \"magma\""), "spec.unknown_option", "panel[0].options.colormap"));
    CHECK(has(scene("labels = 1"), "spec.wrong_type", "panel[0].options.labels"));
    CHECK(parse(head + "[[panel]]\nid = \"w\"\nprimitive = \"scene\"\nbind = { scene = \"live:model\" }\n"
                       "options = { view_camera = \"cameras/main\", include = [\"modules/**\"], labels = false, "
                       "scale_bar = true }\n")
              .ok());
}

TEST_CASE("the item decoders take controls and scene panels") {
    auto c = cs::control_from_json(json::parse(R"({"id": "lr", "kind": "slider", "bind": "param:lr",
                                                   "options": {"min": 0.001, "max": 1, "log": true}})"),
                                   "control[0]");
    REQUIRE_MESSAGE(c.ok(), dump(c.diagnostics));
    CHECK(c.value.options.max == std::optional<double>(1));
    CHECK(cs::to_json(c.value) == json::parse(R"({"id": "lr", "kind": "slider", "bind": "param:lr",
                                                  "options": {"min": 0.001, "max": 1.0, "log": true}})"));
    auto bad = cs::control_from_json(json::parse(R"({"id": "k", "kind": "dial", "bind": "value"})"), "control[3]");
    CHECK(has(bad.diagnostics, "spec.unknown_control_kind", "control[3].kind"));
    // A value built in code can hold infinity or NaN, which JSON text cannot; check_values refuses both.
    json huge = json::parse(R"({"id": "k", "kind": "slider", "bind": "value", "options": {}})");
    huge["options"]["max"] = std::numeric_limits<double>::infinity();
    CHECK(has(cs::control_from_json(huge, "control[0]").diagnostics, "spec.out_of_range", "control[0].options.max"));
    json nan_min = json::parse(R"({"id": "k", "kind": "slider", "bind": "value", "options": {}})");
    nan_min["options"]["min"] = std::numeric_limits<double>::quiet_NaN();
    CHECK(has(cs::control_from_json(nan_min, "control[0]").diagnostics, "spec.out_of_range", "control[0].options.min"));

    auto p = cs::panel_from_json(json::parse(R"({"id": "w", "primitive": "scene", "bind": {"scene": "live:model"},
                                                 "options": {"axes": false}})"));
    REQUIRE_MESSAGE(p.ok(), dump(p.diagnostics));
    CHECK(p.value.primitive == cs::Primitive::scene);
    CHECK(cs::to_json(p.value).at("options") == json::parse(R"({"axes": false})"));
}

TEST_CASE("the bind grammar of a control") {
    using K = cs::ControlTarget::Kind;
    CHECK(cs::parse_control_target("param:lr") == cs::ControlTarget{K::param, "lr"});
    CHECK(cs::parse_control_target("action:save checkpoint") == cs::ControlTarget{K::action, "save checkpoint"});
    CHECK(cs::parse_control_target("run:pause") == cs::ControlTarget{K::run_pause, ""});
    CHECK(cs::parse_control_target("value") == cs::ControlTarget{K::value, ""});
    CHECK(cs::parse_control_target("param:" + std::string(63, 'a')).has_value());
    for (const std::string& bad : std::vector<std::string>{"param:", "param:" + std::string(64, 'a'), "param:a\nb",
                                                           "action:", "run:step", "intervene:", "values", "",
                                                           "lr"}) {
        CAPTURE(bad);
        CHECK_FALSE(cs::parse_control_target(bad).has_value());
    }
    CHECK(cs::parse_control_target("intervene:layer.0") == cs::ControlTarget{K::intervene, "layer.0"});
    for (const std::string& text :
         std::vector<std::string>{"param:lr", "action:save", "run:pause", "value", "intervene:layer.0#2"})
        CHECK(cs::control_target_string(*cs::parse_control_target(text)) == text);
    CHECK(cs::control_takes_target(cs::ControlKind::slider, {K::param, "lr"}));
    CHECK(cs::control_takes_target(cs::ControlKind::slider, {K::value, ""}));
    CHECK_FALSE(cs::control_takes_target(cs::ControlKind::slider, {K::action, "save"}));
    CHECK(cs::control_takes_target(cs::ControlKind::button, {K::run_pause, ""}));
    CHECK_FALSE(cs::control_takes_target(cs::ControlKind::button, {K::param, "lr"}));
    CHECK(cs::control_takes_target(cs::ControlKind::toggle, {K::param, "lr"}));
    CHECK(cs::control_takes_target(cs::ControlKind::toggle, {K::intervene, "layer.0"}));
    CHECK(cs::control_takes_target(cs::ControlKind::toggle, {K::value, ""}));
    CHECK_FALSE(cs::control_takes_target(cs::ControlKind::range, {K::param, "lr"}));
    CHECK(cs::live_scene_name("live:model") == std::optional<std::string>("model"));
    CHECK_FALSE(cs::live_scene_name("live:").has_value());
    CHECK_FALSE(cs::live_scene_name("model").has_value());
    CHECK_FALSE(cs::live_scene_name("tensor:model").has_value());
}
