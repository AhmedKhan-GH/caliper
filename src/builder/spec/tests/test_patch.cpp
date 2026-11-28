// JSON merge patches of one panel or source: RFC 7386 itself, what is applied,
// what is refused and the revalidation.
#include <ostream>
#include <string>

#include <doctest/doctest.h>

#include <caliper_builder/spec/patch.h>

#include "helpers.h"

using namespace spec_test;
using nlohmann::json;

namespace {

cs::Spec example() {
    auto r = cs::parse_toml(fixture("attention-entropy.spec.toml"));
    REQUIRE(r.ok());
    return r.spec;
}

}  // namespace

TEST_CASE("merge_patch follows the examples of RFC 7386") {
    struct Case {
        const char* target;
        const char* patch;
        const char* result;
    };
    const Case cases[] = {
        {R"({"a":"b"})", R"({"a":"c"})", R"({"a":"c"})"},
        {R"({"a":"b"})", R"({"b":"c"})", R"({"a":"b","b":"c"})"},
        {R"({"a":"b"})", R"({"a":null})", R"({})"},
        {R"({"a":"b","b":"c"})", R"({"a":null})", R"({"b":"c"})"},
        {R"({"a":["b"]})", R"({"a":"c"})", R"({"a":"c"})"},
        {R"({"a":"c"})", R"({"a":["b"]})", R"({"a":["b"]})"},
        {R"({"a":{"b":"c"}})", R"({"a":{"b":"d","c":null}})", R"({"a":{"b":"d"}})"},
        {R"({"a":[{"b":"c"}]})", R"({"a":[1]})", R"({"a":[1]})"},
        {R"(["a","b"])", R"(["c","d"])", R"(["c","d"])"},
        {R"({"a":"b"})", R"(["c"])", R"(["c"])"},
        {R"({"a":"foo"})", R"(null)", R"(null)"},
        {R"({"a":"foo"})", R"("bar")", R"("bar")"},
        {R"({"e":null})", R"({"a":1})", R"({"e":null,"a":1})"},
        {R"([1,2])", R"({"a":"b","c":null})", R"({"a":"b"})"},
        {R"({})", R"({"a":{"bb":{"ccc":null}}})", R"({"a":{"bb":{}}})"},
    };
    for (const Case& c : cases) {
        CAPTURE(c.patch);
        CHECK(cs::merge_patch(json::parse(c.target), json::parse(c.patch)) == json::parse(c.result));
    }
}

TEST_CASE("a patch of a panel merges into it and revalidates the spec") {
    const cs::Spec before = example();
    auto r = cs::patch_panel(before, "heat", json::parse(R"({"title": null, "options": {"colormap": "magma",
                                                              "range": [0, 4]}})"),
                             fixture_resolver());
    REQUIRE_MESSAGE(r.ok(), dump(r.diagnostics));
    const cs::Panel& p = r.spec.panels[0];
    CHECK_FALSE(p.title.has_value());
    CHECK(p.options.colormap == std::optional<std::string>("magma"));
    CHECK(p.options.x_label == std::optional<std::string>("Head"));  // untouched keys stay
    CHECK(p.options.range.has_value());
    CHECK(r.spec.panels[1] == before.panels[1]);
    CHECK(r.spec.sources == before.sources);

    // A list in the patch replaces the list.
    auto y = cs::patch_panel(before, "loss", json::parse(R"({"bind": {"y": "loss.loss"}})"));
    REQUIRE(y.ok());
    CHECK_FALSE(y.spec.panels[1].bind.at("y").list);
    CHECK(y.spec.panels[1].bind.at("x").bindings == std::vector<std::string>{"loss.step"});

    // Extensions merge too, and an emptied x disappears.
    auto x1 = cs::patch_panel(before, "heat", json::parse(R"({"x": {"a": 1}})"));
    REQUIRE(x1.ok());
    CHECK(x1.spec.panels[0].x == json::parse(R"({"a": 1})"));
    auto x2 = cs::patch_panel(x1.spec, "heat", json::parse(R"({"x": {"a": null}})"));
    REQUIRE(x2.ok());
    CHECK(x2.spec.panels[0].x.is_null());
}

TEST_CASE("a patch that leaves errors elsewhere is applied and reports them") {
    auto r = cs::patch_panel(example(), "heat", json::parse(R"({"bind": {"values": "missing"}})"),
                             fixture_resolver());
    CHECK(r.applied);
    CHECK_FALSE(r.ok());
    CHECK(has(r.diagnostics, "spec.unknown_source", "panel[0].bind.values"));
    CHECK(r.spec.panels[0].bind.at("values").bindings == std::vector<std::string>{"missing"});

    // A shape that no longer fits is found through the resolver.
    auto s = cs::patch_panel(example(), "heat", json::parse(R"({"bind": {"values": "attn"}})"), fixture_resolver());
    CHECK(s.applied);
    CHECK(has(s.diagnostics, "spec.slot_mismatch", "panel[0].bind.values"));
}

TEST_CASE("a patch that does not decode is refused and leaves the spec unchanged") {
    const cs::Spec before = example();
    auto unknown = cs::patch_panel(before, "heat", json::parse(R"({"colour": "red"})"));
    CHECK_FALSE(unknown.applied);
    CHECK(unknown.spec == before);
    CHECK(has(unknown.diagnostics, "spec.unknown_key", "panel[0].colour"));

    // Changing the primitive keeps options the new primitive does not take.
    auto prim = cs::patch_panel(before, "heat", json::parse(R"({"primitive": "table"})"));
    CHECK_FALSE(prim.applied);
    CHECK(has(prim.diagnostics, "spec.unknown_slot", "panel[0].bind.values"));
    CHECK(has(prim.diagnostics, "spec.unknown_option", "panel[0].options.colormap"));

    auto null_in_list = cs::patch_panel(before, "loss", json::parse(R"({"bind": {"y": ["loss.loss", null]}})"));
    CHECK_FALSE(null_in_list.applied);
    CHECK(has(null_in_list.diagnostics, "spec.wrong_type", "panel[1].bind.y[1]"));
}

TEST_CASE("a patch cannot change an id, and must be a bounded object") {
    const cs::Spec before = example();
    CHECK(has(cs::patch_panel(before, "heat", json{{"id", "other"}}).diagnostics, "spec.bad_patch", "panel[0].id"));
    CHECK(has(cs::patch_panel(before, "heat", json{{"id", nullptr}}).diagnostics, "spec.bad_patch", "panel[0].id"));
    CHECK(has(cs::patch_source(before, "attn", json{{"id", 3}}).diagnostics, "spec.bad_patch", "source[0].id"));
    CHECK(has(cs::patch_panel(before, "heat", json::array({1})).diagnostics, "spec.bad_patch", "panel[0]"));
    CHECK(has(cs::patch_panel(before, "heat", json("text")).diagnostics, "spec.bad_patch"));
    json deep = json::object();
    for (int i = 0; i < 40; ++i) deep = json{{"a", deep}};
    auto r = cs::patch_panel(before, "heat", json{{"x", deep}});
    CHECK_FALSE(r.applied);
    CHECK(has(r.diagnostics, "spec.bad_patch"));
    CHECK(has(cs::patch_panel(before, "nope", json::object()).diagnostics, "spec.not_found"));
    CHECK(has(cs::patch_source(before, "nope", json::object()).diagnostics, "spec.not_found"));
}

TEST_CASE("a patch of a source revalidates what derives from it") {
    const cs::Spec before = example();
    auto ok = cs::patch_source(before, "attn", json{{"catalog", "tensor:attn_weights"}}, fixture_resolver());
    REQUIRE_MESSAGE(ok.ok(), dump(ok.diagnostics));
    CHECK(ok.spec.sources[0].catalog == std::optional<std::string>("tensor:attn_weights"));

    // Turning the origin into SQL makes the transforms of the derived source invalid.
    auto sql = cs::patch_source(before, "attn", json{{"catalog", nullptr}, {"sql", "SELECT 1 AS v"}},
                                fixture_resolver());
    CHECK(sql.applied);
    CHECK(has(sql.diagnostics, "spec.bad_transform", "source[1].transform[0]"));

    // Two forms at once is a source_form error, found by validation.
    auto two = cs::patch_source(before, "attn", json{{"sql", "SELECT 1"}});
    CHECK(two.applied);
    CHECK(has(two.diagnostics, "spec.source_form", "source[0]"));

    // A patch of the transform list replaces the whole list.
    auto list = cs::patch_source(before, "entropy",
                                 json::parse(R"({"transform": [{"op": "reduce", "axis": [2, 3], "fn": "mean"}]})"),
                                 fixture_resolver());
    REQUIRE(list.ok());
    REQUIRE(list.spec.sources[1].transforms.size() == 1);
    CHECK(list.spec.sources[1].transforms[0].axis_list);
}

TEST_CASE("a control takes a merge patch like a panel") {
    auto parsed = cs::parse_toml(fixture("instruments.spec.toml"));
    REQUIRE(parsed.ok());
    auto r = cs::patch_control(parsed.spec, "lr", json::parse(R"({"title": "Rate", "options": {"log": null}})"));
    REQUIRE_MESSAGE(r.ok(), dump(r.diagnostics));
    CHECK(r.spec.controls[0].title == std::optional<std::string>("Rate"));
    CHECK_FALSE(r.spec.controls[0].options.log.has_value());
    CHECK(r.spec.controls[0].options.min.has_value());
    CHECK(has(cs::patch_control(parsed.spec, "nope", json::object()).diagnostics, "spec.not_found"));
    CHECK(has(cs::patch_control(parsed.spec, "lr", json::parse(R"({"id": "rate"})")).diagnostics, "spec.bad_patch",
              "control[0].id"));
    CHECK(has(cs::patch_control(parsed.spec, "lr", json::parse(R"({"kind": "dial"})")).diagnostics,
              "spec.unknown_control_kind", "control[0].kind"));
    auto mismatch = cs::patch_control(parsed.spec, "pause", json::parse(R"({"bind": "param:lr"})"));
    CHECK(mismatch.applied);
    CHECK(has(mismatch.diagnostics, "spec.control_mismatch", "control[1].bind"));
}

TEST_CASE("a patch that makes a panel a scene keeps the spec at version 2") {
    cs::Spec s = example();
    REQUIRE(s.version == 2);
    REQUIRE_FALSE(s.panels.empty());
    const std::string id = s.panels[0].id;
    auto r = cs::patch_panel(s, id, json::parse(R"({"primitive": "scene", "bind": null, "options": null})"));
    CHECK(r.applied);
    CHECK(r.spec.version == 2);
    CHECK(r.spec.panels[0].primitive == cs::Primitive::scene);
    CHECK(has(r.diagnostics, "spec.missing_slot", "panel[0].bind.scene"));
}
