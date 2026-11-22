// The spec tests of scene entities, the scene options box and views, and the colormaps
// (design/instrument-interface.md, Scene Entities and Layout and Viewport).
//
// Each test reads a version 1 TOML document with check_toml(), which parses
// and validates it, and then looks for a diagnostic by its code and its exact
// path. The valid cases check that no diagnostic of the entity checks
// appears. Other checks may add their own diagnostics, for example
// spec.unused_source, and those are not part of these tests. The entity_style
// cases read the typed view of the options of each draw kind. The colormaps
// cases check the sixteen names and their ids. All test case names start with
// "entities: " or "colormaps: ", so that a filter can run them alone.
#include <doctest/doctest.h>

#include <caliper_builder/spec/colormaps.h>
#include <caliper_builder/spec/entity.h>
#include <caliper_builder/spec/format.h>
#include <caliper_builder/spec/validate.h>

#include <set>
#include <string>
#include <vector>

#include "helpers.h"

using namespace spec_test;
using nlohmann::json;

namespace {

// The codes that validate_entities.cpp emits. A valid document has none of them.
const std::set<std::string>& entity_codes() {
    static const std::set<std::string> c = {"spec.bad_value",    "spec.duplicate_id", "spec.too_many",
                                            "spec.bad_binding",  "spec.unknown_source", "spec.missing_slot",
                                            "spec.unknown_slot", "spec.slot_mismatch", "spec.unknown_option",
                                            "spec.out_of_range", "spec.bad_reference"};
    return c;
}

std::string entity_errors(const std::vector<cs::Diagnostic>& ds) {
    std::string s;
    for (const auto& d : ds)
        if (entity_codes().count(d.code) || d.severity == cs::Severity::error)
            s += d.code + " at " + d.path + ": " + d.message + "\n";
    return s;
}

const std::string kHead = "caliper_spec = 1\n[applet]\nid = \"a\"\n"
                          "[[source]]\nid = \"grid\"\nsql = \"SELECT 1\"\n"
                          "[[source]]\nid = \"loss\"\nsql = \"SELECT 1\"\n"
                          "[[source]]\nid = \"cloud\"\nsql = \"SELECT 1\"\n";

// A panel with the given primitive, options table and entity text.
std::string panel(const std::string& entities, const std::string& options = "{ up = \"+z\" }",
                  const std::string& primitive = "scene") {
    return kHead + "[[panel]]\nid = \"p\"\nprimitive = \"" + primitive + "\"\noptions = " + options + "\n" + entities;
}

// One entity of the first panel.
std::string entity(const std::string& path, const std::string& draw, const std::string& bind,
                   const std::string& options = "", const std::string& pose = "") {
    std::string s = "  [[panel.entity]]\n  path = \"" + path + "\"\n  draw = \"" + draw + "\"\n  bind = " + bind + "\n";
    if (!options.empty()) s += "  options = " + options + "\n";
    if (!pose.empty()) s += "  pose = " + pose + "\n";
    return s;
}

std::vector<cs::Diagnostic> check(const std::string& text) { return cs::check_toml(text).diagnostics; }

// The diagnostics of one surface, lines or points entity with the given options.
std::vector<cs::Diagnostic> with_options(const std::string& draw, const std::string& bind,
                                         const std::string& options) {
    return check(panel(entity("e", draw, bind, options)));
}

const std::string kSurface = "\"grid\"";
const std::string kLines = "{ x = \"loss.step\", z = \"loss.value\" }";
const std::string kPoints = "{ positions = \"cloud.xyz\" }";

}  // namespace

TEST_CASE("entities: a valid scene panel has no entity diagnostic") {
    const std::string text =
        panel(entity("terrain", "surface", "\"grid\"") +
                  entity("loss", "lines", "{ x = \"loss.step\", z = \"loss.value\" }",
                         "{ mode = \"strip\", y = \"max\", own = [\"z\"], width_px = 2.0 }") +
                  entity("cloud/a", "points", "{ positions = \"cloud.xyz\" }"),
              "{ up = \"+z\", box = [3.0, 1.0, 0.8], views = [{ name = \"overview\", azimuth = 30.0, "
              "distance = 4.0, elevation = 20.0, target = [0.0, 1.0, 0.0] }, { name = \"top\", azimuth = 0.0, "
              "distance = 2.0, elevation = 89.0, target = [0.0, 0.0, 0.0] }] }");
    const auto r = cs::check_toml(text);
    for (const auto& d : r.diagnostics) {
        INFO(dump(r.diagnostics));
        CHECK_FALSE(entity_codes().count(d.code));
    }
    CHECK(entity_errors(r.diagnostics).empty());
}

TEST_CASE("entities: a scene panel with only entities needs no scene slot") {
    const auto ds = check(panel(entity("terrain", "surface", kSurface)));
    INFO(dump(ds));
    CHECK(entity_errors(ds).empty());
}

TEST_CASE("entities: refusals of the path") {
    auto ds = check(panel(entity("Bad Path", "surface", kSurface)));
    CHECK(has(ds, "spec.bad_value", "panel[0].entity[0].path"));
    ds = check(panel(entity("a//b", "surface", kSurface)));
    CHECK(has(ds, "spec.bad_value", "panel[0].entity[0].path"));
    ds = check(panel(entity("a/b/c/d/e/f/g/h/i", "surface", kSurface)));
    CHECK(has(ds, "spec.bad_value", "panel[0].entity[0].path"));
    ds = check(panel(entity("same", "surface", kSurface) + entity("same", "surface", kSurface)));
    CHECK(has(ds, "spec.duplicate_id", "panel[0].entity[1].path"));
}

TEST_CASE("entities: refusals of the panel and the limits") {
    auto ds = check(panel(entity("a", "surface", kSurface), "{}", "heatmap"));
    INFO(dump(ds));
    CHECK(has(ds, "spec.bad_value", "panel[0].entity"));

    std::string many;
    for (int i = 0; i < 65; ++i) many += entity("e" + std::to_string(i), "surface", kSurface);
    ds = check(panel(many));
    CHECK(has(ds, "spec.too_many", "panel[0].entity"));

    std::string max;
    for (int i = 0; i < 64; ++i) max += entity("e" + std::to_string(i), "surface", kSurface);
    CHECK_FALSE(has(check(panel(max)), "spec.too_many"));
}

TEST_CASE("entities: refusals of the bind") {
    auto ds = check(panel(entity("a", "surface", "{ height = \"grid.v\", wat = \"grid.w\" }")));
    CHECK(has(ds, "spec.unknown_slot", "panel[0].entity[0].bind.wat"));
    ds = check(panel(entity("a", "surface", "{ values = \"grid.v\" }")));
    CHECK(has(ds, "spec.missing_slot", "panel[0].entity[0].bind"));
    ds = check(panel(entity("a", "points", "{ x = \"cloud.x\" }")));
    CHECK(has(ds, "spec.missing_slot", "panel[0].entity[0].bind"));
    ds = check(panel(entity("a", "points", "{ positions = \"cloud.xyz\", colors = \"cloud.c\", values = \"cloud.v\" }")));
    CHECK(has(ds, "spec.slot_mismatch", "panel[0].entity[0].bind"));
    ds = check(panel(entity("a", "points", "{ positions = \"cloud.xyz\", x = \"cloud.x\" }")));
    CHECK(has(ds, "spec.slot_mismatch", "panel[0].entity[0].bind"));
    ds = check(panel(entity("a", "surface", "\"nowhere\"")));
    CHECK(has(ds, "spec.unknown_source", "panel[0].entity[0].bind"));
    ds = check(panel(entity("a", "lines", "{ x = \"nowhere.x\", z = \"loss.value\" }")));
    CHECK(has(ds, "spec.unknown_source", "panel[0].entity[0].bind.x"));
}

TEST_CASE("entities: refusals of the options") {
    CHECK(has(with_options("lines", kLines, "{ size_px = 3.0 }"), "spec.unknown_option",
              "panel[0].entity[0].options.size_px"));
    CHECK(has(with_options("surface", kSurface, "{ colormap = \"jet\" }"), "spec.bad_value",
              "panel[0].entity[0].options.colormap"));
    CHECK(has(with_options("surface", kSurface, "{ depth_offset = 9 }"), "spec.out_of_range",
              "panel[0].entity[0].options.depth_offset"));
    CHECK(has(with_options("surface", kSurface, "{ vmin = 2.0, vmax = 1.0 }"), "spec.out_of_range",
              "panel[0].entity[0].options.vmin"));
    CHECK(has(with_options("points", kPoints, "{ shape = \"star\" }"), "spec.bad_value",
              "panel[0].entity[0].options.shape"));
    CHECK(has(with_options("lines", kLines, "{ mode = \"loop\" }"), "spec.bad_value",
              "panel[0].entity[0].options.mode"));
    CHECK(has(with_options("lines", kLines, "{ own = [\"z\", \"z\"] }"), "spec.bad_value",
              "panel[0].entity[0].options.own"));
    CHECK(has(with_options("lines", kLines, "{ own = [\"w\"] }"), "spec.bad_value",
              "panel[0].entity[0].options.own"));
    CHECK(has(with_options("surface", kSurface, "{ highlight = { along = \"index\", state = \"s\" } }"),
              "spec.bad_value", "panel[0].entity[0].options.highlight.along"));
    CHECK(has(with_options("points", kPoints, "{ size_px = 100.0 }"), "spec.out_of_range",
              "panel[0].entity[0].options.size_px"));
    CHECK(has(with_options("lines", kLines, "{ width_px = 0.0 }"), "spec.out_of_range",
              "panel[0].entity[0].options.width_px"));
}

TEST_CASE("entities: refusals of the pose") {
    auto ds = check(panel(entity("a", "surface", kSurface, "", "{ rotation = [0.0, 0.0, 0.0, 0.0] }")));
    INFO(dump(ds));
    CHECK(has(ds, "spec.bad_value", "panel[0].entity[0].pose.rotation"));
    ds = check(panel(entity("a", "surface", kSurface, "", "{ scale = [1.0, 0.0, 1.0] }")));
    INFO(dump(ds));
    CHECK(has(ds, "spec.out_of_range", "panel[0].entity[0].pose.scale"));
}

TEST_CASE("entities: the box option") {
    auto ds = check(panel("", "{ box = [3.0, 1.0] }"));
    INFO(dump(ds));
    CHECK(ds.size() > 0);
    CHECK((has(ds, "spec.bad_value", "panel[0].options.box") || has(ds, "spec.wrong_type", "panel[0].options.box")));
    ds = check(panel("", "{ box = [3.0, 1.0, 0.0] }"));
    CHECK(has(ds, "spec.bad_value", "panel[0].options.box"));
    ds = check(panel("", "{ box = [3.0, 1.0, 0.8] }", "heatmap"));
    INFO(dump(ds));
    CHECK(has(ds, "spec.unknown_option", "panel[0].options.box"));
    ds = check(panel("", "{ box = \"auto\" }"));
    CHECK_FALSE(has(ds, "spec.bad_value", "panel[0].options.box"));
}

TEST_CASE("entities: the saved views") {
    auto view = [](const std::string& name, double distance, double elevation) {
        return "{ name = \"" + name + "\", azimuth = 0.0, distance = " + std::to_string(distance) +
               ", elevation = " + std::to_string(elevation) + ", target = [0.0, 0.0, 0.0] }";
    };
    auto views = [&](const std::vector<std::string>& vs) {
        std::string s = "{ views = [";
        for (std::size_t i = 0; i < vs.size(); ++i) s += (i ? ", " : "") + vs[i];
        return s + "] }";
    };
    std::vector<std::string> seventeen, sixteen;
    for (int i = 0; i < 17; ++i) seventeen.push_back(view("v" + std::to_string(i), 2.0, 10.0));
    sixteen.assign(seventeen.begin(), seventeen.begin() + 16);
    CHECK(has(check(panel("", views(seventeen))), "spec.too_many", "panel[0].options.views"));
    CHECK_FALSE(has(check(panel("", views(sixteen))), "spec.too_many"));
    CHECK(has(check(panel("", views({view("same", 2, 10), view("same", 2, 10)}))), "spec.duplicate_id",
              "panel[0].options.views[1].name"));
    CHECK(has(check(panel("", views({view("a", 0, 10)}))), "spec.out_of_range",
              "panel[0].options.views[0].distance"));
    CHECK(has(check(panel("", views({view("a", 2, 90)}))), "spec.out_of_range",
              "panel[0].options.views[0].elevation"));
    CHECK(has(check(panel("", views({view("a", 2, 10)}), "heatmap")),"spec.unknown_option",
              "panel[0].options.views"));
}

namespace {

cs::Entity make_entity(cs::DrawKind draw, const json& options) {
    cs::Entity e;
    e.path = "e";
    e.draw = draw;
    e.source = "src";
    e.options = options;
    return e;
}

}  // namespace

TEST_CASE("entities: entity_style reads the options of a points entity") {
    const auto s = cs::entity_style(make_entity(
        cs::DrawKind::points, json{{"colormap", "magma"}, {"vmin", -1.5}, {"vmax", 2.0}, {"size_px", 4.0},
                                   {"shape", "square"}, {"x", 0.25}, {"y", "min"}, {"z", "max"},
                                   {"depth_offset", -2}, {"visible", false}}));
    REQUIRE(s.colormap);
    CHECK(std::get<std::string>(*s.colormap) == "magma");
    REQUIRE(s.vmin);
    CHECK(std::get<double>(*s.vmin) == -1.5);
    REQUIRE(s.vmax);
    CHECK(std::get<double>(*s.vmax) == 2.0);
    CHECK(s.size_px == std::optional<double>(4.0));
    CHECK(s.shape == std::optional<std::string>("square"));
    REQUIRE(s.x);
    CHECK(s.x->kind == cs::AxisPlace::Kind::value);
    CHECK(s.x->value == 0.25);
    REQUIRE(s.y);
    CHECK(s.y->kind == cs::AxisPlace::Kind::min);
    REQUIRE(s.z);
    CHECK(s.z->kind == cs::AxisPlace::Kind::max);
    CHECK(s.depth_offset == std::optional<std::int64_t>(-2));
    REQUIRE(s.visible);
    CHECK(std::get<bool>(*s.visible) == false);
    CHECK_FALSE(s.width_px);
    CHECK_FALSE(s.mode);
    CHECK(s.own.empty());
    CHECK_FALSE(s.highlight);
}

TEST_CASE("entities: entity_style reads the options of a lines entity") {
    const auto s = cs::entity_style(make_entity(
        cs::DrawKind::lines, json{{"mode", "segments"}, {"width_px", 2.0}, {"x", "max"}, {"own", {"y", "z"}}}));
    CHECK(s.mode == std::optional<std::string>("segments"));
    CHECK(s.width_px == std::optional<double>(2.0));
    REQUIRE(s.x);
    CHECK(s.x->kind == cs::AxisPlace::Kind::max);
    CHECK(s.own == std::vector<std::string>{"y", "z"});
    CHECK_FALSE(s.size_px);
    CHECK_FALSE(s.depth_offset);
    CHECK_FALSE(s.visible);
}

TEST_CASE("entities: entity_style reads references, highlight and an entity without options") {
    const auto s = cs::entity_style(make_entity(
        cs::DrawKind::surface,
        json{{"colormap", {{"state", "palette"}}},
             {"vmin", {{"control", "lo"}}},
             {"highlight", {{"state", "token"}, {"along", "column"}}},
             {"visible", {{"control", "show"}}}}));
    REQUIRE(s.colormap);
    REQUIRE(cs::is_reference(*s.colormap));
    CHECK(std::get<cs::Reference>(*s.colormap).kind == cs::Reference::Kind::state);
    CHECK(std::get<cs::Reference>(*s.colormap).target == "palette");
    REQUIRE(s.vmin);
    CHECK(std::get<cs::Reference>(*s.vmin).kind == cs::Reference::Kind::control);
    REQUIRE(s.highlight);
    CHECK(s.highlight->kind == cs::Reference::Kind::state);
    CHECK(s.highlight->target == "token");
    CHECK(s.highlight_along == std::optional<std::string>("column"));
    REQUIRE(s.visible);
    CHECK(std::get<cs::Reference>(*s.visible).target == "show");

    const auto bare = cs::entity_style(make_entity(cs::DrawKind::surface, json()));
    CHECK_FALSE(bare.colormap);
    CHECK_FALSE(bare.highlight);
    CHECK_FALSE(bare.visible);
}

TEST_CASE("entities: entity_style leaves a bad highlight and an option of the wrong form empty") {
    const auto s = cs::entity_style(
        make_entity(cs::DrawKind::points, json{{"highlight", {{"along", "index"}}}, {"depth_offset", 1.5},
                                                {"size_px", "big"}, {"x", "middle"}}));
    CHECK_FALSE(s.highlight);
    CHECK_FALSE(s.highlight_along);
    CHECK_FALSE(s.depth_offset);
    CHECK_FALSE(s.size_px);
    CHECK_FALSE(s.x);
}

TEST_CASE("colormaps: the sixteen names are in order") {
    const std::vector<std::string> expected = {"viridis", "magma",     "rdbu",   "greys",   "cividis", "inferno",
                                               "plasma",  "turbo",     "twilight", "coolwarm", "batlow", "vik",
                                               "roma",    "oslo",      "lajolla", "berlin"};
    CHECK(cs::colormap_names() == expected);
}

TEST_CASE("colormaps: colormap_id gives the index of each name and nothing for an unknown name") {
    const auto& names = cs::colormap_names();
    REQUIRE(names.size() == 16);
    for (std::size_t i = 0; i < names.size(); ++i) {
        INFO(names[i]);
        CHECK(cs::colormap_id(names[i]) == std::optional<int>(static_cast<int>(i)));
    }
    CHECK_FALSE(cs::colormap_id("jet"));
    CHECK_FALSE(cs::colormap_id(""));
}
