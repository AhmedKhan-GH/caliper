// The instruments of the spec runtime (design/instruments.md) without a
// window: the events of a scene panel's picks and selections, and the test
// seam that asks for a pick.
#include "spec_runtime/spec_runtime.h"

#include <caliper_builder/session/events.h>
#include <caliper_builder/spec/format.h>

#include <doctest/doctest.h>

#include <fstream>
#include <sstream>

using namespace caliper_host;
namespace spec = caliper_builder::spec;
using nlohmann::json;

TEST_CASE("scene panel: a pick becomes a pick event with the panel id and the scene as its view") {
    caliper_builder::Event ev;
    const std::string detail =
        R"({"entity": "modules", "primitive": "boxes", "index": 2, "row_id": 2, "position": [0.0, 2.0, 0.0]})";
    REQUIRE(spec_scene_event("pick", "world", "model", detail, &ev));
    CHECK(ev.kind == caliper_builder::EventKind::pick);
    CHECK(ev.panel == "world");
    CHECK(ev.view == "model");
    CHECK(ev.origin == "ui");
    CHECK(ev.fields == json::parse(detail));
}

TEST_CASE("scene panel: a selection, and a cleared selection, become select events") {
    caliper_builder::Event ev;
    const std::string detail =
        R"({"entities": [{"entity": "modules", "primitive": "boxes", "indices": [0, 1], "row_ids": [0, 1]}], "truncated": false})";
    REQUIRE(spec_scene_event("select", "world", "model", detail, &ev));
    CHECK(ev.kind == caliper_builder::EventKind::select);
    CHECK(ev.fields.at("entities").size() == 1);
    REQUIRE(spec_scene_event("select", "world", "model", "{}", &ev));
    CHECK(ev.fields == json{{"entities", json::array()}, {"truncated", false}});
    CHECK_FALSE(spec_scene_event("pick", "world", "model", "not json", &ev));
    CHECK_FALSE(spec_scene_event("pick", "world", "model", "[1, 2]", &ev));
}

TEST_CASE("scene panel: the test seam asks for a pick only in a scene panel of the spec") {
    std::ifstream in(std::string(CALIPER_SPEC_FIXTURES) + "/instruments.spec.toml", std::ios::binary);
    std::stringstream ss;
    ss << in.rdbuf();
    auto parsed = spec::parse_toml(ss.str());
    REQUIRE(parsed.ok());
    SpecTextures textures;
    SpecRuntime runtime(textures);
    CHECK_FALSE(runtime.pick_scene("world", 0.5f, 0.5f));   // no spec yet
    runtime.set_spec(parsed.spec);
    CHECK(runtime.pick_scene("world", 0.5f, 0.5f));
    CHECK_FALSE(runtime.pick_scene("lr", 0.5f, 0.5f));      // a control
    CHECK_FALSE(runtime.pick_scene("nothing", 0.5f, 0.5f));
}
