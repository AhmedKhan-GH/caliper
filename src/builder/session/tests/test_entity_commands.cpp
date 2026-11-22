// The session tests of entity.set, entity.remove and camera.save
// (design/instrument-interface.md, Commands), through apply_spec_edit(), and
// of the entities in panel.data (encode_scene_entities).
#include <doctest/doctest.h>

#include <array>
#include <string>
#include <vector>

#include <caliper_builder/session/instrument_commands.h>
#include <caliper_builder/session/panel_data.h>
#include <caliper_builder/spec/format.h>

using namespace caliper_builder;
using nlohmann::json;

namespace {

// A version 1 spec with a scene panel "world", a heatmap panel "grid_view"
// and two sources.
spec::Spec make_spec() {
    const std::string text =
        "caliper_spec = 1\n[applet]\nid = \"a\"\n"
        "[[source]]\nid = \"grid\"\ncatalog = \"tensor:grid\"\n"
        "[[source]]\nid = \"cloud\"\ncatalog = \"tensor:cloud\"\n"
        "[[panel]]\nid = \"world\"\nprimitive = \"scene\"\nbind = { scene = \"live:model\" }\n"
        "options = { up = \"+z\" }\n"
        "[[panel]]\nid = \"flat\"\nprimitive = \"heatmap\"\nbind = { values = \"grid\" }\n";
    auto parsed = spec::parse_toml(text);
    REQUIRE_MESSAGE(parsed.ok(), "the test spec parses");
    return parsed.spec;
}

json entity_json(const std::string& path, const std::string& draw = "surface", const std::string& bind = "grid") {
    return {{"path", path}, {"draw", draw}, {"bind", bind}};
}

SpecEdit set_entity(const spec::Spec& s, const std::string& panel, const json& entity) {
    return apply_spec_edit(s, "entity.set", {{"panel", panel}, {"entity", entity}});
}

const spec::Panel& world(const SpecEdit& e) { return *e.spec.find_panel("world"); }

json orbit(double distance) {
    return {{"target", {1.0, 2.0, 3.0}}, {"distance", distance}, {"azimuth", 30.0}, {"elevation", 15.0}};
}

SpecEdit save_view(const spec::Spec& s, const std::string& panel, const std::string& name, const json& o) {
    json args{{"panel", panel}, {"name", name}};
    if (!o.is_null()) args["orbit"] = o;
    return apply_spec_edit(s, "camera.save", args);
}

SlotData tensor_slot(const std::string& slot, std::vector<std::int64_t> shape) {
    SlotData s;
    s.slot = slot;
    s.binding = "src." + slot;
    auto r = TensorData::allocate(DType::f32, std::move(shape));
    REQUIRE(r.error.ok());
    s.tensor = std::move(r.tensor);
    return s;
}

}  // namespace

TEST_CASE("entity commands: set appends and then replaces by path") {
    const spec::Spec s = make_spec();
    SpecEdit a = set_entity(s, "world", entity_json("terrain"));
    INFO(a.error.code << ": " << a.error.message);
    REQUIRE(a.error.ok());
    CHECK(a.result["panel"] == "world");
    REQUIRE(world(a).entities.size() == 1);
    CHECK(world(a).entities[0].path == "terrain");

    SpecEdit c = set_entity(a.spec, "world", entity_json("second"));
    REQUIRE(c.error.ok());
    REQUIRE(world(c).entities.size() == 2);

    json again = entity_json("terrain");
    again["title"] = "Terrain";
    SpecEdit d = set_entity(c.spec, "world", again);
    REQUIRE(d.error.ok());
    REQUIRE(world(d).entities.size() == 2);
    CHECK(world(d).entities[0].path == "terrain");
    REQUIRE(world(d).entities[0].title.has_value());
    CHECK(*world(d).entities[0].title == "Terrain");
    CHECK(world(d).entities[1].path == "second");
}

TEST_CASE("entity commands: remove takes an entity out") {
    const spec::Spec s = make_spec();
    SpecEdit a = set_entity(s, "world", entity_json("one"));
    SpecEdit b = set_entity(a.spec, "world", entity_json("two"));
    REQUIRE(b.error.ok());
    SpecEdit r = apply_spec_edit(b.spec, "entity.remove", {{"panel", "world"}, {"path", "one"}});
    INFO(r.error.code << ": " << r.error.message);
    REQUIRE(r.error.ok());
    CHECK(r.result["panel"] == "world");
    REQUIRE(world(r).entities.size() == 1);
    CHECK(world(r).entities[0].path == "two");
}

TEST_CASE("entity commands: refusals of the panel and the path") {
    const spec::Spec s = make_spec();
    CHECK(set_entity(s, "flat", entity_json("t")).error.code == "entity.not_scene_panel");
    CHECK(apply_spec_edit(s, "entity.remove", {{"panel", "flat"}, {"path", "t"}}).error.code ==
          "entity.not_scene_panel");
    CHECK(set_entity(s, "nowhere", entity_json("t")).error.code == "panel.not_found");
    CHECK(apply_spec_edit(s, "entity.remove", {{"panel", "world"}, {"path", "t"}}).error.code == "entity.not_found");
}

TEST_CASE("entity commands: a bad entity is refused with the codec's diagnostic") {
    const spec::Spec s = make_spec();
    SpecEdit e = set_entity(s, "world", {{"path", "t"}, {"bind", "grid"}});  // no draw
    CHECK_FALSE(e.error.ok());
    CHECK(e.error.code.rfind("spec.", 0) == 0);
    SpecEdit f = set_entity(s, "world", {{"path", "t"}, {"draw", "cube"}, {"bind", "grid"}});
    CHECK_FALSE(f.error.ok());
    SpecEdit g = apply_spec_edit(s, "entity.set", {{"panel", "world"}, {"entity", "text"}});
    CHECK_FALSE(g.error.ok());
}

TEST_CASE("entity commands: the validator refuses a 65th entity") {
    spec::Spec s = make_spec();
    for (int i = 0; i < 64; ++i) {
        SpecEdit e = set_entity(s, "world", entity_json("e" + std::to_string(i)));
        REQUIRE_MESSAGE(e.error.ok(), e.error.code);
        s = e.spec;
    }
    SpecEdit over = set_entity(s, "world", entity_json("extra"));
    CHECK(over.error.code == "spec.too_many");
    // Replacing an existing path stays within the limit.
    CHECK(set_entity(s, "world", entity_json("e3")).error.ok());
}

TEST_CASE("entity commands: camera.save appends and replaces by name") {
    const spec::Spec s = make_spec();
    SpecEdit a = save_view(s, "world", "front", orbit(4.0));
    INFO(a.error.code << ": " << a.error.message);
    REQUIRE(a.error.ok());
    CHECK(a.result["panel"] == "world");
    REQUIRE(world(a).options.views.size() == 1);
    const spec::SavedView& v = world(a).options.views[0];
    CHECK(v.name == "front");
    CHECK(v.target == std::array<double, 3>{1.0, 2.0, 3.0});
    CHECK(v.distance == 4.0);
    CHECK(v.azimuth == 30.0);
    CHECK(v.elevation == 15.0);

    SpecEdit b = save_view(a.spec, "world", "side", orbit(5.0));
    REQUIRE(b.error.ok());
    CHECK(world(b).options.views.size() == 2);
    SpecEdit c = save_view(b.spec, "world", "front", orbit(9.0));
    REQUIRE(c.error.ok());
    REQUIRE(world(c).options.views.size() == 2);
    CHECK(world(c).options.views[0].name == "front");
    CHECK(world(c).options.views[0].distance == 9.0);
    CHECK(world(c).options.views[1].name == "side");
}

TEST_CASE("entity commands: camera.save refusals") {
    const spec::Spec s = make_spec();
    SpecEdit none = save_view(s, "world", "front", nullptr);
    CHECK(none.error.code == "camera.no_orbit");
    CHECK(none.error.message == "camera.save needs the app's current orbit.");
    CHECK(save_view(s, "world", "front", "left").error.code == "camera.no_orbit");
    CHECK(save_view(s, "flat", "front", orbit(4.0)).error.code == "camera.not_scene_panel");
}

TEST_CASE("entity commands: the validator refuses a 17th saved view") {
    spec::Spec s = make_spec();
    for (int i = 0; i < 16; ++i) {
        SpecEdit e = save_view(s, "world", "v" + std::to_string(i), orbit(2.0 + i));
        REQUIRE_MESSAGE(e.error.ok(), e.error.code);
        s = e.spec;
    }
    CHECK(save_view(s, "world", "v16", orbit(3.0)).error.code == "spec.too_many");
    CHECK(save_view(s, "world", "v2", orbit(3.0)).error.ok());
}

TEST_CASE("entity commands: panel.data gives the entity errors and the slot shapes") {
    spec::Panel panel;
    panel.id = "world";
    panel.primitive = spec::Primitive::scene;
    auto add = [&](const std::string& path, spec::DrawKind draw) {
        spec::Entity e;
        e.path = path;
        e.draw = draw;
        panel.entities.push_back(e);
    };
    add("good", spec::DrawKind::surface);
    add("flat", spec::DrawKind::surface);
    add("badx", spec::DrawKind::surface);
    add("pts", spec::DrawKind::points);
    add("pts2", spec::DrawKind::points);
    add("ragged", spec::DrawKind::points);
    add("resolve", spec::DrawKind::lines);
    add("absent", spec::DrawKind::lines);

    PanelData data;
    auto entry = [&](const std::string& path, std::vector<SlotData> slots) {
        EntityData d;
        d.path = path;
        d.slots = std::move(slots);
        d.generation = 7;
        data.entities.push_back(std::move(d));
    };
    entry("good", {tensor_slot("height", {3, 4}), tensor_slot("x", {3}), tensor_slot("y", {4}),
                   tensor_slot("values", {3, 4})});
    entry("flat", {tensor_slot("height", {12})});
    entry("badx", {tensor_slot("height", {3, 4}), tensor_slot("x", {5})});
    entry("pts", {tensor_slot("positions", {10, 3}), tensor_slot("values", {10})});
    entry("pts2", {tensor_slot("positions", {10, 2})});
    entry("ragged", {tensor_slot("positions", {10, 3}), tensor_slot("values", {9})});
    entry("resolve", {});
    data.entities.back().error = Error("resolve.unavailable", "The source is gone.");

    json out = json::object();
    encode_scene_entities(panel, data, PanelDataLimits{}, &out);
    REQUIRE(out["entities"].is_array());
    const json& list = out["entities"];
    REQUIRE(list.size() == 8);
    CHECK(list[0]["path"] == "good");
    CHECK(list[0]["draw"] == "surface");
    CHECK_FALSE(list[0].contains("error"));
    CHECK(list[0]["generation"] == 7);
    REQUIRE(list[0]["slots"].size() == 4);
    CHECK(list[0]["slots"][0]["slot"] == "height");
    CHECK(list[0]["slots"][0]["binding"] == "src.height");
    CHECK(list[0]["slots"][0]["shape"] == json::array({3, 4}));
    CHECK(list[1]["error"]["code"] == "entity.shape");
    CHECK(list[2]["error"]["code"] == "entity.length");
    CHECK(list[3]["slots"][1]["shape"] == json::array({10}));
    CHECK(list[4]["error"]["code"] == "entity.shape");
    CHECK(list[5]["error"]["code"] == "entity.length");
    CHECK(list[6]["error"]["code"] == "resolve.unavailable");
    CHECK(list[7]["error"]["code"] == "entity.no_data");
    CHECK(list[7]["draw"] == "lines");
}
