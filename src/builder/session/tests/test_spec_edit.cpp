// Tests of the spec edits as pure functions: each command's own rules, the
// refusal of an edit that leaves an error, the layout insertions and removals,
// the flat layout and its hostile forms, and spec text checks.
#include <caliper_builder/session/spec_edit.h>
#include <caliper_builder/spec/format.h>

#include <doctest/doctest.h>

#include "helpers.h"

using namespace caliper_builder;
using nlohmann::json;
using spec::LayoutNode;

namespace {

spec::Spec base() {
    spec::Spec s;
    s.applet.id = "t";
    return s;
}

SpecEdit edit(const spec::Spec& s, const char* name, const json& args, const spec::SourceResolver& r = {}) {
    return apply_spec_edit(s, name, args, r);
}

spec::Spec must(const SpecEdit& e) {
    INFO(e.error.code << " at " << e.error.where << ": " << e.error.message);
    REQUIRE(e.error.ok());
    return e.spec;
}

// A spec with a table source m (columns step, loss) and a tensor source h.
spec::Spec two_sources() {
    spec::Spec s = base();
    s = must(edit(s, "source.add", {{"id", "m"}, {"catalog", "table:metrics"}}));
    s = must(edit(s, "source.add", {{"id", "h"}, {"catalog", "tensor:heat"}}));
    return s;
}

std::optional<spec::ResolvedSource> resolver(const std::string& id) {
    spec::ResolvedSource r;
    if (id == "table:metrics") {
        r.kind = "table";
        r.columns = std::vector<spec::ResolvedColumn>{{"step", "BIGINT"}, {"loss", "DOUBLE"}};
        r.wire = true;
        return r;
    }
    if (id == "tensor:heat") {
        r.kind = "tensor";
        r.dtype = "f32";
        r.shape = std::vector<std::int64_t>{4, 5};
        r.wire = false;
        return r;
    }
    return std::nullopt;
}

std::string layout_text(const spec::Spec& s) {
    if (!s.layout) return "none";
    return spec::to_json(*s.layout).dump();
}

}  // namespace

TEST_CASE("source.add, source.patch and source.remove keep their rules") {
    spec::Spec s = two_sources();
    CHECK(edit(s, "source.add", {{"id", "m"}, {"catalog", "table:other"}}).error.code == "source.exists");
    CHECK(edit(s, "source.add", {{"id", "d"}, {"from", "nope"}}).error.code == "spec.unknown_source");
    CHECK(edit(s, "source.add", {{"id", "d"}, {"catalog", "table:x"}, {"sql", "SELECT 1"}}).error.code == "spec.source_form");
    CHECK(edit(s, "source.add", {{"id", "d"}, {"from", "m"}, {"transform", {{{"op", "normalise"}, {"mode", "l2"}}}}})
              .error.code == "spec.bad_transform");
    // With the resolver, a transform is checked against the known shape.
    CHECK(edit(s, "source.add",
               {{"id", "d"}, {"from", "h"}, {"transform", {{{"op", "slice"}, {"axis", 2}, {"index", 0}}}}}, resolver)
              .error.code == "spec.bad_transform");
    const SpecEdit added =
        edit(s, "source.add", {{"id", "d"}, {"from", "h"}, {"transform", {{{"op", "reduce"}, {"axis", 1}, {"fn", "mean"}}}}},
             resolver);
    REQUIRE(added.error.ok());
    CHECK(added.result == json{{"id", "d"}});
    bool unused = false;
    for (const auto& w : added.warnings) unused = unused || w.code == "spec.unused_source";
    CHECK(unused);

    CHECK(edit(s, "source.patch", {{"id", "zz"}, {"patch", {{"sql", "SELECT 1"}}}}).error.code == "source.not_found");
    const SpecEdit patched = edit(s, "source.patch", {{"id", "m"}, {"patch", {{"catalog", "table:other"}}}});
    REQUIRE(patched.error.ok());
    CHECK(patched.result["source"] == json{{"id", "m"}, {"catalog", "table:other"}});
    CHECK(edit(s, "source.patch", {{"id", "m"}, {"patch", {{"catalog", nullptr}}}}).error.code == "spec.source_form");

    spec::Spec used = must(edit(s, "panel.create", {{"id", "p"}, {"primitive", "series"}, {"bind", {{"y", "m.loss"}}}}));
    used = must(edit(used, "source.add", {{"id", "d"}, {"from", "h"}}));
    const SpecEdit in_panel = edit(used, "source.remove", {{"id", "m"}});
    CHECK(in_panel.error.code == "source.in_use");
    CHECK(in_panel.error.where == "panel[0].bind.y");
    const SpecEdit in_source = edit(used, "source.remove", {{"id", "h"}});
    CHECK(in_source.error.code == "source.in_use");
    CHECK(in_source.error.where == "source[2].from");
    CHECK(edit(used, "source.remove", {{"id", "zz"}}).error.code == "source.not_found");
    const spec::Spec removed = must(edit(used, "source.remove", {{"id", "d"}}));
    CHECK(removed.sources.size() == 2);
}

TEST_CASE("panel.create places a panel and refuses one that leaves an error") {
    spec::Spec s = two_sources();
    CHECK(edit(s, "panel.create", {{"id", "p"}, {"primitive", "series"}, {"bind", {{"wheel", "m.loss"}}}}).error.code ==
          "spec.unknown_slot");
    CHECK(edit(s, "panel.create", {{"id", "p"}, {"primitive", "series"}, {"bind", json::object()}}).error.code ==
          "spec.missing_slot");
    CHECK(edit(s, "panel.create", {{"id", "p"}, {"primitive", "series"}, {"bind", {{"y", "nope.loss"}}}}).error.code ==
          "spec.unknown_source");
    CHECK(edit(s, "panel.create", {{"id", "p"}, {"primitive", "series"}, {"bind", {{"y", "m.nope"}}}}, resolver)
              .error.code == "spec.unknown_column");
    CHECK(edit(s, "panel.create", {{"id", "p"}, {"primitive", "heatmap"}, {"bind", {{"values", "h"}}},
                                   {"options", {{"bins", 3}}}})
              .error.code == "spec.unknown_option");

    // The first panel is the whole layout; the next goes to its right.
    spec::Spec one = must(edit(s, "panel.create", {{"id", "a"}, {"primitive", "series"}, {"bind", {{"y", "m.loss"}}}}));
    CHECK(layout_text(one) == "\"a\"");
    spec::Spec two = must(edit(one, "panel.create", {{"id", "b"}, {"primitive", "heatmap"}, {"bind", {{"values", "h"}}}}));
    CHECK(layout_text(two) == R"({"children":["a","b"],"split":"horizontal"})");
    // A warning about portability comes back with the resolver.
    const SpecEdit warned =
        edit(one, "panel.create", {{"id", "b"}, {"primitive", "heatmap"}, {"bind", {{"values", "h"}}}}, resolver);
    REQUIRE(warned.error.ok());
    bool portable_warning = false;
    for (const auto& w : warned.warnings) portable_warning = portable_warning || w.code == "spec.not_portable";
    CHECK(portable_warning);

    CHECK(edit(two, "panel.create", {{"id", "a"}, {"primitive", "series"}, {"bind", {{"y", "m.loss"}}}}).error.code ==
          "panel.exists");
    CHECK(edit(two, "panel.create",
               {{"id", "c"}, {"primitive", "series"}, {"bind", {{"y", "m.step"}}}, {"place", {{"target", "zz"}, {"where", "below"}}}})
              .error.code == "panel.not_found");
    spec::Spec below = must(edit(two, "panel.create",
                                 {{"id", "c"}, {"primitive", "series"}, {"bind", {{"y", "m.step"}}},
                                  {"place", {{"target", "a"}, {"where", "below"}, {"ratio", 0.4}}}}));
    CHECK(layout_text(below) ==
          R"({"children":[{"children":["a","c"],"ratio":0.6,"split":"vertical"},"b"],"split":"horizontal"})");
    spec::Spec above = must(edit(two, "panel.create",
                                 {{"id", "c"}, {"primitive", "series"}, {"bind", {{"y", "m.step"}}},
                                  {"place", {{"target", "b"}, {"where", "above"}, {"ratio", 0.25}}}}));
    CHECK(layout_text(above) ==
          R"({"children":["a",{"children":["c","b"],"ratio":0.25,"split":"vertical"}],"split":"horizontal"})");
    spec::Spec tabbed = must(edit(two, "panel.create",
                                  {{"id", "c"}, {"primitive", "series"}, {"bind", {{"y", "m.step"}}},
                                   {"place", {{"target", "b"}, {"where", "tabbed"}}}}));
    CHECK(layout_text(tabbed) == R"({"children":["a",{"tabs":["b","c"]}],"split":"horizontal"})");
    // Tabbing next to a panel already in tabs adds a tab after it.
    spec::Spec more = must(edit(tabbed, "panel.create",
                                {{"id", "d"}, {"primitive", "series"}, {"bind", {{"y", "m.step"}}},
                                 {"place", {{"target", "b"}, {"where", "tabbed"}}}}));
    CHECK(layout_text(more) == R"({"children":["a",{"tabs":["b","d","c"]}],"split":"horizontal"})");
}

TEST_CASE("panel.patch and panel.remove keep the layout whole") {
    spec::Spec s = two_sources();
    s = must(edit(s, "panel.create", {{"id", "a"}, {"primitive", "series"}, {"bind", {{"y", "m.loss"}}}}));
    s = must(edit(s, "panel.create", {{"id", "b"}, {"primitive", "heatmap"}, {"bind", {{"values", "h"}}}}));
    s = must(edit(s, "panel.create", {{"id", "c"}, {"primitive", "series"}, {"bind", {{"y", "m.step"}}},
                                      {"place", {{"target", "a"}, {"where", "below"}}}}));
    const SpecEdit titled = edit(s, "panel.patch", {{"id", "b"}, {"patch", {{"title", "Heat"}, {"options", {{"colormap", "magma"}}}}}});
    REQUIRE(titled.error.ok());
    CHECK(titled.result["panel"] == json{{"id", "b"},
                                         {"primitive", "heatmap"},
                                         {"title", "Heat"},
                                         {"bind", {{"values", "h"}}},
                                         {"options", {{"colormap", "magma"}}}});
    CHECK(edit(s, "panel.patch", {{"id", "zz"}, {"patch", {{"title", "x"}}}}).error.code == "panel.not_found");
    CHECK(edit(s, "panel.patch", {{"id", "b"}, {"patch", {{"primitive", "series"}}}}).error.code == "spec.unknown_slot");
    CHECK(edit(s, "panel.patch", {{"id", "b"}, {"patch", {{"options", {{"colormap", "jet"}}}}}}).error.code ==
          "spec.bad_value");

    const spec::Spec without_c = must(edit(s, "panel.remove", {{"id", "c"}}));
    CHECK(layout_text(without_c) == R"({"children":["a","b"],"split":"horizontal"})");
    const spec::Spec without_a = must(edit(without_c, "panel.remove", {{"id", "a"}}));
    CHECK(layout_text(without_a) == "\"b\"");
    const spec::Spec empty = must(edit(without_a, "panel.remove", {{"id", "b"}}));
    CHECK(!empty.layout);
    CHECK(edit(empty, "panel.remove", {{"id", "b"}}).error.code == "panel.not_found");
}

TEST_CASE("layout.place moves a panel next to another") {
    spec::Spec s = two_sources();
    s = must(edit(s, "panel.create", {{"id", "a"}, {"primitive", "series"}, {"bind", {{"y", "m.loss"}}}}));
    s = must(edit(s, "panel.create", {{"id", "b"}, {"primitive", "heatmap"}, {"bind", {{"values", "h"}}}}));
    s = must(edit(s, "panel.create", {{"id", "c"}, {"primitive", "series"}, {"bind", {{"y", "m.step"}}}}));
    CHECK(edit(s, "layout.place", {{"panel", "a"}, {"target", "a"}, {"where", "below"}}).error.code == "layout.same_panel");
    CHECK(edit(s, "layout.place", {{"panel", "zz"}, {"target", "a"}, {"where", "below"}}).error.code == "panel.not_found");
    CHECK(edit(s, "layout.place", {{"panel", "a"}, {"target", "zz"}, {"where", "below"}}).error.code == "panel.not_found");
    const SpecEdit moved = edit(s, "layout.place", {{"panel", "c"}, {"target", "a"}, {"where", "above"}, {"ratio", 0.3}});
    REQUIRE(moved.error.ok());
    CHECK(layout_text(moved.spec) ==
          R"({"children":[{"children":["c","a"],"ratio":0.3,"split":"vertical"},"b"],"split":"horizontal"})");
    CHECK(moved.result["layout"]["root"] == "n1");
    CHECK(moved.result["layout"]["nodes"].size() == 2);
}

TEST_CASE("the flat layout refuses hostile forms") {
    spec::Spec s = two_sources();
    s = must(edit(s, "panel.create", {{"id", "a"}, {"primitive", "series"}, {"bind", {{"y", "m.loss"}}}}));
    s = must(edit(s, "panel.create", {{"id", "b"}, {"primitive", "heatmap"}, {"bind", {{"values", "h"}}}}));
    auto node = [](const char* id, const char* split, json children, json ratio = nullptr) {
        return json{{"id", id}, {"split", split}, {"ratio", ratio}, {"children", std::move(children)}};
    };
    auto set = [&](json root, json nodes) {
        return edit(s, "layout.set", {{"layout", {{"root", std::move(root)}, {"nodes", std::move(nodes)}}}});
    };
    CHECK(set("n1", json::array({node("n1", "vertical", {"a", "zz"})})).error.code == "panel.not_found");
    CHECK(set("n1", json::array({node("n1", "vertical", {"a", "zz"})})).error.where == "args.layout.nodes[0].children[1]");
    CHECK(set("zz", json::array()).error.code == "panel.not_found");
    CHECK(set("a", json::array({node("a", "vertical", {"a", "b"})})).error.code == "layout.invalid");
    CHECK(set("n1", json::array({node("n1", "vertical", {"a", "b"}), node("n1", "vertical", {"a", "b"})})).error.code == "layout.invalid");
    CHECK(set("n1", json::array({node("n1", "vertical", {"n2", "n2"}), node("n2", "tabs", {"a"})})).error.code == "layout.invalid");
    CHECK(set("n1", json::array({node("n1", "vertical", {"a", "n2"}), node("n2", "vertical", {"b", "n1"})})).error.code ==
          "layout.invalid");
    CHECK(set("n1", json::array({node("n1", "vertical", {"a", "b"}), node("n2", "tabs", {"a"})})).error.code == "layout.invalid");
    CHECK(set("n1", json::array({node("n1", "tabs", {"a", "b"}, 0.5)})).error.code == "layout.invalid");
    CHECK(set("n1", json::array({node("n1", "vertical", {"a"})})).error.code == "spec.bad_layout");
    CHECK(set("n1", json::array({node("n1", "vertical", {"a", "a"})})).error.code == "spec.duplicate_placement");
    // Deeper than the format's 8 nodes.
    json chain = json::array();
    for (int i = 1; i <= 9; ++i)
        chain.push_back(node(("n" + std::to_string(i)).c_str(), "tabs",
                             json::array({i < 9 ? json("n" + std::to_string(i + 1)) : json("a")})));
    CHECK(set("n1", chain).error.code == "spec.too_deep");
    // A good one, with a title, round trips.
    json titled = node("n2", "vertical", {"a", "b"}, 0.7);
    titled["title"] = "Both";
    const SpecEdit good = set("n1", json::array({node("n1", "tabs", {"n2"}), titled}));
    REQUIRE(good.error.ok());
    CHECK(good.result["layout"] == json{{"root", "n1"}, {"nodes", {node("n1", "tabs", {"n2"}), titled}}});
    // A panel left out is a warning, not an error.
    const SpecEdit partial = set("a", json::array());
    REQUIRE(partial.error.ok());
    CHECK(partial.warnings.at(0).code == "spec.unplaced_panel");
}

TEST_CASE("spec.set_text checks the text and locates its errors") {
    const SpecEdit bad = edit(base(), "spec.set_text", {{"text", "caliper_spec = 0\n[applet]\nid = \"x\"\n[[panel]]\nid = \"p\"\nprimitive = \"pie\"\n"}});
    CHECK(bad.error.code == "spec.unknown_primitive");
    const SpecEdit syntax = edit(base(), "spec.set_text", {{"text", "caliper_spec = \n"}});
    CHECK(syntax.error.code == "spec.syntax");
    CHECK(syntax.error.where.find(':') != std::string::npos);
    const SpecEdit good = edit(base(), "spec.set_text",
                               {{"text", session_test::read_text(session_test::fixture_path("attention-entropy.spec.toml"))},
                                {"dry_run", true}});
    REQUIRE(good.error.ok());
    CHECK(good.spec == session_test::parse_fixture("attention-entropy.spec.toml"));
    CHECK(good.result == json{{"id", "attention-entropy"}});
    CHECK(apply_spec_edit(base(), "panel.explode", json::object()).error.code == "command.unknown");
}

TEST_CASE("an edit leaves the spec within the format's limits") {
    spec::Spec s = base();
    for (int i = 0; i < 64; ++i) s = must(edit(s, "source.add", {{"id", "s" + std::to_string(i)}, {"sql", "SELECT 1"}}));
    CHECK(edit(s, "source.add", {{"id", "one_more"}, {"sql", "SELECT 1"}}).error.code == "spec.too_many");
}

// ---- spec version 1: scene panels and controls in the layout ----------------

namespace {

json scene_args(const std::string& id) {
    return {{"id", id}, {"primitive", "scene"}, {"title", "Model"}, {"bind", {{"scene", "live:model"}}}};
}

spec::Control slider(const std::string& id) {
    spec::Control c;
    c.id = id;
    c.kind = spec::ControlKind::slider;
    c.bind = "param:" + id;
    return c;
}

}  // namespace

TEST_CASE("panel.create of a scene panel raises a version 0 spec to version 1 in the same edit") {
    const spec::Spec s = base();
    REQUIRE(s.version == 0);
    SpecEdit e = edit(s, "panel.create", scene_args("world"));
    spec::Spec next = must(e);
    CHECK(next.version == 1);
    REQUIRE(next.panels.size() == 1);
    CHECK(next.panels[0].primitive == spec::Primitive::scene);
    CHECK(next.panels[0].bind.at("scene").bindings == std::vector<std::string>{"live:model"});
    CHECK(e.result == json{{"id", "world"}});
    CHECK(e.warnings.empty());   // no resolver: the scene is not looked up
    // A resolver that does not know the scene gives the warning of an offline source.
    SpecEdit offline = edit(s, "panel.create", scene_args("world"), resolver);
    REQUIRE(offline.error.ok());
    REQUIRE(offline.warnings.size() == 1);
    CHECK(offline.warnings[0].code == "spec.unresolved_source");
    // A panel of version 0 leaves the version as it is.
    CHECK(must(edit(two_sources(), "panel.create", {{"id", "p"}, {"primitive", "series"}, {"bind", {{"y", "m.loss"}}}},
                    resolver))
              .version == 0);
    // The scene slot holds live:<scene name>.
    SpecEdit bad = edit(s, "panel.create", {{"id", "w"}, {"primitive", "scene"}, {"bind", {{"scene", "model"}}}});
    CHECK(bad.error.code == "spec.bad_binding");
    // panel.patch takes the scene options.
    SpecEdit patched = edit(next, "panel.patch", {{"id", "world"}, {"patch", {{"options", {{"up", "+z"}, {"axes", false}}}}}});
    spec::Spec after = must(patched);
    CHECK(after.panels[0].options.scene.up == std::optional<std::string>("+z"));
    CHECK(patched.result.at("panel").at("options") == json{{"axes", false}, {"up", "+z"}});
}

TEST_CASE("the default layout puts a row of controls under the panels") {
    spec::Spec s = base();
    s.version = 1;
    s.controls.push_back(slider("lr"));
    REQUIRE(effective_layout(s));
    CHECK(*effective_layout(s) == LayoutNode::of_panel("lr"));
    s.controls.push_back(slider("momentum"));
    s = must(edit(s, "panel.create", scene_args("world")));
    // panel.create writes the layout: the panel above the controls it found.
    REQUIRE(s.layout);
    CHECK(layout_text(s) == layout_text([] {
              spec::Spec t;
              t.applet.id = "t";
              LayoutNode row;
              row.kind = LayoutNode::Kind::split;
              row.split = spec::SplitDirection::horizontal;
              row.children = {LayoutNode::of_panel("lr"), LayoutNode::of_panel("momentum")};
              LayoutNode root;
              root.kind = LayoutNode::Kind::split;
              root.split = spec::SplitDirection::vertical;
              root.children = {LayoutNode::of_panel("world"), row};
              t.layout = root;
              return t;
          }()));
    // A second panel goes beside the first, and the controls stay under both.
    s = must(edit(s, "panel.create", scene_args("other")));
    REQUIRE(s.layout->children.size() == 2);
    CHECK(s.layout->children[0].kind == LayoutNode::Kind::split);
    CHECK(s.layout->children[0].split == spec::SplitDirection::horizontal);
    CHECK(s.layout->children[1].children.size() == 2);
    // Without a layout of its own, the spec places panels as version 0 and the controls under them.
    spec::Spec d = base();
    d.version = 1;
    d.panels = s.panels;
    d.controls = s.controls;
    const auto root = effective_layout(d);
    REQUIRE(root);
    CHECK(root->split == spec::SplitDirection::vertical);
    REQUIRE(root->children.size() == 2);
    CHECK(root->children[0].children.size() == 2);   // the panels, as version 0 places them
    CHECK(root->children[1].split == spec::SplitDirection::horizontal);
}

TEST_CASE("layout.place and the flat layout accept control ids") {
    spec::Spec s = base();
    s = must(edit(s, "panel.create", scene_args("world")));
    s.controls.push_back(slider("lr"));
    s.controls.push_back(slider("momentum"));
    s.layout.reset();
    s = must(edit(s, "layout.place", {{"panel", "momentum"}, {"target", "world"}, {"where", "beside"}}));
    const json flat = layout_to_flat(*s.layout, s);
    for (const json& node : flat.at("nodes")) {
        CHECK(node.at("id") != "lr");
        CHECK(node.at("id") != "momentum");
    }
    FlatLayoutParse back = layout_from_flat(flat, s);
    REQUIRE(back.error.ok());
    CHECK(back.root == *s.layout);
    SpecEdit set = edit(s, "layout.set", {{"layout", {{"root", "n1"}, {"nodes", json::array({{{"id", "n1"}, {"split", "vertical"}, {"children", {"world", "lr", "momentum"}}}})}}}});
    spec::Spec placed = must(set);
    CHECK(placed.layout->children.size() == 3);
    SpecEdit clash = edit(s, "layout.set", {{"layout", {{"root", "lr"}, {"nodes", json::array({{{"id", "lr"}, {"split", "vertical"}, {"children", {"world", "momentum"}}}})}}}});
    CHECK(clash.error.code == "layout.invalid");
    CHECK(edit(s, "layout.place", {{"panel", "nothing"}, {"target", "world"}, {"where", "beside"}}).error.code ==
          "panel.not_found");
}
