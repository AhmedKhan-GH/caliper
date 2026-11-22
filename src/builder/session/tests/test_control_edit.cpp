// Tests of the control commands of spec version 1 (design/instruments.md,
// Control Commands and Control Layout): the layout steps of control.create,
// the commands' own refusals, and, once the spec library reads and validates
// version 1, the edits through the session with their diff, review, dry run
// and undo.
#include "helpers.h"

#include <caliper_builder/session/session.h>
#include <caliper_builder/session/spec_edit.h>
#include <caliper_builder/spec/instrument.h>

#include <doctest/doctest.h>

using namespace caliper_builder;
using namespace session_test;
using nlohmann::json;
using spec::LayoutNode;

namespace caliper_builder::control_layout {
// Defined in control_edit.cpp; no header declares them.
std::optional<spec::LayoutNode> layout_with_controls(const spec::Spec& s);
spec::LayoutNode add_to_control_row(std::optional<spec::LayoutNode> layout, const spec::Spec& s, const std::string& id);
}  // namespace caliper_builder::control_layout

namespace {

namespace cl = caliper_builder::control_layout;

spec::Panel panel(const std::string& id) {
    spec::Panel p;
    p.id = id;
    return p;
}

spec::Control control(const std::string& id, const std::string& bind = "param:lr") {
    spec::Control c;
    c.id = id;
    c.kind = bind.rfind("param:", 0) == 0 || bind == "value" ? spec::ControlKind::slider : spec::ControlKind::button;
    c.bind = bind;
    return c;
}

LayoutNode split(spec::SplitDirection d, std::vector<LayoutNode> children) {
    LayoutNode n;
    n.kind = LayoutNode::Kind::split;
    n.split = d;
    n.children = std::move(children);
    return n;
}

LayoutNode leaf(const std::string& id) { return LayoutNode::of_panel(id); }

constexpr auto V = spec::SplitDirection::vertical;
constexpr auto H = spec::SplitDirection::horizontal;

}  // namespace

TEST_CASE("control layout: the default places the panels, then one row of the controls") {
    spec::Spec s;
    CHECK_FALSE(cl::layout_with_controls(s));
    s.controls = {control("lr")};
    CHECK(*cl::layout_with_controls(s) == leaf("lr"));
    s.controls.push_back(control("pause", "run:pause"));
    CHECK(*cl::layout_with_controls(s) == split(H, {leaf("lr"), leaf("pause")}));
    s.panels = {panel("world")};
    CHECK(*cl::layout_with_controls(s) == split(V, {leaf("world"), split(H, {leaf("lr"), leaf("pause")})}));
    s.panels.push_back(panel("loss"));
    s.controls.pop_back();
    CHECK(*cl::layout_with_controls(s) == split(V, {split(V, {leaf("world"), leaf("loss")}), leaf("lr")}));
    // A spec's own layout is kept as it is.
    s.layout = leaf("world");
    CHECK(*cl::layout_with_controls(s) == leaf("world"));
}

TEST_CASE("control layout: a new control joins the row of controls under the whole layout") {
    spec::Spec s;
    // No layout: the control alone.
    s.controls = {control("lr")};
    CHECK(cl::add_to_control_row(std::nullopt, s, "lr") == leaf("lr"));

    // Panels and no row yet: a vertical split of the layout and the control.
    s.panels = {panel("world"), panel("loss")};
    const LayoutNode panels = split(H, {leaf("world"), leaf("loss")});
    LayoutNode l = cl::add_to_control_row(panels, s, "lr");
    CHECK(l == split(V, {panels, leaf("lr")}));

    // The row of one control becomes a horizontal split, then grows.
    s.controls.push_back(control("pause", "run:pause"));
    l = cl::add_to_control_row(l, s, "pause");
    CHECK(l == split(V, {panels, split(H, {leaf("lr"), leaf("pause")})}));
    s.controls.push_back(control("save", "action:save"));
    LayoutNode two = l;
    two.children[1].ratio = 0.3;
    l = cl::add_to_control_row(two, s, "save");
    CHECK(l == split(V, {panels, split(H, {leaf("lr"), leaf("pause"), leaf("save")})}));
    CHECK_FALSE(l.children[1].ratio);

    // A root that is a row of controls takes the new one.
    s.panels.clear();
    CHECK(cl::add_to_control_row(leaf("lr"), s, "pause") == split(H, {leaf("lr"), leaf("pause")}));

    // A control placed elsewhere is no row: the new control goes under the whole layout.
    s.panels = {panel("world")};
    const LayoutNode beside = split(H, {leaf("world"), leaf("lr")});
    CHECK(cl::add_to_control_row(beside, s, "pause") == split(V, {beside, leaf("pause")}));
    // A vertical root whose last child is a panel gets a new row.
    const LayoutNode stacked = split(V, {leaf("lr"), leaf("world")});
    CHECK(cl::add_to_control_row(stacked, s, "pause") == split(V, {stacked, leaf("pause")}));
}

TEST_CASE("control commands: their own refusals come before the validator") {
    spec::Spec s;
    s.applet.id = "m1";
    s.panels = {panel("world")};
    s.controls = {control("lr")};
    s.version = spec::kInstrumentVersion;

    SpecEdit e = apply_spec_edit(s, "control.patch", {{"id", "momentum"}, {"patch", {{"title", "M"}}}});
    CHECK(e.error.code == "control.not_found");
    CHECK(e.error.where == "args.id");
    e = apply_spec_edit(s, "control.remove", {{"id", "momentum"}});
    CHECK(e.error.code == "control.not_found");
    e = apply_spec_edit(s, "control.create", {{"id", "lr"}, {"kind", "slider"}, {"bind", "param:lr"}});
    CHECK(e.error.code == "control.exists");
    e = apply_spec_edit(s, "control.create", {{"id", "world"}, {"kind", "slider"}, {"bind", "param:lr"}});
    CHECK(e.error.code == "spec.duplicate_id");
    e = apply_spec_edit(s, "control.create", {{"id", "x"}, {"kind", "dial"}, {"bind", "param:lr"}});
    CHECK(e.error.code == "spec.unknown_control_kind");
    e = apply_spec_edit(s, "control.create",
                        {{"id", "x"}, {"kind", "slider"}, {"bind", "param:lr"}, {"place", {{"target", "nope"}, {"where", "below"}}}});
    CHECK(e.error.code == "panel.not_found");
    CHECK(e.error.where == "args.place.target");
    e = apply_spec_edit(s, "control.patch", {{"id", "lr"}, {"patch", {{"options", {{"knob", 1}}}}}});
    CHECK(e.error.code == "spec.unknown_option");
    e = apply_spec_edit(s, "control.patch", {{"id", "lr"}, {"patch", {{"options", {{"min", "low"}}}}}});
    CHECK(e.error.code == "spec.wrong_type");
    CHECK(apply_control_edit(s, "control.move", json::object(), {}).error.code == "command.unknown");
}

TEST_CASE("control commands: create, patch and remove edit the open spec with a diff and undo") {
    auto s = open_session(fresh_dir("control_edit") + "/workspace");
    ok(*s, "spec.new", {{"id", "m1"}});
    CHECK(s->open_spec()->spec.version == 0);

    // A dry run checks and shows the diff, and changes nothing.
    json r = ok(*s, "control.create", {{"id", "lr"}, {"kind", "slider"}, {"bind", "param:lr"}, {"dry_run", true}});
    CHECK(r["applied"] == false);
    CHECK(r["diff"].get<std::string>().find("param:lr") != std::string::npos);
    CHECK(s->open_spec()->spec.controls.empty());

    // The first control raises the version to 1 in the same step.
    r = ok(*s, "control.create",
           {{"id", "lr"}, {"kind", "slider"}, {"bind", "param:lr"}, {"title", "Learning rate"}, {"options", {{"log", true}}}});
    CHECK(r["id"] == "lr");
    CHECK(r["applied"] == true);
    spec::Spec now = s->open_spec()->spec;
    CHECK(now.version == spec::kInstrumentVersion);
    REQUIRE(now.controls.size() == 1);
    CHECK(now.controls[0].title == std::optional<std::string>("Learning rate"));
    CHECK(now.controls[0].options.log == std::optional<bool>(true));
    CHECK(*effective_layout(now) == leaf("lr"));
    CHECK(err(*s, "control.create", {{"id", "lr"}, {"kind", "slider"}, {"bind", "param:lr"}}) == "control.exists");

    // A second control joins the row.
    ok(*s, "control.create", {{"id", "pause"}, {"kind", "button"}, {"bind", "run:pause"}});
    CHECK(*s->open_spec()->spec.layout == split(H, {leaf("lr"), leaf("pause")}));

    // A patch merges its keys; null clears one.
    r = ok(*s, "control.patch", {{"id", "lr"}, {"patch", {{"title", nullptr}, {"options", {{"min", 1e-4}, {"max", 0.1}, {"log", nullptr}}}}}});
    CHECK(r["control"] == json{{"id", "lr"}, {"kind", "slider"}, {"title", nullptr}, {"bind", "param:lr"},
                               {"options", {{"min", 1e-4}, {"max", 0.1}}}});
    CHECK(err(*s, "control.patch", {{"id", "lr"}, {"patch", {{"bind", "action:save"}}}}) == "spec.control_mismatch");

    // A remove takes the control out of the layout; undo puts it back in its place.
    ok(*s, "control.remove", {{"id", "lr"}});
    CHECK(s->open_spec()->spec.find_control("lr") == nullptr);
    CHECK(*s->open_spec()->spec.layout == leaf("pause"));
    ok(*s, "history.undo");
    CHECK(s->open_spec()->spec.find_control("lr") != nullptr);
    CHECK(*s->open_spec()->spec.layout == split(H, {leaf("lr"), leaf("pause")}));
    CHECK(err(*s, "control.remove", {{"id", "momentum"}}) == "control.not_found");
}

TEST_CASE("control commands: a control placed below a scene panel, and the version raise is one undo step") {
    auto s = open_session(fresh_dir("control_edit_place") + "/workspace");
    ok(*s, "spec.new", {{"id", "m1"}});
    ok(*s, "panel.create", {{"id", "world"}, {"primitive", "scene"}, {"bind", {{"scene", "live:model"}}}});
    const spec::Spec before = s->open_spec()->spec;
    ok(*s, "control.create",
       {{"id", "lr"}, {"kind", "slider"}, {"bind", "param:lr"}, {"place", {{"target", "world"}, {"where", "below"}, {"ratio", 0.15}}}});
    const spec::Spec after = s->open_spec()->spec;
    LayoutNode want = split(V, {leaf("world"), leaf("lr")});
    want.ratio = 0.85;
    CHECK(*after.layout == want);
    // A control is a target of place too.
    ok(*s, "control.create",
       {{"id", "pause"}, {"kind", "button"}, {"bind", "run:pause"}, {"place", {{"target", "lr"}, {"where", "beside"}}}});
    ok(*s, "history.undo");
    ok(*s, "history.undo");
    CHECK(s->open_spec()->spec == before);
}
