// The builder pane's state and gestures without a window, over a real
// builder Session whose workspace imported a Parquet table and a safetensors
// tensor through the pane's file drop: the dashboard of dashboard.spec.toml
// built gesture by gesture, each one undo step of commands from the builder
// origin, journaled as such; a binding, a primitive and a drop that do not
// fit refused before anything is sent; drops on slots, edges, centres and
// the whole layout; the spec text both ways, an invalid draft included;
// undo, redo and save; a busy session; renames.
#include "builder/builder_pane.h"

#include "spec_test_workspace.h"

#include <caliper_builder/spec/format.h>

#include <doctest/doctest.h>

#include <fstream>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

using namespace caliper_host;
using namespace spec_test;
namespace cb = caliper_builder;
namespace spec = caliper_builder::spec;
using nlohmann::json;

namespace {

// A port that can be marked busy, as AppSession::worker_busy() does.
struct BusyPort : SessionBuilderPort {
    using SessionBuilderPort::SessionBuilderPort;
    bool busy = false;
    bool ready() const override { return !busy; }
};

struct Rig {
    std::string root, ws, data;
    std::unique_ptr<cb::Session> s;
    BusyPort* port = nullptr;
    std::unique_ptr<BuilderPane> pane;

    size_t steps() const { return s->registry().history_state(200).undo.size(); }
    std::vector<json> journal() const {
        std::vector<json> out;
        std::ifstream in(ws + "/journal/commands.jsonl");
        std::string line;
        while (std::getline(in, line))
            if (!line.empty()) out.push_back(json::parse(line));
        return out;
    }
};

// A workspace with an empty spec of the fixture's applet, and the two files
// dropped on the pane and imported.
Rig rig(const std::string& name) {
    Rig r;
    r.root = fresh_dir(name);
    r.ws = r.root + "/ws";
    r.data = r.root + "/data";
    write_parquet(r.data + "/run42_metrics.parquet",
                  "SELECT range AS step, 0.1 + 2.0 * exp(-range / 50.0) AS loss, "
                  "0.1 + 0.8 * (1 - exp(-range / 40.0)) AS accuracy FROM range(200)");
    std::vector<float> w(2 * 3 * 8 * 8, 0.0f);
    for (int l = 0; l < 2; ++l)
        for (int h = 0; h < 3; ++h)
            for (int q = 0; q < 8; ++q) w[((l * 3 + h) * 8 + q) * 8 + q] = 1.0f + (float)h;
    write_safetensors_f32(r.data + "/attn.safetensors", "weights", {2, 3, 8, 8}, w);
    cb::SessionOptions options;
    options.registry.check_results = true;
    auto opened = cb::Session::open(r.ws, options);
    REQUIRE_MESSAGE(opened.error.ok(), opened.error.message);
    r.s = std::move(opened.session);
    const spec::Spec d = dashboard_fixture();
    cb::CommandInvocation inv;
    inv.name = "spec.new";
    inv.args = {{"id", d.applet.id}, {"title", *d.applet.title}, {"description", *d.applet.description}};
    inv.origin = cb::CommandOrigin::builder;
    REQUIRE(r.s->invoke(inv).ok);
    auto port = std::make_unique<BusyPort>(*r.s);
    r.port = port.get();
    BuilderPane::Options po;
    po.group_prefix = "t";
    r.pane = std::make_unique<BuilderPane>(std::move(port), po);
    r.pane->set_build_mode(true);
    REQUIRE(r.pane->drop_files({r.data + "/run42_metrics.parquet", r.data + "/attn.safetensors"}));
    CHECK(r.pane->importing());
    r.s->queue().drain();   // the app's frame loop, at once
    r.pane->update();
    REQUIRE_FALSE(r.pane->importing());
    return r;
}

DropHit at(DropZone zone, const std::string& panel, const std::string& where, int slot = -1) {
    DropHit h;
    h.zone = zone;
    h.panel = panel;
    h.where = where;
    h.slot = slot;
    return h;
}

DragPayload primitive(spec::Primitive p) {
    DragPayload d;
    d.kind = DragPayload::Kind::primitive;
    d.primitive = p;
    return d;
}

DragPayload source(const std::string& key, const std::string& column = {}) {
    DragPayload d;
    d.kind = column.empty() ? DragPayload::Kind::source : DragPayload::Kind::column;
    d.source = key;
    d.column = column;
    return d;
}

DragPayload panel(const std::string& id) {
    DragPayload d;
    d.kind = DragPayload::Kind::panel;
    d.panel = id;
    return d;
}

// The commands one gesture sent: those after `before` in sent().
std::vector<cb::CommandInvocation> since(const BuilderPane& p, size_t before) {
    const auto& all = p.sent();
    return std::vector<cb::CommandInvocation>(all.begin() + (std::ptrdiff_t)before, all.end());
}

// One gesture: `fn` returns true, adds exactly one undo step, and sends at
// least one command, all from the builder origin under one group.
template <class F>
std::vector<cb::CommandInvocation> gesture(Rig& r, const char* what, F fn) {
    INFO(what);
    const size_t steps = r.steps(), sent = r.pane->sent().size();
    const bool ok = fn();
    INFO(r.pane->error().code << ": " << r.pane->error().message);
    REQUIRE(ok);
    CHECK(r.steps() == steps + 1);
    const auto cmds = since(*r.pane, sent);
    REQUIRE_FALSE(cmds.empty());
    for (const auto& c : cmds) {
        CHECK(c.origin == cb::CommandOrigin::builder);
        CHECK(c.group == cmds.front().group);
        CHECK_FALSE(c.group.empty());
    }
    return cmds;
}

// Builds the fixture's dashboard by hand, as caliper-builder-shot does
// through the pane's input.
void build_dashboard(Rig& r) {
    BuilderPane& p = *r.pane;
    // The curves.
    p.select("table:run42_metrics");
    const size_t sent = p.sent().size();
    REQUIRE(p.set_source_id("metrics"));   // kept until the source is added
    CHECK(p.sent().size() == sent);
    auto c = gesture(r, "series", [&] { return p.add_primitive(spec::Primitive::series); });
    REQUIRE(c.size() == 2);
    CHECK(c[0].name == "source.add");
    CHECK(c[0].args == json{{"id", "metrics"}, {"catalog", "table:run42_metrics"}});
    CHECK(c[1].name == "panel.create");
    CHECK(p.focused_panel() == "series");
    CHECK(p.selected() == "spec:metrics");
    c = gesture(r, "rename", [&] { return p.set_panel_id("curves"); });
    CHECK(c.front().name == "spec.set_text");
    CHECK(p.focused_panel() == "curves");
    c = gesture(r, "title typed", [&] {
        p.begin_gesture("title");
        bool ok = p.set_title("L") && p.set_title("Loss") && p.set_title("Loss and accuracy");
        p.end_gesture();
        return ok;
    });
    CHECK(c.size() == 3);
    for (const auto& x : c) {
        CHECK(x.name == "panel.patch");
        CHECK(x.merge_key == c.front().merge_key);
        CHECK_FALSE(x.merge_key.empty());
    }
    gesture(r, "x label", [&] {
        p.begin_gesture("x_label");
        bool ok = p.set_option("x_label", "S") && p.set_option("x_label", "Step");
        p.end_gesture();
        return ok;
    });
    gesture(r, "y log", [&] { return p.set_option("y_log", true); });

    // The attention: its source with three transforms, the heatmap beside the curves.
    p.select("tensor:attn_weights");
    REQUIRE(p.set_source_id("attn"));
    c = gesture(r, "slice", [&] { return p.add_transform(spec::TransformOp::slice); });
    CHECK(c.front().name == "source.add");
    CHECK(p.selected() == "spec:attn");
    gesture(r, "reduce", [&] { return p.add_transform(spec::TransformOp::reduce); });
    gesture(r, "normalise", [&] { return p.add_transform(spec::TransformOp::normalise); });
    const std::vector<PaneShape> shapes = p.transform_shapes();
    REQUIRE(shapes.size() == 4);
    CHECK(pane_shape_text(shapes[0]) == "f32 2\xc3\x97" "3\xc3\x97" "8\xc3\x97" "8");
    CHECK(pane_shape_text(shapes[1]) == "f32 3\xc3\x97" "8\xc3\x97" "8");
    CHECK(pane_shape_text(shapes[2]) == "f32 8\xc3\x97" "8");
    CHECK(pane_shape_text(shapes[3]) == "f32 8\xc3\x97" "8");
    c = gesture(r, "heatmap dropped beside", [&] {
        return p.drop(primitive(spec::Primitive::heatmap), at(DropZone::edge, "curves", "beside"));
    });
    REQUIRE(c.size() == 1);
    CHECK(c[0].args["place"] == json{{"target", "curves"}, {"where", "beside"}});
    CHECK(p.focused_panel() == "heatmap");
    gesture(r, "rename", [&] { return p.set_panel_id("attention"); });
    gesture(r, "title", [&] { return p.set_title("Mean attention of layer 0"); });
    gesture(r, "colormap", [&] { return p.set_option("colormap", "magma"); });
    gesture(r, "range", [&] { return p.set_option("range", json::array({0.0, 1.0})); });
    gesture(r, "x label", [&] { return p.set_option("x_label", "Key"); });
    gesture(r, "y label", [&] { return p.set_option("y_label", "Query"); });

    // The table under everything.
    p.select("spec:metrics");
    c = gesture(r, "table below everything", [&] {
        return p.drop(primitive(spec::Primitive::table), at(DropZone::root, "", "below"));
    });
    REQUIRE(c.size() == 2);
    CHECK(c[0].name == "panel.create");
    CHECK(c[1].name == "layout.set");
    gesture(r, "rename", [&] { return p.set_panel_id("rows"); });
    gesture(r, "columns", [&] { return p.set_option("columns", json::array({"step", "loss"})); });
    c = gesture(r, "divider dragged", [&] {
        p.begin_gesture("ratio.");
        bool ok = p.set_ratio("", 0.55) && p.set_ratio("", 0.6) && p.set_ratio("", 0.65);
        p.end_gesture();
        return ok;
    });
    CHECK(c.size() == 3);
}

}  // namespace

TEST_CASE("builder pane: dropped files are imported from the builder origin and selected") {
    Rig r = rig("pane-drop");
    const BuilderPane& p = *r.pane;
    CHECK(p.fresh() == std::vector<std::string>{"table:run42_metrics", "tensor:attn_weights"});
    CHECK(p.selected() == "table:run42_metrics");
    CHECK(p.focused_source() == "table:run42_metrics");
    REQUIRE(p.sent().size() == 1);
    CHECK(p.sent()[0].name == "data.import");
    CHECK(p.sent()[0].origin == cb::CommandOrigin::builder);
    const PaneSource* t = p.find_source("table:run42_metrics");
    REQUIRE(t);
    CHECK(t->group == PaneSource::Group::table);
    CHECK(t->shape_text == "200 \xc3\x97 3");
    const PaneSource* x = p.find_source("tensor:attn_weights");
    REQUIRE(x);
    CHECK(x->shape_text == "f32 2\xc3\x97" "3\xc3\x97" "8\xc3\x97" "8");
    // The import is one step from the builder origin in the journal.
    bool found = false;
    for (const auto& e : r.journal())
        if (e["name"] == "data.import") {
            found = true;
            CHECK(e["origin"] == "builder");
        }
    CHECK(found);
    // The filter.
    r.pane->set_filter("attn");
    REQUIRE(p.visible_sources().size() == 1);
    CHECK(p.visible_sources()[0]->key == "tensor:attn_weights");
    r.pane->set_filter("accur");   // a column's name
    CHECK(p.visible_sources().size() == 1);

    // A file that cannot be read: the import answers it, nothing is selected anew.
    REQUIRE(r.pane->drop_files({r.data + "/missing.parquet"}));
    CHECK_FALSE(r.pane->drop_files({r.data + "/run42_metrics.parquet"}));   // one import at a time
    r.s->queue().drain();
    r.pane->update();
    CHECK(r.pane->error().code.rfind("loader.", 0) == 0);
    CHECK(p.fresh().empty());
}

TEST_CASE("builder pane: the dashboard by hand, one undo step per gesture, equal to the fixture") {
    Rig r = rig("pane-dashboard");
    build_dashboard(r);
    const spec::Spec d = dashboard_fixture();
    const auto& open = r.pane->open_spec();
    REQUIRE(open);
    CHECK(open->spec == d);
    CHECK(r.pane->text() == spec::to_toml(d));
    // Every command the pane sent, and every line of the journal, is the builder's.
    for (const auto& c : r.pane->sent()) CHECK(c.origin == cb::CommandOrigin::builder);
    for (const auto& e : r.journal()) CHECK(e["origin"] == "builder");

    // Two undos take back the divider and the columns, two redos bring them back.
    const size_t steps = r.steps();
    REQUIRE(r.pane->undo());
    REQUIRE(r.pane->undo());
    CHECK(r.steps() == steps - 2);
    CHECK(r.pane->open_spec()->spec.layout->ratio == std::nullopt);
    CHECK_FALSE(r.pane->open_spec()->spec.find_panel("rows")->options.columns.has_value());
    REQUIRE(r.pane->redo());
    REQUIRE(r.pane->redo());
    CHECK(r.pane->open_spec()->spec == d);
    CHECK_FALSE(r.pane->redo());
    CHECK(r.pane->error().code == "history.nothing_to_redo");

    // Saved as the spec applet, whose file is the fixture.
    REQUIRE(r.pane->save());
    REQUIRE(r.pane->open_spec()->path);
    auto saved = spec::parse_toml(read_text(r.ws + "/" + *r.pane->open_spec()->path));
    REQUIRE(saved.ok());
    CHECK(saved.spec == d);
    CHECK_FALSE(r.pane->open_spec()->dirty);
    // A second save overwrites its own applet.
    REQUIRE(r.pane->save());
}

TEST_CASE("builder pane: what does not fit is refused before anything is sent") {
    Rig r = rig("pane-refuse");
    build_dashboard(r);
    BuilderPane& p = *r.pane;
    const spec::Spec before = p.open_spec()->spec;
    const size_t steps = r.steps(), sent = p.sent().size();
    // A column in the heatmap's values: faint in the choices, with the validator's message.
    p.focus_panel("attention");
    const auto& choices = p.binding_choices("values", 0);
    bool saw_attn = false, saw_loss = false;
    std::string why;
    for (const auto& c : choices) {
        if (c.binding == "attn") saw_attn = c.fits;
        if (c.binding == "metrics.loss") {
            saw_loss = true;
            CHECK_FALSE(c.fits);
            why = c.message;
        }
    }
    CHECK(saw_attn);
    CHECK(saw_loss);
    CHECK_FALSE(why.empty());
    CHECK_FALSE(p.set_binding("values", 0, "metrics.loss"));
    CHECK(p.error().message == why);
    // A required slot cannot be unbound.
    CHECK_FALSE(p.remove_binding("values", 0));
    // A heatmap of the metrics table.
    p.select("spec:metrics");
    const PanePlan& plan = p.primitive_plan(spec::Primitive::heatmap);
    CHECK(plan.refusal);
    CHECK(plan.refusal.code.rfind("spec.", 0) == 0);
    CHECK_FALSE(p.add_primitive(spec::Primitive::heatmap));
    // The raw attention tensor dropped on the series' lines: rank 4.
    CHECK_FALSE(p.drop(source("tensor:attn_weights"), at(DropZone::slot, "curves", "", 0)));
    CHECK(p.error().code == "spec.slot_mismatch");
    // A panel dropped on itself.
    CHECK_FALSE(p.drop(panel("curves"), at(DropZone::edge, "curves", "below")));
    CHECK(p.error().code == "layout.same_panel");
    // An id that is not one, or that is taken.
    p.focus_panel("curves");
    CHECK_FALSE(p.set_panel_id("Curves"));
    CHECK(p.error().code == "spec.bad_id");
    CHECK_FALSE(p.set_panel_id("rows"));
    CHECK(p.error().code == "spec.duplicate_id");
    // A source in use.
    p.focus_source("spec:metrics");
    CHECK_FALSE(p.remove_focused());
    CHECK(p.error().code == "source.in_use");
    // Nothing reached the session.
    CHECK(p.sent().size() == sent);
    CHECK(r.steps() == steps);
    CHECK(p.open_spec()->spec == before);
}

TEST_CASE("builder pane: drops on a slot, on edges, in a centre and at the whole layout's edges") {
    Rig r = rig("pane-drops");
    BuilderPane& p = *r.pane;
    // A column alone as a series over the first column.
    p.select("table:run42_metrics", "loss");
    gesture(r, "column added", [&] { return p.add_selected(); });
    const spec::Panel* s = p.open_spec()->spec.find_panel("series");
    REQUIRE(s);
    CHECK(s->bind.at("y").bindings == std::vector<std::string>{"run42_metrics.loss"});
    CHECK(s->bind.at("x").bindings == std::vector<std::string>{"run42_metrics.step"});
    // Another column dropped on its y slot: a second line.
    const std::vector<PaneSlot> slots = p.drop_slots("series");
    REQUIRE(slots.size() == 2);
    CHECK(std::string(slots[0].name) == "y");
    auto c = gesture(r, "column on a slot", [&] {
        return p.drop(source("spec:run42_metrics", "accuracy"), at(DropZone::slot, "series", "", 0));
    });
    REQUIRE(c.size() == 1);
    CHECK(c[0].args["patch"]["bind"]["y"] == json::array({"run42_metrics.loss", "run42_metrics.accuracy"}));
    // A source dropped on a panel's edge: a new panel of its default primitive there.
    c = gesture(r, "table above", [&] { return p.drop(source("spec:run42_metrics"), at(DropZone::edge, "series", "above")); });
    CHECK(c.back().args["primitive"] == "series");
    CHECK(c.back().args["place"]["where"] == "above");
    const std::string second = p.focused_panel();
    CHECK(second == "series-2");
    // A panel moved by its strip: below another, then to its left, then
    // tabbed with it, then under the whole layout.
    c = gesture(r, "moved below", [&] { return p.drop(panel(second), at(DropZone::edge, "series", "below")); });
    CHECK(c[0].name == "layout.place");
    c = gesture(r, "moved left", [&] { return p.drop(panel(second), at(DropZone::edge, "series", "left")); });
    CHECK(c[0].name == "layout.set");
    auto lay = p.open_spec()->spec.layout;
    REQUIRE(lay);
    CHECK(pane_layout_node(*lay, "0")->panel == second);
    c = gesture(r, "tabbed", [&] { return p.drop(panel(second), at(DropZone::centre, "series", "tabbed")); });
    CHECK(p.open_spec()->spec.layout->kind == spec::LayoutNode::Kind::tabs);
    c = gesture(r, "a tensor at the whole layout's left", [&] {
        p.select("tensor:attn_weights");
        return p.drop(source("tensor:attn_weights"), at(DropZone::root, "", "left"));
    });
    lay = p.open_spec()->spec.layout;
    CHECK(lay->kind == spec::LayoutNode::Kind::split);
    CHECK(pane_layout_node(*lay, "0")->panel == p.focused_panel());
    CHECK(p.open_spec()->spec.find_panel(p.focused_panel())->primitive == spec::Primitive::histogram);
    // The panel removed.
    const std::string gone = p.focused_panel();
    gesture(r, "removed", [&] { return p.remove_focused(); });
    CHECK_FALSE(p.open_spec()->spec.find_panel(gone));
    CHECK(p.focused_panel().empty());
}

TEST_CASE("builder pane: the spec text follows the canvas, and the canvas the text") {
    Rig r = rig("pane-text");
    BuilderPane& p = *r.pane;
    p.select("table:run42_metrics");
    REQUIRE(p.add_primitive(spec::Primitive::series));
    REQUIRE(p.set_title("Curves"));
    // A canvas edit is in the text at once.
    CHECK(p.text().find("title = \"Curves\"") != std::string::npos);
    CHECK(p.text() == p.open_spec()->text);

    // An edit of the text, applied while it validates: one step for the
    // whole edit.
    p.set_editing(true);
    const std::string base = p.text();
    std::string draft = base;
    draft.replace(draft.find("title = \"Curves\""), 16, "title = \"Loss\"");
    const size_t steps = r.steps();
    p.begin_gesture("text");
    REQUIRE(p.set_draft(draft));
    std::string draft2 = draft;
    draft2.replace(draft2.find("title = \"Loss\""), 14, "title = \"Loss curves\"");
    REQUIRE(p.set_draft(draft2));
    p.end_gesture();
    CHECK(r.steps() == steps + 1);
    CHECK(*p.open_spec()->spec.find_panel("series")->title == "Loss curves");
    CHECK(p.text() == draft2);   // as typed
    CHECK(p.sent().back().name == "spec.set_text");
    CHECK(p.sent().back().origin == cb::CommandOrigin::builder);

    // An edit that does not validate: its diagnostic at its line, nothing
    // sent, the canvas keeps the last valid spec, and every other edit waits.
    const size_t sent = p.sent().size();
    std::string bad = draft2;
    const size_t at_applet = bad.find("[applet]\n") + 9;
    bad.insert(at_applet, "colour = 3\n");
    CHECK_FALSE(p.set_draft(bad));
    CHECK(p.draft_pending());
    REQUIRE_FALSE(p.diagnostics().empty());
    CHECK(p.diagnostics()[0].code == "spec.unknown_key");
    int line = 1;
    for (size_t i = 0; i < at_applet; ++i)
        if (bad[i] == '\n') ++line;
    CHECK(p.diagnostics()[0].line == line);
    CHECK(p.sent().size() == sent);
    CHECK(*p.open_spec()->spec.find_panel("series")->title == "Loss curves");
    p.focus_panel("series");
    CHECK_FALSE(p.set_title("Other"));
    CHECK(p.error().code == "builder.draft");
    CHECK(p.sent().size() == sent);
    // Fixed, it applies.
    REQUIRE(p.set_draft(draft2 + "\n"));
    CHECK_FALSE(p.draft_pending());
    CHECK(p.diagnostics().empty());
    // A canvas edit while the draft is applied replaces it with the new text.
    REQUIRE(p.set_title("Again"));
    CHECK(p.text() == p.open_spec()->text);
    // Undo through the history puts the text back too.
    REQUIRE(p.undo());
    CHECK(*p.open_spec()->spec.find_panel("series")->title == "Loss curves");
    CHECK(p.text() == p.open_spec()->text);
    p.set_editing(false);
    CHECK(p.text() == p.open_spec()->text);
}

TEST_CASE("builder pane: renames reach every reference, and the spec's own title") {
    Rig r = rig("pane-rename");
    BuilderPane& p = *r.pane;
    build_dashboard(r);
    p.focus_source("spec:metrics");
    gesture(r, "source renamed", [&] { return p.set_source_id("m"); });
    const spec::Spec& s = p.open_spec()->spec;
    CHECK(s.find_source("m"));
    CHECK(s.find_panel("rows")->bind.at("rows").bindings == std::vector<std::string>{"m"});
    CHECK(p.focused_source() == "spec:m");
    p.clear_focus();
    gesture(r, "spec title", [&] { return p.set_spec_title("Run 42, again"); });
    CHECK(*p.open_spec()->spec.applet.title == "Run 42, again");
}

TEST_CASE("builder pane: a busy session sends nothing") {
    Rig r = rig("pane-busy");
    BuilderPane& p = *r.pane;
    p.select("table:run42_metrics");
    r.port->busy = true;
    const size_t sent = p.sent().size();
    CHECK_FALSE(p.add_primitive(spec::Primitive::series));
    CHECK(p.error().code == "builder.busy");
    CHECK_FALSE(p.undo());
    CHECK(p.sent().size() == sent);
    r.port->busy = false;
    CHECK(p.add_primitive(spec::Primitive::series));
}
