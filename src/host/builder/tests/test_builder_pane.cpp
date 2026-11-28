// The builder pane's state and gestures without a window, over a real
// builder Session whose workspace imported a Parquet table and a safetensors
// tensor through the pane's file drop: the dashboard of dashboard.spec.toml
// built gesture by gesture, each one undo step of commands from the builder
// origin, journaled as such; a binding, a primitive and a drop that do not
// fit refused before anything is sent; drops on slots, edges, centres and
// the whole layout; the spec text both ways, an invalid draft included;
// undo, redo and save; a busy session; renames.
#include "builder/builder_pane.h"
#include "builder/widget_agent.h"
#include "builder/widget_surface.h"

#include "spec_test_workspace.h"

#include "catalog_providers/metrics_catalog.h"
#include "metrics_store.h"

#include <caliper_builder/spec/format.h>

#include <doctest/doctest.h>

#include <algorithm>
#include <chrono>
#include <filesystem>
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

// A workspace with an empty spec of the fixture's instrument, and the two files
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
    inv.args = {{"id", d.instrument.id}, {"title", *d.instrument.title}, {"description", *d.instrument.description}};
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
    CHECK(p.sent()[0].origin == cb::CommandOrigin::ui);   // a file the person dropped
    const PaneSource* t = p.find_source("table:run42_metrics");
    REQUIRE(t);
    CHECK(t->group == PaneSource::Group::table);
    CHECK(t->shape_text == "200 \xc3\x97 3");
    const PaneSource* x = p.find_source("tensor:attn_weights");
    REQUIRE(x);
    CHECK(x->shape_text == "f32 2\xc3\x97" "3\xc3\x97" "8\xc3\x97" "8");
    // The import is one step from the ui origin in the journal.
    bool found = false;
    for (const auto& e : r.journal())
        if (e["name"] == "data.import") {
            found = true;
            CHECK(e["origin"] == "ui");
        }
    CHECK(found);
    // The filter.
    r.pane->set_filter("attn");
    REQUIRE(p.visible_sources().size() == 1);
    CHECK(p.visible_sources()[0]->key == "tensor:attn_weights");
    r.pane->set_filter("accur");   // a column's name
    CHECK(p.visible_sources().size() == 1);

    // A file that cannot be read and a pickle: the import refuses each with
    // its code and reason, which the pane keeps for the palette, and selects
    // nothing anew.
    { std::ofstream(r.data + "/model.pkl") << "not a model"; }
    REQUIRE(r.pane->drop_files({r.data + "/missing.parquet", r.data + "/model.pkl"}));
    CHECK_FALSE(r.pane->drop_files({r.data + "/run42_metrics.parquet"}));   // one import at a time
    r.s->queue().drain();
    r.pane->update();
    CHECK(r.pane->error().code.rfind("loader.", 0) == 0);
    CHECK(p.fresh().empty());
    REQUIRE(p.refused().size() == 2);
    bool pickle = false;
    for (const auto& f : p.refused()) {
        CHECK_FALSE(f.message.empty());
        if (f.path.find("model.pkl") != std::string::npos) pickle = f.code == "loader.pickle_refused";
    }
    CHECK(pickle);
}

// Metrics in Sources, grouped by run: a real MetricsStore with more metrics
// than one catalog.list returns, through the app's MetricsCatalog.
TEST_CASE("builder pane: metrics grouped by run, newest first, the last few until Show all, the filter across runs") {
    Rig r = rig("pane-metric-runs");
    MetricsStore store;
    REQUIRE(store.open(":memory:"));
    std::vector<std::uint64_t> runs;
    for (int i = 0; i < 1100; ++i) {
        const bool base = i == 6;
        const std::uint64_t run = store.begin_run("sweep", base ? "baseline" : "r" + std::to_string(i));
        store.scalar(run, "loss", 0, 1.0);
        if (base) store.scalar(run, "accuracy", 0, 0.5);
        runs.push_back(run);
    }
    MetricsCatalog::Options mo;
    mo.thread = false;
    auto metrics = std::make_shared<MetricsCatalog>(store, mo);
    metrics->refresh_now();
    REQUIRE(r.s->catalog().add_provider("metrics", metrics).ok());
    BuilderPane& p = *r.pane;
    p.update();

    // Every metric is listed, and the tables and tensors are still there,
    // though one list of metrics is cut at 1000.
    size_t metric_rows = 0;
    for (const auto& row : p.sources()) metric_rows += row.group == PaneSource::Group::metric;
    CHECK(metric_rows == 1101);
    CHECK(p.find_source("table:run42_metrics"));
    CHECK(p.find_source("tensor:attn_weights"));
    const PaneSource* base = p.find_source("metric:" + std::to_string(runs[6]) + "/accuracy");
    REQUIRE(base);
    CHECK(base->run == (std::int64_t)runs[6]);
    CHECK(base->run_name == "baseline");
    CHECK(base->tag == "accuracy");

    auto run_ids = [&] {
        std::vector<std::int64_t> ids;
        for (const auto& it : p.list_items())
            if (it.kind == PaneListItem::Kind::run) ids.push_back(it.run);
        return ids;
    };
    // The newest five runs, closed, then Show all with every run counted.
    const std::vector<std::int64_t> newest = {(std::int64_t)runs[1099], (std::int64_t)runs[1098], (std::int64_t)runs[1097],
                                              (std::int64_t)runs[1096], (std::int64_t)runs[1095]};
    CHECK(run_ids() == newest);
    for (const auto& it : p.list_items()) CHECK_FALSE((it.source && it.source->group == PaneSource::Group::metric));
    REQUIRE((p.list_items().back().kind == PaneListItem::Kind::more));
    CHECK(p.list_items().back().name == "Show all");
    CHECK(p.list_items().back().metrics == 1100);
    // The tables and tensors are listed as before. The pieces of the built-in
    // library are listed too, and the count leaves them out, so that it does
    // not depend on what the library holds.
    size_t listed = 0;
    for (const PaneSource* s : p.visible_sources()) listed += s->group != PaneSource::Group::piece;
    CHECK(listed == 2);

    // A run opens to its metrics, by tag, under it.
    p.toggle_expanded("run:" + std::to_string(runs[1098]));
    const auto& items = p.list_items();
    auto at = std::find_if(items.begin(), items.end(), [&](const PaneListItem& it) {
        return it.kind == PaneListItem::Kind::run && it.run == (std::int64_t)runs[1098];
    });
    REQUIRE(at != items.end());
    CHECK(at->open);
    CHECK(at->metrics == 1);
    REQUIRE((at + 1) != items.end());
    CHECK(((at + 1)->kind == PaneListItem::Kind::source));
    CHECK((at + 1)->nested);
    CHECK((at + 1)->source->key == "metric:" + std::to_string(runs[1098]) + "/loss");

    // Show all lists every run, newest first, and Show fewer goes back.
    p.set_show_all_runs(true);
    const auto all = run_ids();
    CHECK(all.size() == 1100);
    CHECK(all.front() == (std::int64_t)runs[1099]);
    CHECK(all.back() == (std::int64_t)runs[0]);
    CHECK(std::is_sorted(all.rbegin(), all.rend()));
    CHECK(p.list_items().back().name == "Show fewer");
    p.set_show_all_runs(false);
    CHECK(run_ids().size() == 5);

    // The filter searches every run: an old run by its name, open with all
    // its metrics; a tag in every run; and no Show all while it filters.
    p.set_filter("baseline");
    CHECK(run_ids() == std::vector<std::int64_t>{(std::int64_t)runs[6]});
    REQUIRE(p.visible_sources().size() == 2);
    CHECK(p.visible_sources()[0]->tag == "accuracy");
    CHECK(p.visible_sources()[1]->tag == "loss");
    p.set_filter("accuracy");
    CHECK(run_ids() == std::vector<std::int64_t>{(std::int64_t)runs[6]});
    // The table's column matches too.
    CHECK(p.visible_sources().size() == 2);
    p.set_filter("loss");
    CHECK(run_ids().size() == 1100);
    CHECK((p.list_items().back().kind != PaneListItem::Kind::more));
    p.set_filter("");
    CHECK(run_ids().size() == 5);

    // A metric selected inside its run is added as a panel like any row.
    p.select("metric:" + std::to_string(runs[1098]) + "/loss");
    REQUIRE(p.add_selected());
    r.s->queue().drain();
    p.update();
    bool bound = false;
    for (const auto& src : p.open_spec()->spec.sources)
        bound |= src.catalog && *src.catalog == "metric:" + std::to_string(runs[1098]) + "/loss";
    CHECK(bound);
    CHECK(p.open_spec()->spec.panels.size() == 1);

    r.s->catalog().remove_provider("metrics");
}

namespace {
// The panel ids of a layout, and whether every split and tab group in it has
// what it needs (two children, one tab): a removed panel leaves no gap.
void layout_ids(const spec::LayoutNode& n, std::vector<std::string>& ids, bool& whole) {
    if (n.kind == spec::LayoutNode::Kind::panel) {
        ids.push_back(n.panel);
        return;
    }
    if (n.children.size() < (n.kind == spec::LayoutNode::Kind::split ? 2u : 1u)) whole = false;
    for (const auto& c : n.children) layout_ids(c, ids, whole);
}
std::vector<std::string> laid_out(const BuilderPane& p, bool* whole) {
    std::vector<std::string> ids;
    *whole = true;
    if (p.open_spec() && p.open_spec()->spec.layout) layout_ids(*p.open_spec()->spec.layout, ids, *whole);
    std::sort(ids.begin(), ids.end());
    return ids;
}
}  // namespace

// The page's × on a panel, its menu's Remove panel and ⌫ all come to
// remove_panel(): panel.remove as one undo step, from the ui origin on the
// dashboard page and the builder origin in build mode, which closes the gap
// in tabs and in a split, and ⌘Z brings the panel back where it was.
TEST_CASE("builder pane: remove_panel closes the gap in tabs and in a split, one undo step") {
    Rig r = rig("pane-remove-panel");
    BuilderPane& p = *r.pane;
    std::string text = read_text(std::string(CALIPER_SPEC_FIXTURES) + "/dashboard.spec.toml");
    const std::string split = R"(children = [{ split = "horizontal", children = ["curves", "attention"] }, "rows"])";
    const size_t at = text.find(split);
    REQUIRE(at != std::string::npos);
    text.replace(at, split.size(), R"(children = [{ tabs = ["curves", "attention"] }, "rows"])");
    cb::CommandInvocation inv;
    inv.name = "spec.set_text";
    inv.args = {{"text", text}};
    inv.origin = cb::CommandOrigin::builder;
    REQUIRE_MESSAGE(r.s->invoke(inv).ok, "the fixture with tabs");
    p.set_build_mode(false);
    p.update();
    const spec::Spec before = p.open_spec()->spec;
    bool whole = false;
    REQUIRE(laid_out(p, &whole) == std::vector<std::string>{"attention", "curves", "rows"});

    // A tab, from the dashboard page: the ui origin, one step.
    const size_t steps = r.steps();
    REQUIRE(p.remove_panel("attention"));
    CHECK_FALSE(p.open_spec()->spec.find_panel("attention"));
    CHECK(laid_out(p, &whole) == std::vector<std::string>{"curves", "rows"});
    CHECK(whole);
    CHECK(r.steps() == steps + 1);
    CHECK(r.journal().back()["name"] == "panel.remove");
    CHECK(r.journal().back()["origin"] == "ui");
    REQUIRE(p.undo());
    CHECK(p.open_spec()->spec == before);

    // A panel of the split: the tabs take its place.
    REQUIRE(p.remove_panel("rows"));
    CHECK(laid_out(p, &whole) == std::vector<std::string>{"attention", "curves"});
    CHECK(whole);
    REQUIRE(p.undo());
    CHECK(p.open_spec()->spec == before);

    // In build mode the same removal is the pane's gesture, from the builder
    // origin; a panel that is not there is refused before anything is sent.
    p.set_build_mode(true);
    REQUIRE(p.remove_panel("curves"));
    CHECK(r.journal().back()["origin"] == "builder");
    CHECK_FALSE(p.remove_panel("curves"));
    CHECK(p.error().code == "spec.unknown_panel");
    REQUIRE(p.undo());
    CHECK(p.open_spec()->spec == before);
}

TEST_CASE("builder pane: the dashboard by hand, one undo step per gesture, equal to the fixture") {
    Rig r = rig("pane-dashboard");
    build_dashboard(r);
    const spec::Spec d = dashboard_fixture();
    const auto& open = r.pane->open_spec();
    REQUIRE(open);
    CHECK(open->spec == d);
    CHECK(r.pane->text() == spec::to_toml(d));
    // Every command the pane sent, and every line of the journal, is the
    // builder's, but the import of the files the person dropped, the ui's.
    for (const auto& c : r.pane->sent())
        CHECK(c.origin == (c.name == "data.import" ? cb::CommandOrigin::ui : cb::CommandOrigin::builder));
    for (const auto& e : r.journal()) CHECK(e["origin"] == (e["name"] == "data.import" ? "ui" : "builder"));

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

    // Saved as an instrument, whose file is the fixture.
    REQUIRE(r.pane->save());
    REQUIRE(r.pane->open_spec()->path);
    auto saved = spec::parse_toml(read_text(r.ws + "/" + *r.pane->open_spec()->path));
    REQUIRE(saved.ok());
    CHECK(saved.spec == d);
    CHECK_FALSE(r.pane->open_spec()->dirty);
    // A second save overwrites its own instrument.
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
    const size_t at_instrument = bad.find("[instrument]\n") + 13;
    bad.insert(at_instrument, "colour = 3\n");
    CHECK_FALSE(p.set_draft(bad));
    CHECK(p.draft_pending());
    REQUIRE_FALSE(p.diagnostics().empty());
    CHECK(p.diagnostics()[0].code == "spec.unknown_key");
    int line = 1;
    for (size_t i = 0; i < at_instrument; ++i)
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

// The read-only strip follows each change of the open spec: it brings the
// last line that changed into view, so a panel added at the end of the text
// or a title changed in the middle shows where it changed.
TEST_CASE("builder pane: the spec text strip follows the last changed line") {
    // The helper: the last changed line of the new text, 0-based.
    CHECK(pane_text_follow_line("a\nb\nc\n", "a\nb\nc\n") == -1);
    CHECK(pane_text_follow_line("a\nb\nc\n", "a\nB\nc\n") == 1);
    CHECK(pane_text_follow_line("a\nb\n", "a\nb\nc\nd\n") == 3);
    CHECK(pane_text_follow_line("a\nb\nc\nd\n", "a\nd\n") == 1);   // lines taken out: where they were
    CHECK(pane_text_follow_line("", "a\n") == 0);

    Rig r = rig("pane-follow");
    BuilderPane& p = *r.pane;
    const auto line_of = [&](const std::string& needle) {
        const std::string& t = p.text();
        const size_t at = t.find(needle);
        REQUIRE(at != std::string::npos);
        return (int)std::count(t.begin(), t.begin() + (long)at, '\n');
    };
    const auto last_line = [&] {
        const std::string& t = p.text();
        const size_t end = t.find_last_not_of('\n');
        return (int)std::count(t.begin(), t.begin() + (long)end, '\n');
    };
    // A new panel: its block ends the text.
    p.select("table:run42_metrics");
    REQUIRE(p.add_primitive(spec::Primitive::series));
    CHECK(p.text_follow_line() == last_line());
    // A second panel, then the first one's title, which is above it.
    p.select("spec:run42_metrics");
    REQUIRE(p.add_primitive(spec::Primitive::table));
    CHECK(p.text_follow_line() == last_line());
    p.focus_panel("series");
    REQUIRE(p.set_title("Curves"));
    CHECK(p.text_follow_line() == line_of("title = \"Curves\""));
    CHECK(p.text_follow_line() < last_line());
    // An undo is a change too.
    REQUIRE(p.undo());
    CHECK(p.text_follow_line() >= line_of("[[panel]]\nid = \"series\""));
    CHECK(p.text_follow_line() < line_of("[[panel]]\nid = \"table\""));
}

// A primitive that the selection does not fit says why in the selection's
// own terms, after the validator's words: the attention tensor has four
// dimensions, a heatmap takes two, and Properties slices or reduces it.
TEST_CASE("builder pane: a primitive that does not fit says what it takes") {
    Rig r = rig("pane-fit");
    BuilderPane& p = *r.pane;
    const size_t sent = p.sent().size();
    p.select("tensor:attn_weights");
    std::string why;
    {
        const PanePlan& heat = p.primitive_plan(spec::Primitive::heatmap);
        REQUIRE(heat.refusal);
        CHECK(heat.refusal.code == "spec.missing_slot");
        why = heat.refusal.message;
    }
    CHECK(why.find("attn_weights is f32 2\xc3\x97" "3\xc3\x97" "8\xc3\x97" "8") != std::string::npos);
    CHECK(why.find("a heatmap takes a tensor of two dimensions") != std::string::npos);
    CHECK(why.find("slice or reduce it in Properties") != std::string::npos);
    CHECK_FALSE(p.add_primitive(spec::Primitive::heatmap));
    CHECK(p.error().message == why);
    // A table for a heatmap.
    p.select("table:run42_metrics");
    {
        const PanePlan& t = p.primitive_plan(spec::Primitive::heatmap);
        REQUIRE(t.refusal);
        CHECK(t.refusal.message.find("run42_metrics is a table") != std::string::npos);
        CHECK(t.refusal.message.find("Properties") == std::string::npos);
    }
    CHECK(p.sent().size() == sent);
    // Two slices later, layer 0 and its head 0, the tensor fits.
    p.select("tensor:attn_weights");
    REQUIRE(p.add_transform(spec::TransformOp::slice));
    REQUIRE(p.add_transform(spec::TransformOp::slice));
    CHECK(p.selected() == "spec:attn_weights");
    CHECK_FALSE(p.primitive_plan(spec::Primitive::heatmap).refusal);
    REQUIRE(p.add_primitive(spec::Primitive::heatmap));
}

// A slice's and a reduce's axis in Properties: each axis of the step's
// input with its index range, and with its name where the source carries
// names (a live view's dims; a safetensors tensor has none); a slice's index
// outside its axis's range is set to the nearer end, and the pane says so.
TEST_CASE("builder pane: a transform's axes with their ranges and names, and an index kept in its range") {
    const std::string dot = "\xc2\xb7", dash = "\xe2\x80\x93";
    const auto ax = [&](int a, int last, const std::string& name = {}) {
        return "axis " + std::to_string(a) + (name.empty() ? "" : " " + name) + " " + dot + " 0" + dash +
               std::to_string(last);
    };
    // The helpers.
    CHECK(pane_index_range(4) == "0" + dash + "3");
    CHECK(pane_index_range(1) == "0");
    CHECK(pane_index_range(0).empty());
    CHECK(pane_index_range(-1).empty());
    CHECK(pane_axis_text(1, 4) == ax(1, 3));
    CHECK(pane_axis_text(0, 4, "layer") == ax(0, 3, "layer"));
    CHECK(pane_axis_text(2, -1) == "axis 2");
    bool clamped = true;
    CHECK(pane_clamp_index(2, 4, &clamped) == 2);
    CHECK_FALSE(clamped);
    CHECK(pane_clamp_index(-1, 4, &clamped) == -1);   // from the end
    CHECK_FALSE(clamped);
    CHECK(pane_clamp_index(7, 4, &clamped) == 3);
    CHECK(clamped);
    CHECK(pane_clamp_index(-9, 4, &clamped) == 0);
    CHECK(clamped);
    CHECK(pane_clamp_index(7, -1, &clamped) == 7);   // a size not known
    CHECK_FALSE(clamped);

    Rig r = rig("pane-axes");
    BuilderPane& p = *r.pane;
    const auto index_of = [&](size_t step) {
        const spec::Source* src = p.open_spec()->spec.find_source("attn_weights");
        REQUIRE(src);
        REQUIRE(step < src->transforms.size());
        return src->transforms[step].index.value_or(-100);
    };
    // The tensor is [2, 3, 8, 8], from a safetensors file: no names.
    p.select("tensor:attn_weights");
    REQUIRE(p.add_transform(spec::TransformOp::slice));
    CHECK(p.transform_axes(0) == std::vector<std::string>{ax(0, 1), ax(1, 2), ax(2, 7), ax(3, 7)});
    CHECK(p.transform_index_range(0) == "0" + dash + "1");
    // An index past the axis's end is set to its last index, and said.
    REQUIRE(p.set_slice_index(0, 7));
    CHECK(index_of(0) == 1);
    CHECK(p.transform_note() == "7 is outside 0" + dash + "1");
    CHECK(p.transform_note_step() == 0);
    // One within it is taken as it is, and the note goes.
    REQUIRE(p.set_slice_index(0, 0));
    CHECK(index_of(0) == 0);
    CHECK(p.transform_note().empty());
    // Another axis: its range, and the index kept within it.
    REQUIRE(p.set_transform_axis(0, 1));
    CHECK(p.transform_index_range(0) == "0" + dash + "2");
    REQUIRE(p.set_slice_index(0, 2));
    REQUIRE(p.set_transform_axis(0, 0));
    CHECK(index_of(0) == 1);
    CHECK(p.transform_note() == "2 is outside 0" + dash + "1");
    // A reduce after the slice sees the slice's result, [3, 8, 8].
    REQUIRE(p.add_transform(spec::TransformOp::reduce));
    CHECK(p.transform_axes(1) == std::vector<std::string>{ax(0, 2), ax(1, 7), ax(2, 7)});

    // A source that carries names, as a live view does: each axis is named,
    // and a slice's index takes its axis's name away.
    r.port->set_dim_names([](const std::string& id) {
        return id == "tensor:attn_weights" ? std::vector<std::string>{"layer", "head", "query", "key"}
                                           : std::vector<std::string>{};
    });
    p.focus_source("spec:attn_weights");
    CHECK(p.transform_axes(0) ==
          std::vector<std::string>{ax(0, 1, "layer"), ax(1, 2, "head"), ax(2, 7, "query"), ax(3, 7, "key")});
    CHECK(p.transform_axes(1) == std::vector<std::string>{ax(0, 2, "head"), ax(1, 7, "query"), ax(2, 7, "key")});
    const std::vector<PaneShape> shapes = p.transform_shapes();
    REQUIRE(shapes.size() == 3);
    CHECK(shapes[2].names == std::vector<std::string>{"query", "key"});   // the reduce took head away
    // Names that do not match the rank are not used.
    std::vector<PaneShape> odd = shapes;
    for (auto& sh : odd) sh.names.clear();
    pane_name_dims(odd, p.open_spec()->spec.find_source("attn_weights")->transforms, {"a", "b"});
    CHECK(odd[0].names.empty());
}

// A reduce over several axes keeps them all: a click on an axis in
// Properties adds it to the list or takes it out, never leaving the list
// empty, and every chosen axis is highlighted. A slice's axis is one.
TEST_CASE("builder pane: a reduce's axes toggle in a list, never empty") {
    Rig r = rig("pane-reduce-axes");
    BuilderPane& p = *r.pane;
    const auto axes_of = [&](size_t step) {
        const spec::Source* src = p.open_spec()->spec.find_source("attn_weights");
        REQUIRE(src);
        REQUIRE(step < src->transforms.size());
        return src->transforms[step].axis;
    };
    // The tensor is [2, 3, 8, 8]; a reduce starts on axis 0.
    p.select("tensor:attn_weights");
    REQUIRE(p.add_transform(spec::TransformOp::reduce));
    CHECK(axes_of(0) == std::vector<int64_t>{0});
    CHECK(p.transform_axes_chosen(0) == std::vector<int64_t>{0});
    // A click on axis 2 adds it; the reduce is over both.
    REQUIRE(p.set_transform_axis(0, 2));
    CHECK(axes_of(0) == std::vector<int64_t>{0, 2});
    CHECK(p.transform_axes_chosen(0) == std::vector<int64_t>{0, 2});
    REQUIRE(p.transform_shapes().size() == 2);
    CHECK(p.transform_shapes()[1].dims == std::vector<int64_t>{3, 8});
    // A click on a chosen axis takes it out.
    REQUIRE(p.set_transform_axis(0, 0));
    CHECK(axes_of(0) == std::vector<int64_t>{2});
    CHECK(p.transform_axes_chosen(0) == std::vector<int64_t>{2});
    // The last one stays: a reduce over no axis is not one.
    const size_t sent = p.sent().size();
    p.set_transform_axis(0, 2);
    CHECK(axes_of(0) == std::vector<int64_t>{2});
    CHECK(p.sent().size() == sent);
    // A spec written with a list keeps all of its axes through a click.
    REQUIRE(p.set_transform(0, nlohmann::json{{"op", "reduce"}, {"axis", {1, 3}}, {"fn", "mean"}}));
    CHECK(p.transform_axes_chosen(0) == std::vector<int64_t>{1, 3});
    REQUIRE(p.set_transform_axis(0, 0));
    CHECK(axes_of(0) == std::vector<int64_t>{0, 1, 3});
    REQUIRE(p.set_transform_axis(0, 1));
    REQUIRE(p.set_transform_axis(0, 3));
    CHECK(axes_of(0) == std::vector<int64_t>{0});
    // A slice's axis is replaced, as before: [3, 8, 8], axis 1 of three.
    REQUIRE(p.add_transform(spec::TransformOp::slice));
    REQUIRE(p.set_transform_axis(1, 1));
    CHECK(axes_of(1) == std::vector<int64_t>{1});
    CHECK(p.transform_axes_chosen(1) == std::vector<int64_t>{1});
}

// A window, a surface and a path come from commands. Properties shows each
// with its name and every parameter as text, a reference as its table and a
// left-out key as its default, and the pane neither edits nor adds them.
TEST_CASE("builder pane: window, surface and path steps show their parameters read-only") {
    using Fields = std::vector<std::pair<std::string, std::string>>;
    const auto step = [](const json& t) {
        const auto d = spec::source_from_json(json{{"id", "s"}, {"from", "act"}, {"transform", json::array({t})}});
        REQUIRE(d.value.transforms.size() == 1);
        return d.value.transforms[0];
    };
    CHECK(pane_step_fields(step({{"op", "window"},
                                 {"start", {{"control", "window"}, {"field", "lo"}}},
                                 {"stop", {{"control", "window"}, {"field", "hi"}}}})) ==
          Fields{{"start", "{ control = \"window\", field = \"lo\" }"},
                 {"stop", "{ control = \"window\", field = \"hi\" }"}});
    CHECK(pane_step_fields(step({{"op", "window"}, {"steps", 2048}})) == Fields{{"steps", "2048"}});
    CHECK(pane_step_fields(step({{"op", "window"}, {"steps", {{"state", "depth"}, {"default", 64}}}})) ==
          Fields{{"steps", "{ state = \"depth\", default = 64 }"}});
    CHECK(pane_step_fields(step({{"op", "window"}})) == Fields{{"steps", "every kept step (default)"}});
    CHECK(pane_step_fields(step({{"op", "surface"}, {"max_cells", 4096}, {"pool", "max"}})) ==
          Fields{{"max_cells", "4096"}, {"pool", "max"}});
    CHECK(pane_step_fields(step({{"op", "surface"}})) ==
          Fields{{"max_cells", "32768 (default)"}, {"pool", "mean (default)"}});
    CHECK(pane_step_fields(step({{"op", "path"}, {"columns", {0, 2}}, {"max_points", 500}})) ==
          Fields{{"columns", "0, 2"}, {"max_points", "500"}});
    CHECK(pane_step_fields(step({{"op", "path"}})) ==
          Fields{{"columns", "the first three (default)"}, {"max_points", "65536 (default)"}});
    // The steps that the pane edits have their own fields.
    CHECK(pane_step_fields(step({{"op", "normalise"}, {"mode", "zscore"}})).empty());

    // A window puts the step axis in front of the names; an operator ends them.
    PaneShape a;
    a.known = true;
    a.dims = {64};
    PaneShape b = a;
    b.dims = {16, 64};
    std::vector<PaneShape> shapes = {a, b};
    pane_name_dims(shapes, {step({{"op", "window"}, {"steps", 16}})}, {"token"});
    CHECK(shapes[1].names == std::vector<std::string>{"step", "token"});
    PaneShape g;
    g.known = true;
    g.dims = {16, 64};
    shapes = {b, g};
    pane_name_dims(shapes, {step({{"op", "surface"}})}, {"step", "token"});
    CHECK(shapes[1].names.empty());

    // The pane adds none of them, and sends nothing.
    Rig r = rig("pane-read-only-steps");
    BuilderPane& p = *r.pane;
    p.select("tensor:attn_weights");
    const size_t sent = p.sent().size();
    for (spec::TransformOp op : {spec::TransformOp::window, spec::TransformOp::surface, spec::TransformOp::path})
        CHECK_FALSE(p.add_transform(op));
    CHECK(p.sent().size() == sent);
}

// A new dashboard's empty canvas says how to start in build mode, and
// nothing more once it has a panel or outside build mode.
TEST_CASE("builder pane: an empty canvas says how to start") {
    Rig r = rig("pane-empty");
    BuilderPane& p = *r.pane;
    CHECK(p.canvas_hint() == "Drop a data file, or pick a source in Sources");
    p.set_build_mode(false);
    CHECK(p.canvas_hint().empty());
    p.set_build_mode(true);
    p.select("table:run42_metrics");
    REQUIRE(p.add_primitive(spec::Primitive::series));
    CHECK(p.canvas_hint().empty());
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
    CHECK(*p.open_spec()->spec.instrument.title == "Run 42, again");
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

TEST_CASE("builder pane: an approved agent edit, a ⌘K Accept and a hand edit share one history") {
    Rig r = rig("pane-history");
    BuilderPane& p = *r.pane;
    p.select("table:run42_metrics");
    REQUIRE(p.add_primitive(spec::Primitive::series));   // the hand's first step
    auto title = [&] {
        const spec::Panel* x = p.open_spec()->spec.find_panel("series");
        return x && x->title ? *x->title : std::string();
    };
    auto top_origin = [&] { return r.s->registry().history_state(1).undo.front().origin; };

    // An agent's edit, held for review and approved on the cards: through the
    // session's queue, as Caliper API submits it, and the queue's approve().
    cb::CommandInvocation agent;
    agent.name = "panel.patch";
    agent.args = {{"id", "series"}, {"patch", {{"title", "Agent"}}}};
    agent.origin = cb::CommandOrigin::agent;
    auto held = r.s->queue().submit(agent);
    r.s->queue().drain();
    const auto pending = r.s->registry().pending_approvals();
    REQUIRE(pending.size() == 1);
    auto decided = r.s->queue().approve(pending[0].approval);
    r.s->queue().drain();
    REQUIRE(held.get().ok);
    REQUIRE(decided.get().ok);
    p.update();
    CHECK(title() == "Agent");
    CHECK(top_origin() == cb::CommandOrigin::agent);
    // Undo in the page bar takes back the agent's edit, the last one.
    REQUIRE(p.undo());
    CHECK(title().empty());
    REQUIRE(p.redo());
    CHECK(title() == "Agent");

    // ⌘K on the panel: the inline surface's preview, accepted as one step.
    auto clock = std::chrono::steady_clock::now();
    auto now = [&clock] { return clock; };
    WidgetScript script;
    WidgetScriptTurn turn;
    WidgetScriptStep edit;
    edit.at = std::chrono::milliseconds(100);
    edit.kind = WidgetScriptStep::Kind::edit;
    edit.edit.name = "panel.patch";
    edit.edit.args = {{"id", "series"}, {"patch", {{"options", {{"y_log", true}}}}}};
    WidgetScriptStep reply;
    reply.at = std::chrono::milliseconds(200);
    reply.kind = WidgetScriptStep::Kind::message;
    reply.text = "A log y axis.";
    turn.steps = {edit, reply};
    turn.end_at = std::chrono::milliseconds(300);
    script.turns = {turn};
    ScriptedWidgetAgent widget_agent(
        script, [&](cb::CommandInvocation inv) { return r.s->invoke(std::move(inv)); }, now,
        [&](const std::string& id) { return (int)r.s->preview(id).edits; });
    SessionWidgetSession host(*r.s);
    WidgetSurface::Hooks hooks;
    hooks.now = now;
    WidgetSurface surface(host, widget_agent, hooks);
    REQUIRE(surface.open(WidgetTarget::for_panel("series")));
    surface.set_text("Use a log y axis");
    surface.submit();
    for (int i = 0; i < 20 && surface.state() != WidgetState::preview; ++i) {
        clock += std::chrono::milliseconds(50);
        surface.update();
    }
    REQUIRE(surface.state() == WidgetState::preview);
    surface.accept();
    REQUIRE(surface.state() == WidgetState::applied);
    p.update();
    auto y_log = [&] { return p.open_spec()->spec.find_panel("series")->options.y_log.value_or(false); };
    CHECK(y_log());
    // The Accept's step holds the agent's edit under the preview's group.
    CHECK(top_origin() == cb::CommandOrigin::agent);
    CHECK(r.s->registry().history_state(1).undo.front().group.rfind("widget-", 0) == 0);
    REQUIRE(p.undo());   // the Accept, as one step
    CHECK_FALSE(y_log());
    CHECK(title() == "Agent");
    REQUIRE(p.redo());
    CHECK(y_log());

    // A hand edit after both: Undo takes back the hand's, then the Accept,
    // then the agent's.
    REQUIRE(p.set_title("Hand"));
    CHECK(top_origin() == cb::CommandOrigin::builder);
    REQUIRE(p.undo());
    CHECK(title() == "Agent");
    CHECK(y_log());
    REQUIRE(p.undo());
    CHECK_FALSE(y_log());
    REQUIRE(p.undo());
    CHECK(title().empty());
}

TEST_CASE("builder pane: a new spec's id from its title") {
    auto none = [](const std::string&) { return false; };
    CHECK(pane_new_spec_id("Run 42", none) == "run-42");
    CHECK(pane_new_spec_id("  Loss & accuracy!  ", none) == "loss-accuracy");
    CHECK(pane_new_spec_id("", none) == "dashboard");
    CHECK(pane_new_spec_id("42 runs", none) == "d-42-runs");
    CHECK(pane_new_spec_id("Untitled", [](const std::string& id) { return id == "untitled"; }) == "untitled-2");
    CHECK(pane_valid_id(pane_new_spec_id(std::string(200, 'x'), none)));
    // The spec's own id, renamed through its text.
    Rig r = rig("pane-spec-id");
    REQUIRE(r.pane->set_spec_id("run42-board"));
    CHECK(r.pane->open_spec()->spec.instrument.id == "run42-board");
    CHECK(r.pane->sent().back().name == "spec.set_text");
    CHECK_FALSE(r.pane->set_spec_id("Run 42"));
    CHECK(r.pane->error().code == "spec.bad_id");
}

namespace {

// A workspace piece as piece.update leaves it: pieces/<name>/1/ with its
// piece.toml and its WGSL files.
void write_piece(const std::string& ws, const std::string& name, const std::string& manifest,
                 const std::vector<std::pair<std::string, std::string>>& files) {
    const std::string dir = ws + "/pieces/" + name + "/1";
    std::filesystem::create_directories(dir);
    std::ofstream(dir + "/piece.toml") << manifest;
    for (const auto& [file, text] : files) std::ofstream(dir + "/" + file + ".wgsl") << text;
}

// A piece that only draws and reads no input, and one with a tensor input.
void write_pieces(const std::string& ws) {
    write_piece(ws, "glow",
                "name = \"glow\"\nversion = 1\ndescription = \"A soft glow over the panel\"\n\n"
                "[[params]]\nname = \"gain\"\ndefault = 1.0\nmin = 0.0\nmax = 2.0\n\n"
                "[draw]\ncamera = \"none\"\npasses = [{ kind = \"render\", fragment = \"shade\" }]\n",
                {{"draw", "@fragment fn shade() -> @location(0) vec4<f32> {\n"
                          "    return vec4<f32>(0.2, 0.6, 1.0, 1.0) * params.gain;\n}\n"}});
    write_piece(ws, "row_norms",
                "name = \"row_norms\"\nversion = 1\ndescription = \"The norm of each row of a matrix, as bars\"\n\n"
                "[[inputs]]\nname = \"w\"\nkind = \"tensor\"\nrank_min = 2\nrank_max = 2\n\n"
                "[[resources]]\nid = \"norms\"\nkind = \"buffer\"\nelement = \"f32\"\ncount = 64\n\n"
                "[derive]\npasses = [{ entry = \"measure\", workgroups = [1, 1, 1], writes = [\"norms\"] }]\n\n"
                "[draw]\ncamera = \"none\"\npasses = [{ kind = \"render\", fragment = \"bars\" }]\n",
                {{"derive", "@compute @workgroup_size(64) fn measure(@builtin(global_invocation_id) id: vec3<u32>) {\n"
                            "    norms[id.x] = w[id.x];\n}\n"},
                 {"draw", "@fragment fn bars() -> @location(0) vec4<f32> {\n"
                          "    return vec4<f32>(norms[0], 0.0, 0.0, 1.0);\n}\n"}});
}

}  // namespace

TEST_CASE("builder pane: the rows of piece.list") {
    CHECK(pane_piece_rows(json()).empty());
    CHECK(pane_piece_rows(json::array()).empty());
    CHECK(pane_piece_rows({{"pieces", 3}}).empty());
    const json list = {{"pieces",
                        {{{"name", "points"}, {"version", 1}, {"from", "library"}, {"description", "Points"},
                          {"draws", true}, {"inputs", json::array()}},
                         {{"version", 2}, {"from", "workspace"}},
                         {{"name", "bars"}, {"version", 3}, {"from", "workspace"}, {"description", 7},
                          {"draws", "yes"}, {"inputs", {"w", 2, "v"}}}}}};
    const std::vector<PaneSource> rows = pane_piece_rows(list);
    REQUIRE(rows.size() == 2);
    CHECK(rows[0].group == PaneSource::Group::piece);
    CHECK(rows[0].key == "piece:library/points@1");
    CHECK(rows[0].name == "points@1");
    CHECK(rows[0].shape_text == "library \xc2\xb7 Points");
    CHECK(rows[0].piece == json{{"name", "points"}, {"version", 1}, {"from", "library"}});
    CHECK(rows[0].draws);
    CHECK(rows[1].key == "piece:workspace/bars@3");
    CHECK(rows[1].shape_text == "workspace");
    CHECK(rows[1].inputs == std::vector<std::string>{"w", "v"});
    CHECK_FALSE(rows[1].draws);
    CHECK(std::string(pane_group_label(PaneSource::Group::piece)) == "Pieces");
    spec::Spec s;
    CHECK(pane_program_id("glow", s) == "glow");
    s.programs.emplace_back();
    s.programs.back().id = "glow";
    CHECK(pane_program_id("glow", s) == "glow-2");
}

TEST_CASE("builder pane: pieces in Sources, and a piece dropped on a scene panel as its program") {
    Rig r = rig("pane-pieces");
    build_dashboard(r);
    BuilderPane& p = *r.pane;
    write_pieces(r.ws);
    cb::CommandInvocation inv;
    inv.name = "panel.create";
    inv.args = {{"id", "world"}, {"primitive", "scene"}, {"bind", {{"scene", "live:model"}}},
                {"place", {{"target", "rows"}, {"where", "beside"}}}};
    inv.origin = cb::CommandOrigin::builder;
    const cb::CommandResult made = r.s->invoke(inv);
    REQUIRE_MESSAGE(made.ok, made.error.message);
    p.update();

    // The group Pieces after the tensors, in piece.list's order; a piece row
    // keeps the focus when selected. The pieces of the built-in library are
    // left out, so that the test does not depend on what the library holds.
    const auto library = [](const PaneSource* s) { return s->key.rfind("piece:library/", 0) == 0; };
    const std::string glow = "piece:workspace/glow@1", norms = "piece:workspace/row_norms@1";
    const PaneSource* g = p.find_source(glow);
    REQUIRE(g);
    CHECK(g->name == "glow@1");
    CHECK(g->shape_text == "workspace \xc2\xb7 A soft glow over the panel");
    CHECK(g->draws);
    REQUIRE(p.find_source(norms));
    CHECK(p.find_source(norms)->inputs == std::vector<std::string>{"w"});
    std::vector<std::string> order;
    for (const PaneSource* s : p.visible_sources())
        if ((s->group == PaneSource::Group::piece && !library(s)) || s->group == PaneSource::Group::tensor)
            order.push_back(s->key);
    CHECK(order == std::vector<std::string>{"tensor:attn_weights", glow, norms});
    p.set_filter("glow");
    bool only = true;
    for (const PaneSource* s : p.visible_sources()) only = only && (library(s) || s->key == glow);
    CHECK(only);
    p.set_filter("");
    p.focus_panel("curves");
    p.select(glow);
    CHECK(p.focused_panel() == "curves");
    CHECK(p.focused_source().empty());

    // Refused before anything is sent: a piece with an input to bind, a
    // primitive, an edge, a panel that is no scene panel.
    const spec::Spec before = p.open_spec()->spec;
    const size_t steps = r.steps(), sent = p.sent().size();
    CHECK_FALSE(p.add_selected());
    CHECK(p.error().code == "program.bad_target");
    CHECK(p.primitive_plan(spec::Primitive::scene).refusal.code == "program.bad_target");
    CHECK_FALSE(p.drop(source(glow), at(DropZone::edge, "world", "below")));
    CHECK(p.error().code == "program.bad_target");
    CHECK_FALSE(p.drop(source(norms), at(DropZone::centre, "world", "")));
    CHECK(p.error().code == "piece.bad_binding");
    CHECK(p.error().message.rfind("The piece row_norms@1 reads the input w, which its program binds", 0) == 0);
    CHECK(p.sent().size() == sent);
    CHECK(r.steps() == steps);
    CHECK(p.open_spec()->spec == before);

    // Dropped in the scene panel's centre: one program.create from the
    // builder origin, one undo step, pinned with the hash the session wrote.
    REQUIRE_MESSAGE(p.drop(source(glow), at(DropZone::centre, "world", "")), p.error().message);
    const spec::Program* prog = p.open_spec()->spec.find_program("glow");
    REQUIRE(prog);
    REQUIRE(prog->piece);
    CHECK(prog->piece->name == "glow");
    CHECK(prog->piece->from == spec::PieceFrom::workspace);
    CHECK(prog->piece->hash.rfind("sha256:", 0) == 0);
    REQUIRE(prog->panel);
    CHECK(*prog->panel == "world");
    CHECK(r.journal().back()["name"] == "program.create");
    CHECK(r.journal().back()["origin"] == "builder");
    CHECK(r.steps() == steps + 1);
    CHECK(p.focused_panel() == "world");
    // Selected, then Add with the scene panel focused: the next id.
    p.select(glow);
    REQUIRE_MESSAGE(p.add_selected(), p.error().message);
    CHECK(p.open_spec()->spec.find_program("glow-2"));
    REQUIRE(p.undo());
    REQUIRE(p.undo());
    CHECK(p.open_spec()->spec == before);
}

TEST_CASE("builder pane: non-closable panels and Lock Layout refuse close and layout edits") {
    Rig r = rig("pane-lock-closable");
    build_dashboard(r);
    BuilderPane& p = *r.pane;

    // Marking a panel non-closable refuses both remove_panel and remove_focused without sending a command.
    p.focus_panel("curves");
    REQUIRE_MESSAGE(p.set_closable(false), p.error().message);
    CHECK(p.open_spec()->spec.find_panel("curves")->closable == std::optional<bool>(false));
    const size_t sent_before_refusal = p.sent().size();
    CHECK_FALSE(p.remove_panel("curves"));
    CHECK(p.error().code == "panel.not_closable");
    CHECK_FALSE(p.remove_focused());
    CHECK(p.error().code == "panel.not_closable");
    CHECK(p.sent().size() == sent_before_refusal);
    CHECK(p.open_spec()->spec.find_panel("curves") != nullptr);

    // Lock Layout prevents closing any panel, resizing dividers, moving panels, or adding panels.
    REQUIRE_MESSAGE(p.set_layout_locked(true), p.error().message);
    CHECK(p.layout_locked());
    const size_t sent_locked = p.sent().size();
    CHECK_FALSE(p.remove_panel("rows"));
    CHECK(p.error().code == "layout.locked");
    CHECK_FALSE(p.set_ratio("", 0.4));
    CHECK(p.error().code == "layout.locked");
    DragPayload move;
    move.kind = DragPayload::Kind::panel;
    move.panel = "rows";
    CHECK_FALSE(p.drop(move, at(DropZone::edge, "curves", "below")));
    CHECK(p.error().code == "layout.locked");
    p.select("table:run42_metrics");
    CHECK_FALSE(p.add_primitive(spec::Primitive::text));
    CHECK(p.error().code == "layout.locked");
    CHECK(p.sent().size() == sent_locked);

    // Editing a panel's title still works while the layout is locked.
    p.focus_panel("rows");
    REQUIRE_MESSAGE(p.set_title("Metrics table"), p.error().message);
    CHECK(p.open_spec()->spec.find_panel("rows")->title == std::optional<std::string>("Metrics table"));

    // Unlocking restores layout edits and removal of closable panels; non-closable panels stay protected.
    REQUIRE_MESSAGE(p.set_layout_locked(false), p.error().message);
    CHECK_FALSE(p.layout_locked());
    REQUIRE_MESSAGE(p.set_ratio("", 0.4), p.error().message);
    REQUIRE_MESSAGE(p.remove_panel("rows"), p.error().message);
    CHECK(p.open_spec()->spec.find_panel("rows") == nullptr);
    CHECK_FALSE(p.remove_panel("curves"));
    CHECK(p.error().code == "panel.not_closable");

    // Re-enabling closable lets curves be removed too.
    p.focus_panel("curves");
    REQUIRE_MESSAGE(p.set_closable(true), p.error().message);
    REQUIRE_MESSAGE(p.remove_panel("curves"), p.error().message);
    CHECK(p.open_spec()->spec.find_panel("curves") == nullptr);
}
