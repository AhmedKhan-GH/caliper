// The inline surface's other targets, headless over a real Session with the
// dashboard of dashboard.spec.toml and the scripted agent: a selection in a
// panel (its rows in the context bundle, a new table panel previewed as the
// whole spec's preview, committed as one undo step or discarded without a
// trace), a dropped file (an answer without a change, a proposed panel, and
// a spec.new that a discard undoes when no spec was open) and a property
// (an edit of that panel, previewed in it). Each session's tool scope is the
// one the scope rule in widget_scopes.json gives, read through the session
// library, not written here; and the runtime's selection gives its rows.
#include "builder/widget_agent.h"
#include "builder/widget_surface.h"
#include "spec_runtime/spec_prepare.h"
#include "spec_runtime/spec_runtime.h"

#include "spec_test_workspace.h"

#include <caliper_builder/session/context_bundle.h>

#include <doctest/doctest.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <memory>
#include <set>
#include <string>
#include <vector>

using namespace caliper_host;
using namespace spec_test;
using caliper_builder::CommandOrigin;
using nlohmann::json;
using std::chrono::milliseconds;

namespace {

struct Sink : WidgetPreviewSink {
    std::vector<std::string> panels;            // show()
    std::vector<std::set<std::string>> specs;   // show_spec(): the added panels
    int cleared = 0, cleared_specs = 0;
    void show(const caliper_builder::spec::Panel& panel, const caliper_builder::PanelData&,
              const caliper_builder::spec::Spec&) override {
        panels.push_back(panel.id);
    }
    void clear(const std::string&) override { ++cleared; }
    void show_spec(const caliper_builder::spec::Spec&, const std::set<std::string>& added) override {
        specs.push_back(added);
    }
    void clear_spec() override { ++cleared_specs; }
};

WidgetScriptStep edit(int ms, const std::string& name, json args) {
    WidgetScriptStep s;
    s.at = milliseconds(ms);
    s.kind = WidgetScriptStep::Kind::edit;
    s.edit.name = name;
    s.edit.args = std::move(args);
    return s;
}

WidgetScriptStep message(int ms, std::string text) {
    WidgetScriptStep s;
    s.at = milliseconds(ms);
    s.kind = WidgetScriptStep::Kind::message;
    s.text = std::move(text);
    return s;
}

// The table of the selected steps beside the curves.
WidgetScriptTurn table_turn() {
    WidgetScriptTurn t;
    t.steps = {edit(100, "source.add",
                    {{"id", "selected_steps"},
                     {"sql", "SELECT step, loss, accuracy FROM run42_metrics WHERE step BETWEEN 60 AND 90"}}),
               edit(200, "panel.create",
                    {{"id", "selected_rows"},
                     {"primitive", "table"},
                     {"title", "Steps 60 to 90"},
                     {"bind", {{"rows", "selected_steps"}}}}),
               edit(300, "layout.place", {{"panel", "selected_rows"}, {"target", "curves"}, {"where", "beside"}}),
               message(350, "A table of the 31 selected steps beside the curves.")};
    t.end_at = milliseconds(400);
    return t;
}

struct Rig {
    std::string root, ws;
    std::unique_ptr<caliper_builder::Session> s;
    std::chrono::steady_clock::time_point t = std::chrono::steady_clock::now();
    std::unique_ptr<ScriptedWidgetAgent> agent;
    std::unique_ptr<SessionWidgetSession> host;
    Sink sink;
    std::unique_ptr<WidgetSurface> surface;
    std::string before;
    size_t steps_before = 0;

    Rig(const std::string& name, std::vector<WidgetScriptTurn> turns, bool with_spec = true) {
        root = fresh_dir("widget-target-" + name);
        ws = root + "/ws";
        if (with_spec) {
            s = dashboard_workspace(ws, root + "/data");
        } else {
            write_parquet(root + "/data/validation.parquet",
                          "SELECT range AS step, 0.5 + 0.1 * cos(range / 7.0) AS val_loss FROM range(50)");
            auto opened = caliper_builder::Session::open(ws);
            REQUIRE(opened.error.ok());
            s = std::move(opened.session);
            ok(*s, "data.import", {{"paths", {root + "/data/validation.parquet"}}, {"run", true}});
        }
        WidgetScript script;
        script.start_after = milliseconds(50);
        script.turns = std::move(turns);
        agent = std::make_unique<ScriptedWidgetAgent>(
            script, [this](caliper_builder::CommandInvocation inv) { return s->invoke(std::move(inv)); },
            [this] { return t; }, [this](const std::string& id) { return (int)s->preview(id).edits; });
        host = std::make_unique<SessionWidgetSession>(*s);
        host->bundle_options.vision = false;
        WidgetSurface::Hooks hooks;
        hooks.now = [this] { return t; };
        surface = std::make_unique<WidgetSurface>(*host, *agent, hooks);
        surface->set_preview_sink(&sink);
        if (auto open = s->open_spec()) before = open->text;
        steps_before = s->registry().history_state().undo.size();
    }
    ~Rig() {
        surface.reset();
        s->close();
    }
    void run(int ms) {
        for (int i = 0; i < ms; i += 50) {
            t += milliseconds(50);
            surface->update();
        }
    }
    void ask(const WidgetTarget& target, const std::string& text) {
        REQUIRE(surface->open(target));
        surface->set_text(text);
        surface->submit();
        REQUIRE_MESSAGE(surface->state() == WidgetState::running, surface->error().code << " " << surface->error().message);
        run(600);
    }
    size_t undo_steps() const { return s->registry().history_state().undo.size(); }
    // The scope rule's scope for a target, read through the session library.
    std::vector<std::string> rule_scope(const std::string& key) const {
        caliper_builder::Error e;
        const auto scope = caliper_builder::widget_tool_scope(s->registry().descriptors(), key, false, &e);
        REQUIRE(!e);
        return scope;
    }
};

caliper_builder::WidgetSelection selected_steps() {
    caliper_builder::WidgetSelection sel;
    sel.kind = "range";
    sel.columns = {"step", "loss", "accuracy"};
    for (int step = 60; step <= 90; ++step)
        sel.rows.push_back({step, 0.1 + 2.0 * std::exp(-step / 50.0), 0.1 + 0.8 * (1 - std::exp(-step / 40.0))});
    sel.range = {{"step", {60, 90}}};
    return sel;
}

}  // namespace

TEST_CASE("widget targets: a selection's rows go to the agent, and a new table panel is the whole spec's preview") {
    Rig r("selection", {table_turn()});
    r.ask(WidgetTarget::for_selection("curves", selected_steps()), "Make a table of these points");
    REQUIRE_MESSAGE(r.surface->state() == WidgetState::preview, r.surface->error().code << " " << r.surface->error().message);
    CHECK(r.surface->edits() == 3);
    // The session's scope is the rule's for a selection, and the bundle
    // carries the selected rows.
    REQUIRE(r.agent->scopes().size() == 1);
    CHECK(r.agent->scopes()[0] == r.rule_scope("selection"));
    const std::string bundle = r.agent->contexts()[0].dump();
    CHECK(bundle.find("selection") != std::string::npos);
    CHECK(bundle.find("columns") != std::string::npos);
    CHECK(bundle.find("[60,90]") != std::string::npos);
    // Drawn as the whole spec's preview, the new panel marked as added.
    REQUIRE(r.sink.specs.size() == 1);
    CHECK(r.sink.specs[0] == std::set<std::string>{"selected_rows"});
    CHECK(r.sink.panels.empty());
    CHECK(r.s->open_spec()->text == r.before);
    bool added_line = false;
    for (const auto& d : r.surface->diff()) added_line = added_line || (d.key == "panel selected_rows" && d.kind == WidgetDiffLine::Kind::added);
    CHECK(added_line);
    // Accept: one undo step that adds the source, the panel and its place.
    r.surface->accept();
    CHECK(r.surface->state() == WidgetState::applied);
    CHECK(r.sink.cleared_specs == 1);
    CHECK(r.undo_steps() == r.steps_before + 1);
    const auto open = r.s->open_spec();
    REQUIRE(open->spec.find_panel("selected_rows"));
    const caliper_builder::PanelData data = r.s->resolve("selected_rows");
    CHECK(!data.error);
    ok(*r.s, "history.undo");
    CHECK(r.s->open_spec()->text == r.before);
}

TEST_CASE("widget targets: Reject leaves no trace of a preview that added a panel") {
    Rig r("selection-reject", {table_turn()});
    r.ask(WidgetTarget::for_selection("curves", selected_steps()), "Make a table of these points");
    REQUIRE(r.surface->state() == WidgetState::preview);
    const std::string preview = r.surface->preview_id();
    r.surface->reject();
    CHECK(r.surface->state() == WidgetState::idle);
    CHECK(r.sink.cleared_specs == 1);
    CHECK(r.s->preview(preview).error);
    CHECK(r.s->open_spec()->text == r.before);
    CHECK(!r.s->open_spec()->spec.find_panel("selected_rows"));
    CHECK(r.undo_steps() == r.steps_before);
}

TEST_CASE("widget targets: a dropped file is answered in the card, and a proposed panel is previewed") {
    WidgetScriptTurn answer;
    answer.steps = {message(100, "A table of 200 rows with three columns: step, loss and accuracy. Loss falls from 2.1 to "
                                 "about 0.1, and accuracy rises from 0.1 to about 0.9.")};
    answer.end_at = milliseconds(200);
    WidgetScriptTurn plot;
    plot.steps = {edit(100, "source.add", {{"id", "acc"}, {"catalog", "table:run42_metrics"}}),
                  edit(200, "panel.create",
                       {{"id", "accuracy"}, {"primitive", "series"}, {"bind", {{"x", "acc.step"}, {"y", {"acc.accuracy"}}}}}),
                  message(300, "A plot of accuracy against step.")};
    plot.end_at = milliseconds(400);
    Rig r("file", {answer, plot});
    const WidgetTarget target =
        WidgetTarget::for_dropped_file("run42_metrics", {"table:run42_metrics"}, "source:table:run42_metrics");
    CHECK(target.frame_panel().empty());   // drawn under its row, not in a panel
    r.ask(target, "What's in this?");
    REQUIRE(r.surface->state() == WidgetState::preview);
    CHECK(r.surface->edits() == 0);
    CHECK(r.surface->tag().empty());
    CHECK(r.surface->reply().find("three columns") != std::string::npos);
    CHECK(r.surface->card_actions() == std::vector<WidgetAction>{WidgetAction::refine, WidgetAction::close});
    CHECK(r.agent->scopes()[0] == r.rule_scope("dropped_file"));
    const std::string bundle = r.agent->contexts()[0].dump();
    CHECK(bundle.find("run42_metrics") != std::string::npos);
    CHECK(bundle.find("loader") != std::string::npos);
    CHECK(r.sink.specs.empty());
    // A follow-up proposes a panel: the whole spec's preview.
    r.surface->refine();
    r.surface->set_text("Plot accuracy against step");
    r.surface->submit();
    r.run(600);
    REQUIRE(r.surface->state() == WidgetState::preview);
    CHECK(r.surface->edits() == 2);
    REQUIRE(r.sink.specs.size() == 1);
    CHECK(r.sink.specs[0] == std::set<std::string>{"accuracy"});
    r.surface->accept();
    CHECK(r.undo_steps() == r.steps_before + 1);
    CHECK(r.s->open_spec()->spec.find_panel("accuracy"));
}

TEST_CASE("widget targets: a dropped file with no open spec gets a spec.new, which Reject undoes") {
    WidgetScriptTurn plot;
    plot.steps = {edit(100, "source.add", {{"id", "val"}, {"catalog", "table:validation"}}),
                  edit(200, "panel.create",
                       {{"id", "val_loss"}, {"primitive", "series"}, {"bind", {{"x", "val.step"}, {"y", {"val.val_loss"}}}}}),
                  message(300, "A plot of the validation loss.")};
    plot.end_at = milliseconds(400);
    Rig r("file-nospec", {plot}, /*with_spec=*/false);
    REQUIRE(!r.s->open_spec());
    r.ask(WidgetTarget::for_dropped_file("validation", {"table:validation"}, "source:table:validation"),
          "Plot the validation loss");
    REQUIRE_MESSAGE(r.surface->state() == WidgetState::preview, r.surface->error().code << " " << r.surface->error().message);
    REQUIRE(r.s->open_spec());   // the spec.new the preview needed
    CHECK(r.s->open_spec()->spec.panels.empty());
    CHECK(r.surface->edits() == 2);
    r.surface->reject();
    CHECK(!r.s->open_spec());   // undone with the preview
    CHECK(r.undo_steps() == r.steps_before);
}

namespace {

WidgetScriptTurn validation_plot() {
    WidgetScriptTurn plot;
    plot.steps = {edit(100, "source.add", {{"id", "val"}, {"catalog", "table:validation"}}),
                  edit(200, "panel.create",
                       {{"id", "val_loss"}, {"primitive", "series"}, {"bind", {{"x", "val.step"}, {"y", {"val.val_loss"}}}}}),
                  message(300, "A plot of the validation loss.")};
    plot.end_at = milliseconds(400);
    return plot;
}

}  // namespace

TEST_CASE("widget targets: a discard undoes only its own spec.new, never a step committed after it") {
    Rig r("file-nospec-later", {validation_plot()}, /*with_spec=*/false);
    r.ask(WidgetTarget::for_dropped_file("validation", {"table:validation"}, "source:table:validation"),
          "Plot the validation loss");
    REQUIRE(r.surface->state() == WidgetState::preview);
    REQUIRE(r.s->open_spec());
    // Another step lands on the new spec while the preview is open (the
    // person, the side chat, a program over Caliper API).
    ok(*r.s, "source.add", {{"id", "later"}, {"catalog", "table:validation"}});
    const size_t steps = r.undo_steps();
    REQUIRE(steps == r.steps_before + 2);   // the spec.new, then the later step
    r.surface->reject();
    CHECK(r.surface->state() == WidgetState::idle);
    // The later step stays, and so does the spec it needs.
    REQUIRE(r.s->open_spec());
    CHECK(r.s->open_spec()->spec.find_source("later") != nullptr);
    CHECK(r.undo_steps() == steps);
}

TEST_CASE("widget targets: a dropped file's preview undoes its spec.new when the surface goes, as a discard does") {
    Rig r("file-nospec-gone", {validation_plot()}, /*with_spec=*/false);
    r.ask(WidgetTarget::for_dropped_file("validation", {"table:validation"}, "source:table:validation"),
          "Plot the validation loss");
    REQUIRE(r.surface->state() == WidgetState::preview);
    REQUIRE(r.s->open_spec());
    r.surface.reset();   // the page drops its widget (a settings change, a ⌘K with another agent)
    CHECK(!r.s->open_spec());
    CHECK(r.undo_steps() == r.steps_before);
}

TEST_CASE("widget targets: Accept on a dropped file with no open spec leaves two undo steps, the spec.new and the change") {
    Rig r("file-nospec-accept", {validation_plot()}, /*with_spec=*/false);
    r.ask(WidgetTarget::for_dropped_file("validation", {"table:validation"}, "source:table:validation"),
          "Plot the validation loss");
    REQUIRE(r.surface->state() == WidgetState::preview);
    r.surface->accept();
    CHECK(r.surface->state() == WidgetState::applied);
    CHECK(r.undo_steps() == r.steps_before + 2);
    REQUIRE(r.s->open_spec());
    CHECK(r.s->open_spec()->spec.find_panel("val_loss"));
    ok(*r.s, "history.undo");   // the change
    REQUIRE(r.s->open_spec());
    CHECK(r.s->open_spec()->spec.panels.empty());
    ok(*r.s, "history.undo");   // the spec.new
    CHECK(!r.s->open_spec());
}

TEST_CASE("widget targets: a property's edit is previewed in its panel") {
    WidgetScriptTurn t;
    t.steps = {edit(100, "panel.patch", {{"id", "attention"}, {"patch", {{"options", {{"colormap", "rdbu"}}}}}}),
               message(200, "The attention uses the diverging colormap.")};
    t.end_at = milliseconds(300);
    Rig r("property", {t});
    const WidgetTarget target = WidgetTarget::for_property("attention", "", "options.colormap");
    CHECK(target.anchor == "prop:options.colormap");
    CHECK(target.frame_panel().empty());
    CHECK(target.preview_scope() == json{{"panel", "attention"}});
    r.ask(target, "Use the diverging colormap");
    REQUIRE(r.surface->state() == WidgetState::preview);
    CHECK(r.agent->scopes()[0] == r.rule_scope("panel_property"));
    CHECK(r.agent->contexts()[0].dump().find("options.colormap") != std::string::npos);
    REQUIRE(r.sink.panels == std::vector<std::string>{"attention"});
    CHECK(r.sink.specs.empty());
    REQUIRE(r.surface->diff().size() == 1);
    CHECK(r.surface->diff()[0].key == "colormap");
    CHECK(r.surface->diff()[0].after == "\"rdbu\"");
    r.surface->accept();
    CHECK(r.undo_steps() == r.steps_before + 1);
}

TEST_CASE("widget targets: a line whose y is shorter than its x gives NaN in the selection's rows") {
    PreparedPanel p;
    p.id = "curves";
    p.primitive = caliper_builder::spec::Primitive::series;
    SeriesLine full, ragged;
    full.name = "full";
    ragged.name = "ragged";
    for (int i = 0; i < 4000; ++i) {
        full.x.push_back(i);
        full.y.push_back(1.0);
    }
    // Fewer steps than the first line, so its values are looked up by x, and
    // only two values for them.
    for (int i = 0; i < 3000; ++i) ragged.x.push_back(i);
    ragged.y = {7.0, 8.0};
    p.series.lines = {full, ragged};
    SpecSelection sel;
    sel.kind = SpecSelection::Kind::range;
    sel.panel = "curves";
    sel.x0 = 0.0;
    sel.x1 = 2999.0;
    const SpecSelectionRows rows = spec_selection_rows(p, sel, 4000);
    REQUIRE(rows.rows.size() == 3000);
    CHECK(rows.rows[0][2] == 7.0);
    CHECK(rows.rows[1][2] == 8.0);
    size_t beyond = 0;
    for (size_t i = 2; i < rows.rows.size(); ++i) beyond += std::isnan(rows.rows[i][2]) ? 0 : 1;
    CHECK(beyond == 0);   // no value where the line has none
}

TEST_CASE("widget targets: the runtime's selection gives its rows from the prepared panel") {
    PreparedPanel p;
    p.id = "curves";
    p.primitive = caliper_builder::spec::Primitive::series;
    p.series.x_name = "step";
    SeriesLine loss, acc;
    loss.name = "loss";
    acc.name = "accuracy";
    for (int i = 0; i < 100; ++i) {
        loss.x.push_back(i);
        loss.y.push_back(1.0 / (i + 1));
        acc.x.push_back(i);
        acc.y.push_back(i / 100.0);
    }
    p.series.lines = {loss, acc};
    SpecSelection sel;
    sel.kind = SpecSelection::Kind::range;
    sel.panel = "curves";
    sel.x0 = 10.5;
    sel.x1 = 20.0;
    const SpecSelectionRows rows = spec_selection_rows(p, sel, 1000);
    CHECK(rows.columns == std::vector<std::string>{"step", "loss", "accuracy"});
    REQUIRE(rows.rows.size() == 10);   // 11 .. 20
    CHECK(rows.total == 10);
    CHECK(rows.rows[0][0] == 11.0);
    CHECK(rows.rows[0][2] == doctest::Approx(0.11));
    CHECK(spec_selection_rows(p, sel, 3).rows.size() == 3);
    CHECK(spec_selection_rows(p, sel, 3).total == 10);
    sel.panel = "other";
    CHECK(spec_selection_rows(p, sel).rows.empty());
    // Points inside a rectangle.
    PreparedPanel q;
    q.id = "scatter";
    q.primitive = caliper_builder::spec::Primitive::points;
    q.points.x = {0, 1, 2, 3};
    q.points.y = {0, 1, 2, 3};
    SpecSelection box;
    box.kind = SpecSelection::Kind::points;
    box.panel = "scatter";
    box.x0 = 0.5;
    box.x1 = 2.5;
    box.y0 = 0.5;
    box.y1 = 3.5;
    const SpecSelectionRows pts = spec_selection_rows(q, box);
    CHECK(pts.rows.size() == 2);
    CHECK(pts.columns.size() == 2);
}

TEST_CASE("widget layout: an answer without a change wraps its reply over several lines") {
    WidgetLayoutInput in;
    in.panel = {0, 0, 400, 600};
    in.strip = 0;
    in.state = WidgetState::preview;
    in.reply = true;
    in.reply_lines = 4;
    in.card_actions = {WidgetAction::refine, WidgetAction::close};
    const auto m = WidgetLayoutMetrics::for_text_scale(1.0f);
    const WidgetLayout L = layout_widget_surface(in, m, [](const std::string& t, WidgetFont) { return 7.0f * (float)t.size(); });
    CHECK(L.reply.h == doctest::Approx(4 * m.line));
    in.reply_lines = 1;
    const WidgetLayout one = layout_widget_surface(in, m, [](const std::string& t, WidgetFont) { return 7.0f * (float)t.size(); });
    CHECK(L.band == doctest::Approx(one.band + 3 * m.line));
}
