// The widget agent's inline surface without a window and without an agent:
// its state machine driven by input events and by the scripted agent's
// events on a hand-moved clock, over a real builder Session that holds the
// dashboard of dashboard.spec.toml with a Parquet table and a safetensors
// tensor. Every path: accept, reject, refine twice, cancel while running, a
// refusal, a handoff, no provider, an agent that exits, a preview that
// conflicts because the spec changed underneath, a session busy with a long
// command; and the preview's life on the command layer: the open spec
// unchanged until Accept, one undo step after it, journaled under the agent
// origin with the preview's group, and nothing left after Reject.
#include "builder/widget_agent.h"
#include "builder/widget_surface.h"

#include "spec_test_workspace.h"

#include <caliper_builder/commands/journal.h>
#include <caliper_builder/spec/format.h>

#include <doctest/doctest.h>

#include <algorithm>
#include <chrono>
#include <map>
#include <memory>
#include <string>
#include <vector>

using namespace caliper_host;
using namespace spec_test;
using caliper_builder::CommandOrigin;
using nlohmann::json;
using std::chrono::milliseconds;

namespace {

// Records what the surface asked to draw.
struct FakeSink : WidgetPreviewSink {
    std::vector<std::string> shown, cleared;
    json last;   // the panel last shown, as JSON
    void show(const caliper_builder::spec::Panel& panel, const caliper_builder::PanelData& data,
              const caliper_builder::spec::Spec&) override {
        shown.push_back(panel.id);
        last = caliper_builder::spec::to_json(panel);
        CHECK(data.panel == panel.id);
        CHECK(!data.error);
    }
    void clear(const std::string& panel) override { cleared.push_back(panel); }
};

// A session that can be marked busy, as AppSession::worker_busy() does.
struct BusySession : SessionWidgetSession {
    using SessionWidgetSession::SessionWidgetSession;
    bool busy = false;
    bool ready() const override { return !busy; }
};

WidgetScriptStep step(int ms, WidgetScriptStep::Kind kind, std::string text = {}) {
    WidgetScriptStep s;
    s.at = milliseconds(ms);
    s.kind = kind;
    s.text = std::move(text);
    return s;
}

WidgetScriptStep patch(int ms, json p, std::string panel = "curves") {
    WidgetScriptStep s;
    s.at = milliseconds(ms);
    s.kind = WidgetScriptStep::Kind::edit;
    s.edit.name = "panel.patch";
    s.edit.args = {{"id", panel}, {"patch", std::move(p)}};
    return s;
}

WidgetScriptTurn edit_turn(json p, std::string reply) {
    WidgetScriptTurn t;
    t.steps = {step(100, WidgetScriptStep::Kind::progress, "spec.get"), patch(300, std::move(p)),
               step(400, WidgetScriptStep::Kind::message, std::move(reply))};
    t.end_at = milliseconds(500);
    return t;
}

WidgetScriptTurn linear_turn() { return edit_turn({{"options", {{"y_log", false}}}}, "Turns off the log scale."); }

struct Rig {
    std::string root, ws;
    std::unique_ptr<caliper_builder::Session> s;
    std::chrono::steady_clock::time_point t = std::chrono::steady_clock::now();
    std::unique_ptr<ScriptedWidgetAgent> agent;
    std::unique_ptr<BusySession> host;
    FakeSink sink;
    std::unique_ptr<WidgetSurface> surface;
    std::vector<WidgetChatRequest> chats;
    int settings = 0;
    std::string before;
    size_t steps_before = 0;

    Rig(const std::string& name, WidgetScript script) {
        root = fresh_dir("widget-" + name);
        ws = root + "/ws";
        s = dashboard_workspace(ws, root + "/data");
        agent = std::make_unique<ScriptedWidgetAgent>(
            std::move(script), [this](caliper_builder::CommandInvocation inv) { return s->invoke(std::move(inv)); },
            [this] { return t; }, [this](const std::string& id) { return (int)s->preview(id).edits; });
        host = std::make_unique<BusySession>(*s);
        WidgetSurface::Hooks hooks;
        hooks.now = [this] { return t; };
        hooks.ask_in_chat = [this](const WidgetChatRequest& r) { chats.push_back(r); };
        hooks.open_settings = [this] { ++settings; };
        surface = std::make_unique<WidgetSurface>(*host, *agent, hooks);
        surface->set_preview_sink(&sink);
        before = s->open_spec()->text;
        steps_before = undo_steps();
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
    size_t undo_steps() const { return s->registry().history_state().undo.size(); }
    std::string open_text() const { return s->open_spec()->text; }
    // Opens the prompt on curves, types and submits.
    void ask(const std::string& text) {
        if (surface->state() == WidgetState::idle) REQUIRE(surface->open(WidgetTarget::for_panel("curves")));
        surface->set_text(text);
        surface->submit();
    }
};

WidgetScript script_of(std::vector<WidgetScriptTurn> turns) {
    WidgetScript s;
    s.start_after = milliseconds(100);
    s.turns = std::move(turns);
    return s;
}

}  // namespace

TEST_CASE("widget surface: Accept commits the preview as one undo step journaled under the agent") {
    Rig r("accept", script_of({linear_turn()}));
    CHECK(r.surface->state() == WidgetState::idle);
    CHECK_FALSE(r.surface->open(WidgetTarget::for_panel("nope")));   // not in the open spec
    REQUIRE(r.surface->open(WidgetTarget::for_panel("curves")));
    CHECK(r.surface->state() == WidgetState::prompt);
    CHECK(r.surface->row_actions() == std::vector<WidgetAction>{WidgetAction::ask_in_chat});
    r.surface->submit();   // an empty prompt does nothing
    CHECK(r.surface->state() == WidgetState::prompt);
    r.surface->set_text("  Use a linear y axis ");
    r.surface->submit();
    CHECK(r.surface->state() == WidgetState::running);
    CHECK(r.surface->request() == "Use a linear y axis");
    const std::string preview = r.surface->preview_id();
    REQUIRE(preview.rfind("widget-", 0) == 0);
    CHECK(r.agent->previews() == std::vector<std::string>{preview});
    CHECK(r.s->preview(preview).panel == "curves");   // scoped to the target
    // The context bundle went before the first request, its scope with the session.
    CHECK_FALSE(r.surface->bundle_error());
    REQUIRE(r.agent->contexts().size() == 1);
    REQUIRE(r.agent->contexts()[0].is_array());
    CHECK(r.agent->contexts()[0].size() >= 2);
    CHECK(r.agent->contexts()[0].dump().find("curves") != std::string::npos);
    CHECK(r.agent->contexts()[0].dump().find("run42_metrics") != std::string::npos);
    REQUIRE(r.agent->scopes().size() == 1);
    const auto& scope = r.agent->scopes()[0];
    CHECK(std::find(scope.begin(), scope.end(), "panel.patch") != scope.end());
    CHECK(std::find(scope.begin(), scope.end(), "preview.commit") == scope.end());

    r.run(250);   // started at 100 ms, spec.get at 200 ms
    CHECK(r.surface->state() == WidgetState::running);
    CHECK(r.surface->progress() == "spec.get");
    CHECK(r.surface->model_label() == "Scripted");
    r.run(500);
    REQUIRE(r.surface->state() == WidgetState::preview);
    CHECK(r.surface->edits() == 1);
    CHECK(r.surface->reply() == "Turns off the log scale.");
    REQUIRE(r.surface->diff().size() == 1);
    CHECK(r.surface->diff()[0].key == "y_log");
    CHECK(r.surface->diff()[0].before == "true");
    CHECK(r.surface->diff()[0].after == "false");
    CHECK(r.surface->full_diff().find("y_log = true") != std::string::npos);
    CHECK(r.surface->card_actions() ==
          std::vector<WidgetAction>{WidgetAction::accept, WidgetAction::reject, WidgetAction::refine, WidgetAction::diff});
    // Drawn in place from the preview; the open spec, the history and the
    // journal have not moved.
    REQUIRE(r.sink.shown == std::vector<std::string>{"curves"});
    CHECK(r.sink.last["options"]["y_log"] == false);
    CHECK(r.open_text() == r.before);
    CHECK(r.undo_steps() == r.steps_before);
    const size_t journal_before = caliper_builder::read_journal(r.ws + "/journal/commands.jsonl").events.size();

    r.surface->accept();
    CHECK(r.surface->state() == WidgetState::applied);
    CHECK(r.surface->preview_id().empty());
    CHECK(r.sink.cleared == std::vector<std::string>{"curves"});
    CHECK(r.agent->sessions_closed() == 1);
    CHECK(r.open_text().find("y_log = false") != std::string::npos);
    CHECK(r.undo_steps() == r.steps_before + 1);
    const auto journal = caliper_builder::read_journal(r.ws + "/journal/commands.jsonl");
    int agent_edits = 0, commits = 0;
    for (size_t i = journal_before; i < journal.events.size(); ++i) {
        const auto& e = journal.events[i];
        if (e.name == "preview.commit") {
            ++commits;
            CHECK(e.origin == CommandOrigin::ui);
        }
        if (e.name == "panel.patch") {
            ++agent_edits;
            CHECK(e.origin == CommandOrigin::agent);
            CHECK(e.group == preview);
        }
    }
    CHECK(commits == 1);
    CHECK(agent_edits == 1);
    // One undo takes it back.
    ok(*r.s, "history.undo");
    CHECK(r.open_text() == r.before);
    r.surface->escape();
    CHECK(r.surface->state() == WidgetState::idle);
}

TEST_CASE("widget surface: Reject and Escape discard the preview and leave no trace") {
    Rig r("reject", script_of({linear_turn(), linear_turn()}));
    r.ask("Use a linear y axis");
    r.run(700);
    REQUIRE(r.surface->state() == WidgetState::preview);
    const std::string preview = r.surface->preview_id();
    const size_t journal_before = caliper_builder::read_journal(r.ws + "/journal/commands.jsonl").events.size();
    r.surface->reject();
    CHECK(r.surface->state() == WidgetState::idle);
    CHECK(r.s->preview(preview).error);   // gone
    CHECK(r.sink.cleared == std::vector<std::string>{"curves"});
    CHECK(r.agent->sessions_closed() == 1);
    CHECK(r.open_text() == r.before);
    CHECK(r.undo_steps() == r.steps_before);
    const auto journal = caliper_builder::read_journal(r.ws + "/journal/commands.jsonl");
    for (size_t i = journal_before; i < journal.events.size(); ++i) CHECK(journal.events[i].name != "panel.patch");

    // Escape on the card is Reject.
    r.ask("Use a linear y axis");
    r.run(700);
    REQUIRE(r.surface->state() == WidgetState::preview);
    const std::string second = r.surface->preview_id();
    CHECK(second != preview);
    r.surface->escape();
    CHECK(r.surface->state() == WidgetState::idle);
    CHECK(r.s->preview(second).error);
    CHECK(r.open_text() == r.before);
}

TEST_CASE("widget surface: Refine twice stacks on the same preview in the same agent session") {
    Rig r("refine", script_of({linear_turn(), edit_turn({{"title", "Training curves"}}, "Renames it."),
                               edit_turn({{"options", {{"x_label", "Training step"}}}}, "Names the x axis.")}));
    r.ask("Use a linear y axis");
    r.run(700);
    REQUIRE(r.surface->state() == WidgetState::preview);
    const std::string preview = r.surface->preview_id();

    r.surface->refine();
    CHECK(r.surface->state() == WidgetState::prompt);
    CHECK(r.surface->refining());
    CHECK(r.surface->text().empty());
    // Escape from the follow-up goes back to the card, the preview kept.
    r.surface->escape();
    CHECK(r.surface->state() == WidgetState::preview);
    CHECK(r.surface->preview_id() == preview);

    r.surface->refine();
    r.surface->set_text("Call it Training curves");
    r.surface->submit();
    CHECK(r.surface->state() == WidgetState::running);
    r.run(700);
    REQUIRE(r.surface->state() == WidgetState::preview);
    CHECK(r.surface->edits() == 2);
    CHECK(r.surface->diff().size() == 2);

    r.surface->refine();
    r.surface->set_text("Name the x axis Training step");
    r.surface->submit();
    r.run(700);
    REQUIRE(r.surface->state() == WidgetState::preview);
    CHECK(r.surface->edits() == 3);
    CHECK(r.surface->preview_id() == preview);
    CHECK(r.agent->sessions_started() == 1);
    CHECK(r.agent->prompts().size() == 3);
    // Only the session's first prompt carries the bundle.
    CHECK(r.agent->contexts()[0].is_array());
    CHECK(r.agent->contexts()[1].is_null());
    CHECK(r.agent->contexts()[2].is_null());
    CHECK(r.surface->conversation().size() == 3);
    CHECK(r.surface->conversation()[1].reply == "Renames it.");
    CHECK(r.open_text() == r.before);

    r.surface->accept();
    CHECK(r.surface->state() == WidgetState::applied);
    CHECK(r.undo_steps() == r.steps_before + 1);   // three edits, one step
    const std::string after = r.open_text();
    CHECK(after.find("y_log = false") != std::string::npos);
    CHECK(after.find("Training curves") != std::string::npos);
    CHECK(after.find("Training step") != std::string::npos);
    ok(*r.s, "history.undo");
    CHECK(r.open_text() == r.before);
}

TEST_CASE("widget surface: cancel while running, by Escape or by the action, discards the preview") {
    WidgetScriptTurn slow = linear_turn();
    slow.end_at = milliseconds(5000);
    Rig r("cancel", script_of({slow, slow}));
    r.ask("Use a linear y axis");
    r.run(600);   // the edit has landed on the preview
    REQUIRE(r.surface->state() == WidgetState::running);
    const std::string preview = r.surface->preview_id();
    CHECK(r.s->preview(preview).edits == 1);
    CHECK(r.surface->row_actions() == std::vector<WidgetAction>{WidgetAction::cancel});
    r.surface->escape();
    CHECK(r.surface->state() == WidgetState::prompt);
    CHECK_FALSE(r.surface->refining());
    CHECK(r.surface->text() == "Use a linear y axis");   // kept to send again
    CHECK(r.agent->cancels() == 1);
    CHECK(r.agent->sessions_closed() == 1);
    CHECK(r.s->preview(preview).error);
    CHECK(r.open_text() == r.before);
    r.run(6000);   // nothing arrives from the cancelled turn
    CHECK(r.surface->state() == WidgetState::prompt);

    // Again, with the Cancel action: a new preview and a new session.
    r.surface->submit();
    CHECK(r.surface->preview_id() != preview);
    CHECK(r.agent->sessions_started() == 2);
    r.run(200);
    r.surface->act(WidgetAction::cancel);
    CHECK(r.surface->state() == WidgetState::prompt);
    CHECK(r.agent->cancels() == 2);
    CHECK(r.undo_steps() == r.steps_before);
}

TEST_CASE("widget surface: a refusal is declined with three ways forward") {
    WidgetScriptTurn refused;
    refused.steps = {step(100, WidgetScriptStep::Kind::message, "I can't help with that.")};
    refused.end_at = milliseconds(200);
    refused.stop_reason = "refusal";
    Rig r("declined", script_of({refused, refused, linear_turn()}));
    r.ask("Use a linear y axis");
    r.run(500);
    REQUIRE(r.surface->state() == WidgetState::declined);
    CHECK(r.surface->tag() == "DECLINED");
    CHECK(r.surface->card_actions() ==
          std::vector<WidgetAction>{WidgetAction::rephrase, WidgetAction::retry, WidgetAction::model});
    CHECK(r.surface->row_actions() == std::vector<WidgetAction>{WidgetAction::ask_in_chat});
    CHECK(r.surface->conversation().empty());   // the declined prompt leaves the conversation
    CHECK(r.sink.shown.empty());
    CHECK(r.open_text() == r.before);

    r.surface->act(WidgetAction::model);
    CHECK(r.settings == 1);
    r.surface->retry();   // the same request again, in the same session
    CHECK(r.surface->state() == WidgetState::running);
    r.run(500);
    REQUIRE(r.surface->state() == WidgetState::declined);
    r.surface->rephrase();
    CHECK(r.surface->state() == WidgetState::prompt);
    CHECK(r.surface->text() == "Use a linear y axis");
    r.surface->set_text("Turn the log scale of the y axis off");
    r.surface->submit();
    r.run(700);
    CHECK(r.surface->state() == WidgetState::preview);
    CHECK(r.agent->sessions_started() == 1);
    CHECK(r.agent->prompts().size() == 3);
}

TEST_CASE("widget surface: a handoff drops the preview and Ask in chat carries the conversation") {
    WidgetScriptTurn handoff = linear_turn();   // an edit first, then the agent hands the request over
    handoff.handoff = WidgetHandoff{"loader", "A loader is work for the side chat."};
    Rig r("handoff", script_of({handoff}));
    r.ask("Write a loader for the validation logs");
    r.run(700);
    REQUIRE(r.surface->state() == WidgetState::handoff);
    CHECK(r.surface->tag() == "LOADER");
    CHECK(r.surface->preview_id().empty());
    CHECK(r.agent->sessions_closed() == 1);
    CHECK(r.surface->card_actions().front() == WidgetAction::ask_in_chat);
    CHECK(r.open_text() == r.before);
    r.surface->ask_in_chat();
    CHECK(r.surface->state() == WidgetState::idle);
    REQUIRE(r.chats.size() == 1);
    CHECK(r.chats[0].request == "Write a loader for the validation logs");
    CHECK(r.chats[0].target.panel == "curves");
    REQUIRE(r.chats[0].handoff);
    CHECK(r.chats[0].handoff->kind == "loader");
    REQUIRE(r.chats[0].conversation.size() == 1);
    CHECK(r.chats[0].conversation[0].reply == "Turns off the log scale.");
}

TEST_CASE("widget surface: with no provider the prompt offers only Ask in chat, and no agent starts") {
    WidgetScript none = script_of({});
    none.available = false;
    Rig r("unavailable", none);
    REQUIRE(r.surface->open(WidgetTarget::for_panel("curves")));
    CHECK(r.surface->state() == WidgetState::unavailable);
    CHECK(r.surface->row_actions() == std::vector<WidgetAction>{WidgetAction::setup, WidgetAction::ask_in_chat});
    CHECK(r.surface->card_actions().empty());
    r.surface->act(WidgetAction::setup);
    CHECK(r.settings == 1);
    r.surface->set_text("Use a linear y axis");
    r.surface->submit();   // Enter is Ask in chat
    CHECK(r.surface->state() == WidgetState::idle);
    REQUIRE(r.chats.size() == 1);
    CHECK(r.chats[0].request == "Use a linear y axis");
    CHECK(r.chats[0].diff.empty());
    CHECK(r.agent->sessions_started() == 0);
    CHECK(r.open_text() == r.before);
}

TEST_CASE("widget surface: an agent that exits fails the turn, and Retry starts a new session") {
    WidgetScriptTurn crash = linear_turn();
    crash.exit = true;
    Rig r("exits", script_of({crash, linear_turn()}));
    r.ask("Use a linear y axis");
    r.run(700);
    REQUIRE(r.surface->state() == WidgetState::failed);
    CHECK(r.surface->tag() == "STOPPED");
    CHECK(r.surface->error().code == "acp.agent_exited");
    CHECK(r.surface->preview_id().empty());
    CHECK(r.surface->card_actions() == std::vector<WidgetAction>{WidgetAction::retry, WidgetAction::close});
    CHECK(r.open_text() == r.before);
    r.surface->retry();
    CHECK(r.agent->sessions_started() == 2);
    r.run(700);
    CHECK(r.surface->state() == WidgetState::preview);
    r.surface->close();
    CHECK(r.surface->state() == WidgetState::idle);
}

TEST_CASE("widget surface: a preview conflicts when the spec changes underneath, and Refine begins anew") {
    Rig r("conflict", script_of({linear_turn(), linear_turn(), linear_turn()}));
    r.ask("Use a linear y axis");
    r.run(700);
    REQUIRE(r.surface->state() == WidgetState::preview);

    // A change that leaves the edit applicable: the preview follows the open spec.
    ok(*r.s, "panel.patch", {{"id", "attention"}, {"patch", {{"title", "Attention"}}}});
    r.surface->update();
    CHECK(r.surface->state() == WidgetState::preview);
    CHECK(r.surface->diff().size() == 1);

    // The panel goes away under the preview: its edit no longer applies.
    const std::string first = r.surface->preview_id();
    ok(*r.s, "panel.remove", {{"id", "curves"}});
    r.surface->update();
    REQUIRE(r.surface->state() == WidgetState::conflict);
    CHECK(r.surface->tag() == "CONFLICT");
    CHECK(r.surface->card_actions().front() == WidgetAction::refine);
    ok(*r.s, "history.undo");   // the panel is back
    r.surface->refine();
    CHECK(r.surface->state() == WidgetState::prompt);
    CHECK(r.surface->text() == "Use a linear y axis");
    CHECK(r.s->preview(first).error);
    r.surface->submit();
    CHECK(r.surface->preview_id() != first);
    CHECK(r.agent->sessions_started() == 2);
    r.run(700);
    REQUIRE(r.surface->state() == WidgetState::preview);

    // Accept itself meets the conflict when the change lands between two frames.
    ok(*r.s, "panel.remove", {{"id", "curves"}});
    const size_t steps = r.undo_steps();
    r.surface->accept();
    CHECK(r.surface->state() == WidgetState::conflict);
    CHECK(r.surface->error().code == "history.conflict");
    CHECK(r.undo_steps() == steps);
    r.surface->reject();
    CHECK(r.surface->state() == WidgetState::idle);
}

TEST_CASE("widget surface: what needs the session waits while a long command holds it") {
    Rig r("busy", script_of({linear_turn()}));
    REQUIRE(r.surface->open(WidgetTarget::for_panel("curves")));
    r.surface->set_text("Use a linear y axis");
    r.host->busy = true;
    r.surface->submit();
    CHECK(r.surface->state() == WidgetState::running);
    CHECK(r.surface->preview_id().empty());   // preview.begin waits
    CHECK(r.agent->sessions_started() == 0);
    r.run(200);
    CHECK(r.surface->preview_id().empty());
    r.host->busy = false;
    r.surface->update();
    CHECK_FALSE(r.surface->preview_id().empty());
    CHECK(r.agent->sessions_started() == 1);
    r.run(700);
    REQUIRE(r.surface->state() == WidgetState::preview);
    r.host->busy = true;
    r.surface->accept();   // the commit waits too
    CHECK(r.surface->state() == WidgetState::preview);
    CHECK(r.open_text() == r.before);
    r.host->busy = false;
    r.surface->update();
    CHECK(r.surface->state() == WidgetState::applied);
    CHECK(r.undo_steps() == r.steps_before + 1);
}

TEST_CASE("widget surface: the compact diff names changed, added and removed keys") {
    const caliper_builder::spec::Spec a = dashboard_fixture();
    caliper_builder::spec::Spec b = a;
    caliper_builder::spec::Panel* curves = nullptr;
    for (auto& p : b.panels)
        if (p.id == "curves") curves = &p;
    REQUIRE(curves);
    auto j = caliper_builder::spec::to_json(*curves);
    j["options"].erase("y_log");
    j["options"]["x_range"] = {50, 150};
    j["title"] = "Training curves";
    auto decoded = caliper_builder::spec::panel_from_json(j);
    REQUIRE(decoded.ok());
    *curves = decoded.value;
    b.panels.erase(std::remove_if(b.panels.begin(), b.panels.end(), [](const auto& p) { return p.id == "rows"; }),
                   b.panels.end());
    const auto d = widget_compact_diff(a, b, "curves");
    std::map<std::string, WidgetDiffLine> by;
    for (const auto& l : d) by[l.key] = l;
    REQUIRE(by.count("y_log"));
    CHECK(by["y_log"].kind == WidgetDiffLine::Kind::removed);
    CHECK(by["y_log"].before == "true");
    REQUIRE(by.count("x_range"));
    CHECK(by["x_range"].kind == WidgetDiffLine::Kind::added);
    CHECK(by["x_range"].after == "[50, 150]");
    REQUIRE(by.count("title"));
    CHECK(by["title"].before == "\"Loss and accuracy\"");
    CHECK(by["title"].after == "\"Training curves\"");
    REQUIRE(by.count("panel rows"));
    CHECK(by["panel rows"].kind == WidgetDiffLine::Kind::removed);
    CHECK(d.front().key.find(' ') == std::string::npos);   // the target's keys come first
    // Long values are cut.
    j["title"] = std::string(200, 'a');
    *curves = caliper_builder::spec::panel_from_json(j).value;
    for (const auto& l : widget_compact_diff(a, b, "curves", 20))
        if (l.key == "title") CHECK(l.after.size() <= 22);
}

TEST_CASE("widget surface: InlineWidget, the page's one object, owns the session adapter, the agent and the surface") {
    const std::string root = fresh_dir("widget-inline");
    auto s = dashboard_workspace(root + "/ws", root + "/data");
    const std::string before = s->open_spec()->text;
    auto t = std::chrono::steady_clock::now();
    bool busy = false;
    InlineWidget::Options o;
    o.session = s.get();
    o.busy = [&busy] { return busy; };
    o.agent = std::make_unique<ScriptedWidgetAgent>(
        script_of({linear_turn()}), [&s](caliper_builder::CommandInvocation inv) { return s->invoke(std::move(inv)); },
        [&t] { return t; });
    o.hooks.now = [&t] { return t; };
    std::string preview;
    {
        InlineWidget w(std::move(o));
        REQUIRE(w.surface().open(WidgetTarget::for_panel("curves")));
        w.surface().set_text("Use a linear y axis");
        busy = true;   // the page's worker holds the session: nothing runs yet
        w.surface().submit();
        CHECK(w.surface().preview_id().empty());
        busy = false;
        for (int i = 0; i < 20; ++i) {
            t += milliseconds(50);
            w.surface().update();
        }
        REQUIRE(w.surface().state() == WidgetState::preview);
        preview = w.surface().preview_id();
        CHECK(s->preview(preview).edits == 1);
    }
    // Dropping the object discards its open preview.
    CHECK(s->preview(preview).error);
    CHECK(s->open_spec()->text == before);
    s->close();
}

TEST_CASE("widget surface: the agent's thoughts show as the progress line, and a model is chosen from the row") {
    WidgetScriptTurn thinking = linear_turn();
    thinking.steps.insert(thinking.steps.begin(),
                          step(50, WidgetScriptStep::Kind::thought, "Reading the panel.\nThe y axis is on a log scale"));
    WidgetScript script = script_of({thinking, linear_turn()});
    script.models = {{"claude-haiku-5-5", "Claude Haiku 5.5", false}, {"claude-opus-5-5", "Claude Opus 5.5", false}};
    script.model = "Claude Haiku 5.5";
    Rig r("models", script);
    CHECK(r.agent->models().empty());   // nothing to choose before a session
    r.ask("Use a linear y axis");
    r.run(150);   // started at 100 ms, the thought at 150 ms
    REQUIRE(r.surface->state() == WidgetState::running);
    CHECK(r.surface->progress() == "The y axis is on a log scale");
    r.run(500);   // spec.get, then the edit: a command name replaces the thought
    REQUIRE(r.surface->state() == WidgetState::preview);
    const auto choices = r.agent->models();
    REQUIRE(choices.size() == 2);
    CHECK(choices[0].current);
    r.agent->set_model("claude-opus-5-5");
    r.surface->update();
    CHECK(r.surface->model_label() == "Claude Opus 5.5");
    CHECK(r.agent->models()[1].current);
}
