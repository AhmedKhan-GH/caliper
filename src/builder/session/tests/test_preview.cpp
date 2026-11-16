// Tests of stacked previews: three edits that build on each other stacked on
// a preview from the agent origin while the open spec, the history and the
// journal stay as they were; the preview read by spec.get, preview.diff,
// panel.render and Session::resolve(); a commit from the ui origin as one
// undo step that one undo reverts and the journal replays; a commit refused
// to every agent origin; a discard that leaves no trace; a stack that
// survives a change of the open spec and one that conflicts; expiry and the
// bounds; the scope of a preview of one panel; and the X-Caliper-Preview
// header of the stand-in's server.
#include <caliper_builder/session/command_server.h>
#include <caliper_builder/session/session.h>

#include <caliper_builder/commands/journal.h>
#include <caliper_builder/spec/format.h>

#include <doctest/doctest.h>

#include <atomic>
#include <chrono>
#include <thread>

#include "helpers.h"

using namespace caliper_builder;
using namespace session_test;
using nlohmann::json;

namespace {

const CommandOrigin kUi = CommandOrigin::ui;
const CommandOrigin kAgent = CommandOrigin::agent;

// A clock the tests move by hand.
struct Clock {
    std::chrono::steady_clock::time_point now = std::chrono::steady_clock::now();
};

CommandResult in_preview(Session& s, const std::string& preview, const std::string& name, json args,
                         CommandOrigin origin = kAgent) {
    CommandInvocation inv;
    inv.name = name;
    inv.args = std::move(args);
    inv.origin = origin;
    inv.preview = preview;
    return s.invoke(std::move(inv));
}

json ok_in(Session& s, const std::string& preview, const std::string& name, json args, CommandOrigin origin = kAgent) {
    const CommandResult r = in_preview(s, preview, name, std::move(args), origin);
    INFO(name << ": " << r.error.code << " at " << r.error.where << ": " << r.error.message);
    REQUIRE(r.ok);
    return r.result;
}

std::string err_in(Session& s, const std::string& preview, const std::string& name, json args,
                   CommandOrigin origin = kAgent) {
    const CommandResult r = in_preview(s, preview, name, std::move(args), origin);
    INFO(name << " answered " << (r.ok ? r.result.dump() : r.error.message));
    CHECK_FALSE(r.ok);
    return r.error.code;
}

// A workspace with a metrics table and a 2 by 3 tensor, and an open spec of
// one series panel over the table.
struct Fixture {
    std::string ws;
    std::shared_ptr<Clock> clock = std::make_shared<Clock>();
    std::unique_ptr<Session> s;

    explicit Fixture(const std::string& name, PreviewLimits limits = {}) {
        ws = fresh_dir(name) + "/workspace";
        write_text(ws + "/data/metrics.csv", "step,loss\n0,1.0\n1,0.5\n2,0.25\n");
        write_safetensors_f32(ws + "/data/heat.safetensors", "w", {2, 3}, {1, 2, 3, 4, 5, 6});
        SessionOptions options;
        options.registry.check_results = true;
        options.preview = limits;
        auto c = clock;
        options.preview.now = [c] { return c->now; };
        auto opened = Session::open(ws, options);
        REQUIRE_MESSAGE(opened.error.ok(), opened.error.message);
        s = std::move(opened.session);
        ok(*s, "data.import", {{"paths", {"data/metrics.csv", "data/heat.safetensors"}}});
        ok(*s, "spec.new", {{"id", "dash"}, {"title", "Dashboard"}});
        ok(*s, "source.add", {{"id", "m"}, {"catalog", "table:metrics"}});
        ok(*s, "panel.create", {{"id", "loss"}, {"primitive", "series"}, {"bind", {{"x", "m.step"}, {"y", "m.loss"}}}});
    }

    std::size_t journal_lines() const { return read_journal(ws + "/journal/commands.jsonl").events.size(); }
    std::size_t steps() const { return s->registry().history_state().undo.size(); }
    std::string text() const { return s->open_spec()->text; }

    // Three edits that build on each other: a tensor source, a derived source
    // and a heatmap of it placed beside the loss panel.
    void stack_three(const std::string& preview) {
        ok_in(*s, preview, "source.add", {{"id", "h"}, {"catalog", "tensor:heat_w"}});
        ok_in(*s, preview, "source.add",
              {{"id", "hn"}, {"from", "h"}, {"transform", {{{"op", "normalise"}, {"mode", "minmax"}}}}});
        ok_in(*s, preview, "panel.create", {{"id", "heat"}, {"primitive", "heatmap"}, {"bind", {{"values", "hn"}}}});
    }
};

}  // namespace

TEST_CASE("three edits that build on each other stack on a preview and commit as one undo step") {
    Fixture f("preview-commit");
    const std::string before = f.text();
    const std::size_t lines = f.journal_lines();
    const std::size_t steps = f.steps();

    const json begun = ok(*f.s, "preview.begin", {{"id", "widget-1"}, {"panel", "loss"}}, kUi);
    CHECK(begun == json{{"id", "widget-1"}, {"panel", "loss"}, {"source", nullptr}});
    // The second edit binds the first's source and the third binds the
    // second's, so a dry run of each against the open spec would fail.
    CHECK(err(*f.s, "source.add", {{"id", "hn"}, {"from", "h"}, {"dry_run", true}}, kAgent) == "spec.unknown_source");
    f.stack_three("widget-1");
    const json placed = ok_in(*f.s, "widget-1", "layout.place",
                              {{"panel", "heat"}, {"target", "loss"}, {"where", "beside"}, {"ratio", 0.4}});
    CHECK(placed["applied"] == true);
    CHECK(placed["diff"].get<std::string>().find("heat") != std::string::npos);

    // Nothing reached the open spec, the history or the journal.
    CHECK(f.text() == before);
    CHECK(f.journal_lines() == lines);
    CHECK(f.steps() == steps);

    // The preview is visible to spec.get, preview.diff, resolve and panel.render.
    const json got = ok_in(*f.s, "widget-1", "spec.get", json::object());
    CHECK(got["json"]["panel"].size() == 2);
    CHECK(got["dirty"] == true);
    CHECK(ok(*f.s, "spec.get")["json"]["panel"].size() == 1);
    const json diff = ok(*f.s, "preview.diff", {{"id", "widget-1"}}, kAgent);
    CHECK(diff["edits"] == 4);
    CHECK(diff["panel"] == "loss");
    const std::string d = diff["diff"].get<std::string>();
    CHECK(d.find("+id = \"hn\"") != std::string::npos);
    CHECK(d.find("+primitive = \"heatmap\"") != std::string::npos);
    const PanelData drawn = f.s->resolve("heat", "widget-1");
    REQUIRE_MESSAGE(drawn.error.ok(), drawn.error.message);
    REQUIRE(drawn.slots.size() == 1);
    CHECK(drawn.slots[0].tensor.shape() == std::vector<int64_t>{2, 3});
    CHECK(f.s->resolve("heat").error.code == "panel.not_found");
    CHECK(f.s->resolve("heat", "nope").error.code == "preview.not_found");
    const PreviewState state = f.s->preview("widget-1");
    REQUIRE(state.error.ok());
    CHECK(state.edits == 4);
    CHECK(state.panel == "loss");
    CHECK(state.spec.spec.find_panel("heat") != nullptr);

    // A dry run in the preview checks against the shadow and leaves it.
    const json dry = ok_in(*f.s, "widget-1", "panel.patch", {{"id", "heat"}, {"patch", {{"title", "Heat"}}}, {"dry_run", true}});
    CHECK(dry["applied"] == false);
    CHECK(ok(*f.s, "preview.diff", {{"id", "widget-1"}})["edits"] == 4);

    // Accept: one step, with the preview's id as its group.
    const json committed = ok(*f.s, "preview.commit", {{"id", "widget-1"}}, kUi);
    CHECK(committed["edits"] == 4);
    CHECK(committed["diff"] == diff["diff"]);
    CHECK(f.s->open_spec()->spec.find_panel("heat") != nullptr);
    const auto history = f.s->registry().history_state();
    REQUIRE(history.undo.size() == steps + 1);
    CHECK(history.undo[0].count == 4);
    CHECK(history.undo[0].group == "widget-1");
    CHECK(history.undo[0].origin == kAgent);
    CHECK(err(*f.s, "preview.diff", {{"id", "widget-1"}}) == "preview.not_found");

    // The journal holds the commit and then the four edits, under the agent
    // origin and the preview's group.
    const JournalRead read = read_journal(f.ws + "/journal/commands.jsonl");
    REQUIRE(read.events.size() == lines + 5);
    CHECK(read.events[lines].name == "preview.commit");
    CHECK(read.events[lines].origin == kUi);
    for (std::size_t i = lines + 1; i < lines + 5; ++i) {
        CHECK(read.events[i].origin == kAgent);
        CHECK(read.events[i].group == "widget-1");
        CHECK_FALSE(read.events[i].args.contains("dry_run"));
    }
    CHECK(read.events[lines + 4].name == "layout.place");
    const spec::Spec after = f.s->open_spec()->spec;
    const SpecReplay replay = replay_spec_journal(f.ws);
    REQUIRE_MESSAGE(replay.error.ok(), replay.error.message);
    CHECK(*replay.spec == after);

    // One undo reverts the whole stack, one redo puts it back.
    const json undone = ok(*f.s, "history.undo");
    CHECK(undone["steps"][0]["count"] == 4);
    CHECK(f.text() == before);
    ok(*f.s, "history.redo");
    CHECK(f.s->open_spec()->spec == after);
}

TEST_CASE("only the person commits a preview, and a discard leaves no trace") {
    Fixture f("preview-discard");
    const std::string before = f.text();
    ok(*f.s, "preview.begin", {{"id", "p1"}}, kAgent);
    f.stack_three("p1");
    const std::size_t lines = f.journal_lines();
    for (CommandOrigin o : {CommandOrigin::agent, CommandOrigin::mcp, CommandOrigin::api})
        CHECK(err(*f.s, "preview.commit", {{"id", "p1"}}, o) == "preview.not_allowed");
    // A refused commit is journaled as a failure, and changes nothing.
    CHECK(f.journal_lines() == lines + 3);
    CHECK(f.text() == before);
    CHECK(ok(*f.s, "preview.diff", {{"id", "p1"}})["edits"] == 3);

    const std::size_t steps = f.steps();
    const json dropped = ok(*f.s, "preview.discard", {{"id", "p1"}}, kAgent);
    CHECK(dropped == json{{"id", "p1"}, {"edits", 3}});
    CHECK(f.text() == before);
    CHECK(f.steps() == steps);
    CHECK(f.journal_lines() == lines + 3);
    CHECK(err(*f.s, "preview.discard", {{"id", "p1"}}) == "preview.not_found");
    CHECK(err_in(*f.s, "p1", "spec.get", json::object()) == "preview.not_found");

    // An empty preview commits nothing and files no step.
    ok(*f.s, "preview.begin", {{"id", "p2"}});
    CHECK(ok(*f.s, "preview.commit", {{"id", "p2"}}, CommandOrigin::builder)["edits"] == 0);
    CHECK(f.steps() == steps);
}

TEST_CASE("a stack survives a change of the open spec that it does not touch, and conflicts with one that it does") {
    Fixture f("preview-conflict");
    ok(*f.s, "preview.begin", {{"id", "p"}});
    ok_in(*f.s, "p", "panel.patch", {{"id", "loss"}, {"patch", {{"title", "Loss"}}}});
    // Something else changes the open spec meanwhile: the stack is applied
    // again on the new open spec, and the commit keeps both changes.
    ok(*f.s, "source.add", {{"id", "q"}, {"sql", "SELECT 1 AS one"}}, kUi);
    CHECK(ok(*f.s, "preview.diff", {{"id", "p"}})["diff"].get<std::string>().find("+title = \"Loss\"") !=
          std::string::npos);
    ok(*f.s, "preview.commit", {{"id", "p"}}, kUi);
    const spec::Spec now = f.s->open_spec()->spec;
    CHECK(now.find_source("q") != nullptr);
    CHECK(now.find_panel("loss")->title == std::optional<std::string>("Loss"));

    // A change that removes what the stack edits: every read of the preview
    // and its commit answer history.conflict, and nothing changes.
    ok(*f.s, "preview.begin", {{"id", "p2"}});
    ok_in(*f.s, "p2", "panel.patch", {{"id", "loss"}, {"patch", {{"title", "Again"}}}});
    ok(*f.s, "panel.remove", {{"id", "loss"}}, kUi);
    const std::string before = f.text();
    const std::size_t steps = f.steps();
    const std::size_t lines = f.journal_lines();
    CHECK(err(*f.s, "preview.diff", {{"id", "p2"}}) == "history.conflict");
    CHECK(err_in(*f.s, "p2", "spec.get", json::object()) == "history.conflict");
    CHECK(f.s->resolve("loss", "p2").error.code == "history.conflict");
    CHECK(err(*f.s, "preview.commit", {{"id", "p2"}}, kUi) == "history.conflict");
    CHECK(f.text() == before);
    CHECK(f.steps() == steps);
    CHECK(f.journal_lines() == lines + 1);  // the refused commit
    ok(*f.s, "preview.discard", {{"id", "p2"}});

    // A preview begun on a spec that is no longer open conflicts too.
    ok(*f.s, "preview.begin", {{"id", "p3"}});
    ok(*f.s, "spec.new", {{"id", "other"}});
    CHECK(err_in(*f.s, "p3", "source.add", {{"id", "z"}, {"sql", "SELECT 2 AS two"}}) == "history.conflict");
}

TEST_CASE("previews expire after their idle time and are bounded in number, edits and scope") {
    PreviewLimits limits;
    limits.max_previews = 2;
    limits.max_edits = 2;
    Fixture f("preview-bounds", limits);
    ok(*f.s, "preview.begin", {{"id", "a"}});
    const json made = ok(*f.s, "preview.begin");
    CHECK(made["id"] == "preview-1");
    CHECK(err(*f.s, "preview.begin", {{"id", "c"}}) == "preview.too_many");
    CHECK(err(*f.s, "preview.begin", {{"id", "a"}}) == "preview.exists");

    // Edits per preview; a merge key takes the place of its predecessor.
    ok_in(*f.s, "a", "panel.patch", {{"id", "loss"}, {"patch", {{"title", "One"}}}});
    CommandInvocation drag;
    drag.name = "panel.patch";
    drag.origin = CommandOrigin::builder;
    drag.preview = "a";
    drag.merge_key = "title";
    for (const char* t : {"T1", "T2", "T3"}) {
        drag.args = {{"id", "loss"}, {"patch", {{"title", t}}}};
        REQUIRE(f.s->invoke(drag).ok);
    }
    CHECK(ok(*f.s, "preview.diff", {{"id", "a"}})["edits"] == 2);
    CHECK(err_in(*f.s, "a", "panel.patch", {{"id", "loss"}, {"patch", {{"title", "Four"}}}}) ==
          "preview.too_many_edits");

    // Idle previews are dropped; a late call learns why.
    f.clock->now += std::chrono::minutes(20);
    ok(*f.s, "preview.diff", {{"id", "a"}});  // using it keeps it
    f.clock->now += std::chrono::minutes(20);
    CHECK(err(*f.s, "preview.diff", {{"id", "preview-1"}}) == "preview.expired");
    CHECK(ok(*f.s, "preview.diff", {{"id", "a"}})["edits"] == 2);
    f.clock->now += std::chrono::minutes(31);
    CHECK(err_in(*f.s, "a", "spec.get", json::object()) == "preview.expired");
    CHECK(f.s->preview("a").error.code == "preview.expired");
    CHECK(ok(*f.s, "preview.begin", {{"id", "a"}})["id"] == "a");

    // In a preview only the edits of the open spec run.
    const std::size_t lines = f.journal_lines();
    CHECK(err_in(*f.s, "a", "data.import", {{"paths", {"data/metrics.csv"}}}) == "preview.not_previewable");
    CHECK(err_in(*f.s, "a", "spec.new", {{"id", "x"}}, kUi) == "preview.not_previewable");
    CHECK(f.journal_lines() == lines);
}

TEST_CASE("a preview of one panel changes only that panel, its sources and what it adds") {
    Fixture f("preview-scope");
    ok(*f.s, "source.add", {{"id", "h"}, {"catalog", "tensor:heat_w"}});
    ok(*f.s, "panel.create", {{"id", "heat"}, {"primitive", "heatmap"}, {"bind", {{"values", "h"}}}});
    CHECK(err(*f.s, "preview.begin", {{"panel", "nope"}}) == "panel.not_found");
    CHECK(err(*f.s, "preview.begin", {{"panel", "loss"}, {"source", "m"}}) == "preview.bad_scope");
    ok(*f.s, "preview.begin", {{"id", "w"}, {"panel", "loss"}});
    ok_in(*f.s, "w", "panel.patch", {{"id", "loss"}, {"patch", {{"title", "Loss"}}}});
    ok_in(*f.s, "w", "source.patch", {{"id", "m"}, {"patch", {{"catalog", "table:metrics"}}}});
    ok_in(*f.s, "w", "panel.create", {{"id", "t"}, {"primitive", "table"}, {"bind", {{"rows", "m"}}}});
    ok_in(*f.s, "w", "layout.place", {{"panel", "t"}, {"target", "heat"}, {"where", "below"}});
    ok_in(*f.s, "w", "panel.remove", {{"id", "t"}});
    CHECK(err_in(*f.s, "w", "panel.patch", {{"id", "heat"}, {"patch", {{"title", "No"}}}}) == "preview.out_of_scope");
    CHECK(err_in(*f.s, "w", "source.remove", {{"id", "h"}}) == "preview.out_of_scope");
    CHECK(err_in(*f.s, "w", "layout.place", {{"panel", "heat"}, {"target", "loss"}, {"where", "below"}}) ==
          "preview.out_of_scope");
    CHECK(err_in(*f.s, "w", "spec.set_text", {{"text", f.text()}}) == "preview.out_of_scope");
}

#ifndef _WIN32
TEST_CASE("the stand-in's server sends a call's edits to the preview its X-Caliper-Preview header names") {
    const std::string ws = fresh_dir("preview-http") + "/workspace";
    auto s = open_session(ws);
    ok(*s, "spec.new", {{"id", "web"}});
    ok(*s, "preview.begin", {{"id", "widget-7"}});
    CommandServerOptions options;
    options.token = generate_session_token();
    options.origin = CommandOrigin::agent;  // the token is the agent's
    CommandServer server(s->queue(), options);
    REQUIRE(server.start().ok());
    std::atomic<bool> stop{false};
    std::thread drainer([&] {
        while (!stop.load()) {
            s->queue().drain();
            std::this_thread::sleep_for(std::chrono::milliseconds(2));
        }
    });
    const int port = server.port();
    auto post = [&](const std::string& name, const json& args, std::vector<std::string> extra) {
        extra.push_back("X-Caliper-Origin: agent");
        return http(port, "POST", "/v0/commands/" + name, options.token, args.dump(), extra).json_body();
    };
    const json added = post("source.add", {{"id", "one"}, {"sql", "SELECT 1 AS one"}}, {"X-Caliper-Preview: widget-7"});
    CHECK(added["ok"] == true);
    CHECK(added["result"]["applied"] == true);
    CHECK(s->open_spec()->spec.sources.empty());
    CHECK(post("spec.get", json::object(), {"X-Caliper-Preview: widget-7"})["result"]["json"]["source"].size() == 1);
    CHECK(post("preview.commit", {{"id", "widget-7"}}, {})["error"]["code"] == "preview.not_allowed");
    CHECK(post("spec.get", json::object(), {"X-Caliper-Preview: bad id"})["error"]["code"] ==
          "command.invalid_invocation");
    const std::string twice = "POST /v0/commands/spec.get HTTP/1.1\r\nHost: 127.0.0.1:" + std::to_string(port) +
                              "\r\nAuthorization: Bearer " + options.token +
                              "\r\nX-Caliper-Preview: a\r\nX-Caliper-Preview: b\r\nContent-Type: "
                              "application/json\r\nContent-Length: 2\r\n\r\n{}";
    CHECK(http_raw(port, twice).status == 400);
    ok(*s, "preview.commit", {{"id", "widget-7"}}, kUi);
    CHECK(s->open_spec()->spec.sources.size() == 1);
    s->queue().shutdown();
    server.stop();
    stop.store(true);
    drainer.join();
}
#endif
