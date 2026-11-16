// Tests of the review of spec edits in the command layer: for each origin
// through the session and its registry, an edit that changes the open spec is
// parked with its diff and the validator's warnings, then approved or denied,
// while a dry run, an edit of a preview, an edit that changes nothing and one
// that fails run at once; an approval after the open spec changed applies only
// the change that was shown; the queue and the stand-in's server hold a
// reviewed call until it is approved, denied or timed out; and the stand-in
// host reviews the origins its --review option names and prints each pending
// approval with its diff. The journal and the history are checked after each
// outcome.
#include <caliper_builder/acp/subprocess.h>
#include <caliper_builder/session/command_server.h>
#include <caliper_builder/session/session.h>

#include <caliper_builder/commands/journal.h>

#include <doctest/doctest.h>

#include <atomic>
#include <chrono>
#include <map>
#include <thread>

#include "helpers.h"

using namespace caliper_builder;
using namespace session_test;
using nlohmann::json;

namespace {

const CommandOrigin kUi = CommandOrigin::ui;
const CommandOrigin kAgent = CommandOrigin::agent;
const CommandOrigin kMcp = CommandOrigin::mcp;
const CommandOrigin kApi = CommandOrigin::api;

// A workspace with a metrics table and an open spec of one series panel.
struct Fixture {
    std::string ws;
    std::unique_ptr<Session> s;

    explicit Fixture(const std::string& name) {
        ws = fresh_dir(name) + "/workspace";
        write_text(ws + "/data/metrics.csv", "step,loss\n0,1.0\n1,0.5\n2,0.25\n");
        s = open_session(ws);
        ok(*s, "data.import", {{"paths", {"data/metrics.csv"}}});
        ok(*s, "spec.new", {{"id", "dash"}});
        ok(*s, "source.add", {{"id", "m"}, {"catalog", "table:metrics"}});
        ok(*s, "panel.create", {{"id", "loss"}, {"primitive", "series"}, {"bind", {{"x", "m.step"}, {"y", "m.loss"}}}});
    }

    JournalRead journal() const { return read_journal(ws + "/journal/commands.jsonl"); }
    std::size_t lines() const { return journal().events.size(); }
    std::size_t steps() const { return s->registry().history_state().undo.size(); }
    std::string text() const { return s->open_spec()->text; }
    std::optional<std::string> title() const { return s->open_spec()->spec.find_panel("loss")->title; }

    json title_patch(const std::string& title) const { return {{"id", "loss"}, {"patch", {{"title", title}}}}; }
};

#ifndef _WIN32
// Reads one line of a child's standard output.
std::string read_line(acp::Process& p, std::string& pending, int timeout_ms) {
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(timeout_ms);
    for (;;) {
        const auto eol = pending.find('\n');
        if (eol != std::string::npos) {
            std::string line = pending.substr(0, eol);
            pending.erase(0, eol + 1);
            return line;
        }
        const auto left = std::chrono::duration_cast<std::chrono::milliseconds>(deadline - std::chrono::steady_clock::now());
        if (left.count() <= 0) return {};
        char buf[4096];
        std::size_t got = 0;
        const acp::IoStatus st = p.read_stdout(buf, sizeof buf, &got, static_cast<int>(left.count()));
        if (st == acp::IoStatus::ok) pending.append(buf, got);
        else if (st != acp::IoStatus::timeout) return {};
    }
}
#endif

}  // namespace

TEST_CASE("an agent's spec edit waits for the person's review with its diff, and the person decides") {
    Fixture f("review-decide");
    CHECK(f.s->review().agent);
    CHECK(f.s->review().mcp);
    CHECK_FALSE(f.s->review().ui);
    CHECK_FALSE(f.s->review().api);
    // The person's own edits, the builder's and a program's run at once.
    ok(*f.s, "panel.patch", f.title_patch("By hand"), kUi);
    ok(*f.s, "panel.patch", f.title_patch("By builder"), CommandOrigin::builder);
    ok(*f.s, "panel.patch", f.title_patch("By program"), kApi);
    CHECK(f.title() == std::optional<std::string>("By program"));

    // An agent's edit is parked with its review; nothing changes yet.
    const std::string before = f.text();
    const std::size_t lines = f.lines();
    const std::size_t steps = f.steps();
    const CommandResult parked = invoke(*f.s, "panel.patch", f.title_patch("From the agent"), kAgent, "turn-1");
    REQUIRE(parked.error.code == "command.needs_approval");
    CHECK(parked.error.message.find("reviews") != std::string::npos);
    const auto pending = f.s->registry().pending_approvals();
    REQUIRE(pending.size() == 1);
    CHECK(pending[0].approval == parked.approval);
    CHECK(pending[0].invocation.group == "turn-1");
    const json review = pending[0].review;
    REQUIRE(review.is_object());
    const std::string diff = review["diff"].get<std::string>();
    CHECK(diff.rfind("--- a/dash.spec.toml\n+++ b/dash.spec.toml\n", 0) == 0);
    CHECK(diff.find("-title = \"By program\"") != std::string::npos);
    CHECK(diff.find("+title = \"From the agent\"") != std::string::npos);
    CHECK(review["warnings"].is_array());
    CHECK(pending_approval_to_json(pending[0])["review"] == review);
    CHECK(f.text() == before);
    CHECK(f.lines() == lines);
    CHECK(f.steps() == steps);

    // Approved: it runs, is journaled under the agent and becomes a step.
    const CommandResult approved = f.s->registry().approve(parked.approval);
    REQUIRE(approved.ok);
    CHECK(approved.result["diff"] == diff);
    CHECK(f.title() == std::optional<std::string>("From the agent"));
    JournalRead read = f.journal();
    REQUIRE(read.events.size() == lines + 1);
    CHECK(read.events.back().origin == kAgent);
    CHECK(read.events.back().ok);
    CHECK(read.events.back().group == "turn-1");
    CHECK(f.steps() == steps + 1);

    // Denied: nothing changes, and the denial is journaled.
    const CommandResult from_mcp = invoke(*f.s, "panel.patch", f.title_patch("From outside"), kMcp);
    REQUIRE(from_mcp.error.code == "command.needs_approval");
    CHECK(f.s->registry().deny(from_mcp.approval).error.code == "command.denied");
    CHECK(f.title() == std::optional<std::string>("From the agent"));
    read = f.journal();
    REQUIRE(read.events.size() == lines + 2);
    CHECK_FALSE(read.events.back().ok);
    CHECK(read.events.back().error == "command.denied");
    CHECK(read.events.back().origin == kMcp);
    CHECK(f.steps() == steps + 1);
    CHECK(f.s->registry().pending_approvals().empty());
}

TEST_CASE("a dry run, an edit of a preview, an edit that changes nothing and one that fails need no review") {
    Fixture f("review-exempt");
    const std::size_t lines = f.lines();
    const json dry = ok(*f.s, "panel.patch", {{"id", "loss"}, {"patch", {{"title", "Dry"}}}, {"dry_run", true}}, kAgent);
    CHECK(dry["applied"] == false);
    CHECK(dry["diff"].get<std::string>().find("+title = \"Dry\"") != std::string::npos);
    // An edit that leaves the open spec as it is: an empty diff, run at once.
    ok(*f.s, "panel.patch", {{"id", "loss"}, {"patch", {{"title", nullptr}}}}, kMcp);
    CHECK(err(*f.s, "panel.patch", {{"id", "nope"}, {"patch", {{"title", "X"}}}}, kAgent) == "panel.not_found");
    CHECK(err(*f.s, "source.add", {{"id", "bad"}, {"from", "missing"}}, kAgent) == "spec.unknown_source");
    CommandInvocation in_preview;
    ok(*f.s, "preview.begin", {{"id", "p"}});
    in_preview.name = "panel.patch";
    in_preview.args = f.title_patch("Preview");
    in_preview.origin = kAgent;
    in_preview.preview = "p";
    CHECK(f.s->invoke(in_preview).ok);
    CHECK(f.s->registry().pending_approvals().empty());
    // The person commits the preview: the person's act, so no review.
    ok(*f.s, "preview.commit", {{"id", "p"}}, kUi);
    CHECK(f.title() == std::optional<std::string>("Preview"));
    // The two failures were journaled as failures; the reads and the dry run were not.
    const JournalRead read = f.journal();
    std::size_t failures = 0;
    for (std::size_t i = lines; i < read.events.size(); ++i) failures += read.events[i].ok ? 0 : 1;
    CHECK(failures == 2);
}

TEST_CASE("review is set per origin and review.status tells each caller") {
    Fixture f("review-settings");
    for (CommandOrigin o : {CommandOrigin::ui, CommandOrigin::builder, kAgent, kMcp, kApi}) {
        const json status = ok(*f.s, "review.status", json::object(), o);
        CHECK(status["origin"] == command_origin_name(o));
        CHECK(status["spec_edits"] == (o == kAgent || o == kMcp));
        CHECK(status["destructive"] == (o == kAgent || o == kMcp || o == kApi));
    }
    f.s->set_review(kAgent, false);
    f.s->set_review(kApi, true);
    CHECK(ok(*f.s, "review.status", json::object(), kAgent)["spec_edits"] == false);
    ok(*f.s, "panel.patch", f.title_patch("Unreviewed"), kAgent);
    CHECK(invoke(*f.s, "panel.patch", f.title_patch("Reviewed"), kApi).error.code == "command.needs_approval");
    CHECK(f.s->review().api);
    CHECK_FALSE(f.s->review().agent);

    // A session opened with other settings starts with them.
    SessionOptions options;
    options.review = ReviewSettings{false, false, false, false, false};
    auto opened = Session::open(fresh_dir("review-settings-off") + "/workspace", options);
    REQUIRE(opened.error.ok());
    ok(*opened.session, "spec.new", {{"id", "x"}}, kMcp);
    ok(*opened.session, "source.add", {{"id", "one"}, {"sql", "SELECT 1 AS one"}}, kMcp);
}

TEST_CASE("an approval after the open spec changed applies only the change that was shown") {
    Fixture f("review-changed");
    // A change elsewhere: the shown change still applies, and both stay.
    const CommandResult a = invoke(*f.s, "panel.patch", f.title_patch("Shown"), kAgent);
    REQUIRE(a.error.code == "command.needs_approval");
    ok(*f.s, "source.add", {{"id", "q"}, {"sql", "SELECT 1 AS one"}}, kUi);
    REQUIRE(f.s->registry().approve(a.approval).ok);
    CHECK(f.title() == std::optional<std::string>("Shown"));
    CHECK(f.s->open_spec()->spec.find_source("q") != nullptr);

    // The same value changed meanwhile: the change would differ from the one
    // shown, so it is refused, journaled as a failure, and files no step.
    const CommandResult b = invoke(*f.s, "panel.patch", f.title_patch("Agent's"), kAgent);
    REQUIRE(b.error.code == "command.needs_approval");
    ok(*f.s, "panel.patch", f.title_patch("Person's"), kUi);
    const std::size_t lines = f.lines();
    const std::size_t steps = f.steps();
    CHECK(f.s->registry().approve(b.approval).error.code == "history.conflict");
    CHECK(f.title() == std::optional<std::string>("Person's"));
    CHECK(f.lines() == lines + 1);
    CHECK(f.journal().events.back().error == "history.conflict");
    CHECK(f.steps() == steps);

    // What the edit changes is gone: refused too.
    const CommandResult c = invoke(*f.s, "panel.patch", f.title_patch("Gone"), kAgent);
    REQUIRE(c.error.code == "command.needs_approval");
    ok(*f.s, "panel.remove", {{"id", "loss"}}, kUi);
    CHECK(f.s->registry().approve(c.approval).error.code == "history.conflict");

    // Another spec opened meanwhile: refused.
    const CommandResult d = invoke(*f.s, "source.add", {{"id", "z"}, {"sql", "SELECT 2 AS two"}}, kMcp);
    REQUIRE(d.error.code == "command.needs_approval");
    ok(*f.s, "spec.new", {{"id", "other"}}, kUi);
    CHECK(f.s->registry().approve(d.approval).error.code == "history.conflict");
    CHECK(f.s->open_spec()->spec.sources.empty());
}

#ifndef _WIN32
TEST_CASE("the stand-in's server holds a reviewed call until it is approved, denied or timed out") {
    const std::string ws = fresh_dir("review-http") + "/workspace";
    auto s = open_session(ws);
    ok(*s, "spec.new", {{"id", "web"}});
    // One session behind a server for each origin a token can have; the
    // decisions are made when the call is parked, and only the "late" server
    // has a short wait, for the call that nobody decides.
    std::map<std::string, std::unique_ptr<CommandServer>> servers;
    std::map<std::string, std::string> tokens;
    auto serve = [&](const std::string& key, CommandOrigin origin, std::chrono::milliseconds wait) {
        CommandServerOptions options;
        options.token = generate_session_token();
        options.origin = origin;
        options.approval_wait = wait;
        tokens[key] = options.token;
        servers[key] = std::make_unique<CommandServer>(s->queue(), options);
        REQUIRE(servers[key]->start().ok());
    };
    serve("agent", CommandOrigin::agent, std::chrono::milliseconds(120000));
    serve("mcp", CommandOrigin::mcp, std::chrono::milliseconds(120000));
    serve("late", CommandOrigin::agent, std::chrono::milliseconds(1500));
    serve("api", CommandOrigin::api, std::chrono::milliseconds(120000));
    std::atomic<bool> stop{false};
    std::thread drainer([&] {
        while (!stop.load()) {
            s->queue().drain();
            std::this_thread::sleep_for(std::chrono::milliseconds(2));
        }
    });
    auto post = [&](const std::string& key, const std::string& name, const json& args) {
        return http(servers[key]->port(), "POST", "/v0/commands/" + name, tokens[key], args.dump(),
                    {"X-Caliper-Group: turn-9"})
            .json_body();
    };
    // Waits for the call to be parked and decides it as the app's UI does.
    auto decide_when_pending = [&](bool allow) {
        for (int i = 0; i < 4000; ++i) {
            const auto pending = s->registry().pending_approvals();
            if (!pending.empty()) {
                CHECK(pending[0].review.is_object());
                if (allow) s->queue().approve(pending[0].approval);
                else s->queue().deny(pending[0].approval);
                return;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(5));
        }
        FAIL("no call was parked");
    };
    const std::size_t lines = read_journal(ws + "/journal/commands.jsonl").events.size();

    json reply;
    std::thread approver([&] { decide_when_pending(true); });
    reply = post("agent", "source.add", {{"id", "one"}, {"sql", "SELECT 1 AS one"}});
    approver.join();
    CHECK(reply["ok"] == true);
    CHECK(reply["result"]["applied"] == true);
    CHECK(s->open_spec()->spec.sources.size() == 1);

    std::thread denier([&] { decide_when_pending(false); });
    reply = post("mcp", "source.add", {{"id", "two"}, {"sql", "SELECT 2 AS two"}});
    denier.join();
    CHECK(reply["error"]["code"] == "command.denied");

    // Nobody decides: withdrawn after the wait, and journaled like a denial.
    reply = post("late", "source.add", {{"id", "three"}, {"sql", "SELECT 3 AS three"}});
    CHECK(reply["error"]["code"] == "command.approval_timeout");
    CHECK(s->registry().pending_approvals().empty());
    CHECK(s->open_spec()->spec.sources.size() == 1);

    // A program gets the approval id at once; a dry run is answered at once.
    reply = post("api", "source.add", {{"id", "four"}, {"sql", "SELECT 4 AS four"}});
    CHECK(reply["ok"] == true);  // review is off for api by default
    s->set_review(CommandOrigin::api, true);
    reply = post("api", "source.add", {{"id", "five"}, {"sql", "SELECT 5 AS five"}});
    CHECK(reply["error"]["code"] == "command.needs_approval");
    CHECK(reply["error"]["approval"].is_string());
    reply = post("agent", "source.add", {{"id", "six"}, {"sql", "SELECT 6 AS six"}, {"dry_run", true}});
    CHECK(reply["result"]["applied"] == false);

    const JournalRead read = read_journal(ws + "/journal/commands.jsonl");
    REQUIRE(read.events.size() == lines + 4);  // approved, denied, timed out, the api's add
    CHECK(read.events[lines].ok);
    CHECK(read.events[lines].origin == CommandOrigin::agent);
    CHECK(read.events[lines + 1].error == "command.denied");
    CHECK(read.events[lines + 1].origin == CommandOrigin::mcp);
    CHECK(read.events[lines + 2].error == "command.approval_timeout");
    CHECK(read.events[lines + 2].group == "turn-9");
    CHECK(read.events[lines + 3].origin == CommandOrigin::api);
    s->queue().shutdown();
    for (auto& server : servers) server.second->stop();
    stop.store(true);
    drainer.join();
}

TEST_CASE("the stand-in host reviews the origins --review names and prints each approval with its diff") {
    const std::string root = fresh_dir("review-host");
    const std::string ws = root + "/workspace";
    acp::SpawnOptions so;
    so.argv = {CALIPER_BUILDER_HOST_EXE, "--workspace", ws, "--api-dir", root + "/api", "--approve", "ask", "--review",
               "mcp"};
    Error e;
    std::unique_ptr<acp::Process> host = acp::Process::spawn(so, &e);
    REQUIRE_MESSAGE(host, e.message);
    const std::string api_file = root + "/api/api.json";
    for (int i = 0; i < 400 && !fs::exists(api_file); ++i) std::this_thread::sleep_for(std::chrono::milliseconds(25));
    REQUIRE(fs::exists(api_file));
    const json api = json::parse(read_text(api_file));
    const int port = api["port"].get<int>();
    const std::string token = api["token"].get<std::string>();
    auto post = [&](const std::string& name, const json& args, const std::string& origin) {
        return http(port, "POST", "/v0/commands/" + name, token, args.dump(), {"X-Caliper-Origin: " + origin}, 30000)
            .json_body();
    };
    std::string pending;
    auto call_and_decide = [&](const std::string& name, const json& args, const std::string& decision) {
        json reply;
        std::thread caller([&] { reply = post(name, args, "mcp"); });
        const json line = json::parse(read_line(*host, pending, 20000), nullptr, false);
        REQUIRE(line.is_object());
        CHECK(line["name"] == name);
        CHECK(line["origin"] == "mcp");
        CHECK_FALSE(line.contains("decided"));
        REQUIRE(line["review"].is_object());
        CHECK(line["review"]["diff"].get<std::string>().find("+id = \"" + args["id"].get<std::string>() + "\"") !=
              std::string::npos);
        const std::string answer = decision + " " + line["approval"].get<std::string>() + "\n";
        CHECK(host->write_stdin_all(answer.data(), answer.size(), 5000) == acp::IoStatus::ok);
        caller.join();
        return reply;
    };
    CHECK(post("spec.new", {{"id", "hosted"}}, "mcp")["ok"] == true);  // replaces the document; no diff to review
    CHECK(call_and_decide("source.add", {{"id", "one"}, {"sql", "SELECT 1 AS one"}}, "allow")["ok"] == true);
    CHECK(call_and_decide("source.add", {{"id", "two"}, {"sql", "SELECT 2 AS two"}}, "deny")["error"]["code"] ==
          "command.denied");
    CHECK(post("review.status", json::object(), "mcp")["result"]["spec_edits"] == true);
    // The token is for mcp: a request cannot become agent or api.
    CHECK(http(port, "POST", "/v0/commands/spec.get", token, "{}", {"X-Caliper-Origin: agent"}).status == 400);
    CHECK(http(port, "POST", "/v0/commands/spec.get", token, "{}", {"X-Caliper-Origin: api"}).status == 400);
    const acp::ExitStatus status = host->terminate(10000);
    CHECK(status.exited);
    CHECK(status.code == 0);

    // A host whose token is for the agent origin, which --review does not
    // name: the agent's edit runs at once, and its journal line says agent.
    {
        const std::string root2 = fresh_dir("review-host-agent");
        acp::SpawnOptions as_agent;
        as_agent.argv = {CALIPER_BUILDER_HOST_EXE, "--workspace", root2 + "/workspace", "--api-dir", root2 + "/api",
                         "--origin", "agent", "--approve", "ask", "--review", "mcp"};
        std::unique_ptr<acp::Process> agent_host = acp::Process::spawn(as_agent, &e);
        REQUIRE_MESSAGE(agent_host, e.message);
        const std::string api2 = root2 + "/api/api.json";
        for (int i = 0; i < 400 && !fs::exists(api2); ++i) std::this_thread::sleep_for(std::chrono::milliseconds(25));
        REQUIRE(fs::exists(api2));
        const json a2 = json::parse(read_text(api2));
        auto post2 = [&](const std::string& name, const json& args) {
            return http(a2["port"].get<int>(), "POST", "/v0/commands/" + name, a2["token"].get<std::string>(),
                        args.dump(), {}, 30000)
                .json_body();
        };
        CHECK(post2("spec.new", {{"id", "hosted"}})["ok"] == true);
        CHECK(post2("source.add", {{"id", "three"}, {"sql", "SELECT 3 AS three"}})["ok"] == true);
        CHECK(post2("review.status", json::object())["result"]["origin"] == "agent");
        CHECK(post2("review.status", json::object())["result"]["spec_edits"] == false);
        CHECK(agent_host->terminate(10000).code == 0);
        CHECK(read_journal(root2 + "/workspace/journal/commands.jsonl").events.back().origin == CommandOrigin::agent);
    }

    // Unknown origins are refused at start, for --review and for --origin.
    for (const std::vector<std::string>& extra :
         {std::vector<std::string>{"--review", "mcp,robot"}, std::vector<std::string>{"--origin", "ui"}}) {
        acp::SpawnOptions bad;
        bad.argv = {CALIPER_BUILDER_HOST_EXE, "--workspace", ws, "--api-dir", root + "/api"};
        bad.argv.insert(bad.argv.end(), extra.begin(), extra.end());
        std::unique_ptr<acp::Process> refused = acp::Process::spawn(bad, &e);
        REQUIRE(refused);
        acp::ExitStatus refused_status;
        CHECK(refused->wait_for(10000, &refused_status));
        CHECK(refused_status.code == 2);
    }
}
#endif
