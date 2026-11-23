// Tests of agent permissions in the session (permissions.h, design/
// command-layer.md, Agent permissions): each mode of the side chat's agent,
// the widget agent left as it was, rules for this session and saved ones that
// survive a restart and are revoked, the lines no mode or rule crosses, a mode
// changed while an approval waits, and that no command sets a mode or a rule.
#include <caliper_builder/session/session.h>

#include <caliper_builder/commands/journal.h>

#include <doctest/doctest.h>

#include "helpers.h"

using namespace caliper_builder;
using namespace session_test;
using nlohmann::json;

namespace {

CommandResult send(Session& s, const std::string& name, json args, CommandOrigin origin, const std::string& client) {
    CommandInvocation inv;
    inv.name = name;
    inv.args = std::move(args);
    inv.origin = origin;
    inv.client = client;
    return s.invoke(std::move(inv));
}

CommandResult chat(Session& s, const std::string& name, json args = json::object()) {
    return send(s, name, std::move(args), CommandOrigin::agent, "chat");
}

// A workspace with a loader and an open spec of one panel, saved as dash.
struct Fixture {
    std::string ws;
    std::string perms;
    std::unique_ptr<Session> s;

    explicit Fixture(const std::string& name) {
        const std::string root = fresh_dir(name);
        ws = root + "/workspace";
        perms = root + "/app-data/agent-permissions.json";
        write_text(ws + "/data/metrics.csv", "step,loss\n0,1.0\n1,0.5\n");
        reopen();
        ok(*s, "data.import", {{"paths", {"data/metrics.csv"}}});
        ok(*s, "spec.new", {{"id", "dash"}});
        ok(*s, "source.add", {{"id", "m"}, {"catalog", "table:metrics"}});
        ok(*s, "panel.create", {{"id", "loss"}, {"primitive", "series"}, {"bind", {{"x", "m.step"}, {"y", "m.loss"}}}});
        ok(*s, "applet.save");
    }
    void reopen() {
        s.reset();
        SessionOptions o;
        o.registry.check_results = true;
        o.permissions_file = perms;
        auto opened = Session::open(ws, o);
        REQUIRE(opened.error.ok());
        s = std::move(opened.session);
        s->set_chat_agent("claude");
    }
    std::string title() const { return s->open_spec()->spec.panels.at(0).title.value_or(""); }
    std::string loader() { return ok(*s, "loader.list")["loaders"][0]["id"].get<std::string>(); }
    JournalEvent last_event() const {
        const auto read = read_journal(ws + "/journal/commands.jsonl");
        REQUIRE_FALSE(read.events.empty());
        return read.events.back();
    }
};

json retitle(const std::string& title) { return {{"id", "loss"}, {"patch", {{"title", title}}}}; }

}  // namespace

TEST_CASE("Ask: the chat's agent's edits wait on a card, as before") {
    Fixture f("modes-ask");
    CHECK(f.s->agent_mode(AgentPrincipal::chat) == AgentMode::ask);
    const CommandResult r = chat(*f.s, "panel.patch", retitle("Asked"));
    CHECK(r.error.code == "command.needs_approval");
    CHECK(f.title() != "Asked");
}

TEST_CASE("Accept edits: an edit applies at once, journaled with its mode, and the person can undo it") {
    Fixture f("modes-accept");
    REQUIRE(f.s->set_agent_mode(AgentPrincipal::chat, AgentMode::accept_edits).ok());
    const CommandResult r = chat(*f.s, "panel.patch", retitle("Applied"));
    REQUIRE(r.ok);
    CHECK(f.title() == "Applied");
    const JournalEvent e = f.last_event();
    CHECK(e.origin == CommandOrigin::agent);
    CHECK(e.allowed_by == "accept_edits");
    // It is one undo step, the newest, which the person takes back.
    const auto state = f.s->registry().history_state();
    REQUIRE_FALSE(state.undo.empty());
    CHECK(state.undo.front().seqs == std::vector<std::uint64_t>{e.seq});
    ok(*f.s, "history.undo");
    CHECK(f.title() != "Applied");
    // What is not an edit of the open spec still asks.
    CHECK(chat(*f.s, "loader.delete", {{"id", f.loader()}}).error.code == "command.needs_approval");
}

TEST_CASE("Read only: reads and proposals run, every change is refused with a plain message") {
    Fixture f("modes-read-only");
    REQUIRE(f.s->set_agent_mode(AgentPrincipal::chat, AgentMode::read_only).ok());
    const CommandResult r = chat(*f.s, "panel.patch", retitle("Refused"));
    CHECK(r.error.code == "command.denied");
    CHECK(r.error.message.find("Read only") != std::string::npos);
    CHECK(chat(*f.s, "panel.patch", {{"id", "loss"}, {"patch", {{"title", "Dry"}}}, {"dry_run", true}}).ok);
    CHECK(chat(*f.s, "spec.get").ok);
    CHECK(chat(*f.s, "preview.begin", {{"id", "p1"}}).ok);
    CHECK(chat(*f.s, "data.import", {{"paths", {"data/metrics.csv"}}}).error.code == "command.denied");
    CHECK(f.title() != "Refused");
    CHECK(f.s->registry().pending_approvals().empty());
}

TEST_CASE("Allow all: saved for each principal, loaded at launch, no card but a data-loss card") {
    Fixture f("modes-allow-all");
    REQUIRE(f.s->set_agent_mode(AgentPrincipal::chat, AgentMode::allow_all).ok());
    CHECK(f.s->agent_mode(AgentPrincipal::chat) == AgentMode::allow_all);
    CHECK(read_text(f.perms).find("\"chat\": \"allow_all\"") != std::string::npos);
    // An edit that review would show runs at once, journaled as Allow all's.
    REQUIRE(chat(*f.s, "panel.patch", retitle("Agent")).ok);
    CHECK(f.last_event().allowed_by == "allow_all");
    // The replacement of an open spec with unsaved changes still asks.
    ok(*f.s, "panel.patch", retitle("Unsaved"));
    CHECK(chat(*f.s, "spec.new", {{"id", "other"}}).error.code == "command.needs_approval");
    CHECK(chat(*f.s, "applet.open", {{"id", "dash"}}).error.code == "command.needs_approval");
    CHECK(f.title() == "Unsaved");
    // So does an overwrite of a saved applet, and undo stays the person's.
    CHECK(chat(*f.s, "applet.save", {{"overwrite", true}}).error.code == "command.needs_approval");
    CHECK(chat(*f.s, "history.undo").error.code == "command.not_allowed");
    f.s->registry().cancel_pending();
    // A deletion runs without a card.
    REQUIRE(chat(*f.s, "loader.delete", {{"id", f.loader()}}).ok);
    CHECK(f.last_event().allowed_by == "allow_all");
    // For mcp clients too.
    REQUIRE(f.s->set_agent_mode(AgentPrincipal::mcp, AgentMode::allow_all).ok());
    CHECK(read_text(f.perms).find("\"mcp\": \"allow_all\"") != std::string::npos);
    write_text(f.ws + "/data/other.csv", "step,acc\n0,0.5\n");
    ok(*f.s, "data.import", {{"paths", {"data/other.csv"}}});
    REQUIRE(send(*f.s, "loader.delete", {{"id", f.loader()}}, CommandOrigin::mcp, "").ok);
    CHECK(f.last_event().allowed_by == "allow_all");
    CHECK(send(*f.s, "applet.save", {{"overwrite", true}}, CommandOrigin::mcp, "").error.code == "command.needs_approval");
    f.s->registry().cancel_pending();
    // The end of the chat's agent leaves it on, and a new launch loads it.
    f.s->end_chat_session();
    CHECK(f.s->agent_mode(AgentPrincipal::chat) == AgentMode::allow_all);
    f.reopen();
    CHECK(f.s->permission_problems().empty());
    CHECK(f.s->agent_mode(AgentPrincipal::chat) == AgentMode::allow_all);
    CHECK(f.s->agent_mode(AgentPrincipal::mcp) == AgentMode::allow_all);
    // It stays on until the person sets another mode.
    REQUIRE(f.s->set_agent_mode(AgentPrincipal::chat, AgentMode::ask).ok());
    f.reopen();
    CHECK(f.s->agent_mode(AgentPrincipal::chat) == AgentMode::ask);
    CHECK(f.s->agent_mode(AgentPrincipal::mcp) == AgentMode::allow_all);
}

TEST_CASE("Allow all runs a Python loader without a card, and no rule does") {
    Fixture f("modes-allow-all-python");
    const std::string python = CALIPER_BUILDER_TEST_PYTHON;
    write_text(f.ws + "/scripts/s.py", "def load(path, out):\n    out.tensor('w', [1.0, 2.0, 3.0])\n");
    write_text(f.ws + "/data/in.bin", "anything");
    auto recipe = [&](const std::string& id) {
        return "caliper_loader = 0\n[loader]\nid = \"" + id +
               "\"\n[input]\npath = \"data/in.bin\"\nreader = \"python\"\n[reader]\nscript = \"scripts/s.py\"\n" +
               (python.empty() ? std::string() : "python = \"" + python + "\"\n") + "[[output]]\nname = \"" + id +
               "_w\"\ntensor = \"w\"\n";
    };
    // Ask: the run waits on a card, which offers no rule.
    const CommandResult asked = chat(*f.s, "loader.create", {{"text", recipe("py")}, {"run", true}});
    REQUIRE(asked.error.code == "command.needs_approval");
    const PendingApproval p = f.s->registry().pending_approvals().at(0);
    CHECK_FALSE(f.s->approval_offers_rules(p.invocation));
    f.s->registry().cancel_pending();
    // Allow all: no card, and the handler holds the permission to run it.
    REQUIRE(f.s->set_agent_mode(AgentPrincipal::chat, AgentMode::allow_all).ok());
    const CommandResult run = chat(*f.s, "loader.create", {{"text", recipe("py")}, {"run", true}});
    CHECK(run.error.code != "command.needs_approval");
    CHECK(run.error.code != "loader.python_needs_approval");
    CHECK(f.s->registry().pending_approvals().empty());
    CHECK(f.last_event().allowed_by == "allow_all");
    if (python.empty()) {
        MESSAGE("Skipped the run's result: CALIPER_BUILDER_TEST_PYTHON is not set.");
        return;
    }
    INFO(run.error.code << ": " << run.error.message);
    CHECK(run.ok);
    CHECK(ok(*f.s, "loader.describe", {{"id", "py"}})["status"] == "ok");
}

TEST_CASE("the widget agent keeps its preview with Accept, whatever the chat's mode") {
    Fixture f("modes-widget");
    REQUIRE(f.s->set_agent_mode(AgentPrincipal::chat, AgentMode::allow_all).ok());
    CHECK(send(*f.s, "loader.delete", {{"id", f.loader()}}, CommandOrigin::agent, "widget").error.code ==
          "command.needs_approval");
    REQUIRE(f.s->set_agent_mode(AgentPrincipal::chat, AgentMode::read_only).ok());
    CHECK(send(*f.s, "panel.patch", retitle("W"), CommandOrigin::agent, "widget").error.code == "command.needs_approval");
}

TEST_CASE("rules: for this session, and saved ones that survive a restart and are revoked") {
    Fixture f("modes-rules");
    const std::string loader = f.loader();
    CommandResult r = chat(*f.s, "panel.patch", retitle("One"));
    REQUIRE(r.error.code == "command.needs_approval");
    PendingApproval p = f.s->registry().pending_approvals().at(0);
    REQUIRE(f.s->approval_offers_rules(p.invocation));
    REQUIRE(f.s->allow_always(p.invocation, RuleScope::session).ok());
    REQUIRE(f.s->registry().approve(p.approval).ok);
    CHECK(chat(*f.s, "panel.patch", retitle("Two")).ok);
    CHECK(f.last_event().allowed_by == "session_rule");
    // The rule is the chat's: an mcp client still asks; the session's end drops it.
    CHECK(send(*f.s, "panel.patch", retitle("M"), CommandOrigin::mcp, "").error.code == "command.needs_approval");
    f.s->registry().cancel_pending();
    f.s->end_chat_session();
    CHECK(chat(*f.s, "panel.patch", retitle("Three")).error.code == "command.needs_approval");
    f.s->registry().cancel_pending();

    // Saved: written owner-only, kept across a restart, keyed by the chat's agent.
    r = chat(*f.s, "loader.delete", {{"id", loader}});
    REQUIRE(r.error.code == "command.needs_approval");
    p = f.s->registry().pending_approvals().at(0);
    REQUIRE(f.s->allow_always(p.invocation, RuleScope::saved).ok());
    f.s->registry().cancel_pending();
    CHECK(f.s->saved_agent_rules() == std::vector<AgentRule>{{"claude", "loader.delete"}});
    f.reopen();
    CHECK(f.s->saved_agent_rules() == std::vector<AgentRule>{{"claude", "loader.delete"}});
    CHECK(chat(*f.s, "loader.delete", {{"id", loader}}).ok);
    CHECK(f.last_event().allowed_by == "saved_rule");
    f.s->set_chat_agent("codex");  // another agent: not its rule
    ok(*f.s, "data.import", {{"paths", {"data/metrics.csv"}}});
    CHECK(chat(*f.s, "loader.delete", {{"id", f.loader()}}).error.code == "command.needs_approval");
    f.s->registry().cancel_pending();
    f.s->set_chat_agent("claude");
    // Read only beats the saved rule.
    REQUIRE(f.s->set_agent_mode(AgentPrincipal::chat, AgentMode::read_only).ok());
    CHECK(chat(*f.s, "loader.delete", {{"id", f.loader()}}).error.code == "command.denied");
    REQUIRE(f.s->set_agent_mode(AgentPrincipal::chat, AgentMode::ask).ok());
    CHECK(f.s->revoke_agent_rule({"claude", "loader.delete"}));
    f.reopen();
    CHECK(f.s->saved_agent_rules().empty());
    CHECK(chat(*f.s, "loader.delete", {{"id", f.loader()}}).error.code == "command.needs_approval");
}

TEST_CASE("rules: a run command's rule allows its one parameter or action, for this session and saved") {
    Fixture f("modes-run-rules");
    const json lr = {{"name", "lr"}, {"value", 0.01}};
    CommandResult r = send(*f.s, "run.set_param", lr, CommandOrigin::mcp, "");
    REQUIRE(r.error.code == "command.needs_approval");
    PendingApproval p = f.s->registry().pending_approvals().at(0);
    REQUIRE(f.s->approval_offers_rules(p.invocation));
    REQUIRE(f.s->allow_always(p.invocation, RuleScope::session).ok());
    f.s->registry().cancel_pending();
    // lr runs without a card now; this session has no run control, so the
    // command answers command.not_available. Another parameter still asks.
    CHECK(send(*f.s, "run.set_param", lr, CommandOrigin::mcp, "").error.code == "command.not_available");
    CHECK(send(*f.s, "run.set_param", {{"name", "momentum"}, {"value", 0.9}}, CommandOrigin::mcp, "").error.code ==
          "command.needs_approval");
    f.s->registry().cancel_pending();

    r = send(*f.s, "run.action", {{"name", "save"}}, CommandOrigin::mcp, "");
    REQUIRE(r.error.code == "command.needs_approval");
    p = f.s->registry().pending_approvals().at(0);
    REQUIRE(f.s->allow_always(p.invocation, RuleScope::saved).ok());
    f.s->registry().cancel_pending();
    CHECK(f.s->saved_agent_rules() == std::vector<AgentRule>{{"mcp", "run.action:save"}});
    f.reopen();
    CHECK(send(*f.s, "run.action", {{"name", "save"}}, CommandOrigin::mcp, "").error.code == "command.not_available");
    CHECK(send(*f.s, "run.action", {{"name", "other"}}, CommandOrigin::mcp, "").error.code ==
          "command.needs_approval");
    f.s->registry().cancel_pending();
}

TEST_CASE("no rule or mode makes a viewer write or covers what binds to content or runs Python") {
    Fixture f("modes-ceilings");
    REQUIRE(f.s->set_agent_mode(AgentPrincipal::chat, AgentMode::allow_all).ok());
    // A viewer only reads, whatever client it names.
    CHECK(send(*f.s, "panel.patch", retitle("V"), CommandOrigin::viewer, "chat").error.code == "command.not_allowed");
    // An overwrite's card offers no rule.
    const CommandResult r = chat(*f.s, "applet.save", {{"overwrite", true}});
    REQUIRE(r.error.code == "command.needs_approval");
    const PendingApproval p = f.s->registry().pending_approvals().at(0);
    CHECK_FALSE(f.s->approval_offers_rules(p.invocation));
    CHECK(f.s->allow_always(p.invocation, RuleScope::saved).code == "permissions.no_rule");
    CHECK(f.s->saved_agent_rules().empty());
}

TEST_CASE("a mode changed while an approval waits leaves it waiting for the person") {
    Fixture f("modes-pending");
    const CommandResult r = chat(*f.s, "panel.patch", retitle("Waiting"));
    REQUIRE(r.error.code == "command.needs_approval");
    REQUIRE(f.s->set_agent_mode(AgentPrincipal::chat, AgentMode::allow_all).ok());
    REQUIRE(f.s->set_agent_mode(AgentPrincipal::chat, AgentMode::accept_edits).ok());
    CHECK(f.s->registry().pending_approvals().size() == 1);
    CHECK(f.title() != "Waiting");
    f.s->end_chat_session();
    CHECK(f.s->registry().pending_approvals().size() == 1);
    REQUIRE(f.s->registry().approve(r.approval).ok);
    CHECK(f.title() == "Waiting");
}

TEST_CASE("no command reads or sets a mode or a rule, from any origin") {
    Fixture f("modes-no-command");
    for (const auto& d : command_table()) {
        INFO(d.name);
        for (const char* word : {"mode", "rule", "permission", "allow"}) CHECK(d.name.find(word) == std::string::npos);
    }
    for (CommandOrigin o : {CommandOrigin::agent, CommandOrigin::mcp, CommandOrigin::api, CommandOrigin::viewer})
        for (const char* name : {"permissions.set", "agent.set_mode", "rules.add", "settings.set", "app.settings",
                                 "permissions.allow_all"})
            CHECK(send(*f.s, name, {{"mode", "allow_all"}, {"who", "mcp"}}, o, o == CommandOrigin::agent ? "chat" : "")
                      .error.code == "command.unknown");
    // No argument of a real command sets it either: the schemas refuse what they do not name.
    CHECK(chat(*f.s, "panel.patch", {{"id", "loss"}, {"patch", {{"title", "X"}}}, {"mode", "allow_all"}}).error.code ==
          "command.invalid_args");
    CHECK(f.s->agent_mode(AgentPrincipal::mcp) == AgentMode::ask);
    CHECK(read_text(f.perms).find("allow_all") == std::string::npos);
    // A malformed client is refused, and one that is not "chat" is not governed.
    CHECK(send(*f.s, "spec.get", json::object(), CommandOrigin::agent, "Chat!").error.code ==
          "command.invalid_invocation");
    CHECK(f.s->agent_mode(AgentPrincipal::chat) == AgentMode::ask);
}

TEST_CASE("no change from an agent writes into the settings folder, whatever the mode") {
    // A workspace inside the folder of the permissions file.
    const std::string root = fresh_dir("modes-settings-folder");
    const std::string settings = root + "/app-data";
    const std::string ws = settings + "/workspace";
    write_text(ws + "/data/metrics.csv", "step,loss\n0,1.0\n");
    SessionOptions o;
    o.permissions_file = settings + "/agent-permissions.json";
    auto opened = Session::open(ws, o);
    REQUIRE(opened.error.ok());
    Session& s = *opened.session;
    REQUIRE(s.set_agent_mode(AgentPrincipal::chat, AgentMode::allow_all).ok());
    REQUIRE(s.set_agent_mode(AgentPrincipal::mcp, AgentMode::allow_all).ok());
    const std::string before = read_text(o.permissions_file);
    for (const auto& who : {std::pair{CommandOrigin::agent, "chat"}, std::pair{CommandOrigin::agent, "widget"},
                            std::pair{CommandOrigin::mcp, ""}, std::pair{CommandOrigin::api, ""}}) {
        const CommandOrigin origin = who.first;
        const std::string client = who.second;
        INFO(command_origin_name(origin) << " " << client);
        for (const auto& [name, args] : std::vector<std::pair<std::string, json>>{
                 {"data.import", {{"paths", {"data/metrics.csv"}}}},
                 {"spec.new", {{"id", "x"}}},
                 {"applet.save", json::object()}}) {
            const CommandResult r = send(s, name, args, origin, client);
            CHECK(r.error.code == "command.denied");
            CHECK(r.error.message.find("settings folder") != std::string::npos);
        }
        // Reads still run.
        CHECK(send(s, "loader.list", json::object(), origin, client).ok);
    }
    CHECK(s.registry().pending_approvals().empty());
    CHECK(read_text(o.permissions_file) == before);
    // The person's own UI is not refused.
    ok(s, "data.import", {{"paths", {"data/metrics.csv"}}});
}

#ifndef _WIN32
TEST_CASE("a link from a folder of the workspace into the settings folder refuses an agent's change") {
    Fixture f("modes-settings-link");
    REQUIRE(f.s->set_agent_mode(AgentPrincipal::chat, AgentMode::allow_all).ok());
    const std::string settings = fs::path(f.perms).parent_path().string();
    const std::string before = read_text(f.perms);
    for (const char* folder : {"applets", "loaders", "runs"}) {
        INFO(folder);
        const fs::path link = fs::path(f.ws) / folder;
        const fs::path aside = fs::path(f.ws) / (std::string(folder) + ".aside");
        if (fs::exists(link)) fs::rename(link, aside);
        fs::create_directory_symlink(settings, link);
        const CommandResult r = chat(*f.s, "applet.save", {{"id", "agent-permissions"}});
        CHECK(r.error.code == "command.denied");
        CHECK(r.error.message.find("settings folder") != std::string::npos);
        CHECK(send(*f.s, "loader.delete", {{"id", "x"}}, CommandOrigin::mcp, "").error.code == "command.denied");
        fs::remove(link);
        if (fs::exists(aside)) fs::rename(aside, link);
    }
    CHECK(read_text(f.perms) == before);
    CHECK_FALSE(fs::exists(fs::path(f.ws) / ".." / "app-data" / "agent-permissions"));
    // Without the link, the same change runs.
    CHECK(chat(*f.s, "applet.save", {{"id", "copy"}}).ok);
}
#endif
