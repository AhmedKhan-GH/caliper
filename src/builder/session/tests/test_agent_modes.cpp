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
    // Nor does a spec that names it, opened by the person or written by an agent.
    const std::string named = read_text(f.ws + "/applets/dash/dash.spec.toml") +
                              "\n[permissions]\nchat = \"allow_all\"\nmcp = \"allow_all\"\n";
    write_text(f.ws + "/named.spec.toml", named);
    invoke(*f.s, "spec.open", {{"path", "named.spec.toml"}});
    chat(*f.s, "spec.set_text", {{"text", named}});
    send(*f.s, "spec.set_text", {{"text", named}}, CommandOrigin::mcp, "");
    CHECK(f.s->agent_mode(AgentPrincipal::mcp) == AgentMode::ask);
    CHECK(read_text(f.perms).find("allow_all") == std::string::npos);
    // A malformed client is refused, and one that is not "chat" is not governed.
    CHECK(send(*f.s, "spec.get", json::object(), CommandOrigin::agent, "Chat!").error.code ==
          "command.invalid_invocation");
    CHECK(f.s->agent_mode(AgentPrincipal::chat) == AgentMode::ask);
}

TEST_CASE("a workspace in the settings folder, or that is the settings folder, works for agents") {
    const std::string root = fresh_dir("modes-settings-workspace");
    const std::string settings = root + "/app-data";
    for (const std::string& ws : {settings + "/workspace", settings}) {
        INFO(ws);
        write_text(ws + "/data/metrics.csv", "step,loss\n0,1.0\n1,0.5\n");
        SessionOptions o;
        o.permissions_file = settings + "/agent-permissions.json";
        o.protected_files = {settings + "/side-chat.json"};
        auto opened = Session::open(ws, o);
        REQUIRE(opened.error.ok());
        Session& s = *opened.session;
        REQUIRE(s.set_agent_mode(AgentPrincipal::chat, AgentMode::allow_all).ok());
        REQUIRE(s.set_agent_mode(AgentPrincipal::mcp, AgentMode::allow_all).ok());
        const std::string before = read_text(o.permissions_file);
        REQUIRE(send(s, "data.import", {{"paths", {"data/metrics.csv"}}}, CommandOrigin::agent, "chat").ok);
        REQUIRE(send(s, "spec.new", {{"id", "dash"}}, CommandOrigin::mcp, "").ok);
        REQUIRE(send(s, "source.add", {{"id", "m"}, {"catalog", "table:metrics"}}, CommandOrigin::mcp, "").ok);
        REQUIRE(send(s, "panel.create", {{"id", "loss"}, {"primitive", "series"}, {"bind", {{"x", "m.step"}, {"y", "m.loss"}}}},
                     CommandOrigin::agent, "chat")
                    .ok);
        const CommandResult saved = send(s, "applet.save", json::object(), CommandOrigin::agent, "chat");
        INFO(saved.error.code << ": " << saved.error.message);
        CHECK(saved.ok);
        CHECK(fs::exists(ws + "/applets/dash/dash.spec.toml"));
        CHECK(read_text(o.permissions_file) == before);
    }
}

TEST_CASE("no command writes a protected file, by its name, a relative path, a link, a case variant or a hard link") {
    // The hardest case: the workspace is the settings folder itself.
    const std::string root = fresh_dir("modes-protected-files");
    const std::string settings = root + "/app-data";
    const std::string perms = settings + "/agent-permissions.json";
    const std::string chat_file = settings + "/side-chat.json";
    write_text(settings + "/data/metrics.csv", "step,loss\n0,1.0\n1,0.5\n");
    write_text(chat_file, "{\"agent\": \"claude\"}");
    SessionOptions o;
    o.permissions_file = perms;
    o.protected_files = {chat_file};
    auto opened = Session::open(settings, o);
    REQUIRE(opened.error.ok());
    Session& s = *opened.session;
    REQUIRE(s.set_agent_mode(AgentPrincipal::chat, AgentMode::allow_all).ok());
    const std::string perms_text = read_text(perms);
    const std::string chat_text = read_text(chat_file);
    auto unchanged = [&] {
        CHECK(read_text(perms) == perms_text);
        CHECK(read_text(chat_file) == chat_text);
    };
    ok(s, "data.import", {{"paths", {"data/metrics.csv"}}});
    ok(s, "spec.new", {{"id", "dash"}});
    // Under the protected names: every write has its own folder and suffix.
    CHECK(chat(s, "applet.save", {{"id", "agent-permissions"}}).ok);
    CHECK(fs::exists(settings + "/applets/agent-permissions/agent-permissions.spec.toml"));
    unchanged();
    // A relative path or ".." in an id is refused before anything is written.
    for (const char* id : {"../agent-permissions", "../../app-data/side-chat", "applets/../agent-permissions"})
        CHECK_FALSE(chat(s, "applet.save", {{"id", id}}).ok);
    CHECK(chat(s, "loader.create", {{"text", "caliper_loader = 0\n[loader]\nid = \"../agent-permissions\"\n[input]\n"
                                             "path = \"data/metrics.csv\"\n[[output]]\nname = \"x\"\n"}})
              .error.code == "loader.invalid");
    unchanged();
#ifndef _WIN32
    // A link from an applet's spec file to each protected file, by its name
    // and by a case variant: the save is refused even after the person
    // approved the overwrite.
    auto approve_save = [&](const std::string& id) {
        const CommandResult r = chat(s, "applet.save", {{"id", id}, {"overwrite", true}});
        if (r.error.code != "command.needs_approval") return r;
        return s.registry().approve(r.approval);
    };
    const std::vector<std::pair<std::string, std::string>> links = {
        {"linked", perms}, {"chatlink", chat_file}, {"casevariant", settings + "/AGENT-PERMISSIONS.JSON"}};
    for (const auto& link : links) {
        INFO(link.first);
        fs::create_directories(settings + "/applets/" + link.first);
        fs::create_symlink(link.second, settings + "/applets/" + link.first + "/" + link.first + ".spec.toml");
        const CommandResult r = approve_save(link.first);
        CHECK(r.error.code == "applet.write_failed");
        CHECK(r.error.message.find("protected") != std::string::npos);
        unchanged();
    }
    // A hard link to the permissions file under an applet's name.
    fs::create_directories(settings + "/applets/hard");
    fs::create_hard_link(perms, settings + "/applets/hard/hard.spec.toml");
    CHECK(approve_save("hard").error.code == "applet.write_failed");
    unchanged();
    // A loader file that is a link to the permissions file.
    fs::create_directories(settings + "/loaders");
    fs::create_symlink(perms, settings + "/loaders/linked.loader.toml");
    const CommandResult made = chat(s, "loader.create", {{"text", "caliper_loader = 0\n[loader]\nid = \"linked\"\n[input]\n"
                                                                  "path = \"data/metrics.csv\"\n[[output]]\nname = \"l\"\n"}});
    CHECK(made.error.code == "loader.protected_file");
    unchanged();
    // A folder of loader outputs that leads to the folder above the settings
    // folder: deleting the loader app-data, whose outputs folder would be the
    // settings folder, leaves it in place.
    write_text(settings + "/data/app-data.csv", "step,acc\n0,0.5\n");
    const json imported = ok(s, "data.import", {{"paths", {"data/app-data.csv"}}});
    REQUIRE(imported["loaders"][0]["id"] == "app-data");
    fs::remove_all(settings + "/loader-outputs");
    fs::create_directory_symlink(root, settings + "/loader-outputs");
    CHECK(chat(s, "loader.delete", {{"id", "app-data"}}).ok);
    CHECK(fs::exists(perms));
    CHECK(fs::exists(settings + "/data/metrics.csv"));
    unchanged();
#endif
}
