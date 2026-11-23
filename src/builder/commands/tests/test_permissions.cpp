// Tests of agent permissions (permissions.h): the decision of each mode for
// each class of command, the rules and what beats them, the classification of
// the descriptor table, and the saved file, which grants nothing when it is
// malformed or others may write it.
#include <caliper_builder/commands/command_table.h>
#include <caliper_builder/commands/permissions.h>

#include <doctest/doctest.h>
#include <nlohmann/json.hpp>

#include <filesystem>
#include <fstream>

#ifndef _WIN32
#include <sys/stat.h>
#include <unistd.h>
#endif

using namespace caliper_builder;
using nlohmann::json;
namespace fs = std::filesystem;

namespace {

const CommandDescriptor& cmd(const char* name) {
    const CommandDescriptor* d = find_command_descriptor(name);
    REQUIRE_MESSAGE(d, name);
    return *d;
}

CommandInvocation chat(const char* name, json args = json::object()) {
    CommandInvocation inv;
    inv.name = name;
    inv.args = std::move(args);
    inv.origin = CommandOrigin::agent;
    inv.client = "chat";
    return inv;
}

PermissionVerdict decide(const char* name, AgentMode mode, ApprovalDecision base, json args = json::object(),
                         PermissionContext c = {}) {
    c.mode = mode;
    return decide_agent_permission(cmd(name), chat(name, std::move(args)), base, c);
}

std::string dir(const std::string& name) {
    const fs::path d = fs::path(CALIPER_BUILDER_TEST_TMP) / "permissions" / name;
    std::error_code ec;
    fs::remove_all(d, ec);
    fs::create_directories(d);
    return d.string();
}

void write(const std::string& path, const std::string& text) {
    std::ofstream(path, std::ios::binary | std::ios::trunc) << text;
}

}  // namespace

TEST_CASE("every command of the table says what it does, and the classes agree with the flags") {
    CHECK(command_table_build_errors().empty());
    for (const auto& d : command_table()) {
        INFO(d.name);
        REQUIRE(d.effect != CommandEffect::unset);
        if (d.effect == CommandEffect::read) CHECK_FALSE(d.mutates);
        if (d.effect == CommandEffect::edit) {
            CHECK(d.mutates);
            CHECK_FALSE(d.destructive);  // an edit of the open spec is never destructive
        }
        if (d.destructive) CHECK(d.effect == CommandEffect::change);
        if (d.effect == CommandEffect::run) {
            // Acting on the running script changes no workspace state and has no undo.
            CHECK_FALSE(d.mutates);
            CHECK_FALSE(d.destructive);
        }
        CHECK(command_descriptor_to_json(d)["effect"] == command_effect_name(d.effect));
    }
    // The run commands of design/instruments.md act on the running script.
    for (const char* name : {"run.set_param", "run.action", "run.pause"}) CHECK(cmd(name).effect == CommandEffect::run);
    CHECK(cmd("run.state").effect == CommandEffect::read);
    // The model runner's commands of design/hosted-runs.md: a start, a fork and
    // a stop act on runs; the list and the logs read.
    for (const char* name : {"run.start", "run.fork", "run.stop"}) CHECK(cmd(name).effect == CommandEffect::run);
    for (const char* name : {"run.list", "run.logs"}) CHECK(cmd(name).effect == CommandEffect::read);
    CHECK(cmd("events.wait").effect == CommandEffect::read);
    // design/instrument-interface.md, Commands: the edits are edits of the open spec, and the
    // commands that set state, a control, a pick, the cursor or the camera change the session.
    for (const char* name : {"state.create", "state.patch", "state.remove", "entity.set", "entity.remove",
                             "annotation.set", "annotation.remove", "camera.save"})
        CHECK(cmd(name).effect == CommandEffect::edit);
    for (const char* name : {"state.set", "control.set", "selection.set", "cursor.set", "cursor.step", "camera.set"})
        CHECK(cmd(name).effect == CommandEffect::change);
    for (const char* name : {"state.get", "selection.get", "cursor.get", "camera.get"})
        CHECK(cmd(name).effect == CommandEffect::read);
    // design/gpu-programs.md: programs are edits of the open spec,
    // and an intervention acts on the running script.
    for (const char* name : {"program.create", "program.patch", "program.remove"})
        CHECK(cmd(name).effect == CommandEffect::edit);
    for (const char* name : {"program.status", "piece.check", "piece.list", "piece.show", "program.read",
                             "intervene.points", "intervene.list"})
        CHECK(cmd(name).effect == CommandEffect::read);
    // design/programmable-graphics.md: the piece commands that write the
    // workspace's pieces/ folder change it as the loader commands do.
    for (const char* name : {"piece.save", "piece.copy", "piece.update", "piece.remove"})
        CHECK(cmd(name).effect == CommandEffect::change);
    for (const char* name : {"intervene.set", "intervene.clear"}) CHECK(cmd(name).effect == CommandEffect::run);
    CHECK(command_effect_from_name("run") == CommandEffect::run);
    // Replacing the open spec is not a plain edit.
    for (const char* name : {"spec.new", "spec.open", "applet.open", "app.launch"})
        CHECK(cmd(name).effect == CommandEffect::change);
    // The JSON form of the table carries the class, and an unset one reads as change.
    CommandDescriptor bare;
    CHECK(command_descriptor_to_json(bare)["effect"] == "change");
}

TEST_CASE("Ask keeps the policy's decision, and only the chat's agent and mcp clients are governed") {
    for (auto base : {ApprovalDecision::allow, ApprovalDecision::ask, ApprovalDecision::deny})
        CHECK(decide("panel.patch", AgentMode::ask, base).decision == base);
    // The widget agent, the person's api token and the UI are never governed.
    PermissionContext all;
    all.mode = AgentMode::allow_all;
    for (auto [origin, client] : {std::pair{CommandOrigin::agent, "widget"}, std::pair{CommandOrigin::agent, ""},
                                  std::pair{CommandOrigin::api, ""}, std::pair{CommandOrigin::ui, ""}}) {
        CommandInvocation inv = chat("loader.delete");
        inv.origin = origin;
        inv.client = client;
        CHECK(agent_principal(inv) == AgentPrincipal::none);
        CHECK(decide_agent_permission(cmd("loader.delete"), inv, ApprovalDecision::ask, all).decision ==
              ApprovalDecision::ask);
    }
    CommandInvocation mcp = chat("panel.patch");
    mcp.origin = CommandOrigin::mcp;
    mcp.client.clear();
    CHECK(agent_principal(mcp) == AgentPrincipal::mcp);
}

TEST_CASE("Accept edits applies an edit of the open spec and nothing else") {
    const auto edit = decide("panel.patch", AgentMode::accept_edits, ApprovalDecision::ask);
    CHECK(edit.decision == ApprovalDecision::allow);
    CHECK(edit.allowed_by == "accept_edits");
    // A change, and a replacement of the open spec, still ask.
    for (const char* name : {"loader.run", "loader.delete", "applet.save", "spec.new", "data.import"})
        CHECK(decide(name, AgentMode::accept_edits, ApprovalDecision::ask).decision == ApprovalDecision::ask);
    // A run command is not an edit: it still asks.
    CHECK(decide("run.set_param", AgentMode::accept_edits, ApprovalDecision::ask).decision == ApprovalDecision::ask);
    // What the policy allowed stays allowed; a denial stays a denial.
    CHECK(decide("loader.create", AgentMode::accept_edits, ApprovalDecision::allow).decision == ApprovalDecision::allow);
    CHECK(decide("panel.patch", AgentMode::accept_edits, ApprovalDecision::deny).decision == ApprovalDecision::deny);
}

TEST_CASE("Read only refuses every change with a plain message and lets reads and proposals run") {
    for (const auto& d : command_table()) {
        if (!d.agents) continue;
        INFO(d.name);
        const auto v = decide_agent_permission(d, chat(d.name.c_str()), ApprovalDecision::allow,
                                               PermissionContext{AgentMode::read_only});
        if (d.effect == CommandEffect::read) {
            CHECK(v.decision == ApprovalDecision::allow);
        } else {
            CHECK(v.decision == ApprovalDecision::deny);
            CHECK(v.message.find("Read only") != std::string::npos);
        }
    }
    // A dry run and an edit into a preview are proposals.
    CHECK(decide("panel.patch", AgentMode::read_only, ApprovalDecision::allow, {{"dry_run", true}}).decision ==
          ApprovalDecision::allow);
    CommandInvocation into = chat("panel.patch");
    into.preview = "p1";
    CHECK(decide_agent_permission(cmd("panel.patch"), into, ApprovalDecision::allow, {AgentMode::read_only}).decision ==
          ApprovalDecision::allow);
}

TEST_CASE("Allow all removes every card but a data-loss card, for the chat's agent and for mcp clients") {
    const auto v = decide("loader.delete", AgentMode::allow_all, ApprovalDecision::ask);
    CHECK(v.decision == ApprovalDecision::allow);
    CHECK(v.allowed_by == "allow_all");
    // Running Python code, which no rule covers: a loader, a managed run's start or fork.
    PermissionContext python;
    python.python = true;
    for (const char* name : {"loader.run", "run.start", "run.fork"})
        CHECK(decide(name, AgentMode::allow_all, ApprovalDecision::ask, json::object(), python).decision ==
              ApprovalDecision::allow);
    CHECK(decide("panel.patch", AgentMode::allow_all, ApprovalDecision::deny).decision == ApprovalDecision::deny);
    // Every command of the table, for both principals: what asks runs, unless
    // its card is bound to the content it would lose. A destructive command
    // declared later keeps its card by binding it.
    PermissionContext bound;
    bound.bound = true;
    for (const auto& d : command_table()) {
        if (!d.agents) continue;
        INFO(d.name);
        CommandInvocation c = chat(d.name.c_str());
        CommandInvocation m = c;
        m.origin = CommandOrigin::mcp;
        m.client.clear();
        for (const CommandInvocation& inv : {c, m}) {
            CHECK(decide_agent_permission(d, inv, ApprovalDecision::ask, {AgentMode::allow_all}).decision ==
                  ApprovalDecision::allow);
            bound.mode = AgentMode::allow_all;
            CHECK(decide_agent_permission(d, inv, ApprovalDecision::ask, bound).decision == ApprovalDecision::ask);
        }
    }
}

TEST_CASE("rules allow their command; Read only, content binding and a Python run beat them") {
    PermissionContext c;
    c.session_rule = true;
    auto v = decide("loader.delete", AgentMode::ask, ApprovalDecision::ask, json::object(), c);
    CHECK(v.decision == ApprovalDecision::allow);
    CHECK(v.allowed_by == "session_rule");
    c = {};
    c.saved_rule = true;
    v = decide("loader.delete", AgentMode::ask, ApprovalDecision::ask, json::object(), c);
    CHECK(v.allowed_by == "saved_rule");
    CHECK(decide("loader.delete", AgentMode::read_only, ApprovalDecision::ask, json::object(), c).decision ==
          ApprovalDecision::deny);
    c.bound = true;
    CHECK(decide("applet.save", AgentMode::ask, ApprovalDecision::ask, json::object(), c).decision ==
          ApprovalDecision::ask);
    c.bound = false;
    c.python = true;
    CHECK(decide("loader.run", AgentMode::ask, ApprovalDecision::ask, json::object(), c).decision ==
          ApprovalDecision::ask);
    // A rule names a command that changes something and is open to agents.
    CHECK(rule_command_allowed("loader.delete"));
    CHECK_FALSE(rule_command_allowed("history.undo"));
    CHECK_FALSE(rule_command_allowed("spec.get"));
    CHECK_FALSE(rule_command_allowed("no.such"));
}

TEST_CASE("the saved permissions round-trip owner-only, and a bad file grants nothing") {
    const std::string d = dir("file");
    const std::string path = d + "/agent-permissions.json";
    CHECK(read_agent_permissions(path).problems.empty());  // missing: the defaults
    AgentPermissionsFile f;
    f.chat = AgentMode::accept_edits;
    f.mcp = AgentMode::allow_all;
    f.rules = {{"claude", "panel.patch"}, {"mcp", "loader.delete"}};
    REQUIRE(write_agent_permissions(path, f).ok());
#ifndef _WIN32
    struct stat st {};
    REQUIRE(::stat(path.c_str(), &st) == 0);
    CHECK((st.st_mode & 0777) == 0600);
#endif
    AgentPermissionsRead r = read_agent_permissions(path);
    CHECK(r.problems.empty());
    CHECK(r.file.chat == AgentMode::accept_edits);
    CHECK(r.file.mcp == AgentMode::allow_all);
    CHECK(r.file.rules == f.rules);

    // Unknown fields, bad agents and commands that are no agent's are ignored
    // and reported, and Allow all with them, in a file with a problem.
    write(path, R"({"version": 1, "extra": 1, "modes": {"chat": "allow_all", "mcp": "read_only", "x": "ask"},
                   "rules": [{"agent": "claude", "command": "history.undo"}, {"agent": "../x", "command": "panel.patch"},
                             {"agent": "codex", "command": "panel.patch", "more": 1}, "no",
                             {"agent": "codex", "command": "source.add"}]})");
#ifndef _WIN32
    ::chmod(path.c_str(), 0600);
#endif
    r = read_agent_permissions(path);
    CHECK(r.file.chat == AgentMode::ask);
    CHECK(r.file.mcp == AgentMode::read_only);
    CHECK(r.file.rules == std::vector<AgentRule>{{"codex", "source.add"}});
    CHECK(r.problems.size() == 7);

    for (const char* bad : {"not json", "[]", R"({"version": 2, "rules": [{"agent": "claude", "command": "panel.patch"}]})"}) {
        write(path, bad);
        r = read_agent_permissions(path);
        CHECK(r.file.rules.empty());
        CHECK(r.file.chat == AgentMode::ask);
        CHECK(r.problems.size() == 1);
    }
#ifndef _WIN32
    // A file others may write grants nothing.
    REQUIRE(write_agent_permissions(path, f).ok());
    ::chmod(path.c_str(), 0666);
    r = read_agent_permissions(path);
    CHECK(r.file.rules.empty());
    CHECK(r.file.chat == AgentMode::ask);
    CHECK(r.problems.size() == 1);
#endif
}

TEST_CASE("the modes' names, words and cycle") {
    for (AgentMode m : {AgentMode::ask, AgentMode::accept_edits, AgentMode::read_only, AgentMode::allow_all})
        CHECK(agent_mode_from_name(agent_mode_name(m)) == m);
    CHECK(std::string(agent_mode_label(AgentMode::accept_edits)) == "Accept edits");
    CHECK(next_agent_mode(AgentMode::ask) == AgentMode::accept_edits);
    CHECK(next_agent_mode(AgentMode::accept_edits) == AgentMode::read_only);
    CHECK(next_agent_mode(AgentMode::read_only) == AgentMode::ask);
    CHECK(next_agent_mode(AgentMode::allow_all) == AgentMode::ask);
}

TEST_CASE("a rule of a run command names one parameter or one action") {
    // The key of run.set_param and run.action carries the target, so a card's
    // rule allows "changes of lr" and not every parameter.
    CHECK(permission_rule_key(chat("run.set_param", {{"name", "lr"}, {"value", 0.01}})) == "run.set_param:lr");
    CHECK(permission_rule_key(chat("run.action", {{"name", "save_checkpoint"}})) == "run.action:save_checkpoint");
    CHECK(permission_rule_key(chat("run.pause", {{"op", "pause"}})) == "run.pause");
    CHECK(permission_rule_key(chat("panel.patch", {{"id", "lr"}})) == "panel.patch");
    // A request without a valid name gets a key that no rule holds.
    CHECK(permission_rule_key(chat("run.set_param", {{"value", 1}})) == "run.set_param:");
    CHECK(permission_rule_key(chat("run.action", {{"name", std::string("a\nb")}})) == "run.action:");
    CHECK(permission_rule_key(chat("run.action", {{"name", std::string(64, 'a')}})) == "run.action:");
    CHECK_FALSE(rule_command_allowed("run.action:"));

    CHECK(rule_command_allowed("run.set_param:lr"));
    CHECK(rule_command_allowed("run.action:save_checkpoint"));
    CHECK(rule_command_allowed("run.set_param:" + std::string(63, 'x')));
    CHECK_FALSE(rule_command_allowed("run.set_param:" + std::string(64, 'x')));
    CHECK_FALSE(rule_command_allowed("run.set_param:a\tb"));
    CHECK_FALSE(rule_command_allowed("run.pause:x"));       // takes no target
    CHECK_FALSE(rule_command_allowed("panel.patch:lr"));
    CHECK_FALSE(rule_command_allowed("run.state:lr"));      // a read
    CHECK(rule_command_allowed("run.pause"));
    // The bare name passes as the card's check of the command.
    CHECK(rule_command_allowed("run.set_param"));

    // A rule of the session or a saved rule allows a run command's card.
    PermissionContext c;
    c.saved_rule = true;
    CHECK(decide("run.set_param", AgentMode::ask, ApprovalDecision::ask, {{"name", "lr"}, {"value", 0.01}}, c).decision ==
          ApprovalDecision::allow);
    // Accept edits does not cover a run command; Read only refuses it; Allow all runs it.
    CHECK(decide("run.set_param", AgentMode::accept_edits, ApprovalDecision::ask, {{"name", "lr"}, {"value", 0.01}})
              .decision == ApprovalDecision::ask);
    CHECK(decide("run.pause", AgentMode::read_only, ApprovalDecision::ask, {{"op", "pause"}}).decision ==
          ApprovalDecision::deny);
    CHECK(decide("run.action", AgentMode::allow_all, ApprovalDecision::ask, {{"name", "save"}}).decision ==
          ApprovalDecision::allow);
}

TEST_CASE("a rule of an intervention names one point, and clearing all always asks") {
    CHECK(permission_rule_key(chat("intervene.set", {{"point", "blocks.0.attn#3"}, {"op", "ablate"}})) ==
          "intervene.set:blocks.0.attn#3");
    CHECK(permission_rule_key(chat("intervene.clear", {{"point", "blocks.0.attn"}})) == "intervene.clear:blocks.0.attn");
    CHECK(permission_rule_key(chat("intervene.clear", {{"all", true}})) == "intervene.clear:");
    CHECK(permission_rule_key(chat("intervene.set", {{"name", "head"}, {"op", "ablate"}})) == "intervene.set:");
    CHECK(permission_rule_key(chat("intervene.set", {{"point", std::string(255, 'p')}, {"op", "ablate"}})) ==
          "intervene.set:" + std::string(255, 'p'));
    CHECK(permission_rule_key(chat("intervene.set", {{"point", std::string(256, 'p')}, {"op", "ablate"}})) ==
          "intervene.set:");
    CHECK(permission_rule_key(chat("intervene.points", {{"match", "blocks.*"}})) == "intervene.points");

    CHECK(rule_command_allowed("intervene.set:blocks.0.attn#3"));
    CHECK(rule_command_allowed("intervene.clear:head"));
    CHECK(rule_command_allowed("intervene.set:" + std::string(255, 'x')));
    CHECK_FALSE(rule_command_allowed("intervene.set:" + std::string(256, 'x')));
    CHECK_FALSE(rule_command_allowed("intervene.clear:"));
    CHECK_FALSE(rule_command_allowed("intervene.list:head"));   // a read
    CHECK(rule_command_allowed("intervene.set"));               // the card's check of the command

    PermissionContext c;
    c.saved_rule = true;
    CHECK(decide("intervene.set", AgentMode::ask, ApprovalDecision::ask, {{"point", "head"}, {"op", "ablate"}}, c)
              .decision == ApprovalDecision::allow);
    CHECK(decide("intervene.set", AgentMode::accept_edits, ApprovalDecision::ask, {{"point", "head"}, {"op", "ablate"}})
              .decision == ApprovalDecision::ask);
    CHECK(decide("intervene.clear", AgentMode::read_only, ApprovalDecision::ask, {{"all", true}}).decision ==
          ApprovalDecision::deny);

    // A saved rule of an intervention without its point is ignored.
    const std::string path = dir("intervene") + "/agent-permissions.json";
    write(path, R"({"version": 1, "rules": [{"agent": "mcp", "command": "intervene.set"},
                   {"agent": "mcp", "command": "intervene.set:blocks.0.attn#3"}]})");
#ifndef _WIN32
    ::chmod(path.c_str(), 0600);
#endif
    const AgentPermissionsRead r = read_agent_permissions(path);
    CHECK(r.file.rules == std::vector<AgentRule>{{"mcp", "intervene.set:blocks.0.attn#3"}});
    CHECK(r.problems.size() == 1);
}

TEST_CASE("saved rules with a target round-trip, and a run rule without its target is ignored") {
    const std::string d = dir("targets");
    const std::string path = d + "/agent-permissions.json";
    AgentPermissionsFile f;
    f.rules = {{"mcp", "run.set_param:lr"}, {"claude", "run.action:save_checkpoint"}, {"mcp", "run.pause"}};
    REQUIRE(write_agent_permissions(path, f).ok());
    AgentPermissionsRead r = read_agent_permissions(path);
    CHECK(r.problems.empty());
    CHECK(r.file.rules == f.rules);

    write(path, R"({"version": 1, "rules": [{"agent": "mcp", "command": "run.set_param"},
                   {"agent": "mcp", "command": "run.action:"}, {"agent": "mcp", "command": "run.pause:x"},
                   {"agent": "mcp", "command": "run.set_param:momentum"}]})");
#ifndef _WIN32
    ::chmod(path.c_str(), 0600);
#endif
    r = read_agent_permissions(path);
    CHECK(r.file.rules == std::vector<AgentRule>{{"mcp", "run.set_param:momentum"}});
    CHECK(r.problems.size() == 3);
}

TEST_CASE("Allow all is read only from a clean file, and every other file leaves it at Ask") {
    const std::string d = dir("allow-all");
    const std::string path = d + "/agent-permissions.json";
    auto read = [&](const std::string& text) {
        write(path, text);
#ifndef _WIN32
        ::chmod(path.c_str(), 0600);
#endif
        return read_agent_permissions(path);
    };
    // A clean file: Allow all for each principal, at launch as it was saved.
    AgentPermissionsRead r = read(R"({"version": 1, "modes": {"chat": "allow_all", "mcp": "allow_all"}, "rules": []})");
    CHECK(r.problems.empty());
    CHECK(r.file.chat == AgentMode::allow_all);
    CHECK(r.file.mcp == AgentMode::allow_all);

    // Unknown keys, wrong types, other spellings and a malformed file never turn it on.
    for (const char* text : {
             R"({"version": 1, "modes": {"chat": "allow_all", "mcp": "allow_all"}, "allow_all": true})",
             R"({"version": 1, "modes": {"chat": "allow_all", "mcp": "allow_all", "widget": "allow_all"}})",
             R"({"version": 1, "modes": {"chat": "allow_all", "mcp": 1}})",
             R"({"version": 1, "modes": {"chat": "allow_all", "mcp": "allow_all"}, "rules": {"agent": "mcp"}})",
             R"({"version": 1, "modes": {"chat": "allow_all", "mcp": "allow_all"}, "rules": ["x"]})",
             R"({"version": 1, "modes": {"chat": true, "mcp": ["allow_all"]}})",
             R"({"version": 1, "modes": {"chat": "Allow all", "mcp": "ALLOW_ALL"}})",
             R"({"version": 1, "modes": {"chat": "allow-all", "mcp": "allow_all "}})",
             R"({"version": 1, "modes": [["chat", "allow_all"]]})",
             R"({"version": "1", "modes": {"chat": "allow_all", "mcp": "allow_all"}})",
             R"({"modes": {"chat": "allow_all", "mcp": "allow_all"}})",
             R"({"version": 1, "modes": {"chat": "allow_all", "mcp": "allow_all"})",
             R"({"version": 1, "modes": {"chat": "allow_all", "mcp": "allow_all"}} trailing)",
             "",
             "null"}) {
        INFO(text);
        r = read(text);
        CHECK(r.file.chat == AgentMode::ask);
        CHECK(r.file.mcp == AgentMode::ask);
        CHECK_FALSE(r.problems.empty());
    }
#ifndef _WIN32
    // A clean file that others may write grants nothing.
    write(path, R"({"version": 1, "modes": {"chat": "allow_all", "mcp": "allow_all"}})");
    ::chmod(path.c_str(), 0620);
    r = read_agent_permissions(path);
    CHECK(r.file.chat == AgentMode::ask);
    CHECK(r.file.mcp == AgentMode::ask);
    // A clean file that another user owns grants nothing.
    write(path, R"({"version": 1, "modes": {"chat": "allow_all", "mcp": "allow_all"}})");
    ::chmod(path.c_str(), 0600);
    REQUIRE(read_agent_permissions(path).file.chat == AgentMode::allow_all);
    r = read_agent_permissions(path, static_cast<long>(::geteuid()) + 1);
    CHECK(r.file.chat == AgentMode::ask);
    CHECK(r.file.mcp == AgentMode::ask);
    CHECK(r.problems == std::vector<std::string>{"The agent permissions file belongs to another user, so it grants nothing."});
    // Saving over a file of another mode leaves it owner-only.
    ::chmod(path.c_str(), 0644);
    AgentPermissionsFile f;
    f.chat = AgentMode::allow_all;
    REQUIRE(write_agent_permissions(path, f).ok());
    struct stat st {};
    REQUIRE(::stat(path.c_str(), &st) == 0);
    CHECK((st.st_mode & 0777) == 0600);
    CHECK(read_agent_permissions(path).file.chat == AgentMode::allow_all);
#endif
}
