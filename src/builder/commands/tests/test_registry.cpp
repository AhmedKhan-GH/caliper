// Tests of CommandRegistry: registration, every common error, handlers that
// throw, undo and redo by closures and by inverse invocations, rollback of a
// failed undo or redo, approval (also of history moves) and the journal.
#include <caliper_builder/commands/registry.h>

#include <doctest/doctest.h>

#include <filesystem>
#include <map>
#include <stdexcept>
#include <string>
#include <thread>

using caliper_builder::ApprovalDecision;
using caliper_builder::CommandContext;
using caliper_builder::CommandDescriptor;
using caliper_builder::CommandEffect;
using caliper_builder::CommandInvocation;
using caliper_builder::CommandJournal;
using caliper_builder::CommandOrigin;
using caliper_builder::CommandOutput;
using caliper_builder::CommandRegistry;
using caliper_builder::CommandResult;
using caliper_builder::RegistryOptions;
using caliper_builder::UndoRecord;
using nlohmann::json;
namespace cb = caliper_builder;
namespace fs = std::filesystem;

namespace {

CommandDescriptor descriptor(const std::string& name, bool mutates, bool destructive = false,
                             const char* args = R"({"type": "object", "properties": {"key": {"type": "string"},
                                                    "value": {"type": "integer"}}, "additionalProperties": false})") {
    CommandDescriptor d;
    d.name = name;
    d.summary = "A test command.";
    d.args = json::parse(args);
    d.result = json::parse(R"({"type": "object"})");
    d.mutates = mutates;
    d.destructive = destructive;
    return d;
}

// A store of integers edited by toy commands, each undone by a closure pair.
struct Store {
    std::map<std::string, int> values;
    int fail_undo_of = -1;  // the value whose undo fails
    int fail_redo_of = -1;  // the value whose redo fails
};

CommandRegistry::HistoryState state(const CommandRegistry& r) { return r.history_state(); }

CommandInvocation inv(const std::string& name, json args = json::object(), CommandOrigin origin = CommandOrigin::ui,
                      const std::string& group = {}, const std::string& merge_key = {}) {
    CommandInvocation i;
    i.name = name;
    i.args = std::move(args);
    i.origin = origin;
    i.group = group;
    i.merge_key = merge_key;
    return i;
}

// toy.set {key, value}: sets a value; undo restores the previous one.
void add_toy(CommandRegistry& r, Store& s, bool destructive = false) {
    REQUIRE(r.add(descriptor("toy.set", true, destructive), [&s](const CommandContext& c) {
                 const std::string key = c.invocation.args.value("key", "k");
                 const int value = c.invocation.args.value("value", 0);
                 const bool had = s.values.count(key) > 0;
                 const int before = had ? s.values[key] : 0;
                 s.values[key] = value;
                 return CommandOutput::ok(
                     {{"key", key}},
                     UndoRecord::closures(
                         [&s, key, before, had, value] {
                             if (value == s.fail_undo_of) return cb::Error("toy.conflict", "The value changed.");
                             if (had) s.values[key] = before;
                             else s.values.erase(key);
                             return cb::Error();
                         },
                         [&s, key, value] {
                             if (value == s.fail_redo_of) return cb::Error("toy.conflict", "Cannot redo.");
                             s.values[key] = value;
                             return cb::Error();
                         }));
             }).ok());
}

}  // namespace

TEST_CASE("registration checks descriptors and keeps the history commands built in") {
    CommandRegistry r;
    const auto names = r.descriptors();
    REQUIRE(names.size() == 3);
    CHECK(names[0].name == "history.undo");
    CHECK(r.is_bound("history.list"));
    CHECK(r.add(descriptor("toy.get", false)).ok());
    CHECK(r.add(descriptor("toy.get", false)).code == "command.bad_descriptor");
    CHECK(r.add(descriptor("Toy.get", false)).code == "command.bad_descriptor");
    CHECK_FALSE(r.is_bound("toy.get"));
    CHECK(r.bind("toy.get", [](const CommandContext&) { return CommandOutput::ok({}); }).ok());
    CHECK(r.is_bound("toy.get"));
    CHECK(r.bind("toy.nope", {}).code == "command.unknown");
    CHECK(r.bind("history.undo", {}).code == "command.bad_descriptor");
    CHECK(r.add_all({descriptor("toy.get", false), descriptor("toy.other", false)}).ok());
    CHECK(r.descriptors().size() == 5);
    CHECK(r.descriptor("toy.other"));
    CHECK(!r.descriptor("toy.missing"));
}

TEST_CASE("every refusal before the handler runs has its own code") {
    CommandRegistry r;
    REQUIRE(r.add(descriptor("toy.unbound", false)).ok());
    REQUIRE(r.add(descriptor("toy.get", false), [](const CommandContext&) { return CommandOutput::ok({}); }).ok());

    CommandInvocation bad = inv("toy.get");
    bad.id = "has space";
    CHECK(r.invoke(bad).error.code == "command.invalid_invocation");
    CHECK(r.invoke(inv("toy.missing")).error.code == "command.unknown");
    CHECK(r.invoke(inv("toy.unbound")).error.code == "command.not_bound");
    const auto invalid = r.invoke(inv("toy.get", {{"value", "seven"}}));
    CHECK(invalid.error.code == "command.invalid_args");
    CHECK(invalid.error.where == "args.value");
    CHECK(r.invoke(inv("toy.get", {{"other", 1}})).error.where == "args.other");

    const auto ok = r.invoke(inv("toy.get"));
    CHECK(ok.ok);
    CHECK(ok.id == "c-5");  // one counter, refused invocations included
    CommandInvocation named = inv("toy.get");
    named.id = "mine";
    CHECK(r.invoke(named).id == "mine");
}

TEST_CASE("the viewer origin only reads: a command that mutates or runs is refused before its handler") {
    CommandRegistry r;
    int ran = 0;
    const auto count = [&ran](const CommandContext&) {
        ++ran;
        return CommandOutput::ok({});
    };
    CommandDescriptor run = descriptor("toy.run", false);
    run.effect = CommandEffect::run;
    REQUIRE(r.add(run, count).ok());
    REQUIRE(r.add(descriptor("toy.get", false), count).ok());

    const auto refused = r.invoke(inv("toy.run", {}, CommandOrigin::viewer));
    CHECK(refused.error.code == "command.not_allowed");
    CHECK(refused.error.where == "origin");
    CHECK(refused.error.message.find("running script") != std::string::npos);
    CHECK(ran == 0);
    CHECK(r.invoke(inv("toy.get", {}, CommandOrigin::viewer)).ok);
    CHECK(r.invoke(inv("toy.run", {}, CommandOrigin::ui)).ok);
    CHECK(ran == 2);
}

TEST_CASE("a frame_thread command runs only on the frame thread once one is named") {
    CommandRegistry r;
    CommandDescriptor d = descriptor("toy.draw", false);
    d.frame_thread = true;
    REQUIRE(r.add(d, [](const CommandContext&) { return CommandOutput::ok({}); }).ok());
    REQUIRE(r.add(descriptor("toy.read", false), [](const CommandContext&) { return CommandOutput::ok({}); }).ok());
    CHECK(r.invoke(inv("toy.draw")).ok);  // no frame thread named yet
    std::thread other([] {});
    const auto other_id = other.get_id();
    other.join();
    r.set_frame_thread(other_id);
    CHECK(r.invoke(inv("toy.draw")).error.code == "command.wrong_thread");
    CHECK(r.invoke(inv("toy.read")).ok);
    r.set_frame_thread(std::this_thread::get_id());
    CHECK(r.invoke(inv("toy.draw")).ok);
}

TEST_CASE("no handler throws across the registry, and handlers cannot invoke commands") {
    CommandRegistry r;
    REQUIRE(r.add(descriptor("toy.throw", true), [](const CommandContext&) -> CommandOutput {
                 throw std::runtime_error("disk on fire");
             }).ok());
    REQUIRE(r.add(descriptor("toy.weird", false), [](const CommandContext&) -> CommandOutput { throw 42; }).ok());
    REQUIRE(r.add(descriptor("toy.nest", false), [&r](const CommandContext&) {
                 const CommandResult inner = r.invoke(inv("toy.weird"));
                 return CommandOutput::fail(inner.error);
             }).ok());
    const auto thrown = r.invoke(inv("toy.throw"));
    CHECK(thrown.error.code == "command.handler_failed");
    CHECK(thrown.error.message == "The handler of toy.throw failed: disk on fire");
    CHECK(r.invoke(inv("toy.weird")).error.code == "command.handler_failed");
    CHECK(r.invoke(inv("toy.nest")).error.code == "command.reentrant");
    // The history is untouched by a failure.
    CHECK_FALSE(state(r).can_undo);
}

TEST_CASE("results are checked against their schema when asked") {
    RegistryOptions options;
    options.check_results = true;
    CommandRegistry r(options);
    CommandDescriptor d = descriptor("toy.count", false);
    d.result = json::parse(R"({"type": "object", "properties": {"n": {"type": "integer"}}, "required": ["n"]})");
    REQUIRE(r.add(d, [](const CommandContext& c) {
                 if (c.invocation.args.value("value", 0) == 1) return CommandOutput::ok({{"n", "one"}});
                 return CommandOutput::ok({{"n", 2}});
             }).ok());
    CHECK(r.invoke(inv("toy.count")).ok);
    const auto bad = r.invoke(inv("toy.count", {{"value", 1}}));
    CHECK(bad.error.code == "command.invalid_result");
    CHECK(bad.error.where == "result.n");
}

TEST_CASE("undo and redo by closure pairs, and the redo rule") {
    CommandRegistry r;
    Store s;
    add_toy(r, s);
    REQUIRE(r.invoke(inv("toy.set", {{"key", "a"}, {"value", 1}})).ok);
    REQUIRE(r.invoke(inv("toy.set", {{"key", "a"}, {"value", 2}})).ok);
    REQUIRE(r.invoke(inv("toy.set", {{"key", "b"}, {"value", 3}})).ok);
    CHECK(state(r).undo.size() == 3);

    auto undone = r.invoke(inv("history.undo", {{"steps", 2}}));
    REQUIRE(undone.ok);
    CHECK(undone.result["steps"].size() == 2);
    CHECK(undone.result["steps"][0]["label"] == "toy.set");
    CHECK(s.values == std::map<std::string, int>{{"a", 1}});
    CHECK(undone.result["can_redo"] == true);

    REQUIRE(r.invoke(inv("history.redo")).ok);
    CHECK(s.values["a"] == 2);
    // A new command clears what can be redone.
    REQUIRE(r.invoke(inv("toy.set", {{"key", "c"}, {"value", 4}})).ok);
    CHECK(r.invoke(inv("history.redo")).error.code == "history.nothing_to_redo");

    // Asking for more steps than there are undoes what there is.
    const auto all = r.invoke(inv("history.undo", {{"steps", 100}}));
    REQUIRE(all.ok);
    CHECK(all.result["steps"].size() == 3);
    CHECK(s.values.empty());
    CHECK(r.invoke(inv("history.undo")).error.code == "history.nothing_to_undo");
    CHECK(r.invoke(inv("history.undo", {{"steps", 0}})).error.code == "command.invalid_args");

    const auto list = r.invoke(inv("history.list"));
    REQUIRE(list.ok);
    CHECK(list.result["undo"].empty());
    CHECK(list.result["redo"].size() == 3);
}

TEST_CASE("groups undo as one step, and merge keys merge a drag") {
    RegistryOptions options;
    auto now = std::chrono::steady_clock::time_point{};
    options.steady_now = [&now] { return now; };
    CommandRegistry r(options);
    Store s;
    add_toy(r, s);
    for (int v = 1; v <= 3; ++v)
        REQUIRE(r.invoke(inv("toy.set", {{"key", "g" + std::to_string(v)}, {"value", v}}, CommandOrigin::agent,
                             "turn-1"))
                    .ok);
    CHECK(state(r).undo.size() == 1);
    CHECK(state(r).undo[0].count == 3);
    REQUIRE(r.invoke(inv("history.undo")).ok);
    CHECK(s.values.empty());
    REQUIRE(r.invoke(inv("history.redo")).ok);
    CHECK(s.values.size() == 3);

    for (int v = 10; v <= 50; v += 10) {
        now += std::chrono::milliseconds(16);
        REQUIRE(r.invoke(inv("toy.set", {{"key", "ratio"}, {"value", v}}, CommandOrigin::builder, {}, "slider:ratio")).ok);
    }
    CHECK(state(r).undo.size() == 2);
    CHECK(state(r).undo[0].count == 5);
    REQUIRE(r.invoke(inv("history.undo")).ok);
    CHECK(s.values.count("ratio") == 0);
    REQUIRE(r.invoke(inv("history.redo")).ok);
    CHECK(s.values["ratio"] == 50);
}

TEST_CASE("undo by an inverse invocation, and redo by running the command again") {
    CommandRegistry r;
    std::map<std::string, int> panels;
    int creates = 0;
    REQUIRE(r.add(descriptor("toy.add", true), [&](const CommandContext& c) {
                 const std::string key = c.invocation.args.value("key", "");
                 if (panels.count(key)) return CommandOutput::fail(cb::Error("toy.exists", "It exists."));
                 panels[key] = c.invocation.args.value("value", 0);
                 ++creates;
                 return CommandOutput::ok({{"key", key}}, UndoRecord::inverse("toy.remove", {{"key", key}}));
             }).ok());
    REQUIRE(r.add(descriptor("toy.remove", true), [&](const CommandContext& c) {
                 const std::string key = c.invocation.args.value("key", "");
                 if (!panels.count(key)) return CommandOutput::fail(cb::Error("toy.not_found", "No such key."));
                 const int value = panels[key];
                 panels.erase(key);
                 return CommandOutput::ok({}, UndoRecord::inverse("toy.add", {{"key", key}, {"value", value}}));
             }).ok());
    REQUIRE(r.invoke(inv("toy.add", {{"key", "heat"}, {"value", 7}})).ok);
    REQUIRE(r.invoke(inv("history.undo")).ok);
    CHECK(panels.empty());
    REQUIRE(r.invoke(inv("history.redo")).ok);
    CHECK(panels["heat"] == 7);
    CHECK(creates == 2);
    REQUIRE(r.invoke(inv("history.undo")).ok);
    CHECK(panels.empty());

    // The workspace changed underneath: the inverse fails, and the history says so.
    REQUIRE(r.invoke(inv("history.redo")).ok);
    panels.erase("heat");  // changed outside the command layer
    const auto failed = r.invoke(inv("history.undo"));
    CHECK(failed.error.code == "history.undo_failed");
    CHECK(failed.error.where == "toy.add");
    CHECK(failed.error.message ==
          "The undo of toy.add failed: No such key. The step and every older step were dropped from the history.");
    CHECK_FALSE(state(r).can_undo);
}

TEST_CASE("a failed undo rolls the step back and drops the older steps") {
    CommandRegistry r;
    Store s;
    add_toy(r, s);
    REQUIRE(r.invoke(inv("toy.set", {{"key", "old"}, {"value", 100}})).ok);
    for (int v = 1; v <= 3; ++v)
        REQUIRE(r.invoke(inv("toy.set", {{"key", "k" + std::to_string(v)}, {"value", v}}, CommandOrigin::ui, "g")).ok);
    REQUIRE(r.invoke(inv("toy.set", {{"key", "new"}, {"value", 200}})).ok);
    REQUIRE(r.invoke(inv("history.undo")).ok);  // undo "new", so there is a redo step
    s.fail_undo_of = 2;
    const auto before = s.values;
    const auto failed = r.invoke(inv("history.undo"));
    CHECK(failed.error.code == "history.undo_failed");
    // k3 was undone and then redone: the state is as it was.
    CHECK(s.values == before);
    const auto st = state(r);
    CHECK_FALSE(st.can_undo);  // the group step and the older "old" step were dropped
    CHECK(st.can_redo);        // the redo stack is kept
    REQUIRE(r.invoke(inv("history.redo")).ok);
    CHECK(s.values["new"] == 200);
}

TEST_CASE("a failed redo rolls back, and a failed rollback leaves the history cleared") {
    {
        CommandRegistry r;
        Store s;
        add_toy(r, s);
        for (int v = 1; v <= 3; ++v)
            REQUIRE(r.invoke(inv("toy.set", {{"key", "k" + std::to_string(v)}, {"value", v}}, CommandOrigin::ui, "g")).ok);
        REQUIRE(r.invoke(inv("toy.set", {{"key", "z"}, {"value", 9}})).ok);
        REQUIRE(r.invoke(inv("history.undo", {{"steps", 2}})).ok);
        CHECK(s.values.empty());
        s.fail_redo_of = 3;
        const auto failed = r.invoke(inv("history.redo"));
        CHECK(failed.error.code == "history.redo_failed");
        CHECK(s.values.empty());  // k1 and k2 were redone, then undone again
        CHECK_FALSE(state(r).can_redo);
        CHECK_FALSE(state(r).can_undo);
    }
    {
        CommandRegistry r;
        Store s;
        add_toy(r, s);
        for (int v = 1; v <= 3; ++v)
            REQUIRE(r.invoke(inv("toy.set", {{"key", "k" + std::to_string(v)}, {"value", v}}, CommandOrigin::ui, "g")).ok);
        s.fail_undo_of = 2;
        s.fail_redo_of = 3;  // the rollback of k3 fails too
        const auto failed = r.invoke(inv("history.undo"));
        CHECK(failed.error.code == "history.inconsistent");
        CHECK_FALSE(state(r).can_undo);
        CHECK_FALSE(state(r).can_redo);
    }
}

TEST_CASE("a command that cannot be undone is passed by and clears the redo stack") {
    CommandRegistry r;
    Store s;
    add_toy(r, s);
    int saves = 0;
    REQUIRE(r.add(descriptor("toy.save", true), [&](const CommandContext&) {
                 ++saves;
                 return CommandOutput::ok({});
             }).ok());
    REQUIRE(r.invoke(inv("toy.set", {{"key", "a"}, {"value", 1}})).ok);
    REQUIRE(r.invoke(inv("toy.set", {{"key", "a"}, {"value", 2}})).ok);
    REQUIRE(r.invoke(inv("history.undo")).ok);
    REQUIRE(r.invoke(inv("toy.save")).ok);
    CHECK_FALSE(state(r).can_redo);
    CHECK(state(r).undo.size() == 1);
    REQUIRE(r.invoke(inv("history.undo")).ok);  // passes the save by
    CHECK(s.values.empty());
    CHECK(saves == 1);
}

TEST_CASE("an outcome with no effect is a read: the redo stack, the history and the journal stay as they were") {
    const fs::path dir = fs::path(CALIPER_BUILDER_TEST_TMP) / "registry-no-effect";
    std::error_code ec;
    fs::remove_all(dir, ec);
    const std::string path = (dir / "commands.jsonl").string();
    auto opened = CommandJournal::open(path);
    REQUIRE(opened.error.ok());
    CommandRegistry r;
    r.set_journal(opened.journal.get());
    Store s;
    add_toy(r, s);
    int undo_runs = 0;
    // toy.preview: a dry run of toy.set. Its undo record, if any, is ignored.
    REQUIRE(r.add(descriptor("toy.preview", true), [&](const CommandContext& c) {
                 if (c.invocation.args.value("key", "") == "bad") {
                     CommandOutput failed = CommandOutput::fail(cb::Error("toy.refused", "No."));
                     failed.no_effect = true;
                     return failed;
                 }
                 CommandOutput out = CommandOutput::unchanged({{"preview", true}});
                 out.undo = UndoRecord::closures([&] { ++undo_runs; return cb::Error(); }, [] { return cb::Error(); });
                 return out;
             }).ok());

    REQUIRE(r.invoke(inv("toy.set", {{"key", "a"}, {"value", 1}})).ok);
    REQUIRE(r.invoke(inv("toy.set", {{"key", "a"}, {"value", 2}})).ok);
    REQUIRE(r.invoke(inv("history.undo")).ok);
    CHECK(r.invoke(inv("toy.preview", {{"key", "a"}})).ok);
    CHECK(r.invoke(inv("toy.preview", {{"key", "bad"}})).error.code == "toy.refused");
    CHECK(state(r).can_redo);
    CHECK(state(r).undo.size() == 1);
    REQUIRE(r.invoke(inv("history.redo")).ok);
    CHECK(s.values["a"] == 2);
    CHECK(undo_runs == 0);
    // The previews wrote nothing: two do events, one undo and one redo.
    const auto read = cb::read_journal(path);
    REQUIRE(read.events.size() == 4);
    for (const auto& e : read.events) CHECK(e.name != "toy.preview");

    // Merging stays open across a preview: a drag interrupted by one is still one step.
    REQUIRE(r.invoke(inv("toy.set", {{"key", "b"}, {"value", 1}}, CommandOrigin::ui, {}, "drag")).ok);
    REQUIRE(r.invoke(inv("toy.preview", {{"key", "b"}})).ok);
    REQUIRE(r.invoke(inv("toy.set", {{"key", "b"}, {"value", 2}}, CommandOrigin::ui, {}, "drag")).ok);
    CHECK(state(r).undo.front().count == 2);

    // A journal that records reads records the preview as a read.
    const fs::path reads = dir / "reads" / "commands.jsonl";
    cb::JournalOptions options;
    options.include_read_only = true;
    auto with_reads = CommandJournal::open(reads.string(), options);
    REQUIRE(with_reads.error.ok());
    r.set_journal(with_reads.journal.get());
    REQUIRE(r.invoke(inv("toy.preview", {{"key", "a"}})).ok);
    const auto read2 = cb::read_journal(reads.string());
    REQUIRE(read2.events.size() == 1);
    CHECK(read2.events[0].name == "toy.preview");
}

TEST_CASE("the default policy asks for a destructive command from an agent, and the person decides") {
    RegistryOptions options;
    options.max_pending = 2;
    CommandRegistry r(options);
    int deletes = 0;
    REQUIRE(r.add(descriptor("toy.delete", true, true), [&](const CommandContext&) {
                 ++deletes;
                 return CommandOutput::ok({{"deleted", true}});
             }).ok());

    CHECK(r.invoke(inv("toy.delete", {}, CommandOrigin::ui)).ok);
    CHECK(r.invoke(inv("toy.delete", {}, CommandOrigin::builder)).ok);
    CHECK(deletes == 2);

    const auto asked = r.invoke(inv("toy.delete", {{"key", "x"}}, CommandOrigin::mcp));
    CHECK(asked.error.code == "command.needs_approval");
    CHECK(asked.approval == "ap-1");
    CHECK(cb::command_result_to_json(asked)["error"]["approval"] == "ap-1");
    CHECK(deletes == 2);
    const auto pending = r.pending_approvals();
    REQUIRE(pending.size() == 1);
    CHECK(pending[0].invocation.args["key"] == "x");
    CHECK(pending[0].summary == "A test command.");
    CHECK(cb::pending_approval_to_json(pending[0])["invocation"]["origin"] == "mcp");

    const auto approved = r.approve("ap-1");
    CHECK(approved.ok);
    CHECK(approved.id == asked.id);
    CHECK(deletes == 3);
    CHECK(r.approve("ap-1").error.code == "command.unknown_approval");

    const auto asked2 = r.invoke(inv("toy.delete", {}, CommandOrigin::agent));
    const auto denied = r.deny(asked2.approval);
    CHECK(denied.error.code == "command.denied");
    CHECK(deletes == 3);

    CHECK(r.invoke(inv("toy.delete", {}, CommandOrigin::api)).error.code == "command.needs_approval");
    CHECK(r.invoke(inv("toy.delete", {}, CommandOrigin::api)).error.code == "command.needs_approval");
    CHECK(r.invoke(inv("toy.delete", {}, CommandOrigin::api)).error.code == "command.too_many_pending");
    CHECK(r.cancel_pending() == 2);
    CHECK(r.pending_approvals().empty());
}

TEST_CASE("a custom policy can deny, and a throwing policy denies") {
    CommandRegistry r;
    REQUIRE(r.add(descriptor("toy.get", false), [](const CommandContext&) { return CommandOutput::ok({}); }).ok());
    r.set_approval_policy([](const CommandDescriptor& d, const CommandInvocation& i) {
        return d.name == "toy.get" && i.origin == CommandOrigin::api ? ApprovalDecision::deny : ApprovalDecision::allow;
    });
    CHECK(r.invoke(inv("toy.get", {}, CommandOrigin::api)).error.code == "command.denied");
    CHECK(r.invoke(inv("toy.get", {}, CommandOrigin::mcp)).ok);
    r.set_approval_policy([](const CommandDescriptor&, const CommandInvocation&) -> ApprovalDecision {
        throw std::runtime_error("policy bug");
    });
    CHECK(r.invoke(inv("toy.get")).error.code == "command.denied");
    r.set_approval_policy({});
    CHECK(r.invoke(inv("toy.get")).ok);
}

TEST_CASE("a program cannot redo or undo a destructive step without approval") {
    CommandRegistry r;
    Store s;
    add_toy(r, s, /*destructive=*/true);
    REQUIRE(r.invoke(inv("toy.set", {{"key", "a"}, {"value", 1}})).ok);
    // The person may undo it; a program over Caliper API must ask (an agent
    // may not undo at all).
    const auto asked = r.invoke(inv("history.undo", {}, CommandOrigin::api));
    REQUIRE(asked.error.code == "command.needs_approval");
    CHECK(s.values["a"] == 1);
    REQUIRE(r.approve(asked.approval).ok);
    CHECK(s.values.empty());

    const auto redo = r.invoke(inv("history.redo", {}, CommandOrigin::api));
    REQUIRE(redo.error.code == "command.needs_approval");
    // The history changes before the approval: the approved redo is refused.
    REQUIRE(r.invoke(inv("toy.set", {{"key", "b"}, {"value", 2}})).ok);
    CHECK(r.approve(redo.approval).error.code == "history.changed");
    REQUIRE(r.invoke(inv("history.undo", {}, CommandOrigin::ui)).ok);
    CHECK(s.values.empty());
}

TEST_CASE("mutating commands, failures, denials and history moves are journaled") {
    const fs::path dir = fs::path(CALIPER_BUILDER_TEST_TMP) / "registry-journal";
    std::error_code ec;
    fs::remove_all(dir, ec);
    const std::string path = (dir / "commands.jsonl").string();
    auto opened = CommandJournal::open(path);
    REQUIRE(opened.error.ok());

    CommandRegistry r;
    r.set_journal(opened.journal.get());
    Store s;
    add_toy(r, s);
    REQUIRE(r.add(descriptor("toy.get", false), [](const CommandContext&) { return CommandOutput::ok({}); }).ok());
    REQUIRE(r.add(descriptor("toy.fail", true), [](const CommandContext&) {
                 return CommandOutput::fail(cb::Error("toy.broken", "Broken."));
             }).ok());
    REQUIRE(r.add(descriptor("toy.wipe", true, true), [](const CommandContext&) { return CommandOutput::ok({}); }).ok());

    REQUIRE(r.invoke(inv("toy.set", {{"key", "a"}, {"value", 1}}, CommandOrigin::ui, "g1")).ok);  // 1
    REQUIRE(r.invoke(inv("toy.get")).ok);                                                        // not journaled
    CHECK(r.invoke(inv("toy.set", {{"value", "x"}})).error.code == "command.invalid_args");      // not journaled
    CHECK(r.invoke(inv("toy.fail")).error.code == "toy.broken");                                 // 2
    const auto asked = r.invoke(inv("toy.wipe", {}, CommandOrigin::mcp));                         // not yet
    REQUIRE(r.deny(asked.approval).error.code == "command.denied");                              // 3
    REQUIRE(r.invoke(inv("history.undo")).ok);                                                   // 4
    REQUIRE(r.invoke(inv("history.redo")).ok);                                                   // 5
    CHECK(r.journal_error().ok());

    const auto read = cb::read_journal(path);
    REQUIRE(read.problems.empty());
    REQUIRE(read.events.size() == 5);
    CHECK(read.events[0].name == "toy.set");
    CHECK(read.events[0].group == "g1");
    CHECK(read.events[1].error == "toy.broken");
    CHECK(read.events[2].name == "toy.wipe");
    CHECK(read.events[2].error == "command.denied");
    CHECK(read.events[2].origin == CommandOrigin::mcp);
    CHECK(read.events[3].event == "undo");
    CHECK(read.events[3].seqs == std::vector<std::uint64_t>{1});
    CHECK(read.events[3].group == "g1");
    CHECK(read.events[4].event == "redo");

    // A journal that cannot be written does not fail the command.
    opened.journal->close();
    CHECK(r.invoke(inv("toy.set", {{"key", "b"}, {"value", 2}})).ok);
    CHECK(r.journal_error().code == "journal.closed");
}

TEST_CASE("read-only commands are journaled only when the journal asks for it") {
    const fs::path dir = fs::path(CALIPER_BUILDER_TEST_TMP) / "registry-reads";
    std::error_code ec;
    fs::remove_all(dir, ec);
    cb::JournalOptions options;
    options.include_read_only = true;
    auto opened = CommandJournal::open((dir / "commands.jsonl").string(), options);
    REQUIRE(opened.error.ok());
    CommandRegistry r;
    r.set_journal(opened.journal.get());
    REQUIRE(r.add(descriptor("toy.get", false), [](const CommandContext&) { return CommandOutput::ok({}); }).ok());
    REQUIRE(r.invoke(inv("toy.get", {{"key", "q"}})).ok);
    const auto read = cb::read_journal((dir / "commands.jsonl").string());
    REQUIRE(read.events.size() == 1);
    CHECK(read.events[0].name == "toy.get");
}

TEST_CASE("a handler's follow-ups run as one new step, journaled after it, and roll back together when one fails") {
    const fs::path dir = fs::path(CALIPER_BUILDER_TEST_TMP) / "registry-follow-ups";
    std::error_code ec;
    fs::remove_all(dir, ec);
    const std::string path = (dir / "commands.jsonl").string();
    auto opened = CommandJournal::open(path);
    REQUIRE(opened.error.ok());
    CommandRegistry r;
    r.set_journal(opened.journal.get());
    Store s;
    add_toy(r, s);
    REQUIRE(r.add(descriptor("toy.wipe", true, true)).ok());  // destructive, unbound
    REQUIRE(r.add(descriptor("toy.look", false), [](const CommandContext&) { return CommandOutput::ok({}); }).ok());
    // toy.batch {value}: sets a, b and c to value, value + 1 and value + 2 as
    // follow-ups; with key "mixed" its follow-ups do not share a group, with
    // "wipe" one is destructive, with "look" one does not mutate.
    REQUIRE(r.add(descriptor("toy.batch", true), [](const CommandContext& c) {
                 const int v = c.invocation.args.value("value", 0);
                 const std::string key = c.invocation.args.value("key", "");
                 CommandOutput out = CommandOutput::ok({{"batched", 3}});
                 for (int k = 0; k < 3; ++k) {
                     CommandInvocation sub = inv("toy.set", {{"key", std::string(1, char('a' + k))}, {"value", v + k}},
                                                 CommandOrigin::agent, "batch");
                     if (key == "mixed" && k == 2) sub.group = "other";
                     if (key == "wipe" && k == 1) sub.name = "toy.wipe";
                     if (key == "look" && k == 1) sub.name = "toy.look";
                     out.follow_ups.push_back(sub);
                 }
                 return out;
             }).ok());

    // An earlier step of the same group stays a step of its own.
    REQUIRE(r.invoke(inv("toy.set", {{"key", "z"}, {"value", 9}}, CommandOrigin::agent, "batch")).ok);
    const CommandResult done = r.invoke(inv("toy.batch", {{"value", 10}}));
    REQUIRE(done.ok);
    CHECK(done.result == json{{"batched", 3}});
    CHECK(s.values["a"] == 10);
    CHECK(s.values["c"] == 12);
    REQUIRE(state(r).undo.size() == 2);
    CHECK(state(r).undo[0].count == 3);
    CHECK(state(r).undo[0].group == "batch");
    CHECK(state(r).undo[0].origin == CommandOrigin::agent);
    REQUIRE(r.invoke(inv("history.undo")).ok);
    CHECK(s.values.count("a") == 0);
    CHECK(s.values.count("c") == 0);
    CHECK(s.values["z"] == 9);
    REQUIRE(r.invoke(inv("history.redo")).ok);
    CHECK(s.values["b"] == 11);

    // A follow-up that fails undoes the ones that ran, and only the failed
    // invocation is journaled.
    REQUIRE(r.add(descriptor("toy.fail", true), [](const CommandContext&) {
                 return CommandOutput::fail(cb::Error("toy.nope", "It fails."));
             }).ok());
    REQUIRE(r.add(descriptor("toy.half", true), [](const CommandContext&) {
                 CommandOutput out = CommandOutput::ok({});
                 out.follow_ups.push_back(inv("toy.set", {{"key", "h"}, {"value", 1}}, CommandOrigin::agent, "g"));
                 out.follow_ups.push_back(inv("toy.fail", {}, CommandOrigin::agent, "g"));
                 return out;
             }).ok());
    const CommandResult half = r.invoke(inv("toy.half"));
    CHECK(half.error.code == "toy.nope");
    CHECK(s.values.count("h") == 0);
    CHECK(state(r).undo.size() == 2);

    // Follow-ups that cannot run are refused before any runs.
    for (const char* key : {"mixed", "wipe", "look"}) {
        CAPTURE(key);
        const CommandResult bad = r.invoke(inv("toy.batch", {{"key", key}, {"value", 50}}));
        CHECK(bad.error.code == "command.handler_failed");
        CHECK(s.values["a"] == 10);
    }

    const auto read = cb::read_journal(path);
    REQUIRE(read.error.ok());
    // z, the batch and its three sets, an undo and a redo, the failed half and
    // the three refused batches.
    REQUIRE(read.events.size() == 11);
    CHECK(read.events[1].name == "toy.batch");
    CHECK(read.events[2].name == "toy.set");
    CHECK(read.events[2].group == "batch");
    CHECK(read.events[4].args == json{{"key", "c"}, {"value", 12}});
    CHECK(read.events[5].event == "undo");
    CHECK(read.events[5].seqs == std::vector<std::uint64_t>{3, 4, 5});
    CHECK(read.events[7].name == "toy.half");
    CHECK_FALSE(read.events[7].ok);
    CHECK(read.events[7].error == "toy.nope");
}

TEST_CASE("the approval review is kept with each parked invocation and shown in its JSON") {
    CommandRegistry r;
    Store s;
    add_toy(r, s, true);
    int reviews = 0;
    r.set_approval_review([&](const CommandDescriptor& d, const CommandInvocation& i) -> json {
        ++reviews;
        if (i.args.value("value", 0) == 13) throw std::runtime_error("no review");
        return {{"command", d.name}, {"value", i.args.value("value", 0)}};
    });
    REQUIRE(r.invoke(inv("toy.set", {{"key", "a"}, {"value", 1}})).ok);  // ui: allowed, no review asked
    CHECK(reviews == 0);
    const CommandResult asked = r.invoke(inv("toy.set", {{"key", "a"}, {"value", 2}}, CommandOrigin::agent));
    REQUIRE(asked.error.code == "command.needs_approval");
    CHECK(asked.error.message.find("reviews") != std::string::npos);
    const CommandResult thrown = r.invoke(inv("toy.set", {{"key", "a"}, {"value", 13}}, CommandOrigin::mcp));
    REQUIRE(thrown.error.code == "command.needs_approval");
    CHECK(thrown.error.message.find("deletes or overwrites") != std::string::npos);
    CHECK(reviews == 2);
    const auto pending = r.pending_approvals();
    REQUIRE(pending.size() == 2);
    CHECK(pending[0].review == json{{"command", "toy.set"}, {"value", 2}});
    CHECK(pending[1].review.is_null());
    CHECK(cb::pending_approval_to_json(pending[0])["review"]["value"] == 2);
    CHECK(cb::pending_approval_to_json(pending[1])["review"].is_null());
    REQUIRE(r.approve(asked.approval).ok);
    CHECK(s.values["a"] == 2);
}

TEST_CASE("history.undo and history.redo are the person's: agent and mcp are refused before any policy or history work") {
    const fs::path dir = fs::path(CALIPER_BUILDER_TEST_TMP) / "registry-history-agents";
    std::error_code ec;
    fs::remove_all(dir, ec);
    const std::string path = (dir / "commands.jsonl").string();
    auto opened = CommandJournal::open(path);
    REQUIRE(opened.error.ok());
    CommandRegistry r;
    r.set_journal(opened.journal.get());
    Store s;
    add_toy(r, s);
    int asked = 0;
    r.set_approval_policy([&](const CommandDescriptor& d, const CommandInvocation& i) {
        ++asked;
        return cb::default_approval_policy(d, i);
    });
    REQUIRE(r.invoke(inv("toy.set", {{"key", "a"}, {"value", 1}}, CommandOrigin::agent)).ok);
    const std::size_t lines = cb::read_journal(path).events.size();
    for (const char* name : {"history.undo", "history.redo"})
        for (CommandOrigin o : {CommandOrigin::agent, CommandOrigin::mcp}) {
            CAPTURE(name);
            asked = 0;
            const CommandResult refused = r.invoke(inv(name, json::object(), o));
            CHECK(refused.error.code == "command.not_allowed");
            CHECK(refused.error.message.find("edit") != std::string::npos);
            CHECK(asked == 0);
        }
    CHECK(s.values["a"] == 1);
    CHECK(r.history_state().undo.size() == 1);
    CHECK(cb::read_journal(path).events.size() == lines);
    CHECK(r.pending_approvals().empty());
    // history.list stays readable for agents; the person and programs undo and redo.
    CHECK(r.invoke(inv("history.list", json::object(), CommandOrigin::mcp)).ok);
    REQUIRE(r.invoke(inv("history.undo", json::object(), CommandOrigin::ui)).ok);
    CHECK(s.values.count("a") == 0);
    REQUIRE(r.invoke(inv("history.redo", json::object(), CommandOrigin::api)).ok);
    CHECK(s.values["a"] == 1);
    REQUIRE(r.invoke(inv("history.undo", json::object(), CommandOrigin::builder)).ok);
    // The descriptors say so, for the clients that build tools from them.
    for (const auto& d : cb::history_command_descriptors())
        CHECK(cb::command_descriptor_to_json(d)["agents"] == (d.name == "history.list"));
}
