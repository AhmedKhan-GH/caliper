// The worked test of the command layer: a toy document with panels, three
// commands on a registry, driven through the queue from two threads, with
// undo and redo across a group, a merged slider drag, a failing handler, a
// destructive command from the mcp origin that waits for approval, and a
// journal that is closed, reopened and replayed to the same final document.
#include <caliper_builder/commands/journal.h>
#include <caliper_builder/commands/queue.h>
#include <caliper_builder/commands/registry.h>

#include <doctest/doctest.h>

#include <atomic>
#include <filesystem>
#include <future>
#include <thread>
#include <vector>

using caliper_builder::CommandContext;
using caliper_builder::CommandDescriptor;
using caliper_builder::CommandInvocation;
using caliper_builder::CommandJournal;
using caliper_builder::CommandOrigin;
using caliper_builder::CommandOutput;
using caliper_builder::CommandQueue;
using caliper_builder::CommandRegistry;
using caliper_builder::CommandResult;
using caliper_builder::Error;
using caliper_builder::JournalEvent;
using caliper_builder::RegistryOptions;
using caliper_builder::UndoRecord;
using nlohmann::json;
using namespace std::chrono_literals;
namespace cb = caliper_builder;
namespace fs = std::filesystem;

namespace {

// The toy document: {"panels": {"<id>": {"title": "...", "ratio": 0.5}}}.
// Each command is a function of the document and the arguments, used both by
// the handlers and by the replay, so the replay rebuilds exactly what the
// commands did.
Error add_panel(json& doc, const json& args) {
    const std::string id = args.at("id").get<std::string>();
    if (doc["panels"].contains(id)) return Error("doc.exists", "A panel with this id exists.", id);
    doc["panels"][id] = {{"title", args.value("title", id)}, {"ratio", args.value("ratio", 0.5)}};
    return {};
}

Error remove_panel(json& doc, const json& args) {
    const std::string id = args.at("id").get<std::string>();
    if (!doc["panels"].contains(id)) return Error("doc.not_found", "No panel has this id.", id);
    doc["panels"].erase(id);
    return {};
}

Error set_ratio(json& doc, const json& args) {
    // at() throws for a missing panel: a handler bug, which the registry must contain.
    doc["panels"].at(args.at("id").get<std::string>())["ratio"] = args.at("ratio").get<double>();
    return {};
}

Error apply(json& doc, const JournalEvent& e) {
    if (e.name == "doc.add_panel") return add_panel(doc, e.args);
    if (e.name == "doc.remove_panel") return remove_panel(doc, e.args);
    if (e.name == "doc.set_ratio") return set_ratio(doc, e.args);
    return Error("doc.unknown", "The replay does not know this command.", e.name);
}

CommandDescriptor describe(const char* name, const char* summary, bool destructive, const char* args) {
    CommandDescriptor d;
    d.name = name;
    d.summary = summary;
    d.args = json::parse(args);
    d.result = json::parse(R"({"type": "object", "additionalProperties": false, "required": [],
                               "properties": {"id": {"type": "string", "description": "The panel's id."}}})");
    d.mutates = true;
    d.destructive = destructive;
    d.frame_thread = true;  // the document belongs to the frame thread
    return d;
}

void register_commands(CommandRegistry& registry, json& doc) {
    REQUIRE(registry
                .add(describe("doc.add_panel", "Adds a panel to the document.", false,
                              R"({"type": "object", "additionalProperties": false, "required": ["id"],
                                  "properties": {"id": {"type": "string", "pattern": "^[a-z][a-z0-9_-]{0,63}$"},
                                                 "title": {"type": "string"},
                                                 "ratio": {"type": "number", "minimum": 0, "maximum": 1}}})"),
                     [&doc](const CommandContext& c) {
                         if (Error err = add_panel(doc, c.invocation.args)) return CommandOutput::fail(err);
                         const std::string id = c.invocation.args["id"];
                         return CommandOutput::ok({{"id", id}}, UndoRecord::inverse("doc.remove_panel", {{"id", id}}));
                     })
                .ok());
    REQUIRE(registry
                .add(describe("doc.remove_panel", "Removes a panel from the document.", true,
                              R"({"type": "object", "additionalProperties": false, "required": ["id"],
                                  "properties": {"id": {"type": "string"}}})"),
                     [&doc](const CommandContext& c) {
                         const std::string id = c.invocation.args["id"];
                         const json before = doc["panels"].value(id, json());
                         if (Error err = remove_panel(doc, c.invocation.args)) return CommandOutput::fail(err);
                         json restore = before;
                         restore["id"] = id;
                         return CommandOutput::ok({{"id", id}}, UndoRecord::inverse("doc.add_panel", restore));
                     })
                .ok());
    REQUIRE(registry
                .add(describe("doc.set_ratio", "Sets a panel's ratio.", false,
                              R"({"type": "object", "additionalProperties": false, "required": ["id", "ratio"],
                                  "properties": {"id": {"type": "string"},
                                                 "ratio": {"type": "number", "minimum": 0, "maximum": 1}}})"),
                     [&doc](const CommandContext& c) {
                         const std::string id = c.invocation.args["id"];
                         const double before = doc["panels"].contains(id) ? doc["panels"][id].value("ratio", 0.5) : 0.0;
                         set_ratio(doc, c.invocation.args);
                         const double after = c.invocation.args["ratio"];
                         // A closure pair. The command is meant to merge (a slider), so its
                         // undo checks only what a whole drag leaves unchanged: that the
                         // panel still exists. A merged record keeps the first undo.
                         return CommandOutput::ok(
                             {{"id", id}},
                             UndoRecord::closures(
                                 [&doc, id, before] {
                                     if (!doc["panels"].contains(id))
                                         return Error("history.conflict", "The panel is gone.", id);
                                     doc["panels"][id]["ratio"] = before;
                                     return Error();
                                 },
                                 [&doc, id, after] {
                                     if (!doc["panels"].contains(id)) return Error("history.conflict", "The panel is gone.", id);
                                     doc["panels"][id]["ratio"] = after;
                                     return Error();
                                 }));
                     })
                .ok());
}

CommandInvocation inv(const std::string& name, json args, CommandOrigin origin, const std::string& group = {},
                      const std::string& merge_key = {}) {
    CommandInvocation i;
    i.name = name;
    i.args = std::move(args);
    i.origin = origin;
    i.group = group;
    i.merge_key = merge_key;
    return i;
}

// Drains on this thread until every future is ready.
void drain_until_ready(CommandQueue& queue, const std::vector<std::future<CommandResult>*>& futures) {
    for (int spin = 0; spin < 200000; ++spin) {
        queue.drain();
        bool all = true;
        for (const auto* f : futures) all = all && f->wait_for(0s) == std::future_status::ready;
        if (all) return;
        std::this_thread::yield();
    }
    FAIL("the futures did not complete");
}

CommandResult run(CommandQueue& queue, CommandInvocation i) {
    auto f = queue.submit(std::move(i));
    drain_until_ready(queue, {&f});
    return f.get();
}

}  // namespace

TEST_CASE("a worked session: two threads, a group, a drag, a failure, an approval, undo, redo and replay") {
    const fs::path dir = fs::path(CALIPER_BUILDER_TEST_TMP) / "worked";
    std::error_code ec;
    fs::remove_all(dir, ec);
    const std::string journal_path = (dir / "journal" / "commands.jsonl").string();

    json doc = {{"panels", json::object()}};
    RegistryOptions options;
    std::atomic<long long> ticks{0};
    options.steady_now = [&ticks] {  // 16 ms between commands, like frames
        return std::chrono::steady_clock::time_point(std::chrono::milliseconds(ticks += 16));
    };
    options.check_results = true;
    CommandRegistry registry(options);
    register_commands(registry, doc);
    auto journal = CommandJournal::open(journal_path);
    REQUIRE(journal.error.ok());
    registry.set_journal(journal.journal.get());
    registry.set_frame_thread(std::this_thread::get_id());  // this thread is the frame thread
    CommandQueue queue(registry);

    // Two threads submit; this thread drains. The agent's thread waits for the
    // UI thread's batch so that the group is not interleaved, which keeps the
    // expected history exact.
    std::promise<void> ui_done;
    std::shared_future<void> ui_done_f = ui_done.get_future().share();
    std::vector<std::future<CommandResult>> ui_futures;
    std::vector<std::future<CommandResult>> agent_futures;
    std::thread ui([&] {
        ui_futures.push_back(queue.submit(inv("doc.add_panel", {{"id", "heat"}, {"title", "Entropy"}}, CommandOrigin::ui)));
        ui_futures.push_back(queue.submit(inv("doc.add_panel", {{"id", "loss"}, {"title", "Loss"}}, CommandOrigin::ui)));
        // A slider drag: eleven commands with one merge key.
        for (int k = 0; k <= 10; ++k)
            ui_futures.push_back(queue.submit(inv("doc.set_ratio", {{"id", "heat"}, {"ratio", 0.30 + 0.03 * k}},
                                                  CommandOrigin::builder, {}, "slider:heat/ratio")));
        ui_done.set_value();
    });
    std::thread agent([&] {
        ui_done_f.wait();
        // The agent's proposal: three edits in one group.
        agent_futures.push_back(queue.submit(inv("doc.add_panel", {{"id", "mean"}}, CommandOrigin::agent, "turn-1")));
        agent_futures.push_back(
            queue.submit(inv("doc.set_ratio", {{"id", "mean"}, {"ratio", 0.25}}, CommandOrigin::agent, "turn-1")));
        agent_futures.push_back(queue.submit(inv("doc.add_panel", {{"id", "hist"}}, CommandOrigin::agent, "turn-1")));
        // A handler that throws: the panel does not exist.
        agent_futures.push_back(queue.submit(inv("doc.set_ratio", {{"id", "ghost"}, {"ratio", 0.5}}, CommandOrigin::agent)));
        // A destructive command from an outside agent waits for approval.
        agent_futures.push_back(queue.submit(inv("doc.remove_panel", {{"id", "loss"}}, CommandOrigin::mcp)));
    });

    // Drain until everything but the approval has completed.
    for (int spin = 0; spin < 200000; ++spin) {
        queue.drain();
        bool done = ui_futures.size() == 13 && agent_futures.size() == 5;
        if (done) {
            for (auto& f : ui_futures) done = done && f.wait_for(0s) == std::future_status::ready;
            for (std::size_t i = 0; i + 1 < agent_futures.size(); ++i)
                done = done && agent_futures[i].wait_for(0s) == std::future_status::ready;
        }
        if (done && registry.pending_approvals().size() == 1) break;
        std::this_thread::yield();
    }
    ui.join();
    agent.join();
    for (auto& f : ui_futures) CHECK(f.get().ok);
    for (std::size_t i = 0; i < 3; ++i) CHECK(agent_futures[i].get().ok);
    const CommandResult failed = agent_futures[3].get();
    CHECK(failed.error.code == "command.handler_failed");
    CHECK(failed.error.message.find("doc.set_ratio") != std::string::npos);

    // The removal waits: the document still has "loss" until the person decides.
    std::future<CommandResult>& removal = agent_futures[4];
    CHECK(removal.wait_for(0s) == std::future_status::timeout);
    CHECK(doc["panels"].contains("loss"));
    const auto pending = registry.pending_approvals();
    REQUIRE(pending.size() == 1);
    CHECK(pending[0].invocation.origin == CommandOrigin::mcp);
    CHECK(pending[0].invocation.name == "doc.remove_panel");

    CHECK(doc["panels"]["heat"]["ratio"] == doctest::Approx(0.60));
    CHECK(doc["panels"]["mean"]["ratio"] == 0.25);
    CHECK(doc["panels"].size() == 4);

    // The person approves in the UI, from another thread.
    std::future<CommandResult> approver;
    std::thread person([&] { approver = queue.approve(pending[0].approval); });
    person.join();
    drain_until_ready(queue, {&removal, &approver});
    CHECK(removal.get().ok);
    CHECK(approver.get().ok);
    CHECK_FALSE(doc["panels"].contains("loss"));

    // The history: add heat, add loss, the drag (11 merged), the group (3), the removal.
    auto history = registry.history_state();
    REQUIRE(history.undo.size() == 5);
    CHECK(history.undo[0].label == "doc.remove_panel");
    CHECK(history.undo[1].label == "doc.add_panel");
    CHECK(history.undo[1].group == "turn-1");
    CHECK(history.undo[1].count == 3);
    CHECK(history.undo[2].label == "doc.set_ratio");
    CHECK(history.undo[2].count == 11);

    // Close the journal and reopen it in the middle of the session: seq continues.
    const std::uint64_t seq_before = journal.journal->last_seq();
    journal.journal->close();
    auto reopened = CommandJournal::open(journal_path);
    REQUIRE(reopened.error.ok());
    CHECK(reopened.journal->last_seq() == seq_before);
    registry.set_journal(reopened.journal.get());

    // Undo the removal: "loss" comes back through the inverse invocation.
    CHECK(run(queue, inv("history.undo", json::object(), CommandOrigin::ui)).ok);
    CHECK(doc["panels"]["loss"]["title"] == "Loss");
    // Undo the agent's group as one step.
    const CommandResult group_undo = run(queue, inv("history.undo", json::object(), CommandOrigin::ui));
    REQUIRE(group_undo.ok);
    CHECK(group_undo.result["steps"][0]["group"] == "turn-1");
    CHECK_FALSE(doc["panels"].contains("mean"));
    CHECK_FALSE(doc["panels"].contains("hist"));
    // Redo it as one step.
    CHECK(run(queue, inv("history.redo", json::object(), CommandOrigin::ui)).ok);
    CHECK(doc["panels"]["mean"]["ratio"] == 0.25);
    CHECK(doc["panels"].contains("hist"));
    // Undo two steps: the group, then the whole drag.
    CHECK(run(queue, inv("history.undo", {{"steps", 2}}, CommandOrigin::ui)).ok);
    CHECK_FALSE(doc["panels"].contains("mean"));
    CHECK(doc["panels"]["heat"]["ratio"] == 0.5);
    // Redo the drag.
    CHECK(run(queue, inv("history.redo", json::object(), CommandOrigin::ui)).ok);
    CHECK(doc["panels"]["heat"]["ratio"] == doctest::Approx(0.60));
    // A new command clears what could still be redone (the group).
    CHECK(run(queue, inv("doc.set_ratio", {{"id", "loss"}, {"ratio", 0.7}}, CommandOrigin::ui)).ok);
    CHECK(run(queue, inv("history.redo", json::object(), CommandOrigin::ui)).error.code == "history.nothing_to_redo");
    CHECK(registry.journal_error().ok());

    const json final_doc = doc;
    CHECK(final_doc == json{{"panels",
                             {{"heat", {{"title", "Entropy"}, {"ratio", final_doc["panels"]["heat"]["ratio"]}}},
                              {"loss", {{"title", "Loss"}, {"ratio", 0.7}}}}}});

    // Close the journal, read it back and replay it onto an empty document.
    reopened.journal->close();
    queue.shutdown();
    const auto read = cb::read_journal(journal_path);
    REQUIRE(read.error.ok());
    CHECK(read.problems.empty());
    for (std::size_t i = 0; i < read.events.size(); ++i) CHECK(read.events[i].seq == i + 1);
    // 2 adds, 11 drags, 3 group edits, 1 failure, 1 approved removal, then
    // 6 history events and the last set_ratio.
    CHECK(read.events.size() == 25);
    CHECK(read.events[16].name == "doc.set_ratio");
    CHECK(read.events[16].error == "command.handler_failed");
    CHECK(read.events[17].name == "doc.remove_panel");
    CHECK(read.events[17].origin == CommandOrigin::mcp);

    json replayed = {{"panels", json::object()}};
    const auto replay = cb::replay_journal(read.events, [&replayed](const JournalEvent& e) { return apply(replayed, e); });
    REQUIRE(replay.error.ok());
    CHECK(replayed == final_doc);
}
