// Tests of CommandHistory alone: steps, groups, merging and its window, the
// redo rule and the bounds on depth and memory. The registry's tests run the
// records; these check only how they are filed.
#include <caliper_builder/commands/history.h>

#include <doctest/doctest.h>

#include <string>

using caliper_builder::CommandHistory;
using caliper_builder::CommandOrigin;
using caliper_builder::HistoryLimits;
using caliper_builder::HistoryRecord;
using caliper_builder::UndoRecord;
using nlohmann::json;
using Clock = std::chrono::steady_clock;
using std::chrono::milliseconds;
namespace cb = caliper_builder;

namespace {

HistoryRecord rec(const std::string& name, Clock::time_point at, const std::string& group = {},
                  const std::string& merge_key = {}, CommandOrigin origin = CommandOrigin::ui,
                  std::uint64_t seq = 0) {
    HistoryRecord r;
    r.invocation.name = name;
    r.invocation.group = group;
    r.invocation.merge_key = merge_key;
    r.invocation.origin = origin;
    r.undo = UndoRecord::closures([] { return cb::Error(); }, [] { return cb::Error(); });
    r.last = at;
    if (seq) r.seqs.push_back(seq);
    return r;
}

// A closure pair whose undo and redo report the values they would restore.
HistoryRecord set_record(int before, int after, std::vector<int>& log, Clock::time_point at,
                         const std::string& key) {
    HistoryRecord r = rec("toy.set", at, {}, key);
    r.undo = UndoRecord::closures(
        [&log, before] {
            log.push_back(before);
            return cb::Error();
        },
        [&log, after] {
            log.push_back(after);
            return cb::Error();
        });
    return r;
}

}  // namespace

TEST_CASE("each command without a group is a step, and a group joins the top step") {
    CommandHistory h;
    const auto t = Clock::now();
    h.record(rec("panel.create", t), "t1");
    h.record(rec("panel.patch", t, "turn-1"), "t2");
    h.record(rec("panel.patch", t, "turn-1"), "t3");
    h.record(rec("layout.place", t, "turn-1"), "t4");
    REQUIRE(h.undo_stack().size() == 2);
    CHECK(h.undo_stack()[1].records.size() == 3);
    CHECK(h.undo_stack()[1].group == "turn-1");
    CHECK(h.undo_stack()[1].at == "t4");

    // Another command between two commands of a group begins a second step.
    h.record(rec("spec.set_text", t), "t5");
    h.record(rec("panel.remove", t, "turn-1"), "t6");
    CHECK(h.undo_stack().size() == 4);
    const auto list = h.undo_summaries(10);
    REQUIRE(list.size() == 4);
    CHECK(list[0].label == "panel.remove");
    CHECK(list[2].label == "panel.patch");
    CHECK(list[2].count == 3);
    CHECK(list[2].group == "turn-1");
    CHECK(h.undo_summaries(1).size() == 1);
}

TEST_CASE("commands with one merge key merge within the window into one record") {
    CommandHistory h;  // a 1000 ms window
    std::vector<int> log;
    const auto t = Clock::now();
    h.record(set_record(0, 1, log, t, "slider"), "a");
    h.record(set_record(1, 2, log, t + milliseconds(16), "slider"), "b");
    h.record(set_record(2, 3, log, t + milliseconds(900), "slider"), "c");
    h.record(set_record(3, 4, log, t + milliseconds(1800), "slider"), "d");  // 900 ms after the last: still merges
    REQUIRE(h.undo_stack().size() == 1);
    const auto& step = h.undo_stack()[0];
    REQUIRE(step.records.size() == 1);
    CHECK(step.records[0].merged == 4);
    CHECK(step.count() == 4);
    // The record keeps the first undo and the newest redo.
    CHECK(step.records[0].undo.undo().ok());
    CHECK(step.records[0].undo.redo().ok());
    CHECK(log == std::vector<int>{0, 4});

    // A pause longer than the window begins a new step.
    h.record(set_record(4, 5, log, t + milliseconds(2801), "slider"), "e");
    CHECK(h.undo_stack().size() == 2);
}

TEST_CASE("merging needs the same key, name, origin and group, and an open step") {
    const auto t = Clock::now();
    {
        CommandHistory h;
        h.record(rec("toy.set", t, {}, "a"), "");
        h.record(rec("toy.set", t, {}, "b"), "");
        CHECK(h.undo_stack().size() == 2);
    }
    {
        CommandHistory h;
        h.record(rec("toy.set", t, {}, "a"), "");
        h.record(rec("toy.other", t, {}, "a"), "");
        CHECK(h.undo_stack().size() == 2);
    }
    {
        CommandHistory h;
        h.record(rec("toy.set", t, {}, "a", CommandOrigin::ui), "");
        h.record(rec("toy.set", t, {}, "a", CommandOrigin::agent), "");
        CHECK(h.undo_stack().size() == 2);
    }
    {
        CommandHistory h;  // no merge key: no merge
        h.record(rec("toy.set", t), "");
        h.record(rec("toy.set", t), "");
        CHECK(h.undo_stack().size() == 2);
    }
    {
        CommandHistory h;  // an undo and a redo close the step
        h.record(rec("toy.set", t, {}, "a"), "");
        h.undone();
        h.redone();
        h.record(rec("toy.set", t, {}, "a"), "");
        CHECK(h.undo_stack().size() == 2);
    }
    {
        CommandHistory h;  // a command that cannot be undone closes merging too
        h.record(rec("toy.set", t, {}, "a"), "");
        h.pass_by();
        h.record(rec("toy.set", t, {}, "a"), "");
        CHECK(h.undo_stack().size() == 2);
    }
    {
        CommandHistory h;  // within a group, merging applies to the step's last record
        h.record(rec("toy.add", t, "g"), "");
        h.record(rec("toy.set", t, "g", "a"), "");
        h.record(rec("toy.set", t, "g", "a"), "");
        REQUIRE(h.undo_stack().size() == 1);
        CHECK(h.undo_stack()[0].records.size() == 2);
        CHECK(h.undo_stack()[0].records[1].merged == 2);
    }
    {
        HistoryLimits limits;
        limits.max_merged = 3;
        CommandHistory h(limits);
        for (int i = 0; i < 7; ++i) h.record(rec("toy.set", t, {}, "a"), "");
        CHECK(h.undo_stack().size() == 3);  // 3 + 3 + 1
    }
}

TEST_CASE("a merged inverse record keeps its first inverse and takes the newest invocation") {
    CommandHistory h;
    const auto t = Clock::now();
    for (int v = 1; v <= 3; ++v) {
        HistoryRecord r;
        r.invocation.name = "toy.set";
        r.invocation.merge_key = "k";
        r.invocation.args = {{"value", v}};
        r.undo = UndoRecord::inverse("toy.set", {{"value", v - 1}});
        r.last = t;
        r.seqs = {static_cast<std::uint64_t>(10 + v)};
        h.record(std::move(r), "");
    }
    REQUIRE(h.undo_stack().size() == 1);
    const auto& r = h.undo_stack()[0].records[0];
    CHECK(r.undo.inverse_args == json{{"value", 0}});
    CHECK(r.invocation.args == json{{"value", 3}});
    CHECK(h.undo_stack()[0].seqs() == std::vector<std::uint64_t>{11, 12, 13});
}

TEST_CASE("a new command and a command that cannot be undone clear the redo stack") {
    CommandHistory h;
    const auto t = Clock::now();
    h.record(rec("a.one", t), "");
    h.record(rec("a.two", t), "");
    h.undone();
    CHECK(h.can_redo());
    h.record(rec("a.three", t), "");
    CHECK_FALSE(h.can_redo());
    h.undone();
    CHECK(h.can_redo());
    h.pass_by();
    CHECK_FALSE(h.can_redo());
    CHECK(h.undo_stack().size() == 1);  // pass_by keeps the undo stack
}

TEST_CASE("the undo stack keeps at most max_steps steps") {
    HistoryLimits limits;
    limits.max_steps = 5;
    CommandHistory h(limits);
    for (int i = 0; i < 12; ++i) h.record(rec("a.n" + std::to_string(i), Clock::now()), "");
    REQUIRE(h.undo_stack().size() == 5);
    CHECK(h.undo_stack().front().records[0].invocation.name == "a.n7");
    CHECK(h.undo_stack().back().records[0].invocation.name == "a.n11");
}

TEST_CASE("memory beyond max_bytes drops the oldest steps but keeps the newest") {
    HistoryLimits limits;
    limits.max_bytes = 10000;
    CommandHistory h(limits);
    auto big = [](std::size_t bytes) {
        HistoryRecord r = rec("a.big", Clock::now());
        r.undo.bytes = bytes;
        return r;
    };
    for (int i = 0; i < 5; ++i) h.record(big(3000), "");
    CHECK(h.undo_stack().size() == 3);
    CHECK(h.bytes() == 9000);
    h.record(big(50000), "");  // larger than the bound alone
    CHECK(h.undo_stack().size() == 1);
    CHECK(h.bytes() == 50000);
    h.record(big(1), "");  // closures cost at least 256 bytes
    CHECK(h.undo_stack().size() == 1);
    CHECK(h.bytes() == 256);

    HistoryRecord inv = rec("a.inv", Clock::now());
    inv.invocation.args = {{"text", std::string(1000, 't')}};
    inv.undo = UndoRecord::inverse("a.inv", {{"text", std::string(500, 'u')}});
    CHECK(CommandHistory::record_cost(inv) == 1011 + 511 + 128);
}

TEST_CASE("undone and redone move whole steps, and summaries are newest first") {
    CommandHistory h;
    const auto t = Clock::now();
    h.record(rec("a.one", t, {}, {}, CommandOrigin::ui, 1), "t1");
    h.record(rec("a.two", t, "g", {}, CommandOrigin::agent, 2), "t2");
    h.record(rec("a.three", t, "g", {}, CommandOrigin::agent, 3), "t3");
    REQUIRE(h.top_undo());
    CHECK(h.top_undo()->seqs() == std::vector<std::uint64_t>{2, 3});
    h.undone();
    REQUIRE(h.top_redo());
    CHECK(h.top_redo()->records.size() == 2);
    const auto redo = h.redo_summaries(10);
    REQUIRE(redo.size() == 1);
    CHECK(history_step_summary_to_json(redo[0]) ==
          json{{"id", 2}, {"label", "a.two"}, {"count", 2}, {"group", "g"}, {"origin", "agent"}, {"at", "t3"}});
    h.redone();
    CHECK(h.undo_stack().size() == 2);
    CHECK_FALSE(h.can_redo());
    h.clear_undo();
    CHECK_FALSE(h.can_undo());
    CHECK(h.bytes() == 0);
}
