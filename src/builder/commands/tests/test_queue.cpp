// Tests of CommandQueue: submit from many threads and drain on one, the
// bounds, the wait for approval with and without a time limit, and shutdown,
// which never leaves a future without a value.
#include <caliper_builder/commands/queue.h>

#include <doctest/doctest.h>

#include <atomic>
#include <thread>
#include <vector>

using caliper_builder::CommandContext;
using caliper_builder::CommandDescriptor;
using caliper_builder::CommandInvocation;
using caliper_builder::CommandOrigin;
using caliper_builder::CommandOutput;
using caliper_builder::CommandQueue;
using caliper_builder::CommandRegistry;
using caliper_builder::CommandResult;
using caliper_builder::SubmitOptions;
using nlohmann::json;
using namespace std::chrono_literals;
namespace cb = caliper_builder;

namespace {

CommandDescriptor descriptor(const std::string& name, bool destructive = false) {
    CommandDescriptor d;
    d.name = name;
    d.summary = "A test command.";
    d.args = json::parse(R"({"type": "object", "properties": {"n": {"type": "integer"}}})");
    d.result = json::parse(R"({"type": "object"})");
    d.mutates = true;
    d.destructive = destructive;
    return d;
}

CommandInvocation inv(const std::string& name, int n = 0, CommandOrigin origin = CommandOrigin::ui) {
    CommandInvocation i;
    i.name = name;
    i.args = {{"n", n}};
    i.origin = origin;
    return i;
}

bool ready(const std::future<CommandResult>& f) { return f.wait_for(0s) == std::future_status::ready; }

}  // namespace

TEST_CASE("invocations submitted from several threads run in order on the draining thread") {
    CommandRegistry registry;
    std::vector<int> ran;
    std::thread::id drain_thread;
    REQUIRE(registry.add(descriptor("toy.append"), [&](const CommandContext& c) {
                         CHECK(std::this_thread::get_id() == drain_thread);
                         ran.push_back(c.invocation.args["n"].get<int>());
                         return CommandOutput::ok({{"n", c.invocation.args["n"]}});
                     }).ok());
    CommandQueue queue(registry);
    drain_thread = std::this_thread::get_id();
    registry.set_frame_thread(drain_thread);

    constexpr int kThreads = 4;
    constexpr int kEach = 50;
    std::vector<std::vector<std::future<CommandResult>>> futures(kThreads);
    std::atomic<int> submitted{0};
    std::vector<std::thread> threads;
    for (int t = 0; t < kThreads; ++t)
        threads.emplace_back([&, t] {
            for (int i = 0; i < kEach; ++i) {
                futures[t].push_back(queue.submit(inv("toy.append", t * 1000 + i)));
                ++submitted;
            }
        });
    // Drain on this thread while the others submit.
    std::size_t total = 0;
    while (total < kThreads * kEach) {
        total += queue.drain();
        std::this_thread::yield();
    }
    for (auto& th : threads) th.join();
    CHECK(queue.drain() == 0);
    CHECK(ran.size() == kThreads * kEach);
    for (int t = 0; t < kThreads; ++t) {
        int last = -1;
        for (int i = 0; i < kEach; ++i) {
            const CommandResult r = futures[t][i].get();
            REQUIRE(r.ok);
            CHECK(r.result["n"] == t * 1000 + i);
        }
        // Each thread's invocations ran in the order it submitted them.
        for (const int n : ran)
            if (n / 1000 == t) {
                CHECK(n > last);
                last = n;
            }
    }
}

TEST_CASE("drain runs at most the given number, and the queue has a bound") {
    CommandRegistry registry;
    REQUIRE(registry.add(descriptor("toy.noop"), [](const CommandContext&) { return CommandOutput::ok({}); }).ok());
    CommandQueue::Options options;
    options.max_waiting = 3;
    CommandQueue queue(registry, options);
    auto a = queue.submit(inv("toy.noop"));
    auto b = queue.submit(inv("toy.noop"));
    auto c = queue.submit(inv("toy.noop"));
    auto d = queue.submit(inv("toy.noop"));
    REQUIRE(ready(d));
    CHECK(d.get().error.code == "command.queue_full");
    CHECK(queue.waiting() == 3);
    CHECK(queue.drain(2) == 2);
    CHECK(ready(a));
    CHECK(ready(b));
    CHECK_FALSE(ready(c));
    CHECK(queue.drain() == 1);
    CHECK(c.get().ok);
}

TEST_CASE("a submit waits for the person's decision, which another thread gives") {
    CommandRegistry registry;
    int deletes = 0;
    REQUIRE(registry.add(descriptor("toy.delete", true), [&](const CommandContext&) {
                         ++deletes;
                         return CommandOutput::ok({{"deleted", true}});
                     }).ok());
    CommandQueue queue(registry);
    auto f = queue.submit(inv("toy.delete", 1, CommandOrigin::mcp));
    queue.drain();
    CHECK_FALSE(ready(f));
    const auto pending = registry.pending_approvals();
    REQUIRE(pending.size() == 1);

    std::future<CommandResult> approver;
    std::thread ui([&] { approver = queue.approve(pending[0].approval); });
    ui.join();
    CHECK(deletes == 0);
    queue.drain();
    REQUIRE(ready(f));
    const CommandResult r = f.get();
    CHECK(r.ok);
    CHECK(r.result["deleted"] == true);
    CHECK(approver.get().ok);
    CHECK(deletes == 1);
    REQUIRE(queue.approval_outcome(pending[0].approval));
    CHECK(queue.approval_outcome(pending[0].approval)->ok);

    // A denial completes the waiting future with command.denied.
    auto g = queue.submit(inv("toy.delete", 2, CommandOrigin::agent));
    queue.drain();
    auto denier = queue.deny(registry.pending_approvals().at(0).approval);
    queue.drain();
    CHECK(g.get().error.code == "command.denied");
    CHECK(denier.get().error.code == "command.denied");
    CHECK(deletes == 1);

    // An unknown approval id answers only the caller.
    auto unknown = queue.approve("ap-99");
    queue.drain();
    CHECK(unknown.get().error.code == "command.unknown_approval");
}

TEST_CASE("a caller that must answer at once gets needs_approval and reads the outcome later") {
    CommandRegistry registry;
    REQUIRE(registry.add(descriptor("toy.delete", true), [](const CommandContext&) { return CommandOutput::ok({}); }).ok());
    CommandQueue queue(registry);
    SubmitOptions now;
    now.wait_for_approval = false;
    auto f = queue.submit(inv("toy.delete", 1, CommandOrigin::api), now);
    queue.drain();
    const CommandResult asked = f.get();
    REQUIRE(asked.error.code == "command.needs_approval");
    CHECK(!queue.approval_outcome(asked.approval));
    queue.approve(asked.approval);
    queue.drain();
    REQUIRE(queue.approval_outcome(asked.approval));
    CHECK(queue.approval_outcome(asked.approval)->ok);
}

TEST_CASE("a wait with a time limit is withdrawn when nobody decides") {
    CommandRegistry registry;
    int deletes = 0;
    REQUIRE(registry.add(descriptor("toy.delete", true), [&](const CommandContext&) {
                         ++deletes;
                         return CommandOutput::ok({});
                     }).ok());
    auto clock = std::chrono::steady_clock::time_point{};
    CommandQueue::Options options;
    options.now = [&clock] { return clock; };
    CommandQueue queue(registry, options);
    SubmitOptions limited;
    limited.approval_timeout = 120s;
    auto f = queue.submit(inv("toy.delete", 1, CommandOrigin::mcp), limited);
    queue.drain();
    const std::string approval = registry.pending_approvals().at(0).approval;
    clock += 119s;
    queue.drain();
    CHECK_FALSE(ready(f));
    clock += 2s;
    queue.drain();
    REQUIRE(ready(f));
    CHECK(f.get().error.code == "command.approval_timeout");
    CHECK(registry.pending_approvals().empty());
    CHECK(queue.approval_outcome(approval)->error.code == "command.approval_timeout");
    // A late approval finds nothing to approve.
    auto late = queue.approve(approval);
    queue.drain();
    CHECK(late.get().error.code == "command.unknown_approval");
    CHECK(deletes == 0);
}

TEST_CASE("shutdown completes every future, and later submits complete at once") {
    CommandRegistry registry;
    REQUIRE(registry.add(descriptor("toy.noop"), [](const CommandContext&) { return CommandOutput::ok({}); }).ok());
    REQUIRE(registry.add(descriptor("toy.delete", true), [](const CommandContext&) { return CommandOutput::ok({}); }).ok());
    CommandQueue queue(registry);
    auto parked = queue.submit(inv("toy.delete", 1, CommandOrigin::mcp));
    queue.drain();
    auto queued = queue.submit(inv("toy.noop"));
    CHECK(registry.pending_approvals().size() == 1);

    std::thread other([&] { queue.shutdown(); });
    other.join();
    CHECK(queue.is_shut_down());
    CHECK(parked.get().error.code == "command.shutdown");
    CHECK(queued.get().error.code == "command.shutdown");
    CHECK(registry.pending_approvals().empty());
    auto after = queue.submit(inv("toy.noop"));
    REQUIRE(ready(after));
    CHECK(after.get().error.code == "command.shutdown");
    CHECK(queue.drain() == 0);
    queue.shutdown();  // twice is fine
}

TEST_CASE("peek tells what drain runs next without running it") {
    CommandRegistry registry;
    int runs = 0;
    REQUIRE(registry.add(descriptor("toy.noop"), [&](const CommandContext&) {
                         ++runs;
                         return CommandOutput::ok({});
                     }).ok());
    REQUIRE(registry.add(descriptor("toy.delete", true), [](const CommandContext&) { return CommandOutput::ok({}); }).ok());
    CommandQueue queue(registry);
    CHECK_FALSE(queue.peek().has_value());

    auto a = queue.submit(inv("toy.noop", 7, CommandOrigin::mcp));
    auto b = queue.submit(inv("toy.delete", 1, CommandOrigin::mcp));
    auto next = queue.peek();
    REQUIRE(next.has_value());
    CHECK(next->kind == cb::QueuedItem::Kind::invoke);
    CHECK(next->invocation.name == "toy.noop");
    CHECK(next->invocation.args == json{{"n", 7}});
    CHECK(next->invocation.origin == CommandOrigin::mcp);
    CHECK(queue.peek()->invocation.name == "toy.noop");  // peeking twice changes nothing
    CHECK(runs == 0);
    CHECK(queue.waiting() == 2);

    CHECK(queue.drain(1) == 1);
    CHECK(a.get().ok);
    CHECK(queue.peek()->invocation.name == "toy.delete");
    CHECK(queue.drain(1) == 1);  // parked for approval
    REQUIRE(registry.pending_approvals().size() == 1);
    const std::string id = registry.pending_approvals()[0].approval;
    CHECK_FALSE(queue.peek().has_value());

    auto decided = queue.approve(id);
    next = queue.peek();
    REQUIRE(next.has_value());
    CHECK(next->kind == cb::QueuedItem::Kind::approve);
    CHECK(next->approval == id);
    CHECK(next->invocation.name.empty());
    CHECK(queue.drain(1) == 1);
    CHECK(decided.get().ok);
    CHECK(b.get().ok);

    queue.submit(inv("toy.noop"));
    queue.shutdown();
    CHECK_FALSE(queue.peek().has_value());
}

TEST_CASE("the destructor shuts the queue down") {
    CommandRegistry registry;
    REQUIRE(registry.add(descriptor("toy.noop"), [](const CommandContext&) { return CommandOutput::ok({}); }).ok());
    std::future<CommandResult> f;
    {
        CommandQueue queue(registry);
        f = queue.submit(inv("toy.noop"));
    }
    CHECK(f.get().error.code == "command.shutdown");
}
