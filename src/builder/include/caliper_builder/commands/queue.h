#pragma once
// CommandQueue: brings invocations from any thread to the thread that drains
// the queue (design/command-layer.md, Threads and the queue).
//
// submit() may be called from any thread and returns a std::future of the
// result; it refuses with command.queue_full when 1024 items already wait.
// drain() runs the waiting items in submission order on the calling thread,
// at most `max` of them; the app calls it on the frame thread between frames.
// When the approval policy asks, the future waits for the person's decision
// by default: approve() and deny(), callable from any thread, queue the
// decision, which runs at the next drain() and completes both the waiting
// future and the future they return. A caller that must answer at once
// submits with wait_for_approval = false, gets command.needs_approval, and
// reads the outcome later with approval_outcome() (the last 64 are kept).
// A submit may also set approval_timeout: when the person has not decided by
// then, the next drain() withdraws the approval and completes the future with
// command.approval_timeout (journaled like a denial). Caliper API holds a call
// from the mcp and agent origins this way for up to 120 seconds.
// shutdown() completes every waiting future, those waiting for an approval
// included, with command.shutdown, and drops the registry's pending
// approvals; later submits complete at once with that error. The destructor
// shuts down. No future is ever left without a value. No function throws.
//
// peek() tells what drain() runs next without running it, so that an app can
// choose where to run it: the app runs a long command, such as loader.run, on
// a worker with drain(1) while its frame thread leaves the queue alone, and
// waits for a frame to be read back before it drains capture.frame
// (src/host/builder/app_session.h). Only the draining side may rely on the
// answer, since nothing but drain() and shutdown() removes an item.
#include <caliper_builder/commands/command.h>
#include <caliper_builder/commands/registry.h>

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <functional>
#include <future>
#include <limits>
#include <map>
#include <mutex>
#include <optional>
#include <string>

namespace caliper_builder {

// The item that drain() runs next: an invocation, or the person's decision on
// a pending approval, with its approval id.
struct QueuedItem {
    enum class Kind { invoke, approve, deny };
    Kind kind = Kind::invoke;
    CommandInvocation invocation;  // for invoke
    std::string approval;          // for approve and deny
};

struct SubmitOptions {
    bool wait_for_approval = true;
    // How long the future waits for the person's decision; zero waits without a limit.
    std::chrono::milliseconds approval_timeout{0};
};

class CommandQueue {
public:
    struct Options {
        std::size_t max_waiting = 1024;
        std::size_t max_outcomes = 64;
        std::function<std::chrono::steady_clock::time_point()> now;  // for the timeouts; tests set it
    };

    explicit CommandQueue(CommandRegistry& registry);
    CommandQueue(CommandRegistry& registry, Options options);
    ~CommandQueue();
    CommandQueue(const CommandQueue&) = delete;
    CommandQueue& operator=(const CommandQueue&) = delete;

    std::future<CommandResult> submit(CommandInvocation invocation, SubmitOptions options = {});
    std::future<CommandResult> approve(std::string approval_id);
    std::future<CommandResult> deny(std::string approval_id);

    // Runs up to `max` waiting items on the calling thread; returns how many ran.
    std::size_t drain(std::size_t max = std::numeric_limits<std::size_t>::max());

    // The item drain() runs next, or nothing when none waits or the queue is shut down.
    std::optional<QueuedItem> peek() const;

    void shutdown();
    bool is_shut_down() const;
    std::size_t waiting() const;
    std::optional<CommandResult> approval_outcome(const std::string& approval_id) const;

private:
    enum class Kind { invoke, approve, deny };
    struct Item {
        Kind kind = Kind::invoke;
        CommandInvocation invocation;
        std::string approval;
        bool wait = true;
        std::chrono::milliseconds timeout{0};
        std::promise<CommandResult> promise;
    };
    struct Waiting {
        std::promise<CommandResult> promise;
        std::optional<std::chrono::steady_clock::time_point> deadline;
    };
    std::future<CommandResult> enqueue(Item item);
    void remember(const std::string& approval, const CommandResult& result);
    void expire_waits();

    CommandRegistry& registry_;
    Options options_;
    mutable std::mutex mutex_;
    std::deque<Item> items_;
    std::map<std::string, Waiting> waiting_approval_;
    std::deque<std::pair<std::string, CommandResult>> outcomes_;
    bool shut_ = false;
};

}  // namespace caliper_builder
