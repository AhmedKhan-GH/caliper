#include <caliper_builder/commands/queue.h>

#include <algorithm>
#include <utility>
#include <vector>

namespace caliper_builder {

namespace {

CommandResult shutdown_result(const std::string& id) {
    return CommandResult::failure(id, Error("command.shutdown", "The command queue shut down before this completed.", ""));
}

std::future<CommandResult> ready(CommandResult result) {
    std::promise<CommandResult> p;
    p.set_value(std::move(result));
    return p.get_future();
}

}  // namespace

CommandQueue::CommandQueue(CommandRegistry& registry) : CommandQueue(registry, Options{}) {}

CommandQueue::CommandQueue(CommandRegistry& registry, Options options) : registry_(registry), options_(options) {
    if (options_.max_waiting == 0) options_.max_waiting = 1;
    if (!options_.now) options_.now = [] { return std::chrono::steady_clock::now(); };
}

CommandQueue::~CommandQueue() { shutdown(); }

std::future<CommandResult> CommandQueue::enqueue(Item item) {
    std::lock_guard<std::mutex> lock(mutex_);
    const std::string id = item.kind == Kind::invoke ? item.invocation.id : std::string();
    if (shut_) return ready(shutdown_result(id));
    if (items_.size() >= options_.max_waiting)
        return ready(CommandResult::failure(
            id, Error("command.queue_full", "Too many commands already wait to run; try again later.", "")));
    std::future<CommandResult> f = item.promise.get_future();
    items_.push_back(std::move(item));
    return f;
}

std::future<CommandResult> CommandQueue::submit(CommandInvocation invocation, SubmitOptions options) {
    Item item;
    item.kind = Kind::invoke;
    item.invocation = std::move(invocation);
    item.wait = options.wait_for_approval;
    item.timeout = options.approval_timeout;
    return enqueue(std::move(item));
}

std::future<CommandResult> CommandQueue::approve(std::string approval_id) {
    Item item;
    item.kind = Kind::approve;
    item.approval = std::move(approval_id);
    return enqueue(std::move(item));
}

std::future<CommandResult> CommandQueue::deny(std::string approval_id) {
    Item item;
    item.kind = Kind::deny;
    item.approval = std::move(approval_id);
    return enqueue(std::move(item));
}

void CommandQueue::remember(const std::string& approval, const CommandResult& result) {
    outcomes_.emplace_back(approval, result);
    while (outcomes_.size() > options_.max_outcomes) outcomes_.pop_front();
}

void CommandQueue::expire_waits() {
    std::vector<std::pair<std::string, std::promise<CommandResult>>> expired;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        if (waiting_approval_.empty()) return;
        const auto now = options_.now();
        for (auto it = waiting_approval_.begin(); it != waiting_approval_.end();) {
            if (it->second.deadline && now >= *it->second.deadline) {
                expired.emplace_back(it->first, std::move(it->second.promise));
                it = waiting_approval_.erase(it);
            } else {
                ++it;
            }
        }
    }
    for (auto& e : expired) {
        CommandResult r = registry_.expire_approval(e.first);
        {
            std::lock_guard<std::mutex> lock(mutex_);
            if (r.error.code != "command.unknown_approval") remember(e.first, r);
        }
        e.second.set_value(std::move(r));
    }
}

std::size_t CommandQueue::drain(std::size_t max) {
    expire_waits();
    std::size_t ran = 0;
    while (ran < max) {
        Item item;
        {
            std::lock_guard<std::mutex> lock(mutex_);
            if (shut_ || items_.empty()) break;
            item = std::move(items_.front());
            items_.pop_front();
        }
        ++ran;
        if (item.kind == Kind::invoke) {
            CommandResult r = registry_.invoke(std::move(item.invocation));
            if (r.error.code == "command.needs_approval" && item.wait) {
                std::lock_guard<std::mutex> lock(mutex_);
                if (shut_) {
                    registry_.cancel_approval(r.approval);
                    item.promise.set_value(shutdown_result(r.id));
                } else {
                    Waiting w;
                    w.promise = std::move(item.promise);
                    if (item.timeout.count() > 0) w.deadline = options_.now() + item.timeout;
                    waiting_approval_.emplace(r.approval, std::move(w));
                }
                continue;
            }
            item.promise.set_value(std::move(r));
            continue;
        }
        CommandResult r =
            item.kind == Kind::approve ? registry_.approve(item.approval) : registry_.deny(item.approval);
        {
            std::lock_guard<std::mutex> lock(mutex_);
            if (r.error.code != "command.unknown_approval" && r.error.code != "command.wrong_thread") {
                remember(item.approval, r);
                const auto it = waiting_approval_.find(item.approval);
                if (it != waiting_approval_.end()) {
                    it->second.promise.set_value(r);
                    waiting_approval_.erase(it);
                }
            }
        }
        item.promise.set_value(std::move(r));
    }
    return ran;
}

void CommandQueue::shutdown() {
    std::deque<Item> items;
    std::map<std::string, Waiting> waiting;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        if (shut_) return;
        shut_ = true;
        items.swap(items_);
        waiting.swap(waiting_approval_);
    }
    for (auto& item : items)
        item.promise.set_value(shutdown_result(item.kind == Kind::invoke ? item.invocation.id : std::string()));
    for (auto& w : waiting) w.second.promise.set_value(shutdown_result(""));
    registry_.cancel_pending();
}

std::optional<QueuedItem> CommandQueue::peek() const {
    std::lock_guard<std::mutex> lock(mutex_);
    if (shut_ || items_.empty()) return std::nullopt;
    const Item& front = items_.front();
    QueuedItem out;
    out.kind = front.kind == Kind::invoke ? QueuedItem::Kind::invoke
               : front.kind == Kind::approve ? QueuedItem::Kind::approve
                                             : QueuedItem::Kind::deny;
    if (front.kind == Kind::invoke) out.invocation = front.invocation;
    out.approval = front.approval;
    return out;
}

bool CommandQueue::is_shut_down() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return shut_;
}

std::size_t CommandQueue::waiting() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return items_.size();
}

std::optional<CommandResult> CommandQueue::approval_outcome(const std::string& approval_id) const {
    std::lock_guard<std::mutex> lock(mutex_);
    for (auto it = outcomes_.rbegin(); it != outcomes_.rend(); ++it)
        if (it->first == approval_id) return it->second;
    return std::nullopt;
}

std::optional<std::chrono::milliseconds> CommandQueue::approval_time_left(const std::string& approval_id) const {
    std::lock_guard<std::mutex> lock(mutex_);
    const auto it = waiting_approval_.find(approval_id);
    if (it == waiting_approval_.end() || !it->second.deadline) return std::nullopt;
    const auto left = std::chrono::duration_cast<std::chrono::milliseconds>(*it->second.deadline - options_.now());
    return std::max(left, std::chrono::milliseconds(0));
}

}  // namespace caliper_builder
