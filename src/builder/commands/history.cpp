#include <caliper_builder/commands/history.h>

#include <algorithm>
#include <utility>

namespace caliper_builder {

namespace {

using nlohmann::json;

std::size_t json_size(const json& v) { return v.dump(-1, ' ', false, json::error_handler_t::replace).size(); }

}  // namespace

UndoRecord UndoRecord::closures(std::function<Error()> undo, std::function<Error()> redo, std::size_t bytes) {
    UndoRecord r;
    r.kind = Kind::closures;
    r.undo = std::move(undo);
    r.redo = std::move(redo);
    r.bytes = bytes;
    return r;
}

UndoRecord UndoRecord::inverse(std::string name, json args) {
    UndoRecord r;
    r.kind = Kind::inverse;
    r.inverse_name = std::move(name);
    r.inverse_args = std::move(args);
    return r;
}

UndoRecord UndoRecord::none() { return UndoRecord{}; }

std::vector<std::uint64_t> HistoryStep::seqs() const {
    std::vector<std::uint64_t> out;
    for (const auto& r : records)
        for (const auto s : r.seqs)
            if (s != 0) out.push_back(s);
    return out;
}

std::size_t HistoryStep::count() const {
    std::size_t n = 0;
    for (const auto& r : records) n += r.merged;
    return n;
}

json history_step_summary_to_json(const HistoryStepSummary& s) {
    return json{{"id", s.id},
                {"label", s.label},
                {"count", s.count},
                {"group", s.group.empty() ? json(nullptr) : json(s.group)},
                {"origin", command_origin_name(s.origin)},
                {"at", s.at}};
}

CommandHistory::CommandHistory(HistoryLimits limits) : limits_(limits) {
    if (limits_.max_steps == 0) limits_.max_steps = 1;
    if (limits_.max_merged == 0) limits_.max_merged = 1;
}

std::size_t CommandHistory::record_cost(const HistoryRecord& r) {
    switch (r.undo.kind) {
        case UndoRecord::Kind::closures: return std::max<std::size_t>(r.undo.bytes, 256);
        case UndoRecord::Kind::inverse:
            return json_size(r.undo.inverse_args) + json_size(r.invocation.args) + 128;
        case UndoRecord::Kind::none: return 0;
    }
    return 0;
}

void CommandHistory::record(HistoryRecord rec, std::string at) {
    redo_.clear();
    rec.cost = record_cost(rec);
    const CommandInvocation& inv = rec.invocation;
    if (!undo_.empty() && !sealed_) {
        HistoryStep& top = undo_.back();
        if (!inv.merge_key.empty() && top.id == open_step_ && top.group == inv.group) {
            HistoryRecord& last = top.records.back();
            const bool same = last.invocation.merge_key == inv.merge_key && last.invocation.name == inv.name &&
                              last.invocation.origin == inv.origin && last.undo.kind == rec.undo.kind;
            const bool soon = rec.last >= last.last && rec.last - last.last <= limits_.merge_window;
            if (same && soon && last.merged < limits_.max_merged) {
                // Keep the first undo; take the newest redo and invocation.
                if (rec.undo.kind == UndoRecord::Kind::closures) {
                    last.undo.redo = std::move(rec.undo.redo);
                    last.undo.bytes = std::max(last.undo.bytes, rec.undo.bytes);
                }
                last.invocation = std::move(rec.invocation);
                last.seqs.insert(last.seqs.end(), rec.seqs.begin(), rec.seqs.end());
                last.merged += 1;
                last.last = rec.last;
                last.destructive = last.destructive || rec.destructive;
                last.cost = record_cost(last);
                top.at = std::move(at);
                enforce();
                return;
            }
        }
        if (!inv.group.empty() && top.group == inv.group) {
            top.records.push_back(std::move(rec));
            top.at = std::move(at);
            open_step_ = top.id;
            enforce();
            return;
        }
    }
    sealed_ = false;
    HistoryStep step;
    step.id = next_id_++;
    step.group = inv.group;
    step.origin = inv.origin;
    step.at = std::move(at);
    step.records.push_back(std::move(rec));
    open_step_ = step.id;
    undo_.push_back(std::move(step));
    enforce();
}

void CommandHistory::pass_by() {
    redo_.clear();
    open_step_ = 0;
    enforce();
}

void CommandHistory::seal() {
    sealed_ = true;
    open_step_ = 0;
}

HistoryStep* CommandHistory::top_undo() { return undo_.empty() ? nullptr : &undo_.back(); }
HistoryStep* CommandHistory::top_redo() { return redo_.empty() ? nullptr : &redo_.back(); }

void CommandHistory::undone() {
    if (undo_.empty()) return;
    redo_.push_back(std::move(undo_.back()));
    undo_.pop_back();
    open_step_ = 0;
}

void CommandHistory::redone() {
    if (redo_.empty()) return;
    undo_.push_back(std::move(redo_.back()));
    redo_.pop_back();
    open_step_ = 0;
    enforce();
}

void CommandHistory::clear_undo() {
    undo_.clear();
    open_step_ = 0;
    enforce();
}

void CommandHistory::clear_redo() {
    redo_.clear();
    enforce();
}

void CommandHistory::clear() {
    undo_.clear();
    redo_.clear();
    open_step_ = 0;
    sealed_ = false;
    bytes_ = 0;
}

void CommandHistory::enforce() {
    auto step_cost = [](HistoryStep& s) {
        s.cost = 0;
        for (const auto& r : s.records) s.cost += r.cost;
        return s.cost;
    };
    while (undo_.size() > limits_.max_steps) undo_.erase(undo_.begin());
    std::size_t total = 0;
    for (auto& s : undo_) total += step_cost(s);
    for (auto& s : redo_) total += step_cost(s);
    // Drop the oldest undo steps, never the newest, until the total fits.
    std::size_t drop = 0;
    while (total > limits_.max_bytes && drop + 1 < undo_.size()) total -= undo_[drop++].cost;
    if (drop > 0) undo_.erase(undo_.begin(), undo_.begin() + static_cast<std::ptrdiff_t>(drop));
    bytes_ = total;
}

HistoryStepSummary CommandHistory::summarize(const HistoryStep& step) {
    HistoryStepSummary s;
    s.id = step.id;
    s.label = step.records.empty() ? std::string() : step.records.front().invocation.name;
    s.count = step.count();
    s.group = step.group;
    s.origin = step.origin;
    s.at = step.at;
    return s;
}

std::vector<HistoryStepSummary> CommandHistory::undo_summaries(std::size_t limit) const {
    std::vector<HistoryStepSummary> out;
    for (auto it = undo_.rbegin(); it != undo_.rend() && out.size() < limit; ++it) out.push_back(summarize(*it));
    return out;
}

std::vector<HistoryStepSummary> CommandHistory::redo_summaries(std::size_t limit) const {
    std::vector<HistoryStepSummary> out;
    for (auto it = redo_.rbegin(); it != redo_.rend() && out.size() < limit; ++it) out.push_back(summarize(*it));
    return out;
}

}  // namespace caliper_builder
