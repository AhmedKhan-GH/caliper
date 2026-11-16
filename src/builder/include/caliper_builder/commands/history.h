#pragma once
// The undo history of a workspace (docs/design/command-layer.md, Undo): one undo
// stack and one redo stack of steps, each step one or more records that undo
// and redo together.
//
// A handler of a mutating command returns an UndoRecord: a closure pair, an
// inverse invocation, or none. CommandHistory only keeps the records; the
// registry runs them. record() files a successful command's record:
//   - with a merge key, it merges into the last record of the top step when
//     that record has the same key, name, origin and kind, the step has the
//     same group and is still open (no undo or redo since), the last command
//     ran within the merge window and fewer than max_merged commands merged;
//     a merged record keeps its first undo and takes the newest redo;
//   - with a group equal to the top step's group, it joins that step;
//   - otherwise it begins a new step.
// Every record() and pass_by() clears the redo stack; pass_by() is for a
// mutating command that cannot be undone, and also closes merging. The undo
// stack holds at most max_steps steps, and both stacks at most max_bytes,
// dropping the oldest undo steps first but never the newest one.
//
// Not thread-safe: the registry guards it with its lock. No function throws.
#include <caliper_builder/commands/command.h>
#include <caliper_builder/error.h>

#include <nlohmann/json.hpp>

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <string>
#include <vector>

namespace caliper_builder {

struct UndoRecord {
    enum class Kind { none, closures, inverse };
    Kind kind = Kind::none;
    // A closure pair: each returns an empty Error on success. An undo must
    // check that the state it replaces is the one its command left, and fail
    // with "history.conflict" when it is not.
    std::function<Error()> undo;
    std::function<Error()> redo;
    std::size_t bytes = 0;  // the memory the closures hold, as the handler estimates it
    // An inverse invocation, run through the registry's handlers.
    std::string inverse_name;
    nlohmann::json inverse_args = nlohmann::json::object();

    static UndoRecord closures(std::function<Error()> undo, std::function<Error()> redo, std::size_t bytes = 0);
    static UndoRecord inverse(std::string name, nlohmann::json args);
    static UndoRecord none();
};

struct HistoryLimits {
    std::size_t max_steps = 200;
    std::size_t max_bytes = std::size_t(64) << 20;
    std::chrono::milliseconds merge_window{1000};
    std::size_t max_merged = 4096;
};

struct HistoryRecord {
    CommandInvocation invocation;  // the newest invocation merged into the record
    UndoRecord undo;
    std::vector<std::uint64_t> seqs;  // the journal's seq of each do event (none when there is no journal)
    std::size_t merged = 1;
    std::chrono::steady_clock::time_point last;  // when the newest merged command ran
    bool destructive = false;                    // the command is destructive
    bool inverse_destructive = false;            // its inverse names a destructive command
    std::size_t cost = 0;                        // set by the history
};

struct HistoryStep {
    std::uint64_t id = 0;
    std::string group;
    CommandOrigin origin = CommandOrigin::ui;
    std::vector<HistoryRecord> records;
    std::string at;  // UTC time of the newest command
    std::size_t cost = 0;

    std::vector<std::uint64_t> seqs() const;  // of every record, in order
    std::size_t count() const;                // commands, merged ones included
};

struct HistoryStepSummary {
    std::uint64_t id = 0;
    std::string label;  // the name of the step's first command
    std::size_t count = 0;
    std::string group;
    CommandOrigin origin = CommandOrigin::ui;
    std::string at;
};

nlohmann::json history_step_summary_to_json(const HistoryStepSummary& summary);

class CommandHistory {
public:
    explicit CommandHistory(HistoryLimits limits = {});

    void record(HistoryRecord record, std::string at);
    void pass_by();

    HistoryStep* top_undo();
    HistoryStep* top_redo();
    const std::vector<HistoryStep>& undo_stack() const { return undo_; }  // back is the top
    const std::vector<HistoryStep>& redo_stack() const { return redo_; }
    bool can_undo() const { return !undo_.empty(); }
    bool can_redo() const { return !redo_.empty(); }

    void undone();  // moves the top undo step to the redo stack and closes merging
    void redone();  // moves the top redo step to the undo stack and closes merging
    void clear_undo();
    void clear_redo();
    void clear();

    std::size_t bytes() const { return bytes_; }
    const HistoryLimits& limits() const { return limits_; }

    // Newest first, at most `limit`.
    std::vector<HistoryStepSummary> undo_summaries(std::size_t limit) const;
    std::vector<HistoryStepSummary> redo_summaries(std::size_t limit) const;
    static HistoryStepSummary summarize(const HistoryStep& step);

    // A closure pair costs what its handler declares, at least 256 bytes; an
    // inverse costs its arguments and the original's as JSON text, plus 128.
    static std::size_t record_cost(const HistoryRecord& record);

private:
    void enforce();

    HistoryLimits limits_;
    std::vector<HistoryStep> undo_;
    std::vector<HistoryStep> redo_;
    std::uint64_t next_id_ = 1;
    std::uint64_t open_step_ = 0;  // the step that may still take merges
    std::size_t bytes_ = 0;
};

}  // namespace caliper_builder
