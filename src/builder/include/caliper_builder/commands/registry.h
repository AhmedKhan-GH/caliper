#pragma once
// CommandRegistry: the command layer's core (docs/plans/command-layer.md). It
// maps command names to descriptors and handlers, and runs an invocation:
//   1. checks the invocation, finds the command and its handler, checks the
//      thread for a frame_thread command, and validates the arguments against
//      the schema (command.invalid_invocation, command.unknown,
//      command.not_bound, command.wrong_thread, command.invalid_args);
//   2. asks the approval policy, and on ask parks the invocation and answers
//      command.needs_approval with an approval id;
//   3. runs the handler; an exception that escapes it becomes
//      command.handler_failed, and a handler that invokes a command gets
//      command.reentrant;
//   4. for a mutating command, writes the journal's do event (also for a
//      failure or a denial) and files the undo record in the history.
// history.undo, history.redo and history.list are built in: every registry
// has them, bound to the history it owns. Undoing or redoing a step asks the
// approval policy about each destructive command in it, with the history
// command's origin; a failed undo or redo rolls the step back, drops the
// steps that depend on it and answers history.undo_failed or
// history.redo_failed (history.inconsistent if the rollback fails too).
//
// The registry runs one invocation at a time under one lock that also covers
// the history, the journal and the approvals. It is safe to call from any
// thread, and invoke() runs the command on the calling thread; the queue
// (queue.h) brings invocations to the frame thread. No function here throws.
#include <caliper_builder/commands/approval.h>
#include <caliper_builder/commands/command.h>
#include <caliper_builder/commands/history.h>
#include <caliper_builder/commands/journal.h>
#include <caliper_builder/error.h>

#include <nlohmann/json.hpp>

#include <chrono>
#include <cstdint>
#include <functional>
#include <map>
#include <mutex>
#include <optional>
#include <string>
#include <string_view>
#include <thread>
#include <utility>
#include <vector>

namespace caliper_builder {

struct CommandContext {
    const CommandInvocation& invocation;
    const CommandDescriptor& descriptor;
    bool redo = false;  // the handler runs again to redo its command
};

// What a handler returns: a result object or an error, and for a mutating
// command that succeeded, its undo record (UndoRecord::none() when it cannot
// be undone).
struct CommandOutput {
    nlohmann::json result = nlohmann::json::object();
    Error error;
    UndoRecord undo;

    static CommandOutput ok(nlohmann::json result, UndoRecord undo = UndoRecord::none());
    static CommandOutput fail(Error error);
};

using CommandHandler = std::function<CommandOutput(const CommandContext&)>;

struct RegistryOptions {
    bool check_results = false;  // validate every result against its schema (the tests do)
    HistoryLimits history;
    std::size_t max_pending = 64;
    std::function<std::chrono::steady_clock::time_point()> steady_now;  // for the merge window; tests set it
    std::function<std::chrono::system_clock::time_point()> wall_now;    // for timestamps; tests set it
};

// The descriptors of history.undo, history.redo and history.list.
std::vector<CommandDescriptor> history_command_descriptors();

class CommandRegistry {
public:
    explicit CommandRegistry(RegistryOptions options = {});
    CommandRegistry(const CommandRegistry&) = delete;
    CommandRegistry& operator=(const CommandRegistry&) = delete;

    // Adds a command, with or without its handler (command.bad_descriptor
    // for a descriptor that fails check_command_descriptor() or a name taken).
    Error add(CommandDescriptor descriptor, CommandHandler handler = {});
    // Adds every descriptor that is not there yet, such as the descriptor
    // table; the history commands keep their built-in handlers.
    Error add_all(const std::vector<CommandDescriptor>& descriptors);
    Error bind(std::string_view name, CommandHandler handler);

    std::optional<CommandDescriptor> descriptor(std::string_view name) const;
    std::vector<CommandDescriptor> descriptors() const;  // in the order added
    bool is_bound(std::string_view name) const;

    void set_approval_policy(ApprovalPolicy policy);  // empty: the default policy
    void set_journal(CommandJournal* journal);        // not owned; may be null
    void set_frame_thread(std::optional<std::thread::id> id);

    CommandResult invoke(CommandInvocation invocation);
    CommandResult approve(const std::string& approval_id);
    CommandResult deny(const std::string& approval_id);
    // Withdraws an invocation whose wait for approval ran out: it is journaled
    // like a denial and answers command.approval_timeout.
    CommandResult expire_approval(const std::string& approval_id);
    bool cancel_approval(const std::string& approval_id);  // drops it unanswered, unjournaled
    std::size_t cancel_pending();                          // drops them all
    std::vector<PendingApproval> pending_approvals() const;

    // The last error from writing the journal; commands do not fail for it.
    Error journal_error() const;

    struct HistoryState {
        bool can_undo = false;
        bool can_redo = false;
        std::vector<HistoryStepSummary> undo;  // newest first
        std::vector<HistoryStepSummary> redo;
        std::size_t bytes = 0;
    };
    HistoryState history_state(std::size_t limit = 50) const;

private:
    struct Entry {
        CommandDescriptor descriptor;
        CommandHandler handler;
    };
    struct Parked {
        std::string approval;
        CommandInvocation invocation;
        std::string requested_at;
        std::vector<std::uint64_t> steps;  // for history.undo and history.redo: the steps it would touch
    };

    CommandResult run(const CommandInvocation& inv, const Entry& entry, const std::vector<std::uint64_t>& steps);
    CommandResult run_history_move(const CommandInvocation& inv, bool undo, const std::vector<std::uint64_t>& steps);
    CommandResult run_history_list(const CommandInvocation& inv);
    std::pair<ApprovalDecision, std::vector<std::uint64_t>> history_decision(const CommandInvocation& inv, bool undo);
    ApprovalDecision ask_policy(const CommandDescriptor& d, const CommandInvocation& inv);
    CommandOutput call(const Entry& entry, const CommandInvocation& inv, bool redo);
    Error undo_record(HistoryRecord& record, const CommandInvocation& by);
    Error redo_record(HistoryRecord& record, const CommandInvocation& by);
    Error undo_step(HistoryStep& step, const CommandInvocation& by);
    Error redo_step(HistoryStep& step, const CommandInvocation& by);
    std::uint64_t journal_do(const CommandInvocation& inv, bool ok, const std::string& error);
    void journal_history(const CommandInvocation& inv, bool undo, const std::string& group,
                         std::vector<std::uint64_t> seqs, bool ok, const std::string& error);
    bool wrong_thread(const CommandDescriptor& d) const;
    std::string now_text() const;
    CommandResult finish_refused(const CommandInvocation& inv, const Entry& entry, const std::string& code,
                                 std::string message);
    CommandResult decide(const std::string& approval_id, const std::string& code);

    RegistryOptions options_;
    mutable std::recursive_mutex mutex_;
    std::map<std::string, Entry, std::less<>> commands_;
    std::vector<std::string> order_;
    std::vector<Parked> pending_;
    CommandHistory history_;
    CommandJournal* journal_ = nullptr;
    ApprovalPolicy policy_;
    std::optional<std::thread::id> frame_thread_;
    std::uint64_t next_invocation_ = 1;
    std::uint64_t next_approval_ = 1;
    Error journal_error_;
};

}  // namespace caliper_builder
