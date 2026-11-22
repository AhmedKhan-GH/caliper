#include <caliper_builder/commands/registry.h>
#include <caliper_builder/commands/schema.h>

#include <algorithm>
#include <exception>
#include <tuple>
#include <utility>

namespace caliper_builder {

namespace {

using nlohmann::json;

// Set while a handler, an undo closure or a redo closure runs on this thread,
// so that a handler cannot invoke a command.
thread_local bool g_in_handler = false;

struct HandlerScope {
    bool previous = g_in_handler;
    HandlerScope() { g_in_handler = true; }
    ~HandlerScope() { g_in_handler = previous; }
};

const char* kStepSummarySchema = R"json({
  "type": "object", "additionalProperties": false,
  "required": ["id", "label", "count", "group", "origin", "at"],
  "properties": {
    "id": {"type": "integer", "description": "The step's number; each new step gets a higher one."},
    "label": {"type": "string", "description": "The name of the step's first command, such as panel.patch."},
    "count": {"type": "integer", "description": "How many commands the step holds, merged ones included."},
    "group": {"type": ["string", "null"], "description": "The group the step's commands were sent with, or null."},
    "origin": {"type": "string", "enum": ["ui", "builder", "agent", "mcp", "api"],
               "description": "Who sent the step's first command: the app's UI, the builder, the agent pane, an outside agent over MCP, or a program over Caliper API."},
    "at": {"type": "string", "description": "The UTC time of the step's newest command, in ISO 8601."}}})json";

json move_result_schema(const char* verb) {
    json s = json::parse(R"json({
      "type": "object", "additionalProperties": false, "required": ["steps", "can_undo", "can_redo"],
      "properties": {
        "steps": {"type": "array", "description": ""},
        "can_undo": {"type": "boolean", "description": "Whether a step is left to undo."},
        "can_redo": {"type": "boolean", "description": "Whether a step is left to redo."}}})json");
    s["properties"]["steps"]["description"] = std::string("One summary for each step ") + verb + ", in the order " +
                                              verb + ".";
    s["properties"]["steps"]["items"] = json::parse(kStepSummarySchema);
    return s;
}

CommandDescriptor make_history(const char* name, const char* summary, json args, json result, json example,
                               bool mutates) {
    CommandDescriptor d;
    d.agents = !mutates;  // undo and redo are the person's; the list is everyone's
    d.name = name;
    d.summary = summary;
    d.args = std::move(args);
    d.result = std::move(result);
    d.examples = {std::move(example)};
    d.mutates = mutates;
    d.destructive = false;
    d.frame_thread = true;
    // Moving the history changes the workspace; listing it reads.
    d.effect = mutates ? CommandEffect::change : CommandEffect::read;
    return d;
}

bool is_history(std::string_view name) {
    return name == "history.undo" || name == "history.redo" || name == "history.list";
}

ApprovalDecision stricter(ApprovalDecision a, ApprovalDecision b) {
    auto rank = [](ApprovalDecision d) { return d == ApprovalDecision::deny ? 2 : d == ApprovalDecision::ask ? 1 : 0; };
    return rank(a) >= rank(b) ? a : b;
}

std::size_t steps_arg(const json& args, std::size_t fallback) {
    if (args.contains("steps") && args["steps"].is_number()) {
        const double d = args["steps"].get<double>();
        if (d >= 1) return static_cast<std::size_t>(d);
    }
    return fallback;
}

}  // namespace

ApprovalDecision default_approval_policy(const CommandDescriptor& d, const CommandInvocation& inv) {
    if (inv.origin == CommandOrigin::ui || inv.origin == CommandOrigin::builder) return ApprovalDecision::allow;
    if (!d.destructive) return ApprovalDecision::allow;
    return ApprovalDecision::ask;
}

json pending_approval_to_json(const PendingApproval& p) {
    return json{{"approval", p.approval},
                {"invocation", command_invocation_to_json(p.invocation)},
                {"summary", p.summary},
                {"requested_at", p.requested_at},
                {"review", p.review}};
}

CommandOutput CommandOutput::ok(json result, UndoRecord undo) {
    CommandOutput out;
    out.result = std::move(result);
    out.undo = std::move(undo);
    return out;
}

CommandOutput CommandOutput::unchanged(json result) {
    CommandOutput out;
    out.result = std::move(result);
    out.no_effect = true;
    return out;
}

CommandOutput CommandOutput::fail(Error error) {
    CommandOutput out;
    out.error = std::move(error);
    if (out.error.code.empty()) out.error.code = "command.failed";
    return out;
}

std::vector<CommandDescriptor> history_command_descriptors() {
    const json steps_arg_schema = json::parse(R"json({
      "type": "object", "additionalProperties": false, "required": [],
      "properties": {"steps": {"type": "integer", "minimum": 1, "maximum": 100, "default": 1,
                               "description": "How many steps to move, from 1 to 100; 1 when absent."}}})json");
    std::vector<CommandDescriptor> out;
    out.push_back(make_history(
        "history.undo",
        "Undoes the most recent change to the workspace, or the given number of steps. A step is one command, "
        "several commands sent with the same group, or a run of commands merged by a merge key, such as a slider "
        "drag. Undo restores the open spec and the saved loader recipes; it does not undo saving an applet or "
        "running a loader. Undo is the person's tool: the app's UI, the builder and programs over Caliper API "
        "may undo, and an agent may not; an agent takes back a change with an edit. history.list shows the "
        "steps first.",
        steps_arg_schema, move_result_schema("undone"), json{{"steps", 1}}, true));
    out.push_back(make_history(
        "history.redo",
        "Redoes the step that was undone most recently, or the given number of steps. Any new change to the "
        "workspace clears what can be redone. Redo is the person's tool, as undo is: an agent may not redo.",
        steps_arg_schema, move_result_schema("redone"), json{{"steps", 1}}, true));
    json list_args = json::parse(R"json({
      "type": "object", "additionalProperties": false, "required": [],
      "properties": {"limit": {"type": "integer", "minimum": 1, "maximum": 200, "default": 50,
                               "description": "The most steps to list on each stack, from 1 to 200; 50 when absent."}}})json");
    json list_result = json::parse(R"json({
      "type": "object", "additionalProperties": false, "required": ["undo", "redo", "can_undo", "can_redo"],
      "properties": {
        "undo": {"type": "array", "description": "The steps that history.undo would undo, newest first."},
        "redo": {"type": "array", "description": "The steps that history.redo would redo, the next one first."},
        "can_undo": {"type": "boolean", "description": "Whether a step is left to undo."},
        "can_redo": {"type": "boolean", "description": "Whether a step is left to redo."}}})json");
    list_result["properties"]["undo"]["items"] = json::parse(kStepSummarySchema);
    list_result["properties"]["redo"]["items"] = json::parse(kStepSummarySchema);
    out.push_back(make_history(
        "history.list",
        "Lists the steps that can be undone and redone, newest first, with the command that began each step, how "
        "many commands it holds, its group, who sent it and when. It changes nothing.",
        list_args, list_result, json{{"limit", 20}}, false));
    return out;
}

CommandRegistry::CommandRegistry(RegistryOptions options) : options_(std::move(options)), history_(options_.history) {
    if (!options_.steady_now) options_.steady_now = [] { return std::chrono::steady_clock::now(); };
    if (!options_.wall_now) options_.wall_now = [] { return std::chrono::system_clock::now(); };
    if (options_.max_pending == 0) options_.max_pending = 1;
    for (auto& d : history_command_descriptors()) {
        const std::string name = d.name;
        commands_.emplace(name, Entry{std::move(d), [](const CommandContext&) { return CommandOutput(); }});
        order_.push_back(name);
    }
}

Error CommandRegistry::add(CommandDescriptor d, CommandHandler handler) {
    if (Error err = check_command_descriptor(d)) return err;
    std::lock_guard<std::recursive_mutex> lock(mutex_);
    if (commands_.count(d.name))
        return Error("command.bad_descriptor", "A command of this name is already in the registry.", d.name);
    const std::string name = d.name;
    commands_.emplace(name, Entry{std::move(d), std::move(handler)});
    order_.push_back(name);
    return {};
}

Error CommandRegistry::add_all(const std::vector<CommandDescriptor>& descriptors) {
    for (const auto& d : descriptors) {
        {
            std::lock_guard<std::recursive_mutex> lock(mutex_);
            if (commands_.count(d.name)) continue;
        }
        if (Error err = add(d)) return err;
    }
    return {};
}

Error CommandRegistry::bind(std::string_view name, CommandHandler handler) {
    std::lock_guard<std::recursive_mutex> lock(mutex_);
    const auto it = commands_.find(name);
    if (it == commands_.end()) return Error("command.unknown", "No command has this name.", std::string(name));
    if (is_history(name)) return Error("command.bad_descriptor", "The history commands are built in.", std::string(name));
    it->second.handler = std::move(handler);
    return {};
}

std::optional<CommandDescriptor> CommandRegistry::descriptor(std::string_view name) const {
    std::lock_guard<std::recursive_mutex> lock(mutex_);
    const auto it = commands_.find(name);
    if (it == commands_.end()) return std::nullopt;
    return it->second.descriptor;
}

std::vector<CommandDescriptor> CommandRegistry::descriptors() const {
    std::lock_guard<std::recursive_mutex> lock(mutex_);
    std::vector<CommandDescriptor> out;
    for (const auto& name : order_) out.push_back(commands_.find(name)->second.descriptor);
    return out;
}

bool CommandRegistry::is_bound(std::string_view name) const {
    std::lock_guard<std::recursive_mutex> lock(mutex_);
    const auto it = commands_.find(name);
    return it != commands_.end() && static_cast<bool>(it->second.handler);
}

void CommandRegistry::set_approval_policy(ApprovalPolicy policy) {
    std::lock_guard<std::recursive_mutex> lock(mutex_);
    policy_ = std::move(policy);
}

void CommandRegistry::set_approval_review(ApprovalReview review) {
    std::lock_guard<std::recursive_mutex> lock(mutex_);
    review_ = std::move(review);
}

void CommandRegistry::set_approval_check(ApprovalCheck check) {
    std::lock_guard<std::recursive_mutex> lock(mutex_);
    check_ = std::move(check);
}

void CommandRegistry::set_approval_notes(ApprovalNotes notes) {
    std::lock_guard<std::recursive_mutex> lock(mutex_);
    notes_ = std::move(notes);
}

void CommandRegistry::set_approval_observer(ApprovalObserver observer) {
    std::lock_guard<std::recursive_mutex> lock(mutex_);
    observer_ = std::move(observer);
}

void CommandRegistry::observe(const Parked& parked, std::string_view state) {
    if (!observer_) return;
    const auto it = commands_.find(parked.invocation.name);
    try {
        observer_(PendingApproval{parked.approval, parked.invocation,
                                  it != commands_.end() ? it->second.descriptor.summary : std::string(),
                                  parked.requested_at, parked.review},
                  state);
    } catch (...) {
        // An observer that throws loses its own news only.
    }
}

void CommandRegistry::set_journal(CommandJournal* journal) {
    std::lock_guard<std::recursive_mutex> lock(mutex_);
    journal_ = journal;
}

void CommandRegistry::set_frame_thread(std::optional<std::thread::id> id) {
    std::lock_guard<std::recursive_mutex> lock(mutex_);
    frame_thread_ = id;
}

Error CommandRegistry::journal_error() const {
    std::lock_guard<std::recursive_mutex> lock(mutex_);
    return journal_error_;
}

std::vector<PendingApproval> CommandRegistry::pending_approvals() const {
    std::lock_guard<std::recursive_mutex> lock(mutex_);
    std::vector<PendingApproval> out;
    for (const auto& p : pending_) {
        const auto it = commands_.find(p.invocation.name);
        out.push_back(PendingApproval{p.approval, p.invocation,
                                      it != commands_.end() ? it->second.descriptor.summary : std::string(),
                                      p.requested_at, p.review});
    }
    return out;
}

CommandRegistry::HistoryState CommandRegistry::history_state(std::size_t limit) const {
    std::lock_guard<std::recursive_mutex> lock(mutex_);
    HistoryState s;
    s.can_undo = history_.can_undo();
    s.can_redo = history_.can_redo();
    s.undo = history_.undo_summaries(limit);
    s.redo = history_.redo_summaries(limit);
    s.bytes = history_.bytes();
    return s;
}

std::string CommandRegistry::now_text() const { return command_utc_timestamp(options_.wall_now()); }

bool CommandRegistry::wrong_thread(const CommandDescriptor& d) const {
    return d.frame_thread && frame_thread_ && *frame_thread_ != std::this_thread::get_id();
}

ApprovalDecision CommandRegistry::ask_policy(const CommandDescriptor& d, const CommandInvocation& inv) {
    try {
        return policy_ ? policy_(d, inv) : default_approval_policy(d, inv);
    } catch (...) {
        return ApprovalDecision::deny;  // fail closed
    }
}

ApprovalNote CommandRegistry::ask_notes(const CommandDescriptor& d, const CommandInvocation& inv) {
    if (!notes_) return {};
    try {
        return notes_(d, inv);
    } catch (...) {
        return {};
    }
}

std::uint64_t CommandRegistry::journal_do(const CommandInvocation& inv, bool ok, const std::string& error,
                                          const std::string& allowed_by) {
    if (!journal_) return 0;
    JournalEvent e;
    e.allowed_by = allowed_by;
    e.ts = now_text();
    e.origin = inv.origin;
    e.name = inv.name;
    e.args = inv.args;
    e.ok = ok;
    e.group = inv.group;
    e.event = "do";
    e.error = error;
    const auto appended = journal_->append(std::move(e));
    if (appended.error) journal_error_ = appended.error;
    return appended.seq;
}

void CommandRegistry::journal_history(const CommandInvocation& inv, bool undo, const std::string& group,
                                      std::vector<std::uint64_t> seqs, bool ok, const std::string& error) {
    if (!journal_) return;
    JournalEvent e;
    e.ts = now_text();
    e.origin = inv.origin;
    e.name = inv.name;
    e.args = json::object();
    e.ok = ok;
    e.group = group;
    e.event = undo ? "undo" : "redo";
    e.error = error;
    e.seqs = std::move(seqs);
    const auto appended = journal_->append(std::move(e));
    if (appended.error) journal_error_ = appended.error;
}

CommandOutput CommandRegistry::call(const Entry& entry, const CommandInvocation& inv, bool redo) {
    CommandContext ctx{inv, entry.descriptor, redo};
    HandlerScope scope;
    try {
        CommandOutput out = entry.handler(ctx);
        if (out.error && out.error.code.empty()) out.error.code = "command.failed";
        if (out.result.is_null()) out.result = json::object();
        return out;
    } catch (const std::exception& e) {
        return CommandOutput::fail(
            Error("command.handler_failed", std::string("The handler of ") + inv.name + " failed: " + e.what(), inv.name));
    } catch (...) {
        return CommandOutput::fail(Error("command.handler_failed",
                                         std::string("The handler of ") + inv.name + " threw a non-standard exception.",
                                         inv.name));
    }
}

CommandResult CommandRegistry::invoke(CommandInvocation inv) {
    if (g_in_handler)
        return CommandResult::failure(inv.id, Error("command.reentrant",
                                                    "A handler cannot invoke a command; it calls library functions.",
                                                    inv.name));
    std::lock_guard<std::recursive_mutex> lock(mutex_);
    if (inv.id.empty()) inv.id = "c-" + std::to_string(next_invocation_++);
    if (inv.args.is_null()) inv.args = json::object();  // no arguments, as when the field is absent
    if (Error err = check_command_invocation(inv)) return CommandResult::failure(inv.id, err);
    const auto it = commands_.find(inv.name);
    if (it == commands_.end())
        return CommandResult::failure(inv.id, Error("command.unknown", "No command is named " + inv.name + ".", "name"));
    const Entry& entry = it->second;
    // A null on an optional argument is the argument left out, so the policy,
    // the review, the journal and the handler see the call as it is understood.
    drop_null_optional_args(entry.descriptor.args, &inv.args);
    // The arguments first: a caller learns that its call is malformed even
    // from an app that has no handler for the command yet.
    if (Error err = validate_command_value(entry.descriptor.args, inv.args, "args")) {
        err.code = "command.invalid_args";
        return CommandResult::failure(inv.id, err);
    }
    // The browser's spec page reads; a command that changes the workspace or
    // acts on the running script (the effect run, which does not mutate) is
    // refused before any policy sees it, so it is never journaled.
    if (inv.origin == CommandOrigin::viewer && (entry.descriptor.mutates || entry.descriptor.effect == CommandEffect::run))
        return CommandResult::failure(
            inv.id, Error("command.not_allowed",
                          inv.name + (entry.descriptor.mutates ? " changes the workspace" : " acts on the running script") +
                              ", and the viewer origin only reads.",
                          "origin"));
    // The person's own commands (history.undo and history.redo) are refused
    // to agents before any policy or history work, so they are never journaled.
    if (!entry.descriptor.agents && (inv.origin == CommandOrigin::agent || inv.origin == CommandOrigin::mcp))
        return CommandResult::failure(
            inv.id, Error("command.not_allowed",
                          inv.name + " is the person's own tool, so an agent cannot run it; to take back a change, "
                                     "send an edit that reverts it, which the person reviews.",
                          "origin"));
    if (!entry.handler)
        return CommandResult::failure(
            inv.id, Error("command.not_bound", inv.name + " is in the command table, but this app has no handler for it yet.",
                          "name"));
    if (wrong_thread(entry.descriptor))
        return CommandResult::failure(
            inv.id, Error("command.wrong_thread", inv.name + " runs only on the app's frame thread.", "name"));

    ApprovalDecision decision;
    std::vector<std::uint64_t> steps;
    ApprovalNote note;
    if (inv.name == "history.undo" || inv.name == "history.redo") {
        std::tie(decision, steps) = history_decision(inv, inv.name == "history.undo");
    } else {
        decision = ask_policy(entry.descriptor, inv);
        note = ask_notes(entry.descriptor, inv);
    }
    if (decision == ApprovalDecision::deny)
        return finish_refused(inv, entry, "command.denied",
                              note.message.empty() ? inv.name + " was denied by the approval policy." : note.message);
    if (decision == ApprovalDecision::ask) return park(inv, entry, std::move(steps));
    return run(inv, entry, steps, note.allowed_by);
}

CommandResult CommandRegistry::park(const CommandInvocation& inv, const Entry& entry, std::vector<std::uint64_t> steps) {
    if (pending_.size() >= options_.max_pending)
        return CommandResult::failure(
            inv.id, Error("command.too_many_pending",
                          "Too many invocations already wait for the person's approval; try again later.", ""));
    Parked parked;
    parked.approval = "ap-" + std::to_string(next_approval_++);
    parked.invocation = inv;
    parked.requested_at = now_text();
    parked.steps = std::move(steps);
    if (review_) {
        try {
            parked.review = review_(entry.descriptor, inv);
        } catch (...) {
            parked.review = nullptr;  // the card shows no review rather than nothing at all
        }
    }
    const std::string why =
        parked.review.is_null()
            ? " needs the person's approval in Caliper, because it deletes or overwrites saved data or runs a Python "
              "loader. It runs once they approve it."
        : inv.name == "applet.save"
            ? " overwrites a saved applet, and the person reviews the content it writes in Caliper first. It runs "
              "once they approve it."
            : " changes the open spec, and the person reviews such changes in Caliper before they apply. It runs "
              "once they approve it.";
    CommandResult r = CommandResult::failure(inv.id, Error("command.needs_approval", inv.name + why, ""));
    r.approval = parked.approval;
    observe(parked, "pending");
    pending_.push_back(std::move(parked));
    return r;
}

CommandResult CommandRegistry::finish_refused(const CommandInvocation& inv, const Entry& entry, const std::string& code,
                                              std::string message) {
    if (inv.name == "history.undo" || inv.name == "history.redo")
        journal_history(inv, inv.name == "history.undo", inv.group, {}, false, code);
    else if (entry.descriptor.mutates || (journal_ && journal_->options().include_read_only))
        journal_do(inv, false, code);
    return CommandResult::failure(inv.id, Error(code, std::move(message), ""));
}

CommandResult CommandRegistry::approve(const std::string& approval_id, bool reissue) {
    std::lock_guard<std::recursive_mutex> lock(mutex_);
    const auto it = std::find_if(pending_.begin(), pending_.end(),
                                 [&](const Parked& p) { return p.approval == approval_id; });
    if (it == pending_.end())
        return CommandResult::failure(
            "", Error("command.unknown_approval", "No invocation waits for approval under this id.", approval_id));
    const auto cmd = commands_.find(it->invocation.name);
    if (cmd == commands_.end()) {
        pending_.erase(it);
        return CommandResult::failure("", Error("command.unknown", "The command is gone.", approval_id));
    }
    if (wrong_thread(cmd->second.descriptor))
        return CommandResult::failure(it->invocation.id,
                                      Error("command.wrong_thread",
                                            it->invocation.name + " runs only on the app's frame thread.", "name"));
    Parked parked = std::move(*it);
    pending_.erase(it);
    observe(parked, "approved");
    bool holds = true;
    if (check_) {
        try {
            holds = check_(cmd->second.descriptor, parked.invocation);
        } catch (...) {
            holds = false;
        }
    }
    if (!holds) {
        // What the person approved changed while the card waited.
        if (!reissue)
            return finish_refused(parked.invocation, cmd->second, "history.conflict",
                                  parked.invocation.name +
                                      " was approved for content that changed while the person decided, so it did "
                                      "not run; send it again to have the new content approved.");
        const ApprovalDecision again = ask_policy(cmd->second.descriptor, parked.invocation);
        if (again == ApprovalDecision::deny)
            return finish_refused(parked.invocation, cmd->second, "command.denied",
                                  parked.invocation.name + " was denied by the approval policy.");
        if (again == ApprovalDecision::ask) return park(parked.invocation, cmd->second, parked.steps);
    }
    return run(parked.invocation, cmd->second, parked.steps);
}

CommandResult CommandRegistry::deny(const std::string& approval_id) { return decide(approval_id, "command.denied"); }

CommandResult CommandRegistry::expire_approval(const std::string& approval_id) {
    return decide(approval_id, "command.approval_timeout");
}

CommandResult CommandRegistry::decide(const std::string& approval_id, const std::string& code) {
    std::lock_guard<std::recursive_mutex> lock(mutex_);
    const auto it = std::find_if(pending_.begin(), pending_.end(),
                                 [&](const Parked& p) { return p.approval == approval_id; });
    if (it == pending_.end())
        return CommandResult::failure(
            "", Error("command.unknown_approval", "No invocation waits for approval under this id.", approval_id));
    Parked parked = std::move(*it);
    pending_.erase(it);
    observe(parked, code == "command.denied" ? "denied" : "expired");
    const std::string message = code == "command.denied"
                                    ? "The person denied " + parked.invocation.name + "."
                                    : "Nobody approved " + parked.invocation.name + " in time, so it was withdrawn.";
    const auto cmd = commands_.find(parked.invocation.name);
    if (cmd == commands_.end()) return CommandResult::failure(parked.invocation.id, Error(code, message, ""));
    return finish_refused(parked.invocation, cmd->second, code, message);
}

bool CommandRegistry::cancel_approval(const std::string& approval_id) {
    std::lock_guard<std::recursive_mutex> lock(mutex_);
    const auto it = std::find_if(pending_.begin(), pending_.end(),
                                 [&](const Parked& p) { return p.approval == approval_id; });
    if (it == pending_.end()) return false;
    pending_.erase(it);
    return true;
}

std::size_t CommandRegistry::cancel_pending() {
    std::lock_guard<std::recursive_mutex> lock(mutex_);
    const std::size_t n = pending_.size();
    pending_.clear();
    return n;
}

CommandResult CommandRegistry::run(const CommandInvocation& inv, const Entry& entry,
                                   const std::vector<std::uint64_t>& steps, const std::string& allowed_by) {
    CommandResult result;
    if (inv.name == "history.undo" || inv.name == "history.redo") {
        result = run_history_move(inv, inv.name == "history.undo", steps);
    } else if (inv.name == "history.list") {
        result = run_history_list(inv);
    } else {
        CommandOutput out = call(entry, inv, false);
        if (!out.error && !out.follow_ups.empty()) return run_follow_ups(inv, entry, out);
        // An outcome that changed nothing is a read, even of a mutating command.
        const bool changed = entry.descriptor.mutates && !out.no_effect;
        const bool journal_this = changed || (journal_ && journal_->options().include_read_only);
        if (out.error) {
            if (journal_this) journal_do(inv, false, out.error.code, allowed_by);
            return CommandResult::failure(inv.id, out.error);
        }
        const std::uint64_t seq = journal_this ? journal_do(inv, true, {}, allowed_by) : 0;
        if (changed) {
            if (out.undo.kind == UndoRecord::Kind::none) {
                history_.pass_by();
            } else {
                HistoryRecord record;
                record.invocation = inv;
                record.destructive = entry.descriptor.destructive;
                if (out.undo.kind == UndoRecord::Kind::inverse) {
                    const auto inverse = commands_.find(out.undo.inverse_name);
                    record.inverse_destructive = inverse != commands_.end() && inverse->second.descriptor.destructive;
                }
                record.undo = std::move(out.undo);
                if (seq) record.seqs.push_back(seq);
                record.last = options_.steady_now();
                history_.record(std::move(record), now_text());
            }
        }
        result = CommandResult::success(inv.id, std::move(out.result));
    }
    if (result.ok && options_.check_results)
        if (Error err = validate_command_value(entry.descriptor.result, result.result, "result")) {
            err.code = "command.invalid_result";
            return CommandResult::failure(inv.id, err);
        }
    return result;
}

CommandResult CommandRegistry::run_follow_ups(const CommandInvocation& inv, const Entry& entry, CommandOutput& out) {
    const bool journal_parent = entry.descriptor.mutates || (journal_ && journal_->options().include_read_only);
    auto fail = [&](Error e) {
        if (journal_parent) journal_do(inv, false, e.code);
        return CommandResult::failure(inv.id, std::move(e));
    };
    auto bad = [&](std::size_t k, const std::string& why) {
        return fail(Error("command.handler_failed",
                          "The handler of " + inv.name + " returned a follow-up that cannot run: " + why,
                          "follow_ups[" + std::to_string(k) + "]"));
    };
    // Every follow-up is checked before any runs.
    std::vector<CommandInvocation>& subs = out.follow_ups;
    std::vector<const Entry*> entries;
    const std::string group = subs.front().group;
    if (group.empty()) return bad(0, "the follow-ups share one non-empty group.");
    for (std::size_t k = 0; k < subs.size(); ++k) {
        CommandInvocation& sub = subs[k];
        sub.id = inv.id.substr(0, 112) + "." + std::to_string(k + 1);
        if (sub.args.is_null()) sub.args = json::object();
        if (sub.group != group) return bad(k, "the follow-ups share one non-empty group.");
        if (Error err = check_command_invocation(sub)) return bad(k, err.message);
        const auto it = commands_.find(sub.name);
        if (it == commands_.end() || !it->second.handler) return bad(k, sub.name + " is not a bound command.");
        const CommandDescriptor& d = it->second.descriptor;
        drop_null_optional_args(d.args, &sub.args);
        if (!d.mutates || d.destructive || is_history(sub.name))
            return bad(k, sub.name + " does not mutate, is destructive or is a history command.");
        if (Error err = validate_command_value(d.args, sub.args, "args")) return bad(k, err.where + ": " + err.message);
        if (wrong_thread(d)) return bad(k, sub.name + " runs only on the app's frame thread.");
        entries.push_back(&it->second);
    }
    std::vector<CommandOutput> outs;
    for (std::size_t k = 0; k < subs.size(); ++k) {
        CommandOutput o = call(*entries[k], subs[k], false);
        if (o.error) {
            // Undo the follow-ups that ran, newest first, so that a failed
            // invocation changes nothing.
            for (std::size_t j = outs.size(); j-- > 0;) {
                if (outs[j].no_effect || outs[j].undo.kind == UndoRecord::Kind::none) continue;
                HistoryRecord r;
                r.invocation = subs[j];
                r.undo = std::move(outs[j].undo);
                if (Error again = undo_record(r, inv)) {
                    history_.clear();
                    return fail(Error("history.inconsistent",
                                      subs[k].name + " failed (" + o.error.message + "), and undoing " + subs[j].name +
                                          " to roll it back failed too (" + again.message +
                                          "). The history was cleared; the workspace may be partly changed.",
                                      subs[k].name));
                }
            }
            Error e = o.error;
            e.message = subs[k].name + " failed: " + e.message;
            return fail(std::move(e));
        }
        outs.push_back(std::move(o));
    }
    if (journal_parent) journal_do(inv, true, {});
    history_.seal();
    for (std::size_t k = 0; k < subs.size(); ++k) {
        CommandOutput& o = outs[k];
        if (o.no_effect) continue;  // a follow-up that changed nothing is a read
        const std::uint64_t seq = journal_do(subs[k], true, {});
        if (o.undo.kind == UndoRecord::Kind::none) {
            history_.pass_by();
            continue;
        }
        HistoryRecord record;
        record.invocation = subs[k];
        record.destructive = false;
        if (o.undo.kind == UndoRecord::Kind::inverse) {
            const auto inverse = commands_.find(o.undo.inverse_name);
            record.inverse_destructive = inverse != commands_.end() && inverse->second.descriptor.destructive;
        }
        record.undo = std::move(o.undo);
        if (seq) record.seqs.push_back(seq);
        record.last = options_.steady_now();
        history_.record(std::move(record), now_text());
    }
    CommandResult result = CommandResult::success(inv.id, std::move(out.result));
    if (options_.check_results)
        if (Error err = validate_command_value(entry.descriptor.result, result.result, "result")) {
            err.code = "command.invalid_result";
            return CommandResult::failure(inv.id, err);
        }
    return result;
}

std::pair<ApprovalDecision, std::vector<std::uint64_t>> CommandRegistry::history_decision(const CommandInvocation& inv,
                                                                                          bool undo) {
    const auto& stack = undo ? history_.undo_stack() : history_.redo_stack();
    const std::size_t n = std::min(steps_arg(inv.args, 1), stack.size());
    ApprovalDecision decision = ApprovalDecision::allow;
    std::vector<std::uint64_t> ids;
    for (std::size_t i = 0; i < n; ++i) {
        const HistoryStep& step = stack[stack.size() - 1 - i];
        ids.push_back(step.id);
        for (const auto& record : step.records) {
            // The command that would really run: for an undo by inverse, the
            // inverse; for a closure undo or any redo, the command itself.
            std::string name = record.invocation.name;
            bool destructive = record.destructive;
            if (undo && record.undo.kind == UndoRecord::Kind::inverse) {
                name = record.undo.inverse_name;
                destructive = record.inverse_destructive;
            }
            if (!destructive) continue;
            const auto it = commands_.find(name);
            if (it == commands_.end()) continue;
            CommandInvocation as = inv;
            as.name = name;
            as.args = undo && record.undo.kind == UndoRecord::Kind::inverse ? record.undo.inverse_args
                                                                              : record.invocation.args;
            decision = stricter(decision, ask_policy(it->second.descriptor, as));
        }
    }
    return {decision, ids};
}

Error CommandRegistry::undo_record(HistoryRecord& record, const CommandInvocation& by) {
    if (record.undo.kind == UndoRecord::Kind::closures) {
        if (!record.undo.undo) return Error("history.undo_failed", "The record has no undo.", record.invocation.name);
        HandlerScope scope;
        try {
            return record.undo.undo();
        } catch (const std::exception& e) {
            return Error("command.handler_failed", e.what(), record.invocation.name);
        } catch (...) {
            return Error("command.handler_failed", "The undo threw a non-standard exception.", record.invocation.name);
        }
    }
    const auto it = commands_.find(record.undo.inverse_name);
    if (it == commands_.end() || !it->second.handler)
        return Error("history.undo_failed", "The inverse command " + record.undo.inverse_name + " is not available.",
                     record.invocation.name);
    CommandInvocation inverse;
    inverse.id = by.id + ".undo";
    inverse.name = record.undo.inverse_name;
    inverse.args = record.undo.inverse_args;
    inverse.origin = by.origin;
    inverse.group = record.invocation.group;
    if (Error err = validate_command_value(it->second.descriptor.args, inverse.args, "args")) return err;
    return call(it->second, inverse, false).error;
}

Error CommandRegistry::redo_record(HistoryRecord& record, const CommandInvocation& by) {
    if (record.undo.kind == UndoRecord::Kind::closures) {
        if (!record.undo.redo) return Error("history.redo_failed", "The record has no redo.", record.invocation.name);
        HandlerScope scope;
        try {
            return record.undo.redo();
        } catch (const std::exception& e) {
            return Error("command.handler_failed", e.what(), record.invocation.name);
        } catch (...) {
            return Error("command.handler_failed", "The redo threw a non-standard exception.", record.invocation.name);
        }
    }
    // An inverse record redoes by running its command again; the inverse kept
    // is the one from the first time it ran.
    (void)by;
    const auto it = commands_.find(record.invocation.name);
    if (it == commands_.end() || !it->second.handler)
        return Error("history.redo_failed", "The command " + record.invocation.name + " is not available.",
                     record.invocation.name);
    return call(it->second, record.invocation, true).error;
}

Error CommandRegistry::undo_step(HistoryStep& step, const CommandInvocation& by) {
    for (std::size_t k = step.records.size(); k-- > 0;) {
        if (Error err = undo_record(step.records[k], by)) {
            // Roll back: redo the records already undone, oldest first.
            for (std::size_t j = k + 1; j < step.records.size(); ++j)
                if (Error again = redo_record(step.records[j], by))
                    return Error("history.inconsistent",
                                 "The undo of " + step.records[k].invocation.name + " failed (" + err.message +
                                     "), and redoing " + step.records[j].invocation.name +
                                     " to roll it back failed too (" + again.message +
                                     "). The history was cleared; the workspace may be partly changed.",
                                 step.records[k].invocation.name);
            return Error("history.undo_failed",
                         "The undo of " + step.records[k].invocation.name + " failed: " + err.message,
                         step.records[k].invocation.name);
        }
    }
    return {};
}

Error CommandRegistry::redo_step(HistoryStep& step, const CommandInvocation& by) {
    for (std::size_t k = 0; k < step.records.size(); ++k) {
        if (Error err = redo_record(step.records[k], by)) {
            for (std::size_t j = k; j-- > 0;)
                if (Error again = undo_record(step.records[j], by))
                    return Error("history.inconsistent",
                                 "The redo of " + step.records[k].invocation.name + " failed (" + err.message +
                                     "), and undoing " + step.records[j].invocation.name +
                                     " to roll it back failed too (" + again.message +
                                     "). The history was cleared; the workspace may be partly changed.",
                                 step.records[k].invocation.name);
            return Error("history.redo_failed",
                         "The redo of " + step.records[k].invocation.name + " failed: " + err.message,
                         step.records[k].invocation.name);
        }
    }
    return {};
}

CommandResult CommandRegistry::run_history_move(const CommandInvocation& inv, bool undo,
                                                const std::vector<std::uint64_t>& expected) {
    const auto& stack = undo ? history_.undo_stack() : history_.redo_stack();
    // An approved history command checks that the steps it was approved for
    // are still the ones on top.
    for (std::size_t i = 0; i < expected.size(); ++i)
        if (i >= stack.size() || stack[stack.size() - 1 - i].id != expected[i])
            return CommandResult::failure(
                inv.id, Error("history.changed",
                              "The history changed while the request waited for approval; list it and ask again.", ""));

    const std::size_t want = steps_arg(inv.args, 1);
    json summaries = json::array();
    std::size_t done = 0;
    for (; done < want; ++done) {
        HistoryStep* step = undo ? history_.top_undo() : history_.top_redo();
        if (!step) {
            if (done == 0)
                return CommandResult::failure(
                    inv.id, Error(undo ? "history.nothing_to_undo" : "history.nothing_to_redo",
                                  undo ? "There is nothing to undo." : "There is nothing to redo.", ""));
            break;
        }
        Error err = undo ? undo_step(*step, inv) : redo_step(*step, inv);
        const std::string group = step->group;
        if (err) {
            if (err.code == "history.inconsistent") {
                history_.clear();
            } else if (undo) {
                history_.clear_undo();
                err.message += " The step and every older step were dropped from the history.";
            } else {
                history_.clear_redo();
                err.message += " The step and every step after it were dropped from the history.";
            }
            if (done > 0)
                err.message += " " + std::to_string(done) + (done == 1 ? " step was " : " steps were ") +
                               (undo ? "undone" : "redone") + " before it.";
            journal_history(inv, undo, group, {}, false, err.code);
            return CommandResult::failure(inv.id, err);
        }
        summaries.push_back(history_step_summary_to_json(CommandHistory::summarize(*step)));
        std::vector<std::uint64_t> seqs = step->seqs();
        if (undo)
            history_.undone();
        else
            history_.redone();
        journal_history(inv, undo, group, std::move(seqs), true, {});
    }
    return CommandResult::success(
        inv.id, json{{"steps", summaries}, {"can_undo", history_.can_undo()}, {"can_redo", history_.can_redo()}});
}

CommandResult CommandRegistry::run_history_list(const CommandInvocation& inv) {
    std::size_t limit = 50;
    if (inv.args.contains("limit") && inv.args["limit"].is_number())
        limit = static_cast<std::size_t>(inv.args["limit"].get<double>());
    json undo = json::array();
    json redo = json::array();
    for (const auto& s : history_.undo_summaries(limit)) undo.push_back(history_step_summary_to_json(s));
    for (const auto& s : history_.redo_summaries(limit)) redo.push_back(history_step_summary_to_json(s));
    return CommandResult::success(inv.id, json{{"undo", undo},
                                               {"redo", redo},
                                               {"can_undo", history_.can_undo()},
                                               {"can_redo", history_.can_redo()}});
}

}  // namespace caliper_builder
