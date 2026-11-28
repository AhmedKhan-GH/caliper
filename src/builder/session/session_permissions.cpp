// session_permissions.cpp: content-bound approvals, Python/run approval gates,
// approval policy, and agent permission persistence (session_internal.h).
#include "session_internal.h"

#include <algorithm>

#ifndef _WIN32
#include <sys/stat.h>
#endif

namespace caliper_builder {

namespace {

namespace fs = std::filesystem;
using nlohmann::json;
using namespace session_detail;
using namespace session_internal;

}  // namespace

// ---- approvals bound to content --------------------------------------------

bool Session::Impl::replaces_document(const std::string& name) {
    return name == "spec.new" || name == "spec.open" || name == "instrument.open";
}

Session::Impl::Content Session::Impl::content_of(const CommandInvocation& inv) {
    Content out;
    std::lock_guard<std::mutex> lock(doc_mutex);
    if (!doc) return out;
    std::string before;
    std::string after;
    std::string before_label;
    std::string after_label;
    std::vector<spec::Diagnostic> diagnostics;
    if (inv.name == "instrument.save") {
        // The saved file against the open spec it would be overwritten with.
        const std::string id = str_arg(inv.args, "id", doc->spec.instrument.id);
        if (!valid_id(id)) return out;
        spec::Spec saved = doc->spec;
        saved.instrument.id = id;
        after = spec::to_toml(saved);
        const fs::path file = ws() / "instruments" / fs::u8path(id) / fs::u8path(id + ".spec.toml");
        if (read_file_bounded(file.u8string(), spec::limits::kTextBytes, &before, "spec.too_large")) before.clear();
        before_label = after_label = "instruments/" + id + "/" + id + ".spec.toml";
        diagnostics = spec::validate(saved, resolver()).diagnostics;
    } else if (replaces_document(inv.name)) {
        // The open spec with unsaved changes against the one that would replace it.
        if (!doc->dirty()) return out;
        spec::Spec next;
        if (inv.name == "spec.new") {
            next = new_spec_from_args(inv.args);
            diagnostics = spec::validate(next, resolver()).diagnostics;
            if (first_spec_error(diagnostics)) return out;
        } else {
            LoadedSpec loaded = inv.name == "spec.open" ? load_spec_file(str_arg(inv.args, "path"), resolver())
                                                        : load_instrument(str_arg(inv.args, "id"), resolver());
            if (loaded.error) return out;
            next = std::move(loaded.spec);
            diagnostics = std::move(loaded.diagnostics);
        }
        before = spec::to_toml(doc->spec);
        after = spec::to_toml(next);
        before_label = spec_label();
        after_label = next.instrument.id + ".spec.toml";
    } else {
        return out;
    }
    out.applies = true;
    out.doc = doc->id;
    out.digest = command_sha256_hex(before + std::string(1, '\0') + after);
    out.review = {{"diff", unified_line_diff(before, after, "a/" + before_label, "b/" + after_label)},
                  {"warnings", warnings_json(diagnostics)}};
    return out;
}

// Binds the invocation's approval to the content its card shows, and
// keeps the review for the registry; null when it does not apply.
json Session::Impl::bind_content(const CommandInvocation& inv) {
    Content c = content_of(inv);
    if (!c.applies) return nullptr;
    const std::string key = invocation_key(inv);
    std::lock_guard<std::mutex> lock(review_mutex);
    bound.push_back(Bound{key, c.doc, c.digest});
    while (bound.size() > 256) bound.pop_front();
    last_review = {key, c.review};
    return c.review;
}

// The approval check: an approval bound to content holds only while the
// document and what it would write or open are what the card showed.
bool Session::Impl::approval_holds(const CommandInvocation& inv) {
    const std::string key = invocation_key(inv);
    std::optional<Bound> b;
    {
        std::lock_guard<std::mutex> lock(review_mutex);
        const auto it = std::find_if(bound.begin(), bound.end(), [&](const Bound& x) { return x.key == key; });
        if (it == bound.end()) return true;
        b = *it;
        bound.erase(it);
    }
    const Content now = content_of(inv);
    return now.applies && now.doc == b->doc && now.digest == b->digest;
}

// ---- Python and managed-run approval tracking ------------------------------

std::string Session::Impl::invocation_key(const CommandInvocation& inv) {
    return inv.id + "\n" + inv.name + "\n" + inv.args.dump(-1, ' ', false, json::error_handler_t::replace);
}

bool Session::Impl::is_python(const Loader& l) {
    const auto r = declared_reader(l);
    return r && *r == ReaderKind::python;
}

// Whether a command starts a managed run, which executes Python code
// (design/hosted-runs.md, Review Card).
bool Session::Impl::starts_run(const std::string& name) { return name == "run.start" || name == "run.fork"; }

// Whether an invocation runs a Python loader's script: loader.run of one,
// or loader.create or loader.update of one with run set.
bool Session::Impl::runs_python(const CommandInvocation& inv) {
    if (inv.name == "loader.run") {
        LoaderStore::LoaderGet got = store->get(str_arg(inv.args, "id"));
        return !got.error && is_python(got.loader);
    }
    if ((inv.name == "loader.create" || inv.name == "loader.update") && bool_arg(inv.args, "run", false)) {
        LoaderParse p = parse_loader(str_arg(inv.args, "text"));
        return !p.error && is_python(p.loader);
    }
    return false;
}

// The permission a handler gives a run: the person's own click from ui and
// builder; for another origin, only an invocation the policy asked about,
// which reaches its handler only once the person approved it.
bool Session::Impl::python_approved(const CommandInvocation& inv) {
    if (user_origin(inv.origin)) return true;
    const std::string key = invocation_key(inv);
    std::lock_guard<std::mutex> lock(python_mutex);
    const auto it = std::find(python_asked.begin(), python_asked.end(), key);
    if (it == python_asked.end()) return false;
    python_asked.erase(it);
    return true;
}

// Whether the policy asked about this start or fork, which reaches its
// handler only once the person approved it; the key is used up.
bool Session::Impl::run_approved(const CommandInvocation& inv) {
    const std::string key = invocation_key(inv);
    std::lock_guard<std::mutex> lock(python_mutex);
    const auto it = std::find(python_asked.begin(), python_asked.end(), key);
    if (it == python_asked.end()) return false;
    python_asked.erase(it);
    return true;
}

// ---- agent permissions -----------------------------------------------------

bool Session::Impl::is_bound(const std::string& key) {
    std::lock_guard<std::mutex> lock(review_mutex);
    return std::any_of(bound.begin(), bound.end(), [&](const Bound& b) { return b.key == key; });
}

bool Session::Impl::python_pending(const std::string& key) {
    std::lock_guard<std::mutex> lock(python_mutex);
    return std::find(python_asked.begin(), python_asked.end(), key) != python_asked.end();
}

// The saved rules' agent of a principal: "mcp", or the chat's agent id.
std::string Session::Impl::rule_agent(AgentPrincipal who) const {
    if (who == AgentPrincipal::mcp) return "mcp";
    if (who == AgentPrincipal::chat && chat_agent != "mcp" && valid_rule_agent(chat_agent)) return chat_agent;
    return {};
}

AgentMode Session::Impl::mode_of(AgentPrincipal who) const {
    if (who == AgentPrincipal::chat) return perms.chat;
    if (who == AgentPrincipal::mcp) return perms.mcp;
    return AgentMode::ask;
}

Session::Impl::FileStamp Session::Impl::file_stamp() const {
    FileStamp out;
    if (options.permissions_file.empty()) return out;
#ifndef _WIN32
    struct stat st {};
    if (::stat(fs::u8path(options.permissions_file).c_str(), &st) != 0) return out;
    out.exists = true;
    out.size = static_cast<long long>(st.st_size);
    out.dev = static_cast<unsigned long long>(st.st_dev);
    out.ino = static_cast<unsigned long long>(st.st_ino);
#ifdef __APPLE__
    out.mtime_ns = static_cast<long long>(st.st_mtimespec.tv_sec) * 1000000000LL + st.st_mtimespec.tv_nsec;
#else
    out.mtime_ns = static_cast<long long>(st.st_mtim.tv_sec) * 1000000000LL + st.st_mtim.tv_nsec;
#endif
#else
    std::error_code ec;
    const fs::path p = fs::u8path(options.permissions_file);
    out.exists = fs::exists(p, ec);
    if (out.exists) {
        out.size = static_cast<long long>(fs::file_size(p, ec));
        out.mtime_ns = static_cast<long long>(fs::last_write_time(p, ec).time_since_epoch().count());
    }
#endif
    return out;
}

int Session::Impl::strictness(AgentMode m) {
    switch (m) {
        case AgentMode::read_only: return 0;
        case AgentMode::ask: return 1;
        case AgentMode::accept_edits: return 2;
        case AgentMode::allow_all: return 3;
    }
    return 1;
}

AgentMode Session::Impl::stricter(AgentMode a, AgentMode b) { return strictness(a) <= strictness(b) ? a : b; }

// Before a decision that Allow all would make without a card: when the
// file changed since the last read or write and no longer holds Allow all
// for the principal, its mode is lowered to the file's at once. This only
// ever lowers. Under perm_mutex.
void Session::Impl::recheck_allow_all(AgentPrincipal who) {
    if (options.permissions_file.empty() || mode_of(who) != AgentMode::allow_all) return;
    const FileStamp now = file_stamp();
    if (now == stamp) return;
    const AgentPermissionsRead r = read_agent_permissions(options.permissions_file);
    stamp = now;
    const AgentMode in_file = who == AgentPrincipal::chat ? r.file.chat : r.file.mcp;
    if (in_file == AgentMode::allow_all) return;
    (who == AgentPrincipal::chat ? perms.chat : perms.mcp) = in_file;
    (who == AgentPrincipal::chat ? saved.chat : saved.mcp) = in_file;
    (who == AgentPrincipal::chat ? pending_chat : pending_mcp) = false;
}

Error Session::Impl::failed_save(Error e) {
    save_error = e.message;
    return e;
}

// Saves the permissions, under a lock between apps. The file is read
// again first, and what the person chose in this app (the pending modes
// and rules) is always written into it; the rest stays as the file holds
// it, so a second app's change stays. A re-read only lowers: afterwards
// a mode in memory takes the file's mode only where that is stricter,
// and a rule the file no longer holds leaves memory, but nothing the
// file adds enters it. A file that cannot be read fails the save; one
// that reads with a problem is replaced with what the session holds.
Error Session::Impl::save_perms() {
    if (options.permissions_file.empty()) {
        saved = perms;
        pending_chat = pending_mcp = false;
        pending_added.clear();
        pending_revoked.clear();
        return {};
    }
    const AgentPermissionsLock lock(options.permissions_file);
    if (lock.error()) return failed_save(lock.error());
    const AgentPermissionsRead now = read_agent_permissions(options.permissions_file);
    if (!now.readable)
        return failed_save(Error("permissions.read_failed",
                                 "The agent permissions could not be saved: their file cannot be read.",
                                 options.permissions_file));
    AgentPermissionsFile merged = now.problems.empty() ? now.file : perms;
    if (pending_chat) merged.chat = perms.chat;
    if (pending_mcp) merged.mcp = perms.mcp;
    auto has = [](const std::vector<AgentRule>& v, const AgentRule& r) {
        return std::find(v.begin(), v.end(), r) != v.end();
    };
    for (const AgentRule& r : pending_added)
        if (!has(merged.rules, r)) merged.rules.push_back(r);
    for (const AgentRule& r : pending_revoked)
        merged.rules.erase(std::remove(merged.rules.begin(), merged.rules.end(), r), merged.rules.end());
    if (Error e = write_agent_permissions(options.permissions_file, merged)) return failed_save(e);
    stamp = file_stamp();
    perms.chat = stricter(perms.chat, merged.chat);
    perms.mcp = stricter(perms.mcp, merged.mcp);
    perms.rules.erase(std::remove_if(perms.rules.begin(), perms.rules.end(),
                                     [&](const AgentRule& r) { return !has(merged.rules, r); }),
                      perms.rules.end());
    saved = std::move(merged);
    pending_chat = pending_mcp = false;
    pending_added.clear();
    pending_revoked.clear();
    save_error.clear();
    return {};
}

// The session's policy with the agent's mode and rules applied, and the
// note the registry asks for next.
ApprovalDecision Session::Impl::policy(const CommandDescriptor& d, const CommandInvocation& inv) {
    const ApprovalDecision base = base_policy(d, inv);
    const AgentPrincipal who = agent_principal(inv);
    if (who == AgentPrincipal::none) return base;
    const std::string key = invocation_key(inv);
    PermissionContext c;
    // A data-loss card (bound to content) asks in every mode. A start or
    // a fork of a managed run runs Python code, as a Python loader does:
    // no rule covers it, and only its card or Allow all lets it run.
    c.bound = base == ApprovalDecision::ask && is_bound(key);
    c.python = base == ApprovalDecision::ask && (python_pending(key) || starts_run(inv.name));
    std::lock_guard<std::mutex> lock(perm_mutex);
    if (base == ApprovalDecision::ask) recheck_allow_all(who);
    c.mode = mode_of(who);
    const std::string rule = permission_rule_key(inv);
    c.session_rule = (who == AgentPrincipal::chat ? chat_session_rules : mcp_session_rules).count(rule) > 0;
    const std::string agent = rule_agent(who);
    c.saved_rule = !agent.empty() &&
                   std::find(perms.rules.begin(), perms.rules.end(), AgentRule{agent, rule}) != perms.rules.end();
    const PermissionVerdict v = decide_agent_permission(d, inv, base, c);
    last_note = {key, ApprovalNote{v.message, v.allowed_by}};
    return v.decision;
}

ApprovalNote Session::Impl::note_of(const CommandInvocation& inv) {
    const std::string key = invocation_key(inv);
    std::lock_guard<std::mutex> lock(perm_mutex);
    if (last_note.first != key) return {};
    ApprovalNote out = std::move(last_note.second);
    last_note = {};
    return out;
}

ApprovalDecision Session::Impl::base_policy(const CommandDescriptor& d, const CommandInvocation& inv) {
    if (starts_run(inv.name)) {
        // A start or a fork of a managed run runs Python code: it waits for
        // a card from every origin, the person's own panel too
        // (design/hosted-runs.md, Review Card). A request that its handler
        // refuses anyway goes through, so that the reason comes at once.
        if (runner_precheck(inv, model_runner.load(), workspace)) return ApprovalDecision::allow;
        std::lock_guard<std::mutex> lock(python_mutex);
        python_asked.push_back(invocation_key(inv));
        while (python_asked.size() > 256) python_asked.pop_front();
        return ApprovalDecision::ask;
    }
    if (!user_origin(inv.origin) && runs_python(inv)) {
        // Running a script runs the user's code, so every such invocation
        // asks, whatever the command's flags say.
        std::lock_guard<std::mutex> lock(python_mutex);
        python_asked.push_back(invocation_key(inv));
        while (python_asked.size() > 256) python_asked.pop_front();
        return ApprovalDecision::ask;
    }
    if (inv.name == "instrument.save" && !user_origin(inv.origin)) {
        // Writing a new instrument destroys nothing; overwriting one asks.
        if (!bool_arg(inv.args, "overwrite", false)) return ApprovalDecision::allow;
        std::string id = str_arg(inv.args, "id");
        if (id.empty()) {
            std::lock_guard<std::mutex> lock(doc_mutex);
            if (doc) id = doc->spec.instrument.id;
        }
        if (!instrument_saved(id)) return ApprovalDecision::allow;
        bind_content(inv);  // the card shows the file as it would be written
        return ApprovalDecision::ask;
    }
    // Replacing an open spec with unsaved changes is a change to the open
    // spec, reviewed as an edit is, with the diff of the two documents.
    if (replaces_document(inv.name) && review_for(inv.origin) && !bind_content(inv).is_null())
        return ApprovalDecision::ask;
    // With review on for its origin, an edit that changes the open spec
    // waits for the person, with its diff.
    if (!review_of(inv).is_null()) return ApprovalDecision::ask;
    // A request ID alone reads a retained reply. The transport verifies
    // the method and run, and an unknown ID cannot submit an operation.
    if ((inv.name == "trace.run" || inv.name == "trace.validate" || inv.name == "patch.sweep") &&
        inv.args.is_object() && inv.args.size() == 1) {
        const auto request = inv.args.find("request_id");
        if (request != inv.args.end() && request->is_number_integer() && request->get<std::uint64_t>() > 0 &&
            request->get<std::uint64_t>() <= 0xffffffffu)
            return ApprovalDecision::allow;
    }
    // Acting on the running script waits for the person when an agent
    // asks (design/instruments.md, Run Effect Class); the modes and the
    // rules then apply.
    if (d.effect == CommandEffect::run && (inv.origin == CommandOrigin::agent || inv.origin == CommandOrigin::mcp))
        return ApprovalDecision::ask;
    return default_approval_policy(d, inv);
}

// ---- Session public permission API -----------------------------------------

AgentMode Session::agent_mode(AgentPrincipal who) const {
    std::lock_guard<std::mutex> lock(impl_->perm_mutex);
    return impl_->mode_of(who);
}

Error Session::set_agent_mode(AgentPrincipal who, AgentMode mode) {
    Impl& im = *impl_;
    std::lock_guard<std::mutex> lock(im.perm_mutex);
    if (who == AgentPrincipal::none)
        return Error("permissions.no_principal", "Only the side chat's agent and mcp clients have a mode.");
    AgentMode& slot = who == AgentPrincipal::chat ? im.perms.chat : im.perms.mcp;
    bool& pending = who == AgentPrincipal::chat ? im.pending_chat : im.pending_mcp;
    const AgentMode before = slot;
    const bool was_pending = pending;
    // The person's choice is always written, also when memory holds it already.
    slot = mode;
    pending = true;
    const Error e = im.save_perms();
    // Allow all holds only once the file says so. Another mode holds at once
    // even when its save failed, and saved_agent_mode() and
    // permission_save_error() say that the file still holds the old one.
    if (e && mode == AgentMode::allow_all) {
        slot = before;
        pending = was_pending;
    }
    return e;
}

AgentMode Session::saved_agent_mode(AgentPrincipal who) const {
    std::lock_guard<std::mutex> lock(impl_->perm_mutex);
    return who == AgentPrincipal::chat ? impl_->saved.chat
                                       : who == AgentPrincipal::mcp ? impl_->saved.mcp : AgentMode::ask;
}

std::string Session::permission_save_error() const {
    std::lock_guard<std::mutex> lock(impl_->perm_mutex);
    return impl_->save_error;
}

void Session::set_chat_agent(const std::string& agent_id) {
    std::lock_guard<std::mutex> lock(impl_->perm_mutex);
    impl_->chat_agent = agent_id;
}

std::string Session::chat_agent() const {
    std::lock_guard<std::mutex> lock(impl_->perm_mutex);
    return impl_->chat_agent;
}

void Session::end_chat_session() {
    std::lock_guard<std::mutex> lock(impl_->perm_mutex);
    impl_->chat_session_rules.clear();
}

bool Session::approval_offers_rules(const CommandInvocation& parked) const {
    const AgentPrincipal who = agent_principal(parked);
    if (who == AgentPrincipal::none || !rule_command_allowed(permission_rule_key(parked))) return false;
    const std::string key = Impl::invocation_key(parked);
    if (impl_->is_bound(key) || impl_->python_pending(key) || Impl::starts_run(parked.name)) return false;
    std::lock_guard<std::mutex> lock(impl_->perm_mutex);
    return who == AgentPrincipal::mcp || !impl_->rule_agent(who).empty();
}

Error Session::allow_always(const CommandInvocation& parked, RuleScope scope) {
    if (!approval_offers_rules(parked))
        return Error("permissions.no_rule", "This approval cannot become a rule; it is decided on its card each time.",
                     parked.name);
    Impl& im = *impl_;
    const AgentPrincipal who = agent_principal(parked);
    // The key the policy looks rules up by: a run command's carries its
    // target (run.set_param:lr), so that the rule allows one parameter.
    const std::string key = permission_rule_key(parked);
    std::lock_guard<std::mutex> lock(im.perm_mutex);
    if (scope == RuleScope::session) {
        (who == AgentPrincipal::chat ? im.chat_session_rules : im.mcp_session_rules).insert(key);
        return {};
    }
    const AgentRule rule{im.rule_agent(who), key};
    if (std::find(im.perms.rules.begin(), im.perms.rules.end(), rule) == im.perms.rules.end())
        im.perms.rules.push_back(rule);
    // The person's choice is always written, also when memory holds it already.
    im.pending_revoked.erase(std::remove(im.pending_revoked.begin(), im.pending_revoked.end(), rule),
                             im.pending_revoked.end());
    if (std::find(im.pending_added.begin(), im.pending_added.end(), rule) == im.pending_added.end())
        im.pending_added.push_back(rule);
    return im.save_perms();
}

std::vector<AgentRule> Session::saved_agent_rules() const {
    std::lock_guard<std::mutex> lock(impl_->perm_mutex);
    return impl_->perms.rules;
}

bool Session::revoke_agent_rule(const AgentRule& rule) {
    Impl& im = *impl_;
    std::lock_guard<std::mutex> lock(im.perm_mutex);
    const auto it = std::find(im.perms.rules.begin(), im.perms.rules.end(), rule);
    if (it == im.perms.rules.end()) return false;
    im.perms.rules.erase(it);
    im.pending_added.erase(std::remove(im.pending_added.begin(), im.pending_added.end(), rule), im.pending_added.end());
    if (std::find(im.pending_revoked.begin(), im.pending_revoked.end(), rule) == im.pending_revoked.end())
        im.pending_revoked.push_back(rule);
    im.save_perms();
    return true;
}

std::vector<std::string> Session::permission_problems() const {
    std::lock_guard<std::mutex> lock(impl_->perm_mutex);
    return impl_->perm_problems;
}

}  // namespace caliper_builder
