// session_internal.h: private types, helpers, and Session::Impl declaration
// shared across session.cpp, session_catalog.cpp, session_spec.cpp,
// session_preview.cpp, and session_permissions.cpp.
#pragma once

#include <caliper_builder/session/session.h>

#include <caliper_builder/catalog/catalog_entry.h>
#include <caliper_builder/commands/command_table.h>
#include <caliper_builder/commands/schema.h>
#include <caliper_builder/commands/sha256.h>
#include <caliper_builder/loaders/inference.h>
#include <caliper_builder/loaders/loader.h>
#include <caliper_builder/loaders/paths.h>
#include <caliper_builder/session/interactive_commands.h>
#include <caliper_builder/session/intervene_commands.h>
#include <caliper_builder/session/line_diff.h>
#include <caliper_builder/session/panel_data.h>
#include <caliper_builder/session/piece_commands.h>
#include <caliper_builder/session/program_commands.h>
#include <caliper_builder/session/resource_commands.h>
#include <caliper_builder/session/spec_edit.h>
#include <caliper_builder/spec/format.h>

#include <array>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <filesystem>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <set>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "instrument_files.h"
#include "util.h"

namespace caliper_builder {

namespace session_internal {

namespace fs = std::filesystem;
using nlohmann::json;

inline constexpr std::size_t kMaxQueryJson = std::size_t(8) << 20;  // data.query rows, as the table says
inline constexpr std::size_t kMaxInstrumentFolders = 4096;

inline CommandOutput fail(std::string code, std::string message, std::string where = {}) {
    return CommandOutput::fail(Error(std::move(code), std::move(message), std::move(where)));
}

inline bool user_origin(CommandOrigin o) { return o == CommandOrigin::ui || o == CommandOrigin::builder; }

inline std::string str_arg(const json& args, const char* key, const std::string& fallback = {}) {
    const auto it = args.find(key);
    return it != args.end() && it->is_string() ? it->get<std::string>() : fallback;
}

inline std::int64_t int_arg(const json& args, const char* key, std::int64_t fallback) {
    const auto it = args.find(key);
    if (it == args.end() || !it->is_number()) return fallback;
    return static_cast<std::int64_t>(it->get<double>());
}

inline bool bool_arg(const json& args, const char* key, bool fallback) {
    const auto it = args.find(key);
    return it != args.end() && it->is_boolean() ? it->get<bool>() : fallback;
}

inline std::vector<std::int64_t> int_list_arg(const json& args, const char* key) {
    std::vector<std::int64_t> out;
    const auto it = args.find(key);
    if (it == args.end() || !it->is_array()) return out;
    for (const auto& v : *it)
        if (v.is_number()) out.push_back(static_cast<std::int64_t>(v.get<double>()));
    return out;
}

inline Error conflict(const std::string& what) {
    return Error("history.conflict",
                 "The open spec changed since this step, so " + what + " would overwrite a change it never saw.");
}

inline const char* loader_status(const std::string& s) {
    if (s == "ok") return "ok";
    if (s == "error") return "failed";
    return "never_run";
}

// The text of a loader's reader, from its recipe or its path's extension.
inline std::string reader_text(const Loader& l) {
    const auto r = declared_reader(l);
    return r ? reader_name(*r) : "";
}

// The fixed part of a path or glob: the segments before the first with a
// wildcard, and whether there was one.
inline std::pair<fs::path, bool> fixed_part(const fs::path& pattern) {
    fs::path fixed;
    for (const auto& part : pattern) {
        const std::string s = part.u8string();
        if (s.find_first_of("*?") != std::string::npos) return {fixed, true};
        fixed /= part;
    }
    return {fixed, false};
}

}  // namespace session_internal

// The open spec and where it came from. `id` names the document, so that an
// undo can tell it apart from another spec opened since.
struct Document {
    std::uint64_t id = 0;
    spec::Spec spec;
    std::optional<std::string> path;  // relative to the workspace
    std::string saved_text;           // the canonical text when last opened or saved

    bool dirty() const { return !path || spec::to_toml(spec) != saved_text; }
};

struct LoadedSpec {
    Error error;
    spec::Spec spec;
    std::string path;  // relative to the workspace
    std::vector<spec::Diagnostic> diagnostics;
};

// A stacked preview: the open spec with the edits of `stack` applied, kept in
// `shadow`, which was last computed from `base`.
struct Preview {
    std::string id;
    std::uint64_t doc = 0;  // the document it was begun on
    spec::Spec base;
    spec::Spec shadow;
    std::vector<CommandInvocation> stack;  // the edits that changed the shadow, in order, without dry_run
    std::size_t bytes = 0;                 // their arguments as JSON text
    std::string panel;                     // the scope: a panel or a source, or neither
    std::string source;
    std::chrono::steady_clock::time_point used;
};

struct Session::Impl {
    // Declared first, so that it outlives the journal that feeds it.
    std::shared_ptr<EventRing> events = std::make_shared<EventRing>();
    std::atomic<RunControl*> run_control{nullptr};
    std::atomic<ModelRunner*> model_runner{nullptr};
    std::atomic<ProgramHost*> program_host{nullptr};
    std::atomic<DrawStatusHost*> draw_host{nullptr};
    // What the program host was last given (reconcile_hosts()).
    std::mutex reconcile_mutex;
    std::optional<spec::Spec> reconciled;
    ProgramHost* reconciled_program_host = nullptr;
    std::string workspace;
    SessionOptions options;
    std::unique_ptr<LoaderStore> store;
    std::shared_ptr<LoaderCatalogProvider> provider;
    std::unique_ptr<Catalog> catalog;
    DataSources sources;  // of the kinds the app serves
    std::unique_ptr<CommandJournal> journal;
    mutable std::mutex host_mutex;
    std::shared_ptr<SessionHost> host;
    std::unique_ptr<CommandRegistry> registry;
    std::unique_ptr<CommandQueue> queue;
    // The session state of instruments (design/instrument-interface.md):
    // the values that references read, the readers of each input and the
    // kept steps. reconcile_hosts() resets them when the open spec changes.
    ValueStore values;
    StepHistory history;
    DependencyIndex deps;
    std::mutex instrument_mutex;
    std::optional<spec::Spec> instrumented;  // the spec they were last reset for, under instrument_mutex

    mutable std::mutex doc_mutex;
    std::optional<Document> doc;
    std::uint64_t next_doc = 1;
    // The stacked previews, under doc_mutex, and the ids of the last ones that
    // expired, so that a late call learns why its preview is gone.
    std::map<std::string, Preview> previews;
    std::deque<std::string> expired;
    std::uint64_t next_preview = 1;
    std::map<std::string, double> last_elapsed;  // by loader id; touched only by handlers
    // Invocations from agent, mcp and api that the policy asked about because
    // they run a Python loader, keyed by their id, name and arguments; a
    // handler that finds its invocation here runs it with the approval the
    // person gave. Bounded, oldest first out.
    std::mutex python_mutex;
    std::deque<std::string> python_asked;
    // Review of spec edits: whether it is on for each origin, the edits parked
    // for review (by invocation key, with the document and the lines the
    // shown change removes and adds), oldest first out, and the review the
    // policy computed last, which the registry keeps with the parked call.
    struct Reviewed {
        std::string key;
        std::uint64_t doc = 0;
        std::string change;
    };
    std::array<std::atomic<bool>, 6> review_on{};  // by CommandOrigin
    std::mutex review_mutex;
    std::deque<Reviewed> reviewed;
    std::pair<std::string, nlohmann::json> last_review;
    // Approvals bound to content (an overwrite of a saved instrument, or the
    // replacement of an open spec with unsaved changes): the document and the
    // digest of what the card showed, by invocation key, oldest first out.
    struct Bound {
        std::string key;
        std::uint64_t doc = 0;
        std::string digest;
    };
    std::deque<Bound> bound;  // under review_mutex
    // Agent permissions (permissions.h), under perm_mutex: the saved modes
    // and rules, the chat's agent id, the rules of this session by principal,
    // and the note of the invocation the policy decided last, which the
    // registry asks for next.
    mutable std::mutex perm_mutex;
    AgentPermissionsFile perms;
    AgentPermissionsFile saved;  // what the file holds, as last read or written
    std::string save_error;      // why the last save failed; "" after a successful one
    // What the person chose in this app and no save has written yet: always
    // written by the next save, whatever the file holds.
    bool pending_chat = false, pending_mcp = false;
    std::vector<AgentRule> pending_added, pending_revoked;
    // The permissions file's size, modification time and identity at the
    // last read or write, so that a changed file is noticed without a read.
    struct FileStamp {
        bool exists = false;
        long long size = 0, mtime_ns = 0;
        unsigned long long dev = 0, ino = 0;
        bool operator==(const FileStamp& o) const {
            return exists == o.exists && size == o.size && mtime_ns == o.mtime_ns && dev == o.dev && ino == o.ino;
        }
    };
    FileStamp stamp;
    std::string chat_agent;
    std::set<std::string> chat_session_rules;
    std::set<std::string> mcp_session_rules;
    std::vector<std::string> perm_problems;
    std::pair<std::string, ApprovalNote> last_note;
    std::mutex close_mutex;
    bool closed = false;

    struct MadeLoader {
        std::string id;
        std::string recipe;
        bool ran = false;
    };

    struct Content {
        bool applies = false;
        std::uint64_t doc = 0;
        std::string digest;
        nlohmann::json review;
    };

    // ---- shared helpers and app commands (session.cpp) ----
    std::shared_ptr<SessionHost> current_host() const;
    spec::SourceResolver resolver() const;
    std::string rel(const std::filesystem::path& p) const;
    std::filesystem::path ws() const;
    std::string spec_label() const;
    LoadedSpec load_spec_file(const std::string& rel_path, const spec::SourceResolver& res) const;
    LoadedSpec load_instrument(const std::string& id, const spec::SourceResolver& res) const;
    void refuse_unresolved_pins(const spec::Spec& s);
    UndoRecord replace_document(Document next);
    UndoRecord change_spec(spec::Spec after, bool merging, std::size_t bytes);
    bool instrument_saved(const std::string& id) const;
    CommandOutput panel_render(const CommandContext& ctx);
    CommandOutput run_cmd(const CommandContext& ctx);
    CommandOutput runner_cmd(const CommandContext& ctx);
    CommandOutput events_wait(const CommandContext& ctx);
    spec::Spec open_copy() const;
    CommandOutput piece_cmd(const CommandContext& ctx);
    CommandOutput piece_save(const CommandContext& ctx);
    CommandOutput resource_cmd(const CommandContext& ctx);
    CommandOutput program_cmd(const CommandContext& ctx);
    CommandOutput interactive_cmd(const CommandContext& ctx);
    ChainContext chain_context() const;
    void reset_instruments();
    CommandOutput interpretation_cmd(const CommandContext& ctx);
    CommandOutput intervene_cmd(const CommandContext& ctx);
    void approval_event(const PendingApproval& p, std::string_view state);
    std::string open_instrument_id() const;
    void reconcile_hosts();
    void bind_all();

    // ---- catalog, data and loaders (session_catalog.cpp) ----
    Error check_loader_input(const Loader& l, CommandOrigin origin);
    static Error loader_save_error(Error e);
    nlohmann::json run_outputs(const RunResult& r) const;
    std::map<std::string, std::vector<std::string>> outputs_by_loader();
    RunResult run_loader(const std::string& id, bool allow_python, double* elapsed_ms = nullptr);
    CommandOutput catalog_list(const CommandContext& ctx);
    CommandOutput catalog_describe(const CommandContext& ctx);
    CommandOutput catalog_sample(const CommandContext& ctx);
    CommandOutput data_query(const CommandContext& ctx);
    UndoRecord made_loaders_undo(std::vector<MadeLoader> made);
    CommandOutput data_import(const CommandContext& ctx);
    nlohmann::json loader_summary(const LoaderRecord& r,
                                  const std::map<std::string, std::vector<std::string>>& outputs);
    CommandOutput loader_list(const CommandContext& ctx);
    CommandOutput loader_describe(const CommandContext& ctx);
    CommandOutput loader_create(const CommandContext& ctx);
    CommandOutput loader_update(const CommandContext& ctx);
    CommandOutput loader_run(const CommandContext& ctx);
    CommandOutput loader_delete(const CommandContext& ctx);

    // ---- open spec, instruments and panel data (session_spec.cpp) ----
    CommandOutput no_open_spec();
    CommandOutput spec_new(const CommandContext& ctx);
    CommandOutput open_loaded(LoadedSpec loaded, nlohmann::json result);
    CommandOutput spec_open(const CommandContext& ctx);
    CommandOutput instrument_open(const CommandContext& ctx);
    Error view_spec(const std::string& instrument, const std::string& preview, spec::Spec* out);
    CommandOutput spec_get(const CommandContext& ctx);
    CommandOutput spec_validate(const CommandContext& ctx);
    SpecEdit edit_spec(const spec::Spec& s, const std::string& name, nlohmann::json& args);
    CommandOutput spec_edit(const CommandContext& ctx);
    CommandOutput instrument_save(const CommandContext& ctx);
    CommandOutput instrument_list(const CommandContext& ctx);
    CommandOutput panel_data(const CommandContext& ctx);

    // ---- review of spec edits and stacked previews (session_preview.cpp) ----
    bool review_for(CommandOrigin o) const;
    static std::string change_lines(const std::string& before, const std::string& after);
    nlohmann::json review_of(const CommandInvocation& inv);
    nlohmann::json parked_review(const CommandInvocation& inv);
    std::optional<Reviewed> take_reviewed(const CommandInvocation& inv);
    CommandOutput review_status(const CommandContext& ctx);
    std::chrono::steady_clock::time_point preview_now() const;
    void sweep_previews();
    Error find_preview(const std::string& id, const std::string& where, Preview** out);
    Error refresh_preview(Preview& p);
    Error use_preview(const std::string& id, const std::string& where, Preview** out);
    Error check_scope(const Preview& p, const std::string& name, const nlohmann::json& args) const;
    CommandOutput preview_edit(const CommandContext& ctx, bool dry);
    static nlohmann::json scope_json(const std::string& id);
    CommandOutput preview_begin(const CommandContext& ctx);
    CommandOutput preview_diff(const CommandContext& ctx);
    CommandOutput preview_commit(const CommandContext& ctx);
    CommandOutput preview_discard(const CommandContext& ctx);

    // ---- content-bound approvals and agent permissions (session_permissions.cpp) ----
    static bool replaces_document(const std::string& name);
    Content content_of(const CommandInvocation& inv);
    nlohmann::json bind_content(const CommandInvocation& inv);
    bool approval_holds(const CommandInvocation& inv);
    static std::string invocation_key(const CommandInvocation& inv);
    static bool is_python(const Loader& l);
    static bool starts_run(const std::string& name);
    bool runs_python(const CommandInvocation& inv);
    bool python_approved(const CommandInvocation& inv);
    bool run_approved(const CommandInvocation& inv);
    bool is_bound(const std::string& key);
    bool python_pending(const std::string& key);
    std::string rule_agent(AgentPrincipal who) const;
    AgentMode mode_of(AgentPrincipal who) const;
    FileStamp file_stamp() const;
    static int strictness(AgentMode m);
    static AgentMode stricter(AgentMode a, AgentMode b);
    void recheck_allow_all(AgentPrincipal who);
    Error failed_save(Error e);
    Error save_perms();
    ApprovalDecision policy(const CommandDescriptor& d, const CommandInvocation& inv);
    ApprovalNote note_of(const CommandInvocation& inv);
    ApprovalDecision base_policy(const CommandDescriptor& d, const CommandInvocation& inv);
};

}  // namespace caliper_builder
