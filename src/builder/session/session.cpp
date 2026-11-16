#include <caliper_builder/session/session.h>

#include <caliper_builder/catalog/catalog_entry.h>
#include <caliper_builder/commands/command_table.h>
#include <caliper_builder/commands/schema.h>
#include <caliper_builder/commands/sha256.h>
#include <caliper_builder/loaders/inference.h>
#include <caliper_builder/loaders/loader.h>
#include <caliper_builder/loaders/paths.h>
#include <caliper_builder/session/line_diff.h>
#include <caliper_builder/session/panel_data.h>
#include <caliper_builder/session/spec_edit.h>
#include <caliper_builder/spec/format.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <exception>
#include <filesystem>
#include <deque>
#include <map>
#include <mutex>
#include <set>
#include <utility>

#include "applet_files.h"
#include "util.h"

namespace caliper_builder {

namespace {

namespace fs = std::filesystem;
using nlohmann::json;
using namespace session_detail;

constexpr std::size_t kMaxQueryJson = std::size_t(8) << 20;  // data.query rows, as the table says
constexpr std::size_t kMaxAppletFolders = 4096;

CommandOutput fail(std::string code, std::string message, std::string where = {}) {
    return CommandOutput::fail(Error(std::move(code), std::move(message), std::move(where)));
}

bool user_origin(CommandOrigin o) { return o == CommandOrigin::ui || o == CommandOrigin::builder; }

std::string str_arg(const json& args, const char* key, const std::string& fallback = {}) {
    const auto it = args.find(key);
    return it != args.end() && it->is_string() ? it->get<std::string>() : fallback;
}

std::int64_t int_arg(const json& args, const char* key, std::int64_t fallback) {
    const auto it = args.find(key);
    if (it == args.end() || !it->is_number()) return fallback;
    return static_cast<std::int64_t>(it->get<double>());
}

bool bool_arg(const json& args, const char* key, bool fallback) {
    const auto it = args.find(key);
    return it != args.end() && it->is_boolean() ? it->get<bool>() : fallback;
}

std::vector<std::int64_t> int_list_arg(const json& args, const char* key) {
    std::vector<std::int64_t> out;
    const auto it = args.find(key);
    if (it == args.end() || !it->is_array()) return out;
    for (const auto& v : *it)
        if (v.is_number()) out.push_back(static_cast<std::int64_t>(v.get<double>()));
    return out;
}

Error conflict(const std::string& what) {
    return Error("history.conflict",
                 "The open spec changed since this step, so " + what + " would overwrite a change it never saw.");
}

const char* loader_status(const std::string& s) {
    if (s == "ok") return "ok";
    if (s == "error") return "failed";
    return "never_run";
}

// The text of a loader's reader, from its recipe or its path's extension.
std::string reader_text(const Loader& l) {
    const auto r = declared_reader(l);
    return r ? reader_name(*r) : "";
}

// The fixed part of a path or glob: the segments before the first with a
// wildcard, and whether there was one.
std::pair<fs::path, bool> fixed_part(const fs::path& pattern) {
    fs::path fixed;
    for (const auto& part : pattern) {
        const std::string s = part.u8string();
        if (s.find_first_of("*?") != std::string::npos) return {fixed, true};
        fixed /= part;
    }
    return {fixed, false};
}

}  // namespace

const std::vector<std::string>& session_host_commands() {
    static const std::vector<std::string> names = {"app.info", "app.applets", "app.launch", "panel.render",
                                                   "capture.frame"};
    return names;
}

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
    std::pair<std::string, json> last_review;
    // Approvals bound to content (an overwrite of a saved applet, or the
    // replacement of an open spec with unsaved changes): the document and the
    // digest of what the card showed, by invocation key, oldest first out.
    struct Bound {
        std::string key;
        std::uint64_t doc = 0;
        std::string digest;
    };
    std::deque<Bound> bound;  // under review_mutex
    // Agent permissions (permissions.h), under perm_mutex: the saved modes
    // and rules, Allow all of the chat's session, the chat's agent id, the
    // rules of this session by principal, and the note of the invocation the
    // policy decided last, which the registry asks for next.
    mutable std::mutex perm_mutex;
    AgentPermissionsFile perms;
    bool chat_allow_all = false;
    std::string chat_agent;
    std::set<std::string> chat_session_rules;
    std::set<std::string> mcp_session_rules;
    std::vector<std::string> perm_problems;
    std::pair<std::string, ApprovalNote> last_note;
    std::mutex close_mutex;
    bool closed = false;

    // ---- shared helpers ------------------------------------------------

    std::shared_ptr<SessionHost> current_host() const {
        std::lock_guard<std::mutex> lock(host_mutex);
        return host;
    }

    spec::SourceResolver resolver() const {
        const Catalog* cat = catalog.get();
        return [cat](const std::string& id) -> std::optional<spec::ResolvedSource> {
            CatalogDescribeResult d = cat->describe(id);
            if (d.error || !d.entry) return std::nullopt;
            const CatalogEntry& e = *d.entry;
            spec::ResolvedSource r;
            r.kind = catalog_kind_name(e.kind);
            if (e.columns) {
                std::vector<spec::ResolvedColumn> cols;
                for (const auto& c : *e.columns) cols.push_back({c.name, c.type});
                r.columns = std::move(cols);
            }
            if (e.dtype) r.dtype = catalog_dtype_name(*e.dtype);
            r.shape = e.shape;
            r.wire = e.wire;
            return r;
        };
    }

    std::string rel(const fs::path& p) const {
        const std::string c = p.u8string();
        if (path_within(c, workspace) && c.size() > workspace.size())
            return fs::u8path(c.substr(workspace.size() + 1)).generic_u8string();
        return p.generic_u8string();
    }

    fs::path ws() const { return fs::u8path(workspace); }

    std::string spec_label() const {
        if (doc && doc->path) return *doc->path;
        return (doc ? doc->spec.applet.id : std::string("spec")) + ".spec.toml";
    }

    // Reads, parses and validates a spec file given relative to the workspace.
    LoadedSpec load_spec_file(const std::string& rel_path, const spec::SourceResolver& res) const {
        LoadedSpec out;
        const fs::path p = fs::u8path(rel_path);
        bool dotdot = false;
        for (const auto& part : p) dotdot = dotdot || part == "..";
        if (rel_path.empty() || p.is_absolute() || p.has_root_name() || p.has_root_directory() || dotdot) {
            out.error = Error("spec.path_refused", "A spec path is relative to the workspace and has no .. segment.", "args.path");
            return out;
        }
        std::error_code ec;
        const fs::path canonical = fs::canonical(ws() / p, ec);
        if (ec || !fs::is_regular_file(canonical, ec)) {
            out.error = Error("spec.not_found", "The workspace has no spec file " + rel_path + ".", "args.path");
            return out;
        }
        if (!path_within(canonical.u8string(), workspace)) {
            out.error = Error("spec.path_refused", "The spec file lies outside the workspace.", "args.path");
            return out;
        }
        std::string text;
        if (Error e = read_file_bounded(canonical.u8string(), spec::limits::kTextBytes, &text, "spec.too_large")) {
            out.error = e.code == "session.io" ? Error("spec.not_found", "The spec file could not be read.", "args.path") : e;
            return out;
        }
        SpecTextCheck checked = check_spec_text(text, res);
        if (checked.error) {
            out.error = checked.error;
            return out;
        }
        out.spec = std::move(checked.spec);
        out.diagnostics = std::move(checked.diagnostics);
        out.path = rel(canonical);
        return out;
    }

    // The spec of a saved applet, by its folder id.
    LoadedSpec load_applet(const std::string& id, const spec::SourceResolver& res) const {
        LoadedSpec out;
        const fs::path folder = ws() / "applets" / fs::u8path(id);
        const fs::path manifest = folder / fs::u8path(id + ".caliper.toml");
        std::error_code ec;
        if (!valid_id(id) || !fs::is_regular_file(manifest, ec)) {
            out.error = Error("applet.not_found", "The workspace has no applet " + id + "; applet.list shows the saved ones.",
                              "args.id");
            return out;
        }
        ManifestInfo m = read_manifest(manifest.u8string());
        if (m.error) {
            out.error = m.error;
            out.error.where = "args.id";
            return out;
        }
        if (m.kind != "spec" || !plain_file_name(m.spec)) {
            out.error = Error("applet.not_spec", "The applet " + id + " is not a spec applet, so it cannot be edited here.",
                              "args.id");
            return out;
        }
        return load_spec_file(rel(folder / fs::u8path(m.spec)), res);
    }

    // Replaces the open document. Its undo restores the document that was
    // open before; its redo puts back the one it replaced.
    UndoRecord replace_document(Document next) {
        struct State {
            std::optional<Document> previous;
            std::uint64_t current = 0;
            std::optional<Document> redo;
        };
        auto state = std::make_shared<State>();
        state->previous = doc;
        state->current = next.id;
        const std::size_t bytes = 2 * (spec::to_toml(next.spec).size() + (doc ? spec::to_toml(doc->spec).size() : 0)) + 512;
        doc = std::move(next);
        auto undo = [this, state]() -> Error {
            std::lock_guard<std::mutex> lock(doc_mutex);
            if (!doc || doc->id != state->current) return conflict("restoring the spec open before");
            state->redo = doc;
            doc = state->previous;
            return {};
        };
        auto redo = [this, state]() -> Error {
            std::lock_guard<std::mutex> lock(doc_mutex);
            const bool same = state->previous ? (doc && doc->id == state->previous->id) : !doc;
            if (!same || !state->redo) return conflict("opening the spec again");
            doc = state->redo;
            return {};
        };
        return UndoRecord::closures(std::move(undo), std::move(redo), bytes);
    }

    // Changes the open document's spec. A command meant to merge checks only
    // that the document is the same; others also check its content.
    UndoRecord change_spec(spec::Spec after, bool merging, std::size_t bytes) {
        spec::Spec before = doc->spec;
        doc->spec = after;
        const std::uint64_t id = doc->id;
        auto undo = [this, id, before, after, merging]() -> Error {
            std::lock_guard<std::mutex> lock(doc_mutex);
            if (!doc || doc->id != id || (!merging && doc->spec != after)) return conflict("this undo");
            doc->spec = before;
            return {};
        };
        auto redo = [this, id, before, after, merging]() -> Error {
            std::lock_guard<std::mutex> lock(doc_mutex);
            if (!doc || doc->id != id || (!merging && doc->spec != before)) return conflict("this redo");
            doc->spec = after;
            return {};
        };
        return UndoRecord::closures(std::move(undo), std::move(redo), bytes);
    }

    // Checks the input path of a loader for its origin, and grants a path
    // outside the workspace that the person named.
    Error check_loader_input(const Loader& l, CommandOrigin origin) {
        const fs::path pattern = fs::u8path(l.path);
        if (!pattern.is_absolute()) return {};  // inside the workspace; the store refuses `..` and escapes
        const auto [fixed, wildcard] = fixed_part(pattern);
        std::error_code ec;
        const fs::path canonical = fs::weakly_canonical(fixed, ec);
        if (ec) return Error("loader.path_refused", "The input path cannot be resolved.", "input.path");
        const std::string c = canonical.u8string();
        if (path_within(c, workspace)) return {};
        const AllowedRoots roots{workspace, store->granted_folders(), store->granted_files()};
        if (path_allowed(c, roots)) return {};
        if (!user_origin(origin))
            return Error("loader.path_refused",
                         "The input path lies outside the workspace and outside every file and folder the person "
                         "granted; only the person can add one, by dropping it on Caliper.",
                         "input.path");
        (void)wildcard;  // a glob grants its fixed folder; a plain path grants the file or folder itself
        if (Error e = store->grant(c)) return Error("loader.path_refused", e.message, "input.path");
        return {};
    }

    static Error loader_save_error(Error e) {
        if (e.code == "loader.path_not_allowed") e.code = "loader.path_refused";
        return e;
    }

    json run_outputs(const RunResult& r) const {
        json outputs = json::array();
        for (const LoadedSource& s : r.sources) outputs.push_back(s.id);
        return outputs;
    }

    std::map<std::string, std::vector<std::string>> outputs_by_loader() {
        std::map<std::string, std::vector<std::string>> out;
        for (const LoadedSource& s : store->sources()) out[s.loader].push_back(s.id);
        return out;
    }

    // Runs a loader, records how long it took and tells the catalog.
    // Runs a loader; a Python loader's script runs only with `allow_python`.
    RunResult run_loader(const std::string& id, bool allow_python, double* elapsed_ms = nullptr) {
        const auto t0 = std::chrono::steady_clock::now();
        RunOptions options;
        options.allow_python = allow_python;
        RunResult r = store->run(id, options);
        const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
        last_elapsed[id] = ms;
        if (elapsed_ms) *elapsed_ms = ms;
        provider->touch();
        return r;
    }

    bool applet_saved(const std::string& id) const {
        if (!valid_id(id)) return false;
        std::error_code ec;
        const fs::path folder = ws() / "applets" / fs::u8path(id);
        return fs::exists(folder / fs::u8path(id + ".caliper.toml"), ec) ||
               fs::exists(folder / fs::u8path(id + ".spec.toml"), ec);
    }

    // ---- catalog and data ------------------------------------------------

    CommandOutput catalog_list(const CommandContext& ctx) {
        const json& a = ctx.invocation.args;
        CatalogFilter f;
        if (a.contains("kind")) f.kind = catalog_kind_from_name(str_arg(a, "kind"));
        f.prefix = str_arg(a, "prefix");
        f.limit = static_cast<std::size_t>(int_arg(a, "limit", 100));
        CatalogMergedList r = catalog->list(f);
        if (r.error) return CommandOutput::fail(r.error);
        json entries = json::array();
        for (const auto& e : r.entries) entries.push_back(catalog_entry_to_json(e));
        return CommandOutput::ok({{"entries", std::move(entries)}, {"truncated", r.truncated}, {"warnings", r.warnings}});
    }

    CommandOutput catalog_describe(const CommandContext& ctx) {
        const std::string id = str_arg(ctx.invocation.args, "id");
        CatalogDescribeResult r = catalog->describe(id);
        if (r.error) return CommandOutput::fail(r.error);
        if (!r.entry) return fail("catalog.not_found", "The catalog has no source " + id + "; catalog.list shows them.", "args.id");
        return CommandOutput::ok({{"entry", catalog_entry_to_json(*r.entry)}});
    }

    CommandOutput catalog_sample(const CommandContext& ctx) {
        const json& a = ctx.invocation.args;
        CatalogSampleRequest req;
        req.limit = static_cast<std::size_t>(int_arg(a, "limit", 20));
        req.start = int_list_arg(a, "start");
        req.count = int_list_arg(a, "count");
        CatalogSampleResult r = catalog->sample(str_arg(a, "id"), req);
        if (r.error) return CommandOutput::fail(r.error);
        return CommandOutput::ok(catalog_sample_to_json(r.sample));
    }

    CommandOutput data_query(const CommandContext& ctx) {
        const json& a = ctx.invocation.args;
        QueryResult q = store->query(str_arg(a, "sql"), static_cast<std::uint64_t>(int_arg(a, "limit", 1000)),
                                     static_cast<std::uint32_t>(int_arg(a, "timeout_ms", 5000)));
        if (q.error) {
            Error e = q.error;
            if (e.code == "loader.sql_not_select") e.code = "data.not_select";
            else if (e.code == "query.timeout") e.code = "data.timeout";
            else if (e.code == "loader.sql" || e.code == "loader.path_not_allowed") e.code = "data.sql_error";
            else e.code = "data.unavailable";
            e.where = "args.sql";
            return CommandOutput::fail(e);
        }
        fix_nonfinite(q);
        json columns = json::array();
        for (const auto& c : q.columns) columns.push_back(c.to_json());
        json rows = json::array();
        std::size_t bytes = 0;
        bool truncated = q.truncated;
        for (auto& row : q.rows) {
            bytes += row.dump(-1, ' ', false, json::error_handler_t::replace).size() + 1;
            if (bytes > kMaxQueryJson) {
                truncated = true;
                break;
            }
            rows.push_back(std::move(row));
        }
        return CommandOutput::ok(
            {{"columns", std::move(columns)}, {"rows", std::move(rows)}, {"truncated", truncated}, {"elapsed_ms", q.elapsed_ms}});
    }

    // The undo of commands that created loaders: deletes them if their recipes
    // are unchanged; the redo saves them again and runs those that ran.
    struct MadeLoader {
        std::string id;
        std::string recipe;
        bool ran = false;
    };

    UndoRecord made_loaders_undo(std::vector<MadeLoader> made) {
        std::size_t bytes = 256;
        for (const auto& m : made) bytes += m.recipe.size() + m.id.size();
        auto undo = [this, made]() -> Error {
            for (const auto& m : made) {
                LoaderStore::LoaderGet got = store->get(m.id);
                if (got.error.code == "loader.not_found" || got.record.recipe != m.recipe)
                    return Error("history.conflict", "The loader " + m.id + " changed since it was created.", m.id);
            }
            for (const auto& m : made)
                if (Error e = store->remove(m.id)) return e;
            provider->touch();
            return {};
        };
        auto redo = [this, made]() -> Error {
            for (const auto& m : made)
                if (store->get(m.id).error.code != "loader.not_found")
                    return Error("history.conflict", "A loader " + m.id + " exists again.", m.id);
            for (const auto& m : made) {
                LoaderParse p = parse_loader(m.recipe);
                if (p.error) return p.error;
                if (Error e = store->save(p.loader)) return e;
                if (m.ran) run_loader(m.id, false);  // a redo never runs a Python script
            }
            provider->touch();
            return {};
        };
        return UndoRecord::closures(std::move(undo), std::move(redo), bytes);
    }

    CommandOutput data_import(const CommandContext& ctx) {
        const json& a = ctx.invocation.args;
        const bool run = bool_arg(a, "run", true);
        const bool user = user_origin(ctx.invocation.origin);
        json refused = json::array();
        std::vector<std::string> accepted;
        std::vector<std::string> given;
        for (const auto& item : a["paths"]) {
            const std::string p = item.get<std::string>();
            fs::path path = fs::u8path(p);
            if (!path.is_absolute()) path = ws() / path;
            std::error_code ec;
            const fs::path canonical = fs::canonical(path, ec);
            if (ec || !fs::is_regular_file(canonical, ec)) {
                refused.push_back({{"path", p}, {"code", "loader.io"}, {"message", "The path is not a readable file."}});
                continue;
            }
            if (!user && !path_within(canonical.u8string(), workspace)) {
                refused.push_back({{"path", p},
                                   {"code", "loader.path_refused"},
                                   {"message",
                                    "Only the person can import a file from outside the workspace, by dropping it on "
                                    "Caliper or picking it in a file dialog; an agent imports files inside the workspace."}});
                continue;
            }
            accepted.push_back(path.u8string());
            given.push_back(p);
        }
        const auto items = store->import_files(accepted, user);
        json loaders = json::array();
        std::vector<MadeLoader> made;
        for (std::size_t i = 0; i < items.size(); ++i) {
            const auto& item = items[i];
            if (item.error) {
                refused.push_back({{"path", given[i]}, {"code", loader_save_error(item.error).code}, {"message", item.error.message}});
                continue;
            }
            LoaderStore::LoaderGet got = store->get(item.loader_id);
            MadeLoader m{item.loader_id, got.record.recipe, false};
            json outputs = json::array();
            const bool python = got.loader.reader && *got.loader.reader == ReaderKind::python;
            if (run && !python) {
                RunResult r = run_loader(item.loader_id, false);
                m.ran = !r.error;
                outputs = run_outputs(r);
            }
            loaders.push_back({{"id", item.loader_id},
                               {"path", got.loader.path},
                               {"reader", reader_text(got.loader)},
                               {"outputs", std::move(outputs)},
                               {"ran", m.ran}});
            made.push_back(std::move(m));
        }
        provider->touch();
        json result{{"loaders", std::move(loaders)}, {"refused", std::move(refused)}};
        if (made.empty()) return CommandOutput::ok(std::move(result));
        return CommandOutput::ok(std::move(result), made_loaders_undo(std::move(made)));
    }

    // ---- loaders -----------------------------------------------------------

    json loader_summary(const LoaderRecord& r, const std::map<std::string, std::vector<std::string>>& outputs) {
        LoaderParse p = parse_loader(r.recipe);
        const auto it = outputs.find(r.id);
        return {{"id", r.id},
                {"title", r.title},
                {"reader", p.error ? std::string() : reader_text(p.loader)},
                {"path", p.error ? std::string() : p.loader.path},
                {"status", loader_status(r.last_status)},
                {"outputs", it == outputs.end() ? json::array() : json(it->second)},
                {"updated_at", r.updated_at.empty() ? json(nullptr) : json(r.updated_at)}};
    }

    CommandOutput loader_list(const CommandContext&) {
        const auto outputs = outputs_by_loader();
        json loaders = json::array();
        for (const LoaderRecord& r : store->list()) loaders.push_back(loader_summary(r, outputs));
        return CommandOutput::ok({{"loaders", std::move(loaders)}});
    }

    CommandOutput loader_describe(const CommandContext& ctx) {
        const std::string id = str_arg(ctx.invocation.args, "id");
        LoaderStore::LoaderGet got = store->get(id);
        if (got.error.code == "loader.not_found")
            return fail("loader.not_found", "There is no saved loader " + id + "; loader.list shows them.", "args.id");
        const auto outputs = outputs_by_loader();
        json out = loader_summary(got.record, outputs);
        out.erase("updated_at");
        out["text"] = got.record.recipe;
        json last_run = nullptr;
        if (!got.record.last_run_at.empty()) {
            const auto t = last_elapsed.find(id);
            last_run = {{"at", got.record.last_run_at},
                        {"ok", got.record.last_status == "ok"},
                        {"error", got.record.last_error_code.empty() ? json(nullptr) : json(got.record.last_error_code)},
                        {"elapsed_ms", t == last_elapsed.end() ? json(nullptr) : json(t->second)}};
        }
        out["last_run"] = std::move(last_run);
        return CommandOutput::ok(std::move(out));
    }

    CommandOutput loader_create(const CommandContext& ctx) {
        const json& a = ctx.invocation.args;
        LoaderParse parsed = parse_loader(str_arg(a, "text"));
        if (parsed.error)
            return fail("loader.invalid", parsed.error.message + " (" + parsed.error.code + ")",
                        parsed.error.where.empty() ? "args.text" : "args.text " + parsed.error.where);
        const Loader& l = parsed.loader;
        if (store->get(l.id).error.code != "loader.not_found")
            return fail("loader.exists", "A loader " + l.id + " exists already; loader.update changes it.", "args.text");
        if (Error e = check_loader_input(l, ctx.invocation.origin)) return CommandOutput::fail(e);
        if (Error e = store->save(l)) return CommandOutput::fail(loader_save_error(e));
        provider->touch();
        MadeLoader m{l.id, store->get(l.id).record.recipe, false};
        json outputs = json::array();
        if (bool_arg(a, "run", false)) {
            RunResult r = run_loader(l.id, python_approved(ctx.invocation));
            if (r.error) {
                // A loader that fails its first run is not kept, so a failed
                // command changes nothing.
                store->remove(l.id);
                provider->touch();
                return CommandOutput::fail(r.error);
            }
            m.ran = true;
            outputs = run_outputs(r);
        }
        return CommandOutput::ok({{"id", l.id}, {"outputs", std::move(outputs)}}, made_loaders_undo({m}));
    }

    CommandOutput loader_update(const CommandContext& ctx) {
        const json& a = ctx.invocation.args;
        const std::string id = str_arg(a, "id");
        LoaderStore::LoaderGet before = store->get(id);
        if (before.error.code == "loader.not_found")
            return fail("loader.not_found", "There is no saved loader " + id + "; loader.list shows them.", "args.id");
        LoaderParse parsed = parse_loader(str_arg(a, "text"));
        if (parsed.error)
            return fail("loader.invalid", parsed.error.message + " (" + parsed.error.code + ")",
                        parsed.error.where.empty() ? "args.text" : "args.text " + parsed.error.where);
        if (parsed.loader.id != id)
            return fail("loader.id_mismatch", "The recipe's [loader] id is " + parsed.loader.id + ", not " + id + ".",
                        "args.text");
        if (Error e = check_loader_input(parsed.loader, ctx.invocation.origin)) return CommandOutput::fail(e);
        const std::string previous = before.record.recipe;
        if (Error e = store->save(parsed.loader)) return CommandOutput::fail(loader_save_error(e));
        provider->touch();
        const std::string now = store->get(id).record.recipe;
        const bool run = bool_arg(a, "run", false);
        json outputs = json::array();
        if (run) {
            RunResult r = run_loader(id, python_approved(ctx.invocation));
            if (r.error) {
                LoaderParse old = parse_loader(previous);
                if (!old.error) store->save(old.loader);
                provider->touch();
                return CommandOutput::fail(r.error);
            }
            outputs = run_outputs(r);
        }
        json result{{"id", id}, {"outputs", std::move(outputs)}};
        LoaderParse old = parse_loader(previous);
        if (old.error) return CommandOutput::ok(std::move(result));  // a broken recipe cannot be saved back
        auto undo = [this, id, previous, now]() -> Error {
            if (store->get(id).record.recipe != now)
                return Error("history.conflict", "The loader " + id + " changed since it was updated.", id);
            LoaderParse p = parse_loader(previous);
            if (p.error) return p.error;
            Error e = store->save(p.loader);
            provider->touch();
            return e;
        };
        auto redo = [this, id, previous, now, run]() -> Error {
            if (store->get(id).record.recipe != previous)
                return Error("history.conflict", "The loader " + id + " changed since its update was undone.", id);
            LoaderParse p = parse_loader(now);
            if (p.error) return p.error;
            if (Error e = store->save(p.loader)) return e;
            if (run) run_loader(id, false);  // a redo never runs a Python script
            provider->touch();
            return {};
        };
        return CommandOutput::ok(std::move(result),
                                 UndoRecord::closures(std::move(undo), std::move(redo), previous.size() + now.size() + 256));
    }

    CommandOutput loader_run(const CommandContext& ctx) {
        const std::string id = str_arg(ctx.invocation.args, "id");
        double ms = 0;
        RunResult r = run_loader(id, python_approved(ctx.invocation), &ms);
        if (r.error) {
            Error e = r.error;
            if (e.code == "loader.not_found") e.where = "args.id";
            return CommandOutput::fail(e);
        }
        json outputs = json::array();
        for (const LoadedSource& s : r.sources)
            outputs.push_back({{"id", s.id},
                               {"rows", s.rows ? json(*s.rows) : json(nullptr)},
                               {"shape", s.shape ? json(*s.shape) : json(nullptr)}});
        return CommandOutput::ok({{"id", id}, {"outputs", std::move(outputs)}, {"elapsed_ms", ms}});
    }

    CommandOutput loader_delete(const CommandContext& ctx) {
        const json& a = ctx.invocation.args;
        const std::string id = str_arg(a, "id");
        if (bool_arg(a, "keep_outputs", false))
            return fail("loader.unsupported",
                        "Keeping the outputs of a deleted loader is not available yet; delete it without keep_outputs, "
                        "or keep the loader.",
                        "args.keep_outputs");
        LoaderStore::LoaderGet got = store->get(id);
        if (got.error.code == "loader.not_found")
            return fail("loader.not_found", "There is no saved loader " + id + "; loader.list shows them.", "args.id");
        const auto outputs = outputs_by_loader();
        const auto it = outputs.find(id);
        const json removed = it == outputs.end() ? json::array() : json(it->second);
        const std::string recipe = got.record.recipe;
        if (Error e = store->remove(id)) return CommandOutput::fail(e);
        provider->touch();
        json result{{"id", id}, {"removed", removed}};
        if (got.error) return CommandOutput::ok(std::move(result));  // a broken recipe cannot be saved back
        auto undo = [this, id, recipe]() -> Error {
            if (store->get(id).error.code != "loader.not_found")
                return Error("history.conflict", "A loader " + id + " exists again.", id);
            LoaderParse p = parse_loader(recipe);
            if (p.error) return p.error;
            Error e = store->save(p.loader);
            provider->touch();
            return e;
        };
        auto redo = [this, id, recipe]() -> Error {
            LoaderStore::LoaderGet now = store->get(id);
            if (now.error.code == "loader.not_found" || now.record.recipe != recipe)
                return Error("history.conflict", "The loader " + id + " changed since its deletion was undone.", id);
            Error e = store->remove(id);
            provider->touch();
            return e;
        };
        return CommandOutput::ok(std::move(result),
                                 UndoRecord::closures(std::move(undo), std::move(redo), recipe.size() + 256));
    }

    // ---- the open spec -----------------------------------------------------

    CommandOutput no_open_spec() {
        return fail("spec.no_open_spec", "No spec is open; spec.new starts one and applet.open opens a saved one.");
    }

    CommandOutput spec_new(const CommandContext& ctx) {
        std::lock_guard<std::mutex> lock(doc_mutex);
        Document next;
        next.id = next_doc++;
        next.spec = new_spec_from_args(ctx.invocation.args);
        spec::ValidationReport report = spec::validate(next.spec, resolver());
        if (Error e = first_spec_error(report.diagnostics)) {
            if (e.code == "spec.bad_id") e.code = "spec.invalid_id";
            return CommandOutput::fail(e);
        }
        const std::string text = spec::to_toml(next.spec);
        const std::string id = next.spec.applet.id;
        return CommandOutput::ok({{"id", id}, {"text", text}}, replace_document(std::move(next)));
    }

    CommandOutput open_loaded(LoadedSpec loaded, json result) {
        Document next;
        next.id = next_doc++;
        next.saved_text = spec::to_toml(loaded.spec);
        next.spec = std::move(loaded.spec);
        next.path = loaded.path;
        result["warnings"] = warnings_json(loaded.diagnostics);
        return CommandOutput::ok(std::move(result), replace_document(std::move(next)));
    }

    CommandOutput spec_open(const CommandContext& ctx) {
        std::lock_guard<std::mutex> lock(doc_mutex);
        LoadedSpec loaded = load_spec_file(str_arg(ctx.invocation.args, "path"), resolver());
        if (loaded.error) return CommandOutput::fail(loaded.error);
        json result{{"id", loaded.spec.applet.id}, {"path", loaded.path}};
        return open_loaded(std::move(loaded), std::move(result));
    }

    CommandOutput applet_open(const CommandContext& ctx) {
        std::lock_guard<std::mutex> lock(doc_mutex);
        const std::string id = str_arg(ctx.invocation.args, "id");
        LoadedSpec loaded = load_applet(id, resolver());
        if (loaded.error) return CommandOutput::fail(loaded.error);
        json result{{"id", id}, {"title", loaded.spec.applet.title.value_or(loaded.spec.applet.id)}};
        return open_loaded(std::move(loaded), std::move(result));
    }

    // The spec that the read-only commands of the browser's spec page read: a
    // saved applet's, read from its files without opening it, or else the
    // open spec or the preview the invocation names.
    Error view_spec(const std::string& applet, const std::string& preview, spec::Spec* out) {
        if (!applet.empty()) {
            LoadedSpec loaded = load_applet(applet, {});
            if (loaded.error) return Error(loaded.error.code, loaded.error.message, "args.applet");
            *out = std::move(loaded.spec);
            return {};
        }
        std::lock_guard<std::mutex> lock(doc_mutex);
        if (!doc) return no_open_spec().error;
        *out = doc->spec;
        Preview* p = nullptr;
        if (!preview.empty()) {
            if (Error e = use_preview(preview, "preview", &p)) return e;
            *out = p->shadow;
        }
        return {};
    }

    CommandOutput spec_get(const CommandContext& ctx) {
        const std::string applet = str_arg(ctx.invocation.args, "applet");
        if (!applet.empty()) {
            LoadedSpec loaded = load_applet(applet, {});
            if (loaded.error) return fail(loaded.error.code, loaded.error.message, "args.applet");
            return CommandOutput::ok({{"open", true},
                                      {"id", loaded.spec.applet.id},
                                      {"path", loaded.path},
                                      {"dirty", false},
                                      {"text", spec::to_toml(loaded.spec)},
                                      {"json", spec::to_json(loaded.spec)}});
        }
        std::lock_guard<std::mutex> lock(doc_mutex);
        // No spec open: a first look, answered as a read, not as a failure
        // (a preview needs an open spec, so naming one still fails).
        if (!doc && ctx.invocation.preview.empty()) return CommandOutput::ok({{"open", false}});
        if (!doc) return no_open_spec();
        const spec::Spec* s = &doc->spec;
        if (!ctx.invocation.preview.empty()) {
            Preview* p = nullptr;
            if (Error e = use_preview(ctx.invocation.preview, "preview", &p)) return CommandOutput::fail(e);
            s = &p->shadow;
        }
        const std::string text = spec::to_toml(*s);
        return CommandOutput::ok({{"open", true},
                                  {"id", s->applet.id},
                                  {"path", doc->path ? json(*doc->path) : json(nullptr)},
                                  {"dirty", !doc->path || text != doc->saved_text},
                                  {"text", text},
                                  {"json", spec::to_json(*s)}});
    }

    CommandOutput spec_validate(const CommandContext& ctx) {
        std::vector<spec::Diagnostic> diagnostics;
        if (ctx.invocation.args.contains("text")) {
            diagnostics = check_spec_text(str_arg(ctx.invocation.args, "text"), resolver()).diagnostics;
        } else {
            std::lock_guard<std::mutex> lock(doc_mutex);
            if (!doc) return no_open_spec();
            const spec::Spec* s = &doc->spec;
            if (!ctx.invocation.preview.empty()) {
                Preview* p = nullptr;
                if (Error e = use_preview(ctx.invocation.preview, "preview", &p)) return CommandOutput::fail(e);
                s = &p->shadow;
            }
            diagnostics = spec::validate(*s, resolver()).diagnostics;
        }
        json errors = json::array();
        for (const auto& d : diagnostics)
            if (d.is_error()) errors.push_back(diagnostic_json(d));
        const bool valid = errors.empty();
        return CommandOutput::ok({{"valid", valid}, {"errors", std::move(errors)}, {"warnings", warnings_json(diagnostics)}});
    }

    // A dry run, whether it succeeds or fails, and an edit that leaves the
    // spec as it was report no effect, so the registry treats them as reads:
    // the redo stack stays, no undo step is filed and the journal is not
    // written.
    CommandOutput spec_edit(const CommandContext& ctx) {
        std::lock_guard<std::mutex> lock(doc_mutex);
        const json& args = ctx.invocation.args;
        const bool dry = bool_arg(args, "dry_run", false);
        if (!ctx.invocation.preview.empty()) return preview_edit(ctx, dry);
        auto refused = [dry](CommandOutput out) {
            out.no_effect = dry;
            return out;
        };
        if (!doc) return refused(no_open_spec());
        // An edit the person reviewed runs only as it was shown.
        const std::optional<Reviewed> shown = dry ? std::nullopt : take_reviewed(ctx.invocation);
        if (shown && doc->id != shown->doc)
            return fail("history.conflict",
                        "Another spec was opened after the person was shown this change, so it was not applied; read "
                        "the open spec and send the edit again.");
        SpecEdit e = apply_spec_edit(doc->spec, ctx.invocation.name, args, resolver());
        if (e.error && shown)
            return fail("history.conflict",
                        "The open spec changed after the person was shown this change, and it no longer applies: " +
                            e.error.message + " (" + e.error.code + ")",
                        e.error.where);
        if (e.error) return refused(CommandOutput::fail(e.error));
        const std::string before_text = spec::to_toml(doc->spec);
        const std::string after_text = spec::to_toml(e.spec);
        if (shown && change_lines(before_text, after_text) != shown->change)
            return fail("history.conflict",
                        "The open spec changed after the person was shown this change, and the change it would make "
                        "now differs from the one they approved, so it was not applied; read the open spec and send "
                        "the edit again.");
        const std::string label = spec_label();
        json result = std::move(e.result);
        result["warnings"] = warnings_json(e.warnings);
        result["diff"] = unified_line_diff(before_text, after_text, "a/" + label, "b/" + label);
        result["applied"] = !dry;
        if (dry || e.spec == doc->spec) return CommandOutput::unchanged(std::move(result));
        const std::size_t bytes = 2 * (before_text.size() + after_text.size()) + 512;
        return CommandOutput::ok(std::move(result),
                                 change_spec(std::move(e.spec), !ctx.invocation.merge_key.empty(), bytes));
    }

    // ---- review of spec edits ---------------------------------------------------

    bool review_for(CommandOrigin o) const { return review_on[static_cast<std::size_t>(o)].load(); }

    // The lines a change removes and adds, in order, without the line numbers
    // and the context of a unified diff, so that the same change made on a
    // spec that changed elsewhere compares equal.
    static std::string change_lines(const std::string& before, const std::string& after) {
        const std::string diff = unified_line_diff(before, after, "a", "b", 0);
        std::string out;
        std::size_t start = 0;
        int line = 0;
        while (start < diff.size()) {
            std::size_t end = diff.find('\n', start);
            if (end == std::string::npos) end = diff.size();
            const std::string_view l(diff.data() + start, end - start);
            if (++line > 2 && l.rfind("@@", 0) != 0) {
                out.append(l);
                out += '\n';
            }
            start = end + 1;
        }
        return out;
    }

    // The review of an invocation, or null when it needs none: review is off
    // for its origin, it is not an edit of the open spec, it is a dry run or
    // names a preview, or it would fail or change nothing. A review is kept
    // for the handler, which runs the edit only as it was shown.
    json review_of(const CommandInvocation& inv) {
        if (!is_spec_edit_command(inv.name) || !review_for(inv.origin) || !inv.preview.empty() ||
            bool_arg(inv.args, "dry_run", false))
            return nullptr;
        Reviewed kept;
        json review;
        {
            std::lock_guard<std::mutex> lock(doc_mutex);
            if (!doc) return nullptr;
            SpecEdit e = apply_spec_edit(doc->spec, inv.name, inv.args, resolver());
            if (e.error || e.spec == doc->spec) return nullptr;
            const std::string before = spec::to_toml(doc->spec);
            const std::string after = spec::to_toml(e.spec);
            const std::string label = spec_label();
            review = {{"diff", unified_line_diff(before, after, "a/" + label, "b/" + label)},
                      {"warnings", warnings_json(e.warnings)}};
            kept = Reviewed{invocation_key(inv), doc->id, change_lines(before, after)};
        }
        std::lock_guard<std::mutex> lock(review_mutex);
        reviewed.push_back(std::move(kept));
        while (reviewed.size() > 256) reviewed.pop_front();
        last_review = {invocation_key(inv), review};
        return review;
    }

    // What the registry keeps with a parked invocation: the review the policy
    // just computed for it.
    json parked_review(const CommandInvocation& inv) {
        std::lock_guard<std::mutex> lock(review_mutex);
        if (last_review.first != invocation_key(inv)) return nullptr;
        json out = std::move(last_review.second);
        last_review = {};
        return out;
    }

    std::optional<Reviewed> take_reviewed(const CommandInvocation& inv) {
        const std::string key = invocation_key(inv);
        std::lock_guard<std::mutex> lock(review_mutex);
        const auto it = std::find_if(reviewed.begin(), reviewed.end(), [&](const Reviewed& r) { return r.key == key; });
        if (it == reviewed.end()) return std::nullopt;
        Reviewed out = std::move(*it);
        reviewed.erase(it);
        return out;
    }

    // ---- approvals bound to content -------------------------------------------

    // What an overwrite or a replacement would write or open, as its card
    // shows it: the diff and warnings, and the document and digest it is
    // bound to. `applies` is false when the invocation is not one of these
    // or would fail, so that it needs no card of this kind.
    struct Content {
        bool applies = false;
        std::uint64_t doc = 0;
        std::string digest;
        json review;
    };

    static bool replaces_document(const std::string& name) {
        return name == "spec.new" || name == "spec.open" || name == "applet.open";
    }

    Content content_of(const CommandInvocation& inv) {
        Content out;
        std::lock_guard<std::mutex> lock(doc_mutex);
        if (!doc) return out;
        std::string before;
        std::string after;
        std::string before_label;
        std::string after_label;
        std::vector<spec::Diagnostic> diagnostics;
        if (inv.name == "applet.save") {
            // The saved file against the open spec it would be overwritten with.
            const std::string id = str_arg(inv.args, "id", doc->spec.applet.id);
            if (!valid_id(id)) return out;
            spec::Spec saved = doc->spec;
            saved.applet.id = id;
            after = spec::to_toml(saved);
            const fs::path file = ws() / "applets" / fs::u8path(id) / fs::u8path(id + ".spec.toml");
            if (read_file_bounded(file.u8string(), spec::limits::kTextBytes, &before, "spec.too_large")) before.clear();
            before_label = after_label = "applets/" + id + "/" + id + ".spec.toml";
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
                                                            : load_applet(str_arg(inv.args, "id"), resolver());
                if (loaded.error) return out;
                next = std::move(loaded.spec);
                diagnostics = std::move(loaded.diagnostics);
            }
            before = spec::to_toml(doc->spec);
            after = spec::to_toml(next);
            before_label = spec_label();
            after_label = next.applet.id + ".spec.toml";
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
    json bind_content(const CommandInvocation& inv) {
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
    bool approval_holds(const CommandInvocation& inv) {
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

    CommandOutput review_status(const CommandContext& ctx) {
        const CommandOrigin o = ctx.invocation.origin;
        return CommandOutput::ok(
            {{"origin", command_origin_name(o)}, {"spec_edits", review_for(o)}, {"destructive", !user_origin(o)}});
    }

    // ---- stacked previews (all under doc_mutex) ------------------------------

    std::chrono::steady_clock::time_point preview_now() const {
        return options.preview.now ? options.preview.now() : std::chrono::steady_clock::now();
    }

    // Drops the previews left unused for longer than the idle limit.
    void sweep_previews() {
        const auto now = preview_now();
        for (auto it = previews.begin(); it != previews.end();) {
            if (now - it->second.used > options.preview.idle) {
                expired.push_back(it->first);
                while (expired.size() > 64) expired.pop_front();
                it = previews.erase(it);
            } else {
                ++it;
            }
        }
    }

    // Finds a preview, after dropping the expired ones, and marks it used.
    Error find_preview(const std::string& id, const std::string& where, Preview** out) {
        sweep_previews();
        const auto it = previews.find(id);
        if (it == previews.end()) {
            if (std::find(expired.begin(), expired.end(), id) != expired.end()) {
                const auto minutes = std::chrono::duration_cast<std::chrono::minutes>(options.preview.idle).count();
                return Error("preview.expired",
                             "The preview " + id + " was dropped after " + std::to_string(minutes) +
                                 " minutes without use; preview.begin starts a new one.",
                             where);
            }
            return Error("preview.not_found", "No preview has the id " + id + "; preview.begin starts one.", where);
        }
        it->second.used = preview_now();
        *out = &it->second;
        return {};
    }

    // Makes the preview's shadow the open spec with its stack applied again,
    // when the open spec changed since the shadow was computed.
    Error refresh_preview(Preview& p) {
        if (!doc || doc->id != p.doc)
            return Error("history.conflict", "The preview " + p.id +
                                                 " was made on a spec that is no longer open; discard it and begin a "
                                                 "new one on the open spec.");
        if (doc->spec == p.base) return {};
        spec::Spec s = doc->spec;
        const spec::SourceResolver res = resolver();
        for (const CommandInvocation& inv : p.stack) {
            SpecEdit e = apply_spec_edit(s, inv.name, inv.args, res);
            if (e.error)
                return Error("history.conflict",
                             "The open spec changed since the preview's edits were made, and its " + inv.name +
                                 " no longer applies: " + e.error.message + " (" + e.error.code + ")",
                             e.error.where);
            s = std::move(e.spec);
        }
        p.base = doc->spec;
        p.shadow = std::move(s);
        return {};
    }

    Error use_preview(const std::string& id, const std::string& where, Preview** out) {
        if (Error e = find_preview(id, where, out)) return e;
        return refresh_preview(**out);
    }

    // The rule of a scoped preview: it takes new sources and panels, and
    // changes only its target, the sources a target panel binds (following
    // `from`) and the items it added itself.
    Error check_scope(const Preview& p, const std::string& name, const json& args) const {
        if (p.panel.empty() && p.source.empty()) return {};
        const std::string target = p.panel.empty() ? "source " + p.source : "panel " + p.panel;
        if (name == "spec.set_text" || name == "layout.set")
            return Error("preview.out_of_scope",
                         "A preview of the " + target + " cannot replace the whole spec or the whole layout; change "
                                                        "the target with the other edits.",
                         "name");
        if (name == "source.add" || name == "panel.create") return {};
        std::set<std::string> panels;
        std::set<std::string> sources;
        if (!p.panel.empty()) panels.insert(p.panel);
        if (!p.source.empty()) sources.insert(p.source);
        for (const CommandInvocation& inv : p.stack) {
            if (inv.name == "panel.create") panels.insert(str_arg(inv.args, "id"));
            if (inv.name == "source.add") sources.insert(str_arg(inv.args, "id"));
        }
        if (const spec::Panel* panel = p.panel.empty() ? nullptr : p.shadow.find_panel(p.panel)) {
            std::vector<std::string> todo;
            for (const auto& slot : panel->bind)
                for (const std::string& b : slot.second.bindings) todo.push_back(b.substr(0, b.find('.')));
            while (!todo.empty() && sources.size() < 4096) {
                const std::string id = todo.back();
                todo.pop_back();
                if (!sources.insert(id).second) continue;
                const spec::Source* src = p.shadow.find_source(id);
                if (src && src->from) todo.push_back(*src->from);
            }
        }
        const bool panel_edit = name == "panel.patch" || name == "panel.remove" || name == "layout.place";
        const std::string id = str_arg(args, name == "layout.place" ? "panel" : "id");
        if (panel_edit ? panels.count(id) : sources.count(id)) return {};
        return Error("preview.out_of_scope",
                     "This preview is about the " + target + ", so it changes only that target, the sources it "
                     "binds and what the preview added; " + id + " is outside it.",
                     name == "layout.place" ? "args.panel" : "args.id");
    }

    // An edit that names a preview: it applies to the shadow and reports no
    // effect, so the registry neither journals it nor files a step.
    CommandOutput preview_edit(const CommandContext& ctx, bool dry) {
        auto quiet = [](CommandOutput out) {
            out.no_effect = true;
            return out;
        };
        if (!doc) return quiet(no_open_spec());
        Preview* p = nullptr;
        if (Error e = use_preview(ctx.invocation.preview, "preview", &p)) return quiet(CommandOutput::fail(e));
        const std::string& name = ctx.invocation.name;
        const json& args = ctx.invocation.args;
        if (Error e = check_scope(*p, name, args)) return quiet(CommandOutput::fail(e));
        SpecEdit e = apply_spec_edit(p->shadow, name, args, resolver());
        if (e.error) return quiet(CommandOutput::fail(e.error));
        const std::string label = spec_label();
        json result = std::move(e.result);
        result["warnings"] = warnings_json(e.warnings);
        result["diff"] = unified_line_diff(spec::to_toml(p->shadow), spec::to_toml(e.spec), "a/" + label, "b/" + label);
        result["applied"] = !dry;
        if (dry || e.spec == p->shadow) return CommandOutput::unchanged(std::move(result));
        CommandInvocation kept;
        kept.id = ctx.invocation.id;
        kept.name = name;
        kept.args = args;
        kept.args.erase("dry_run");
        kept.origin = ctx.invocation.origin;
        kept.merge_key = ctx.invocation.merge_key;
        const std::size_t bytes = kept.args.dump(-1, ' ', false, json::error_handler_t::replace).size() + name.size();
        // A run with one merge key sets the same thing each time, so its
        // newest edit takes the place of the one before.
        const bool merges = !kept.merge_key.empty() && !p->stack.empty() && p->stack.back().merge_key == kept.merge_key &&
                            p->stack.back().name == name;
        const std::size_t replaced = merges ? p->stack.back().args.dump(-1, ' ', false, json::error_handler_t::replace).size() +
                                                  name.size()
                                            : 0;
        if (!merges && p->stack.size() >= options.preview.max_edits)
            return quiet(fail("preview.too_many_edits",
                              "The preview holds " + std::to_string(p->stack.size()) +
                                  " edits, the most it takes; it can be committed or discarded, or a new one begun.",
                              "preview"));
        if (p->bytes - replaced + bytes > options.preview.max_bytes)
            return quiet(fail("preview.too_large",
                              "The preview's edits would hold more than " +
                                  std::to_string(options.preview.max_bytes >> 10) + " KiB of arguments.",
                              "preview"));
        if (merges) {
            p->bytes -= replaced;
            p->stack.back() = std::move(kept);
        } else {
            p->stack.push_back(std::move(kept));
        }
        p->bytes += bytes;
        p->shadow = std::move(e.spec);
        return CommandOutput::unchanged(std::move(result));
    }

    static json scope_json(const std::string& id) { return id.empty() ? json(nullptr) : json(id); }

    CommandOutput preview_begin(const CommandContext& ctx) {
        std::lock_guard<std::mutex> lock(doc_mutex);
        if (!doc) return no_open_spec();
        sweep_previews();
        const json& a = ctx.invocation.args;
        std::string id = str_arg(a, "id");
        const std::string panel = str_arg(a, "panel");
        const std::string source = str_arg(a, "source");
        if (!panel.empty() && !source.empty())
            return fail("preview.bad_scope", "A preview is about a panel or a source, not both.", "args.source");
        if (!panel.empty() && !doc->spec.find_panel(panel))
            return fail("panel.not_found", "The open spec has no panel " + panel + "; spec.get lists them.", "args.panel");
        if (!source.empty() && !doc->spec.find_source(source))
            return fail("source.not_found", "The open spec has no source " + source + "; spec.get lists them.",
                        "args.source");
        if (!id.empty() && previews.count(id))
            return fail("preview.exists", "A preview " + id + " is open already; preview.discard ends it.", "args.id");
        if (previews.size() >= options.preview.max_previews)
            return fail("preview.too_many", std::to_string(previews.size()) +
                                                " previews are open, the most a workspace keeps; commit or discard "
                                                "one first.");
        while (id.empty() || previews.count(id)) id = "preview-" + std::to_string(next_preview++);
        expired.erase(std::remove(expired.begin(), expired.end(), id), expired.end());
        Preview p;
        p.id = id;
        p.doc = doc->id;
        p.base = doc->spec;
        p.shadow = doc->spec;
        p.panel = panel;
        p.source = source;
        p.used = preview_now();
        previews.emplace(id, std::move(p));
        return CommandOutput::ok({{"id", id}, {"panel", scope_json(panel)}, {"source", scope_json(source)}});
    }

    CommandOutput preview_diff(const CommandContext& ctx) {
        std::lock_guard<std::mutex> lock(doc_mutex);
        if (!doc) return no_open_spec();
        Preview* p = nullptr;
        if (Error e = use_preview(str_arg(ctx.invocation.args, "id"), "args.id", &p)) return CommandOutput::fail(e);
        const std::string label = spec_label();
        return CommandOutput::ok(
            {{"id", p->id},
             {"edits", p->stack.size()},
             {"diff", unified_line_diff(spec::to_toml(doc->spec), spec::to_toml(p->shadow), "a/" + label, "b/" + label)},
             {"warnings", warnings_json(spec::validate(p->shadow, resolver()).diagnostics)},
             {"panel", scope_json(p->panel)},
             {"source", scope_json(p->source)}});
    }

    // Accept: the person's act, so only ui and builder commit. The edits run
    // as the registry's follow-ups, after being applied again to the open
    // spec here, so a stack that no longer applies changes nothing.
    CommandOutput preview_commit(const CommandContext& ctx) {
        if (!user_origin(ctx.invocation.origin))
            return fail("preview.not_allowed",
                        "Only the person commits a preview, with Accept in Caliper; it stays open for them to accept "
                        "or reject.");
        std::lock_guard<std::mutex> lock(doc_mutex);
        if (!doc) return no_open_spec();
        Preview* p = nullptr;
        if (Error e = use_preview(str_arg(ctx.invocation.args, "id"), "args.id", &p)) return CommandOutput::fail(e);
        const std::string label = spec_label();
        json result{
            {"id", p->id},
            {"edits", p->stack.size()},
            {"diff", unified_line_diff(spec::to_toml(doc->spec), spec::to_toml(p->shadow), "a/" + label, "b/" + label)},
            {"warnings", warnings_json(spec::validate(p->shadow, resolver()).diagnostics)}};
        std::vector<CommandInvocation> edits = std::move(p->stack);
        const std::string group = p->id;
        previews.erase(group);
        if (edits.empty()) return CommandOutput::unchanged(std::move(result));
        for (CommandInvocation& inv : edits) {
            inv.id.clear();
            inv.group = group;
            inv.merge_key.clear();
            inv.preview.clear();
        }
        CommandOutput out = CommandOutput::ok(std::move(result));
        out.follow_ups = std::move(edits);
        return out;
    }

    CommandOutput preview_discard(const CommandContext& ctx) {
        std::lock_guard<std::mutex> lock(doc_mutex);
        Preview* p = nullptr;
        if (Error e = find_preview(str_arg(ctx.invocation.args, "id"), "args.id", &p)) return CommandOutput::fail(e);
        const std::string id = p->id;
        json result{{"id", id}, {"edits", p->stack.size()}};
        previews.erase(id);
        return CommandOutput::ok(std::move(result));
    }

    // ---- applets -----------------------------------------------------------

    CommandOutput applet_save(const CommandContext& ctx) {
        std::lock_guard<std::mutex> lock(doc_mutex);
        if (!doc) return no_open_spec();
        const json& a = ctx.invocation.args;
        const std::string id = str_arg(a, "id", doc->spec.applet.id);
        const bool overwrite = bool_arg(a, "overwrite", false);
        if (Error e = first_spec_error(spec::validate(doc->spec, resolver()).diagnostics)) return CommandOutput::fail(e);
        if (!valid_id(id)) return fail("spec.invalid_id", "An applet id matches [a-z][a-z0-9_-]{0,63}.", "args.id");

        const fs::path folder = ws() / "applets" / fs::u8path(id);
        const fs::path manifest = folder / fs::u8path(id + ".caliper.toml");
        const fs::path spec_file = folder / fs::u8path(id + ".spec.toml");
        const bool exists = applet_saved(id);
        if (exists && !overwrite)
            return fail("applet.exists", "An applet " + id + " is saved already; set overwrite to replace it.", "args.id");
        const std::string manifest_id = manifest_id_for(id);
        std::error_code ec;
        for (fs::directory_iterator it(ws() / "applets", ec), end; !ec && it != end; it.increment(ec)) {
            const std::string other = it->path().filename().u8string();
            if (other == id || !valid_id(other)) continue;
            const fs::path m = it->path() / fs::u8path(other + ".caliper.toml");
            std::error_code mec;
            if (!fs::is_regular_file(m, mec)) continue;
            if (read_manifest(m.u8string()).id == manifest_id)
                return fail("applet.exists",
                            "The applet " + other + " already has the manifest id " + manifest_id + "; choose another id.",
                            "args.id");
        }
        std::string version = "0.1.0";
        if (exists) {
            ManifestInfo old = read_manifest(manifest.u8string());
            if (!old.error && !old.version.empty()) version = old.version;
        }
        spec::Spec saved = doc->spec;
        saved.applet.id = id;
        const std::string text = spec::to_toml(saved);
        std::optional<std::string> summary;
        if (saved.applet.description && saved.applet.description->find_first_of("\r\n") == std::string::npos &&
            command_code_points(*saved.applet.description) <= 200)
            summary = *saved.applet.description;
        const std::string name = saved.applet.title.value_or(id);
        if (Error e = write_file_atomic(spec_file.u8string(), text))
            return fail("applet.write_failed", "The spec file could not be written: " + e.message, rel(spec_file));
        if (Error e = write_file_atomic(manifest.u8string(),
                                        manifest_text(manifest_id, name, version, summary, id + ".spec.toml")))
            return fail("applet.write_failed", "The manifest could not be written: " + e.message, rel(manifest));
        // Saving under the spec's own id makes the files the open spec's
        // home; saving under another id writes a copy and leaves it as it was.
        if (id == doc->spec.applet.id) {
            doc->path = rel(spec_file);
            doc->saved_text = text;
        }
        return CommandOutput::ok({{"id", id}, {"manifest", rel(manifest)}, {"spec", rel(spec_file)}});
    }

    CommandOutput applet_list(const CommandContext&) {
        std::vector<json> found;
        std::error_code ec;
        std::size_t seen = 0;
        for (fs::directory_iterator it(ws() / "applets", ec), end; !ec && it != end; it.increment(ec)) {
            if (++seen > kMaxAppletFolders) break;
            const std::string id = it->path().filename().u8string();
            std::error_code dec;
            if (!valid_id(id) || !it->is_directory(dec)) continue;
            const fs::path manifest = it->path() / fs::u8path(id + ".caliper.toml");
            if (!fs::is_regular_file(manifest, dec)) continue;
            ManifestInfo m = read_manifest(manifest.u8string());
            if (m.error || m.kind != "spec" || !plain_file_name(m.spec)) continue;
            auto updated = file_mtime_utc((it->path() / fs::u8path(m.spec)).u8string());
            if (!updated) updated = file_mtime_utc(manifest.u8string());
            found.push_back({{"id", id},
                             {"title", m.name.empty() ? id : m.name},
                             {"path", rel(it->path())},
                             {"updated_at", updated ? json(*updated) : json(nullptr)}});
        }
        std::sort(found.begin(), found.end(), [](const json& x, const json& y) { return x["id"] < y["id"]; });
        return CommandOutput::ok({{"applets", json(found)}});
    }

    // One panel's data for the browser's spec page (panel_data.h); with
    // `known` equal to the version, nothing is resolved.
    CommandOutput panel_data(const CommandContext& ctx) {
        const json& args = ctx.invocation.args;
        spec::Spec s;
        if (Error e = view_spec(str_arg(args, "applet"), ctx.invocation.preview, &s)) return CommandOutput::fail(e);
        const std::string id = str_arg(args, "panel");
        if (!s.find_panel(id)) return fail("panel.not_found", "The spec has no panel " + id + "; spec.get lists them.", "args.panel");
        const std::uint64_t revision = catalog->revision();
        const std::string version = panel_data_version(s, id, revision, panel_generation(s, id, sources));
        if (str_arg(args, "known") == version) return CommandOutput::ok(panel_data_unchanged(s, id, version));
        return CommandOutput::ok(panel_data_result(s, resolve_panel(s, id, *store, options.resolve, &sources), revision));
    }

    // ---- the app's commands ------------------------------------------------

    CommandOutput panel_render(const CommandContext& ctx) {
        std::shared_ptr<SessionHost> h = current_host();
        if (!h) return fail("command.not_bound", "panel.render needs the app.");
        spec::Spec s;
        {
            std::lock_guard<std::mutex> lock(doc_mutex);
            if (!doc) return no_open_spec();
            s = doc->spec;
            if (!ctx.invocation.preview.empty()) {
                Preview* p = nullptr;
                if (Error e = use_preview(ctx.invocation.preview, "preview", &p)) return CommandOutput::fail(e);
                s = p->shadow;
            }
        }
        const std::string id = str_arg(ctx.invocation.args, "id");
        if (!s.find_panel(id))
            return fail("panel.not_found",
                        std::string("The ") + (ctx.invocation.preview.empty() ? "open spec" : "preview") +
                            " has no panel " + id + ".",
                        "args.id");
        // A resolve error goes to the host with the data, which draws it as the
        // panel's state ("Source offline", "Missing column"), as the app shows it.
        PanelData data = resolve_panel(s, id, *store, options.resolve, &sources);
        return h->render_panel(s, data, static_cast<int>(int_arg(ctx.invocation.args, "width", 800)),
                               static_cast<int>(int_arg(ctx.invocation.args, "height", 500)));
    }

    void bind_all() {
        auto bind = [this](const char* name, CommandOutput (Impl::*fn)(const CommandContext&)) {
            registry->bind(name, [this, fn](const CommandContext& ctx) {
                // In a preview only the edits of the open spec run; any other
                // change of the workspace is refused, quietly.
                const std::string& n = ctx.invocation.name;
                if (!ctx.invocation.preview.empty() && ctx.descriptor.mutates && !is_spec_edit_command(n) &&
                    n != "preview.commit") {
                    CommandOutput out = fail("preview.not_previewable",
                                             n + " changes the workspace beyond the open spec, so it cannot run in a "
                                                 "preview; send it without the preview.",
                                             "preview");
                    out.no_effect = true;
                    return out;
                }
                return (this->*fn)(ctx);
            });
        };
        bind("catalog.list", &Impl::catalog_list);
        bind("catalog.describe", &Impl::catalog_describe);
        bind("catalog.sample", &Impl::catalog_sample);
        bind("data.query", &Impl::data_query);
        bind("data.import", &Impl::data_import);
        bind("loader.list", &Impl::loader_list);
        bind("loader.describe", &Impl::loader_describe);
        bind("loader.create", &Impl::loader_create);
        bind("loader.update", &Impl::loader_update);
        bind("loader.run", &Impl::loader_run);
        bind("loader.delete", &Impl::loader_delete);
        bind("spec.new", &Impl::spec_new);
        bind("spec.open", &Impl::spec_open);
        bind("spec.get", &Impl::spec_get);
        bind("spec.validate", &Impl::spec_validate);
        for (const char* name : {"spec.set_text", "source.add", "source.patch", "source.remove", "panel.create",
                                 "panel.patch", "panel.remove", "layout.place", "layout.set"})
            bind(name, &Impl::spec_edit);
        bind("preview.begin", &Impl::preview_begin);
        bind("preview.diff", &Impl::preview_diff);
        bind("preview.commit", &Impl::preview_commit);
        bind("preview.discard", &Impl::preview_discard);
        bind("applet.save", &Impl::applet_save);
        bind("applet.open", &Impl::applet_open);
        bind("applet.list", &Impl::applet_list);
        bind("panel.data", &Impl::panel_data);
        bind("review.status", &Impl::review_status);
    }

    static std::string invocation_key(const CommandInvocation& inv) {
        return inv.id + "\n" + inv.name + "\n" + inv.args.dump(-1, ' ', false, json::error_handler_t::replace);
    }

    static bool is_python(const Loader& l) {
        const auto r = declared_reader(l);
        return r && *r == ReaderKind::python;
    }

    // Whether an invocation runs a Python loader's script: loader.run of one,
    // or loader.create or loader.update of one with run set.
    bool runs_python(const CommandInvocation& inv) {
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
    bool python_approved(const CommandInvocation& inv) {
        if (user_origin(inv.origin)) return true;
        const std::string key = invocation_key(inv);
        std::lock_guard<std::mutex> lock(python_mutex);
        const auto it = std::find(python_asked.begin(), python_asked.end(), key);
        if (it == python_asked.end()) return false;
        python_asked.erase(it);
        return true;
    }

    // ---- agent permissions -----------------------------------------------------

    bool is_bound(const std::string& key) {
        std::lock_guard<std::mutex> lock(review_mutex);
        return std::any_of(bound.begin(), bound.end(), [&](const Bound& b) { return b.key == key; });
    }

    bool python_pending(const std::string& key) {
        std::lock_guard<std::mutex> lock(python_mutex);
        return std::find(python_asked.begin(), python_asked.end(), key) != python_asked.end();
    }

    // The saved rules' agent of a principal: "mcp", or the chat's agent id.
    std::string rule_agent(AgentPrincipal who) const {
        if (who == AgentPrincipal::mcp) return "mcp";
        if (who == AgentPrincipal::chat && chat_agent != "mcp" && valid_rule_agent(chat_agent)) return chat_agent;
        return {};
    }

    AgentMode mode_of(AgentPrincipal who) const {
        if (who == AgentPrincipal::chat) return chat_allow_all ? AgentMode::allow_all : perms.chat;
        if (who == AgentPrincipal::mcp) return perms.mcp;
        return AgentMode::ask;
    }

    Error save_perms() {
        if (options.permissions_file.empty()) return {};
        return write_agent_permissions(options.permissions_file, perms);
    }

    // The session's policy with the agent's mode and rules applied, and the
    // note the registry asks for next.
    ApprovalDecision policy(const CommandDescriptor& d, const CommandInvocation& inv) {
        const ApprovalDecision base = base_policy(d, inv);
        const AgentPrincipal who = agent_principal(inv);
        if (who == AgentPrincipal::none) return base;
        const std::string key = invocation_key(inv);
        PermissionContext c;
        c.bound = base == ApprovalDecision::ask && is_bound(key);
        c.python = base == ApprovalDecision::ask && python_pending(key);
        std::lock_guard<std::mutex> lock(perm_mutex);
        c.mode = mode_of(who);
        c.session_rule = (who == AgentPrincipal::chat ? chat_session_rules : mcp_session_rules).count(inv.name) > 0;
        const std::string agent = rule_agent(who);
        c.saved_rule = !agent.empty() && std::find(perms.rules.begin(), perms.rules.end(),
                                                   AgentRule{agent, inv.name}) != perms.rules.end();
        const PermissionVerdict v = decide_agent_permission(d, inv, base, c);
        last_note = {key, ApprovalNote{v.message, v.allowed_by}};
        return v.decision;
    }

    ApprovalNote note_of(const CommandInvocation& inv) {
        const std::string key = invocation_key(inv);
        std::lock_guard<std::mutex> lock(perm_mutex);
        if (last_note.first != key) return {};
        ApprovalNote out = std::move(last_note.second);
        last_note = {};
        return out;
    }

    ApprovalDecision base_policy(const CommandDescriptor& d, const CommandInvocation& inv) {
        if (!user_origin(inv.origin) && runs_python(inv)) {
            // Running a script runs the user's code, so every such invocation
            // asks, whatever the command's flags say.
            std::lock_guard<std::mutex> lock(python_mutex);
            python_asked.push_back(invocation_key(inv));
            while (python_asked.size() > 256) python_asked.pop_front();
            return ApprovalDecision::ask;
        }
        if (inv.name == "applet.save" && !user_origin(inv.origin)) {
            // Writing a new applet destroys nothing; overwriting one asks.
            if (!bool_arg(inv.args, "overwrite", false)) return ApprovalDecision::allow;
            std::string id = str_arg(inv.args, "id");
            if (id.empty()) {
                std::lock_guard<std::mutex> lock(doc_mutex);
                if (doc) id = doc->spec.applet.id;
            }
            if (!applet_saved(id)) return ApprovalDecision::allow;
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
        return default_approval_policy(d, inv);
    }
};

Session::Session() : impl_(std::make_unique<Impl>()) {}

Session::~Session() { close(); }

Session::Open Session::open(const std::string& workspace, SessionOptions options) {
    Open out;
    try {
        std::unique_ptr<Session> s(new Session());
        Impl& im = *s->impl_;
        im.options = std::move(options);
        StoreOpen opened = LoaderStore::open(workspace, im.options.store);
        if (opened.error) {
            out.error = opened.error;
            return out;
        }
        im.store = std::move(opened.store);
        im.workspace = im.store->workspace();
        std::error_code ec;
        fs::create_directories(im.ws() / "applets", ec);
        im.provider = std::make_shared<LoaderCatalogProvider>(*im.store);
        im.catalog = std::make_unique<Catalog>();
        if (Error e = im.catalog->add_provider("loaders", im.provider, 10)) {
            out.error = e;
            return out;
        }
        CommandJournal::Open j = CommandJournal::open((im.ws() / "journal" / "commands.jsonl").u8string(), im.options.journal);
        if (j.error) {
            out.error = j.error;
            return out;
        }
        im.journal = std::move(j.journal);
        im.registry = std::make_unique<CommandRegistry>(im.options.registry);
        if (Error e = im.registry->add_all(command_table())) {
            out.error = e;
            return out;
        }
        im.registry->set_journal(im.journal.get());
        Impl* raw = &im;
        im.registry->set_approval_policy(
            [raw](const CommandDescriptor& d, const CommandInvocation& inv) { return raw->policy(d, inv); });
        im.registry->set_approval_review(
            [raw](const CommandDescriptor&, const CommandInvocation& inv) { return raw->parked_review(inv); });
        im.registry->set_approval_check(
            [raw](const CommandDescriptor&, const CommandInvocation& inv) { return raw->approval_holds(inv); });
        im.registry->set_approval_notes(
            [raw](const CommandDescriptor&, const CommandInvocation& inv) { return raw->note_of(inv); });
        if (!im.options.permissions_file.empty()) {
            AgentPermissionsRead read = read_agent_permissions(im.options.permissions_file);
            im.perms = std::move(read.file);
            im.perm_problems = std::move(read.problems);
        }
        for (CommandOrigin o : {CommandOrigin::ui, CommandOrigin::builder, CommandOrigin::agent, CommandOrigin::mcp,
                                CommandOrigin::api})
            im.review_on[static_cast<std::size_t>(o)].store(im.options.review.on(o));
        im.bind_all();
        im.queue = std::make_unique<CommandQueue>(*im.registry, im.options.queue);
        out.session = std::move(s);
    } catch (const std::exception& e) {
        out.error = Error("session.open_failed", std::string("The workspace could not be opened: ") + e.what(), workspace);
    }
    return out;
}

const std::string& Session::workspace() const { return impl_->workspace; }
CommandRegistry& Session::registry() { return *impl_->registry; }
CommandQueue& Session::queue() { return *impl_->queue; }
LoaderStore& Session::store() { return *impl_->store; }
Catalog& Session::catalog() { return *impl_->catalog; }
LoaderCatalogProvider& Session::loader_catalog() { return *impl_->provider; }

CommandResult Session::invoke(CommandInvocation invocation) {
    {
        std::lock_guard<std::mutex> lock(impl_->close_mutex);
        if (impl_->closed)
            return CommandResult::failure(invocation.id, Error("command.shutdown", "The session is closed."));
    }
    return impl_->registry->invoke(std::move(invocation));
}

void Session::set_host(std::shared_ptr<SessionHost> host) {
    Impl& im = *impl_;
    {
        std::lock_guard<std::mutex> lock(im.host_mutex);
        im.host = host;
    }
    if (!host) {
        for (const auto& name : session_host_commands()) im.registry->bind(name, {});
        return;
    }
    Impl* raw = &im;
    im.registry->bind("app.info", [raw](const CommandContext&) {
        auto h = raw->current_host();
        return h ? h->app_info() : fail("command.not_bound", "app.info needs the app.");
    });
    im.registry->bind("app.applets", [raw](const CommandContext&) {
        auto h = raw->current_host();
        return h ? h->app_applets() : fail("command.not_bound", "app.applets needs the app.");
    });
    im.registry->bind("app.launch", [raw](const CommandContext& ctx) {
        auto h = raw->current_host();
        return h ? h->app_launch(str_arg(ctx.invocation.args, "id"), ctx.invocation.origin)
                 : fail("command.not_bound", "app.launch needs the app.");
    });
    im.registry->bind("capture.frame", [raw](const CommandContext&) {
        auto h = raw->current_host();
        return h ? h->capture_frame() : fail("command.not_bound", "capture.frame needs the app.");
    });
    im.registry->bind("panel.render", [raw](const CommandContext& ctx) { return raw->panel_render(ctx); });
}

void Session::set_review(CommandOrigin origin, bool on) {
    impl_->review_on[static_cast<std::size_t>(origin)].store(on);
}

AgentMode Session::agent_mode(AgentPrincipal who) const {
    std::lock_guard<std::mutex> lock(impl_->perm_mutex);
    return impl_->mode_of(who);
}

AgentMode Session::saved_agent_mode(AgentPrincipal who) const {
    std::lock_guard<std::mutex> lock(impl_->perm_mutex);
    return who == AgentPrincipal::chat ? impl_->perms.chat : who == AgentPrincipal::mcp ? impl_->perms.mcp : AgentMode::ask;
}

Error Session::set_agent_mode(AgentPrincipal who, AgentMode mode) {
    Impl& im = *impl_;
    std::lock_guard<std::mutex> lock(im.perm_mutex);
    if (who == AgentPrincipal::none) return Error("permissions.no_principal", "Only the side chat's agent and mcp clients have a mode.");
    if (mode == AgentMode::allow_all) {
        if (who != AgentPrincipal::chat)
            return Error("permissions.allow_all_chat_only", "Allow all is only for the side chat's agent, for its session.");
        im.chat_allow_all = true;  // never saved
        return {};
    }
    if (who == AgentPrincipal::chat) {
        im.chat_allow_all = false;
        im.perms.chat = mode;
    } else {
        im.perms.mcp = mode;
    }
    return im.save_perms();
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
    impl_->chat_allow_all = false;
    impl_->chat_session_rules.clear();
}

bool Session::approval_offers_rules(const CommandInvocation& parked) const {
    const AgentPrincipal who = agent_principal(parked);
    if (who == AgentPrincipal::none || !rule_command_allowed(parked.name)) return false;
    const std::string key = Impl::invocation_key(parked);
    if (impl_->is_bound(key) || impl_->python_pending(key)) return false;
    std::lock_guard<std::mutex> lock(impl_->perm_mutex);
    return who == AgentPrincipal::mcp || !impl_->rule_agent(who).empty();
}

Error Session::allow_always(const CommandInvocation& parked, RuleScope scope) {
    if (!approval_offers_rules(parked))
        return Error("permissions.no_rule", "This approval cannot become a rule; it is decided on its card each time.",
                     parked.name);
    Impl& im = *impl_;
    const AgentPrincipal who = agent_principal(parked);
    std::lock_guard<std::mutex> lock(im.perm_mutex);
    if (scope == RuleScope::session) {
        (who == AgentPrincipal::chat ? im.chat_session_rules : im.mcp_session_rules).insert(parked.name);
        return {};
    }
    const AgentRule rule{im.rule_agent(who), parked.name};
    if (std::find(im.perms.rules.begin(), im.perms.rules.end(), rule) == im.perms.rules.end()) im.perms.rules.push_back(rule);
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
    im.save_perms();
    return true;
}

std::vector<std::string> Session::permission_problems() const {
    std::lock_guard<std::mutex> lock(impl_->perm_mutex);
    return impl_->perm_problems;
}

ReviewSettings Session::review() const {
    ReviewSettings out;
    for (CommandOrigin o : {CommandOrigin::ui, CommandOrigin::builder, CommandOrigin::agent, CommandOrigin::mcp,
                            CommandOrigin::api})
        out.set(o, impl_->review_for(o));
    return out;
}

std::optional<OpenSpec> Session::open_spec() const {
    std::lock_guard<std::mutex> lock(impl_->doc_mutex);
    if (!impl_->doc) return std::nullopt;
    const Document& d = *impl_->doc;
    return OpenSpec{d.spec, d.path, d.dirty(), spec::to_toml(d.spec)};
}

spec::SourceResolver Session::source_resolver() const { return impl_->resolver(); }

PanelData Session::resolve(std::string_view panel_id, std::string_view preview) const {
    spec::Spec s;
    {
        std::lock_guard<std::mutex> lock(impl_->doc_mutex);
        PanelData out;
        out.panel = std::string(panel_id);
        if (!impl_->doc) {
            out.error = Error("spec.no_open_spec", "No spec is open.");
            return out;
        }
        s = impl_->doc->spec;
        if (!preview.empty()) {
            Preview* p = nullptr;
            if (Error e = impl_->use_preview(std::string(preview), "preview", &p)) {
                out.error = e;
                return out;
            }
            s = p->shadow;
        }
    }
    return resolve_panel(s, panel_id, *impl_->store, impl_->options.resolve, &impl_->sources);
}

PreviewState Session::preview(std::string_view id) const {
    PreviewState out;
    out.id = std::string(id);
    std::lock_guard<std::mutex> lock(impl_->doc_mutex);
    if (!impl_->doc) {
        out.error = Error("spec.no_open_spec", "No spec is open.");
        return out;
    }
    Preview* p = nullptr;
    if (Error e = impl_->use_preview(out.id, "preview", &p)) {
        out.error = e;
        return out;
    }
    const Document& d = *impl_->doc;
    const std::string text = spec::to_toml(p->shadow);
    out.spec = OpenSpec{p->shadow, d.path, !d.path || text != d.saved_text, text};
    out.edits = p->stack.size();
    out.panel = p->panel;
    out.source = p->source;
    return out;
}

DataSources& Session::data_sources() { return impl_->sources; }

void Session::close() {
    {
        std::lock_guard<std::mutex> lock(impl_->close_mutex);
        if (impl_->closed) return;
        impl_->closed = true;
    }
    if (impl_->queue) impl_->queue->shutdown();
    if (impl_->registry) impl_->registry->set_journal(nullptr);
    if (impl_->journal) impl_->journal->close();
}

SpecReplay replay_spec_journal(const std::string& workspace) {
    SpecReplay out;
    try {
        std::error_code ec;
        const fs::path canonical = fs::canonical(fs::u8path(workspace), ec);
        if (ec) {
            out.error = Error("session.io", "The workspace folder does not exist.", workspace);
            return out;
        }
        const JournalRead read = read_journal((canonical / "journal" / "commands.jsonl").u8string());
        out.problems = read.problems;
        if (read.error) {
            out.error = read.error;
            return out;
        }
        // A session-less Impl serves the file reading of spec.open and applet.open.
        Session::Impl im;
        im.workspace = canonical.u8string();
        const spec::SourceResolver none;
        std::optional<spec::Spec> current;
        auto apply = [&](const JournalEvent& e) -> Error {
            const json& args = e.args;
            if (e.name == "spec.new") {
                current = new_spec_from_args(args);
                return {};
            }
            if (e.name == "spec.open" || e.name == "applet.open") {
                LoadedSpec loaded = e.name == "spec.open" ? im.load_spec_file(str_arg(args, "path"), none)
                                                          : im.load_applet(str_arg(args, "id"), none);
                if (loaded.error) return loaded.error;
                current = std::move(loaded.spec);
                return {};
            }
            if (!is_spec_edit_command(e.name) || bool_arg(args, "dry_run", false)) return {};
            if (!current) return Error("journal.apply_failed", e.name + " ran with no open spec.");
            SpecEdit edit = apply_spec_edit(*current, e.name, args, none);
            if (edit.error) return edit.error;
            current = std::move(edit.spec);
            return {};
        };
        JournalReplay r = replay_journal(read.events, apply);
        out.applied = std::move(r.applied);
        out.error = r.error;
        out.spec = std::move(current);
    } catch (const std::exception& e) {
        out.error = Error("journal.apply_failed", e.what());
    }
    return out;
}

}  // namespace caliper_builder
