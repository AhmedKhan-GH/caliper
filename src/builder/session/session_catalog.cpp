// session_catalog.cpp: catalog, data query/import, and loader command handlers
// of Session::Impl (session_internal.h).
#include "session_internal.h"

namespace caliper_builder {

namespace {

namespace fs = std::filesystem;
using nlohmann::json;
using namespace session_detail;
using namespace session_internal;

}  // namespace

// Checks the input path of a loader for its origin, and grants a path
// outside the workspace that the person named.
Error Session::Impl::check_loader_input(const Loader& l, CommandOrigin origin) {
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

Error Session::Impl::loader_save_error(Error e) {
    if (e.code == "loader.path_not_allowed") e.code = "loader.path_refused";
    return e;
}

json Session::Impl::run_outputs(const RunResult& r) const {
    json outputs = json::array();
    for (const LoadedSource& s : r.sources) outputs.push_back(s.id);
    return outputs;
}

std::map<std::string, std::vector<std::string>> Session::Impl::outputs_by_loader() {
    std::map<std::string, std::vector<std::string>> out;
    for (const LoadedSource& s : store->sources()) out[s.loader].push_back(s.id);
    return out;
}

// Runs a loader, records how long it took and tells the catalog.
// A Python loader's script runs only with `allow_python`.
RunResult Session::Impl::run_loader(const std::string& id, bool allow_python, double* elapsed_ms) {
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

// ---- catalog and data ------------------------------------------------------

CommandOutput Session::Impl::catalog_list(const CommandContext& ctx) {
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

CommandOutput Session::Impl::catalog_describe(const CommandContext& ctx) {
    const std::string id = str_arg(ctx.invocation.args, "id");
    CatalogDescribeResult r = catalog->describe(id);
    if (r.error) return CommandOutput::fail(r.error);
    if (!r.entry)
        return fail("catalog.not_found", "The catalog has no source " + id + "; catalog.list shows them.", "args.id");
    return CommandOutput::ok({{"entry", catalog_entry_to_json(*r.entry)}});
}

CommandOutput Session::Impl::catalog_sample(const CommandContext& ctx) {
    const json& a = ctx.invocation.args;
    CatalogSampleRequest req;
    req.limit = static_cast<std::size_t>(int_arg(a, "limit", 20));
    req.start = int_list_arg(a, "start");
    req.count = int_list_arg(a, "count");
    CatalogSampleResult r = catalog->sample(str_arg(a, "id"), req);
    if (r.error) return CommandOutput::fail(r.error);
    return CommandOutput::ok(catalog_sample_to_json(r.sample));
}

CommandOutput Session::Impl::data_query(const CommandContext& ctx) {
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
    return CommandOutput::ok({{"columns", std::move(columns)},
                              {"rows", std::move(rows)},
                              {"truncated", truncated},
                              {"elapsed_ms", q.elapsed_ms}});
}

// The undo of commands that created loaders: deletes them if their recipes
// are unchanged; the redo saves them again and runs those that ran.
UndoRecord Session::Impl::made_loaders_undo(std::vector<MadeLoader> made) {
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

CommandOutput Session::Impl::data_import(const CommandContext& ctx) {
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
            refused.push_back(
                {{"path", given[i]}, {"code", loader_save_error(item.error).code}, {"message", item.error.message}});
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

// ---- loaders ---------------------------------------------------------------

json Session::Impl::loader_summary(const LoaderRecord& r,
                                   const std::map<std::string, std::vector<std::string>>& outputs) {
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

CommandOutput Session::Impl::loader_list(const CommandContext&) {
    const auto outputs = outputs_by_loader();
    json loaders = json::array();
    for (const LoaderRecord& r : store->list()) loaders.push_back(loader_summary(r, outputs));
    return CommandOutput::ok({{"loaders", std::move(loaders)}});
}

CommandOutput Session::Impl::loader_describe(const CommandContext& ctx) {
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

CommandOutput Session::Impl::loader_create(const CommandContext& ctx) {
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

CommandOutput Session::Impl::loader_update(const CommandContext& ctx) {
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

CommandOutput Session::Impl::loader_run(const CommandContext& ctx) {
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

CommandOutput Session::Impl::loader_delete(const CommandContext& ctx) {
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

}  // namespace caliper_builder
