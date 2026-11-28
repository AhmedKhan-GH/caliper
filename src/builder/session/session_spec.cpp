// session_spec.cpp: open-spec, instrument save/list/open, and panel.data command
// handlers of Session::Impl (session_internal.h).
#include "session_internal.h"

#include <algorithm>
#include <functional>

namespace caliper_builder {

namespace {

namespace fs = std::filesystem;
using nlohmann::json;
using namespace session_detail;
using namespace session_internal;

}  // namespace

// ---- the open spec ---------------------------------------------------------

CommandOutput Session::Impl::no_open_spec() {
    return fail("spec.no_open_spec", "No spec is open; spec.new starts one and instrument.open opens a saved one.");
}

CommandOutput Session::Impl::spec_new(const CommandContext& ctx) {
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
    const std::string id = next.spec.instrument.id;
    return CommandOutput::ok({{"id", id}, {"text", text}}, replace_document(std::move(next)));
}

CommandOutput Session::Impl::open_loaded(LoadedSpec loaded, json result) {
    Document next;
    next.id = next_doc++;
    next.saved_text = spec::to_toml(loaded.spec);
    next.spec = std::move(loaded.spec);
    next.path = loaded.path;
    result["warnings"] = warnings_json(loaded.diagnostics);
    return CommandOutput::ok(std::move(result), replace_document(std::move(next)));
}

CommandOutput Session::Impl::spec_open(const CommandContext& ctx) {
    std::lock_guard<std::mutex> lock(doc_mutex);
    LoadedSpec loaded = load_spec_file(str_arg(ctx.invocation.args, "path"), resolver());
    if (loaded.error) return CommandOutput::fail(loaded.error);
    json result{{"id", loaded.spec.instrument.id}, {"path", loaded.path}};
    return open_loaded(std::move(loaded), std::move(result));
}

CommandOutput Session::Impl::instrument_open(const CommandContext& ctx) {
    std::lock_guard<std::mutex> lock(doc_mutex);
    const std::string id = str_arg(ctx.invocation.args, "id");
    LoadedSpec loaded = load_instrument(id, resolver());
    if (loaded.error) return CommandOutput::fail(loaded.error);
    json result{{"id", id}, {"title", loaded.spec.instrument.title.value_or(loaded.spec.instrument.id)}};
    return open_loaded(std::move(loaded), std::move(result));
}

// The spec that the read-only commands of the browser's spec page read: a
// saved instrument's, read from its files without opening it, or else the
// open spec or the preview the invocation names.
Error Session::Impl::view_spec(const std::string& instrument, const std::string& preview, spec::Spec* out) {
    if (!instrument.empty()) {
        LoadedSpec loaded = load_instrument(instrument, {});
        if (loaded.error) return Error(loaded.error.code, loaded.error.message, "args.instrument");
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

CommandOutput Session::Impl::spec_get(const CommandContext& ctx) {
    const std::string instrument = str_arg(ctx.invocation.args, "instrument");
    if (!instrument.empty()) {
        LoadedSpec loaded = load_instrument(instrument, {});
        if (loaded.error) return fail(loaded.error.code, loaded.error.message, "args.instrument");
        // Whether the app has this instrument open: only then do its live sources, the session's
        // values and the kept steps reach panel.data (the browser's viewer says so when not).
        bool shown = false;
        {
            std::lock_guard<std::mutex> lock(doc_mutex);
            shown = doc && doc->spec.instrument.id == loaded.spec.instrument.id;
        }
        json out = {{"open", true},
                    {"id", loaded.spec.instrument.id},
                    {"path", loaded.path},
                    {"dirty", false},
                    {"open_in_app", shown},
                    {"text", spec::to_toml(loaded.spec)},
                    {"json", spec::to_json(loaded.spec)}};
        if (shown) {
            if (RunControl* rc = run_control.load()) {
                const json prog = rc->progress_state();
                if (prog.is_object()) out["progress"] = prog;
            }
        }
        return CommandOutput::ok(std::move(out));
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
                              {"id", s->instrument.id},
                              {"path", doc->path ? json(*doc->path) : json(nullptr)},
                              {"dirty", !doc->path || text != doc->saved_text},
                              {"text", text},
                              {"json", spec::to_json(*s)}});
}

CommandOutput Session::Impl::spec_validate(const CommandContext& ctx) {
    std::vector<spec::Diagnostic> diagnostics;
    // The programs of one panel share its camera: the draw host refuses the later program
    // of a conflict when it runs, and the check here tells an agent before it applies.
    auto cameras = [&](const spec::Spec& s) {
        for (spec::Diagnostic& d : check_panel_cameras(PieceLibrary::builtin(), workspace, s))
            diagnostics.push_back(std::move(d));
    };
    if (ctx.invocation.args.contains("text")) {
        SpecTextCheck checked = check_spec_text(str_arg(ctx.invocation.args, "text"), resolver());
        diagnostics = std::move(checked.diagnostics);
        cameras(checked.spec);  // an empty spec when the text does not parse
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
        cameras(*s);
    }
    json errors = json::array();
    for (const auto& d : diagnostics)
        if (d.is_error()) errors.push_back(diagnostic_json(d));
    const bool valid = errors.empty();
    return CommandOutput::ok({{"valid", valid}, {"errors", std::move(errors)}, {"warnings", warnings_json(diagnostics)}});
}

// apply_spec_edit(), and for program.create and program.patch of an
// instance of a piece the resolution of its pin against the library, the
// workspace and the spec's inline pieces (piece_commands.h): the missing
// or changed piece refuses the edit, a pin without a hash takes the
// piece's, and `args` gets that hash so that the journal replays it.
SpecEdit Session::Impl::edit_spec(const spec::Spec& s, const std::string& name, json& args) {
    const spec::SourceResolver res = resolver();
    SpecEdit e = apply_spec_edit(s, name, args, res);
    if (e.error || (name != "program.create" && name != "program.patch")) return e;
    const std::string id = str_arg(args, "id");
    if (Error err = complete_instance(PieceLibrary::builtin(), workspace, e.spec, id, res, &args)) {
        SpecEdit refused;
        refused.error = std::move(err);
        return refused;
    }
    return e;
}

// A dry run, whether it succeeds or fails, and an edit that leaves the
// spec as it was report no effect, so the registry treats them as reads:
// the redo stack stays, no undo step is filed and the journal is not
// written.
CommandOutput Session::Impl::spec_edit(const CommandContext& ctx) {
    // camera.save saves the app's current orbit (interactive_commands.h):
    // the session asks the app for it before the edit, without the lock.
    json camera_args;
    if (ctx.invocation.name == "camera.save" && !ctx.invocation.args.contains("orbit")) {
        std::shared_ptr<SessionHost> h = current_host();
        if (!h) return fail("command.not_bound", "camera.save needs the app.");
        CommandOutput got = h->camera_command("camera.get", {{"panel", str_arg(ctx.invocation.args, "panel")}});
        if (got.error) return got;
        camera_args = ctx.invocation.args;
        camera_args["orbit"] = got.result.value("orbit", json());
    }
    std::lock_guard<std::mutex> lock(doc_mutex);
    json args = camera_args.is_null() ? ctx.invocation.args : camera_args;
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
    SpecEdit e = edit_spec(doc->spec, ctx.invocation.name, args);
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
    CommandOutput out =
        CommandOutput::ok(std::move(result), change_spec(std::move(e.spec), !ctx.invocation.merge_key.empty(), bytes));
    // camera.save journals the orbit it saved, so that a replay of the
    // journal and a redo need no app; a program edit journals the hash
    // of the piece it pinned.
    out.recorded_args = camera_args;
    if (args != ctx.invocation.args) out.recorded_args = args;
    return out;
}

// ---- instruments and panel data --------------------------------------------

CommandOutput Session::Impl::instrument_save(const CommandContext& ctx) {
    std::lock_guard<std::mutex> lock(doc_mutex);
    if (!doc) return no_open_spec();
    const json& a = ctx.invocation.args;
    const std::string id = str_arg(a, "id", doc->spec.instrument.id);
    const bool overwrite = bool_arg(a, "overwrite", false);
    if (Error e = first_spec_error(spec::validate(doc->spec, resolver()).diagnostics)) return CommandOutput::fail(e);
    if (!valid_id(id)) return fail("spec.invalid_id", "An instrument id matches [a-z][a-z0-9_-]{0,63}.", "args.id");

    const fs::path folder = ws() / "instruments" / fs::u8path(id);
    const fs::path manifest = folder / fs::u8path(id + ".caliper.toml");
    const fs::path spec_file = folder / fs::u8path(id + ".spec.toml");
    const bool exists = instrument_saved(id);
    if (exists && !overwrite)
        return fail("instrument.exists", "An instrument " + id + " is saved already; set overwrite to replace it.", "args.id");
    // The manifest id is the one the caller names (the schema holds it to the
    // manifest's grammar). Without one, an overwrite keeps the saved manifest's
    // id and version, and a new instrument is local.<id>.
    const bool named = a.contains("manifest_id");
    std::string manifest_id = str_arg(a, "manifest_id");
    std::string version = "0.1.0";
    if (exists) {
        ManifestInfo old = read_manifest(manifest.u8string());
        if (!old.error && !old.version.empty()) version = old.version;
        if (!named && !old.error) manifest_id = old.id;
    }
    if (manifest_id.empty()) manifest_id = manifest_id_for(id);
    std::error_code ec;
    for (fs::directory_iterator it(ws() / "instruments", ec), end; !ec && it != end; it.increment(ec)) {
        const std::string other = it->path().filename().u8string();
        if (other == id || !valid_id(other)) continue;
        const fs::path m = it->path() / fs::u8path(other + ".caliper.toml");
        std::error_code mec;
        if (!fs::is_regular_file(m, mec)) continue;
        if (read_manifest(m.u8string()).id == manifest_id)
            return fail("instrument.exists",
                        "The instrument " + other + " already has the manifest id " + manifest_id + "; choose another " +
                            (named ? "manifest id." : "id."),
                        named ? "args.manifest_id" : "args.id");
    }
    spec::Spec saved = doc->spec;
    saved.instrument.id = id;
    const std::string text = spec::to_toml(saved);
    std::optional<std::string> summary;
    if (saved.instrument.description && saved.instrument.description->find_first_of("\r\n") == std::string::npos &&
        command_code_points(*saved.instrument.description) <= 200)
        summary = *saved.instrument.description;
    const std::string name = saved.instrument.title.value_or(id);
    if (Error e = write_file_atomic(spec_file.u8string(), text))
        return fail("instrument.write_failed", "The spec file could not be written: " + e.message, rel(spec_file));
    if (Error e = write_file_atomic(manifest.u8string(),
                                    manifest_text(manifest_id, name, version, summary, id + ".spec.toml")))
        return fail("instrument.write_failed", "The manifest could not be written: " + e.message, rel(manifest));
    // Saving under the spec's own id makes the files the open spec's
    // home; saving under another id writes a copy and leaves it as it was.
    if (id == doc->spec.instrument.id) {
        doc->path = rel(spec_file);
        doc->saved_text = text;
    }
    return CommandOutput::ok({{"id", id}, {"manifest", rel(manifest)}, {"spec", rel(spec_file)}});
}

CommandOutput Session::Impl::instrument_list(const CommandContext&) {
    std::vector<json> found;
    std::error_code ec;
    std::size_t seen = 0;
    for (fs::directory_iterator it(ws() / "instruments", ec), end; !ec && it != end; it.increment(ec)) {
        if (++seen > kMaxInstrumentFolders) break;
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
    return CommandOutput::ok({{"instruments", json(found)}});
}

// One panel's data for the browser's spec page (panel_data.h); with
// `known` equal to the version, nothing is resolved.
CommandOutput Session::Impl::panel_data(const CommandContext& ctx) {
    const json& args = ctx.invocation.args;
    spec::Spec s;
    if (Error e = view_spec(str_arg(args, "instrument"), ctx.invocation.preview, &s)) return CommandOutput::fail(e);
    const std::string id = str_arg(args, "panel");
    // The session's values and kept steps belong to its open spec: a page
    // of that instrument reads them, a page of another saved instrument does not.
    ViewerInputs in;
    in.library = &PieceLibrary::builtin();
    in.workspace = workspace;
    {
        std::lock_guard<std::mutex> lock(doc_mutex);
        const std::string instrument = str_arg(args, "instrument");
        if (doc && (instrument.empty() || doc->spec.instrument.id == instrument)) {
            in.values = &values;
            in.history = &history;
        }
    }
    const std::uint64_t revision = catalog->revision();
    if (!s.find_panel(id)) {
        // A control of the layout: its state, which the page shows.
        const spec::Control* c = s.find_control(id);
        if (!c) return fail("panel.not_found", "The spec has no panel " + id + "; spec.get lists them.", "args.panel");
        json r = control_data_result(s, *c, in, run_control.load(), "");
        const std::string version =
            panel_data_version(s, id, revision, std::hash<std::string>{}(r["data"].dump()));
        if (str_arg(args, "known") == version) return CommandOutput::ok(panel_data_unchanged(s, id, version));
        r["version"] = version;
        return CommandOutput::ok(std::move(r));
    }
    // The cursor goes into the version, so that a move of a held cursor
    // gives new data; for a scene panel, so do its programs and their
    // newest resources.
    const ChainContext chain = chain_context();
    std::uint64_t generation = panel_generation(s, id, sources, &chain);
    json programs;
    std::string digest;
    if (s.find_panel(id)->primitive == spec::Primitive::scene) {
        programs = viewer_programs(s, id, in, &digest);
        generation ^= std::hash<std::string>{}(digest);
    }
    const std::string version = panel_data_version(s, id, revision, generation);
    if (str_arg(args, "known") == version) return CommandOutput::ok(panel_data_unchanged(s, id, version));
    json r = panel_data_result(s, resolve_panel(s, id, *store, options.resolve, &sources, &chain), revision);
    if (!programs.is_null()) {
        r["version"] = version;
        if (r["data"].is_object()) r["data"]["programs"] = std::move(programs);
    }
    return CommandOutput::ok(std::move(r));
}

}  // namespace caliper_builder
