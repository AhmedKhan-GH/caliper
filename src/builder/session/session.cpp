// session.cpp: Session lifecycle, shared Impl helpers, command binding, host
// bridges, and journal replay. Domain command groups live in
// session_catalog.cpp, session_spec.cpp, session_preview.cpp, and
// session_permissions.cpp.
#include "session_internal.h"

#include <exception>

namespace caliper_builder {

namespace {

namespace fs = std::filesystem;
using nlohmann::json;
using namespace session_detail;
using namespace session_internal;

}  // namespace

const std::vector<std::string>& session_host_commands() {
    static const std::vector<std::string> names = {"app.info", "app.applets", "app.launch", "panel.render",
                                                   "capture.frame", "camera.get", "camera.set"};
    return names;
}

// ---- shared helpers --------------------------------------------------------

std::shared_ptr<SessionHost> Session::Impl::current_host() const {
    std::lock_guard<std::mutex> lock(host_mutex);
    return host;
}

spec::SourceResolver Session::Impl::resolver() const {
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

std::string Session::Impl::rel(const fs::path& p) const {
    const std::string c = p.u8string();
    if (path_within(c, workspace) && c.size() > workspace.size())
        return fs::u8path(c.substr(workspace.size() + 1)).generic_u8string();
    return p.generic_u8string();
}

fs::path Session::Impl::ws() const { return fs::u8path(workspace); }

std::string Session::Impl::spec_label() const {
    if (doc && doc->path) return *doc->path;
    return (doc ? doc->spec.applet.id : std::string("spec")) + ".spec.toml";
}

// Reads, parses and validates a spec file given relative to the workspace.
LoadedSpec Session::Impl::load_spec_file(const std::string& rel_path, const spec::SourceResolver& res) const {
    LoadedSpec out;
    const fs::path p = fs::u8path(rel_path);
    bool dotdot = false;
    for (const auto& part : p) dotdot = dotdot || part == "..";
    if (rel_path.empty() || p.is_absolute() || p.has_root_name() || p.has_root_directory() || dotdot) {
        out.error =
            Error("spec.path_refused", "A spec path is relative to the workspace and has no .. segment.", "args.path");
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
        out.error =
            e.code == "session.io" ? Error("spec.not_found", "The spec file could not be read.", "args.path") : e;
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
LoadedSpec Session::Impl::load_applet(const std::string& id, const spec::SourceResolver& res) const {
    LoadedSpec out;
    const fs::path folder = ws() / "applets" / fs::u8path(id);
    const fs::path manifest = folder / fs::u8path(id + ".caliper.toml");
    std::error_code ec;
    if (!valid_id(id) || !fs::is_regular_file(manifest, ec)) {
        out.error =
            Error("applet.not_found", "The workspace has no applet " + id + "; applet.list shows the saved ones.",
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
        out.error =
            Error("applet.not_spec", "The applet " + id + " is not a spec applet, so it cannot be edited here.",
                  "args.id");
        return out;
    }
    return load_spec_file(rel(folder / fs::u8path(m.spec)), res);
}

// The event `refused` of each program whose pinned piece is missing from
// its source or has another hash, or whose pin has no hash, when a spec
// opens (design/programmable-graphics.md, Status and Events). The spec
// opens all the same, and the host never pins another version on its own.
void Session::Impl::refuse_unresolved_pins(const spec::Spec& s) {
    for (const spec::Program& p : s.programs) {
        if (!p.piece) continue;
        const FoundPiece f = resolve_spec_pin(PieceLibrary::builtin(), workspace, s, *p.piece);
        if (!f.error) continue;
        ProgramStatus st;
        st.id = p.id;
        st.kind = "piece";
        st.state = "refused";
        st.panel = p.panel.value_or("");
        st.error = ProgramError{f.error.code, f.error.message, 0, 0};
        push_program_event(*events, st, "refused");
    }
}

// Replaces the open document. Its undo restores the document that was
// open before; its redo puts back the one it replaced.
UndoRecord Session::Impl::replace_document(Document next) {
    struct State {
        std::optional<Document> previous;
        std::uint64_t current = 0;
        std::optional<Document> redo;
    };
    auto state = std::make_shared<State>();
    state->previous = doc;
    state->current = next.id;
    const std::size_t bytes =
        2 * (spec::to_toml(next.spec).size() + (doc ? spec::to_toml(doc->spec).size() : 0)) + 512;
    doc = std::move(next);
    refuse_unresolved_pins(doc->spec);
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
UndoRecord Session::Impl::change_spec(spec::Spec after, bool merging, std::size_t bytes) {
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

bool Session::Impl::applet_saved(const std::string& id) const {
    if (!valid_id(id)) return false;
    std::error_code ec;
    const fs::path folder = ws() / "applets" / fs::u8path(id);
    return fs::exists(folder / fs::u8path(id + ".caliper.toml"), ec) ||
           fs::exists(folder / fs::u8path(id + ".spec.toml"), ec);
}

// ---- the app's commands ----------------------------------------------------

CommandOutput Session::Impl::panel_render(const CommandContext& ctx) {
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
    const ChainContext chain = chain_context();
    PanelData data = resolve_panel(s, id, *store, options.resolve, &sources, &chain);
    // The panel as it draws now: its option references applied.
    for (spec::Panel& p : s.panels)
        if (p.id == id) p = panel_with_values(p, &chain);
    return h->render_panel(s, data, static_cast<int>(int_arg(ctx.invocation.args, "width", 800)),
                           static_cast<int>(int_arg(ctx.invocation.args, "height", 500)));
}

CommandOutput Session::Impl::run_cmd(const CommandContext& ctx) {
    return run_command(ctx, run_control.load(), model_runner.load());
}

// A start or a fork runs only when the person approved its card, from
// every origin (design/hosted-runs.md, Review Card).
CommandOutput Session::Impl::runner_cmd(const CommandContext& ctx) {
    const bool approved = !starts_run(ctx.invocation.name) || run_approved(ctx.invocation);
    return runner_command(ctx, model_runner.load(), workspace, approved);
}

CommandOutput Session::Impl::events_wait(const CommandContext& ctx) { return events_wait_command(ctx, *events); }

spec::Spec Session::Impl::open_copy() const {
    std::lock_guard<std::mutex> lock(doc_mutex);
    return doc ? doc->spec : spec::Spec{};
}

CommandOutput Session::Impl::piece_cmd(const CommandContext& ctx) { return piece_command(ctx, open_copy(), workspace); }

// piece.save writes the inline piece to the workspace and installs the
// spec without it, its programs pinned to the copy, as one undo step.
CommandOutput Session::Impl::piece_save(const CommandContext& ctx) {
    std::lock_guard<std::mutex> lock(doc_mutex);
    const bool dry = bool_arg(ctx.invocation.args, "dry_run", false);
    auto quiet = [dry](CommandOutput out) {
        out.no_effect = dry || out.error;
        return out;
    };
    if (!doc) return quiet(no_open_spec());
    PieceSave saved = save_inline_piece(workspace, doc->spec, ctx.invocation.args, dry);
    if (saved.error) return quiet(CommandOutput::fail(saved.error));
    if (dry || saved.after == doc->spec) return CommandOutput::unchanged(std::move(saved.result));
    const std::size_t bytes = 2 * (spec::to_toml(doc->spec).size() + spec::to_toml(saved.after).size()) + 512;
    return CommandOutput::ok(std::move(saved.result), change_spec(std::move(saved.after), false, bytes));
}

CommandOutput Session::Impl::resource_cmd(const CommandContext& ctx) {
    return resource_command(ctx, open_copy(), history);
}

CommandOutput Session::Impl::program_cmd(const CommandContext& ctx) {
    spec::Spec open;
    {
        std::lock_guard<std::mutex> lock(doc_mutex);
        if (doc) open = doc->spec;
    }
    return program_command(ctx, open, program_host.load(), draw_host.load(), workspace);
}

// The commands of design/instrument-interface.md that change or read
// session state (instrument_commands.h). They run on a copy of the open
// spec, without the document's lock.
CommandOutput Session::Impl::instrument_cmd(const CommandContext& ctx) {
    spec::Spec open;
    bool is_open = false;
    {
        std::lock_guard<std::mutex> lock(doc_mutex);
        if (doc) {
            open = doc->spec;
            is_open = true;
        }
    }
    InstrumentContext ic{open, is_open, values, history, events.get(), store.get(), &sources, options.resolve,
                         workspace};
    const std::string& n = ctx.invocation.name;
    if (n == "state.set" || n == "state.get" || n == "control.set") return state_command(ctx, ic);
    if (n == "selection.set" || n == "selection.get") return selection_command(ctx, ic);
    return cursor_command(ctx, ic);
}

// What references and kept steps read when a panel resolves.
ChainContext Session::Impl::chain_context() const {
    const CursorState c = history.cursor();
    return ChainContext{&values, &history, c.step, c.live};
}

// Resets the instruments' session state when the open spec differs from
// the one it was last reset for.
void Session::Impl::reset_instruments() {
    spec::Spec now;
    {
        std::lock_guard<std::mutex> doc_lock(doc_mutex);
        if (doc) now = doc->spec;
    }
    std::lock_guard<std::mutex> lock(instrument_mutex);
    if (instrumented && *instrumented == now) return;
    instrumented = now;
    values.reset(now);
    deps.rebuild(now);
    history.configure(now);
}

// The intervention commands (design/gpu-programs.md).
CommandOutput Session::Impl::interpretation_cmd(const CommandContext& ctx) {
    return interpretation_command(ctx, run_control.load());
}

CommandOutput Session::Impl::intervene_cmd(const CommandContext& ctx) {
    return intervene_command(ctx, run_control.load());
}

// ---- the approval feed -----------------------------------------------------
// The registry tells of each approval as it goes (set_approval_observer):
// parked, approved, denied or withdrawn, with its own id, also for the
// commands that the journal does not record, such as run commands. The
// person approves and denies; a park and a withdrawal are the caller's.
void Session::Impl::approval_event(const PendingApproval& p, std::string_view state) {
    Event e;
    e.kind = EventKind::approval;
    const bool person = state == "approved" || state == "denied";
    e.origin = command_origin_name(person ? CommandOrigin::ui : p.invocation.origin);
    e.fields = {{"approval", p.approval}, {"command", p.invocation.name}, {"state", std::string(state)}};
    const std::string& n = p.invocation.name;
    if (n == "run.set_param" || n == "run.action") {
        const std::string target = str_arg(p.invocation.args, "name");
        if (!target.empty()) e.fields["target"] = target;
    }
    events->push(std::move(e));
}

// The applet id of the open spec, for the spec events of the journal.
std::string Session::Impl::open_applet_id() const {
    std::lock_guard<std::mutex> lock(doc_mutex);
    return doc ? doc->spec.applet.id : std::string();
}

// Makes the host of GPU programs run what the open spec holds
// (program_host.h, design/gpu-programs.md, Attachment): after each
// journaled command (an edit, an undo, a redo, an open), each host's
// reconcile() gets the open spec when it differs from what the hosts got
// last, or when a host was set since; with no spec open, an empty one.
// It runs on the thread that ran the command, the frame thread in the app.
void Session::Impl::reconcile_hosts() {
    reset_instruments();
    ProgramHost* programs = program_host.load();
    std::lock_guard<std::mutex> lock(reconcile_mutex);
    if (!programs && !reconciled) return;
    spec::Spec now;
    {
        std::lock_guard<std::mutex> doc_lock(doc_mutex);
        if (doc) now = doc->spec;
    }
    const bool changed = !reconciled || !(*reconciled == now);
    const bool new_hosts = programs != reconciled_program_host;
    if (!changed && !new_hosts) return;
    reconciled = now;
    reconciled_program_host = programs;
    if (programs) programs->reconcile(now);
}

void Session::Impl::bind_all() {
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
                             "panel.patch", "panel.remove", "layout.place", "layout.set", "layout.lock",
                             "control.create", "control.patch", "control.remove", "program.create", "program.patch",
                             "program.remove", "state.create", "state.patch", "state.remove", "entity.set",
                             "entity.remove", "annotation.set", "annotation.remove", "camera.save"})
        bind(name, &Impl::spec_edit);
    // The session state of instruments (design/instrument-interface.md).
    for (const char* name : {"state.set", "state.get", "control.set", "selection.set", "selection.get", "cursor.get",
                             "cursor.set", "cursor.step"})
        bind(name, &Impl::instrument_cmd);
    // The programs and interventions (design/gpu-programs.md).
    bind("program.status", &Impl::program_cmd);
    for (const char* name : {"piece.check", "piece.list", "piece.show", "piece.copy", "piece.update", "piece.remove"})
        bind(name, &Impl::piece_cmd);
    bind("piece.save", &Impl::piece_save);
    bind("program.read", &Impl::resource_cmd);
    for (const char* name : {"intervene.points", "intervene.set", "intervene.clear", "intervene.list"})
        bind(name, &Impl::intervene_cmd);
    for (const char* name : {"feature.dictionary", "feature.show", "feature.examples", "trace.run", "trace.status",
                             "trace.read", "patch.sweep", "trace.validate"})
        bind(name, &Impl::interpretation_cmd);
    // The run commands and events.wait (design/instruments.md).
    for (const char* name : {"run.state", "run.set_param", "run.action", "run.pause"}) bind(name, &Impl::run_cmd);
    for (const char* name : {"run.list", "run.start", "run.fork", "run.stop", "run.logs"})
        bind(name, &Impl::runner_cmd);
    bind("events.wait", &Impl::events_wait);
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

// ---- Session public interface ----------------------------------------------

Session::Session() : impl_(std::make_unique<Impl>()) {}

Session::~Session() { close(); }

Session::Open Session::open(const std::string& workspace, SessionOptions options) {
    Open out;
    try {
        std::unique_ptr<Session> s(new Session());
        Impl& im = *s->impl_;
        im.options = std::move(options);
        // The person's settings that set what an agent may do: no command
        // writes, replaces or removes them (loaders/paths.h).
        protect_file(im.options.permissions_file);
        for (const std::string& f : im.options.protected_files) protect_file(f);
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
        // Each journaled command also reaches the event ring, before the app's
        // own hook (design/instruments.md, Event Feeds).
        im.options.journal.on_append = [ring = im.events, self = &im, app = std::move(im.options.journal.on_append)](
                                           const JournalEvent& e) {
            // The journal appends after the command's handler returned, so
            // the open spec is not locked here.
            note_journal_event(*ring, e, self->open_applet_id());
            // A host's reconcile() runs here, under the journal's lock: it
            // sends and compiles, and never invokes a command.
            self->reconcile_hosts();
            if (app) app(e);
        };
        const std::string journal_file = (im.ws() / "journal" / "commands.jsonl").u8string();
        if (touches_protected_file(journal_file)) {
            out.error = Error("session.protected_file",
                              "The command journal would be written into a protected settings file, so the workspace "
                              "was not opened.",
                              journal_file);
            return out;
        }
        CommandJournal::Open j = CommandJournal::open(journal_file, im.options.journal);
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
        im.registry->set_approval_observer(
            [raw](const PendingApproval& p, std::string_view state) { raw->approval_event(p, state); });
        if (!im.options.permissions_file.empty()) {
            AgentPermissionsRead read = read_agent_permissions(im.options.permissions_file);
            im.perms = std::move(read.file);
            im.saved = im.perms;
            im.stamp = im.file_stamp();
            im.perm_problems = std::move(read.problems);
        }
        for (CommandOrigin o : {CommandOrigin::ui, CommandOrigin::builder, CommandOrigin::agent, CommandOrigin::mcp,
                                CommandOrigin::api})
            im.review_on[static_cast<std::size_t>(o)].store(im.options.review.on(o));
        im.bind_all();
        im.queue = std::make_unique<CommandQueue>(*im.registry, im.options.queue);
        out.session = std::move(s);
    } catch (const std::exception& e) {
        out.error =
            Error("session.open_failed", std::string("The workspace could not be opened: ") + e.what(), workspace);
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

void Session::set_run_control(RunControl* control) { impl_->run_control.store(control); }
void Session::set_model_runner(ModelRunner* runner) { impl_->model_runner.store(runner); }
ModelRunner* Session::model_runner() const { return impl_->model_runner.load(); }
void Session::set_program_host(ProgramHost* host) { impl_->program_host.store(host); }
ProgramHost* Session::program_host() const { return impl_->program_host.load(); }
void Session::set_draw_host(DrawStatusHost* host) { impl_->draw_host.store(host); }
DrawStatusHost* Session::draw_host() const { return impl_->draw_host.load(); }
RunControl* Session::run_control() const { return impl_->run_control.load(); }
EventRing& Session::events() { return *impl_->events; }
ValueStore& Session::values() { return impl_->values; }
StepHistory& Session::history() { return impl_->history; }

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
    for (const char* name : {"camera.get", "camera.set"})
        im.registry->bind(name, [raw](const CommandContext& ctx) {
            auto h = raw->current_host();
            return h ? h->camera_command(ctx.invocation.name, ctx.invocation.args)
                     : fail("command.not_bound", ctx.invocation.name + " needs the app.");
        });
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
    const ChainContext chain = impl_->chain_context();
    return resolve_panel(s, panel_id, *impl_->store, impl_->options.resolve, &impl_->sources, &chain);
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
    impl_->events->close();  // wakes a caller held in wait_for_events()
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
