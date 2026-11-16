#pragma once
// Session: the headless object that binds the builder libraries for one
// workspace, which the app will own after the wiring
// (design/builder-session.md).
//
// Session::open() opens a workspace folder (sprint 3 contracts, section 2):
// the LoaderStore and its workspace.duckdb, a Catalog with the loader outputs'
// provider, the command journal at journal/commands.jsonl, a CommandRegistry
// that holds the whole descriptor table, and a CommandQueue over it. It binds
// a handler for every command that needs no window: catalog.*, data.*,
// loader.*, spec.*, source.*, panel.create, panel.patch, panel.remove,
// layout.*, preview.*, applet.*, review.status (the history commands come
// with the registry).
// app.info, app.applets, app.launch, panel.render and capture.frame stay
// unbound until set_host() gives a SessionHost, which the app implements.
//
// The open spec. At most one spec is open; the spec, source, panel and layout
// commands edit it. Every edit is validated with a SourceResolver that asks
// the catalog, and an edit that leaves the spec with an error is refused with
// the validator's code, so the open spec is valid after every command. Each
// edit returns the validator's warnings and the change as a unified diff of
// the canonical TOML (`diff`), and takes `dry_run` to return them without
// applying. A dry run, refused or not, and an edit that leaves the spec as it
// was report no effect (CommandOutput::no_effect), so they file no undo step,
// keep the redo stack and write no journal line. Each edit has an exact undo: a closure pair that restores the spec
// as it was, after checking that the open spec is still the document the
// command changed (history.conflict otherwise).
//
// Stacked previews. preview.begin makes a preview of the open spec: an id, a
// scratch copy (the shadow) that starts equal to the open spec, and the stack
// of edits made on it. An edit of the open spec whose invocation names the
// preview (CommandInvocation::preview, the header X-Caliper-Preview) applies
// to the shadow instead, validated as a real edit, and reports no effect, so
// it never reaches the journal or the history; with dry_run it is only
// checked against the shadow. spec.get, spec.validate, panel.render and
// resolve() read the shadow when they name the preview, and any other command
// that mutates is refused with preview.not_previewable. A preview is always
// the open spec with its stack applied: when the open spec changed under it,
// the stack is applied again to the open spec before the preview is read,
// and an edit that no longer applies answers history.conflict, as does a
// preview begun on a spec that is no longer open. preview.diff gives the
// stacked diff; preview.commit, from the ui and builder origins only
// (preview.not_allowed otherwise), applies the stack to the open spec as the
// registry's follow-ups, one undo step whose group is the preview's id;
// preview.discard drops it without a trace. A preview scoped to a panel or a
// source takes only edits of that target, of the sources a target panel binds
// (through `from` too) and of the items the preview added
// (preview.out_of_scope). PreviewLimits bounds the previews open at once,
// the edits and bytes of one, and its idle time, after which it is dropped
// (preview.expired).
//
// Review. With review on for an origin (ReviewSettings: agent and mcp by
// default, settable per origin), an edit of the open spec from that origin
// that would change it does not run at once: the policy asks, and the parked
// invocation carries its review, {"diff", "warnings"}, the unified diff of
// the canonical TOML before and after and the validator's warnings
// (PendingApproval::review). A dry run, an edit that names a preview, one
// that would fail validation and one that would change nothing run at once.
// When the person approves, the edit is made again on the open spec as it is
// then, and it runs only when the change it makes is the change that was shown
// (the same lines removed and added, wherever they now are); otherwise, or
// when another spec was opened meanwhile, it answers history.conflict. Each
// edit is reviewed on its own; a held call answers before the agent sends its
// next edit, so each diff is against the spec the earlier approvals left. A
// stacked review of several edits is a preview. review.status tells a caller
// which of its commands wait for the person.
//
// Paths. A path outside the workspace is accepted by data.import,
// loader.create and loader.update only from the ui and builder origins, which
// stand for a file the person dropped or named, and is then granted; from
// agent, mcp and api it is refused with loader.path_refused unless an earlier
// grant already covers it, so an agent cannot widen what loaders may read.
//
// Approval. The session's policy is the default policy of the command layer,
// with the review above and two rules that look at the invocation itself:
// - Running a Python loader's script (loader.run of a Python loader, or
//   loader.create or loader.update of one with run set) asks for every origin
//   but ui and builder, each time. The handler passes RunOptions::allow_python
//   to the store only for ui and builder, or for an invocation that the policy
//   asked about and the person approved; a redo never runs a script.
// - applet.save from agent, mcp or api asks only when it would overwrite a
//   saved applet, since writing a new one destroys nothing.
//
// Threads. Commands run where the registry runs them: invoke() on the calling
// thread, or the queue's drain(). open_spec() and resolve() may be called
// from any thread; they read the open spec under the session's own lock. No
// function here throws.
#include <caliper_builder/catalog/catalog.h>
#include <caliper_builder/commands/journal.h>
#include <caliper_builder/commands/queue.h>
#include <caliper_builder/commands/registry.h>
#include <caliper_builder/error.h>
#include <caliper_builder/loaders/store.h>
#include <caliper_builder/session/loader_catalog.h>
#include <caliper_builder/session/resolve.h>
#include <caliper_builder/spec/spec.h>
#include <caliper_builder/spec/validate.h>

#include <nlohmann/json.hpp>

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace caliper_builder {

// The commands that need the running app. The app implements this interface
// and gives it to Session::set_host(), which binds the five commands to it.
// Each function runs on the thread that runs the command (the frame thread in
// the app) and returns the command's result, as its descriptor in the table
// describes it, or its error. A function must not throw.
class SessionHost {
public:
    virtual ~SessionHost() = default;
    virtual CommandOutput app_info() = 0;
    virtual CommandOutput app_applets() = 0;
    virtual CommandOutput app_launch(const std::string& id) = 0;
    // Draws one panel of the open spec off screen. The session has found the
    // panel and resolved its data; the host draws `data` and answers
    // {"png_base64", "width", "height"} or panel.render_failed. When `data`
    // carries a resolve error, such as resolve.unavailable for a source the
    // app serves, the host draws the panel in that state.
    virtual CommandOutput render_panel(const spec::Spec& spec, const PanelData& data, int width, int height) = 0;
    virtual CommandOutput capture_frame() = 0;
};

// The names of the commands that SessionHost serves.
const std::vector<std::string>& session_host_commands();

// Which origins' edits of the open spec wait for the person's review. The
// side chat, the widget agent and outside agents are reviewed by default; the
// person's own UI, the builder and programs over Caliper API are not.
struct ReviewSettings {
    bool ui = false;
    bool builder = false;
    bool agent = true;
    bool mcp = true;
    bool api = false;

    bool on(CommandOrigin origin) const {
        switch (origin) {
            case CommandOrigin::ui: return ui;
            case CommandOrigin::builder: return builder;
            case CommandOrigin::agent: return agent;
            case CommandOrigin::mcp: return mcp;
            case CommandOrigin::api: return api;
            case CommandOrigin::viewer: return false;  // it only reads
        }
        return true;
    }
    void set(CommandOrigin origin, bool value) {
        switch (origin) {
            case CommandOrigin::ui: ui = value; break;
            case CommandOrigin::builder: builder = value; break;
            case CommandOrigin::agent: agent = value; break;
            case CommandOrigin::mcp: mcp = value; break;
            case CommandOrigin::api: api = value; break;
            case CommandOrigin::viewer: break;
        }
    }
};

// The bounds of stacked previews.
struct PreviewLimits {
    std::size_t max_previews = 16;                  // open at once (preview.too_many)
    std::size_t max_edits = 64;                     // stacked on one preview (preview.too_many_edits)
    std::size_t max_bytes = std::size_t(4) << 20;   // the arguments of one preview's edits as JSON (preview.too_large)
    std::chrono::milliseconds idle{30 * 60 * 1000};  // dropped after this long without use (preview.expired)
    std::function<std::chrono::steady_clock::time_point()> now;  // for the idle time; tests set it
};

struct SessionOptions {
    StoreOptions store;
    JournalOptions journal;
    RegistryOptions registry;
    CommandQueue::Options queue;
    ResolveLimits resolve;
    PreviewLimits preview;
    ReviewSettings review;
};

// The open spec as the app reads it.
struct OpenSpec {
    spec::Spec spec;
    std::optional<std::string> path;  // relative to the workspace; empty when never saved
    bool dirty = true;                // changed since it was opened or saved
    std::string text;                 // its canonical TOML
};

// A stacked preview as the app draws it in place.
struct PreviewState {
    Error error;         // spec.no_open_spec, preview.not_found, preview.expired, history.conflict
    std::string id;
    OpenSpec spec;       // the open spec with every edit of the preview applied
    std::size_t edits = 0;
    std::string panel;   // the panel it is scoped to, or empty
    std::string source;  // the source it is scoped to, or empty
};

class Session {
public:
    struct Open {
        Error error;
        std::unique_ptr<Session> session;
    };
    // Opens the workspace folder, creating it and its layout when missing.
    static Open open(const std::string& workspace, SessionOptions options = {});

    ~Session();
    Session(const Session&) = delete;
    Session& operator=(const Session&) = delete;

    // The canonical path of the workspace folder.
    const std::string& workspace() const;

    CommandRegistry& registry();
    CommandQueue& queue();
    LoaderStore& store();
    Catalog& catalog();
    LoaderCatalogProvider& loader_catalog();

    // Runs one invocation on the calling thread (registry().invoke()).
    CommandResult invoke(CommandInvocation invocation);

    // Binds app.info, app.applets, app.launch, panel.render and capture.frame
    // to the host, or leaves them unbound (command.not_bound) for null.
    void set_host(std::shared_ptr<SessionHost> host);

    // Turns the review of spec edits on or off for one origin, from the
    // app's settings; it holds from the next invocation the policy sees.
    void set_review(CommandOrigin origin, bool on);
    ReviewSettings review() const;

    // The open spec, or nothing.
    std::optional<OpenSpec> open_spec() const;

    // The resolver that the edits validate with: what the catalog knows of a
    // catalog id.
    spec::SourceResolver source_resolver() const;

    // The data of one panel of the open spec (resolve.h); spec.no_open_spec
    // when no spec is open. With `preview`, the panel of that preview, which
    // the app draws in place (preview.not_found, preview.expired,
    // history.conflict); reading a preview counts as using it.
    PanelData resolve(std::string_view panel_id, std::string_view preview = {}) const;

    // A preview's spec, for the app to draw; reading it counts as using it.
    PreviewState preview(std::string_view id) const;

    // The data sources of the kinds the app serves (live, feed and metric),
    // which resolve() and panel.render read; the app registers one per kind
    // with data_sources().set() (resolve.h).
    DataSources& data_sources();

    // Shuts the queue down (every waiting future completes with
    // command.shutdown) and closes the journal. The destructor does it too.
    void close();

    struct Impl;

private:
    Session();
    std::unique_ptr<Impl> impl_;
};

// Replays the journal of a workspace onto an empty open spec: the spec,
// source, panel and layout edits and spec.new, spec.open and applet.open that
// are in effect after every undo and redo, in order (dry runs and the
// commands that do not touch the open spec are passed by). spec.open and
// applet.open read the files as they are now. The result is the open spec the
// session would have, or nothing when no spec was opened.
struct SpecReplay {
    Error error;
    std::optional<spec::Spec> spec;
    std::vector<std::uint64_t> applied;  // the journal seqs applied
    std::vector<std::string> problems;   // lines the reader skipped
};
SpecReplay replay_spec_journal(const std::string& workspace);

}  // namespace caliper_builder
