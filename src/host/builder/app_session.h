#pragma once
// ============================================================================
// app_session.h: the builder session in the app (design/builder-session.md,
// design/builder-overview.md, item 2 of the integration order). The app, and
// `caliper shot`, own one AppSession for the workspace of the command line
// (host_cli.h, workspace_choice). It holds a caliper_builder::Session, drains
// its command queue on the frame thread between two frames, and implements the
// session's SessionHost for app.info, app.applets, app.launch, capture.frame
// and panel.render.
//
// Opening. start() opens nothing. A workspace that the command line chose
// (--workspace, CALIPER_WORKSPACE) is opened on a worker once the first frame
// has been drawn; the default workspace, workspace/ under the app-data folder,
// only when something first needs it (ensure_open(), wait_open(), the command
// script), so the app without --workspace starts as before and creates
// nothing. The frame thread adopts the opened session at its next frame():
// it names itself to the registry as the frame thread and binds the host. A
// workspace that cannot be opened leaves the app without the builder; the
// Activity pane and standard error say why.
//
// The frame loop. frame() runs once per frame, where the app takes the calls
// of Caliper API. It drains the queue within a time budget (4 ms by default,
// at least one command per frame), so a long queue cannot stall drawing. It
// looks at the next command before running it (CommandQueue::peek()):
// - data.import, data.query, loader.run, and loader.create or loader.update
//   with `run`, and the approval of one of them, run on a worker (the host's
//   job system in the app), whose result reaches its caller through the
//   queue's future. While that worker runs, the frame thread does not call
//   the queue, the registry, the store or the catalog at all: the registry
//   runs every command under one lock and the store under another, so this is
//   what keeps a frame from waiting on a loader. Later commands wait in the
//   queue, in order, until the worker is done. worker_busy() says so to other
//   code of the frame thread, such as the spec runtime.
// - capture.frame arms a read-back of the frame drawn next and waits, without
//   draining further, until the app hands the pixels over (capture_ready()) or
//   says it could not (capture_failed()); then it runs and answers with the
//   PNG. Where the app cannot read its frame back, it answers
//   capture.unavailable at once with the reason Caliper API gives for its 501.
// - every other command runs on the frame thread.
//
// Activity. Every command the journal records, that is every mutating command
// that ran or was denied (not a dry run), becomes one Activity line, with the
// event "<origin> command" (ui, builder, agent, mcp or api), its name as the
// object and, when it failed, its error code as the detail, and one
// "[session]" line on standard error.
//
// panel.render needs the spec runtime, which draws a panel off screen; until
// one is given to set_panel_renderer() (panel_renderer.h), it answers
// command.not_available. The app gives it one that draws on a canvas sharing
// the app's font atlas (spec_page.h).
//
// The command script, a testing and demo aid: CALIPER_COMMAND_SCRIPT names a
// file of JSON lines, each an invocation such as {"name": "spec.new", "args":
// {...}} ("origin" defaults to "ui"; empty lines and lines starting with #
// are skipped). Once the session is open, a thread of its own submits them one
// at a time through the queue, as an outside client does, and appends each
// result to CALIPER_COMMAND_RESULTS when it is set: one JSON line per command,
// {"line", "name", "id", "ok", "result" | "error"}, then {"done": true,
// "commands", "failed"}. A command that asks for the person's approval (a
// destructive one, or an agent's spec edit under review) never holds the
// script: it answers command.needs_approval with its approval id, which stays
// pending, unless its line has "approve": true, in which case the script
// approves it as the person would and writes the result with "approved".
//
// Shutdown. close() shuts the queue down (waiting calls get command.shutdown),
// waits for the worker and the script, and closes the journal and the
// workspace's database before it returns. The app calls it before the host's
// stores close, because DuckDB aborts when it is left to static destructors.
//
// Built only with CALIPER_WITH_BUILDER; without it start() says, for a chosen
// workspace or a command script, that the builder is not in this build, and
// everything else does nothing. Free of any graphics dependency, so its tests
// drive it with a fake frame loop.
// ============================================================================
#include <chrono>
#include <cstdint>
#include <functional>
#include <memory>
#include <ostream>
#include <string>
#include <vector>

namespace caliper_builder {
class Session;
}

namespace caliper_api {
class CommandGateway;   // api/command_gateway.h
}

namespace caliper_host {

class PanelRenderSeam;   // panel_renderer.h

// What the app's navigation answers a launch by manifest id, for POST
// /v0/applets/{id}/launch and app.launch: status 200 with the body of
// launched_json(), or 404 or 409 with a message, and the command's error code
// (app.applet_not_found, app.launch_refused, app.launch_failed).
struct LaunchOutcome {
    int status = 200;
    std::string body;   // 200: the JSON body; otherwise the message
    std::string code;
};

class AppSession {
public:
    // What the app gives the session. Every function runs on the frame thread
    // unless it says otherwise.
    struct Hooks {
        // Runs fn once on a worker thread, under a label for the jobs tray.
        // Without it, fn runs on a thread of its own.
        std::function<void(const std::string& label, std::function<void()> fn)> run_job;
        std::function<std::string()> info_json;      // the body of GET /v0/info
        std::function<std::string()> applets_json;   // the body of GET /v0/applets
        std::function<LaunchOutcome(const std::string& id)> launch;
        // "" when the app can read its frame back; otherwise why it cannot,
        // which capture.frame answers as capture.unavailable.
        std::string capture_unavailable;
        // Arms a read-back of the frame drawn next; the app then calls
        // capture_ready() or capture_failed().
        std::function<void()> request_capture;
        std::function<bool(const uint8_t* rgba, uint32_t w, uint32_t h, std::string* png)> encode_png;
        // One Activity line (WHEN / EVENT / OBJECT / DETAIL).
        std::function<void(const std::string& event, const std::string& object, const std::string& detail)> log;
        std::ostream* err = nullptr;   // the [session] lines; standard error when null
    };

    struct Options {
        std::string workspace;     // the folder, absolute (workspace_choice)
        bool chosen = false;       // chosen on the command line: opened after the first frame
        std::string from;          // "--workspace", "CALIPER_WORKSPACE" or "default", for the log
        double budget_ms = 4.0;    // the queue's time per frame
        std::size_t max_per_frame = 256;
        double capture_timeout_s = 5.0;   // a read-back that never arrives fails then
        std::string script;        // CALIPER_COMMAND_SCRIPT
        std::string results;       // CALIPER_COMMAND_RESULTS
    };

    // The ticket under which the app's WindowCaptures serves the session's
    // capture, beside the API's own tickets.
    static constexpr uint64_t kCaptureTicket = ~uint64_t(0);

    AppSession();
    ~AppSession();   // close()
    AppSession(const AppSession&) = delete;
    AppSession& operator=(const AppSession&) = delete;

    // On the frame thread, before the first frame. Opens nothing.
    void start(Options options, Hooks hooks);
    // On the frame thread, once per frame, between two frames.
    void frame();
    // The app's read-back of the frame drawn after request_capture(): RGBA8,
    // top row first. Frame thread.
    void capture_ready(uint32_t w, uint32_t h, std::vector<uint8_t> rgba);
    void capture_failed(const std::string& why);
    // Before the host's stores close; the destructor calls it too.
    void close();

    // Asks for the workspace to be opened, if it is not; never waits. Any thread.
    void ensure_open();
    // Opens the workspace if needed and waits until the frame thread has
    // adopted it, at most `limit`. Any thread but the frame thread, where it
    // never waits. Null when it is not open (open_error() then says why, or it
    // is still opening). The session stays valid until close().
    caliper_builder::Session* wait_open(std::chrono::milliseconds limit);
    // The adopted session, or null. Frame thread.
    caliper_builder::Session* session();
    // Why the workspace could not be opened; "" otherwise.
    std::string open_error() const;
    const std::string& workspace() const;

    // A long command runs on the worker: the frame thread must leave the
    // session's queue, registry, store and catalog alone this frame.
    bool worker_busy() const;
    bool script_running() const;

    // The spec runtime's seam for panel.render; null unbinds it.
    void set_panel_renderer(std::shared_ptr<PanelRenderSeam> renderer);

    // Counters the spec page (spec_page.h) reads on the frame thread to know
    // when to look at the session again, without asking it every frame. Any
    // thread may read them; the journal's observer moves them on the thread
    // that ran the command.
    // Every command the journal recorded: any of them may have changed the
    // open spec.
    uint64_t journaled() const;
    // Every applet.open that succeeded, from any origin: the app shows the
    // spec page for each.
    uint64_t applets_opened() const;
    // Every applet.save that succeeded: the app lists the workspace's
    // applets again for each.
    uint64_t applets_saved() const;

    // Caliper API's command gateway over this session (api/command_gateway.h),
    // for AppApi::set_command_gateway(), so that the app serves
    // /v0/commands. It serves the command table at once. A command that
    // comes before the workspace is open opens it, the default workspace on
    // first use as wait_open() does, and goes to the queue once the frame
    // thread has adopted the session; when the workspace cannot be opened it
    // answers command.not_available with the reason. GET /v0/approvals/{id}
    // reads the pending approvals as the last frame without a worker left
    // them, so a lookup never waits for a loader that holds the registry.
    // After close() its commands answer command.shutdown. The app clears it
    // from the API before the session closes. Null without the builder.
    std::shared_ptr<caliper_api::CommandGateway> command_gateway();

    struct Impl;

private:
    std::unique_ptr<Impl> impl_;
};

}  // namespace caliper_host
