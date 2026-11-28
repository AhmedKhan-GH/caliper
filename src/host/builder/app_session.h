#pragma once
// ============================================================================
// app_session.h: the builder session in the app (design/builder-session.md,
// design/builder-overview.md, item 2 of the integration order). The app, and
// `caliper shot`, own one AppSession for the workspace of the command line
// (host_cli.h, workspace_choice). It holds a caliper_builder::Session, drains
// its command queue on the frame thread between two frames, and implements the
// session's SessionHost for app.info, app.instruments, app.launch, capture.frame
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
// object and, when it failed, its error code as the detail, or what let it
// run without a card ("allowed by accept_edits"), and one "[session]" line on
// standard error.
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

#if CALIPER_WITH_BUILDER
#include <caliper_builder/commands/registry.h>
#include <caliper_builder/session/events.h>
#include <caliper_builder/session/run_control.h>
#include "live/live_attach.h"

#include <map>
#endif

namespace caliper_builder {
class Session;
class RunControl;
class ModelRunner;
class ProgramHost;
class DrawStatusHost;
}

namespace caliper_live {
class RunLink;   // live/live_attach.h
}

namespace caliper_net {
class Json;   // net/json.h
}

namespace caliper_api {
class CommandGateway;   // api/command_gateway.h
}

namespace caliper_host {

class PanelRenderSeam;   // panel_renderer.h

#if CALIPER_WITH_BUILDER
// The app's run control (caliper_builder::RunControl, design/instruments.md,
// Run Control in App): run.state, run.set_param, run.action and run.pause act
// on the attached script through a caliper_live::RunLink, which in the app is
// the attachment's RunControls (live_controls.h). Each request goes out as a
// protocol 3.1 CONTROL message through the app's own viewer connection, as
// the run bar sends it; the agent never talks to the script. A request fails
// with run.not_attached without a script, run.control_not_allowed when the
// connection takes no control, run.no_step for a pause, a step or an action
// before the first step, and run.refused for anything else RunControls
// refuses, with its reason.
//
// feed() runs once per frame and turns what the script's CONTROLS and its
// CONTROL_REPLYs changed into run events (events.h): `attach` when a script is
// attached or detached, `state` when the run state changes, `param` when the
// script applies a value (the value after its clamp, and the step), and
// `action` when an action ends with ok or error. The origin of each is
// "script". The first CONTROLS after an attach is the baseline: its values
// are registrations, not changes. Frame thread.
class AppRunControl final : public caliper_builder::RunControl {
public:
    explicit AppRunControl(caliper_live::RunLink* link = nullptr) : link_(link) {}
    // The link to act through; null detaches. Frame thread.
    void set_link(caliper_live::RunLink* link) { link_ = link; }
    caliper_live::RunLink* link() const { return link_; }
    void set_model_runner(caliper_builder::ModelRunner* runner) { runner_ = runner; }

    caliper_builder::RunState state() override;
    nlohmann::json progress_state() override;
    caliper_builder::Error set_param(const std::string& name, const nlohmann::json& value) override;
    std::string param_error(const std::string& name) const override {
        return link_ ? link_->param_error(name) : std::string();
    }
    std::string control_session() const override {
        return link_ ? link_->control_session() : std::string();
    }
    caliper_builder::Error run_action(const std::string& name) override;
    caliper_builder::Error pause(caliper_builder::RunPauseOp op, std::int64_t count) override;

    // The interventions (design/gpu-programs.md, Intervention Wire): the
    // points and the interventions come from the CONTROLS key
    // "interventions", which a script sends only when it announced the
    // capability "interventions"; a change goes out as CONTROL op
    // "intervene" through the link, and fails as the requests above do.
    caliper_builder::CommandOutput interpret(const std::string& method, const nlohmann::json& params,
                                               std::uint32_t request) override;
    bool interventions_supported() override;
    std::vector<caliper_builder::InterventionPoint> intervention_points() override;
    std::vector<caliper_builder::Intervention> interventions() override;
    caliper_builder::Error intervene_set(const caliper_builder::Intervention& intervention) override;
    caliper_builder::Error intervene_clear(const std::string& point) override;

    // Pushes the run events of the changes since the last call into `sink`,
    // with `intervene` when the script applied an intervention (its point,
    // its op and the step it applied it at) or stopped one (op null). The
    // interventions applied when the link dropped stay known: when the same
    // script attaches again, each one that it no longer applies, such as one
    // it ended because no viewer that may control stayed connected, is a
    // stop (op null) at the script's step.
    void feed(caliper_builder::EventSink& sink);
    // The `intervene` events of the CONTROLS key "interventions" against the
    // last one seen; `baseline` records the new ones without events, and
    // still reports a stop of one known before it. Called by feed().
    void feed_interventions(const nlohmann::json& interventions, const nlohmann::json& step, bool baseline,
                            caliper_builder::EventSink& sink);

private:
    caliper_builder::Error send(const std::string& op, const std::string& name, const nlohmann::json& value,
                                std::int64_t count);
    caliper_builder::Error send_net(const std::string& op, const std::string& name, const caliper_net::Json& value,
                                    std::int64_t count);

    caliper_live::RunLink* link_ = nullptr;
    caliper_builder::ModelRunner* runner_ = nullptr;
    std::map<std::string, double> starting_since_;
    // What feed() saw last.
    bool attached_ = false;
    std::string script_;
    bool have_ = false;                          // a CONTROLS since the attach
    std::uint64_t version_ = 0;                  // its version
    std::string run_state_ = "none";
    std::map<std::string, std::pair<nlohmann::json, nlohmann::json>> params_;   // name: value, pending
    std::map<std::string, std::uint64_t> results_;                              // action: at_ns of its last result
    std::map<std::string, nlohmann::json> applied_;    // canonical specification key: applied event fields
    std::string dropped_script_;                                                // the script of the dropped link
    std::map<std::string, nlohmann::json> dropped_;    // its applied_ then
};

// The points and the interventions of a CONTROLS key "interventions"
// ({points, active}) in the run control's form, the pending interventions
// first; empty for anything else.
std::vector<caliper_builder::InterventionPoint> intervention_points_of(const nlohmann::json& interventions);
std::vector<caliper_builder::Intervention> interventions_of(const nlohmann::json& interventions);
#endif

// What the app's navigation answers a launch by manifest id, for POST
// /v0/instruments/{id}/launch and app.launch: status 200 with the body of
// launched_json(), or 404 or 409 with a message, and the command's error code
// (app.instrument_not_found, app.launch_refused, app.launch_failed).
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
        std::function<std::string()> instruments_json;   // the body of GET /v0/instruments
        // `origin` is the command origin of whoever asked ("ui", "api",
        // "mcp", "agent"), which an instrument's instrument.open is sent from.
        std::function<LaunchOutcome(const std::string& id, const std::string& origin)> launch;
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
        // The saved agent permissions (Session's SessionOptions::
        // permissions_file): agent-permissions.json in the app-data folder.
        std::string permissions_file;
        // Other settings files that no command may write
        // (SessionOptions::protected_files).
        std::vector<std::string> protected_files;
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
    // The spec page's answer to camera.get and camera.set (design/instrument-interface.md,
    // Commands), from the scene view that a panel shows; empty unbinds it, and the commands
    // then answer command.not_available. Frame thread, as the commands run.
#if CALIPER_WITH_BUILDER
    using CameraSeam = std::function<caliper_builder::CommandOutput(const std::string& name, const nlohmann::json& args)>;
    void set_camera_seam(CameraSeam seam);
#endif

    // The attached script's run (live/live_attach.h): the app gives its
    // attachment's link once, and null before the attachment goes. The
    // session's run control (run_control()) acts through it, and frame() feeds
    // the run events of what changed into the session's event ring. Frame
    // thread.
    void set_run_link(caliper_live::RunLink* link);
    // The app's run control, which the session holds from its adoption until
    // close(): what the spec runtime's controls act through
    // (SpecLiveHost::run). Null without the builder.
    caliper_builder::RunControl* run_control();
    // The app's model runner (src/host/runner/model_runner.h,
    // design/hosted-runs.md), which the session holds from its adoption
    // until close() for run.list, run.start, run.fork, run.stop, run.logs and
    // the argument `run` of the run commands; null takes it back. The app
    // gives it before the first frame and takes it back before the runner
    // goes. Frame thread.
    void set_model_runner(caliper_builder::ModelRunner* runner);
    // The hosts of the open spec's GPU programs (in the attached script) and
    // the draw halves of its pieces (in the viewer), which the session holds from its adoption until
    // close() for program.status and its reconciles
    // (caliper_builder/session/program_host.h, design/gpu-programs.md); null
    // takes one back. The app gives them before the first frame and takes
    // them back before they go. Frame thread.
    void set_program_host(caliper_builder::ProgramHost* host);
    void set_draw_host(caliper_builder::DrawStatusHost* host);

    // Counters the spec page (spec_page.h) reads on the frame thread to know
    // when to look at the session again, without asking it every frame. Any
    // thread may read them; the journal's observer moves them on the thread
    // that ran the command.
    // Every command the journal recorded: any of them may have changed the
    // open spec.
    uint64_t journaled() const;
    // Every instrument.open that succeeded, from any origin, and every spec.new
    // or spec.open of an agent (the origins agent and mcp) that succeeded:
    // the app shows the spec page for each, so that the person sees the spec
    // whose edits they review. The person's own spec.new shows the page by
    // its own path (SpecPage::take_new()), in build mode.
    uint64_t specs_opened() const;
    // Of those, the ones an agent sent (the origins agent and mcp): an
    // instrument.open, a spec.new or a spec.open.
    uint64_t agent_specs_opened() const;
    // Every instrument.save that succeeded: the app lists the workspace's
    // instruments again for each.
    uint64_t instruments_saved() const;

    // The commands that ran without a card because an agent's mode or a rule
    // allowed them (the journal's allowed_by), since the last call, the
    // newest 64 at most: the side chat shows its agent's as APPLIED rows.
    // Frame thread.
    struct Applied {
        uint64_t seq = 0;          // the journal's seq
        std::string name;          // the command
        std::string origin;        // agent or mcp
        std::string allowed_by;    // accept_edits, allow_all, session_rule, saved_rule
    };
    std::vector<Applied> take_applied();

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

// The command script may claim the ui origin and approve what waits, so it
// runs only where CALIPER_TEST_APPROVALS does: in the hidden app (--hidden)
// and in `caliper shot`, which shows no window (`scripts_allowed`). Otherwise
// the script and its results file are dropped from `o`, and `refused` (when
// given) says why. app_session_options() (app_hooks.h) applies it.
void gate_command_script(AppSession::Options* o, bool scripts_allowed, std::string* refused);

// What every page's status bar says about Allow all, "" for nothing: "allow
// all for side chat", "allow all for MCP clients" or both, and "allow all in
// the saved file for ..." while the file holds Allow all for a principal that
// this app runs in another mode: after a failed save of that mode, or when a
// second app turned it on (Session::saved_agent_mode()). The session also
// says it on standard error when it opens, for a run with no status bar.
// Built with the builder only.
std::string allow_all_status(const caliper_builder::Session& s);
// The Permissions section's note after a failed save, "" when the file holds
// what the session holds: why, which mode the file still holds and returns at
// the next launch, and that choosing the mode again saves it.
std::string permission_save_note(const caliper_builder::Session& s);

}  // namespace caliper_host
