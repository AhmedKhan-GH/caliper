// live_attach.h: "Attach to process" for a host that owns its window and its
// renderer (the Caliper app's landing and live page; host-internal C++).
//
// Two pieces, both thin over the live core so the protocol lives in one place:
//   SessionList  the live sessions under <data_root>/live and the streams of
//                caliper.serve() under <data_root>/streams, relisted at most
//                once a second; caliper_live::list_sessions reclaims a dead or
//                reused pid's session as it reads (so a SIGKILLed script's row
//                disappears) and marks the rest (live_source.h SessionState),
//                and list_streams does the same for a stream's file. A stream's
//                row carries its stream URL, which attach_url() takes.
//   Attachment   one attached session drawn through a hosted embed core
//                (embed_core.h, caliper_core_create_hosted): caliper_core_live_*
//                claim and apply the newest completed snapshot of every view,
//                the typed live views draw them inside the host's ImGui frame,
//                and the transitions a host logs (attached, session ended,
//                detached) are queued as events. While attached it holds the
//                viewer activity (live_activity.h), so a hidden window keeps
//                its heartbeat. Nothing here waits on the training process:
//                polling is a few lock-free loads and stores on shared memory,
//                so training never waits on the host either.
//
// caliper-viewer drives the same core through the embed C ABI; its picker and
// the app's session list share the row rules (session_state, state_text,
// clock_time, age_text). Frame-thread only, like the embed views.
#pragma once

#include "live/live_source.h"

#include <cstdint>
#include <string>
#include <vector>

struct CaliperCore;
namespace caliper_host { class HostRenderer; }

namespace caliper_live {

class RunControls;

// The attached script's run as the builder session's run control reads and
// drives it (src/host/builder/app_session.h, design/instruments.md, Run
// Control in App): in the app, the attachment and its RunControls
// (AttachmentRunLink, live_controls.h); in the tests, a fake. A request goes
// out as RunControls sends it from the run bar, through the app's own viewer
// connection. Frame thread.
class RunLink {
public:
    // The last result of an action: its text, or why it failed, and when it
    // arrived (now_ns() clock); at_ns 0 when there is none.
    struct ActionResult {
        std::string text;
        bool error = false;
        uint64_t at_ns = 0;
    };

    virtual ~RunLink() = default;
    // Whether a script is attached, and its name.
    virtual bool attached() const = 0;
    virtual std::string script() const = 0;
    // The script's latest CONTROLS; null before the first one.
    virtual const Controls* controls() const = 0;
    // Whether the connection takes control at all, and why not.
    virtual ControlAccess access() const = 0;
    // Sends one request: `op` is pause, resume, step (with `count`), set
    // (with `name` and `value`) or run (with `name`). Returns "" when it was
    // sent (or a set value is already what the script shows), else why not,
    // in RunControls' words.
    virtual std::string send(const std::string& op, const std::string& name, const caliper_net::Json& value,
                             uint32_t count) = 0;
    virtual caliper_net::Json interpret(const std::string&, const caliper_net::Json&, uint32_t) {
        auto out = caliper_net::Json::object(), error = caliper_net::Json::object();
        error.set("code", "interpret.unsupported"); error.set("message", "This link does not support interpretation.");
        out.set("error", error); return out;
    }
    // The last result of the action `name`.
    virtual ActionResult action_result(const std::string& name) const = 0;
};

// The state column of a session row: "running", "starting", "closing",
// "viewed by <pid>", "protocol v<n>", and "stream" for a caliper.serve()
// stream.
std::string state_text(const SessionEntry& e);
// Local wall-clock time of a unix-ms stamp, "14:03:27"; "-" for 0.
std::string clock_time(uint64_t unix_ms);
// How long ago, coarse: "12s ago", "3m ago", "2h ago"; "" for 0.
std::string age_text(uint64_t unix_ms);
// The view names, comma-separated; "none yet" when the script has shown none.
std::string view_names(const SessionEntry& e);

class SessionList {
public:
    // `root` "" = the default data root ($CALIPER_DATA_DIR, else the app data
    // folder: ~/Library/Application Support/Caliper on macOS).
    explicit SessionList(std::string root = {});

    // Relists when forced or once a second has passed since the last listing.
    // True when it relisted (the rows may have changed).
    bool refresh(bool force = false);

    const std::vector<SessionEntry>& sessions() const { return sessions_; }
    // Why the last listing failed ("" when it did not).
    const std::string& error() const { return error_; }
    // Index of the session with token `session`, or -1.
    int find(const std::string& session) const;
    // Index of the session of process `pid`, or -1.
    int find_pid(int32_t pid) const;
    // The resolved root the sessions are listed from.
    std::string root() const;

private:
    std::string root_;
    std::vector<SessionEntry> sessions_;
    std::string error_;
    uint64_t listed_ns_ = 0;   // CLOCK_UPTIME_RAW of the last listing, 0 = never
};

class Attachment {
public:
    enum class Event {
        Attached,   // attach() succeeded (the session may still be starting)
        Ended,      // the script exited or died; its last frames stay drawn
        Detached,   // detach(): views released, the script's attached() is false
    };

    struct Status {
        bool        attached = false;
        bool        pending = false;      // waiting for a starting session
        bool        alive = false;        // false once the script has gone
        std::string session;              // the token
        std::string script;
        int32_t     pid = 0;
        uint64_t    start_unix_ms = 0;
        int         views = 0;
        uint64_t    updates = 0;          // view updates applied since attach
        double      fps = 0.0;            // frames drawn per second, last second
        double      latency_ms = 0.0;     // step-to-screen median, last second
    };

    Attachment() = default;
    ~Attachment() { detach(); }
    Attachment(const Attachment&) = delete;
    Attachment& operator=(const Attachment&) = delete;

    // Attaches session `session` through a hosted core over `renderer` (the
    // host's, bound to the tensor bridge). False with *err on a refusal: the
    // one-core slot is taken, unknown token, another viewer attached, no Metal
    // device. A session that is still starting attaches and waits.
    bool attach(caliper_host::HostRenderer* renderer, const std::string& session,
                std::string* err);
    // The same for a wire protocol v3 stream by its link (caliper_core_live_
    // attach_url: ws://127.0.0.1:<port>/v3/stream?token=..., or the page link
    // http://127.0.0.1:<port>/?token=...). Status::session is the session id.
    bool attach_url(caliper_host::HostRenderer* renderer, const std::string& url,
                    std::string* err);
    bool attached() const { return core_ != nullptr; }

    // Once per host frame, OUTSIDE the ImGui frame (like caliper_core_live_poll
    // before caliper_core_frame): claims and applies the newest completed
    // snapshot of every view whose previous GPU read retired, writes the
    // heartbeat, and notices the script's exit. Returns the views updated.
    int poll();
    // Inside the host's ImGui frame: the session's views, tiled into the work
    // area (x, y, w, h in logical points; `scale` physical px per point).
    void draw(float x, float y, float w, float h, float scale);
    // Releases every view (the renderer must still be bound to the bridge, and
    // the host must have rendered the frame that drew them), clears the
    // heartbeat and frees the core. Safe when not attached.
    void detach();

    const Status& status() const { return status_; }
    // Transitions since the last call, oldest first.
    std::vector<Event> take_events();
    // The hosted core (tests and diagnostics: the caliper_core_live_view_*
    // probes); null when not attached.
    CaliperCore* core() const { return core_; }
    // A network stream's controls and selections (wire protocol 3.1,
    // live_controls.h), updated by poll(); null when not attached to a stream.
    RunControls* controls() const;

private:
    void refresh_status(bool reset_window);
    // The common part of attach and attach_url: a hosted core, then `go` on it.
    bool attach_with(caliper_host::HostRenderer* renderer, int (*go)(CaliperCore*, const char*),
                     const std::string& what, const char* prefix, std::string* err);

    CaliperCore*       core_ = nullptr;
    void*              activity_ = nullptr;   // live_activity.h, while attached
    Status             status_;
    bool               ended_ = false;
    uint64_t           window_ns_ = 0;   // start of the fps/latency window
    std::vector<Event> events_;
};

}  // namespace caliper_live
