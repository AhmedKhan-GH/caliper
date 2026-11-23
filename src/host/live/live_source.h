// live_source.h: the transport seam of the live views (host-internal).
//
// live_core.cpp turns snapshots into typed live views (caliper/live_views.h);
// it does not know where a snapshot came from. A Source delivers, per view,
// "the newest completed snapshot": an MTLBuffer + byte offsets (payload, aux),
// the dtype/shape, a CPU pointer to the producer's stats, the mapping
// (colormap, fixed or GPU-computed vmin/vmax) and its seq/stamp plus the kind
// metadata (series_total, points_dims, label_count). Two Sources exist: the
// shared-memory session of another process (live_session.mm) and a wire
// protocol v3 stream over a loopback WebSocket (live_net_source.h); an
// in-process ring over torch-owned buffers can implement the same interface and
// reuse every line of live_core.cpp.
#pragma once

#include "caliper/live.h"   // CALIPER_LIVE_PROTOCOL_VERSION
#include "net/json.h"       // the JSON values of wire protocol 3.1 controls
#include "net/wire.h"       // the scene messages of wire protocol 3.2

#include <cstdint>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <variant>
#include <vector>

namespace caliper_live {

// How a payload maps onto its source (wire protocol v3, "Payload axes"), so
// that a view reports what a person picked or selected in source indices.
// Payload index i along an axis stands for the source positions
// start + step * [floor(i * count / size), ceil((i + 1) * count / size)).
// An empty map means payload index = source index (the shared-memory
// session, where the payload is the whole source).
struct PayloadAxis {
    std::string dim;
    int64_t start = 0, step = 1, count = 0, size = 0;
};
struct PayloadMap {
    std::vector<PayloadAxis> axes;   // payload order; a tiled heatmap's tile axis first
    int64_t tile_rows = 0, tile_cols = 0;   // a tiled heatmap's mosaic of cells, 0 = not tiled
};

struct Snapshot {
    void*    mtl_buffer = nullptr;  // id<MTLBuffer>, borrowed: valid while the view exists
    void*    host_base = nullptr;   // CPU address of the same buffer (shared storage)
    uint64_t byte_offset = 0;       // payload start inside mtl_buffer
    uint64_t aux_offset = 0;        // aux start (POINTS), valid when aux_count > 0
    int64_t  aux_count = 0;         // aux f32 elements, 0 = none
    uint32_t kind = 0;              // CaliperLiveKind
    int32_t  dtype = 0;             // CaliperDType
    int32_t  ndim = 0;
    int64_t  shape[4] = {};
    int32_t  colormap = 0;
    uint32_t flags = 0;             // CALIPER_LIVE_SNAP_*
    float    vmin = 0.0f, vmax = 1.0f;   // fixed ends (used for the non-AUTO ends)
    const float* stats = nullptr;   // producer stats (CPU-readable, in the claimed slot)
    int32_t  stats_count = 0;
    uint64_t series_total = 0;      // SERIES
    int32_t  points_dims = 0;       // POINTS
    int32_t  label_count = 0;       // POINTS
    uint32_t slot = 0;
    uint64_t seq = 0;               // producer's publish counter for this view
    uint64_t stamp_ns = 0;          // producer CLOCK_UPTIME_RAW at show()
    /// The step stamp of the snapshot (the first timeline of its stamp section); INT64_MIN when none.
    int64_t step = INT64_MIN;
    const PayloadMap* map = nullptr; // borrowed until the next take() of the view; null = identity
    // Bytes the source copied to make this payload: a network snapshot
    // received into its slot, or series values appended to the ring; 0 when
    // the producer wrote the slot in place (the shared-memory session). The
    // viewer counts it as one copy (embed/render_stats.h).
    uint64_t copied_bytes = 0;
};

struct SourceView {
    uint64_t    key = 0;            // stable for the view's lifetime
    std::string name;
    uint32_t    kind = 0;
};

// ---- wire protocol 3.2 draft: scenes ------------------------------------------------
// A scene view (kind 6) has no snapshots: its source hands over decoded messages,
// in their order, to the scene view (live_views_scene.h). A message's sections
// point into `keep`, the received frame, which lives as long as the item. A note
// is something the source ignored for the view, as the versioning rules of 3.2
// say (a part, a form or a section this viewer does not build).
struct SceneItem {
    std::shared_ptr<const void> keep;
    std::variant<caliper_net::wire::SceneUpdate, caliper_net::wire::SceneClear, std::string,
                 caliper_net::wire::SceneTransforms, caliper_net::wire::SceneClasses> msg;
};

// ---- wire protocol 3.1: controls and linked selection --------------------------
// What a producer that takes controls registered, as its latest CONTROLS says
// (wire protocol v3, "Controls"). Values are JSON scalars as they travel
// (number, boolean or string); `pending` is null when nothing waits.
struct ControlParam {
    std::string name, type, help;             // type: number, integer, boolean, choice
    caliper_net::Json value, pending;
    bool has_min = false, has_max = false, log = false;
    double min = 0.0, max = 0.0;
    std::vector<std::string> choices;
};
struct ControlAction {
    std::string name, help;
    bool running = false, queued = false;
};
struct Controls {
    std::string run_state = "none";           // none, running, pausing, paused, stepping
    bool has_step = false;
    uint64_t step = 0, steps_left = 0;
    std::vector<ControlParam> params;
    std::vector<ControlAction> actions;
    caliper_net::Json interventions;          // the key "interventions" ({points, active}) as given; null when absent
    uint64_t version = 0;                     // grows with every CONTROLS received
};
// Whether this viewer may send CONTROL, by the protocol's "Who may control":
// the server announced `controls` and set WELCOME's `control`, the attached
// session lists `controls` and is live. `why` says what is missing.
struct ControlAccess {
    bool allowed = false;
    std::string why;
};
// One CONTROL_REPLY, with the request id that send_control returned.
struct ControlReply {
    uint32_t request_id = 0;
    std::string op, status, name;             // status: ok, queued, error
    caliper_net::Json value;                  // set with ok: the accepted value
    std::string message;                      // run with ok: the action's result text
    std::string error_code, error_message;    // status error
    uint64_t controls_version = 0;            // Controls::version when it arrived
};
// The current selection of one item space (SELECTION), with its details.
struct LinkedSelection {
    std::string space, view;                  // view: "" when the script set it
    std::vector<uint64_t> items;              // ascending item ids; empty: nothing selected
    bool truncated = false;
    caliper_net::Json details;                // {"fields", "items", "values"}, null when none
    std::string details_error;
    uint64_t seq = 0;                         // arrival order among all selections
};
// Which axes of a view index the items of a space (VIEWS: `items` on a source
// dim or on a series view), for the highlight of a linked selection.
struct ItemAxis {
    std::string view, source, dim;            // dim "" with source "": a series' own index
    std::string space;
};

// THE CONTRACT (an in-process Source implements exactly this).
//
// Threading: every method is called from ONE thread, the embed core's frame
// thread (inside caliper_core_live_poll / _status / the frame hook), never
// concurrently with itself. The PRODUCER (training thread or process) runs
// concurrently with all of them; synchronisation with it is the Source's job.
//
// Per poll, live_core.cpp calls, in this order: heartbeat(), producer_alive(),
// sync(), then per view: release_prev() once the previous payload's GPU read
// retired, at most one take() (only while no previous claim is held), and after
// it either drew() (applied) or unclaim() (the apply failed).
//
// One version of a published resource of a program, as the stream received it
// (RESOURCE of wire protocol 3.2, design/programmable-graphics.md, Resource
// Message): the run's step at the pass, the valid elements and their bytes.
struct ReceivedResource {
    std::string program, resource;
    int64_t step = 0;
    uint32_t count = 0;                  // valid elements
    uint32_t stride = 0;                 // bytes of one element
    uint32_t width = 0, height = 0;      // a texture's size; 0 for a buffer
    std::string format;                  // a texture's format; empty for rgba8unorm or a buffer
    std::optional<uint32_t> elements;    // a buffer of narrow elements: their count, of which count holds the words
    std::shared_ptr<const std::vector<uint8_t>> bytes;   // count * stride bytes
};

// Claims (what a producer may rely on): a view has at most two claimed
// snapshots. take() moves the current claim to "previous" and claims the
// newest completed snapshot; the previous one stays claimed until
// release_prev() (called once caliper_core_live_view_read_done says its GPU
// read retired: a points view draws its payload every frame until the next
// snapshot's first frame completed). SERIES snapshots are never claimed (an
// append-only ring the viewer copies). The producer never writes a claimed
// slot nor the slot it last published; every other slot is free for it.
class Source {
public:
    virtual ~Source() = default;
    virtual std::string label() const = 0;          // the session token (status display)
    virtual std::string script() const = 0;         // producer's script name ("" unknown)
    virtual uint64_t start_unix_ms() const = 0;
    virtual int32_t producer_pid() const = 0;       // 0 when not a separate process
    // False once the producer has gone (closed or died). Views keep their last
    // pixels. Called once per poll; must be cheap.
    virtual bool producer_alive() = 0;
    // Refresh the view list. Returns false (and leaves *views alone) when
    // nothing changed; true with the FULL new list otherwise. Views keep their
    // `key` for their lifetime; a re-added name gets a new key.
    virtual bool sync(std::vector<SourceView>* views) = 0;
    // Claim the newest COMPLETED snapshot of view `key` with seq > after_seq
    // (after_seq 0 = anything). Routinely called with nothing new: it must then
    // return false quickly and leave the claims unchanged. On true, the
    // previous current claim became the "previous" claim and *out is valid
    // (mtl_buffer borrowed for as long as the view exists in the source).
    virtual bool take(uint64_t key, uint64_t after_seq, Snapshot* out) = 0;
    // The previous claim's GPU read retired: free it for the producer.
    virtual void release_prev(uint64_t key) = 0;
    // The snapshot of the last take() was not applied: return to the claims
    // before it (the old current claim is still what the view shows).
    virtual void unclaim(uint64_t key) = 0;
    // What the consumer drew: seq of the applied snapshot and how many
    // publishes it skipped since the last one (diagnostics only).
    virtual void drew(uint64_t key, uint64_t seq, uint64_t skipped) = 0;
    // Producer-side dropped (busy) snapshots of a view (status only).
    virtual uint64_t producer_dropped(uint64_t key) = 0;
    // Once per poll: tell the producer a consumer is alive (its attached()
    // turns true; it snapshots only while this keeps coming).
    virtual void heartbeat() = 0;

    // ---- outbound: what the viewer tells the producer ------------------------
    // A Source that cannot send keeps these defaults (the shared-memory session
    // has no channel back). Both are called on the frame thread and never wait:
    // a network source queues what it sends.
    //
    // What a person did in view `key` (the view layer's event sink, caliper_
    // live_views::set_event_sink): `type` is "select", "pick" or "hover" and
    // `detail` a JSON object in source indices (wire protocol v3, "Events").
    // The Source adds what the viewer was showing (subscription, seq, data
    // view). False when it was not sent.
    virtual bool send_event(uint64_t key, const std::string& type, const std::string& detail_json) {
        (void)key; (void)type; (void)detail_json;
        return false;
    }
    // Ask the producer for another data view of view `key` (a JSON object;
    // wire protocol v3, "Data views"). The snapshots that follow carry the
    // resolved one. False with *err when it was not sent.
    virtual bool request_data_view(uint64_t key, const std::string& data_view_json, std::string* err) {
        (void)key; (void)data_view_json;
        if (err) *err = "this source cannot request data views";
        return false;
    }

    // ---- wire protocol 3.1 (frame thread) -------------------------------------
    // A Source without a channel back, or a producer of 3.0, keeps these
    // defaults: no controls and no linked selection.
    //
    // The latest CONTROLS: false (and *out left alone) when none has arrived
    // for the attached session.
    virtual bool controls(Controls* out) { (void)out; return false; }
    virtual ControlAccess control_access() { return {false, "this stream takes no controls"}; }
    // Sends CONTROL `op` (pause, resume, step, set, run) for parameter or action
    // `name` with `value` (set) or `count` (step). Returns the request id its
    // CONTROL_REPLY will carry, 0 with *err when it was not sent.
    virtual uint32_t send_control(const std::string& op, const std::string& name,
                                  const caliper_net::Json& value, uint32_t count, std::string* err) {
        (void)op; (void)name; (void)value; (void)count;
        if (err) *err = "this stream takes no controls";
        return 0;
    }
    // The CONTROL_REPLYs received since the last call, oldest first.
    virtual std::vector<ControlReply> take_control_replies() { return {}; }
    // The current selection of every item space and a version that grows with
    // every change; 0 when none ever arrived.
    virtual uint64_t selections(std::vector<LinkedSelection>* out) { (void)out; return 0; }
    // The views' axes that index item spaces.
    virtual std::vector<ItemAxis> item_axes() { return {}; }
    // The resolved data view and the offer (VIEWS entry) of view `key`, as JSON
    // objects; false when unknown.
    virtual bool data_view_of(uint64_t key, caliper_net::Json* resolved, caliper_net::Json* offer) {
        (void)key; (void)resolved; (void)offer;
        return false;
    }

    // ---- wire protocol 3.2 draft: scenes (frame thread) -------------------------
    // The scene messages of view `key` (kind 6) that arrived since the last call,
    // oldest first, appended to *out; false when there are none. A source that
    // holds messages the frame thread has not taken keeps only what still
    // matters: a reset drops what came before it, and a newer update of an
    // entity part replaces an older one queued after the last clear of it.
    virtual bool take_scene(uint64_t key, std::vector<SceneItem>* out) {
        (void)key; (void)out;
        return false;
    }
    // The frame thread will never show scene view `key` (the core refused it for good):
    // the source ends its subscription and drops what still arrives for it.
    virtual void stop_scene(uint64_t key) { (void)key; }

    // ---- wire protocol 3.2 draft: the views' cadence (any thread) ----------------
    // The shared cadence that the newest snapshot of each view carried (the
    // SNAPSHOT key `cadence`: an object with every, passes_per_s, step_ms,
    // pass_ms, share, budget and limit), as JSON text by view name; a view
    // whose snapshots carry none is left out.
    virtual std::map<std::string, std::string> view_cadences() { return {}; }

    // ---- wire protocol 3.2 draft: resources (frame thread) -----------------------
    // Whether the attached script takes instances of pieces: it and the server
    // announced `programs` and `resources`, and programs may go out now.
    virtual bool takes_pieces() { return false; }
    // The RESOURCE versions since the last call, the newest of each resource,
    // in the order each began to wait.
    virtual std::vector<ReceivedResource> take_resources() { return {}; }
};

// Why open_shm_source returned NULL.
enum class OpenFail {
    None,
    Missing,    // no control segment (yet): a session file may say it is starting
    Starting,   // the control segment exists but is not initialised yet: retry
    Refused     // wrong magic/version/layout, another viewer attached, no device
};

// Shared-memory session of another process (live_session.mm). `mtl_device` is
// the renderer's id<MTLDevice>: each segment's MTLBuffer is created on it, so
// the embed core's import accepts it (registryID match). NULL + *err + *why on
// failure.
std::unique_ptr<Source> open_shm_source(const std::string& session, void* mtl_device,
                                        std::string* err, OpenFail* why);

// Session files (live_session.mm): the live sessions under <root>/live after
// reclaiming stale ones. Also used by the picker and attach-by-pid.
struct SessionEntry {
    std::string path;
    std::string session;
    std::string script;
    int32_t     pid = 0;
    int32_t     protocol = 0;
    uint64_t    start_unix_ms = 0;
    int32_t     view_count = 0;
    int32_t     viewer_pid = 0;
    bool        alive = false;      // the process runs and its session is open
    bool        starting = false;   // the process runs, its control segment is not there yet
    std::vector<std::string> views; // view names, from the session file
    // A caliper.serve() stream (list_streams): its stream URL with the
    // session token, ws://127.0.0.1:<port>/v3/stream?token=..., which attaches
    // it (caliper_core_live_attach_url); "" for a shared-memory session.
    std::string stream_url;
    bool        control = false;    // a stream: its viewers may control the script
};
bool list_sessions(const std::string& root, std::vector<SessionEntry>* out, std::string* err);

// Stream files (live_session.mm): the streams of caliper.serve() on this
// machine, <root>/streams/<pid>.json, which python/caliper/_net.py
// (StreamFile) writes while its server listens. A file is listed only when it
// is a regular file of this user that no one else can read (mode 0600), of
// version 1, written on this host (its "host" is this machine's name), by a
// process that still runs and started before the file's start time (a reused
// pid's file is stale), with a stream URL on 127.0.0.1 at its port. A file of
// a process that has gone is removed, as list_sessions reclaims a killed
// script's session; one of another user or host is left alone. Every entry is
// alive and has a stream_url.
bool list_streams(const std::string& root, std::vector<SessionEntry>* out, std::string* err);
inline std::string streams_dir(const std::string& root) { return root + "/streams"; }

// What a listed session offers a viewer choosing one to attach to. One rule for
// caliper-viewer's picker and the Caliper app's session list (live_attach.h).
enum class SessionState {
    Running,        // alive, nobody attached: attach
    Starting,       // its control segment is not initialised yet: attach and wait
    Closing,        // the script is closing its session
    Viewed,         // another viewer holds it (one viewer per session)
    Incompatible    // written by another protocol version
};
inline SessionState session_state(const SessionEntry& e) {
    if (!e.stream_url.empty()) return SessionState::Running;   // listed only while its server listens
    if (e.protocol != (int32_t)CALIPER_LIVE_PROTOCOL_VERSION) return SessionState::Incompatible;
    if (e.starting) return SessionState::Starting;
    if (!e.alive) return SessionState::Closing;
    if (e.viewer_pid) return SessionState::Viewed;
    return SessionState::Running;
}
inline bool can_attach(const SessionEntry& e) {
    const SessionState s = session_state(e);
    return s == SessionState::Running || s == SessionState::Starting;
}

}  // namespace caliper_live
