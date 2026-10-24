// live_source.h: the transport seam of the live views (host-internal).
//
// live_core.cpp turns snapshots into typed live views (caliper/live_views.h);
// it does not know where a snapshot came from. A Source delivers, per view,
// "the newest completed snapshot": an MTLBuffer + byte offsets (payload, aux),
// the dtype/shape, a CPU pointer to the producer's stats, the mapping
// (colormap, fixed or GPU-computed vmin/vmax) and its seq/stamp plus the kind
// metadata (series_total, points_dims, label_count). Today the only Source is
// the shared-memory session of another process (live_session.mm); an
// in-process ring over torch-owned buffers can implement the same interface and
// reuse every line of live_core.cpp.
#pragma once

#include "caliper/live.h"   // CALIPER_LIVE_PROTOCOL_VERSION

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace caliper_live {

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
};

struct SourceView {
    uint64_t    key = 0;            // stable for the view's lifetime
    std::string name;
    uint32_t    kind = 0;
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
};
bool list_sessions(const std::string& root, std::vector<SessionEntry>* out, std::string* err);

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
