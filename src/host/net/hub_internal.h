#pragma once
// hub_internal.h: private data structures and Hub::Impl declaration shared by
// the hub translation units (src/host/net/hub*.cpp). Not part of the public
// interface; external callers include net/hub.h.
#include "net/hub.h"

#include "caliper/live_net.h"
#include "net/http_door.h"
#include "net/json.h"
#include "net/wire.h"
#include "net/ws.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <set>
#include <string>
#include <utility>
#include <vector>

namespace caliper_net {

namespace hub_detail {

inline uint64_t now_us() {
    using namespace std::chrono;
    return (uint64_t)duration_cast<microseconds>(steady_clock::now().time_since_epoch()).count();
}

constexpr double kDefaultRate = CALIPER_NET_DEFAULT_RATE_HZ;
constexpr double kLimitRate = CALIPER_NET_LIMIT_RATE_HZ;
// Each viewer subscription gets max_rate messages per second on average, with
// bursts of two: a producer that sends at exactly the rate, with some jitter,
// then passes through without waiting for an interval it just missed.
constexpr double kBurst = 2.0;
constexpr size_t kSeriesKeep = CALIPER_NET_SERIES_RETAIN_MIN;   // a history never shrinks below this
constexpr size_t kEventQueueLimit = size_t(1) << 20;            // producer send queue that drops events
constexpr uint16_t kCloseOverflow = 1008;                        // RFC 6455 policy violation

inline std::string canon(const Json& j) { return j.is_object() ? j.dump() : std::string("{}"); }

// The latest snapshot of a channel: the producer's frame, shared by every
// viewer connection that sends it, and what is needed to give it a new head.
struct Snap {
    SharedBytes frame;
    Json meta;                       // the metadata, "sections" included
    std::vector<uint32_t> lens;      // section lengths
    size_t tail = 0;                 // offset of the first section in `frame`
    uint16_t flags = 0;
    size_t bytes() const { return frame ? frame->size() : 0; }
    size_t tail_len() const { return frame ? frame->size() - tail : 0; }
};
using SnapPtr = std::shared_ptr<const Snap>;

// ---- 3.2 draft: scenes (capability "scenes"; docs/reference/wire-protocol-3-2.md) ----
// One entity transform as the hub keeps it: the column-major matrix, and its stamp row (T values,
// none for a producer without timelines).
struct TransformRow {
    float m[16] = {};
    bool is_static = false;
    uint64_t seq = 0, stamp_ns = 0;
    std::vector<int64_t> stamp;
};
using TransformRows = std::map<std::string, TransformRow>;
struct SceneMsg;
using ScenePtr = std::shared_ptr<const SceneMsg>;
// One scene message of a producer (SCENE_UPDATE, SCENE_TRANSFORMS, SCENE_CLEAR or SCENE_CLASSES),
// shared by every viewer connection that sends it; or a message the hub makes: the reset that opens
// a late viewer's state (no frame), the entity transforms it keeps (`rows`, sent as SCENE_TRANSFORMS
// built at the time, the static ones apart), or a split update put together (`pieces`, sent in order).
struct SceneMsg {
    uint16_t type = 0;
    SharedBytes frame;               // null for a message the hub makes
    Json meta;                       // the metadata, "sections" included when there are sections
    std::vector<uint32_t> lens;      // section lengths
    size_t tail = 0;                 // offset of the first section in `frame`
    uint16_t flags = 0;
    std::string path, part;          // part: "" for a clear
    bool is_static = false, recursive = false, reset = false;
    std::vector<std::string> parts;  // a clear's parts; empty: all
    std::vector<ScenePtr> pieces;    // a split update: its pieces, in order; or a bundle of the piece form
    // The piece form (capability "scene_pieces"): `piece_form` marks a bundle, the RESOURCE frames of a
    // scene program and then the update that names them, in `pieces`, sent only to a viewer that
    // announced the capability; `raw` marks one of those RESOURCE frames, sent as it came.
    bool piece_form = false, raw = false;
    size_t pieces_expected = 0;      // a split update being put together: the pieces it has
    std::shared_ptr<const TransformRows> rows;   // the entity transforms the hub keeps
    size_t bytes() const {
        size_t n = frame ? frame->size() : 0;
        for (const ScenePtr& p : pieces) n += p->bytes();
        if (rows) n += rows->size() * 160;
        return n;
    }
    size_t tail_len() const { return frame ? frame->size() - tail : 0; }
};
// (path, part): the byte order of the paths, then the parts in the order points, lines, arrows,
// boxes, mesh, image, camera. The class table is kept under ("", kClassesOrder) and the entity
// transforms under ("", kTransformsOrder), so that a late viewer's state sends them first, in the
// draft's order: the class table, the entity transforms, then the batches.
using SceneKey = std::pair<std::string, int>;
constexpr int kCameraOrder = 6;

struct History {
    uint64_t view_id = 0;
    std::deque<float> v;             // values [first, first + v.size())
    uint64_t first = 0;
    uint64_t total = 0;              // the producer's count of appended values
    uint64_t stamp_ns = 0;
    uint64_t end() const { return first + v.size(); }
};

struct SubKey {
    ConnId conn = 0;
    uint32_t sub = 0;
    bool operator<(const SubKey& o) const { return conn != o.conn ? conn < o.conn : sub < o.sub; }
};

struct Waiter {
    ConnId conn = 0;
    uint32_t sub = 0;
    uint32_t request_id = 0;
    uint64_t ticket = 0;             // valid while the viewer's sub still holds it
    bool view_request = false;
    double rate = kDefaultRate;
    Json requested;                  // the data view the viewer asked for
};

struct Channel {
    uint64_t id = 0;
    std::string view;
    uint64_t view_id = 0;
    int64_t kind = 0;
    uint32_t up_sub = 0;             // the producer subscription, 0 when none
    uint32_t up_request = 0;         // a SUBSCRIBE waiting for its VIEW_REPLY
    double up_rate = 0;
    Json requested = Json::object(); // the data view sent to the producer
    std::set<std::string> aliases;   // requested data views that resolved here
    Json resolved;
    std::string resolved_key;
    bool resolved_ok = false;
    bool keeper = false;             // a series view the hub keeps without viewers
    std::set<SubKey> members;
    std::vector<Waiter> waiters;
    SnapPtr latest;
    uint64_t used_ms = 0;
    // 3.2 draft, a scene: the newest batch of each entity part, for a late
    // viewer; the newest seq; whether the producer has sent the state of this
    // subscription (its reset) since the hub subscribed; whether the hub forgot
    // that state (over its budget, or past CALIPER_NET_MAX_BATCHES batches) and
    // keeps nothing of the scene until the producer's next state.
    std::map<SceneKey, ScenePtr> scene_held;
    uint64_t scene_seq = 0;
    bool scene_seen = false;
    bool scene_forgotten = false;
    // The split updates being put together, oldest first: their update counter and the pieces so far.
    std::vector<std::pair<uint64_t, std::shared_ptr<SceneMsg>>> scene_building;
    bool idle() const { return members.empty() && waiters.empty() && !keeper; }
};

struct ViewInfo {
    uint64_t id = 0;
    int64_t kind = 0;
};

// The latest SELECTION of one item space, as the producer sent it (3.1).
struct SelectionState {
    SharedBytes frame;
    bool empty = true;               // nothing selected: not sent to a late viewer
};

// A CONTROL forwarded to a producer, waiting for its CONTROL_REPLY (3.1).
struct ControlRoute {
    ConnId viewer = 0;
    uint32_t request_id = 0;         // the viewer's own
    std::string op;
    std::optional<std::string> name;
    uint64_t deadline_ms = 0;        // dropped with `timeout` after this
};

// A PROGRAM forwarded to a producer, waiting for its PROGRAM_STATE (3.2 draft).
struct ProgramRoute {
    ConnId viewer = 0;
    uint32_t request_id = 0;         // the viewer's own
    uint64_t deadline_ms = 0;        // dropped with `timeout` after this
};

// One version of a resource as the hub forwards and keeps it: its frame, or the frames of its parts in order.
using ResourceVersion = std::shared_ptr<const std::vector<SharedBytes>>;
inline uint64_t version_bytes(const ResourceVersion& v) {
    uint64_t n = 0;
    for (const SharedBytes& b : *v) n += b->size();
    return n;
}

struct Session {
    std::string id;
    std::string proposed;            // the id the producer asked for
    wire::Identity identity;         // with the id the hub chose
    std::vector<std::string> capabilities;
    bool controls_cap = false, selection_cap = false, events_cap = false;   // the producer's
    bool programs_cap = false;       // 3.2 draft: the producer announced "programs"
    bool interpretation_cap = false;
    bool text_params_cap = false;
    bool interventions_cap = false;  // 3.2 draft: the producer announced "interventions"
    ConnId producer = 0;
    bool live = true;
    uint64_t order = 0;
    uint64_t ended_ms = 0;
    bool have_views = false;
    bool resubscribe = false;        // after a resume: subscribe again on the next VIEWS
    Json views = Json::array();
    std::map<std::string, ViewInfo> view_index;
    std::map<uint64_t, Channel> channels;
    std::map<uint32_t, uint64_t> by_up_sub, by_up_request;
    std::map<std::string, History> series;   // by view name
    std::set<ConnId> viewers;        // attached
    uint32_t next_sub = 1, next_request = 1;
    // 3.1. VIEWS without `items`, for viewers that did not announce `selection`;
    // the latest CONTROLS and the latest SELECTION of each item space, as the
    // producer sent them (kept after the session ends); the CONTROL requests
    // waiting for the producer's reply, by the hub's own request_id.
    Json views_plain = Json::array();
    SharedBytes controls;
    std::set<std::string> text_params;
    std::map<std::string, SelectionState> selections;
    std::map<uint32_t, ControlRoute> routes;
    uint32_t next_control = 1;
    // 3.2 draft. `controls_plain` is `controls` without the key "interventions",
    // for viewers that did not announce it; null when `controls` has no such key
    // (then `controls` goes to every viewer). `program_state` is the latest
    // unrequested PROGRAM_STATE, kept for a viewer with programs that attaches
    // later; `program_routes` are the PROGRAMs waiting for the producer's
    // PROGRAM_STATE, by the hub's own request_id.
    SharedBytes controls_plain;
    SharedBytes program_state;
    std::map<uint32_t, ProgramRoute> program_routes;
    uint32_t next_program = 1;
    // 3.2 draft, capability "resources": whether the producer announced it; the last draw half of each
    // program as the producer relayed it (a PROGRAM set of an instance), and the newest RESOURCE of
    // each "<program>/<resource>", both kept for a viewer with "resources" that attaches later.
    bool resources_cap = false;
    struct DrawHalf {
        std::string hash;            // the piece's hash, from the pin
        SharedBytes frame;
    };
    std::map<std::string, DrawHalf> draw_halves;
    std::map<std::string, ResourceVersion> kept_resources;
    // The producer's versions in parts being put together: the order checks without copies, and the
    // frames of each incomplete version, which the checks bound to two of at most
    // CALIPER_NET_RESOURCE_VERSION_MAX_BYTES.
    wire::SplitResources resource_parts{false};
    std::map<std::string, std::vector<SharedBytes>> resource_building;
    // 3.2 draft, capability "scene_pieces": the newest whole version of each scene program's resource,
    // "<program>/<resource>", with its count and its frames, until the update in the piece form that
    // names it comes on the same connection and takes it into a bundle; and its parts being put together.
    struct SceneVersion {
        uint32_t count = 0;
        std::vector<SharedBytes> frames;
    };
    std::map<std::string, SceneVersion> scene_versions;
    wire::SplitResources scene_parts{false};
    std::map<std::string, std::vector<SharedBytes>> scene_building;
};

// The most resources a session keeps, and a viewer has waiting: every resource of every program.
constexpr size_t kResourcesMax = size_t(CALIPER_NET_MAX_PROGRAMS) * CALIPER_NET_MAX_RESOURCES;

struct VSub {
    uint32_t sub = 0;
    std::string view;
    uint64_t channel = 0;
    uint64_t ticket = 0;
    bool series = false;
    double rate = kDefaultRate;
    double tokens = 0;               // the rate limit: a bucket of kBurst messages
    uint64_t refill_us = 0;          // that refills at `rate` per second
    uint64_t last_us = 0;
    bool sent_any = false;
    SnapPtr pending;                 // latest wins
    uint64_t pending_n = 0;          // counts the snapshots made pending (a held CONTROLS waits for one)
    uint64_t next = 0;               // series: next absolute index to send
    uint64_t backfill = std::numeric_limits<uint64_t>::max();
    // 3.2 draft, a scene: messages sent first and in order (clears and states),
    // each with whether it goes as state; then the newest batch of each entity
    // part, each part at most at `rate`.
    bool scene = false;
    bool scene_pieces = false;          // its viewer announced "scene_pieces": it takes the bundles of the piece form
    std::deque<std::pair<ScenePtr, bool>> scene_first;
    std::map<SceneKey, ScenePtr> scene_pending;
    std::map<SceneKey, uint64_t> scene_last_us;
    TransformRows transforms_pending;   // merged into SCENE_TRANSFORMS when sent, the newest per entity
};

struct Peer {
    enum class Role { None, Viewer, Producer, Retired };
    ConnId conn = 0;
    Role role = Role::None;
    uint64_t opened_ms = 0;
    uint64_t pinged_at = 0;          // last_input_ms when the last PING went out
    uint32_t nonce = 0;
    bool closing = false;
    bool watch_only = false;         // its token may watch, not control (HubOptions::stream_access)
    // viewer
    std::string attached;            // session id, "" when not attached
    bool follows_default = true;
    std::map<uint32_t, VSub> subs;
    std::map<uint32_t, uint64_t> pending_sub;   // sub -> ticket of its waiting SUBSCRIBE
    uint64_t tickets = 0;
    bool in_flight = false;          // a data message is in the send queue
    // One scene bundle continues after each frame drains. No other data
    // enters the queue before its final update. The subscription ticket
    // prevents a replacement subscription from receiving the old bundle.
    std::deque<ScenePtr> sending_scene;
    uint32_t sending_scene_sub = 0;
    uint64_t sending_scene_ticket = 0;
    bool sending_scene_state = false;
    bool dirty = false;
    uint64_t due_us = 0;             // a rate-limited message is due then
    // 3.1: what the viewer announced, its coalesced CONTROLS and SELECTIONs
    // (latest wins while its send queue is full), and its waiting CONTROLs.
    bool cap_controls = false, cap_selection = false;
    bool cap_scenes = false;         // 3.2 draft: the viewer announced "scenes"
    bool cap_programs = false;       // 3.2 draft: the viewer announced "programs"
    bool cap_text_params = false;
    bool cap_interpretation = false;
    bool cap_interventions = false;  // 3.2 draft: the viewer announced "interventions"
    bool cap_resources = false;      // 3.2 draft: the viewer announced "resources"
    bool cap_scene_pieces = false;   // 3.2 draft: the viewer announced "scene_pieces"
    // RESOURCEs waiting for the send queue: the newest of each "<program>/<resource>", with the order
    // in which it began to wait, so that a resource with frequent versions holds no other back.
    std::map<std::string, std::pair<uint64_t, ResourceVersion>> pending_resources;
    uint64_t resource_order = 0;
    // A version in parts that is being sent, its key and its next part. Its parts go in order, and at
    // most one waiting version in one frame goes between two of them, so the viewer gets the next part
    // well within CALIPER_NET_RESOURCE_PART_WAIT frames and a newer version of the same resource waits.
    ResourceVersion sending_resource;
    std::string sending_key;
    size_t sending_next = 0;
    bool sent_part = false;
    SharedBytes pending_controls;
    std::map<std::string, SharedBytes> pending_selections;
    size_t controls_waiting = 0;
    // A CONTROLS held for the data the hub had for this viewer when it came:
    // per subscription, the pending snapshot (its pending_n) and the series
    // index to reach. A newer CONTROLS replaces it and adds what it waits for.
    SharedBytes held_controls;
    uint64_t held_since_us = 0;
    std::map<uint32_t, uint64_t> held_snaps, held_series;
    // producer
    std::string session;
};

inline double clamp_rate(const std::optional<double>& r) {
    double x = r ? *r : kDefaultRate;
    if (!(x > 0) || !std::isfinite(x)) x = kDefaultRate;
    return std::min(x, kLimitRate);
}

inline uint64_t backfill_of(const Json& resolved) {
    const Json* b = resolved.find("budget");
    const Json* n = b && b->is_object() ? b->find("max_backfill") : nullptr;
    uint64_t v = 0;
    if (n && n->get_u64(&v)) return v;
    return std::numeric_limits<uint64_t>::max();
}

}  // namespace hub_detail

// ============================================================================
// the hub
// ============================================================================
struct Hub::Impl {
    explicit Impl(HubOptions o) : opt(std::move(o)), server(ws_options(opt)) {}

    static Options ws_options(const HubOptions& h);

    HubOptions opt;
    Server server;
    StaticFiles files;
    std::unique_ptr<HttpDoor> door;
    std::map<ConnId, hub_detail::Peer> peers;
    std::set<ConnId> watch_only;   // upgraded with a watching token, not yet open
    std::map<std::string, hub_detail::Session> sessions;
    uint64_t next_channel = 1;
    uint64_t next_order = 1;
    uint64_t snap_bytes = 0, series_bytes = 0;
    uint64_t meta_bytes = 0;         // 3.1: the CONTROLS and SELECTIONs kept per session
    uint64_t scene_bytes = 0;        // 3.2 draft: the scene batches kept for late viewers
    uint64_t resource_bytes = 0;     // 3.2 draft: the newest RESOURCEs kept for late viewers
    uint64_t retained() const { return snap_bytes + series_bytes + meta_bytes + scene_bytes + resource_bytes; }
    HubStats c;                      // the counters
    mutable std::mutex stats_mu;
    HubStats published;
    uint64_t last_publish_ms = 0, last_expire_ms = 0;

    // Responses to on_request's requests, queued by Hub::respond() on any
    // thread and written by the loop.
    struct Reply {
        ConnId conn = 0;
        int status = 0;
        std::string type, body;
        std::vector<std::pair<std::string, std::string>> headers;
    };
    std::mutex replies_mu;
    std::vector<Reply> replies;
    bool running = false;            // under replies_mu: run() takes replies

    void write_replies();
    void log(const std::string& line) {
        if (opt.log) opt.log(line);
    }

    // ---- sending (hub.cpp) ---------------------------------------------------------
    bool send_bytes(ConnId conn, std::vector<uint8_t>&& b);
    bool send(ConnId conn, const wire::Message& m);
    void overflow(ConnId conn);
    void close(ConnId conn, uint16_t code, const std::string& reason);
    void bye(ConnId conn, const char* reason, uint16_t code, const std::string& text);
    void error(ConnId conn, const std::string& code, const std::string& text,
               std::optional<uint32_t> in_reply_to = {}, bool fatal = false);
    void fatal(ConnId conn, uint16_t close_code, const std::string& code, const std::string& text);
    void reply_error(ConnId conn, uint32_t request_id, uint32_t sub, const std::string& view,
                     const std::string& code, const std::string& text);
    void reply_ok(ConnId conn, uint32_t request_id, uint32_t sub, const std::string& view, const Json& resolved);

    // ---- sessions directory & viewers (hub_viewer.cpp) -----------------------------
    std::vector<hub_detail::Session*> ordered_sessions();
    wire::Sessions sessions_msg();
    void broadcast_sessions();
    wire::Views views_msg(const hub_detail::Session* s, bool with_items, bool with_scenes);

    size_t live_count() const;
    void attach(hub_detail::Peer& p, const std::string& sid);
    void detach_all(hub_detail::Peer& p);
    void reconsider_defaults();

    hub_detail::Session* session_of(const hub_detail::Peer& p);
    hub_detail::Channel* channel(hub_detail::Session& s, uint64_t id);
    void mark_dirty(ConnId conn);
    hub_detail::VSub* vsub(const hub_detail::SubKey& k);
    void set_latest(hub_detail::Channel& ch, hub_detail::SnapPtr snap);
    double want_rate(const hub_detail::Channel& ch);
    void send_subscribe(hub_detail::Session& s, hub_detail::Channel& ch);
    bool subscribe_upstream(hub_detail::Session& s, hub_detail::Channel& ch);
    void unsubscribe_upstream(hub_detail::Session& s, hub_detail::Channel& ch);
    void adjust_rate(hub_detail::Session& s, hub_detail::Channel& ch);
    void settle(hub_detail::Session& s, uint64_t id);
    void trim_idle(hub_detail::Session& s);
    void destroy_channel(hub_detail::Session& s, uint64_t id, const char* code = CALIPER_NET_ERR_UNKNOWN_VIEW,
                         const std::string& text = "the view is gone");
    bool waiter_valid(hub_detail::Session& s, const hub_detail::Waiter& w, hub_detail::Peer** out);
    void fail_waiter(hub_detail::Session& s, const std::string& view, const hub_detail::Waiter& w,
                     const std::string& code, const std::string& text);
    void join(hub_detail::Session& s, hub_detail::Peer& p, hub_detail::VSub& v, hub_detail::Channel& ch);
    void leave(hub_detail::Session& s, hub_detail::Peer& p, hub_detail::VSub& v);
    void end_vsub(hub_detail::Peer& p, uint32_t sub);
    void serve_waiter(hub_detail::Session& s, hub_detail::Channel& ch, const hub_detail::Waiter& w);
    hub_detail::Channel* ended_lookup(hub_detail::Session& s, const std::string& view, const std::string& key,
                                      uint64_t exclude = 0);
    void request(hub_detail::Session& s, const std::string& view, const Json& requested,
                 const hub_detail::Waiter& w);

    static constexpr size_t kMaxControlsWaiting = 64;       // per viewer connection
    static constexpr size_t kMaxRoutes = 1024;              // per session
    static constexpr uint64_t kReplyWaitMs = 60000;         // a reply the producer owes
    static constexpr uint64_t kQueuedWaitMs = CALIPER_NET_ACTION_WAIT_MS + kReplyWaitMs;

    void viewer_subscribe(hub_detail::Peer& p, const wire::Subscribe& m);
    void viewer_view_request(hub_detail::Peer& p, const wire::ViewRequest& m);
    void viewer_event(hub_detail::Peer& p, const wire::DecodeResult& r, const wire::Event& e);
    void viewer_control(hub_detail::Peer& p, const wire::Control& ctl);
    void viewer_program(hub_detail::Peer& p, const wire::Program& m);
    void answer_program_route(const hub_detail::ProgramRoute& r, const char* code, const std::string& text);
    void answer_route(const hub_detail::ControlRoute& r, const char* code, const std::string& text);
    void control_error(hub_detail::Peer& p, uint32_t request_id, const std::string& op,
                       const std::optional<std::string>& name, const char* code, const std::string& text);
    void on_viewer_message(hub_detail::Peer& p, const wire::DecodeResult& r, wire::Message& m);

    // ---- flow control & delivery (hub_flow.cpp) ------------------------------------
    static constexpr size_t kMetaQueueLimit = size_t(256) << 10;
    // How long a CONTROLS waits for the data before it, at most: the senders'
    // CALIPER_NET_ORDER_WAIT_MS.
#ifdef CALIPER_NET_ORDER_WAIT_MS
    static constexpr uint64_t kOrderWaitUs = uint64_t(CALIPER_NET_ORDER_WAIT_MS) * 1000;
#else
    static constexpr uint64_t kOrderWaitUs = 2000000;
#endif

    void offer_controls(hub_detail::Peer& p, const SharedBytes& frame);
    void offer_selection(hub_detail::Peer& p, const std::string& space, const SharedBytes& frame);
    void flush_meta(hub_detail::Peer& p);
    void deliver_controls(hub_detail::Peer& p, const SharedBytes& frame);
    void release_controls(hub_detail::Peer& p, uint64_t now);
    void drop_held_controls(hub_detail::Peer& p);
    static SharedBytes controls_for(const hub_detail::Peer& p, const hub_detail::Session& s);
    void send_state(hub_detail::Peer& p, const hub_detail::Session& s);
    void offer_resource(hub_detail::Peer& p, const std::string& key, const hub_detail::ResourceVersion& buf);
    static void stop_sending_resource(hub_detail::Peer& p);
    void pump(hub_detail::Peer& p, uint64_t now);
    void pump_data(hub_detail::Peer& p, uint64_t now);
    bool send_series(hub_detail::Peer& p, hub_detail::Session& s, hub_detail::VSub& v);
    bool send_snapshot(hub_detail::Peer& p, hub_detail::VSub& v);

    // ---- producers & retention (hub_producer.cpp) ----------------------------------
    static bool valid_id_char(char ch);
    void on_producer_hello(hub_detail::Peer& p, const wire::Hello& h);
    static void set_producer_caps(hub_detail::Session& s, const std::vector<std::string>& caps);
    void welcome_producer(hub_detail::Peer& p, hub_detail::Session& s);
    void drop_upstream(hub_detail::Session& s);
    void end_session(hub_detail::Session& s);
    size_t ended_count() const;
    void evict_oldest_ended();
    void remove_session(const std::string& id, bool reattach);
    static Json strip_items(const Json& views);
    void forget_meta(hub_detail::Session& s);
    void forget_program(hub_detail::Session& s, const std::string& id);
    void forget_programs(hub_detail::Session& s);
    void producer_program(hub_detail::Session& s, const wire::Program& m, SharedBytes buf);
    void producer_resource(hub_detail::Session& s, const wire::Resource& m, SharedBytes buf);
    void producer_control_reply(hub_detail::Session& s, const wire::ControlReply& m);
    void producer_controls(hub_detail::Session& s, const wire::Controls& m, SharedBytes buf);
    void producer_program_state(hub_detail::Session& s, const wire::ProgramState& m, SharedBytes buf);
    void producer_selection(hub_detail::Session& s, const wire::Selection& m, SharedBytes buf);
    void producer_views(hub_detail::Session& s, const wire::Views& m);
    void producer_reply(hub_detail::Session& s, const wire::ViewReply& m);
    void producer_snapshot(hub_detail::Session& s, wire::DecodeResult& r, const wire::Snapshot& m, SharedBytes buf);
    void producer_series(hub_detail::Session& s, const wire::Series& m);
    void enforce_budget();
    void on_producer_message(hub_detail::Peer& p, wire::DecodeResult& r, wire::Message& m, SharedBytes buf);

    // ---- 3.2 draft: scenes (hub_scene.cpp) -----------------------------------------
    void scene_resource(hub_detail::Session& s, const wire::Resource& m, SharedBytes buf);
    std::shared_ptr<hub_detail::SceneMsg> scene_bundle(hub_detail::Session& s, const wire::SceneUpdate& up,
                                                       const hub_detail::ScenePtr& update);
    void producer_scene(hub_detail::Session& s, wire::DecodeResult& r, const wire::Message& m, SharedBytes buf);
    std::shared_ptr<hub_detail::SceneMsg> scene_piece(hub_detail::Channel& ch, const wire::SceneUpdate& up,
                                                      const hub_detail::ScenePtr& piece);
    void forget_scene(hub_detail::Channel& ch);
    void release_scene(hub_detail::Channel& ch);
    void join_scene(hub_detail::VSub& v, const hub_detail::Channel& ch);
    void resync_scene(hub_detail::Session& s, hub_detail::Channel& ch, hub_detail::VSub& v);
    bool send_scene_frame(hub_detail::Peer& p, hub_detail::VSub& v, const hub_detail::ScenePtr& msg, bool state);
    bool send_scene_part(hub_detail::Peer& p, hub_detail::VSub& v);
    bool send_scene(hub_detail::Peer& p, hub_detail::VSub& v, const hub_detail::ScenePtr& msg, bool state);

    // ---- connections, timers & loop (hub.cpp) --------------------------------------
    static bool has_cap(const std::vector<std::string>& caps, const char* cap);
    wire::Welcome hub_welcome(bool client_has_controls);
    void on_hello(hub_detail::Peer& p, const wire::Hello& h);
    void on_message(ConnId conn, Bytes&& data, bool binary);
    void on_closed(ConnId conn);
    void handle(Event& e);
    void tick();
    int timeout_ms(uint64_t now);
    void publish(bool force);
    void run(const std::atomic<bool>& stop);
};

}  // namespace caliper_net
