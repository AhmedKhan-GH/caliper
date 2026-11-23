// hub.cpp: the wire protocol v3 hub of hub.h.
//
// The state, all owned by the thread in run():
//   Peer      one upgraded connection: a viewer, a producer, or a producer
//             connection that a resumed session replaced (retired).
//   Session   one producer process: its identity, its VIEWS, its channels and
//             its retained series. Kept after the producer leaves (ended).
//   Channel   one subscription of the hub at a producer (`up_sub`): a view
//             under one resolved data view. Viewer subscriptions whose
//             resolved data views are equal share it. It keeps the latest
//             snapshot, and stays (idle, with that snapshot) after its last
//             viewer leaves, so a late viewer is served at once.
//   History   the retained values of one series view, by absolute index.
//   VSub      one subscription of a viewer: its own `sub`, its channel, its
//             rate, its pending snapshot (latest wins) or its series cursor.
//   Waiter    a viewer's SUBSCRIBE or VIEW_REQUEST waiting for the producer's
//             VIEW_REPLY on a channel.
#include "net/hub.h"

#include "caliper/live_net.h"
#include "net/http_door.h"
#include "net/json.h"
#include "net/wire.h"
#include "net/ws.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <deque>
#include <filesystem>
#include <fstream>
#include <limits>
#include <map>
#include <mutex>
#include <optional>
#include <random>
#include <set>
#include <sstream>
#include <system_error>
#include <vector>

#ifdef _WIN32
#  include <process.h>
#else
#  include <fcntl.h>
#  include <sys/stat.h>
#  include <unistd.h>
#endif
#ifdef __APPLE__
#  include <stdlib.h>   // arc4random_buf
#endif

namespace caliper_net {

namespace {

namespace fs = std::filesystem;

uint64_t now_us() {
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

std::string canon(const Json& j) { return j.is_object() ? j.dump() : std::string("{}"); }

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
// The messages a viewer's scene subscription may have waiting in order (a state, a reset and at
// most CALIPER_NET_MAX_BATCHES batches, and as many clears); past it the hub's own state replaces them.
constexpr size_t kSceneFirstMax = 2 * CALIPER_NET_MAX_BATCHES;
// A viewer's pending batches (one per entity part) and pending entity transforms (one per entity)
// wait while its send is in flight and count in no budget, so each holds at most this many; past
// it the hub's own state replaces them, as past kSceneFirstMax.
constexpr size_t kScenePendingMax = CALIPER_NET_MAX_BATCHES;
// (path, part): the byte order of the paths, then the parts in the order points, lines, arrows,
// boxes, mesh, image, camera. The class table is kept under ("", kClassesOrder) and the entity
// transforms under ("", kTransformsOrder), so that a late viewer's state sends them first, in the
// draft's order: the class table, the entity transforms, then the batches.
using SceneKey = std::pair<std::string, int>;
constexpr int kClassesOrder = -2, kTransformsOrder = -1, kCameraOrder = 6;
int scene_part_order(const std::string& part) {
    const int i = wire::scene_part_index(part);
    return i < 0 ? 7 : i;
}
bool scene_reaches(const SceneMsg& c, const std::string& p) {
    return p == c.path || (c.recursive && (c.path.empty() || (p.size() > c.path.size() &&
                                                              p.compare(0, c.path.size(), c.path) == 0 &&
                                                              p[c.path.size()] == '/')));
}
// Whether clear `c` removes the batch of `part` at `path`: a reset removes
// everything; otherwise static items stay unless the clear is static too.
// Only a reset removes the class table, which belongs to no entity.
bool scene_clears(const SceneMsg& c, const SceneMsg& b) {
    if (c.reset) return true;
    if (b.type == CALIPER_NET_MSG_SCENE_CLASSES) return false;
    if (b.is_static && !c.is_static) return false;
    return scene_reaches(c, b.path) &&
           (c.parts.empty() || std::find(c.parts.begin(), c.parts.end(), b.part) != c.parts.end());
}
// Whether clear `c` removes the entity transform of `path`.
bool scene_clears_transform(const SceneMsg& c, const std::string& path, const TransformRow& t) {
    if (c.reset) return true;
    if (t.is_static && !c.is_static) return false;
    return scene_reaches(c, path) && (c.parts.empty() || std::find(c.parts.begin(), c.parts.end(),
                                                                    CALIPER_NET_PART_TRANSFORM) != c.parts.end());
}
// SCENE_TRANSFORMS frames of transform rows: the static ones first, then the others, grouped by the
// length of their stamps, at most CALIPER_NET_MAX_TRANSFORMS in a message.
std::vector<ScenePtr> transform_messages(const std::string& view, const TransformRows& rows) {
    std::vector<ScenePtr> out;
    std::map<std::pair<int, size_t>, std::vector<const std::pair<const std::string, TransformRow>*>> groups;
    for (const auto& kv : rows) groups[{kv.second.is_static ? 0 : 1, kv.second.stamp.size()}].push_back(&kv);
    for (const auto& g : groups) {
        const auto& items = g.second;
        for (size_t at = 0; at < items.size(); at += CALIPER_NET_MAX_TRANSFORMS) {
            const size_t k = std::min<size_t>(CALIPER_NET_MAX_TRANSFORMS, items.size() - at);
            wire::SceneTransforms t;
            t.view = view;
            t.is_static = g.first.first == 0;
            std::vector<float> mats(k * 16);
            std::vector<int64_t> stamps(k * g.first.second);
            for (size_t i = 0; i < k; ++i) {
                const auto& kv = *items[at + i];
                t.paths.push_back(kv.first);
                t.seq = std::max(t.seq, kv.second.seq);
                t.stamp_ns = std::max(t.stamp_ns, kv.second.stamp_ns);
                std::memcpy(mats.data() + 16 * i, kv.second.m, 64);
                std::copy(kv.second.stamp.begin(), kv.second.stamp.end(), stamps.begin() + (ptrdiff_t)(i * g.first.second));
            }
            t.matrices = wire::Section::make("matrices", CALIPER_NET_DTYPE_F32, {(int64_t)k, 16}, mats.data(), mats.size() * 4);
            if (g.first.second)
                t.stamps = wire::Section::make("stamps", CALIPER_NET_DTYPE_I64, {(int64_t)k, (int64_t)g.first.second},
                                               stamps.data(), stamps.size() * 8);
            std::vector<uint8_t> bytes;
            if (!wire::encode(wire::Message(t), &bytes)) continue;
            auto msg = std::make_shared<SceneMsg>();
            wire::Frame f;
            wire::FrameError fe;
            if (!wire::decode_frame(bytes.data(), bytes.size(), &f, &fe)) continue;
            msg->type = CALIPER_NET_MSG_SCENE_TRANSFORMS;
            msg->meta = std::move(f.meta);
            msg->flags = f.flags;
            for (const auto& sec : f.sections) msg->lens.push_back((uint32_t)sec.size);
            msg->tail = f.sections.front().offset;
            msg->is_static = t.is_static;
            msg->frame = std::make_shared<const Bytes>(std::move(bytes));
            out.push_back(std::move(msg));
        }
    }
    return out;
}

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
    bool dirty = false;
    uint64_t due_us = 0;             // a rate-limited message is due then
    // 3.1: what the viewer announced, its coalesced CONTROLS and SELECTIONs
    // (latest wins while its send queue is full), and its waiting CONTROLs.
    bool cap_controls = false, cap_selection = false;
    bool cap_scenes = false;         // 3.2 draft: the viewer announced "scenes"
    bool cap_programs = false;       // 3.2 draft: the viewer announced "programs"
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

double clamp_rate(const std::optional<double>& r) {
    double x = r ? *r : kDefaultRate;
    if (!(x > 0) || !std::isfinite(x)) x = kDefaultRate;
    return std::min(x, kLimitRate);
}

uint64_t backfill_of(const Json& resolved) {
    const Json* b = resolved.find("budget");
    const Json* n = b && b->is_object() ? b->find("max_backfill") : nullptr;
    uint64_t v = 0;
    if (n && n->get_u64(&v)) return v;
    return std::numeric_limits<uint64_t>::max();
}

std::string base64url(const uint8_t* p, size_t n) {
    static const char* k = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_";
    std::string out;
    uint32_t acc = 0;
    int bits = 0;
    for (size_t i = 0; i < n; ++i) {
        acc = (acc << 8) | p[i];
        bits += 8;
        while (bits >= 6) {
            bits -= 6;
            out += k[(acc >> bits) & 63];
        }
    }
    if (bits > 0) out += k[(acc << (6 - bits)) & 63];
    return out;
}

void secure_random(uint8_t* p, size_t n) {
#if defined(__APPLE__)
    arc4random_buf(p, n);
#elif defined(_WIN32)
    std::random_device rd;   // rand_s, the system's cryptographic generator, on MSVC
    for (size_t i = 0; i < n; ++i) p[i] = (uint8_t)rd();
#else
    std::ifstream f("/dev/urandom", std::ios::binary);
    f.read((char*)p, (std::streamsize)n);
    if (!f) {
        std::random_device rd;
        for (size_t i = 0; i < n; ++i) p[i] = (uint8_t)rd();
    }
#endif
}

long this_pid() {
#ifdef _WIN32
    return (long)_getpid();
#else
    return (long)getpid();
#endif
}

}  // namespace

// ============================================================================
// the hub
// ============================================================================
struct Hub::Impl {
    explicit Impl(HubOptions o) : opt(std::move(o)), server(ws_options(opt)) {}

    static Options ws_options(const HubOptions& h) {
        Options o;
        o.max_queued_bytes = h.max_queued_bytes;
        o.max_connections = h.max_tcp_connections;
        o.header_timeout_ms = h.handshake_timeout_ms;
        o.body_limit = h.body_limit;
        o.body_timeout_ms = h.handshake_timeout_ms;
        return o;
    }

    HubOptions opt;
    Server server;
    StaticFiles files;
    std::unique_ptr<HttpDoor> door;
    std::map<ConnId, Peer> peers;
    std::set<ConnId> watch_only;   // upgraded with a watching token, not yet open
    std::map<std::string, Session> sessions;
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

    void write_replies() {
        std::vector<Reply> r;
        {
            std::lock_guard<std::mutex> g(replies_mu);
            r.swap(replies);
        }
        for (Reply& x : r) server.respond(x.conn, x.status, x.type, x.body, x.headers);
    }

    void log(const std::string& line) {
        if (opt.log) opt.log(line);
    }

    // ---- sending ------------------------------------------------------------------
    bool send_bytes(ConnId conn, std::vector<uint8_t>&& b) {
        auto shared = std::make_shared<const Bytes>(std::move(b));
        if (server.send(conn, shared)) return true;
        overflow(conn);
        return false;
    }

    bool send(ConnId conn, const wire::Message& m) {
        std::vector<uint8_t> b;
        if (!wire::encode(m, &b)) return false;
        return send_bytes(conn, std::move(b));
    }

    // A send queue past max_queued_bytes: the peer reads far too slowly.
    void overflow(ConnId conn) {
        auto it = peers.find(conn);
        if (it == peers.end() || it->second.closing) return;
        close(conn, kCloseOverflow, "send queue full");
    }

    void close(ConnId conn, uint16_t code, const std::string& reason) {
        auto it = peers.find(conn);
        if (it != peers.end()) it->second.closing = true;
        server.close(conn, code, reason);
    }

    void bye(ConnId conn, const char* reason, uint16_t code, const std::string& text) {
        wire::Bye b;
        b.reason = reason;
        b.message = text;
        send(conn, b);
        close(conn, code, reason);
    }

    void error(ConnId conn, const std::string& code, const std::string& text, std::optional<uint32_t> in_reply_to = {},
               bool fatal = false) {
        wire::Error e;
        e.code = code;
        e.message = text;
        e.fatal = fatal;
        e.in_reply_to = in_reply_to;
        send(conn, e);
    }

    void fatal(ConnId conn, uint16_t close_code, const std::string& code, const std::string& text) {
        error(conn, code, text, {}, true);
        close(conn, close_code, text.substr(0, 120));
    }

    void reply_error(ConnId conn, uint32_t request_id, uint32_t sub, const std::string& view,
                     const std::string& code, const std::string& text) {
        wire::ViewReply r;
        r.request_id = request_id;
        r.sub = sub;
        r.view = view;
        Json e = Json::object();
        e.set("code", code);
        e.set("message", text);
        r.error = e;
        send(conn, r);
    }

    void reply_ok(ConnId conn, uint32_t request_id, uint32_t sub, const std::string& view, const Json& resolved) {
        wire::ViewReply r;
        r.request_id = request_id;
        r.sub = sub;
        r.view = view;
        r.data_view = resolved.is_object() ? resolved : Json::object();
        send(conn, r);
    }

    // ---- sessions list --------------------------------------------------------------
    std::vector<Session*> ordered_sessions() {
        std::vector<Session*> v;
        for (auto& kv : sessions) v.push_back(&kv.second);
        std::sort(v.begin(), v.end(), [](Session* a, Session* b) { return a->order < b->order; });
        return v;
    }

    wire::Sessions sessions_msg() {
        wire::Sessions m;
        Json list = Json::array();
        for (Session* s : ordered_sessions()) {
            Json e = s->identity.to_json();
            e.set("state", s->live ? CALIPER_NET_SESSION_LIVE : CALIPER_NET_SESSION_ENDED);
            e.set("views", (uint64_t)s->views.size());
            Json caps = Json::array();
            for (const auto& x : s->capabilities) caps.push(x);
            e.set("capabilities", caps);
            list.push(e);
        }
        m.sessions = list;
        return m;
    }

    void broadcast_sessions() {
        std::vector<uint8_t> b;
        if (!wire::encode(sessions_msg(), &b)) return;
        auto shared = std::make_shared<const Bytes>(std::move(b));
        for (auto& kv : peers)
            if (kv.second.role == Peer::Role::Viewer && !kv.second.closing && !server.send(kv.first, shared))
                overflow(kv.first);
    }

    // VIEWS for one viewer: the item spaces of 3.1 only to a viewer that
    // announced `selection`.
    // 3.2 draft: the scenes only to a viewer that announced `scenes`.
    wire::Views views_msg(const Session* s, bool with_items, bool with_scenes) {
        wire::Views v;
        if (s) {
            v.session = s->id;
            v.views = with_items ? s->views : s->views_plain;
            if (!with_scenes) {
                Json out = Json::array();
                for (const Json& e : v.views.items()) {
                    const Json* k = e.find("kind");
                    int64_t kind = 0;
                    if (!(k && k->get_i64(&kind) && kind == CALIPER_NET_KIND_SCENE)) out.push(e);
                }
                v.views = out;
            }
        } else {
            v.views = Json::array();
        }
        return v;
    }

    // ---- 3.1: CONTROLS and SELECTION to viewers ----------------------------------------
    // Control messages, sent at once in order; when a viewer's send queue is
    // full they wait, and a newer CONTROLS (or SELECTION of the same item
    // space) replaces the waiting one, as the protocol coalesces them.
    static constexpr size_t kMetaQueueLimit = size_t(256) << 10;
    // How long a CONTROLS waits for the data before it, at most: the senders'
    // CALIPER_NET_ORDER_WAIT_MS.
#ifdef CALIPER_NET_ORDER_WAIT_MS
    static constexpr uint64_t kOrderWaitUs = uint64_t(CALIPER_NET_ORDER_WAIT_MS) * 1000;
#else
    static constexpr uint64_t kOrderWaitUs = 2000000;
#endif

    void offer_controls(Peer& p, const SharedBytes& frame) {
        if (!p.cap_controls || !frame) return;
        if (!p.pending_controls && server.queued_bytes(p.conn) < kMetaQueueLimit) {
            if (!server.send(p.conn, frame)) overflow(p.conn);
            ++c.controls_out;
        } else {
            p.pending_controls = frame;
        }
    }

    void offer_selection(Peer& p, const std::string& space, const SharedBytes& frame) {
        if (!p.cap_selection || !frame) return;
        if (!p.pending_selections.count(space) && server.queued_bytes(p.conn) < kMetaQueueLimit) {
            if (!server.send(p.conn, frame)) overflow(p.conn);
            ++c.selections_out;
        } else {
            p.pending_selections[space] = frame;
        }
    }

    // In pump(), before data: what waited, once the queue has room.
    void flush_meta(Peer& p) {
        if (server.queued_bytes(p.conn) >= kMetaQueueLimit) return;
        if (p.pending_controls) {
            if (!server.send(p.conn, p.pending_controls)) overflow(p.conn);
            p.pending_controls.reset();
            ++c.controls_out;
        }
        for (auto& kv : p.pending_selections) {
            if (!server.send(p.conn, kv.second)) overflow(p.conn);
            ++c.selections_out;
        }
        p.pending_selections.clear();
    }

    // A hub delivers a run state after the data it received before it, or
    // after 2 s, so that a paused view shows the step the run stopped at. A
    // CONTROLS for a viewer waits until each of its subscriptions has sent, or
    // replaced with a newer one, the snapshot that was pending when the
    // CONTROLS came, and has sent its series up to where they then ended. It
    // never waits for the viewer to drain: while data streams, an echo waits
    // for at most one message per subscription. kOrderWaitUs after the first
    // CONTROLS held for a viewer it goes anyway, with any newer one in its place.
    void deliver_controls(Peer& p, const SharedBytes& frame) {
        if (!p.cap_controls || !frame) return;
        if (Session* s = session_of(p))
            for (const auto& kv : p.subs) {
                const VSub& v = kv.second;
                if (v.series) {
                    auto h = s->series.find(v.view);
                    if (h != s->series.end() && std::max(v.next, h->second.first) < h->second.end()) {
                        uint64_t& need = p.held_series[kv.first];
                        need = std::max(need, h->second.end());
                    }
                } else if (v.pending) {
                    p.held_snaps[kv.first] = v.pending_n;
                }
            }
        if (!p.held_controls && p.held_snaps.empty() && p.held_series.empty()) {
            offer_controls(p, frame);
            return;
        }
        if (!p.held_controls) {
            p.held_since_us = now_us();
            ++c.controls_held;
        }
        p.held_controls = frame;
        p.dirty = true;
    }

    // In pump(): sends a held CONTROLS once what it waits for has gone out or
    // been replaced, or its time is up.
    void release_controls(Peer& p, uint64_t now) {
        if (!p.held_controls) return;
        Session* s = session_of(p);
        for (auto it = p.held_snaps.begin(); it != p.held_snaps.end();) {
            auto v = p.subs.find(it->first);
            const bool waits = v != p.subs.end() && v->second.pending && v->second.pending_n == it->second;
            it = waits ? std::next(it) : p.held_snaps.erase(it);
        }
        for (auto it = p.held_series.begin(); it != p.held_series.end();) {
            auto v = p.subs.find(it->first);
            bool waits = false;
            if (s && v != p.subs.end() && v->second.series) {
                auto h = s->series.find(v->second.view);
                if (h != s->series.end())
                    waits = std::max(v->second.next, h->second.first) < std::min(it->second, h->second.end());
            }
            it = waits ? std::next(it) : p.held_series.erase(it);
        }
        const bool waits = !p.held_snaps.empty() || !p.held_series.empty();
        if (waits && now < p.held_since_us + kOrderWaitUs) return;
        if (waits) ++c.controls_hold_expired;
        SharedBytes frame = std::move(p.held_controls);
        drop_held_controls(p);
        offer_controls(p, frame);
    }

    void drop_held_controls(Peer& p) {
        p.held_controls.reset();
        p.held_snaps.clear();
        p.held_series.clear();
        p.held_since_us = 0;
    }

    // The session's latest CONTROLS as one viewer may see it: without the key
    // "interventions" unless the viewer announced that capability (3.2 draft).
    static const SharedBytes& controls_for(const Peer& p, const Session& s) {
        return (!p.cap_interventions && s.controls_plain) ? s.controls_plain : s.controls;
    }

    // Right after the VIEWS of an attachment: the session's latest CONTROLS and
    // every item space's selection that is not empty. A new attachment has no
    // data pending, so its CONTROLS goes at once, before any SUBSCRIBE.
    void send_state(Peer& p, const Session& s) {
        if (p.cap_controls && s.controls) deliver_controls(p, controls_for(p, s));
        if (p.cap_programs && s.program_state) {
            if (!server.send(p.conn, s.program_state)) overflow(p.conn);
            ++c.program_states_out;
        }
        if (p.cap_selection)
            for (const auto& kv : s.selections)
                if (!kv.second.empty) offer_selection(p, kv.first, kv.second.frame);
        if (p.cap_resources) {   // 3.2 draft: the draw halves first, then their resources as data
            for (const auto& kv : s.draw_halves) {
                if (!server.send(p.conn, kv.second.frame)) {
                    overflow(p.conn);
                    return;
                }
                ++c.program_relays_out;
            }
            for (const auto& kv : s.kept_resources) offer_resource(p, kv.first, kv.second);
        }
    }

    // ---- 3.2 draft: resources (capability "resources") ----------------------------------------
    // One RESOURCE for a viewer: it replaces a waiting version of the same resource and keeps that
    // version's place; a new resource past kResourcesMax waiting ones is dropped.
    void offer_resource(Peer& p, const std::string& key, const ResourceVersion& buf) {
        auto it = p.pending_resources.find(key);
        if (it != p.pending_resources.end()) {
            ++c.resources_superseded;
            it->second.second = buf;
        } else if (p.pending_resources.size() >= kResourcesMax) {
            ++c.resources_dropped;
            return;
        } else {
            p.pending_resources[key] = {++p.resource_order, buf};
        }
        p.dirty = true;
    }

    // Forgets what the session keeps of one program: its draw half and its resources' newest
    // versions; and drops the versions of them that wait at each viewer, which belong to the
    // program's earlier install.
    void forget_program(Session& s, const std::string& id) {
        if (auto it = s.draw_halves.find(id); it != s.draw_halves.end()) {
            meta_bytes -= it->second.frame->size();
            s.draw_halves.erase(it);
        }
        const std::string prefix = id + "/";
        for (auto it = s.kept_resources.lower_bound(prefix);
             it != s.kept_resources.end() && it->first.compare(0, prefix.size(), prefix) == 0;) {
            resource_bytes -= version_bytes(it->second);
            it = s.kept_resources.erase(it);
        }
        for (ConnId v : s.viewers) {
            auto pt = peers.find(v);
            if (pt == peers.end()) continue;
            auto& pending = pt->second.pending_resources;
            for (auto it = pending.lower_bound(prefix);
                 it != pending.end() && it->first.compare(0, prefix.size(), prefix) == 0;)
                it = pending.erase(it);
            if (pt->second.sending_key.compare(0, prefix.size(), prefix) == 0) stop_sending_resource(pt->second);
        }
    }

    static void stop_sending_resource(Peer& p) {
        p.sending_resource.reset();
        p.sending_key.clear();
        p.sending_next = 0;
        p.sent_part = false;
    }

    void forget_programs(Session& s) {
        std::vector<std::string> ids;
        for (const auto& kv : s.draw_halves) ids.push_back(kv.first);
        for (const auto& kv : s.kept_resources) ids.push_back(kv.first.substr(0, kv.first.find('/')));
        for (const std::string& id : ids) forget_program(s, id);
    }

    // The producer's relay of a draw half (a PROGRAM set of an instance of a piece) or of a removal.
    // The hub keeps the last draw half of each program, at most CALIPER_NET_MAX_PROGRAMS, and sends
    // the relay to every attached viewer that announced "resources". A removal, or the draw half of
    // another piece, also drops the program's kept resources; the same piece relayed again (after an
    // attachment) keeps them, so that a static resource, which the script publishes once, stays. A
    // producer without the capability, and any other PROGRAM from a producer, are ignored.
    void producer_program(Session& s, const wire::Program& m, SharedBytes buf) {
        const bool set = m.op == CALIPER_NET_PROGRAM_SET && !m.piece.is_null();
        if (!s.resources_cap || !(set || m.op == CALIPER_NET_PROGRAM_REMOVE)) return;
        ++c.program_relays_in;
        const Json* h = set ? m.piece.find("hash") : nullptr;
        const std::string hash = h && h->is_string() ? h->as_string() : std::string();
        auto old = s.draw_halves.find(m.id);
        if (!set || (old != s.draw_halves.end() && old->second.hash != hash)) {
            forget_program(s, m.id);
        } else if (old != s.draw_halves.end()) {
            meta_bytes -= old->second.frame->size();
            s.draw_halves.erase(old);
        }
        if (set && s.draw_halves.size() < CALIPER_NET_MAX_PROGRAMS) {
            s.draw_halves[m.id] = Session::DrawHalf{hash, buf};
            meta_bytes += buf->size();
        }
        for (ConnId v : s.viewers) {
            auto pt = peers.find(v);
            if (pt == peers.end() || pt->second.closing || !pt->second.cap_resources) continue;
            if (!server.send(pt->second.conn, buf)) {
                overflow(pt->second.conn);
                continue;
            }
            ++c.program_relays_out;
        }
        if (retained() > opt.max_retained_bytes) enforce_budget();
    }

    // The producer's RESOURCE: offered to every attached viewer that announced "resources", with one
    // waiting version per resource and viewer, and kept as the newest version for a later viewer
    // while it fits the budget of retained bytes. A version over the budget is forwarded, not kept.
    // The parts of a version in parts wait until the version is whole, which the order checks of
    // wire::SplitResources decide; a part they drop or refuse is counted and goes nowhere.
    void producer_resource(Session& s, const wire::Resource& m, SharedBytes buf) {
        if (wire::scene_program_id(m.program)) return scene_resource(s, m, std::move(buf));
        if (!s.resources_cap) return;
        const std::string key = m.program + "/" + m.resource;
        std::string err;
        const uint64_t let_go = s.resource_parts.discarded() + s.resource_parts.expired();
        const wire::SplitResources::Result got = s.resource_parts.add(m, nullptr, &err);
        std::vector<SharedBytes> frames;
        if (!m.split) {
            frames.push_back(std::move(buf));
        } else if (got == wire::SplitResources::Result::Pending) {
            if (m.split->part == 0) s.resource_building[key].clear();
            s.resource_building[key].push_back(std::move(buf));
        } else if (got == wire::SplitResources::Result::Whole) {
            frames = std::move(s.resource_building[key]);
            frames.push_back(std::move(buf));
            s.resource_building.erase(key);
        } else {
            ++c.resource_parts_dropped;
        }
        // The frames of the versions that the checks no longer hold: refused, replaced, or expired.
        for (auto it = s.resource_building.begin(); it != s.resource_building.end();) {
            const size_t slash = it->first.find('/');
            if (!s.resource_parts.holds(it->first.substr(0, slash), it->first.substr(slash + 1)))
                it = s.resource_building.erase(it);
            else
                ++it;
        }
        c.resource_versions_incomplete += s.resource_parts.discarded() + s.resource_parts.expired() - let_go -
                                          (got == wire::SplitResources::Result::Refused ? 1 : 0);
        if (frames.empty()) return;
        ++c.resources_in;
        auto version = std::make_shared<const std::vector<SharedBytes>>(std::move(frames));
        const uint64_t bytes = version_bytes(version);
        if (auto it = s.kept_resources.find(key); it != s.kept_resources.end()) {
            resource_bytes -= version_bytes(it->second);
            s.kept_resources.erase(it);
        }
        if (s.kept_resources.size() < kResourcesMax && retained() + bytes <= opt.max_retained_bytes) {
            s.kept_resources[key] = version;
            resource_bytes += bytes;
        }
        for (ConnId v : s.viewers) {
            auto pt = peers.find(v);
            if (pt == peers.end() || pt->second.closing || !pt->second.cap_resources) continue;
            offer_resource(pt->second, key, version);
        }
    }

    // A scene program's RESOURCE (the piece form of SCENE_UPDATE): its whole version waits for the update
    // that names it, in the place of an older one, and never takes the program route. At most every batch
    // of a scene times the resources of a piece wait, so that updates that never come cannot grow it.
    void scene_resource(Session& s, const wire::Resource& m, SharedBytes buf) {
        const std::string key = m.program + "/" + m.resource;
        std::string err;
        const wire::SplitResources::Result got = s.scene_parts.add(m, nullptr, &err);
        std::vector<SharedBytes> frames;
        if (!m.split) {
            frames.push_back(std::move(buf));
        } else if (got == wire::SplitResources::Result::Pending) {
            if (m.split->part == 0) s.scene_building[key].clear();
            s.scene_building[key].push_back(std::move(buf));
        } else if (got == wire::SplitResources::Result::Whole) {
            frames = std::move(s.scene_building[key]);
            frames.push_back(std::move(buf));
            s.scene_building.erase(key);
        } else {
            ++c.resource_parts_dropped;
        }
        for (auto it = s.scene_building.begin(); it != s.scene_building.end();) {
            const size_t slash = it->first.find('/');
            if (!s.scene_parts.holds(it->first.substr(0, slash), it->first.substr(slash + 1)))
                it = s.scene_building.erase(it);
            else
                ++it;
        }
        if (frames.empty()) return;
        if (s.scene_versions.size() >= size_t(CALIPER_NET_MAX_BATCHES) * CALIPER_NET_MAX_RESOURCES &&
            !s.scene_versions.count(key))
            s.scene_versions.clear();
        Session::SceneVersion& v = s.scene_versions[key];
        v.count = m.count;
        v.frames = std::move(frames);
    }

    // The update in the piece form takes the versions it names into one bundle: their frames as they came,
    // then the update. Null when a version did not come or has another count, which drops the update.
    std::shared_ptr<SceneMsg> scene_bundle(Session& s, const wire::SceneUpdate& up, const ScenePtr& update) {
        auto group = std::make_shared<SceneMsg>();
        group->type = CALIPER_NET_MSG_SCENE_UPDATE;
        group->path = up.path;
        group->part = up.part;
        group->is_static = up.is_static;
        group->piece_form = true;
        group->meta = Json::object();
        if (up.state) group->meta.set("state", true);
        for (const auto& kv : up.resources.members()) {
            auto it = s.scene_versions.find(*up.program + "/" + kv.first);
            uint64_t count = 0;
            if (it == s.scene_versions.end() || !kv.second.get_u64(&count) || it->second.count != count) return nullptr;
            for (const SharedBytes& f : it->second.frames) {
                auto raw = std::make_shared<SceneMsg>();
                raw->type = CALIPER_NET_MSG_RESOURCE;
                raw->raw = true;
                raw->frame = f;
                group->pieces.push_back(std::move(raw));
            }
        }
        for (const auto& kv : up.resources.members()) s.scene_versions.erase(*up.program + "/" + kv.first);
        group->pieces.push_back(update);
        return group;
    }

    size_t live_count() const {
        size_t n = 0;
        for (const auto& kv : sessions) n += kv.second.live ? 1 : 0;
        return n;
    }

    // ---- viewers: attaching ------------------------------------------------------------
    void attach(Peer& p, const std::string& sid) {
        if (p.attached == sid) {   // already there: confirm
            Session* s = sid.empty() ? nullptr : &sessions.at(sid);
            send(p.conn, views_msg(s, p.cap_selection, p.cap_scenes));
            if (s) send_state(p, *s);
            return;
        }
        detach_all(p);
        p.attached = sid;
        Session* s = nullptr;
        if (!sid.empty()) {
            s = &sessions.at(sid);
            s->viewers.insert(p.conn);
        }
        send(p.conn, views_msg(s, p.cap_selection, p.cap_scenes));
        if (s) send_state(p, *s);
    }

    // Ends every subscription of a viewer and leaves its session (no message).
    void detach_all(Peer& p) {
        std::vector<uint32_t> subs;
        for (auto& kv : p.subs) subs.push_back(kv.first);
        for (uint32_t s : subs) end_vsub(p, s);
        p.pending_sub.clear();
        // The controls and selections of the session it leaves are not sent.
        p.pending_controls.reset();
        p.pending_selections.clear();
        p.pending_resources.clear();   // 3.2 draft
        stop_sending_resource(p);
        drop_held_controls(p);
        auto it = sessions.find(p.attached);
        if (it != sessions.end()) it->second.viewers.erase(p.conn);
        p.attached.clear();
    }

    // The server's default: a viewer that never sent ATTACH and is not attached
    // to a live session follows the one live session, when there is one.
    void reconsider_defaults() {
        if (live_count() != 1) return;
        std::string live;
        for (auto& kv : sessions)
            if (kv.second.live) live = kv.first;
        for (auto& kv : peers) {
            Peer& p = kv.second;
            if (p.role != Peer::Role::Viewer || p.closing || !p.follows_default || p.attached == live) continue;
            auto it = sessions.find(p.attached);
            if (it != sessions.end() && it->second.live) continue;
            attach(p, live);
        }
    }

    // ---- channels ----------------------------------------------------------------------
    Session* session_of(const Peer& p) {
        auto it = sessions.find(p.attached);
        return it == sessions.end() ? nullptr : &it->second;
    }

    Channel* channel(Session& s, uint64_t id) {
        auto it = s.channels.find(id);
        return it == s.channels.end() ? nullptr : &it->second;
    }

    void mark_dirty(ConnId conn) {
        auto it = peers.find(conn);
        if (it != peers.end()) it->second.dirty = true;
    }

    VSub* vsub(const SubKey& k) {
        auto it = peers.find(k.conn);
        if (it == peers.end()) return nullptr;
        auto jt = it->second.subs.find(k.sub);
        return jt == it->second.subs.end() ? nullptr : &jt->second;
    }

    void set_latest(Channel& ch, SnapPtr snap) {
        if (ch.latest) snap_bytes -= ch.latest->bytes();
        ch.latest = std::move(snap);
        if (ch.latest) snap_bytes += ch.latest->bytes();
    }

    double want_rate(const Channel& ch) {
        double r = 0;
        for (const SubKey& k : ch.members)
            if (VSub* v = vsub(k)) r = std::max(r, v->rate);
        for (const Waiter& w : ch.waiters) r = std::max(r, w.rate);
        if (ch.keeper || ch.kind == CALIPER_NET_KIND_SERIES) r = std::max(r, kDefaultRate);
        return r > 0 ? r : kDefaultRate;
    }

    void send_subscribe(Session& s, Channel& ch) {
        ch.up_request = s.next_request++;
        if (s.next_request == 0) s.next_request = 1;
        s.by_up_request[ch.up_request] = ch.id;
        ch.up_rate = want_rate(ch);
        wire::Subscribe m;
        m.sub = ch.up_sub;
        m.view = ch.view;
        m.request_id = ch.up_request;
        m.max_rate = ch.up_rate;
        m.data_view = ch.requested;
        send(s.producer, m);
    }

    // A new producer subscription for a channel that has none. False at the
    // protocol's limit of subscriptions on one connection.
    bool subscribe_upstream(Session& s, Channel& ch) {
        if (!s.producer || ch.up_sub) return false;
        if (s.by_up_sub.size() >= CALIPER_NET_MAX_SUBSCRIPTIONS) return false;
        ch.up_sub = s.next_sub++;
        if (s.next_sub == 0) s.next_sub = 1;
        s.by_up_sub[ch.up_sub] = ch.id;
        ch.scene_seen = false;       // 3.2 draft: the producer sends the scene's state again
        ch.scene_forgotten = false;
        send_subscribe(s, ch);
        return true;
    }

    void unsubscribe_upstream(Session& s, Channel& ch) {
        if (!ch.up_sub) return;
        if (s.producer) {
            wire::Unsubscribe u;
            u.sub = ch.up_sub;
            send(s.producer, u);
        }
        s.by_up_sub.erase(ch.up_sub);
        if (ch.up_request) s.by_up_request.erase(ch.up_request);
        ch.up_sub = 0;
        ch.up_request = 0;
        ch.up_rate = 0;
    }

    // The producer subscription follows the largest max_rate of the channel's
    // viewers (series only ever go up, since a new SUBSCRIBE resends backfill).
    void adjust_rate(Session& s, Channel& ch) {
        if (!ch.up_sub || ch.up_request || !s.producer) return;
        const double want = want_rate(ch);
        const bool series = ch.kind == CALIPER_NET_KIND_SERIES;
        if (series ? want > ch.up_rate + 1e-9 : std::fabs(want - ch.up_rate) > 1e-9) send_subscribe(s, ch);
    }

    // After a change of membership: a channel nobody watches gives up its
    // producer subscription and stays idle with its last snapshot.
    void settle(Session& s, uint64_t id) {
        Channel* ch = channel(s, id);
        if (!ch) return;
        if (ch->idle()) {
            unsubscribe_upstream(s, *ch);
            ch->used_ms = monotonic_ms();
            const bool keep = ch->resolved_ok && (ch->latest || s.series.count(ch->view) || !ch->scene_held.empty());
            if (!keep) {
                destroy_channel(s, id);
                return;
            }
            trim_idle(s);
        } else {
            adjust_rate(s, *ch);
        }
    }

    // At most max_idle_channels idle channels per session, the least recently used go.
    void trim_idle(Session& s) {
        for (;;) {
            size_t n = 0;
            Channel* lru = nullptr;
            for (auto& kv : s.channels) {
                Channel& ch = kv.second;
                if (!ch.idle()) continue;
                ++n;
                if (!lru || ch.used_ms < lru->used_ms) lru = &ch;
            }
            if (n <= opt.max_idle_channels || !lru) return;
            destroy_channel(s, lru->id);
        }
    }

    // Removes a channel: its viewer subscriptions end without a message (the
    // VIEWS that caused it tells the viewers), its waiters are answered.
    void destroy_channel(Session& s, uint64_t id, const char* code = CALIPER_NET_ERR_UNKNOWN_VIEW,
                         const std::string& text = "the view is gone") {
        auto it = s.channels.find(id);
        if (it == s.channels.end()) return;
        Channel ch = std::move(it->second);
        s.channels.erase(it);
        if (ch.up_sub) {
            if (s.producer) {
                wire::Unsubscribe u;
                u.sub = ch.up_sub;
                send(s.producer, u);
            }
            s.by_up_sub.erase(ch.up_sub);
        }
        if (ch.up_request) s.by_up_request.erase(ch.up_request);
        if (ch.latest) snap_bytes -= ch.latest->bytes();
        release_scene(ch);
        for (const SubKey& k : ch.members) {
            auto pt = peers.find(k.conn);
            if (pt == peers.end()) continue;
            auto st = pt->second.subs.find(k.sub);
            if (st != pt->second.subs.end() && st->second.channel == id) pt->second.subs.erase(st);
        }
        for (const Waiter& w : ch.waiters) fail_waiter(s, ch.view, w, code, text);
    }

    bool waiter_valid(Session& s, const Waiter& w, Peer** out) {
        auto pt = peers.find(w.conn);
        if (pt == peers.end() || pt->second.closing || pt->second.attached != s.id) return false;
        Peer& p = pt->second;
        if (w.view_request) {
            auto st = p.subs.find(w.sub);
            if (st == p.subs.end() || st->second.ticket != w.ticket) return false;
        } else {
            auto pi = p.pending_sub.find(w.sub);
            if (pi == p.pending_sub.end() || pi->second != w.ticket) return false;
        }
        *out = &p;
        return true;
    }

    void fail_waiter(Session& s, const std::string& view, const Waiter& w, const std::string& code,
                     const std::string& text) {
        Peer* p = nullptr;
        if (!waiter_valid(s, w, &p)) return;
        if (!w.view_request) p->pending_sub.erase(w.sub);
        reply_error(p->conn, w.request_id, w.sub, view, code, text);
    }

    // Puts a viewer subscription on a channel.
    void join(Session& s, Peer& p, VSub& v, Channel& ch) {
        v.channel = ch.id;
        v.series = ch.kind == CALIPER_NET_KIND_SERIES;
        v.pending.reset();
        ch.members.insert({p.conn, v.sub});
        ch.used_ms = monotonic_ms();
        if (ch.kind == CALIPER_NET_KIND_SCENE) {   // 3.2 draft
            v.scene_pieces = p.cap_scene_pieces;
            // The hub forgot the scene's state: it subscribes again for a fresh one, which
            // reaches this viewer as the producer's own state.
            if (ch.scene_forgotten && ch.up_sub && !ch.up_request && s.producer) {
                ch.scene_forgotten = false;
                send_subscribe(s, ch);
            }
            join_scene(v, ch);
            p.dirty = true;
            return;
        }
        if (v.series) {
            v.backfill = backfill_of(ch.resolved);
            v.next = 0;
            v.sent_any = false;
        } else {
            v.pending = ch.latest;
            ++v.pending_n;
        }
        p.dirty = true;
        (void)s;
    }

    void leave(Session& s, Peer& p, VSub& v) {
        v.pending.reset();
        v.scene_first.clear();
        v.scene_pending.clear();
        v.transforms_pending.clear();
        Channel* ch = channel(s, v.channel);
        v.channel = 0;
        if (!ch) return;
        ch->members.erase({p.conn, v.sub});
        settle(s, ch->id);
    }

    void end_vsub(Peer& p, uint32_t sub) {
        auto it = p.subs.find(sub);
        if (it == p.subs.end()) return;
        if (Session* s = session_of(p)) leave(*s, p, it->second);
        p.subs.erase(sub);
        if (p.held_controls) p.dirty = true;   // it may have waited for this one
    }

    // The viewer's request is answered by a channel that has resolved.
    void serve_waiter(Session& s, Channel& ch, const Waiter& w) {
        Peer* p = nullptr;
        if (!waiter_valid(s, w, &p)) return;
        if (!w.view_request) {
            p->pending_sub.erase(w.sub);
            VSub v;
            v.sub = w.sub;
            v.view = ch.view;
            v.ticket = w.ticket;
            v.rate = w.rate;
            reply_ok(p->conn, w.request_id, w.sub, ch.view, ch.resolved);
            VSub& in = p->subs[w.sub] = v;
            join(s, *p, in, ch);
        } else {
            VSub& v = p->subs.at(w.sub);
            reply_ok(p->conn, w.request_id, w.sub, ch.view, ch.resolved);
            if (v.channel != ch.id) {
                const uint64_t old = v.channel;
                Channel* oc = channel(s, old);
                if (oc) oc->members.erase({p->conn, v.sub});
                const uint64_t keep_next = v.next;
                const bool was_sent = v.sent_any;
                join(s, *p, v, ch);
                if (v.series) {   // the same series: carry on from where it was
                    v.next = keep_next;
                    v.sent_any = was_sent;
                }
                if (oc) settle(s, old);
            }
        }
        adjust_rate(s, ch);
    }

    // What an ended session can still show for (view, data view): the channel
    // that resolved this request, else the most recently used one of the view
    // that retained something.
    Channel* ended_lookup(Session& s, const std::string& view, const std::string& key, uint64_t exclude = 0) {
        auto vi = s.view_index.find(view);
        if (vi == s.view_index.end()) return nullptr;
        Channel* best = nullptr;
        for (auto& kv : s.channels) {
            Channel& ch = kv.second;
            if (ch.id == exclude || ch.view != view || ch.view_id != vi->second.id || !ch.resolved_ok) continue;
            const bool has = ch.latest || s.series.count(view) || !ch.scene_held.empty();
            if (!has) continue;
            if (ch.aliases.count(key) || ch.resolved_key == key) return &ch;
            if (!best || ch.used_ms > best->used_ms) best = &ch;
        }
        return best;
    }

    // A SUBSCRIBE or VIEW_REQUEST of a viewer for (view, data view).
    void request(Session& s, const std::string& view, const Json& requested, const Waiter& w) {
        auto vi = s.view_index.find(view);
        if (vi == s.view_index.end()) {
            fail_waiter(s, view, w, CALIPER_NET_ERR_UNKNOWN_VIEW, "no view '" + view + "' in session " + s.id);
            return;
        }
        const std::string key = canon(requested);
        if (!s.live) {
            Channel* ch = ended_lookup(s, view, key);
            if (ch) serve_waiter(s, *ch, w);
            else
                fail_waiter(s, view, w, CALIPER_NET_ERR_UNKNOWN_VIEW,
                            "session " + s.id + " has ended and the hub kept nothing of view '" + view + "'");
            return;
        }
        Channel* found = nullptr;
        for (auto& kv : s.channels) {
            Channel& ch = kv.second;
            if (ch.view != view || ch.view_id != vi->second.id) continue;
            if (ch.aliases.count(key) || (ch.resolved_ok && ch.resolved_key == key) ||
                (!ch.resolved_ok && canon(ch.requested) == key)) {
                found = &ch;
                if (ch.up_sub) break;   // prefer one that is subscribed
            }
        }
        if (found) {
            if (found->up_sub && !found->up_request && found->resolved_ok) {
                serve_waiter(s, *found, w);
                return;
            }
            found->waiters.push_back(w);
            if (!found->up_sub && !subscribe_upstream(s, *found)) {
                found->waiters.pop_back();
                fail_waiter(s, view, w, CALIPER_NET_ERR_LIMIT,
                            "the hub holds " + std::to_string(CALIPER_NET_MAX_SUBSCRIPTIONS) +
                                " subscriptions at this producer");
                settle(s, found->id);
            }
            return;
        }
        if (s.by_up_sub.size() >= CALIPER_NET_MAX_SUBSCRIPTIONS) {
            fail_waiter(s, view, w, CALIPER_NET_ERR_LIMIT,
                        "the hub holds " + std::to_string(CALIPER_NET_MAX_SUBSCRIPTIONS) +
                            " subscriptions at this producer");
            return;
        }
        Channel ch;
        ch.id = next_channel++;
        ch.view = view;
        ch.view_id = vi->second.id;
        ch.kind = vi->second.kind;
        ch.requested = requested.is_object() ? requested : Json::object();
        ch.used_ms = monotonic_ms();
        ch.waiters.push_back(w);
        Channel& in = s.channels[ch.id] = std::move(ch);
        subscribe_upstream(s, in);
    }

    // ---- viewers: messages ---------------------------------------------------------------
    void viewer_subscribe(Peer& p, const wire::Subscribe& m) {
        Session* s = session_of(p);
        if (!s) {
            reply_error(p.conn, m.request_id, m.sub, m.view, CALIPER_NET_ERR_NOT_ATTACHED,
                        "attach to a session first");
            return;
        }
        const bool known = p.subs.count(m.sub) || p.pending_sub.count(m.sub);
        if (!known && p.subs.size() + p.pending_sub.size() >= CALIPER_NET_MAX_SUBSCRIPTIONS) {
            reply_error(p.conn, m.request_id, m.sub, m.view, CALIPER_NET_ERR_LIMIT,
                        "at most " + std::to_string(CALIPER_NET_MAX_SUBSCRIPTIONS) + " subscriptions per connection");
            return;
        }
        if (auto vi = s->view_index.find(m.view);   // 3.2 draft: no scene for a viewer without `scenes`
            vi != s->view_index.end() && vi->second.kind == CALIPER_NET_KIND_SCENE && !p.cap_scenes) {
            reply_error(p.conn, m.request_id, m.sub, m.view, CALIPER_NET_ERR_UNKNOWN_VIEW,
                        "no view '" + m.view + "' in session " + s->id);
            return;
        }
        end_vsub(p, m.sub);
        Waiter w;
        w.conn = p.conn;
        w.sub = m.sub;
        w.request_id = m.request_id;
        w.ticket = ++p.tickets;
        w.rate = clamp_rate(m.max_rate);
        w.requested = m.data_view.is_object() ? m.data_view : Json::object();
        p.pending_sub[m.sub] = w.ticket;
        request(*s, m.view, w.requested, w);
    }

    void viewer_view_request(Peer& p, const wire::ViewRequest& m) {
        Session* s = session_of(p);
        auto it = p.subs.find(m.sub);
        if (!s || it == p.subs.end()) {
            reply_error(p.conn, m.request_id, m.sub, m.view, CALIPER_NET_ERR_UNKNOWN_SUBSCRIPTION,
                        "no subscription " + std::to_string(m.sub));
            return;
        }
        if (it->second.view != m.view) {
            reply_error(p.conn, m.request_id, m.sub, m.view, CALIPER_NET_ERR_BAD_REQUEST,
                        "subscription " + std::to_string(m.sub) + " is to view '" + it->second.view + "'");
            return;
        }
        Waiter w;
        w.conn = p.conn;
        w.sub = m.sub;
        w.request_id = m.request_id;
        w.ticket = it->second.ticket;
        w.view_request = true;
        w.rate = it->second.rate;
        w.requested = m.data_view;
        request(*s, m.view, m.data_view, w);
    }

    void viewer_event(Peer& p, const wire::DecodeResult& r, const wire::Event& e) {
        Session* s = session_of(p);
        if (!s || !s->live || !s->producer) {
            ++c.events_dropped;
            return;
        }
        if (!(s->events_cap || s->selection_cap) || server.queued_bytes(s->producer) > kEventQueueLimit) {
            ++c.events_dropped;
            return;
        }
        Json meta = r.frame.meta;
        meta.erase("sections");
        meta.erase("sub");
        // 3.1: a selection's item space and ids only to a producer that keeps
        // linked selections; it reads the rest of the event as before.
        if (!s->selection_cap) {
            if (const Json* d = meta.find("detail"); d && d->is_object() && (d->has("space") || d->has("items"))) {
                Json detail = *d;
                detail.erase("space");
                detail.erase("items");
                meta.set("detail", detail);
            }
        }
        if (e.sub) {
            auto it = p.subs.find(*e.sub);
            if (it != p.subs.end())
                if (Channel* ch = channel(*s, it->second.channel); ch && ch->up_sub) meta.set("sub", ch->up_sub);
        }
        std::vector<uint8_t> b;
        if (!wire::encode_frame(CALIPER_NET_MSG_EVENT, meta, {}, &b)) {
            ++c.events_dropped;
            return;
        }
        send_bytes(s->producer, std::move(b));
        ++c.events_forwarded;
    }

    // ---- 3.1: controls ----------------------------------------------------------------
    static constexpr size_t kMaxControlsWaiting = 64;       // per viewer connection
    static constexpr size_t kMaxRoutes = 1024;              // per session
    static constexpr uint64_t kReplyWaitMs = 60000;         // a reply the producer owes
    static constexpr uint64_t kQueuedWaitMs = CALIPER_NET_ACTION_WAIT_MS + kReplyWaitMs;

    // A viewer's CONTROL: forwarded to the producer of its session only when
    // the hub allows control, else answered here, as "Controls through a hub"
    // says. Selecting is not a control: select events go on as before.
    void viewer_control(Peer& p, const wire::Control& ctl) {
        if (!p.cap_controls) {
            // CONTROL_REPLY goes only to a peer that announced `controls`.
            wire::Error e;
            e.code = CALIPER_NET_ERR_BAD_REQUEST;
            e.message = "a viewer announces the capability 'controls' before it sends CONTROL";
            e.in_reply_to = CALIPER_NET_MSG_CONTROL;
            e.request_id = ctl.request_id;
            send(p.conn, e);
            return;
        }
        auto refuse = [&](const char* code, const std::string& text) {
            ++c.controls_refused;
            control_error(p, ctl.request_id, ctl.op, ctl.name, code, text);
        };
        if (!opt.allow_control) {
            refuse(CALIPER_NET_ERR_FORBIDDEN, "this hub does not allow control; start it with --allow-control");
            return;
        }
        if (p.watch_only) {
            refuse(CALIPER_NET_ERR_FORBIDDEN, "this connection's token may watch the stream, not control a run");
            return;
        }
        Session* s = session_of(p);
        if (!s) {
            refuse(CALIPER_NET_ERR_NOT_ATTACHED, "attach to a session first");
            return;
        }
        if (!s->live || !s->producer) {
            refuse(CALIPER_NET_ERR_UNSUPPORTED, "session " + s->id + " has ended");
            return;
        }
        if (!s->controls_cap) {
            refuse(CALIPER_NET_ERR_UNSUPPORTED, "the producer of session " + s->id + " does not take controls");
            return;
        }
        if (ctl.op == "intervene" && !(p.cap_interventions && s->interventions_cap)) {
            refuse(CALIPER_NET_ERR_UNSUPPORTED, p.cap_interventions
                       ? "the producer of session " + s->id + " does not take interventions"
                       : "a viewer announces the capability 'interventions' before it sends intervene");
            return;
        }
        if (p.controls_waiting >= kMaxControlsWaiting || s->routes.size() >= kMaxRoutes) {
            refuse(CALIPER_NET_ERR_LIMIT, "too many controls wait for the producer's answer");
            return;
        }
        // The hub's own request_id at the producer: viewers' ids may collide.
        uint32_t up = 0;
        do {
            up = s->next_control++;
        } while (up == 0 || s->routes.count(up));
        ControlRoute route;
        route.viewer = p.conn;
        route.request_id = ctl.request_id;
        route.op = ctl.op;
        route.name = ctl.name;
        route.deadline_ms = monotonic_ms() + kReplyWaitMs;
        s->routes[up] = route;
        ++p.controls_waiting;
        wire::Control fwd = ctl;   // only the keys of its op; never `args`
        fwd.request_id = up;
        send(s->producer, fwd);
        ++c.controls_forwarded;
    }

    // The producer's answer, routed to the viewer that asked under its own
    // request_id; the route of a `queued` run stays until the final answer.
    void producer_control_reply(Session& s, const wire::ControlReply& m) {
        auto it = s.routes.find(m.request_id);
        if (it == s.routes.end()) return;
        ControlRoute route = it->second;
        const bool queued = m.status == CALIPER_NET_STATUS_QUEUED;
        if (queued) {
            it->second.deadline_ms = monotonic_ms() + kQueuedWaitMs;
        } else {
            s.routes.erase(it);
        }
        auto pt = peers.find(route.viewer);
        if (pt == peers.end()) return;
        Peer& v = pt->second;
        if (!queued && v.controls_waiting) --v.controls_waiting;
        if (v.closing) return;
        wire::ControlReply r = m;
        r.request_id = route.request_id;
        send(v.conn, r);
        ++c.control_replies;
    }

    // The producer's CONTROLS: kept (also after the session ends) and passed
    // to every viewer of the session that announced `controls`, after the data
    // the hub received before it (deliver_controls).
    // 3.2 draft: a viewer without `interventions` gets the CONTROLS without that key.
    void producer_controls(Session& s, const wire::Controls& m, SharedBytes buf) {
        if (s.controls) meta_bytes -= s.controls->size();
        if (s.controls_plain) meta_bytes -= s.controls_plain->size();
        s.controls = buf;
        s.controls_plain.reset();
        meta_bytes += buf->size();
        if (!m.interventions.is_null()) {
            wire::Controls plain = m;
            plain.interventions = Json();
            std::vector<uint8_t> b;
            if (wire::encode(wire::Message(plain), &b)) {
                s.controls_plain = std::make_shared<const Bytes>(std::move(b));
                meta_bytes += s.controls_plain->size();
            }
        }
        for (ConnId v : s.viewers) {
            auto pt = peers.find(v);
            if (pt != peers.end() && !pt->second.closing) deliver_controls(pt->second, controls_for(pt->second, s));
        }
        if (retained() > opt.max_retained_bytes) enforce_budget();
    }

    // 3.2 draft: a viewer's PROGRAM, forwarded to the producer of its session
    // under a request_id of the hub's, when the hub allows control. Every
    // refusal is a non-fatal ERROR in reply to the PROGRAM.
    void viewer_program(Peer& p, const wire::Program& m) {
        auto refuse = [&](const char* code, const std::string& text) {
            ++c.programs_refused;
            wire::Error e;
            e.code = code;
            e.message = text.substr(0, CALIPER_NET_TEXT_MAX);
            e.in_reply_to = CALIPER_NET_MSG_PROGRAM;
            e.request_id = m.request_id;
            send(p.conn, e);
        };
        if (!p.cap_programs) {
            refuse(CALIPER_NET_ERR_UNSUPPORTED, "a viewer announces the capability 'programs' before it sends PROGRAM");
            return;
        }
        if (!opt.allow_control) {
            refuse(CALIPER_NET_ERR_FORBIDDEN, "this hub does not allow control; start it with --allow-control");
            return;
        }
        if (p.watch_only) {
            refuse(CALIPER_NET_ERR_FORBIDDEN, "this connection's token may watch the stream, not control a run");
            return;
        }
        Session* s = session_of(p);
        if (!s) {
            refuse(CALIPER_NET_ERR_NOT_ATTACHED, "attach to a session first");
            return;
        }
        if (!s->live || !s->producer) {
            refuse(CALIPER_NET_ERR_UNSUPPORTED, "session " + s->id + " has ended");
            return;
        }
        if (!s->programs_cap) {
            refuse(CALIPER_NET_ERR_UNSUPPORTED, "the producer of session " + s->id + " does not take programs");
            return;
        }
        if (!m.piece.is_null() && !(p.cap_resources && s->resources_cap)) {   // an instance of a piece
            refuse(CALIPER_NET_ERR_UNSUPPORTED, p.cap_resources ? "the producer of session " + s->id +
                                                                      " does not take pieces"
                                                                : std::string("a viewer announces the capability "
                                                                              "'resources' before it sends a piece"));
            return;
        }
        if (s->program_routes.size() >= kMaxRoutes) {
            refuse(CALIPER_NET_ERR_LIMIT, "too many programs wait for the producer's answer");
            return;
        }
        uint32_t up = 0;
        do {
            up = s->next_program++;
        } while (up == 0 || s->program_routes.count(up));
        ProgramRoute route;
        route.viewer = p.conn;
        route.request_id = m.request_id;
        route.deadline_ms = monotonic_ms() + kReplyWaitMs;
        s->program_routes[up] = route;
        wire::Program fwd = m;
        fwd.request_id = up;
        send(s->producer, fwd);
        ++c.programs_forwarded;
    }

    // 3.2 draft: the producer's PROGRAM_STATE. With a request_id it goes to the
    // viewer that asked, under its own id. Without one it goes to every viewer
    // of the session that announced `programs`, and the latest is kept for a
    // viewer that attaches later.
    void producer_program_state(Session& s, const wire::ProgramState& m, SharedBytes buf) {
        if (m.request_id) {
            auto it = s.program_routes.find(*m.request_id);
            if (it == s.program_routes.end()) return;
            ProgramRoute route = it->second;
            s.program_routes.erase(it);
            auto pt = peers.find(route.viewer);
            if (pt == peers.end() || pt->second.closing || !pt->second.cap_programs) return;
            wire::ProgramState r = m;
            r.request_id = route.request_id;
            send(pt->second.conn, r);
            ++c.program_states_out;
            return;
        }
        if (s.program_state) meta_bytes -= s.program_state->size();
        s.program_state = buf;
        meta_bytes += buf->size();
        for (ConnId v : s.viewers) {
            auto pt = peers.find(v);
            if (pt == peers.end() || pt->second.closing || !pt->second.cap_programs) continue;
            if (!server.send(pt->second.conn, buf)) overflow(pt->second.conn);
            ++c.program_states_out;
        }
        if (retained() > opt.max_retained_bytes) enforce_budget();
    }

    // The producer's SELECTION of one item space: kept per space, at most
    // CALIPER_NET_MAX_ITEM_SPACES spaces per session, and passed to every
    // viewer of the session that announced `selection`.
    void producer_selection(Session& s, const wire::Selection& m, SharedBytes buf) {
        auto it = s.selections.find(m.space);
        if (it == s.selections.end()) {
            if (s.selections.size() >= CALIPER_NET_MAX_ITEM_SPACES) {
                ++c.selections_dropped;
                return;
            }
            it = s.selections.emplace(m.space, SelectionState()).first;
        }
        if (it->second.frame) meta_bytes -= it->second.frame->size();
        it->second.frame = buf;
        it->second.empty = m.items.size() == 0;
        meta_bytes += buf->size();
        for (ConnId v : s.viewers) {
            auto pt = peers.find(v);
            if (pt != peers.end() && !pt->second.closing) offer_selection(pt->second, m.space, buf);
        }
        if (retained() > opt.max_retained_bytes) enforce_budget();
    }

    void on_viewer_message(Peer& p, const wire::DecodeResult& r, wire::Message& m) {
        if (auto* x = std::get_if<wire::Ping>(&m)) {
            send(p.conn, wire::Pong{x->nonce});
        } else if (std::holds_alternative<wire::Pong>(m)) {
        } else if (auto* a = std::get_if<wire::Attach>(&m)) {
            p.follows_default = false;
            if (a->session && !sessions.count(*a->session)) {
                wire::Error e;
                e.code = CALIPER_NET_ERR_UNKNOWN_SESSION;
                e.message = "no session '" + *a->session + "'";
                e.in_reply_to = CALIPER_NET_MSG_ATTACH;
                e.session = *a->session;
                send(p.conn, e);
                return;
            }
            attach(p, a->session ? *a->session : std::string());
        } else if (auto* s = std::get_if<wire::Subscribe>(&m)) {
            viewer_subscribe(p, *s);
        } else if (auto* u = std::get_if<wire::Unsubscribe>(&m)) {
            end_vsub(p, u->sub);
            p.pending_sub.erase(u->sub);
        } else if (auto* q = std::get_if<wire::ViewRequest>(&m)) {
            viewer_view_request(p, *q);
        } else if (auto* e = std::get_if<wire::Event>(&m)) {
            viewer_event(p, r, *e);
        } else if (auto* ctl = std::get_if<wire::Control>(&m)) {
            viewer_control(p, *ctl);
        } else if (auto* pg = std::get_if<wire::Program>(&m)) {
            viewer_program(p, *pg);   // 3.2 draft
        } else if (std::holds_alternative<wire::Bye>(m)) {
            close(p.conn, CALIPER_NET_CLOSE_NORMAL, "bye");
        } else if (auto* er = std::get_if<wire::Error>(&m)) {
            if (er->fatal) close(p.conn, CALIPER_NET_CLOSE_NORMAL, "peer error");
        } else if (std::holds_alternative<wire::Unknown>(m)) {
        } else {
            fatal(p.conn, CALIPER_NET_CLOSE_PROTOCOL_ERROR, CALIPER_NET_ERR_BAD_REQUEST,
                  std::string("a viewer does not send ") + wire::type_name(r.type));
        }
    }

    // ---- producers ---------------------------------------------------------------------
    static bool valid_id_char(char ch) {
        return (ch >= 'A' && ch <= 'Z') || (ch >= 'a' && ch <= 'z') || (ch >= '0' && ch <= '9') || ch == '.' ||
               ch == '_' || ch == '-';
    }

    void on_producer_hello(Peer& p, const wire::Hello& h) {
        // A connection whose token may only watch does not take the producer
        // role, as caliper.serve() refuses producers (HubOptions::stream_access).
        if (p.watch_only) {
            fatal(p.conn, CALIPER_NET_CLOSE_ROLE_REFUSED, CALIPER_NET_ERR_UNSUPPORTED,
                  "this connection's token may watch the stream, not publish to it");
            return;
        }
        wire::Identity id;
        if (!wire::Identity::parse(h.session, &id)) {
            fatal(p.conn, CALIPER_NET_CLOSE_PROTOCOL_ERROR, CALIPER_NET_ERR_BAD_REQUEST, "a producer's hello needs 'session'");
            return;
        }
        const std::string proposed = id.id;
        // Resume: the same process (id, pid, start time) on a new connection.
        Session* s = nullptr;
        for (auto& kv : sessions) {
            Session& x = kv.second;
            if ((x.id == proposed || x.proposed == proposed) && x.identity.pid == id.pid &&
                x.identity.start_unix_ms == id.start_unix_ms)
                s = &x;
        }
        if (s) {
            const bool was_live = s->live;
            if (s->producer) {
                auto old = peers.find(s->producer);
                if (old != peers.end()) {
                    old->second.role = Peer::Role::Retired;
                    bye(old->first, CALIPER_NET_BYE_REPLACED, CALIPER_NET_CLOSE_NORMAL,
                        "the producer resumed its session on a new connection");
                }
            }
            if (!was_live && live_count() >= opt.max_producers) {
                fatal(p.conn, CALIPER_NET_CLOSE_SERVER_FULL, CALIPER_NET_ERR_LIMIT,
                      "the hub holds " + std::to_string(opt.max_producers) + " live sessions");
                return;
            }
            drop_upstream(*s);
            s->producer = p.conn;
            s->live = true;
            s->ended_ms = 0;
            s->resubscribe = true;
            s->identity.script = id.script;
            set_producer_caps(*s, h.capabilities);
            forget_programs(*s);   // 3.2 draft: the producer relays its draw halves again
            p.role = Peer::Role::Producer;
            p.session = s->id;
            welcome_producer(p, *s);
            log("session " + s->id + " resumed (" + s->identity.script + ", pid " + std::to_string(id.pid) + ")");
            if (!was_live) {
                broadcast_sessions();
                reconsider_defaults();
            }
            return;
        }
        if (live_count() >= opt.max_producers) {
            fatal(p.conn, CALIPER_NET_CLOSE_SERVER_FULL, CALIPER_NET_ERR_LIMIT,
                  "the hub holds " + std::to_string(opt.max_producers) + " live sessions");
            return;
        }
        // The id: kept unless a live session of another process holds it.
        std::string chosen = proposed;
        for (int n = 2; sessions.count(chosen) && sessions.at(chosen).live; ++n) {
            const std::string suffix = "-" + std::to_string(n);
            chosen = proposed.substr(0, CALIPER_NET_SESSION_ID_MAX - suffix.size()) + suffix;
        }
        // An ended session of another process under that id gives way: its
        // viewers move to the new session, as after a notebook kernel restart.
        std::vector<ConnId> movers;
        if (auto it = sessions.find(chosen); it != sessions.end()) {
            movers.assign(it->second.viewers.begin(), it->second.viewers.end());
            remove_session(chosen, /*reattach=*/false);
        }
        Session ns;
        ns.id = chosen;
        ns.proposed = proposed;
        ns.identity = id;
        ns.identity.id = chosen;
        set_producer_caps(ns, h.capabilities);
        ns.producer = p.conn;
        ns.order = next_order++;
        Session& s2 = sessions[chosen] = std::move(ns);
        p.role = Peer::Role::Producer;
        p.session = chosen;
        welcome_producer(p, s2);
        log("session " + chosen + " started (" + id.script + ", pid " + std::to_string(id.pid) + ")");
        broadcast_sessions();
        for (ConnId c2 : movers) {
            auto pt = peers.find(c2);
            if (pt != peers.end() && pt->second.role == Peer::Role::Viewer && !pt->second.closing)
                attach(pt->second, chosen);
        }
        reconsider_defaults();
    }

    static bool has_cap(const std::vector<std::string>& caps, const char* cap) {
        return std::find(caps.begin(), caps.end(), cap) != caps.end();
    }

    static void set_producer_caps(Session& s, const std::vector<std::string>& caps) {
        s.capabilities = caps;
        s.controls_cap = has_cap(caps, CALIPER_NET_CAP_CONTROLS);
        s.selection_cap = has_cap(caps, CALIPER_NET_CAP_SELECTION);
        s.events_cap = has_cap(caps, CALIPER_NET_CAP_EVENTS);
        s.programs_cap = has_cap(caps, CALIPER_NET_CAP_PROGRAMS);
        s.interventions_cap = has_cap(caps, CALIPER_NET_CAP_INTERVENTIONS);
        s.resources_cap = has_cap(caps, CALIPER_NET_CAP_RESOURCES);
        s.resource_parts = wire::SplitResources(false);   // a new producer connection starts no version
        s.resource_building.clear();
        s.scene_versions.clear();
        s.scene_parts = wire::SplitResources(false);
        s.scene_building.clear();
    }

    // The hub's WELCOME. 3.1: it announces `controls` and `selection`, and to a
    // client that announced `controls` it says whether it takes (to a viewer)
    // or forwards (to a producer) controls: only with allow_control.
    wire::Welcome hub_welcome(bool client_has_controls) {
        wire::Welcome w;
        w.role = CALIPER_NET_ROLE_HUB;
        w.capabilities = {CALIPER_NET_CAP_CONTROLS, CALIPER_NET_CAP_SCENES, CALIPER_NET_CAP_SELECTION,
                          CALIPER_NET_CAP_PROGRAMS, CALIPER_NET_CAP_INTERVENTIONS,
                          CALIPER_NET_CAP_RESOURCES,       // 3.2 draft: forwarded
                          CALIPER_NET_CAP_SCENE_PIECES};   // 3.2 draft: bundled per entity part
        Json sw = Json::object();
        sw.set("name", "caliper-hub");
        sw.set("version", opt.software_version);
        w.software = sw;
        if (client_has_controls) w.control = opt.allow_control;
        return w;
    }

    void welcome_producer(Peer& p, Session& s) {
        wire::Welcome w = hub_welcome(s.controls_cap);
        w.session = s.identity.to_json();
        send(p.conn, w);
    }

    // The producer connection is gone or replaced: no producer subscription remains.
    // The producer connection is gone or replaced: no producer subscription
    // remains, and the CONTROL requests that waited for it are answered.
    void drop_upstream(Session& s) {
        for (auto& kv : s.channels) {
            kv.second.up_sub = 0;
            kv.second.up_request = 0;
            kv.second.up_rate = 0;
        }
        s.by_up_sub.clear();
        s.by_up_request.clear();
        s.producer = 0;
        std::map<uint32_t, ControlRoute> routes;
        routes.swap(s.routes);
        for (auto& kv : routes)
            answer_route(kv.second, CALIPER_NET_ERR_UNSUPPORTED,
                         "the session's producer disconnected before it answered");
        std::map<uint32_t, ProgramRoute> programs;
        programs.swap(s.program_routes);
        for (auto& kv : programs)
            answer_program_route(kv.second, CALIPER_NET_ERR_UNSUPPORTED,
                                 "the session's producer disconnected before it answered");
    }

    // A non-fatal ERROR in reply to the PROGRAM of a route (3.2 draft).
    void answer_program_route(const ProgramRoute& r, const char* code, const std::string& text) {
        auto pt = peers.find(r.viewer);
        if (pt == peers.end() || pt->second.closing) return;
        wire::Error e;
        e.code = code;
        e.message = text.substr(0, CALIPER_NET_TEXT_MAX);
        e.in_reply_to = CALIPER_NET_MSG_PROGRAM;
        e.request_id = r.request_id;
        send(pt->second.conn, e);
    }

    // A CONTROL_REPLY with an error to the viewer of a route (3.1).
    void answer_route(const ControlRoute& r, const char* code, const std::string& text) {
        auto pt = peers.find(r.viewer);
        if (pt == peers.end()) return;
        if (pt->second.controls_waiting) --pt->second.controls_waiting;
        if (pt->second.closing) return;
        control_error(pt->second, r.request_id, r.op, r.name, code, text);
    }

    void control_error(Peer& p, uint32_t request_id, const std::string& op, const std::optional<std::string>& name,
                       const char* code, const std::string& text) {
        wire::ControlReply r;
        r.request_id = request_id;
        r.op = op;
        r.status = CALIPER_NET_STATUS_ERROR;
        r.name = name;
        Json e = Json::object();
        e.set("code", code);
        e.set("message", text.substr(0, CALIPER_NET_TEXT_MAX));
        r.error = e;
        send(p.conn, r);
    }

    void end_session(Session& s) {
        drop_upstream(s);
        s.live = false;
        s.ended_ms = monotonic_ms();
        s.resubscribe = true;
        // Requests that waited for the producer get what the session kept.
        std::vector<uint64_t> ids;
        for (auto& kv : s.channels) ids.push_back(kv.first);
        for (uint64_t id : ids) {
            Channel* ch = channel(s, id);
            if (!ch || ch->waiters.empty()) continue;
            std::vector<Waiter> ws;
            ws.swap(ch->waiters);
            const std::string view = ch->view;
            for (const Waiter& w : ws) {
                Channel* alt = ended_lookup(s, view, canon(w.requested), ch->resolved_ok ? 0 : id);
                if (alt) serve_waiter(s, *alt, w);
                else
                    fail_waiter(s, view, w, CALIPER_NET_ERR_UNKNOWN_VIEW,
                                "session " + s.id + " ended before its producer answered");
            }
        }
        for (uint64_t id : ids) {
            Channel* ch = channel(s, id);
            if (ch && ch->members.empty() && ch->waiters.empty() && !ch->resolved_ok) destroy_channel(s, id);
        }
        log("session " + s.id + " ended");
        broadcast_sessions();
        reconsider_defaults();
        while (ended_count() > opt.max_ended_sessions) evict_oldest_ended();
    }

    size_t ended_count() const { return sessions.size() - live_count(); }

    void evict_oldest_ended() {
        std::string oldest;
        uint64_t t = 0;
        for (auto& kv : sessions)
            if (!kv.second.live && (oldest.empty() || kv.second.ended_ms < t)) {
                oldest = kv.first;
                t = kv.second.ended_ms;
            }
        if (oldest.empty()) return;
        ++c.sessions_evicted;
        remove_session(oldest, /*reattach=*/true);
        log("session " + oldest + " forgotten");
        broadcast_sessions();
    }

    // Forgets a session. Its viewers are detached; with `reattach` they follow
    // the default or hear that they are attached to nothing.
    void remove_session(const std::string& id, bool reattach) {
        auto it = sessions.find(id);
        if (it == sessions.end()) return;
        Session& s = it->second;
        std::vector<ConnId> viewers(s.viewers.begin(), s.viewers.end());
        for (ConnId v : viewers) {
            auto pt = peers.find(v);
            if (pt != peers.end()) detach_all(pt->second);
        }
        std::vector<uint64_t> ids;
        for (auto& kv : s.channels) ids.push_back(kv.first);
        for (uint64_t cid : ids) destroy_channel(s, cid);
        for (auto& kv : s.series) series_bytes -= kv.second.v.size() * sizeof(float);
        forget_meta(s);
        for (auto& kv : s.routes) answer_route(kv.second, CALIPER_NET_ERR_UNSUPPORTED, "the session is gone");
        s.routes.clear();
        for (auto& kv : s.program_routes) answer_program_route(kv.second, CALIPER_NET_ERR_UNSUPPORTED, "the session is gone");
        s.program_routes.clear();
        if (s.producer) {
            auto pt = peers.find(s.producer);
            if (pt != peers.end()) pt->second.role = Peer::Role::Retired;
        }
        sessions.erase(it);
        if (!reattach) return;
        for (ConnId v : viewers) {
            auto pt = peers.find(v);
            if (pt == peers.end() || pt->second.closing) continue;
            send(v, views_msg(nullptr, pt->second.cap_selection, pt->second.cap_scenes));
        }
        reconsider_defaults();
    }

    // VIEWS without the item spaces of 3.1 (`items` on a view and on a dim),
    // for a viewer that did not announce `selection`.
    static Json strip_items(const Json& views) {
        Json out = Json::array();
        for (const Json& v : views.items()) {
            Json e = v;
            e.erase("items");
            if (const Json* srcs = v.find("sources"); srcs && srcs->is_array()) {
                Json ns = Json::array();
                for (const Json& src : srcs->items()) {
                    Json sc = src;
                    if (const Json* dims = src.find("dims"); dims && dims->is_array()) {
                        Json nd = Json::array();
                        for (const Json& d : dims->items()) {
                            Json dc = d;
                            dc.erase("items");
                            nd.push(dc);
                        }
                        sc.set("dims", nd);
                    }
                    ns.push(sc);
                }
                e.set("sources", ns);
            }
            out.push(e);
        }
        return out;
    }

    void forget_meta(Session& s) {
        if (s.controls) meta_bytes -= s.controls->size();
        s.controls.reset();
        if (s.controls_plain) meta_bytes -= s.controls_plain->size();
        s.controls_plain.reset();
        if (s.program_state) meta_bytes -= s.program_state->size();
        s.program_state.reset();
        forget_programs(s);   // 3.2 draft
        for (auto& kv : s.selections)
            if (kv.second.frame) meta_bytes -= kv.second.frame->size();
        s.selections.clear();
    }

    void producer_views(Session& s, const wire::Views& m) {
        Json list = Json::array();
        std::map<std::string, ViewInfo> index;
        for (size_t i = 0; i < m.list.size() && i < CALIPER_NET_MAX_VIEWS; ++i) {
            list.push(m.views.at(i));
            index[m.list[i].name] = ViewInfo{m.list[i].id, m.list[i].kind};
        }
        const size_t before = s.views.size();
        // Channels and series of views that went away or were re-created.
        std::vector<uint64_t> gone;
        for (auto& kv : s.channels) {
            auto vi = index.find(kv.second.view);
            if (vi == index.end() || vi->second.id != kv.second.view_id) gone.push_back(kv.first);
        }
        for (uint64_t id : gone) destroy_channel(s, id);
        for (auto it = s.series.begin(); it != s.series.end();) {
            auto vi = index.find(it->first);
            if (vi == index.end() || vi->second.id != it->second.view_id) {
                series_bytes -= it->second.v.size() * sizeof(float);
                it = s.series.erase(it);
            } else {
                ++it;
            }
        }
        s.views = list;
        s.views_plain = strip_items(list);
        s.view_index = index;
        s.have_views = true;
        // After a resume: subscribe again for every channel someone still wants.
        if (s.resubscribe) {
            s.resubscribe = false;
            std::vector<uint64_t> ids;
            for (auto& kv : s.channels) ids.push_back(kv.first);
            for (uint64_t id : ids) {
                Channel* ch = channel(s, id);
                if (ch && !ch->up_sub && (!ch->members.empty() || !ch->waiters.empty() || ch->keeper))
                    subscribe_upstream(s, *ch);
            }
        }
        // Keep every series view from the start, for late viewers.
        for (const auto& kv : index) {
            if (kv.second.kind != CALIPER_NET_KIND_SERIES) continue;
            bool have = false;
            for (auto& ck : s.channels)
                if (ck.second.keeper && ck.second.view == kv.first && ck.second.view_id == kv.second.id) have = true;
            if (have) continue;
            Channel ch;
            ch.id = next_channel++;
            ch.view = kv.first;
            ch.view_id = kv.second.id;
            ch.kind = kv.second.kind;
            ch.keeper = true;
            ch.used_ms = monotonic_ms();
            const uint64_t cid = ch.id;
            Channel& in = s.channels[cid] = std::move(ch);
            if (!subscribe_upstream(s, in)) s.channels.erase(cid);
        }
        // The viewers attached to the session hear the new list, with the item
        // spaces only when they announced `selection`.
        for (bool with_items : {true, false})
            for (bool with_scenes : {true, false}) {
                std::vector<uint8_t> b;
                if (!wire::encode(views_msg(&s, with_items, with_scenes), &b)) continue;
                auto shared = std::make_shared<const Bytes>(std::move(b));
                for (ConnId v : s.viewers) {
                    auto pt = peers.find(v);
                    if (pt == peers.end() || pt->second.cap_selection != with_items ||
                        pt->second.cap_scenes != with_scenes)
                        continue;
                    if (!server.send(v, shared)) overflow(v);
                }
            }
        if (s.views.size() != before) broadcast_sessions();
    }

    void producer_reply(Session& s, const wire::ViewReply& m) {
        auto it = s.by_up_request.find(m.request_id);
        if (it == s.by_up_request.end()) return;
        const uint64_t id = it->second;
        s.by_up_request.erase(it);
        Channel* ch = channel(s, id);
        if (!ch || ch->up_request != m.request_id) return;
        ch->up_request = 0;
        std::vector<Waiter> ws;
        ws.swap(ch->waiters);
        if (!m.error.is_null()) {
            // The producer holds no subscription now.
            s.by_up_sub.erase(ch->up_sub);
            ch->up_sub = 0;
            const Json* code = m.error.find("code");
            const Json* text = m.error.find("message");
            const std::string cs = code && code->is_string() ? code->as_string() : CALIPER_NET_ERR_INTERNAL;
            const std::string ts = text && text->is_string() ? text->as_string() : std::string();
            for (const Waiter& w : ws) fail_waiter(s, ch->view, w, cs, ts);
            if (ch->members.empty() && !ch->resolved_ok) destroy_channel(s, id);
            else settle(s, id);
            return;
        }
        const std::string key = canon(m.data_view);
        // Equal resolved data views share one producer subscription.
        Channel* other = nullptr;
        for (auto& kv : s.channels) {
            Channel& o = kv.second;
            if (o.id == id || o.view != ch->view || o.view_id != ch->view_id || !o.resolved_ok ||
                o.resolved_key != key || o.up_request)
                continue;
            if (!other || (o.up_sub && !other->up_sub)) other = &o;
        }
        if (other && other->up_sub) {
            // Merge this channel into the subscribed one.
            other->aliases.insert(ch->aliases.begin(), ch->aliases.end());
            other->aliases.insert(canon(ch->requested));
            other->keeper = other->keeper || ch->keeper;
            for (const SubKey& k : ch->members)
                if (VSub* v = vsub(k)) {
                    v->channel = other->id;
                    other->members.insert(k);
                }
            ch->members.clear();
            if (!other->latest && ch->latest) set_latest(*other, ch->latest);
            if (other->scene_held.empty()) other->scene_held.swap(ch->scene_held);   // 3.2 draft
            const uint64_t oid = other->id;
            destroy_channel(s, id);
            Channel* o = channel(s, oid);
            for (const Waiter& w : ws)
                if (o) serve_waiter(s, *o, w);
            if (o) adjust_rate(s, *o);
            return;
        }
        ch->resolved = m.data_view;
        ch->resolved_key = key;
        ch->resolved_ok = true;
        ch->aliases.insert(canon(ch->requested));
        if (other && !other->up_sub) {
            // An idle channel with the same data view gives this one its aliases.
            ch->aliases.insert(other->aliases.begin(), other->aliases.end());
            if (!ch->latest && other->latest) set_latest(*ch, other->latest);
            if (ch->scene_held.empty()) ch->scene_held.swap(other->scene_held);      // 3.2 draft
            for (const SubKey& k : other->members)
                if (VSub* v = vsub(k)) {
                    v->channel = ch->id;
                    ch->members.insert(k);
                }
            other->members.clear();
            const uint64_t oid = other->id;
            destroy_channel(s, oid);
            ch = channel(s, id);
            if (!ch) return;
        }
        for (const Waiter& w : ws) {
            Channel* x = channel(s, id);
            if (x) serve_waiter(s, *x, w);
        }
        settle(s, id);
    }

    void producer_snapshot(Session& s, wire::DecodeResult& r, const wire::Snapshot& m, SharedBytes buf) {
        auto it = s.by_up_sub.find(m.sub);
        if (it == s.by_up_sub.end()) return;
        Channel* ch = channel(s, it->second);
        if (!ch || r.frame.sections.empty()) return;
        auto snap = std::make_shared<Snap>();
        snap->frame = std::move(buf);
        snap->meta = std::move(r.frame.meta);
        for (const auto& sec : r.frame.sections) snap->lens.push_back((uint32_t)sec.size);
        snap->tail = r.frame.sections.front().offset;
        snap->flags = r.frame.flags;
        set_latest(*ch, snap);
        ch->used_ms = monotonic_ms();
        for (const SubKey& k : ch->members) {
            VSub* v = vsub(k);
            if (!v) continue;
            if (v->pending) ++c.snapshots_superseded;
            v->pending = snap;
            ++v->pending_n;
            mark_dirty(k.conn);
        }
        if (retained() > opt.max_retained_bytes) enforce_budget();
    }

    void producer_series(Session& s, const wire::Series& m) {
        auto it = s.by_up_sub.find(m.sub);
        if (it == s.by_up_sub.end()) return;
        Channel* ch = channel(s, it->second);
        if (!ch || ch->kind != CALIPER_NET_KIND_SERIES) return;
        History& h = s.series[ch->view];
        if (h.view_id != ch->view_id) {
            series_bytes -= h.v.size() * sizeof(float);
            h = History();
            h.view_id = ch->view_id;
        }
        const size_t before = h.v.size();
        const uint64_t n = (uint64_t)m.values.size / sizeof(float);
        const uint8_t* p = m.values.data;
        auto val = [&](uint64_t i) {
            float f;
            std::memcpy(&f, p + i * sizeof(float), sizeof f);
            return f;
        };
        if (h.v.empty()) h.first = m.start;
        if (m.start > h.end()) {   // a gap the hub cannot fill: start again from here
            h.v.clear();
            h.first = m.start;
        }
        const uint64_t a = std::max(m.start, h.first), b = std::min(m.start + n, h.end());
        for (uint64_t i = a; i < b; ++i) h.v[(size_t)(i - h.first)] = val(i - m.start);
        for (uint64_t i = std::max(m.start, h.end()); i < m.start + n; ++i) h.v.push_back(val(i - m.start));
        h.total = std::max({h.total, m.total, h.end()});
        h.stamp_ns = std::max(h.stamp_ns, m.stamp_ns);
        // Bounded: series_retain values, and no growth past the byte budget.
        size_t cap = opt.series_retain;
        if (retained() + (h.v.size() - before) * sizeof(float) > opt.max_retained_bytes)
            cap = std::max(kSeriesKeep, std::min(cap, before));
        while (h.v.size() > cap) {
            h.v.pop_front();
            ++h.first;
        }
        series_bytes = series_bytes - before * sizeof(float) + h.v.size() * sizeof(float);
        for (auto& kv : s.channels) {
            if (kv.second.view != ch->view) continue;
            for (const SubKey& k : kv.second.members) mark_dirty(k.conn);
        }
        if (retained() > opt.max_retained_bytes) enforce_budget();
    }

    // ---- 3.2 draft: scenes ---------------------------------------------------------------
    // A scene message of a producer: the hub keeps, for a late viewer, the newest batch of each
    // entity part of the channel (a camera is one), the class table and the newest entity transform
    // of each entity, applies a clear to what it keeps, and offers the message to the channel's
    // viewers. Clears and the producer's state go to a viewer in order and are never dropped (a clear
    // first drops the viewer's pending batches and transforms it clears); any other batch replaces
    // the viewer's pending one of the same entity part, a class table the pending one, and entity
    // transforms merge into the viewer's pending ones, the newest per entity. The pieces of a split
    // update are put together first and then go as one update, replaced only as a whole.
    void producer_scene(Session& s, wire::DecodeResult& r, const wire::Message& m, SharedBytes buf) {
        const auto* up = std::get_if<wire::SceneUpdate>(&m);
        const auto* cl = std::get_if<wire::SceneClear>(&m);
        const auto* tr = std::get_if<wire::SceneTransforms>(&m);
        const auto* kl = std::get_if<wire::SceneClasses>(&m);
        const uint32_t sub = up ? up->sub : cl ? cl->sub : tr ? tr->sub : kl->sub;
        auto it = s.by_up_sub.find(sub);
        if (it == s.by_up_sub.end()) return;
        Channel* ch = channel(s, it->second);
        if (!ch || ch->kind != CALIPER_NET_KIND_SCENE) return;
        ++c.scenes_in;
        auto msg = std::make_shared<SceneMsg>();
        msg->type = r.type;
        msg->meta = std::move(r.frame.meta);
        msg->flags = r.frame.flags;
        for (const auto& sec : r.frame.sections) msg->lens.push_back((uint32_t)sec.size);
        msg->tail = r.frame.sections.empty() ? buf->size() : r.frame.sections.front().offset;
        msg->frame = std::move(buf);
        bool state = false;
        if (up) {
            msg->path = up->path;
            msg->part = up->part;
            msg->is_static = up->is_static;
            state = up->state;
            ch->scene_seq = std::max(ch->scene_seq, up->seq);
        } else if (cl) {
            msg->path = cl->path;
            msg->is_static = cl->is_static;
            msg->recursive = cl->recursive;
            msg->reset = cl->reset;
            if (cl->parts) msg->parts = *cl->parts;
            state = cl->state;
            ch->scene_seq = std::max(ch->scene_seq, cl->seq);
            if (cl->reset) {   // the producer's state begins: the hub keeps it again
                ch->scene_seen = true;
                ch->scene_forgotten = false;
            }
        } else if (tr) {
            msg->is_static = tr->is_static;
            state = tr->state;
            ch->scene_seq = std::max(ch->scene_seq, tr->seq);
        } else {
            msg->is_static = kl->is_static;
            state = kl->state;
            ch->scene_seq = std::max(ch->scene_seq, kl->seq);
        }
        ch->used_ms = monotonic_ms();
        if (up && !up->split.is_null()) {
            std::shared_ptr<SceneMsg> group = scene_piece(*ch, *up, msg);
            if (!group) return;   // more pieces to come, or a piece of an update not held
            msg = group;
            state = group->meta.find("state") && group->meta.find("state")->is_bool() &&
                    group->meta.find("state")->as_bool();
        } else if (up && up->program) {
            std::shared_ptr<SceneMsg> group = scene_bundle(s, *up, msg);
            if (!group) {   // its versions did not come just before it
                ++c.scene_pieces_dropped;
                return;
            }
            msg = group;
        }
        // The entity transforms of the message, as rows.
        TransformRows rows;
        if (tr) {
            const int64_t t = tr->stamps ? tr->stamps->shape[1] : tr->stamp ? tr->stamp->shape[0] : 0;
            for (size_t k = 0; k < tr->paths.size(); ++k) {
                TransformRow row;
                std::memcpy(row.m, tr->matrices.data + 64 * k, 64);
                row.is_static = tr->is_static;
                row.seq = tr->seq;
                row.stamp_ns = tr->stamp_ns;
                row.stamp.resize((size_t)t);
                if (tr->stamps) std::memcpy(row.stamp.data(), tr->stamps->data + 8 * t * k, 8 * (size_t)t);
                else if (tr->stamp) std::memcpy(row.stamp.data(), tr->stamp->data, 8 * (size_t)t);
                rows[tr->paths[k]] = std::move(row);
            }
        }
        const SceneKey key = kl ? SceneKey{"", kClassesOrder} : SceneKey{msg->path, scene_part_order(msg->part)};
        if (cl) {
            for (auto h = ch->scene_held.begin(); h != ch->scene_held.end();) {
                if (h->first.second == kTransformsOrder) {
                    auto kept = std::make_shared<TransformRows>(*h->second->rows);
                    for (auto q = kept->begin(); q != kept->end();)
                        q = scene_clears_transform(*msg, q->first, q->second) ? kept->erase(q) : std::next(q);
                    scene_bytes -= h->second->bytes();
                    if (kept->empty()) {
                        h = ch->scene_held.erase(h);
                        continue;
                    }
                    auto set = std::make_shared<SceneMsg>();
                    set->type = CALIPER_NET_MSG_SCENE_TRANSFORMS;
                    set->rows = kept;
                    scene_bytes += set->bytes();
                    h->second = set;
                    ++h;
                } else if (scene_clears(*msg, *h->second)) {
                    scene_bytes -= h->second->bytes();
                    h = ch->scene_held.erase(h);
                } else {
                    ++h;
                }
            }
        } else if (!ch->scene_forgotten) {
            const SceneKey hold = tr ? SceneKey{"", kTransformsOrder} : key;
            auto h = ch->scene_held.find(hold);
            if (h == ch->scene_held.end() && ch->scene_held.size() >= CALIPER_NET_MAX_BATCHES) {
                forget_scene(*ch);   // more batches than a scene holds: the producer's state serves a late viewer
            } else if (tr) {
                auto kept = std::make_shared<TransformRows>(h == ch->scene_held.end() ? TransformRows()
                                                                                       : *h->second->rows);
                for (const auto& kv : rows) (*kept)[kv.first] = kv.second;
                auto set = std::make_shared<SceneMsg>();
                set->type = CALIPER_NET_MSG_SCENE_TRANSFORMS;
                set->rows = kept;
                ScenePtr& slot = ch->scene_held[hold];
                if (slot) scene_bytes -= slot->bytes();
                slot = set;
                scene_bytes += set->bytes();
            } else {
                ScenePtr& slot = ch->scene_held[hold];
                if (slot) scene_bytes -= slot->bytes();
                slot = msg;
                scene_bytes += msg->bytes();
            }
        }
        for (const SubKey& k : ch->members) {
            VSub* v = vsub(k);
            if (!v || !v->scene) continue;
            if (msg->piece_form && !v->scene_pieces) continue;   // a viewer of the per-shape parts: the structure alone
            if (cl) {
                if (cl->reset) {   // a new state: nothing queued before it matters
                    v->scene_first.clear();
                    v->transforms_pending.clear();
                }
                for (auto q = v->scene_pending.begin(); q != v->scene_pending.end();)
                    q = scene_clears(*msg, *q->second) ? v->scene_pending.erase(q) : std::next(q);
                for (auto q = v->transforms_pending.begin(); q != v->transforms_pending.end();)
                    q = scene_clears_transform(*msg, q->first, q->second) ? v->transforms_pending.erase(q) : std::next(q);
                v->scene_first.emplace_back(msg, false);
                if (v->scene_first.size() > kSceneFirstMax) resync_scene(s, *ch, *v);
            } else if (state) {
                v->scene_first.emplace_back(msg, false);
                if (v->scene_first.size() > kSceneFirstMax) resync_scene(s, *ch, *v);
            } else if (tr) {
                for (const auto& kv : rows) {
                    if (v->transforms_pending.count(kv.first)) ++c.scenes_superseded;
                    v->transforms_pending[kv.first] = kv.second;
                }
                if (v->transforms_pending.size() > kScenePendingMax) resync_scene(s, *ch, *v);
            } else {
                ScenePtr& slot = v->scene_pending[key];
                if (slot) ++c.scenes_superseded;
                slot = msg;
                if (v->scene_pending.size() > kScenePendingMax) resync_scene(s, *ch, *v);
            }
            mark_dirty(k.conn);
        }
        if (retained() > opt.max_retained_bytes) enforce_budget();
    }

    // A piece of a split update (the draft's "Splitting a large update"): the channel puts the
    // pieces of an update together, piece 0 first, and returns the whole update once its last piece
    // has arrived, else null. A newer piece 0 of the same entity part discards an incomplete update,
    // at most CALIPER_NET_MAX_INCOMPLETE_UPDATES are put together at once (the oldest goes), and a
    // piece of an update not held is dropped. The viewers check the pieces as they put them together.
    std::shared_ptr<SceneMsg> scene_piece(Channel& ch, const wire::SceneUpdate& up, const ScenePtr& piece) {
        const Json& sp = up.split;
        uint64_t update = 0, num = 0, pieces = 0;
        sp.find("update")->get_u64(&update);
        sp.find("piece")->get_u64(&num);
        sp.find("pieces")->get_u64(&pieces);
        auto& building = ch.scene_building;
        if (num == 0) {
            for (auto b = building.begin(); b != building.end();) {
                if (b->first == update || (b->second->path == up.path && b->second->part == up.part)) {
                    scene_bytes -= b->second->bytes();
                    b = building.erase(b);
                } else {
                    ++b;
                }
            }
            auto group = std::make_shared<SceneMsg>();
            group->type = CALIPER_NET_MSG_SCENE_UPDATE;
            group->path = up.path;
            group->part = up.part;
            group->is_static = up.is_static;
            group->meta = Json::object();
            if (up.state) group->meta.set("state", true);
            group->pieces_expected = (size_t)pieces;
            building.emplace_back(update, group);
            while (building.size() > CALIPER_NET_MAX_INCOMPLETE_UPDATES) {
                scene_bytes -= building.front().second->bytes();
                building.erase(building.begin());
            }
        }
        auto b = std::find_if(building.begin(), building.end(), [&](const auto& x) { return x.first == update; });
        if (b == building.end()) {
            ++c.scene_pieces_dropped;
            return nullptr;
        }
        std::shared_ptr<SceneMsg> group = b->second;
        group->pieces.push_back(piece);
        scene_bytes += piece->bytes();
        if (group->pieces.size() < group->pieces_expected) return nullptr;
        scene_bytes -= group->bytes();   // counted again where it is kept
        building.erase(b);
        return group;
    }

    // The hub forgets what it keeps of a scene. The viewers that watch it keep
    // receiving its messages; a later viewer makes the hub subscribe again (join),
    // and the producer's fresh state reaches it.
    void forget_scene(Channel& ch) {
        release_scene(ch);
        ch.scene_seen = false;
        ch.scene_forgotten = true;
    }

    // What a channel keeps of a scene, and the split updates it puts together, leave the budget.
    void release_scene(Channel& ch) {
        for (const auto& kv : ch.scene_held) scene_bytes -= kv.second->bytes();
        ch.scene_held.clear();
        for (const auto& b : ch.scene_building) scene_bytes -= b.second->bytes();
        ch.scene_building.clear();
    }

    // A viewer joins a scene channel. Once the producer's state has passed on
    // it, the hub sends its own: a reset, then every batch it keeps, as state.
    void join_scene(VSub& v, const Channel& ch) {
        v.scene = true;
        v.scene_first.clear();
        v.scene_pending.clear();
        v.scene_last_us.clear();
        if (!ch.scene_seen) return;   // the producer's own state is on its way
        auto reset = std::make_shared<SceneMsg>();
        reset->type = CALIPER_NET_MSG_SCENE_CLEAR;
        reset->reset = true;
        Json meta = Json::object();
        meta.set("view", ch.view);
        meta.set("seq", ch.scene_seq);
        meta.set("stamp_ns", (uint64_t)now_us() * 1000u);
        meta.set("path", "");
        meta.set("reset", true);
        reset->meta = std::move(meta);
        v.scene_first.emplace_back(reset, true);
        v.transforms_pending.clear();
        // In the order of the keys: the class table, the entity transforms, then every camera and
        // batch in the byte order of their paths.
        for (const auto& kv : ch.scene_held)
            if (!kv.second->piece_form || v.scene_pieces) v.scene_first.emplace_back(kv.second, true);
    }

    // A viewer's in-order queue passed kSceneFirstMax, or its pending batches or entity transforms
    // passed kScenePendingMax (it stopped reading while the producer clears or writes new entities):
    // the hub's own state replaces them, since it holds the effect of every message so far.
    // While the hub holds no complete state, the producer's next state does instead: it opens with a
    // reset, which drops what was queued before it, so the queue is dropped now.
    void resync_scene(Session& s, Channel& ch, VSub& v) {
        if (ch.scene_seen) {
            join_scene(v, ch);
            return;
        }
        v.scene_first.clear();
        v.scene_pending.clear();
        v.transforms_pending.clear();
        if (ch.scene_forgotten && ch.up_sub && !ch.up_request && s.producer) {
            ch.scene_forgotten = false;
            send_subscribe(s, ch);
        }
    }

    bool send_scene(Peer& p, VSub& v, const ScenePtr& msg, bool state) {
        if (msg->raw) {   // a scene program's RESOURCE, as it came
            if (!server.send(p.conn, msg->frame)) {
                overflow(p.conn);
                return false;
            }
            return true;
        }
        if (msg->rows) {   // the entity transforms the hub keeps: SCENE_TRANSFORMS built now
            for (const ScenePtr& t : transform_messages(v.view, *msg->rows))
                if (!send_scene(p, v, t, state)) return false;
            return true;
        }
        if (!msg->pieces.empty()) {   // a split update: its pieces in order
            for (const ScenePtr& piece : msg->pieces)
                if (!send_scene(p, v, piece, state)) return false;
            return true;
        }
        Json meta = msg->meta;
        meta.set("sub", v.sub);
        if (state) meta.set("state", true);
        if (!msg->lens.empty()) {
            std::vector<uint8_t> head;
            if (!wire::encode_head(msg->type, meta, msg->lens, msg->tail_len(), &head, nullptr,
                                   CALIPER_NET_PROTOCOL_MINOR, msg->flags))
                return false;
            if (!server.send(p.conn, std::move(head), msg->frame, msg->tail, msg->tail_len())) {
                overflow(p.conn);
                return false;
            }
        } else {
            meta.erase("sections");
            std::vector<uint8_t> b;
            if (!wire::encode_frame(msg->type, meta, {}, &b)) return false;
            if (!server.send(p.conn, std::make_shared<const Bytes>(std::move(b)))) {
                overflow(p.conn);
                return false;
            }
        }
        ++c.scenes_out;
        return true;
    }

    // Over the byte budget: ended sessions go first (the oldest), then the
    // last snapshots of channels nobody watches (the least recently used), then
    // the states of watched scenes (the least recently used), which the hub
    // forgets (3.2 draft).
    void enforce_budget() {
        while (retained() > opt.max_retained_bytes) {
            if (ended_count() > 0) {
                evict_oldest_ended();
                continue;
            }
            Session* ls = nullptr;
            Channel* lru = nullptr;
            for (auto& kv : sessions)
                for (auto& ck : kv.second.channels) {
                    Channel& ch = ck.second;
                    if ((!ch.latest && ch.scene_held.empty()) || !ch.members.empty() || !ch.waiters.empty()) continue;
                    if (!lru || ch.used_ms < lru->used_ms) {
                        lru = &ch;
                        ls = &kv.second;
                    }
                }
            if (!lru) {
                Channel* scene = nullptr;
                for (auto& kv : sessions)
                    for (auto& ck : kv.second.channels)
                        if (!ck.second.scene_held.empty() && (!scene || ck.second.used_ms < scene->used_ms))
                            scene = &ck.second;
                if (!scene) return;
                forget_scene(*scene);
                continue;
            }
            set_latest(*lru, nullptr);
            release_scene(*lru);   // 3.2 draft
            if (!lru->up_sub && lru->idle()) destroy_channel(*ls, lru->id);
        }
    }

    void on_producer_message(Peer& p, wire::DecodeResult& r, wire::Message& m, SharedBytes buf) {
        auto it = sessions.find(p.session);
        if (it == sessions.end()) return;
        Session& s = it->second;
        if (std::holds_alternative<wire::Snapshot>(m)) {
            ++c.snapshots_in;
            producer_snapshot(s, r, std::get<wire::Snapshot>(m), std::move(buf));
        } else if (auto* x = std::get_if<wire::Series>(&m)) {
            ++c.series_in;
            producer_series(s, *x);
        } else if (auto* v = std::get_if<wire::Views>(&m)) {
            producer_views(s, *v);
        } else if (std::holds_alternative<wire::SceneUpdate>(m) || std::holds_alternative<wire::SceneClear>(m) ||
                   std::holds_alternative<wire::SceneTransforms>(m) || std::holds_alternative<wire::SceneClasses>(m)) {
            producer_scene(s, r, m, std::move(buf));                 // 3.2 draft
        } else if (auto* q = std::get_if<wire::ViewReply>(&m)) {
            producer_reply(s, *q);
        } else if (auto* cs = std::get_if<wire::Controls>(&m)) {
            producer_controls(s, *cs, std::move(buf));
        } else if (auto* ps = std::get_if<wire::ProgramState>(&m)) {
            producer_program_state(s, *ps, std::move(buf));   // 3.2 draft
        } else if (auto* rs = std::get_if<wire::Resource>(&m)) {
            producer_resource(s, *rs, std::move(buf));        // 3.2 draft
        } else if (auto* pg = std::get_if<wire::Program>(&m)) {
            producer_program(s, *pg, std::move(buf));         // 3.2 draft: a relayed draw half
        } else if (auto* sel = std::get_if<wire::Selection>(&m)) {
            producer_selection(s, *sel, std::move(buf));
        } else if (auto* cr = std::get_if<wire::ControlReply>(&m)) {
            producer_control_reply(s, *cr);
        } else if (auto* pi = std::get_if<wire::Ping>(&m)) {
            send(p.conn, wire::Pong{pi->nonce});
        } else if (std::holds_alternative<wire::Pong>(m)) {
        } else if (std::holds_alternative<wire::Bye>(m)) {
            close(p.conn, CALIPER_NET_CLOSE_NORMAL, "bye");
        } else if (auto* er = std::get_if<wire::Error>(&m)) {
            if (er->fatal) close(p.conn, CALIPER_NET_CLOSE_NORMAL, "peer error");
        } else if (std::holds_alternative<wire::Unknown>(m)) {
        } else {
            fatal(p.conn, CALIPER_NET_CLOSE_PROTOCOL_ERROR, CALIPER_NET_ERR_BAD_REQUEST,
                  std::string("a producer does not send ") + wire::type_name(r.type) + " to a hub");
        }
    }

    // ---- connections ---------------------------------------------------------------------
    void on_hello(Peer& p, const wire::Hello& h) {
        if (h.role == CALIPER_NET_ROLE_PRODUCER) {
            on_producer_hello(p, h);
            return;
        }
        size_t viewers = 0;
        for (auto& kv : peers) viewers += kv.second.role == Peer::Role::Viewer ? 1 : 0;
        if (viewers >= opt.max_viewers) {
            ++c.refused_full;
            fatal(p.conn, CALIPER_NET_CLOSE_SERVER_FULL, CALIPER_NET_ERR_LIMIT,
                  "the hub holds " + std::to_string(opt.max_viewers) + " viewer connections");
            return;
        }
        p.role = Peer::Role::Viewer;
        p.cap_controls = has_cap(h.capabilities, CALIPER_NET_CAP_CONTROLS);
        p.cap_selection = has_cap(h.capabilities, CALIPER_NET_CAP_SELECTION);
        p.cap_scenes = has_cap(h.capabilities, CALIPER_NET_CAP_SCENES);
        p.cap_programs = has_cap(h.capabilities, CALIPER_NET_CAP_PROGRAMS);
        p.cap_interventions = has_cap(h.capabilities, CALIPER_NET_CAP_INTERVENTIONS);
        p.cap_resources = has_cap(h.capabilities, CALIPER_NET_CAP_RESOURCES);
        p.cap_scene_pieces = p.cap_scenes && has_cap(h.capabilities, CALIPER_NET_CAP_SCENE_PIECES);
        wire::Welcome w = hub_welcome(p.cap_controls);
        if (p.cap_controls && p.watch_only) w.control = false;
        send(p.conn, w);
        send(p.conn, sessions_msg());
        reconsider_defaults();
    }

    void on_message(ConnId conn, Bytes&& data, bool binary) {
        auto pt = peers.find(conn);
        if (pt == peers.end() || pt->second.closing) return;
        Peer& p = pt->second;
        if (!binary) {
            close(conn, CALIPER_NET_CLOSE_UNSUPPORTED_DATA, "binary frames only");
            return;
        }
        if (p.role == Peer::Role::Retired) return;
        auto buf = std::make_shared<const Bytes>(std::move(data));
        wire::Message m;
        wire::DecodeResult r = wire::decode(buf->data(), buf->size(), &m);
        if (r.status == wire::DecodeResult::Status::BadFrame) {
            close(conn, (uint16_t)r.close_code, r.reason);
            return;
        }
        if (r.status == wire::DecodeResult::Status::BadMessage) {
            if (p.role == Peer::Role::None) {
                fatal(conn, CALIPER_NET_CLOSE_PROTOCOL_ERROR, r.error_code, r.message);
                return;
            }
            error(conn, r.error_code, r.message, r.type);
            return;
        }
        switch (p.role) {
            case Peer::Role::None:
                if (auto* h = std::get_if<wire::Hello>(&m)) on_hello(p, *h);
                else if (std::holds_alternative<wire::Bye>(m)) close(conn, CALIPER_NET_CLOSE_NORMAL, "bye");
                else
                    fatal(conn, CALIPER_NET_CLOSE_PROTOCOL_ERROR, CALIPER_NET_ERR_BAD_REQUEST,
                          "the first message is hello");
                break;
            case Peer::Role::Viewer: on_viewer_message(p, r, m); break;
            case Peer::Role::Producer: on_producer_message(p, r, m, std::move(buf)); break;
            case Peer::Role::Retired: break;
        }
    }

    void on_closed(ConnId conn) {
        watch_only.erase(conn);
        auto pt = peers.find(conn);
        if (pt == peers.end()) return;
        Peer& p = pt->second;
        if (p.role == Peer::Role::Viewer) {
            detach_all(p);
            // Its waiting controls get no answer: nobody would read it.
            for (auto& kv : sessions)
                for (auto it = kv.second.routes.begin(); it != kv.second.routes.end();)
                    it = it->second.viewer == conn ? kv.second.routes.erase(it) : std::next(it);
            for (auto& kv : sessions)
                for (auto it = kv.second.program_routes.begin(); it != kv.second.program_routes.end();)
                    it = it->second.viewer == conn ? kv.second.program_routes.erase(it) : std::next(it);
        } else if (p.role == Peer::Role::Producer) {
            auto it = sessions.find(p.session);
            if (it != sessions.end() && it->second.producer == conn) end_session(it->second);
        }
        peers.erase(conn);
    }

    void handle(Event& e) {
        switch (e.type) {
            case Event::Type::Request:
                if (door->answer(server, e.conn, e.request) && opt.stream_access) {
                    std::string t;
                    e.request.query_param(CALIPER_NET_TOKEN_PARAM, &t);
                    if (opt.stream_access(t) != HubOptions::StreamAccess::kFull) watch_only.insert(e.conn);
                }
                break;
            case Event::Type::Open: {
                Peer p;
                p.conn = e.conn;
                p.opened_ms = monotonic_ms();
                p.watch_only = watch_only.erase(e.conn) > 0;
                peers[e.conn] = std::move(p);
                break;
            }
            case Event::Type::Message:
                on_message(e.conn, std::move(e.data), e.binary);
                break;
            case Event::Type::Writable: {
                auto pt = peers.find(e.conn);
                if (pt != peers.end() && pt->second.in_flight && server.queued_bytes(e.conn) == 0) {
                    pt->second.in_flight = false;
                    pt->second.dirty = true;
                }
                // CONTROLS and SELECTIONs that waited for room (3.1).
                if (pt != peers.end() && (pt->second.pending_controls || !pt->second.pending_selections.empty()))
                    pt->second.dirty = true;
                break;
            }
            case Event::Type::Pong:
                break;
            case Event::Type::Closed:
                on_closed(e.conn);
                break;
        }
    }

    // ---- delivery --------------------------------------------------------------------------
    // A held CONTROLS goes right behind the data it waited for, in the same
    // send queue; until then the viewer is due again when its wait runs out.
    void pump(Peer& p, uint64_t now) {
        p.due_us = 0;
        if (p.closing) return;
        release_controls(p, now);
        pump_data(p, now);
        if (!p.held_controls) return;
        release_controls(p, now);
        if (!p.held_controls) return;
        const uint64_t until = p.held_since_us + kOrderWaitUs;
        if (!p.due_us || until < p.due_us) p.due_us = until;
    }

    // Hands the next data message of one viewer to its socket, if the last one
    // has drained: series before snapshots, the subscription that waited
    // longest first, each at most at its max_rate.
    void pump_data(Peer& p, uint64_t now) {
        if (p.pending_controls || !p.pending_selections.empty()) flush_meta(p);   // control messages first
        if (p.in_flight) return;
        Session* s = session_of(p);
        if (!s) return;
        VSub* best = nullptr;
        bool best_series = false;
        uint64_t due_min = 0;
        VSub* scene_v = nullptr;           // 3.2 draft: the scene message to send next, if any
        bool scene_head = false;           // from the in-order queue (else a pending batch)
        SceneKey scene_key;
        for (auto& kv : p.subs) {
            VSub& v = kv.second;
            if (v.scene) {
                // In-order messages and pending entity transforms go now; a pending batch at most
                // at `rate` per entity part, a camera and a class table at once.
                if (!v.scene_first.empty() || !v.transforms_pending.empty()) {
                    if (!scene_v || !scene_head || v.last_us < scene_v->last_us) {
                        scene_v = &v;
                        scene_head = true;
                    }
                } else if (!scene_head) {
                    const uint64_t iv = (uint64_t)(1e6 / v.rate);
                    for (const auto& q : v.scene_pending) {
                        auto lt = v.scene_last_us.find(q.first);
                        const bool unlimited = q.first.second < 0 || q.first.second == kCameraOrder;
                        const uint64_t due = lt == v.scene_last_us.end() || unlimited ? 0 : lt->second + iv;
                        if (due <= now) {
                            if (!scene_v || v.last_us < scene_v->last_us) {
                                scene_v = &v;
                                scene_key = q.first;
                            }
                            break;
                        }
                        if (!due_min || due < due_min) due_min = due;
                    }
                }
                continue;
            }
            bool has = false;
            if (v.series) {
                auto h = s->series.find(v.view);
                has = h != s->series.end() && (std::max(v.next, h->second.first) < h->second.end());
            } else {
                has = (bool)v.pending;
            }
            if (!has) continue;
            if (!v.refill_us) v.tokens = kBurst;
            else v.tokens = std::min(kBurst, v.tokens + (double)(now - v.refill_us) * v.rate / 1e6);
            v.refill_us = now;
            if (v.tokens < 1.0) {
                const uint64_t due = now + (uint64_t)std::ceil((1.0 - v.tokens) * 1e6 / v.rate);
                if (!due_min || due < due_min) due_min = due;
                continue;
            }
            if (!best || (v.series && !best_series) ||
                (v.series == best_series && v.last_us < best->last_us)) {
                best = &v;
                best_series = v.series;
            }
        }
        p.due_us = due_min;
        if (scene_v && !(best && best->series)) {   // 3.2 draft: scenes after series, before snapshots
            bool ok;
            if (scene_head && !scene_v->scene_first.empty()) {
                auto item = scene_v->scene_first.front();
                scene_v->scene_first.pop_front();
                ok = send_scene(p, *scene_v, item.first, item.second);
            } else if (scene_head) {   // the pending entity transforms, merged
                auto set = std::make_shared<SceneMsg>();
                set->type = CALIPER_NET_MSG_SCENE_TRANSFORMS;
                set->rows = std::make_shared<const TransformRows>(std::move(scene_v->transforms_pending));
                scene_v->transforms_pending.clear();
                ok = send_scene(p, *scene_v, set, false);
            } else {
                auto q = scene_v->scene_pending.find(scene_key);
                ScenePtr msg = q->second;
                scene_v->scene_pending.erase(q);
                scene_v->scene_last_us[scene_key] = now;
                ok = send_scene(p, *scene_v, msg, false);
            }
            if (!ok) return;
            scene_v->last_us = now;
            scene_v->sent_any = true;
            p.in_flight = true;
            return;
        }
        // 3.2 draft: resources after series and scenes, before snapshots; the one that waited longest.
        // The parts of a version in parts go in order, one at a time; between two of them at most one
        // waiting version in one frame goes, and another version in parts waits for the end of this one.
        if ((p.sending_resource || !p.pending_resources.empty()) && !(best && best->series)) {
            auto oldest = p.pending_resources.end(), single = p.pending_resources.end();
            for (auto it = p.pending_resources.begin(); it != p.pending_resources.end(); ++it) {
                if (oldest == p.pending_resources.end() || it->second.first < oldest->second.first) oldest = it;
                if (it->second.second->size() == 1 &&
                    (single == p.pending_resources.end() || it->second.first < single->second.first))
                    single = it;
            }
            SharedBytes buf;
            if (p.sending_resource && (!p.sent_part || single == p.pending_resources.end())) {
                buf = (*p.sending_resource)[p.sending_next++];
                p.sent_part = true;
                if (p.sending_next == p.sending_resource->size()) stop_sending_resource(p);
            } else {
                auto next = p.sending_resource ? single : oldest;
                if (next == p.pending_resources.end()) return;
                ResourceVersion v = std::move(next->second.second);
                const std::string key = next->first;
                p.pending_resources.erase(next);
                buf = (*v)[0];
                p.sent_part = false;
                if (v->size() > 1) {
                    p.sending_resource = std::move(v);
                    p.sending_key = key;
                    p.sending_next = 1;
                    p.sent_part = true;
                }
            }
            if (!server.send(p.conn, buf)) {
                overflow(p.conn);
                return;
            }
            ++c.resources_out;
            p.in_flight = true;
            return;
        }
        if (!best) return;
        bool ok;
        if (best->series) ok = send_series(p, *s, *best);
        else ok = send_snapshot(p, *best);
        if (!ok) return;
        best->last_us = now;
        best->sent_any = true;
        best->tokens -= 1.0;
        p.in_flight = true;
    }

    bool send_series(Peer& p, Session& s, VSub& v) {
        History& h = s.series.at(v.view);
        uint64_t start = std::max(v.next, h.first);
        if (!v.sent_any && v.backfill != std::numeric_limits<uint64_t>::max())
            start = std::max(start, h.end() > v.backfill ? h.end() - v.backfill : 0);
        const uint64_t stop = std::min<uint64_t>(h.end(), start + CALIPER_NET_SERIES_MAX_VALUES);
        if (stop <= start) return false;
        std::vector<float> vals((size_t)(stop - start));
        for (uint64_t i = start; i < stop; ++i) vals[(size_t)(i - start)] = h.v[(size_t)(i - h.first)];
        wire::Series m;
        m.sub = v.sub;
        m.view = v.view;
        m.start = start;
        m.total = std::max(h.total, stop);
        m.stamp_ns = h.stamp_ns;
        m.values = wire::Section::make("values", CALIPER_NET_DTYPE_F32, {(int64_t)vals.size()}, vals.data(),
                                       vals.size() * sizeof(float));
        v.next = stop;
        if (!send(p.conn, m)) return false;
        ++c.series_out;
        c.series_values_out += stop - start;
        return true;
    }

    bool send_snapshot(Peer& p, VSub& v) {
        SnapPtr snap = std::move(v.pending);
        v.pending.reset();
        Json meta = snap->meta;
        meta.set("sub", v.sub);
        std::vector<uint8_t> head;
        if (!wire::encode_head(CALIPER_NET_MSG_SNAPSHOT, meta, snap->lens, snap->tail_len(), &head, nullptr,
                               CALIPER_NET_PROTOCOL_MINOR, snap->flags))
            return false;
        if (!server.send(p.conn, std::move(head), snap->frame, snap->tail, snap->tail_len())) {
            overflow(p.conn);
            return false;
        }
        ++c.snapshots_out;
        return true;
    }

    // ---- timers ------------------------------------------------------------------------------
    void tick() {
        const uint64_t ms = monotonic_ms();
        for (auto& kv : peers) {
            Peer& p = kv.second;
            if (p.closing) continue;
            if (p.role == Peer::Role::None) {
                if (ms - p.opened_ms >= (uint64_t)opt.handshake_timeout_ms)
                    close(p.conn, CALIPER_NET_CLOSE_HANDSHAKE_TIMEOUT, "no hello");
                continue;
            }
            if (p.role == Peer::Role::Retired) continue;
            const uint64_t last = server.last_input_ms(p.conn);
            if (ms - last >= (uint64_t)opt.idle_timeout_ms) {
                close(p.conn, CALIPER_NET_CLOSE_IDLE_TIMEOUT, "idle");
            } else if (ms - last >= (uint64_t)opt.ping_interval_ms && p.pinged_at != last) {
                p.pinged_at = last;
                send(p.conn, wire::Ping{++p.nonce});
            }
        }
        // CONTROL requests whose answer never came (3.1).
        for (auto& kv : sessions)
            for (auto it = kv.second.routes.begin(); it != kv.second.routes.end();) {
                if (it->second.deadline_ms > ms) {
                    ++it;
                    continue;
                }
                answer_route(it->second, CALIPER_NET_ERR_TIMEOUT, "the producer did not answer in time");
                it = kv.second.routes.erase(it);
            }
        for (auto& kv : sessions)   // 3.2 draft: PROGRAMs whose PROGRAM_STATE never came
            for (auto it = kv.second.program_routes.begin(); it != kv.second.program_routes.end();) {
                if (it->second.deadline_ms > ms) {
                    ++it;
                    continue;
                }
                answer_program_route(it->second, CALIPER_NET_ERR_TIMEOUT, "the producer did not answer in time");
                it = kv.second.program_routes.erase(it);
            }
        if (ms - last_expire_ms >= 100) {
            last_expire_ms = ms;
            for (;;) {
                std::string old;
                for (auto& kv : sessions)
                    if (!kv.second.live && ms - kv.second.ended_ms >= opt.ended_ttl_ms) old = kv.first;
                if (old.empty()) break;
                ++c.sessions_evicted;
                remove_session(old, true);
                log("session " + old + " forgotten");
                broadcast_sessions();
            }
        }
    }

    int timeout_ms(uint64_t now) {
        int t = 250;
        for (auto& kv : peers) {
            const Peer& p = kv.second;
            if (p.dirty) return 0;
            if (p.due_us) {
                const uint64_t d = p.due_us > now ? (p.due_us - now + 999) / 1000 : 0;
                t = std::min<int>(t, (int)d);
            }
        }
        return t;
    }

    // Stats for other threads: every 20 ms while busy, and before each wait.
    void publish(bool force) {
        const uint64_t ms = monotonic_ms();
        if (!force && ms - last_publish_ms < 20) return;
        last_publish_ms = ms;
        HubStats st = c;
        st.connections = peers.size();
        for (auto& kv : peers) {
            st.viewers += kv.second.role == Peer::Role::Viewer ? 1 : 0;
            st.producers += kv.second.role == Peer::Role::Producer ? 1 : 0;
            st.viewer_subscriptions += kv.second.subs.size();
            st.queued_bytes += server.queued_bytes(kv.first);
        }
        for (auto& kv : sessions) {
            (kv.second.live ? st.sessions_live : st.sessions_ended) += 1;
            st.channels += kv.second.channels.size();
            st.channels_open += kv.second.by_up_sub.size();
        }
        st.retained_snapshot_bytes = snap_bytes;
        st.retained_series_bytes = series_bytes;
        st.retained_control_bytes = meta_bytes;
        st.retained_scene_bytes = scene_bytes;
        st.retained_resource_bytes = resource_bytes;
        for (auto& kv : sessions) st.controls_waiting += kv.second.routes.size();
        const DoorStats& d = door->stats();
        st.refused_host = d.refused_host;
        st.refused_origin = d.refused_origin;
        st.refused_token = d.refused_token;
        st.http_files = d.files;
        st.http_not_found = d.not_found;
        std::lock_guard<std::mutex> g(stats_mu);
        published = st;
    }

    void run(const std::atomic<bool>& stop) {
        {
            std::lock_guard<std::mutex> g(replies_mu);
            running = true;
        }
        while (!stop.load()) {
            uint64_t now = now_us();
            int wait = timeout_ms(now);
            if (opt.tick_ms > 0) wait = std::min(wait, opt.tick_ms);
            publish(wait > 0);   // current whenever the loop is about to wait
            std::vector<Event> evs = server.poll(wait);
            for (Event& e : evs) handle(e);
            write_replies();
            if (opt.on_tick) opt.on_tick();
            tick();
            now = now_us();
            for (auto& kv : peers) {
                Peer& p = kv.second;
                if (p.role != Peer::Role::Viewer) continue;
                if (p.dirty || (p.due_us && p.due_us <= now)) {
                    p.dirty = false;
                    pump(p, now);
                }
            }
            ++c.loops;
        }
        {
            std::lock_guard<std::mutex> g(replies_mu);
            running = false;
            replies.clear();
        }
        // Shut down: BYE to every connection, then the close handshake.
        for (auto& kv : peers)
            if (!kv.second.closing && server.is_open(kv.first)) {
                wire::Bye b;
                b.reason = CALIPER_NET_BYE_SHUTDOWN;
                b.message = "the hub is shutting down";
                send(kv.first, b);
            }
        server.shutdown(CALIPER_NET_CLOSE_GOING_AWAY, opt.shutdown_timeout_ms);
        peers.clear();
        publish(true);
    }
};

// ---- Hub ---------------------------------------------------------------------------------------

Hub::Hub(HubOptions opt) : impl_(std::make_unique<Impl>(std::move(opt))) {}
Hub::~Hub() = default;

bool Hub::listen(std::string* err) {
    Impl& m = *impl_;
    if (!valid_token(m.opt.token)) {
        *err = "the session token must be 22 to 128 characters of A-Z, a-z, 0-9, '_' and '-'";
        return false;
    }
    if (!m.opt.assets_dir.empty() && !m.files.open(m.opt.assets_dir, err)) return false;
    if (!m.server.listen(m.opt.port, err, m.opt.host)) return false;
    const std::string note =
        "This is a Caliper hub. It serves no viewer page because it was started without --assets.\n"
        "The Python package ships the viewer page in caliper/_web/; start the hub with\n"
        "--assets <that folder> to open the stream in a browser at this address.\n";
    m.door = std::make_unique<HttpDoor>(m.opt.token, m.server.port(), &m.files, note,
                                        m.opt.stream_path, m.opt.on_request, m.opt.stream_alias);
    if (m.opt.stream_access) {
        auto access = m.opt.stream_access;
        m.door->set_token_check(
            [access](const std::string& t) { return access(t) != HubOptions::StreamAccess::kRefused; });
    }
    m.publish(true);
    return true;
}

uint16_t Hub::port() const { return impl_->server.port(); }
const std::string& Hub::token() const { return impl_->opt.token; }

std::string Hub::viewer_url() const {
    return "http://127.0.0.1:" + std::to_string(port()) + "/?" CALIPER_NET_TOKEN_PARAM "=" + impl_->opt.token;
}

std::string Hub::stream_url() const {
    return "ws://127.0.0.1:" + std::to_string(port()) + impl_->opt.stream_path;
}

void Hub::respond(ConnId c, int status, std::string content_type, std::string body,
                  std::vector<std::pair<std::string, std::string>> headers) {
    // Woken under the lock: once run() has cleared `running`, nothing touches
    // the server from another thread.
    std::lock_guard<std::mutex> g(impl_->replies_mu);
    if (!impl_->running) return;
    impl_->replies.push_back({c, status, std::move(content_type), std::move(body), std::move(headers)});
    impl_->server.wake();
}

void Hub::run() {
    if (!impl_->door) return;
    impl_->run(stop_);
}

void Hub::stop() {
    stop_.store(true);
    impl_->server.wake();
}

HubStats Hub::stats() const {
    std::lock_guard<std::mutex> g(impl_->stats_mu);
    return impl_->published;
}

// ---- tokens and the hub file ------------------------------------------------------------------

bool valid_token(const std::string& t) {
    if (t.size() < CALIPER_NET_TOKEN_MIN || t.size() > CALIPER_NET_TOKEN_MAX) return false;
    for (char ch : t)
        if (!((ch >= 'A' && ch <= 'Z') || (ch >= 'a' && ch <= 'z') || (ch >= '0' && ch <= '9') || ch == '_' ||
              ch == '-'))
            return false;
    return true;
}

std::string random_token() {
    uint8_t b[24];
    secure_random(b, sizeof b);
    return base64url(b, sizeof b);
}

bool read_token_file(const std::string& path, std::string* token, std::string* err) {
#ifndef _WIN32
    struct stat st{};
    if (::stat(path.c_str(), &st) != 0) {
        *err = "cannot read the token file " + path;
        return false;
    }
    if (st.st_mode & 077) {
        *err = "the token file " + path + " can be read or written by other users; make it private (chmod 600)";
        return false;
    }
#endif
    std::ifstream f(path, std::ios::binary);
    if (!f) {
        *err = "cannot read the token file " + path;
        return false;
    }
    std::string line;
    std::getline(f, line);
    while (!line.empty() && (line.back() == '\r' || line.back() == ' ' || line.back() == '\t')) line.pop_back();
    size_t a = 0;
    while (a < line.size() && (line[a] == ' ' || line[a] == '\t')) ++a;
    line = line.substr(a);
    if (!valid_token(line)) {
        *err = "the token file " + path +
               " does not start with a session token (22 to 128 characters of A-Z, a-z, 0-9, '_' and '-')";
        return false;
    }
    *token = line;
    return true;
}

std::string hub_file_text(uint16_t port, const std::string& token, long pid) {
    Json j = Json::object();
    j.set("version", 1);
    j.set("pid", (int64_t)pid);
    j.set("port", (uint64_t)port);
    j.set("token", token);
    j.set("stream_url", "ws://127.0.0.1:" + std::to_string(port) + CALIPER_NET_PATH);
    j.set("url", "http://127.0.0.1:" + std::to_string(port) + "/?" CALIPER_NET_TOKEN_PARAM "=" + token);
    return j.dump() + "\n";
}

bool write_hub_file(const std::string& path, const std::string& text, std::string* err) {
    std::error_code ec;
    const fs::path p(path);
    if (p.has_parent_path()) fs::create_directories(p.parent_path(), ec);
    const std::string tmp = path + ".tmp" + std::to_string(this_pid());
#ifndef _WIN32
    ::unlink(tmp.c_str());
    const int fd = ::open(tmp.c_str(), O_WRONLY | O_CREAT | O_EXCL | O_TRUNC, 0600);
    if (fd < 0) {
        *err = "cannot write " + tmp;
        return false;
    }
    size_t off = 0;
    while (off < text.size()) {
        const ssize_t w = ::write(fd, text.data() + off, text.size() - off);
        if (w <= 0) break;
        off += (size_t)w;
    }
    ::close(fd);
    if (off != text.size()) {
        ::unlink(tmp.c_str());
        *err = "cannot write " + tmp;
        return false;
    }
#else
    {
        std::ofstream f(tmp, std::ios::binary | std::ios::trunc);
        f << text;
        if (!f) {
            *err = "cannot write " + tmp;
            return false;
        }
    }
#endif
    fs::rename(tmp, p, ec);
    if (ec) {
        fs::remove(tmp, ec);
        *err = "cannot write " + path;
        return false;
    }
    return true;
}

void remove_hub_file(const std::string& path, const std::string& text) {
    std::ifstream f(path, std::ios::binary);
    if (!f) return;
    std::stringstream ss;
    ss << f.rdbuf();
    f.close();
    if (ss.str() != text) return;
    std::error_code ec;
    fs::remove(path, ec);
}

}  // namespace caliper_net
