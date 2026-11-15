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
    bool idle() const { return members.empty() && waiters.empty() && !keeper; }
};

struct ViewInfo {
    uint64_t id = 0;
    int64_t kind = 0;
};

struct Session {
    std::string id;
    std::string proposed;            // the id the producer asked for
    wire::Identity identity;         // with the id the hub chose
    std::vector<std::string> capabilities;
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
};

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
    uint64_t next = 0;               // series: next absolute index to send
    uint64_t backfill = std::numeric_limits<uint64_t>::max();
};

struct Peer {
    enum class Role { None, Viewer, Producer, Retired };
    ConnId conn = 0;
    Role role = Role::None;
    uint64_t opened_ms = 0;
    uint64_t pinged_at = 0;          // last_input_ms when the last PING went out
    uint32_t nonce = 0;
    bool closing = false;
    // viewer
    std::string attached;            // session id, "" when not attached
    bool follows_default = true;
    std::map<uint32_t, VSub> subs;
    std::map<uint32_t, uint64_t> pending_sub;   // sub -> ticket of its waiting SUBSCRIBE
    uint64_t tickets = 0;
    bool in_flight = false;          // a data message is in the send queue
    bool dirty = false;
    uint64_t due_us = 0;             // a rate-limited message is due then
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
        return o;
    }

    HubOptions opt;
    Server server;
    StaticFiles files;
    std::unique_ptr<HttpDoor> door;
    std::map<ConnId, Peer> peers;
    std::map<std::string, Session> sessions;
    uint64_t next_channel = 1;
    uint64_t next_order = 1;
    uint64_t snap_bytes = 0, series_bytes = 0;
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

    wire::Views views_msg(const Session* s) {
        wire::Views v;
        if (s) {
            v.session = s->id;
            v.views = s->views;
        } else {
            v.views = Json::array();
        }
        return v;
    }

    size_t live_count() const {
        size_t n = 0;
        for (const auto& kv : sessions) n += kv.second.live ? 1 : 0;
        return n;
    }

    // ---- viewers: attaching ------------------------------------------------------------
    void attach(Peer& p, const std::string& sid) {
        if (p.attached == sid) {   // already there: confirm
            send(p.conn, views_msg(sid.empty() ? nullptr : &sessions.at(sid)));
            return;
        }
        detach_all(p);
        p.attached = sid;
        Session* s = nullptr;
        if (!sid.empty()) {
            s = &sessions.at(sid);
            s->viewers.insert(p.conn);
        }
        send(p.conn, views_msg(s));
    }

    // Ends every subscription of a viewer and leaves its session (no message).
    void detach_all(Peer& p) {
        std::vector<uint32_t> subs;
        for (auto& kv : p.subs) subs.push_back(kv.first);
        for (uint32_t s : subs) end_vsub(p, s);
        p.pending_sub.clear();
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
            const bool keep = ch->resolved_ok && (ch->latest || s.series.count(ch->view));
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
        if (v.series) {
            v.backfill = backfill_of(ch.resolved);
            v.next = 0;
            v.sent_any = false;
        } else {
            v.pending = ch.latest;
        }
        p.dirty = true;
        (void)s;
    }

    void leave(Session& s, Peer& p, VSub& v) {
        v.pending.reset();
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
            const bool has = ch.latest || s.series.count(view);
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
        if (std::find(s->capabilities.begin(), s->capabilities.end(), CALIPER_NET_CAP_EVENTS) ==
                s->capabilities.end() ||
            server.queued_bytes(s->producer) > kEventQueueLimit) {
            ++c.events_dropped;
            return;
        }
        Json meta = r.frame.meta;
        meta.erase("sections");
        meta.erase("sub");
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
            s->capabilities = h.capabilities;
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
        ns.capabilities = h.capabilities;
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

    void welcome_producer(Peer& p, Session& s) {
        wire::Welcome w;
        w.role = CALIPER_NET_ROLE_HUB;
        Json sw = Json::object();
        sw.set("name", "caliper-hub");
        sw.set("version", opt.software_version);
        w.software = sw;
        w.session = s.identity.to_json();
        send(p.conn, w);
    }

    // The producer connection is gone or replaced: no producer subscription remains.
    void drop_upstream(Session& s) {
        for (auto& kv : s.channels) {
            kv.second.up_sub = 0;
            kv.second.up_request = 0;
            kv.second.up_rate = 0;
        }
        s.by_up_sub.clear();
        s.by_up_request.clear();
        s.producer = 0;
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
        if (s.producer) {
            auto pt = peers.find(s.producer);
            if (pt != peers.end()) pt->second.role = Peer::Role::Retired;
        }
        sessions.erase(it);
        if (!reattach) return;
        for (ConnId v : viewers) {
            auto pt = peers.find(v);
            if (pt == peers.end() || pt->second.closing) continue;
            send(v, views_msg(nullptr));
        }
        reconsider_defaults();
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
        // The viewers attached to the session hear the new list.
        std::vector<uint8_t> b;
        if (wire::encode(views_msg(&s), &b)) {
            auto shared = std::make_shared<const Bytes>(std::move(b));
            for (ConnId v : s.viewers)
                if (!server.send(v, shared)) overflow(v);
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
            mark_dirty(k.conn);
        }
        if (snap_bytes + series_bytes > opt.max_retained_bytes) enforce_budget();
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
        if (snap_bytes + series_bytes + (h.v.size() - before) * sizeof(float) > opt.max_retained_bytes)
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
        if (snap_bytes + series_bytes > opt.max_retained_bytes) enforce_budget();
    }

    // Over the byte budget: ended sessions go first (the oldest), then the
    // last snapshots of channels nobody watches (the least recently used).
    void enforce_budget() {
        while (snap_bytes + series_bytes > opt.max_retained_bytes) {
            if (ended_count() > 0) {
                evict_oldest_ended();
                continue;
            }
            Session* ls = nullptr;
            Channel* lru = nullptr;
            for (auto& kv : sessions)
                for (auto& ck : kv.second.channels) {
                    Channel& ch = ck.second;
                    if (!ch.latest || !ch.members.empty() || !ch.waiters.empty()) continue;
                    if (!lru || ch.used_ms < lru->used_ms) {
                        lru = &ch;
                        ls = &kv.second;
                    }
                }
            if (!lru) return;
            set_latest(*lru, nullptr);
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
        } else if (auto* q = std::get_if<wire::ViewReply>(&m)) {
            producer_reply(s, *q);
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
        wire::Welcome w;
        w.role = CALIPER_NET_ROLE_HUB;
        Json sw = Json::object();
        sw.set("name", "caliper-hub");
        sw.set("version", opt.software_version);
        w.software = sw;
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
        auto pt = peers.find(conn);
        if (pt == peers.end()) return;
        Peer& p = pt->second;
        if (p.role == Peer::Role::Viewer) {
            detach_all(p);
        } else if (p.role == Peer::Role::Producer) {
            auto it = sessions.find(p.session);
            if (it != sessions.end() && it->second.producer == conn) end_session(it->second);
        }
        peers.erase(conn);
    }

    void handle(Event& e) {
        switch (e.type) {
            case Event::Type::Request:
                door->answer(server, e.conn, e.request);
                break;
            case Event::Type::Open: {
                Peer p;
                p.conn = e.conn;
                p.opened_ms = monotonic_ms();
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
    // Hands the next data message of one viewer to its socket, if the last one
    // has drained: series before snapshots, the subscription that waited
    // longest first, each at most at its max_rate.
    void pump(Peer& p, uint64_t now) {
        p.due_us = 0;
        if (p.in_flight || p.closing) return;
        Session* s = session_of(p);
        if (!s) return;
        VSub* best = nullptr;
        bool best_series = false;
        uint64_t due_min = 0;
        for (auto& kv : p.subs) {
            VSub& v = kv.second;
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
