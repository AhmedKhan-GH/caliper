// caliper._netio: the native sender of caliper.serve() (wire protocol v3).
//
// WHY. A producer's network work in Python (framing, encoding, the send loop,
// asyncio's wake-ups) holds the GIL that the training loop needs: measured at 3
// to 8 % of a training loop's throughput with one viewer. This module does all
// of it on a native thread that never takes the GIL.
//
// WHO DOES WHAT.
//   training thread (Python)  reduces a snapshot on its device, queues a
//                             non-blocking copy into pinned host memory and
//                             records a torch.mps.Event (caliper/_net_ring.py).
//   pump thread (Python)      waits in control() with the GIL released; wakes
//                             for a control message, a wake() or its poll tick,
//                             polls the MPS events (that needs the GIL), and
//                             hands each completed snapshot over with publish()
//                             and each series chunk with series_append().
//   native thread (here)      listens on 127.0.0.1; HTTP (the viewer's files,
//                             the upgrade checks); HELLO, WELCOME, SESSIONS,
//                             ATTACH, VIEWS; PING and the timeouts; latest-wins
//                             snapshots and coalesced series per subscription;
//                             encoding and sending. SUBSCRIBE, UNSUBSCRIBE,
//                             VIEW_REQUEST and EVENT go to Python (control()),
//                             which answers with send() and sub_open()/sub_close().
//
// OWNERSHIP. publish() and series_append() copy the bytes they are given
// before they return (with the GIL released), so the caller's pinned memory
// must stay valid only for the duration of the call; from then on this module
// owns the copy. A snapshot is stored once per subscription group
// (shared_ptr) and each connection encodes its own frame from it when the
// connection has room, so a newer snapshot simply replaces the reference
// (latest wins) and an older one is freed when nothing refers to it.
//
// LOCKING. One mutex guards the state below. Python-thread calls take it
// briefly (the GIL released) and wake the native thread; the native thread
// takes it around each round of events. The ws::Server is touched only by the
// native thread.
#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/stl/vector.h>

#include "caliper/live_net.h"
#include "net/json.h"
#include "net/wire.h"
#include "net/ws.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstring>
#include <deque>
#include <fstream>
#include <map>
#include <memory>
#include <mutex>
#include <set>
#include <sstream>
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace nb = nanobind;
using caliper_net::ConnId;
using caliper_net::Json;
using caliper_net::SharedBytes;
namespace wire = caliper_net::wire;

namespace {

uint64_t now_ns() {
    return (uint64_t)std::chrono::duration_cast<std::chrono::nanoseconds>(
               std::chrono::steady_clock::now().time_since_epoch()).count();
}

constexpr size_t kRoom = 64 * 1024;          // send the next frame when less than this is queued
constexpr size_t kMaxControlPerConn = 256;   // messages for Python waiting per connection
constexpr int kMaxPollMs = 50;

// One completed snapshot, owned here (copied out of the caller's memory).
struct Snap {
    std::string view;
    uint64_t seq = 0, stamp_ns = 0;
    int64_t kind = 0;
    int64_t colormap = 0;
    uint64_t flags = 0;
    double vmin = 0.0, vmax = 1.0;
    std::vector<double> stats;
    Json axes = Json::array();
    Json tile;
    std::optional<int64_t> points_dims;
    std::optional<uint64_t> label_count;
    int64_t dtype = CALIPER_NET_DTYPE_F32;
    std::vector<int64_t> shape;
    std::vector<uint8_t> data;
    bool has_aux = false;
    std::vector<uint8_t> aux;                 // (N,) f32
};

struct Group {
    Json data_view = Json::object();
    std::set<std::pair<ConnId, uint32_t>> members;
};

// The retained values of one series (absolute indices [first(), total)).
struct History {
    std::vector<float> buf = std::vector<float>(4096);
    uint64_t total = 0, stamp_ns = 0;
    size_t retain = 1 << 20;
    uint64_t first() const { return total > buf.size() ? total - buf.size() : 0; }
    void append(const float* v, size_t n, uint64_t stamp) {
        if (!n) return;
        size_t cap = buf.size();
        if (total + n > cap && cap < retain) {   // grow while nothing has wrapped
            size_t nc = cap;
            while (nc < total + n && nc < retain) nc *= 2;
            nc = std::min(nc, retain);
            buf.resize(nc);
            cap = nc;
        }
        if (n > cap) {
            v += n - cap;
            total += n - cap;
            n = cap;
        }
        const size_t i = (size_t)(total % cap);
        const size_t k = std::min(n, cap - i);
        std::memcpy(buf.data() + i, v, k * 4);
        if (k < n) std::memcpy(buf.data(), v + k, (n - k) * 4);
        total += n;
        stamp_ns = stamp;
    }
    void values(uint64_t start, uint64_t stop, std::vector<float>* out) const {
        const size_t cap = buf.size();
        out->resize((size_t)(stop - start));
        for (uint64_t a = start, j = 0; a < stop;) {
            const size_t i = (size_t)(a % cap);
            const size_t k = std::min<size_t>((size_t)(stop - a), cap - i);
            std::memcpy(out->data() + j, buf.data() + i, k * 4);
            a += k;
            j += k;
        }
    }
};

struct Sub {
    uint32_t sub = 0;
    std::string view;
    int64_t kind = 0;
    int64_t group = -1;                       // snapshot kinds
    bool series = false;
    uint64_t interval_ns = 16'666'667;
    uint64_t last_sent_ns = 0;
    std::shared_ptr<const Snap> snap;         // the pending snapshot (latest wins)
    uint64_t next = 0;                        // series: the next absolute index to send
};

struct Link {
    ConnId id = 0;
    uint64_t opened_ms = 0;
    bool helloed = false;
    bool attached = false;
    bool follows_default = true;
    bool pinged = false;
    bool closing = false;
    uint32_t nonce = 0;
    std::deque<SharedBytes> control;          // frames before any data
    bool views_pending = false;
    std::map<uint32_t, Sub> subs;
    size_t to_python = 0;                     // its messages waiting in the control queue
    uint64_t frames_out = 0, bytes_out = 0, superseded = 0, snapshots_sent = 0, series_sent = 0;
};

struct Ctl {                                  // a message for Python
    std::string kind;                         // "msg", "attached", "detached", "closed"
    ConnId conn = 0;
    std::vector<uint8_t> frame;               // "msg": the frame as received
};

struct Asset {
    std::string content_type;
    std::string path;
};

SharedBytes encode(const wire::Message& m) {
    auto out = std::make_shared<caliper_net::Bytes>();
    wire::encode(m, out.get());
    return out;
}

const char* kNotBuilt =
    "<!doctype html><meta charset=utf-8><title>Caliper</title>"
    "<p>The Caliper viewer assets are not built.</p>\n";

class NetIO {
public:
    NetIO(int port, std::string token, std::string identity_json, std::string software_json,
          std::vector<std::tuple<std::string, std::string, std::string>> assets, int max_connections,
          size_t series_retain, int handshake_ms, int ping_ms, int idle_ms)
        : token_(std::move(token)), retain_(series_retain), max_conns_(max_connections),
          handshake_ms_(handshake_ms), ping_ms_(ping_ms), idle_ms_(idle_ms) {
        std::string err;
        if (!Json::parse(identity_json.data(), identity_json.size(), &identity_, &err) ||
            !wire::Identity::parse(identity_, &ident_))
            throw std::invalid_argument("identity: " + err);
        if (!software_json.empty() && !Json::parse(software_json.data(), software_json.size(), &software_, &err))
            throw std::invalid_argument("software: " + err);
        for (auto& a : assets) assets_[std::get<0>(a)] = Asset{std::get<1>(a), std::get<2>(a)};
        caliper_net::Options opt;
        opt.max_message_bytes = CALIPER_NET_MAX_FRAME_BYTES;
        opt.max_connections = (size_t)std::max(max_connections, 1) * 2 + 16;   // HTTP requests too
        server_ = std::make_unique<caliper_net::Server>(opt);
        if (!server_->listen((uint16_t)port, &err)) throw std::runtime_error("listen: " + err);
        thread_ = std::thread([this] { loop(); });
    }

    ~NetIO() { close(); }

    uint16_t port() const { return server_->port(); }
    int attached() const { return attached_.load(std::memory_order_relaxed); }

    // ---- Python-thread calls (the GIL is released around them) ----------------------
    void close() {
        if (stop_.exchange(true)) {
            if (thread_.joinable()) thread_.join();
            return;
        }
        server_->wake();
        if (thread_.joinable()) thread_.join();
        std::lock_guard<std::mutex> g(cmu_);
        ctl_wake_ = true;
        ccv_.notify_all();
    }

    void wake() {                             // make control() return
        std::lock_guard<std::mutex> g(cmu_);
        ctl_wake_ = true;
        ccv_.notify_all();
    }

    // Wait up to timeout_ms (<0: until something arrives) for messages for Python.
    std::vector<Ctl> wait_control(int timeout_ms) {
        std::unique_lock<std::mutex> lk(cmu_);
        auto ready = [this] { return !ctl_.empty() || ctl_wake_ || stop_.load(); };
        if (timeout_ms < 0) ccv_.wait(lk, ready);
        else ccv_.wait_for(lk, std::chrono::milliseconds(timeout_ms), ready);
        ctl_wake_ = false;
        std::vector<Ctl> out(std::make_move_iterator(ctl_.begin()), std::make_move_iterator(ctl_.end()));
        ctl_.clear();
        lk.unlock();                          // lock order: mu_ before cmu_, never the reverse
        if (!out.empty()) {
            std::lock_guard<std::mutex> g(mu_);
            for (auto& c : out) {
                auto it = links_.find(c.conn);
                if (it != links_.end() && c.kind == "msg" && it->second.to_python) --it->second.to_python;
            }
        }
        return out;
    }

    void set_views(std::string views_json) {
        Json v;
        std::string err;
        if (!Json::parse(views_json.data(), views_json.size(), &v, &err, 64) || !v.is_array())
            throw std::invalid_argument("views: " + err);
        std::lock_guard<std::mutex> g(mu_);
        views_ = std::move(v);
        for (auto& [id, l] : links_)
            if (l.attached) l.views_pending = true;
        server_->wake();
    }

    bool send(ConnId c, std::string frame) {
        std::lock_guard<std::mutex> g(mu_);
        auto it = links_.find(c);
        if (it == links_.end()) return false;
        it->second.control.push_back(std::make_shared<caliper_net::Bytes>(frame.begin(), frame.end()));
        server_->wake();
        return true;
    }

    bool sub_open(ConnId c, uint32_t sub, std::string view, int64_t kind, int64_t group,
                  std::string data_view_json, double max_rate, int64_t backfill) {
        Json dv;
        std::string err;
        if (!Json::parse(data_view_json.data(), data_view_json.size(), &dv, &err, 32))
            throw std::invalid_argument("data_view: " + err);
        std::lock_guard<std::mutex> g(mu_);
        auto it = links_.find(c);
        if (it == links_.end()) return false;
        Link& l = it->second;
        drop_sub(l, sub);
        Sub s;
        s.sub = sub;
        s.view = std::move(view);
        s.kind = kind;
        s.interval_ns = (uint64_t)(1e9 / std::max(max_rate, 1e-3));
        if (kind == CALIPER_NET_KIND_SERIES) {
            s.series = true;
            History& h = histories_[s.view];
            h.retain = retain_;
            const uint64_t back = backfill < 0 ? h.total : (uint64_t)backfill;
            s.next = std::max(h.first(), h.total > back ? h.total - back : 0);
        } else {
            s.group = group;
            Group& gr = groups_[group];
            gr.data_view = std::move(dv);
            gr.members.insert({c, sub});
        }
        l.subs[sub] = std::move(s);
        server_->wake();
        return true;
    }

    void sub_close(ConnId c, uint32_t sub) {
        std::lock_guard<std::mutex> g(mu_);
        auto it = links_.find(c);
        if (it != links_.end()) drop_sub(it->second, sub);
    }

    void series_drop(const std::string& view) {
        std::lock_guard<std::mutex> g(mu_);
        histories_.erase(view);
    }

    // A completed snapshot of group `group`, copied out of the given memory.
    void publish(int64_t group, std::string view, uint64_t seq, uint64_t stamp_ns, int64_t kind,
                 std::string axes_json, std::string tile_json, int64_t colormap, uint64_t flags, double vmin,
                 double vmax, uint64_t stats_ptr, size_t stats_n, int64_t points_dims, int64_t label_count,
                 int64_t dtype, std::vector<int64_t> shape, uint64_t data_ptr, size_t data_nbytes,
                 uint64_t aux_ptr, size_t aux_n) {
        auto s = std::make_shared<Snap>();
        s->view = std::move(view);
        s->seq = seq;
        s->stamp_ns = stamp_ns;
        s->kind = kind;
        s->colormap = colormap;
        s->flags = flags;
        s->vmin = vmin;
        s->vmax = vmax;
        std::string err;
        if (!axes_json.empty()) Json::parse(axes_json.data(), axes_json.size(), &s->axes, &err, 8);
        if (!tile_json.empty()) Json::parse(tile_json.data(), tile_json.size(), &s->tile, &err, 8);
        if (stats_ptr && stats_n) {
            const float* f = reinterpret_cast<const float*>(stats_ptr);
            s->stats.assign(f, f + std::min<size_t>(stats_n, CALIPER_NET_MAX_STATS));
        }
        if (points_dims > 0) s->points_dims = points_dims;
        if (label_count >= 0 && kind == CALIPER_NET_KIND_POINTS) s->label_count = (uint64_t)label_count;
        s->dtype = dtype;
        s->shape = std::move(shape);
        s->data.resize(data_nbytes);
        if (data_nbytes) std::memcpy(s->data.data(), reinterpret_cast<const void*>(data_ptr), data_nbytes);
        if (aux_ptr) {
            s->has_aux = true;
            s->aux.resize(aux_n * 4);
            std::memcpy(s->aux.data(), reinterpret_cast<const void*>(aux_ptr), aux_n * 4);
        }
        std::shared_ptr<const Snap> snap = std::move(s);
        std::lock_guard<std::mutex> g(mu_);
        ++published_;
        auto it = groups_.find(group);
        if (it == groups_.end()) return;
        for (auto& [conn, sub] : it->second.members) {
            auto lt = links_.find(conn);
            if (lt == links_.end()) continue;
            auto st = lt->second.subs.find(sub);
            if (st == lt->second.subs.end()) continue;
            if (st->second.snap) ++lt->second.superseded;
            st->second.snap = snap;
        }
        server_->wake();
    }

    void series_append(const std::string& view, uint64_t ptr, size_t n, uint64_t stamp_ns) {
        std::lock_guard<std::mutex> g(mu_);
        History& h = histories_[view];
        h.retain = retain_;
        h.append(reinterpret_cast<const float*>(ptr), n, stamp_ns);
        series_values_ += n;
        server_->wake();
    }

    struct Stats {
        std::vector<std::tuple<uint64_t, bool, size_t, uint64_t, uint64_t, uint64_t, uint64_t, uint64_t>> ls;
        uint64_t published = 0, http = 0, refused = 0, values = 0;
        std::map<std::string, uint64_t> totals;
    };
    // Called with the GIL released; to_dict() runs with it.
    Stats collect() {
        Stats st;
        std::lock_guard<std::mutex> g(mu_);
        for (auto& [id, l] : links_)
            st.ls.emplace_back(id, l.attached, l.subs.size(), l.frames_out, l.bytes_out, l.superseded,
                               l.snapshots_sent, l.series_sent);
        st.published = published_;
        st.http = http_requests_;
        st.refused = refused_;
        st.values = series_values_;
        for (auto& [v, h] : histories_) st.totals[v] = h.total;
        return st;
    }
    static nb::dict to_dict(const Stats& st) {
        const auto& ls = st.ls;
        const auto& totals = st.totals;
        const uint64_t published = st.published, http = st.http, refused = st.refused, values = st.values;
        nb::dict d;
        nb::list conns;
        for (auto& t : ls) {
            nb::dict c;
            c["id"] = std::get<0>(t);
            c["attached"] = std::get<1>(t);
            c["subscriptions"] = std::get<2>(t);
            c["frames_out"] = std::get<3>(t);
            c["bytes_out"] = std::get<4>(t);
            c["superseded"] = std::get<5>(t);
            c["snapshots_sent"] = std::get<6>(t);
            c["series_sent"] = std::get<7>(t);
            conns.append(c);
        }
        d["connections"] = conns;
        d["published"] = published;
        d["http_requests"] = http;
        d["refused"] = refused;
        d["series_values"] = values;
        nb::dict s;
        for (auto& [v, n] : totals) s[nb::str(v.c_str())] = n;
        d["series"] = s;
        return d;
    }

private:
    // ---- the native thread ------------------------------------------------------------
    void loop() {
        while (!stop_.load()) {
            const int timeout = next_timeout_ms();
            std::vector<caliper_net::Event> evs = server_->poll(timeout);
            std::lock_guard<std::mutex> g(mu_);
            for (auto& e : evs) handle(e);
            timers();
            for (auto& [id, l] : links_) pump(l);
        }
        std::vector<ConnId> open;
        {
            std::lock_guard<std::mutex> g(mu_);
            wire::Bye bye;
            bye.reason = CALIPER_NET_BYE_SHUTDOWN;
            bye.message = "the producer is closing";
            SharedBytes b = encode(bye);
            for (auto& [id, l] : links_) {
                // what is queued for it goes first, then BYE
                pump(l, /*final=*/true);
                if (server_->is_open(id)) server_->send(id, b);
            }
        }
        server_->shutdown(CALIPER_NET_CLOSE_GOING_AWAY, 2000);
        std::lock_guard<std::mutex> g(mu_);
        for (auto& [id, l] : links_) push_ctl("closed", id);
        links_.clear();
        attached_.store(0);
    }

    int next_timeout_ms() {
        std::lock_guard<std::mutex> g(mu_);
        if (links_.empty()) return kMaxPollMs * 4;
        const uint64_t now = now_ns();
        uint64_t wait = (uint64_t)kMaxPollMs * 1'000'000;
        for (auto& [id, l] : links_) {
            if (server_->queued_bytes(id) >= kRoom) continue;
            for (auto& [n, s] : l.subs) {
                bool has = s.snap != nullptr;
                if (s.series) {
                    auto h = histories_.find(s.view);
                    has = h != histories_.end() && s.next < h->second.total;
                }
                if (!has) continue;
                const uint64_t due = s.last_sent_ns + s.interval_ns;
                wait = std::min(wait, due > now ? due - now : 0);
            }
        }
        return (int)((wait + 999'999) / 1'000'000);
    }

    void push_ctl(const char* kind, ConnId c, std::vector<uint8_t> frame = {}) {
        std::lock_guard<std::mutex> g(cmu_);
        ctl_.push_back(Ctl{kind, c, std::move(frame)});
        ccv_.notify_all();
    }

    void set_attached(Link& l, bool on) {
        if (l.attached == on) return;
        l.attached = on;
        attached_.fetch_add(on ? 1 : -1);
        push_ctl(on ? "attached" : "detached", l.id);
    }

    void queue(Link& l, const wire::Message& m) { l.control.push_back(encode(m)); }

    void fatal(Link& l, const std::string& code, const std::string& msg, uint16_t close_code) {
        wire::Error e;
        e.code = code;
        e.message = msg;
        e.fatal = true;
        server_->send(l.id, encode(e));
        server_->close(l.id, close_code, msg.substr(0, 120));
        l.closing = true;
    }

    void drop_sub(Link& l, uint32_t sub) {
        auto it = l.subs.find(sub);
        if (it == l.subs.end()) return;
        if (it->second.group >= 0) {
            auto g = groups_.find(it->second.group);
            if (g != groups_.end()) {
                g->second.members.erase({l.id, sub});
                if (g->second.members.empty()) groups_.erase(g);
            }
        }
        l.subs.erase(it);
    }

    void respond_static(ConnId c, const caliper_net::HttpRequest& r) {
        std::string path;
        if (!caliper_net::url_decode(r.path, &path)) path = r.path;
        std::string name = path.size() > 1 ? path.substr(1) : "index.html";
        if (!name.empty() && name.back() == '/') name += "index.html";
        std::vector<std::pair<std::string, std::string>> h = {{"Cache-Control", "no-store"},
                                                              {"X-Content-Type-Options", "nosniff"}};
        if (r.has_origin && wire::origin_allowed(r.origin)) {
            h.emplace_back("Access-Control-Allow-Origin", r.origin);
            h.emplace_back("Vary", "Origin");
        }
        auto it = assets_.find(name);
        if (it == assets_.end()) {
            if (name == "index.html" && (assets_.empty() || !assets_.count("index.html"))) {
                server_->respond(c, 200, "text/html; charset=utf-8", kNotBuilt, h);
                return;
            }
            server_->respond(c, 404, "text/plain; charset=utf-8", "Not Found\n", h);
            return;
        }
        std::ifstream f(it->second.path, std::ios::binary);
        if (!f) {
            server_->respond(c, 404, "text/plain; charset=utf-8", "Not Found\n", h);
            return;
        }
        std::ostringstream ss;
        ss << f.rdbuf();
        server_->respond(c, 200, it->second.content_type, ss.str(), h);
    }

    void handle(caliper_net::Event& e) {
        using T = caliper_net::Event::Type;
        switch (e.type) {
            case T::Request: {
                ++http_requests_;
                const auto& r = e.request;
                if (!wire::host_allowed(r.host, port())) {
                    ++refused_;
                    server_->respond(e.conn, 403, "text/plain; charset=utf-8", "Forbidden: Host\n",
                                     {{"Cache-Control", "no-store"}});
                } else if (r.method != "GET") {
                    server_->respond(e.conn, 405, "text/plain; charset=utf-8", "Method Not Allowed\n",
                                     {{"Allow", "GET"}, {"Cache-Control", "no-store"}});
                } else if (r.path == CALIPER_NET_PATH) {
                    const int st = wire::upgrade_status(r, token_, port());
                    if (st) {
                        ++refused_;
                        server_->reject(e.conn, st);
                    } else if (!r.websocket) {
                        server_->respond(e.conn, 426, "text/plain; charset=utf-8", "Upgrade Required\n",
                                         {{"Cache-Control", "no-store"}});
                    } else {
                        server_->accept(e.conn);
                    }
                } else {
                    respond_static(e.conn, r);
                }
                break;
            }
            case T::Open: {
                Link l;
                l.id = e.conn;
                l.opened_ms = caliper_net::monotonic_ms();
                links_[e.conn] = std::move(l);
                break;
            }
            case T::Message: {
                auto it = links_.find(e.conn);
                if (it != links_.end()) on_message(it->second, e);
                break;
            }
            case T::Writable:
            case T::Pong:
                break;
            case T::Closed: {
                auto it = links_.find(e.conn);
                if (it != links_.end()) {
                    Link& l = it->second;
                    for (auto sub = l.subs.begin(); sub != l.subs.end();) drop_sub(l, (sub++)->first);
                    if (l.attached) attached_.fetch_sub(1);
                    if (l.helloed) push_ctl("closed", l.id);
                    links_.erase(it);
                }
                break;
            }
        }
    }

    void on_message(Link& l, caliper_net::Event& e) {
        if (l.closing) return;
        if (!e.binary) {
            server_->close(l.id, CALIPER_NET_CLOSE_UNSUPPORTED_DATA, "binary frames only");
            l.closing = true;
            return;
        }
        l.pinged = false;
        wire::Message m;
        const wire::DecodeResult r = wire::decode(e.data.data(), e.data.size(), &m);
        if (r.status == wire::DecodeResult::Status::BadFrame) {
            server_->close(l.id, (uint16_t)r.close_code, r.reason);
            l.closing = true;
            return;
        }
        if (!l.helloed) {
            if (r.status == wire::DecodeResult::Status::BadMessage) {
                fatal(l, r.error_code, r.message, CALIPER_NET_CLOSE_PROTOCOL_ERROR);
                return;
            }
            if (std::holds_alternative<wire::Bye>(m)) {
                server_->close(l.id, CALIPER_NET_CLOSE_NORMAL);
                l.closing = true;
                return;
            }
            auto* h = std::get_if<wire::Hello>(&m);
            if (!h) {
                fatal(l, CALIPER_NET_ERR_BAD_REQUEST, "the first message is hello",
                      CALIPER_NET_CLOSE_PROTOCOL_ERROR);
                return;
            }
            if (h->role != CALIPER_NET_ROLE_VIEWER) {
                fatal(l, CALIPER_NET_ERR_UNSUPPORTED,
                      "caliper.serve() accepts viewers only; producers connect to a hub",
                      CALIPER_NET_CLOSE_ROLE_REFUSED);
                return;
            }
            size_t helloed = 0;
            for (auto& [id, x] : links_) helloed += x.helloed;
            if ((int)helloed >= max_conns_) {
                fatal(l, CALIPER_NET_ERR_LIMIT, "the server has reached its connection limit",
                      CALIPER_NET_CLOSE_SERVER_FULL);
                return;
            }
            l.helloed = true;
            wire::Welcome w;
            w.role = CALIPER_NET_ROLE_PRODUCER;
            w.capabilities = {CALIPER_NET_CAP_EVENTS};
            w.software = software_;
            w.session = identity_;
            queue(l, w);
            wire::Sessions s;
            Json entry = identity_;
            entry.set("state", CALIPER_NET_SESSION_LIVE);
            entry.set("views", (uint64_t)views_.size());
            Json caps = Json::array();
            caps.push(CALIPER_NET_CAP_EVENTS);
            entry.set("capabilities", caps);
            s.sessions.push(entry);
            queue(l, s);
            push_ctl("opened", l.id);         // before "attached": Python makes its state first
            attach(l, ident_.id);
            return;
        }
        if (r.status == wire::DecodeResult::Status::BadMessage) {
            wire::Error err;
            err.code = r.error_code;
            err.message = r.message;
            queue(l, err);
            return;
        }
        if (auto* p = std::get_if<wire::Ping>(&m)) {
            queue(l, wire::Pong{p->nonce});
        } else if (std::holds_alternative<wire::Pong>(m) || std::holds_alternative<wire::Unknown>(m)) {
        } else if (auto* a = std::get_if<wire::Attach>(&m)) {
            l.follows_default = false;
            if (a->session && *a->session != ident_.id) {
                wire::Error err;
                err.code = CALIPER_NET_ERR_UNKNOWN_SESSION;
                err.message = "no session " + *a->session;
                err.in_reply_to = CALIPER_NET_MSG_ATTACH;
                err.session = *a->session;
                queue(l, err);
            } else {
                attach(l, a->session ? *a->session : std::string());
            }
        } else if (std::holds_alternative<wire::Bye>(m)) {
            server_->close(l.id, CALIPER_NET_CLOSE_NORMAL, "bye");
            l.closing = true;
        } else if (auto* er = std::get_if<wire::Error>(&m)) {
            if (er->fatal) {
                server_->close(l.id, CALIPER_NET_CLOSE_NORMAL, "peer error");
                l.closing = true;
            }
        } else if (std::holds_alternative<wire::Subscribe>(m) || std::holds_alternative<wire::Unsubscribe>(m) ||
                   std::holds_alternative<wire::ViewRequest>(m) || std::holds_alternative<wire::Event>(m)) {
            to_python(l, m, e.data);
        } else {
            fatal(l, CALIPER_NET_ERR_BAD_REQUEST,
                  std::string("a producer does not accept ") + wire::type_name(r.type),
                  CALIPER_NET_CLOSE_PROTOCOL_ERROR);
        }
    }

    // A message only Python can answer, through a bounded queue: beyond
    // kMaxControlPerConn waiting, an EVENT is dropped (the protocol keeps the
    // newest 256 per connection) and a request is refused with `limit` here.
    void to_python(Link& l, const wire::Message& m, caliper_net::Bytes& data) {
        if (l.to_python >= kMaxControlPerConn) {
            ++refused_;
            if (auto* s = std::get_if<wire::Subscribe>(&m)) {
                wire::ViewReply rep;
                rep.request_id = s->request_id;
                rep.sub = s->sub;
                rep.view = s->view;
                Json err = Json::object();
                err.set("code", CALIPER_NET_ERR_LIMIT);
                err.set("message", "too many requests waiting");
                rep.error = err;
                queue(l, rep);
            } else if (auto* v = std::get_if<wire::ViewRequest>(&m)) {
                wire::ViewReply rep;
                rep.request_id = v->request_id;
                rep.sub = v->sub;
                rep.view = v->view;
                Json err = Json::object();
                err.set("code", CALIPER_NET_ERR_LIMIT);
                err.set("message", "too many requests waiting");
                rep.error = err;
                queue(l, rep);
            }
            return;
        }
        ++l.to_python;
        push_ctl("msg", l.id, std::move(data));
    }

    void attach(Link& l, const std::string& session) {
        if (session.empty()) {
            for (auto it = l.subs.begin(); it != l.subs.end();) drop_sub(l, (it++)->first);
            set_attached(l, false);
        } else {
            set_attached(l, true);
        }
        l.views_pending = true;
    }

    void timers() {
        const uint64_t now = caliper_net::monotonic_ms();
        for (auto& [id, l] : links_) {
            if (l.closing) continue;
            if (!l.helloed) {
                if (now - l.opened_ms >= (uint64_t)handshake_ms_) {
                    server_->close(id, CALIPER_NET_CLOSE_HANDSHAKE_TIMEOUT, "no hello");
                    l.closing = true;
                }
                continue;
            }
            const uint64_t idle = now - server_->last_input_ms(id);
            if (idle >= (uint64_t)idle_ms_) {
                server_->close(id, CALIPER_NET_CLOSE_IDLE_TIMEOUT, "idle");
                l.closing = true;
            } else if (idle >= (uint64_t)ping_ms_ && !l.pinged) {
                queue(l, wire::Ping{++l.nonce});
                l.pinged = true;
            }
        }
    }

    // Send what the connection has room for: control frames, then series,
    // then snapshots, each subscription at its own rate.
    void pump(Link& l, bool final = false) {
        if (l.closing || !server_->is_open(l.id)) return;
        const uint64_t now = now_ns();
        while (final || server_->queued_bytes(l.id) < kRoom) {
            SharedBytes f;
            if (l.views_pending && l.control.empty()) {
                wire::Views v;
                if (l.attached) {
                    v.session = ident_.id;
                    v.views = views_;
                }
                f = encode(v);
                l.views_pending = false;
            } else if (!l.control.empty()) {
                f = l.control.front();
                l.control.pop_front();
            } else {
                f = next_data(l, now, final);
            }
            if (!f) break;
            if (!server_->send(l.id, f)) break;
            ++l.frames_out;
            l.bytes_out += f->size();
        }
    }

    SharedBytes next_data(Link& l, uint64_t now, bool final) {
        for (auto& [n, s] : l.subs) {
            if (!s.series) continue;
            auto h = histories_.find(s.view);
            if (h == histories_.end() || s.next >= h->second.total) continue;
            if (!final && s.last_sent_ns && now < s.last_sent_ns + s.interval_ns) continue;
            const uint64_t start = std::max(s.next, h->second.first());
            const uint64_t stop = std::min<uint64_t>(h->second.total, start + CALIPER_NET_SERIES_MAX_VALUES);
            h->second.values(start, stop, &scratch_);
            wire::Series m;
            m.sub = s.sub;
            m.view = s.view;
            m.start = start;
            m.total = h->second.total;
            m.stamp_ns = h->second.stamp_ns;
            m.values = wire::Section::make("values", CALIPER_NET_DTYPE_F32, {(int64_t)(stop - start)},
                                           scratch_.data(), (stop - start) * 4);
            s.next = stop;
            s.last_sent_ns = now;
            ++l.series_sent;
            return encode(m);
        }
        Sub* best = nullptr;
        for (auto& [n, s] : l.subs) {
            if (!s.snap) continue;
            if (!final && s.last_sent_ns && now < s.last_sent_ns + s.interval_ns) continue;
            if (!best || s.last_sent_ns < best->last_sent_ns) best = &s;
        }
        if (!best) return nullptr;
        std::shared_ptr<const Snap> sn = std::move(best->snap);
        best->snap.reset();
        best->last_sent_ns = now;
        ++l.snapshots_sent;
        wire::Snapshot m;
        m.sub = best->sub;
        m.view = sn->view;
        m.seq = sn->seq;
        m.stamp_ns = sn->stamp_ns;
        m.kind = sn->kind;
        auto g = groups_.find(best->group);
        m.data_view = g != groups_.end() ? g->second.data_view : Json::object();
        m.data = wire::Section::make("data", sn->dtype, sn->shape, sn->data.data(), sn->data.size());
        if (sn->has_aux)
            m.aux = wire::Section::make("aux", CALIPER_NET_DTYPE_F32, {(int64_t)(sn->aux.size() / 4)},
                                        sn->aux.data(), sn->aux.size());
        m.colormap = sn->colormap;
        m.flags = sn->flags;
        m.vmin = sn->vmin;
        m.vmax = sn->vmax;
        m.stats = sn->stats;
        m.axes = sn->axes;
        m.tile = sn->tile;
        m.points_dims = sn->points_dims;
        m.label_count = sn->label_count;
        return encode(m);
    }

    std::unique_ptr<caliper_net::Server> server_;
    std::thread thread_;
    std::atomic<bool> stop_{false};
    std::atomic<int> attached_{0};
    std::string token_;
    Json identity_;
    wire::Identity ident_;
    Json software_;
    size_t retain_;
    int max_conns_;
    int handshake_ms_, ping_ms_, idle_ms_;
    std::map<std::string, Asset> assets_;

    std::mutex mu_;                           // everything below
    std::map<ConnId, Link> links_;
    std::map<int64_t, Group> groups_;
    std::map<std::string, History> histories_;
    Json views_ = Json::array();
    std::vector<float> scratch_;
    uint64_t published_ = 0, http_requests_ = 0, refused_ = 0, series_values_ = 0;

    std::mutex cmu_;                          // the queue to Python
    std::condition_variable ccv_;
    std::deque<Ctl> ctl_;
    bool ctl_wake_ = false;
};

}  // namespace

NB_MODULE(_netio, m) {
    m.doc() = "The native sender of caliper.serve() (wire protocol v3); internal.";
    m.attr("MAX_CONTROL_PER_CONNECTION") = kMaxControlPerConn;
    using R = nb::call_guard<nb::gil_scoped_release>;
    nb::class_<NetIO>(m, "Server")
        .def(nb::init<int, std::string, std::string, std::string,
                      std::vector<std::tuple<std::string, std::string, std::string>>, int, size_t, int, int,
                      int>(),
             nb::arg("port"), nb::arg("token"), nb::arg("identity"), nb::arg("software"), nb::arg("assets"),
             nb::arg("max_connections"), nb::arg("series_retain"), nb::arg("handshake_ms"),
             nb::arg("ping_ms"), nb::arg("idle_ms"), R())
        .def_prop_ro("port", &NetIO::port)
        .def("attached", &NetIO::attached)
        .def("close", &NetIO::close, R())
        .def("wake", &NetIO::wake, R())
        .def("control",
             [](NetIO& self, int timeout_ms) {
                 std::vector<Ctl> got;
                 {
                     nb::gil_scoped_release release;
                     got = self.wait_control(timeout_ms);
                 }
                 nb::list out;
                 for (auto& c : got)
                     out.append(nb::make_tuple(nb::str(c.kind.c_str()), c.conn,
                                               nb::bytes(reinterpret_cast<const char*>(c.frame.data()),
                                                         c.frame.size())));
                 return out;
             },
             nb::arg("timeout_ms"))
        .def("set_views", &NetIO::set_views, R())
        .def("send", [](NetIO& self, ConnId c, nb::bytes frame) {
                 std::string f(frame.c_str(), frame.size());
                 nb::gil_scoped_release release;
                 return self.send(c, std::move(f));
             })
        .def("sub_open", &NetIO::sub_open, nb::arg("conn"), nb::arg("sub"), nb::arg("view"), nb::arg("kind"),
             nb::arg("group"), nb::arg("data_view"), nb::arg("max_rate"), nb::arg("backfill"), R())
        .def("sub_close", &NetIO::sub_close, R())
        .def("series_drop", &NetIO::series_drop, R())
        .def("publish", &NetIO::publish, R())
        .def("series_append", &NetIO::series_append, R())
        .def("stats", [](NetIO& self) {
                 NetIO::Stats st;
                 {
                     nb::gil_scoped_release release;
                     st = self.collect();
                 }
                 return NetIO::to_dict(st);
             });
}
