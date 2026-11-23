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
//                             VIEW_REQUEST, EVENT and (3.1) CONTROL go to Python
//                             (control()), which answers with send() and
//                             sub_open()/sub_close().
//
// SCENES (wire protocol 3.2 draft, capability "scenes"). WELCOME and the HELLO
// to a hub announce "scenes"; Python (caliper/_net_scene.py) makes every scene
// message and hands it over with send(), one link at a time while queued()
// is low, and the views of kind scene leave the VIEWS of a peer that did not
// announce "scenes".
//
// CONTROLS (wire protocol 3.1). After enable_controls(), WELCOME, SESSIONS and
// the HELLO to a hub announce "controls" and "selection" (caliper._controls
// in Python implements them), WELCOME tells a viewer that announced "controls"
// whether it may control, and the "opened" message hands Python what a peer
// announced as JSON. A CONTROL from a peer that did not announce "controls" is
// refused here with ERROR bad_request. Python sends CONTROLS, CONTROL_REPLY
// and SELECTION with send(); a late viewer's state goes with
// send_after_views() so that it follows the VIEWS of its attachment, and
// data_pending() tells Python when the run state of a pause may follow the
// data sent before it. A peer
// that announced "selection" gets the VIEWS with item spaces (set_views()'s
// second list).
//
// TIMELINES (wire protocol 3.2 draft, capability "timelines"). WELCOME
// announces "timelines". To a peer that announced it, the VIEWS of an
// attachment declare the timelines `step` (sequence) and `time` (time, with an
// epoch), and each SNAPSHOT carries the section `stamp` i64 (2,): the run's
// step when the view call took it (or CALIPER_NET_STAMP_ABSENT) and the wall
// time in nanoseconds, which publish() takes from Python. A peer that did not
// announce it gets neither, as from the Python sender.
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

#if defined(__ARM_NEON)
#include <arm_neon.h>
#endif

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstring>
#include <deque>
#include <fstream>
#include <future>
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

// 3.2 draft: whether a caliper.watch(model, optimizer) is on (caliper._net_programs), so that a
// server with controls also announces "programs" and "interventions".
std::atomic<bool> g_watch_caps{false};

uint64_t now_ns() {
    return (uint64_t)std::chrono::duration_cast<std::chrono::nanoseconds>(
               std::chrono::steady_clock::now().time_since_epoch()).count();
}

constexpr size_t kRoom = 64 * 1024;          // send the next frame when less than this is queued
constexpr size_t kMaxControlPerConn = 256;   // messages for Python waiting per connection
// The control frames waiting to be sent on one connection (replies, state,
// errors; a PONG is kept apart, at most one). A peer that keeps sending while
// it reads nothing fills them; past either limit the connection is closed with
// 1008, as the hub closes a connection whose send queue overflows.
constexpr size_t kMaxControlFrames = 65536;
constexpr size_t kMaxControlBytes = 16u << 20;
// 3.2 draft: scene frames travel in the same queue, in order with the control
// frames, but they are the producer's data and not answers a peer provoked: Python
// hands a connection at most NATIVE_BACKLOG (2) of them waiting, each at most a
// frame of CALIPER_NET_MAX_FRAME_BYTES. They are counted apart, against this bound,
// so that a scene frame the protocol allows never closes the connection.
constexpr size_t kSceneBacklog = 2;
constexpr size_t kMaxSceneBytes = kSceneBacklog * size_t(CALIPER_NET_MAX_FRAME_BYTES);
constexpr uint16_t kCloseOverflow = 1008;
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
    Json cadence;                             // 3.2 draft: the views' cadence, for "programs" peers
    bool has_stamp = false;                   // 3.2 draft: the stamp (step, wall ns), for "timelines" peers
    int64_t stamp[2] = {CALIPER_NET_STAMP_ABSENT, 0};
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
    bool upstream = false;                    // the producer's own connection to a hub
    uint64_t opened_ms = 0;
    bool helloed = false;
    bool attached = false;
    bool follows_default = true;
    bool pinged = false;
    bool closing = false;
    uint32_t nonce = 0;
    std::deque<SharedBytes> control;          // frames before any data
    bool views_pending = false;
    bool cap_controls = false;                // 3.1: the peer announced "controls"
    bool cap_selection = false;               // 3.1: the peer announced "selection"
    bool cap_scenes = false;                  // 3.2 draft: the peer announced "scenes"
    bool cap_timelines = false;               // 3.2 draft: the peer announced "timelines"
    bool cap_programs = false;                // 3.2 draft: both sides announced "programs"
    bool cap_resources = false;               // 3.2 draft: "resources" beside "programs" (pieces)
    bool cap_interventions = false;           // 3.2 draft: both sides announced "interventions"
    bool may_control = false;                 // 3.1: a CONTROL from it is accepted
    std::deque<SharedBytes> after_views;      // 3.1: frames that follow the next VIEWS
    size_t control_bytes = 0;                 // the bytes in control and after_views, scene frames aside
    size_t scene_bytes = 0;                   // 3.2 draft: the bytes of the scene frames among them
    bool overflow = false;                    // past kMaxControlFrames/Bytes: closed with 1008
    bool pong_pending = false;                // one PONG at most, to the latest PING
    uint32_t pong_nonce = 0;
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

// Messages that could not be encoded (a frame over CALIPER_NET_MAX_FRAME_BYTES or
// its metadata over CALIPER_NET_MAX_META_BYTES): counted and not sent.
std::atomic<uint64_t> g_encode_failed{0};

SharedBytes encode(const wire::Message& m) {
    auto out = std::make_shared<caliper_net::Bytes>();
    if (!wire::encode(m, out.get())) {
        g_encode_failed.fetch_add(1, std::memory_order_relaxed);
        return nullptr;
    }
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
        opt_ = opt;
        server_ = std::make_unique<caliper_net::Server>(opt);
        if (port >= 0 && !server_->listen((uint16_t)port, &err)) throw std::runtime_error("listen: " + err);
        thread_ = std::thread([this] { loop(); });
    }

    ~NetIO() { close(); }

    uint16_t port() const { return server_->port(); }
    bool listening() const { return server_->listening(); }

    // Start listening later (a producer that connected to a hub first): done on
    // the native thread, which owns the server.
    uint16_t listen(int port) {
        std::promise<std::string> done;
        std::future<std::string> f = done.get_future();
        {
            std::lock_guard<std::mutex> g(mu_);
            // Once the native loop has exited nothing would answer the promise.
            if (loop_done_) throw std::runtime_error("listen: the server is closed");
            listen_port_ = port;
            listen_done_ = &done;
        }
        server_->wake();
        const std::string err = f.get();
        if (!err.empty()) throw std::runtime_error("listen: " + err);
        return server_->port();
    }

    // ---- the producer's connection to a hub (caliper.connect) -------------------------
    void connect_upstream(std::string url, std::string identity_json) {
        Json id;
        std::string err;
        if (!Json::parse(identity_json.data(), identity_json.size(), &id, &err) || !id.is_object())
            throw std::invalid_argument("identity: " + err);
        disconnect_upstream();
        {
            std::lock_guard<std::mutex> g(mu_);
            up_url_ = std::move(url);
            up_identity_ = std::move(id);
            up_state_ = "connecting";
            up_error_.clear();
            up_attempts_ = 0;
            up_replaced_ = false;
        }
        std::lock_guard<std::mutex> j(up_join_mu_);
        up_stop_ = false;
        up_thread_ = std::thread([this] { up_loop(); });
    }

    void disconnect_upstream() {
        std::lock_guard<std::mutex> j(up_join_mu_);    // two threads never join it at once
        if (!up_thread_.joinable()) return;
        up_stop_ = true;
        {
            std::lock_guard<std::mutex> g(mu_);
            if (client_) client_->wake();
        }
        up_cv_.notify_all();
        up_thread_.join();
    }

    struct UpState {
        std::string state, session, error, url;
        uint64_t attempts = 0, connects = 0;
        bool connected = false;
    };
    UpState upstream() {
        std::lock_guard<std::mutex> g(mu_);
        UpState u;
        u.state = up_state_;
        u.session = up_session_;
        u.error = up_error_;
        u.attempts = up_attempts_;
        u.connects = up_connects_;
        u.connected = up_state_ == "connected";
        u.url = up_url_;
        return u;
    }
    int attached() const { return attached_.load(std::memory_order_relaxed); }

    // ---- Python-thread calls (the GIL is released around them) ----------------------
    void close() {
        disconnect_upstream();
        // close() may come from two threads at once (the script's close() and
        // caliper.close() at exit, say): one of them joins, the other waits.
        std::lock_guard<std::mutex> j(join_mu_);
        stop_ = true;
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

    // views_json: the VIEWS list; selection_json (3.1, optional): the same list
    // with item spaces, for peers that announced "selection".
    void set_views(std::string views_json, std::string selection_json) {
        Json v, sv;
        std::string err;
        if (!Json::parse(views_json.data(), views_json.size(), &v, &err, 64) || !v.is_array())
            throw std::invalid_argument("views: " + err);
        if (!selection_json.empty() &&
            (!Json::parse(selection_json.data(), selection_json.size(), &sv, &err, 64) || !sv.is_array()))
            throw std::invalid_argument("selection views: " + err);
        std::lock_guard<std::mutex> g(mu_);
        views_ = std::move(v);
        views_sel_ = std::move(sv);
        views_plain_ = without_scenes(views_);           // 3.2 draft: for peers without "scenes"
        views_sel_plain_ = views_sel_.is_array() ? without_scenes(views_sel_) : Json();
        for (auto& [id, l] : links_)
            if (l.attached) l.views_pending = true;
        wake_all();
    }

    bool send(ConnId c, std::string frame) {
        std::lock_guard<std::mutex> g(mu_);
        auto it = links_.find(c);
        if (it == links_.end()) return false;
        const bool ok = push_control(it->second, std::make_shared<caliper_net::Bytes>(frame.begin(), frame.end()));
        wake_all();
        return ok;
    }

    // 3.1: as send(), but after the VIEWS that the connection waits for, if any
    // (the state a viewer receives right after the VIEWS of its attachment).
    bool send_after_views(ConnId c, std::string frame) {
        std::lock_guard<std::mutex> g(mu_);
        auto it = links_.find(c);
        if (it == links_.end()) return false;
        Link& l = it->second;
        const bool ok = push_control(l, std::make_shared<caliper_net::Bytes>(frame.begin(), frame.end()),
                                     l.views_pending);
        wake_all();
        return ok;
    }

    // 3.2 draft: the frames a connection still has waiting before its data (Python
    // hands it the next scene message only while this is low).
    size_t queued(ConnId c) {
        std::lock_guard<std::mutex> g(mu_);
        auto it = links_.find(c);
        return it == links_.end() ? 0 : it->second.control.size() + it->second.after_views.size();
    }

    // 3.2 draft: the bytes of the control frames that a connection still has waiting, which
    // bound the RESOURCE frames that Python gives a viewer that falls behind.
    size_t control_bytes(ConnId c) {
        std::lock_guard<std::mutex> g(mu_);
        auto it = links_.find(c);
        return it == links_.end() ? 0 : it->second.control_bytes;
    }

    static Json without_scenes(const Json& views) {
        Json out = Json::array();
        for (const Json& e : views.items()) {
            const Json* k = e.find("kind");
            int64_t kind = 0;
            if (!(k && k->get_i64(&kind) && kind == CALIPER_NET_KIND_SCENE)) out.push(e);
        }
        return out;
    }

    // 3.1: whether a connection still has a snapshot or series values to send
    // (Python sends the run state of a pause only once none has).
    bool data_pending() {
        std::lock_guard<std::mutex> g(mu_);
        for (const auto& [id, l] : links_) {
            if (l.closing) continue;
            for (const auto& [n, sub] : l.subs) {
                if (sub.snap) return true;
                if (!sub.series) continue;
                auto h = histories_.find(sub.view);
                if (h != histories_.end() && sub.next < h->second.total) return true;
            }
        }
        return false;
    }

    // 3.1: announce "controls" and "selection" from now on; ``allow``: whether
    // viewers may control (WELCOME's "control").
    void enable_controls(bool allow) {
        std::lock_guard<std::mutex> g(mu_);
        controls_ = true;
        allow_control_ = allow;
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
        wake_all();
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
                 uint64_t aux_ptr, size_t aux_n, std::string cadence_json, bool has_stamp, int64_t stamp_step,
                 int64_t stamp_wall_ns) {
        auto s = std::make_shared<Snap>();
        s->has_stamp = has_stamp;
        s->stamp[0] = stamp_step;
        s->stamp[1] = stamp_wall_ns;
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
        if (!cadence_json.empty()) Json::parse(cadence_json.data(), cadence_json.size(), &s->cadence, &err, 8);
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
        if (stats_ptr == 0) host_stats(*s);
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
        wake_all();
    }

    // The stats of a snapshot whose producer did not compute them on its device,
    // from the payload as sent (NaN left out; NaN when nothing is left):
    // heatmap [min, max]; points [aux min, aux max, x min, x max, y min, y max,
    // z min, z max] (z 0 for 2-D points). Run with the GIL released.
    static void host_stats(Snap& s) {
        const double nan = std::nan("");
        // On arm64 with NEON's minNum and maxNum, which leave a NaN out as the
        // scalar compares below do (a NaN compares false and never replaces a
        // bound); the scalar loops do the rest and the other targets.
        auto range = [&](const float* p, size_t n, double* lo, double* hi) {
            float a = INFINITY, b = -INFINITY;
            size_t i = 0;
#if defined(__ARM_NEON) && defined(__aarch64__)
            if (n >= 8) {
                float32x4_t a0 = vdupq_n_f32(INFINITY), a1 = a0, b0 = vdupq_n_f32(-INFINITY), b1 = b0;
                for (; i + 8 <= n; i += 8) {
                    const float32x4_t x = vld1q_f32(p + i), y = vld1q_f32(p + i + 4);
                    a0 = vminnmq_f32(a0, x);
                    b0 = vmaxnmq_f32(b0, x);
                    a1 = vminnmq_f32(a1, y);
                    b1 = vmaxnmq_f32(b1, y);
                }
                a = vminnmvq_f32(vminnmq_f32(a0, a1));
                b = vmaxnmvq_f32(vmaxnmq_f32(b0, b1));
            }
#endif
            for (; i < n; ++i) {
                const float v = p[i];
                a = v < a ? v : a;
                b = v > b ? v : b;
            }
            *lo = a <= b ? (double)a : nan;
            *hi = a <= b ? (double)b : nan;
        };
        const float* f = reinterpret_cast<const float*>(s.data.data());
        if (s.kind == CALIPER_NET_KIND_HEATMAP && s.dtype == CALIPER_NET_DTYPE_F32) {
            s.stats.assign(2, nan);
            range(f, s.data.size() / 4, &s.stats[0], &s.stats[1]);
        } else if (s.kind == CALIPER_NET_KIND_POINTS && s.dtype == CALIPER_NET_DTYPE_F32 && s.shape.size() == 2) {
            const size_t n = (size_t)s.shape[0], d = (size_t)s.shape[1];
            s.stats.assign(8, 0.0);
            if (s.has_aux) range(reinterpret_cast<const float*>(s.aux.data()), s.aux.size() / 4,
                                 &s.stats[0], &s.stats[1]);
            if (d < 1 || d > 3 || n * d * 4 > s.data.size()) return;
            // one pass over the interleaved coordinates
            float a[3] = {INFINITY, INFINITY, INFINITY}, b[3] = {-INFINITY, -INFINITY, -INFINITY};
            size_t i = 0;
#if defined(__ARM_NEON) && defined(__aarch64__)
            if (d == 3 || d == 2) {
                const float32x4_t inf = vdupq_n_f32(INFINITY), ninf = vdupq_n_f32(-INFINITY);
                float32x4_t lx = inf, ly = inf, lz = inf, hx = ninf, hy = ninf, hz = ninf;
                if (d == 3) {
                    for (; i + 4 <= n; i += 4) {
                        const float32x4x3_t v = vld3q_f32(f + i * 3);
                        lx = vminnmq_f32(lx, v.val[0]);
                        hx = vmaxnmq_f32(hx, v.val[0]);
                        ly = vminnmq_f32(ly, v.val[1]);
                        hy = vmaxnmq_f32(hy, v.val[1]);
                        lz = vminnmq_f32(lz, v.val[2]);
                        hz = vmaxnmq_f32(hz, v.val[2]);
                    }
                    a[2] = vminnmvq_f32(lz);
                    b[2] = vmaxnmvq_f32(hz);
                } else {
                    for (; i + 4 <= n; i += 4) {
                        const float32x4x2_t v = vld2q_f32(f + i * 2);
                        lx = vminnmq_f32(lx, v.val[0]);
                        hx = vmaxnmq_f32(hx, v.val[0]);
                        ly = vminnmq_f32(ly, v.val[1]);
                        hy = vmaxnmq_f32(hy, v.val[1]);
                    }
                }
                a[0] = vminnmvq_f32(lx);
                b[0] = vmaxnmvq_f32(hx);
                a[1] = vminnmvq_f32(ly);
                b[1] = vmaxnmvq_f32(hy);
            }
#endif
            for (; i < n; ++i) {
                const float* q = f + i * d;
                for (size_t k = 0; k < d; ++k) {
                    a[k] = q[k] < a[k] ? q[k] : a[k];
                    b[k] = q[k] > b[k] ? q[k] : b[k];
                }
            }
            for (size_t k = 0; k < d; ++k) {
                s.stats[2 + 2 * k] = a[k] <= b[k] ? (double)a[k] : nan;
                s.stats[3 + 2 * k] = a[k] <= b[k] ? (double)b[k] : nan;
            }
        }
    }

    void series_append(const std::string& view, uint64_t ptr, size_t n, uint64_t stamp_ns) {
        std::lock_guard<std::mutex> g(mu_);
        History& h = histories_[view];
        h.retain = retain_;
        h.append(reinterpret_cast<const float*>(ptr), n, stamp_ns);
        series_values_ += n;
        wake_all();
    }

    struct Stats {
        std::vector<std::tuple<uint64_t, bool, size_t, uint64_t, uint64_t, uint64_t, uint64_t, uint64_t>> ls;
        std::vector<std::pair<size_t, size_t>> queued;    // per connection: control frames and bytes
        uint64_t published = 0, http = 0, refused = 0, values = 0, overflow = 0;
        std::map<std::string, uint64_t> totals;
    };
    // Called with the GIL released; to_dict() runs with it.
    Stats collect() {
        Stats st;
        std::lock_guard<std::mutex> g(mu_);
        for (auto& [id, l] : links_)
        {
            st.ls.emplace_back(id, l.attached, l.subs.size(), l.frames_out, l.bytes_out, l.superseded,
                               l.snapshots_sent, l.series_sent);
            st.queued.emplace_back(l.control.size() + l.after_views.size(), l.control_bytes);
        }
        st.overflow = overflow_closed_;
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
        size_t k = 0;
        for (auto& t : ls) {
            nb::dict c;
            c["control_frames"] = st.queued[k].first;
            c["control_bytes"] = st.queued[k].second;
            ++k;
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
        d["overflow_closed"] = st.overflow;
        d["encode_failed"] = g_encode_failed.load(std::memory_order_relaxed);
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
            if (listen_done_) {
                std::string err;
                if (!server_->listening() && !server_->listen((uint16_t)listen_port_, &err) && err.empty())
                    err = "listen failed";
                listen_done_->set_value(err);
                listen_done_ = nullptr;
            }
            for (auto& e : evs) handle(e);
            timers();
            for (auto& [id, l] : links_)
                if (!l.upstream) pump(l);
        }
        std::vector<ConnId> open;
        {
            std::lock_guard<std::mutex> g(mu_);
            wire::Bye bye;
            bye.reason = CALIPER_NET_BYE_SHUTDOWN;
            bye.message = "the producer is closing";
            SharedBytes b = encode(bye);
            for (auto& [id, l] : links_) {
                if (l.upstream) continue;
                // what is queued for it goes first, then BYE
                pump(l, /*final=*/true);
                if (b && server_->is_open(id)) server_->send(id, b);
            }
        }
        server_->shutdown(CALIPER_NET_CLOSE_GOING_AWAY, 2000);
        std::lock_guard<std::mutex> g(mu_);
        loop_done_ = true;                    // a later listen() fails at once
        if (listen_done_) {
            listen_done_->set_value("closed");
            listen_done_ = nullptr;
        }
        for (auto it = links_.begin(); it != links_.end();) {
            if (it->second.upstream) {
                ++it;
                continue;
            }
            push_ctl("closed", it->first);
            if (it->second.attached) attached_.fetch_sub(1);
            it = links_.erase(it);
        }
    }

    int next_timeout_ms() {
        std::lock_guard<std::mutex> g(mu_);
        if (links_.empty()) return kMaxPollMs * 4;
        const uint64_t now = now_ns();
        uint64_t wait = (uint64_t)kMaxPollMs * 1'000'000;
        for (auto& [id, l] : links_) {
            if (l.upstream || server_->queued_bytes(id) >= kRoom) continue;
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

    // The socket operations of a link: the server's connections, or the
    // client of the hub connection (touched only by the thread that owns it).
    bool xsend(Link& l, const SharedBytes& f) {
        return l.upstream ? (client_ && client_->send(f)) : server_->send(l.id, f);
    }
    void xclose(Link& l, uint16_t code, const std::string& reason) {
        if (!l.upstream) server_->close(l.id, code, reason);
        else if (client_) client_->close(code, reason);
        l.closing = true;
    }
    size_t xqueued(const Link& l) const {
        return l.upstream ? (client_ ? client_->queued_bytes() : 0) : server_->queued_bytes(l.id);
    }
    bool xopen(const Link& l) const {
        return l.upstream ? (client_ && client_->state() == caliper_net::Client::State::Open)
                          : server_->is_open(l.id);
    }
    uint64_t xlast_input(const Link& l) const {
        return l.upstream ? (client_ ? client_->last_input_ms() : 0) : server_->last_input_ms(l.id);
    }
    void wake_all() {                         // under mu_
        server_->wake();
        if (client_) client_->wake();
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

    void queue(Link& l, const wire::Message& m) {
        if (SharedBytes b = encode(m)) push_control(l, std::move(b));
    }

    // A SCENE_UPDATE or SCENE_CLEAR frame (the type at offset 8 of the header).
    static bool scene_frame(const caliper_net::Bytes& b) {
        if (b.size() < 10) return false;
        const uint16_t type = uint16_t(b[8] | (b[9] << 8));
        return type == CALIPER_NET_MSG_SCENE_UPDATE || type == CALIPER_NET_MSG_SCENE_CLEAR;
    }

    // Queue a control frame (under mu_). Past kMaxControlFrames, kMaxControlBytes
    // or, for scene frames, kMaxSceneBytes the queue is dropped and the connection
    // marked, and the thread that owns its socket closes it with 1008 at its next pump.
    bool push_control(Link& l, SharedBytes b, bool after_views = false) {
        if (l.overflow || l.closing) return false;
        const bool scene = scene_frame(*b);
        size_t& bytes = scene ? l.scene_bytes : l.control_bytes;
        if (l.control.size() + l.after_views.size() >= kMaxControlFrames ||
            bytes + b->size() > (scene ? kMaxSceneBytes : kMaxControlBytes)) {
            l.overflow = true;
            ++overflow_closed_;
            l.control.clear();
            l.after_views.clear();
            l.control_bytes = 0;
            l.scene_bytes = 0;
            wake_all();
            return false;
        }
        bytes += b->size();
        (after_views ? l.after_views : l.control).push_back(std::move(b));
        return true;
    }

    // 3.1 (under mu_): the server's capabilities, and what a peer announced as
    // the JSON of its "opened" message for Python.
    std::vector<std::string> capabilities() const {
        if (!controls_) return {CALIPER_NET_CAP_EVENTS, CALIPER_NET_CAP_SCENES, CALIPER_NET_CAP_TIMELINES};
        if (g_watch_caps.load())
            return {CALIPER_NET_CAP_CONTROLS,  CALIPER_NET_CAP_EVENTS, CALIPER_NET_CAP_INTERVENTIONS,
                    CALIPER_NET_CAP_PROGRAMS,  CALIPER_NET_CAP_RESOURCES, CALIPER_NET_CAP_SCENES,
                    CALIPER_NET_CAP_SELECTION, CALIPER_NET_CAP_TIMELINES};
        return {CALIPER_NET_CAP_CONTROLS, CALIPER_NET_CAP_EVENTS, CALIPER_NET_CAP_SCENES,
                CALIPER_NET_CAP_SELECTION, CALIPER_NET_CAP_TIMELINES};
    }
    // 3.2 draft: the timelines of the VIEWS to a peer that announced "timelines", as the
    // Python sender declares them (caliper/_net.py, views_msg()).
    static const Json& timelines() {
        static const Json t = [] {
            const std::string text =
                "[{\"kind\":\"sequence\",\"name\":\"step\"},{\"epoch\":true,\"kind\":\"time\",\"name\":\"time\"}]";
            Json v;
            std::string err;
            Json::parse(text.data(), text.size(), &v, &err, 8);
            return v;
        }();
        return t;
    }
    static bool announces(const std::vector<std::string>& caps, const char* cap) {
        return std::find(caps.begin(), caps.end(), cap) != caps.end();
    }
    static std::vector<uint8_t> link_info(const Link& l) {
        Json o = Json::object();
        o.set("controls", l.cap_controls);
        o.set("selection", l.cap_selection);
        o.set("scenes", l.cap_scenes);
        o.set("programs", l.cap_programs);
        o.set("resources", l.cap_resources);
        o.set("interventions", l.cap_interventions);
        o.set("control", l.may_control);
        o.set("upstream", l.upstream);
        const std::string t = o.dump();
        return std::vector<uint8_t>(t.begin(), t.end());
    }

    void fatal(Link& l, const std::string& code, const std::string& msg, uint16_t close_code) {
        wire::Error e;
        e.code = code;
        e.message = msg;
        e.fatal = true;
        if (SharedBytes b = encode(e)) xsend(l, b);
        xclose(l, close_code, msg.substr(0, 120));
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
            xclose(l, CALIPER_NET_CLOSE_UNSUPPORTED_DATA, "binary frames only");
            return;
        }
        l.pinged = false;
        wire::Message m;
        const wire::DecodeResult r = wire::decode(e.data.data(), e.data.size(), &m);
        if (r.status == wire::DecodeResult::Status::BadFrame) {
            xclose(l, (uint16_t)r.close_code, r.reason);
            return;
        }
        if (!l.helloed) {
            if (r.status == wire::DecodeResult::Status::BadMessage) {
                fatal(l, r.error_code, r.message, CALIPER_NET_CLOSE_PROTOCOL_ERROR);
                return;
            }
            if (std::holds_alternative<wire::Bye>(m)) {
                xclose(l, CALIPER_NET_CLOSE_NORMAL, "");
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
            l.cap_controls = controls_ && announces(h->capabilities, CALIPER_NET_CAP_CONTROLS);
            l.cap_selection = controls_ && announces(h->capabilities, CALIPER_NET_CAP_SELECTION);
            l.cap_scenes = announces(h->capabilities, CALIPER_NET_CAP_SCENES);
            l.cap_timelines = announces(h->capabilities, CALIPER_NET_CAP_TIMELINES);
            l.cap_programs = controls_ && g_watch_caps.load() && announces(h->capabilities, CALIPER_NET_CAP_PROGRAMS);
            l.cap_resources = l.cap_programs && announces(h->capabilities, CALIPER_NET_CAP_RESOURCES);
            l.cap_interventions =
                controls_ && g_watch_caps.load() && announces(h->capabilities, CALIPER_NET_CAP_INTERVENTIONS);
            l.may_control = l.cap_controls && allow_control_;
            wire::Welcome w;
            w.role = CALIPER_NET_ROLE_PRODUCER;
            w.capabilities = capabilities();
            w.software = software_;
            w.session = identity_;
            if (l.cap_controls) w.control = allow_control_;
            queue(l, w);
            wire::Sessions s;
            Json entry = identity_;
            entry.set("state", CALIPER_NET_SESSION_LIVE);
            entry.set("views", (uint64_t)views_.size());
            Json caps = Json::array();
            for (const auto& c : capabilities()) caps.push(c);
            entry.set("capabilities", caps);
            s.sessions.push(entry);
            queue(l, s);
            push_ctl("opened", l.id, link_info(l));   // before "attached": Python makes its state first
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
            l.pong_pending = true;                // a PING flood leaves one PONG, to the latest
            l.pong_nonce = p->nonce;
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
        } else if (auto* bye = std::get_if<wire::Bye>(&m)) {
            if (l.upstream && bye->reason == CALIPER_NET_BYE_REPLACED) up_replaced_ = true;
            xclose(l, CALIPER_NET_CLOSE_NORMAL, "bye");
        } else if (auto* er = std::get_if<wire::Error>(&m)) {
            if (er->fatal) xclose(l, CALIPER_NET_CLOSE_NORMAL, "peer error");
        } else if (std::holds_alternative<wire::Subscribe>(m) || std::holds_alternative<wire::Unsubscribe>(m) ||
                   std::holds_alternative<wire::ViewRequest>(m) || std::holds_alternative<wire::Event>(m)) {
            to_python(l, m, e.data);
        } else if (auto* c = std::get_if<wire::Control>(&m)) {
            if (l.cap_controls) {
                to_python(l, m, e.data);
            } else {                          // the protocol's rule for a peer without "controls"
                wire::Error err;
                err.code = CALIPER_NET_ERR_BAD_REQUEST;
                err.message = "a CONTROL needs the capability controls in HELLO";
                err.in_reply_to = CALIPER_NET_MSG_CONTROL;
                err.request_id = c->request_id;
                queue(l, err);
            }
        } else if (auto* pg = std::get_if<wire::Program>(&m)) {   // 3.2 draft
            if (l.cap_programs) {
                to_python(l, m, e.data);
            } else {
                wire::Error err;
                err.code = CALIPER_NET_ERR_UNSUPPORTED;
                err.message = "a PROGRAM needs the capability programs in HELLO and caliper.watch(model, optimizer)";
                err.in_reply_to = CALIPER_NET_MSG_PROGRAM;
                err.request_id = pg->request_id;
                queue(l, err);
            }
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
            } else if (auto* c = std::get_if<wire::Control>(&m)) {
                wire::ControlReply rep;
                rep.request_id = c->request_id;
                rep.op = c->op;
                rep.status = CALIPER_NET_STATUS_ERROR;
                rep.name = c->name;
                Json err = Json::object();
                err.set("code", CALIPER_NET_ERR_LIMIT);
                err.set("message", "too many requests waiting");
                rep.error = err;
                queue(l, rep);
            } else if (auto* pg = std::get_if<wire::Program>(&m)) {
                wire::Error err;
                err.code = CALIPER_NET_ERR_LIMIT;
                err.message = "too many requests waiting";
                err.in_reply_to = CALIPER_NET_MSG_PROGRAM;
                err.request_id = pg->request_id;
                queue(l, err);
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
        for (auto& [id, l] : links_)
            if (!l.upstream) timer(l, now);
    }

    void timer(Link& l, uint64_t now) {
        if (l.closing) return;
        if (!l.helloed) {
            if (now - l.opened_ms >= (uint64_t)handshake_ms_)
                xclose(l, CALIPER_NET_CLOSE_HANDSHAKE_TIMEOUT, "no hello");
            return;
        }
        const uint64_t idle = now - xlast_input(l);
        if (idle >= (uint64_t)idle_ms_) {
            xclose(l, CALIPER_NET_CLOSE_IDLE_TIMEOUT, "idle");
        } else if (idle >= (uint64_t)ping_ms_ && !l.pinged) {
            queue(l, wire::Ping{++l.nonce});
            l.pinged = true;
        }
    }

    // Send what the connection has room for: control frames, then series,
    // then snapshots, each subscription at its own rate.
    void pump(Link& l, bool final = false) {
        if (l.overflow && !l.closing) {
            xclose(l, kCloseOverflow, "control queue full");
            return;
        }
        if (l.closing || !xopen(l)) return;
        const uint64_t now = now_ns();
        while (final || xqueued(l) < kRoom) {
            SharedBytes f;
            if (l.pong_pending) {
                f = encode(wire::Pong{l.pong_nonce});
                l.pong_pending = false;
            } else if (l.views_pending && l.control.empty()) {
                wire::Views v;
                if (l.attached) {
                    v.session = l.upstream && !up_session_.empty() ? up_session_ : ident_.id;
                    if (l.cap_scenes) v.views = l.cap_selection && views_sel_.is_array() ? views_sel_ : views_;
                    else v.views = l.cap_selection && views_sel_.is_array() ? views_sel_plain_ : views_plain_;
                    if (l.cap_timelines) v.timelines = timelines();   // 3.2 draft
                }
                f = encode(v);
                l.views_pending = false;
                l.control.swap(l.after_views);    // the state that follows the VIEWS
            } else if (!l.control.empty()) {
                f = l.control.front();
                l.control.pop_front();
                size_t& bytes = scene_frame(*f) ? l.scene_bytes : l.control_bytes;
                bytes -= std::min(bytes, f->size());
            } else {
                f = next_data(l, now, final);
            }
            if (!f) break;
            if (!xsend(l, f)) break;
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
        for (;;) {                                // a snapshot that cannot be encoded is dropped
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
            if (l.cap_programs && sn->cadence.is_object()) m.cadence = sn->cadence;
            if (l.cap_timelines && sn->has_stamp)   // 3.2 draft: i64 (2,), step then wall ns
                m.stamp = wire::Section::make("stamp", CALIPER_NET_DTYPE_I64, {2}, sn->stamp, sizeof sn->stamp);
            if (SharedBytes f = encode(m)) return f;
            --l.snapshots_sent;
        }
    }

    // ---- the hub connection's thread ------------------------------------------------------
    // Connect, HELLO as a producer, WELCOME from the hub; then the connection is
    // one more link (the hub is this session's one viewer). On a loss it waits
    // (0.5 s, doubling to 30 s) and connects again, proposing the session id the
    // hub assigned, so the hub resumes the session. BYE `replaced` ends it.
    void up_loop() {
        int backoff_ms = 500;
        while (!up_stop_.load()) {
            std::string url;
            Json hello_id;
            {
                std::lock_guard<std::mutex> g(mu_);
                client_ = std::make_unique<caliper_net::Client>(opt_);
                url = up_url_;
                hello_id = up_identity_;
                if (!up_session_.empty()) hello_id.set("id", up_session_);
                ++up_attempts_;
                up_state_ = "connecting";
            }
            std::string err;
            bool ok = client_->connect(url, &err);
            const ConnId cid = next_up_id_++;
            const uint64_t t0 = caliper_net::monotonic_ms();
            bool helloed = false, closed = !ok;
            while (ok && !closed && !up_stop_.load()) {
                int timeout = kMaxPollMs;
                std::vector<caliper_net::Event> evs = client_->poll(timeout);
                std::lock_guard<std::mutex> g(mu_);
                for (auto& e : evs) {
                    using T = caliper_net::Event::Type;
                    if (e.type == T::Open) {
                        wire::Hello h;
                        h.role = CALIPER_NET_ROLE_PRODUCER;
                        h.capabilities = capabilities();
                        h.software = software_;
                        h.session = hello_id;
                        if (SharedBytes b = encode(h)) client_->send(b);
                    } else if (e.type == T::Message && !helloed) {
                        wire::Message m;
                        const wire::DecodeResult r = wire::decode(e.data.data(), e.data.size(), &m);
                        auto* w = r.ok() ? std::get_if<wire::Welcome>(&m) : nullptr;
                        wire::Identity got;
                        if (!w || w->role != CALIPER_NET_ROLE_HUB || !wire::Identity::parse(w->session, &got)) {
                            err = "the hub's first message was not a WELCOME with a session";
                            client_->close(CALIPER_NET_CLOSE_PROTOCOL_ERROR, "no welcome");
                            closed = true;
                            break;
                        }
                        helloed = true;
                        up_session_ = got.id;
                        up_state_ = "connected";
                        ++up_connects_;
                        backoff_ms = 500;
                        Link l;
                        l.id = cid;
                        l.upstream = true;
                        l.helloed = true;
                        l.follows_default = false;
                        l.opened_ms = caliper_net::monotonic_ms();
                        l.cap_controls = controls_ && announces(w->capabilities, CALIPER_NET_CAP_CONTROLS);
                        l.cap_selection = controls_ && announces(w->capabilities, CALIPER_NET_CAP_SELECTION);
                        l.cap_scenes = announces(w->capabilities, CALIPER_NET_CAP_SCENES);
                        l.cap_timelines = announces(w->capabilities, CALIPER_NET_CAP_TIMELINES);
                        l.cap_programs = controls_ && g_watch_caps.load() &&
                                         announces(w->capabilities, CALIPER_NET_CAP_PROGRAMS);
                        l.cap_resources = l.cap_programs && announces(w->capabilities, CALIPER_NET_CAP_RESOURCES);
                        l.cap_interventions = controls_ && g_watch_caps.load() &&
                                              announces(w->capabilities, CALIPER_NET_CAP_INTERVENTIONS);
                        l.may_control = l.cap_controls && w->control.value_or(false);
                        std::vector<uint8_t> info = link_info(l);
                        links_[cid] = std::move(l);
                        push_ctl("opened", cid, std::move(info));
                        set_attached(links_[cid], true);
                        links_[cid].views_pending = true;
                    } else if (e.type == T::Message) {
                        auto it = links_.find(cid);
                        if (it != links_.end()) on_message(it->second, e);
                    } else if (e.type == T::Closed) {
                        closed = true;
                        err = e.reason.empty() ? "closed with " + std::to_string(e.code) : e.reason;
                    }
                }
                if (closed) break;
                if (!helloed && caliper_net::monotonic_ms() - t0 >= (uint64_t)handshake_ms_) {
                    err = "no WELCOME from the hub";
                    client_->close(CALIPER_NET_CLOSE_HANDSHAKE_TIMEOUT, "no welcome");
                    closed = true;
                    break;
                }
                auto it = links_.find(cid);
                if (it != links_.end()) {
                    timer(it->second, caliper_net::monotonic_ms());
                    pump(it->second);
                }
            }
            if (up_stop_.load() && client_ && client_->state() == caliper_net::Client::State::Open) {
                std::lock_guard<std::mutex> g(mu_);
                auto it = links_.find(cid);
                if (it != links_.end()) pump(it->second, /*final=*/true);
                wire::Bye bye;
                bye.reason = CALIPER_NET_BYE_SHUTDOWN;
                bye.message = "the producer is closing";
                if (SharedBytes b = encode(bye)) client_->send(b);
                client_->close(CALIPER_NET_CLOSE_GOING_AWAY, "shutdown");
            }
            if (up_stop_.load()) {                // let the close go out, briefly
                const uint64_t end = caliper_net::monotonic_ms() + 1000;
                while (client_->state() != caliper_net::Client::State::Closed &&
                       caliper_net::monotonic_ms() < end)
                    client_->poll(20);
            }
            {
                std::lock_guard<std::mutex> g(mu_);
                auto it = links_.find(cid);
                if (it != links_.end()) {
                    for (auto sub = it->second.subs.begin(); sub != it->second.subs.end();)
                        drop_sub(it->second, (sub++)->first);
                    if (it->second.attached) attached_.fetch_sub(1);
                    push_ctl("closed", cid);
                    links_.erase(it);
                }
                client_.reset();
                if (!err.empty()) up_error_ = err;
                up_state_ = up_stop_.load() ? "closed" : up_replaced_ ? "replaced" : "waiting";
            }
            if (up_stop_.load() || up_replaced_) break;
            std::unique_lock<std::mutex> lk(up_mu_);
            up_cv_.wait_for(lk, std::chrono::milliseconds(backoff_ms), [this] { return up_stop_.load(); });
            backoff_ms = std::min(backoff_ms * 2, 30000);
        }
    }

    caliper_net::Options opt_;
    std::unique_ptr<caliper_net::Client> client_;   // the hub connection (its thread; pointer under mu_)
    std::thread up_thread_;
    std::atomic<bool> up_stop_{false};
    std::atomic<bool> up_replaced_{false};
    std::mutex up_mu_;
    std::condition_variable up_cv_;
    std::string up_url_, up_state_ = "closed", up_error_, up_session_;
    Json up_identity_;
    uint64_t up_attempts_ = 0, up_connects_ = 0;
    ConnId next_up_id_ = ConnId(1) << 62;     // never a server connection id
    int listen_port_ = 0;
    std::promise<std::string>* listen_done_ = nullptr;
    bool loop_done_ = false;                  // the native loop has exited (under mu_)
    std::mutex join_mu_, up_join_mu_;         // one thread at a time joins thread_, up_thread_

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
    Json views_sel_;                          // 3.1: views_ with item spaces (an array), or null
    Json views_plain_ = Json::array();        // 3.2 draft: views_ without the scenes
    Json views_sel_plain_;                    // 3.2 draft: views_sel_ without the scenes, or null
    bool controls_ = false;                   // 3.1: announce "controls" and "selection"
    bool allow_control_ = false;              // 3.1: viewers may control
    std::vector<float> scratch_;
    uint64_t published_ = 0, http_requests_ = 0, refused_ = 0, series_values_ = 0;
    uint64_t overflow_closed_ = 0;            // connections closed for a full control queue

    std::mutex cmu_;                          // the queue to Python
    std::condition_variable ccv_;
    std::deque<Ctl> ctl_;
    bool ctl_wake_ = false;
};

}  // namespace

NB_MODULE(_netio, m) {
    m.doc() = "The native sender of caliper.serve() (wire protocol v3); internal.";
    m.attr("MAX_CONTROL_PER_CONNECTION") = kMaxControlPerConn;
    m.attr("MAX_CONTROL_FRAMES") = kMaxControlFrames;
    m.attr("MAX_CONTROL_BYTES") = kMaxControlBytes;
    m.attr("MAX_SCENE_BYTES") = kMaxSceneBytes;
    m.def("set_watch_capabilities", [](bool on) { g_watch_caps.store(on); }, nb::arg("on"),
          "Whether a caliper.watch(model, optimizer) is on, so that a server with controls announces "
          "programs and interventions to the peers that connect from now on.");
    using R = nb::call_guard<nb::gil_scoped_release>;
    nb::class_<NetIO>(m, "Server")
        .def(nb::init<int, std::string, std::string, std::string,
                      std::vector<std::tuple<std::string, std::string, std::string>>, int, size_t, int, int,
                      int>(),
             nb::arg("port"), nb::arg("token"), nb::arg("identity"), nb::arg("software"), nb::arg("assets"),
             nb::arg("max_connections"), nb::arg("series_retain"), nb::arg("handshake_ms"),
             nb::arg("ping_ms"), nb::arg("idle_ms"), R())
        .def_prop_ro("port", &NetIO::port)
        .def_prop_ro("listening", &NetIO::listening)
        .def("listen", &NetIO::listen, R())
        .def("connect_upstream", &NetIO::connect_upstream, R())
        .def("disconnect_upstream", &NetIO::disconnect_upstream, R())
        .def("upstream", [](NetIO& self) {
                 NetIO::UpState u;
                 {
                     nb::gil_scoped_release release;
                     u = self.upstream();
                 }
                 nb::dict d;
                 d["state"] = u.state;
                 d["session"] = u.session;
                 d["error"] = u.error;
                 d["attempts"] = u.attempts;
                 d["connects"] = u.connects;
                 d["connected"] = u.connected;
                 d["url"] = u.url;
                 return d;
             })
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
        .def("set_views", &NetIO::set_views, nb::arg("views"), nb::arg("selection_views") = "", R())
        .def("send", [](NetIO& self, ConnId c, nb::bytes frame) {
                 std::string f(frame.c_str(), frame.size());
                 nb::gil_scoped_release release;
                 return self.send(c, std::move(f));
             })
        .def("send_after_views", [](NetIO& self, ConnId c, nb::bytes frame) {
                 std::string f(frame.c_str(), frame.size());
                 nb::gil_scoped_release release;
                 return self.send_after_views(c, std::move(f));
             })
        .def("enable_controls", &NetIO::enable_controls, nb::arg("allow"), R())
        .def("data_pending", &NetIO::data_pending, R())
        .def("queued", &NetIO::queued, R())
        .def("control_bytes", &NetIO::control_bytes, nb::arg("conn"), R())
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
