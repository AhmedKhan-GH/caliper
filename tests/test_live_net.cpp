// caliper_live_tests, network part: the native viewer of wire protocol v3
// (src/host/live/live_net_source.h, caliper_core_live_attach_url) end to end in
// ONE process. A fake producer (the WebSocket server of net/ws.h and the
// encoder of net/wire.h, on its own thread, behaving as caliper.serve() does)
// streams a heatmap, an image, a points view, a series and a histogram; an
// OFFSCREEN embed core attaches by URL and the caliper_core_live_view_* probes
// and texture readbacks check what it drew: heatmap pixels against the CPU
// colormap reference, image bytes, points in their palette colours, series and
// histogram values. GPU cases skip without a device (gpu_skip.h).
#include <doctest/doctest.h>
#include "gpu_skip.h"

#include <caliper/embed.h>
#include <caliper/live.h>
#include <caliper/live_views.h>

#include "core_lifecycle.h"            // the app-like host of the Attachment case
#include "host_services.h"
#include "embed/embed_core.h"        // caliper_live_controls, the views' event sink
#include "embed/render_stats.h"      // copy counters (white-box)
#include "live/live_attach.h"
#include "live/live_controls.h"
#include "live/live_ui_driver.h"
#include "live/live_net_source.h"      // parse_stream_url
#include "live/live_scene.h"           // the scene view (protocol 3.2 draft)
#include "live/live_views_scene.h"
#include "live/live_views_state.h"     // kPalette (the exact categorical colours)
#include "live/scene_queue.h"          // what a source keeps of a scene while the display is paused
#include "net/wire.h"
#include "net/ws.h"
#include "tensor_bridge.h"             // map_f32_to_rgba8 / colormap_lut (CPU reference)

#include <imgui.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstring>
#include <deque>
#include <functional>
#include <fstream>
#include <future>
#include <map>
#include <tuple>
#include <memory>
#include <mutex>
#include <optional>
#include <set>
#include <string>
#include <thread>
#include <vector>

namespace wire = caliper_net::wire;
using caliper_net::Json;

namespace {

// ---- the fake producer -------------------------------------------------------------------
// One thread owns the server; the test hands it work through call(), which
// runs a function on that thread and returns its result.
class FakeProducer {
public:
    std::string token = "fake_token-0123456789abcdef";
    wire::Identity id{"clp-test-4242", "fake_train.py", 4242, 1763200000000ull};
    Json views = Json::array();
    bool auto_attach = true;            // as caliper.serve(): attach a viewer to its one session

    // Producer-thread state (read through call()).
    caliper_net::ConnId conn = 0;
    int hellos = 0;
    std::vector<std::string> hello_caps;      // the capabilities of the last HELLO
    std::map<std::string, uint32_t> sub_of;   // view -> its open subscription
    std::map<std::string, int> subscribes;    // view -> SUBSCRIBEs received for it
    std::vector<uint32_t> unsubscribed;       // the subs of the UNSUBSCRIBEs received
    std::vector<float> series;                // the retained values of "loss"
    std::vector<std::string> errors;          // ERROR codes the viewer sent
    std::vector<uint16_t> closes;             // close codes of ended connections
    std::vector<std::string> byes;            // BYE reasons the viewer sent
    std::vector<uint32_t> pongs;              // nonces of the viewer's PONGs
    uint64_t seq = 0;

    // ---- 3.1 ------------------------------------------------------------------
    std::vector<std::string> welcome_caps;    // WELCOME's capabilities
    std::optional<bool> welcome_control;      // WELCOME's control
    // As a script that calls a checkpoint and registered "lr" and "save": its
    // views name item spaces ("samples": the rows of "heat" and the points of
    // "pts"; "step": the series "loss"), CONTROLS follows VIEWS, a CONTROL is
    // answered and applied at once, and a select event with a space becomes
    // that space's SELECTION, with details.
    bool script31 = false;
    std::string run_state = CALIPER_NET_RUN_RUNNING;
    double lr = 0.001;
    int saves = 0;
    std::vector<wire::Control> controls_in;   // every CONTROL received
    std::vector<wire::Event> events;          // every EVENT received
    std::vector<wire::ViewRequest> view_requests;   // every VIEW_REQUEST received

    FakeProducer() {
        std::string err;
        REQUIRE(server_.listen(0, &err));
        add_view(1, "heat", CALIPER_NET_KIND_HEATMAP, {{"rows", 4}, {"cols", 8}});
        add_view(2, "img", CALIPER_NET_KIND_IMAGE, {{"h", 4}, {"w", 4}});
        add_view(3, "pts", CALIPER_NET_KIND_POINTS, {{"point", 6}});
        add_view(4, "loss", CALIPER_NET_KIND_SERIES, {});
        add_view(5, "hist", CALIPER_NET_KIND_HISTOGRAM, {{"x", 100}});
    }
    void start() { thread_ = std::thread([this] { loop(); }); }
    ~FakeProducer() {
        stop_ = true;
        server_.wake();
        if (thread_.joinable()) thread_.join();
    }

    uint16_t port() const { return server_.port(); }
    std::string url() const {
        return "ws://127.0.0.1:" + std::to_string(port()) + "/v3/stream?token=" + token;
    }

    template <typename F>
    auto call(F f) -> decltype(f(*this)) {
        using R = decltype(f(*this));
        auto task = std::make_shared<std::packaged_task<R()>>([this, f]() mutable { return f(*this); });
        std::future<R> fut = task->get_future();
        {
            std::lock_guard<std::mutex> g(mu_);
            tasks_.push_back([task] { (*task)(); });
        }
        server_.wake();
        return fut.get();
    }

    // ---- producer-thread actions (inside call()) ----------------------------------------
    void send(const wire::Message& m) {
        std::vector<uint8_t> b;
        REQUIRE(wire::encode(m, &b));
        if (conn) server_.send(conn, b.data(), b.size());
    }
    void send_raw(const std::vector<uint8_t>& b, bool binary = true) {
        if (conn) server_.send(conn, b.data(), b.size(), binary);
    }
    void close(uint16_t code) {
        if (conn) server_.close(conn, code, "test");
    }

    wire::Snapshot snap(const std::string& view, int64_t kind) {
        wire::Snapshot s;
        s.sub = sub_of.at(view);
        s.view = view;
        s.seq = ++seq;
        s.stamp_ns = clock_gettime_nsec_np(CLOCK_UPTIME_RAW);
        s.kind = kind;
        return s;
    }

    void heatmap(const std::vector<float>& v, int H, int W, int cm, double vmin, double vmax, uint64_t flags,
                 std::vector<double> stats = {}) {
        wire::Snapshot s = snap("heat", CALIPER_NET_KIND_HEATMAP);
        // The payload axes: every row and column of the source, unpooled.
        s.axes = Json::array();
        for (auto [dim, n] : {std::pair<const char*, int>{"rows", H}, {"cols", W}}) {
            Json a = Json::object();
            a.set("dim", dim);
            a.set("start", 0);
            a.set("step", 1);
            a.set("count", n);
            a.set("size", n);
            s.axes.push(a);
        }
        s.data = wire::Section::make("data", CALIPER_NET_DTYPE_F32, {H, W}, v.data(), v.size() * 4);
        s.colormap = cm;
        s.vmin = vmin;
        s.vmax = vmax;
        s.flags = flags;
        s.stats = std::move(stats);
        send(s);
    }

    // "world" (add_scene): a scene of the protocol 3.2 draft, 3-D with +z up.
    void add_scene() {
        Json v = Json::object();
        v.set("id", 7);
        v.set("name", "world");
        v.set("kind", CALIPER_NET_KIND_SCENE);
        v.set("sources", Json::array());
        Json sc = Json::object();
        sc.set("space", CALIPER_NET_SPACE_3D);
        sc.set("up", "+z");
        sc.set("units", "m");
        v.set("scene", sc);
        views.push(v);
    }
    wire::SceneUpdate scene_update(const std::string& path, const char* part, const std::vector<float>& pos,
                                   bool is_static) {
        wire::SceneUpdate u;
        u.sub = sub_of.at("world");
        u.view = "world";
        u.seq = ++seq;
        u.stamp_ns = clock_gettime_nsec_np(CLOCK_UPTIME_RAW);
        u.path = path;
        u.part = part;
        u.is_static = is_static;
        if (std::string(part) == CALIPER_NET_PART_LINES) u.params.set("mode", CALIPER_NET_LINES_SEGMENTS);
        u.arrays.push_back(wire::Section::make("positions", CALIPER_NET_DTYPE_F32, {(int64_t)pos.size() / 3, 3},
                                               pos.data(), pos.size() * 4));
        return u;
    }

    // "attn" (add_attention): attention maps of 4 heads, 8 queries by 6 keys.
    void add_attention() {
        add_view(6, "attn", CALIPER_NET_KIND_HEATMAP, {{"head", 4}, {"query", 8}, {"key", 6}});
    }
    static Json attn_default() {
        Json dv = Json::object(), index = Json::object();
        dv.set("source", "src");
        index.set("head", 0);
        dv.set("index", index);
        dv.set("reduce", Json::object());
        return dv;
    }
    // One snapshot of "attn" for the data view `dv`: one head (8 x 6), or with
    // the heads tiled, a 2 x 2 mosaic of them (16 x 12).
    void attention(const Json& dv) {
        const bool mosaic = dv.find("tile") != nullptr;
        const int H = mosaic ? 16 : 8, W = mosaic ? 12 : 6;
        wire::Snapshot s = snap("attn", CALIPER_NET_KIND_HEATMAP);
        std::vector<float> v((size_t)W * H);
        for (int i = 0; i < W * H; ++i) v[(size_t)i] = (float)i / (float)(W * H - 1);
        s.data = wire::Section::make("data", CALIPER_NET_DTYPE_F32, {H, W}, v.data(), v.size() * 4);
        s.data_view = dv;
        s.axes = Json::array();
        auto axis = [&](const char* dim, int n, int size) {
            Json a = Json::object();
            a.set("dim", dim);
            a.set("start", 0);
            a.set("step", 1);
            a.set("count", n);
            a.set("size", size);
            s.axes.push(a);
        };
        if (mosaic) {
            axis("head", 4, 4);
            Json tile = Json::object();
            tile.set("dim", "head");
            tile.set("rows", 2);
            tile.set("cols", 2);
            s.tile = tile;
        }
        axis("query", 8, 8);
        axis("key", 6, 6);
        s.flags = CALIPER_NET_SNAP_AUTO_VMIN | CALIPER_NET_SNAP_AUTO_VMAX;
        s.stats = {0.0, 1.0};
        send(s);
    }

    void append(const std::vector<float>& v) {
        const uint64_t start = series.size();
        series.insert(series.end(), v.begin(), v.end());
        send_series(start, v);
    }

    // Retained values reach a later subscriber as its backfill (on SUBSCRIBE).
    void send_series(uint64_t start, const std::vector<float>& v) {
        if (!sub_of.count("loss")) return;
        wire::Series s;
        s.sub = sub_of.at("loss");
        s.view = "loss";
        s.start = start;
        s.total = std::max<uint64_t>(series.size(), start + v.size());
        s.stamp_ns = clock_gettime_nsec_np(CLOCK_UPTIME_RAW);
        s.values = wire::Section::make("values", CALIPER_NET_DTYPE_F32, {(int64_t)v.size()}, v.data(), v.size() * 4);
        send(s);
    }

private:
    void add_view(uint64_t vid, const char* name, int kind, std::vector<std::pair<const char*, int>> dims) {
        Json v = Json::object();
        v.set("id", vid);
        v.set("name", name);
        v.set("kind", kind);
        Json sources = Json::array();
        if (!dims.empty()) {
            Json src = Json::object();
            src.set("name", "src");
            src.set("dtype", "f32");
            Json d = Json::array();
            for (auto& x : dims) {
                Json e = Json::object();
                e.set("name", x.first);
                e.set("size", x.second);
                d.push(e);
            }
            src.set("dims", d);
            sources.push(src);
        }
        v.set("sources", sources);
        views.push(v);
    }

    void send_views() {
        wire::Views v;
        v.session = id.id;
        v.views = script31 ? views_with_items() : views;
        send(v);
        if (script31) send_controls();
    }

    Json views_with_items() const {
        Json with = Json::array();
        for (const Json& e : views.items()) {
            Json x = e;
            const std::string name = x.find("name")->as_string();
            if (name == "loss") x.set("items", "step");
            if (name == "heat" || name == "pts") {
                Json src = x.find("sources")->at(0);
                Json dims = Json::array();
                for (size_t i = 0; i < src.find("dims")->size(); ++i) {
                    Json d = src.find("dims")->at(i);
                    if (i == 0) d.set("items", "samples");   // heat: rows; pts: the point dim
                    dims.push(d);
                }
                src.set("dims", dims);
                Json srcs = Json::array();
                srcs.push(src);
                x.set("sources", srcs);
            }
            with.push(x);
        }
        return with;
    }

public:
    void send_controls() {
        wire::Controls c;
        c.run.set("state", run_state);
        c.run.set("step", 7);
        Json param = Json::object();
        param.set("name", "lr");
        param.set("type", CALIPER_NET_PARAM_NUMBER);
        param.set("value", lr);
        param.set("min", 1e-05);
        param.set("max", 0.1);
        param.set("log", true);
        c.params.push(param);
        Json action = Json::object();
        action.set("name", "save");
        action.set("running", false);
        action.set("queued", false);
        c.actions.push(action);
        send(c);
    }

private:
    void on_control(const wire::Control& c) {
        controls_in.push_back(c);
        wire::ControlReply r;
        r.request_id = c.request_id;
        r.op = c.op;
        r.name = c.name;
        r.status = CALIPER_NET_STATUS_OK;
        auto refuse = [&](const char* code, const char* msg) {
            r.status = CALIPER_NET_STATUS_ERROR;
            Json e = Json::object();
            e.set("code", code);
            e.set("message", msg);
            r.error = e;
            send(r);
        };
        if (c.op == CALIPER_NET_OP_PAUSE) {
            run_state = CALIPER_NET_RUN_PAUSED;
        } else if (c.op == CALIPER_NET_OP_RESUME) {
            run_state = CALIPER_NET_RUN_RUNNING;
        } else if (c.op == CALIPER_NET_OP_STEP) {
            run_state = CALIPER_NET_RUN_PAUSED;   // the step ran at once
        } else if (c.op == CALIPER_NET_OP_SET) {
            if (c.name != std::string("lr")) return refuse(CALIPER_NET_ERR_UNKNOWN_CONTROL, "no such parameter");
            if (!c.value.is_number()) return refuse(CALIPER_NET_ERR_BAD_VALUE, "lr is a number");
            lr = std::min(0.1, std::max(1e-05, c.value.as_double()));
            r.value = Json(lr);
        } else if (c.op == CALIPER_NET_OP_RUN) {
            if (c.name != std::string("save")) return refuse(CALIPER_NET_ERR_UNKNOWN_CONTROL, "no such action");
            r.status = CALIPER_NET_STATUS_QUEUED;
            send(r);
            ++saves;
            r.status = CALIPER_NET_STATUS_OK;
            r.message = "saved checkpoint " + std::to_string(saves);
        }
        send(r);
        send_controls();
    }

    void on_event(const wire::Event& e) {
        events.push_back(e);
        const Json* space = e.detail.find("space");
        if (!script31 || e.type != "select" || !space || !space->is_string()) return;
        wire::Selection sel;
        sel.space = space->as_string();
        sel.view = e.view;
        Json ids = Json::array(), rows = Json::array();
        if (const Json* items = e.detail.find("items"); items && items->is_array())
            for (const Json& x : items->items()) {
                sel.items.push(x);
                if (ids.size() < CALIPER_NET_MAX_DETAIL_ITEMS) {
                    ids.push(x);
                    Json row = Json::array();
                    row.push(x.as_double() * 0.5);
                    row.push(std::string("sample ") + std::to_string((int)x.as_double()));
                    rows.push(row);
                }
            }
        if (sel.items.size() > 0) {
            Json fields = Json::array();
            fields.push("score");
            fields.push("label");
            Json d = Json::object();
            d.set("fields", fields);
            d.set("items", ids);
            d.set("values", rows);
            sel.details = d;
        }
        send(sel);
    }

    void on_message(const caliper_net::Bytes& data) {
        wire::Message m;
        const wire::DecodeResult r = wire::decode(data.data(), data.size(), &m);
        if (!r.ok()) return;
        if (auto* hh = std::get_if<wire::Hello>(&m)) {
            ++hellos;
            hello_caps = hh->capabilities;
            wire::Welcome w;
            w.role = CALIPER_NET_ROLE_PRODUCER;
            w.session = id.to_json();
            w.capabilities = welcome_caps;
            w.control = welcome_control;
            send(w);
            wire::Sessions s;
            wire::SessionInfo si{id, CALIPER_NET_SESSION_LIVE, views.size()};
            Json entry = si.to_json();
            if (!welcome_caps.empty()) {
                Json caps = Json::array();
                for (const auto& c : welcome_caps) caps.push(c);
                entry.set("capabilities", caps);
            }
            s.sessions.push(entry);
            send(s);
            if (auto_attach) send_views();
        } else if (auto* a = std::get_if<wire::Attach>(&m)) {
            if (a->session == id.id) {
                send_views();
            } else {
                wire::Error e;
                e.code = CALIPER_NET_ERR_UNKNOWN_SESSION;
                send(e);
            }
        } else if (auto* s = std::get_if<wire::Subscribe>(&m)) {
            sub_of[s->view] = s->sub;
            ++subscribes[s->view];
            wire::ViewReply rep;
            rep.request_id = s->request_id;
            rep.sub = s->sub;
            rep.view = s->view;
            rep.data_view = Json::object();
            if (s->view == "attn") rep.data_view = attn_default();
            send(rep);
            if (s->view == "loss" && !series.empty()) send_series(0, series);   // the backfill
        } else if (auto* un = std::get_if<wire::Unsubscribe>(&m)) {
            unsubscribed.push_back(un->sub);   // sub_of keeps it: what is in flight still goes
        } else if (auto* p = std::get_if<wire::Ping>(&m)) {
            send(wire::Pong{p->nonce});
        } else if (auto* po = std::get_if<wire::Pong>(&m)) {
            pongs.push_back(po->nonce);
        } else if (auto* vr = std::get_if<wire::ViewRequest>(&m)) {
            // Resolved as asked, the source named or the first.
            view_requests.push_back(*vr);
            wire::ViewReply rep;
            rep.request_id = vr->request_id;
            rep.sub = vr->sub;
            rep.view = vr->view;
            rep.data_view = vr->data_view;
            if (!rep.data_view.has("source")) rep.data_view.set("source", "src");
            send(rep);
        } else if (auto* c = std::get_if<wire::Control>(&m)) {
            on_control(*c);
        } else if (auto* ev = std::get_if<wire::Event>(&m)) {
            on_event(*ev);
        } else if (auto* e = std::get_if<wire::Error>(&m)) {
            errors.push_back(e->code);
        } else if (auto* b = std::get_if<wire::Bye>(&m)) {
            byes.push_back(b->reason);
        }
    }

    void loop() {
        while (!stop_) {
            for (auto& e : server_.poll(5)) {
                using T = caliper_net::Event::Type;
                if (e.type == T::Request) {
                    const int st = wire::upgrade_status(e.request, token, port());
                    if (st) server_.reject(e.conn, st);
                    else server_.accept(e.conn);
                } else if (e.type == T::Open) {
                    conn = e.conn;
                    sub_of.clear();
                } else if (e.type == T::Message && e.conn == conn) {
                    on_message(e.data);
                } else if (e.type == T::Closed) {
                    closes.push_back(e.code);
                    if (e.conn == conn) conn = 0;
                }
            }
            std::deque<std::function<void()>> todo;
            {
                std::lock_guard<std::mutex> g(mu_);
                todo.swap(tasks_);
            }
            for (auto& t : todo) t();
        }
    }

    caliper_net::Server server_;
    std::thread thread_;
    std::atomic<bool> stop_{false};
    std::mutex mu_;
    std::deque<std::function<void()>> tasks_;
};

// ---- the viewer side ------------------------------------------------------------------------

// The core's log lines (CaliperCoreDesc::log_fn), for a test that waits on one.
struct LogLines {
    std::mutex mu;
    std::vector<std::string> lines;
    static void fn(void* self, int, const char* m) {
        auto* l = static_cast<LogLines*>(self);
        std::lock_guard<std::mutex> g(l->mu);
        l->lines.push_back(m);
    }
    bool has(const std::string& s) {
        std::lock_guard<std::mutex> g(mu);
        for (const auto& x : lines)
            if (x.find(s) != std::string::npos) return true;
        return false;
    }
};

CaliperCore* net_core(int W, int H, std::string* why, LogLines* log = nullptr) {
    CaliperCoreDesc d{};
    d.struct_size = sizeof d;
    d.renderer = CALIPER_RENDERER_DEFAULT;
    if (log) {
        d.log_fn = &LogLines::fn;
        d.userdata = log;
    }
    CaliperCore* core = caliper_core_create(&d);
    REQUIRE(core != nullptr);
    CaliperCanvasDesc c{};
    c.struct_size = sizeof c;
    c.mode = CALIPER_CANVAS_OFFSCREEN;
    c.width = W;
    c.height = H;
    c.content_scale = 1.0f;
    if (caliper_test::force_no_gpu() || !caliper_core_attach_canvas(core, nullptr, &c)) {
        *why = caliper_core_last_error(core);
        caliper_core_shutdown(core);
        return nullptr;
    }
    return core;
}

CaliperLiveStatus status(CaliperCore* core) {
    CaliperLiveStatus st{};
    st.struct_size = sizeof st;
    caliper_core_live_status(core, &st, 0);
    return st;
}

CaliperLiveViewStatus view_status(CaliperCore* core, const std::string& name) {
    const CaliperLiveStatus st = status(core);
    for (int i = 0; i < st.views; ++i) {
        CaliperLiveViewStatus vs{};
        vs.struct_size = sizeof vs;
        if (caliper_core_live_view_status(core, i, &vs) && name == vs.name) return vs;
    }
    CaliperLiveViewStatus none{};
    none.name = "";
    none.error = "";
    return none;
}

CaliperLiveViewInfo view_info(CaliperCore* core, const char* name) {
    CaliperLiveViewInfo vi{};
    vi.struct_size = sizeof vi;
    caliper_core_live_view_info(core, name, &vi);
    return vi;
}

std::vector<float> values(CaliperCore* core, const char* name) {
    size_t n = 0;
    caliper_core_live_view_values(core, name, nullptr, 0, &n);
    std::vector<float> v(n);
    if (n) caliper_core_live_view_values(core, name, v.data(), v.size(), &n);
    return v;
}

std::vector<uint8_t> readback(CaliperCore* core, const char* name, int W, int H) {
    std::vector<uint8_t> px((size_t)W * H * 4, 0);
    if (!caliper_core_live_view_readback(core, name, px.data(), px.size())) px.clear();
    return px;
}

std::vector<uint8_t> map_ref(const std::vector<float>& v, int W, int H, int32_t cm, float lo, float hi) {
    std::vector<uint8_t> out((size_t)W * H * 4);
    caliper_host::map_f32_to_rgba8(v.data(), W, H, caliper_host::colormap_lut(cm), lo, hi, out.data());
    return out;
}

// Waits until `done` holds without polling the core, so nothing is drawn meanwhile.
bool wait_until(const std::function<bool()>& done, int ms = 10000) {
    const auto end = std::chrono::steady_clock::now() + std::chrono::milliseconds(ms);
    while (std::chrono::steady_clock::now() < end) {
        if (done()) return true;
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    return done();
}

// The producer pings the viewer and waits for the PONG, which the viewer's
// network thread sends only after it has handled every message the producer
// sent before the PING: then those snapshots sit in its slots and any ERROR
// it sent back has reached the producer.
bool viewer_caught_up(FakeProducer& p) {
    static uint32_t nonce = 7000;
    const uint32_t n = ++nonce;
    p.call([n](FakeProducer& f) {
        f.send(wire::Ping{n});
        return 0;
    });
    return wait_until([&] {
        return p.call([n](FakeProducer& f) { return std::count(f.pongs.begin(), f.pongs.end(), n) > 0; });
    });
}

// Polls and draws until `done` holds (the producer runs on its own thread).
bool poll_until(CaliperCore* core, const std::function<bool()>& done, int ms = 5000) {
    const auto end = std::chrono::steady_clock::now() + std::chrono::milliseconds(ms);
    while (std::chrono::steady_clock::now() < end) {
        caliper_core_live_poll(core);
        caliper_core_frame(core);
        if (done()) return true;
        std::this_thread::sleep_for(std::chrono::milliseconds(2));
    }
    return false;
}

std::vector<float> ramp(int W, int H, float scale) {
    std::vector<float> v((size_t)W * H);
    for (int i = 0; i < W * H; ++i) v[(size_t)i] = scale * (float)i / (float)(W * H - 1);
    return v;
}

bool same_values(const std::vector<float>& got, const std::vector<float>& want) {
    if (got.size() != want.size()) return false;
    for (size_t i = 0; i < got.size(); ++i) {
        if (std::isnan(want[i]) ? !std::isnan(got[i]) : got[i] != want[i]) return false;
    }
    return true;
}

}  // namespace

TEST_CASE("live/net: stream links are checked and normalised") {
    caliper_live::StreamUrl u;
    std::string err;
    const std::string tok = "abcdefghijklmnopqrstuv";
    REQUIRE(caliper_live::parse_stream_url("ws://127.0.0.1:8765/v3/stream?token=" + tok, &u, &err));
    CHECK(u.port == 8765);
    CHECK(u.token == tok);
    CHECK(u.session.empty());
    CHECK(u.ws_url == "ws://127.0.0.1:8765/v3/stream?token=" + tok);
    CHECK(u.display == "ws://127.0.0.1:8765/v3/stream");
    REQUIRE(caliper_live::parse_stream_url(" http://localhost:9/?token=" + tok + "&session=clp-1-a \n", &u, &err));
    CHECK(u.host == "localhost");
    CHECK(u.session == "clp-1-a");
    CHECK(u.ws_url == "ws://localhost:9/v3/stream?token=" + tok);
    CHECK_FALSE(caliper_live::parse_stream_url("ws://10.0.0.2:9/v3/stream?token=" + tok, &u, &err));
    CHECK(err.find("loopback") != std::string::npos);
    CHECK_FALSE(caliper_live::parse_stream_url("ws://127.0.0.1:9/v3/stream?token=short", &u, &err));
    CHECK_FALSE(caliper_live::parse_stream_url("ws://127.0.0.1:9/v3/stream", &u, &err));
    CHECK_FALSE(caliper_live::parse_stream_url("ws://127.0.0.1:9/other?token=" + tok, &u, &err));
    CHECK_FALSE(caliper_live::parse_stream_url("wss://127.0.0.1:9/v3/stream?token=" + tok, &u, &err));
    CHECK_FALSE(caliper_live::parse_stream_url("ws://127.0.0.1:9/v3/stream?token=" + tok + "&session=a/b", &u, &err));
    // No error repeats the token, whatever is wrong with the link.
    for (const std::string bad : {"ftp://127.0.0.1:9/v3/stream?token=", "ws:///v3/stream?token=",
                                  "ws://127.0.0.1:99999/v3/stream?token=", "http://127.0.0.1:0/?token=",
                                  "ws://[::1]:9/v3/stream?token=", "ws://10.0.0.2:9/v3/stream?token=",
                                  "ws://127.0.0.1:9/other?token=", "wss://127.0.0.1:9/v3/stream?token="}) {
        CAPTURE(bad);
        err.clear();
        CHECK_FALSE(caliper_live::parse_stream_url(bad + tok, &u, &err));
        CHECK_FALSE(err.empty());
        CHECK(err.find(tok) == std::string::npos);
    }
}

TEST_CASE("live/net: a scene view (protocol 3.2 draft) takes the scene's messages in order through the app's "
          "source, applies every part, transforms and class tables, refuses an encoded image, and draws") {
    std::string no_gpu;
    CaliperCore* core = net_core(320, 240, &no_gpu);
    if (!core) CALIPER_SKIP_NO_GPU("no embeddable GPU: " << no_gpu);
    FakeProducer p;
    p.add_scene();
    p.start();
    REQUIRE_MESSAGE(caliper_core_live_attach_url(core, p.url().c_str()) == 1, caliper_core_last_error(core));
    REQUIRE(poll_until(core, [&] { return p.call([](FakeProducer& f) { return f.sub_of.count("world"); }) == 1; }));
    const std::vector<std::string> caps = p.call([](FakeProducer& f) { return f.hello_caps; });
    CHECK(std::find(caps.begin(), caps.end(), CALIPER_NET_CAP_SCENES) != caps.end());

    // The state of a subscription, then two updates of one part (the newer wins wherever they
    // are taken), arrows, an entity transform, a class table, and an encoded image, the one form
    // this viewer does not draw yet.
    const std::vector<float> frame = {-1, -1, -1, 1, 1, 1}, a = {0, 0, 0, 1, 0, 0}, b = {0, 0, 0, 0, 1, 0, 0, 0, 1};
    static const uint8_t jpeg[4] = {0xff, 0xd8, 0xff, 0xd9};
    p.call([&](FakeProducer& f) {
        wire::SceneClear reset;
        reset.sub = f.sub_of.at("world");
        reset.view = "world";
        reset.seq = ++f.seq;
        reset.path = "";
        reset.reset = true;
        reset.state = true;
        f.send(reset);
        wire::SceneUpdate fr = f.scene_update("", CALIPER_NET_PART_LINES, frame, true);
        fr.state = true;
        f.send(fr);
        f.send(f.scene_update("cloud", CALIPER_NET_PART_POINTS, a, false));
        f.send(f.scene_update("cloud", CALIPER_NET_PART_POINTS, b, false));
        wire::SceneUpdate arrows = f.scene_update("arrows", CALIPER_NET_PART_POINTS, a, false);
        arrows.part = CALIPER_NET_PART_ARROWS;
        arrows.arrays.at(0).name = "origins";
        wire::Section vec = arrows.arrays.at(0);
        vec.name = "vectors";
        arrows.arrays.push_back(vec);
        f.send(arrows);
        static const float shift[16] = {1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 2, 0, 0, 1};
        wire::SceneTransforms tr;
        tr.sub = f.sub_of.at("world");
        tr.view = "world";
        tr.seq = ++f.seq;
        tr.paths = {"cloud"};
        tr.matrices = wire::Section::make("matrices", CALIPER_NET_DTYPE_F32, {1, 16}, shift, sizeof shift);
        f.send(tr);
        wire::SceneClasses cl;
        cl.sub = f.sub_of.at("world");
        cl.view = "world";
        cl.seq = ++f.seq;
        Json entry = Json::object();
        entry.set("id", Json(3));
        entry.set("label", Json("rock"));
        Json col = Json::array();
        for (int c : {10, 20, 30, 255}) col.push(Json(c));
        entry.set("color", col);
        cl.classes.push(entry);
        f.send(cl);
        wire::SceneUpdate enc = f.scene_update("photo", CALIPER_NET_PART_POINTS, a, false);
        enc.part = CALIPER_NET_PART_IMAGE;
        enc.params = Json::object();
        enc.params.set("form", Json(CALIPER_NET_FORM_ENCODED));
        enc.params.set("format", Json(CALIPER_NET_FORMAT_JPEG));
        enc.arrays = {wire::Section::make("pixels", CALIPER_NET_DTYPE_U8, {4}, jpeg, sizeof jpeg)};
        f.send(enc);
        return 0;
    });
    caliper_live_views::SceneView* sv = nullptr;
    REQUIRE(poll_until(core, [&] {
        sv = caliper_live_views::scene_view(core, "world");
        return sv && sv->counts().batches == 3 && !sv->tree().classes().empty() && !sv->notes().empty() &&
               sv->has_target();
    }));
    CHECK(sv->tree().up() == caliper_scene::Axis::PosZ);
    const caliper_scene::EntityId cl = sv->tree().find("cloud");
    const caliper_scene::Batch* cloud = sv->tree().batch(cl, caliper_scene::Primitive::Points);
    REQUIRE(cloud != nullptr);
    CHECK(cloud->count() == 3);   // the newer update
    CHECK(sv->tree().batch_static(sv->tree().root(), caliper_scene::Primitive::Lines));
    REQUIRE(sv->tree().batch(sv->tree().find("arrows"), caliper_scene::Primitive::Arrows));
    REQUIRE(sv->tree().transform(cl));
    CHECK(sv->tree().transform(cl)->at(0, 3) == 2.0);
    CHECK(sv->tree().class_label(3) == "rock");
    const caliper_scene::EntityId photo = sv->tree().find("photo");
    CHECK((!photo || sv->tree().batch(photo, caliper_scene::Primitive::Image) == nullptr));
    const auto noted = [&](const char* what) {
        for (const std::string& n : sv->notes())
            if (n.find(what) != std::string::npos) return true;
        return false;
    };
    CHECK_MESSAGE(noted("encoded images (JPEG or PNG) are not drawn yet"), sv->notes().front());
    REQUIRE(poll_until(core, [&] {
        const auto errs = p.call([](FakeProducer& f) { return f.errors; });
        return std::find(errs.begin(), errs.end(), CALIPER_NET_ERR_UNSUPPORTED) != errs.end();
    }));
    // The view in the registry: kind 6, its items (three points, one segment, two arrows), its
    // target and a readback of it.
    CaliperLiveViewInfo vi = view_info(core, "world");
    CHECK(vi.kind == CALIPER_NET_KIND_SCENE);
    CHECK(vi.count == 3 + 1 + 2);
    CHECK(std::string(vi.path) == "copied-from-host");
    REQUIRE(vi.width > 0);
    std::vector<uint8_t> px((size_t)vi.width * vi.height * 4);
    CHECK(caliper_core_live_view_readback(core, "world", px.data(), px.size()) == 1);
    // One copy per uploaded array (the frame's and the cloud's positions, the arrows' origins and
    // vectors) and the class table's texture, uploaded with the frame that drew it.
    caliper_stats::ViewCopies vc{};
    REQUIRE(poll_until(core, [&] {
        return caliper_stats::view_copies(core, "world", &vc) && vc.total.bytes >= (frame.size() + b.size() + 2 * a.size()) * 4 + 65536 * 4;
    }));
    caliper_core_live_detach(core);
    caliper_core_shutdown(core);
}

// A view's status says "" when its last update succeeded (CaliperLiveViewStatus.error): a scene
// update the view refuses shows its reason, and the next one it applies clears it.
TEST_CASE("live/net: a scene view's error clears once an update applies again") {
    std::string no_gpu;
    CaliperCore* core = net_core(320, 240, &no_gpu);
    if (!core) CALIPER_SKIP_NO_GPU("no embeddable GPU: " << no_gpu);
    FakeProducer p;
    p.add_scene();
    p.start();
    REQUIRE_MESSAGE(caliper_core_live_attach_url(core, p.url().c_str()) == 1, caliper_core_last_error(core));
    REQUIRE(poll_until(core, [&] { return p.call([](FakeProducer& f) { return f.sub_of.count("world"); }) == 1; }));
    p.call([&](FakeProducer& f) {
        wire::SceneClear reset;
        reset.sub = f.sub_of.at("world");
        reset.view = "world";
        reset.seq = ++f.seq;
        reset.reset = true;
        reset.state = true;
        f.send(reset);
        // 2-D positions in a 3-D scene: the view refuses the batch.
        wire::SceneUpdate flat = f.scene_update("flat", CALIPER_NET_PART_POINTS, {}, false);
        const std::vector<float> xy = {0, 0, 1, 1};
        flat.arrays.clear();
        flat.arrays.push_back(wire::Section::make("positions", CALIPER_NET_DTYPE_F32, {2, 2}, xy.data(), xy.size() * 4));
        f.send(flat);
        return 0;
    });
    REQUIRE(poll_until(core, [&] { return std::string(view_status(core, "world").error).find("flat") != std::string::npos; }));
    const uint64_t updates = view_status(core, "world").updates;
    p.call([&](FakeProducer& f) {
        f.send(f.scene_update("cloud", CALIPER_NET_PART_POINTS, {0, 0, 0, 1, 1, 1}, false));
        return 0;
    });
    REQUIRE(poll_until(core, [&] { return view_status(core, "world").updates > updates; }));
    CHECK(std::string(view_status(core, "world").error).empty());
    caliper_core_live_detach(core);
    caliper_core_shutdown(core);
}

// A scene view the core refuses for good (its name is taken by an embed v1.2 view) stops its
// subscription: the source unsubscribes and drops what still arrives, rather than queueing it
// until the queue's bound and subscribing again, over and over.
TEST_CASE("live/net: a scene view refused for good unsubscribes instead of queueing its messages") {
    std::string no_gpu;
    CaliperCore* core = net_core(320, 240, &no_gpu);
    if (!core) CALIPER_SKIP_NO_GPU("no embeddable GPU: " << no_gpu);
    // An embed v1.2 view named "world", from the host's CPU memory.
    std::vector<float> px(4 * 4, 0.5f);
    CaliperTensor t{};
    t.struct_size = sizeof t;
    t.data = px.data();
    t.dtype = CALIPER_DT_F32;
    t.ndim = 2;
    t.shape[0] = t.shape[1] = 4;
    t.strides[0] = 4;
    t.strides[1] = 1;
    t.device = CALIPER_DEV_CPU;
    CaliperViewDesc vd{};
    vd.struct_size = sizeof vd;
    vd.name = "world";
    vd.tensor = &t;
    vd.vmax = 1.0f;
    REQUIRE_MESSAGE(caliper_core_view_tensor(core, &vd) == 1, caliper_core_last_error(core));
    FakeProducer p;
    p.add_scene();
    p.start();
    REQUIRE_MESSAGE(caliper_core_live_attach_url(core, p.url().c_str()) == 1, caliper_core_last_error(core));
    REQUIRE(poll_until(core, [&] {
        return p.call([](FakeProducer& f) { return f.sub_of.count("world") + f.sub_of.count("loss"); }) == 2;
    }));
    REQUIRE(poll_until(core, [&] { return std::string(view_status(core, "world").error).find("taken") != std::string::npos; }));
    const uint32_t sub = p.call([](FakeProducer& f) { return f.sub_of.at("world"); });
    // What still arrives for it is dropped, past the queue's bound too, and asks for nothing.
    p.call([&](FakeProducer& f) {
        for (size_t i = 0; i < caliper_live::kSceneQueueMax + 10; ++i) {
            wire::SceneClear c;
            c.sub = sub;
            c.view = "world";
            c.seq = ++f.seq;
            c.path = "c/" + std::to_string(i);
            f.send(c);
        }
        f.append({1.0f});   // after them on the connection: once it arrives, so have they
        return 0;
    });
    REQUIRE(poll_until(core, [&] { return view_status(core, "loss").series_total >= 1; }));
    CHECK(p.call([](FakeProducer& f) { return f.subscribes["world"]; }) == 1);
    REQUIRE(poll_until(core, [&] {
        const auto un = p.call([](FakeProducer& f) { return f.unsubscribed; });
        return std::find(un.begin(), un.end(), sub) != un.end();
    }));
    caliper_core_live_detach(core);
    caliper_core_shutdown(core);
}

// ---- the scene queue (live/scene_queue.h) --------------------------------------------------------
namespace {
using SceneQ = caliper_live::SceneQueue;
bool qpush(SceneQ& q, caliper_live::SceneItem it) { return caliper_live::scene_queue_push(q, std::move(it)); }
std::deque<caliper_live::SceneItem>& qitems(SceneQ& q) { return q.items; }
bool qwaits(const SceneQ& q) { return q.resync; }

caliper_live::SceneItem sq_update(const std::string& path, const char* part, bool is_static = false) {
    wire::SceneUpdate u;
    u.sub = 1;
    u.view = "world";
    u.path = path;
    u.part = part;
    u.is_static = is_static;
    return caliper_live::SceneItem{nullptr, std::move(u)};
}
caliper_live::SceneItem sq_clear(const std::string& path, bool recursive = false, bool is_static = false,
                                 std::optional<std::vector<std::string>> parts = std::nullopt, bool reset = false) {
    wire::SceneClear c;
    c.sub = 1;
    c.view = "world";
    c.path = path;
    c.recursive = recursive;
    c.is_static = is_static;
    c.parts = std::move(parts);
    c.reset = reset;
    return caliper_live::SceneItem{nullptr, std::move(c)};
}
caliper_live::SceneItem sq_reset() { return sq_clear("", false, false, std::nullopt, true); }
// The queue as text: "reset", "clear <path>", "<path>:<part>", with " static" for a static item.
std::vector<std::string> sq_log(SceneQ& q) {
    std::vector<std::string> out;
    for (const auto& it : qitems(q)) {
        if (const auto* u = std::get_if<wire::SceneUpdate>(&it.msg))
            out.push_back(u->path + ":" + u->part + (u->is_static ? " static" : ""));
        else if (const auto* c = std::get_if<wire::SceneClear>(&it.msg))
            out.push_back(c->reset ? "reset" : "clear " + (c->path.empty() ? std::string("<root>") : c->path));
        else if (const auto* t = std::get_if<wire::SceneTransforms>(&it.msg)) {
            std::string paths;
            for (const std::string& p : t->paths) paths += (paths.empty() ? "" : ",") + p;
            out.push_back("transforms " + paths + (t->is_static ? " static" : ""));
        } else if (const auto* k = std::get_if<wire::SceneClasses>(&it.msg))
            out.push_back(std::string("classes") + (k->is_static ? " static" : "") + " " + std::to_string(k->seq));
        else
            out.push_back(std::get<std::string>(it.msg));
    }
    return out;
}
caliper_live::SceneItem sq_transforms(std::vector<std::string> paths, bool is_static = false) {
    wire::SceneTransforms t;
    t.sub = 1;
    t.view = "world";
    t.paths = std::move(paths);
    t.is_static = is_static;
    return caliper_live::SceneItem{nullptr, std::move(t)};
}
caliper_live::SceneItem sq_classes(uint64_t seq, bool is_static = true) {
    wire::SceneClasses c;
    c.sub = 1;
    c.view = "world";
    c.seq = seq;
    c.is_static = is_static;
    return caliper_live::SceneItem{nullptr, std::move(c)};
}
}  // namespace

// Entity transforms replace a queued message of the same layer whose every path they set again,
// and a class table replaces a queued one of the same layer, so that an arm whose joints move
// every step keeps one message while the display is paused.
TEST_CASE("live/scene queue: entity transforms and class tables keep the newest of each layer") {
    SceneQ q;
    qpush(q, sq_reset());
    qpush(q, sq_transforms({"rig/left", "rig/right"}, true));
    for (int i = 0; i < 9000; ++i) {
        qpush(q, sq_transforms({"arm/link0", "arm/link0/link1"}));
        qpush(q, sq_classes(uint64_t(i), false));
    }
    CHECK(sq_log(q) == std::vector<std::string>{"reset", "transforms rig/left,rig/right static",
                                                "transforms arm/link0,arm/link0/link1", "classes 8999"});
    CHECK_FALSE(qwaits(q));
    // A message that sets only some of a queued one's paths keeps it; a static table stays beside a
    // stamped one; a reset drops both.
    qpush(q, sq_transforms({"arm/link0"}));
    qpush(q, sq_classes(1, true));
    CHECK(sq_log(q) == std::vector<std::string>{"reset", "transforms rig/left,rig/right static",
                                                "transforms arm/link0,arm/link0/link1", "classes 8999",
                                                "transforms arm/link0", "classes static 1"});
    qpush(q, sq_reset());
    CHECK(sq_log(q) == std::vector<std::string>{"reset"});
}

// While the display is paused a source keeps what SceneView::apply would leave: a clear removes
// the queued updates it removes and the clears it covers, so that clearing and logging one entity
// every step keeps one clear and one update; resets and clears are never dropped.
TEST_CASE("live/scene queue: a clear drops the queued updates it removes and the clears it covers; resets stay") {
    SceneQ q;
    qpush(q, sq_reset());
    qpush(q, sq_update("", CALIPER_NET_PART_LINES, true));
    for (int i = 0; i < 9000; ++i) {
        qpush(q, sq_clear("cloud"));
        qpush(q, sq_update("cloud", CALIPER_NET_PART_POINTS));
    }
    CHECK(sq_log(q) == std::vector<std::string>{"reset", ":lines static", "clear cloud", "cloud:points"});
    CHECK_FALSE(qwaits(q));
    // What a clear keeps: a static item (unless it is static itself), a part it does not name, a
    // path it does not reach.
    SceneQ k;
    qpush(k, sq_update("a", CALIPER_NET_PART_POINTS, true));
    qpush(k, sq_update("a", CALIPER_NET_PART_MESH));
    qpush(k, sq_update("ab", CALIPER_NET_PART_POINTS));
    qpush(k, sq_update("a/b", CALIPER_NET_PART_POINTS));
    qpush(k, sq_clear("a", true, false, std::vector<std::string>{CALIPER_NET_PART_POINTS}));
    CHECK(sq_log(k) == std::vector<std::string>{"a:points static", "a:mesh", "ab:points", "clear a"});
    // A static recursive clear of the root removes everything queued, the earlier clear included.
    qpush(k, sq_clear("", true, true));
    CHECK(sq_log(k) == std::vector<std::string>{"clear <root>"});
}

// A sender that clears ever new paths while the display is paused: past kSceneQueueMax messages
// the queue empties and waits for the scene's next state, which the source asks for; nothing but
// that state's reset is queued meanwhile.
TEST_CASE("live/scene queue: past kSceneQueueMax messages the queue waits for the scene's next state") {
    SceneQ q;
    qpush(q, sq_reset());
    qpush(q, sq_update("", CALIPER_NET_PART_LINES, true));
    bool asked = false;
    for (size_t i = 0; i < caliper_live::kSceneQueueMax + 10; ++i) asked = qpush(q, sq_clear("c/" + std::to_string(i))) || asked;
    CHECK(asked);
    CHECK(qwaits(q));
    CHECK(qitems(q).empty());
    CHECK_FALSE(qpush(q, sq_update("x", CALIPER_NET_PART_POINTS)));
    CHECK(qitems(q).empty());
    qpush(q, sq_reset());
    CHECK_FALSE(qwaits(q));
    CHECK(sq_log(q) == std::vector<std::string>{"reset"});
}

// The app's source: a paused scene view whose queue passes kSceneQueueMax subscribes again on the
// same sub, and the producer's new state is what the view shows once the display resumes.
TEST_CASE("live/net: a paused scene view whose queue passes its bound asks for the scene's state again") {
    std::string no_gpu;
    CaliperCore* core = net_core(320, 240, &no_gpu);
    if (!core) CALIPER_SKIP_NO_GPU("no embeddable GPU: " << no_gpu);
    FakeProducer p;
    p.add_scene();
    p.start();
    REQUIRE_MESSAGE(caliper_core_live_attach_url(core, p.url().c_str()) == 1, caliper_core_last_error(core));
    REQUIRE(poll_until(core, [&] { return p.call([](FakeProducer& f) { return f.sub_of.count("world"); }) == 1; }));
    const std::vector<float> frame = {-1, -1, -1, 1, 1, 1}, cloud = {0, 0, 0, 1, 0, 0};
    auto state = [&](const std::string& path, const char* part, const std::vector<float>& pos, bool is_static) {
        p.call([&](FakeProducer& f) {
            wire::SceneClear reset;
            reset.sub = f.sub_of.at("world");
            reset.view = "world";
            reset.seq = ++f.seq;
            reset.reset = true;
            reset.state = true;
            f.send(reset);
            wire::SceneUpdate u = f.scene_update(path, part, pos, is_static);
            u.state = true;
            f.send(u);
            return 0;
        });
    };
    state("", CALIPER_NET_PART_LINES, frame, true);
    caliper_live_views::SceneView* sv = nullptr;
    REQUIRE(poll_until(core, [&] {
        sv = caliper_live_views::scene_view(core, "world");
        return sv && sv->counts().batches == 1;
    }));
    const uint32_t sub = p.call([](FakeProducer& f) { return f.sub_of.at("world"); });
    caliper_core_live_set_paused(core, 1);
    p.call([&](FakeProducer& f) {
        for (size_t i = 0; i < caliper_live::kSceneQueueMax + 10; ++i) {
            wire::SceneClear c;
            c.sub = f.sub_of.at("world");
            c.view = "world";
            c.seq = ++f.seq;
            c.path = "c/" + std::to_string(i);
            f.send(c);
        }
        return 0;
    });
    REQUIRE(poll_until(core, [&] { return p.call([](FakeProducer& f) { return f.subscribes["world"]; }) == 2; }));
    CHECK(p.call([](FakeProducer& f) { return f.sub_of.at("world"); }) == sub);
    // The producer answers with the scene's state, which the view takes once the display resumes.
    state("cloud", CALIPER_NET_PART_POINTS, cloud, false);
    caliper_core_live_set_paused(core, 0);
    REQUIRE(poll_until(core, [&] {
        sv = caliper_live_views::scene_view(core, "world");
        return sv && sv->tree().find("cloud") &&
               sv->tree().batch(sv->tree().find("cloud"), caliper_scene::Primitive::Points) != nullptr;
    }));
    CHECK(sv->counts().batches == 1);   // the reset of the new state removed the frame
    caliper_core_live_detach(core);
    caliper_core_shutdown(core);
}

// The scene view reads a batch's colormap and flags as the integers they are: a flag far above
// the known ones keeps AUTO_VMIN (2^60 + 1 is not a double, and through one it loses bit 0), and
// a colormap id past 32 bits, or below 0, is no colormap.
TEST_CASE("live/net: a scene batch's colormap and flags are read as integers, whatever their size") {
    std::string no_gpu;
    CaliperCore* core = net_core(320, 240, &no_gpu);
    if (!core) CALIPER_SKIP_NO_GPU("no embeddable GPU: " << no_gpu);
    FakeProducer p;
    p.add_scene();
    p.start();
    REQUIRE_MESSAGE(caliper_core_live_attach_url(core, p.url().c_str()) == 1, caliper_core_last_error(core));
    REQUIRE(poll_until(core, [&] { return p.call([](FakeProducer& f) { return f.sub_of.count("world"); }) == 1; }));
    const std::vector<float> two = {0, 0, 0, 1, 1, 1};
    p.call([&](FakeProducer& f) {
        wire::SceneClear reset;
        reset.sub = f.sub_of.at("world");
        reset.view = "world";
        reset.seq = ++f.seq;
        reset.reset = true;
        reset.state = true;
        f.send(reset);
        Json stats = Json::array();
        for (double x : {5.0, 9.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0}) stats.push(x);
        wire::SceneUpdate a = f.scene_update("auto", CALIPER_NET_PART_POINTS, two, false);
        a.params.set("flags", (uint64_t(1) << 60) | (uint64_t)CALIPER_NET_SNAP_AUTO_VMIN);
        a.params.set("vmin", 0.0);
        a.params.set("vmax", 1.0);
        a.params.set("stats", stats);
        f.send(a);
        wire::SceneUpdate big = f.scene_update("big", CALIPER_NET_PART_POINTS, two, false);
        big.params.set("colormap", (int64_t)((uint64_t(1) << 40) | 2u));
        f.send(big);
        wire::SceneUpdate neg = f.scene_update("negative", CALIPER_NET_PART_POINTS, two, false);
        neg.params.set("colormap", (int64_t)-2);
        f.send(neg);
        return 0;
    });
    caliper_live_views::SceneView* sv = nullptr;
    REQUIRE(poll_until(core, [&] {
        sv = caliper_live_views::scene_view(core, "world");
        return sv && sv->counts().batches == 3;
    }));
    using caliper_scene::Primitive;
    const caliper_scene::Batch* a = sv->tree().batch(sv->tree().find("auto"), Primitive::Points);
    const caliper_scene::Batch* big = sv->tree().batch(sv->tree().find("big"), Primitive::Points);
    const caliper_scene::Batch* neg = sv->tree().batch(sv->tree().find("negative"), Primitive::Points);
    REQUIRE((a && big && neg));
    CHECK(a->vmin == 5.0f);   // AUTO_VMIN: the producer's stats
    CHECK(a->vmax == 1.0f);   // fixed
    CHECK(big->colormap == 0u);
    CHECK(neg->colormap == 0u);
    caliper_core_live_detach(core);
    caliper_core_shutdown(core);
}

TEST_CASE("live/net: a producer streams a heatmap, an image, points, a series and a histogram to "
          "an offscreen core attached by URL") {
    std::string no_gpu;
    CaliperCore* core = net_core(96, 64, &no_gpu);
    if (!core) CALIPER_SKIP_NO_GPU("no embeddable GPU: " << no_gpu);
    FakeProducer p;
    p.start();

    REQUIRE_MESSAGE(caliper_core_live_attach_url(core, p.url().c_str()) == 1, caliper_core_last_error(core));
    CHECK(caliper_core_live_attach_url(core, p.url().c_str()) == 0);   // already attached
    CaliperLiveStatus st = status(core);
    CHECK(st.attached == 1);
    CHECK(st.pending == 0);
    CHECK(st.client_alive == 1);
    CHECK(std::string(st.session) == "clp-test-4242");   // the session id, never the token
    CHECK(std::string(st.script) == "fake_train.py");
    CHECK(st.client_pid == 4242);
    CHECK(st.start_unix_ms == 1763200000000ull);
    // Every view of VIEWS is subscribed with its default data view.
    REQUIRE(poll_until(core, [&] { return p.call([](FakeProducer& f) { return f.sub_of.size(); }) == 5; }));
    CHECK(status(core).views == 5);

    // One snapshot of every kind.
    const int HW = 4, HH = 4 * 1;   // heat is 4 rows x 8 columns
    const std::vector<float> heat = ramp(8, HH, 1.0f);
    std::vector<uint8_t> img(4 * 4 * 4);
    for (size_t i = 0; i < img.size(); ++i) img[i] = (uint8_t)(i * 13 + 7);
    const int N = 6;
    std::vector<float> xy, labels;
    for (int i = 0; i < N; ++i) {
        xy.push_back((float)(i % 3));
        xy.push_back((float)(i / 3));
        labels.push_back((float)i);
    }
    const std::vector<float> bins = {1, 4, 9, 16, 25, 16, 9, 4};
    const std::vector<float> loss = {2.5f, 2.0f, 1.5f, 1.25f, 1.0f, 0.9f, 0.8f, 0.7f, 0.65f, 0.6f};
    (void)HW;
    p.call([&](FakeProducer& f) {
        f.heatmap(heat, HH, 8, CALIPER_NET_COLORMAP_MAGMA, 0.0, 1.0, CALIPER_NET_SNAP_RAW, {0.0, 1.0});
        wire::Snapshot i = f.snap("img", CALIPER_NET_KIND_IMAGE);
        i.data = wire::Section::make("data", CALIPER_NET_DTYPE_U8, {4, 4, 4}, img.data(), img.size());
        i.flags = CALIPER_NET_SNAP_RAW;
        f.send(i);
        wire::Snapshot pt = f.snap("pts", CALIPER_NET_KIND_POINTS);
        pt.data = wire::Section::make("data", CALIPER_NET_DTYPE_F32, {N, 2}, xy.data(), xy.size() * 4);
        pt.aux = wire::Section::make("aux", CALIPER_NET_DTYPE_F32, {N}, labels.data(), labels.size() * 4);
        pt.points_dims = 2;
        pt.label_count = (uint64_t)N;
        pt.stats = {0, 5, 0, 2, 0, 1, 0, 0};
        f.send(pt);
        wire::Snapshot h = f.snap("hist", CALIPER_NET_KIND_HISTOGRAM);
        h.data = wire::Section::make("data", CALIPER_NET_DTYPE_F32, {(int64_t)bins.size()}, bins.data(), bins.size() * 4);
        h.stats = {-2.0, 2.0};
        f.send(h);
        f.append(loss);
        return 0;
    });
    REQUIRE(poll_until(core, [&] {
        return caliper_core_live_view_count(core) == 5 && view_status(core, "heat").updates >= 1 &&
               view_status(core, "img").updates >= 1 && view_status(core, "pts").updates >= 1 &&
               view_status(core, "hist").updates >= 1 && view_status(core, "loss").updates >= 1;
    }));
    for (const char* n : {"heat", "img", "pts", "loss", "hist"}) {
        CAPTURE(std::string(n));
        CHECK(std::string(view_status(core, n).error).empty());
        CHECK(std::string(view_status(core, n).apply) == "live_views");
    }

    // Heatmap: byte-exact against the CPU colormap over the fixed range.
    CaliperLiveViewInfo vi = view_info(core, "heat");
    CHECK(vi.kind == CALIPER_LIVE_KIND_HEATMAP);
    CHECK(vi.width == 8);
    CHECK(vi.height == HH);
    CHECK(vi.imported == 1);   // read in place from the slot's MTLBuffer
    CHECK(readback(core, "heat", 8, HH) == map_ref(heat, 8, HH, CALIPER_NET_COLORMAP_MAGMA, 0.0f, 1.0f));
    // Image: the bytes as sent.
    CHECK(readback(core, "img", 4, 4) == img);
    // Copy counters (embed/render_stats.h): a network snapshot is copied once,
    // into its slot as it arrives, and the view then reads the slot in place.
    // A series and a histogram also copy their values at apply, as on every route.
    for (const auto& [n, bytes, per] : std::vector<std::tuple<const char*, uint64_t, uint64_t>>{
             {"heat", 8u * HH * 4u, 1}, {"img", 4u * 4u * 4u, 1}, {"pts", N * 12u + N * 4u, 1},
             {"hist", 2u * 8u * 4u, 2}, {"loss", 2u * 10u * 4u, 2}}) {
        CAPTURE(std::string(n));
        caliper_stats::ViewCopies vc{};
        REQUIRE(caliper_stats::view_copies(core, n, &vc));
        const uint64_t upd = view_status(core, n).updates;
        CHECK(upd == 1);
        CHECK(vc.total.copies == per * upd);
        CHECK(vc.total.bytes == bytes * upd);
    }
    // Points: (N, 2) arrived as the (N, 3) the view draws, in palette colours.
    CaliperLiveViewStatus ps = view_status(core, "pts");
    CHECK(ps.points_dims == 2);
    CHECK(ps.label_count == N);
    CHECK(ps.stats_count == 8);
    vi = view_info(core, "pts");
    CHECK(vi.count == N);
    CHECK(std::string(vi.path) == "points-imported");
    for (int i = 0; i < 3; ++i) caliper_core_frame(core);
    vi = view_info(core, "pts");
    REQUIRE(vi.width > 0);
    const std::vector<uint8_t> px = readback(core, "pts", vi.width, vi.height);
    std::set<uint32_t> seen;
    for (size_t i = 0; i + 4 <= px.size(); i += 4) {
        uint32_t c;
        std::memcpy(&c, &px[i], 4);
        seen.insert(c);
    }
    int hits = 0;
    for (int i = 0; i < N; ++i) hits += seen.count(caliper_live_views::kPalette[i]) ? 1 : 0;
    CHECK(hits == N);
    // Histogram and series: the values as sent.
    CHECK(values(core, "hist") == bins);
    CHECK(values(core, "loss") == loss);
    CHECK(view_status(core, "loss").series_total == loss.size());

    // Updates: an autoscaled heatmap replaces the first; the range is the stats.
    const std::vector<float> heat2 = ramp(8, HH, -3.0f);
    p.call([&](FakeProducer& f) {
        f.heatmap(heat2, HH, 8, CALIPER_NET_COLORMAP_RDBU, 0.0, 0.0,
                  CALIPER_NET_SNAP_RAW | CALIPER_NET_SNAP_AUTO_VMIN | CALIPER_NET_SNAP_AUTO_VMAX, {-3.0, 0.0});
        return 0;
    });
    REQUIRE(poll_until(core, [&] { return view_status(core, "heat").updates >= 2; }));
    CHECK(readback(core, "heat", 8, HH) == map_ref(heat2, 8, HH, CALIPER_NET_COLORMAP_RDBU, -3.0f, 0.0f));
    CHECK(view_info(core, "heat").creates == 1);   // in place

    // Series deltas append; a message past the end leaves a gap of NaN.
    p.call([&](FakeProducer& f) {
        f.append({0.5f, 0.45f});
        f.send_series(15, {0.3f, 0.2f});   // indices 12..14 never arrive
        f.series.resize(17, 0.0f);
        return 0;
    });
    std::vector<float> want = loss;
    want.push_back(0.5f);
    want.push_back(0.45f);
    for (int i = 0; i < 3; ++i) want.push_back(NAN);
    want.push_back(0.3f);
    want.push_back(0.2f);
    REQUIRE(poll_until(core, [&] { return values(core, "loss").size() == want.size(); }));
    CHECK(same_values(values(core, "loss"), want));
    CHECK(view_status(core, "loss").series_total == 17);

    // Several snapshots between two polls: the newest wins and the skipped ones count.
    p.call([&](FakeProducer& f) {
        for (int k = 1; k <= 4; ++k)
            f.heatmap(std::vector<float>((size_t)8 * HH, 0.2f * (float)k), HH, 8, CALIPER_NET_COLORMAP_VIRIDIS, 0.0,
                      1.0, CALIPER_NET_SNAP_RAW);
        return 0;
    });
    REQUIRE(viewer_caught_up(p));   // all four are in the viewer's slots before the next poll
    REQUIRE(poll_until(core, [&] {
        return readback(core, "heat", 8, HH) ==
               map_ref(std::vector<float>((size_t)8 * HH, 0.8f), 8, HH, CALIPER_NET_COLORMAP_VIRIDIS, 0.0f, 1.0f);
    }));
    CHECK(view_status(core, "heat").skipped >= 1);
    st = status(core);
    CHECK(st.client_alive == 1);
    CHECK(st.errors == 0);
    CHECK(p.call([](FakeProducer& f) { return f.errors.size(); }) == 0);   // the viewer had no complaint

    // The producer ends: the views keep their last pixels.
    p.call([](FakeProducer& f) {
        f.send(wire::Bye{CALIPER_NET_BYE_SHUTDOWN, "the producer is exiting"});
        f.close(CALIPER_NET_CLOSE_GOING_AWAY);
        return 0;
    });
    REQUIRE(poll_until(core, [&] { return status(core).client_alive == 0; }));
    CHECK(status(core).views == 5);
    CHECK(caliper_core_live_view_count(core) == 5);
    CHECK(readback(core, "heat", 8, HH) ==
          map_ref(std::vector<float>((size_t)8 * HH, 0.8f), 8, HH, CALIPER_NET_COLORMAP_VIRIDIS, 0.0f, 1.0f));
    CHECK(same_values(values(core, "loss"), want));

    caliper_core_live_detach(core);
    CHECK(caliper_core_live_poll(core) == -1);
    caliper_core_shutdown(core);
}

TEST_CASE("live/net: refusals and bad links fail the attach with the reason") {
    std::string no_gpu;
    CaliperCore* core = net_core(64, 64, &no_gpu);
    if (!core) CALIPER_SKIP_NO_GPU("no embeddable GPU: " << no_gpu);
    FakeProducer p;
    p.start();
    const std::string port = std::to_string(p.port());
    auto refused = [&](const std::string& url) {
        CAPTURE(url);
        const int ok = caliper_core_live_attach_url(core, url.c_str());
        const std::string why = caliper_core_last_error(core);
        if (ok) caliper_core_live_detach(core);
        return ok ? std::string() : why;
    };
    std::string why = refused("ws://127.0.0.1:" + port + "/v3/stream?token=wrong-token-0123456789abc");
    CHECK(why.find("403") != std::string::npos);
    CHECK(why.find("wrong-token") == std::string::npos);   // the token is never repeated
    // Nothing listens on a port the producer's server gave back.
    uint16_t dead = 0;
    {
        caliper_net::Server s;
        std::string err;
        REQUIRE(s.listen(0, &err));
        dead = s.port();
    }
    why = refused("ws://127.0.0.1:" + std::to_string(dead) + "/v3/stream?token=" + p.token);
    CHECK(why.find("cannot connect") != std::string::npos);
    why = refused("ws://192.168.1.2:" + port + "/v3/stream?token=" + p.token);
    CHECK(why.find("loopback") != std::string::npos);
    why = refused("ws://127.0.0.1:" + port + "/v3/stream?token=short");
    CHECK(why.find("token") != std::string::npos);
    // What the app's attach pane and caliper-viewer print for a bad link
    // (caliper_core_last_error) never repeats its token.
    for (const std::string bad : {std::string("ws:///v3/stream?token="), std::string("ws://127.0.0.1:99999/v3/stream?token="),
                                  "ftp://127.0.0.1:" + port + "/v3/stream?token=", std::string("ws://[::1]:9/v3/stream?token=")}) {
        why = refused(bad + p.token);
        CHECK_FALSE(why.empty());
        CHECK(why.find(p.token) == std::string::npos);
    }
    CHECK(caliper_core_live_attach_url(core, nullptr) == 0);
    CHECK(caliper_core_live_poll(core) == -1);   // nothing attached

    // The page link that caliper.serve() prints attaches as well.
    REQUIRE_MESSAGE(caliper_core_live_attach_url(core, ("http://127.0.0.1:" + port + "/?token=" + p.token).c_str()) == 1,
                    caliper_core_last_error(core));
    CHECK(poll_until(core, [&] { return status(core).views == 5; }));   // listed from the first poll
    caliper_core_live_detach(core);
    // Detaching says goodbye before it closes.
    REQUIRE(poll_until(core, [&] {
        return p.call([](FakeProducer& f) { return f.byes.size() == 1 && f.conn == 0; });
    }));
    CHECK(p.call([](FakeProducer& f) { return f.byes[0]; }) == CALIPER_NET_BYE_NORMAL);

    // A named session is attached with ATTACH when the server does not attach by itself.
    p.call([](FakeProducer& f) {
        f.auto_attach = false;
        return 0;
    });
    REQUIRE_MESSAGE(caliper_core_live_attach_url(
                        core, (p.url() + "&session=" + p.id.id).c_str()) == 1, caliper_core_last_error(core));
    CHECK(status(core).pending == 0);
    CHECK(poll_until(core, [&] { return status(core).views == 5; }));
    caliper_core_shutdown(core);
}

TEST_CASE("live/net: after a dropped connection the stream reconnects; views and the series continue") {
    std::string no_gpu;
    CaliperCore* core = net_core(64, 64, &no_gpu);
    if (!core) CALIPER_SKIP_NO_GPU("no embeddable GPU: " << no_gpu);
    FakeProducer p;
    p.start();
    REQUIRE_MESSAGE(caliper_core_live_attach_url(core, p.url().c_str()) == 1, caliper_core_last_error(core));
    REQUIRE(poll_until(core, [&] { return p.call([](FakeProducer& f) { return f.sub_of.size(); }) == 5; }));
    const std::vector<float> first = {1, 2, 3, 4, 5};
    p.call([&](FakeProducer& f) {
        f.append(first);
        f.heatmap(ramp(8, 4, 1.0f), 4, 8, 0, 0.0, 1.0, CALIPER_NET_SNAP_RAW);
        return 0;
    });
    REQUIRE(poll_until(core, [&] { return values(core, "loss").size() == 5 && view_status(core, "heat").updates == 1; }));

    // The connection drops without a goodbye; the producer keeps appending.
    p.call([](FakeProducer& f) {
        f.close(CALIPER_NET_CLOSE_GOING_AWAY);
        f.series.insert(f.series.end(), {6, 7, 8});
        return 0;
    });
    REQUIRE(poll_until(core, [&] { return status(core).client_alive == 0; }));
    REQUIRE(poll_until(core, [&] {
        return p.call([](FakeProducer& f) { return f.hellos == 2 && f.sub_of.size() == 5; }) &&
               status(core).client_alive == 1;
    }));
    // The backfill of the new subscription overlaps what the view holds and adds what it missed.
    REQUIRE(poll_until(core, [&] { return values(core, "loss").size() == 8; }));
    CHECK(values(core, "loss") == std::vector<float>{1, 2, 3, 4, 5, 6, 7, 8});
    // Same session: the views kept their identity (no rebuild) and update again.
    CHECK(status(core).views == 5);
    CHECK(caliper_core_live_view_count(core) == 2);   // the two that received data
    p.call([&](FakeProducer& f) {
        f.heatmap(ramp(8, 4, 0.5f), 4, 8, 0, 0.0, 1.0, CALIPER_NET_SNAP_RAW);
        return 0;
    });
    REQUIRE(poll_until(core, [&] { return view_status(core, "heat").updates == 2; }));
    CHECK(view_info(core, "heat").creates == 1);
    CHECK(readback(core, "heat", 8, 4) == map_ref(ramp(8, 4, 0.5f), 8, 4, 0, 0.0f, 1.0f));
    caliper_core_shutdown(core);
}

// Protocol 3.1: the native source announces `controls` and `selection`. A
// script's CONTROLS, its CONTROL_REPLYs and SELECTIONs reach the page's
// RunControls (live_controls.h), and its requests reach the script. The 3.1
// golden frames, sent at any moment, never disturb the stream.
TEST_CASE("live/net: the native source announces controls and selection, and a script's controls reach the page") {
    std::string no_gpu;
    CaliperCore* core = net_core(64, 64, &no_gpu);
    if (!core) CALIPER_SKIP_NO_GPU("no embeddable GPU: " << no_gpu);
    FakeProducer p;
    p.welcome_caps = {CALIPER_NET_CAP_EVENTS, CALIPER_NET_CAP_CONTROLS, CALIPER_NET_CAP_SELECTION};
    p.welcome_control = true;
    p.script31 = true;
    p.start();
    REQUIRE_MESSAGE(caliper_core_live_attach_url(core, p.url().c_str()) == 1, caliper_core_last_error(core));
    caliper_live::RunControls* rc = caliper_live_controls(core);
    REQUIRE(rc != nullptr);
    REQUIRE(poll_until(core, [&] { return rc->have() && rc->state().run_state == "running"; }));
    CHECK(p.call([](FakeProducer& f) { return f.hello_caps; }) ==
          std::vector<std::string>{CALIPER_NET_CAP_CONTROLS, CALIPER_NET_CAP_SELECTION, CALIPER_NET_CAP_SCENES});
    CHECK(rc->access().allowed);
    CHECK(rc->why_not(CALIPER_NET_OP_PAUSE).empty());
    REQUIRE(rc->state().params.size() == 1);
    CHECK(rc->state().params[0].name == "lr");
    CHECK(rc->state().params[0].log);
    CHECK(rc->state().params[0].max == doctest::Approx(0.1));
    REQUIRE(rc->state().actions.size() == 1);
    CHECK(rc->state().has_step);
    CHECK(rc->state().step == 7);

    // Pause, then a value above the range (clamped), then an action.
    CHECK(rc->pause());
    REQUIRE(poll_until(core, [&] { return rc->state().run_state == "paused"; }));
    CHECK(rc->set("lr", Json(0.5)));
    CHECK(rc->param_waiting(rc->state().params[0]));
    REQUIRE(poll_until(core, [&] { return !rc->param_waiting(rc->state().params[0]); }));
    CHECK(rc->state().params[0].value.as_double() == doctest::Approx(0.1));
    CHECK(rc->run("save"));
    CHECK(rc->action_busy(rc->state().actions[0]));
    REQUIRE(poll_until(core, [&] { return rc->action_result("save") != nullptr; }));
    CHECK(rc->action_result("save")->text == "saved checkpoint 1");
    CHECK_FALSE(rc->action_result("save")->error);
    CHECK_FALSE(rc->action_busy(rc->state().actions[0]));
    // A refusal is shown and changes nothing.
    CHECK(rc->set("lr", Json("fast")));
    REQUIRE(poll_until(core, [&] { return !rc->notice().text.empty(); }));
    CHECK(rc->notice().text.find("bad_value") != std::string::npos);
    CHECK(rc->state().params[0].value.as_double() == doctest::Approx(0.1));
    const auto in = p.call([](FakeProducer& f) { return f.controls_in; });
    REQUIRE(in.size() == 4);
    CHECK(in[0].op == "pause");
    CHECK(in[1].op == "set");
    CHECK(in[1].value.as_double() == doctest::Approx(0.5));
    CHECK(in[2].op == "run");
    CHECK(*in[2].name == "save");
    CHECK(in[0].request_id != in[1].request_id);

    // The golden frames of 3.1 at any moment: tolerated, the views keep drawing.
    p.call([](FakeProducer& f) {
        for (const char* name : {"controls", "control_reply_queued", "selection", "selection_clear"}) {
            std::ifstream in(std::string(CALIPER_TEST_SOURCE_ROOT) + "/tests/fixtures/wire_v3/v3_1/" + name + ".bin",
                             std::ios::binary);
            f.send_raw(std::vector<uint8_t>((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>()));
        }
        f.heatmap(ramp(8, 4, 1.0f), 4, 8, 0, 0.0, 1.0, CALIPER_NET_SNAP_RAW);
        return 0;
    });
    REQUIRE(poll_until(core, [&] { return view_status(core, "heat").updates >= 1; }));
    REQUIRE(viewer_caught_up(p));   // an ERROR or a close for any frame above would be in by now
    caliper_core_live_poll(core);
    CHECK(status(core).client_alive == 1);
    CHECK(p.call([](FakeProducer& f) { return f.errors.empty() && f.closes.empty() && f.hellos == 1; }));
    CHECK(readback(core, "heat", 8, 4) == map_ref(ramp(8, 4, 1.0f), 8, 4, 0, 0.0f, 1.0f));
    caliper_core_shutdown(core);
}

// Who may control: a server that does not let this viewer control (WELCOME's
// `control` false, as a hub without --allow-control) or a script without
// `controls` leaves every control off with the reason, and nothing is sent.
TEST_CASE("live/net: without WELCOME's control, or without the script's controls, the page sends no control") {
    std::string no_gpu;
    CaliperCore* core = net_core(64, 64, &no_gpu);
    if (!core) CALIPER_SKIP_NO_GPU("no embeddable GPU: " << no_gpu);
    {
        FakeProducer p;
        p.welcome_caps = {CALIPER_NET_CAP_CONTROLS, CALIPER_NET_CAP_SELECTION};
        p.welcome_control = false;
        p.script31 = true;
        p.start();
        REQUIRE_MESSAGE(caliper_core_live_attach_url(core, p.url().c_str()) == 1, caliper_core_last_error(core));
        caliper_live::RunControls* rc = caliper_live_controls(core);
        REQUIRE(poll_until(core, [&] { return rc->have(); }));
        CHECK_FALSE(rc->access().allowed);
        CHECK(rc->why_not(CALIPER_NET_OP_PAUSE).find("--allow-control") != std::string::npos);
        CHECK_FALSE(rc->pause());
        CHECK_FALSE(rc->set("lr", Json(0.01)));
        CHECK(rc->notice().text.find("pause") == std::string::npos);   // the last refusal is the set
        CHECK(rc->notice().text.find("set lr") != std::string::npos);
        poll_until(core, [] { return false; }, 200);
        CHECK(p.call([](FakeProducer& f) { return f.controls_in.size(); }) == 0);
        caliper_core_live_detach(core);
    }
    {
        FakeProducer p;   // a 3.1 server whose script registered no controls
        p.welcome_caps = {CALIPER_NET_CAP_SELECTION};
        p.welcome_control = true;
        p.start();
        REQUIRE_MESSAGE(caliper_core_live_attach_url(core, p.url().c_str()) == 1, caliper_core_last_error(core));
        caliper_live::RunControls* rc = caliper_live_controls(core);
        REQUIRE(rc != nullptr);
        REQUIRE(poll_until(core, [&] { return view_status(core, "heat").name[0] != 0; }));
        CHECK_FALSE(rc->have());
        CHECK(rc->why_not(CALIPER_NET_OP_SET).find("server takes no controls") != std::string::npos);
        CHECK_FALSE(rc->step());
        poll_until(core, [] { return false; }, 100);
        CHECK(p.call([](FakeProducer& f) { return f.controls_in.size(); }) == 0);
    }
    caliper_core_shutdown(core);
}

// A data view request goes out on the view's own subscription from the frame
// thread without waiting, and the producer's resolved form is kept with the
// view's offer from VIEWS (what a data view picker reads).
TEST_CASE("live/net: request_data_view sends VIEW_REQUEST on the view's subscription; the resolved form is kept") {
    std::string no_gpu;
    CaliperCore* core = net_core(64, 64, &no_gpu);
    if (!core) CALIPER_SKIP_NO_GPU("no embeddable GPU: " << no_gpu);
    FakeProducer p;
    p.start();
    REQUIRE_MESSAGE(caliper_core_live_attach_url(core, p.url().c_str()) == 1, caliper_core_last_error(core));
    REQUIRE(poll_until(core, [&] { return p.call([](FakeProducer& f) { return f.sub_of.size(); }) == 5; }));
    caliper_live::Source* src = caliper_live_stream(core);
    REQUIRE(src != nullptr);
    const uint64_t key = caliper_live_view_key(core, "heat");
    REQUIRE(key != 0);
    CHECK(caliper_live_view_key(core, "nope") == 0);
    std::string err;
    CHECK_FALSE(src->request_data_view(key, "[1]", &err));
    CHECK(err.find("JSON object") != std::string::npos);
    CHECK_FALSE(src->request_data_view(key + 999, "{}", &err));
    REQUIRE(src->request_data_view(key, R"({"index":{"rows":[0,2]}})", &err));
    REQUIRE(poll_until(core, [&] { return p.call([](FakeProducer& f) { return f.view_requests.size(); }) == 1; }));
    const auto vr = p.call([](FakeProducer& f) { return f.view_requests[0]; });
    CHECK(vr.view == "heat");
    CHECK(vr.sub == p.call([](FakeProducer& f) { return f.sub_of.at("heat"); }));
    CHECK(vr.data_view.find("index")->find("rows")->size() == 2);
    Json resolved, offer;
    REQUIRE(poll_until(core, [&] {
        return src->data_view_of(key, &resolved, &offer) && resolved.is_object() && resolved.has("index");
    }));
    CHECK(resolved.find("source")->as_string() == "src");
    CHECK(offer.find("name")->as_string() == "heat");
    CHECK(offer.find("sources")->at(0).find("dims")->size() == 2);
    caliper_core_shutdown(core);
}

// Linked selection: a select event of a view whose axis indexes an item space
// carries `space` and `items` (wire protocol 3.1, "Selecting"): the source rows
// of a heatmap, the row ids of points, the steps of a series; a clear clears
// the space. The script's SELECTION then reaches the page with its details.
TEST_CASE("live/net: a select event names its item space and items, and the SELECTION comes back with details") {
    std::string no_gpu;
    CaliperCore* core = net_core(96, 64, &no_gpu);
    if (!core) CALIPER_SKIP_NO_GPU("no embeddable GPU: " << no_gpu);
    FakeProducer p;
    p.welcome_caps = {CALIPER_NET_CAP_CONTROLS, CALIPER_NET_CAP_SELECTION};
    p.welcome_control = false;   // a selection is not a control
    p.script31 = true;
    p.start();
    REQUIRE_MESSAGE(caliper_core_live_attach_url(core, p.url().c_str()) == 1, caliper_core_last_error(core));
    caliper_live::RunControls* rc = caliper_live_controls(core);
    REQUIRE(poll_until(core, [&] { return p.call([](FakeProducer& f) { return f.sub_of.size(); }) == 5; }));
    p.call([](FakeProducer& f) {
        f.heatmap(ramp(8, 4, 1.0f), 4, 8, 0, 0.0, 1.0, CALIPER_NET_SNAP_RAW);
        return 0;
    });
    REQUIRE(poll_until(core, [&] { return view_status(core, "heat").updates >= 1; }));
    // The item axes the views declared.
    REQUIRE(poll_until(core, [&] { return rc->item_axes().size() == 3; }));

    // Events as the view layer reports them, through the core's event sink.
    auto emit = [&](const char* view, const char* detail) {
        caliper_live_views::LiveEvent e;
        e.view = view;
        e.type = "select";
        e.detail = detail;
        REQUIRE(core->live_views);
        REQUIRE(core->live_views->events);
        core->live_views->events(e);
    };
    emit("heat", R"({"rows":[1,3],"cols":[0,8]})");
    REQUIRE(poll_until(core, [&] { return p.call([](FakeProducer& f) { return f.events.size(); }) == 1; }));
    REQUIRE(poll_until(core, [&] { return rc->details() != nullptr; }));
    const caliper_live::LinkedSelection* d = rc->details();
    CHECK(d->space == "samples");
    CHECK(d->view == "heat");
    CHECK(d->items == std::vector<uint64_t>{1, 2});
    CHECK(d->details.find("fields")->size() == 2);
    CHECK(d->details.find("values")->at(1).at(1).as_string() == "sample 2");

    emit("pts", R"({"indices":[0,1,2],"row_ids":[4,2,4]})");
    emit("loss", R"({"steps":[2,5]})");
    emit("hist", R"({"range":[0.0,1.0]})");   // no item space: unchanged
    emit("heat", "{}");                        // clears "samples"
    REQUIRE(poll_until(core, [&] { return p.call([](FakeProducer& f) { return f.events.size(); }) == 5; }));
    const auto ev = p.call([](FakeProducer& f) { return f.events; });
    auto items_of = [](const wire::Event& e) {
        std::vector<uint64_t> v;
        if (const Json* it = e.detail.find("items"))
            for (const Json& x : it->items()) v.push_back((uint64_t)x.as_double());
        return v;
    };
    CHECK(ev[0].detail.find("space")->as_string() == "samples");
    CHECK(ev[0].sub.has_value());
    CHECK(ev[0].data_view.is_object());
    CHECK(ev[1].detail.find("space")->as_string() == "samples");
    CHECK(items_of(ev[1]) == std::vector<uint64_t>{2, 4});
    CHECK(ev[1].detail.find("row_ids")->size() == 3);   // the view layer's keys stay
    CHECK(ev[2].detail.find("space")->as_string() == "step");
    CHECK(items_of(ev[2]) == std::vector<uint64_t>{2, 3, 4});
    CHECK(ev[3].detail.find("space") == nullptr);
    CHECK(ev[4].detail.find("space")->as_string() == "samples");
    CHECK(ev[4].detail.find("items")->size() == 0);
    REQUIRE(poll_until(core, [&] {
        for (const auto& s : rc->selections())
            if (s.space == "step" && s.items.size() == 3) return true;
        return false;
    }));
    // "samples" was cleared last; its details go with it.
    REQUIRE(poll_until(core, [&] {
        for (const auto& s : rc->selections())
            if (s.space == "samples") return s.items.empty();
        return false;
    }));
    CHECK(rc->details() == nullptr);
    caliper_core_shutdown(core);
}

TEST_CASE("live/net: a bad frame closes with the decoder's code, a bad message is answered with ERROR") {
    std::string no_gpu;
    LogLines log;
    CaliperCore* core = net_core(64, 64, &no_gpu, &log);
    if (!core) CALIPER_SKIP_NO_GPU("no embeddable GPU: " << no_gpu);
    FakeProducer p;
    p.start();
    REQUIRE_MESSAGE(caliper_core_live_attach_url(core, p.url().c_str()) == 1, caliper_core_last_error(core));
    REQUIRE(poll_until(core, [&] { return p.call([](FakeProducer& f) { return f.sub_of.size(); }) == 5; }));

    // A snapshot without its data section: well formed, invalid -> ERROR bad_request.
    p.call([](FakeProducer& f) {
        Json meta = Json::object();
        meta.set("sub", f.sub_of.at("heat"));
        meta.set("view", "heat");
        meta.set("seq", 1);
        meta.set("stamp_ns", 1);
        meta.set("kind", 1);
        meta.set("data_view", Json::object());
        std::vector<uint8_t> b;
        wire::encode_frame(CALIPER_NET_MSG_SNAPSHOT, meta, {}, &b);
        f.send_raw(b);
        return 0;
    });
    REQUIRE(poll_until(core, [&] { return p.call([](FakeProducer& f) { return f.errors.size(); }) == 1; }));
    CHECK(p.call([](FakeProducer& f) { return f.errors[0]; }) == CALIPER_NET_ERR_BAD_REQUEST);
    CHECK(status(core).client_alive == 1);   // the connection stays

    // A frame with another major version: the viewer closes with 4001 and
    // does not come back (the server speaks another protocol).
    p.call([](FakeProducer& f) {
        std::vector<uint8_t> b;
        wire::encode(wire::Ping{1}, &b);
        b[4] = 4;
        f.send_raw(b);
        return 0;
    });
    REQUIRE(poll_until(core, [&] {
        return p.call([](FakeProducer& f) { return !f.closes.empty(); });
    }));
    CHECK(p.call([](FakeProducer& f) { return f.closes.back(); }) == CALIPER_NET_CLOSE_UNSUPPORTED_MAJOR);
    REQUIRE(poll_until(core, [&] { return status(core).client_alive == 0; }));
    // The source gave up for good: it says why, and its thread then ends
    // without another attempt, whatever its back-off.
    REQUIRE(poll_until(core, [&] { return log.has("live: the server speaks another major version of the protocol"); }));
    CHECK(p.call([](FakeProducer& f) { return f.hellos; }) == 1);   // no reconnect

    // A text message closes with 1003.
    caliper_core_live_detach(core);
    REQUIRE_MESSAGE(caliper_core_live_attach_url(core, p.url().c_str()) == 1, caliper_core_last_error(core));
    p.call([](FakeProducer& f) {
        f.send_raw({'h', 'i'}, /*binary=*/false);
        return 0;
    });
    REQUIRE(poll_until(core, [&] {
        return p.call([](FakeProducer& f) { return f.closes.size() >= 2; });
    }));
    CHECK(p.call([](FakeProducer& f) { return f.closes.back(); }) == CALIPER_NET_CLOSE_UNSUPPORTED_DATA);
    caliper_core_shutdown(core);
}

TEST_CASE("live/app: Attachment::attach_url through a hosted core, as the Caliper app attaches") {
    caliper_host::core_create_ui_context();
    std::unique_ptr<caliper_host::HostRenderer> renderer = caliper_host::make_metal_renderer();
    const bool ok = !caliper_test::force_no_gpu() && renderer &&
                    renderer->canvas_init(nullptr, caliper_host::HostRenderer::CANVAS_OFFSCREEN, 320, 240);
    caliper_host::services_set_renderer(renderer.get());
    struct Cleanup {
        std::unique_ptr<caliper_host::HostRenderer>& r;
        ~Cleanup() {
            caliper_host::services_set_renderer(nullptr);
            if (r) r->canvas_shutdown();
            r.reset();
            caliper_host::core_destroy_ui_context();
        }
    } cleanup{renderer};
    if (!ok) CALIPER_SKIP_NO_GPU("no Metal device for the host's offscreen canvas");

    FakeProducer p;
    p.start();
    caliper_live::Attachment a;
    std::string err;
    CHECK_FALSE(a.attach_url(renderer.get(), "ws://127.0.0.1:" + std::to_string(p.port()) +
                                                 "/v3/stream?token=wrong-token-0123456789abc", &err));
    CHECK(err.find("403") != std::string::npos);
    CHECK_FALSE(a.attached());
    REQUIRE_MESSAGE(a.attach_url(renderer.get(), p.url(), &err), err);
    CHECK(a.status().attached);
    CHECK(a.status().session == "clp-test-4242");
    CHECK(a.status().script == "fake_train.py");
    p.call([&](FakeProducer& f) {
        f.append({3.0f, 2.0f, 1.0f});
        return 0;
    });
    bool got = false;
    for (int i = 0; i < 2000 && !got; ++i) {
        a.poll();
        ImGuiIO& io = ImGui::GetIO();
        io.DisplayFramebufferScale = ImVec2(1.0f, 1.0f);
        io.DisplaySize = ImVec2(320.0f, 240.0f);
        io.DeltaTime = 1.0f / 60.0f;
        renderer->canvas_new_frame();
        a.draw(0.0f, 0.0f, 320.0f, 240.0f, 1.0f);
        renderer->canvas_render();
        got = values(a.core(), "loss") == std::vector<float>{3.0f, 2.0f, 1.0f};
        if (!got) std::this_thread::sleep_for(std::chrono::milliseconds(2));
    }
    CHECK(got);
    CHECK(a.status().views == 5);
    a.detach();
    CHECK_FALSE(a.attached());
}

// The run bar as the app draws it, driven by real ImGui input (live_ui_driver.h):
// clicks on Pause and on an action, a typed parameter value and a shift-drag
// across a heatmap reach the script, and the bar shows what the script echoes.
// Then, against a server that refuses controls, the same clicks send nothing
// while a selection still goes out.
namespace {
struct HostFrame {
    std::unique_ptr<caliper_host::HostRenderer> renderer;
    bool ok = false;
    HostFrame() {
        caliper_host::core_create_ui_context();
        renderer = caliper_host::make_metal_renderer();
        ok = !caliper_test::force_no_gpu() && renderer &&
             renderer->canvas_init(nullptr, caliper_host::HostRenderer::CANVAS_OFFSCREEN, 1000, 480);
        caliper_host::services_set_renderer(renderer.get());
    }
    ~HostFrame() {
        caliper_host::services_set_renderer(nullptr);
        if (renderer) renderer->canvas_shutdown();
        renderer.reset();
        caliper_host::core_destroy_ui_context();
    }
    // One frame of a live page: the run bar on top, the views below, the
    // shared selection over them. The driver's input goes in first.
    void frame(caliper_live::Attachment& a, caliper_live::UiDriver* driver) {
        a.poll();
        if (driver) driver->feed(a.core(), a.controls());
        ImGuiIO& io = ImGui::GetIO();
        io.DisplayFramebufferScale = ImVec2(1.0f, 1.0f);
        io.DisplaySize = ImVec2(1000.0f, 480.0f);
        io.DeltaTime = 1.0f / 60.0f;
        renderer->canvas_new_frame();
        const caliper_live::ControlsLook look;
        if (caliper_live::RunControls* rc = a.controls()) {
            ImGui::SetNextWindowPos(ImVec2(0, 0));
            ImGui::SetNextWindowSize(ImVec2(1000, 40));
            ImGui::Begin("##bar", nullptr, ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_NoSavedSettings);
            caliper_live::draw_run_bar(*rc, look, 980.0f);
            ImGui::End();
        }
        a.draw(0.0f, 44.0f, 1000.0f, 436.0f, 1.0f);
        renderer->canvas_render();
    }
};
}  // namespace

TEST_CASE("live/app: clicks on the run bar and a shift-drag in a view reach the script; the bar shows its echo") {
    HostFrame host;
    if (!host.ok) CALIPER_SKIP_NO_GPU("no Metal device for the host's offscreen canvas");
    FakeProducer p;
    p.welcome_caps = {CALIPER_NET_CAP_CONTROLS, CALIPER_NET_CAP_SELECTION};
    p.welcome_control = true;
    p.script31 = true;
    p.start();
    caliper_live::Attachment a;
    std::string err;
    REQUIRE_MESSAGE(a.attach_url(host.renderer.get(), p.url(), &err), err);
    REQUIRE(a.controls() != nullptr);
    // The heatmap's rows index "samples"; it needs pixels to be selected in.
    caliper_live::UiDriver driver("state running; click pause; state paused; set lr 0.02; wait 5; "
                                  "click action:save; wait 5; select heat; wait 5");
    bool sent_heat = false;
    std::vector<std::string> steps;
    for (int i = 0; i < 3000 && !driver.done(); ++i) {
        if (!sent_heat && p.call([](FakeProducer& f) { return f.sub_of.count("heat"); })) {
            p.call([](FakeProducer& f) {
                f.heatmap(ramp(8, 4, 1.0f), 4, 8, 0, 0.0, 1.0, 0);   // with chrome: RAW views take no gestures
                return 0;
            });
            sent_heat = true;
        }
        host.frame(a, &driver);
        for (auto& line : driver.take_log()) steps.push_back(line);
        std::this_thread::sleep_for(std::chrono::milliseconds(2));
    }
    INFO("driver: " << driver.error());
    for (const auto& s : steps) INFO(s);
    REQUIRE(driver.done());
    REQUIRE(driver.error().empty());
    caliper_live::RunControls* rc = a.controls();
    // Let the last echoes arrive.
    const auto echoes_by = std::chrono::steady_clock::now() + std::chrono::seconds(10);
    while (!(rc->details() && rc->action_result("save")) && std::chrono::steady_clock::now() < echoes_by) {
        host.frame(a, nullptr);
        std::this_thread::sleep_for(std::chrono::milliseconds(2));
    }
    REQUIRE(rc->details());
    REQUIRE(rc->action_result("save"));
    const auto in = p.call([](FakeProducer& f) { return f.controls_in; });
    REQUIRE(in.size() == 3);
    CHECK(in[0].op == "pause");
    CHECK(in[1].op == "set");
    CHECK(*in[1].name == "lr");
    CHECK(in[1].value.as_double() == doctest::Approx(0.02));
    CHECK(in[2].op == "run");
    CHECK(*in[2].name == "save");
    // The echo: paused, the value confirmed, the action's result.
    CHECK(rc->state().run_state == "paused");
    CHECK(rc->state().params[0].value.as_double() == doctest::Approx(0.02));
    CHECK_FALSE(rc->param_waiting(rc->state().params[0]));
    REQUIRE(rc->action_result("save") != nullptr);
    CHECK(rc->action_result("save")->text == "saved checkpoint 1");
    // The shift-drag selected the top rows of "heat" in source indices.
    const auto ev = p.call([](FakeProducer& f) { return f.events; });
    const wire::Event* sel = nullptr;
    for (const auto& e : ev)
        if (e.type == "select" && e.view == "heat") sel = &e;
    REQUIRE(sel != nullptr);
    CHECK(sel->detail.find("space")->as_string() == "samples");
    REQUIRE(sel->detail.find("items")->size() >= 1);
    CHECK(sel->detail.find("items")->at(0).as_double() == 0.0);
    REQUIRE(rc->details() != nullptr);
    CHECK(rc->details()->space == "samples");
    CHECK(rc->details()->view == "heat");
    a.detach();
}

TEST_CASE("live/app: against a server that refuses controls the bar's clicks send nothing; a selection still goes") {
    HostFrame host;
    if (!host.ok) CALIPER_SKIP_NO_GPU("no Metal device for the host's offscreen canvas");
    FakeProducer p;
    p.welcome_caps = {CALIPER_NET_CAP_CONTROLS, CALIPER_NET_CAP_SELECTION};
    p.welcome_control = false;
    p.script31 = true;
    p.start();
    caliper_live::Attachment a;
    std::string err;
    REQUIRE_MESSAGE(a.attach_url(host.renderer.get(), p.url(), &err), err);
    caliper_live::UiDriver driver("state running; click pause; wait 10; click step; wait 10; "
                                  "click action:save; wait 10; select heat; wait 10");
    bool sent_heat = false;
    for (int i = 0; i < 3000 && !driver.done(); ++i) {
        if (!sent_heat && p.call([](FakeProducer& f) { return f.sub_of.count("heat"); })) {
            p.call([](FakeProducer& f) {
                f.heatmap(ramp(8, 4, 1.0f), 4, 8, 0, 0.0, 1.0, 0);   // with chrome: RAW views take no gestures
                return 0;
            });
            sent_heat = true;
        }
        host.frame(a, &driver);
        std::this_thread::sleep_for(std::chrono::milliseconds(2));
    }
    INFO("driver: " << driver.error());
    REQUIRE(driver.done());
    REQUIRE(driver.error().empty());
    caliper_live::RunControls* rc = a.controls();
    // The selection's details come back after the select event, so a control
    // sent before it would have reached the script by then.
    const auto details_by = std::chrono::steady_clock::now() + std::chrono::seconds(10);
    while (!rc->details() && std::chrono::steady_clock::now() < details_by) {
        host.frame(a, nullptr);
        std::this_thread::sleep_for(std::chrono::milliseconds(2));
    }
    REQUIRE(rc->details());
    CHECK(p.call([](FakeProducer& f) { return f.controls_in.size(); }) == 0);
    CHECK(rc->state().run_state == "running");
    CHECK(rc->why_not(CALIPER_NET_OP_PAUSE).find("does not let this viewer control") != std::string::npos);
    CHECK(rc->counts().sent == 0);
    REQUIRE(rc->details() != nullptr);
    CHECK(rc->details()->space == "samples");
    a.detach();
}

// ---- RunControls on a scripted source (no network, no GPU) -----------------------------------
namespace {
// A Source whose 3.1 side the test scripts: what it allows, its CONTROLS, the
// replies it hands over, and every CONTROL it was asked to send (with the
// test's clock at the time).
class ScriptedSource : public caliper_live::Source {
public:
    caliper_live::ControlAccess access{true, ""};
    bool have = true;
    caliper_live::Controls ctl;
    std::vector<caliper_live::ControlReply> replies;
    struct Sent {
        std::string op, name;
        Json value;
        uint32_t id = 0;
        uint64_t at_ns = 0;
    };
    std::vector<Sent> sent;
    uint64_t clock_ns = 0;
    std::vector<caliper_live::LinkedSelection> sel;
    uint64_t sel_version = 0;

    std::string label() const override { return "scripted"; }
    std::string script() const override { return "train.py"; }
    uint64_t start_unix_ms() const override { return 0; }
    int32_t producer_pid() const override { return 0; }
    bool producer_alive() override { return true; }
    bool sync(std::vector<caliper_live::SourceView>*) override { return false; }
    bool take(uint64_t, uint64_t, caliper_live::Snapshot*) override { return false; }
    void release_prev(uint64_t) override {}
    void unclaim(uint64_t) override {}
    void drew(uint64_t, uint64_t, uint64_t) override {}
    uint64_t producer_dropped(uint64_t) override { return 0; }
    void heartbeat() override {}

    bool controls(caliper_live::Controls* out) override {
        if (!have) return false;
        *out = ctl;
        return true;
    }
    caliper_live::ControlAccess control_access() override { return access; }
    uint32_t send_control(const std::string& op, const std::string& name, const Json& value, uint32_t,
                          std::string*) override {
        sent.push_back({op, name, value, (uint32_t)sent.size() + 1, clock_ns});
        return (uint32_t)sent.size();
    }
    std::vector<caliper_live::ControlReply> take_control_replies() override {
        std::vector<caliper_live::ControlReply> out;
        out.swap(replies);
        return out;
    }
    uint64_t selections(std::vector<caliper_live::LinkedSelection>* out) override {
        *out = sel;
        return sel_version;
    }

    // A reply to request `id`, as it arrives after the current CONTROLS.
    void reply(uint32_t id, const char* status, Json value = Json(), const char* code = "", std::string message = "") {
        caliper_live::ControlReply r;
        r.request_id = id;
        r.op = sent.at(id - 1).op;
        r.name = sent.at(id - 1).name;
        r.status = status;
        r.value = value;
        r.message = std::move(message);
        if (*code) {
            r.error_code = code;
            r.error_message = "refused";
        }
        r.controls_version = ctl.version;
        replies.push_back(r);
    }
    // A new CONTROLS with lr's value and pending value.
    void controls_now(const char* state, double lr, Json pending = Json()) {
        ctl.run_state = state;
        ctl.params.resize(1);
        ctl.params[0].name = "lr";
        ctl.params[0].type = CALIPER_NET_PARAM_NUMBER;
        ctl.params[0].value = Json(lr);
        ctl.params[0].pending = pending;
        ctl.actions.resize(1);
        ctl.actions[0].name = "save";
        ++ctl.version;
    }
};

constexpr uint64_t kMs = 1000000ull;
}  // namespace

TEST_CASE("live/controls: why a control is off, and what a checkpoint unlocks") {
    ScriptedSource src;
    caliper_live::RunControls rc;
    rc.update(nullptr, 0);
    CHECK(rc.why_not(CALIPER_NET_OP_SET) == "not attached to a session");
    src.access = {false, "the session has ended"};
    src.controls_now(CALIPER_NET_RUN_RUNNING, 0.001);
    rc.update(&src, 1);
    CHECK(rc.why_not(CALIPER_NET_OP_PAUSE) == "the session has ended");
    CHECK_FALSE(rc.set("lr", Json(0.01)));
    CHECK(rc.notice().text == "set lr: the session has ended");
    CHECK(src.sent.empty());
    src.access = {true, ""};
    src.have = false;
    rc.update(&src, 2);
    CHECK(rc.why_not(CALIPER_NET_OP_PAUSE) == "the script has sent no controls");
    src.have = true;
    src.controls_now(CALIPER_NET_RUN_NONE, 0.001);
    rc.update(&src, 3);
    // Before the first checkpoint: pause, step and actions wait for it, a value may be set.
    CHECK(rc.why_not(CALIPER_NET_OP_PAUSE) == "the script has called no checkpoint yet");
    CHECK(rc.why_not(CALIPER_NET_OP_STEP) == "the script has called no checkpoint yet");
    CHECK(rc.why_not(CALIPER_NET_OP_RUN) == "the script has called no checkpoint yet");
    CHECK(rc.why_not(CALIPER_NET_OP_SET).empty());
    src.controls_now(CALIPER_NET_RUN_RUNNING, 0.001);
    rc.update(&src, 4);
    for (const char* op : {"pause", "resume", "step", "set", "run"}) CHECK(rc.why_not(op).empty());
    const auto log = rc.take_log();
    CHECK(std::find(log.begin(), log.end(), "controls off: the session has ended") != log.end());
    CHECK(std::find(log.begin(), log.end(), "controls on") != log.end());
    CHECK(std::find(log.begin(), log.end(), "run running") != log.end());
}

TEST_CASE("live/controls: set goes out at most 30 times a second, and the value where a drag stops goes last") {
    ScriptedSource src;
    src.controls_now(CALIPER_NET_RUN_RUNNING, 0.001);
    caliper_live::RunControls rc;
    // A drag: a new value every millisecond for 100 ms.
    double v = 0.0;
    for (uint64_t t = 0; t < 100; ++t) {
        src.clock_ns = t * kMs;
        rc.update(&src, src.clock_ns);
        v = 0.001 + 0.0001 * (double)t;
        CHECK(rc.set("lr", Json(v)));
    }
    // Then nothing for 50 ms: the last value goes out when its turn comes.
    for (uint64_t t = 100; t < 150; ++t) {
        src.clock_ns = t * kMs;
        rc.update(&src, src.clock_ns);
    }
    REQUIRE(src.sent.size() >= 2);
    CHECK(src.sent.size() <= 5);
    for (size_t i = 1; i < src.sent.size(); ++i)
        CHECK(src.sent[i].at_ns - src.sent[i - 1].at_ns >= 1000000000ull / CALIPER_NET_CONTROL_RATE_HZ);
    CHECK(src.sent.back().value.as_double() == doctest::Approx(v));
    CHECK(rc.counts().sets == src.sent.size());
}

TEST_CASE("live/controls: a value set here is pending until a CONTROLS after its reply confirms it") {
    ScriptedSource src;
    src.controls_now(CALIPER_NET_RUN_RUNNING, 0.001);
    caliper_live::RunControls rc;
    rc.update(&src, 0);
    CHECK(rc.set("lr", Json(0.02)));
    REQUIRE(src.sent.size() == 1);
    const caliper_live::ControlParam* lr = &rc.state().params[0];
    CHECK(rc.shown_value(*lr).as_double() == doctest::Approx(0.02));
    CHECK(rc.param_waiting(*lr));
    // A CONTROLS that crossed the request changes nothing yet.
    src.controls_now(CALIPER_NET_RUN_RUNNING, 0.001);
    rc.update(&src, 100 * kMs);
    lr = &rc.state().params[0];
    CHECK(rc.param_waiting(*lr));
    CHECK(rc.shown_value(*lr).as_double() == doctest::Approx(0.02));
    // The reply, then the CONTROLS that follows it with the script's pending value.
    src.reply(1, CALIPER_NET_STATUS_OK, Json(0.02));
    rc.update(&src, 200 * kMs);
    lr = &rc.state().params[0];
    CHECK(rc.param_waiting(*lr));
    src.controls_now(CALIPER_NET_RUN_RUNNING, 0.001, Json(0.02));
    rc.update(&src, 300 * kMs);
    lr = &rc.state().params[0];
    CHECK_FALSE(rc.param_waiting(*lr));
    CHECK(rc.param_pending(*lr));
    CHECK(rc.shown_value(*lr).as_double() == doctest::Approx(0.02));
    // The value the control already shows is not sent again.
    CHECK(rc.set("lr", Json(0.02)));
    CHECK(src.sent.size() == 1);
    // Clamped: the reply's value shows until confirmed.
    CHECK(rc.set("lr", Json(0.5)));
    src.reply(2, CALIPER_NET_STATUS_OK, Json(0.1));
    rc.update(&src, 400 * kMs);
    lr = &rc.state().params[0];
    CHECK(rc.shown_value(*lr).as_double() == doctest::Approx(0.1));
    // Refused: dropped, the script's value shows again, and the reason.
    src.controls_now(CALIPER_NET_RUN_RUNNING, 0.1);
    rc.update(&src, 500 * kMs);
    CHECK(rc.set("lr", Json("fast")));
    src.reply(3, CALIPER_NET_STATUS_ERROR, Json(), CALIPER_NET_ERR_BAD_VALUE);
    rc.update(&src, 600 * kMs);
    lr = &rc.state().params[0];
    CHECK_FALSE(rc.param_waiting(*lr));
    CHECK(rc.shown_value(*lr).as_double() == doctest::Approx(0.1));
    CHECK(rc.notice().text == "set lr: bad_value: refused");
    CHECK(rc.notice().error);
}

TEST_CASE("live/controls: an action is busy until its final reply and keeps its result; a refused pause is shown") {
    ScriptedSource src;
    src.controls_now(CALIPER_NET_RUN_RUNNING, 0.001);
    caliper_live::RunControls rc;
    rc.update(&src, 0);
    CHECK(rc.run("save"));
    CHECK(rc.action_busy(rc.state().actions[0]));
    src.reply(1, CALIPER_NET_STATUS_QUEUED);
    rc.update(&src, 1);
    CHECK(rc.action_busy(rc.state().actions[0]));
    CHECK(rc.action_result("save") == nullptr);
    src.reply(1, CALIPER_NET_STATUS_OK, Json(), "", "saved at step 9");
    rc.update(&src, 2);
    CHECK_FALSE(rc.action_busy(rc.state().actions[0]));
    REQUIRE(rc.action_result("save") != nullptr);
    CHECK(rc.action_result("save")->text == "saved at step 9");
    // The script says it runs: busy again, without a request of ours.
    src.ctl.actions[0].running = true;
    ++src.ctl.version;
    rc.update(&src, 3);
    CHECK(rc.action_busy(rc.state().actions[0]));
    src.ctl.actions[0].running = false;
    ++src.ctl.version;
    CHECK(rc.run("save"));
    src.reply(2, CALIPER_NET_STATUS_ERROR, Json(), CALIPER_NET_ERR_ACTION_FAILED);
    rc.update(&src, 4);
    REQUIRE(rc.action_result("save") != nullptr);
    CHECK(rc.action_result("save")->error);
    CHECK(rc.action_result("save")->text == "action_failed: refused");
    CHECK(rc.pause());
    src.reply(3, CALIPER_NET_STATUS_ERROR, Json(), CALIPER_NET_ERR_FORBIDDEN);
    rc.update(&src, 5);
    CHECK(rc.notice().text == "pause: forbidden: refused");
    CHECK(rc.counts().refused == 2);
}

TEST_CASE("live/controls: the details shown are those of the selection that changed last") {
    ScriptedSource src;
    caliper_live::RunControls rc;
    caliper_live::LinkedSelection a;
    a.space = "samples";
    a.view = "emb";
    a.items = {1, 2};
    a.details = Json::object();
    a.seq = 1;
    src.sel = {a};
    src.sel_version = 1;
    rc.update(&src, 0);
    REQUIRE(rc.details() != nullptr);
    CHECK(rc.details()->space == "samples");
    caliper_live::LinkedSelection b;
    b.space = "step";
    b.items = {3};
    b.seq = 2;
    src.sel = {a, b};
    src.sel_version = 2;
    rc.update(&src, 1);
    CHECK(rc.details() == nullptr);   // "step" changed last and has none
    b.details_error = "the details function timed out";
    b.seq = 3;
    src.sel = {a, b};
    src.sel_version = 3;
    rc.update(&src, 2);
    REQUIRE(rc.details() != nullptr);
    CHECK(rc.details()->space == "step");
    const auto log = rc.take_log();
    CHECK(std::find(log.begin(), log.end(), "selection samples: 2 items, with details") != log.end());
    CHECK(std::find(log.begin(), log.end(), "selection step: 1 item, details failed") != log.end());
}

// The data view picker under a view's header, clicked through ImGui's input
// queue as the run bar is: a heatmap of attention maps (head x query x key)
// shows one head; picking another head, then "all" heads, sends VIEW_REQUEST
// with the data view the menus describe, and the producer's resolved data
// view and the snapshots of its new shape (a 2 x 2 mosaic of the heads) arrive
// in the view.
TEST_CASE("live/app: the data view picker changes the head of an attention map, and the new shape arrives") {
    HostFrame host;
    if (!host.ok) CALIPER_SKIP_NO_GPU("no Metal device for the host's offscreen canvas");
    FakeProducer p;
    p.add_attention();
    p.start();
    caliper_live::Attachment a;
    std::string err;
    REQUIRE_MESSAGE(a.attach_url(host.renderer.get(), p.url(), &err), err);
    auto frames_until = [&](const std::function<bool()>& done, caliper_live::UiDriver* d = nullptr) {
        for (int i = 0; i < 3000; ++i) {
            host.frame(a, d);
            if (done()) return true;
            std::this_thread::sleep_for(std::chrono::milliseconds(2));
        }
        return false;
    };
    REQUIRE(frames_until([&] { return p.call([](FakeProducer& f) { return f.sub_of.count("attn"); }) == 1; }));
    p.call([](FakeProducer& f) {
        f.attention(FakeProducer::attn_default());
        return 0;
    });
    CaliperViewInfo vi{};
    vi.struct_size = sizeof vi;
    const bool shown = frames_until([&] { return caliper_core_view_info(a.core(), "attn", &vi) == 1 && vi.height == 8; });
    INFO("attn: updates " << view_status(a.core(), "attn").updates << ", error " << view_status(a.core(), "attn").error
                          << ", info " << vi.width << "x" << vi.height);
    REQUIRE(shown);
    CHECK(vi.width == 6);
    // The menus: one per dimension (a heatmap keeps two of three), no source menu.
    REQUIRE(frames_until([&] { return caliper_live::bar_widgets().count("picker:attn:head") == 1; }));
    CHECK(caliper_live::bar_widgets().count("picker:attn:query") == 1);
    CHECK(caliper_live::bar_widgets().count("picker:attn:source") == 0);

    caliper_live::UiDriver pick("click picker:attn:head; wait 3; click picker:attn:head=2; wait 3");
    REQUIRE(frames_until([&] { return pick.done(); }, &pick));
    INFO("driver: " << pick.error());
    REQUIRE(pick.error().empty());
    REQUIRE(frames_until([&] { return p.call([](FakeProducer& f) { return f.view_requests.size(); }) == 1; }));
    wire::ViewRequest vr = p.call([](FakeProducer& f) { return f.view_requests[0]; });
    CHECK(vr.view == "attn");
    CHECK(vr.data_view.find("source")->as_string() == "src");
    int64_t head = -1;
    REQUIRE(vr.data_view.find("index")->find("head")->get_i64(&head));
    CHECK(head == 2);
    CHECK(vr.data_view.find("tile") == nullptr);
    caliper_live::Source* src = caliper_live_stream(a.core());
    const uint64_t key = caliper_live_view_key(a.core(), "attn");
    Json resolved, offer;
    REQUIRE(frames_until([&] {
        return src->data_view_of(key, &resolved, &offer) && resolved.find("index") &&
               resolved.find("index")->find("head") && resolved.find("index")->find("head")->as_double() == 2.0;
    }));

    // All heads: the heatmap keeps three dims, so the first becomes the mosaic.
    caliper_live::UiDriver all("click picker:attn:head; wait 3; click picker:attn:head=all; wait 3");
    REQUIRE(frames_until([&] { return all.done(); }, &all));
    REQUIRE(all.error().empty());
    REQUIRE(frames_until([&] { return p.call([](FakeProducer& f) { return f.view_requests.size(); }) == 2; }));
    vr = p.call([](FakeProducer& f) { return f.view_requests[1]; });
    REQUIRE(vr.data_view.find("tile") != nullptr);
    CHECK(vr.data_view.find("tile")->as_string() == "head");
    CHECK(vr.data_view.find("index")->find("head") == nullptr);
    p.call([vr](FakeProducer& f) {
        f.attention(vr.data_view);
        return 0;
    });
    REQUIRE(frames_until([&] { return caliper_core_view_info(a.core(), "attn", &vi) == 1 && vi.height == 16; }));
    CHECK(vi.width == 12);
    REQUIRE(frames_until([&] { return src->data_view_of(key, &resolved, &offer) && resolved.find("tile"); }));
    // The view's payload map now names the mosaic's tile axis.
    const caliper_live_views::View* v = caliper_live_views::find(a.core(), "attn");
    REQUIRE(v != nullptr);
    CHECK(v->map.tile_cols == 2);
    a.detach();
}
