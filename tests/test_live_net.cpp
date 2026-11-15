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
#include "live/live_attach.h"
#include "live/live_net_source.h"      // parse_stream_url
#include "live/live_views_state.h"     // kPalette (the exact categorical colours)
#include "net/wire.h"
#include "net/ws.h"
#include "tensor_bridge.h"             // map_f32_to_rgba8 / colormap_lut (CPU reference)

#include <imgui.h>

#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstring>
#include <deque>
#include <functional>
#include <future>
#include <map>
#include <memory>
#include <mutex>
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
    std::map<std::string, uint32_t> sub_of;   // view -> its open subscription
    std::vector<float> series;                // the retained values of "loss"
    std::vector<std::string> errors;          // ERROR codes the viewer sent
    std::vector<uint16_t> closes;             // close codes of ended connections
    std::vector<std::string> byes;            // BYE reasons the viewer sent
    uint64_t seq = 0;

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
        s.data = wire::Section::make("data", CALIPER_NET_DTYPE_F32, {H, W}, v.data(), v.size() * 4);
        s.colormap = cm;
        s.vmin = vmin;
        s.vmax = vmax;
        s.flags = flags;
        s.stats = std::move(stats);
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
        v.views = views;
        send(v);
    }

    void on_message(const caliper_net::Bytes& data) {
        wire::Message m;
        const wire::DecodeResult r = wire::decode(data.data(), data.size(), &m);
        if (!r.ok()) return;
        if (std::holds_alternative<wire::Hello>(m)) {
            ++hellos;
            wire::Welcome w;
            w.role = CALIPER_NET_ROLE_PRODUCER;
            w.session = id.to_json();
            send(w);
            wire::Sessions s;
            wire::SessionInfo si{id, CALIPER_NET_SESSION_LIVE, views.size()};
            s.sessions.push(si.to_json());
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
            wire::ViewReply rep;
            rep.request_id = s->request_id;
            rep.sub = s->sub;
            rep.view = s->view;
            rep.data_view = Json::object();
            send(rep);
            if (s->view == "loss" && !series.empty()) send_series(0, series);   // the backfill
        } else if (auto* p = std::get_if<wire::Ping>(&m)) {
            send(wire::Pong{p->nonce});
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

CaliperCore* net_core(int W, int H, std::string* why) {
    CaliperCoreDesc d{};
    d.struct_size = sizeof d;
    d.renderer = CALIPER_RENDERER_DEFAULT;
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
    std::this_thread::sleep_for(std::chrono::milliseconds(100));   // all four arrive before the next poll
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

TEST_CASE("live/net: a bad frame closes with the decoder's code, a bad message is answered with ERROR") {
    std::string no_gpu;
    CaliperCore* core = net_core(64, 64, &no_gpu);
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
    std::this_thread::sleep_for(std::chrono::milliseconds(600));
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
