// The hub (src/host/net/hub.h) with the scenes subset of the protocol 3.2
// draft, in one process: a Hub on its own thread, a fake producer that serves
// one scene as caliper.serve() does (the scene's state on SUBSCRIBE: a reset,
// then its batches, each with "state"), and fake viewers with and without the
// capability "scenes". It checks that the hub forwards SCENE_UPDATE and
// SCENE_CLEAR with each viewer's sub and shared sections, keeps the newest
// batch of each entity part with clears applied and sends it as the state of
// a late viewer after a reset of its own, coalesces a slow viewer's batches per
// entity part, and hides scenes from a viewer that did not announce them.
#include <doctest/doctest.h>

#include "caliper/live_net.h"
#include "net/hub.h"
#include "net/wire.h"
#include "net/ws.h"

#include <algorithm>
#include <chrono>
#include <cstring>
#include <deque>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <functional>
#include <map>
#include <memory>
#include <string>
#include <thread>
#include <vector>

namespace wire = caliper_net::wire;
using caliper_net::Bytes;
using caliper_net::Hub;
using caliper_net::HubOptions;
using caliper_net::Json;

namespace {

const std::string kToken32 = "hub_scene_token-0123456789abcdef";

uint64_t mono_ns32() {
    using namespace std::chrono;
    return (uint64_t)duration_cast<nanoseconds>(steady_clock::now().time_since_epoch()).count();
}

struct Runner {
    std::unique_ptr<Hub> hub;
    std::thread thread;
    explicit Runner(size_t max_retained_bytes = size_t(1) << 30, bool allow_control = false,
                    size_t max_queued_bytes = size_t(160) << 20) {
        HubOptions o;
        o.token = kToken32;
        o.allow_control = allow_control;
        o.max_retained_bytes = max_retained_bytes;
        o.max_queued_bytes = max_queued_bytes;
        hub = std::make_unique<Hub>(std::move(o));
        std::string err;
        REQUIRE_MESSAGE(hub->listen(&err), err);
        thread = std::thread([this] { hub->run(); });
    }
    ~Runner() {
        hub->stop();
        thread.join();
    }
    std::string url() const { return "ws://127.0.0.1:" + std::to_string(hub->port()) + "/v3/stream?token=" + kToken32; }
};

struct In {
    std::shared_ptr<Bytes> buf;
    wire::Message m;
};

class Client {
public:
    virtual ~Client() = default;
    bool connect(const std::string& url) {
        std::string err;
        return client_.connect(url, &err);
    }
    void send(const wire::Message& m) {
        std::vector<uint8_t> b;
        REQUIRE(wire::encode(m, &b));
        client_.send(b.data(), b.size());
    }
    void poll() {
        for (auto& e : client_.poll(0)) {
            using T = caliper_net::Event::Type;
            if (e.type == T::Open) {
                open = true;
                on_open();
            } else if (e.type == T::Message) {
                In in;
                in.buf = std::make_shared<Bytes>(std::move(e.data));
                const wire::DecodeResult r = wire::decode(in.buf->data(), in.buf->size(), &in.m);
                if (!r.ok()) continue;
                if (auto* pi = std::get_if<wire::Ping>(&in.m)) send(wire::Pong{pi->nonce});
                inbox.push_back(in);
                on(inbox.back());
            }
        }
    }
    template <typename T>
    std::vector<const T*> all() const {
        std::vector<const T*> v;
        for (const auto& x : inbox)
            if (auto* p = std::get_if<T>(&x.m)) v.push_back(p);
        return v;
    }
    template <typename T>
    const T* last() const {
        for (auto it = inbox.rbegin(); it != inbox.rend(); ++it)
            if (auto* p = std::get_if<T>(&it->m)) return p;
        return nullptr;
    }
    // The scene messages of `sub`, in arrival order: SCENE_CLEAR as "clear", SCENE_UPDATE as its path.
    std::vector<std::string> scene_log(uint32_t sub, bool states_only = false) const {
        std::vector<std::string> out;
        for (const auto& x : inbox) {
            if (auto* u = std::get_if<wire::SceneUpdate>(&x.m)) {
                if (u->sub == sub && (!states_only || u->state)) out.push_back(u->path + ":" + u->part);
            } else if (auto* c = std::get_if<wire::SceneClear>(&x.m)) {
                if (c->sub == sub && (!states_only || c->state)) out.push_back(c->reset ? "reset" : "clear " + c->path);
            }
        }
        return out;
    }
    std::deque<In> inbox;
    bool open = false;

protected:
    virtual void on_open() {}
    virtual void on(In&) {}
    caliper_net::Client client_;
};

// One scene "world" (3-D, up +z) and a series "loss". The scene's batches are
// kept here, as Caliper LIB keeps them, and sent as its state on SUBSCRIBE.
class SceneProducer : public Client {
public:
    bool welcomed = false;
    std::vector<std::string> hub_caps;
    std::map<uint32_t, std::string> subs;     // sub -> view
    int subscribes = 0, scene_subscribes = 0;
    uint64_t seq = 0;
    std::vector<float> frame = {-1, -1, -1, 1, -1, -1};   // a segment, static
    std::map<std::string, std::vector<float>> clouds;    // path -> positions of a points batch

    wire::SceneUpdate update(uint32_t sub, const std::string& path, const std::vector<float>& pos, bool is_static,
                             bool state, const char* part = CALIPER_NET_PART_POINTS) {
        wire::SceneUpdate u;
        u.sub = sub;
        u.view = "world";
        u.seq = ++seq;
        u.stamp_ns = mono_ns32();
        u.path = path;
        u.part = part;
        u.is_static = is_static;
        u.state = state;
        if (std::string(part) == CALIPER_NET_PART_LINES) u.params.set("mode", CALIPER_NET_LINES_SEGMENTS);
        u.arrays.push_back(wire::Section::make("positions", CALIPER_NET_DTYPE_F32, {(int64_t)pos.size() / 3, 3},
                                               pos.data(), pos.size() * 4));
        return u;
    }
    std::vector<uint32_t> scene_subs() const {
        std::vector<uint32_t> v;
        for (auto& kv : subs)
            if (kv.second == "world") v.push_back(kv.first);
        return v;
    }
    void points(const std::string& path, std::vector<float> pos) {
        clouds[path] = pos;
        for (uint32_t sub : scene_subs()) send(update(sub, path, clouds[path], false, false));
    }
    void clear(const std::string& path) {
        clouds.erase(path);
        for (uint32_t sub : scene_subs()) {
            wire::SceneClear c;
            c.sub = sub;
            c.view = "world";
            c.seq = ++seq;
            c.stamp_ns = mono_ns32();
            c.path = path;
            send(c);
        }
    }

protected:
    void on_open() override {
        wire::Hello h;
        h.role = CALIPER_NET_ROLE_PRODUCER;
        h.capabilities = {CALIPER_NET_CAP_EVENTS, CALIPER_NET_CAP_SCENES};
        wire::Identity id;
        id.id = "clp-scene";
        id.script = "scene.py";
        id.pid = 7;
        id.start_unix_ms = 1763200000000ull;
        h.session = id.to_json();
        send(h);
    }
    void on(In& in) override {
        if (auto* w = std::get_if<wire::Welcome>(&in.m)) {
            welcomed = true;
            hub_caps = w->capabilities;
            wire::Views v;
            Json world = Json::object();
            world.set("id", 1);
            world.set("name", "world");
            world.set("kind", (int)CALIPER_NET_KIND_SCENE);
            world.set("sources", Json::array());
            Json sc = Json::object();
            sc.set("space", "3d");
            sc.set("up", "+z");
            sc.set("units", "m");
            world.set("scene", sc);
            v.views.push(world);
            Json loss = Json::object();
            loss.set("id", 2);
            loss.set("name", "loss");
            loss.set("kind", (int)CALIPER_NET_KIND_SERIES);
            loss.set("sources", Json::array());
            v.views.push(loss);
            send(v);
        } else if (auto* s = std::get_if<wire::Subscribe>(&in.m)) {
            ++subscribes;
            wire::ViewReply r;
            r.request_id = s->request_id;
            r.sub = s->sub;
            r.view = s->view;
            if (s->view == "world") {
                r.data_view = wire::scene_data_view();
            } else {
                Json b = Json::object();
                b.set("max_backfill", 4096);
                Json dv = Json::object();
                dv.set("budget", b);
                r.data_view = dv;
            }
            subs[s->sub] = s->view;
            send(r);
            if (s->view != "world") return;
            ++scene_subscribes;
            wire::SceneClear reset;
            reset.sub = s->sub;
            reset.view = "world";
            reset.seq = seq;
            reset.stamp_ns = mono_ns32();
            reset.reset = true;
            reset.state = true;
            send(reset);
            send(update(s->sub, "", frame, true, true, CALIPER_NET_PART_LINES));
            for (auto& kv : clouds) send(update(s->sub, kv.first, kv.second, false, true));
        } else if (auto* u = std::get_if<wire::Unsubscribe>(&in.m)) {
            subs.erase(u->sub);
        }
    }
};

struct World32 {
    std::vector<Client*> peers;
    bool until(const std::function<bool()>& done, int ms = 5000) {
        const auto end = std::chrono::steady_clock::now() + std::chrono::milliseconds(ms);
        while (std::chrono::steady_clock::now() < end) {
            for (Client* p : peers) p->poll();
            if (done()) return true;
            std::this_thread::sleep_for(std::chrono::microseconds(500));
        }
        return done();
    }
    void spin(int ms) { until([] { return false; }, ms); }
};

std::unique_ptr<Client> scene_viewer(World32& w, const Runner& h, std::vector<std::string> caps) {
    auto v = std::make_unique<Client>();
    REQUIRE(v->connect(h.url()));
    w.peers.push_back(v.get());
    REQUIRE(w.until([&] { return v->open; }));
    wire::Hello hello;
    hello.role = CALIPER_NET_ROLE_VIEWER;
    hello.capabilities = std::move(caps);
    v->send(hello);
    REQUIRE(w.until([&] { return v->last<wire::Views>() && v->last<wire::Views>()->session; }));
    return v;
}

void subscribe(Client& v, uint32_t sub, const std::string& view, double rate = 0) {
    wire::Subscribe s;
    s.sub = sub;
    s.view = view;
    s.request_id = 100 + sub;
    if (rate > 0) s.max_rate = rate;
    v.send(s);
}

const wire::ViewReply* reply_of(const Client& v, uint32_t sub) {
    for (auto* r : v.all<wire::ViewReply>())
        if (r->request_id == 100 + sub) return r;
    return nullptr;
}

bool has_view(const Client& v, const std::string& name) {
    const wire::Views* views = v.last<wire::Views>();
    if (!views) return false;
    for (const auto& e : views->list)
        if (e.name == name) return true;
    return false;
}

std::vector<float> floats(const wire::Section& s) {
    std::vector<float> out(s.size / 4);
    std::memcpy(out.data(), s.data, s.size);
    return out;
}

}  // namespace

TEST_CASE("net/hub 3.2: scene messages reach the subscription's viewers; a late viewer gets the hub's state") {
    Runner h;
    World32 w;
    auto p = std::make_unique<SceneProducer>();
    REQUIRE(p->connect(h.url()));
    w.peers.push_back(p.get());
    REQUIRE(w.until([&] { return p->welcomed; }));
    CHECK(std::find(p->hub_caps.begin(), p->hub_caps.end(), CALIPER_NET_CAP_SCENES) != p->hub_caps.end());
    p->clouds["world/cloud"] = {0, 0, 0, 1, 0, 0.5f};

    auto a = scene_viewer(w, h, {CALIPER_NET_CAP_SCENES});
    REQUIRE(w.until([&] { return has_view(*a, "world"); }));
    const wire::Views* va = a->last<wire::Views>();
    for (const auto& e : va->list)
        if (e.name == "world") {
            wire::SceneView sv;
            std::string err;
            REQUIRE(wire::scene_view_info(va->views.at(0), &sv, &err));
            CHECK(sv.up == "+z");
        }
    subscribe(*a, 5, "world");
    REQUIRE(w.until([&] { return a->scene_log(5).size() == 3; }));
    CHECK(reply_of(*a, 5)->data_view == wire::scene_data_view());
    CHECK(a->scene_log(5) == std::vector<std::string>{"reset", ":lines", "world/cloud:points"});
    CHECK(a->scene_log(5, true).size() == 3);
    // A new batch reaches the viewer with its own sub and the producer's arrays.
    p->points("world/cloud", {2, 2, 2});
    REQUIRE(w.until([&] { return a->scene_log(5).size() == 4; }));
    const wire::SceneUpdate* u = a->last<wire::SceneUpdate>();
    CHECK(u->sub == 5);
    CHECK(!u->state);
    CHECK(floats(*u->array("positions")) == std::vector<float>{2, 2, 2});
    p->points("world/marker", {9, 9, 9});
    REQUIRE(w.until([&] { return a->scene_log(5).size() == 5; }));

    // A late viewer shares the producer subscription and gets the hub's own state.
    auto b = scene_viewer(w, h, {CALIPER_NET_CAP_SCENES});
    subscribe(*b, 9, "world");
    REQUIRE(w.until([&] { return b->scene_log(9).size() == 4; }));
    CHECK(b->scene_log(9) == std::vector<std::string>{"reset", ":lines", "world/cloud:points", "world/marker:points"});
    CHECK(b->scene_log(9, true).size() == 4);
    const auto ups = b->all<wire::SceneUpdate>();
    CHECK(ups.back()->sub == 9);
    CHECK(floats(*ups[1]->array("positions")) == std::vector<float>{2, 2, 2});   // the newest batch
    CHECK(ups[0]->is_static);
    CHECK(p->scene_subscribes == 1);

    // A clear reaches both viewers in order and leaves the hub's state.
    p->clear("world/marker");
    REQUIRE(w.until([&] {
        return a->scene_log(5).back() == "clear world/marker" && b->scene_log(9).back() == "clear world/marker";
    }));
    auto c = scene_viewer(w, h, {CALIPER_NET_CAP_SCENES});
    subscribe(*c, 1, "world");
    REQUIRE(w.until([&] { return c->scene_log(1).size() == 3; }));
    // The hub publishes its counters at the end of a loop round: wait for them, not for a time.
    REQUIRE(w.until([&] { return h.hub->stats().scenes_out >= 14; }));
    CHECK(c->scene_log(1) == std::vector<std::string>{"reset", ":lines", "world/cloud:points"});
    const caliper_net::HubStats st = h.hub->stats();
    CHECK(st.scenes_in == 6);
    CHECK(st.scenes_out == 14);
    CHECK(st.retained_scene_bytes > 0);
}

TEST_CASE("net/hub 3.2: a viewer without scenes sees no scene and cannot subscribe to one") {
    Runner h;
    World32 w;
    auto p = std::make_unique<SceneProducer>();
    REQUIRE(p->connect(h.url()));
    w.peers.push_back(p.get());
    REQUIRE(w.until([&] { return p->welcomed; }));
    auto v = scene_viewer(w, h, {});
    REQUIRE(w.until([&] { return has_view(*v, "loss"); }));
    CHECK(!has_view(*v, "world"));
    subscribe(*v, 3, "world");
    REQUIRE(w.until([&] { return reply_of(*v, 3) != nullptr; }));
    CHECK(reply_of(*v, 3)->error.find("code")->as_string() == CALIPER_NET_ERR_UNKNOWN_VIEW);
    w.spin(50);
    CHECK(p->scene_subscribes == 0);
    CHECK(v->all<wire::SceneUpdate>().empty());
}

// The hub sends each entity part of a subscription at most once per 1/max_rate and keeps
// one batch of it waiting, which a newer one replaces. The test asserts on what is sent and
// on the hub's counters, never on how many batches arrive per second, so it holds on a
// runner of any speed: the slow subscription's interval (1,000 s) is far longer than the
// test, so after its first batch of each part the 39 that follow can only wait as one, and
// the paced one (4 a second) receives each part in order, the newest last, with every batch
// it did not receive replaced by a newer one.
TEST_CASE("net/hub 3.2: a slow viewer's batches coalesce per entity part, the newest last") {
    Runner h;
    World32 w;
    auto p = std::make_unique<SceneProducer>();
    REQUIRE(p->connect(h.url()));
    w.peers.push_back(p.get());
    REQUIRE(w.until([&] { return p->welcomed; }));
    auto paced = scene_viewer(w, h, {CALIPER_NET_CAP_SCENES});
    auto slow = scene_viewer(w, h, {CALIPER_NET_CAP_SCENES});
    subscribe(*paced, 2, "world", 4.0);   // 4 batches a second per entity part
    REQUIRE(w.until([&] { return paced->scene_log(2).size() == 2; }));
    subscribe(*slow, 3, "world", 0.001);  // 1 batch per entity part every 1,000 s
    REQUIRE(w.until([&] { return slow->scene_log(3).size() == 2; }));
    // The value of each batch of "a" or "b", in arrival order.
    const auto got = [](const Client& v, const std::string& path) {
        std::vector<float> out;
        for (auto* u : v.all<wire::SceneUpdate>())
            if (u->path == path) out.push_back(floats(*u->array("positions"))[path == "a" ? 0 : 1]);
        return out;
    };
    // The first batch of each part goes to both viewers at once.
    p->points("a", {0, 0, 0});
    p->points("b", {0, 0, 0});
    REQUIRE(w.until([&] {
        for (const Client* v : {paced.get(), slow.get()})
            if (got(*v, "a").size() != 1 || got(*v, "b").size() != 1) return false;
        return true;
    }));
    for (int i = 1; i < 40; ++i) {
        p->points("a", {(float)i, 0, 0});
        p->points("b", {0, (float)i, 0});
    }
    REQUIRE(w.until([&] {
        const std::vector<float> a = got(*paced, "a"), b = got(*paced, "b");
        return a.back() == 39.0f && b.back() == 39.0f;
    }, 20000));
    size_t paced_n = 0;
    for (const std::string path : {"a", "b"}) {
        INFO(path);
        const std::vector<float> pv = got(*paced, path);
        paced_n += pv.size();
        CHECK(pv.front() == 0.0f);
        CHECK(std::adjacent_find(pv.begin(), pv.end(), std::greater_equal<float>()) == pv.end());   // newer each time
        CHECK(got(*slow, path) == std::vector<float>{0.0f});   // the rest wait as one
    }
    // Every batch is sent or replaced by a newer one: the paced viewer's that it did not
    // receive, and 38 of each part's 39 that wait for the slow one, which keeps the newest.
    CHECK(p->scene_subscribes == 1);                   // both viewers share one producer subscription
    REQUIRE(w.until([&] { return h.hub->stats().scenes_in == 2 + 80; }));   // its state, then the batches
    CHECK(h.hub->stats().scenes_superseded == (80 - paced_n) + 2 * 38);
}

// The hub keeps a watched scene's state within its budget of retained bytes: past it, it forgets
// the state (the viewers watching keep receiving every message), and a later viewer gets a fresh
// state from the producer, for which the hub subscribes again.
TEST_CASE("net/hub 3.2: a watched scene's state stays within the hub's budget; a late viewer gets a fresh state") {
    const size_t budget = size_t(64) << 10;
    Runner h(budget);
    World32 w;
    auto p = std::make_unique<SceneProducer>();
    REQUIRE(p->connect(h.url()));
    w.peers.push_back(p.get());
    REQUIRE(w.until([&] { return p->welcomed; }));
    auto a = scene_viewer(w, h, {CALIPER_NET_CAP_SCENES});
    subscribe(*a, 5, "world");
    REQUIRE(w.until([&] { return a->scene_log(5).size() == 2; }));
    // 40 entities of 1,024 points (12 KiB) each: 480 KiB, past the budget of 64 KiB.
    std::vector<float> cloud(3 * 1024, 0.5f);
    for (int i = 0; i < 40; ++i) p->points("step/" + std::to_string(i), cloud);
    REQUIRE(w.until([&] { return h.hub->stats().scenes_in == 2 + 40; }));
    REQUIRE(w.until([&] { return a->scene_log(5).size() == 2 + 40; }));   // the viewer missed nothing
    CHECK(h.hub->stats().retained_scene_bytes <= budget);
    // A late viewer: the hub subscribes again, and the producer's own state reaches it whole.
    auto b = scene_viewer(w, h, {CALIPER_NET_CAP_SCENES});
    subscribe(*b, 9, "world");
    REQUIRE(w.until([&] { return b->scene_log(9).size() == 2 + 40; }));
    CHECK(p->scene_subscribes == 2);
    const std::vector<std::string> got = b->scene_log(9, true);
    REQUIRE(got.size() == 2 + 40);
    CHECK(got[0] == "reset");
    CHECK(got[1] == ":lines");
    CHECK(got.back() == "step/9:points");   // the producer's state, in the byte order of the paths
    REQUIRE(w.until([&] { return h.hub->stats().scenes_in == 2 + 40 + 2 + 40; }));
    CHECK(h.hub->stats().retained_scene_bytes <= budget);
}

// A producer that sends more batches than a scene holds (CALIPER_NET_MAX_BATCHES) makes the hub
// forget the scene's state rather than keep every batch; a late viewer gets the producer's state.
TEST_CASE("net/hub 3.2: past CALIPER_NET_MAX_BATCHES batches the hub forgets a scene's state") {
    Runner h;
    World32 w;
    auto p = std::make_unique<SceneProducer>();
    REQUIRE(p->connect(h.url()));
    w.peers.push_back(p.get());
    REQUIRE(w.until([&] { return p->welcomed; }));
    auto a = scene_viewer(w, h, {CALIPER_NET_CAP_SCENES});
    subscribe(*a, 5, "world");
    REQUIRE(w.until([&] { return a->scene_log(5).size() == 2; }));
    const int n = (int)CALIPER_NET_MAX_BATCHES + 8;
    for (int i = 0; i < n; ++i) p->points("e/" + std::to_string(i), {(float)i, 0, 0});
    REQUIRE(w.until([&] { return h.hub->stats().scenes_in == 2 + (uint64_t)n; }, 20000));
    REQUIRE(w.until([&] { return a->scene_log(5).size() == 2 + (size_t)n; }, 20000));
    // The hub holds at most CALIPER_NET_MAX_BATCHES batches of the scene (each frame is under 512 bytes).
    CHECK(h.hub->stats().retained_scene_bytes <= (uint64_t)CALIPER_NET_MAX_BATCHES * 512);
    auto b = scene_viewer(w, h, {CALIPER_NET_CAP_SCENES});
    subscribe(*b, 9, "world");
    REQUIRE(w.until([&] { return b->scene_log(9).size() == 2 + (size_t)n; }, 20000));
    CHECK(p->scene_subscribes == 2);
    CHECK(b->scene_log(9, true).front() == "reset");
}

// A viewer that stops reading: its connection holds one batch of 24 MB that its socket cannot
// take, so every later scene message waits in the hub's queue for it. Returns the scene log of
// `sub` the viewer had before it stopped.
namespace {
size_t stall(World32& w, const Runner& h, SceneProducer& p, Client& slow, uint32_t sub) {
    w.peers.erase(std::find(w.peers.begin(), w.peers.end(), &slow));
    const uint64_t out = h.hub->stats().scenes_out;
    p.points("big", std::vector<float>(3 * 2000000, 0.25f));
    REQUIRE(w.until([&] { return h.hub->stats().scenes_out > out && h.hub->stats().queued_bytes > (8u << 20); }));
    return slow.scene_log(sub).size();
}
// Reads `slow` again until `done` holds for its scene log of `sub`.
bool resume(Client& slow, uint32_t sub, const std::function<bool(const std::vector<std::string>&)>& done) {
    const auto end = std::chrono::steady_clock::now() + std::chrono::seconds(20);
    while (std::chrono::steady_clock::now() < end) {
        slow.poll();
        if (done(slow.scene_log(sub))) return true;
        std::this_thread::sleep_for(std::chrono::microseconds(500));
    }
    return false;
}
size_t clears_in(const std::vector<std::string>& log, size_t from) {
    size_t n = 0;
    for (size_t i = from; i < log.size(); ++i) n += log[i].rfind("clear ", 0) == 0;
    return n;
}
}  // namespace

// The hub's in-order queue of a viewer that stopped reading stays bounded: past 2 ×
// CALIPER_NET_MAX_BATCHES messages, the hub's own state (a reset and every batch it keeps, which
// hold the effect of every message so far) replaces what waits.
TEST_CASE("net/hub 3.2: a viewer that stopped reading while the producer clears gets the hub's state") {
    Runner h;
    World32 w;
    auto p = std::make_unique<SceneProducer>();
    REQUIRE(p->connect(h.url()));
    w.peers.push_back(p.get());
    REQUIRE(w.until([&] { return p->welcomed; }));
    auto slow = scene_viewer(w, h, {CALIPER_NET_CAP_SCENES});
    subscribe(*slow, 4, "world");
    REQUIRE(w.until([&] { return slow->scene_log(4).size() == 2; }));
    const size_t before = stall(w, h, *p, *slow, 4);
    const int n = 2 * (int)CALIPER_NET_MAX_BATCHES + 900;
    const uint64_t in = h.hub->stats().scenes_in;
    for (int i = 0; i < n; ++i) p->clear("c/" + std::to_string(i));
    REQUIRE(w.until([&] { return h.hub->stats().scenes_in == in + (uint64_t)n; }, 20000));
    const std::string last = "clear c/" + std::to_string(n - 1);
    REQUIRE(resume(*slow, 4, [&](const std::vector<std::string>& log) { return !log.empty() && log.back() == last; }));
    const std::vector<std::string> log = slow->scene_log(4);
    CHECK(clears_in(log, before) < 2 * CALIPER_NET_MAX_BATCHES);
    CHECK(std::find(log.begin() + (long)before, log.end(), "reset") != log.end());
}

// A reset opens a new state of the scene and drops what a viewer has queued before it.
TEST_CASE("net/hub 3.2: a reset drops what a viewer that stopped reading has queued before it") {
    Runner h;
    World32 w;
    auto p = std::make_unique<SceneProducer>();
    REQUIRE(p->connect(h.url()));
    w.peers.push_back(p.get());
    REQUIRE(w.until([&] { return p->welcomed; }));
    auto slow = scene_viewer(w, h, {CALIPER_NET_CAP_SCENES});
    subscribe(*slow, 4, "world");
    REQUIRE(w.until([&] { return slow->scene_log(4).size() == 2; }));
    const size_t before = stall(w, h, *p, *slow, 4);
    const uint64_t in = h.hub->stats().scenes_in;
    for (int i = 0; i < 100; ++i) p->clear("c/" + std::to_string(i));
    REQUIRE(w.until([&] { return h.hub->stats().scenes_in == in + 100; }));
    // A viewer at a higher rate makes the hub subscribe again, and the producer sends its state.
    auto fast = scene_viewer(w, h, {CALIPER_NET_CAP_SCENES});
    subscribe(*fast, 6, "world", 120.0);
    // The fast viewer can receive the hub's cached state before the producer's new state.
    // Poll the producer until all three replay messages reach the hub. resume() polls only slow.
    REQUIRE(w.until([&] { return p->scene_subscribes == 2 && h.hub->stats().scenes_in == in + 100 + 3; }));
    REQUIRE(resume(*slow, 4, [&](const std::vector<std::string>& log) {
        return log.size() >= before + 4 && log.back() == "big:points" && log[log.size() - 3] == "reset";
    }));
    const std::vector<std::string> log = slow->scene_log(4);
    CHECK(clears_in(log, before) == 0);
    CHECK(std::vector<std::string>(log.begin() + (long)before, log.end()) ==
          std::vector<std::string>{"big:points", "reset", ":lines", "big:points"});
}

TEST_CASE("net/hub 3.2: a cached scene does not acknowledge the producer's new state") {
    Runner h;
    World32 w;
    auto p = std::make_unique<SceneProducer>();
    REQUIRE(p->connect(h.url()));
    w.peers.push_back(p.get());
    REQUIRE(w.until([&] { return p->welcomed; }));
    auto slow = scene_viewer(w, h, {CALIPER_NET_CAP_SCENES});
    subscribe(*slow, 4, "world");
    REQUIRE(w.until([&] { return slow->scene_log(4).size() == 2; }));
    p->points("big", {0.25f, 0.25f, 0.25f});
    REQUIRE(w.until([&] { return slow->scene_log(4).size() == 3 && h.hub->stats().scenes_in == 3; }));
    const size_t before = slow->scene_log(4).size();
    const uint64_t in = h.hub->stats().scenes_in;

    // Let the fast viewer receive the cached state before the producer reads its new SUBSCRIBE.
    w.peers.erase(std::find(w.peers.begin(), w.peers.end(), p.get()));
    w.peers.erase(std::find(w.peers.begin(), w.peers.end(), slow.get()));
    auto fast = scene_viewer(w, h, {CALIPER_NET_CAP_SCENES});
    subscribe(*fast, 6, "world", 120.0);
    REQUIRE(w.until([&] { return fast->scene_log(6).size() == 3; }));
    CHECK(p->scene_subscribes == 1);
    CHECK(fast->scene_log(6, true) == std::vector<std::string>{"reset", ":lines", "big:points"});

    // The callback queues the replay after Client::poll() returns its transport events.
    // Stop at that callback, before the next poll can flush those messages.
    World32 producer;
    producer.peers.push_back(p.get());
    REQUIRE(producer.until([&] { return p->scene_subscribes == 2; }));
    CHECK((p->scene_subscribes == 2 && fast->scene_log(6).size() >= 3));
    CHECK(h.hub->stats().scenes_in == in);

    w.peers.push_back(p.get());
    REQUIRE(w.until([&] { return p->scene_subscribes == 2 && h.hub->stats().scenes_in == in + 3; }));
    REQUIRE(resume(*slow, 4, [&](const std::vector<std::string>& log) { return log.size() >= before + 3; }));
    const std::vector<std::string> log = slow->scene_log(4);
    CHECK(std::vector<std::string>(log.begin() + (long)before, log.end()) ==
          std::vector<std::string>{"reset", ":lines", "big:points"});
}

// The pending batches of a viewer that stopped reading stay bounded: past CALIPER_NET_MAX_BATCHES
// entity parts (the hub has forgotten the scene's state by then), the hub drops them and asks the
// producer for its state, which opens with a reset.
TEST_CASE("net/hub 3.2: a viewer that stopped reading while the producer writes new entities gets a new state") {
    Runner h;
    World32 w;
    auto p = std::make_unique<SceneProducer>();
    REQUIRE(p->connect(h.url()));
    w.peers.push_back(p.get());
    REQUIRE(w.until([&] { return p->welcomed; }));
    auto slow = scene_viewer(w, h, {CALIPER_NET_CAP_SCENES});
    subscribe(*slow, 4, "world");
    REQUIRE(w.until([&] { return slow->scene_log(4).size() == 2; }));
    const size_t before = stall(w, h, *p, *slow, 4);
    const int n = (int)CALIPER_NET_MAX_BATCHES + 100;
    const uint64_t in = h.hub->stats().scenes_in;
    for (int i = 0; i < n; ++i) p->points("e/" + std::to_string(i), {(float)i, 0, 0});
    // The producer's state, which the hub asked for: a reset, its static frame, "big" and the n entities.
    // The producer is polled until the hub has all of it, since `resume` reads the viewer alone.
    REQUIRE(w.until([&] { return p->scene_subscribes == 2 && h.hub->stats().scenes_in == in + 2 * (uint64_t)n + 3; },
                    20000));
    auto after_reset = [&](const std::vector<std::string>& log) -> long {
        auto r = std::find(log.begin() + (long)before, log.end(), "reset");
        return r == log.end() ? -1 : (long)(log.end() - r) - 1;
    };
    REQUIRE(resume(*slow, 4, [&](const std::vector<std::string>& log) { return after_reset(log) >= n + 2; }));
    // Without the cap the viewer got one batch per entity part and no reset.
    const std::vector<std::string> log = slow->scene_log(4);
    CHECK(std::count(log.begin() + (long)before, log.end(), "reset") == 1);
    CHECK(after_reset(log) == n + 2);
}

// The pending entity transforms of a viewer that stopped reading stay bounded: past
// CALIPER_NET_MAX_BATCHES entities the hub's own state, which holds every transform, replaces them.
TEST_CASE("net/hub 3.2: a viewer that stopped reading while the producer moves many entities gets the hub's state") {
    Runner h;
    World32 w;
    auto p = std::make_unique<SceneProducer>();
    REQUIRE(p->connect(h.url()));
    w.peers.push_back(p.get());
    REQUIRE(w.until([&] { return p->welcomed; }));
    auto slow = scene_viewer(w, h, {CALIPER_NET_CAP_SCENES});
    subscribe(*slow, 4, "world");
    REQUIRE(w.until([&] { return slow->scene_log(4).size() == 2; }));
    const size_t before = stall(w, h, *p, *slow, 4);
    const size_t start = slow->inbox.size();
    const uint32_t up = p->scene_subs().at(0);
    // Three SCENE_TRANSFORMS of 1000 new entities each: 3000 pending transforms pass the cap.
    std::vector<float> mats;
    for (int k = 0; k < 1000; ++k) {
        for (int e = 0; e < 16; ++e) mats.push_back(e % 5 == 0 ? 1.f : 0.f);
    }
    const uint64_t in = h.hub->stats().scenes_in;
    for (int m = 0; m < 3; ++m) {
        wire::SceneTransforms t;
        t.sub = up;
        t.view = "world";
        t.seq = ++p->seq;
        t.stamp_ns = mono_ns32();
        for (int k = 0; k < 1000; ++k) t.paths.push_back("m" + std::to_string(m) + "/" + std::to_string(k));
        t.is_static = true;
        t.matrices = wire::Section::make("matrices", CALIPER_NET_DTYPE_F32, {1000, 16}, mats.data(), mats.size() * 4);
        p->send(t);
    }
    REQUIRE(w.until([&] { return h.hub->stats().scenes_in == in + 3; }));
    REQUIRE(resume(*slow, 4, [&](const std::vector<std::string>& log) {
        size_t paths = 0;
        for (size_t i = start; i < slow->inbox.size(); ++i)
            if (auto* t = std::get_if<wire::SceneTransforms>(&slow->inbox[i].m)) paths += t->paths.size();
        return std::find(log.begin() + (long)before, log.end(), "reset") != log.end() && paths == 3000;
    }));
    // Without the cap the transforms waited in the viewer's queue and came with no reset; now they
    // come once, in the hub's state after its reset.
    const std::vector<std::string> log = slow->scene_log(4);
    CHECK(std::count(log.begin() + (long)before, log.end(), "reset") == 1);
    size_t paths = 0;
    for (size_t i = start; i < slow->inbox.size(); ++i)
        if (auto* t = std::get_if<wire::SceneTransforms>(&slow->inbox[i].m)) paths += t->paths.size();
    CHECK(paths == 3000);
}

namespace {
// Every scene message of `sub` in arrival order, the new ones included: "classes", "transforms
// static|stamped p1,p2" and "path:part" (with "#k" for a piece of a split update).
std::vector<std::string> scene_all(const Client& c, uint32_t sub) {
    std::vector<std::string> out;
    for (const auto& x : c.inbox) {
        if (auto* u = std::get_if<wire::SceneUpdate>(&x.m)) {
            if (u->sub != sub) continue;
            std::string s = u->path + ":" + u->part;
            if (!u->split.is_null()) s += "#" + std::to_string((int)u->split.find("piece")->as_double());
            out.push_back(s);
        } else if (auto* cl = std::get_if<wire::SceneClear>(&x.m)) {
            if (cl->sub == sub) out.push_back(cl->reset ? "reset" : "clear " + cl->path);
        } else if (auto* t = std::get_if<wire::SceneTransforms>(&x.m)) {
            if (t->sub != sub) continue;
            std::string s = std::string("transforms ") + (t->is_static ? "static " : "stamped ");
            for (size_t i = 0; i < t->paths.size(); ++i) s += (i ? "," : "") + t->paths[i];
            out.push_back(s);
        } else if (auto* k = std::get_if<wire::SceneClasses>(&x.m)) {
            if (k->sub == sub) out.push_back("classes");
        }
    }
    return out;
}

float translation_x(const wire::SceneTransforms& t, size_t k) {
    float x = 0;
    std::memcpy(&x, t.matrices.data + 64 * k + 48, 4);   // column-major: element 12
    return x;
}
}  // namespace

TEST_CASE("net/hub 3.2: class tables, entity transforms, cameras and split updates are kept, merged and forwarded") {
    Runner h;
    World32 w;
    auto p = std::make_unique<SceneProducer>();
    REQUIRE(p->connect(h.url()));
    w.peers.push_back(p.get());
    REQUIRE(w.until([&] { return p->welcomed; }));
    auto a = scene_viewer(w, h, {CALIPER_NET_CAP_SCENES});
    subscribe(*a, 5, "world");
    REQUIRE(w.until([&] { return scene_all(*a, 5).size() == 2; }));   // reset, the static frame
    const uint32_t up = p->scene_subs().at(0);

    std::vector<float> eye(16, 0.f);
    eye[0] = eye[5] = eye[10] = eye[15] = 1.f;
    auto transforms = [&](std::vector<std::string> paths, float x, bool is_static) {
        static std::vector<float> mats;
        mats.clear();
        for (size_t k = 0; k < paths.size(); ++k) {
            std::vector<float> m = eye;
            m[12] = x;
            mats.insert(mats.end(), m.begin(), m.end());
        }
        wire::SceneTransforms t;
        t.sub = up;
        t.view = "world";
        t.seq = ++p->seq;
        t.stamp_ns = mono_ns32();
        t.paths = std::move(paths);
        t.is_static = is_static;
        t.matrices = wire::Section::make("matrices", CALIPER_NET_DTYPE_F32, {(int64_t)t.paths.size(), 16}, mats.data(),
                                         mats.size() * 4);
        p->send(t);
    };
    wire::SceneClasses k;
    k.sub = up;
    k.view = "world";
    k.seq = ++p->seq;
    k.stamp_ns = mono_ns32();
    k.is_static = true;
    Json entry = Json::object();
    entry.set("id", 1);
    entry.set("label", "obstacle");
    Json color = Json::array();
    for (int c : {230, 80, 60, 255}) color.push(c);
    entry.set("color", color);
    k.classes.push(entry);
    p->send(k);
    transforms({"rig"}, 7.f, true);
    transforms({"a", "b"}, 1.f, false);
    transforms({"a"}, 2.f, false);
    wire::SceneUpdate cam;
    cam.sub = up;
    cam.view = "world";
    cam.seq = ++p->seq;
    cam.stamp_ns = mono_ns32();
    cam.path = "rig/cam";
    cam.part = CALIPER_NET_PART_CAMERA;
    for (const char* key : {"fx", "fy", "cx", "cy"}) cam.params.set(key, 2.0);
    cam.params.set("width", 4);
    cam.params.set("height", 4);
    cam.params.set("convention", "rdf");
    p->send(cam);
    // A split update of four points in two pieces.
    static const float big[12] = {0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3};
    auto piece = [&](int num, uint64_t update) {
        wire::SceneUpdate u;
        u.sub = up;
        u.view = "world";
        u.seq = 99;
        u.stamp_ns = 1;
        u.path = "big";
        u.part = CALIPER_NET_PART_POINTS;
        Json sp = Json::object(), first = Json::object();
        sp.set("update", update);
        sp.set("piece", num);
        sp.set("pieces", 2);
        first.set("positions", (uint64_t)(2 * num));
        sp.set("first", first);
        if (num == 0) {
            Json rows = Json::object();
            rows.set("positions", (uint64_t)4);
            sp.set("rows", rows);
        }
        u.split = sp;
        u.arrays.push_back(wire::Section::make("positions", CALIPER_NET_DTYPE_F32, {2, 3}, big + 6 * num, 24));
        p->send(u);
    };
    piece(0, 1);
    piece(1, 1);
    piece(1, 7);   // a piece of an update whose piece 0 never came: dropped
    REQUIRE(w.until([&] {
        const auto log = scene_all(*a, 5);
        // The hub forwards the parts of a view in its own order, so wait for each part the checks
        // below read and not only for the last one sent.
        return std::count(log.begin(), log.end(), "big:points#1") == 1 && log.size() >= 7 &&
               std::count(log.begin(), log.end(), "classes") == 1 &&
               std::count(log.begin(), log.end(), "rig/cam:camera") == 1;
    }));
    {
        const auto log = scene_all(*a, 5);
        CHECK(std::count(log.begin(), log.end(), "classes") == 1);
        CHECK(std::count(log.begin(), log.end(), "rig/cam:camera") == 1);
        // the pieces go together and in order
        const auto p0 = std::find(log.begin(), log.end(), "big:points#0");
        REQUIRE(p0 != log.end());
        CHECK(*(p0 + 1) == "big:points#1");
        // the transforms reached the viewer, merged where they waited: "a" at its newest
        float ax = 0;
        for (auto* t : a->all<wire::SceneTransforms>())
            for (size_t i = 0; i < t->paths.size(); ++i)
                if (t->paths[i] == "a") ax = translation_x(*t, i);
        CHECK(ax == 2.f);
    }
    REQUIRE(w.until([&] { return h.hub->stats().scene_pieces_dropped == 1; }));

    // A late viewer gets the hub's state in the draft's order.
    auto b = scene_viewer(w, h, {CALIPER_NET_CAP_SCENES});
    subscribe(*b, 9, "world");
    const std::vector<std::string> want = {"reset", "classes", "transforms static rig", "transforms stamped a,b",
                                           ":lines", "big:points#0", "big:points#1", "rig/cam:camera"};
    REQUIRE(w.until([&] { return scene_all(*b, 9).size() == want.size(); }));
    CHECK(scene_all(*b, 9) == want);
    for (auto* t : b->all<wire::SceneTransforms>()) {
        CHECK(t->state);
        if (t->paths.size() == 2) {
            CHECK(translation_x(*t, 0) == 2.f);   // "a", the newest
            CHECK(translation_x(*t, 1) == 1.f);
        }
    }

    // A clear of "a" removes its transform from what the hub keeps; the static one stays.
    p->clear("a");
    REQUIRE(w.until([&] { return scene_all(*b, 9).back() == "clear a"; }));
    auto c = scene_viewer(w, h, {CALIPER_NET_CAP_SCENES});
    subscribe(*c, 3, "world");
    REQUIRE(w.until([&] { return scene_all(*c, 3).size() == want.size(); }));
    CHECK(scene_all(*c, 3)[3] == "transforms stamped b");
    CHECK(p->scene_subscribes == 1);
}

// ---- 3.2 draft: PROGRAM, PROGRAM_STATE and interventions --------------------------------------

namespace {

// A golden message of the draft, decoded; the buffers stay alive for the test run.
wire::Message golden32(const char* name) {
    static std::deque<std::vector<uint8_t>> keep;
    const auto path = std::filesystem::path(CALIPER_TEST_SOURCE_ROOT) / "tests" / "fixtures" / "wire_v3" / "v3_2" /
                      (std::string(name) + ".bin");
    std::ifstream f(path, std::ios::binary);
    keep.emplace_back((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    wire::Message m;
    const wire::DecodeResult r = wire::decode(keep.back().data(), keep.back().size(), &m);
    REQUIRE_MESSAGE(r.ok(), name << ": " << r.message);
    return m;
}

// A producer with the capabilities it is given, that records the PROGRAMs and CONTROLs it gets.
class CapProducer : public Client {
public:
    explicit CapProducer(std::vector<std::string> caps) : caps_(std::move(caps)) {}
    bool welcomed = false;
    std::vector<wire::Program> programs;
    std::vector<wire::Control> controls;

protected:
    void on_open() override {
        wire::Hello h;
        h.role = CALIPER_NET_ROLE_PRODUCER;
        h.capabilities = caps_;
        wire::Identity id;
        id.id = "clp-prog";
        id.script = "prog.py";
        id.pid = 8;
        id.start_unix_ms = 1763200000000ull;
        h.session = id.to_json();
        send(h);
    }
    void on(In& in) override {
        if (std::holds_alternative<wire::Welcome>(in.m)) {
            welcomed = true;
            send(wire::Views{});
        } else if (auto* p = std::get_if<wire::Program>(&in.m)) {
            programs.push_back(*p);
        } else if (auto* c = std::get_if<wire::Control>(&in.m)) {
            controls.push_back(*c);
        }
    }

private:
    std::vector<std::string> caps_;
};

std::unique_ptr<CapProducer> cap_producer(World32& w, const Runner& h, std::vector<std::string> caps) {
    auto p = std::make_unique<CapProducer>(std::move(caps));
    REQUIRE(p->connect(h.url()));
    w.peers.push_back(p.get());
    REQUIRE(w.until([&] { return p->welcomed; }));
    return p;
}

wire::Program program_remove(uint32_t request_id) {
    wire::Program m = std::get<wire::Program>(golden32("program_remove"));
    m.request_id = request_id;
    return m;
}

wire::ProgramState program_state(std::optional<uint32_t> request_id) {
    wire::ProgramState m = std::get<wire::ProgramState>(golden32("program_state_waiting"));
    m.request_id = request_id;
    return m;
}

const wire::Error* error_for(const Client& v, uint16_t type, uint32_t request_id) {
    for (auto* e : v.all<wire::Error>())
        if (e->in_reply_to && *e->in_reply_to == type && e->request_id && *e->request_id == request_id) return e;
    return nullptr;
}

const std::vector<std::string> kProgramCaps = {CALIPER_NET_CAP_PROGRAMS};

}  // namespace

TEST_CASE("net/hub 3.2: PROGRAM and PROGRAM_STATE are forwarded with the viewers' own request ids") {
    Runner h(size_t(1) << 30, true);
    World32 w;
    auto p = cap_producer(w, h, kProgramCaps);
    auto a = scene_viewer(w, h, kProgramCaps);
    auto b = scene_viewer(w, h, {});
    auto a2 = scene_viewer(w, h, kProgramCaps);

    a->send(program_remove(7));
    a2->send(program_remove(7));   // the same id from another viewer must not collide
    REQUIRE(w.until([&] { return p->programs.size() == 2; }));
    CHECK(p->programs[0].op == CALIPER_NET_PROGRAM_REMOVE);
    CHECK(p->programs[0].request_id != 0);
    CHECK(p->programs[0].request_id != p->programs[1].request_id);
    const uint32_t up_a = p->programs[0].request_id, up_a2 = p->programs[1].request_id;

    // The answer goes to the viewer that asked, under its own id, and to no one else.
    p->send(program_state(up_a2));
    REQUIRE(w.until([&] { return a2->last<wire::ProgramState>() != nullptr; }));
    CHECK(a2->last<wire::ProgramState>()->request_id.value_or(0) == 7);
    p->send(program_state(up_a));
    REQUIRE(w.until([&] { return a->last<wire::ProgramState>() != nullptr; }));
    CHECK(a->last<wire::ProgramState>()->request_id.value_or(0) == 7);
    p->send(program_state(up_a));   // the route is gone: dropped
    p->send(program_state(std::nullopt));
    REQUIRE(w.until([&] { return a->all<wire::ProgramState>().size() == 2; }));
    CHECK(!a->all<wire::ProgramState>()[1]->request_id.has_value());
    CHECK(a2->all<wire::ProgramState>().size() == 2);
    CHECK(b->all<wire::ProgramState>().empty());
    CHECK(h.hub->stats().programs_forwarded == 2);

    // A viewer with programs that attaches later gets the latest unrequested state; one without, none.
    auto late = scene_viewer(w, h, kProgramCaps);
    auto late_none = scene_viewer(w, h, {});
    REQUIRE(w.until([&] { return late->last<wire::ProgramState>() != nullptr; }));
    CHECK(!late->last<wire::ProgramState>()->request_id.has_value());
    w.spin(50);
    CHECK(late_none->all<wire::ProgramState>().empty());
}

TEST_CASE("net/hub 3.2: a viewer without programs gets ERROR unsupported for PROGRAM") {
    Runner h(size_t(1) << 30, true);
    World32 w;
    auto p = cap_producer(w, h, kProgramCaps);
    auto v = scene_viewer(w, h, {});
    v->send(program_remove(5));
    REQUIRE(w.until([&] { return error_for(*v, CALIPER_NET_MSG_PROGRAM, 5) != nullptr; }));
    const wire::Error* e = error_for(*v, CALIPER_NET_MSG_PROGRAM, 5);
    CHECK(e->code == CALIPER_NET_ERR_UNSUPPORTED);
    CHECK(!e->fatal);
    w.spin(50);
    CHECK(p->programs.empty());
}

TEST_CASE("net/hub 3.2: a producer without programs gives the viewer ERROR unsupported") {
    Runner h(size_t(1) << 30, true);
    World32 w;
    auto p = cap_producer(w, h, {});
    auto v = scene_viewer(w, h, kProgramCaps);
    v->send(program_remove(6));
    REQUIRE(w.until([&] { return error_for(*v, CALIPER_NET_MSG_PROGRAM, 6) != nullptr; }));
    CHECK(error_for(*v, CALIPER_NET_MSG_PROGRAM, 6)->code == CALIPER_NET_ERR_UNSUPPORTED);
    w.spin(50);
    CHECK(p->programs.empty());
}

TEST_CASE("net/hub 3.2: a hub without allow_control answers PROGRAM forbidden") {
    Runner h;   // allow_control is off
    World32 w;
    auto p = cap_producer(w, h, kProgramCaps);
    auto v = scene_viewer(w, h, kProgramCaps);
    v->send(program_remove(8));
    REQUIRE(w.until([&] { return error_for(*v, CALIPER_NET_MSG_PROGRAM, 8) != nullptr; }));
    CHECK(error_for(*v, CALIPER_NET_MSG_PROGRAM, 8)->code == CALIPER_NET_ERR_FORBIDDEN);
    w.spin(50);
    CHECK(p->programs.empty());
}

TEST_CASE("net/hub 3.2: CONTROLS loses the key interventions for a viewer that did not announce it") {
    Runner h;
    World32 w;
    auto p = cap_producer(w, h, {CALIPER_NET_CAP_CONTROLS, CALIPER_NET_CAP_INTERVENTIONS});
    const std::vector<std::string> with = {CALIPER_NET_CAP_CONTROLS, CALIPER_NET_CAP_INTERVENTIONS};
    auto a = scene_viewer(w, h, with);
    auto b = scene_viewer(w, h, {CALIPER_NET_CAP_CONTROLS});
    const wire::Message m = golden32("controls_interventions");
    p->send(m);
    REQUIRE(w.until([&] { return a->last<wire::Controls>() && b->last<wire::Controls>(); }));
    CHECK(a->last<wire::Controls>()->interventions.is_object());
    CHECK(a->last<wire::Controls>()->interventions.find("points")->size() == 2);
    CHECK(b->last<wire::Controls>()->interventions.is_null());
    CHECK(b->last<wire::Controls>()->run.is_object());

    // A late viewer gets the stored CONTROLS the same way.
    auto c = scene_viewer(w, h, with);
    auto d = scene_viewer(w, h, {CALIPER_NET_CAP_CONTROLS});
    REQUIRE(w.until([&] { return c->last<wire::Controls>() && d->last<wire::Controls>(); }));
    CHECK(c->last<wire::Controls>()->interventions.is_object());
    CHECK(d->last<wire::Controls>()->interventions.is_null());
}

TEST_CASE("net/hub 3.2: CONTROL intervene to a producer without interventions gets CONTROL_REPLY unsupported") {
    Runner h(size_t(1) << 30, true);
    World32 w;
    auto p = cap_producer(w, h, {CALIPER_NET_CAP_CONTROLS});
    auto v = scene_viewer(w, h, {CALIPER_NET_CAP_CONTROLS, CALIPER_NET_CAP_INTERVENTIONS});
    wire::Control c = std::get<wire::Control>(golden32("control_intervene_clear_all"));
    c.request_id = 21;
    v->send(c);
    REQUIRE(w.until([&] { return v->last<wire::ControlReply>() != nullptr; }));
    const wire::ControlReply* r = v->last<wire::ControlReply>();
    CHECK(r->request_id == 21);
    CHECK(r->status == CALIPER_NET_STATUS_ERROR);
    CHECK(r->error.find("code")->as_string() == CALIPER_NET_ERR_UNSUPPORTED);
    w.spin(50);
    CHECK(p->controls.empty());
}

// The piece form (capability "scene_pieces"): the producer sends a scene program's RESOURCE and then the
// update that names it; the hub forwards both, in that order, to a viewer that announced the capability,
// keeps them as one bundle for a late viewer, gives a viewer without it the structure alone, and drops an
// update whose version did not come.
TEST_CASE("net/hub 3.2: a scene program's RESOURCE travels with the update in the piece form that names it") {
    Runner h;
    World32 w;
    auto p = std::make_unique<SceneProducer>();
    REQUIRE(p->connect(h.url()));
    w.peers.push_back(p.get());
    REQUIRE(w.until([&] { return p->welcomed; }));
    CHECK(std::find(p->hub_caps.begin(), p->hub_caps.end(), CALIPER_NET_CAP_SCENE_PIECES) != p->hub_caps.end());
    auto a = scene_viewer(w, h, {CALIPER_NET_CAP_SCENES, CALIPER_NET_CAP_SCENE_PIECES});
    auto old = scene_viewer(w, h, {CALIPER_NET_CAP_SCENES});
    REQUIRE(w.until([&] { return has_view(*a, "world") && has_view(*old, "world"); }));
    subscribe(*a, 5, "world");
    subscribe(*old, 6, "world");
    REQUIRE(w.until([&] { return a->scene_log(5).size() == 2 && old->scene_log(6).size() == 2; }));
    std::vector<uint8_t> bytes(2 * 32, 0);
    for (uint32_t i = 0; i < 2; ++i) {
        const float pt[4] = {float(i), 0, 0, 1};
        std::memcpy(bytes.data() + i * 32, pt, 16);
        std::memcpy(bytes.data() + i * 32 + 20, &i, 4);
    }
    auto send_piece = [&](bool with_resource, uint32_t count) {
        if (with_resource) {
            wire::Resource r;
            r.program = "scene.1";
            r.resource = "points";
            r.step = 3;
            r.count = 2;
            r.stride = 32;
            r.data = wire::Section::make(CALIPER_NET_SECTION_RESOURCE, CALIPER_NET_DTYPE_U8, {64}, bytes.data(), bytes.size());
            p->send(r);
        }
        for (uint32_t sub : p->scene_subs()) {
            wire::SceneUpdate u;
            u.sub = sub;
            u.view = "world";
            u.seq = ++p->seq;
            u.stamp_ns = mono_ns32();
            u.path = "world/piece";
            u.part = CALIPER_NET_PART_POINTS;
            u.program = std::string("scene.1");
            u.piece = Json::object();
            u.piece.set("name", "points");
            u.piece.set("version", 1);
            u.resources = Json::object();
            u.resources.set("points", Json(uint64_t(count)));
            u.names = {"positions"};
            p->send(u);
        }
    };
    send_piece(true, 2);
    REQUIRE(w.until([&] { return a->scene_log(5).size() == 3; }));
    CHECK(a->scene_log(5).back() == "world/piece:points");
    // The RESOURCE came right before its update, as it came from the producer.
    {
        const auto& in = a->inbox;
        size_t at = 0;
        for (size_t i = 0; i < in.size(); ++i)
            if (auto* u = std::get_if<wire::SceneUpdate>(&in[i].m); u && u->program) at = i;
        REQUIRE(at > 0);
        const auto* r = std::get_if<wire::Resource>(&in[at - 1].m);
        REQUIRE(r);
        CHECK(r->program == "scene.1");
        CHECK(r->count == 2);
        CHECK(std::memcmp(r->data.data, bytes.data(), 64) == 0);
    }
    // The viewer of the per-shape parts gets neither; a batch with its arrays still reaches both.
    p->points("world/cloud", {1, 2, 3});
    REQUIRE(w.until([&] { return old->scene_log(6).size() == 3 && a->scene_log(5).size() == 4; }));
    CHECK(old->scene_log(6).back() == "world/cloud:points");
    CHECK(old->all<wire::Resource>().empty());
    // A late viewer gets the bundle in the hub's state.
    auto b = scene_viewer(w, h, {CALIPER_NET_CAP_SCENES, CALIPER_NET_CAP_SCENE_PIECES});
    subscribe(*b, 9, "world");
    REQUIRE(w.until([&] { return b->scene_log(9).size() == 4; }));
    CHECK(b->scene_log(9) == std::vector<std::string>{"reset", ":lines", "world/cloud:points", "world/piece:points"});
    CHECK(b->all<wire::Resource>().size() == 1);
    // An update whose version did not come, or with another count, is dropped.
    send_piece(false, 2);
    send_piece(true, 5);
    p->points("world/cloud", {4, 5, 6});
    REQUIRE(w.until([&] { return a->scene_log(5).size() == 5; }));
    CHECK(a->scene_log(5).back() == "world/cloud:points");
    REQUIRE(w.until([&] { return h.hub->stats().scene_pieces_dropped >= 2; }));
}


TEST_CASE("net/hub 3.2: scene bundles larger than the send queue drain in order for live and late viewers") {
    constexpr std::uint32_t parts = 8, part_bytes = 4096, stride = 32;
    Runner h(size_t(1) << 20, false, 16 * 1024);
    World32 w;
    auto p = std::make_unique<SceneProducer>();
    REQUIRE(p->connect(h.url()));
    w.peers.push_back(p.get());
    REQUIRE(w.until([&] { return p->welcomed; }));
    auto a = scene_viewer(w, h, {CALIPER_NET_CAP_SCENES, CALIPER_NET_CAP_SCENE_PIECES});
    subscribe(*a, 5, "world");
    REQUIRE(w.until([&] { return a->scene_log(5).size() == 2; }));

    for (std::uint32_t part = 0; part < parts; ++part) {
        const std::vector<std::uint8_t> bytes(part_bytes, std::uint8_t(part));
        wire::Resource r;
        r.program = "scene.1";
        r.resource = "points";
        r.step = 3;
        r.count = parts * part_bytes / stride;
        r.stride = stride;
        r.split = wire::Resource::Split{3, part, parts, part * part_bytes / stride};
        r.data = wire::Section::make(CALIPER_NET_SECTION_RESOURCE, CALIPER_NET_DTYPE_U8,
                                    {part_bytes}, bytes.data(), bytes.size());
        p->send(r);
    }
    for (std::uint32_t sub : p->scene_subs()) {
        wire::SceneUpdate u;
        u.sub = sub;
        u.view = "world";
        u.seq = ++p->seq;
        u.stamp_ns = mono_ns32();
        u.path = "bundle";
        u.part = CALIPER_NET_PART_POINTS;
        u.program = "scene.1";
        u.piece = Json::object();
        u.piece.set("name", "points");
        u.piece.set("version", 1);
        u.resources = Json::object();
        u.resources.set("points", parts * part_bytes / stride);
        u.names = {"positions"};
        p->send(u);
    }
    p->points("later", {1, 2, 3});
    REQUIRE(w.until([&] { return a->scene_log(5).size() == 4; }));

    auto check_bundle = [&](const Client& viewer, std::uint32_t sub, bool state) {
        std::size_t at = 0;
        for (; at < viewer.inbox.size(); ++at)
            if (const auto* u = std::get_if<wire::SceneUpdate>(&viewer.inbox[at].m); u && u->program) break;
        REQUIRE(at < viewer.inbox.size());
        REQUIRE(at >= parts);
        const auto& update = std::get<wire::SceneUpdate>(viewer.inbox[at].m);
        CHECK(update.sub == sub);
        CHECK(update.state == state);
        CHECK(update.path == "bundle");
        for (std::uint32_t part = 0; part < parts; ++part) {
            const auto* r = std::get_if<wire::Resource>(&viewer.inbox[at - parts + part].m);
            REQUIRE(r);
            REQUIRE(r->split);
            CHECK(r->program == "scene.1");
            CHECK(r->split->part == part);
            CHECK(r->split->parts == parts);
            REQUIRE(r->data.size == part_bytes);
            CHECK(r->data.data[0] == part);
            CHECK(r->data.data[part_bytes - 1] == part);
        }
        CHECK(viewer.all<wire::Resource>().size() == parts);
        CHECK(viewer.scene_log(sub).back() == "later:points");
    };
    check_bundle(*a, 5, false);
    auto b = scene_viewer(w, h, {CALIPER_NET_CAP_SCENES, CALIPER_NET_CAP_SCENE_PIECES});
    subscribe(*b, 9, "world");
    REQUIRE(w.until([&] { return b->scene_log(9).size() == 4; }));
    check_bundle(*b, 9, true);
    CHECK(h.hub->stats().viewers == 2);
}
