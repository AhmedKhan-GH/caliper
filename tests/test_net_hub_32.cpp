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
    Runner() {
        HubOptions o;
        o.token = kToken32;
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
