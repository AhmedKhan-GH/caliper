// The hub of wire protocol v3 (src/host/net/hub.h) in ONE process: a Hub runs
// on its own thread on 127.0.0.1, and fake producers and fake viewers, built
// from the WebSocket client of net/ws.h and the codec of net/wire.h, talk to
// it from the test thread. The fake producer behaves as caliper.serve() does
// on a hub connection (python/caliper/_net.py): it answers SUBSCRIBE with a
// resolved data view, sends series backfill, snapshots for open
// subscriptions and records events. No other process, no window.
#include <doctest/doctest.h>

#include "caliper/live_net.h"
#include "net/http_door.h"
#include "net/hub.h"
#include "net/wire.h"
#include "net/ws.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstring>
#include <deque>
#include <filesystem>
#include <fstream>
#include <functional>
#include <map>
#include <memory>
#include <set>
#include <string>
#include <thread>
#include <vector>

#ifdef _WIN32
#  ifndef WIN32_LEAN_AND_MEAN
#    define WIN32_LEAN_AND_MEAN
#  endif
#  ifndef NOMINMAX
#    define NOMINMAX
#  endif
#  include <winsock2.h>
#  include <ws2tcpip.h>
using raw_sock = SOCKET;
#  define RAW_BAD INVALID_SOCKET
#  define raw_close closesocket
#else
#  include <arpa/inet.h>
#  include <netinet/in.h>
#  include <sys/socket.h>
#  include <sys/time.h>
#  include <unistd.h>
using raw_sock = int;
#  define RAW_BAD (-1)
#  define raw_close ::close
#endif
#ifdef __APPLE__
#  include <mach/mach.h>
#endif

namespace wire = caliper_net::wire;
using caliper_net::Bytes;
using caliper_net::Hub;
using caliper_net::HubOptions;
using caliper_net::Json;

namespace {

const std::string kToken = "hub_test_token-0123456789abcdef";

uint64_t mono_ns() {
    using namespace std::chrono;
    return (uint64_t)duration_cast<nanoseconds>(steady_clock::now().time_since_epoch()).count();
}

// ---- the hub on a thread -------------------------------------------------------------------
struct HubRunner {
    std::unique_ptr<Hub> hub;
    std::thread thread;

    explicit HubRunner(HubOptions o = HubOptions()) {
        if (o.token.empty()) o.token = kToken;
        hub = std::make_unique<Hub>(std::move(o));
        std::string err;
        REQUIRE_MESSAGE(hub->listen(&err), err);
        thread = std::thread([this] { hub->run(); });
    }
    ~HubRunner() { stop(); }
    void stop() {
        if (thread.joinable()) {
            hub->stop();
            thread.join();
        }
    }
    uint16_t port() const { return hub->port(); }
    std::string url(const std::string& token = kToken) const {
        return "ws://127.0.0.1:" + std::to_string(port()) + "/v3/stream?token=" + token;
    }
    caliper_net::HubStats stats() const { return hub->stats(); }
};

// ---- a client peer -------------------------------------------------------------------------
struct Msg {
    std::shared_ptr<Bytes> buf;
    wire::Message m;
    uint16_t type = 0;
    uint64_t at_ns = 0;
};

class Peer {
public:
    explicit Peer(caliper_net::Options o = caliper_net::Options()) : client_(o) {}
    virtual ~Peer() = default;

    bool connect(const std::string& url, const std::string& origin = {},
                 const std::vector<std::pair<std::string, std::string>>& headers = {}) {
        std::string err;
        return client_.connect(url, &err, origin, headers);
    }
    void send(const wire::Message& m) {
        std::vector<uint8_t> b;
        REQUIRE(wire::encode(m, &b));
        client_.send(b.data(), b.size());
    }
    void hello_viewer() {
        wire::Hello h;
        h.role = CALIPER_NET_ROLE_VIEWER;
        send(h);
    }
    void poll(int ms = 0) {
        for (auto& e : client_.poll(ms)) {
            using T = caliper_net::Event::Type;
            if (e.type == T::Open) {
                open = true;
                on_open();
            } else if (e.type == T::Closed) {
                closed = true;
                close_code = e.code;
                http_status = e.http_status;
            } else if (e.type == T::Message) {
                Msg msg;
                msg.buf = std::make_shared<Bytes>(std::move(e.data));
                msg.at_ns = mono_ns();
                const wire::DecodeResult r = wire::decode(msg.buf->data(), msg.buf->size(), &msg.m);
                if (!r.ok()) continue;
                msg.type = r.type;
                if (answer_pings)
                    if (auto* pi = std::get_if<wire::Ping>(&msg.m)) send(wire::Pong{pi->nonce});
                inbox.push_back(msg);
                on(inbox.back());
            }
        }
    }
    void close(uint16_t code = 1000) { client_.close(code); }
    size_t queued() const { return client_.queued_bytes(); }

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
    template <typename T>
    size_t count() const {
        return all<T>().size();
    }
    const wire::ViewReply* reply(uint32_t request_id) const {
        for (auto* r : all<wire::ViewReply>())
            if (r->request_id == request_id) return r;
        return nullptr;
    }
    std::vector<const wire::Snapshot*> snapshots(uint32_t sub) const {
        std::vector<const wire::Snapshot*> v;
        for (auto* s : all<wire::Snapshot>())
            if (s->sub == sub) v.push_back(s);
        return v;
    }
    // Values of the series received on `sub`, by absolute index.
    std::map<uint64_t, float> series(uint32_t sub) const {
        std::map<uint64_t, float> out;
        for (auto* s : all<wire::Series>()) {
            if (s->sub != sub) continue;
            const size_t n = s->values.size / 4;
            for (size_t i = 0; i < n; ++i) {
                float f;
                std::memcpy(&f, s->values.data + 4 * i, 4);
                out[s->start + i] = f;
            }
        }
        return out;
    }
    const wire::Sessions* sessions() const { return last<wire::Sessions>(); }
    const wire::Views* views() const { return last<wire::Views>(); }

    void subscribe(uint32_t sub, const std::string& view, uint32_t request_id, Json data_view = Json(),
                   double rate = 0) {
        wire::Subscribe s;
        s.sub = sub;
        s.view = view;
        s.request_id = request_id;
        s.data_view = data_view;
        if (rate > 0) s.max_rate = rate;
        send(s);
    }
    void attach(const std::string& session) {
        wire::Attach a;
        if (!session.empty()) a.session = session;
        send(a);
    }

    std::deque<Msg> inbox;
    bool answer_pings = true;   // as every real peer does
    bool open = false, closed = false;
    uint16_t close_code = 0;
    int http_status = 0;

protected:
    virtual void on_open() {}
    virtual void on(Msg&) {}
    caliper_net::Client client_;
};

// ---- the fake producer ---------------------------------------------------------------------
// Views: "heat" (a heatmap of a (rows 64, cols 64) source "w"), "loss" (a
// series). Resolution: a heatmap data view gets source "w" and its budget
// filled in; a data view with the key "bad" is refused; a series resolves to
// {"budget": {"max_backfill": n}}.
class FakeProducer : public Peer {
public:
    wire::Identity id;
    std::string chosen;                   // the id the hub chose (WELCOME)
    bool welcomed = false;
    struct Sub {
        std::string view;
        Json resolved;
        double rate = 60;
    };
    std::map<uint32_t, Sub> subs;
    std::vector<float> loss;              // values [loss_base, total())
    uint64_t loss_base = 0;
    std::vector<wire::Event> events;
    std::vector<std::string> byes;
    int subscribes = 0, unsubscribes = 0;
    uint64_t seq = 0;
    bool send_views_on_welcome = true;

    FakeProducer(const std::string& sid, uint64_t pid, uint64_t start = 1763200000000ull) {
        id.id = sid;
        id.script = "fake_train.py";
        id.pid = pid;
        id.start_unix_ms = start;
    }

    void hello() {
        wire::Hello h;
        h.role = CALIPER_NET_ROLE_PRODUCER;
        h.capabilities = {CALIPER_NET_CAP_EVENTS};
        h.session = id.to_json();
        send(h);
    }

    static Json views_json() {
        Json views = Json::array();
        Json heat = Json::object();
        heat.set("id", 1);
        heat.set("name", "heat");
        heat.set("kind", (int)CALIPER_NET_KIND_HEATMAP);
        Json src = Json::object();
        src.set("name", "w");
        src.set("dtype", "f32");
        Json dims = Json::array();
        for (const char* d : {"rows", "cols"}) {
            Json e = Json::object();
            e.set("name", d);
            e.set("size", 64);
            dims.push(e);
        }
        src.set("dims", dims);
        Json sources = Json::array();
        sources.push(src);
        heat.set("sources", sources);
        views.push(heat);
        Json loss = Json::object();
        loss.set("id", 2);
        loss.set("name", "loss");
        loss.set("kind", (int)CALIPER_NET_KIND_SERIES);
        loss.set("sources", Json::array());
        views.push(loss);
        return views;
    }

    void send_views() {
        wire::Views v;
        v.views = views_json();
        send(v);
    }

    std::vector<uint32_t> subs_of(const std::string& view) const {
        std::vector<uint32_t> v;
        for (auto& kv : subs)
            if (kv.second.view == view) v.push_back(kv.first);
        return v;
    }

    // One snapshot of "heat" (H x W, every value `fill`) for each open subscription.
    int snapshot(float fill, int H = 4, int W = 8) {
        std::vector<float> data((size_t)H * W, fill);
        int n = 0;
        ++seq;
        for (uint32_t sub : subs_of("heat")) {
            wire::Snapshot s;
            s.sub = sub;
            s.view = "heat";
            s.seq = seq;
            s.stamp_ns = mono_ns();
            s.kind = CALIPER_NET_KIND_HEATMAP;
            s.data_view = subs[sub].resolved;
            s.data = wire::Section::make("data", CALIPER_NET_DTYPE_F32, {H, W}, data.data(), data.size() * 4);
            s.flags = CALIPER_NET_SNAP_AUTO_VMIN | CALIPER_NET_SNAP_AUTO_VMAX;
            s.stats = {fill, fill};
            send(s);
            ++n;
        }
        return n;
    }

    uint64_t total() const { return loss_base + loss.size(); }

    void append(const std::vector<float>& v) {
        const uint64_t start = total();
        loss.insert(loss.end(), v.begin(), v.end());
        for (uint32_t sub : subs_of("loss")) send_series(sub, start, total());
    }

    // Keeps only the newest `keep` values (absolute indices go on).
    void forget(size_t keep) {
        if (loss.size() <= keep) return;
        const size_t drop = loss.size() - keep;
        loss.erase(loss.begin(), loss.begin() + (std::ptrdiff_t)drop);
        loss_base += drop;
    }

    void send_series(uint32_t sub, uint64_t a, uint64_t b) {
        while (a < b) {
            const uint64_t e = std::min<uint64_t>(b, a + CALIPER_NET_SERIES_MAX_VALUES);
            wire::Series s;
            s.sub = sub;
            s.view = "loss";
            s.start = a;
            s.total = total();
            s.stamp_ns = mono_ns();
            s.values = wire::Section::make("values", CALIPER_NET_DTYPE_F32, {(int64_t)(e - a)},
                                           loss.data() + (a - loss_base), (e - a) * 4);
            send(s);
            a = e;
        }
    }

protected:
    void on_open() override { hello(); }

    void on(Msg& msg) override {
        if (auto* w = std::get_if<wire::Welcome>(&msg.m)) {
            welcomed = true;
            wire::Identity got;
            if (wire::Identity::parse(w->session, &got)) chosen = got.id;
            if (send_views_on_welcome) send_views();
        } else if (auto* s = std::get_if<wire::Subscribe>(&msg.m)) {
            ++subscribes;
            subs.erase(s->sub);
            wire::ViewReply r;
            r.request_id = s->request_id;
            r.sub = s->sub;
            r.view = s->view;
            const Json dv = s->data_view.is_object() ? s->data_view : Json::object();
            if (s->view != "heat" && s->view != "loss") {
                Json e = Json::object();
                e.set("code", CALIPER_NET_ERR_UNKNOWN_VIEW);
                e.set("message", "no view");
                r.error = e;
                send(r);
                return;
            }
            if (dv.has("bad")) {
                Json e = Json::object();
                e.set("code", CALIPER_NET_ERR_BAD_DATA_VIEW);
                e.set("message", "bad");
                r.error = e;
                send(r);
                return;
            }
            Json res = Json::object();
            uint64_t back = total() + (1u << 20);
            if (s->view == "heat") {
                res.set("source", "w");
                for (auto& m : dv.members())
                    if (m.first != "budget") res.set(m.first, m.second);
                Json budget = Json::object();
                uint64_t ext = 512;
                if (const Json* b = dv.find("budget"))
                    if (const Json* x = b->find("max_extent")) x->get_u64(&ext);
                budget.set("max_extent", ext);
                budget.set("pool", "mean");
                res.set("budget", budget);
            } else {
                if (const Json* b = dv.find("budget"))
                    if (const Json* x = b->find("max_backfill")) x->get_u64(&back);
                Json budget = Json::object();
                budget.set("max_backfill", back);
                res.set("budget", budget);
            }
            r.data_view = res;
            subs[s->sub] = Sub{s->view, res, s->max_rate ? *s->max_rate : 60.0};
            send(r);
            if (s->view == "loss") {
                const uint64_t from = std::max(loss_base, total() > back ? total() - back : 0);
                if (from < total()) send_series(s->sub, from, total());
            }
        } else if (auto* u = std::get_if<wire::Unsubscribe>(&msg.m)) {
            ++unsubscribes;
            subs.erase(u->sub);
        } else if (auto* e = std::get_if<wire::Event>(&msg.m)) {
            events.push_back(*e);
        } else if (auto* b = std::get_if<wire::Bye>(&msg.m)) {
            byes.push_back(b->reason);
        }
    }
};

// ---- pumping every client -------------------------------------------------------------------
struct World {
    std::vector<Peer*> peers;
    void add(Peer* p) { peers.push_back(p); }
    void once() {
        for (Peer* p : peers) p->poll(0);
    }
    bool until(const std::function<bool()>& done, int ms = 5000) {
        const auto end = std::chrono::steady_clock::now() + std::chrono::milliseconds(ms);
        while (std::chrono::steady_clock::now() < end) {
            once();
            if (done()) return true;
            std::this_thread::sleep_for(std::chrono::microseconds(500));
        }
        once();
        return done();
    }
    void spin(int ms) {
        until([] { return false; }, ms);
    }
};

// A viewer connected and welcomed.
std::unique_ptr<Peer> viewer(World& w, const HubRunner& h) {
    auto v = std::make_unique<Peer>();
    REQUIRE(v->connect(h.url()));
    w.add(v.get());
    REQUIRE(w.until([&] { return v->open; }));
    v->hello_viewer();
    REQUIRE(w.until([&] { return v->count<wire::Welcome>() == 1 && v->sessions(); }));
    return v;
}

std::unique_ptr<FakeProducer> producer(World& w, const HubRunner& h, const std::string& id, uint64_t pid,
                                       uint64_t start = 1763200000000ull) {
    auto p = std::make_unique<FakeProducer>(id, pid, start);
    REQUIRE(p->connect(h.url()));
    w.add(p.get());
    REQUIRE(w.until([&] { return p->welcomed; }));
    return p;
}

const wire::SessionInfo* find_session(const Peer& v, const std::string& id) {
    const wire::Sessions* s = v.sessions();
    if (!s) return nullptr;
    for (const auto& x : s->list)
        if (x.identity.id == id) return &x;
    return nullptr;
}

Json obj(std::initializer_list<std::pair<const char*, Json>> kv) {
    Json j = Json::object();
    for (auto& x : kv) j.set(x.first, x.second);
    return j;
}

// ---- raw HTTP -----------------------------------------------------------------------------------
std::string http(uint16_t port, const std::string& request) {
    raw_sock fd = ::socket(AF_INET, SOCK_STREAM, 0);
    sockaddr_in a{};
    a.sin_family = AF_INET;
    a.sin_port = htons(port);
    inet_pton(AF_INET, "127.0.0.1", &a.sin_addr);
    if (::connect(fd, (sockaddr*)&a, sizeof a) != 0) {
        raw_close(fd);
        return {};
    }
#ifdef _WIN32
    DWORD ms = 2000;
    setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, (const char*)&ms, sizeof ms);
#else
    timeval tv{2, 0};
    setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof tv);
#endif
    ::send(fd, request.data(), (int)request.size(), 0);
    std::string out;
    char buf[65536];
    for (;;) {
        const int r = (int)::recv(fd, buf, sizeof buf, 0);
        if (r <= 0) break;
        out.append(buf, (size_t)r);
    }
    raw_close(fd);
    return out;
}

std::string get(uint16_t port, const std::string& path, const std::string& extra = {}, const char* host = nullptr) {
    const std::string h = host ? host : "127.0.0.1:" + std::to_string(port);
    return http(port, "GET " + path + " HTTP/1.1\r\nHost: " + h + "\r\n" + extra + "\r\n");
}

int status_of(const std::string& response) {
    if (response.size() < 12) return 0;
    return std::atoi(response.substr(9, 3).c_str());
}

size_t rss_bytes() {
#ifdef __APPLE__
    mach_task_basic_info info{};
    mach_msg_type_number_t n = MACH_TASK_BASIC_INFO_COUNT;
    if (task_info(mach_task_self(), MACH_TASK_BASIC_INFO, (task_info_t)&info, &n) == KERN_SUCCESS)
        return (size_t)info.resident_size;
#endif
    return 0;
}

}  // namespace

// ============================================================================
TEST_CASE("net/hub: sessions are listed, a viewer follows the one live session, attach and detach") {
    HubRunner h;
    World w;
    auto pa = producer(w, h, "clp-a", 100);
    CHECK(pa->chosen == "clp-a");
    auto v1 = viewer(w, h);
    const wire::Welcome* wel = v1->last<wire::Welcome>();
    REQUIRE(wel);
    CHECK(wel->role == CALIPER_NET_ROLE_HUB);
    CHECK(wel->session.is_null());
    // The one live session: the viewer is attached to it right after WELCOME.
    REQUIRE(w.until([&] { return v1->views() && v1->views()->session && v1->views()->list.size() == 2; }));
    CHECK(*v1->views()->session == "clp-a");
    REQUIRE(find_session(*v1, "clp-a"));
    CHECK(find_session(*v1, "clp-a")->state == CALIPER_NET_SESSION_LIVE);
    CHECK(find_session(*v1, "clp-a")->views == 2);

    // A second producer: the list grows, the attached viewer stays.
    auto pb = producer(w, h, "clp-b", 200);
    REQUIRE(w.until([&] { return find_session(*v1, "clp-b") != nullptr; }));
    CHECK(*v1->views()->session == "clp-a");
    // Another process proposing a live session's id gets a suffix.
    auto pc = producer(w, h, "clp-a", 300);
    CHECK(pc->chosen == "clp-a-2");

    // A viewer arriving with several live sessions is not attached.
    auto v2 = viewer(w, h);
    w.spin(100);
    CHECK(v2->count<wire::Views>() == 0);
    v2->subscribe(1, "heat", 11);
    REQUIRE(w.until([&] { return v2->reply(11); }));
    CHECK(v2->reply(11)->error.find("code")->as_string() == CALIPER_NET_ERR_NOT_ATTACHED);
    v2->attach("clp-b");
    REQUIRE(w.until([&] { return v2->views() && v2->views()->session; }));
    CHECK(*v2->views()->session == "clp-b");
    v2->attach("nope");
    REQUIRE(w.until([&] { return v2->last<wire::Error>(); }));
    CHECK(v2->last<wire::Error>()->code == CALIPER_NET_ERR_UNKNOWN_SESSION);
    CHECK(v2->last<wire::Error>()->in_reply_to == CALIPER_NET_MSG_ATTACH);
    v2->attach("");
    REQUIRE(w.until([&] { return v2->views() && !v2->views()->session; }));
    CHECK(v2->views()->list.empty());
    CHECK(h.stats().sessions_live == 3);
}

TEST_CASE("net/hub: snapshots are forwarded with the viewer's sub, and data views are routed") {
    HubRunner h;
    World w;
    auto p = producer(w, h, "clp-fwd", 1);
    auto v = viewer(w, h);
    REQUIRE(w.until([&] { return v->views() && v->views()->list.size() == 2; }));

    v->subscribe(7, "heat", 70);
    REQUIRE(w.until([&] { return v->reply(70) && p->subs_of("heat").size() == 1; }));
    const wire::ViewReply* r = v->reply(70);
    CHECK(r->sub == 7);
    CHECK(r->view == "heat");
    CHECK(r->error.is_null());
    CHECK(r->data_view.find("source")->as_string() == "w");
    // The hub numbers its own subscription at the producer.
    const uint32_t up = p->subs_of("heat")[0];

    CHECK(p->snapshot(2.5f, 4, 8) == 1);
    REQUIRE(w.until([&] { return v->snapshots(7).size() == 1; }));
    const wire::Snapshot* s = v->snapshots(7)[0];
    CHECK(s->view == "heat");
    CHECK(s->seq == p->seq);
    CHECK(s->kind == CALIPER_NET_KIND_HEATMAP);
    CHECK(s->data.shape == std::vector<int64_t>{4, 8});
    float f;
    std::memcpy(&f, s->data.data + 4 * 31, 4);
    CHECK(f == 2.5f);
    CHECK(s->data_view == p->subs[up].resolved);
    CHECK(s->stats.size() == 2);

    // A data view request moves this viewer to a new producer subscription.
    wire::ViewRequest q;
    q.request_id = 71;
    q.sub = 7;
    q.view = "heat";
    q.data_view = obj({{"index", obj({{"rows", Json::array()}})}});
    Json idx = Json::object();
    Json rng = Json::array();
    rng.push(0);
    rng.push(2);
    idx.set("rows", rng);
    q.data_view = obj({{"index", idx}});
    v->send(q);
    REQUIRE(w.until([&] { return v->reply(71); }));
    CHECK(v->reply(71)->sub == 7);
    CHECK(v->reply(71)->data_view.find("index") != nullptr);
    REQUIRE(w.until([&] { return p->subs_of("heat").size() == 1 && p->subs_of("heat")[0] != up; }));
    CHECK(p->unsubscribes == 1);   // nobody watches the old data view any more
    p->snapshot(3.0f);
    REQUIRE(w.until([&] { return v->snapshots(7).size() == 2; }));
    CHECK(v->snapshots(7)[1]->data_view.find("index") != nullptr);

    // A data view the producer refuses: the error reaches the viewer with its
    // own ids, and the subscription keeps its data view.
    q.request_id = 72;
    q.data_view = obj({{"bad", true}});
    v->send(q);
    REQUIRE(w.until([&] { return v->reply(72); }));
    CHECK(v->reply(72)->error.find("code")->as_string() == CALIPER_NET_ERR_BAD_DATA_VIEW);
    p->snapshot(4.0f);
    REQUIRE(w.until([&] { return v->snapshots(7).size() == 3; }));
    CHECK(v->snapshots(7)[2]->data_view.find("index") != nullptr);

    // A view the session does not have.
    v->subscribe(8, "nope", 80);
    REQUIRE(w.until([&] { return v->reply(80); }));
    CHECK(v->reply(80)->error.find("code")->as_string() == CALIPER_NET_ERR_UNKNOWN_VIEW);

    // Unsubscribing ends the producer subscription.
    wire::Unsubscribe u;
    u.sub = 7;
    v->send(u);
    REQUIRE(w.until([&] { return p->subs_of("heat").empty(); }));
}

TEST_CASE("net/hub: a late viewer gets the series backfill from the hub, then new values") {
    HubRunner h;
    World w;
    auto p = producer(w, h, "clp-series", 2);
    // The hub subscribes to every series view as soon as VIEWS names it.
    REQUIRE(w.until([&] { return p->subs_of("loss").size() == 1; }));
    std::vector<float> vals(300000);
    for (size_t i = 0; i < vals.size(); ++i) vals[i] = (float)i * 0.5f;
    p->append(vals);
    REQUIRE(w.until([&] { return h.stats().retained_series_bytes == vals.size() * 4; }));

    auto v = viewer(w, h);
    REQUIRE(w.until([&] { return v->views() && v->views()->session; }));
    const int subs_before = p->subscribes;
    v->subscribe(3, "loss", 30);
    REQUIRE(w.until([&] { return v->series(3).size() == vals.size(); }));
    CHECK(p->subscribes == subs_before);   // served from the hub's keeper subscription
    auto got = v->series(3);
    CHECK(got.begin()->first == 0);
    CHECK(got[123457] == 123457 * 0.5f);
    CHECK(got[299999] == 299999 * 0.5f);
    // Split into messages of at most CALIPER_NET_SERIES_MAX_VALUES values.
    for (auto* s : v->all<wire::Series>()) CHECK(s->values.size / 4 <= CALIPER_NET_SERIES_MAX_VALUES);

    p->append({1.0f, 2.0f, 3.0f});
    REQUIRE(w.until([&] { return v->series(3).size() == vals.size() + 3; }));
    CHECK(v->series(3)[300002] == 3.0f);
    CHECK(v->all<wire::Series>().back()->total == 300003);

    // A viewer that asks for the last 100 values only.
    auto v2 = viewer(w, h);
    REQUIRE(w.until([&] { return v2->views() && v2->views()->session; }));
    v2->subscribe(4, "loss", 40, obj({{"budget", obj({{"max_backfill", 100}})}}));
    REQUIRE(w.until([&] { return v2->reply(40) && v2->series(4).size() == 100; }));
    w.spin(50);
    CHECK(v2->series(4).size() == 100);
    CHECK(v2->series(4).begin()->first == 300003 - 100);
    CHECK(v2->reply(40)->data_view.find("budget")->find("max_backfill")->as_double() == 100);
}

TEST_CASE("net/hub: viewers with equal resolved data views share one producer subscription") {
    HubRunner h;
    World w;
    auto p = producer(w, h, "clp-share", 3);
    auto a = viewer(w, h);
    auto b = viewer(w, h);
    auto c = viewer(w, h);
    for (Peer* v : {a.get(), b.get(), c.get()})
        REQUIRE(w.until([&] { return v->views() && v->views()->session; }));

    a->subscribe(1, "heat", 1);
    REQUIRE(w.until([&] { return a->reply(1); }));
    // {"source": "w"} resolves to the same data view as {}: the hub merges.
    b->subscribe(5, "heat", 2, obj({{"source", "w"}}));
    REQUIRE(w.until([&] { return b->reply(2); }));
    REQUIRE(w.until([&] { return p->subs_of("heat").size() == 1; }));
    const int subs_now = p->subscribes;
    // Asking again with {} needs no new producer subscription.
    c->subscribe(9, "heat", 3);
    REQUIRE(w.until([&] { return c->reply(3); }));
    w.spin(50);
    CHECK(p->subscribes == subs_now);
    REQUIRE(w.until([&] { return h.stats().channels_open == 2; }));   // heat, and the series keeper of loss

    CHECK(p->snapshot(7.0f) == 1);
    REQUIRE(w.until([&] {
        return a->snapshots(1).size() == 1 && b->snapshots(5).size() == 1 && c->snapshots(9).size() == 1;
    }));

    // The producer subscription runs at the largest max_rate among its viewers.
    const uint32_t up = p->subs_of("heat")[0];
    CHECK(p->subs[up].rate == doctest::Approx(60));
    c->subscribe(9, "heat", 4, Json(), 120);
    REQUIRE(w.until([&] { return c->reply(4) && p->subs.count(up) && p->subs[up].rate > 100; }));
    CHECK(p->subs[up].rate == doctest::Approx(120));
    wire::Unsubscribe u;
    u.sub = 9;
    c->send(u);
    REQUIRE(w.until([&] { return p->subs.count(up) && p->subs[up].rate < 100; }));

    // The last viewer leaving ends it at the producer.
    u.sub = 1;
    a->send(u);
    u.sub = 5;
    b->send(u);
    REQUIRE(w.until([&] { return p->subs_of("heat").empty(); }));
    // The last snapshot stays with the hub: a new viewer sees it at once.
    auto d = viewer(w, h);
    REQUIRE(w.until([&] { return d->views() && d->views()->session; }));
    d->subscribe(2, "heat", 5);
    REQUIRE(w.until([&] { return d->reply(5) && d->snapshots(2).size() == 1; }));
    CHECK(d->snapshots(2)[0]->seq == p->seq);
}

TEST_CASE("net/hub: events reach the producer of the viewer's session, with its own sub") {
    HubRunner h;
    World w;
    auto pa = producer(w, h, "clp-ev-a", 10);
    auto pb = producer(w, h, "clp-ev-b", 11);
    auto v = viewer(w, h);
    v->attach("clp-ev-b");
    REQUIRE(w.until([&] { return v->views() && v->views()->session; }));
    v->subscribe(42, "heat", 1);
    REQUIRE(w.until([&] { return v->reply(1) && pb->subs_of("heat").size() == 1; }));
    wire::Event e;
    e.view = "heat";
    e.type = CALIPER_NET_EVENT_PICK;
    e.detail = obj({{"row", 3}, {"col", 4}});
    e.sub = 42;
    e.seq = 9;
    v->send(e);
    REQUIRE(w.until([&] { return pb->events.size() == 1; }));
    CHECK(pa->events.empty());
    CHECK(pb->events[0].type == CALIPER_NET_EVENT_PICK);
    CHECK(pb->events[0].detail == e.detail);
    CHECK(pb->events[0].seq == 9u);
    REQUIRE(pb->events[0].sub);
    CHECK(*pb->events[0].sub == pb->subs_of("heat")[0]);
    REQUIRE(w.until([&] { return h.stats().events_forwarded == 1; }));
}

TEST_CASE("net/hub: a producer that leaves keeps its session viewable; a resumed one replaces it") {
    HubRunner h;
    World w;
    auto p = producer(w, h, "clp-leave", 20, 555);
    auto v = viewer(w, h);
    REQUIRE(w.until([&] { return v->views() && v->views()->session; }));
    v->subscribe(1, "heat", 1);
    v->subscribe(2, "loss", 2);
    REQUIRE(w.until([&] { return v->reply(1) && v->reply(2) && p->subs_of("heat").size() == 1; }));
    p->snapshot(5.0f);
    p->append({1, 2, 3, 4});
    REQUIRE(w.until([&] { return v->snapshots(1).size() == 1 && v->series(2).size() == 4; }));

    // The producer says BYE and goes: the session is ended, still listed.
    wire::Bye bye;
    bye.reason = CALIPER_NET_BYE_NORMAL;
    p->send(bye);
    p->close(1000);
    REQUIRE(w.until([&] {
        const auto* s = find_session(*v, "clp-leave");
        return s && s->state == CALIPER_NET_SESSION_ENDED;
    }));
    CHECK(*v->views()->session == "clp-leave");   // still attached

    // A later viewer attaches to the ended session and gets what it kept.
    auto late = viewer(w, h);
    // With no live session there is no default: it attaches by id.
    w.spin(50);
    CHECK(late->count<wire::Views>() == 0);
    late->attach("clp-leave");
    REQUIRE(w.until([&] { return late->views() && late->views()->session; }));
    CHECK(*late->views()->session == "clp-leave");
    late->subscribe(3, "heat", 3);
    late->subscribe(4, "loss", 4);
    REQUIRE(w.until([&] { return late->snapshots(3).size() == 1 && late->series(4).size() == 4; }));
    CHECK(late->snapshots(3)[0]->seq == p->seq);
    CHECK(late->series(4)[3] == 4.0f);

    // The same process comes back (same id, pid and start): the session resumes,
    // its viewers stay attached and the hub subscribes again.
    auto p2 = std::make_unique<FakeProducer>("clp-leave", 20, 555);
    p2->loss = p->loss;
    REQUIRE(p2->connect(h.url()));
    w.add(p2.get());
    REQUIRE(w.until([&] { return p2->welcomed && p2->subs_of("heat").size() == 1; }));
    CHECK(p2->chosen == "clp-leave");
    REQUIRE(w.until([&] {
        const auto* s = find_session(*v, "clp-leave");
        return s && s->state == CALIPER_NET_SESSION_LIVE;
    }));
    p2->snapshot(6.0f);
    REQUIRE(w.until([&] { return v->snapshots(1).size() >= 2 && late->snapshots(3).size() >= 2; }));
    CHECK(v->snapshots(1).back()->stats[0] == 6.0);

    // Again while it is still connected: the old connection gets BYE replaced.
    auto p3 = std::make_unique<FakeProducer>("clp-leave", 20, 555);
    REQUIRE(p3->connect(h.url()));
    w.add(p3.get());
    REQUIRE(w.until([&] { return p3->welcomed && p2->closed; }));
    CHECK(p2->byes == std::vector<std::string>{CALIPER_NET_BYE_REPLACED});
    CHECK(p2->close_code == CALIPER_NET_CLOSE_NORMAL);
    REQUIRE(w.until([&] { return p3->subs_of("heat").size() == 1; }));
    p3->snapshot(8.0f);
    REQUIRE(w.until([&] { return v->snapshots(1).back()->stats[0] == 8.0; }));
    CHECK(h.stats().sessions_live == 1);
    CHECK(h.stats().sessions_ended == 0);
}

TEST_CASE("net/hub: a new process under an ended session's id takes it over, as a kernel restart") {
    HubRunner h;
    World w;
    auto p = producer(w, h, "notebook", 30);
    auto v = viewer(w, h);
    REQUIRE(w.until([&] { return v->views() && v->views()->session; }));
    p->close(1000);
    REQUIRE(w.until([&] { return find_session(*v, "notebook") && find_session(*v, "notebook")->state == "ended"; }));
    const size_t views_before = v->count<wire::Views>();
    auto p2 = producer(w, h, "notebook", 31);
    CHECK(p2->chosen == "notebook");
    REQUIRE(w.until([&] { return v->count<wire::Views>() > views_before && find_session(*v, "notebook") &&
                                 find_session(*v, "notebook")->identity.pid == 31; }));
    CHECK(*v->views()->session == "notebook");
    CHECK(h.stats().sessions_ended == 0);
}

TEST_CASE("net/hub: the token, Host and Origin checks, and the static files") {
    namespace fs = std::filesystem;
    const fs::path dir = fs::temp_directory_path() / "caliper-hub-assets";
    std::error_code ec;
    fs::remove_all(dir, ec);
    fs::create_directories(dir / "fonts");
    std::ofstream(dir / "index.html") << "<!doctype html><title>Caliper</title>";
    std::ofstream(dir / "caliper-view.js") << "export {}";
    std::ofstream(dir / "views.wasm", std::ios::binary) << std::string("\0asm", 4);
    std::ofstream(dir / "fonts" / "a.woff2") << "font";
    std::ofstream(dir / ".hidden") << "secret";
    std::ofstream(dir.parent_path() / "caliper-hub-outside.txt") << "outside";

    HubOptions o;
    o.assets_dir = dir.string();
    HubRunner h(o);
    World w;

    Peer bad_token;
    REQUIRE(bad_token.connect(h.url("wrong_token-0123456789abcdef")));
    w.add(&bad_token);
    REQUIRE(w.until([&] { return bad_token.closed; }));
    CHECK(bad_token.http_status == 403);

    Peer bad_host;
    REQUIRE(bad_host.connect(h.url(), {}, {{"Host", "evil.example:" + std::to_string(h.port())}}));
    w.add(&bad_host);
    REQUIRE(w.until([&] { return bad_host.closed; }));
    CHECK(bad_host.http_status == 403);

    Peer bad_origin;
    REQUIRE(bad_origin.connect(h.url(), "http://evil.example"));
    w.add(&bad_origin);
    REQUIRE(w.until([&] { return bad_origin.closed; }));
    CHECK(bad_origin.http_status == 403);

    Peer notebook;   // JupyterLab on loopback
    REQUIRE(notebook.connect(h.url(), "http://localhost:8888"));
    w.add(&notebook);
    REQUIRE(w.until([&] { return notebook.open; }));

    REQUIRE(w.until([&] { return h.stats().refused_origin == 1; }));
    const auto st = h.stats();
    CHECK(st.refused_token == 1);
    CHECK(st.refused_host == 1);
    CHECK(st.refused_origin == 1);

    const uint16_t port = h.port();
    std::string r = get(port, "/");
    CHECK(status_of(r) == 200);
    CHECK(r.find("Content-Type: text/html; charset=utf-8") != std::string::npos);
    CHECK(r.find("Cache-Control: no-store") != std::string::npos);
    CHECK(r.find("<title>Caliper</title>") != std::string::npos);
    r = get(port, "/views.wasm");
    CHECK(status_of(r) == 200);
    CHECK(r.find("Content-Type: application/wasm") != std::string::npos);
    r = get(port, "/caliper-view.js", "Origin: http://localhost:8888\r\n");
    CHECK(r.find("Content-Type: text/javascript; charset=utf-8") != std::string::npos);
    CHECK(r.find("Access-Control-Allow-Origin: http://localhost:8888") != std::string::npos);
    CHECK(r.find("Vary: Origin") != std::string::npos);
    r = get(port, "/caliper-view.js", "Origin: http://evil.example\r\n");
    CHECK(status_of(r) == 200);
    CHECK(r.find("Access-Control-Allow-Origin") == std::string::npos);
    CHECK(status_of(get(port, "/fonts/a.woff2")) == 200);
    CHECK(status_of(get(port, "/.hidden")) == 404);
    CHECK(status_of(get(port, "/../caliper-hub-outside.txt")) == 404);
    CHECK(status_of(get(port, "/%2e%2e/caliper-hub-outside.txt")) == 404);
    CHECK(status_of(get(port, "/fonts/../../caliper-hub-outside.txt")) == 404);
    CHECK(status_of(get(port, "/missing.js")) == 404);
    CHECK(status_of(get(port, "/", {}, "evil.example")) == 403);
    CHECK(status_of(http(port, "POST / HTTP/1.1\r\nHost: 127.0.0.1:" + std::to_string(port) + "\r\n\r\n")) == 405);
    CHECK(status_of(get(port, "/v3/stream?token=" + kToken)) == 426);
    CHECK(status_of(get(port, "/v3/stream?token=nope")) == 403);
    fs::remove_all(dir, ec);
    fs::remove(dir.parent_path() / "caliper-hub-outside.txt", ec);

    // Without a folder, "/" says where the viewer page is.
    HubRunner bare;
    r = get(bare.port(), "/");
    CHECK(status_of(r) == 200);
    CHECK(r.find("caliper/_web/") != std::string::npos);
    CHECK(status_of(get(bare.port(), "/caliper-view.js")) == 404);
}

TEST_CASE("net/hub: limits on viewers, producers, subscriptions and ended sessions") {
    HubOptions o;
    o.max_viewers = 2;
    o.max_producers = 2;
    o.max_ended_sessions = 1;
    HubRunner h(o);
    World w;
    auto p1 = producer(w, h, "lim-1", 1);
    auto p2 = producer(w, h, "lim-2", 2);
    FakeProducer p3("lim-3", 3);
    REQUIRE(p3.connect(h.url()));
    w.add(&p3);
    REQUIRE(w.until([&] { return p3.closed; }));
    CHECK(p3.close_code == CALIPER_NET_CLOSE_SERVER_FULL);
    REQUIRE(p3.last<wire::Error>());
    CHECK(p3.last<wire::Error>()->code == CALIPER_NET_ERR_LIMIT);
    CHECK(p3.last<wire::Error>()->fatal);

    auto v1 = viewer(w, h);
    auto v2 = viewer(w, h);
    Peer v3;
    REQUIRE(v3.connect(h.url()));
    w.add(&v3);
    REQUIRE(w.until([&] { return v3.open; }));
    v3.hello_viewer();
    REQUIRE(w.until([&] { return v3.closed; }));
    CHECK(v3.close_code == CALIPER_NET_CLOSE_SERVER_FULL);

    // At most 256 subscriptions per viewer connection.
    v1->attach("lim-1");
    REQUIRE(w.until([&] { return v1->views() && v1->views()->session; }));
    for (uint32_t i = 0; i < CALIPER_NET_MAX_SUBSCRIPTIONS; ++i) v1->subscribe(1000 + i, "loss", 1000 + i);
    REQUIRE(w.until([&] { return v1->count<wire::ViewReply>() == CALIPER_NET_MAX_SUBSCRIPTIONS; }));
    v1->subscribe(9999, "loss", 9999);
    REQUIRE(w.until([&] { return v1->reply(9999); }));
    CHECK(v1->reply(9999)->error.find("code")->as_string() == CALIPER_NET_ERR_LIMIT);

    // One ended session is kept: the older one goes when the next one ends.
    p1->close(1000);
    REQUIRE(w.until([&] { return h.stats().sessions_ended == 1; }));
    p2->close(1000);
    REQUIRE(w.until([&] { return h.stats().sessions_ended == 1 && h.stats().sessions_live == 0 &&
                                 h.stats().sessions_evicted == 1; }));
    REQUIRE(w.until([&] { return find_session(*v2, "lim-2") && !find_session(*v2, "lim-1"); }));
}

TEST_CASE("net/hub: a session that ended is forgotten after its time; silence is pinged, then closed") {
    HubOptions o;
    o.ended_ttl_ms = 300;
    o.handshake_timeout_ms = 300;
    o.ping_interval_ms = 200;
    o.idle_timeout_ms = 600;
    HubRunner h(o);
    World w;
    auto p = producer(w, h, "ttl", 1);
    auto v = viewer(w, h);
    REQUIRE(w.until([&] { return v->views() && v->views()->session; }));
    p->close(1000);
    REQUIRE(w.until([&] { return find_session(*v, "ttl") && find_session(*v, "ttl")->state == "ended"; }));
    REQUIRE(w.until([&] { return v->sessions() && v->sessions()->list.empty(); }, 3000));
    REQUIRE(w.until([&] { return v->views() && !v->views()->session; }));

    // A connection that never says HELLO is closed with 4003.
    Peer mute;
    REQUIRE(mute.connect(h.url()));
    w.add(&mute);
    REQUIRE(w.until([&] { return mute.closed; }, 3000));
    CHECK(mute.close_code == CALIPER_NET_CLOSE_HANDSHAKE_TIMEOUT);

    // A viewer that answers PING stays; one that answers nothing gets PING
    // after 200 ms of silence and is closed with 4004 after 600 ms.
    auto quiet = viewer(w, h);
    quiet->answer_pings = false;
    REQUIRE(w.until([&] { return quiet->count<wire::Ping>() >= 1; }, 3000));
    REQUIRE(w.until([&] { return quiet->closed; }, 3000));
    CHECK(quiet->close_code == CALIPER_NET_CLOSE_IDLE_TIMEOUT);
    CHECK_FALSE(v->closed);
    CHECK(v->count<wire::Ping>() >= 1);
}

TEST_CASE("net/hub: a slow viewer does not slow a fast one or the producer") {
    HubRunner h;
    World w;
    auto p = producer(w, h, "clp-slow", 5);
    auto fast = viewer(w, h);
    auto slow = viewer(w, h);
    for (Peer* v : {fast.get(), slow.get()}) REQUIRE(w.until([&] { return v->views() && v->views()->session; }));
    fast->subscribe(1, "heat", 1, Json(), 240);
    slow->subscribe(1, "heat", 1, Json(), 240);
    REQUIRE(w.until([&] { return fast->reply(1) && slow->reply(1) && p->subs_of("heat").size() == 1; }));

    // The slow viewer stops reading. The producer sends 1 MiB snapshots, each
    // as soon as the fast viewer has the previous one: the fast viewer must
    // get every one of them while the slow one is stuck.
    const int N = 240;
    size_t max_producer_queue = 0;
    uint64_t max_hub_queue = 0;
    bool kept_up = true;
    for (int i = 0; i < N && kept_up; ++i) {
        p->snapshot((float)i, 512, 512);
        const auto limit = std::chrono::steady_clock::now() + std::chrono::seconds(5);
        for (;;) {
            p->poll(0);
            fast->poll(1);
            max_producer_queue = std::max(max_producer_queue, p->queued());
            if (!fast->snapshots(1).empty() && fast->snapshots(1).back()->seq == p->seq) break;
            if (std::chrono::steady_clock::now() > limit) {
                kept_up = false;
                break;
            }
        }
        max_hub_queue = std::max<uint64_t>(max_hub_queue, h.stats().queued_bytes);
        fast->inbox.erase(fast->inbox.begin(), fast->inbox.end() - 1);   // keep the last one only
    }
    MESSAGE("the hub's send queues peaked at " << max_hub_queue << " bytes; the producer's at " << max_producer_queue);
    CHECK(kept_up);
    // The producer's queue never backs up: the hub reads it at once.
    CHECK(max_producer_queue < (size_t)(8u << 20));
    // The slow viewer holds at most one snapshot in the hub's send queue.
    CHECK(max_hub_queue < (uint64_t)(4u << 20));
    REQUIRE(w.until([&] { return h.stats().snapshots_superseded > 0; }));

    // When it reads again it gets the newest snapshot, not a backlog.
    REQUIRE(w.until([&] { return !slow->snapshots(1).empty() && slow->snapshots(1).back()->seq == p->seq; }));
    CHECK(slow->snapshots(1).size() < (size_t)N / 2);
}

TEST_CASE("net/hub: retained memory stays bounded over a long run") {
    HubOptions o;
    o.max_retained_bytes = size_t(3) << 19;   // 1.5 MiB: idle snapshots must be evicted
    o.series_retain = 50000;
    o.max_idle_channels = 8;
    HubRunner h(o);
    World w;
    auto p = producer(w, h, "clp-long", 6);
    auto v = viewer(w, h);
    REQUIRE(w.until([&] { return v->views() && v->views()->session; }));
    v->subscribe(1, "heat", 1, Json(), 240);
    v->subscribe(2, "loss", 2, Json(), 240);
    REQUIRE(w.until([&] { return v->reply(1) && v->reply(2) && p->subs_of("heat").size() == 1; }));

    std::vector<float> chunk(1000);
    size_t rss_warm = 0;
    uint64_t max_retained = 0;
    uint32_t req = 100;
    for (int i = 0; i < 1500; ++i) {
        p->snapshot((float)i, 256, 256);
        for (float& x : chunk) x = (float)i;
        p->append(chunk);
        p->forget(4096);
        if (i % 10 == 0) {
            // Zooming: a new data view every ten steps leaves an idle channel behind.
            wire::ViewRequest q;
            q.request_id = ++req;
            q.sub = 1;
            q.view = "heat";
            q.data_view = obj({{"budget", obj({{"max_extent", (uint64_t)(16 + i)}})}});
            v->send(q);
        }
        for (int k = 0; k < 3; ++k) w.once();
        if (i % 50 == 0) {
            v->inbox.clear();   // the test's own memory
            p->inbox.clear();
            const auto st = h.stats();
            max_retained = std::max<uint64_t>(max_retained, st.retained_snapshot_bytes + st.retained_series_bytes);
            // Channels waiting for the producer's reply to a data view request
            // come on top of the bound; they are few unless the producer lags.
            CHECK(st.channels <= 2 + 8 + CALIPER_NET_MAX_SUBSCRIPTIONS);
        }
        if (i == 300) rss_warm = rss_bytes();
    }
    // Once every reply is in: the subscribed heat and loss channels, the loss
    // keeper and at most eight idle ones.
    REQUIRE(w.until([&] { return v->reply(req) && h.stats().channels <= 2 + 8 + 1; }, 20000));
    const auto st = h.stats();
    CHECK(st.retained_series_bytes <= 50000 * 4);
    CHECK(max_retained <= o.max_retained_bytes);
    const size_t rss_end = rss_bytes();
    MESSAGE("retained bytes peaked at " << max_retained << "; process RSS " << (rss_warm >> 20) << " MiB at step 300, "
                                        << (rss_end >> 20) << " MiB at step 1500");
    if (rss_warm) CHECK(rss_end < rss_warm + (size_t(16) << 20));
}

TEST_CASE("net/hub: shutting down says BYE shutdown to every connection and closes with 1001") {
    HubRunner h;
    World w;
    auto p = producer(w, h, "clp-down", 7);
    auto v = viewer(w, h);
    REQUIRE(w.until([&] { return v->views() && v->views()->session; }));
    const uint16_t port = h.port();
    std::thread stopper([&] { h.stop(); });
    REQUIRE(w.until([&] { return p->closed && v->closed; }));
    stopper.join();
    CHECK(p->byes == std::vector<std::string>{CALIPER_NET_BYE_SHUTDOWN});
    REQUIRE(v->last<wire::Bye>());
    CHECK(v->last<wire::Bye>()->reason == CALIPER_NET_BYE_SHUTDOWN);
    CHECK(v->close_code == CALIPER_NET_CLOSE_GOING_AWAY);
    CHECK(p->close_code == CALIPER_NET_CLOSE_GOING_AWAY);
    // Nothing listens any more.
    CHECK(get(port, "/").empty());
}

TEST_CASE("net/hub: the token and the hub file") {
    namespace fs = std::filesystem;
    const std::string t = caliper_net::random_token();
    CHECK(t.size() == 32);
    CHECK(caliper_net::valid_token(t));
    CHECK(caliper_net::random_token() != t);
    CHECK_FALSE(caliper_net::valid_token("short"));
    CHECK_FALSE(caliper_net::valid_token("has a space-0123456789abcdef"));

    const fs::path dir = fs::temp_directory_path() / "caliper-hub-file";
    std::error_code ec;
    fs::remove_all(dir, ec);
    fs::create_directories(dir);
    const std::string tf = (dir / "token").string();
    std::ofstream(tf) << t << "\n";
    std::string got, err;
#ifndef _WIN32
    fs::permissions(tf, fs::perms::owner_read | fs::perms::owner_write | fs::perms::group_read);
    CHECK_FALSE(caliper_net::read_token_file(tf, &got, &err));
    CHECK(err.find("chmod 600") != std::string::npos);
    fs::permissions(tf, fs::perms::owner_read | fs::perms::owner_write);
#endif
    REQUIRE(caliper_net::read_token_file(tf, &got, &err));
    CHECK(got == t);

    const std::string hf = (dir / "sub" / "hub.json").string();
    const std::string text = caliper_net::hub_file_text(7340, t, 4242);
    REQUIRE(caliper_net::write_hub_file(hf, text, &err));
    Json j;
    std::ifstream f(hf);
    std::string body((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    REQUIRE(Json::parse(body.data(), body.size(), &j, &err));
    CHECK(j.find("token")->as_string() == t);
    CHECK(j.find("stream_url")->as_string() == "ws://127.0.0.1:7340/v3/stream");
    CHECK(j.find("url")->as_string() == "http://127.0.0.1:7340/?token=" + t);
#ifndef _WIN32
    CHECK((fs::status(hf).permissions() & (fs::perms::group_all | fs::perms::others_all)) == fs::perms::none);
#endif
    caliper_net::remove_hub_file(hf, "something else");
    CHECK(fs::exists(hf));
    caliper_net::remove_hub_file(hf, text);
    CHECK_FALSE(fs::exists(hf));
    fs::remove_all(dir, ec);
}

// The relay's latency and throughput: a 512 x 512 f32 heatmap (1 MiB) at 60 Hz
// to four viewers on loopback, in one process so that one clock times it. The
// producer runs on the test thread and each viewer on a thread of its own, so
// the numbers are the hub's (and the sockets'), not a shared test loop's. The
// viewers subscribe at the default max_rate, 60. Skipped by default; run with
//   caliper_tests -tc="net/hub: relay*" --no-skip
TEST_CASE("net/hub: relay latency and throughput, 512 x 512 heatmap at 60 Hz to four viewers" *
          doctest::skip(true)) {
    HubRunner h;
    World w;
    auto p = producer(w, h, "clp-bench", 8);
    std::vector<std::unique_ptr<Peer>> vs;
    for (int i = 0; i < 4; ++i) {
        vs.push_back(viewer(w, h));
        Peer* v = vs.back().get();
        REQUIRE(w.until([&] { return v->views() && v->views()->session; }));
        v->subscribe(1, "heat", 1);
        REQUIRE(w.until([&] { return v->reply(1); }));
    }
    REQUIRE(w.until([&] { return p->subs_of("heat").size() == 1; }));
    struct Got {
        std::vector<double> lat_ms;
        size_t bytes = 0;
    };
    std::vector<Got> got(vs.size());
    std::atomic<bool> done{false};
    std::vector<std::thread> threads;
    for (size_t i = 0; i < vs.size(); ++i)
        threads.emplace_back([&, i] {
            Peer& v = *vs[i];
            v.inbox.clear();
            while (!done.load()) {
                v.poll(5);
                for (const Msg& m : v.inbox)
                    if (auto* s = std::get_if<wire::Snapshot>(&m.m)) {
                        got[i].lat_ms.push_back((double)(m.at_ns - s->stamp_ns) / 1e6);
                        got[i].bytes += m.buf->size();
                    }
                v.inbox.clear();
            }
        });
    const int N = 600;   // ten seconds
    const auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < N; ++i) {
        p->snapshot((float)i, 512, 512);
        const auto next = t0 + std::chrono::microseconds((int64_t)(i + 1) * 16667);
        while (std::chrono::steady_clock::now() < next) p->poll(1);
    }
    const double secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    std::this_thread::sleep_for(std::chrono::milliseconds(300));
    done = true;
    for (auto& t : threads) t.join();
    std::vector<double> lat;
    size_t bytes = 0;
    for (auto& g : got) {
        lat.insert(lat.end(), g.lat_ms.begin(), g.lat_ms.end());
        bytes += g.bytes;
    }
    std::sort(lat.begin(), lat.end());
    REQUIRE(!lat.empty());
    const auto st = h.stats();
    MESSAGE("snapshots received " << lat.size() << " of " << 4 * N << " (superseded in the hub "
                                  << st.snapshots_superseded << "), " << (double)bytes / secs / (1 << 20)
                                  << " MiB/s to four viewers; latency from the producer's send to a viewer's receive: p50 "
                                  << lat[lat.size() / 2] << " ms, p99 " << lat[lat.size() * 99 / 100] << " ms, max "
                                  << lat.back() << " ms");
}
