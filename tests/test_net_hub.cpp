// The hub of wire protocol v3 (src/host/net/hub.h) in ONE process: a Hub runs
// on its own thread on 127.0.0.1, and fake producers and fake viewers, built
// from the WebSocket client of net/ws.h and the codec of net/wire.h, talk to
// it from the test thread. The fake producer behaves as caliper.serve() does
// on a hub connection (python/caliper/_net.py): it answers SUBSCRIBE with a
// resolved data view, sends series backfill, snapshots for open
// subscriptions and records events. No other process, no window.
#include <doctest/doctest.h>

#include "caliper/live_net.h"
#include "api/api_server.h"
#include "net/http_door.h"
#include "net/hub.h"
#include "net/wire.h"
#include "net/ws.h"
#include "test_temp.h"

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
#include <optional>
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
        std::string error;
        REQUIRE_MESSAGE(wire::encode(m, &b, &error), wire::type_name(wire::message_type(m)), ": ", error);
        client_.send(b.data(), b.size());
    }
    void hello_viewer(std::vector<std::string> caps = {}) {
        wire::Hello h;
        h.role = CALIPER_NET_ROLE_VIEWER;
        h.capabilities = std::move(caps);
        send(h);
    }
    // 3.1: CONTROL from this viewer.
    void control(uint32_t request_id, const std::string& op, const std::string& name = {}, Json value = Json()) {
        wire::Control c;
        c.request_id = request_id;
        c.op = op;
        if (!name.empty()) c.name = name;
        c.value = value;
        send(c);
    }
    std::vector<const wire::ControlReply*> control_replies(uint32_t request_id) const {
        std::vector<const wire::ControlReply*> v;
        for (auto* r : all<wire::ControlReply>())
            if (r->request_id == request_id) v.push_back(r);
        return v;
    }
    // The position of the first message of a type in the inbox, -1 when none.
    int first_index(uint16_t type) const {
        for (size_t i = 0; i < inbox.size(); ++i)
            if (inbox[i].type == type) return (int)i;
        return -1;
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
    void close(uint16_t code = 1000, const std::string& reason = {}) { client_.close(code, reason); }
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
    // 3.1
    std::vector<std::string> caps = {CALIPER_NET_CAP_EVENTS};
    std::optional<wire::Welcome> welcome;
    bool send_controls_on_welcome = false;         // CONTROLS right after VIEWS
    std::vector<wire::Control> controls_in;        // the CONTROLs the hub forwarded
    std::vector<uint32_t> queued_runs;             // request ids of runs answered `queued`

    FakeProducer(const std::string& sid, uint64_t pid, uint64_t start = 1763200000000ull) {
        id.id = sid;
        id.script = "fake_train.py";
        id.pid = pid;
        id.start_unix_ms = start;
    }

    void hello() {
        wire::Hello h;
        h.role = CALIPER_NET_ROLE_PRODUCER;
        h.capabilities = caps;
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

    // With `selection` announced (3.1), the rows of "heat" index the item space
    // "samples" and the series "loss" the space "step".
    void send_views() {
        wire::Views v;
        v.views = views_json();
        if (std::find(caps.begin(), caps.end(), CALIPER_NET_CAP_SELECTION) != caps.end()) {
            Json with = Json::array();
            for (const Json& e : v.views.items()) {
                Json x = e;
                if (x.find("name")->as_string() == "loss") {
                    x.set("items", "step");
                } else {
                    Json src = x.find("sources")->at(0);
                    Json dims = Json::array();
                    for (size_t i = 0; i < src.find("dims")->size(); ++i) {
                        Json d = src.find("dims")->at(i);
                        if (i == 0) d.set("items", "samples");
                        dims.push(d);
                    }
                    src.set("dims", dims);
                    Json srcs = Json::array();
                    srcs.push(src);
                    x.set("sources", srcs);
                }
                with.push(x);
            }
            v.views = with;
        }
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
            if (std::find(caps.begin(), caps.end(), CALIPER_NET_CAP_SELECTION) != caps.end()) {
                // 3.1 viewers map a selection onto the source: the 64 x 64
                // source pooled into the H x W payload.
                s.axes = Json::array();
                for (auto [dim, n] : {std::pair<const char*, int>{"rows", H}, {"cols", W}}) {
                    Json a = Json::object();
                    a.set("dim", dim);
                    a.set("start", 0);
                    a.set("step", 1);
                    a.set("count", 64);
                    a.set("size", n);
                    s.axes.push(a);
                }
            }
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
            welcome = *w;
            wire::Identity got;
            if (wire::Identity::parse(w->session, &got)) chosen = got.id;
            if (send_views_on_welcome) send_views();
            if (send_controls_on_welcome) send_controls("running", 0.001);
        } else if (auto* c = std::get_if<wire::Control>(&msg.m)) {
            // As a script would: pause/resume/step at once, set with the
            // clamped value, run queued and finished later (finish_run).
            controls_in.push_back(*c);
            wire::ControlReply r;
            r.request_id = c->request_id;
            r.op = c->op;
            r.name = c->name;
            if (c->op == CALIPER_NET_OP_RUN) {
                r.status = CALIPER_NET_STATUS_QUEUED;
                queued_runs.push_back(c->request_id);
            } else {
                r.status = CALIPER_NET_STATUS_OK;
                if (c->op == CALIPER_NET_OP_INTERPRET) { r.result = Json::object(); r.result.set("state", "idle"); }
                if (c->op == CALIPER_NET_OP_SET) r.value = Json(std::min(0.1, c->value.as_double()));
            }
            send(r);
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

public:
    // 3.1: the run state and one parameter, "lr".
    void send_controls(const char* state, double lr, int64_t step = -1) {
        wire::Controls c;
        c.run.set("state", state);
        if (step >= 0) c.run.set("step", step);
        Json param = Json::object();
        param.set("name", "lr");
        param.set("type", CALIPER_NET_PARAM_NUMBER);
        param.set("value", lr);
        param.set("min", 1e-05);
        param.set("max", 0.1);
        c.params.push(param);
        Json action = Json::object();
        action.set("name", "save");
        action.set("running", false);
        action.set("queued", !queued_runs.empty());
        c.actions.push(action);
        send(c);
    }
    void send_selection(const std::string& space, std::vector<uint64_t> items, const std::string& view = {}) {
        wire::Selection sel;
        sel.space = space;
        for (uint64_t i : items) sel.items.push(i);
        if (!view.empty()) sel.view = view;
        send(sel);
    }
    void finish_run(uint32_t request_id, const std::string& message) {
        wire::ControlReply r;
        r.request_id = request_id;
        r.op = CALIPER_NET_OP_RUN;
        r.status = CALIPER_NET_STATUS_OK;
        r.name = std::string("save");
        r.message = message;
        send(r);
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

// A barrier for the checks that something did not arrive: the peer pings the
// hub and waits for the PONG, which the hub sends after it has handled every
// message the peer sent before, behind everything it had queued for the peer.
void sync(World& w, Peer& peer) {
    static uint32_t nonce = 9000;
    const uint32_t n = ++nonce;
    peer.send(wire::Ping{n});
    REQUIRE(w.until([&] {
        for (auto* x : peer.all<wire::Pong>())
            if (x->nonce == n) return true;
        return false;
    }));
}

// A viewer connected and welcomed (3.1: with these capabilities).
std::unique_ptr<Peer> viewer(World& w, const HubRunner& h, std::vector<std::string> caps = {}) {
    auto v = std::make_unique<Peer>();
    REQUIRE(v->connect(h.url()));
    w.add(v.get());
    REQUIRE(w.until([&] { return v->open; }));
    v->hello_viewer(std::move(caps));
    REQUIRE(w.until([&] { return v->count<wire::Welcome>() == 1 && v->sessions(); }));
    return v;
}

std::unique_ptr<FakeProducer> producer(World& w, const HubRunner& h, const std::string& id, uint64_t pid,
                                       uint64_t start = 1763200000000ull, std::vector<std::string> caps = {},
                                       bool controls_on_welcome = false) {
    auto p = std::make_unique<FakeProducer>(id, pid, start);
    if (!caps.empty()) p->caps = std::move(caps);
    p->send_controls_on_welcome = controls_on_welcome;
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
    sync(w, *v2);
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
    sync(w, *v2);
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
    sync(w, *p);
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
    sync(w, *late);
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
    const fs::path dir = caliper_test::unique_temp_path("caliper-hub-assets");
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

TEST_CASE("net/hub: close diagnostics identify the endpoint without peer text") {
    bool local = false;
    SUBCASE("Peer Close") { local = false; }
    SUBCASE("Hub Idle Timeout") { local = true; }
    std::vector<std::string> logs;
    HubOptions o;
    o.ping_interval_ms = 100;
    o.idle_timeout_ms = 300;
    o.log = [&](const std::string& line) { logs.push_back(line); };
    HubRunner h(o);
    World w;
    auto v = viewer(w, h);
    if (local) v->answer_pings = false;
    else v->close(1000, "private-close-marker\nprivate-second-line");
    REQUIRE(w.until([&] { return v->closed; }));
    h.stop();   // The log callback ends before the test reads its output.
    const std::string expected = local ? "closed: role=viewer code=4004 local=true" :
                                         "closed: role=viewer code=1000 local=false";
    size_t matches = 0;
    for (const auto& line : logs) {
        CHECK(line.find("private-close-marker") == std::string::npos);
        CHECK(line.find("private-second-line") == std::string::npos);
        matches += line.find(expected) != std::string::npos;
    }
    CHECK(matches == 1);
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

    const fs::path dir = caliper_test::unique_temp_path("caliper-hub-file");
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

// ============================================================================
// Protocol 3.1 through the hub: controls and linked selection.
namespace {
const std::vector<std::string> kCaps31 = {CALIPER_NET_CAP_CONTROLS, CALIPER_NET_CAP_SELECTION};
// What the hub announces: 3.1 and the scenes of the 3.2 draft (tests/test_net_hub_32.cpp).
// The hub forwards controls, scenes and selections, and the 3.2 draft's programs and interventions.
const std::vector<std::string> kHubCaps = {CALIPER_NET_CAP_CONTROLS, CALIPER_NET_CAP_SCENES, CALIPER_NET_CAP_SELECTION,
                                           CALIPER_NET_CAP_PROGRAMS, CALIPER_NET_CAP_INTERVENTIONS,
                                           CALIPER_NET_CAP_INTERPRETATION, CALIPER_NET_CAP_TEXT_PARAMS,
                                           CALIPER_NET_CAP_RESOURCES, CALIPER_NET_CAP_SCENE_PIECES};
const std::vector<std::string> kProducer31 = {CALIPER_NET_CAP_EVENTS, CALIPER_NET_CAP_CONTROLS,
                                              CALIPER_NET_CAP_SELECTION};

bool has_items_key(const Json& views) {
    for (const Json& v : views.items()) {
        if (v.has("items")) return true;
        if (const Json* srcs = v.find("sources"))
            for (const Json& src : srcs->items())
                for (const Json& d : src.find("dims")->items())
                    if (d.has("items")) return true;
    }
    return false;
}
}  // namespace

TEST_CASE("net/hub 3.1: a late viewer gets the controls and selections right after VIEWS; a 3.0 viewer neither") {
    HubRunner h;
    World w;
    auto p = producer(w, h, "clp-31", 1, 1763200000000ull, kProducer31, true);
    // The hub announces both capabilities, and tells the producer it forwards no control.
    REQUIRE(p->welcome);
    CHECK(p->welcome->capabilities == kHubCaps);
    CHECK(p->welcome->control == false);
    p->send_selection("samples", {1, 5}, "heat");
    p->send_selection("step", {});   // an empty selection is kept, not sent to a late viewer
    REQUIRE(w.until([&] { return h.stats().retained_control_bytes > 0 && h.stats().controls_out == 0; }));
    sync(w, *p);   // both selections are kept before the viewer comes

    auto v31 = viewer(w, h, kCaps31);
    REQUIRE(w.until([&] { return v31->views() && v31->views()->session && v31->last<wire::Selection>(); }));
    const wire::Welcome* wel = v31->last<wire::Welcome>();
    CHECK(wel->capabilities == kHubCaps);
    CHECK(wel->control == false);
    // VIEWS (with the item spaces), then CONTROLS, then the one non-empty SELECTION.
    const int iv = v31->first_index(CALIPER_NET_MSG_VIEWS), ic = v31->first_index(CALIPER_NET_MSG_CONTROLS),
              is = v31->first_index(CALIPER_NET_MSG_SELECTION);
    CHECK(iv >= 0);
    CHECK(ic > iv);
    CHECK(is > ic);
    CHECK(has_items_key(v31->views()->views));
    CHECK(v31->last<wire::Controls>()->params.at(0).find("name")->as_string() == "lr");
    CHECK(v31->count<wire::Selection>() == 1);
    CHECK(v31->last<wire::Selection>()->space == "samples");

    // A 3.0 viewer: no CONTROLS, no SELECTION, no item spaces, no `control`.
    auto v30 = viewer(w, h);
    REQUIRE(w.until([&] { return v30->views() && v30->views()->session; }));
    CHECK_FALSE(v30->last<wire::Welcome>()->control.has_value());
    CHECK_FALSE(has_items_key(v30->views()->views));

    // A newer state reaches the 3.1 viewer only.
    p->send_controls("paused", 0.01);
    p->send_selection("samples", {2});
    REQUIRE(w.until([&] { return v31->count<wire::Controls>() == 2 && v31->count<wire::Selection>() == 2; }));
    CHECK(v31->last<wire::Controls>()->run.find("state")->as_string() == "paused");
    sync(w, *p);
    sync(w, *v30);
    CHECK(v30->count<wire::Controls>() == 0);
    CHECK(v30->count<wire::Selection>() == 0);
    CHECK(v30->count<wire::ControlReply>() == 0);
}

TEST_CASE("net/hub 3.1: without --allow-control every CONTROL is answered forbidden and none reaches the script") {
    HubRunner h;
    World w;
    auto p = producer(w, h, "clp-ro", 1, 1763200000000ull, kProducer31, true);
    auto v = viewer(w, h, kCaps31);
    REQUIRE(w.until([&] { return v->views() && v->views()->session; }));
    v->control(5, CALIPER_NET_OP_PAUSE);
    v->control(6, CALIPER_NET_OP_SET, "lr", Json(0.5));
    REQUIRE(w.until([&] { return v->control_replies(5).size() == 1 && v->control_replies(6).size() == 1; }));
    for (uint32_t id : {5u, 6u}) {
        const wire::ControlReply* r = v->control_replies(id)[0];
        CHECK(r->status == CALIPER_NET_STATUS_ERROR);
        CHECK(r->error.find("code")->as_string() == CALIPER_NET_ERR_FORBIDDEN);
    }
    CHECK(v->control_replies(6)[0]->name == std::string("lr"));
    sync(w, *p);
    CHECK(p->controls_in.empty());
    // Selecting is not a control: a select event still reaches the script.
    wire::Event e;
    e.view = "heat";
    e.type = CALIPER_NET_EVENT_SELECT;
    e.detail = obj({{"space", "samples"}, {"items", Json::array()}});
    v->send(e);
    REQUIRE(w.until([&] { return p->events.size() == 1; }));
    CHECK(p->events[0].detail.has("space"));
}

TEST_CASE("net/hub 3.1: with --allow-control a CONTROL goes to its session's script and the reply to its viewer") {
    HubOptions o;
    o.allow_control = true;
    HubRunner h(o);
    World w;
    auto pa = producer(w, h, "clp-a", 1, 1, kProducer31, true);
    auto pb = producer(w, h, "clp-b", 2, 2, kProducer31, true);
    CHECK(pa->welcome->control == true);
    auto va = viewer(w, h, kCaps31);
    auto vb = viewer(w, h, kCaps31);
    CHECK(va->last<wire::Welcome>()->control == true);
    // Not attached yet (two live sessions): the hub answers itself.
    va->control(1, CALIPER_NET_OP_PAUSE);
    REQUIRE(w.until([&] { return va->control_replies(1).size() == 1; }));
    CHECK(va->control_replies(1)[0]->error.find("code")->as_string() == CALIPER_NET_ERR_NOT_ATTACHED);
    va->attach("clp-a");
    vb->attach("clp-a");
    REQUIRE(w.until([&] { return va->views() && va->views()->session && vb->views() && vb->views()->session; }));

    // set: forwarded under the hub's own request id, answered with the clamped value.
    va->control(7, CALIPER_NET_OP_SET, "lr", Json(0.5));
    REQUIRE(w.until([&] { return va->control_replies(7).size() == 1; }));
    REQUIRE(pa->controls_in.size() == 1);
    CHECK(pa->controls_in[0].op == CALIPER_NET_OP_SET);
    CHECK(pa->controls_in[0].name == std::string("lr"));
    CHECK(pa->controls_in[0].value.as_double() == 0.5);
    CHECK(pb->controls_in.empty());   // only the script of the viewer's own session
    const wire::ControlReply* r = va->control_replies(7)[0];
    CHECK(r->status == CALIPER_NET_STATUS_OK);
    CHECK(r->value.as_double() == 0.1);

    // Two viewers with the same request id: distinct ids at the script, each
    // reply back to its own viewer, the queued run and its final answer too.
    va->control(9, CALIPER_NET_OP_RUN, "save");
    vb->control(9, CALIPER_NET_OP_RUN, "save");
    REQUIRE(w.until([&] { return pa->queued_runs.size() == 2; }));
    CHECK(pa->queued_runs[0] != pa->queued_runs[1]);
    REQUIRE(w.until([&] { return va->control_replies(9).size() == 1 && vb->control_replies(9).size() == 1; }));
    CHECK(va->control_replies(9)[0]->status == CALIPER_NET_STATUS_QUEUED);
    pa->finish_run(pa->queued_runs[0], "first");
    pa->finish_run(pa->queued_runs[1], "second");
    REQUIRE(w.until([&] { return va->control_replies(9).size() == 2 && vb->control_replies(9).size() == 2; }));
    const std::string ma = *va->control_replies(9)[1]->message, mb = *vb->control_replies(9)[1]->message;
    CHECK(ma != mb);
    CHECK((ma == "first" || ma == "second"));
    REQUIRE(w.until([&] { return h.stats().controls_waiting == 0; }));

    // A viewer that did not announce `controls` gets an ERROR, not a CONTROL_REPLY.
    auto plain = viewer(w, h);
    plain->attach("clp-a");
    REQUIRE(w.until([&] { return plain->views() && plain->views()->session; }));
    plain->control(3, CALIPER_NET_OP_PAUSE);
    REQUIRE(w.until([&] { return plain->last<wire::Error>(); }));
    CHECK(plain->last<wire::Error>()->in_reply_to == (uint32_t)CALIPER_NET_MSG_CONTROL);
    CHECK(plain->count<wire::ControlReply>() == 0);

    // A 3.0 script (no `controls`): the hub answers unsupported, and its
    // WELCOME carries no `control`.
    auto p30 = producer(w, h, "clp-30", 3, 3);
    CHECK_FALSE(p30->welcome->control.has_value());
    va->attach("clp-30");
    REQUIRE(w.until([&] { return va->views() && va->views()->session == std::string("clp-30"); }));
    va->control(11, CALIPER_NET_OP_PAUSE);
    REQUIRE(w.until([&] { return va->control_replies(11).size() == 1; }));
    CHECK(va->control_replies(11)[0]->error.find("code")->as_string() == CALIPER_NET_ERR_UNSUPPORTED);
    CHECK(p30->count<wire::Control>() == 0);
}

// A viewer connected with `url` and welcomed (3.1 capabilities).
std::unique_ptr<Peer> viewer_at(World& w, const std::string& url) {
    auto v = std::make_unique<Peer>();
    REQUIRE(v->connect(url));
    w.add(v.get());
    REQUIRE(w.until([&] { return v->open; }));
    v->hello_viewer(kCaps31);
    REQUIRE(w.until([&] { return v->count<wire::Welcome>() == 1 && v->sessions(); }));
    return v;
}

// A producer's HELLO on a connection opened with `url` whose token may only
// watch: refused as caliper.serve() refuses producers, ERROR `unsupported`
// and close 4002, and no session appears.
void publish_refused(World& w, const std::string& url, const std::function<caliper_net::HubStats()>& stats) {
    const size_t before = stats().sessions_live + stats().sessions_ended;
    FakeProducer p("clp-watcher", 9);
    REQUIRE(p.connect(url));
    w.add(&p);
    REQUIRE(w.until([&] { return p.closed; }));
    CHECK(p.close_code == CALIPER_NET_CLOSE_ROLE_REFUSED);
    REQUIRE(p.last<wire::Error>());
    CHECK(p.last<wire::Error>()->code == CALIPER_NET_ERR_UNSUPPORTED);
    CHECK(p.last<wire::Error>()->fatal);
    CHECK_FALSE(p.welcomed);
    CHECK(stats().sessions_live + stats().sessions_ended == before);
}

// The upgrade's HTTP status with `url`: 101 when it opened, else the refusal.
int upgrade_with(const std::string& url) {
    caliper_net::Client c;
    std::string why;
    if (!c.connect(url, &why)) return -1;
    const auto end = std::chrono::steady_clock::now() + std::chrono::seconds(10);
    while (std::chrono::steady_clock::now() < end) {
        for (auto& e : c.poll(10)) {
            if (e.type == caliper_net::Event::Type::Open) return 101;
            if (e.type == caliper_net::Event::Type::Closed) return e.http_status;
        }
    }
    return 0;
}

// A watching viewer and a controlling one on the same session: the same data
// and the same controls' state, and only the controlling one's CONTROL set
// reaches the script; the watcher's is answered forbidden, as without
// --allow-control. Shared by the hub's and Caliper API's cases.
void watch_and_control(World& w, FakeProducer& p, Peer& watcher, Peer& full) {
    CHECK(watcher.last<wire::Welcome>()->control == false);
    CHECK(full.last<wire::Welcome>()->control == true);
    REQUIRE(w.until([&] {
        return watcher.views() && watcher.views()->session && full.views() && full.views()->session &&
               watcher.last<wire::Controls>() && full.last<wire::Controls>();
    }));
    // Data flows to the watcher.
    watcher.subscribe(7, "heat", 70);
    REQUIRE(w.until([&] { return watcher.reply(70) && p.subs_of("heat").size() == 1; }));
    CHECK(p.snapshot(2.5f, 4, 8) >= 1);
    REQUIRE(w.until([&] { return watcher.snapshots(7).size() == 1; }));
    // Its CONTROL set is refused by the hub and never reaches the script.
    watcher.control(6, CALIPER_NET_OP_SET, "lr", Json(0.5));
    REQUIRE(w.until([&] { return watcher.control_replies(6).size() == 1; }));
    CHECK(watcher.control_replies(6)[0]->status == CALIPER_NET_STATUS_ERROR);
    CHECK(watcher.control_replies(6)[0]->error.find("code")->as_string() == CALIPER_NET_ERR_FORBIDDEN);
    sync(w, p);
    CHECK(p.controls_in.empty());
    // The controlling viewer's same CONTROL reaches the script, as today.
    full.control(7, CALIPER_NET_OP_SET, "lr", Json(0.5));
    REQUIRE(w.until([&] { return full.control_replies(7).size() == 1; }));
    REQUIRE(p.controls_in.size() == 1);
    CHECK(p.controls_in[0].name == std::string("lr"));
    CHECK(full.control_replies(7)[0]->status == CALIPER_NET_STATUS_OK);
    // The watcher still gets data after it was refused.
    CHECK(p.snapshot(3.5f, 4, 8) >= 1);
    REQUIRE(w.until([&] { return watcher.snapshots(7).size() == 2; }));
}

TEST_CASE("net/hub 3.1: with stream_access, a watching token sees the data and the controls but may not control") {
    const std::string watch = "hub_watch_token-0123456789abcdef";
    HubOptions o;
    o.allow_control = true;
    o.stream_access = [watch](const std::string& t) {
        if (wire::token_equal(t, kToken)) return HubOptions::StreamAccess::kFull;
        if (wire::token_equal(t, watch)) return HubOptions::StreamAccess::kWatch;
        return HubOptions::StreamAccess::kRefused;
    };
    HubRunner h(o);
    World w;
    auto p = producer(w, h, "clp-watch", 1, 1763200000000ull, kProducer31, true);
    auto watcher = viewer_at(w, h.url(watch));
    auto full = viewer_at(w, h.url());
    watch_and_control(w, *p, *watcher, *full);
    // Any other token is refused at the upgrade, as a wrong token always was.
    CHECK(upgrade_with(h.url("hub_other_token-0123456789abcdef")) == 403);
    // A watching token may not publish; the full one did, above.
    publish_refused(w, h.url(watch), [&] { return h.stats(); });
}

TEST_CASE("net/hub 3.1: Caliper API's stream: the person's token controls, the mcp and agent tokens watch, and the "
          "viewer, a revoked and a wrong token are refused, on both paths") {
    const std::string person = "api_person_token-0123456789abcdef";
    const std::string mcp = "api_mcp_token-0123456789abcdefghij";
    const std::string viewer_token = "api_viewer_token-0123456789abcdef";
    caliper_api::Options o;
    o.port = 0;
    o.token = person;
    o.mcp_token = mcp;
    o.viewer_token = viewer_token;
    o.allow_control = true;
    o.info_json = "{\"host\":\"test\"}";
    caliper_api::Server server(o);
    std::string err;
    REQUIRE_MESSAGE(server.start(&err), err);
    const std::string agent = server.issue_token("agent");
    const std::string revoked = server.issue_token("agent");
    server.revoke_token(revoked);
    for (const char* path : {"/v3/stream", "/v0/stream"}) {
        CAPTURE(path);
        const std::string base = "ws://127.0.0.1:" + std::to_string(server.port()) + path + "?token=";
        CHECK(upgrade_with(base + person) == 101);
        CHECK(upgrade_with(base + mcp) == 101);
        CHECK(upgrade_with(base + agent) == 101);
        CHECK(upgrade_with(base + revoked) == 403);
        CHECK(upgrade_with(base + viewer_token) == 403);
        CHECK(upgrade_with(base + "api_wrong_token-0123456789abcdefg") == 403);
        // The API file's token and an agent's may not publish a session.
        World w;
        publish_refused(w, base + mcp, [&] { return server.hub_stats(); });
        publish_refused(w, base + agent, [&] { return server.hub_stats(); });
    }
    // A run's script streams in with the person's token; an agent and the API
    // file's token watch it and may not control it, the person may.
    const std::string base = "ws://127.0.0.1:" + std::to_string(server.port()) + "/v3/stream?token=";
    for (const std::string& watching : {agent, mcp}) {
        CAPTURE(watching == agent ? "agent" : "mcp");
        World w;
        auto p = std::make_unique<FakeProducer>(watching == agent ? "clp-api-a" : "clp-api-m", 1, 1763200000000ull);
        p->caps = kProducer31;
        p->send_controls_on_welcome = true;
        REQUIRE(p->connect(base + person));
        w.add(p.get());
        REQUIRE(w.until([&] { return p->welcomed; }));
        auto watcher = viewer_at(w, base + watching);
        auto full = viewer_at(w, base + person);
        watcher->attach(p->chosen);
        full->attach(p->chosen);
        watch_and_control(w, *p, *watcher, *full);
    }
    server.stop();
}

TEST_CASE("net/hub 3.1: SELECTION reaches every viewer of the session; select events keep their items for a 3.1 script") {
    HubRunner h;
    World w;
    auto p = producer(w, h, "clp-sel", 1, 1, kProducer31);
    auto other = producer(w, h, "clp-other", 2, 2, kProducer31);
    auto a = viewer(w, h, kCaps31);
    auto b = viewer(w, h, kCaps31);
    auto c = viewer(w, h, kCaps31);
    a->attach("clp-sel");
    b->attach("clp-sel");
    c->attach("clp-other");
    REQUIRE(w.until([&] { return a->views() && b->views() && c->views() && c->views()->session; }));
    // The script's selection after a's select event goes to a and b, not c.
    wire::Event e;
    e.view = "heat";
    e.type = CALIPER_NET_EVENT_SELECT;
    Json items = Json::array();
    items.push(5);
    items.push(9);
    e.detail = obj({{"rows", Json::array()}, {"space", "samples"}, {"items", items}});
    a->send(e);
    REQUIRE(w.until([&] { return p->events.size() == 1; }));
    CHECK(p->events[0].detail.find("items")->size() == 2);
    p->send_selection("samples", {5, 9}, "heat");
    REQUIRE(w.until([&] { return a->count<wire::Selection>() == 1 && b->count<wire::Selection>() == 1; }));
    CHECK(b->last<wire::Selection>()->view == std::string("heat"));
    sync(w, *c);
    CHECK(c->count<wire::Selection>() == 0);

    // A script without `selection` gets the select event without its item keys.
    auto p30 = producer(w, h, "clp-30sel", 3, 3);
    a->attach("clp-30sel");
    REQUIRE(w.until([&] { return a->views() && a->views()->session == std::string("clp-30sel"); }));
    a->send(e);
    REQUIRE(w.until([&] { return p30->events.size() == 1; }));
    CHECK_FALSE(p30->events[0].detail.has("space"));
    CHECK_FALSE(p30->events[0].detail.has("items"));
    CHECK(p30->events[0].detail.has("rows"));

    // At most 16 item spaces per session are kept.
    for (int i = 0; i < 20; ++i) p->send_selection("space" + std::to_string(i), {1});
    REQUIRE(w.until([&] { return h.stats().selections_dropped == 5; }));
}

TEST_CASE("net/hub 3.1: an ended session keeps its controls visible and refuses controls; waiting ones are answered") {
    HubOptions o;
    o.allow_control = true;
    HubRunner h(o);
    World w;
    auto p = producer(w, h, "clp-end", 1, 1, kProducer31, true);
    auto v = viewer(w, h, kCaps31);
    REQUIRE(w.until([&] { return v->views() && v->views()->session && v->last<wire::Controls>(); }));
    // A run the script queued, then the script goes.
    v->control(4, CALIPER_NET_OP_RUN, "save");
    REQUIRE(w.until([&] { return v->control_replies(4).size() == 1; }));
    p->close(1000);
    REQUIRE(w.until([&] { return v->control_replies(4).size() == 2; }));
    CHECK(v->control_replies(4)[1]->error.find("code")->as_string() == CALIPER_NET_ERR_UNSUPPORTED);
    REQUIRE(w.until([&] { return find_session(*v, "clp-end") && find_session(*v, "clp-end")->state == "ended"; }));
    // A late viewer of the ended session still sees its last controls.
    auto late = viewer(w, h, kCaps31);
    late->attach("clp-end");
    REQUIRE(w.until([&] { return late->views() && late->views()->session && late->last<wire::Controls>(); }));
    CHECK(late->last<wire::Controls>()->params.at(0).find("name")->as_string() == "lr");
    late->control(8, CALIPER_NET_OP_RESUME);
    REQUIRE(w.until([&] { return late->control_replies(8).size() == 1; }));
    CHECK(late->control_replies(8)[0]->error.find("code")->as_string() == CALIPER_NET_ERR_UNSUPPORTED);
    CHECK(h.stats().controls_waiting == 0);
}

TEST_CASE("net/hub 3.1: controls waiting for a script are bounded per viewer") {
    HubOptions o;
    o.allow_control = true;
    HubRunner h(o);
    World w;
    auto p = producer(w, h, "clp-mute", 1, 1, kProducer31, true);
    auto v = viewer(w, h, kCaps31);
    REQUIRE(w.until([&] { return v->views() && v->views()->session; }));
    // The fake script answers each run `queued` and never finishes it, so
    // every route stays open.
    for (uint32_t i = 1; i <= 64; ++i) v->control(1000 + i, CALIPER_NET_OP_RUN, "save");
    REQUIRE(w.until([&] { return v->count<wire::ControlReply>() == 64; }));
    v->control(2000, CALIPER_NET_OP_PAUSE);
    REQUIRE(w.until([&] { return v->control_replies(2000).size() == 1; }));
    CHECK(v->control_replies(2000)[0]->error.find("code")->as_string() == CALIPER_NET_ERR_LIMIT);
    CHECK(h.stats().controls_waiting == 64);
}

// ---- 3.1: a run state after the data before it -------------------------------------------
// A hub delivers a run state after the data it received before it, or after
// 2 s, so that a paused view shows the step the run stopped at.
namespace {
int64_t step_of(const wire::Controls& c) {
    uint64_t n = 0;
    const Json* s = c.run.find("step");
    return s && s->get_u64(&n) ? (int64_t)n : -1;
}

// The position in v's inbox of the first CONTROLS with this state and step, -1 when none.
int controls_index(const Peer& v, const std::string& state, int64_t step) {
    for (size_t i = 0; i < v.inbox.size(); ++i)
        if (auto* c = std::get_if<wire::Controls>(&v.inbox[i].m))
            if (c->run.find("state")->as_string() == state && step_of(*c) == step) return (int)i;
    return -1;
}

int snapshot_index(const Peer& v, uint32_t sub, uint64_t seq) {
    for (size_t i = 0; i < v.inbox.size(); ++i)
        if (auto* s = std::get_if<wire::Snapshot>(&v.inbox[i].m))
            if (s->sub == sub && s->seq == seq) return (int)i;
    return -1;
}

// One past the highest series index that the first n messages of v's inbox carry on sub.
uint64_t series_end_before(const Peer& v, uint32_t sub, int n) {
    uint64_t end = 0;
    for (int i = 0; i < n && i < (int)v.inbox.size(); ++i)
        if (auto* s = std::get_if<wire::Series>(&v.inbox[(size_t)i].m))
            if (s->sub == sub) end = std::max<uint64_t>(end, s->start + s->values.size / 4);
    return end;
}

// Polls the producer all the time and the viewer only every few milliseconds,
// so that the hub holds data in the viewer's send queue and pending for it.
bool drain_slowly(Peer& p, Peer& v, const std::function<bool()>& done, int ms = 10000) {
    const auto end = std::chrono::steady_clock::now() + std::chrono::milliseconds(ms);
    for (int i = 0; std::chrono::steady_clock::now() < end; ++i) {
        p.poll(0);
        if (i % 6 == 0) v.poll(0);
        if (done()) return true;
        std::this_thread::sleep_for(std::chrono::microseconds(500));
    }
    return done();
}

double secs_between(uint64_t a_ns, uint64_t b_ns) { return (double)(int64_t)(b_ns - a_ns) / 1e9; }

std::vector<float> filled(size_t n, float x) { return std::vector<float>(n, x); }
}  // namespace

TEST_CASE("net/hub 3.1: a slowly draining viewer gets each paused step's data before the run state, every time") {
    HubOptions o;
    o.allow_control = true;
    HubRunner h(o);
    World w;
    auto p = producer(w, h, "clp-order", 1, 1, kProducer31, true);
    auto v = viewer(w, h, kCaps31);
    REQUIRE(w.until([&] { return v->views() && v->views()->session && v->last<wire::Controls>(); }));
    v->subscribe(1, "heat", 1, Json(), 240);
    v->subscribe(2, "loss", 2, Json(), 240);
    REQUIRE(w.until([&] { return v->reply(1) && v->reply(2) && p->subs_of("heat").size() == 1; }));

    // A run paused step by step: each step sends series values and two
    // snapshots, the second its last, then CONTROLS paused at that step. The
    // viewer drains slowly, so the step's data is still pending or in the
    // viewer's send queue when the CONTROLS reaches the hub.
    const int N = 100;
    int in_order = 0;
    for (int k = 1; k <= N; ++k) {
        p->send_controls("stepping", 0.01, k - 1);
        p->append(filled(3000, (float)k));
        p->snapshot((float)k, 256, 256);
        p->snapshot((float)k + 0.5f, 256, 256);
        const uint64_t seq = p->seq, total = p->total();
        p->send_controls("paused", 0.01, k);
        REQUIRE(drain_slowly(*p, *v, [&] { return controls_index(*v, "paused", k) >= 0; }));
        const int ic = controls_index(*v, "paused", k);
        const int is = snapshot_index(*v, 1, seq);
        const bool ok = is >= 0 && is < ic && series_end_before(*v, 2, ic) >= total;
        if (!ok) MESSAGE("step " << k << ": CONTROLS at " << ic << ", its snapshot at " << is << ", series to "
                                 << series_end_before(*v, 2, ic) << " of " << total);
        in_order += ok ? 1 : 0;
        v->inbox.clear();
    }
    CHECK(in_order == N);
    CHECK(h.stats().controls_hold_expired == 0);
    MESSAGE(h.stats().controls_held << " of the run states waited for their data");
}

TEST_CASE("net/hub 3.1: a run state waits at most 2 s for data the viewer's max_rate holds back") {
    HubRunner h;
    World w;
    auto p = producer(w, h, "clp-rate", 1, 1, kProducer31, true);
    auto v = viewer(w, h, kCaps31);
    REQUIRE(w.until([&] { return v->views() && v->views()->session && v->last<wire::Controls>(); }));
    v->subscribe(1, "heat", 1, Json(), 0.2);   // one snapshot per 5 s, after a burst of two
    REQUIRE(w.until([&] { return v->reply(1) && p->subs_of("heat").size() == 1; }));
    for (int i = 1; i <= 2; ++i) {
        p->snapshot((float)i);
        REQUIRE(w.until([&] { return v->snapshots(1).size() == (size_t)i; }));
    }
    p->snapshot(3);   // pending for about 5 s; the CONTROLS follows it on the same connection
    const uint64_t sent = mono_ns();
    p->send_controls("paused", 0.01, 3);
    REQUIRE(w.until([&] { return controls_index(*v, "paused", 3) >= 0; }, 5000));
    const double waited = secs_between(sent, v->inbox[(size_t)controls_index(*v, "paused", 3)].at_ns);
    MESSAGE("the run state waited " << waited << " s");
    CHECK(waited > 1.8);
    CHECK(waited < 4.5);   // the 2 s bound, not the snapshot's release at about 5 s
    CHECK(v->snapshots(1).size() == 2);   // it went without the snapshot before it
    CHECK(h.stats().controls_held == 1);
    CHECK(h.stats().controls_hold_expired == 1);
}

TEST_CASE("net/hub 3.1: a viewer that stops reading gets the run state after 2 s, ahead of the data it waited for") {
    HubRunner h;
    World w;
    auto p = producer(w, h, "clp-stuck", 1, 1, kProducer31, true);
    auto v = viewer(w, h, kCaps31);
    REQUIRE(w.until([&] { return v->views() && v->views()->session && v->last<wire::Controls>(); }));
    v->subscribe(1, "heat", 1, Json(), 240);
    REQUIRE(w.until([&] { return v->reply(1) && p->subs_of("heat").size() == 1; }));

    // The viewer reads nothing more: 1 MiB snapshots fill its socket until the
    // hub replaces the one pending for it.
    const auto limit = std::chrono::steady_clock::now() + std::chrono::seconds(10);
    for (int i = 0; h.stats().snapshots_superseded < 3 && std::chrono::steady_clock::now() < limit; ++i) {
        p->snapshot((float)i, 512, 512);
        p->poll(0);
        std::this_thread::sleep_for(std::chrono::milliseconds(2));
    }
    REQUIRE(h.stats().snapshots_superseded >= 3);
    const uint64_t sent = mono_ns();
    p->send_controls("paused", 0.01, 1);
    const auto until = std::chrono::steady_clock::now() + std::chrono::seconds(5);
    while (h.stats().controls_hold_expired == 0 && std::chrono::steady_clock::now() < until) {
        p->poll(0);
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    const double waited = secs_between(sent, mono_ns());
    MESSAGE("the hub let the run state go after " << waited << " s");
    CHECK(h.stats().controls_held == 1);
    REQUIRE(h.stats().controls_hold_expired == 1);
    CHECK(waited > 1.9);
    CHECK(waited < 4.5);
    // When it reads again it gets the run state.
    REQUIRE(w.until([&] { return controls_index(*v, "paused", 1) >= 0; }));
}

TEST_CASE("net/hub 3.1: parameter echoes are not held back while data streams") {
    HubRunner h;
    World w;
    auto p = producer(w, h, "clp-echo", 1, 1, kProducer31, true);
    auto v = viewer(w, h, kCaps31);
    REQUIRE(w.until([&] { return v->views() && v->views()->session && v->last<wire::Controls>(); }));
    // The viewer takes 20 messages a second per subscription while the
    // producer sends 200: the hub always has a snapshot and series values
    // pending for it, and is never drained.
    v->subscribe(1, "heat", 1, Json(), 20);
    v->subscribe(2, "loss", 2, Json(), 20);
    REQUIRE(w.until([&] { return v->reply(1) && v->reply(2) && p->subs_of("heat").size() == 1; }));
    std::map<int, uint64_t> sent;   // echo number -> when the producer sent it
    const auto start = std::chrono::steady_clock::now();
    auto next_data = start, next_echo = start + std::chrono::milliseconds(50);
    int echoes = 0;
    while (std::chrono::steady_clock::now() < start + std::chrono::milliseconds(1600)) {
        const auto now = std::chrono::steady_clock::now();
        if (now >= next_data) {
            p->snapshot((float)p->seq, 128, 128);
            p->append(filled(16, 1.0f));
            next_data += std::chrono::milliseconds(5);
        }
        if (now >= next_echo && echoes < 15) {
            ++echoes;
            sent[echoes] = mono_ns();
            p->send_controls("running", 0.001 * echoes, 1000 + echoes);   // as a set's answer
            next_echo += std::chrono::milliseconds(100);
        }
        p->poll(0);
        v->poll(0);
        std::this_thread::sleep_for(std::chrono::microseconds(300));
    }
    REQUIRE(w.until([&] { return controls_index(*v, "running", 1000 + 15) >= 0; }));
    double worst = 0;
    int got = 0;
    for (int k = 1; k <= 15; ++k) {
        const int i = controls_index(*v, "running", 1000 + k);
        if (i < 0) continue;
        worst = std::max(worst, secs_between(sent[k], v->inbox[(size_t)i].at_ns));
        ++got;
    }
    MESSAGE(got << " echoes, the slowest after " << worst * 1000 << " ms; " << h.stats().controls_held
                << " waited for data");
    CHECK(got == 15);
    CHECK(worst < 1.5);   // well under the 2 s bound; no echo waited for it:
    CHECK(h.stats().controls_held > 0);   // they did wait for the data before them, briefly
    CHECK(h.stats().controls_hold_expired == 0);
}

TEST_CASE("net/hub 3.1: a single step shows the step's data before the run state that says paused") {
    HubOptions o;
    o.allow_control = true;
    HubRunner h(o);
    World w;
    auto p = producer(w, h, "clp-step", 1, 1, kProducer31, true);
    auto v = viewer(w, h, kCaps31);
    REQUIRE(w.until([&] { return v->views() && v->views()->session && v->last<wire::Controls>(); }));
    v->subscribe(1, "heat", 1, Json(), 240);
    v->subscribe(2, "loss", 2, Json(), 240);
    REQUIRE(w.until([&] { return v->reply(1) && v->reply(2) && p->subs_of("heat").size() == 1; }));
    // Paused at step 4.
    p->append(filled(4, 4.0f));
    p->snapshot(4, 512, 512);
    p->send_controls("paused", 0.01, 4);
    REQUIRE(drain_slowly(*p, *v, [&] { return controls_index(*v, "paused", 4) >= 0; }));
    v->inbox.clear();

    // The viewer steps once; the script answers, says stepping, runs step 5
    // and pauses at it.
    v->control(21, CALIPER_NET_OP_STEP);
    REQUIRE(w.until([&] { return v->control_replies(21).size() == 1 && p->controls_in.size() == 1; }));
    CHECK(p->controls_in[0].op == CALIPER_NET_OP_STEP);
    p->send_controls("stepping", 0.01, 4);
    p->append(filled(30000, 5.0f));
    p->snapshot(5, 512, 512);
    const uint64_t seq = p->seq, total = p->total();
    p->send_controls("paused", 0.01, 5);
    REQUIRE(drain_slowly(*p, *v, [&] { return controls_index(*v, "paused", 5) >= 0; }));
    const int stepping = controls_index(*v, "stepping", 4), snap = snapshot_index(*v, 1, seq),
              paused = controls_index(*v, "paused", 5);
    CHECK(stepping >= 0);
    CHECK(snap > stepping);
    CHECK(paused > snap);
    CHECK(series_end_before(*v, 2, paused) == total);
    CHECK(v->snapshots(1).back()->seq == seq);
}

TEST_CASE("net/hub 3.1: a late viewer of a paused run gets the run state right after VIEWS and the step's data after "
          "it subscribes") {
    HubRunner h;
    World w;
    auto p = producer(w, h, "clp-late", 1, 1, kProducer31, true);
    auto early = viewer(w, h, kCaps31);
    REQUIRE(w.until([&] { return early->views() && early->views()->session; }));
    early->subscribe(1, "heat", 1, Json(), 240);
    early->subscribe(2, "loss", 2, Json(), 240);
    REQUIRE(w.until([&] { return early->reply(1) && early->reply(2) && p->subs_of("heat").size() == 1; }));
    for (int k = 1; k <= 9; ++k) {
        p->append(filled(100, (float)k));
        p->snapshot((float)k, 256, 256);
    }
    const uint64_t seq = p->seq, total = p->total();
    p->send_controls("paused", 0.01, 9);
    REQUIRE(w.until([&] { return controls_index(*early, "paused", 9) >= 0; }));
    CHECK(snapshot_index(*early, 1, seq) < controls_index(*early, "paused", 9));
    const uint64_t held = h.stats().controls_held;

    auto late = viewer(w, h, kCaps31);
    REQUIRE(w.until([&] { return late->views() && late->views()->session && late->last<wire::Controls>(); }));
    const int iv = late->first_index(CALIPER_NET_MSG_VIEWS), ic = controls_index(*late, "paused", 9);
    CHECK(iv >= 0);
    CHECK(ic == iv + 1);   // the kept run state, right after VIEWS, before any SUBSCRIBE
    late->subscribe(1, "heat", 1, Json(), 240);
    late->subscribe(2, "loss", 2, Json(), 240);
    REQUIRE(w.until([&] { return !late->snapshots(1).empty() && late->series(2).size() == total; }));
    CHECK(late->snapshots(1).front()->seq == seq);   // the paused step's snapshot
    CHECK(snapshot_index(*late, 1, seq) > ic);
    CHECK(late->series(2).rbegin()->first == total - 1);
    CHECK(h.stats().controls_held == held);   // nothing was pending for it: nothing waited
    CHECK(late->count<wire::Controls>() == 1);
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

// ============================================================================
// The Caliper app as a 3.1 viewer through the hub: the real app, its window
// never shown (--hidden), attached at start with --url, its live page's run
// bar clicked and a view shift-dragged by scripted ImGui input (CALIPER_TEST_UI,
// live/live_ui_driver.h). The fake script answers like one that calls a
// checkpoint. With --allow-control the clicks pause it, set "lr" and run
// "save", and the page logs the state the script echoes; without it the bar
// is off and nothing reaches the script, while a selection still does.
#if defined(__APPLE__)
#include "test_proc.h"

namespace {

namespace fs = std::filesystem;

struct AppRun {
    bool ran = false;                 // false: no GPU, skipped
    std::string text;                 // the app's output
    std::vector<wire::Control> controls;
    std::vector<wire::Event> events;
    int64_t bar_running_at = -1;      // the fake script's step when the page's bar first showed "running..."
    int64_t paused_at = -1;           // the step at which a pause or step stopped it
};

AppRun run_hidden_app(const char* exe, bool allow_control, const std::string& ui) {
    AppRun out;
    HubOptions o;
    o.allow_control = allow_control;
    HubRunner h(o);
    World w;
    auto p = producer(w, h, "clp-app", 4242, 1763200000000ull,
                      {CALIPER_NET_CAP_EVENTS, CALIPER_NET_CAP_CONTROLS, CALIPER_NET_CAP_SELECTION}, true);
    // As caliper.step(step) does: the first checkpoint says running at step 0,
    // and no CONTROLS follows while the loop runs on (the step in CONTROLS is
    // that of the last change of state).
    int64_t step = 0, state_step = 0;
    p->send_controls(CALIPER_NET_RUN_RUNNING, 0.001, state_step);
    const fs::path dir = fs::temp_directory_path() / (std::string("caliper-app-live-") + (allow_control ? "a" : "b"));
    fs::remove_all(dir);
    fs::create_directories(dir / "data");
    fs::create_directories(dir / "user");
    Proc app;
    REQUIRE(app.start(exe, {"--hidden", "--url", h.url(), "--api", "--api-port", "0"},
                      {"CALIPER_DATA_DIR=" + (dir / "data").string(), "CALIPER_USER_INSTRUMENTS_DIR=" + (dir / "user").string(),
                       "CALIPER_WINDOW_SIZE=1280x800", "CALIPER_TEST_UI=" + ui}));
    std::string state = CALIPER_NET_RUN_RUNNING;
    double lr = 0.001;
    size_t controls_seen = 0, events_seen = 0;
    float fill = 0.0f;
    auto last_snap = std::chrono::steady_clock::now();
    const auto deadline = last_snap + std::chrono::seconds(90);
    while (std::chrono::steady_clock::now() < deadline) {
        w.once();
        app.read_some(2);
        if (app.text.find("[ui] done") != std::string::npos || app.text.find("[ui] stopped") != std::string::npos)
            break;
        // A renderer other than Metal (its whole line read): the run is skipped.
        if (const size_t r = app.text.find("[renderer] "); r != std::string::npos &&
            app.text.find('\n', r) != std::string::npos && app.text.find("[renderer] metal") == std::string::npos)
            break;
        if (::waitpid(app.pid, &app.status, WNOHANG) == app.pid) {
            app.pid = -1;
            break;
        }
        if (std::chrono::steady_clock::now() - last_snap > std::chrono::milliseconds(50)) {
            p->snapshot(fill += 0.01f);
            p->append({fill});
            last_snap = std::chrono::steady_clock::now();
            if (state == CALIPER_NET_RUN_RUNNING) ++step;
        }
        if (out.bar_running_at < 0 && app.text.find("[ui] bar shows 'running") != std::string::npos)
            out.bar_running_at = step;
        // As a script at its checkpoint: apply what was accepted, say so in CONTROLS.
        for (; controls_seen < p->controls_in.size(); ++controls_seen) {
            const wire::Control& c = p->controls_in[controls_seen];
            if (c.op == CALIPER_NET_OP_PAUSE || c.op == CALIPER_NET_OP_STEP) {
                state = CALIPER_NET_RUN_PAUSED;
                state_step = out.paused_at = step;
            } else if (c.op == CALIPER_NET_OP_RESUME) {
                state = CALIPER_NET_RUN_RUNNING;
                state_step = step;
            } else if (c.op == CALIPER_NET_OP_SET) {
                lr = std::min(0.1, c.value.as_double());
            } else if (c.op == CALIPER_NET_OP_RUN) {
                p->queued_runs.clear();
                p->finish_run(c.request_id, "saved at step 7");
            }
            p->send_controls(state.c_str(), lr, state_step);
        }
        // A selection made in a view: every viewer gets it, with details.
        for (; events_seen < p->events.size(); ++events_seen) {
            const wire::Event& e = p->events[events_seen];
            const Json* space = e.detail.find("space");
            if (e.type != "select" || !space) continue;
            wire::Selection sel;
            sel.space = space->as_string();
            sel.view = e.view;
            Json ids = Json::array(), rows = Json::array();
            for (const Json& x : e.detail.find("items")->items()) {
                sel.items.push(x);
                if (ids.size() < CALIPER_NET_MAX_DETAIL_ITEMS) {
                    ids.push(x);
                    Json row = Json::array();
                    row.push(x.as_double() * 0.25);
                    rows.push(row);
                }
            }
            Json fields = Json::array();
            fields.push("score");
            sel.details = obj({{"fields", fields}, {"items", ids}, {"values", rows}});
            p->send(sel);
        }
    }
    out.ran = app.text.find("[renderer] metal") != std::string::npos;
    if (app.pid > 0) {
        ::kill(app.pid, SIGTERM);
        const int code = app.wait(20000);
        CHECK_MESSAGE(code == 0, "exit " << code << ", signal " << (WIFSIGNALED(app.status) ? WTERMSIG(app.status) : 0)
                                         << "\n" << app.text);
    }
    out.text = app.text;
    out.controls = p->controls_in;
    out.events = p->events;
    fs::remove_all(dir);
    return out;
}

bool has(const std::string& text, const std::string& needle) { return text.find(needle) != std::string::npos; }

}  // namespace

TEST_CASE("app: the hidden app's run bar controls a script through a hub with --allow-control; without it only "
          "selections go") {
    const char* exe = std::getenv("CALIPER_TEST_CALIPER_EXE");
    if (!exe || !*exe) {
        MESSAGE("CALIPER_TEST_CALIPER_EXE is not set: run the ctest entry caliper_app_live_controls");
        return;
    }
    // The bar while the fake script runs on past the step 0 of its only
    // CONTROLS, and once it is paused.
    AppRun a = run_hidden_app(exe, true,
                              "state running; wait 30; bar running; click pause; state paused; bar paused; "
                              "set lr 0.02; wait 10; click action:save; wait 30; select heat; wait 30");
    if (!a.ran) {
        const char* require = std::getenv("CALIPER_REQUIRE_GPU");
        REQUIRE_MESSAGE(!(require && std::string(require) == "1"), a.text);
        MESSAGE("caliper --hidden --url: run skipped: no GPU (no Metal device)");
        return;
    }
    INFO(a.text);
    CHECK(has(a.text, "[hidden] the window is not shown (visible 0, focused 0, background policy 1"));
    CHECK(has(a.text, "[ui] done"));
    REQUIRE(a.controls.size() == 3);
    CHECK(a.controls[0].op == "pause");
    CHECK(a.controls[1].op == "set");
    CHECK(a.controls[1].value.as_double() == doctest::Approx(0.02));
    CHECK(a.controls[2].op == "run");
    CHECK(*a.controls[2].name == "save");
    // The bar says running, without the step of the state's change, while the
    // script is past it, and names the step at which the run is paused.
    CHECK(has(a.text, "[ui] bar shows 'running'\n"));
    CHECK(a.bar_running_at > 0);
    REQUIRE(a.paused_at > 0);
    CHECK(has(a.text, "[ui] bar shows 'paused at step " + std::to_string(a.paused_at) + "'\n"));
    // The page shows what the script echoed.
    CHECK(has(a.text, "[live] controls on"));
    CHECK(has(a.text, "[live] run paused"));
    CHECK(has(a.text, "[live] param lr = 0.02"));
    CHECK(has(a.text, "[live] save: saved at step 7"));
    CHECK(has(a.text, "[live] selection samples: "));
    CHECK(has(a.text, "with details"));
    const wire::Event* sel = nullptr;
    for (const auto& e : a.events)
        if (e.type == "select" && e.view == "heat") sel = &e;
    REQUIRE(sel != nullptr);
    CHECK(sel->detail.find("space")->as_string() == "samples");
    CHECK(sel->detail.find("items")->size() >= 1);

    AppRun b = run_hidden_app(exe, false,
                              "state running; click pause; wait 20; click step; wait 20; click action:save; "
                              "wait 20; select heat; wait 30");
    INFO(b.text);
    REQUIRE(b.ran);
    CHECK(has(b.text, "[ui] done"));
    CHECK(b.controls.empty());
    CHECK(has(b.text, "[live] controls off: the server does not let this viewer control"));
    CHECK_FALSE(has(b.text, "[live] run paused"));
    CHECK(has(b.text, "[live] selection samples: "));
    bool selected = false;
    for (const auto& e : b.events) selected = selected || (e.type == "select" && e.detail.find("space"));
    CHECK(selected);
}
#endif

TEST_CASE("net/hub: text parameters follow viewer capabilities on attach and update") {
    HubRunner hub;
    World world;
    auto caps = kProducer31;
    caps.push_back(CALIPER_NET_CAP_TEXT_PARAMS);
    auto producer_link = producer(world, hub, "text-params", 1, 1, caps);
    wire::Controls controls;
    controls.run = obj({{"state", "running"}, {"step", 1}});
    controls.params.push(obj({{"name", "clamp"}, {"type", "number"}, {"value", 0.},
                              {"min", -4.}, {"max", 4.}}));
    controls.params.push(obj({{"name", "prompt"}, {"type", "text"}, {"value", "AB"}}));
    controls.params.push(obj({{"name", "retrace"}, {"type", "boolean"}, {"value", false}}));
    controls.actions.push(obj({{"name", "apply_clamp"}, {"running", false}, {"queued", false}}));
    const int64_t stamp[] = {17, 23};
    controls.stamp = wire::Section::make("stamp", CALIPER_NET_DTYPE_I64, {2}, stamp, sizeof(stamp));
    producer_link->send(controls);
    sync(world, *producer_link);

    auto legacy = viewer(world, hub, kCaps31);
    auto text_reader = viewer(world, hub, caps);
    REQUIRE(world.until([&] { return legacy->last<wire::Controls>() && text_reader->last<wire::Controls>(); }));
    const auto check_controls = [&](const wire::Controls& expected) {
        const auto* plain = legacy->last<wire::Controls>();
        const auto* full = text_reader->last<wire::Controls>();
        REQUIRE(plain);
        REQUIRE(full);
        REQUIRE(plain->params.size() == 2);
        CHECK(plain->params.at(0).dump() == expected.params.at(0).dump());
        CHECK(plain->params.at(1).dump() == expected.params.at(2).dump());
        CHECK(full->params.dump() == expected.params.dump());
        CHECK(plain->run.dump() == expected.run.dump());
        CHECK(full->run.dump() == expected.run.dump());
        CHECK(plain->actions.dump() == expected.actions.dump());
        CHECK(full->actions.dump() == expected.actions.dump());
        for (const auto* value : {plain, full}) {
            REQUIRE(value->stamp);
            CHECK(value->stamp->dtype == expected.stamp->dtype);
            CHECK(value->stamp->shape == expected.stamp->shape);
            REQUIRE(value->stamp->size == expected.stamp->size);
            CHECK(std::memcmp(value->stamp->data, expected.stamp->data, expected.stamp->size) == 0);
        }
    };
    CHECK(legacy->first_index(CALIPER_NET_MSG_CONTROLS) > legacy->first_index(CALIPER_NET_MSG_VIEWS));
    CHECK(text_reader->first_index(CALIPER_NET_MSG_CONTROLS) > text_reader->first_index(CALIPER_NET_MSG_VIEWS));
    check_controls(controls);

    controls.run = obj({{"state", "paused"}, {"step", 2}});
    controls.params = Json::array();
    controls.params.push(obj({{"name", "clamp"}, {"type", "number"}, {"value", 2.},
                              {"min", -4.}, {"max", 4.}}));
    controls.params.push(obj({{"name", "prompt"}, {"type", "text"}, {"value", "AC"}}));
    controls.params.push(obj({{"name", "retrace"}, {"type", "boolean"}, {"value", true}}));
    producer_link->send(controls);
    REQUIRE(world.until([&] {
        return legacy->count<wire::Controls>() == 2 && text_reader->count<wire::Controls>() == 2;
    }));
    check_controls(controls);
}

TEST_CASE("net/hub: text sets require both peer capabilities and preserve legacy choices") {
    HubOptions options;
    options.allow_control = true;
    HubRunner hub(options);
    World world;
    auto caps = kProducer31;
    bool text = false;
    bool viewer_text = true;
    SUBCASE("legacy producer") {}
    SUBCASE("text producer") { caps.push_back(CALIPER_NET_CAP_TEXT_PARAMS); text = true; }
    SUBCASE("legacy viewer") { caps.push_back(CALIPER_NET_CAP_TEXT_PARAMS); viewer_text = false; }
    auto source = producer(world, hub, "text-admission", 1, 1, caps, true);
    auto viewer_caps = kCaps31;
    if (viewer_text) viewer_caps.push_back(CALIPER_NET_CAP_TEXT_PARAMS);
    auto client = viewer(world, hub, viewer_caps);
    REQUIRE(world.until([&] { return client->views() && client->views()->session; }));
    uint32_t id = 0;
    size_t forwarded = 0;
    for (const auto& value : {std::string("mode"), std::string("line\n\tbreak"), std::string(63, 'x'), std::string(),
                              std::string(64, 'x'), std::string(4096, 'x')}) {
        client->control(++id, "set", "mode", Json(value));
        REQUIRE(world.until([&] { return client->control_replies(id).size() == 1; }));
        const bool admitted = text || (!value.empty() && value.size() <= 63);
        if (admitted) {
            ++forwarded;
            REQUIRE(source->controls_in.size() == forwarded);
            CHECK(source->controls_in.back().value.as_string() == value);
            CHECK(client->control_replies(id)[0]->status == "ok");
        } else {
            CHECK(source->controls_in.size() == forwarded);
            CHECK(client->control_replies(id)[0]->error.find("code")->as_string() == CALIPER_NET_ERR_UNSUPPORTED);
        }
    }
    client->control(++id, "set", "lr", Json(0.01));
    REQUIRE(world.until([&] { return client->control_replies(id).size() == 1; }));
    CHECK(source->controls_in.size() == forwarded + 1);
    CHECK(client->control_replies(id)[0]->status == "ok");

    wire::Controls controls;
    controls.run = obj({{"state", "running"}, {"step", 1}});
    controls.params.push(obj({{"name", "prompt"}, {"type", "text"}, {"value", "initial"}}));
    source->send(controls);
    sync(world, *source);
    client->control(++id, "set", "prompt", Json("short"));
    REQUIRE(world.until([&] { return client->control_replies(id).size() == 1; }));
    CHECK(client->control_replies(id)[0]->status == (text ? "ok" : "error"));
    CHECK(source->controls_in.size() == forwarded + 1 + (text ? 1 : 0));

    controls.params = Json::array();
    Json choices = Json::array();
    choices.push("initial");
    choices.push("short");
    controls.params.push(obj({{"name", "prompt"}, {"type", "choice"}, {"value", "initial"},
                              {"choices", choices}}));
    source->send(controls);
    sync(world, *source);
    client->control(++id, "set", "prompt", Json("short"));
    REQUIRE(world.until([&] { return client->control_replies(id).size() == 1; }));
    CHECK(client->control_replies(id)[0]->status == "ok");
}

TEST_CASE("net/hub: interpretation reads preserve capability gates and write permissions") {
    HubRunner hub;
    World world;
    auto caps = kProducer31;
    caps.push_back(CALIPER_NET_CAP_INTERPRETATION);
    auto producer_link = producer(world, hub, "interpretation", 1, 1, caps, true);
    auto reader = viewer(world, hub, caps);
    const auto* welcome = reader->last<wire::Welcome>();
    REQUIRE(welcome);
    CHECK(std::find(welcome->capabilities.begin(), welcome->capabilities.end(),
                    CALIPER_NET_CAP_INTERPRETATION) != welcome->capabilities.end());
    CHECK(std::find(welcome->capabilities.begin(), welcome->capabilities.end(),
                    CALIPER_NET_CAP_TEXT_PARAMS) != welcome->capabilities.end());
    REQUIRE(world.until([&] { return reader->views() && reader->views()->session; }));
    wire::Control request;
    request.request_id = 1;
    request.op = CALIPER_NET_OP_INTERPRET;
    request.interpretation = obj({{"version",1},{"method","trace.status"},{"params",Json::object()}});
    reader->send(request);
    REQUIRE(world.until([&] { return reader->control_replies(1).size() == 1; }));
    CHECK(reader->control_replies(1)[0]->result.find("state")->as_string() == "idle");
    request.request_id = 2;
    request.interpretation = obj({{"version",1},{"method","trace.run"},{"params",obj({{"prompt","A"}})}});
    reader->send(request);
    REQUIRE(world.until([&] { return reader->control_replies(2).size() == 1; }));
    CHECK(reader->control_replies(2)[0]->error.find("code")->as_string() == CALIPER_NET_ERR_FORBIDDEN);
    auto legacy = viewer(world, hub, kCaps31);
    REQUIRE(world.until([&] { return legacy->views() && legacy->views()->session; }));
    request.request_id = 3;
    request.interpretation = obj({{"version",1},{"method","trace.status"},{"params",Json::object()}});
    legacy->send(request);
    REQUIRE(world.until([&] { return legacy->control_replies(3).size() == 1; }));
    CHECK(legacy->control_replies(3)[0]->error.find("code")->as_string() == "interpret.unsupported");
    CHECK(producer_link->controls_in.size() == 1);
}

TEST_CASE("net/hub: interpretation refuses a producer without the capability") {
    HubRunner hub;
    World world;
    auto producer_link = producer(world, hub, "legacy-interpretation", 1, 1, kProducer31, true);
    auto caps = kCaps31;
    caps.push_back(CALIPER_NET_CAP_INTERPRETATION);
    auto reader = viewer(world, hub, caps);
    REQUIRE(world.until([&] { return reader->views() && reader->views()->session; }));
    wire::Control request;
    request.request_id = 1;
    request.op = CALIPER_NET_OP_INTERPRET;
    request.interpretation = obj({{"version",1},{"method","trace.status"},{"params",Json::object()}});
    reader->send(request);
    REQUIRE(world.until([&] { return reader->control_replies(1).size() == 1; }));
    CHECK(reader->control_replies(1)[0]->error.find("code")->as_string() == "interpret.unsupported");
    CHECK(producer_link->controls_in.empty());
}
