// Caliper API v0 (src/host/api/api_server.h). The routes, the token and the
// checks of a plain request as functions; the server in ONE process on
// 127.0.0.1 with a fake frame thread; and, when the ctest entry
// caliper_cli_api names the executable, `caliper shot --api` as a process,
// which is the app's windowless mode: every endpoint, a capture that is a PNG
// of the shot's size, the stream's handshake, and a clean stop on SIGTERM.
// Nothing here opens a window.
#include <doctest/doctest.h>

#include "api/api_server.h"
#include "api/window_captures.h"
#include "caliper/live_net.h"
#include "net/json.h"
#include "net/wire.h"
#include "net/ws.h"
#include "test_proc.h"
#include "test_temp.h"

#define STB_IMAGE_STATIC
#define STB_IMAGE_IMPLEMENTATION
#define STBI_ONLY_PNG
#include <stb_image.h>

#include <atomic>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <map>
#include <sstream>
#include <string>
#include <thread>
#include <unordered_set>
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
using api_sock = SOCKET;
#  define api_close closesocket
#else
#  include <arpa/inet.h>
#  include <netinet/in.h>
#  include <sys/socket.h>
#  include <sys/time.h>
#  include <unistd.h>
using api_sock = int;
#  define api_close ::close
#endif

namespace fs = std::filesystem;
namespace wire = caliper_net::wire;
using caliper_api::Route;
using caliper_net::HttpRequest;
using caliper_net::Json;

namespace {

const std::string kToken = "api_test_token-0123456789abcdef";

HttpRequest req(const std::string& method, const std::string& target,
                std::vector<std::pair<std::string, std::string>> headers = {}) {
    HttpRequest r;
    r.method = method;
    r.target = target;
    const size_t q = target.find('?');
    r.path = target.substr(0, q);
    r.query = q == std::string::npos ? std::string() : target.substr(q + 1);
    for (const auto& h : headers) {
        if (h.first == "origin") {
            r.has_origin = true;
            r.origin = h.second;
        }
        if (h.first == "host") r.host = h.second;
    }
    r.headers = std::move(headers);
    return r;
}

// ---- raw HTTP -----------------------------------------------------------------------------
struct Response {
    int status = 0;
    std::map<std::string, std::string> headers;   // names lower-cased
    std::string body;
};

Response parse_response(const std::string& raw) {
    Response r;
    const size_t end = raw.find("\r\n\r\n");
    if (raw.size() < 12 || end == std::string::npos) return r;
    r.status = std::atoi(raw.substr(9, 3).c_str());
    std::istringstream head(raw.substr(0, end));
    std::string line;
    std::getline(head, line);
    while (std::getline(head, line)) {
        if (!line.empty() && line.back() == '\r') line.pop_back();
        const size_t c = line.find(':');
        if (c == std::string::npos) continue;
        std::string name = line.substr(0, c);
        for (char& ch : name) ch = (char)std::tolower((unsigned char)ch);
        size_t v = c + 1;
        while (v < line.size() && line[v] == ' ') ++v;
        r.headers[name] = line.substr(v);
    }
    r.body = raw.substr(end + 4);
    return r;
}

api_sock open_socket(uint16_t port) {
    api_sock fd = ::socket(AF_INET, SOCK_STREAM, 0);
    sockaddr_in a{};
    a.sin_family = AF_INET;
    a.sin_port = htons(port);
    inet_pton(AF_INET, "127.0.0.1", &a.sin_addr);
    if (::connect(fd, (sockaddr*)&a, sizeof a) != 0) {
        api_close(fd);
        return (api_sock)-1;
    }
#ifdef _WIN32
    DWORD ms = 10000;
    setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, (const char*)&ms, sizeof ms);
#else
    timeval tv{10, 0};
    setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof tv);
#endif
    return fd;
}

std::string read_all(api_sock fd) {
    std::string out;
    char buf[65536];
    for (;;) {
        const int r = (int)::recv(fd, buf, sizeof buf, 0);
        if (r <= 0) break;
        out.append(buf, (size_t)r);
    }
    return out;
}

// One request; `host` replaces the Host header, `extra` ends with "\r\n" per line.
Response http(uint16_t port, const std::string& method, const std::string& target,
              const std::string& extra = {}, const std::string& host = {}) {
    const api_sock fd = open_socket(port);
    if (fd == (api_sock)-1) return {};
    const std::string h = host.empty() ? "127.0.0.1:" + std::to_string(port) : host;
    const std::string request = method + " " + target + " HTTP/1.1\r\nHost: " + h + "\r\n" + extra + "\r\n";
    ::send(fd, request.data(), (int)request.size(), 0);
    const std::string raw = read_all(fd);
    api_close(fd);
    return parse_response(raw);
}

std::string bearer(const std::string& token = kToken) { return "Authorization: Bearer " + token + "\r\n"; }

Json parse_json(const std::string& text) {
    Json j;
    std::string err;
    const bool ok = Json::parse(text.data(), text.size(), &j, &err);
    REQUIRE_MESSAGE(ok, std::string(err + ": " + text));
    return j;
}

// ---- a viewer's handshake on the stream ------------------------------------------------------
// HELLO as a viewer; true once WELCOME arrives, with its role in *role.
bool viewer_handshake(const std::string& url, std::string* role, std::string* why) {
    caliper_net::Client c;
    if (!c.connect(url, why)) return false;
    bool open = false;
    const auto end = std::chrono::steady_clock::now() + std::chrono::seconds(10);
    while (std::chrono::steady_clock::now() < end) {
        for (auto& e : c.poll(10)) {
            using T = caliper_net::Event::Type;
            if (e.type == T::Closed) {
                *why = "closed: " + std::to_string(e.http_status ? e.http_status : e.code) + " " + e.reason;
                return false;
            }
            if (e.type == T::Open && !open) {
                open = true;
                wire::Hello h;
                h.role = CALIPER_NET_ROLE_VIEWER;
                std::vector<uint8_t> b;
                REQUIRE(wire::encode(h, &b));
                c.send(b.data(), b.size());
            }
            if (e.type == T::Message) {
                wire::Message m;
                if (!wire::decode(e.data.data(), e.data.size(), &m).ok()) continue;
                if (auto* w = std::get_if<wire::Welcome>(&m)) {
                    *role = w->role;
                    c.close();
                    return true;
                }
            }
        }
    }
    *why = "no WELCOME within 10 s";
    return false;
}

// A producer that streams one 8 x 8 heatmap view, "heat", into the app's hub:
// it answers SUBSCRIBE and sends a snapshot every 30 ms for each open
// subscription. Pumped from the test thread.
struct TinyProducer {
    caliper_net::Client c;
    bool open = false, welcomed = false;
    int subscribes = 0;
    std::map<uint32_t, Json> subs;   // sub -> resolved data view
    uint64_t seq = 0;
    std::chrono::steady_clock::time_point next = std::chrono::steady_clock::now();

    bool connect(const std::string& url) {
        std::string err;
        return c.connect(url, &err);
    }
    void send(const wire::Message& m) {
        std::vector<uint8_t> b;
        REQUIRE(wire::encode(m, &b));
        c.send(b.data(), b.size());
    }
    void pump(int ms) {
        for (auto& e : c.poll(ms)) {
            using T = caliper_net::Event::Type;
            if (e.type == T::Open && !open) {
                open = true;
                wire::Hello h;
                h.role = CALIPER_NET_ROLE_PRODUCER;
                h.session = wire::Identity{"tiny-producer", "tiny.py", 7, 1}.to_json();
                send(h);
            }
            if (e.type != T::Message) continue;
            wire::Message m;
            if (!wire::decode(e.data.data(), e.data.size(), &m).ok()) continue;
            if (std::holds_alternative<wire::Welcome>(m)) {
                welcomed = true;
                wire::Views v;
                Json views = Json::array(), heat = Json::object(), src = Json::object(), dims = Json::array();
                for (const char* d : {"rows", "cols"}) {
                    Json dim = Json::object();
                    dim.set("name", d);
                    dim.set("size", 8);
                    dims.push(dim);
                }
                src.set("name", "w");
                src.set("dims", dims);
                Json sources = Json::array();
                sources.push(src);
                heat.set("id", 1);
                heat.set("name", "heat");
                heat.set("kind", (int)CALIPER_NET_KIND_HEATMAP);
                heat.set("sources", sources);
                views.push(heat);
                v.views = views;
                send(v);
            } else if (auto* s = std::get_if<wire::Subscribe>(&m)) {
                ++subscribes;
                wire::ViewReply r;
                r.request_id = s->request_id;
                r.sub = s->sub;
                r.view = s->view;
                Json dv = Json::object();
                dv.set("source", "w");
                r.data_view = dv;
                subs[s->sub] = dv;
                send(r);
            } else if (auto* u = std::get_if<wire::Unsubscribe>(&m)) {
                subs.erase(u->sub);
            } else if (auto* pi = std::get_if<wire::Ping>(&m)) {
                send(wire::Pong{pi->nonce});
            }
        }
        if (std::chrono::steady_clock::now() < next || subs.empty()) return;
        next = std::chrono::steady_clock::now() + std::chrono::milliseconds(30);
        std::vector<float> data(64);
        ++seq;
        for (size_t i = 0; i < data.size(); ++i) data[i] = (float)((i + seq) % 9);
        for (const auto& kv : subs) {
            wire::Snapshot s;
            s.sub = kv.first;
            s.view = "heat";
            s.seq = seq;
            s.kind = CALIPER_NET_KIND_HEATMAP;
            s.data_view = kv.second;
            s.data = wire::Section::make("data", CALIPER_NET_DTYPE_F32, {8, 8}, data.data(), data.size() * 4);
            s.flags = CALIPER_NET_SNAP_AUTO_VMIN | CALIPER_NET_SNAP_AUTO_VMAX;
            s.stats = {0.0, 8.0};
            send(s);
        }
    }
};

// ---- the server with a fake frame thread ------------------------------------------------
struct FakeApp {
    std::unique_ptr<caliper_api::Server> server;
    std::thread frame;
    std::atomic<bool> stop{false};
    std::atomic<bool> frames_on{true};   // false: the frame thread takes nothing
    std::atomic<int> launches{0};

    explicit FakeApp(caliper_api::Options o = caliper_api::Options()) {
        o.port = 0;
        if (o.token.empty()) o.token = kToken;
        if (o.info_json.empty()) o.info_json = "{\"host\":\"0.6.0\"}";
        if (!o.encode_png)
            o.encode_png = [](const uint8_t* px, uint32_t w, uint32_t h, std::string* out) {
                *out = "PNG " + std::to_string(w) + "x" + std::to_string(h) + " " + std::to_string(px[0]);
                return true;
            };
        server = std::make_unique<caliper_api::Server>(std::move(o));
        std::string err;
        REQUIRE_MESSAGE(server->start(&err), err);
        frame = std::thread([this] {
            while (!stop.load()) {
                if (frames_on.load()) {
                    for (const caliper_api::Call& c : server->take()) {
                        using K = caliper_api::Call::Kind;
                        if (c.kind == K::Applets) {
                            server->reply(c.ticket, 200, "{\"applets\":[]}");
                        } else if (c.kind == K::Launch) {
                            ++launches;
                            if (c.applet_id == "dev.caliper.hello")
                                server->reply(c.ticket, 200, "{\"id\":\"dev.caliper.hello\"}");
                            else
                                server->reply(c.ticket, 404, caliper_api::error_json("no applet " + c.applet_id));
                        } else {
                            server->reply_capture(c.ticket, 2, 2, std::vector<uint8_t>(16, 7));
                        }
                    }
                }
                std::this_thread::sleep_for(std::chrono::milliseconds(2));
            }
        });
    }
    ~FakeApp() {
        stop.store(true);
        if (frame.joinable()) frame.join();
        server->stop();
    }
    uint16_t port() const { return server->port(); }
};

// A new folder of this run's own (test_temp.h), removed when the case ends.
struct TempDir {
    fs::path path;
    explicit TempDir(const std::string& name) : path(caliper_test::unique_temp_path("caliper-api-" + name)) {
        fs::create_directories(path);
    }
    ~TempDir() {
        std::error_code ec;
        fs::remove_all(path, ec);
    }
    std::string str(const std::string& sub) const { return (path / sub).string(); }
};

}  // namespace

// ---- routing, the token and the checks --------------------------------------------------------

TEST_CASE("api: the routes of v0 and the method each takes") {
    CHECK(caliper_api::route("GET", "/v0/info").route == Route::Info);
    CHECK(caliper_api::route("GET", "/v0/applets").route == Route::Applets);
    CHECK(caliper_api::route("GET", "/v0/capture.png").route == Route::Capture);
    CHECK(caliper_api::route("GET", "/v3/stream").route == Route::Stream);
    CHECK(caliper_api::route("GET", "/v0/stream").route == Route::Stream);
    auto launch = caliper_api::route("POST", "/v0/applets/dev.caliper.hello/launch");
    CHECK(launch.route == Route::Launch);
    CHECK(launch.method_ok);
    CHECK(launch.applet_id == "dev.caliper.hello");
    auto wrong = caliper_api::route("GET", "/v0/applets/x/launch");
    CHECK(wrong.route == Route::Launch);
    CHECK_FALSE(wrong.method_ok);
    CHECK(std::string(wrong.allow) == "POST");
    CHECK_FALSE(caliper_api::route("POST", "/v0/info").method_ok);
    CHECK_FALSE(caliper_api::route("HEAD", "/v0/capture.png").method_ok);
    for (const char* none : {"/", "/v0", "/v0/", "/v0/info/", "/v1/info", "/v3/stream/", "/v3/", "/v0/applets/",
                             "/v0/applets//launch", "/v0/applets/a/b/launch", "/v0/applets/x/launch/",
                             "/v0/applets/x", "/V0/info", "/v0/capture", "/index.html"}) {
        CAPTURE(none);
        CHECK(caliper_api::route("GET", none).route == Route::None);
        CHECK(caliper_api::route("POST", none).route == Route::None);
    }
}

TEST_CASE("api: an applet id in the path is 1 to 128 characters of [A-Za-z0-9._-]") {
    CHECK(caliper_api::plausible_applet_id("dev.caliper.hello"));
    CHECK(caliper_api::plausible_applet_id("com.example.my-viewer_2"));
    CHECK(caliper_api::plausible_applet_id(std::string(128, 'a')));
    for (const std::string bad : {std::string(), std::string(129, 'a'), std::string("a%2Fb"),
                                  std::string("a b"), std::string("a\\b"), std::string("a:b"),
                                  std::string("caf\xc3\xa9")}) {
        CAPTURE(bad);
        CHECK_FALSE(caliper_api::plausible_applet_id(bad));
    }
}

TEST_CASE("api: the token travels in the Authorization header, or in the query of a GET") {
    using caliper_api::token_ok;
    CHECK(token_ok(req("GET", "/v0/info", {{"authorization", "Bearer " + kToken}}), kToken, true));
    CHECK(token_ok(req("POST", "/v0/info", {{"authorization", "bearer  " + kToken + " "}}), kToken, false));
    CHECK(token_ok(req("GET", "/v0/info?token=" + kToken), kToken, true));
    CHECK_FALSE(token_ok(req("POST", "/v0/info?token=" + kToken), kToken, false));
    CHECK_FALSE(token_ok(req("GET", "/v0/info"), kToken, true));
    CHECK_FALSE(token_ok(req("GET", "/v0/info", {{"authorization", "Bearer " + kToken + "x"}}), kToken, true));
    CHECK_FALSE(token_ok(req("GET", "/v0/info", {{"authorization", "Basic " + kToken}}), kToken, true));
    CHECK_FALSE(token_ok(req("GET", "/v0/info", {{"authorization", kToken}}), kToken, true));
    CHECK_FALSE(token_ok(req("GET", "/v0/info", {{"authorization", "Bearer"}}), kToken, true));
    // A wrong header is not rescued by a right query parameter.
    CHECK_FALSE(token_ok(req("GET", "/v0/info?token=" + kToken, {{"authorization", "Bearer nope"}}), kToken, true));
    CHECK_FALSE(token_ok(req("GET", "/v0/info?token="), kToken, true));
    CHECK_FALSE(token_ok(req("GET", "/v0/info?token=" + kToken), "", true));
}

TEST_CASE("api: a plain request is checked for Origin, route, method, token, body and id, in order") {
    using caliper_api::check;
    const auto auth = std::make_pair(std::string("authorization"), "Bearer " + kToken);
    CHECK(check(req("GET", "/v0/info", {auth}), kToken).status == 0);
    CHECK(check(req("GET", "/v0/info", {auth, {"origin", "http://localhost:8888"}}), kToken).status == 0);
    CHECK(check(req("GET", "/v0/info", {auth, {"origin", "null"}}), kToken).status == 0);
    CHECK(check(req("GET", "/nope", {{"origin", "https://evil.example"}}), kToken).status == 403);
    CHECK(check(req("GET", "/v0/info", {auth, {"origin", "http://127.0.0.1.evil.example"}}), kToken).status == 403);
    CHECK(check(req("GET", "/nope"), kToken).status == 404);
    CHECK(check(req("POST", "/v0/info"), kToken).status == 405);
    CHECK(check(req("GET", "/v0/info"), kToken).status == 401);
    CHECK(check(req("POST", "/v0/applets/x/launch?token=" + kToken), kToken).status == 401);
    CHECK(check(req("POST", "/v0/applets/x/launch", {auth, {"content-length", "4"}}), kToken).status == 413);
    CHECK(check(req("POST", "/v0/applets/x/launch", {auth, {"transfer-encoding", "chunked"}}), kToken).status == 413);
    CHECK(check(req("POST", "/v0/applets/x/launch", {auth, {"content-length", "0"}}), kToken).status == 0);
    CHECK(check(req("POST", "/v0/applets/a%2Fb/launch", {auth}), kToken).status == 400);
    const caliper_api::Verdict ok = check(req("POST", "/v0/applets/x.y/launch", {auth}), kToken);
    CHECK(ok.status == 0);
    CHECK(ok.routed.applet_id == "x.y");
}

// ---- the server in this process ----------------------------------------------------------------

TEST_CASE("api: the server answers every endpoint, through the frame thread where it must") {
    FakeApp app;
    const uint16_t port = app.port();
    REQUIRE(port != 0);
    CHECK(app.server->base_url() == "http://127.0.0.1:" + std::to_string(port) + "/v0");
    CHECK(app.server->stream_url() == "ws://127.0.0.1:" + std::to_string(port) + "/v3/stream");

    Response info = http(port, "GET", "/v0/info", bearer());
    CHECK(info.status == 200);
    CHECK(info.body == "{\"host\":\"0.6.0\"}");
    CHECK(info.headers["content-type"] == "application/json");
    CHECK(info.headers["cache-control"] == "no-store");
    CHECK(info.headers["x-content-type-options"] == "nosniff");
    CHECK(http(port, "GET", "/v0/info?token=" + kToken).status == 200);

    Response applets = http(port, "GET", "/v0/applets", bearer());
    CHECK(applets.status == 200);
    CHECK(applets.body == "{\"applets\":[]}");

    Response launch = http(port, "POST", "/v0/applets/dev.caliper.hello/launch", bearer());
    CHECK(launch.status == 200);
    CHECK(launch.body == "{\"id\":\"dev.caliper.hello\"}");
    CHECK(http(port, "POST", "/v0/applets/no.such/launch", bearer()).status == 404);
    CHECK(app.launches.load() == 2);

    Response capture = http(port, "GET", "/v0/capture.png?token=" + kToken);
    CHECK(capture.status == 200);
    CHECK(capture.headers["content-type"] == "image/png");
    CHECK(capture.body == "PNG 2x2 7");

    // Refusals.
    Response anon = http(port, "GET", "/v0/info");
    CHECK(anon.status == 401);
    CHECK(anon.headers["www-authenticate"].rfind("Bearer", 0) == 0);
    CHECK(http(port, "GET", "/v0/info", bearer("api_test_token-0123456789abcdeX")).status == 401);
    Response post_query = http(port, "POST", "/v0/applets/dev.caliper.hello/launch?token=" + kToken);
    CHECK(post_query.status == 401);
    Response method = http(port, "DELETE", "/v0/applets", bearer());
    CHECK(method.status == 405);
    CHECK(method.headers["allow"] == "GET");
    CHECK(http(port, "GET", "/", bearer()).status == 404);
    CHECK(http(port, "GET", "/index.html", bearer()).status == 404);
    CHECK(http(port, "GET", "/v3/info?token=" + kToken).status == 404);
    CHECK(http(port, "POST", "/v0/applets/dev.caliper.hello/launch", bearer() + "Content-Length: 2\r\n").status == 413);
    CHECK(app.launches.load() == 2);   // refused requests never reached the app

    const caliper_api::Stats st = app.server->stats();
    CHECK(st.answered == 5);
    CHECK(st.pending == 0);
}

TEST_CASE("api: Host and Origin are refused on every path, the stream included") {
    FakeApp app;
    const uint16_t port = app.port();
    for (const char* target : {"/v0/info", "/v0/applets", "/v0/capture.png", "/v3/stream?token=api_test_token-0123456789abcdef",
                               "/v0/stream?token=api_test_token-0123456789abcdef"}) {
        CAPTURE(target);
        CHECK(http(port, "GET", target, bearer(), "evil.example:" + std::to_string(port)).status == 403);
        CHECK(http(port, "GET", target, bearer(), "127.0.0.1:1").status == 403);
        CHECK(http(port, "GET", target, bearer() + "Origin: https://evil.example\r\n").status == 403);
    }
    CHECK(http(port, "GET", "/v0/info", bearer(), "localhost:" + std::to_string(port)).status == 200);
    CHECK(app.launches.load() == 0);
}

TEST_CASE("api: /v3/stream, and /v0/stream as its alias, is wire protocol v3 with the app as the hub") {
    FakeApp app;
    for (const char* path : {"/v3/stream", "/v0/stream"}) {
        CAPTURE(path);
        const std::string base = "ws://127.0.0.1:" + std::to_string(app.port()) + path + "?token=";
        std::string role, why;
        REQUIRE_MESSAGE(viewer_handshake(base + kToken, &role, &why), why);
        CHECK(role == CALIPER_NET_ROLE_HUB);
        CHECK_FALSE(viewer_handshake(base + "api_test_token-0123456789abcdeX", &role, &why));
        CHECK(why.find("403") != std::string::npos);
        // A plain GET of the stream path is not an upgrade.
        CHECK(http(app.port(), "GET", std::string(path) + "?token=" + kToken).status == 426);
    }
    // Other paths under /v3/ are not the API's.
    CHECK(http(app.port(), "GET", "/v3/other", bearer()).status == 404);
}

TEST_CASE("api: a call the frame thread does not answer times out, and waiting calls are bounded") {
    caliper_api::Options o;
    o.frame_timeout_ms = 300;
    o.max_pending = 2;
    FakeApp app(o);
    app.frames_on.store(false);
    const uint16_t port = app.port();
    // Two calls wait; the third is refused at once.
    const api_sock a = open_socket(port), b = open_socket(port);
    const std::string get = "GET /v0/applets HTTP/1.1\r\nHost: 127.0.0.1:" + std::to_string(port) + "\r\n" + bearer() + "\r\n";
    ::send(a, get.data(), (int)get.size(), 0);
    ::send(b, get.data(), (int)get.size(), 0);
    const auto t0 = std::chrono::steady_clock::now();
    while (app.server->stats().pending < 2 && std::chrono::steady_clock::now() - t0 < std::chrono::seconds(2))
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
    Response busy = http(port, "GET", "/v0/applets", bearer());
    CHECK(busy.status == 503);
    CHECK(busy.headers["retry-after"] == "1");
    CHECK(http(port, "GET", "/v0/info", bearer()).status == 200);   // info needs no frame
    const Response ra = parse_response(read_all(a)), rb = parse_response(read_all(b));
    api_close(a);
    api_close(b);
    const double waited = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    CHECK(ra.status == 504);
    CHECK(rb.status == 504);
    CHECK(waited < 3.0);
    // A late answer is dropped, and the next call is served.
    app.frames_on.store(true);
    CHECK(http(port, "GET", "/v0/applets", bearer()).status == 200);
    const caliper_api::Stats st = app.server->stats();
    CHECK(st.timeouts == 2);
    CHECK(st.busy == 1);
}

TEST_CASE("api: stopping answers the calls still waiting with 503") {
    auto app = std::make_unique<FakeApp>();
    app->frames_on.store(false);
    const uint16_t port = app->port();
    const api_sock a = open_socket(port);
    const std::string get = "GET /v0/capture.png?token=" + kToken + " HTTP/1.1\r\nHost: 127.0.0.1:" + std::to_string(port) + "\r\n\r\n";
    ::send(a, get.data(), (int)get.size(), 0);
    const auto t0 = std::chrono::steady_clock::now();
    while (app->server->stats().pending < 1 && std::chrono::steady_clock::now() - t0 < std::chrono::seconds(2))
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
    app.reset();
    const Response r = parse_response(read_all(a));
    api_close(a);
    CHECK(r.status == 503);
    CHECK(r.body.find("closing") != std::string::npos);
}

TEST_CASE("api: capture answers 501 where the app cannot read its frame back") {
    caliper_api::Options o;
    o.capture_unavailable = "the window cannot be read back";
    FakeApp app(o);
    Response r = http(app.port(), "GET", "/v0/capture.png", bearer());
    CHECK(r.status == 501);
    CHECK(r.body == caliper_api::error_json("the window cannot be read back"));
}

TEST_CASE("api: the server refuses a token that is not a session token, and a port in use") {
    caliper_api::Options o;
    o.port = 0;
    o.token = "short";
    caliper_api::Server s(o);
    std::string err;
    CHECK_FALSE(s.start(&err));
    CHECK(err.find("session token") != std::string::npos);
    FakeApp app;
    caliper_api::Options same;
    same.port = app.port();
    same.token = kToken;
    caliper_api::Server second(same);
    CHECK_FALSE(second.start(&err));
}

// ---- the window's captures (api/window_captures.h) --------------------------------------
namespace {
// A renderer whose window read-back the test scripts: requests are counted, and
// poll() returns what `next` says.
struct ScriptedRenderer final : caliper_host::HostRenderer {
    bool supported = true;
    int requests = 0;
    WindowCapture next = WindowCapture::Pending;
    int w = 3, h = 2;
    bool init(GLFWwindow*) override { return true; }
    void new_frame() override {}
    void render(int, int) override {}
    void shutdown() override {}
    const char* name() const override { return "scripted"; }
    uint64_t tex_create_rgba8(int, int) override { return 0; }
    bool tex_upload_rgba8(uint64_t, const void*, int, int) override { return false; }
    void tex_release(uint64_t) override {}
    uint64_t tex_imtexture_id(uint64_t) override { return 0; }
    bool tex_update_from_device(uint64_t, const CaliperTensor&, const uint32_t*, float, float) override {
        return false;
    }
    void window_hints() override {}
    bool window_capture_supported() const override { return supported; }
    bool window_capture_request() override {
        if (!supported) return false;
        ++requests;
        return true;
    }
    WindowCapture window_capture_poll(std::vector<uint8_t>* rgba, int* pw, int* ph) override {
        if (next == WindowCapture::Ready) {
            rgba->assign((size_t)w * (size_t)h * 4, 7);
            *pw = w;
            *ph = h;
        }
        return next;
    }
};
}  // namespace

TEST_CASE("api: a capture in the window waits for the read-back of a presented frame, never the frame loop") {
    ScriptedRenderer r;
    caliper_host::WindowCaptures caps;
    std::vector<std::pair<uint64_t, std::string>> answers;   // ticket, "WxH" or "status: reason"
    auto done = [&](uint64_t t, uint32_t w, uint32_t h, std::vector<uint8_t> rgba) {
        CHECK(rgba.size() == (size_t)w * h * 4);
        answers.emplace_back(t, std::to_string(w) + "x" + std::to_string(h));
    };
    auto fail = [&](uint64_t t, int status, const std::string& why) {
        answers.emplace_back(t, std::to_string(status) + ": " + why);
    };
    // Two calls of one frame arm one read-back.
    caps.add(1);
    caps.add(2);
    caps.before_frame(r, fail);
    CHECK(r.requests == 1);
    caps.after_frame(r, done, fail);   // still copying: nothing answered, nothing waited for
    CHECK(answers.empty());
    // A call while that copy is in flight waits for the next read-back.
    caps.add(3);
    caps.before_frame(r, fail);
    CHECK(r.requests == 1);
    r.next = caliper_host::HostRenderer::WindowCapture::Ready;
    caps.after_frame(r, done, fail);
    REQUIRE(answers.size() == 2);
    CHECK(answers[0] == std::make_pair(uint64_t(1), std::string("3x2")));
    CHECK(answers[1] == std::make_pair(uint64_t(2), std::string("3x2")));
    CHECK(caps.pending() == 1);
    caps.before_frame(r, fail);
    CHECK(r.requests == 2);
    r.next = caliper_host::HostRenderer::WindowCapture::Failed;
    caps.after_frame(r, done, fail);
    REQUIRE(answers.size() == 3);
    CHECK(answers[2].first == 3);
    CHECK(answers[2].second.rfind("500: could not read the window's frame back from the scripted renderer", 0) == 0);
    CHECK(caps.pending() == 0);
    // A renderer that cannot read its window back answers 501 with the reason.
    r.supported = false;
    caps.add(4);
    caps.before_frame(r, fail);
    REQUIRE(answers.size() == 4);
    CHECK(answers[3].second == "501: " + caliper_host::WindowCaptures::unsupported("scripted"));
    CHECK(answers[3].second.find("caliper shot --api") != std::string::npos);
    CHECK(caps.pending() == 0);
}

// ---- caliper shot --api as a process ------------------------------------------------------
// Runs the caliper executable named by CALIPER_TEST_CALIPER_EXE (the ctest
// entry caliper_cli_api sets it, and CALIPER_TEST_APPLETS to the build's
// applets folder) as `caliper shot --api`: no window, the page drawn into the
// offscreen canvas. Without a GPU, shot cannot draw and the run is reported
// as skipped, as the shot tests do. POSIX only (it sends a signal).
#ifndef _WIN32
TEST_CASE("api: caliper shot --api serves every endpoint as a process and stops on SIGTERM") {
    const char* exe = std::getenv("CALIPER_TEST_CALIPER_EXE");
    const char* applets = std::getenv("CALIPER_TEST_APPLETS");
    if (!exe || !*exe || !applets || !*applets) {
        MESSAGE("CALIPER_TEST_CALIPER_EXE or CALIPER_TEST_APPLETS is not set: run the ctest entry caliper_cli_api");
        return;
    }
    TempDir t("process");
    const std::string data = t.str("data");
    const std::string out_png = t.str("shot.png");
    Proc p;
    REQUIRE(p.start(exe,
                    {"shot", "--applets", applets, "--launch", "dev.caliper.hello", "--size", "640x400",
                     "--frames", "36000", "--out", out_png, "--api", "--api-port", "0"},
                    {"CALIPER_DATA_DIR=" + data}));
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(60);
    while (p.text.find("[api] Caliper API on ") == std::string::npos &&
           p.text.find("shot cannot draw offscreen") == std::string::npos &&
           std::chrono::steady_clock::now() < deadline && p.pid > 0) {
        p.read_some(20);
        if (::waitpid(p.pid, &p.status, WNOHANG) == p.pid) p.pid = -1;
    }
    p.read_some(50);
    if (p.text.find("shot cannot draw offscreen") != std::string::npos) {
        const char* require = std::getenv("CALIPER_REQUIRE_GPU");
        REQUIRE_MESSAGE(!(require && std::string(require) == "1"), p.text);
        MESSAGE("caliper shot --api: run skipped: no GPU");
        return;
    }
    REQUIRE_MESSAGE(p.text.find("[api] Caliper API on ") != std::string::npos, p.text);

    // The API file: owner only, with the port and the token the line printed.
    const std::string api_file = (fs::path(data) / "api.json").string();
    REQUIRE(fs::exists(api_file));
    CHECK((fs::status(api_file).permissions() & (fs::perms::group_all | fs::perms::others_all)) == fs::perms::none);
    std::ifstream af(api_file);
    const std::string api_text((std::istreambuf_iterator<char>(af)), std::istreambuf_iterator<char>());
    const Json api = parse_json(api_text);
    uint64_t port64 = 0;
    REQUIRE(api.find("port")->get_u64(&port64));
    const uint16_t port = (uint16_t)port64;
    const std::string token = api.find("token")->as_string();
    CHECK(caliper_net::valid_token(token));
    CHECK(api.find("url")->as_string() == "http://127.0.0.1:" + std::to_string(port) + "/v0");
    CHECK(api.find("stream_url")->as_string() == "ws://127.0.0.1:" + std::to_string(port) + "/v3/stream");
    CHECK(p.text.find("http://127.0.0.1:" + std::to_string(port) + "/v0 with the session token " + token +
                      " (also in " + api_file + ")") != std::string::npos);

    // info
    Response info = http(port, "GET", "/v0/info", bearer(token));
    REQUIRE(info.status == 200);
    const Json ij = parse_json(info.body);
    CHECK_FALSE(ij.find("host")->as_string().empty());
    int64_t epoch = 0;
    CHECK(ij.find("abi_epoch")->get_i64(&epoch));
    CHECK(epoch >= 2);
    CHECK(ij.find("services")->size() > 0);
    CHECK(ij.has("sdk"));

    // applets: the shot's applet is ACTIVE, the other one READY.
    auto status_of = [&](const std::string& id) {
        Response r = http(port, "GET", "/v0/applets", bearer(token));
        REQUIRE(r.status == 200);
        const Json j = parse_json(r.body);
        for (const Json& e : j.find("applets")->items())
            if (e.find("id")->as_string() == id) return e.find("status")->as_string();
        return std::string("absent");
    };
    CHECK(status_of("dev.caliper.hello") == "ACTIVE");
    CHECK(status_of("dev.example.sine-scope") == "READY");

    // capture: a PNG of the shot's size that is not one flat colour.
    auto capture = [&] {
        Response r = http(port, "GET", "/v0/capture.png?token=" + token);
        REQUIRE(r.status == 200);
        CHECK(r.headers["content-type"] == "image/png");
        int w = 0, h = 0, n = 0;
        unsigned char* px = stbi_load_from_memory((const unsigned char*)r.body.data(), (int)r.body.size(), &w, &h, &n, 4);
        REQUIRE_MESSAGE(px != nullptr, stbi_failure_reason());
        std::unordered_set<uint32_t> colours;
        for (size_t i = 0; i < (size_t)w * (size_t)h; ++i) {
            const unsigned char* q = px + i * 4;
            colours.insert((uint32_t)q[0] | (uint32_t)q[1] << 8 | (uint32_t)q[2] << 16 | (uint32_t)q[3] << 24);
        }
        stbi_image_free(px);
        CHECK(w == 640);
        CHECK(h == 400);
        CHECK(colours.size() > 1);
        return colours.size();
    };
    capture();

    // launch: another applet, the same one again, an unknown one.
    Response launched = http(port, "POST", "/v0/applets/dev.example.sine-scope/launch", bearer(token));
    REQUIRE_MESSAGE(launched.status == 200, launched.body);
    const Json lj = parse_json(launched.body);
    CHECK(lj.find("id")->as_string() == "dev.example.sine-scope");
    CHECK_FALSE(lj.find("already_open")->as_bool());
    CHECK(status_of("dev.example.sine-scope") == "ACTIVE");
    CHECK(status_of("dev.caliper.hello") == "READY");
    Response again = http(port, "POST", "/v0/applets/dev.example.sine-scope/launch", bearer(token));
    CHECK(again.status == 200);
    CHECK(parse_json(again.body).find("already_open")->as_bool());
    CHECK(http(port, "POST", "/v0/applets/dev.example.nowhere/launch", bearer(token)).status == 404);
    CHECK(http(port, "POST", "/v0/applets/dev.caliper.hello/launch?token=" + token).status == 401);
    CHECK(http(port, "GET", "/v0/applets/dev.caliper.hello/launch", bearer(token)).status == 405);
    capture();

    // The stream's handshake on both paths: the app answers as the hub.
    for (const char* path : {"/v3/stream", "/v0/stream"}) {
        std::string role, why;
        REQUIRE_MESSAGE(viewer_handshake("ws://127.0.0.1:" + std::to_string(port) + path + "?token=" + token, &role, &why), why);
        CHECK(role == CALIPER_NET_ROLE_HUB);
    }

    // The native viewer (caliper_viewer --headless, the app's NetSource)
    // attaches through /v3/stream on the API port, unchanged, to a session that
    // a fake producer streams into the app.
    if (const char* viewer = std::getenv("CALIPER_TEST_VIEWER_EXE"); viewer && *viewer) {
        TinyProducer prod;
        REQUIRE(prod.connect("ws://127.0.0.1:" + std::to_string(port) + "/v3/stream?token=" + token));
        const auto ready = std::chrono::steady_clock::now() + std::chrono::seconds(10);
        while (!prod.welcomed && std::chrono::steady_clock::now() < ready) prod.pump(10);
        REQUIRE(prod.welcomed);
        Proc v;
        REQUIRE(v.start(viewer, {"--headless", "--url", "ws://127.0.0.1:" + std::to_string(port) + "/v3/stream?token=" + token,
                                 "--exit-after", "3", "--stats", "--data-dir", t.str("viewer-data")}, {}));
        int code = -1;
        const auto until = std::chrono::steady_clock::now() + std::chrono::seconds(30);
        while (std::chrono::steady_clock::now() < until) {
            prod.pump(5);
            v.read_some(0);
            if (::waitpid(v.pid, &v.status, WNOHANG) == v.pid) {
                v.pid = -1;
                v.read_some(50);
                code = WIFEXITED(v.status) ? WEXITSTATUS(v.status) : -1;
                break;
            }
        }
        CHECK_MESSAGE(code == 0, v.text);
        CHECK_MESSAGE(v.text.find("received session=tiny-producer ") != std::string::npos, v.text);
        const size_t at = v.text.find("received view=heat kind=heatmap updates=");
        REQUIRE_MESSAGE(at != std::string::npos, v.text);
        CHECK(std::atoi(v.text.c_str() + at + std::strlen("received view=heat kind=heatmap updates=")) > 0);
        CHECK(prod.subscribes >= 1);
    } else {
        MESSAGE("CALIPER_TEST_VIEWER_EXE is not set: the native viewer's attach through /v3/stream is not run");
    }

    // Refusals.
    CHECK(http(port, "GET", "/v0/info").status == 401);
    CHECK(http(port, "GET", "/v0/info", bearer(token), "evil.example:" + std::to_string(port)).status == 403);
    CHECK(http(port, "GET", "/v0/info", bearer(token) + "Origin: https://evil.example\r\n").status == 403);
    CHECK(http(port, "GET", "/", bearer(token)).status == 404);

    // SIGTERM ends the frames; the shot writes its image of the applet it
    // shows and leaves neither the API file nor a listening socket.
    REQUIRE(p.pid > 0);
    ::kill(p.pid, SIGTERM);
    CHECK(p.wait(15000) == 0);
    CHECK_MESSAGE(p.text.find("wrote " + out_png + ": dev.example.sine-scope at 640x400 after ") != std::string::npos, p.text);
    CHECK(fs::exists(out_png));
    CHECK_FALSE(fs::exists(api_file));
    CHECK(open_socket(port) == (api_sock)-1);
    CHECK(p.text.find(token, p.text.find("(also in")) == std::string::npos);   // printed once
}

// ---- the app itself, hidden (--hidden), as a process -------------------------------------
// Runs the caliper executable as the app with its window never shown
// (--hidden: no Dock icon, no menu bar, no focus) and Caliper API on, with
// its data folder, user applets folder and window size set for the test. A
// client reads the token from the API file and uses every endpoint; the
// capture is the frame the hidden window presents, read back on Metal. SIGTERM
// closes the app normally. Without a Metal device the app falls back to GL,
// which cannot read its window back, and the run is reported as skipped.
namespace {
bool decode_png(const std::string& body, int* w, int* h, std::vector<uint8_t>* px) {
    int n = 0;
    unsigned char* p = stbi_load_from_memory((const unsigned char*)body.data(), (int)body.size(), w, h, &n, 4);
    if (!p) return false;
    px->assign(p, p + (size_t)*w * (size_t)*h * 4);
    stbi_image_free(p);
    return true;
}
}  // namespace

TEST_CASE("api: the hidden app serves every endpoint through its window and stops on SIGTERM") {
    const char* exe = std::getenv("CALIPER_TEST_CALIPER_EXE");
    const char* applets = std::getenv("CALIPER_TEST_APPLETS");
    if (!exe || !*exe || !applets || !*applets) {
        MESSAGE("CALIPER_TEST_CALIPER_EXE or CALIPER_TEST_APPLETS is not set: run the ctest entry caliper_cli_app_hidden");
        return;
    }
    TempDir t("app-hidden");
    const std::string data = t.str("data");
    fs::create_directories(t.str("user-applets"));
    Proc p;
    REQUIRE(p.start(exe, {"--hidden", "--applets", applets, "--api", "--api-port", "0"},
                    {"CALIPER_DATA_DIR=" + data, "CALIPER_USER_APPLETS_DIR=" + t.str("user-applets"),
                     "CALIPER_WINDOW_SIZE=640x400"}));
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(60);
    while (p.text.find("[api] Caliper API on ") == std::string::npos && std::chrono::steady_clock::now() < deadline &&
           p.pid > 0) {
        p.read_some(20);
        if (::waitpid(p.pid, &p.status, WNOHANG) == p.pid) p.pid = -1;
    }
    p.read_some(50);
    REQUIRE_MESSAGE(p.text.find("[api] Caliper API on ") != std::string::npos, p.text);
    if (p.text.find("[renderer] metal") == std::string::npos) {
        const char* require = std::getenv("CALIPER_REQUIRE_GPU");
        REQUIRE_MESSAGE(!(require && std::string(require) == "1"), p.text);
        MESSAGE("caliper --hidden --api: run skipped: no GPU (no Metal device)");
        return;
    }
    // Never shown, never focused, no Dock icon or menu bar.
    CHECK_MESSAGE(p.text.find("[hidden] the window is not shown (visible 0, focused 0, background policy 1") !=
                      std::string::npos,
                  p.text);

    // The token, read from the API file as a client reads it.
    const std::string api_file = (fs::path(data) / "api.json").string();
    REQUIRE(fs::exists(api_file));
    std::ifstream af(api_file);
    const Json api = parse_json(std::string((std::istreambuf_iterator<char>(af)), std::istreambuf_iterator<char>()));
    uint64_t port64 = 0;
    REQUIRE(api.find("port")->get_u64(&port64));
    const uint16_t port = (uint16_t)port64;
    const std::string token = api.find("token")->as_string();
    CHECK(api.find("stream_url")->as_string() == "ws://127.0.0.1:" + std::to_string(port) + "/v3/stream");

    Response info = http(port, "GET", "/v0/info", bearer(token));
    REQUIRE(info.status == 200);
    CHECK(parse_json(info.body).find("services")->size() > 0);
    auto status_of = [&](const std::string& id) {
        Response r = http(port, "GET", "/v0/applets", bearer(token));
        REQUIRE(r.status == 200);
        const Json j = parse_json(r.body);
        for (const Json& e : j.find("applets")->items())
            if (e.find("id")->as_string() == id) return e.find("status")->as_string();
        return std::string("absent");
    };
    CHECK(status_of("dev.caliper.hello") == "READY");

    // A capture: the frame the hidden window presents, at its framebuffer's
    // size (640 x 400 points, one or two pixels per point), not one colour.
    auto capture = [&](std::vector<uint8_t>* px) {
        Response r = http(port, "GET", "/v0/capture.png?token=" + token);
        REQUIRE_MESSAGE(r.status == 200, r.body);
        CHECK(r.headers["content-type"] == "image/png");
        int w = 0, h = 0;
        REQUIRE(decode_png(r.body, &w, &h, px));
        CHECK(((w == 640 && h == 400) || (w == 1280 && h == 800)));
        std::unordered_set<uint32_t> colours;
        for (size_t i = 0; i < (size_t)w * (size_t)h; ++i) {
            uint32_t c;
            std::memcpy(&c, px->data() + 4 * i, 4);
            colours.insert(c);
        }
        CHECK(colours.size() > 1);
    };
    std::vector<uint8_t> landing, page;
    capture(&landing);

    // Launch the hello fixture: the window then shows its page.
    Response launched = http(port, "POST", "/v0/applets/dev.caliper.hello/launch", bearer(token));
    REQUIRE_MESSAGE(launched.status == 200, launched.body);
    CHECK(parse_json(launched.body).find("id")->as_string() == "dev.caliper.hello");
    CHECK(status_of("dev.caliper.hello") == "ACTIVE");
    capture(&page);
    CHECK(page.size() == landing.size());
    CHECK(page != landing);

    // The stream's handshake on the protocol's path.
    std::string role, why;
    REQUIRE_MESSAGE(viewer_handshake("ws://127.0.0.1:" + std::to_string(port) + "/v3/stream?token=" + token, &role, &why),
                    why);
    CHECK(role == CALIPER_NET_ROLE_HUB);

    // SIGTERM closes the hidden window as a click would: the teardown runs, the
    // API file goes, nothing listens, no layout was written.
    REQUIRE(p.pid > 0);
    ::kill(p.pid, SIGTERM);
    CHECK_MESSAGE(p.wait(20000) == 0, p.text);
    CHECK_FALSE(fs::exists(api_file));
    CHECK_FALSE(fs::exists(fs::path(data) / "imgui.ini"));
    CHECK(open_socket(port) == (api_sock)-1);
    CHECK(p.text.find("the window became visible") == std::string::npos);
}
#endif
