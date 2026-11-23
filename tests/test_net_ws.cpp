// The WebSocket layer of wire protocol v3 (src/host/net/ws.h): its client
// against its own server on 127.0.0.1, in one thread that pumps both sides
// (every call is non-blocking, so one loop drives the two ends). A raw TCP
// socket stands in for a peer that breaks the rules (unmasked frames, plain
// HTTP, a slow or silent client). No other process, no window.
#include <doctest/doctest.h>

#include "net/sha1.h"
#include "net/ws.h"

#include <atomic>
#include <chrono>
#include <cstring>
#include <functional>
#include <memory>
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
#  include <poll.h>
#  include <sys/ioctl.h>
#  include <sys/socket.h>
#  include <sys/time.h>
#  include <unistd.h>
using raw_sock = int;
#  define RAW_BAD (-1)
#  define raw_close ::close
#endif

using namespace caliper_net;
using Type = Event::Type;

namespace {

// ---- a blocking raw TCP peer ------------------------------------------------------

struct Raw {
    raw_sock fd = RAW_BAD;
    explicit Raw(uint16_t port) {
        fd = ::socket(AF_INET, SOCK_STREAM, 0);
        sockaddr_in a{};
        a.sin_family = AF_INET;
        a.sin_port = htons(port);
        inet_pton(AF_INET, "127.0.0.1", &a.sin_addr);
        if (::connect(fd, (sockaddr*)&a, sizeof a) != 0) {
            raw_close(fd);
            fd = RAW_BAD;
        }
#ifdef _WIN32
        DWORD ms = 100;
        setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, (const char*)&ms, sizeof ms);
#else
        timeval tv{0, 100000};
        setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof tv);
#endif
    }
    ~Raw() {
        if (fd != RAW_BAD) raw_close(fd);
    }
    void send_all(const std::string& s) {
        size_t off = 0;
        while (off < s.size()) {
            const int w = (int)::send(fd, s.data() + off, (int)(s.size() - off), 0);
            if (w <= 0) return;
            off += (size_t)w;
        }
    }
    // Reads what is there within one receive timeout (100 ms). Empty on EOF or
    // timeout; *eof says which.
    std::string recv_some(bool* eof = nullptr) {
        char buf[65536];
        const int r = (int)::recv(fd, buf, sizeof buf, 0);
        if (eof) *eof = r == 0;
        return r > 0 ? std::string(buf, (size_t)r) : std::string();
    }
};

// ---- pumping both ends --------------------------------------------------------------

struct Pump {
    Server* server = nullptr;
    std::vector<Client*> clients;
    std::vector<Event> sev;                  // server events, in order
    std::vector<std::vector<Event>> cev;     // per client

    explicit Pump(Server* s, std::vector<Client*> c = {}) : server(s), clients(std::move(c)) {
        cev.resize(clients.size());
    }
    void add(Client* c) {
        clients.push_back(c);
        cev.emplace_back();
    }
    void once(int server_wait_ms = 1) {
        if (server) {
            auto e = server->poll(server_wait_ms);
            for (auto& x : e) sev.push_back(std::move(x));
        }
        for (size_t i = 0; i < clients.size(); ++i) {
            if (!clients[i]) continue;
            auto e = clients[i]->poll(0);
            for (auto& x : e) cev[i].push_back(std::move(x));
        }
    }
    bool until(const std::function<bool()>& done, int ms = 5000) {
        const auto end = std::chrono::steady_clock::now() + std::chrono::milliseconds(ms);
        while (std::chrono::steady_clock::now() < end) {
            once();
            if (done()) return true;
        }
        return false;
    }
    static const Event* find(const std::vector<Event>& v, Type t, ConnId c = 0) {
        for (const auto& e : v)
            if (e.type == t && (c == 0 || e.conn == c)) return &e;
        return nullptr;
    }
    static size_t count(const std::vector<Event>& v, Type t, ConnId c = 0) {
        size_t n = 0;
        for (const auto& e : v) n += e.type == t && (c == 0 || e.conn == c);
        return n;
    }
};

std::string url(uint16_t port, const std::string& rest = "/v3/stream?token=abcdefghijklmnopqrstuv") {
    return "ws://127.0.0.1:" + std::to_string(port) + rest;
}

// Accepts every upgrade request the server reported so far (and answers a
// plain request with 404).
void accept_all(Server& s, Pump& p, size_t* handled) {
    for (; *handled < p.sev.size(); ++*handled) {
        const Event& e = p.sev[*handled];
        if (e.type != Type::Request) continue;
        if (e.request.websocket) s.accept(e.conn);
        else s.reject(e.conn, 404);
    }
}

// One server and one open client connection.
struct Pair {
    Server server;
    Client client;
    Pump pump{&server};
    ConnId conn = 0;   // the server's id for the client
    size_t handled = 0;

    explicit Pair(Options so = Options(), Options co = Options()) : server(so), client(co) {
        std::string err;
        REQUIRE(server.listen(0, &err));
        REQUIRE(client.connect(url(server.port()), &err));
        pump.add(&client);
        REQUIRE(pump.until([&] {
            accept_all(server, pump, &handled);
            return Pump::find(pump.cev[0], Type::Open) && Pump::find(pump.sev, Type::Open);
        }));
        conn = Pump::find(pump.sev, Type::Open)->conn;
    }
};

Bytes pattern(size_t n, uint32_t seed) {
    Bytes b(n);
    uint32_t x = seed * 2654435761u + 1;
    for (size_t i = 0; i < n; ++i) {
        x ^= x << 13; x ^= x >> 17; x ^= x << 5;
        b[i] = uint8_t(x);
    }
    return b;
}

// A masked client frame built by hand (the raw peer's own frames).
std::string client_frame(uint8_t first, const std::string& payload, bool mask = true) {
    std::string f;
    f.push_back((char)first);
    const uint8_t mbit = mask ? 0x80 : 0;
    if (payload.size() < 126) {
        f.push_back((char)(mbit | payload.size()));
    } else {
        f.push_back((char)(mbit | 126));
        f.push_back((char)(payload.size() >> 8));
        f.push_back((char)(payload.size() & 0xff));
    }
    const uint8_t key[4] = {0x12, 0x34, 0x56, 0x78};
    if (mask) f.append((const char*)key, 4);
    for (size_t i = 0; i < payload.size(); ++i) f.push_back((char)(payload[i] ^ (mask ? key[i & 3] : 0)));
    return f;
}

std::string raw_upgrade(uint16_t port, const std::string& extra = "") {
    return "GET /v3/stream?token=t HTTP/1.1\r\nHost: 127.0.0.1:" + std::to_string(port) +
           "\r\nUpgrade: websocket\r\nConnection: Upgrade\r\nSec-WebSocket-Key: dGhlIHNhbXBsZSBub25jZQ==\r\n"
           "Sec-WebSocket-Version: 13\r\n" + extra + "\r\n";
}

}  // namespace

TEST_CASE("net/ws: SHA-1, base64 and the RFC 6455 accept key") {
    uint8_t d[20];
    sha1("abc", 3, d);
    CHECK(base64_encode(d, 20) == "qZk+NkcGgWq6PiVxeFDCbJzQ2J0=");
    sha1("", 0, d);
    CHECK(base64_encode(d, 20) == "2jmj7l5rSw0yVb/vlWAYkK/YBwk=");
    const std::string long_in(1000, 'a');   // more than one block, and a two-block tail
    sha1(long_in.data(), long_in.size(), d);
    CHECK(base64_encode(d, 20) == "KR6abGaZSUm1e6XmUDYemPw2sbo=");
    // RFC 6455 section 1.3: the sample key and its accept value.
    const std::string k = std::string("dGhlIHNhbXBsZSBub25jZQ==") + "258EAFA5-E914-47DA-95CA-C5AB0DC85B11";
    sha1(k.data(), k.size(), d);
    CHECK(base64_encode(d, 20) == "s3pPLMBiTxaQ9kYGzzhZRbK+xOo=");
    std::string out;
    CHECK(base64_decode("aGVsbG8=", &out));
    CHECK(out == "hello");
    CHECK(base64_decode("", &out));
    CHECK(out.empty());
    CHECK_FALSE(base64_decode("aGVsbG8", &out));     // length
    CHECK_FALSE(base64_decode("aGV=bG8=", &out));    // '=' inside
    CHECK_FALSE(base64_decode("aGVs*G8=", &out));    // alphabet
}

TEST_CASE("net/ws: URLs and query parameters") {
    WsUrl u;
    std::string err;
    REQUIRE(parse_ws_url("ws://127.0.0.1:8765/v3/stream?token=abc%2Bd&x=1", &u, &err));
    CHECK(u.host == "127.0.0.1");
    CHECK(u.port == 8765);
    CHECK(u.path == "/v3/stream");
    CHECK(u.query == "token=abc%2Bd&x=1");
    CHECK(u.target == "/v3/stream?token=abc%2Bd&x=1");
    REQUIRE(parse_ws_url("WS://LocalHost:1/", &u, &err));
    CHECK(u.host == "localhost");
    REQUIRE(parse_ws_url("ws://127.0.0.1:9?token=t", &u, &err));
    CHECK(u.target == "/?token=t");
    CHECK_FALSE(parse_ws_url("http://127.0.0.1:9/", &u, &err));
    CHECK_FALSE(parse_ws_url("ws://127.0.0.1:0/", &u, &err));
    CHECK_FALSE(parse_ws_url("ws://127.0.0.1:70000/", &u, &err));
    CHECK_FALSE(parse_ws_url("ws:///x", &u, &err));
    CHECK_FALSE(parse_ws_url("ws://[::1]:9/", &u, &err));
    // An error shows a link without its query or fragment, where a stream
    // link carries the session token.
    const std::string secret = "secret-token-0123456789abcdef";
    for (const std::string bad : {"http://127.0.0.1:9/v3/stream?token=", "ws:///v3/stream?token=",
                                  "ws://127.0.0.1:0/v3/stream?token=", "ws://127.0.0.1:70000/v3/stream?token=",
                                  "ws://[::1]:9/v3/stream?token=", "ws://u@127.0.0.1:9/v3/stream?token=",
                                  "ws://127.0.0.1:x/v3/stream#token="}) {
        CAPTURE(bad);
        err.clear();
        CHECK_FALSE(parse_ws_url(bad + secret, &u, &err));
        CHECK_FALSE(err.empty());
        CHECK(err.find(secret) == std::string::npos);
        CHECK(err.find(bad.substr(0, bad.find_first_of("?#"))) != std::string::npos);
    }

    HttpRequest r;
    r.query = "token=abc%2Bd&flag&name=a+b";
    std::string v;
    REQUIRE(r.query_param("token", &v));
    CHECK(v == "abc+d");
    REQUIRE(r.query_param("flag", &v));
    CHECK(v.empty());
    REQUIRE(r.query_param("name", &v));
    CHECK(v == "a b");
    CHECK_FALSE(r.query_param("missing", &v));
    CHECK(url_encode("a b/c~") == "a%20b%2Fc~");
}

TEST_CASE("net/ws: handshake, request fields, messages both ways, close codes") {
    Server server;
    std::string err;
    REQUIRE(server.listen(0, &err));
    REQUIRE(server.port() != 0);
    Client client;
    REQUIRE(client.connect(url(server.port()), &err, "http://127.0.0.1:" + std::to_string(server.port())));
    Pump p(&server, {&client});
    REQUIRE(p.until([&] { return Pump::find(p.sev, Type::Request) != nullptr; }));
    const Event req = *Pump::find(p.sev, Type::Request);
    CHECK(req.request.websocket);
    CHECK(req.request.method == "GET");
    CHECK(req.request.path == "/v3/stream");
    CHECK(req.request.host == "127.0.0.1:" + std::to_string(server.port()));
    CHECK(req.request.has_origin);
    CHECK(req.request.origin == "http://127.0.0.1:" + std::to_string(server.port()));
    std::string token;
    REQUIRE(req.request.query_param("token", &token));
    CHECK(token == "abcdefghijklmnopqrstuv");
    CHECK(req.request.header("sec-websocket-version") == "13");
    CHECK(server.request(req.conn) != nullptr);
    CHECK(client.state() == Client::State::Handshake);

    REQUIRE(server.accept(req.conn));
    REQUIRE(p.until([&] { return Pump::find(p.cev[0], Type::Open) && Pump::find(p.sev, Type::Open); }));
    CHECK(client.state() == Client::State::Open);
    CHECK(server.is_open(req.conn));

    REQUIRE(client.send("hello", 5));
    REQUIRE(client.send("text", 4, /*binary=*/false));
    REQUIRE(server.send(req.conn, "world", 5));
    REQUIRE(p.until([&] { return Pump::count(p.sev, Type::Message) == 2 && Pump::find(p.cev[0], Type::Message); }));
    std::vector<const Event*> got;
    for (const auto& e : p.sev)
        if (e.type == Type::Message) got.push_back(&e);
    CHECK(std::string(got[0]->data.begin(), got[0]->data.end()) == "hello");
    CHECK(got[0]->binary);
    CHECK(std::string(got[1]->data.begin(), got[1]->data.end()) == "text");
    CHECK_FALSE(got[1]->binary);   // a text message reaches the caller (the protocol closes with 1003)
    const Event* cm = Pump::find(p.cev[0], Type::Message);
    CHECK(std::string(cm->data.begin(), cm->data.end()) == "world");
    // Every queue drained: the server said so for its connection.
    CHECK(Pump::find(p.sev, Type::Writable, req.conn) != nullptr);
    CHECK(server.queued_bytes(req.conn) == 0);

    // The client closes with a protocol code; both ends report it.
    client.close(4004, "idle");
    REQUIRE(p.until([&] { return Pump::find(p.sev, Type::Closed) && Pump::find(p.cev[0], Type::Closed); }));
    const Event* sc = Pump::find(p.sev, Type::Closed);
    CHECK(sc->code == 4004);
    CHECK(sc->reason == "idle");
    CHECK_FALSE(sc->initiated_here);
    const Event* cc = Pump::find(p.cev[0], Type::Closed);
    CHECK(cc->code == 4004);
    CHECK(cc->initiated_here);
    CHECK(client.state() == Client::State::Closed);
    CHECK(server.connections() == 0);
}

TEST_CASE("net/ws: the server closes with a code; ping and pong; shutdown says going away") {
    Pair a;
    REQUIRE(a.client.ping("c1"));
    REQUIRE(a.server.ping(a.conn, "s1"));
    REQUIRE(a.pump.until([&] {
        return Pump::find(a.pump.sev, Type::Pong) && Pump::find(a.pump.cev[0], Type::Pong);
    }));
    // Each side's Pong echoes its own ping's payload.
    const Event* sp = Pump::find(a.pump.sev, Type::Pong);
    CHECK(std::string(sp->data.begin(), sp->data.end()) == "s1");
    const Event* cp = Pump::find(a.pump.cev[0], Type::Pong);
    CHECK(std::string(cp->data.begin(), cp->data.end()) == "c1");
    CHECK(Pump::count(a.pump.sev, Type::Message) == 0);   // pings are answered, not delivered
    CHECK(a.client.last_input_ms() > 0);

    a.server.close(a.conn, 4002, "role refused");
    REQUIRE(a.pump.until([&] { return Pump::find(a.pump.cev[0], Type::Closed) != nullptr; }));
    CHECK(Pump::find(a.pump.cev[0], Type::Closed)->code == 4002);
    CHECK(Pump::find(a.pump.cev[0], Type::Closed)->reason == "role refused");
    REQUIRE(a.pump.until([&] { return Pump::find(a.pump.sev, Type::Closed) != nullptr; }));
    CHECK(Pump::find(a.pump.sev, Type::Closed)->initiated_here);

    Pair b;
    std::vector<Event> fin = b.server.shutdown(kCloseGoingAway, 1000);
    CHECK_FALSE(b.server.listening());
    CHECK(b.server.connections() == 0);
    REQUIRE(b.pump.until([&] { return Pump::find(b.pump.cev[0], Type::Closed) != nullptr; }));
    CHECK(Pump::find(b.pump.cev[0], Type::Closed)->code == kCloseGoingAway);
    bool server_closed = false;
    for (const auto& e : fin) server_closed |= e.type == Type::Closed;
    CHECK(server_closed);
}

TEST_CASE("net/ws: refusals reach the client as an HTTP status (token, Host, Origin)") {
    // The rules belong to the caller (the hub, caliper.serve()); this is what
    // the layer gives it to apply them, and what a refused client sees.
    Server server;
    std::string err;
    REQUIRE(server.listen(0, &err));
    const std::string port = std::to_string(server.port());
    auto policy = [&](const HttpRequest& r) {
        std::string token;
        if (!r.query_param("token", &token) || token != "abcdefghijklmnopqrstuv") return 403;
        if (r.host != "127.0.0.1:" + port && r.host != "localhost:" + port) return 403;
        if (r.has_origin && r.origin != "http://127.0.0.1:" + port) return 403;
        return 0;
    };
    struct Case {
        std::string url, origin;
        std::vector<std::pair<std::string, std::string>> extra;
        int status;
    };
    const std::vector<Case> cases = {
        {url(server.port(), "/v3/stream?token=wrong"), "", {}, 403},
        {url(server.port(), "/v3/stream"), "", {}, 403},
        {url(server.port()), "", {{"Host", "evil.example:" + port}}, 403},
        {url(server.port()), "http://evil.example", {}, 403},
        {url(server.port()), "http://127.0.0.1:" + port, {}, 0},
        {"ws://localhost:" + port + "/v3/stream?token=abcdefghijklmnopqrstuv", "", {}, 0},
    };
    for (const Case& c : cases) {
        CAPTURE(c.url);
        CAPTURE(c.origin);
        Client client;
        REQUIRE(client.connect(c.url, &err, c.origin, c.extra));
        Pump p(&server, {&client});
        size_t handled = 0;
        REQUIRE(p.until([&] {
            for (; handled < p.sev.size(); ++handled) {
                const Event& e = p.sev[handled];
                if (e.type != Type::Request) continue;
                const int st = policy(e.request);
                if (st) server.reject(e.conn, st);
                else server.accept(e.conn);
            }
            return Pump::find(p.cev[0], Type::Closed) || Pump::find(p.cev[0], Type::Open);
        }));
        if (c.status) {
            const Event* e = Pump::find(p.cev[0], Type::Closed);
            REQUIRE(e != nullptr);
            CHECK(e->http_status == c.status);
            CHECK(e->code == kCloseAbnormal);
        } else {
            CHECK(Pump::find(p.cev[0], Type::Open) != nullptr);
            client.close();
            REQUIRE(p.until([&] { return Pump::find(p.cev[0], Type::Closed) != nullptr; }));
        }
        p.until([&] { return server.connections() == 0; }, 1000);
    }
}

TEST_CASE("net/ws: large and fragmented messages, both directions") {
    Options frag;
    frag.send_fragment_bytes = 1000;
    Pair a(frag, frag);
    const Bytes big = pattern(9u << 20, 1);       // 9 MiB: the 64-bit length form
    const Bytes mid = pattern(70000, 2);          // the 16-bit length form, 70 fragments
    const Bytes tiny = pattern(125, 3);
    const Bytes empty;
    REQUIRE(a.client.send(big.data(), big.size()));
    REQUIRE(a.client.send(mid.data(), mid.size()));
    REQUIRE(a.client.send(tiny.data(), tiny.size()));
    REQUIRE(a.client.send(empty.data(), 0));
    REQUIRE(a.server.send(a.conn, big.data(), big.size()));
    REQUIRE(a.server.send(a.conn, mid.data(), mid.size()));
    REQUIRE(a.pump.until([&] {
        return Pump::count(a.pump.sev, Type::Message) == 4 && Pump::count(a.pump.cev[0], Type::Message) == 2;
    }, 20000));
    std::vector<const Event*> s, c;
    for (const auto& e : a.pump.sev)
        if (e.type == Type::Message) s.push_back(&e);
    for (const auto& e : a.pump.cev[0])
        if (e.type == Type::Message) c.push_back(&e);
    CHECK(s[0]->data == big);
    CHECK(s[1]->data == mid);
    CHECK(s[2]->data == tiny);
    CHECK(s[3]->data.empty());
    CHECK(c[0]->data == big);
    CHECK(c[1]->data == mid);
}

TEST_CASE("net/ws: frames that break RFC 6455 close with 1002, an oversized message with 1009") {
    Options small;
    small.max_message_bytes = 1000;
    Server server(small);
    std::string err;
    REQUIRE(server.listen(0, &err));
    Pump p(&server);
    size_t handled = 0;

    auto open_raw = [&](Raw& r) {
        r.send_all(raw_upgrade(server.port()));
        std::string resp;
        p.until([&] {
            accept_all(server, p, &handled);
            resp += r.recv_some();
            return resp.find("\r\n\r\n") != std::string::npos;
        });
        CHECK(resp.rfind("HTTP/1.1 101", 0) == 0);
        CHECK(resp.find("Sec-WebSocket-Accept: s3pPLMBiTxaQ9kYGzzhZRbK+xOo=") != std::string::npos);
        CHECK(resp.find("Sec-WebSocket-Extensions") == std::string::npos);
    };
    // The server's close frame as the raw peer reads it: 0x88, length, code.
    auto close_code_seen = [&](Raw& r) {
        std::string got;
        p.until([&] {
            got += r.recv_some();
            return got.size() >= 4;
        }, 3000);
        if (got.size() < 4 || (uint8_t)got[0] != 0x88) return -1;
        return ((uint8_t)got[2] << 8) | (uint8_t)got[3];
    };
    struct Case {
        const char* what;
        std::string frame;
        int code;
    };
    const std::vector<Case> cases = {
        {"unmasked client frame", client_frame(0x82, "abc", false), 1002},
        {"reserved bit", client_frame(0x82 | 0x40, "abc"), 1002},
        {"unknown opcode", client_frame(0x83, "abc"), 1002},
        {"continuation without a message", client_frame(0x80, "abc"), 1002},
        {"fragmented ping", client_frame(0x09, "p"), 1002},
        {"one-byte close", client_frame(0x88, "x"), 1002},
        {"invalid close code", client_frame(0x88, std::string("\x03\xe7", 2)), 1002},   // 999
        {"message over the limit", client_frame(0x82, std::string(1001, 'z')), 1009},
        {"new message inside a fragmented one", client_frame(0x02, "ab") + client_frame(0x82, "cd"), 1002},
    };
    for (const Case& c : cases) {
        CAPTURE(c.what);
        Raw r(server.port());
        REQUIRE(r.fd != RAW_BAD);
        open_raw(r);
        p.sev.clear();
        handled = 0;
        r.send_all(c.frame);
        CHECK(close_code_seen(r) == c.code);
        REQUIRE(p.until([&] { return Pump::find(p.sev, Type::Closed) != nullptr; }));
        CHECK(Pump::find(p.sev, Type::Closed)->code == c.code);
        p.sev.clear();
        handled = 0;
    }

    // A fragmented message whose parts pass the limit only together.
    {
        Raw r(server.port());
        open_raw(r);
        p.sev.clear();
        handled = 0;
        r.send_all(client_frame(0x02, std::string(600, 'a')) + client_frame(0x80, std::string(600, 'b')));
        CHECK(close_code_seen(r) == 1009);
    }
}

TEST_CASE("net/ws: plain HTTP is answered with a body; bad and oversized heads are refused") {
    Server server;
    std::string err;
    REQUIRE(server.listen(0, &err));
    Pump p(&server);
    size_t handled = 0;
    const std::string page = "<!doctype html><title>viewer</title>";

    auto exchange = [&](const std::string& request) {
        Raw r(server.port());
        r.send_all(request);
        std::string resp;
        bool eof = false;
        p.until([&] {
            for (; handled < p.sev.size(); ++handled) {
                const Event& e = p.sev[handled];
                if (e.type != Type::Request) continue;
                if (e.request.websocket) server.reject(e.conn, 400);
                else if (!e.request.header("upgrade").empty()) server.reject(e.conn, 426);
                else if (e.request.path == "/") server.respond(e.conn, 200, "text/html; charset=utf-8", page);
                else server.reject(e.conn, 404);
            }
            bool now_eof = false;
            resp += r.recv_some(&now_eof);
            eof |= now_eof;
            return eof;
        }, 3000);
        return resp;
    };
    const std::string port = std::to_string(server.port());
    std::string resp = exchange("GET /?token=t HTTP/1.1\r\nHost: 127.0.0.1:" + port + "\r\n\r\n");
    CHECK(resp.rfind("HTTP/1.1 200 OK\r\n", 0) == 0);
    CHECK(resp.find("Content-Length: " + std::to_string(page.size())) != std::string::npos);
    CHECK(resp.find("Content-Type: text/html; charset=utf-8") != std::string::npos);
    CHECK(resp.substr(resp.size() - page.size()) == page);

    resp = exchange("HEAD / HTTP/1.1\r\nHost: 127.0.0.1:" + port + "\r\n\r\n");
    CHECK(resp.rfind("HTTP/1.1 200 OK\r\n", 0) == 0);
    CHECK(resp.find(page) == std::string::npos);   // no body for HEAD

    resp = exchange("GET /missing HTTP/1.1\r\nHost: 127.0.0.1:" + port + "\r\n\r\n");
    CHECK(resp.rfind("HTTP/1.1 404 Not Found\r\n", 0) == 0);

    // An upgrade with the wrong version is reported as a plain request (this
    // handler answers it with 426).
    resp = exchange("GET /v3/stream HTTP/1.1\r\nHost: x\r\nUpgrade: websocket\r\nConnection: Upgrade\r\n"
                    "Sec-WebSocket-Key: dGhlIHNhbXBsZSBub25jZQ==\r\nSec-WebSocket-Version: 8\r\n\r\n");
    CHECK(resp.rfind("HTTP/1.1 426 Upgrade Required", 0) == 0);

    resp = exchange("NOT A REQUEST\r\n\r\n");
    CHECK(resp.rfind("HTTP/1.1 400", 0) == 0);
    resp = exchange("GET / HTTP/1.1\r\nX-Big: " + std::string(20000, 'x') + "\r\n\r\n");
    CHECK(resp.rfind("HTTP/1.1 431", 0) == 0);
    p.until([&] { return server.connections() == 0; }, 1000);
    CHECK(server.connections() == 0);
}

TEST_CASE("net/ws: many connections at once") {
    Server server;
    std::string err;
    REQUIRE(server.listen(0, &err));
    constexpr int kClients = 64;
    std::vector<std::unique_ptr<Client>> clients;
    Pump p(&server);
    for (int i = 0; i < kClients; ++i) {
        clients.push_back(std::make_unique<Client>());
        REQUIRE(clients.back()->connect(url(server.port()), &err));
        p.add(clients.back().get());
    }
    size_t handled = 0;
    REQUIRE(p.until([&] {
        accept_all(server, p, &handled);
        for (int i = 0; i < kClients; ++i)
            if (!Pump::find(p.cev[(size_t)i], Type::Open)) return false;
        return true;
    }));
    CHECK(server.connections() == (size_t)kClients);
    CHECK(server.connection_ids().size() == (size_t)kClients);
    for (int i = 0; i < kClients; ++i) {
        const std::string m = "client " + std::to_string(i);
        REQUIRE(clients[(size_t)i]->send(m.data(), m.size()));
    }
    // Echo every message back on its own connection.
    size_t echoed = 0;
    REQUIRE(p.until([&] {
        for (; echoed < p.sev.size(); ++echoed) {
            const Event& e = p.sev[echoed];
            if (e.type == Type::Message) server.send(e.conn, e.data.data(), e.data.size());
        }
        for (int i = 0; i < kClients; ++i)
            if (!Pump::find(p.cev[(size_t)i], Type::Message)) return false;
        return true;
    }));
    for (int i = 0; i < kClients; ++i) {
        const Event* e = Pump::find(p.cev[(size_t)i], Type::Message);
        CHECK(std::string(e->data.begin(), e->data.end()) == "client " + std::to_string(i));
    }
    // One shared buffer sent to every connection is stored once.
    auto shared = std::make_shared<const Bytes>(pattern(4096, 9));
    for (ConnId c : server.connection_ids()) REQUIRE(server.send(c, shared));
    REQUIRE(p.until([&] {
        for (int i = 0; i < kClients; ++i)
            if (Pump::count(p.cev[(size_t)i], Type::Message) < 2) return false;
        return true;
    }));
    for (auto& c : clients) c->close();
    REQUIRE(p.until([&] { return server.connections() == 0; }));
}

TEST_CASE("net/ws: a slow reader never blocks the sender; back-pressure is visible") {
    Server server;
    std::string err;
    REQUIRE(server.listen(0, &err));
    Client slow, fast;
    REQUIRE(slow.connect(url(server.port()), &err));
    REQUIRE(fast.connect(url(server.port()), &err));
    Pump p(&server, {&slow, &fast});
    size_t handled = 0;
    REQUIRE(p.until([&] {
        accept_all(server, p, &handled);
        return Pump::find(p.cev[0], Type::Open) && Pump::find(p.cev[1], Type::Open);
    }));
    // Which server connection is which: the order of the Open events follows accept.
    std::vector<ConnId> ids = server.connection_ids();
    REQUIRE(ids.size() == 2);
    // From here the slow client is never polled.
    p.clients[0] = nullptr;
    // Find the slow one: send a probe to both and see which the fast client gets.
    ConnId slow_id = 0, fast_id = 0;
    for (ConnId c : ids) {
        const std::string probe = "probe " + std::to_string(c);
        server.send(c, probe.data(), probe.size());
    }
    REQUIRE(p.until([&] { return Pump::find(p.cev[1], Type::Message) != nullptr; }));
    const Event* fm = Pump::find(p.cev[1], Type::Message);
    fast_id = (ConnId)std::stoull(std::string(fm->data.begin() + 6, fm->data.end()));
    slow_id = ids[0] == fast_id ? ids[1] : ids[0];

    // 64 MiB to each: the sends and polls return at once although the slow
    // client reads nothing (the kernel buffers fill, the queue holds the rest).
    const auto chunk = std::make_shared<const Bytes>(pattern(256 * 1024, 4));
    const auto t0 = std::chrono::steady_clock::now();
    size_t max_queued = 0;
    for (int i = 0; i < 256; ++i) {
        REQUIRE(server.send(slow_id, chunk));
        REQUIRE(server.send(fast_id, chunk));
        p.once(0);
        max_queued = std::max(max_queued, server.queued_bytes(slow_id));
    }
    const double send_ms =
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    MESSAGE("queued 256 x 256 KiB to a silent and a reading client in " << send_ms << " ms; the silent one's "
            "queue peaked at " << (max_queued >> 20) << " MiB");
    CHECK(send_ms < 5000);
    CHECK(max_queued > (16u << 20));                      // most of it waits in the queue
    CHECK(server.queued_messages(slow_id) > 0);
    // The fast client gets everything meanwhile.
    REQUIRE(p.until([&] { return Pump::count(p.cev[1], Type::Message) == 257; }, 20000));
    CHECK(server.queued_bytes(fast_id) == 0);
    CHECK(server.queued_bytes(slow_id) > 0);

    // A sender over its limit is refused, not blocked.
    Options tight;
    tight.max_queued_bytes = 1 << 20;
    Pair t(tight);
    p.clients[0] = nullptr;
    size_t accepted = 0;
    for (int i = 0; i < 64; ++i) accepted += t.server.send(t.conn, chunk);   // the client never reads
    CHECK(accepted < 64);
    CHECK(t.server.queued_bytes(t.conn) <= tight.max_queued_bytes);

    // The slow client finally reads: everything arrives, in order, and the
    // server reports the drained queue.
    p.clients[0] = &slow;
    REQUIRE(p.until([&] {
        return Pump::count(p.cev[0], Type::Message) == 257 && server.queued_bytes(slow_id) == 0;
    }, 30000));
    size_t k = 0;
    for (const auto& e : p.cev[0]) {
        if (e.type != Type::Message) continue;
        if (k++ == 0) continue;   // the probe
        if (e.data != *chunk) {
            FAIL("message " << k << " differs");
            break;
        }
    }
    CHECK(Pump::find(p.sev, Type::Writable, slow_id) != nullptr);
}

// A raw peer upgraded on `s`, and the server's id for it.
static ConnId upgrade_raw(Server& s, Pump& p, Raw& r) {
    REQUIRE(r.fd != RAW_BAD);
    r.send_all(raw_upgrade(s.port()));
    size_t handled = 0;
    REQUIRE(p.until([&] {
        accept_all(s, p, &handled);
        return Pump::find(p.sev, Type::Open) != nullptr;
    }));
    return Pump::find(p.sev, Type::Open)->conn;
}

TEST_CASE("net/ws: a peer that sends pings and never reads keeps at most one pong queued") {
    Server s;
    std::string err;
    REQUIRE(s.listen(0, &err));
    Pump p{&s};
    Raw r(s.port());
    const ConnId c = upgrade_raw(s, p, r);
    // 40,000 pings of 125 bytes, about 5 MiB of pongs if each were queued,
    // then one binary message: once the server reports it, it has read every
    // ping before it.
    const std::string ping = client_frame(0x89, std::string(125, 'p'));
    std::string flood;
    for (int i = 0; i < 40000; ++i) flood += ping;
    flood += client_frame(0x82, "end");
    std::thread writer([&] { r.send_all(flood); });
    size_t peak = 0;
    const bool read_all = p.until([&] {
        peak = std::max(peak, s.queued_bytes(c));
        return Pump::find(p.sev, Type::Message, c) != nullptr;
    }, 60000);
    writer.join();
    MESSAGE("after 40,000 pings the server's queue holds " << s.queued_bytes(c) << " bytes; at most " << peak);
    REQUIRE(read_all);
    // A pong partly written to the full socket, and the newest one behind it.
    CHECK(s.queued_bytes(c) <= 2 * (2 + 125));
    CHECK(peak <= 2 * (2 + 125));
    CHECK(Pump::find(p.sev, Type::Closed, c) == nullptr);
}

TEST_CASE("net/ws: a ping that would take the send queue past its limit closes the connection with 1008") {
    Options so;
    so.max_queued_bytes = size_t(64) << 10;
    so.close_timeout_ms = 200;
    Server s(so);
    std::string err;
    REQUIRE(s.listen(0, &err));
    Pump p{&s};
    Raw r(s.port());
    const ConnId c = upgrade_raw(s, p, r);
    // The peer never reads: fill the socket, then the queue, to within a
    // few bytes of its limit.
    for (size_t n : {size_t(16) << 10, size_t(1) << 10, size_t(64), size_t(1)}) {
        auto b = std::make_shared<const Bytes>(n, uint8_t(7));
        while (s.send(c, b)) p.once();
    }
    CHECK(s.queued_bytes(c) + 15 > so.max_queued_bytes);
    r.send_all(client_frame(0x89, std::string(125, 'x')));
    REQUIRE(p.until([&] { return Pump::find(p.sev, Type::Closed, c) != nullptr; }, 10000));
    const Event* e = Pump::find(p.sev, Type::Closed, c);
    CHECK(e->code == kClosePolicy);
    CHECK(e->reason == "send queue full");
    CHECK(e->initiated_here);
}

TEST_CASE("net/ws: connect failures, header timeout, wake from another thread") {
    // Nothing listens on a port the server just gave back.
    uint16_t dead_port = 0;
    {
        Server s;
        std::string err;
        REQUIRE(s.listen(0, &err));
        dead_port = s.port();
    }
    Client c;
    std::string err;
    REQUIRE(c.connect(url(dead_port), &err));
    Pump p(nullptr, {&c});
    REQUIRE(p.until([&] { return Pump::find(p.cev[0], Type::Closed) != nullptr; }));
    const Event* e = Pump::find(p.cev[0], Type::Closed);
    CHECK(e->code == kCloseAbnormal);
    CHECK(e->http_status == 0);
    CHECK(e->reason.find("connect") != std::string::npos);

    CHECK_FALSE(c.connect("ws://example.com:80/", &err));   // no name resolution
    CHECK(err.find("numeric") != std::string::npos);
    CHECK_FALSE(c.connect("http://127.0.0.1:1/", &err));

    // A client that never finishes its request head is dropped.
    Options quick;
    quick.header_timeout_ms = 100;
    Server s(quick);
    REQUIRE(s.listen(0, &err));
    Raw r(s.port());
    r.send_all("GET / HTTP/1.1\r\n");
    Pump q(&s);
    bool eof = false;
    q.until([&] {
        bool now = false;
        r.recv_some(&now);
        eof |= now;
        return eof;
    }, 3000);
    CHECK(eof);
    CHECK(q.sev.empty());   // never reported: no complete request

    // wake() from another thread ends a long wait at once.
    std::atomic<bool> go{false};
    std::thread t([&] {
        while (!go) std::this_thread::yield();
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
        s.wake();
    });
    go = true;
    const auto t0 = std::chrono::steady_clock::now();
    s.poll(5000);
    const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    t.join();
    CHECK(ms < 2000);
}

TEST_CASE("net/ws: a message sent as a prefix and a shared body arrives whole") {
    Pair a;
    auto body = std::make_shared<const Bytes>(pattern(300000, 3));
    Bytes prefix = pattern(37, 4);
    for (const auto& r : std::vector<std::pair<size_t, size_t>>{{0, 300000}, {1000, 70000}, {5, 100}, {0, 0}})
        REQUIRE(a.server.send(a.conn, prefix, body, r.first, r.second));
    CHECK_FALSE(a.server.send(a.conn, prefix, body, 299999, 2));   // past the body's end
    REQUIRE(a.pump.until([&] { return Pump::count(a.pump.cev[0], Type::Message) == 4; }));
    std::vector<const Event*> got;
    for (const auto& e : a.pump.cev[0])
        if (e.type == Type::Message) got.push_back(&e);
    size_t i = 0;
    for (const auto& r : std::vector<std::pair<size_t, size_t>>{{0, 300000}, {1000, 70000}, {5, 100}, {0, 0}}) {
        Bytes want = prefix;
        want.insert(want.end(), body->begin() + (std::ptrdiff_t)r.first, body->begin() + (std::ptrdiff_t)(r.first + r.second));
        CHECK(got[i]->binary);
        CHECK(got[i]->data == want);
        ++i;
    }
}


TEST_CASE("net/ws: a request body is read only when body_limit allows it, within its time limit") {
    Options o;
    o.body_timeout_ms = 300;
    // Bodies up to 64 bytes on /body; none anywhere else.
    o.body_limit = [](const HttpRequest& r) { return r.path == "/body" ? size_t(64) : size_t(0); };
    Server server(o);
    std::string err;
    REQUIRE(server.listen(0, &err));
    Pump p(&server);
    size_t handled = 0;
    std::vector<std::string> bodies;   // as the server reported them

    // Sends `parts` with a pause between them and returns the response.
    auto exchange = [&](const std::vector<std::string>& parts) {
        Raw r(server.port());
        std::string resp;
        bool eof = false;
        for (size_t i = 0; i < parts.size(); ++i) {
            r.send_all(parts[i]);
            if (i + 1 < parts.size()) p.until([] { return false; }, 50);
        }
        p.until([&] {
            for (; handled < p.sev.size(); ++handled) {
                const Event& e = p.sev[handled];
                if (e.type != Type::Request) continue;
                bodies.push_back(e.request.body);
                server.respond(e.conn, 200, "text/plain", "ok");
            }
            bool now_eof = false;
            resp += r.recv_some(&now_eof);
            eof |= now_eof;
            return eof;
        }, 3000);
        return resp;
    };
    const std::string host = "Host: 127.0.0.1:" + std::to_string(server.port()) + "\r\n";
    // A body in two pieces, after the head, arrives whole; bytes beyond it are dropped.
    std::string resp = exchange({"POST /body HTTP/1.1\r\n" + host + "Content-Length: 11\r\n\r\n{\"a\":", "[1,2]}tail"});
    CHECK(resp.rfind("HTTP/1.1 200 OK\r\n", 0) == 0);
    REQUIRE(bodies.size() == 1);
    CHECK(bodies[0] == "{\"a\":[1,2]}");
    // Over the limit, another path, or chunked: reported at once, without a body.
    for (const std::string& head :
         {"POST /body HTTP/1.1\r\n" + host + "Content-Length: 65\r\n\r\n",
          "POST /other HTTP/1.1\r\n" + host + "Content-Length: 4\r\n\r\n",
          "POST /body HTTP/1.1\r\n" + host + "Content-Length: 4\r\nTransfer-Encoding: chunked\r\n\r\n",
          "POST /body HTTP/1.1\r\n" + host + "Content-Length: x4\r\n\r\n"}) {
        CAPTURE(head);
        const size_t before = bodies.size();
        resp = exchange({head});
        CHECK(resp.rfind("HTTP/1.1 200 OK\r\n", 0) == 0);
        REQUIRE(bodies.size() == before + 1);
        CHECK(bodies.back().empty());
    }
    // A body that does not arrive within body_timeout_ms gets 408, unreported.
    const size_t before = bodies.size();
    const auto t0 = std::chrono::steady_clock::now();
    resp = exchange({"POST /body HTTP/1.1\r\n" + host + "Content-Length: 10\r\n\r\n{\"a\""});
    const double waited = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    CHECK(resp.rfind("HTTP/1.1 408", 0) == 0);
    CHECK(bodies.size() == before);
    CHECK(waited < 2.0);
}

#ifndef _WIN32
namespace {

// A peer that ends with a reset while the client has a write queued: the
// client's next write fails with EPIPE or ECONNRESET, but what the peer sent
// before the reset still waits in the client's socket and must arrive before
// Closed. The model runner reads a script's last action reply this way when
// the script's process exits right after it (design/hosted-runs.md). The peer
// is a raw socket that answers the upgrade by hand; the test reads the state
// of the client's own socket (POSIX poll) only to wait for the reset.
struct ResetPeer {
    Client& c;
    Pump p;
    int sfd = -1;    // the peer's end
    int cfd = -1;    // the client's own socket, found by its port
    explicit ResetPeer(Client& client) : c(client), p(nullptr, {&client}) {}
    ~ResetPeer() {
        if (sfd >= 0) ::close(sfd);
    }
    // Connects the client to a raw listener and answers its upgrade.
    void open() {
        const int lfd = ::socket(AF_INET, SOCK_STREAM, 0);
        REQUIRE(lfd >= 0);
        sockaddr_in la{};
        la.sin_family = AF_INET;
        inet_pton(AF_INET, "127.0.0.1", &la.sin_addr);
        socklen_t len = sizeof la;
        REQUIRE(::bind(lfd, (sockaddr*)&la, sizeof la) == 0);
        REQUIRE(::listen(lfd, 1) == 0);
        REQUIRE(::getsockname(lfd, (sockaddr*)&la, &len) == 0);
        std::string err;
        REQUIRE(c.connect(url(ntohs(la.sin_port)), &err));
        sfd = ::accept(lfd, nullptr, nullptr);   // the system completes the TCP handshake
        ::close(lfd);
        REQUIRE(sfd >= 0);
        timeval tv{0, 20000};
        setsockopt(sfd, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof tv);
        std::string head;
        REQUIRE(p.until([&] {
            char buf[4096];
            const ssize_t r = ::recv(sfd, buf, sizeof buf, 0);
            if (r > 0) head.append(buf, (size_t)r);
            return head.find("\r\n\r\n") != std::string::npos;
        }));
        const std::string field = "Sec-WebSocket-Key: ";
        const size_t at = head.find(field);
        REQUIRE(at != std::string::npos);
        const size_t from = at + field.size();
        const std::string key = head.substr(from, head.find("\r\n", from) - from) + "258EAFA5-E914-47DA-95CA-C5AB0DC85B11";
        uint8_t digest[20];
        sha1(key.data(), key.size(), digest);
        const std::string upgrade = "HTTP/1.1 101 Switching Protocols\r\nUpgrade: websocket\r\nConnection: Upgrade\r\n"
                                    "Sec-WebSocket-Accept: " + base64_encode(digest, 20) + "\r\n\r\n";
        REQUIRE(::send(sfd, upgrade.data(), upgrade.size(), 0) == (ssize_t)upgrade.size());
        REQUIRE(p.until([&] { return Pump::find(p.cev[0], Type::Open) != nullptr; }));

        // The client writes a message that the peer never reads, so that the
        // peer's close is a reset.
        REQUIRE(c.send("unread", 6, true));
        REQUIRE(p.until([&] {
            char peek[16];
            return ::recv(sfd, peek, sizeof peek, MSG_PEEK) > 0;
        }));
        sockaddr_in remote{};
        len = sizeof remote;
        REQUIRE(::getpeername(sfd, (sockaddr*)&remote, &len) == 0);
        for (int fd = 0; fd < 4096 && cfd < 0; ++fd) {
            sockaddr_in mine{}, theirs{};
            socklen_t ml = sizeof mine, tl = sizeof theirs;
            if (fd == sfd || ::getsockname(fd, (sockaddr*)&mine, &ml) != 0 || mine.sin_family != AF_INET) continue;
            if (::getpeername(fd, (sockaddr*)&theirs, &tl) != 0) continue;
            if (mine.sin_port == remote.sin_port && theirs.sin_port == la.sin_port) cfd = fd;
        }
        REQUIRE(cfd >= 0);
    }
    // Sends `frames` as the peer's last bytes, then closes with unread input:
    // a reset. Returns when the client's socket took it (a hang-up), without
    // a read, so that the client's next write fails.
    void send_and_reset(const std::string& frames) {
        REQUIRE(::send(sfd, frames.data(), frames.size(), 0) == (ssize_t)frames.size());
        ::close(sfd);
        sfd = -1;
        bool reset = false;
        const auto until = std::chrono::steady_clock::now() + std::chrono::seconds(5);
        while (!reset && std::chrono::steady_clock::now() < until) {
            pollfd pf{cfd, POLLIN | POLLOUT, 0};
            reset = ::poll(&pf, 1, 0) == 1 && (pf.revents & POLLHUP);
            if (!reset) std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
        REQUIRE(reset);
    }
    // Queues a write, polls until Closed and returns the client's events.
    const std::vector<Event>& write_and_close() {
        REQUIRE(c.send("after the reset", 15, true));
        REQUIRE(p.until([&] { return Pump::find(p.cev[0], Type::Closed) != nullptr; }));
        return p.cev[0];
    }
};

// An unmasked binary frame of the server side.
std::string server_frame(const std::string& payload) {
    std::string f = "\x82";
    if (payload.size() < 126) {
        f.push_back((char)payload.size());
    } else {
        f.push_back((char)126);
        f.push_back((char)(payload.size() >> 8));
        f.push_back((char)(payload.size() & 0xff));
    }
    return f + payload;
}

}  // namespace

TEST_CASE("net/ws: a message sent just before the peer's reset arrives, also when a write fails first") {
    Client c;
    ResetPeer peer(c);
    peer.open();
    peer.send_and_reset(server_frame("reply"));
    const std::vector<Event>& got = peer.write_and_close();
    size_t message_at = got.size(), closed_at = got.size();
    for (size_t i = 0; i < got.size(); ++i) {
        if (message_at == got.size() && got[i].type == Type::Message &&
            std::string(got[i].data.begin(), got[i].data.end()) == "reply")
            message_at = i;
        if (closed_at == got.size() && got[i].type == Type::Closed) closed_at = i;
    }
    INFO("closed: " << Pump::find(got, Type::Closed)->reason);
    CHECK(message_at < closed_at);
}

TEST_CASE("net/ws: the read after a failed write takes no more than one poll's read budget") {
    Options o;
    o.read_budget_bytes = 16 * 1024;
    Client c(o);
    ResetPeer peer(c);
    peer.open();
    // Sixteen messages of 4 KiB, 64 KiB in all, sent to the client's socket.
    std::string frames;
    for (int i = 0; i < 16; ++i) frames += server_frame(std::string(4096, char('a' + i)));
    peer.send_and_reset(frames);
    int waiting = 0;
    REQUIRE(::ioctl(peer.cfd, FIONREAD, &waiting) == 0);
    REQUIRE((size_t)waiting > o.read_budget_bytes);   // more than the budget is there to read
    const std::vector<Event>& got = peer.write_and_close();
    size_t bytes = 0, messages = 0;
    for (const auto& e : got)
        if (e.type == Type::Message) {
            ++messages;
            bytes += e.data.size();
        }
    INFO("messages: " << messages << ", closed: " << Pump::find(got, Type::Closed)->reason);
    CHECK(messages >= 1);                     // what fits in the budget arrives
    CHECK(bytes <= o.read_budget_bytes);      // and nothing past it
    CHECK(messages < 16);
}
#endif
