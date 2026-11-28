// The loopback HTTP client against an in-process server that misbehaves:
// slow, oversized, garbage, a wrong Content-Length, a redirect. POSIX only.
#include <caliper_builder/mcp/loopback_http.h>

#include "fake_http.h"

#include <doctest/doctest.h>

#include <arpa/inet.h>
#include <atomic>
#include <chrono>
#include <netinet/in.h>
#include <sys/socket.h>
#include <thread>
#include <unistd.h>

using namespace caliper_builder::mcp;
using caliper_builder::mcp::test::FakeConnection;
using caliper_builder::mcp::test::FakeHttpServer;
using caliper_builder::mcp::test::FakeRequest;
using caliper_builder::mcp::test::http_response;

namespace {

HttpLimits quick() {
    HttpLimits l;
    l.connect_timeout_ms = 1000;
    l.read_timeout_ms = 1000;
    l.total_timeout_ms = 3000;
    return l;
}

HttpResponse get(FakeHttpServer& s, const HttpLimits& limits = quick(), const std::string& target = "/v0/info") {
    HttpRequest r;
    r.target = target;
    return loopback_http_request("127.0.0.1", s.port(), r, limits);
}

// A port on which nothing listens: bound, then closed.
int dead_port() {
    const int fd = ::socket(AF_INET, SOCK_STREAM, 0);
    sockaddr_in a{};
    a.sin_family = AF_INET;
    a.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    REQUIRE(::bind(fd, reinterpret_cast<sockaddr*>(&a), sizeof a) == 0);
    socklen_t len = sizeof a;
    getsockname(fd, reinterpret_cast<sockaddr*>(&a), &len);
    ::close(fd);
    return ntohs(a.sin_port);
}

}  // namespace

TEST_CASE("a well-formed exchange carries the request and the response") {
    FakeHttpServer s;
    REQUIRE(s.start([](const FakeRequest& req, FakeConnection& c) {
        c.write(http_response(200, "application/json", "{\"echo\":\"" + req.method + " " + req.target + "\"}",
                              {{"X-Extra", "  padded value  "}}));
    }));
    HttpRequest r;
    r.method = "POST";
    r.target = "/v0/commands/app.info";
    r.body = "{\"a\":1}";
    r.headers = {{"Authorization", "Bearer abcdefghijklmnopqrstuvwxyz012345"}, {"X-Caliper-Origin", "mcp"}};
    const HttpResponse resp = loopback_http_request("127.0.0.1", s.port(), r, quick());
    REQUIRE(resp.ok());
    CHECK(resp.status == 200);
    CHECK(resp.reason == "OK");
    CHECK(resp.body == "{\"echo\":\"POST /v0/commands/app.info\"}");
    CHECK(resp.header("x-extra") == "padded value");
    CHECK(resp.header("content-type") == "application/json");
    const auto reqs = s.requests();
    REQUIRE(reqs.size() == 1);
    CHECK(reqs[0].header("host") == "127.0.0.1:" + std::to_string(s.port()));
    CHECK(reqs[0].header("connection") == "close");
    CHECK(reqs[0].header("content-length") == "7");
    CHECK(reqs[0].header("x-caliper-origin") == "mcp");
    CHECK(reqs[0].body == "{\"a\":1}");

    // A POST without a body says so with Content-Length: 0; a GET sends none.
    s.clear_requests();
    HttpRequest post;
    post.method = "POST";
    post.target = "/v0/instruments/x/launch";
    CHECK(loopback_http_request("127.0.0.1", s.port(), post, quick()).ok());
    CHECK(get(s).ok());
    const auto two = s.requests();
    REQUIRE(two.size() == 2);
    CHECK(two[0].header("content-length") == "0");
    CHECK(two[1].header("content-length").empty());
}

TEST_CASE("a body without Content-Length runs to the close") {
    FakeHttpServer s;
    REQUIRE(s.start([](const FakeRequest&, FakeConnection& c) {
        c.write("HTTP/1.1 200 OK\r\nContent-Type: text/plain\r\n\r\nhello ");
        c.sleep_ms(50);
        c.write("world");
    }));
    const HttpResponse r = get(s);
    REQUIRE(r.ok());
    CHECK(r.body == "hello world");
}

TEST_CASE("only 127.0.0.1 is reached, and a request that could smuggle a header is refused") {
    HttpRequest r;
    for (const char* host : {"localhost", "127.0.0.2", "0.0.0.0", "10.0.0.1", "::1", "127.0.0.1 ", ""}) {
        const HttpResponse resp = loopback_http_request(host, 80, r, quick());
        CHECK_MESSAGE(resp.error.code == "http.refused_host", host);
    }
    CHECK(loopback_http_request("127.0.0.1", 0, r, quick()).error.code == "http.bad_request");
    CHECK(loopback_http_request("127.0.0.1", 70000, r, quick()).error.code == "http.bad_request");

    auto refused = [](HttpRequest bad) {
        const HttpResponse resp = loopback_http_request("127.0.0.1", 9, bad, quick());
        CHECK(resp.error.code == "http.bad_request");
        CHECK(resp.error.message.find("secret") == std::string::npos);
    };
    HttpRequest b;
    b.method = "GET /x HTTP/1.1\r\nX:";
    refused(b);
    b = {};
    b.method = "get";
    refused(b);
    b = {};
    b.target = "/v0/info HTTP/1.1\r\nX-Evil: 1\r\n";
    refused(b);
    b = {};
    b.target = "v0/info";
    refused(b);
    b = {};
    b.headers = {{"Authorization", "Bearer secret\r\nX-Evil: 1"}};
    refused(b);
    b = {};
    b.headers = {{"Bad Name", "secret"}};
    refused(b);
    b = {};
    b.headers = {{"Host", "evil.example"}};
    refused(b);
    b = {};
    b.headers = {{"content-length", "5"}};
    refused(b);
}

TEST_CASE("connection faults and time limits are reported") {
    HttpRequest r;
    const HttpResponse refused = loopback_http_request("127.0.0.1", dead_port(), r, quick());
    CHECK(refused.error.code == "http.connect_refused");

    FakeHttpServer slow;
    REQUIRE(slow.start([](const FakeRequest&, FakeConnection& c) { c.sleep_ms(3000); }));
    HttpLimits l = quick();
    l.read_timeout_ms = 300;
    const auto t0 = std::chrono::steady_clock::now();
    CHECK(get(slow, l).error.code == "http.read_timeout");
    CHECK(std::chrono::steady_clock::now() - t0 < std::chrono::milliseconds(2500));

    // A server that trickles bytes keeps the idle limit happy but not the total one.
    FakeHttpServer trickle;
    REQUIRE(trickle.start([](const FakeRequest&, FakeConnection& c) {
        c.write("HTTP/1.1 200 OK\r\nContent-Length: 1000\r\n\r\n");
        for (int i = 0; i < 100; ++i) {
            if (!c.write("x")) return;
            c.sleep_ms(50);
        }
    }));
    HttpLimits t = quick();
    t.read_timeout_ms = 500;
    t.total_timeout_ms = 800;
    CHECK(get(trickle, t).error.code == "http.timeout");

    // A cancel flag stops a wait early.
    std::atomic<bool> cancel{false};
    std::thread canceller([&] {
        std::this_thread::sleep_for(std::chrono::milliseconds(200));
        cancel = true;
    });
    HttpLimits long_wait = quick();
    long_wait.read_timeout_ms = long_wait.total_timeout_ms = 10000;
    const auto t1 = std::chrono::steady_clock::now();
    HttpRequest cr;
    CHECK(loopback_http_request("127.0.0.1", slow.port(), cr, long_wait, &cancel).error.code == "http.cancelled");
    CHECK(std::chrono::steady_clock::now() - t1 < std::chrono::milliseconds(2000));
    canceller.join();
}

TEST_CASE("oversized responses are refused before they are kept") {
    FakeHttpServer big;
    REQUIRE(big.start([](const FakeRequest& req, FakeConnection& c) {
        if (req.target == "/declared") {
            c.write("HTTP/1.1 200 OK\r\nContent-Length: 999999999\r\n\r\nabc");
        } else if (req.target == "/undeclared") {
            c.write("HTTP/1.1 200 OK\r\n\r\n");
            const std::string block(64 << 10, 'x');
            for (int i = 0; i < 64; ++i)
                if (!c.write(block)) return;
        } else if (req.target == "/head") {
            std::string head = "HTTP/1.1 200 OK\r\n";
            for (int i = 0; i < 2000; ++i) head += "X-Filler-" + std::to_string(i) + ": aaaaaaaaaa\r\n";
            c.write(head + "\r\n");
        } else if (req.target == "/endless-head") {
            c.write("HTTP/1.1 200 OK\r\nX: ");
            const std::string block(4096, 'y');
            for (int i = 0; i < 16; ++i)
                if (!c.write(block)) return;
        } else if (req.target == "/many-headers") {
            std::string head = "HTTP/1.1 200 OK\r\n";
            for (int i = 0; i < 150; ++i) head += "X" + std::to_string(i) + ": 1\r\n";
            c.write(head + "Content-Length: 0\r\n\r\n");
        }
    }));
    HttpLimits l = quick();
    l.max_body_bytes = 1 << 20;
    CHECK(get(big, l, "/declared").error.code == "http.too_large");
    CHECK(get(big, l, "/undeclared").error.code == "http.too_large");
    CHECK(get(big, l, "/head").error.code == "http.too_large");
    CHECK(get(big, l, "/endless-head").error.code == "http.too_large");
    CHECK(get(big, l, "/many-headers").error.code == "http.bad_response");
}

TEST_CASE("garbage, a wrong Content-Length, a transfer coding and a redirect are refused") {
    FakeHttpServer s;
    REQUIRE(s.start([](const FakeRequest& req, FakeConnection& c) {
        const std::string& t = req.target;
        if (t == "/garbage") c.write("SSH-2.0-OpenSSH_9.6\r\n\r\n");
        else if (t == "/binary") c.write(std::string("\x00\x01\x02\xff\r\n\r\n", 8));
        else if (t == "/no-head-end") c.write("HTTP/1.1 200 OK\r\nContent-Length: 2\r\n");
        else if (t == "/nothing") return;
        else if (t == "/short") c.write("HTTP/1.1 200 OK\r\nContent-Length: 10\r\n\r\nabc");
        else if (t == "/long") c.write("HTTP/1.1 200 OK\r\nContent-Length: 2\r\n\r\nabcdef");
        else if (t == "/two-lengths") c.write("HTTP/1.1 200 OK\r\nContent-Length: 2\r\nContent-Length: 3\r\n\r\nabc");
        else if (t == "/same-lengths") c.write("HTTP/1.1 200 OK\r\nContent-Length: 3\r\nContent-Length: 3\r\n\r\nabc");
        else if (t == "/bad-length") c.write("HTTP/1.1 200 OK\r\nContent-Length: -3\r\n\r\nabc");
        else if (t == "/chunked")
            c.write("HTTP/1.1 200 OK\r\nTransfer-Encoding: chunked\r\n\r\n3\r\nabc\r\n0\r\n\r\n");
        else if (t == "/redirect")
            c.write("HTTP/1.1 302 Found\r\nLocation: http://example.com/\r\nContent-Length: 0\r\n\r\n");
        else if (t == "/interim") c.write("HTTP/1.1 100 Continue\r\n\r\nHTTP/1.1 200 OK\r\nContent-Length: 0\r\n\r\n");
        else if (t == "/folded") c.write("HTTP/1.1 200 OK\r\nX-A: 1\r\n continued\r\nContent-Length: 0\r\n\r\n");
        else if (t == "/status") c.write("HTTP/1.1 2x0 OK\r\nContent-Length: 0\r\n\r\n");
        else if (t == "/control") c.write("HTTP/1.1 200 OK\r\nX-A: a\x01" "b\r\nContent-Length: 0\r\n\r\n");
        else if (t == "/no-colon") c.write("HTTP/1.1 200 OK\r\nJustAName\r\nContent-Length: 0\r\n\r\n");
        else if (t == "/http2") c.write("HTTP/2 200\r\ncontent-length: 0\r\n\r\n");
        else if (t == "/no-reason") c.write("HTTP/1.1 404\r\nContent-Length: 0\r\n\r\n");
    }));
    for (const char* t : {"/garbage", "/binary", "/no-head-end", "/nothing", "/short", "/long", "/two-lengths",
                          "/bad-length", "/chunked", "/interim", "/folded", "/status", "/control", "/no-colon",
                          "/http2"}) {
        const HttpResponse r = get(s, quick(), t);
        CHECK_MESSAGE(r.error.code == "http.bad_response", t, " gave ", r.error.code);
    }
    const HttpResponse same = get(s, quick(), "/same-lengths");
    CHECK(same.ok());
    CHECK(same.body == "abc");
    const HttpResponse redirect = get(s, quick(), "/redirect");
    CHECK(redirect.error.code == "http.redirect");
    CHECK(redirect.status == 302);
    CHECK(s.request_count() == 17);   // followed nowhere
    const HttpResponse no_reason = get(s, quick(), "/no-reason");
    CHECK(no_reason.ok());
    CHECK(no_reason.status == 404);
}

TEST_CASE("an early answer to an unread body is still read") {
    FakeHttpServer s;
    REQUIRE(s.start([](const FakeRequest&, FakeConnection& c) {
        c.write(http_response(413, "application/json", "{\"error\":\"too big\"}"));
    }));
    HttpRequest r;
    r.method = "POST";
    r.target = "/v0/commands/x.y";
    r.body = std::string(64 << 10, 'x');
    const HttpResponse resp = loopback_http_request("127.0.0.1", s.port(), r, quick());
    if (resp.ok()) {
        CHECK(resp.status == 413);
    } else {
        // The server closed with unread bytes, so the system may reset the
        // connection before the answer is read; that is reported, not hidden.
        CHECK(resp.error.code == "http.io_error");
    }
}
