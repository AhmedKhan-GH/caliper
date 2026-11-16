// Tests of the stand-in's HTTP server, in process: the two endpoints, every
// check of the request in its order with its status, hostile heads and
// bodies, the origins and the approval waits, the connection bound, and the
// API file and the token helpers.
#include <caliper_builder/session/command_server.h>
#include <caliper_builder/session/session.h>

#include <doctest/doctest.h>

#include <atomic>
#include <thread>

#include "helpers.h"

#ifndef _WIN32
#include <sys/stat.h>
#endif

using namespace caliper_builder;
using namespace session_test;
using nlohmann::json;

#ifndef _WIN32

namespace {

// A session, a drain loop on a thread of its own and a server.
struct Served {
    std::unique_ptr<Session> s;
    std::unique_ptr<CommandServer> server;
    std::atomic<bool> stop{false};
    std::thread drainer;
    std::string token = generate_session_token();
    int port = 0;

    explicit Served(const std::string& name, CommandServerOptions options = {}) {
        s = open_session(fresh_dir(name) + "/workspace");
        options.token = token;
        server = std::make_unique<CommandServer>(s->queue(), options);
        REQUIRE(server->start().ok());
        port = server->port();
        drainer = std::thread([this] {
            while (!stop.load()) {
                s->queue().drain();
                std::this_thread::sleep_for(std::chrono::milliseconds(2));
            }
        });
    }
    ~Served() {
        s->queue().shutdown();
        server->stop();
        stop.store(true);
        drainer.join();
    }

    HttpReply post(const std::string& name, const json& args, const std::vector<std::string>& extra = {}) {
        return http(port, "POST", "/v0/commands/" + name, token, args.dump(), extra);
    }
};

bool has_header(const HttpReply& r, const std::string& line) { return r.head.find(line) != std::string::npos; }

}  // namespace

TEST_CASE("GET /v0/commands gives the descriptor table and POST runs a command") {
    Served sv("server-basic");
    const HttpReply list = http(sv.port, "GET", "/v0/commands", sv.token);
    CHECK(list.status == 200);
    CHECK(list.json_body() == command_table_to_json());
    CHECK(has_header(list, "Cache-Control: no-store"));
    CHECK(has_header(list, "X-Content-Type-Options: nosniff"));
    CHECK(has_header(list, "Connection: close"));
    CHECK(list.body.find(sv.token) == std::string::npos);
    // A GET may carry the token in its query; a POST may not.
    CHECK(http(sv.port, "GET", "/v0/commands?token=" + sv.token, "").status == 200);
    CHECK(http(sv.port, "POST", "/v0/commands/spec.get?token=" + sv.token, "", "{}").status == 401);

    const HttpReply made = sv.post("spec.new", {{"id", "web"}}, {"X-Caliper-Origin: mcp", "X-Caliper-Group: turn-1"});
    CHECK(made.status == 200);
    const json env = made.json_body();
    CHECK(env["ok"] == true);
    CHECK(env["id"].get<std::string>().rfind("http-", 0) == 0);
    CHECK(env["result"]["id"] == "web");
    const HttpReply got = sv.post("spec.get", json::object());
    CHECK(got.json_body()["result"]["id"] == "web");
    // A failure is still HTTP 200, with the envelope.
    const HttpReply failed = sv.post("panel.remove", {{"id", "none"}});
    CHECK(failed.status == 200);
    CHECK(failed.json_body()["error"]["code"] == "panel.not_found");
    const HttpReply unknown = sv.post("panel.explode", json::object());
    CHECK(unknown.status == 200);
    CHECK(unknown.json_body()["error"]["code"] == "command.unknown");
    const HttpReply journal = sv.post("history.list", json::object());
    CHECK(journal.json_body()["result"]["undo"][0]["origin"] == "mcp");
    CHECK(journal.json_body()["result"]["undo"][0]["group"] == "turn-1");
}

TEST_CASE("each check of a request answers with its status") {
    CommandServerOptions options;
    options.head_timeout_ms = 300;
    options.body_timeout_ms = 300;
    Served sv("server-checks", options);
    const std::string host = "Host: 127.0.0.1:" + std::to_string(sv.port) + "\r\n";
    const std::string auth = "Authorization: Bearer " + sv.token + "\r\n";

    const HttpReply no_token = http(sv.port, "GET", "/v0/commands", "");
    CHECK(no_token.status == 401);
    CHECK(has_header(no_token, "WWW-Authenticate: Bearer"));
    CHECK(http(sv.port, "GET", "/v0/commands", sv.token + "x").status == 401);
    CHECK(http(sv.port, "GET", "/v0/commands", sv.token.substr(1)).status == 401);
    CHECK(http_raw(sv.port, "GET /v0/commands HTTP/1.1\r\nHost: example.com:" + std::to_string(sv.port) + "\r\n" + auth + "\r\n")
              .status == 403);
    CHECK(http_raw(sv.port, "GET /v0/commands HTTP/1.1\r\nHost: 127.0.0.1:1\r\n" + auth + "\r\n").status == 403);
    CHECK(http_raw(sv.port, "GET /v0/commands HTTP/1.1\r\n" + auth + "\r\n").status == 403);
    CHECK(http(sv.port, "GET", "/v0/commands", sv.token, "", {"Origin: https://evil.example"}).status == 403);
    CHECK(http(sv.port, "GET", "/v0/commands", sv.token, "", {"Origin: http://localhost.evil.example"}).status == 403);
    CHECK(http(sv.port, "GET", "/v0/commands", sv.token, "", {"Origin: http://localhost:5173"}).status == 200);
    CHECK(http(sv.port, "GET", "/v0/commands", sv.token, "", {"Origin: null"}).status == 200);
    CHECK(http_raw(sv.port, "GET /v0/commands HTTP/1.1\r\nHost: LOCALHOST:" + std::to_string(sv.port) + "\r\n" + auth +
                                "\r\n")
              .status == 200);

    CHECK(http(sv.port, "GET", "/v0/info", sv.token).status == 404);
    CHECK(http(sv.port, "POST", "/v0/commands/a/b", sv.token, "{}").status == 404);
    CHECK(http(sv.port, "POST", "/v0/commands/", sv.token, "{}").status == 404);
    const HttpReply wrong_method = http(sv.port, "GET", "/v0/commands/spec.get", sv.token);
    CHECK(wrong_method.status == 405);
    CHECK(has_header(wrong_method, "Allow: POST"));
    CHECK(http(sv.port, "POST", "/v0/commands", sv.token, "{}").status == 405);

    CHECK(http_raw(sv.port, "POST /v0/commands/spec.get HTTP/1.1\r\n" + host + auth + "Content-Type: application/json\r\n\r\n{}")
              .status == 411);
    CHECK(http(sv.port, "POST", "/v0/commands/spec.get", sv.token, "{}", {}).status == 200);
    CHECK(http_raw(sv.port, "POST /v0/commands/spec.get HTTP/1.1\r\n" + host + auth +
                                "Content-Type: text/plain\r\nContent-Length: 2\r\n\r\n{}")
              .status == 415);
    CHECK(http_raw(sv.port, "POST /v0/commands/spec.get HTTP/1.1\r\n" + host + auth +
                                "Content-Type: Application/JSON; charset=utf-8\r\nContent-Length: 2\r\n\r\n{}")
              .status == 200);
    CHECK(http_raw(sv.port, "GET /v0/commands HTTP/1.1\r\n" + host + auth + "Content-Length: 2\r\n\r\n{}").status == 413);
    const std::string big(std::size_t(1) << 20 | 1, ' ');
    CHECK(http(sv.port, "POST", "/v0/commands/spec.get", sv.token, big).status == 413);
    CHECK(http_raw(sv.port, "POST /v0/commands/spec.get HTTP/1.1\r\n" + host + auth +
                                "Content-Type: application/json\r\nContent-Length: 99999999999999999999\r\n\r\n")
              .status == 413);
    CHECK(http_raw(sv.port, "POST /v0/commands/spec.get HTTP/1.1\r\n" + host + auth +
                                "Content-Type: application/json\r\nContent-Length: -1\r\n\r\n")
              .status == 400);
    CHECK(http(sv.port, "POST", "/v0/commands/spec.get", sv.token, "{nope").status == 400);
    const HttpReply array = http(sv.port, "POST", "/v0/commands/spec.get", sv.token, "[1, 2]");
    CHECK(array.status == 200);
    CHECK(array.json_body()["error"]["code"] == "command.invalid_args");
    const HttpReply deep = http(sv.port, "POST", "/v0/commands/spec.get", sv.token,
                                "{\"a\":" + std::string(100000, '[') + std::string(100000, ']') + "}");
    CHECK(deep.status == 200);
    CHECK(deep.json_body()["error"]["code"] == "command.invalid_args");
    const HttpReply bad_args = sv.post("catalog.list", {{"limit", 5000}});
    CHECK(bad_args.json_body()["error"]["where"] == "args.limit");
    CHECK(sv.post("spec.get", json::object(), {"X-Caliper-Origin: ui"}).status == 400);
    CHECK(sv.post("spec.get", json::object(), {"X-Caliper-Origin: builder"}).status == 400);
    CHECK(sv.post("spec.get", json::object(), {"X-Caliper-Group: bad group"}).json_body()["error"]["code"] ==
          "command.invalid_invocation");

    // Hostile heads.
    CHECK(http_raw(sv.port, "GET /v0/commands HTTP/1.1\r\n" + host + auth + "X-Big: " + std::string(20000, 'a') + "\r\n\r\n")
              .status == 431);
    CHECK(http_raw(sv.port, "GET  /v0/commands HTTP/1.1\r\n" + host + auth + "\r\n").status == 400);
    CHECK(http_raw(sv.port, "GET /v0/commands HTTP/2.0\r\n" + host + auth + "\r\n").status == 400);
    CHECK(http_raw(sv.port, "get /v0/commands HTTP/1.1\r\n" + host + auth + "\r\n").status == 400);
    CHECK(http_raw(sv.port, "GET /v0/commands HTTP/1.1\r\n" + host + host + auth + "\r\n").status == 400);
    CHECK(http_raw(sv.port, "GET /v0/commands HTTP/1.1\r\n" + host + auth + " folded\r\n\r\n").status == 400);
    CHECK(http_raw(sv.port, "GET /v0/commands HTTP/1.1\r\n" + host + auth + "Bad\x01Header: x\r\n\r\n").status == 400);
    CHECK(http_raw(sv.port, "GET /v0/commands HTTP/1.1\n" + host + auth + "\r\n").status == 400);
    CHECK(http_raw(sv.port, "POST /v0/commands/spec.get HTTP/1.1\r\n" + host + auth +
                                "Content-Type: application/json\r\nTransfer-Encoding: chunked\r\n\r\n2\r\n{}\r\n0\r\n\r\n")
              .status == 501);
    // A client that sends half a head, or half a body, gets 408.
    CHECK(http_raw(sv.port, "GET /v0/commands HTTP/1.1\r\n" + host, 5000, false).status == 408);
    CHECK(http_raw(sv.port, "POST /v0/commands/spec.get HTTP/1.1\r\n" + host + auth +
                                "Content-Type: application/json\r\nContent-Length: 10\r\n\r\n{}",
                   5000, false)
              .status == 408);
    // A client that goes away before its head is complete gets nothing.
    CHECK(http_raw(sv.port, "GET /v0/commands HTTP/1.1\r\n" + host).status == 0);
    CHECK(sv.server->served() > 40);
}

TEST_CASE("a call that needs approval waits for mcp and agent and answers at once for api") {
    CommandServerOptions options;
    options.approval_wait = std::chrono::milliseconds(400);
    Served sv("server-approval", options);
    write_text(sv.s->workspace() + "/data/a.csv", "x\n1\n");
    REQUIRE(sv.post("data.import", {{"paths", {"data/a.csv"}}, {"run", false}}).json_body()["ok"] == true);

    // api: command.needs_approval with the id, at once.
    const json now = sv.post("loader.run", {{"id", "a"}}).json_body();
    CHECK(now["error"]["code"] == "command.needs_approval");
    CHECK(now["error"]["approval"].is_string());

    // mcp: held until the person approves.
    std::atomic<bool> answered{false};
    json held;
    std::thread caller([&] {
        held = sv.post("loader.run", {{"id", "a"}}, {"X-Caliper-Origin: mcp"}).json_body();
        answered.store(true);
    });
    std::string approval;
    for (int i = 0; i < 200 && approval.empty(); ++i) {
        for (const auto& p : sv.s->registry().pending_approvals())
            if (p.invocation.origin == CommandOrigin::mcp) approval = p.approval;
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }
    REQUIRE(!approval.empty());
    CHECK(!answered.load());
    sv.s->queue().approve(approval);
    caller.join();
    CHECK(held["ok"] == true);
    CHECK(held["result"]["outputs"][0]["id"] == "table:a");

    // agent: withdrawn after the wait.
    const json late = sv.post("loader.delete", {{"id", "a"}}, {"X-Caliper-Origin: agent"}).json_body();
    CHECK(late["error"]["code"] == "command.approval_timeout");
    // Denied.
    std::thread denier([&] {
        for (int i = 0; i < 200; ++i) {
            for (const auto& p : sv.s->registry().pending_approvals())
                if (p.invocation.origin == CommandOrigin::agent) {
                    sv.s->queue().deny(p.approval);
                    return;
                }
            std::this_thread::sleep_for(std::chrono::milliseconds(5));
        }
    });
    const json denied = sv.post("loader.delete", {{"id", "a"}}, {"X-Caliper-Origin: agent"}).json_body();
    denier.join();
    CHECK(denied["error"]["code"] == "command.denied");
}

TEST_CASE("connections beyond the bound get 503") {
    CommandServerOptions options;
    options.max_connections = 1;
    options.head_timeout_ms = 1500;
    Served sv("server-busy", options);
    // The first connection sends nothing and holds the one slot.
    std::thread slow([&] { http_raw(sv.port, "GET /v0/commands HTTP/1.1\r\n", 5000, false); });
    std::this_thread::sleep_for(std::chrono::milliseconds(300));
    const HttpReply busy = http(sv.port, "GET", "/v0/commands", sv.token);
    CHECK(busy.status == 503);
    CHECK(has_header(busy, "Retry-After: 1"));
    slow.join();
    CHECK(http(sv.port, "GET", "/v0/commands", sv.token).status == 200);
}

TEST_CASE("the API file is private and removed only by its owner, and tokens are well formed") {
    const std::string dir = fresh_dir("api-file") + "/data";
    std::string path;
    REQUIRE(write_api_file(dir, 4242, "tok", &path).ok());
    struct stat st;
    REQUIRE(::stat(path.c_str(), &st) == 0);
    CHECK((st.st_mode & 0777) == 0600);
    const json api = json::parse(read_text(path));
    CHECK(api["version"] == 1);
    CHECK(api["port"] == 4242);
    CHECK(api["token"] == "tok");
    CHECK(api["url"] == "http://127.0.0.1:4242/v0");
    CHECK(api["pid"] == static_cast<std::int64_t>(::getpid()));
    // Another process's file is left alone.
    write_text(path, json{{"version", 1}, {"pid", 1}, {"port", 1}, {"token", "x"}, {"url", "u"}}.dump());
    remove_api_file(path);
    CHECK(fs::exists(path));
    REQUIRE(write_api_file(dir, 4242, "tok", &path).ok());
    remove_api_file(path);
    CHECK(!fs::exists(path));
    CHECK(write_api_file(path + "/sub/deeper", 1, "t", &path).code == "server.io");

    const std::string a = generate_session_token();
    const std::string b = generate_session_token();
    CHECK(a.size() == 32);
    CHECK(a != b);
    for (char c : a) CHECK((std::isalnum(static_cast<unsigned char>(c)) || c == '-' || c == '_'));
    CHECK(token_equal(a, a));
    CHECK(!token_equal(a, b));
    CHECK(!token_equal(a, a.substr(0, 31)));
    CHECK(!token_equal("", a));

    CommandServerOptions options;  // no token
    auto s = open_session(fresh_dir("server-no-token") + "/workspace");
    CommandServer server(s->queue(), options);
    CHECK(server.start().code == "server.no_token");
}

#endif
