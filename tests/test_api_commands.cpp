// Caliper API's command endpoints over a real builder session
// (src/host/api/session_gateway.h): the app's API server (api_server.h) in
// this process, a caliper_builder::Session over a temporary workspace behind
// it, and a fake frame thread that does what the app's frame loop does: it
// takes the API's calls, answers those of API v0, and drains the session's
// command queue. Everything is driven over HTTP on 127.0.0.1, and the last
// case runs the built caliper-mcp as a child against the server's API file.
// Nothing here opens a window. POSIX only, like the other socket tests.
#include <doctest/doctest.h>

#include "api/api_server.h"
#include "api/session_gateway.h"
#include "caliper/live_net.h"
#include "net/wire.h"
#include "net/ws.h"
#include "test_temp.h"

#include "mcp_child.h"

#include <caliper_builder/commands/command_table.h>
#include <caliper_builder/session/session.h>

#include <nlohmann/json.hpp>

#include <atomic>
#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <functional>
#include <map>
#include <memory>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <sys/time.h>
#include <unistd.h>

namespace fs = std::filesystem;
namespace wire = caliper_net::wire;
using nlohmann::json;

namespace {

const std::string kToken = "api_commands_token-0123456789abcdef";

// ---- raw HTTP -----------------------------------------------------------------------------
struct Response {
    int status = 0;
    std::map<std::string, std::string> headers;   // names lower-cased
    std::string body;
    json value() const {
        json j = json::parse(body, nullptr, false);
        REQUIRE_MESSAGE(!j.is_discarded(), "not JSON: " << body);
        return j;
    }
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

int open_socket(uint16_t port) {
    const int fd = ::socket(AF_INET, SOCK_STREAM, 0);
    sockaddr_in a{};
    a.sin_family = AF_INET;
    a.sin_port = htons(port);
    inet_pton(AF_INET, "127.0.0.1", &a.sin_addr);
    if (::connect(fd, (sockaddr*)&a, sizeof a) != 0) {
        ::close(fd);
        return -1;
    }
    timeval tv{20, 0};
    setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof tv);
    return fd;
}

Response read_response(int fd) {
    std::string out;
    char buf[65536];
    for (;;) {
        const ssize_t r = ::recv(fd, buf, sizeof buf, 0);
        if (r <= 0) break;
        out.append(buf, (size_t)r);
    }
    ::close(fd);
    return parse_response(out);
}

int send_request(uint16_t port, const std::string& method, const std::string& target, const std::string& extra,
                 const std::string& body = {}) {
    const int fd = open_socket(port);
    REQUIRE(fd >= 0);
    const std::string r = method + " " + target + " HTTP/1.1\r\nHost: 127.0.0.1:" + std::to_string(port) + "\r\n" +
                          extra + "\r\n" + body;
    ::send(fd, r.data(), r.size(), 0);
    return fd;
}

std::string bearer() { return "Authorization: Bearer " + kToken + "\r\n"; }

Response get(uint16_t port, const std::string& target) {
    return read_response(send_request(port, "GET", target, bearer()));
}

// POST /v0/commands/<name> on a socket of its own; read the answer with read_response().
int send_command(uint16_t port, const std::string& name, const json& args, const std::string& origin = {},
                 const std::string& group = {}, const std::string& preview = {}) {
    const std::string body = args.dump();
    std::string extra = bearer() + "Content-Type: application/json\r\nContent-Length: " + std::to_string(body.size()) +
                        "\r\n";
    if (!origin.empty()) extra += "X-Caliper-Origin: " + origin + "\r\n";
    if (!group.empty()) extra += "X-Caliper-Group: " + group + "\r\n";
    if (!preview.empty()) extra += "X-Caliper-Preview: " + preview + "\r\n";
    return send_request(port, "POST", "/v0/commands/" + name, extra, body);
}

// The envelope of one command, which must come with 200.
json command(uint16_t port, const std::string& name, const json& args, const std::string& origin = {},
             const std::string& group = {}, const std::string& preview = {}) {
    const Response r = read_response(send_command(port, name, args, origin, group, preview));
    INFO(name << ": " << r.status << " " << r.body.substr(0, 1000));
    REQUIRE(r.status == 200);
    const json env = r.value();
    CHECK(env["id"].get<std::string>().rfind("api-", 0) == 0);
    return env;
}

std::string error_code(const json& env) {
    INFO(env.dump());
    REQUIRE(env["ok"] == false);
    return env["error"]["code"].get<std::string>();
}

bool wait_for(const std::function<bool()>& done, int ms = 10000) {
    const auto end = std::chrono::steady_clock::now() + std::chrono::milliseconds(ms);
    while (std::chrono::steady_clock::now() < end) {
        if (done()) return true;
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }
    return done();
}

// HELLO as a viewer on the stream; true once WELCOME arrives.
bool viewer_handshake(uint16_t port, std::string* why) {
    caliper_net::Client c;
    if (!c.connect("ws://127.0.0.1:" + std::to_string(port) + "/v3/stream?token=" + kToken, why)) return false;
    bool open = false;
    const auto end = std::chrono::steady_clock::now() + std::chrono::seconds(10);
    while (std::chrono::steady_clock::now() < end) {
        for (auto& e : c.poll(10)) {
            using T = caliper_net::Event::Type;
            if (e.type == T::Closed) {
                *why = "closed: " + e.reason;
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
                if (std::holds_alternative<wire::Welcome>(m)) {
                    c.close();
                    return true;
                }
            }
        }
    }
    *why = "no WELCOME within 10 s";
    return false;
}

// ---- the app: the API server, a session and a frame thread ----------------------------------
struct SessionApp {
    fs::path root;
    std::unique_ptr<caliper_builder::Session> session;
    std::unique_ptr<caliper_api::Server> server;
    std::thread frame;
    std::atomic<bool> stop{false};

    explicit SessionApp(caliper_api::Options o = caliper_api::Options())
        : root(caliper_test::unique_temp_path("caliper-api-commands")) {
        fs::create_directories(root);
        // Review of spec edits is off here, so that an agent's edit answers at
        // once; the review case below turns it on for one origin.
        caliper_builder::SessionOptions session_options;
        session_options.review.agent = false;
        session_options.review.mcp = false;
        caliper_builder::Session::Open opened =
            caliper_builder::Session::open((root / "workspace").string(), session_options);
        REQUIRE_MESSAGE(opened.error.ok(), opened.error.message);
        session = std::move(opened.session);
        o.port = 0;
        o.token = kToken;
        o.info_json = "{\"host\":\"test\"}";
        server = std::make_unique<caliper_api::Server>(std::move(o));
        server->set_command_gateway(caliper_api::make_session_gateway(*session));
        std::string err;
        REQUIRE_MESSAGE(server->start(&err), err);
        // The frame thread, as the app's frame loop: it names itself to the
        // registry, takes the API's calls and drains the command queue.
        frame = std::thread([this] {
            session->registry().set_frame_thread(std::this_thread::get_id());
            while (!stop.load()) {
                for (const caliper_api::Call& c : server->take()) {
                    if (c.kind == caliper_api::Call::Kind::Applets) server->reply(c.ticket, 200, "{\"applets\":[]}");
                    else server->reply(c.ticket, 404, caliper_api::error_json("not in this test"));
                }
                session->queue().drain();
                std::this_thread::sleep_for(std::chrono::milliseconds(2));
            }
        });
    }
    ~SessionApp() {
        stop.store(true);
        if (frame.joinable()) frame.join();
        server->set_command_gateway(nullptr);
        server->stop();
        session->close();
        session.reset();
        std::error_code ec;
        fs::remove_all(root, ec);
    }
    uint16_t port() const { return server->port(); }

    // The approval the registry waits on for `name`, once it is there.
    std::string pending_approval(const std::string& name, const std::string& origin) {
        std::string id;
        wait_for([&] {
            for (const auto& p : session->registry().pending_approvals()) {
                if (p.invocation.name == name &&
                    caliper_builder::command_origin_name(p.invocation.origin) == origin) {
                    id = p.approval;
                    return true;
                }
            }
            return false;
        });
        return id;
    }
};

// A spec with one applet id, saved once, so that saving again overwrites.
void save_one_applet(uint16_t port) {
    REQUIRE(command(port, "spec.new", {{"id", "dash"}, {"title", "Dash"}}, "mcp")["ok"] == true);
    const json saved = command(port, "applet.save", json::object(), "mcp");
    INFO(saved.dump());
    REQUIRE(saved["ok"] == true);
}

}  // namespace

TEST_CASE("api commands: the table, a read, spec edits and undo over a real session") {
    SessionApp app;
    const uint16_t port = app.port();

    const Response table = get(port, "/v0/commands");
    REQUIRE(table.status == 200);
    CHECK(table.value() == caliper_builder::command_table_to_json());

    // A read.
    const json listed = command(port, "applet.list", json::object());
    REQUIRE(listed["ok"] == true);
    CHECK(listed["result"]["applets"] == json::array());
    CHECK(error_code(command(port, "spec.get", json::object())) == "spec.no_open_spec");

    // Spec edits, which run on the frame thread, with a group from the header.
    CHECK(command(port, "spec.new", {{"id", "dash"}, {"title", "Dash"}}, "mcp", "turn-1")["ok"] == true);
    const json made = command(port, "panel.create",
                              {{"id", "note"}, {"primitive", "text"}, {"options", {{"text", "Hi"}}}}, "agent");
    INFO(made.dump());
    REQUIRE(made["ok"] == true);
    CHECK(made["result"]["applied"] == true);
    const json got = command(port, "spec.get", json::object());
    REQUIRE(got["ok"] == true);
    CHECK(got["result"]["id"] == "dash");
    REQUIRE(app.session->open_spec().has_value());
    CHECK(app.session->open_spec()->spec.panels.size() == 1);
    const json history = command(port, "history.list", json::object());
    REQUIRE(history["ok"] == true);
    REQUIRE(history["result"]["undo"].size() == 2);
    CHECK(history["result"]["undo"][0]["origin"] == "agent");
    CHECK(history["result"]["undo"][1]["group"] == "turn-1");
    CHECK(history["result"]["undo"][1]["origin"] == "mcp");

    // Undo through history.undo removes the panel; redo brings it back.
    const json undone = command(port, "history.undo", json::object(), "mcp");
    INFO(undone.dump());
    REQUIRE(undone["ok"] == true);
    CHECK(app.session->open_spec()->spec.panels.empty());
    CHECK(app.session->open_spec()->spec.applet.id == "dash");
    CHECK(command(port, "history.redo", json::object(), "mcp")["ok"] == true);
    CHECK(app.session->open_spec()->spec.panels.size() == 1);

    // An edit that names a stacked preview in X-Caliper-Preview applies to the
    // preview, not to the open spec; only the person commits it.
    const json begun = command(port, "preview.begin", {{"id", "try"}}, "agent");
    INFO(begun.dump());
    REQUIRE(begun["ok"] == true);
    const json patched = command(port, "panel.patch", {{"id", "note"}, {"patch", {{"title", "Draft"}}}}, "agent", {},
                                 "try");
    INFO(patched.dump());
    REQUIRE(patched["ok"] == true);
    CHECK_FALSE(app.session->open_spec()->spec.panels[0].title.has_value());
    const json diff = command(port, "preview.diff", {{"id", "try"}}, "agent");
    REQUIRE(diff["ok"] == true);
    CHECK(diff["result"]["edits"] == 1);
    CHECK(diff["result"]["diff"].get<std::string>().find("Draft") != std::string::npos);
    CHECK(error_code(command(port, "preview.commit", {{"id", "try"}}, "agent")) == "preview.not_allowed");
    CHECK(error_code(command(port, "panel.patch", {{"id", "note"}, {"patch", {{"title", "x"}}}}, "agent", {},
                             "Not A Preview")) == "command.invalid_invocation");
    CHECK(command(port, "preview.discard", {{"id", "try"}}, "agent")["ok"] == true);

    // What the command layer refuses comes with 200.
    CHECK(error_code(command(port, "no.such", json::object())) == "command.unknown");
    const json bad = command(port, "spec.new", {{"id", 5}});
    CHECK(error_code(bad) == "command.invalid_args");
    CHECK(bad["error"]["where"].get<std::string>().rfind("args", 0) == 0);
    CHECK(error_code(command(port, "spec.new", {{"id", "x"}}, "mcp", "not a group")) == "command.invalid_invocation");
    CHECK(error_code(command(port, "panel.create", {{"id", "note"}, {"primitive", "text"}}, "mcp")) != "");
    // An invalid group or name ran nothing.
    CHECK(app.session->open_spec()->spec.applet.id == "dash");
}

TEST_CASE("api commands: the viewer token opens the spec page and runs its reads from the viewer origin only") {
    caliper_api::Options o;
    o.viewer_token = "api_viewer_token-0123456789abcdef";
    o.spec_page_html = "<!doctype html><title>Spec</title>";
    SessionApp app(o);
    const uint16_t port = app.port();
    const std::string viewer = "Authorization: Bearer " + o.viewer_token + "\r\n";
    auto as_viewer = [&](const std::string& name, const json& args, const std::string& extra = {}) {
        const std::string body = args.dump();
        return read_response(send_request(port, "POST", "/v0/commands/" + name,
                                          viewer + extra + "Content-Type: application/json\r\nContent-Length: " +
                                              std::to_string(body.size()) + "\r\n",
                                          body));
    };
    REQUIRE(command(port, "spec.new", {{"id", "dash"}, {"title", "Dash"}})["ok"] == true);
    REQUIRE(command(port, "panel.create", {{"id", "note"}, {"primitive", "text"}, {"options", {{"text", "Hi"}}}})["ok"] ==
            true);
    REQUIRE(command(port, "applet.save", json::object())["ok"] == true);
    CHECK(app.server->spec_url() == "http://127.0.0.1:" + std::to_string(port) + "/v0/spec?token=" + o.viewer_token);

    // The page, with the viewer token in its address, and with the session token.
    const Response page = read_response(send_request(port, "GET", "/v0/spec?token=" + o.viewer_token + "&applet=dash", ""));
    CHECK(page.status == 200);
    CHECK(page.body == o.spec_page_html);
    CHECK(page.headers.at("content-type") == "text/html; charset=utf-8");
    CHECK(page.headers.at("content-security-policy").find("connect-src 'self'") != std::string::npos);
    CHECK(page.headers.at("referrer-policy") == "no-referrer");
    CHECK(get(port, "/v0/spec").status == 200);
    CHECK(read_response(send_request(port, "GET", "/v0/spec", "")).status == 401);

    // The page's three reads, from the viewer origin.
    const Response got = as_viewer("spec.get", {{"applet", "dash"}});
    REQUIRE(got.status == 200);
    CHECK(got.value()["result"]["json"]["panel"][0]["id"] == "note");
    CHECK(as_viewer("applet.list", json::object()).value()["result"]["applets"].size() == 1);
    const Response data = as_viewer("panel.data", {{"applet", "dash"}, {"panel", "note"}});
    REQUIRE(data.status == 200);
    CHECK(data.value()["result"]["data"]["value"].is_null());
    CHECK(as_viewer("panel.data", {{"applet", "dash"}, {"panel", "note"}}, "X-Caliper-Origin: viewer\r\n").status == 200);
    CHECK(as_viewer("panel.data", {{"applet", "dash"}, {"panel", "note"}}, "X-Caliper-Origin: api\r\n").status == 400);

    // Nothing else: other commands, other endpoints and the stream.
    for (const char* name : {"panel.remove", "spec.set_text", "applet.save", "data.query", "capture.frame",
                             "history.undo"}) {
        CAPTURE(name);
        CHECK(as_viewer(name, json::object()).status == 403);
    }
    for (const char* path : {"/v0/info", "/v0/applets", "/v0/commands", "/v0/capture.png", "/v0/approvals/ap-1"}) {
        CAPTURE(path);
        CHECK(read_response(send_request(port, "GET", path, viewer)).status == 403);
    }
    CHECK(read_response(send_request(port, "POST", "/v0/applets/x/launch", viewer)).status == 403);
    caliper_net::Client c;
    std::string why;
    bool refused = !c.connect("ws://127.0.0.1:" + std::to_string(port) + "/v3/stream?token=" + o.viewer_token, &why);
    if (!refused) {
        for (int i = 0; i < 200 && !refused; ++i)
            for (auto& e : c.poll(10))
                if (e.type == caliper_net::Event::Type::Closed) refused = true;
    }
    CHECK(refused);

    // With the session token, a request may lower itself to the viewer origin,
    // and the command layer then runs reads only.
    CHECK(command(port, "panel.data", {{"applet", "dash"}, {"panel", "note"}}, "viewer")["ok"] == true);
    CHECK(error_code(command(port, "panel.remove", {{"id", "note"}}, "viewer")) == "command.not_allowed");
    REQUIRE(app.session->open_spec().has_value());
    CHECK(app.session->open_spec()->spec.panels.size() == 1);
}

TEST_CASE("api commands: a destructive command from mcp or agent is held while the person decides") {
    SessionApp app;
    const uint16_t port = app.port();
    save_one_applet(port);

    // Overwriting the saved applet from mcp: held until the person approves.
    const int held = send_command(port, "applet.save", {{"overwrite", true}}, "mcp");
    const std::string id = app.pending_approval("applet.save", "mcp");
    REQUIRE_FALSE(id.empty());
    CHECK(app.server->stats().commands_held == 1);
    // Meanwhile API v0, the stream, the table, approvals and other commands answer.
    CHECK(get(port, "/v0/info").status == 200);
    CHECK(get(port, "/v0/applets").status == 200);
    CHECK(get(port, "/v0/commands").status == 200);
    const Response look = get(port, "/v0/approvals/" + id);
    REQUIRE(look.status == 200);
    CHECK(look.value() == json{{"approval", id}, {"state", "pending"}, {"result", nullptr}, {"name", "applet.save"},
                               {"origin", "mcp"}, {"review", nullptr}});
    std::string why;
    CHECK_MESSAGE(viewer_handshake(port, &why), why);
    CHECK(command(port, "applet.list", json::object(), "mcp")["result"]["applets"].size() == 1);
    // Approved in the app, it runs and answers.
    CHECK(app.session->queue().approve(id).get().ok);
    const Response approved = read_response(held);
    REQUIRE(approved.status == 200);
    const json env = approved.value();
    INFO(env.dump());
    CHECK(env["ok"] == true);
    CHECK(env["result"]["id"] == "dash");
    CHECK(app.server->stats().commands_held == 0);
    CHECK(get(port, "/v0/approvals/" + id).value()["state"] == "decided");

    // Deleting a loader from agent: denied in the app.
    const int denied_fd = send_command(port, "loader.delete", {{"id", "nope"}}, "agent");
    const std::string id2 = app.pending_approval("loader.delete", "agent");
    REQUIRE_FALSE(id2.empty());
    app.session->queue().deny(id2);
    const Response denied = read_response(denied_fd);
    REQUIRE(denied.status == 200);
    CHECK(error_code(denied.value()) == "command.denied");
    CHECK(app.session->registry().pending_approvals().empty());
}

TEST_CASE("api commands: with review on, an agent's spec edit is held with its diff until the person approves") {
    SessionApp app;
    const uint16_t port = app.port();
    REQUIRE(command(port, "spec.new", {{"id", "dash"}, {"title", "Dash"}}, "mcp")["ok"] == true);
    app.session->set_review(caliper_builder::CommandOrigin::mcp, true);

    const int held = send_command(port, "panel.create",
                                  {{"id", "note"}, {"primitive", "text"}, {"options", {{"text", "Hi"}}}}, "mcp");
    const std::string id = app.pending_approval("panel.create", "mcp");
    REQUIRE_FALSE(id.empty());
    // The approval's state carries the change for the person's card: the
    // command, its origin and the diff of the spec's canonical text.
    const json look = get(port, "/v0/approvals/" + id).value();
    INFO(look.dump());
    CHECK(look["state"] == "pending");
    CHECK(look["name"] == "panel.create");
    CHECK(look["origin"] == "mcp");
    REQUIRE(look["review"].is_object());
    CHECK(look["review"]["diff"].get<std::string>().find("+id = \"note\"") != std::string::npos);
    CHECK(look["review"]["warnings"].is_array());
    // The open spec is unchanged while the edit waits.
    CHECK(command(port, "spec.get", json::object(), "api")["result"]["json"].value("panel", json::array()).empty());
    CHECK(app.session->queue().approve(id).get().ok);
    const Response approved = read_response(held);
    REQUIRE(approved.status == 200);
    CHECK(approved.value()["ok"] == true);
    CHECK(command(port, "spec.get", json::object(), "api")["result"]["json"]["panel"].size() == 1);
}

TEST_CASE("api commands: a held command that nobody decides is withdrawn with command.approval_timeout") {
    caliper_api::Options o;
    o.command_approval_wait_ms = 300;
    SessionApp app(o);
    const uint16_t port = app.port();
    const auto t0 = std::chrono::steady_clock::now();
    const json env = command(port, "loader.delete", {{"id", "nope"}}, "mcp");
    const double waited = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    CHECK(error_code(env) == "command.approval_timeout");
    CHECK(waited >= 0.25);
    CHECK(waited < 5.0);
    CHECK(app.session->registry().pending_approvals().empty());
    CHECK(app.server->stats().commands_held == 0);
}

TEST_CASE("api commands: an api call gets command.needs_approval at once and reads the outcome later") {
    SessionApp app;
    const uint16_t port = app.port();
    save_one_applet(port);

    const json asked = command(port, "applet.save", {{"overwrite", true}});
    CHECK(error_code(asked) == "command.needs_approval");
    const std::string id = asked["error"]["approval"].get<std::string>();
    REQUIRE_FALSE(id.empty());
    CHECK(app.server->stats().commands_held == 0);
    CHECK(get(port, "/v0/approvals/" + id).value() == json{{"approval", id}, {"state", "pending"}, {"result", nullptr},
                                                           {"name", "applet.save"}, {"origin", "api"},
                                                           {"review", nullptr}});

    CHECK(app.session->queue().approve(id).get().ok);
    const json decided = get(port, "/v0/approvals/" + id).value();
    INFO(decided.dump());
    CHECK(decided["state"] == "decided");
    CHECK(decided["result"]["ok"] == true);
    CHECK(decided["result"]["result"]["id"] == "dash");

    CHECK(get(port, "/v0/approvals/ap-999").value() ==
          json{{"approval", "ap-999"}, {"state", "unknown"}, {"result", nullptr}});
    CHECK(get(port, "/v0/approvals/a%2Fb").status == 400);
}

TEST_CASE("api commands: caliper-mcp drives a session through the app's API server") {
    SessionApp app;
    const fs::path api_file = app.root / "api.json";
    {
        json f{{"version", 1},
               {"pid", (long long)::getpid()},
               {"port", app.port()},
               {"token", kToken},
               {"url", app.server->base_url()},
               {"stream_url", app.server->stream_url()}};
        std::ofstream(api_file) << f.dump() << "\n";
        fs::permissions(api_file, fs::perms::owner_read | fs::perms::owner_write, fs::perm_options::replace);
    }
    caliper_builder::mcp::test::McpChild child({"--api-file", api_file.string()});
    const json tools = child.request("tools", "tools/list", json::object());
    REQUIRE(tools.contains("result"));
    std::size_t agents_may = 0;  // caliper-mcp offers no tool for the person's undo and redo
    for (const auto& c : caliper_builder::command_table_to_json()["commands"]) agents_may += c["agents"] == true;
    CHECK(tools["result"]["tools"].size() == agents_may);
    const json result = child.call("new", "spec_new", {{"id", "from-mcp"}, {"title", "From MCP"}});
    INFO(result.dump());
    CHECK(result["isError"] == false);
    REQUIRE(app.session->open_spec().has_value());
    CHECK(app.session->open_spec()->spec.applet.id == "from-mcp");
    CHECK(child.finish().success());
    CHECK(child.all_err().find("the app's command table") != std::string::npos);
    CHECK(child.all_out().find(kToken) == std::string::npos);
    CHECK(child.all_err().find(kToken) == std::string::npos);
}
