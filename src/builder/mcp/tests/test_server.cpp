// The protocol engine on its own, with an in-memory sink and no app: the
// JSON-RPC handling of hostile lines, both eras of MCP, the tools from the
// built-in table, and the resources.
#include <caliper_builder/programs/pieces.h>
#include <caliper_builder/mcp/server.h>

#include <caliper_builder/commands/command_table.h>
#include <caliper_builder/mcp/resources.h>

#include <doctest/doctest.h>

#include <nlohmann/json.hpp>

#include <chrono>
#include <condition_variable>
#include <filesystem>
#include <fstream>
#include <mutex>
#include <set>

using namespace caliper_builder;
using namespace caliper_builder::mcp;
using nlohmann::json;
namespace fs = std::filesystem;

namespace {

json modern_meta() {
    return {{"io.modelcontextprotocol/protocolVersion", "2026-07-28"},
            {"io.modelcontextprotocol/clientCapabilities", json::object()},
            {"io.modelcontextprotocol/clientInfo", {{"name", "test-client"}, {"version", "1.0"}}}};
}

json modern(json params = json::object()) {
    params["_meta"] = modern_meta();
    return params;
}

struct Harness {
    std::mutex mu;
    std::condition_variable cv;
    std::vector<json> out;
    std::vector<std::string> logs;
    bool open_sink = true;
    std::unique_ptr<Server> server;

    explicit Harness(ServerOptions o = {}) {
        if (o.api_file.empty())
            o.api_file = (fs::path(CALIPER_BUILDER_TEST_TMP) / "server-unit" / "no-app" / "api.json").string();
        server = std::make_unique<Server>(
            o,
            [this](const std::string& line) {
                std::lock_guard<std::mutex> g(mu);
                REQUIRE(line.find('\n') == std::string::npos);
                out.push_back(json::parse(line));
                cv.notify_all();
                return open_sink;
            },
            [this](const std::string& line) {
                std::lock_guard<std::mutex> g(mu);
                logs.push_back(line);
            });
        REQUIRE(server->open().ok());
    }
    ~Harness() { server->shutdown(); }

    void send(const json& j) { server->handle_line(j.dump()); }

    // The response with this id, waiting up to 10 s for it.
    json wait_id(const json& id) {
        std::unique_lock<std::mutex> g(mu);
        json found;
        cv.wait_for(g, std::chrono::seconds(10), [&] {
            for (const auto& m : out)
                if (m.contains("id") && m["id"] == id && !m.contains("method")) {
                    found = m;
                    return true;
                }
            return false;
        });
        REQUIRE_MESSAGE(!found.is_null(), "no response with id " << id.dump());
        return found;
    }

    json request(const json& id, const std::string& method, const json& params) {
        send({{"jsonrpc", "2.0"}, {"id", id}, {"method", method}, {"params", params}});
        return wait_id(id);
    }

    std::vector<json> take() {
        std::lock_guard<std::mutex> g(mu);
        std::vector<json> t;
        t.swap(out);
        return t;
    }

    // The single message produced by a line that is answered at once.
    json only(const std::string& line) {
        take();
        server->handle_line(line);
        auto t = take();
        REQUIRE(t.size() == 1);
        return t[0];
    }
};

void check_error(const json& m, int code, const json* id = nullptr) {
    CAPTURE(m.dump());
    CHECK(m["jsonrpc"] == "2.0");
    REQUIRE(m.contains("error"));
    CHECK(m["error"]["code"] == code);
    CHECK(m["error"]["message"].is_string());
    if (id) CHECK(m["id"] == *id);
    else CHECK_FALSE(m.contains("id"));
    CHECK_FALSE(m.contains("result"));
}

json tool_error(const json& response) {
    const json& r = response["result"];
    REQUIRE(r["isError"] == true);
    return json::parse(r["content"][0]["text"].get<std::string>())["error"];
}

}  // namespace

TEST_CASE("hostile lines get the JSON-RPC error they call for") {
    Harness h;
    const json one = 1;
    check_error(h.only("this is not json"), -32700);
    check_error(h.only("{\"jsonrpc\":\"2.0\",\"id\":1,\"method\":\"ping\""), -32700);
    check_error(h.only(std::string("{\"jsonrpc\":\"2.0\",\"id\":1,\"method\":\"p\xff\"}")), -32700);
    check_error(h.only("[{\"jsonrpc\":\"2.0\",\"id\":1,\"method\":\"ping\"}]"), -32600);
    check_error(h.only("42"), -32600);
    check_error(h.only("\"ping\""), -32600);
    check_error(h.only("{\"id\":1,\"method\":\"ping\"}"), -32600, &one);
    check_error(h.only("{\"jsonrpc\":\"1.0\",\"id\":1,\"method\":\"ping\"}"), -32600, &one);
    check_error(h.only("{\"jsonrpc\":\"2.0\",\"id\":{\"a\":1},\"method\":\"ping\"}"), -32600);
    check_error(h.only("{\"jsonrpc\":\"2.0\",\"id\":null,\"method\":\"ping\"}"), -32600);
    check_error(h.only("{\"jsonrpc\":\"2.0\",\"id\":1.5,\"method\":\"ping\"}"), -32600);
    check_error(h.only("{\"jsonrpc\":\"2.0\",\"id\":true,\"method\":\"ping\"}"), -32600);
    check_error(h.only("{\"jsonrpc\":\"2.0\",\"id\":1,\"method\":7}"), -32600, &one);
    check_error(h.only("{\"jsonrpc\":\"2.0\",\"id\":1}"), -32600, &one);
    check_error(h.only("{\"jsonrpc\":\"2.0\",\"id\":1,\"method\":\"ping\",\"params\":[1]}"), -32602, &one);
    std::string deep = "{\"jsonrpc\":\"2.0\",\"id\":1,\"method\":\"ping\",\"params\":{\"x\":";
    deep += std::string(200, '[') + std::string(200, ']') + "}}";
    check_error(h.only(deep), -32600);
    h.take();
    h.server->handle_oversized_line();
    auto t = h.take();
    REQUIRE(t.size() == 1);
    check_error(t[0], -32600);

    // Lines that need no answer.
    h.take();
    h.server->handle_line("");
    h.server->handle_line("   \r");
    h.server->handle_line("{\"jsonrpc\":\"2.0\",\"method\":\"notifications/whatever\"}");
    h.server->handle_line("{\"jsonrpc\":\"2.0\",\"method\":\"notifications/cancelled\",\"params\":{\"requestId\":99}}");
    h.server->handle_line("{\"jsonrpc\":\"2.0\",\"method\":\"notifications/cancelled\",\"params\":[]}");
    h.server->handle_line("{\"jsonrpc\":\"2.0\",\"id\":5,\"result\":{}}");
    h.server->handle_line("{\"jsonrpc\":\"2.0\",\"id\":5,\"error\":{\"code\":1,\"message\":\"x\"}}");
    CHECK(h.take().empty());

    // The server still answers afterwards.
    const json pong = h.request("after", "ping", json::object());
    CHECK(pong["result"] == json::object());
}

TEST_CASE("the legacy handshake negotiates a revision and serves the client") {
    Harness h;
    // ping works before initialize; other requests need a revision.
    CHECK(h.request(1, "ping", json::object())["result"] == json::object());
    const json early = h.request(2, "tools/list", json::object());
    const json two = 2;
    check_error(early, -32602, &two);
    CHECK(early["error"]["data"]["supported"].size() == supported_protocol_versions().size());

    const json bad = h.request(3, "initialize", {{"capabilities", json::object()}});
    const json three = 3;
    check_error(bad, -32602, &three);

    const json init = h.request(4, "initialize",
                                {{"protocolVersion", "2025-06-18"},
                                 {"capabilities", json::object()},
                                 {"clientInfo", {{"name", "legacy"}, {"version", "1"}}}});
    REQUIRE(init.contains("result"));
    CHECK(init["result"]["protocolVersion"] == "2025-06-18");
    CHECK(init["result"]["serverInfo"]["name"] == "caliper-mcp");
    CHECK(init["result"]["capabilities"]["tools"]["listChanged"] == true);
    CHECK(init["result"]["capabilities"].contains("resources"));
    CHECK(init["result"]["instructions"].is_string());
    CHECK_FALSE(init["result"].contains("resultType"));
    h.send({{"jsonrpc", "2.0"}, {"method", "notifications/initialized"}});

    for (const char* asked : {"2024-11-05", "2026-07-28", "1900-01-01"}) {
        const json again = h.request(std::string("init-") + asked, "initialize",
                                     {{"protocolVersion", asked}, {"capabilities", json::object()}});
        CHECK(again["result"]["protocolVersion"] == "2025-11-25");
    }
    const json latest = h.request(5, "initialize", {{"protocolVersion", "2025-11-25"}, {"capabilities", json::object()}});
    CHECK(latest["result"]["protocolVersion"] == "2025-11-25");

    const json list = h.request(6, "tools/list", json::object());
    REQUIRE(list.contains("result"));
    CHECK_FALSE(list["result"].contains("resultType"));
    CHECK_FALSE(list["result"].contains("ttlMs"));
    CHECK(list["result"]["tools"].size() == command_table().size() - 2);  // not undo, redo

    const json missing = h.request(7, "resources/read", {{"uri", "caliper://docs/none.md"}});
    const json seven = 7;
    check_error(missing, -32002, &seven);
    const json server_discover = h.request(8, "server/discover", json::object());
    const json eight = 8;
    check_error(server_discover, -32601, &eight);
    const json unknown = h.request(9, "prompts/list", json::object());
    const json nine = 9;
    check_error(unknown, -32601, &nine);
}

TEST_CASE("modern requests carry their revision and get the revision's result fields") {
    Harness h;
    const json d = h.request("d", "server/discover", modern());
    REQUIRE(d.contains("result"));
    const json& r = d["result"];
    CHECK(r["resultType"] == "complete");
    CHECK(r["supportedVersions"] == json(supported_protocol_versions()));
    CHECK(r["supportedVersions"][0] == "2026-07-28");
    CHECK(r["capabilities"]["tools"]["listChanged"] == true);
    CHECK(r["_meta"]["io.modelcontextprotocol/serverInfo"]["name"] == "caliper-mcp");
    CHECK(r["ttlMs"].is_number_integer());
    CHECK(r["cacheScope"] == "public");

    json old = modern();
    old["_meta"]["io.modelcontextprotocol/protocolVersion"] = "1900-01-01";
    const json unsupported = h.request("u", "tools/list", old);
    const json u = "u";
    check_error(unsupported, -32022, &u);
    CHECK(unsupported["error"]["data"]["requested"] == "1900-01-01");
    CHECK(unsupported["error"]["data"]["supported"] == json(supported_protocol_versions()));

    json no_caps = modern();
    no_caps["_meta"].erase("io.modelcontextprotocol/clientCapabilities");
    const json n = "n";
    check_error(h.request("n", "tools/list", no_caps), -32602, &n);
    json not_string = modern();
    not_string["_meta"]["io.modelcontextprotocol/protocolVersion"] = 20260728;
    const json s = "s";
    check_error(h.request("s", "tools/list", not_string), -32602, &s);

    const json list = h.request(10, "tools/list", modern());
    CHECK(list["result"]["resultType"] == "complete");
    CHECK(list["result"]["ttlMs"] == 0);
    CHECK(list["result"]["cacheScope"] == "public");
    CHECK(list["result"]["tools"].size() == command_table().size() - 2);  // not undo, redo
    const json cursor = h.request(11, "tools/list", modern({{"cursor", "abc"}}));
    const json eleven = 11;
    check_error(cursor, -32602, &eleven);

    const json init = h.request(12, "initialize", modern({{"protocolVersion", "2026-07-28"}}));
    const json twelve = 12;
    check_error(init, -32601, &twelve);
    const json ping = h.request(13, "ping", modern());
    CHECK(ping["result"]["resultType"] == "complete");
}

TEST_CASE("tools/call checks the tool and its arguments before it looks for the app") {
    Harness h;
    const auto& table = command_table();
    const std::string first_tool = command_tool_name(table.front().name);

    const json unknown = h.request(1, "tools/call", modern({{"name", "no_such_tool"}, {"arguments", json::object()}}));
    const json one = 1;
    check_error(unknown, -32602, &one);
    // The person's undo and redo are no tools: a forced call is refused as unknown.
    for (const char* forced : {"history_undo", "history_redo"}) {
        const json r = h.request(std::string("forced-") + forced, "tools/call",
                                 modern({{"name", forced}, {"arguments", json::object()}}));
        const json id = std::string("forced-") + forced;
        check_error(r, -32602, &id);
    }
    const json no_name = h.request(2, "tools/call", modern({{"arguments", json::object()}}));
    const json two = 2;
    check_error(no_name, -32602, &two);
    const json array_args = h.request(3, "tools/call", modern({{"name", first_tool}, {"arguments", json::array()}}));
    const json three = 3;
    check_error(array_args, -32602, &three);

    // Every tool refuses an argument its closed schema does not name, as a
    // tool error the model can correct.
    int id = 100;
    for (const auto& d : table) {
        if (!d.agents) continue;  // no tool: the person's undo and redo
        const json r = h.request(id++, "tools/call",
                                 modern({{"name", command_tool_name(d.name)}, {"arguments", {{"no_such_argument", 1}}}}));
        CAPTURE(d.name);
        REQUIRE(r.contains("result"));
        CHECK(r["result"]["resultType"] == "complete");
        CHECK(tool_error(r)["code"] == "command.invalid_args");
    }

    // Valid arguments with no app running: the API file is missing.
    const json r = h.request(4, "tools/call",
                             modern({{"name", first_tool}, {"arguments", table.front().examples.at(0)}}));
    CHECK(tool_error(r)["code"] == "mcp.no_api_file");
    const json no_args = h.request(5, "tools/call", modern({{"name", first_tool}}));
    REQUIRE(no_args.contains("result"));

    // A null on an optional argument is read as the argument left out, as the
    // command layer reads it, so this call a local model sent passes the check
    // and goes on to look for the app; a null on a required argument does not.
    const json nulls = {{"id", "layer0"}, {"catalog", "live:entropy.l0"}, {"sql", nullptr}, {"from", nullptr},
                        {"transform", nullptr}, {"dry_run", false}};
    CHECK(tool_error(h.request(6, "tools/call", modern({{"name", "source_add"}, {"arguments", nulls}})))["code"] ==
          "mcp.no_api_file");
    CHECK(tool_error(h.request(7, "tools/call",
                               modern({{"name", "source_add"}, {"arguments", {{"id", nullptr}, {"catalog", "x"}}}})))["code"] ==
          "command.invalid_args");
}

TEST_CASE("subscriptions are acknowledged with what the server honours and closed at shutdown") {
    Harness h;
    h.take();
    h.send({{"jsonrpc", "2.0"},
            {"id", "sub-1"},
            {"method", "subscriptions/listen"},
            {"params", modern({{"notifications", {{"toolsListChanged", true}, {"resourcesListChanged", true},
                                                  {"resourceSubscriptions", {"caliper://docs/index.md"}}}}})}});
    h.send({{"jsonrpc", "2.0"},
            {"id", "sub-2"},
            {"method", "subscriptions/listen"},
            {"params", modern({{"notifications", {{"promptsListChanged", true}}}})}});
    auto t = h.take();
    REQUIRE(t.size() == 2);
    CHECK(t[0]["method"] == "notifications/subscriptions/acknowledged");
    CHECK(t[0]["params"]["_meta"]["io.modelcontextprotocol/subscriptionId"] == "sub-1");
    CHECK(t[0]["params"]["notifications"] == json{{"toolsListChanged", true}});
    CHECK(t[1]["params"]["notifications"] == json::object());
    CHECK_FALSE(t[0].contains("id"));

    // A second request with a subscription's id is refused while it runs.
    const json dup = h.request("sub-1", "ping", modern());
    const json sub1 = "sub-1";
    check_error(dup, -32600, &sub1);

    // Cancelling sub-2 ends it without a response; sub-1 closes at shutdown.
    h.send({{"jsonrpc", "2.0"}, {"method", "notifications/cancelled"}, {"params", {{"requestId", "sub-2"}}}});
    h.take();
    h.server->shutdown();
    t = h.take();
    REQUIRE(t.size() == 1);
    CHECK(t[0]["id"] == "sub-1");
    CHECK(t[0]["result"]["resultType"] == "complete");
    CHECK(t[0]["result"]["_meta"]["io.modelcontextprotocol/subscriptionId"] == "sub-1");
    // After shutdown nothing more is answered.
    h.server->handle_line(json{{"jsonrpc", "2.0"}, {"id", 1}, {"method", "ping"}}.dump());
    CHECK(h.take().empty());
}

TEST_CASE("resources are listed in pages and read by URI") {
    const fs::path docs = fs::path(CALIPER_BUILDER_TEST_TMP) / "server-unit" / "docs";
    fs::remove_all(docs);
    fs::create_directories(docs / "many");
    for (int i = 0; i < 450; ++i) std::ofstream(docs / "many" / ("p" + std::to_string(1000 + i) + ".md")) << "# Page\n";
    std::ofstream(docs / "index.md") << "# Index\n";
    ServerOptions o;
    o.docs_dir = docs.string();
    Harness h(o);

    std::set<std::string> seen;
    json cursor;
    int pages = 0;
    for (;;) {
        json p = modern();
        if (!cursor.is_null()) p["cursor"] = cursor;
        const json r = h.request("page-" + std::to_string(pages), "resources/list", p);
        REQUIRE(r.contains("result"));
        CHECK(r["result"]["cacheScope"] == "public");
        for (const auto& e : r["result"]["resources"]) CHECK(seen.insert(e["uri"].get<std::string>()).second);
        ++pages;
        if (!r["result"].contains("nextCursor")) break;
        cursor = r["result"]["nextCursor"];
        REQUIRE(pages < 10);
    }
    CHECK(pages == 3);
    // 451 pages, the spec schema compiled in, the command table, and
    // the files of the built-in library of pieces.
    std::size_t library = 0;
    for (const std::string& uri : seen) library += uri.rfind(kLibraryUriPrefix, 0) == 0;
    CHECK(library == caliper_builder::programs::library_files().size());
    CHECK(seen.size() - library == 453);
    CHECK(seen.count(std::string(kLibraryUriPrefix) + "lib/colormap/1.wgsl") == 1);
    CHECK(seen.count(kCommandTableUri) == 1);
    CHECK(seen.count(kSpecSchemaUri) == 1);
    CHECK(seen.count("caliper://docs/index.md") == 1);

    for (const json& bad : {json("x"), json("-1"), json(5), json("99999")}) {
        const json r = h.request("bad-cursor", "resources/list", modern({{"cursor", bad}}));
        const json id = "bad-cursor";
        check_error(r, -32602, &id);
    }

    const json page = h.request("read-1", "resources/read", modern({{"uri", "caliper://docs/index.md"}}));
    CHECK(page["result"]["contents"][0]["text"] == "# Index\n");
    CHECK(page["result"]["contents"][0]["mimeType"] == "text/markdown");
    CHECK(page["result"]["resultType"] == "complete");

    const json table = h.request("read-2", "resources/read", modern({{"uri", kCommandTableUri}}));
    // The table the agent reads is its tools': without the person's undo and redo.
    json agents_table = command_table_to_json();
    json kept = json::array();
    for (const auto& c : agents_table["commands"])
        if (c["agents"] == true) kept.push_back(c);
    agents_table["commands"] = kept;
    CHECK(json::parse(table["result"]["contents"][0]["text"].get<std::string>()) == agents_table);
    const json schema = h.request("read-3", "resources/read", modern({{"uri", kSpecSchemaUri}}));
    const json parsed = json::parse(schema["result"]["contents"][0]["text"].get<std::string>(), nullptr, false);
    CHECK(parsed.is_object());
    CHECK(parsed.at("$id") == "urn:caliper:schema:spec:v2");
    CHECK(parsed.at("properties").at("caliper_spec").at("const") == 2);
    CHECK(parsed.at("$defs").contains("panel_scene"));
    CHECK(parsed.at("$defs").contains("control"));
    CHECK(schema["result"]["ttlMs"] == 3600000);

    const json missing = h.request("read-4", "resources/read", modern({{"uri", "caliper://docs/../index.md"}}));
    const json id4 = "read-4";
    check_error(missing, -32602, &id4);
    CHECK(missing["error"]["data"]["uri"] == "caliper://docs/../index.md");

    const json templates = h.request("tpl", "resources/templates/list", modern());
    CHECK(templates["result"]["resourceTemplates"].size() == 1);
    CHECK(templates["result"]["resourceTemplates"][0]["uriTemplate"] == "caliper://docs/{+path}");
}

TEST_CASE("a closed output stops the server from writing") {
    Harness h;
    h.open_sink = false;
    h.request(1, "ping", json::object());
    CHECK(h.server->output_closed());
    h.take();
    h.server->handle_line(json{{"jsonrpc", "2.0"}, {"id", 2}, {"method", "ping"}}.dump());
    CHECK(h.take().empty());
}

TEST_CASE("bad options are refused when the server opens") {
    ServerOptions o;
    o.origin = "ui";
    Server s(o, [](const std::string&) { return true; }, nullptr);
    CHECK(s.open().code == "mcp.bad_option");
    ServerOptions d;
    d.docs_dir = (fs::path(CALIPER_BUILDER_TEST_TMP) / "server-unit" / "missing-docs").string();
    Server s2(d, [](const std::string&) { return true; }, nullptr);
    CHECK(s2.open().code == "mcp.bad_resource_dir");
}
