// The MCP protocol engine of caliper-mcp (server.h).
#include <caliper_builder/mcp/server.h>

#include "codec.h"

#include <caliper_builder/mcp/api_file.h>
#include <caliper_builder/mcp/app_client.h>
#include <caliper_builder/mcp/resources.h>
#include <caliper_builder/mcp/tools.h>

#include <nlohmann/json.hpp>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <list>
#include <map>
#include <mutex>
#include <thread>

#ifdef _WIN32
#include <fcntl.h>
#include <io.h>
#include <stdio.h>
#else
#include <cerrno>
#include <csignal>
#include <unistd.h>
#endif

#ifndef CALIPER_MCP_VERSION
#define CALIPER_MCP_VERSION "0.1.0"
#endif

namespace caliper_builder::mcp {

namespace detail {
// The JSON Schema of the spec format, compiled in by the build
// (spec_schema_embed.cpp); empty when the build had none.
extern const char* const kSpecSchemaText;
}  // namespace detail

using nlohmann::json;

namespace {

constexpr int kParseError = -32700;
constexpr int kInvalidRequest = -32600;
constexpr int kMethodNotFound = -32601;
constexpr int kInvalidParams = -32602;
constexpr int kInternalError = -32603;
constexpr int kLegacyResourceNotFound = -32002;
constexpr int kUnsupportedProtocolVersion = -32022;

const char* const kMetaVersion = "io.modelcontextprotocol/protocolVersion";
const char* const kMetaClientCapabilities = "io.modelcontextprotocol/clientCapabilities";
const char* const kMetaServerInfo = "io.modelcontextprotocol/serverInfo";
const char* const kMetaSubscriptionId = "io.modelcontextprotocol/subscriptionId";

const char* const kInstructions =
    "caliper-mcp drives the Caliper app that runs on this machine, through Caliper API. app_info and app_instruments "
    "describe the app and its instruments, app_launch opens an instrument's page, and capture_frame returns the frame the app "
    "shows as a PNG. The other tools are the commands of Caliper's command layer: they read the data catalog, make "
    "loaders and build instruments written as specs. An app without a command layer answers them with the error "
    "mcp.no_command_layer. A command that deletes or overwrites saved data waits until the person approves it in the "
    "app, for up to two minutes. The resources hold the Caliper documentation, the SDK headers, the JSON Schema of "
    "the spec format and the command table.";

constexpr std::size_t kResourcePage = 200;
constexpr int kTtlStatic = 3600000;   // what changes only with caliper-mcp itself
constexpr int kTtlFiles = 60000;      // the docs and SDK folders
constexpr int kTtlApp = 0;            // what depends on the running app

std::string dump(const json& j) { return j.dump(-1, ' ', false, json::error_handler_t::replace); }

json server_info() { return {{"name", "caliper-mcp"}, {"title", "Caliper"}, {"version", CALIPER_MCP_VERSION}}; }

json capabilities() { return {{"tools", {{"listChanged", true}}}, {"resources", json::object()}}; }

bool valid_id(const json& id) { return id.is_string() || id.is_number_integer(); }

std::string id_key(const json& id) { return dump(id); }

// What a handler gives back: a result, or a JSON-RPC error.
struct Reply {
    bool is_error = false;
    int code = 0;
    std::string message;
    json data;
    json result;
    int ttl_ms = -1;   // -1: not a cacheable result

    static Reply ok(json r, int ttl = -1) {
        Reply x;
        x.result = std::move(r);
        x.ttl_ms = ttl;
        return x;
    }
    static Reply error(int code, std::string message, json data = nullptr) {
        Reply x;
        x.is_error = true;
        x.code = code;
        x.message = std::move(message);
        x.data = std::move(data);
        return x;
    }
};

// One request in its era.
struct Ctx {
    json id;
    bool modern = false;
};

}  // namespace

const std::vector<std::string>& supported_protocol_versions() {
    static const std::vector<std::string> v = {"2026-07-28", "2025-11-25", "2025-06-18"};
    return v;
}

const char* server_version() { return CALIPER_MCP_VERSION; }

struct Server::Impl {
    ServerOptions opt;
    Sink sink;
    Log log;
    std::unique_ptr<AppClient> app;
    ResourceStore resources;

    mutable std::mutex out_mu;   // the sink
    bool out_closed = false;

    mutable std::mutex mu;   // everything below
    struct Flight {
        json id;
        std::shared_ptr<std::atomic<bool>> cancel;    // stops the work: the client cancelled, or the server stops
        std::shared_ptr<std::atomic<bool>> dropped;   // the client cancelled, so no answer is sent
    };
    std::map<std::string, Flight> flights;
    struct Worker {
        std::thread thread;
        std::shared_ptr<std::atomic<bool>> done;
    };
    std::list<Worker> workers;
    struct Subscription {
        json id;
        bool tools = false;
    };
    std::map<std::string, Subscription> subscriptions;
    bool legacy = false;          // an initialize was answered
    std::string legacy_version;
    bool legacy_ready = false;    // notifications/initialized arrived
    bool stopping = false;
    std::string last_fingerprint;

    // ---- output ----------------------------------------------------------------------------

    void diag(std::string line) {
        if (app)
            for (const auto& t : app->known_tokens()) detail::redact(&line, t);
        if (log) log(line);
    }

    void send(const json& message) {
        std::string line = dump(message);
        if (app)
            for (const auto& t : app->known_tokens()) detail::redact(&line, t);
        std::lock_guard<std::mutex> g(out_mu);
        if (out_closed) return;
        if (!sink(line)) out_closed = true;
    }

    void send_result(const json& id, json result) {
        send({{"jsonrpc", "2.0"}, {"id", id}, {"result", std::move(result)}});
    }

    void send_error(const json* id, int code, const std::string& message, const json& data = nullptr) {
        json e = {{"code", code}, {"message", message}};
        if (!data.is_null()) e["data"] = data;
        json m = {{"jsonrpc", "2.0"}, {"error", std::move(e)}};
        if (id) m["id"] = *id;
        send(m);
    }

    json decorate(json result, const Ctx& c, int ttl_ms) const {
        if (!c.modern) return result;
        result["resultType"] = "complete";
        result["_meta"][kMetaServerInfo] = server_info();
        if (ttl_ms >= 0) {
            result["ttlMs"] = ttl_ms;
            result["cacheScope"] = "public";
        }
        return result;
    }

    void answer(const Ctx& c, const Reply& r) {
        if (r.is_error) send_error(&c.id, r.code, r.message, r.data);
        else send_result(c.id, decorate(r.result, c, r.ttl_ms));
    }

    // ---- workers ---------------------------------------------------------------------------

    void reap_locked() {
        for (auto it = workers.begin(); it != workers.end();) {
            if (it->done->load()) {
                it->thread.join();
                it = workers.erase(it);
            } else {
                ++it;
            }
        }
    }

    void run_async(const Ctx& c, std::function<Reply(const std::atomic<bool>&)> fn) {
        auto cancel = std::make_shared<std::atomic<bool>>(false);
        auto dropped = std::make_shared<std::atomic<bool>>(false);
        auto done = std::make_shared<std::atomic<bool>>(false);
        {
            std::lock_guard<std::mutex> g(mu);
            reap_locked();
            if (stopping) return;
            if (flights.size() >= opt.max_in_flight) {
                // Answered below, outside the lock.
            } else {
                const std::string key = id_key(c.id);
                flights[key] = {c.id, cancel, dropped};
                workers.push_back({std::thread([this, c, fn = std::move(fn), cancel, dropped, done, key] {
                                       Reply r = fn(*cancel);
                                       bool drop;
                                       {
                                           std::lock_guard<std::mutex> g2(mu);
                                           drop = dropped->load();
                                           flights.erase(key);
                                       }
                                       if (!drop) answer(c, r);
                                       done->store(true);
                                   }),
                                   done});
                return;
            }
        }
        send_error(&c.id, kInternalError,
                   "Too many requests are running: at most " + std::to_string(opt.max_in_flight) + " run at once.");
    }

    // ---- the command table and its changes ---------------------------------------------------

    void note_table(const std::shared_ptr<const ToolTable>& t, bool notify) {
        std::vector<json> targets;
        bool legacy_target = false;
        {
            std::lock_guard<std::mutex> g(mu);
            const bool changed = !last_fingerprint.empty() && last_fingerprint != t->fingerprint;
            last_fingerprint = t->fingerprint;
            if (!changed || !notify) return;
            legacy_target = legacy && legacy_ready;
            for (const auto& s : subscriptions)
                if (s.second.tools) targets.push_back(s.second.id);
        }
        diag("the tool list changed (" + t->source + " table, " + std::to_string(t->commands.size()) + " commands)");
        if (legacy_target) send({{"jsonrpc", "2.0"}, {"method", "notifications/tools/list_changed"}});
        for (const auto& id : targets)
            send({{"jsonrpc", "2.0"},
                  {"method", "notifications/tools/list_changed"},
                  {"params", {{"_meta", {{kMetaSubscriptionId, id}}}}}});
    }

    // ---- handlers --------------------------------------------------------------------------

    Reply discover() {
        return Reply::ok({{"supportedVersions", supported_protocol_versions()},
                          {"capabilities", capabilities()},
                          {"instructions", kInstructions}},
                         kTtlStatic);
    }

    Reply initialize(const json& params) {
        const auto v = params.find("protocolVersion");
        if (v == params.end() || !v->is_string())
            return Reply::error(kInvalidParams, "initialize needs params.protocolVersion, a string.");
        const std::string asked = v->get<std::string>();
        std::string chosen = "2025-11-25";
        for (const auto& s : supported_protocol_versions())
            if (s == asked && s != kModernProtocolVersion) chosen = s;
        {
            std::lock_guard<std::mutex> g(mu);
            legacy = true;
            legacy_version = chosen;
        }
        diag("initialize: a legacy client asked for " + asked.substr(0, 32) + "; serving " + chosen);
        return Reply::ok({{"protocolVersion", chosen},
                          {"capabilities", capabilities()},
                          {"serverInfo", server_info()},
                          {"instructions", kInstructions}});
    }

    Reply tools_list(const json& params) {
        if (params.contains("cursor")) return Reply::error(kInvalidParams, "The cursor is not one this server gave.");
        auto t = app->table();
        note_table(t, false);
        return Reply::ok({{"tools", t->tools}}, kTtlApp);
    }

    Reply tools_call(const json& params, const std::atomic<bool>& cancel) {
        const auto name = params.find("name");
        if (name == params.end() || !name->is_string())
            return Reply::error(kInvalidParams, "tools/call needs params.name, a string.");
        json args = json::object();
        const auto a = params.find("arguments");
        if (a != params.end() && !a->is_null()) {
            if (!a->is_object()) return Reply::error(kInvalidParams, "params.arguments must be a JSON object.");
            args = *a;
        }
        auto t = app->table();
        note_table(t, true);
        const std::string& tool = name->get_ref<const std::string&>();
        const CommandDescriptor* d = t->find_tool(tool);
        if (!d) {
            const std::string shown = tool.size() <= 128 ? tool : tool.substr(0, 128) + "...";
            return Reply::error(kInvalidParams, "Unknown tool: " + shown, json{{"name", shown}});
        }
        const auto t0 = std::chrono::steady_clock::now();
        json result = app->call(*d, args, &cancel);
        const auto ms =
            std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - t0).count();
        std::string outcome = "ok";
        if (result.value("isError", false)) {
            const json e = json::parse(result["content"][0].value("text", std::string("{}")), nullptr, false);
            outcome = e.is_object() && e.contains("error") ? e["error"].value("code", std::string("error")) : "error";
        }
        diag("tools/call " + tool + ": " + outcome + " in " + std::to_string(ms) + " ms");
        return Reply::ok(std::move(result));
    }

    Reply resources_list(const json& params) {
        std::size_t offset = 0;
        const auto cursor = params.find("cursor");
        if (cursor != params.end()) {
            bool good = cursor->is_string() && !cursor->get_ref<const std::string&>().empty() &&
                        cursor->get_ref<const std::string&>().size() <= 9;
            if (good)
                for (char ch : cursor->get_ref<const std::string&>())
                    if (ch < '0' || ch > '9') good = false;
            if (!good) return Reply::error(kInvalidParams, "The cursor is not one this server gave.");
            offset = std::stoul(cursor->get<std::string>());
        }
        bool truncated = false;
        std::vector<ResourceEntry> entries = resources.list(&truncated);
        if (truncated) diag("a resource folder holds more files than are listed");
        ResourceEntry table;
        table.uri = kCommandTableUri;
        table.name = "command_table.json";
        table.title = "Command table";
        table.description =
            "The command descriptors the tools are made from: the app's table when the app serves one, otherwise the "
            "table built into caliper-mcp.";
        table.mime_type = "application/json";
        entries.push_back(table);
        std::sort(entries.begin(), entries.end(),
                  [](const ResourceEntry& x, const ResourceEntry& y) { return x.uri < y.uri; });
        if (offset > entries.size() || (offset == entries.size() && offset != 0))
            return Reply::error(kInvalidParams, "The cursor is not one this server gave.");
        json list = json::array();
        const std::size_t end = std::min(entries.size(), offset + kResourcePage);
        for (std::size_t i = offset; i < end; ++i) list.push_back(entries[i].to_json());
        json result = {{"resources", std::move(list)}};
        if (end < entries.size()) result["nextCursor"] = std::to_string(end);
        return Reply::ok(std::move(result), kTtlFiles);
    }

    Reply resources_read(const json& params, bool modern) {
        const auto uri = params.find("uri");
        if (uri == params.end() || !uri->is_string())
            return Reply::error(kInvalidParams, "resources/read needs params.uri, a string.");
        const std::string& u = uri->get_ref<const std::string&>();
        if (u == kCommandTableUri) {
            auto t = app->table();
            note_table(t, true);
            json c = {{"uri", u}, {"mimeType", "application/json"}, {"text", t->to_json().dump(2, ' ', false, json::error_handler_t::replace)}};
            return Reply::ok({{"contents", json::array({std::move(c)})}}, kTtlApp);
        }
        ResourceContent rc = resources.read(u);
        const std::string shown = u.size() <= 1024 ? u : u.substr(0, 1024);
        if (rc.error.code == "mcp.resource_not_found")
            return Reply::error(modern ? kInvalidParams : kLegacyResourceNotFound, "Resource not found",
                                json{{"uri", shown}, {"reason", rc.error.message}});
        if (rc.error) return Reply::error(kInternalError, rc.error.message, json{{"uri", shown}});
        json c = {{"uri", u}, {"mimeType", rc.mime_type}, {"text", std::move(rc.text)}};
        return Reply::ok({{"contents", json::array({std::move(c)})}}, u == kSpecSchemaUri ? kTtlStatic : kTtlFiles);
    }

    Reply templates_list(const json& params) {
        if (params.contains("cursor")) return Reply::error(kInvalidParams, "The cursor is not one this server gave.");
        return Reply::ok({{"resourceTemplates", resources.templates()}}, kTtlStatic);
    }

    void subscribe(const Ctx& c, const json& params) {
        const auto n = params.find("notifications");
        if (n != params.end() && !n->is_object()) {
            send_error(&c.id, kInvalidParams, "params.notifications must be an object.");
            return;
        }
        bool tools = false;
        if (n != params.end()) {
            const auto t = n->find("toolsListChanged");
            tools = t != n->end() && t->is_boolean() && t->get<bool>();
        }
        bool registered = false;
        {
            std::lock_guard<std::mutex> g(mu);
            if (stopping) return;
            if (subscriptions.size() < opt.max_subscriptions) {
                subscriptions[id_key(c.id)] = {c.id, tools};
                registered = true;
            }
        }
        if (!registered) {
            send_error(&c.id, kInternalError,
                       "Too many subscriptions: at most " + std::to_string(opt.max_subscriptions) + " at once.");
            return;
        }
        // Only the tool list can change; the other kinds are left out of the
        // acknowledged filter, as the revision asks.
        json agreed = json::object();
        if (tools) agreed["toolsListChanged"] = true;
        send({{"jsonrpc", "2.0"},
              {"method", "notifications/subscriptions/acknowledged"},
              {"params", {{"_meta", {{kMetaSubscriptionId, c.id}}}, {"notifications", agreed}}}});
    }

    // ---- dispatch --------------------------------------------------------------------------

    void request(const json& id, const std::string& method, const json& params) {
        Ctx c;
        c.id = id;
        bool busy_id;
        {
            std::lock_guard<std::mutex> g(mu);
            const std::string key = id_key(id);
            busy_id = flights.count(key) > 0 || subscriptions.count(key) > 0;
        }
        if (busy_id) {
            send_error(&id, kInvalidRequest, "A request with this id is still running.");
            return;
        }
        const json* meta = nullptr;
        const auto m = params.find("_meta");
        if (m != params.end() && m->is_object()) meta = &*m;
        const bool has_version = meta && meta->contains(kMetaVersion);
        if (has_version) {
            const json& v = (*meta)[kMetaVersion];
            if (!v.is_string()) {
                send_error(&id, kInvalidParams, std::string(kMetaVersion) + " must be a string.");
                return;
            }
            if (v.get<std::string>() != kModernProtocolVersion) {
                send_error(&id, kUnsupportedProtocolVersion, "Unsupported protocol version",
                           json{{"supported", supported_protocol_versions()}, {"requested", v.get<std::string>().substr(0, 64)}});
                return;
            }
            const auto caps = meta->find(kMetaClientCapabilities);
            if (caps == meta->end() || !caps->is_object()) {
                send_error(&id, kInvalidParams,
                           std::string("A request of protocol version ") + kModernProtocolVersion + " needs " +
                               kMetaClientCapabilities + " in params._meta.");
                return;
            }
            c.modern = true;
        } else if (method == "initialize") {
            answer(c, initialize(params));
            return;
        } else if (method == "ping") {
            answer(c, Reply::ok(json::object()));
            return;
        } else {
            bool is_legacy;
            {
                std::lock_guard<std::mutex> g(mu);
                is_legacy = legacy;
            }
            if (!is_legacy) {
                send_error(&id, kInvalidParams,
                           std::string("The request has no ") + kMetaVersion +
                               " in params._meta, and no initialize came before it. Supported versions: 2026-07-28 "
                               "(per request), 2025-11-25 and 2025-06-18 (through initialize).",
                           json{{"supported", supported_protocol_versions()}});
                return;
            }
        }

        if (method == "ping") {
            answer(c, Reply::ok(json::object()));
        } else if (method == "server/discover" && c.modern) {
            answer(c, discover());
        } else if (method == "subscriptions/listen" && c.modern) {
            subscribe(c, params);
        } else if (method == "tools/list") {
            run_async(c, [this, params](const std::atomic<bool>&) { return tools_list(params); });
        } else if (method == "tools/call") {
            run_async(c, [this, params](const std::atomic<bool>& cancel) { return tools_call(params, cancel); });
        } else if (method == "resources/list") {
            run_async(c, [this, params](const std::atomic<bool>&) { return resources_list(params); });
        } else if (method == "resources/read") {
            const bool modern = c.modern;
            run_async(c, [this, params, modern](const std::atomic<bool>&) { return resources_read(params, modern); });
        } else if (method == "resources/templates/list") {
            answer(c, templates_list(params));
        } else {
            const std::string shown = method.size() <= 128 ? method : method.substr(0, 128) + "...";
            send_error(&id, kMethodNotFound, "Method not found: " + shown);
        }
    }

    void notification(const std::string& method, const json& params) {
        if (method == "notifications/initialized") {
            std::lock_guard<std::mutex> g(mu);
            legacy_ready = true;
            return;
        }
        if (method == "notifications/cancelled") {
            const auto rid = params.find("requestId");
            if (rid == params.end() || !valid_id(*rid)) return;
            const std::string key = id_key(*rid);
            std::lock_guard<std::mutex> g(mu);
            const auto f = flights.find(key);
            if (f != flights.end()) {
                f->second.dropped->store(true);
                f->second.cancel->store(true);
            }
            subscriptions.erase(key);
            return;
        }
        // Other notifications need nothing from this server.
    }

    void line(std::string_view text) {
        {
            std::lock_guard<std::mutex> g(mu);
            if (stopping) return;
        }
        const auto first = text.find_first_not_of(" \t\r");
        if (first == std::string_view::npos) return;
        if (!detail::json_depth_ok(text, 128)) {
            send_error(nullptr, kInvalidRequest, "The message nests deeper than 128 levels.");
            return;
        }
        const json j = json::parse(text.begin(), text.end(), nullptr, false);
        if (j.is_discarded()) {
            send_error(nullptr, kParseError, "Parse error: the line is not one JSON value in UTF-8.");
            return;
        }
        if (j.is_array()) {
            send_error(nullptr, kInvalidRequest, "Batches are not supported; send one message per line.");
            return;
        }
        if (!j.is_object()) {
            send_error(nullptr, kInvalidRequest, "A message is a JSON object.");
            return;
        }
        const auto id = j.find("id");
        const json* good_id = id != j.end() && valid_id(*id) ? &*id : nullptr;
        const auto rpc = j.find("jsonrpc");
        if (rpc == j.end() || *rpc != "2.0") {
            send_error(good_id, kInvalidRequest, "The message is not JSON-RPC 2.0.");
            return;
        }
        const auto method = j.find("method");
        if (method == j.end()) {
            if (j.contains("result") || j.contains("error")) {
                diag("a response arrived, but this server sends no requests; it is ignored");
                return;
            }
            send_error(good_id, kInvalidRequest, "The message has no method.");
            return;
        }
        if (!method->is_string()) {
            send_error(good_id, kInvalidRequest, "The method is not a string.");
            return;
        }
        const auto params = j.find("params");
        static const json empty = json::object();
        if (params != j.end() && !params->is_object()) {
            if (id != j.end()) send_error(good_id, kInvalidParams, "params must be a JSON object.");
            return;
        }
        const json& p = params != j.end() ? *params : empty;
        if (id == j.end()) {
            notification(method->get<std::string>(), p);
            return;
        }
        if (!good_id) {
            send_error(nullptr, kInvalidRequest, "A request id is a string or an integer.");
            return;
        }
        request(*good_id, method->get<std::string>(), p);
    }

    void stop() {
        {
            std::lock_guard<std::mutex> g(mu);
            stopping = true;
        }
        // Calls that finish within the grace period are answered as usual; a
        // client may close its input and still read the answers.
        const auto grace_end = std::chrono::steady_clock::now() + std::chrono::milliseconds(opt.shutdown_grace_ms);
        for (;;) {
            {
                std::lock_guard<std::mutex> g(mu);
                if (flights.empty()) break;
            }
            if (std::chrono::steady_clock::now() >= grace_end) break;
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
        std::list<Worker> joining;
        std::vector<json> closing;
        {
            std::lock_guard<std::mutex> g(mu);
            // The rest stop at once and answer with what their cancellation gives.
            for (auto& f : flights) f.second.cancel->store(true);
            joining.swap(workers);
            for (const auto& s : subscriptions) closing.push_back(s.second.id);
            subscriptions.clear();
        }
        for (auto& w : joining)
            if (w.thread.joinable()) w.thread.join();
        for (const auto& id : closing)
            send_result(id, {{"resultType", "complete"},
                             {"_meta", {{kMetaSubscriptionId, id}, {kMetaServerInfo, server_info()}}}});
    }
};

Server::Server(ServerOptions options, Sink sink, Log log) : impl_(std::make_unique<Impl>()) {
    impl_->opt = std::move(options);
    impl_->sink = std::move(sink);
    impl_->log = std::move(log);
    if (impl_->opt.api_file.empty()) impl_->opt.api_file = default_api_file_path();
    if (impl_->opt.spec_schema_text.empty() && detail::kSpecSchemaText) impl_->opt.spec_schema_text = detail::kSpecSchemaText;
    AppClientOptions ao;
    ao.api_file = impl_->opt.api_file;
    ao.origin = impl_->opt.origin;
    ao.command_timeout_ms = impl_->opt.command_timeout_ms;
    ao.app_timeout_ms = impl_->opt.app_timeout_ms;
    ao.builtin = impl_->opt.builtin_table;
    Impl* impl = impl_.get();
    impl_->app = std::make_unique<AppClient>(std::move(ao), [impl](const std::string& l) { impl->diag(l); });
}

Server::~Server() { shutdown(); }

Error Server::open() {
    if (impl_->opt.origin != "mcp" && impl_->opt.origin != "agent")
        return {"mcp.bad_option", "The origin is mcp or agent.", "--origin"};
    if (impl_->opt.command_timeout_ms < 1000)
        return {"mcp.bad_option", "The command time limit is at least one second.", "--command-timeout"};
    ResourceOptions ro;
    ro.docs_dir = impl_->opt.docs_dir;
    ro.sdk_dir = impl_->opt.sdk_dir;
    ro.spec_schema_text = impl_->opt.spec_schema_text;
    ro.library = true;  // the built-in library of pieces (design/programmable-graphics.md, Library Pieces)
    return impl_->resources.open(ro);
}

void Server::handle_line(std::string_view line) { impl_->line(line); }

void Server::handle_oversized_line() {
    impl_->send_error(nullptr, kInvalidRequest,
                      "The message is longer than " + std::to_string(impl_->opt.max_message_bytes) + " bytes.");
}

void Server::shutdown() { impl_->stop(); }

bool Server::output_closed() const {
    std::lock_guard<std::mutex> g(impl_->out_mu);
    return impl_->out_closed;
}

std::size_t Server::in_flight() const {
    std::lock_guard<std::mutex> g(impl_->mu);
    return impl_->flights.size();
}

// ---- the stdio loop -------------------------------------------------------------------------

namespace {

bool write_all(int fd, const std::string& data) {
    std::size_t done = 0;
    while (done < data.size()) {
#ifdef _WIN32
        const int n = _write(fd, data.data() + done, unsigned(std::min<std::size_t>(data.size() - done, 1u << 30)));
        if (n <= 0) return false;
#else
        const ssize_t n = ::write(fd, data.data() + done, data.size() - done);
        if (n < 0) {
            if (errno == EINTR) continue;
            return false;
        }
        if (n == 0) return false;
#endif
        done += std::size_t(n);
    }
    return true;
}

long read_some(int fd, char* buffer, std::size_t size) {
#ifdef _WIN32
    return _read(fd, buffer, unsigned(size));
#else
    for (;;) {
        const ssize_t n = ::read(fd, buffer, size);
        if (n < 0 && errno == EINTR) continue;
        return long(n);
    }
#endif
}

}  // namespace

int run_stdio_server(const ServerOptions& options) {
#ifdef _WIN32
    _setmode(_fileno(stdin), _O_BINARY);
    _setmode(_fileno(stdout), _O_BINARY);
#else
    std::signal(SIGPIPE, SIG_IGN);
#endif
    std::mutex err_mu;
    auto log = [&err_mu](const std::string& line) {
        std::lock_guard<std::mutex> g(err_mu);
        (void)write_all(2, "caliper-mcp: " + line + "\n");
    };
    Server server(
        options, [](const std::string& line) { return write_all(1, line + "\n"); }, log);
    if (Error e = server.open()) {
        log(e.message + (e.where.empty() ? std::string() : " (" + e.where + ")"));
        return 2;
    }
    log(std::string("serving MCP on standard input and output; version ") + server_version());

    const std::size_t max = options.max_message_bytes;
    std::string buf;
    bool dropping = false;
    std::vector<char> chunk(1u << 16);
    for (;;) {
        const long n = read_some(0, chunk.data(), chunk.size());
        if (n <= 0) break;
        std::size_t start = 0;
        for (std::size_t i = 0; i < std::size_t(n); ++i) {
            if (chunk[i] != '\n') continue;
            if (dropping) {
                dropping = false;
                server.handle_oversized_line();
            } else {
                buf.append(chunk.data() + start, i - start);
                if (buf.size() > max) {
                    server.handle_oversized_line();
                } else {
                    if (!buf.empty() && buf.back() == '\r') buf.pop_back();
                    server.handle_line(buf);
                }
            }
            buf.clear();
            start = i + 1;
        }
        if (!dropping) {
            buf.append(chunk.data() + start, std::size_t(n) - start);
            if (buf.size() > max) {
                dropping = true;
                buf.clear();
                buf.shrink_to_fit();
            }
        }
        if (server.output_closed()) break;
    }
    if (dropping) server.handle_oversized_line();
    else if (!buf.empty()) server.handle_line(buf);
    server.shutdown();
    log("standard input ended; stopping");
    return 0;
}

}  // namespace caliper_builder::mcp
