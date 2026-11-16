// How caliper-mcp reaches the running app (app_client.h).
#include <caliper_builder/mcp/app_client.h>

#include "codec.h"

#include <caliper_builder/commands/command_table.h>
#include <caliper_builder/commands/schema.h>

namespace caliper_builder::mcp {

using nlohmann::json;

namespace {

std::string dump(const json& j) { return j.dump(-1, ' ', false, json::error_handler_t::replace); }

// The message of an API error body {"error": "..."}, or the reason phrase.
std::string server_message(const HttpResponse& r) {
    if (r.body.size() <= 64u << 10 && detail::json_depth_ok(r.body, 16)) {
        const json j = json::parse(r.body, nullptr, false);
        if (j.is_object()) {
            const auto e = j.find("error");
            if (e != j.end() && e->is_string()) return e->get<std::string>().substr(0, 1000);
        }
    }
    return r.reason.substr(0, 200);
}

std::string seconds(int ms) {
    const int s = (ms + 999) / 1000;
    return std::to_string(s) + (s == 1 ? " second" : " seconds");
}

// A transport fault, or an API file fault, as a tool error.
Error transport_error(const Error& http, int timeout_ms) {
    const std::string& c = http.code;
    if (c == "http.connect_refused")
        return {"mcp.app_unreachable",
                "Caliper API does not answer on " + http.where +
                    ": the app named in the API file may have exited. Start the app with --api.",
                http.where};
    if (c == "http.connect_timeout" || c == "http.read_timeout" || c == "http.timeout")
        return {"mcp.timeout", "The app did not answer within " + seconds(timeout_ms) + ".", http.where};
    if (c == "http.cancelled") return {"mcp.cancelled", "The call was cancelled.", ""};
    if (c == "http.too_large") return {"mcp.response_too_large", http.message, http.where};
    if (c == "http.bad_response" || c == "http.redirect")
        return {"mcp.bad_response", "Caliper API gave an answer caliper-mcp does not take: " + http.message, http.where};
    return {"mcp.app_unreachable", "Caliper API could not be reached: " + http.message, http.where};
}

// The tool error of an HTTP status that is not success, common to every route.
Error status_error(const HttpResponse& r, const std::string& what) {
    const std::string msg = server_message(r);
    switch (r.status) {
        case 401:
            return {"mcp.unauthorized",
                    "Caliper API refused the session token from the API file; the app may have restarted. Retry the call.",
                    what};
        case 403: return {"mcp.forbidden", "Caliper API refused the request (403): " + msg, what};
        case 503: return {"mcp.app_busy", "The app is busy or closing (503): " + msg + ". Retry in a moment.", what};
        case 504:
            return {"mcp.app_timeout", "The app did not take the request in time (504): " + msg + ". Retry in a moment.",
                    what};
        default:
            return {"mcp.http_error", "Caliper API answered " + std::to_string(r.status) + ": " + msg, what};
    }
}

bool applet_id_ok(const std::string& id) {
    if (id.empty() || id.size() > 128) return false;
    for (char c : id) {
        const bool ok = (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9') || c == '.' ||
                        c == '_' || c == '-';
        if (!ok) return false;
    }
    return true;
}

}  // namespace

AppClient::AppClient(AppClientOptions options, Log log) : options_(std::move(options)), log_(std::move(log)) {
    if (!log_) log_ = [](const std::string&) {};
    std::vector<CommandDescriptor> builtin = options_.builtin.empty() ? command_table() : options_.builtin;
    ToolTableBuild b = build_tool_table(std::move(builtin), "built-in");
    for (const auto& d : b.dropped) log_("built-in command " + d.where + " left out: " + d.message);
    if (b.table) {
        builtin_ = b.table;
    } else {
        log_("the built-in command table is unusable: " + b.error.message);
        builtin_ = build_tool_table({}, "built-in").table;
    }
}

std::vector<std::string> AppClient::known_tokens() const {
    std::lock_guard<std::mutex> g(mu_);
    return {tokens_.begin(), tokens_.end()};
}

ApiFileRead AppClient::read_endpoint() {
    ApiFileRead f = read_api_file(options_.api_file);
    if (f.error.ok()) {
        std::lock_guard<std::mutex> g(mu_);
        tokens_.insert(f.endpoint.token);
    }
    return f;
}

AppClient::Sent AppClient::send(HttpRequest request, const HttpLimits& limits, const std::atomic<bool>* cancel) {
    Sent out;
    ApiFileRead f = read_endpoint();
    if (f.error) {
        out.file_error = f.error;
        return out;
    }
    const std::size_t plain_headers = request.headers.size();
    auto attempt = [&](const ApiEndpoint& ep) {
        request.headers.resize(plain_headers);
        request.headers.push_back({"Authorization", "Bearer " + ep.token});
        out.endpoint = ep;
        out.response = loopback_http_request("127.0.0.1", ep.port, request, limits, cancel);
    };
    attempt(f.endpoint);
    if (out.response.error.code == "http.connect_refused" || (out.response.ok() && out.response.status == 401)) {
        ApiFileRead again = read_endpoint();
        if (again.error.ok() && !again.endpoint.same_as(f.endpoint)) {
            log_("the API file names another port or token now; sending the request again");
            attempt(again.endpoint);
        }
    }
    return out;
}

std::shared_ptr<const ToolTable> AppClient::table(bool* changed) {
    std::lock_guard<std::mutex> refresh(table_mu_);
    std::shared_ptr<const ToolTable> chosen;
    std::string note;
    const ApiFileRead f = read_endpoint();
    if (f.error) {
        chosen = builtin_;
        note = "the built-in command table, since the app is not running";
    } else {
        {
            std::lock_guard<std::mutex> g(mu_);
            if (have_cached_ && cached_endpoint_.same_as(f.endpoint)) chosen = cached_table_;
        }
        if (!chosen) {
            HttpRequest req;
            req.method = "GET";
            req.target = "/v0/commands";
            HttpLimits limits;
            limits.read_timeout_ms = limits.total_timeout_ms = options_.table_timeout_ms;
            limits.max_body_bytes = options_.max_table_bytes;
            Sent s = send(req, limits, nullptr);
            bool cache = false;
            if (s.file_error) {
                chosen = builtin_;
                note = "the built-in command table, since the app is not running";
            } else if (s.response.error) {
                chosen = builtin_;
                note = "the built-in command table, since GET /v0/commands failed: " + s.response.error.message;
            } else if (s.response.status == 404) {
                chosen = builtin_;
                cache = true;
                note = "the built-in command table, since this app has no command layer yet";
            } else if (s.response.status != 200) {
                chosen = builtin_;
                note = "the built-in command table, since GET /v0/commands answered " +
                       std::to_string(s.response.status);
            } else {
                const std::string& body = s.response.body;
                const json j = detail::json_depth_ok(body, 128) ? json::parse(body, nullptr, false) : json();
                ToolTableBuild b = j.is_discarded() || j.is_null()
                                       ? ToolTableBuild{nullptr, {}, {"mcp.bad_command_table", "it is not JSON", ""}}
                                       : parse_app_command_table(j);
                for (const auto& d : b.dropped) log_("the app's command " + d.where + " is left out: " + d.message);
                cache = true;
                if (b.table) {
                    chosen = b.table;
                    note = "the app's command table";
                } else {
                    chosen = builtin_;
                    note = "the built-in command table, since the app's is malformed: " + b.error.message;
                }
            }
            if (cache) {
                std::lock_guard<std::mutex> g(mu_);
                have_cached_ = true;
                cached_endpoint_ = s.endpoint;
                cached_table_ = chosen;
            }
        }
    }
    bool say = false;
    {
        std::lock_guard<std::mutex> g(mu_);
        if (!note.empty() && note != last_source_note_) {
            say = true;
            last_source_note_ = note;
        }
        if (changed) *changed = !last_fingerprint_.empty() && last_fingerprint_ != chosen->fingerprint;
        last_fingerprint_ = chosen->fingerprint;
    }
    // The log may ask for known_tokens(), which takes mu_.
    if (say) log_("tools come from " + note + " (" + std::to_string(chosen->commands.size()) + " commands)");
    return chosen;
}

json AppClient::call(const CommandDescriptor& d, const json& args, const std::atomic<bool>* cancel) {
    if (Error err = validate_command_value(d.args, args, "args")) return tool_error_result(err);
    const std::string body = dump(args);
    if (body.size() > (1u << 20))
        return tool_error_result({"command.invalid_args", "The arguments exceed 1 MiB as JSON text.", "args"});

    HttpRequest req;
    HttpLimits limits;
    limits.max_body_bytes = options_.max_response_bytes;
    limits.read_timeout_ms = limits.total_timeout_ms = options_.app_timeout_ms;
    enum class Route { info, applets, launch, capture, command } route = Route::command;
    if (d.name == "app.info") {
        route = Route::info;
        req.target = "/v0/info";
    } else if (d.name == "app.applets") {
        route = Route::applets;
        req.target = "/v0/applets";
    } else if (d.name == "app.launch") {
        route = Route::launch;
        const auto id = args.find("id");
        if (id == args.end() || !id->is_string() || !applet_id_ok(id->get<std::string>()))
            return tool_error_result({"command.invalid_args",
                                      "An applet id is 1 to 128 characters of A-Z, a-z, 0-9, '.', '_' and '-'.",
                                      "args.id"});
        req.method = "POST";
        req.target = "/v0/applets/" + id->get<std::string>() + "/launch";
    } else if (d.name == "capture.frame") {
        route = Route::capture;
        req.target = "/v0/capture.png";
    } else {
        req.method = "POST";
        req.target = "/v0/commands/" + d.name;
        req.body = body;
        req.headers.push_back({"Content-Type", "application/json"});
        req.headers.push_back({"X-Caliper-Origin", options_.origin});
        limits.read_timeout_ms = limits.total_timeout_ms = options_.command_timeout_ms;
    }

    Sent s = send(req, limits, cancel);
    if (s.file_error) return tool_error_result(s.file_error);
    if (s.response.error) return tool_error_result(transport_error(s.response.error, limits.total_timeout_ms));
    const HttpResponse& r = s.response;

    if (route == Route::capture) {
        if (r.status == 501)
            return tool_error_result({"capture.unavailable",
                                      "The app cannot capture its frame on this renderer: it reads its window back "
                                      "only on Metal, and answers 501 on the GL and Vulkan renderers (" +
                                          server_message(r) + ").",
                                      "GET /v0/capture.png"});
        if (r.status != 200) return tool_error_result(status_error(r, "GET /v0/capture.png"));
        std::uint32_t w = 0, h = 0;
        if (!detail::png_size(r.body, &w, &h))
            return tool_error_result({"mcp.bad_response", "The capture is not a PNG.", "GET /v0/capture.png"});
        json result = {{"png_base64", detail::base64_encode(r.body)}, {"width", w}, {"height", h}};
        return tool_result_from_command(d, result);
    }

    if (route == Route::command && r.status == 404)
        return tool_error_result({"mcp.no_command_layer",
                                  "This version of the app has no command layer yet, so " + d.name +
                                      " cannot run. Only app_info, app_applets, app_launch and capture_frame work "
                                      "against it.",
                                  "POST /v0/commands/" + d.name});
    if (route == Route::launch && r.status != 200) {
        const std::string msg = server_message(r);
        if (r.status == 404) return tool_error_result({"app.applet_not_found", msg, "args.id"});
        if (r.status == 409) return tool_error_result({"app.launch_refused", msg, "args.id"});
        if (r.status == 400) return tool_error_result({"command.invalid_args", msg, "args.id"});
    }
    if (r.status != 200) return tool_error_result(status_error(r, req.method + " " + req.target));

    if (!detail::json_depth_ok(r.body, 128))
        return tool_error_result({"mcp.bad_response", "The app's answer nests too deeply.", req.target});
    const json j = json::parse(r.body, nullptr, false);
    if (j.is_discarded()) return tool_error_result({"mcp.bad_response", "The app's answer is not JSON.", req.target});
    if (route != Route::command) return tool_result_from_command(d, j);

    CommandResultParse env = command_result_from_json(j);
    if (env.error)
        return tool_error_result(
            {"mcp.bad_response", "The app's answer is not a command result: " + env.error.message, req.target});
    if (!env.result.ok) return tool_error_result(env.result.error, env.result.approval);
    return tool_result_from_command(d, env.result.result);
}

}  // namespace caliper_builder::mcp
