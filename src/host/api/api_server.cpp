// Caliper API v0 (api_server.h): the routes, the checks of a plain request,
// and the server around an embedded hub.
#include "api/api_server.h"

#include "net/json.h"
#include "net/wire.h"

#include <algorithm>
#include <atomic>
#include <cstring>
#include <condition_variable>
#include <deque>
#include <map>
#include <mutex>
#include <thread>
#include <utility>

namespace caliper_api {

using caliper_net::ConnId;
using caliper_net::HttpRequest;

namespace {

const char kPrefix[] = "/v0/";
const char kLaunchHead[] = "/v0/applets/";
const char kLaunchTail[] = "/launch";
const char kCommands[] = "/v0/commands";
const char kCommandHead[] = "/v0/commands/";
const char kApprovalHead[] = "/v0/approvals/";

bool starts_with(const std::string& s, const char* p) {
    const size_t n = std::char_traits<char>::length(p);
    return s.size() >= n && s.compare(0, n, p) == 0;
}

bool ends_with(const std::string& s, const char* p) {
    const size_t n = std::char_traits<char>::length(p);
    return s.size() >= n && s.compare(s.size() - n, n, p) == 0;
}

std::string lower(std::string s) {
    for (char& c : s)
        if (c >= 'A' && c <= 'Z') c = char(c - 'A' + 'a');
    return s;
}

std::string trim(const std::string& s) {
    size_t a = 0, b = s.size();
    while (a < b && (s[a] == ' ' || s[a] == '\t')) ++a;
    while (b > a && (s[b - 1] == ' ' || s[b - 1] == '\t')) --b;
    return s.substr(a, b - a);
}

// The part of a path after `head`, when it is 1 to 128 characters without a '/'.
bool tail_after(const std::string& path, const char* head, std::string* tail) {
    if (!starts_with(path, head)) return false;
    const size_t a = std::char_traits<char>::length(head);
    if (path.size() == a || path.size() - a > 128 || path.find('/', a) != std::string::npos) return false;
    *tail = path.substr(a);
    return true;
}

size_t header_count(const HttpRequest& r, const char* name) {
    size_t n = 0;
    for (const auto& h : r.headers)
        if (h.first == name) ++n;
    return n;
}

// ---- the shape of a JSON body, without building it ----------------------------------------
// A recursive descent over RFC 8259 that keeps nothing. The caller has checked
// that the text is UTF-8; depth stops at kMaxJsonDepth + 1.
struct ShapeReader {
    const char* p;
    const char* end;
    bool too_deep = false;

    void ws() {
        while (p < end && (*p == ' ' || *p == '\t' || *p == '\n' || *p == '\r')) ++p;
    }
    bool lit(const char* w) {
        const size_t n = std::char_traits<char>::length(w);
        if ((size_t)(end - p) < n || std::char_traits<char>::compare(p, w, n) != 0) return false;
        p += n;
        return true;
    }
    static int hex(char c) {
        if (c >= '0' && c <= '9') return c - '0';
        if (c >= 'a' && c <= 'f') return c - 'a' + 10;
        if (c >= 'A' && c <= 'F') return c - 'A' + 10;
        return -1;
    }
    // The four hex digits of a u escape, after the backslash and the 'u'; -1 when malformed.
    long u4() {
        if (end - p < 4) return -1;
        long v = 0;
        for (int i = 0; i < 4; ++i) {
            const int h = hex(p[i]);
            if (h < 0) return -1;
            v = v * 16 + h;
        }
        p += 4;
        return v;
    }
    bool string() {
        ++p;   // the opening quote
        while (p < end) {
            const unsigned char c = (unsigned char)*p++;
            if (c == '"') return true;
            if (c < 0x20) return false;
            if (c != '\\') continue;
            if (p == end) return false;
            const char e = *p++;
            if (e == 'u') {
                const long v = u4();
                if (v < 0 || (v >= 0xDC00 && v <= 0xDFFF)) return false;
                if (v >= 0xD800 && v <= 0xDBFF) {
                    if (end - p < 2 || p[0] != '\\' || p[1] != 'u') return false;
                    p += 2;
                    const long lo = u4();
                    if (lo < 0xDC00 || lo > 0xDFFF) return false;
                }
            } else if (!std::strchr("\"\\/bfnrt", e) || e == 0) {
                return false;
            }
        }
        return false;
    }
    bool digits() {
        const char* s0 = p;
        while (p < end && *p >= '0' && *p <= '9') ++p;
        return p > s0;
    }
    bool number() {
        if (p < end && *p == '-') ++p;
        if (p < end && *p == '0') ++p;
        else if (!(p < end && *p >= '1' && *p <= '9') || !digits()) return false;
        if (p < end && *p == '.') {
            ++p;
            if (!digits()) return false;
        }
        if (p < end && (*p == 'e' || *p == 'E')) {
            ++p;
            if (p < end && (*p == '+' || *p == '-')) ++p;
            if (!digits()) return false;
        }
        return true;
    }
    bool value(int depth) {
        ws();
        if (p == end) return false;
        const char c = *p;
        if (c == '{' || c == '[') {
            if (depth + 1 > kMaxJsonDepth) {
                too_deep = true;
                return false;
            }
            const char close = c == '{' ? '}' : ']';
            ++p;
            ws();
            if (p < end && *p == close) {
                ++p;
                return true;
            }
            for (;;) {
                if (c == '{') {
                    ws();
                    if (p == end || *p != '"' || !string()) return false;
                    ws();
                    if (p == end || *p != ':') return false;
                    ++p;
                }
                if (!value(depth + 1)) return false;
                ws();
                if (p == end) return false;
                if (*p == close) {
                    ++p;
                    return true;
                }
                if (*p != ',') return false;
                ++p;
            }
        }
        if (c == '"') return string();
        if (c == 't') return lit("true");
        if (c == 'f') return lit("false");
        if (c == 'n') return lit("null");
        return number();
    }
};

}  // namespace

// ---- routing ------------------------------------------------------------------------------

Routed route(const std::string& method, const std::string& path) {
    Routed r;
    if (path == "/v0/info") {
        r.route = Route::Info;
    } else if (path == "/v0/applets") {
        r.route = Route::Applets;
    } else if (path == "/v0/capture.png") {
        r.route = Route::Capture;
    } else if (path == kStreamPath || path == kStreamAlias) {
        r.route = Route::Stream;
    } else if (path == kCommands) {
        r.route = Route::Commands;
    } else if (path == kSpecPagePath) {
        r.route = Route::SpecPage;
    } else if (tail_after(path, kCommandHead, &r.name)) {
        r.route = Route::Command;
    } else if (tail_after(path, kApprovalHead, &r.name)) {
        r.route = Route::Approval;
    } else if (starts_with(path, kLaunchHead) && ends_with(path, kLaunchTail)) {
        const size_t a = sizeof kLaunchHead - 1;
        const size_t b = path.size() - (sizeof kLaunchTail - 1);
        if (b > a && path.find('/', a) == b) {
            r.route = Route::Launch;
            r.applet_id = path.substr(a, b - a);
        }
    }
    if (r.route == Route::None) return r;
    r.allow = r.route == Route::Launch || r.route == Route::Command ? "POST" : "GET";
    r.method_ok = method == r.allow;
    return r;
}

bool viewer_command(const std::string& name) {
    for (const char* n : kViewerCommands)
        if (name == n) return true;
    return false;
}

bool plausible_applet_id(const std::string& id) {
    if (id.empty() || id.size() > 128) return false;
    for (char c : id)
        if (!((c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9') || c == '.' ||
              c == '_' || c == '-'))
            return false;
    return true;
}

bool plausible_approval_id(const std::string& id) {
    if (id.empty() || id.size() > 128) return false;
    for (char c : id)
        if (!((c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9') || c == '.' ||
              c == '_' || c == ':' || c == '-'))
            return false;
    return true;
}

bool token_ok(const HttpRequest& r, const std::string& token, bool query_ok) {
    if (token.empty()) return false;
    bool has_header = false;
    for (const auto& h : r.headers)
        if (h.first == "authorization") has_header = true;
    if (has_header) {
        // "Bearer <token>", the scheme in any case. Only the first header counts.
        const std::string v = trim(r.header("authorization"));
        const size_t sp = v.find_first_of(" \t");
        if (sp == std::string::npos || lower(v.substr(0, sp)) != "bearer") return false;
        return caliper_net::wire::token_equal(trim(v.substr(sp + 1)), token);
    }
    std::string q;
    if (!query_ok || !r.query_param(CALIPER_NET_TOKEN_PARAM, &q)) return false;
    return caliper_net::wire::token_equal(q, token);
}

std::string TokenTable::find(const HttpRequest& r, bool query_ok) const {
    // The token the request presents: the first Authorization header's
    // "Bearer <token>", else, when query_ok, the query parameter.
    std::string presented;
    bool has_header = false;
    for (const auto& h : r.headers)
        if (h.first == "authorization") has_header = true;
    if (has_header) {
        const std::string v = trim(r.header("authorization"));
        const size_t sp = v.find_first_of(" \t");
        if (sp == std::string::npos || lower(v.substr(0, sp)) != "bearer") return {};
        presented = trim(v.substr(sp + 1));
    } else if (!query_ok || !r.query_param(CALIPER_NET_TOKEN_PARAM, &presented)) {
        return {};
    }
    if (presented.empty()) return {};
    // Every entry is compared, in constant time each, whichever matches.
    std::string origin;
    for (const auto& e : entries)
        if (!e.first.empty() && caliper_net::wire::token_equal(presented, e.first) && origin.empty()) origin = e.second;
    return origin;
}

Verdict check(const HttpRequest& r, const std::string& token, bool command_layer, const std::string& viewer_token) {
    TokenTable t;
    t.entries.push_back({token, "api"});
    if (!viewer_token.empty()) t.entries.push_back({viewer_token, "viewer"});
    return check(r, t, command_layer);
}

Verdict check(const HttpRequest& r, const TokenTable& tokens, bool command_layer) {
    Verdict v;
    if (r.has_origin && !caliper_net::wire::origin_allowed(r.origin)) {
        v.status = 403;
        v.message = "this Origin is not allowed";
        return v;
    }
    v.routed = route(r.method, r.path);
    const Route rt = v.routed.route;
    if (!command_layer && (rt == Route::Commands || rt == Route::Command || rt == Route::Approval)) {
        v.status = 404;
        v.message = "this app has no command layer";
        return v;
    }
    if (v.routed.route == Route::None) {
        v.status = 404;
        v.message = starts_with(r.path, kPrefix) ? "no such endpoint in Caliper API v0"
                                                 : "Caliper API serves only /v0/";
        return v;
    }
    if (!v.routed.method_ok) {
        v.status = 405;
        v.message = std::string(r.path) + " takes " + v.routed.allow;
        return v;
    }
    const std::string token_origin = tokens.find(r, r.method == "GET");
    if (token_origin.empty()) {
        v.status = 401;
        v.message = r.method == "GET" ? "the session token is missing or wrong"
                                      : "the session token is missing or wrong (a POST takes it in "
                                        "the Authorization header only)";
        return v;
    }
    if (token_origin == "viewer") {
        // The viewer token opens the spec page and its read-only commands only.
        v.viewer = true;
        if (rt != Route::SpecPage && !(rt == Route::Command && viewer_command(v.routed.name))) {
            v.status = 403;
            v.message = "the viewer token opens only the spec page and the commands spec.get, applet.list and "
                        "panel.data";
            return v;
        }
    }
    // The token decides who asks, on every route: the command table an
    // agent's token is shown, and a command's origin below.
    v.origin = token_origin;
    const std::string length = trim(r.header("content-length"));
    if (rt == Route::Command) {
        for (const char* h : {"authorization", "content-length", "content-type", "x-caliper-group",
                              "x-caliper-preview"}) {
            if (header_count(r, h) > 1) {
                v.status = 400;
                v.message = std::string("the ") + h + " header is repeated";
                return v;
            }
        }
        if (header_count(r, "transfer-encoding") > 0) {
            v.status = 501;
            v.message = "Transfer-Encoding is not accepted; send the body with Content-Length";
            return v;
        }
        if (header_count(r, "content-length") == 0) {
            v.status = 411;
            v.message = "a command's arguments come with Content-Length";
            return v;
        }
        if (length.empty() || length.find_first_not_of("0123456789") != std::string::npos) {
            v.status = 400;
            v.message = "Content-Length is not a number";
            return v;
        }
        if (length.size() > 18 || std::stoull(length) > kMaxCommandBody) {
            v.status = 413;
            v.message = "a command's arguments are at most 1 MiB";
            return v;
        }
        v.body_ok = true;
        std::string type = lower(r.header("content-type"));
        type = trim(type.substr(0, type.find(';')));
        if (type != "application/json") {
            v.status = 415;
            v.message = "a command's arguments are application/json";
            return v;
        }
        if (v.viewer) {
            // The viewer token's commands come from the viewer origin; a
            // request that claims another beside it is refused, as before.
            const std::string claimed = r.header("x-caliper-origin");
            if (header_count(r, "x-caliper-origin") > 1 || (header_count(r, "x-caliper-origin") > 0 && claimed != "viewer")) {
                v.status = 400;
                v.message = "with the viewer token, X-Caliper-Origin is viewer";
                return v;
            }
        }
        // The token decides the origin; X-Caliper-Origin is ignored, so no
        // request becomes api by leaving it out or by naming another origin.
        v.origin = token_origin;
        return v;
    }
    if ((!length.empty() && length != "0") || !r.header("transfer-encoding").empty()) {
        v.status = 413;
        v.message = "Caliper API v0 takes no request body";
        return v;
    }
    if (v.routed.route == Route::Launch && !plausible_applet_id(v.routed.applet_id)) {
        v.status = 400;
        v.message = "not an applet id: an id is 1 to 128 characters of A-Z, a-z, 0-9, '.', '_' and '-'";
        return v;
    }
    if (rt == Route::Approval && !plausible_approval_id(v.routed.name)) {
        v.status = 400;
        v.message = "not an approval id: an id is 1 to 128 characters of A-Z, a-z, 0-9, '.', '_', ':' and '-'";
        return v;
    }
    return v;
}

BodyShape body_shape(const std::string& body) {
    if (!caliper_net::valid_utf8(body.data(), body.size())) return BodyShape::NotJson;
    ShapeReader rd{body.data(), body.data() + body.size()};
    rd.ws();
    const bool object = rd.p < rd.end && *rd.p == '{';
    if (!rd.value(0)) return rd.too_deep ? BodyShape::TooDeep : BodyShape::NotJson;
    rd.ws();
    if (rd.p != rd.end) return BodyShape::NotJson;
    return object ? BodyShape::Object : BodyShape::NotObject;
}

std::string command_error_json(const std::string& id, const std::string& code, const std::string& message,
                               const std::string& where) {
    caliper_net::Json e = caliper_net::Json::object();
    e.set("code", code);
    e.set("message", message);
    e.set("where", where);
    caliper_net::Json j = caliper_net::Json::object();
    j.set("id", id);
    j.set("ok", false);
    j.set("error", e);
    return j.dump();
}

std::string error_json(const std::string& message) {
    caliper_net::Json j = caliper_net::Json::object();
    j.set("error", message);
    return j.dump();
}

// ---- the server ---------------------------------------------------------------------------

struct Server::Impl {
    struct Pending {
        ConnId conn = 0;
        Call::Kind kind = Call::Kind::Applets;
        uint64_t deadline_ms = 0;
        bool layer = false;              // a command request or an approval lookup
        bool held = false;               // a command the frame thread handed to the command layer
        std::unique_ptr<PendingCommand> result;   // its answer, once submit() returned
    };
    // A request for the command layer that waits for the frame thread.
    struct LayerCall {
        uint64_t ticket = 0;
        bool lookup = false;             // GET /v0/approvals/{id}
        CommandRequest request;          // POST /v0/commands/{name}
        std::string approval;            // the lookup's approval id
    };
    struct Encode {
        uint64_t ticket = 0;
        uint32_t w = 0, h = 0;
        std::vector<uint8_t> rgba;
    };

    const Options& opt;
    std::unique_ptr<caliper_net::Hub> hub;
    std::thread hub_thread;

    mutable std::mutex mu;               // everything below
    std::map<uint64_t, Pending> pending;
    std::vector<Call> queue;             // not yet taken by the frame thread
    std::vector<LayerCall> layer_queue;  // not yet taken by the frame thread
    std::shared_ptr<CommandGateway> gateway;
    size_t held = 0;                     // entries of `pending` with held set
    uint64_t next_ticket = 1;
    bool stopping = false;
    Stats st;
    // Read without the lock: queue.size(), layer_queue.size() and held.
    std::atomic<size_t> queued{0}, layer_waiting{0}, held_count{0};
    std::atomic<uint64_t> next_command{1};

    std::mutex enc_mu;
    std::condition_variable enc_cv;
    std::deque<Encode> enc_jobs;
    bool enc_stop = false;
    std::thread enc_thread;

    explicit Impl(const Options& o, const Server* s) : opt(o), server(s) {}
    const Server* server;   // its token table

    std::shared_ptr<CommandGateway> current_gateway() const {
        std::lock_guard<std::mutex> g(mu);
        return gateway;
    }

    // Under mu: the counters read without the lock.
    void publish_counts() {
        queued.store(queue.size());
        layer_waiting.store(layer_queue.size());
        held_count.store(held);
    }

    // Every answer of a plain request goes through here. The WebSocket layer
    // adds Cache-Control: no-store and Connection: close to every response.
    void answer(ConnId c, int status, const std::string& type, std::string body,
                std::vector<std::pair<std::string, std::string>> extra = {}) {
        extra.emplace_back("X-Content-Type-Options", "nosniff");
        {
            std::lock_guard<std::mutex> g(mu);
            if (status >= 200 && status < 300) ++st.answered;
            else if (status >= 400 && status < 500) ++st.refused;
            else if (status == 503) ++st.busy;
            else if (status == 504) ++st.timeouts;
        }
        hub->respond(c, status, type, std::move(body), std::move(extra));
    }

    void answer_error(ConnId c, int status, const std::string& message,
                      std::vector<std::pair<std::string, std::string>> extra = {}) {
        answer(c, status, "application/json", error_json(message), std::move(extra));
    }

    // The hub's thread, before a request's body is read: the body of a command
    // that came with the token and an acceptable length (Verdict::body_ok), and
    // none of any other request, so that nothing is buffered for a request
    // without the token or the right Host and Origin.
    size_t body_limit(const HttpRequest& r) {
        if (!caliper_net::wire::host_allowed(r.host, hub->port())) return 0;
        const Verdict v = check(r, server->tokens(), current_gateway() != nullptr);
        return v.routed.route == Route::Command && v.body_ok ? kMaxCommandBody : 0;
    }

    // The hub's thread: a request on any path but the stream's.
    void on_request(ConnId c, const HttpRequest& r) {
        {
            std::lock_guard<std::mutex> g(mu);
            ++st.requests;
        }
        const std::shared_ptr<CommandGateway> gw = current_gateway();
        const Verdict v = check(r, server->tokens(), gw != nullptr);
        if (v.status == 401) {
            answer_error(c, 401, v.message, {{"WWW-Authenticate", "Bearer realm=\"caliper\""}});
            return;
        }
        if (v.status == 405) {
            answer_error(c, 405, v.message, {{"Allow", v.routed.allow}});
            return;
        }
        if (v.status != 0) {
            answer_error(c, v.status, v.message);
            return;
        }
        Call call;
        LayerCall layer;
        bool to_layer = false;
        switch (v.routed.route) {
            case Route::Info:
                answer(c, 200, "application/json", opt.info_json);
                return;
            case Route::SpecPage:
                if (opt.spec_page_html.empty()) {
                    answer_error(c, 404, "this build has no spec page");
                    return;
                }
                // The page reaches nothing but its own origin, and its address,
                // which holds a token, is never sent as a referrer.
                answer(c, 200, "text/html; charset=utf-8", opt.spec_page_html,
                       {{"Content-Security-Policy",
                         "default-src 'none'; script-src 'unsafe-inline'; style-src 'unsafe-inline'; "
                         "font-src data:; img-src data: blob:; connect-src 'self'; base-uri 'none'; "
                         "form-action 'none'; frame-ancestors 'none'"},
                        {"Referrer-Policy", "no-referrer"}});
                return;
            case Route::Commands:
                // An agent's token is not shown the person's own commands.
                answer(c, 200, "application/json",
                       v.origin == "mcp" || v.origin == "agent" ? gw->agent_table_json() : gw->table_json());
                return;
            case Route::Command: {
                // check() passed: the Content-Length is a number of at most 1 MiB.
                if (r.body.size() != (size_t)std::stoull(trim(r.header("content-length")))) {
                    // The command layer arrived between the head and here, so
                    // the body was not read with the request.
                    answer_error(c, 503, "the body was not read; send the request again", {{"Retry-After", "1"}});
                    return;
                }
                const std::string id = "api-" + std::to_string(next_command.fetch_add(1));
                switch (body_shape(r.body)) {
                    case BodyShape::NotJson:
                        answer_error(c, 400, "the body is not JSON");
                        return;
                    case BodyShape::TooDeep:
                        answer(c, 200, "application/json",
                               command_error_json(id, "command.invalid_args",
                                                  "The arguments nest more than 64 levels deep.", "args"));
                        return;
                    case BodyShape::NotObject:
                        answer(c, 200, "application/json",
                               command_error_json(id, "command.invalid_args", "The arguments are a JSON object.",
                                                  "args"));
                        return;
                    case BodyShape::Object:
                        break;
                }
                layer.request.id = id;
                layer.request.name = v.routed.name;
                layer.request.args = r.body;
                layer.request.origin = v.origin;
                layer.request.group = r.header("x-caliper-group");
                layer.request.preview = r.header("x-caliper-preview");
                layer.request.approval_wait_ms =
                    v.origin == "mcp" || v.origin == "agent" ? opt.command_approval_wait_ms : 0;
                to_layer = true;
                break;
            }
            case Route::Approval:
                layer.lookup = true;
                layer.approval = v.routed.name;
                to_layer = true;
                break;
            case Route::Capture:
                if (!opt.capture_unavailable.empty()) {
                    answer_error(c, 501, opt.capture_unavailable);
                    return;
                }
                call.kind = Call::Kind::Capture;
                break;
            case Route::Applets:
                call.kind = Call::Kind::Applets;
                break;
            case Route::Launch:
                call.kind = Call::Kind::Launch;
                call.applet_id = v.routed.applet_id;
                call.origin = v.origin;
                break;
            default:   // the stream path never reaches here (http_door.h)
                answer_error(c, 404, "no such endpoint in Caliper API v0");
                return;
        }
        std::string busy;
        {
            std::lock_guard<std::mutex> g(mu);
            const size_t waiting = pending.size() - held;
            size_t commands = held;
            for (const LayerCall& l : layer_queue)
                if (!l.lookup) ++commands;
            if (stopping) {
                busy = "the app is closing";
            } else if (waiting >= opt.max_pending) {
                busy = "the app is busy: " + std::to_string(waiting) + " requests are waiting for it";
            } else if (to_layer && !layer.lookup && commands >= opt.max_commands) {
                busy = "the command layer is busy: " + std::to_string(commands) +
                       " commands are running or waiting for approval";
            } else {
                const uint64_t ticket = next_ticket++;
                Pending& p = pending[ticket];
                p.conn = c;
                p.kind = call.kind;
                p.deadline_ms = caliper_net::monotonic_ms() + (uint64_t)opt.frame_timeout_ms;
                p.layer = to_layer;
                if (to_layer) {
                    layer.ticket = ticket;
                    layer_queue.push_back(std::move(layer));
                } else {
                    call.ticket = ticket;
                    queue.push_back(call);
                }
                publish_counts();
            }
        }
        if (!busy.empty()) answer_error(c, 503, busy, {{"Retry-After", "1"}});
    }

    // Answers the held commands that the command layer has answered. Called on
    // the frame thread (take()) and the hub's thread (on_tick()); it never
    // waits for the command layer.
    void poll_commands() {
        if (held_count.load() == 0) return;
        std::vector<std::pair<ConnId, std::unique_ptr<PendingCommand>>> done;
        {
            std::lock_guard<std::mutex> g(mu);
            for (auto it = pending.begin(); it != pending.end();) {
                if (it->second.result && it->second.result->ready()) {
                    done.emplace_back(it->second.conn, std::move(it->second.result));
                    --held;
                    it = pending.erase(it);
                } else {
                    ++it;
                }
            }
            publish_counts();
        }
        for (auto& d : done) answer(d.first, 200, "application/json", d.second->envelope());
    }

    // The hub's thread, every round: held commands that have been answered,
    // and calls past their deadline, which get 504.
    void on_tick() {
        poll_commands();
        const uint64_t now = caliper_net::monotonic_ms();
        std::vector<std::pair<ConnId, bool>> late;   // the connection, and whether it was held
        {
            std::lock_guard<std::mutex> g(mu);
            for (auto it = pending.begin(); it != pending.end();) {
                if (it->second.deadline_ms > now) {
                    ++it;
                    continue;
                }
                late.emplace_back(it->second.conn, it->second.held);
                const uint64_t t = it->first;
                queue.erase(std::remove_if(queue.begin(), queue.end(),
                                           [t](const Call& q) { return q.ticket == t; }),
                            queue.end());
                layer_queue.erase(std::remove_if(layer_queue.begin(), layer_queue.end(),
                                                 [t](const LayerCall& q) { return q.ticket == t; }),
                                  layer_queue.end());
                if (it->second.held) --held;
                it = pending.erase(it);
            }
            publish_counts();
        }
        const std::string why = "the app did not answer within " +
                                std::to_string(opt.frame_timeout_ms) + " ms";
        const std::string why_held = "the command did not finish within " +
                                     std::to_string(opt.command_timeout_ms / 1000) +
                                     " s; it may still complete";
        for (const auto& l : late) answer_error(l.first, 504, l.second ? why_held : why);
    }

    // Takes a call out of `pending`; false when it timed out or was answered.
    bool claim(uint64_t ticket, ConnId* conn) {
        std::lock_guard<std::mutex> g(mu);
        auto it = pending.find(ticket);
        if (it == pending.end()) return false;
        *conn = it->second.conn;
        if (it->second.held) --held;
        pending.erase(it);
        publish_counts();
        return true;
    }

    // The frame thread, from take(): hands one request to the command layer.
    void serve_layer(const std::shared_ptr<CommandGateway>& gw, LayerCall l) {
        ConnId c = 0;
        if (!gw) {
            if (claim(l.ticket, &c)) answer_error(c, 404, "this app has no command layer");
            return;
        }
        if (l.lookup) {
            std::string body = gw->approval_json(l.approval);
            if (claim(l.ticket, &c)) answer(c, 200, "application/json", std::move(body));
            return;
        }
        std::unique_ptr<PendingCommand> result = gw->submit(std::move(l.request));
        if (!result) {
            if (claim(l.ticket, &c)) answer_error(c, 500, "the command layer did not take the command");
            return;
        }
        std::lock_guard<std::mutex> g(mu);
        auto it = pending.find(l.ticket);
        if (it != pending.end()) it->second.result = std::move(result);   // else answered meanwhile
    }

    void encoder() {
        for (;;) {
            Encode job;
            {
                std::unique_lock<std::mutex> g(enc_mu);
                enc_cv.wait(g, [this] { return enc_stop || !enc_jobs.empty(); });
                if (enc_stop) return;
                job = std::move(enc_jobs.front());
                enc_jobs.pop_front();
            }
            {
                // Skip the work for a call that has timed out meanwhile.
                std::lock_guard<std::mutex> g(mu);
                if (!pending.count(job.ticket)) continue;
            }
            std::string png;
            const bool ok = opt.encode_png && opt.encode_png(job.rgba.data(), job.w, job.h, &png);
            ConnId c = 0;
            if (!claim(job.ticket, &c)) continue;
            if (ok) answer(c, 200, "image/png", std::move(png));
            else answer_error(c, 500, "the frame could not be encoded as a PNG");
        }
    }
};

Server::Server(Options o) : opt_(std::move(o)) {}

TokenTable Server::tokens() const {
    TokenTable t;
    t.entries.push_back({opt_.token, "api"});
    if (!opt_.mcp_token.empty()) t.entries.push_back({opt_.mcp_token, "mcp"});
    if (!opt_.viewer_token.empty()) t.entries.push_back({opt_.viewer_token, "viewer"});
    std::lock_guard<std::mutex> g(issued_mu_);
    for (const auto& e : issued_) t.entries.push_back(e);
    return t;
}

caliper_net::HubOptions::StreamAccess Server::stream_access(const std::string& presented) const {
    using A = caliper_net::HubOptions::StreamAccess;
    if (presented.empty()) return A::kRefused;
    // Every token is compared, in constant time each, whichever matches.
    const TokenTable t = tokens();
    std::string origin;
    for (const auto& e : t.entries)
        if (!e.first.empty() && caliper_net::wire::token_equal(presented, e.first) && origin.empty()) origin = e.second;
    if (origin == "api") return A::kFull;
    if (origin == "mcp" || origin == "agent") return A::kWatch;
    return A::kRefused;   // the viewer token, a revoked or an unknown one
}

std::string Server::issue_token(const std::string& origin) {
    if (origin != "agent" && origin != "mcp") return {};
    std::lock_guard<std::mutex> g(issued_mu_);
    for (;;) {
        std::string t = caliper_net::random_token();
        bool taken = t == opt_.token || t == opt_.mcp_token || t == opt_.viewer_token;
        for (const auto& e : issued_) taken = taken || e.first == t;
        if (taken) continue;
        issued_.push_back({t, origin});
        return t;
    }
}

void Server::revoke_token(const std::string& token) {
    std::lock_guard<std::mutex> g(issued_mu_);
    for (auto it = issued_.begin(); it != issued_.end();)
        it = it->first == token ? issued_.erase(it) : it + 1;
}

Server::~Server() { stop(); }

bool Server::start(std::string* err) {
    if (impl_) {
        *err = "Caliper API is running already";
        return false;
    }
    if (!caliper_net::valid_token(opt_.token)) {
        *err = "the session token must be 22 to 128 characters of A-Z, a-z, 0-9, '_' and '-'";
        return false;
    }
    if (!opt_.viewer_token.empty() && (!caliper_net::valid_token(opt_.viewer_token) || opt_.viewer_token == opt_.token)) {
        *err = "the viewer token must be a valid token other than the session token";
        return false;
    }
    if (!opt_.mcp_token.empty() && (!caliper_net::valid_token(opt_.mcp_token) || opt_.mcp_token == opt_.token ||
                                    opt_.mcp_token == opt_.viewer_token)) {
        *err = "the mcp token must be a valid token other than the session and viewer tokens";
        return false;
    }
    auto m = std::make_unique<Impl>(opt_, this);
    Impl* p = m.get();
    m->gateway = gateway_;
    caliper_net::HubOptions h;
    h.port = opt_.port;
    h.token = opt_.token;
    h.software_version = opt_.software_version;
    h.allow_control = opt_.allow_control;
    h.log = opt_.log;
    h.stream_path = kStreamPath;
    h.stream_alias = kStreamAlias;
    h.on_request = [p](ConnId c, const HttpRequest& r) { p->on_request(c, r); };
    h.on_tick = [p] { p->on_tick(); };
    h.tick_ms = 50;
    h.body_limit = [p](const HttpRequest& r) { return p->body_limit(r); };
    // The stream: the person's token may control a run (with allow_control);
    // the API file's token and the agents' watch it and may not. The viewer
    // token never opens it.
    h.stream_access = [this](const std::string& t) { return stream_access(t); };
    m->hub = std::make_unique<caliper_net::Hub>(std::move(h));
    if (!m->hub->listen(err)) return false;
    m->enc_thread = std::thread([p] { p->encoder(); });
    m->hub_thread = std::thread([p] { p->hub->run(); });
    impl_ = std::move(m);
    return true;
}

void Server::stop() {
    if (!impl_) return;
    Impl& m = *impl_;
    std::vector<ConnId> waiting;
    {
        std::lock_guard<std::mutex> g(m.mu);
        m.stopping = true;
        for (auto& kv : m.pending) waiting.push_back(kv.second.conn);
        m.pending.clear();
        m.queue.clear();
        m.layer_queue.clear();
        m.held = 0;
        m.publish_counts();
    }
    for (ConnId c : waiting) m.answer_error(c, 503, "the app is closing");
    {
        std::lock_guard<std::mutex> g(m.enc_mu);
        m.enc_stop = true;
    }
    m.enc_cv.notify_all();
    if (m.enc_thread.joinable()) m.enc_thread.join();
    m.hub->stop();
    if (m.hub_thread.joinable()) m.hub_thread.join();
    impl_.reset();
}

uint16_t Server::port() const { return impl_ ? impl_->hub->port() : 0; }

std::string Server::base_url() const { return "http://127.0.0.1:" + std::to_string(port()) + "/v0"; }

std::string Server::stream_url() const {
    return "ws://127.0.0.1:" + std::to_string(port()) + kStreamPath;
}

std::string Server::spec_url() const {
    if (opt_.viewer_token.empty()) return {};
    return "http://127.0.0.1:" + std::to_string(port()) + kSpecPagePath + "?" + CALIPER_NET_TOKEN_PARAM + "=" +
           opt_.viewer_token;
}

std::vector<Call> Server::take() {
    std::vector<Call> out;
    if (!impl_) return out;
    Impl& m = *impl_;
    if (m.queued.load() == 0 && m.layer_waiting.load() == 0 && m.held_count.load() == 0) return out;
    m.poll_commands();
    std::vector<Impl::LayerCall> layer;
    std::shared_ptr<CommandGateway> gw;
    {
        std::lock_guard<std::mutex> g(m.mu);
        out.swap(m.queue);
        layer.swap(m.layer_queue);
        gw = m.gateway;
        // A command the frame thread takes is held from now on, with the
        // command layer's own time limits and command_timeout_ms behind them.
        const uint64_t deadline = caliper_net::monotonic_ms() + (uint64_t)opt_.command_timeout_ms;
        for (const Impl::LayerCall& l : layer) {
            auto it = m.pending.find(l.ticket);
            if (it == m.pending.end() || l.lookup || !gw) continue;
            it->second.held = true;
            it->second.deadline_ms = deadline;
            ++m.held;
        }
        m.publish_counts();
    }
    for (Impl::LayerCall& l : layer) m.serve_layer(gw, std::move(l));
    m.poll_commands();
    return out;
}

void Server::reply(uint64_t ticket, int status, std::string json) {
    if (!impl_) return;
    ConnId c = 0;
    if (!impl_->claim(ticket, &c)) return;
    impl_->answer(c, status, "application/json", std::move(json));
}

void Server::reply_capture(uint64_t ticket, uint32_t w, uint32_t h, std::vector<uint8_t> rgba) {
    if (!impl_) return;
    {
        std::lock_guard<std::mutex> g(impl_->mu);
        if (!impl_->pending.count(ticket)) return;
    }
    {
        std::lock_guard<std::mutex> g(impl_->enc_mu);
        impl_->enc_jobs.push_back({ticket, w, h, std::move(rgba)});
    }
    impl_->enc_cv.notify_one();
}

void Server::set_command_gateway(std::shared_ptr<CommandGateway> gateway) {
    gateway_ = gateway;
    if (!impl_) return;
    std::lock_guard<std::mutex> g(impl_->mu);
    impl_->gateway = std::move(gateway);
}

Stats Server::stats() const {
    if (!impl_) return {};
    std::lock_guard<std::mutex> g(impl_->mu);
    Stats s = impl_->st;
    s.pending = impl_->pending.size();
    s.commands_held = impl_->held;
    return s;
}

caliper_net::HubStats Server::hub_stats() const { return impl_ ? impl_->hub->stats() : caliper_net::HubStats(); }

}  // namespace caliper_api
