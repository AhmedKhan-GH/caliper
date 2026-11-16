#pragma once
// ============================================================================
// api_server.h: Caliper API v0, an HTTP interface to the running app on
// 127.0.0.1 (docs/references/api.md). It is off unless the app starts with
// --api or CALIPER_API=1; `caliper`, `caliper dev` and `caliper shot` take it
// (api/app_api.h starts it for them).
//
//   GET  /v0/info                   the host's version, epoch, services and SDK
//   GET  /v0/applets                the applets the app's loader lists, as
//                                   `caliper list` shows them
//   POST /v0/applets/{id}/launch    open that applet's page
//   GET  /v0/capture.png            the app's current frame as a PNG
//   GET  /v3/stream                 the WebSocket of wire protocol v3
//   GET  /v0/stream                 the same WebSocket, under the API's prefix
//
// THE STREAM. /v3/stream is wire protocol v3 with the app in the hub role: the
// hub of `caliper serve` (net/hub.h) runs on the API's port, so producers
// connect to the app and become sessions, and viewers list them and attach to
// one, as with `caliper serve`. It is the protocol's own path, so every client
// of the protocol (the app's NetSource, caliper-viewer --url, the browser
// element) attaches to it unchanged; /v0/stream is an alias of it. Every check
// of the upgrade is the protocol's, on either path.
//
// SECURITY. Every request gets the stream's checks (net/wire.h,
// net/http_door.h): the server listens on 127.0.0.1 only; the Host header must
// name 127.0.0.1 or localhost with the API's port (403); an Origin header, when
// present, must be one the protocol accepts (403); and the session token must
// come with it, compared in constant time. A plain request carries the token
// in an "Authorization: Bearer <token>" header, and a GET may carry it as the
// query parameter `token` instead, as the WebSocket does; a POST takes the
// header only. A missing or wrong token answers 401 (403 on the stream, as the
// protocol says). Nothing else is served: no files, no command layer, no way
// to run code. A request with a body is refused (413), and launch takes only
// an applet id in its path that the app's loader lists.
//
// THREADS. The hub's thread answers every request and never waits for the
// app. A request that reads or changes the app's state is a Call that the
// frame thread takes at the start of a frame (take()) and answers with
// reply(), or with reply_capture() at the end of that frame for a capture. A
// call not answered within frame_timeout_ms gets 504, and at most max_pending
// calls wait at once (503 beyond). A capture's PNG is encoded on a worker
// thread, so neither the frame loop nor the stream waits for the encoder.
// Host-internal C++17 on net/; no graphics dependency.
// ============================================================================
#include "net/hub.h"
#include "net/ws.h"

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace caliper_api {

// The port the API listens on without --api-port.
constexpr uint16_t kDefaultPort = 7341;
// Where the stream is mounted: the protocol's path, and an alias under /v0/.
inline constexpr char kStreamPath[] = CALIPER_NET_PATH;
inline constexpr char kStreamAlias[] = "/v0/stream";

// ---- routing --------------------------------------------------------------------------

enum class Route { None, Info, Applets, Launch, Capture, Stream };

struct Routed {
    Route route = Route::None;   // None: 404
    const char* allow = "";      // the method the path takes ("GET" or "POST")
    bool method_ok = false;      // false: 405 with Allow
    std::string applet_id;       // Launch: the id in the path
};
// The endpoint a method and path name. The path is matched exactly, without
// percent-decoding: an applet id holds no character that needs escaping.
Routed route(const std::string& method, const std::string& path);

// The applet ids launch accepts in its path: 1 to 128 characters of
// [A-Za-z0-9._-] (every valid manifest id is one).
bool plausible_applet_id(const std::string& id);

// True when the request carries the session token: an "Authorization: Bearer
// <token>" header or, when query_ok, the query parameter `token`. Both are
// compared with wire::token_equal, in constant time.
bool token_ok(const caliper_net::HttpRequest& r, const std::string& token, bool query_ok);

// The checks of a plain request after the door's Host check, in this order:
// Origin (403), route (404, then 405), token (401), body (413), applet id
// (400). status 0 means it may be served.
struct Verdict {
    int status = 0;
    std::string message;   // for the JSON error body
    Routed routed;
};
Verdict check(const caliper_net::HttpRequest& r, const std::string& token);

// {"error": message}
std::string error_json(const std::string& message);

// ---- the server -----------------------------------------------------------------------

struct Options {
    uint16_t port = kDefaultPort;      // 0: any free port (see Server::port())
    std::string token;                 // the session token (caliper_net::valid_token)
    std::string info_json;             // the body of GET /v0/info
    std::string software_version;      // the hub's WELCOME software.version
    // The stream's hub forwards viewers' controls (wire protocol 3.1) only with
    // this (--allow-control with --api); off by default, as in caliper serve.
    bool allow_control = false;
    // Not empty: the app cannot read its frame back, and GET /v0/capture.png
    // answers 501 with this reason without asking the frame thread.
    std::string capture_unavailable;
    int frame_timeout_ms = 5000;       // a call waits this long for the frame thread (504)
    size_t max_pending = 8;            // calls waiting for the frame thread at once (503)
    // Encodes tightly packed top-down RGBA8 as a PNG (export_encode_png).
    std::function<bool(const uint8_t*, uint32_t, uint32_t, std::string*)> encode_png;
    // One line per session that starts or ends on the stream; never the token.
    std::function<void(const std::string&)> log;
};

// What the frame thread is asked to do.
struct Call {
    enum class Kind { Applets, Launch, Capture };
    uint64_t ticket = 0;
    Kind kind = Kind::Applets;
    std::string applet_id;   // Launch
};

struct Stats {
    uint64_t requests = 0;             // plain requests handed to the API
    uint64_t answered = 0;             // of which answered 2xx
    uint64_t refused = 0;              // answered 4xx
    uint64_t timeouts = 0;             // 504
    uint64_t busy = 0;                 // 503
    size_t pending = 0;                // calls waiting now
};

class Server {
public:
    explicit Server(Options o);
    ~Server();                         // stop()
    Server(const Server&) = delete;
    Server& operator=(const Server&) = delete;

    // Listens on 127.0.0.1 and starts the hub's thread and the encoder's.
    // False with *err (an invalid token, the port is taken).
    bool start(std::string* err);
    // Answers every waiting call with 503, stops the hub and joins the threads.
    void stop();

    uint16_t port() const;
    const std::string& token() const { return opt_.token; }
    std::string base_url() const;      // http://127.0.0.1:<port>/v0
    std::string stream_url() const;    // ws://127.0.0.1:<port>/v3/stream (no token)

    // ---- the frame thread ----
    // The calls waiting for the frame thread, oldest first. Cheap when none
    // wait: call it once per frame.
    std::vector<Call> take();
    // Answers an Applets or Launch call with a JSON body. A call that has
    // timed out meanwhile is dropped.
    void reply(uint64_t ticket, int status, std::string json);
    // Answers a Capture call with the frame: w x h tightly packed RGBA8, which
    // the encoder thread turns into the PNG.
    void reply_capture(uint64_t ticket, uint32_t w, uint32_t h, std::vector<uint8_t> rgba);

    Stats stats() const;               // any thread
    caliper_net::HubStats hub_stats() const;

private:
    struct Impl;
    Options opt_;
    std::unique_ptr<Impl> impl_;
};

}  // namespace caliper_api
