#pragma once
// ============================================================================
// api_server.h: Caliper API v0, an HTTP interface to the running app on
// 127.0.0.1 (docs/agents/http-api.md). It is off unless the app starts with
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
//   GET  /v0/commands               the command layer's descriptor table
//   POST /v0/commands/{name}        run a command; the args object is the body
//   GET  /v0/approvals/{id}         what became of a command that needed approval
//   GET  /v0/spec                   the browser viewer of spec applets, a page
//
// THE COMMAND LAYER. The last three are served only while a CommandGateway
// (api/command_gateway.h) is set with set_command_gateway(); without one they
// answer 404, as for any path the API does not serve. POST /v0/commands/{name}
// is the one request that has a body: a JSON object of at most 1 MiB
// (kMaxCommandBody) with Content-Type application/json, read by the hub's
// thread only after the request head passed every check. It answers the
// command layer's result envelope with 200, whether `ok` is true or false;
// faults of the request itself keep their status. The origin of the
// invocation is the origin of the token that authenticated the request (see
// TOKENS), never a header: X-Caliper-Origin is ignored, so no request can
// claim an origin its token does not have. X-Caliper-Group gives its group,
// and X-Caliper-Preview the id of a stacked preview that its edits apply to.
// A command that needs approval is held for the origins mcp and agent while
// the person decides, up to Options::command_approval_wait_ms, and answers
// command.needs_approval at once for api, whose caller reads the outcome from
// GET /v0/approvals/{id}.
//
// TOKENS. Each token has a fixed origin. Options::token is the person's own
// (api): unreviewed scripting, printed on the app's start line and written to
// no file. Options::mcp_token (mcp) is the one the API file holds, for an
// outside agent's caliper-mcp: its spec edits and destructive commands are
// reviewed in the app. issue_token() makes a token for an agent the app
// starts (agent: the side chat's agent, the widget agent's sidecar), and
// revoke_token() ends it when that agent's session ends; an unknown or
// revoked token is 401. Options::viewer_token opens the spec page only (THE
// SPEC PAGE). Every comparison is constant-time. The stream (/v3/stream) takes
// the person's token, which may control a run when allow_control is on, and
// the mcp and agent tokens, which watch it and may not control
// (HubOptions::stream_access); the viewer token does not open it.
//
// THE SPEC PAGE. GET /v0/spec answers Options::spec_page_html, the page of
// the web package that draws a saved spec applet in a browser
// (web/src/spec/), embedded in the executable; no file is read. The page
// takes the token and the applet's id from its own address and calls
// spec.get, applet.list and panel.data. Options::viewer_token is a second
// token for it: a request that carries the viewer token instead of the
// session token may only GET the page and POST those three commands
// (kViewerCommands), always from the origin viewer, so the link printed for
// the page never opens the rest of the API or the command layer's edits;
// anything else is refused with 403.
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
// protocol says). Nothing else is served: no files and no way to run code but
// the commands of the command layer, which build data. A request with a body
// is refused (413), except a command's arguments, and launch takes only an
// applet id in its path that the app's loader lists.
//
// THREADS. The hub's thread answers every request and never waits for the
// app. A request that reads or changes the app's state is a Call that the
// frame thread takes at the start of a frame (take()) and answers with
// reply(), or with reply_capture() at the end of that frame for a capture. A
// call not answered within frame_timeout_ms gets 504, and at most max_pending
// calls wait at once (503 beyond). A capture's PNG is encoded on a worker
// thread, so neither the frame loop nor the stream waits for the encoder.
// A command request waits for the frame thread the same way: take() hands it
// to the gateway on the frame thread, and from then on it is held, without a
// thread of its own, until the command layer answers it; take() and the hub's
// thread look for its answer without waiting. At most max_commands command
// requests are held or waiting at once (503 beyond), and one the frame thread
// took that has no answer within command_timeout_ms gets 504.
// Host-internal C++17 on net/; no graphics dependency.
// ============================================================================
#include "api/command_gateway.h"
#include "net/hub.h"
#include "net/ws.h"

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

namespace caliper_api {

// The port the API listens on without --api-port.
constexpr uint16_t kDefaultPort = 7341;
// Where the stream is mounted: the protocol's path, and an alias under /v0/.
inline constexpr char kStreamPath[] = CALIPER_NET_PATH;
inline constexpr char kStreamAlias[] = "/v0/stream";
// The largest body POST /v0/commands/{name} takes.
constexpr size_t kMaxCommandBody = size_t(1) << 20;

// ---- routing --------------------------------------------------------------------------

enum class Route { None, Info, Applets, Launch, Capture, Stream, Commands, Command, Approval, SpecPage, SpecAsset };

// The spec page's path, and the commands a request with the viewer token may run.
inline constexpr char kSpecPagePath[] = "/v0/spec";
// The files of the WebAssembly views that the spec page loads to draw the
// programs of its scene panels, GET /v0/spec/<name> (Options::spec_assets_dir):
// these three names only, never a path that the request names.
inline constexpr char kSpecAssetHead[] = "/v0/spec/";
inline constexpr const char* kSpecAssets[] = {"caliper-views-loader.js", "caliper-views.js", "caliper-views.wasm"};
bool spec_asset(const std::string& name);
inline constexpr const char* kViewerCommands[] = {"spec.get", "applet.list", "panel.data"};
bool viewer_command(const std::string& name);

struct Routed {
    Route route = Route::None;   // None: 404
    const char* allow = "";      // the method the path takes ("GET" or "POST")
    bool method_ok = false;      // false: 405 with Allow
    std::string applet_id;       // Launch: the id in the path
    std::string name;            // Command: the command's name; Approval: the approval id
};
// The endpoint a method and path name. The path is matched exactly, without
// percent-decoding: an applet id, a command name and an approval id hold no
// character that needs escaping. A command name or an approval id in the path
// is 1 to 128 characters without a '/'.
Routed route(const std::string& method, const std::string& path);

// The applet ids launch accepts in its path: 1 to 128 characters of
// [A-Za-z0-9._-] (every valid manifest id is one).
bool plausible_applet_id(const std::string& id);
// The approval ids GET /v0/approvals/{id} accepts: 1 to 128 characters of
// [A-Za-z0-9._:-] (the command layer's are "ap-<n>").
bool plausible_approval_id(const std::string& id);

// True when the request carries the session token: an "Authorization: Bearer
// <token>" header or, when query_ok, the query parameter `token`. Both are
// compared with wire::token_equal, in constant time.
bool token_ok(const caliper_net::HttpRequest& r, const std::string& token, bool query_ok);

// The tokens the server accepts, each with its origin: "api", "mcp", "agent"
// or "viewer". find() compares the request's token with every entry in
// constant time and answers its origin, or "" for a missing or unknown token,
// and gives the token's client in *client: which agent the app issued it for
// ("chat", "widget"), from `clients` at the entry's index, "" when none.
struct TokenTable {
    std::vector<std::pair<std::string, std::string>> entries;   // token, origin
    std::vector<std::string> clients;                           // by entry; shorter means ""
    std::string find(const caliper_net::HttpRequest& r, bool query_ok, std::string* client = nullptr) const;
};

// The checks of a plain request after the door's Host check, in this order:
// Origin (403), route (404, then 405), token (401), then for a command: a
// repeated header that the checks read or the command takes (400), Transfer-Encoding (501), no
// Content-Length (411), one that is not a number (400) or over
// kMaxCommandBody (413), a Content-Type other than application/json (415),
// and with the viewer token an X-Caliper-Origin other than viewer (400); for
// every other endpoint: a body (413), the applet id of a launch and the
// approval id of an approval (400). The command layer's three endpoints are
// 404 unless command_layer is true. The command's origin is the token's
// (TOKENS); X-Caliper-Origin is ignored for every other token. A request with
// the viewer token may GET the spec page and POST the kViewerCommands from the
// origin viewer only, and anything else is 403. status 0 means it may be
// served; the body itself is checked once it has arrived.
struct Verdict {
    int status = 0;
    std::string message;   // for the JSON error body
    Routed routed;
    std::string origin;    // Command: "api", "mcp", "agent" or "viewer"
    std::string client;    // the token's client ("chat", "widget"), "" for none
    bool viewer = false;   // the request carries the viewer token
    // Command: the request came with the token and a Content-Length within
    // kMaxCommandBody, so its body is read even when a later check (415, the
    // origin's 400) refuses it, and the refusal reaches the client before the
    // connection closes.
    bool body_ok = false;
};
Verdict check(const caliper_net::HttpRequest& r, const TokenTable& tokens, bool command_layer = false);
// The same with the person's token (api) and a viewer token alone.
Verdict check(const caliper_net::HttpRequest& r, const std::string& token, bool command_layer = false,
              const std::string& viewer_token = {});

// What a command's body holds, checked without building it: a JSON object,
// another JSON value, JSON nested deeper than kMaxJsonDepth, or not JSON (RFC
// 8259 in UTF-8, with paired surrogate escapes).
enum class BodyShape { Object, NotObject, TooDeep, NotJson };
constexpr int kMaxJsonDepth = 64;
BodyShape body_shape(const std::string& body);

// {"error": message}
std::string error_json(const std::string& message);

// ---- the server -----------------------------------------------------------------------

struct Options {
    uint16_t port = kDefaultPort;      // 0: any free port (see Server::port())
    std::string token;                 // the person's session token, origin api (caliper_net::valid_token)
    // The token of the API file, origin mcp (caliper_net::valid_token, not the
    // session token), or "" for none.
    std::string mcp_token;
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
    // The command layer (set_command_gateway()).
    int command_approval_wait_ms = 120000;   // mcp, agent: a command held for approval waits this long
    int command_timeout_ms = 600000;         // a command the frame thread took answers within this (504)
    size_t max_commands = 32;                // command requests held or waiting at once (503)
    // The viewer token (caliper_net::valid_token, not the session token), or
    // "" for none; and the spec page that GET /v0/spec answers, "" for a 404.
    std::string viewer_token;
    std::string spec_page_html;
    // The folder that holds the files of kSpecAssets, which GET /v0/spec/<name>
    // answers with either token, each at most kSpecAssetMaxBytes; "" or a
    // missing file is a 404, and the page then shows its programs' panels
    // without them.
    std::string spec_assets_dir;
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
    std::string origin;      // the origin of the request's token: "api", "mcp" or "agent"
};

struct Stats {
    uint64_t requests = 0;             // plain requests handed to the API
    uint64_t answered = 0;             // of which answered 2xx
    uint64_t refused = 0;              // answered 4xx
    uint64_t timeouts = 0;             // 504
    uint64_t busy = 0;                 // 503
    size_t pending = 0;                // calls waiting now, held commands included
    size_t commands_held = 0;          // of which commands the command layer holds
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
    const std::string& mcp_token() const { return opt_.mcp_token; }

    // A new token with the origin "agent" or "mcp" ("" for any other origin),
    // accepted until revoke_token(); any thread, while the server runs or not.
    // `client` names the agent the app issued it for ("chat" for the side
    // chat's agent, "widget" for the widget agent's sidecar): its commands
    // carry it (CommandRequest::client), so the command layer applies that
    // agent's mode; no request can name another (0 to 32 of a-z).
    std::string issue_token(const std::string& origin, const std::string& client = {});
    // Ends an issued token: a request that carries it is then 401.
    void revoke_token(const std::string& token);
    std::string base_url() const;      // http://127.0.0.1:<port>/v0
    std::string stream_url() const;    // ws://127.0.0.1:<port>/v3/stream (no token)
    // http://127.0.0.1:<port>/v0/spec?token=<viewer token>, "" without a
    // viewer token; a saved spec applet opens with &applet=<id> added.
    std::string spec_url() const;

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
    // take() also hands the command requests and approval lookups that wait
    // to the command gateway, and answers the commands it has answered; it
    // returns only the calls above.

    // The command layer the API serves (command_gateway.h), or none (the
    // command endpoints answer 404). Before start(), or while the server runs
    // on the thread that calls take(); clear it before the command layer
    // closes. Requests that wait for the frame thread then get 404, and
    // commands already held are answered when their PendingCommand is.
    void set_command_gateway(std::shared_ptr<CommandGateway> gateway);

    Stats stats() const;               // any thread
    caliper_net::HubStats hub_stats() const;

private:
    struct Impl;
    TokenTable tokens() const;
    // The stream's access for a token: full for the person's (api), watch
    // for mcp and agent tokens, refused for the viewer's and any other.
    caliper_net::HubOptions::StreamAccess stream_access(const std::string& presented) const;
    Options opt_;
    mutable std::mutex issued_mu_;
    struct Issued {
        std::string token, origin, client;
    };
    std::vector<Issued> issued_;   // issue_token()'s
    std::shared_ptr<CommandGateway> gateway_;
    std::unique_ptr<Impl> impl_;
};

}  // namespace caliper_api
