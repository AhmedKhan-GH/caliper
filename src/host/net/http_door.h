#pragma once
// http_door.h: the front door of a wire protocol v3 server on a ws.h Server.
// Every HTTP request on the stream port arrives here first: the WebSocket
// upgrade of CALIPER_NET_PATH after the reference page's checks (path, Host,
// Origin, constant-time token; net/wire.h), and plain GET requests for the
// files of one folder, such as the browser viewer page and its modules. The
// `caliper serve` hub uses it, and so can any later native server on the same
// layer. Host-internal C++17, no graphics dependency.
//
// THE RULES (those of caliper.serve() in python/caliper/_net.py and of the
// package's _assets.py, so a page behaves the same from either server):
//   - The Host check applies to every request: 403 unless the Host header is
//     127.0.0.1:<port> or localhost:<port> (a DNS rebinding guard).
//   - Only GET is served: 405 with "Allow: GET" otherwise.
//   - CALIPER_NET_PATH: 403 for a refused Origin or a missing or wrong token,
//     426 for a GET that is not a WebSocket upgrade, else the upgrade.
//   - Any other path names a file of the folder: the path is percent-decoded
//     once, "/" and a trailing "/" mean index.html, and a name with a part that
//     is empty, ".", "..", or starts with ".", or holds "\", ":" or NUL, is
//     refused, as is a file whose real path (symbolic links resolved) is not
//     inside the folder. Such requests and missing files get 404.
//   - A file is answered with its content type from a fixed table (.wasm is
//     application/wasm), "Cache-Control: no-store", "X-Content-Type-Options:
//     nosniff", and, when the request carries an Origin the protocol accepts,
//     "Access-Control-Allow-Origin: <that origin>" with "Vary: Origin" (never
//     a wildcard). Static files need no token: they are public code, no data.
//   - Without a folder, "/" and "/index.html" answer with a short plain page
//     given by the server (the hub says where the viewer assets are).
//   - A server that answers other requests itself (Caliper API, api/
//     api_server.h) gives the door a handler and the path it mounts the stream
//     at. Then every request whose path is not the stream's goes to the
//     handler after the Host check, whatever its method, and the door serves
//     no files and no page.
#include "caliper/live_net.h"
#include "net/ws.h"

#include <cstdint>
#include <functional>
#include <string>
#include <utility>
#include <vector>

namespace caliper_net {

// The files of one folder, read on each request.
class StaticFiles {
public:
    // Serves `dir` (it must be an existing folder). False with *err.
    bool open(const std::string& dir, std::string* err);
    bool enabled() const { return !root_.empty(); }
    const std::string& root() const { return root_; }

    // The asset name a request path asks for ("/" -> "index.html"); false for a
    // path that could leave the folder or does not decode.
    static bool name_for_path(const std::string& url_path, std::string* name);
    // The Content-Type of a file name, by its extension.
    static const char* content_type(const std::string& name);

    // The bytes of the file a request path names; false when it is refused or
    // missing. Files over max_bytes are refused.
    bool read(const std::string& url_path, std::string* name, std::string* body,
              size_t max_bytes = size_t(64) << 20) const;

private:
    std::string root_;   // the folder's real path, without a trailing separator
};

// Counters of what the door answered.
struct DoorStats {
    uint64_t upgrades = 0;
    uint64_t files = 0;
    uint64_t refused_host = 0, refused_origin = 0, refused_token = 0;
    uint64_t not_found = 0, bad_method = 0, not_upgrade = 0;
    uint64_t handed_over = 0;   // requests given to the handler
};

class HttpDoor {
public:
    // Answers a request the door does not serve itself; it must answer it
    // through the server, now or later.
    using Handler = std::function<void(ConnId, const HttpRequest&)>;

    // `token` is the session token; `port` the port the server listens on;
    // `files` may be null; `fallback_page` answers "/" without files.
    // `stream_path` is where the stream is upgraded, and `other`, when set,
    // takes every request on another path (see the rules above).
    HttpDoor(std::string token, uint16_t port, const StaticFiles* files, std::string fallback_page,
             std::string stream_path = CALIPER_NET_PATH, Handler other = nullptr)
        : token_(std::move(token)), port_(port), files_(files), fallback_(std::move(fallback_page)),
          stream_path_(std::move(stream_path)), other_(std::move(other)) {}

    // Answers one Event::Request on `server`: accepts the upgrade (true; an
    // Event::Open follows) or responds and closes (false).
    bool answer(Server& server, ConnId c, const HttpRequest& r);
    const DoorStats& stats() const { return stats_; }

private:
    std::string token_;
    uint16_t port_;
    const StaticFiles* files_;
    std::string fallback_;
    std::string stream_path_;
    Handler other_;
    DoorStats stats_;
};

}  // namespace caliper_net
