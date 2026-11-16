// upgrade.cpp: the server's checks on a WebSocket upgrade request (wire.h,
// "the upgrade request"), split out of wire.cpp so that the frame and message
// codec needs nothing of the socket layer (ws.h): the browser's views
// (web/wasm) compile wire.cpp and json.cpp alone. Native servers link this
// file through the caliper_net library.
#include "net/wire.h"

#include "net/ws.h"

#include <string>

namespace caliper_net {
namespace wire {

// ---- the upgrade request -------------------------------------------------------------

namespace {
std::string lower_ascii(std::string s) {
    for (char& c : s)
        if (c >= 'A' && c <= 'Z') c = char(c - 'A' + 'a');
    return s;
}
bool all_digits(const std::string& s) {
    if (s.empty() || s.size() > 5) return false;
    for (char c : s)
        if (c < '0' || c > '9') return false;
    return true;
}
}  // namespace

bool host_allowed(const std::string& host, uint16_t port) {
    const std::string h = lower_ascii(host);
    const std::string p = ":" + std::to_string(port);
    if (h == "127.0.0.1" + p || h == "localhost" + p) return true;
    return port == 80 && (h == "127.0.0.1" || h == "localhost");
}

bool origin_allowed(const std::string& origin) {
    if (origin == "null") return true;
    const std::string o = lower_ascii(origin);
    const std::string vs = "vscode-webview://";
    if (o.compare(0, vs.size(), vs) == 0) {
        const std::string auth = o.substr(vs.size());
        if (auth.empty()) return false;
        for (char c : auth)
            if (c == '/' || c == '?' || c == '#' || c == '@' || (unsigned char)c <= ' ') return false;
        return true;
    }
    std::string rest;
    if (o.compare(0, 7, "http://") == 0) rest = o.substr(7);
    else if (o.compare(0, 8, "https://") == 0) rest = o.substr(8);
    else return false;
    for (const char* host : {"127.0.0.1", "localhost", "[::1]"}) {
        const std::string h = host;
        if (rest == h) return true;
        if (rest.compare(0, h.size() + 1, h + ":") == 0 && all_digits(rest.substr(h.size() + 1))) return true;
    }
    return false;
}

bool token_equal(const std::string& a, const std::string& b) {
    // Every byte of the expected token is read whatever the input, so the time
    // says nothing about how much of it matched.
    unsigned diff = a.size() != b.size() ? 1u : 0u;
    for (size_t i = 0; i < b.size(); ++i) diff |= (unsigned)(uint8_t)(i < a.size() ? a[i] : 0) ^ (uint8_t)b[i];
    return diff == 0;
}

int upgrade_status(const HttpRequest& r, const std::string& session_token, uint16_t port,
                   const std::string& path) {
    return upgrade_status(
        r, [&](const std::string& token) { return !session_token.empty() && token_equal(token, session_token); }, port,
        path);
}

int upgrade_status(const HttpRequest& r, const std::function<bool(const std::string&)>& token_ok, uint16_t port,
                   const std::string& path) {
    if (r.path != path) return 404;
    if (!host_allowed(r.host, port)) return 403;
    if (r.has_origin && !origin_allowed(r.origin)) return 403;
    std::string token;
    if (!r.query_param(CALIPER_NET_TOKEN_PARAM, &token) || !token_ok || !token_ok(token)) return 403;
    return 0;
}

}  // namespace wire
}  // namespace caliper_net
