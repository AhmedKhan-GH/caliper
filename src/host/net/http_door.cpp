// http_door.cpp: the front door of http_door.h.
#include "net/http_door.h"

#include "caliper/live_net.h"
#include "net/json.h"
#include "net/wire.h"

#include <algorithm>
#include <cctype>
#include <filesystem>
#include <fstream>
#include <system_error>

namespace caliper_net {

namespace fs = std::filesystem;

namespace {

int hex(char c) {
    if (c >= '0' && c <= '9') return c - '0';
    if (c >= 'a' && c <= 'f') return c - 'a' + 10;
    if (c >= 'A' && c <= 'F') return c - 'A' + 10;
    return -1;
}

// Percent escapes decoded once ('+' stays '+', as in a path); false on a bad
// escape or bytes that are not UTF-8.
bool decode_path(const std::string& in, std::string* out) {
    std::string s;
    for (size_t i = 0; i < in.size(); ++i) {
        if (in[i] != '%') {
            s += in[i];
            continue;
        }
        if (i + 2 >= in.size()) return false;
        const int h = hex(in[i + 1]), l = hex(in[i + 2]);
        if (h < 0 || l < 0) return false;
        s += char(h * 16 + l);
        i += 2;
    }
    if (!valid_utf8(s.data(), s.size())) return false;
    *out = std::move(s);
    return true;
}

bool name_ok(const std::string& name) {
    if (name.empty()) return false;
    for (char c : name)
        if (c == '\\' || c == '\0' || c == ':') return false;
    if (name[0] == '/') return false;
    size_t start = 0;
    while (start <= name.size()) {
        size_t end = name.find('/', start);
        if (end == std::string::npos) end = name.size();
        const std::string part = name.substr(start, end - start);
        if (part.empty() || part == "." || part == ".." || part[0] == '.') return false;
        start = end + 1;
    }
    return true;
}

bool inside(const fs::path& root, const fs::path& p) {
    auto r = root.begin(), x = p.begin();
    for (; r != root.end(); ++r, ++x)
        if (x == p.end() || *x != *r) return false;
    return x != p.end();
}

}  // namespace

bool StaticFiles::open(const std::string& dir, std::string* err) {
    std::error_code ec;
    const fs::path real = fs::canonical(fs::path(dir), ec);
    if (ec || !fs::is_directory(real, ec)) {
        *err = "not a folder: " + dir;
        return false;
    }
    root_ = real.string();
    return true;
}

bool StaticFiles::name_for_path(const std::string& url_path, std::string* name) {
    std::string path = url_path.substr(0, url_path.find_first_of("?#"));
    std::string decoded;
    if (!decode_path(path, &decoded) || decoded.empty() || decoded[0] != '/') return false;
    std::string n = decoded.substr(1);
    if (n.empty() || n.back() == '/') n += "index.html";
    if (!name_ok(n)) return false;
    *name = n;
    return true;
}

const char* StaticFiles::content_type(const std::string& name) {
    static const std::pair<const char*, const char*> kTypes[] = {
        {".html", "text/html; charset=utf-8"},
        {".js", "text/javascript; charset=utf-8"},
        {".mjs", "text/javascript; charset=utf-8"},
        {".css", "text/css; charset=utf-8"},
        {".json", "application/json"},
        {".map", "application/json"},
        {".txt", "text/plain; charset=utf-8"},
        {".svg", "image/svg+xml"},
        {".png", "image/png"},
        {".ico", "image/x-icon"},
        {".woff2", "font/woff2"},
        {".woff", "font/woff"},
        {".ttf", "font/ttf"},
        {".otf", "font/otf"},
        {".wasm", "application/wasm"},
    };
    const size_t slash = name.rfind('/');
    const size_t dot = name.rfind('.');
    if (dot != std::string::npos && (slash == std::string::npos || dot > slash)) {
        std::string ext = name.substr(dot);
        for (char& c : ext) c = (char)std::tolower((unsigned char)c);
        for (const auto& t : kTypes)
            if (ext == t.first) return t.second;
    }
    return "application/octet-stream";
}

bool StaticFiles::read(const std::string& url_path, std::string* name, std::string* body, size_t max_bytes) const {
    if (root_.empty() || !name_for_path(url_path, name)) return false;
    std::error_code ec;
    const fs::path root(root_);
    fs::path p = root;
    size_t start = 0;
    while (start <= name->size()) {
        size_t end = name->find('/', start);
        if (end == std::string::npos) end = name->size();
        p /= name->substr(start, end - start);
        start = end + 1;
    }
    const fs::path real = fs::canonical(p, ec);
    if (ec || !inside(root, real) || !fs::is_regular_file(real, ec)) return false;
    const uintmax_t size = fs::file_size(real, ec);
    if (ec || size > max_bytes) return false;
    std::ifstream f(real, std::ios::binary);
    if (!f) return false;
    body->assign((size_t)size, '\0');
    f.read(body->data(), (std::streamsize)size);
    return (size_t)f.gcount() == size;
}

bool HttpDoor::answer(Server& server, ConnId c, const HttpRequest& r) {
    const std::vector<std::pair<std::string, std::string>> nosniff = {{"X-Content-Type-Options", "nosniff"}};
    if (!wire::host_allowed(r.host, port_)) {
        ++stats_.refused_host;
        server.respond(c, 403, "text/plain; charset=utf-8", "Forbidden: Host\n", nosniff);
        return false;
    }
    if (r.method != "GET") {
        ++stats_.bad_method;
        auto h = nosniff;
        h.emplace_back("Allow", "GET");
        server.respond(c, 405, "text/plain; charset=utf-8", "Method Not Allowed\n", h);
        return false;
    }
    if (r.path == CALIPER_NET_PATH) {
        const int status = wire::upgrade_status(r, token_, port_);
        if (status != 0) {
            if (r.has_origin && !wire::origin_allowed(r.origin)) {
                ++stats_.refused_origin;
                server.respond(c, status, "text/plain; charset=utf-8", "Forbidden: Origin\n", nosniff);
            } else {
                ++stats_.refused_token;
                server.respond(c, status, "text/plain; charset=utf-8", "Forbidden: token\n", nosniff);
            }
            return false;
        }
        if (!r.websocket) {
            ++stats_.not_upgrade;
            auto h = nosniff;
            h.emplace_back("Upgrade", "websocket");
            server.respond(c, 426, "text/plain; charset=utf-8", "Upgrade Required\n", h);
            return false;
        }
        ++stats_.upgrades;
        return server.accept(c);
    }
    std::string name, body;
    const char* type = nullptr;
    if (files_ && files_->enabled()) {
        if (files_->read(r.path, &name, &body)) type = StaticFiles::content_type(name);
    } else if (r.path == "/" || r.path == "/index.html") {
        body = fallback_;
        type = "text/plain; charset=utf-8";
    }
    if (!type) {
        ++stats_.not_found;
        server.respond(c, 404, "text/plain; charset=utf-8", "Not Found\n", nosniff);
        return false;
    }
    ++stats_.files;
    auto h = nosniff;
    if (r.has_origin && wire::origin_allowed(r.origin)) {
        h.emplace_back("Access-Control-Allow-Origin", r.origin);
        h.emplace_back("Vary", "Origin");
    }
    server.respond(c, 200, type, body, h);
    return false;
}

}  // namespace caliper_net
