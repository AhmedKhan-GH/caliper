#include <caliper_builder/session/command_server.h>

#include <caliper_builder/commands/command.h>
#include <caliper_builder/commands/command_table.h>

#include <nlohmann/json.hpp>

#include <algorithm>
#include <atomic>
#include <cctype>
#include <exception>
#include <fstream>
#include <map>
#include <mutex>
#include <random>
#include <thread>
#include <vector>

#ifndef _WIN32
#include <arpa/inet.h>
#include <cerrno>
#include <fcntl.h>
#include <netinet/in.h>
#include <poll.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

namespace caliper_builder {

using nlohmann::json;

bool token_equal(std::string_view a, std::string_view b) {
    unsigned char diff = a.size() == b.size() ? 0 : 1;
    const std::size_t n = std::max(a.size(), b.size());
    for (std::size_t i = 0; i < n; ++i) {
        const unsigned char x = i < a.size() ? static_cast<unsigned char>(a[i]) : 0;
        const unsigned char y = i < b.size() ? static_cast<unsigned char>(b[i]) : 0;
        diff = static_cast<unsigned char>(diff | (x ^ y));
    }
    return diff == 0;
}

std::string generate_session_token() {
    unsigned char bytes[24] = {};
    bool filled = false;
#ifndef _WIN32
    const int fd = ::open("/dev/urandom", O_RDONLY | O_CLOEXEC);
    if (fd >= 0) {
        std::size_t got = 0;
        while (got < sizeof bytes) {
            const ssize_t n = ::read(fd, bytes + got, sizeof bytes - got);
            if (n <= 0) break;
            got += static_cast<std::size_t>(n);
        }
        filled = got == sizeof bytes;
        ::close(fd);
    }
#endif
    if (!filled) {
        std::random_device rd;
        for (auto& b : bytes) b = static_cast<unsigned char>(rd() & 0xFF);
    }
    static const char* kAlphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_";
    std::string out;
    for (std::size_t i = 0; i < sizeof bytes; i += 3) {
        const unsigned v = (unsigned(bytes[i]) << 16) | (unsigned(bytes[i + 1]) << 8) | unsigned(bytes[i + 2]);
        out += kAlphabet[(v >> 18) & 63];
        out += kAlphabet[(v >> 12) & 63];
        out += kAlphabet[(v >> 6) & 63];
        out += kAlphabet[v & 63];
    }
    return out;
}

#ifdef _WIN32

struct CommandServer::Impl {
    CommandServerOptions options;
};

CommandServer::CommandServer(CommandQueue&, CommandServerOptions options) : impl_(std::make_unique<Impl>()) {
    impl_->options = std::move(options);
}
CommandServer::~CommandServer() = default;
Error CommandServer::start() {
    return Error("server.unsupported", "The stand-in command server runs on macOS and Linux only for now.");
}
std::uint16_t CommandServer::port() const { return 0; }
void CommandServer::stop() {}
std::uint64_t CommandServer::served() const { return 0; }
Error write_api_file(const std::string&, std::uint16_t, const std::string&, std::string*) {
    return Error("server.unsupported", "The stand-in command server runs on macOS and Linux only for now.");
}
void remove_api_file(const std::string&) {}

#else

namespace {

constexpr std::size_t kMaxHeaders = 100;
constexpr int kMaxJsonDepth = 64;
constexpr std::size_t kMaxDrainBytes = std::size_t(8) << 20;
constexpr int kDrainMs = 2000;

std::int64_t now_ms() {
    return std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now().time_since_epoch())
        .count();
}

std::string lower(std::string s) {
    for (char& c : s) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    return s;
}

std::string trim(const std::string& s) {
    std::size_t a = 0;
    std::size_t b = s.size();
    while (a < b && (s[a] == ' ' || s[a] == '\t')) ++a;
    while (b > a && (s[b - 1] == ' ' || s[b - 1] == '\t')) --b;
    return s.substr(a, b - a);
}

const char* reason(int status) {
    switch (status) {
        case 200: return "OK";
        case 400: return "Bad Request";
        case 401: return "Unauthorized";
        case 403: return "Forbidden";
        case 404: return "Not Found";
        case 405: return "Method Not Allowed";
        case 408: return "Request Timeout";
        case 411: return "Length Required";
        case 413: return "Payload Too Large";
        case 415: return "Unsupported Media Type";
        case 431: return "Request Header Fields Too Large";
        case 501: return "Not Implemented";
        case 503: return "Service Unavailable";
        case 504: return "Gateway Timeout";
    }
    return "Error";
}

struct Response {
    int status = 200;
    std::string body;
    std::vector<std::pair<std::string, std::string>> headers;
};

Response error_response(int status, const std::string& message) {
    Response r;
    r.status = status;
    r.body = json{{"error", message}}.dump();
    return r;
}

Response json_response(const json& value) {
    Response r;
    r.body = value.dump(-1, ' ', false, json::error_handler_t::replace);
    return r;
}

// Writes all of `data` within the time limit.
bool write_all(int fd, const std::string& data, int timeout_ms) {
    const std::int64_t deadline = now_ms() + timeout_ms;
    std::size_t sent = 0;
    while (sent < data.size()) {
        const std::int64_t left = deadline - now_ms();
        if (left <= 0) return false;
        pollfd p{fd, POLLOUT, 0};
        const int r = ::poll(&p, 1, static_cast<int>(left));
        if (r < 0 && errno == EINTR) continue;
        if (r <= 0) return false;
        int flags = 0;
#ifdef MSG_NOSIGNAL
        flags |= MSG_NOSIGNAL;
#endif
        const ssize_t n = ::send(fd, data.data() + sent, data.size() - sent, flags);
        if (n < 0) {
            if (errno == EINTR || errno == EAGAIN) continue;
            return false;
        }
        sent += static_cast<std::size_t>(n);
    }
    return true;
}

// The nesting depth of a JSON text, counted without parsing it; strings are
// skipped. Stops counting once it passes `limit`.
int json_depth(const std::string& text, int limit) {
    int depth = 0;
    int max_depth = 0;
    bool in_string = false;
    bool escape = false;
    for (char c : text) {
        if (in_string) {
            if (escape) escape = false;
            else if (c == '\\') escape = true;
            else if (c == '"') in_string = false;
            continue;
        }
        if (c == '"') in_string = true;
        else if (c == '[' || c == '{') {
            max_depth = std::max(max_depth, ++depth);
            if (max_depth > limit) return max_depth;
        } else if (c == ']' || c == '}') {
            --depth;
        }
    }
    return max_depth;
}

bool loopback_origin(const std::string& origin) {
    if (origin == "null") return true;
    if (origin.rfind("vscode-webview://", 0) == 0) return true;
    std::string rest;
    if (origin.rfind("http://", 0) == 0) rest = origin.substr(7);
    else if (origin.rfind("https://", 0) == 0) rest = origin.substr(8);
    else return false;
    std::string host = rest;
    std::string port;
    if (!rest.empty() && rest[0] == '[') {
        const auto close = rest.find(']');
        if (close == std::string::npos) return false;
        host = rest.substr(0, close + 1);
        port = rest.substr(close + 1);
    } else {
        const auto colon = rest.find(':');
        if (colon != std::string::npos) {
            host = rest.substr(0, colon);
            port = rest.substr(colon);
        }
    }
    if (!port.empty()) {
        if (port[0] != ':' || port.size() < 2 || port.size() > 6) return false;
        for (std::size_t i = 1; i < port.size(); ++i)
            if (!std::isdigit(static_cast<unsigned char>(port[i]))) return false;
    }
    host = lower(host);
    return host == "127.0.0.1" || host == "localhost" || host == "[::1]";
}

struct Request {
    std::string method;
    std::string target;
    std::string path;
    std::string query;
    std::map<std::string, std::string> headers;  // lower-case names
    std::string body;                             // what followed the head in the first reads
};

}  // namespace

struct CommandServer::Impl {
    CommandQueue& queue;
    CommandServerOptions options;
    int listener = -1;
    int wake[2] = {-1, -1};
    std::uint16_t port = 0;
    std::thread acceptor;
    std::atomic<bool> stopping{false};
    std::atomic<std::uint64_t> served{0};
    std::atomic<std::uint64_t> next_id{1};

    struct Worker {
        std::thread thread;
        std::shared_ptr<std::atomic<bool>> done;
    };
    std::mutex workers_mutex;
    std::vector<Worker> workers;

    Impl(CommandQueue& q, CommandServerOptions o) : queue(q), options(std::move(o)) {}

    void reap(bool all) {
        std::vector<Worker> finished;
        {
            std::lock_guard<std::mutex> lock(workers_mutex);
            for (auto it = workers.begin(); it != workers.end();) {
                if (all || it->done->load()) {
                    finished.push_back(std::move(*it));
                    it = workers.erase(it);
                } else {
                    ++it;
                }
            }
        }
        for (auto& w : finished)
            if (w.thread.joinable()) w.thread.join();
    }

    std::size_t active() {
        std::lock_guard<std::mutex> lock(workers_mutex);
        std::size_t n = 0;
        for (const auto& w : workers) n += w.done->load() ? 0 : 1;
        return n;
    }

    void send_response(int fd, const Response& r) {
        std::string head = "HTTP/1.1 " + std::to_string(r.status) + " " + reason(r.status) + "\r\n";
        head += "Content-Type: application/json\r\n";
        head += "Content-Length: " + std::to_string(r.body.size()) + "\r\n";
        head += "Cache-Control: no-store\r\nX-Content-Type-Options: nosniff\r\nConnection: close\r\n";
        for (const auto& [k, v] : r.headers) head += k + ": " + v + "\r\n";
        head += "\r\n";
        write_all(fd, head + r.body, options.write_timeout_ms);
        served.fetch_add(1);
    }

    // Ends a connection: no more writes, then what the client still sends is
    // read and dropped for a moment, so that closing does not reset the
    // connection before the client has read the response.
    void finish(int fd) {
        ::shutdown(fd, SHUT_WR);
        const std::int64_t deadline = now_ms() + kDrainMs;
        std::size_t drained = 0;
        char buf[16384];
        while (drained < kMaxDrainBytes) {
            const std::int64_t left = deadline - now_ms();
            if (left <= 0) break;
            pollfd p{fd, POLLIN, 0};
            const int r = ::poll(&p, 1, static_cast<int>(left));
            if (r < 0 && errno == EINTR) continue;
            if (r <= 0) break;
            const ssize_t n = ::recv(fd, buf, sizeof buf, 0);
            if (n <= 0) break;
            drained += static_cast<std::size_t>(n);
        }
        ::close(fd);
    }

    // Reads the head; on a fault returns its status (0 when the client went away).
    int read_head(int fd, Request& req) {
        std::string buf;
        const std::int64_t deadline = now_ms() + options.head_timeout_ms;
        std::size_t end = std::string::npos;
        char chunk[4096];
        while (end == std::string::npos) {
            if (buf.size() > options.max_head_bytes) return 431;
            const std::int64_t left = deadline - now_ms();
            if (left <= 0) return 408;
            pollfd p{fd, POLLIN, 0};
            const int r = ::poll(&p, 1, static_cast<int>(left));
            if (r < 0 && errno == EINTR) continue;
            if (r == 0) return 408;
            if (r < 0) return 0;
            const ssize_t n = ::recv(fd, chunk, sizeof chunk, 0);
            if (n < 0 && (errno == EINTR || errno == EAGAIN)) continue;
            if (n <= 0) return 0;
            buf.append(chunk, static_cast<std::size_t>(n));
            end = buf.find("\r\n\r\n");
        }
        if (end + 4 > options.max_head_bytes) return 431;
        req.body = buf.substr(end + 4);
        const std::string head = buf.substr(0, end + 2);
        std::size_t pos = 0;
        bool first = true;
        while (pos < head.size()) {
            const std::size_t eol = head.find("\r\n", pos);
            if (eol == std::string::npos) return 400;
            const std::string line = head.substr(pos, eol - pos);
            pos = eol + 2;
            if (line.find('\n') != std::string::npos || line.find('\r') != std::string::npos) return 400;
            for (unsigned char c : line)
                if (c == 0 || (c < 0x20 && c != '\t') || c == 0x7F) return 400;
            if (first) {
                first = false;
                const auto s1 = line.find(' ');
                const auto s2 = s1 == std::string::npos ? s1 : line.find(' ', s1 + 1);
                if (s1 == std::string::npos || s2 == std::string::npos || line.find(' ', s2 + 1) != std::string::npos)
                    return 400;
                req.method = line.substr(0, s1);
                req.target = line.substr(s1 + 1, s2 - s1 - 1);
                const std::string version = line.substr(s2 + 1);
                if (version != "HTTP/1.1" && version != "HTTP/1.0") return 400;
                if (req.method.empty() || req.target.empty() || req.target[0] != '/') return 400;
                for (char c : req.method)
                    if (c < 'A' || c > 'Z') return 400;
                continue;
            }
            if (line.empty() || line[0] == ' ' || line[0] == '\t') return 400;  // no folded headers
            const auto colon = line.find(':');
            if (colon == std::string::npos || colon == 0) return 400;
            const std::string name = lower(line.substr(0, colon));
            for (char c : name)
                if (c == ' ' || c == '\t') return 400;
            if (req.headers.size() >= kMaxHeaders) return 431;
            if (!req.headers.emplace(name, trim(line.substr(colon + 1))).second) {
                // Repeating a header that the checks read could smuggle a value past them.
                static const char* kSingle[] = {"host", "origin", "authorization", "content-length", "content-type",
                                                "transfer-encoding", "x-caliper-origin", "x-caliper-group",
                                                "x-caliper-preview"};
                for (const char* s : kSingle)
                    if (name == s) return 400;
            }
        }
        const auto q = req.target.find('?');
        req.path = req.target.substr(0, q);
        req.query = q == std::string::npos ? std::string() : req.target.substr(q + 1);
        return 200;
    }

    std::string header(const Request& req, const char* name) const {
        const auto it = req.headers.find(name);
        return it == req.headers.end() ? std::string() : it->second;
    }

    bool host_ok(const Request& req) const {
        const auto it = req.headers.find("host");
        if (it == req.headers.end()) return false;
        const std::string want = ":" + std::to_string(port);
        const std::string h = lower(it->second);
        return h == "127.0.0.1" + want || h == "localhost" + want;
    }

    bool token_ok(const Request& req) const {
        const std::string auth = header(req, "authorization");
        if (auth.size() > 7 && lower(auth.substr(0, 7)) == "bearer ")
            return token_equal(trim(auth.substr(7)), options.token);
        if (req.method == "GET" && !req.query.empty()) {
            std::size_t pos = 0;
            while (pos <= req.query.size()) {
                const auto amp = req.query.find('&', pos);
                const std::string part = req.query.substr(pos, amp == std::string::npos ? std::string::npos : amp - pos);
                if (part.rfind("token=", 0) == 0) return token_equal(part.substr(6), options.token);
                if (amp == std::string::npos) break;
                pos = amp + 1;
            }
        }
        return false;
    }

    // Reads the rest of a body of `length` bytes; false on a timeout or a
    // client that went away.
    bool read_body(int fd, Request& req, std::size_t length) {
        const std::int64_t deadline = now_ms() + options.body_timeout_ms;
        char chunk[16384];
        while (req.body.size() < length) {
            const std::int64_t left = deadline - now_ms();
            if (left <= 0) return false;
            pollfd p{fd, POLLIN, 0};
            const int r = ::poll(&p, 1, static_cast<int>(left));
            if (r < 0 && errno == EINTR) continue;
            if (r <= 0) return false;
            const ssize_t n = ::recv(fd, chunk, std::min(sizeof chunk, length - req.body.size()), 0);
            if (n < 0 && (errno == EINTR || errno == EAGAIN)) continue;
            if (n <= 0) return false;
            req.body.append(chunk, static_cast<std::size_t>(n));
        }
        if (req.body.size() > length) req.body.resize(length);
        return true;
    }

    Response envelope_error(const std::string& id, const std::string& code, const std::string& message,
                            const std::string& where) {
        return json_response(command_result_to_json(CommandResult::failure(id, Error(code, message, where))));
    }

    Response handle(int fd, Request& req) {
        if (!host_ok(req)) return error_response(403, "The Host header is refused.");
        const auto origin = req.headers.find("origin");
        if (origin != req.headers.end() && !loopback_origin(origin->second))
            return error_response(403, "The Origin header is refused.");
        if (req.headers.count("transfer-encoding"))
            return error_response(501, "Transfer-Encoding is not accepted; send the body with Content-Length.");

        const std::string prefix = "/v0/commands/";
        std::string name;
        if (req.path == "/v0/commands") {
            if (req.method != "GET") {
                Response r = error_response(405, "GET /v0/commands lists the commands.");
                r.headers.emplace_back("Allow", "GET");
                return r;
            }
        } else if (req.path.size() > prefix.size() && req.path.compare(0, prefix.size(), prefix) == 0) {
            name = req.path.substr(prefix.size());
            if (name.find('/') != std::string::npos || name.size() > 128) return error_response(404, "No such endpoint.");
            if (req.method != "POST") {
                Response r = error_response(405, "POST /v0/commands/{name} runs a command.");
                r.headers.emplace_back("Allow", "POST");
                return r;
            }
        } else {
            return error_response(404, "No such endpoint.");
        }
        if (!token_ok(req)) {
            Response r = error_response(401, "The session token is missing or wrong.");
            r.headers.emplace_back("WWW-Authenticate", "Bearer");
            return r;
        }

        std::size_t length = 0;
        const std::string cl = header(req, "content-length");
        if (!cl.empty()) {
            if (cl.size() > 18) return error_response(413, "The body is larger than 1 MiB.");
            for (char c : cl)
                if (!std::isdigit(static_cast<unsigned char>(c))) return error_response(400, "Content-Length is not a number.");
            length = static_cast<std::size_t>(std::stoull(cl));
        }
        if (req.method == "GET") {
            if (length > 0 || !req.body.empty()) return error_response(413, "A GET takes no body.");
            return json_response(command_table_to_json());
        }
        if (cl.empty()) return error_response(411, "A POST needs Content-Length.");
        if (length > options.max_body_bytes) return error_response(413, "The body is larger than 1 MiB.");
        std::string type = lower(header(req, "content-type"));
        type = trim(type.substr(0, type.find(';')));
        if (type != "application/json") return error_response(415, "The body must be application/json.");
        if (!read_body(fd, req, length)) return error_response(408, "The body did not arrive in time.");

        const std::string id = "http-" + std::to_string(next_id.fetch_add(1));
        if (json_depth(req.body, kMaxJsonDepth) > kMaxJsonDepth)
            return envelope_error(id, "command.invalid_args", "The arguments nest more than 64 levels deep.", "args");
        json args = json::parse(req.body, nullptr, false);
        if (args.is_discarded()) return error_response(400, "The body is not JSON.");
        if (!args.is_object()) return envelope_error(id, "command.invalid_args", "The arguments are a JSON object.", "args");

        CommandInvocation inv;
        inv.id = id;
        inv.name = name;
        inv.args = std::move(args);
        // The origin belongs to the token, not to the request.
        inv.origin = options.origin;
        const std::string claimed = header(req, "x-caliper-origin");
        if (!claimed.empty() && claimed != command_origin_name(options.origin))
            return error_response(400, std::string("This server's token is for the origin ") +
                                           command_origin_name(options.origin) +
                                           "; X-Caliper-Origin cannot name another.");
        inv.group = header(req, "x-caliper-group");
        inv.preview = header(req, "x-caliper-preview");

        SubmitOptions submit;
        if (inv.origin == CommandOrigin::api) {
            submit.wait_for_approval = false;
        } else {
            submit.wait_for_approval = true;
            submit.approval_timeout = options.approval_wait;
        }
        std::future<CommandResult> future = queue.submit(std::move(inv), submit);
        if (future.wait_for(options.max_command_wait) != std::future_status::ready)
            return error_response(504, "The command did not finish in time.");
        return json_response(command_result_to_json(future.get()));
    }

    void serve(int fd) {
        try {
#ifdef SO_NOSIGPIPE
            int one = 1;
            ::setsockopt(fd, SOL_SOCKET, SO_NOSIGPIPE, &one, sizeof one);
#endif
            Request req;
            const int status = read_head(fd, req);
            if (status == 0) {
                ::close(fd);
                return;
            }
            if (status != 200) {
                send_response(fd, error_response(status, status == 431 ? "The request head is too large."
                                                          : status == 408 ? "The request did not arrive in time."
                                                                          : "The request is malformed."));
                finish(fd);
                return;
            }
            send_response(fd, handle(fd, req));
            finish(fd);
        } catch (const std::exception&) {
            send_response(fd, error_response(503, "The request could not be served."));
            finish(fd);
        }
    }

    void accept_loop() {
        while (!stopping.load()) {
            pollfd fds[2] = {{listener, POLLIN, 0}, {wake[0], POLLIN, 0}};
            const int r = ::poll(fds, 2, 250);
            reap(false);
            if (r < 0 && errno == EINTR) continue;
            if (r <= 0) continue;
            if (fds[1].revents) break;
            if (!(fds[0].revents & POLLIN)) continue;
            sockaddr_in peer{};
            socklen_t len = sizeof peer;
            const int fd = ::accept(listener, reinterpret_cast<sockaddr*>(&peer), &len);
            if (fd < 0) continue;
            ::fcntl(fd, F_SETFD, FD_CLOEXEC);
            if (active() >= options.max_connections) {
                // Answered here, quickly; the client may retry.
                Response busy = error_response(503, "Too many connections; try again.");
                busy.headers.emplace_back("Retry-After", "1");
                std::string text = "HTTP/1.1 503 Service Unavailable\r\nContent-Type: application/json\r\nContent-Length: " +
                                   std::to_string(busy.body.size()) +
                                   "\r\nRetry-After: 1\r\nCache-Control: no-store\r\nX-Content-Type-Options: "
                                   "nosniff\r\nConnection: close\r\n\r\n" +
                                   busy.body;
#ifdef SO_NOSIGPIPE
                int one = 1;
                ::setsockopt(fd, SOL_SOCKET, SO_NOSIGPIPE, &one, sizeof one);
#endif
                write_all(fd, text, 1000);
                ::shutdown(fd, SHUT_RDWR);
                ::close(fd);
                served.fetch_add(1);
                continue;
            }
            auto done = std::make_shared<std::atomic<bool>>(false);
            std::lock_guard<std::mutex> lock(workers_mutex);
            workers.push_back(Worker{std::thread([this, fd, done] {
                                         serve(fd);
                                         done->store(true);
                                     }),
                                     done});
        }
    }
};

CommandServer::CommandServer(CommandQueue& queue, CommandServerOptions options)
    : impl_(std::make_unique<Impl>(queue, std::move(options))) {}

CommandServer::~CommandServer() { stop(); }

Error CommandServer::start() {
    Impl& im = *impl_;
    if (im.listener >= 0) return Error("server.started", "The server is running already.");
    if (im.options.token.empty()) return Error("server.no_token", "The server needs a session token.");
    if (im.options.origin != CommandOrigin::mcp && im.options.origin != CommandOrigin::agent &&
        im.options.origin != CommandOrigin::api)
        return Error("server.bad_origin", "The server's origin is mcp, agent or api.");
    if (::pipe(im.wake) != 0) return Error("server.io", "The server could not make its wake pipe.");
    ::fcntl(im.wake[0], F_SETFD, FD_CLOEXEC);
    ::fcntl(im.wake[1], F_SETFD, FD_CLOEXEC);
    const int fd = ::socket(AF_INET, SOCK_STREAM, 0);
    if (fd < 0) return Error("server.io", "The server could not make a socket.");
    ::fcntl(fd, F_SETFD, FD_CLOEXEC);
    int one = 1;
    ::setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, &one, sizeof one);
    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_port = htons(im.options.port);
    addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    if (::bind(fd, reinterpret_cast<sockaddr*>(&addr), sizeof addr) != 0 || ::listen(fd, 64) != 0) {
        ::close(fd);
        return Error("server.port_taken", "The port " + std::to_string(im.options.port) + " cannot be bound on 127.0.0.1.");
    }
    socklen_t len = sizeof addr;
    ::getsockname(fd, reinterpret_cast<sockaddr*>(&addr), &len);
    im.port = ntohs(addr.sin_port);
    im.listener = fd;
    im.acceptor = std::thread([&im] { im.accept_loop(); });
    return {};
}

std::uint16_t CommandServer::port() const { return impl_->port; }

std::uint64_t CommandServer::served() const { return impl_->served.load(); }

void CommandServer::stop() {
    Impl& im = *impl_;
    if (im.listener < 0) return;
    im.stopping.store(true);
    const char byte = 1;
    if (::write(im.wake[1], &byte, 1) < 0) {
        // The acceptor also wakes on its poll timeout.
    }
    if (im.acceptor.joinable()) im.acceptor.join();
    ::close(im.listener);
    im.listener = -1;
    im.reap(true);
    ::close(im.wake[0]);
    ::close(im.wake[1]);
    im.wake[0] = im.wake[1] = -1;
}

Error write_api_file(const std::string& dir, std::uint16_t port, const std::string& token, std::string* path) {
    if (dir.empty()) return Error("server.io", "The API folder is not given.");
    std::string folder = dir;
    while (folder.size() > 1 && folder.back() == '/') folder.pop_back();
    struct stat st;
    if (::stat(folder.c_str(), &st) != 0) {
        if (::mkdir(folder.c_str(), 0700) != 0) return Error("server.io", "The API folder could not be created.", folder);
    } else if (!S_ISDIR(st.st_mode)) {
        return Error("server.io", "The API folder is not a folder.", folder);
    }
    const std::string target = folder + "/api.json";
    const std::string temp = folder + "/.api.json." + std::to_string(::getpid()) + ".tmp";
    ::unlink(temp.c_str());
    const int fd = ::open(temp.c_str(), O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC, 0600);
    if (fd < 0) return Error("server.io", "The API file could not be written.", target);
    ::fchmod(fd, 0600);
    const std::string text = json{{"version", 1},
                                  {"pid", static_cast<std::int64_t>(::getpid())},
                                  {"port", port},
                                  {"token", token},
                                  {"url", "http://127.0.0.1:" + std::to_string(port) + "/v0"}}
                                 .dump() +
                             "\n";
    std::size_t written = 0;
    while (written < text.size()) {
        const ssize_t n = ::write(fd, text.data() + written, text.size() - written);
        if (n < 0 && errno == EINTR) continue;
        if (n <= 0) break;
        written += static_cast<std::size_t>(n);
    }
    ::fsync(fd);
    ::close(fd);
    if (written != text.size() || ::rename(temp.c_str(), target.c_str()) != 0) {
        ::unlink(temp.c_str());
        return Error("server.io", "The API file could not be written.", target);
    }
    if (path) *path = target;
    return {};
}

void remove_api_file(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) return;
    std::string text((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
    if (text.size() > 4096) return;
    const json value = json::parse(text, nullptr, false);
    if (!value.is_object() || !value.contains("pid") || !value["pid"].is_number_integer()) return;
    if (value["pid"].get<std::int64_t>() != static_cast<std::int64_t>(::getpid())) return;
    ::unlink(path.c_str());
}

#endif

}  // namespace caliper_builder
