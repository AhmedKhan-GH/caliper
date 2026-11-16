// The loopback HTTP/1.1 client (loopback_http.h).
#include <caliper_builder/mcp/loopback_http.h>

#include <algorithm>
#include <chrono>
#include <cstring>

#ifndef _WIN32
#include <arpa/inet.h>
#include <cerrno>
#include <fcntl.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <poll.h>
#include <sys/socket.h>
#include <unistd.h>
#endif

namespace caliper_builder::mcp {

std::string HttpResponse::header(const std::string& lower_name) const {
    for (const auto& h : headers)
        if (h.name == lower_name) return h.value;
    return {};
}

namespace {

bool is_tchar(char c) {
    if ((c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9')) return true;
    return std::strchr("!#$%&'*+-.^_`|~", c) != nullptr && c != '\0';
}

std::string lower(std::string s) {
    for (char& c : s)
        if (c >= 'A' && c <= 'Z') c = char(c - 'A' + 'a');
    return s;
}

// A header value may hold visible ASCII, space, tab and bytes from 0x80.
bool value_ok(const std::string& v) {
    for (unsigned char c : v)
        if ((c < 0x20 && c != '\t') || c == 0x7f) return false;
    return true;
}

Error check_request(const std::string& host, int port, const HttpRequest& r) {
    if (host != "127.0.0.1")
        return {"http.refused_host", "This client connects only to 127.0.0.1.", "host"};
    if (port < 1 || port > 65535) return {"http.bad_request", "The port is not between 1 and 65535.", "port"};
    if (r.method.empty() || r.method.size() > 16) return {"http.bad_request", "The method is not a token.", "method"};
    for (char c : r.method)
        if (c < 'A' || c > 'Z') return {"http.bad_request", "The method is not an upper-case token.", "method"};
    if (r.target.empty() || r.target[0] != '/' || r.target.size() > 8192)
        return {"http.bad_request", "The target must start with '/' and have at most 8192 characters.", "target"};
    for (unsigned char c : r.target)
        if (c <= 0x20 || c >= 0x7f) return {"http.bad_request", "The target holds a character that is not visible ASCII.", "target"};
    for (const auto& h : r.headers) {
        if (h.name.empty() || h.name.size() > 256) return {"http.bad_request", "A header name is empty or too long.", "headers"};
        for (char c : h.name)
            if (!is_tchar(c)) return {"http.bad_request", "A header name is not a token.", "headers"};
        const std::string n = lower(h.name);
        if (n == "host" || n == "connection" || n == "content-length" || n == "transfer-encoding")
            return {"http.bad_request", "The client sets Host, Connection and Content-Length itself.", "headers"};
        // The value is never quoted in a message: it may be a token.
        if (!value_ok(h.value) || h.value.size() > 8192)
            return {"http.bad_request", "A header value holds a control character or is too long.", "headers"};
    }
    return {};
}

}  // namespace

#ifdef _WIN32

HttpResponse loopback_http_request(const std::string& host, int port, const HttpRequest& request,
                                   const HttpLimits&, const std::atomic<bool>*) {
    HttpResponse out;
    out.error = check_request(host, port, request);
    if (out.error) return out;
    out.error = {"http.unsupported", "The loopback HTTP client is not available on Windows yet.", ""};
    return out;
}

#else

namespace {

using Clock = std::chrono::steady_clock;

struct Fd {
    int fd = -1;
    ~Fd() {
        if (fd >= 0) ::close(fd);
    }
};

std::string where_of(int port) { return "127.0.0.1:" + std::to_string(port); }

int ms_left(Clock::time_point deadline) {
    const auto left = std::chrono::duration_cast<std::chrono::milliseconds>(deadline - Clock::now()).count();
    return left <= 0 ? 0 : int(std::min<long long>(left, 1 << 30));
}

// Parses a response head (without the blank line). False with a message.
bool parse_head(const std::string& head, const HttpLimits& limits, HttpResponse* out, std::string* why) {
    std::size_t pos = head.find("\r\n");
    const std::string status_line = head.substr(0, pos);
    if (status_line.size() < 12 || (status_line.compare(0, 9, "HTTP/1.1 ") != 0 && status_line.compare(0, 9, "HTTP/1.0 ") != 0)) {
        *why = "the status line is not HTTP/1.x";
        return false;
    }
    for (int i = 9; i < 12; ++i)
        if (status_line[i] < '0' || status_line[i] > '9') {
            *why = "the status code is not three digits";
            return false;
        }
    if (status_line.size() > 12 && status_line[12] != ' ') {
        *why = "the status code is not followed by a space";
        return false;
    }
    out->status = std::stoi(status_line.substr(9, 3));
    out->reason = status_line.size() > 13 ? status_line.substr(13) : std::string();
    if (!value_ok(out->reason)) {
        *why = "the reason phrase holds a control character";
        return false;
    }
    while (pos != std::string::npos) {
        const std::size_t start = pos + 2;
        pos = head.find("\r\n", start);
        const std::string line = head.substr(start, pos == std::string::npos ? std::string::npos : pos - start);
        if (line.empty()) {
            *why = "an empty line inside the head";
            return false;
        }
        if (line[0] == ' ' || line[0] == '\t') {
            *why = "a folded header line";
            return false;
        }
        const std::size_t colon = line.find(':');
        if (colon == std::string::npos || colon == 0) {
            *why = "a header line without a name";
            return false;
        }
        HttpHeader h;
        h.name = line.substr(0, colon);
        for (char c : h.name)
            if (!is_tchar(c)) {
                *why = "a header name that is not a token";
                return false;
            }
        h.name = lower(h.name);
        std::size_t a = colon + 1, b = line.size();
        while (a < b && (line[a] == ' ' || line[a] == '\t')) ++a;
        while (b > a && (line[b - 1] == ' ' || line[b - 1] == '\t')) --b;
        h.value = line.substr(a, b - a);
        if (!value_ok(h.value)) {
            *why = "a header value holds a control character";
            return false;
        }
        if (out->headers.size() >= limits.max_header_count) {
            *why = "more than " + std::to_string(limits.max_header_count) + " header fields";
            return false;
        }
        out->headers.push_back(std::move(h));
    }
    return true;
}

// The declared body length: -1 when there is none, -2 when it is malformed.
long long content_length(const HttpResponse& r) {
    long long found = -1;
    for (const auto& h : r.headers) {
        if (h.name != "content-length") continue;
        if (h.value.empty() || h.value.size() > 18) return -2;
        long long v = 0;
        for (char c : h.value) {
            if (c < '0' || c > '9') return -2;
            v = v * 10 + (c - '0');
        }
        if (found >= 0 && found != v) return -2;
        found = v;
    }
    return found;
}

}  // namespace

HttpResponse loopback_http_request(const std::string& host, int port, const HttpRequest& request,
                                   const HttpLimits& limits, const std::atomic<bool>* cancel) {
    HttpResponse out;
    out.error = check_request(host, port, request);
    if (out.error) return out;
    const std::string where = where_of(port);
    const auto start = Clock::now();
    const auto deadline = start + std::chrono::milliseconds(std::max(1, limits.total_timeout_ms));
    auto cancelled = [&] { return cancel && cancel->load(); };
    auto fail = [&](const char* code, std::string message) {
        out.error = {code, std::move(message), where};
        return out;
    };

    // The request bytes.
    std::string wire;
    wire.reserve(256 + request.body.size());
    wire += request.method + " " + request.target + " HTTP/1.1\r\n";
    wire += "Host: 127.0.0.1:" + std::to_string(port) + "\r\n";
    wire += "Connection: close\r\n";
    for (const auto& h : request.headers) wire += h.name + ": " + h.value + "\r\n";
    if (!request.body.empty() || request.method == "POST" || request.method == "PUT")
        wire += "Content-Length: " + std::to_string(request.body.size()) + "\r\n";
    wire += "\r\n";
    wire += request.body;

    Fd sock;
    sock.fd = ::socket(AF_INET, SOCK_STREAM, 0);
    if (sock.fd < 0) return fail("http.connect_failed", std::string("No socket: ") + std::strerror(errno) + ".");
    (void)fcntl(sock.fd, F_SETFD, FD_CLOEXEC);
    const int fl = fcntl(sock.fd, F_GETFL);
    if (fl < 0 || fcntl(sock.fd, F_SETFL, fl | O_NONBLOCK) != 0)
        return fail("http.connect_failed", "The socket cannot be made non-blocking.");
#ifdef SO_NOSIGPIPE
    {
        int one = 1;
        (void)setsockopt(sock.fd, SOL_SOCKET, SO_NOSIGPIPE, &one, sizeof one);
    }
#endif
#ifdef MSG_NOSIGNAL
    const int send_flags = MSG_NOSIGNAL;
#else
    const int send_flags = 0;
#endif

    // Connect.
    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_port = htons(static_cast<uint16_t>(port));
    addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    if (::connect(sock.fd, reinterpret_cast<sockaddr*>(&addr), sizeof addr) != 0) {
        if (errno == ECONNREFUSED) return fail("http.connect_refused", "Nothing listens on " + where + ".");
        if (errno != EINPROGRESS && errno != EINTR)
            return fail("http.connect_failed", "The connection to " + where + " failed: " + std::strerror(errno) + ".");
        const auto connect_deadline =
            std::min(deadline, start + std::chrono::milliseconds(std::max(1, limits.connect_timeout_ms)));
        for (;;) {
            if (cancelled()) return fail("http.cancelled", "The request was cancelled.");
            const int left = ms_left(connect_deadline);
            if (left == 0) return fail("http.connect_timeout", "No connection to " + where + " in time.");
            pollfd p{sock.fd, POLLOUT, 0};
            const int r = ::poll(&p, 1, std::min(left, 100));
            if (r < 0 && errno != EINTR) return fail("http.connect_failed", std::string("poll failed: ") + std::strerror(errno) + ".");
            if (r > 0) break;
        }
        int soerr = 0;
        socklen_t len = sizeof soerr;
        if (getsockopt(sock.fd, SOL_SOCKET, SO_ERROR, &soerr, &len) != 0) soerr = errno;
        if (soerr == ECONNREFUSED) return fail("http.connect_refused", "Nothing listens on " + where + ".");
        if (soerr != 0)
            return fail("http.connect_failed", "The connection to " + where + " failed: " + std::strerror(soerr) + ".");
    }

    // Send. A server may answer early and close (a refusal); then the answer is
    // still read below.
    std::size_t sent = 0;
    bool send_broken = false;
    while (sent < wire.size()) {
        if (cancelled()) return fail("http.cancelled", "The request was cancelled.");
        const int left = ms_left(deadline);
        if (left == 0) return fail("http.timeout", "The request to " + where + " was not sent in time.");
        pollfd p{sock.fd, POLLOUT, 0};
        const int r = ::poll(&p, 1, std::min(left, 100));
        if (r < 0) {
            if (errno == EINTR) continue;
            return fail("http.io_error", std::string("poll failed: ") + std::strerror(errno) + ".");
        }
        if (r == 0) continue;
        const ssize_t n = ::send(sock.fd, wire.data() + sent, wire.size() - sent, send_flags);
        if (n > 0) {
            sent += std::size_t(n);
        } else if (n < 0 && (errno == EAGAIN || errno == EWOULDBLOCK || errno == EINTR)) {
            continue;
        } else {
            send_broken = true;
            break;
        }
    }
    // The writing side stays open: a server may take an early end of input for a
    // client that has gone away and drop the request.

    // Read until the server closes.
    std::string buf;
    std::size_t head_end = std::string::npos;   // the index after the blank line
    long long declared = -1;
    bool head_parsed = false;
    auto last_byte = Clock::now();
    bool eof = false;
    bool reset = false;
    char chunk[16384];
    while (!eof) {
        if (cancelled()) return fail("http.cancelled", "The request was cancelled.");
        const int left_total = ms_left(deadline);
        if (left_total == 0) return fail("http.timeout", "No complete answer from " + where + " in time.");
        const int left_idle = ms_left(last_byte + std::chrono::milliseconds(std::max(1, limits.read_timeout_ms)));
        if (left_idle == 0) return fail("http.read_timeout", "The answer from " + where + " stalled.");
        pollfd p{sock.fd, POLLIN, 0};
        const int r = ::poll(&p, 1, std::min({left_total, left_idle, 100}));
        if (r < 0) {
            if (errno == EINTR) continue;
            return fail("http.io_error", std::string("poll failed: ") + std::strerror(errno) + ".");
        }
        if (r == 0) continue;
        const ssize_t n = ::recv(sock.fd, chunk, sizeof chunk, 0);
        if (n == 0) {
            eof = true;
            break;
        }
        if (n < 0) {
            if (errno == EAGAIN || errno == EWOULDBLOCK || errno == EINTR) continue;
            if (errno == ECONNRESET) {
                reset = true;
                break;
            }
            return fail("http.io_error", std::string("Reading the answer failed: ") + std::strerror(errno) + ".");
        }
        last_byte = Clock::now();
        buf.append(chunk, std::size_t(n));
        if (!head_parsed) {
            head_end = buf.find("\r\n\r\n");
            if (head_end == std::string::npos) {
                if (buf.size() > limits.max_head_bytes)
                    return fail("http.too_large", "The response head exceeds " + std::to_string(limits.max_head_bytes) + " bytes.");
                continue;
            }
            if (head_end > limits.max_head_bytes)
                return fail("http.too_large", "The response head exceeds " + std::to_string(limits.max_head_bytes) + " bytes.");
            std::string why;
            if (!parse_head(buf.substr(0, head_end), limits, &out, &why))
                return fail("http.bad_response", "The response from " + where + " is malformed: " + why + ".");
            head_end += 4;
            head_parsed = true;
            if (!out.header("transfer-encoding").empty())
                return fail("http.bad_response", "The response uses a transfer coding, which this client does not take.");
            declared = content_length(out);
            if (declared == -2) return fail("http.bad_response", "The response has a malformed Content-Length.");
            if (declared >= 0 && std::size_t(declared) > limits.max_body_bytes)
                return fail("http.too_large", "The response body exceeds " + std::to_string(limits.max_body_bytes) + " bytes.");
        }
        const std::size_t body_now = buf.size() - head_end;
        if (body_now > limits.max_body_bytes)
            return fail("http.too_large", "The response body exceeds " + std::to_string(limits.max_body_bytes) + " bytes.");
        if (declared >= 0 && body_now > std::size_t(declared))
            return fail("http.bad_response", "The response holds more bytes than its Content-Length.");
    }

    if (!head_parsed) {
        if (buf.empty() && (reset || send_broken))
            return fail("http.io_error", "The connection to " + where + " was closed before an answer.");
        return fail("http.bad_response", buf.empty() ? "The server closed the connection without an answer."
                                                     : "The response ended inside its head.");
    }
    out.body = buf.substr(head_end);
    if (declared >= 0 && out.body.size() != std::size_t(declared)) {
        if (reset) return fail("http.io_error", "The connection was reset before the whole body arrived.");
        return fail("http.bad_response", "The response body is shorter than its Content-Length.");
    }
    if (out.status < 200) return fail("http.bad_response", "An interim (1xx) response was not expected.");
    if (out.status >= 300 && out.status < 400) {
        out.body.clear();
        return fail("http.redirect", "The server answered with a redirect (" + std::to_string(out.status) +
                                         "), which this client does not follow.");
    }
    return out;
}

#endif

}  // namespace caliper_builder::mcp
