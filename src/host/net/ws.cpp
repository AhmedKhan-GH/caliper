// ws.cpp: the WebSocket layer of ws.h. One Conn per TCP connection carries the
// HTTP handshake, the frame parser and the send queue; Server and Client own
// Conns and drive them from poll(2). See ws.h for the model.
#include "net/ws.h"

#include "net/sha1.h"

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cstring>
#include <deque>
#include <random>

#ifdef _WIN32
#  ifndef WIN32_LEAN_AND_MEAN
#    define WIN32_LEAN_AND_MEAN
#  endif
#  ifndef NOMINMAX
#    define NOMINMAX
#  endif
#  include <winsock2.h>
#  include <ws2tcpip.h>
// The Winsock differences, kept to these few names.
using sock_t = SOCKET;
using pollfd_t = WSAPOLLFD;
#  define CNET_BAD_SOCKET INVALID_SOCKET
#  define cnet_closesocket(s) ::closesocket(s)
#  define cnet_errno() ::WSAGetLastError()
#  define cnet_poll(f, n, t) ::WSAPoll((f), (ULONG)(n), (t))
#  define CNET_WOULDBLOCK(e) ((e) == WSAEWOULDBLOCK)
#  define CNET_INPROGRESS(e) ((e) == WSAEWOULDBLOCK || (e) == WSAEINPROGRESS)
#  define CNET_INTR(e) ((e) == WSAEINTR)
#  define CNET_SEND_FLAGS 0
#  define CNET_SHUT_WR SD_SEND
#else
#  include <arpa/inet.h>
#  include <cerrno>
#  include <fcntl.h>
#  include <netinet/in.h>
#  include <netinet/tcp.h>
#  include <poll.h>
#  include <sys/socket.h>
#  include <unistd.h>
using sock_t = int;
using pollfd_t = struct pollfd;
#  define CNET_BAD_SOCKET (-1)
#  define cnet_closesocket(s) ::close(s)
#  define cnet_errno() errno
#  define cnet_poll(f, n, t) ::poll((f), (nfds_t)(n), (t))
#  define CNET_WOULDBLOCK(e) ((e) == EAGAIN || (e) == EWOULDBLOCK)
#  define CNET_INPROGRESS(e) ((e) == EINPROGRESS)
#  define CNET_INTR(e) ((e) == EINTR)
#  ifdef MSG_NOSIGNAL
#    define CNET_SEND_FLAGS MSG_NOSIGNAL
#  else
#    define CNET_SEND_FLAGS 0   // Apple: SO_NOSIGPIPE on the socket instead
#  endif
#  define CNET_SHUT_WR SHUT_WR
#endif

namespace caliper_net {

uint64_t monotonic_ms() {
    using namespace std::chrono;
    return (uint64_t)duration_cast<milliseconds>(steady_clock::now().time_since_epoch()).count();
}

namespace {

constexpr const char* kWsGuid = "258EAFA5-E914-47DA-95CA-C5AB0DC85B11";
constexpr size_t kReadChunk = 64 * 1024;
constexpr size_t kReadBudget = 4u << 20;   // bytes read per connection per poll (fairness)

// ---- sockets ------------------------------------------------------------------

void net_init() {
#ifdef _WIN32
    static const bool ok = [] {
        WSADATA d;
        return ::WSAStartup(MAKEWORD(2, 2), &d) == 0;
    }();
    (void)ok;
#endif
}

std::string sock_error_text(int e) {
#ifdef _WIN32
    return "socket error " + std::to_string(e);
#else
    return std::strerror(e);
#endif
}

bool set_nonblocking(sock_t fd) {
#ifdef _WIN32
    u_long on = 1;
    return ::ioctlsocket(fd, FIONBIO, &on) == 0;
#else
    const int fl = ::fcntl(fd, F_GETFL, 0);
    return fl >= 0 && ::fcntl(fd, F_SETFL, fl | O_NONBLOCK) == 0;
#endif
}

void tune_socket(sock_t fd) {
    int on = 1;
    ::setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, (const char*)&on, sizeof on);
#ifdef SO_NOSIGPIPE
    ::setsockopt(fd, SOL_SOCKET, SO_NOSIGPIPE, (const char*)&on, sizeof on);
#endif
}

long long sock_send(sock_t fd, const uint8_t* p, size_t n) {
#ifdef _WIN32
    return ::send(fd, (const char*)p, (int)std::min<size_t>(n, 1u << 30), 0);
#else
    return ::send(fd, p, n, CNET_SEND_FLAGS);
#endif
}

long long sock_recv(sock_t fd, uint8_t* p, size_t n) {
#ifdef _WIN32
    return ::recv(fd, (char*)p, (int)n, 0);
#else
    return ::recv(fd, p, n, 0);
#endif
}

// Numeric IPv4 or "localhost" (127.0.0.1); no name resolution on purpose.
bool resolve_ipv4(const std::string& host, in_addr* out) {
    const std::string h = host == "localhost" ? "127.0.0.1" : host;
    return ::inet_pton(AF_INET, h.c_str(), out) == 1;
}

// ---- strings --------------------------------------------------------------------

std::string lower(std::string s) {
    for (char& c : s) c = (char)std::tolower((unsigned char)c);
    return s;
}

std::string trim(const std::string& s) {
    size_t a = 0, b = s.size();
    while (a < b && (s[a] == ' ' || s[a] == '\t')) ++a;
    while (b > a && (s[b - 1] == ' ' || s[b - 1] == '\t')) --b;
    return s.substr(a, b - a);
}

// True when the comma-separated header value holds `token` (case-insensitive).
bool has_token(const std::string& value, const char* token) {
    size_t i = 0;
    const std::string want = lower(token);
    while (i <= value.size()) {
        size_t j = value.find(',', i);
        if (j == std::string::npos) j = value.size();
        if (lower(trim(value.substr(i, j - i))) == want) return true;
        i = j + 1;
    }
    return false;
}

const char* status_text(int status) {
    switch (status) {
        case 101: return "Switching Protocols";
        case 200: return "OK";
        case 204: return "No Content";
        case 301: return "Moved Permanently";
        case 302: return "Found";
        case 304: return "Not Modified";
        case 400: return "Bad Request";
        case 401: return "Unauthorized";
        case 403: return "Forbidden";
        case 404: return "Not Found";
        case 405: return "Method Not Allowed";
        case 408: return "Request Timeout";
        case 426: return "Upgrade Required";
        case 429: return "Too Many Requests";
        case 431: return "Request Header Fields Too Large";
        case 500: return "Internal Server Error";
        case 503: return "Service Unavailable";
        default: return "Status";
    }
}

std::string accept_key(const std::string& client_key) {
    const std::string s = client_key + kWsGuid;
    uint8_t d[20];
    sha1(s.data(), s.size(), d);
    return base64_encode(d, sizeof d);
}

// Splits an HTTP head (without the final CRLF CRLF) into its first line and
// lower-cased header fields. False on a malformed field or an obsolete folded
// line.
bool split_head(const std::string& head, std::string* first,
                std::vector<std::pair<std::string, std::string>>* fields) {
    size_t pos = head.find("\r\n");
    *first = head.substr(0, pos);
    while (pos != std::string::npos && pos + 2 <= head.size()) {
        const size_t start = pos + 2;
        if (start >= head.size()) break;
        size_t end = head.find("\r\n", start);
        if (end == std::string::npos) end = head.size();
        const std::string line = head.substr(start, end - start);
        if (line.empty()) break;
        if (line[0] == ' ' || line[0] == '\t') return false;   // obs-fold
        const size_t colon = line.find(':');
        if (colon == std::string::npos || colon == 0) return false;
        const std::string name = line.substr(0, colon);
        for (char c : name)
            if (c == ' ' || c == '\t') return false;
        fields->emplace_back(lower(name), trim(line.substr(colon + 1)));
        pos = end == head.size() ? std::string::npos : end;
    }
    return true;
}

std::string field(const std::vector<std::pair<std::string, std::string>>& f, const char* name) {
    for (const auto& kv : f)
        if (kv.first == name) return kv.second;
    return {};
}

bool has_field(const std::vector<std::pair<std::string, std::string>>& f, const char* name) {
    for (const auto& kv : f)
        if (kv.first == name) return true;
    return false;
}

std::mt19937_64& rng() {
    static thread_local std::mt19937_64 g{[] {
        std::random_device rd;
        return ((uint64_t)rd() << 32) ^ rd();
    }()};
    return g;
}

void random_fill(uint8_t* p, size_t n) {
    std::random_device rd;
    for (size_t i = 0; i < n; i += 4) {
        const uint32_t v = rd();
        for (size_t j = 0; j < 4 && i + j < n; ++j) p[i + j] = uint8_t(v >> (8 * j));
    }
}

// XOR `n` bytes with the 4-byte key, starting at key position `pos`.
void apply_mask(uint8_t* p, size_t n, const uint8_t key[4], size_t pos) {
    size_t i = 0;
    // Byte-wise until the key phase is 0, then 8 bytes at a time.
    while (i < n && ((pos + i) & 3) != 0) { p[i] ^= key[(pos + i) & 3]; ++i; }
    uint64_t k8;
    uint8_t kk[8] = {key[0], key[1], key[2], key[3], key[0], key[1], key[2], key[3]};
    std::memcpy(&k8, kk, 8);
    for (; i + 8 <= n; i += 8) {
        uint64_t w;
        std::memcpy(&w, p + i, 8);
        w ^= k8;
        std::memcpy(p + i, &w, 8);
    }
    for (; i < n; ++i) p[i] ^= key[(pos + i) & 3];
}

bool valid_close_code(uint16_t c) {
    if (c >= 3000 && c <= 4999) return true;
    return c == 1000 || c == 1001 || c == 1002 || c == 1003 || (c >= 1007 && c <= 1011);
}

}  // namespace

// ---- HttpRequest, URLs ------------------------------------------------------------

std::string HttpRequest::header(const std::string& lower_name) const {
    for (const auto& kv : headers)
        if (kv.first == lower_name) return kv.second;
    return {};
}

bool HttpRequest::query_param(const std::string& name, std::string* value) const {
    size_t i = 0;
    while (i <= query.size()) {
        size_t j = query.find('&', i);
        if (j == std::string::npos) j = query.size();
        const std::string part = query.substr(i, j - i);
        const size_t eq = part.find('=');
        std::string k, v;
        if (url_decode(part.substr(0, eq), &k) && k == name) {
            if (eq == std::string::npos) v.clear();
            else if (!url_decode(part.substr(eq + 1), &v)) return false;
            *value = v;
            return true;
        }
        i = j + 1;
    }
    return false;
}

bool url_decode(const std::string& in, std::string* out) {
    out->clear();
    for (size_t i = 0; i < in.size(); ++i) {
        const char c = in[i];
        if (c == '+') {
            out->push_back(' ');
        } else if (c == '%') {
            if (i + 2 >= in.size() || !std::isxdigit((unsigned char)in[i + 1]) ||
                !std::isxdigit((unsigned char)in[i + 2]))
                return false;
            out->push_back((char)std::stoi(in.substr(i + 1, 2), nullptr, 16));
            i += 2;
        } else {
            out->push_back(c);
        }
    }
    return true;
}

std::string url_encode(const std::string& in) {
    static const char* hex = "0123456789ABCDEF";
    std::string out;
    for (unsigned char c : in) {
        if (std::isalnum(c) || c == '-' || c == '_' || c == '.' || c == '~') {
            out.push_back((char)c);
        } else {
            out.push_back('%');
            out.push_back(hex[c >> 4]);
            out.push_back(hex[c & 15]);
        }
    }
    return out;
}

bool parse_ws_url(const std::string& url, WsUrl* out, std::string* err) {
    const std::string scheme = "ws://";
    if (lower(url.substr(0, scheme.size())) != scheme) {
        *err = "not a ws:// URL: " + url;
        return false;
    }
    std::string rest = url.substr(scheme.size());
    const size_t frag = rest.find('#');
    if (frag != std::string::npos) rest.resize(frag);
    const size_t slash = rest.find_first_of("/?");
    const std::string authority = rest.substr(0, slash);
    std::string target = slash == std::string::npos ? "/" : rest.substr(slash);
    if (target[0] == '?') target = "/" + target;
    if (authority.empty() || authority.find('@') != std::string::npos) {
        *err = "no host in " + url;
        return false;
    }
    WsUrl u;
    const size_t colon = authority.rfind(':');
    if (authority[0] == '[') {
        *err = "IPv6 hosts are not supported: " + url;
        return false;
    }
    if (colon == std::string::npos) {
        u.host = authority;
        u.port = 80;
    } else {
        u.host = authority.substr(0, colon);
        const std::string p = authority.substr(colon + 1);
        if (p.empty() || p.size() > 5 || !std::all_of(p.begin(), p.end(), ::isdigit) ||
            std::stoul(p) == 0 || std::stoul(p) > 65535) {
            *err = "bad port in " + url;
            return false;
        }
        u.port = (uint16_t)std::stoul(p);
    }
    if (u.host.empty()) {
        *err = "no host in " + url;
        return false;
    }
    u.host = lower(u.host);
    u.target = target;
    const size_t q = target.find('?');
    u.path = target.substr(0, q);
    u.query = q == std::string::npos ? std::string() : target.substr(q + 1);
    *out = u;
    return true;
}

// ---- the waker ----------------------------------------------------------------------
// A UDP socket on 127.0.0.1 connected to itself: wake() sends one datagram, a
// waiting poll() sees it readable. The same code works with Winsock.
class Waker {
public:
    Waker() {
        net_init();
        fd_ = ::socket(AF_INET, SOCK_DGRAM, 0);
        if (fd_ == CNET_BAD_SOCKET) return;
        sockaddr_in a{};
        a.sin_family = AF_INET;
        a.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
        a.sin_port = 0;
        socklen_t len = sizeof a;
        if (::bind(fd_, (sockaddr*)&a, sizeof a) != 0 || ::getsockname(fd_, (sockaddr*)&a, &len) != 0 ||
            ::connect(fd_, (sockaddr*)&a, sizeof a) != 0 || !set_nonblocking(fd_)) {
            cnet_closesocket(fd_);
            fd_ = CNET_BAD_SOCKET;
        }
    }
    ~Waker() {
        if (fd_ != CNET_BAD_SOCKET) cnet_closesocket(fd_);
    }
    sock_t fd() const { return fd_; }
    void wake() {
        if (fd_ == CNET_BAD_SOCKET) return;
        const uint8_t b = 1;
        (void)sock_send(fd_, &b, 1);
    }
    void drain() {
        uint8_t buf[64];
        for (int i = 0; i < 64; ++i)
            if (sock_recv(fd_, buf, sizeof buf) <= 0) break;
    }

private:
    sock_t fd_ = CNET_BAD_SOCKET;
};

// ---- one connection ---------------------------------------------------------------

class Conn {
public:
    enum class Role { Server, Client };
    enum class Phase {
        Connecting,     // client: TCP connect in progress
        ReadHead,       // server: reading the request head
        Decide,         // server: Request reported, waiting for accept/respond/reject
        AwaitUpgrade,   // client: request sent, reading the response head
        Open,
        Closing,        // a close frame went out; waiting for the peer's
        Flush,          // write what is queued, then end the TCP connection
        Dead,
    };

    Conn(ConnId id, sock_t fd, Role role, const Options& opt)
        : id_(id), fd_(fd), role_(role), opt_(opt) {
        last_input_ms_ = monotonic_ms();
    }
    ~Conn() { drop_socket(); }

    ConnId id() const { return id_; }
    sock_t fd() const { return fd_; }
    Role role() const { return role_; }
    Phase phase() const { return phase_; }
    void set_phase(Phase p) { phase_ = p; }
    void set_deadline(uint64_t t) { deadline_ms_ = t; }
    uint64_t deadline() const { return deadline_ms_; }
    uint64_t last_input_ms() const { return last_input_ms_; }
    size_t queued_bytes() const { return out_bytes_; }
    size_t queued_messages() const { return out_msgs_; }
    const HttpRequest& request() const { return req_; }
    bool dead() const { return phase_ == Phase::Dead; }
    bool reported() const { return reported_; }   // a Request or Open went out

    // poll(2) interest.
    short interest() const {
        short ev = 0;
        if (phase_ == Phase::Connecting) return POLLOUT;
        if (phase_ == Phase::ReadHead || phase_ == Phase::AwaitUpgrade || phase_ == Phase::Open ||
            phase_ == Phase::Closing)
            ev |= POLLIN;
        if (!out_.empty()) ev |= POLLOUT;
        return ev;
    }

    // ---- client setup -----------------------------------------------------------
    void client_start(const WsUrl& u, const std::string& origin,
                      const std::vector<std::pair<std::string, std::string>>& extra) {
        uint8_t k[16];
        random_fill(k, sizeof k);
        key_ = base64_encode(k, sizeof k);
        std::string host = u.host + ":" + std::to_string(u.port);
        for (const auto& kv : extra)
            if (lower(kv.first) == "host") host = kv.second;
        std::string r = "GET " + u.target + " HTTP/1.1\r\n";
        r += "Host: " + host + "\r\n";
        r += "Upgrade: websocket\r\nConnection: Upgrade\r\n";
        r += "Sec-WebSocket-Key: " + key_ + "\r\n";
        r += "Sec-WebSocket-Version: 13\r\n";
        if (!origin.empty()) r += "Origin: " + origin + "\r\n";
        for (const auto& kv : extra)
            if (lower(kv.first) != "host") r += kv.first + ": " + kv.second + "\r\n";
        r += "\r\n";
        queue_raw(r);
    }

    // ---- events from poll ---------------------------------------------------------
    void on_poll(short revents, std::vector<Event>& ev) {
        if (phase_ == Phase::Dead) return;
        if (phase_ == Phase::Connecting) {
            if (!(revents & (POLLOUT | POLLERR | POLLHUP))) return;
            int e = 0;
            socklen_t len = sizeof e;
            ::getsockopt(fd_, SOL_SOCKET, SO_ERROR, (char*)&e, &len);
            if (e != 0) {
                die(ev, kCloseAbnormal, "connect: " + sock_error_text(e));
                return;
            }
            phase_ = Phase::AwaitUpgrade;
            deadline_ms_ = monotonic_ms() + (uint64_t)opt_.header_timeout_ms;
        }
        if (revents & (POLLIN | POLLHUP | POLLERR)) {
            if (phase_ == Phase::Decide || phase_ == Phase::Flush) {
                // Not reading: only notice that the peer went away.
                if (revents & (POLLHUP | POLLERR)) die(ev, kCloseAbnormal, "the peer closed the connection");
            } else {
                read_some(ev);
            }
        }
        if (phase_ != Phase::Dead) flush(ev);
    }

    void on_timer(uint64_t now, std::vector<Event>& ev) {
        if (phase_ == Phase::Dead || deadline_ms_ == 0 || now < deadline_ms_) return;
        switch (phase_) {
            case Phase::Connecting: die(ev, kCloseAbnormal, "connect timed out"); break;
            case Phase::ReadHead: die(ev, kCloseAbnormal, "request head timed out"); break;
            case Phase::AwaitUpgrade: die(ev, kCloseAbnormal, "upgrade response timed out"); break;
            case Phase::Closing:
            case Phase::Flush: die(ev, close_code_ ? close_code_ : kCloseAbnormal, close_reason_); break;
            default: break;
        }
    }

    // ---- server decisions -------------------------------------------------------------
    bool accept(const std::vector<std::pair<std::string, std::string>>& extra, std::vector<Event>& ev) {
        if (phase_ != Phase::Decide || !req_.websocket) return false;
        std::string r = "HTTP/1.1 101 Switching Protocols\r\nUpgrade: websocket\r\nConnection: Upgrade\r\n";
        r += "Sec-WebSocket-Accept: " + accept_key(req_.header("sec-websocket-key")) + "\r\n";
        for (const auto& kv : extra) r += kv.first + ": " + kv.second + "\r\n";
        r += "\r\n";
        queue_raw(r);
        phase_ = Phase::Open;
        deadline_ms_ = 0;
        reported_ = true;
        Event e;
        e.type = Event::Type::Open;
        e.conn = id_;
        ev.push_back(std::move(e));
        // Frames that arrived right behind the request head.
        if (!leftover_.empty()) {
            Bytes b;
            b.swap(leftover_);
            feed(b.data(), b.size(), ev);
        }
        return true;
    }

    bool respond(int status, const std::string& type, const std::string& body,
                 const std::vector<std::pair<std::string, std::string>>& extra) {
        if (phase_ != Phase::Decide) return false;
        std::string r = "HTTP/1.1 " + std::to_string(status) + " " + status_text(status) + "\r\n";
        if (!type.empty()) r += "Content-Type: " + type + "\r\n";
        r += "Content-Length: " + std::to_string(body.size()) + "\r\n";
        r += "Cache-Control: no-store\r\nConnection: close\r\n";
        for (const auto& kv : extra) r += kv.first + ": " + kv.second + "\r\n";
        r += "\r\n";
        if (req_.method != "HEAD") r += body;
        queue_raw(r);
        close_code_ = 0;
        close_reason_ = "HTTP " + std::to_string(status);
        initiated_here_ = true;
        phase_ = Phase::Flush;
        deadline_ms_ = monotonic_ms() + (uint64_t)opt_.close_timeout_ms;
        return true;
    }

    // ---- sending ------------------------------------------------------------------------
    bool can_send() const { return phase_ == Phase::Open; }

    bool send_message(SharedBytes msg, bool binary) {
        if (!can_send() || !msg) return false;
        const size_t n = msg->size();
        if (out_bytes_ + n + 14 > opt_.max_queued_bytes) return false;
        const uint8_t op = binary ? 2 : 1;
        const size_t frag = opt_.send_fragment_bytes ? opt_.send_fragment_bytes : n;
        size_t off = 0;
        bool first = true;
        do {
            const size_t len = std::min(frag ? frag : n, n - off);
            const bool fin = off + len == n;
            queue_frame(first ? op : 0, fin, msg, off, len, /*data=*/true, /*priority=*/false);
            off += len;
            first = false;
        } while (off < n);
        ++out_msgs_;
        return true;
    }

    // One binary message: `prefix`, then body[off, off + len). On the server
    // side (unmasked, unfragmented) the prefix rides in the frame head and the
    // body is shared; otherwise the message is joined first.
    bool send_prefixed(Bytes prefix, SharedBytes body, size_t off, size_t len) {
        if (!can_send() || !body || off > body->size() || len > body->size() - off) return false;
        const size_t n = prefix.size() + len;
        if (out_bytes_ + n + 14 > opt_.max_queued_bytes) return false;
        if (role_ == Role::Client || opt_.send_fragment_bytes) {
            auto whole = std::make_shared<Bytes>(std::move(prefix));
            whole->insert(whole->end(), body->begin() + (std::ptrdiff_t)off,
                          body->begin() + (std::ptrdiff_t)(off + len));
            return send_message(std::move(whole), true);
        }
        Out o;
        o.head.push_back(uint8_t(0x80 | 0x2));
        if (n < 126) {
            o.head.push_back(uint8_t(n));
        } else if (n <= 0xffff) {
            o.head.push_back(126);
            o.head.push_back(uint8_t(n >> 8));
            o.head.push_back(uint8_t(n));
        } else {
            o.head.push_back(127);
            for (int i = 7; i >= 0; --i) o.head.push_back(uint8_t((uint64_t)n >> (8 * i)));
        }
        o.head.insert(o.head.end(), prefix.begin(), prefix.end());
        o.body = std::move(body);
        o.begin = off;
        o.end = off + len;
        o.data = true;
        o.ends_message = true;
        out_bytes_ += o.head.size() + len;
        out_.push_back(std::move(o));
        ++out_msgs_;
        return true;
    }

    bool send_control(uint8_t op, const std::string& payload, bool priority) {
        if (phase_ != Phase::Open && !(op == 0x8 && phase_ == Phase::Closing)) return false;
        auto b = std::make_shared<Bytes>(payload.begin(), payload.end());
        queue_frame(op, true, b, 0, b->size(), false, priority);
        return true;
    }

    // Starts the close handshake (an open connection) or ends the exchange.
    void close(uint16_t code, const std::string& reason, bool priority, std::vector<Event>& ev) {
        switch (phase_) {
            case Phase::Open: {
                std::string p;
                p.push_back(char(code >> 8));
                p.push_back(char(code & 0xff));
                p += reason.substr(0, 123);
                if (priority) drop_unsent_data();
                send_control(0x8, p, priority);
                close_sent_ = true;
                close_code_ = code;
                close_reason_ = reason;
                initiated_here_ = true;
                phase_ = Phase::Closing;
                deadline_ms_ = monotonic_ms() + (uint64_t)opt_.close_timeout_ms;
                break;
            }
            case Phase::Closing:
            case Phase::Flush:
            case Phase::Dead:
                break;
            default:   // not upgraded: just end it
                initiated_here_ = true;
                die(ev, code, reason);
                break;
        }
    }

    // Writes what the socket takes now. Reports Writable when the queue drains,
    // and ends the connection when a Flush finished.
    void flush(std::vector<Event>& ev) {
        if (phase_ == Phase::Dead || phase_ == Phase::Connecting) return;
        const bool had = !out_.empty();
        while (!out_.empty()) {
            Out& o = out_.front();
            const size_t total = o.head.size() + (o.end - o.begin);
            while (o.sent < total) {
                const uint8_t* p;
                size_t n;
                if (o.sent < o.head.size()) {
                    p = o.head.data() + o.sent;
                    n = o.head.size() - o.sent;
                } else {
                    const size_t b = o.begin + (o.sent - o.head.size());
                    p = o.body->data() + b;
                    n = o.end - b;
                }
                const long long w = sock_send(fd_, p, n);
                if (w < 0) {
                    const int e = cnet_errno();
                    if (CNET_WOULDBLOCK(e) || CNET_INTR(e)) return;
                    die(ev, close_code_ ? close_code_ : kCloseAbnormal, "send: " + sock_error_text(e));
                    return;
                }
                o.sent += (size_t)w;
                out_bytes_ -= (size_t)w;
            }
            if (o.ends_message) --out_msgs_;
            out_.pop_front();
        }
        if (had && phase_ == Phase::Open) {
            Event e;
            e.type = Event::Type::Writable;
            e.conn = id_;
            ev.push_back(std::move(e));
        }
        if (phase_ == Phase::Flush) {
            ::shutdown(fd_, CNET_SHUT_WR);
            die(ev, close_code_, close_reason_);
        }
    }

    // Ends the connection now and reports Closed (once, and only for a
    // connection the owner has heard of).
    void die(std::vector<Event>& ev, uint16_t code, const std::string& reason) {
        if (phase_ == Phase::Dead) return;
        phase_ = Phase::Dead;
        drop_socket();
        if (!reported_) return;
        Event e;
        e.type = Event::Type::Closed;
        e.conn = id_;
        e.code = code;
        e.reason = reason;
        e.http_status = http_status_;
        e.initiated_here = initiated_here_;
        ev.push_back(std::move(e));
    }

    void mark_reported() { reported_ = true; }

private:
    struct Out {
        Bytes head;            // the frame header (or raw bytes when body is null)
        SharedBytes body;      // shared payload (server) or a masked copy (client)
        size_t begin = 0, end = 0;
        size_t sent = 0;       // over head + [begin, end)
        bool data = false;     // a data frame (dropped by a failing close)
        bool ends_message = false;
    };

    void drop_socket() {
        if (fd_ != CNET_BAD_SOCKET) {
            cnet_closesocket(fd_);
            fd_ = CNET_BAD_SOCKET;
        }
    }

    void queue_raw(const std::string& s) {
        Out o;
        o.head.assign(s.begin(), s.end());
        out_bytes_ += o.head.size();
        out_.push_back(std::move(o));
    }

    void queue_frame(uint8_t op, bool fin, const SharedBytes& body, size_t off, size_t len, bool data,
                     bool priority) {
        Out o;
        o.head.push_back(uint8_t((fin ? 0x80 : 0) | op));
        const uint8_t mbit = role_ == Role::Client ? 0x80 : 0;
        if (len < 126) {
            o.head.push_back(uint8_t(mbit | len));
        } else if (len <= 0xffff) {
            o.head.push_back(uint8_t(mbit | 126));
            o.head.push_back(uint8_t(len >> 8));
            o.head.push_back(uint8_t(len));
        } else {
            o.head.push_back(uint8_t(mbit | 127));
            for (int i = 7; i >= 0; --i) o.head.push_back(uint8_t((uint64_t)len >> (8 * i)));
        }
        if (role_ == Role::Client) {
            uint8_t key[4];
            const uint64_t r = rng()();
            std::memcpy(key, &r, 4);
            o.head.insert(o.head.end(), key, key + 4);
            auto masked = std::make_shared<Bytes>(body->begin() + (std::ptrdiff_t)off,
                                                  body->begin() + (std::ptrdiff_t)(off + len));
            apply_mask(masked->data(), masked->size(), key, 0);
            o.body = std::move(masked);
            o.begin = 0;
            o.end = len;
        } else {
            o.body = body;
            o.begin = off;
            o.end = off + len;
        }
        o.data = data;
        o.ends_message = data && fin;
        out_bytes_ += o.head.size() + len;
        if (priority) {
            // After a partly written frame (frames never interleave), before the rest.
            auto at = out_.begin();
            if (at != out_.end() && at->sent > 0) ++at;
            out_.insert(at, std::move(o));
        } else {
            out_.push_back(std::move(o));
        }
    }

    // A failing close: unsent data frames go (nothing may follow a close).
    void drop_unsent_data() {
        for (auto it = out_.begin(); it != out_.end();) {
            if (it->data && it->sent == 0) {
                out_bytes_ -= it->head.size() + (it->end - it->begin);
                if (it->ends_message) --out_msgs_;
                it = out_.erase(it);
            } else {
                ++it;
            }
        }
    }

    // ---- reading ------------------------------------------------------------------------
    void read_some(std::vector<Event>& ev) {
        uint8_t buf[kReadChunk];
        size_t budget = kReadBudget;
        while (budget > 0 && phase_ != Phase::Dead && phase_ != Phase::Decide && phase_ != Phase::Flush) {
            const long long r = sock_recv(fd_, buf, sizeof buf);
            if (r == 0) {
                // The peer ended the TCP connection.
                if (close_received_ || close_sent_)
                    die(ev, close_code_ ? close_code_ : kCloseNoStatus, close_reason_);
                else
                    die(ev, kCloseAbnormal, "the peer closed the connection without a close frame");
                return;
            }
            if (r < 0) {
                const int e = cnet_errno();
                if (CNET_WOULDBLOCK(e) || CNET_INTR(e)) return;
                die(ev, kCloseAbnormal, "recv: " + sock_error_text(e));
                return;
            }
            last_input_ms_ = monotonic_ms();
            budget -= std::min(budget, (size_t)r);
            switch (phase_) {
                case Phase::ReadHead:
                case Phase::AwaitUpgrade:
                    read_head(buf, (size_t)r, ev);
                    break;
                case Phase::Open:
                case Phase::Closing:
                    feed(buf, (size_t)r, ev);
                    break;
                default:
                    break;
            }
        }
    }

    void read_head(const uint8_t* p, size_t n, std::vector<Event>& ev) {
        const size_t before = head_.size();
        head_.append((const char*)p, n);
        const size_t from = before >= 3 ? before - 3 : 0;
        const size_t end = head_.find("\r\n\r\n", from);
        if (end == std::string::npos) {
            if (head_.size() > opt_.max_header_bytes) {
                if (role_ == Role::Server) server_bad_head(431, ev);
                else die(ev, kCloseAbnormal, "upgrade response head too large");
            }
            return;
        }
        if (end + 4 > opt_.max_header_bytes) {
            if (role_ == Role::Server) server_bad_head(431, ev);
            else die(ev, kCloseAbnormal, "upgrade response head too large");
            return;
        }
        const std::string head = head_.substr(0, end);
        leftover_.assign(head_.begin() + (std::ptrdiff_t)(end + 4), head_.end());
        head_.clear();
        head_.shrink_to_fit();
        if (role_ == Role::Server) server_head(head, ev);
        else client_head(head, ev);
    }

    // A request head the server cannot use: answer, then close quietly.
    void server_bad_head(int status, std::vector<Event>& ev) {
        phase_ = Phase::Decide;
        req_.method = "GET";
        respond(status, "text/plain; charset=utf-8", std::string(status_text(status)) + "\n", {});
        flush(ev);
    }

    void server_head(const std::string& head, std::vector<Event>& ev) {
        std::string first;
        std::vector<std::pair<std::string, std::string>> f;
        if (!split_head(head, &first, &f)) {
            server_bad_head(400, ev);
            return;
        }
        // METHOD SP request-target SP HTTP-version
        const size_t a = first.find(' ');
        const size_t b = a == std::string::npos ? a : first.find(' ', a + 1);
        if (a == std::string::npos || b == std::string::npos || first.find(' ', b + 1) != std::string::npos) {
            server_bad_head(400, ev);
            return;
        }
        HttpRequest r;
        r.method = first.substr(0, a);
        r.target = first.substr(a + 1, b - a - 1);
        const std::string version = first.substr(b + 1);
        if (r.method.empty() || r.target.empty() || version.rfind("HTTP/1.", 0) != 0) {
            server_bad_head(400, ev);
            return;
        }
        const size_t q = r.target.find('?');
        r.path = r.target.substr(0, q);
        r.query = q == std::string::npos ? std::string() : r.target.substr(q + 1);
        r.host = field(f, "host");
        r.has_origin = has_field(f, "origin");
        r.origin = field(f, "origin");
        std::string key_raw;
        r.websocket = r.method == "GET" && version == "HTTP/1.1" &&
                      has_token(field(f, "upgrade"), "websocket") &&
                      has_token(field(f, "connection"), "upgrade") &&
                      field(f, "sec-websocket-version") == "13" &&
                      base64_decode(field(f, "sec-websocket-key"), &key_raw) && key_raw.size() == 16;
        r.headers = std::move(f);
        req_ = r;
        phase_ = Phase::Decide;
        deadline_ms_ = 0;
        reported_ = true;
        Event e;
        e.type = Event::Type::Request;
        e.conn = id_;
        e.request = req_;
        ev.push_back(std::move(e));
    }

    void client_head(const std::string& head, std::vector<Event>& ev) {
        std::string first;
        std::vector<std::pair<std::string, std::string>> f;
        reported_ = true;   // the owner started this connection: it hears how it ends
        if (!split_head(head, &first, &f) || first.rfind("HTTP/1.1 ", 0) != 0 || first.size() < 12) {
            die(ev, kCloseAbnormal, "malformed upgrade response");
            return;
        }
        const int status = std::atoi(first.c_str() + 9);
        if (status != 101) {
            http_status_ = status;
            die(ev, kCloseAbnormal, "the server refused the upgrade: " + first.substr(9));
            return;
        }
        if (!has_token(field(f, "upgrade"), "websocket") || !has_token(field(f, "connection"), "upgrade") ||
            field(f, "sec-websocket-accept") != accept_key(key_)) {
            die(ev, kCloseAbnormal, "bad upgrade response (Upgrade, Connection or Sec-WebSocket-Accept)");
            return;
        }
        if (has_field(f, "sec-websocket-extensions") || has_field(f, "sec-websocket-protocol")) {
            die(ev, kCloseAbnormal, "the server chose an extension or a subprotocol nobody offered");
            return;
        }
        phase_ = Phase::Open;
        deadline_ms_ = 0;
        Event e;
        e.type = Event::Type::Open;
        e.conn = id_;
        ev.push_back(std::move(e));
        if (!leftover_.empty()) {
            Bytes b;
            b.swap(leftover_);
            feed(b.data(), b.size(), ev);
        }
    }

    // A frame breaking RFC 6455: close with `code` at once and drop the rest.
    void fail(uint16_t code, const std::string& why, std::vector<Event>& ev) {
        if (phase_ == Phase::Open) {
            close(code, why, /*priority=*/true, ev);
        }
        close_code_ = code;
        close_reason_ = why;
        initiated_here_ = true;
        phase_ = Phase::Flush;   // write the close, then end the TCP connection
        deadline_ms_ = monotonic_ms() + (uint64_t)opt_.close_timeout_ms;
    }

    // The frame parser: header bytes, then the payload streamed into the
    // message (or control) buffer, unmasked on the way.
    void feed(const uint8_t* p, size_t n, std::vector<Event>& ev) {
        while (n > 0 && (phase_ == Phase::Open || phase_ == Phase::Closing)) {
            if (!in_payload_) {
                const size_t want = hdr_need_ - hdr_have_;
                const size_t take = std::min(want, n);
                std::memcpy(hdr_ + hdr_have_, p, take);
                hdr_have_ += take;
                p += take;
                n -= take;
                if (hdr_have_ < hdr_need_) continue;
                if (!parse_header(ev)) return;
                continue;
            }
            const size_t take = (size_t)std::min<uint64_t>(payload_left_, n);
            Bytes& dst = cur_op_ >= 0x8 ? ctrl_ : msg_;
            const size_t at = dst.size();
            dst.insert(dst.end(), p, p + take);
            if (masked_) apply_mask(dst.data() + at, take, mask_, mask_pos_);
            mask_pos_ += take;
            payload_left_ -= take;
            p += take;
            n -= take;
            if (payload_left_ == 0) end_frame(ev);
        }
    }

    // Parses the header bytes gathered so far. Asks for more (hdr_need_) while
    // the extended length or the key are missing. False when the connection failed.
    bool parse_header(std::vector<Event>& ev) {
        const uint8_t b0 = hdr_[0], b1 = hdr_[1];
        const uint8_t len7 = b1 & 0x7f;
        const bool masked = (b1 & 0x80) != 0;
        const size_t need = 2 + (len7 == 126 ? 2 : len7 == 127 ? 8 : 0) + (masked ? 4 : 0);
        if (hdr_need_ < need) {
            hdr_need_ = need;
            return true;
        }
        const bool fin = (b0 & 0x80) != 0;
        const uint8_t op = b0 & 0x0f;
        uint64_t len = len7;
        size_t pos = 2;
        if (len7 == 126) {
            len = (uint64_t(hdr_[2]) << 8) | hdr_[3];
            pos = 4;
        } else if (len7 == 127) {
            len = 0;
            for (int i = 0; i < 8; ++i) len = (len << 8) | hdr_[2 + i];
            pos = 10;
        }
        hdr_have_ = 0;
        hdr_need_ = 2;
        if (b0 & 0x70) { fail(kCloseProtocolError, "reserved bits set (no extension was agreed)", ev); return false; }
        const bool control = op >= 0x8;
        if (!(op == 0 || op == 1 || op == 2 || op == 8 || op == 9 || op == 10)) {
            fail(kCloseProtocolError, "unknown opcode " + std::to_string(op), ev);
            return false;
        }
        if (masked != (role_ == Role::Server)) {
            fail(kCloseProtocolError, role_ == Role::Server ? "a client frame without a mask"
                                                            : "a masked server frame", ev);
            return false;
        }
        if (len7 == 127 && (len >> 63)) { fail(kCloseProtocolError, "bad payload length", ev); return false; }
        if (control) {
            if (!fin || len > 125) {
                fail(kCloseProtocolError, "a fragmented or long control frame", ev);
                return false;
            }
        } else if (op == 0) {
            if (msg_op_ == 0) { fail(kCloseProtocolError, "a continuation without a message", ev); return false; }
        } else {
            if (msg_op_ != 0) { fail(kCloseProtocolError, "a new message inside a fragmented one", ev); return false; }
        }
        if (!control) {
            if (len > opt_.max_message_bytes || msg_.size() + len > opt_.max_message_bytes) {
                fail(kCloseTooBig, "message larger than " + std::to_string(opt_.max_message_bytes) + " bytes", ev);
                return false;
            }
            if (op != 0) {
                msg_op_ = op;
                msg_.clear();
                if (fin) msg_.reserve((size_t)len);
            }
        } else {
            ctrl_.clear();
        }
        masked_ = masked;
        if (masked) std::memcpy(mask_, hdr_ + pos, 4);
        mask_pos_ = 0;
        cur_op_ = op;
        cur_fin_ = fin;
        payload_left_ = len;
        in_payload_ = true;
        if (len == 0) end_frame(ev);
        return phase_ != Phase::Dead;
    }

    void end_frame(std::vector<Event>& ev) {
        in_payload_ = false;
        const uint8_t op = cur_op_;
        if (op >= 0x8) {
            Bytes c;
            c.swap(ctrl_);
            on_control(op, c, ev);
            return;
        }
        if (!cur_fin_) return;   // more fragments follow
        const uint8_t mop = msg_op_;
        msg_op_ = 0;
        if (phase_ != Phase::Open) {   // data after our close: ignored
            msg_.clear();
            return;
        }
        Event e;
        e.type = Event::Type::Message;
        e.conn = id_;
        e.binary = mop == 2;
        e.data.swap(msg_);
        ev.push_back(std::move(e));
    }

    void on_control(uint8_t op, const Bytes& payload, std::vector<Event>& ev) {
        if (op == 0x9) {   // ping: pong with the same payload, ahead of queued data
            if (phase_ == Phase::Open) send_control(0xA, std::string(payload.begin(), payload.end()), true);
            return;
        }
        if (op == 0xA) {   // pong: reported, so a sender can measure the round trip
            Event e;
            e.type = Event::Type::Pong;
            e.conn = id_;
            e.data = payload;
            ev.push_back(std::move(e));
            return;
        }
        // close
        uint16_t code = kCloseNoStatus;
        std::string reason;
        if (payload.size() == 1) { fail(kCloseProtocolError, "a close frame of one byte", ev); return; }
        if (payload.size() >= 2) {
            code = uint16_t((payload[0] << 8) | payload[1]);
            reason.assign(payload.begin() + 2, payload.end());
            if (!valid_close_code(code)) { fail(kCloseProtocolError, "invalid close code " + std::to_string(code), ev); return; }
        }
        close_received_ = true;
        if (!close_sent_) {
            // The peer started: echo its code, then end the connection.
            std::string p;
            if (code != kCloseNoStatus) {
                p.push_back(char(code >> 8));
                p.push_back(char(code & 0xff));
            }
            drop_unsent_data();
            send_control(0x8, p, true);
            close_sent_ = true;
            close_code_ = code;
            close_reason_ = reason;
            initiated_here_ = false;
        }
        // Both close frames exchanged: write what is left and end TCP.
        phase_ = Phase::Flush;
        deadline_ms_ = monotonic_ms() + (uint64_t)opt_.close_timeout_ms;
        flush(ev);
    }

    ConnId id_;
    sock_t fd_;
    Role role_;
    const Options& opt_;
    Phase phase_ = Phase::ReadHead;
    uint64_t deadline_ms_ = 0;
    uint64_t last_input_ms_ = 0;
    bool reported_ = false;

    // handshake
    std::string head_;
    Bytes leftover_;
    HttpRequest req_;
    std::string key_;
    int http_status_ = 0;

    // frame parser
    uint8_t hdr_[14] = {};
    size_t hdr_have_ = 0, hdr_need_ = 2;
    bool in_payload_ = false;
    uint64_t payload_left_ = 0;
    bool masked_ = false;
    uint8_t mask_[4] = {};
    size_t mask_pos_ = 0;
    uint8_t cur_op_ = 0;
    bool cur_fin_ = false;
    uint8_t msg_op_ = 0;   // opcode of the message being reassembled, 0 = none
    Bytes msg_;
    Bytes ctrl_;

    // send queue
    std::deque<Out> out_;
    size_t out_bytes_ = 0;
    size_t out_msgs_ = 0;

    // close
    bool close_sent_ = false, close_received_ = false, initiated_here_ = false;
    uint16_t close_code_ = 0;
    std::string close_reason_;

    friend class Server;
    friend class Client;
};

namespace {

// One poll(2) round over a waker and the given connections (plus a listener).
// Returns the revents per entry in the same order.
std::vector<short> wait_round(Waker* waker, intptr_t listen_fd, bool listen_ready,
                              const std::vector<Conn*>& conns, int timeout_ms) {
    std::vector<pollfd_t> fds;
    fds.reserve(conns.size() + 2);
    auto add = [&](sock_t fd, short events) {
        pollfd_t p{};
        p.fd = fd;
        p.events = events;
        fds.push_back(p);
    };
    add(waker && waker->fd() != CNET_BAD_SOCKET ? waker->fd() : CNET_BAD_SOCKET, POLLIN);
    add(listen_ready ? (sock_t)listen_fd : CNET_BAD_SOCKET, POLLIN);
    for (Conn* c : conns) add(c->fd(), c->interest());
    // Entries with no socket are skipped by poll(2) when negative; WSAPoll
    // rejects them, so move them out of the array on every platform.
    std::vector<int> map;
    std::vector<pollfd_t> live;
    for (size_t i = 0; i < fds.size(); ++i) {
        if (fds[i].fd == CNET_BAD_SOCKET) continue;
        map.push_back((int)i);
        live.push_back(fds[i]);
    }
    std::vector<short> rev(fds.size(), 0);
    if (live.empty()) return rev;
    int r;
    do {
        r = cnet_poll(live.data(), live.size(), timeout_ms);
    } while (r < 0 && CNET_INTR(cnet_errno()));
    if (r <= 0) return rev;
    for (size_t i = 0; i < live.size(); ++i) rev[(size_t)map[i]] = live[i].revents;
    return rev;
}

int clamp_timeout(int timeout_ms, uint64_t now, uint64_t deadline) {
    if (deadline == 0) return timeout_ms;
    const int until = deadline <= now ? 0 : (int)std::min<uint64_t>(deadline - now, 1u << 30);
    return timeout_ms < 0 ? until : std::min(timeout_ms, until);
}

}  // namespace

// ---- Server -----------------------------------------------------------------------------

Server::Server(Options opt) : opt_(opt), waker_(std::make_unique<Waker>()) { net_init(); }

Server::~Server() {
    conns_.clear();
    if (listen_fd_ != -1) cnet_closesocket((sock_t)listen_fd_);
}

bool Server::listening() const { return listen_fd_ != -1; }

bool Server::listen(uint16_t port, std::string* err, const std::string& host) {
    if (listen_fd_ != -1) {
        *err = "already listening on port " + std::to_string(port_);
        return false;
    }
    in_addr ip{};
    if (::inet_pton(AF_INET, host.c_str(), &ip) != 1) {
        *err = "not a numeric IPv4 address: " + host;
        return false;
    }
    sock_t fd = ::socket(AF_INET, SOCK_STREAM, 0);
    if (fd == CNET_BAD_SOCKET) {
        *err = "socket: " + sock_error_text(cnet_errno());
        return false;
    }
#ifndef _WIN32
    int on = 1;   // a restarted server can take its port back at once
    ::setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, &on, sizeof on);
#endif
    sockaddr_in a{};
    a.sin_family = AF_INET;
    a.sin_addr = ip;
    a.sin_port = htons(port);
    if (::bind(fd, (sockaddr*)&a, sizeof a) != 0) {
        *err = "bind " + host + ":" + std::to_string(port) + ": " + sock_error_text(cnet_errno());
        cnet_closesocket(fd);
        return false;
    }
    if (::listen(fd, 128) != 0 || !set_nonblocking(fd)) {
        *err = "listen: " + sock_error_text(cnet_errno());
        cnet_closesocket(fd);
        return false;
    }
    socklen_t len = sizeof a;
    ::getsockname(fd, (sockaddr*)&a, &len);
    port_ = ntohs(a.sin_port);
    listen_fd_ = (intptr_t)fd;
    return true;
}

void Server::wake() { waker_->wake(); }

Conn* Server::find(ConnId c) const {
    auto it = conns_.find(c);
    return it == conns_.end() || it->second->dead() ? nullptr : it->second.get();
}

std::vector<Event> Server::poll(int timeout_ms) {
    std::vector<Event> ev;
    ev.swap(pending_);
    uint64_t now = monotonic_ms();
    // Write what was queued since the last poll before waiting.
    std::vector<Conn*> list;
    uint64_t next_deadline = 0;
    for (auto& kv : conns_) {
        Conn* c = kv.second.get();
        c->flush(ev);
        if (!c->dead()) {
            list.push_back(c);
            if (c->deadline() && (!next_deadline || c->deadline() < next_deadline)) next_deadline = c->deadline();
        }
    }
    const int wait = ev.empty() ? clamp_timeout(timeout_ms, now, next_deadline) : 0;
    const bool can_accept = listen_fd_ != -1;
    const std::vector<short> rev = wait_round(waker_.get(), listen_fd_, can_accept, list, wait);
    if (rev[0]) waker_->drain();
    if (can_accept && (rev[1] & POLLIN)) {
        for (;;) {
            sockaddr_in a{};
            socklen_t len = sizeof a;
            const sock_t fd = ::accept((sock_t)listen_fd_, (sockaddr*)&a, &len);
            if (fd == CNET_BAD_SOCKET) break;
            if (conns_.size() >= opt_.max_connections || !set_nonblocking(fd)) {
                cnet_closesocket(fd);
                continue;
            }
            tune_socket(fd);
            auto c = std::make_unique<Conn>(next_id_++, fd, Conn::Role::Server, opt_);
            c->set_phase(Conn::Phase::ReadHead);
            c->set_deadline(monotonic_ms() + (uint64_t)opt_.header_timeout_ms);
            conns_.emplace(c->id(), std::move(c));
        }
    }
    for (size_t i = 0; i < list.size(); ++i)
        if (rev[i + 2]) list[i]->on_poll(rev[i + 2], ev);
    now = monotonic_ms();
    for (auto& kv : conns_) kv.second->on_timer(now, ev);
    for (auto it = conns_.begin(); it != conns_.end();) {
        if (it->second->dead()) it = conns_.erase(it);
        else ++it;
    }
    return ev;
}

bool Server::accept(ConnId c, const std::vector<std::pair<std::string, std::string>>& extra) {
    Conn* k = find(c);
    return k && k->accept(extra, pending_);   // Open is reported by the next poll
}

bool Server::respond(ConnId c, int status, const std::string& content_type, const std::string& body,
                     const std::vector<std::pair<std::string, std::string>>& extra) {
    Conn* k = find(c);
    return k && k->respond(status, content_type, body, extra);
}

bool Server::reject(ConnId c, int status) {
    return respond(c, status, "text/plain; charset=utf-8",
                   std::to_string(status) + " " + status_text(status) + "\n");
}

bool Server::send(ConnId c, SharedBytes msg, bool binary) {
    Conn* k = find(c);
    return k && k->send_message(std::move(msg), binary);
}

bool Server::send(ConnId c, const void* data, size_t n, bool binary) {
    const uint8_t* p = static_cast<const uint8_t*>(data);
    return send(c, std::make_shared<const Bytes>(p, p + n), binary);
}

bool Server::send(ConnId c, Bytes prefix, SharedBytes body, size_t offset, size_t length) {
    Conn* k = find(c);
    return k && k->send_prefixed(std::move(prefix), std::move(body), offset, length);
}

bool Server::ping(ConnId c, const std::string& payload) {
    Conn* k = find(c);
    return k && payload.size() <= 125 && k->send_control(0x9, payload, true);
}

void Server::close(ConnId c, uint16_t code, const std::string& reason) {
    if (Conn* k = find(c)) k->close(code, reason, /*priority=*/false, pending_);
}

size_t Server::queued_bytes(ConnId c) const {
    const Conn* k = find(c);
    return k ? k->queued_bytes() : 0;
}

size_t Server::queued_messages(ConnId c) const {
    const Conn* k = find(c);
    return k ? k->queued_messages() : 0;
}

uint64_t Server::last_input_ms(ConnId c) const {
    const Conn* k = find(c);
    return k ? k->last_input_ms() : 0;
}

bool Server::is_open(ConnId c) const {
    const Conn* k = find(c);
    return k && k->phase() == Conn::Phase::Open;
}

std::vector<ConnId> Server::connection_ids() const {
    std::vector<ConnId> out;
    for (const auto& kv : conns_)
        if (!kv.second->dead() && kv.second->reported()) out.push_back(kv.first);
    return out;
}

const HttpRequest* Server::request(ConnId c) const {
    const Conn* k = find(c);
    return k && k->reported() ? &k->request() : nullptr;
}

std::vector<Event> Server::shutdown(uint16_t code, int timeout_ms) {
    if (listen_fd_ != -1) {
        cnet_closesocket((sock_t)listen_fd_);
        listen_fd_ = -1;
    }
    for (auto& kv : conns_) {
        Conn* k = kv.second.get();
        if (k->phase() == Conn::Phase::Open) k->close(code, "server shutting down", false, pending_);
        else if (k->phase() != Conn::Phase::Closing && k->phase() != Conn::Phase::Flush)
            k->close(code, "server shutting down", false, pending_);
    }
    std::vector<Event> out;
    const uint64_t end = monotonic_ms() + (uint64_t)std::max(0, timeout_ms);
    while (!conns_.empty()) {
        const uint64_t now = monotonic_ms();
        if (now >= end) break;
        std::vector<Event> ev = poll((int)std::min<uint64_t>(end - now, 50));
        out.insert(out.end(), std::make_move_iterator(ev.begin()), std::make_move_iterator(ev.end()));
    }
    for (auto& kv : conns_) kv.second->die(out, code, "server shut down");
    conns_.clear();
    out.insert(out.end(), std::make_move_iterator(pending_.begin()), std::make_move_iterator(pending_.end()));
    pending_.clear();
    return out;
}

// ---- Client -----------------------------------------------------------------------------

Client::Client(Options opt) : opt_(opt), waker_(std::make_unique<Waker>()) { net_init(); }

Client::~Client() = default;

bool Client::connect(const std::string& url, std::string* err, const std::string& origin,
                     const std::vector<std::pair<std::string, std::string>>& extra_headers,
                     int connect_timeout_ms) {
    if (conn_ && !conn_->dead()) {
        *err = "already connected";
        return false;
    }
    conn_.reset();
    WsUrl u;
    if (!parse_ws_url(url, &u, err)) return false;
    sockaddr_in a{};
    a.sin_family = AF_INET;
    a.sin_port = htons(u.port);
    if (!resolve_ipv4(u.host, &a.sin_addr)) {
        *err = "the host must be a numeric IPv4 address or localhost: " + u.host;
        return false;
    }
    const sock_t fd = ::socket(AF_INET, SOCK_STREAM, 0);
    if (fd == CNET_BAD_SOCKET) {
        *err = "socket: " + sock_error_text(cnet_errno());
        return false;
    }
    if (!set_nonblocking(fd)) {
        *err = "non-blocking socket: " + sock_error_text(cnet_errno());
        cnet_closesocket(fd);
        return false;
    }
    tune_socket(fd);
    conn_ = std::make_unique<Conn>(1, fd, Conn::Role::Client, opt_);
    conn_->mark_reported();
    const int r = ::connect(fd, (sockaddr*)&a, sizeof a);
    if (r == 0) {
        conn_->set_phase(Conn::Phase::AwaitUpgrade);
        conn_->set_deadline(monotonic_ms() + (uint64_t)opt_.header_timeout_ms);
    } else {
        const int e = cnet_errno();
        if (!CNET_INPROGRESS(e)) {
            *err = "connect " + u.host + ":" + std::to_string(u.port) + ": " + sock_error_text(e);
            conn_.reset();
            return false;
        }
        conn_->set_phase(Conn::Phase::Connecting);
        conn_->set_deadline(monotonic_ms() + (uint64_t)std::max(1, connect_timeout_ms));
    }
    conn_->client_start(u, origin, extra_headers);
    return true;
}

Client::State Client::state() const {
    if (!conn_) return State::Idle;
    switch (conn_->phase()) {
        case Conn::Phase::Connecting: return State::Connecting;
        case Conn::Phase::AwaitUpgrade: return State::Handshake;
        case Conn::Phase::Open: return State::Open;
        case Conn::Phase::Closing:
        case Conn::Phase::Flush: return State::Closing;
        default: return State::Closed;
    }
}

std::vector<Event> Client::poll(int timeout_ms) {
    std::vector<Event> ev;
    ev.swap(pending_);
    if (!conn_ || conn_->dead()) {
        if (ev.empty() && timeout_ms != 0) {
            // Nothing to watch but the waker.
            wait_round(waker_.get(), -1, false, {}, timeout_ms);
            waker_->drain();
        }
        return ev;
    }
    uint64_t now = monotonic_ms();
    conn_->flush(ev);
    std::vector<Conn*> list;
    if (!conn_->dead()) list.push_back(conn_.get());
    const int wait = ev.empty() ? clamp_timeout(timeout_ms, now, conn_->deadline()) : 0;
    const std::vector<short> rev = wait_round(waker_.get(), -1, false, list, wait);
    if (rev[0]) waker_->drain();
    if (!list.empty() && rev[2]) conn_->on_poll(rev[2], ev);
    now = monotonic_ms();
    conn_->on_timer(now, ev);
    return ev;
}

void Client::wake() { waker_->wake(); }

bool Client::send(const void* data, size_t n, bool binary) {
    const uint8_t* p = static_cast<const uint8_t*>(data);
    return send(std::make_shared<const Bytes>(p, p + n), binary);
}

bool Client::send(SharedBytes msg, bool binary) {
    return conn_ && conn_->send_message(std::move(msg), binary);
}

bool Client::ping(const std::string& payload) {
    return conn_ && payload.size() <= 125 && conn_->send_control(0x9, payload, true);
}

void Client::close(uint16_t code, const std::string& reason) {
    if (conn_) conn_->close(code, reason, /*priority=*/false, pending_);
}

size_t Client::queued_bytes() const { return conn_ ? conn_->queued_bytes() : 0; }

uint64_t Client::last_input_ms() const { return conn_ ? conn_->last_input_ms() : 0; }

}  // namespace caliper_net
