#pragma once
// ws.h: a small WebSocket layer (RFC 6455) for wire protocol v3 (caliper/
// live_net.h): the client side (NetSource, tests) and the server side (the
// `caliper serve` hub, tests) over plain TCP. Host-internal C++17.
//
// SCOPE. Plain ws:// over TCP (no TLS), no extensions (permessage-deflate is
// never offered or accepted), no subprotocols. The HTTP/1.1 opening handshake
// in both directions; on the server side a plain HTTP GET can be answered with
// a body (the hub serves the viewer page on the same port). Masking (client
// frames are masked, server frames are not; anything else closes with 1002),
// fragmentation on receive (continuation frames are reassembled, control
// frames may interleave) and optionally on send, ping and pong (a ping is
// answered automatically), close with a code and a reason, and size limits.
//
// WHY IN-TREE. libcurl (cmake/Dependencies.cmake) has a WebSocket client API
// but no server, and the hub needs both. One small layer for both sides keeps
// the framing rules, the limits and the close codes in one place, and it has
// no dependency to configure on Windows.
//
// THE MODEL. Event-driven and single-threaded: a Server or a Client is owned
// by one thread, which calls poll(timeout_ms). poll() waits (with poll(2), or
// WSAPoll on Windows) for socket activity or a wake(), then accepts, reads,
// writes and returns what happened as a list of Events. send(), accept(),
// respond(), reject() and close() never block: they queue bytes, and the next
// poll() writes them. Every socket is non-blocking. This layer creates no
// thread. wake() is the one function that may be called from another thread:
// it makes a waiting poll() return at once.
//
// BACK-PRESSURE. Every connection has a send queue. queued_bytes() tells a
// sender how much is still unwritten, and an Event::Writable is reported once
// each time a non-empty queue drains, so a sender can keep "the latest
// snapshot" itself and hand it over only when the connection has room (the
// protocol's flow control: a slow viewer never blocks a producer). A send that
// would take the queue past Options::max_queued_bytes is refused (false) and
// queues nothing. Queued payloads are shared_ptr buffers, so one encoded
// snapshot sent to many server connections is stored once. A peer's ping is
// answered with one pong; while it waits unsent, a newer ping's pong replaces
// it, and a pong that would take the queue past max_queued_bytes closes the
// connection with 1008 instead, so a peer that pings and never reads cannot
// grow the queue.
//
// LIMITS. A message (all fragments together) larger than max_message_bytes
// closes the connection with 1009; the default is the protocol's
// CALIPER_NET_MAX_FRAME_BYTES. An HTTP request or response head larger than
// max_header_bytes, or one not complete within header_timeout_ms, closes the
// TCP connection (a request gets 400 or 431 first when possible).
//
// LOOPBACK. The server listens on 127.0.0.1 unless told otherwise, and the
// client connects only to numeric IPv4 hosts and "localhost" (127.0.0.1): there
// is no name resolution. Applying the protocol's 403 rules (token, Host,
// Origin) is the caller's job; Event::Request carries what it needs.
//
// PORTABILITY. BSD sockets; the few Winsock differences (closesocket,
// WSAGetLastError, WSAPoll, ioctlsocket, WSAStartup) sit behind macros in
// ws.cpp. The Windows branch has been checked against the documented Winsock
// declarations but not yet compiled with MSVC; WSAPoll gets no entry for a
// connection that asks for no event.
#include <cstddef>
#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace caliper_net {

using ConnId = uint64_t;   // 0 is never a connection
using Bytes = std::vector<uint8_t>;
using SharedBytes = std::shared_ptr<const Bytes>;

// WebSocket close codes this layer sends or reports on its own (RFC 6455
// section 7.4.1). 1005 and 1006 are only ever reported, never sent.
enum : uint16_t {
    kCloseNormal = 1000,
    kCloseGoingAway = 1001,
    kCloseProtocolError = 1002,     // a frame breaking RFC 6455
    kCloseUnsupportedData = 1003,
    kCloseNoStatus = 1005,          // reported: the peer's close carried no code
    kCloseAbnormal = 1006,          // reported: the TCP connection ended without a close
    kClosePolicy = 1008,            // a peer's ping when the send queue is full
    kCloseTooBig = 1009,            // a message over max_message_bytes
};

struct Options {
    size_t max_message_bytes = size_t(64) << 20;   // CALIPER_NET_MAX_FRAME_BYTES
    size_t max_header_bytes = 16 * 1024;           // the HTTP head of the handshake
    size_t max_queued_bytes = size_t(512) << 20;   // per connection; send() refuses beyond
    size_t send_fragment_bytes = 0;    // >0: split outgoing data messages into
                                       // fragments of this size (tests, interop)
    int header_timeout_ms = 10000;     // the HTTP head must complete in this time
    int close_timeout_ms = 2000;       // wait this long for the peer's close frame
    size_t max_connections = 1024;     // server: TCP connections beyond are closed at once
};

// What the server parsed from a request head (Event::Request).
struct HttpRequest {
    std::string method;    // "GET"
    std::string target;    // as sent: "/v3/stream?token=..."
    std::string path;      // the target before '?': "/v3/stream"
    std::string query;     // after '?', without it ("" when none)
    std::string host;      // the Host header ("" when absent)
    std::string origin;    // the Origin header ("" when absent)
    bool has_origin = false;
    bool websocket = false;   // a valid RFC 6455 upgrade request (GET, HTTP/1.1,
                              // Upgrade: websocket, Connection: upgrade, version 13,
                              // a 16-byte Sec-WebSocket-Key)
    std::vector<std::pair<std::string, std::string>> headers;   // names lower-cased

    // The first header of that (lower-case) name, "" when absent.
    std::string header(const std::string& lower_name) const;
    // The first query parameter of that name, percent-decoded ('+' is a space);
    // false when absent.
    bool query_param(const std::string& name, std::string* value) const;
};

// Percent-decodes a query component ('+' becomes a space). False on a bad escape.
bool url_decode(const std::string& in, std::string* out);
// Percent-encodes everything but unreserved characters (RFC 3986).
std::string url_encode(const std::string& in);

// A parsed ws:// URL. host is as written ("127.0.0.1", "localhost"); port is
// 80 when the URL names none; target is the path plus "?query" ("/" when the
// URL has no path).
struct WsUrl {
    std::string host;
    uint16_t port = 80;
    std::string path;
    std::string query;
    std::string target;
};
bool parse_ws_url(const std::string& url, WsUrl* out, std::string* err);

struct Event {
    enum class Type {
        Request,    // server: a complete HTTP request head. Answer it with accept()
                    // (a WebSocket upgrade), respond() or reject(); until then the
                    // connection reads nothing more.
        Open,       // the upgrade completed: messages may flow (server: after
                    // accept(); client: after a valid 101 response)
        Message,    // one complete data message (fragments reassembled)
        Writable,   // the send queue drained after holding bytes
        Pong,       // a pong arrived (data: its payload); pings are answered
                    // automatically and not reported
        Closed,     // the connection is gone; its id is no longer valid
    };
    Type type = Type::Message;
    ConnId conn = 0;
    HttpRequest request;        // Request
    Bytes data;                 // Message, Pong
    bool binary = true;         // Message: false for a text message
    uint16_t code = 0;          // Closed: the close code (the peer's, or the one
                                // this side sent; 1005, 1006 as above)
    std::string reason;         // Closed: the close reason, or what went wrong locally
    int http_status = 0;        // Closed (client): the status of a refused upgrade
                                // (403, ...), 0 when the upgrade was not refused
    bool initiated_here = false;   // Closed: this side started the close
};

class Conn;   // ws.cpp
class Waker;  // ws.cpp

// The server side: listens, accepts many connections, answers HTTP requests,
// upgrades to WebSocket, and keeps a send queue per connection.
class Server {
public:
    explicit Server(Options opt = Options());
    ~Server();   // closes every connection at once (no close handshake) and the listener
    Server(const Server&) = delete;
    Server& operator=(const Server&) = delete;

    // Binds `host` (numeric IPv4; default loopback) on `port` (0: an ephemeral
    // port, see port()) and listens. False with *err.
    bool listen(uint16_t port, std::string* err, const std::string& host = "127.0.0.1");
    uint16_t port() const { return port_; }
    bool listening() const;

    // Waits up to timeout_ms (0: do not wait, <0: until something happens) for
    // activity or wake(), handles it, and returns the events, in order.
    std::vector<Event> poll(int timeout_ms);
    // Thread-safe: makes a waiting poll() return.
    void wake();

    // Answers a pending Request.
    // accept: sends 101 Switching Protocols (the request must be a valid upgrade;
    //   extra_headers are added to the response). Event::Open follows.
    bool accept(ConnId c, const std::vector<std::pair<std::string, std::string>>& extra_headers = {});
    // respond: a plain HTTP response with this body (no body for HEAD), then
    //   the connection closes once it is written.
    bool respond(ConnId c, int status, const std::string& content_type, const std::string& body,
                 const std::vector<std::pair<std::string, std::string>>& extra_headers = {});
    // reject: respond(status) with a short text body ("403 Forbidden").
    bool reject(ConnId c, int status);

    // Queues one data message (binary unless `binary` is false). False when the
    // connection is not open (or closing) or the queue limit would be passed.
    bool send(ConnId c, SharedBytes msg, bool binary = true);
    bool send(ConnId c, const void* data, size_t n, bool binary = true);
    // Queues one binary message made of `prefix` followed by body[offset,
    // offset + length), without copying the body: a forwarder sends a frame
    // with its own head in front of a payload shared by many connections.
    // With send_fragment_bytes set, the message is joined into one buffer and
    // sent as above.
    bool send(ConnId c, Bytes prefix, SharedBytes body, size_t offset, size_t length);
    bool ping(ConnId c, const std::string& payload = {});
    // Starts the close handshake with `code` (an upgraded connection), or ends
    // the HTTP exchange. Event::Closed follows once the peer answered, the
    // close timeout passed or the TCP connection ended.
    void close(ConnId c, uint16_t code = kCloseNormal, const std::string& reason = {});

    // Back-pressure: bytes still to be written (0 when the id is unknown).
    size_t queued_bytes(ConnId c) const;
    size_t queued_messages(ConnId c) const;
    // CLOCK_MONOTONIC-like ms of the last byte received (protocol idle timers).
    uint64_t last_input_ms(ConnId c) const;
    bool is_open(ConnId c) const;   // upgraded, not closing
    size_t connections() const { return conns_.size(); }
    std::vector<ConnId> connection_ids() const;
    const HttpRequest* request(ConnId c) const;   // the request head of a connection

    // Closes every upgraded connection with `code`, answers nothing more, stops
    // listening, and keeps polling until all connections are gone or
    // timeout_ms passed (then drops the rest). Returns the final events.
    std::vector<Event> shutdown(uint16_t code = kCloseGoingAway, int timeout_ms = 1000);

private:
    Conn* find(ConnId c) const;
    Options opt_;
    intptr_t listen_fd_ = -1;
    uint16_t port_ = 0;
    std::unique_ptr<Waker> waker_;
    std::map<ConnId, std::unique_ptr<Conn>> conns_;
    ConnId next_id_ = 1;
    std::vector<Event> pending_;   // events of calls between polls, reported by the next poll
};

// The client side: one connection to a ws:// URL.
class Client {
public:
    enum class State { Idle, Connecting, Handshake, Open, Closing, Closed };

    explicit Client(Options opt = Options());
    ~Client();   // drops the connection at once
    Client(const Client&) = delete;
    Client& operator=(const Client&) = delete;

    // Starts a non-blocking connect to `url` (ws://host:port/path?query). The
    // upgrade request carries `origin` when it is not empty, and extra_headers
    // (a "Host" entry replaces the default "<host>:<port>"). False with *err for
    // a bad URL, a host this layer does not resolve, or a socket error;
    // otherwise poll() reports Open or Closed.
    bool connect(const std::string& url, std::string* err, const std::string& origin = {},
                 const std::vector<std::pair<std::string, std::string>>& extra_headers = {},
                 int connect_timeout_ms = 5000);
    State state() const;

    std::vector<Event> poll(int timeout_ms);
    void wake();   // thread-safe

    bool send(const void* data, size_t n, bool binary = true);
    bool send(SharedBytes msg, bool binary = true);
    bool ping(const std::string& payload = {});
    void close(uint16_t code = kCloseNormal, const std::string& reason = {});
    size_t queued_bytes() const;
    uint64_t last_input_ms() const;

private:
    Options opt_;
    std::unique_ptr<Waker> waker_;
    std::unique_ptr<Conn> conn_;
    std::vector<Event> pending_;
};

// Milliseconds on a monotonic clock (the timers of this layer).
uint64_t monotonic_ms();

}  // namespace caliper_net
