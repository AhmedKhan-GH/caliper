#pragma once
// caliper_builder::mcp::loopback_http_request(): a small, strict HTTP/1.1
// client for Caliper API on the loopback address, as caliper-mcp uses it.
//
// - It connects only to 127.0.0.1. Any other host, including "localhost" and
//   other loopback addresses, is refused before a socket is made.
// - One request per connection: it sends "Connection: close" and reads the
//   response until the server closes. It speaks no TLS, follows no redirect (a
//   3xx answer is the error http.redirect) and refuses a chunked or otherwise
//   transfer-encoded response.
// - Three time limits: the connect, the wait for each read (the server may be
//   silent while a command waits for approval) and the whole exchange. A
//   cancel flag, when given, is checked at least every 100 ms.
// - The response head is at most 16 KiB with at most 100 header fields, and the
//   body at most `max_body_bytes`. A Content-Length must be one decimal number
//   and must match the body exactly; without one the body runs to the close.
// - The request is checked before it is sent: the method is a token, the
//   target starts with "/" and holds only visible ASCII, and no header name or
//   value holds a control character, so nothing can be smuggled into the head.
//
// Header values are never logged or put into an Error, so a bearer token in a
// request header stays out of every message this client makes. On Windows the
// client is a stub that answers http.unsupported.
//
// Error codes: http.refused_host, http.bad_request, http.connect_refused,
// http.connect_failed, http.connect_timeout, http.read_timeout, http.timeout,
// http.cancelled, http.io_error, http.too_large, http.bad_response,
// http.redirect, http.unsupported.
#include <caliper_builder/error.h>

#include <atomic>
#include <cstddef>
#include <string>
#include <utility>
#include <vector>

namespace caliper_builder::mcp {

struct HttpHeader {
    std::string name;
    std::string value;
};

struct HttpRequest {
    std::string method = "GET";
    std::string target = "/";          // the path and query, starting with "/"
    std::vector<HttpHeader> headers;   // Host, Connection and Content-Length are added
    std::string body;                  // sent with a Content-Length; POST always gets one
};

struct HttpLimits {
    int connect_timeout_ms = 2000;
    int read_timeout_ms = 15000;     // the longest silence while waiting for bytes
    int total_timeout_ms = 15000;    // the whole exchange, connect included
    std::size_t max_head_bytes = 16u << 10;
    std::size_t max_header_count = 100;
    std::size_t max_body_bytes = 64u << 20;
};

struct HttpResponse {
    Error error;   // set when no well-formed response arrived (status may still be set for http.redirect)
    int status = 0;
    std::string reason;
    std::vector<HttpHeader> headers;   // names in lower case, in the order received
    std::string body;

    bool ok() const { return error.ok(); }
    // The value of the first header with this lower-case name, or an empty string.
    std::string header(const std::string& lower_name) const;
};

// Sends one request to `host`:`port` and reads its response. `host` must be
// "127.0.0.1"; `port` 1 to 65535.
HttpResponse loopback_http_request(const std::string& host, int port, const HttpRequest& request,
                                   const HttpLimits& limits, const std::atomic<bool>* cancel = nullptr);

}  // namespace caliper_builder::mcp
