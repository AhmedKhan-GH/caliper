#pragma once
// A small HTTP server for the tests of the mcp module, on 127.0.0.1 and a
// free port. Each connection runs on a thread of its own and is handed to the
// test's handler with its parsed request; the handler writes whatever bytes
// it likes, well-formed or not, and may wait before or between them, so the
// tests can play a server that misbehaves. The connection closes when the
// handler returns. POSIX only.
#include <atomic>
#include <functional>
#include <list>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace caliper_builder::mcp::test {

struct FakeRequest {
    std::string method;
    std::string target;
    std::vector<std::pair<std::string, std::string>> headers;   // names in lower case
    std::string body;
    std::string head;   // the raw head, without the blank line

    std::string header(const std::string& lower_name) const;
};

class FakeConnection {
public:
    explicit FakeConnection(int fd, const std::atomic<bool>* stopping) : fd_(fd), stopping_(stopping) {}
    // Writes all of `bytes`; false when the peer has gone.
    bool write(const std::string& bytes);
    // Waits, returning early when the server stops.
    void sleep_ms(int ms);
    // True when the client has closed its end (checked without blocking).
    bool peer_closed();

private:
    int fd_;
    const std::atomic<bool>* stopping_;
};

// A whole response with a Content-Length and Connection: close.
std::string http_response(int status, const std::string& content_type, const std::string& body,
                          const std::vector<std::pair<std::string, std::string>>& extra = {});

class FakeHttpServer {
public:
    using Handler = std::function<void(const FakeRequest&, FakeConnection&)>;

    FakeHttpServer() = default;
    ~FakeHttpServer();
    FakeHttpServer(const FakeHttpServer&) = delete;
    FakeHttpServer& operator=(const FakeHttpServer&) = delete;

    // Listens on 127.0.0.1 and a free port. False when that fails.
    bool start(Handler handler);
    void stop();
    int port() const { return port_; }

    // Every request read so far.
    std::vector<FakeRequest> requests() const;
    std::size_t request_count() const;
    void clear_requests();

private:
    void accept_loop();
    void serve(int fd);

    Handler handler_;
    int listen_fd_ = -1;
    int port_ = 0;
    std::atomic<bool> stopping_{false};
    std::thread acceptor_;
    mutable std::mutex mu_;
    std::list<std::thread> connections_;
    std::vector<FakeRequest> requests_;
};

}  // namespace caliper_builder::mcp::test
