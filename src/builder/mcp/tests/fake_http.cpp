// The fake HTTP server of the mcp tests (fake_http.h).
#include "fake_http.h"

#include <arpa/inet.h>
#include <cerrno>
#include <chrono>
#include <cstring>
#include <fcntl.h>
#include <netinet/in.h>
#include <poll.h>
#include <sys/socket.h>
#include <unistd.h>

namespace caliper_builder::mcp::test {

namespace {

std::string lower(std::string s) {
    for (char& c : s)
        if (c >= 'A' && c <= 'Z') c = char(c - 'A' + 'a');
    return s;
}

const char* status_text(int status) {
    switch (status) {
        case 200: return "OK";
        case 302: return "Found";
        case 400: return "Bad Request";
        case 401: return "Unauthorized";
        case 403: return "Forbidden";
        case 404: return "Not Found";
        case 409: return "Conflict";
        case 413: return "Payload Too Large";
        case 501: return "Not Implemented";
        case 503: return "Service Unavailable";
        case 504: return "Gateway Timeout";
        default: return "Status";
    }
}

}  // namespace

std::string FakeRequest::header(const std::string& lower_name) const {
    for (const auto& h : headers)
        if (h.first == lower_name) return h.second;
    return {};
}

bool FakeConnection::write(const std::string& bytes) {
    std::size_t done = 0;
    while (done < bytes.size()) {
#ifdef MSG_NOSIGNAL
        const ssize_t n = ::send(fd_, bytes.data() + done, bytes.size() - done, MSG_NOSIGNAL);
#else
        const ssize_t n = ::send(fd_, bytes.data() + done, bytes.size() - done, 0);
#endif
        if (n < 0) {
            if (errno == EINTR) continue;
            if (errno == EAGAIN || errno == EWOULDBLOCK) {
                pollfd p{fd_, POLLOUT, 0};
                (void)::poll(&p, 1, 50);
                if (stopping_->load()) return false;
                continue;
            }
            return false;
        }
        done += std::size_t(n);
    }
    return true;
}

void FakeConnection::sleep_ms(int ms) {
    const auto end = std::chrono::steady_clock::now() + std::chrono::milliseconds(ms);
    while (std::chrono::steady_clock::now() < end && !stopping_->load())
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
}

bool FakeConnection::peer_closed() {
    pollfd p{fd_, POLLIN, 0};
    if (::poll(&p, 1, 0) <= 0) return false;
    char c;
    const ssize_t n = ::recv(fd_, &c, 1, MSG_PEEK);
    return n == 0;
}

std::string http_response(int status, const std::string& content_type, const std::string& body,
                          const std::vector<std::pair<std::string, std::string>>& extra) {
    std::string r = "HTTP/1.1 " + std::to_string(status) + " " + status_text(status) + "\r\n";
    if (!content_type.empty()) r += "Content-Type: " + content_type + "\r\n";
    r += "Content-Length: " + std::to_string(body.size()) + "\r\n";
    r += "Cache-Control: no-store\r\nConnection: close\r\n";
    for (const auto& h : extra) r += h.first + ": " + h.second + "\r\n";
    r += "\r\n";
    r += body;
    return r;
}

FakeHttpServer::~FakeHttpServer() { stop(); }

bool FakeHttpServer::start(Handler handler) {
    handler_ = std::move(handler);
    listen_fd_ = ::socket(AF_INET, SOCK_STREAM, 0);
    if (listen_fd_ < 0) return false;
    (void)fcntl(listen_fd_, F_SETFD, FD_CLOEXEC);
    int one = 1;
    (void)setsockopt(listen_fd_, SOL_SOCKET, SO_REUSEADDR, &one, sizeof one);
    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    addr.sin_port = 0;
    if (::bind(listen_fd_, reinterpret_cast<sockaddr*>(&addr), sizeof addr) != 0 || ::listen(listen_fd_, 64) != 0) {
        ::close(listen_fd_);
        listen_fd_ = -1;
        return false;
    }
    socklen_t len = sizeof addr;
    (void)getsockname(listen_fd_, reinterpret_cast<sockaddr*>(&addr), &len);
    port_ = ntohs(addr.sin_port);
    stopping_ = false;
    acceptor_ = std::thread([this] { accept_loop(); });
    return true;
}

void FakeHttpServer::stop() {
    if (listen_fd_ < 0 && !acceptor_.joinable()) return;
    stopping_ = true;
    if (acceptor_.joinable()) acceptor_.join();
    if (listen_fd_ >= 0) {
        ::close(listen_fd_);
        listen_fd_ = -1;
    }
    std::list<std::thread> joining;
    {
        std::lock_guard<std::mutex> g(mu_);
        joining.swap(connections_);
    }
    for (auto& t : joining)
        if (t.joinable()) t.join();
}

std::vector<FakeRequest> FakeHttpServer::requests() const {
    std::lock_guard<std::mutex> g(mu_);
    return requests_;
}

std::size_t FakeHttpServer::request_count() const {
    std::lock_guard<std::mutex> g(mu_);
    return requests_.size();
}

void FakeHttpServer::clear_requests() {
    std::lock_guard<std::mutex> g(mu_);
    requests_.clear();
}

void FakeHttpServer::accept_loop() {
    while (!stopping_.load()) {
        pollfd p{listen_fd_, POLLIN, 0};
        if (::poll(&p, 1, 50) <= 0) continue;
        const int fd = ::accept(listen_fd_, nullptr, nullptr);
        if (fd < 0) continue;
        (void)fcntl(fd, F_SETFD, FD_CLOEXEC);
#ifdef SO_NOSIGPIPE
        int one = 1;
        (void)setsockopt(fd, SOL_SOCKET, SO_NOSIGPIPE, &one, sizeof one);
#endif
        std::lock_guard<std::mutex> g(mu_);
        connections_.emplace_back([this, fd] { serve(fd); });
    }
}

void FakeHttpServer::serve(int fd) {
    std::string buf;
    char chunk[8192];
    std::size_t head_end = std::string::npos;
    std::size_t want = 0;
    bool have_length = false;
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(10);
    while (!stopping_.load() && std::chrono::steady_clock::now() < deadline) {
        if (head_end != std::string::npos && buf.size() >= head_end + 4 + want) break;
        pollfd p{fd, POLLIN, 0};
        if (::poll(&p, 1, 50) <= 0) continue;
        const ssize_t n = ::recv(fd, chunk, sizeof chunk, 0);
        if (n <= 0) break;
        buf.append(chunk, std::size_t(n));
        if (buf.size() > (4u << 20)) break;
        if (head_end == std::string::npos) {
            head_end = buf.find("\r\n\r\n");
            if (head_end != std::string::npos) {
                const std::string head = lower(buf.substr(0, head_end));
                const auto at = head.find("\r\ncontent-length:");
                if (at != std::string::npos) {
                    have_length = true;
                    want = std::stoul(head.substr(at + 17));
                }
            }
        }
    }
    FakeRequest req;
    if (head_end != std::string::npos) {
        req.head = buf.substr(0, head_end);
        std::size_t line_end = req.head.find("\r\n");
        const std::string first = req.head.substr(0, line_end);
        const auto sp1 = first.find(' ');
        const auto sp2 = first.find(' ', sp1 + 1);
        if (sp1 != std::string::npos && sp2 != std::string::npos) {
            req.method = first.substr(0, sp1);
            req.target = first.substr(sp1 + 1, sp2 - sp1 - 1);
        }
        while (line_end != std::string::npos) {
            const std::size_t start = line_end + 2;
            line_end = req.head.find("\r\n", start);
            const std::string line = req.head.substr(start, line_end == std::string::npos ? std::string::npos : line_end - start);
            const auto colon = line.find(':');
            if (colon == std::string::npos) continue;
            std::string value = line.substr(colon + 1);
            while (!value.empty() && value.front() == ' ') value.erase(value.begin());
            req.headers.emplace_back(lower(line.substr(0, colon)), value);
        }
        if (have_length) req.body = buf.substr(head_end + 4, want);
        {
            std::lock_guard<std::mutex> g(mu_);
            requests_.push_back(req);
        }
        FakeConnection conn(fd, &stopping_);
        if (handler_) handler_(req, conn);
    }
    ::close(fd);
}

}  // namespace caliper_builder::mcp::test
