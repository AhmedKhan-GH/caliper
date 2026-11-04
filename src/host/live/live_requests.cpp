#include "live/live_requests.h"
#include <cerrno>
#include <cstring>
#ifndef _WIN32
#include <fcntl.h>
#include <sys/socket.h>
#include <sys/un.h>
#include <unistd.h>
#include <poll.h>
#endif

namespace caliper_live {
namespace { constexpr size_t max_message = 4 * 1024 * 1024; }
void RequestChannel::close() {
#ifndef _WIN32
    if (fd_ >= 0) ::close(fd_);
#endif
    fd_ = -1;
    connecting_ = false;
    incoming_.clear(); outgoing_.clear();
}
void RequestChannel::fail(const char* reason) { error_ = reason; close(); }
bool RequestChannel::connect(const std::string& session) {
    close(); error_.clear();
#ifndef _WIN32
    if (session.empty() || session.size() > 20 ||
        session.find_first_not_of("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-") != std::string::npos)
        return false;
    sockaddr_un addr{}; addr.sun_family = AF_UNIX;
    const std::string path = "/tmp/caliper-" + std::to_string(getuid()) + "/" + session + ".sock";
    std::memcpy(addr.sun_path, path.c_str(), path.size() + 1);
    fd_ = socket(AF_UNIX, SOCK_STREAM, 0);
    if (fd_ < 0) return false;
    fcntl(fd_, F_SETFD, FD_CLOEXEC);
    fcntl(fd_, F_SETFL, O_NONBLOCK);
#ifdef SO_NOSIGPIPE
    int one = 1; setsockopt(fd_, SOL_SOCKET, SO_NOSIGPIPE, &one, sizeof one);
#endif
    if (::connect(fd_, reinterpret_cast<sockaddr*>(&addr), sizeof addr) != 0) {
        if (errno == EINPROGRESS) connecting_ = true;
        else { fail(std::strerror(errno)); return false; }
    }
    return true;
#else
    error_ = "local requests unavailable on this platform";
    return false;
#endif
}
bool RequestChannel::send(const std::string& json) {
    if (fd_ < 0) return false;
    if (json.size() + 1 >= max_message || outgoing_.size() + json.size() + 1 > max_message ||
        json.find('\n') != std::string::npos) { error_ = "request queue full or invalid frame"; return false; }
    outgoing_ += json; outgoing_ += '\n'; return true;
}
std::vector<std::string> RequestChannel::poll() {
    std::vector<std::string> result;
#ifndef _WIN32
    if (fd_ < 0) return result;
    if (connecting_) {
        pollfd p{fd_, POLLOUT, 0};
        if (::poll(&p, 1, 0) <= 0) return result;
        int err = 0; socklen_t len = sizeof err;
        if (getsockopt(fd_, SOL_SOCKET, SO_ERROR, &err, &len) || err) {
            fail("request connection failed"); return result;
        }
        connecting_ = false;
    }
    if (!outgoing_.empty()) {
#ifdef MSG_NOSIGNAL
        constexpr int flags = MSG_NOSIGNAL;
#else
        constexpr int flags = 0;
#endif
        const auto n = ::send(fd_, outgoing_.data(), outgoing_.size(), flags);
        if (n > 0) outgoing_.erase(0, static_cast<size_t>(n));
        else if (n < 0 && errno != EAGAIN && errno != EWOULDBLOCK && errno != EINTR) {
            fail("request send failed"); return result;
        }
    }
    // Bound per-frame work, even when a producer sends continuously.
    for (int chunk = 0; chunk < 16; ++chunk) {
        char buf[16384];
        const auto n = recv(fd_, buf, sizeof buf, 0);
        if (n == 0) { fail("request host disconnected"); break; }
        if (n < 0) {
            if (errno != EAGAIN && errno != EWOULDBLOCK && errno != EINTR) fail("request receive failed");
            break;
        }
        incoming_.append(buf, static_cast<size_t>(n));
        size_t start = 0, end;
        while ((end = incoming_.find('\n', start)) != std::string::npos) {
            if (end - start >= max_message) { fail("reply too large"); return {}; }
            result.push_back(incoming_.substr(start, end - start)); start = end + 1;
        }
        incoming_.erase(0, start);
        if (incoming_.size() >= max_message) { fail("reply too large"); return {}; }
    }
#endif
    return result;
}
} // namespace caliper_live
