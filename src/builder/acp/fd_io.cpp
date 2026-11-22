#include "fd_io.h"

#ifndef _WIN32

#include <cerrno>
#include <csignal>
#include <ctime>
#include <fcntl.h>
#include <poll.h>
#include <pthread.h>
#include <unistd.h>

namespace caliper_builder::acp::detail {

bool set_cloexec(int fd) {
    const int flags = fcntl(fd, F_GETFD);
    return flags >= 0 && fcntl(fd, F_SETFD, flags | FD_CLOEXEC) == 0;
}

bool set_nonblocking(int fd) {
    const int flags = fcntl(fd, F_GETFL);
    return flags >= 0 && fcntl(fd, F_SETFL, flags | O_NONBLOCK) == 0;
}

void set_nosigpipe(int fd) {
#ifdef F_SETNOSIGPIPE
    (void)fcntl(fd, F_SETNOSIGPIPE, 1);
#else
    (void)fd;
#endif
}

bool make_pipe(int fds[2]) {
#if defined(__linux__)
    return pipe2(fds, O_CLOEXEC) == 0;
#else
    if (pipe(fds) != 0) return false;
    if (!set_cloexec(fds[0]) || !set_cloexec(fds[1])) {
        close(fds[0]);
        close(fds[1]);
        return false;
    }
    return true;
#endif
}

namespace {

// Waits for `events` on `fd`. Returns 1 when ready, 0 on timeout, -1 on error;
// `revents` receives what poll reported.
int wait_fd(int fd, short events, int timeout_ms, short* revents) {
    pollfd p{};
    p.fd = fd;
    p.events = events;
    const int r = poll(&p, 1, timeout_ms < 0 ? -1 : timeout_ms);
    if (r < 0) return errno == EINTR ? 0 : -1;
    *revents = p.revents;
    return r == 0 ? 0 : 1;
}

ssize_t write_without_sigpipe(int fd, const char* data, std::size_t size) {
#ifdef F_SETNOSIGPIPE
    // set_nosigpipe() was applied to the descriptor when it was made.
    return write(fd, data, size);
#else
    // Block SIGPIPE in this thread for the write and swallow the one it raises.
    sigset_t pipe_set;
    sigset_t old_set;
    sigset_t pending;
    sigemptyset(&pipe_set);
    sigaddset(&pipe_set, SIGPIPE);
    sigpending(&pending);
    const bool was_pending = sigismember(&pending, SIGPIPE) == 1;
    pthread_sigmask(SIG_BLOCK, &pipe_set, &old_set);
    const ssize_t n = write(fd, data, size);
    const int saved = errno;
    if (n < 0 && saved == EPIPE && !was_pending) {
        timespec zero{0, 0};
        sigtimedwait(&pipe_set, nullptr, &zero);
    }
    pthread_sigmask(SIG_SETMASK, &old_set, nullptr);
    errno = saved;
    return n;
#endif
}

}  // namespace

IoStatus read_fd(int fd, char* buffer, std::size_t capacity, std::size_t* got, int timeout_ms) {
    *got = 0;
    if (fd < 0 || capacity == 0) return IoStatus::error;
    short revents = 0;
    const int ready = wait_fd(fd, POLLIN, timeout_ms, &revents);
    if (ready < 0) return IoStatus::error;
    if (ready == 0) return IoStatus::timeout;
    const ssize_t n = read(fd, buffer, capacity);
    if (n > 0) {
        *got = static_cast<std::size_t>(n);
        return IoStatus::ok;
    }
    if (n == 0) return IoStatus::eof;
    if (errno == EAGAIN || errno == EWOULDBLOCK || errno == EINTR) return IoStatus::timeout;
    return IoStatus::error;
}

IoStatus write_fd(int fd, const char* data, std::size_t size, std::size_t* written, int timeout_ms) {
    *written = 0;
    if (fd < 0) return IoStatus::error;
    if (size == 0) return IoStatus::ok;
    short revents = 0;
    const int ready = wait_fd(fd, POLLOUT, timeout_ms, &revents);
    if (ready < 0) return IoStatus::error;
    if (ready == 0) return IoStatus::timeout;
    if ((revents & (POLLERR | POLLHUP)) != 0 && (revents & POLLOUT) == 0) return IoStatus::eof;
    const ssize_t n = write_without_sigpipe(fd, data, size);
    if (n > 0) {
        *written = static_cast<std::size_t>(n);
        return IoStatus::ok;
    }
    if (n < 0 && errno == EPIPE) return IoStatus::eof;
    if (n < 0 && (errno == EAGAIN || errno == EWOULDBLOCK || errno == EINTR)) return IoStatus::timeout;
    return IoStatus::error;
}

IoStatus write_fd_all(int fd, const char* data, std::size_t size, int timeout_ms) {
    std::size_t done = 0;
    while (done < size) {
        std::size_t n = 0;
        const IoStatus st = write_fd(fd, data + done, size - done, &n, timeout_ms);
        if (st != IoStatus::ok) return st;
        done += n;
    }
    return IoStatus::ok;
}

}  // namespace caliper_builder::acp::detail

#endif  // _WIN32
