#pragma once
// Internal to the acp module: reads and writes on POSIX file descriptors with
// a timeout, shared by Process and the pipe channels. A write never raises
// SIGPIPE; a closed reader shows as IoStatus::eof.
#ifndef _WIN32

#include <caliper_builder/acp/subprocess.h>

#include <cstddef>

namespace caliper_builder::acp::detail {

// Marks a descriptor close-on-exec and, for the parent's ends of the pipes,
// non-blocking. Returns false on failure.
bool set_cloexec(int fd);
bool set_nonblocking(int fd);
// On platforms that support it, turns SIGPIPE off for writes to `fd`.
void set_nosigpipe(int fd);
// Creates a pipe whose two ends are close-on-exec. Returns false on failure.
bool make_pipe(int fds[2]);

IoStatus read_fd(int fd, char* buffer, std::size_t capacity, std::size_t* got, int timeout_ms);
IoStatus write_fd(int fd, const char* data, std::size_t size, std::size_t* written, int timeout_ms);
IoStatus write_fd_all(int fd, const char* data, std::size_t size, int timeout_ms);

}  // namespace caliper_builder::acp::detail

#endif
