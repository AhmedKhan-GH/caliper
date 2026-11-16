#pragma once
// CommandServer: the two command endpoints of Caliper API (sprint 3
// contracts, sections 8 and 15) on a minimal, strict HTTP/1.1 server that
// listens on 127.0.0.1 only. It serves caliper-builder-host, the development
// stand-in that lets caliper-mcp and caliper-agent be tested end to end before
// the app is wired; the app's own API server replaces it at the wiring
// (design/builder-session.md, The stand-in host).
//
//   GET  /v0/commands         the descriptor table, command_table_to_json()
//   POST /v0/commands/{name}  the `args` object as a JSON body; the result
//                             envelope with HTTP 200 whether ok or not
//
// Every request is checked in this order, and the first fault answers:
//   - the request head: at most 16 KiB (431), a well-formed request line and
//     headers with CRLF endings, no repeated header that matters (400), and
//     no Transfer-Encoding (501);
//   - `Host` must be 127.0.0.1:<port> or localhost:<port> (403), and an
//     `Origin`, when present, must be null, an http or https origin on
//     127.0.0.1, localhost or [::1], or vscode-webview:// (403);
//   - the path and method (404, 405 with Allow);
//   - the session token, in `Authorization: Bearer <token>`, or for a GET in
//     the query parameter `token`, compared in constant time (401 with
//     WWW-Authenticate: Bearer);
//   - a GET has no body (413); a POST has a Content-Length (411) of at most
//     1 MiB (413) and the type application/json (415); the body is read
//     within its time limit (408) and must be JSON (400).
// A body that is JSON but not an object, or nests deeper than 64 levels, is
// answered 200 with command.invalid_args, and a name that is not a command
// with command.unknown, so that a 404 always means "no command layer".
//
// The origin is api, or mcp or agent when X-Caliper-Origin says so; any other
// value is refused with 400, so a request can never claim the ui or builder
// origin. X-Caliper-Group gives the group, and X-Caliper-Preview the id of a
// stacked preview that the call's edits apply to. A call from mcp or agent that needs
// approval is held while the person decides, up to `approval_wait` (120 s);
// a call from api gets command.needs_approval with the approval id at once.
//
// One request per connection: the response carries Connection: close,
// Cache-Control: no-store and X-Content-Type-Options: nosniff, and errors are
// {"error": "..."}. At most `max_connections` connections are served at once;
// beyond that a connection gets 503. The token is never logged or echoed.
//
// POSIX only for now: on Windows start() fails with server.unsupported. No
// function here throws.
#include <caliper_builder/commands/queue.h>
#include <caliper_builder/error.h>

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <string_view>

namespace caliper_builder {

struct CommandServerOptions {
    std::uint16_t port = 0;  // 0 takes any free port
    std::string token;       // the session token; start() refuses an empty one
    std::size_t max_connections = 32;
    std::size_t max_head_bytes = 16 * 1024;
    std::size_t max_body_bytes = std::size_t(1) << 20;
    int head_timeout_ms = 5000;
    int body_timeout_ms = 10000;
    int write_timeout_ms = 10000;
    std::chrono::milliseconds approval_wait{120000};
    // How long a request waits for its command once it is queued; then 504.
    std::chrono::milliseconds max_command_wait{600000};
};

class CommandServer {
public:
    CommandServer(CommandQueue& queue, CommandServerOptions options);
    ~CommandServer();
    CommandServer(const CommandServer&) = delete;
    CommandServer& operator=(const CommandServer&) = delete;

    // Binds 127.0.0.1 and starts accepting on a thread of its own.
    Error start();
    std::uint16_t port() const;
    // Stops accepting and waits for the connections being served. A request
    // that waits for its command waits until the queue answers it, so shut the
    // queue down first.
    void stop();
    // The requests answered so far, of any status.
    std::uint64_t served() const;

    struct Impl;

private:
    std::unique_ptr<Impl> impl_;
};

// A new session token: 32 characters of A-Z a-z 0-9 _ - from 24 random bytes.
std::string generate_session_token();

// Compares two tokens in time that depends only on their lengths.
bool token_equal(std::string_view a, std::string_view b);

// Writes <dir>/api.json, as the app writes its API file: {"version": 1,
// "pid", "port", "token", "url": "http://127.0.0.1:<port>/v0"}, with mode
// 0600, through a temporary file and a rename. The folder is created (0700)
// when missing. Fills `path` with the file's path.
Error write_api_file(const std::string& dir, std::uint16_t port, const std::string& token, std::string* path);

// Removes the API file when it is still this process's (its pid is ours).
void remove_api_file(const std::string& path);

}  // namespace caliper_builder
