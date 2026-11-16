// caliper-builder-host: a headless development tool that stands in for the
// command endpoints of the app's Caliper API, so that caliper-mcp and
// caliper-agent can be tested end to end before the libraries are wired into
// the app (docs/design/builder-session.md, The stand-in host). The app's own
// API server replaces it at the wiring; it is not part of the product.
//
//   caliper-builder-host --workspace <dir> --api-dir <dir>
//                        [--port <n>] [--approve allow|deny|ask]
//
// It opens the workspace as a Session, serves GET /v0/commands and
// POST /v0/commands/{name} on 127.0.0.1 (a free port unless --port says), and
// writes <api-dir>/api.json as the app writes its API file (mode 0600,
// removed on exit). The main thread drains the command queue, as the app's
// frame thread will. A command that needs approval waits; --approve decides
// what it gets here: allow approves it, deny denies it, and ask (the default)
// prints one JSON line per pending approval on standard output,
// {"approval", "name", "origin", "args"}, and reads "allow <id>" or
// "deny <id>" lines on standard input. SIGTERM or SIGINT stops it cleanly:
// waiting requests are answered with command.shutdown, the API file is
// removed, and it exits with 0. It never prints the session token.
#include <caliper_builder/session/command_server.h>
#include <caliper_builder/session/session.h>

#include <nlohmann/json.hpp>

#include <cstdio>
#include <cstring>
#include <iostream>
#include <set>
#include <string>
#include <thread>

#ifndef _WIN32
#include <csignal>
#include <poll.h>
#include <unistd.h>
#endif

namespace {

void usage() {
    std::fprintf(stderr,
                 "usage: caliper-builder-host --workspace <dir> --api-dir <dir> [--port <n>] "
                 "[--approve allow|deny|ask]\n");
}

}  // namespace

#ifdef _WIN32

int main() {
    std::fprintf(stderr, "caliper-builder-host runs on macOS and Linux only for now.\n");
    return 1;
}

#else

namespace {

volatile std::sig_atomic_t g_stop = 0;

void on_signal(int) { g_stop = 1; }

}  // namespace

int main(int argc, char** argv) {
    using namespace caliper_builder;
    std::string workspace;
    std::string api_dir;
    std::string approve = "ask";
    long port = 0;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto value = [&](std::string* out) {
            if (i + 1 >= argc) return false;
            *out = argv[++i];
            return true;
        };
        std::string text;
        if (a == "--workspace" && value(&workspace)) continue;
        if (a == "--api-dir" && value(&api_dir)) continue;
        if (a == "--approve" && value(&approve)) continue;
        if (a == "--port" && value(&text)) {
            char* end = nullptr;
            port = std::strtol(text.c_str(), &end, 10);
            if (!end || *end != '\0' || port < 0 || port > 65535) {
                usage();
                return 2;
            }
            continue;
        }
        usage();
        return 2;
    }
    if (workspace.empty() || api_dir.empty() || (approve != "allow" && approve != "deny" && approve != "ask")) {
        usage();
        return 2;
    }

    std::signal(SIGPIPE, SIG_IGN);
    struct sigaction sa;
    std::memset(&sa, 0, sizeof sa);
    sa.sa_handler = on_signal;
    sigemptyset(&sa.sa_mask);
    sigaction(SIGTERM, &sa, nullptr);
    sigaction(SIGINT, &sa, nullptr);

    Session::Open opened = Session::open(workspace);
    if (opened.error) {
        std::fprintf(stderr, "caliper-builder-host: the workspace could not be opened: %s (%s)\n",
                     opened.error.message.c_str(), opened.error.code.c_str());
        return 1;
    }
    Session& session = *opened.session;
    // The main thread drains the queue, so it is the frame thread here.
    session.registry().set_frame_thread(std::this_thread::get_id());

    CommandServerOptions options;
    options.port = static_cast<std::uint16_t>(port);
    options.token = generate_session_token();
    CommandServer server(session.queue(), options);
    if (Error e = server.start()) {
        std::fprintf(stderr, "caliper-builder-host: %s\n", e.message.c_str());
        return 1;
    }
    std::string api_file;
    if (Error e = write_api_file(api_dir, server.port(), options.token, &api_file)) {
        std::fprintf(stderr, "caliper-builder-host: %s\n", e.message.c_str());
        server.stop();
        return 1;
    }
    std::fprintf(stderr,
                 "caliper-builder-host: the command endpoints listen on http://127.0.0.1:%u/v0; the API file is %s; "
                 "approvals: %s\n",
                 static_cast<unsigned>(server.port()), api_file.c_str(), approve.c_str());

    std::set<std::string> announced;
    std::string input;
    bool stdin_open = approve == "ask";
    while (!g_stop) {
        session.queue().drain();
        for (const PendingApproval& p : session.registry().pending_approvals()) {
            if (!announced.insert(p.approval).second) continue;
            if (approve == "allow") {
                session.queue().approve(p.approval);
            } else if (approve == "deny") {
                session.queue().deny(p.approval);
            } else {
                const nlohmann::json line{{"approval", p.approval},
                                          {"name", p.invocation.name},
                                          {"origin", command_origin_name(p.invocation.origin)},
                                          {"args", p.invocation.args}};
                std::printf("%s\n", line.dump(-1, ' ', false, nlohmann::json::error_handler_t::replace).c_str());
                std::fflush(stdout);
            }
        }
        pollfd p{0, POLLIN, 0};
        const int ready = stdin_open ? ::poll(&p, 1, 5) : (::poll(nullptr, 0, 5), 0);
        if (ready > 0) {
            char buf[1024];
            const ssize_t n = ::read(0, buf, sizeof buf);
            if (n <= 0) {
                stdin_open = false;
            } else {
                input.append(buf, static_cast<std::size_t>(n));
                std::size_t eol;
                while ((eol = input.find('\n')) != std::string::npos) {
                    const std::string line = input.substr(0, eol);
                    input.erase(0, eol + 1);
                    if (line.rfind("allow ", 0) == 0) session.queue().approve(line.substr(6));
                    else if (line.rfind("deny ", 0) == 0) session.queue().deny(line.substr(5));
                }
                if (input.size() > 4096) input.clear();
            }
        }
    }

    remove_api_file(api_file);
    session.queue().shutdown();
    server.stop();
    session.close();
    std::fprintf(stderr, "caliper-builder-host: stopped\n");
    return 0;
}

#endif
