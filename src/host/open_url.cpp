// Opening a link of the browser viewer in the default browser (open_url.h).
#include "open_url.h"

#include <cctype>
#include <cstring>
#include <thread>

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#include <shellapi.h>
#else
#include <fcntl.h>
#include <signal.h>
#include <spawn.h>
#include <sys/wait.h>
#include <unistd.h>
extern char** environ;
#endif

namespace caliper_host {

bool viewer_link_openable(const std::string& url) {
    static const char kStart[] = "http://127.0.0.1:";
    if (url.size() > 1024 || url.compare(0, sizeof kStart - 1, kStart) != 0) return false;
    size_t i = sizeof kStart - 1;
    const size_t digits = i;
    while (i < url.size() && std::isdigit((unsigned char)url[i])) ++i;
    if (i == digits || i - digits > 5) return false;
    if (url.compare(i, 9, "/v0/spec?") != 0) return false;
    // The query: the token, then at most one instrument, and nothing else.
    std::string query = url.substr(i + 9), token, instrument;
    bool has_instrument = false;
    while (!query.empty()) {
        const size_t amp = query.find('&');
        const std::string part = query.substr(0, amp);
        query = amp == std::string::npos ? std::string() : query.substr(amp + 1);
        if (part.rfind("token=", 0) == 0 && token.empty()) {
            token = part.substr(6);
            if (token.empty()) return false;
            for (const char c : token)
                if (!std::isalnum((unsigned char)c) && c != '_' && c != '-') return false;
        } else if (part.rfind("instrument=", 0) == 0 && !has_instrument) {
            has_instrument = true;
            instrument = part.substr(11);
        } else {
            return false;
        }
        if (amp == std::string::npos) break;
    }
    return !token.empty() && (!has_instrument || viewer_instrument_id(instrument));
}

bool viewer_instrument_id(const std::string& id) {
    // The workspace's rule for an instrument id, the spec format's: a lower-case
    // letter, then at most 63 lower-case letters, digits, _ or -.
    if (id.empty() || id.size() > 64 || id[0] < 'a' || id[0] > 'z') return false;
    for (const char c : id)
        if (!((c >= 'a' && c <= 'z') || (c >= '0' && c <= '9') || c == '_' || c == '-')) return false;
    return true;
}

#ifndef _WIN32
bool spawn_detached(const std::vector<std::string>& args, std::string* why) {
    if (args.empty()) {
        if (why) *why = "nothing to start";
        return false;
    }
    std::vector<std::string> copy = args;
    std::vector<char*> argv;
    for (auto& a : copy) argv.push_back(a.data());
    argv.push_back(nullptr);
    // Only the standard descriptors reach the child, each on /dev/null, so
    // that no descriptor of the app (a listening socket, a pipe) passes to
    // it or to a browser it starts, and nothing it prints lands in the app's
    // output; the signal mask and dispositions are the defaults, and it
    // leads a process group of its own.
    posix_spawn_file_actions_t actions;
    posix_spawnattr_t attr;
    posix_spawn_file_actions_init(&actions);
    posix_spawnattr_init(&attr);
    posix_spawn_file_actions_addopen(&actions, STDIN_FILENO, "/dev/null", O_RDONLY, 0);
    posix_spawn_file_actions_addopen(&actions, STDOUT_FILENO, "/dev/null", O_WRONLY, 0);
    posix_spawn_file_actions_addopen(&actions, STDERR_FILENO, "/dev/null", O_WRONLY, 0);
    short flags = POSIX_SPAWN_SETSIGMASK | POSIX_SPAWN_SETSIGDEF | POSIX_SPAWN_SETPGROUP;
#if defined(POSIX_SPAWN_CLOEXEC_DEFAULT)
    flags |= POSIX_SPAWN_CLOEXEC_DEFAULT;   // every descriptor not acted on above is closed
#elif defined(__GLIBC__) && (__GLIBC__ > 2 || (__GLIBC__ == 2 && __GLIBC_MINOR__ >= 34))
    posix_spawn_file_actions_addclosefrom_np(&actions, 3);
#endif
    posix_spawnattr_setflags(&attr, flags);
    posix_spawnattr_setpgroup(&attr, 0);
    sigset_t empty_set;
    sigemptyset(&empty_set);
    posix_spawnattr_setsigmask(&attr, &empty_set);
    sigset_t defaults;
    sigemptyset(&defaults);
    for (int sig : {SIGPIPE, SIGINT, SIGTERM, SIGHUP, SIGQUIT, SIGCHLD}) sigaddset(&defaults, sig);
    posix_spawnattr_setsigdefault(&attr, &defaults);
    pid_t pid = 0;
#ifdef __APPLE__
    const int rc = posix_spawn(&pid, argv[0], &actions, &attr, argv.data(), environ);
#else
    const int rc = posix_spawnp(&pid, argv[0], &actions, &attr, argv.data(), environ);
#endif
    posix_spawn_file_actions_destroy(&actions);
    posix_spawnattr_destroy(&attr);
    if (rc != 0) {
        if (why) *why = std::string("cannot start ") + args[0] + ": " + std::strerror(rc);
        return false;
    }
    // It hands its work over and exits; it is reaped off the calling thread.
    std::thread([pid] {
        int status = 0;
        waitpid(pid, &status, 0);
    }).detach();
    return true;
}
#endif

bool open_in_default_browser(const std::string& url, std::string* why) {
    if (!viewer_link_openable(url)) {
        if (why) *why = "not a link of the browser viewer";
        return false;
    }
#ifdef _WIN32
    const int n = MultiByteToWideChar(CP_UTF8, 0, url.c_str(), -1, nullptr, 0);
    std::wstring wide((size_t)(n > 0 ? n : 1), L'\0');
    MultiByteToWideChar(CP_UTF8, 0, url.c_str(), -1, wide.data(), n);
    const HINSTANCE r = ShellExecuteW(nullptr, L"open", wide.c_str(), nullptr, nullptr, SW_SHOWNORMAL);
    if ((INT_PTR)r <= 32) {
        if (why) *why = "the browser did not start";
        return false;
    }
    return true;
#else
#ifdef __APPLE__
    const char* program = "/usr/bin/open";
#else
    const char* program = "xdg-open";
#endif
    std::string err;
    if (!spawn_detached({program, url}, &err)) {
        if (why) *why = "the browser did not start: " + err;
        return false;
    }
    return true;
#endif
}

}  // namespace caliper_host
