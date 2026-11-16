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
#include <spawn.h>
#include <sys/wait.h>
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
    // The query: the token, then at most one applet, and nothing else.
    std::string query = url.substr(i + 9), token, applet;
    bool has_applet = false;
    while (!query.empty()) {
        const size_t amp = query.find('&');
        const std::string part = query.substr(0, amp);
        query = amp == std::string::npos ? std::string() : query.substr(amp + 1);
        if (part.rfind("token=", 0) == 0 && token.empty()) {
            token = part.substr(6);
            if (token.empty()) return false;
            for (const char c : token)
                if (!std::isalnum((unsigned char)c) && c != '_' && c != '-') return false;
        } else if (part.rfind("applet=", 0) == 0 && !has_applet) {
            has_applet = true;
            applet = part.substr(7);
        } else {
            return false;
        }
        if (amp == std::string::npos) break;
    }
    return !token.empty() && (!has_applet || viewer_applet_id(applet));
}

bool viewer_applet_id(const std::string& id) {
    // The workspace's rule for an applet id, the spec format's: a lower-case
    // letter, then at most 63 lower-case letters, digits, _ or -.
    if (id.empty() || id.size() > 64 || id[0] < 'a' || id[0] > 'z') return false;
    for (const char c : id)
        if (!((c >= 'a' && c <= 'z') || (c >= '0' && c <= '9') || c == '_' || c == '-')) return false;
    return true;
}

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
    std::string arg0 = program, arg1 = url;
    char* argv[] = {arg0.data(), arg1.data(), nullptr};
    pid_t pid = 0;
#ifdef __APPLE__
    const int rc = posix_spawn(&pid, program, nullptr, nullptr, argv, environ);
#else
    const int rc = posix_spawnp(&pid, program, nullptr, nullptr, argv, environ);
#endif
    if (rc != 0) {
        if (why) *why = std::string("the browser did not start: ") + std::strerror(rc);
        return false;
    }
    // The opener hands the link over and exits; it is reaped off the frame thread.
    std::thread([pid] {
        int status = 0;
        waitpid(pid, &status, 0);
    }).detach();
    return true;
#endif
}

}  // namespace caliper_host
