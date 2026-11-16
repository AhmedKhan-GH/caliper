#pragma once
// A child process for the process-level tests (test_cli.cpp, test_api.cpp):
// started with posix_spawn, its standard output and error read through one
// pipe, killed if it is still running when the test ends. POSIX only.
#ifndef _WIN32
#include <chrono>
#include <string>
#include <vector>

#include <fcntl.h>
#include <poll.h>
#include <signal.h>
#include <spawn.h>
#include <sys/wait.h>
#include <unistd.h>
extern char** environ;

struct Proc {
    pid_t pid = -1;
    int out = -1;
    std::string text;
    int status = -1;

    bool start(const std::string& exe, const std::vector<std::string>& args, const std::vector<std::string>& env_add) {
        int fds[2];
        if (::pipe(fds) != 0) return false;
        posix_spawn_file_actions_t fa;
        posix_spawn_file_actions_init(&fa);
        posix_spawn_file_actions_adddup2(&fa, fds[1], 1);
        posix_spawn_file_actions_adddup2(&fa, fds[1], 2);
        posix_spawn_file_actions_addclose(&fa, fds[0]);
        std::vector<std::string> argv_s = {exe};
        argv_s.insert(argv_s.end(), args.begin(), args.end());
        std::vector<char*> argv;
        for (auto& a : argv_s) argv.push_back(a.data());
        argv.push_back(nullptr);
        // The test's own environment, with each variable of env_add in place
        // of an inherited one of the same name (ctest sets CALIPER_DATA_DIR
        // for every entry, and a child's own folder must win over it).
        std::vector<std::string> env_s;
        auto name_of = [](const std::string& kv) { return kv.substr(0, kv.find('=')); };
        for (char** e = environ; *e; ++e) {
            const std::string kv(*e);
            bool replaced = false;
            for (const auto& a : env_add)
                if (name_of(a) == name_of(kv)) replaced = true;
            if (!replaced) env_s.push_back(kv);
        }
        env_s.insert(env_s.end(), env_add.begin(), env_add.end());
        std::vector<char*> envp;
        for (auto& e : env_s) envp.push_back(e.data());
        envp.push_back(nullptr);
        const int rc = posix_spawn(&pid, exe.c_str(), &fa, nullptr, argv.data(), envp.data());
        posix_spawn_file_actions_destroy(&fa);
        ::close(fds[1]);
        out = fds[0];
        ::fcntl(out, F_SETFL, ::fcntl(out, F_GETFL) | O_NONBLOCK);
        return rc == 0;
    }
    void read_some(int ms) {
        pollfd p{out, POLLIN, 0};
        if (::poll(&p, 1, ms) <= 0) return;
        char buf[4096];
        for (;;) {
            const ssize_t r = ::read(out, buf, sizeof buf);
            if (r <= 0) break;
            text.append(buf, (size_t)r);
        }
    }
    bool read_until(const std::string& needle, int ms) {
        const auto end = std::chrono::steady_clock::now() + std::chrono::milliseconds(ms);
        while (text.find(needle) == std::string::npos && std::chrono::steady_clock::now() < end) read_some(20);
        return text.find(needle) != std::string::npos;
    }
    // The exit code, or -1 when it did not exit within `ms` (then it is killed).
    int wait(int ms) {
        const auto end = std::chrono::steady_clock::now() + std::chrono::milliseconds(ms);
        while (std::chrono::steady_clock::now() < end) {
            read_some(10);
            if (::waitpid(pid, &status, WNOHANG) == pid) {
                pid = -1;
                read_some(0);
                return WIFEXITED(status) ? WEXITSTATUS(status) : -1;
            }
        }
        return -1;
    }
    ~Proc() {
        if (pid > 0) {
            ::kill(pid, SIGKILL);
            ::waitpid(pid, &status, 0);
        }
        if (out >= 0) ::close(out);
    }
};

#endif
