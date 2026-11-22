// The child process on POSIX (child_process.h): posix_spawn with the output
// file as stdout and stderr (mode 0600, never through a link), /dev/null as
// stdin, a process group of its own and the default handlers of the common
// signals. A signal goes to the child's whole process group.
#include "child_process.h"

#include <cerrno>
#include <chrono>
#include <csignal>
#include <cstring>
#include <thread>

#include <fcntl.h>
#include <spawn.h>
#include <sys/wait.h>
#include <unistd.h>

#if defined(__APPLE__)
#include <crt_externs.h>
#define CALIPER_ENVIRON (*_NSGetEnviron())
#else
extern char** environ;
#define CALIPER_ENVIRON environ
#endif

namespace caliper_runner {

namespace {

// Asks posix_spawn to change the child's working folder. POSIX.1-2024 names
// the call posix_spawn_file_actions_addchdir; older systems have the _np form.
bool add_chdir(posix_spawn_file_actions_t* actions, const std::string& dir) {
#if defined(__APPLE__) || (defined(__GLIBC__) && (__GLIBC__ > 2 || (__GLIBC__ == 2 && __GLIBC_MINOR__ >= 29)))
#if defined(__clang__)
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
#endif
    return posix_spawn_file_actions_addchdir_np(actions, dir.c_str()) == 0;
#if defined(__clang__)
#pragma clang diagnostic pop
#endif
#else
    (void)actions;
    (void)dir;
    return false;
#endif
}

class PosixChild final : public ChildProcess {
public:
    explicit PosixChild(pid_t pid) : pid_(pid) {}

    std::int64_t pid() const override { return pid_; }

    bool poll(std::int64_t* exit_code) override {
        if (ended_) {
            *exit_code = code_;
            return true;
        }
        int status = 0;
        const pid_t r = ::waitpid(pid_, &status, WNOHANG);
        if (r == 0) return false;
        if (r == pid_) {
            code_ = WIFEXITED(status) ? WEXITSTATUS(status) : WIFSIGNALED(status) ? -WTERMSIG(status) : -1;
        } else if (r < 0 && errno == EINTR) {
            return false;
        } else {
            code_ = -1;   // ECHILD: someone else reaped it; the code is lost
        }
        ended_ = true;
        *exit_code = code_;
        return true;
    }

    void terminate() override { signal(SIGTERM); }
    void kill() override { signal(SIGKILL); }

    bool wait(int timeout_ms, std::int64_t* exit_code) override {
        const auto until = std::chrono::steady_clock::now() + std::chrono::milliseconds(timeout_ms);
        for (;;) {
            if (poll(exit_code)) return true;
            if (std::chrono::steady_clock::now() >= until) return false;
            std::this_thread::sleep_for(std::chrono::milliseconds(5));
        }
    }

private:
    void signal(int sig) {
        std::int64_t code = 0;
        // The id is still this child's, and the group's: it has not been
        // reaped. The group holds the processes that the child started, such
        // as the interpreter behind a launcher, unless one left it.
        if (!poll(&code) && ::kill(-pid_, sig) != 0) ::kill(pid_, sig);
    }

    pid_t pid_;
    bool ended_ = false;
    std::int64_t code_ = 0;
};

}  // namespace

std::vector<std::pair<std::string, std::string>> current_environment() {
    std::vector<std::pair<std::string, std::string>> out;
    for (char** e = CALIPER_ENVIRON; e && *e; ++e) {
        const std::string kv(*e);
        const std::size_t eq = kv.find('=');
        if (eq == std::string::npos || eq == 0) continue;
        out.emplace_back(kv.substr(0, eq), kv.substr(eq + 1));
    }
    return out;
}

std::int64_t current_process_id() { return static_cast<std::int64_t>(::getpid()); }

std::unique_ptr<ChildProcess> start_child(const ChildSpec& spec, std::string* error) {
    auto failed = [&](const std::string& why) -> std::unique_ptr<ChildProcess> {
        if (error) *error = "The program " + spec.program + " could not start: " + why;
        return nullptr;
    };
    if (spec.program.empty()) return failed("no program was named.");

    posix_spawn_file_actions_t actions;
    posix_spawnattr_t attr;
    posix_spawn_file_actions_init(&actions);
    posix_spawnattr_init(&attr);
    posix_spawn_file_actions_addopen(&actions, STDIN_FILENO, "/dev/null", O_RDONLY, 0);
    posix_spawn_file_actions_addopen(&actions, STDOUT_FILENO, spec.output.empty() ? "/dev/null" : spec.output.c_str(),
                                     O_WRONLY | O_CREAT | O_APPEND | O_NOFOLLOW, 0600);
    posix_spawn_file_actions_adddup2(&actions, STDOUT_FILENO, STDERR_FILENO);
    const bool chdir_ok = spec.cwd.empty() || add_chdir(&actions, spec.cwd);

    short flags = POSIX_SPAWN_SETSIGMASK | POSIX_SPAWN_SETSIGDEF | POSIX_SPAWN_SETPGROUP;
#ifdef POSIX_SPAWN_CLOEXEC_DEFAULT
    flags |= POSIX_SPAWN_CLOEXEC_DEFAULT;   // only stdin, stdout and stderr reach the child
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

    std::vector<std::string> argv_s{spec.program};
    argv_s.insert(argv_s.end(), spec.args.begin(), spec.args.end());
    std::vector<char*> argv;
    for (auto& a : argv_s) argv.push_back(a.data());
    argv.push_back(nullptr);
    std::vector<std::string> env_s;
    for (const auto& kv : spec.env) env_s.push_back(kv.first + "=" + kv.second);
    std::vector<char*> envp;
    for (auto& v : env_s) envp.push_back(v.data());
    envp.push_back(nullptr);

    pid_t pid = -1;
    const int rc = chdir_ok ? ::posix_spawn(&pid, spec.program.c_str(), &actions, &attr, argv.data(), envp.data())
                            : ENOTSUP;
    posix_spawn_file_actions_destroy(&actions);
    posix_spawnattr_destroy(&attr);
    if (!chdir_ok) return failed("this platform cannot start a child in another working folder.");
    if (rc != 0) return failed(std::string(std::strerror(rc)) + ".");
    return std::make_unique<PosixChild>(pid);
}

}  // namespace caliper_runner
