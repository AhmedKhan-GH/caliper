#include <caliper_builder/acp/subprocess.h>

#include <chrono>
#include <cstdlib>
#include <mutex>
#include <set>
#include <thread>

#ifndef _WIN32
#include "fd_io.h"

#include <cerrno>
#include <csignal>
#include <cstring>
#include <fcntl.h>
#include <poll.h>
#include <spawn.h>
#include <sys/stat.h>
#include <sys/wait.h>
#include <unistd.h>
#if defined(__APPLE__)
#include <crt_externs.h>
#endif
#endif

namespace caliper_builder::acp {

std::string ExitStatus::describe() const {
    if (exited) return "exit code " + std::to_string(code);
    if (signal != 0) return "signal " + std::to_string(signal);
    return "unknown status";
}

namespace {

bool has_nul(const std::string& s) { return s.find('\0') != std::string::npos; }

std::string quote_arg(const std::string& a) {
    if (!a.empty() && a.find_first_of(" \t\n'\"\\$`") == std::string::npos) return a;
    std::string out = "'";
    for (char c : a) {
        if (c == '\'') out += "'\\''";
        else out += c;
    }
    return out + "'";
}

Error check_options(const SpawnOptions& options) {
    if (options.argv.empty() || options.argv[0].empty())
        return {"process.bad_argument", "The program to run is empty."};
    for (const auto& a : options.argv)
        if (has_nul(a)) return {"process.bad_argument", "An argument contains a NUL byte."};
    for (const auto& v : options.env) {
        if (v.name.empty() || v.name.find('=') != std::string::npos || has_nul(v.name))
            return {"process.bad_argument", "An environment variable has an invalid name.", v.name};
        if (has_nul(v.value))
            return {"process.bad_argument", "An environment variable's value contains a NUL byte.", v.name};
    }
    if (has_nul(options.cwd)) return {"process.bad_argument", "The working folder contains a NUL byte."};
    return {};
}

}  // namespace

std::string describe_spawn(const SpawnOptions& options) {
    std::string out;
    for (std::size_t i = 0; i < options.argv.size(); ++i) {
        if (i) out += ' ';
        out += quote_arg(options.argv[i]);
    }
    if (!options.cwd.empty()) out += " (in " + quote_arg(options.cwd) + ")";
    if (!options.env.empty()) {
        out += " env set:";
        for (const auto& v : options.env) out += " " + v.name;
    }
    if (!options.env_remove.empty()) {
        out += " env removed:";
        for (const auto& n : options.env_remove) out += " " + n;
    }
    if (!options.inherit_environment) out += " (clean environment)";
    return out;
}

#ifndef _WIN32

namespace {

char** current_environ() {
#if defined(__APPLE__)
    return *_NSGetEnviron();
#else
    extern char** environ;
    return environ;
#endif
}

bool is_executable_file(const std::string& p) {
    struct stat st {};
    return stat(p.c_str(), &st) == 0 && S_ISREG(st.st_mode) && access(p.c_str(), X_OK) == 0;
}

// The child's environment as NAME=VALUE strings.
std::vector<std::string> build_environment(const SpawnOptions& options) {
    std::vector<std::string> out;
    std::set<std::string> replaced;
    for (const auto& n : options.env_remove) replaced.insert(n);
    for (const auto& v : options.env) replaced.insert(v.name);
    if (options.inherit_environment) {
        for (char** e = current_environ(); e && *e; ++e) {
            const std::string entry(*e);
            const auto eq = entry.find('=');
            const std::string name = eq == std::string::npos ? entry : entry.substr(0, eq);
            if (replaced.count(name) == 0) out.push_back(entry);
        }
    }
    for (const auto& v : options.env) out.push_back(v.name + "=" + v.value);
    return out;
}

std::string path_of(const std::vector<std::string>& env) {
    for (const auto& e : env)
        if (e.rfind("PATH=", 0) == 0) return e.substr(5);
    const char* p = std::getenv("PATH");
    return p ? std::string(p) : std::string("/usr/bin:/bin");
}

ExitStatus decode_status(int raw) {
    ExitStatus s;
    if (WIFEXITED(raw)) {
        s.exited = true;
        s.code = WEXITSTATUS(raw);
    } else if (WIFSIGNALED(raw)) {
        s.signal = WTERMSIG(raw);
    }
    return s;
}

// Asks posix_spawn to change the child's working folder. POSIX.1-2024 names
// the call posix_spawn_file_actions_addchdir; older systems have the _np form.
bool add_chdir(posix_spawn_file_actions_t* actions, const std::string& dir) {
#if defined(__APPLE__) && defined(__MAC_26_0) && __MAC_OS_X_VERSION_MIN_REQUIRED >= __MAC_26_0
    return posix_spawn_file_actions_addchdir(actions, dir.c_str()) == 0;
#elif defined(__APPLE__) || (defined(__GLIBC__) && (__GLIBC__ > 2 || (__GLIBC__ == 2 && __GLIBC_MINOR__ >= 29)))
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

void close_fd(int& fd) {
    if (fd >= 0) {
        close(fd);
        fd = -1;
    }
}

}  // namespace

std::string find_executable(const std::string& name, const std::string& path_list) {
    if (name.empty() || has_nul(name)) return {};
    if (name.find('/') != std::string::npos) return is_executable_file(name) ? name : std::string();
    std::size_t start = 0;
    while (start <= path_list.size()) {
        const auto colon = path_list.find(':', start);
        std::string dir = path_list.substr(start, colon == std::string::npos ? std::string::npos : colon - start);
        if (dir.empty()) dir = ".";
        const std::string candidate = dir + "/" + name;
        if (is_executable_file(candidate)) return candidate;
        if (colon == std::string::npos) break;
        start = colon + 1;
    }
    return {};
}

struct Process::Impl {
    pid_t pid = -1;
    bool group = false;
    int in = -1;   // the child's stdin, our write end
    int out = -1;  // the child's stdout, our read end
    int err = -1;  // the child's stderr, our read end
    std::mutex wait_mutex;
    bool reaped = false;
    ExitStatus status;

    bool poll_exit() {
        std::lock_guard<std::mutex> lock(wait_mutex);
        if (reaped) return true;
        int raw = 0;
        const pid_t r = waitpid(pid, &raw, WNOHANG);
        if (r == pid) {
            reaped = true;
            status = decode_status(raw);
            return true;
        }
        if (r < 0 && errno == ECHILD) {  // reaped elsewhere; nothing more to learn
            reaped = true;
            return true;
        }
        return false;
    }
};

struct ProcessAccess {
    static Process::Impl& impl(Process& p) { return *p.impl_; }
};

Process::Process() : impl_(std::make_unique<Impl>()) {}

Process::~Process() {
    if (impl_ && impl_->pid > 0) terminate(2000);
    if (impl_) {
        close_fd(impl_->in);
        close_fd(impl_->out);
        close_fd(impl_->err);
    }
}

std::unique_ptr<Process> Process::spawn(const SpawnOptions& options, Error* error) {
    Error dummy;
    Error& e = error ? *error : dummy;
    e = check_options(options);
    if (e) return nullptr;

    const std::vector<std::string> env = build_environment(options);
    const std::string program = find_executable(options.argv[0], path_of(env));
    if (program.empty()) {
        e = {"process.not_found", "No program named " + quote_arg(options.argv[0]) + " was found.", options.argv[0]};
        return nullptr;
    }

    int in_pipe[2] = {-1, -1};
    int out_pipe[2] = {-1, -1};
    int err_pipe[2] = {-1, -1};
    auto close_all = [&] {
        for (int* p : {in_pipe, out_pipe, err_pipe}) {
            close_fd(p[0]);
            close_fd(p[1]);
        }
    };
    if (!detail::make_pipe(in_pipe) || !detail::make_pipe(out_pipe) || !detail::make_pipe(err_pipe)) {
        close_all();
        e = {"process.spawn_failed", std::string("Could not create pipes: ") + std::strerror(errno) + "."};
        return nullptr;
    }

    posix_spawn_file_actions_t actions;
    posix_spawnattr_t attr;
    posix_spawn_file_actions_init(&actions);
    posix_spawnattr_init(&attr);
    posix_spawn_file_actions_adddup2(&actions, in_pipe[0], STDIN_FILENO);
    posix_spawn_file_actions_adddup2(&actions, out_pipe[1], STDOUT_FILENO);
    posix_spawn_file_actions_adddup2(&actions, err_pipe[1], STDERR_FILENO);
    const bool chdir_ok = options.cwd.empty() || add_chdir(&actions, options.cwd);

    short flags = POSIX_SPAWN_SETSIGMASK | POSIX_SPAWN_SETSIGDEF;
    if (options.new_process_group) {
        flags |= POSIX_SPAWN_SETPGROUP;
        posix_spawnattr_setpgroup(&attr, 0);
    }
#ifdef POSIX_SPAWN_CLOEXEC_DEFAULT
    flags |= POSIX_SPAWN_CLOEXEC_DEFAULT;  // only the three pipes reach the child
#endif
    posix_spawnattr_setflags(&attr, flags);
    sigset_t empty_set;
    sigemptyset(&empty_set);
    posix_spawnattr_setsigmask(&attr, &empty_set);
    sigset_t defaults;
    sigemptyset(&defaults);
    for (int sig : {SIGPIPE, SIGINT, SIGTERM, SIGHUP, SIGQUIT, SIGCHLD}) sigaddset(&defaults, sig);
    posix_spawnattr_setsigdefault(&attr, &defaults);

    std::vector<char*> argv_c;
    for (const auto& a : options.argv) argv_c.push_back(const_cast<char*>(a.c_str()));
    argv_c.push_back(nullptr);
    std::vector<char*> env_c;
    for (const auto& v : env) env_c.push_back(const_cast<char*>(v.c_str()));
    env_c.push_back(nullptr);

    pid_t pid = -1;
    int rc = chdir_ok ? posix_spawn(&pid, program.c_str(), &actions, &attr, argv_c.data(), env_c.data()) : ENOTSUP;
    posix_spawn_file_actions_destroy(&actions);
    posix_spawnattr_destroy(&attr);
    close_fd(in_pipe[0]);
    close_fd(out_pipe[1]);
    close_fd(err_pipe[1]);
    if (rc != 0) {
        close_all();
        if (!chdir_ok)
            e = {"process.unsupported", "This platform cannot start a child in another working folder.", options.cwd};
        else if (rc == ENOENT || rc == EACCES || rc == ENOEXEC)
            e = {"process.not_found", "The program " + quote_arg(program) + " could not be run: " + std::strerror(rc) + ".", program};
        else
            e = {"process.spawn_failed", "Could not start " + quote_arg(program) + ": " + std::strerror(rc) + ".", program};
        return nullptr;
    }

    std::unique_ptr<Process> p(new Process());
    p->impl_->pid = pid;
    p->impl_->group = options.new_process_group;
    p->impl_->in = in_pipe[1];
    p->impl_->out = out_pipe[0];
    p->impl_->err = err_pipe[0];
    detail::set_nonblocking(p->impl_->in);
    detail::set_nonblocking(p->impl_->out);
    detail::set_nonblocking(p->impl_->err);
    detail::set_nosigpipe(p->impl_->in);
    e = {};
    return p;
}

std::int64_t Process::pid() const { return impl_->pid; }

IoStatus Process::read_stdout(char* buffer, std::size_t capacity, std::size_t* got, int timeout_ms) {
    return detail::read_fd(impl_->out, buffer, capacity, got, timeout_ms);
}

IoStatus Process::read_stderr(char* buffer, std::size_t capacity, std::size_t* got, int timeout_ms) {
    return detail::read_fd(impl_->err, buffer, capacity, got, timeout_ms);
}

IoStatus Process::write_stdin(const char* data, std::size_t size, std::size_t* written, int timeout_ms) {
    return detail::write_fd(impl_->in, data, size, written, timeout_ms);
}

IoStatus Process::write_stdin_all(const char* data, std::size_t size, int timeout_ms) {
    return detail::write_fd_all(impl_->in, data, size, timeout_ms);
}

void Process::close_stdin() { close_fd(impl_->in); }

bool Process::try_wait(ExitStatus* status) {
    if (!impl_->poll_exit()) return false;
    if (status) *status = impl_->status;
    return true;
}

bool Process::wait_for(int timeout_ms, ExitStatus* status) {
    using clock = std::chrono::steady_clock;
    const auto deadline = clock::now() + std::chrono::milliseconds(timeout_ms < 0 ? 0 : timeout_ms);
    int nap_ms = 1;
    for (;;) {
        if (try_wait(status)) return true;
        if (timeout_ms >= 0 && clock::now() >= deadline) return false;
        std::this_thread::sleep_for(std::chrono::milliseconds(nap_ms));
        nap_ms = nap_ms < 20 ? nap_ms * 2 : 20;
    }
}

ExitStatus Process::terminate(int grace_ms) {
    close_stdin();
    ExitStatus status;
    if (try_wait(&status)) return status;
    const pid_t target = impl_->group ? -impl_->pid : impl_->pid;
    kill(target, SIGTERM);
    if (!wait_for(grace_ms < 0 ? 0 : grace_ms, &status)) {
        kill(target, SIGKILL);
        wait_for(-1, &status);
    }
    // Members of the group that ignored SIGTERM do not outlive their leader.
    if (impl_->group) kill(-impl_->pid, SIGKILL);
    return status;
}

RunResult run_process(const SpawnOptions& options, const RunLimits& limits) {
    RunResult result;
    std::unique_ptr<Process> p = Process::spawn(options, &result.error);
    if (!p) return result;
    Process::Impl& im = ProcessAccess::impl(*p);

    using clock = std::chrono::steady_clock;
    const auto deadline = clock::now() + std::chrono::milliseconds(limits.timeout_ms < 0 ? 0 : limits.timeout_ms);
    std::size_t fed = 0;
    if (limits.stdin_data.empty()) p->close_stdin();
    bool out_open = true;
    bool err_open = true;
    std::vector<char> buf(64 * 1024);

    auto take = [&](int& fd, std::string& sink, bool& truncated, bool& open) {
        std::size_t got = 0;
        const IoStatus st = detail::read_fd(fd, buf.data(), buf.size(), &got, 0);
        if (st == IoStatus::ok) {
            const std::size_t room = limits.max_output_bytes > sink.size() ? limits.max_output_bytes - sink.size() : 0;
            sink.append(buf.data(), got < room ? got : room);
            if (got > room) truncated = true;
        } else if (st == IoStatus::eof || st == IoStatus::error) {
            open = false;
        }
    };

    while (out_open || err_open) {
        const auto now = clock::now();
        if (limits.timeout_ms >= 0 && now >= deadline) {
            result.timed_out = true;
            break;
        }
        const int wait_ms = limits.timeout_ms < 0
            ? 100
            : static_cast<int>(std::chrono::duration_cast<std::chrono::milliseconds>(deadline - now).count()) + 1;
        pollfd fds[3];
        nfds_t n = 0;
        int out_i = -1, err_i = -1, in_i = -1;
        if (out_open) { fds[n] = {im.out, POLLIN, 0}; out_i = static_cast<int>(n++); }
        if (err_open) { fds[n] = {im.err, POLLIN, 0}; err_i = static_cast<int>(n++); }
        if (im.in >= 0) { fds[n] = {im.in, POLLOUT, 0}; in_i = static_cast<int>(n++); }
        const int r = poll(fds, n, wait_ms < 100 ? wait_ms : 100);
        if (r < 0 && errno != EINTR) break;
        if (r <= 0) continue;
        if (out_i >= 0 && fds[out_i].revents) take(im.out, result.out, result.out_truncated, out_open);
        if (err_i >= 0 && fds[err_i].revents) take(im.err, result.err, result.err_truncated, err_open);
        if (in_i >= 0 && fds[in_i].revents) {
            std::size_t w = 0;
            const IoStatus st = detail::write_fd(im.in, limits.stdin_data.data() + fed,
                                                 limits.stdin_data.size() - fed, &w, 0);
            fed += w;
            if (st == IoStatus::eof || st == IoStatus::error || fed >= limits.stdin_data.size()) p->close_stdin();
        }
    }

    if (result.timed_out) {
        result.status = p->terminate(limits.kill_grace_ms);
    } else {
        const auto now = clock::now();
        const int left = limits.timeout_ms < 0
            ? -1
            : static_cast<int>(std::chrono::duration_cast<std::chrono::milliseconds>(deadline - now).count());
        if (!p->wait_for(left < 0 && limits.timeout_ms >= 0 ? 0 : left, &result.status)) {
            result.timed_out = true;
            result.status = p->terminate(limits.kill_grace_ms);
        }
    }
    return result;
}

#else  // _WIN32

std::string find_executable(const std::string&, const std::string&) { return {}; }

struct Process::Impl {};
struct ProcessAccess {};

Process::Process() : impl_(std::make_unique<Impl>()) {}
Process::~Process() = default;

std::unique_ptr<Process> Process::spawn(const SpawnOptions& options, Error* error) {
    if (error) {
        *error = check_options(options);
        if (!*error) *error = {"process.unsupported", "Starting a child process is not implemented on Windows yet."};
    }
    return nullptr;
}

std::int64_t Process::pid() const { return -1; }
IoStatus Process::read_stdout(char*, std::size_t, std::size_t* got, int) { *got = 0; return IoStatus::error; }
IoStatus Process::read_stderr(char*, std::size_t, std::size_t* got, int) { *got = 0; return IoStatus::error; }
IoStatus Process::write_stdin(const char*, std::size_t, std::size_t* written, int) { *written = 0; return IoStatus::error; }
IoStatus Process::write_stdin_all(const char*, std::size_t, int) { return IoStatus::error; }
void Process::close_stdin() {}
bool Process::try_wait(ExitStatus*) { return true; }
bool Process::wait_for(int, ExitStatus*) { return true; }
ExitStatus Process::terminate(int) { return {}; }

RunResult run_process(const SpawnOptions& options, const RunLimits&) {
    RunResult r;
    Process::spawn(options, &r.error);
    return r;
}

#endif  // _WIN32

}  // namespace caliper_builder::acp
