#pragma once
// caliper_builder::acp::Process: a child process started from an argument
// vector, never through a shell, with pipes for its standard input, output and
// error. The ACP client runs an agent with it, and the Python loader will run
// its scripts with it, so it knows nothing about either.
//
// - The program is argv[0]. A name without a slash is searched for on the PATH
//   of the child's environment (or of this process when the child's has none);
//   nothing is ever passed to a shell, so arguments need no quoting.
// - The child's environment starts from this process's environment when
//   `inherit_environment` is true, then `env_remove` names are removed and
//   `env` entries are set. Values are never logged, printed or put into an
//   Error: `describe_spawn` shows variable names only. A secret such as an API
//   key travels to a child this way and never in argv, which other users of
//   the machine can read.
// - The child gets a process group of its own by default, so terminate()
//   reaches the processes it starts in turn (an `npx` launcher, for example).
// - Reads and writes take a timeout in milliseconds (0 polls once, a negative
//   value waits without limit) and report an IoStatus. A write to a child that
//   has closed its input reports `eof`; it never raises SIGPIPE.
// - Each pipe may be used by one thread at a time; the three pipes may be used
//   by three threads at once. close_stdin() and the destructor must not run
//   while another thread uses a pipe.
// - The destructor terminates a child that still runs (with the grace period
//   of terminate()) and reaps it.
//
// POSIX only for now: on Windows spawn() fails with `process.unsupported` and
// run_process() returns that error.
//
// Error codes: process.bad_argument, process.not_found, process.spawn_failed,
// process.unsupported.
#include <caliper_builder/error.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace caliper_builder::acp {

// One environment variable given to a child.
struct EnvVar {
    std::string name;
    std::string value;
};

struct SpawnOptions {
    std::vector<std::string> argv;        // argv[0] is the program
    bool inherit_environment = true;      // start from this process's environment
    std::vector<std::string> env_remove;  // names removed from the inherited environment
    std::vector<EnvVar> env;              // set or replaced in the child, applied last
    std::string cwd;                      // the child's working folder; empty keeps this process's
    bool new_process_group = true;        // the child leads a process group of its own
};

// How a child ended. `exited` is true when it returned from main or called
// exit with `code`; otherwise `signal` names the signal that ended it.
struct ExitStatus {
    bool exited = false;
    int code = -1;
    int signal = 0;

    bool success() const { return exited && code == 0; }
    std::string describe() const;  // "exit code 3" or "signal 9"
};

enum class IoStatus {
    ok,       // some bytes were read or written
    timeout,  // nothing happened within the timeout
    eof,      // the other end is closed
    error,    // an I/O error (or the pipe was already closed here)
};

class Process {
public:
    // Starts the child. On failure returns null and fills `error`.
    static std::unique_ptr<Process> spawn(const SpawnOptions& options, Error* error);

    ~Process();
    Process(const Process&) = delete;
    Process& operator=(const Process&) = delete;

    std::int64_t pid() const;

    IoStatus read_stdout(char* buffer, std::size_t capacity, std::size_t* got, int timeout_ms);
    IoStatus read_stderr(char* buffer, std::size_t capacity, std::size_t* got, int timeout_ms);
    IoStatus write_stdin(const char* data, std::size_t size, std::size_t* written, int timeout_ms);
    // Writes all of `size` bytes or stops at the first eof, error or timeout
    // (the timeout applies to each wait for the pipe to drain).
    IoStatus write_stdin_all(const char* data, std::size_t size, int timeout_ms);
    void close_stdin();

    // True when the child has ended; fills `status`. Never blocks.
    bool try_wait(ExitStatus* status);
    // Waits up to `timeout_ms` (negative: without limit) for the child to end.
    bool wait_for(int timeout_ms, ExitStatus* status);
    // Closes stdin, sends SIGTERM to the child (to its group when it has one),
    // waits up to `grace_ms`, then sends SIGKILL, and reaps it. Safe to call on
    // a child that has already ended.
    ExitStatus terminate(int grace_ms);

    struct Impl;

private:
    Process();
    std::unique_ptr<Impl> impl_;
    friend struct ProcessAccess;
};

// What run_process() collects.
struct RunLimits {
    int timeout_ms = 60000;                      // then the child is terminated
    std::size_t max_output_bytes = 16u << 20;    // per stream; the rest is read and dropped
    int kill_grace_ms = 2000;
    std::string stdin_data;                      // written to the child's input, which is then closed
};

struct RunResult {
    Error error;                 // set when the child could not be started
    ExitStatus status;
    std::string out;
    std::string err;
    bool timed_out = false;
    bool out_truncated = false;
    bool err_truncated = false;
};

// Runs a child to completion, feeding it `stdin_data` and reading both of its
// output streams at once, so a child that writes a lot never deadlocks on a
// full pipe.
RunResult run_process(const SpawnOptions& options, const RunLimits& limits);

// Finds `name` on `path_list` (a PATH value). A name with a slash is returned
// as it is when it is executable. Returns an empty string when nothing is found.
std::string find_executable(const std::string& name, const std::string& path_list);

// A one-line description of a spawn for logs: the arguments, the working
// folder and the names (never the values) of the variables set or removed.
std::string describe_spawn(const SpawnOptions& options);

}  // namespace caliper_builder::acp
