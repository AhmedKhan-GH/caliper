#pragma once
// A child process of the model runner (design/hosted-runs.md, Child
// Processes): a program with its arguments, its whole environment, a working
// folder and one output file that takes its standard output and its standard
// error. The runner reads the exit without a wait and ends the process by its
// own id, never by a name.
//
// POSIX (child_process_posix.cpp): posix_spawn, the child in a process group
// of its own so that a terminal's signal to the app does not reach it, and
// the default handlers of the common signals. Windows
// (child_process_win.cpp): CreateProcessW in a job object that ends the child
// when the app's last handle of the job closes, also in a crash. The Windows
// code compiles and is not tested.
//
// Threads: one ChildProcess is used from one thread at a time. Nothing here
// throws.
#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace caliper_runner {

// What to start.
struct ChildSpec {
    std::string program;                                    // the executable, an absolute path
    std::vector<std::string> args;                          // the arguments after the program
    std::vector<std::pair<std::string, std::string>> env;   // the child's whole environment
    std::string cwd;                                        // the working folder; "" keeps the app's
    std::string output;                                     // the file of stdout and stderr, appended
};

class ChildProcess {
public:
    // Does not end the process: the owner ends it first (kill(), then wait()).
    virtual ~ChildProcess() = default;
    // The process id.
    virtual std::int64_t pid() const = 0;
    // True once the process ended, with its exit code: a signal as its
    // negative number. Never waits. After the first true, the process id is
    // no longer used, so no later call can reach another process.
    virtual bool poll(std::int64_t* exit_code) = 0;
    // Asks the process to end: SIGTERM on POSIX, the same as kill() on
    // Windows. Nothing once it ended.
    virtual void terminate() = 0;
    // Ends the process at once: SIGKILL, or TerminateProcess. Nothing once it
    // ended.
    virtual void kill() = 0;
    // Waits at most `timeout_ms` for the end; true with the exit code when it
    // ended.
    virtual bool wait(int timeout_ms, std::int64_t* exit_code) = 0;
};

// Starts the child; null with *error ("The program ... could not start: ...")
// when the program, the folder or the output file fails.
std::unique_ptr<ChildProcess> start_child(const ChildSpec& spec, std::string* error);

// This process's environment as name and value pairs, in its order.
std::vector<std::pair<std::string, std::string>> current_environment();

}  // namespace caliper_runner
