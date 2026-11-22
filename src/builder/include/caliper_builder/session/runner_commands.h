#pragma once
// The model runner's commands (design/hosted-runs.md): run.list, run.start,
// run.fork, run.stop and run.logs, and the optional argument `run` of the run
// commands of M1 (run_control.h).
//
// The app implements ModelRunner in its host component (src/host/runner/
// model_runner.h): it starts a script of the workspace as a Python child
// process, keeps its own viewer connection to each managed run and attaches
// the app to a new run. The app gives its runner to
// Session::set_model_runner(); a session without one (the stand-in host of the
// tests, caliper-mcp without the app) answers the five commands with
// command.not_available.
//
// No function here waits for a child process: run.start and run.fork answer
// when the process started, in the state starting, and the runner reports
// the later states through run.list and run events of change runner.
//
// Threads: the commands are frame_thread commands, as the run commands are;
// a ModelRunner is safe to call from any thread. Nothing here throws.
#include <caliper_builder/commands/registry.h>
#include <caliper_builder/error.h>
#include <caliper_builder/session/run_control.h>

#include <nlohmann/json.hpp>

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace caliper_builder {

// The id of the person's own attached script in run.list and in the argument
// `run`: a script that the runner did not start.
inline constexpr const char* kUserRunId = "user";

// The error codes of the runner commands, besides command.not_available.
inline constexpr const char* kRunScriptRefused = "run.script_refused";  // outside the workspace, or not a file
inline constexpr const char* kRunBadId = "run.bad_id";                  // not [a-z0-9-]{1,40}
inline constexpr const char* kRunExists = "run.exists";                 // a run of this id exists in the workspace
inline constexpr const char* kRunNotFound = "run.not_found";            // no run of this id
inline constexpr const char* kRunNotReady = "run.not_ready";            // no connection yet, or the run ended
inline constexpr const char* kRunNotManaged = "run.not_managed";        // `user` where a managed run is needed
inline constexpr const char* kRunStartFailed = "run.start_failed";      // the interpreter or the process failed

// The most arguments and parameters of a start, and the longest run id.
inline constexpr std::size_t kRunMaxArgs = 64;
inline constexpr std::size_t kRunMaxParams = 64;
inline constexpr std::size_t kRunMaxIdBytes = 40;

// One run of run.list (the schema run_entry of the command table).
struct RunEntry {
    std::string id;
    bool managed = true;
    std::string state = "starting";         // starting, running, paused, exited, failed
    std::string script;                     // relative to the workspace for a managed run
    std::vector<std::string> args;
    nlohmann::json params = nlohmann::json::object();
    std::optional<std::string> parent;      // the run a fork started from
    std::optional<std::uint64_t> step;      // the step of the latest checkpoint
    std::optional<std::int64_t> exit_code;  // a signal as its negative number
    bool attached = false;                  // the app shows this run now
    bool stopping = false;                  // run.stop asked it to stop
    std::optional<std::string> error;       // why it failed
};

// A RunEntry in the command table's form.
nlohmann::json run_entry_json(const RunEntry& entry);

// Whether `id` is a valid run id: 1 to 40 bytes of a-z, 0-9 and '-'.
bool valid_run_id(const std::string& id);

// A start of run.start, after the handler checked it.
struct RunStartRequest {
    std::string script_path;     // absolute and canonical, a file in the workspace
    std::string script;          // the same path relative to the workspace, with '/'
    std::vector<std::string> args;
    nlohmann::json params = nlohmann::json::object();
    std::string id;              // empty: the runner chooses one from the script's name
};

// A fork of run.fork, after the handler checked it.
struct RunForkRequest {
    std::string parent;
    nlohmann::json params = nlohmann::json::object();
    std::string id;              // empty: the runner chooses one
};

class ModelRunner {
public:
    virtual ~ModelRunner() = default;
    // The managed runs in the order of their start, then the person's own
    // attached script (id `user`, managed false) when there is one.
    virtual std::vector<RunEntry> list() = 0;
    // Starts the script as a managed run; *out is the new run, in the state
    // starting. Fails with run.bad_id, run.exists or run.start_failed.
    virtual Error start(const RunStartRequest& request, RunEntry* out) = 0;
    // Asks the parent for a checkpoint and starts the fork when it comes;
    // *out is the fork, in the state starting. Fails with run.not_found,
    // run.not_managed, run.not_ready, run.bad_id or run.exists.
    virtual Error fork(const RunForkRequest& request, RunEntry* out) = 0;
    // Sends caliper.stop and ends the process after `kill_after_s` seconds;
    // *out is the run with stopping true. A run that ended already is
    // returned as it is. Fails with run.not_found or run.not_managed.
    virtual Error stop(const std::string& run, double kill_after_s, RunEntry* out) = 0;
    // The last `lines` lines of the run's output.log, oldest first. Fails with
    // run.not_found or run.not_managed.
    virtual Error logs(const std::string& run, int lines, std::vector<std::string>* out) = 0;
    // The run control that acts on `run` for the run commands of M1, and the
    // app's switch to that run at its next frame when it does not show it.
    // Null with *error (run.not_found, run.not_ready) when there is none. The
    // pointer stays valid until the next call of the frame thread.
    virtual RunControl* control_of(const std::string& run, Error* error) = 0;
};

// The handler of run.list, run.start, run.fork, run.stop and run.logs, by the
// invocation's name. `runner` is the session's model runner, or null, which
// answers command.not_available. `workspace` is the session's canonical
// workspace folder. `approved` says whether the person approved this start or
// fork on its card, or the origin is the person's own (ui, builder); the
// handler refuses a start or a fork without it.
CommandOutput runner_command(const CommandContext& context, ModelRunner* runner, const std::string& workspace,
                             bool approved);

// The handler of the run commands of M1 with the argument `run`: without it,
// run_command(context, control); with it, the run control that `runner`
// gives for that run (`user` is the person's script, through `control`).
CommandOutput run_command(const CommandContext& context, RunControl* control, ModelRunner* runner);

}  // namespace caliper_builder
