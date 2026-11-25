#pragma once
// The model runner's commands (design/hosted-runs.md): run.list, run.start,
// run.fork, run.stop and run.logs, and the optional argument `run` of the run
// commands of run_control.h (run.state, run.set_param, run.action, run.pause).
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
// The byte limits of the arguments and of the parameters, which keep the card
// short enough to show them in full: one argument, all arguments, one string
// value, and all parameter names and string values. The handlers refuse a
// start or a fork past a limit before it gets a card.
inline constexpr std::size_t kRunMaxArgBytes = 256;
inline constexpr std::size_t kRunMaxArgsBytes = 1024;
inline constexpr std::size_t kRunMaxParamValueBytes = 256;
inline constexpr std::size_t kRunMaxParamsBytes = 1024;

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

// Parameters as name=value pairs separated by spaces, "lr=0.05 opt=adam",
// from an object of values by name or from the command table's pairs, with
// each name and string value as run_card_text() shows every text: the card's
// parameters are this text, and the runs panel's Parameters field reads it
// back to the same names and values.
std::string run_params_text(const nlohmann::json& params);

// The card text of run.start, run.fork and run.stop (design/hosted-runs.md,
// Review Card), the one source of the agent's card and of the runs panel's:
// the script, its arguments and its parameters of a start; the run and the
// parameters of a fork, with `script` the parent's ("" when unknown); the run
// and the limit of a stop. Every text shows in full: a word of printable
// ASCII with no space, quote, backslash or comma shows bare, and any other
// text in double quotes with JSON's escapes and each control or format
// character as \uXXXX, so that no argument reads as two and no value ends the
// card. A parameter name with '=' and a string value that reads as a number,
// true or false are in quotes too. The handlers refuse what passes the
// limits above, so nothing is cut. "" for any other command.
std::string run_card_text(const CommandInvocation& invocation, const std::string& script);

// A start of run.start, after the handler checked it.
struct RunStartRequest {
    std::string kind = "script";
    std::string model, dictionary, prompt;  // local inference resources and bounded text
    std::string replication_profile, private_output_directory;  // paired inference options, empty when absent
    std::string script_path;     // absolute and canonical, a file in the workspace
    std::string script;          // the same path relative to the workspace, with '/'
    std::vector<std::string> args;
    nlohmann::json params = nlohmann::json::object();
    std::string id;              // empty: the runner chooses one from the script's name
};

// The most interventions of a fork, and the byte limit of all their points
// together, which keeps the card short enough to show each point in full.
inline constexpr std::size_t kRunMaxForkInterventions = 16;
inline constexpr std::size_t kRunMaxInterventionsBytes = 1024;

// A fork of run.fork, after the handler checked it.
struct RunForkRequest {
    std::string parent;
    nlohmann::json params = nlohmann::json::object();
    std::string id;              // empty: the runner chooses one
    // Start from the parent's newest kept checkpoint at or before this step,
    // in place of the checkpoint that the parent saves at its next step.
    std::optional<std::uint64_t> step;
    // The interventions that the fork applies before its first step, each an
    // object as intervene.set takes it (point, op ablate, from_step and
    // until_step), in the order given.
    nlohmann::json interventions = nlohmann::json::array();
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
    // Asks the parent for a checkpoint and starts the fork when it comes, or
    // with `step` starts the fork at once from the parent's newest kept
    // checkpoint at or before that step; *out is the fork, in the state
    // starting. Fails with run.not_found, run.not_managed, run.not_ready (also
    // when no kept checkpoint is at or before `step`), run.bad_id, run.exists,
    // run.start_failed or intervene.bad_range (an until_step at or before
    // the kept checkpoint's step).
    virtual Error fork(const RunForkRequest& request, RunEntry* out) = 0;
    // Sends caliper.stop and ends the process after `kill_after_s` seconds;
    // *out is the run with stopping true. A run that ended already is
    // returned as it is. Fails with run.not_found or run.not_managed.
    virtual Error stop(const std::string& run, double kill_after_s, RunEntry* out) = 0;
    // The last `lines` lines of the run's output.log, oldest first. Fails with
    // run.not_found or run.not_managed.
    virtual Error logs(const std::string& run, int lines, std::vector<std::string>* out) = 0;
    // The run control that acts on `run` for the run commands of
    // run_control.h, and the app's switch to that run at its next frame when it does not show it.
    // Null with *error (run.not_found, run.not_ready) when there is none. The
    // pointer stays valid until the next call of the frame thread.
    virtual RunControl* control_of(const std::string& run, Error* error) = 0;
    // The intervention points that the managed run `run` lists, without
    // switching the app to it, for the check of a fork's interventions;
    // empty when the run lists none or is not there.
    virtual std::vector<InterventionPoint> intervention_points(const std::string& run) {
        (void)run;
        return {};
    }
};

// The handler of run.list, run.start, run.fork, run.stop and run.logs, by the
// invocation's name. `runner` is the session's model runner, or null, which
// answers command.not_available. `workspace` is the session's canonical
// workspace folder. `approved` says whether the person approved this start or
// fork on its card; the handler refuses a start or a fork without it.
CommandOutput runner_command(const CommandContext& context, ModelRunner* runner, const std::string& workspace,
                             bool approved);

// Why a start or a fork would be refused before anything runs: no runner, a
// script outside the workspace, bad parameters or id, the person's script
// as a parent, or a parent that is not there or not running. An empty Error
// when it may run. The session's policy lets a refused request through
// without a card, so that its handler answers with the reason at once, and
// asks the person on a card about every other start or fork, from every
// origin (design/hosted-runs.md, Review Card).
Error runner_precheck(const CommandInvocation& invocation, ModelRunner* runner, const std::string& workspace);

// The handler of the run commands of run_control.h with the argument `run`:
// without it, run_command(context, control); with it, the run control that
// `runner` gives for that run (`user` is the person's script, through
// `control`).
CommandOutput run_command(const CommandContext& context, RunControl* control, ModelRunner* runner);

}  // namespace caliper_builder
