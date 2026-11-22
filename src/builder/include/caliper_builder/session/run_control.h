#pragma once
// The run control (design/instruments.md, Run Control): what the commands
// run.state, run.set_param, run.action and run.pause call to act on the
// script that the app is attached to.
//
// The app implements RunControl over its own viewer connection: it sends
// each request as a protocol 3.1 CONTROL message (include/caliper/live_net.h,
// docs/reference/run-control-messages.md) through the attachment's
// caliper_live::RunControls, and reads the run's state from the CONTROLS
// message the script last sent. The app gives its RunControl to
// Session::set_run_control(); a session without one (the stand-in host of the
// tests, caliper-mcp without the app) answers the four commands with
// command.not_available.
//
// The app may control the script only when its viewer connection allows it:
// the stream announced the capability `controls`, WELCOME carried control
// true and the session lists `controls` (live_net_source.mm, access_locked).
// When it does not, every request fails with run.control_not_allowed and the
// reason the connection gives. No request waits for the script: a value or an
// action reaches the script at its next step, and what it did comes back as
// a run event (events.h).
//
// Threads: every function runs on the frame thread, where the app's
// RunControls lives; the four commands are frame_thread commands. Nothing
// here throws.
//
// This header is the interface of the run control: the commands are in
// src/builder/session/run_commands.cpp, and the app has its own
// implementation.
#include <caliper_builder/commands/registry.h>
#include <caliper_builder/error.h>

#include <nlohmann/json.hpp>

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace caliper_builder {

// One parameter of the script, as its CONTROLS message gives it.
struct RunParam {
    std::string name;
    std::string type;                 // "number", "integer", "boolean" or "choice"
    nlohmann::json value;             // the value the script uses now
    nlohmann::json pending;           // a value sent and not applied yet; null when none
    std::optional<double> min, max;
    bool log = false;
    std::vector<std::string> choices; // a choice parameter's values
    std::string help;
};

// One action of the script.
struct RunAction {
    std::string name;
    std::string help;
    bool running = false;
    bool queued = false;
};

// What run.state returns, as a value.
struct RunState {
    bool attached = false;            // a script is attached to the app
    std::string script;               // its name; empty when none
    bool control = false;             // the app may send CONTROL
    std::string why_not;              // when !control: the reason, one sentence
    std::string state = "none";       // "none", "running", "pausing", "paused" or "stepping"
    std::optional<std::int64_t> step;        // null before the script's first step
    std::optional<std::int64_t> steps_left;  // while stepping
    std::vector<RunParam> params;
    std::vector<RunAction> actions;
};

// The result of run.state in the command table's form.
nlohmann::json run_state_json(const RunState& state);

// The requests of run.pause.
enum class RunPauseOp { pause, resume, step };

// The error codes of the run commands, besides command.not_available.
inline constexpr const char* kRunNotAttached = "run.not_attached";            // no script is attached
inline constexpr const char* kRunControlNotAllowed = "run.control_not_allowed";  // the connection takes no control
inline constexpr const char* kRunUnknownParam = "run.unknown_param";          // the script has no such parameter
inline constexpr const char* kRunUnknownAction = "run.unknown_action";        // the script has no such action
inline constexpr const char* kRunBadValue = "run.bad_value";                  // wrong type, or not a choice
inline constexpr const char* kRunNoStep = "run.no_step";                      // pause or step before the first step
inline constexpr const char* kRunRefused = "run.refused";                     // the script or the hub refused it

class RunControl {
public:
    virtual ~RunControl() = default;
    // The attached script's state, from its last CONTROLS message.
    virtual RunState state() = 0;
    // Sends {op: "set", name, value}. The value applies at the script's next
    // step; until then state() lists it as the parameter's pending value.
    virtual Error set_param(const std::string& name, const nlohmann::json& value) = 0;
    // Sends {op: "run", name}. The script runs the action at its next step.
    virtual Error run_action(const std::string& name) = 0;
    // Sends {op: "pause"}, {op: "resume"} or {op: "step", count}.
    virtual Error pause(RunPauseOp op, std::int64_t count) = 0;
};

// The handler of run.state, run.set_param, run.action and run.pause, by the
// invocation's name. `control` is the session's run control, or null, which
// answers command.not_available.
CommandOutput run_command(const CommandContext& context, RunControl* control);

}  // namespace caliper_builder
