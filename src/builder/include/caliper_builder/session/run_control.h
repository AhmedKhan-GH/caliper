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

// ---- interventions (design/gpu-programs.md, Interventions)
// The error codes of the intervention commands.
inline constexpr const char* kInterveneUnknownPoint = "intervene.unknown_point";  // the model has no such point
inline constexpr const char* kInterveneOpReserved = "intervene.op_reserved";      // steer remains reserved
inline constexpr const char* kInterveneBadRange = "intervene.bad_range";          // until_step not above from_step
inline constexpr const char* kInterveneNotSupported = "intervene.not_supported";  // no capability interventions

// One intervention point, from the script's CONTROLS key interventions.points:
// each module of model.named_modules() by its dotted name, and each head of
// an attention module as "<module>#<h>".
struct InterventionPoint {
    std::string point;
    std::string kind = "module";               // "module" or "head"
    std::string module;                        // the module's dotted name
    std::optional<std::int64_t> head;          // a head's index
    std::string type;                          // the module's class name
    std::optional<std::vector<std::int64_t>> shape;   // the last output's shape; null before the first step
    std::optional<std::string> probe;           // semantic form; kind stays "module"
    std::optional<std::int64_t> layer;          // decoder layer index
    // The outer optional preserves absence. The inner optional preserves JSON null.
    std::optional<std::optional<std::string>> pattern;
};

// One intervention, as intervene.set sends it and as CONTROLS key
// interventions.active gives it back.
struct Intervention {
    std::string point;
    std::string op = "ablate";                 // ablate, add, scale, clamp, or patch
    std::optional<std::int64_t> from_step;     // null: the next step
    std::optional<std::int64_t> until_step;    // null: until cleared
    std::optional<std::int64_t> since_step;    // the step the script applied it at; null while pending
    std::string state = "pending";             // "pending", "active" or "ended"
    nlohmann::json specification = nlohmann::json::object();
};

// Validated wire records prefer a non-null kind, then the legacy op.
inline std::string intervention_op(const nlohmann::json& specification) {
    for (const auto* key : {"kind", "op"}) {
        const auto it = specification.find(key);
        if (it != specification.end() && !it->is_null()) return it->get<std::string>();
    }
    return "ablate";
}

inline nlohmann::json intervention_set_spec(const Intervention& i) {
    auto result = i.specification.is_object() ? i.specification : nlohmann::json::object();
    for (const auto* key : {"since_step", "state", "error", "error_code"}) result.erase(key);
    result["point"] = i.point;
    result["op"] = i.op;
    if (result.contains("kind")) result["kind"] = i.op;
    result["from_step"] = i.from_step ? nlohmann::json(*i.from_step) : nlohmann::json();
    result["until_step"] = i.until_step ? nlohmann::json(*i.until_step) : nlohmann::json();
    return result;
}

class RunControl {
public:
    virtual ~RunControl() = default;
    // The attached script's state, from its last CONTROLS message.
    virtual RunState state() = 0;
    // Sends {op: "set", name, value}. The value applies at the script's next
    // step; until then state() lists it as the parameter's pending value.
    virtual Error set_param(const std::string& name, const nlohmann::json& value) = 0;
    // Text drafts read refusals and session changes without another request.
    virtual std::string param_error(const std::string&) const { return {}; }
    virtual std::string control_session() const { return {}; }
    // Sends {op: "run", name}. The script runs the action at its next step.
    virtual Error run_action(const std::string& name) = 0;
    // Sends {op: "pause"}, {op: "resume"} or {op: "step", count}.
    virtual Error pause(RunPauseOp op, std::int64_t count) = 0;

    virtual CommandOutput interpret(const std::string&, const nlohmann::json&, std::uint32_t) {
        return CommandOutput::fail(Error("interpret.unsupported", "This run has no interpretation transport."));
    }

    // The interventions (design/gpu-programs.md, Intervention Wire). A script
    // that announced the capability `interventions` sends its points and its
    // interventions in CONTROLS, and takes CONTROL {op: "intervene"}. The
    // defaults answer for a run control without interventions: no points, and
    // command.not_available for a change. intervene.points, intervene.list,
    // intervene.set and intervene.clear call these four.
    //
    // Whether the attached script announced `interventions`.
    virtual bool interventions_supported() { return false; }
    // The points of the attached model, from its last CONTROLS message.
    virtual std::vector<InterventionPoint> intervention_points() { return {}; }
    // The interventions, the pending ones first.
    virtual std::vector<Intervention> interventions() { return {}; }
    // Sends {op: "intervene", intervene: {set: {point, op, from_step,
    // until_step}}}. The script applies it at its next step.
    virtual Error intervene_set(const Intervention& intervention) {
        (void)intervention;
        return Error("command.not_available", "This run control takes no interventions.");
    }
    // Sends {op: "intervene", intervene: {clear: point}}, or {clear_all: true}
    // when `point` is empty.
    virtual Error intervene_clear(const std::string& point) {
        (void)point;
        return Error("command.not_available", "This run control takes no interventions.");
    }
};

// The handler of run.state, run.set_param, run.action and run.pause, by the
// invocation's name. `control` is the session's run control, or null, which
// answers command.not_available.
CommandOutput run_command(const CommandContext& context, RunControl* control);

}  // namespace caliper_builder
