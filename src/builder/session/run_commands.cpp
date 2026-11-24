// The run commands (run_control.h, design/instruments.md, Run Commands).
//
// run.state reads the attached script's state; run.set_param, run.action and
// run.pause check the request against that state and hand it to the app's
// RunControl, which sends it as a CONTROL message. The checks here are those
// that need no answer from the script: a script is attached, the connection
// takes control, the parameter or the action exists, the value has the
// parameter's type, and a pause or a step comes after the first step. What
// the script then does comes back as a run event, never as this result.
#include <caliper_builder/session/run_control.h>

#include <cmath>
#include <exception>
#include <utility>

namespace caliper_builder {

namespace {

using nlohmann::json;

Error fail(std::string code, std::string message, std::string where = {}) {
    return Error(std::move(code), std::move(message), std::move(where));
}

std::string str_arg(const json& args, const char* key) {
    const auto it = args.find(key);
    return it != args.end() && it->is_string() ? it->get<std::string>() : std::string();
}

json param_json(const RunParam& p) {
    json out{{"name", p.name}, {"type", p.type}, {"value", p.value}, {"pending", p.pending}, {"log", p.log},
             {"help", p.help}};
    if (p.min) out["min"] = *p.min;
    if (p.max) out["max"] = *p.max;
    if (!p.choices.empty()) out["choices"] = p.choices;
    return out;
}

// The refusal of a request that needs an attached script that takes control,
// or an empty Error.
Error check_control(const RunState& state) {
    if (!state.attached)
        return fail(kRunNotAttached,
                    "No script is attached to Caliper; start a script with caliper.watch or caliper.serve and attach "
                    "the app to its stream first.");
    if (!state.control)
        return fail(kRunControlNotAllowed,
                    state.why_not.empty() ? "The stream of " + state.script + " takes no control." : state.why_not);
    return {};
}

const RunParam* find_param(const RunState& state, const std::string& name) {
    for (const auto& p : state.params)
        if (p.name == name) return &p;
    return nullptr;
}

const RunAction* find_action(const RunState& state, const std::string& name) {
    for (const auto& a : state.actions)
        if (a.name == name) return &a;
    return nullptr;
}

std::string names_of_params(const RunState& state) {
    std::string out;
    for (const auto& p : state.params) out += (out.empty() ? "" : ", ") + p.name;
    return out.empty() ? "none" : out;
}

std::string names_of_actions(const RunState& state) {
    std::string out;
    for (const auto& a : state.actions) out += (out.empty() ? "" : ", ") + a.name;
    return out.empty() ? "none" : out;
}

// Whether `value` has the type of `param`: a number for a number, a whole
// number for an integer, true or false for a boolean, one of the choices for a
// choice. The range is not checked: the script clamps to it.
Error check_value(const RunParam& param, const json& value) {
    const std::string where = "args.value";
    if (param.type == "number") {
        if (!value.is_number() || !std::isfinite(value.get<double>()))
            return fail(kRunBadValue, "The parameter " + param.name + " takes a finite number.", where);
        return {};
    }
    if (param.type == "integer") {
        if (value.is_number_integer()) return {};
        if (value.is_number_float()) {
            const double v = value.get<double>();
            if (std::isfinite(v) && std::floor(v) == v && std::fabs(v) < 9.007199254740992e15) return {};
        }
        return fail(kRunBadValue, "The parameter " + param.name + " takes a whole number.", where);
    }
    if (param.type == "boolean") {
        if (!value.is_boolean()) return fail(kRunBadValue, "The parameter " + param.name + " takes true or false.", where);
        return {};
    }
    if (param.type == "text") {
        if (!value.is_string() || value.get_ref<const std::string&>().size() > 1024 ||
            value.get_ref<const std::string&>().find('\0') != std::string::npos)
            return fail(kRunBadValue, "The text parameter accepts at most 1024 UTF-8 bytes.", where);
        try { (void)value.dump(); } catch (...) { return fail(kRunBadValue, "The text is not valid UTF-8.", where); }
        return {};
    }
    if (param.type == "choice") {
        if (value.is_string())
            for (const auto& c : param.choices)
                if (c == value.get<std::string>()) return {};
        std::string list;
        for (const auto& c : param.choices) list += (list.empty() ? "" : ", ") + c;
        return fail(kRunBadValue, "The parameter " + param.name + " takes one of its choices: " + list + ".", where);
    }
    // A type this Caliper does not know: the script decides.
    return {};
}

// An integer parameter gets an integer, so that the CONTROL message carries
// 3 and not 3.0.
json normalized_value(const RunParam& param, const json& value) {
    if (param.type == "integer" && value.is_number_float())
        return static_cast<std::int64_t>(value.get<double>());
    return value;
}

CommandOutput set_param(const json& args, RunControl& control) {
    const RunState state = control.state();
    if (Error e = check_control(state)) return CommandOutput::fail(std::move(e));
    const std::string name = str_arg(args, "name");
    const RunParam* param = find_param(state, name);
    if (!param)
        return CommandOutput::fail(fail(kRunUnknownParam,
                                        "The script " + state.script + " has no parameter " + name +
                                            "; its parameters are " + names_of_params(state) + ".",
                                        "args.name"));
    const json value = args.contains("value") ? args["value"] : json();
    if (Error e = check_value(*param, value)) return CommandOutput::fail(std::move(e));
    const json sent = normalized_value(*param, value);
    if (Error e = control.set_param(name, sent)) return CommandOutput::fail(std::move(e));
    return CommandOutput::ok(json{{"name", name}, {"value", sent}, {"pending", true}});
}

CommandOutput run_action(const json& args, RunControl& control) {
    const RunState state = control.state();
    if (Error e = check_control(state)) return CommandOutput::fail(std::move(e));
    const std::string name = str_arg(args, "name");
    if (!find_action(state, name))
        return CommandOutput::fail(fail(kRunUnknownAction,
                                        "The script " + state.script + " has no action " + name +
                                            "; its actions are " + names_of_actions(state) + ".",
                                        "args.name"));
    if (Error e = control.run_action(name)) return CommandOutput::fail(std::move(e));
    return CommandOutput::ok(json{{"name", name}, {"queued", true}});
}

CommandOutput pause(const json& args, RunControl& control) {
    const RunState state = control.state();
    if (Error e = check_control(state)) return CommandOutput::fail(std::move(e));
    const std::string op_name = str_arg(args, "op");
    RunPauseOp op = RunPauseOp::pause;
    if (op_name == "resume") {
        op = RunPauseOp::resume;
    } else if (op_name == "step") {
        op = RunPauseOp::step;
    } else if (op_name != "pause") {
        return CommandOutput::fail(fail("command.invalid_args", "op is pause, resume or step.", "args.op"));
    }
    std::int64_t count = 1;
    if (op == RunPauseOp::step && args.contains("count") && args["count"].is_number_integer())
        count = args["count"].get<std::int64_t>();
    if (op != RunPauseOp::resume && (state.state == "none" || !state.step))
        return CommandOutput::fail(fail(kRunNoStep,
                                        "The script " + state.script +
                                            " has not made its first step, and a run pauses only at a step; try again "
                                            "after its first step.",
                                        "args.op"));
    if (Error e = control.pause(op, count)) return CommandOutput::fail(std::move(e));
    return CommandOutput::ok(json{{"op", op_name}, {"state", state.state}});
}

}  // namespace

json run_state_json(const RunState& state) {
    json out{{"attached", state.attached}, {"control", state.control}, {"state", state.state}};
    if (!state.script.empty()) out["script"] = state.script;
    if (!state.control && !state.why_not.empty()) out["why_not"] = state.why_not;
    out["step"] = state.step ? json(*state.step) : json();
    out["steps_left"] = state.steps_left ? json(*state.steps_left) : json();
    json params = json::array();
    for (const auto& p : state.params) params.push_back(param_json(p));
    out["params"] = std::move(params);
    json actions = json::array();
    for (const auto& a : state.actions)
        actions.push_back(json{{"name", a.name}, {"help", a.help}, {"running", a.running}, {"queued", a.queued}});
    out["actions"] = std::move(actions);
    return out;
}

CommandOutput run_command(const CommandContext& context, RunControl* control) {
    const std::string& name = context.invocation.name;
    if (!control)
        return CommandOutput::fail(fail("command.not_available",
                                        "Run commands act on the script that the Caliper app is attached to, and this "
                                        "session has no app with a script.",
                                        name));
    try {
        const json& args = context.invocation.args;
        if (name == "run.state") return CommandOutput::ok(run_state_json(control->state()));
        if (name == "run.set_param") return set_param(args, *control);
        if (name == "run.action") return run_action(args, *control);
        if (name == "run.pause") return pause(args, *control);
        return CommandOutput::fail(fail("command.unknown", "This is not a run command.", name));
    } catch (const std::exception& e) {
        return CommandOutput::fail(fail(kRunRefused, std::string("The request could not be sent: ") + e.what(), name));
    }
}

}  // namespace caliper_builder
