// The intervention commands (intervene_commands.h, design/gpu-programs.md,
// Intervention Commands).
//
// intervene.points and intervene.list read what the attached script last sent
// in CONTROLS. intervene.set and intervene.clear check the request against
// that state and hand it to the run control, which sends it as CONTROL op
// intervene; the script applies it at its next step, and what it did comes
// back as a run event of change intervene, never as this result. The checks
// here need no answer from the script: a script is attached, it announced
// `interventions`, the connection takes control for a change, the op is
// built, the point is one the script listed (when it listed any), and the
// step range is not empty or past. The script checks again.
#include <caliper_builder/session/intervene_commands.h>

#include <caliper_builder/commands/command_table.h>
#include <caliper_builder/commands/schema.h>
#include "../../../include/caliper/live_net.h"
#include <set>
#include <cmath>
#include <algorithm>
#include <cstdint>
#include <exception>
#include <optional>
#include <string>
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

std::optional<std::int64_t> step_arg(const json& args, const char* key) {
    const auto it = args.find(key);
    if (it == args.end() || !it->is_number_integer()) return std::nullopt;
    return it->get<std::int64_t>();
}

// Whether `name` matches the glob `pattern`: `*` matches any run of
// characters, dots included, and `?` one character.
bool glob_match(const std::string& pattern, const std::string& name) {
    std::size_t p = 0, n = 0, star = std::string::npos, mark = 0;
    while (n < name.size()) {
        if (p < pattern.size() && (pattern[p] == '?' || pattern[p] == name[n])) {
            ++p;
            ++n;
        } else if (p < pattern.size() && pattern[p] == '*') {
            star = p++;
            mark = n;
        } else if (star != std::string::npos) {
            p = star + 1;
            n = ++mark;
        } else {
            return false;
        }
    }
    while (p < pattern.size() && pattern[p] == '*') ++p;
    return p == pattern.size();
}

json opt_json(const std::optional<std::int64_t>& v) { return v ? json(*v) : json(); }

json point_json(const InterventionPoint& p) {
    json shape = json();
    if (p.shape) shape = *p.shape;
    json out{{"point", p.point}, {"kind", p.kind},       {"module", p.module},
             {"head", opt_json(p.head)}, {"type", p.type}, {"shape", std::move(shape)}};
    if (p.probe) out["probe"] = *p.probe;
    if (p.layer) out["layer"] = *p.layer;
    if (p.pattern) out["pattern"] = *p.pattern ? json(**p.pattern) : json();
    return out;
}

json intervention_json(const Intervention& i) {
    json result = i.specification.is_object() ? i.specification : json::object();
    result.update(json{{"point", i.point}, {"op", i.op}, {"from_step", opt_json(i.from_step)},
                       {"until_step", opt_json(i.until_step)}, {"since_step", opt_json(i.since_step)}, {"state", i.state}});
    if (result.contains("kind")) result["kind"] = i.op;
    return result;
}

// The refusal of any intervention command without an attached script that
// announced `interventions`; with `change`, also without control. Empty when
// it may go on.
Error check(const RunState& state, RunControl& control, bool change) {
    if (!state.attached)
        return fail(kRunNotAttached,
                    "No script is attached to Caliper. Start a script with caliper.watch(model) and "
                    "attach the app to its stream first.");
    if (!control.interventions_supported())
        return fail(kInterveneNotSupported,
                    "The script " + state.script +
                        " takes no interventions. It announces them under caliper.watch(model).");
    if (change && !state.control)
        return fail(kRunControlNotAllowed,
                    state.why_not.empty() ? "The stream of " + state.script + " takes no control." : state.why_not);
    return {};
}

// The refusal of a point that the script did not list, or an empty Error.
// Without a list (the script sent none yet), the script decides.
Error check_point(const std::vector<InterventionPoint>& points, const std::string& point, const std::string& where) {
    if (points.empty()) return {};
    for (const auto& p : points)
        if (p.point == point) return {};
    return fail(kInterveneUnknownPoint,
                "The model has no point " + point + "; intervene.points lists the points.", where);
}

CommandOutput points(const json& args, RunControl& control) {
    const RunState state = control.state();
    if (Error e = check(state, control, false)) return CommandOutput::fail(std::move(e));
    const std::string match = str_arg(args, "match");
    json out = json::array();
    for (const auto& p : control.intervention_points())
        if (match.empty() || glob_match(match, p.point)) out.push_back(point_json(p));
    return CommandOutput::ok(json{{"points", std::move(out)}, {"step", opt_json(state.step)}});
}

CommandOutput list(RunControl& control) {
    const RunState state = control.state();
    if (Error e = check(state, control, false)) return CommandOutput::fail(std::move(e));
    json out = json::array();
    for (const auto& i : control.interventions()) out.push_back(intervention_json(i));
    return CommandOutput::ok(json{{"interventions", std::move(out)}});
}

CommandOutput set(const json& args, RunControl& control) {
    const RunState state = control.state();
    if (Error e = check(state, control, true)) return CommandOutput::fail(std::move(e));
    Intervention i;
    i.point = str_arg(args, "point");
    i.specification = args;
    i.from_step = step_arg(args, "from_step");
    i.until_step = step_arg(args, "until_step");
    if (Error e = check_intervention_spec(args, "args")) return CommandOutput::fail(std::move(e));
    i.op = intervention_op(args);
    if (Error e = check_point(control.intervention_points(), i.point, "args.point"))
        return CommandOutput::fail(std::move(e));
    if (i.from_step && i.until_step && *i.until_step <= *i.from_step)
        return CommandOutput::fail(fail(kInterveneBadRange,
                                        "until_step " + std::to_string(*i.until_step) + " is not above from_step " +
                                            std::to_string(*i.from_step) + ".",
                                        "args.until_step"));
    // The script's next step is past the step it last reported.
    if (i.until_step && state.step && *i.until_step <= *state.step + 1)
        return CommandOutput::fail(fail(kInterveneBadRange,
                                        "until_step " + std::to_string(*i.until_step) +
                                            " is in the past: the script is at step " + std::to_string(*state.step) +
                                            ".",
                                        "args.until_step"));
    if (Error e = control.intervene_set(i)) return CommandOutput::fail(std::move(e));
    return CommandOutput::ok(json{{"point", i.point}, {"op", i.op}, {"pending", true}});
}

CommandOutput clear(const json& args, RunControl& control) {
    const RunState state = control.state();
    if (Error e = check(state, control, true)) return CommandOutput::fail(std::move(e));
    const auto all_it = args.find("all");
    const bool all = all_it != args.end() && all_it->is_boolean() && all_it->get<bool>();
    const std::string point = str_arg(args, "point");
    if (all == !point.empty())
        return CommandOutput::fail(
            fail("command.invalid_args", "intervene.clear takes point or all true, and not both.", "args"));
    if (!all)
        if (Error e = check_point(control.intervention_points(), point, "args.point"))
            return CommandOutput::fail(std::move(e));
    json cleared = json::array();
    for (const auto& i : control.interventions())
        if ((all || i.point == point) && i.state != "ended" &&
            std::find(cleared.begin(), cleared.end(), json(i.point)) == cleared.end())
            cleared.push_back(i.point);
    if (Error e = control.intervene_clear(all ? std::string() : point)) return CommandOutput::fail(std::move(e));
    return CommandOutput::ok(json{{"cleared", std::move(cleared)}, {"pending", true}});
}

}  // namespace

Error check_intervention_spec(const nlohmann::json& spec, const std::string& where) {
    const auto* descriptor = find_command_descriptor("intervene.set");
    if (auto error = validate_command_value(descriptor->args, spec, where)) return error;
    const std::string op = str_arg(spec,"op"), kind = str_arg(spec,"kind");
    if ((!op.empty() && !kind.empty() && op != kind) || (op.empty() && kind.empty()))
        return fail("command.invalid_args", "Supply matching op or kind discriminators.", where);
    const std::string name = kind.empty() ? op : kind;
    if (name == "steer") return fail(kInterveneOpReserved, "Steer remains reserved.", where);
    std::set<std::string> fields = {"point","op","kind","from_step","until_step"};
    std::vector<std::string> required;
    if (name == "add") { fields.insert({"vector","coefficient","positions"}); required = {"vector","coefficient"}; }
    if (name == "scale") { fields.insert({"feature","multiplier","positions","dictionary"}); required = {"feature","multiplier"}; }
    if (name == "clamp") { fields.insert({"feature","value","positions","dictionary"}); required = {"feature","value"}; }
    if (name == "patch") { fields.insert({"source_prompt","source_position","positions"}); required = {"source_prompt"}; }
    for (const auto& key : required)
        if (!spec.contains(key)) return fail("command.invalid_args", "The operation requires " + key + ".", where + "." + key);
    for (auto it = spec.begin(); it != spec.end(); ++it)
        if (!fields.count(it.key())) return fail("command.invalid_args", "The operation does not accept " + it.key() + ".", where + "." + it.key());
    for (const auto* key : {"from_step","until_step","feature","source_position"})
        if (spec.contains(key) && !spec[key].is_null() && !spec[key].is_number_integer())
            return fail("command.invalid_args", "The field requires an integer.", where + "." + key);
    const auto finite = [](const json& value) {
        return value.is_number() && std::isfinite(value.get<double>()) &&
            (!value.is_number_integer() || std::abs(value.get<double>()) <= 9007199254740991.);
    };
    for (const auto* key : {"coefficient","multiplier","value"})
        if (spec.contains(key) && !finite(spec[key]))
            return fail("command.invalid_args", "The value must be finite and representable.", where + "." + key);
    if (spec.contains("vector"))
        for (const auto& value : spec["vector"])
            if (!finite(value)) return fail("command.invalid_args", "Vector values must be finite and representable.", where + ".vector");
    if (spec.contains("positions")) {
        std::set<int> seen;
        for (const auto& value : spec["positions"])
            if (!value.is_number_integer() || !seen.insert(value.get<int>()).second) return fail("command.invalid_args", "Positions must be unique.", where + ".positions");
    }
    for (const auto* key : {"point","dictionary","source_prompt"}) {
        if (!spec.contains(key)) continue;
        const auto& value = spec[key].get_ref<const std::string&>();
        const std::size_t limit = std::string(key) == "source_prompt" ? 4096 : 255;
        if (value.size() > limit || value.find('\0') != std::string::npos)
            return fail("command.invalid_args", "Text exceeds its UTF-8 byte limit or contains a zero character.", where + "." + key);
    }
    try {
        if (spec.dump().size() > 262144) return fail("command.invalid_args", "The specification exceeds 256 KiB.", where);
    } catch (const std::exception&) { return fail("command.invalid_args", "The specification must contain valid UTF-8.", where); }
    const auto start = step_arg(spec,"from_step"), end = step_arg(spec,"until_step");
    if (start && end && *end <= *start) return fail(kInterveneBadRange,"until_step must exceed from_step.",where);
    return {};
}

CommandOutput interpretation_command(const CommandContext& context, RunControl* control) {
    if (!control) return CommandOutput::fail(fail("interpret.unsupported", "This session has no run transport."));
    const auto state = control->state();
    if (!state.attached) return CommandOutput::fail(fail(kRunNotAttached, "No run is attached."));
    const auto& name = context.invocation.name;
    const bool change = name == "trace.run" || name == "trace.validate" || name == "patch.sweep";
    if (change && !state.control) return CommandOutput::fail(fail(kRunControlNotAllowed, state.why_not));
    json params = context.invocation.args;
    const auto request = params.value("request_id", std::uint32_t(0));
    params.erase("request_id");
    if (request && !params.empty())
        return CommandOutput::fail(fail("command.invalid_args", "A reply query accepts only request_id."));
    if (!request && (name == "feature.show" || name == "feature.examples")) {
        const bool node = params.contains("node"), layer = params.contains("layer"), feature = params.contains("feature");
        if (!((node && !layer && !feature) || (!node && layer && feature)))
            return CommandOutput::fail(fail("command.invalid_args", "Supply node or both layer and feature."));
    }
    if (!request && name == "trace.run" && !params.contains("prompt"))
        return CommandOutput::fail(fail("command.invalid_args", "A trace requires a prompt."));
    if (!request && name == "patch.sweep") {
        if (!params.contains("clean_prompt") || !params.contains("corrupted_prompt") || !params.contains("metric"))
            return CommandOutput::fail(fail("command.invalid_args", "A sweep requires both prompts and its metric."));
        if (params["metric"]["positive_token"] == params["metric"]["negative_token"])
            return CommandOutput::fail(fail("command.invalid_args", "The metric token IDs must differ."));
    }
    for (const auto* key : {"prompt", "clean_prompt", "corrupted_prompt"}) {
        const auto it = params.find(key);
        if (it == params.end()) continue;
        if (!it->is_string())
            return CommandOutput::fail(fail("command.invalid_args", "The prompt must be text.", std::string("args.") + key));
        const auto& text = it->get_ref<const std::string&>();
        if (text.empty() || text.size() > CALIPER_NET_INTERPRET_STRING_MAX || text.find('\0') != std::string::npos)
            return CommandOutput::fail(fail("command.invalid_args",
                "Use a nonempty UTF-8 prompt without NUL and at most " +
                    std::to_string(CALIPER_NET_INTERPRET_STRING_MAX) + " bytes.", std::string("args.") + key));
    }
    try {
        if (params.dump().size() > CALIPER_NET_INTERPRET_MAX_BYTES)
            return CommandOutput::fail(fail("command.invalid_args", "Reduce the interpretation request to 256 KiB."));
    } catch (const json::exception&) {
        return CommandOutput::fail(fail("command.invalid_args", "The interpretation request must contain valid UTF-8."));
    }
    auto output = control->interpret(name, params, request);
    if (request) output.no_effect = true;
    return output;
}

CommandOutput intervene_command(const CommandContext& context, RunControl* control) {
    const std::string& name = context.invocation.name;
    if (!control)
        return CommandOutput::fail(fail("command.not_available",
                                        "Intervention commands act on the script that the Caliper app is attached "
                                        "to, and this session has no app with a script.",
                                        name));
    try {
        const json& args = context.invocation.args;
        if (name == "intervene.points") return points(args, *control);
        if (name == "intervene.list") return list(*control);
        if (name == "intervene.set") return set(args, *control);
        if (name == "intervene.clear") return clear(args, *control);
        return CommandOutput::fail(fail("command.unknown", "This is not an intervention command.", name));
    } catch (const std::exception& e) {
        return CommandOutput::fail(fail(kRunRefused, std::string("The request could not be sent: ") + e.what(), name));
    }
}

}  // namespace caliper_builder
