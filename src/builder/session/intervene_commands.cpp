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
    return json{{"point", p.point}, {"kind", p.kind},       {"module", p.module},
                {"head", opt_json(p.head)}, {"type", p.type}, {"shape", std::move(shape)}};
}

json intervention_json(const Intervention& i) {
    return json{{"point", i.point},
                {"op", i.op},
                {"from_step", opt_json(i.from_step)},
                {"until_step", opt_json(i.until_step)},
                {"since_step", opt_json(i.since_step)},
                {"state", i.state}};
}

// The refusal of any intervention command without an attached script that
// announced `interventions`; with `change`, also without control. Empty when
// it may go on.
Error check(const RunState& state, RunControl& control, bool change) {
    if (!state.attached)
        return fail(kRunNotAttached,
                    "No script is attached to Caliper; start a script with caliper.watch(model, optimizer) and "
                    "attach the app to its stream first.");
    if (!control.interventions_supported())
        return fail(kInterveneNotSupported,
                    "The script " + state.script +
                        " takes no interventions: it announces them only under caliper.watch(model, optimizer).");
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
    i.op = str_arg(args, "op");
    i.from_step = step_arg(args, "from_step");
    i.until_step = step_arg(args, "until_step");
    if (i.op != "ablate")
        return CommandOutput::fail(fail(kInterveneOpReserved,
                                        "The op " + i.op + " comes later; ablate is the only op built.", "args.op"));
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
