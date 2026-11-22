// The state commands and control.set
// (design/instrument-interface.md, Commands).
//
// state.create, state.patch and state.remove edit the spec's `state` array
// as the control edits do: on a copy, refusing an edit whose own rule fails
// (state.exists, state.not_found, state.in_use) and handing the result to the
// validator, whose first error refuses the edit. An edit that adds a state
// variable to a version 0 spec raises the version to 1.
//
// state.set, state.get and control.set change or read the session's values:
// they have no undo step and no journal entry, and they add the same `state`
// and `control` events that a person's edit adds, with the caller's origin.
#include <caliper_builder/session/instrument_commands.h>

#include <caliper_builder/spec/state.h>

#include <set>

namespace caliper_builder {

namespace {

using nlohmann::json;

SpecEdit refuse(Error err) {
    SpecEdit out;
    out.error = std::move(err);
    return out;
}

// Validates the edited spec: the first error refuses the edit; otherwise the
// warnings are kept.
SpecEdit finish(spec::Spec next, const spec::SourceResolver& resolver, json result) {
    SpecEdit out;
    spec::ValidationReport report = spec::validate(next, resolver);
    if (Error err = first_spec_error(report.diagnostics)) {
        out.error = std::move(err);
        return out;
    }
    for (auto& d : report.diagnostics)
        if (!d.is_error()) out.warnings.push_back(std::move(d));
    out.spec = std::move(next);
    out.result = std::move(result);
    return out;
}

std::string str_arg(const json& args, const char* key) {
    const auto it = args.find(key);
    return it != args.end() && it->is_string() ? it->get<std::string>() : std::string();
}

std::size_t state_index(const spec::Spec& s, const std::string& id) {
    for (std::size_t i = 0; i < s.states.size(); ++i)
        if (s.states[i].id == id) return i;
    return s.states.size();
}

// Sets one key of a state variable from a JSON value; null clears a key that
// may be absent. An error for an unknown key or a value of the wrong type.
Error set_key(spec::StateVar& s, const std::string& key, const json& v, const std::string& where) {
    auto wrong = [&](const char* what) { return Error("spec.wrong_type", "The key " + key + " is " + what + ".", where); };
    if (key == "type") {
        const auto t = v.is_string() ? spec::state_type_from_name(v.get<std::string>()) : std::nullopt;
        if (!t)
            return Error("spec.bad_value",
                         "A state variable's type is number, integer, boolean, text, range or list.", where);
        s.type = *t;
    } else if (key == "title") {
        if (v.is_null()) s.title.reset();
        else if (v.is_string()) s.title = v.get<std::string>();
        else return wrong("a text");
    } else if (key == "default") {
        s.default_value = v;
    } else if (key == "min" || key == "max") {
        std::optional<double>& slot = key == "min" ? s.min : s.max;
        if (v.is_null()) slot.reset();
        else if (v.is_number()) slot = v.get<double>();
        else return wrong("a number");
    } else if (key == "choices") {
        if (v.is_null()) s.choices.clear();
        else if (v.is_array()) s.choices.assign(v.begin(), v.end());
        else return wrong("a list of values");
    } else if (key == "x") {
        if (!v.is_null() && !v.is_object()) return wrong("an extension table");
        s.x = v;
    } else if (key == "id") {
        return Error("state.id_immutable", "A state variable's id cannot change; remove it and create another.",
                     where);
    } else {
        return Error("spec.unknown_key", "A state variable has no key " + key + ".", where);
    }
    return {};
}

SpecEdit state_create(const spec::Spec& s, const json& args, const spec::SourceResolver& resolver) {
    const std::string id = str_arg(args, "id");
    if (state_index(s, id) != s.states.size())
        return refuse(Error("state.exists", "The spec already has a state variable " + id + "; choose another id.",
                            "args.id"));
    spec::StateVar v;
    v.id = id;
    for (const auto& [key, value] : args.items()) {
        if (key == "id" || key == "dry_run") continue;
        if (Error e = set_key(v, key, value, "args." + key)) return refuse(std::move(e));
    }
    spec::Spec next = s;
    if (next.version < spec::kInstrumentVersion) next.version = spec::kInstrumentVersion;
    next.states.push_back(std::move(v));
    return finish(std::move(next), resolver, json{{"id", id}});
}

SpecEdit state_patch(const spec::Spec& s, const json& args, const spec::SourceResolver& resolver) {
    const std::string id = str_arg(args, "id");
    const std::size_t index = state_index(s, id);
    if (index == s.states.size())
        return refuse(Error("state.not_found", "The spec has no state variable " + id + ".", "args.id"));
    const auto set = args.find("set");
    if (set == args.end() || !set->is_object())
        return refuse(Error("spec.wrong_type", "set is a table of the keys to change.", "args.set"));
    spec::Spec next = s;
    for (const auto& [key, value] : set->items())
        if (Error e = set_key(next.states[index], key, value, "args.set." + key)) return refuse(std::move(e));
    return finish(std::move(next), resolver, json{{"id", id}});
}

// What reads or writes the state variable `id`: a reference, a control bound
// to it, or a panel whose picks write it. Empty when nothing does.
std::string user_of(const spec::Spec& s, const std::string& id) {
    for (const auto& c : s.controls)
        if (c.bind == "state:" + id) return "the control " + c.id + " writes it";
    for (const auto& p : s.panels)
        if (p.options.select && p.options.select->state == id) return "the picks of the panel " + p.id + " write it";
    ValueStore probe;
    for (const auto& st : probe.describe_states(s))
        if (st["id"] == id && !st["readers"].empty())
            return "the reference at " + st["readers"][0].get<std::string>() + " reads it";
    return {};
}

SpecEdit state_remove(const spec::Spec& s, const json& args, const spec::SourceResolver& resolver) {
    const std::string id = str_arg(args, "id");
    const std::size_t index = state_index(s, id);
    if (index == s.states.size())
        return refuse(Error("state.not_found", "The spec has no state variable " + id + ".", "args.id"));
    const std::string user = user_of(s, id);
    if (!user.empty())
        return refuse(Error("state.in_use", "The state variable " + id + " is in use: " + user + ".", "args.id"));
    spec::Spec next = s;
    next.states.erase(next.states.begin() + static_cast<std::ptrdiff_t>(index));
    return finish(std::move(next), resolver, json{{"id", id}});
}

CommandOutput no_spec() { return CommandOutput::fail(Error("spec.no_open_spec", "No spec is open.")); }

CommandOutput state_set(const CommandContext& ctx, InstrumentContext& ic) {
    const json& args = ctx.invocation.args;
    const std::string id = str_arg(args, "id");
    spec::Reference ref;
    ref.kind = spec::Reference::Kind::state;
    ref.target = id;
    const json previous = ic.values.value(ref);
    if (Error e = ic.values.set_state(id, args.contains("value") ? args["value"] : json(), ctx.invocation.name,
                                      ctx.invocation.origin, ic.events))
        return CommandOutput::fail(std::move(e));
    return CommandOutput::ok(json{{"value", ic.values.value(ref)}, {"previous", previous}});
}

CommandOutput state_get(const CommandContext& ctx, InstrumentContext& ic) {
    const json& args = ctx.invocation.args;
    json all = ic.values.describe_states(ic.spec);
    if (!args.contains("ids") || !args["ids"].is_array()) return CommandOutput::ok(json{{"states", std::move(all)}});
    json out = json::array();
    std::set<std::string> seen;
    for (const auto& id : args["ids"]) {
        const std::string want = id.is_string() ? id.get<std::string>() : std::string();
        if (!seen.insert(want).second) continue;
        bool found = false;
        for (const auto& st : all)
            if (st["id"] == want) {
                out.push_back(st);
                found = true;
            }
        if (!found)
            return CommandOutput::fail(Error("state.not_found", "The open spec has no state variable " + want + ".",
                                             "args.ids"));
    }
    return CommandOutput::ok(json{{"states", std::move(out)}});
}

CommandOutput control_set(const CommandContext& ctx, InstrumentContext& ic) {
    const json& args = ctx.invocation.args;
    const std::string id = str_arg(args, "id");
    const spec::Control* c = ic.spec.find_control(id);
    if (!c) return CommandOutput::fail(Error("control.not_found", "The open spec has no control " + id + ".", "args.id"));
    const auto t = spec::parse_control_target(c->bind);
    if (!t || (t->kind != spec::ControlTarget::Kind::value && t->kind != spec::ControlTarget::Kind::state))
        return CommandOutput::fail(Error("control.not_settable",
                                         "The control " + id + " binds " + c->bind +
                                             (t && t->kind == spec::ControlTarget::Kind::param
                                                  ? "; run.set_param sets a parameter."
                                                  : "; control.set sets a control bound to value or to a state."),
                                         "args.id"));
    // The bounds of a range may read the cursor; read them as they are now.
    ic.values.set_cursor(ic.history.cursor());
    if (Error e = ic.values.set_control(id, args.contains("value") ? args["value"] : json(), ctx.invocation.origin,
                                        ic.events))
        return CommandOutput::fail(std::move(e));
    return CommandOutput::ok(json{{"value", ic.values.control_value(id)}});
}

}  // namespace

SpecEdit apply_state_edit(const spec::Spec& spec, std::string_view name, const nlohmann::json& args,
                          const spec::SourceResolver& resolver) {
    if (name == "state.create") return state_create(spec, args, resolver);
    if (name == "state.patch") return state_patch(spec, args, resolver);
    if (name == "state.remove") return state_remove(spec, args, resolver);
    return refuse(not_implemented(name));
}

CommandOutput state_command(const CommandContext& ctx, InstrumentContext& ic) {
    if (!ic.spec_open) return no_spec();
    const std::string& name = ctx.invocation.name;
    if (name == "state.set") return state_set(ctx, ic);
    if (name == "state.get") return state_get(ctx, ic);
    if (name == "control.set") return control_set(ctx, ic);
    return CommandOutput::fail(not_implemented(name));
}

}  // namespace caliper_builder
