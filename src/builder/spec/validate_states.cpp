// The validator's checks of state variables, of the option `select` and of the
// controls bound to "state:<id>" (design/instrument-interface.md, Shared
// State).
//
// A state variable has a unique id other than "cursor", a type, and bounds,
// choices and a default that fit the type. A pick writes a state variable
// through the option `select` of a scene panel, with a pick field whose type
// fits the variable. A control bound to a state variable has a kind that
// writes the variable's type, and two controls that write one variable give
// the warning spec.state_writers.
#include "validate_additions.h"

#include <caliper_builder/spec/state.h>

#include <cmath>
#include <map>
#include <set>

namespace caliper_builder::spec {
// state.cpp: why `v` does not fit `s` (its type, bounds, choices and sizes),
// or empty when it fits.
std::string state_value_problem(const StateVar& s, const nlohmann::json& v);
}  // namespace caliper_builder::spec

namespace caliper_builder::spec::detail {

namespace {

// The type of the value of a pick field (Pick Fields): 'i' an integer, 'n' a
// number, 't' a text; 0 for a name that is not a pick field.
char pick_field_type(const std::string& field) {
    static const std::map<std::string, char> types = {
        {"entity", 't'}, {"index", 'i'}, {"row_id", 'i'}, {"row", 'i'}, {"column", 'i'}, {"x", 'n'},
        {"y", 'n'},      {"z", 'n'},     {"value", 'n'},  {"step", 'i'}, {"annotation", 't'}};
    const auto it = types.find(field);
    return it == types.end() ? 0 : it->second;
}

// Whether a pick field of type `field` may write a state variable of `type`.
bool pick_writes(char field, StateType type) {
    switch (field) {
        case 'i': return type == StateType::integer || type == StateType::number;
        case 'n': return type == StateType::number;
        case 't': return type == StateType::text;
    }
    return false;
}

// The state types that a control kind writes (Control Kinds).
std::vector<StateType> kind_writes(ControlKind kind) {
    switch (kind) {
        case ControlKind::slider: return {StateType::number, StateType::integer};
        case ControlKind::range: return {StateType::range};
        case ControlKind::toggle: return {StateType::boolean};
        case ControlKind::picker: return {StateType::text, StateType::integer};
        case ControlKind::text: return {StateType::text};
        case ControlKind::segmented: return {StateType::text, StateType::integer, StateType::list};
        default: return {};
    }
}

std::string type_list(const std::vector<StateType>& types) {
    std::vector<std::string> names;
    for (StateType t : types) names.push_back(state_type_name(t));
    return join(names);
}

bool valid_entity_path(const std::string& path) {
    if (path.empty()) return false;
    std::size_t segments = 1, len = 0;
    for (char ch : path) {
        if (ch == '/') {
            if (len == 0) return false;
            ++segments;
            len = 0;
            continue;
        }
        const bool ok = (ch >= 'a' && ch <= 'z') || (ch >= '0' && ch <= '9') || ch == '_' || ch == '-';
        if (!ok) return false;
        ++len;
    }
    return len > 0 && segments <= 8;
}

void check_state(const AdditionsCheck& c, std::size_t i, std::set<std::string>& seen) {
    const StateVar& s = c.spec.states[i];
    const std::string path = index_path("state", i);
    if (!valid_id(s.id)) {
        c.sink.error("spec.bad_id",
                     quote(s.id) + " is not an id; an id starts with a lowercase letter and holds lowercase letters, "
                                   "digits, - and _, at most 64 in all.",
                     path + ".id");
    } else if (s.id == kCursorState) {
        c.sink.error("spec.bad_id",
                     "The id cursor is reserved for the built-in state of the time cursor; choose another id.",
                     path + ".id");
    } else if (!seen.insert(s.id).second) {
        c.sink.error("spec.duplicate_id", "Two state variables have the id " + quote(s.id) + ".", path + ".id");
    }
    if (s.title && (s.title->empty() || code_points(*s.title) > limits::kTitleChars || has_control(*s.title, TextKind::line)))
        c.sink.error("spec.bad_value",
                     path + ".title must be one line of 1 to " + std::to_string(limits::kTitleChars) + " characters.",
                     path + ".title");

    // Bounds: number, integer and range only; finite, with min below max.
    const bool bounded = s.type == StateType::number || s.type == StateType::integer || s.type == StateType::range;
    for (const auto& [key, bound] : {std::pair<const char*, const std::optional<double>*>{"min", &s.min},
                                     std::pair<const char*, const std::optional<double>*>{"max", &s.max}}) {
        if (!*bound) continue;
        if (!bounded)
            c.sink.error("spec.bad_value",
                         std::string(key) + " is a bound of a number, an integer or a range; a " +
                             state_type_name(s.type) + " has none.",
                         path + "." + key);
        else if (!std::isfinite(**bound))
            c.sink.error("spec.out_of_range", path + "." + key + " must be a finite number.", path + "." + key);
    }
    if (bounded && s.min && s.max && std::isfinite(*s.min) && std::isfinite(*s.max) && !(*s.min < *s.max))
        c.sink.error("spec.out_of_range", path + ".min must be below " + path + ".max.", path + ".min");

    // Choices: text, integer and list only; 1 to 1,024 distinct values that
    // fit the type of a value (or of an item of a list).
    if (!s.choices.empty()) {
        const bool takes = s.type == StateType::text || s.type == StateType::integer || s.type == StateType::list;
        if (!takes) {
            c.sink.error("spec.bad_value",
                         "choices are the values of a text, an integer or a list; a " +
                             std::string(state_type_name(s.type)) + " has none.",
                         path + ".choices");
        } else if (s.choices.size() > limits::kStateChoices) {
            c.sink.error("spec.too_many",
                         "The state variable has " + std::to_string(s.choices.size()) +
                             " choices; the limit is 1,024.",
                         path + ".choices");
        } else {
            StateVar item = s;
            item.choices.clear();
            if (s.type == StateType::list) item.type = StateType::text;
            std::set<std::string> distinct;
            for (std::size_t k = 0; k < s.choices.size(); ++k) {
                const std::string cp = index_path(path + ".choices", k);
                const std::string why = state_value_problem(item, s.choices[k]);
                if (!why.empty())
                    c.sink.error("spec.bad_value", cp + " " + why + ".", cp);
                else if (!distinct.insert(s.choices[k].dump()).second)
                    c.sink.error("spec.bad_value", cp + " repeats an earlier choice.", cp);
            }
        }
    }

    // The default fits the type, the bounds and the choices.
    if (!s.default_value.is_null()) {
        const std::string why = state_value_problem(s, s.default_value);
        if (!why.empty())
            c.sink.error("spec.out_of_range", path + ".default " + why + ".", path + ".default");
    }
}

// The option `select` of panel `i`.
void check_select(const AdditionsCheck& c, std::size_t i) {
    const Panel& p = c.spec.panels[i];
    const StateWrite& w = *p.options.select;
    const std::string path = index_path("panel", i) + ".options.select";
    if (p.primitive != Primitive::scene) {
        c.sink.error("spec.unknown_option",
                     std::string("select is an option of a scene panel, where a person picks items; a ") + to_string(p.primitive) +
                         " panel has no picks.",
                     path);
        return;
    }
    const StateVar* s = nullptr;
    for (const auto& v : c.spec.states)
        if (v.id == w.state) s = &v;
    if (w.state == kCursorState) {
        c.sink.error("spec.reference_type", "A pick cannot write the cursor; only the transport moves it.",
                     path + ".state");
    } else if (!s) {
        c.sink.error("spec.unknown_state", quote(w.state) + " is not a state variable of this spec.", path + ".state");
    }
    const char type = pick_field_type(w.field);
    if (!type) {
        c.sink.error("spec.unknown_field",
                     quote(w.field) + " is not a pick field; the fields are entity, index, row_id, row, column, x, y, "
                                      "z, value and step.",
                     path + ".field");
    } else if (s && !pick_writes(type, s->type)) {
        c.sink.error("spec.reference_type",
                     "The pick field " + quote(w.field) + " gives " +
                         (type == 'i' ? "an integer" : type == 'n' ? "a number" : "a text") +
                         ", which does not fit the " + state_type_name(s->type) + " state " + quote(s->id) + ".",
                     path + ".field");
    }
    if (w.entity && !valid_entity_path(*w.entity))
        c.sink.error("spec.bad_value",
                     path + ".entity must be an entity path: 1 to 8 segments of a-z, 0-9, _ and -, joined by /.",
                     path + ".entity");
}

}  // namespace

void validate_states(const AdditionsCheck& c) {
    if (c.spec.states.size() > limits::kStates)
        c.sink.error("spec.too_many",
                     "The spec has " + std::to_string(c.spec.states.size()) + " state variables; the limit is 64.",
                     "state");
    std::set<std::string> seen;
    for (std::size_t i = 0; i < c.spec.states.size() && i < limits::kStates; ++i) check_state(c, i, seen);
    for (std::size_t i = 0; i < c.spec.panels.size(); ++i)
        if (c.spec.panels[i].options.select) check_select(c, i);

    // The controls bound to a state variable: a known variable of a type the
    // kind writes, and one control for each variable.
    std::map<std::string, std::vector<std::string>> writers;
    for (std::size_t i = 0; i < c.spec.controls.size(); ++i) {
        const Control& ctl = c.spec.controls[i];
        const auto target = parse_control_target(ctl.bind);
        if (!target || target->kind != ControlTarget::Kind::state) continue;
        const std::string path = index_path("control", i);
        const StateVar* s = nullptr;
        for (const auto& v : c.spec.states)
            if (v.id == target->name) s = &v;
        if (target->name == kCursorState) {
            c.sink.error("spec.control_mismatch",
                         "A control cannot bind state:cursor; a transport binds the cursor with bind = \"cursor\".",
                         path + ".bind");
            continue;
        }
        if (!s) {
            c.sink.error("spec.unknown_state", quote(target->name) + " is not a state variable of this spec.",
                         path + ".bind");
            continue;
        }
        const auto types = kind_writes(ctl.kind);
        bool fits = false;
        for (StateType t : types) fits = fits || t == s->type;
        if (!types.empty() && !fits)
            c.sink.error("spec.control_mismatch",
                         "A " + std::string(control_kind_name(ctl.kind)) + " writes a " + type_list(types) +
                             ", and the state " + quote(s->id) + " is a " + state_type_name(s->type) + ".",
                         path + ".bind");
        if (ctl.options.value || !ctl.options.value_other.is_null())
            c.sink.error("spec.control_mismatch",
                         "A control bound to a state starts at the state's value; set the state's default instead "
                         "of the option value.",
                         path + ".options.value");
        writers[s->id].push_back(ctl.id);
    }
    for (const auto& [state, ids] : writers)
        if (ids.size() > 1)
            c.sink.warning("spec.state_writers",
                           "The controls " + join(ids) + " all write the state " + quote(state) +
                               "; one setting usually has one control.",
                           "state");
}

}  // namespace caliper_builder::spec::detail
