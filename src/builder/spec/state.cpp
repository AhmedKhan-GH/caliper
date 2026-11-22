// State variables (state.h): the names of the types, the comparison and the
// value that a state variable starts with.
#include <caliper_builder/spec/state.h>

#include <cmath>

namespace caliper_builder::spec {

const char* state_type_name(StateType type) {
    switch (type) {
        case StateType::number: return "number";
        case StateType::integer: return "integer";
        case StateType::boolean: return "boolean";
        case StateType::text: return "text";
        case StateType::range: return "range";
        case StateType::list: return "list";
    }
    return "number";
}

std::optional<StateType> state_type_from_name(std::string_view name) {
    for (StateType t : {StateType::number, StateType::integer, StateType::boolean, StateType::text, StateType::range,
                        StateType::list})
        if (name == state_type_name(t)) return t;
    return std::nullopt;
}

bool operator==(const StateVar& a, const StateVar& b) {
    return a.id == b.id && a.title == b.title && a.type == b.type && a.default_value == b.default_value &&
           a.min == b.min && a.max == b.max && a.choices == b.choices && a.x == b.x;
}

nlohmann::json state_initial_value(const StateVar& state) {
    if (!state.default_value.is_null()) return state.default_value;
    switch (state.type) {
        case StateType::number: return 0.0;
        case StateType::integer: return state.choices.empty() ? nlohmann::json(0) : state.choices.front();
        case StateType::boolean: return false;
        case StateType::text: return state.choices.empty() ? nlohmann::json("") : state.choices.front();
        case StateType::range: return nlohmann::json::array({state.min.value_or(0.0), state.max.value_or(0.0)});
        case StateType::list: return nlohmann::json::array();
    }
    return nlohmann::json();
}

// ---- the fit of a value -------------------------------------------------------
// Shared by the validator (validate_states.cpp, validate_controls.cpp) and the
// session's values (values.cpp); state.h declares it.

namespace {

bool is_integral(const nlohmann::json& v) {
    if (v.is_number_integer()) return true;
    if (!v.is_number_float()) return false;
    const double d = v.get<double>();
    return std::isfinite(d) && std::floor(d) == d && std::fabs(d) <= 9.007199254740992e15;
}

bool in_choices(const StateVar& s, const nlohmann::json& v) {
    for (const auto& c : s.choices) {
        if (c == v) return true;
        if (c.is_number() && v.is_number() && c.get<double>() == v.get<double>()) return true;
    }
    return false;
}

std::string number_text(double d) {
    nlohmann::json j = d;
    if (is_integral(j)) return std::to_string(static_cast<long long>(d));
    return j.dump();
}

std::string bounds_problem(const StateVar& s, double d) {
    if (s.min && d < *s.min) return "is below the minimum " + number_text(*s.min);
    if (s.max && d > *s.max) return "is above the maximum " + number_text(*s.max);
    return {};
}

}  // namespace

std::string state_value_problem(const StateVar& s, const nlohmann::json& v) {
    switch (s.type) {
        case StateType::number:
            if (!v.is_number() || !std::isfinite(v.get<double>())) return "is not a finite number";
            return bounds_problem(s, v.get<double>());
        case StateType::integer:
            if (!is_integral(v)) return "is not an integer";
            if (!s.choices.empty() && !in_choices(s, v)) return "is not one of the choices";
            return bounds_problem(s, v.get<double>());
        case StateType::boolean: return v.is_boolean() ? std::string() : "is not true or false";
        case StateType::text:
            if (!v.is_string()) return "is not a text";
            if (v.get_ref<const std::string&>().size() > limits::kStateTextBytes) return "has more than 1,024 bytes";
            if (!s.choices.empty() && !in_choices(s, v)) return "is not one of the choices";
            return {};
        case StateType::range: {
            if (!v.is_array() || v.size() != 2 || !v[0].is_number() || !v[1].is_number() ||
                !std::isfinite(v[0].get<double>()) || !std::isfinite(v[1].get<double>()))
                return "is not a pair [lo, hi] of finite numbers";
            if (v[0].get<double>() > v[1].get<double>()) return "has its lo above its hi";
            std::string why = bounds_problem(s, v[0].get<double>());
            return why.empty() ? bounds_problem(s, v[1].get<double>()) : why;
        }
        case StateType::list:
            if (!v.is_array()) return "is not a list of texts";
            if (v.size() > limits::kStateListItems) return "has more than 256 items";
            for (const auto& item : v) {
                if (!item.is_string()) return "is not a list of texts";
                if (item.get_ref<const std::string&>().size() > limits::kStateTextBytes)
                    return "has an item of more than 1,024 bytes";
                if (!s.choices.empty() && !in_choices(s, item)) return "has an item that is not one of the choices";
            }
            return {};
    }
    return {};
}

}  // namespace caliper_builder::spec
