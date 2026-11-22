// State variables (state.h): the names of the types, the comparison and the
// value that a state variable starts with.
#include <caliper_builder/spec/state.h>

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

}  // namespace caliper_builder::spec
