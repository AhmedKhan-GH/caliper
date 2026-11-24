#pragma once
// caliper_builder::spec: state variables (design/instrument-interface.md,
// section Shared State).
//
// A state variable is a named value of the spec. Controls bound to
// "state:<id>", picks through the panel option `select` and the command
// state.set write it, and references read it. The declaration is spec
// content; the current value is session state (session/values.h) that starts
// at the default when the spec opens and is never saved.
//
// The codec reads and writes the top-level array `state` into Spec::states.
// The validator's checks of state variables are in validate_states.cpp.
// Nothing here throws.
#include <cstddef>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include <nlohmann/json.hpp>

namespace caliper_builder::spec {

namespace limits {
inline constexpr std::size_t kStates = 64;            // spec.too_many
inline constexpr std::size_t kStateChoices = 1024;
inline constexpr std::size_t kStateTextBytes = 1024;
inline constexpr std::size_t kParamTextBytes = 4096;
inline constexpr std::size_t kStateListItems = 256;
}  // namespace limits

// The built-in state variable of the time cursor (history.h). No state
// variable of a spec may take this id, and nothing but the transport, the
// cursor commands and new steps writes it.
inline constexpr const char* kCursorState = "cursor";

enum class StateType { number, integer, boolean, text, range, list };

// "number", "integer", "boolean", "text", "range", "list".
const char* state_type_name(StateType type);
std::optional<StateType> state_type_from_name(std::string_view name);

// One entry of the `state` array.
struct StateVar {
    std::string id;
    std::optional<std::string> title;           // the id when absent
    StateType type = StateType::number;
    nlohmann::json default_value;               // `default`, a value of the type; null when absent
    std::optional<double> min;                  // number, integer and range
    std::optional<double> max;
    std::vector<nlohmann::json> choices;        // text, integer and the items of a list; empty when absent
    nlohmann::json x;                           // the extension table, null when absent
};
bool operator==(const StateVar& a, const StateVar& b);
inline bool operator!=(const StateVar& a, const StateVar& b) { return !(a == b); }

// The value a state variable starts with: its default, else 0, false, the
// first choice or an empty text, [min, max] (0 for a missing bound) or an
// empty list.
nlohmann::json state_initial_value(const StateVar& state);

// Why `v` does not fit `s` (its type, bounds, choices and sizes), or empty
// when it fits. Defined in state.cpp.
std::string state_value_problem(const StateVar& s, const nlohmann::json& v);

}  // namespace caliper_builder::spec
