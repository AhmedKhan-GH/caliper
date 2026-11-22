#pragma once
// caliper_builder::spec: binding references (design/instrument-interface.md,
// section Binding References).
//
// A binding reference is a small table that stands where the spec takes a
// literal value, and the app evaluates it; the script never sees one. Exactly
// one of the keys `control`, `selection` and `state` names the input:
//
//   { control = "<control id>" [, field = "lo" | "hi"] [, default = <literal>] }
//   { selection = "<panel id>", field = "<pick field>" [, entity = "<path>"] [, default = <literal>] }
//   { state = "<state id>" [, field = "<field>"] [, default = <literal>] }
//
// A reference stands only at the value places that the design page lists.
// Items that existed before references keep their literal fields and hold the
// references of their value places in a map from the key name to the
// Reference (Transform::references, Options::references,
// ControlOptions::references, SlotValue::reference); the literal field is
// then empty. Items that references came with hold ValueOr<T> or keep the
// table in their JSON (Entity::options, Annotation).
//
// parse_reference() only recognises the form; the validator checks the
// targets, the fields, the default's type and the places
// (validate_references.cpp). Nothing here throws.
#include <optional>
#include <string>
#include <string_view>
#include <variant>

#include <nlohmann/json.hpp>

namespace caliper_builder::spec {

// The input keys of a reference, and the other keys it may hold.
inline constexpr const char* kReferenceInputKeys[] = {"control", "selection", "state"};
inline constexpr const char* kReferenceOtherKeys[] = {"field", "entity", "default"};

struct Reference {
    enum class Kind { control, selection, state };
    Kind kind = Kind::state;
    std::string target;                  // the control id, the panel id or the state id ("cursor" is built in)
    std::optional<std::string> field;    // "lo" or "hi", a pick field, a state or cursor field
    std::optional<std::string> entity;   // selection only: only a pick on this entity counts
    nlohmann::json fallback;             // the `default` literal; null when absent
};
bool operator==(const Reference& a, const Reference& b);
inline bool operator!=(const Reference& a, const Reference& b) { return !(a == b); }

// "control", "selection" or "state".
const char* reference_kind_name(Reference::Kind kind);

// A value place that new items declare: a literal of type T or a reference.
template <class T>
using ValueOr = std::variant<T, Reference>;

template <class T>
bool is_reference(const ValueOr<T>& v) {
    return std::holds_alternative<Reference>(v);
}

// Whether `v` has the form of a reference: a table with at least one of the
// input keys. Such a table at a value place is read as a reference, and the
// validator reports a wrong form with spec.bad_reference.
bool looks_like_reference(const nlohmann::json& v);

// The reference that `v` writes, or empty when `v` does not have exactly one
// input key with a string value. Keys other than those of the form are left
// to the validator, which reads the table again.
std::optional<Reference> parse_reference(const nlohmann::json& v);

// The table of a reference, the inverse of parse_reference(): the input key,
// then `field`, `entity` and `default` when set.
nlohmann::json reference_json(const Reference& r);

}  // namespace caliper_builder::spec
