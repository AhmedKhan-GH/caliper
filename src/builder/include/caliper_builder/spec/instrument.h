#pragma once
// caliper_builder::spec: the value types that spec version 1 adds for
// instruments (design/instruments.md): the `scene` primitive bound to a live
// scene, with its options, and the `control` table.
//
// Version 1 is version 0 with three additions: the primitive `scene`, the
// top-level array `control` and, later, binding references.
// A version 0 document stays valid and reads the same way. Spec::controls
// holds the controls (spec.h); a scene panel is a Panel whose primitive is
// scene, whose slot `scene` holds "live:<scene name>" and whose options hold
// a SceneOptions. Controls and panels share one set of ids, and a layout leaf
// names either one.
//
// The spec library reads and writes these types, and the control edits and
// the spec runtime edit and draw them. The functions are declared here and defined in
// src/builder/spec/instrument.cpp. Nothing here throws.
#include <nlohmann/json.hpp>

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace caliper_builder::spec {

// The format version that adds instruments.
inline constexpr std::int64_t kInstrumentVersion = 1;

namespace limits {
inline constexpr std::size_t kControls = 64;           // spec.too_many
inline constexpr std::size_t kScenePatterns = 64;      // include and exclude, each
inline constexpr std::size_t kTargetNameBytes = 63;    // a parameter's or an action's name, as caliper.param allows
}  // namespace limits

// The slot of a scene panel, and the prefix of the live scene it binds.
inline constexpr const char* kSceneSlot = "scene";
inline constexpr const char* kLiveScenePrefix = "live:";

// The options of a scene panel. An option that is not set is empty, and the
// scene's own value (its VIEWS entry) applies.
struct SceneOptions {
    std::optional<std::string> up;            // "+x", "-x", "+y", "-y", "+z" or "-z"
    nlohmann::json view_camera;               // null, an orbit object or a camera's entity path
    std::vector<std::string> include;         // entity path patterns; empty: every entity
    std::vector<std::string> exclude;         // entity path patterns left out, after include
    std::optional<bool> labels;               // default true
    std::optional<bool> axes;                 // default true
    std::optional<bool> scale_bar;            // default false
};
bool operator==(const SceneOptions& a, const SceneOptions& b);
inline bool operator!=(const SceneOptions& a, const SceneOptions& b) { return !(a == b); }

// The scene name of a scene slot's binding: "model" for "live:model", or
// empty when the binding has another form.
std::optional<std::string> live_scene_name(std::string_view binding);

// The kinds of a control. The first version builds slider and button; the others are part of
// the format and refused by the validator with spec.unsupported_control until
// they are built.
enum class ControlKind { slider, button, range, toggle, picker, text };

// "slider", "button", "range", "toggle", "picker", "text".
const char* control_kind_name(ControlKind kind);
std::optional<ControlKind> control_kind_from_name(std::string_view name);
// Whether this Caliper draws the kind (slider and button in the first version).
bool control_kind_built(ControlKind kind);

// The one target of a control, parsed from its `bind` string:
//   param:<name>   a parameter the script registered with caliper.param
//   action:<name>  an action the script registered with caliper.action
//   run:pause      the run's pause and resume
//   value          a value the spec holds, which references read later
// A name has 1 to 63 bytes of UTF-8 and no control characters.
struct ControlTarget {
    enum class Kind { param, action, run_pause, value };
    Kind kind = Kind::value;
    std::string name;   // param and action only
};
bool operator==(const ControlTarget& a, const ControlTarget& b);

// The target of a bind string, or empty when it does not match the grammar
// (the validator's spec.bad_binding).
std::optional<ControlTarget> parse_control_target(std::string_view bind);
// The bind string of a target: the inverse of parse_control_target().
std::string control_target_string(const ControlTarget& target);
// Whether a control of `kind` may bind `target`: a slider binds param: or
// value, a button binds action: or run:pause (the validator's
// spec.control_mismatch). The reserved kinds bind nothing in the first version.
bool control_takes_target(ControlKind kind, const ControlTarget& target);

// The options of a control. An option that is not set is empty.
struct ControlOptions {
    std::optional<double> min;     // slider: the range's low end
    std::optional<double> max;     // slider: the range's high end, above min
    std::optional<double> step;    // slider: the step, above 0
    std::optional<bool> log;       // slider: a log scale, with min above 0
    std::optional<double> value;   // slider bound to value: the start value
    nlohmann::json x;              // the extension table, null when absent

    bool empty() const { return !min && !max && !step && !log && !value && x.is_null(); }
};
bool operator==(const ControlOptions& a, const ControlOptions& b);
inline bool operator!=(const ControlOptions& a, const ControlOptions& b) { return !(a == b); }

// One entry of the `control` array.
struct Control {
    std::string id;                    // shares the panels' set of ids
    ControlKind kind = ControlKind::slider;
    std::optional<std::string> title;  // the id when absent
    std::string bind;                  // as written: "param:lr", "action:save", "run:pause", "value"
    ControlOptions options;
    nlohmann::json x;                  // the extension table, null when absent
};
bool operator==(const Control& a, const Control& b);
inline bool operator!=(const Control& a, const Control& b) { return !(a == b); }

}  // namespace caliper_builder::spec
