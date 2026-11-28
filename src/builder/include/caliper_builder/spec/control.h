#pragma once
// caliper_builder::spec: the value types for scene panels and controls
// (design/instruments.md): the `scene` primitive bound to a live scene, with
// its options, and the `control` table.
//
// Spec::controls holds the controls (spec.h); a scene panel is a Panel whose
// primitive is scene, whose slot `scene` holds "live:<scene name>" and whose
// options hold a SceneOptions. Controls and panels share one set of ids, and a
// layout leaf names either one.
//
// The spec library reads and writes these types, and the control edits and
// the spec runtime edit and draw them. The functions are declared here and
// defined in src/builder/spec/control.cpp. Nothing here throws.
#include <nlohmann/json.hpp>

#include <cstddef>
#include <cstdint>
#include <map>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include <caliper_builder/spec/annotation.h>
#include <caliper_builder/spec/reference.h>

namespace caliper_builder::spec {

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

// The kinds of a control (design/instruments.md, Control Table): slider, button and toggle,
// and from design/instrument-interface.md range, picker, text, segmented,
// readout and transport. The app draws every kind.
enum class ControlKind { slider, button, range, toggle, picker, text, segmented, readout, transport };

// "slider", "button", "range", "toggle", "picker", "text", "segmented",
// "readout", "transport".
const char* control_kind_name(ControlKind kind);
std::optional<ControlKind> control_kind_from_name(std::string_view name);
// Whether this Caliper draws the kind; every kind of the plan is built.
bool control_kind_built(ControlKind kind);
// The keys that the app reserves for itself (Keyboard Shortcuts), which a
// shortcut of a choice never takes, with or without Shift. Defined in
// validate_controls.cpp.
const std::vector<std::string>& reserved_shortcut_keys();

// The one target of a control, parsed from its `bind` string:
//   param:<name>   a parameter the script registered with caliper.param
//   action:<name>  an action the script registered with caliper.action
//   run:pause      the run's pause and resume
//   value          a value the spec holds, which references read later
//   intervene:<point>  an intervention point of the attached model, which a
//                  toggle ablates while it is on (design/gpu-programs.md)
//   state:<id>     a state variable of the spec (state.h)
//   cursor         the time cursor, which only a transport binds
// A name has 1 to 63 bytes of UTF-8 and no control characters; a point's
// name has up to 255 bytes.
struct ControlTarget {
    enum class Kind { param, action, run_pause, value, intervene, state, cursor };
    Kind kind = Kind::value;
    std::string name;   // param, action, intervene and state only
};
bool operator==(const ControlTarget& a, const ControlTarget& b);

// The target of a bind string, or empty when it does not match the grammar
// (the validator's spec.bad_binding).
std::optional<ControlTarget> parse_control_target(std::string_view bind);
// The bind string of a target: the inverse of parse_control_target().
std::string control_target_string(const ControlTarget& target);
// Whether a control of `kind` may bind `target`: a slider binds param: or
// value, a button binds action: or run:pause, a toggle binds intervene: or
// param: (the validator's spec.control_mismatch). The reserved kinds bind nothing.
bool control_takes_target(ControlKind kind, const ControlTarget& target);

// One choice of a picker or a segmented group: a value, or the table
// { value, label, key }. `key` is a shortcut: a letter or a digit with an
// optional "shift+".
struct Choice {
    nlohmann::json value;
    std::optional<std::string> label;
    std::optional<std::string> key;
    bool table = false;            // written as a table
};
bool operator==(const Choice& a, const Choice& b);

// The options of a control. An option that is not set is empty.
struct ControlOptions {
    std::optional<double> min;     // slider and range: the low end
    std::optional<double> max;     // slider and range: the high end, above min
    std::optional<double> step;    // slider and range: the step, above 0
    std::optional<bool> log;       // slider: a log scale, with min above 0
    std::optional<double> value;   // slider bound to value: the start value
    // Version 1 (design/instrument-interface.md):
    nlohmann::json value_other;    // the start value when it is not one number: [lo, hi], a boolean, a text or a list
    std::vector<Choice> choices;   // picker and segmented
    std::optional<std::string> choices_from;  // picker: "<source>.<column>"
    std::optional<bool> multiple;  // segmented
    std::optional<std::string> placeholder;   // text
    std::optional<std::int64_t> max_length;   // text
    std::optional<double> min_span;           // range
    std::optional<bool> follow;    // range: moves with its upper bound
    std::optional<double> speed;   // transport: steps per second at open, 0.1 to 1,000
    std::optional<bool> keys;      // transport: Space, the arrows, Home and End
    std::optional<BoundText> bound_text;      // readout: the options `format` and `fields`
    // The references at the value places, by key: `min` and `max` of a
    // slider or a range. The literal field is then empty.
    std::map<std::string, Reference> references;
    nlohmann::json x;              // the extension table, null when absent

    bool empty() const {
        return !min && !max && !step && !log && !value && value_other.is_null() && choices.empty() && !choices_from &&
               !multiple && !placeholder && !max_length && !min_span && !follow && !speed && !keys && !bound_text &&
               references.empty() && x.is_null();
    }
};
bool operator==(const ControlOptions& a, const ControlOptions& b);
inline bool operator!=(const ControlOptions& a, const ControlOptions& b) { return !(a == b); }

// One entry of the `control` array.
struct Control {
    std::string id;                    // shares the panels' set of ids
    ControlKind kind = ControlKind::slider;
    std::optional<std::string> title;  // the id when absent
    std::optional<bool> closable;      // default true when absent
    std::string bind;                  // as written: "param:lr", "action:save", "run:pause", "value"
    ControlOptions options;
    nlohmann::json x;                  // the extension table, null when absent
};
bool operator==(const Control& a, const Control& b);
inline bool operator!=(const Control& a, const Control& b) { return !(a == b); }

}  // namespace caliper_builder::spec
