// The value types of spec version 1 (instrument.h, design/instruments.md).
//
// The comparisons and the names of the kinds are complete. The bind grammar
// and the scene binding are stubs of milestone M1's interface commit: the
// track "spec1" fills them with the codec and the validator.
#include <caliper_builder/spec/instrument.h>

namespace caliper_builder::spec {

bool operator==(const SceneOptions& a, const SceneOptions& b) {
    return a.up == b.up && a.view_camera == b.view_camera && a.include == b.include && a.exclude == b.exclude &&
           a.labels == b.labels && a.axes == b.axes && a.scale_bar == b.scale_bar;
}

std::optional<std::string> live_scene_name(std::string_view binding) {
    (void)binding;
    return std::nullopt;   // stub: track spec1
}

const char* control_kind_name(ControlKind kind) {
    switch (kind) {
        case ControlKind::slider: return "slider";
        case ControlKind::button: return "button";
        case ControlKind::range: return "range";
        case ControlKind::toggle: return "toggle";
        case ControlKind::picker: return "picker";
        case ControlKind::text: return "text";
    }
    return "slider";
}

std::optional<ControlKind> control_kind_from_name(std::string_view name) {
    for (ControlKind k : {ControlKind::slider, ControlKind::button, ControlKind::range, ControlKind::toggle,
                          ControlKind::picker, ControlKind::text})
        if (name == control_kind_name(k)) return k;
    return std::nullopt;
}

bool control_kind_built(ControlKind kind) { return kind == ControlKind::slider || kind == ControlKind::button; }

bool operator==(const ControlTarget& a, const ControlTarget& b) { return a.kind == b.kind && a.name == b.name; }

std::optional<ControlTarget> parse_control_target(std::string_view bind) {
    (void)bind;
    return std::nullopt;   // stub: track spec1
}

std::string control_target_string(const ControlTarget& target) {
    (void)target;
    return {};   // stub: track spec1
}

bool control_takes_target(ControlKind kind, const ControlTarget& target) {
    (void)kind;
    (void)target;
    return false;   // stub: track spec1
}

bool operator==(const ControlOptions& a, const ControlOptions& b) {
    return a.min == b.min && a.max == b.max && a.step == b.step && a.log == b.log && a.value == b.value && a.x == b.x;
}

bool operator==(const Control& a, const Control& b) {
    return a.id == b.id && a.kind == b.kind && a.title == b.title && a.bind == b.bind && a.options == b.options &&
           a.x == b.x;
}

}  // namespace caliper_builder::spec
