// The value types of spec version 1 (instrument.h, design/instruments.md).
//
// The comparisons, the names of the kinds, the bind grammar of a control and
// the binding of a scene slot.
#include <caliper_builder/spec/instrument.h>

#include "internal.h"

namespace caliper_builder::spec {

namespace {

// A target's name: 1 to 63 bytes of valid UTF-8 without a control character,
// as caliper.param requires.
bool valid_target_name(std::string_view name) {
    return !name.empty() && name.size() <= limits::kTargetNameBytes && detail::valid_utf8(name) &&
           !detail::has_control(name, detail::TextKind::line);
}

bool starts_with(std::string_view s, std::string_view prefix) { return s.substr(0, prefix.size()) == prefix; }

}  // namespace

bool operator==(const SceneOptions& a, const SceneOptions& b) {
    return a.up == b.up && a.view_camera == b.view_camera && a.include == b.include && a.exclude == b.exclude &&
           a.labels == b.labels && a.axes == b.axes && a.scale_bar == b.scale_bar;
}

std::optional<std::string> live_scene_name(std::string_view binding) {
    if (!starts_with(binding, kLiveScenePrefix)) return std::nullopt;
    const std::string_view name = binding.substr(std::string_view(kLiveScenePrefix).size());
    if (name.empty() || !detail::valid_utf8(name) || detail::has_control(name, detail::TextKind::line))
        return std::nullopt;
    return std::string(name);
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
    ControlTarget t;
    if (bind == "value") {
        t.kind = ControlTarget::Kind::value;
        return t;
    }
    if (bind == "run:pause") {
        t.kind = ControlTarget::Kind::run_pause;
        return t;
    }
    std::string_view name;
    if (starts_with(bind, "param:")) {
        t.kind = ControlTarget::Kind::param;
        name = bind.substr(6);
    } else if (starts_with(bind, "action:")) {
        t.kind = ControlTarget::Kind::action;
        name = bind.substr(7);
    } else {
        return std::nullopt;   // intervene: is not available yet
    }
    if (!valid_target_name(name)) return std::nullopt;
    t.name = std::string(name);
    return t;
}

std::string control_target_string(const ControlTarget& target) {
    switch (target.kind) {
        case ControlTarget::Kind::param: return "param:" + target.name;
        case ControlTarget::Kind::action: return "action:" + target.name;
        case ControlTarget::Kind::run_pause: return "run:pause";
        case ControlTarget::Kind::value: return "value";
        case ControlTarget::Kind::intervene: return "intervene:" + target.name;
    }
    return "value";
}

bool control_takes_target(ControlKind kind, const ControlTarget& target) {
    switch (kind) {
        case ControlKind::slider:
            return target.kind == ControlTarget::Kind::param || target.kind == ControlTarget::Kind::value;
        case ControlKind::button:
            return target.kind == ControlTarget::Kind::action || target.kind == ControlTarget::Kind::run_pause;
        case ControlKind::range:
        case ControlKind::toggle:
        case ControlKind::picker:
        case ControlKind::text: return false;
    }
    return false;
}

bool operator==(const ControlOptions& a, const ControlOptions& b) {
    return a.min == b.min && a.max == b.max && a.step == b.step && a.log == b.log && a.value == b.value && a.x == b.x;
}

bool operator==(const Control& a, const Control& b) {
    return a.id == b.id && a.kind == b.kind && a.title == b.title && a.bind == b.bind && a.options == b.options &&
           a.x == b.x;
}

}  // namespace caliper_builder::spec
