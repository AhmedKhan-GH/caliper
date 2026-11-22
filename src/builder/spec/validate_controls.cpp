// The validator's checks of the options of the control kinds range, picker,
// text, segmented and readout, of their choices and of the shortcut keys of
// segmented groups (design/instrument-interface.md, Controls).
//
// validate.cpp checks each control's bind and the numbers of a slider and a
// range; this file checks which options each kind takes, the start values, the
// choices and the keys. The reserved keys are one list here,
// reserved_shortcut_keys(), which the app's tests compare with the keys that
// its views use.
#include "validate_additions.h"

#include <caliper_builder/spec/state.h>

#include <cmath>
#include <map>
#include <set>

namespace caliper_builder::spec {
// state.cpp: why `v` does not fit `s`, or empty when it fits.
std::string state_value_problem(const StateVar& s, const nlohmann::json& v);

// The keys that the app reserves for itself (Keyboard Shortcuts): the
// transport's and the text fields' keys, and each single key that a view of
// the app uses (F fits a live view's camera, U turns a live scene's up axis).
// A shortcut of a choice never takes one of them, with or without Shift.
const std::vector<std::string>& reserved_shortcut_keys() {
    static const std::vector<std::string> keys = {"space", "left", "right", "up",     "down", "home", "end",
                                                  "escape", "tab", "return", "delete", "f",   "u"};
    return keys;
}
}  // namespace caliper_builder::spec

namespace caliper_builder::spec::detail {

namespace {

using json = nlohmann::json;

constexpr std::size_t kPlaceholderChars = 256;
constexpr std::size_t kSegmentedMin = 2;
constexpr std::size_t kSegmentedMax = 16;

bool is_integral(const json& v) {
    if (v.is_number_integer()) return true;
    if (!v.is_number_float()) return false;
    const double d = v.get<double>();
    return std::isfinite(d) && std::floor(d) == d && std::fabs(d) <= 9.007199254740992e15;
}

// The canonical text of a choice's value, so that 3 and 3.0 are one value.
std::string value_key(const json& v) { return is_integral(v) ? std::to_string(v.get<std::int64_t>()) : v.dump(); }

// The start value of a control as written: the number of `value`, or any
// other value.
json start_value(const ControlOptions& o) { return o.value ? json(*o.value) : o.value_other; }

// Whether `key` follows the grammar: an optional "shift+", then one
// lowercase letter or digit.
bool valid_key(const std::string& key) {
    std::string_view k = key;
    if (k.rfind("shift+", 0) == 0) k.remove_prefix(6);
    return k.size() == 1 && ((k[0] >= 'a' && k[0] <= 'z') || (k[0] >= '0' && k[0] <= '9'));
}

// The key without its Shift, which the reserved list holds.
std::string bare_key(const std::string& key) { return key.rfind("shift+", 0) == 0 ? key.substr(6) : key; }

struct Check {
    const AdditionsCheck& c;
    std::map<std::string, std::string> keys;   // a shortcut key to the path where it first stands

    const StateVar* bound_state(const Control& ctl) const {
        const auto t = parse_control_target(ctl.bind);
        if (!t || t->kind != ControlTarget::Kind::state) return nullptr;
        for (const auto& s : c.spec.states)
            if (s.id == t->name) return &s;
        return nullptr;
    }

    void mismatch(const char* option, const char* takes, const Control& ctl, const std::string& path) {
        c.sink.error("spec.control_mismatch",
                     quote(option) + " is an option of " + takes + "; a " + control_kind_name(ctl.kind) +
                         " does not take it.",
                     path + "." + option);
    }

    // Each option that the control's kind does not take.
    void options_of_kind(const Control& ctl, const std::string& op) {
        const ControlOptions& o = ctl.options;
        const ControlKind k = ctl.kind;
        // validate.cpp refuses the slider options on a button and a toggle, and
        // the transport's own options are validate_history.cpp's.
        const bool slider = k == ControlKind::slider, range = k == ControlKind::range;
        const bool picker = k == ControlKind::picker, text = k == ControlKind::text;
        const bool segmented = k == ControlKind::segmented, readout = k == ControlKind::readout;
        const bool toggle = k == ControlKind::toggle, button = k == ControlKind::button;
        if (!(slider || range || button || toggle)) {
            // validate.cpp refuses these on a button and a toggle.
            if (o.min || o.references.count("min")) mismatch("min", "a slider and a range", ctl, op);
            if (o.max || o.references.count("max")) mismatch("max", "a slider and a range", ctl, op);
            if (o.step) mismatch("step", "a slider and a range", ctl, op);
        }
        if ((button || toggle) && (o.references.count("min") || o.references.count("max")))
            mismatch(o.references.count("min") ? "min" : "max", "a slider and a range", ctl, op);
        if (o.log && !(slider || button || toggle)) mismatch("log", "a slider", ctl, op);
        if (o.value && !(slider || picker || segmented || button || toggle))
            c.sink.error("spec.bad_value",
                         "The start value of a " + std::string(control_kind_name(k)) + " is " +
                             (range ? "a pair [lo, hi]." : text ? "a text." : "not a number; a readout has none."),
                         op + ".value");
        if (!o.value_other.is_null() && (slider || button || readout))
            c.sink.error("spec.bad_value",
                         slider ? std::string("The start value of a slider is a number.")
                                : "A " + std::string(control_kind_name(k)) + " has no start value.",
                         op + ".value");
        if (!o.choices.empty() && !(picker || segmented)) mismatch("choices", "a picker and a segmented group", ctl, op);
        if (o.choices_from && !picker) mismatch("choices_from", "a picker", ctl, op);
        if (o.multiple && !segmented) mismatch("multiple", "a segmented group", ctl, op);
        if (o.placeholder && !text) mismatch("placeholder", "a text control", ctl, op);
        if (o.max_length && !text) mismatch("max_length", "a text control", ctl, op);
        if (o.min_span && !range) mismatch("min_span", "a range", ctl, op);
        if (o.follow && !range) mismatch("follow", "a range", ctl, op);
        if (o.bound_text && !readout) mismatch("format", "a readout", ctl, op);
    }

    // The bounds of a range: from its options, else from its state.
    void range_options(const Control& ctl, const std::string& op) {
        const ControlOptions& o = ctl.options;
        const StateVar* s = bound_state(ctl);
        const bool has_min = o.min || o.references.count("min") || (s && s->min);
        const bool has_max = o.max || o.references.count("max") || (s && s->max);
        if (!has_min || !has_max)
            c.sink.error("spec.missing_key",
                         "A range needs min and max: numbers, references or the bounds of its state.",
                         op + (has_min ? ".max" : ".min"));
        if (o.min_span && !(std::isfinite(*o.min_span) && *o.min_span >= 0))
            c.sink.error("spec.out_of_range", op + ".min_span must be a finite number of at least 0.", op + ".min_span");
        else if (o.min_span && o.min && o.max && *o.min_span > *o.max - *o.min)
            c.sink.error("spec.out_of_range", op + ".min_span must fit between min and max.", op + ".min_span");
        if (o.value_other.is_null()) return;
        const json& v = o.value_other;
        if (!v.is_array() || v.size() != 2 || !v[0].is_number() || !v[1].is_number() ||
            !std::isfinite(v[0].get<double>()) || !std::isfinite(v[1].get<double>()) ||
            v[0].get<double>() > v[1].get<double>()) {
            c.sink.error("spec.bad_value", op + ".value of a range must be a pair [lo, hi] of finite numbers, lo first.",
                         op + ".value");
            return;
        }
        const double lo = v[0].get<double>(), hi = v[1].get<double>();
        if (o.min_span && std::isfinite(*o.min_span) && hi - lo < *o.min_span)
            c.sink.error("spec.out_of_range", op + ".value spans less than min_span.", op + ".value");
        // A literal bound clamps the value; a bound that a reference gives moves
        // with the run, and the value then clamps when the app reads it.
        if ((o.min && lo < *o.min) || (o.max && hi > *o.max))
            c.sink.error("spec.out_of_range", op + ".value must lie within min and max.", op + ".value");
    }

    // The choices of a picker or a segmented group, and their keys.
    void choices(const Control& ctl, const std::string& op) {
        const ControlOptions& o = ctl.options;
        const bool segmented = ctl.kind == ControlKind::segmented;
        if (ctl.kind == ControlKind::picker) {
            if (o.choices.empty() && !o.choices_from)
                c.sink.error("spec.missing_key", "A picker needs choices or choices_from.", op + ".choices");
            if (!o.choices.empty() && o.choices_from)
                c.sink.error("spec.bad_value", "A picker takes choices or choices_from, not both.", op + ".choices_from");
            if (o.choices.size() > limits::kStateChoices)
                c.sink.error("spec.too_many",
                             "The picker has " + std::to_string(o.choices.size()) + " choices; the limit is 1,024.",
                             op + ".choices");
            if (o.choices_from) choices_from(*o.choices_from, op + ".choices_from");
        } else if (o.choices.size() < kSegmentedMin || o.choices.size() > kSegmentedMax) {
            c.sink.error("spec.out_of_range",
                         "A segmented group has 2 to 16 choices, and this one has " + std::to_string(o.choices.size()) +
                             ".",
                         op + ".choices");
        }
        const StateVar* s = bound_state(ctl);
        StateVar item;   // the state a choice must fit, for a list one of its items
        if (s) {
            item = *s;
            if (item.type == StateType::list) item.type = StateType::text;
        }
        std::set<std::string> distinct;
        for (std::size_t k = 0; k < o.choices.size() && k < limits::kStateChoices; ++k) {
            const Choice& ch = o.choices[k];
            const std::string cp = index_path(op + ".choices", k);
            const bool typed = (ch.value.is_string() && ch.value.get_ref<const std::string&>().size() <=
                                                            limits::kStateTextBytes) ||
                               is_integral(ch.value);
            if (!typed) {
                c.sink.error("spec.bad_value", cp + " must be a text of at most 1,024 bytes or an integer.",
                             cp + (ch.table ? ".value" : ""));
                continue;
            }
            if (!distinct.insert(value_key(ch.value)).second)
                c.sink.error("spec.bad_value", cp + " repeats an earlier choice.", cp);
            if (s) {
                const std::string why = state_value_problem(item, ch.value);
                if (!why.empty())
                    c.sink.error("spec.control_mismatch",
                                 cp + " " + why + " of the state " + quote(s->id) + ", which the control writes.", cp);
            }
            if (ch.label && (ch.label->empty() || code_points(*ch.label) > limits::kTitleChars ||
                             has_control(*ch.label, TextKind::line)))
                c.sink.error("spec.bad_value",
                             cp + ".label must be one line of 1 to " + std::to_string(limits::kTitleChars) +
                                 " characters.",
                             cp + ".label");
            if (ch.key) shortcut(*ch.key, segmented, cp + ".key");
        }
        // The start value is a choice, or with `multiple` a list of them.
        const json start = start_value(o);
        if (start.is_null() || o.choices.empty()) return;
        auto known = [&](const json& v) { return (v.is_string() || is_integral(v)) && distinct.count(value_key(v)); };
        if (segmented && o.multiple && *o.multiple) {
            bool ok = start.is_array();
            std::set<std::string> on;
            if (ok)
                for (const auto& v : start) ok = ok && known(v) && on.insert(value_key(v)).second;
            if (!ok)
                c.sink.error("spec.out_of_range", op + ".value of a group with multiple must be a list of its choices.",
                             op + ".value");
        } else if (!known(start)) {
            c.sink.error("spec.out_of_range", op + ".value must be one of the choices.", op + ".value");
        }
    }

    void choices_from(const std::string& from, const std::string& path) {
        const auto dot = from.find('.');
        const std::string source = dot == std::string::npos ? from : from.substr(0, dot);
        if (dot == std::string::npos || dot + 1 >= from.size() || !valid_id(source)) {
            c.sink.error("spec.bad_value", path + " must be <source>.<column>.", path);
            return;
        }
        if (!c.spec.find_source(source))
            c.sink.error("spec.unknown_source", quote(source) + " is not a source of this spec.", path);
    }

    void shortcut(const std::string& key, bool segmented, const std::string& path) {
        if (!segmented) {
            c.sink.error("spec.control_mismatch", "A key is a shortcut of a choice of a segmented group.", path);
            return;
        }
        if (!valid_key(key)) {
            c.sink.error("spec.bad_value",
                         quote(key) + " is not a shortcut; a shortcut is one letter or digit with an optional "
                                      "shift+, and the app keeps every chord with Command, Control or Option.",
                         path);
            return;
        }
        if (contains(reserved_shortcut_keys(), bare_key(key))) {
            c.sink.error("spec.reserved_shortcut",
                         "The app uses the key " + quote(bare_key(key)) + " itself; choose another key.", path);
            return;
        }
        const auto [it, fresh] = keys.emplace(key, path);
        if (!fresh)
            c.sink.error("spec.duplicate_shortcut",
                         "The key " + quote(key) + " is already the shortcut at " + it->second + ".", path);
    }

    void text_options(const Control& ctl, const std::string& op) {
        const ControlOptions& o = ctl.options;
        if (o.max_length && (*o.max_length < 1 || *o.max_length > static_cast<std::int64_t>(limits::kStateTextBytes)))
            c.sink.error("spec.out_of_range", op + ".max_length must be from 1 to 1,024.", op + ".max_length");
        if (o.placeholder &&
            (code_points(*o.placeholder) > kPlaceholderChars || has_control(*o.placeholder, TextKind::line)))
            c.sink.error("spec.bad_value", op + ".placeholder must be one line of at most 256 characters.",
                         op + ".placeholder");
        if (o.value_other.is_null()) return;
        const std::size_t limit = o.max_length ? static_cast<std::size_t>(std::max<std::int64_t>(*o.max_length, 0))
                                               : limits::kStateTextBytes;
        if (!o.value_other.is_string())
            c.sink.error("spec.bad_value", op + ".value of a text control must be a text.", op + ".value");
        else if (o.value_other.get_ref<const std::string&>().size() > limit)
            c.sink.error("spec.out_of_range", op + ".value is longer than max_length.", op + ".value");
    }

    void control(std::size_t i) {
        const Control& ctl = c.spec.controls[i];
        if (!control_kind_built(ctl.kind)) return;   // validate.cpp refuses the kind
        const std::string op = index_path("control", i) + ".options";
        options_of_kind(ctl, op);
        switch (ctl.kind) {
            case ControlKind::range: range_options(ctl, op); break;
            case ControlKind::picker:
            case ControlKind::segmented: choices(ctl, op); break;
            case ControlKind::text: text_options(ctl, op); break;
            case ControlKind::toggle:
                if (!ctl.options.value_other.is_null() && !ctl.options.value_other.is_boolean())
                    c.sink.error("spec.bad_value", op + ".value of a toggle must be true or false.", op + ".value");
                break;
            default: break;
        }
        if (ctl.kind == ControlKind::segmented) {
            const StateVar* s = bound_state(ctl);
            const bool multiple = ctl.options.multiple && *ctl.options.multiple;
            if (s && (s->type == StateType::list) != multiple)
                c.sink.error("spec.control_mismatch",
                             multiple ? "A group with multiple writes a list, and the state " + quote(s->id) + " is a " +
                                            state_type_name(s->type) + "."
                                      : "A group without multiple writes one value, and the state " + quote(s->id) +
                                            " is a list; set multiple = true.",
                             op + ".multiple");
        }
    }
};

}  // namespace

void validate_controls(const AdditionsCheck& c) {
    Check check{c, {}};
    for (std::size_t i = 0; i < c.spec.controls.size() && i < limits::kControls; ++i) check.control(i);
}

}  // namespace caliper_builder::spec::detail
