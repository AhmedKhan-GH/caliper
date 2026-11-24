// The controls of a spec (spec_instruments.h, design/instruments.md, Control
// Table, and design/instrument-interface.md, Controls).
//
// A control is one row: its title on the left, its widget on the right. This file
// draws these kinds:
//   slider     bound to param:<name>, it shows the script's parameter on the
//              parameter's own range and scale, narrowed by the options, and
//              sends each new value through the run control while the person
//              drags; bound to `value` or to a state, it holds the value in the
//              session's values.
//   button     bound to action:<name>, it queues the action; bound to
//              run:pause, it pauses a run that runs and resumes a paused one.
//   toggle     bound to intervene:<point>, on it ablates that point of the
//              attached model and off it clears the point's interventions
//              (design/gpu-programs.md, Toggle Control); bound to param:<name>
//              of a boolean parameter, it sets the parameter. It shows the
//              script's state, and the person's last click until the script
//              shows it or for a few seconds. Bound to `value` or to a state,
//              it holds a boolean.
//   range      two grabs on one track, [lo, hi], between bounds that may read
//              references; with `follow` the session moves it with its upper
//              bound.
//   picker     a menu of its choices.
//   text       a field that applies its text on Return or when it loses focus.
//   segmented  a row of 2 to 16 square buttons, the ones that are on in the
//              accent colour; a choice's key shows in its tooltip.
//   readout    a bound text whose fields read references.
// A widget bound to the script is disabled, with the reason as its tooltip,
// when no script is attached, when the connection takes no control or when
// the script has no such parameter, action or intervention point, or takes no
// interventions.
//
// The widget sends through RunControl directly with the origin of the person
// (design/instruments.md, Open Points): the change is the person's own, the
// run commands' card is for agents, and the runtime has no command queue. A
// `control` event goes to the ring once per edit, when the person lets go of
// the slider or presses the button, with the value sent; the drag itself is
// rate-limited by the run control and not recorded value by value.
//
// A control bound to `value` or to a state writes the session's ValueStore
// (values.h) at once, so that references read each new value at the next
// frame; the `control` event, and the `state` event of a state, go to the ring
// once, when the edit ends. Without a session the control holds its value
// itself.
#include "spec_runtime/spec_instruments.h"
#include "spec_runtime/spec_transport.h"

#include <caliper_builder/spec/state.h>
#include <caliper_builder/text/bound_text.h>

#include "spec_runtime/spec_annotations.h"
#include "spec_runtime/label_layout.h"   // fit_with_ellipsis

#include <algorithm>
#include <array>
#include <chrono>
#include <cctype>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <map>
#include <set>
#include <string>
#include <vector>

namespace caliper_host {

namespace {

namespace cb = caliper_builder;
namespace spec = caliper_builder::spec;
using nlohmann::json;

// `text` cut to `width` with an ellipsis at the current font (label_layout.h).
std::string fit_row_text(const std::string& text, float width) {
    return fit_with_ellipsis(text, width, [](std::string_view t) {
        return ImGui::CalcTextSize(t.data(), t.data() + t.size()).x;
    });
}

// The whole of a cut text as a tooltip while the pointer is over `hover` in
// this window. A text that fits needs none.
void cut_text_tooltip(const std::string& text, const std::string& shown, const ImRect& hover) {
    if (shown == text || text.empty()) return;
    if (ImGui::IsWindowHovered(ImGuiHoveredFlags_AllowWhenBlockedByActiveItem) &&
        ImGui::IsMouseHoveringRect(hover.Min, hover.Max))
        ImGui::SetTooltip("%s", text.c_str());
}

// A value the person set that the script has not shown yet: the slider keeps
// showing it until the script's value or pending value matches it, or for a
// few seconds.
struct Held {
    double value = 0.0;
    std::chrono::steady_clock::time_point at;
};
std::map<std::string, Held>& held_values() {
    static std::map<std::string, Held> m;
    return m;
}
// A toggle's state the person set that the script has not shown yet.
struct HeldToggle {
    bool on = false;
    std::chrono::steady_clock::time_point at;
};
std::map<std::string, HeldToggle>& held_toggles() {
    static std::map<std::string, HeldToggle> m;
    return m;
}
constexpr std::chrono::seconds kHoldFor{5};

std::string key_of(const spec::Control& c) {
    return std::to_string(ImGui::GetID(c.id.c_str())) + "/" + c.id + "/" + c.bind;
}

bool same(double a, double b) { return std::fabs(a - b) <= 1e-12 * std::max(1.0, std::max(std::fabs(a), std::fabs(b))); }

void push_control_event(SpecInstrumentHost& host, const spec::Control& c, json value) {
    if (!host.events) return;
    cb::Event e;
    e.kind = cb::EventKind::control;
    e.panel = c.id;
    e.origin = "ui";
    e.fields = json{{"control", c.id}, {"bind", c.bind}, {"value", std::move(value)}};
    host.events->push(std::move(e));
}

// The bind string's target; a string that is not one reads as `value`, so
// the widget does not depend on the validator.
spec::ControlTarget target_of(const std::string& bind) {
    if (auto t = spec::parse_control_target(bind)) return *t;
    spec::ControlTarget t;
    t.kind = spec::ControlTarget::Kind::value;
    return t;
}

const cb::RunParam* find_param(const cb::RunState& st, const std::string& name) {
    for (const auto& p : st.params)
        if (p.name == name) return &p;
    return nullptr;
}

const cb::RunAction* find_action(const cb::RunState& st, const std::string& name) {
    for (const auto& a : st.actions)
        if (a.name == name) return &a;
    return nullptr;
}

// Why a widget bound to the script cannot act now, or "".
std::string why_disabled(const SpecInstrumentHost& host, const cb::RunState& st) {
    if (!host.run || !st.attached) return "No script is attached.";
    if (!st.control) return st.why_not.empty() ? "The stream takes no control." : st.why_not;
    return "";
}

// ---- values the spec holds (design/instrument-interface.md, Control Values) ----

// The values of controls without a session, by window and control.
std::map<std::string, json>& own_json() {
    static std::map<std::string, json> m;
    return m;
}
// The value a control showed when the person's drag began, by window and control.
std::map<std::string, json>& drag_starts() {
    static std::map<std::string, json> m;
    return m;
}

bool holds_value(const spec::ControlTarget& t) {
    return t.kind == spec::ControlTarget::Kind::value || t.kind == spec::ControlTarget::Kind::state;
}

// The state variable a control binds, from the spec shown.
const spec::StateVar* bound_state(const SpecInstrumentHost& host, const spec::Control& c) {
    const auto t = spec::parse_control_target(c.bind);
    if (!host.spec || !t || t->kind != spec::ControlTarget::Kind::state) return nullptr;
    for (const auto& s : host.spec->states)
        if (s.id == t->name) return &s;
    return nullptr;
}

// The value a reference reads now, or its default.
json read_reference(const SpecInstrumentHost& host, const spec::Reference& r) {
    if (r.kind == spec::Reference::Kind::state && r.target == spec::kCursorState && host.history) {
        const cb::CursorState cur = host.history->cursor();
        const std::string f = r.field.value_or("step");
        if (f == "first") return cur.first;
        if (f == "latest") return cur.latest;
        if (f == "step") return cur.step;
    }
    return host.values ? host.values->value(r) : r.fallback;
}

// The low or high bound of a slider or a range: a number, a reference's
// value, or the bound of its state.
std::optional<double> bound_of(const SpecInstrumentHost& host, const spec::Control& c, const char* key) {
    const auto ref = c.options.references.find(key);
    if (ref != c.options.references.end()) {
        const json v = read_reference(host, ref->second);
        if (v.is_number() && std::isfinite(v.get<double>())) return v.get<double>();
        return std::nullopt;
    }
    const std::optional<double>& literal = std::string_view(key) == "min" ? c.options.min : c.options.max;
    if (literal) return literal;
    if (const spec::StateVar* s = bound_state(host, c)) return std::string_view(key) == "min" ? s->min : s->max;
    return std::nullopt;
}

// The value a control starts with when nothing holds one.
json start_of(const SpecInstrumentHost& host, const spec::Control& c) {
    if (const spec::StateVar* s = bound_state(host, c)) return spec::state_initial_value(*s);
    const spec::ControlOptions& o = c.options;
    if (o.value) return *o.value;
    if (!o.value_other.is_null()) return o.value_other;
    switch (c.kind) {
        case spec::ControlKind::slider: return bound_of(host, c, "min").value_or(0.0);
        case spec::ControlKind::range: {
            const double lo = bound_of(host, c, "min").value_or(0.0);
            return json::array({lo, bound_of(host, c, "max").value_or(lo)});
        }
        case spec::ControlKind::toggle: return false;
        case spec::ControlKind::text: return "";
        case spec::ControlKind::picker:
        case spec::ControlKind::segmented:
            if (o.multiple && *o.multiple) return json::array();
            return o.choices.empty() ? json() : o.choices.front().value;
        default: return nullptr;
    }
}

// The value a control bound to `value` or to a state shows now.
json current(const SpecInstrumentHost& host, const spec::Control& c) {
    if (host.values) {
        json v = host.values->control_value(c.id);
        if (!v.is_null()) return v;
    }
    const auto it = own_json().find(key_of(c));
    return it != own_json().end() ? it->second : start_of(host, c);
}

// Writes the value of a control bound to `value` or to a state. While a drag
// goes on (`end` false) the value goes to the session without an event; at
// the end of the edit the value goes again with its events, from the value
// the drag began with, so that the `state` event gives the right previous
// value. Returns whether the session took it.
bool write(SpecInstrumentHost& host, const spec::Control& c, const json& v, bool end) {
    const std::string key = key_of(c);
    if (!end && !drag_starts().count(key)) drag_starts()[key] = current(host, c);
    if (end) {
        const auto start = drag_starts().find(key);
        if (start != drag_starts().end()) {
            if (host.values) host.values->set_control(c.id, start->second, cb::CommandOrigin::ui, nullptr);
            drag_starts().erase(start);
        }
    }
    if (host.values) {
        const cb::Error e = host.values->set_control(c.id, v, cb::CommandOrigin::ui, end ? host.events : nullptr);
        return e.ok();
    }
    own_json()[key] = v;
    if (end) push_control_event(host, c, v);
    return true;
}

std::string choice_text(const spec::Choice& ch) {
    if (ch.label) return *ch.label;
    return ch.value.is_string() ? ch.value.get<std::string>() : ch.value.dump();
}

bool same_json(const json& a, const json& b) {
    if (a.is_number() && b.is_number()) return a.get<double>() == b.get<double>();
    return a == b;
}

struct Style {
    int colors = 0;
    int vars = 0;
    bool font = false;
    Style(const SpecLook& look) {
        if (look.ui) {
            ImGui::PushFont(look.ui, look.ui_size * look.text_scale);
            font = true;
        }
        const auto col = [&](ImGuiCol c, ImU32 v) {
            ImGui::PushStyleColor(c, v);
            ++colors;
        };
        col(ImGuiCol_Text, look.text);
        col(ImGuiCol_TextDisabled, look.faint);
        col(ImGuiCol_FrameBg, look.plate);
        col(ImGuiCol_FrameBgHovered, look.plate_hover);
        col(ImGuiCol_FrameBgActive, look.plate_hover);
        col(ImGuiCol_SliderGrab, look.accent);
        col(ImGuiCol_SliderGrabActive, look.line);
        col(ImGuiCol_Button, look.plate);
        col(ImGuiCol_ButtonHovered, look.plate_hover);
        col(ImGuiCol_ButtonActive, look.rule);
        col(ImGuiCol_Border, look.rule_faint);
        ImGui::PushStyleVar(ImGuiStyleVar_FrameBorderSize, 1.0f);
        ++vars;
    }
    ~Style() {
        ImGui::PopStyleVar(vars);
        ImGui::PopStyleColor(colors);
        if (font) ImGui::PopFont();
    }
};

// The rectangle of the widget: the row's right part, one frame high, in the
// middle of the row; the title takes the left part.
struct Row {
    ImRect title, widget;
};
Row row_of(const ImRect& rect, const SpecLook& look, const std::string& title, bool with_title) {
    const float pad = std::round(6.0f * look.text_scale);
    const float frame_h = ImGui::GetFrameHeight();
    const float y = std::round(rect.Min.y + std::max(0.0f, (rect.GetHeight() - frame_h) * 0.5f));
    Row r;
    float x = rect.Min.x + pad;
    if (with_title) {
        const float want = ImGui::CalcTextSize(title.c_str()).x + 2.0f * pad;
        const float w = std::min(want, std::max(0.0f, rect.GetWidth() * 0.35f));
        r.title = ImRect(x, y, x + w, y + frame_h);
        x += w;
    }
    r.widget = ImRect(x, y, std::max(x + 1.0f, rect.Max.x - pad), y + frame_h);
    return r;
}

void draw_title(const Row& row, const std::string& title, const SpecLook& look) {
    if (row.title.GetWidth() <= 0.0f) return;
    ImDrawList* dl = ImGui::GetWindowDrawList();
    const float ty = row.title.Min.y + (row.title.GetHeight() - ImGui::GetTextLineHeight()) * 0.5f;
    // A title wider than its part of the row ends in an ellipsis before the
    // widget's gap, and a hover shows it whole.
    const float gap = std::round(6.0f * look.text_scale);
    const std::string shown = fit_row_text(title, std::max(0.0f, row.title.GetWidth() - gap));
    dl->PushClipRect(row.title.Min, row.title.Max, true);
    dl->AddText(ImVec2(row.title.Min.x, ty), look.secondary, shown.c_str());
    dl->PopClipRect();
    cut_text_tooltip(title, shown, row.title);
}

void disabled_tooltip(const std::string& why) {
    if (!why.empty() && ImGui::IsItemHovered(ImGuiHoveredFlags_AllowWhenDisabled)) ImGui::SetTooltip("%s", why.c_str());
}

// The largest size of a number a slider takes, as the spec validator allows
// (validate.cpp). A script's own range and value are not validated, so every
// number is clamped to it here: ImGui asserts on a range beyond DBL_MAX / 2,
// and an integer parameter's value is converted to an int64.
constexpr double kSliderLimit = 1e15;

double bounded(double v, double fallback) {
    return std::isfinite(v) ? std::clamp(v, -kSliderLimit, kSliderLimit) : fallback;
}

double snap(double v, double lo, const std::optional<double>& step) {
    if (!step || *step <= 0.0) return v;
    return lo + std::round((v - lo) / *step) * *step;
}

bool draw_slider(const spec::Control& c, const spec::ControlTarget& t, const Row& row, SpecInstrumentHost& host) {
    const std::string key = key_of(c);
    double lo = c.options.min.value_or(0.0), hi = c.options.max.value_or(1.0);
    bool log = c.options.log.value_or(false);
    double shown = 0.0;
    std::string why;
    const cb::RunParam* param = nullptr;
    cb::RunState st;
    if (t.kind == spec::ControlTarget::Kind::param) {
        st = host.run ? host.run->state() : cb::RunState{};
        why = why_disabled(host, st);
        param = why.empty() ? find_param(st, t.name) : nullptr;
        if (why.empty() && !param) why = "The script " + st.script + " has no parameter " + t.name + ".";
        if (param && param->type != "number" && param->type != "integer") {
            why = "The parameter " + t.name + " is not a number.";
            param = nullptr;
        }
        if (param) {
            // The parameter's own range and scale where the options set none;
            // the script clamps, so the slider shows at most its range.
            if (!c.options.min && param->min) lo = *param->min;
            if (!c.options.max && param->max) hi = *param->max;
            if (!c.options.log) log = param->log;
            if (!c.options.min && !param->min) lo = std::min(0.0, param->value.is_number() ? param->value.get<double>() : 0.0);
            if (!c.options.max && !param->max)
                hi = std::max(lo + 1.0, param->value.is_number() ? 2.0 * std::fabs(param->value.get<double>()) : 1.0);
            const json& now = param->pending.is_number() ? param->pending : param->value;
            shown = now.is_number() ? now.get<double>() : lo;
            auto& held = held_values();
            const auto h = held.find(key);
            if (h != held.end()) {
                const bool confirmed = same(h->second.value, shown) ||
                                       (param->value.is_number() && same(h->second.value, param->value.get<double>()));
                if (confirmed || std::chrono::steady_clock::now() - h->second.at > kHoldFor) held.erase(h);
                else shown = h->second.value;
            }
        }
    } else {
        // Bound to `value` or to a state: the bounds may read references or
        // the state, and the session holds the value.
        lo = bound_of(host, c, "min").value_or(0.0);
        hi = bound_of(host, c, "max").value_or(std::max(lo + 1.0, 1.0));
        const json now = current(host, c);
        shown = now.is_number() ? now.get<double>() : lo;
    }
    lo = bounded(lo, 0.0);
    hi = bounded(hi, lo + 1.0);
    shown = bounded(shown, lo);
    if (log && lo <= 0.0) log = false;   // a log scale needs a positive range
    if (hi <= lo) hi = lo + 1.0;

    ImGui::SetCursorScreenPos(row.widget.Min);
    ImGui::SetNextItemWidth(row.widget.GetWidth());
    const bool disabled = !why.empty();
    ImGui::BeginDisabled(disabled);
    double v = shown;
    ImGuiSliderFlags flags = ImGuiSliderFlags_AlwaysClamp;
    if (log) flags |= ImGuiSliderFlags_Logarithmic;
    const spec::StateVar* state = bound_state(host, c);
    const bool integer = (param && param->type == "integer") || (state && state->type == spec::StateType::integer);
    // ImGui takes the smallest value a log slider reaches from the format's
    // precision (10^-precision), so the format carries enough digits for the
    // low end of the range: lr from 1e-5 needs six.
    std::string format = integer ? "%.0f" : "%.4g";
    if (log && !integer)
        format = "%." + std::to_string(std::clamp((int)std::ceil(-std::log10(lo)) + 1, 3, 12)) + "g";
    bool changed = ImGui::SliderScalar("##value", ImGuiDataType_Double, &v, &lo, &hi, format.c_str(), flags);
    const bool done = ImGui::IsItemDeactivatedAfterEdit();
    ImGui::EndDisabled();
    disabled_tooltip(why);
    if (disabled) return false;
    if (changed) {
        v = std::clamp(bounded(snap(v, lo, c.options.step), lo), lo, hi);   // a tiny step can overflow
        if (integer) v = std::round(v);
        changed = !same(v, shown);
    }
    const json sent = integer ? json(static_cast<std::int64_t>(v)) : json(v);
    if (param) {
        if (changed) {
            host.run->set_param(t.name, sent);
            held_values()[key] = Held{v, std::chrono::steady_clock::now()};
        }
        if (done) push_control_event(host, c, sent);
        return changed;
    }
    if (changed && !done) write(host, c, sent, false);
    if (done) write(host, c, sent, true);
    return changed;
}

bool draw_button(const spec::Control& c, const spec::ControlTarget& t, const Row& row, SpecInstrumentHost& host) {
    const cb::RunState st = host.run ? host.run->state() : cb::RunState{};
    std::string why = why_disabled(host, st);
    const std::string label = spec_button_label(c, st);
    const bool pause = st.state != "paused";
    if (t.kind == spec::ControlTarget::Kind::action) {
        if (why.empty() && !find_action(st, t.name)) why = "The script " + st.script + " has no action " + t.name + ".";
    } else if (t.kind == spec::ControlTarget::Kind::run_pause) {
        if (why.empty() && (st.state == "none" || !st.step))
            why = "The script has not made its first step; a run pauses only at a step.";
    } else {
        why = "A button binds an action or run:pause.";
    }
    ImGui::SetCursorScreenPos(row.widget.Min);
    const bool disabled = !why.empty();
    ImGui::BeginDisabled(disabled);
    const bool pressed = ImGui::Button((label + "###button").c_str(), row.widget.GetSize());
    ImGui::EndDisabled();
    disabled_tooltip(why);
    if (!disabled && t.kind == spec::ControlTarget::Kind::action && ImGui::IsItemHovered())
        ImGui::SetTooltip("A click runs the action %s of %s.", t.name.c_str(), st.script.c_str());
    if (!pressed || disabled) return false;
    if (t.kind == spec::ControlTarget::Kind::action) {
        host.run->run_action(t.name);
        push_control_event(host, c, true);
    } else {
        host.run->pause(pause ? cb::RunPauseOp::pause : cb::RunPauseOp::resume, 1);
        push_control_event(host, c, pause ? "pause" : "resume");
    }
    return true;
}

// The intervention of a toggle's point that the script holds and has not
// ended, or null.
const cb::Intervention* find_intervention(const std::vector<cb::Intervention>& ivs, const std::string& point) {
    for (const auto& i : ivs)
        if (i.point == point && i.op == "ablate" && i.state != "ended") return &i;
    return nullptr;
}

bool draw_toggle(const spec::Control& c, const spec::ControlTarget& t, const Row& row, const SpecLook& look,
                 SpecInstrumentHost& host) {
    const cb::RunState st = host.run ? host.run->state() : cb::RunState{};
    std::string why = why_disabled(host, st);
    bool script_on = false;
    std::string text;
    if (t.kind == spec::ControlTarget::Kind::intervene) {
        std::vector<cb::Intervention> ivs;
        if (why.empty() && !host.run->interventions_supported())
            why = "The script " + st.script + " takes no interventions; it announces them under caliper.watch(model, "
                  "optimizer).";
        if (why.empty()) {
            const auto points = host.run->intervention_points();
            bool known = points.empty();   // before the script lists them, it decides
            for (const auto& p : points) known = known || p.point == t.name;
            if (!known) why = "The model has no point " + t.name + ".";
            ivs = host.run->interventions();
        }
        const cb::Intervention* iv = find_intervention(ivs, t.name);
        script_on = iv != nullptr;
        text = "ablate " + t.name;
        if (iv && iv->state == "active" && iv->since_step)
            text += "  \xC2\xB7 since step " + std::to_string(*iv->since_step);   // ·
        else if (iv)
            text += "  \xC2\xB7 at the next step";
    } else if (t.kind == spec::ControlTarget::Kind::param) {
        const cb::RunParam* p = st.attached ? find_param(st, t.name) : nullptr;
        if (why.empty() && !p) why = "The script " + st.script + " has no parameter " + t.name + ".";
        if (why.empty() && p->type != "boolean") why = "A toggle binds a boolean parameter; " + t.name + " is not one.";
        if (p) {
            const nlohmann::json& v = p->pending.is_boolean() ? p->pending : p->value;
            script_on = v.is_boolean() && v.get<bool>();
        }
        text = t.name;
    } else {
        why = "A toggle binds intervene:<point> or param:<name> of a boolean parameter.";
    }
    // The person's click shows until the script shows it, or for a few seconds.
    const std::string key = key_of(c);
    const auto now = std::chrono::steady_clock::now();
    bool on = script_on;
    if (const auto it = held_toggles().find(key); it != held_toggles().end()) {
        if (it->second.on == script_on || now - it->second.at > kHoldFor)
            held_toggles().erase(it);
        else
            on = it->second.on;
    }
    const bool disabled = !why.empty();
    ImGui::SetCursorScreenPos(row.widget.Min);
    ImGui::BeginDisabled(disabled);
    const bool pressed = ImGui::InvisibleButton("###toggle", row.widget.GetSize());
    const bool hovered = ImGui::IsItemHovered();
    ImGui::EndDisabled();
    disabled_tooltip(why);
    // The switch: a track with a knob at its left (off) or right (on), then the text.
    ImDrawList* dl = ImGui::GetWindowDrawList();
    const float h = std::round(ImGui::GetFrameHeight() * 0.7f);
    const float w = std::round(h * 1.8f);
    const ImVec2 a(row.widget.Min.x, std::round(row.widget.GetCenter().y - h * 0.5f));
    const ImVec2 b(a.x + w, a.y + h);
    const ImU32 track = disabled ? look.plate : on ? look.accent : hovered ? look.plate_hover : look.plate;
    dl->AddRectFilled(a, b, track, h * 0.5f);
    dl->AddRect(a, b, look.rule_faint, h * 0.5f);
    const float r = h * 0.5f - 2.0f;
    dl->AddCircleFilled(ImVec2(on ? b.x - h * 0.5f : a.x + h * 0.5f, a.y + h * 0.5f), r,
                        disabled ? look.faint : look.text);
    const float tx = b.x + std::round(6.0f * look.text_scale);
    const std::string shown = fit_row_text(text, std::max(0.0f, row.widget.Max.x - tx));
    dl->PushClipRect(ImVec2(tx, row.widget.Min.y), row.widget.Max, true);
    dl->AddText(ImVec2(tx, row.widget.GetCenter().y - ImGui::GetTextLineHeight() * 0.5f),
                disabled ? look.faint : look.secondary, shown.c_str());
    dl->PopClipRect();
    cut_text_tooltip(text, shown, ImRect(ImVec2(tx, row.widget.Min.y), row.widget.Max));
    if (!pressed || disabled) return false;
    const bool want = !on;
    cb::Error e;
    if (t.kind == spec::ControlTarget::Kind::intervene) {
        if (want) {
            cb::Intervention i;
            i.point = t.name;
            i.op = "ablate";
            e = host.run->intervene_set(i);
        } else {
            e = host.run->intervene_clear(t.name);
        }
    } else {
        e = host.run->set_param(t.name, want);
    }
    if (e) return false;
    held_toggles()[key] = HeldToggle{want, now};
    push_control_event(host, c, want);
    return true;
}

// ---- the kinds that hold a value (design/instrument-interface.md, Control Kinds) ----

// A toggle bound to `value` or to a state: the switch of draw_toggle() over a
// boolean that the session holds.
bool draw_value_toggle(const spec::Control& c, const Row& row, const SpecLook& look, SpecInstrumentHost& host) {
    const json now = current(host, c);
    const bool on = now.is_boolean() && now.get<bool>();
    ImGui::SetCursorScreenPos(row.widget.Min);
    const bool pressed = ImGui::InvisibleButton("###toggle", row.widget.GetSize());
    const bool hovered = ImGui::IsItemHovered();
    ImDrawList* dl = ImGui::GetWindowDrawList();
    const float h = std::round(ImGui::GetFrameHeight() * 0.7f);
    const float w = std::round(h * 1.8f);
    const ImVec2 a(row.widget.Min.x, std::round(row.widget.GetCenter().y - h * 0.5f));
    const ImVec2 b(a.x + w, a.y + h);
    dl->AddRectFilled(a, b, on ? look.accent : hovered ? look.plate_hover : look.plate, h * 0.5f);
    dl->AddRect(a, b, look.rule_faint, h * 0.5f);
    dl->AddCircleFilled(ImVec2(on ? b.x - h * 0.5f : a.x + h * 0.5f, a.y + h * 0.5f), h * 0.5f - 2.0f, look.text);
    const float tx = b.x + std::round(6.0f * look.text_scale);
    dl->AddText(ImVec2(tx, row.widget.GetCenter().y - ImGui::GetTextLineHeight() * 0.5f), look.secondary,
                on ? "on" : "off");
    if (!pressed) return false;
    return write(host, c, !on, true);
}

// The x of `v` on the track [x0, x1] for the bounds [lo, hi].
float track_x(double v, double lo, double hi, float x0, float x1) {
    const double t = hi > lo ? std::clamp((v - lo) / (hi - lo), 0.0, 1.0) : 0.0;
    return x0 + static_cast<float>(t) * (x1 - x0);
}

std::string number_label(double v) {
    char buf[64];
    if (std::fabs(v - std::round(v)) < 1e-9 && std::fabs(v) < 1e15) std::snprintf(buf, sizeof buf, "%.0f", v);
    else std::snprintf(buf, sizeof buf, "%.4g", v);
    return buf;
}

// Which part of a range the drag holds: the low grab, the high grab or the
// span between them, by window and control.
std::map<std::string, int>& range_grips() {
    static std::map<std::string, int> m;
    return m;
}

// A range: a track between its bounds, the span [lo, hi] in the accent
// colour with a grab at each end, and the two numbers over it. A drag moves
// the nearer grab, or the whole span when it starts between the grabs; the
// value snaps to `step` and keeps at least `min_span`.
bool draw_range(const spec::Control& c, const Row& row, const SpecLook& look, SpecInstrumentHost& host) {
    const std::optional<double> mn = bound_of(host, c, "min"), mx = bound_of(host, c, "max");
    const std::string key = key_of(c);
    ImGui::SetCursorScreenPos(row.widget.Min);
    const bool disabled = !mn || !mx;
    ImGui::BeginDisabled(disabled);
    ImGui::InvisibleButton("###range", row.widget.GetSize());
    const bool active = ImGui::IsItemActive(), activated = ImGui::IsItemActivated();
    const bool released = ImGui::IsItemDeactivated();
    ImGui::EndDisabled();
    disabled_tooltip(disabled ? "The range has no bounds yet: its min or max reads a value that is not there." : "");
    ImDrawList* dl = ImGui::GetWindowDrawList();
    const ImRect& r = row.widget;
    dl->AddRectFilled(r.Min, r.Max, look.plate);
    dl->AddRect(r.Min, r.Max, look.rule_faint);
    if (disabled) {
        dl->AddText(ImVec2(r.Min.x + 6.0f, r.GetCenter().y - ImGui::GetTextLineHeight() * 0.5f), look.faint, "no bounds");
        return false;
    }
    const double lo_b = std::min(*mn, *mx), hi_b = std::max(*mn, *mx);
    json now = current(host, c);
    double lo = lo_b, hi = hi_b;
    if (now.is_array() && now.size() == 2 && now[0].is_number() && now[1].is_number()) {
        lo = std::clamp(now[0].get<double>(), lo_b, hi_b);
        hi = std::clamp(now[1].get<double>(), lo_b, hi_b);
    }
    const float grab = std::max(4.0f, std::round(4.0f * look.text_scale));
    // The numbers [lo, hi] stand at the right of the track, as a slider's value does.
    const std::string lo_text = number_label(lo_b), hi_text = number_label(hi_b);
    const std::string& longer = lo_text.size() > hi_text.size() ? lo_text : hi_text;
    const std::string widest = longer + " \xE2\x80\x93 " + longer;
    const float label_w = ImGui::CalcTextSize(widest.c_str()).x + 2.0f * std::round(6.0f * look.text_scale);
    const float x0 = r.Min.x + grab, x1 = std::max(x0 + 1.0f, r.Max.x - grab - std::min(label_w, r.GetWidth() * 0.45f));
    bool changed = false;
    if (active) {
        const float mx_pos = ImGui::GetIO().MousePos.x;
        const double at = lo_b + (hi_b - lo_b) * std::clamp((mx_pos - x0) / std::max(1.0f, x1 - x0), 0.0f, 1.0f);
        if (activated) {
            const float xl = track_x(lo, lo_b, hi_b, x0, x1), xh = track_x(hi, lo_b, hi_b, x0, x1);
            range_grips()[key] = mx_pos > xl + grab && mx_pos < xh - grab ? 2
                                 : std::fabs(mx_pos - xl) <= std::fabs(mx_pos - xh) ? 0
                                                                                    : 1;
            drag_starts().erase(key);
            range_grips()[key + "#at"] = 0;
        }
        const int grip = range_grips()[key];
        double nlo = lo, nhi = hi;
        if (grip == 0) nlo = std::min(at, hi);
        else if (grip == 1) nhi = std::max(at, lo);
        else {
            const double span = hi - lo;
            const double d = ImGui::GetIO().MouseDelta.x / std::max(1.0f, x1 - x0) * (hi_b - lo_b);
            nlo = std::clamp(lo + d, lo_b, hi_b - span);
            nhi = nlo + span;
        }
        if (c.options.step && *c.options.step > 0) {
            nlo = lo_b + std::round((nlo - lo_b) / *c.options.step) * *c.options.step;
            nhi = lo_b + std::round((nhi - lo_b) / *c.options.step) * *c.options.step;
        }
        if (std::fabs(nlo - lo) > 1e-12 || std::fabs(nhi - hi) > 1e-12) {
            changed = write(host, c, json::array({nlo, nhi}), false);
            now = current(host, c);
            if (now.is_array() && now.size() == 2) lo = now[0].get<double>(), hi = now[1].get<double>();
        }
    }
    if (released && drag_starts().count(key)) write(host, c, current(host, c), true);
    // The track, the span and the grabs.
    const float cy = r.GetCenter().y;
    const float th = std::max(2.0f, std::round(3.0f * look.text_scale));
    dl->AddRectFilled(ImVec2(x0, cy - th * 0.5f), ImVec2(x1, cy + th * 0.5f), look.rule_faint);
    const float xl = track_x(lo, lo_b, hi_b, x0, x1), xh = track_x(hi, lo_b, hi_b, x0, x1);
    dl->AddRectFilled(ImVec2(xl, cy - th), ImVec2(xh, cy + th), look.accent);
    for (float x : {xl, xh})
        dl->AddRectFilled(ImVec2(x - grab * 0.5f, r.Min.y + 3.0f), ImVec2(x + grab * 0.5f, r.Max.y - 3.0f),
                          active ? look.line : look.accent);
    const std::string text = number_label(lo) + " \xE2\x80\x93 " + number_label(hi);   // –
    const float right = r.Max.x - std::round(6.0f * look.text_scale);
    const std::string shown = fit_row_text(text, std::max(0.0f, right - (x1 + grab)));
    const ImVec2 ts = ImGui::CalcTextSize(shown.c_str());
    dl->PushClipRect(ImVec2(x1 + grab, r.Min.y), r.Max, true);
    dl->AddText(ImVec2(right - ts.x, cy - ts.y * 0.5f), look.text, shown.c_str());
    dl->PopClipRect();
    cut_text_tooltip(text, shown, ImRect(ImVec2(x1 + grab, r.Min.y), r.Max));
    return changed;
}

// A picker: a menu of its choices, the chosen one shown.
bool draw_picker(const spec::Control& c, const Row& row, SpecInstrumentHost& host) {
    const json now = current(host, c);
    std::string shown = now.is_null() ? std::string("\xE2\x80\x94") : now.is_string() ? now.get<std::string>() : now.dump();
    for (const auto& ch : c.options.choices)
        if (same_json(ch.value, now)) shown = choice_text(ch);
    ImGui::SetCursorScreenPos(row.widget.Min);
    ImGui::SetNextItemWidth(row.widget.GetWidth());
    bool changed = false;
    if (ImGui::BeginCombo("##picker", shown.c_str())) {
        for (std::size_t i = 0; i < c.options.choices.size(); ++i) {
            const spec::Choice& ch = c.options.choices[i];
            ImGui::PushID(static_cast<int>(i));
            const bool on = same_json(ch.value, now);
            if (ImGui::Selectable(choice_text(ch).c_str(), on) && !on) changed = write(host, c, ch.value, true);
            ImGui::PopID();
        }
        ImGui::EndCombo();
    }
    return changed;
}

struct TextEdit {
    std::string text;
    std::optional<std::string> submitted;
    std::string refusal;
    cb::RunControl* run = nullptr;
    cb::ValueStore* values = nullptr;
    std::string session;
    unsigned epoch = 0;
    bool retained = false;
};

// A context owns its drafts. Context shutdown is a deliberate teardown.
std::map<std::string, TextEdit>& text_edits() {
    static std::map<ImGuiContext*, std::map<std::string, TextEdit>> contexts;
    auto [it, added] = contexts.try_emplace(ImGui::GetCurrentContext());
    if (added) {
        ImGuiContextHook hook;
        hook.Type = ImGuiContextHookType_Shutdown;
        hook.Callback = [](ImGuiContext* context, ImGuiContextHook*) { contexts.erase(context); };
        ImGui::AddContextHook(ImGui::GetCurrentContext(), &hook);
    }
    return it->second;
}

// A text field: the person's edit applies on Return or when the field loses
// focus. Parameter edits over max_length are refused without truncation.
bool draw_text_field(const spec::Control& c, const Row& row, SpecInstrumentHost& host) {
    const std::string key = key_of(c);
    const auto target = spec::parse_control_target(c.bind);
    const bool script = target && target->kind == spec::ControlTarget::Kind::param;
    const cb::RunState state = script && host.run ? host.run->state() : cb::RunState{};
    const auto* param = script ? find_param(state, target->name) : nullptr;
    std::string why = script ? why_disabled(host, state) : std::string();
    if (script && why.empty() && (!param || param->type != "text")) why = "The script has no text parameter with this name.";
    const json now = param ? (param->pending.is_null() ? param->value : param->pending) : current(host, c);
    const std::size_t requested_limit = c.options.max_length
                                  ? static_cast<std::size_t>(std::max<std::int64_t>(*c.options.max_length, 0))
                                  : (script ? spec::limits::kParamTextBytes : spec::limits::kStateTextBytes);
    const std::size_t limit = script ? std::min(requested_limit, spec::limits::kParamTextBytes) : requested_limit;
    // Keep the full parameter draft. Fixed value and state buffers retain their limits.
    auto& edits = text_edits();
    auto it = edits.find(key);
    const std::string session = script && host.run ? host.run->control_session() : std::string();
    unsigned epoch = 0;
    if (it != edits.end() && script && (!state.attached || it->second.run != host.run ||
        it->second.values != host.values || it->second.session != session)) {
        epoch = it->second.epoch + 1;
        edits.erase(it);
        it = edits.end();
    }
    if (it == edits.end()) {
        std::string start = now.is_string() ? now.get<std::string>() : std::string();
        if (!script && start.size() > limit) {
            auto end = limit;
            while (end && (static_cast<unsigned char>(start[end]) & 0xc0) == 0x80) --end;
            start.resize(end);
        }
        if (!script) start.resize(limit + 1, '\0');
        it = edits.emplace(key, TextEdit{std::move(start), {}, {}, host.run, host.values, session, epoch}).first;
    }
    TextEdit& edit = it->second;
    if (script && !edit.retained) edit.text = now.is_string() ? now.get<std::string>() : std::string();
    if (script && host.run) {
        const auto error = host.run->param_error(target->name);
        if (!error.empty()) {
            edit.refusal = error;
            edit.submitted.reset();
        }
    }
    std::string& buf = edit.text;
    const bool invalid = script && buf.size() > limit;
    const std::string refusal = invalid ? "Text exceeds " + std::to_string(limit) + " bytes. The draft was retained."
                                        : edit.refusal;
    const float notice_width = !refusal.empty() ? std::min(row.widget.GetWidth() * 0.6f,
        ImGui::CalcTextSize(refusal.c_str()).x + ImGui::GetStyle().ItemSpacing.x) : 0.0f;
    ImGui::SetCursorScreenPos(row.widget.Min);
    ImGui::SetNextItemWidth(row.widget.GetWidth() - notice_width);
    const std::string hint = c.options.placeholder.value_or("");
    ImGui::PushID(static_cast<int>(edit.epoch));
    ImGui::BeginDisabled(!why.empty());
    auto resize = [](ImGuiInputTextCallbackData* data) {
        auto* text = static_cast<std::string*>(data->UserData);
        text->resize(static_cast<std::size_t>(data->BufTextLen));
        data->Buf = text->data();
        return 0;
    };
    const bool enter = script
        ? ImGui::InputTextWithHint("##text", hint.c_str(), buf.data(), buf.capacity() + 1,
            ImGuiInputTextFlags_EnterReturnsTrue | ImGuiInputTextFlags_CallbackResize, resize, &buf)
        : ImGui::InputTextWithHint("##text", hint.c_str(), buf.data(), buf.size(), ImGuiInputTextFlags_EnterReturnsTrue);
    if (script && buf.size() > limit && ImGui::IsItemHovered())
        ImGui::SetTooltip("Text accepts at most %zu UTF-8 bytes. The value was not sent.", limit);
    const bool done = enter || ImGui::IsItemDeactivatedAfterEdit();
    const bool editing = ImGui::IsItemActive();
    ImGui::EndDisabled();
    disabled_tooltip(why);
    bool changed = false;
    if (done && why.empty()) {
        const std::string value(buf.c_str());
        if (script) {
            if (value.size() <= limit && (value != now || !edit.refusal.empty()) && edit.submitted != value) {
                const cb::Error error = host.run->set_param(target->name, value);
                edit.refusal = error ? (error.message.empty() ? error.code : error.message) : std::string();
                if (!error) {
                    edit.submitted = value;
                    changed = true;
                    push_control_event(host, c, value);
                }
            }
        } else changed = write(host, c, value, true);
    }
    ImGui::PopID();
    if (!refusal.empty()) {
        const auto shown = fit_row_text(refusal, notice_width - ImGui::GetStyle().ItemSpacing.x);
        ImGui::GetWindowDrawList()->AddText(ImVec2(row.widget.Max.x - notice_width,
            row.widget.Min.y + ImGui::GetStyle().FramePadding.y), ImGui::GetColorU32(ImGuiCol_Text), shown.c_str());
        cut_text_tooltip(refusal, shown, ImRect(ImVec2(row.widget.Max.x - notice_width, row.widget.Min.y), row.widget.Max));
    }
    if (script) {
        edit.retained = editing || buf.size() > limit || !edit.refusal.empty() || buf != now;
        if (!edit.retained) edit.submitted.reset();
    } else if (!editing) edits.erase(key);
    return changed;
}

// A segmented group: one square button for each choice, the ones that are on
// in the accent colour. Without `multiple` a click chooses one; with it a
// click turns its choice on or off.
bool draw_segmented(const spec::Control& c, const Row& row, const SpecLook& look, SpecInstrumentHost& host) {
    const json now = current(host, c);
    const bool multiple = c.options.multiple && *c.options.multiple;
    const std::size_t n = c.options.choices.size();
    if (n == 0) return false;
    const float w = row.widget.GetWidth() / static_cast<float>(n);
    bool changed = false;
    ImDrawList* dl = ImGui::GetWindowDrawList();
    for (std::size_t i = 0; i < n; ++i) {
        const spec::Choice& ch = c.options.choices[i];
        bool on = false;
        if (multiple && now.is_array()) {
            for (const auto& v : now) on = on || same_json(v, ch.value);
        } else {
            on = same_json(now, ch.value);
        }
        const ImVec2 a(std::round(row.widget.Min.x + w * static_cast<float>(i)), row.widget.Min.y);
        const ImVec2 b(std::round(row.widget.Min.x + w * static_cast<float>(i + 1)), row.widget.Max.y);
        ImGui::SetCursorScreenPos(a);
        ImGui::PushID(static_cast<int>(i));
        const bool pressed = ImGui::InvisibleButton("###segment", ImVec2(std::max(1.0f, b.x - a.x), b.y - a.y));
        const bool hovered = ImGui::IsItemHovered();
        ImGui::PopID();
        // A label wider than its segment ends in an ellipsis, and a hover
        // shows it whole above the choice's shortcut.
        const std::string label = choice_text(ch);
        const std::string shown = fit_row_text(label, std::max(0.0f, b.x - a.x - 4.0f));
        if (hovered && ch.key) {
            std::string k = *ch.key;
            const bool shift = k.rfind("shift+", 0) == 0;
            if (shift) k = k.substr(6);
            for (char& ch2 : k) ch2 = static_cast<char>(std::toupper(static_cast<unsigned char>(ch2)));
            if (shown != label)
                ImGui::SetTooltip("%s\nShortcut: %s%s", label.c_str(), shift ? "Shift+" : "", k.c_str());
            else
                ImGui::SetTooltip("Shortcut: %s%s", shift ? "Shift+" : "", k.c_str());
        } else if (hovered && shown != label) {
            ImGui::SetTooltip("%s", label.c_str());
        }
        dl->AddRectFilled(a, b, on ? look.accent : hovered ? look.plate_hover : look.plate);
        dl->AddRect(a, b, look.rule_faint);
        const ImVec2 ts = ImGui::CalcTextSize(shown.c_str());
        dl->PushClipRect(a, b, true);
        dl->AddText(ImVec2(std::round((a.x + b.x - ts.x) * 0.5f), std::round((a.y + b.y - ts.y) * 0.5f)),
                    on ? look.ground : look.secondary, shown.c_str());
        dl->PopClipRect();
        if (!pressed) continue;
        if (!multiple) {
            if (!on) changed = write(host, c, ch.value, true) || changed;
            continue;
        }
        json list = json::array();
        for (const auto& other : c.options.choices) {
            bool keep = false;
            if (now.is_array())
                for (const auto& v : now) keep = keep || same_json(v, other.value);
            if (&other == &ch) keep = !on;
            if (keep) list.push_back(other.value);
        }
        changed = write(host, c, list, true) || changed;
    }
    return changed;
}

// A readout: its bound text, with each field read from its reference or, for
// a field that names a source, from the source's kept value at the cursor
// (spec_annotations.h). A format that does not parse shows as written, in the
// faint colour, with the reason as its tooltip.
void draw_readout(const spec::Control& c, const Row& row, const SpecLook& look, const SpecInstrumentHost& host) {
    ImDrawList* dl = ImGui::GetWindowDrawList();
    const float ty = row.widget.GetCenter().y - ImGui::GetTextLineHeight() * 0.5f;
    if (!c.options.bound_text) {
        dl->AddText(ImVec2(row.widget.Min.x, ty), look.faint, "\xE2\x80\x94");
        return;
    }
    const spec::BoundText& bt = *c.options.bound_text;
    std::vector<std::string> names;
    for (const auto& [name, field] : bt.fields) names.push_back(name);
    const std::map<std::string, json> values = spec_bound_text_values(bt, host);
    caliper_builder::text::ParsedText parsed;
    const cb::Error e = caliper_builder::text::parse_bound_text(bt.format, names, &parsed);
    const std::string text = e ? bt.format : caliper_builder::text::format_bound_text(parsed, values);
    // A text wider than the row ends in an ellipsis, and a hover shows it whole.
    const std::string shown = fit_row_text(text, row.widget.GetWidth());
    ImGui::SetCursorScreenPos(row.widget.Min);
    ImGui::InvisibleButton("###readout", row.widget.GetSize());
    if (ImGui::IsItemHovered()) {
        if (e) ImGui::SetTooltip("%s", e.message.c_str());
        else if (shown != text) ImGui::SetTooltip("%s", text.c_str());
    }
    dl->PushClipRect(row.widget.Min, row.widget.Max, true);
    dl->AddText(ImVec2(row.widget.Min.x, ty), e ? look.faint : look.text, shown.c_str());
    dl->PopClipRect();
}

}  // namespace

std::string spec_button_label(const caliper_builder::spec::Control& control, const caliper_builder::RunState& state) {
    const spec::ControlTarget t = target_of(control.bind);
    if (t.kind == spec::ControlTarget::Kind::run_pause) return state.state != "paused" ? "Pause" : "Resume";
    std::string label = control.title.value_or(control.id);
    if (t.kind != spec::ControlTarget::Kind::action) return label;
    const cb::RunAction* a = state.attached ? find_action(state, t.name) : nullptr;
    if (a && (a->queued || a->running)) label += a->running ? "  \xC2\xB7 running" : "  \xC2\xB7 queued";   // ·
    return label;
}

float spec_control_height(const caliper_builder::spec::Control& control, const SpecLook& look) {
    (void)control;   // every kind takes one row
    return std::round(look.ui_size * look.text_scale * 2.0f);
}

bool draw_spec_control(const caliper_builder::spec::Control& control, const ImRect& rect, const SpecLook& look,
                       SpecInstrumentHost& host) {
    if (rect.GetWidth() <= 1.0f || rect.GetHeight() <= 1.0f) return false;
    const ImVec2 cursor = ImGui::GetCursorScreenPos();
    ImGui::PushID(control.id.c_str());
    bool changed = false;
    {
        Style style(look);
        const std::string title = control.title.value_or(control.id);
        const spec::ControlTarget t = target_of(control.bind);
        const bool button = control.kind == spec::ControlKind::button;
        const Row row = row_of(rect, look, title, !button);
        if (!button) draw_title(row, title, look);
        if (control.kind == spec::ControlKind::slider) {
            changed = draw_slider(control, t, row, host);
        } else if (button) {
            changed = draw_button(control, t, row, host);
        } else if (control.kind == spec::ControlKind::toggle) {
            changed = holds_value(t) ? draw_value_toggle(control, row, look, host) : draw_toggle(control, t, row, look, host);
        } else if (control.kind == spec::ControlKind::range) {
            changed = draw_range(control, row, look, host);
        } else if (control.kind == spec::ControlKind::picker) {
            changed = draw_picker(control, row, host);
        } else if (control.kind == spec::ControlKind::text) {
            changed = draw_text_field(control, row, host);
        } else if (control.kind == spec::ControlKind::segmented) {
            changed = draw_segmented(control, row, look, host);
        } else if (control.kind == spec::ControlKind::readout) {
            draw_readout(control, row, look, host);
        } else if (control.kind == spec::ControlKind::transport) {
            changed = draw_transport_control(control, row.widget, look, host);   // spec_transport.h
        } else {
            ImGui::GetWindowDrawList()->AddText(row.widget.Min, look.faint,
                                                (std::string(spec::control_kind_name(control.kind)) +
                                                 " is not built yet")
                                                    .c_str());
        }
    }
    ImGui::PopID();
    ImGui::SetCursorScreenPos(cursor);
    return changed;
}

void record_spec_pick(const std::string& panel, const caliper_builder::Event& pick, SpecInstrumentHost& host) {
    if (!host.values) return;
    // The pick's fields: the event's `fields` when the panel gave them, else
    // the keys of a live view's pick (entity, index, row_id, position).
    const json& f = pick.fields.contains("fields") && pick.fields["fields"].is_object() ? pick.fields["fields"] : pick.fields;
    cb::PickFields out;
    // A whole number of the pick, when it is a finite number of a magnitude
    // of at most 2^53 - 1, so that the rounding is exact and defined.
    auto integer = [&](const char* k) -> std::optional<std::int64_t> {
        const auto it = f.find(k);
        if (it == f.end() || !it->is_number()) return std::nullopt;
        const double d = it->get<double>();
        if (!std::isfinite(d) || std::fabs(d) > 9007199254740991.0) return std::nullopt;
        return static_cast<std::int64_t>(std::llround(d));
    };
    auto number = [&](const char* k) -> std::optional<double> {
        const auto it = f.find(k);
        return it != f.end() && it->is_number() ? std::optional<double>(it->get<double>()) : std::nullopt;
    };
    if (auto it = f.find("entity"); it != f.end() && it->is_string()) out.entity = it->get<std::string>();
    out.index = integer("index");
    out.row_id = integer("row_id");
    out.row = integer("row");
    out.column = integer("column");
    out.step = integer("step");
    out.value = number("value");
    if (auto x = number("x"), y = number("y"), z = number("z"); x && y && z) {
        out.position = std::array<double, 3>{*x, *y, *z};
    } else if (auto p = f.find("position"); p != f.end() && p->is_array() && p->size() == 3) {
        bool ok = true;
        for (const auto& v : *p) ok = ok && v.is_number();
        if (ok) out.position = std::array<double, 3>{(*p)[0].get<double>(), (*p)[1].get<double>(), (*p)[2].get<double>()};
    }
    if (auto a = pick.fields.find("annotation"); a != pick.fields.end() && a->is_string()) out.annotation = a->get<std::string>();
    // A program's pick (design/programmable-graphics.md, Picking): its id stands as the entity, so
    // that a select may name the program, and the item map's fields keep their names.
    if (auto pr = pick.fields.find("program"); pr != pick.fields.end() && pr->is_string()) {
        out.program = pr->get<std::string>();
        out.entity = *out.program;
        if (auto it = pick.fields.find("item"); it != pick.fields.end() && it->is_number_integer() &&
                                                 it->get<std::int64_t>() >= 0 && it->get<std::int64_t>() <= 0xffffffffll)
            out.item = static_cast<std::int64_t>(it->get<std::uint64_t>());
        if (auto st = pick.fields.find("step"); st != pick.fields.end() && st->is_number_integer())
            out.step = st->get<std::int64_t>();
        out.index.reset();
        if (auto ix = pick.fields.find("index"); ix != pick.fields.end() && ix->is_array() && ix->size() == 4) {
            std::array<std::int64_t, 4> t{-1, -1, -1, -1};
            for (std::size_t k = 0; k < 4; ++k)
                if ((*ix)[k].is_number_integer()) t[k] = (*ix)[k].get<std::int64_t>();
            out.tensor_index = t;
        }
        if (f.is_object())
            for (auto it = f.begin(); it != f.end() && out.program_fields.size() < 64; ++it)
                if (it.value().is_number() && std::isfinite(it.value().get<double>()))
                    out.program_fields.emplace_back(it.key(), it.value().get<double>());
    }
    // The pick event is in the ring already; the state write adds its own.
    host.values->record_pick(panel, out, cb::CommandOrigin::ui, host.events);
}

}  // namespace caliper_host
