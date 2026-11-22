// The controls of a spec (spec_instruments.h, design/instruments.md, Control
// Table).
//
// A control is one row: its title on the left, its widget on the right. This file
// draws three kinds:
//   slider  bound to param:<name>, it shows the script's parameter on the
//           parameter's own range and scale, narrowed by the options, and
//           sends each new value through the run control while the person
//           drags; bound to `value`, it holds its value itself.
//   button  bound to action:<name>, it queues the action; bound to
//           run:pause, it pauses a run that runs and resumes a paused one.
//   toggle  bound to intervene:<point>, on it ablates that point of the
//           attached model and off it clears the point's interventions
//           (design/gpu-programs.md, Toggle Control); bound to param:<name>
//           of a boolean parameter, it sets the parameter. It shows the
//           script's state, and the person's last click until the script
//           shows it or for a few seconds.
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
#include "spec_runtime/spec_instruments.h"
#include "spec_runtime/spec_transport.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <map>
#include <string>

namespace caliper_host {

namespace {

namespace cb = caliper_builder;
namespace spec = caliper_builder::spec;
using nlohmann::json;

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
// The values of sliders bound to `value`, by window and control.
std::map<std::string, double>& own_values() {
    static std::map<std::string, double> m;
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

// The bind string's target, read here so the widget does not depend on the
// validator: param:<name>, action:<name>, run:pause or value.
spec::ControlTarget target_of(const std::string& bind) {
    spec::ControlTarget t;
    if (bind.rfind("param:", 0) == 0) {
        t.kind = spec::ControlTarget::Kind::param;
        t.name = bind.substr(6);
    } else if (bind.rfind("action:", 0) == 0) {
        t.kind = spec::ControlTarget::Kind::action;
        t.name = bind.substr(7);
    } else if (bind == "run:pause") {
        t.kind = spec::ControlTarget::Kind::run_pause;
    } else if (bind.rfind("intervene:", 0) == 0) {
        t.kind = spec::ControlTarget::Kind::intervene;
        t.name = bind.substr(10);
    } else {
        t.kind = spec::ControlTarget::Kind::value;
    }
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
    dl->PushClipRect(row.title.Min, row.title.Max, true);
    dl->AddText(ImVec2(row.title.Min.x, ty), look.secondary, title.c_str());
    dl->PopClipRect();
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
        auto& own = own_values();
        const auto it = own.find(key);
        shown = it != own.end() ? it->second : c.options.value.value_or(lo);
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
    const bool integer = param && param->type == "integer";
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
    if (changed) {
        if (param) {
            host.run->set_param(t.name, sent);
            held_values()[key] = Held{v, std::chrono::steady_clock::now()};
        } else {
            own_values()[key] = v;
        }
    }
    if (done) push_control_event(host, c, sent);
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
    dl->PushClipRect(ImVec2(tx, row.widget.Min.y), row.widget.Max, true);
    dl->AddText(ImVec2(tx, row.widget.GetCenter().y - ImGui::GetTextLineHeight() * 0.5f),
                disabled ? look.faint : look.secondary, text.c_str());
    dl->PopClipRect();
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

}  // namespace

std::string spec_button_label(const caliper_builder::spec::Control& control, const caliper_builder::RunState& state) {
    const spec::ControlTarget t = target_of(control.bind);
    if (t.kind == spec::ControlTarget::Kind::run_pause) return state.state != "paused" ? "Pause" : "Resume";
    std::string label = control.title.value_or(control.id);
    if (t.kind != spec::ControlTarget::Kind::action) return label;
    label += " (" + spec::control_target_string(t) + ")";
    const cb::RunAction* a = state.attached ? find_action(state, t.name) : nullptr;
    if (a && (a->queued || a->running)) label += a->running ? "  \xC2\xB7 running" : "  \xC2\xB7 queued";   // ·
    return label;
}

float spec_control_height(const caliper_builder::spec::Control& control, const SpecLook& look) {
    (void)control;   // a slider, a button and a toggle each take one row
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
            changed = draw_toggle(control, t, row, look, host);
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
    // Not built yet: picks write no state (design/instrument-interface.md, Pick Fields).
    (void)panel;
    (void)pick;
    (void)host;
}

}  // namespace caliper_host
