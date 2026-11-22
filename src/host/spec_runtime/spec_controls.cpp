// The controls of a spec (spec_instruments.h, design/instruments.md, Control
// Table).
//
// A control is one row: its title on the left, its widget on the right. M1
// draws two kinds:
//   slider  bound to param:<name>, it shows the script's parameter on the
//           parameter's own range and scale, narrowed by the options, and
//           sends each new value through the run control while the person
//           drags; bound to `value`, it holds its value itself.
//   button  bound to action:<name>, it queues the action; bound to
//           run:pause, it pauses a run that runs and resumes a paused one.
// A widget bound to the script is disabled, with the reason as its tooltip,
// when no script is attached, when the connection takes no control or when
// the script has no such parameter or action.
//
// The widget sends through RunControl directly with the origin of the person
// (design/instruments.md, Open Points): the change is the person's own, the
// run commands' card is for agents, and the runtime has no command queue. A
// `control` event goes to the ring once per edit, when the person lets go of
// the slider or presses the button, with the value sent; the drag itself is
// rate-limited by the run control and not recorded value by value.
#include "spec_runtime/spec_instruments.h"

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
    const char* format = integer ? "%.0f" : log ? "%.3g" : "%.4g";
    bool changed = ImGui::SliderScalar("##value", ImGuiDataType_Double, &v, &lo, &hi, format, flags);
    const bool done = ImGui::IsItemDeactivatedAfterEdit();
    ImGui::EndDisabled();
    disabled_tooltip(why);
    if (disabled) return false;
    if (changed) {
        v = snap(v, lo, c.options.step);
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

bool draw_button(const spec::Control& c, const spec::ControlTarget& t, const Row& row, const std::string& title,
                 SpecInstrumentHost& host) {
    const cb::RunState st = host.run ? host.run->state() : cb::RunState{};
    std::string why = why_disabled(host, st);
    std::string label = title;
    bool pause = true;
    if (t.kind == spec::ControlTarget::Kind::action) {
        const cb::RunAction* a = why.empty() ? find_action(st, t.name) : nullptr;
        if (why.empty() && !a) why = "The script " + st.script + " has no action " + t.name + ".";
        if (a && (a->queued || a->running)) label += a->running ? "  \xC2\xB7 running" : "  \xC2\xB7 queued";   // ·
    } else if (t.kind == spec::ControlTarget::Kind::run_pause) {
        pause = st.state != "paused";
        label = pause ? "Pause" : "Resume";
        if (why.empty() && (st.state == "none" || !st.step))
            why = "The script has not made its first step; a run pauses only at a step.";
    } else {
        why = "A button binds an action or run:pause.";
    }
    ImGui::SetCursorScreenPos(row.widget.Min);
    const bool disabled = !why.empty();
    ImGui::BeginDisabled(disabled);
    const bool pressed = ImGui::Button((label + "##button").c_str(), row.widget.GetSize());
    ImGui::EndDisabled();
    disabled_tooltip(why);
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

}  // namespace

float spec_control_height(const caliper_builder::spec::Control& control, const SpecLook& look) {
    (void)control;   // both kinds of M1 take one row
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
            changed = draw_button(control, t, row, title, host);
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

}  // namespace caliper_host
