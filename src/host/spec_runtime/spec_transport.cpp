// The time transport in the spec runtime (spec_transport.h,
// design/instrument-interface.md, Transport Control, Run Pause and Cursor).
// The control moves the cursor of the session's StepHistory and copies it
// into the value store, so that references to the state `cursor` and the
// panels that read kept steps follow it. A hold never pauses the run.
#include "spec_runtime/spec_transport.h"

#include <caliper_builder/session/events.h>
#include <caliper_builder/session/history.h>
#include <caliper_builder/session/run_control.h>
#include <caliper_builder/session/values.h>

#include "implot.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <string>

namespace caliper_host {

namespace cb = caliper_builder;
namespace spec = caliper_builder::spec;

namespace {

// The speeds of the speed menu, in steps a second.
constexpr double kSpeeds[] = {1, 2, 5, 10, 20, 50, 100, 200, 500, 1000};

// Records a move of the cursor by the person: the value store follows it,
// and a `cursor` event says how it changed (Cursor Events).
void moved(SpecInstrumentHost& host, const cb::CursorState& c, const char* change) {
    if (host.values) host.values->set_cursor(c);
    if (!host.events) return;
    cb::Event e;
    e.kind = cb::EventKind::cursor;
    e.origin = "ui";
    e.fields = {{"change", change},      {"step", c.step},       {"first", c.first},
                {"latest", c.latest},    {"playing", c.playing}, {"speed", c.speed}};
    host.events->push(std::move(e));
}

// The transport's actions, shared by the buttons and the keys.
enum class Act { back, play_hold, forward, first, live };

bool act(SpecInstrumentHost& host, Act a) {
    if (!host.history) return false;
    cb::StepHistory& h = *host.history;
    const cb::CursorState now = h.cursor();
    const bool kept = !h.kept_steps().empty();
    switch (a) {
        case Act::back:
        case Act::forward: {
            if (!kept) return false;
            moved(host, h.step_cursor(a == Act::back ? -1 : 1), "moved");
            return true;
        }
        case Act::play_hold: {
            cb::CursorState want = now;
            if (now.live || now.playing) {   // Hold: the view stays at its step
                want.live = false;
                want.playing = false;
                if (!kept) return false;
                moved(host, h.set_cursor(want), "held");
            } else {   // Play: toward the newest step
                want.playing = true;
                moved(host, h.set_cursor(want), "playing");
            }
            return true;
        }
        case Act::first: {
            if (!kept) return false;
            cb::CursorState want = now;
            want.live = false;
            want.playing = false;
            want.step = now.first;
            moved(host, h.set_cursor(want), "moved");
            return true;
        }
        case Act::live: {
            if (now.live) return false;
            cb::CursorState want = now;
            want.live = true;
            moved(host, h.set_cursor(want), "live");
            return true;
        }
    }
    return false;
}

const spec::Control* transport_of(const spec::Spec& s) {
    for (const spec::Control& c : s.controls)
        if (c.kind == spec::ControlKind::transport) return &c;
    return nullptr;
}

std::string speed_text(double speed) {
    char buf[32];
    if (speed >= 10 || std::fabs(speed - std::round(speed)) < 1e-9)
        std::snprintf(buf, sizeof buf, "%.0f/s", speed);
    else
        std::snprintf(buf, sizeof buf, "%.1f/s", speed);
    return buf;
}

}  // namespace

bool draw_transport_control(const spec::Control& control, const ImRect& rect, const SpecLook& look,
                            SpecInstrumentHost& host) {
    (void)control;
    ImDrawList* dl = ImGui::GetWindowDrawList();
    if (!host.history) {
        dl->AddText(ImVec2(rect.Min.x, rect.Min.y + (rect.GetHeight() - ImGui::GetTextLineHeight()) * 0.5f),
                    look.faint, "No kept steps in this view");
        return false;
    }
    cb::StepHistory& h = *host.history;
    const cb::CursorState c = h.cursor();
    const bool kept = !h.kept_steps().empty();
    const float pad = std::round(4.0f * look.text_scale);
    const float fh = rect.GetHeight();
    float x = rect.Min.x;
    bool changed = false;
    auto button = [&](const char* label, const char* id, float w, bool enabled, const char* tip) {
        ImGui::SetCursorScreenPos(ImVec2(x, rect.Min.y));
        ImGui::BeginDisabled(!enabled);
        const bool pressed = ImGui::Button((std::string(label) + "###" + id).c_str(), ImVec2(w, fh));
        ImGui::EndDisabled();
        if (ImGui::IsItemHovered(ImGuiHoveredFlags_AllowWhenDisabled)) ImGui::SetTooltip("%s", tip);
        x += w + pad;
        return pressed && enabled;
    };
    auto text_w = [&](const char* s) { return ImGui::CalcTextSize(s).x + 2.0f * ImGui::GetStyle().FramePadding.x; };
    const float square = fh;
    if (button("<", "back", square, kept, "Step back")) changed = act(host, Act::back);
    const bool holding = c.live || c.playing;
    const char* hold = holding ? "Hold" : "Play";
    if (button(hold, "play", std::max(text_w("Hold"), text_w("Play")), kept,
               holding ? "Hold view" : "Play the kept steps toward the newest step"))
        changed = act(host, Act::play_hold);
    if (button(">", "forward", square, kept, "Step forward")) changed = act(host, Act::forward);
    if (button("Live", "live", text_w("Live"), !c.live, "Follow the newest step")) changed = act(host, Act::live);

    // The speed menu.
    const float speed_w = text_w("1000/s") + fh;
    ImGui::SetCursorScreenPos(ImVec2(x, rect.Min.y));
    ImGui::SetNextItemWidth(speed_w);
    if (ImGui::BeginCombo("###speed", speed_text(c.speed).c_str())) {
        for (double s : kSpeeds) {
            if (ImGui::Selectable(speed_text(s).c_str(), std::fabs(s - c.speed) < 1e-9)) {
                cb::CursorState want = c;
                want.speed = s;
                h.set_cursor(want);
                if (host.values) host.values->set_cursor(h.cursor());
            }
        }
        ImGui::EndCombo();
    }
    if (ImGui::IsItemHovered()) ImGui::SetTooltip("Play speed in steps a second");
    x += speed_w + pad * 2.0f;

    // The step as text: "Live 1234", or "Held 120 / 450" while the run goes on.
    const cb::CursorState after = h.cursor();
    std::string state;
    if (!kept) state = "No kept steps yet";
    else if (after.live) state = "Live " + std::to_string(after.step);
    else state = std::string(after.playing ? "Playing " : "Held ") + std::to_string(after.step) + " / " +
                 std::to_string(after.latest);
    const float ty = rect.Min.y + (fh - ImGui::GetTextLineHeight()) * 0.5f;
    dl->PushClipRect(ImVec2(x, rect.Min.y), rect.Max, true);
    dl->AddText(ImVec2(x, ty), after.live ? look.text : look.accent, state.c_str());
    x += ImGui::CalcTextSize(state.c_str()).x + pad * 3.0f;
    // A chip while the run is paused, which the cursor never causes.
    const bool paused = host.run && host.run->state().state == "paused";
    if (paused) {
        const char* chip = "Run Paused";
        const ImVec2 sz = ImGui::CalcTextSize(chip);
        const ImVec2 a(x, ty - 2.0f), b(x + sz.x + 2.0f * pad, ty + sz.y + 2.0f);
        dl->AddRectFilled(a, b, look.plate, 3.0f);
        dl->AddRect(a, b, look.rule, 3.0f);
        dl->AddText(ImVec2(x + pad, ty), look.secondary, chip);
    }
    dl->PopClipRect();
    return changed;
}

bool handle_transport_keys(const spec::Spec& spec, SpecInstrumentHost& host) {
    const spec::Control* t = transport_of(spec);
    if (!t || !t->options.keys.value_or(true) || !host.history) return false;
    // A text field that holds the keyboard keeps its keys.
    if (ImGui::GetIO().WantTextInput) return false;
    bool changed = false;
    if (ImGui::IsKeyPressed(ImGuiKey_Space, false)) changed = act(host, Act::play_hold) || changed;
    if (ImGui::IsKeyPressed(ImGuiKey_LeftArrow, true)) changed = act(host, Act::back) || changed;
    if (ImGui::IsKeyPressed(ImGuiKey_RightArrow, true)) changed = act(host, Act::forward) || changed;
    if (ImGui::IsKeyPressed(ImGuiKey_Home, false)) changed = act(host, Act::first) || changed;
    if (ImGui::IsKeyPressed(ImGuiKey_End, false)) changed = act(host, Act::live) || changed;
    return changed;
}

void draw_cursor_line(const spec::Panel& panel, const SpecInstrumentHost& host) {
    if (panel.primitive != spec::Primitive::series || !host.history) return;
    const cb::CursorState c = host.history->cursor();
    if (c.live) return;
    const double x = static_cast<double>(c.step);
    const ImPlotRect limits = ImPlot::GetPlotLimits();
    if (x < limits.X.Min || x > limits.X.Max) return;
    ImDrawList* dl = ImPlot::GetPlotDrawList();
    const ImVec2 top = ImPlot::PlotToPixels(x, limits.Y.Max);
    const ImVec2 bottom = ImPlot::PlotToPixels(x, limits.Y.Min);
    const ImU32 color = ImGui::GetColorU32(ImVec4(0xb3 / 255.0f, 0x8b / 255.0f, 0x59 / 255.0f, 1.0f));
    ImPlot::PushPlotClipRect();
    dl->AddLine(ImVec2(std::round(top.x) + 0.5f, top.y), ImVec2(std::round(bottom.x) + 0.5f, bottom.y), color, 1.5f);
    const std::string label = std::to_string(c.step);
    dl->AddText(ImVec2(std::round(top.x) + 4.0f, top.y + 2.0f), color, label.c_str());
    ImPlot::PopPlotClipRect();
}

void tick_transport(SpecInstrumentHost& host, double seconds) {
    if (!host.history) return;
    host.history->tick(seconds);
    if (host.values) host.values->set_cursor(host.history->cursor());
}

}  // namespace caliper_host
