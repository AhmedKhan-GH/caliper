// live_ui_driver.h: a testing aid that plays a short script of mouse and key
// input into the current ImGui context, aimed at the run bar's widgets
// (live_controls.h, bar_widgets()) and at the live views' data areas
// (live_views_state.h, geo_area). The input goes through ImGui's own event
// queue, exactly as the platform backend would feed a person's clicks, so the
// widgets run their real click, drag and text-entry paths. The caliper_live
// tests drive the run bar with it; the app plays a script given in
// CALIPER_TEST_UI only when its window is hidden (--hidden), and never shows
// a window for it.
//
// The script is a list of steps separated by ';':
//   click <widget>        a click on "pause", "resume", "step", "action:<name>", "more"
//   set <param> <text>    ctrl+click the parameter's slider or field, type <text>, Enter
//   toggle <param>        a click on a boolean parameter's box
//   select <view>         a shift-drag across the top quarter of the view's image or plot area
//   leave                 the pointer leaves the window (no hover is left behind)
//   wait <frames>         nothing for that many frames
//   state <run state>     wait until the run state is <run state> (paused, running, ...)
//   bar [<text>]          wait until the run state the bar wrote in the last frame starts
//                         with <text> (any text without one), and log it: bar shows '<text>'
// A step whose target has not been drawn waits for it, up to kPatience frames,
// then the script stops with an error.
//
// Header-only; frame thread. Call feed() once per frame BEFORE ImGui::NewFrame.
#pragma once

#include "live/live_controls.h"
#include "live/live_views_state.h"

#include <imgui.h>

#include <algorithm>
#include <cfloat>
#include <cstdio>
#include <cstdlib>
#include <sstream>
#include <string>
#include <vector>

namespace caliper_live {

class UiDriver {
public:
    static constexpr int kPatience = 1200;   // frames a step waits for its target (20 s at 60 Hz)

    explicit UiDriver(const std::string& script) {
        std::stringstream all(script);
        std::string part;
        while (std::getline(all, part, ';')) {
            std::stringstream words(part);
            Step s;
            words >> s.verb >> s.a;
            std::getline(words, s.b);
            while (!s.b.empty() && s.b.front() == ' ') s.b.erase(s.b.begin());
            if (!s.verb.empty()) steps_.push_back(s);
        }
    }

    bool done() const { return at_ >= steps_.size() || !error_.empty(); }
    const std::string& error() const { return error_; }
    // What the driver did, one line per step, oldest first. Cleared by the call.
    std::vector<std::string> take_log() {
        std::vector<std::string> out;
        out.swap(log_);
        return out;
    }

    // Queues this frame's input. `core` finds the views; `rc` answers `state`.
    void feed(CaliperCore* core, const RunControls* rc) {
        if (done()) return;
        ImGuiIO& io = ImGui::GetIO();
        const Step& s = steps_[at_];
        ++frames_;
        if (s.verb == "wait") {
            if (frames_ >= std::atoi(s.a.c_str())) next("waited " + s.a + " frames");
            return;
        }
        if (s.verb == "leave") {
            io.AddMousePosEvent(-FLT_MAX, -FLT_MAX);
            next("left the window");
            return;
        }
        if (s.verb == "state") {
            if (rc && rc->have() && rc->state().run_state == s.a) next("run state " + s.a);
            else if (frames_ > kPatience) fail("the run state never became " + s.a);
            return;
        }
        if (s.verb == "bar") {
            const std::string want = s.a + (s.b.empty() ? "" : " " + s.b);
            const std::string& shown = bar_state_text();
            if (!shown.empty() && shown.compare(0, want.size(), want) == 0) next("bar shows '" + shown + "'");
            else if (frames_ > kPatience) fail("the bar never showed '" + want + "'; it shows '" + shown + "'");
            return;
        }
        if (s.verb == "click" || s.verb == "toggle" || s.verb == "set") {
            // A widget's name may hold spaces (a view's): click takes the rest of the step.
            const std::string name = s.verb == "click" ? s.a + (s.b.empty() ? "" : " " + s.b) : "param:" + s.a;
            if (phase_ == 0) {
                const auto& w = bar_widgets();
                const auto it = w.find(name);
                if (it == w.end()) {
                    if (frames_ > kPatience) fail("'" + name + "' was never drawn");
                    return;
                }
                // A box is clicked at its left edge (a checkbox's square).
                x_ = s.verb == "toggle" ? it->second.x0 + 6.0f : 0.5f * (it->second.x0 + it->second.x1);
                y_ = 0.5f * (it->second.y0 + it->second.y1);
                io.AddMousePosEvent(x_, y_);
                phase_ = 1;
                return;
            }
            // ImGui's "ctrl" (io.KeyCtrl) is Cmd on macOS: the event a backend
            // sends for Cmd is ImGuiMod_Super, which ImGui swaps (a physical
            // Ctrl+click would be a right click there).
            const ImGuiKey ctrl = io.ConfigMacOSXBehaviors ? ImGuiMod_Super : ImGuiMod_Ctrl;
            const bool typed = s.verb == "set";
            switch (phase_++) {
                case 1:
                    if (typed) io.AddKeyEvent(ctrl, true);
                    break;
                case 2: io.AddMouseButtonEvent(0, true); break;
                case 3: io.AddMouseButtonEvent(0, false); break;
                case 4:
                    if (!typed) next(s.verb + " " + s.a + " at " + where());
                    else io.AddKeyEvent(ctrl, false);
                    break;
                // The slider's text field opens with its text selected, so typing replaces it.
                case 5: io.AddInputCharactersUTF8(s.b.c_str()); break;
                case 6: io.AddKeyEvent(ImGuiKey_Enter, true); break;
                case 7:
                    io.AddKeyEvent(ImGuiKey_Enter, false);
                    next("set " + s.a + " to " + s.b + " at " + where());
                    break;
            }
            return;
        }
        if (s.verb == "select") {
            constexpr int kMoves = 8;
            if (phase_ == 0) {
                const std::string view = s.a + (s.b.empty() ? "" : " " + s.b);
                const caliper_live_views::View* v = core ? caliper_live_views::find(core, view.c_str()) : nullptr;
                if (!v || !v->geo_valid || v->geo_area[2] < 8.0f || v->geo_area[3] < 8.0f) {
                    if (frames_ > kPatience) fail("view '" + view + "' was never drawn");
                    return;
                }
                ax_ = v->geo_area[0];
                ay_ = v->geo_area[1];
                aw_ = v->geo_area[2];
                ah_ = v->geo_area[3];
                // A heatmap or an image: its image, which may sit inside the
                // area with margins (cells keep their aspect).
                const double* gw = v->geo_window;
                if ((v->kind == CALIPER_LIVE_KIND_HEATMAP || v->kind == CALIPER_LIVE_KIND_IMAGE) && v->zoom_w > 0 &&
                    v->zoom_h > 0 && gw[1] > gw[0] && gw[3] > gw[2]) {
                    auto sx = [&](double c) { return ax_ + (float)((c - gw[0]) / (gw[1] - gw[0])) * aw_; };
                    auto sy = [&](double r) { return ay_ + (float)((r - gw[2]) / (gw[3] - gw[2])) * ah_; };
                    const float x0 = std::max(ax_, sx(0.0)), x1 = std::min(ax_ + aw_, sx((double)v->zoom_w));
                    const float y0 = std::max(ay_, sy(0.0)), y1 = std::min(ay_ + ah_, sy((double)v->zoom_h));
                    if (x1 - x0 >= 8.0f && y1 - y0 >= 8.0f) {
                        ax_ = x0;
                        ay_ = y0;
                        aw_ = x1 - x0;
                        ah_ = y1 - y0;
                    }
                }
                x_ = ax_ + 0.02f * aw_;
                y_ = ay_ + 0.02f * ah_;
                io.AddMousePosEvent(x_, y_);
                phase_ = 1;
                return;
            }
            if (phase_ == 1) {
                io.AddKeyEvent(ImGuiMod_Shift, true);
            } else if (phase_ == 2) {
                io.AddMouseButtonEvent(0, true);
            } else if (phase_ < 3 + kMoves) {
                const float t = (float)(phase_ - 2) / (float)kMoves;
                io.AddMousePosEvent(ax_ + (0.02f + 0.96f * t) * aw_, ay_ + (0.02f + 0.23f * t) * ah_);
            } else if (phase_ == 3 + kMoves) {
                io.AddMouseButtonEvent(0, false);
            } else {
                io.AddKeyEvent(ImGuiMod_Shift, false);
                next("select in " + s.a + (s.b.empty() ? "" : " " + s.b));
                return;
            }
            ++phase_;
            return;
        }
        fail("unknown step '" + s.verb + "'");
    }

private:
    struct Step {
        std::string verb, a, b;
    };

    std::string where() const {
        char buf[48];
        std::snprintf(buf, sizeof buf, "(%.0f, %.0f)", x_, y_);
        return buf;
    }
    void next(const std::string& line) {
        log_.push_back(line);
        ++at_;
        phase_ = 0;
        frames_ = 0;
    }
    void fail(const std::string& why) {
        error_ = why;
        log_.push_back("stopped: " + why);
    }

    std::vector<Step> steps_;
    size_t at_ = 0;
    int phase_ = 0, frames_ = 0;
    float x_ = 0, y_ = 0, ax_ = 0, ay_ = 0, aw_ = 0, ah_ = 0;
    std::string error_;
    std::vector<std::string> log_;
};

}  // namespace caliper_live
