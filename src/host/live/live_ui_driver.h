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
//   pick <view> <fx> <fy> a click at the fraction (fx, fy) of the view's data area (a scene's pick)
//   hover <view> <fx> <fy> the pointer moves to that place, nothing pressed, and rests 10 frames
//   rect <view> <fx0> <fy0> <fx1> <fy1>
//                         a shift+alt drag between the two places: a rectangle selection
//   where <view>          logs the view's data area (x y w h in points), the framebuffer's
//                         pixels per point and the view's own geometry text (a scene's camera,
//                         and its scale bar's band as "scale_bar x0 y0 x1 y1" in points)
//   leave                 the pointer leaves the window (no hover is left behind)
//   wait <frames>         nothing for that many frames
//   state <run state>     wait until the run state is <run state> (paused, running, ...)
//   bar [<text>]          wait until the run state the bar wrote in the last frame starts
//                         with <text> (any text without one), and log it: bar shows '<text>'
// Three steps act on the spec page instead, through the hooks the app gives
// (set_spec_hooks()), and wait up to kLongPatience frames, because they wait
// for an agent in another process:
//   panel <id>            wait until the open spec's scene panel <id> draws its live scene
//   pick_panel <id> <fx> <fy>
//                         a pick at the fraction (fx, fy) of the scene panel's content
//                         (SpecPage::pick_scene), once the panel draws its live scene
//   held [<kind>]         wait until a call of events.wait waits on the event ring (a long
//                         poll), with a kind one whose filter takes events of that kind
//   drag_panel <id> <fx0> <fy0> <fx1> <fy1> <frames> [left|right|middle] [stale]
//                         once the panel draws its scene, a drag with that button (left without
//                         one) across its content from (fx0, fy0) to (fx1, fy1), fractions of it,
//                         one move each frame for <frames> frames; each frame logs the time since
//                         the frame before and what the panel drew: "drag <k> ms <time> <the probe
//                         of scene_probe>". With `stale`, as a window that got no cursor moves
//                         before the press: the pointer was last seen at (fx1, fy1), and the press
//                         at (fx0, fy0) comes through the platform's callback (platform_button)
// The app feeds the driver on the spec page only while the step is one of
// these, `wait` or `leave`.
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
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

namespace caliper_live {

class UiDriver {
public:
    static constexpr int kPatience = 1200;   // frames a step waits for its target (20 s at 60 Hz)
    static constexpr int kLongPatience = 10 * kPatience;   // a spec page step waits for an agent

    // What the steps on the spec page ask the app (src/main.cpp).
    struct SpecHooks {
        // Whether the open spec's scene panel `id` draws its live scene.
        std::function<bool(const std::string& id)> scene_ready;
        // Asks the scene panel `id` for a pick at (u, v) at its next draw;
        // false when it cannot take one.
        std::function<bool(const std::string& id, float u, float v)> scene_pick;
        // Whether a call of events.wait waits on the event ring; with a kind
        // ("pick"), one whose filter takes events of that kind ("" for any).
        std::function<bool(const std::string& kind)> long_poll_held;
        // Where the scene panel `id` was drawn last (x, y, width, height in
        // points) and what it drew in words (SpecPage::scene_probe); false
        // before its first draw.
        std::function<bool(const std::string& id, float rect[4], std::string* probe)> scene_probe;
        // A press or a release as the platform delivers it, through the app's GLFW callbacks,
        // with (x, y) as the pointer's place: the `stale` drags' press.
        std::function<void(int button, bool down, float x, float y)> platform_button;
    };
    void set_spec_hooks(SpecHooks hooks) { spec_ = std::move(hooks); }
    // Whether the current step may run on the spec page.
    bool spec_step() const {
        if (done()) return false;
        const std::string& v = steps_[at_].verb;
        return v == "panel" || v == "pick_panel" || v == "held" || v == "drag_panel" || v == "wait" || v == "leave";
    }

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
        if (s.verb == "panel" || s.verb == "pick_panel") {
            const bool ready = spec_.scene_ready && spec_.scene_ready(s.a);
            if (ready && s.verb == "panel") {
                next("panel " + s.a + " draws its scene");
                return;
            }
            float u = 0.5f, v = 0.5f;
            if (ready && std::sscanf(s.b.c_str(), "%f %f", &u, &v) != 2) {
                fail("pick_panel needs <id> <fx> <fy>");
                return;
            }
            if (ready && spec_.scene_pick && spec_.scene_pick(s.a, u, v)) next("pick_panel " + s.a + " " + s.b);
            else if (frames_ > kLongPatience) fail("the scene panel '" + s.a + "' never drew its scene");
            return;
        }
        if (s.verb == "drag_panel") {
            drag_panel(io, s);
            return;
        }
        if (s.verb == "held") {
            if (spec_.long_poll_held && spec_.long_poll_held(s.a))
                next("events.wait is held" + (s.a.empty() ? std::string() : " for " + s.a));
            else if (frames_ > kLongPatience) fail("no call of events.wait was ever held");
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
        if (s.verb == "where") {
            const std::string view = s.a + (s.b.empty() ? "" : " " + s.b);
            const caliper_live_views::View* v = core ? caliper_live_views::find(core, view.c_str()) : nullptr;
            if (!v || !v->geo_valid || v->updates == 0) {
                if (frames_ > kPatience) fail("view '" + view + "' was never drawn");
                return;
            }
            char buf[160];
            std::snprintf(buf, sizeof buf, "where %s area %.2f %.2f %.2f %.2f scale %.3f", view.c_str(), v->geo_area[0],
                          v->geo_area[1], v->geo_area[2], v->geo_area[3], io.DisplayFramebufferScale.x);
            next(std::string(buf) + (v->geo_text.empty() ? "" : " " + v->geo_text));
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
        if (s.verb == "pick" || s.verb == "hover" || s.verb == "rect") {
            // The view's name is one word here; the fractions follow it.
            if (phase_ == 0) {
                const caliper_live_views::View* v = core ? caliper_live_views::find(core, s.a.c_str()) : nullptr;
                if (!v || !v->geo_valid || v->geo_area[2] < 8.0f || v->geo_area[3] < 8.0f) {
                    if (frames_ > kPatience) fail("view '" + s.a + "' was never drawn");
                    return;
                }
                std::stringstream fr(s.b);
                float f[4] = {0.5f, 0.5f, 0.5f, 0.5f};
                for (float& x : f) fr >> x;
                ax_ = v->geo_area[0] + f[0] * v->geo_area[2];
                ay_ = v->geo_area[1] + f[1] * v->geo_area[3];
                aw_ = v->geo_area[0] + f[2] * v->geo_area[2];   // rect: the far corner
                ah_ = v->geo_area[1] + f[3] * v->geo_area[3];
                x_ = ax_;
                y_ = ay_;
                io.AddMousePosEvent(x_, y_);
                phase_ = 1;
                return;
            }
            if (s.verb == "hover") {
                if (++phase_ > 10) next("hover in " + s.a + " at " + where());
                return;
            }
            if (s.verb == "pick") {
                switch (phase_++) {
                    case 1: io.AddMouseButtonEvent(0, true); break;
                    case 2: io.AddMouseButtonEvent(0, false); break;
                    default:
                        if (phase_ > 8) next("pick in " + s.a + " at " + where());   // its readback lands
                        break;
                }
                return;
            }
            constexpr int kMoves = 8;   // rect
            if (phase_ == 1) {
                io.AddKeyEvent(ImGuiMod_Shift, true);
                io.AddKeyEvent(ImGuiMod_Alt, true);
            } else if (phase_ == 2) {
                io.AddMouseButtonEvent(0, true);
            } else if (phase_ < 3 + kMoves) {
                const float t = (float)(phase_ - 2) / (float)kMoves;
                io.AddMousePosEvent(ax_ + (aw_ - ax_) * t, ay_ + (ah_ - ay_) * t);
            } else if (phase_ == 3 + kMoves) {
                io.AddMouseButtonEvent(0, false);
            } else {
                io.AddKeyEvent(ImGuiMod_Alt, false);
                io.AddKeyEvent(ImGuiMod_Shift, false);
                next("rect in " + s.a);
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

    // A drag across a scene panel of the spec page: phase 0 places the pointer, 1 presses,
    // 2 .. moves + 1 move it one step each frame, then the release and one frame after it. Each
    // frame from the press on logs the probe, which tells what the frame before drew.
    void drag_panel(ImGuiIO& io, const Step& s) {
        float rect[4];
        std::string probe;
        const bool ready = spec_.scene_probe && spec_.scene_probe(s.a, rect, &probe);
        if (phase_ == 0) {
            float f[5] = {0, 0, 0, 0, 0};
            char which[16] = "left", how[16] = "";
            const int n =
                std::sscanf(s.b.c_str(), "%f %f %f %f %f %15s %15s", &f[0], &f[1], &f[2], &f[3], &f[4], which, how);
            if (std::strcmp(which, "stale") == 0) {   // the button left out
                std::strcpy(how, "stale");
                std::strcpy(which, "left");
            }
            button_ = std::strcmp(which, "right") == 0 ? 1 : std::strcmp(which, "middle") == 0 ? 2 : 0;
            stale_ = std::strcmp(how, "stale") == 0;
            if (n < 5 || f[4] < 1 || (button_ == 0 && std::strcmp(which, "left") != 0) ||
                (*how && !stale_) || (stale_ && !spec_.platform_button)) {
                fail("drag_panel needs <id> <fx0> <fy0> <fx1> <fy1> <frames> [left|right|middle] [stale]");
                return;
            }
            if (!ready || rect[2] < 8.0f || rect[3] < 8.0f) {
                if (frames_ > kLongPatience) fail("the scene panel '" + s.a + "' never drew its scene");
                return;
            }
            ax_ = rect[0] + f[0] * rect[2];
            ay_ = rect[1] + f[1] * rect[3];
            aw_ = rect[0] + f[2] * rect[2];   // the end of the drag
            ah_ = rect[1] + f[3] * rect[3];
            moves_ = int(f[4]);
            x_ = ax_;
            y_ = ay_;
            // A stale drag: the pointer was last seen at the drag's end, far from its press.
            if (stale_) io.AddMousePosEvent(aw_, ah_);
            else io.AddMousePosEvent(x_, y_);
            phase_ = 1;
            return;
        }
        const auto now = std::chrono::steady_clock::now();
        if (phase_ > 1) {
            char head[64];
            std::snprintf(head, sizeof head, "drag %d ms %.3f ", phase_ - 1,
                          std::chrono::duration<double, std::milli>(now - last_).count());
            log_.push_back(head + (ready ? probe : std::string("no probe")));
        }
        last_ = now;
        if (phase_ == 1) {
            if (stale_) spec_.platform_button(button_, true, x_, y_);
            else io.AddMouseButtonEvent(button_, true);
        } else if (phase_ < 2 + moves_) {
            const float t = float(phase_ - 1) / float(moves_);
            x_ = ax_ + (aw_ - ax_) * t;
            y_ = ay_ + (ah_ - ay_) * t;
            io.AddMousePosEvent(x_, y_);
        } else if (phase_ == 2 + moves_) {
            io.AddMouseButtonEvent(button_, false);
        } else {
            next("drag_panel " + s.a + " " + s.b);
            return;
        }
        ++phase_;
    }

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
    int phase_ = 0, frames_ = 0, moves_ = 0, button_ = 0;
    bool stale_ = false;
    std::chrono::steady_clock::time_point last_{};
    float x_ = 0, y_ = 0, ax_ = 0, ay_ = 0, aw_ = 0, ah_ = 0;
    std::string error_;
    std::vector<std::string> log_;
    SpecHooks spec_;
};

}  // namespace caliper_live
