#pragma once
// ============================================================================
// spec_ui_driver.h: a testing aid that plays a short script of mouse and key
// input into the app's spec page, and reads back named frames as PNG files,
// as live/live_ui_driver.h does for the live page. The input goes through
// ImGui's own event queue, exactly as the platform backend would feed a
// person's clicks and keys, so the page and the widget agent's inline surface
// run their real paths. The app plays a script given in CALIPER_TEST_SPEC_UI
// only when its window is hidden (--hidden), and writes the frames into the
// folder CALIPER_TEST_SPEC_UI_SHOTS; a normal run never enables it.
//
// The script is a list of steps separated by ';':
//   wait <frames>        nothing for that many frames
//   focus <panel>        a click in the middle of a shown panel, which focuses it
//   leave                the pointer leaves the window (no hover is left behind)
//   key <chord>          a press and release of "cmd+k" (Ctrl+K off macOS),
//                        "escape" or "enter"
//   type <text>          the text as the keyboard's characters
//   until <state>        wait until the inline surface's state is <state>
//                        (prompt, running, preview, applied, ...)
//   shot <name>          read the next frame back into <name>.png
//   chat open|close      the side chat's toggle (chat_pane.h)
//   chat focus           puts the keyboard into the chat's input
//   chat until <state>   wait until the chat's state is <state> (ready,
//                        running, exited, ...)
//   chat item <kind> [<command> [<status>]]
//                        wait until the conversation has such an item: a
//                        tool call of a command in a status ("tool
//                        catalog.describe in_progress"), an approval row
//                        ("approval panel.create waiting"), a stop word
//                        ("stop - DONE")
//   touch <path>         creates an empty file (a stub agent's gate)
//   append <path> <line> appends a line to a file (a test's approval decision)
//   quit                 close the app
// A step whose target is not there yet waits for it, up to kPatience frames,
// then the script stops with an error.
//
// Header-only; frame thread. Call feed() once per frame BEFORE ImGui::NewFrame.
// ============================================================================
#include "builder/app_session.h"
#include "builder/chat_pane.h"
#include "builder/spec_page.h"
#include "export_service.h"

#include <imgui.h>

#if CALIPER_WITH_BUILDER
#include <caliper_builder/session/session.h>
#endif

#include <cfloat>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

namespace caliper_host {

class SpecUiDriver {
public:
    static constexpr int kPatience = 1800;   // frames a step waits (30 s at 60 Hz)
    static constexpr uint64_t kShotTicket = ~uint64_t(0) - 1;   // beside AppSession::kCaptureTicket

    SpecUiDriver(const std::string& script, std::string shots) : shots_(std::move(shots)) {
        std::stringstream all(script);
        std::string part;
        while (std::getline(all, part, ';')) {
            std::stringstream words(part);
            Step s;
            words >> s.verb;
            std::getline(words, s.arg);
            while (!s.arg.empty() && s.arg.front() == ' ') s.arg.erase(s.arg.begin());
            while (!s.arg.empty() && s.arg.back() == ' ') s.arg.pop_back();
            if (!s.verb.empty()) steps_.push_back(s);
        }
    }

    bool done() const { return at_ >= steps_.size() || !error_.empty(); }
    bool quit() const { return quit_; }
    const std::string& error() const { return error_; }
    std::vector<std::string> take_log() {
        std::vector<std::string> out;
        out.swap(log_);
        return out;
    }

    // A frame to read back, once: the caller arms a capture under
    // kShotTicket for the frame it draws next.
    bool take_shot_request() {
        const bool r = shot_requested_;
        shot_requested_ = false;
        return r;
    }

    // The read-back of the frame armed for the current `shot` step.
    void write_shot(uint32_t w, uint32_t h, const std::vector<uint8_t>& rgba) {
        if (at_ >= steps_.size() || steps_[at_].verb != "shot") return;
        std::string why;
        const std::string file = (std::filesystem::path(shots_) / (steps_[at_].arg + ".png")).string();
        if (!export_write_png_atomic(file, rgba.data(), w, h, &why)) error_ = "cannot write " + file + ": " + why;
        else next("shot " + steps_[at_].arg + " (" + std::to_string(w) + "x" + std::to_string(h) + ", " + facts_ + ")");
    }
    void shot_failed(const std::string& why) { error_ = "shot failed: " + why; }

    void feed(const SpecPage& page, AppSession& app, ChatPane* chat = nullptr) {
        if (done()) return;
        ImGuiIO& io = ImGui::GetIO();
        // Releases of the last frame's presses.
        for (ImGuiKey k : held_keys_) io.AddKeyEvent(k, false);
        held_keys_.clear();
        if (mouse_down_) {
            io.AddMouseButtonEvent(0, false);
            mouse_down_ = false;
        }
        const Step& s = steps_[at_];
        ++frames_;
        if (s.verb == "wait") {
            if (frames_ >= std::atoi(s.arg.c_str())) next("waited " + s.arg + " frames");
        } else if (s.verb == "leave") {
            io.AddMousePosEvent(-FLT_MAX, -FLT_MAX);
            next("left the window");
        } else if (s.verb == "focus") {
            float r[4];
            if (!page.panel_rect(s.arg, r)) return patience("panel " + s.arg + " is not shown");
            io.AddMousePosEvent(r[0] + r[2] * 0.5f, r[1] + r[3] * 0.5f);
            io.AddMouseButtonEvent(0, true);
            mouse_down_ = true;
            next("clicked panel " + s.arg);
        } else if (s.verb == "key") {
            press(s.arg);
            if (error_.empty()) next("pressed " + s.arg);
        } else if (s.verb == "type") {
            io.AddInputCharactersUTF8(s.arg.c_str());
            next("typed " + s.arg);
        } else if (s.verb == "until") {
            if (page.widget_state() != s.arg) return patience("the surface is " + page.widget_state() + ", not " + s.arg);
            next("the surface is " + s.arg);
        } else if (s.verb == "shot") {
            if (!armed_) {
                facts_ = "surface " + page.widget_state() + ", focus " +
                         (page.focused_panel().empty() ? std::string("none") : page.focused_panel()) + ", undo " +
                         std::to_string(undo_steps(app)) + (chat ? ", chat " + chat->state() : std::string());
                shot_requested_ = true;
                armed_ = true;
            } else if (frames_ > kPatience) {
                error_ = "shot " + s.arg + ": no frame was read back";
            }
        } else if (s.verb == "chat") {
            if (!chat) return void(error_ = "chat: the app has no side chat");
            std::stringstream words(s.arg);
            std::string what, a, b, c;
            words >> what >> a >> b >> c;
            if (what == "open" || what == "close") {
                chat->set_open(what == "open");
                next("chat " + what);
            } else if (what == "focus") {
                if (chat->state() == "none") return patience("the chat is not made yet");
                chat->focus_input();
                next("chat focus");
            } else if (what == "until") {
                if (chat->state() != a) return patience("the chat is " + chat->state() + ", not " + a);
                next("the chat is " + a);
            } else if (what == "item") {
                const std::string cmd = b == "-" ? std::string() : b;
                if (!chat->has_item(a, cmd, c)) return patience("no chat item " + s.arg);
                next("chat " + s.arg);
            } else {
                error_ = "unknown chat step " + what;
            }
        } else if (s.verb == "touch" || s.verb == "append") {
            std::stringstream words(s.arg);
            std::string file, line;
            words >> file;
            std::getline(words, line);
            while (!line.empty() && line.front() == ' ') line.erase(line.begin());
            std::ofstream out(file, std::ios::app);
            if (s.verb == "append") out << line << "\n";
            if (!out) return void(error_ = s.verb + ": cannot write " + file);
            next(s.verb + " " + std::filesystem::path(file).filename().string());
        } else if (s.verb == "quit") {
            quit_ = true;
            next("quit");
        } else {
            error_ = "unknown step " + s.verb;
        }
    }

private:
    struct Step {
        std::string verb, arg;
    };

    void next(const std::string& what) {
        log_.push_back(what);
        ++at_;
        frames_ = 0;
        armed_ = false;
    }

    void patience(const std::string& why) {
        if (frames_ > kPatience) error_ = steps_[at_].verb + " " + steps_[at_].arg + ": " + why;
    }

    void press(const std::string& chord) {
        ImGuiIO& io = ImGui::GetIO();
        std::vector<ImGuiKey> keys;
        if (chord == "cmd+k") {
            // ImGui swaps Cmd and Ctrl at the event level on macOS
            // (ConfigMacOSXBehaviors): Super there becomes its Ctrl, the
            // chord the page reads as ⌘K.
            keys = {io.ConfigMacOSXBehaviors ? ImGuiMod_Super : ImGuiMod_Ctrl, ImGuiKey_K};
        } else if (chord == "escape") {
            keys = {ImGuiKey_Escape};
        } else if (chord == "enter") {
            keys = {ImGuiKey_Enter};
        } else {
            error_ = "unknown key " + chord;
            return;
        }
        for (ImGuiKey k : keys) io.AddKeyEvent(k, true);
        held_keys_ = keys;
    }

    static size_t undo_steps(AppSession& app) {
#if CALIPER_WITH_BUILDER
        if (caliper_builder::Session* s = app.session()) return s->registry().history_state().undo.size();
#else
        (void)app;
#endif
        return 0;
    }

    std::vector<Step> steps_;
    std::string shots_;
    size_t at_ = 0;
    int frames_ = 0;
    bool armed_ = false, shot_requested_ = false, quit_ = false, mouse_down_ = false;
    std::vector<ImGuiKey> held_keys_;
    std::vector<std::string> log_;
    std::string error_, facts_;
};

}  // namespace caliper_host
