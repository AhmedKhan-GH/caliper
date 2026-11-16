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
//   key <chord>          a press and release of a chord: cmd, shift and one
//                        key (a letter, a digit, enter, escape, backspace,
//                        f3, f4, f5, up, down, comma), such as "cmd+k"
//                        (Ctrl+K off macOS) or "cmd+shift+z"
//   type <text>          the text as the keyboard's characters; "\n" is a
//                        line break
//   until <state> [<seconds>]
//                        wait until the inline surface's state is <state>
//                        (prompt, running, preview, applied, ...), for at
//                        most that many seconds of wall time when given (a
//                        real model's turn), else kPatience frames; a state
//                        that does not change by itself (failed, declined,
//                        handoff, conflict, unavailable) stops the script at
//                        once when it is not the one waited for. The log line
//                        adds what the surface shows (SpecPage::widget_facts())
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
//   chat draft           wait until the chat's input holds a handoff's draft
//   chat note <text>|none
//                        wait until the chat's header shows that GPU note
//   chat local on|off    marks the chosen agent as running on a local model,
//                        or clears it, as the picker's Local model row does
//   chat picker          opens the agent picker
//   cards <n>            wait until the approval cards' strip shows at least n
//                        waiting cards at its full height (approval_cards.h)
//   click <item> [fx fy] a click on a named rectangle of the spec page
//                        (SpecPage::item_rect: "page:browser") or of the
//                        builder pane (BuilderPane::item_rect:
//                        "source:table:metrics", "primitive:series",
//                        "prop:title", "bar:save", ...),
//                        at the fractions fx, fy of it (0.5 0.5 by default); a
//                        negative fraction counts pixels from its right or
//                        bottom edge
//   drag <item> [fx fy] to <item> [fx fy] [hold]
//                        a press on the first, a move in steps to the second
//                        and a release there, or none with `hold`
//   release              lets go of a held drag
//   selection <path>     writes the page's selection as JSON to <path>
//                        (SpecPage::selection_json()), once there is one
//   wait-file <path>     wait until <path> exists (a file a test writes)
//   select <panel> <fx0> <fx1> [<fy0> <fy1>]
//                        a shift-drag across a shown panel from the fractions
//                        (fx0, fy0) to (fx1, fy1) of its rectangle (fy 0.5
//                        when not given): a range on a series panel, a
//                        rectangle of points on a points panel
//   drop <path> ...      files dropped on the window, through the app's drop
//                        callback's own function (a hidden window receives no
//                        real drop)
//   pane source <key>    wait until the builder lists that source and no
//                        import runs
//   pane focus <panel>   wait until the builder's focused panel is <panel>
//   page <name>          wait until the app shows that page (landing, spec, live)
//   command-line <text>|none
//                        wait until the landing's command line holds that
//                        text, or nothing
//   landing selected <applet id>
//                        wait until the landing's selected row is the applet
//                        with that manifest id
//   landing projects <applet id>,...|none
//                        wait until Projects lists exactly those dashboards
//                        (manifest ids, in their order), or none
//   widget <label>       presses the inline surface's action of that label
//                        ("Setup", "Ask in chat", ...) once the surface offers it,
//                        through the surface's own act(), as its button does
//   settings open|close  the widget agent's settings (agent_settings.h)
//   settings preset <id> | model <name>
//                        chooses a preset, or sets the model, as a click on its
//                        button or Enter in its field does
//   settings list | probe
//                        lists the server's models, or probes the model, as
//                        List and Probe do
//   settings until open | note <text>|none | state <text>|none | done [<seconds>]
//                        wait until the settings are open, show that GPU note,
//                        or that plain state, or until no list or probe runs
//                        (for at most that many seconds of wall time when
//                        given); `done` logs the capabilities the settings show
//   files <folder> <prefix> <count>
//                        wait until the folder holds that many files whose
//                        names begin with <prefix> (the API's agent files)
//   touch <path>         creates an empty file (a stub agent's gate)
//   append <path> <line> appends a line to a file (a test's approval decision)
//   patience <seconds>   the steps after it wait for their target up to that
//                        many seconds instead of kPatience frames, such as a
//                        real agent's turn on a local model; 0 goes back to
//                        kPatience
//   save until           wait until "Save changes to <name>?" is open
//                        (save_prompt.h)
//   save save|dont-save|cancel
//                        wait until it is open, then answer it as a click on
//                        that button does
//   close                a request to close the window, as its close button
//                        and ⌘Q make, which asks first when the open spec
//                        has unsaved changes
//   quit                 close the app
// A step whose target is not there yet waits for it, up to kPatience frames
// (or the seconds of the last `patience`), then the script stops with an
// error.
//
// Header-only; frame thread. Call feed() once per frame BEFORE ImGui::NewFrame.
// ============================================================================
#include "builder/app_session.h"
#include "builder/chat_pane.h"
#include "builder/save_prompt.h"
#include "builder/spec_page.h"
#include "export_service.h"

#include <imgui.h>

#if CALIPER_WITH_BUILDER
#include "builder/agent_settings.h"
#include "builder/builder_pane.h"

#include <caliper_builder/session/session.h>
#endif

#include <algorithm>
#include <chrono>
#include <cmath>
#include <functional>
#include <optional>

#include <cfloat>
#include <cstdio>
#include <chrono>
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

    // The page the app shows ("landing", "spec", "live"), and the function
    // a dropped file reaches; the app sets both before each feed().
    void set_page(std::string page) { page_ = std::move(page); }
    // The landing's command line, for the `command-line` step; the app sets it.
    void set_command_line(std::string text) { command_line_ = std::move(text); }
    // The landing's selected row (its manifest id, "" for none) and the
    // dashboards Projects lists, for the `landing` steps; the app sets them.
    void set_landing(std::string selected, std::vector<std::string> projects) {
        landing_selected_ = std::move(selected);
        landing_projects_ = std::move(projects);
    }
    void set_drop(std::function<void(std::vector<std::string>)> drop) { drop_ = std::move(drop); }
    // The waiting cards the approval strip shows (ApprovalTray::shown_waiting()), for `cards`.
    void set_cards(std::function<int()> cards) { cards_ = std::move(cards); }
    // "Save changes to <name>?" for the `save` steps, and the window's close
    // request for `close`; the app sets both.
    void set_save_prompt(SavePrompt* prompt) { save_prompt_ = prompt; }
    void set_close(std::function<void()> close) { close_ = std::move(close); }
#if CALIPER_WITH_BUILDER
    // The widget agent's settings, for the `settings` steps; the app sets them.
    void set_agent_settings(AgentSettings* settings) { settings_ = settings; }
#endif

    void feed(SpecPage& page, AppSession& app, ChatPane* chat = nullptr) {
        if (done()) return;
        ImGuiIO& io = ImGui::GetIO();
        // Releases of the last frame's presses.
        for (ImGuiKey k : held_keys_) io.AddKeyEvent(k, false);
        held_keys_.clear();
        if (mouse_down_) {
            io.AddMouseButtonEvent(0, false);
            mouse_down_ = false;
        }
        // After a step of input, frames for the page to act on it and draw
        // its new rectangles, so the next step aims at where things are now.
        if (settle_ > 0) {
            --settle_;
            return;
        }
        const Step& s = steps_[at_];
        ++frames_;
        if (s.verb == "widget") {
            if (!page.widget_action(s.arg)) return patience("the surface offers no " + s.arg);
            next("pressed " + s.arg);
            return;
        }
        if (s.verb == "settings") {
#if CALIPER_WITH_BUILDER
            if (!settings_) return void(error_ = "settings: the app has no widget agent settings");
            std::stringstream words(s.arg);
            std::string what, which, rest;
            words >> what >> which;
            std::getline(words, rest);
            while (!rest.empty() && rest.front() == ' ') rest.erase(rest.begin());
            const std::string want = rest == "none" ? std::string() : rest;
            auto shown = [](const std::string& t) { return t.empty() ? std::string("none") : t; };
            if (what == "open" || what == "close") {
                settings_->set_open(what == "open");
                next("settings " + what);
            } else if (what == "preset" && !which.empty()) {
                settings_->choose_preset(which);
                next("settings preset " + which);
            } else if (what == "model" && !which.empty()) {
                settings_->set_model(which);
                next("settings model " + which);
            } else if (what == "list" || what == "probe") {
                if (settings_->busy()) return patience("a list or a probe runs");
                if (what == "list") settings_->list_models();
                else settings_->probe(true);
                next("settings " + what);
            } else if (what == "until" && which == "done") {
                if (!started_) started_ = std::chrono::steady_clock::now();
                // A few frames without a list or a probe running, so the
                // settings have taken the last answer (AgentSettings::frame()).
                idle_frames_ = settings_->busy() ? 0 : idle_frames_ + 1;
                if (idle_frames_ < 3) return patience_for(rest, "a list or a probe runs");
                next("the settings are done: " + settings_facts());
            } else if (what == "until" && which == "open") {
                if (!settings_->is_open()) return patience("the settings are closed");
                next("the settings are open");
            } else if (what == "until" && which == "note") {
                if (settings_->gpu_note() != want) return patience("the settings' note is " + shown(settings_->gpu_note()));
                next("the settings' note is " + shown(want));
            } else if (what == "until" && which == "state") {
                if (settings_->state() != want) return patience("the settings' state is " + shown(settings_->state()));
                next("the settings' state is " + shown(want));
            } else {
                error_ = "settings takes open, close, preset, model, list, probe or until open|note|state|done";
            }
#else
            error_ = "settings: the builder is not in this build";
#endif
            return;
        }
        if (s.verb == "click") {
            std::stringstream words(s.arg);
            std::string name;
            float fx = 0.5f, fy = 0.5f;
            words >> name >> fx >> fy;
            float x = 0, y = 0;
            if (!item_point(page, name, fx, fy, &x, &y)) return patience("no builder item " + name);
            io.AddMousePosEvent(x, y);
            io.AddMouseButtonEvent(0, true);
            mouse_down_ = true;
            next("clicked " + name);
            return;
        }
        if (s.verb == "drag") {
            // drag <a> [fx fy] to <b> [fx fy] [hold]
            std::stringstream words(s.arg);
            std::vector<std::string> w;
            for (std::string t; words >> t;) w.push_back(t);
            size_t to = 0;
            while (to < w.size() && w[to] != "to") ++to;
            if (to == 0 || to + 1 >= w.size()) return void(error_ = "drag needs <item> to <item>");
            const float afx = to >= 3 ? (float)std::atof(w[1].c_str()) : 0.5f, afy = to >= 3 ? (float)std::atof(w[2].c_str()) : 0.5f;
            const size_t b = to + 1;
            const bool b_frac = b + 2 < w.size() || (b + 2 == w.size() && w.back() != "hold");
            const float bfx = b_frac ? (float)std::atof(w[b + 1].c_str()) : 0.5f, bfy = b_frac ? (float)std::atof(w[b + 2].c_str()) : 0.5f;
            const bool hold = w.back() == "hold";
            constexpr int kSteps = 14;
            if (drag_frame_ == 0) {
                if (!item_point(page, w[0], afx, afy, &drag_a_[0], &drag_a_[1]) ||
                    !item_point(page, w[b], bfx, bfy, &drag_b_[0], &drag_b_[1]))
                    return patience("no builder item " + w[0] + " or " + w[b]);
                io.AddMousePosEvent(drag_a_[0], drag_a_[1]);
                io.AddMouseButtonEvent(0, true);
                drag_frame_ = 1;
            } else if (drag_frame_ <= kSteps) {
                const float t = (float)drag_frame_ / (float)kSteps;
                io.AddMousePosEvent(std::round(drag_a_[0] + (drag_b_[0] - drag_a_[0]) * t),
                                    std::round(drag_a_[1] + (drag_b_[1] - drag_a_[1]) * t));
                ++drag_frame_;
            } else if (drag_frame_ == kSteps + 1) {
                ++drag_frame_;   // one more frame at the target, so the drop's zone is read there
            } else {
                if (!hold) io.AddMouseButtonEvent(0, false);
                next(std::string(hold ? "dragged and held " : "dragged ") + w[0] + " to " + w[b]);
            }
            return;
        }
        if (s.verb == "select") {
            std::stringstream words(s.arg);
            std::string panel;
            float fx0 = 0.3f, fx1 = 0.6f, fy0 = 0.5f, fy1 = 0.5f;
            words >> panel >> fx0 >> fx1;
            if (!(words >> fy0 >> fy1)) fy0 = fy1 = 0.5f;
            constexpr int kSteps = 12;
            if (drag_frame_ == 0) {
                float r[4];
                if (!page.panel_rect(panel, r)) return patience("panel " + panel + " is not shown");
                drag_a_[0] = std::round(r[0] + r[2] * fx0);
                drag_a_[1] = std::round(r[1] + r[3] * fy0);
                drag_b_[0] = std::round(r[0] + r[2] * fx1);
                drag_b_[1] = std::round(r[1] + r[3] * fy1);
                io.AddKeyEvent(ImGuiMod_Shift, true);
                io.AddMousePosEvent(drag_a_[0], drag_a_[1]);
                drag_frame_ = 1;
            } else if (drag_frame_ == 1) {
                io.AddMouseButtonEvent(0, true);
                drag_frame_ = 2;
            } else if (drag_frame_ < kSteps + 2) {
                const float t = (float)(drag_frame_ - 1) / (float)kSteps;
                io.AddMousePosEvent(std::round(drag_a_[0] + (drag_b_[0] - drag_a_[0]) * t),
                                    std::round(drag_a_[1] + (drag_b_[1] - drag_a_[1]) * t));
                ++drag_frame_;
            } else if (drag_frame_ == kSteps + 2) {
                io.AddMouseButtonEvent(0, false);
                ++drag_frame_;
            } else {
                io.AddKeyEvent(ImGuiMod_Shift, false);
                next("selected on " + panel);
            }
            return;
        }
        if (s.verb == "selection") {
            const std::string sel = page.selection_json();
            if (sel.empty()) return patience("no selection");
            std::ofstream(s.arg) << sel << "\n";
            next("wrote the selection to " + std::filesystem::path(s.arg).filename().string());
            return;
        }
        if (s.verb == "files") {
            std::stringstream words(s.arg);
            std::string folder, prefix;
            int want = -1;
            words >> folder >> prefix >> want;
            if (folder.empty() || prefix.empty() || want < 0) return void(error_ = "files takes <folder> <prefix> <count>");
            int n = 0;
            std::error_code ec;
            for (std::filesystem::directory_iterator it(folder, ec), end; !ec && it != end; it.increment(ec))
                if (it->path().filename().string().rfind(prefix, 0) == 0) ++n;
            const std::string what = std::to_string(n) + (n == 1 ? " file " : " files ") + prefix + "*";
            if (n != want) return patience(what);
            next(what);
            return;
        }
        if (s.verb == "wait-file") {
            std::error_code ec;
            if (!std::filesystem::exists(s.arg, ec)) return patience("no file " + s.arg);
            next("found " + std::filesystem::path(s.arg).filename().string());
            return;
        }
        if (s.verb == "release") {
            io.AddMouseButtonEvent(0, false);
            next("released");
            return;
        }
        if (s.verb == "drop") {
            if (!drop_) return void(error_ = "drop: the app gave no drop function");
            std::stringstream words(s.arg);
            std::vector<std::string> paths;
            for (std::string p; words >> p;) paths.push_back(p);
            drop_(paths);
            next("dropped " + std::to_string(paths.size()) + " files");
            return;
        }
        if (s.verb == "page") {
            if (page_ != s.arg) return patience("the app shows " + page_ + ", not " + s.arg);
            next("page " + s.arg);
            return;
        }
        if (s.verb == "command-line") {
            const std::string want = s.arg == "none" ? std::string() : s.arg;
            if (command_line_ != want)
                return patience("the command line holds " + (command_line_.empty() ? std::string("nothing") : "\"" + command_line_ + "\""));
            next("the command line holds " + (want.empty() ? std::string("nothing") : "\"" + want + "\""));
            return;
        }
        if (s.verb == "landing") {
            std::stringstream words(s.arg);
            std::string what, want;
            words >> what >> want;
            auto shown = [](const std::string& t) { return t.empty() ? std::string("none") : t; };
            if (what == "selected") {
                if (landing_selected_ != want) return patience("the landing's selection is " + shown(landing_selected_));
                next("the landing's selection is " + want);
            } else if (what == "projects") {
                std::string listed;
                for (const auto& id : landing_projects_) listed += (listed.empty() ? "" : ",") + id;
                if (shown(listed) != want) return patience("Projects lists " + shown(listed));
                next("Projects lists " + want);
            } else {
                error_ = "landing takes selected <applet id> or projects <applet id>,...|none";
            }
            return;
        }
        if (s.verb == "pane") {
#if CALIPER_WITH_BUILDER
            std::stringstream words(s.arg);
            std::string what, arg;
            words >> what >> arg;
            BuilderPane* b = page.builder();
            if (!b) return patience("no builder pane");
            if (what == "source") {
                if (b->importing() || !b->find_source(arg)) return patience("the builder does not list " + arg);
                next("the builder lists " + arg);
            } else if (what == "focus") {
                if (b->focused_panel() != arg)
                    return patience("the builder's focus is " + b->focused_panel() + " (" + b->error().code + " " +
                                    b->error().message + ")");
                next("the builder's focus is " + arg);
            } else {
                error_ = "unknown pane step " + what;
            }
#else
            error_ = "pane: the builder is not in this build";
#endif
            return;
        }
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
            std::string text;
            for (size_t i = 0; i < s.arg.size(); ++i) {
                if (s.arg[i] == '\\' && i + 1 < s.arg.size() && s.arg[i + 1] == 'n') {
                    text += '\n';
                    ++i;
                } else {
                    text += s.arg[i];
                }
            }
            io.AddInputCharactersUTF8(text.c_str());
            next("typed " + s.arg);
        } else if (s.verb == "until") {
            std::stringstream words(s.arg);
            std::string want, seconds;
            words >> want >> seconds;
            if (!started_) started_ = std::chrono::steady_clock::now();
            const std::string now = page.widget_state();
            const std::string facts = page.widget_facts();
            if (now != want) {
                const std::string why = "the surface is " + now + ", not " + want + (facts.empty() ? "" : " (" + facts + ")");
                // A state that does not change by itself ends the wait.
                if (now == "failed" || now == "declined" || now == "handoff" || now == "conflict" ||
                    now == "unavailable")
                    return void(error_ = "until " + s.arg + ": " + why);
                return patience_for(seconds, why);
            }
            next("the surface is " + want + (facts.empty() ? "" : " (" + facts + ")"));
        } else if (s.verb == "shot") {
            if (!armed_) {
                facts_ = "surface " + page.widget_state() + ", focus " +
                         (page.focused_panel().empty() ? std::string("none") : page.focused_panel()) + ", undo " +
                         std::to_string(undo_steps(app)) + (chat ? ", chat " + chat->state() : std::string());
#if CALIPER_WITH_BUILDER
                if (settings_) facts_ += std::string(", settings ") + (settings_->is_open() ? "open" : "closed");
#endif
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
            } else if (what == "draft") {
                if (chat->draft_chips() == 0) return patience("the chat has no draft");
                next("chat draft of " + std::to_string(chat->draft_chips()) + " blocks");
            } else if (what == "local" && (a == "on" || a == "off")) {
                if (chat->state() == "none") return patience("the chat is not made yet");
                if (!chat->set_local_model(a == "on")) return void(error_ = "chat local: the chat has no agent");
                next("chat local " + a);
            } else if (what == "picker") {
                if (chat->state() == "none") return patience("the chat is not made yet");
                chat->open_picker();
                next("chat picker");
            } else if (what == "note") {
                std::string want = s.arg.substr(std::min(s.arg.size(), std::string("note ").size()));
                if (want == "none") want.clear();
                const std::string shown = chat->note().empty() ? std::string("none") : chat->note();
                if (chat->note() != want) return patience("the chat's note is " + shown);
                next("the chat's note is " + shown);
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
        } else if (s.verb == "cards") {
            if (!cards_) return void(error_ = "cards: the app gave no approval cards");
            const int want = std::atoi(s.arg.c_str());
            if (want < 1) return void(error_ = "cards takes a number of cards from 1");
            const int shown = cards_();
            if (shown < want) return patience("the approval strip shows " + std::to_string(shown) + " waiting cards");
            next("the approval strip shows " + std::to_string(shown) + (shown == 1 ? " waiting card" : " waiting cards"));
        } else if (s.verb == "patience") {
            char* end = nullptr;
            const double seconds = std::strtod(s.arg.c_str(), &end);
            if (s.arg.empty() || !end || *end != '\0' || !(seconds >= 0.0) || seconds > 86400.0)
                return void(error_ = "patience takes a number of seconds from 0 to 86400");
            patience_s_ = seconds;
            next(seconds > 0.0 ? "patience " + s.arg + " s" : "patience " + std::to_string(kPatience) + " frames");
        } else if (s.verb == "save") {
            if (!save_prompt_) return void(error_ = "save: the app gave no save prompt");
            if (!save_prompt_->is_open()) return patience("no Save changes question is open");
            if (s.arg == "until") {
                next("Save changes to " + save_prompt_->name() + "? is open");
            } else if (save_prompt_->answer(s.arg)) {
                next("answered " + s.arg);
            } else {
                error_ = "save takes until, save, dont-save or cancel";
            }
        } else if (s.verb == "close") {
            if (!close_) return void(error_ = "close: the app gave no close request");
            close_();
            next("asked to close the window");
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
        const std::string& verb = steps_[at_].verb;
        if (verb == "click" || verb == "key" || verb == "type" || verb == "drag" || verb == "release" ||
            verb == "drop" || verb == "select")
            settle_ = kSettle;
        log_.push_back(what);
        ++at_;
        frames_ = 0;
        idle_frames_ = 0;
        started_.reset();
        armed_ = false;
        drag_frame_ = 0;
        step_start_ = std::chrono::steady_clock::now();
    }

    // The step's error once it has waited its patience, with how long it
    // waited in frames and in seconds, so that a log says which limit stopped it.
    void patience(const std::string& why) {
        const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - step_start_).count();
        const bool over = patience_s_ > 0.0 ? seconds > patience_s_ : frames_ > kPatience;
        if (!over) return;
        char spent[64];
        std::snprintf(spent, sizeof spent, " (after %d frames, %.1f s)", frames_, seconds);
        error_ = steps_[at_].verb + " " + steps_[at_].arg + ": " + why + spent;
    }

    // As patience(), for at most `seconds` of wall time since the step began
    // when it is a number, else kPatience frames.
    void patience_for(const std::string& seconds, const std::string& why) {
        const double limit = seconds.empty() ? 0.0 : std::atof(seconds.c_str());
        if (limit <= 0.0) return patience(why);
        const double spent = std::chrono::duration<double>(std::chrono::steady_clock::now() - *started_).count();
        if (spent > limit) error_ = steps_[at_].verb + " " + steps_[at_].arg + ": " + why;
    }

#if CALIPER_WITH_BUILDER
    // The capabilities and the state the settings show, for the log.
    std::string settings_facts() const {
        const AgentCapabilities c = settings_->capabilities();
        auto yes_no = [](const std::optional<bool>& v) { return !v ? std::string("unknown") : *v ? "yes" : "no"; };
        const AgentProbeAnswer& a = settings_->probe_answer();
        std::string out = "tools " + yes_no(c.tools) + " (" + c.tools_source + "), vision " + yes_no(c.vision) + " (" +
                          c.vision_source + "), context " + (c.context ? std::to_string(*c.context) : "unknown") +
                          " (" + c.context_source + "), " + std::to_string(settings_->model_choices().size()) +
                          " models, state " + (settings_->state().empty() ? std::string("none") : settings_->state());
        if (!a.failure.empty()) out += ", failure " + a.failure;
        for (const auto& p : a.problems) out += ", " + p;
        return out;
    }
#endif

    // A chord of "cmd", "shift" and one key, joined by '+'.
    void press(const std::string& chord) {
        ImGuiIO& io = ImGui::GetIO();
        std::vector<ImGuiKey> keys;
        std::stringstream parts(chord);
        for (std::string p; std::getline(parts, p, '+');) {
            if (p == "cmd") {
                // ImGui swaps Cmd and Ctrl at the event level on macOS
                // (ConfigMacOSXBehaviors): Super there becomes its Ctrl, the
                // chord the page reads as ⌘.
                keys.push_back(io.ConfigMacOSXBehaviors ? ImGuiMod_Super : ImGuiMod_Ctrl);
            } else if (p == "shift") {
                keys.push_back(ImGuiMod_Shift);
            } else if (p == "escape") {
                keys.push_back(ImGuiKey_Escape);
            } else if (p == "enter") {
                keys.push_back(ImGuiKey_Enter);
            } else if (p == "backspace") {
                keys.push_back(ImGuiKey_Backspace);
            } else if (p == "f3") {
                keys.push_back(ImGuiKey_F3);
            } else if (p == "f4") {
                keys.push_back(ImGuiKey_F4);
            } else if (p == "f5") {
                keys.push_back(ImGuiKey_F5);
            } else if (p == "up") {
                keys.push_back(ImGuiKey_UpArrow);
            } else if (p == "down") {
                keys.push_back(ImGuiKey_DownArrow);
            } else if (p == "comma") {
                keys.push_back(ImGuiKey_Comma);
            } else if (p.size() == 1 && p[0] >= 'a' && p[0] <= 'z') {
                keys.push_back((ImGuiKey)(ImGuiKey_A + (p[0] - 'a')));
            } else if (p.size() == 1 && p[0] >= '0' && p[0] <= '9') {
                keys.push_back((ImGuiKey)(ImGuiKey_0 + (p[0] - '0')));
            } else {
                error_ = "unknown key " + chord;
                return;
            }
        }
        for (ImGuiKey k : keys) io.AddKeyEvent(k, true);
        // Released in reverse order, the key before its modifiers.
        held_keys_.assign(keys.rbegin(), keys.rend());
    }

    // A point of a builder item: fractions of it, or pixels from its far
    // edge for a negative value.
    static bool item_point(SpecPage& page, const std::string& name, float fx, float fy, float* x, float* y) {
#if CALIPER_WITH_BUILDER
        BuilderPane* b = page.builder();
        SpecRect r;
        float pr[4];
        if (page.item_rect(name, pr)) r = SpecRect{pr[0], pr[1], pr[2], pr[3]};
        else if (!b || !b->item_rect(name, &r)) return false;
        *x = std::round(fx < 0.0f ? r.right() + fx : fx > 1.0f ? r.x + fx : r.x + r.w * fx);
        *y = std::round(fy < 0.0f ? r.bottom() + fy : fy > 1.0f ? r.y + fy : r.y + r.h * fy);
        return true;
#else
        (void)page, (void)name, (void)fx, (void)fy, (void)x, (void)y;
        return false;
#endif
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
    // When the current step began, and the seconds a step may wait (0: kPatience frames).
    std::chrono::steady_clock::time_point step_start_ = std::chrono::steady_clock::now();
    double patience_s_ = 0.0;
    bool armed_ = false, shot_requested_ = false, quit_ = false, mouse_down_ = false;
    std::vector<ImGuiKey> held_keys_;
    std::vector<std::string> log_;
    std::string error_, facts_;
#if CALIPER_WITH_BUILDER
    AgentSettings* settings_ = nullptr;
#endif
    std::string page_, command_line_, landing_selected_;
    std::vector<std::string> landing_projects_;
    std::function<void(std::vector<std::string>)> drop_;
    std::function<int()> cards_;
    SavePrompt* save_prompt_ = nullptr;
    std::function<void()> close_;
    float drag_a_[2] = {0, 0}, drag_b_[2] = {0, 0};
    int drag_frame_ = 0;   // the frame of a drag in progress, 0 before its press
    std::optional<std::chrono::steady_clock::time_point> started_;   // a step's first frame, for a wait in seconds
    int idle_frames_ = 0;   // frames in a row without a list or a probe running
    static constexpr int kSettle = 3;   // frames after a step of input
    int settle_ = 0;
};

}  // namespace caliper_host
