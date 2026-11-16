#pragma once
// ============================================================================
// save_prompt.h: "Save changes to <name>?", the question the app asks before
// one of the person's own actions replaces or closes an open spec that has
// unsaved changes: F3 New dashboard (and "new <title>" on the command line),
// opening another dashboard from the landing screen, and closing the window.
// Save sends applet.save of the open spec from the `ui` origin, over its own
// file when it was opened from one (as ⌘S Save does), and goes on once it has
// succeeded; Don't Save goes on; Cancel leaves everything as it
// was. ⌘Z still restores a spec replaced by spec.new, spec.open or
// applet.open, as before.
//
// Only the person's own actions ask: the app calls ask() from the landing
// screen's ways in and from the window's close request, never from the
// command layer. An agent's replacement of an open spec reaches the session
// as a command of its own origin and is reviewed on an approval card
// (approval_cards.h); it never meets this prompt, and this prompt never
// answers an approval.
//
// A hidden app never shows the question to anyone, so it says on stderr when
// it opens ("[save] Save changes to <name>? waiting for Save, Don't Save or
// Cancel"), as the approval cards say a card waits, and how it was answered.
// The input driver answers it with its `save` step (answer()), as a click on
// its button does; a hidden app's SIGTERM and the driver's `quit` close the
// window directly and never ask.
//
// Drawn in the house style: a modal over the page, the question in the UI
// face, ⏎ Save as the primary plate, esc Cancel as a key button, Don't Save
// as a word button. Enter saves and Esc cancels.
//
// Header-only; frame thread. ask() and answer() may be called inside or
// outside the ImGui frame; frame() runs between two frames, so `go` (a
// spec.new, an applet.open, the window's close) never runs inside one.
// ============================================================================
#include "builder/app_session.h"
#include "ui/sternbild_fonts.h"
#include "ui/sternbild_theme.h"

#include <imgui.h>

#if CALIPER_WITH_BUILDER
#include <caliper_builder/commands/command.h>
#include <caliper_builder/session/session.h>
#endif

#include <algorithm>
#include <chrono>
#include <functional>
#include <future>
#include <iostream>
#include <optional>
#include <string>
#include <utility>

namespace caliper_host {

class SavePrompt {
public:
    using Log = std::function<void(const std::string& event, const std::string& object, const std::string& detail)>;
    void set_log(Log log) { log_ = std::move(log); }

    // True, with the open spec's applet id and its name (its title, else its
    // id), when the session's open spec has changes that are not saved; a
    // new spec that was never saved has. False without the builder, before
    // the session opens, and while its worker runs a command.
    static bool unsaved(AppSession& app, std::string* id = nullptr, std::string* name = nullptr) {
#if CALIPER_WITH_BUILDER
        caliper_builder::Session* s = app.session();
        if (!s || app.worker_busy()) return false;
        const std::optional<caliper_builder::OpenSpec> open = s->open_spec();
        if (!open || !open->dirty) return false;
        if (id) *id = open->spec.applet.id;
        if (name) *name = open->spec.applet.title.value_or(open->spec.applet.id);
        return true;
#else
        (void)app, (void)id, (void)name;
        return false;
#endif
    }

    // One of the person's actions that would replace or close the open spec:
    // `go` runs at the next frame() when the open spec has no unsaved
    // changes, or after Save or Don't Save; Cancel drops it. While the
    // session's worker runs a command the check waits for it. False, and
    // `go` dropped, while another question is open.
    bool ask(std::string what, std::function<void()> go) {
        if (state_ != State::idle) return false;
        what_ = std::move(what);
        go_ = std::move(go);
        error_.clear();
        state_ = State::checking;
        return true;
    }

    bool is_open() const { return state_ == State::asking || state_ == State::saving; }
    bool waiting() const { return state_ != State::idle; }
    const std::string& name() const { return name_; }

    // The answer, as a click on its button: "save", "dont-save" or "cancel".
    // False when no question is open, a save runs, or the answer is unknown.
    bool answer(const std::string& a) {
        if (state_ != State::asking) return false;
        if (a == "save") pending_ = Answer::save;
        else if (a == "dont-save") pending_ = Answer::dont_save;
        else if (a == "cancel") pending_ = Answer::cancel;
        else return false;
        return true;
    }

    // Between two frames: decides a waiting check, acts on the answer, reads
    // the save's result, and runs `go`.
    void frame(AppSession& app) {
        if (state_ == State::checking) {
            if (app.session() && app.worker_busy()) return;   // the open spec cannot be read yet
            std::string id;
            if (!unsaved(app, &id, &name_)) {
                finish();
                return;
            }
            state_ = State::asking;
            say("save changes", "Save changes to " + name_ + "? waiting for Save, Don't Save or Cancel (" + what_ + ")");
        }
        if (state_ == State::asking && pending_ != Answer::none) {
            const Answer a = pending_;
            pending_ = Answer::none;
            if (a == Answer::cancel) {
                say("cancel", "Cancel: " + what_ + " is not done");
                state_ = State::idle;
                go_ = nullptr;
            } else if (a == Answer::dont_save) {
                say("don't save", "Don't Save: " + name_ + " is left unsaved");
                finish();
            } else if (!submit_save(app)) {
                error_ = "the session is not open";
                say("save failed", "Save failed: " + error_);
            }
        }
#if CALIPER_WITH_BUILDER
        if (state_ == State::saving && request_ &&
            request_->wait_for(std::chrono::seconds(0)) == std::future_status::ready) {
            const caliper_builder::CommandResult r = request_->get();
            request_.reset();
            if (r.ok) {
                say("saved", "Save: " + name_ + " saved");
                finish();
            } else {
                // The question stays, with the reason, so nothing is lost.
                error_ = r.error.message.empty() ? r.error.code : r.error.message;
                state_ = State::asking;
                say("save failed", "Save failed: " + error_);
            }
        }
#endif
    }

    // Inside the ImGui frame, after the page: the modal question while open.
    void draw() {
        static const char* const kId = "##save_changes";
        if (!is_open()) return;
        if (!ImGui::IsPopupOpen(kId)) ImGui::OpenPopup(kId);
        const ImGuiViewport* vp = ImGui::GetMainViewport();
        ImGui::SetNextWindowPos(ImVec2(vp->WorkPos.x + vp->WorkSize.x * 0.5f, vp->WorkPos.y + vp->WorkSize.y * 0.4f),
                                ImGuiCond_Always, ImVec2(0.5f, 0.5f));
        ImGui::PushStyleColor(ImGuiCol_ModalWindowDimBg, sternbild::to_vec4(sternbild::with_alpha(sternbild::col::deep, 150)));
        ImGui::PushStyleColor(ImGuiCol_PopupBg, sternbild::to_vec4(sternbild::col::plate));
        ImGui::PushStyleColor(ImGuiCol_Border, sternbild::to_vec4(sternbild::col::rule));
        ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(sternbild::metric::gap, sternbild::metric::gap));
        ImGui::PushStyleVar(ImGuiStyleVar_WindowBorderSize, 1.0f);
        const ImGuiWindowFlags flags = ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoResize |
                                       ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoSavedSettings |
                                       ImGuiWindowFlags_AlwaysAutoResize;
        if (ImGui::BeginPopupModal(kId, nullptr, flags)) {
            if (sternbild::fonts().ui) ImGui::PushFont(sternbild::fonts().ui, sternbild::font_size::ui);
            const std::string question = "Save changes to " + name_ + "?";
            ImGui::TextUnformatted(question.c_str());
            ImGui::Dummy(ImVec2(std::max(360.0f, ImGui::CalcTextSize(question.c_str()).x), sternbild::metric::pad));
            const bool asking = state_ == State::asking;
            ImGui::BeginDisabled(!asking);
            if (sternbild::primary_button("Save", ImVec2(0.0f, 24.0f), "\xe2\x8f\x8e")) answer("save");   // ⏎
            ImGui::SameLine(0.0f, sternbild::metric::gap);
            if (sternbild::key_button("esc", "Cancel")) answer("cancel");
            ImGui::SameLine(0.0f, sternbild::metric::gap);
            if (sternbild::word_button("Don't Save")) answer("dont-save");
            ImGui::EndDisabled();
            if (asking && (ImGui::IsKeyPressed(ImGuiKey_Enter, false) || ImGui::IsKeyPressed(ImGuiKey_KeypadEnter, false)))
                answer("save");
            if (asking && ImGui::IsKeyPressed(ImGuiKey_Escape, false)) answer("cancel");
            if (!error_.empty()) {
                ImGui::Dummy(ImVec2(0.0f, sternbild::metric::pad * 0.5f));
                ImGui::PushStyleColor(ImGuiCol_Text, sternbild::col::faint);
                ImGui::TextUnformatted(error_.c_str());
                ImGui::PopStyleColor();
            }
            if (sternbild::fonts().ui) ImGui::PopFont();
            ImGui::EndPopup();
        }
        ImGui::PopStyleVar(2);
        ImGui::PopStyleColor(3);
    }

private:
    enum class State { idle, checking, asking, saving };
    enum class Answer { none, save, dont_save, cancel };

    void finish() {
        state_ = State::idle;
        std::function<void()> go = std::move(go_);
        go_ = nullptr;
        if (go) go();
    }

    bool submit_save(AppSession& app) {
#if CALIPER_WITH_BUILDER
        caliper_builder::Session* s = app.session();
        if (!s) return false;
        caliper_builder::CommandInvocation inv;
        inv.name = "applet.save";
        inv.args = nlohmann::json::object();
        // The open spec's own applet is saved over; a new one is written new
        // (BuilderPane::save's rule).
        const std::optional<caliper_builder::OpenSpec> open = s->open_spec();
        if (open && open->path) inv.args["overwrite"] = true;
        inv.origin = caliper_builder::CommandOrigin::ui;
        request_ = s->queue().submit(std::move(inv));
        state_ = State::saving;
        return true;
#else
        (void)app;
        return false;
#endif
    }

    void say(const char* event, const std::string& line) {
        std::cerr << "[save] " << line << std::endl;
        if (log_) log_(event, name_, what_);
    }

    State state_ = State::idle;
    Answer pending_ = Answer::none;
    std::string what_, name_, error_;
    std::function<void()> go_;
    Log log_;
#if CALIPER_WITH_BUILDER
    std::optional<std::future<caliper_builder::CommandResult>> request_;
#endif
};

}  // namespace caliper_host
