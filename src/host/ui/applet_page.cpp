#include "ui/applet_page.h"

#include "ui/landing_screen.h"   // draw_host_status_items
#include "ui/sternbild_theme.h"
#include "ui/ui_recovery.h"
#include "host_services.h"
#include "host_version.h"

#include <imgui.h>
#include <imgui_internal.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstring>

namespace caliper_host {

namespace {

double now_ms() {
    using clock = std::chrono::steady_clock;
    return std::chrono::duration<double, std::milli>(clock::now().time_since_epoch())
        .count();
}

} // namespace

CaliperHost applet_host_proto() {
    CaliperHost proto{};
    proto.struct_size  = sizeof(CaliperHost);
    proto.abi_epoch    = 2;
    proto.host_version = kHostVersionU32;
    proto.applet_data_dir = nullptr;   // the loader fills it per applet
    proto.get_service = [](const CaliperHost*, const char* id) {
        return services_get(id);
    };
    return proto;
}

bool begin_page_bar() {
    const float bar_pad_y =
        (sternbild::metric::command_bar - ImGui::GetFontSize()) * 0.5f;
    ImGui::PushStyleVar(ImGuiStyleVar_FramePadding,
                        ImVec2(ImGui::GetStyle().FramePadding.x, bar_pad_y));
    const bool open = ImGui::BeginMainMenuBar();
    ImGui::PopStyleVar();
    return open;
}

AppletPage::Result AppletPage::draw(AppletLoader& loader, int applet,
                                    const CaliperFrameInfo& fi,
                                    const char* renderer_name, bool chromeless,
                                    JobsTray* tray) {
    Result r;
    // The page bar and the status bar, then the jobs tray's strip above the
    // status bar while it is open, unless the page is chromeless
    // (CALIPER_CHROMELESS=1), which promises the applet the whole window.
    if (!chromeless) {
        if (draw_bars(loader, applet, renderer_name)) r.home = true;
        if (tray) tray->draw();
    }

    // Docked desktop. A passthru dock space fills the work area the bars
    // leave this frame (work_area.h), so the applet's windows can be docked
    // and none runs under a bar. The central node stays transparent while
    // empty, so a full-viewport applet docked into it looks exactly as before.
    const ImGuiID dock_id =
        dock_space_over_work_area(0, ImGuiDockNodeFlags_PassthruCentralNode);

    // Docking by role (dock_layout.h): windows the applet hinted, or began
    // without a hint, and that have no saved layout are placed the first time
    // they appear. A layout the user arranged is never touched.
    dock_layout_.apply(dock_id);

    // The UI stacks are checkpointed around the applet's frame: a throw or
    // fault partway through it leaves its windows, plots and pushes open, and
    // they are unwound before the host draws anything more into this frame.
    const UiCheckpoint ui_checkpoint = UiCheckpoint::store();
    const double t0 = now_ms();
    dock_layout_.begin_applet_frame();
    r.alive = loader.frame(applet, fi);
    dock_layout_.end_applet_frame();
    watchdog_.feed(now_ms() - t0);
    if (!r.alive) ui_checkpoint.recover();   // quarantined mid-frame
    return r;
}

// The applet page's chrome: the page bar over the work area and the status
// bar under it. True when the page bar's Home was clicked.
bool AppletPage::draw_bars(const AppletLoader& loader, int applet,
                           const char* renderer_name) {
    bool home = false;
    // Applet bar: Home as one key button holding its key, esc (recognition
    // over recall, nothing printed beside it), the applet's name as metadata,
    // and the watchdog warning in bronze, the palette's only accent. The
    // page's one action lives here; the status bar below carries only facts.
    if (begin_page_bar()) {
        if (sternbild::key_button("esc", "Home")) home = true;
        ImGui::Separator();
        ImGui::PushStyleColor(ImGuiCol_Text, sternbild::col::secondary);
        ImGui::TextUnformatted(loader.at(applet).manifest.name.c_str());
        ImGui::PopStyleColor();
        if (watchdog_.flagged()) {
            ImGui::Separator();
            ImGui::PushStyleColor(ImGuiCol_Text, sternbild::col::bronze);
            ImGui::TextUnformatted("slow: long work belongs in background jobs");
            ImGui::PopStyleColor();
        }
        ImGui::EndMainMenuBar();
    }

    // Status bar (VS Code): the same persistent strip as the landing's, so
    // the machine is always in the same place; facts only, as there (Home is
    // in the applet bar above). Like the menu bar it is a viewport side bar,
    // drawn before the dock space that fills what it leaves.
    sternbild::status_bar_begin();
    ::draw_host_status_items(renderer_name);
    {
        int running = 0;
        for (const auto& v : host_job_system().views()) running += v.running ? 1 : 0;
        char jobs[48];
        if (running == 0)
            std::snprintf(jobs, sizeof jobs, "no jobs");
        else
            std::snprintf(jobs, sizeof jobs,
                          running == 1 ? "%d job running" : "%d jobs running", running);
        sternbild::status_item(jobs);
    }
    sternbild::status_bar_end();
    return home;
}

void JobsTray::note_failures(const std::vector<JobSystem::JobView>& views,
                             const std::function<void(const Failure&)>& on_new) {
    for (const auto& v : views) {
        if (!v.failed) continue;
        bool known = false;
        for (const auto& f : failed_) known |= f.id == v.id;
        if (known) continue;
        // JobSystem writes the reason as the message, "failed: <why>".
        std::string reason = v.message;
        constexpr const char* kPrefix = "failed: ";
        if (reason.rfind(kPrefix, 0) == 0) reason.erase(0, std::strlen(kPrefix));
        if (reason.empty()) reason = "no reason given";
        failed_.push_back({v.id, v.label, std::move(reason)});
        if (on_new) on_new(failed_.back());
    }
}

void JobsTray::begin_frame(std::vector<JobSystem::JobView> views, double now) {
    views_ = std::move(views);
    bool wanted = !failed_.empty();
    for (const auto& v : views_) wanted |= v.running;
    presence_.update(now, wanted);
}

void JobsTray::clear_finished() {
    host_job_system().clear_finished();
    failed_.clear();
    // The user asked for the space back: with nothing running, the strip
    // closes now rather than after the close delay.
    bool running = false;
    for (const auto& v : views_) running |= v.running;
    if (!running) presence_.close();
}

namespace {

constexpr float kTrayBarW = 160.0f;      // a running job's progress bar in the strip
constexpr float kTrayReasonW = 280.0f;   // a failure's reason, cut with an ellipsis
constexpr float kListBarW = 240.0f;      // a progress bar in the popup's list

float text_w(const char* s) { return ImGui::CalcTextSize(s).x; }

// Text in `colour` cut to `max_w` with an ellipsis, on the line's baseline as
// TextUnformatted puts it.
void text_cut(const char* text, float max_w, ImU32 colour) {
    ImGuiWindow* window = ImGui::GetCurrentWindow();
    const ImVec2 size = ImGui::CalcTextSize(text);
    const float w = std::min(size.x, max_w);
    const ImVec2 at(window->DC.CursorPos.x,
                    window->DC.CursorPos.y + window->DC.CurrLineTextBaseOffset);
    const ImRect bb(at, ImVec2(at.x + w, at.y + size.y));
    ImGui::ItemSize(ImVec2(w, size.y), 0.0f);
    if (!ImGui::ItemAdd(bb, 0)) return;
    ImGui::PushStyleColor(ImGuiCol_Text, colour);
    ImGui::RenderTextEllipsis(window->DrawList, bb.Min, bb.Max, bb.Max.x, text, nullptr,
                              &size);
    ImGui::PopStyleColor();
}

} // namespace

// The strip: one line, the running jobs, then the kept failures (newest
// first), as many as fit, then "+N more" for the rest; at the right end, how
// many finished and "clear finished". Actions are house words, not boxes.
// The next strip, when one is set, follows it.
void JobsTray::draw() {
    draw_tray();
    if (next_strip_) next_strip_();
}

void JobsTray::draw_tray() {
    if (!presence_.open()) return;
    sternbild::tray_bar_begin(kJobsTrayHeight);
    ImGuiWindow* window = ImGui::GetCurrentWindow();
    const float gap = sternbild::metric::gap;
    const float pad = sternbild::metric::pad;

    std::vector<const JobSystem::JobView*> running;
    int finished = 0;
    for (const auto& v : views_) {
        if (v.running) running.push_back(&v);
        else if (!v.failed) finished++;
    }
    const bool any_done = finished > 0 || !failed_.empty();
    char finished_text[32] = "";
    if (finished > 0)
        std::snprintf(finished_text, sizeof finished_text, "%d finished", finished);

    // The right end, measured first so the entries stop short of it.
    float right_w = 0.0f;
    if (finished > 0) right_w += text_w(finished_text);
    if (any_done) right_w += (finished > 0 ? gap : 0.0f) + text_w("clear finished");
    const float right_x = window->WorkRect.Max.x - right_w;
    const float more_w = text_w("+999 more");

    sternbild::label("Jobs");
    const int total = (int)running.size() + (int)failed_.size();
    int shown = 0;
    for (int i = 0; i < total; i++) {
        const bool is_running = i < (int)running.size();
        const JobSystem::JobView* v = is_running ? running[(size_t)i] : nullptr;
        const Failure* f =
            is_running ? nullptr : &failed_[failed_.size() - 1 - (size_t)(i - running.size())];
        const char* label = is_running ? v->label.c_str() : f->label.c_str();
        const float entry_w =
            is_running ? text_w(label) + pad + kTrayBarW + pad + text_w("cancel")
                       : text_w(label) + pad + text_w("FAILED") + pad +
                             std::min(text_w(f->reason.c_str()), kTrayReasonW);
        const float after = i + 1 < total ? gap + more_w : 0.0f;
        const float x = window->DC.CursorPosPrevLine.x + gap;   // where SameLine puts it
        if (x + entry_w + after > right_x - gap) break;
        ImGui::SameLine(0.0f, gap);
        ImGui::PushID(is_running ? (int)v->id : -(int)f->id);
        ImGui::TextUnformatted(label);
        ImGui::SameLine(0.0f, pad);
        if (is_running) {
            ImGui::ProgressBar(v->progress, {kTrayBarW, 0.0f},
                               v->message.empty() ? nullptr : v->message.c_str());
            ImGui::SameLine(0.0f, pad);
            if (sternbild::word_button("cancel")) host_job_system().request_cancel(v->id);
        } else {
            // The status word in capitals, as the Instruments STATUS column
            // writes it, in parchment, the strongest ink after bronze (which
            // stays for focus and selection; the palette has no red), then
            // the reason in secondary.
            ImGui::PushStyleColor(ImGuiCol_Text, sternbild::col::parchment);
            ImGui::TextUnformatted("FAILED");
            ImGui::PopStyleColor();
            ImGui::SameLine(0.0f, pad);
            text_cut(f->reason.c_str(), kTrayReasonW, sternbild::col::secondary);
        }
        ImGui::PopID();
        shown++;
    }
    if (shown < total) {
        char more[32];
        std::snprintf(more, sizeof more, "+%d more", total - shown);
        ImGui::SameLine(0.0f, gap);
        if (sternbild::word_button(more)) ImGui::OpenPopup("##jobs_tray_list");
    }
    if (ImGui::BeginPopup("##jobs_tray_list")) {
        draw_list();
        ImGui::EndPopup();
    }

    if (right_w > 0.0f) {
        ImGui::SameLine(right_x - window->Pos.x);
        if (finished > 0) {
            ImGui::PushStyleColor(ImGuiCol_Text, sternbild::col::secondary);
            ImGui::TextUnformatted(finished_text);
            ImGui::PopStyleColor();
            ImGui::SameLine(0.0f, gap);
        }
        if (sternbild::word_button("clear finished")) clear_finished();
    }
    sternbild::tray_bar_end();
}

// Every entry, one per row: running jobs first, then failures (newest first),
// then the jobs that finished.
void JobsTray::draw_list() {
    auto job_row = [&](const JobSystem::JobView& v) {
        ImGui::PushID((int)v.id);
        ImGui::TextUnformatted(v.label.c_str());
        // Fixed width: the popup auto-sizes, so a bar sized from the window
        // width would feed back into it.
        ImGui::ProgressBar(v.progress, {kListBarW, 0.0f},
                           v.message.empty() ? nullptr : v.message.c_str());
        if (v.running) {
            ImGui::SameLine(0.0f, sternbild::metric::gap);
            if (sternbild::word_button("cancel")) host_job_system().request_cancel(v.id);
        }
        ImGui::PopID();
    };
    for (const auto& v : views_)
        if (v.running) job_row(v);
    // A failure in place of its bar, the reason wrapped at the bar's width so
    // a long one never widens the popup.
    for (auto it = failed_.rbegin(); it != failed_.rend(); ++it) {
        ImGui::PushID(-(int)it->id);
        ImGui::TextUnformatted(it->label.c_str());
        const float row_x = ImGui::GetCursorPosX();
        ImGui::PushStyleColor(ImGuiCol_Text, sternbild::col::parchment);
        ImGui::TextUnformatted("FAILED");
        ImGui::PopStyleColor();
        ImGui::SameLine(0.0f, sternbild::metric::pad);
        ImGui::PushTextWrapPos(row_x + kListBarW);
        ImGui::PushStyleColor(ImGuiCol_Text, sternbild::col::secondary);
        ImGui::TextUnformatted(it->reason.c_str());
        ImGui::PopStyleColor();
        ImGui::PopTextWrapPos();
        ImGui::PopID();
    }
    for (const auto& v : views_)
        if (!v.running && !v.failed) job_row(v);
}

} // namespace caliper_host
