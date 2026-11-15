#include "ui/applet_page.h"

#include "ui/landing_screen.h"   // draw_host_status_items
#include "ui/sternbild_theme.h"
#include "ui/ui_recovery.h"
#include "host_services.h"
#include "host_version.h"

#include <imgui.h>

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
                                    const char* renderer_name, bool chromeless) {
    Result r;
    // The page bar and the status bar, unless the page is chromeless
    // (CALIPER_CHROMELESS=1).
    if (!chromeless && draw_bars(loader, applet, renderer_name)) r.home = true;

    // Docked desktop. A passthru dock space fills the viewport's work area
    // (between the page bar and the status bar) so the applet's windows can
    // be docked. The central node stays transparent while empty, so a
    // full-viewport applet docked into it looks exactly as before.
    const ImGuiID dock_id = ImGui::DockSpaceOverViewport(
        0, ImGui::GetMainViewport(), ImGuiDockNodeFlags_PassthruCentralNode);

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

// A plate pinned to the bottom-right of the WORK area, so it sits above the
// status bar instead of over it; sized to its content, never docked, never
// saved. Actions are house words, not boxes.
void JobsTray::draw(const std::vector<JobSystem::JobView>& views) {
    if (views.empty() && failed_.empty()) return;
    const ImGuiViewport* vp = ImGui::GetMainViewport();
    constexpr float inset = 12.0f;
    constexpr float bar_w = 240.0f;
    ImGui::SetNextWindowPos({vp->WorkPos.x + vp->WorkSize.x - inset,
                             vp->WorkPos.y + vp->WorkSize.y - inset},
                            ImGuiCond_Always, {1.0f, 1.0f});
    ImGui::SetNextWindowViewport(vp->ID);
    ImGui::PushStyleColor(ImGuiCol_WindowBg, sternbild::col::plate);
    ImGui::Begin("Jobs", nullptr,
                 ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoResize |
                     ImGuiWindowFlags_NoCollapse | ImGuiWindowFlags_NoDocking |
                     ImGuiWindowFlags_NoSavedSettings |
                     ImGuiWindowFlags_AlwaysAutoResize);
    ImGui::PopStyleColor();
    sternbild::label("Jobs");
    bool any_done = !failed_.empty();
    for (auto& v : views) any_done |= !v.running;
    if (any_done) {
        ImGui::SameLine(0.0f, sternbild::metric::gap);
        if (sternbild::word_button("clear finished")) {
            host_job_system().clear_finished();
            failed_.clear();
        }
    }
    auto job_row = [&](const JobSystem::JobView& v) {
        ImGui::PushID((int)v.id);
        ImGui::TextUnformatted(v.label.c_str());
        // Fixed width: the tray auto-sizes, so a bar sized from the window
        // width would feed back into it.
        ImGui::ProgressBar(v.progress, {bar_w, 0.0f},
                           v.message.empty() ? nullptr : v.message.c_str());
        if (v.running) {
            ImGui::SameLine(0.0f, sternbild::metric::gap);
            if (sternbild::word_button("cancel")) host_job_system().request_cancel(v.id);
        }
        ImGui::PopID();
    };
    for (auto& v : views)
        if (v.running) job_row(v);
    // A failure in place of its bar: the status word in capitals, as the
    // Instruments STATUS column writes it, in parchment, the strongest ink
    // after bronze (which stays for focus and selection; the palette has no
    // red), then the reason in secondary, wrapped at the bar's width so a long
    // one never widens the tray. Newest first.
    for (auto it = failed_.rbegin(); it != failed_.rend(); ++it) {
        ImGui::PushID((int)it->id);
        ImGui::TextUnformatted(it->label.c_str());
        const float row_x = ImGui::GetCursorPosX();
        ImGui::PushStyleColor(ImGuiCol_Text, sternbild::col::parchment);
        ImGui::TextUnformatted("FAILED");
        ImGui::PopStyleColor();
        ImGui::SameLine(0.0f, sternbild::metric::pad);
        ImGui::PushTextWrapPos(row_x + bar_w);
        ImGui::PushStyleColor(ImGuiCol_Text, sternbild::col::secondary);
        ImGui::TextUnformatted(it->reason.c_str());
        ImGui::PopStyleColor();
        ImGui::PopTextWrapPos();
        ImGui::PopID();
    }
    for (auto& v : views)
        if (!v.running && !v.failed) job_row(v);
    ImGui::End();
}

} // namespace caliper_host
