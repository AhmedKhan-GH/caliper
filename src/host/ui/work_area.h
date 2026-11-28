#pragma once
// ============================================================================
// The work area of a page: what the page's bars leave of the window. The page
// bar (or the landing's command bar) takes the top edge, and the status bar,
// then the jobs tray above it, take the bottom edge. Each is an ImGui viewport
// side bar, which reserves its height from the work area, and the page's dock
// space fills what is left, so no instrument window ever runs under a bar.
//
// ImGui moves the viewport's public WorkPos / WorkSize only on the next
// frame, so a dock space laid out from them would run under a bar for the one
// frame in which the bar first appears. dock_space_over_work_area() lays the
// dock space out from the bars submitted so far in this frame instead: call it
// after every bar of the frame, and the dock space never overlaps one, not
// even for a frame.
//
// The jobs tray takes its strip only while it has something to show, and
// JobsTrayPresence decides when, so the work area does not jump: a job must
// run for kOpenDelay before the strip appears, which keeps a job that starts
// and ends within a moment from moving the instrument's panes at all, and the
// strip stays for kCloseDelay after the last running job has ended (showing
// it as finished), so a run of jobs one after another keeps one strip. A kept
// failure holds the strip open until the user clears it, and clearing with
// nothing running closes the strip at once.
//
// Header-only and free of any renderer: the unit test drives it on a
// headless ImGui context.
// ============================================================================
#include <imgui.h>
#include <imgui_internal.h>

namespace caliper_host {

// The jobs tray's strip height: the page bar's height, so a progress bar and
// a word button sit in it with the page bar's margins.
constexpr float kJobsTrayHeight = 28.0f;

// When the jobs tray takes its strip. `wanted` is true while a job runs or a
// failure is kept; `now` is a monotonic clock in seconds.
class JobsTrayPresence {
public:
    static constexpr double kOpenDelay = 0.5;
    static constexpr double kCloseDelay = 2.0;

    bool update(double now, bool wanted) {
        if (wanted) {
            if (wanted_since_ < 0.0) wanted_since_ = now;
            idle_since_ = -1.0;
            if (!open_ && now - wanted_since_ >= kOpenDelay) open_ = true;
        } else {
            wanted_since_ = -1.0;
            if (open_) {
                if (idle_since_ < 0.0) idle_since_ = now;
                if (now - idle_since_ >= kCloseDelay) {
                    open_ = false;
                    idle_since_ = -1.0;
                }
            }
        }
        return open_;
    }
    bool open() const { return open_; }
    // The user cleared the tray with nothing running: the strip goes now,
    // without the close delay, because the user asked for the space back.
    void close() {
        open_ = false;
        idle_since_ = -1.0;
    }

private:
    bool open_ = false;
    double wanted_since_ = -1.0;   // start of the current wanted stretch
    double idle_since_ = -1.0;     // start of the current idle stretch while open
};

// The work area left by the bars submitted so far in this frame.
inline ImRect work_area() {
    return static_cast<ImGuiViewportP*>(ImGui::GetMainViewport())->GetBuildWorkRect();
}

// ImGui::DockSpaceOverViewport() over work_area() rather than last frame's
// WorkPos / WorkSize. The host window and the default dock space id are the
// ones DockSpaceOverViewport() uses, so layouts saved in imgui.ini still apply.
inline ImGuiID dock_space_over_work_area(ImGuiID dockspace_id, ImGuiDockNodeFlags flags) {
    const ImGuiViewport* viewport = ImGui::GetMainViewport();
    const ImRect area = work_area();
    ImGui::SetNextWindowPos(area.Min);
    ImGui::SetNextWindowSize(area.GetSize());
    ImGui::SetNextWindowViewport(viewport->ID);

    ImGuiWindowFlags host_flags = ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoCollapse |
                                  ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoMove |
                                  ImGuiWindowFlags_NoDocking |
                                  ImGuiWindowFlags_NoBringToFrontOnFocus |
                                  ImGuiWindowFlags_NoNavFocus;
    if (flags & ImGuiDockNodeFlags_PassthruCentralNode)
        host_flags |= ImGuiWindowFlags_NoBackground;

    char label[32];
    ImFormatString(label, IM_COUNTOF(label), "WindowOverViewport_%08X", viewport->ID);
    ImGui::PushStyleVar(ImGuiStyleVar_WindowRounding, 0.0f);
    ImGui::PushStyleVar(ImGuiStyleVar_WindowBorderSize, 0.0f);
    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0.0f, 0.0f));
    ImGui::Begin(label, nullptr, host_flags);
    ImGui::PopStyleVar(3);
    if (dockspace_id == 0) dockspace_id = ImGui::GetID("DockSpace");
    ImGui::DockSpace(dockspace_id, ImVec2(0.0f, 0.0f), flags);
    ImGui::End();
    return dockspace_id;
}

} // namespace caliper_host
