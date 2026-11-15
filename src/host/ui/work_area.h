#pragma once
// ============================================================================
// The work area of a page: what the page's bars leave of the window. The page
// bar (or the landing's command bar) takes the top edge, and the status bar
// takes the bottom edge. Each is an ImGui viewport
// side bar, which reserves its height from the work area, and the page's dock
// space fills what is left, so no applet window ever runs under a bar.
//
// ImGui moves the viewport's public WorkPos / WorkSize only on the next
// frame, so a dock space laid out from them would run under a bar for the one
// frame in which the bar first appears. dock_space_over_work_area() lays the
// dock space out from the bars submitted so far in this frame instead: call it
// after every bar of the frame, and the dock space never overlaps one, not
// even for a frame.
//
// Header-only and free of any renderer: the unit test drives it on a
// headless ImGui context.
// ============================================================================
#include <imgui.h>
#include <imgui_internal.h>

namespace caliper_host {

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
