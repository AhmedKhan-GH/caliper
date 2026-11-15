// The work area of a page (src/host/ui/work_area.h): the dock space ends
// above every bar of the page, in the very frame a bar appears. Driven on a headless ImGui
// context the way the host draws a page: the bars as viewport side bars, then
// the dock space over the work area they leave.
#include <doctest/doctest.h>
#include "ui/work_area.h"
#include <cstdint>
#include <functional>


namespace {

constexpr float kWidth = 1400.0f;
constexpr float kHeight = 900.0f;
constexpr float kPageBar = 28.0f;
constexpr float kStatusBar = 22.0f;
constexpr float kJobsTrayHeight = 28.0f;   // a bar that comes and goes above the status bar

struct Page {
    ImGuiContext* ctx = nullptr;
    ImGuiID dockspace = 0;

    Page() {
        ctx = ImGui::CreateContext();
        ImGuiIO& io = ImGui::GetIO();
        io.IniFilename = nullptr;
        io.LogFilename = nullptr;
        io.ConfigFlags |= ImGuiConfigFlags_DockingEnable;
        io.BackendFlags |= ImGuiBackendFlags_RendererHasTextures;
        io.DisplaySize = ImVec2(kWidth, kHeight);
        io.DeltaTime = 1.0f / 60.0f;
    }
    ~Page() { ImGui::DestroyContext(ctx); }

    static void bar(const char* name, ImGuiDir dir, float height) {
        ImGui::BeginViewportSideBar(name, ImGui::GetMainViewport(), dir, height,
                                    ImGuiWindowFlags_NoSavedSettings);
        ImGui::End();
    }

    // One frame of a page: the page bar, the status bar, the tray's strip
    // when `tray` is set, then the dock space; `inside` runs before Render().
    void frame(bool tray, const std::function<void()>& inside = {}) {
        ImGui::NewFrame();
        bar("##page_bar", ImGuiDir_Up, kPageBar);
        bar("##status_bar", ImGuiDir_Down, kStatusBar);
        if (tray) bar("##tray_bar", ImGuiDir_Down, kJobsTrayHeight);
        dockspace = caliper_host::dock_space_over_work_area(
            0, ImGuiDockNodeFlags_PassthruCentralNode);
        if (inside) inside();
        ImGui::Render();
        for (ImTextureData* tex : ImGui::GetPlatformIO().Textures) {
            if (tex->Status == ImTextureStatus_WantCreate ||
                tex->Status == ImTextureStatus_WantUpdates) {
                tex->SetTexID((ImTextureID)(intptr_t)1);
                tex->SetStatus(ImTextureStatus_OK);
            } else if (tex->Status == ImTextureStatus_WantDestroy) {
                tex->SetTexID(ImTextureID_Invalid);
                tex->SetStatus(ImTextureStatus_Destroyed);
            }
        }
    }

    ImRect dock_rect() const {
        ImGuiDockNode* node = ImGui::DockBuilderGetNode(dockspace);
        REQUIRE(node != nullptr);
        return ImRect(node->Pos, ImVec2(node->Pos.x + node->Size.x, node->Pos.y + node->Size.y));
    }
    static ImRect window_rect(const char* name) {
        ImGuiWindow* w = ImGui::FindWindowByName(name);
        REQUIRE(w != nullptr);
        return w->Rect();
    }
};

} // namespace

TEST_CASE("work area: the dock space ends above the tray, the tray above the status bar") {
    Page page;
    page.frame(false);
    page.frame(false);
    const ImRect without = page.dock_rect();
    CHECK(without.Min.y == doctest::Approx(kPageBar));
    CHECK(without.Max.y == doctest::Approx(kHeight - kStatusBar));
    CHECK(without.GetWidth() == doctest::Approx(kWidth));

    // The first frame with the tray: the dock space already ends above it,
    // so no applet window runs under the strip even for one frame.
    page.frame(true);
    const ImRect tray = Page::window_rect("##tray_bar");
    const ImRect status = Page::window_rect("##status_bar");
    const ImRect with = page.dock_rect();
    CHECK(tray.GetHeight() == doctest::Approx(kJobsTrayHeight));
    CHECK(tray.Max.y == doctest::Approx(status.Min.y));
    CHECK(status.Max.y == doctest::Approx(kHeight));
    CHECK(with.Max.y == doctest::Approx(tray.Min.y));
    CHECK(with.Min.y == doctest::Approx(kPageBar));
    CHECK(with.GetHeight() ==
          doctest::Approx(kHeight - kPageBar - kStatusBar - kJobsTrayHeight));

    // A window docked in the dock space ends above the strip too.
    page.frame(true, [&] {
        ImGui::SetNextWindowDockID(page.dockspace, ImGuiCond_Always);
        ImGui::Begin("Applet");
        ImGui::End();
    });
    page.frame(true, [&] {
        ImGui::SetNextWindowDockID(page.dockspace, ImGuiCond_Always);
        ImGui::Begin("Applet");
        ImGui::End();
    });
    const ImRect applet = Page::window_rect("Applet");
    CHECK(applet.Max.y <= tray.Min.y + 0.5f);

    // The tray goes: the dock space reaches the status bar again at once.
    page.frame(false);
    CHECK(page.dock_rect().Max.y == doctest::Approx(kHeight - kStatusBar));
}

TEST_CASE("work area: the dock space keeps DockSpaceOverViewport's id") {
    // So a layout saved in imgui.ini before the change still applies.
    ImGuiID over_viewport = 0, over_work_area = 0;
    {
        Page page;
        ImGui::NewFrame();
        over_viewport = ImGui::DockSpaceOverViewport(0, ImGui::GetMainViewport(),
                                                     ImGuiDockNodeFlags_PassthruCentralNode);
        ImGui::EndFrame();
    }
    {
        Page page;
        ImGui::NewFrame();
        over_work_area = caliper_host::dock_space_over_work_area(
            0, ImGuiDockNodeFlags_PassthruCentralNode);
        ImGui::EndFrame();
    }
    CHECK(over_viewport != 0);
    CHECK(over_viewport == over_work_area);
}
