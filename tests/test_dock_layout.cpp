// Docking by role on the applet page (src/host/ui/dock_layout.h), driven on a
// headless ImGui context the way the host drives it: the dock space, then
// apply(), then the applet's windows between begin_applet_frame() and
// end_applet_frame(), with the hints an applet gives through caliper.ui.v1_1.
#include <doctest/doctest.h>
#include "ui/dock_layout.h"
#include <cmath>
#include <cstdint>
#include <functional>
#include <string>
#include <vector>

using caliper_host::DockLayout;

namespace {

struct Ui {
    ImGuiContext* ctx = nullptr;
    DockLayout layout;
    ImGuiID dockspace = 0;

    explicit Ui(const char* ini = nullptr) {
        ctx = ImGui::CreateContext();
        ImGuiIO& io = ImGui::GetIO();
        io.IniFilename = nullptr;
        io.LogFilename = nullptr;
        io.ConfigFlags |= ImGuiConfigFlags_DockingEnable;
        io.BackendFlags |= ImGuiBackendFlags_RendererHasTextures;
        io.DisplaySize = ImVec2(1400.0f, 900.0f);
        io.DeltaTime = 1.0f / 60.0f;
        if (ini) ImGui::LoadIniSettingsFromMemory(ini);
    }
    ~Ui() { ImGui::DestroyContext(ctx); }

    // One frame of the applet page; `applet` draws the applet's windows.
    void frame(const std::function<void()>& applet) {
        ImGui::NewFrame();
        dockspace = ImGui::DockSpaceOverViewport(0, ImGui::GetMainViewport(),
                                                 ImGuiDockNodeFlags_PassthruCentralNode);
        layout.apply(dockspace);
        layout.begin_applet_frame();
        applet();
        layout.end_applet_frame();
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

    // A window as an applet draws it, its hint before Begin() (role < 0: none).
    void window(const char* title, int role, bool hint_after = false) {
        if (role >= 0 && !hint_after) layout.hint(title, (uint32_t)role);
        ImGui::SetNextWindowSize(ImVec2(300.0f, 200.0f), ImGuiCond_FirstUseEver);
        ImGui::Begin(title);
        ImGui::TextUnformatted(title);
        ImGui::End();
        if (role >= 0 && hint_after) layout.hint(title, (uint32_t)role);
    }

    static ImGuiWindow* find(const char* title) { return ImGui::FindWindowByName(title); }
    static ImGuiID dock_of(const char* title) {
        ImGuiWindow* w = find(title);
        return w ? w->DockId : 0;
    }
};

constexpr int kCentral = CALIPER_DOCK_CENTRAL;
constexpr int kTop = CALIPER_DOCK_SIDE_TOP;
constexpr int kBottom = CALIPER_DOCK_SIDE_BOTTOM;

} // namespace

TEST_CASE("dock layout: roles tile a first run as the title lists did") {
    Ui ui;
    auto applet = [&] {
        ui.window("Main", kCentral);
        ui.window("Controls", kTop);
        ui.window("Details", kBottom);
        ui.window("More details", kBottom);
    };
    ui.frame(applet);
    // Hints given before Begin() dock in the frame the window first appears.
    CHECK(Ui::dock_of("Main") != 0);
    CHECK(Ui::dock_of("Controls") != 0);
    CHECK(Ui::dock_of("Details") != 0);
    for (int i = 0; i < 3; i++) ui.frame(applet);

    ImGuiWindow* main = Ui::find("Main");
    ImGuiWindow* top = Ui::find("Controls");
    ImGuiWindow* bottom = Ui::find("Details");
    CHECK(Ui::dock_of("Main") == ui.layout.node_for_role(kCentral));
    CHECK(Ui::dock_of("Controls") == ui.layout.node_for_role(kTop));
    CHECK(Ui::dock_of("Details") == ui.layout.node_for_role(kBottom));
    CHECK(Ui::dock_of("More details") == Ui::dock_of("Details"));   // tabs
    // The first-run geometry at 1400 x 900: the centre on the left, the side
    // column on the right with about a fifth of the width (the earlier
    // layout's 285 px, to within a pixel), split 60/40 top to bottom.
    CHECK(main->Pos.x == 0.0f);
    CHECK(main->Pos.y == 0.0f);
    CHECK(main->Size.y == 900.0f);
    CHECK(std::fabs(top->Size.x - 285.0f) <= 1.0f);
    CHECK(top->Pos.x == main->Size.x + 2.0f);
    CHECK(top->Pos.y == 0.0f);
    CHECK(top->Size.y == 538.0f);
    CHECK(bottom->Pos.x == top->Pos.x);
    CHECK(bottom->Pos.y == 540.0f);
    CHECK(bottom->Size.y == 360.0f);
}

TEST_CASE("dock layout: a central-only applet fills the work area") {
    Ui ui;
    for (int i = 0; i < 3; i++) ui.frame([&] { ui.window("Only view", kCentral); });
    ImGuiWindow* w = Ui::find("Only view");
    CHECK(w->DockId == ui.layout.node_for_role(kCentral));
    CHECK(w->Size.x == 1400.0f);
    CHECK(w->Size.y == 900.0f);
}

TEST_CASE("dock layout: a window without a hint docks in the centre") {
    Ui ui;
    auto applet = [&] {
        ui.window("Hinted", kCentral);
        ui.window("Unhinted", -1);
        ImGui::Begin("No docking", nullptr, ImGuiWindowFlags_NoDocking);
        ImGui::End();
        ImGui::Begin("Parent", nullptr, ImGuiWindowFlags_NoDocking);
        ImGui::BeginChild("child", ImVec2(50.0f, 50.0f));
        ImGui::EndChild();
        ImGui::End();
    };
    ui.frame(applet);
    CHECK(Ui::dock_of("Unhinted") == 0);   // found after the frame it began in
    ui.frame(applet);
    CHECK(Ui::dock_of("Unhinted") == Ui::dock_of("Hinted"));
    CHECK(Ui::dock_of("No docking") == 0);
    CHECK(Ui::dock_of("Parent") == 0);
}

TEST_CASE("dock layout: a hint after Begin() takes effect on the next frame") {
    Ui ui;
    auto applet = [&] {
        ui.window("Main", kCentral, true);
        ui.window("Controls", kTop, true);
    };
    ui.frame(applet);
    ui.frame(applet);
    CHECK(Ui::dock_of("Main") == ui.layout.node_for_role(kCentral));
    CHECK(Ui::dock_of("Controls") == ui.layout.node_for_role(kTop));
}

TEST_CASE("dock layout: an unknown role is the centre") {
    Ui ui;
    for (int i = 0; i < 2; i++)
        ui.frame([&] {
            ui.window("Main", kCentral);
            ui.window("Odd", 77);
        });
    CHECK(Ui::dock_of("Odd") == Ui::dock_of("Main"));
}

TEST_CASE("dock layout: a saved layout wins over every hint") {
    // An imgui.ini entry for "Controls": the user left it floating there.
    Ui ui("[Window][Controls]\nPos=50,60\nSize=200,100\nCollapsed=0\n");
    for (int i = 0; i < 3; i++)
        ui.frame([&] {
            ui.window("Main", kCentral);
            ui.window("Controls", kTop);
        });
    ImGuiWindow* w = Ui::find("Controls");
    CHECK(w->DockId == 0);
    CHECK(w->Pos.x == 50.0f);
    CHECK(w->Pos.y == 60.0f);
    CHECK(Ui::dock_of("Main") != 0);
}

TEST_CASE("dock layout: a window the user undocks stays undocked") {
    Ui ui;
    auto applet = [&] {
        ui.window("Main", kCentral);
        ui.window("Controls", kTop);
    };
    for (int i = 0; i < 2; i++) ui.frame(applet);
    REQUIRE(Ui::dock_of("Controls") != 0);
    ImGui::DockContextQueueUndockWindow(ImGui::GetCurrentContext(), Ui::find("Controls"));
    for (int i = 0; i < 4; i++) ui.frame(applet);   // hinted every frame
    CHECK(Ui::dock_of("Controls") == 0);
}

TEST_CASE("dock layout: a window that first appears later still docks by role") {
    Ui ui;
    for (int i = 0; i < 3; i++) ui.frame([&] { ui.window("Main", kCentral); });
    CHECK(ui.layout.node_for_role(kCentral) == Ui::dock_of("Main"));
    auto later = [&] {
        ui.window("Main", kCentral);
        ui.window("Late details", kBottom);
        ui.window("Late controls", kTop, true);
    };
    for (int i = 0; i < 3; i++) ui.frame(later);
    CHECK(Ui::dock_of("Late details") == ui.layout.node_for_role(kBottom));
    CHECK(Ui::dock_of("Late controls") == ui.layout.node_for_role(kTop));
    CHECK(Ui::dock_of("Main") == ui.layout.node_for_role(kCentral));
    CHECK(Ui::find("Late details")->Pos.x > Ui::find("Main")->Pos.x);
}

TEST_CASE("dock layout: a restored layout keeps its windows and adds new ones") {
    // First run: build a layout and save it, as imgui.ini would.
    std::string ini;
    {
        Ui first;
        for (int i = 0; i < 3; i++)
            first.frame([&] {
                first.window("Main", kCentral);
                first.window("Controls", kTop);
            });
        ini = ImGui::SaveIniSettingsToMemory();
    }
    // Second run: the saved windows are where the user left them; a window
    // new in this version of the applet still docks by its role.
    Ui ui(ini.c_str());
    auto applet = [&] {
        ui.window("Main", kCentral);
        ui.window("Controls", kTop);
        ui.window("New details", kBottom);
    };
    for (int i = 0; i < 3; i++) ui.frame(applet);
    ImGuiWindow* main = Ui::find("Main");
    ImGuiWindow* controls = Ui::find("Controls");
    CHECK(main->DockId != 0);
    CHECK(controls->DockId != 0);
    CHECK(controls->Pos.x > main->Pos.x);
    CHECK(Ui::dock_of("New details") == ui.layout.node_for_role(kBottom));
    CHECK(Ui::dock_of("New details") != Ui::dock_of("Main"));
}
