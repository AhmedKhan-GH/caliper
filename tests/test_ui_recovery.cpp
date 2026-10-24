// UI recovery after an applet fault (src/host/ui/ui_recovery.h). The throwing
// fixture opens a window, a child, a table, a tree node, a 2-D and a 3-D plot
// and a pile of pushes on the host's contexts, then throws (or writes through
// null) with all of it still open, inside the loader's guarded frame(). The
// host's checkpoint unwinds it, and the host's own window and the next frame,
// plots included, run as if nothing had happened.
//
// GPU-free: ImGui runs headless (no renderer backend; a stand-in marks the
// font atlas uploaded), which is all the stacks need.
#include <doctest/doctest.h>
#include "applet_loader.h"
#include "ui/ui_recovery.h"
#include <caliper/fixture_host.h>
#include <caliper/services/ui_v1.h>
#include <cstdint>
#include <cstdlib>
#include <string>
#ifdef _WIN32
#define setenv(k, v, overwrite) _putenv_s(k, v)
#define unsetenv(k) _putenv_s(k, "")
#endif
using namespace caliper_host;

namespace {

// ImGui, ImPlot and ImPlot3D contexts with no renderer, as the host makes
// them (core_create_ui_context), handed to the fixture through caliper.ui.v1.
struct HeadlessUi {
    ImGuiContext* imgui = nullptr;
    ImPlotContext* implot = nullptr;
    ImPlot3DContext* implot3d = nullptr;

    HeadlessUi() {
        imgui = ImGui::CreateContext();
        implot = ImPlot::CreateContext();
        implot3d = ImPlot3D::CreateContext();
        ImGuiIO& io = ImGui::GetIO();
        io.IniFilename = nullptr;
        io.LogFilename = nullptr;
        io.ConfigFlags |= ImGuiConfigFlags_DockingEnable;
        io.BackendFlags |= ImGuiBackendFlags_RendererHasTextures;
        io.DisplaySize = ImVec2(1280.0f, 800.0f);
        io.DeltaTime = 1.0f / 60.0f;
        // Keep any recoverable error the host itself might make visible to
        // the test, never an abort or a tooltip.
        io.ConfigErrorRecoveryEnableAssert = false;
        io.ConfigErrorRecoveryEnableTooltip = false;
        io.ConfigErrorRecoveryEnableDebugLog = false;
    }
    ~HeadlessUi() {
        if (ImGui::GetCurrentContext() && ImGui::GetCurrentContext()->WithinFrameScope)
            ImGui::EndFrame();
        ImPlot3D::DestroyContext(implot3d);
        ImPlot::DestroyContext(implot);
        ImGui::DestroyContext(imgui);
    }

    void new_frame() { ImGui::NewFrame(); }
    void end_frame() {
        ImGui::Render();
        // The backend's half of the 1.92 texture protocol, minus the GPU.
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
};

// caliper.ui.v1 over the test's contexts. C tables carry no user data, hence
// the static.
HeadlessUi* g_ui = nullptr;
ImGuiContext* ui_imgui() { return g_ui->imgui; }
ImPlotContext* ui_implot() { return g_ui->implot; }
ImPlot3DContext* ui_implot3d() { return g_ui->implot3d; }
void ui_allocators(CaliperImGuiAllocFn* out_alloc, CaliperImGuiFreeFn* out_free,
                   void** out_user) {
    ImGuiMemAllocFunc a = nullptr;
    ImGuiMemFreeFunc f = nullptr;
    ImGui::GetAllocatorFunctions(&a, &f, out_user);
    *out_alloc = reinterpret_cast<CaliperImGuiAllocFn>(a);
    *out_free = reinterpret_cast<CaliperImGuiFreeFn>(f);
}
const CaliperUiV1 kTestUi = {sizeof(CaliperUiV1), &ui_imgui, &ui_implot,
                             &ui_implot3d, &ui_allocators};

// Counts ImGui's recoverable errors (IM_ASSERT_USER_ERROR, which ImPlot and
// ImPlot3D use too) while installed: a stale open window, an unpopped push or
// a "Mismatched BeginPlot()/EndPlot()!" each lands here.
struct ErrorCount {
    int n = 0;
    std::string first;
    ErrorCount() {
        ImGuiContext& g = *ImGui::GetCurrentContext();
        g.ErrorCallback = [](ImGuiContext*, void* user, const char* msg) {
            auto* self = static_cast<ErrorCount*>(user);
            if (self->n++ == 0) self->first = msg ? msg : "";
        };
        g.ErrorCallbackUserData = this;
    }
    ~ErrorCount() {
        ImGuiContext& g = *ImGui::GetCurrentContext();
        g.ErrorCallback = nullptr;
        g.ErrorCallbackUserData = nullptr;
    }
};

// What the host draws after the applet's frame: its own window (the Jobs
// tray), then, on the next frame, a window with a plot of each kind, which
// would trip over a plot the applet left open.
void host_window() {
    ImGui::Begin("Host: Jobs");
    ImGui::TextUnformatted("host");
    ImGui::End();
}
void host_plots() {
    ImGui::SetNextWindowSize(ImVec2(400.0f, 500.0f));
    ImGui::Begin("Host: plots");
    static const double xs[] = {0.0, 1.0};
    if (ImPlot::BeginPlot("host 2-D", ImVec2(300.0f, 200.0f))) {
        ImPlot::PlotLine("l", xs, xs, 2);
        ImPlot::EndPlot();
    }
    static const float ps[] = {0.0f, 1.0f};
    if (ImPlot3D::BeginPlot("host 3-D", ImVec2(300.0f, 200.0f))) {
        ImPlot3D::PlotLine("l", ps, ps, ps, 2);
        ImPlot3D::EndPlot();
    }
    ImGui::End();
}

// The stack sizes the checkpoint has to return to, read from the outside.
struct Stacks {
    int windows, ids, colors, style_vars, fonts, groups, disabled;
    static Stacks now() {
        const ImGuiContext& g = *ImGui::GetCurrentContext();
        return {g.CurrentWindowStack.Size, g.CurrentWindow->IDStack.Size,
                g.ColorStack.Size,         g.StyleVarStack.Size,
                g.FontStack.Size,          g.GroupStack.Size,
                g.DisabledStackSize};
    }
};

int find_throwing(AppletLoader& L) {
    for (int k = 0; k < L.count(); k++)
        if (L.at(k).manifest.id == "dev.caliper.throwing") return k;
    return -1;
}

}  // namespace

TEST_CASE("ui recovery: an applet's half-finished UI is unwound and the next "
          "frame runs") {
    HeadlessUi ui;
    g_ui = &ui;
    caliper::testing::FixtureHost fx;
    fx.provide(CALIPER_UI_V1, &kTestUi);
    CaliperFrameInfo fi{};
    fi.struct_size = sizeof fi;
    const HostCaps caps{2, "0.6.0", {"caliper.ui.v1", "caliper.log.v1"}};
    ImGuiContext& g = *ui.imgui;

    // One frame as main.cpp runs it: checkpoint, the applet's guarded
    // frame() fails, recover, the host draws on. Then a whole next frame.
    auto fail_mid_frame_and_recover = [&](const char* where, FaultKind kind) {
        setenv("CALIPER_THROWING_WHERE", where, 1);
        AppletLoader L(caps, CALIPER_TEST_DATA_ROOT);
        L.scan(CALIPER_TEST_FIXTURES_DIR);
        const int i = find_throwing(L);
        REQUIRE(i >= 0);
        REQUIRE(L.launch(i, *fx.host()));

        ui.new_frame();
        const Stacks before = Stacks::now();
        const UiCheckpoint cp = UiCheckpoint::store();
        CHECK_FALSE(L.frame(i, fi));
        CHECK(L.at(i).status == AppletStatus::Quarantined);
        CHECK(L.at(i).fault_kind == kind);

        // Everything the fixture began is still open: the fault really did
        // land mid-UI, with both plots begun.
        CHECK(g.CurrentWindowStack.Size > 2);   // fallback + window + child
        CHECK(g.ColorStack.Size > 0);
        CHECK(g.StyleVarStack.Size > 0);
        CHECK(g.GroupStack.Size > 0);
        CHECK(g.DisabledStackSize > 0);
        CHECK(g.CurrentTable != nullptr);
        REQUIRE(ui.implot->CurrentPlot != nullptr);
        REQUIRE(ui.implot3d->CurrentPlot != nullptr);
        CHECK(ui.implot->ColorModifiers.Size == 1);
        CHECK(ui.implot3d->ColormapModifiers.Size == 1);

        // Recovery reports through its own channel and puts the host's
        // settings back: the callback the host had installed sees none of the
        // applet's errors, and ImGui's error tooltip never opens even though
        // the host had it on.
        ErrorCount errors;
        g.IO.ConfigErrorRecoveryEnableTooltip = true;
        CHECK(cp.recover() > 0);
        CHECK(errors.n == 0);
        CHECK(g.ErrorCallbackUserData == &errors);
        CHECK(g.IO.ConfigErrorRecoveryEnableTooltip);
        CHECK_FALSE(g.IO.ConfigErrorRecoveryEnableAssert);
        CHECK(ImGui::FindWindowByName("##Tooltip_Error") == nullptr);
        g.IO.ConfigErrorRecoveryEnableTooltip = false;

        // Back to the checkpoint: only ImGui's implicit fallback window, no
        // pushes, no plot, no plot pushes.
        const Stacks after = Stacks::now();
        CHECK(after.windows == before.windows);
        CHECK(g.CurrentWindow->IsFallbackWindow);
        CHECK(after.ids == before.ids);
        CHECK(after.colors == before.colors);
        CHECK(after.style_vars == before.style_vars);
        CHECK(after.fonts == before.fonts);
        CHECK(after.groups == before.groups);
        CHECK(after.disabled == before.disabled);
        CHECK(g.CurrentTable == nullptr);
        CHECK(ui.implot->CurrentPlot == nullptr);
        CHECK(ui.implot->CurrentItems == nullptr);
        CHECK(ui.implot->ColorModifiers.Size == 0);
        CHECK(ui.implot->StyleModifiers.Size == 0);
        CHECK(ui.implot->ColormapModifiers.Size == 0);
        CHECK(ui.implot3d->CurrentPlot == nullptr);
        CHECK(ui.implot3d->ColorModifiers.Size == 0);
        CHECK(ui.implot3d->StyleModifiers.Size == 0);
        CHECK(ui.implot3d->ColormapModifiers.Size == 0);

        // The host draws on: its window is a top-level window, not a child
        // of the applet's, and nothing in the rest of this frame or the next
        // is a recoverable error.
        ImGui::Begin("Host: Jobs");
        CHECK(g.CurrentWindowStack.Size == before.windows + 1);
        CHECK(g.CurrentWindow->ParentWindowInBeginStack->IsFallbackWindow);
        ImGui::End();
        ui.end_frame();

        ui.new_frame();
        host_window();
        host_plots();
        ui.end_frame();
        CHECK_MESSAGE(errors.n == 0, errors.first);
        unsetenv("CALIPER_THROWING_WHERE");
    };

    SUBCASE("after a C++ exception") {
        fail_mid_frame_and_recover("frame-imgui", FaultKind::Exception);
    }
    SUBCASE("after a memory fault") {
        fail_mid_frame_and_recover("frame-imgui-fault", FaultKind::Signal);
    }
    SUBCASE("without the checkpoint the host's window nests in the applet's") {
        // The control: the same fault, no recovery. The host's next window
        // opens inside the applet's half-built one, and ImGui finds the
        // applet's leftovers at the end of the frame.
        setenv("CALIPER_THROWING_WHERE", "frame-imgui", 1);
        AppletLoader L(caps, CALIPER_TEST_DATA_ROOT);
        L.scan(CALIPER_TEST_FIXTURES_DIR);
        const int i = find_throwing(L);
        REQUIRE(i >= 0);
        REQUIRE(L.launch(i, *fx.host()));
        ui.new_frame();
        CHECK_FALSE(L.frame(i, fi));
        ErrorCount errors;
        ImGui::Begin("Host: Jobs");
        CHECK(g.CurrentWindowStack.Size > 2);
        CHECK_FALSE(g.CurrentWindow->ParentWindowInBeginStack->IsFallbackWindow);
        ImGui::End();
        ui.end_frame();
        CHECK(errors.n > 0);
        // EndFrame recovered what it could; the plot contexts it does not
        // know about still hold the applet's plots.
        CHECK(ui.implot->CurrentPlot != nullptr);
        ui.implot->CurrentPlot = nullptr;
        ui.implot->CurrentItems = nullptr;
        ui.implot3d->CurrentPlot = nullptr;
        ui.implot3d->CurrentItems = nullptr;
        unsetenv("CALIPER_THROWING_WHERE");
    }
    SUBCASE("a checkpoint with nothing left open costs nothing to recover") {
        ui.new_frame();
        const UiCheckpoint cp = UiCheckpoint::store();
        host_window();
        CHECK(cp.recover() == 0);
        ui.end_frame();
        // Outside a frame (an applet launched before the first one) the
        // checkpoint is empty and recover() does nothing.
        const UiCheckpoint outside = UiCheckpoint::store();
        CHECK(outside.recover() == 0);
    }
    g_ui = nullptr;
}
