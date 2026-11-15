// Proof that the installed caliper-sdk package is self-contained: the sugar,
// the UI stack and the testing fixture all come from the install prefix.
#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include <caliper/abi.h>
#include <caliper/caliper.hpp>
#include <caliper/fixture_host.h>
#include <caliper/services/log_v1.h>

#include <ImGuiFileDialog.h>
#include <imgui.h>
#include <implot.h>
#include <implot3d.h>

static_assert(CALIPER_ABI_EPOCH == PROBE_ABI_EPOCH,
              "the headers speak the epoch the package records");

TEST_CASE("the installed UI stack draws a headless frame") {
    ImGuiContext* imgui = ImGui::CreateContext();
    ImPlotContext* implot = ImPlot::CreateContext();
    ImPlot3DContext* implot3d = ImPlot3D::CreateContext();
    ImGuiIO& io = ImGui::GetIO();
    io.BackendFlags |= ImGuiBackendFlags_RendererHasTextures;
    io.DisplaySize = ImVec2(640.0f, 480.0f);

    ImGui::NewFrame();
    ImGui::Begin("Probe");
    if (ImPlot::BeginPlot("plot")) {
        const float ys[3] = {1.0f, 2.0f, 3.0f};
        ImPlot::PlotLine("line", ys, 3);
        ImPlot::EndPlot();
    }
    if (ImPlot3D::BeginPlot("plot3d")) {
        ImPlot3D::EndPlot();
    }
    ImGui::End();
    ImGui::Render();
    CHECK(ImGui::GetDrawData() != nullptr);

    ImGuiFileDialog dialog;
    CHECK_FALSE(dialog.IsOpened());

    ImPlot3D::DestroyContext(implot3d);
    ImPlot::DestroyContext(implot);
    ImGui::DestroyContext(imgui);
}

TEST_CASE("the fixture host vends log.v1") {
    caliper::testing::FixtureHost fixture;
    const CaliperHost* host = fixture.host();
    REQUIRE(host != nullptr);
    CHECK(host->abi_epoch == CALIPER_ABI_EPOCH);
    const auto* log = static_cast<const CaliperLogV1*>(host->get_service(host, CALIPER_LOG_V1));
    REQUIRE(log != nullptr);
    log->log(CALIPER_LOG_INFO, "probe");
    CHECK(fixture.log_contains("probe"));
}
