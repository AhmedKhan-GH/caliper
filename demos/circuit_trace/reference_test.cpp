// CPU-only ImGui references. No platform backend, window, or GPU is created.
#include "builder/circuit_reference.h"
#include "cpu_check.h"

int main() {
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO();
    io.IniFilename = nullptr;
    io.LogFilename = nullptr;
    io.DisplaySize = ImVec2(256, 128);
    io.Fonts->AddFontDefault();
    unsigned char* pixels = nullptr;
    int width = 0, height = 0;
    io.Fonts->GetTexDataAsRGBA32(&pixels, &width, &height);
    CIRCUIT_CHECK(pixels && width > 0 && height > 0);
    io.Fonts->SetTexID(ImTextureID(1));
    ImGui::NewFrame();
    ImGui::SetNextWindowPos(ImVec2(0, 0));
    ImGui::SetNextWindowSize(ImVec2(256, 128));
    ImGui::Begin("CPU capture reference", nullptr, ImGuiWindowFlags_NoDecoration);
    auto* draw = ImGui::GetWindowDrawList();
    draw->PushClipRect(ImVec2(12, 14), ImVec2(180, 100));
    draw->AddImage(ImTextureID(99), ImVec2(10, 12), ImVec2(190, 110));
    const auto image = caliper_host::CircuitReference::image(ImTextureID(99));
    CIRCUIT_CHECK(image["rect"] == nlohmann::json({10, 12, 190, 110}));
    CIRCUIT_CHECK(image["clip"] == nlohmann::json({12, 14, 180, 100}));
    bool refused = false;
    try { caliper_host::CircuitReference::image(ImTextureID(100)); }
    catch (const std::runtime_error&) { refused = true; }
    CIRCUIT_CHECK(refused); // An absent actual image command must fail.
    caliper_live_views::DrawnLabel label{"A", 30, 30, 50, 50, ImGui::GetFont(), ImGui::GetFontSize(),
        ImGui::GetFont()->CurrentRasterizerDensity, IM_COL32_WHITE, IM_COL32_BLACK,
        ImVec4(12, 14, 180, 100)};
    const auto expected = caliper_host::CircuitReference::text(label);
    CIRCUIT_CHECK(expected["quads"].size() == 2); // Shadow and ink use the real default font atlas.
    const int actual_vertices = draw->VtxBuffer.Size;
    label.text = "B";
    const auto wrong = caliper_host::CircuitReference::text(label);
    CIRCUIT_CHECK(wrong["quads"][1]["uv"] != expected["quads"][1]["uv"]);
    label.text = "A";
    label.clip.z = 31;
    const auto clipped = caliper_host::CircuitReference::text(label);
    CIRCUIT_CHECK(clipped["quads"].size() == 2);
    CIRCUIT_CHECK(clipped["quads"][1]["rect"] == expected["quads"][1]["rect"]);
    CIRCUIT_CHECK(draw->VtxBuffer.Size == actual_vertices); // References never enter the window.
    draw->PopClipRect();
    ImGui::End();
    ImGui::EndFrame();
    ImGui::DestroyContext();
}
