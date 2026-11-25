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
    ImFontConfig alternate_config;
    alternate_config.SizePixels = 22;
    ImFont* alternate_font = io.Fonts->AddFontDefault(&alternate_config);
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
    label.original_text = "A";
    const auto expected = caliper_host::CircuitReference::text(label);
    CIRCUIT_CHECK(expected["original_text"] == "A");
    CIRCUIT_CHECK(expected["fit"].is_null());
    CIRCUIT_CHECK(expected["quads"].size() == 2); // Shadow and ink use the real default font atlas.
    const int actual_vertices = draw->VtxBuffer.Size;
    label.text = "B";
    label.original_text = "B";
    const auto wrong = caliper_host::CircuitReference::text(label);
    CIRCUIT_CHECK(wrong["quads"][1]["uv"] != expected["quads"][1]["uv"]);
    label.text = "A";
    label.original_text = "A";
    label.clip.z = 31;
    const auto clipped = caliper_host::CircuitReference::text(label);
    CIRCUIT_CHECK(clipped["quads"].size() == 2);
    CIRCUIT_CHECK(clipped["quads"][1]["rect"] == expected["quads"][1]["rect"]);
    CIRCUIT_CHECK(draw->VtxBuffer.Size == actual_vertices); // References never enter the window.

    const auto width_of = [](std::string_view text) {
        return ImGui::CalcTextSize(text.data(), text.data() + text.size()).x;
    };
    auto fitted = label;
    fitted.clip = ImVec4(12, 14, 180, 100);
    fitted.original_text = "[empty token 256]";
    fitted.fit_width = width_of("[emp…");
    fitted.text = "[emp…";
    const auto proof = caliper_host::CircuitReference::text(fitted);
    CIRCUIT_CHECK(proof["original_text"] == "[empty token 256]");
    CIRCUIT_CHECK(proof["text"] == "[emp…");
    CIRCUIT_CHECK(proof["fit"]["width"] == *fitted.fit_width);
    CIRCUIT_CHECK(proof["fit"]["full_width"] == width_of(fitted.original_text));
    CIRCUIT_CHECK(proof["fit"]["prefix_widths"][4] == width_of("[emp…"));
    CIRCUIT_CHECK(proof["fit"]["prefix_widths"][5].get<float>() > *fitted.fit_width);
    CIRCUIT_CHECK(proof["quads"].size() == 10);

    // A later active font or density must not replace the captured font state.
    ImGui::PushFont(alternate_font, 22);
    fitted.font->CurrentRasterizerDensity = 2;
    CIRCUIT_CHECK(caliper_host::CircuitReference::text(fitted) == proof);
    CIRCUIT_CHECK(fitted.font->CurrentRasterizerDensity == 2);
    fitted.font->CurrentRasterizerDensity = fitted.font_density;
    ImGui::PopFont();

    const auto refuses = [](const caliper_live_views::DrawnLabel& value) {
        try { caliper_host::CircuitReference::text(value); }
        catch (const std::runtime_error&) { return true; }
        return false;
    };
    auto changed = fitted;
    changed.text = "[e…";
    CIRCUIT_CHECK(refuses(changed)); // A shorter matching prefix does not prove the exact fit.
    changed.text = "[xyz…";
    CIRCUIT_CHECK(refuses(changed));
    changed.text.clear();
    CIRCUIT_CHECK(refuses(changed));
    changed = fitted;
    changed.original_text = "[other token 256]";
    CIRCUIT_CHECK(refuses(changed));
    changed = fitted;
    changed.fit_width = 0;
    CIRCUIT_CHECK(refuses(changed));
    changed.fit_width = -1;
    CIRCUIT_CHECK(refuses(changed));
    changed.fit_width = NAN;
    CIRCUIT_CHECK(refuses(changed));
    changed = fitted;
    changed.fit_width.reset();
    CIRCUIT_CHECK(refuses(changed)); // Unpinned labels must retain the complete source.

    fitted.original_text = "é你🙂 suffix";
    fitted.fit_width = width_of("é你…");
    fitted.text = "é你…";
    const auto unicode = caliper_host::CircuitReference::text(fitted);
    CIRCUIT_CHECK(unicode["fit"]["prefix_widths"].size() == 9);
    CIRCUIT_CHECK(unicode["fit"]["prefix_widths"][2] == width_of("é你…"));
    CIRCUIT_CHECK(unicode["original_text"] == "é你🙂 suffix");
    CIRCUIT_CHECK(unicode["quads"].size() == 6);
    CIRCUIT_CHECK(draw->VtxBuffer.Size == actual_vertices);

    draw->PopClipRect();
    ImGui::End();
    ImGui::EndFrame();
    ImGui::DestroyContext();
}
