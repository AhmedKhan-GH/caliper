// page_style.cpp: see page_style.h. The colours are the palette of Caliper's
// site and of the element's TypeScript views (web/src/views/theme.ts).
#include "page_style.h"

#include <imgui.h>
#include <implot.h>

namespace caliper_web {

namespace {
constexpr ImU32 kPlate     = IM_COL32(0x27, 0x26, 0x22, 255);
constexpr ImU32 kRuleFaint = IM_COL32(0x36, 0x34, 0x2f, 255);
constexpr ImU32 kRule      = IM_COL32(0x5a, 0x57, 0x51, 255);
constexpr ImU32 kParchment = IM_COL32(0xf0, 0xe9, 0xdb, 255);
constexpr ImU32 kSecondary = IM_COL32(0xd0, 0xc9, 0xba, 255);
constexpr ImU32 kFaint     = IM_COL32(0xa4, 0x9d, 0x8f, 255);
constexpr ImU32 kBronze    = IM_COL32(0xb3, 0x8b, 0x59, 255);

ImVec4 v4(ImU32 c) { return ImGui::ColorConvertU32ToFloat4(c); }
}  // namespace

void apply_page_style(bool bare) {
    ImGuiStyle& st = ImGui::GetStyle();
    st.Colors[ImGuiCol_WindowBg] = v4(kPlate);
    st.Colors[ImGuiCol_ChildBg] = v4(kPlate);
    if (bare) {
        st.WindowBorderSize = 0.0f;
        st.WindowPadding = ImVec2(6.0f, 6.0f);
    }

    ImPlotStyle& ps = ImPlot::GetStyle();
    ps.Colors[ImPlotCol_FrameBg] = v4(kPlate);
    ps.Colors[ImPlotCol_PlotBg] = v4(kPlate);
    ps.Colors[ImPlotCol_PlotBorder] = v4(kRule);
    ps.Colors[ImPlotCol_AxisText] = v4(kFaint);
    ps.Colors[ImPlotCol_AxisGrid] = v4(kRuleFaint);
    ps.Colors[ImPlotCol_AxisTick] = v4(kRule);
    ps.Colors[ImPlotCol_InlayText] = v4(kSecondary);
    ps.Colors[ImPlotCol_TitleText] = v4(kParchment);
    ps.Colors[ImPlotCol_Crosshairs] = v4(kRule);
    ps.Colors[ImPlotCol_Selection] = v4(kBronze);
    // A fitted y range reaches 6% of the data's span above and below it, as
    // the TypeScript series does (ImPlot adds half the padding on each side),
    // so that a loss near zero is not drawn on the plot's bottom border.
    ps.FitPadding = ImVec2(0.0f, 0.12f);
    // Items take the colormap's colours in order: a series line or the
    // histogram's bars first (parchment), then the hover mark (bronze).
    static const ImU32 kItems[] = {kParchment, kBronze, kSecondary, kFaint};
    static ImPlotColormap map = -1;
    if (map < 0) map = ImPlot::AddColormap("caliper-page", kItems, 4, true);
    ps.Colormap = map;
}

}  // namespace caliper_web
