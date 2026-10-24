// ============================================================================
// Sternbild house style for the Caliper chrome: the ImGui style table, the
// draw-list primitives (rules, cut plate, star, diamond, mark, tracked text)
// and the dense workstation widgets declared in sternbild_theme.h.
//
// UX patterns this file leans on, named where they apply:
//   - pre-attentive colour: hue only where it means something. Bronze marks
//     focus, selection and keys; parchment is text; secondary is metadata.
//     No other hue reaches the chrome, so the eye finds the accent at once.
//   - Fitts's law: the type is small, but hit targets are the full row or
//     plate height, never just the glyphs.
//   - honest disabled actions: a disabled control states why it is disabled.
//   - recognition over recall: a button holds the key that runs the same
//     action ("⏎ Launch"), so no label beside it repeats the action.
//   - command bar and status bar strips (Bloomberg command line, VS Code).
//   - booktabs tables with a sticky header (spreadsheets, W&B run tables).
// ============================================================================

#include "ui/sternbild_theme.h"
#include "ui/sternbild_fonts.h"

#include <imgui_internal.h>   // ItemSize/ItemAdd/ButtonBehavior, ImGuiTable

#include <cfloat>
#include <cstring>

namespace sternbild {
namespace {

// Paddings derived from the type sizes so a row or a frame holds its text
// with whole-pixel padding: 15 px type + 2 × 3 = 21, inside the 22 px row
// and the 22 px pane header (tab bar).
constexpr float half_floor(float v) {
    return static_cast<float>(static_cast<int>(v * 0.5f));
}
constexpr ImVec2 kCellPadding(6.0f, half_floor(metric::row - font_size::mono));
constexpr ImVec2 kFramePadding(6.0f, half_floor(metric::pane_header - font_size::ui));

constexpr float kPlateHeight   = 24.0f;  // primary_button default height
constexpr float kPlatePadX     = 12.0f;  // primary_button label inset
constexpr float kFocusOffset   = 3.0f;   // keyboard focus outline, outside the plate
constexpr float kKeyPadX       = 8.0f;   // key_button content inset
constexpr float kKeyGap        = 6.0f;   // a button's key to its action
constexpr float kChevronGap    = 4.0f;   // header label to its sort chevron
constexpr float kReasonGap     = 12.0f;  // disabled action to its reason
constexpr float kStripPadX     = 12.0f;  // command bar / status bar side inset
constexpr float kPaneInsetX    = 12.0f;  // pane header title / shortcut inset
constexpr float kMarkSize      = 18.0f;  // the Caliper mark in the masthead
constexpr float kMarkGap       = 8.0f;   // mark to wordmark
constexpr float kWordmarkTrack = 0.10f;  // em; Display's em equals its size
constexpr float kChevronW      = 8.0f;   // sort chevron: two 45° strokes,
constexpr float kChevronH      = 4.0f;   //   8 wide × 4 tall (SPEC §6.7)

// The strips are fixed furniture: no title, no moving, no saved position,
// never docked into and never raised over the panes when clicked.
constexpr ImGuiWindowFlags kStripFlags =
    ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_NoMove |
    ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoSavedSettings |
    ImGuiWindowFlags_NoDocking | ImGuiWindowFlags_NoBringToFrontOnFocus |
    ImGuiWindowFlags_NoScrollbar | ImGuiWindowFlags_NoScrollWithMouse;

// status_item() separates items on one line. A line is the run of items in
// one window in one frame; the strip begins also reset it explicitly.
struct StatusLine {
    int frame = -1;
    ImGuiID window = 0;
};
StatusLine g_status_line;

// Colour through the style alpha, as ImGui's own widgets do, so BeginDisabled
// (DisabledAlpha) and window fades apply to the house widgets too.
ImU32 styled(ImU32 c) { return ImGui::GetColorU32(c); }

void set(ImGuiStyle& st, ImGuiCol idx, ImU32 c) { st.Colors[idx] = to_vec4(c); }

// Uppercase ASCII into `out`, leaving UTF-8 sequences alone. A long label is
// cut at a code point boundary, never inside a sequence.
void upper_ascii(const char* in, char* out, size_t cap) {
    size_t i = 0;
    for (; in[i] != '\0' && i + 1 < cap; ++i) {
        const char ch = in[i];
        out[i] = (ch >= 'a' && ch <= 'z') ? static_cast<char>(ch - 'a' + 'A') : ch;
    }
    if ((static_cast<unsigned char>(in[i]) & 0xC0) == 0x80) {
        while (i > 0 && (static_cast<unsigned char>(out[i - 1]) & 0xC0) == 0x80)
            --i;
        if (i > 0)
            --i;   // the lead byte of the cut sequence
    }
    out[i] = '\0';
}

// Baseline of a face at a size, in px below the line top. ImGui places every
// glyph on round(baked Ascent), so this is the baseline it will actually use.
float baseline_of(ImFont* face, float size_base) {
    ImGui::PushFont(face, size_base);
    const ImFontBaked* baked = ImGui::GetFontBaked();
    const float size = ImGui::GetFontSize();
    const float a =
        baked ? IM_ROUND(baked->Ascent * (size / baked->Size)) : size * 0.8f;
    ImGui::PopFont();
    return a;
}

// Ink extent of the capitals ('H') below the line top, for optical centring
// of uppercase text (the wordmark) on a mark or a strip.
void caps_ink(ImFont* font, float size, float* top, float* bottom) {
    *top = 0.0f;
    *bottom = size;
    ImFontBaked* baked = font->GetFontBaked(size);
    if (!baked)
        return;
    if (const ImFontGlyph* g = baked->FindGlyphNoFallback('H')) {
        const float s = size / baked->Size;
        *top = g->Y0 * s;
        *bottom = g->Y1 * s;
    }
}

// One run of text laid out like ImGui::TextUnformatted: it honours the line's
// text baseline offset (so it sits level with framed widgets) and reports its
// extent to the layout, but is drawn through text_tracked. `nudge` moves the
// run down so it shares a baseline with a different face on the same line.
void text_run(ImFont* face, float size_base, const char* text, ImU32 c,
              float tracking_em, float nudge = 0.0f) {
    ImGuiWindow* window = ImGui::GetCurrentWindow();
    if (window->SkipItems)
        return;
    ImGui::PushFont(face, size_base);
    ImFont* font = ImGui::GetFont();
    const float size = ImGui::GetFontSize();
    const float tracking = tracking_em * size;
    const float width = text_tracked_width(font, size, text, tracking);
    const float y = window->DC.CursorPos.y + window->DC.CurrLineTextBaseOffset;
    const ImVec2 pos(window->DC.CursorPos.x, y + nudge);
    ImGui::ItemSize(ImVec2(width, size + nudge), 0.0f);
    if (ImGui::ItemAdd(ImRect(pos, ImVec2(pos.x + width, pos.y + size)), 0))
        text_tracked(window->DrawList, font, size, pos, styled(c), text, tracking);
    ImGui::PopFont();
}

// The sort direction mark in a header cell: two 1 px strokes at 45° on half
// pixels, pointing up for ascending and down for descending.
void sort_chevron(ImDrawList* dl, float x, float centre_y, bool ascending, ImU32 c) {
    const float x0 = ImFloor(x) + 0.5f;
    const float y0 = ImFloor(centre_y - kChevronH * 0.5f) + 0.5f;
    const float y_tip = ascending ? y0 : y0 + kChevronH;
    const float y_arm = ascending ? y0 + kChevronH : y0;
    const ImVec2 pts[3] = {ImVec2(x0, y_arm), ImVec2(x0 + kChevronW * 0.5f, y_tip),
                           ImVec2(x0 + kChevronW, y_arm)};
    dl->AddPolyline(pts, 3, c, 1.0f);
}

// Shared by both strips: a viewport side bar across the main viewport (the
// pattern ImGui's own main menu bar uses), with its padding chosen so a
// frame-height line is centred in the strip. A side bar takes its place from
// the work area left by earlier bars this frame (so the status bar still
// lands at the bottom under a main menu bar) and reserves its height from
// WorkPos / WorkSize, so a dock space laid out from them never runs under it.
void strip_begin(const char* name, ImGuiDir dir, float height, float frame_height,
                 ImU32 bg) {
    ImGui::PushStyleColor(ImGuiCol_WindowBg, bg);
    ImGui::PushStyleVar(ImGuiStyleVar_WindowBorderSize, 0.0f);
    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding,
                        ImVec2(kStripPadX, half_floor(height - frame_height)));
    // The strips are shorter than the default 32 px minimum window size.
    ImGui::PushStyleVar(ImGuiStyleVar_WindowMinSize, ImVec2(0.0f, 0.0f));
    ImGui::BeginViewportSideBar(name, ImGui::GetMainViewport(), dir, height,
                                kStripFlags);
    ImGui::PopStyleVar(3);
    ImGui::PopStyleColor();
    g_status_line = StatusLine{};
}

// A rule along one edge of the current window. The window clip rect is inset
// by half the side padding, so widen it to the full window for the rule.
void strip_rule(bool bottom) {
    ImDrawList* dl = ImGui::GetWindowDrawList();
    const ImVec2 p = ImGui::GetWindowPos();
    const ImVec2 s = ImGui::GetWindowSize();
    dl->PushClipRect(p, ImVec2(p.x + s.x, p.y + s.y), false);
    rule_h(dl, p.x, p.x + s.x, bottom ? p.y + s.y - 1.0f : p.y);
    dl->PopClipRect();
}

// A key and its action on one line (key_button, primary_button): the key in
// mono at the small size, then the action in the UI face at `action_size`.
// `height` is the action's line, the taller of the two.
struct KeyRun {
    float key_w = 0.0f;
    float action_w = 0.0f;
    float width = 0.0f;
    float height = 0.0f;
};

KeyRun measure_key_run(const char* key, const char* action, const char* action_end,
                       float action_size) {
    KeyRun run;
    if (key && key[0] != '\0') {
        ImGui::PushFont(fonts().mono, font_size::ui_small);
        run.key_w = ImGui::CalcTextSize(key).x;
        ImGui::PopFont();
    }
    ImGui::PushFont(fonts().ui, action_size);
    run.height = ImGui::GetFontSize();
    if (action && action != action_end && action[0] != '\0')
        run.action_w = ImGui::CalcTextSize(action, action_end, false).x;
    ImGui::PopFont();
    const float gap = run.key_w > 0.0f && run.action_w > 0.0f ? kKeyGap : 0.0f;
    run.width = run.key_w + gap + run.action_w;
    return run;
}

// Draws a measured run with `pos` at the top left of the action's line. The
// key is set on the action's baseline (mono and UI have different ascents),
// so keys stay level across a row of buttons with or without an action.
void draw_key_run(ImDrawList* dl, ImVec2 pos, const KeyRun& run, const char* key,
                  const char* action, const char* action_end, float action_size,
                  ImU32 key_ink, ImU32 action_ink) {
    float x = pos.x;
    if (run.key_w > 0.0f) {
        const float nudge = baseline_of(fonts().ui, action_size) -
                            baseline_of(fonts().mono, font_size::ui_small);
        ImGui::PushFont(fonts().mono, font_size::ui_small);
        dl->AddText(ImGui::GetFont(), ImGui::GetFontSize(), ImVec2(x, pos.y + nudge),
                    key_ink, key);
        ImGui::PopFont();
        x += run.key_w + (run.action_w > 0.0f ? kKeyGap : 0.0f);
    }
    if (run.action_w > 0.0f) {
        ImGui::PushFont(fonts().ui, action_size);
        dl->AddText(ImGui::GetFont(), ImGui::GetFontSize(), ImVec2(x, pos.y),
                    action_ink, action, action_end);
        ImGui::PopFont();
    }
}

// A square 1 px outline just inside `bb`, drawn as four rules.
void box(ImDrawList* dl, const ImRect& bb, ImU32 c) {
    rule_h(dl, bb.Min.x, bb.Max.x, bb.Min.y, c);
    rule_h(dl, bb.Min.x, bb.Max.x, bb.Max.y - 1.0f, c);
    rule_v(dl, bb.Min.x, bb.Min.y + 1.0f, bb.Max.y - 1.0f, c);
    rule_v(dl, bb.Max.x - 1.0f, bb.Min.y + 1.0f, bb.Max.y - 1.0f, c);
}

// Honest disabled action: why the last item cannot run, in a tooltip on
// hover (ImGui's tooltip hover allows disabled items).
void reason_tooltip(const char* reason) {
    if (!reason || reason[0] == '\0' || !ImGui::BeginItemTooltip())
        return;
    ImGui::PushFont(fonts().ui, font_size::ui);
    ImGui::PushStyleColor(ImGuiCol_Text, col::secondary);
    ImGui::TextUnformatted(reason);
    ImGui::PopStyleColor();
    ImGui::PopFont();
    ImGui::EndTooltip();
}

}  // namespace

// ---- Colour -------------------------------------------------------------------

ImVec4 to_vec4(ImU32 c) { return ImGui::ColorConvertU32ToFloat4(c); }

ImU32 with_alpha(ImU32 c, int alpha) {
    const ImU32 a = static_cast<ImU32>(ImClamp(alpha, 0, 255));
    return (c & ~IM_COL32_A_MASK) | (a << IM_COL32_A_SHIFT);
}

// ---- Style --------------------------------------------------------------------

void apply_style() {
    ImGuiStyle& st = ImGui::GetStyle();

    // The UI face at its house size starts every frame. FontSizeBase is the
    // unscaled size NewFrame() begins with; the _NextFrame copy keeps it when
    // this runs mid-frame. The built-in fallback font keeps its own size.
    if (fonts().ui) {
        st.FontSizeBase = font_size::ui;
        st._NextFrameFontSizeBase = font_size::ui;
    }

    // Square corners everywhere (SPEC §3.5); the only cut corners are drawn
    // by cut_plate() and caliper_mark().
    st.WindowRounding = 0.0f;
    st.ChildRounding = 0.0f;
    st.PopupRounding = 0.0f;
    st.FrameRounding = 0.0f;
    st.ScrollbarRounding = 0.0f;
    st.GrabRounding = 0.0f;
    st.ImageRounding = 0.0f;
    st.TabRounding = 0.0f;
    st.TreeLinesRounding = 0.0f;
    st.MenuItemRounding = 0.0f;
    st.SelectableRounding = 0.0f;
    st.DragDropTargetRounding = 0.0f;

    // 1 px rules. Panes are separated by the docking separators, so child
    // windows carry no border of their own (no box inside a box).
    st.WindowBorderSize = 1.0f;
    st.ChildBorderSize = 0.0f;
    st.PopupBorderSize = 1.0f;
    st.FrameBorderSize = 0.0f;
    st.TabBorderSize = 0.0f;
    st.TabBarBorderSize = 1.0f;
    st.TabBarOverlineSize = 1.0f;
    st.DockingSeparatorSize = 1.0f;
    st.SeparatorTextBorderSize = 1.0f;
    st.DragDropTargetBorderSize = 1.0f;
    st.TreeLinesSize = 1.0f;

    // Workstation density: default widgets land on metric::row and the
    // docked tab bar on metric::pane_header.
    st.WindowPadding = ImVec2(metric::pad, 6.0f);
    st.FramePadding = kFramePadding;
    st.CellPadding = kCellPadding;
    st.ItemSpacing = ImVec2(8.0f, 4.0f);
    st.ItemInnerSpacing = ImVec2(6.0f, 4.0f);
    st.ScrollbarSize = 8.0f;
    st.GrabMinSize = 8.0f;
    st.WindowMenuButtonPosition = ImGuiDir_None;

    // Disabled text reads as `faint`: parchment over ground at this alpha
    // lands on #8f897c to within a step per channel.
    st.DisabledAlpha = 0.56f;

    // Pre-attentive colour: the chrome is parchment on the warm night ground
    // in three depths; bronze appears only for focus, selection and keys.
    constexpr ImU32 clear = IM_COL32_BLACK_TRANS;
    set(st, ImGuiCol_Text, col::parchment);
    set(st, ImGuiCol_TextDisabled, col::faint);
    set(st, ImGuiCol_WindowBg, col::ground);
    set(st, ImGuiCol_ChildBg, col::ground);
    set(st, ImGuiCol_PopupBg, col::ground);
    set(st, ImGuiCol_Border, col::rule);
    set(st, ImGuiCol_BorderShadow, clear);
    set(st, ImGuiCol_FrameBg, col::plate);
    set(st, ImGuiCol_FrameBgHovered, col::plate_hover);
    set(st, ImGuiCol_FrameBgActive, col::plate_hover);
    set(st, ImGuiCol_TitleBg, col::ground);
    set(st, ImGuiCol_TitleBgActive, col::ground);
    set(st, ImGuiCol_TitleBgCollapsed, col::ground);
    set(st, ImGuiCol_MenuBarBg, col::ground);
    set(st, ImGuiCol_ScrollbarBg, col::ground);
    set(st, ImGuiCol_ScrollbarGrab, col::rule);
    set(st, ImGuiCol_ScrollbarGrabHovered, col::faint);
    set(st, ImGuiCol_ScrollbarGrabActive, col::faint);
    set(st, ImGuiCol_CheckMark, col::bronze);
    set(st, ImGuiCol_CheckboxSelectedBg, col::plate);
    // Bronze is a mark, never a fill (SPEC §9): the check and the nav cursor
    // stay bronze, but grabs are filled shapes, so they take the neutrals.
    set(st, ImGuiCol_SliderGrab, col::faint);
    set(st, ImGuiCol_SliderGrabActive, col::secondary);
    // ImGui's stock button is not a house button (primary_button, key_button);
    // keep it neutral so a stray ImGui::Button never reads as an accent.
    set(st, ImGuiCol_Button, col::ground);
    set(st, ImGuiCol_ButtonHovered, col::plate_hover);
    set(st, ImGuiCol_ButtonActive, col::plate_hover);
    set(st, ImGuiCol_Header, col::plate);
    set(st, ImGuiCol_HeaderHovered, col::plate_hover);
    set(st, ImGuiCol_HeaderActive, col::plate_hover);
    // A separator under the pointer or being dragged is a focus cue.
    set(st, ImGuiCol_Separator, col::rule);
    set(st, ImGuiCol_SeparatorHovered, col::bronze);
    set(st, ImGuiCol_SeparatorActive, col::bronze);
    set(st, ImGuiCol_ResizeGrip, with_alpha(col::rule, 26));
    set(st, ImGuiCol_ResizeGripHovered, with_alpha(col::rule, 90));
    set(st, ImGuiCol_ResizeGripActive, with_alpha(col::rule, 150));
    set(st, ImGuiCol_InputTextCursor, col::parchment);
    // Docked pane headers: every tab sits on the ground; the focused pane's
    // current tab carries the bronze overline, an unfocused one a rule.
    set(st, ImGuiCol_TabHovered, col::plate_hover);
    set(st, ImGuiCol_Tab, col::ground);
    set(st, ImGuiCol_TabSelected, col::ground);
    set(st, ImGuiCol_TabSelectedOverline, col::bronze);
    set(st, ImGuiCol_TabDimmed, col::ground);
    set(st, ImGuiCol_TabDimmedSelected, col::ground);
    set(st, ImGuiCol_TabDimmedSelectedOverline, col::rule);
    set(st, ImGuiCol_DockingPreview, with_alpha(col::bronze, 77));
    set(st, ImGuiCol_DockingEmptyBg, col::ground);
    set(st, ImGuiCol_PlotLines, col::secondary);
    set(st, ImGuiCol_PlotLinesHovered, col::parchment);
    set(st, ImGuiCol_PlotHistogram, col::secondary);
    set(st, ImGuiCol_PlotHistogramHovered, col::parchment);
    set(st, ImGuiCol_TableHeaderBg, col::ground);
    set(st, ImGuiCol_TableBorderStrong, col::rule);
    set(st, ImGuiCol_TableBorderLight, col::rule);
    set(st, ImGuiCol_TableRowBg, clear);
    set(st, ImGuiCol_TableRowBgAlt, clear);
    set(st, ImGuiCol_TextLink, col::parchment);
    set(st, ImGuiCol_TextSelectedBg, with_alpha(col::bronze, 64));
    set(st, ImGuiCol_TreeLines, col::rule);
    set(st, ImGuiCol_DragDropTarget, col::bronze);
    set(st, ImGuiCol_DragDropTargetBg, clear);
    set(st, ImGuiCol_UnsavedMarker, col::parchment);
    set(st, ImGuiCol_NavCursor, col::bronze);
    set(st, ImGuiCol_NavWindowingHighlight, with_alpha(col::bronze, 178));
    set(st, ImGuiCol_NavWindowingDimBg, with_alpha(col::deep, 153));
    set(st, ImGuiCol_ModalWindowDimBg, with_alpha(col::deep, 153));
}

// ---- Draw-list primitives -----------------------------------------------------

// A 1 px tall rectangle covers exactly one pixel row with no AA fringe at any
// integer framebuffer scale. (Since 1.92.8 AddLine/AddLineH add the 0.5
// offset themselves, so adding it here as well would straddle two rows.)
void rule_h(ImDrawList* dl, float x0, float x1, float y, ImU32 c) {
    x0 = ImFloor(x0);
    x1 = ImFloor(x1);
    y = ImFloor(y);
    if (x1 < x0)
        ImSwap(x0, x1);
    dl->AddRectFilled(ImVec2(x0, y), ImVec2(x1, y + 1.0f), c);
}

void rule_v(ImDrawList* dl, float x, float y0, float y1, ImU32 c) {
    x = ImFloor(x);
    y0 = ImFloor(y0);
    y1 = ImFloor(y1);
    if (y1 < y0)
        ImSwap(y0, y1);
    dl->AddRectFilled(ImVec2(x, y0), ImVec2(x + 1.0f, y1), c);
}

void cut_plate(ImDrawList* dl, ImVec2 p_min, ImVec2 p_max, ImU32 outline,
               ImU32 fill, float cut) {
    // SPEC §6.7 on half pixels: 6.5,0.5  W-6.5,0.5  W-0.5,6.5  W-0.5,H-6.5
    // W-6.5,H-0.5  6.5,H-0.5  0.5,H-6.5  0.5,6.5 (clockwise, as AA fill needs).
    const float x0 = ImFloor(p_min.x) + 0.5f;
    const float y0 = ImFloor(p_min.y) + 0.5f;
    const float x1 = ImFloor(p_max.x) - 0.5f;
    const float y1 = ImFloor(p_max.y) - 0.5f;
    const float c = ImMin(ImFloor(cut), ImFloor(ImMin(x1 - x0, y1 - y0) * 0.5f));
    const ImVec2 pts[8] = {
        ImVec2(x0 + c, y0), ImVec2(x1 - c, y0), ImVec2(x1, y0 + c), ImVec2(x1, y1 - c),
        ImVec2(x1 - c, y1), ImVec2(x0 + c, y1), ImVec2(x0, y1 - c), ImVec2(x0, y0 + c),
    };
    if (fill != 0)
        dl->AddConvexPolyFilled(pts, 8, fill);
    dl->AddPolyline(pts, 8, outline, 1.0f, ImDrawFlags_Closed);
}

void open_star(ImDrawList* dl, ImVec2 centre, float tip_radius, ImU32 fill, ImU32 halo) {
    // SPEC §6.13 at R = 12, scaled: tip at R, base on the inner vertices
    // (±3.31 across, 4.27 from the centre). Each arm is its own triangle so
    // the middle stays open and the selected point's colour shows through.
    const float k = tip_radius / 12.0f;
    const float a = 3.31f * k;
    const float b = 4.27f * k;
    const float r = tip_radius;
    const ImVec2 o = centre;
    const ImVec2 arms[4][3] = {
        {ImVec2(o.x, o.y - r), ImVec2(o.x + a, o.y - b), ImVec2(o.x - a, o.y - b)},
        {ImVec2(o.x + r, o.y), ImVec2(o.x + b, o.y + a), ImVec2(o.x + b, o.y - a)},
        {ImVec2(o.x, o.y + r), ImVec2(o.x - a, o.y + b), ImVec2(o.x + a, o.y + b)},
        {ImVec2(o.x - r, o.y), ImVec2(o.x - b, o.y - a), ImVec2(o.x - b, o.y + a)},
    };
    // paint-order: stroke. The 2 px halo goes under the fill, so 1 px of it
    // shows outside each arm against the plot field.
    for (const auto& t : arms) {
        dl->AddTriangle(t[0], t[1], t[2], halo, 2.0f);
        dl->AddTriangleFilled(t[0], t[1], t[2], fill);
    }
}

void diamond(ImDrawList* dl, ImVec2 centre, float half, ImU32 c, bool filled) {
    const ImVec2 top(centre.x, centre.y - half);
    const ImVec2 right(centre.x + half, centre.y);
    const ImVec2 bottom(centre.x, centre.y + half);
    const ImVec2 left(centre.x - half, centre.y);
    if (filled)
        dl->AddQuadFilled(top, right, bottom, left, c);
    else
        dl->AddQuad(top, right, bottom, left, c, 1.0f);
}

void caliper_mark(ImDrawList* dl, ImVec2 p_min, float size, ImU32 c) {
    // Reduction of caliper-clean-parchment.svg (viewBox 232 232 536 536) for
    // 14–24 px, every stroke 1 px. Coordinates below are fractions of the box.
    const ImVec2 o = ImFloor(p_min);
    const float s = ImFloor(size + 0.5f);

    // Frame: the octagon 344,240 … 240,344, whose 45° cuts are 20% of the
    // side, set on half pixels just inside the box.
    const float x0 = o.x + 0.5f;
    const float y0 = o.y + 0.5f;
    const float x1 = o.x + s - 0.5f;
    const float y1 = o.y + s - 0.5f;
    const float cut = IM_ROUND(0.2f * (s - 1.0f));
    const ImVec2 frame[8] = {
        ImVec2(x0 + cut, y0), ImVec2(x1 - cut, y0),
        ImVec2(x1, y0 + cut), ImVec2(x1, y1 - cut),
        ImVec2(x1 - cut, y1), ImVec2(x0 + cut, y1),
        ImVec2(x0, y1 - cut), ImVec2(x0, y0 + cut),
    };
    dl->AddPolyline(frame, 8, c, 1.0f, ImDrawFlags_Closed);

    // Iris: the ring at the centre, outer radius 91 of 536. AddCircle puts the
    // stroke's outer edge on the radius, matching the ring's outer edge.
    const ImVec2 mid(o.x + s * 0.5f, o.y + s * 0.5f);
    dl->AddCircle(mid, 0.17f * s, c, 16, 1.0f);

    // Bows: each caliper leg runs from a tip at the iris' top pole, up over
    // the crossing, down to the outer hinge at (304, 500), and back to a tip
    // at the bottom pole, so the two legs read as calipers measuring the iris.
    // The upper arm is one cubic fitted to the centreline of the SVG's filled
    // leg (within 0.05 px at 18 px); the lower arm is its mirror about the
    // horizontal axis, and the right leg mirrors the left about the vertical.
    constexpr float tip_u = 0.5614f, tip_v = 0.3275f;   // (533, 408)
    constexpr float c1_u = 0.3636f, c1_v = 0.1384f;
    constexpr float c2_u = 0.2554f, c2_v = 0.3273f;
    constexpr float hinge_u = 0.1343f;                   // (304, 500)
    const float hinge_r = ImMax(0.75f, 0.05f * s);
    for (int side = 0; side < 2; ++side) {
        auto at = [&](float u, float v) {
            return ImVec2(o.x + (side == 0 ? u : 1.0f - u) * s, o.y + v * s);
        };
        dl->PathLineTo(at(tip_u, tip_v));
        dl->PathBezierCubicCurveTo(at(c1_u, c1_v), at(c2_u, c2_v),
                                   at(hinge_u, 0.5f), 8);
        dl->PathBezierCubicCurveTo(at(c2_u, 1.0f - c2_v), at(c1_u, 1.0f - c1_v),
                                   at(tip_u, 1.0f - tip_v), 8);
        dl->PathStroke(c, 1.0f);
        // The hinge, a knob where the leg turns; filled because a ring this
        // small would close up.
        dl->AddCircleFilled(at(hinge_u, 0.5f), hinge_r, c, 12);
    }
}

void text_tracked(ImDrawList* dl, ImFont* font, float size, ImVec2 pos, ImU32 c,
                  const char* text, float tracking_px) {
    if (!dl || !text || text[0] == '\0')
        return;
    if (!font)
        font = ImGui::GetFont();
    if (size <= 0.0f)
        size = ImGui::GetFontSize();
    ImFontBaked* baked = font->GetFontBaked(size);
    if (!baked)
        return;
    const float scale = size / baked->Size;

    // AddText truncates each call's position to a whole logical pixel, which
    // would make fractional tracking uneven glyph to glyph. Snap the run's
    // origin as AddText does, then place each glyph on the device pixel grid
    // (the rasteriser density: 2 on Retina).
    const float density = ImMax(font->CurrentRasterizerDensity, 1.0f);
    const float x0 = ImFloor(pos.x);
    const float y = ImFloor(pos.y);
    const ImDrawListFlags saved = dl->Flags;
    dl->Flags |= ImDrawListFlags_TextNoPixelSnap;
    float pen = 0.0f;
    const char* s = text;
    const char* end = text + std::strlen(text);
    while (s < end) {
        unsigned int cp = 0;
        const int n = ImTextCharFromUtf8(&cp, s, end);
        if (n <= 0)
            break;
        const float x = x0 + ImFloor(pen * density + 0.5f) / density;
        dl->AddText(font, size, ImVec2(x, y), c, s, s + n);
        pen += baked->GetCharAdvance(static_cast<ImWchar>(cp)) * scale + tracking_px;
        s += n;
    }
    dl->Flags = saved;
}

float text_tracked_width(ImFont* font, float size, const char* text, float tracking_px) {
    if (!text || text[0] == '\0')
        return 0.0f;
    if (!font)
        font = ImGui::GetFont();
    if (size <= 0.0f)
        size = ImGui::GetFontSize();
    ImFontBaked* baked = font->GetFontBaked(size);
    if (!baked)
        return 0.0f;
    const float scale = size / baked->Size;
    // The inked extent: the tracking after the last glyph is not part of the
    // word, so centred tracked text stays centred.
    float width = 0.0f;
    int glyphs = 0;
    const char* s = text;
    const char* end = text + std::strlen(text);
    while (s < end) {
        unsigned int cp = 0;
        const int n = ImTextCharFromUtf8(&cp, s, end);
        if (n <= 0)
            break;
        width += baked->GetCharAdvance(static_cast<ImWchar>(cp)) * scale;
        ++glyphs;
        s += n;
    }
    return glyphs > 0 ? width + tracking_px * static_cast<float>(glyphs - 1) : 0.0f;
}

// ---- Widgets --------------------------------------------------------------------

void label(const char* text) { label(text, col::secondary); }

void label(const char* text, ImU32 colour) {
    char upper[256];
    upper_ascii(text ? text : "", upper, sizeof(upper));
    text_run(fonts().ui, font_size::ui_small, upper, colour, metric::tracking);
}

bool key_button(const char* key, const char* action, bool enabled,
                const char* disabled_reason) {
    ImGuiWindow* window = ImGui::GetCurrentWindow();
    if (window->SkipItems)
        return false;
    ImGuiContext& g = *GImGui;

    // The key is part of the id, so an action-less ↑ and ↓ stay distinct.
    ImGui::PushID(key);
    const ImGuiID id = ImGui::GetID(action ? action : "");
    ImGui::PopID();
    const KeyRun run = measure_key_run(key, action, nullptr, font_size::ui);
    const ImVec2 size(ImCeil(run.width + 2.0f * kKeyPadX), metric::key_button);
    const float text_y = ImFloor((size.y - run.height) * 0.5f);

    // On the line's text baseline, as word_button does: in a strip or a menu
    // bar (AlignTextToFramePadding) the plate drops so its words sit level
    // with the text beside it, and the line grows by the drop.
    const float drop = ImMax(0.0f, window->DC.CurrLineTextBaseOffset - text_y);
    const ImVec2 pos(ImFloor(window->DC.CursorPos.x),
                     ImFloor(window->DC.CursorPos.y + drop));
    const ImRect bb(pos, ImVec2(pos.x + size.x, pos.y + size.y));
    ImGui::ItemSize(ImVec2(size.x, size.y + drop), text_y + drop);

    // Fitts: the whole plate is the target.
    ImGui::BeginDisabled(!enabled);
    bool pressed = false;
    const bool visible = ImGui::ItemAdd(bb, id);
    if (visible) {
        bool hovered = false;
        bool held = false;
        pressed = ImGui::ButtonBehavior(bb, id, &hovered, &held);
        const bool disabled = (ImGui::GetItemFlags() & ImGuiItemFlags_Disabled) != 0;

        // Bronze only under the pointer (the key) and on keyboard focus (the
        // outline); at rest the key reads as secondary metadata. Disabled uses
        // its own tokens rather than the style alpha, like primary_button.
        ImU32 outline = styled(col::rule);
        ImU32 fill = 0;
        ImU32 key_ink = styled(col::secondary);
        ImU32 action_ink = styled(col::parchment);
        if (disabled) {
            outline = col::rule_faint;
            key_ink = col::faint;
            action_ink = col::faint;
        } else if (held && hovered) {
            fill = styled(col::parchment);
            outline = fill;
            key_ink = styled(col::deep);
            action_ink = key_ink;
        } else if (hovered) {
            fill = styled(col::plate_hover);
            key_ink = styled(col::bronze);
        }
        if (!disabled && g.NavId == id && g.NavCursorVisible)
            outline = styled(col::bronze);

        ImDrawList* dl = window->DrawList;
        if (fill != 0)
            dl->AddRectFilled(bb.Min, bb.Max, fill);
        box(dl, bb, outline);
        const ImVec2 text_pos(pos.x + ImFloor((size.x - run.width) * 0.5f),
                              pos.y + text_y);
        draw_key_run(dl, text_pos, run, key, action, nullptr, font_size::ui, key_ink,
                     action_ink);
    }
    ImGui::EndDisabled();

    // After EndDisabled, so the tooltip itself is not drawn at disabled alpha.
    if (visible && !enabled)
        reason_tooltip(disabled_reason);
    return pressed;
}

void pane_header(const char* title, const char* subtitle, const char* shortcut,
                 bool focused) {
    ImGuiWindow* window = ImGui::GetCurrentWindow();
    if (window->SkipItems)
        return;
    ImDrawList* dl = window->DrawList;
    const ImGuiStyle& st = ImGui::GetStyle();

    // The strip spans the pane edge to edge, ignoring WindowPadding, directly
    // under any title or dock tab bar. It is placed from the window rect, not
    // the cursor, so it stays put when the pane's content scrolls.
    const float border = window->WindowBorderSize;
    const ImVec2 p0(ImFloor(window->Pos.x + border), ImFloor(window->InnerRect.Min.y));
    const ImVec2 p1(ImFloor(window->Pos.x + window->Size.x - border),
                    p0.y + metric::pane_header);
    // The window clip rect is inset by half the side padding; clip to the
    // strip itself instead, within what the host leaves of the window.
    ImRect clip(p0, p1);
    clip.ClipWith(window->OuterRectClipped);
    dl->PushClipRect(clip.Min, clip.Max, false);
    // The strip is the pane's key button: a click on it focuses the pane (as
    // on any window), the same as the shortcut on its right, so an unfocused
    // strip lifts under the pointer.
    const bool hovered = !focused && ImGui::IsWindowHovered() &&
                         ImGui::IsMouseHoveringRect(clip.Min, clip.Max, false);
    dl->AddRectFilled(p0, p1, styled(hovered ? col::plate_hover : col::plate));
    // Focus is pre-attentive: the focused pane's title, key and closing rule
    // turn bronze together; every other pane stays parchment over a rule.
    rule_h(dl, p0.x, p1.x, p1.y - 1.0f, styled(focused ? col::bronze : col::rule));

    // Text centred optically on the plate above the rule: the capitals' ink,
    // as in masthead().
    const float centre = p0.y + (metric::pane_header - 1.0f) * 0.5f;
    ImGui::PushFont(fonts().ui, font_size::ui_small);
    ImFont* ui = ImGui::GetFont();
    const float size = ImGui::GetFontSize();
    float ink_top = 0.0f;
    float ink_bottom = size;
    caps_ink(ui, size, &ink_top, &ink_bottom);
    const float text_y = ImFloor(centre - (ink_top + ink_bottom) * 0.5f + 0.5f);
    float x = p0.x + kPaneInsetX;
    if (title) {
        char upper[128];
        upper_ascii(title, upper, sizeof(upper));
        const float tracking = metric::tracking * size;
        text_tracked(dl, ui, size, ImVec2(x, text_y),
                     styled(focused ? col::bronze : col::parchment), upper, tracking);
        x += text_tracked_width(ui, size, upper, tracking) + metric::gap;
    }
    if (subtitle)
        dl->AddText(ui, size, ImVec2(x, text_y), styled(col::secondary), subtitle);
    ImGui::PopFont();

    if (shortcut) {
        // Mono on the title's baseline, right-aligned to the same inset.
        const float nudge = baseline_of(fonts().ui, font_size::ui_small) -
                            baseline_of(fonts().mono, font_size::ui_small);
        ImGui::PushFont(fonts().mono, font_size::ui_small);
        ImFont* mono = ImGui::GetFont();
        const float mono_size = ImGui::GetFontSize();
        const float w = mono->CalcTextSizeA(mono_size, FLT_MAX, 0.0f, shortcut).x;
        dl->AddText(mono, mono_size,
                    ImVec2(ImFloor(p1.x - kPaneInsetX - w), text_y + nudge),
                    styled(focused ? col::bronze : col::secondary), shortcut);
        ImGui::PopFont();
    }
    dl->PopClipRect();

    // Pin the strip. Called first in the pane (before any item or clip rect),
    // swap the window clip rect Begin() pushed for one starting below the
    // strip, so scrolled content, and child windows, which clip to it, never
    // draw over the header. End() pops it as usual; the stack stays balanced.
    // Then reserve the strip so content starts one WindowPadding below it.
    const float below = window->DC.CursorStartPos.y + metric::pane_header;
    const bool first_item =
        ImFabs(window->DC.CursorPos.x - window->DC.CursorStartPos.x) < 0.5f &&
        ImFabs(window->DC.CursorPos.y - window->DC.CursorStartPos.y) < 0.5f;
    if (first_item) {
        const ImRect inner = window->InnerClipRect;
        ImGui::PopClipRect();
        ImGui::PushClipRect(ImVec2(inner.Min.x, ImMax(inner.Min.y, p1.y)), inner.Max,
                            true);
        ImGui::ItemSize(ImVec2(0.0f, metric::pane_header - st.ItemSpacing.y));
    } else if (window->DC.CursorPos.y < below) {
        ImGui::SetCursorScreenPos(ImVec2(window->DC.CursorStartPos.x, below));
    }
}

bool primary_button(const char* label, ImVec2 size_arg, const char* key) {
    ImGuiWindow* window = ImGui::GetCurrentWindow();
    if (window->SkipItems)
        return false;
    ImGuiContext& g = *GImGui;

    ImGui::PushFont(fonts().ui, font_size::ui);
    const ImGuiID id = window->GetID(label);
    const char* label_end = ImGui::FindRenderedTextEnd(label);
    const KeyRun run = measure_key_run(key, label, label_end, font_size::ui);
    const ImVec2 size(
        size_arg.x > 0.0f ? size_arg.x : ImCeil(run.width + 2.0f * kPlatePadX),
        size_arg.y > 0.0f ? size_arg.y : kPlateHeight);
    const ImVec2 pos = ImFloor(window->DC.CursorPos);
    const ImRect bb(pos, ImVec2(pos.x + size.x, pos.y + size.y));
    const float text_y = ImFloor((size.y - run.height) * 0.5f);

    // Fitts: the whole plate is the target. Reporting the label's offset as
    // the text baseline lets text on the same line sit level with the label.
    ImGui::ItemSize(size, text_y);
    if (!ImGui::ItemAdd(bb, id)) {
        ImGui::PopFont();
        return false;
    }
    bool hovered = false;
    bool held = false;
    const bool pressed = ImGui::ButtonBehavior(bb, id, &hovered, &held);
    const bool disabled = (ImGui::GetItemFlags() & ImGuiItemFlags_Disabled) != 0;

    // States (SPEC §6.7). Disabled uses its own tokens rather than the style
    // alpha, so the honest disabled plate is exactly rule over faint. The
    // primary's key is bronze, the accent for the one main action.
    ImU32 outline = styled(col::parchment);
    ImU32 fill = 0;
    ImU32 ink = styled(col::parchment);
    ImU32 key_ink = styled(col::bronze);
    if (disabled) {
        outline = col::rule;
        ink = col::faint;
        key_ink = col::faint;
    } else if (held && hovered) {
        fill = styled(col::parchment);
        ink = styled(col::deep);
        key_ink = ink;
    } else if (hovered) {
        fill = styled(col::plate_hover);
    }
    ImDrawList* dl = window->DrawList;
    cut_plate(dl, bb.Min, bb.Max, outline, fill);

    // Keyboard focus: a second cut outline 3 px outside, in bronze (the focus
    // accent). Its cut grows by about 3 × (√2 − 1) to keep the diagonals
    // parallel. Replaces ImGui's square nav rectangle.
    if (!disabled && g.NavId == id && g.NavCursorVisible) {
        const ImVec2 d(kFocusOffset, kFocusOffset);
        cut_plate(dl, ImVec2(bb.Min.x - d.x, bb.Min.y - d.y),
                  ImVec2(bb.Max.x + d.x, bb.Max.y + d.y), styled(col::bronze), 0,
                  metric::cut + 1.0f);
    }

    const ImVec2 text_pos(pos.x + ImFloor((size.x - run.width) * 0.5f), pos.y + text_y);
    draw_key_run(dl, text_pos, run, key, label, label_end, font_size::ui, key_ink, ink);
    ImGui::PopFont();
    return pressed;
}

bool word_button(const char* label) {
    ImGuiWindow* window = ImGui::GetCurrentWindow();
    if (window->SkipItems)
        return false;
    ImGuiContext& g = *GImGui;
    const ImGuiStyle& st = g.Style;

    ImGui::PushFont(fonts().ui, font_size::ui);
    const ImGuiID id = window->GetID(label);
    const char* label_end = ImGui::FindRenderedTextEnd(label);
    const ImVec2 text = ImGui::CalcTextSize(label, label_end, false);

    // Fitts: the word is small but its target is a full frame height, laid
    // out like ImGui::Button so it lines up with framed widgets and text.
    ImVec2 pos = window->DC.CursorPos;
    if (st.FramePadding.y < window->DC.CurrLineTextBaseOffset)
        pos.y += window->DC.CurrLineTextBaseOffset - st.FramePadding.y;
    const ImVec2 size(text.x, text.y + st.FramePadding.y * 2.0f);
    const ImRect bb(pos, ImVec2(pos.x + size.x, pos.y + size.y));
    ImGui::ItemSize(size, st.FramePadding.y);
    if (!ImGui::ItemAdd(bb, id)) {
        ImGui::PopFont();
        return false;
    }
    bool hovered = false;
    bool held = false;
    const bool pressed = ImGui::ButtonBehavior(bb, id, &hovered, &held);
    const bool disabled = (ImGui::GetItemFlags() & ImGuiItemFlags_Disabled) != 0;
    const bool focused = !disabled && g.NavId == id && g.NavCursorVisible;

    ImDrawList* dl = window->DrawList;
    const ImVec2 text_pos(ImFloor(pos.x), ImFloor(pos.y + st.FramePadding.y));
    dl->AddText(ImGui::GetFont(), ImGui::GetFontSize(), text_pos,
                disabled ? col::faint : styled(col::parchment), label, label_end);

    // The underline is the link vocabulary: parchment under the pointer,
    // bronze under keyboard focus, at baseline + 2.
    if (hovered || focused) {
        const float base = baseline_of(nullptr, font_size::ui);
        rule_h(dl, text_pos.x, text_pos.x + text.x, text_pos.y + base + 2.0f,
               styled(focused && !hovered ? col::bronze : col::parchment));
    }
    ImGui::PopFont();
    return pressed;
}

void unavailable_action(const char* label, const char* reason) {
    // Honest disabled action: the action keeps its place and says why it
    // cannot run, in plain sight rather than in a tooltip box. It takes the
    // same frame-height line as a live word_button.
    ImGui::PushFont(fonts().ui, font_size::ui);
    ImGui::AlignTextToFramePadding();
    ImGui::PopFont();
    const float label_base = baseline_of(fonts().ui, font_size::ui);
    const float reason_base = baseline_of(fonts().ui, font_size::ui_small);
    text_run(fonts().ui, font_size::ui, label, col::faint, 0.0f);
    ImGui::SameLine(0.0f, kReasonGap);
    text_run(fonts().ui, font_size::ui_small, reason, col::faint, 0.0f,
             ImMax(0.0f, label_base - reason_base));
}

// ---- Tables -----------------------------------------------------------------------

bool begin_table(const char* id, int columns, ImVec2 size, ImGuiTableFlags extra) {
    // A caller's own sizing policy replaces the default rather than OR-ing
    // into it (the policies are values in a mask, not independent bits).
    const ImGuiTableFlags sizing =
        (extra & ImGuiTableFlags_SizingMask_) ? 0 : ImGuiTableFlags_SizingStretchProp;
    const ImGuiTableFlags flags = ImGuiTableFlags_ScrollY |
                                  ImGuiTableFlags_NoBordersInBody |
                                  ImGuiTableFlags_NoPadOuterX | sizing | extra;

    // CellPadding stays pushed for the rows (ImGui reads it per row) and is
    // popped in end_table(). The scrolling child gets no ground of its own so
    // the bottomrule drawn by end_table() on the host shows through it, and
    // the strong border colour is cleared so ImGui's own line at the frozen
    // header does not double the midrule while scrolled.
    ImGui::PushStyleVar(ImGuiStyleVar_CellPadding, kCellPadding);
    ImGui::PushStyleColor(ImGuiCol_ChildBg, IM_COL32_BLACK_TRANS);
    ImGui::PushStyleColor(ImGuiCol_TableBorderStrong, IM_COL32_BLACK_TRANS);
    const bool open = ImGui::BeginTable(id, columns, flags, size);
    ImGui::PopStyleColor(2);
    if (!open)
        ImGui::PopStyleVar();
    return open;
}

void end_table() {
    ImGuiTable* table = ImGui::GetCurrentTable();
    ImGui::EndTable();
    ImGui::PopStyleVar();
    if (!table)
        return;
    // Bottomrule (booktabs, `faint`): just under the last row, or along the
    // bottom of the scroll area when the rows overflow it. RowPosY2 and the
    // clip rect stay valid after EndTable for this frame.
    const float y = ImMin(table->RowPosY2, table->InnerClipRect.Max.y - 1.0f);
    if (y >= table->InnerClipRect.Min.y)
        rule_h(ImGui::GetWindowDrawList(), table->BorderX1, table->BorderX2, y,
               col::faint);
}

void table_header_row() {
    ImGuiTable* table = ImGui::GetCurrentTable();
    IM_ASSERT(table != nullptr && "table_header_row() outside begin_table()");
    if (!table)
        return;

    // Sticky header: the header row stays put while the body scrolls, as in
    // a spreadsheet. Keeps any column freeze the caller already asked for.
    if (!table->IsLayoutLocked && (table->Flags & ImGuiTableFlags_ScrollY) &&
        table->FreezeRowsRequest == 0)
        ImGui::TableSetupScrollFreeze(table->FreezeColumnsRequest, 1);

    // The header sits on its container's ground, like the scrolling child.
    ImGui::PushStyleColor(ImGuiCol_TableHeaderBg, IM_COL32_BLACK_TRANS);
    ImGui::TableNextRow(ImGuiTableRowFlags_Headers, metric::row);
    ImGui::PopStyleColor();
    if (table->HostSkipItems)
        return;
    const float row_y1 = table->RowPosY1;
    const float row_y2 = table->RowPosY2;

    // ImGui::TableHeader supplies the behaviour (click to sort, shift for
    // multi-sort, drag to reorder, context menu, keyboard nav); its own label
    // and arrow are drawn transparent and replaced by the house label. Only
    // headers that do something on click light up under the pointer.
    const bool sortable = (table->Flags & ImGuiTableFlags_Sortable) != 0;
    const bool clickable =
        (table->Flags & (ImGuiTableFlags_Sortable | ImGuiTableFlags_Reorderable)) != 0;
    ImGui::PushFont(fonts().ui, font_size::ui_small);
    ImFont* font = ImGui::GetFont();
    const float size = ImGui::GetFontSize();
    const float tracking = metric::tracking * size;
    float ink_top = 0.0f;
    float ink_bottom = size;
    caps_ink(font, size, &ink_top, &ink_bottom);   // the chevron centres on the caps
    ImDrawList* dl = ImGui::GetWindowDrawList();
    ImGui::PushStyleColor(ImGuiCol_Text, IM_COL32_BLACK_TRANS);
    if (!clickable) {
        ImGui::PushStyleColor(ImGuiCol_HeaderHovered, IM_COL32_BLACK_TRANS);
        ImGui::PushStyleColor(ImGuiCol_HeaderActive, IM_COL32_BLACK_TRANS);
    }
    for (int i = 0; i < table->ColumnsCount; ++i) {
        if (!ImGui::TableSetColumnIndex(i) && table->LastHeldHeaderColumn != i)
            continue;
        const ImGuiTableColumnFlags flags = ImGui::TableGetColumnFlags(i);
        const bool unnamed = (flags & ImGuiTableColumnFlags_NoHeaderLabel) != 0;
        const char* raw = unnamed ? "" : ImGui::TableGetColumnName(i);
        // A name starting with '>' marks a numeric column; the marker is not
        // drawn.
        const bool numeric = raw[0] == '>';
        char name[128];
        upper_ascii(numeric ? raw + 1 : raw, name, sizeof(name));
        const ImVec2 pos = ImGui::GetCursorScreenPos();
        const float right = pos.x + ImGui::GetContentRegionAvail().x;   // cell edge
        ImGui::PushID(i);
        ImGui::TableHeader("");
        ImGui::PopID();

        // A numeric head is right-aligned like its figures, with the chevron
        // to the left of the word. When it cannot fit it starts at the cell's
        // left edge instead, so it never strays into the previous column.
        const bool sorts = sortable && !(flags & ImGuiTableColumnFlags_NoSort);
        const float chevron_span = sorts ? kChevronGap + kChevronW + 1.0f : 0.0f;
        const float width = text_tracked_width(font, size, name, tracking);
        const float label_x =
            numeric ? ImMax(pos.x + chevron_span, ImFloor(right - width)) : pos.x;
        text_tracked(dl, font, size, ImVec2(label_x, pos.y), styled(col::secondary),
                     name, tracking);
        ImGuiTableColumn* column = &table->Columns[i];
        if (sorts && column->SortOrder != -1) {
            const float chevron_x = numeric ? label_x - chevron_span
                                            : label_x + width + kChevronGap;
            sort_chevron(dl, chevron_x, pos.y + (ink_top + ink_bottom) * 0.5f,
                         column->SortDirection == ImGuiSortDirection_Ascending,
                         styled(col::bronze));
        }
        // TableHeader("") reported no label width: feed the tracked label to
        // auto-fit, and its ink extent to the clip test so an overflowing label
        // stays clipped to its column instead of being merged into the host
        // draw call.
        const float ideal_x = pos.x + width + chevron_span;
        const float ink_x = numeric ? label_x + width : ideal_x;
        float& used = column->ContentMaxXHeadersUsed;
        float& ideal = column->ContentMaxXHeadersIdeal;
        used = ImMax(used, ink_x);
        ideal = ImMax(ideal, ideal_x);
    }
    ImGui::PopStyleColor(clickable ? 1 : 3);
    ImGui::PopFont();

    // Toprule (`faint`) on the header's first pixel row and midrule (`rule`)
    // on its last, across the data width. They go in the table's background
    // channel, which spans every column and sits above the cell fills.
    ImGui::TablePushBackgroundChannel();
    rule_h(dl, table->BorderX1, table->BorderX2, row_y1, col::faint);
    rule_h(dl, table->BorderX1, table->BorderX2, row_y2 - 1.0f, col::rule);
    ImGui::TablePopBackgroundChannel();
}

bool table_row_selectable(const char* id, bool selected, bool* double_clicked) {
    const ImVec2 spacing = ImGui::GetStyle().ItemSpacing;
    const float pad_y = ImGui::GetStyle().CellPadding.y;
    ImGui::PushStyleColor(ImGuiCol_Header, col::plate);
    ImGui::PushStyleColor(ImGuiCol_HeaderHovered, col::plate_hover);
    ImGui::PushStyleColor(ImGuiCol_HeaderActive, col::plate_hover);
    // Fitts: the hit target and the fill are the whole row, padding included.
    // Selectable pads its box by half the item spacing above and below, so
    // spacing of twice the cell padding makes the box exactly metric::row.
    ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, ImVec2(spacing.x, 2.0f * pad_y));
    const bool pressed = ImGui::Selectable(
        id, selected,
        ImGuiSelectableFlags_SpanAllColumns | ImGuiSelectableFlags_AllowDoubleClick |
            ImGuiSelectableFlags_AllowOverlap,
        ImVec2(0.0f, metric::row - 2.0f * pad_y));
    ImGui::PopStyleVar();
    ImGui::PopStyleColor(3);
    if (double_clicked)
        *double_clicked = ImGui::IsItemHovered() &&
                          ImGui::IsMouseDoubleClicked(ImGuiMouseButton_Left);
    return pressed;
}

ImU32 row_text(bool selected, ImU32 rest) {
    // Bloomberg: the focused row's figures turn amber; nothing else changes.
    return selected ? col::bronze : rest;
}

// ---- Strips -----------------------------------------------------------------------

void command_bar_begin() {
    // Frame-height padding centres the command line (a framed InputText) in
    // the strip; AlignTextToFramePadding then centres plain text with it.
    strip_begin("##sternbild_command_bar", ImGuiDir_Up, metric::command_bar,
                ImGui::GetFrameHeight(), col::ground);
    strip_rule(true);
    ImGui::AlignTextToFramePadding();
}

void command_bar_end() { ImGui::End(); }

void status_bar_begin() {
    // Status bar (VS Code / JetBrains): small secondary text on the deep
    // ground. Its contents default to the small UI size.
    const float frame = font_size::ui_small + 2.0f * ImGui::GetStyle().FramePadding.y;
    strip_begin("##sternbild_status_bar", ImGuiDir_Down, metric::status_bar, frame,
                col::deep);
    strip_rule(false);
    ImGui::PushFont(fonts().ui, font_size::ui_small);
    ImGui::AlignTextToFramePadding();
}

void status_bar_end() {
    ImGui::PopFont();
    ImGui::End();
}

void status_item(const char* text, ImU32 c) {
    const int frame = ImGui::GetFrameCount();
    const ImGuiID window = ImGui::GetCurrentWindowRead()->ID;
    const bool first = g_status_line.frame != frame || g_status_line.window != window;
    g_status_line = StatusLine{frame, window};

    ImGui::PushFont(fonts().ui, font_size::ui_small);
    if (!first) {
        // Items stand metric::gap apart with the faint middle dot centred in
        // the gap.
        ImGui::SameLine(0.0f, metric::gap * 0.5f);
        ImGui::PushStyleColor(ImGuiCol_Text, col::faint);
        ImGui::TextUnformatted("\xC2\xB7");
        ImGui::PopStyleColor();
        ImGui::SameLine(0.0f, metric::gap * 0.5f);
    }
    ImGui::PushStyleColor(ImGuiCol_Text, c);
    ImGui::TextUnformatted(text);
    ImGui::PopStyleColor();
    ImGui::PopFont();
}

void masthead() {
    ImGuiWindow* window = ImGui::GetCurrentWindow();
    if (window->SkipItems)
        return;
    ImDrawList* dl = window->DrawList;
    const ImVec2 cursor = window->DC.CursorPos;
    // Centre the block on the current text line (in the command bar, the
    // strip's centre once AlignTextToFramePadding has run).
    const float mid =
        cursor.y + window->DC.CurrLineTextBaseOffset + ImGui::GetFontSize() * 0.5f;

    // The house is present, not loud: the mark once, small, at the top left.
    const ImVec2 mark_min(ImFloor(cursor.x), ImFloor(mid - kMarkSize * 0.5f + 0.5f));
    caliper_mark(dl, mark_min, kMarkSize, styled(col::parchment));

    // Wordmark in Sternbild Display; without it, the UI face in capitals.
    ImFont* face = fonts().display;
    const float base = face ? font_size::display : font_size::ui;
    ImGui::PushFont(face ? face : fonts().ui, base);
    ImFont* font = ImGui::GetFont();
    const float size = ImGui::GetFontSize();
    const float tracking = kWordmarkTrack * font_size::display;
    const char* word = "CALIPER";
    const float width = text_tracked_width(font, size, word, tracking);
    // Optical centring: the capitals' ink, not the line box, on the mark's
    // centre line.
    float ink_top = 0.0f;
    float ink_bottom = size;
    caps_ink(font, size, &ink_top, &ink_bottom);
    const float word_x = mark_min.x + kMarkSize + kMarkGap;
    const float word_y = ImFloor(mid - (ink_top + ink_bottom) * 0.5f + 0.5f);
    text_tracked(dl, font, size, ImVec2(word_x, word_y), styled(col::parchment), word,
                 tracking);
    ImGui::PopFont();

    ImGui::Dummy(ImVec2(word_x + width - cursor.x, ImGui::GetTextLineHeight()));
    ImGui::SameLine(0.0f, metric::gap);
}

}  // namespace sternbild
