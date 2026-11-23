#pragma once
// ============================================================================
// Sternbild house style for the Caliper chrome (SPEC-sternbild.md, workstation
// revision). The window is a register of tiled panes set on 1 px rules, in the
// density of a Bloomberg terminal or Foundry: compact pane headers, dense
// tables with tabular figures, a command bar, a persistent status bar.
//
// Colour belongs to data. The chrome is monochrome parchment on a warm night
// ground, and bronze is the single accent: focus, selection and keys only.
// Square corners everywhere. No gradients, shadows, pills, chips or cards.
//
// Everything here is an ImDrawList rect, line, convex polygon or text, so
// applets can adopt it later without any host dependency beyond Dear ImGui.
// ============================================================================

#include <imgui.h>
#include <sternbild_tokens.h>

namespace sternbild {

// The values below are read from the generated header of the Sternbild design
// tokens (cmake/SternbildTokens.cmake). The organization repository owns them:
// change a value there, never here. scripts/check_theme_colors.py fails when a
// raw colour literal appears in this file outside the region that is marked
// "colours without a token". The tokens are reached as sternbild::tokens::...

// ---- Colour tokens (role and data colours of the design tokens) -------------
namespace col {
// deep         role well: status bar, plot fields, inspector wells
// ground       role background-app: window and pane ground
// plate        role surface: pane headers, menus, the selected row
// plate_hover  role surface-hover: hovered row, menu item or plate
// rule         role rule: every 1 px rule;  rule_faint: hatching, dividers inside tables
// parchment    role text: primary text;  secondary (text-muted): metadata, labels, status text
// faint        role text-faint: placeholders, disabled, booktabs top and bottom rules
// bronze       role accent: focus, selection, keys, the star
constexpr ImU32 deep        = tokens::role::well.imu32;
constexpr ImU32 ground      = tokens::role::background_app.imu32;
constexpr ImU32 plate       = tokens::role::surface.imu32;
constexpr ImU32 plate_hover = tokens::role::surface_hover.imu32;
constexpr ImU32 rule        = tokens::role::rule.imu32;
constexpr ImU32 rule_faint  = tokens::role::rule_faint.imu32;
constexpr ImU32 parchment   = tokens::role::text.imu32;
constexpr ImU32 secondary   = tokens::role::text_muted.imu32;
constexpr ImU32 faint       = tokens::role::text_faint.imu32;
constexpr ImU32 bronze      = tokens::role::accent.imu32;

// Plot class pigments (data colours of the tokens). Data only, never chrome; fixed order.
constexpr ImU32 vehicle = tokens::data::smalt.imu32;
constexpr ImU32 dog     = tokens::data::verdigris.imu32;
constexpr ImU32 bird    = tokens::data::olive_ochre.imu32;
constexpr ImU32 cat     = tokens::data::madder.imu32;
constexpr ImU32 other   = tokens::data::slate.imu32;

// BEGIN colours without a token
// Colours the app draws that the design tokens 1.1.0 do not cover. Each one is
// Caliper's own constant; a token that names the same role replaces it here.

// The axis gizmo of a scene view: x, y and z.
constexpr ImU32 axis_x = IM_COL32(222, 92, 84, 255);
constexpr ImU32 axis_y = IM_COL32(112, 186, 104, 255);
constexpr ImU32 axis_z = IM_COL32(96, 146, 232, 255);

// Status words and messages. The tokens have no status colours.
constexpr ImU32 warning       = IM_COL32(255, 196, 120, 255);  // a stage message over a scene
constexpr ImU32 error_text    = IM_COL32(230, 120, 90, 255);   // a spec panel that failed to draw
constexpr ImU32 status_error  = IM_COL32(255, 102, 77, 255);   // the live page: a picker error
constexpr ImU32 status_failed = IM_COL32(255, 89, 77, 255);    // the live page: the session failed
constexpr ImU32 status_wait   = IM_COL32(255, 204, 77, 255);   // the live page: waiting for the session
constexpr ImU32 status_run    = IM_COL32(115, 217, 115, 255);  // the live page: running

// A ring around the point under the pointer in a plot.
constexpr ImU32 hover_ring = IM_COL32(255, 255, 255, 230);

// A scrim or shadow: black at an alpha, under labels and over pictures.
constexpr ImU32 shade(int alpha) { return IM_COL32(0, 0, 0, alpha); }

// Rule and text of a plot whose host sets no look of its own, and the colour
// of points that carry none.
constexpr ImU32 plot_rule  = IM_COL32(0x5a, 0x57, 0x51, 255);
constexpr ImU32 plot_faint = IM_COL32(0xa4, 0x9d, 0x8f, 255);
constexpr ImU32 flat_point = IM_COL32(220, 220, 230, 255);

// The defaults of a view window header (live_views_state.h Look) and of the
// run bar (live_controls.h ControlsLook), for a host that sets no look.
constexpr ImU32 look_text    = IM_COL32(230, 230, 235, 255);
constexpr ImU32 look_faint   = IM_COL32(140, 140, 150, 255);
constexpr ImU32 look_rule    = IM_COL32(90, 90, 100, 255);
constexpr ImU32 look_accent  = IM_COL32(110, 170, 255, 255);
constexpr ImU32 bar_text      = IM_COL32(255, 255, 255, 255);
constexpr ImU32 bar_secondary = IM_COL32(190, 190, 190, 255);
constexpr ImU32 bar_faint     = IM_COL32(128, 128, 128, 255);
// END colours without a token
}  // namespace col

inline ImVec4 to_vec4(ImU32 c) { return ImGui::ColorConvertU32ToFloat4(c); }
// Same rgb, alpha 0..255.
constexpr ImU32 with_alpha(ImU32 c, int alpha) {
    const ImU32 a = static_cast<ImU32>(alpha < 0 ? 0 : (alpha > 255 ? 255 : alpha));
    return (c & ~IM_COL32_A_MASK) | (a << IM_COL32_A_SHIFT);
}

// ---- Metrics (logical px at 1x; ImGui scales to the framebuffer) ----------
// The application metrics and the spacing scale of the design tokens.
namespace metric {
constexpr float row         = tokens::app::row;           // table / list row: 15 px type + 2 x 3
constexpr float pane_header = tokens::app::pane_header;   // pane header strip (landing/final)
constexpr float command_bar = tokens::app::command_bar;   // masthead + command line strip at the top
constexpr float status_bar  = tokens::app::status_bar;    // status strip at the bottom
constexpr float pad         = tokens::app::pad;           // pane inner padding
constexpr float gap         = tokens::app::gap;           // between inline status items
constexpr float key_button  = tokens::app::key_button;    // key_button height: the 15 px line + 5
constexpr float cut         = tokens::CUT;                // 45 degree cut on the primary plate corners
constexpr float tracking    = tokens::tracking::ui_label; // em tracking for uppercase labels
}  // namespace metric

// ---- Style ------------------------------------------------------------------
// Applies the house style to the current ImGui context: radius 0 everywhere,
// 1 px borders, palette-mapped ImGuiCol_* (window/child/popup = ground,
// frames = plate, header/selection = plate_hover, text = parchment, disabled
// text = faint, borders/separators = rule, tab overline = bronze, docking
// preview = bronze at low alpha, no other hue). Also sets FramePadding and
// ItemSpacing so default widgets land on the dense metrics above. Call once,
// after sternbild::load_fonts() (it reads the UI font size).
void apply_style();

// ---- Draw-list primitives (screen coordinates) -----------------------------
// Rules fill exactly one pixel row/column as a 1 px filled rect (no AA
// fringe at any framebuffer scale); callers pass integral coordinates.
void rule_h(ImDrawList* dl, float x0, float x1, float y, ImU32 c = col::rule);
void rule_v(ImDrawList* dl, float x, float y0, float y1, ImU32 c = col::rule);

// The primary-action plate (SPEC §6.7): a 1 px outline with 45° cuts of
// `cut` px on all four corners, drawn on half pixels; optional fill (0 = none).
void cut_plate(ImDrawList* dl, ImVec2 p_min, ImVec2 p_max, ImU32 outline,
               ImU32 fill = 0, float cut = metric::cut);

// The open star (SPEC §6.13): four separate triangular arms, tip at R, base on
// the inner vertices (±0.276R across, 0.356R from the centre), each arm with a
// 1 px `halo` stroke so the centre stays open and the point's own colour shows.
void open_star(ImDrawList* dl, ImVec2 centre, float tip_radius,
               ImU32 fill = col::bronze, ImU32 halo = col::deep);

// The live diamond (SPEC §6.5): a square rotated 45°, `half` px from centre to
// tip. Filled when a run is live, outlined (1 px) when there is none.
void diamond(ImDrawList* dl, ImVec2 centre, float half, ImU32 c, bool filled);

// The Caliper mark (the cut-corner instrument frame holding the iris and the
// two caliper bows), reduced for 14–24 px use. Source geometry:
// ~/Artisan/caliper-product/sternbild/marks/caliper-clean-parchment.svg,
// viewBox 232 232 536 536. `size` is the box side; strokes are 1 px.
void caliper_mark(ImDrawList* dl, ImVec2 p_min, float size,
                  ImU32 c = col::parchment);

// Text with letter-spacing (ImGui has none): draws glyph by glyph, adding
// `tracking_px` after each. Used for the CALIPER wordmark and uppercase labels.
void  text_tracked(ImDrawList* dl, ImFont* font, float size, ImVec2 pos,
                   ImU32 c, const char* text, float tracking_px);
float text_tracked_width(ImFont* font, float size, const char* text,
                         float tracking_px);

// ---- Widgets (ImGui cursor flow) -------------------------------------------
// Label (column heads, field labels, pane captions): the UI face at the small
// size, `secondary`, uppercase with metric::tracking. Pass the text as written;
// the helper uppercases ASCII. Advances the cursor like ImGui::Text.
void label(const char* text);
void label(const char* text, ImU32 colour);

// Key button ("recognition over recall" without a hint beside the control,
// Bloomberg function keys): one target holding the key and the action it runs,
// "⌘K Command", "esc Home", so the key is learnt from the button that does the
// same thing. The key in mono at the small size, then the action in the UI
// face; a null or empty `action` leaves the key alone (an ↑ ↓ pair). Square
// corners, 1 px outline, metric::key_button tall, sitting on the line's text
// baseline like word_button. States: rest = rule outline, secondary key,
// parchment action, no fill; hover = plate_hover fill, bronze key; keyboard
// focus = bronze outline; pressed = parchment fill with deep ink; disabled
// (`enabled` false, or inside BeginDisabled) = rule_faint outline, faint ink,
// and `disabled_reason` in a tooltip on hover. Returns true when clicked.
bool key_button(const char* key, const char* action, bool enabled = true,
                const char* disabled_reason = nullptr);

// Pane header (Bloomberg Launchpad, VS Code panel; landing/final rows 82/245/
// 633): a metric::pane_header-tall strip on `plate` across the full pane
// width at the top of a docked pane, ignoring WindowPadding. Left: the title
// uppercase and tracked (bronze when `focused`, else parchment), then the
// subtitle in secondary (e.g. "none yet", "11 installed"; may be null). Right:
// the pane's shortcut in mono (bronze when focused, else secondary; may be
// null). A 1 px rule closes the strip: bronze when focused, else `rule`. The
// strip is the pane's own key button: a click on it focuses the pane, as the
// shortcut does, so an unfocused strip lifts to plate_hover under the pointer.
// Advances the cursor to the first content line below the strip.
void pane_header(const char* title, const char* subtitle, const char* shortcut,
                 bool focused);

// Primary action: the cut-corner plate, the house's button for a pane's one
// main action. size.x == 0 fits the label plus 2 × 12 px padding; height is
// 24 px unless size.y is given. A `key` is drawn before the label as in
// key_button, in bronze (the primary's key: "⏎ Launch"). States: rest =
// parchment outline, no fill; hover = plate_hover fill; pressed = parchment
// fill with deep label; disabled (BeginDisabled) = rule outline, faint label
// and key. Returns true when clicked.
bool primary_button(const char* label, ImVec2 size = ImVec2(0, 0),
                    const char* key = nullptr);

// Secondary action: a bare word in parchment, 1 px underline on hover.
bool word_button(const char* label);

// Honest disabled action: the word in faint, and the reason in faint beside it
// ("Open folder…   not yet available"). Never a silent disabled control.
void unavailable_action(const char* label, const char* reason);

// Dense booktabs table (SPEC §6.9 in workstation density): wraps BeginTable with
// no vertical rules, no zebra, header row drawn as labels over a `rule`
// midrule, a `faint` toprule and bottomrule, rows metric::row tall, scrolling
// in Y. Flags passed in `extra` are OR'd in (e.g. ImGuiTableFlags_Sortable).
// Callers still use TableSetupColumn / TableNextRow / TableNextColumn.
bool begin_table(const char* id, int columns, ImVec2 size = ImVec2(0, 0),
                 ImGuiTableFlags extra = 0);
void end_table();
// After the TableSetupColumn calls: emits the header row in label style
// (uppercase, tracked, secondary; sort arrows in bronze when sortable). A
// column whose name begins with '>' is a numeric column: its head is
// right-aligned like its figures, and the marker is not drawn.
void table_header_row();

// Whole-row hit target for the current row (call first, in column 0, then
// ImGui::SameLine() and draw the cell). Hover = plate_hover fill, selected =
// plate fill; the caller colours the selected row's text with row_text().
// Returns true on click; *double_clicked is set on a double click.
bool table_row_selectable(const char* id, bool selected,
                          bool* double_clicked = nullptr);
ImU32 row_text(bool selected, ImU32 rest = col::parchment);  // bronze when selected

// Fixed strips. Each is a viewport side bar (ImGui::BeginViewportSideBar)
// across the main viewport: the command bar at the top, h metric::command_bar
// (ground, rule under); the status bar at the bottom, h metric::status_bar
// (deep, rule above); the tray bar, h `height` (ground, rule above), which
// takes its place above the status bar when it is begun after it. A side bar
// reserves its height from the viewport's work area, so a dock space laid out
// from the work area (ui/work_area.h) never runs under it; draw the bars
// before the dock space each frame. Between begin/end the caller lays out
// inline content.
void command_bar_begin();
void command_bar_end();
void status_bar_begin();
void status_bar_end();
void tray_bar_begin(float height);
void tray_bar_end();
// The control bar: a second strip under the command bar (or the page bar),
// h metric::command_bar (ground, rule under), for a page's own controls such
// as the live page's run bar. Begin it after the bar above it.
void control_bar_begin();
void control_bar_end();
// Inline status item: text in `c`, preceded by a faint " · " separator after
// the first item on the line.
void status_item(const char* text, ImU32 c = col::secondary);

// Masthead block drawn inside the command bar: the Caliper mark (18 px) and the
// CALIPER wordmark (Sternbild Display 16 px, 0.10 em tracking, parchment; falls
// back to the UI face uppercase when the Display font is absent). Advances the
// cursor past the block with a trailing metric::gap.
void masthead();

}  // namespace sternbild
