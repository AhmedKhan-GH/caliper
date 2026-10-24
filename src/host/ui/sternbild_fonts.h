#pragma once
// ============================================================================
// Font loading for the Sternbild chrome.
//
// Three faces, two families in the UI (SPEC §4 as revised for the workstation
// direction): a compact sans for labels, a mono for data and the command line,
// and Sternbild Display for the CALIPER wordmark and nothing else.
//
//   ui       IBM Plex Sans Condensed Regular   SIL OFL 1.1   assets/fonts (in repo)
//   mono     Source Code Pro Regular           SIL OFL 1.1   assets/fonts (in repo)
//   display  Sternbild Display Regular         house-owned   user path, never in repo
//
// Dear ImGui 1.92 rasterises on demand at the viewport's framebuffer scale
// (ImGuiBackendFlags_RendererHasTextures on Metal, GL3 and Vulkan), so one
// ImFont per face serves every size via ImGui::PushFont(font, size) and Retina
// needs no extra work.
// ============================================================================

#include <string>

struct ImFont;

namespace sternbild {

struct Fonts {
    ImFont* ui      = nullptr;   // labels, pane headers, menus, prose
    ImFont* mono    = nullptr;   // data cells, identifiers, paths, the command line
    ImFont* display = nullptr;   // the wordmark only; nullptr when the face is absent
    bool ok() const { return ui && mono; }
};

// Logical sizes in px at 1x. ImGui sizes a face by its ascender-to-descender
// height, not the em a browser uses, so these are the CSS sizes 12 / 11 px
// multiplied by each face's height ratio (Plex Condensed 1.30, Source Code
// Pro 1.257; Display's ratio is 1.0). Display never below 16; no text below
// CSS 11.
namespace font_size {
constexpr float ui       = 15.0f;  // CSS 11.5: labels, pane headers, buttons, prose
constexpr float ui_small = 14.0f;  // CSS 10.8: column heads, key hints, status
constexpr float mono     = 15.0f;  // CSS 11.9: data cells, identifiers, command line
constexpr float display  = 16.0f;  // CALIPER wordmark
}  // namespace font_size

// Where the faces come from. Resolution order:
//   assets_dir:   $CALIPER_FONTS_DIR, else <exe dir>/assets/fonts (copied by
//                 CMake), else CALIPER_SOURCE_DIR/assets/fonts (dev builds).
//   display_font: $CALIPER_DISPLAY_FONT, else
//                 caliper::app_data_path("fonts/SternbildDisplay-Regular.otf").
struct FontPaths {
    std::string assets_dir;
    std::string display_font;
};
FontPaths default_font_paths();

// Adds the faces to the current ImGui context's atlas and sets io.FontDefault
// to the UI face. Call after caliper_host::core_create_ui_context(); the 1.92
// dynamic atlas accepts fonts before or after the renderer backend is up.
// When the in-repo faces cannot be found the app keeps ImGui's built-in font
// for ui and mono and returns false; a missing Display face only leaves
// fonts().display null (the wordmark falls back to the UI face).
bool load_fonts(const FontPaths& paths = default_font_paths());

// The loaded faces; all null until load_fonts() has run.
const Fonts& fonts();

}  // namespace sternbild
