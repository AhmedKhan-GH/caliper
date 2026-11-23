#pragma once
// ============================================================================
// Font loading for the Sternbild chrome.
//
// Three faces, two families in the UI: a compact sans for labels, a mono for
// data and the command line, and Sternbild Display for the CALIPER wordmark and
// nothing else. The font files and the type sizes come from the Sternbild design
// tokens (cmake/SternbildTokens.cmake); nothing is copied into this repository.
//
//   ui       IBM Plex Sans Condensed Regular   SIL OFL 1.1   design token package
//   mono     Source Code Pro Regular           SIL OFL 1.1   design token package
//   display  Sternbild Display Regular         house-owned   design token package
//
// Dear ImGui 1.92 rasterises on demand at the viewport's framebuffer scale
// (ImGuiBackendFlags_RendererHasTextures on Metal, GL3 and Vulkan), so one
// ImFont per face serves every size via ImGui::PushFont(font, size) and Retina
// needs no extra work.
// ============================================================================

#include <sternbild_tokens.h>

#include <cstddef>
#include <string>

struct ImFont;

namespace sternbild {

struct Fonts {
    ImFont* ui      = nullptr;   // labels, pane headers, menus, prose
    ImFont* mono    = nullptr;   // data cells, identifiers, paths, the command line
    ImFont* display = nullptr;   // the wordmark only; nullptr when the face is absent
    bool ok() const { return ui && mono; }
};

// Logical sizes in px at 1x, the application sizes of the design tokens
// (app.font_size). ImGui sizes a face by its ascender-to-descender height, not
// the em a browser uses, so the tokens already hold the CSS sizes multiplied by
// each face's height ratio.
namespace font_size {
constexpr float ui       = tokens::app::font_size::ui;        // labels, pane headers, buttons, prose
constexpr float ui_small = tokens::app::font_size::ui_small;  // column heads, key hints, status
constexpr float mono     = tokens::app::font_size::mono;      // data cells, identifiers, command line
constexpr float display  = tokens::app::font_size::display;   // CALIPER wordmark
}  // namespace font_size

// Where the faces come from. Resolution order:
//   assets_dir:   $CALIPER_FONTS_DIR, else <exe dir>/assets/fonts (copied by
//                 CMake from the token package), else the package's own font
//                 folder (CALIPER_TOKEN_FONTS_DIR, dev builds).
//   display_font: $CALIPER_DISPLAY_FONT, else SternbildDisplay-Regular.otf in
//                 assets_dir.
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
// fonts().display null (the wordmark falls back to the UI face). An empty
// display_font asks for no Display face, and nothing is reported.
bool load_fonts(const FontPaths& paths = default_font_paths());

// The same faces from memory, for a host without a file system (the
// browser's views, web/wasm, embed the in-repo files in the module). The bytes
// are copied; there is no symbol fallback and no Display face. Sets
// io.FontDefault to the UI face; false (and the built-in font) when a face
// does not load.
bool load_fonts_from_memory(const void* ui_ttf, size_t ui_size, const void* mono_ttf,
                            size_t mono_size);

// The loaded faces; all null until load_fonts() has run.
const Fonts& fonts();

}  // namespace sternbild
