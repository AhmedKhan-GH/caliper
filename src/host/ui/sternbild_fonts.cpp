// ============================================================================
// Font loading for the Sternbild chrome. The faces, their licences and where
// each one lives are listed in sternbild_fonts.h and assets/fonts/README.md.
// ============================================================================

#include "ui/sternbild_fonts.h"

#include "app_paths.h"

#include <imgui.h>
#include <imgui_internal.h>   // ImFileLoadToMemory: IM_ALLOC'd bytes the atlas can own

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <string>
#include <system_error>
#include <vector>

#ifdef __APPLE__
  #include <mach-o/dyld.h>
#elif defined(_WIN32)
  #define WIN32_LEAN_AND_MEAN
  #include <windows.h>
#endif

namespace fs = std::filesystem;

namespace sternbild {

namespace {

constexpr const char* k_ui_file   = "IBMPlexSansCondensed-Regular.ttf";
constexpr const char* k_mono_file = "SourceCodePro-Regular.ttf";

Fonts g_fonts;

// Every directory default_font_paths() tried when none held the UI face, so the
// not-found message can name the whole search rather than an empty path.
std::string g_searched;

bool is_file(const fs::path& path) {
    std::error_code ec;
    return fs::is_regular_file(path, ec);
}

// The same lookup main.cpp uses for <exe dir>/applets.
std::string exe_dir() {
    std::string dir;
#ifdef __APPLE__
    char path_buf[1024];
    uint32_t path_size = sizeof(path_buf);
    if (_NSGetExecutablePath(path_buf, &path_size) == 0)
        dir = fs::path(path_buf).parent_path().string();
#elif defined(_WIN32)
    char path_buf[MAX_PATH];
    if (GetModuleFileNameA(nullptr, path_buf, MAX_PATH) > 0)
        dir = fs::path(path_buf).parent_path().string();
#else
    std::error_code ec;
    dir = fs::read_symlink("/proc/self/exe", ec).parent_path().string();
#endif
    return dir;
}

// Vertical metrics as fractions of the ImGui size. stb_truetype, and so ImGui,
// scales a face so its hhea ascender-to-descender height equals the requested
// size, where a browser scales the em. Zero when a table cannot be read.
struct FaceMetrics {
    float em  = 0.0f;   // head unitsPerEm
    float cap = 0.0f;   // OS/2 sCapHeight (OS/2 version 2 and later)
};

FaceMetrics face_metrics(const unsigned char* p, size_t n) {
    auto u16 = [&](size_t at) -> uint32_t {
        return at + 2 <= n ? uint32_t(p[at]) << 8 | p[at + 1] : 0;
    };
    auto u32 = [&](size_t at) -> uint32_t { return u16(at) << 16 | u16(at + 2); };

    // A collection (Menlo.ttc) starts with its own header; read the first face,
    // which is the one ImGui loads with the default FontNo of 0.
    const size_t dir = u32(0) == 0x74746366 /* ttcf */ ? u32(12) : 0;
    size_t head = 0, hhea = 0, os2 = 0;
    for (uint32_t i = 0, count = u16(dir + 4); i < count; i++) {
        const size_t rec = dir + 12 + 16 * size_t(i);
        if (u32(rec) == 0x68656164 /* head */) head = u32(rec + 8);
        if (u32(rec) == 0x68686561 /* hhea */) hhea = u32(rec + 8);
        if (u32(rec) == 0x4F532F32 /* OS/2 */) os2 = u32(rec + 8);
    }
    FaceMetrics m;
    const int height = hhea ? int16_t(u16(hhea + 4)) - int16_t(u16(hhea + 6)) : 0;
    if (height <= 0) return m;
    if (head) m.em = float(u16(head + 18)) / float(height);
    if (os2 && u16(os2) >= 2) m.cap = float(int16_t(u16(os2 + 88))) / float(height);
    if (m.cap < 0.0f) m.cap = 0.0f;
    return m;
}

// A face's bytes, allocated by ImGui because the atlas takes ownership: 1.92
// keeps source data for the atlas lifetime and frees it with IM_FREE.
struct FaceFile {
    void* data = nullptr;
    int   size = 0;
    FaceMetrics metrics;
};

FaceFile read_face(const std::string& path) {
    FaceFile face;
    if (path.empty()) return face;
    size_t size = 0;
    void* data = ImFileLoadToMemory(path.c_str(), "rb", &size);
    // AddFontFromMemoryTTF asserts on anything this small; treat it as absent.
    if (!data || size <= 100) {
        ImGui::MemFree(data);
        return face;
    }
    face.data = data;
    face.size = int(size);
    face.metrics = face_metrics(static_cast<const unsigned char*>(data), size);
    return face;
}

ImFontConfig face_config() {
    ImFontConfig cfg;
    // Fractional advances keep PushFont(font, size) exact at every size, and
    // snapping would also switch off horizontal oversampling. OversampleH stays
    // at the 1.92 auto value, already 2x below a 36 px raster: every size we
    // draw, at 1x and on Retina. Hinting and glyph ranges stay at defaults;
    // 1.92 loads glyphs on demand.
    cfg.PixelSnapH = false;
    return cfg;
}

// Hands the bytes to the atlas, which owns them from here on (also on failure:
// AddFont's rollback frees them).
ImFont* add_face(ImFontAtlas* atlas, const FaceFile& face, const std::string& path,
                 float size, ImFontConfig cfg) {
    std::snprintf(cfg.Name, sizeof(cfg.Name), "%s",
                  fs::path(path).filename().string().c_str());
    return atlas->AddFontFromMemoryTTF(face.data, face.size, size, &cfg);
}

// Plex Sans Condensed and Source Code Pro lack the macOS key glyphs the landing
// puts in key hints: U+2318 (command), U+23CE (return), U+21E7 (shift). A
// system face merged behind each supplies them. Only macOS is covered; other
// platforms get no fallback and should spell those keys out.
std::string find_symbol_face() {
#ifdef __APPLE__
    for (const char* path : {"/System/Library/Fonts/Apple Symbols.ttf",
                             "/System/Library/Fonts/Supplemental/Apple Symbols.ttf",
                             "/System/Library/Fonts/Menlo.ttc"}) {
        if (is_file(path)) return path;
    }
    std::cerr << "[fonts] neither Apple Symbols nor Menlo found; U+2318, U+23CE"
                 " and U+21E7 will not render" << std::endl;
#endif
    return {};
}

// Merges the symbol face into the font added last, so it must directly follow
// its primary. ImGui searches a font's sources in order, so the primary still
// draws every glyph it has and the fallback only fills the gaps.
void merge_symbols(ImFontAtlas* atlas, float base_size, const FaceMetrics& base,
                   const std::string& path) {
    FaceFile face = read_face(path);
    if (!face.data) return;
    const FaceMetrics& sym = face.metrics;
    ImFontConfig cfg = face_config();
    cfg.MergeMode = true;
    // Size the fallback so its capitals match the primary's. Apple Symbols draws
    // its key glyphs at about its cap height, and matching the em instead left
    // U+23CE some 5 px tall beside 8 px capitals at size 15. The em is the
    // fallback when a face has no cap height; 0 keeps the primary's scale.
    float size = 0.0f;
    if (base.cap > 0.0f && sym.cap > 0.0f)
        size = base_size * base.cap / sym.cap;
    else if (base.em > 0.0f && sym.em > 0.0f)
        size = base_size * base.em / sym.em;
    // No GlyphOffset: ImGui sets every merged glyph on the primary's baseline
    // (the primary's ascent plus the glyph's own baseline-relative box), so the
    // baselines already agree. At size 15 U+2318 ends 0.5 px below it and
    // U+23CE 0.5 px above, by design; an offset (rounded to whole pixels)
    // would move both the same way.
    add_face(atlas, face, path, size, cfg);
}

ImFont* load_face(ImFontAtlas* atlas, const std::string& path, float size,
                  const std::string& symbols = {}) {
    FaceFile face = read_face(path);
    if (!face.data) return nullptr;
    ImFont* font = add_face(atlas, face, path, size, face_config());
    if (font && !symbols.empty()) merge_symbols(atlas, size, face.metrics, symbols);
    return font;
}

}  // namespace

FontPaths default_font_paths() {
    FontPaths paths;

    std::vector<std::string> candidates;
    if (const char* env = std::getenv("CALIPER_FONTS_DIR"); env && *env)
        candidates.emplace_back(env);
    if (const std::string dir = exe_dir(); !dir.empty())
        candidates.push_back((fs::path(dir) / "assets" / "fonts").string());
#ifdef CALIPER_SOURCE_DIR
    candidates.emplace_back(CALIPER_SOURCE_DIR "/assets/fonts");
#endif

    // First directory that really holds the UI face: a stale or half-copied
    // directory earlier in the order must not shadow a good one after it.
    for (const std::string& dir : candidates) {
        if (is_file(fs::path(dir) / k_ui_file)) {
            paths.assets_dir = dir;
            break;
        }
    }
    g_searched.clear();
    if (paths.assets_dir.empty()) {
        for (const std::string& dir : candidates)
            g_searched += (g_searched.empty() ? "" : ", ") + dir;
    }

    if (const char* env = std::getenv("CALIPER_DISPLAY_FONT"); env && *env)
        paths.display_font = env;
    else
        paths.display_font =
            caliper::app_data_path("fonts/SternbildDisplay-Regular.otf");
    return paths;
}

bool load_fonts(const FontPaths& paths) {
    ImGuiIO& io = ImGui::GetIO();
    ImFontAtlas* atlas = io.Fonts;
    g_fonts = Fonts{};

    auto in_assets = [&](const char* file) {
        // An empty directory means none was found; never fall back to the cwd.
        return paths.assets_dir.empty() ? std::string()
                                        : (fs::path(paths.assets_dir) / file).string();
    };
    const std::string symbols = find_symbol_face();

    // UI first: if it is missing, the built-in font must be the atlas's first
    // font, which ImGui uses as the default while io.FontDefault is null.
    g_fonts.ui = load_face(atlas, in_assets(k_ui_file), font_size::ui, symbols);
    if (!g_fonts.ui && atlas->Fonts.empty())
        atlas->AddFontDefault();
    g_fonts.mono = load_face(atlas, in_assets(k_mono_file), font_size::mono, symbols);
    g_fonts.display = load_face(atlas, paths.display_font, font_size::display);

    if (g_fonts.ui) io.FontDefault = g_fonts.ui;

    if (!g_fonts.ok()) {
        std::string missing = g_fonts.ui ? "" : k_ui_file;
        if (!g_fonts.mono)
            missing += (missing.empty() ? "" : ", ") + std::string(k_mono_file);
        std::string where = paths.assets_dir;
        if (where.empty()) where = g_searched.empty() ? "(no directory)" : g_searched;
        std::cerr << "[fonts] " << missing << " not found in " << where
                  << "; using ImGui's built-in font (set CALIPER_FONTS_DIR)"
                  << std::endl;
    }
    if (!g_fonts.display) {
        std::cerr << "[fonts] Sternbild Display not found at "
                  << (paths.display_font.empty() ? "(no path)" : paths.display_font)
                  << "; the wordmark uses the UI face" << std::endl;
    }
    return g_fonts.ok();
}

const Fonts& fonts() { return g_fonts; }

}  // namespace sternbild
