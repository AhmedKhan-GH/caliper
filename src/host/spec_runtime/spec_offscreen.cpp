// spec_offscreen.cpp: a spec or one panel drawn into an offscreen canvas
// (spec_offscreen.h).
#include "spec_runtime/spec_offscreen.h"

#include "core_lifecycle.h"
#include "export_service.h"
#include "renderer/host_renderer.h"
#include "ui/sternbild_fonts.h"
#include "ui/sternbild_theme.h"

#include <imgui.h>
#include <imgui_internal.h>
#include <implot.h>
#include <implot3d.h>

#include <algorithm>

namespace caliper_host {

namespace spec = caliper_builder::spec;

namespace {
// The canvas whose atlas sternbild::fonts() points into, when one loaded the
// house faces itself.
SpecOffscreen* g_font_owner = nullptr;
}  // namespace

// The caller's contexts, put back on destruction.
struct SpecOffscreen::Saved {
    ImGuiContext* imgui = ImGui::GetCurrentContext();
    ImPlotContext* implot = ImPlot::GetCurrentContext();
    ImPlot3DContext* implot3d = ImPlot3D::GetCurrentContext();
    ~Saved() {
        ImGui::SetCurrentContext(imgui);
        ImPlot::SetCurrentContext(implot);
        ImPlot3D::SetCurrentContext(implot3d);
    }
};

std::unique_ptr<SpecOffscreen> SpecOffscreen::create(int width, int height, float text_scale, std::string* why) {
    if (width <= 0 || height <= 0 || width > 16384 || height > 16384) {
        if (why) *why = "the canvas size " + std::to_string(width) + "x" + std::to_string(height) + " is out of range";
        return nullptr;
    }
    Saved saved;
    if (!saved.imgui && g_font_owner) {
        if (why) *why = "another offscreen canvas holds the house faces; draw with that one";
        return nullptr;
    }
    std::unique_ptr<SpecOffscreen> c(new SpecOffscreen());
    c->w_ = width;
    c->h_ = height;
    // A shared atlas when another context is current: the house faces of
    // sternbild::fonts() live in it.
    ImFontAtlas* shared = saved.imgui ? ImGui::GetIO().Fonts : nullptr;
    c->imgui_ = ImGui::CreateContext(shared);
    ImGui::SetCurrentContext(c->imgui_);
    c->implot_ = ImPlot::CreateContext();
    ImPlot::SetCurrentContext(c->implot_);
    c->implot3d_ = ImPlot3D::CreateContext();
    ImPlot3D::SetCurrentContext(c->implot3d_);
    ImGuiIO& io = ImGui::GetIO();
    io.IniFilename = nullptr;
    io.LogFilename = nullptr;
    if (!shared) {
        sternbild::FontPaths fonts = sternbild::default_font_paths();
        fonts.display_font.clear();   // the wordmark's face; a spec has no wordmark
        sternbild::load_fonts(fonts);
        c->owns_fonts_ = true;
        g_font_owner = c.get();
    } else if (sternbild::fonts().ui) {
        io.FontDefault = sternbild::fonts().ui;
    }
    sternbild::apply_style();
    c->look_ = house_spec_look(text_scale);

    // The renderer's offscreen canvas: Metal on macOS, Vulkan on Windows. There
    // is no fallback renderer, so a renderer that is not there or cannot start
    // is refused with its reason, as in `caliper shot`.
    std::string refusal;
    c->renderer_ = core_select_renderer(&refusal);
    if (!c->renderer_) {
        if (why) *why = refusal;
        return nullptr;
    }
    if (!c->renderer_->canvas_init(nullptr, HostRenderer::CANVAS_OFFSCREEN, width, height)) {
        if (why)
            *why = renderer_start_refusal(c->renderer_->name(), c->renderer_->start_error()) + " (a " +
                   std::to_string(width) + "x" + std::to_string(height) + " canvas)";
        return nullptr;
    }
    c->canvas_ = true;
    c->textures_ = std::make_unique<SpecTextures>();
    return c;
}

SpecOffscreen::~SpecOffscreen() {
    if (g_font_owner == this) g_font_owner = nullptr;
    if (!imgui_) return;
    Saved saved;
    ImGui::SetCurrentContext(imgui_);
    ImPlot::SetCurrentContext(implot_);
    ImPlot3D::SetCurrentContext(implot3d_);
    // In `caliper shot`'s order: the canvas (its ImGui backend destroys the
    // textures it made), then the textures' records, then the contexts.
    if (canvas_) renderer_->canvas_shutdown();
    textures_.reset();
    renderer_.reset();
    if (implot3d_) ImPlot3D::DestroyContext(implot3d_);
    if (implot_) ImPlot::DestroyContext(implot_);
    ImGui::DestroyContext(imgui_);
    // A context of the caller's that was current is put back by `saved`;
    // ours must not be.
    if (saved.imgui == imgui_) saved.imgui = nullptr;
    if (saved.implot == implot_) saved.implot = nullptr;
    if (saved.implot3d == implot3d_) saved.implot3d = nullptr;
}

const char* SpecOffscreen::renderer_name() const { return renderer_ ? renderer_->name() : ""; }

bool SpecOffscreen::resize(int width, int height) {
    if (width <= 0 || height <= 0 || width > 16384 || height > 16384 || !canvas_) return false;
    if (width == w_ && height == h_) return true;
    Saved saved;
    ImGui::SetCurrentContext(imgui_);
    renderer_->canvas_resize(width, height);
    w_ = width;
    h_ = height;
    return true;
}

void SpecOffscreen::with_context(const std::function<void()>& fn) {
    Saved saved;
    ImGui::SetCurrentContext(imgui_);
    ImPlot::SetCurrentContext(implot_);
    ImPlot3D::SetCurrentContext(implot3d_);
    fn();
}

bool SpecOffscreen::render(int frames, const std::function<void(const ImRect&)>& draw, std::vector<uint8_t>* rgba,
                           std::string* why) {
    if (!canvas_) {
        if (why) *why = "no canvas";
        return false;
    }
    Saved saved;
    ImGui::SetCurrentContext(imgui_);
    ImPlot::SetCurrentContext(implot_);
    ImPlot3D::SetCurrentContext(implot3d_);
    ImGuiIO& io = ImGui::GetIO();
    for (int f = 0; f < std::max(1, frames); ++f) {
        io.DisplaySize = ImVec2((float)w_, (float)h_);
        io.DisplayFramebufferScale = ImVec2(1.0f, 1.0f);
        io.DeltaTime = 1.0f / 60.0f;
        textures_->collect();
        renderer_->canvas_new_frame();
        ImGui::SetNextWindowPos(ImVec2(0, 0));
        ImGui::SetNextWindowSize(io.DisplaySize);
        ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0, 0));
        ImGui::PushStyleVar(ImGuiStyleVar_WindowBorderSize, 0.0f);
        ImGui::Begin("##spec.canvas", nullptr,
                     ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoSavedSettings |
                         ImGuiWindowFlags_NoBringToFrontOnFocus | ImGuiWindowFlags_NoScrollWithMouse);
        ImGui::PopStyleVar(2);
        draw(ImRect(ImVec2(0, 0), io.DisplaySize));
        ImGui::End();
        renderer_->canvas_render();
    }
    if (rgba) {
        rgba->assign((size_t)w_ * (size_t)h_ * 4, 0);
        if (!renderer_->canvas_read_pixels(rgba->data(), w_ * 4)) {
            if (why) *why = std::string("could not read the frame back from the ") + renderer_->name() + " renderer";
            return false;
        }
        for (size_t i = 3; i < rgba->size(); i += 4) (*rgba)[i] = 255;
    }
    return true;
}

// ---- panel.render ---------------------------------------------------------------------

OffscreenPanelRenderer::OffscreenPanelRenderer(SpecOffscreen* canvas, float text_scale, PrepareLimits limits)
    : shared_(canvas), text_scale_(text_scale), limits_(limits) {}

OffscreenPanelRenderer::~OffscreenPanelRenderer() = default;

PanelPng OffscreenPanelRenderer::render_panel(const spec::Spec& s, const caliper_builder::PanelData& data, int width,
                                              int height) {
    PanelPng out;
    const spec::Panel* panel = s.find_panel(data.panel);
    if (!panel) {
        out.error = "The spec has no panel " + data.panel + ".";
        return out;
    }
    std::string why;
    if (!shared_ && !own_) own_ = SpecOffscreen::create(width, height, text_scale_, &why);
    SpecOffscreen* const canvas_ = shared_ ? shared_ : own_.get();
    if (!canvas_) {
        out.error = "The panel cannot be drawn off screen: " + why + ".";
        return out;
    }
    if (!canvas_->resize(width, height)) {
        out.error = "The size " + std::to_string(width) + "x" + std::to_string(height) + " cannot be drawn.";
        return out;
    }
    SpecColorKey key;
    key.reset(s);
    SpecPanelView view;
    view.data = prepare_panel(s, *panel, data, limits_);
    view.key = &key;
    std::vector<uint8_t> rgba;
    const bool drawn = canvas_->render(
        3, [&](const ImRect& area) { draw_spec_panel(view, area, canvas_->look(), canvas_->textures()); }, &rgba, &why);
    canvas_->with_context([&] { canvas_->textures().release(view.texture); });
    if (!drawn) {
        out.error = "The panel could not be drawn: " + why + ".";
        return out;
    }
    if (!export_encode_png(rgba.data(), (uint32_t)width, (uint32_t)height, &out.png)) {
        out.error = "The panel's image could not be encoded as PNG.";
        return out;
    }
    out.width = width;
    out.height = height;
    return out;
}

caliper_builder::CommandOutput panel_render_output(const PanelPng& png) {
    if (!png.ok()) return caliper_builder::CommandOutput::fail(caliper_builder::Error("panel.render_failed", png.error));
    return caliper_builder::CommandOutput::ok(
        {{"png_base64", spec_base64(png.png)}, {"width", png.width}, {"height", png.height}});
}

std::string spec_base64(const std::string& bytes) {
    static const char* const k = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    std::string out;
    out.reserve((bytes.size() + 2) / 3 * 4);
    size_t i = 0;
    for (; i + 3 <= bytes.size(); i += 3) {
        const uint32_t v = (uint32_t)(uint8_t)bytes[i] << 16 | (uint32_t)(uint8_t)bytes[i + 1] << 8 | (uint8_t)bytes[i + 2];
        out += k[v >> 18];
        out += k[(v >> 12) & 63];
        out += k[(v >> 6) & 63];
        out += k[v & 63];
    }
    if (i + 1 == bytes.size()) {
        const uint32_t v = (uint32_t)(uint8_t)bytes[i] << 16;
        out += k[v >> 18];
        out += k[(v >> 12) & 63];
        out += "==";
    } else if (i + 2 == bytes.size()) {
        const uint32_t v = (uint32_t)(uint8_t)bytes[i] << 16 | (uint32_t)(uint8_t)bytes[i + 1] << 8;
        out += k[v >> 18];
        out += k[(v >> 12) & 63];
        out += k[(v >> 6) & 63];
        out += '=';
    }
    return out;
}

}  // namespace caliper_host
