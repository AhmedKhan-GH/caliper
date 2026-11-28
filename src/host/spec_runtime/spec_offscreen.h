#pragma once
// Drawing a spec, or one panel of it, into an offscreen canvas: the path of
// caliper-spec-shot, of the runtime's GPU tests and of panel.render.
//
// SpecOffscreen owns a renderer's offscreen canvas (Metal on macOS, Vulkan on
// Windows, as `caliper shot` uses) and its own ImGui, ImPlot and ImPlot3D
// contexts in the house style. It opens no window. It needs none of the app's
// page machinery: the seam is HostRenderer's canvas, which `caliper shot`
// also draws through, and nothing of the instrument page.
//
// When another ImGui context is current as it is made (the app's, between
// two frames), its contexts share that context's font atlas, so the house
// faces of sternbild::fonts() draw in both; otherwise it loads the house
// faces itself, and sternbild::fonts() points into its atlas while it lives.
// So without a current context only one canvas may exist at a time; a second
// is refused, and its maker draws with the first. Every call puts the
// caller's contexts back as they were, so the app can render a panel between
// two of its frames on the frame thread.
//
// OffscreenPanelRenderer draws one panel of a spec from resolved data and
// encodes it as PNG: the function behind panel.render, which the app plugs
// into its SessionHost with panel_render_output().
#include "spec_runtime/spec_runtime.h"

#include <caliper_builder/commands/registry.h>

#include <functional>
#include <memory>
#include <string>
#include <vector>

struct ImGuiContext;
struct ImPlotContext;
struct ImPlot3DContext;

namespace caliper_host {

class HostRenderer;

class SpecOffscreen {
public:
    // A canvas of w x h pixels, or null with the reason (no GPU, no canvas).
    static std::unique_ptr<SpecOffscreen> create(int width, int height, float text_scale, std::string* why);
    ~SpecOffscreen();
    SpecOffscreen(const SpecOffscreen&) = delete;
    SpecOffscreen& operator=(const SpecOffscreen&) = delete;

    int width() const { return w_; }
    int height() const { return h_; }
    const char* renderer_name() const;
    // The renderer's device and command queue (id<MTLDevice> and id<MTLCommandQueue> on Metal), for
    // a test that makes its own backend of the graphics layer; null when it has none.
    void* native_device() const;
    void* native_queue() const;
    const SpecLook& look() const { return look_; }
    SpecTextures& textures() { return *textures_; }

    bool resize(int width, int height);

    // Draws `frames` frames, each calling `draw` with the canvas's area inside
    // a window that covers it, and reads the last one back as opaque RGBA8
    // rows, top row first. More than one frame lets plots settle their fit.
    bool render(int frames, const std::function<void(const ImRect& area)>& draw, std::vector<uint8_t>* rgba,
                std::string* why);

    // Runs `fn` with this canvas's contexts current, and the caller's after.
    void with_context(const std::function<void()>& fn);

private:
    SpecOffscreen() = default;
    struct Saved;
    std::unique_ptr<HostRenderer> renderer_;
    ImGuiContext* imgui_ = nullptr;
    ImPlotContext* implot_ = nullptr;
    ImPlot3DContext* implot3d_ = nullptr;
    std::unique_ptr<SpecTextures> textures_;
    SpecLook look_;
    int w_ = 0, h_ = 0;
    bool canvas_ = false;
    bool owns_fonts_ = false;   // loaded the house faces into its own atlas
};

// A PNG image of one panel.
struct PanelPng {
    std::string error;    // empty on success
    std::string png;      // the encoded bytes
    int width = 0, height = 0;
    bool ok() const { return error.empty(); }
};

// What the app's SessionHost::render_panel() calls.
class PanelRenderer {
public:
    virtual ~PanelRenderer() = default;
    virtual PanelPng render_panel(const caliper_builder::spec::Spec& spec, const caliper_builder::PanelData& data,
                                  int width, int height) = 0;
};

// Draws the panel alone over the whole canvas, in its title strip and
// frame, as the runtime draws it in a spec, and encodes the last of three
// frames as PNG. It draws on `canvas` when one is given (which must outlive
// it), else on a canvas of its own, made on the first call; the canvas is
// resized for each call.
class OffscreenPanelRenderer : public PanelRenderer {
public:
    explicit OffscreenPanelRenderer(SpecOffscreen* canvas = nullptr, float text_scale = 1.0f, PrepareLimits limits = {});
    ~OffscreenPanelRenderer() override;
    PanelPng render_panel(const caliper_builder::spec::Spec& spec, const caliper_builder::PanelData& data, int width,
                          int height) override;
    // The session's values and kept steps, which the panel's annotations,
    // cursor line and bound texts read; null reads their defaults.
    void set_instrument_state(caliper_builder::ValueStore* values, caliper_builder::StepHistory* history) {
        values_ = values;
        history_ = history;
    }

private:
    caliper_builder::ValueStore* values_ = nullptr;
    caliper_builder::StepHistory* history_ = nullptr;
    SpecOffscreen* shared_ = nullptr;
    float text_scale_;
    PrepareLimits limits_;
    std::unique_ptr<SpecOffscreen> own_;
};

// The result of panel.render from a PanelPng: {"png_base64", "width",
// "height"}, or panel.render_failed with the reason.
caliper_builder::CommandOutput panel_render_output(const PanelPng& png);

// Standard base64 with padding.
std::string spec_base64(const std::string& bytes);

}  // namespace caliper_host
