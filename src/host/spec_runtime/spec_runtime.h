#pragma once
// The spec runtime: draws the panels of a spec natively, in the app's own
// look, from the data the builder session resolves (design/spec-format.md,
// design/builder-session.md, Resolving a panel's data).
//
// A caller hands SpecRuntime a spec and a source of resolved panel data, and
// calls draw() on the frame thread inside a window that covers the area. The
// runtime lays the spec's layout tree out as plain rectangles
// (spec_layout.h), resolves each panel's data when the spec or the source's
// revision changes, makes it ready to draw within bounds (spec_prepare.h),
// and draws each panel with its primitive: series, heatmap, points (2-D with
// ImPlot, 3-D with ImPlot3D), histogram, table, image and text. A panel whose
// data is not there shows a short state in its place.
//
// The live views' look carries over: a series' single line and a
// histogram's bars in parchment, a heatmap and an image drawn as a texture
// stretched over index axes in the mono face with a colorbar beside a
// heatmap, points as squares of the live views' sizes, the live views'
// colormaps and their categorical palette for several lines, and the info
// string of a live view's header. A spec panel and a live view of the same
// data therefore read alike.
//
// Seams for the pages that call it:
//   - draw() takes any area, so a page lays its strips out and hands the
//     runtime what is left;
//   - set_preview() swaps one panel's spec fragment and data for a
//     preview's, which is drawn in that panel's rectangle until
//     clear_preview();
//   - set_overlay() is called for every panel after its content, inside the
//     panel's own rectangle, for a prompt row or a diff card.
//
// Threads: everything here runs on the frame thread, where the ImGui context
// is. resolve() of the source runs there too, when the spec or the data
// changed; nothing else is resolved per frame.
#include "spec_runtime/spec_layout.h"
#include "spec_runtime/spec_prepare.h"

#include <caliper_builder/session/resolve.h>
#include <caliper_builder/spec/spec.h>

#include <imgui.h>
#include <imgui_internal.h>   // ImRect

#include <cstdint>
#include <functional>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace caliper_builder {
class Session;
}

namespace caliper_host {

// The type and colours the runtime draws with. house_spec_look() gives the
// Sternbild faces and tokens; the defaults are the same tokens with ImGui's
// current font, for a context without the house fonts.
struct SpecLook {
    ImFont* ui = nullptr;      // titles, tabs, labels, states
    ImFont* mono = nullptr;    // the info strings, ticks, figures
    float ui_size = 15.0f;
    float small_size = 14.0f;
    float mono_size = 14.0f;
    float text_scale = 1.0f;   // the minimums and margins grow with it
    ImU32 ground = IM_COL32(0x18, 0x17, 0x15, 255);
    ImU32 deep = IM_COL32(0x12, 0x11, 0x0f, 255);
    ImU32 plate = IM_COL32(0x20, 0x1f, 0x1c, 255);
    ImU32 plate_hover = IM_COL32(0x2a, 0x29, 0x25, 255);
    ImU32 rule = IM_COL32(0x4b, 0x49, 0x45, 255);
    ImU32 rule_faint = IM_COL32(0x2c, 0x2b, 0x28, 255);
    ImU32 text = IM_COL32(0xe8, 0xe0, 0xd0, 255);
    ImU32 secondary = IM_COL32(0xc1, 0xba, 0xaa, 255);
    ImU32 faint = IM_COL32(0x8f, 0x89, 0x7c, 255);
    ImU32 accent = IM_COL32(0xb3, 0x8b, 0x59, 255);
    ImU32 line = IM_COL32(0xe8, 0xe0, 0xd0, 255);        // one series line, histogram bars (the live views' look.plot)
    ImU32 flat_point = IM_COL32(220, 220, 230, 255);     // points without colour (the live views' kFlat)
};

// The house look: the Sternbild faces from sternbild::fonts() and its tokens.
SpecLook house_spec_look(float text_scale = 1.0f);

// The data palette: the house's class pigments in their fixed order, then
// the live views' categorical palette.
ImU32 spec_palette_color(size_t i);

// One colour key for a spec: every name a series draws gets the next colour
// of the palette the first time it appears, in the order of the spec's
// panels and bindings, and keeps it in every panel, so `loss` is one colour
// in a series and in a table's head.
class SpecColorKey {
public:
    void reset(const caliper_builder::spec::Spec& spec);
    ImU32 color(const std::string& name);              // assigns a new name the next colour
    bool find(const std::string& name, ImU32* out) const;

private:
    std::map<std::string, size_t> index_;
};

// RGBA8 textures the runtime draws heatmaps and images with, as ImGui user
// textures: the renderer's ImGui backend creates, updates and destroys them
// with the font atlas, so the runtime needs no renderer of its own. Frame
// thread only. A released texture is kept until the backend has destroyed
// it; collect() frees those, and the destructor unregisters the rest.
class SpecTextures {
public:
    SpecTextures() = default;
    ~SpecTextures();
    SpecTextures(const SpecTextures&) = delete;
    SpecTextures& operator=(const SpecTextures&) = delete;

    ImTextureData* create(int w, int h, const uint8_t* rgba);
    void release(ImTextureData* tex);
    void collect();
    size_t live() const { return live_.size(); }

private:
    std::vector<ImTextureData*> live_, dying_;
};

// Where one panel is drawn, for an overlay.
struct SpecPanelFrame {
    std::string id;
    ImRect rect;       // the whole panel
    ImRect header;     // its title strip
    ImRect content;    // under the strip, inside the padding
    bool preview = false;
};

// The data of a spec's panels. resolve() is called on the frame thread when
// the spec or revision() changed.
class SpecPanelSource {
public:
    virtual ~SpecPanelSource() = default;
    virtual caliper_builder::PanelData resolve(const caliper_builder::spec::Spec& spec, std::string_view panel_id) = 0;
    // Grows when the data behind the panels may have changed.
    virtual uint64_t revision() const { return 0; }
};

// A source over a builder session: the open spec's panels through
// Session::resolve(), any other spec's through resolve_panel() over the
// session's store with its limits, and the catalog's revision.
class SessionPanelSource : public SpecPanelSource {
public:
    explicit SessionPanelSource(caliper_builder::Session& session, caliper_builder::ResolveLimits limits = {});
    caliper_builder::PanelData resolve(const caliper_builder::spec::Spec& spec, std::string_view panel_id) override;
    uint64_t revision() const override;

private:
    caliper_builder::Session& session_;
    caliper_builder::ResolveLimits limits_;
};

// One panel ready to draw: its data and the texture of a heatmap or image.
struct SpecPanelView {
    PreparedPanel data;
    SpecColorKey* key = nullptr;    // the spec's colour key; null: colours by position
    bool show_title = true;         // false for a tab, whose label carries it
    ImTextureData* texture = nullptr;
    int fit_frames = 2;          // frames that fit the axes to the data
    double x_extent[2] = {0, 0}, y_extent[2] = {0, 0};   // the data's, for log ticks
    uint64_t version = 0;        // grows with each new data
};

// Draws one prepared panel into `rect` of the current window: the title
// strip, then the primitive or the state. `textures` makes the panel's
// texture when it has none. Returns the panel's frame.
SpecPanelFrame draw_spec_panel(SpecPanelView& view, const ImRect& rect, const SpecLook& look, SpecTextures& textures);

class SpecRuntime {
public:
    explicit SpecRuntime(SpecTextures& textures);
    ~SpecRuntime();
    SpecRuntime(const SpecRuntime&) = delete;
    SpecRuntime& operator=(const SpecRuntime&) = delete;

    // The spec to draw. Panels whose spec fragment changed are resolved again
    // on the next draw; the others keep their data.
    void set_spec(const caliper_builder::spec::Spec& spec);
    void clear_spec();
    const std::optional<caliper_builder::spec::Spec>& spec() const { return spec_; }

    // Where the data comes from; null shows every panel as "No data source".
    void set_source(SpecPanelSource* source);
    // Resolve every panel again on the next draw.
    void invalidate();

    void set_limits(const PrepareLimits& limits) { limits_ = limits; invalidate(); }

    // The tab shown in each tabs node (spec_layout.h); the tab strips change it.
    SpecTabChoice& tabs() { return tabs_; }

    // ⌘K seams: a panel drawn from a preview's fragment and data instead of
    // its own, and a callback after each panel's content.
    void set_preview(const caliper_builder::spec::Panel& panel, const caliper_builder::PanelData& data);
    void clear_preview(const std::string& panel_id);
    void set_overlay(std::function<void(const SpecPanelFrame&)> overlay) { overlay_ = std::move(overlay); }

    // Brings the panels up to date with the spec and the source: resolves
    // and prepares the ones that changed. draw() calls it.
    void update();

    // Lays the spec out in `area` and draws it, on the frame thread, inside a
    // window that covers the area.
    void draw(const ImRect& area, const SpecLook& look);

    // The layout of the last draw.
    const SpecLayout& layout() const { return layout_; }
    // A panel's prepared data, or null.
    const PreparedPanel* prepared(const std::string& panel_id) const;

private:
    struct Entry {
        caliper_builder::spec::Panel panel;   // the fragment the data was resolved for
        SpecPanelView view;
        bool stale = true;
    };
    void prepare(Entry& e, const caliper_builder::PanelData& data);
    void draw_tabs(const SpecLayoutTabs& t, const SpecLook& look);

    SpecTextures& textures_;
    std::optional<caliper_builder::spec::Spec> spec_;
    SpecPanelSource* source_ = nullptr;
    uint64_t revision_ = 0;
    PrepareLimits limits_;
    std::map<std::string, Entry> entries_;
    std::map<std::string, Entry> previews_;
    SpecTabChoice tabs_;
    SpecColorKey key_;
    SpecLayout layout_;
    std::function<void(const SpecPanelFrame&)> overlay_;
};

}  // namespace caliper_host
