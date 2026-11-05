#include "live/live_trajectories.h"
#include "live/live_canvas.h"
#include "live/live_trace.h"
#include "renderer/host_renderer.h"
#include "ui/sternbild_theme.h"
#include <caliper/services/geometry_v1_1.h>
#include <imgui.h>
#include <algorithm>
#include <cmath>
#include <cstdio>

namespace caliper_live {

using caliper_host::HostRenderer;
using caliper_host::HostGeomDraw;

std::vector<int> lock_layers(const TraceRun& run) {
    const int layers = static_cast<int>(run.layers.size());
    std::vector<int> lock(static_cast<size_t>(run.positions()), -1);
    if (layers == 0 || run.k <= 0) return lock;
    for (int t = 0; t < run.positions(); ++t) {
        const size_t out = static_cast<size_t>(t) * run.k;
        if (out >= run.output.size()) continue;
        const int32_t next = run.output[out];
        int from = -1;
        for (int l = layers - 1; l >= 0; --l) {
            const size_t at = run.at(t, l);
            if (at >= run.jlens.size() || run.jlens[at] != next) break;
            from = l;
        }
        lock[static_cast<size_t>(t)] = from;
    }
    return lock;
}

namespace {

constexpr ImU32 kTeal = IM_COL32(0x3c, 0xa9, 0x9c, 255);
// The plane spans about [-1, 1]; the layers rise through the same height.
constexpr float kHeight = 1.0f;

uint32_t rgba(ImU32 colour, uint8_t alpha) { return (colour & 0x00ffffffu) | (static_cast<uint32_t>(alpha) << 24); }

} // namespace

struct TrajectoryView::Impl {
    GeomCanvas canvas;
    std::vector<int> lens_layers;       // the lens layers, bottom to top
    std::vector<float> plane;           // [token][layer][2]
    int count = 0;                      // tokens placed
    std::vector<int32_t> tokens, reads, next;
    std::vector<int> lock;
    int prompt = 0;
    uint64_t positions = 0, colours = 0, guides = 0;
    int uploaded = -1, coloured_for = -2, hovered = -1;
    bool rebuilt = false;               // the vertices changed since they were uploaded
    std::vector<float> vertices, guide_vertices;

    Impl() {
        canvas.camera.target = {0, 0, 0};
        canvas.camera.radius = 1.3;
        canvas.camera.distance = 3.6;
        canvas.camera.azimuth = 0.9;
        canvas.camera.elevation = 0.25;
    }

    int layers() const { return static_cast<int>(lens_layers.size()); }
    float height(int l) const { return layers() > 1 ? -kHeight + 2 * kHeight * l / (layers() - 1) : 0; }
    void point(int l, int t, float& x, float& y, float& z) const {
        const size_t at = (static_cast<size_t>(t) * layers() + l) * 2;
        x = plane[at]; y = height(l); z = plane[at + 1];
    }

    // Line segments between consecutive layers, per token; a frame of layer ticks beside them.
    void build() {
        vertices.clear();
        for (int t = 0; t < count; ++t)
            for (int l = 0; l + 1 < layers(); ++l) {
                float x, y, z;
                point(l, t, x, y, z); vertices.insert(vertices.end(), {x, y, z});
                point(l + 1, t, x, y, z); vertices.insert(vertices.end(), {x, y, z});
            }
        guide_vertices.clear();
        const float g = -1.35f;
        guide_vertices.insert(guide_vertices.end(), {g, height(0), g, g, height(layers() - 1), g});
        for (int l = 0; l < layers(); ++l)
            guide_vertices.insert(guide_vertices.end(), {g, height(l), g, g + 0.08f, height(l), g});
    }

    // A segment's colour: teal once its token's lens reads the next token for good, bronze for the hovered token,
    // otherwise faint for the prompt and secondary for what the model wrote.
    std::vector<uint32_t> segment_colours() const {
        std::vector<uint32_t> out;
        out.reserve(vertices.size() / 3);
        for (int t = 0; t < count; ++t)
            for (int l = 0; l + 1 < layers(); ++l) {
                uint32_t c;
                if (t == hovered) c = rgba(sternbild::col::bronze, 255);
                else if (lock[t] >= 0 && l + 1 >= lock[t]) c = rgba(kTeal, t < prompt ? 150 : 235);
                else c = t < prompt ? rgba(sternbild::col::faint, 70) : rgba(sternbild::col::secondary, 150);
                out.push_back(c); out.push_back(c);
            }
        return out;
    }

    bool upload() {
        if (uploaded != canvas.generation()) {   // a new renderer: the old imports went with the old one
            positions = colours = guides = 0;
            uploaded = canvas.generation();
            rebuilt = true;
        }
        if (rebuilt) {                           // canvas.upload replaces (and releases) what an id held
            if (!canvas.upload(positions, vertices.data(), vertices.size() * sizeof(float)) ||
                !canvas.upload(guides, guide_vertices.data(), guide_vertices.size() * sizeof(float))) return false;
            rebuilt = false;
            coloured_for = -2;
        }
        if (coloured_for != hovered) {
            const auto c = segment_colours();
            if (!canvas.upload(colours, c.data(), c.size() * sizeof(uint32_t))) return false;
            coloured_for = hovered;
        }
        return true;
    }

    bool render(uint64_t view, const model3d::Matrices& m) {
        if (vertices.empty() || !upload()) return false;
        HostGeomDraw frame;
        frame.pos_alloc = guides;
        frame.vertex_count = guide_vertices.size() / 3;
        frame.topology = CALIPER_GEOM_TOPO_LINES;
        frame.color_mode = CALIPER_GEOM_COLOR_FLAT;
        frame.blend_mode = CALIPER_GEOM_BLEND_OPAQUE;
        frame.flat_rgba = sternbild::col::rule;
        HostGeomDraw paths;
        paths.pos_alloc = positions;
        paths.vertex_count = vertices.size() / 3;
        paths.attr_alloc = colours;
        paths.topology = CALIPER_GEOM_TOPO_LINES;
        paths.color_mode = CALIPER_GEOM_COLOR_VERTEX_RGBA;
        paths.blend_mode = CALIPER_GEOM_BLEND_ALPHA;
        const HostGeomDraw draws[2] = {frame, paths};
        return canvas.renderer()->geom_draw_primitives(view, m.view.data(), m.projection.data(), draws, 2,
                                                       sternbild::col::deep);
    }

    // The token whose path passes nearest the pointer, within a few pixels.
    int nearest(const GeomCanvas::Frame& f) const {
        float best = 6.0f * 6.0f;
        int found = -1;
        for (int t = 0; t < count; ++t) {
            float px = 0, py = 0;
            bool have = false;
            for (int l = 0; l < layers(); ++l) {
                float x, y, z, sx, sy;
                point(l, t, x, y, z);
                if (!GeomCanvas::project(f, x, y, z, sx, sy)) { have = false; continue; }
                if (have) {
                    const float dx = sx - px, dy = sy - py, len = dx * dx + dy * dy;
                    float u = len > 0 ? ((f.mouse_x - px) * dx + (f.mouse_y - py) * dy) / len : 0;
                    u = std::clamp(u, 0.0f, 1.0f);
                    const float ex = px + u * dx - f.mouse_x, ey = py + u * dy - f.mouse_y, d = ex * ex + ey * ey;
                    if (d < best) { best = d; found = t; }
                }
                px = sx; py = sy; have = true;
            }
        }
        return found;
    }
};

TrajectoryView::TrajectoryView() : impl_(std::make_unique<Impl>()) {}
TrajectoryView::~TrajectoryView() = default;

void TrajectoryView::show(const TraceRun& run) {
    Impl& s = *impl_;
    const int count = run.placed() ? run.positions() : 0;
    if (count == s.count && run.tokens == s.tokens && run.layers == s.lens_layers) return;
    s.lens_layers = run.layers;
    s.count = count;
    s.plane.assign(run.plane.begin(), run.plane.begin() + static_cast<std::ptrdiff_t>(count ? run.plane.size() : 0));
    s.tokens = run.tokens;
    s.prompt = run.prompt;
    s.lock = lock_layers(run);
    s.next.assign(static_cast<size_t>(run.positions()), -1);
    for (int t = 0; t < run.positions(); ++t)
        if (static_cast<size_t>(t) * run.k < run.output.size()) s.next[t] = run.output[static_cast<size_t>(t) * run.k];
    s.reads.assign(static_cast<size_t>(run.positions()) * run.layers.size(), -1);
    for (int t = 0; t < run.positions(); ++t)
        for (size_t l = 0; l < run.layers.size(); ++l)
            if (run.at(t, static_cast<int>(l)) < run.jlens.size())
                s.reads[static_cast<size_t>(t) * run.layers.size() + l] = run.jlens[run.at(t, static_cast<int>(l))];
    if (s.hovered >= count) s.hovered = -1;
    s.build();
    s.rebuilt = true;
    s.canvas.invalidate();
}

std::string TrajectoryView::subtitle() const {
    const Impl& s = *impl_;
    if (s.count == 0) return "each token's residual stream, layer by layer";
    int settled = 0;
    for (int l : s.lock) settled += l >= 0;
    char text[160];
    std::snprintf(text, sizeof text, "%d tokens  %d lens layers  %d read their next token before the output",
                  s.count, s.layers(), settled);
    return text;
}

void TrajectoryView::draw(HostRenderer* renderer, const std::function<std::string(int32_t)>& word) {
    Impl& s = *impl_;
    s.canvas.attach(renderer);
    if (s.count == 0) {
        ImGui::PushStyleColor(ImGuiCol_Text, sternbild::col::faint);
        ImGui::TextUnformatted(s.tokens.empty() ? "Generate in the Inference pane to trace a run"
                                                : "This run carries no trajectories");
        ImGui::PopStyleColor();
        return;
    }
    if (!s.canvas.ready()) {
        ImGui::PushStyleColor(ImGuiCol_Text, sternbild::col::faint);
        ImGui::TextUnformatted("This view needs the Metal renderer.");
        ImGui::PopStyleColor();
        return;
    }
    const auto f = s.canvas.draw("##trajectories.view", [&](uint64_t view, const model3d::Matrices& m) { return s.render(view, m); });
    ImDrawList* list = ImGui::GetWindowDrawList();
    // Layer numbers beside the ticks.
    for (int l = 0; l < s.layers(); l += std::max(1, s.layers() / 6)) {
        float sx, sy;
        if (GeomCanvas::project(f, -1.35f, s.height(l), -1.35f, sx, sy)) {
            char label[16];
            std::snprintf(label, sizeof label, "%d", s.lens_layers[l]);
            list->AddText(ImVec2(sx - ImGui::CalcTextSize(label).x - 6, sy - ImGui::GetTextLineHeight() * 0.5f),
                          sternbild::col::faint, label);
        }
    }
    float lx, ly;
    if (GeomCanvas::project(f, -1.35f, s.height(s.layers() - 1) + 0.12f, -1.35f, lx, ly))
        list->AddText(ImVec2(lx - 18, ly - ImGui::GetTextLineHeight()), sternbild::col::faint, "layer");
    const char* hint = "Teal: the lens reads the next token from here on   Left drag rotates, wheel zooms";
    const ImVec2 hint_size = ImGui::CalcTextSize(hint);
    list->AddText(ImVec2(static_cast<float>(f.rect.x + f.rect.width) - hint_size.x - 12,
                         static_cast<float>(f.rect.y + f.rect.height) - hint_size.y - 10), sternbild::col::faint, hint);
    const int hovered = f.hovered ? s.nearest(f) : -1;
    if (hovered != s.hovered) { s.hovered = hovered; s.canvas.invalidate(); }
    if (hovered < 0) return;
    // The hovered token: its text at the top of its path, and every layer's reading.
    float x, y, z, sx, sy;
    s.point(s.layers() - 1, hovered, x, y, z);
    const std::string token = display_word(word(s.tokens[hovered]));
    const std::string next = s.next[hovered] >= 0 ? display_word(word(s.next[hovered])) : std::string();
    if (GeomCanvas::project(f, x, y, z, sx, sy))
        list->AddText(ImVec2(sx + 6, sy - ImGui::GetTextLineHeight()), sternbild::col::parchment, (token + " → " + next).c_str());
    ImGui::BeginTooltip();
    ImGui::Text("\"%s\" → \"%s\"", token.c_str(), next.c_str());
    ImGui::PushStyleColor(ImGuiCol_Text, sternbild::col::secondary);
    if (s.lock[hovered] >= 0) ImGui::Text("read from layer %d of %d", s.lens_layers[s.lock[hovered]], s.lens_layers.back());
    else ImGui::TextUnformatted("not read before the output");
    ImGui::Text("%s token %d", hovered < s.prompt ? "prompt" : "written", hovered);
    ImGui::PopStyleColor();
    ImGui::Separator();
    for (int l = 0; l < s.layers(); ++l) {
        const int32_t id = s.reads[static_cast<size_t>(hovered) * s.layers() + l];
        const bool settled = s.lock[hovered] >= 0 && l >= s.lock[hovered];
        ImGui::PushStyleColor(ImGuiCol_Text, settled ? kTeal : sternbild::col::secondary);
        ImGui::Text("%2d  %s", s.lens_layers[l], id >= 0 ? display_word(word(id)).c_str() : "");
        ImGui::PopStyleColor();
    }
    ImGui::EndTooltip();
}

} // namespace caliper_live
