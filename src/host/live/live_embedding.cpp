#include "live/live_embedding.h"
#include "live/live_canvas.h"
#include "live/live_trace.h"
#include "renderer/host_renderer.h"
#include "ui/sternbild_theme.h"
#include <caliper/services/geometry_v1_1.h>
#include <imgui.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>

namespace caliper_live {

using caliper_host::HostRenderer;
using caliper_host::HostGeomDraw;

bool read_embedding_map(int count, const std::vector<uint8_t>& positions, const std::vector<uint8_t>& kinds,
                        size_t kind_count, EmbeddingMap& out) {
    if (count <= 0 || positions.size() != static_cast<size_t>(count) * 3 * 2 || kinds.size() != static_cast<size_t>(count))
        return false;
    out.xyz.resize(static_cast<size_t>(count) * 3);
    for (size_t i = 0; i < out.xyz.size(); ++i) {
        int16_t q;
        std::memcpy(&q, positions.data() + i * 2, 2);   // little-endian, as the host sends it
        out.xyz[i] = q / 32767.0f;
    }
    for (uint8_t k : kinds)
        if (k >= kind_count) return false;
    out.kinds = kinds;
    return true;
}

namespace {

// One bright, saturated hue per kind, in the host's KIND_NAMES order: words, word pieces, punctuation, digits,
// CJK, other scripts, whitespace, code and mixed. The largest neighbouring regions take the hues furthest apart.
constexpr ImU32 kKindColours[] = {
    IM_COL32(0x4d, 0xa3, 0xff, 255), IM_COL32(0xff, 0x7a, 0x33, 255), IM_COL32(0xb4, 0x8c, 0xff, 255),
    IM_COL32(0x5e, 0xe0, 0x5a, 255), IM_COL32(0xff, 0x4f, 0xa3, 255), IM_COL32(0x26, 0xd9, 0xa0, 255),
    IM_COL32(0xff, 0x6b, 0x6b, 255), IM_COL32(0xff, 0xc2, 0x1a, 255),
};
constexpr size_t kKinds = sizeof kKindColours / sizeof kKindColours[0];

// The camera's starting distance; points are kPointPx there and grow as the camera closes in.
constexpr double kHomeDistance = 2.4;
constexpr float kPointPx = 2.0f;

} // namespace

struct EmbeddingView::Impl {
    GeomCanvas canvas;
    EmbeddingMap map;
    std::vector<size_t> counts, first;  // per kind: its rows, and where they start in the uploaded buffer
    bool hidden[kKinds] = {};
    std::string status, error;
    uint64_t positions = 0;
    int uploaded = -1;                  // canvas generation the buffer was uploaded under
    // The last hover search: the frame it ran on and its answer.
    float hover_x = -1, hover_y = -1;
    model3d::Camera hover_camera;
    int hovered = -1;

    Impl() {
        canvas.camera.target = {0, 0, 0};
        canvas.camera.radius = 1.0;
        canvas.camera.distance = kHomeDistance;
        canvas.camera.azimuth = 0.7;
        canvas.camera.elevation = 0.3;
    }

    float point_px() const {
        return std::clamp(kPointPx * static_cast<float>(kHomeDistance / std::max(canvas.camera.distance, 1e-3)), 1.5f, 16.0f);
    }

    // The rows grouped by kind, so each kind draws as one range in its own colour and a hidden kind is skipped.
    bool upload() {
        if (map.rows() == 0) return false;
        if (uploaded == canvas.generation()) return true;
        first.assign(counts.size(), 0);
        for (size_t k = 1; k < counts.size(); ++k) first[k] = first[k - 1] + counts[k - 1];
        std::vector<size_t> next = first;
        std::vector<float> grouped(map.xyz.size());
        for (size_t i = 0; i < map.rows(); ++i)
            std::memcpy(&grouped[next[map.kinds[i]]++ * 3], &map.xyz[i * 3], 3 * sizeof(float));
        positions = 0;
        if (!canvas.upload(positions, grouped.data(), grouped.size() * sizeof(float))) return false;
        uploaded = canvas.generation();
        return true;
    }

    bool render(uint64_t view, const model3d::Matrices& m) {
        HostRenderer* renderer = canvas.renderer();
        if (!upload()) return false;
        std::vector<HostGeomDraw> draws;
        for (size_t k = 0; k < counts.size() && k < kKinds; ++k) {
            if (hidden[k] || counts[k] == 0) continue;
            HostGeomDraw points;
            points.pos_alloc = positions;
            points.pos_offset = first[k] * 3 * sizeof(float);
            points.vertex_count = counts[k];
            points.topology = CALIPER_GEOM_TOPO_POINTS;
            points.color_mode = CALIPER_GEOM_COLOR_FLAT;
            points.flat_rgba = kKindColours[k];
            points.shade_mode = CALIPER_GEOM_SHADE_UNLIT;
            points.blend_mode = CALIPER_GEOM_BLEND_OPAQUE;
            points.depth_flags = CALIPER_GEOM_DEPTH_TEST | CALIPER_GEOM_DEPTH_WRITE;   // the nearest point shows
            points.size_px = point_px();
            draws.push_back(points);
        }
        return renderer->geom_draw_primitives(view, m.view.data(), m.projection.data(), draws.data(), draws.size(),
                                              sternbild::col::deep);
    }

    // The visible row nearest the pointer within a few pixels, searched again only when the pointer or camera moved.
    int nearest(const GeomCanvas::Frame& f) {
        const model3d::Camera& c = canvas.camera;
        if (f.mouse_x == hover_x && f.mouse_y == hover_y && c.azimuth == hover_camera.azimuth &&
            c.elevation == hover_camera.elevation && c.distance == hover_camera.distance &&
            c.target.x == hover_camera.target.x && c.target.y == hover_camera.target.y && c.target.z == hover_camera.target.z)
            return hovered;
        hover_x = f.mouse_x; hover_y = f.mouse_y; hover_camera = c;
        const float reach = std::max(7.0f, point_px());
        float best = reach * reach;
        hovered = -1;
        for (size_t i = 0; i < map.rows(); ++i) {
            if (hidden[map.kinds[i]]) continue;
            float sx, sy;
            if (!GeomCanvas::project(f, map.xyz[i * 3], map.xyz[i * 3 + 1], map.xyz[i * 3 + 2], sx, sy)) continue;
            const float dx = sx - f.mouse_x, dy = sy - f.mouse_y, d = dx * dx + dy * dy;
            if (d < best) { best = d; hovered = static_cast<int>(i); }
        }
        return hovered;
    }

    // The legend: each kind's pigment, name and count; a click hides or shows the kind. True while the pointer
    // is on it, so the view under it does not also report a hover.
    bool legend(const GeomCanvas::Frame& f) {
        bool over_any = false;
        ImDrawList* list = ImGui::GetWindowDrawList();
        const float line = ImGui::GetTextLineHeight() + 6;
        float y = static_cast<float>(f.rect.y) + 10;
        const float x = static_cast<float>(f.rect.x) + 12;
        const size_t shown = std::min(map.kind_names.size(), kKinds);
        std::vector<std::string> labels(shown);
        float widest = 0;
        for (size_t k = 0; k < shown; ++k) {
            char label[96];
            std::snprintf(label, sizeof label, "%s  %zu", map.kind_names[k].c_str(), counts[k]);
            labels[k] = label;
            widest = std::max(widest, ImGui::CalcTextSize(label).x);
        }
        // A backing so the legend stays legible over points the camera has closed in on.
        list->AddRectFilled(ImVec2(x - 8, y - 6), ImVec2(x + widest + 26, y + line * shown + 4),
                            (sternbild::col::deep & 0x00ffffffu) | (0xd8u << 24));
        for (size_t k = 0; k < shown; ++k) {
            const char* label = labels[k].c_str();
            const ImVec2 size = ImGui::CalcTextSize(label);
            ImGui::SetCursorScreenPos(ImVec2(x, y));
            ImGui::PushID(static_cast<int>(k));
            if (ImGui::InvisibleButton("kind", ImVec2(size.x + 18, line))) { hidden[k] = !hidden[k]; canvas.invalidate(); }
            const bool over = ImGui::IsItemHovered();
            over_any = over_any || over;
            ImGui::PopID();
            const ImU32 ink = hidden[k] ? sternbild::col::faint : over ? sternbild::col::parchment : sternbild::col::secondary;
            const ImVec2 dot(x + 4, y + line * 0.5f - 1);
            if (hidden[k]) list->AddCircle(dot, 3.5f, kKindColours[k], 0, 1.2f);
            else list->AddCircleFilled(dot, 3.5f, kKindColours[k]);
            list->AddText(ImVec2(x + 14, y + 2), ink, label);
            y += line;
        }
        return over_any;
    }
};

EmbeddingView::EmbeddingView() : impl_(std::make_unique<Impl>()) {}
EmbeddingView::~EmbeddingView() = default;

void EmbeddingView::set(EmbeddingMap map) {
    impl_->map = std::move(map);
    impl_->counts.assign(impl_->map.kind_names.size(), 0);
    for (uint8_t k : impl_->map.kinds) if (k < impl_->counts.size()) ++impl_->counts[k];
    impl_->uploaded = -1;
    impl_->status.clear();
    impl_->error.clear();
    impl_->hover_x = -1;
    impl_->canvas.invalidate();
}
void EmbeddingView::status(const std::string& text) { impl_->status = text; }
void EmbeddingView::failed(const std::string& error) { impl_->error = error; impl_->status.clear(); }
bool EmbeddingView::loaded() const { return impl_->map.rows() > 0; }

std::string EmbeddingView::subtitle() const {
    if (!impl_->error.empty()) return impl_->error;
    if (impl_->map.rows() == 0) return impl_->status.empty() ? "the embedding table, laid out in 3-D" : impl_->status;
    char text[128];
    std::snprintf(text, sizeof text, "%zu tokens  %s of the embedding rows", impl_->map.rows(),
                  impl_->map.method == "umap" ? "UMAP" : "principal components");
    return text;
}

int EmbeddingView::draw(HostRenderer* renderer, const std::map<int32_t, std::string>& words) {
    Impl& s = *impl_;
    s.canvas.attach(renderer);
    if (s.map.rows() == 0) {
        ImGui::PushStyleColor(ImGuiCol_Text, sternbild::col::faint);
        ImGui::TextUnformatted(!s.error.empty() ? s.error.c_str() : !s.status.empty() ? s.status.c_str()
                                                                     : "Waiting for the map from the script");
        ImGui::PopStyleColor();
        return -1;
    }
    if (!s.canvas.ready()) {
        ImGui::PushStyleColor(ImGuiCol_Text, sternbild::col::faint);
        ImGui::TextUnformatted("This view needs the Metal renderer.");
        ImGui::PopStyleColor();
        return -1;
    }
    const auto f = s.canvas.draw("##embedding.view", [&](uint64_t view, const model3d::Matrices& m) { return s.render(view, m); });
    const bool on_legend = s.legend(f);
    ImDrawList* list = ImGui::GetWindowDrawList();
    const char* hint = "Left drag rotates, right drag moves, wheel zooms";
    const ImVec2 hint_size = ImGui::CalcTextSize(hint);
    list->AddText(ImVec2(static_cast<float>(f.rect.x + f.rect.width) - hint_size.x - 12,
                         static_cast<float>(f.rect.y + f.rect.height) - hint_size.y - 10), sternbild::col::faint, hint);
    int wanted = -1;
    const int row = f.hovered && !on_legend ? s.nearest(f) : -1;
    if (f.hovered && row >= 0) {
        float sx, sy;
        if (GeomCanvas::project(f, s.map.xyz[row * 3], s.map.xyz[row * 3 + 1], s.map.xyz[row * 3 + 2], sx, sy))
            list->AddCircle(ImVec2(sx, sy), 5.0f, sternbild::col::bronze, 0, 1.5f);
        const auto word = words.find(row);
        if (word == words.end()) wanted = row;
        ImGui::BeginTooltip();
        ImGui::TextUnformatted(word == words.end() ? "…" : ("\"" + display_word(word->second) + "\"").c_str());
        ImGui::PushStyleColor(ImGuiCol_Text, sternbild::col::secondary);
        const uint8_t k = s.map.kinds[row];
        ImGui::Text("token %d  %s", row, k < s.map.kind_names.size() ? s.map.kind_names[k].c_str() : "");
        ImGui::PopStyleColor();
        ImGui::EndTooltip();
    }
    return wanted;
}

} // namespace caliper_live
