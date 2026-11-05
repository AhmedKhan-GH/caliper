#include "live/live_model3d.h"

#include "live/live_views_state.h"
#include "live/model3d_presentation.h"
#include "renderer/host_renderer.h"
#include "tensor_bridge.h"
#include "ui/sternbild_theme.h"

#include <caliper/services/geometry_v1_2.h>
#include <imgui.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <map>
#include <utility>
#include <vector>

namespace caliper_live {
namespace {
using caliper_host::HostGeomDraw;
using caliper_host::HostRenderer;
using model3d::Panel;
using model3d::Rect;
using model3d::Vec3;

// One static unit quad serves every tensor. Row zero of a thumbnail is its
// upper edge; Metal and ImGui texture coordinates both start at that edge.
constexpr float positions[] = {
    -0.5f, 0.5f, 0,  0.5f, 0.5f, 0, -0.5f,-0.5f, 0,
    -0.5f,-0.5f, 0,  0.5f, 0.5f, 0,  0.5f,-0.5f, 0,
    -0.5f, 0.5f, 0,  0.5f, 0.5f, 0,  0.5f,-0.5f, 0,
    -0.5f,-0.5f, 0, -0.5f, 0.5f, 0,
};
constexpr float coordinates[] = {
    0, 0, 1, 0, 0, 1,
    0, 1, 1, 0, 1, 1,
};
constexpr uint64_t outline_offset = 6 * 3 * sizeof(float);
constexpr size_t max_tensors = 10000;
constexpr int max_thumbnail_side = 128;
constexpr int max_sample_side = 2048;
constexpr int max_target_side = 8192;
// The scene renders at twice the drawn size and is filtered down when drawn
// (2x2 supersampling), which smooths panel edges, frames and fine texture.
constexpr float supersample = 2.0f;
constexpr int resize_tolerance = 8;

bool same_layout(const std::vector<ModelTensor>& a, const std::vector<ModelTensor>& b) {
    if (a.size() != b.size()) return false;
    for (size_t i = 0; i < a.size(); ++i) {
        const auto& x = a[i];
        const auto& y = b[i];
        if (x.name != y.name || x.shape != y.shape || x.rows != y.rows || x.cols != y.cols ||
            x.group != y.group || x.layer_type != y.layer_type || x.role != y.role ||
            x.layer_index != y.layer_index || x.tied != y.tied) return false;
    }
    return true;
}

void transform(HostGeomDraw& draw, Vec3 center, double width, double height) {
    draw.model[0] = static_cast<float>(width);
    draw.model[5] = static_cast<float>(height);
    draw.model[12] = static_cast<float>(center.x);
    draw.model[13] = static_cast<float>(center.y);
    draw.model[14] = static_cast<float>(center.z);
}

std::string dimensions(const ModelTensor& tensor) {
    if (!tensor.shape_label.empty()) return tensor.shape_label;
    std::string label = "[";
    for (int64_t d : tensor.shape) {
        if (label.size() > 1) label += ", ";
        label += std::to_string(d);
    }
    return label + "]";
}

const char* attention_label(const std::string& type) {
    if (type == "linear_attention" || type == "linear") return "Linear";
    if (type == "full_attention" || type == "full") return "Full";
    return "";
}

bool full_attention(const std::string& type) {
    return type == "full_attention" || type == "full";
}

bool inside(Rect rect, double x, double y) {
    return rect.width > 0 && rect.height > 0 && x >= rect.x && y >= rect.y &&
           x < rect.x + rect.width && y < rect.y + rect.height;
}

std::string group_label(const model3d::Group& group) {
    if (group.layer_index < 0) return group.label;
    char label[48];
    std::snprintf(label, sizeof label, "Layer %02d%s", group.layer_index,
                  full_attention(group.layer_type) ? "  FULL" : "");
    return label;
}

} // namespace

bool model3d::thumbnail_atlas(const Tensor& tensor, int rows, int cols,
                              const std::vector<float>& values, ThumbnailAtlas* output) {
    if (!output || rows < 1 || cols < 1 || rows > max_sample_side || cols > max_thumbnail_side ||
        static_cast<size_t>(rows) * cols > max_thumbnail_side * max_thumbnail_side ||
        values.size() != static_cast<size_t>(rows) * cols ||
        !std::all_of(values.begin(), values.end(), [](float v) { return std::isfinite(v); })) return false;
    ThumbnailAtlas atlas;
    if (tensor.shape.size() == 1) {
        if (values.size() > max_thumbnail_side) return false;
        atlas.width = static_cast<int>(values.size()); atlas.height = 1; atlas.values = values;
    } else {
        const int blocks = model3d::row_blocks(tensor);
        if (blocks < 1 || blocks > max_thumbnail_side || (blocks == 1 && rows > max_thumbnail_side)) return false;
        const int block_width = std::min(cols, max_thumbnail_side / blocks);
        atlas.width = block_width * blocks;
        atlas.height = std::min(max_thumbnail_side, (rows + blocks - 1) / blocks);
        atlas.values.resize(static_cast<size_t>(atlas.width) * atlas.height);
        for (int y = 0; y < atlas.height; ++y) for (int x = 0; x < atlas.width; ++x) {
            const int block = x / block_width;
            const int source_row = std::min(rows - 1,
                static_cast<int>((static_cast<int64_t>(2 * (block * atlas.height + y) + 1) * rows) /
                                 (2 * blocks * atlas.height)));
            const int source_col = std::min(cols - 1,
                ((2 * (x % block_width) + 1) * cols) / (2 * block_width));
            atlas.values[static_cast<size_t>(y) * atlas.width + x] = values[static_cast<size_t>(source_row) * cols + source_col];
        }
    }
    *output = std::move(atlas);
    return true;
}

std::array<model3d::FrameBand, 4> model3d::frame_bands(const Panel& panel, const Camera& camera,
                                                     Rect viewport, double pixel_width) {
    std::array<FrameBand, 4> result{};
    if (!(panel.width > 0 && panel.height > 0 && viewport.width > 0 && viewport.height > 0 && pixel_width > 0) ||
        !std::isfinite(panel.width) || !std::isfinite(panel.height) || !std::isfinite(pixel_width) ||
        !std::isfinite(panel.center.x) || !std::isfinite(panel.center.y) || !std::isfinite(panel.center.z)) return result;
    const double aspect = viewport.width / viewport.height;
    const auto derivative = [&](Vec3 center, bool x_axis) {
        const double probe = std::max(1e-8, (x_axis ? panel.width : panel.height) * .001);
        Vec3 a = center, b = center;
        if (x_axis) { a.x -= probe; b.x += probe; }
        else { a.y -= probe; b.y += probe; }
        const auto pa = model3d::project(a, camera, aspect, viewport);
        const auto pb = model3d::project(b, camera, aspect, viewport);
        return std::array<double, 2>{(pb.x - pa.x) / (2 * probe), (pb.y - pa.y) / (2 * probe)};
    };
    const auto thickness = [&](Vec3 center, bool vertical) {
        const auto dx = derivative(center, true), dy = derivative(center, false);
        const double tangent = vertical ? std::hypot(dy[0], dy[1]) : std::hypot(dx[0], dx[1]);
        const double scale = tangent > 1e-10 ? std::fabs(dx[0] * dy[1] - dx[1] * dy[0]) / tangent : 0;
        const double extent = vertical ? panel.width : panel.height;
        const double wanted = std::isfinite(scale) && scale > 1e-10 ? pixel_width / scale : extent * .1;
        return std::clamp(wanted, extent * 1e-6, extent * .2);
    };
    const double left = panel.center.x - panel.width * .5, right = panel.center.x + panel.width * .5;
    const double top = panel.center.y + panel.height * .5, bottom = panel.center.y - panel.height * .5;
    const double tx0 = thickness({left, panel.center.y, panel.center.z}, true);
    const double tx1 = thickness({right, panel.center.y, panel.center.z}, true);
    const double ty0 = thickness({panel.center.x, top, panel.center.z}, false);
    const double ty1 = thickness({panel.center.x, bottom, panel.center.z}, false);
    result[0] = {{left + tx0 * .5, panel.center.y, panel.center.z}, tx0, panel.height};
    result[1] = {{right - tx1 * .5, panel.center.y, panel.center.z}, tx1, panel.height};
    result[2] = {{panel.center.x, top - ty0 * .5, panel.center.z}, panel.width, ty0};
    result[3] = {{panel.center.x, bottom + ty1 * .5, panel.center.z}, panel.width, ty1};
    return result;
}

struct Model3D::Impl {
    struct Thumbnail {
        std::vector<uint8_t> rgba;
        double minimum = 0, maximum = 0;
        uint64_t revision = 0, texture = 0;
        int width = 0, height = 0;
        bool requested = false, failed = false, upload = false;
    };

    std::vector<ModelTensor> tensors;
    std::map<std::string, size_t> indices;
    std::map<std::string, Thumbnail> thumbnails;
    model3d::Scene scene;
    model3d::Camera camera;
    Rect viewport, label_column;
    enum class Framing { Fit, Focus, Manual };
    Framing framing = Framing::Fit;
    std::string focused;
    double framed_aspect = 0, rendered_aspect = 0;
    bool framing_pending = true;
    std::vector<float> fold_positions;
    HostRenderer* renderer = nullptr;
    uint64_t position_buffer = 0, uv_buffer = 0, fold_buffer = 0, target = 0;
    int target_width = 0, target_height = 0;
    bool visible = false, dirty = true, folds_dirty = true, dragged = false, click_armed = false;
    unsigned drag_armed = 0; // Left and middle presses must originate in the scene.
    bool failed = false, imports_failed = false;
    // The axis navigator: an eased swing to an axis view, and whether a press on it became a drag.
    bool snapping = false, axes_dragged = false;
    double snap_azimuth = 0, snap_elevation = 0;
    std::string selected, output_label = "Output (tied)";

    ~Impl() { release_renderer(); }

    void release_renderer() {
        if (renderer) {
            if (target) renderer->tex_release(target);
            if (position_buffer) renderer->release_external_allocation(position_buffer);
            if (uv_buffer) renderer->release_external_allocation(uv_buffer);
            if (fold_buffer) renderer->release_external_allocation(fold_buffer);
            for (auto& entry : thumbnails) {
                if (entry.second.texture) renderer->tex_release(entry.second.texture);
                entry.second.texture = 0;
                entry.second.upload = !entry.second.rgba.empty();
            }
        }
        renderer = nullptr;
        position_buffer = uv_buffer = fold_buffer = target = 0;
        target_width = target_height = 0;
        failed = imports_failed = false;
        folds_dirty = true;
        rendered_aspect = 0;
    }

    bool bind_renderer(HostRenderer* next) {
        if (renderer != next) {
            release_renderer();
            renderer = next;
            dirty = true;
        }
        if (!renderer || !renderer->supports_geometry_primitives() ||
            !renderer->supports_geometry_textured() || !renderer->supports_external_import()) return false;
        if (renderer->interop_device() != CALIPER_DEV_METAL) return false;
        if (imports_failed) return false;
        if (!position_buffer || !uv_buffer) {
            imports_failed = true;
            void* pos = caliper_live_views::metal_new_shared_buffer(renderer->native_device(), positions, sizeof positions);
            void* uv = caliper_live_views::metal_new_shared_buffer(renderer->native_device(), coordinates, sizeof coordinates);
            if (pos && uv) {
                position_buffer = renderer->import_external_allocation(pos, sizeof positions, CALIPER_ALLOC_HANDLE_MTLBUFFER);
                uv_buffer = renderer->import_external_allocation(uv, sizeof coordinates, CALIPER_ALLOC_HANDLE_MTLBUFFER);
            }
            // The renderer retains successful imports, including in-flight draws.
            caliper_live_views::metal_release(pos);
            caliper_live_views::metal_release(uv);
            if (!position_buffer || !uv_buffer) {
                if (position_buffer) renderer->release_external_allocation(position_buffer);
                if (uv_buffer) renderer->release_external_allocation(uv_buffer);
                position_buffer = uv_buffer = 0;
                return false;
            }
            imports_failed = false;
        }
        if (folds_dirty) {
            if (fold_buffer) renderer->release_external_allocation(fold_buffer);
            fold_buffer = 0;
            folds_dirty = false;
            if (!fold_positions.empty()) {
                const size_t bytes = fold_positions.size() * sizeof(float);
                void* buffer = caliper_live_views::metal_new_shared_buffer(renderer->native_device(), fold_positions.data(), bytes);
                if (buffer) fold_buffer = renderer->import_external_allocation(buffer, bytes, CALIPER_ALLOC_HANDLE_MTLBUFFER);
                caliper_live_views::metal_release(buffer);
                if (!fold_buffer) { imports_failed = true; return false; }
            }
        }
        return true;
    }

    void set_tensors(const std::vector<ModelTensor>& next, const std::string& label) {
        if (next.size() > max_tensors) return;
        const bool changed = !same_layout(tensors, next) || output_label != label;
        std::map<std::string, Thumbnail> retained;
        for (const auto& tensor : next) {
            auto previous = thumbnails.find(tensor.name);
            if (previous == thumbnails.end()) {
                retained.emplace(tensor.name, Thumbnail{});
                continue;
            }
            Thumbnail item = std::move(previous->second);
            if (tensor.revision != item.revision && !item.rgba.empty()) {
                if (item.texture && renderer) renderer->tex_release(item.texture);
                item = Thumbnail{};
            }
            if (item.rgba.empty()) {
                // A refreshed manifest follows any queued list request or a
                // new connection. No thumbnail request from the old manifest
                // remains active in the inspector's serial scheduler.
                item.requested = false;
                item.failed = false;
                item.upload = false;
            }
            retained.emplace(tensor.name, std::move(item));
        }
        if (renderer) {
            for (const auto& entry : thumbnails) {
                if (!retained.count(entry.first) && entry.second.texture) renderer->tex_release(entry.second.texture);
            }
        }
        thumbnails = std::move(retained);
        tensors = next;
        indices.clear();
        for (size_t i = 0; i < tensors.size(); ++i) indices.emplace(tensors[i].name, i);
        output_label = label;
        if (changed) {
            scene = model3d::layout(tensors, output_label);
            fold_positions.clear();
            for (const Panel& panel : scene.panels) {
                for (int block = 1; block < panel.row_blocks; ++block) {
                    const float x = static_cast<float>(panel.center.x - panel.width * .5 + panel.width * block / panel.row_blocks);
                    fold_positions.insert(fold_positions.end(), {x, static_cast<float>(panel.center.y + panel.height * .5), static_cast<float>(panel.center.z),
                        x, static_cast<float>(panel.center.y - panel.height * .5), static_cast<float>(panel.center.z)});
                }
            }
            folds_dirty = true;
            imports_failed = false;
            framing = Framing::Fit;
            focused.clear();
            framing_pending = true;
        }
        dirty = true;
    }

    bool set_thumbnail(const std::string& name, int rows, int cols,
                       const std::vector<float>& values, double minimum, double maximum, uint64_t revision) {
        const auto found = indices.find(name);
        if (found == indices.end() || !std::isfinite(minimum) || !std::isfinite(maximum) || minimum > maximum) return false;
        const ModelTensor& tensor = tensors[found->second];
        model3d::ThumbnailAtlas atlas;
        if (!model3d::thumbnail_atlas(tensor, rows, cols, values, &atlas)) return false;
        Thumbnail& item = thumbnails[name];
        if (revision < tensor.revision) {
            item.requested = false;
            return false;
        }
        item.minimum = minimum;
        item.maximum = maximum;
        item.revision = revision;
        item.width = atlas.width; item.height = atlas.height;
        item.rgba.resize(atlas.values.size() * 4);
        float lo = static_cast<float>(minimum), hi = static_cast<float>(maximum);
        if (lo == hi) { lo -= 0.5f; hi += 0.5f; }
        caliper_host::map_f32_to_rgba8(atlas.values.data(), item.width, item.height,
            caliper_host::colormap_lut(CALIPER_CMAP_VIRIDIS), lo, hi, item.rgba.data());
        item.requested = true;
        item.failed = false;
        item.upload = true;
        dirty = true;
        return true;
    }

    void invalidate(const std::string& name, bool edited) {
        const auto found = indices.find(name);
        if (found == indices.end()) return;
        tensors[found->second].edited = edited;
        Thumbnail& item = thumbnails[name];
        item.rgba.clear();
        item.requested = item.failed = item.upload = false;
        dirty = true;
    }

    std::string next_thumbnail() {
        if (!visible) return {};
        const auto request = [&](const std::string& name) {
            auto found = thumbnails.find(name);
            if (found == thumbnails.end() || found->second.requested || found->second.failed ||
                !found->second.rgba.empty()) return false;
            found->second.requested = true;
            return true;
        };
        if (!selected.empty() && request(selected)) return selected;
        for (const auto& tensor : tensors) if (request(tensor.name)) return tensor.name;
        return {};
    }

    HostGeomDraw base_draw(Vec3 center, double width, double height, bool outline, uint32_t color) const {
        HostGeomDraw draw;
        draw.pos_alloc = position_buffer;
        draw.pos_offset = outline ? outline_offset : 0;
        draw.vertex_count = outline ? 5 : 6;
        draw.topology = outline ? CALIPER_GEOM_TOPO_LINE_STRIP : CALIPER_GEOM_TOPO_TRIANGLES;
        draw.color_mode = CALIPER_GEOM_COLOR_FLAT;
        draw.shade_mode = CALIPER_GEOM_SHADE_UNLIT;
        draw.blend_mode = CALIPER_GEOM_BLEND_OPAQUE;
        draw.depth_flags = CALIPER_GEOM_DEPTH_TEST | (outline ? 0u : CALIPER_GEOM_DEPTH_WRITE);
        draw.flat_rgba = color;
        draw.size_px = 1;
        transform(draw, center, width, height);
        return draw;
    }

    void upload_thumbnails() {
        for (auto& entry : thumbnails) {
            Thumbnail& item = entry.second;
            if (!item.upload || item.rgba.empty()) continue;
            // Mipmapped, so a panel seen from afar is minified without shimmer.
            const uint64_t texture = renderer->tex_create_rgba8_mipmapped(item.width, item.height);
            if (!texture || !renderer->tex_upload_rgba8(texture, item.rgba.data(), item.width, item.height)) {
                if (texture) renderer->tex_release(texture);
                item.upload = false;
                item.failed = true;
                continue;
            }
            renderer->tex_generate_mipmaps(texture);
            // A replacement texture keeps an earlier command buffer's retained
            // sample immutable; do not overwrite pixels still in flight.
            if (item.texture) renderer->tex_release(item.texture);
            item.texture = texture;
            item.upload = false;
        }
    }

    bool render(int width, int height, double aspect) {
        if (!target || std::abs(width - target_width) > resize_tolerance ||
            std::abs(height - target_height) > resize_tolerance) {
            const uint64_t replacement = renderer->geom_create_view_ex(width, height, CALIPER_GEOM_VIEW_DEPTH);
            if (!replacement) return false;
            if (target) renderer->tex_release(target);
            target = replacement;
            target_width = width;
            target_height = height;
            dirty = true;
        }
        if (std::fabs(aspect - rendered_aspect) > 1e-8) dirty = true;
        if (!dirty) return !failed;
        upload_thumbnails();
        std::vector<HostGeomDraw> draws;
        draws.reserve(scene.panels.size() * 2 + scene.groups.size() * 6);
        const auto strong_frame = [&](const Panel& panel, uint32_t color, double pixels) {
            for (const auto& band : model3d::frame_bands(panel, camera, viewport, pixels)) {
                if (!(band.width > 0 && band.height > 0)) continue;
                auto draw = base_draw(band.center, band.width, band.height, false, color);
                draw.depth_flags = CALIPER_GEOM_DEPTH_TEST;
                draws.push_back(draw);
            }
        };
        for (const Panel& panel : scene.panels) {
            if (panel.tensor >= tensors.size()) continue;
            const ModelTensor& tensor = tensors[panel.tensor];
            const auto item = thumbnails.find(tensor.name);
            HostGeomDraw draw = base_draw(panel.center, panel.width, panel.height, false, sternbild::col::plate);
            if (item != thumbnails.end() && item->second.texture && !item->second.rgba.empty()) {
                draw.color_mode = CALIPER_GEOM_COLOR_TEXTURE;
                draw.texture = item->second.texture;
                draw.uv_alloc = uv_buffer;
            }
            draws.push_back(draw);
        }
        // Fold marks share one retained position buffer and one depth-tested
        // draw, rather than adding a separate draw for every block boundary.
        if (fold_buffer && !fold_positions.empty()) {
            HostGeomDraw folds;
            folds.pos_alloc = fold_buffer;
            folds.vertex_count = fold_positions.size() / 3;
            folds.topology = CALIPER_GEOM_TOPO_LINES;
            folds.color_mode = CALIPER_GEOM_COLOR_FLAT;
            folds.shade_mode = CALIPER_GEOM_SHADE_UNLIT;
            folds.blend_mode = CALIPER_GEOM_BLEND_OPAQUE;
            folds.depth_flags = CALIPER_GEOM_DEPTH_TEST;
            folds.flat_rgba = sternbild::col::rule;
            draws.push_back(folds);
        }
        for (const auto& group : scene.groups) {
            Vec3 center{group.bounds.x + group.bounds.width * .5,
                        group.bounds.y + group.bounds.height * .5, group.z};
            draws.push_back(base_draw(center, group.bounds.width, group.bounds.height, true,
                                      full_attention(group.layer_type) ? sternbild::col::parchment : sternbild::col::rule));
            if (full_attention(group.layer_type)) {
                Panel outline;
                outline.center = center; outline.width = group.bounds.width; outline.height = group.bounds.height;
                strong_frame(outline, sternbild::col::parchment, 2.5);
            }
        }
        for (const Panel& panel : scene.panels) {
            if (panel.tensor >= tensors.size()) continue;
            const ModelTensor& tensor = tensors[panel.tensor];
            if (!tensor.edited && tensor.name != selected) continue;
            draws.push_back(base_draw(panel.center, panel.width, panel.height, true,
                                     sternbild::col::bronze));
            strong_frame(panel, sternbild::col::bronze, 2.0);
        }
        const auto matrices = model3d::matrices(camera, aspect);
        failed = !renderer->geom_draw_primitives(target, matrices.view.data(), matrices.projection.data(),
                    draws.empty() ? nullptr : draws.data(), static_cast<uint32_t>(draws.size()), sternbild::col::deep);
        dirty = false;
        rendered_aspect = aspect;
        return !failed;
    }

    void labels(ImDrawList* draw_list, const Rect& column) const {
        draw_list->PushClipRect(ImVec2(static_cast<float>(column.x), static_cast<float>(column.y)),
            ImVec2(static_cast<float>(column.x + column.width), static_cast<float>(column.y + column.height)), true);
        for (const auto& label : model3d::label_layout(scene, camera, viewport, column, ImGui::GetTextLineHeight())) {
            if (label.group >= scene.groups.size()) continue;
            const auto& group = scene.groups[label.group];
            const auto text = group_label(group);
            draw_list->AddText(ImVec2(static_cast<float>(label.bounds.x), static_cast<float>(label.bounds.y)),
                              full_attention(group.layer_type) ? sternbild::col::parchment : sternbild::col::secondary,
                              text.c_str());
        }
        draw_list->PopClipRect();
    }

    std::string draw(HostRenderer* next, float x, float y, float width, float height, const std::string& current) {
        visible = false;
        if (selected != current) { selected = current; dirty = true; }
        ImGui::SetNextWindowPos(ImVec2(x, y), ImGuiCond_FirstUseEver);
        ImGui::SetNextWindowSize(ImVec2(width, height), ImGuiCond_FirstUseEver);
        std::string clicked;
        int linear = 0, full = 0;
        for (const auto& group : scene.groups) {
            if (full_attention(group.layer_type)) ++full;
            else if (*attention_label(group.layer_type)) ++linear;
        }
        char counts[96];
        std::snprintf(counts, sizeof counts, "%zu tensors  %d linear attention  %d full attention", tensors.size(), linear, full);
        // A pane like the inspector's others: its header strip, no scrolling
        // (the wheel zooms the scene).
        if (!ImGui::Begin("Model 3D###caliper.inspector.model3d", nullptr,
                          ImGuiWindowFlags_NoScrollbar | ImGuiWindowFlags_NoScrollWithMouse)) {
            click_armed = false; drag_armed = 0; ImGui::End(); return clicked;
        }
        visible = true;
        sternbild::pane_header("Model 3D", tensors.empty() ? nullptr : counts, nullptr,
                               ImGui::IsWindowFocused(ImGuiFocusedFlags_RootAndChildWindows));
        if (sternbild::word_button("Fit")) {
            framing = Framing::Fit; focused.clear(); framing_pending = true;
        }
        ImGui::SameLine();
        const auto selected_panel = std::find_if(scene.panels.begin(), scene.panels.end(), [&](const Panel& panel) {
            return panel.tensor < tensors.size() && tensors[panel.tensor].name == selected && !panel.output_alias;
        });
        ImGui::BeginDisabled(selected_panel == scene.panels.end());
        if (sternbild::word_button("Focus selected")) {
            // Toolbar items and the label column still change the available
            // viewport. Frame only after their final rectangles are known.
            focused = selected; framing = Framing::Focus; framing_pending = true;
        }
        ImGui::EndDisabled();
        {
            const char* legend = "Bold frame: full attention   Bronze: selected or edited   Left drag rotates, right drag moves, wheel zooms";
            ImGui::SameLine();
            const float width = ImGui::CalcTextSize(legend).x;
            const float x = ImGui::GetCursorPosX() + ImGui::GetContentRegionAvail().x - width;
            if (x > ImGui::GetCursorPosX()) ImGui::SetCursorPosX(x);
            ImGui::AlignTextToFramePadding();
            ImGui::PushStyleColor(ImGuiCol_Text, sternbild::col::faint);
            ImGui::TextUnformatted(legend);
            ImGui::PopStyleColor();
        }
        if (tensors.empty()) {
            ImGui::TextUnformatted("Waiting for tensors");
            ImGui::End(); return clicked;
        }
        const bool supported = bind_renderer(next);
        if (!supported) ImGui::TextUnformatted("3D renderer unavailable");
        if (failed) ImGui::TextUnformatted("3D drawing failed");
        const ImVec2 available = ImGui::GetContentRegionAvail();
        if (available.x < 4 || available.y < 4) { ImGui::End(); return clicked; }
        const ImVec2 origin = ImGui::GetCursorScreenPos();
        double label_width = 0;
        for (const auto& group : scene.groups)
            label_width = std::max(label_width, static_cast<double>(ImGui::CalcTextSize(group_label(group).c_str()).x));
        const auto layout = model3d::viewport_layout({origin.x, origin.y, available.x, available.y}, label_width + 4);
        viewport = layout.scene;
        if (viewport.width < 4 || viewport.height < 4) { ImGui::End(); return clicked; }
        const double aspect = viewport.width / viewport.height;
        const auto frame_camera = [&] {
            if (!framing_pending && std::fabs(aspect - framed_aspect) <= 1e-8) return;
            if (framing == Framing::Fit) camera = model3d::fit(scene, aspect);
            else if (framing == Framing::Focus) {
                const auto panel = std::find_if(scene.panels.begin(), scene.panels.end(), [&](const Panel& p) {
                    return p.tensor < tensors.size() && tensors[p.tensor].name == focused;
                });
                if (panel != scene.panels.end())
                    camera = model3d::focus(scene, static_cast<size_t>(panel - scene.panels.begin()), aspect, camera);
                else { framing = Framing::Fit; camera = model3d::fit(scene, aspect); }
            }
            framing_pending = false; framed_aspect = aspect;
            dirty = true;
        };
        frame_camera();
        label_column = model3d::column_beside_stack(scene, camera, viewport, layout.labels);
        ImGuiIO& io = ImGui::GetIO();
        // The navigation gizmo in the scene's top-right corner, drawn in the
        // house style: an axis navigator (drag it to orbit; click an axis end to
        // turn the camera to look along it) and, under it, round buttons that
        // move the view (drag), zoom it (drag up or down) and fit the model
        // (click). They read the mouse before the scene's own button, so a press
        // on them is theirs, and draw on a second channel above the scene image.
        ImDrawList* scene_list = ImGui::GetWindowDrawList();
        scene_list->ChannelsSplit(2);
        scene_list->ChannelsSetCurrent(1);
        const float cube = std::clamp(static_cast<float>(std::min(viewport.width, viewport.height) * 0.2), 72.0f, 112.0f);
        const ImVec2 cube_min(static_cast<float>(viewport.x + viewport.width) - cube - 8.0f, static_cast<float>(viewport.y) + 8.0f);
        const float button = std::clamp(cube * 0.3f, 26.0f, 34.0f), button_gap = 6.0f;
        const float buttons_width = 3 * button + 2 * button_gap;
        const ImVec2 buttons_min(cube_min.x + (cube - buttons_width) * 0.5f, cube_min.y + cube + 6.0f);
        const ImVec2 nav_min(std::min(cube_min.x, buttons_min.x), cube_min.y);
        const ImVec2 nav_max(std::max(cube_min.x + cube, buttons_min.x + buttons_width), buttons_min.y + button);
        const bool over_nav = io.MousePos.x >= nav_min.x && io.MousePos.x < nav_max.x &&
                              io.MousePos.y >= nav_min.y && io.MousePos.y < nav_max.y;
        {
            // A swing to an axis view eases in over a few frames.
            if (snapping) {
                const double da = std::remainder(snap_azimuth - camera.azimuth, 2 * 3.14159265358979323846);
                const double de = snap_elevation - camera.elevation;
                if (std::fabs(da) < 1e-3 && std::fabs(de) < 1e-3) {
                    camera.azimuth = snap_azimuth; camera.elevation = snap_elevation; snapping = false;
                } else { camera.azimuth += da * 0.25; camera.elevation += de * 0.25; }
                framing = Framing::Manual; dirty = true;
            }
            const ImVec2 centre(cube_min.x + cube * 0.5f, cube_min.y + cube * 0.5f);
            const float radius = cube * 0.5f;
            const double ce = std::cos(camera.elevation);
            const double off[3] = {ce * std::cos(camera.azimuth), std::sin(camera.elevation), ce * std::sin(camera.azimuth)};
            const double fwd[3] = {-off[0], -off[1], -off[2]};
            double right[3] = {-fwd[2], 0.0, fwd[0]};                 // forward x world up
            const double rl = std::sqrt(right[0] * right[0] + right[2] * right[2]);
            if (rl > 1e-9) { right[0] /= rl; right[2] /= rl; } else { right[0] = 1; right[2] = 0; }
            const double up[3] = {right[1] * fwd[2] - right[2] * fwd[1], right[2] * fwd[0] - right[0] * fwd[2],
                                  right[0] * fwd[1] - right[1] * fwd[0]};
            struct End { int axis; int sign; ImVec2 at; double depth; };
            std::array<End, 6> ends{};
            for (int axis = 0; axis < 3; ++axis) for (int sign = 0; sign < 2; ++sign) {
                const double d = sign ? -1.0 : 1.0;
                const ImVec2 at(centre.x + static_cast<float>(right[axis] * d) * radius * 0.7f,
                                centre.y - static_cast<float>(up[axis] * d) * radius * 0.7f);
                ends[axis * 2 + sign] = {axis, sign, at, fwd[axis] * d};
            }
            std::sort(ends.begin(), ends.end(), [](const End& a, const End& b) { return a.depth > b.depth; });
            ImGui::SetCursorScreenPos(cube_min);
            ImGui::InvisibleButton("##model3d.axes", ImVec2(cube, cube));
            const bool axes_hovered = ImGui::IsItemHovered(), axes_active = ImGui::IsItemActive();
            int hovered_end = -1;
            if (axes_hovered) {
                float best = 11.0f * 11.0f;
                for (int i = 0; i < 6; ++i) {
                    const float dx = io.MousePos.x - ends[i].at.x, dy = io.MousePos.y - ends[i].at.y;
                    if (dx * dx + dy * dy < best) { best = dx * dx + dy * dy; hovered_end = i; }
                }
            }
            if (axes_active && ImGui::IsMouseDragging(ImGuiMouseButton_Left, 2) && (io.MouseDelta.x != 0 || io.MouseDelta.y != 0)) {
                camera = model3d::orbit(camera, io.MouseDelta.x, io.MouseDelta.y);
                snapping = false; axes_dragged = true; framing = Framing::Manual; dirty = true;
            }
            if (ImGui::IsItemActivated()) axes_dragged = false;
            if (ImGui::IsItemDeactivated() && !axes_dragged && hovered_end >= 0) {
                // Look along the clicked end, from its side toward the model.
                const int axis = ends[hovered_end].axis;
                const double d = ends[hovered_end].sign ? -1.0 : 1.0;
                const double dir[3] = {axis == 0 ? d : 0.0, axis == 1 ? d : 0.0, axis == 2 ? d : 0.0};
                snap_elevation = std::clamp(std::asin(dir[1]), -1.55, 1.55);
                snap_azimuth = axis == 1 ? camera.azimuth : std::atan2(dir[2], dir[0]);
                snapping = true;
            }
            static const char* const meaning[3] = {"X: across a layer's tensors", "Y: up the tensors' rows",
                                                   "Z: through the layers, in order"};
            if (axes_hovered && !axes_active) {
                if (hovered_end >= 0) ImGui::SetTooltip("%s\nClick to look along %s%c", meaning[ends[hovered_end].axis],
                                                        ends[hovered_end].sign ? "-" : "+", "XYZ"[ends[hovered_end].axis]);
                else ImGui::SetTooltip("Drag to rotate the view\nClick an axis end to look along it");
            }
            if (axes_hovered || axes_active) ImGui::SetMouseCursor(hovered_end >= 0 ? ImGuiMouseCursor_Hand : ImGuiMouseCursor_ResizeAll);
            const ImU32 colours[3] = {sternbild::col::cat, sternbild::col::dog, sternbild::col::vehicle};
            scene_list->AddCircleFilled(centre, radius, sternbild::with_alpha(axes_hovered || axes_active ? sternbild::col::plate_hover
                                                                                                          : sternbild::col::plate, 220));
            scene_list->AddCircle(centre, radius, axes_active ? sternbild::col::bronze : sternbild::col::rule, 0, 1.0f);
            for (int i = 0; i < 6; ++i) {
                const End& e = ends[i];
                const bool far_side = e.depth > 0.05;
                const ImU32 c = sternbild::with_alpha(colours[e.axis], far_side ? 150 : 255);
                const bool lit = i == hovered_end;
                if (!e.sign) scene_list->AddLine(centre, e.at, c, 2.0f);
                if (!e.sign) {
                    scene_list->AddCircleFilled(e.at, 8.5f, c);
                    const char label[2] = {"XYZ"[e.axis], 0};
                    const ImVec2 size = ImGui::CalcTextSize(label);
                    scene_list->AddText(ImVec2(e.at.x - size.x * 0.5f, e.at.y - size.y * 0.5f), sternbild::col::deep, label);
                } else {
                    scene_list->AddCircleFilled(e.at, 6.0f, sternbild::with_alpha(sternbild::col::deep, 200));
                    scene_list->AddCircle(e.at, 6.0f, c, 0, 1.5f);
                }
                if (lit) scene_list->AddCircle(e.at, e.sign ? 8.0f : 10.5f, sternbild::col::bronze, 0, 2.0f);
            }
        }
        enum Nav { Move, Zoom, FitAll };
        const char* const nav_ids[3] = {"##model3d.move", "##model3d.zoom", "##model3d.fit"};
        const char* const nav_tips[3] = {"Move the view: drag", "Zoom: drag up to zoom in, down to zoom out",
                                         "Fit the whole model"};
        for (int i = 0; i < 3; ++i) {
            const ImVec2 at(buttons_min.x + i * (button + button_gap), buttons_min.y);
            ImGui::SetCursorScreenPos(at);
            ImGui::InvisibleButton(nav_ids[i], ImVec2(button, button));
            const bool button_hovered = ImGui::IsItemHovered(), button_active = ImGui::IsItemActive();
            if (button_hovered || button_active) {
                ImGui::SetMouseCursor(i == Move ? ImGuiMouseCursor_ResizeAll
                                      : i == Zoom ? ImGuiMouseCursor_ResizeNS : ImGuiMouseCursor_Hand);
                if (!button_active) ImGui::SetTooltip("%s", nav_tips[i]);
            }
            if (button_active && (io.MouseDelta.x != 0 || io.MouseDelta.y != 0)) {
                if (i == Move) camera = model3d::pan(camera, io.MouseDelta.x, io.MouseDelta.y, viewport.height);
                if (i == Zoom) camera = model3d::zoom(camera, -io.MouseDelta.y * 0.05);
                if (i != FitAll) { framing = Framing::Manual; dirty = true; }
            }
            if (i == FitAll && ImGui::IsItemClicked()) { framing = Framing::Fit; focused.clear(); framing_pending = true; }
            // A round plate with a line icon, bronze while in use.
            const ImVec2 c(at.x + button * 0.5f, at.y + button * 0.5f);
            const float r = button * 0.5f;
            const ImU32 ink = button_active ? sternbild::col::bronze : button_hovered ? sternbild::col::parchment
                                                                                       : sternbild::col::secondary;
            scene_list->AddCircleFilled(c, r, sternbild::with_alpha(button_hovered ? sternbild::col::plate_hover : sternbild::col::plate, 220));
            scene_list->AddCircle(c, r, button_active ? sternbild::col::bronze : sternbild::col::rule, 0, 1.0f);
            const float a = r * 0.55f, head = r * 0.18f;
            if (i == Move) {
                for (int d = 0; d < 4; ++d) {
                    const float dx = d == 0 ? 1.f : d == 1 ? -1.f : 0.f, dy = d == 2 ? 1.f : d == 3 ? -1.f : 0.f;
                    const ImVec2 tip(c.x + dx * a, c.y + dy * a);
                    scene_list->AddLine(c, tip, ink, 1.5f);
                    scene_list->AddTriangleFilled(tip, ImVec2(tip.x - dx * head - dy * head, tip.y - dy * head - dx * head),
                                                  ImVec2(tip.x - dx * head + dy * head, tip.y - dy * head + dx * head), ink);
                }
            } else if (i == Zoom) {
                const ImVec2 lens(c.x - r * 0.12f, c.y - r * 0.12f);
                scene_list->AddCircle(lens, r * 0.34f, ink, 0, 1.5f);
                scene_list->AddLine(ImVec2(lens.x + r * 0.24f, lens.y + r * 0.24f), ImVec2(c.x + r * 0.5f, c.y + r * 0.5f), ink, 2.0f);
                scene_list->AddLine(ImVec2(lens.x - r * 0.16f, lens.y), ImVec2(lens.x + r * 0.16f, lens.y), ink, 1.5f);
                scene_list->AddLine(ImVec2(lens.x, lens.y - r * 0.16f), ImVec2(lens.x, lens.y + r * 0.16f), ink, 1.5f);
            } else {
                const float b = r * 0.48f, l = r * 0.22f;
                for (int sx = -1; sx <= 1; sx += 2) for (int sy = -1; sy <= 1; sy += 2) {
                    const ImVec2 corner(c.x + sx * b, c.y + sy * b);
                    scene_list->AddLine(corner, ImVec2(corner.x - sx * l, corner.y), ink, 1.5f);
                    scene_list->AddLine(corner, ImVec2(corner.x, corner.y - sy * l), ink, 1.5f);
                }
            }
        }
        scene_list->ChannelsSetCurrent(0);
        ImGui::SetCursorScreenPos(ImVec2(static_cast<float>(viewport.x), static_cast<float>(viewport.y)));
        ImGui::InvisibleButton("##model3d.viewport", ImVec2(static_cast<float>(viewport.width), static_cast<float>(viewport.height)),
            ImGuiButtonFlags_MouseButtonLeft | ImGuiButtonFlags_MouseButtonRight | ImGuiButtonFlags_MouseButtonMiddle);
        const bool nav_busy = over_nav;
        const bool hovered = ImGui::IsItemHovered() && !nav_busy && !inside(label_column, io.MousePos.x, io.MousePos.y);
        const bool active = ImGui::IsItemActive();
        if (ImGui::IsMouseClicked(ImGuiMouseButton_Left)) {
            click_armed = hovered && !io.KeyShift;
            dragged = false;
            drag_armed = (drag_armed & ~1u) | (hovered ? 1u : 0u);
        }
        if (ImGui::IsMouseClicked(ImGuiMouseButton_Middle))
            drag_armed = (drag_armed & ~2u) | (hovered ? 2u : 0u);
        if (ImGui::IsMouseClicked(ImGuiMouseButton_Right))
            drag_armed = (drag_armed & ~4u) | (hovered ? 4u : 0u);
        // Left drag rotates (shift+left moves); right or middle drag moves.
        const bool left_drag = (drag_armed & 1u) && ImGui::IsMouseDragging(ImGuiMouseButton_Left, 4);
        const bool middle_drag = (drag_armed & 2u) && ImGui::IsMouseDragging(ImGuiMouseButton_Middle, 0);
        const bool right_drag = (drag_armed & 4u) && ImGui::IsMouseDragging(ImGuiMouseButton_Right, 2);
        if (active && (left_drag || middle_drag || right_drag) && (io.MouseDelta.x != 0 || io.MouseDelta.y != 0)) {
            dragged = true;
            if (middle_drag || right_drag || io.KeyShift)
                camera = model3d::pan(camera, io.MouseDelta.x, io.MouseDelta.y, viewport.height);
            else if (ImGui::IsMouseDown(ImGuiMouseButton_Left))
                camera = model3d::orbit(camera, io.MouseDelta.x, io.MouseDelta.y);
            framing = Framing::Manual;
            dirty = true;
        }
        if (hovered && io.MouseWheel != 0) {
            camera = model3d::zoom_at(camera, io.MouseWheel, viewport, io.MousePos.x, io.MousePos.y);
            framing = Framing::Manual;
            dirty = true;
        }
        if (ImGui::IsWindowFocused() && !io.WantTextInput && ImGui::IsKeyPressed(ImGuiKey_F, false)) {
            framing = Framing::Fit; focused.clear(); framing_pending = true;
            frame_camera();
        }
        label_column = model3d::column_beside_stack(scene, camera, viewport, layout.labels);
        if (supported) {
            const float scale = std::max(0.25f, std::max(io.DisplayFramebufferScale.x, io.DisplayFramebufferScale.y)) * supersample;
            const int physical_width = std::clamp(static_cast<int>(viewport.width * scale), 1, max_target_side);
            const int physical_height = std::clamp(static_cast<int>(viewport.height * scale), 1, max_target_side);
            if (render(physical_width, physical_height, aspect) && target) {
                ImGui::GetWindowDrawList()->AddImage(static_cast<ImTextureID>(renderer->tex_imtexture_id(target)),
                    ImVec2(static_cast<float>(viewport.x), static_cast<float>(viewport.y)),
                    ImVec2(static_cast<float>(viewport.x + viewport.width), static_cast<float>(viewport.y + viewport.height)));
            }
        }
        labels(ImGui::GetWindowDrawList(), label_column);
        if (hovered && !active && !inside(label_column, io.MousePos.x, io.MousePos.y)) {
            const auto hit = model3d::hit_test(scene, camera, viewport, io.MousePos.x, io.MousePos.y);
            if (hit.panel >= 0 && static_cast<size_t>(hit.panel) < scene.panels.size()) {
                const Panel& panel = scene.panels[static_cast<size_t>(hit.panel)];
                if (panel.tensor < tensors.size()) {
                    const ModelTensor& tensor = tensors[panel.tensor];
                    const auto thumbnail = thumbnails.find(tensor.name);
                    ImGui::BeginTooltip();
                    ImGui::TextUnformatted(tensor.name.c_str());
                    if (tensor.tied) ImGui::TextUnformatted("Embedding and output (tied)");
                    ImGui::TextUnformatted(dimensions(tensor).c_str());
                    if (panel.row_blocks > 1)
                        ImGui::Text("%d row blocks, %lld rows per block", panel.row_blocks,
                                    static_cast<long long>(tensor.rows / panel.row_blocks));
                    if (thumbnail != thumbnails.end() && !thumbnail->second.rgba.empty())
                        ImGui::Text("Sample range %.6g to %.6g", thumbnail->second.minimum, thumbnail->second.maximum);
                    else ImGui::TextUnformatted("Sample pending");
                    if (tensor.edited) ImGui::TextUnformatted("Edited");
                    ImGui::EndTooltip();
                    if (ImGui::IsMouseReleased(ImGuiMouseButton_Left) && click_armed && !dragged && !io.KeyShift)
                        clicked = tensor.name;
                }
            }
        }
        if (ImGui::IsMouseReleased(ImGuiMouseButton_Left)) { click_armed = false; drag_armed &= ~1u; }
        if (ImGui::IsMouseReleased(ImGuiMouseButton_Middle)) drag_armed &= ~2u;
        if (ImGui::IsMouseReleased(ImGuiMouseButton_Right)) drag_armed &= ~4u;
        scene_list->ChannelsMerge();
        ImGui::End();
        return clicked;
    }
};

Model3D::Model3D() : impl_(std::make_unique<Impl>()) {}
Model3D::~Model3D() = default;
void Model3D::set_tensors(const std::vector<ModelTensor>& tensors, const std::string& output_label) {
    impl_->set_tensors(tensors, output_label);
}
bool Model3D::set_thumbnail(const std::string& name, int rows, int cols, const std::vector<float>& values,
                            double minimum, double maximum, uint64_t revision) {
    return impl_->set_thumbnail(name, rows, cols, values, minimum, maximum, revision);
}
void Model3D::invalidate(const std::string& name, bool edited) { impl_->invalidate(name, edited); }
std::string Model3D::next_thumbnail() { return impl_->next_thumbnail(); }
void Model3D::thumbnail_failed(const std::string& name) {
    const auto found = impl_->thumbnails.find(name);
    if (found != impl_->thumbnails.end()) { found->second.requested = false; found->second.failed = true; }
}
std::string Model3D::draw(HostRenderer* renderer, float x, float y, float width, float height,
                         const std::string& selected) {
    return impl_->draw(renderer, x, y, width, height, selected);
}
model3d::ViewState Model3D::view_state() const { return {impl_->camera, impl_->viewport, impl_->label_column}; }

} // namespace caliper_live
