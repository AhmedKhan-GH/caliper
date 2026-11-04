#include "live/live_model3d.h"

#include "live/live_views_state.h"
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
constexpr int max_target_side = 4096;
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

} // namespace

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
    HostRenderer* renderer = nullptr;
    uint64_t position_buffer = 0, uv_buffer = 0, target = 0;
    int target_width = 0, target_height = 0;
    bool visible = false, dirty = true, fit_pending = true, dragged = false;
    bool failed = false, imports_failed = false;
    std::string selected, output_label = "Output (tied)";

    ~Impl() { release_renderer(); }

    void release_renderer() {
        if (renderer) {
            if (target) renderer->tex_release(target);
            if (position_buffer) renderer->release_external_allocation(position_buffer);
            if (uv_buffer) renderer->release_external_allocation(uv_buffer);
            for (auto& entry : thumbnails) {
                if (entry.second.texture) renderer->tex_release(entry.second.texture);
                entry.second.texture = 0;
                entry.second.upload = !entry.second.rgba.empty();
            }
        }
        renderer = nullptr;
        position_buffer = uv_buffer = target = 0;
        target_width = target_height = 0;
        failed = imports_failed = false;
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
        if (position_buffer && uv_buffer) return true;
        if (imports_failed) return false;
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
            fit_pending = true;
        }
        dirty = true;
    }

    bool set_thumbnail(const std::string& name, int rows, int cols,
                       const std::vector<float>& values, double minimum, double maximum, uint64_t revision) {
        const auto found = indices.find(name);
        if (found == indices.end() || rows < 1 || cols < 1 || rows > max_thumbnail_side ||
            cols > max_thumbnail_side || values.size() != static_cast<size_t>(rows * cols) ||
            !std::isfinite(minimum) || !std::isfinite(maximum) || minimum > maximum ||
            !std::all_of(values.begin(), values.end(), [](float v) { return std::isfinite(v); })) return false;
        const ModelTensor& tensor = tensors[found->second];
        if (tensor.shape.size() == 1 && values.size() > max_thumbnail_side) return false;
        Thumbnail& item = thumbnails[name];
        if (revision < tensor.revision) {
            item.requested = false;
            return false;
        }
        item.minimum = minimum;
        item.maximum = maximum;
        item.revision = revision;
        // Vectors retain their Nx1 wire layout but occupy a horizontal strip.
        item.width = tensor.shape.size() == 1 ? rows * cols : cols;
        item.height = tensor.shape.size() == 1 ? 1 : rows;
        item.rgba.resize(values.size() * 4);
        caliper_host::map_f32_to_rgba8(values.data(), item.width, item.height,
            caliper_host::colormap_lut(CALIPER_CMAP_VIRIDIS), static_cast<float>(minimum),
            static_cast<float>(maximum), item.rgba.data());
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
            const uint64_t texture = renderer->tex_create_rgba8(item.width, item.height);
            if (!texture || !renderer->tex_upload_rgba8(texture, item.rgba.data(), item.width, item.height)) {
                if (texture) renderer->tex_release(texture);
                item.upload = false;
                item.failed = true;
                continue;
            }
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
        if (!dirty) return !failed;
        upload_thumbnails();
        std::vector<HostGeomDraw> draws;
        draws.reserve(scene.panels.size() * 2 + scene.groups.size() * 2);
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
        // Group boundaries provide a stable language for attention kinds:
        // linear-attention layers have one frame, full-attention layers two.
        for (const auto& group : scene.groups) {
            Vec3 center{group.bounds.x + group.bounds.width * .5,
                        group.bounds.y + group.bounds.height * .5, group.z};
            draws.push_back(base_draw(center, group.bounds.width, group.bounds.height, true, sternbild::col::rule));
            if (full_attention(group.layer_type))
                draws.push_back(base_draw(center, group.bounds.width * .98, group.bounds.height * .98,
                                          true, sternbild::col::secondary));
        }
        for (const Panel& panel : scene.panels) {
            if (panel.tensor >= tensors.size()) continue;
            const ModelTensor& tensor = tensors[panel.tensor];
            if (!tensor.edited && tensor.name != selected) continue;
            draws.push_back(base_draw(panel.center, panel.width, panel.height, true,
                                     tensor.edited ? sternbild::col::bronze : sternbild::col::parchment));
        }
        const auto matrices = model3d::matrices(camera, aspect);
        failed = !renderer->geom_draw_primitives(target, matrices.view.data(), matrices.projection.data(),
                    draws.empty() ? nullptr : draws.data(), static_cast<uint32_t>(draws.size()), sternbild::col::deep);
        dirty = false;
        return !failed;
    }

    void labels(ImDrawList* draw_list, const Rect& viewport, double aspect) const {
        draw_list->PushClipRect(ImVec2(static_cast<float>(viewport.x), static_cast<float>(viewport.y)),
            ImVec2(static_cast<float>(viewport.x + viewport.width), static_cast<float>(viewport.y + viewport.height)), true);
        for (const auto& group : scene.groups) {
            const Vec3 anchor{group.bounds.x, group.bounds.y + group.bounds.height, group.z};
            const auto point = model3d::project(anchor, camera, aspect, viewport);
            if (!point.visible) continue;
            std::string label = group.label;
            const char* kind = attention_label(group.layer_type);
            if (*kind) label += std::string("  ") + kind;
            draw_list->AddText(ImVec2(static_cast<float>(point.x + 3), static_cast<float>(point.y - ImGui::GetTextLineHeight())),
                              full_attention(group.layer_type) ? sternbild::col::parchment : sternbild::col::secondary,
                              label.c_str());
        }
        draw_list->PopClipRect();
    }

    std::string draw(HostRenderer* next, float x, float y, float width, float height, const std::string& current) {
        visible = false;
        if (selected != current) { selected = current; dirty = true; }
        ImGui::SetNextWindowPos(ImVec2(x, y), ImGuiCond_FirstUseEver);
        ImGui::SetNextWindowSize(ImVec2(width, height), ImGuiCond_FirstUseEver);
        std::string clicked;
        if (!ImGui::Begin("Model 3D###caliper.inspector.model3d")) { ImGui::End(); return clicked; }
        visible = true;
        int linear = 0, full = 0;
        for (const auto& group : scene.groups) {
            if (full_attention(group.layer_type)) ++full;
            else if (*attention_label(group.layer_type)) ++linear;
        }
        if (ImGui::Button("Fit")) fit_pending = true;
        ImGui::SameLine();
        const auto selected_panel = std::find_if(scene.panels.begin(), scene.panels.end(), [&](const Panel& panel) {
            return panel.tensor < tensors.size() && tensors[panel.tensor].name == selected && !panel.output_alias;
        });
        ImGui::BeginDisabled(selected_panel == scene.panels.end());
        if (ImGui::Button("Focus selected")) {
            const size_t index = static_cast<size_t>(selected_panel - scene.panels.begin());
            const double aspect = std::max(1.0f, ImGui::GetContentRegionAvail().x) /
                                  std::max(1.0f, ImGui::GetContentRegionAvail().y);
            camera = model3d::focus(scene, index, aspect, camera);
            fit_pending = false;
            dirty = true;
        }
        ImGui::EndDisabled();
        ImGui::SameLine();
        ImGui::Text("%zu tensors  %d Linear  %d Full", tensors.size(), linear, full);
        ImGui::TextUnformatted("Linear: single frame   Full: double frame");
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
        const Rect viewport{origin.x, origin.y, available.x, available.y};
        const double aspect = viewport.width / viewport.height;
        if (fit_pending) {
            camera = model3d::fit(scene, aspect, camera.azimuth, camera.elevation);
            fit_pending = false;
            dirty = true;
        }
        ImGui::InvisibleButton("##model3d.viewport", available,
            ImGuiButtonFlags_MouseButtonLeft | ImGuiButtonFlags_MouseButtonMiddle);
        const bool hovered = ImGui::IsItemHovered();
        const bool active = ImGui::IsItemActive();
        ImGuiIO& io = ImGui::GetIO();
        if (hovered && ImGui::IsMouseClicked(ImGuiMouseButton_Left)) dragged = false;
        if (active && (io.MouseDelta.x != 0 || io.MouseDelta.y != 0)) {
            if (ImGui::IsMouseDragging(ImGuiMouseButton_Left, 4) || ImGui::IsMouseDown(ImGuiMouseButton_Middle)) dragged = true;
            if (ImGui::IsMouseDown(ImGuiMouseButton_Middle) || io.KeyShift)
                camera = model3d::pan(camera, io.MouseDelta.x, io.MouseDelta.y, viewport.height);
            else if (ImGui::IsMouseDown(ImGuiMouseButton_Left))
                camera = model3d::orbit(camera, io.MouseDelta.x, io.MouseDelta.y);
            dirty = true;
        }
        if (hovered && io.MouseWheel != 0) {
            camera = model3d::zoom(camera, io.MouseWheel);
            dirty = true;
        }
        if (ImGui::IsWindowFocused() && !io.WantTextInput && ImGui::IsKeyPressed(ImGuiKey_F, false)) {
            camera = model3d::fit(scene, aspect, camera.azimuth, camera.elevation);
            dirty = true;
        }
        if (supported) {
            const float scale = std::max(0.25f, std::max(io.DisplayFramebufferScale.x, io.DisplayFramebufferScale.y));
            const int physical_width = std::clamp(static_cast<int>(available.x * scale), 1, max_target_side);
            const int physical_height = std::clamp(static_cast<int>(available.y * scale), 1, max_target_side);
            if (render(physical_width, physical_height, aspect) && target) {
                ImGui::GetWindowDrawList()->AddImage(static_cast<ImTextureID>(renderer->tex_imtexture_id(target)), origin,
                    ImVec2(origin.x + available.x, origin.y + available.y));
                labels(ImGui::GetWindowDrawList(), viewport, aspect);
            }
        }
        if (hovered && !active) {
            const auto hit = model3d::hit_test(scene, camera, viewport, io.MousePos.x, io.MousePos.y);
            if (hit.panel >= 0 && static_cast<size_t>(hit.panel) < scene.panels.size()) {
                const Panel& panel = scene.panels[static_cast<size_t>(hit.panel)];
                if (panel.tensor < tensors.size()) {
                    const ModelTensor& tensor = tensors[panel.tensor];
                    const auto thumbnail = thumbnails.find(tensor.name);
                    ImGui::BeginTooltip();
                    ImGui::TextUnformatted(panel.output_alias ? panel.label.c_str() : tensor.name.c_str());
                    if (panel.output_alias) ImGui::Text("Shared with %s", tensor.name.c_str());
                    ImGui::TextUnformatted(dimensions(tensor).c_str());
                    if (thumbnail != thumbnails.end() && !thumbnail->second.rgba.empty())
                        ImGui::Text("Sample range %.6g to %.6g", thumbnail->second.minimum, thumbnail->second.maximum);
                    else ImGui::TextUnformatted("Sample pending");
                    if (tensor.edited) ImGui::TextUnformatted("Edited");
                    ImGui::EndTooltip();
                    if (ImGui::IsMouseReleased(ImGuiMouseButton_Left) && !dragged && !io.KeyShift)
                        clicked = tensor.name;
                }
            }
        }
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

} // namespace caliper_live
