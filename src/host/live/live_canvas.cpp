#include "live/live_canvas.h"
#include "live/live_views_state.h"
#include "renderer/host_renderer.h"
#include "ui/sternbild_theme.h"
#include <caliper/services/geometry_v1_1.h>
#include <caliper/services/tensor_bridge_v1_2.h>
#include <imgui.h>
#include <algorithm>
#include <cmath>

namespace caliper_live {

using caliper_host::HostRenderer;

GeomCanvas::~GeomCanvas() { release(); }

void GeomCanvas::release() {
    if (renderer_) {
        if (target_) renderer_->tex_release(target_);
        for (uint64_t id : imports_) renderer_->release_external_allocation(id);
    }
    imports_.clear();
    target_ = 0;
    width_ = height_ = 0;
    dirty_ = true;
    ++generation_;
}

void GeomCanvas::attach(HostRenderer* renderer) {
    if (renderer == renderer_) return;
    release();
    renderer_ = renderer;
}

bool GeomCanvas::ready() const {
    return renderer_ && renderer_->supports_geometry_primitives() && renderer_->supports_external_import() &&
           renderer_->interop_device() == CALIPER_DEV_METAL;
}

uint64_t GeomCanvas::upload(uint64_t& id, const void* bytes, size_t length) {
    if (id && renderer_) {
        renderer_->release_external_allocation(id);
        imports_.erase(std::remove(imports_.begin(), imports_.end(), id), imports_.end());
    }
    id = 0;
    if (!ready() || !bytes || !length) return 0;
    void* buffer = caliper_live_views::metal_new_shared_buffer(renderer_->native_device(), bytes, length);
    if (!buffer) return 0;
    id = renderer_->import_external_allocation(buffer, length, CALIPER_ALLOC_HANDLE_MTLBUFFER);
    caliper_live_views::metal_release(buffer);   // the renderer retains a successful import, in-flight draws too
    if (id) imports_.push_back(id);
    dirty_ = true;
    return id;
}

GeomCanvas::Frame GeomCanvas::draw(const char* id,
                                   const std::function<bool(uint64_t, const model3d::Matrices&)>& render) {
    Frame f;
    ImGuiIO& io = ImGui::GetIO();
    const ImVec2 at = ImGui::GetCursorScreenPos(), available = ImGui::GetContentRegionAvail();
    const float w = std::max(available.x, 8.0f), h = std::max(available.y, 8.0f);
    f.rect = {at.x, at.y, w, h};
    ImGui::SetNextItemAllowOverlap();
    ImGui::InvisibleButton(id, ImVec2(w, h), ImGuiButtonFlags_MouseButtonLeft | ImGuiButtonFlags_MouseButtonRight |
                                                 ImGuiButtonFlags_MouseButtonMiddle);
    const bool over = ImGui::IsItemHovered(), active = ImGui::IsItemActive();
    if (ImGui::IsMouseClicked(ImGuiMouseButton_Left)) drag_ = (drag_ & ~1u) | (over ? 1u : 0u);
    if (ImGui::IsMouseClicked(ImGuiMouseButton_Right)) drag_ = (drag_ & ~2u) | (over ? 2u : 0u);
    if (ImGui::IsMouseClicked(ImGuiMouseButton_Middle)) drag_ = (drag_ & ~4u) | (over ? 4u : 0u);
    const bool left = (drag_ & 1u) && ImGui::IsMouseDragging(ImGuiMouseButton_Left, 2);
    const bool move = (left && io.KeyShift) || ((drag_ & 2u) && ImGui::IsMouseDragging(ImGuiMouseButton_Right, 2)) ||
                      ((drag_ & 4u) && ImGui::IsMouseDragging(ImGuiMouseButton_Middle, 0));
    const bool rotate = left && !io.KeyShift;
    if (active && (io.MouseDelta.x != 0 || io.MouseDelta.y != 0)) {
        if (move) { camera = model3d::pan(camera, io.MouseDelta.x, io.MouseDelta.y, h); dirty_ = true; }
        else if (rotate) { camera = model3d::orbit(camera, io.MouseDelta.x, io.MouseDelta.y); dirty_ = true; }
    }
    if (over && io.MouseWheel != 0) {
        camera = model3d::zoom_at(camera, io.MouseWheel, f.rect, io.MousePos.x, io.MousePos.y);
        dirty_ = true;
    }
    f.hovered = over && !(active && (rotate || move));
    f.mouse_x = io.MousePos.x;
    f.mouse_y = io.MousePos.y;
    const double aspect = static_cast<double>(w) / h;
    f.m = model3d::matrices(camera, aspect);
    ImDrawList* list = ImGui::GetWindowDrawList();
    list->AddRectFilled(at, ImVec2(at.x + w, at.y + h), sternbild::col::deep);
    if (!ready()) return f;
    // The view is drawn at framebuffer resolution and shown at its logical size.
    const float scale = std::max(1.0f, std::max(io.DisplayFramebufferScale.x, io.DisplayFramebufferScale.y));
    const int pw = std::max(8, static_cast<int>(w * scale)), ph = std::max(8, static_cast<int>(h * scale));
    if (!target_ || std::abs(pw - width_) > 2 || std::abs(ph - height_) > 2) {
        const uint64_t next = renderer_->geom_create_view_ex(pw, ph, CALIPER_GEOM_VIEW_DEPTH);
        if (!next) return f;
        if (target_) renderer_->tex_release(target_);
        target_ = next;
        width_ = pw;
        height_ = ph;
        dirty_ = true;
    }
    if (std::fabs(aspect - aspect_) > 1e-6) dirty_ = true;
    if (dirty_) {
        failed_ = !render(target_, f.m);
        dirty_ = false;
        aspect_ = aspect;
    }
    if (!failed_)
        list->AddImage(static_cast<ImTextureID>(renderer_->tex_imtexture_id(target_)), at, ImVec2(at.x + w, at.y + h));
    return f;
}

bool GeomCanvas::project(const Frame& f, float x, float y, float z, float& sx, float& sy) {
    // The view matrix is column-major and looks down -z (model3d::matrices).
    const auto& v = f.m.view;
    const auto& p = f.m.projection;
    const float vx = v[0] * x + v[4] * y + v[8] * z + v[12];
    const float vy = v[1] * x + v[5] * y + v[9] * z + v[13];
    const float vz = v[2] * x + v[6] * y + v[10] * z + v[14];
    if (!(vz < 0)) return false;
    const float nx = p[0] * vx / -vz, ny = p[5] * vy / -vz;
    sx = static_cast<float>(f.rect.x + (nx * 0.5f + 0.5f) * f.rect.width);
    sy = static_cast<float>(f.rect.y + (0.5f - ny * 0.5f) * f.rect.height);
    return true;
}

} // namespace caliper_live
