#include "mesh_scope.h"

#include <caliper/caliper.hpp>
#include <caliper/adapters/exportable_pool.hpp>
#include <caliper/adapters/torch.hpp>
#include <imgui.h>
#include <torch/torch.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <memory>
#include <vector>

namespace meshscope {
namespace {

constexpr int kGrid = 72;
constexpr float kPi = 3.14159265358979323846f;

struct V3 { float x, y, z; };
V3 operator-(V3 a, V3 b) { return {a.x - b.x, a.y - b.y, a.z - b.z}; }
V3 cross(V3 a, V3 b) {
    return {a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x};
}
float dot(V3 a, V3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
V3 norm3(V3 a) {
    const float l = std::sqrt(dot(a, a));
    return l > 0 ? V3{a.x / l, a.y / l, a.z / l} : V3{0, 1, 0};
}
void look_at(V3 eye, V3 at, V3 up, float* m) {
    const V3 f = norm3(at - eye);
    const V3 s = norm3(cross(f, up));
    const V3 u = cross(s, f);
    const float t[16] = {
        s.x, u.x, -f.x, 0,
        s.y, u.y, -f.y, 0,
        s.z, u.z, -f.z, 0,
        -dot(s, eye), -dot(u, eye), dot(f, eye), 1};
    std::memcpy(m, t, sizeof(t));
}
void perspective(float fovy_rad, float aspect, float zn, float zf, float* m) {
    const float f = 1.0f / std::tan(fovy_rad * 0.5f);
    std::memset(m, 0, 16 * sizeof(float));
    m[0] = f / aspect;
    m[5] = f;
    m[10] = zf / (zn - zf);
    m[11] = -1.0f;
    m[14] = (zn * zf) / (zn - zf);
}

std::vector<int32_t> triangle_indices() {
    std::vector<int32_t> idx;
    idx.reserve((kGrid - 1) * (kGrid - 1) * 6);
    for (int y = 0; y < kGrid - 1; ++y) {
        for (int x = 0; x < kGrid - 1; ++x) {
            const int i0 = y * kGrid + x;
            const int i1 = i0 + 1;
            const int i2 = i0 + kGrid;
            const int i3 = i2 + 1;
            idx.push_back(i0); idx.push_back(i2); idx.push_back(i1);
            idx.push_back(i1); idx.push_back(i2); idx.push_back(i3);
        }
    }
    return idx;
}

std::vector<int32_t> line_indices() {
    std::vector<int32_t> idx;
    idx.reserve((kGrid - 1) * kGrid * 4);
    for (int y = 0; y < kGrid; ++y) {
        for (int x = 0; x < kGrid - 1; ++x) {
            const int i = y * kGrid + x;
            idx.push_back(i); idx.push_back(i + 1);
        }
    }
    for (int y = 0; y < kGrid - 1; ++y) {
        for (int x = 0; x < kGrid; ++x) {
            const int i = y * kGrid + x;
            idx.push_back(i); idx.push_back(i + kGrid);
        }
    }
    return idx;
}

}  // namespace

struct MeshScopeState {
    caliper::Host* host = nullptr;
    caliper::Device device;
    caliper::Bridge bridge;
    caliper::Geometry geometry;
    uint32_t bridge_caps = 0;
    uint32_t geom_caps = 0;

    bool gpu = false;
    bool zero_copy_frame = false;
    bool initialized = false;
    const char* frame_status = "initializing";
    std::unique_ptr<caliper::adapters::ExportablePool> pool;

    torch::Device torch_device = torch::Device(torch::kCPU);
    torch::Tensor base_x, base_y;
    torch::Tensor pos, normal, attr, tri_idx, line_idx;
    int64_t vertex_count = (int64_t)kGrid * (int64_t)kGrid;
    int64_t tri_index_count = 0;
    int64_t line_index_count = 0;

    CaliperTextureId view = 0;
    int view_w = 768, view_h = 768;
    float cam_az = 0.8f, cam_el = 0.55f, cam_dist = 4.5f;
    float phase = 0.0f;
};

MeshScopeApplet::MeshScopeApplet() : s_(std::make_unique<MeshScopeState>()) {}
MeshScopeApplet::~MeshScopeApplet() = default;

bool MeshScopeApplet::initialize(caliper::Host& host) {
    auto* st = s_.get();
    st->host = &host;
    st->device = caliper::Device::query(host);
    st->bridge = caliper::Bridge(host);
    st->geometry = caliper::Geometry(host);
    st->bridge_caps = st->bridge.caps();
    st->geom_caps = st->geometry.caps();

    const bool cuda = torch::cuda::is_available();
#if defined(__APPLE__)
    const bool mps = !cuda && torch::mps::is_available();
#else
    const bool mps = false;
#endif
    st->gpu = cuda || mps;
    st->torch_device = cuda ? torch::Device(torch::kCUDA)
                     : mps  ? torch::Device(torch::kMPS)
                            : torch::Device(torch::kCPU);

    if (st->gpu && st->bridge && st->geometry.has_primitives() &&
        (st->bridge_caps & CALIPER_BRIDGE_CAP_IMPORT_ALLOC)) {
        try {
            auto p = std::make_unique<caliper::adapters::ExportablePool>(0);
            if (p->ok()) st->pool = std::move(p);
        } catch (...) {
        }
    }

    const auto opt_f = torch::TensorOptions(st->torch_device).dtype(torch::kFloat32);
    const auto opt_i = torch::TensorOptions(st->torch_device).dtype(torch::kInt32);
    std::vector<float> bx((size_t)st->vertex_count), by((size_t)st->vertex_count);
    for (int y = 0; y < kGrid; ++y) {
        for (int x = 0; x < kGrid; ++x) {
            const size_t i = (size_t)y * kGrid + (size_t)x;
            bx[i] = -1.6f + 3.2f * (float)x / (float)(kGrid - 1);
            by[i] = -1.6f + 3.2f * (float)y / (float)(kGrid - 1);
        }
    }
    st->base_x = torch::from_blob(bx.data(), {st->vertex_count},
                                  torch::TensorOptions().dtype(torch::kFloat32)).clone().to(st->torch_device);
    st->base_y = torch::from_blob(by.data(), {st->vertex_count},
                                  torch::TensorOptions().dtype(torch::kFloat32)).clone().to(st->torch_device);

    auto tri = triangle_indices();
    auto lines = line_indices();
    st->tri_index_count = (int64_t)tri.size();
    st->line_index_count = (int64_t)lines.size();

    auto allocate_render_tensors = [&] {
        st->pos = torch::empty({st->vertex_count, 3}, opt_f);
        st->normal = torch::empty({st->vertex_count, 3}, opt_f);
        st->attr = torch::empty({st->vertex_count}, opt_f);
        st->tri_idx = torch::empty({st->tri_index_count}, opt_i);
        st->line_idx = torch::empty({st->line_index_count}, opt_i);
    };
    if (st->pool) { auto scope = st->pool->use(); allocate_render_tensors(); }
    else allocate_render_tensors();

    st->tri_idx.copy_(torch::from_blob(tri.data(), {st->tri_index_count},
                                       torch::TensorOptions().dtype(torch::kInt32)).clone().to(st->torch_device));
    st->line_idx.copy_(torch::from_blob(lines.data(), {st->line_index_count},
                                        torch::TensorOptions().dtype(torch::kInt32)).clone().to(st->torch_device));

    char msg[256];
    std::snprintf(msg, sizeof(msg),
                  "mesh-scope: init gpu=%d bridge=%d bridge_caps=0x%x geom_caps=0x%x pool=%d",
                  st->gpu ? 1 : 0, st->bridge ? 1 : 0, st->bridge_caps,
                  st->geom_caps, st->pool ? 1 : 0);
    host.log_info(msg);
    st->initialized = true;
    return true;
}

void MeshScopeApplet::draw_ui() {
    auto* st = s_.get();
    ImGui::Begin("MeshScope: Surface");
    ImGui::TextDisabled("geometry.v1_1: indexed triangles + lines + depth + Lambert, from imported tensors");
    ImGui::SameLine();
    if (st->zero_copy_frame)
        ImGui::TextColored({0.55f, 0.9f, 0.6f, 1.f}, "zero-copy primitives");
    else
        ImGui::TextColored({1.f, 0.7f, 0.4f, 1.f}, "fallback: %s",
                           st->frame_status);

    const ImVec2 avail = ImGui::GetContentRegionAvail();
    const float fb_scale = ImGui::GetIO().DisplayFramebufferScale.y > 0.f
        ? ImGui::GetIO().DisplayFramebufferScale.y : 1.f;
    auto clampi = [](int v, int lo, int hi) { return v < lo ? lo : (v > hi ? hi : v); };
    const int dw = clampi((int)(avail.x * fb_scale), 64, 4096);
    const int dh = clampi((int)(avail.y * fb_scale), 64, 4096);
    if (st->geometry.has_primitives() && avail.x >= 64 && avail.y >= 64 &&
        (st->view == 0 || std::abs(dw - st->view_w) >= 3 || std::abs(dh - st->view_h) >= 3)) {
        if (st->view != 0) st->geometry.release_view(st->view);
        st->view = st->geometry.create_view_ex((uint32_t)dw, (uint32_t)dh,
                                               CALIPER_GEOM_VIEW_DEPTH);
        st->view_w = dw;
        st->view_h = dh;
    }

    st->phase += ImGui::GetIO().DeltaTime;
    st->zero_copy_frame = false;
    st->frame_status =
        !st->gpu ? "torch CPU"
                 : !st->bridge ? "no tensor bridge"
                 : !(st->bridge_caps & CALIPER_BRIDGE_CAP_IMPORT_ALLOC) ? "no import cap"
                 : !st->geometry.has_primitives() ? "no geometry.v1_1 backend"
                 : !st->pool ? "pool unavailable"
                 : st->view == 0 ? "no geometry view"
                 : !st->initialized ? "initializing"
                                    : "not drawn";
    if (st->pool && st->view != 0 && st->initialized) {
        torch::NoGradGuard ng;
        const auto x = st->base_x;
        const auto y = st->base_y;
        const auto a = x * 3.0f + st->phase;
        const auto b = y * 3.0f + st->phase * 0.7f;
        const auto z = 0.36f * torch::sin(a) * torch::cos(b);
        st->pos.select(1, 0).copy_(x);
        st->pos.select(1, 1).copy_(z);
        st->pos.select(1, 2).copy_(y);
        const auto dzdx = 1.08f * torch::cos(a) * torch::cos(b);
        const auto dzdy = -1.08f * torch::sin(a) * torch::sin(b);
        const auto inv_len = torch::rsqrt(dzdx * dzdx + dzdy * dzdy + 1.0f);
        st->normal.select(1, 0).copy_(-dzdx * inv_len);
        st->normal.select(1, 1).copy_(inv_len);
        st->normal.select(1, 2).copy_(-dzdy * inv_len);
        st->attr.copy_(z);
        if (torch::cuda::is_available()) torch::cuda::synchronize();
#if defined(__APPLE__)
        else if (torch::mps::is_available()) caliper::adapters::detail::mps_synchronize_serialized();
#endif

        auto pref = st->pool->to_bridge(st->bridge, st->pos);
        auto nref = st->pool->to_bridge(st->bridge, st->normal);
        auto aref = st->pool->to_bridge(st->bridge, st->attr);
        auto tref = st->pool->to_bridge(st->bridge, st->tri_idx);
        auto lref = st->pool->to_bridge(st->bridge, st->line_idx);
        if (pref && nref && aref && tref && lref) {
            const float ce = std::cos(st->cam_el), se = std::sin(st->cam_el);
            const float ca = std::cos(st->cam_az), sa = std::sin(st->cam_az);
            const V3 eye{st->cam_dist * ce * ca, st->cam_dist * se,
                         st->cam_dist * ce * sa};
            CaliperGeomCamera cam{};
            look_at(eye, {0, 0, 0}, {0, 1, 0}, cam.view);
            perspective(45.f * kPi / 180.f, (float)st->view_w / (float)st->view_h,
                        0.05f, 50.f, cam.proj);

            CaliperGeomDraw mesh = caliper::geom_draw_defaults();
            mesh.pos_alloc = pref->alloc;
            mesh.pos_offset = pref->offset;
            mesh.vertex_count = (uint64_t)st->vertex_count;
            mesh.index_alloc = tref->alloc;
            mesh.index_offset = tref->offset;
            mesh.index_count = (uint64_t)st->tri_index_count;
            mesh.normal_alloc = nref->alloc;
            mesh.normal_offset = nref->offset;
            mesh.attr_alloc = aref->alloc;
            mesh.attr_offset = aref->offset;
            mesh.topology = CALIPER_GEOM_TOPO_TRIANGLES;
            mesh.color_mode = CALIPER_GEOM_COLOR_COLORMAP;
            mesh.shade_mode = CALIPER_GEOM_SHADE_LAMBERT;
            mesh.blend_mode = CALIPER_GEOM_BLEND_OPAQUE;
            mesh.depth_flags = CALIPER_GEOM_DEPTH_TEST | CALIPER_GEOM_DEPTH_WRITE;
            mesh.colormap = CALIPER_CMAP_VIRIDIS;
            mesh.vmin = -0.36f;
            mesh.vmax = 0.36f;

            CaliperGeomDraw wire = caliper::geom_draw_defaults();
            wire.pos_alloc = pref->alloc;
            wire.pos_offset = pref->offset;
            wire.vertex_count = (uint64_t)st->vertex_count;
            wire.index_alloc = lref->alloc;
            wire.index_offset = lref->offset;
            wire.index_count = (uint64_t)st->line_index_count;
            wire.topology = CALIPER_GEOM_TOPO_LINES;
            wire.color_mode = CALIPER_GEOM_COLOR_FLAT;
            wire.blend_mode = CALIPER_GEOM_BLEND_ALPHA;
            wire.depth_flags = CALIPER_GEOM_DEPTH_TEST;
            wire.flat_rgba = 0x99ffffffu;

            CaliperGeomDraw draws[2] = {mesh, wire};
            st->zero_copy_frame = st->geometry.draw_primitives(
                st->view, cam, draws, 2, 0xff05050au);
            st->frame_status = st->zero_copy_frame ? "zero-copy primitives"
                                                   : "draw_primitives refused";
        } else {
            st->frame_status =
                !pref ? "position import failed"
                      : !nref ? "normal import failed"
                              : !aref ? "attribute import failed"
                                      : !tref ? "triangle-index import failed"
                                              : "line-index import failed";
        }
    }

    if (st->zero_copy_frame) {
        ImGui::Image(caliper::Bridge::imtex(st->view),
                     ImVec2((float)st->view_w / fb_scale,
                            (float)st->view_h / fb_scale));
        const bool hovered = ImGui::IsItemHovered();
        ImGuiIO& io = ImGui::GetIO();
        if (hovered && ImGui::IsMouseDown(ImGuiMouseButton_Right)) {
            st->cam_az += io.MouseDelta.x * 0.008f;
            st->cam_el += io.MouseDelta.y * 0.008f;
            st->cam_el = std::clamp(st->cam_el, -1.45f, 1.45f);
        }
        if (hovered && io.MouseWheel != 0.f) {
            st->cam_dist *= (1.f - io.MouseWheel * 0.08f);
            st->cam_dist = std::clamp(st->cam_dist, 2.0f, 9.0f);
        }
    } else {
        ImGui::Dummy(ImVec2(std::max(64.f, avail.x), std::max(64.f, avail.y)));
    }
    ImGui::End();
}

void MeshScopeApplet::cleanup() {
    auto* st = s_.get();
    if (st->view != 0) {
        st->geometry.release_view(st->view);
        st->view = 0;
    }
    st->pos = torch::Tensor();
    st->normal = torch::Tensor();
    st->attr = torch::Tensor();
    st->tri_idx = torch::Tensor();
    st->line_idx = torch::Tensor();
    st->pool.reset();
    if (st->host) st->host->log_info("mesh-scope: on_cleanup");
}

}  // namespace meshscope
