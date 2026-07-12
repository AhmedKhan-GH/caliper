// ============================================================================
// MeshScope — a small MLP's learned 2-D function surface, drawn with ZERO
// copies of the vertex data (id dev.caliper.mesh-scope 0.1.0).
//
// The ML-visualization exemplar for caliper.geometry.v1_1 (GEOMETRY.md §9.3):
// a 2->64->64->1 tanh net trains live against a fixed target surface; every
// optimizer step its prediction over a 72x72 grid is written into imported
// device tensors and drawn the SAME frame as Lambert-lit indexed triangles
// colored by per-vertex squared error through the MAGMA LUT, a white wireframe
// overlay, and the training minibatch as an additive point cloud — watching a
// function being learned, as geometry.
//
// Built on the flow_scope/sculpt_scope backbone, reused verbatim: triple-
// buffered ExportablePool slots, the ready/display invariant, worker/frame
// threading through caliper.jobs.v1 with one publish mutex, orbit/zoom camera
// (applet-owned math), DPI-correct view sizing, and the honest fallback ladder
// (no caps / no pool / CPU torch / no view -> an input-locked ImPlot heatmap of
// the same per-vertex error). Status line reports "zero-copy (imported
// geometry)" only when draw_primitives actually drew this frame.
// ============================================================================
#include "mesh_scope.h"
#include "mesh_model.h"

#include <caliper/caliper.hpp>
#include <caliper/adapters/exportable_pool.hpp>
#include <caliper/adapters/torch.hpp>
#include <imgui.h>
#include <implot.h>
#include <torch/torch.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>

namespace meshscope {
namespace {

constexpr int   kSlots = 3;      // triple buffer: write / ready / displayed
constexpr float kPi    = 3.14159265358979323846f;
constexpr uint64_t kResetSeed = 1234;   // reproducible re-init on the reset button

// Read-only viewer flags for the fallback plot — it is a heatmap you look at,
// not a widget you drive (the recurring "input-lock read-only plots" rule).
constexpr ImPlotFlags kLockedPlot = ImPlotFlags_NoInputs | ImPlotFlags_NoMenus |
                                    ImPlotFlags_NoBoxSelect | ImPlotFlags_NoMouseText;

// ---- tiny column-major mat4 helpers (verbatim from flow_scope) -------------
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
    m[0]  = f / aspect;
    m[5]  = f;
    m[10] = zf / (zn - zf);
    m[11] = -1.0f;
    m[14] = (zn * zf) / (zn - zf);
}

// Static index buffers for the kGrid x kGrid vertex lattice (i = y*kGrid + x).
std::vector<int32_t> triangle_indices() {
    std::vector<int32_t> idx;
    idx.reserve((size_t)(kGrid - 1) * (kGrid - 1) * 6);
    for (int y = 0; y < kGrid - 1; ++y) {
        for (int x = 0; x < kGrid - 1; ++x) {
            const int i0 = y * kGrid + x, i1 = i0 + 1;
            const int i2 = i0 + kGrid, i3 = i2 + 1;
            idx.push_back(i0); idx.push_back(i2); idx.push_back(i1);
            idx.push_back(i1); idx.push_back(i2); idx.push_back(i3);
        }
    }
    return idx;
}
std::vector<int32_t> line_indices() {
    std::vector<int32_t> idx;
    idx.reserve((size_t)(kGrid - 1) * kGrid * 4);
    for (int y = 0; y < kGrid; ++y)
        for (int x = 0; x < kGrid - 1; ++x) {
            const int i = y * kGrid + x;
            idx.push_back(i); idx.push_back(i + 1);
        }
    for (int y = 0; y < kGrid - 1; ++y)
        for (int x = 0; x < kGrid; ++x) {
            const int i = y * kGrid + x;
            idx.push_back(i); idx.push_back(i + kGrid);
        }
    return idx;
}

}  // namespace

// ---------------------------------------------------------------------------
// pImpl state. Cross-thread fields live under `mtx`; the frame-thread-only
// block at the bottom never locks.
// ---------------------------------------------------------------------------
struct MeshScopeState {
    caliper::Host*    host = nullptr;
    caliper::Jobs     jobs;
    caliper::Device   device;
    caliper::Bridge   bridge;
    caliper::Geometry geometry;
    uint32_t geom_caps = 0;

    uint64_t job_id = 0;
    std::atomic<bool> stop{false};

    // frame -> worker knobs (atomics; torn reads harmless).
    std::atomic<bool>  train_on{true};
    std::atomic<bool>  reset_req{false};
    std::atomic<bool>  reset_target_req{false};   // "reset target" button; worker applies
    std::atomic<float> lr{3e-3f};

    // A painted Gaussian bump on the target: domain center, radius, signed amp.
    struct Stroke { float cx, cy, radius, amp; };

    std::mutex mtx;  // guards everything below, down to the frame block

    // Brush strokes pushed by the frame thread (never torch), drained by the
    // worker at the top of each step before train_step (the threading rule).
    std::vector<Stroke> strokes;

    // Triple-buffered render slots (pool-born on the zero-copy path) + the two
    // static index buffers, imported once.
    std::unique_ptr<caliper::adapters::ExportablePool> pool;
    torch::Tensor pos[kSlots], normal[kSlots], attr[kSlots], sample_pos[kSlots];
    torch::Tensor tri_idx, line_idx;
    int  ready_slot   = -1;
    int  display_slot = -1;
    int64_t vertex_count   = (int64_t)kGrid * kGrid;
    int64_t sample_count   = kBatch;
    int64_t tri_index_count = 0;
    int64_t line_index_count = 0;
    bool  on_gpu = false;

    // published scalars for the status line.
    float   last_loss = 0.f;
    float   grid_mse  = 0.f;
    int64_t steps_trained = 0;

    // CPU fallback: the per-vertex err^2 grid, refreshed ~4 Hz by the worker.
    std::vector<float> err_grid;   // kGrid*kGrid, row-major (i = y*kGrid + x)
    uint64_t err_gen = 0;

    // ------- frame-thread-only -------
    CaliperTextureId view = 0;
    int   view_w = 768, view_h = 768;
    float cam_az = 0.8f, cam_el = 0.55f, cam_dist = 4.5f;
    float color_vmax = 0.05f;   // err^2 LUT ceiling (UI-tunable)
    float brush_radius = 0.35f;   // paint brush radius, domain units
    float brush_strength = 0.8f;  // paint rate, target units/sec
    bool  zero_copy_frame = false;
    bool  logged_first_draw = false;
    const char* frame_status = "initializing";
};

namespace {

// ---- the worker: build the model, then train + publish forever -------------
void mesh_job(MeshScopeState* st, const CaliperJobControl* ctl) {
    // NB: no worker-wide NoGradGuard — train_step needs autograd; the inference
    // paths (publish, normals) scope their own NoGradGuard.
    const bool cuda = torch::cuda::is_available();
#if defined(__APPLE__)
    const bool mps  = !cuda && torch::mps::is_available();
#else
    const bool mps  = false;
#endif
    const bool gpu  = cuda || mps;
    const torch::Device dev = cuda ? torch::Device(torch::kCUDA)
                            : mps  ? torch::Device(torch::kMPS)
                                   : torch::Device(torch::kCPU);

    // Zero-copy opt-in, decided once: geometry primitives caps + import caps +
    // a GPU device. Falls back to a plain-tensor CPU heatmap otherwise.
    std::unique_ptr<caliper::adapters::ExportablePool> pool;
    if (gpu && (st->geom_caps & CALIPER_GEOM_CAP_PRIMITIVES)) {
        try {
            auto p = std::make_unique<caliper::adapters::ExportablePool>(0);
            if (p->ok()) pool = std::move(p);
        } catch (...) { /* pool absent -> fallback path, never a crash */ }
    }
    if (st->host) {
        if (pool)
            st->host->log_info(cuda ? "mesh-scope: zero-copy pool ready (cuda)"
                                    : "mesh-scope: zero-copy pool ready (mps)");
        else
            st->host->log_info(
                !gpu ? "mesh-scope: fallback (torch CPU)"
                     : !(st->geom_caps & CALIPER_GEOM_CAP_PRIMITIVES)
                           ? "mesh-scope: fallback (no geometry.v1_1 backend)"
                           : "mesh-scope: fallback (pool unavailable)");
    }

    // The learner (worker-thread-only; the frame never touches it).
    MeshModel model(dev, st->lr.load());
    auto gx = model.grid.select(1, 0);   // constant world-x column (view)
    auto gy = model.grid.select(1, 1);   // constant world-z column (view)

    // Static index buffers (int32, non-negative bit patterns the shader reads).
    auto tri   = triangle_indices();
    auto lines = line_indices();
    const int64_t N   = (int64_t)kGrid * kGrid;
    const int64_t nTi = (int64_t)tri.size();
    const int64_t nLi = (int64_t)lines.size();

    // Render slots + indices: pool-born when we have a pool (all the renderer
    // reads). Model tensors + minibatch stay default-allocated.
    torch::Tensor pos[kSlots], normal[kSlots], attr[kSlots], sample_pos[kSlots];
    torch::Tensor tri_idx, line_idx;
    {
        auto opt_f = torch::TensorOptions(dev).dtype(torch::kFloat32);
        auto opt_i = torch::TensorOptions(dev).dtype(torch::kInt32);
        auto alloc_slots = [&] {
            for (int i = 0; i < kSlots; ++i) {
                pos[i]        = torch::empty({N, 3}, opt_f);
                normal[i]     = torch::empty({N, 3}, opt_f);
                attr[i]       = torch::empty({N}, opt_f);
                sample_pos[i] = torch::empty({(int64_t)kBatch, 3}, opt_f);
            }
            tri_idx  = torch::empty({nTi}, opt_i);
            line_idx = torch::empty({nLi}, opt_i);
        };
        if (pool) { auto scope = pool->use(); alloc_slots(); }
        else      { alloc_slots(); }
    }
    tri_idx.copy_(torch::from_blob(tri.data(), {nTi},
                  torch::TensorOptions().dtype(torch::kInt32)).clone().to(dev));
    line_idx.copy_(torch::from_blob(lines.data(), {nLi},
                   torch::TensorOptions().dtype(torch::kInt32)).clone().to(dev));

    // Publish helper: write slot `w` from the current surface, device-sync, then
    // flip ready_slot under the one mutex (the memory-stability contract).
    auto publish = [&](int w, int64_t step) {
        float mse_now = 0.f;
        {
            torch::NoGradGuard ng;
            auto z = model.net->forward(model.grid);            // (N,) heights
            pos[w].select(1, 0).copy_(gx);
            pos[w].select(1, 1).copy_(z);
            pos[w].select(1, 2).copy_(gy);
            auto err = (z - model.target.grid).pow(2);          // (N,) err^2
            attr[w].copy_(err);
            normal[w].copy_(finite_diff_normals(z));
            sample_pos[w].select(1, 0).copy_(model.batch_xy.select(1, 0));
            sample_pos[w].select(1, 1).copy_(model.batch_pred);
            sample_pos[w].select(1, 2).copy_(model.batch_xy.select(1, 1));
            mse_now = err.mean().item<float>();                 // syncs
        }
        if (cuda) torch::cuda::synchronize();                   // writes done BEFORE publish
#if defined(__APPLE__)
        else if (mps) caliper::adapters::detail::mps_synchronize_serialized();
#endif
        int next_write;
        {
            std::lock_guard<std::mutex> lk(st->mtx);
            st->grid_mse   = mse_now;
            st->ready_slot = w;
            next_write = 0;
            for (int i = 0; i < kSlots; ++i)
                if (i != st->ready_slot && i != st->display_slot) { next_write = i; break; }
        }
        // Fallback err^2 grid, ~4 Hz (cheap: 5184 floats DtoH).
        static int64_t last_err = -1000;
        if (step - last_err >= 8) {
            last_err = step;
            auto cpu = attr[w].to(torch::kCPU).contiguous();
            const float* ep = cpu.data_ptr<float>();
            std::lock_guard<std::mutex> lk(st->mtx);
            st->err_grid.assign(ep, ep + N);
            st->err_gen++;
        }
        return next_write;
    };

    // Hand the slots to the state under the mutex before the first publish.
    {
        std::lock_guard<std::mutex> lk(st->mtx);
        st->pool = std::move(pool);
        for (int i = 0; i < kSlots; ++i) {
            st->pos[i] = pos[i]; st->normal[i] = normal[i];
            st->attr[i] = attr[i]; st->sample_pos[i] = sample_pos[i];
        }
        st->tri_idx = tri_idx; st->line_idx = line_idx;
        st->tri_index_count = nTi; st->line_index_count = nLi;
        st->vertex_count = N; st->on_gpu = gpu;
    }

    int write = 0;
    int64_t step = 0;
    write = publish(write, step);   // untrained surface appears immediately

    while (!st->stop.load() && !(ctl && ctl->cancelled(ctl))) {
        if (st->reset_req.exchange(false)) {
            model.reset(kResetSeed);
            std::lock_guard<std::mutex> lk(st->mtx);
            st->steps_trained = 0;
            st->last_loss = 0.f;
        }

        // Drain the paint queue BEFORE train_step: the single-writer of the
        // target grid is always the worker (the frame thread only enqueues).
        if (st->reset_target_req.exchange(false)) model.target.reset_preset();
        {
            std::vector<MeshScopeState::Stroke> pending;
            {
                std::lock_guard<std::mutex> lk(st->mtx);
                pending.swap(st->strokes);
            }
            for (const auto& s : pending)
                model.target.brush(s.cx, s.cy, s.radius, s.amp);
        }

        model.set_lr(st->lr.load());

        if (st->train_on.load()) {
            const float lv = model.train_step();
            std::lock_guard<std::mutex> lk(st->mtx);
            st->last_loss = lv;
            ++st->steps_trained;
        } else {
            // Policy runs (inference) even when training is off: the surface
            // holds still, so avoid busy-spinning the publish loop.
            std::this_thread::sleep_for(std::chrono::milliseconds(16));
        }
        ++step;
        write = publish(write, step);
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
}

void mesh_job_tramp(void* user, const CaliperJobControl* ctl) {
    mesh_job(static_cast<MeshScopeState*>(user), ctl);
}

}  // namespace

MeshScopeApplet::MeshScopeApplet() : s_(std::make_unique<MeshScopeState>()) {}
MeshScopeApplet::~MeshScopeApplet() = default;

bool MeshScopeApplet::initialize(caliper::Host& host) {
    s_->host     = &host;
    s_->jobs     = caliper::Jobs(host);
    s_->device   = caliper::Device::query(host);
    s_->bridge   = caliper::Bridge(host);
    s_->geometry = caliper::Geometry(host);
    s_->geom_caps = s_->geometry.caps();
    host.log_info("mesh-scope: on_init");
    s_->job_id = s_->jobs.submit("mesh_scope: train", &mesh_job_tramp, s_.get());
    return true;
}

void MeshScopeApplet::draw_ui() {
    auto* st = s_.get();

    // ---- snapshot worker-published state under the mutex ----
    torch::Tensor draw_pos, draw_normal, draw_attr, draw_sample, tri_idx, line_idx;
    int64_t n = 0, nTi = 0, nLi = 0, nS = 0, steps = 0;
    bool gpu = false;
    float loss = 0.f, mse = 0.f;
    caliper::adapters::ExportablePool* pool = nullptr;
    std::vector<float> err_grid;
    {
        std::lock_guard<std::mutex> lk(st->mtx);
        if (st->ready_slot >= 0) {
            st->display_slot = st->ready_slot;
            draw_pos    = st->pos[st->display_slot];
            draw_normal = st->normal[st->display_slot];
            draw_attr   = st->attr[st->display_slot];
            draw_sample = st->sample_pos[st->display_slot];
        }
        tri_idx = st->tri_idx; line_idx = st->line_idx;
        n = st->vertex_count; nTi = st->tri_index_count; nLi = st->line_index_count;
        nS = st->sample_count;
        gpu = st->on_gpu; loss = st->last_loss; mse = st->grid_mse;
        steps = st->steps_trained;
        pool = st->pool.get();
        err_grid = st->err_grid;
    }

    // Give the window room on first use so the 3-D view can bootstrap: without
    // a view there is no image, so an auto-sizing floating window collapses to
    // the HUD height and never leaves >=64px for create_view_ex — exactly what
    // happens under embed_host, where no dock layout sizes us (instance_scope
    // precedent; docked layouts override this harmlessly).
    ImGui::SetNextWindowSize(ImVec2(1120, 860), ImGuiCond_FirstUseEver);
    ImGui::Begin("MeshScope: Surface");

    // ---- toolbar: controls + honest status (last frame's provenance) ----
    const float bar_h = ImGui::GetFrameHeight() + ImGui::GetStyle().WindowPadding.y * 2.f;
    if (ImGui::BeginChild("##toolbar", ImVec2(0, bar_h), ImGuiChildFlags_Borders)) {
        bool train = st->train_on.load();
        if (ImGui::Checkbox("train", &train)) st->train_on.store(train);
        ImGui::SameLine();
        if (ImGui::Button("reset")) st->reset_req.store(true);
        ImGui::SameLine();
        float lr = st->lr.load();
        ImGui::SetNextItemWidth(120);
        if (ImGui::SliderFloat("lr", &lr, 1e-4f, 1e-2f, "%.4f",
                               ImGuiSliderFlags_Logarithmic))
            st->lr.store(lr);
        ImGui::SameLine();
        ImGui::SetNextItemWidth(110);
        ImGui::SliderFloat("err vmax", &st->color_vmax, 0.005f, 0.2f, "%.3f");
        ImGui::SameLine();
        ImGui::SetNextItemWidth(110);
        ImGui::SliderFloat("brush", &st->brush_radius, 0.08f, 1.0f, "%.2f");
        ImGui::SameLine();
        ImGui::SetNextItemWidth(110);
        ImGui::SliderFloat("strength", &st->brush_strength, 0.1f, 2.0f, "%.2f");
        ImGui::SameLine();
        if (ImGui::Button("reset target")) st->reset_target_req.store(true);
        ImGui::SameLine();
        ImGui::TextDisabled("|");
        ImGui::SameLine();
        if (st->zero_copy_frame)
            ImGui::TextColored({0.55f, 0.9f, 0.6f, 1.f},
                "zero-copy (imported geometry) · step %lld · loss %.4f · grid-MSE %.4f",
                (long long)steps, loss, mse);
        else
            ImGui::TextColored({1.f, 0.7f, 0.4f, 1.f},
                "fallback: %s · step %lld · loss %.4f · grid-MSE %.4f",
                st->frame_status, (long long)steps, loss, mse);
        ImGui::SameLine();
        ImGui::TextDisabled("   (left-drag: paint · alt: lower · right-drag: orbit · wheel: zoom)");
    }
    ImGui::EndChild();

    // ---- the 3-D view fills all remaining space (flow_scope DPI discipline) ----
    const ImVec2 avail = ImGui::GetContentRegionAvail();
    const bool geom_live = st->geometry.has_primitives();

    const float fb_scale = ImGui::GetIO().DisplayFramebufferScale.y > 0.f
                               ? ImGui::GetIO().DisplayFramebufferScale.y : 1.f;
    auto clampi = [](int v, int lo, int hi) { return v < lo ? lo : (v > hi ? hi : v); };
    const int dw = clampi((int)(avail.x * fb_scale), 64, 4096);
    const int dh = clampi((int)(avail.y * fb_scale), 64, 4096);
    if (geom_live && avail.x >= 64 && avail.y >= 64 &&
        (st->view == 0 || std::abs(dw - st->view_w) >= 3 ||
         std::abs(dh - st->view_h) >= 3)) {
        if (st->view != 0) st->geometry.release_view(st->view);
        st->view = st->geometry.create_view_ex((uint32_t)dw, (uint32_t)dh,
                                               CALIPER_GEOM_VIEW_DEPTH);
        st->view_w = dw; st->view_h = dh;
    }

    st->zero_copy_frame = false;
    st->frame_status =
        !gpu ? "torch CPU"
             : !geom_live ? "no geometry.v1_1 backend"
             : !pool ? "pool unavailable"
             : st->view == 0 ? "no geometry view"
             : !draw_pos.defined() ? "no surface yet"
                                   : "not drawn";

    if (geom_live && st->view != 0 && pool && draw_pos.defined()) {
        const float ce = std::cos(st->cam_el), se = std::sin(st->cam_el);
        const float ca = std::cos(st->cam_az), sa = std::sin(st->cam_az);
        const V3 eye{st->cam_dist * ce * ca, st->cam_dist * se,
                     st->cam_dist * ce * sa};
        CaliperGeomCamera cam{};
        look_at(eye, {0, 0, 0}, {0, 1, 0}, cam.view);
        perspective(45.f * kPi / 180.f,
                    (float)st->view_w / (float)st->view_h, 0.05f, 50.f, cam.proj);

        // import-once per pool block, cached; offsets thereafter.
        auto pref = pool->to_bridge(st->bridge, draw_pos);
        auto nref = pool->to_bridge(st->bridge, draw_normal);
        auto aref = pool->to_bridge(st->bridge, draw_attr);
        auto sref = pool->to_bridge(st->bridge, draw_sample);
        auto tref = pool->to_bridge(st->bridge, tri_idx);
        auto lref = pool->to_bridge(st->bridge, line_idx);
        if (pref && nref && aref && sref && tref && lref) {
            // Draw 0: the learned surface, indexed triangles, MAGMA err^2,
            // Lambert-lit from the finite-difference normals, opaque, depth R/W.
            CaliperGeomDraw surf = caliper::geom_draw_defaults();
            surf.pos_alloc = pref->alloc;       surf.pos_offset = pref->offset;
            surf.vertex_count = (uint64_t)n;
            surf.index_alloc = tref->alloc;     surf.index_offset = tref->offset;
            surf.index_count = (uint64_t)nTi;
            surf.normal_alloc = nref->alloc;    surf.normal_offset = nref->offset;
            surf.attr_alloc = aref->alloc;      surf.attr_offset = aref->offset;
            surf.topology   = CALIPER_GEOM_TOPO_TRIANGLES;
            surf.color_mode = CALIPER_GEOM_COLOR_COLORMAP;
            surf.shade_mode = CALIPER_GEOM_SHADE_LAMBERT;
            surf.blend_mode = CALIPER_GEOM_BLEND_OPAQUE;
            surf.depth_flags = CALIPER_GEOM_DEPTH_TEST | CALIPER_GEOM_DEPTH_WRITE;
            surf.colormap = CALIPER_CMAP_MAGMA;
            surf.vmin = 0.0f;  surf.vmax = st->color_vmax;

            // Draw 1: the coplanar wireframe overlay, indexed lines, flat white
            // at low alpha, depth-tested only (LESS_OR_EQUAL lets edges win).
            CaliperGeomDraw wire = caliper::geom_draw_defaults();
            wire.pos_alloc = pref->alloc;       wire.pos_offset = pref->offset;
            wire.vertex_count = (uint64_t)n;
            wire.index_alloc = lref->alloc;     wire.index_offset = lref->offset;
            wire.index_count = (uint64_t)nLi;
            wire.topology   = CALIPER_GEOM_TOPO_LINES;
            wire.color_mode = CALIPER_GEOM_COLOR_FLAT;
            wire.shade_mode = CALIPER_GEOM_SHADE_UNLIT;
            wire.blend_mode = CALIPER_GEOM_BLEND_ALPHA;
            wire.depth_flags = CALIPER_GEOM_DEPTH_TEST;
            wire.flat_rgba = 0x59ffffffu;   // white, alpha ~0.35

            // Draw 2: the training minibatch at (x, f_θ(x,y), y), additive amber
            // points — "where is supervision" over the surface.
            CaliperGeomDraw pts = caliper::geom_draw_defaults();
            pts.pos_alloc = sref->alloc;        pts.pos_offset = sref->offset;
            pts.vertex_count = (uint64_t)nS;
            pts.topology   = CALIPER_GEOM_TOPO_POINTS;
            pts.color_mode = CALIPER_GEOM_COLOR_FLAT;
            pts.shade_mode = CALIPER_GEOM_SHADE_UNLIT;
            pts.blend_mode = CALIPER_GEOM_BLEND_ADDITIVE;
            pts.depth_flags = CALIPER_GEOM_DEPTH_TEST;
            pts.flat_rgba = 0xff33bfffu;    // amber
            pts.size_px = 3.0f * fb_scale;

            CaliperGeomDraw draws[3] = {surf, wire, pts};
            st->zero_copy_frame =
                st->geometry.draw_primitives(st->view, cam, draws, 3, 0xff05050au);
            if (!st->zero_copy_frame) st->frame_status = "draw_primitives refused";
            // One-shot provenance line: the log is the artifact that the
            // zero-copy path actually drew (host UI text can't be grepped).
            if (st->zero_copy_frame && !st->logged_first_draw && st->host) {
                st->host->log_info("mesh-scope: first zero-copy frame drawn "
                                   "(imported geometry, 3 draws)");
                st->logged_first_draw = true;
            }
        } else {
            st->frame_status =
                !pref ? "position import failed"
                      : !nref ? "normal import failed"
                              : !aref ? "attribute import failed"
                                      : !sref ? "sample import failed"
                                              : !tref ? "triangle-index import failed"
                                                      : "line-index import failed";
        }
    }

    if (st->zero_copy_frame) {
        ImGui::Image(caliper::Bridge::imtex(st->view),
                     ImVec2((float)st->view_w / fb_scale,
                            (float)st->view_h / fb_scale));
        const bool hovered = ImGui::IsItemHovered();
        const ImVec2 mn = ImGui::GetItemRectMin();
        const ImVec2 sz = ImGui::GetItemRectSize();
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
        // Left-drag paints the target: unproject the cursor through the SAME
        // view/proj the applet built above, intersect the base plane y=0, and
        // enqueue a brush at those domain coords. Frame thread never touches
        // torch — it only pushes a Stroke under the publish mutex.
        if (hovered && ImGui::IsMouseDown(ImGuiMouseButton_Left) &&
            sz.x > 0.f && sz.y > 0.f) {
            const float ce = std::cos(st->cam_el), se = std::sin(st->cam_el);
            const float ca = std::cos(st->cam_az), sa = std::sin(st->cam_az);
            const V3 eye{st->cam_dist * ce * ca, st->cam_dist * se,
                         st->cam_dist * ce * sa};
            const V3 fwd = norm3(V3{0, 0, 0} - eye);
            const V3 right = norm3(cross(fwd, {0, 1, 0}));
            const V3 up = cross(right, fwd);
            const float u = ((io.MousePos.x - mn.x) / sz.x) * 2.f - 1.f;
            const float v = 1.f - ((io.MousePos.y - mn.y) / sz.y) * 2.f;
            const float ta = std::tan(45.f * kPi / 360.f);
            const float aspect = (float)st->view_w / (float)st->view_h;
            const V3 dir = norm3({fwd.x + u * ta * aspect * right.x + v * ta * up.x,
                                  fwd.y + u * ta * aspect * right.y + v * ta * up.y,
                                  fwd.z + u * ta * aspect * right.z + v * ta * up.z});
            if (std::abs(dir.y) > 1e-6f) {
                const float t = -eye.y / dir.y;   // ray . plane y=0
                const float wx = eye.x + t * dir.x;   // world x == domain x
                const float wz = eye.z + t * dir.z;   // world z == domain y
                if (t > 0.f && std::abs(wx) <= kDomain && std::abs(wz) <= kDomain) {
                    const float sign = io.KeyAlt ? -1.f : 1.f;
                    const float amp = sign * st->brush_strength * io.DeltaTime;
                    std::lock_guard<std::mutex> lk(st->mtx);
                    st->strokes.push_back({wx, wz, st->brush_radius, amp});
                }
            }
        }
    } else {
        // Fallback ladder: the same per-vertex err^2, as an input-locked CPU
        // heatmap. Never a blank rectangle.
        char label[96];
        std::snprintf(label, sizeof(label), "fallback: %s — CPU heatmap",
                      st->frame_status);
        ImGui::TextDisabled("%s", label);
        if ((int64_t)err_grid.size() == (int64_t)kGrid * kGrid) {
            ImPlot::PushColormap(ImPlotColormap_Viridis);
            if (ImPlot::BeginPlot("##mesh_err_heatmap", ImVec2(-1, -1),
                                  ImPlotFlags_NoLegend | kLockedPlot)) {
                ImPlot::SetupAxes("x", "y",
                                  ImPlotAxisFlags_NoDecorations,
                                  ImPlotAxisFlags_NoDecorations);
                ImPlot::SetupAxisLimits(ImAxis_X1, -kDomain, kDomain, ImGuiCond_Always);
                ImPlot::SetupAxisLimits(ImAxis_Y1, -kDomain, kDomain, ImGuiCond_Always);
                ImPlot::PlotHeatmap("err^2", err_grid.data(), kGrid, kGrid,
                                    0.0, (double)st->color_vmax, nullptr,
                                    ImPlotPoint(-kDomain, -kDomain),
                                    ImPlotPoint(kDomain, kDomain));
                ImPlot::EndPlot();
            }
            ImPlot::PopColormap();
        } else {
            ImGui::TextDisabled("waiting for the first surface…");
        }
    }
    ImGui::End();
}

void MeshScopeApplet::cleanup() {
    auto* st = s_.get();
    st->stop.store(true);
    if (st->job_id != 0) {
        st->jobs.request_cancel(st->job_id);
        for (int i = 0; i < 2000 && st->jobs.is_running(st->job_id); ++i)
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    if (st->view != 0) { st->geometry.release_view(st->view); st->view = 0; }
    // Pool teardown mirrors flow_scope: drop pool-backed tensors first, then the
    // pool; leak deliberately if the worker somehow outlived the grace rather
    // than risk a use-after-free.
    {
        std::lock_guard<std::mutex> lk(st->mtx);
        for (int i = 0; i < kSlots; ++i) {
            st->pos[i] = torch::Tensor(); st->normal[i] = torch::Tensor();
            st->attr[i] = torch::Tensor(); st->sample_pos[i] = torch::Tensor();
        }
        st->tri_idx = torch::Tensor(); st->line_idx = torch::Tensor();
    }
    if (st->job_id != 0 && st->jobs.is_running(st->job_id)) {
        (void)st->pool.release();
        if (st->host)
            st->host->log_info("mesh-scope: worker still live at cleanup — "
                               "pool deliberately leaked");
    } else {
        st->pool.reset();
    }
    if (st->host) st->host->log_info("mesh-scope: on_cleanup");
}

}  // namespace meshscope
