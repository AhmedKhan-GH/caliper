// ============================================================================
// InstanceScope — the caliper.geometry.v1_3 instancing exemplar (id
// dev.caliper.instance-scope 0.1.0).
//
// ONE job: make instancing's value viscerally obvious. A field of N procedural
// gems bobs and spins in a traveling wave, drawn with ONE instanced draw call
// and ZERO copies of the mesh. Every frame the worker recomputes, ON DEVICE
// (torch, MPS/CUDA/CPU like the other applets), a RIGID (N,16) f32 column-major
// pose tensor and a (N,) f32 sin-phase tint, drains the device, and publishes a
// triple-buffered slot; the frame thread imports both zero-copy (ExportablePool
// → bridge) and issues a single draw_primitives with the v1.3 instance tail.
//
// Discipline reused verbatim from mesh_scope/flow_scope: triple-buffered
// ExportablePool slots, the ready/display invariant, worker/frame threading
// through one publish mutex, the drain-before-publish memory-stability contract
// (geometry_v1.h §TEMPORAL), applet-owned orbit camera math, DPI-correct view
// sizing. The tint rides a FIXED [-1,1] MAGMA window (analytic sin — never
// saturates, the TwinScope lesson). The status line claims zero-copy only when
// the instanced draw actually drew this frame (flow_scope provenance).
//
// Honest ladder: has_instanced() false → "instancing unavailable on this
// backend" + draw ONE gem via the non-instanced path; no geometry/pool/GPU →
// a text notice (there is nothing to plot — the whole point is the 3-D field).
// ============================================================================
#include "instance_scope.h"
#include "instance_field.h"

#include <caliper/caliper.hpp>
#include <caliper/adapters/exportable_pool.hpp>
#include <caliper/adapters/torch.hpp>
#include <imgui.h>
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

namespace instancescope {
namespace {

constexpr int      kSlots   = 3;       // triple buffer: write / ready / displayed
constexpr int      kNmax    = 5000;    // slider ceiling; slots sized once to this
constexpr int      kNdefault= 1000;
constexpr float    kSpacing = 2.2f;    // grid neighbor spacing (world units)

// ---- tiny column-major mat4 helpers (verbatim from mesh_scope) -------------
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

}  // namespace

// ---------------------------------------------------------------------------
// pImpl state. Cross-thread fields live under `mtx`; the frame-thread-only
// block at the bottom never locks.
// ---------------------------------------------------------------------------
struct InstanceScopeState {
    caliper::Host*    host = nullptr;
    caliper::Jobs     jobs;
    caliper::Device   device;
    caliper::Bridge   bridge;
    caliper::Geometry geometry;
    uint32_t geom_caps = 0;

    uint64_t job_id = 0;
    std::atomic<bool> stop{false};

    // frame -> worker knobs (atomics; torn reads harmless).
    std::atomic<int>  req_n{kNdefault};
    std::atomic<bool> paused{false};

    std::mutex mtx;  // guards everything below, down to the frame block

    // Static gem mesh (pool-born, imported once) + triple-buffered instance
    // pose/tint slots (pool-born). All the renderer reads.
    std::unique_ptr<caliper::adapters::ExportablePool> pool;
    torch::Tensor mesh_pos, mesh_nrm, mesh_idx;   // static
    torch::Tensor pose[kSlots], tint[kSlots];     // (kNmax,16) / (kNmax,)
    int  ready_slot   = -1;
    int  display_slot = -1;
    int64_t mesh_vtx = 0;
    int64_t mesh_idx_count = 0;
    bool on_gpu = false;

    // published scalars for the HUD / camera.
    int   cur_n = kNdefault;
    float grid_extent = kSpacing * 32.0f;
    double sim_time = 0.0;

    // ------- frame-thread-only -------
    CaliperTextureId view = 0;
    int   view_w = 900, view_h = 720;
    float cam_az = 0.7f, cam_el = 0.62f, cam_zoom = 0.85f;
    bool  zero_copy_frame = false;
    bool  logged_first_draw = false;
    const char* frame_status = "initializing";
};

namespace {

// Write instance pose+tint rows [0,n) of slot `w` on device, from the analytic
// wave at sim time `t`. Mirrors instance_field.h::pose_matrix / tint_signal
// exactly, vectorized. cx,cz are the (n,) grid-center columns for this n.
void build_slot(const torch::Tensor& pose_w, const torch::Tensor& tint_w,
                const torch::Tensor& cx, const torch::Tensor& cz, int64_t n,
                double t, const WaveParams& wp) {
    torch::NoGradGuard ng;
    auto ph  = (cx + cz) * wp.k - (wp.omega * (float)t);   // (n,)
    auto s   = torch::sin(ph);
    auto cph = torch::cos(ph);
    const float ca = std::cos(wp.tilt), sa = std::sin(wp.tilt);
    auto dst = pose_w.narrow(0, 0, n);                     // (n,16) column-major
    dst.select(1, 0).copy_(cph);       dst.select(1, 1).zero_();
    dst.select(1, 2).copy_(-s);        dst.select(1, 3).zero_();
    dst.select(1, 4).copy_(s * sa);    dst.select(1, 5).fill_(ca);
    dst.select(1, 6).copy_(cph * sa);  dst.select(1, 7).zero_();
    dst.select(1, 8).copy_(s * ca);    dst.select(1, 9).fill_(-sa);
    dst.select(1, 10).copy_(cph * ca); dst.select(1, 11).zero_();
    dst.select(1, 12).copy_(cx);
    dst.select(1, 13).copy_(wp.amp * s);
    dst.select(1, 14).copy_(cz);
    dst.select(1, 15).fill_(1.0f);
    tint_w.narrow(0, 0, n).copy_(s);
}

// ---- the worker: build the mesh + slots, then animate + publish forever ----
void instance_job(InstanceScopeState* st, const CaliperJobControl* ctl) {
    torch::NoGradGuard ng;
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

    // Zero-copy opt-in, decided once: geometry primitives caps + a GPU device.
    std::unique_ptr<caliper::adapters::ExportablePool> pool;
    if (gpu && (st->geom_caps & CALIPER_GEOM_CAP_PRIMITIVES)) {
        try {
            auto p = std::make_unique<caliper::adapters::ExportablePool>(0);
            if (p->ok()) pool = std::move(p);
        } catch (...) { /* pool absent -> fallback, never a crash */ }
    }
    if (st->host) {
        if (pool)
            st->host->log_info(cuda ? "instance-scope: zero-copy pool ready (cuda)"
                                    : "instance-scope: zero-copy pool ready (mps)");
        else
            st->host->log_info(
                !gpu ? "instance-scope: fallback (torch CPU, no zero-copy)"
                     : !(st->geom_caps & CALIPER_GEOM_CAP_PRIMITIVES)
                           ? "instance-scope: fallback (no geometry backend)"
                           : "instance-scope: fallback (pool unavailable)");
    }

    const WaveParams wp;
    const Mesh gem = gem_mesh();
    const int64_t V = gem.vertex_count();
    const int64_t I = gem.index_count();

    auto opt_f = torch::TensorOptions(dev).dtype(torch::kFloat32);
    auto opt_i = torch::TensorOptions(dev).dtype(torch::kInt32);

    // Mesh (static) + instance slots, pool-born when we have a pool.
    torch::Tensor mesh_pos, mesh_nrm, mesh_idx;
    torch::Tensor pose[kSlots], tint[kSlots];
    auto alloc_all = [&] {
        mesh_pos = torch::empty({V, 3}, opt_f);
        mesh_nrm = torch::empty({V, 3}, opt_f);
        mesh_idx = torch::empty({I}, opt_i);
        for (int i = 0; i < kSlots; ++i) {
            pose[i] = torch::empty({(int64_t)kNmax, 16}, opt_f);
            tint[i] = torch::empty({(int64_t)kNmax}, opt_f);
        }
    };
    if (pool) { auto scope = pool->use(); alloc_all(); }
    else      { alloc_all(); }

    mesh_pos.copy_(torch::from_blob((void*)gem.pos.data(), {V, 3},
                   torch::TensorOptions().dtype(torch::kFloat32)).clone().to(dev));
    mesh_nrm.copy_(torch::from_blob((void*)gem.normal.data(), {V, 3},
                   torch::TensorOptions().dtype(torch::kFloat32)).clone().to(dev));
    mesh_idx.copy_(torch::from_blob((void*)gem.index.data(), {I},
                   torch::TensorOptions().dtype(torch::kInt32)).clone().to(dev));

    // (n,) grid-center columns, rebuilt on CPU only when N changes (cheap;
    // the PER-FRAME work — phases, matrices, tint — stays on device).
    int cur_n = 0;
    torch::Tensor cx, cz;
    auto rebuild_grid = [&](int n) {
        std::vector<float> hx(n), hz(n);
        for (int i = 0; i < n; ++i) {
            const Vec3 c = grid_center(i, n, kSpacing);
            hx[i] = c.x; hz[i] = c.z;
        }
        cx = torch::from_blob(hx.data(), {(int64_t)n},
                 torch::TensorOptions().dtype(torch::kFloat32)).clone().to(dev);
        cz = torch::from_blob(hz.data(), {(int64_t)n},
                 torch::TensorOptions().dtype(torch::kFloat32)).clone().to(dev);
        cur_n = n;
    };

    auto drain = [&] {
        if (cuda) torch::cuda::synchronize();
#if defined(__APPLE__)
        else if (mps) caliper::adapters::detail::mps_synchronize_serialized();
#endif
    };

    // Publish helper: write slot `w`, device-sync, flip ready_slot under the
    // one mutex (the drain-before-publish memory-stability contract).
    auto publish = [&](int w, int n, double t) {
        build_slot(pose[w], tint[w], cx, cz, n, t, wp);
        drain();                                  // writes done BEFORE publish
        const GridDims gd = grid_dims(n);
        const float extent = (float)std::max(gd.cols, gd.rows) * kSpacing;
        int next_write;
        {
            std::lock_guard<std::mutex> lk(st->mtx);
            st->ready_slot = w;
            st->cur_n = n;
            st->grid_extent = extent;
            st->sim_time = t;
            next_write = 0;
            for (int i = 0; i < kSlots; ++i)
                if (i != st->ready_slot && i != st->display_slot) { next_write = i; break; }
        }
        return next_write;
    };

    // Hand the static tensors to the state before the first publish.
    {
        std::lock_guard<std::mutex> lk(st->mtx);
        st->pool = std::move(pool);
        st->mesh_pos = mesh_pos; st->mesh_nrm = mesh_nrm; st->mesh_idx = mesh_idx;
        for (int i = 0; i < kSlots; ++i) { st->pose[i] = pose[i]; st->tint[i] = tint[i]; }
        st->mesh_vtx = V; st->mesh_idx_count = I; st->on_gpu = gpu;
    }

    rebuild_grid(std::clamp(st->req_n.load(), 1, kNmax));
    int  write = 0;
    double t   = 0.0;
    write = publish(write, cur_n, t);           // first field appears immediately

    auto last = std::chrono::steady_clock::now();
    while (!st->stop.load() && !(ctl && ctl->cancelled(ctl))) {
        const auto now = std::chrono::steady_clock::now();
        const double dt = std::chrono::duration<double>(now - last).count();
        last = now;

        const int want_n = std::clamp(st->req_n.load(), 1, kNmax);
        const bool grid_changed = (want_n != cur_n);
        if (grid_changed) rebuild_grid(want_n);

        const bool paused = st->paused.load();
        if (!paused) t += dt;

        // Republish when the animation advances OR the field size changed; when
        // fully paused and unchanged, idle instead of spinning the publish loop.
        if (!paused || grid_changed) {
            write = publish(write, cur_n, t);
        } else {
            std::this_thread::sleep_for(std::chrono::milliseconds(16));
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
}

void instance_job_tramp(void* user, const CaliperJobControl* ctl) {
    instance_job(static_cast<InstanceScopeState*>(user), ctl);
}

}  // namespace

InstanceScopeApplet::InstanceScopeApplet() : s_(std::make_unique<InstanceScopeState>()) {}
InstanceScopeApplet::~InstanceScopeApplet() = default;

bool InstanceScopeApplet::initialize(caliper::Host& host) {
    s_->host     = &host;
    s_->jobs     = caliper::Jobs(host);
    s_->device   = caliper::Device::query(host);
    s_->bridge   = caliper::Bridge(host);
    s_->geometry = caliper::Geometry(host);
    s_->geom_caps = s_->geometry.caps();
    host.log_info("instance-scope: on_init");
    s_->job_id = s_->jobs.submit("instance_scope: animate", &instance_job_tramp, s_.get());
    return true;
}

void InstanceScopeApplet::draw_ui() {
    auto* st = s_.get();

    // ---- snapshot worker-published state under the mutex ----
    torch::Tensor draw_pose, draw_tint, mesh_pos, mesh_nrm, mesh_idx;
    int64_t V = 0, I = 0;
    int n = 0;
    bool gpu = false;
    float extent = st->grid_extent;
    caliper::adapters::ExportablePool* pool = nullptr;
    {
        std::lock_guard<std::mutex> lk(st->mtx);
        if (st->ready_slot >= 0) {
            st->display_slot = st->ready_slot;
            draw_pose = st->pose[st->display_slot];
            draw_tint = st->tint[st->display_slot];
        }
        mesh_pos = st->mesh_pos; mesh_nrm = st->mesh_nrm; mesh_idx = st->mesh_idx;
        V = st->mesh_vtx; I = st->mesh_idx_count;
        n = st->cur_n; gpu = st->on_gpu; extent = st->grid_extent;
        pool = st->pool.get();
    }

    const bool instanced = st->geometry.has_instanced();
    const bool geom_live = st->geometry.has_primitives();

    // Give the window room on first use so the 3-D view can bootstrap: without a
    // view there is no image, so an auto-sizing floating window would collapse to
    // the HUD height and never leave >=64px for create_view (docked layouts
    // override this harmlessly).
    ImGui::SetNextWindowSize(ImVec2(1120, 860), ImGuiCond_FirstUseEver);
    ImGui::Begin("InstanceScope");

    // ---- the hero pitch: BIG, front and center ----
    ImGui::SetWindowFontScale(1.9f);
    ImGui::TextColored({0.98f, 0.86f, 0.55f, 1.0f},
                       "%d objects   \xC2\xB7   1 draw call   \xC2\xB7   0 copies of the mesh", n);
    ImGui::SetWindowFontScale(1.0f);
    ImGui::Spacing();

    // ---- controls: N slider, pause, live FPS ----
    int req_n = st->req_n.load();
    ImGui::SetNextItemWidth(360);
    if (ImGui::SliderInt("N (objects)", &req_n, 1, kNmax))
        st->req_n.store(std::clamp(req_n, 1, kNmax));
    ImGui::SameLine();
    bool paused = st->paused.load();
    if (ImGui::Button(paused ? "Resume" : "Pause")) st->paused.store(!paused);
    ImGui::SameLine();
    ImGui::Text("   %.0f FPS", ImGui::GetIO().Framerate);
    ImGui::SameLine();
    ImGui::SetNextItemWidth(120);
    ImGui::SliderFloat("zoom", &st->cam_zoom, 0.35f, 2.2f, "%.2f");

    // ---- honest provenance (last frame's actual path) ----
    if (st->zero_copy_frame)
        ImGui::TextColored({0.55f, 0.9f, 0.6f, 1.0f},
            "zero-copy instanced draw \xC2\xB7 %d gems, 1 draw_primitives call, "
            "0 mesh copies \xC2\xB7 device: %s", n, st->device.name);
    else if (!instanced && geom_live)
        ImGui::TextColored({1.0f, 0.7f, 0.4f, 1.0f},
            "instancing unavailable on this backend \xE2\x80\x94 drawing ONE gem "
            "(non-instanced path)");
    else
        ImGui::TextColored({1.0f, 0.7f, 0.4f, 1.0f},
            "fallback: %s", st->frame_status);
    ImGui::TextDisabled("   (right-drag: orbit \xC2\xB7 wheel/zoom: dolly \xC2\xB7 "
                        "rigid poses only \xE2\x80\x94 G14 never refuses)");

    // ---- the 3-D view fills all remaining space (mesh_scope DPI discipline) ----
    const ImVec2 avail = ImGui::GetContentRegionAvail();
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
        !gpu ? "torch CPU (no zero-copy import)"
             : !geom_live ? "no geometry backend"
             : !pool ? "pool unavailable"
             : st->view == 0 ? "no geometry view"
             : !draw_pose.defined() ? "no field yet"
                                    : "not drawn";

    if (geom_live && st->view != 0 && pool && draw_pose.defined() && mesh_pos.defined()) {
        // camera: orbit at a distance that frames the whole field.
        const float dist = std::max(extent * 1.25f * st->cam_zoom, 4.0f);
        const float ce = std::cos(st->cam_el), se = std::sin(st->cam_el);
        const float ca = std::cos(st->cam_az), sa = std::sin(st->cam_az);
        const V3 eye{dist * ce * ca, dist * se, dist * ce * sa};
        CaliperGeomCamera cam{};
        look_at(eye, {0, 0, 0}, {0, 1, 0}, cam.view);
        perspective(45.f * kPi / 180.f, (float)st->view_w / (float)st->view_h,
                    0.1f, std::max(extent * 4.0f, 100.0f), cam.proj);

        // import-once per pool block, cached; offsets thereafter.
        auto pr = pool->to_bridge(st->bridge, mesh_pos);
        auto nr = pool->to_bridge(st->bridge, mesh_nrm);
        auto ir = pool->to_bridge(st->bridge, mesh_idx);
        auto Pr = pool->to_bridge(st->bridge, draw_pose);
        auto Tr = pool->to_bridge(st->bridge, draw_tint);
        if (pr && nr && ir && Pr && Tr) {
            if (instanced) {
                // THE hero: one instanced draw. Base record carries the shared
                // mesh (LAMBERT + OPAQUE + depth); the v1.3 tail carries the
                // per-instance rigid poses and the sin-phase tint. color_mode is
                // FLAT — the instance tint carries its own MAGMA LUT (§3.3), so
                // no per-vertex attr is needed; the fixed [-1,1] window can never
                // saturate (analytic sin).
                CaliperGeomDrawV1_3 d = caliper::geom_draw_v1_3_defaults();
                auto& b = d.base.base;
                b.pos_alloc = pr->alloc;    b.pos_offset = pr->offset;
                b.vertex_count = (uint64_t)V;
                b.index_alloc = ir->alloc;  b.index_offset = ir->offset;
                b.index_count = (uint64_t)I;
                b.normal_alloc = nr->alloc; b.normal_offset = nr->offset;
                b.topology   = CALIPER_GEOM_TOPO_TRIANGLES;
                b.color_mode = CALIPER_GEOM_COLOR_FLAT;
                b.shade_mode = CALIPER_GEOM_SHADE_LAMBERT;
                b.blend_mode = CALIPER_GEOM_BLEND_OPAQUE;
                b.depth_flags = CALIPER_GEOM_DEPTH_TEST | CALIPER_GEOM_DEPTH_WRITE;
                b.colormap = CALIPER_CMAP_MAGMA;   // resolves the tint LUT (G12)
                b.vmin = -1.0f; b.vmax = 1.0f;
                d.instance_alloc = Pr->alloc;  d.instance_offset = Pr->offset;
                d.instance_count = (uint64_t)n;
                d.instance_attr_alloc = Tr->alloc; d.instance_attr_offset = Tr->offset;

                st->zero_copy_frame =
                    st->geometry.draw_primitives(st->view, cam, &d, 1, 0xff0a0810u);
                if (!st->zero_copy_frame) st->frame_status = "instanced draw refused (G14?)";
                if (st->zero_copy_frame && !st->logged_first_draw && st->host) {
                    char msg[160];
                    std::snprintf(msg, sizeof(msg),
                        "instance-scope: first zero-copy instanced frame drawn "
                        "\xE2\x80\x94 %d objects, 1 draw call, 0 mesh copies", n);
                    st->host->log_info(msg);
                    st->logged_first_draw = true;
                }
            } else {
                // Honest ladder: no instancing — draw ONE gem via the v1.1
                // non-instanced path, at instance 0's live rigid pose.
                CaliperGeomDraw one = caliper::geom_draw_defaults();
                one.pos_alloc = pr->alloc;    one.pos_offset = pr->offset;
                one.vertex_count = (uint64_t)V;
                one.index_alloc = ir->alloc;  one.index_offset = ir->offset;
                one.index_count = (uint64_t)I;
                one.normal_alloc = nr->alloc; one.normal_offset = nr->offset;
                one.topology   = CALIPER_GEOM_TOPO_TRIANGLES;
                one.color_mode = CALIPER_GEOM_COLOR_FLAT;
                one.shade_mode = CALIPER_GEOM_SHADE_LAMBERT;
                one.blend_mode = CALIPER_GEOM_BLEND_OPAQUE;
                one.depth_flags = CALIPER_GEOM_DEPTH_TEST | CALIPER_GEOM_DEPTH_WRITE;
                one.flat_rgba = 0xff5fb0ffu;   // amber
                const auto m = pose_matrix(0, std::max(n, 1), kSpacing,
                                           st->sim_time, WaveParams{});
                std::memcpy(one.model, m.data(), sizeof(one.model));
                const bool drew =
                    st->geometry.draw_primitives(st->view, cam, &one, 1, 0xff0a0810u);
                if (!drew) st->frame_status = "single-gem draw refused";
                st->frame_status = drew ? "one gem (non-instanced)" : st->frame_status;
                // not zero_copy_frame: the hero claim is instanced-only.
            }
        } else {
            st->frame_status =
                !pr ? "mesh-position import failed"
                    : !nr ? "mesh-normal import failed"
                          : !ir ? "mesh-index import failed"
                                : !Pr ? "pose import failed" : "tint import failed";
        }
    }

    if (st->view != 0 && (st->zero_copy_frame ||
                          (geom_live && !instanced && draw_pose.defined()))) {
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
            st->cam_zoom *= (1.f - io.MouseWheel * 0.08f);
            st->cam_zoom = std::clamp(st->cam_zoom, 0.35f, 2.2f);
        }
    } else {
        ImGui::Dummy(ImVec2(1, 8));
        ImGui::TextDisabled("The whole point is the 3-D field; without the "
                            "geometry/zero-copy path there is nothing to show.");
        ImGui::TextDisabled("status: %s", st->frame_status);
    }
    ImGui::End();
}

void InstanceScopeApplet::cleanup() {
    auto* st = s_.get();
    st->stop.store(true);
    if (st->job_id != 0) {
        st->jobs.request_cancel(st->job_id);
        for (int i = 0; i < 2000 && st->jobs.is_running(st->job_id); ++i)
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    if (st->view != 0) { st->geometry.release_view(st->view); st->view = 0; }
    // Pool teardown mirrors mesh_scope: drop pool-backed tensors first, then the
    // pool; leak deliberately if the worker somehow outlived the grace rather
    // than risk a use-after-free.
    {
        std::lock_guard<std::mutex> lk(st->mtx);
        st->mesh_pos = torch::Tensor(); st->mesh_nrm = torch::Tensor();
        st->mesh_idx = torch::Tensor();
        for (int i = 0; i < kSlots; ++i) {
            st->pose[i] = torch::Tensor(); st->tint[i] = torch::Tensor();
        }
    }
    if (st->job_id != 0 && st->jobs.is_running(st->job_id)) {
        (void)st->pool.release();
        if (st->host)
            st->host->log_info("instance-scope: worker still live at cleanup — "
                               "pool deliberately leaked");
    } else {
        st->pool.reset();
    }
    if (st->host) st->host->log_info("instance-scope: on_cleanup");
}

}  // namespace instancescope
