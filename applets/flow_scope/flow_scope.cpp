// ============================================================================
// FlowScope — a million particles advected through an analytic curl field,
// drawn with ZERO copies of the point data (id dev.caliper.flow-scope 0.1.0).
//
// The digital-twin exemplar for caliper.geometry.v1 on top of bridge v1.2:
//   - the sim's position/speed tensors are born in the ExportablePool, so the
//     host imports each block ONCE and the point pass reads them in place;
//   - triple-buffered slots make the memory-stability contract literal — the
//     worker never writes the slot the (fenced) draw is reading;
//   - every miss (no geometry caps, no pool, CPU torch, GL renderer) falls
//     back to an ImPlot3D scatter of a 10k CPU subsample, honestly labeled.
//
// Interaction: right-drag orbits, wheel zooms, LEFT-drag pushes the field —
// the cursor ray becomes a radial impulse the sim applies as a force splat.
// Threading is the cookbook spine: one worker job steps the sim and publishes
// slots under one mutex; the frame thread snapshots, draws, and never blocks
// on the sim (the flip is the only contended instant).
// ============================================================================
#include "flow_scope.h"

#include <caliper/caliper.hpp>
#include <caliper/adapters/torch.hpp>
#include <caliper/adapters/exportable_pool.hpp>
#include <imgui.h>
#include <implot3d.h>
#include <torch/torch.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace flowscope {
namespace {

constexpr int   kSlots     = 3;      // triple buffer: write / ready / displayed
constexpr float kBoxL      = 1.5f;   // particles wrap in [-L, L)^3
constexpr int   kSubsample = 10000;  // CPU fallback scatter size
constexpr float kDt        = 1.0f / 120.0f;

// ---- tiny column-major mat4 helpers (no new applet dependencies) -----------
struct V3 { float x, y, z; };
V3 operator-(V3 a, V3 b) { return {a.x - b.x, a.y - b.y, a.z - b.z}; }
V3 cross(V3 a, V3 b) {
    return {a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x};
}
float dot(V3 a, V3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
V3 norm3(V3 a) {
    const float l = std::sqrt(dot(a, a));
    return l > 0 ? V3{a.x / l, a.y / l, a.z / l} : V3{0, 0, 1};
}

// Right-handed lookAt, column-major (GLSL layout).
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

// Vulkan-style perspective (z in [0,1]; +y-up NDC comes from the backend's
// negative-viewport convention, so no Y flip here).
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
struct FlowScopeState {
    caliper::Host*    host = nullptr;
    caliper::Jobs     jobs;
    caliper::Device   device;
    caliper::Bridge   bridge;    // v1.2 imports (through the pool's to_bridge)
    caliper::Geometry geometry;  // the new service
    uint32_t geom_caps = 0;      // snapshot at init (frame-thread-only wrappers)

    uint64_t job_id = 0;
    std::atomic<bool> stop{false};
    std::atomic<bool> paused{false};

    // frame -> worker parameters (atomics: torn reads are harmless here).
    std::atomic<float> field_strength{0.9f};
    std::atomic<float> noise_scale{2.2f};
    std::atomic<float> damping{0.6f};
    std::atomic<float> impulse_strength{6.0f};
    std::atomic<float> turbulence{0.03f};   // Langevin velocity noise (entropy)

    std::mutex mtx;  // guards everything below, down to the frame block
    // Impulse ray (frame writes, worker reads).
    bool  imp_active = false;
    float imp_o[3] = {0, 0, 0}, imp_d[3] = {0, 0, 1};

    // Triple-buffered sim outputs. Tensors are pool-born on the zero-copy
    // path (CUDA + pool ok) or plain device/CPU tensors otherwise.
    std::unique_ptr<caliper::adapters::ExportablePool> pool;
    torch::Tensor pos[kSlots], speed[kSlots];
    int  ready_slot   = -1;   // latest completed slot (worker publishes)
    int  display_slot = -1;   // slot the frame is drawing (worker avoids)
    int64_t n_particles = 0;
    bool  sim_on_cuda = false;
    float steps_per_sec = 0.f;

    // CPU fallback subsample (worker refreshes ~4 Hz when needed).
    std::vector<float> sub_x, sub_y, sub_z;
    uint64_t sub_gen = 0;

    // ------- frame-thread-only -------
    CaliperTextureId view = 0;
    int   view_w = 768, view_h = 768;
    float cam_az = 0.8f, cam_el = 0.45f, cam_dist = 5.5f;
    float color_vmax = 3.0f;   // wide window: speeds spread into magenta→orange
                               // instead of saturating the LUT's white end
    bool  zero_copy_frame = false;   // provenance of the current view content
    uint64_t frames = 0;
};

namespace {

// ---- the simulation step (worker thread, pure torch) -----------------------
// Divergence-free analytic field: each component is independent of its own
// axis, so div F = 0 — swirls without sinks. All heavy tensors preallocated;
// temporaries go to torch's default allocator (NOT the pool: the pool holds
// only what the renderer reads).
void sim_step(FlowScopeState* st, torch::Tensor& p_in, torch::Tensor& p_out,
              torch::Tensor& speed_out, torch::Tensor& vel, float t) {
    torch::NoGradGuard ng;
    const float k = st->noise_scale.load();
    const float A = st->field_strength.load();
    const float damp = st->damping.load();
    const float turb = st->turbulence.load();

    auto x = p_in.select(1, 0), y = p_in.select(1, 1), z = p_in.select(1, 2);
    const float tw = 0.6f * t;
    auto F = torch::stack({torch::sin(k * z + tw) + torch::cos(k * y + tw),
                           torch::sin(k * x + tw) + torch::cos(k * z + tw),
                           torch::sin(k * y + tw) + torch::cos(k * x + tw)}, 1)
                 .mul_(A);

    // Cursor impulse: radial push away from the mouse ray (a 3-D cursor wake).
    bool active = false;
    float o3[3], d3[3], strength = 0.f;
    {
        std::lock_guard<std::mutex> lk(st->mtx);
        active = st->imp_active;
        std::memcpy(o3, st->imp_o, sizeof(o3));
        std::memcpy(d3, st->imp_d, sizeof(d3));
    }
    if (active) {
        strength = st->impulse_strength.load();
        auto opt = torch::TensorOptions(p_in.device()).dtype(torch::kFloat32);
        auto o = torch::from_blob(o3, {1, 3}, torch::kFloat32).to(opt.device());
        auto d = torch::from_blob(d3, {1, 3}, torch::kFloat32).to(opt.device());
        auto a = p_in - o;
        auto tproj = (a * d).sum(1, true).clamp_min(0.f);
        auto r = p_in - (o + tproj * d);              // point -> ray offset
        auto dist2 = (r * r).sum(1, true);
        const float rad = 0.35f;
        F = F + r / (dist2.sqrt() + 1e-4f) *
                (torch::exp(dist2 / (-2.f * rad * rad)) * strength);
    }

    vel.add_(F, kDt).mul_(1.f - damp * kDt);
    // Langevin entropy: a per-step random velocity kick (temporary in the
    // default allocator, not the pool). Damping + this noise form an
    // Ornstein–Uhlenbeck process — the laminar field gains turbulence, and
    // since color follows speed the cloud gains color variation too.
    if (turb > 0.f) vel.add_(torch::randn_like(vel), turb);
    p_out.copy_(p_in).add_(vel, kDt);
    // Wrap into [-L, L): continuous flow, no respawn popping.
    p_out.sub_(torch::floor((p_out + kBoxL) / (2 * kBoxL)) * (2 * kBoxL));
    speed_out.copy_(vel.norm(2, {1}));
}

// ---- the worker job: allocate (pool-first), then step + publish forever ----
void sim_job(FlowScopeState* st, const CaliperJobControl* ctl) {
    torch::NoGradGuard ng;
    const bool cuda = torch::cuda::is_available();
    const torch::Device dev = cuda ? torch::Device(torch::kCUDA)
                                   : torch::Device(torch::kCPU);
    const int64_t N = cuda ? 1'000'000 : 50'000;

    // Zero-copy opt-in, decided once: geometry caps + import caps + CUDA.
    std::unique_ptr<caliper::adapters::ExportablePool> pool;
    if (cuda && (st->geom_caps & CALIPER_GEOM_CAP_IMPORTED_POINTS)) {
        try {
            auto p = std::make_unique<caliper::adapters::ExportablePool>(0);
            if (p->ok()) pool = std::move(p);
        } catch (...) { /* pool absent -> fallback path, never a crash */ }
    }

    // Slot tensors: inside the pool scope when we have one (they are ALL the
    // renderer reads); velocities and temporaries stay default-allocated.
    torch::Tensor pos[kSlots], speed[kSlots];
    {
        auto opt = torch::TensorOptions(dev).dtype(torch::kFloat32);
        auto alloc_slots = [&] {
            for (int i = 0; i < kSlots; ++i) {
                pos[i]   = torch::rand({N, 3}, opt) * 2.f - 1.f;
                speed[i] = torch::zeros({N}, opt);
            }
        };
        if (pool) { auto scope = pool->use(); alloc_slots(); }
        else      { alloc_slots(); }
    }
    torch::Tensor vel = torch::zeros({N, 3},
                                     torch::TensorOptions(dev).dtype(torch::kFloat32));
    {
        std::lock_guard<std::mutex> lk(st->mtx);
        st->pool = std::move(pool);
        for (int i = 0; i < kSlots; ++i) { st->pos[i] = pos[i]; st->speed[i] = speed[i]; }
        st->n_particles = N;
        st->sim_on_cuda = cuda;
        st->ready_slot  = 0;
    }

    int write = 1;
    float t = 0.f;
    int   steps = 0;
    uint64_t last_sub_step = 0;
    auto rate_t0 = std::chrono::steady_clock::now();
    int  rate_steps = 0;

    while (!st->stop.load() && !(ctl && ctl->cancelled(ctl))) {
        if (st->paused.load()) {
            std::this_thread::sleep_for(std::chrono::milliseconds(16));
            continue;
        }
        int read;
        {
            std::lock_guard<std::mutex> lk(st->mtx);
            read = st->ready_slot;
        }
        sim_step(st, pos[read], pos[write], speed[write], vel, t);
        if (cuda) torch::cuda::synchronize();   // writes done BEFORE publish
        t += kDt;
        ++steps;
        ++rate_steps;

        {
            std::lock_guard<std::mutex> lk(st->mtx);
            st->ready_slot = write;
            // Next write slot: neither the new ready nor whatever the frame
            // is displaying (the triple-buffer invariant that makes the
            // memory-stability contract literal).
            for (int i = 0; i < kSlots; ++i)
                if (i != st->ready_slot && i != st->display_slot) { write = i; break; }
        }

        // Fallback subsample, ~4 Hz (cheap: 10k floats DtoH).
        if (steps - last_sub_step >= 30) {
            last_sub_step = steps;
            auto sub = pos[read].narrow(0, 0, std::min<int64_t>(kSubsample, N))
                           .to(torch::kCPU).contiguous();
            const float* sp = sub.data_ptr<float>();
            const int64_t n = sub.size(0);
            std::lock_guard<std::mutex> lk(st->mtx);
            st->sub_x.resize((size_t)n); st->sub_y.resize((size_t)n);
            st->sub_z.resize((size_t)n);
            for (int64_t i = 0; i < n; ++i) {
                st->sub_x[(size_t)i] = sp[i * 3 + 0];
                st->sub_y[(size_t)i] = sp[i * 3 + 1];
                st->sub_z[(size_t)i] = sp[i * 3 + 2];
            }
            st->sub_gen++;
        }

        const auto now = std::chrono::steady_clock::now();
        const float el = std::chrono::duration<float>(now - rate_t0).count();
        if (el >= 0.5f) {
            std::lock_guard<std::mutex> lk(st->mtx);
            st->steps_per_sec = (float)rate_steps / el;
            rate_t0 = now;
            rate_steps = 0;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(2));
    }
}

void flow_job_tramp(void* user, const CaliperJobControl* ctl) {
    sim_job(static_cast<FlowScopeState*>(user), ctl);
}

}  // namespace

FlowScopeApplet::FlowScopeApplet() : s_(std::make_unique<FlowScopeState>()) {}
FlowScopeApplet::~FlowScopeApplet() = default;

bool FlowScopeApplet::initialize(caliper::Host& host) {
    s_->host     = &host;
    s_->jobs     = caliper::Jobs(host);
    s_->device   = caliper::Device::query(host);
    s_->bridge   = caliper::Bridge(host);
    s_->geometry = caliper::Geometry(host);
    s_->geom_caps = s_->geometry.caps();
    host.log_info("flow-scope: on_init");
    // The demo moves immediately — no button gate.
    s_->job_id = s_->jobs.submit("flow_scope: sim", &flow_job_tramp, s_.get());
    return true;
}

void FlowScopeApplet::draw_ui() {
    auto* st = s_.get();

    // ---- snapshot worker-published state under the mutex ----
    torch::Tensor draw_pos, draw_speed;
    int64_t n = 0;
    bool cuda = false;
    float sps = 0.f;
    caliper::adapters::ExportablePool* pool = nullptr;
    std::vector<float> sx, sy, sz;
    {
        std::lock_guard<std::mutex> lk(st->mtx);
        if (st->ready_slot >= 0) {
            st->display_slot = st->ready_slot;
            draw_pos   = st->pos[st->display_slot];
            draw_speed = st->speed[st->display_slot];
        }
        n = st->n_particles;
        cuda = st->sim_on_cuda;
        sps = st->steps_per_sec;
        pool = st->pool.get();
        sx = st->sub_x; sy = st->sub_y; sz = st->sub_z;
    }

    // No SetNextWindowSize: this window docks into the host's central node
    // (main.cpp central_windows list), so it fills the viewport work area.
    ImGui::Begin("FlowScope: Field");

    // ---- toolbar panel: controls + honest status (from last frame's draw) --
    // Fixed-height bordered child so the 3-D view below gets everything else.
    const float bar_h = ImGui::GetFrameHeight() + ImGui::GetStyle().WindowPadding.y * 2.f;
    if (ImGui::BeginChild("##toolbar", ImVec2(0, bar_h), ImGuiChildFlags_Borders)) {
        bool paused = st->paused.load();
        if (ImGui::Checkbox("pause", &paused)) st->paused.store(paused);
        ImGui::SameLine();
        float fs = st->field_strength.load();
        ImGui::SetNextItemWidth(100);
        if (ImGui::SliderFloat("field", &fs, 0.f, 2.5f)) st->field_strength.store(fs);
        ImGui::SameLine();
        float ns = st->noise_scale.load();
        ImGui::SetNextItemWidth(100);
        if (ImGui::SliderFloat("scale", &ns, 0.5f, 6.f)) st->noise_scale.store(ns);
        ImGui::SameLine();
        float dp = st->damping.load();
        ImGui::SetNextItemWidth(100);
        if (ImGui::SliderFloat("damping", &dp, 0.f, 2.f)) st->damping.store(dp);
        ImGui::SameLine();
        float tb = st->turbulence.load();
        ImGui::SetNextItemWidth(100);
        if (ImGui::SliderFloat("turb", &tb, 0.f, 0.12f, "%.3f"))
            st->turbulence.store(tb);
        ImGui::SameLine();
        ImGui::SetNextItemWidth(100);
        ImGui::SliderFloat("color", &st->color_vmax, 0.5f, 6.f);
        // Status reflects last frame's provenance (imperceptible 1-frame lag);
        // "zero-copy (imported geometry)" only when that path actually drew.
        ImGui::SameLine();
        ImGui::TextDisabled("|");
        ImGui::SameLine();
        if (st->zero_copy_frame)
            ImGui::TextColored({0.55f, 0.9f, 0.6f, 1.f},
                "%lld particles — zero-copy (imported geometry) · %.0f steps/s",
                (long long)n, sps);
        else
            ImGui::TextColored({1.f, 0.7f, 0.4f, 1.f},
                "%lld particles — CPU fallback (subsampled %d) · %.0f steps/s · %s",
                (long long)n, (int)sx.size(), sps,
                !cuda ? "torch CPU"
                      : (st->geom_caps ? "pool unavailable" : "no geometry service"));
        ImGui::SameLine();
        ImGui::TextDisabled("   (left-drag: push · right-drag: orbit · wheel: zoom)");
    }
    ImGui::EndChild();

    // ---- the 3-D view fills all remaining space ----
    const ImVec2 avail = ImGui::GetContentRegionAvail();
    const bool geom_live =
        st->geometry && (st->geom_caps & CALIPER_GEOM_CAP_IMPORTED_POINTS);

    // Size the offscreen view to the content region; recreate on real change
    // (a few-px threshold avoids reallocating on sub-pixel jitter). Clamp to a
    // sane range so a collapsed/huge dock node can't ask for a degenerate RT.
    auto clampi = [](int v, int lo, int hi) { return v < lo ? lo : (v > hi ? hi : v); };
    const int dw = clampi((int)avail.x, 64, 4096);
    const int dh = clampi((int)avail.y, 64, 4096);
    if (geom_live && avail.x >= 64 && avail.y >= 64 &&
        (st->view == 0 || std::abs(dw - st->view_w) >= 3 ||
         std::abs(dh - st->view_h) >= 3)) {
        if (st->view != 0) st->geometry.release_view(st->view);
        st->view = st->geometry.create_view((uint32_t)dw, (uint32_t)dh);
        st->view_w = dw; st->view_h = dh;
    }

    st->zero_copy_frame = false;
    if (geom_live && st->view != 0 && pool && draw_pos.defined()) {
        // Camera from the orbit state; aspect matches the live view size.
        const float ce = std::cos(st->cam_el), se = std::sin(st->cam_el);
        const float ca = std::cos(st->cam_az), sa = std::sin(st->cam_az);
        const V3 eye{st->cam_dist * ce * ca, st->cam_dist * se,
                     st->cam_dist * ce * sa};
        CaliperGeomCamera cam{};
        look_at(eye, {0, 0, 0}, {0, 1, 0}, cam.view);
        perspective(45.f * 3.14159265f / 180.f,
                    (float)st->view_w / (float)st->view_h, 0.05f, 50.f, cam.proj);

        // to_bridge: import-once per pool block, cached; offsets thereafter.
        auto pref = pool->to_bridge(st->bridge, draw_pos);
        auto sref = pool->to_bridge(st->bridge, draw_speed);
        if (pref && sref) {
            // Baseline color: a negative vmin lifts speed-0 off the magma
            // LUT's black floor to ~25% up (a dim magenta), so still
            // particles stay visible; faster particles climb to the bright
            // end. Purely the mapping window — no shader/ABI change.
            const float vmin = -0.33f * st->color_vmax;
            st->zero_copy_frame = st->geometry.draw_points(
                st->view, &cam, pref->alloc, pref->offset, (uint64_t)n,
                sref->alloc, sref->offset, CALIPER_CMAP_MAGMA, vmin,
                st->color_vmax, 1.5f, 0xFF000000u);
        }
    }

    if (st->zero_copy_frame) {
        // Fill the region with the offscreen view; interaction rides the image.
        ImGui::Image(caliper::Bridge::imtex(st->view),
                     ImVec2((float)st->view_w, (float)st->view_h));
        const bool hovered = ImGui::IsItemHovered();
        const ImVec2 mn = ImGui::GetItemRectMin();
        const ImVec2 sz = ImGui::GetItemRectSize();
        ImGuiIO& io = ImGui::GetIO();
        if (hovered && ImGui::IsMouseDown(ImGuiMouseButton_Right)) {
            st->cam_az += io.MouseDelta.x * 0.008f;
            st->cam_el += io.MouseDelta.y * 0.008f;
            if (st->cam_el > 1.5f) st->cam_el = 1.5f;
            if (st->cam_el < -1.5f) st->cam_el = -1.5f;
        }
        if (hovered && io.MouseWheel != 0.f) {
            st->cam_dist *= (1.f - io.MouseWheel * 0.08f);
            if (st->cam_dist < 1.2f) st->cam_dist = 1.2f;
            if (st->cam_dist > 12.f) st->cam_dist = 12.f;
        }
        const bool pushing = hovered && ImGui::IsMouseDown(ImGuiMouseButton_Left);
        {
            std::lock_guard<std::mutex> lk(st->mtx);
            st->imp_active = pushing;
            if (pushing && sz.x > 0.f && sz.y > 0.f) {
                // Cursor ray in world space from the SAME camera parameters.
                const float ce = std::cos(st->cam_el), se = std::sin(st->cam_el);
                const float ca = std::cos(st->cam_az), sa = std::sin(st->cam_az);
                const V3 eye{st->cam_dist * ce * ca, st->cam_dist * se,
                             st->cam_dist * ce * sa};
                const V3 fwd = norm3(V3{0, 0, 0} - eye);
                const V3 right = norm3(cross(fwd, {0, 1, 0}));
                const V3 up = cross(right, fwd);
                const float u = ((io.MousePos.x - mn.x) / sz.x) * 2.f - 1.f;
                const float v = 1.f - ((io.MousePos.y - mn.y) / sz.y) * 2.f;
                const float ta = std::tan(45.f * 3.14159265f / 360.f);
                const float aspect = (float)st->view_w / (float)st->view_h;
                V3 dir = norm3({fwd.x + u * ta * aspect * right.x + v * ta * up.x,
                                fwd.y + u * ta * aspect * right.y + v * ta * up.y,
                                fwd.z + u * ta * aspect * right.z + v * ta * up.z});
                st->imp_o[0] = eye.x; st->imp_o[1] = eye.y; st->imp_o[2] = eye.z;
                st->imp_d[0] = dir.x; st->imp_d[1] = dir.y; st->imp_d[2] = dir.z;
            }
        }
    } else {
        // Fallback ladder: subsampled CPU scatter, filling the same region.
        if (!sx.empty() &&
            ImPlot3D::BeginPlot("##flow_fallback", ImVec2(-1, -1))) {
            ImPlot3D::SetupAxesLimits(-kBoxL, kBoxL, -kBoxL, kBoxL, -kBoxL,
                                      kBoxL, ImPlot3DCond_Once);
            ImPlot3D::PlotScatter("particles", sx.data(), sy.data(), sz.data(),
                                  (int)sx.size());
            ImPlot3D::EndPlot();
        } else if (sx.empty()) {
            ImGui::TextDisabled("waiting for the first sim step…");
        }
    }
    ImGui::End();
    st->frames++;
}

void FlowScopeApplet::cleanup() {
    auto* st = s_.get();
    st->stop.store(true);
    if (st->job_id != 0) {
        st->jobs.request_cancel(st->job_id);
        for (int i = 0; i < 2000 && st->jobs.is_running(st->job_id); ++i)
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    if (st->view != 0) { st->geometry.release_view(st->view); st->view = 0; }
    // Pool teardown mirrors gpt_scope: drop pool-backed tensors first, then
    // the pool (its dtor releases cached bridge imports) — and if the worker
    // somehow outlived the grace, leak deliberately rather than UAF.
    {
        std::lock_guard<std::mutex> lk(st->mtx);
        for (int i = 0; i < kSlots; ++i) {
            st->pos[i] = torch::Tensor();
            st->speed[i] = torch::Tensor();
        }
    }
    if (st->job_id != 0 && st->jobs.is_running(st->job_id)) {
        (void)st->pool.release();
        if (st->host)
            st->host->log_info("flow-scope: worker still live at cleanup — "
                               "pool deliberately leaked");
    } else {
        st->pool.reset();
    }
    if (st->host) st->host->log_info("flow-scope: on_cleanup");
}

}  // namespace flowscope
