// ============================================================================
// FieldScope — a self-consistent electrostatic Particle-In-Cell (PIC) plasma,
// drawn with ZERO copies of the point data (id dev.caliper.field-scope).
//
// The particles generate the field that pushes them. Each worker step:
//   rho   = deposit_cic(pos)          particles -> grid charge (Cloud-In-Cell)
//   E     = solve_field(rho)          FFT Poisson on the density contrast
//   Epart = coupling * gather_cic(E)  self-field back to the particles
//   boris_push(pos, vel, Epart, B)    half E-kick / B-rotation / half E-kick
// so real collective phenomena EMERGE — the two-stream instability rolls the
// cloud into phase-space vortices, warm plasmas ring at the plasma frequency —
// rather than being scripted by a canned analytic field (the prior version).
// See em_pic.h for the physics (host-free, unit-tested in tests/test_em_pic.cpp).
//
// Built on the flow_scope/sculpt_scope zero-copy spine, reused verbatim: the
// triple-buffered ExportablePool slots, the ready/display invariant, worker/
// frame threading + one publish mutex, orbit/zoom camera, DPI-correct view
// sizing, magma-by-speed colour, and the honest fallback ladder (no caps / no
// pool / CPU / GL -> a subsampled ImPlot3D scatter). Only the physics changed.
// ============================================================================
#include "field_scope.h"
#include "em_pic.h"

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
#include <cstdio>
#include <cstring>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace fieldscope {
namespace {

constexpr int   kSlots     = 3;                 // write / ready / displayed
constexpr int   kG         = 48;                // PIC grid is kG^3
constexpr float kL         = 10.0f;             // solver domain [0,L)^3; cloud
                                                // floats near the centre (open)
constexpr float kDt        = 0.02f;
constexpr float kV0        = 1.2f;              // stream drift speed
constexpr float kCenter    = kL * 0.5f;         // cloud centre (camera target)
constexpr int   kSubsample = 10000;             // CPU fallback scatter size

// ---- tiny column-major mat4 helpers (verbatim from flow_scope) -------------
struct V3 { float x, y, z; };
V3 operator+(V3 a, V3 b) { return {a.x + b.x, a.y + b.y, a.z + b.z}; }
V3 operator-(V3 a, V3 b) { return {a.x - b.x, a.y - b.y, a.z - b.z}; }
V3 cross(V3 a, V3 b) {
    return {a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x};
}
float dot(V3 a, V3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
V3 norm3(V3 a) {
    const float l = std::sqrt(dot(a, a));
    return l > 0 ? V3{a.x / l, a.y / l, a.z / l} : V3{0, 0, 1};
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
struct FieldScopeState {
    caliper::Host*    host = nullptr;
    caliper::Jobs     jobs;
    caliper::Device   device;
    caliper::Bridge   bridge;
    caliper::Geometry geometry;
    uint32_t geom_caps = 0;

    uint64_t job_id = 0;
    std::atomic<bool> stop{false};
    std::atomic<bool> paused{false};

    // frame -> worker knobs (atomics; torn reads harmless).
    std::atomic<int>   field_mode{(int)Field::kToroidal}; // which field to observe
    std::atomic<float> b_field{8.0f};        // field strength (B0 / magnetization)
    std::atomic<float> vth{1.6f};            // test-particle speed (field modes)
    std::atomic<float> coupling{6.0f};       // self-field scale (plasma mode)
    std::atomic<float> trap{0.6f};           // axial (z) confinement (plasma mode)
    std::atomic<float> temperature{0.03f};   // thermal spread (plasma mode)
    std::atomic<bool>  reseed_req{false};
    std::atomic<float> impulse_strength{6.0f};

    std::mutex mtx;  // guards everything below, down to the frame block
    bool  imp_active = false;
    float imp_o[3] = {0, 0, 0}, imp_d[3] = {0, 0, 1};

    std::unique_ptr<caliper::adapters::ExportablePool> pool;
    torch::Tensor pos[kSlots], speed[kSlots];   // pool-born on the zero-copy path
    int  ready_slot   = -1;
    int  display_slot = -1;
    int64_t n_particles = 0;
    bool  on_gpu      = false;
    float steps_per_sec = 0.f;
    float last_ke     = 0.f;

    std::vector<float> sub_x, sub_y, sub_z;
    uint64_t sub_gen = 0;

    // ------- frame-thread-only -------
    CaliperTextureId view = 0;
    int   view_w = 768, view_h = 768;
    float cam_az = 0.9f, cam_el = 0.6f, cam_dist = 9.0f;
    float color_vmax = 1.2f;
    bool  zero_copy_frame = false;
    uint64_t frames = 0;
};

namespace {

// ---- the worker: init a plasma, then PIC-step + publish forever ------------
void sim_job(FieldScopeState* st, const CaliperJobControl* ctl) {
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
    const int64_t N = gpu ? 200'000 : 20'000;

    std::unique_ptr<caliper::adapters::ExportablePool> pool;
    if (gpu && (st->geom_caps & CALIPER_GEOM_CAP_IMPORTED_POINTS)) {
        try {
            auto p = std::make_unique<caliper::adapters::ExportablePool>(0);
            if (p->ok()) pool = std::move(p);
        } catch (...) { /* pool absent -> fallback path, never a crash */ }
    }
    if (st->host) {
        if (pool)
            st->host->log_info(cuda ? "field-scope: zero-copy pool ready (cuda)"
                                    : "field-scope: zero-copy pool ready (mps)");
        else
            st->host->log_info(
                !gpu ? "field-scope: fallback (torch CPU)"
                     : !(st->geom_caps & CALIPER_GEOM_CAP_IMPORTED_POINTS)
                           ? "field-scope: fallback (no geometry service)"
                           : "field-scope: fallback (pool unavailable)");
    }

    // Render slots: pool-born (all the renderer reads). Velocity is default-
    // allocated (never rendered). The plasma's initial state seeds slot 0.
    torch::Tensor pos[kSlots], speed[kSlots];
    {
        auto opt = torch::TensorOptions(dev).dtype(torch::kFloat32);
        auto alloc_slots = [&] {
            for (int i = 0; i < kSlots; ++i) {
                pos[i]   = torch::zeros({N, 3}, opt);
                speed[i] = torch::zeros({N}, opt);
            }
        };
        if (pool) { auto scope = pool->use(); alloc_slots(); }
        else      { alloc_slots(); }
    }
    const auto c3 = torch::tensor({kCenter, kCenter, kCenter},
                                  torch::TensorOptions(dev).dtype(torch::kFloat32));
    // Seed pos/vel/charge for the selected mode: plasma = the self-consistent
    // magnetized clumps; the rest = test particles in a prescribed field.
    auto seed = [&] {
        const auto mode = (Field)st->field_mode.load();
        return mode == Field::kPlasma
            ? init_state(IC::kClumps, N, kL, st->temperature.load(), kV0, dev)
            : init_field(mode, N, kL, st->vth.load(), dev);
    };
    auto [p0, vel, charge] = seed();
    pos[0].copy_(p0);

    {
        std::lock_guard<std::mutex> lk(st->mtx);
        st->pool = std::move(pool);
        for (int i = 0; i < kSlots; ++i) { st->pos[i] = pos[i]; st->speed[i] = speed[i]; }
        st->n_particles = N;
        st->on_gpu   = gpu;
        st->ready_slot = 0;
    }

    auto sync = [&] {
        if (cuda) torch::cuda::synchronize();
#if defined(__APPLE__)
        else if (mps) caliper::adapters::detail::mps_synchronize_serialized();
#endif
    };

    int write = 1;
    int64_t step = 0;
    auto rate_t0 = std::chrono::steady_clock::now();
    int  rate_steps = 0;
    int64_t last_sub = -1000;
    bool logged_first = false;

    while (!st->stop.load() && !(ctl && ctl->cancelled(ctl))) {
        if (st->paused.load()) {
            std::this_thread::sleep_for(std::chrono::milliseconds(16));
            continue;
        }
        int read;
        { std::lock_guard<std::mutex> lk(st->mtx); read = st->ready_slot; }

        // Re-seed (mode change or reset button): rebuild into the write slot
        // and publish it, skipping a physics step.
        if (st->reseed_req.exchange(false)) {
            auto [pp, vv, cc] = seed();
            vel = vv; charge = cc;
            pos[write].copy_(pp);
            speed[write].copy_(vel.norm(2, {1}));
            sync();
            std::lock_guard<std::mutex> lk(st->mtx);
            st->ready_slot = write;
            for (int i = 0; i < kSlots; ++i)
                if (i != st->ready_slot && i != st->display_slot) { write = i; break; }
            continue;
        }

        // ---- one step ----
        const auto opt_dev = torch::TensorOptions(dev).dtype(torch::kFloat32);
        const auto mode = (Field)st->field_mode.load();
        torch::Tensor accel, B;
        if (mode == Field::kPlasma) {
            // Self-consistent magnetized plasma: signed charge -> grid -> FFT
            // free-space Poisson -> gather, plus an anisotropic trap (strong z,
            // gentle xy; B does the transverse confinement).
            auto rho = deposit_cic(pos[read], kG, kL, charge.squeeze(1));
            auto E   = poisson_E_free(rho, kL) * (st->coupling.load() / (float)N);
            accel = charge * gather_cic(E, pos[read], kL);
            const float kz = st->trap.load();
            auto kvec = torch::tensor({0.12f * kz, 0.12f * kz, kz}, opt_dev);
            accel = accel - (pos[read] - c3) * kvec;
            B = torch::tensor({0.f, 0.f, st->b_field.load()}, opt_dev);
        } else {
            // Prescribed field geometry: pure test-particle motion (no self-
            // field, no trap) in B(x) — gyration / mirroring / drifts.
            B = external_B(pos[read], mode, kCenter, st->b_field.load(), kL);
            accel = torch::zeros_like(pos[read]);
        }

        // Cursor-ray impulse: a radial push away from the mouse ray (an
        // external, charge-independent perturbation).
        bool active; float o3[3], d3[3];
        {
            std::lock_guard<std::mutex> lk(st->mtx);
            active = st->imp_active;
            std::memcpy(o3, st->imp_o, sizeof(o3));
            std::memcpy(d3, st->imp_d, sizeof(d3));
        }
        if (active) {
            const float strength = st->impulse_strength.load();
            auto o = torch::from_blob(o3, {1, 3}, torch::kFloat32).to(dev);
            auto d = torch::from_blob(d3, {1, 3}, torch::kFloat32).to(dev);
            auto a = pos[read] - o;
            auto tproj = (a * d).sum(1, true).clamp_min(0.f);
            auto r = pos[read] - (o + tproj * d);
            auto dist2 = (r * r).sum(1, true);
            const float rad = 0.8f;
            accel = accel + r / (dist2.sqrt() + 1e-4f) *
                            (torch::exp(dist2 / (-2.f * rad * rad)) * strength);
        }

        boris_push(pos[read], pos[write], vel, accel, charge, B, kDt);
        speed[write].copy_(vel.norm(2, {1}));
        sync();
        ++step; ++rate_steps;

        {
            std::lock_guard<std::mutex> lk(st->mtx);
            st->ready_slot = write;
            for (int i = 0; i < kSlots; ++i)
                if (i != st->ready_slot && i != st->display_slot) { write = i; break; }
        }

        // Fallback subsample, ~4 Hz.
        if (step - last_sub >= 8) {
            last_sub = step;
            auto sub = pos[read].narrow(0, 0, std::min<int64_t>(kSubsample, N))
                           .to(torch::kCPU).contiguous();
            const float* sp = sub.data_ptr<float>();
            const int64_t ns = sub.size(0);
            std::lock_guard<std::mutex> lk(st->mtx);
            st->sub_x.resize((size_t)ns); st->sub_y.resize((size_t)ns);
            st->sub_z.resize((size_t)ns);
            for (int64_t i = 0; i < ns; ++i) {
                st->sub_x[(size_t)i] = sp[i * 3 + 0];
                st->sub_y[(size_t)i] = sp[i * 3 + 1];
                st->sub_z[(size_t)i] = sp[i * 3 + 2];
            }
            st->sub_gen++;
        }

        const auto now = std::chrono::steady_clock::now();
        const float el = std::chrono::duration<float>(now - rate_t0).count();
        if (el >= 0.5f) {
            const float ke = 0.5f * (vel * vel).sum().item<float>();   // syncs
            std::lock_guard<std::mutex> lk(st->mtx);
            st->steps_per_sec = (float)rate_steps / el;
            st->last_ke = ke;
            rate_t0 = now; rate_steps = 0;
            if (st->host && !logged_first) {
                logged_first = true;
                const char* mn[] = {"plasma", "uniform", "mirror", "toroidal", "dipole"};
                char m[128];
                std::snprintf(m, sizeof(m),
                              "field-scope: running — %lld particles, field=%s, B=%.1f",
                              (long long)N, mn[st->field_mode.load() & 7], st->b_field.load());
                st->host->log_info(m);
            }
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
}

void field_job_tramp(void* user, const CaliperJobControl* ctl) {
    sim_job(static_cast<FieldScopeState*>(user), ctl);
}

}  // namespace

FieldScopeApplet::FieldScopeApplet() : s_(std::make_unique<FieldScopeState>()) {}
FieldScopeApplet::~FieldScopeApplet() = default;

bool FieldScopeApplet::initialize(caliper::Host& host) {
    s_->host     = &host;
    s_->jobs     = caliper::Jobs(host);
    s_->device   = caliper::Device::query(host);
    s_->bridge   = caliper::Bridge(host);
    s_->geometry = caliper::Geometry(host);
    s_->geom_caps = s_->geometry.caps();
    if (const char* fm = std::getenv("FS_FIELD")) s_->field_mode.store(std::atoi(fm));
    host.log_info("field-scope: on_init");
    s_->job_id = s_->jobs.submit("field_scope: PIC", &field_job_tramp, s_.get());
    return true;
}

void FieldScopeApplet::draw_ui() {
    auto* st = s_.get();

    torch::Tensor draw_pos, draw_speed;
    int64_t n = 0;
    bool gpu = false;
    float sps = 0.f, ke = 0.f;
    caliper::adapters::ExportablePool* pool = nullptr;
    std::vector<float> sx, sy, sz;
    {
        std::lock_guard<std::mutex> lk(st->mtx);
        if (st->ready_slot >= 0) {
            st->display_slot = st->ready_slot;
            draw_pos   = st->pos[st->display_slot];
            draw_speed = st->speed[st->display_slot];
        }
        n = st->n_particles; gpu = st->on_gpu; sps = st->steps_per_sec;
        ke = st->last_ke; pool = st->pool.get();
        sx = st->sub_x; sy = st->sub_y; sz = st->sub_z;
    }

    ImGui::Begin("FieldScope: EM Field");

    const float bar_h = ImGui::GetFrameHeight() + ImGui::GetStyle().WindowPadding.y * 2.f;
    if (ImGui::BeginChild("##toolbar", ImVec2(0, bar_h), ImGuiChildFlags_Borders)) {
        bool paused = st->paused.load();
        if (ImGui::Checkbox("pause", &paused)) st->paused.store(paused);
        ImGui::SameLine();
        const char* modes[] = {"plasma (self-field)", "uniform B", "mirror",
                               "toroidal", "dipole"};
        int fm = st->field_mode.load();
        ImGui::SetNextItemWidth(150);
        if (ImGui::Combo("field", &fm, modes, IM_ARRAYSIZE(modes))) {
            st->field_mode.store(fm);
            st->reseed_req.store(true);           // re-seed on selection
        }
        const bool plasma = (fm == (int)Field::kPlasma);
        ImGui::SameLine();
        if (ImGui::Button("reseed")) st->reseed_req.store(true);
        ImGui::SameLine();
        float b = st->b_field.load();
        ImGui::SetNextItemWidth(90);
        if (ImGui::SliderFloat("B (strength)", &b, 0.f, 40.f)) st->b_field.store(b);
        ImGui::SameLine();
        if (plasma) {
            float cpl = st->coupling.load();
            ImGui::SetNextItemWidth(70);
            if (ImGui::SliderFloat("charge", &cpl, 1.f, 30.f)) st->coupling.store(cpl);
            ImGui::SameLine();
            float tr = st->trap.load();
            ImGui::SetNextItemWidth(60);
            if (ImGui::SliderFloat("trap", &tr, 0.05f, 1.5f)) st->trap.store(tr);
        } else {
            float vs = st->vth.load();
            ImGui::SetNextItemWidth(90);
            if (ImGui::SliderFloat("speed", &vs, 0.2f, 4.f)) st->vth.store(vs);
        }
        ImGui::SameLine();
        ImGui::SetNextItemWidth(80);
        ImGui::SliderFloat("color", &st->color_vmax, 0.1f, 3.f);
        ImGui::SameLine();
        ImGui::TextDisabled("|");
        ImGui::SameLine();
        if (st->zero_copy_frame)
            ImGui::TextColored({0.55f, 0.9f, 0.6f, 1.f},
                "%lld particles — zero-copy (imported geometry) · %s · KE %.2f · %.0f steps/s",
                (long long)n, modes[fm], ke, sps);
        else
            ImGui::TextColored({1.f, 0.7f, 0.4f, 1.f},
                "%lld particles — CPU fallback (subsampled %d) · %s · KE %.2f · %s",
                (long long)n, (int)sx.size(), modes[fm], ke,
                !gpu ? "torch CPU"
                     : (st->geom_caps ? "pool unavailable" : "no geometry service"));
        ImGui::SameLine();
        ImGui::TextDisabled("   (left-drag: perturb · right-drag: orbit · wheel: zoom)");
    }
    ImGui::EndChild();

    const ImVec2 avail = ImGui::GetContentRegionAvail();
    const bool geom_live =
        st->geometry && (st->geom_caps & CALIPER_GEOM_CAP_IMPORTED_POINTS);

    const float fb_scale = ImGui::GetIO().DisplayFramebufferScale.y > 0.f
                               ? ImGui::GetIO().DisplayFramebufferScale.y : 1.f;
    auto clampi = [](int v, int lo, int hi) { return v < lo ? lo : (v > hi ? hi : v); };
    const int dw = clampi((int)(avail.x * fb_scale), 64, 4096);
    const int dh = clampi((int)(avail.y * fb_scale), 64, 4096);
    if (geom_live && avail.x >= 64 && avail.y >= 64 &&
        (st->view == 0 || std::abs(dw - st->view_w) >= 3 ||
         std::abs(dh - st->view_h) >= 3)) {
        if (st->view != 0) st->geometry.release_view(st->view);
        st->view = st->geometry.create_view((uint32_t)dw, (uint32_t)dh);
        st->view_w = dw; st->view_h = dh;
    }

    st->zero_copy_frame = false;
    if (geom_live && st->view != 0 && pool && draw_pos.defined()) {
        const float ce = std::cos(st->cam_el), se = std::sin(st->cam_el);
        const float ca = std::cos(st->cam_az), sa = std::sin(st->cam_az);
        const V3 c{kCenter, kCenter, kCenter};
        const V3 eye = c + V3{st->cam_dist * ce * ca, st->cam_dist * se,
                              st->cam_dist * ce * sa};
        CaliperGeomCamera cam{};
        look_at(eye, c, {0, 1, 0}, cam.view);
        perspective(45.f * 3.14159265f / 180.f,
                    (float)st->view_w / (float)st->view_h, 0.1f, 100.f, cam.proj);

        auto pref = pool->to_bridge(st->bridge, draw_pos);
        auto sref = pool->to_bridge(st->bridge, draw_speed);
        if (pref && sref) {
            const float vmin = -0.15f * st->color_vmax;
            st->zero_copy_frame = st->geometry.draw_points(
                st->view, &cam, pref->alloc, pref->offset, (uint64_t)n,
                sref->alloc, sref->offset, CALIPER_CMAP_MAGMA, vmin,
                st->color_vmax, 1.5f * fb_scale, 0xFF000000u);
            static bool logged_draw = false;
            if (st->zero_copy_frame && !logged_draw && st->host) {
                logged_draw = true;
                char m[96];
                std::snprintf(m, sizeof(m),
                              "field-scope: first zero-copy draw — %lld particles in place",
                              (long long)n);
                st->host->log_info(m);
            }
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
            if (st->cam_el > 1.5f) st->cam_el = 1.5f;
            if (st->cam_el < -1.5f) st->cam_el = -1.5f;
        }
        if (hovered && io.MouseWheel != 0.f) {
            st->cam_dist *= (1.f - io.MouseWheel * 0.08f);
            if (st->cam_dist < 2.5f) st->cam_dist = 2.5f;
            if (st->cam_dist > 20.f) st->cam_dist = 20.f;
        }
        const bool pushing = hovered && ImGui::IsMouseDown(ImGuiMouseButton_Left);
        {
            std::lock_guard<std::mutex> lk(st->mtx);
            st->imp_active = pushing;
            if (pushing && sz.x > 0.f && sz.y > 0.f) {
                const float ce = std::cos(st->cam_el), se = std::sin(st->cam_el);
                const float ca = std::cos(st->cam_az), sa = std::sin(st->cam_az);
                const V3 c{kCenter, kCenter, kCenter};
                const V3 eye = c + V3{st->cam_dist * ce * ca, st->cam_dist * se,
                                      st->cam_dist * ce * sa};
                const V3 fwd = norm3(c - eye);
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
        if (!sx.empty() &&
            ImPlot3D::BeginPlot("##field_fallback", ImVec2(-1, -1))) {
            ImPlot3D::SetupAxesLimits(kCenter - 3, kCenter + 3, kCenter - 3,
                                      kCenter + 3, kCenter - 3, kCenter + 3,
                                      ImPlot3DCond_Once);
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

void FieldScopeApplet::cleanup() {
    auto* st = s_.get();
    st->stop.store(true);
    if (st->job_id != 0) {
        st->jobs.request_cancel(st->job_id);
        for (int i = 0; i < 2000 && st->jobs.is_running(st->job_id); ++i)
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    if (st->view != 0) { st->geometry.release_view(st->view); st->view = 0; }
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
            st->host->log_info("field-scope: worker still live at cleanup — "
                               "pool deliberately leaked");
    } else {
        st->pool.reset();
    }
    if (st->host) st->host->log_info("field-scope: on_cleanup");
}

}  // namespace fieldscope
