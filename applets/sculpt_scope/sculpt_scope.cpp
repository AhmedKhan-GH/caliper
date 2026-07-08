// ============================================================================
// SculptScope — a generator MLP g_θ: R^k -> R^3 trained live to match a target
// shape, drawn with ZERO copies of the point data (id dev.caliper.sculpt-scope).
//
// The fusion this applet exists to demonstrate: the (N,3) tensor the network's
// FINAL layer writes is the SAME device buffer the renderer draws in place.
// Every publish, under NoGrad, the display path runs
//     h = g_θ.hidden(z_all);                       // (N,128), default allocator
//     torch::addmm_out(pts[write], fc_out.bias, h, fc_out.weight.t());  // (N,3)
// so the last Linear's matmul lands directly in a pool-born slot — no .copy_()
// between the net's output and geometry.v1's imported-points read.
//
// Built on the field_scope/flow_scope backbone, reused verbatim: triple-buffered
// ExportablePool slots, the ready/display invariant, worker/frame threading + one
// publish mutex, orbit/zoom camera, DPI-correct view sizing, the magma-by-motion
// colormap, and the honest fallback ladder (no caps / no pool / CPU / GL -> a
// subsampled ImPlot3D scatter). The physics is replaced by libtorch training.
// ============================================================================
#include "sculpt_scope.h"
#include "sculpt_model.h"

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

namespace sculptscope {
namespace {

constexpr int   kSlots     = 3;      // triple buffer: write / ready / displayed
constexpr int   kSubsample = 10000;  // CPU fallback scatter size
constexpr float kBox       = 1.5f;   // fallback plot half-extent

// ---- tiny column-major mat4 helpers (verbatim from field_scope) ------------
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
struct SculptScopeState {
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
    std::atomic<bool>  train_on{true};
    std::atomic<bool>  reset_req{false};      // frame requests a weight re-init
    std::atomic<float> lr{2e-3f};
    std::atomic<int>   shape{(int)Shape::kTorus};

    std::mutex mtx;  // guards everything below, down to the frame block

    std::unique_ptr<caliper::adapters::ExportablePool> pool;
    torch::Tensor pos[kSlots], speed[kSlots];   // pool-born on the zero-copy path
    int  ready_slot   = -1;
    int  display_slot = -1;
    int64_t n_points  = 0;
    bool  on_gpu      = false;
    float steps_per_sec = 0.f;
    float last_loss   = 0.f;
    int64_t steps_trained = 0;

    // CPU fallback subsample (worker refreshes ~4 Hz when needed).
    std::vector<float> sub_x, sub_y, sub_z;
    uint64_t sub_gen = 0;

    // ------- frame-thread-only -------
    CaliperTextureId view = 0;
    int   view_w = 768, view_h = 768;
    float cam_az = 0.9f, cam_el = 0.45f, cam_dist = 4.2f;
    float color_vmax = 0.06f;   // motion (per-publish) scale for the LUT
    bool  zero_copy_frame = false;
    uint64_t frames = 0;
};

namespace {

// ---- the worker: build the net, then train + publish forever --------------
void sculpt_job(SculptScopeState* st, const CaliperJobControl* ctl) {
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
    const int64_t N = gpu ? 150'000 : 20'000;   // rendered points
    const int64_t B = gpu ? 1024 : 256;         // training minibatch

    // Zero-copy opt-in, decided once (identical gate to field_scope).
    std::unique_ptr<caliper::adapters::ExportablePool> pool;
    if (gpu && (st->geom_caps & CALIPER_GEOM_CAP_IMPORTED_POINTS)) {
        try {
            auto p = std::make_unique<caliper::adapters::ExportablePool>(0);
            if (p->ok()) pool = std::move(p);
        } catch (...) { /* pool absent -> fallback path, never a crash */ }
    }
    if (st->host) {
        if (pool)
            st->host->log_info(cuda ? "sculpt-scope: zero-copy pool ready (cuda)"
                                    : "sculpt-scope: zero-copy pool ready (mps)");
        else
            st->host->log_info(
                !gpu ? "sculpt-scope: fallback (torch CPU)"
                     : !(st->geom_caps & CALIPER_GEOM_CAP_IMPORTED_POINTS)
                           ? "sculpt-scope: fallback (no geometry service)"
                           : "sculpt-scope: fallback (pool unavailable)");
    }

    // The generator + optimizer (worker-thread-only; the frame never touches
    // them). Latents are persistent and default-allocated (never rendered).
    SculptNet net;
    net->to(dev);
    torch::Tensor z_all = torch::randn(
        {N, kLatentDim}, torch::TensorOptions(dev).dtype(torch::kFloat32));
    auto make_opt = [&] {
        return std::make_unique<torch::optim::Adam>(
            net->parameters(), torch::optim::AdamOptions(st->lr.load()));
    };
    std::unique_ptr<torch::optim::Adam> opt = make_opt();

    // Render slots: pool-born (all the renderer reads). prev holds the last
    // published positions so speed = per-publish motion (the color channel).
    torch::Tensor pos[kSlots], speed[kSlots];
    {
        auto opt_t = torch::TensorOptions(dev).dtype(torch::kFloat32);
        auto alloc_slots = [&] {
            for (int i = 0; i < kSlots; ++i) {
                pos[i]   = torch::zeros({N, 3}, opt_t);
                speed[i] = torch::zeros({N}, opt_t);
            }
        };
        if (pool) { auto scope = pool->use(); alloc_slots(); }
        else      { alloc_slots(); }
    }
    torch::Tensor prev = torch::zeros({N, 3},
                                      torch::TensorOptions(dev).dtype(torch::kFloat32));

    {
        std::lock_guard<std::mutex> lk(st->mtx);
        st->pool = std::move(pool);
        for (int i = 0; i < kSlots; ++i) { st->pos[i] = pos[i]; st->speed[i] = speed[i]; }
        st->n_points = N;
        st->on_gpu   = gpu;
    }

    // Publish helper: forward the whole latent set into `write`, under NoGrad,
    // with the final layer fused straight into the pool slot (the money op).
    auto publish = [&](int write, int read_for_sub, int64_t step) {
        {
            torch::NoGradGuard ng;
            auto h = net->hidden(z_all);                  // (N,128), default alloc
            torch::addmm_out(pos[write], net->fc_out->bias, h,
                             net->fc_out->weight.t());     // (N,3) INTO the slot
            speed[write].copy_((pos[write] - prev).norm(2, {1}));
            prev.copy_(pos[write]);
        }
        if (cuda) torch::cuda::synchronize();             // writes done BEFORE publish
#if defined(__APPLE__)
        else if (mps) caliper::adapters::detail::mps_synchronize_serialized();
#endif
        int next_write;
        {
            std::lock_guard<std::mutex> lk(st->mtx);
            st->ready_slot = write;
            next_write = 0;
            for (int i = 0; i < kSlots; ++i)
                if (i != st->ready_slot && i != st->display_slot) { next_write = i; break; }
        }
        // Fallback subsample, ~4 Hz.
        static int64_t last_sub = -1000;
        if (step - last_sub >= 8) {
            last_sub = step;
            auto sub = pos[write].narrow(0, 0, std::min<int64_t>(kSubsample, N))
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
        (void)read_for_sub;
        return next_write;
    };

    int write = 0;
    int64_t step = 0;
    auto rate_t0 = std::chrono::steady_clock::now();
    int  rate_steps = 0;

    // First publish so the cloud (an untrained blob) appears immediately.
    write = publish(write, write, step);

    while (!st->stop.load() && !(ctl && ctl->cancelled(ctl))) {
        if (st->paused.load()) {
            std::this_thread::sleep_for(std::chrono::milliseconds(16));
            continue;
        }
        // Weight reset requested from the UI: fresh net + optimizer, blob returns.
        if (st->reset_req.exchange(false)) {
            net = SculptNet();
            net->to(dev);
            opt = make_opt();
            st->steps_trained = 0;
        }
        // Keep the optimizer LR in sync with the slider.
        const float lr = st->lr.load();
        for (auto& g : opt->param_groups())
            static_cast<torch::optim::AdamOptions&>(g.options()).lr(lr);

        // ---- one training step (frozen if train is off) ----
        if (st->train_on.load()) {
            const auto shape = (Shape)st->shape.load();
            auto idx = torch::randint(0, N, {B},
                           torch::TensorOptions(dev).dtype(torch::kLong));
            auto gen  = net->forward(z_all.index_select(0, idx));   // WITH grad
            auto tgt  = sample_target(shape, B, dev);
            auto loss = energy_distance(gen, tgt);
            opt->zero_grad();
            loss.backward();
            opt->step();
            const float lv = loss.item<float>();                    // syncs
            int64_t s_now;
            {
                std::lock_guard<std::mutex> lk(st->mtx);
                st->last_loss = lv;
                s_now = ++st->steps_trained;
            }
            // Periodic, greppable learning-curve line (headless verification):
            // the loss is real and falling. ~ every 200 steps.
            if (st->host && (s_now == 1 || s_now % 25 == 0)) {
                char msg[96];
                std::snprintf(msg, sizeof(msg),
                              "sculpt-scope: step %lld  loss %.4f",
                              (long long)s_now, lv);
                st->host->log_info(msg);
            }
        }
        ++step;
        ++rate_steps;

        write = publish(write, write, step);

        const auto now = std::chrono::steady_clock::now();
        const float el = std::chrono::duration<float>(now - rate_t0).count();
        if (el >= 0.5f) {
            std::lock_guard<std::mutex> lk(st->mtx);
            st->steps_per_sec = (float)rate_steps / el;
            rate_t0 = now;
            rate_steps = 0;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
}

void sculpt_job_tramp(void* user, const CaliperJobControl* ctl) {
    sculpt_job(static_cast<SculptScopeState*>(user), ctl);
}

}  // namespace

SculptScopeApplet::SculptScopeApplet() : s_(std::make_unique<SculptScopeState>()) {}
SculptScopeApplet::~SculptScopeApplet() = default;

bool SculptScopeApplet::initialize(caliper::Host& host) {
    s_->host     = &host;
    s_->jobs     = caliper::Jobs(host);
    s_->device   = caliper::Device::query(host);
    s_->bridge   = caliper::Bridge(host);
    s_->geometry = caliper::Geometry(host);
    s_->geom_caps = s_->geometry.caps();
    host.log_info("sculpt-scope: on_init");
    s_->job_id = s_->jobs.submit("sculpt_scope: train", &sculpt_job_tramp, s_.get());
    return true;
}

void SculptScopeApplet::draw_ui() {
    auto* st = s_.get();

    // ---- snapshot worker-published state under the mutex ----
    torch::Tensor draw_pos, draw_speed;
    int64_t n = 0, steps = 0;
    bool gpu = false;
    float sps = 0.f, loss = 0.f;
    caliper::adapters::ExportablePool* pool = nullptr;
    std::vector<float> sx, sy, sz;
    {
        std::lock_guard<std::mutex> lk(st->mtx);
        if (st->ready_slot >= 0) {
            st->display_slot = st->ready_slot;
            draw_pos   = st->pos[st->display_slot];
            draw_speed = st->speed[st->display_slot];
        }
        n = st->n_points; gpu = st->on_gpu; sps = st->steps_per_sec;
        loss = st->last_loss; steps = st->steps_trained;
        pool = st->pool.get();
        sx = st->sub_x; sy = st->sub_y; sz = st->sub_z;
    }

    ImGui::Begin("SculptScope: Learned Cloud");

    // ---- toolbar: controls + honest status (last frame's provenance) ----
    const float bar_h = ImGui::GetFrameHeight() + ImGui::GetStyle().WindowPadding.y * 2.f;
    if (ImGui::BeginChild("##toolbar", ImVec2(0, bar_h), ImGuiChildFlags_Borders)) {
        bool paused = st->paused.load();
        if (ImGui::Checkbox("pause", &paused)) st->paused.store(paused);
        ImGui::SameLine();
        bool train = st->train_on.load();
        if (ImGui::Checkbox("train", &train)) st->train_on.store(train);
        ImGui::SameLine();
        if (ImGui::Button("reset")) st->reset_req.store(true);
        ImGui::SameLine();
        const char* shapes[] = {"sphere", "torus", "helix", "two lobes"};
        int shape = st->shape.load();
        ImGui::SetNextItemWidth(110);
        if (ImGui::Combo("shape", &shape, shapes, IM_ARRAYSIZE(shapes)))
            st->shape.store(shape);
        ImGui::SameLine();
        float lr = st->lr.load();
        ImGui::SetNextItemWidth(120);
        if (ImGui::SliderFloat("lr", &lr, 1e-4f, 1e-2f, "%.4f",
                               ImGuiSliderFlags_Logarithmic))
            st->lr.store(lr);
        ImGui::SameLine();
        ImGui::SetNextItemWidth(100);
        ImGui::SliderFloat("color", &st->color_vmax, 0.005f, 0.2f, "%.3f");
        ImGui::SameLine();
        ImGui::TextDisabled("|");
        ImGui::SameLine();
        if (st->zero_copy_frame)
            ImGui::TextColored({0.55f, 0.9f, 0.6f, 1.f},
                "%lld pts — zero-copy (imported geometry) · loss %.4f · %lld steps · %.0f/s",
                (long long)n, loss, (long long)steps, sps);
        else
            ImGui::TextColored({1.f, 0.7f, 0.4f, 1.f},
                "%lld pts — CPU fallback (subsampled %d) · loss %.4f · %lld steps · %s",
                (long long)n, (int)sx.size(), loss, (long long)steps,
                !gpu ? "torch CPU"
                     : (st->geom_caps ? "pool unavailable" : "no geometry service"));
        ImGui::SameLine();
        ImGui::TextDisabled("   (right-drag: orbit · wheel: zoom)");
    }
    ImGui::EndChild();

    // ---- the 3-D view fills all remaining space (field_scope DPI discipline) ----
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
        const V3 eye{st->cam_dist * ce * ca, st->cam_dist * se,
                     st->cam_dist * ce * sa};
        CaliperGeomCamera cam{};
        look_at(eye, {0, 0, 0}, {0, 1, 0}, cam.view);
        perspective(45.f * 3.14159265f / 180.f,
                    (float)st->view_w / (float)st->view_h, 0.05f, 50.f, cam.proj);

        auto pref = pool->to_bridge(st->bridge, draw_pos);
        auto sref = pool->to_bridge(st->bridge, draw_speed);
        if (pref && sref) {
            // Motion (per-publish displacement) -> magma. A small negative vmin
            // lifts a settled cloud off the LUT floor so it stays visible.
            const float vmin = -0.2f * st->color_vmax;
            st->zero_copy_frame = st->geometry.draw_points(
                st->view, &cam, pref->alloc, pref->offset, (uint64_t)n,
                sref->alloc, sref->offset, CALIPER_CMAP_MAGMA, vmin,
                st->color_vmax, 1.5f * fb_scale, 0xFF000000u);
            // One-shot proof the fusion path actually DREW (not just imported):
            // the net's own forward-output buffer went to the renderer in place.
            static bool logged_first = false;
            if (st->zero_copy_frame && !logged_first && st->host) {
                logged_first = true;
                char msg[96];
                std::snprintf(msg, sizeof(msg),
                              "sculpt-scope: first zero-copy draw — %lld pts drawn in place",
                              (long long)n);
                st->host->log_info(msg);
            }
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
            if (st->cam_el > 1.5f) st->cam_el = 1.5f;
            if (st->cam_el < -1.5f) st->cam_el = -1.5f;
        }
        if (hovered && io.MouseWheel != 0.f) {
            st->cam_dist *= (1.f - io.MouseWheel * 0.08f);
            if (st->cam_dist < 1.2f) st->cam_dist = 1.2f;
            if (st->cam_dist > 12.f) st->cam_dist = 12.f;
        }
    } else {
        if (!sx.empty() &&
            ImPlot3D::BeginPlot("##sculpt_fallback", ImVec2(-1, -1))) {
            ImPlot3D::SetupAxesLimits(-kBox, kBox, -kBox, kBox, -kBox, kBox,
                                      ImPlot3DCond_Once);
            ImPlot3D::PlotScatter("points", sx.data(), sy.data(), sz.data(),
                                  (int)sx.size());
            ImPlot3D::EndPlot();
        } else if (sx.empty()) {
            ImGui::TextDisabled("waiting for the first forward pass…");
        }
    }
    ImGui::End();
    st->frames++;
}

void SculptScopeApplet::cleanup() {
    auto* st = s_.get();
    st->stop.store(true);
    if (st->job_id != 0) {
        st->jobs.request_cancel(st->job_id);
        for (int i = 0; i < 2000 && st->jobs.is_running(st->job_id); ++i)
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    if (st->view != 0) { st->geometry.release_view(st->view); st->view = 0; }
    // Pool teardown mirrors field_scope/gpt_scope: drop pool-backed tensors
    // first, then the pool; leak deliberately if the worker somehow outlived
    // the grace rather than risk a use-after-free.
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
            st->host->log_info("sculpt-scope: worker still live at cleanup — "
                               "pool deliberately leaked");
    } else {
        st->pool.reset();
    }
    if (st->host) st->host->log_info("sculpt-scope: on_cleanup");
}

}  // namespace sculptscope
