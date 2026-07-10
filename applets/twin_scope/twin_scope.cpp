// TwinScope v2 — the surface twin (design doc §6-§9). A subdivided-mesh
// cotangent-Laplacian heat sim (twin_surface.h / twin_model.h) flows heat over
// the 3-D housing surface on a background worker; a small MLP chases the field.
// The hero splits sim|net side by side, drawn zero-copy from imported device
// tensors at texture resolution (geometry.v1_2), with a live textured↔per-vertex
// (v1.1) toggle exposing the R2 resolution gap. Frame-thread discipline is
// absolute: every torch op lives on the init/worker job; draw_ui only snapshots
// (a shared_ptr + tensor handles under the mutex) and renders.

// Windows headers before torch (NOMINMAX / lean) so <windows.h> min/max macros
// never clash with torch — the module_dir() helper needs GetModuleHandleExW.
#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#else
#include <dlfcn.h>
#endif

#include "twin_scope.h"
#include "twin_model.h"
#include "twin_surface.h"

#include <caliper/adapters/exportable_pool.hpp>
#include <caliper/adapters/obj.hpp>
#include <caliper/adapters/orbit_camera.hpp>
#include <caliper/adapters/torch.hpp>
#include <caliper/caliper.hpp>
#include <imgui.h>
#include <implot.h>
#include <torch/torch.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <cfloat>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <thread>
#include <vector>

namespace twinscope {

using caliper::adapters::V3;
using caliper::adapters::cross;
using caliper::adapters::cursor_ray;
using caliper::adapters::dot;
using caliper::adapters::look_at;
using caliper::adapters::normalized;
using caliper::adapters::orbit_eye;
using caliper::adapters::perspective;

namespace {

constexpr int kSlots = 3;
constexpr int kVariants = 50;      // B boundary-condition variants (variant 0 = hero)
constexpr int kSubdiv = 2;         // render mesh -> sim mesh (×16 vertices)
constexpr int kTexH = 256;
constexpr int kTexW = 256;
constexpr float kPi = caliper::adapters::kOrbitPi;

// Display LUT ranges (fixed per FIELD kind — see mode_range). Temperature is
// the same physical quantity for sim and net, so they share ONE range and read
// comparably in the split hero. Error uses its own [0, span]. Fixing the range
// per field makes the donor's mode-switch LUT flash structurally impossible:
// each field's texture is created once with its own range and never remapped.
constexpr float kTempMin = kAmbient;      // 22 °C
constexpr float kTempMax = 100.0f;        // legible ceiling (calibrated gain, below)
constexpr float kErrSpan = 25.0f;         // |sim − net| °C

// Display modes. published_mode (§8.d) carries the mode that SELECTED a publish;
// draw_ui keys both the drawn-field set and each field's LUT on it.
enum DisplayMode { MODE_SPLIT = 0, MODE_SIM = 1, MODE_NET = 2, MODE_ERR = 3 };
// Field kinds — index into cpu_fields and the per-field texture/vertex slots.
enum FieldKind { FIELD_SIM = 0, FIELD_NET = 1, FIELD_ERR = 2, FIELD_COUNT = 3 };

// The ONE range helper (§8.d): textured (baked into the bridge texture) and
// per-vertex (passed straight to draw_primitives) rungs both call it, so they
// cannot disagree about what a color means.
inline void mode_range(int field, float& vmin, float& vmax) {
    if (field == FIELD_ERR) { vmin = 0.0f; vmax = kErrSpan; }
    else { vmin = kTempMin; vmax = kTempMax; }
}

// Provenance rungs for the honest status line (§6/§9). ZEROCOPY only when the
// imported-allocation path actually drew this frame (flow_scope discipline).
enum Prov { PROV_WAIT, PROV_HEATMAP, PROV_PERVERTEX, PROV_CPU_TEX, PROV_ZEROCOPY };

// Directory of the loaded twin_scope shared library (§8.c). The staged asset
// (<applets>/assets/twin_scope/housing.obj) lives next to it.
std::string module_dir() {
#ifdef _WIN32
    HMODULE mod = nullptr;
    if (GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS |
                               GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                           reinterpret_cast<LPCWSTR>(&module_dir), &mod) &&
        mod) {
        wchar_t buf[MAX_PATH];
        const DWORD n = GetModuleFileNameW(mod, buf, MAX_PATH);
        if (n > 0 && n < MAX_PATH) {
            std::wstring w(buf, n);
            const size_t slash = w.find_last_of(L"\\/");
            if (slash != std::wstring::npos) w.resize(slash);
            const int len = WideCharToMultiByte(CP_UTF8, 0, w.c_str(),
                                                static_cast<int>(w.size()),
                                                nullptr, 0, nullptr, nullptr);
            std::string s(static_cast<size_t>(len), '\0');
            WideCharToMultiByte(CP_UTF8, 0, w.c_str(), static_cast<int>(w.size()),
                                s.data(), len, nullptr, nullptr);
            return s;
        }
    }
    return {};
#else
    Dl_info info;
    if (dladdr(reinterpret_cast<void*>(&module_dir), &info) && info.dli_fname) {
        std::string p = info.dli_fname;
        const size_t slash = p.find_last_of('/');
        if (slash != std::string::npos) p.resize(slash);
        return p;
    }
    return {};
#endif
}

// DLL-relative staged copy first, source-tree macro as the dev fallback; log the
// choice (§8.c).
std::string resolve_asset(caliper::Host& host) {
    const std::string dir = module_dir();
    if (!dir.empty()) {
        const std::string staged = dir + "/assets/twin_scope/housing.obj";
        std::ifstream probe(staged);
        if (probe.good()) {
            host.log_info(("twin-scope: asset (DLL-relative) " + staged).c_str());
            return staged;
        }
    }
    host.log_info("twin-scope: asset (source-tree fallback) " TWIN_SCOPE_ASSET_PATH);
    return TWIN_SCOPE_ASSET_PATH;
}

// Möller–Trumbore ray/triangle (transcribed from the donor). Returns the hit
// distance and the barycentric (b1,b2) of the second/third vertices.
bool intersect_triangle(V3 origin, V3 direction, V3 a, V3 b, V3 c,
                        float& distance, float& bary_b, float& bary_c) {
    const V3 e1 = b - a, e2 = c - a;
    const V3 p = cross(direction, e2);
    const float determinant = dot(e1, p);
    if (std::abs(determinant) < 1e-7f) return false;
    const float inverse = 1.f / determinant;
    const V3 t = origin - a;
    bary_b = dot(t, p) * inverse;
    if (bary_b < 0.f || bary_b > 1.f) return false;
    const V3 q = cross(t, e1);
    bary_c = dot(direction, q) * inverse;
    if (bary_c < 0.f || bary_b + bary_c > 1.f) return false;
    distance = dot(e2, q) * inverse;
    return distance > 0.f;
}

CaliperTensor cpu_field_desc(const float* base, int h, int w) {
    CaliperTensor desc{};
    desc.struct_size = sizeof(desc);
    desc.data = const_cast<float*>(base);
    desc.dtype = CALIPER_DT_F32;
    desc.ndim = 2;
    desc.shape[0] = h; desc.shape[1] = w;
    desc.strides[0] = w; desc.strides[1] = 1;
    desc.device = CALIPER_DEV_CPU;
    return desc;
}

}  // namespace

// ===========================================================================
// State
// ===========================================================================
struct TwinScopeState {
    caliper::Host* host = nullptr;
    caliper::Jobs jobs;
    caliper::Bridge bridge;
    caliper::Geometry geometry;
    uint32_t bridge_caps = 0;
    uint32_t geom_caps = 0;
    uint64_t job_id = 0;

    // Worker control (atomics — direct store from the widgets, §8.f).
    std::atomic<bool> stop{false};
    std::atomic<bool> sim_on{true};
    std::atomic<bool> train_on{true};
    std::atomic<bool> reset_req{false};
    std::atomic<float> learning_rate{2e-3f};
    std::atomic<int> display_mode{MODE_SPLIT};
    std::atomic<bool> textured_pref{true};   // R2 toggle
    // Per-source override: <0 → follow the seeded duty schedule; [0,1] → held
    // user duty factor (source has left the cycle, §7). One atomic per source,
    // stored directly where the change happens.
    std::array<std::atomic<float>, kSourceCount> src_override;

    // Shared worker → frame (guarded by mutex).
    std::mutex mutex;
    caliper::obj::Mesh mesh_cpu;
    std::array<V3, kSourceCount> sites_cpu{};
    std::unique_ptr<caliper::adapters::ExportablePool> pool;
    torch::Tensor positions, normals, uvs, indices, wire_indices;   // render mesh
    torch::Tensor tex_slot[FIELD_COUNT][kSlots];   // (H,W) sim/net/err per slot
    torch::Tensor vert_slot[FIELD_COUNT][kSlots];  // (V_render,) per slot
    int ready_slot = -1;
    int display_slot = -1;
    int published_mode = MODE_SPLIT;
    int field_h = 0, field_w = 0;
    int64_t v_render = 0, v_sim = 0;
    bool gpu = false;
    std::string device_name = "CPU";
    std::shared_ptr<std::vector<float>> cpu_fields;  // (FIELD_COUNT*H*W), one snapshot
    float loss = 0.f;
    float peak = kAmbient;
    float sim_sps = 0.f, train_sps = 0.f;
    int64_t sim_steps = 0, train_steps = 0;
    std::string worker_error;

    // Frame-thread-only renderer state.
    CaliperTextureId view = 0;
    CaliperTextureId field_tex[FIELD_COUNT] = {0, 0, 0};
    int view_w = 1000, view_h = 720;
    V3 cam_target{0.f, 0.5f, 0.f};
    float model_offset = 2.3f;   // ±x split offset (set from mesh bounds at init)
    float camera_azimuth = 0.9f;
    float camera_elevation = 0.55f;
    float camera_distance = 15.0f;
    bool wireframe = false;
    int selected_source = -1;
    int logged_prov = -1;
    bool logged_first_draw = false;
    std::vector<float> loss_history;

    TwinScopeState() {
        for (auto& v : src_override) v.store(-1.f);
    }
};

// ===========================================================================
// Worker
// ===========================================================================
namespace {

void twin_job(TwinScopeState* state, const CaliperJobControl* control) {
    const bool cuda = torch::cuda::is_available();
#if defined(__APPLE__)
    const bool mps = !cuda && torch::mps::is_available();
#else
    const bool mps = false;
#endif
    const bool gpu = cuda || mps;
    const torch::Device device = cuda ? torch::Device(torch::kCUDA)
                               : mps ? torch::Device(torch::kMPS)
                                     : torch::Device(torch::kCPU);

    // --- init job: OBJ (already loaded) -> render mesh -> subdivide -> sim ---
    SurfaceMesh render_mesh;
    {
        const int64_t vr = static_cast<int64_t>(state->mesh_cpu.vertex_count());
        render_mesh.positions =
            torch::from_blob(state->mesh_cpu.positions.data(), {vr, 3},
                             torch::kFloat32).clone();
        render_mesh.uvs =
            torch::from_blob(state->mesh_cpu.uvs.data(), {vr, 2},
                             torch::kFloat32).clone();
        render_mesh.indices =
            torch::from_blob(state->mesh_cpu.indices.data(),
                             {static_cast<int64_t>(state->mesh_cpu.indices.size())},
                             torch::kInt32).clone().to(torch::kLong);
    }
    const int64_t V_render = render_mesh.positions.size(0);
    SurfaceMesh sim_mesh = subdivide_midpoint(render_mesh, kSubdiv);
    const int64_t V_sim = sim_mesh.positions.size(0);

    // Calibrate the source deposition so the hero field settles WITHIN the
    // learner's representable range [ambient, ambient+span]; the default gain of
    // 50 drives this asset past 480 °C, well outside span (122 °C), where the
    // net saturates and can never chase (design §6). The twin claim is the
    // dataflow, not FEA — units are model-relative — so this is calibration, not
    // physics (design §3, "calibrated by source_gain, not physical W").
    ThermalConfig cfg;
    cfg.source_gain = 8.0f;
    ThermalSim sim = make_thermal_sim(sim_mesh, kVariants, device, /*seed=*/1, cfg);
    std::optional<ThermalLearner> learner;
    learner.emplace(sim, /*seed=*/7, state->learning_rate.load());

    // Bake matrix + gutter (state -> texture) on device, and the texel 3-D
    // positions the net is evaluated at (gutter-filled so outside-chart texels
    // read a valid nearby surface point rather than the origin).
    BakeResult bake = bake_matrix(sim_mesh, kTexH, kTexW);
    auto S_dev = bake.S.to(device);
    auto gutter_dev = bake.gutter_src.to(device);
    auto texel_pos =
        torch::mm(S_dev, sim.positions).index_select(0, gutter_dev).contiguous();
    auto render_pos = sim.positions.slice(0, 0, V_render).contiguous();

    // Pool-backed display allocations (import-once bases stay stable, §exportable
    // pool). Only the tensors handed to the bridge/geometry live in the pool;
    // the sim/learner tensors are ordinary device allocations.
    std::unique_ptr<caliper::adapters::ExportablePool> pool;
    if (gpu && (state->bridge_caps & CALIPER_BRIDGE_CAP_IMPORT_ALLOC) &&
        (state->geom_caps & CALIPER_GEOM_CAP_PRIMITIVES)) {
        try {
            auto candidate = std::make_unique<caliper::adapters::ExportablePool>(0);
            if (candidate->ok()) pool = std::move(candidate);
        } catch (...) {}
    }

    const int64_t index_count = static_cast<int64_t>(state->mesh_cpu.indices.size());
    std::vector<int32_t> wire_cpu;
    wire_cpu.reserve(state->mesh_cpu.indices.size() * 2);
    for (size_t i = 0; i + 2 < state->mesh_cpu.indices.size(); i += 3) {
        const int32_t a = state->mesh_cpu.indices[i];
        const int32_t b = state->mesh_cpu.indices[i + 1];
        const int32_t c = state->mesh_cpu.indices[i + 2];
        wire_cpu.insert(wire_cpu.end(), {a, b, b, c, c, a});
    }

    torch::Tensor positions, normals, uvs, indices, wire_indices;
    torch::Tensor tex_slot[FIELD_COUNT][kSlots];
    torch::Tensor vert_slot[FIELD_COUNT][kSlots];
    auto fopt = torch::TensorOptions(device).dtype(torch::kFloat32);
    auto iopt = torch::TensorOptions(device).dtype(torch::kInt32);
    auto allocate = [&] {
        positions = torch::empty({V_render, 3}, fopt);
        normals = torch::empty({V_render, 3}, fopt);
        uvs = torch::empty({V_render, 2}, fopt);
        indices = torch::empty({index_count}, iopt);
        wire_indices = torch::empty({static_cast<int64_t>(wire_cpu.size())}, iopt);
        for (int f = 0; f < FIELD_COUNT; ++f)
            for (int s = 0; s < kSlots; ++s) {
                tex_slot[f][s] = torch::empty({kTexH, kTexW}, fopt);
                vert_slot[f][s] = torch::empty({V_render}, fopt);
            }
    };
    if (pool) { auto use_pool = pool->use(); allocate(); }
    else allocate();

    positions.copy_(torch::from_blob(state->mesh_cpu.positions.data(),
        {V_render, 3}, torch::kFloat32).clone().to(device));
    normals.copy_(torch::from_blob(state->mesh_cpu.normals.data(),
        {V_render, 3}, torch::kFloat32).clone().to(device));
    uvs.copy_(torch::from_blob(state->mesh_cpu.uvs.data(),
        {V_render, 2}, torch::kFloat32).clone().to(device));
    indices.copy_(torch::from_blob(state->mesh_cpu.indices.data(),
        {index_count}, torch::kInt32).clone().to(device));
    wire_indices.copy_(torch::from_blob(wire_cpu.data(),
        {static_cast<int64_t>(wire_cpu.size())}, torch::kInt32).clone().to(device));

    // Publish the immutable geometry + metadata once.
    {
        std::lock_guard<std::mutex> lock(state->mutex);
        state->pool = std::move(pool);
        state->positions = positions; state->normals = normals;
        state->uvs = uvs; state->indices = indices;
        state->wire_indices = wire_indices;
        for (int f = 0; f < FIELD_COUNT; ++f)
            for (int s = 0; s < kSlots; ++s) {
                state->tex_slot[f][s] = tex_slot[f][s];
                state->vert_slot[f][s] = vert_slot[f][s];
            }
        state->field_h = kTexH; state->field_w = kTexW;
        state->v_render = V_render; state->v_sim = V_sim;
        state->gpu = gpu;
        state->device_name = cuda ? "CUDA" : mps ? "MPS" : "CPU";
    }
    if (state->host)
        state->host->log_info(("twin-scope: render V=" + std::to_string(V_render) +
            " sim V=" + std::to_string(V_sim) + " on " +
            std::string(cuda ? "CUDA" : mps ? "MPS" : "CPU") +
            " — sim+train starting").c_str());

    auto bake_field = [&](const torch::Tensor& vertfield) {
        auto col = vertfield.reshape({V_sim, 1});
        auto tex = torch::mm(S_dev, col).squeeze(1);   // (H*W,)
        return tex.index_select(0, gutter_dev).reshape({kTexH, kTexW});
    };

    // Local override tracking so we touch the device override tensors only on a
    // real change (direct-store atomics, worker applies).
    std::array<float, kSourceCount> last_ov;
    last_ov.fill(-2.f);
    auto set_lr = [&](float lr) {
        for (auto& g : learner->optimizer->param_groups())
            static_cast<torch::optim::AdamOptions&>(g.options()).lr(lr);
    };
    float applied_lr = state->learning_rate.load();

    int write_slot = 0;
    auto t_start = std::chrono::steady_clock::now();
    auto last_publish = std::chrono::steady_clock::time_point{};
    auto rate_start = t_start;
    auto last_status_log = t_start;
    int rate_sim = 0, rate_train = 0;

    while (!state->stop.load() && !(control && control->cancelled(control))) {
        const auto now = std::chrono::steady_clock::now();
        const double t = std::chrono::duration<double>(now - t_start).count();

        if (state->reset_req.exchange(false)) {
            sim.T.fill_(kAmbient);
            sim.active = sim.active_intensities(t);
            learner.emplace(sim, /*seed=*/7, applied_lr);
            std::lock_guard<std::mutex> lock(state->mutex);
            state->sim_steps = 0; state->train_steps = 0;
            state->loss = 0.f; state->peak = kAmbient;
        }

        const float want_lr = state->learning_rate.load();
        if (want_lr != applied_lr) { applied_lr = want_lr; set_lr(applied_lr); }

        // Apply source overrides (only when changed) to the hero variant.
        bool ov_changed = false;
        for (int k = 0; k < kSourceCount; ++k) {
            const float v = state->src_override[k].load();
            if (v != last_ov[k]) { last_ov[k] = v; ov_changed = true; }
        }
        if (ov_changed) {
            auto flag = torch::empty({kSourceCount}, torch::kBool);
            auto val = torch::empty({kSourceCount}, torch::kFloat32);
            auto fa = flag.accessor<bool, 1>();
            auto va = val.accessor<float, 1>();
            for (int k = 0; k < kSourceCount; ++k) {
                fa[k] = last_ov[k] >= 0.f;
                va[k] = std::clamp(last_ov[k], 0.f, 1.f);
            }
            sim.override_flag.copy_(flag.to(device));
            sim.override_value.copy_(val.to(device));
        }

        if (state->sim_on.load()) { sim.step(t); ++rate_sim; }
        float loss = state->loss;
        if (state->train_on.load()) { loss = learner->train_step(sim); ++rate_train; }

        // Publish ≤30 Hz; sim+train run uncapped between publishes.
        if (last_publish.time_since_epoch().count() == 0 ||
            now - last_publish >= std::chrono::milliseconds(33)) {
            last_publish = now;
            const int mode = state->display_mode.load();
            auto T0 = sim.T.select(0, 0);              // (V_sim,)
            auto s_hero = sim.active.select(0, 0);     // (K,)

            auto sim_texf = bake_field(T0);
            auto net_texf = learner->predict(texel_pos, s_hero).reshape({kTexH, kTexW});
            auto err_texf = (sim_texf - net_texf).abs();
            auto sim_vf = T0.slice(0, 0, V_render).contiguous();
            auto net_vf = learner->predict(render_pos, s_hero);
            auto err_vf = (sim_vf - net_vf).abs();

            // The ONE .to(kCPU) per publish (§8.f): stack all three textures,
            // synchronize once, and compute the peak on the CPU copy. The
            // device→device slot copies below never add a sync.
            auto cpu3 = torch::stack({sim_texf, net_texf, err_texf}, 0)
                            .to(torch::kCPU).contiguous();
            const float* cpu_ptr = cpu3.data_ptr<float>();
            const float peak = cpu3.select(0, 0).max().item<float>();
            auto cpuvec = std::make_shared<std::vector<float>>(
                cpu_ptr, cpu_ptr + static_cast<size_t>(FIELD_COUNT) * kTexH * kTexW);

            tex_slot[FIELD_SIM][write_slot].copy_(sim_texf);
            tex_slot[FIELD_NET][write_slot].copy_(net_texf);
            tex_slot[FIELD_ERR][write_slot].copy_(err_texf);
            vert_slot[FIELD_SIM][write_slot].copy_(sim_vf);
            vert_slot[FIELD_NET][write_slot].copy_(net_vf);
            vert_slot[FIELD_ERR][write_slot].copy_(err_vf);

            std::lock_guard<std::mutex> lock(state->mutex);
            state->cpu_fields = cpuvec;
            state->peak = peak;
            state->loss = loss;
            state->published_mode = mode;   // tag the publish with its mode (§8.d)
            state->ready_slot = write_slot;
            for (int s = 0; s < kSlots; ++s)
                if (s != state->ready_slot && s != state->display_slot) {
                    write_slot = s; break;
                }
        }

        // Uncapped step-rate counters over a 0.5 s window.
        const float elapsed = std::chrono::duration<float>(now - rate_start).count();
        if (elapsed >= 0.5f) {
            std::lock_guard<std::mutex> lock(state->mutex);
            state->sim_sps = static_cast<float>(rate_sim) / elapsed;
            state->train_sps = static_cast<float>(rate_train) / elapsed;
            state->sim_steps += rate_sim;    // running totals for the HUD
            state->train_steps += rate_train;
            rate_start = now; rate_sim = 0; rate_train = 0;
        }

        // Periodic status to the log (run-proof without GUI clicking).
        if (now - last_status_log >= std::chrono::milliseconds(1500)) {
            last_status_log = now;
            float sps, tps, pk, ls; int pm;
            { std::lock_guard<std::mutex> lock(state->mutex);
              sps = state->sim_sps; tps = state->train_sps; pk = state->peak;
              ls = state->loss; pm = state->published_mode; }
            if (state->host)
                state->host->log_info(("twin-scope: sim " + std::to_string((int)sps) +
                    " step/s  train " + std::to_string((int)tps) +
                    " step/s  peak " + std::to_string(pk).substr(0, 5) +
                    " C  loss " + std::to_string(ls).substr(0, 7) +
                    "  published_mode " + std::to_string(pm)).c_str());
        }

        // No hot-loop sleep (§8.f): only idle when nothing is running.
        if (!state->sim_on.load() && !state->train_on.load())
            std::this_thread::sleep_for(std::chrono::milliseconds(16));
    }
}

void twin_job_trampoline(void* user, const CaliperJobControl* control) {
    twin_job(static_cast<TwinScopeState*>(user), control);
}

}  // namespace

// ===========================================================================
// Applet
// ===========================================================================
TwinScopeApplet::TwinScopeApplet() : state_(std::make_unique<TwinScopeState>()) {}
TwinScopeApplet::~TwinScopeApplet() = default;

bool TwinScopeApplet::initialize(caliper::Host& host) {
    auto* state = state_.get();
    state->host = &host;
    state->jobs = caliper::Jobs(host);
    state->bridge = caliper::Bridge(host);
    state->geometry = caliper::Geometry(host);
    state->bridge_caps = state->bridge.caps();
    state->geom_caps = state->geometry.caps();

    const std::string asset = resolve_asset(host);
    std::string error;
    if (!caliper::obj::load_file(asset, state->mesh_cpu, &error)) {
        host.log_error(error.c_str());
        return false;
    }

    // Model offset for the split hero + camera target, from the mesh bounds.
    float minx = 1e30f, maxx = -1e30f, miny = 1e30f, maxy = -1e30f;
    const auto& p = state->mesh_cpu.positions;
    for (size_t i = 0; i + 2 < p.size(); i += 3) {
        minx = std::min(minx, p[i]); maxx = std::max(maxx, p[i]);
        miny = std::min(miny, p[i + 1]); maxy = std::max(maxy, p[i + 1]);
    }
    state->model_offset = 0.62f * (maxx - minx);
    state->cam_target = V3{0.f, 0.5f * (miny + maxy), 0.f};

    // Single source-site definition, read once for frame-thread picking (the
    // sim owns the same table on the worker — twin_model.h defines it ONCE).
    auto sites = source_sites(torch::kCPU);
    auto sa = sites.accessor<float, 2>();
    for (int k = 0; k < kSourceCount; ++k)
        state->sites_cpu[k] = V3{sa[k][0], sa[k][1], sa[k][2]};

    host.log_info("twin-scope: housing loaded; starting the surface thermal twin");
    state->job_id = state->jobs.submit("twin_scope: simulate + learn",
                                       &twin_job_trampoline, state);
    return state->job_id != 0;
}

void TwinScopeApplet::draw_ui() {
    auto* state = state_.get();

    // --- snapshot under the mutex: tensor handles + a shared_ptr, no big copy ---
    torch::Tensor positions, normals, uvs, indices, wire_indices;
    torch::Tensor tex[FIELD_COUNT], vert[FIELD_COUNT];
    caliper::adapters::ExportablePool* pool = nullptr;
    std::shared_ptr<std::vector<float>> cpu_fields;
    int h = 0, w = 0, published_mode = MODE_SPLIT;
    int64_t v_render = 0, v_sim = 0;
    float loss = 0.f, peak = 0.f, sim_sps = 0.f, train_sps = 0.f;
    int64_t sim_steps = 0, train_steps = 0;
    std::string device_name;
    bool have_field = false;
    {
        std::lock_guard<std::mutex> lock(state->mutex);
        if (state->ready_slot >= 0) {
            state->display_slot = state->ready_slot;
            const int s = state->display_slot;
            for (int f = 0; f < FIELD_COUNT; ++f) {
                tex[f] = state->tex_slot[f][s];
                vert[f] = state->vert_slot[f][s];
            }
            have_field = true;
        }
        positions = state->positions; normals = state->normals;
        uvs = state->uvs; indices = state->indices;
        wire_indices = state->wire_indices;
        pool = state->pool.get();
        cpu_fields = state->cpu_fields;
        published_mode = state->published_mode;
        h = state->field_h; w = state->field_w;
        v_render = state->v_render; v_sim = state->v_sim;
        loss = state->loss; peak = state->peak;
        sim_sps = state->sim_sps; train_sps = state->train_sps;
        sim_steps = state->sim_steps; train_steps = state->train_steps;
        device_name = state->device_name;
    }

    const bool has_prim = (state->geom_caps & CALIPER_GEOM_CAP_PRIMITIVES) != 0;
    const bool has_textured = state->geometry.has_textured();
    const bool stream_ordered =
        (state->bridge_caps & CALIPER_BRIDGE_CAP_STREAM_ORDERED) != 0;

    ImGui::SetNextWindowSize(ImVec2(1180.f, 800.f), ImGuiCond_FirstUseEver);
    ImGui::Begin("TwinScope: the surface twin");

    // ----------------------------- toolbar -----------------------------------
    const float toolbar_h = ImGui::GetFrameHeight() * 2.f +
        ImGui::GetStyle().WindowPadding.y * 2.f;
    if (ImGui::BeginChild("##twin_toolbar", ImVec2(0, toolbar_h),
                          ImGuiChildFlags_Borders)) {
        bool sim = state->sim_on.load();
        if (ImGui::Checkbox("simulate", &sim)) state->sim_on.store(sim);
        ImGui::SameLine();
        bool train = state->train_on.load();
        if (ImGui::Checkbox("learn", &train)) state->train_on.store(train);
        ImGui::SameLine();
        if (ImGui::Button("reset")) state->reset_req.store(true);
        ImGui::SameLine();
        float lr = state->learning_rate.load();
        ImGui::SetNextItemWidth(105.f);
        if (ImGui::SliderFloat("lr", &lr, 1e-4f, 1e-2f, "%.4f",
                               ImGuiSliderFlags_Logarithmic))
            state->learning_rate.store(lr);
        ImGui::SameLine();
        ImGui::TextUnformatted("|");
        ImGui::SameLine();

        int mode = state->display_mode.load();
        if (ImGui::RadioButton("split sim|net", mode == MODE_SPLIT))
            state->display_mode.store(MODE_SPLIT);
        ImGui::SameLine();
        if (ImGui::RadioButton("sim", mode == MODE_SIM))
            state->display_mode.store(MODE_SIM);
        ImGui::SameLine();
        if (ImGui::RadioButton("net", mode == MODE_NET))
            state->display_mode.store(MODE_NET);
        ImGui::SameLine();
        if (ImGui::RadioButton("|error|", mode == MODE_ERR))
            state->display_mode.store(MODE_ERR);
        ImGui::SameLine();
        ImGui::TextUnformatted("|");
        ImGui::SameLine();

        // R2 toggle — only when BOTH the textured and primitives caps are live.
        bool textured = state->textured_pref.load();
        if (!has_textured) ImGui::BeginDisabled();
        if (ImGui::Checkbox("textured (v1.2)", &textured))
            state->textured_pref.store(textured);
        if (!has_textured) ImGui::EndDisabled();
        ImGui::SameLine();
        ImGui::Checkbox("wire", &state->wireframe);

        // Second row: sources (direct store — no else-if re-detection, §8.f).
        for (int k = 0; k < kSourceCount; ++k) {
            if (k) ImGui::SameLine();
            ImGui::PushID(k);
            const float cur = state->src_override[k].load();
            bool on = cur > 0.01f;                 // schedule-driven shows as off
            if (ImGui::Checkbox("##on", &on))
                state->src_override[k].store(on ? 1.f : 0.f);
            ImGui::SameLine();
            ImGui::SetNextItemWidth(90.f);
            float held = cur < 0.f ? 0.f : cur;
            const char* label = k == 2 ? "core %.2f"
                              : k == 3 ? "fin %.2f" : "bolt %.2f";
            if (ImGui::SliderFloat("##src", &held, 0.f, 1.f, label))
                state->src_override[k].store(held);
            ImGui::PopID();
        }
        if (ImGui::SmallButton("auto")) {        // release all overrides
            for (int k = 0; k < kSourceCount; ++k) state->src_override[k].store(-1.f);
            state->selected_source = -1;
        }
    }
    ImGui::EndChild();

    // ------------------------------- HUD -------------------------------------
    {
        // Status line uses last frame's provenance (flow_scope 1-frame-lag
        // reporting pattern): the current frame's rung isn't known until after
        // the draw below, so the honest label lags by exactly one frame.
        ImGui::Text("provenance: %s   |   %s   |   render V=%lld  sim V=%lld",
                    device_name.c_str(),
                    (state->logged_prov == PROV_ZEROCOPY) ? "zero-copy imported"
                    : (state->logged_prov == PROV_CPU_TEX) ? "CPU-staged texture"
                    : (state->logged_prov == PROV_PERVERTEX) ? "per-vertex"
                    : (state->logged_prov == PROV_HEATMAP) ? "CPU heatmap" : "…",
                    static_cast<long long>(v_render),
                    static_cast<long long>(v_sim));
        ImGui::SameLine();
        ImGui::TextDisabled("sim %.0f step/s   train %.0f step/s   peak %.1f C   loss %.5f",
                            sim_sps, train_sps, peak, loss);
        // loss sparkline
        if (cpu_fields) {
            if (state->loss_history.size() > 240) state->loss_history.erase(
                state->loss_history.begin());
            state->loss_history.push_back(loss);
        }
        if (!state->loss_history.empty()) {
            ImGui::PlotLines("##loss", state->loss_history.data(),
                             static_cast<int>(state->loss_history.size()), 0,
                             "loss", FLT_MAX, FLT_MAX, ImVec2(220, 34));
            ImGui::SameLine();
        }
        // MAGMA legend with the actual °C temperature range.
        ImDrawList* dl = ImGui::GetWindowDrawList();
        const ImVec2 p0 = ImGui::GetCursorScreenPos();
        const float lw = 150.f, lh = 14.f;
        const ImU32 stops[5] = {
            IM_COL32(0, 0, 4, 255), IM_COL32(81, 18, 124, 255),
            IM_COL32(183, 55, 121, 255), IM_COL32(252, 137, 97, 255),
            IM_COL32(252, 253, 191, 255)};
        for (int i = 0; i < 4; ++i)
            dl->AddRectFilledMultiColor(
                ImVec2(p0.x + lw * i / 4, p0.y),
                ImVec2(p0.x + lw * (i + 1) / 4, p0.y + lh),
                stops[i], stops[i + 1], stops[i + 1], stops[i]);
        ImGui::Dummy(ImVec2(lw, lh));
        ImGui::SameLine();
        ImGui::Text("MAGMA %.0f–%.0f C", kTempMin, kTempMax);
    }

    // --------------------------- geometry view -------------------------------
    const ImVec2 available = ImGui::GetContentRegionAvail();
    const float fb_scale = std::max(1.f, ImGui::GetIO().DisplayFramebufferScale.y);
    const int desired_w = std::clamp(static_cast<int>(available.x * fb_scale), 64, 4096);
    const int desired_h = std::clamp(static_cast<int>(available.y * fb_scale), 64, 4096);
    if (has_prim && available.x >= 64.f && available.y >= 64.f &&
        (state->view == 0 || std::abs(desired_w - state->view_w) >= 3 ||
         std::abs(desired_h - state->view_h) >= 3)) {
        if (state->view) state->geometry.release_view(state->view);
        state->view = state->geometry.create_view_ex(desired_w, desired_h,
                                                     CALIPER_GEOM_VIEW_DEPTH);
        state->view_w = desired_w; state->view_h = desired_h;
    }

    // Which fields to draw, and at what ±x offset (published_mode drives it).
    struct Spec { int field; float offset; };
    std::vector<Spec> specs;
    if (published_mode == MODE_SPLIT) {
        specs.push_back({FIELD_SIM, -state->model_offset});
        specs.push_back({FIELD_NET, +state->model_offset});
    } else {
        const int f = published_mode == MODE_NET ? FIELD_NET
                    : published_mode == MODE_ERR ? FIELD_ERR : FIELD_SIM;
        specs.push_back({f, 0.f});
    }

    // ONE eye computation, used for BOTH the draw and the pick (§8.f).
    const V3 target = state->cam_target;
    const V3 eye = orbit_eye(state->camera_azimuth, state->camera_elevation,
                             state->camera_distance, target);
    const float aspect = static_cast<float>(state->view_w) /
                         std::max(1, state->view_h);
    CaliperGeomCamera camera{};
    look_at(eye, target, {0.f, 1.f, 0.f}, camera.view);
    perspective(45.f * kPi / 180.f, aspect, 0.05f, 60.f, camera.proj);

    bool textured_drew = false, pervertex_drew = false, imported_any = false;
    const bool want_textured = state->textured_pref.load() && has_textured;

    if (state->view && pool && positions.defined() && have_field) {
        auto pref = pool->to_bridge(state->bridge, positions);
        auto nref = pool->to_bridge(state->bridge, normals);
        auto iref = pool->to_bridge(state->bridge, indices);
        auto uref = pool->to_bridge(state->bridge, uvs);
        auto wref = pool->to_bridge(state->bridge, wire_indices);

        auto set_base = [&](CaliperGeomDraw& b, float offset) {
            b.pos_alloc = pref->alloc; b.pos_offset = pref->offset;
            b.vertex_count = static_cast<uint64_t>(v_render);
            b.index_alloc = iref->alloc; b.index_offset = iref->offset;
            b.index_count = static_cast<uint64_t>(indices.numel());
            b.normal_alloc = nref->alloc; b.normal_offset = nref->offset;
            b.topology = CALIPER_GEOM_TOPO_TRIANGLES;
            b.shade_mode = CALIPER_GEOM_SHADE_LAMBERT;
            b.blend_mode = CALIPER_GEOM_BLEND_OPAQUE;
            b.depth_flags = CALIPER_GEOM_DEPTH_TEST | CALIPER_GEOM_DEPTH_WRITE;
            b.model[12] = offset;
        };
        auto wire_of = [&](const CaliperGeomDraw& surf) {
            CaliperGeomDraw wd = surf;
            wd.index_alloc = wref->alloc; wd.index_offset = wref->offset;
            wd.index_count = static_cast<uint64_t>(wire_indices.numel());
            wd.color_mode = CALIPER_GEOM_COLOR_FLAT;
            wd.shade_mode = CALIPER_GEOM_SHADE_UNLIT;
            wd.topology = CALIPER_GEOM_TOPO_LINES;
            wd.depth_flags = CALIPER_GEOM_DEPTH_TEST;
            wd.flat_rgba = 0x70FFFFFFu;
            return wd;
        };

        // Create-once bridge texture per field (range baked from mode_range).
        auto ensure_tex = [&](int field) -> CaliperTextureId {
            if (state->field_tex[field] == 0 && cpu_fields) {
                float vmin, vmax; mode_range(field, vmin, vmax);
                auto d = cpu_field_desc(cpu_fields->data() +
                    static_cast<size_t>(field) * h * w, h, w);
                state->field_tex[field] = state->bridge.texture_from_tensor_mapped(
                    &d, CALIPER_CMAP_MAGMA, vmin, vmax, 0);
            }
            return state->field_tex[field];
        };
        // Update a field texture: zero-copy imported path when STREAM_ORDERED is
        // present (§8.b), else CPU-staged. Returns true if imported.
        auto update_tex = [&](int field, CaliperTextureId t) -> bool {
            if (stream_ordered) {
                auto ref = pool->to_bridge(state->bridge, tex[field]);
                auto desc = caliper::adapters::stream_to_tensor(tex[field],
                                                                state->bridge_caps);
                if (ref && desc &&
                    state->bridge.update_texture_from_alloc(
                        t, ref->alloc, ref->offset, &*desc))
                    return true;
            }
            if (cpu_fields) {
                auto d = cpu_field_desc(cpu_fields->data() +
                    static_cast<size_t>(field) * h * w, h, w);
                state->bridge.update_texture(t, &d);
            }
            return false;
        };

        if (pref && nref && iref) {
            // ---- textured (v1.2) rung ----
            if (want_textured && uref) {
                std::vector<CaliperGeomDrawV1_2> draws;
                for (const Spec& sp : specs) {
                    const CaliperTextureId ft = ensure_tex(sp.field);
                    if (ft == 0) continue;
                    imported_any |= update_tex(sp.field, ft);
                    CaliperGeomDrawV1_2 d = caliper::geom_draw_v1_2_defaults();
                    set_base(d.base, sp.offset);
                    d.base.color_mode = CALIPER_GEOM_COLOR_TEXTURE;
                    d.uv_alloc = uref->alloc; d.uv_offset = uref->offset;
                    d.texture = ft;
                    draws.push_back(d);
                    if (state->wireframe && wref) {
                        CaliperGeomDrawV1_2 wd = caliper::geom_draw_v1_2_defaults();
                        wd.base = wire_of(d.base);
                        draws.push_back(wd);
                    }
                }
                if (!draws.empty())
                    textured_drew = state->geometry.draw_primitives(
                        state->view, camera, draws.data(),
                        static_cast<uint32_t>(draws.size()), 0xFF090B0Eu);
            }

            // ---- per-vertex (v1.1) rung: chosen when textured is off, the cap
            // is absent, OR the textured draw was refused at runtime (§9). ----
            if (!textured_drew) {
                std::vector<CaliperGeomDraw> draws;
                for (const Spec& sp : specs) {
                    auto aref = pool->to_bridge(state->bridge, vert[sp.field]);
                    if (!aref) continue;
                    float vmin, vmax; mode_range(sp.field, vmin, vmax);
                    CaliperGeomDraw d = caliper::geom_draw_defaults();
                    set_base(d, sp.offset);
                    d.attr_alloc = aref->alloc; d.attr_offset = aref->offset;
                    d.color_mode = CALIPER_GEOM_COLOR_COLORMAP;
                    d.colormap = CALIPER_CMAP_MAGMA;
                    d.vmin = vmin; d.vmax = vmax;
                    draws.push_back(d);
                    if (state->wireframe && wref) draws.push_back(wire_of(d));
                }
                if (!draws.empty())
                    pervertex_drew = state->geometry.draw_primitives(
                        state->view, camera, draws.data(),
                        static_cast<uint32_t>(draws.size()), 0xFF090B0Eu);
            }
        }
    }
    const bool geometry_drew = textured_drew || pervertex_drew;

    // provenance (honest, post-draw): claimed only for what actually drew.
    Prov prov = !geometry_drew ? (cpu_fields ? PROV_HEATMAP : PROV_WAIT)
              : textured_drew ? (imported_any ? PROV_ZEROCOPY : PROV_CPU_TEX)
              : PROV_PERVERTEX;
    if (state->logged_prov != prov && state->host) {
        state->logged_prov = prov;
        const char* line =
            prov == PROV_ZEROCOPY ? "twin-scope: zero-copy — primitives drawn from imported allocation(s)"
          : prov == PROV_CPU_TEX  ? "twin-scope: CPU-staged texture (no stream-ordered handoff)"
          : prov == PROV_PERVERTEX ? "twin-scope: per-vertex fallback (no textured geometry)"
          : prov == PROV_HEATMAP  ? "twin-scope: CPU heatmap fallback (no geometry service)"
                                  : "twin-scope: waiting for the first published field";
        state->host->log_info(line);
    }

    // ------------------------------ present ----------------------------------
    if (geometry_drew) {
        if (!state->logged_first_draw && state->host) {
            state->logged_first_draw = true;
            state->host->log_info(("twin-scope: geometry view drawn — " +
                std::to_string(specs.size()) + " mesh half(s), " +
                std::string(textured_drew ? "textured v1.2" : "per-vertex v1.1")).c_str());
        }
        ImGui::Image(caliper::Bridge::imtex(state->view),
                     ImVec2(state->view_w / fb_scale, state->view_h / fb_scale));
        const bool hovered = ImGui::IsItemHovered();
        const ImVec2 item_min = ImGui::GetItemRectMin();
        const ImVec2 item_size = ImGui::GetItemRectSize();
        ImGuiIO& io = ImGui::GetIO();
        if (hovered && ImGui::IsMouseDown(ImGuiMouseButton_Right)) {
            state->camera_azimuth += io.MouseDelta.x * 0.008f;
            state->camera_elevation = std::clamp(
                state->camera_elevation + io.MouseDelta.y * 0.008f, -1.35f, 1.35f);
        }
        if (hovered && io.MouseWheel != 0.f)
            state->camera_distance = std::clamp(
                state->camera_distance * (1.f - io.MouseWheel * 0.08f), 6.f, 40.f);

        // Click → nearest source site toggle; ray cast against the RENDER mesh
        // (each active ±x offset tested by shifting the ray origin). Same eye.
        if (hovered && ImGui::IsMouseClicked(ImGuiMouseButton_Left) &&
            item_size.x > 0.f && item_size.y > 0.f) {
            const float ndc_x = ((io.MousePos.x - item_min.x) / item_size.x) * 2.f - 1.f;
            const float ndc_y = 1.f - ((io.MousePos.y - item_min.y) / item_size.y) * 2.f;
            const V3 ray = cursor_ray(eye, target, 45.f, aspect, ndc_x, ndc_y);
            float closest = 1e30f; V3 hit_local{};
            for (const Spec& sp : specs) {
                const V3 origin = eye - V3{sp.offset, 0.f, 0.f};
                for (size_t i = 0; i + 2 < state->mesh_cpu.indices.size(); i += 3) {
                    auto vp = [&](int idx) {
                        return V3{state->mesh_cpu.positions[idx * 3],
                                  state->mesh_cpu.positions[idx * 3 + 1],
                                  state->mesh_cpu.positions[idx * 3 + 2]}; };
                    const V3 a = vp(state->mesh_cpu.indices[i]);
                    const V3 b = vp(state->mesh_cpu.indices[i + 1]);
                    const V3 c = vp(state->mesh_cpu.indices[i + 2]);
                    float dist, b1, b2;
                    if (intersect_triangle(origin, ray, a, b, c, dist, b1, b2) &&
                        dist < closest) {
                        closest = dist;
                        hit_local = origin + ray * dist;
                    }
                }
            }
            if (closest < 1e29f) {
                int nearest = -1; float best = 0.35f * 0.35f;
                for (int k = 0; k < kSourceCount; ++k) {
                    const V3 d = hit_local - state->sites_cpu[k];
                    const float d2 = dot(d, d);
                    if (d2 < best) { best = d2; nearest = k; }
                }
                state->selected_source = nearest;
                if (nearest >= 0) {   // toggle: leaves the duty cycle either way
                    const float cur = state->src_override[nearest].load();
                    state->src_override[nearest].store(cur > 0.01f ? 0.f : 1.f);
                }
            }
        }
        // Drag → scale the held intensity of the selected source.
        if (hovered && state->selected_source >= 0 &&
            ImGui::IsMouseDown(ImGuiMouseButton_Left) && io.MouseDelta.y != 0.f) {
            auto& src = state->src_override[state->selected_source];
            const float cur = src.load() < 0.f ? 0.5f : src.load();
            src.store(std::clamp(cur - io.MouseDelta.y * 0.01f, 0.f, 1.f));
        }
    } else if (cpu_fields && ImPlot::BeginPlot("##twin_heatmap", ImVec2(-1, -1),
                                               ImPlotFlags_NoInputs)) {
        // Lowest rung: 2-D heatmap of the primary field (mesh_scope ladder).
        const int pf = published_mode == MODE_NET ? FIELD_NET
                     : published_mode == MODE_ERR ? FIELD_ERR : FIELD_SIM;
        float vmin, vmax; mode_range(pf, vmin, vmax);
        ImPlot::SetupAxes(nullptr, nullptr, ImPlotAxisFlags_NoDecorations,
                          ImPlotAxisFlags_NoDecorations);
        ImPlot::PlotHeatmap("synthetic surface heat field",
                            cpu_fields->data() + static_cast<size_t>(pf) * h * w,
                            h, w, vmin, vmax, nullptr,
                            ImPlotPoint(0, 0), ImPlotPoint(1, 1));
        ImPlot::EndPlot();
        ImGui::TextDisabled("CPU heatmap fallback (no geometry service)");
    } else {
        ImGui::TextDisabled("waiting for the first synthetic surface heat field");
    }
    ImGui::End();
}

void TwinScopeApplet::cleanup() {
    auto* state = state_.get();
    state->stop.store(true);
    if (state->job_id) {
        state->jobs.request_cancel(state->job_id);
        for (int i = 0; i < 3000 && state->jobs.is_running(state->job_id); ++i)
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    if (state->view) state->geometry.release_view(state->view);
    for (int f = 0; f < FIELD_COUNT; ++f)
        if (state->field_tex[f]) state->bridge.release_texture(state->field_tex[f]);
    state->view = 0;
    for (int f = 0; f < FIELD_COUNT; ++f) state->field_tex[f] = 0;
    {
        std::lock_guard<std::mutex> lock(state->mutex);
        state->positions = torch::Tensor();
        state->normals = torch::Tensor();
        state->uvs = torch::Tensor();
        state->indices = torch::Tensor();
        state->wire_indices = torch::Tensor();
        for (int f = 0; f < FIELD_COUNT; ++f)
            for (int s = 0; s < kSlots; ++s) {
                state->tex_slot[f][s] = torch::Tensor();
                state->vert_slot[f][s] = torch::Tensor();
            }
    }
    // The pool import lifetime rule (donor): if the worker is somehow still
    // live, retain the pool deliberately rather than free memory it may read.
    if (state->job_id && state->jobs.is_running(state->job_id)) {
        (void)state->pool.release();
        if (state->host) state->host->log_error(
            "twin-scope: worker still running; export pool deliberately retained");
    } else {
        state->pool.reset();
    }
    if (state->host) state->host->log_info("twin-scope: on_cleanup");
}

}  // namespace twinscope
