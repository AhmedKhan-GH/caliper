// live_views.cpp: embed ABI v1.3 typed live views (caliper/live_views.h): the
// C entry points, per-kind apply, slot-claim bookkeeping and lifetime. The
// ImGui/ImPlot windows live in live_views_draw.cpp.
//
// ONE VIEW LAYER, TRANSPORT-BLIND. Both transports (the shm session of another
// process, the in-process window's caliper-made shared MTLBuffers) end in
// caliper_core_live_apply on the frame thread. Nothing here waits on the GPU
// unless CALIPER_LIVE_APPLY_WAIT_GPU asks (tests): every completion question is
// a status poll of a retained command buffer.
//
// SLOT-CLAIM RULE PER KIND (what read_done() answers; 1 = the transport may
// rewrite every slot this view was given, except as noted):
//   HEATMAP / IMAGE  the colormap/blit kernel of each apply reads the payload
//                    once, into the view's own texture. read_done = 1 once the
//                    device update of EVERY apply so far has completed, so the
//                    last applied payload is free too. CPU payloads are copied
//                    at apply (always 1).
//   POINTS           the current payload is re-read by the GPU whenever the
//                    view redraws, so it stays claimed for as long as it is
//                    current. read_done = 1 once the last frame that drew any
//                    SUPERSEDED payload has retired (their imports are released
//                    then). A CPU payload is staged into a view-owned buffer at
//                    apply and is free at once.
//   SERIES/HISTOGRAM copied at apply; always 1.
#include "caliper/live_views.h"

#include "embed/embed_core.h"
#include "live/live_views_state.h"
#include "renderer/host_renderer.h"
#include "tensor_bridge.h"   // colormap_lut

#include <caliper/services/tensor_bridge_v1_2.h>   // CALIPER_ALLOC_HANDLE_MTLBUFFER

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <thread>

using caliper_embed_views::TensorView;
using caliper_host::HostRenderer;

#ifndef __APPLE__
namespace caliper_live_views {
BufferInfo metal_buffer_info(void*) { return {}; }
void* metal_new_shared_buffer(void*, const void*, uint64_t) { return nullptr; }
void metal_release(void*) {}
uint64_t now_ns() {
    return (uint64_t)std::chrono::duration_cast<std::chrono::nanoseconds>(
               std::chrono::steady_clock::now().time_since_epoch()).count();
}
}  // namespace caliper_live_views
#endif

namespace caliper_live_views {

namespace {

constexpr uint32_t kKnownFlags = CALIPER_LIVE_APPLY_RAW | CALIPER_LIVE_APPLY_AUTO_VMIN |
                                 CALIPER_LIVE_APPLY_AUTO_VMAX | CALIPER_LIVE_APPLY_WAIT_GPU;
constexpr int64_t kMaxTexDim = 16384;

const char* kind_name(int32_t k) {
    switch (k) {
        case CALIPER_LIVE_KIND_HEATMAP:   return "heatmap";
        case CALIPER_LIVE_KIND_IMAGE:     return "image";
        case CALIPER_LIVE_KIND_POINTS:    return "points";
        case CALIPER_LIVE_KIND_SERIES:    return "series";
        case CALIPER_LIVE_KIND_HISTOGRAM: return "histogram";
        default:                          return "?";
    }
}

bool texture_kind(int32_t k) {
    return k == CALIPER_LIVE_KIND_HEATMAP || k == CALIPER_LIVE_KIND_IMAGE;
}

const char* dtype_name(CaliperDType d) {
    switch (d) {
        case CALIPER_DT_F32:  return "f32";
        case CALIPER_DT_F16:  return "f16";
        case CALIPER_DT_BF16: return "bf16";
        case CALIPER_DT_I64:  return "i64";
        case CALIPER_DT_I32:  return "i32";
        case CALIPER_DT_U8:   return "u8";
    }
    return "unknown";
}

std::string shape_str(const CaliperTensor& t) {
    std::string s = "(";
    for (int i = 0; i < t.ndim && i < 8; ++i) {
        if (i) s += ",";
        s += std::to_string(t.shape[i]);
    }
    return s + ")";
}

std::string describe(const CaliperTensor& t) {
    return "ndim=" + std::to_string(t.ndim) + " shape=" + shape_str(t) +
           " dtype=" + dtype_name(t.dtype);
}

bool contiguous(const CaliperTensor& t) {
    int64_t expect = 1;
    for (int i = t.ndim - 1; i >= 0; --i) {
        if (t.shape[i] != 1 && t.strides[i] != expect) return false;
        expect *= t.shape[i];
    }
    return true;
}

CaliperLiveViewsState* state(CaliperCore* core) {
    if (!core->live_views) core->live_views = new CaliperLiveViewsState();
    return core->live_views;
}

TensorView* find_v12(CaliperCore* core, const std::string& name) {
    for (auto& v : core->views)
        if (v.name == name) return &v;
    return nullptr;
}

// ---- releases ------------------------------------------------------------
void unref_import(CaliperCore* core, View& v, Import* im) {
    if (!im || --im->refs > 0) return;
    if (core->renderer && im->rid) core->renderer->release_external_allocation(im->rid);
    metal_release(im->owned);
    v.imports.erase(std::remove_if(v.imports.begin(), v.imports.end(),
                                   [&](const std::unique_ptr<Import>& p) { return p.get() == im; }),
                    v.imports.end());
}

void release_payload(CaliperCore* core, View& v, PointsPayload& p) {
    if (p.last_draw && core->renderer) core->renderer->device_update_release(p.last_draw);
    p.last_draw = nullptr;
    unref_import(core, v, p.pos);
    unref_import(core, v, p.aux);
    p = PointsPayload{};
}

// Every GPU resource and handle of a view (telemetry and placement stay).
void release_typed(CaliperCore* core, View& v) {
    HostRenderer* r = core->renderer.get();
    for (void* h : v.pending)
        if (r) r->device_update_release(h);
    v.pending.clear();
    if (v.have_cur) release_payload(core, v, v.cur);
    v.have_cur = false;
    for (auto& p : v.retired) release_payload(core, v, p);
    v.retired.clear();
    for (auto& im : v.imports) {   // empty unless a payload leaked a ref
        if (r && im->rid) r->release_external_allocation(im->rid);
        metal_release(im->owned);
    }
    v.imports.clear();
    if (r && v.view_rid) r->tex_release(v.view_rid);
    v.view_rid = 0;
    v.tex_w = v.tex_h = 0;
    if (v.owns_v12 && find_v12(core, v.name)) caliper_core_view_remove(core, v.name.c_str());
    v.owns_v12 = false;
    v.values.clear();
    v.xs.clear();
    v.ys.clear();
}

void erase_view(CaliperCore* core, CaliperLiveViewsState* S, View* v) {
    release_typed(core, *v);
    S->views.erase(std::remove_if(S->views.begin(), S->views.end(),
                                  [&](const std::unique_ptr<View>& p) { return p.get() == v; }),
                   S->views.end());
}

// ---- mapping range -------------------------------------------------------
// AUTO ends from stats[0..1] (the producer's GPU aminmax), fixed ends from the
// snapshot. A non-finite AUTO range (an all-NaN tensor) keeps the previous
// finite one; a flat AUTO range widens by +-0.5 (like _convert.py).
bool resolve_range(View& v, const CaliperLiveSnapshot& s, float* lo, float* hi,
                   std::string* why) {
    const bool amin = (s.flags & CALIPER_LIVE_APPLY_AUTO_VMIN) != 0;
    const bool amax = (s.flags & CALIPER_LIVE_APPLY_AUTO_VMAX) != 0;
    float a = s.vmin, b = s.vmax;
    if (amin) {
        if (!s.stats || s.stats_count < 1) {
            *why = "AUTO_VMIN needs stats[0] (stats is NULL or stats_count < 1)";
            return false;
        }
        a = s.stats[0];
    }
    if (amax) {
        if (!s.stats || s.stats_count < 2) {
            *why = "AUTO_VMAX needs stats[1] (stats is NULL or stats_count < 2)";
            return false;
        }
        b = s.stats[1];
    }
    if (!amin && !std::isfinite(a)) { *why = "vmin must be finite"; return false; }
    if (!amax && !std::isfinite(b)) { *why = "vmax must be finite"; return false; }
    if (amin || amax) {
        if (!std::isfinite(a) || !std::isfinite(b)) {
            a = v.have_range ? v.vmin : 0.0f;
            b = v.have_range ? v.vmax : 1.0f;
        } else {
            if (a == b) { a -= 0.5f; b += 0.5f; }
            v.vmin = a;
            v.vmax = b;
            v.have_range = true;
        }
    }
    *lo = a;
    *hi = b;
    return true;
}

// ---- per-kind apply ------------------------------------------------------
bool apply_texture(CaliperCore* core, View& v, const CaliperLiveSnapshot& s,
                   std::string* why) {
    // Canonical row-major strides (validate() accepted any stride on a size-1
    // dim, which the v1.2 gate would refuse).
    CaliperTensor t = *s.data;
    t.struct_size = sizeof t;
    int64_t st = 1;
    for (int i = t.ndim - 1; i >= 0; --i) { t.strides[i] = st; st *= t.shape[i]; }
    CaliperViewDesc d{};
    d.struct_size = sizeof d;
    d.name = s.name;
    d.tensor = &t;
    d.byte_offset = s.data_offset;
    d.flags = ((s.flags & CALIPER_LIVE_APPLY_RAW) ? CALIPER_VIEW_FLAG_RAW : 0u) |
              ((s.flags & CALIPER_LIVE_APPLY_WAIT_GPU) ? CALIPER_VIEW_FLAG_WAIT_GPU : 0u);
    if (s.kind == CALIPER_LIVE_KIND_HEATMAP) {
        d.colormap = s.colormap;
        if (!resolve_range(v, s, &d.vmin, &d.vmax, why)) return false;
    } else {
        d.colormap = 0;
        d.vmin = 0.0f;
        d.vmax = 1.0f;
    }
    if (!caliper_core_view_tensor(core, &d)) {
        std::string e = core->last_error;
        const std::string pre = "view_tensor('" + v.name + "'): ";
        if (e.compare(0, pre.size(), pre) == 0) e = e.substr(pre.size());
        *why = e + " (embed v1.2 view_tensor)";
        return false;
    }
    v.owns_v12 = true;
    const TensorView* tv = find_v12(core, v.name);
    v.imported = tv && tv->imported;
    v.path = tv ? tv->path : "";
    v.count = 0;
    // Hover values: the CPU side of a METAL payload (NULL for private storage
    // and for CPU payloads, whose bytes the caller may free on return). Read
    // only while the v1.2 view still imports this very buffer.
    v.tex_buf = nullptr;
    v.tex_cpu = nullptr;
    if (s.data->device == CALIPER_DEV_METAL) {
        const BufferInfo bi = metal_buffer_info(s.data->data);
        if (bi.contents) {
            v.tex_buf = s.data->data;
            v.tex_cpu = (const uint8_t*)bi.contents + s.data_offset;
        }
    }
    // The device update was committed on the renderer's queue and not waited
    // on: keep a +1 handle on its command buffer until it retires (WAIT_GPU
    // already waited, and dropped the renderer's own reference).
    if (v.imported && !(s.flags & CALIPER_LIVE_APPLY_WAIT_GPU)) {
        if (void* h = core->renderer->device_update_retain_last()) v.pending.push_back(h);
    }
    return true;
}

// Bind one points buffer (positions or aux) as a renderer import: METAL is
// imported in place (cached by buffer pointer, the whole buffer so every slot
// of a shm segment shares it); CPU is staged into a fresh view-owned shared
// MTLBuffer (copied now, so the caller's bytes are free on return).
bool bind_buffer(CaliperCore* core, View& v, const CaliperTensor& t, uint64_t off,
                 uint64_t bytes, const char* what, Import** out, uint64_t* out_off,
                 const uint8_t** out_cpu, std::string* why) {
    HostRenderer* r = core->renderer.get();
    if (t.device == CALIPER_DEV_METAL) {
        const BufferInfo bi = metal_buffer_info(t.data);
        if (!bi.ok) { *why = std::string(what) + ": not an MTLBuffer"; return false; }
        if (off > bi.length || bytes > bi.length - off) {
            *why = std::string(what) + ": window [" + std::to_string(off) + ", " +
                   std::to_string(off + bytes) + ") exceeds the MTLBuffer length " +
                   std::to_string(bi.length);
            return false;
        }
        Import* im = nullptr;
        for (auto& p : v.imports)
            if (!p->owned && p->buf == t.data && p->bytes >= bi.length) { im = p.get(); break; }
        if (!im) {
            const uint64_t rid = r->import_external_allocation(t.data, bi.length,
                                                               CALIPER_ALLOC_HANDLE_MTLBUFFER);
            if (!rid) {
                *why = std::string(what) + ": MTLBuffer import refused (the buffer lives "
                       "on another GPU, or the renderer cannot import)";
                return false;
            }
            auto p = std::make_unique<Import>();
            p->buf = t.data;
            p->rid = rid;
            p->bytes = bi.length;
            im = p.get();
            v.imports.push_back(std::move(p));
        }
        ++im->refs;
        *out = im;
        *out_off = off;
        *out_cpu = bi.contents ? (const uint8_t*)bi.contents + off : nullptr;
        return true;
    }
    void* dev = r->native_device();
    if (!dev) { *why = std::string(what) + ": CPU payload needs a Metal renderer"; return false; }
    void* buf = metal_new_shared_buffer(dev, (const uint8_t*)t.data + off, bytes);
    if (!buf) { *why = std::string(what) + ": staging buffer allocation failed"; return false; }
    const uint64_t rid = r->import_external_allocation(buf, bytes, CALIPER_ALLOC_HANDLE_MTLBUFFER);
    if (!rid) {
        metal_release(buf);
        *why = std::string(what) + ": staging buffer import refused";
        return false;
    }
    auto p = std::make_unique<Import>();
    p->buf = buf;
    p->rid = rid;
    p->bytes = bytes;
    p->owned = buf;
    p->refs = 1;
    *out = p.get();
    *out_off = 0;
    *out_cpu = (const uint8_t*)metal_buffer_info(buf).contents;
    v.imports.push_back(std::move(p));
    return true;
}

bool apply_points(CaliperCore* core, View& v, const CaliperLiveSnapshot& s,
                  std::string* why) {
    const CaliperTensor& d = *s.data;
    const uint64_t n = (uint64_t)d.shape[0];
    const bool has_aux = s.aux != nullptr;
    const bool categorical = has_aux && s.label_count > 0;

    // Mapping first (pure validation), so a refusal acquires nothing.
    float lo = 0.0f, hi = 255.0f;
    if (has_aux && !categorical && !resolve_range(v, s, &lo, &hi, why)) return false;

    PointsPayload p;
    p.n = n;
    if (!bind_buffer(core, v, d, s.data_offset, n * 12u, "positions", &p.pos, &p.pos_off,
                     &p.pos_cpu, why)) {
        return false;
    }
    if (has_aux && !bind_buffer(core, v, *s.aux, s.aux_offset, n * 4u, "aux", &p.aux,
                                &p.aux_off, &p.aux_cpu, why)) {
        release_payload(core, v, p);
        return false;
    }

    // Commit: the current payload becomes superseded (still claimed until the
    // last frame that drew it retires); the new one is drawn from the next frame.
    if (v.have_cur) v.retired.push_back(v.cur);
    const bool first = !v.have_cur;
    v.cur = p;
    v.have_cur = true;
    if (v.dims != s.points_dims) v.fit_pending = true;   // 2-D <-> 3-D: new camera model
    v.dims = s.points_dims;
    v.label_count = has_aux ? s.label_count : 0;
    v.colormap = s.colormap;
    if (categorical) {
        // Label L (an integer 0..255 stored as f32) maps through the shader's
        // idx = (uint)(clamp((L - 0) / (255 - 0), 0, 1) * 255 + 0.5) == L
        // exactly (the f32 error of L/255*255 is far below 0.5), and LUT entry
        // L holds palette colour L mod 16. Labels above 255 clamp to entry 255.
        for (int i = 0; i < 256; ++i) v.lut[i] = kPalette[i % 16];
        v.pvmin = 0.0f;
        v.pvmax = 255.0f;
    } else if (has_aux) {
        std::memcpy(v.lut, caliper_host::colormap_lut(s.colormap), sizeof v.lut);
        v.pvmin = lo;
        v.pvmax = hi;
    }
    if (s.stats && s.stats_count >= 8) {
        bool fin = true;
        for (int i = 0; i < 6; ++i) fin = fin && std::isfinite(s.stats[2 + i]);
        if (fin) {
            std::memcpy(v.bbox, s.stats + 2, sizeof v.bbox);
            if (!v.has_bbox) v.fit_pending = true;   // first bbox seen: fit to it
            v.has_bbox = true;
        }
    }
    if (first) v.fit_pending = true;
    v.dirty = true;
    v.hover_dirty = true;
    v.count = (int64_t)n;
    v.imported = d.device == CALIPER_DEV_METAL;
    v.path = v.imported ? "points-imported" : "points-cpu-staged";

    poll_handles(core, v);   // a superseded payload that was never drawn retires now
    if (s.flags & CALIPER_LIVE_APPLY_WAIT_GPU) {
        // Tests only: spin on the completion probe until every superseded
        // payload's last frame retired (bounded; never on a default path).
        const auto t_end = std::chrono::steady_clock::now() + std::chrono::seconds(5);
        while (!v.retired.empty() && std::chrono::steady_clock::now() < t_end) {
            std::this_thread::sleep_for(std::chrono::microseconds(100));
            poll_handles(core, v);
        }
    }
    return true;
}

// CPU pointer to `bytes` of a SERIES/HISTOGRAM payload, or NULL with *why.
const uint8_t* cpu_view(const CaliperTensor& t, uint64_t off, uint64_t bytes,
                        int32_t kind, std::string* why) {
    if (t.device == CALIPER_DEV_CPU) return (const uint8_t*)t.data + off;
    const BufferInfo bi = metal_buffer_info(t.data);
    if (!bi.ok) { *why = "not an MTLBuffer"; return nullptr; }
    if (!bi.contents) {
        *why = std::string(kind_name(kind)) + " needs a CPU-visible payload: the MTLBuffer has "
               "private storage ([buffer contents] is NULL); hand over a shared-storage slot";
        return nullptr;
    }
    if (off > bi.length || bytes > bi.length - off) {
        *why = "window [" + std::to_string(off) + ", " + std::to_string(off + bytes) +
               ") exceeds the MTLBuffer length " + std::to_string(bi.length);
        return nullptr;
    }
    return (const uint8_t*)bi.contents + off;
}

bool apply_series(View& v, const CaliperLiveSnapshot& s, std::string* why) {
    const CaliperTensor& d = *s.data;
    const uint64_t M = (uint64_t)d.shape[0];
    const uint8_t* base = cpu_view(d, s.data_offset, M * 4u, s.kind, why);
    if (!base) return false;
    const float* ring = (const float*)base;
    const uint64_t total = s.series_total;
    const uint64_t count = std::min(total, M);
    const uint64_t first = total - count;
    v.values.resize((size_t)count);
    for (uint64_t j = 0; j < count; ++j) v.values[(size_t)j] = ring[(first + j) % M];
    v.first = first;
    v.count = (int64_t)count;
    v.path = "cpu-read";
    v.imported = false;
    v.draw_dirty = true;
    return true;
}

bool apply_histogram(View& v, const CaliperLiveSnapshot& s, std::string* why) {
    const CaliperTensor& d = *s.data;
    const uint64_t B = (uint64_t)d.shape[0];
    const uint8_t* base = cpu_view(d, s.data_offset, B * 4u, s.kind, why);
    if (!base) return false;
    const float* bins = (const float*)base;
    v.values.assign(bins, bins + B);
    // Range: stats[0..1] (lo, hi). Missing or non-finite: bin indices [0, B).
    float lo = 0.0f, hi = (float)B;
    if (s.stats && s.stats_count >= 2 && std::isfinite(s.stats[0]) &&
        std::isfinite(s.stats[1])) {
        lo = s.stats[0];
        hi = s.stats[1];
        if (hi < lo) std::swap(lo, hi);
        if (hi == lo) { lo -= 0.5f; hi += 0.5f; }
    }
    v.lo = lo;
    v.hi = hi;
    v.count = (int64_t)B;
    v.path = "cpu-read";
    v.imported = false;
    v.draw_dirty = true;
    return true;
}

// ---- validation ------------------------------------------------------------
// Every gate that does not need GPU state, with a specific reason.
bool validate(CaliperCore* core, const CaliperLiveSnapshot& s, std::string* why) {
    auto no = [&](const std::string& w) { *why = w; return false; };
    if (s.kind < CALIPER_LIVE_KIND_HEATMAP || s.kind > CALIPER_LIVE_KIND_HISTOGRAM)
        return no("bad kind " + std::to_string(s.kind) +
                  " (1 heatmap, 2 image, 3 points, 4 series, 5 histogram)");
    if (s.flags & ~kKnownFlags) {
        char b[96];
        std::snprintf(b, sizeof b, "unknown flag bits 0x%x (CALIPER_LIVE_APPLY_* only)",
                      (unsigned)(s.flags & ~kKnownFlags));
        return no(b);
    }
    if (s.stats_count < 0) return no("stats_count must be >= 0");
    if (s.stats_count > 0 && !s.stats) return no("stats is NULL but stats_count > 0");
    if (!s.data) return no("null data tensor");
    const CaliperTensor& d = *s.data;
    const char* kn = kind_name(s.kind);
    if (d.ndim < 1 || d.ndim > 8) return no("data ndim must be 1..8; got " + std::to_string(d.ndim));
    if (d.device != CALIPER_DEV_CPU && d.device != CALIPER_DEV_METAL)
        return no("device must be CPU or METAL; got device kind " + std::to_string(d.device));
    const bool metal = d.device == CALIPER_DEV_METAL;
    if (!d.data) return no(metal ? "null MTLBuffer (data->data)" : "null data pointer");
    if (metal && core->renderer->interop_device() != CALIPER_DEV_METAL)
        return no("METAL payload, but the active renderer is not Metal");
    for (int i = 0; i < d.ndim; ++i)
        if (d.shape[i] < 1) return no(std::string("every dim must be >= 1; got shape ") + shape_str(d));
    if (!contiguous(d))
        return no("payload must be row-major contiguous (strides in elements); make it "
                  "contiguous first");
    const bool f32 = d.dtype == CALIPER_DT_F32;
    if ((f32 || metal) && s.data_offset % 4 != 0)
        return no("data_offset must be a multiple of 4; got " + std::to_string(s.data_offset));

    switch (s.kind) {
        case CALIPER_LIVE_KIND_HEATMAP:
            if (!f32 || d.ndim != 2)
                return no(std::string(kn) + " wants a 2-D f32 (H,W) payload; got " + describe(d));
            if (d.shape[0] > kMaxTexDim || d.shape[1] > kMaxTexDim)
                return no("dims must be in 1..16384; got " + shape_str(d));
            if (!caliper_host::colormap_lut(s.colormap))
                return no("bad colormap " + std::to_string(s.colormap) + " (0 viridis, 1 magma, 2 rdbu)");
            break;
        case CALIPER_LIVE_KIND_IMAGE:
            if (d.dtype != CALIPER_DT_U8 || d.ndim != 3 || d.shape[2] != 4)
                return no(std::string(kn) + " wants a 3-D u8 (H,W,4) RGBA payload; got " + describe(d));
            if (d.shape[0] > kMaxTexDim || d.shape[1] > kMaxTexDim)
                return no("dims must be in 1..16384; got " + shape_str(d));
            break;
        case CALIPER_LIVE_KIND_POINTS: {
            if (!f32 || d.ndim != 2 || d.shape[1] != 3)
                return no(std::string(kn) + " wants a 2-D f32 (N,3) payload (2-D callers pad "
                          "z = 0); got " + describe(d));
            if ((uint64_t)d.shape[0] > 0xffffffffull)
                return no("points: N must be <= 2^32-1; got " + std::to_string(d.shape[0]));
            if (s.points_dims != 2 && s.points_dims != 3)
                return no("points_dims must be 2 or 3; got " + std::to_string(s.points_dims));
            if (s.label_count < 0)
                return no("label_count must be >= 0; got " + std::to_string(s.label_count));
            if (metal && s.data_offset / 4 > 0xffffffffull)
                return no("points: data_offset / 4 must fit 32 bits");
            if (s.label_count > 0 && !s.aux)
                return no("label_count > 0 needs aux labels (N,) f32; aux is NULL");
            if (s.aux) {
                const CaliperTensor& a = *s.aux;
                if (a.dtype != CALIPER_DT_F32 || a.ndim != 1 || a.shape[0] != d.shape[0])
                    return no("aux must be (N,) f32 with N = " + std::to_string(d.shape[0]) +
                              "; got " + describe(a));
                if (!contiguous(a)) return no("aux must be contiguous");
                if (a.device != d.device)
                    return no("aux must live on the same device as data (both CPU or both METAL)");
                if (!a.data) return no(metal ? "null aux MTLBuffer" : "null aux pointer");
                if (s.aux_offset % 4 != 0)
                    return no("aux_offset must be a multiple of 4; got " + std::to_string(s.aux_offset));
                if (metal && s.aux_offset / 4 > 0xffffffffull)
                    return no("points: aux_offset / 4 must fit 32 bits");
                if (s.label_count == 0 && !caliper_host::colormap_lut(s.colormap))
                    return no("bad colormap " + std::to_string(s.colormap) +
                              " (0 viridis, 1 magma, 2 rdbu)");
            }
            if (!core->renderer->supports_geometry_primitives())
                return no("points: the renderer has no geometry pipeline (points need Metal)");
            break;
        }
        case CALIPER_LIVE_KIND_SERIES:
        case CALIPER_LIVE_KIND_HISTOGRAM:
            if (!f32 || d.ndim != 1)
                return no(std::string(kn) + " wants a 1-D f32 payload; got " + describe(d));
            break;
    }
    return true;
}

double ms_since(std::chrono::steady_clock::time_point t0) {
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
}

}  // namespace

// ---- shared with live_views_draw.cpp -----------------------------------
View* find(CaliperCore* core, const char* name) {
    if (!core->live_views || !name) return nullptr;
    for (auto& v : core->live_views->views)
        if (v->name == name) return v.get();
    return nullptr;
}

void poll_handles(CaliperCore* core, View& v) {
    HostRenderer* r = core->renderer.get();
    if (!r) return;
    for (size_t i = 0; i < v.pending.size();) {
        if (r->device_update_is_done(v.pending[i])) {
            r->device_update_release(v.pending[i]);
            v.pending.erase(v.pending.begin() + (std::ptrdiff_t)i);
        } else {
            ++i;
        }
    }
    for (size_t i = 0; i < v.retired.size();) {
        if (r->device_update_is_done(v.retired[i].last_draw)) {   // NULL (never drawn) = done
            PointsPayload p = v.retired[i];
            v.retired.erase(v.retired.begin() + (std::ptrdiff_t)i);
            release_payload(core, v, p);
        } else {
            ++i;
        }
    }
}

void fit_points(View& v, float aspect) {
    float b[6] = {-1.0f, 1.0f, -1.0f, 1.0f, -1.0f, 1.0f};
    bool have = false;
    if (v.has_bbox) {
        std::memcpy(b, v.bbox, sizeof b);
        have = true;
    } else if (v.have_cur && v.cur.pos_cpu) {
        // No producer bbox: scan the CPU-visible payload (sampled above 4M).
        const float* P = (const float*)v.cur.pos_cpu;
        const uint64_t n = v.cur.n;
        const uint64_t stride = n > (4u << 20) ? (n + (4u << 20) - 1) / (4u << 20) : 1;
        float lo[3] = {INFINITY, INFINITY, INFINITY}, hi[3] = {-INFINITY, -INFINITY, -INFINITY};
        for (uint64_t i = 0; i < n; i += stride) {
            const float* q = P + 3 * i;
            if (!std::isfinite(q[0]) || !std::isfinite(q[1]) || !std::isfinite(q[2])) continue;
            for (int k = 0; k < 3; ++k) {
                lo[k] = std::min(lo[k], q[k]);
                hi[k] = std::max(hi[k], q[k]);
            }
            have = true;
        }
        if (have)
            for (int k = 0; k < 3; ++k) { b[2 * k] = lo[k]; b[2 * k + 1] = hi[k]; }
    }
    (void)have;
    if (v.dims == 2) b[4] = b[5] = 0.0f;
    const float cx = 0.5f * (b[0] + b[1]), cy = 0.5f * (b[2] + b[3]), cz = 0.5f * (b[4] + b[5]);
    const float ex = 0.5f * std::fabs(b[1] - b[0]), ey = 0.5f * std::fabs(b[3] - b[2]),
                ez = 0.5f * std::fabs(b[5] - b[4]);
    Camera& c = v.cam;
    c.tx = cx;
    c.ty = cy;
    c.tz = cz;
    if (!(aspect > 0.0f)) aspect = 1.0f;
    if (v.dims == 3) {
        float r = std::sqrt(ex * ex + ey * ey + ez * ez);
        if (!(r > 1e-12f) || !std::isfinite(r)) r = 1.0f;
        c.radius = r;
        const float half_fovy = 22.5f * 3.14159265f / 180.0f;
        const float half_fovx = std::atan(std::tan(half_fovy) * aspect);
        c.dist = r / std::sin(std::min(half_fovy, half_fovx)) * 1.05f;
    } else {
        float hh = std::max(ey, ex / aspect);
        if (!(hh > 1e-12f) || !std::isfinite(hh)) hh = 1.0f;
        c.half_h = hh * 1.08f;
        c.radius = std::max(std::max(ex, ey), 1e-6f);
    }
    v.fit_pending = false;
    v.dirty = true;
    v.hover_dirty = true;
}

}  // namespace caliper_live_views

using namespace caliper_live_views;

// ===========================================================================
// C ABI
// ===========================================================================
extern "C" int caliper_core_live_apply(CaliperCore* core, const CaliperLiveSnapshot* snap) {
    if (!core) return 0;
    const auto t0 = std::chrono::steady_clock::now();
    if (!snap || snap->struct_size < sizeof(CaliperLiveSnapshot)) {
        core->fail("live_apply: null/short CaliperLiveSnapshot (struct_size " +
                   std::to_string(snap ? snap->struct_size : 0) + " < " +
                   std::to_string(sizeof(CaliperLiveSnapshot)) + ")");
        return 0;
    }
    if (!snap->name || !snap->name[0]) {
        core->fail("live_apply: name must be a non-empty string");
        return 0;
    }
    const CaliperLiveSnapshot& s = *snap;
    const std::string name = s.name;
    auto lfail = [&](const std::string& why) {
        core->fail("live_apply('" + name + "'): " + why);
        return 0;
    };
    if (!core->canvas_ready || !core->renderer)
        return lfail("attach a canvas first (the renderer's device exists only after attach_canvas)");
    std::string why;
    if (!validate(core, s, &why)) return lfail(why);

    CaliperLiveViewsState* S = state(core);
    View* v = find(core, name.c_str());
    if (!texture_kind(s.kind) && (!v || !v->owns_v12) && find_v12(core, name))
        return lfail("the name is taken by an embed v1.2 view (caliper.show); remove it or "
                     "pick another name");

    if (S->paused) {
        // Display frozen: accept (the producer's claims progress), count, drop.
        if (!v) {
            auto nv = std::make_unique<View>();
            nv->name = name;
            nv->kind = s.kind;
            v = nv.get();
            S->views.push_back(std::move(nv));
        }
        ++v->dropped;
        core->last_error.clear();
        return 1;
    }

    // A new view, or a kind change between non-texture kinds, is built in a
    // fresh record and swapped in only on success (a failed first apply leaves
    // no view; a failed update keeps the previous one). heatmap <-> image stays
    // in place: the v1.2 view handles that dtype change itself.
    std::unique_ptr<View> fresh;
    View* target = v;
    if (!v || (v->kind != s.kind && !(texture_kind(v->kind) && texture_kind(s.kind)))) {
        fresh = std::make_unique<View>();
        fresh->name = name;
        fresh->kind = s.kind;
        if (v) {   // keep the window where the user put it
            fresh->placed = v->placed;
            fresh->pos_x = v->pos_x; fresh->pos_y = v->pos_y;
            fresh->size_x = v->size_x; fresh->size_y = v->size_y;
        }
        target = fresh.get();
    } else if (v->kind != s.kind) {
        v->kind = s.kind;   // heatmap <-> image
    }

    bool ok = false;
    switch (s.kind) {
        case CALIPER_LIVE_KIND_HEATMAP:
        case CALIPER_LIVE_KIND_IMAGE:     ok = apply_texture(core, *target, s, &why); break;
        case CALIPER_LIVE_KIND_POINTS:    ok = apply_points(core, *target, s, &why); break;
        case CALIPER_LIVE_KIND_SERIES:    ok = apply_series(*target, s, &why); break;
        case CALIPER_LIVE_KIND_HISTOGRAM: ok = apply_histogram(*target, s, &why); break;
    }
    if (!ok) {
        if (fresh) {
            // Don't let the fresh record's release remove a v1.2 view the old
            // record still owns (texture apply failed before adopting it).
            if (v && v->owns_v12) fresh->owns_v12 = false;
            release_typed(core, *fresh);
        }
        return lfail(why);
    }

    if (fresh) {
        if (v) {
            // Old kind out: when the new kind is a texture kind it now owns the
            // v1.2 view of this name, so the old record must not remove it.
            if (texture_kind(s.kind)) v->owns_v12 = false;
            release_typed(core, *v);
            for (auto& p : S->views)
                if (p.get() == v) { p = std::move(fresh); break; }
        } else {
            S->views.push_back(std::move(fresh));
        }
    }
    View& t = *find(core, name.c_str());
    if (!texture_kind(t.kind) && t.creates == 0 && t.kind != CALIPER_LIVE_KIND_POINTS) t.creates = 1;
    ++t.updates;
    t.seq = s.seq;
    t.stamp_ns = s.stamp_ns;
    const uint64_t now = now_ns();
    t.latency_ms = (s.stamp_ns && now > s.stamp_ns) ? (double)(now - s.stamp_ns) * 1e-6 : 0.0;
    t.last_apply_ms = ms_since(t0);
    core->last_error.clear();
    return 1;
}

extern "C" int caliper_core_live_view_remove(CaliperCore* core, const char* name) {
    if (!core) return 0;
    View* v = find(core, name);
    if (!v) {
        core->fail(std::string("live_view_remove: no live view named '") + (name ? name : "") + "'");
        return 0;
    }
    erase_view(core, core->live_views, v);
    return 1;
}

extern "C" int caliper_core_live_view_count(CaliperCore* core) {
    return (core && core->live_views) ? (int)core->live_views->views.size() : 0;
}

extern "C" int caliper_core_live_view_info(CaliperCore* core, const char* name,
                                           CaliperLiveViewInfo* out) {
    if (!core) return 0;
    if (!name || !out || out->struct_size < sizeof(CaliperLiveViewInfo)) {
        core->fail("live_view_info: null name or null/short CaliperLiveViewInfo");
        return 0;
    }
    View* v = find(core, name);
    if (!v) {
        core->fail(std::string("live_view_info: no live view named '") + name + "'");
        return 0;
    }
    poll_handles(core, *v);
    const size_t sz = out->struct_size;
    std::memset(out, 0, sizeof(CaliperLiveViewInfo));
    out->struct_size = sz;
    out->kind = v->kind;
    out->path = v->path.c_str();
    out->updates = v->updates;
    out->creates = v->creates;
    out->dropped = v->dropped;
    out->seq = v->seq;
    out->stamp_ns = v->stamp_ns;
    out->last_apply_ms = v->last_apply_ms;
    out->latency_ms = v->latency_ms;
    out->imported = v->imported ? 1 : 0;
    out->read_done = (v->pending.empty() && v->retired.empty()) ? 1 : 0;
    out->count = v->count;
    if (texture_kind(v->kind)) {
        if (const TensorView* tv = find_v12(core, v->name)) {
            out->width = tv->w;
            out->height = tv->h;
            out->creates = tv->creates;
        }
    } else if (v->kind == CALIPER_LIVE_KIND_POINTS) {
        out->width = v->tex_w;
        out->height = v->tex_h;
    }
    return 1;
}

extern "C" int caliper_core_live_view_read_done(CaliperCore* core, const char* name) {
    if (!core || !name) return 1;
    View* v = find(core, name);
    if (!v) return 1;
    poll_handles(core, *v);
    return (v->pending.empty() && v->retired.empty()) ? 1 : 0;
}

extern "C" int caliper_core_live_view_readback(CaliperCore* core, const char* name, void* buf,
                                               size_t buf_bytes) {
    if (!core) return 0;
    if (!name || !buf) {
        core->fail("live_view_readback: null name/buffer");
        return 0;
    }
    View* v = find(core, name);
    if (!v) {
        core->fail(std::string("live_view_readback: no live view named '") + name + "'");
        return 0;
    }
    if (texture_kind(v->kind)) {
        if (!find_v12(core, v->name)) {
            core->fail(std::string("live_view_readback('") + name + "'): no pixels yet (nothing applied while unpaused)");
            return 0;
        }
        return caliper_core_view_readback(core, name, buf, buf_bytes);
    }
    if (v->kind != CALIPER_LIVE_KIND_POINTS) {
        core->fail(std::string("live_view_readback('") + name + "'): a " + kind_name(v->kind) +
                   " view has no texture; use caliper_core_live_view_values");
        return 0;
    }
    if (!v->view_rid) {
        core->fail(std::string("live_view_readback('") + name +
                   "'): no render target yet (run a frame after the first apply)");
        return 0;
    }
    const size_t need = (size_t)v->tex_w * (size_t)v->tex_h * 4;
    if (buf_bytes < need) {
        core->fail("live_view_readback: buffer too small (need " + std::to_string(need) + " bytes)");
        return 0;
    }
    std::vector<uint8_t> px = core->renderer->debug_readback_rgba8(v->view_rid, v->tex_w, v->tex_h);
    if (px.size() != need) {
        core->fail("live_view_readback: renderer readback failed");
        return 0;
    }
    std::memcpy(buf, px.data(), need);
    return 1;
}

extern "C" int caliper_core_live_view_values(CaliperCore* core, const char* name, float* out,
                                             size_t cap, size_t* count) {
    if (!core) return 0;
    if (!name || (!out && cap > 0)) {
        core->fail("live_view_values: null name, or null out with cap > 0");
        return 0;
    }
    View* v = find(core, name);
    if (!v) {
        core->fail(std::string("live_view_values: no live view named '") + name + "'");
        return 0;
    }
    if (v->kind != CALIPER_LIVE_KIND_SERIES && v->kind != CALIPER_LIVE_KIND_HISTOGRAM) {
        core->fail(std::string("live_view_values('") + name + "'): a " + kind_name(v->kind) +
                   " view holds no values (series and histogram only)");
        return 0;
    }
    const size_t n = v->values.size();
    if (out && cap) std::memcpy(out, v->values.data(), std::min(cap, n) * sizeof(float));
    if (count) *count = n;
    return 1;
}

extern "C" void caliper_core_live_set_paused(CaliperCore* core, int paused) {
    if (core) state(core)->paused = paused != 0;
}

extern "C" int caliper_core_live_paused(CaliperCore* core) {
    return (core && core->live_views && core->live_views->paused) ? 1 : 0;
}

// ===========================================================================
// embed_core.cpp hooks
// ===========================================================================
void caliper_live_views_frame(CaliperCore* core) {
    if (!core || !core->live_views) return;
    CaliperLiveViewsState* S = core->live_views;
    for (auto& v : S->views) {
        poll_handles(core, *v);
        draw_view(core, S, *v);
    }
}

// Called by caliper_core_shutdown BEFORE the v1.2 views are destroyed and
// while the renderer is alive: every typed resource goes back through it.
void caliper_live_views_shutdown(CaliperCore* core) {
    if (!core || !core->live_views) return;
    for (auto& v : core->live_views->views) release_typed(core, *v);
    delete core->live_views;
    core->live_views = nullptr;
}
