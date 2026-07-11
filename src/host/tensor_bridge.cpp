#include "tensor_bridge.h"
#include "renderer/host_renderer.h"
#include <caliper/services/log_v1.h>

#include <cmath>
#include <cstdio>
#include <cstring>
#include <limits>
#include <utility>

namespace caliper_host {

// ---------------------------------------------------------------------------
// Colormap LUTs — generated once, shared by the CPU staging path AND the gfx
// tests' expected values (single source of truth for §16 pixel-exactness).
// ---------------------------------------------------------------------------
namespace {

inline uint32_t pack_rgba(float r, float g, float b) {
    auto q = [](float x) -> uint32_t {
        if (x < 0.0f) x = 0.0f;
        if (x > 1.0f) x = 1.0f;
        return (uint32_t)(x * 255.0f + 0.5f);
    };
    return q(r) | (q(g) << 8) | (q(b) << 16) | (0xffu << 24);
}

// viridis / magma: matplotlib-derived degree-6 polynomial fits (Matt Zucker,
// https://www.shadertoy.com/view/WlfXRN) — accurate to <2/255 vs matplotlib.
// c[k] holds the RGB coefficient of t^k for k = 0..6.
void gen_poly(uint32_t* lut, const float c[7][3]) {
    for (int i = 0; i < 256; ++i) {
        float t = i / 255.0f;
        float r = c[0][0], g = c[0][1], b = c[0][2], tp = 1.0f;
        for (int k = 1; k < 7; ++k) {
            tp *= t;
            r += c[k][0] * tp;
            g += c[k][1] * tp;
            b += c[k][2] * tp;
        }
        lut[i] = pack_rgba(r, g, b);
    }
}

// RdBu: ColorBrewer 11-class diverging anchors (the same control points
// matplotlib's RdBu is built from), piecewise-linear over t in [0,1].
void gen_rdbu(uint32_t* lut) {
    static const float a[11][3] = {
        {103,   0,  31}, {178,  24,  43}, {214,  96,  77}, {244, 165, 130},
        {253, 219, 199}, {247, 247, 247}, {209, 229, 240}, {146, 197, 222},
        { 67, 147, 195}, { 33, 102, 172}, {  5,  48,  97},
    };
    for (int i = 0; i < 256; ++i) {
        float p = (i / 255.0f) * 10.0f;    // 0..10 spans the 11 anchors
        int lo = (int)p; if (lo > 9) lo = 9;
        float f = p - (float)lo;
        float r = a[lo][0] + (a[lo + 1][0] - a[lo][0]) * f;
        float g = a[lo][1] + (a[lo + 1][1] - a[lo][1]) * f;
        float b = a[lo][2] + (a[lo + 1][2] - a[lo][2]) * f;
        lut[i] = pack_rgba(r / 255.0f, g / 255.0f, b / 255.0f);
    }
}

struct Luts {
    uint32_t viridis[256];
    uint32_t magma[256];
    uint32_t rdbu[256];
    Luts() {
        static const float kViridis[7][3] = {
            { 0.2777273272234177f,   0.005407344544966578f,  0.3340998053353061f  },
            { 0.1050930431085774f,   1.404613529898575f,     1.384590162594685f   },
            {-0.3308618287255563f,   0.214847559468213f,     0.09509516302823659f },
            {-4.634230498983486f,   -5.799100973351585f,   -19.33244095627987f    },
            { 6.228269936347081f,   14.17993336680509f,     56.69055260068105f    },
            { 4.776384997670288f,  -13.74514537774601f,    -65.35303263337234f    },
            {-5.435455855934631f,    4.645852612178535f,     26.3124352495832f    },
        };
        static const float kMagma[7][3] = {
            {-0.002136485053939582f, -0.000749655052795221f, -0.005386127855323933f},
            { 0.2516605407371642f,    0.6775232436837668f,    2.494026599312351f   },
            { 8.353717279216625f,    -3.577719514958484f,     0.3144679030132573f  },
            {-27.66873308576866f,    14.26473078096533f,    -13.64921318813922f    },
            { 52.17613981234068f,   -27.94360607168351f,    12.94416944238394f    },
            {-50.76852536473588f,    29.04658282127291f,     4.23415299384598f    },
            { 18.65570506591883f,   -11.48977351997711f,    -5.601961508734096f   },
        };
        gen_poly(viridis, kViridis);
        gen_poly(magma, kMagma);
        gen_rdbu(rdbu);
    }
};

const Luts& luts() { static const Luts L; return L; }

// Bridge rejections route through caliper.log.v1 when the host installs a sink
// (host_services::set_bridge_log_sink); the unit/gfx test binaries link this TU
// without host_services, so the sink stays null and rejections still surface on
// stderr — a rejection is never a silent misinterpreted texture (§ rules).
void (*g_log_sink)(CaliperLogLevel, const char*) = nullptr;

void bridge_log(const char* what) {
    if (g_log_sink)
        g_log_sink(CALIPER_LOG_WARN, what);
    else
        std::fprintf(stderr, "[tensor_bridge] reject: %s\n", what);
}

bool is_contiguous(const CaliperTensor& t) {
    int64_t expected = 1;
    for (int i = t.ndim - 1; i >= 0; --i) {
        if (t.strides[i] != expected) return false;
        expected *= t.shape[i];
    }
    return true;
}

int dtype_size(CaliperDType d) {
    switch (d) {
        case CALIPER_DT_F32:
        case CALIPER_DT_I32:  return 4;
        case CALIPER_DT_F16:
        case CALIPER_DT_BF16: return 2;
        case CALIPER_DT_I64:  return 8;
        case CALIPER_DT_U8:   return 1;
    }
    return 0;
}

// Finding #1 (C2 review): compute the tensor's extent — max linear element
// index + 1 from shape×strides — and sanity-bound it BEFORE any device path,
// so the bridge never forwards a tensor addressing memory it hasn't reasoned
// about. Returns false on non-positive dims, negative strides, or overflow.
constexpr int64_t kMaxExtentElems = int64_t(1) << 40;   // ~1e12 elements
bool safe_extent_elems(const CaliperTensor& t, int64_t* out) {
    int64_t maxidx = 0;
    for (int i = 0; i < t.ndim; ++i) {
        if (t.shape[i] <= 0 || t.strides[i] < 0) return false;
        maxidx += (t.shape[i] - 1) * t.strides[i];
        if (maxidx < 0 || maxidx >= kMaxExtentElems) return false;  // overflow/sane
    }
    *out = maxidx + 1;
    return true;
}

}  // namespace

// Install the host's caliper.log.v1 route for bridge rejections. Declared (not
// in the frozen tensor_bridge.h) by host_services.cpp, which owns log_impl.
void set_bridge_log_sink(void (*sink)(CaliperLogLevel, const char*)) {
    g_log_sink = sink;
}

const uint32_t* colormap_lut(int32_t cm) {
    switch (cm) {
        case CALIPER_CMAP_VIRIDIS: return luts().viridis;
        case CALIPER_CMAP_MAGMA:   return luts().magma;
        case CALIPER_CMAP_RDBU:    return luts().rdbu;
        default:                   return nullptr;
    }
}

void map_f32_to_rgba8(const float* src, int w, int h, const uint32_t* lut,
                      float vmin, float vmax, uint8_t* dst) {
    const float denom = vmax - vmin;
    const int n = w * h;
    for (int i = 0; i < n; ++i) {
        const float v = src[i];
        int idx;
        if (std::isnan(v)) {
            idx = 0;   // NaN -> index 0, never a misinterpreted texel
        } else {
            float t = (denom != 0.0f) ? (v - vmin) / denom : 0.0f;
            if (t < 0.0f) t = 0.0f;
            else if (t > 1.0f) t = 1.0f;
            idx = (int)(t * 255.0f + 0.5f);
            if (idx < 0) idx = 0;
            else if (idx > 255) idx = 255;
        }
        const uint32_t p = lut[idx];
        dst[i * 4 + 0] = (uint8_t)(p & 0xff);
        dst[i * 4 + 1] = (uint8_t)((p >> 8) & 0xff);
        dst[i * 4 + 2] = (uint8_t)((p >> 16) & 0xff);
        dst[i * 4 + 3] = (uint8_t)((p >> 24) & 0xff);
    }
}

void expand_u8_to_rgba8(const uint8_t* src, int w, int h, int c, uint8_t* dst) {
    const int n = w * h;
    for (int i = 0; i < n; ++i) {
        const uint8_t* s = src + (size_t)i * c;
        uint8_t* d = dst + (size_t)i * 4;
        if (c == 1) {
            d[0] = d[1] = d[2] = s[0]; d[3] = 255;
        } else if (c == 3) {
            d[0] = s[0]; d[1] = s[1]; d[2] = s[2]; d[3] = 255;
        } else {  // c == 4
            d[0] = s[0]; d[1] = s[1]; d[2] = s[2]; d[3] = s[3];
        }
    }
}

// ---------------------------------------------------------------------------
// TensorBridge
// ---------------------------------------------------------------------------

TensorBridge::TensorBridge(HostRenderer& renderer) : renderer_(renderer) {
    // The backend declares the device its zero-copy path imports (spec §3.4).
    // Vulkan reports CUDA only when a UUID-matched CUDA device is actually
    // paired, so on a hybrid/CPU-Vulkan box the bridge advertises CPU and
    // never accepts a CUDA tensor it would only end up staging.
    active_device_ = renderer.interop_device();
}

uint32_t TensorBridge::caps() const {
    uint32_t c = renderer_.honors_stream_ordered_handoff()
        ? CALIPER_BRIDGE_CAP_STREAM_ORDERED : 0u;
    if (renderer_.supports_external_import())
        c |= CALIPER_BRIDGE_CAP_IMPORT_ALLOC;
    return c;
}

namespace {
// The frozen shape gates minus the null-data check: device is CPU or the active
// backend, known dtype, row-major contiguous, a bounded extent. Fills *extent
// with the element count. Shared by accept_common (CPU/device tensors, where
// data must be present) AND the imported-alloc path (where desc->data is
// ignored — the allocation + offset are the address), so the contiguity/dtype/
// extent gate logic lives in exactly one place.
bool accept_shape(const CaliperTensor& t, CaliperDeviceKind active,
                  int64_t* extent) {
    if (t.device != CALIPER_DEV_CPU && t.device != active)
                                         { bridge_log("foreign device");   return false; }
    if (dtype_size(t.dtype) == 0)        { bridge_log("unknown dtype");    return false; }
    if (!is_contiguous(t))               { bridge_log("non-contiguous");   return false; }
    if (!safe_extent_elems(t, extent))   { bridge_log("bad extent");       return false; }
    return true;
}

// Shared acceptance checks for BOTH entry points: data present, device is CPU
// or the active backend, row-major contiguous, and a bounded extent.
bool accept_common(const CaliperTensor& t, CaliperDeviceKind active) {
    if (t.data == nullptr)               { bridge_log("null data");        return false; }
    int64_t extent = 0;
    return accept_shape(t, active, &extent);
}
}  // namespace

CaliperTextureId TensorBridge::texture_from_tensor(const CaliperTensor* t, uint32_t) {
    if (!t)                                     { bridge_log("null tensor");  return 0; }
    if (t->dtype != CALIPER_DT_U8 || t->ndim != 3) { bridge_log("direct: want 3D u8"); return 0; }
    const int c = (int)t->shape[2];
    if (c < 1 || c > 4)                         { bridge_log("direct: C not in 1..4"); return 0; }
    if (!accept_common(*t, active_device_))     return 0;

    const int h = (int)t->shape[0], w = (int)t->shape[1];
    const uint64_t tex = renderer_.tex_create_rgba8(w, h);
    if (tex == 0)                               { bridge_log("tex_create failed"); return 0; }
    // The PUBLIC CaliperTextureId is the renderer's ImGui-compatible handle
    // (§5.4): the value imgui_impl_{metal,opengl3} bind directly as ImTextureID.
    // The internal renderer id (`tex`) stays in Entry for upload/release.
    const CaliperTextureId id = renderer_.tex_imtexture_id(tex);
    if (id == 0) { renderer_.tex_release(tex); bridge_log("direct: null imtexture id"); return 0; }

    Entry e;
    e.tex = tex; e.w = w; e.h = h; e.dtype = CALIPER_DT_U8;
    e.channels = c; e.mapped = false;
    if (!upload_into(e, t)) { renderer_.tex_release(tex); bridge_log("direct: upload failed"); return 0; }
    entries_[id] = std::move(e);
    return id;
}

CaliperTextureId TensorBridge::texture_from_tensor_mapped(const CaliperTensor* t,
        int32_t colormap, float vmin, float vmax, uint32_t) {
    if (!t)                                        { bridge_log("null tensor"); return 0; }
    if (t->dtype != CALIPER_DT_F32 || t->ndim != 2) { bridge_log("mapped: want 2D f32"); return 0; }
    if (colormap_lut(colormap) == nullptr)         { bridge_log("mapped: bad colormap"); return 0; }
    if (!accept_common(*t, active_device_))        return 0;

    const int h = (int)t->shape[0], w = (int)t->shape[1];
    const uint64_t tex = renderer_.tex_create_rgba8(w, h);
    if (tex == 0)                                  { bridge_log("tex_create failed"); return 0; }
    const CaliperTextureId id = renderer_.tex_imtexture_id(tex);   // ImGui handle (§5.4)
    if (id == 0) { renderer_.tex_release(tex); bridge_log("mapped: null imtexture id"); return 0; }

    Entry e;
    e.tex = tex; e.w = w; e.h = h; e.dtype = CALIPER_DT_F32;
    e.mapped = true; e.colormap = colormap; e.vmin = vmin; e.vmax = vmax;
    if (!upload_into(e, t)) { renderer_.tex_release(tex); bridge_log("mapped: upload failed"); return 0; }
    entries_[id] = std::move(e);
    return id;
}

bool TensorBridge::desc_matches_entry(const Entry& e, const CaliperTensor& t) const {
    if (t.dtype != e.dtype) { bridge_log("update: dtype mismatch"); return false; }
    if (e.mapped) {
        if (t.ndim != 2 || (int)t.shape[0] != e.h || (int)t.shape[1] != e.w) {
            bridge_log("update: shape mismatch"); return false;
        }
    } else {
        if (t.ndim != 3 || (int)t.shape[0] != e.h || (int)t.shape[1] != e.w
            || (int)t.shape[2] != e.channels) {
            bridge_log("update: shape mismatch"); return false;
        }
    }
    return true;
}

bool TensorBridge::update_texture(CaliperTextureId tex, const CaliperTensor* t) {
    auto it = entries_.find(tex);
    if (it == entries_.end() || !t) return false;
    Entry& e = it->second;
    if (e.view) { bridge_log("update: id is a geometry view"); return false; }

    if (!desc_matches_entry(e, *t)) return false;
    if (!accept_common(*t, active_device_)) return false;
    return upload_into(e, t);
}

void TensorBridge::release_texture(CaliperTextureId tex) {
    auto it = entries_.find(tex);
    if (it == entries_.end()) return;
    if (it->second.view) { bridge_log("release: id is a geometry view"); return; }
    renderer_.tex_release(it->second.tex);
    entries_.erase(it);
}

bool TensorBridge::alloc_shared(CaliperDType dtype, int32_t ndim,
                                const int64_t* shape, CaliperTensor* out,
                                CaliperTextureId* out_texture) {
    if (!out || !out_texture || !shape) return false;

    int w, h, channels = 1;
    bool mapped;
    if (dtype == CALIPER_DT_F32 && ndim == 2) {
        mapped = true;  h = (int)shape[0]; w = (int)shape[1];
    } else if (dtype == CALIPER_DT_U8 && ndim == 3) {
        channels = (int)shape[2];
        if (channels < 1 || channels > 4) { bridge_log("alloc_shared: C not in 1..4"); return false; }
        mapped = false; h = (int)shape[0]; w = (int)shape[1];
    } else {
        bridge_log("alloc_shared: unsupported shape/dtype"); return false;
    }
    if (w <= 0 || h <= 0) { bridge_log("alloc_shared: bad dims"); return false; }

    int64_t numel = 1;
    for (int i = 0; i < ndim; ++i) {
        if (shape[i] <= 0) return false;
        numel *= shape[i];
    }
    const size_t bytes = (size_t)numel * (size_t)dtype_size(dtype);

    const uint64_t tex = renderer_.tex_create_rgba8(w, h);
    if (tex == 0) { bridge_log("alloc_shared: tex_create failed"); return false; }
    const CaliperTextureId id = renderer_.tex_imtexture_id(tex);   // ImGui handle (§5.4)
    if (id == 0) { renderer_.tex_release(tex); bridge_log("alloc_shared: null imtexture id"); return false; }

    Entry e;
    e.tex = tex; e.w = w; e.h = h; e.dtype = dtype; e.channels = channels;
    e.mapped = mapped; e.colormap = CALIPER_CMAP_VIRIDIS; e.vmin = 0.0f; e.vmax = 1.0f;
    e.shared = true;

    // Literal zero-copy (§3.5): when the backend imports our device, back the
    // shared tensor with a device buffer the applet's kernels write in place —
    // update_texture then does no copy. Falls back to a CPU-unified vector when
    // the backend has no device-shared texture path (GL or CUDA unpaired).
    void* device_ptr = nullptr;
    const bool device_shared =
        active_device_ != CALIPER_DEV_CPU &&
        renderer_.alloc_device_shared(tex, (uint64_t)bytes, &device_ptr) &&
        device_ptr != nullptr;
    if (!device_shared) e.shared_buf.assign(bytes, 0);
    Entry& stored = (entries_[id] = std::move(e));

    std::memset(out, 0, sizeof(*out));
    out->struct_size = sizeof(CaliperTensor);
    out->data = device_shared ? device_ptr : stored.shared_buf.data();
    out->dtype = dtype;
    out->ndim = ndim;
    int64_t st = 1;
    for (int i = ndim - 1; i >= 0; --i) {
        out->shape[i] = shape[i];
        out->strides[i] = st;
        st *= shape[i];
    }
    out->device = device_shared ? active_device_ : CALIPER_DEV_CPU;
    out->device_index = 0;

    *out_texture = id;
    return true;
}

void TensorBridge::free_shared(CaliperTextureId tex) { release_texture(tex); }

// ---------------------------------------------------------------------------
// caliper.tensor_bridge.v1.2 — imported external allocations
// ---------------------------------------------------------------------------

CaliperAllocId TensorBridge::import_allocation(void* os_handle,
                                               uint64_t size_bytes,
                                               uint32_t handle_type) {
    if (os_handle == nullptr)        { bridge_log("import: null handle");    return 0; }
    if (size_bytes == 0)             { bridge_log("import: zero size");      return 0; }
    if (handle_type != CALIPER_ALLOC_HANDLE_OPAQUE_WIN32 &&
        handle_type != CALIPER_ALLOC_HANDLE_OPAQUE_FD &&
        handle_type != CALIPER_ALLOC_HANDLE_MTLBUFFER)
                                     { bridge_log("import: bad handle type"); return 0; }
    const uint64_t rid =
        renderer_.import_external_allocation(os_handle, size_bytes, handle_type);
    if (rid == 0) { bridge_log("import: renderer refused"); return 0; }  // no insert

    const CaliperAllocId id = next_alloc_id_++;
    imported_[id] = ImportedAlloc{rid, size_bytes};
    return id;
}

void TensorBridge::release_allocation(CaliperAllocId a) {
    auto it = imported_.find(a);
    if (it == imported_.end()) return;   // invalid id / double release: no-op
    renderer_.release_external_allocation(it->second.renderer_id);
    imported_.erase(it);
}

bool TensorBridge::update_texture_from_alloc(CaliperTextureId tex, CaliperAllocId a,
                                             uint64_t offset_bytes,
                                             const CaliperTensor* desc) {
    if (!desc) { bridge_log("update_alloc: null desc"); return false; }
    auto te = entries_.find(tex);
    if (te == entries_.end()) { bridge_log("update_alloc: unknown texture"); return false; }
    auto ia = imported_.find(a);
    if (ia == imported_.end()) { bridge_log("update_alloc: unknown alloc"); return false; }
    Entry& e = te->second;
    if (e.view) { bridge_log("update_alloc: id is a geometry view"); return false; }
    const ImportedAlloc& alloc = ia->second;

    // Same frozen acceptance gates as update_texture — dtype/shape vs the entry,
    // then contiguity/dtype/extent — but desc->data is IGNORED (the imported
    // allocation + offset are the address), so the null-data check is skipped.
    if (!desc_matches_entry(e, *desc)) return false;
    int64_t extent = 0;
    if (!accept_shape(*desc, active_device_, &extent)) return false;

    // Host-side bounds (the analog of cuMemGetAddressRange): the byte window the
    // desc addresses must lie inside the imported allocation. The renderer
    // re-checks against the real device allocation before touching it.
    const uint64_t bytes = (uint64_t)extent * (uint64_t)dtype_size(desc->dtype);
    if (offset_bytes > alloc.size_bytes || bytes > alloc.size_bytes - offset_bytes) {
        bridge_log("update_alloc: window out of imported bounds"); return false;
    }

    // Forward to the device update-from-imported path with the texture's stored
    // (pinned-at-create) colormap/vmin/vmax — same values update_texture uses.
    return renderer_.tex_update_from_imported(e.tex, alloc.renderer_id,
                                              offset_bytes, *desc,
                                              e.colormap, e.vmin, e.vmax);
}

// --- caliper.geometry.v1: imported 3-D points into offscreen views ---------

uint32_t TensorBridge::geom_caps() const {
    const bool primitives = renderer_.supports_geometry_primitives();
    uint32_t c = (renderer_.supports_geometry() || primitives)
        ? CALIPER_GEOM_CAP_IMPORTED_POINTS : 0u;
    if (primitives)
        c |= CALIPER_GEOM_CAP_PRIMITIVES;
    if (primitives && renderer_.supports_geometry_textured())
        c |= CALIPER_GEOM_CAP_TEXTURED;
    if (primitives && renderer_.supports_geometry_instanced())
        c |= CALIPER_GEOM_CAP_INSTANCED;
    return c;
}

CaliperTextureId TensorBridge::geom_create_view(uint32_t w, uint32_t h) {
    if (w == 0 || h == 0 || w > 16384 || h > 16384) {
        bridge_log("geom_view: bad size"); return 0;
    }
    const uint64_t rid = renderer_.geom_create_view((int)w, (int)h);
    if (rid == 0) { bridge_log("geom_view: renderer refused"); return 0; }
    const CaliperTextureId pub = renderer_.tex_imtexture_id(rid);
    Entry e;
    e.tex  = rid;
    e.w    = (int)w;
    e.h    = (int)h;
    e.view = true;
    e.view_depth = false;
    entries_[pub] = std::move(e);
    return pub;
}

CaliperTextureId TensorBridge::geom_create_view_ex(uint32_t w, uint32_t h,
                                                   uint32_t flags) {
    if (w == 0 || h == 0 || w > 16384 || h > 16384) {
        bridge_log("geom_view_ex: bad size"); return 0;
    }
    if ((flags & ~CALIPER_GEOM_VIEW_DEPTH) != 0u) {
        bridge_log("geom_view_ex: bad flags"); return 0;
    }
    if (!renderer_.supports_geometry_primitives()) {
        bridge_log("geom_view_ex: primitives unsupported"); return 0;
    }
    const uint64_t rid = renderer_.geom_create_view_ex((int)w, (int)h, flags);
    if (rid == 0) { bridge_log("geom_view_ex: renderer refused"); return 0; }
    const CaliperTextureId pub = renderer_.tex_imtexture_id(rid);
    Entry e;
    e.tex  = rid;
    e.w    = (int)w;
    e.h    = (int)h;
    e.view = true;
    e.view_depth = (flags & CALIPER_GEOM_VIEW_DEPTH) != 0u;
    entries_[pub] = std::move(e);
    return pub;
}

void TensorBridge::geom_release_view(CaliperTextureId view) {
    auto it = entries_.find(view);
    if (it == entries_.end() || !it->second.view) return;   // wrong door: no-op
    renderer_.tex_release(it->second.tex);
    entries_.erase(it);
}

bool TensorBridge::geom_draw_points(CaliperTextureId view,
                                    const CaliperGeomCamera* cam,
                                    CaliperAllocId pos_alloc,
                                    uint64_t pos_offset, uint64_t count,
                                    CaliperAllocId attr_alloc,
                                    uint64_t attr_offset,
                                    int32_t colormap, float vmin, float vmax,
                                    float size_px, uint32_t clear_rgba) {
    if (!cam) { bridge_log("geom_draw: null camera"); return false; }
    auto vt = entries_.find(view);
    if (vt == entries_.end() || !vt->second.view) {
        bridge_log("geom_draw: unknown view"); return false;
    }
    if (!(size_px > 0.0f)) { bridge_log("geom_draw: bad point size"); return false; }

    uint64_t pos_rid = 0, attr_rid = 0;
    const uint32_t* lut = nullptr;
    if (count > 0) {
        // Positions: (count,3) f32, 4-byte-aligned offset, overflow-safe
        // bounds against the imported allocation (the renderer re-checks).
        auto pa = imported_.find(pos_alloc);
        if (pa == imported_.end()) { bridge_log("geom_draw: unknown pos alloc"); return false; }
        if (pos_offset % 4 != 0) { bridge_log("geom_draw: pos offset misaligned"); return false; }
        if (count > UINT64_MAX / 12u) { bridge_log("geom_draw: count overflow"); return false; }
        const uint64_t pos_bytes = count * 12u;
        if (pos_offset > pa->second.size_bytes ||
            pos_bytes > pa->second.size_bytes - pos_offset) {
            bridge_log("geom_draw: positions out of imported bounds"); return false;
        }
        pos_rid = pa->second.renderer_id;

        if (attr_alloc != 0) {
            auto aa = imported_.find(attr_alloc);
            if (aa == imported_.end()) { bridge_log("geom_draw: unknown attr alloc"); return false; }
            if (attr_offset % 4 != 0) { bridge_log("geom_draw: attr offset misaligned"); return false; }
            const uint64_t attr_bytes = count * 4u;   // bounded by the *12 check
            if (attr_offset > aa->second.size_bytes ||
                attr_bytes > aa->second.size_bytes - attr_offset) {
                bridge_log("geom_draw: attr out of imported bounds"); return false;
            }
            lut = colormap_lut(colormap);
            if (!lut) { bridge_log("geom_draw: bad colormap"); return false; }
            attr_rid = aa->second.renderer_id;
        }
    }
    return renderer_.geom_draw_points(vt->second.tex, cam->view, cam->proj,
                                      pos_rid, pos_offset, count,
                                      attr_rid, attr_offset,
                                      lut, vmin, vmax, size_px, clear_rgba);
}

bool TensorBridge::geom_draw_primitives(CaliperTextureId view,
                                        const CaliperGeomCamera* cam,
                                        const CaliperGeomDraw* draws,
                                        uint32_t draw_count,
                                        uint32_t draw_stride,
                                        uint32_t clear_rgba) {
    return geom_draw_primitives_impl(view, cam, draws, draw_count, draw_stride,
                                     GeomRev::V1_1, clear_rgba);
}

bool TensorBridge::geom_draw_primitives_v1_2(
        CaliperTextureId view, const CaliperGeomCamera* cam,
        const CaliperGeomDrawV1_2* draws, uint32_t draw_count,
        uint32_t draw_stride, uint32_t clear_rgba) {
    return geom_draw_primitives_impl(view, cam, draws, draw_count, draw_stride,
                                     GeomRev::V1_2, clear_rgba);
}

bool TensorBridge::geom_draw_primitives_v1_3(
        CaliperTextureId view, const CaliperGeomCamera* cam,
        const CaliperGeomDrawV1_3* draws, uint32_t draw_count,
        uint32_t draw_stride, uint32_t clear_rgba) {
    return geom_draw_primitives_impl(view, cam, draws, draw_count, draw_stride,
                                     GeomRev::V1_3, clear_rgba);
}

bool TensorBridge::geom_draw_primitives_impl(
        CaliperTextureId view, const CaliperGeomCamera* cam,
        const void* draws, uint32_t draw_count, uint32_t draw_stride,
        GeomRev rev, uint32_t clear_rgba) {
    // The single revision axis derives everything: v1.2+ records carry the
    // UV/texture tail and may request COLOR_TEXTURE; v1.3 additionally carries
    // the instance tail (read only under V1_3). v1.3 adds no color mode.
    const uint32_t min_stride =
        rev == GeomRev::V1_3 ? sizeof(CaliperGeomDrawV1_3)
      : rev == GeomRev::V1_2 ? sizeof(CaliperGeomDrawV1_2)
                             : sizeof(CaliperGeomDraw);
    const uint32_t max_color =
        rev == GeomRev::V1_1 ? CALIPER_GEOM_COLOR_VERTEX_RGBA
                             : CALIPER_GEOM_COLOR_TEXTURE;
    if (!renderer_.supports_geometry_primitives()) {
        bridge_log("geom_prims: primitives unsupported"); return false;
    }
    if (!cam) { bridge_log("geom_prims: null camera"); return false; }
    if (draw_stride < min_stride) {
        bridge_log("geom_prims: short stride"); return false;
    }
    if (draw_count > 0 && draws == nullptr) {
        bridge_log("geom_prims: null draws"); return false;
    }
    if (draw_count > 0 &&
        (size_t)draw_count > std::numeric_limits<size_t>::max() / draw_stride) {
        bridge_log("geom_prims: draw array overflow"); return false;
    }
    auto vt = entries_.find(view);
    if (vt == entries_.end() || !vt->second.view) {
        bridge_log("geom_prims: unknown view"); return false;
    }

    auto range_ok = [](uint64_t size, uint64_t offset, uint64_t elem_count,
                       uint64_t elem_size, const char* what) -> bool {
        if (offset % 4u != 0u) {
            char msg[96];
            std::snprintf(msg, sizeof msg, "geom_prims: %s offset misaligned", what);
            bridge_log(msg);
            return false;
        }
        if (elem_size != 0 && elem_count > UINT64_MAX / elem_size) {
            char msg[96];
            std::snprintf(msg, sizeof msg, "geom_prims: %s byte count overflow", what);
            bridge_log(msg);
            return false;
        }
        const uint64_t bytes = elem_count * elem_size;
        if (offset > size || bytes > size - offset) {
            char msg[96];
            std::snprintf(msg, sizeof msg, "geom_prims: %s out of imported bounds", what);
            bridge_log(msg);
            return false;
        }
        return true;
    };

    std::vector<HostGeomDraw> resolved;
    resolved.reserve(draw_count);
    for (uint32_t i = 0; i < draw_count; ++i) {
        const auto* record = reinterpret_cast<const uint8_t*>(draws) +
                             (uint64_t)i * draw_stride;
        const auto* d = reinterpret_cast<const CaliperGeomDraw*>(record);
        const auto* d12 = reinterpret_cast<const CaliperGeomDrawV1_2*>(record);
        const auto* d13 = reinterpret_cast<const CaliperGeomDrawV1_3*>(record);
        auto reject_i = [i](const char* reason) {
            char msg[128];
            std::snprintf(msg, sizeof msg, "geom_prims: draw %u refused: %s", i, reason);
            bridge_log(msg);
        };

        if (d->topology > CALIPER_GEOM_TOPO_TRIANGLE_STRIP) {
            reject_i("bad topology"); return false;
        }
        if (d->color_mode > max_color) {
            reject_i("bad color mode"); return false;
        }
        if (d->color_mode == CALIPER_GEOM_COLOR_TEXTURE &&
            !renderer_.supports_geometry_textured()) {
            reject_i("textured geometry unsupported"); return false;
        }
        if (d->shade_mode > CALIPER_GEOM_SHADE_LAMBERT) {
            reject_i("bad shade mode"); return false;
        }
        if (d->blend_mode > CALIPER_GEOM_BLEND_ADDITIVE) {
            reject_i("bad blend mode"); return false;
        }
        if ((d->depth_flags & ~(CALIPER_GEOM_DEPTH_TEST | CALIPER_GEOM_DEPTH_WRITE)) != 0u) {
            reject_i("bad depth flags"); return false;
        }
        if (d->reserved[0] != 0u || d->reserved[1] != 0u) {
            reject_i("reserved nonzero"); return false;
        }
        if (d->depth_flags != 0u && !vt->second.view_depth) {
            reject_i("depth flags on depthless view"); return false;
        }
        if (d->vertex_count == 0) {
            reject_i("zero vertices"); return false;
        }
        if (d->vertex_count > UINT32_MAX) {
            reject_i("too many vertices"); return false;
        }
        if (d->topology == CALIPER_GEOM_TOPO_POINTS && !(d->size_px > 0.0f)) {
            reject_i("bad point size"); return false;
        }

        auto pa = imported_.find(d->pos_alloc);
        if (pa == imported_.end()) { reject_i("unknown pos alloc"); return false; }
        if (!range_ok(pa->second.size_bytes, d->pos_offset, d->vertex_count,
                      12u, "positions")) {
            return false;
        }

        uint64_t consumed = d->vertex_count;
        uint64_t index_rid = 0;
        if (d->index_alloc != 0) {
            auto ia = imported_.find(d->index_alloc);
            if (ia == imported_.end()) { reject_i("unknown index alloc"); return false; }
            if (d->index_count == 0) { reject_i("zero indices"); return false; }
            if (d->index_count > UINT32_MAX) { reject_i("too many indices"); return false; }
            if (!range_ok(ia->second.size_bytes, d->index_offset, d->index_count,
                          4u, "indices")) {
                return false;
            }
            index_rid = ia->second.renderer_id;
            consumed = d->index_count;
        }
        if ((d->topology == CALIPER_GEOM_TOPO_LINES ||
             d->topology == CALIPER_GEOM_TOPO_LINE_STRIP) && consumed < 2) {
            reject_i("line needs two vertices"); return false;
        }
        if ((d->topology == CALIPER_GEOM_TOPO_TRIANGLES ||
             d->topology == CALIPER_GEOM_TOPO_TRIANGLE_STRIP) && consumed < 3) {
            reject_i("triangle needs three vertices"); return false;
        }

        uint64_t normal_rid = 0;
        if (d->shade_mode == CALIPER_GEOM_SHADE_LAMBERT && d->normal_alloc == 0) {
            reject_i("lambert needs normals"); return false;
        }
        if (d->normal_alloc != 0) {
            auto na = imported_.find(d->normal_alloc);
            if (na == imported_.end()) { reject_i("unknown normal alloc"); return false; }
            if (!range_ok(na->second.size_bytes, d->normal_offset, d->vertex_count,
                          12u, "normals")) {
                return false;
            }
            normal_rid = na->second.renderer_id;
        }

        uint64_t attr_rid = 0;
        const uint32_t* lut = nullptr;
        if (d->color_mode == CALIPER_GEOM_COLOR_COLORMAP ||
            d->color_mode == CALIPER_GEOM_COLOR_VERTEX_RGBA) {
            auto aa = imported_.find(d->attr_alloc);
            if (aa == imported_.end()) { reject_i("unknown attr alloc"); return false; }
            if (!range_ok(aa->second.size_bytes, d->attr_offset, d->vertex_count,
                          4u, "attributes")) {
                return false;
            }
            attr_rid = aa->second.renderer_id;
            if (d->color_mode == CALIPER_GEOM_COLOR_COLORMAP) {
                lut = colormap_lut(d->colormap);
                if (!lut) { reject_i("bad colormap"); return false; }
            }
        }

        // Instance tail (v1.3 only) — the G1-G12 host validator battery (§5).
        // Runs after color_mode/topology otherwise validate; every failure
        // refuses the whole frame atomically before any backend call, pixels
        // untouched. G13 (LAMBERT needs normals) is the existing gate above;
        // G14 (rigidity) executes in each backend re-gate (§5.1 placement), not
        // here. Resolved renderer ids mirror how uv_alloc holds a resolved id.
        uint64_t instance_rid = 0;
        uint64_t instance_attr_rid = 0;
        if (rev == GeomRev::V1_3) {
            const bool has_instances = d13->instance_alloc != 0;
            const bool has_tint = d13->instance_attr_alloc != 0;

            // G1: either instance stream present requires the caps bit live.
            if ((has_instances || has_tint) &&
                !renderer_.supports_geometry_instanced()) {
                reject_i("instanced geometry unsupported"); return false;
            }

            if (has_instances) {
                // G2: N>0 with a pose alloc (mirrors the zero-vertices gate).
                if (d13->instance_count == 0) {
                    reject_i("instanced draw needs N>0"); return false;
                }
                // G3: N bound (Vulkan/Metal instanceCount re-bind to u32).
                if (d13->instance_count > UINT32_MAX) {
                    reject_i("too many instances"); return false;
                }
                // G4: matrix offset 4-byte aligned.
                if (d13->instance_offset % 4u != 0u) {
                    reject_i("instance offset misaligned"); return false;
                }
                // G6: matrix byte base / 4 fits a u32 PrimParams base.
                if (d13->instance_offset / 4u > UINT32_MAX) {
                    reject_i("instance base exceeds 32 bits"); return false;
                }
                // G7: matrix alloc resolves.
                auto ma = imported_.find(d13->instance_alloc);
                if (ma == imported_.end()) {
                    reject_i("unknown instance alloc"); return false;
                }
                // G5: 16 f32 = 64 B/instance, overflow-safe + in imported bounds.
                if (!range_ok(ma->second.size_bytes, d13->instance_offset,
                              d13->instance_count, 64u, "instances")) {
                    return false;
                }
                instance_rid = ma->second.renderer_id;
            }

            if (has_tint) {
                // G8: a tint with nothing to tint is refused, not ignored.
                if (!has_instances || d13->instance_count == 0) {
                    reject_i("instance attr without instances"); return false;
                }
                // G9: attr offset 4-byte aligned.
                if (d13->instance_attr_offset % 4u != 0u) {
                    reject_i("instance attr offset misaligned"); return false;
                }
                // G11: attr alloc resolves.
                auto ta = imported_.find(d13->instance_attr_alloc);
                if (ta == imported_.end()) {
                    reject_i("unknown instance attr alloc"); return false;
                }
                // G10: 4 B/instance scalar, overflow-safe + in imported bounds.
                if (!range_ok(ta->second.size_bytes, d13->instance_attr_offset,
                              d13->instance_count, 4u, "instance attr")) {
                    return false;
                }
                instance_attr_rid = ta->second.renderer_id;
                // G12: the tint needs a resolvable colormap. Tint-LUT rule
                // (§3.3): an instanced-tint draw carries a real LUT regardless
                // of base color_mode, resolved from the base record's colormap.
                if (!lut) lut = colormap_lut(d->colormap);
                if (!lut) { reject_i("instance tint needs colormap"); return false; }
            }
        }

        uint64_t uv_rid = 0;
        uint64_t texture_rid = 0;
        if (d->color_mode == CALIPER_GEOM_COLOR_TEXTURE) {
            auto ua = imported_.find(d12->uv_alloc);
            if (ua == imported_.end()) { reject_i("unknown uv alloc"); return false; }
            if (!range_ok(ua->second.size_bytes, d12->uv_offset,
                          d->vertex_count, 8u, "uvs")) {
                return false;
            }
            auto te = entries_.find(d12->texture);
            if (te == entries_.end()) { reject_i("unknown texture"); return false; }
            if (te->second.view) { reject_i("geometry view used as texture"); return false; }
            uv_rid = ua->second.renderer_id;
            texture_rid = te->second.tex;
        }

        HostGeomDraw hd;
        hd.pos_alloc = pa->second.renderer_id;
        hd.pos_offset = d->pos_offset;
        hd.vertex_count = d->vertex_count;
        hd.index_alloc = index_rid;
        hd.index_offset = index_rid ? d->index_offset : 0u;
        hd.index_count = index_rid ? d->index_count : 0u;
        hd.normal_alloc = normal_rid;
        hd.normal_offset = normal_rid ? d->normal_offset : 0u;
        hd.attr_alloc = attr_rid;
        hd.attr_offset = attr_rid ? d->attr_offset : 0u;
        hd.uv_alloc = uv_rid;
        hd.uv_offset = uv_rid ? d12->uv_offset : 0u;
        hd.texture = texture_rid;
        hd.topology = d->topology;
        hd.color_mode = d->color_mode;
        hd.shade_mode = d->shade_mode;
        hd.blend_mode = d->blend_mode;
        hd.depth_flags = d->depth_flags;
        hd.flat_rgba = d->flat_rgba;
        hd.lut256 = lut;
        hd.vmin = d->vmin;
        hd.vmax = d->vmax;
        hd.size_px = d->size_px;
        std::memcpy(hd.model, d->model, sizeof(hd.model));
        hd.instance_alloc = instance_rid;
        hd.instance_offset = instance_rid ? d13->instance_offset : 0u;
        hd.instance_count = instance_rid ? d13->instance_count : 0u;
        hd.instance_attr_alloc = instance_attr_rid;
        hd.instance_attr_offset =
            instance_attr_rid ? d13->instance_attr_offset : 0u;
        resolved.push_back(hd);
    }

    const HostGeomDraw* resolved_ptr = resolved.empty() ? nullptr : resolved.data();
    return renderer_.geom_draw_primitives(vt->second.tex, cam->view, cam->proj,
                                          resolved_ptr, draw_count, clear_rgba);
}

bool TensorBridge::upload_into(Entry& e, const CaliperTensor* t) {
    if (t->device != CALIPER_DEV_CPU) {
        // Device path — extent already bounded by accept_common (finding #1).
        const uint32_t* lut = e.mapped ? colormap_lut(e.colormap) : nullptr;
        return renderer_.tex_update_from_device(e.tex, *t, lut, e.vmin, e.vmax);
    }
    std::vector<uint8_t> staging((size_t)e.w * (size_t)e.h * 4);
    if (e.mapped)
        map_f32_to_rgba8((const float*)t->data, e.w, e.h,
                         colormap_lut(e.colormap), e.vmin, e.vmax, staging.data());
    else
        expand_u8_to_rgba8((const uint8_t*)t->data, e.w, e.h, e.channels, staging.data());
    return renderer_.tex_upload_rgba8(e.tex, staging.data(), e.w, e.h);
}

}  // namespace caliper_host
