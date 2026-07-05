#include "tensor_bridge.h"
#include "renderer/host_renderer.h"
#include <caliper/services/log_v1.h>

#include <cmath>
#include <cstdio>
#include <cstring>
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

namespace {
// Shared acceptance checks for BOTH entry points: data present, device is CPU
// or the active backend, row-major contiguous, and a bounded extent.
bool accept_common(const CaliperTensor& t, CaliperDeviceKind active) {
    if (t.data == nullptr)               { bridge_log("null data");        return false; }
    if (t.device != CALIPER_DEV_CPU && t.device != active)
                                         { bridge_log("foreign device");   return false; }
    if (dtype_size(t.dtype) == 0)        { bridge_log("unknown dtype");    return false; }
    if (!is_contiguous(t))               { bridge_log("non-contiguous");   return false; }
    int64_t extent = 0;
    if (!safe_extent_elems(t, &extent))  { bridge_log("bad extent");       return false; }
    return true;
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

bool TensorBridge::update_texture(CaliperTextureId tex, const CaliperTensor* t) {
    auto it = entries_.find(tex);
    if (it == entries_.end() || !t) return false;
    Entry& e = it->second;

    if (t->dtype != e.dtype) { bridge_log("update: dtype mismatch"); return false; }
    if (e.mapped) {
        if (t->ndim != 2 || (int)t->shape[0] != e.h || (int)t->shape[1] != e.w) {
            bridge_log("update: shape mismatch"); return false;
        }
    } else {
        if (t->ndim != 3 || (int)t->shape[0] != e.h || (int)t->shape[1] != e.w
            || (int)t->shape[2] != e.channels) {
            bridge_log("update: shape mismatch"); return false;
        }
    }
    if (!accept_common(*t, active_device_)) return false;
    return upload_into(e, t);
}

void TensorBridge::release_texture(CaliperTextureId tex) {
    auto it = entries_.find(tex);
    if (it == entries_.end()) return;
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
    // the backend has no device interop (GL, Metal-today, or CUDA unpaired).
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
