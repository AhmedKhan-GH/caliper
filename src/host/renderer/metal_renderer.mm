// Metal backend for the HostRenderer seam (PLATFORM.md §5.4). Sibling to
// gl_renderer.cpp; make_metal_renderer() (declared in host_renderer.h)
// constructs it and main.cpp selects via CALIPER_RENDERER=metal, falling back
// to the GL backend if init() fails. Apple-only translation unit.
//
// Frame ordering matches the C1 contract: the CLEAR happens in new_frame()
// (the Metal render pass is opened there with loadAction=Clear), the applet
// draws its ImGui frame, then render() flushes RenderDrawData and presents.
// This mirrors the GL backend's top-of-frame clear exactly.
//
// Compiled with ARC (-fobjc-arc, set in CMakeLists): ObjC strong references
// live as members / in an internal id table; the uint64 texture ids handed to
// callers are sequential handles, never raw retained pointers (§5.4).
#include "host_renderer.h"
#include <caliper/services/tensor_bridge_v1_2.h>   // CALIPER_ALLOC_HANDLE_MTLBUFFER
#include <caliper/services/geometry_v1_1.h>
#include <caliper/services/geometry_v1_2.h>
#include <caliper/services/geometry_v1_3.h>   // CALIPER_GEOM_RIGID_TOL (G14)

#import <Metal/Metal.h>
#import <QuartzCore/QuartzCore.h>

#define GLFW_EXPOSE_NATIVE_COCOA
#include <GLFW/glfw3.h>
#include <GLFW/glfw3native.h>

#include <imgui.h>
#include <backends/imgui_impl_glfw.h>
#include <backends/imgui_impl_metal.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <unordered_map>

namespace caliper_host {

// Resolve a colormap id to its 256-entry RGBA8 LUT (tensor_bridge.cpp, present
// in every exe/test link scope that pulls this backend). Unlike
// tex_update_from_device — where the bridge hands the renderer a resolved
// lut256 — the v1.2 imported path receives the raw colormap id (the texture's
// pinned mapping), so the renderer resolves it here.
const uint32_t* colormap_lut(int32_t colormap);

namespace {

// Cross-backend determinism contract (§16): the compute colormap must produce
// the SAME bytes as the CPU reference. The index math below is byte-identical
//   idx = clamp((v - vmin)/(vmax - vmin), 0, 1) * 255 + 0.5  (floor)
// to the staged path. LUT is 256 RGBA8 entries packed little-endian as
// r | g<<8 | b<<16 | a<<24; RGBA8Unorm store rounds byte/255 back to byte.
static const char* kColormapShaderSrc = R"metal(
#include <metal_stdlib>
using namespace metal;

struct CmapParams {
    uint  w;
    uint  h;
    uint  sx;    // element stride along x
    uint  sy;    // element stride along y
    float vmin;
    float vmax;
};

kernel void cmap_f32(device const float*  src  [[buffer(0)]],
                     device const uint*   lut  [[buffer(1)]],
                     constant CmapParams& p    [[buffer(2)]],
                     texture2d<float, access::write> dst [[texture(0)]],
                     uint2 gid [[thread_position_in_grid]])
{
    if (gid.x >= p.w || gid.y >= p.h) return;
    float v = src[gid.y * p.sy + gid.x * p.sx];
    float denom = p.vmax - p.vmin;
    float t = denom != 0.0f ? (v - p.vmin) / denom : 0.0f;
    t = clamp(t, 0.0f, 1.0f);
    uint idx = (uint)(t * 255.0f + 0.5f);
    uint packed = lut[idx];
    float4 c = float4(float(packed        & 0xffu),
                      float((packed >> 8)  & 0xffu),
                      float((packed >> 16) & 0xffu),
                      float((packed >> 24) & 0xffu)) / 255.0f;
    dst.write(c, gid);
}
)metal";

struct CmapParams {
    uint32_t w, h, sx, sy;
    float    vmin, vmax;
};

// Vertex-pulled point pipeline (caliper.geometry.v1) — MSL port of
// shaders/points.vert + points.frag. Element-base addressing (byte offset / 4)
// into whole-bound buffers, same 4-byte alignment gate as Vulkan. Color index
// math byte-identical to points.vert / map_f32_to_rgba8 (NaN->0, degenerate
// range->0). Square points, no discard — deterministic rasterization.
static const char* kPointsShaderSrc = R"metal(
#include <metal_stdlib>
using namespace metal;

struct GeomParams {
    float4x4 mvp;        // proj*view, premultiplied host-side (column-major)
    uint  pos_base;      // element base = byte offset / 4
    uint  attr_base;
    uint  use_attr;
    float vmin;
    float vmax;
    float size_px;
};

struct VOut {
    float4 pos   [[position]];
    float  size  [[point_size]];
    float4 color;
};

vertex VOut points_vs(uint vid [[vertex_id]],
                      device const float* pos  [[buffer(0)]],
                      device const float* attr [[buffer(1)]],
                      device const uint*  lut  [[buffer(2)]],
                      constant GeomParams& p   [[buffer(3)]])
{
    VOut o;
    float3 wp = float3(pos[p.pos_base + 3u * vid + 0u],
                       pos[p.pos_base + 3u * vid + 1u],
                       pos[p.pos_base + 3u * vid + 2u]);
    o.pos  = p.mvp * float4(wp, 1.0f);
    o.size = p.size_px;
    if (p.use_attr != 0u) {
        float v = attr[p.attr_base + vid];
        float t = (v == v && p.vmax > p.vmin)
                ? clamp((v - p.vmin) / (p.vmax - p.vmin), 0.0f, 1.0f) : 0.0f;
        uint packed = lut[(uint)(t * 255.0f + 0.5f)];
        o.color = float4(float(packed         & 0xffu),
                         float((packed >> 8)  & 0xffu),
                         float((packed >> 16) & 0xffu),
                         float((packed >> 24) & 0xffu)) / 255.0f;
    } else {
        o.color = float4(1.0f);
    }
    return o;
}

fragment float4 points_fs(VOut in [[stage_in]]) { return in.color; }
)metal";

struct GeomParams {          // must match the MSL struct byte-for-byte (88 B)
    float    mvp[16];
    uint32_t pos_base, attr_base, use_attr;
    float    vmin, vmax, size_px;
};
static_assert(sizeof(GeomParams) == 88, "MSL constant-buffer layout");

// caliper.geometry.v1_1 primitive pipeline: vertex-pulled from imported
// MTLBuffers. The shader reads all sources as whole buffers plus element bases,
// mirroring the Vulkan plan and the existing point shader. Indices are pulled
// as u32 and clamped against vertex_count to make out-of-range values defined.
static const char* kGeomShaderSrc = R"metal(
#include <metal_stdlib>
using namespace metal;

struct PrimParams {
    float4x4 mvp;
    float4 nmat0;
    float4 nmat1;
    float4 nmat2;
    uint pos_base;
    uint idx_base;
    uint nrm_base;
    uint attr_base;
    uint use_index;
    uint vertex_count;
    uint color_mode;
    uint shade_mode;
    uint flat_rgba;
    float vmin;
    float vmax;
    float size_px;
    uint uv_base;             // 160
    uint use_instance;        // 164 — 0/1 (v1.3 instance tail)
    uint inst_base;           // 168 — instance_offset / 4
    uint use_instance_attr;   // 172 — 0/1
    uint inst_attr_base;      // 176 — instance_attr_offset / 4
    uint pad0;                // 180
    uint pad1;                // 184
    uint pad2;                // 188  (192)
};

/* Metal validates [[point_size]] against the pipeline's topology class: a
 * vertex function that writes it is rejected at pipeline creation for Line
 * and Triangle classes. So the point class gets its own entry point and the
 * two variants share one body. */
struct VOut {
    float4 pos [[position]];
    float4 color;
    float2 uv;
};

struct VOutPoint {
    float4 pos [[position]];
    float  size [[point_size]];
    float4 color;
    float2 uv;
};

static inline float4 unpack_rgba(uint packed) {
    return float4(float(packed         & 0xffu),
                  float((packed >> 8)  & 0xffu),
                  float((packed >> 16) & 0xffu),
                  float((packed >> 24) & 0xffu)) / 255.0f;
}

static inline VOut geom_compute(uint vid, uint iid,
                                device const float* pos,
                                device const uint*  idx,
                                device const float* nrm,
                                device const uint*  attr,
                                device const float* uv,
                                device const float* im,
                                device const uint*  iattr,
                                constant uint*      lut,
                                constant PrimParams& p)
{
    VOut o;
    uint vi = p.use_index != 0u ? min(idx[p.idx_base + vid], p.vertex_count - 1u)
                                : vid;
    float3 wp = float3(pos[p.pos_base + 3u * vi + 0u],
                       pos[p.pos_base + 3u * vi + 1u],
                       pos[p.pos_base + 3u * vi + 2u]);

    // v1.3 (§4.1): the per-instance model matrix is applied to the world
    // position FIRST, then mvp. use_instance==0 takes the exact v1.2 expression
    // (bit-identical). M columns are pulled column-major at inst_base + 16*iid.
    bool inst = (p.use_instance != 0u);
    float4x4 M;
    if (inst) {
        uint b = p.inst_base + 16u * iid;
        M = float4x4(float4(im[b + 0u], im[b + 1u], im[b + 2u], im[b + 3u]),
                     float4(im[b + 4u], im[b + 5u], im[b + 6u], im[b + 7u]),
                     float4(im[b + 8u], im[b + 9u], im[b + 10u], im[b + 11u]),
                     float4(im[b + 12u], im[b + 13u], im[b + 14u], im[b + 15u]));
        o.pos = p.mvp * (M * float4(wp, 1.0f));
    } else {
        o.pos = p.mvp * float4(wp, 1.0f);
    }
    o.uv = p.color_mode == 3u
        ? float2(uv[p.uv_base + 2u * vi + 0u],
                 uv[p.uv_base + 2u * vi + 1u])
        : float2(0.0f);

    float4 c;
    if (inst && p.use_instance_attr != 0u) {
        // §4.3: per-instance tint overrides the color_mode source, looked up once
        // per instance through the same COLORMAP idx math (LUT bound at index 4).
        float v = as_type<float>(iattr[p.inst_attr_base + iid]);
        float t = (v == v && p.vmax > p.vmin)
                ? clamp((v - p.vmin) / (p.vmax - p.vmin), 0.0f, 1.0f) : 0.0f;
        c = unpack_rgba(lut[(uint)(t * 255.0f + 0.5f)]);
    } else if (p.color_mode == 1u) {
        float v = as_type<float>(attr[p.attr_base + vi]);
        float t = (v == v && p.vmax > p.vmin)
                ? clamp((v - p.vmin) / (p.vmax - p.vmin), 0.0f, 1.0f) : 0.0f;
        c = unpack_rgba(lut[(uint)(t * 255.0f + 0.5f)]);
    } else if (p.color_mode == 2u) {
        c = unpack_rgba(attr[p.attr_base + vi]);
    } else if (p.color_mode == 3u) {
        c = float4(1.0f);
    } else {
        c = unpack_rgba(p.flat_rgba);
    }

    if (p.shade_mode == 1u) {
        float3 n = normalize(float3(nrm[p.nrm_base + 3u * vi + 0u],
                                   nrm[p.nrm_base + 3u * vi + 1u],
                                   nrm[p.nrm_base + 3u * vi + 2u]));
        float3 nvs;
        if (inst) {
            // §4.4: instance upper-3x3 (M columns 0/1/2, xyz) applied to n first,
            // then the per-draw normal matrix — EXACT float op order from the spec.
            float3 ni;
            ni.x = M[0].x * n.x + M[1].x * n.y + M[2].x * n.z;
            ni.y = M[0].y * n.x + M[1].y * n.y + M[2].y * n.z;
            ni.z = M[0].z * n.x + M[1].z * n.y + M[2].z * n.z;
            nvs = normalize(ni.x * p.nmat0.xyz +
                            ni.y * p.nmat1.xyz +
                            ni.z * p.nmat2.xyz);
        } else {
            nvs = normalize(n.x * p.nmat0.xyz +
                            n.y * p.nmat1.xyz +
                            n.z * p.nmat2.xyz);
        }
        float lit = 0.30f + 0.70f * max(dot(nvs, float3(0.0f, 0.0f, 1.0f)), 0.0f);
        c.rgb *= lit;
    }
    o.color = c;
    return o;
}

vertex VOut geom_vs(uint vid [[vertex_id]],
                    uint iid [[instance_id]],
                    device const float* pos   [[buffer(0)]],
                    device const uint*  idx   [[buffer(1)]],
                    device const float* nrm   [[buffer(2)]],
                    device const uint*  attr  [[buffer(3)]],
                    constant uint*      lut   [[buffer(4)]],
                    constant PrimParams& p    [[buffer(5)]],
                    device const float* uv    [[buffer(6)]],
                    device const float* im    [[buffer(7)]],
                    device const uint*  iattr [[buffer(8)]])
{
    return geom_compute(vid, iid, pos, idx, nrm, attr, uv, im, iattr, lut, p);
}

vertex VOutPoint geom_vs_point(uint vid [[vertex_id]],
                               uint iid [[instance_id]],
                               device const float* pos   [[buffer(0)]],
                               device const uint*  idx   [[buffer(1)]],
                               device const float* nrm   [[buffer(2)]],
                               device const uint*  attr  [[buffer(3)]],
                               constant uint*      lut   [[buffer(4)]],
                               constant PrimParams& p    [[buffer(5)]],
                               device const float* uv    [[buffer(6)]],
                               device const float* im    [[buffer(7)]],
                               device const uint*  iattr [[buffer(8)]])
{
    VOut b = geom_compute(vid, iid, pos, idx, nrm, attr, uv, im, iattr, lut, p);
    VOutPoint o;
    o.pos   = b.pos;
    o.size  = p.size_px;
    o.color = b.color;
    o.uv    = b.uv;
    return o;
}

fragment float4 geom_fs(VOut in [[stage_in]]) { return in.color; }
// geom_tex_fs is also paired with geom_vs_point for textured POINT draws:
// valid MSL — [[position]]/[[point_size]] are builtins excluded from stage-in
// matching, and the user varyings (color, uv) are identical in VOut and
// VOutPoint. Hardware-verified by the gfx row "textured POINT draw renders
// the sampled texel".
fragment float4 geom_tex_fs(VOut in [[stage_in]],
                            texture2d<float> tex [[texture(0)]]) {
    constexpr sampler smp(coord::normalized, address::clamp_to_edge,
                          filter::linear, mip_filter::none);
    float4 sampled = tex.sample(smp, in.uv);
    sampled.rgb *= in.color.rgb;
    return sampled;
}
)metal";

struct PrimParams {
    float    mvp[16];
    float    nmat0[4];
    float    nmat1[4];
    float    nmat2[4];
    uint32_t pos_base, idx_base, nrm_base, attr_base;
    uint32_t use_index, vertex_count, color_mode, shade_mode;
    uint32_t flat_rgba;
    float    vmin, vmax, size_px;
    uint32_t uv_base, use_instance, inst_base, use_instance_attr;   // v1.3 tail
    uint32_t inst_attr_base, pad0, pad1, pad2;
};
static_assert(sizeof(PrimParams) == 192, "MSL primitive params layout");

void mat4_mul_cm(const float* a, const float* b, float* out) {
    for (int c = 0; c < 4; ++c)
        for (int r = 0; r < 4; ++r) {
            float acc = 0.f;
            for (int k = 0; k < 4; ++k)
                acc += a[k * 4 + r] * b[c * 4 + k];
            out[c * 4 + r] = acc;
        }
}

void normal_matrix_columns(const float* view_model, float* c0, float* c1, float* c2) {
    const double a00 = view_model[0], a01 = view_model[4], a02 = view_model[8];
    const double a10 = view_model[1], a11 = view_model[5], a12 = view_model[9];
    const double a20 = view_model[2], a21 = view_model[6], a22 = view_model[10];
    const double det = a00 * (a11 * a22 - a12 * a21)
                     - a01 * (a10 * a22 - a12 * a20)
                     + a02 * (a10 * a21 - a11 * a20);
    if (std::abs(det) < 1e-12) {
        c0[0] = 1.f; c0[1] = 0.f; c0[2] = 0.f; c0[3] = 0.f;
        c1[0] = 0.f; c1[1] = 1.f; c1[2] = 0.f; c1[3] = 0.f;
        c2[0] = 0.f; c2[1] = 0.f; c2[2] = 1.f; c2[3] = 0.f;
        return;
    }
    const double inv_det = 1.0 / det;
    const double inv00 =  (a11 * a22 - a12 * a21) * inv_det;
    const double inv01 = -(a01 * a22 - a02 * a21) * inv_det;
    const double inv02 =  (a01 * a12 - a02 * a11) * inv_det;
    const double inv10 = -(a10 * a22 - a12 * a20) * inv_det;
    const double inv11 =  (a00 * a22 - a02 * a20) * inv_det;
    const double inv12 = -(a00 * a12 - a02 * a10) * inv_det;
    const double inv20 =  (a10 * a21 - a11 * a20) * inv_det;
    const double inv21 = -(a00 * a21 - a01 * a20) * inv_det;
    const double inv22 =  (a00 * a11 - a01 * a10) * inv_det;

    // Columns of transpose(inverse(A)) are rows of inverse(A).
    c0[0] = (float)inv00; c0[1] = (float)inv01; c0[2] = (float)inv02; c0[3] = 0.f;
    c1[0] = (float)inv10; c1[1] = (float)inv11; c1[2] = (float)inv12; c1[3] = 0.f;
    c2[0] = (float)inv20; c2[1] = (float)inv21; c2[2] = (float)inv22; c2[3] = 0.f;
}

uint32_t topo_class(uint32_t topology) {
    if (topology == CALIPER_GEOM_TOPO_POINTS) return 0u;
    if (topology == CALIPER_GEOM_TOPO_LINES ||
        topology == CALIPER_GEOM_TOPO_LINE_STRIP) return 1u;
    return 2u;
}

bool metal_geom_fail(const char* reason) {
    std::fprintf(stderr, "[metal] geom_prims: %s\n", reason);
    return false;
}

// G14 (spec §5.1): an instance upper-3x3 must be orthogonal-up-to-uniform-scale
// for the §4.4 normal chain (raw-upper-3x3 + normalize) to be exact-compose.
// c0,c1,c2 are the columns of the upper-3x3 (column-major 4x4: col j at m[4j..]),
// read as f32 straight from the imported buffer. s̄² = (‖c0‖²+‖c1‖²+‖c2‖²)/3;
// refuse unless every pair is orthogonal and every column equal-length within
// CALIPER_GEOM_RIGID_TOL·s̄², and s̄²>0. The float op order is the byte-exact
// contract shared with Vulkan (T4 transcribes this verbatim).
bool instance_upper3x3_rigid(const float* m) {
    const float c0[3] = {m[0], m[1], m[2]};
    const float c1[3] = {m[4], m[5], m[6]};
    const float c2[3] = {m[8], m[9], m[10]};
    const float n0 = c0[0] * c0[0] + c0[1] * c0[1] + c0[2] * c0[2];
    const float n1 = c1[0] * c1[0] + c1[1] * c1[1] + c1[2] * c1[2];
    const float n2 = c2[0] * c2[0] + c2[1] * c2[1] + c2[2] * c2[2];
    const float sbar2 = (n0 + n1 + n2) / 3.0f;
    if (!(sbar2 > 0.0f)) return false;
    const float tol = CALIPER_GEOM_RIGID_TOL * sbar2;
    const float d01 = c0[0] * c1[0] + c0[1] * c1[1] + c0[2] * c1[2];
    const float d02 = c0[0] * c2[0] + c0[1] * c2[1] + c0[2] * c2[2];
    const float d12 = c1[0] * c2[0] + c1[1] * c2[1] + c1[2] * c2[2];
    if (std::fabs(d01) > tol || std::fabs(d02) > tol || std::fabs(d12) > tol)
        return false;
    if (std::fabs(n0 - sbar2) > tol || std::fabs(n1 - sbar2) > tol ||
        std::fabs(n2 - sbar2) > tol)
        return false;
    return true;
}

// Byte extent a tensor addresses: (max linear element index + 1) * elem_size,
// from shape×strides. The bridge already bounds this in *elements* against a
// sane cap, but only the backend sees the actual id<MTLBuffer> — so the device
// paths bound the buffer's real byte length here before dispatch. A short
// buffer (e.g. a caller passing a half-sized MTLBuffer) is rejected rather than
// letting the shader/blit read past the allocation and fault the GPU.
static uint64_t tensor_extent_bytes(const CaliperTensor& t, uint64_t elem_size) {
    uint64_t maxidx = 0;
    for (int i = 0; i < t.ndim; ++i)
        maxidx += (uint64_t)(t.shape[i] - 1) * (uint64_t)t.strides[i];
    return (maxidx + 1) * elem_size;
}

class MetalRenderer final : public HostRenderer {
public:
    const char* name() const override { return "metal"; }
    const char* last_device_path() const override { return last_device_path_; }
    CaliperDeviceKind interop_device() const override { return CALIPER_DEV_METAL; }

    // M2b shipped: a non-NULL t.stream is GPU-ordered after the producer
    // queue in both device paths above (D24).
    bool honors_stream_ordered_handoff() const override { return true; }

    // Metal owns no GL context: the window must be created with NO_API so GLFW
    // does not attach an OpenGL context we would fight over. Runs before
    // glfwCreateWindow (sibling to the GL backend's profile hints).
    void window_hints() override {
        glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
    }

    bool init(GLFWwindow* window) override {
        window_ = window;
        device_ = MTLCreateSystemDefaultDevice();
        if (device_ == nil) return false;
        queue_ = [device_ newCommandQueue];
        if (queue_ == nil) return false;

        NSWindow* nswin = glfwGetCocoaWindow(window);
        if (nswin == nil) return false;
        layer_ = [CAMetalLayer layer];
        layer_.device = device_;
        layer_.pixelFormat = MTLPixelFormatBGRA8Unorm;
        layer_.framebufferOnly = YES;
        nswin.contentView.layer = layer_;
        nswin.contentView.wantsLayer = YES;

        // ImGui core/ImPlot contexts are host-owned and already created; only
        // the GLFW + Metal *backends* belong to this renderer.
        if (!ImGui_ImplGlfw_InitForOther(window, true)) return false;
        if (!ImGui_ImplMetal_Init(device_)) {
            ImGui_ImplGlfw_Shutdown();   // undo the GLFW backend for a clean fallback
            return false;
        }

        textures_ = [NSMutableDictionary dictionary];
        events_ = [NSMutableDictionary dictionary];
        pass_desc_ = [MTLRenderPassDescriptor new];
        return true;
    }

    // Start-of-frame: size the layer, acquire a drawable, open the render pass
    // with a CLEAR matching the GL backend's background, then ImGui NewFrame.
    // The clear lives here (not render()) per the C1 contract — anything the
    // app draws this frame lands on top of it.
    void new_frame() override {
        @autoreleasepool {
            int w = 0, h = 0;
            glfwGetFramebufferSize(window_, &w, &h);
            if (w > 0 && h > 0)
                layer_.drawableSize = CGSizeMake((CGFloat)w, (CGFloat)h);

            drawable_ = [layer_ nextDrawable];   // may be nil under load

            pass_desc_.colorAttachments[0].texture = drawable_ ? drawable_.texture : nil;
            pass_desc_.colorAttachments[0].loadAction = MTLLoadActionClear;
            pass_desc_.colorAttachments[0].storeAction = MTLStoreActionStore;
            pass_desc_.colorAttachments[0].clearColor =
                MTLClearColorMake(0.05, 0.05, 0.08, 1.0);   // == GL glClearColor

            frame_cmd_ = [queue_ commandBuffer];
            if (drawable_) {
                frame_enc_ = [frame_cmd_ renderCommandEncoderWithDescriptor:pass_desc_];
            } else {
                frame_enc_ = nil;   // still keep ImGui frame balanced below
            }

            ImGui_ImplMetal_NewFrame(pass_desc_);
            ImGui_ImplGlfw_NewFrame();
            ImGui::NewFrame();
        }
    }

    void render(int /*fb_w*/, int /*fb_h*/) override {
        @autoreleasepool {
            ImGui::Render();                       // consume the frame regardless
            if (frame_enc_ && drawable_) {
                ImGui_ImplMetal_RenderDrawData(ImGui::GetDrawData(),
                                               frame_cmd_, frame_enc_);
                [frame_enc_ endEncoding];
                [frame_cmd_ presentDrawable:drawable_];
                [frame_cmd_ commit];
            } else if (frame_cmd_) {
                [frame_cmd_ commit];               // nothing to draw; flush
            }
            frame_enc_ = nil;
            frame_cmd_ = nil;
            drawable_ = nil;
        }
    }

    void shutdown() override {
        ImGui_ImplMetal_Shutdown();
        ImGui_ImplGlfw_Shutdown();
        [events_ removeAllObjects]; events_ = nil; event_values_.clear();
        depth_textures_.clear();
        [textures_ removeAllObjects];
        textures_ = nil;
        imported_.clear();
        geom_prim_inst_staging_ = nil;   // ARC releases the grow-only G14 staging
        cmap_pipeline_ = nil;
        points_pipeline_ = nil;
        geom_lib_ = nil;
        geom_pipelines_.clear();
        depth_states_.clear();
        pass_desc_ = nil;
        queue_ = nil;
        layer_ = nil;
        device_ = nil;
    }

    // ---- Texture ops. id table maps an opaque uint64 -> strong MTLTexture;
    // the raw Metal handle never leaves this file as an id. ----
    uint64_t tex_create_rgba8(int w, int h) override {
        if (w <= 0 || h <= 0 || device_ == nil) return 0;
        MTLTextureDescriptor* d =
            [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA8Unorm
                                                               width:(NSUInteger)w
                                                              height:(NSUInteger)h
                                                           mipmapped:NO];
        d.storageMode = MTLStorageModeShared;                  // CPU upload + readback
        d.usage = MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite;  // compute write
        id<MTLTexture> tex = [device_ newTextureWithDescriptor:d];
        if (tex == nil) return 0;

        uint64_t id = next_id_++;
        textures_[@(id)] = tex;
        return id;
    }

    bool tex_upload_rgba8(uint64_t tex, const void* data, int w, int h) override {
        id<MTLTexture> t = lookup(tex);
        if (t == nil || data == nullptr || w <= 0 || h <= 0) return false;
        [t replaceRegion:MTLRegionMake2D(0, 0, (NSUInteger)w, (NSUInteger)h)
             mipmapLevel:0
               withBytes:data
             bytesPerRow:(NSUInteger)w * 4];
        return true;
    }

    void tex_release(uint64_t tex) override {
        [textures_ removeObjectForKey:@(tex)];
        [events_ removeObjectForKey:@(tex)]; event_values_.erase(tex);
        depth_textures_.erase(tex);
    }

    uint64_t tex_imtexture_id(uint64_t tex) override {
        id<MTLTexture> t = lookup(tex);
        return t == nil ? 0 : (uint64_t)(__bridge void*)t;
    }

    // Device-resident update: t.data is an id<MTLBuffer> living on this device.
    // No CPU roundtrip on either path. Returns false -> caller CPU-stages.
    bool tex_update_from_device(uint64_t tex, const CaliperTensor& t,
                                const uint32_t* lut256,
                                float vmin, float vmax) override {
        id<MTLTexture> dst = lookup(tex);
        if (dst == nil) return false;
        if (t.device != CALIPER_DEV_METAL || t.data == nullptr) return false;
        id<MTLBuffer> src = (__bridge id<MTLBuffer>)t.data;
        if (src == nil) return false;

        if (t.dtype == CALIPER_DT_F32 && lut256 != nullptr)
            return colormap_compute_from(tex, dst, src, 0, t, lut256, vmin, vmax,
                                         /*imported=*/false);
        if (t.dtype == CALIPER_DT_U8)
            return blit_u8_from(tex, dst, src, 0, t, /*imported=*/false);
        return false;
    }

    // ---- v1.2 imported allocations (in-process MTLBuffer import) -----------
    // The Metal analog of Vulkan's DuplicateHandle + VkImportMemory: there is
    // no OS handle transfer on Apple unified memory, so the "dup" is an ObjC
    // strong retain. Lights up CALIPER_BRIDGE_CAP_IMPORT_ALLOC via the bridge.
    bool supports_external_import() const override { return device_ != nil; }

    uint64_t import_external_allocation(void* os_handle, uint64_t size_bytes,
                                        uint32_t handle_type) override {
        if (handle_type != CALIPER_ALLOC_HANDLE_MTLBUFFER) return 0;
        if (os_handle == nullptr || size_bytes == 0 || device_ == nil) return 0;
        id<MTLBuffer> buf = (__bridge id<MTLBuffer>)os_handle;
        if (buf == nil) return 0;
        if (buf.device.registryID != device_.registryID) return 0;  // wrong GPU
        if (buf.length < size_bytes) return 0;   // caller overclaims — refuse
        const uint64_t iid = next_import_id_++;
        imported_[iid] = buf;                    // ARC strong ref IS the dup
        return iid;
    }

    void release_external_allocation(uint64_t iid) override { imported_.erase(iid); }

    // Colormap/blit a texture FROM an imported buffer at a byte offset, with NO
    // data copy (the applet's kernels already wrote the bytes). Guards mirror
    // tex_update_from_device; the byte offset rides setBuffer:offset: (f32) or
    // sourceOffset: (u8). Colormap/vmin/vmax are the texture's pinned mapping.
    bool tex_update_from_imported(uint64_t tex, uint64_t alloc, uint64_t offset_bytes,
                                  const CaliperTensor& desc, int32_t colormap,
                                  float vmin, float vmax) override {
        id<MTLTexture> dst = lookup(tex);
        id<MTLBuffer>  src = lookup_import(alloc);
        if (dst == nil || src == nil) return false;
        if (offset_bytes % 4 != 0 || offset_bytes > src.length) return false;
        if (desc.dtype == CALIPER_DT_F32) {
            const uint32_t* lut = colormap_lut(colormap);
            if (lut == nullptr) return false;
            return colormap_compute_from(tex, dst, src, offset_bytes, desc,
                                         lut, vmin, vmax, /*imported=*/true);
        }
        if (desc.dtype == CALIPER_DT_U8)
            return blit_u8_from(tex, dst, src, offset_bytes, desc, /*imported=*/true);
        return false;
    }

    // ---- caliper.geometry.v1: instanced points from imported allocations ----
    // Same gate as the imported-texture path: point data lives in v1.2 imported
    // MTLBuffers, which only exist when external import is up.
    bool supports_geometry() const override { return supports_external_import(); }
    bool supports_geometry_primitives() const override { return supports_external_import(); }
    bool supports_geometry_textured() const override { return supports_external_import(); }
    // geometry.v1_3: N>1 instanced draws ride the same imported-MTLBuffer path
    // as textured meshes, so the gate is identical (mirrors _textured above).
    bool supports_geometry_instanced() const override { return supports_external_import(); }

    // An offscreen render target that is ALSO an ordinary sampled texture: it
    // lives in textures_, so tex_imtexture_id / debug_readback / tex_release
    // work unchanged; the RenderTarget usage makes it drawable by the point
    // pass. Cleared to opaque black at create so ImGui sampling before the
    // first draw is defined.
    uint64_t geom_create_view(int w, int h) override {
        if (w <= 0 || h <= 0 || device_ == nil) return 0;
        MTLTextureDescriptor* d =
            [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA8Unorm
                                                               width:(NSUInteger)w
                                                              height:(NSUInteger)h
                                                           mipmapped:NO];
        d.storageMode = MTLStorageModeShared;                // unified memory: renderable + readback
        d.usage = MTLTextureUsageShaderRead | MTLTextureUsageRenderTarget;
        id<MTLTexture> tex = [device_ newTextureWithDescriptor:d];
        if (tex == nil) return 0;
        @autoreleasepool {                                    // defined pre-first-draw: opaque black
            MTLRenderPassDescriptor* rp = [MTLRenderPassDescriptor renderPassDescriptor];
            rp.colorAttachments[0].texture     = tex;
            rp.colorAttachments[0].loadAction  = MTLLoadActionClear;
            rp.colorAttachments[0].storeAction = MTLStoreActionStore;
            rp.colorAttachments[0].clearColor  = MTLClearColorMake(0, 0, 0, 1);
            id<MTLCommandBuffer> cb = [queue_ commandBuffer];
            id<MTLRenderCommandEncoder> enc = [cb renderCommandEncoderWithDescriptor:rp];
            [enc endEncoding];
            [cb commit];
        }
        uint64_t tid = next_id_++;
        textures_[@(tid)] = tex;
        return tid;
    }

    uint64_t geom_create_view_ex(int w, int h, uint32_t flags) override {
        if ((flags & ~CALIPER_GEOM_VIEW_DEPTH) != 0u) return 0;
        if (w <= 0 || h <= 0 || device_ == nil) return 0;

        MTLTextureDescriptor* d =
            [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA8Unorm
                                                               width:(NSUInteger)w
                                                              height:(NSUInteger)h
                                                           mipmapped:NO];
        d.storageMode = MTLStorageModeShared;
        d.usage = MTLTextureUsageShaderRead | MTLTextureUsageRenderTarget;
        id<MTLTexture> color = [device_ newTextureWithDescriptor:d];
        if (color == nil) return 0;

        id<MTLTexture> depth = nil;
        if ((flags & CALIPER_GEOM_VIEW_DEPTH) != 0u) {
            MTLTextureDescriptor* dd =
                [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatDepth32Float
                                                                   width:(NSUInteger)w
                                                                  height:(NSUInteger)h
                                                               mipmapped:NO];
            dd.storageMode = MTLStorageModePrivate;
            dd.usage = MTLTextureUsageRenderTarget;
            depth = [device_ newTextureWithDescriptor:dd];
            if (depth == nil) return 0;
        }

        @autoreleasepool {
            MTLRenderPassDescriptor* rp = [MTLRenderPassDescriptor renderPassDescriptor];
            rp.colorAttachments[0].texture     = color;
            rp.colorAttachments[0].loadAction  = MTLLoadActionClear;
            rp.colorAttachments[0].storeAction = MTLStoreActionStore;
            rp.colorAttachments[0].clearColor  = MTLClearColorMake(0, 0, 0, 1);
            if (depth != nil) {
                rp.depthAttachment.texture     = depth;
                rp.depthAttachment.loadAction  = MTLLoadActionClear;
                rp.depthAttachment.storeAction = MTLStoreActionDontCare;
                rp.depthAttachment.clearDepth  = 1.0;
            }
            id<MTLCommandBuffer> cb = [queue_ commandBuffer];
            id<MTLRenderCommandEncoder> enc = [cb renderCommandEncoderWithDescriptor:rp];
            [enc endEncoding];
            [cb commit];
        }

        uint64_t tid = next_id_++;
        textures_[@(tid)] = color;
        if (depth != nil) depth_textures_[tid] = depth;
        return tid;
    }

    // One view frame, atomically: clear + draw `count` vertex-pulled points.
    // Positions/attr are element bases into whole-bound imported buffers, so the
    // offset gate is 4-byte alignment (same as Vulkan). Additive blend, no
    // depth. Metal NDC is +y up with a positive-height viewport, which lands the
    // GL-style ndc_for_pixel mapping directly (NO y-flip, unlike Vulkan). Every
    // gate runs BEFORE the encoder exists — creating the encoder performs the
    // clear, so any late refusal would violate pixels-untouched.
    bool geom_draw_points(uint64_t view_tex, const float* view16, const float* proj16,
                          uint64_t pos_alloc, uint64_t pos_offset, uint64_t count,
                          uint64_t attr_alloc, uint64_t attr_offset,
                          const uint32_t* lut256, float vmin, float vmax,
                          float size_px, uint32_t clear_rgba) override {
        @autoreleasepool {
            // ---- every gate BEFORE the encoder exists: false = pixels untouched ----
            id<MTLTexture> t = lookup(view_tex);
            if (t == nil || view16 == nullptr || proj16 == nullptr) return false;
            id<MTLBuffer> pos = nil, attr = nil;
            if (count > 0) {
                pos = lookup_import(pos_alloc);
                if (pos == nil || pos_offset % 4 != 0) return false;
                if (count > UINT64_MAX / 12) return false;
                if (pos_offset > pos.length || count * 12 > pos.length - pos_offset)
                    return false;
                if (attr_alloc != 0) {
                    attr = lookup_import(attr_alloc);
                    if (attr == nil || attr_offset % 4 != 0 || lut256 == nullptr)
                        return false;
                    if (attr_offset > attr.length || count * 4 > attr.length - attr_offset)
                        return false;
                }
                if (!ensure_points_pipeline()) return false;
            }

            MTLRenderPassDescriptor* rp = [MTLRenderPassDescriptor renderPassDescriptor];
            rp.colorAttachments[0].texture     = t;
            rp.colorAttachments[0].loadAction  = MTLLoadActionClear;
            rp.colorAttachments[0].storeAction = MTLStoreActionStore;
            rp.colorAttachments[0].clearColor  = MTLClearColorMake(
                (double)( clear_rgba        & 0xFFu) / 255.0,
                (double)((clear_rgba >> 8)  & 0xFFu) / 255.0,
                (double)((clear_rgba >> 16) & 0xFFu) / 255.0,
                (double)((clear_rgba >> 24) & 0xFFu) / 255.0);
            id<MTLCommandBuffer> cb = [queue_ commandBuffer];
            id<MTLRenderCommandEncoder> enc = [cb renderCommandEncoderWithDescriptor:rp];
            if (enc == nil) return false;   // nothing encoded, nothing cleared

            if (count > 0) {
                GeomParams p{};
                // mvp = proj * view, column-major — same loop as vulkan_renderer.cpp.
                for (int c = 0; c < 4; ++c)
                    for (int r = 0; r < 4; ++r) {
                        float acc = 0.f;
                        for (int k = 0; k < 4; ++k)
                            acc += proj16[k * 4 + r] * view16[c * 4 + k];
                        p.mvp[c * 4 + r] = acc;
                    }
                p.pos_base  = (uint32_t)(pos_offset / 4);
                p.attr_base = (uint32_t)(attr_offset / 4);
                p.use_attr  = attr != nil ? 1u : 0u;
                p.vmin = vmin; p.vmax = vmax;
                p.size_px = std::min(std::max(size_px, 1.0f), 511.0f);  // Metal point-size cap

                static const uint32_t kZeroLut[256] = {};   // valid-but-unread when flat
                [enc setRenderPipelineState:points_pipeline_];
                MTLViewport vp = {0.0, 0.0, (double)t.width, (double)t.height, 0.0, 1.0};
                [enc setViewport:vp];                        // positive height: no Y flip on Metal
                [enc setVertexBuffer:pos offset:0 atIndex:0];
                [enc setVertexBuffer:(attr != nil ? attr : pos) offset:0 atIndex:1];
                [enc setVertexBytes:(lut256 ? lut256 : kZeroLut)
                             length:256 * sizeof(uint32_t) atIndex:2];  // 1 KB < 4 KB setBytes cap
                [enc setVertexBytes:&p length:sizeof(p) atIndex:3];
                [enc drawPrimitives:MTLPrimitiveTypePoint
                        vertexStart:0 vertexCount:(NSUInteger)count];
            }
            [enc endEncoding];
            [cb commit];   // no CPU wait: same-queue_ commit order covers the frame's
                           // sampling; producer (MPS) writes are already CPU-drained
                           // before publish (flow_scope sync contract).
            last_device_path_ = "points-imported";
            return true;
        }
    }

    bool geom_draw_primitives(uint64_t view_tex,
                              const float* view16, const float* proj16,
                              const HostGeomDraw* draws, uint32_t count,
                              uint32_t clear_rgba) override {
        @autoreleasepool {
            id<MTLTexture> color = lookup(view_tex);
            if (color == nil) return metal_geom_fail("unknown color view");
            if (view16 == nullptr || proj16 == nullptr) return metal_geom_fail("null camera matrices");
            if (count > 0 && draws == nullptr) return metal_geom_fail("null draw array");
            id<MTLTexture> depth = nil;
            auto dt = depth_textures_.find(view_tex);
            if (dt != depth_textures_.end()) depth = dt->second;

            struct EncodedDraw {
                const HostGeomDraw* d = nullptr;
                id<MTLBuffer> pos = nil;
                id<MTLBuffer> idx = nil;
                id<MTLBuffer> nrm = nil;
                id<MTLBuffer> attr = nil;
                id<MTLBuffer> uv = nil;
                id<MTLBuffer> inst = nil;    // (N,16) f32 instance matrices (idx 7)
                id<MTLBuffer> iattr = nil;   // (N,) per-instance tint scalar (idx 8)
                id<MTLTexture> texture = nil;
                NSUInteger consumed = 0;
                NSUInteger n_inst = 1;       // instanceCount (1 == non-instanced)
                bool use_instance = false;
                MTLPrimitiveType prim = MTLPrimitiveTypeTriangle;
                id<MTLRenderPipelineState> pipeline = nil;
                id<MTLDepthStencilState> depth_state = nil;
                PrimParams params{};
            };
            std::vector<EncodedDraw> encs;
            encs.reserve(count);

            for (uint32_t i = 0; i < count; ++i) {
                const HostGeomDraw& d = draws[i];
                if (d.vertex_count == 0 || d.vertex_count > UINT32_MAX)
                    return metal_geom_fail("bad vertex count");
                id<MTLBuffer> pos = lookup_import(d.pos_alloc);
                if (pos == nil) return metal_geom_fail("unknown pos import");
                if (d.pos_offset % 4 != 0) return metal_geom_fail("pos offset misaligned");
                if (d.vertex_count > UINT64_MAX / 12u) return metal_geom_fail("position byte overflow");
                if (d.pos_offset > pos.length || d.vertex_count * 12u > pos.length - d.pos_offset)
                    return metal_geom_fail("positions out of bounds");

                id<MTLBuffer> idx = nil;
                uint64_t consumed = d.vertex_count;
                if (d.index_alloc != 0) {
                    idx = lookup_import(d.index_alloc);
                    if (idx == nil || d.index_offset % 4 != 0 || d.index_count == 0)
                        return metal_geom_fail("bad index import/range");
                    if (d.index_count > UINT32_MAX || d.index_count > UINT64_MAX / 4u)
                        return metal_geom_fail("index count overflow");
                    if (d.index_offset > idx.length || d.index_count * 4u > idx.length - d.index_offset)
                        return metal_geom_fail("indices out of bounds");
                    consumed = d.index_count;
                }
                if ((d.topology == CALIPER_GEOM_TOPO_LINES ||
                     d.topology == CALIPER_GEOM_TOPO_LINE_STRIP) && consumed < 2)
                    return metal_geom_fail("line draw has too few vertices");
                if ((d.topology == CALIPER_GEOM_TOPO_TRIANGLES ||
                     d.topology == CALIPER_GEOM_TOPO_TRIANGLE_STRIP) && consumed < 3)
                    return metal_geom_fail("triangle draw has too few vertices");
                if (d.topology == CALIPER_GEOM_TOPO_POINTS && !(d.size_px > 0.0f))
                    return metal_geom_fail("bad point size");
                if (d.depth_flags != 0 && depth == nil) return metal_geom_fail("depth draw on depthless view");

                id<MTLBuffer> nrm = nil;
                if (d.shade_mode == CALIPER_GEOM_SHADE_LAMBERT) {
                    nrm = lookup_import(d.normal_alloc);
                    if (nrm == nil) return metal_geom_fail("unknown normal import");
                    if (d.normal_offset % 4 != 0) return metal_geom_fail("normal offset misaligned");
                    if (d.normal_offset > nrm.length ||
                        d.vertex_count * 12u > nrm.length - d.normal_offset)
                        return metal_geom_fail("normals out of bounds");
                } else if (d.normal_alloc != 0) {
                    nrm = lookup_import(d.normal_alloc);
                    if (nrm == nil) return metal_geom_fail("unknown optional normal import");
                    if (d.normal_offset % 4 != 0) return metal_geom_fail("optional normal offset misaligned");
                    if (d.normal_offset > nrm.length ||
                        d.vertex_count * 12u > nrm.length - d.normal_offset)
                        return metal_geom_fail("optional normals out of bounds");
                }

                id<MTLBuffer> attr = nil;
                if (d.color_mode == CALIPER_GEOM_COLOR_COLORMAP ||
                    d.color_mode == CALIPER_GEOM_COLOR_VERTEX_RGBA) {
                    attr = lookup_import(d.attr_alloc);
                    if (attr == nil) return metal_geom_fail("unknown attr import");
                    if (d.attr_offset % 4 != 0) return metal_geom_fail("attr offset misaligned");
                    if (d.vertex_count > UINT64_MAX / 4u) return metal_geom_fail("attr byte overflow");
                    if (d.attr_offset > attr.length ||
                        d.vertex_count * 4u > attr.length - d.attr_offset)
                        return metal_geom_fail("attributes out of bounds");
                    if (d.color_mode == CALIPER_GEOM_COLOR_COLORMAP && d.lut256 == nullptr)
                        return metal_geom_fail("missing colormap LUT");
                }

                id<MTLBuffer> uv = nil;
                id<MTLTexture> texture = nil;
                if (d.color_mode == CALIPER_GEOM_COLOR_TEXTURE) {
                    uv = lookup_import(d.uv_alloc);
                    texture = lookup(d.texture);
                    if (uv == nil) return metal_geom_fail("unknown uv import");
                    // Refuse sampling ANY geometry view: unknown id, the current
                    // target, or any entry created by geom_create_view* (which
                    // uniquely carry MTLTextureUsageRenderTarget — tex_create_rgba8
                    // never does). Metal views share textures_ with sampled
                    // textures, so the usage flag is the marker create_view sets.
                    // Mirrors Vulkan's `fb != VK_NULL_HANDLE` refusal.
                    if (texture == nil || d.texture == view_tex ||
                        (texture.usage & MTLTextureUsageRenderTarget) != 0)
                        return metal_geom_fail("bad sampled texture");
                    if (d.uv_offset % 4 != 0)
                        return metal_geom_fail("uv offset misaligned");
                    if (d.vertex_count > UINT64_MAX / 8u)
                        return metal_geom_fail("uv byte overflow");
                    if (d.uv_offset > uv.length ||
                        d.vertex_count * 8u > uv.length - d.uv_offset)
                        return metal_geom_fail("uvs out of bounds");
                }

                // ---- v1.3 instance tail re-gate (§5/§5.1): G1-G12 against this
                // renderer's imported_ table, byte-same reasons as the host
                // battery, plus G14 which executes ONLY here. HostGeomDraw carries
                // resolved renderer ids + byte offsets (all zero -> non-instanced,
                // the exact v1.2 path). Every failure returns before the encoder
                // exists, so the view's pixels are bit-untouched. ----
                id<MTLBuffer> inst = nil;
                id<MTLBuffer> iattr = nil;
                bool use_instance = false;
                bool use_instance_attr = false;
                NSUInteger n_inst = 1;
                if (d.instance_alloc != 0) {
                    // G1 (cap) is granted upstream; a resolved alloc is instanced.
                    if (d.instance_count == 0)
                        return metal_geom_fail("instanced draw needs N>0");        // G2
                    if (d.instance_count > UINT32_MAX)
                        return metal_geom_fail("too many instances");              // G3
                    if (d.instance_offset % 4 != 0)
                        return metal_geom_fail("instance offset misaligned");      // G4
                    if (d.instance_offset / 4u > UINT32_MAX)
                        return metal_geom_fail("instance base exceeds 32 bits");   // G6
                    inst = lookup_import(d.instance_alloc);
                    if (inst == nil)
                        return metal_geom_fail("unknown instance alloc");          // G7
                    if (d.instance_count > UINT64_MAX / 64u)
                        return metal_geom_fail("instances byte count overflow");
                    if (d.instance_offset > inst.length ||
                        d.instance_count * 64u > inst.length - d.instance_offset)
                        return metal_geom_fail("instances out of imported bounds"); // G5
                    use_instance = true;
                    n_inst = (NSUInteger)d.instance_count;
                }
                if (d.instance_attr_alloc != 0) {
                    if (!use_instance)
                        return metal_geom_fail("instance attr without instances"); // G8
                    if (d.instance_attr_offset % 4 != 0)
                        return metal_geom_fail("instance attr offset misaligned"); // G9
                    iattr = lookup_import(d.instance_attr_alloc);
                    if (iattr == nil)
                        return metal_geom_fail("unknown instance attr alloc");     // G11
                    if (d.instance_count > UINT64_MAX / 4u)
                        return metal_geom_fail("instance attr byte count overflow");
                    if (d.instance_attr_offset > iattr.length ||
                        d.instance_count * 4u > iattr.length - d.instance_attr_offset)
                        return metal_geom_fail("instance attr out of imported bounds"); // G10
                    if (d.lut256 == nullptr)
                        return metal_geom_fail("instance tint needs colormap");    // G12
                    use_instance_attr = true;
                }
                // G14 (§5.1) rigidity is NOT checked per-draw here: it needs the
                // instance bytes on the host, and a real device tensor (torch MPS
                // buffer) imports as MTLStorageModePrivate — contents() is not
                // host-readable — so a private read needs a blit round-trip. Doing
                // that per draw would fire a fresh alloc + commit + wait for every
                // LAMBERT-instanced draw. Instead the batched G14 pass runs ONCE
                // after this metadata gate loop (below), mirroring the Vulkan
                // reference shape: one grow-only staging buffer, one blit command
                // buffer for all private draws, one wait, then the comparisons —
                // and still before any render encoder exists (refusal = pixels
                // untouched).

                MTLPrimitiveType prim;
                switch (d.topology) {
                    case CALIPER_GEOM_TOPO_POINTS:         prim = MTLPrimitiveTypePoint; break;
                    case CALIPER_GEOM_TOPO_LINES:          prim = MTLPrimitiveTypeLine; break;
                    case CALIPER_GEOM_TOPO_LINE_STRIP:     prim = MTLPrimitiveTypeLineStrip; break;
                    case CALIPER_GEOM_TOPO_TRIANGLES:      prim = MTLPrimitiveTypeTriangle; break;
                    case CALIPER_GEOM_TOPO_TRIANGLE_STRIP: prim = MTLPrimitiveTypeTriangleStrip; break;
                    default: return metal_geom_fail("bad topology");
                }

                id<MTLRenderPipelineState> pipe =
                    geom_pipeline(topo_class(d.topology), d.blend_mode,
                                  depth != nil,
                                  d.color_mode == CALIPER_GEOM_COLOR_TEXTURE);
                id<MTLDepthStencilState> ds = geom_depth_state(d.depth_flags);
                if (pipe == nil) return metal_geom_fail("pipeline creation failed");
                if (ds == nil) return metal_geom_fail("depth-state creation failed");

                EncodedDraw e;
                e.d = &d;
                e.pos = pos;
                e.idx = idx;
                e.nrm = nrm;
                e.attr = attr;
                e.uv = uv;
                e.inst = inst;
                e.iattr = iattr;
                e.texture = texture;
                e.consumed = (NSUInteger)consumed;
                e.n_inst = n_inst;
                e.use_instance = use_instance;
                e.prim = prim;
                e.pipeline = pipe;
                e.depth_state = ds;

                float view_model[16], mvp[16];
                mat4_mul_cm(view16, d.model, view_model);
                mat4_mul_cm(proj16, view_model, mvp);
                std::memcpy(e.params.mvp, mvp, sizeof(mvp));
                normal_matrix_columns(view_model, e.params.nmat0, e.params.nmat1, e.params.nmat2);
                e.params.pos_base = (uint32_t)(d.pos_offset / 4u);
                e.params.idx_base = (uint32_t)(d.index_offset / 4u);
                e.params.nrm_base = (uint32_t)(d.normal_offset / 4u);
                e.params.attr_base = (uint32_t)(d.attr_offset / 4u);
                e.params.use_index = d.index_alloc != 0 ? 1u : 0u;
                e.params.vertex_count = (uint32_t)d.vertex_count;
                e.params.color_mode = d.color_mode;
                e.params.shade_mode = d.shade_mode;
                e.params.flat_rgba = d.flat_rgba;
                e.params.vmin = d.vmin;
                e.params.vmax = d.vmax;
                e.params.size_px = std::min(std::max(d.size_px, 1.0f), 511.0f);
                if (d.uv_offset / 4 > UINT32_MAX)
                    return metal_geom_fail("uv base exceeds 32 bits");
                e.params.uv_base = (uint32_t)(d.uv_offset / 4u);
                e.params.use_instance = use_instance ? 1u : 0u;
                e.params.inst_base = (uint32_t)(d.instance_offset / 4u);
                e.params.use_instance_attr = use_instance_attr ? 1u : 0u;
                e.params.inst_attr_base = (uint32_t)(d.instance_attr_offset / 4u);
                encs.push_back(e);
            }

            // ---- G14 (§5.1): every instance upper-3x3 on a LAMBERT-instanced
            // draw must be rigid+uniform-scale. Mirrors the Vulkan reference
            // shape (vulkan_renderer.cpp §5.1): collect ALL such draws first;
            // shared-storage imports are read via contents() directly (zero-copy,
            // unified memory), while private-storage imports (real device tensors,
            // e.g. torch MPS buffers whose contents() is not host-readable) are
            // batched into ONE grow-only staging buffer, copied by ONE blit
            // command buffer, and read after ONE waitUntilCompleted — never a
            // per-draw alloc+commit+wait. Placed after the metadata gate loop and
            // BEFORE any render encoder exists, so a refusal leaves the view's
            // pixels bit-untouched. The float-order rigidity check
            // (instance_upper3x3_rigid) is identical for both storage paths. ----
            {
                struct RigidRead {
                    id<MTLBuffer> inst;
                    NSUInteger    src_off;
                    NSUInteger    dst_off;
                    NSUInteger    n;
                };
                std::vector<RigidRead> priv_reads;
                NSUInteger stage_bytes = 0;
                for (const EncodedDraw& e : encs) {
                    if (!(e.use_instance &&
                          e.params.shade_mode == CALIPER_GEOM_SHADE_LAMBERT))
                        continue;
                    if (e.inst.storageMode == MTLStorageModeShared) {
                        // Zero-copy read straight from unified memory (no blit).
                        const float* base = (const float*)
                            ((const uint8_t*)e.inst.contents + e.d->instance_offset);
                        for (NSUInteger k = 0; k < e.n_inst; ++k)
                            if (!instance_upper3x3_rigid(base + 16u * k))
                                return metal_geom_fail("instanced lambert needs rigid+uniform-scale"); // G14
                    } else {
                        priv_reads.push_back({e.inst,
                                              (NSUInteger)e.d->instance_offset,
                                              stage_bytes, e.n_inst});
                        stage_bytes += e.n_inst * 64u;
                    }
                }
                if (!priv_reads.empty()) {
                    // Grow-only member staging buffer (created once, ×2 on demand,
                    // reused across frames, released in shutdown()).
                    if (geom_prim_inst_staging_ == nil ||
                        geom_prim_inst_staging_.length < stage_bytes) {
                        NSUInteger grow = geom_prim_inst_staging_ ?
                            geom_prim_inst_staging_.length : 64u;
                        while (grow < stage_bytes) grow *= 2;
                        geom_prim_inst_staging_ =
                            [device_ newBufferWithLength:grow
                                                 options:MTLResourceStorageModeShared];
                        if (geom_prim_inst_staging_ == nil)
                            return metal_geom_fail("instance rigidity staging alloc failed");
                    }
                    id<MTLCommandBuffer> cb = [queue_ commandBuffer];
                    id<MTLBlitCommandEncoder> blit = [cb blitCommandEncoder];
                    for (const RigidRead& r : priv_reads)
                        [blit copyFromBuffer:r.inst
                                sourceOffset:r.src_off
                                    toBuffer:geom_prim_inst_staging_
                           destinationOffset:r.dst_off
                                        size:r.n * 64u];
                    [blit endEncoding];
                    [cb commit];
                    [cb waitUntilCompleted];
                    for (const RigidRead& r : priv_reads) {
                        const float* base = (const float*)
                            ((const uint8_t*)geom_prim_inst_staging_.contents + r.dst_off);
                        for (NSUInteger k = 0; k < r.n; ++k)
                            if (!instance_upper3x3_rigid(base + 16u * k))
                                return metal_geom_fail("instanced lambert needs rigid+uniform-scale"); // G14
                    }
                }
            }

            MTLRenderPassDescriptor* rp = [MTLRenderPassDescriptor renderPassDescriptor];
            rp.colorAttachments[0].texture     = color;
            rp.colorAttachments[0].loadAction  = MTLLoadActionClear;
            rp.colorAttachments[0].storeAction = MTLStoreActionStore;
            rp.colorAttachments[0].clearColor  = MTLClearColorMake(
                (double)( clear_rgba        & 0xFFu) / 255.0,
                (double)((clear_rgba >> 8)  & 0xFFu) / 255.0,
                (double)((clear_rgba >> 16) & 0xFFu) / 255.0,
                (double)((clear_rgba >> 24) & 0xFFu) / 255.0);
            if (depth != nil) {
                rp.depthAttachment.texture     = depth;
                rp.depthAttachment.loadAction  = MTLLoadActionClear;
                rp.depthAttachment.storeAction = MTLStoreActionDontCare;
                rp.depthAttachment.clearDepth  = 1.0;
            }

            id<MTLCommandBuffer> cb = [queue_ commandBuffer];
            id<MTLRenderCommandEncoder> re = [cb renderCommandEncoderWithDescriptor:rp];
            if (re == nil) return metal_geom_fail("render encoder creation failed");
            MTLViewport vp = {0.0, 0.0, (double)color.width, (double)color.height, 0.0, 1.0};
            [re setViewport:vp];
            static const uint32_t kZeroLut[256] = {};
            for (const EncodedDraw& e : encs) {
                [re setRenderPipelineState:e.pipeline];
                [re setDepthStencilState:e.depth_state];
                [re setVertexBuffer:e.pos offset:0 atIndex:0];
                [re setVertexBuffer:(e.idx != nil ? e.idx : e.pos) offset:0 atIndex:1];
                [re setVertexBuffer:(e.nrm != nil ? e.nrm : e.pos) offset:0 atIndex:2];
                [re setVertexBuffer:(e.attr != nil ? e.attr : e.pos) offset:0 atIndex:3];
                [re setVertexBytes:(e.d->lut256 ? e.d->lut256 : kZeroLut)
                            length:256 * sizeof(uint32_t) atIndex:4];
                [re setVertexBytes:&e.params length:sizeof(e.params) atIndex:5];
                [re setVertexBuffer:(e.uv != nil ? e.uv : e.pos) offset:0 atIndex:6];
                // v1.3 instance streams at 7/8; placeholder-bind e.pos when a
                // stream is unused (the shader's use_instance/_attr guards make
                // the read harmless), mirroring the idx/nrm/attr trick above.
                [re setVertexBuffer:(e.inst  != nil ? e.inst  : e.pos) offset:0 atIndex:7];
                [re setVertexBuffer:(e.iattr != nil ? e.iattr : e.pos) offset:0 atIndex:8];
                if (e.texture != nil) [re setFragmentTexture:e.texture atIndex:0];
                [re drawPrimitives:e.prim vertexStart:0 vertexCount:e.consumed
                     instanceCount:(e.use_instance ? e.n_inst : 1)];
            }
            [re endEncoding];
            [cb commit];
            last_device_path_ = "primitives-imported";
            return true;
        }
    }

    // Test-only (spec §3.4 / M1): copy a texture back on the RENDERER's own
    // queue — commit order retires every previously committed tensor op, so
    // this reads fully-updated texels without the hot path ever waiting. The
    // gfx harness passes the PUBLIC bridge id (the bridged texture pointer),
    // so resolve by pointer value against the id table; internal renderer ids
    // resolve via lookup(). NB: parameter is tex_id, never `id` (ObjC keyword).
    std::vector<uint8_t> debug_readback_rgba8(uint64_t tex_id, int w, int h) override {
        @autoreleasepool {
            id<MTLTexture> t = lookup(tex_id);
            if (t == nil) {
                for (NSNumber* key in textures_) {
                    id<MTLTexture> cand = textures_[key];
                    if ((uint64_t)(__bridge void*)cand == tex_id) { t = cand; break; }
                }
            }
            if (t == nil || w <= 0 || h <= 0) return {};
            const NSUInteger bpr = (NSUInteger)w * 4;
            id<MTLBuffer> out = [device_ newBufferWithLength:bpr * (NSUInteger)h
                                                     options:MTLResourceStorageModeShared];
            if (out == nil) return {};
            id<MTLCommandBuffer> cb = [queue_ commandBuffer];
            id<MTLBlitCommandEncoder> blit = [cb blitCommandEncoder];
            [blit copyFromTexture:t
                      sourceSlice:0
                      sourceLevel:0
                     sourceOrigin:MTLOriginMake(0, 0, 0)
                       sourceSize:MTLSizeMake((NSUInteger)w, (NSUInteger)h, 1)
                         toBuffer:out
                destinationOffset:0
           destinationBytesPerRow:bpr
         destinationBytesPerImage:bpr * (NSUInteger)h];
            [blit endEncoding];
            [cb commit];
            [cb waitUntilCompleted];   // waits live in test readbacks, not the hot path
            std::vector<uint8_t> px((size_t)w * h * 4);
            std::memcpy(px.data(), out.contents, px.size());
            return px;
        }
    }

private:
    id<MTLTexture> lookup(uint64_t id) {
        if (id == 0) return nil;
        return textures_[@(id)];
    }

    id<MTLBuffer> lookup_import(uint64_t iid) {
        auto it = imported_.find(iid);
        return it == imported_.end() ? nil : it->second;
    }

    bool ensure_pipeline() {
        if (cmap_pipeline_ != nil) return true;
        NSError* err = nil;
        id<MTLLibrary> lib =
            [device_ newLibraryWithSource:[NSString stringWithUTF8String:kColormapShaderSrc]
                                  options:nil
                                    error:&err];
        if (lib == nil) return false;
        id<MTLFunction> fn = [lib newFunctionWithName:@"cmap_f32"];
        if (fn == nil) return false;
        cmap_pipeline_ = [device_ newComputePipelineStateWithFunction:fn error:&err];
        return cmap_pipeline_ != nil;
    }

    // Lazy point pipeline (once; released at shutdown). Additive ONE/ONE blend
    // on both channels, matching the Vulkan geom pipeline: a 1-px point at a
    // pixel center lands exactly the LUT color on top of the cleared background.
    bool ensure_points_pipeline() {
        if (points_pipeline_ != nil) return true;
        NSError* err = nil;
        id<MTLLibrary> lib =
            [device_ newLibraryWithSource:[NSString stringWithUTF8String:kPointsShaderSrc]
                                  options:nil error:&err];
        if (lib == nil) return false;
        id<MTLFunction> vs = [lib newFunctionWithName:@"points_vs"];
        id<MTLFunction> fs = [lib newFunctionWithName:@"points_fs"];
        if (vs == nil || fs == nil) return false;
        MTLRenderPipelineDescriptor* d = [MTLRenderPipelineDescriptor new];
        d.vertexFunction   = vs;
        d.fragmentFunction = fs;
        d.inputPrimitiveTopology = MTLPrimitiveTopologyClassPoint;
        d.colorAttachments[0].pixelFormat         = MTLPixelFormatRGBA8Unorm;
        d.colorAttachments[0].blendingEnabled     = YES;   // additive ONE/ONE, both channels
        d.colorAttachments[0].rgbBlendOperation   = MTLBlendOperationAdd;
        d.colorAttachments[0].alphaBlendOperation = MTLBlendOperationAdd;
        d.colorAttachments[0].sourceRGBBlendFactor        = MTLBlendFactorOne;
        d.colorAttachments[0].destinationRGBBlendFactor   = MTLBlendFactorOne;
        d.colorAttachments[0].sourceAlphaBlendFactor      = MTLBlendFactorOne;
        d.colorAttachments[0].destinationAlphaBlendFactor = MTLBlendFactorOne;
        points_pipeline_ = [device_ newRenderPipelineStateWithDescriptor:d error:&err];
        return points_pipeline_ != nil;
    }

    id<MTLLibrary> geom_library() {
        if (geom_lib_ != nil) return geom_lib_;
        NSError* err = nil;
        geom_lib_ = [device_ newLibraryWithSource:[NSString stringWithUTF8String:kGeomShaderSrc]
                                          options:nil error:&err];
        if (geom_lib_ == nil) {
            const char* msg = err ? [[err localizedDescription] UTF8String] : "unknown error";
            std::fprintf(stderr, "[metal] geom_prims: shader compile failed: %s\n", msg);
        }
        return geom_lib_;
    }

    id<MTLRenderPipelineState> geom_pipeline(uint32_t cls, uint32_t blend,
                                             bool has_depth, bool textured) {
        const uint32_t key = cls | (blend << 2) |
            (has_depth ? (1u << 4) : 0u) | (textured ? (1u << 5) : 0u);
        auto hit = geom_pipelines_.find(key);
        if (hit != geom_pipelines_.end()) return hit->second;

        id<MTLLibrary> lib = geom_library();
        if (lib == nil) return nil;
        // cls==0 && textured pairs geom_vs_point (VOutPoint) with geom_tex_fs
        // (VOut stage_in): valid — builtin members are excluded from stage-in
        // matching and the user varyings agree, so textured POINT draws render
        // (hardware-verified gfx row "textured POINT draw renders the sampled
        // texel"). Any genuinely bad pairing still fails closed: nil -> the
        // caller refuses the whole frame before any encoding.
        id<MTLFunction> vs = [lib newFunctionWithName:
            (cls == 0 ? @"geom_vs_point" : @"geom_vs")];
        id<MTLFunction> fs = [lib newFunctionWithName:
            (textured ? @"geom_tex_fs" : @"geom_fs")];
        if (vs == nil || fs == nil) {
            std::fprintf(stderr, "[metal] geom_prims: missing geom shader function\n");
            return nil;
        }

        MTLRenderPipelineDescriptor* d = [MTLRenderPipelineDescriptor new];
        d.vertexFunction = vs;
        d.fragmentFunction = fs;
        d.colorAttachments[0].pixelFormat = MTLPixelFormatRGBA8Unorm;
        d.depthAttachmentPixelFormat = has_depth ? MTLPixelFormatDepth32Float
                                                 : MTLPixelFormatInvalid;
        switch (cls) {
            case 0: d.inputPrimitiveTopology = MTLPrimitiveTopologyClassPoint; break;
            case 1: d.inputPrimitiveTopology = MTLPrimitiveTopologyClassLine; break;
            default: d.inputPrimitiveTopology = MTLPrimitiveTopologyClassTriangle; break;
        }

        if (blend == CALIPER_GEOM_BLEND_ALPHA) {
            d.colorAttachments[0].blendingEnabled = YES;
            d.colorAttachments[0].rgbBlendOperation = MTLBlendOperationAdd;
            d.colorAttachments[0].alphaBlendOperation = MTLBlendOperationAdd;
            d.colorAttachments[0].sourceRGBBlendFactor = MTLBlendFactorSourceAlpha;
            d.colorAttachments[0].destinationRGBBlendFactor = MTLBlendFactorOneMinusSourceAlpha;
            d.colorAttachments[0].sourceAlphaBlendFactor = MTLBlendFactorOne;
            d.colorAttachments[0].destinationAlphaBlendFactor = MTLBlendFactorOneMinusSourceAlpha;
        } else if (blend == CALIPER_GEOM_BLEND_ADDITIVE) {
            d.colorAttachments[0].blendingEnabled = YES;
            d.colorAttachments[0].rgbBlendOperation = MTLBlendOperationAdd;
            d.colorAttachments[0].alphaBlendOperation = MTLBlendOperationAdd;
            d.colorAttachments[0].sourceRGBBlendFactor = MTLBlendFactorOne;
            d.colorAttachments[0].destinationRGBBlendFactor = MTLBlendFactorOne;
            d.colorAttachments[0].sourceAlphaBlendFactor = MTLBlendFactorOne;
            d.colorAttachments[0].destinationAlphaBlendFactor = MTLBlendFactorOne;
        }

        NSError* err = nil;
        id<MTLRenderPipelineState> p = [device_ newRenderPipelineStateWithDescriptor:d error:&err];
        if (p == nil) {
            const char* msg = err ? [[err localizedDescription] UTF8String] : "unknown error";
            std::fprintf(stderr, "[metal] geom_prims: render pipeline failed: %s\n", msg);
        }
        if (p != nil) geom_pipelines_[key] = p;
        return p;
    }

    id<MTLDepthStencilState> geom_depth_state(uint32_t flags) {
        const uint32_t key = flags & (CALIPER_GEOM_DEPTH_TEST | CALIPER_GEOM_DEPTH_WRITE);
        auto hit = depth_states_.find(key);
        if (hit != depth_states_.end()) return hit->second;
        MTLDepthStencilDescriptor* d = [MTLDepthStencilDescriptor new];
        d.depthCompareFunction = (key & CALIPER_GEOM_DEPTH_TEST)
            ? MTLCompareFunctionLessEqual : MTLCompareFunctionAlways;
        d.depthWriteEnabled = (key & CALIPER_GEOM_DEPTH_WRITE) != 0u;
        id<MTLDepthStencilState> s = [device_ newDepthStencilStateWithDescriptor:d];
        if (s != nil) depth_states_[key] = s;
        return s;
    }

    // M2b (spec §4): GPU-order this texture's update after the producer
    // queue's already-committed work. A tiny command buffer on the PRODUCER's
    // queue signals value v (queue order puts it after the producer's
    // committed kernels); the tensor-op command buffer waits v before any
    // encoder runs. No CPU block. If the event can't be created, fall back to
    // a CPU wait on the producer queue — slower, never silently unordered.
    void order_after_producer(uint64_t tex, id<MTLCommandBuffer> cb, void* stream) {
        id<MTLCommandQueue> producer = (__bridge id<MTLCommandQueue>)stream;
        if (producer == nil) return;
        id<MTLSharedEvent> ev = events_[@(tex)];
        if (ev == nil) {
            ev = [device_ newSharedEvent];
            if (ev != nil) events_[@(tex)] = ev;
        }
        id<MTLCommandBuffer> sig = [producer commandBuffer];
        if (ev != nil && sig != nil) {
            const uint64_t v = ++event_values_[tex];
            [sig encodeSignalEvent:ev value:v];
            [sig commit];
            [cb encodeWaitForEvent:ev value:v];
        } else if (sig != nil) {
            [sig commit];
            [sig waitUntilCompleted];   // rare fallback: CPU-ordered, still correct
        }
    }

    // f32 + LUT -> runtime-compiled compute shader. Sources from src at
    // src_offset bytes (0 on the direct path). Records "compute" (direct) or
    // "compute-imported" (v1.2 imported path).
    bool colormap_compute_from(uint64_t tex, id<MTLTexture> dst, id<MTLBuffer> src,
                               uint64_t src_offset, const CaliperTensor& t,
                               const uint32_t* lut256, float vmin, float vmax,
                               bool imported) {
        if (!ensure_pipeline()) return false;
        if (tensor_extent_bytes(t, sizeof(float)) > src.length - src_offset)
            return false;   // buffer too short for the declared extent at offset

        CmapParams p{};
        p.w = (uint32_t)dst.width;
        p.h = (uint32_t)dst.height;
        // Strides are in elements (§7.2). Fall back to row-major contiguous.
        p.sx = (t.ndim >= 1) ? (uint32_t)t.strides[t.ndim - 1] : 1u;
        p.sy = (t.ndim >= 2) ? (uint32_t)t.strides[t.ndim - 2] : p.w;
        p.vmin = vmin;
        p.vmax = vmax;

        id<MTLBuffer> lutbuf = [device_ newBufferWithBytes:lut256
                                                    length:256 * sizeof(uint32_t)
                                                   options:MTLResourceStorageModeShared];
        if (lutbuf == nil) return false;

        id<MTLCommandBuffer> cb = [queue_ commandBuffer];
        if (t.stream != nullptr) order_after_producer(tex, cb, t.stream);
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        [enc setComputePipelineState:cmap_pipeline_];
        [enc setBuffer:src offset:(NSUInteger)src_offset atIndex:0];
        [enc setBuffer:lutbuf offset:0 atIndex:1];
        [enc setBytes:&p length:sizeof(p) atIndex:2];
        [enc setTexture:dst atIndex:0];

        MTLSize tg = MTLSizeMake(16, 16, 1);
        MTLSize groups = MTLSizeMake((p.w + 15) / 16, (p.h + 15) / 16, 1);
        [enc dispatchThreadgroups:groups threadsPerThreadgroup:tg];
        [enc endEncoding];
        // No CPU wait (M1/D23): the frame's command buffer commits AFTER this
        // one on the same queue_, so draw ordering is free by commit order; cb
        // retains src, lutbuf, and dst until the GPU retires it (default,
        // non-`unretained` encoding), so lifetime is free too. Test readbacks
        // retire the queue themselves (debug_readback_rgba8).
        [cb commit];

        last_device_path_ = imported ? "compute-imported" : "compute";
        return true;
    }

    // u8 HWC (RGBA8) -> blit straight into the texture. Sources from src at
    // src_offset bytes (0 on the direct path). Records "blit" (direct) or
    // "blit-imported" (v1.2 imported path).
    bool blit_u8_from(uint64_t tex, id<MTLTexture> dst, id<MTLBuffer> src,
                      uint64_t src_offset, const CaliperTensor& t, bool imported) {
        if (t.ndim < 2) return false;
        NSUInteger h = (NSUInteger)t.shape[0];
        NSUInteger w = (NSUInteger)t.shape[1];
        NSUInteger c = (t.ndim >= 3) ? (NSUInteger)t.shape[2] : 1;
        if (c != 4 || w != dst.width || h != dst.height) return false;  // RGBA8 only
        if (tensor_extent_bytes(t, 1) > src.length - src_offset)
            return false;   // buffer too short for the declared extent at offset

        NSUInteger bytesPerRow = w * 4;
        id<MTLCommandBuffer> cb = [queue_ commandBuffer];
        if (t.stream != nullptr) order_after_producer(tex, cb, t.stream);
        id<MTLBlitCommandEncoder> blit = [cb blitCommandEncoder];
        [blit copyFromBuffer:src
                sourceOffset:(NSUInteger)src_offset
           sourceBytesPerRow:bytesPerRow
         sourceBytesPerImage:bytesPerRow * h
                  sourceSize:MTLSizeMake(w, h, 1)
                   toTexture:dst
            destinationSlice:0
            destinationLevel:0
           destinationOrigin:MTLOriginMake(0, 0, 0)];
        [blit endEncoding];
        [cb commit];   // no CPU wait (M1/D23): same-queue commit order + retention

        last_device_path_ = imported ? "blit-imported" : "blit";
        return true;
    }

    GLFWwindow*                 window_ = nullptr;
    id<MTLDevice>               device_ = nil;
    id<MTLCommandQueue>         queue_  = nil;
    CAMetalLayer*               layer_  = nil;
    MTLRenderPassDescriptor*    pass_desc_ = nil;
    id<MTLComputePipelineState> cmap_pipeline_ = nil;
    id<MTLRenderPipelineState>  points_pipeline_ = nil;  // caliper.geometry.v1 point pass
    id<MTLLibrary>              geom_lib_ = nil;
    std::unordered_map<uint32_t, id<MTLRenderPipelineState>> geom_pipelines_;
    std::unordered_map<uint32_t, id<MTLDepthStencilState>> depth_states_;

    // Per-frame transients created in new_frame(), consumed in render().
    id<CAMetalDrawable>          drawable_  = nil;
    id<MTLCommandBuffer>         frame_cmd_ = nil;
    id<MTLRenderCommandEncoder>  frame_enc_ = nil;

    NSMutableDictionary<NSNumber*, id<MTLTexture>>* textures_ = nil;
    std::unordered_map<uint64_t, id<MTLTexture>> depth_textures_;

    // M2b: per-texture producer-ordering events (D23 — MTLSharedEvent appears
    // ONLY where cross-queue ordering genuinely exists). Values are a per-
    // texture monotonic timeline, the Metal analog of Vulkan's semaphores.
    NSMutableDictionary<NSNumber*, id<MTLSharedEvent>>* events_ = nil;
    std::unordered_map<uint64_t, uint64_t> event_values_;

    // v1.2 imported allocations: in-process MTLBuffers, strong-retained (ARC) —
    // the Metal analog of Vulkan's DuplicateHandle+VkImportMemory. 0 invalid.
    std::unordered_map<uint64_t, id<MTLBuffer>> imported_;
    uint64_t next_import_id_ = 1;

    // Grow-only host-visible G14 rigidity-readback staging (§5.1): one shared
    // buffer, reused across frames, blit-target for private-storage instance
    // matrices. The Metal analog of Vulkan's geom_prim_inst_staging_.
    id<MTLBuffer> geom_prim_inst_staging_ = nil;

    uint64_t next_id_ = 1;          // 0 is the invalid id
    const char* last_device_path_ = "";
};

}  // namespace

std::unique_ptr<HostRenderer> make_metal_renderer() {
    return std::make_unique<MetalRenderer>();
}

}  // namespace caliper_host
