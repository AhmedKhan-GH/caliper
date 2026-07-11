#pragma once
// TensorBridge — host-side core of caliper.tensor_bridge.v1 (PLATFORM.md §7.4).
// Pure logic over the HostRenderer texture seam: it validates a CaliperTensor
// against the frozen v1 acceptance rules, colormaps/expands it into RGBA8 on the
// CPU staging path, or forwards a device tensor to the backend's device path —
// and keeps the id -> backend-handle bookkeeping (§5.4). It never links torch
// (D11) and never names a graphics API; the renderer stays swappable.
//
// This lives in the exe / gfx-test link scope, NOT caliper_host_lib: it depends
// on the HostRenderer interface only (header), so unit tests drive it with a
// stub renderer and no window.
#include <caliper/tensor.h>
#include <caliper/services/tensor_bridge_v1.h>
#include <caliper/services/tensor_bridge_v1_1.h>
#include <caliper/services/tensor_bridge_v1_2.h>
#include <caliper/services/geometry_v1.h>
#include <caliper/services/geometry_v1_1.h>
#include <caliper/services/geometry_v1_2.h>
#include <caliper/services/geometry_v1_3.h>
#include <caliper/services/device_v1.h>

#include <cstdint>
#include <unordered_map>
#include <vector>

namespace caliper_host {

class HostRenderer;

// ---- Built-in 256-entry RGBA8 colormap LUTs -------------------------------
// Packed little-endian as r | g<<8 | b<<16 | a<<24 — the exact byte layout the
// CPU staging path writes and the Metal compute path uploads, so both backends
// emit identical bytes (Phase-2C colormaps v1). Returned by value-stable
// pointer; nullptr for an out-of-range colormap id.
const uint32_t* colormap_lut(int32_t colormap);

// ---- CPU reference conversions (single source of truth for §16) -----------
// Colormap a row-major (h,w) f32 buffer through lut256 into a tightly-packed
// RGBA8 dst (w*h*4 bytes). Index rule, byte-identical to the Metal shader:
//   idx = clamp((v - vmin)/(vmax - vmin), 0, 1) * 255 + 0.5   (truncated)
// with vmax==vmin -> t=0, and NaN -> index 0 (never a misinterpreted texel).
void map_f32_to_rgba8(const float* src, int w, int h,
                      const uint32_t* lut256, float vmin, float vmax,
                      uint8_t* dst);

// Expand a row-major (h,w,c) u8 buffer (c in 1..4) into tightly-packed RGBA8:
//   c==1 -> gray replicated to RGB, a=255;  c==3 -> RGB, a=255;  c==4 -> copy.
void expand_u8_to_rgba8(const uint8_t* src, int w, int h, int c, uint8_t* dst);

class TensorBridge {
public:
    explicit TensorBridge(HostRenderer& renderer);

    // caliper.tensor_bridge.v1 ops. Returns 0/false on an acceptance-rule
    // violation (reason emitted via log; never a misinterpreted texture).
    CaliperTextureId texture_from_tensor(const CaliperTensor* t, uint32_t flags);
    bool             update_texture(CaliperTextureId tex, const CaliperTensor* t);
    void             release_texture(CaliperTextureId tex);
    CaliperTextureId texture_from_tensor_mapped(const CaliperTensor* t,
                                                int32_t colormap,
                                                float vmin, float vmax,
                                                uint32_t flags);
    bool             alloc_shared(CaliperDType dtype, int32_t ndim,
                                  const int64_t* shape,
                                  CaliperTensor* out_tensor,
                                  CaliperTextureId* out_texture);
    void             free_shared(CaliperTextureId tex);

    // Bridge-v1.1 capability bits (D24). Bit 0 = the active renderer honors
    // stream-ordered handoff, so adapters may skip the device drain. Bit 1
    // (v1.2) = the renderer can import an applet-exported device allocation.
    uint32_t caps() const;

    // caliper.tensor_bridge.v1.2 ops (imported external allocations). The host
    // dups the applet's OS shareable handle into a renderer-internal id and
    // runs device texture updates FROM the imported bytes — zero copies. All
    // return 0/false/no-op when the renderer can't import or on a rule breach
    // (reason logged), and update_texture_from_alloc reuses the SAME frozen
    // acceptance gates as update_texture plus a host-side bounds check.
    CaliperAllocId import_allocation(void* os_handle, uint64_t size_bytes,
                                     uint32_t handle_type);
    void           release_allocation(CaliperAllocId a);
    bool           update_texture_from_alloc(CaliperTextureId tex,
                                             CaliperAllocId a,
                                             uint64_t offset_bytes,
                                             const CaliperTensor* desc);

    // caliper.geometry.v1 (imported 3-D points). Lives on the SAME object as
    // the bridge so views share the texture table (ImGui-drawable ids) and
    // points address v1.2 imported allocations directly. Every gate fails
    // closed with a logged reason; a false draw leaves the view's pixels and
    // telemetry untouched.
    uint32_t         geom_caps() const;
    CaliperTextureId geom_create_view(uint32_t w, uint32_t h);
    CaliperTextureId geom_create_view_ex(uint32_t w, uint32_t h,
                                         uint32_t flags);
    void             geom_release_view(CaliperTextureId view);
    bool             geom_draw_points(CaliperTextureId view,
                                      const CaliperGeomCamera* cam,
                                      CaliperAllocId pos_alloc,
                                      uint64_t pos_offset, uint64_t count,
                                      CaliperAllocId attr_alloc,
                                      uint64_t attr_offset,
                                      int32_t colormap, float vmin, float vmax,
                                      float size_px, uint32_t clear_rgba);
    bool             geom_draw_primitives(CaliperTextureId view,
                                          const CaliperGeomCamera* cam,
                                          const CaliperGeomDraw* draws,
                                          uint32_t draw_count,
                                          uint32_t draw_stride,
                                          uint32_t clear_rgba);
    bool             geom_draw_primitives_v1_2(CaliperTextureId view,
                                          const CaliperGeomCamera* cam,
                                          const CaliperGeomDrawV1_2* draws,
                                          uint32_t draw_count,
                                          uint32_t draw_stride,
                                          uint32_t clear_rgba);
    bool             geom_draw_primitives_v1_3(CaliperTextureId view,
                                          const CaliperGeomCamera* cam,
                                          const CaliperGeomDrawV1_3* draws,
                                          uint32_t draw_count,
                                          uint32_t draw_stride,
                                          uint32_t clear_rgba);

private:
    // The revision axis for the shared draw_primitives validator: selects the
    // minimum stride (192/216/256), the color-mode ceiling, and whether the
    // instance tail is read. Replaced the earlier bool v12 (single-axis) — a
    // second bool was the reviewed defect, do not reintroduce it.
    enum class GeomRev : uint32_t { V1_1, V1_2, V1_3 };
    // Per-texture bookkeeping. The public CaliperTextureId handed to callers is
    // the renderer's ImGui handle (tex_imtexture_id — what imgui_impl_{metal,
    // opengl3} bind as ImTextureID; §5.4), and this table is keyed by it. The
    // internal renderer id (`tex` below) drives upload/release and never leaves
    // the host. Entry adds the shape/dtype/colormap needed to re-upload on
    // update_texture.
    struct Entry {
        uint64_t     tex = 0;       // renderer texture id (internal, for up/release)
        int          w = 0;
        int          h = 0;
        CaliperDType dtype = CALIPER_DT_F32;
        int          channels = 1;  // direct-u8 source channel count
        bool         mapped = false;// f32 -> colormap LUT
        int32_t      colormap = 0;
        float        vmin = 0.0f;
        float        vmax = 1.0f;
        bool         shared = false;
        std::vector<uint8_t> shared_buf;  // alloc_shared CPU-unified backing
        bool         view = false;        // geometry.v1 render target: updates
                                          // and bridge-release refuse it
        bool         view_depth = false;  // geometry.v1_1 depth attachment
    };

    // Stage/forward a validated tensor into an existing entry's texture.
    bool upload_into(Entry& e, const CaliperTensor* t);

    // dtype + ndim + per-dim shape match of an update/alloc desc against an
    // existing entry (mapped: 2D h×w; direct: 3D h×w×channels). Shared by
    // update_texture and update_texture_from_alloc so the shape gate is
    // written once; logs "update: ..." and returns false on a mismatch.
    bool desc_matches_entry(const Entry& e, const CaliperTensor& t) const;

    // Shared validation for all draw_primitives entry points. `rev` is the
    // single revision axis: it selects the minimum stride (192/216/256), the
    // color-mode ceiling (VERTEX_RGBA vs TEXTURE), and whether the instance
    // tail is read — the records are otherwise validated identically.
    bool geom_draw_primitives_impl(CaliperTextureId view,
                                   const CaliperGeomCamera* cam,
                                   const void* draws, uint32_t draw_count,
                                   uint32_t draw_stride, GeomRev rev,
                                   uint32_t clear_rgba);

    // Imported external allocations (v1.2): public CaliperAllocId -> the
    // renderer-internal id + its byte size (for the host-side bounds check).
    struct ImportedAlloc {
        uint64_t renderer_id = 0;
        uint64_t size_bytes  = 0;
    };

    HostRenderer&                          renderer_;
    CaliperDeviceKind                      active_device_;  // backend's device
    std::unordered_map<uint64_t, Entry>    entries_;
    std::unordered_map<uint64_t, ImportedAlloc> imported_;   // alloc id -> entry
    uint64_t                               next_alloc_id_ = 1;
};

}  // namespace caliper_host
