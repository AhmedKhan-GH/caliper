#pragma once
// TensorBridge — host-side core of caliper.tensor_bridge.v1 (docs/references/service-tensor-bridge-v1.md).
// Pure logic over the graphics layer (src/host/gfx): it validates a CaliperTensor
// against the frozen v1 acceptance rules and puts it into a layer texture with
// the layer's update_texture_from_tensor, which runs the colormap by keys on the
// GPU for f32 data and fills RGBA8 texels for u8 data, reading a device tensor
// or an imported allocation in place and copying host bytes once; and it keeps
// the id -> layer-handle bookkeeping (§5.4). Its imported allocations are layer
// buffers (import_buffer), which the geometry service's draws on the Metal
// renderer read through an adapter until the geometry service goes, so there is
// one allocation table. It never links torch and never names a graphics API.
// Where the renderer runs no backend of the layer (HostRenderer::gfx() is null:
// the Vulkan renderer until Windows parity), every texture, import and view is
// refused with that reason.
//
// This lives in the exe / gfx-test link scope, NOT caliper_host_lib: it depends
// on the HostRenderer interface and the layer's header only, so unit tests drive
// it with a stub renderer on the layer's recording backend and no window.
#include <caliper/tensor.h>
#include <caliper/services/tensor_bridge_v1.h>
#include <caliper/services/tensor_bridge_v1_1.h>
#include <caliper/services/tensor_bridge_v1_2.h>
#include <caliper/services/geometry_v1.h>
#include <caliper/services/geometry_v1_1.h>
#include <caliper/services/geometry_v1_2.h>
#include <caliper/services/geometry_v1_3.h>
#include <caliper/services/device_v1.h>

#include "gfx/gfx.h"

#include <cstdint>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace caliper_gfx::cmap {
class KeyBlockCache;
}

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
// with vmax==vmin -> t=0, and NaN -> index 0 (never a misinterpreted texel);
// a NaN t, from an infinite end or span, is index 0 as well.
void map_f32_to_rgba8(const float* src, int w, int h,
                      const uint32_t* lut256, float vmin, float vmax,
                      uint8_t* dst);

// The thresholds that give map_f32_to_rgba8's index with exact float
// comparisons alone, for GPU shaders whose division is not exact (WGSL allows
// 2.5 ULP of error; Metal's default fast math too): for k in 1..255, T[k] is
// the first float whose index is k or more for an increasing range (mode 1),
// the last one for a decreasing range (mode 2), and the index of v is the
// number of thresholds it has passed. Mode 0 is a range whose width is zero,
// infinite or NaN, where the reference gives every value index 0. NaN is
// index 0 in every mode. Writes T[0..255] (T[0] unused) and returns the mode.
// web/src/render/colormap.ts (colormapThresholds) derives the same.
uint32_t colormap_thresholds(float vmin, float vmax, float* T);

// Expand a row-major (h,w,c) u8 buffer (c in 1..4) into tightly-packed RGBA8:
//   c==1 -> gray replicated to RGB, a=255;  c==3 -> RGB, a=255;  c==4 -> copy.
void expand_u8_to_rgba8(const uint8_t* src, int w, int h, int c, uint8_t* dst);

// Host-internal diagnostics (embed v1.2 views): the text of the most recent
// bridge rejection on THIS thread ("" if none since the last clear), so the
// embed core can put the bridge's own reason into caliper_core_last_error
// instead of a generic "refused". The log sink still receives every line.
const char* bridge_last_reject();
void        bridge_clear_last_reject();

class TensorBridge {
public:
    explicit TensorBridge(HostRenderer& renderer);
    ~TensorBridge();   // gives every texture, view and import it holds back to the layer

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

    // Bridge-v1.1 capability bits, from the layer's caps. Bit 0 = an update
    // honors a tensor's stream by ordering its read after the producer's queue
    // on the GPU, so adapters may skip the device drain. Bit 1 (v1.2) = the
    // layer can import an applet-exported device allocation in place. 0 with
    // no backend of the layer.
    uint32_t caps() const;
    // The device kind whose memory the layer reads in place (METAL on Metal,
    // CUDA with an exportable-memory import, else CPU): a device tensor of any
    // other kind is refused.
    CaliperDeviceKind active_device() const;

    // caliper.tensor_bridge.v1.2 ops (imported external allocations). The
    // layer imports the applet's buffer in place (import_buffer, with no
    // copy) and updates read it at a byte offset. All return 0/false/no-op
    // when the layer cannot import or on a rule breach (reason logged), and
    // update_texture_from_alloc reuses the SAME frozen acceptance gates as
    // update_texture plus a host-side bounds check.
    CaliperAllocId import_allocation(void* os_handle, uint64_t size_bytes,
                                     uint32_t handle_type);
    void           release_allocation(CaliperAllocId a);
    bool           update_texture_from_alloc(CaliperTextureId tex,
                                             CaliperAllocId a,
                                             uint64_t offset_bytes,
                                             const CaliperTensor* desc);

    // Host-internal (NOT on any frozen C table — the embed core's built-in
    // tensor views use it): re-pin a MAPPED texture's colormap/vmin/vmax so the
    // NEXT update_texture / update_texture_from_alloc maps with the new values,
    // without recreating the texture. Existing texels are not re-rendered.
    // false (nothing changed): unknown id, a direct-u8/geometry texture, or an
    // out-of-range colormap.
    bool           set_mapping(CaliperTextureId tex, int32_t colormap,
                               float vmin, float vmax);

    // Host-internal: the path label of the last successful update, as the layer
    // returned it with its route ("compute", "compute-imported", "blit",
    // "blit-imported" or "cpu-staged"). The embed views report it.
    const char*    last_path() const { return last_path_; }
    // Host-internal: completion of the updates, by the layer's serials. The
    // serial of the last successful update's GPU read of its source (0 when it
    // needed none: host texels the layer wrote at once), whether a serial has
    // completed (never blocks), and a blocking wait for one (embed views'
    // CALIPER_VIEW_FLAG_WAIT_GPU, tests).
    uint64_t       last_serial() const { return last_serial_; }
    bool           completed(uint64_t serial) const;
    void           wait(uint64_t serial);
    // Host-internal: a texture's or a geometry view's texels, tightly packed
    // RGBA8 rows, top row first, read after every update and draw made before;
    // false for an unknown id or a failed readback.
    bool           readback(CaliperTextureId tex, std::vector<uint8_t>* rgba);
    // Host-internal: the layer's texture behind a texture or view id (none for
    // an unknown id), and how many textures and views the bridge holds.
    caliper_gfx::Texture texture(CaliperTextureId tex) const;
    size_t         texture_count() const { return entries_.size(); }

    // caliper.geometry.v1 (imported 3-D points). Lives on the SAME object as
    // the bridge so views share the texture table (ImGui-drawable ids) and
    // points address v1.2 imported allocations directly. A view is a layer
    // texture (and a depth texture with CALIPER_GEOM_VIEW_DEPTH) that the Metal
    // renderer's geometry draws render into through the layer's adapter. Every
    // gate fails closed with a logged reason; a false draw leaves the view's
    // pixels and telemetry untouched.
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
    // the layer's ImGui handle for the texture (Backend::imgui_texture_id — the
    // id<MTLTexture> imgui_impl_metal binds, the view imgui_impl_wgpu binds;
    // §5.4), and this table is keyed by it. Entry adds the shape/dtype/colormap
    // needed to re-upload on update_texture.
    struct Entry {
        caliper_gfx::Texture tex;   // the layer's texture
        caliper_gfx::Texture depth; // a geometry view's depth target, or none
        int          w = 0;
        int          h = 0;
        CaliperDType dtype = CALIPER_DT_F32;
        int          channels = 1;  // direct-u8 source channel count
        bool         mapped = false;// f32 -> colormap
        int32_t      colormap = 0;
        float        vmin = 0.0f;
        float        vmax = 1.0f;
        bool         shared = false;
        std::vector<uint8_t> shared_buf;  // alloc_shared CPU-unified backing
        bool         view = false;        // geometry.v1 render target: updates
                                          // and bridge-release refuse it
        bool         view_depth = false;  // geometry.v1_1 depth attachment
    };

    // The layer's backend, or null (with the reason logged) where the renderer
    // runs none.
    caliper_gfx::Backend* gfx(const char* what) const;
    // A new layer texture for a tensor, registered under its ImGui id; 0 when
    // the layer refuses it.
    CaliperTextureId add_texture(Entry e, const char* what);
    // A geometry view's layer textures, cleared; 0 when the layer refuses them.
    CaliperTextureId make_view(uint32_t w, uint32_t h, bool depth, const char* what);
    void drop(Entry& e);

    // Put a validated tensor (or, with alloc, an imported allocation's window
    // described by t) into an existing entry's texture through the layer.
    bool upload_into(Entry& e, const CaliperTensor* t, caliper_gfx::Buffer alloc = {},
                     uint64_t offset_bytes = 0);
    const char* last_path_ = "cpu-staged";
    uint64_t    last_serial_ = 0;

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
    // layer's buffer + its byte size (for the host-side bounds check).
    struct ImportedAlloc {
        caliper_gfx::Buffer buffer;
        uint64_t size_bytes = 0;
    };

    HostRenderer&                          renderer_;
    std::unordered_map<uint64_t, Entry>    entries_;
    std::unordered_map<uint64_t, ImportedAlloc> imported_;   // alloc id -> entry
    uint64_t                               next_alloc_id_ = 1;
    // The key blocks of the colormaps and ranges the textures map with.
    std::unique_ptr<caliper_gfx::cmap::KeyBlockCache> keys_;
};

}  // namespace caliper_host
