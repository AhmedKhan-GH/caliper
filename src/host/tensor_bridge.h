#pragma once
// TensorBridge — host-side core of caliper.tensor_bridge.v1 (docs/instruments/services/caliper-tensor-bridge-v1.md).
// Pure logic over the HostRenderer texture seam: it validates a CaliperTensor
// against the frozen v1 acceptance rules, colormaps/expands it into RGBA8 on the
// CPU staging path, or forwards a device tensor to the backend's device path —
// and keeps the id -> backend-handle bookkeeping (§5.4). It never links torch
// and never names a graphics API; the renderer stays swappable.
//
// This lives in the exe / gfx-test link scope, NOT caliper_host_lib: it depends
// on the HostRenderer interface only (header), so unit tests drive it with a
// stub renderer and no window.
#include "gfx/colormap_ref.h"
#include <caliper/tensor.h>
#include <caliper/services/tensor_bridge_v1.h>
#include <caliper/services/tensor_bridge_v1_1.h>
#include <caliper/services/tensor_bridge_v1_2.h>
#include <caliper/services/device_v1.h>

#include <cstdint>
#include <unordered_map>
#include <vector>

namespace caliper_host {

class HostRenderer;

// ---- Built-in 256-entry RGBA8 colormap LUTs & CPU reference conversions ----
// Declared in gfx/colormap_ref.h (included above) so gfx headers depend
// downward on gfx/ rather than pulling tensor_bridge.h back up.

// Host-internal diagnostics (embed v1.2 views): the text of the most recent
// bridge rejection on THIS thread ("" if none since the last clear), so the
// embed core can put the bridge's own reason into caliper_core_last_error
// instead of a generic "refused". The log sink still receives every line.
const char* bridge_last_reject();
void        bridge_clear_last_reject();

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

    // Bridge-v1.1 capability bits. Bit 0 = the active renderer honors
    // stream-ordered handoff, so adapters may skip the device drain. Bit 1
    // (v1.2) = the renderer can import an instrument-exported device allocation.
    uint32_t caps() const;

    // caliper.tensor_bridge.v1.2 ops (imported external allocations). The host
    // dups the instrument's OS shareable handle into a renderer-internal id and
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

    // Host-internal (NOT on any frozen C table — the embed core's built-in
    // tensor views use it): re-pin a MAPPED texture's colormap/vmin/vmax so the
    // NEXT update_texture / update_texture_from_alloc maps with the new values,
    // without recreating the texture. Existing texels are not re-rendered.
    // false (nothing changed): unknown id, a direct-u8 texture, or an
    // out-of-range colormap.
    bool           set_mapping(CaliperTextureId tex, int32_t colormap,
                               float vmin, float vmax);

    // Host-internal: how the last host-memory (CPU) upload reached its texture,
    // "cpu-staged" (colormapped or expanded on the CPU, then uploaded) or the
    // renderer's last_device_path() when the renderer mapped the host bytes on
    // its GPU (tex_update_mapped_from_host). The embed views report it.
    const char*    last_upload_path() const { return last_upload_path_; }


private:
    // Per-texture bookkeeping. The public CaliperTextureId handed to callers is
    // the renderer's ImGui handle (tex_imtexture_id — what imgui_impl_metal or
    // imgui_impl_vulkan binds as ImTextureID; §5.4), and this table is keyed by it. The
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
    };

    // Stage/forward a validated tensor into an existing entry's texture.
    bool upload_into(Entry& e, const CaliperTensor* t);
    const char* last_upload_path_ = "cpu-staged";

    // dtype + ndim + per-dim shape match of an update/alloc desc against an
    // existing entry (mapped: 2D h×w; direct: 3D h×w×channels). Shared by
    // update_texture and update_texture_from_alloc so the shape gate is
    // written once; logs "update: ..." and returns false on a mismatch.
    bool desc_matches_entry(const Entry& e, const CaliperTensor& t) const;

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
