# Design — `caliper.geometry.v1` + flow_scope (imported 3-D geometry)

**Approved:** 2026-07-07 (brainstormed; approach 1 of 3; user chose demo=flow-field
particles, interaction level=touchable). **Branch:** `feat/flowscope`.

## Goal

Prove the digital-twin primitive on top of bridge v1.2: a GPU simulation's
state, born in the exportable pool, is drawn as **millions of instanced 3-D
points directly from simulation memory** — zero copies of the data, per
frame — inside a Caliper applet. Platform increment first (a new frozen
service), exemplar applet second, hardware verification third.

## Non-goals (v1)

- Meshes, lines, indexed geometry — increment two, same machinery.
- Metal/GL implementations of the service (ABI stays graphics-API-neutral;
  hosts without support simply don't vend the capability).
- Pipelined (semaphore-only) draw submission — v1 draws with a fenced
  submit like the v1.2 sync fallback; pipelining is a later perf increment.
- Multi-draw scenes / draw lists — one `draw_points` call renders one view
  frame (clear + draw), atomically.

## 1. ABI — `sdk/include/caliper/services/geometry_v1.h`

New frozen service `caliper.geometry.v1`, additive forever after; NOT a
tensor_bridge revision (the bridge's identity — a tensor becomes an image —
stays clean). It shares two id spaces with the bridge on purpose:

- `CaliperTextureId`: `create_view` returns a texture id in the SAME table
  the bridge uses, so a view is drawable with `ImGui::Image` and readable by
  `debug_readback` with zero new plumbing.
- `CaliperAllocId`: positions/attributes are addressed as
  (imported allocation, byte offset) — the v1.2 import machinery, cache,
  gates, and lifecycle are reused as-is.

```c
#define CALIPER_GEOMETRY_V1 "caliper.geometry.v1"
#define CALIPER_GEOM_CAP_IMPORTED_POINTS (1u << 0)

typedef struct CaliperGeomCamera {   /* column-major, applet-owned math */
    float view[16];
    float proj[16];
} CaliperGeomCamera;

typedef struct CaliperGeometryV1 {
    uint32_t struct_size;
    uint32_t (*caps)(void);
    CaliperTextureId (*create_view)(uint32_t width, uint32_t height);
    void (*release_view)(CaliperTextureId view);
    /* Render one frame of `view`: clear to clear_rgba, then draw `count`
       points whose positions are (count,3) f32 contiguous at
       pos_alloc+pos_offset. attr_alloc != 0 selects a (count,) f32 scalar
       at attr_offset, colormapped through the tensor-bridge LUTs over
       [vmin,vmax]; attr_alloc == 0 draws flat white. size_px = point size.
       Additive blending, no depth (v1 — built for particle clouds).
       Gates mirror update_texture_from_alloc: live allocations only,
       4-byte-aligned offsets, overflow-safe bounds
       (offset > size || bytes > size - offset). false = view unchanged. */
    bool (*draw_points)(CaliperTextureId view,
                        const CaliperGeomCamera* cam,
                        CaliperAllocId pos_alloc, uint64_t pos_offset,
                        uint64_t count,
                        CaliperAllocId attr_alloc, uint64_t attr_offset,
                        int32_t colormap, float vmin, float vmax,
                        float size_px, uint32_t clear_rgba);
} CaliperGeometryV1;
```

Alignment note: positions are read by vertex pulling with a push-constant
**element base** (buffer bound whole), so the gate is 4-byte alignment —
deliberately looser than v1.2's `minStorageBufferOffsetAlignment` descriptor
offsets. Torch pool tensors (512-aligned) trivially pass.

C++ wrapper `caliper::Geometry` in `caliper.hpp`, null-guarded like Bridge
(D24: inert on hosts without the service).

## 2. Host bookkeeping

Implemented inside the same host object as the TensorBridge (one backing
object, two vended tables) so `CaliperAllocId` resolution is trivially
coherent. Views live in the texture id table with a `kind=view` flag;
`release_view` refuses ids that aren't views. All gates fail closed with a
`caliper.log.v1` line; `draw_points` on any gate failure leaves pixels and
telemetry untouched. Caps bit granted only when the renderer reports
support (Vulkan + paired CUDA device, same gate as IMPORT_ALLOC).

## 3. Renderer seam + Vulkan implementation

`HostRenderer` gains defaulted-unsupported virtuals:
`geom_create_view`, `geom_release_view`, `geom_draw_points`.

VulkanRenderer:
- **View** = RGBA8 `VkImage` (COLOR_ATTACHMENT | SAMPLED), framebuffer +
  render pass (clear-on-load, store), registered in the existing texture
  table so ImGui sampling and `debug_readback_rgba8` work unchanged.
- **Pipeline** (created lazily once): POINT_LIST topology, vertex shader
  pulls `vec3` positions from a storage buffer at
  `pos_base + gl_VertexIndex`, applies `proj*view`, writes `gl_PointSize`;
  fragment shader samples the shared colormap LUT by the point's attr value
  (second storage buffer; a specialization/flag handles attr-absent = flat
  white); **additive blending, no depth** — a glowing particle cloud, and
  order-independent so no sort is needed.
- **Submission**: record + fenced submit per draw (v1), matching the v1.2
  sync-fallback discipline. `last_device_path()` reports
  `"points-imported"` on success.
- Point size clamped to device limits; `largePoints` not required for the
  1-px verification rows.

## 4. Verification (`gfx-cuda` rows + stub tests)

Stub-renderer tests (portable): caps gate off ⇒ every call inert; view
lifecycle; draw with unknown alloc / misaligned offset / out-of-bounds
window / released alloc ⇒ false.

Hardware rows (UUID-gated like all gfx-cuda): rasterization of 1-px points
at exact pixel centers is deterministic, so the byte-exact discipline
applies where defined:
- N known points at pixel centers of a 64×64 view, flat attr ⇒ the exact
  expected pixels equal the exact expected LUT color; every other pixel
  equals the clear color. Byte-exact, no tolerances.
- attr-colormapped row: two points with attr {vmin, vmax} ⇒ LUT[0] and
  LUT[255] exactly.
- count=0 ⇒ pure clear. Negative rows: pixels untouched, telemetry
  untouched.
- positions at a nonzero pool offset (the flow_scope shape) ⇒ same
  byte-exact result.

## 5. flow_scope (applet id `dev.caliper.flow-scope`)

- **Sim** (worker job, pure torch CUDA, all state allocated ONCE inside
  `pool.use()` — no per-step allocation): positions ping/pong `(N,3)` f32,
  velocities `(N,3)`, speed attr `(N,)`. Per step:
  `v += F(p,t)·dt + impulse(p)·dt − damping·v·dt; p' = p + v·dt` written
  in place into the write buffer; out-of-bounds particles respawn. Field
  `F` is analytic and divergence-free:
  `F(p) = A·(sin(k·z+t)+cos(k·y+t), sin(k·x+t)+cos(k·z+t), sin(k·y+t)+cos(k·x+t))`
  (each component independent of its own axis ⇒ div F = 0).
- **Ping/pong** is the memory-stability contract in practice: the frame
  thread draws buffer A while the sim writes buffer B; the swap publishes
  under the state mutex. `to_bridge` once per buffer at startup; the pool
  block imports once and everything after is offsets.
- **Frame thread**: orbit/zoom camera (hand-rolled lookAt/perspective, no
  new deps); click-drag ray-casts to the y=0 plane and publishes an
  impulse (position, strength) the sim applies as a Gaussian force splat;
  `draw_points(view, cam, posA, speed, MAGMA, …)`; `ImGui::Image(view)`.
  N default 1,000,000 (const for v1).
- **Honest status line**: "N particles — zero-copy (imported geometry)"
  only when this frame's draw returned true with pool-backed buffers;
  otherwise the fallback labels itself.
- **Fallback** (service absent / pool absent / any gate false): ImPlot3D
  scatter of a 10k CPU subsample, labeled "CPU fallback (subsampled)".
  Worst case slow and honest, never broken.

## 6. Risks / open points

- `gl_PointSize` semantics and additive blend are well-trodden Vulkan;
  the risky integration is the offscreen view sharing the texture table —
  mitigated by reusing the existing texture registration path.
- Sim-vs-draw overlap is handled by ping/pong; if hardware shows tearing
  anyway, the fix is a third buffer, not a redesign.
- 1M×3 f32 ≈ 12 MB/buffer; pool blocks are 2 MiB-granularity — fine.
- Draw cost at 1M additive 1-2px points on RTX 500 Ada: expected well
  under 2 ms; if not, N is a constant.
