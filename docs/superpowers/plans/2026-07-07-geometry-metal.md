# Metal/MPS geometry.v1 Zero-Copy Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Give flow_scope (and any geometry.v1/import-path applet) the same zero-copy GPU path on macOS/Apple Silicon that it has on Windows/NVIDIA — torch MPS tensors rendered as points by the Metal renderer with no CPU staging.

**Architecture:** Three additive layers. (1) A new bridge handle kind `CALIPER_ALLOC_HANDLE_MTLBUFFER` — on unified memory every MPS tensor already lives in an `id<MTLBuffer>`, so "import" is an in-process retain, not an OS-handle dup. (2) The Metal renderer grows the import table, imported-texture updates, and a geometry.v1 point pipeline mirroring the Vulkan one byte-for-byte (same vertex-pulling, same colormap index math, same additive blend / no depth / square points). (3) An `#elif __APPLE__` MPS variant of `ExportablePool` with the identical public surface, so flow_scope's pool-path code compiles near-unchanged; there is no allocation routing to do — `use()` is a no-op and `to_bridge()` extracts (buffer, size, offset) from the tensor's storage.

**Tech Stack:** ObjC++/ARC (`metal_renderer.mm`), MSL compiled at runtime via `newLibraryWithSource` (repo convention — no `.metal` files), libtorch 2.5.1 MPS, doctest (`caliper_tests`, `caliper_gfx_tests`, `caliper_torch_tests`).

## Global Constraints

- **Frozen-contract discipline:** ABI changes are ADDITIVE ONLY — new `#define` constants, no struct layout changes, no epoch bump (mirrors how v1.2 itself was added).
- **Byte-exact cross-backend contract (§16):** colormap index math is `idx = uint(clamp((v-vmin)/(vmax-vmin),0,1)*255 + 0.5)`, NaN→0, degenerate range→0 — bit-identical to `points.vert`, `cmap_f32`, and the CPU reference. Square points, no round-sprite discard, no AA.
- **Fail closed, pixels untouched:** every gate that returns false must leave view pixels and `last_device_path()` unchanged. Never claim a fast path not taken.
- **No CPU waits on the hot path:** commit and rely on command-buffer retention + same-`queue_` commit order (M1/D23). `waitUntilCompleted` only in test readbacks.
- **NDC convention:** GL-style +Y up. The gfx test `ndc_for_pixel` bakes `y = 1 - 2*(py+0.5)/h`. Metal NDC is already +Y-up with top-left framebuffer origin → **plain positive-height viewport, do NOT flip**.
- **Threading:** all renderer/bridge/geometry calls are frame-thread-only; no locks in the renderer. Worker↔frame handoff is flow_scope's triple-buffer + full-device drain before publish.
- **Applet targets Apple Silicon** (arm64 libtorch, unified memory). `setBuffer:offset:` uses 4-byte alignment (Apple GPU family); the bridge's existing 4-byte gates cover it.
- Build dir: `cmake-build-debug` (CLion). Test binaries: `caliper_tests`, `caliper_gfx_tests`, `caliper_torch_tests`.

## File Map

| File | Change |
|---|---|
| `sdk/include/caliper/services/tensor_bridge_v1_2.h` | +1 constant: `CALIPER_ALLOC_HANDLE_MTLBUFFER 3u` |
| `src/host/tensor_bridge.cpp` | accept handle kind 3 in the import gate (~L402) |
| `tests/test_abi.cpp` | pin the new constant |
| `src/host/renderer/metal_renderer.mm` | import table + `tex_update_from_imported` + geometry.v1 (view + point pipeline) |
| `tests/gfx/gfx_main.cpp` | Metal-gated: imported-texture rows + geometry byte-exact rows |
| `sdk/include/caliper/adapters/exportable_pool.hpp` | `#elif __APPLE__` MPS variant of `ExportablePool` |
| `tests/test_torch_adapter.cpp` | MPS storage-ref extraction tests |
| `applets/flow_scope/flow_scope.cpp` | enable the zero-copy gate for MPS |
| `ZEROCOPY.md` | one row: Metal imported allocations |

---

### Task 1: ABI handle kind + host import gate

**Files:**
- Modify: `sdk/include/caliper/services/tensor_bridge_v1_2.h` (after the two existing handle-kind defines, ~L17-19)
- Modify: `src/host/tensor_bridge.cpp` (`import_allocation` handle-type gate, ~L402-404)
- Test: `tests/test_abi.cpp`

**Interfaces:**
- Produces: `CALIPER_ALLOC_HANDLE_MTLBUFFER` (`3u`) — used by Tasks 2, 4.

- [ ] **Step 1: Write the failing test.** In `tests/test_abi.cpp`, find the existing v1.2 pin block (search `CALIPER_ALLOC_HANDLE_OPAQUE_FD`) and add alongside, matching local style:

```cpp
// Additive v1.2 handle kind: an in-process id<MTLBuffer> (Apple). Value is
// frozen — 1=win32, 2=fd, 3=mtlbuffer; renumbering breaks shipped applets.
static_assert(CALIPER_ALLOC_HANDLE_MTLBUFFER == 3u, "frozen handle kind");
```

- [ ] **Step 2: Verify it fails.** `cmake --build cmake-build-debug --target caliper_tests 2>&1 | tail -5` → compile error: `CALIPER_ALLOC_HANDLE_MTLBUFFER` undeclared.

- [ ] **Step 3: Add the constant.** In `tensor_bridge_v1_2.h` after the `OPAQUE_FD` define:

```c
/* void* is an in-process id<MTLBuffer> (Apple unified memory). No OS handle
 * transfer: the "dup" the host performs is an ObjC strong retain. Additive,
 * same discipline as the two kinds above. */
#define CALIPER_ALLOC_HANDLE_MTLBUFFER    3u
```

- [ ] **Step 4: Widen the host gate.** In `src/host/tensor_bridge.cpp` `import_allocation`, extend the `handle_type` validation (currently accepts kinds 1 and 2, logs `"import: bad handle type"`) to also accept `CALIPER_ALLOC_HANDLE_MTLBUFFER`. The bridge stays kind-agnostic — it forwards the tag to `renderer_.import_external_allocation(...)` unchanged; a renderer that doesn't speak kind 3 returns 0 (fail closed).

- [ ] **Step 5: Build + run.** `cmake --build cmake-build-debug --target caliper_tests && ./cmake-build-debug/tests/caliper_tests` (adjust path if the binary lands elsewhere; find with `find cmake-build-debug -name caliper_tests -type f`). Expected: PASS.

- [ ] **Step 6: Commit.** `git add -A sdk src/host/tensor_bridge.cpp tests/test_abi.cpp && git commit -m "feat(sdk,host): CALIPER_ALLOC_HANDLE_MTLBUFFER — additive v1.2 handle kind for in-process Metal buffers"`

---

### Task 2: Metal renderer — imported allocations + imported-texture updates

**Files:**
- Modify: `src/host/renderer/metal_renderer.mm`
- Test: `tests/gfx/gfx_main.cpp` (Metal section, after the existing `mat_*` cases ~L437)

**Interfaces:**
- Consumes: `CALIPER_ALLOC_HANDLE_MTLBUFFER` (Task 1).
- Produces (overrides of `host_renderer.h` virtuals, exact signatures from L61-70):
  - `bool supports_external_import() const` → `device_ != nil`
  - `uint64_t import_external_allocation(void* os_handle, uint64_t size_bytes, uint32_t handle_type)`
  - `void release_external_allocation(uint64_t id)`
  - `bool tex_update_from_imported(uint64_t tex, uint64_t alloc, uint64_t offset_bytes, const CaliperTensor& desc, int32_t colormap, float vmin, float vmax)`
  - Internal: `std::unordered_map<uint64_t, id<MTLBuffer>> imported_; uint64_t next_import_id_ = 1;` and a resolver `id<MTLBuffer> lookup_import(uint64_t)` — Task 3 reuses these.

**Notes for the implementer:**
- `tensor_bridge_v1_2.h` must be included (for the handle-kind constant); check what `metal_renderer.mm`/`host_renderer.h` already pull in first.
- Device match: torch's MPS device and `device_` are both the system default GPU but not guaranteed pointer-equal — compare `buf.device.registryID == device_.registryID`, decline on mismatch.
- `tex_update_from_imported` reuses `colormap_compute`/`blit_u8` logic but sources from the imported buffer at `offset_bytes` via `setBuffer:offset:` / `sourceOffset:` — **no shader changes**. Resolve the LUT with `colormap_lut(colormap)` (declared in `src/host/tensor_bridge.h:32`; F32 requires a non-null LUT, U8 ignores it — mirror `tex_update_from_device`'s dispatch). Bounds: `offset_bytes <= buf.length && tensor_extent_bytes(desc, elem) <= buf.length - offset_bytes` (overflow-safe order). Set `last_device_path_ = "compute-imported"` / `"blit-imported"` (must match Vulkan's strings, `vulkan_renderer.cpp:2052/2065`).
- Stream ordering: keep the `desc.stream != nullptr → order_after_producer(...)` call exactly as in the non-imported paths.

- [ ] **Step 1: Write the failing gfx tests.** In `tests/gfx/gfx_main.cpp`'s `#ifdef CALIPER_HAVE_METAL` section, add after the existing Metal cases (style-match the `mat_*` cases; `device_buffer(src, bytes)` at ~L406 fabricates shared MTLBuffers; `Backend bk = metal_backend()` gives `bk.bridge`/`bk.readback`):

```cpp
// ---- v1.2 imported-allocation rows (Metal: in-process MTLBuffer import) ----

TEST_CASE("gfx/metal: import in-process MTLBuffer, colormap from a nonzero offset, byte-exact") {
    if (!metal_env().ok) { MESSAGE("no Metal device — skipping"); return; }
    Backend bk = metal_backend();
    REQUIRE(bk.bridge->caps() & CALIPER_BRIDGE_CAP_IMPORT_ALLOC);

    // 4×4 f32 ramp at byte offset 256 inside a 4096-byte buffer.
    const int W = 4, H = 4;
    const uint64_t off = 256;
    std::vector<uint8_t> bytes(4096, 0);
    float ramp[W * H];
    for (int i = 0; i < W * H; ++i) ramp[i] = (float)i / (float)(W * H - 1);
    std::memcpy(bytes.data() + off, ramp, sizeof(ramp));
    id<MTLBuffer> buf = device_buffer(bytes.data(), bytes.size());
    REQUIRE(buf != nil);

    CaliperAllocId alloc = bk.bridge->import_allocation(
        (__bridge void*)buf, bytes.size(), CALIPER_ALLOC_HANDLE_MTLBUFFER);
    REQUIRE(alloc != 0);

    CaliperTextureId tex = bk.bridge->create_texture(W, H);   // match the mat_* cases' creation call
    REQUIRE(tex != 0);

    CaliperTensor d{};
    d.struct_size = sizeof(CaliperTensor);
    d.dtype = CALIPER_DT_F32; d.ndim = 2;
    d.shape[0] = H; d.shape[1] = W; d.strides[0] = W; d.strides[1] = 1;
    d.device = CALIPER_DEV_METAL;    // data/stream stay null: alloc+offset IS the address
    REQUIRE(bk.bridge->update_texture_from_alloc(tex, alloc, off, &d));
    CHECK(std::string(bk.renderer->last_device_path()) == "compute-imported");

    // CPU reference: identical to how the existing mat_f32 cases build theirs
    // (map_f32_to_rgba8 over the same LUT) — byte-exact CHECK against readback.
    auto got = bk.readback(tex, W, H);
    REQUIRE(got.size() == (size_t)W * H * 4);
    const uint32_t* lut = caliper_host::colormap_lut(CALIPER_CMAP_VIRIDIS);
    for (int i = 0; i < W * H; ++i) {
        float t = ramp[i];                       // vmin=0, vmax=1 below
        uint32_t idx = (uint32_t)(t * 255.0f + 0.5f);
        uint32_t packed = lut[idx];
        CHECK(got[(size_t)i*4+0] == (uint8_t)(packed & 0xFF));
        CHECK(got[(size_t)i*4+1] == (uint8_t)((packed >> 8) & 0xFF));
        CHECK(got[(size_t)i*4+2] == (uint8_t)((packed >> 16) & 0xFF));
        CHECK(got[(size_t)i*4+3] == (uint8_t)((packed >> 24) & 0xFF));
    }
    bk.bridge->release_allocation(alloc);
    CHECK_FALSE(bk.bridge->update_texture_from_alloc(tex, alloc, off, &d));  // released → refuses
}

TEST_CASE("gfx/metal: import gates fail closed") {
    if (!metal_env().ok) { MESSAGE("no Metal device — skipping"); return; }
    Backend bk = metal_backend();
    std::vector<uint8_t> z(1024, 0);
    id<MTLBuffer> buf = device_buffer(z.data(), z.size());

    // wrong handle kind / null handle / zero size / size overclaim
    CHECK(bk.bridge->import_allocation((__bridge void*)buf, 1024,
                                       CALIPER_ALLOC_HANDLE_OPAQUE_FD) == 0);
    CHECK(bk.bridge->import_allocation(nullptr, 1024,
                                       CALIPER_ALLOC_HANDLE_MTLBUFFER) == 0);
    CHECK(bk.bridge->import_allocation((__bridge void*)buf, 0,
                                       CALIPER_ALLOC_HANDLE_MTLBUFFER) == 0);
    CHECK(bk.bridge->import_allocation((__bridge void*)buf, 4096,
                                       CALIPER_ALLOC_HANDLE_MTLBUFFER) == 0);  // buf.length is 1024

    // OOB offset on a valid import refuses and leaves pixels untouched
    CaliperAllocId alloc = bk.bridge->import_allocation(
        (__bridge void*)buf, 1024, CALIPER_ALLOC_HANDLE_MTLBUFFER);
    REQUIRE(alloc != 0);
    CaliperTextureId tex = bk.bridge->create_texture(4, 4);
    CaliperTensor d{};
    d.struct_size = sizeof(CaliperTensor);
    d.dtype = CALIPER_DT_F32; d.ndim = 2;
    d.shape[0] = 4; d.shape[1] = 4; d.strides[0] = 4; d.strides[1] = 1;
    d.device = CALIPER_DEV_METAL;
    CHECK_FALSE(bk.bridge->update_texture_from_alloc(tex, alloc, 1024, &d));   // offset==length
    CHECK_FALSE(bk.bridge->update_texture_from_alloc(tex, alloc, 1000, &d));   // extent past end
    bk.bridge->release_allocation(alloc);
}
```

  **Adaptation note (required, not optional):** before finalizing, read one existing `mat_*` Metal case end-to-end and match its exact texture-creation call, vmin/vmax passing (the mat cases show where colormap/vmin/vmax are configured — replicate that mechanism), and reference-building helper if one exists (prefer the shared helper over the inline loop above). The test intent (import → update from offset → byte-exact → gates → release refuses) is fixed; the plumbing must match the file's house style.

- [ ] **Step 2: Verify they fail.** `cmake --build cmake-build-debug --target caliper_gfx_tests && find cmake-build-debug -name caliper_gfx_tests -type f -exec {} \;` → new cases FAIL (`caps()` lacks `CALIPER_BRIDGE_CAP_IMPORT_ALLOC` since `supports_external_import()` is false).

- [ ] **Step 3: Implement in `metal_renderer.mm`.** Members (next to `textures_`):

```objc
// v1.2 imported allocations: in-process MTLBuffers, strong-retained (ARC) —
// the Metal analog of Vulkan's DuplicateHandle+VkImportMemory. 0 invalid.
std::unordered_map<uint64_t, id<MTLBuffer>> imported_;
uint64_t next_import_id_ = 1;
```

Overrides (public section):

```objc
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
        return colormap_compute_from(tex, dst, src, offset_bytes, desc, lut, vmin, vmax);
    }
    if (desc.dtype == CALIPER_DT_U8)
        return blit_u8_from(tex, dst, src, offset_bytes, desc);
    return false;
}
```

Private: `id<MTLBuffer> lookup_import(uint64_t iid) { auto it = imported_.find(iid); return it == imported_.end() ? nil : it->second; }`

Refactor `colormap_compute`/`blit_u8` into `_from` variants taking `uint64_t src_offset` (existing callers pass 0): in `colormap_compute_from`, the extent gate becomes `tensor_extent_bytes(t, sizeof(float)) <= src.length - src_offset` and the bind becomes `[enc setBuffer:src offset:(NSUInteger)src_offset atIndex:0]`; on success set `last_device_path_ = src_offset || from_import ? "compute-imported" : "compute"` — cleaner: pass a `bool imported` flag; strings must be exactly `"compute-imported"`/`"blit-imported"` on the imported path, `"compute"`/`"blit"` unchanged on the direct path. Same for `blit_u8_from` with `sourceOffset:(NSUInteger)src_offset`. Keep `order_after_producer` calls intact (imported path: `desc.stream` is normally null — alloc+offset addressing — but honor it if set). Clean up `imported_` in `shutdown()` (`imported_.clear();`).

Also check the includes: add `#include <caliper/services/tensor_bridge_v1_2.h>` if the constant isn't already visible via `host_renderer.h`'s includes.

- [ ] **Step 4: Build + run gfx tests.** Same command as Step 2. Expected: new cases PASS, all pre-existing Metal cases still PASS (the `_from` refactor is covered by them).

- [ ] **Step 5: Commit.** `git commit -am "feat(metal): v1.2 imported allocations — in-process MTLBuffer import + imported-texture updates"`

---

### Task 3: Metal renderer — geometry.v1 point pipeline

**Files:**
- Modify: `src/host/renderer/metal_renderer.mm`
- Test: `tests/gfx/gfx_main.cpp` (Metal section)

**Interfaces:**
- Consumes: `imported_` table + `lookup_import` (Task 2).
- Produces (overrides, signatures from `host_renderer.h:82-92`):
  - `bool supports_geometry() const` → `supports_external_import()`
  - `uint64_t geom_create_view(int w, int h)`
  - `bool geom_draw_points(uint64_t view_tex, const float* view16, const float* proj16, uint64_t pos_alloc, uint64_t pos_offset, uint64_t count, uint64_t attr_alloc, uint64_t attr_offset, const uint32_t* lut256, float vmin, float vmax, float size_px, uint32_t clear_rgba)`

- [ ] **Step 1: Write the failing gfx tests.** Mirror the two Vulkan geometry cases (`gfx_main.cpp:1422-1532`) in the Metal section. The Vulkan-gated helpers (`ndc_for_pixel`, `identity_cam`, `geom_ref`, quoted in full at gfx_main.cpp:1387-1418) are compiled out on Mac — duplicate them verbatim into the Metal section's anonymous namespace. Then:

```cpp
// ---- caliper.geometry.v1 rows (Metal): byte-exact mirror of the Vulkan cases,
// alloc source = in-process shared MTLBuffer instead of CUDA VMM. ----

TEST_CASE("gfx/metal geometry: imported points byte-exact — colormap extremes at a nonzero offset") {
    if (!metal_env().ok) { MESSAGE("no Metal device — skipping"); return; }
    Backend bk = metal_backend();
    REQUIRE(bk.bridge->geom_caps() & CALIPER_GEOM_CAP_IMPORTED_POINTS);

    const int W = 64, H = 64;
    const uint64_t pos_off = 512, attr_off = 2048;
    const std::vector<std::pair<int,int>> px = {{3, 5}, {40, 22}, {63, 63}};
    const float attrs[3] = {0.0f, 1.0f, 0.5f};      // LUT[0], LUT[255], LUT[128]

    std::vector<uint8_t> bytes(4096, 0);
    float pos[9];
    for (int i = 0; i < 3; ++i) ndc_for_pixel(px[i].first, px[i].second, W, H, &pos[i*3]);
    std::memcpy(bytes.data() + pos_off,  pos,   sizeof(pos));
    std::memcpy(bytes.data() + attr_off, attrs, sizeof(attrs));
    id<MTLBuffer> buf = device_buffer(bytes.data(), bytes.size());
    REQUIRE(buf != nil);

    CaliperAllocId alloc = bk.bridge->import_allocation(
        (__bridge void*)buf, bytes.size(), CALIPER_ALLOC_HANDLE_MTLBUFFER);
    REQUIRE(alloc != 0);
    CaliperTextureId view = bk.bridge->geom_create_view(W, H);
    REQUIRE(view != 0);

    CaliperGeomCamera cam = identity_cam();
    REQUIRE(bk.bridge->geom_draw_points(view, &cam, alloc, pos_off, 3,
                                        alloc, attr_off, CALIPER_CMAP_VIRIDIS,
                                        0.f, 1.f, 1.f, 0xFF000000u));
    CHECK(std::string(bk.renderer->last_device_path()) == "points-imported");

    const uint32_t* lut = caliper_host::colormap_lut(CALIPER_CMAP_VIRIDIS);
    auto ref = geom_ref(W, H, 0xFF000000u, px, {lut[0], lut[255], lut[128]});
    auto got = bk.readback(view, W, H);
    REQUIRE(got.size() == ref.size());
    for (size_t i = 0; i < got.size(); ++i)
        if (got[i] != ref[i]) { FAIL("first diff at byte ", i, ": got ", (int)got[i], " ref ", (int)ref[i]); }
    CHECK(got == ref);

    // flat-white path: attr_alloc = 0
    REQUIRE(bk.bridge->geom_draw_points(view, &cam, alloc, pos_off, 3,
                                        0, 0, CALIPER_CMAP_VIRIDIS,
                                        0.f, 1.f, 1.f, 0xFF000000u));
    auto ref2 = geom_ref(W, H, 0xFF000000u, px,
                         {0xFFFFFFFFu, 0xFFFFFFFFu, 0xFFFFFFFFu});
    CHECK(bk.readback(view, W, H) == ref2);
    bk.bridge->release_allocation(alloc);
    bk.bridge->geom_release_view(view);
}

TEST_CASE("gfx/metal geometry: count 0 clears; gates keep prior pixels; released refuses") {
    if (!metal_env().ok) { MESSAGE("no Metal device — skipping"); return; }
    Backend bk = metal_backend();
    if ((bk.bridge->geom_caps() & CALIPER_GEOM_CAP_IMPORTED_POINTS) == 0) return;

    const int W = 32, H = 32;
    CaliperTextureId view = bk.bridge->geom_create_view(W, H);
    REQUIRE(view != 0);
    CaliperGeomCamera cam = identity_cam();

    // count==0 = pure clear (teal), alloc ids 0
    const uint32_t teal = 10u | (20u << 8) | (30u << 16) | (255u << 24);
    REQUIRE(bk.bridge->geom_draw_points(view, &cam, 0, 0, 0, 0, 0,
                                        CALIPER_CMAP_VIRIDIS, 0.f, 1.f, 1.f, teal));
    CHECK(bk.readback(view, W, H) == geom_ref(W, H, teal, {}, {}));

    // one real point, then assert every gate refuses AND pixels stay put
    std::vector<uint8_t> bytes(1024, 0);
    float p3[3]; ndc_for_pixel(7, 9, W, H, p3);
    std::memcpy(bytes.data(), p3, sizeof(p3));
    id<MTLBuffer> buf = device_buffer(bytes.data(), bytes.size());
    CaliperAllocId alloc = bk.bridge->import_allocation(
        (__bridge void*)buf, bytes.size(), CALIPER_ALLOC_HANDLE_MTLBUFFER);
    REQUIRE(alloc != 0);
    REQUIRE(bk.bridge->geom_draw_points(view, &cam, alloc, 0, 1, 0, 0,
                                        CALIPER_CMAP_VIRIDIS, 0.f, 1.f, 1.f, 0xFF000000u));
    auto snap = bk.readback(view, W, H);
    std::string path = bk.renderer->last_device_path();

    CHECK_FALSE(bk.bridge->geom_draw_points(view, &cam, alloc, 2, 1, 0, 0,
                CALIPER_CMAP_VIRIDIS, 0.f, 1.f, 1.f, 0xFF000000u));      // misaligned
    CHECK_FALSE(bk.bridge->geom_draw_points(view, &cam, alloc, 0, 1024/12 + 1, 0, 0,
                CALIPER_CMAP_VIRIDIS, 0.f, 1.f, 1.f, 0xFF000000u));      // OOB count
    CHECK_FALSE(bk.bridge->geom_draw_points(view, &cam, 999, 0, 1, 0, 0,
                CALIPER_CMAP_VIRIDIS, 0.f, 1.f, 1.f, 0xFF000000u));      // unknown alloc
    CHECK_FALSE(bk.bridge->geom_draw_points(view, nullptr, alloc, 0, 1, 0, 0,
                CALIPER_CMAP_VIRIDIS, 0.f, 1.f, 1.f, 0xFF000000u));      // null cam
    CHECK(bk.readback(view, W, H) == snap);
    CHECK(std::string(bk.renderer->last_device_path()) == path);

    bk.bridge->release_allocation(alloc);
    CHECK_FALSE(bk.bridge->geom_draw_points(view, &cam, alloc, 0, 1, 0, 0,
                CALIPER_CMAP_VIRIDIS, 0.f, 1.f, 1.f, 0xFF000000u));      // released alloc
    bk.bridge->geom_release_view(view);
    CHECK_FALSE(bk.bridge->geom_draw_points(view, &cam, 0, 0, 0, 0, 0,
                CALIPER_CMAP_VIRIDIS, 0.f, 1.f, 1.f, 0xFF000000u));      // released view
}
```

  **Adaptation note:** diff these against the Vulkan originals at L1422-1532 first; where the original asserts something these omit (or names a call differently), follow the original. Note the Vulkan OOB-count case derives count from the *real padded block size* — for Metal the buffer length IS the real size, so `1024/12 + 1` is correct here.

- [ ] **Step 2: Verify they fail.** Build + run `caliper_gfx_tests` → both new cases FAIL (`geom_caps() == 0`).

- [ ] **Step 3: Implement.** In `metal_renderer.mm`:

**(a) MSL point shader** (file-scope, sibling to `kColormapShaderSrc`). The color math mirrors `points.vert` lines 32-41 exactly (including `vmax > vmin`, which differs from `cmap_f32`'s `denom != 0` — points.vert is the spec here):

```objc
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
```

**(b) C++ params mirror + members:**

```cpp
struct GeomParams {          // must match the MSL struct byte-for-byte (88 B)
    float    mvp[16];
    uint32_t pos_base, attr_base, use_attr;
    float    vmin, vmax, size_px;
};
static_assert(sizeof(GeomParams) == 88, "MSL constant-buffer layout");
```

Members: `id<MTLRenderPipelineState> points_pipeline_ = nil;` (nil it in `shutdown()`).

**(c) `ensure_points_pipeline()`** (private, sibling of `ensure_pipeline()`):

```objc
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
    d.colorAttachments[0].pixelFormat        = MTLPixelFormatRGBA8Unorm;
    d.colorAttachments[0].blendingEnabled    = YES;   // additive ONE/ONE, both channels
    d.colorAttachments[0].rgbBlendOperation  = MTLBlendOperationAdd;
    d.colorAttachments[0].alphaBlendOperation = MTLBlendOperationAdd;
    d.colorAttachments[0].sourceRGBBlendFactor        = MTLBlendFactorOne;
    d.colorAttachments[0].destinationRGBBlendFactor   = MTLBlendFactorOne;
    d.colorAttachments[0].sourceAlphaBlendFactor      = MTLBlendFactorOne;
    d.colorAttachments[0].destinationAlphaBlendFactor = MTLBlendFactorOne;
    points_pipeline_ = [device_ newRenderPipelineStateWithDescriptor:d error:&err];
    return points_pipeline_ != nil;
}
```

**(d) Overrides:**

```objc
bool supports_geometry() const override { return supports_external_import(); }

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
    uint64_t id = next_id_++;
    textures_[@(id)] = tex;
    return id;
}

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
            // mvp = proj * view, column-major — same loop as vulkan_renderer.cpp:627-635
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
        [cb commit];   // no CPU wait: same-queue_ commit order covers the frame's sampling;
                       // producer (MPS) writes are already CPU-drained before publish
                       // (flow_scope sync contract), so no cross-queue event is needed here.
        last_device_path_ = "points-imported";
        return true;
    }
}
```

Add `#include <algorithm>` if `std::min/max` aren't already available.

- [ ] **Step 4: Build + run gfx tests.** Expected: both geometry cases PASS byte-exact, everything else still green. If the byte-exact case fails on pixel positions (not colors), the NDC mapping is off — re-check the viewport (must be positive-height) before touching anything else. If it fails on colors only, diff the MSL index math against `points.vert` character-by-character.

- [ ] **Step 5: Commit.** `git commit -am "feat(metal): geometry.v1 point pipeline — vertex-pulled from imported MTLBuffers, byte-exact vs CPU reference"`

---

### Task 4: MPS ExportablePool variant

**Files:**
- Modify: `sdk/include/caliper/adapters/exportable_pool.hpp` (replace the `#else // !CALIPER_EXPORTABLE_POOL_CUDA` fallback block at L502-524 with an `#elif` chain)
- Test: `tests/test_torch_adapter.cpp`

**Interfaces:**
- Consumes: `CALIPER_ALLOC_HANDLE_MTLBUFFER` (Task 1), `caliper::Bridge::import_allocation/release_allocation` (existing, caliper.hpp L307-318).
- Produces: `ExportablePool` on `__APPLE__` with the same public surface flow_scope uses — `ExportablePool(int)`, `ok()`, `use()` → `Scope`, `to_bridge(Bridge&, const at::Tensor&)` → `std::optional<BridgeRef>`, `registry()`. Plus a testable static: `ExportablePool::storage_ref(const at::Tensor&)` → `std::optional<MpsStorageRef>`.

- [ ] **Step 1: Write the failing tests.** In `tests/test_torch_adapter.cpp`, after the CUDA pool case (~L307):

```cpp
TEST_CASE("mps exportable pool: storage_ref extracts (buffer, size, offset) "
          "from tensor storage" * doctest::skip(!torch::mps::is_available())) {
#if defined(__APPLE__)
    caliper::adapters::ExportablePool pool(0);
    REQUIRE(pool.ok());

    auto t = torch::rand({17, 9}, torch::TensorOptions()
                                      .device(torch::kMPS).dtype(torch::kFloat32));
    auto ref = caliper::adapters::ExportablePool::storage_ref(t);
    REQUIRE(ref.has_value());
    CHECK(ref->buffer == t.storage().mutable_data());   // the id<MTLBuffer> bridge pointer
    CHECK(ref->size >= (uint64_t)t.numel() * sizeof(float));
    CHECK(ref->offset == 0);

    // a slice that shares storage carries its byte offset — this is exactly
    // what the (alloc, offset) addressing fixes vs the offset-rejecting v1 path
    auto view = t.reshape({17 * 9}).slice(0, 9, 18);    // storage_offset 9 elements
    auto vref = caliper::adapters::ExportablePool::storage_ref(view);
    REQUIRE(vref.has_value());
    CHECK(vref->buffer == ref->buffer);
    CHECK(vref->offset == 9 * sizeof(float));

    // rejections: CPU tensor, non-contiguous
    CHECK_FALSE(caliper::adapters::ExportablePool::storage_ref(
        torch::rand({4, 4})).has_value());
    CHECK_FALSE(caliper::adapters::ExportablePool::storage_ref(
        t.transpose(0, 1)).has_value());

    // to_bridge against a null (default) Bridge declines without crashing
    caliper::Bridge nobridge{};
    CHECK_FALSE(pool.to_bridge(nobridge, t).has_value());
#endif
}
```

  **Adaptation note:** check `caliper::Bridge`'s default constructibility in `caliper.hpp` first; if `Bridge{}` isn't default-constructible, drop only that last assertion (the decline path is then covered by `import_allocation` returning 0 end-to-end in the app). Check whether `slice` on a reshaped MPS tensor stays contiguous — it does for a 1-D slice with step 1; keep `is_contiguous()` semantics identical to the CUDA path (which never sees non-contiguous pool hits anyway).

- [ ] **Step 2: Verify it fails.** `cmake --build cmake-build-debug --target caliper_torch_tests && find cmake-build-debug -name caliper_torch_tests -type f -exec {} \;` → FAIL: `storage_ref` not a member (the current `#else` stub has no such method and `ok()` is false).

- [ ] **Step 3: Implement.** In `exportable_pool.hpp`, change the tail of the file from `#else` to a three-way chain. Add near `BridgeRef` (outside the `#if`, it's shared):

```cpp
// The MPS variant's storage view: the ObjC bridge pointer of the tensor
// storage's id<MTLBuffer>, the storage's full byte size, and the tensor's
// byte offset within it.
struct MpsStorageRef {
    void*    buffer;
    uint64_t size;
    uint64_t offset;
};
```

Then replace `#else // !CALIPER_EXPORTABLE_POOL_CUDA` + stub with:

```cpp
#elif defined(__APPLE__)
// ===========================================================================
// MPS variant. On unified memory every MPS tensor's storage IS an
// id<MTLBuffer> (see adapters/torch.hpp:152) — there is nothing to route, so
// use() is a no-op and to_bridge() imports the tensor's own storage buffer
// in-process (handle kind CALIPER_ALLOC_HANDLE_MTLBUFFER; the host's "dup"
// is an ObjC retain). Unlike the offset-rejecting texture path, (alloc,
// offset) addressing carries storage_offset explicitly.
//
// Lifetime contract: a tensor handed to to_bridge must stay alive until the
// pool is destroyed (or the import released) — the MPS caching allocator may
// otherwise hand the same MTLBuffer to an unrelated tensor while the host
// still renders from it. flow_scope's persistent slot tensors satisfy this.
// ===========================================================================
class ExportablePool {
public:
    class Scope { public: Scope() = default; };

    explicit ExportablePool(int /*device_index*/)
        : ok_(torch::mps::is_available()) {}

    ~ExportablePool() {
        if (import_bridge_ != nullptr)
            for (auto& [buf, id] : import_cache_)
                if (id != 0) import_bridge_->release_allocation(id);
    }

    ExportablePool(const ExportablePool&)            = delete;
    ExportablePool& operator=(const ExportablePool&) = delete;

    bool                 ok() const       { return ok_; }
    const AllocRegistry& registry() const { return registry_; }
    Scope                use()            { return Scope{}; }

    // (buffer, size, offset) of an MPS tensor's storage; nullopt for
    // non-MPS / non-contiguous / empty storage. Static and bridge-free so
    // the extraction math is unit-testable without a host.
    static std::optional<MpsStorageRef> storage_ref(const at::Tensor& t) {
        if (!t.is_mps() || !t.is_contiguous()) return std::nullopt;
        void* buf = t.storage().mutable_data();
        if (buf == nullptr) return std::nullopt;
        return MpsStorageRef{
            buf,
            static_cast<uint64_t>(t.storage().nbytes()),
            static_cast<uint64_t>(t.storage_offset()) *
                static_cast<uint64_t>(t.element_size())};
    }

    std::optional<BridgeRef> to_bridge(caliper::Bridge& bridge,
                                       const at::Tensor& t) {
        if (!ok_) return std::nullopt;
        auto ref = storage_ref(t);
        if (!ref) return std::nullopt;
        import_bridge_ = &bridge;
        auto it = import_cache_.find(ref->buffer);
        if (it == import_cache_.end()) {
            // Import once per storage buffer. 0 = host declined; cached as a
            // permanent negative (caller stays on fallback), same discipline
            // as the CUDA variant.
            const CaliperAllocId id = bridge.import_allocation(
                ref->buffer, ref->size, CALIPER_ALLOC_HANDLE_MTLBUFFER);
            it = import_cache_.emplace(ref->buffer, id).first;
        }
        if (it->second == 0) return std::nullopt;
        return BridgeRef{it->second, ref->offset};
    }

private:
    bool ok_ = false;
    AllocRegistry registry_;
    std::map<void*, CaliperAllocId> import_cache_;   // storage buffer -> id
    caliper::Bridge* import_bridge_ = nullptr;       // frame-thread-only, like the bridge
};

#else   // neither CUDA nor Apple
// ... (keep the existing stub class verbatim) ...
#endif  // CALIPER_EXPORTABLE_POOL_CUDA / __APPLE__
```

Include `<map>` and `<optional>` unconditionally at the top (they're currently inside the CUDA guard). `torch::mps::is_available` comes via the existing `#include <torch/torch.h>`; if the build says otherwise, add `#include <torch/mps.h>` under `#if defined(__APPLE__)`.

- [ ] **Step 4: Build + run.** `caliper_torch_tests` → new case PASSES on this machine (MPS available); all prior cases still green. Also rebuild `caliper_tests` to confirm no non-Apple compile regressions in the header (the CUDA/stub branches are untouched text, but the shared `MpsStorageRef` addition must compile everywhere).

- [ ] **Step 5: Commit.** `git commit -am "feat(adapters): ExportablePool MPS variant — in-process MTLBuffer import of tensor storage"`

---

### Task 5: flow_scope MPS enablement

**Files:**
- Modify: `applets/flow_scope/flow_scope.cpp` (worker gate ~L198-212, sync ~L257, state publish ~L235)

**Interfaces:**
- Consumes: MPS `ExportablePool` (Task 4), `caliper::adapters::detail::mps_synchronize_serialized()` (existing, `torch.hpp:105` — flow_scope already includes `caliper/adapters/torch.hpp`).

- [ ] **Step 1: Widen the device gate.** Replace L200-203:

```cpp
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
    const int64_t N = gpu ? 1'000'000 : 50'000;
```

- [ ] **Step 2: Widen the pool opt-in.** L205-212's condition `if (cuda && ...)` becomes `if (gpu && ...)` — the comment updates to "geometry caps + import caps + a GPU device (CUDA or MPS)". The pool construction is identical (`ExportablePool(0)`; the MPS variant ignores the ordinal).

- [ ] **Step 3: Sync before publish.** L257 `if (cuda) torch::cuda::synchronize();` becomes:

```cpp
        if (cuda) torch::cuda::synchronize();     // writes done BEFORE publish
#if defined(__APPLE__)
        else if (mps) caliper::adapters::detail::mps_synchronize_serialized();
        // serialized on torch's MPS stream queue — a bare synchronize() races
        // any other applet's encodes (see torch.hpp:70-111); full drain keeps
        // the renderer's imported-points read ordered without events.
#endif
```

- [ ] **Step 4: Publish flag.** L235 `st->sim_on_cuda = cuda;` becomes `st->sim_on_cuda = gpu;` — first check every read of `sim_on_cuda` in `flow_scope.cpp`/`flow_scope.h` (it feeds the status line's device wording); if any reader prints "CUDA" literally, generalize that string to GPU/device-name or add a `sim_on_mps` field — keep the status line honest per the whitepaper contract.

- [ ] **Step 5: Build the app + run headless checks.** `cmake --build cmake-build-debug --target caliper 2>&1 | tail -3` → clean. Then run the full test sweep: `caliper_tests`, `caliper_gfx_tests`, `caliper_torch_tests` — all green.

- [ ] **Step 6: Live verification (the point of the whole plan).** Launch the app (`./cmake-build-debug/caliper` or the repo's run convention — check `docs/` or ask), open flow_scope, and confirm the status line reads `1000000 particles — zero-copy (imported geometry) · N steps/s` — NOT "CPU fallback". Confirm the point cloud renders and orbits. If the status line says "pool unavailable" or "no geometry service", the corresponding cap didn't light up — debug from `geom_caps` outward, do not soften the status line.

- [ ] **Step 7: Commit.** `git commit -am "feat(flow_scope): zero-copy imported geometry on Apple Silicon — MPS joins the CUDA gate"`

---

### Task 6: Docs + final sweep

**Files:**
- Modify: `ZEROCOPY.md` (the per-allocation-origin table)

- [ ] **Step 1: Add the Metal row.** Read `ZEROCOPY.md` (16 lines) and add one row/line documenting: imported allocations on Metal = in-process `id<MTLBuffer>` retain (handle kind 3), geometry.v1 + imported-texture updates zero-copy on Apple Silicon; the in-VRAM-copy floor is now per-allocation-origin on BOTH GPU platforms.

- [ ] **Step 2: Full test sweep, all three binaries.** All green, plus `git status` shows only intended files.

- [ ] **Step 3: Commit.** `git commit -am "docs(zerocopy): Metal imported-allocation row — zero-copy floor now per-origin on both GPU platforms"`

---

## Self-Review Notes

- **Spec coverage:** Mac equivalent of the Windows zero-copy flow_scope demo = geometry caps (T3) + import caps (T2) + pool (T4) + applet gate (T5); honesty contract preserved (status line, `last_device_path`, fail-closed gates); byte-exact verification mirrored (T2/T3 tests).
- **Known deliberate scope cut:** no `MTLSharedEvent` ordering for the geometry path — flow_scope's full-drain-before-publish makes it unnecessary (same coarse contract as the CUDA path, which also skips STREAM_ORDERED). If a future applet wants stream-ordered points, that's a new task.
- **Type consistency check:** `CALIPER_ALLOC_HANDLE_MTLBUFFER` (T1) used in T2 impl + tests, T4 impl; `lookup_import` produced in T2, consumed in T3; `storage_ref`/`MpsStorageRef` defined and tested in T4; `mps_synchronize_serialized` is pre-existing (torch.hpp:105).
- **Interface drift risk flagged to implementers:** exact call names on `bk.bridge` (`create_texture` vs whatever the mat_* cases use) and vmin/vmax plumbing in the imported-texture test MUST be taken from the neighboring cases in gfx_main.cpp, not from this plan.
