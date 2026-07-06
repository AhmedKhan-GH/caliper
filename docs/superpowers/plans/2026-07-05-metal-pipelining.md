# Metal / MPS Handoff Pipelining (M1 + M2) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Remove the CPU sync stalls from the Metal backend's device-update path (M1) and retire the adapter's full-device drain at the tensor handoff by giving `CaliperTensor.stream` semantics behind a bridge-v1.1 `caps()` negotiation (M2), per `docs/metal-pipelining.md`.

**Architecture:** M1 drops the two `waitUntilCompleted` calls in `metal_renderer.mm` (single-queue commit order makes draw ordering free — D23) and moves the one real hazard, test readback, onto the renderer's own queue via `MetalRenderer::debug_readback_rgba8`. M2 adds an additive service `caliper.tensor_bridge.v1_1` (same six ops + `caps()`); when bit 0 is set the torch adapter populates `t.stream` instead of draining the device, and the renderer GPU-orders its update after the producer's queue/stream (per-texture `MTLSharedEvent` on Metal; the existing timeline-semaphore chain, re-aimed at the producer stream, on Vulkan/CUDA).

**Tech Stack:** Objective-C++ (ARC) / Metal, C++20, doctest, CMake (build dir `build` (Makefiles, BUILD_TESTS=ON)), libtorch (adapter header only — applet side), Vulkan/CUDA driver API (Windows-only TU, code-only here).

## Global Constraints

- FROZEN, do not edit: `sdk/include/caliper/tensor.h` (the `stream` field already exists), `sdk/include/caliper/services/tensor_bridge_v1.h`, the ABI epoch. v1.1 is a NEW header + NEW service id only (spec §5, D24).
- §16 pixel-exactness: every existing byte-equality gfx test must keep passing UNMODIFIED (spec §5).
- `stream == NULL` keeps exact v1 semantics on every path; every rung of the degradation ladder survives (spec §4).
- Adapters skip their device drain ONLY when `caps()` bit 0 (`CALIPER_BRIDGE_CAP_STREAM_ORDERED`) is present; adapters default to drain (spec §7 D24, §8 risk 3).
- After M1, no per-op CPU wait remains on the Metal hot path; `waitUntilCompleted` is allowed only in test readbacks and rare correctness fallbacks (spec §3, §6 M1 exit).
- The host never links torch (D11): all torch code stays in `sdk/include/caliper/adapters/torch.hpp`, which must remain compilable as plain C++ (applet TUs are `.cpp`, not `.mm`).
- Metal command buffers use default (retained) encoding — never `unretained` (spec §8 risk 4); document at encode sites.
- Build/test commands (macOS, this machine):
  - unit: `cmake --build build --target caliper_tests && ./build/tests/caliper_tests`
  - gfx (needs GUI session): `cmake --build build --target caliper_gfx_tests && ./build/tests/caliper_gfx_tests`
  - torch: `cmake --build build --target caliper_torch_tests && ./build/tests/caliper_torch_tests`
  - host app compile check: `cmake --build build --target caliper`
- `src/host/renderer/vulkan_renderer.cpp` is a Windows-only TU: Task 9 is code-only on this machine; its verification is deferred to a Windows checklist (spec §6 M2a).
- Commit style (from repo log): `feat(scope): …`, `fix(scope): …`, `test(gfx): …`, `docs(scope): …`.

---

### Task 1: M1 — `MetalRenderer::debug_readback_rgba8` on the renderer's queue; switch the Mac gfx harness to it

The gfx harness's `metal_readback` blits on its OWN `MTLCommandQueue` — safe today only because the renderer waits per op; it becomes a stale-read hazard the moment Task 2 drops those waits (spec §3.4). Fix the readback FIRST so Task 2 lands against a correct harness.

**Files:**
- Modify: `tests/gfx/gfx_main.cpp:390-430` (replace `metal_readback` with the renderer hook)
- Modify: `src/host/renderer/metal_renderer.mm` (implement `debug_readback_rgba8`)

**Interfaces:**
- Consumes: `HostRenderer::debug_readback_rgba8(uint64_t, int, int)` virtual (already on the seam, `src/host/renderer/host_renderer.h:61`; base returns `{}`).
- Produces: `MetalRenderer::debug_readback_rgba8(uint64_t tex_id, int w, int h) -> std::vector<uint8_t>` that resolves BOTH the internal table id and the public bridge id (the bridged `id<MTLTexture>` pointer value), blits on the renderer's `queue_`, and CPU-waits only there. Tasks 2 and 8 rely on this readback retiring all prior renderer-queue work by commit order.

- [x] **Step 1: Switch the harness readback to the renderer hook (this is the failing test)**

In `tests/gfx/gfx_main.cpp`, delete the whole `metal_readback` function (lines 388–419, the comment included) and change `metal_backend()` to:

```objc
Backend metal_backend() {
    Backend b;
    b.bridge = metal_env().bridge.get();
    b.renderer = metal_env().renderer.get();
    HostRenderer* r = b.renderer;
    b.readback = [r](CaliperTextureId id, int w, int h) {
        return r->debug_readback_rgba8(id, w, h);   // renderer-queue readback (M1)
    };
    return b;
}
```

- [x] **Step 2: Run the Metal gfx tests to verify they fail**

Run: `cmake --build build --target caliper_gfx_tests && ./build/tests/caliper_gfx_tests -tc='*Metal*'`
Expected: FAIL — every Metal pixel-comparison case fails (readback returns an empty vector from the base-class default).

- [x] **Step 3: Implement `debug_readback_rgba8` in `MetalRenderer`**

In `src/host/renderer/metal_renderer.mm`: add `#include <cstring>` next to `#include <cstdio>`. Then add this method to `MetalRenderer` (public section, after `tex_update_from_device`):

```objc
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
```

- [x] **Step 4: Run the full gfx suite to verify it passes**

Run: `./build/tests/caliper_gfx_tests` (after rebuild)
Expected: PASS — all GL, Metal cases green (Vulkan cases absent on macOS).

- [x] **Step 5: Commit**

```bash
git add tests/gfx/gfx_main.cpp src/host/renderer/metal_renderer.mm
git commit -m "feat(metal): renderer-queue debug_readback_rgba8; gfx harness reads back through it (M1 prep)"
```

---

### Task 2: M1 — drop the per-op `waitUntilCompleted`; add the Metal burst test

**Files:**
- Modify: `src/host/renderer/metal_renderer.mm:303-304, 333-334` (the two wait sites)
- Test: `tests/gfx/gfx_main.cpp` (new burst test, after the "short device buffer" case)

**Interfaces:**
- Consumes: Task 1's renderer-queue readback (the burst test's final readback retires the whole in-flight chain).
- Produces: a Metal hot path with zero CPU waits; the frame and readback order by queue commit order alone (D23).

- [x] **Step 1: Write the burst test (the Vulkan one's twin, `gfx_main.cpp:812`)**

Add inside the `#ifdef CALIPER_HAVE_METAL` block, after the "short device buffer is rejected" case:

```objc
// M1 pipelining proof (the Vulkan burst test's twin): several device updates
// enqueued back-to-back with NO readback between them, so successive compute
// passes are in flight together, ordered only by queue commit order (D23).
// The final readback must equal the LAST write byte-for-byte. Fresh source
// buffers per generation keep CPU writes outside the contract (a NULL-stream
// caller owns producer quiescence); dropping each buffer's last strong ref
// mid-flight also exercises command-buffer resource retention (spec §3.2).
TEST_CASE("gfx/Metal: burst updates pipeline in order, final readback pixel-exact") {
    if (!metal_env().ok) { MESSAGE("no Metal device — skipping"); return; }
    Backend bk = metal_backend();

    const int w = 17, h = 9, n = w * h;   // non-16-multiple edge sizes
    auto gen_data = [&](int gen) {
        std::vector<float> d(n);
        for (int i = 0; i < n; ++i) d[i] = (float)((i * 7 + gen * 13) % n);
        return d;
    };

    std::vector<float> d0 = gen_data(0);
    id<MTLBuffer> buf0 = device_buffer(d0.data(), (size_t)n * sizeof(float));
    CaliperTensor t{};
    t.struct_size = sizeof(t); t.data = (__bridge void*)buf0; t.dtype = CALIPER_DT_F32;
    t.ndim = 2; t.shape[0] = h; t.shape[1] = w; t.strides[0] = w; t.strides[1] = 1;
    t.device = CALIPER_DEV_METAL;

    CaliperTextureId id = bk.bridge->texture_from_tensor_mapped(
        &t, CALIPER_CMAP_VIRIDIS, 0.0f, (float)(n - 1), 0);
    REQUIRE(id != 0);

    std::vector<float> last;
    for (int gen = 1; gen <= 8; ++gen) {
        last = gen_data(gen);
        id<MTLBuffer> b = device_buffer(last.data(), (size_t)n * sizeof(float));
        t.data = (__bridge void*)b;      // b's last strong ref dies each loop turn
        REQUIRE(bk.bridge->update_texture(id, &t));
    }
    CHECK(std::string(bk.renderer->last_device_path()) == "compute");

    std::vector<uint8_t> ref((size_t)n * 4);
    map_f32_to_rgba8(last.data(), w, h, colormap_lut(CALIPER_CMAP_VIRIDIS),
                     0.0f, (float)(n - 1), ref.data());
    CHECK(bk.readback(id, w, h) == ref);
    bk.bridge->release_texture(id);
}
```

- [x] **Step 2: Run it — expect PASS (waits still in place; this pins behavior before the change)**

Run: `cmake --build build --target caliper_gfx_tests && ./build/tests/caliper_gfx_tests -tc='*burst*'`
Expected: PASS.

- [x] **Step 3: Remove the two waits**

In `colormap_compute` (`metal_renderer.mm:303-304`), replace:

```objc
        [cb commit];
        [cb waitUntilCompleted];   // texture must be ready for the frame/readback
```

with:

```objc
        // No CPU wait (M1/D23): the frame's command buffer commits AFTER this
        // one on the same queue_, so draw ordering is free by commit order; cb
        // retains src, lutbuf, and dst until the GPU retires it (default,
        // non-`unretained` encoding), so lifetime is free too. Test readbacks
        // retire the queue themselves (debug_readback_rgba8).
        [cb commit];
```

In `blit_u8` (`metal_renderer.mm:333-334`), replace:

```objc
        [cb commit];
        [cb waitUntilCompleted];
```

with:

```objc
        [cb commit];   // no CPU wait (M1/D23): same-queue commit order + retention
```

- [x] **Step 4: Run the full gfx suite — byte-exact with no waits**

Run: `./build/tests/caliper_gfx_tests` (after rebuild)
Expected: PASS — all cases, burst included. Also grep-verify the exit criterion:
`grep -n waitUntilCompleted src/host/renderer/metal_renderer.mm` → the only hit is inside `debug_readback_rgba8`.

- [x] **Step 5: Commit**

```bash
git add src/host/renderer/metal_renderer.mm tests/gfx/gfx_main.cpp
git commit -m "feat(metal)!: drop per-op waitUntilCompleted from the device paths (M1, D23) + burst test"
```

---

### Task 3: M2 — `caliper.tensor_bridge.v1_1` header + ABI tests

**Files:**
- Create: `sdk/include/caliper/services/tensor_bridge_v1_1.h`
- Modify: `tests/test_abi.cpp` (new case beside the existing id-string checks)
- Modify: `tests/abi_c_check.c` (add the include so the header stays C-compilable)

**Interfaces:**
- Produces: `CALIPER_TENSOR_BRIDGE_V1_1` (`"caliper.tensor_bridge.v1_1"`), `CALIPER_BRIDGE_CAP_STREAM_ORDERED` (`1u << 0`), and `struct CaliperTensorBridgeV1_1` — the six v1 function pointers in identical order plus `uint32_t (*caps)(void)`. Tasks 5–7 consume these names exactly.

- [x] **Step 1: Write the failing ABI test**

In `tests/test_abi.cpp`, add `#include <caliper/services/tensor_bridge_v1_1.h>` and `#include <cstddef>` next to the existing includes, and this case beside the v1 id-string check (line 62):

```cpp
TEST_CASE("tensor_bridge v1_1 is an additive, prefix-compatible superset of v1 (D24)") {
    CHECK(std::string(CALIPER_TENSOR_BRIDGE_V1_1) == "caliper.tensor_bridge.v1_1");
    CHECK(CALIPER_BRIDGE_CAP_STREAM_ORDERED == (1u << 0));
    // Same table: every v1 member sits at the same offset in v1_1.
    CHECK(offsetof(CaliperTensorBridgeV1_1, struct_size) ==
          offsetof(CaliperTensorBridgeV1, struct_size));
    CHECK(offsetof(CaliperTensorBridgeV1_1, texture_from_tensor) ==
          offsetof(CaliperTensorBridgeV1, texture_from_tensor));
    CHECK(offsetof(CaliperTensorBridgeV1_1, update_texture) ==
          offsetof(CaliperTensorBridgeV1, update_texture));
    CHECK(offsetof(CaliperTensorBridgeV1_1, release_texture) ==
          offsetof(CaliperTensorBridgeV1, release_texture));
    CHECK(offsetof(CaliperTensorBridgeV1_1, texture_from_tensor_mapped) ==
          offsetof(CaliperTensorBridgeV1, texture_from_tensor_mapped));
    CHECK(offsetof(CaliperTensorBridgeV1_1, alloc_shared) ==
          offsetof(CaliperTensorBridgeV1, alloc_shared));
    CHECK(offsetof(CaliperTensorBridgeV1_1, free_shared) ==
          offsetof(CaliperTensorBridgeV1, free_shared));
    // Plus exactly one query at the end.
    CHECK(sizeof(CaliperTensorBridgeV1_1) ==
          offsetof(CaliperTensorBridgeV1_1, caps) + sizeof(void*));
}
```

In `tests/abi_c_check.c`, add `#include <caliper/services/tensor_bridge_v1_1.h>` next to the existing `tensor_bridge_v1.h` include.

- [x] **Step 2: Run to verify it fails**

Run: `cmake --build build --target caliper_tests`
Expected: FAIL to compile — `tensor_bridge_v1_1.h: file not found`.

- [x] **Step 3: Write the header**

Create `sdk/include/caliper/services/tensor_bridge_v1_1.h`:

```c
#pragma once
/* caliper.tensor_bridge.v1_1 — ADDITIVE revision of tensor_bridge.v1 (D24,
 * docs/metal-pipelining.md §4): the SAME six operations, prefix-identical
 * layout, plus one query — caps(). Bit 0 set means the host honors
 * stream-ordered handoff: a non-NULL CaliperTensor.stream orders the device
 * update on the producer's stream/queue (CUstream on CUDA, MTLCommandQueue*
 * on Metal), so the adapter may SKIP its full device drain. Hosts that don't
 * vend this id keep the v1 contract: adapters drain, stream stays NULL. The
 * v1 header, table, and id are untouched (frozen); no ABI epoch bump. */
#include <caliper/services/tensor_bridge_v1.h>

#define CALIPER_TENSOR_BRIDGE_V1_1 "caliper.tensor_bridge.v1_1"

/* caps() bit 0: non-NULL CaliperTensor.stream is honored — producer-stream
 * GPU ordering replaces the adapter's device drain. Adapters must treat a
 * missing bit (or a missing v1_1 service) as "drain as v1". */
#define CALIPER_BRIDGE_CAP_STREAM_ORDERED (1u << 0)

#ifdef __cplusplus
extern "C" {
#endif

typedef struct CaliperTensorBridgeV1_1 {
    uint32_t struct_size;
    /* v1-identical prefix — same semantics as CaliperTensorBridgeV1. */
    CaliperTextureId (*texture_from_tensor)(const CaliperTensor* t, uint32_t flags);
    bool (*update_texture)(CaliperTextureId tex, const CaliperTensor* t);
    void (*release_texture)(CaliperTextureId tex);
    CaliperTextureId (*texture_from_tensor_mapped)(const CaliperTensor* t,
                                                   int32_t colormap,
                                                   float vmin, float vmax,
                                                   uint32_t flags);
    bool (*alloc_shared)(CaliperDType dtype, int32_t ndim, const int64_t* shape,
                         CaliperTensor* out_tensor, CaliperTextureId* out_texture);
    void (*free_shared)(CaliperTextureId tex);
    /* v1.1 addition: capability bits (CALIPER_BRIDGE_CAP_*). */
    uint32_t (*caps)(void);
} CaliperTensorBridgeV1_1;

#ifdef __cplusplus
}
#endif
```

- [x] **Step 4: Run to verify it passes**

Run: `cmake --build build --target caliper_tests && ./build/tests/caliper_tests`
Expected: PASS (all unit cases, new ABI case included).

- [x] **Step 5: Commit**

```bash
git add sdk/include/caliper/services/tensor_bridge_v1_1.h tests/test_abi.cpp tests/abi_c_check.c
git commit -m "feat(sdk): caliper.tensor_bridge.v1_1 — additive caps() query for stream-ordered handoff (D24)"
```

---

### Task 4: M2 — renderer capability signal + `TensorBridge::caps()`

**Files:**
- Modify: `src/host/renderer/host_renderer.h` (new virtual)
- Modify: `src/host/tensor_bridge.h`, `src/host/tensor_bridge.cpp` (caps())
- Test: `tests/test_tensor_bridge.cpp`

**Interfaces:**
- Consumes: `CALIPER_BRIDGE_CAP_STREAM_ORDERED` (Task 3).
- Produces: `virtual bool HostRenderer::honors_stream_ordered_handoff() const { return false; }` and `uint32_t TensorBridge::caps() const`. Task 5's `br_caps` thunk and Tasks 8–9's overrides rely on these exact names.

- [x] **Step 1: Write the failing unit test**

In `tests/test_tensor_bridge.cpp`, add after the existing StubRenderer-based cases:

```cpp
TEST_CASE("bridge caps() surfaces the renderer's stream-handoff capability (D24)") {
    // Default renderer: no stream honor -> caps 0 (adapters must drain).
    StubRenderer plain("gl");
    TensorBridge b_plain(plain);
    CHECK(b_plain.caps() == 0u);

    // A backend that honors the stream channel -> bit 0.
    class StreamStub : public StubRenderer {
    public:
        using StubRenderer::StubRenderer;
        bool honors_stream_ordered_handoff() const override { return true; }
    };
    StreamStub honored("metal");
    TensorBridge b_honored(honored);
    CHECK(b_honored.caps() == CALIPER_BRIDGE_CAP_STREAM_ORDERED);
}
```

- [x] **Step 2: Run to verify it fails**

Run: `cmake --build build --target caliper_tests`
Expected: FAIL to compile — `honors_stream_ordered_handoff` / `caps` not declared.

- [x] **Step 3: Implement**

In `src/host/renderer/host_renderer.h`, after the `debug_readback_rgba8` virtual (line 64):

```cpp
    // D24 (docs/metal-pipelining.md §4): true when this backend honors a
    // non-NULL CaliperTensor.stream by GPU-ordering the device update after
    // the producer's stream/queue. Surfaced to applets as bridge-v1.1 caps()
    // bit 0. Default false: a backend that ignores stream must never let an
    // adapter skip its drain.
    virtual bool honors_stream_ordered_handoff() const { return false; }
```

In `src/host/tensor_bridge.h`: add `#include <caliper/services/tensor_bridge_v1_1.h>` next to the v1 include, and declare in the public section of `TensorBridge`:

```cpp
    // Bridge-v1.1 capability bits (D24). Bit 0 = the active renderer honors
    // stream-ordered handoff, so adapters may skip the device drain.
    uint32_t caps() const;
```

In `src/host/tensor_bridge.cpp`, after the constructor:

```cpp
uint32_t TensorBridge::caps() const {
    return renderer_.honors_stream_ordered_handoff()
        ? CALIPER_BRIDGE_CAP_STREAM_ORDERED : 0u;
}
```

- [x] **Step 4: Run to verify it passes**

Run: `cmake --build build --target caliper_tests && ./build/tests/caliper_tests`
Expected: PASS.

- [x] **Step 5: Commit**

```bash
git add src/host/renderer/host_renderer.h src/host/tensor_bridge.h src/host/tensor_bridge.cpp tests/test_tensor_bridge.cpp
git commit -m "feat(host): HostRenderer stream-handoff capability + TensorBridge::caps() (D24)"
```

---

### Task 5: M2 — register `caliper.tensor_bridge.v1_1` in the services registry

`host_services.cpp` compiles only into the `caliper` exe (headless-unit-testable host_lib excludes it), so this task is verified by the exe compiling plus the ABI/sugar tests around it; the thunk is a two-liner over Task 4's tested `caps()`.

**Files:**
- Modify: `src/host/host_services.cpp`

**Interfaces:**
- Consumes: `TensorBridge::caps()` (Task 4), `CaliperTensorBridgeV1_1` (Task 3).
- Produces: `services_get(CALIPER_TENSOR_BRIDGE_V1_1)` returns a `CaliperTensorBridgeV1_1*`; the id appears in `service_ids()`. Task 6's sugar and real applets consume this.

- [x] **Step 1: Implement**

In `src/host/host_services.cpp`: add `#include <caliper/services/tensor_bridge_v1_1.h>` next to the v1 include (line 17). After `kBridge` (line 243), add:

```cpp
// v1.1 (D24): the same six thunks plus caps(). Bit 0 reflects the ACTIVE
// renderer (Metal/Vulkan honor it once M2 lands there; GL and headless never
// do) — 0 with no renderer bound, so adapters keep draining.
uint32_t br_caps(void) {
    TensorBridge* b = bridge();
    return b ? b->caps() : 0u;
}
const CaliperTensorBridgeV1_1 kBridge11 = {sizeof(CaliperTensorBridgeV1_1),
    &br_texture_from_tensor, &br_update_texture, &br_release_texture,
    &br_texture_from_tensor_mapped, &br_alloc_shared, &br_free_shared,
    &br_caps};
```

Add `CALIPER_TENSOR_BRIDGE_V1_1,` to `kIds` (line 245-249, after `CALIPER_TENSOR_BRIDGE_V1,`), and in `services_get` (line 320):

```cpp
    if (std::strcmp(id, CALIPER_TENSOR_BRIDGE_V1_1) == 0) return &kBridge11;
```

- [x] **Step 2: Verify the exe builds and the unit suite still passes**

Run: `cmake --build build --target caliper caliper_tests && ./build/tests/caliper_tests`
Expected: both build; unit suite PASS (kIds only widens the host's offer — negotiation of applet-required ids is unaffected).

- [x] **Step 3: Commit**

```bash
git add src/host/host_services.cpp
git commit -m "feat(host): vend caliper.tensor_bridge.v1_1 (caps thunk) from the services registry (D24)"
```

---

### Task 6: M2 — sugar: `caliper::Bridge::caps()`

**Files:**
- Modify: `sdk/include/caliper/caliper.hpp` (Bridge class, lines 256-297)
- Test: `tests/test_sugar_services.cpp`

**Interfaces:**
- Consumes: `CALIPER_TENSOR_BRIDGE_V1_1` service lookup (fake host in tests, real host at runtime).
- Produces: `uint32_t Bridge::caps() const` — 0 on v1-only/headless hosts. Task 10's applet call sites consume this.

- [x] **Step 1: Write the failing test**

In `tests/test_sugar_services.cpp`, find the fixture that provides `CALIPER_TENSOR_BRIDGE_V1` (line 354) and add a sibling case (reuse the file's existing fixture type and fake-table idiom — mirror how `kFakeBridge` is declared there):

```cpp
static uint32_t fake_caps(void) { return CALIPER_BRIDGE_CAP_STREAM_ORDERED; }

TEST_CASE("sugar Bridge::caps(): v1_1 bit surfaces; absent service -> 0") {
    ServiceFixture fx;                       // the file's existing fixture type
    // v1-only host: caps() must be 0 (adapters drain).
    fx.provide(CALIPER_TENSOR_BRIDGE_V1, &kFakeBridge);
    {
        caliper::Bridge b(fx.host());
        CHECK(b.caps() == 0u);
    }
    // v1.1 host: the bit crosses.
    static const CaliperTensorBridgeV1_1 kFake11 = {sizeof(CaliperTensorBridgeV1_1),
        kFakeBridge.texture_from_tensor, kFakeBridge.update_texture,
        kFakeBridge.release_texture, kFakeBridge.texture_from_tensor_mapped,
        kFakeBridge.alloc_shared, kFakeBridge.free_shared, &fake_caps};
    fx.provide(CALIPER_TENSOR_BRIDGE_V1_1, &kFake11);
    {
        caliper::Bridge b(fx.host());
        CHECK(b.caps() == CALIPER_BRIDGE_CAP_STREAM_ORDERED);
    }
}
```

(Adapt fixture/ctor names to the file's actual idiom at line 354 — the assertions and table shape above are the contract.)

- [x] **Step 2: Run to verify it fails**

Run: `cmake --build build --target caliper_tests`
Expected: FAIL to compile — `caps` is not a member of `caliper::Bridge`.

- [x] **Step 3: Implement**

In `sdk/include/caliper/caliper.hpp`: add `#include <caliper/services/tensor_bridge_v1_1.h>` next to the `tensor_bridge_v1.h` include (line 12). In `class Bridge`:

```cpp
    explicit Bridge(const Host& host)
        : t_(static_cast<const CaliperTensorBridgeV1*>(
              host.service(CALIPER_TENSOR_BRIDGE_V1))),
          t11_(static_cast<const CaliperTensorBridgeV1_1*>(
              host.service(CALIPER_TENSOR_BRIDGE_V1_1))) {}
```

after `free_shared`:

```cpp
    // v1.1 capability bits — 0 on a v1-only or headless host (D24). Query
    // once per handoff site and pass to adapters::stream_to_tensor; bit
    // CALIPER_BRIDGE_CAP_STREAM_ORDERED means a non-NULL CaliperTensor.stream
    // replaces the adapter's device drain.
    uint32_t caps() const { return (t11_ && t11_->caps) ? t11_->caps() : 0u; }
```

and the member:

```cpp
    const CaliperTensorBridgeV1_1* t11_ = nullptr;
```

- [x] **Step 4: Run to verify it passes**

Run: `cmake --build build --target caliper_tests && ./build/tests/caliper_tests`
Expected: PASS.

- [x] **Step 5: Commit**

```bash
git add sdk/include/caliper/caliper.hpp tests/test_sugar_services.cpp
git commit -m "feat(sdk): Bridge::caps() sugar over tensor_bridge.v1_1 (D24)"
```

---

### Task 7: M2 — adapter: `stream_to_tensor` in `adapters/torch.hpp`

**Files:**
- Modify: `sdk/include/caliper/adapters/torch.hpp`
- Test: `tests/test_torch_adapter.cpp`

**Interfaces:**
- Consumes: `CALIPER_BRIDGE_CAP_STREAM_ORDERED` (Task 3); `synced_to_tensor` / `to_tensor` (existing).
- Produces: `std::optional<CaliperTensor> caliper::adapters::stream_to_tensor(const at::Tensor& t, uint32_t bridge_caps)`. Contract: caps bit absent → identical to `synced_to_tensor` (drain); bit present → MPS commits-without-waiting and sets `stream` to the producer `MTLCommandQueue*`, CUDA sets `stream` to the current `cudaStream_t`, CPU stays `stream == NULL` with no sync. Task 10 consumes this.

- [x] **Step 1: Write the failing tests**

In `tests/test_torch_adapter.cpp`, add `using caliper::adapters::stream_to_tensor;` next to the existing usings, and:

```cpp
TEST_CASE("stream_to_tensor: no caps bit -> exactly the v1 drained handoff (stream NULL)") {
    torch::Tensor t = torch::arange(12, torch::kFloat).reshape({3, 4}).contiguous();
    auto ct = stream_to_tensor(t, 0);
    REQUIRE(ct.has_value());
    CHECK(ct->stream == nullptr);
    CHECK(ct->device == CALIPER_DEV_CPU);
    CHECK(ct->data == t.data_ptr());
}

TEST_CASE("stream_to_tensor: cpu tensor never carries a stream, even when honored") {
    torch::Tensor t = torch::arange(12, torch::kFloat).reshape({3, 4}).contiguous();
    auto ct = stream_to_tensor(t, CALIPER_BRIDGE_CAP_STREAM_ORDERED);
    REQUIRE(ct.has_value());
    CHECK(ct->stream == nullptr);
}

TEST_CASE("stream_to_tensor: mps tensor carries the producer queue when honored; drains when not") {
    if (!torch::mps::is_available()) { MESSAGE("no MPS device — skipping"); return; }
    torch::Tensor t = torch::ones({4, 4},
        torch::TensorOptions().device(torch::kMPS)) * 2.0f;

    auto honored = stream_to_tensor(t, CALIPER_BRIDGE_CAP_STREAM_ORDERED);
    REQUIRE(honored.has_value());
    CHECK(honored->device == CALIPER_DEV_METAL);
    CHECK(honored->stream != nullptr);           // the MTLCommandQueue*

    auto v1 = stream_to_tensor(t, 0);            // negotiation pin, other direction
    REQUIRE(v1.has_value());
    CHECK(v1->stream == nullptr);
}
```

- [x] **Step 2: Run to verify it fails**

Run: `cmake --build build --target caliper_torch_tests`
Expected: FAIL to compile — `stream_to_tensor` not declared.

- [x] **Step 3: Implement**

In `sdk/include/caliper/adapters/torch.hpp`, extend the includes:

```cpp
#include <caliper/services/tensor_bridge_v1_1.h>
#if defined(__APPLE__)
#include <objc/message.h>   // producer-queue lookup without an ObjC++ TU
#endif
#if __has_include(<c10/cuda/CUDAStream.h>)
#include <c10/cuda/CUDAStream.h>
#endif
```

In `namespace detail`, after `map_dtype`:

```cpp
#if defined(__APPLE__)
// [cb commandQueue] via the C ObjC runtime, so this header stays compilable
// as plain C++ (applet TUs are .cpp, not .mm). The queue is torch's global
// MPS command queue — process-lifetime, safe to hand across the ABI.
inline void* mtl_command_queue_of(void* command_buffer) {
    using Send = void* (*)(void*, SEL);
    return command_buffer
        ? ((Send)objc_msgSend)(command_buffer, sel_registerName("commandQueue"))
        : nullptr;
}
#endif
```

After `synced_to_tensor` (line 114), add:

```cpp
// M2 (docs/metal-pipelining.md §4, D24): hand over ORDER instead of a drained
// device. When the host's bridge-v1.1 caps carry
// CALIPER_BRIDGE_CAP_STREAM_ORDERED, populate t.stream with the producer's
// stream/queue and SKIP the full-device drain; the renderer GPU-orders its
// update after the producer's already-enqueued work. Without the bit (v1
// host, GL fallback, headless) this is exactly synced_to_tensor — the adapter
// never skips a drain the host didn't promise to replace. Thread-safety story
// unchanged from v1: the caller still hands over a tensor it owns at a
// quiescent point in its own logic (spec §4, last paragraph).
inline std::optional<CaliperTensor> stream_to_tensor(const at::Tensor& t,
                                                     uint32_t bridge_caps) {
    if (!(bridge_caps & CALIPER_BRIDGE_CAP_STREAM_ORDERED))
        return synced_to_tensor(t);
#if defined(__APPLE__)
    if (t.is_mps()) {
        auto out = to_tensor(t);
        if (!out) return out;
        // Queue BEFORE commit: torch may release the command-buffer object
        // once the GPU retires it; the queue lives for the process.
        void* queue = detail::mtl_command_queue_of(torch::mps::get_command_buffer());
        if (queue == nullptr) { torch::mps::synchronize(); return out; }
        // Enqueue, not drain: pending torch kernels become committed GPU work
        // the renderer's producer-queue signal is ordered after (M2b).
        torch::mps::commit();
        out->stream = queue;
        return out;
    }
#endif
#if __has_include(<c10/cuda/CUDAStream.h>)
    if (t.is_cuda()) {
        auto out = to_tensor(t);
        if (!out) return out;
        // Stream order puts the renderer's DtoD copy after the producer's
        // kernels — torch::cuda::synchronize() elided entirely (M2a).
        out->stream = (void*)at::cuda::getCurrentCUDAStream(
                          t.device().index()).stream();
        return out;
    }
#endif
    return synced_to_tensor(t);   // CPU: no sync needed; unknown devices drain
}
```

Also update the header's top comment block, third bullet: after "v1 device story: `stream == NULL`…", append one line: `stream_to_tensor (M2/D24) supersedes this when the host's bridge-v1.1 caps() grants stream-ordered handoff.`

- [x] **Step 4: Run to verify it passes**

Run: `cmake --build build --target caliper_torch_tests && ./build/tests/caliper_torch_tests`
Expected: PASS (MPS cases run on this machine; CUDA branch is compiled out — no `c10/cuda` headers in the mac libtorch).

- [x] **Step 5: Commit**

```bash
git add sdk/include/caliper/adapters/torch.hpp tests/test_torch_adapter.cpp
git commit -m "feat(sdk): adapters::stream_to_tensor — caps-gated stream handoff, drain-by-default (M2, D24)"
```

---

### Task 8: M2b — Metal renderer honors the stream channel (per-texture `MTLSharedEvent`)

**Files:**
- Modify: `src/host/renderer/metal_renderer.mm`
- Test: `tests/gfx/gfx_main.cpp` (gated-producer ordering test)

**Interfaces:**
- Consumes: `t.stream` as `MTLCommandQueue*` (Task 7's contract); `honors_stream_ordered_handoff()` virtual (Task 4).
- Produces: `MetalRenderer::honors_stream_ordered_handoff()` returns `true`; a non-NULL `t.stream` GPU-orders the texture update after the producer queue's committed work with no CPU block.

- [x] **Step 1: Write the failing gfx test (deterministic via a CPU-releasable gate)**

Add to `tests/gfx/gfx_main.cpp` inside `#ifdef CALIPER_HAVE_METAL`, after the burst test:

```objc
// M2b: a non-NULL t.stream (the producer's MTLCommandQueue*) must GPU-order
// the update AFTER the producer's committed work. Deterministic, no timing
// luck: the producer's payload write is gated behind an MTLSharedEvent the
// TEST only fires after update_texture returns. A renderer that ignores
// t.stream colormaps the stale bytes (fails); one that orders reads the fresh.
TEST_CASE("gfx/Metal: non-NULL stream orders the update after the producer queue") {
    if (!metal_env().ok) { MESSAGE("no Metal device — skipping"); return; }
    Backend bk = metal_backend();
    REQUIRE(bk.renderer->honors_stream_ordered_handoff());

    const int w = 8, h = 8, n = w * h;
    std::vector<float> stale(n, 0.0f);
    std::vector<float> fresh(n);
    for (int i = 0; i < n; ++i) fresh[i] = (float)i;

    id<MTLBuffer> tensor_buf = device_buffer(stale.data(), (size_t)n * sizeof(float));
    id<MTLBuffer> payload    = device_buffer(fresh.data(), (size_t)n * sizeof(float));

    CaliperTensor t{};
    t.struct_size = sizeof(t);
    t.data = (__bridge void*)tensor_buf;
    t.dtype = CALIPER_DT_F32;
    t.ndim = 2; t.shape[0] = h; t.shape[1] = w; t.strides[0] = w; t.strides[1] = 1;
    t.device = CALIPER_DEV_METAL;

    CaliperTextureId id = bk.bridge->texture_from_tensor_mapped(
        &t, CALIPER_CMAP_VIRIDIS, 0.0f, (float)(n - 1), 0);
    REQUIRE(id != 0);

    // Producer queue with pending committed work: a payload blit blocked on a
    // gate the CPU holds — the stand-in for torch's just-committed kernels.
    id<MTLDevice> dev = metal_env().device;
    id<MTLCommandQueue> producer = [dev newCommandQueue];
    id<MTLSharedEvent> gate = [dev newSharedEvent];
    id<MTLCommandBuffer> pc = [producer commandBuffer];
    [pc encodeWaitForEvent:gate value:1];
    id<MTLBlitCommandEncoder> pb = [pc blitCommandEncoder];
    [pb copyFromBuffer:payload sourceOffset:0
              toBuffer:tensor_buf destinationOffset:0
                  size:(NSUInteger)n * sizeof(float)];
    [pb endEncoding];
    [pc commit];

    // Handoff with the producer queue in t.stream and NO drain anywhere.
    t.stream = (__bridge void*)producer;
    REQUIRE(bk.bridge->update_texture(id, &t));

    gate.signaledValue = 1;   // only NOW may the producer's write run

    std::vector<uint8_t> ref((size_t)n * 4);
    map_f32_to_rgba8(fresh.data(), w, h, colormap_lut(CALIPER_CMAP_VIRIDIS),
                     0.0f, (float)(n - 1), ref.data());
    CHECK(bk.readback(id, w, h) == ref);
    bk.bridge->release_texture(id);
}
```

- [x] **Step 2: Run to verify it fails**

Run: `cmake --build build --target caliper_gfx_tests && ./build/tests/caliper_gfx_tests -tc='*stream orders*'`
Expected: FAIL at `REQUIRE(bk.renderer->honors_stream_ordered_handoff())` (clean red — no GPU race involved).

- [x] **Step 3: Implement in `metal_renderer.mm`**

Add `#include <unordered_map>` to the includes. Add members next to `textures_`:

```objc
    // M2b: per-texture producer-ordering events (D23 — MTLSharedEvent appears
    // ONLY where cross-queue ordering genuinely exists). Values are a per-
    // texture monotonic timeline, the Metal analog of Vulkan's semaphores.
    NSMutableDictionary<NSNumber*, id<MTLSharedEvent>>* events_ = nil;
    std::unordered_map<uint64_t, uint64_t> event_values_;
```

In `init()`, next to `textures_ = …`: `events_ = [NSMutableDictionary dictionary];`
In `shutdown()`, before `textures_` teardown: `[events_ removeAllObjects]; events_ = nil; event_values_.clear();`
In `tex_release()`: add `[events_ removeObjectForKey:@(tex)]; event_values_.erase(tex);`

Add the private helper (after `ensure_pipeline`):

```objc
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
```

Thread the internal texture id into the two op helpers and hook the wait in. Change the signatures and the call site:

```objc
    bool tex_update_from_device(uint64_t tex, const CaliperTensor& t,
                                const uint32_t* lut256,
                                float vmin, float vmax) override {
        id<MTLTexture> dst = lookup(tex);
        if (dst == nil) return false;
        if (t.device != CALIPER_DEV_METAL || t.data == nullptr) return false;
        id<MTLBuffer> src = (__bridge id<MTLBuffer>)t.data;
        if (src == nil) return false;

        if (t.dtype == CALIPER_DT_F32 && lut256 != nullptr)
            return colormap_compute(tex, dst, src, t, lut256, vmin, vmax);
        if (t.dtype == CALIPER_DT_U8)
            return blit_u8(tex, dst, src, t);
        return false;
    }
```

In `colormap_compute(uint64_t tex, id<MTLTexture> dst, id<MTLBuffer> src, const CaliperTensor& t, const uint32_t* lut256, float vmin, float vmax)`, after `id<MTLCommandBuffer> cb = [queue_ commandBuffer];` and BEFORE `computeCommandEncoder` (waits must be encoded outside encoder scopes):

```objc
        if (t.stream != nullptr) order_after_producer(tex, cb, t.stream);
```

Same one-liner in `blit_u8(uint64_t tex, id<MTLTexture> dst, id<MTLBuffer> src, const CaliperTensor& t)` after its `cb` creation, before `blitCommandEncoder`.

Finally, advertise the capability (next to `interop_device()`):

```objc
    // M2b shipped: a non-NULL t.stream is GPU-ordered after the producer
    // queue in both device paths above (D24).
    bool honors_stream_ordered_handoff() const override { return true; }
```

- [x] **Step 4: Run the full gfx suite**

Run: `./build/tests/caliper_gfx_tests` (after rebuild)
Expected: PASS — the new ordering test green, everything else byte-exact and unmodified (NULL-stream cases prove the fallback rung).

- [x] **Step 5: Commit**

```bash
git add src/host/renderer/metal_renderer.mm tests/gfx/gfx_main.cpp
git commit -m "feat(metal): stream-ordered handoff — per-texture MTLSharedEvent orders updates after the producer queue (M2b, D24)"
```

---

### Task 9: M2a — Vulkan renderer orders on `t.stream` (code-only here; Windows verification deferred)

`vulkan_renderer.cpp` does not compile on macOS. Make the minimal, mechanical edit V4 was designed for; verification runs on the Windows box (checklist below).

**Files:**
- Modify: `src/host/renderer/vulkan_renderer.cpp:1243-1255` (`update_pipelined`), `:33` (header comment), plus the capability override next to `interop_device()` (line 176).

**Interfaces:**
- Consumes: `t.stream` as `cudaStream_t` (Task 7); `pipelined_ok_` member (existing, line 164).
- Produces: the pipelined copy+signal run on the producer stream; `honors_stream_ordered_handoff()` returns `pipelined_ok_` (the sync fallback reports false, so adapters drain there — strictest reading of D24).

- [x] **Step 1: Implement**

In `update_pipelined`, replace the CUDA-side block (lines 1243-1255) with:

```cpp
        // CUDA side, ordered on the PRODUCER's stream when the adapter
        // supplied one (M2a, D24) — stream order puts the copy after the
        // producer's kernels, so the adapter's torch::cuda::synchronize() is
        // elided. NULL keeps the legacy default stream (v1 drained handoff).
        const cudadrv::Api* cu = cudadrv::api();
        cudadrv::CUstream stream = (cudadrv::CUstream)t.stream;
        const uint64_t base = io.timeline_value;
        if (!shared_in_place &&
            cu->cuMemcpyDtoDAsync(io.cuda_ptr, src, (size_t)bytes, stream)
                != cudadrv::CUDA_SUCCESS)
            return dev_bail("pipelined: cuMemcpyDtoDAsync failed");
        cudadrv::ExternalSemaphoreSignalParams sp{};
        sp.params.fence.value = base + 1;
        if (cu->cuSignalExternalSemaphoresAsync(&io.cuda_sem, &sp, 1, stream)
                != cudadrv::CUDA_SUCCESS)
            return dev_bail("pipelined: semaphore signal failed");
```

Next to `interop_device()` (line 176), add:

```cpp
    // M2a (D24): only the pipelined path GPU-orders after t.stream; the
    // synchronous fallback must keep the v1 drained contract, so advertise
    // stream handoff only when pipelining is actually live.
    bool honors_stream_ordered_handoff() const override { return pipelined_ok_; }
```

Update the file-top comment at line 33: replace `eliding it needs a stream channel in CaliperTensor (a v2 ABI question).` with `elided when the adapter populates CaliperTensor.stream under bridge-v1.1 caps (M2a, D24) — the copy+signal then ride the producer's stream.`

- [x] **Step 2: Syntax sanity check (no Windows toolchain here)**

Run: `python3 -c "print(open('src/host/renderer/vulkan_renderer.cpp').read().count('cuMemcpyDtoDAsync'))"` → expect `2` (header comment + call), and re-read the edited hunks for balance.
Expected: edits are local, no other `nullptr` stream args remain in `update_pipelined`.

- [x] **Step 3: Record the deferred Windows verification checklist**

Append to the plan's own Notes section (bottom of this file):
- [ ] Windows: `caliper_gfx_tests` green including burst test (NULL-stream fallback = every existing case).
- [ ] Windows: gpt_scope/embed_scope training shows no `torch::cuda::synchronize` in the handoff profile (spec §6 M2a exit).

- [x] **Step 4: Commit**

```bash
git add src/host/renderer/vulkan_renderer.cpp docs/superpowers/plans/2026-07-05-metal-pipelining.md
git commit -m "feat(vulkan): pipelined copy+signal ride the producer CUDA stream from t.stream (M2a, D24) — Windows verify deferred"
```

---

### Task 10: switch the applet handoff sites to `stream_to_tensor`

**Files:**
- Modify: `applets/embed_scope/embed_scope.cpp:905-925` (the two `synced_to_tensor` sites + comments), `:454-456` (the stale cost comment — only after measuring)
- Modify: `applets/gpt_scope/gpt_scope.cpp:338-352` (`upload_mapped`)

**Interfaces:**
- Consumes: `Bridge::caps()` (Task 6), `stream_to_tensor` (Task 7).
- Produces: no `torch::mps::synchronize()` at the handoff when the host honors streams; v1 drain preserved everywhere else.

- [x] **Step 1: embed_scope**

At `applets/embed_scope/embed_scope.cpp:905-918`, replace the two call sites:

```cpp
            bool device_rejected = false;
            // stream_: with a stream-honoring host (bridge v1.1 caps bit 0)
            // the handoff rides the producer queue — no full-device drain.
            // Otherwise this IS synced_to_tensor: drain the producer before
            // the device upload reads it (no-op on CPU).
            const uint32_t bcaps = st->bridge.caps();
            for (int k = 0; k < 8; k++) {
                auto ct = caliper::adapters::stream_to_tensor(disp_conv[k], bcaps);
                st->conv_tex[k] = ct ? st->bridge.texture_from_tensor_mapped(
                                           &*ct, CALIPER_CMAP_MAGMA,
                                           -w_km, w_km) : 0;
                device_rejected |= (st->conv_tex[k] == 0);
            }
            {
                auto ct = caliper::adapters::stream_to_tensor(disp_embw, bcaps);
                st->embw_tex = ct ? st->bridge.texture_from_tensor_mapped(
                                        &*ct, CALIPER_CMAP_RDBU,
                                        -w_wm, w_wm) : 0;
                device_rejected |= (st->embw_tex == 0);
            }
```

(The old 4-line `// synced_:` comment above the loop is superseded by the new one.)

- [x] **Step 2: gpt_scope**

In `upload_mapped` (`gpt_scope.cpp:338-352`), replace the device branch of `view`:

```cpp
    auto view = [&](bool cpu) -> std::optional<CaliperTensor> {
        if (cpu) { host_t = dev_t.to(torch::kCPU);
                   return caliper::adapters::to_tensor(host_t); }
        // Device handoff (M2/D24): with a stream-honoring host the bridge's
        // in-VRAM copy is GPU-ordered after the producer stream — no drain.
        // On a v1 host this drains (synced) exactly as before.
        return caliper::adapters::stream_to_tensor(dev_t, bridge.caps());
    };
```

- [x] **Step 3: Build the applets + run every automated suite**

Run: `cmake --build build --target caliper && cmake --build build` then `./build/tests/caliper_tests && ./build/tests/caliper_gfx_tests && ./build/tests/caliper_torch_tests`
Expected: all PASS.

- [ ] **Step 4: Measure (spec §6 M2b exit: "retired with measurement")**

Run the app (`./build/caliper` or the run skill), open EmbedScope → Train, and compare steps/sec (the progress line shows step counts; time 100 steps by wall clock) against a stash of this task's diff (`git stash` / re-run / `git stash pop`). Record both numbers.
Expected: step time drops (the per-step `torch::mps::synchronize` is gone from the handoff). ONLY if measured: update the `embed_scope.cpp:454-456` comment to:

```cpp
            // Cloud stream: re-embed the full test subset EVERY step — cheap
            // now that the handoff is stream-ordered (M2b): the per-step MPS
            // drain that used to dominate step time is gone (measured
            // YYY→ZZZ steps/s on M-series, 2026-07).
```

(with the real numbers). If the measurement shows no improvement, leave the comment, note the numbers in the commit message, and flag it in the final report (spec §8 last risk).

- [x] **Step 5: Commit**

```bash
git add applets/embed_scope/embed_scope.cpp applets/gpt_scope/gpt_scope.cpp
git commit -m "feat(applets): stream-ordered tensor handoff via bridge v1.1 caps (M2) — drain elided on honoring hosts"
```

---

### Task 11: documentation sweep

**Files:**
- Modify: `docs/metal-pipelining.md` (status header + §6 table: M1 ✅, M2b ✅, M2a code-complete/Windows-verify-pending)
- Modify: `ZEROCOPY.md` (two places)

**Interfaces:** none (docs).

- [x] **Step 1: metal-pipelining.md**

Change the header `| **Status** | Draft for review |` to `| **Status** | Implemented — M1 + M2b shipped and gfx-verified on Apple Silicon; M2a code-complete, Windows hardware verification pending |`. In the §6 table, append to each row's exit-criterion cell: M1 — `**Shipped:** gfx suite + burst test byte-exact; sole remaining wait is the test readback.`; M2a — `**Code-complete:** stream pass-through + caps; Windows profile/gfx verification pending.`; M2b — `**Shipped:** gated-producer ordering test byte-exact; adapter commits + hands the queue.`

- [x] **Step 2: ZEROCOPY.md**

In the MPS sequence diagram (line 87), change the note `one torch::mps synchronize at the handoff —<br/>pending kernels finish before the GPU reads` to `handoff sync: v1 drains (torch::mps::synchronize); with bridge-v1.1 stream handoff the renderer GPU-orders<br/>after the producer queue instead (docs/metal-pipelining.md)`. In the Vulkan implementation-notes paragraph (line 144-145), replace `torch::cuda::synchronize() at the adapter is still the v1 producer→consumer barrier (the ABI has no stream channel yet).` with `The adapter's torch::cuda::synchronize() barrier is elided when bridge-v1.1 caps grant stream-ordered handoff — the copy+signal then ride the producer's CUDA stream (docs/metal-pipelining.md M2a).`

- [x] **Step 3: Verify diagrams still render**

Run: `grep -c '```mermaid' ZEROCOPY.md` → unchanged count (4); visually re-read the edited note lines for mermaid syntax (no stray `|`).

- [x] **Step 4: Commit**

```bash
git add docs/metal-pipelining.md ZEROCOPY.md
git commit -m "docs(metal-pipelining): mark M1/M2b shipped, M2a Windows-verify pending; ZEROCOPY handoff notes updated"
```

---

## Notes / Deferred (Windows)

Superseded by the full agent brief: `docs/m2a-windows-verification.md` (encodes the macOS session findings — the wrong-once CUDA guard, the proven handoff-vs-training thread race and its `545a2f7` fix — plus tasks T1–T7).

- [ ] Windows: `caliper_gfx_tests` green including the CUDA burst test with the new stream pass-through (NULL-stream fallback = every existing case passes unmodified).
- [ ] Windows: gpt_scope/embed_scope training profile shows no `torch::cuda::synchronize` in the handoff (spec §6 M2a exit).
- [ ] Windows: consider a `cuStreamCreate`-based non-default-stream gfx test (needs a new `cuda_driver.h` entry) — not required for M2a exit.
- [ ] Windows: verify the adapter's CUDA guard (`#if !defined(__APPLE__) && __has_include(<c10/cuda/CUDAStream.h>)`, torch.hpp) admits the branch there — Windows libtorch has both the c10/cuda headers AND the CUDA toolkit headers (mac ships only the former, which is why presence-of-header alone was not usability).

## Self-Review (done at planning time)

- Spec coverage: §3 (M1) → Tasks 1–2; §4 M2 contract → Tasks 7–9; capability negotiation → Tasks 3–6; §6 exit criteria → each task's run steps + Task 10 measurement + deferred Windows list; §5 "does not change" → Global Constraints; D23/D24 → comments landed at the code sites; §8 risks → readback fixed inside M1, caps-default-drain pinned by tests in Tasks 4/6/7, retention documented at encode sites (Task 2), torch-MPS API risk isolated in the adapter with a drain fallback (Task 7).
- Placeholder scan: Task 6 asks the implementer to adapt fixture names to the file's idiom — the assertions and fake-table shape are fully specified; everything else is complete code.
- Type consistency: `honors_stream_ordered_handoff` (Tasks 4/8/9), `caps()` (4/5/6), `stream_to_tensor(const at::Tensor&, uint32_t)` (7/10), `debug_readback_rgba8(uint64_t, int, int)` (1/2/8), `order_after_producer(uint64_t, id<MTLCommandBuffer>, void*)` (8 only) — checked consistent.
