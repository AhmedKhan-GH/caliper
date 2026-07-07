# Zero-Copy for Arbitrary CUDA Tensors (bridge import-allocation) — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Eliminate the one-in-VRAM-copy floor for torch CUDA tensors allocated from an applet-side exportable memory pool, by letting the bridge import the pool's allocation into Vulkan and run the buffer→image pass directly from it — true zero-copy for arbitrary *computations*, opt-in per allocation.

**Architecture:** Additive bridge capability (v1.1 precedent): a new caps bit + `import_allocation`/`release_allocation`/`update_texture_from_alloc` entry points; the frozen `CaliperTensor` struct is untouched. Applet side, a small SDK helper allocates torch tensors from a `cuMemCreate`-backed, shareable-handle-exporting pool (scoped — training memory stays on torch's caching allocator) and maps any tensor pointer to (allocation, offset). Host side, the Vulkan renderer imports the OS handle once per pool block (`VkImportMemoryWin32HandleInfoKHR`, OPAQUE_WIN32), binds a `VkBuffer` at the tensor's offset, and reuses the existing SPIR-V colormap / copy-to-image passes and per-texture timeline-semaphore ordering — the `cuMemcpyDtoD` stage simply disappears. Every failure (unsupported renderer, bad alignment, import failure) returns 0/false and falls back to the existing D2D-copy path.

**Tech Stack:** C (frozen SDK ABI), C++17 host, Vulkan (`VK_KHR_external_memory_win32`), CUDA driver API (VMM: `cuMemCreate`/`cuMemExportToShareableHandle` — applet side), libtorch (applet side only, D11), Catch2-style existing test harness, ctest labels for hardware gating.

## Global Constraints

- **No ABI break.** `CaliperTensor` is frozen (no storage-offset channel, no new fields). All additions are new entry points + a caps bit, following the exact v1→v1.1 additive pattern (recon to confirm mechanism).
- **The host never links torch (D11).** All torch/MemPool code lives in `sdk/include/caliper/adapters/` (header-only, applet-compiled). Host-side CUDA only via the runtime-loaded driver table (`src/host/cuda_driver.h`).
- **Byte-identical colormaps.** The frozen index rule is untouched; the import path must read back byte-equal to `map_f32_to_rgba8` in `gfx-cuda` tests.
- **Degradation ladder (§19).** Import failure at any rung → `0`/`false` → existing D2D interop path → CPU staging. Never a crash, never a wrong image, one `caliper.log.v1` line per rejection.
- **Wording discipline (§7.4).** "Zero-copy" only for paths with zero copies of the data; status lines must say which path ran.
- **macOS build unaffected.** All Vulkan/CUDA TUs stay behind existing gates; full mac `ctest` suite must stay green; new mac-runnable unit tests must not require CUDA.
- **TDD.** Every task: failing test first, minimal implementation, green, commit. Branch: `feat/zerocopy-arbitrary-cuda`.

---

## Task Map

| # | Task | Verifiable on macOS? |
|---|---|---|
| 1 | SDK v1.2 header (`CALIPER_BRIDGE_CAP_IMPORT_ALLOC`, `import_allocation`/`release_allocation`/`update_texture_from_alloc`) + `caliper.hpp` wrapper + layout tests | ✅ |
| 2 | Host bridge bookkeeping + seam virtuals + v1_2 table registration, stub-renderer TDD | ✅ |
| 3 | `AllocRegistry` — pure-C++ pointer→(allocation, offset) interval lookup | ✅ |
| 4 | `ExportablePool` — CUDA-gated torch MemPool over `cuMemCreate` shareable allocations + bridge glue + tripwire | ⚠️ compile-guard + tripwire only |
| 5 | Vulkan import path (`VkImportMemoryWin32HandleInfoKHR`, descriptor-offset binding, copy-free semaphore chain) | build-only (WIN32 TU) |
| 6 | Optional `VmmApi` driver table + five `gfx-cuda` hardware rows (byte-exact, offsets, misalign-fallback, release stress, bounds) | ⛔ Windows/NVIDIA box |
| 7 | gpt_scope exemplar opt-in + honest status line | ⛔ visual check on Windows |
| 8 | Docs: ZEROCOPY.md row, vulkan-cuda-backend.md V5 entry, WHITEPAPER.md §9 | ✅ |

Tasks 1–4 and 8 are fully executable and verifiable in this macOS session. Tasks 5–7 build here (gated TUs / cap-absent fallbacks), verify on the Windows/NVIDIA box per the `m2a-windows-verification.md` pattern.

---

### Task 1: SDK v1.2 header + C++ wrapper + layout tests

**Files:**
- Create: `sdk/include/caliper/services/tensor_bridge_v1_2.h`
- Modify: `sdk/include/caliper/caliper.hpp` (Bridge wrapper, ~line 257–306)
- Test: `tests/test_abi.cpp` (append layout case)

**Interfaces:**
- Consumes: `CaliperTensorBridgeV1_1` (six v1 ops + `caps`), `CaliperTensor`, `CaliperTextureId`.
- Produces: `CALIPER_TENSOR_BRIDGE_V1_2` = `"caliper.tensor_bridge.v1_2"`, `CALIPER_BRIDGE_CAP_IMPORT_ALLOC` = `(1u << 1)`, `CaliperAllocId` (uint64, 0 = invalid), `CALIPER_ALLOC_HANDLE_OPAQUE_WIN32` = 1, `CALIPER_ALLOC_HANDLE_OPAQUE_FD` = 2, struct `CaliperTensorBridgeV1_2` (v1.1-prefix-identical + 3 new fn pointers), `caliper::Bridge::import_allocation/release_allocation/update_texture_from_alloc`.

- [ ] **Step 1: Write the failing layout test** (append to `tests/test_abi.cpp`):

```cpp
#include <caliper/services/tensor_bridge_v1_2.h>
#include <cstddef>

TEST_CASE("tensor_bridge v1_2 is prefix-identical to v1_1 (additive, D24 pattern)") {
    static_assert(offsetof(CaliperTensorBridgeV1_2, struct_size) ==
                  offsetof(CaliperTensorBridgeV1_1, struct_size));
    static_assert(offsetof(CaliperTensorBridgeV1_2, texture_from_tensor) ==
                  offsetof(CaliperTensorBridgeV1_1, texture_from_tensor));
    static_assert(offsetof(CaliperTensorBridgeV1_2, update_texture) ==
                  offsetof(CaliperTensorBridgeV1_1, update_texture));
    static_assert(offsetof(CaliperTensorBridgeV1_2, release_texture) ==
                  offsetof(CaliperTensorBridgeV1_1, release_texture));
    static_assert(offsetof(CaliperTensorBridgeV1_2, texture_from_tensor_mapped) ==
                  offsetof(CaliperTensorBridgeV1_1, texture_from_tensor_mapped));
    static_assert(offsetof(CaliperTensorBridgeV1_2, alloc_shared) ==
                  offsetof(CaliperTensorBridgeV1_1, alloc_shared));
    static_assert(offsetof(CaliperTensorBridgeV1_2, free_shared) ==
                  offsetof(CaliperTensorBridgeV1_1, free_shared));
    static_assert(offsetof(CaliperTensorBridgeV1_2, caps) ==
                  offsetof(CaliperTensorBridgeV1_1, caps));
    CHECK(CALIPER_BRIDGE_CAP_IMPORT_ALLOC == (1u << 1));
    CHECK(std::string(CALIPER_TENSOR_BRIDGE_V1_2) == "caliper.tensor_bridge.v1_2");
}
```

- [ ] **Step 2: Run to verify it fails**

Run: `cmake --build build --target caliper_tests 2>&1 | tail -5`
Expected: FAIL — `caliper/services/tensor_bridge_v1_2.h: No such file or directory`

- [ ] **Step 3: Create the header** (`sdk/include/caliper/services/tensor_bridge_v1_2.h`):

```c
#pragma once
/* caliper.tensor_bridge.v1_2 — ADDITIVE revision of tensor_bridge.v1_1 (same
 * D24 pattern): the SAME seven members, prefix-identical layout, plus three
 * entry points for imported external allocations. Caps bit 1 set means the
 * host can import an applet-exported device allocation (CUDA VMM shareable
 * handle) and run device texture updates directly FROM it — zero copies of
 * the tensor data. Hosts without the bit: applets keep the v1/v1.1 contract
 * (the D2D-copy interop path). The v1/v1.1 headers, tables, and ids are
 * untouched (frozen); no ABI epoch bump. */
#include <caliper/services/tensor_bridge_v1_1.h>

#define CALIPER_TENSOR_BRIDGE_V1_2 "caliper.tensor_bridge.v1_2"

/* caps() bit 1: import_allocation/update_texture_from_alloc are live. */
#define CALIPER_BRIDGE_CAP_IMPORT_ALLOC (1u << 1)

/* OS handle types accepted by import_allocation. */
#define CALIPER_ALLOC_HANDLE_OPAQUE_WIN32 1u
#define CALIPER_ALLOC_HANDLE_OPAQUE_FD    2u

#ifdef __cplusplus
extern "C" {
#endif

/* Opaque imported-allocation id; 0 = invalid. Compare-only, host-internal. */
typedef uint64_t CaliperAllocId;

typedef struct CaliperTensorBridgeV1_2 {
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
    /* v1.1-identical member. */
    uint32_t (*caps)(void);
    /* v1.2 additions. import_allocation: hand the host an OS shareable handle
     * (from cuMemExportToShareableHandle) plus the allocation's byte size;
     * returns 0 when the host cannot import (missing cap, bad handle, no
     * hardware pair) — the applet then stays on the v1 path. The host dups
     * the handle; the applet keeps ownership of its copy. */
    CaliperAllocId (*import_allocation)(void* os_handle, uint64_t size_bytes,
                                        uint32_t handle_type);
    void (*release_allocation)(CaliperAllocId alloc);
    /* Update an existing texture (create it first via texture_from_tensor*)
     * from tensor bytes living INSIDE an imported allocation at offset_bytes.
     * desc describes shape/dtype/strides/stream; desc->data is IGNORED (the
     * imported allocation + offset are the address). Same acceptance gates
     * as update_texture; false = not updated, caller falls back. */
    bool (*update_texture_from_alloc)(CaliperTextureId tex, CaliperAllocId alloc,
                                      uint64_t offset_bytes,
                                      const CaliperTensor* desc);
} CaliperTensorBridgeV1_2;

#ifdef __cplusplus
}
#endif
```

- [ ] **Step 4: Extend the SDK wrapper** (`sdk/include/caliper/caliper.hpp`) — add beside the existing `t11_` member and fetch (`caliper.hpp:257-306`):

```cpp
// member, next to t11_:
const CaliperTensorBridgeV1_2* t12_ = nullptr;
// in the Bridge(const Host&) ctor initializer list, after t11_:
, t12_(static_cast<const CaliperTensorBridgeV1_2*>(
      host.service(CALIPER_TENSOR_BRIDGE_V1_2)))
// methods, next to caps():
CaliperAllocId import_allocation(void* h, uint64_t size, uint32_t type) const {
    return (t12_ && t12_->import_allocation)
               ? t12_->import_allocation(h, size, type) : 0;
}
void release_allocation(CaliperAllocId a) const {
    if (t12_ && t12_->release_allocation) t12_->release_allocation(a);
}
bool update_texture_from_alloc(CaliperTextureId tex, CaliperAllocId a,
                               uint64_t off, const CaliperTensor* d) const {
    return (t12_ && t12_->update_texture_from_alloc)
               ? t12_->update_texture_from_alloc(tex, a, off, d) : false;
}
```
Add `#include <caliper/services/tensor_bridge_v1_2.h>` beside the v1_1 include.

- [ ] **Step 5: Build + run, verify pass**

Run: `cmake --build build --target caliper_tests && ./build/tests/caliper_tests "*v1_2*"`
Expected: PASS (1 test case)

- [ ] **Step 6: Commit**

```bash
git add sdk/include/caliper/services/tensor_bridge_v1_2.h sdk/include/caliper/caliper.hpp tests/test_abi.cpp
git commit -m "feat(sdk): tensor_bridge.v1_2 — additive import-allocation entry points + caps bit"
```

---

### Task 2: Host bridge bookkeeping + v1_2 registration (stub-renderer TDD)

**Files:**
- Modify: `src/host/tensor_bridge.h`, `src/host/tensor_bridge.cpp` (caps at `:215`, device mapping at `:205`)
- Modify: `src/host/renderer/host_renderer.h` (three defaulted virtuals + one probe)
- Modify: `src/host/host_services.cpp` (`kBridge11` pattern at `:253-256`, ids at `:258-263`, dispatch at `:334-335`)
- Test: `tests/test_tensor_bridge.cpp` (StubRenderer at `:21-79`, caps test pattern at `:336-351`)

**Interfaces:**
- Consumes: Task 1 header; `HostRenderer` seam.
- Produces: `HostRenderer` virtuals `import_external_allocation(void*,uint64_t,uint32_t) -> uint64_t` (default 0), `release_external_allocation(uint64_t)` (default no-op), `tex_update_from_imported(uint64_t tex, uint64_t alloc, uint64_t offset, const CaliperTensor& desc, int32_t colormap, float vmin, float vmax) -> bool` (default false), `supports_external_import() -> bool` (default false); `TensorBridge::import_allocation/release_allocation/update_texture_from_alloc` mirroring the ABI; caps() adds `CALIPER_BRIDGE_CAP_IMPORT_ALLOC` iff `supports_external_import()`.

- [ ] **Step 1: Write failing tests** (append to `tests/test_tensor_bridge.cpp`, following the `StreamStub` pattern at `:336`):

```cpp
namespace {
struct ImportStub : StubRenderer {
    using StubRenderer::StubRenderer;
    uint64_t next_id = 1;
    std::vector<uint64_t> released;
    struct Update { uint64_t tex, alloc, offset; };
    std::vector<Update> updates;
    bool supports_external_import() const override { return true; }
    uint64_t import_external_allocation(void*, uint64_t size, uint32_t type) override {
        if (type != CALIPER_ALLOC_HANDLE_OPAQUE_WIN32 &&
            type != CALIPER_ALLOC_HANDLE_OPAQUE_FD) return 0;
        if (size == 0) return 0;
        return next_id++;
    }
    void release_external_allocation(uint64_t id) override { released.push_back(id); }
    bool tex_update_from_imported(uint64_t tex, uint64_t alloc, uint64_t off,
                                  const CaliperTensor&, int32_t, float, float) override {
        updates.push_back({tex, alloc, off});
        return true;
    }
};
} // namespace

TEST_CASE("caps() adds IMPORT_ALLOC only when the renderer supports it") {
    StubRenderer plain("vulkan");
    TensorBridge b1(plain);
    CHECK((b1.caps() & CALIPER_BRIDGE_CAP_IMPORT_ALLOC) == 0u);
    ImportStub imp("vulkan");
    TensorBridge b2(imp);
    CHECK((b2.caps() & CALIPER_BRIDGE_CAP_IMPORT_ALLOC) != 0u);
}

TEST_CASE("import_allocation: id lifecycle, invalid args, double release") {
    ImportStub imp("vulkan");
    TensorBridge b(imp);
    uint64_t dummy = 42;
    CaliperAllocId a = b.import_allocation(&dummy, 4096,
                                           CALIPER_ALLOC_HANDLE_OPAQUE_WIN32);
    REQUIRE(a != 0);
    CHECK(b.import_allocation(nullptr, 4096,
                              CALIPER_ALLOC_HANDLE_OPAQUE_WIN32) == 0);   // null handle
    CHECK(b.import_allocation(&dummy, 0,
                              CALIPER_ALLOC_HANDLE_OPAQUE_WIN32) == 0);   // zero size
    CHECK(b.import_allocation(&dummy, 4096, 99u) == 0);                   // bad type
    b.release_allocation(a);
    CHECK(imp.released.size() == 1);
    b.release_allocation(a);              // double release: no-op, no crash
    CHECK(imp.released.size() == 1);
    b.release_allocation(0);              // invalid id: no-op
    CHECK(imp.released.size() == 1);
}

TEST_CASE("update_texture_from_alloc: acceptance gates + bounds + fallback contract") {
    ImportStub imp("vulkan");
    TensorBridge b(imp);
    uint64_t dummy = 42;
    // 4x4 f32 mapped texture created through the normal path first
    std::vector<float> px(16, 0.5f);
    CaliperTensor t = make_f32_2d(px.data(), 4, 4);        // existing test helper
    CaliperTextureId tex = b.texture_from_tensor_mapped(&t, 0, 0.f, 1.f, 0);
    REQUIRE(tex != 0);
    CaliperAllocId a = b.import_allocation(&dummy, 4 * 4 * sizeof(float),
                                           CALIPER_ALLOC_HANDLE_OPAQUE_WIN32);
    REQUIRE(a != 0);
    CaliperTensor d = t; d.data = nullptr;                  // desc: data ignored
    CHECK(b.update_texture_from_alloc(tex, a, 0, &d));
    CHECK(imp.updates.size() == 1);
    // offset + extent exceeding the imported size must be rejected host-side
    CHECK_FALSE(b.update_texture_from_alloc(tex, a, 8, &d));
    // unknown alloc / unknown texture / null desc reject without renderer call
    CHECK_FALSE(b.update_texture_from_alloc(tex, 999u, 0, &d));
    CHECK_FALSE(b.update_texture_from_alloc(0, a, 0, &d));
    CHECK_FALSE(b.update_texture_from_alloc(tex, a, 0, nullptr));
    // non-contiguous desc rejected by the same frozen gate
    CaliperTensor bad = d; bad.strides[0] = 5;
    CHECK_FALSE(b.update_texture_from_alloc(tex, a, 0, &bad));
    CHECK(imp.updates.size() == 1);
}
```

(If `make_f32_2d` does not exist under that name, reuse the exact tensor-builder helper the acceptance-matrix case at `tests/test_tensor_bridge.cpp:170-227` uses — same shape, do not invent a new helper.)

- [ ] **Step 2: Run to verify failure**

Run: `cmake --build build --target caliper_tests 2>&1 | tail -5`
Expected: FAIL — `supports_external_import` / `import_allocation` not members.

- [ ] **Step 3: Implement.**
`src/host/renderer/host_renderer.h` — add after `alloc_device_shared` (keeping the seam's comment style):

```cpp
/* External-allocation import (bridge v1.2). Default: unsupported — the
 * bridge then never grants CALIPER_BRIDGE_CAP_IMPORT_ALLOC. */
virtual bool supports_external_import() const { return false; }
virtual uint64_t import_external_allocation(void* /*os_handle*/,
                                            uint64_t /*size_bytes*/,
                                            uint32_t /*handle_type*/) { return 0; }
virtual void release_external_allocation(uint64_t /*id*/) {}
virtual bool tex_update_from_imported(uint64_t /*tex*/, uint64_t /*alloc*/,
                                      uint64_t /*offset_bytes*/,
                                      const CaliperTensor& /*desc*/,
                                      int32_t /*colormap*/,
                                      float /*vmin*/, float /*vmax*/) { return false; }
```

`src/host/tensor_bridge.h` — add public methods + a private table:

```cpp
CaliperAllocId import_allocation(void* os_handle, uint64_t size_bytes,
                                 uint32_t handle_type);
void release_allocation(CaliperAllocId a);
bool update_texture_from_alloc(CaliperTextureId tex, CaliperAllocId a,
                               uint64_t offset_bytes, const CaliperTensor* desc);
// private:
struct ImportedAlloc { uint64_t renderer_id; uint64_t size_bytes; };
std::unordered_map<uint64_t, ImportedAlloc> imported_;   // bridge id -> entry
uint64_t next_alloc_id_ = 1;
```

`src/host/tensor_bridge.cpp` — implementation. Validation order in `update_texture_from_alloc`: (1) desc non-null, tex known, alloc known; (2) the SAME frozen acceptance gates `update_texture` applies to shape/dtype/strides — call the existing validator, not a copy of it; (3) byte extent from shape×strides×dtype, then `offset + extent <= size_bytes` (the host-side analog of the `cuMemGetAddressRange` check — the renderer re-checks against the real allocation); (4) forward to `renderer_.tex_update_from_imported(...)` with the texture's stored colormap/vmin/vmax (the same stored values `update_texture` uses for mapped textures — pinned-at-create, no re-range). Any rejection emits one `bridge_log` line (existing sink). `caps()` (at `tensor_bridge.cpp:215`) gains: `if (renderer_.supports_external_import()) c |= CALIPER_BRIDGE_CAP_IMPORT_ALLOC;`. `import_allocation` validates handle non-null, size > 0, type ∈ {OPAQUE_WIN32, OPAQUE_FD} BEFORE forwarding; renderer returning 0 → return 0 without inserting.

`src/host/host_services.cpp` — clone the `kBridge11` block (`:253-256`): three new C thunks `br_import_allocation`/`br_release_allocation`/`br_update_texture_from_alloc` (null-bridge → 0/no-op/false, the `br_caps` pattern at `:249-252`), static `kBridge12` table with `sizeof(CaliperTensorBridgeV1_2)`, id added to `kIds` (`:258-263`), dispatch line `if (std::strcmp(id, CALIPER_TENSOR_BRIDGE_V1_2) == 0) return &kBridge12;` beside `:335`.

- [ ] **Step 4: Build + run**

Run: `cmake --build build --target caliper_tests && ./build/tests/caliper_tests`
Expected: ALL PASS (new cases + the full existing suite — the acceptance matrix must be untouched).

- [ ] **Step 5: Commit**

```bash
git add src/host/tensor_bridge.h src/host/tensor_bridge.cpp src/host/renderer/host_renderer.h src/host/host_services.cpp tests/test_tensor_bridge.cpp
git commit -m "feat(host): bridge import-allocation bookkeeping + v1_2 table, stub-renderer tested"
```

---

### Task 3: Applet-side allocation registry (pure C++, CUDA-free)

**Files:**
- Create: `sdk/include/caliper/adapters/alloc_registry.hpp`
- Test: create `tests/test_alloc_registry.cpp`; modify `tests/CMakeLists.txt` (add to `caliper_tests` sources)

**Interfaces:**
- Consumes: nothing (standalone, `<cstdint>`, `<map>`, `<mutex>`, `<optional>`).
- Produces: `caliper::adapters::AllocRegistry` with `void add(uintptr_t base, uint64_t size, void* os_handle)`, `void remove(uintptr_t base)`, `struct Hit { void* os_handle; uint64_t size; uint64_t offset; uintptr_t base; }`, `std::optional<Hit> find(const void* p, uint64_t extent_bytes) const` — thread-safe; `find` succeeds only when `[p, p+extent)` lies wholly inside one registered range.

- [ ] **Step 1: Write the failing test** (`tests/test_alloc_registry.cpp`):

```cpp
#include <doctest/doctest.h>
#include <caliper/adapters/alloc_registry.hpp>

using caliper::adapters::AllocRegistry;

TEST_CASE("AllocRegistry: interval lookup with offset and extent bounds") {
    AllocRegistry r;
    int h1 = 0, h2 = 0;
    r.add(0x1000, 0x1000, &h1);              // [0x1000, 0x2000)
    r.add(0x8000, 0x0800, &h2);              // [0x8000, 0x8800)

    auto hit = r.find(reinterpret_cast<void*>(0x1200), 0x100);
    REQUIRE(hit.has_value());
    CHECK(hit->os_handle == &h1);
    CHECK(hit->offset == 0x200);
    CHECK(hit->size == 0x1000);
    CHECK(hit->base == 0x1000);

    CHECK(r.find(reinterpret_cast<void*>(0x1000), 0x1000).has_value());  // exact fit
    CHECK_FALSE(r.find(reinterpret_cast<void*>(0x1F00), 0x200).has_value()); // spills out
    CHECK_FALSE(r.find(reinterpret_cast<void*>(0x0FFF), 1).has_value());     // below
    CHECK_FALSE(r.find(reinterpret_cast<void*>(0x2000), 1).has_value());     // end is exclusive
    CHECK_FALSE(r.find(reinterpret_cast<void*>(0x3000), 1).has_value());     // gap

    r.remove(0x1000);
    CHECK_FALSE(r.find(reinterpret_cast<void*>(0x1200), 1).has_value());
    CHECK(r.find(reinterpret_cast<void*>(0x8400), 0x100)->os_handle == &h2);
}
```

- [ ] **Step 2: Run to verify failure**

Run: `cmake --build build --target caliper_tests 2>&1 | tail -3`
Expected: FAIL — header not found.

- [ ] **Step 3: Implement** (`sdk/include/caliper/adapters/alloc_registry.hpp`):

```cpp
#pragma once
/* caliper/adapters/alloc_registry.hpp — maps a raw device pointer back to the
 * exportable allocation that contains it (base -> {handle, size}), so the
 * adapter can hand the bridge (alloc, offset) instead of a bare pointer.
 * Pure C++, no CUDA dependency: unit-tested on every platform. */
#include <cstdint>
#include <map>
#include <mutex>
#include <optional>

namespace caliper::adapters {

class AllocRegistry {
public:
    struct Hit {
        void*     os_handle;
        uint64_t  size;
        uint64_t  offset;
        uintptr_t base;
    };

    void add(uintptr_t base, uint64_t size, void* os_handle) {
        std::lock_guard<std::mutex> lock(mu_);
        ranges_[base] = Range{size, os_handle};
    }
    void remove(uintptr_t base) {
        std::lock_guard<std::mutex> lock(mu_);
        ranges_.erase(base);
    }
    std::optional<Hit> find(const void* p, uint64_t extent_bytes) const {
        const auto addr = reinterpret_cast<uintptr_t>(p);
        std::lock_guard<std::mutex> lock(mu_);
        auto it = ranges_.upper_bound(addr);   // first base > addr
        if (it == ranges_.begin()) return std::nullopt;
        --it;                                  // candidate: greatest base <= addr
        const uintptr_t base = it->first;
        const Range& r = it->second;
        if (addr < base) return std::nullopt;
        const uint64_t off = addr - base;
        if (off > r.size || r.size - off < extent_bytes) return std::nullopt;
        return Hit{r.os_handle, r.size, off, base};
    }

private:
    struct Range { uint64_t size; void* os_handle; };
    std::map<uintptr_t, Range> ranges_;
    mutable std::mutex mu_;
};

} // namespace caliper::adapters
```

- [ ] **Step 4: Register the test** in `tests/CMakeLists.txt` (add `test_alloc_registry.cpp` to the `caliper_tests` source list), build, run:

Run: `cmake --build build --target caliper_tests && ./build/tests/caliper_tests "*AllocRegistry*"`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add sdk/include/caliper/adapters/alloc_registry.hpp tests/test_alloc_registry.cpp tests/CMakeLists.txt
git commit -m "feat(adapters): AllocRegistry — pointer→(allocation, offset) interval lookup"
```

---

### Task 4: Exportable pool adapter (CUDA-gated) + tripwire

**Files:**
- Create: `sdk/include/caliper/adapters/exportable_pool.hpp`
- Test: `tests/test_torch_adapter.cpp` (append; torch target only, label "torch")

**Interfaces:**
- Consumes: `AllocRegistry` (Task 3), bridge v1.2 wrapper (Task 1), libtorch 2.5.1 (`c10::cuda::MemPool`, `torch::cuda::CUDAPluggableAllocator`), CUDA driver API — **runtime-loaded by the adapter itself** (a small inline loader in the header: `LoadLibraryA("nvcuda.dll")` / `dlopen("libcuda.so.1")` + `GetProcAddress`/`dlsym`, the exact mechanism of `src/host/cuda_driver.cpp:15-77` but self-contained in the SDK header — applets must NOT include host-internal headers and must NOT link the CUDA toolkit): `cuMemCreate`, `cuMemAddressReserve`, `cuMemMap`, `cuMemSetAccess`, `cuMemExportToShareableHandle`, `cuMemUnmap`, `cuMemRelease`, `cuMemAddressFree`, `cuMemGetAllocationGranularity`. Any symbol missing → `ok() == false` → callers fall back; never a crash.
- Produces: `caliper::adapters::ExportablePool` — ctor `(int device_index)`; `bool ok() const`; `at::cuda::MemPool-scoped` RAII guard `ExportablePool::Scope use()` (tensors allocated inside the scope land in the pool); `std::optional<BridgeRef> to_bridge(caliper::Bridge&, const at::Tensor&)` where `BridgeRef{CaliperAllocId alloc; uint64_t offset;}` — imports each pool block into the bridge once (cached by base), returns nullopt for tensors outside the pool (caller falls back to `stream_to_tensor` + `update_texture`).

**Compile guard (same discipline as `torch.hpp:37`):**
```cpp
#if !defined(__APPLE__) && __has_include(<c10/cuda/CUDAStream.h>) && \
    __has_include(<c10/cuda/CUDACachingAllocator.h>)
```
Windows handle type `CU_MEM_HANDLE_TYPE_WIN32` / POSIX `CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR` selected by `#ifdef _WIN32`, mapped to `CALIPER_ALLOC_HANDLE_OPAQUE_WIN32` / `_FD` at the bridge call.

- [ ] **Step 1: Write the failing tripwire test** (append to `tests/test_torch_adapter.cpp`, mirroring the loud-guard pattern at `:181-186`):

```cpp
TEST_CASE("exportable pool: allocations are pool-backed, registry-resolvable, "
          "and export a shareable handle" * doctest::skip(!torch::cuda::is_available())) {
#if !defined(__APPLE__) && __has_include(<c10/cuda/CUDAStream.h>) && \
    __has_include(<c10/cuda/CUDACachingAllocator.h>)
    caliper::adapters::ExportablePool pool(0);
    REQUIRE_MESSAGE(pool.ok(), "cuMemCreate-backed pool failed on a CUDA machine "
                               "— VMM or export unsupported by this driver?");
    at::Tensor t;
    {
        auto scope = pool.use();
        t = torch::rand({17, 9}, torch::TensorOptions()
                                     .device(torch::kCUDA).dtype(torch::kFloat32));
        // a DERIVED tensor inside the scope must also land in the pool:
        t = t.square().contiguous();
    }
    auto hit = pool.registry().find(t.data_ptr(), t.numel() * sizeof(float));
    REQUIRE_MESSAGE(hit.has_value(),
        "pool-scoped tensor not resolvable in the AllocRegistry — "
        "MemPool routing broke (torch 2.5.1 API drift?)");
    CHECK(hit->os_handle != nullptr);
    CHECK(hit->size >= t.numel() * sizeof(float));
#else
    REQUIRE_MESSAGE(!torch::cuda::is_available(),
        "CUDA machine but the exportable-pool branch is compiled out");
#endif
}
```

- [ ] **Step 2: Verify it fails to compile** (macOS: guard compiles the body out; the failure surface here is header-not-found once the include is added):

Run: `cmake --build build --target caliper_torch_tests 2>&1 | tail -3`
Expected: FAIL — `caliper/adapters/exportable_pool.hpp: No such file or directory` (after adding the include at the top of the test file).

- [ ] **Step 3: Implement** (`sdk/include/caliper/adapters/exportable_pool.hpp`). Core structure (the CUDA-gated interior; outside the guard the class exists with `ok() == false` and `to_bridge` returning nullopt so applet code compiles everywhere):

```cpp
#pragma once
/* caliper/adapters/exportable_pool.hpp — a torch MemPool whose blocks are
 * cuMemCreate'd with a shareable handle type, so the bridge can IMPORT them
 * (tensor_bridge.v1_2) and update textures with zero copies of the data.
 * Applet-side only (D11). Every failure degrades to "not pool-backed":
 * the caller falls back to the v1/v1.1 path. */
#include <caliper/adapters/alloc_registry.hpp>
#include <caliper/caliper.hpp>
#include <torch/torch.h>
```

Interior (guarded): a `CUDAPluggableAllocator` whose alloc fn: rounds size to `cuMemGetAllocationGranularity`, `cuMemCreate` with `CUmemAllocationProp{ .type = CU_MEM_ALLOCATION_TYPE_PINNED, .location = {CU_MEM_LOCATION_TYPE_DEVICE, device}, .requestedHandleTypes = <WIN32|FD> }`, `cuMemAddressReserve` + `cuMemMap` + `cuMemSetAccess(RW, device)`, `cuMemExportToShareableHandle` → registry.add(va, padded_size, handle); free fn reverses (registry.remove, unmap, release, address-free, CloseHandle/close). `c10::cuda::MemPool` constructed with that allocator; `Scope` = RAII over torch's pool-context (in 2.5.1: `c10::cuda::MemPoolContext` + `c10::cuda::CUDACachingAllocator::beginAllocateToPool`/`releasePool` pair — implementer: pin exact 2.5.1 signatures from `third_party/libtorch/include/c10/cuda/CUDACachingAllocator.h` and `c10/cuda/MemPool.h` (mac libtorch ships these headers — readable on this machine even though the branch is compiled out) before writing this; if `MemPool.h` is absent in 2.5.1, fall back to `beginAllocateToPool(device, pool_id, filter)` directly, which exists in 2.5). `to_bridge`: `registry().find(t.data_ptr(), byte_extent)` → per-base cache lookup → miss: `bridge.import_allocation(handle, size, <type>)`, 0 → negative-cache and return nullopt (permanent fallback for that block); hit: `BridgeRef{alloc, hit->offset}`.

- [ ] **Step 4: Build both test targets on macOS** (compiles the guard-out path; full CUDA execution happens in Task 6 on hardware):

Run: `cmake --build build --target caliper_torch_tests caliper_tests && ctest --test-dir build -L torch --output-on-failure`
Expected: PASS (CUDA cases skip on macOS; MPS/CPU suite untouched).

- [ ] **Step 5: Commit**

```bash
git add sdk/include/caliper/adapters/exportable_pool.hpp tests/test_torch_adapter.cpp
git commit -m "feat(adapters): ExportablePool — cuMemCreate shareable-handle MemPool + bridge import glue"
```

---

### Task 5: Vulkan import path (Windows-gated TU; builds on Windows, verified in Task 6)

**Files:**
- Modify: `src/host/renderer/vulkan_renderer.cpp` (Tex/Interop structs at `:434-467`, `ensure_buffer` donor at `:735-770`, `ensure_shared_buffer` donor at `:985-1031`, `tex_update_from_device` donor at `:333-397`, cmap/blit recorders at `:1103-1206`, pipelining at `:852-912` + `:1216-1300`)

**Interfaces:**
- Consumes: Task 2 seam virtuals; `CALIPER_ALLOC_HANDLE_OPAQUE_WIN32`; existing `colormap.comp` (unchanged — it already reads element strides; the byte offset rides the descriptor).
- Produces: overrides of `supports_external_import` / `import_external_allocation` / `release_external_allocation` / `tex_update_from_imported`; new `last_device_path()` strings `"compute-imported"` / `"blit-imported"`.

**Design (locked):**
- Renderer state: `std::unordered_map<uint64_t, ImportedAlloc> imported_; uint64_t next_import_id_ = 1;` with `struct ImportedAlloc { VkBuffer buf; VkDeviceMemory memory; VkDeviceSize size; void* handle_dup; };` — **one VkBuffer per imported allocation, textures reference it at descriptor offsets.** No per-texture buffer.
- `supports_external_import()` → `interop_ok_ && external_memory_ok_` (`_WIN32` only; `#else` false).
- `import_external_allocation(h, size, type)`: reject unless type == OPAQUE_WIN32; `DuplicateHandle` (host owns its copy; `CloseHandle` at release); `vkCreateBuffer` with `VkExternalMemoryBufferCreateInfo{OPAQUE_WIN32}` (donor `:740-741`), usage `STORAGE | TRANSFER_SRC`, size = the applet's padded allocation size; `vkGetBufferMemoryRequirements`; `vkAllocateMemory` with chained `VkImportMemoryWin32HandleInfoKHR{ .handleType = OPAQUE_WIN32_BIT, .handle = dup }` (the import twin of the export at `:751-754`), DEVICE_LOCAL memory type; `vkBindBufferMemory(buf, mem, 0)`. Any failure → full cleanup → return 0 (bridge falls back).
- `tex_update_from_imported(tex, alloc, offset, desc, colormap, vmin, vmax)`: guards mirror `tex_update_from_device:336-346` (texture exists, alloc exists, dtype ∈ {f32, u8}, `bytes = tensor_extent_bytes(desc, elem)`, `offset + bytes <= imported.size`); **f32 path additionally requires `offset % minStorageBufferOffsetAlignment == 0`** (query `VkPhysicalDeviceLimits` once at init; torch sub-allocations are 512-aligned so this passes in practice; violation → false → fallback). Recording reuses the existing bodies with two mechanical changes: cmap descriptor binding 0 becomes `VkDescriptorBufferInfo{ imported.buf, offset, bytes }` (donor `write_cmap_set:1103-1115` binds the per-texture `Interop.set` — rewrite it per update; per-texture chains are already serialized by `retire()` backpressure); blit uses `VkBufferImageCopy{ .bufferOffset = offset }` (donor `:1139-1154`). **No D2D copy anywhere in this path.**
- Ordering: if `desc.stream != NULL && pipelined_ok_` → the `update_pipelined` chain (`:1216-1300`) minus the `cuMemcpyDtoDAsync` step: `cuSignalExternalSemaphoresAsync` (already loaded) signals `base+1` on the producer's stream, the Vulkan pass GPU-waits `base+1`/signals `base+2`, `pending_frame_waits_` as today. Else → synchronous fenced submit (adapter has drained per the v1 rung contract — same shape as `:377-390` without the copy). Factor `ensure_pipeline_objects` (`:852-912`) so the imported path creates cb/descriptor-set/LUT/timeline **without** `ensure_shared_buffer` (no interop buffer needed).
- `release_external_allocation(id)`: `vkQueueWaitIdle` (the existing synchronous-release model, `tex_release` precedent), destroy buffer, free memory, `CloseHandle(handle_dup)`, erase. Textures referencing a released alloc simply fail their next `update_texture_from_alloc` (bridge id lookup) → fallback; document in the v1_2 header comment.

- [ ] **Step 1:** Implement the four overrides + `ImportedAlloc` map per the locked design above, donor-anchored. All new code inside the existing `#ifdef _WIN32` discipline (`:61-66` pattern).
- [ ] **Step 2:** Verify macOS build is untouched: `cmake --build build 2>&1 | tail -3` (the TU is WIN32-only; expected: full build green, zero new warnings elsewhere).
- [ ] **Step 3:** Cross-check on the Windows box (or defer to Task 6's hardware pass): `cmake --build build --target caliper_vulkan_backend` green.
- [ ] **Step 4: Commit** — `git commit -m "feat(vulkan): import applet-exported allocations; texture updates with zero data copies"`

---

### Task 6: Optional VMM driver table + gfx-cuda hardware rows

**Files:**
- Modify: `src/host/cuda_driver.h` (Api at `:91-131`), `src/host/cuda_driver.cpp` (entries at `:39-63`, all-or-nothing rule at `:66`)
- Modify: `tests/gfx/gfx_main.cpp` (VkEnv at `:754-787`, `vk_cuda_ready()` at `:806-817`, donor CUDA case at `:844-877`)

**Interfaces:**
- Consumes: Tasks 1–5.
- Produces: `cudadrv::VmmApi` (separate, OPTIONAL table — `vmm_api()` returns nullptr if ANY symbol missing; the core `Api` all-or-nothing rule at `cuda_driver.cpp:66` is untouched so older drivers keep working): `cuMemCreate`, `cuMemAddressReserve`, `cuMemMap`, `cuMemSetAccess`, `cuMemExportToShareableHandle`, `cuMemUnmap`, `cuMemRelease`, `cuMemAddressFree`, `cuMemGetAllocationGranularity` + the `CUmemAllocationProp`/`CUmemAccessDesc` structs (with `reserved[16]` padding, matching the existing desc style at `cuda_driver.h:39-88`).

- [ ] **Step 1: Write the hardware tests first** (append to `gfx_main.cpp` beside the donor case at `:844`, same double skip-guard pattern plus `cudadrv::vmm_api()` presence):
  - **`import path byte-exact at offset 0 and at a 512-byte offset`**: build a VMM allocation via `vmm_api()` (granularity-padded, WIN32 shareable, mapped + RW access), `cuMemcpyHtoD` a known 17×9 f32 grid at offset 0 and a second 5×3 grid at offset 512; export the handle; `bridge.import_allocation(...)` must return nonzero; create textures via `texture_from_tensor_mapped` (CPU seed of matching shape); `update_texture_from_alloc(tex, alloc, 0/512, &desc)` must return true, `last_device_path() == "compute-imported"`, and `debug_readback_rgba8` must equal `map_f32_to_rgba8` byte-for-byte for BOTH offsets.
  - **`misaligned offset falls back, never wrong`**: `offset = 4` (fails minStorageBufferOffsetAlignment) → `update_texture_from_alloc` returns false, texture pixels unchanged, `last_device_path()` untouched.
  - **`u8 blit-imported row`**: 3-channel u8 pattern at nonzero offset → `"blit-imported"`, readback equals `expand_u8_to_rgba8`.
  - **`release + reuse stress`**: import, update, `release_allocation`, subsequent update returns false (fallback contract); loop 50× import/release, no validation errors, ids never reused while live.
  - **`bounds`**: `offset + extent > size` → false (host-side check from Task 2 plus renderer re-check).
- [ ] **Step 2:** Verify the suite still builds+passes on macOS (`ctest --test-dir build -L gfx`) — new cases are inside `#ifdef CALIPER_HAVE_VULKAN` and skip.
- [ ] **Step 3:** Implement the `VmmApi` optional table.
- [ ] **Step 4:** Full hardware pass on the Windows/NVIDIA box: `ctest -L gfx` → all rows green including the five new cases; record results in the plan's verification log (m2a-windows-verification.md pattern).
- [ ] **Step 5: Commit** — `git commit -m "test(gfx): imported-allocation byte-exactness rows + optional VMM driver table"`

---

### Task 7: Exemplar opt-in (gpt_scope)

**Files:**
- Modify: `applets/gpt_scope/` (the attention-snapshot worker + status line; exact call site: where the worker currently produces the attention tensor handed to `stream_to_tensor`)

**Interfaces:**
- Consumes: `ExportablePool` (Task 4), bridge v1.2 wrapper (Task 1).

- [ ] **Step 1:** Construct one `ExportablePool` in the applet when `bridge.caps() & CALIPER_BRIDGE_CAP_IMPORT_ALLOC` and torch reports CUDA; snapshot the attention tensor inside `pool.use()` scope (the existing snapshot clone moves inside the scope — one-line move).
- [ ] **Step 2:** At upload: `pool.to_bridge(bridge, snap)` → hit: `update_texture_from_alloc(tex, ref.alloc, ref.offset, &desc)` (desc from `stream_to_tensor(snap, caps)` with `data` ignored); miss or false: existing `update_texture` path unchanged.
- [ ] **Step 3:** Status line: `"zero-copy (imported pool)"` when the imported path ran this frame; existing strings otherwise. Wording discipline: only this path may say "zero-copy" for arbitrary tensors.
- [ ] **Step 4:** macOS: build + run gpt_scope on Metal — behavior identical (cap absent → pool never constructed). Commit — `git commit -m "feat(gpt_scope): zero-copy attention uploads via exportable pool when host grants IMPORT_ALLOC"`

---

### Task 8: Docs

- [ ] `ZEROCOPY.md`: crossings table gains `| Vulkan / CUDA (Windows), exportable-pool tensor | 0 host copies; **0 in-VRAM copies** (bridge imports the pool block; pass reads it at offset) | Implemented — hardware verification pending/complete |`; a short "Imported allocations" paragraph after the `alloc_shared` one (the floor is now per-allocation-origin: foreign allocations keep the 1-copy floor).
- [ ] `docs/vulkan-cuda-backend.md`: As-Built gains a V5 row (import path, descriptor-offset binding, no new host driver symbols, optional VmmApi for tests).
- [ ] `WHITEPAPER.md` §9: replace the floor bullet with: "Arbitrary framework-allocated CUDA tensors carry a one in-VRAM-copy floor **unless allocated from Caliper's exportable pool, which removes it**; the floor persists only for memory born unshareable."
- [ ] Commit — `git commit -m "docs(zerocopy): imported-pool row — the in-VRAM-copy floor is now per-allocation-origin"`
