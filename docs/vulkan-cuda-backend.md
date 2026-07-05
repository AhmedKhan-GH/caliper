# Vulkan + CUDA Backend — Specification

| | |
|---|---|
| **Status** | Draft for review |
| **Date** | 2026-07-03 |
| **Owner** | Ahmed Khan |
| **Scope** | The NVIDIA half of PLATFORM.md §5.4/D13: a `VulkanRenderer` implementing the `HostRenderer` seam on Windows (Linux-ready), plus CUDA external-memory interop in the tensor bridge, so `CALIPER_DEV_CUDA` tensors reach the screen with no CPU staging. Assigned to Phase 4 by §17. |
| **Parent** | `PLATFORM.md` §5.4 (renderer strategy), §7.4 (bridge), §16 (testing), §17 Phase 4, §19 (GPU-interop risk row) |

> **How to read this document.** §1 states what exists today (all of it verified in-tree). Everything from §3 onward is the proposed target, written in spec present tense. §6 sequences the work into four increments, each independently shippable; §8 lists the decisions awaiting ratification (proposed as D19–D22, continuing PLATFORM.md §18's log).

---

## 1. Where We Are

| Asset | Location | State |
|---|---|---|
| `HostRenderer` seam: texture ops + `tex_update_from_device` + `last_device_path()` | `src/host/renderer/host_renderer.h` | **Done.** The interface this spec implements. Host-internal — nothing here touches the ABI (§5.4 guarantee holds: no epoch bump, no applet rebuilds). |
| Metal backend — the parity donor | `src/host/renderer/metal_renderer.mm` | **Done.** Frame ordering (clear in `new_frame()`), `0.05/0.05/0.08` clear parity, id-table texture handles, runtime-compiled `cmap_f32` compute path, `blit_u8` path, byte-extent bounds check before dispatch. |
| GL backend — frozen fallback | `src/host/renderer/gl_renderer.cpp` | **Done.** `tex_update_from_device` returns `false` unconditionally → bridge CPU-stages. This never changes. |
| Bridge routing + CPU reference conversions | `src/host/tensor_bridge.{h,cpp}` | **Done.** `upload_into` forwards device tensors to the backend; `map_f32_to_rgba8` is the single source of truth both GPU paths must match byte-for-byte. |
| Backend selection | `src/main.cpp` (~line 47) | Metal-or-GL only, Apple-gated. Vulkan slot does not exist. |
| Device query | `src/host/device_query.h` + `device_query_stub.cpp` | Non-Apple reports CPU. Header already says: "CUDA detection arrives with Phase 4 hardware." |
| `CALIPER_DEV_CUDA` | `sdk/include/caliper/tensor.h` | Enum value exists; nothing produces or consumes it yet. |
| Bridge active-device mapping | `tensor_bridge.cpp:205` | `strcmp(name, "metal") → METAL else CPU`. A CUDA tensor is rejected as "foreign device" today — the concrete bug this spec fixes. |

**Consequence today:** on Windows/NVIDIA every tensor takes the CPU-staged path (colormap on CPU + `glTexSubImage2D`), and a `CALIPER_DEV_CUDA` tensor is refused outright. The USP (§7.4: GPU-resident, in-loop visualization) exists only on macOS.

---

## 2. Constraints Inherited From the Platform

These are settled upstream and constrain every choice below:

1. **No ABI change.** Applets see `CaliperTensor` in and opaque `CaliperTextureId` out. All work lands behind `HostRenderer` (§5.4 rule 1).
2. **The host never links torch** (D11). The interop layer speaks the CUDA *driver/runtime* API over raw pointers only.
3. **Byte-identical colormaps across backends** (§16 "pixel-exact vs CPU reference, run per backend"). The index rule is frozen: `idx = clamp((v−vmin)/(vmax−vmin), 0, 1) * 255 + 0.5` (truncated), `vmax==vmin → t=0`, NaN → index 0, LUT packed `r | g<<8 | b<<16 | a<<24`.
4. **Per-tensor degradation ladder** (§19 risk row): device path → device-side copy → CPU-staged upload. A failed interop is a `false` return and a staged frame, never a crash or a wrong image.
5. **Frame ordering (C1 contract):** the clear happens in `new_frame()` (render pass opened with load-op CLEAR, color `0.05, 0.05, 0.08, 1.0`), applet draws, `render()` flushes ImGui draw data and presents.
6. **TDD** (repo standing rule + §16): every acceptance rule and both GPU paths get failing tests first; determinism tests compare GPU output bytes against `map_f32_to_rgba8`/`expand_u8_to_rgba8` golden output.

---

## 3. Design — `VulkanRenderer`

New translation unit `src/host/renderer/vulkan_renderer.cpp`, sibling of the Metal backend, `name() == "vulkan"`. Factory `make_vulkan_renderer()` declared in `host_renderer.h` (defined only when `CALIPER_HAVE_VULKAN`), selected by `main.cpp` — never a branch inside the frozen `make_renderer()` (same rule that kept Metal out of `gl_renderer.cpp`).

### 3.1 Core objects

- `VkInstance` (+ debug messenger in debug builds), `VkSurfaceKHR` via `glfwCreateWindowSurface` (`window_hints()` sets `GLFW_NO_API`, exactly like Metal).
- **Physical-device selection is UUID-driven, not "first discrete":** enumerate Vulkan devices, read `VkPhysicalDeviceIDProperties::deviceUUID`, and prefer a device whose UUID matches an enumerated CUDA device (`cudaDeviceProp::uuid`). Match found → that pair is *the* interop pair for the session and `device.v1` reports `CALIPER_DEV_CUDA`. No match (no NVIDIA GPU, no CUDA runtime, hybrid-GPU mismatch) → Vulkan still renders, interop is disabled, bridge CPU-stages, `device.v1` reports CPU. This single rule resolves the §19 "hybrid GPUs" risk.
- One graphics+compute queue; swapchain (FIFO present, BGRA8/RGBA8 UNORM), **2 frames in flight**; render pass with load-op CLEAR per §2.5.
- ImGui: `ImGui_ImplGlfw_InitForOther` + `ImGui_ImplVulkan_Init` (descriptor pool sized for bridge textures + ImGui internals). Host-owned ImGui/ImPlot contexts are created by the caller, as with both existing backends.

### 3.2 Texture ops (the four `HostRenderer` methods)

| Method | Implementation |
|---|---|
| `tex_create_rgba8` | `VkImage` RGBA8_UNORM, `SAMPLED \| TRANSFER_DST \| STORAGE`, device-local; sequential `uint64` handle in an id table — raw Vulkan handles never leave the file (Metal precedent). |
| `tex_upload_rgba8` | Persistent host-visible staging ring → `vkCmdCopyBufferToImage` → layout to `SHADER_READ_ONLY_OPTIMAL`. Synchronous (fence) in v1, matching Metal's `waitUntilCompleted` semantics. |
| `tex_imtexture_id` | `ImGui_ImplVulkan_AddTexture(sampler, view, SHADER_READ_ONLY_OPTIMAL)` → the `VkDescriptorSet` **is** the `ImTextureID`/`CaliperTextureId`. Allocated once at create, cached in the entry (the bridge calls this once and keys its table on it). |
| `tex_release` | **Deferred destruction:** image/view/descriptor/interop resources enter a per-frame retirement queue and are destroyed `kFramesInFlight` frames later — the one lifetime problem Metal's ARC hid that Vulkan makes explicit. `release_texture` mid-frame must be safe because applets call the bridge from `frame()`. |

### 3.3 Device-resident update — `tex_update_from_device`

Accepts `t.device == CALIPER_DEV_CUDA` (and, symmetrically with Metal, only when interop is live). Interop happens **at buffer level only** — `VkImage` tiling never meets CUDA, which sidesteps the §19 "alignment/tiling quirks" row entirely:

1. **Per-texture interop buffer, created lazily on first device update:** `VkBuffer` + `VkDeviceMemory` allocated with `VkExportMemoryAllocateInfo` (`OPAQUE_WIN32` handle type on Windows, `OPAQUE_FD` on Linux), exported via `vkGetMemoryWin32HandleKHR`, imported with `cudaImportExternalMemory` + `cudaExternalMemoryGetMappedBuffer` → one allocation visible as both a `VkBuffer` and a CUDA `void*`. Sized `w × h × elem` for f32-mapped entries, `w × h × 4` for u8 entries.
2. **Bounds check before any copy** — the CUDA analog of Metal's `src.length` check (`metal_renderer.mm:82`): `cuMemGetAddressRange` on `t.data` yields the owning allocation's base+size; the tensor's byte extent from `shape × strides` must fit inside it, else reject → staged fallback. The bridge has already bounded the extent in elements (`safe_extent_elems`); this re-bounds it in bytes against the real allocation, per the same finding-#1 rationale.
3. **Device-to-device copy** from `t.data` into the interop buffer: `cudaMemcpyDeviceToDevice` when the tensor is contiguous (the bridge guarantees contiguity today; the strided `cudaMemcpy2D` variant is unlocked only if the bridge ever relaxes that rule). VRAM→VRAM — the CPU never touches the bytes.
4. **Synchronize CUDA → Vulkan.** v1: `cudaStreamSynchronize` (honoring `t.stream` when non-null) before the Vulkan submit — the exact synchronous model the Metal path ships (`waitUntilCompleted`). Imported-semaphore pipelining (`cudaImportExternalSemaphore` / signal-wait) is increment V4, an optimization behind the same interface, not a correctness requirement.
5. **Buffer → image on the Vulkan timeline:**
   - **f32 + LUT → compute path.** A compute shader (GLSL → SPIR-V at build time via `glslangValidator`, embedded as a byte-array header — no runtime compiler dependency; the determinism contract makes runtime codegen pointless) implements the frozen index rule of §2.3 over push-constant `{w, h, sx, sy, vmin, vmax}` and a 256-entry LUT SSBO, `imageStore`-ing RGBA8. Records `last_device_path_ = "compute"`.
   - **u8 RGBA → copy path.** `vkCmdCopyBufferToImage`, C==4 only, dimensions must equal the texture's (same acceptance as `blit_u8`). Records `"blit"`.
6. Any failure at any step → `false` → the caller CPU-stages this tensor this frame (§2.4 ladder). Rejections route through the existing `bridge_log` sink.

### 3.4 Bridge + device-service integration (small, explicit diffs)

- `HostRenderer` gains one defaulted virtual: `virtual CaliperDeviceKind interop_device() const { return CALIPER_DEV_CPU; }`. Metal overrides → `CALIPER_DEV_METAL`; Vulkan overrides → `CALIPER_DEV_CUDA` when the UUID-matched pair exists, else CPU. `TensorBridge`'s constructor replaces its `strcmp(name, "metal")` mapping with `renderer.interop_device()` — deleting the string-matching that would otherwise grow a third arm.
- `device_query`: new `device_query_cuda.cpp` (compiled when `CALIPER_HAVE_CUDA`) reporting kind/index/name/`cudaMemGetInfo` free-memory hint for the matched device; stub retained otherwise. `device.v1` thereby tells applets to allocate torch tensors on `cuda:N` — mirroring how Metal hosts steer applets to MPS.
- `main.cpp` selection: non-Apple default becomes Vulkan with automatic GL fallback on `init()` failure (identical recreate-window flow the Metal path uses); `CALIPER_RENDERER=gl|vulkan` overrides. GL remains the frozen fallback everywhere (D13) — and per §17 Phase 4, the GLEW→GLAD-or-delete decision is made only once Vulkan is the Windows default.

### 3.5 `alloc_shared` on CUDA (increment V3)

Today `alloc_shared` hands out a CPU vector even on Metal — correct but staged on update. The CUDA upgrade makes it the literal-zero-copy story of §7.4/D14: allocate the interop buffer (§3.3.1) *as* the tensor's backing store, return `out_tensor.data =` the CUDA-mapped pointer with `device = CALIPER_DEV_CUDA`; the applet wraps it with `torch::from_blob` and its kernels write into memory the texture pass reads directly — the update call reduces to steps 4–5: zero movement of the tensor's data. What remains is D19's buffer→image pass (cmap conversion for f32, `vkCmdCopyBufferToImage` for u8), which no Vulkan path can elide: an optimal-tiled `VkImage` cannot alias linear buffer memory the way a unified-memory `MTLBuffer` can back a Metal texture. "Zero-copy" on this backend therefore means zero copies of the data, not zero GPU work — the same accounting §7.4 uses for Metal's "at most a layout transition," one pass heavier. Falls back to the CPU-vector behavior when interop is absent.

The full ladder, for doc/demo wording (each rung strictly no-CPU):

| Tensor source | Data movement | Honest label |
|---|---|---|
| Arbitrary torch CUDA tensor | one `cudaMemcpy` D2D into the interop buffer, then the buffer→image pass | "GPU-resident, no CPU staging" |
| `alloc_shared` tensor | none — kernels write the interop buffer in place; buffer→image pass only | "zero-copy" | The Metal `alloc_shared` upgrade (MTLBuffer-backed) is a sibling task, noted here for symmetry, out of scope for this spec.

**Honesty rule carried over from the design discussions:** an *arbitrary* torch CUDA tensor (from torch's caching allocator) is not importable into Vulkan — for those, one D2D copy (§3.3.3) is the floor. Zero-copy means `alloc_shared`. Docs and demo copy must say "GPU-resident, no CPU staging" for the general path and reserve "zero-copy" for the shared-allocation path — same wording discipline as §7.4.

---

## 4. What Deliberately Does NOT Change

| Thing | Why |
|---|---|
| ABI, SDK headers, any applet | §5.4 rule 1 — backend swaps are host-internal forever. The exit test proves it: `gpt_scope`/`embed_scope` binaries run unmodified on the Vulkan host. |
| `gl_renderer.cpp` | Frozen fallback; `tex_update_from_device` keeps returning `false`. |
| `tensor_bridge.cpp` acceptance rules | Same contiguity/extent/dtype gates on every backend; only the constructor's device mapping changes (§3.4). |
| Colormap LUTs / index rule | Single source of truth stays `tensor_bridge.{h,cpp}`; the SPIR-V shader is an implementation of it, verified byte-identical by test. |
| Metal backend | Untouched. |

---

## 5. Testing Strategy (tests first, per §16)

| Layer | Test (written before the code it gates) | Harness / gate |
|---|---|---|
| Bridge device mapping | stub renderer reporting each `interop_device()` value → bridge accepts/rejects CPU/CUDA/METAL tensors per §7.4 rules | existing unit suite (no GPU) |
| SPIR-V colormap determinism | offscreen `VkImage`, cmap dispatch over synthetic f32 grids (incl. NaN, vmin==vmax, clamp edges, non-unit `sx/sy`) → readback **byte-equal** to `map_f32_to_rgba8` | new `gfx-vulkan` ctest; **runs on lavapipe/SwiftShader in CI** — no NVIDIA hardware needed for the pure-Vulkan half |
| u8 copy path | RGBA pattern buffer → copy → readback equals `expand_u8_to_rgba8` output; C≠4 rejected | `gfx-vulkan` (lavapipe) |
| Deferred destruction | create → draw-reference → release → pump `kFramesInFlight` frames → validation layer clean, descriptor reuse sane | `gfx-vulkan` with validation layers fatal-on-error |
| External-memory round-trip | export buffer → `cudaImportExternalMemory` → pattern via `cudaMemcpy` → Vulkan readback equals pattern | `gfx-cuda` ctest, **gated on hardware** (label `cuda`; runs on the Phase-4 Windows/NVIDIA machine, skipped elsewhere) |
| Extent bounds (finding #1 parity) | tensor declaring a larger extent than its `cuMemGetAddressRange` allocation → rejected, `last_device_path()` untouched, staged fallback taken | `gfx-cuda` |
| End-to-end device path | CUDA f32 tensor → `texture_from_tensor_mapped` → readback byte-equal to CPU reference; `last_device_path() == "compute"`; u8 → `"blit"` | `gfx-cuda` |
| Degradation ladder | interop artificially disabled → same calls succeed staged; no visual diff | `gfx-cuda` |
| Backend boot & fallback | `CALIPER_RENDERER=vulkan` boots headless; ICD absent → GL fallback path taken (log asserted) | integration test, Windows CI |
| Golden applets | `gpt_scope` + `embed_scope` bundles load and render on the Vulkan host **unmodified** | host CI (the §13.1 wall, extended to the Windows runner) |

The lavapipe split matters: everything except the four `gfx-cuda` rows runs in ordinary CI, so the backend can't rot between sessions on real hardware.

---

## 6. Increments (each shippable, strangler-style)

| # | Deliverable | Exit criterion |
|---|---|---|
| **V1** | `VulkanRenderer` core: swapchain, frame loop, ImGui backend, staging-upload texture ops, deferred destruction; `main.cpp` selection + GL fallback | Windows host runs on Vulkan by default; all applets render via CPU-staged bridge; `gfx-vulkan` (lavapipe) suite green; clear/frame-order parity with Metal verified |
| **V2** | CUDA interop: UUID pairing, exportable interop buffers, D2D copy, SPIR-V cmap + copy paths, `interop_device()`, `device_query_cuda`, bounds check | `gfx-cuda` suite green on hardware; `gpt_scope` attention heatmaps take the `"compute"` path with zero CPU staging — the §17 Phase 4 exit brought forward |
| **V3** | `alloc_shared` CUDA upgrade (§3.5) | shared-tensor training-loop demo writes weights into texture-backed memory; update path does no copy |
| **V4** | Semaphore pipelining (`cudaImportExternalSemaphore`), staging-ring elision, perf pass | frame-time regression suite shows no sync stalls; purely internal |

V1 has no CUDA dependency at all and already retires GL as the Windows default; V2 is where the USP lands on NVIDIA. A slip in V2+ blocks nothing else in Phase 4 (§19 mitigation row: renderer work is host-internal).

---

## 7. Build & CI Wiring

- `CALIPER_HAVE_VULKAN`: `find_package(Vulkan)` (LunarG SDK on Windows); backend TU compiled only when found — macOS builds are untouched by default.
- `CALIPER_HAVE_CUDA`: existing `USE_CUDA` gate reused; interop lives in its own TU (`cuda_interop.cpp`) so V1 builds and ships without any CUDA toolkit present.
- SPIR-V compiled at build time (`glslangValidator` from the Vulkan SDK) into a generated header; committed golden copy so a missing compiler degrades to the checked-in bytes.
- CI: Windows runner gains lavapipe for `gfx-vulkan`; `gfx-cuda` tests carry a ctest label and run on the self-hosted NVIDIA box only.

---

## 8. Decisions for Ratification (continuing PLATFORM.md §18)

| # | Decision | Rationale / trade accepted |
|---|---|---|
| D19 | **Interop at buffer level only** — CUDA never touches `VkImage` memory; buffer→image conversion happens on the Vulkan timeline | Sidesteps tiling/alignment quirks (§19) at the cost of one on-GPU pass per update — the same shape the Metal compute path already has |
| D20 | **UUID-matched device pairing**, interop disabled (not erred) on mismatch | Hybrid laptops and CPU-Vulkan environments keep working via the staged ladder; no "wrong GPU" corruption class exists |
| D21 | **Synchronous interop in v1** (`cudaStreamSynchronize` + fenced submits), semaphores deferred to V4 | Matches Metal's shipped `waitUntilCompleted` semantics; correctness first, pipelining as measured optimization |
| D22 | **Build-time SPIR-V, embedded** (vs. runtime compilation à la Metal's `newLibraryWithSource`) | Determinism contract fixes the shader forever; runtime codegen buys nothing and adds a shaderc dependency |

---

## 9. Risks

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| Driver variance in external-memory handle support | Low (core-adjacent KHR extensions, universal on NVIDIA) | Medium | Extension presence checked at pairing time; absence = interop off, staged path (D20) |
| Descriptor/lifetime bugs under `release_texture` mid-frame | Medium | Medium | Deferred-destruction queue is TDD'd first (§5 row 4); validation layers fatal in CI |
| `cuMemGetAddressRange` unavailable for exotic allocators (pools, VMM) | Low | Low | Query failure → treat as unbounded → reject to staged path; never dispatch unchecked |
| No NVIDIA hardware in hosted CI | Certain | Medium | lavapipe covers all pure-Vulkan behavior; `gfx-cuda` label runs on the Phase-4 machine; golden bytes make hardware runs reproducible |
| V2 slips | Medium | Low | V1 alone already ships Vulkan-by-default with full functional parity via staging |

---

*Companion to `PLATFORM.md` (§5.4, §7.4, §16, §17 Phase 4, §18 D11/D13/D14, §19). Implementation donors: `src/host/renderer/metal_renderer.mm` (paths, parity, bounds pattern), `src/host/tensor_bridge.{h,cpp}` (acceptance rules, CPU references), `src/host/renderer/gl_renderer.cpp` (frozen fallback contract).*
