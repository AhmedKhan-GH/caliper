# Phase 2C — HostRenderer + Metal Backend + `tensor_bridge.v1` Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:subagent-driven-development. Checkbox steps. This plan is design-heavy: where a task gives acceptance criteria + skeleton instead of verbatim code, the criteria are the contract — report DONE_WITH_CONCERNS with specifics if reality fights them.

**Goal:** The USP, productized (PLATFORM.md §5.4/§7.4, step 3 of the ratified sequencing): a renderer-agnostic `HostRenderer` with **Metal** (primary-in-waiting) and **GL** (current, default until 2D) backends, the frozen `caliper.tensor_bridge.v1` service on top, a torch adapter in the SDK sugar, and the payoff demo — **MLScope's first-layer conv kernels as live textures, with zero CPU staging on Metal**.

**Architecture (grounded, 2026-07-02):**
- **Donor pattern being replaced:** both real applets CPU-stage today (`repnet_demo.cpp:324`, `opengllama.cpp:662`: float array → RGBA8 → `glTexImage2D` → `ImGui::Image`). The bridge's GL path formalizes that (staged, frozen fallback); the Metal path deletes the CPU hop.
- **MPS interop seam:** a torch MPS tensor's storage pointer IS its `id<MTLBuffer>` (the documented custom-Metal-kernel idiom: `(__bridge id<MTLBuffer>)t.storage().data()`); `torch/mps.h` publicly exposes `synchronize()/commit()/get_command_buffer()`. The SDK's torch adapter (applet-side, C7) fills `CaliperTensor{data=buffer-bridge-ptr, device=METAL}` and synchronizes before handing over; `CaliperTensor.stream` stays NULL/reserved in v1 (sync-on-update — ~18 updates/run in the demo, negligible; async ordering is a later optimization, noted in docs).
- **No `byte_offset` in the frozen tensor:** adapter REQUIRES `storage_offset()==0 && is_contiguous()` for METAL tensors (fresh weight/kernel tensors satisfy this; `.contiguous()` fixes the rest — asserted with a clear message).
- **The ladder (spec §7.4/§19):** per-texture degradation `alias → device blit → CPU staging`. Metal tries a linear-texture alias over the buffer when `bytesPerRow` meets `minimumLinearTextureAlignment`, else a blit/compute-encoder copy (still GPU-resident — the exit criterion is NO CPU STAGING on Metal, alias not required); GL is always staged.
- **Backend selection:** `CALIPER_RENDERER=metal|gl` env var, default **gl** until 2D removes the last raw-GL applet (the ratified gate). Metal init failure → GL fallback with a stderr warning, never a crash.

**Tech Stack:** as 2B; plus `imgui_impl_metal.mm` (vendored ✓), QuartzCore/CAMetalLayer, a `.metal`-free approach for the colormap (compute shader compiled from source string at runtime — no build-time metallib plumbing), new windowed graphics test binary `caliper_gfx_tests` (hidden window; runs in the mac GUI session, ctest-registered).

## Global Constraints

- All 2A/2B constraints carry over (TDD where logic is testable, green tasks, trailer `Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>`, docs-ride-along, explicit-path staging, `build/`, orchestrator merges, index.lock retry).
- **Branch:** `platform/phase-2c` from `main`.
- **Frozen id:** `"caliper.tensor_bridge.v1"`; `CaliperTextureId` = opaque `uint64_t`, 0 invalid, casts to `ImTextureID`, NEVER a raw GL/Metal handle (§5.4 — the bridge keeps an id→backend-handle table).
- **Colormaps v1:** `CALIPER_CMAP_VIRIDIS=0, CALIPER_CMAP_MAGMA=1, CALIPER_CMAP_RDBU=2` (256-entry RGBA8 LUTs, shared constant table in the host, same numeric output on both backends).
- **§16 contract:** tensor→texture **pixel-exact vs a CPU reference, run per backend** — `caliper_gfx_tests` uploads known tensors, reads the texture back, compares byte-for-byte.
- **Host never links torch** (D11): the bridge consumes `CaliperTensor` only; all torch knowledge lives in the applet-side adapter header.
- **The §6c rule holds:** applets still never issue graphics calls — MLScope's kernel grid goes through the bridge exclusively. NO CPU-staged kernel visualization anywhere in the exemplar, even as a fallback branch (on GL the *bridge* stages; the applet code is identical).
- **Do not touch:** `applets/*` internals (2D/2F migrate them), other examples, shipped frozen headers, `third_party/`, `cmake-build-debug/`.

## File Map

```
src/host/renderer/host_renderer.h        C1 (interface + factory)
src/host/renderer/gl_renderer.cpp        C1 (extraction of today's path)
src/host/renderer/metal_renderer.mm      C2
src/main.cpp                             C1 (render path via interface), C2 (selection)
sdk/include/caliper/services/tensor_bridge_v1.h  C3 (frozen §7.4)
tests/test_abi.cpp, tests/abi_c_check.c  C3
src/host/colormaps.h                     C4 (LUT tables, header-only)
src/host/tensor_bridge.h/.cpp            C4 (service impl over HostRenderer)
tests/gfx/test_bridge_gl.cpp             C4 (windowed pixel-exact, GL)
tests/gfx/CMakeLists.txt                 C4 (caliper_gfx_tests)
tests/gfx/test_bridge_metal.cpp          C5 (same fixtures, Metal)
sdk/include/caliper/adapters/torch.hpp   C7 (applet-side, torch-including)
examples/ml_scope/ml_scope.cpp           C8 (kernel grid — THE demo)
examples/ml_scope/ml_scope.caliper.toml  C8 (tensor_bridge moves required? NO — stays optional, probe pattern)
src/host/host_services.h/.cpp            C6 (vend bridge; kIds=6)
docs/wiki/reference/services/tensor-bridge-v1.md  C3 stub → C9 semantics
docs/wiki/explanation/architecture.md    C9 (HostRenderer section)
CMakeLists.txt                           C1/C2/C4 wiring
```

Task order note: C6 (vend) lands between C5 and C7 so the gfx tests exercise the real vended table.

---

### Task C1: `HostRenderer` interface + GL extraction (pure refactor)

**Files:** Create `src/host/renderer/host_renderer.h`, `src/host/renderer/gl_renderer.cpp`; Modify `src/main.cpp`, root `CMakeLists.txt`.

**Interface (this is the frozen-for-the-phase contract — later tasks consume it):**
```cpp
#pragma once
#include <caliper/tensor.h>
#include <cstdint>
#include <memory>
struct GLFWwindow;

namespace caliper_host {

// Host-internal renderer seam (PLATFORM.md §5.4). The ABI never sees this;
// backends are swappable forever because applets only see ImGui + bridge ids.
class HostRenderer {
public:
    virtual ~HostRenderer() = default;
    virtual bool init(GLFWwindow* window) = 0;      // after glfwCreateWindow
    virtual void new_frame() = 0;                   // backend NewFrame calls
    virtual void render(int fb_w, int fb_h) = 0;    // clear + RenderDrawData + present
    virtual void shutdown() = 0;
    virtual const char* name() const = 0;           // "gl" / "metal"

    // Texture ops the bridge builds on. data is tightly-packed RGBA8.
    virtual uint64_t tex_create_rgba8(int w, int h) = 0;            // 0 on fail
    virtual bool tex_upload_rgba8(uint64_t tex, const void* data,
                                  int w, int h) = 0;                // full update
    virtual void tex_release(uint64_t tex) = 0;
    virtual uint64_t tex_imtexture_id(uint64_t tex) = 0;            // for ImGui::Image

    // Device-resident update: src is a CaliperTensor whose data lives on this
    // backend's device (METAL buffer for metal). Returns false -> caller
    // falls back to CPU staging. GL always returns false (frozen fallback).
    virtual bool tex_update_from_device(uint64_t tex, const CaliperTensor& t,
                                        const uint32_t* lut256 /*nullable*/,
                                        float vmin, float vmax) = 0;

    // GLFW pre-window hint setup for this backend (GL profile vs NO_API).
    virtual void window_hints() = 0;
};

std::unique_ptr<HostRenderer> make_renderer(const char* name); // "gl"|"metal"|nullptr->default
}
```

**Steps:** (this is glue refactor — no new unit tests; the proof is behavior-identical run)
- [ ] `gl_renderer.cpp`: move the existing main.cpp GL/ImGui-backend code (`glfwWindowHint` GL bits, `glewInit`, `ImGui_ImplGlfw_InitForOpenGL` + `ImGui_ImplOpenGL3_Init`, per-frame `*_NewFrame`, viewport/clear/`RenderDrawData`/`glfwSwapBuffers`, shutdowns) behind the interface. Texture ops = the classic pattern from the donors (`glGenTextures`/`GL_RGBA8`/`glTexSubImage2D`/`glDeleteTextures`; `tex_imtexture_id` returns the GL name; `tex_update_from_device` returns false).
- [ ] `main.cpp`: hold `std::unique_ptr<HostRenderer> renderer_`; call `window_hints()` before window creation, `init()` after; replace the inlined GL calls in `run()`/`cleanup()` with `new_frame()`/`render(dw,dh)`/`shutdown()`. IntroScreen's own GL (`render_3d`) stays direct for now (it's host-internal GL, dies with the backend flip in 2D — add a `renderer_->name()=="gl"` guard so Metal skips it, with a TODO(2D)).
- [ ] Verify: build; full ctest; headless 10s; the human sees identical app behavior (checklist at C9).
- [ ] Commit `refactor(host): HostRenderer seam — GL path extracted, behavior identical`.

### Task C2: Metal backend

**Files:** Create `src/host/renderer/metal_renderer.mm`; Modify `src/main.cpp` (selection), root `CMakeLists.txt` (QuartzCore framework, imgui_impl_metal into the ui-stack objc build).

**Acceptance criteria (implementer designs within these):**
- [ ] `window_hints()` sets `GLFW_CLIENT_API, GLFW_NO_API`. `init()`: `MTLCreateSystemDefaultDevice`, command queue, `CAMetalLayer` attached to the GLFW window's NSWindow contentView (`glfwGetCocoaWindow`), `ImGui_ImplGlfw_InitForOther` + `ImGui_ImplMetal_Init`. Per-frame: drawable acquire, render pass (clear to the host's background color), `ImGui_ImplMetal_NewFrame/RenderDrawData`, present, commit. Handle drawable-size changes from framebuffer size each frame.
- [ ] Texture ops: `MTLTextureDescriptor` RGBA8Unorm shared-storage textures; `tex_upload_rgba8` via `replaceRegion`; `tex_imtexture_id` returns the `(__bridge void*)id<MTLTexture>` as uint64 (what imgui_impl_metal expects as ImTextureID). Map uint64 ids to strong ObjC references in an internal table (never hand raw retained pointers to callers as ids — the table owns lifetime).
- [ ] `tex_update_from_device`: v1 = GPU-side path for METAL tensors — blit/compute encoder consuming the incoming `id<MTLBuffer>` (bridge-cast from `t.data`): f32 + LUT → compute shader (source-string compiled once, 256-entry LUT in a small buffer, vmin/vmax normalize, write RGBA8 texture); u8 HWC → blit or the same shader passthrough. Alias attempt (linear texture over the buffer) may be included if alignment permits but is NOT required for this task's acceptance — the requirement is **no CPU roundtrip**.
- [ ] Selection in `main.cpp`: `make_renderer(getenv("CALIPER_RENDERER"))`; default "gl"; if metal `init()` fails → destroy, fall back to GL with a stderr warning.
- [ ] `enable_language(OBJCXX)` already exists (A3); imgui_impl_metal.mm compiled into a small `imgui_metal_backend` objc library or直接 into the caliper exe sources — implementer's call, state it.
- [ ] Verify: build both paths; `CALIPER_RENDERER=metal ./build/caliper` headless 10s crash-free AND the log line names the active backend; GL default unchanged; full ctest.
- [ ] Commit `feat(host): Metal renderer backend (CAMetalLayer + imgui_impl_metal), env-selectable, GL default until 2D`.

### Task C3: `tensor_bridge_v1.h` — frozen

**Files:** Create `sdk/include/caliper/services/tensor_bridge_v1.h`; Modify `tests/test_abi.cpp`, `tests/abi_c_check.c`; doc stub + nav.

- [ ] TDD as A1/B2. Header verbatim from PLATFORM.md §7.4 with the C-hygiene pass: id define, `CaliperTextureId` typedef (uint64_t, 0 invalid, doc comment "casts to ImTextureID; never a raw graphics handle"), colormap enum (VIRIDIS/MAGMA/RDBU per Global Constraints), table `{struct_size, texture_from_tensor(t,flags), update_texture(tex,t), release_texture(tex), texture_from_tensor_mapped(t,colormap,vmin,vmax,flags), alloc_shared(dtype,ndim,shape,out_tensor,out_texture), free_shared(tex)}` — includes `caliper/tensor.h`. Doc comment states v1 acceptance: 2-D `(H,W)` f32 (mapped) or 3-D `(H,W,C<=4)` u8 (direct), contiguous, device CPU or the active backend's device; anything else returns 0/false with a log line.
- [ ] Commit `feat(sdk): tensor_bridge.v1 service header (Phase 2C)`.

### Task C4: Bridge core + GL path + the pixel-exact harness

**Files:** Create `src/host/colormaps.h`, `src/host/tensor_bridge.h`, `src/host/tensor_bridge.cpp`, `tests/gfx/CMakeLists.txt`, `tests/gfx/test_bridge_gl.cpp`; Modify root `CMakeLists.txt` (gfx tests subdir under BUILD_TESTS), `tests/CMakeLists.txt` untouched.

**Design (binding):**
- `colormaps.h`: three `constexpr uint32_t kCmap*[256]` RGBA8 LUT tables (generate viridis/magma/RdBu numerically — matplotlib-derived values, comment the provenance) + `map_f32_to_rgba8(src,w,h,lut,vmin,vmax,dst)` CPU reference used by BOTH the GL staging path and the tests' expected-value computation (single source of truth for pixel-exactness).
- `TensorBridge` class: owns `HostRenderer&`, an id→{renderer_tex, w, h} table, and implements the six service ops. Validation per the C3 doc comment; the CPU staging path handles: CPU f32 (H,W) mapped → LUT; CPU u8 (H,W,C) → RGBA8 expand; device tensors → try `renderer.tex_update_from_device`, if false and the tensor is CPU-reachable stage, else fail with log. `alloc_shared` in THIS task: CPU malloc-backed (`device=CALIPER_DEV_CPU`, tensor.data = the buffer) + a texture; `free_shared` releases both; the Metal shared-buffer upgrade is C5's.
- **Harness:** `caliper_gfx_tests` — doctest binary creating ONE hidden GLFW window (`GLFW_VISIBLE, GLFW_FALSE`) + GL renderer + bridge per run. Tests: known 4×4 f32 ramp → `texture_from_tensor_mapped(viridis)` → `glGetTexImage` readback → byte-compare vs `map_f32_to_rgba8` reference; known 2×2×3 u8 → direct → readback compare; `update_texture` changes pixels; invalid tensors (wrong ndim, non-contiguous strides, f16) → id 0; `alloc_shared` roundtrip (write buffer → update → readback). Register with ctest (label `gfx`; note in the plan: requires a GUI session — fine on this machine, skipped-with-message if `glfwInit` fails so headless CI doesn't redline).
- [ ] TDD: harness + tests first (RED: no `tensor_bridge.h`), implement, GREEN (both `ctest` suites).
- [ ] Commit `feat(host): tensor bridge core + GL staging path, pixel-exact gfx harness`.

### Task C5: Metal bridge path — pixel-exact on the GPU

**Files:** Create `tests/gfx/test_bridge_metal.cpp`; Modify `src/host/renderer/metal_renderer.mm` (complete `tex_update_from_device` per C2 criteria if any gap remains), `tests/gfx/CMakeLists.txt`.

- [ ] Same fixtures as C4 but: renderer = metal, and the interesting cases feed **device-resident** tensors — the test allocates an `MTLBuffer` directly (no torch in tests!), fills it via `contents` (unified memory), builds `CaliperTensor{device=METAL, data=(__bridge void*)buffer}`, and expects the bridge to produce pixel-identical output to the C4 CPU reference **without any CPU staging** (assert via the renderer's returned path: extend `tex_update_from_device` to be observable — e.g., the bridge records `last_path` = "device-blit"/"device-compute"/"staged" per texture, and the test asserts it's a device path; this observability also feeds the C8 demo line).
- [ ] Readback for verification is allowed to use a blit-to-shared-buffer + `synchronize` (readback is test-only, not the render path).
- [ ] The compute-shader LUT output must byte-match the CPU reference exactly (integer LUT indexing — define the index math once in a comment shared by both implementations: `idx = clamp((v - vmin)/(vmax-vmin),0,1)*255 + 0.5`).
- [ ] Verify: `ctest` incl. gfx label green on BOTH backends; full suite green.
- [ ] Commit `feat(host): Metal tensor-bridge path — GPU-resident updates, pixel-exact vs CPU reference`.

### Task C6: Vend `tensor_bridge.v1` (+ Metal `alloc_shared` upgrade)

**Files:** Modify `src/host/host_services.h/.cpp` (bridge thunks over a host-global `TensorBridge` bound to the active renderer — expose `host_tensor_bridge()`; `kIds`=6), `src/main.cpp` (construct bridge after renderer init, hand to services), `src/host/tensor_bridge.cpp` (Metal alloc_shared: shared `MTLBuffer`, tensor.data = `contents` pointer as `CALIPER_DEV_CPU` **documented v1 semantics**: unified-memory zero-copy for CPU writers; device writers use `update_texture` — the honest ladder, noted in docs and the header was already written to allow it), `sdk/include/caliper/caliper.hpp` (thin `caliper::TensorBridge` wrapper, falsy-inert), `tests/test_sugar_services.cpp` (fake-table routing tests).

- [ ] TDD on the sugar wrapper; service-thunk wiring mirrors metrics (B4 pattern, incl. teardown-order comment — bridge global declared per the load-bearing-order rule, and it must be destroyed AFTER applets are torn down: it lives in main-owned scope handed to services, not a TU-static — state the chosen lifetime in the report).
- [ ] Full ctest + gfx green; loader green with 6 ids.
- [ ] Commit `feat(host): vend tensor_bridge.v1; caliper::TensorBridge sugar; Metal alloc_shared`.

### Task C7: Torch adapter — `sdk/include/caliper/adapters/torch.hpp`

**Files:** Create the header; Create `tests/test_torch_adapter.cpp` (linked into a NEW small test target `caliper_torch_tests` that links torch — keep torch out of `caliper_tests`); Modify `tests/CMakeLists.txt`.

- [ ] Header-only, applet-side (includes `<torch/torch.h>` — never included by the host): `caliper::torch_adapter::to_tensor(const at::Tensor&) -> CaliperTensor` — CPU: `data_ptr()`; MPS: enforce `is_contiguous() && storage_offset()==0` (TORCH_CHECK with an actionable message: "call .contiguous() / clone the view"), `data = t.storage().data()` (the buffer-bridge idiom, comment explains), `device=CALIPER_DEV_METAL`, and the helper `sync_before_handoff()` calling `torch::mps::synchronize()` (adapter's `to_tensor_synced` does both — the one MLScope uses). dtype map f32/f16/bf16/i64/i32/u8; shape/strides copy.
- [ ] Tests (CPU-only assertions run everywhere; MPS branch asserts under `torch::hasMPS()`): field fidelity for a CPU tensor; contiguity TORCH_CHECK fires on a transposed MPS view; MPS tensor's `data` equals `storage().data()`.
- [ ] Docs stub note in tensor.md ("adapters/torch.hpp" section).
- [ ] Commit `feat(sdk): torch adapter — CaliperTensor from at::Tensor with MPS buffer-bridge`.

### Task C8: MLScope kernel grid — THE demo

**Files:** Modify `examples/ml_scope/ml_scope.cpp`, `examples/ml_scope/CMakeLists.txt` (nothing new — torch already linked), manifest UNCHANGED (bridge stays optional — the probe pattern is the exemplar's point).

- [ ] `ML-EXEMPLAR 7`: probe `caliper::TensorBridge` in on_init. At every accuracy-eval cadence point in the job, if the bridge is truthy: take conv1 weights `(8,1,3,3)` and build a single tiled f32 tensor **on the training device** — squeeze to `(8,3,3)`, then `torch::cat` the eight `(3,3)` kernels along columns with a 1-px separator column (filled with the tile's min value) between them, yielding `(3, 8*3+7)`; compute `wmin/wmax` from the tile; hand off via the adapter's `to_tensor_synced`, then `texture_from_tensor_mapped(..., CALIPER_CMAP_VIRIDIS, wmin, wmax, 0)` on first call / `update_texture` on subsequent ones (same texture id, same shape every time). The UI shows `ImGui::Image` scaled ~8× with nearest sampling caveat noted, captioned with the bridge path (`device-compute` on Metal / `staged` on GL — read from a new tiny status the applet keeps; the bridge logs it).
- [ ] The tile tensor is built on the TRAINING DEVICE and handed device-resident on Metal — on the Metal backend this whole path never touches CPU memory (the demo claim). On GL, identical applet code; the bridge stages (say so in the caption).
- [ ] Update the "watch this space" text → gone; ML-EXEMPLAR 4 comment rewritten to its fulfilled form.
- [ ] Verify: build, full ctest + gfx, headless 10s both backends. Human demo checklist (C9).
- [ ] Commit `feat(ml_scope): live conv-kernel grid via tensor_bridge.v1 — the zero-copy demo`.

### Task C9: Docs + demo + merge (orchestrator merges)

- [ ] `tensor-bridge-v1.md` semantics: the ladder (alias/blit vs staged, per backend), v1 acceptance rules, `alloc_shared` v1 semantics (unified-memory CPU-writer zero-copy), sync model (adapter synchronizes; stream reserved), `CaliperTextureId` lifetime rules. `architecture.md`: HostRenderer section replacing the GL-era text. `tensor.md`: adapter section. Strict build.
- [ ] **Human demo checklist:** (1) default GL run — app identical to yesterday, MLScope kernel grid appears during training, caption says `staged`; (2) `CALIPER_RENDERER=metal` — full app parity (cards, dashboard, MLScope training), kernel grid caption says a device path, kernels visibly sharpen from noise as loss falls; (3) SignalScope/watchdog/quarantine spot-checks on Metal (the §15 machinery is backend-agnostic — prove it); (4) `ctest` gfx label green on both backends.
- [ ] Ledger + merge `Phase 2C: HostRenderer + Metal + tensor_bridge.v1 (PLATFORM.md §17 Phase 2, step 3)`.

## Exit Criteria

| Requirement | Proof |
|---|---|
| Renderer-agnostic seam; ABI untouched | C1 refactor + no sdk header changes except C3's new frozen file |
| Metal backend at app parity | C2/C9 human checklist |
| Bridge pixel-exact per backend (§16) | C4/C5 gfx harness, byte-compare vs one CPU reference |
| **No CPU staging on Metal** (the USP claim) | C5 observable-path assertion + C8 device-resident tile handoff |
| Torch never enters the host | C7 adapter is applet-side; host targets unchanged (link-graph check in C9) |
| Kernel-grid demo live on the benchmark | C8 + human checklist item 2 |
| GL default until 2D (ratified gate) | C2 selection logic + checklist item 1 |

## Spec Deviations (deliberate)

1. `alloc_shared` v1 returns a CPU-device unified-memory tensor (zero-copy for CPU writers; MPS writers use `update_texture`) — torch cannot wrap foreign MTLBuffers as MPS tensors via public API; D14's full vision revisits when/if torch grows the hook. Documented, not silent.
2. Sync model is synchronize-on-handoff in the adapter; `CaliperTensor.stream` reserved. Noted for a future `.v2` or flags bit.
3. Metal alias path is best-effort (alignment-gated), blit/compute is the guaranteed GPU-resident path — the exit criterion is no-CPU-staging, not literal aliasing.
4. IntroScreen's own GL 3D background renders only on the GL backend until 2D (TODO'd).
5. gfx tests require a GUI session; they self-skip (with a visible message) when glfwInit fails.

## Risks

- CAMetalLayer/GLFW integration quirks (drawable resize, retina scale) — C2 budgets for iteration; parity checklist is the gate.
- Compute-shader LUT float determinism vs CPU reference — pinned by the shared index-math definition; if a GPU rounding edge appears, quantize the normalize step identically on both sides (the test will catch it, the fix is spec'd).
- torch storage-bridge idiom is internal-adjacent (public headers, documented usage in custom-kernel ecosystem) — pinned to the vendored torch 2.5.1; revisit at any torch bump (golden note in ledger).

