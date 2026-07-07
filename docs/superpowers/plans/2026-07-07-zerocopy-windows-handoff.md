# Windows Handoff — Zero-Copy Import v1.2 (Tasks 6–8 + Hardware Verification)

**For:** an agent on the Windows/NVIDIA box (RTX 500 Ada precedent machine, driver ≥ 596.47).
**Branch:** `feat/zerocopy-arbitrary-cuda` (do all work here; do not merge to main — the human merges after the final review).
**Parent plan:** `docs/superpowers/plans/2026-07-07-zerocopy-arbitrary-cuda.md` — read its Global Constraints section first; every one of them binds you. Task briefs live at `.superpowers/sdd/task-{6,7,8}-brief.md`; append your progress to `.superpowers/sdd/progress.md` (section `=== ZEROCOPY IMPORT v1.2 ===`).

## What this feature is

Bridge v1.2 lets an applet allocate torch CUDA tensors from an *exportable* memory pool (`cuMemCreate` + shareable Win32 handle); the host imports the pool block into Vulkan once and updates textures **directly from it at byte offsets** — deleting the one `cuMemcpyDtoD` per update that the general path pays. "Zero-copy" wording is reserved for exactly this path (PLATFORM.md §7.4 discipline).

## State as delivered from macOS (all reviewed & approved, mac suites green)

| Commit | Task | Delivered |
|---|---|---|
| `030a9bc` | — | The TDD plan |
| `8ce3863` | 1 | `sdk/include/caliper/services/tensor_bridge_v1_2.h` (id `caliper.tensor_bridge.v1_2`, caps bit `CALIPER_BRIDGE_CAP_IMPORT_ALLOC (1u<<1)`, `import_allocation`/`release_allocation`/`update_texture_from_alloc`) + `caliper::Bridge` wrapper + layout tests |
| `30953fb` + fix `22a5426` | 2 | Host `TensorBridge` bookkeeping (id table, frozen-gate reuse, **overflow-safe** bounds check), `HostRenderer` seam virtuals (defaulted unsupported), `kBridge12` table; stub-renderer TDD (caps gate, lifecycle, double-release, wrap-case) |
| `94b934a` | 3 | `sdk/include/caliper/adapters/alloc_registry.hpp` — pointer→(allocation, offset) interval map, pure C++, unit-tested |
| `23d823a` | 4 | `sdk/include/caliper/adapters/exportable_pool.hpp` — torch-2.5.1 MemPool over `cuMemCreate` shareable allocations (API pins verified against `third_party/libtorch/include`: `createCustomAllocator` CUDAPluggableAllocator.h:49-53, `MemPool` CUDACachingAllocator.h:461-464, `beginAllocateToPool` :372) + self-contained driver mini-loader + `to_bridge()` glue with per-base import cache/negative cache; tripwire in `tests/test_torch_adapter.cpp` |
| `cef24d1` | 5 | `VulkanRenderer` overrides: `import_external_allocation` (VkImportMemoryWin32HandleInfoKHR, DuplicateHandle, full unwind on failure), `tex_update_from_imported` (descriptor-offset f32 / bufferOffset u8, **no D2D copy**, pipelined signal-without-copy + fenced sync fallback), `release_external_allocation`, `supports_external_import`; paths `"compute-imported"`/`"blit-imported"` |

**NEVER compiled or run on Windows yet.** Task 5's TU is WIN32-only; it passed by-inspection review (10 named risks traced) but your first build is its first compile.

## Your job, in order

### 0. Build + regression floor
Configure per README (LIBTORCH_VARIANT=cu121+; Vulkan via volk/FetchContent — no SDK needed). Then:
- `cmake --build build` — **expect Task 5's TU to surface compile errors**; fix mechanically (types/includes), keep the locked design, note every fix in your report.
- Full existing suites green before any new work: `ctest --test-dir build --output-on-failure` (unit + torch + gfx). The pre-existing `gfx-cuda` rows (device f32 compute path, alloc_shared, burst pipelining) must pass — that proves the T5 refactors (`write_cmap_set_src`, `record_blit_body_src`, `ensure_pipeline_objects` split) kept existing behavior byte-identical. **If these fail, stop and fix before proceeding — it means the refactor broke the donor paths.**

### 1. Task 6 — VmmApi + five gfx-cuda rows (brief: `.superpowers/sdd/task-6-brief.md`)
- Add a **separate optional** `VmmApi` table to `src/host/cuda_driver.{h,cpp}` — `vmm_api()` returns nullptr if ANY symbol missing; do NOT touch the core `Api`'s all-or-nothing rule (cuda_driver.cpp:66). Symbols: `cuMemCreate`, `cuMemAddressReserve`, `cuMemMap`, `cuMemSetAccess`, `cuMemExportToShareableHandle`, `cuMemUnmap`, `cuMemRelease`, `cuMemAddressFree`, `cuMemGetAllocationGranularity` (none have `_v2` exports; mirror the existing entries[] style). Struct layouts: cross-check field-for-field against the transcriptions in `exportable_pool.hpp` (do not include it host-side).
- Five test rows in `tests/gfx/gfx_main.cpp`, donor case at :844-877, same skip-guards plus `vmm_api()` presence, TDD (write first, watch fail, implement helper, green):
  1. **Byte-exact at offsets**: VMM alloc (granularity-padded, WIN32 handle type, mapped+RW), `cuMemcpyHtoD` a 17×9 f32 grid at offset 0 and 5×3 at offset 512; export handle; `bridge.import_allocation(...)` nonzero; textures created via `texture_from_tensor_mapped` (CPU seed of matching shape); `update_texture_from_alloc(tex, alloc, 0/512, &desc)` true; `last_device_path() == "compute-imported"`; `debug_readback_rgba8` **byte-equal** to `map_f32_to_rgba8` for BOTH offsets.
  2. **Misaligned offset 4** → false, pixels unchanged, `last_device_path()` untouched.
  3. **u8 3-channel at offset 512** → `"blit-imported"`, readback equals `expand_u8_to_rgba8`.
  4. **Release + reuse**: update-after-release false (fallback contract); 50× import/release loop, ids strictly increasing, no validation errors.
  5. **Bounds**: `offset + extent > size` → false (bridge gate + renderer re-check).

### 2. Task 7 — gpt_scope opt-in (brief: `.superpowers/sdd/task-7-brief.md`)
Lazy `ExportablePool` when `caps & CALIPER_BRIDGE_CAP_IMPORT_ALLOC` and torch CUDA available; attention snapshot materialized inside `pool.use()`; upload tries `pool.to_bridge()` → `update_texture_from_alloc`, falls through to the existing `update_texture` path on any miss/false. Status line: **"zero-copy (imported pool)"** only when the imported path ran this frame. Determine bridge-side release ownership by reading `exportable_pool.hpp` (if `to_bridge`'s import cache doesn't release, own a small map in the applet). Fallback purity is a hard requirement: with the cap absent the applet must behave byte-identically to today.

### 3. Hardware verification checklist (proves Task 5)
- [ ] All five Task-6 rows green (`ctest -L gfx`), plus the pre-existing gfx-cuda rows still green.
- [ ] gpt_scope live: train on CUDA, attention heatmaps sharpen, status line shows **zero-copy (imported pool)**, no hitching; cancel/relaunch mid-run clean.
- [ ] `CALIPER_RENDERER=gl` run: identical visuals, CPU-staged status, fallback intact.
- [ ] Review-flagged first-suspects if import unexpectedly falls back or fails:
  1. **Memory-type intersection** (vulkan_renderer.cpp ~:504-510): import uses the buffer's `memoryTypeBits` only; the strict path intersects with `vkGetMemoryWin32HandlePropertiesKHR::memoryTypeBits`. If `vkAllocateMemory` fails on import, add that intersection.
  2. **`allocationSize = size_bytes < mr.size`** (~:498): if bind/alloc fails, the padded-size assumption broke; log both sizes.
  3. Torch pool sub-allocation offsets should be 512-aligned; if `minStorageBufferOffsetAlignment` rejections appear in logs, report actual offsets — do NOT silently relax the gate.
- [ ] Stress: concurrent-training + imported updates (mirror the 10× stress precedent in `tests/test_torch_adapter.cpp` CUDA cases).

### 4. Task 8 — docs (brief: `.superpowers/sdd/task-8-brief.md`) — only after hardware green
ZEROCOPY.md crossings-table row + "Imported allocations" paragraph; `docs/vulkan-cuda-backend.md` As-Built V5 entry (include your compile-fix notes and measured update timings if easy); WHITEPAPER.md §9 floor-bullet replacement (exact sentence in the brief). State only what you verified — "hardware-verified" only after §3 is green.

### 5. Close out
- Fix-forward commits on the branch, conventional style, trailer: `Co-Authored-By: <your model> <noreply@anthropic.com>`.
- Stage ONLY files you changed — never `WHITEPAPER.md` (root; the human's separate work), `graphify-out/`, or `.superpowers/` scratch.
- Dispatch/perform the **final whole-branch review** (base `f84b27c` = merge-base with main) per superpowers:requesting-code-review; triage the Minors ledgered in `progress.md` (T1 layout-test sizeof pin, T3 extent-0/dead-branch, T4 fd-0 sentinel + locking asymmetry + TOCTOU comment, T5 memory-type intersection + pending_frame_waits_ comment).
- Ledger every step; leave the branch unpushed unless the human says otherwise.

## Honesty rules (non-negotiable, from PLATFORM.md)
Byte-exact means byte-exact — no tolerance comparisons. A failed import is a `false` and a staged frame, never a crash or a wrong image. Status lines and docs claim only the path that actually ran. If a §16-style row cannot pass, report it failing; do not weaken the assertion.
