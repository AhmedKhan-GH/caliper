# Windows CUDA direct image — Implementation Plan

**Goal:** Implement `docs/vulkan-update-costs.md` so a Windows tensor update reads the tensor in place, with no in-VRAM copy and no Vulkan compute pass.

**Architecture:** A runtime-loaded CUDA kernel, launched on the producer's stream, writes an exported `VkImage` through a surface object. The buffer path stays as the rung below, per update and per texture.

**Tech Stack:** C++17, Vulkan, CUDA driver API, PTX checked in and embedded (no toolkit at build or link time, D11), doctest gfx suite.

**Spec:** `docs/vulkan-update-costs.md`

## Global Constraints

- Metal is not touched. Everything is behind the Vulkan backend.
- Output stays byte-identical to `map_f32_to_rgba8`. Build the kernels with `-fmad=false` and never `-use_fast_math`.
- Failure at any rung falls back; never a crash or a wrong image. Each fallback logs once through `dev_note`.
- "Zero-copy" only names paths with zero copies of the data (§7.4).
- The kernel runs in torch's context, where a bad read poisons training. Validate shape, strides and alignment against the allocation before launch.

## Review Focus

1. A frame still sampling the image when the kernel writes it: the kernel waits on a frame timeline value.
2. Textured geometry and the offscreen canvas sample without the frame's timeline waits: retire before sampling; the direct path stays off offscreen.
3. Tensor shape differing from the texture, non-positive strides, misaligned data: fall back before launch.
4. Colormap edge values (NaN, ±inf, denormals, rounding boundaries, `vmin == vmax`): byte-exact test against the CPU reference.
5. Mixing direct, buffer and CPU-staged updates on one texture, and streams changing between updates: ordered through the texture's timeline.

---

## Tasks

- [x] **W1:** `kMinImageCount = 3`. Gate: gfx suite.
- [x] **W0:** Update cost benchmark and a working `device_update_wait()`. Gate: before numbers recorded.
- [x] **W2.1:** Optional `direct_api()` driver table, kernels and embedded PTX. Gate: module loads and both functions resolve.
- [x] **W2.2:** Exportable images resting in `GENERAL`, with the buffer path unchanged. Gate: existing suite byte-exact.
- [x] **W2.3:** Direct path, frame timeline, sampling hazards. Gate: new byte-exact tests for Review Focus 1–5, plus the old tests forced onto the buffer path.
- [x] **W2.4 (added):** A second image per texture. MNIST showed one image costs training at least 37%.
- [ ] **W3:** Deferred release and a small image pool. Not done: W2 already took release+create from about 3–4 ms to about 1 ms.
- [x] **Proof:** `embed_scope` (MNIST) runs training on the real app; results in `docs/vulkan-update-costs.md` §7.
- [x] **Docs and site:** ratify D25, update status docs and `site/`.
