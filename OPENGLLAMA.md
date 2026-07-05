# llama.cpp & OpenGLlama — Status and Plan

| | |
|---|---|
| **Status** | Plan / draft for review |
| **Date** | 2026-07-05 |
| **Owner** | Ahmed Khan |
| **Scope** | Why `third_party/llama.cpp` is in the tree, what the archived `OpenGLlama` applet is, and the phased plan to revive it as a built, cross-platform applet that runs arbitrary Ollama (GGUF) models through embedded llama.cpp and visualizes their attention live through `caliper.tensor_bridge.v1`. |
| **Related** | `PLATFORM.md` (applet/loader model, runtime packs), `ZEROCOPY.md` + `docs/vulkan-cuda-backend.md` (the torch/Vulkan device path this eventually reuses), `APPLETS.md` |

> **TL;DR.** llama.cpp is pinned, builds, and is currently linked by *nothing* — it exists only for the archived `OpenGLlama` applet, which is fully written (~2,850 lines) but not built (it lives under `applets/legacy/`, which the build glob skips). The applet embeds llama.cpp, exposes an **Ollama-compatible HTTP server**, discovers models from the local **Ollama model store**, and streams **attention heatmaps** to the tensor bridge. The plan below un-archives it, gets it building on Windows, and — as a stretch — routes its GPU attention tensors through the same Vulkan/CUDA device path the torch applets use.

---

## 1. Where things stand today

### 1.1 llama.cpp (the dependency)

| Fact | Detail |
|---|---|
| **Pin** | Tag `b9873` (`a4107133a`), fetchable. The earlier ghost-pin (an unfetchable local-fork commit) is fixed. |
| **Build** | Configured via `add_subdirectory` in `cmake/Dependencies.cmake`; forced static. On Windows the ggml backend is **CPU-only** by default. |
| **Linked by** | **Nothing active.** Not in `CALIPER_DEPENDENCY_LIBS`; no active applet or example includes `llama.h`. The only consumer is `applets/legacy/opengllama/`, which is not built. |
| **Net** | Currently **compiled dead weight** — build cost, zero runtime consumers. Parked, correctly pinned. |

**In-flight local edit (uncommitted).** `cmake/Dependencies.cmake` has a change that enables ggml's **CUDA backend on Windows when a CUDA 13+ toolkit is present** — pinning `nvcc`, setting `CMAKE_CUDA_ARCHITECTURES native`, and relying on CUDA 13's distinct DLL names (`cudart64_13.dll`) to avoid colliding with the `cudart64_12.dll` that libtorch's cu12x build ships. The reasoning is sound, but the edit is **currently inert for the active build** (nothing links ggml, so enabling CUDA there compiles nothing extra). It is speculative prep for exactly the applet this document plans. Keep it local until Phase L2.

### 1.2 OpenGLlama (the applet)

Fully implemented, **not built**. Lives at `applets/legacy/opengllama/` (~2,850 lines across `opengllama.{h,cpp}`, `ollama_server.{h,cpp}`, `ollama_models.{h,cpp}`, `model_profiles.{h,cpp}`, `plugin.cpp`). It has a working `CMakeLists.txt` and manifest (`opengllama.caliper.toml`, id `dev.ahmed.opengllama`, ABI epoch 2) — the root glob simply never descends into `legacy/`.

> **Naming note.** "OpenGLlama" is historical. The manifest already says *"bridge-native, no raw GL"*: it migrated off raw OpenGL to `caliper.tensor_bridge.v1` in Phase 2D. Consider renaming to `LlamaScope` (or similar) on revival to match the `*_scope` applet family and drop the misleading "GL".

---

## 2. What OpenGLlama actually is (architecture)

Four cooperating pieces, all already written:

1. **Embedded llama.cpp inference.** Links the `llama` static lib directly and loads GGUF models in-process — not a client of an external server. This is the only way to see inside inference; a pure Ollama client can't, because Ollama's HTTP API never exposes attention.

2. **Ollama model discovery** (`OllamaModelStore`, `ollama_models.cpp`). Scans the local Ollama install's store — `~/.ollama/models` on Unix, `%USERPROFILE%\.ollama\models` on Windows (already handled) — reads Ollama's manifests under `manifests/registry.ollama.ai/library`, and resolves the GGUF blob paths. So the applet **reuses models the user already pulled with `ollama pull`**, no re-download. "Arbitrary Ollama models" = any GGUF in that store.

3. **Ollama-compatible HTTP server** (`OllamaServer`, `ollama_server.cpp`) on port **11435**, exposing `/api/tags`, `/api/generate`, `/api/chat` backed by the embedded model (chat templating via `llama_chat_apply_template`). Any Ollama client — including `demos/ollama_client.py` — can point at it and drive inference. Uses `cpp-httplib` (vendored inside `third_party/llama.cpp/vendor/cpp-httplib`).

4. **Attention capture → bridge** (`eval_callback` in `opengllama.cpp`). Registers a ggml eval callback that intercepts `kq_soft_max-*` (softmaxed attention) and layer-output tensors during graph evaluation, pulls them off the backend with `ggml_backend_tensor_get`, averages across heads, and uploads the heatmaps through `caliper.tensor_bridge.v1` (EMA / max / recent-window aggregations, per-layer). Bridge-native, no raw GL.

**Data-path caveat that matters for the roadmap.** `ggml_backend_tensor_get` copies the attention tensor **device → host**. So today OpenGLlama feeds the bridge **CPU tensors** (the CPU-staged path) — it does *not* use the zero-copy device path, even when ggml runs on GPU. This is the seam Phase L5 addresses.

---

## 3. The goal

A built, cross-platform applet that:
- lists the models already in the user's Ollama store and loads any of them via llama.cpp;
- runs interactive inference (in-app, and via the Ollama-compatible endpoint for external clients);
- visualizes attention live as it generates, through the tensor bridge;
- on NVIDIA/Windows, eventually keeps that attention **GPU-resident** through the same Vulkan/CUDA interop the torch applets use.

---

## 4. The plan (phased, each independently shippable)

| Phase | Deliverable | Exit criterion |
|---|---|---|
| **L0 — Parked (now)** | llama.cpp CPU-only, not linked. | Current state. No action; don't commit the CUDA-13 edit yet. |
| **L1 — llama.cpp GPU on Windows** | Commit the CUDA-13 ggml enablement; verify a clean from-scratch configure/build of ggml with CUDA 13 (distinct DLLs, `-allow-unsupported-compiler`, `CMAKE_CUDA_ARCHITECTURES`). | `ggml-cuda` compiles and loads alongside torch's cu12x without DLL collision; a trivial ggml GPU op runs. Gated: only meaningful once L2 links it. |
| **L2 — Un-archive & build** | Move `opengllama/` out of `legacy/` (or teach the glob to include it), link `llama` + `cpp-httplib`, resolve the Windows build gaps in §5. | Applet builds on Windows + macOS, loads in the host, appears in the launcher, no raw-GL. |
| **L3 — Model discovery + load** | Verify `OllamaModelStore` on Windows (`%USERPROFILE%\.ollama`), load an arbitrary GGUF blob, run a generation. | Pick a model from the local Ollama store in-app; tokens stream. |
| **L4 — Visualization parity (CPU path)** | Attention heatmaps render through the bridge on the CPU-staged path (works on every backend incl. Vulkan/GL). | Live per-layer attention heatmaps during generation; the Ollama endpoint drives them too. |
| **L5 — Zero-copy LLM attention (stretch)** | A `ggml → CaliperTensor` **device** adapter: when ggml runs on the CUDA backend, keep the `kq_soft_max` tensor on-device and route it through the bridge's `CALIPER_DEV_CUDA` path (Vulkan external-memory interop) instead of `ggml_backend_tensor_get`. | Attention heatmaps take the `"compute"` device path (no CPU staging) — the LLM analog of the torch zero-copy result. |

L2–L4 deliver the user-visible goal (arbitrary Ollama-model inference + visualization). L5 is the performance capstone and the point where this work rejoins the zero-copy effort.

---

## 5. Windows-specific work items (surfaced for L2)

Concrete gaps a first Windows build will hit, none blocking:

- **cpp-httplib on Windows.** Needs Winsock (`ws2_32`) linked and works HTTP-only (no OpenSSL required for the local Ollama-compatible server). Confirm the vendored `cpp-httplib` target links `ws2_32` on MSVC; add it if not.
- **llama.cpp CUDA vs torch cudart collision.** Covered by the L1 CUDA-13 approach (distinct `cudart64_13.dll`). If staying CPU-only for L2/L3/L4, no collision — GPU is an L5 concern.
- **Applet DLL placement.** Match the other applets: `RUNTIME_OUTPUT_DIRECTORY = applets/` (the `CMakeLists.txt` already sets both LIBRARY and RUNTIME output dirs — good).
- **Model store path.** Already handled (`USERPROFILE`); verify the manifest scan against a real Windows Ollama install layout.
- **Port conflict.** 11435 avoids Ollama's own 11434; keep it (or make it configurable) so the applet and a running Ollama daemon coexist.
- **`min_host` / ABI.** Manifest declares `abi_epoch = 2`, `min_host = "0.6.0"` — matches the current host; no negotiation work expected.

---

## 6. Relationship to the zero-copy work

llama.cpp's `ggml` tensors are a **separate tensor source** from torch. Everything in `ZEROCOPY.md` / `docs/vulkan-cuda-backend.md` is torch-CUDA → Vulkan. The bridge and the Vulkan device path are **source-agnostic** — they accept a `CaliperTensor{device = CALIPER_DEV_CUDA, data = <device ptr>}` regardless of who produced it. So L5 does **not** re-do the interop; it writes a small **ggml→CaliperTensor device adapter** (the analog of `sdk/include/caliper/adapters/torch.hpp`) that hands a ggml CUDA tensor's device pointer to the bridge, plus a sync-at-handoff (`ggml_backend_synchronize`). The heavy lifting — external-memory import, bounds check, in-VRAM copy, on-GPU colormap — is already done and verified.

One caveat to design around in L5: the ggml CUDA backend must be the *same* CUDA device the Vulkan renderer paired with by UUID (spec §3.1). On a single-GPU box that's automatic; document the assumption.

---

## 7. Open decisions & risks

| # | Item | Note |
|---|---|---|
| D1 | Rename `OpenGLlama` → `LlamaScope`? | "GL" is historical/misleading; align with `*_scope` family. Cosmetic, do at L2. |
| D2 | Keep the embedded Ollama-compatible server, or in-app-only? | The server is what enables external clients + `demos/ollama_client.py`. Low cost to keep; it's already written. |
| D3 | Commit the CUDA-13 ggml edit now or at L1? | Recommend **L1** — committing now adds build time for zero current benefit and ties `main` to a CUDA-13 toolchain assumption before anything uses it. |
| R1 | ggml CUDA build under VS2026/MSVC | Mitigated by `-allow-unsupported-compiler` + CUDA 13; verify in L1 with a clean configure. |
| R2 | cpp-httplib / Winsock quirks | Low; HTTP-only, well-trodden. Link `ws2_32`. |
| R3 | Ollama store format drift | `ollama_models.cpp` parses a specific manifest layout; re-verify against current Ollama versions in L3. |
| R4 | L5 device-UUID mismatch (multi-GPU) | ggml CUDA device must match the Vulkan-paired device; assert and fall back to the CPU path (already the safe default). |

---

## 8. Immediate recommendation

Leave llama.cpp **parked and CPU-only** (Phase L0) and keep the CUDA-13 edit as a local experiment until you actually start L2. The revival is a clean, self-contained project that does **not** block or entangle the torch/Vulkan zero-copy finish line — it only *reuses* it at L5. Start at L2 (un-archive + Windows build) whenever LLM visualization becomes the priority.
