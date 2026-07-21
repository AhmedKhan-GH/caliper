# Decisions

This is the framework decision log, mirroring `PLATFORM.md` §18.

!!! note "Source of truth"
    `PLATFORM.md` §18 remains the source of truth for these decisions until dedicated ADR files start landing here. This page is a convenience mirror; on any discrepancy, the spec wins.

| # | Decision | Status | Rationale / trade accepted |
|---|---|---|---|
| D1 | In-process C ABI + C++ sugar (not C++ interfaces, not IPC) | **Ratified** (existing code + this plan) | Zero-copy same-frame-loop USP; longevity. IPC deferred to Phase 6 for trust, not foundation. |
| D2 | Host context = service registry (`get_service`), CLAP-style | Proposed | Growth without ABI breaks for years. |
| D3 | Torch/DuckDB types never cross the ABI; `CaliperTensor` (DLPack-aligned) + Arrow C streams are the interchange | Proposed | Survives version skew; trivial conversions at the edge. |
| D4 | UI-stack pin defines the ABI epoch; applets write raw ImGui/ImPlot | Proposed | DX superpower kept; cost = applets rebuild on epoch bump (rare, CI-flagged). |
| D5 | Runtime packs host-managed; **one libtorch per process per session** | Proposed | Kills 2 GB-per-applet; honest about the multi-version limit (Phase 6 solves fully). |
| D6 | Registry = git repo (Homebrew-tap model); sideloading always works | Proposed | Zero infrastructure; curation = PR review; upgrade path exists. |
| D7 | Windows moves to `/MD`; `ui.v1` hands over ImGui allocators | Proposed | Fixes the latent static-CRT/DLL-heap crash class. |
| D8 | Applet repos migrate with history (`git filter-repo`), not from scratch | Proposed | "Own lives and histories" includes the history already written. |
| D9 | CLI = subcommands of the host binary (`caliper new/dev/package/install/publish`) | Proposed | One artifact to install; the `code` CLI model. |
| D10 | SDK license: MIT (host may stay separate) | **Decide by Phase 3** | Ecosystem needs a permissive SDK; MIT matches ImGui/ImPlot neighborhood. Apache-2.0 acceptable if patent grant desired. |
| D11 | Host ships without libtorch (bridge uses raw CUDA/Metal; metrics use embedded DuckDB) | Proposed | 50 MB host; packs on demand. |
| D12 | Audience: (b) source-building collaborators now, contracts sized for (c) later | Proposed (assumption) | Stated in §3; revisit when a real (c) user appears. |
| D13 | Renderer-agnostic ABI from epoch 2; native backends as the target — **Metal (macOS) + Vulkan (Windows)** primary, OpenGL 3.3 frozen fallback; textures cross as opaque `CaliperTextureId` | Proposed | The USP demands GPU-resident pixels; GL is deprecated on macOS and cannot touch MPS memory (today every Mac texture takes a CPU round-trip). Decided while zero external applets exist, so the renderer stays host-internal forever — no epoch bump, no applet rebuilds. GLEW dies with the fallback refactor (GLAD 3.3-core loader, §5.4). Live evidence for the GL dead end: sibling project Compass stayed on "cross-platform GL" and is stranded between 2.1 fixed-function and macOS's capped 4.1 core, with per-platform `#ifdef` include paths. |
| D14 | The bridge *allocates* texture-backed shared tensors (`alloc_shared`), not just mirrors existing ones | Proposed | Upgrades "fast device copy" to literal zero-copy for live weights/saliency; applets adopt it with one `torch::from_blob`. |
| D15 | Documentation is docs-as-code from Phase 1: MkDocs Material wiki in-repo (`docs/wiki/`, Diátaxis layout), doc pages updated in the same commit as the change, `mkdocs build --strict` gate, reference pages embed the real headers/manifests via snippets (`check_paths` — moved files fail the build). API reference generated from headers via `mkdocs-cxxdox` (libclang) adopted at Phase 2; publishing (Pages) + versioning (`mike`) at Phase 5 | **Ratified** (2026-07-01) | Docs written retroactively rot; same-commit + embed-don't-paste + strict link checks keep the wiki true mechanically. cxxdox is alpha (kfrlib, v0.1.x) but additive to the same MkDocs site and trivially droppable — the snippet-embedded pages remain the fallback. |
