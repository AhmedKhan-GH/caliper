# Caliper Platform — Status & Roadmap

| | |
|---|---|
| **As of** | 2026-07-03 |
| **Branch / tip** | `platform/phase-2f` @ `4dde8af`+docs (Phase 2F′; ready to merge to `main`) |
| **Governing spec** | `PLATFORM.md` (repo root) — decision log D1–D18 |
| **Execution record** | `docs/superpowers/plans/*` (per-phase plans) + `.superpowers/sdd/progress.md` (ledger) |
| **Position** | **Phase 2 complete — all 8 services shipped and consumed; at the ratified strategic stop-and-evaluate line before Phase 3** |

---

## 1. The scope, as agreed in this project

Convert Caliper from a monorepo application into a **platform**: a frozen C-ABI contract, named/versioned host **services**, applets as independently-shippable artifacts, and GPU-resident ML visualization running in the same frame loop as training. Six strangler-fig phases, each ending shippable. Two structural decisions taken mid-stream shape the rest:

- **The sufficiency line (agreed):** Phases 0–2 build the platform *for an audience of one* (you). Phases 3–6 build it *for other people*. The honest stopping point to evaluate "do I invite the world?" is **the end of Phase 2** — which is exactly where we are now.
- **The flagship pivot (D16):** `repnet_demo` is defunct and will **not** be service-migrated (it stays a working legacy example). The Phase-2 generality proof and long-term reference applet is **GPTScope** — a mini-GPT on TinyShakespeare. Consequently `data.v1`/`artifacts.v1` are **demand-driven**: designed and frozen only when a real applet consumes them.

---

## 2. Progress map

### ✅ Shipped (all merged to `main`)

| Phase | Milestone | What it delivered |
|---|---|---|
| **0** | SDK extraction | Installable `caliper::sdk` CMake package; applets build against it, not the host tree; `find_package` install probe |
| **1** | ABI epoch 2 | `caliper_applet_descriptor()` + `get_service` registry; manifest-gated loader v2 with friendly refusal cards; crash quarantine (signal trampoline); frame watchdog; `CALIPER_APPLET` sugar + fixture host; **v1 ABI deleted** |
| **2A** | Compute services | `caliper.jobs.v1` (thread-per-job, ≤100 ms cancel contract) + jobs tray; `caliper.device.v1` (Metal-native detection, no ML framework linked); MLScope exemplar born |
| **2B** | Observability | MNIST CNN training; `caliper.metrics.v1` on embedded DuckDB (10k-ordered §16 contract); host **Runs dashboard** (run list, per-tag plots, EMA smoothing) |
| **2C** | **The USP** | `HostRenderer` seam + **Metal backend**; `caliper.tensor_bridge.v1` **pixel-exact on both backends** (windowed gfx harness); torch adapter (`caliper/adapters/torch.hpp`); MLScope live conv-kernel grid, GPU-resident |
| **2D** | Native coherence | Every applet bridge-native (**zero raw GL anywhere**); **Metal is the macOS default** (GL = frozen fallback); MLScope real-data viz (probe digit + feature maps) |
| **2E′** | **Flagship** | **GPTScope** — nanoGPT-style char transformer on TinyShakespeare: jobs-trained, metrics-streamed, **live evolving samples**, **per-head attention heatmaps** + hover char-highlight, temperature, perplexity |
| **2F′** | **Last two services + all-services exemplar** (D18) | `caliper.artifacts.v1` (content-addressed, deduped, run-lineaged checkpoints on DuckDB+blob files) + `caliper.data.v1` (SQL over the host store, results out as **Arrow C streams**); **EmbedScope** — MNIST net with a learned **3-D embedding bottleneck** on live ImPlot3D (blob→10 lobes, renderer-agnostic), the honest consumer of **all 8 services** (Save/Load load-bearing on artifacts; live SQL centroids/misclassified on data). Commits `626af5f..4dde8af` (+docs) |

### 🔧 Post-ship fixes (from your hands-on runs — all merged)

| Fix | Root cause | Commit |
|---|---|---|
| Accuracy learning curve | per-epoch sampling hid MNIST's sub-epoch convergence | `3c17dd1` |
| Blank Metal landing page | GL-only IntroScreen owned the entire launcher; Metal skipped it | `913c891` |
| SIGSEGV on first `ImGui::Image` (Metal) | bridge handed out integer texture ids; ImGui-Metal retained `1` as an object pointer | `4267753` |
| Docked desktop layout | floating windows → ImGui **docking branch** pin + host dockspace + first-run tiling (D17) | `7310a52` |

### ⬜ Not started

| Phase | Milestone | Nature |
|---|---|---|
| **3** | **Independence** | reach — the "platform moment" |
| **4** | **Distribution** | reach |
| **5** | **Ecosystem** | reach |
| **6** | **Demand-driven expansion** | reach / conditional |

---

## 3. What "done" looks like today (inventory)

**Services vended (8):** `ui.v1`, `log.v1`, `jobs.v1`, `device.v1`, `metrics.v1`, `tensor_bridge.v1`, `artifacts.v1`, `data.v1` (+ `CaliperTensor` and the Arrow C Data Interface as ABI-boundary interchange types). Every service now has an honest in-tree consumer.

**Applets (8):**
- *Flagship:* GPTScope (`applets/gpt_scope`)
- *All-services exemplar:* EmbedScope (`applets/embed_scope`) — 3-D embedding projector; consumes all 8 services
- *Exemplars:* SignalScope (general idioms), MLScope (MNIST CNN + kernel/digit/feature-map viz)
- *Fixture:* Hello (`examples/hello`)
- *Legacy (epoch-2, bridge-native, never service-migrated):* CircuitNet, OpenGllama, RepNet Demo

**Tests:** ~80 cases across three ctest suites — `caliper_tests` (host units + sugar + ABI/C-gate), `caliper_gfx_tests` (windowed, pixel-exact bridge + ImGui-draw-path both backends, label `gfx`), `caliper_torch_tests` (adapter, label `torch`).

**Docs:** MkDocs wiki (`docs/wiki/`), strict-build gated, reference pages embedding real headers (incl. the two new service pages + an Arrow C Data Interface note).

**Infra:** GitHub org `caliper-platform` (repo transferred); host self-versions `0.6.0`.

---

## 4. What's left to reach the full scope

Everything below the sufficiency line. Phase 2F′ (the last two demand-driven services) is now **done** — it was the final piece of *completeness for the author*, and it closed on real demand exactly as D12/D16 prescribed. Everything remaining below it exists to onboard **other people**, and none of it is required for the platform to be complete for you.

### Phase 2F′ — demand-driven services *(done — the last two services shipped with their first honest consumer, D18)*
- [x] `caliper.artifacts.v1` — content-addressed, deduped, run-lineaged checkpoint store (blob files + DuckDB index); **consumer:** EmbedScope Save/Load (load-bearing — Load restores the cloud without training).
- [x] `caliper.data.v1` — named datasets + SQL over the host store, results out as Arrow C streams; **consumer:** EmbedScope's live class-centroid and misclassified-count queries over the published embedding table.
- [x] `EmbedScope` — the all-services exemplar; 3-D embedding bottleneck on live ImPlot3D, renderer-agnostic, exercising the whole platform surface.
- *Extract-don't-invent honored: each header was frozen only once its consumer existed.*

### Phase 3 — Independence *("this is the moment Caliper becomes a platform")*
- [ ] Split `caliper-sdk` into its own repo via `git filter-repo` (history preserved), tag `v0.1.0`.
- [ ] Move the pinned UI stack (imgui docking + implot + implot3d + ImGuiFileDialog) into the SDK repo.
- [ ] `caliper-applet-template` repo: 10-line CMake, manifest, fixture-host tests, CI matrix.
- [ ] Move **CircuitNet out first** (smallest, torch-free) into its own repo with migrated history, built by CI that never checks out Caliper.
- [ ] **Golden-applet wall** in host CI: keep `.caliperapp` artifacts built against every supported SDK release; every host change must still load them.
- [ ] Decide SDK license (D10 — MIT/Apache-2.0), due this phase.
- [ ] **Exit:** a `circuitnet.caliperapp` built by foreign CI, dropped into a stock host, runs.
- *Carry-in from earlier reviews:* add the deferred service-table/nested-`api` `struct_size` forward-compat checks and the `AppletMeta.services` `static_assert` when epoch-3 planning begins; `main.cpp` card-loop dedupe on next touch.

### Phase 4 — Distribution
- [ ] `.caliperapp` bundle format + manifest pre-flight before `dlopen`.
- [ ] Runtime packs (libtorch downloaded once, shared, checksum-verified) + `caliper::torch_stub`.
- [ ] `caliper new` (scaffold) + `caliper dev` (file-watch hot reload) as host subcommands.
- [ ] **Vulkan backend + CUDA interop** on Windows — *where "CUDA-ready-by-construction" becomes CUDA-verified* (needs hardware/CI); GLEW→GLAD or delete the GL fallback.
- [ ] Move repnet-lab/opengllama-style applets to their own repos with pack dependencies.
- [ ] **Exit:** fresh machine → 50 MB host → install a bundle → pack fetched once → live training.

### Phase 5 — Ecosystem
- [ ] `caliper-registry` git repo (`index.json` + `packs.json`); publishing = a PR.
- [ ] In-app Browse tab (install/update/uninstall, compatibility filtering) + sideloading.
- [ ] Docs go public: GitHub Pages + `mike` versioning + `mkdocs-cxxdox` generated API reference (D15).
- [ ] macOS codesign/notarize + update check; delete `applets/` from the monorepo.
- [ ] **Exit (the real one):** someone who isn't you ships an applet end-to-end without a question the docs can't answer.

### Phase 6 — Demand-driven expansion *(conditional)*
- [ ] Out-of-process applet host for untrusted binaries (the real sandbox).
- [ ] Python bindings (pybind over the sugar; notebook-driven prototyping via the fixture host).
- [ ] `libcaliper` extraction + **Compass** as the second host (wxWidgets chrome; validates the host-neutral service rule).
- [ ] Linux as a first-class triple; bundle signing; hosted registry if PR volume demands.

---

## 5. Open decisions & standing items

- **The strategic question:** do Phases 3–6 happen at all? They add zero capability for a solo user and carry real maintenance cost. The architecture was deliberately built so stopping here is a *complete* outcome, not an abandoned one.
- **Push:** `main` is ~50 commits ahead of `origin`; nothing has been pushed since early Phase 2A.
- **Hands-on demo pass:** the visual/interaction checks that only human eyes close — GPTScope's sample evolution + attention grid, MLScope's live kernels/feature-maps, the docked layout, crash-quarantine and refusal cards.
- **Two subagent anomalies** were observed and recovered (instant zero-tool failures); logged in the ledger. A third would trigger a halt-and-investigate rather than another retry.

---

## 6. One-line summary

**Phase 2 is complete and the platform is sufficient for its author: all 8 services shipped and each honestly consumed, the USP proven, a flagship you chose plus an all-services exemplar (EmbedScope) running end-to-end on public services in a docked desktop. Everything remaining is about inviting other people in — and that's now a choice, not a task.**
