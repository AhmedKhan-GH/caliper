# Phase 2F′ — data.v1 + artifacts.v1 + EmbedScope (3D embedding projector) Plan

> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:subagent-driven-development. Checkbox steps.

**Goal:** Finish Phase 2 by (a) building the two demand-driven services `caliper.artifacts.v1` (content-addressed checkpoints) and `caliper.data.v1` (SQL/Arrow tabular access), and (b) shipping **EmbedScope** — a data-driven ML exemplar that puts **3D visualization front and center** (ImPlot3D, renderer-agnostic) and exercises **every** service. EmbedScope is the first honest consumer that un-parks both services (D16 demand-driven clause satisfied).

**Architecture.** Both stores reuse the host's already-linked DuckDB (`caliper_host_lib`, from 2B). `artifacts.v1`: blobs written as files under `<data>/artifacts/<sha256>`, a DuckDB index table linking name→digest→run (dedup + lineage). `data.v1`: DuckDB is the query engine; results cross the ABI as **Arrow C Data Interface** streams (D3 — the ratified tabular boundary), defined in a vendored `caliper/arrow_c.h` (the canonical single-header spec, C-clean). No DuckDB or Arrow C++ type crosses the frozen ABI. EmbedScope's 3D is **ImPlot3D** (already in `caliper::ui_stack`, proven in repnet) — OpenGL-rasterized on the GL backend, Metal on the default; §6c-clean, portable.

**EmbedScope design.** Small net on MNIST with a **3-D embedding bottleneck** (coordinates are literally learned — no projection): `conv→conv→fc→Linear(*,3)→ReLU→Linear(3,10)`. The star panel is an **ImPlot3D scatter of test-set embeddings, 10 colors by class, rotating, updating live during training** — you watch one blob split into ten lobes. Services used: `jobs`(train), `device`(MPS), `metrics`(loss/acc→dashboard), `tensor_bridge`(hover a 3D point → that digit as a GPU texture), `ui`/`log`, **`artifacts`**(Save/Load model — reload skips training; load-bearing), **`data`**(register the embedding table, live SQL: class centroids as a 3D mesh, misclassified gallery, kNN-to-cursor — Arrow round-trip on real learned data). MNIST reuse: the cached file from MLScope's data dir if present, else the same download recipe.

## Global Constraints
- All prior-phase constraints carry over (trailer `Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>`, explicit-path staging, no agent merges — orchestrator merges, build/, TDD for host units, strict mkdocs on docs changes, aliveness `kill -0` for headless run checks — NOT log-grep).
- **Branch:** `platform/phase-2f` from `main`.
- **Ids (fixed):** `"caliper.artifacts.v1"`, `"caliper.data.v1"`; EmbedScope id `dev.caliper.embed-scope`, name `EmbedScope`, version `0.1.0`, tag `ML`.
- **Frozen headers are immutable on merge** — freeze exactly to §7.7/§7.8 shapes with sibling C-hygiene (`stdint.h`+`stdbool.h`, extern-C, struct_size first, no third-party types). Arrow structs enter only via the vendored `caliper/arrow_c.h`.
- **§16 contracts:** artifacts — `put` then `exists`/`path_of` round-trips a digest; identical bytes dedup to one file; wrong digest → not-found (not crash). data — a known table queried back yields exact rows through the Arrow stream; `last_error` set on bad SQL, no crash.
- Both services thread-safe (job threads will call them): one mutex per store over its DuckDB connection (the MetricsStore pattern).
- **Do not touch:** frozen headers already shipped, `src/main.cpp` beyond service-init wiring, other applets/examples, `third_party/`, `cmake-build-debug/`.

## Tasks

### F1 — `artifacts.v1` frozen header
Create `sdk/include/caliper/services/artifacts_v1.h` (§7.8 shape: `put(name,bytes,len,run,out_digest[65])→bool`, `path_of(digest_or_name)→const char*`, `exists(...)→bool`). Extend `tests/test_abi.cpp` (standard-layout, struct_size@0, id string) + `tests/abi_c_check.c` (newest-first include). Doc stub `docs/wiki/reference/services/artifacts-v1.md` (H1+intro+`--8<--` embed+`*Semantics: F6*`) + mkdocs nav. TDD RED→GREEN full ctest; strict mkdocs.
Commit: `feat(sdk): artifacts.v1 service header (Phase 2F)`.

### F2 — ArtifactStore + vend + sugar (TDD)
`src/host/artifact_store.{h,cpp}` (DuckDB-free header, pimpl): blobs → `<root>/artifacts/<digest>` files; index table `artifacts(digest TEXT PK, name TEXT, run BIGINT, len BIGINT, ts)`; sha256 over bytes (vendor a small public-domain sha256 in `src/host/` or use an available lib — state which). `put` dedups (existing digest → no rewrite), links run; `path_of`/`exists` accept digest OR name (name → newest). `tests/test_artifact_store.cpp`: round-trip, dedup-to-one-file, name-resolves-newest, unknown inert, run-lineage query. Vend in `host_services.{h,cpp}` (open at `services_init`, non-fatal on failure like MetricsStore; `host_artifact_store()` accessor; image-gate-style validation not needed). `caliper::Artifacts` sugar in `caliper.hpp` (falsy-inert; `put(name,span,run)→std::string digest`, `path_of`, `exists`). Extend `tests/test_sugar_services.cpp`. kIds→7.
Commit: `feat(host): ArtifactStore — content-addressed checkpoints; vend artifacts.v1 + sugar`.

### F3 — Arrow C header + `data.v1` frozen header
Vendor `sdk/include/caliper/arrow_c.h` = the canonical Arrow C Data Interface (`ArrowSchema`/`ArrowArray`/`ArrowArrayStream` with release callbacks — the public single-header spec, verbatim, C-clean). Create `sdk/include/caliper/services/data_v1.h` (§7.7 shape: `query(sql,ArrowArrayStream*)→bool`, `register_dataset(name,uri)→bool`, `open_dataset(name,ArrowArrayStream*)→bool`, `last_error()→const char*`; includes arrow_c.h). ABI tests + C-gate + doc stub + nav as F1. TDD; strict mkdocs.
Commit: `feat(sdk): arrow_c.h + data.v1 service header (Phase 2F)`.

### F4 — DataStore + vend + sugar (TDD)
`src/host/data_store.{h,cpp}` (pimpl): DuckDB connection; `register_dataset(name,uri)` (parquet/csv/table); `open_dataset`/`query` execute and export results as an `ArrowArrayStream` (DuckDB's Arrow export — the caller drains+releases); `last_error`. `tests/test_data_store.cpp`: create an in-memory table, query it back, drain the Arrow stream, assert exact values; bad SQL → false + last_error non-empty, no crash; register+open a temp parquet round-trip. Vend in host_services (open at init, non-fatal; `host_data_store()`). `caliper::Data` sugar in `caliper.hpp`: a thin Arrow-stream→typed-row adapter (drains a stream into `std::vector` of a caller struct via column accessors — keep minimal: expose the raw stream + a helper for the numeric-columns case EmbedScope needs). Extend sugar tests. kIds→8.
Commit: `feat(host): DataStore — SQL/Arrow tabular access; vend data.v1 + sugar`.

### F5 — EmbedScope (the 3D exemplar)
`applets/embed_scope/{embed_model.h, embed_scope.cpp, plugin.cpp, embed_scope.caliper.toml, CMakeLists.txt}`. Required services ui/log/jobs/device; optional metrics/tensor_bridge/artifacts/data (all probed). Model with 3-D bottleneck; training job (jobs.v1, cancel-per-step, loss/acc→metrics); worker publishes under a mutex, each eval tick, the test-set 3-D embeddings + labels + preds (owned clones). Frame: **ImPlot3D scatter** (BeginPlot, PlotScatter per class colour, live) as the centerpiece; class centroids as an ImPlot3D mesh/markers computed via **data.v1 SQL** over the registered embedding table; hover a point → **tensor_bridge** shows that digit; loss/acc ImPlot; **Save/Load model** buttons → **artifacts.v1** (Load reconstructs and skips training — the load-bearing demand); a small SQL-results panel (misclassified count / gallery via data.v1). Inherit ml_scope contracts (download recipe, bounded-wait cleanup, curl pairing, generation/mutex texture lifecycle). Root `applets/*` glob auto-builds it; manifest POST_BUILD copy; torch+curl link like ml_scope; id/version byte-match.
Verification: full build; ctest + gfx + torch green; both renderers `kill -0` alive 10s; card count = 8. Live 3D/train/save-load = F6 human checklist.
Commit: `feat(embed_scope): 3D embedding projector — ImPlot3D live clusters, artifacts + data services, all-feature exemplar`.

### F6 — Docs + demo checklist + STATUS + merge
Semantics on `artifacts-v1.md`/`data-v1.md` + a `reference/arrow.md` note; `index.md` + `tutorials/first-applet.md` mention EmbedScope; extend the tensor-bridge demo checklist with EmbedScope's 3D items; update `docs/STATUS.md` (2F′ ✅, 8 services, 8 applets). Human demo checklist: (1) launch EmbedScope → train → the 3D cloud splits blob→10 lobes live, rotating; (2) hover a point → its digit texture; (3) centroids/misclassified via SQL update live; (4) Save model → relaunch → Load → skips training, cloud restored; (5) run in Runs dashboard; (6) both renderers. Strict mkdocs; full suites.
Commit: `feat(embed_scope)+docs: semantics, demo checklist, STATUS — Phase 2 complete`. Then orchestrator merges `platform/phase-2f`.

## Exit Criteria (= true Phase-2 completion)
| Requirement | Proof |
|---|---|
| artifacts.v1 frozen, content-addressed, dedup+lineage | F1/F2 + §16 tests |
| data.v1 frozen, Arrow round-trip, error-safe | F3/F4 + §16 tests |
| Both services their first honest consumer | F5 (artifacts load-bearing; data queries live) |
| 3D front-and-center, renderer-agnostic, data-driven | F5 ImPlot3D live embeddings + human checklist |
| All 8 services exercised by one applet | F5 |
| Suites green both backends | every task |

## Risks / Notes
- **Arrow is the phase's heaviest plumbing.** arrow_c.h is a verbatim stable spec (no invention); DuckDB's Arrow export is documented. If DuckDB's Arrow API in the vendored version differs, adapt the store, never the frozen header.
- **data.v1's EmbedScope use is demonstrative-but-honest** (SQL over a genuinely tabular embedding set: centroids, kNN, misclassified) while **artifacts.v1's is load-bearing** (no reload without it). Neither is contrived.
- 3-D bottleneck may cluster imperfectly on MNIST in 3 dims — that's honest and still visually clear (the blob→lobes transition is the point, not linear separability). Checklist language reflects this.
- ImPlot3D scatter of 10k test points per frame is fine; if heavy, subsample to ~2k for the live view (state it).
- sha256: prefer a tiny vendored public-domain impl over adding a dependency.
