# Compass as a libcaliper CONSUMER — beyond picture-in-picture

**Date:** 2026-07-12
**Status:** DESIGN — downstream of `2026-07-11-libcaliper-compass-design.md`
(R4). That doc built the embeddable core and gated L3/Compass on a named
workflow (§5). This doc answers the owner's next question — *"what would a
Compass app that USES Caliper, not just embeds it, look like?"* — by
concretizing the app shape, the one ABI extension it needs, and the candidate
v0 workflows. The §5 gate stands: building Compass still requires the owner to
name ONE workflow from §5 below. Everything here is designed so that naming it
is the only remaining decision.
**Authority:** the R4 design doc (§4.2 canvas ownership — unchanged here),
PLATFORM.md Phase 6 (wx chrome, AUI, property grids, document-style), D5 (one
libtorch per process), D13 (native backends), D3 (torch/DuckDB types never
cross the ABI; `CaliperTensor` + Arrow C streams are the interchange).
**Verified baseline:** `include/caliper/embed.h` v1 exposes create/attach_
canvas/frame/event/load_applet/unload/read_pixels/last_error/shutdown — and
NOTHING else. Today an embedder can only run applets in a pane. This doc's ABI
section (§3) is what turns "embeds" into "consumes."

---

## 1. One paragraph

Compass is a native wxWidgets **document application** (projects, tabs,
property grids, AUI docking) whose documents are *made of Caliper's data
plane*: it opens the same metrics/artifacts/data stores the instrument
writes, renders live viewports through the same libcaliper canvas, and steers
running applets through the same service registry — without owning a renderer,
a loader, a torch runtime, or one line of the applet contract. Caliper is the
fast-thinking face (watch it happen); Compass is the slow-thinking face
(study it, arrange it, author it, report it). The integration is three planes
(§2), of which embedding is only the first.

## 2. The three integration planes ("embed" vs "consume")

| Plane | What it is | ABI surface | Status |
|---|---|---|---|
| **P1 View** | A live applet canvas docked as one pane among native panes | embed.h v1 as shipped | DONE (L2) |
| **P2 Data** | Compass's OWN native UI reads the platform's stores: metrics (DuckDB via `metrics.v1`), artifacts, `data.v1` Arrow streams — no applet involved | §3's `get_service` | MISSING — the v1.1 extension |
| **P3 Steering** | Compass's property grids/config push INTO a running applet; Compass-side code feeds tensors the viewport draws | same `get_service` (+ each service's existing semantics) | MISSING — rides P2 |

The line the R4 doc drew stays drawn: **the canvas is libcaliper's** (§4.2 —
chrome never paints applet pixels). What P2/P3 add is that the *data plane*
becomes shared: a Compass "document" is a native window onto the same stores
and services, with the canvas as its live viewport rather than a foreign app
in a box.

## 3. The one ABI extension: embed v1.1 (`get_service` for hosts)

### 3.1 The call

```c
/* v1.1 — additive (struct_size discipline unchanged). Returns the same
 * service table an applet gets from CaliperHost.get_service, or NULL if the
 * id is unknown / the core is not created. The pointer is valid until
 * caliper_core_shutdown. */
const void* caliper_core_get_service(CaliperCore* core, const char* id);
```

One call, the applets' own vocabulary (`"caliper.metrics.v1"`, …). No new
types cross the seam: services already speak C ABI + `CaliperTensor` + Arrow
C streams (D3), so the host consumes them exactly as an applet would, with
`caliper.hpp`'s existing sugar available to C++ hosts.

### 3.2 The threading contract (pinned here, verified before v1.1 ships)

- **Frame-thread services** — `ui.v1` (meaningless to a host anyway),
  `tensor_bridge`/`geometry` (draw-adjacent): host calls only from the thread
  that calls `caliper_core_frame`.
- **Any-thread services** — `jobs.v1` is inherently cross-thread; `metrics.v1`
  reads (DuckDB) and `artifacts.v1`/`data.v1` are *believed* thread-safe for
  reads but this has only ever been exercised from applet job threads — the
  v1.1 execution pass MUST pin each service's actual thread rules in embed.h
  and test host-thread reads racing an applet's writes (the Compass case:
  a wx UI thread querying metrics while a live run streams into it).
- **No torch in Compass.** D5 binds the embedder: Compass never links torch.
  P3 tensor-feeding uses bridge CPU-staged uploads or pool tensors produced
  by an APPLET's worker — Compass pushes *parameters*, applets produce
  *tensors*. (A Compass that needs its own device tensors is a different,
  later conversation — likely an applet in disguise.)

### 3.3 v0 gaps that graduate to requirements at v1.1

- `data_dir` must stop being ignored (a document app needs per-project data
  roots — this is the field's designed purpose).
- Applet `log.v1` lines must route to the embedder's `log_fn` (a document app
  surfaces logs in a native pane, not stderr) — the ledgered
  de-singletonization work.
- Multi-canvas (one core, N canvases) becomes real demand the moment a
  document wants two viewports; v1.1 may still ship one-canvas — but the
  refusal must stay honest and the ABI shape must not preclude N.

## 4. The app shape (wx, document-style)

- **Chrome:** wxWidgets native — menu bar, AUI docking, wxPropertyGrid,
  wxDataViewCtrl tables, document/view with project files (`.compass` — a
  small manifest naming stores, applet ids, layouts; no tensor data inside).
- **Panes (the standard set):**
  - *Viewport* — the libcaliper canvas (P1), one per document v0.
  - *Inspector* — wxPropertyGrid bound to applet/document config (P3).
  - *Tables* — wxDataViewCtrl over `metrics.v1` queries / `data.v1` streams
    (P2), native sort/filter/select.
  - *Runs/Artifacts browser* — `artifacts.v1` + the metrics run index (P2).
  - *Log* — the embedder `log_fn` sink (§3.3).
- **Event loop:** wx owns it; `caliper_core_frame` from a wxTimer/idle at
  display rate ONLY while a viewport is visible (a document app spends most
  of its life not pumping — the core must tolerate long frame gaps; applets
  already must per the honest-degradation rules).
- **Process:** separate repo (`compass`), consumes libcaliper as the SDK
  artifact + the embed header — the Phase-3 independence discipline applied
  to hosts. It never includes `src/host/*` (the include-root PRIVATE flip,
  already ledgered, becomes a hard prerequisite here).

## 5. Candidate v0 workflows (the §5 gate — owner picks ONE)

| Candidate | Planes | New machinery | Why it might be first |
|---|---|---|---|
| **(a) Run-comparison documents** — open N runs from the metrics store side by side; native tables/plots; annotate; export a report | P2 only (P1 optional garnish) | §3.1 PLUS a metrics READ surface across the C ABI (see the survey correction below) | Cheapest true "consumer"; exercises P2 alone; useful the day it exists; zero applet changes |

**Survey correction (2026-07-12, C1 groundwork — supersedes the original "(a)
needs nothing beyond §3.1" claim):** `metrics.v1` is WRITE-ONLY at the ABI
(`begin_run/scalar/histogram/image/hparams_json`; header immutable); the
readers (`runs()/scalars()`) are host-private C++ a consumer cannot include,
and `data.v1.query()` targets a separate `data.duckdb`, not the applet's
`metrics.duckdb`. So workflow (a) additionally requires ONE of: (i) the data
service's SQL→Arrow read surface gaining read-only visibility into the
metrics store (DuckDB ATTACH at the host layer — no new service, one read
vocabulary, D3-aligned; preferred if the single-writer/attach semantics prove
out), or (ii) an additive metrics query revision (`metrics.v1_1`: list runs /
stream scalars). This is C0b in the phasing; C1's live-table gate depends on
it.

**C0b RESOLUTION (2026-07-12, SHIPPED — option (ii) won):** Option (i) was
probed same-process and REJECTED on evidence. A read-only `ATTACH
'metrics.duckdb' (READ_ONLY)` from a second DuckDB instance sees only rows the
writer has CHECKPOINTed to the file — the live writer's freshly-committed rows
were invisible across ALL variants (same attach, DETACH+re-attach, a brand-new
instance) until an explicit writer `CHECKPOINT`. A live metrics writer never
checkpoints per-scalar, so option (i) cannot satisfy "the table shows live rows
from the running applet" (C1's acceptance). (The probe also found DuckDB does
NOT file-lock a second same-process handle here — irrelevant to the verdict,
but noted so a future OUT-OF-PROCESS Compass knows the checkpoint-visibility
barrier — not the lock — is the real blocker, and bites cross-process too.)
Shipped: **`caliper.metrics.v1_1`** — the frozen v1 writer prefix plus one
`query(sql, out_arrow_stream)` read entry, running SQL against the metrics
store's OWN live connection (the C0-proven one-connection-one-mutex path, so a
host UI thread reads rows the instant a worker writes them), reusing
`data.v1`'s SQL→Arrow producer verbatim. READ-ONLY is ENFORCED here (unlike
`data.v1`, which does not): the SQL is parsed, not executed, and refused unless
it is exactly one `SELECT` — the connection is the live writer. Reached by both
applets and embedders via `caliper_core_get_service`. Commit: `ba1d85b`.
| **(b) Twin authoring** — property-grid documents that configure a twin (assets, cameras, colormap ranges, per-unit metadata), viewport shows it live, "run in Caliper" hands off | P1+P2+P3 | A config channel INTO applets (P3 semantics per applet; likely a small applet-side convention, zero ABI) | The Adobe-shaped vision; but needs (a)'s plumbing anyway plus P3 conventions |
| **(c) Dataset/embedding curation** — tables over `data.v1` Arrow streams, filter/label/flag, write back | P2 (+P3 for write-back) | data.v1 write-back semantics (today it's read-oriented) | Real ML utility; blocked on a data.v1 revision — heaviest |

**Recommendation (not a decision):** (a) first. It is P2-pure, needs nothing
but §3.1, produces a genuinely useful artifact (the run-comparison report),
and every pane it builds (tables, runs browser, log) is reused verbatim by
(b) and (c). (b) is the flagship shape but should be second, riding proven
plumbing. The §5 gate is satisfied the moment the owner says "build (a)" (or
names something better).

## 6. What Compass does NOT do (inherited walls, restated)

- No second torch, no renderer of its own, no applet-contract changes, no wx
  `HostRenderer` backend, no painting applet pixels with wx (R4 §4.2).
- No pack manager / registry in v0 (sideload; Phases 4–5 own that).
- Not a replacement for Caliper: anything realtime-first (watch training,
  steer a sim) stays in the instrument. If a Compass document wants a live
  loop, it embeds a viewport — it does not grow one.

## 7. Phasing (each ships alone; C0 is useful even if Compass never happens)

- **C0 — embed v1.1 in the caliper repo.** §3.1 `get_service` + the §3.2
  thread-rule pinning + the §3.3 gap closures (data_dir, log routing).
  Acceptance: a headless test host queries `metrics.v1` rows written by a
  live applet, from a non-frame thread, races clean under TSan-style
  scrutiny; embed battery grows accordingly; applets unaffected (suites
  green). *This is worth shipping regardless — it makes embed_host and any
  future host a real consumer.*
- **C1 — Compass skeleton (separate repo).** wx app, one document, Viewport
  pane (P1) + Log pane + ONE native Table pane over a metrics query (P2).
  Acceptance: same `.caliperapp` binary runs in Caliper and in the Compass
  pane; the table shows live rows from the running applet; chrome is native;
  process links no torch.
- **C2 — the named workflow document** (per the §5 pick). For (a):
  multi-run open, side-by-side tables + native plots, annotations in the
  `.compass` project, report export. Acceptance: the owner performs the
  workflow end-to-end without touching Caliper's shell.
- **C3 — the Windows pass.** Same protocol as every hardware pass; requires
  the libcaliper embed Vulkan/Windows pass
  (`2026-07-11-libcaliper-embed-vulkan-windows-pass.md`) landed first.

## Invariants (hold forever)

- The canvas is libcaliper's; the chrome is the host's; the data plane is
  shared through `get_service` — three planes, one seam, nothing leaks into
  the applet contract.
- Compass documents contain references and configuration, never tensor data.
- One torch per process, and it is never Compass's.
- Honest degradation travels: a service the core can't vend → the pane says
  so; a canvas the backend refuses → the viewport says so; never a blank
  pane pretending.
