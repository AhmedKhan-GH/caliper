# EmbedScope "training freezes" postmortem — the database was the bug

| | |
|---|---|
| **Date** | 2026-07-05 |
| **Platform observed** | Windows 11, RTX 500 Ada Laptop GPU (4 GB), NVMe/NTFS |
| **Symptom** | Whole-window freeze while training: rhythmic hitches early in a session, degrading to a constant ~2 fps (window unresponsive, even OS resize stuttered). macOS: same code, smooth. |
| **Root cause** | data.v1 SQL table rebuild (`CREATE OR REPLACE TABLE` + 2,000-row `INSERT` into the **on-disk** DuckDB store) executed **on the frame thread**, with per-rebuild cost that **grows over the session** (WAL append + catalog churn), until it exceeded its own 0.5 s throttle and ran every frame. |
| **Fixed in** | `3c465e7`, `b14633a`, `6535820`, plus the SQL-worker/`clear finished` commits landing with this doc |
| **Audience** | Future contributors on either platform; the macOS-side agent (see §6) |

---

## 1. TL;DR

EmbedScope rebuilt a DuckDB table on the UI thread twice a second. On macOS/APFS
that costs a few ms and nobody noticed. On Windows/NTFS the same rebuild started
around ~80 ms and **grew with session age** to ~500 ms, at which point the 0.5 s
throttle degenerated into "every frame" and the app ran at 2 fps. The durable
lessons: **no I/O on the frame thread, ever** (even "fast" I/O is only fast on
one platform, on a fresh store), and **live-visualization tables belong in
memory (`TEMP`), not in the persistent store** — an on-disk table that is
rebuilt from scratch every 0.5 s is pure WAL/catalog churn that nothing ever
reads back.

Three other real defects were found and fixed on the way (texture re-creation
per step, 10 forced stream drains per eval, jobs-panel litter), but the freeze
itself was the database.

## 2. Why Windows and not the Mac — the actual asymmetry

The applet code is identical on both platforms. The asymmetry is in what sits
under `data.v1`:

- The data store lives at `%APPDATA%\Caliper\data.duckdb` (Windows) /
  `~/Library/Application Support/Caliper/data.duckdb` (macOS) — **on disk**.
- Every rebuild ran `CREATE OR REPLACE TABLE embed_points(...)` (a DDL /
  catalog transaction) plus one 2,000-row `INSERT` (~170 KB SQL literal).
  Each is a WAL-append transaction against the on-disk file.
- **The cost compounds over a session.** Each rebuild appends the full table
  contents to the WAL; each `CREATE OR REPLACE` leaves dropped-table state
  behind until checkpoint. Measured: an identical rebuild takes **~78 ms**
  against a fresh copy of the store, but the in-app rebuild measured
  **417–566 ms** after a day of sessions. The WAL had grown to 7.5 MB.
- NTFS commit/fsync and Windows Defender make each transaction meaningfully
  more expensive than on APFS, so macOS never surfaced the growth as a freeze.
- The killer interaction: the rebuild was throttled to every 0.5 s **measured
  from its start**. Once a rebuild itself takes ≥ 500 ms, the throttle is
  always elapsed — the rebuild runs **every frame**, on the frame thread.
  Frame period ≈ rebuild duration ≈ 2 fps, event pump starved, whole window
  (including OS resize) frozen.

Ruled out along the way, with measurements: the M2a stream handoff (benchmarked
0.27 ms/update, byte-exact), Vulkan/CUDA interop per se, WDDM present stalls,
eval compute cadence (the "every 4%" perception was a coincidental cadence —
the 0.5 s SQL throttle ≈ the 0.54 s eval window).

## 3. What changed — database aspects first

### 3.1 The SQL rebuild (the freeze) — commit `6535820` + the SQL-worker commit

- **The table is now `CREATE OR REPLACE TEMP TABLE`.** DuckDB temp tables are
  in-memory and per-connection: no WAL append, no on-disk catalog churn, no
  session-age growth. Rebuild cost stays at its floor permanently. The panels
  (class centroids via `AVG(x,y,z) GROUP BY label`, misclassified/total counts)
  query the temp table identically — same SQL, same results.
- **Persistence semantics changed deliberately:** `embed_points` no longer
  exists in `data.duckdb` across restarts. Nothing ever read it back (it was
  `CREATE OR REPLACE`'d from live data every 0.5 s; the only consumer is
  embed_scope itself — verified repo-wide). A one-time
  `DROP TABLE IF EXISTS main.embed_points` reclaims the legacy on-disk table.
- **The rebuild runs on an applet-owned worker thread**, not the frame thread
  and not a jobs.v1 job (it is internal plumbing, not user-meaningful work —
  it must not occupy the Jobs panel). The frame thread stages input copies
  under the applet mutex and notifies a condition variable; the worker
  rebuilds and commits results (centroids/counts/status) under the same mutex;
  draw sites read mutex-guarded copies. Staging overwrites, so a slow rebuild
  coalesces to the newest snapshot. `cleanup()` flags exit, notifies, joins.
- data.v1's single connection is only ever touched from that worker thread —
  the store stays single-user, and the TEMP table (per-connection state)
  lives exactly as long as the host's data store connection.

### 3.2 Texture churn (frame-thread interop cost) — commit `b14633a`

The Tensors panel re-created all 9 weight-heatmap textures every optimizer step
because the colormap range chased `max|weight|`. On Vulkan+CUDA a mapped-texture
**create** is the full interop construction (exportable `VkImage` + Win32 handle
+ `cuImportExternalMemory` + per-texture timeline semaphore create/export/
import): measured **1.4 ms create vs 0.27 ms update** (gfx timing case,
`3c465e7`, kept as a regression probe). 9 × 1.4 ms ≈ 12.6 ms of frame budget per
step. Now the worker normalizes display tensors to a fixed [-1, 1] on the
training device and the frame thread creates each texture once and
`update_texture`s in place. Metal never punished the old pattern (cheap
`newTexture`), but the reuse pattern is strictly better there too.

### 3.3 Eval stream drains — commit `b14633a`

`evaluate()` called `.item()` per 1,000-image test batch — 10 forced stream
drains per eval; on WDDM each is a scheduler round-trip. Correct-counts now
accumulate in a device tensor with **one** readback. Same batches, same math,
identical accuracy, structure and cadence unchanged (monolithic every 50 steps,
exactly as on the Mac — an earlier experiment that sliced the eval across steps
was reverted on owner request).

### 3.4 Jobs panel hygiene — this commit set

The jobs list only ever grew (minimal §7.5 surface: list + cancel). Added
`JobSystem::clear_finished()` (joins finished threads before erasing) and a
`clear finished` button in the Jobs window. Job entries are in-memory UI state
only — clearing loses nothing.

## 4. Measured outcome (Windows, RTX 500 Ada)

| Metric | Before | After |
|---|---|---|
| `draw_ui` frames > 100 ms during training | 43+/run, 417–566 ms each | **0** |
| Frame rate while training | ~2 fps | **86–124 fps** |
| Worst frame gap over a full run | ~900 ms | ~42 ms |
| SQL rebuild cost trajectory | ~80 ms → ~500 ms over a session | pinned at floor (in-memory) |
| All test suites | — | green (unit 33,295 / gfx 883 / torch 1,027 assertions) |

## 5. Rules to carry forward

1. **The frame thread does zero I/O.** No SQL, no file writes, no store calls.
   The host's own watchdog motto ("long work belongs in background jobs")
   applies to anything > ~5 ms — and platform variance means you should assume
   your fast path is someone else's slow path.
2. **Live-viz tables are `TEMP`.** The on-disk data store is for deliberate,
   read-back persistence. High-frequency rebuild targets in the persistent
   store are self-inflicted WAL growth — cost rises monotonically with session
   age and the regression is invisible on a fresh dev store.
3. **jobs.v1 is for user-meaningful work** (training, downloads — things a user
   watches or cancels). Internal maintenance loops get applet-owned threads.
4. **Perceived rhythm is data, but verify the cadence match with probes** —
   two independent 0.5 s-ish cadences coexisted here and the obvious one was
   innocent. The probe ladder that worked: inter-frame gap around
   `renderer_->render()` in the host loop → loop-phase breakdown
   (poll / new_frame / applet / render) → draw_ui-phase breakdown
   (snapshot / textures / plots / sql). Applet-side CPU timers alone were
   misleading (they once showed a healthy 120 fps while the user saw freezes).
5. `CALIPER_AUTOLAUNCH=<manifest id>` skips the landing page — halves the
   clicking in verification runs.

## 6. Notes for the macOS agent

Everything in §3 is **shared applet/host code** — the Mac picks all of it up on
pull. Worth a verification pass on Apple Silicon:

- **Behavioral parity check:** one full EmbedScope training run. Expect: cloud
  + heatmaps + panels identical; centroids/misclassified refresh ~2 Hz; a
  single training job in the Jobs panel (no `SQL panels` entries — that worker
  is intentionally invisible); `clear finished` button appears after the run.
- **TEMP-table path on the Mac store:** the Data panel should read
  "data.v1: table rebuilt, SQL panels live" during training. First run also
  executes the one-time `DROP TABLE IF EXISTS main.embed_points` against the
  Mac's `data.duckdb`.
- **Single-sync eval on MPS:** `evaluate()` now accumulates correct-counts in a
  device tensor with one `.item()` — verify no MPS regression (it removes 9
  syncs there too; expected neutral-to-positive).
- **Texture reuse on Metal:** the create-once/`update_texture` path replaces
  release+create per step; M2b's per-texture `MTLSharedEvent` ordering now sees
  updates only. The gfx suite + the `3c465e7` timing case (Vulkan-only case
  self-skips on Metal) should stay green.
- **Point-motion note:** there is deliberately **no smoothing** on the cloud
  (raw coordinates from the last completed optimizer step, both platforms).
  Windows now advances ~90+ gens/s vs the Mac's ~30, so Windows looks more
  fluid; the Mac's "more violent" look is the same per-step deltas sampled 3×
  less often. Any change here is a design decision, not a bug fix.
- **Known same-class hazard, unfixed:** gpt_scope's head heatmap does
  release+recreate per probe (its `vmax` changes per probe — the same
  stored-range problem embed_scope had). Fix shape: normalize on the producer,
  fixed range, update in place. Lower priority (probe cadence, not per-step).

---

*Diagnosis history and crash-level detail live in the commit messages of
`3c465e7`, `b14633a`, `6535820`, and the SQL-worker commit. The M2a/stream
handoff verification that preceded (and was initially, wrongly, suspected) is
`docs/m2a-windows-verification.md`.*
