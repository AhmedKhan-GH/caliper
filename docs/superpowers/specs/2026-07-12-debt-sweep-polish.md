# Debt sweep — the final polish pass (four fixes, one registry)

**Date:** 2026-07-12
**Status:** execution spec, small. Scope: the fix-now tier of the accumulated
review-debt ledger (sources: the v1_3 final review, the T5 fleet review, the
feed T3 review, and the C1/C2 Compass reviews — each item carries its origin).
One session, one branch (`chore/debt-sweep`), one Opus implementer + review.
**Rule:** anchors below were re-verified against `main` @ `1174771` on
2026-07-12 — one ledger item was found STALE (§1.1) precisely because the
tree moved under it; the implementer trusts THESE anchors, not the old
review texts.

---

## 1. The four fixes

### 1.1 TwinScope leftovers — LEDGER CORRECTION + one deletion

- **STALE, do NOT remove:** `MODE_SPLIT` was flagged dead by the T5 fleet
  review — but the fleet was reverted (`85698f6`) and the restored TwinScope
  uses the split view as its DEFAULT (`twin_scope.cpp:204`
  `display_mode{MODE_SPLIT}`, `:221 published_mode = MODE_SPLIT`). The enum
  value is load-bearing again. The ledger entry is retired as overtaken by
  events; no code change.
- **REAL, remove:** `model_offset` (`twin_scope.cpp:238` declaration, `:581`
  assignment) is write-only — grep shows no read anywhere in the file, in
  the restored pre-fleet version too. Delete the member and the assignment.
  Verify with a fresh grep in the diff, and confirm the split view still
  renders (it positions halves through other state): the run-proof is the
  existing autolaunch line `geometry view drawn — 2 mesh half(s)`.

### 1.2 `inst_attr_base` u32 guard (the one real gate gap)

Origin: v1_3 T3 review. The instance-MATRIX base has a 32-bit guard
(`tensor_bridge.cpp:796-797`: `instance_offset / 4u > UINT32_MAX` →
`"instance base exceeds 32 bits"`) because it rides a `PrimParams` u32; the
instance-ATTR base rides the same kind of u32 (`inst_attr_base =
instance_attr_offset / 4`) but has NO equivalent gate — only G10's bounds
check keeps it sane (unreachable below 16 GB attr offsets, hence
"theoretical"; gates don't get to have known holes regardless).

- **Fix (host validator, platform-neutral):** in the `GeomRev::V1_3`
  attr-gate block, after G9/G11/G10, add:
  `if (d13->instance_attr_offset / 4u > UINT32_MAX) reject_i("instance attr
  base exceeds 32 bits")` — string in the exact G6 register.
- **Test:** one case in the G-battery of `test_tensor_bridge.cpp` (mirror
  the existing G6 case: purpose-built large alloc so bounds pass and the
  base check is the one that fires; MSVC-safe locals).
- **Backend re-gates: deferred symmetrically.** The host refuses before any
  backend sees the draw; today NEITHER backend re-gates this (symmetric).
  Adding the Metal twin here would force a 3-line Vulkan transcription into
  pending-Windows state for defense-in-depth — not worth it. Each backend
  picks it up on its next natural pass; note it in the gate-table comment.

### 1.3 `pulse_scope` needless atomic

Origin: feed T3 review. `pulse_scope.cpp:99` `std::atomic<float> window_s`
is only touched from `draw_ui` (`:261-264` — load and store, same thread).
Make it a plain `float` with a one-line comment (draw-thread-only). No
behavior change; the applet's tests and run-proof stand.

### 1.4 The `--clean-first` GLOB footnote (docs only)

Origin: feed T3 report (pre-existing behavior, rediscovery risk). Root
`CMakeLists.txt:407` globs applets with `CONFIGURE_DEPENDS`; a partial
`cmake --build --clean-first` on a target subset can prune sibling applet
dylibs from `build/applets/` until a full build restores them. Add a short
warning box to `docs/wiki/tutorials/development-basics.md` (the build page)
saying exactly that. mkdocs --strict stays green.

## 2. The deferred registry (recorded here so the ledger has a durable home)

NOT in this sweep, each waiting for its natural session — with owners:

| Item | Origin | Lands with |
|---|---|---|
| Torch-free/lazy-torch libcaliper link variant | C1 review | first external embedder that needs a small binary |
| Foreign-root live metrics reading (DuckDB checkpoint barrier) | C0b/C2 | next metrics-adjacent caliper work or out-of-process design |
| Wall-clock `started` column in the runs schema | C2 review | same metrics session as above |
| Compass: foreign-root picker coherence; `.compass` layout authority; virtual/diffing table model | C2 review | next Compass session |
| Metal G1–G12 re-gate not test-isolated (host refuses first) | v1_3 T3 review | accepted layered-design property; revisit only if the host battery ever thins |
| Windows feed provider | feed spec §6.2 | `2026-07-12-feed-v1-windows-provider-pass.md` (queued, executable) |
| Multi-canvas per core; embed v2 wishlist (api_version getter, canvas-info query) | L2 reviews | first Compass document wanting two viewports |

**Owner-manual item (30 seconds, needs a human at the GUI):** float→re-dock
the Compass Viewport pane once; crisp re-render at correct scale closes the
last UNVERIFIED-MANUAL ledger line (C1-⚠️2). If it blurs: file it against
`RepushViewport`.

## 3. Gate

Full ctest green; gfx 49/49 untouched (1.2's gate is host-level — the
byte-exact rows don't move); TwinScope + pulse_scope autolaunch run-proofs
re-captured (the §1.1 split-view line, the §1.3 unchanged dashboard); mkdocs
--strict green; one review pass; merge to main + push. Commits:
chore(debt)/fix(geometry)/docs(wiki) as fits, Fable trailer.
