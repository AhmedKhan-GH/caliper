# export.v1 on Windows — the verify pass

**Date:** 2026-07-12
**Status:** **EXECUTED on the box** (2026-07-13, RTX 500 Ada laptop,
Windows 11, driver 596.47) — `cf196f0` (Vulkan battery twins + §2 NTFS
cases) + the closeout docs commit. Every checkbox below is ticked by a
run artifact; findings in the box ledger
(`.superpowers/sdd/export-v1-windows-pass-report.md`). NEARLY pure
verification as designed: `caliper.export.v1` shipped platform-neutral by
composition (merge `1796a0f` — service, battery, exemplar affordances,
build-time git stamp), and every piece it composes was ALREADY
hardware-proven on the box (`geom_create_view_ex` / `draw_primitives` /
`debug_readback_rgba8` ran there in the v1_3 and embed passes). The export
veneer now has, too — both platform-sensitive claims (§2) held on NTFS
exactly as contracted; zero production-code changes were needed.
Protocol: D24, same as every prior box pass.
**Authority:** PUBLISHING.md §3 (Rung E, SHIPPED row — this pass removes its
"Windows battery pending" qualifier); the execution spec
`2026-07-12-export-v1-execution.md` (EXECUTED header); the E1/E2/final review
verdicts in the Mac ledger — their Windows must-not-forget lists are §2–§4
here.
**Checkbox discipline (inherited):** a box is checked only when the suite is
green on the box, the path is run-proven by a logged artifact, and the commit
is named. A determinism or purity miss is STOP-and-diagnose, never a loosened
comparison.

---

## 0. What already exists (all merged on `main`)

- `sdk/include/caliper/services/export_v1.h` — frozen; frame-thread contract
  in the header. No ABI work in this pass.
- `src/host/host_services.cpp` export block + `src/host/export_service.{h,cpp}`
  (PNG/sidecar/atomic-write helpers) — platform-neutral C++17/20 +
  `std::filesystem`; compiles on Windows today.
- `third_party/stb/` (write v1.16 / image v2.30, pinned) — header-only.
- `cmake/git_commit_stamp.cmake` + the `caliper_git_stamp` build-time target.
- `tests/test_export.cpp` (`caliper_export_tests`, 12 cases) — the LIVE cases
  are gated on a real renderer; **check how the gate is expressed** (E1 built
  it Metal-gated per house pattern): if the gating is Apple-only, the pass's
  ONE expected test-side edit is extending it to the Vulkan backend the same
  way the gfx harness gates its Windows rows. Mirror, don't redesign.
- `applets/{mesh_scope,twin_scope}` export/record affordances +
  `CALIPER_EXPORT_SELFTEST` (env-gated, the headless run-proof hook).

## 1. Build gate

- [x] **1.1** Configure+build green via the box wrappers; the
  `caliper_git_stamp` step runs under vcvars (`cmake -P` + git on PATH —
  the wrappers already provide git; verify the generated header appears and
  carries the true HEAD, and that a dirty tree stamps `-dirty`).
  *(Header matched `rev-parse --short=12`; `-dirty` verified to fire on a
  dirtied tree and clear on restore.)*
- [x] **1.2** `caliper_export_tests` builds; if the live cases were
  Apple-gated, land the Vulkan gating edit (test-side only; MSVC REQUIRE
  discipline per the standing gotcha). *(They were `CALIPER_HAVE_METAL`-gated;
  the CMake half already defined `CALIPER_HAVE_VULKAN=1` under WIN32 — the
  test-file half landed in `cf196f0`: VkHostEnv in the gfx-harness pattern,
  geometry staged via gfx_main's VMM shareable-handle path, all 7 live cases
  mirrored.)*

## 2. The two platform-sensitive claims (the real point of this pass)

- [x] **2.1 Atomic replace-existing rename on NTFS.** The filesystem-purity
  design writes `<name>.caliper_tmp` then `std::filesystem::rename` over the
  target. POSIX rename atomically replaces; on Windows, `fs::rename` maps to
  `MoveFileExW(MOVEFILE_REPLACE_EXISTING)` in modern MSVC STLs — VERIFY on
  the box, not from docs: export twice to the same path (file exists,
  replaced, contents = second export); export over a file held open by
  another handle (expect refusal rc 0 with the original intact — record the
  actual behavior; if fs::rename throws/fails non-atomically anywhere, that
  is a STOP-and-diagnose, fix in export_service with the Win32 call, not by
  loosening the contract). *(Both verified in-battery on the box: the new
  `_WIN32` case holds the target open with `share=READ, no DELETE` —
  `export_write_text_atomic` returned false, original byte-identical, temp
  removed; handle closed → replace succeeded with the new contents. The
  error_code path fired cleanly; no throw, no partial write, no STOP.)*
- [x] **2.2 Truncation/refusal purity on NTFS.** The battery's sentinel case
  (pre-existing file byte-identical after a refused export) must RUN there,
  not skip. Plus the E1-review LOW path: sidecar-write failure rolls the PNG
  back (if the case is Mac-gated, port it). *(Sentinel ran inside the
  Vulkan refusal-purity twin — byte-identical after the `pos_alloc==0`
  refusal. The rollback path had NO test on ANY platform — a new
  Vulkan-gated case plants a directory at `<png>.json`: view_png returned 0,
  PNG rolled back, directory untouched. host_services.cpp:534 is now
  proven, not just reviewed.)*

## 3. Byte-exactness + determinism on Vulkan

- [x] **3.1** The decoded-quad row: exported PNG pixels == the CPU reference
  on Vulkan (top-down row order — the asymmetric corner checks must pass;
  a flip here is a backend readback bug, not a test to adjust). *(Passed —
  no flip, no adjustment.)*
- [x] **3.2** Double-export memcmp byte-identity on Vulkan (same draws →
  identical PNG bytes). *(Passed.)*
- [x] **3.3** Full `caliper_export_tests` green on the box, 0 skipped where
  hardware is present; full ctest suite green; gfx unregressed. *(14 cases,
  84 assertions, 0 failed, 0 skipped — the live Vulkan cases RAN; full
  10-suite ctest 100%, gfx label green.)*

## 4. The exemplar run-proof (live, artifacts)

- [x] **4.1** `CALIPER_EXPORT_SELFTEST=1` autolaunch of twin_scope on
  Vulkan+CUDA: the 3840×2160 figure + sidecar land (sidecar says
  `backend=vulkan platform=windows`, git_commit = the box's HEAD); the
  300-frame sequence completes with a finalized `sequence.json`.
  *(Figure 3840×2160 + sidecar `backend=vulkan platform=windows
  git_commit=cf196f018b58` = HEAD; 300/300 frames, `frame_count: 300`
  finalized. Note: PNG encode paces a debug build well under 30 fps —
  the 10 s budget needs ~60 s wall-clock; `CALIPER_EXIT_AFTER=90` used.)*
- [x] **4.2** mesh_scope figure likewise (draw_count=3, MAGMA colormap id in
  the sidecar). *(draw_count=3, colormaps=[1]=`CALIPER_CMAP_MAGMA`; bonus
  finalized 300-frame record.)*
- [x] **4.3** ffmpeg assembly: if ffmpeg exists on the box, assemble and
  ffprobe (h264, 300 frames); else record the exact command +
  UNVERIFIED-TOOLING per house rule. *(No ffmpeg on the box —
  UNVERIFIED-TOOLING. The command for the produced sequence dir:
  `ffmpeg -framerate 30 -i frame_%06d.png -pix_fmt yuv420p out.mp4`.)*
- [x] **4.4** The interrupted-record orphan-guard: kill the app mid-record;
  `sequence.json` must still be finalized with the honest partial
  frame_count (the Mac reviewer verified this accidentally; do it on
  purpose here). *(Done via the real close path (`CALIPER_EXIT_AFTER`
  firing mid-record): `frame_count: 106` = exactly the 106 frames on disk,
  sidecar fully formed.)*

## 5. Closeout (only with artifacts)

- [x] **5.1** wiki `export-v1.md`: the Windows row flips from
  "compiles/battery pending" to run-proven with this pass's evidence
  (fs::rename semantics finding recorded either way). *(Platform-status
  admonition + the POSIX parenthetical both updated.)*
- [x] **5.2** PUBLISHING.md §2 Rung E row + §3 status line: drop the Windows
  qualifier → "run-proven both ecosystems"; name the commits. *(`cf196f0`
  named in both.)*
- [x] **5.3** The execution spec + this spec: status headers updated,
  commits named; box scratch ledger records the NTFS findings.
  *(`.superpowers/sdd/export-v1-windows-pass-report.md`.)*
- [x] **5.4** Commits in house style, Fable trailer; any code fix rides the
  full battery + a live re-proof. *(No code fix was needed — the pass is
  `cf196f0` + the closeout docs commit.)*

## Invariants (hold forever)

- Refusal purity extends to the filesystem ON EVERY PLATFORM — a refused
  export leaves NTFS exactly as it was, same as APFS.
- The sidecar never lies: backend/platform/git_commit reflect the actual
  box, build, and source state (`-dirty` included).
- Determinism is per-backend; cross-backend byte-identity is NOT claimed
  (Lambert ±2 LSB carries over) — do not "fix" a cross-platform diff that
  the contract already scopes.
- No ABI change, no applet change (the selftest hook already ships); this
  pass verifies, it does not redesign.
