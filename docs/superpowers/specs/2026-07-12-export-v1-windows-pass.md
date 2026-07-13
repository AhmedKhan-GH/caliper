# export.v1 on Windows — the verify pass

**Date:** 2026-07-12
**Status:** execution spec for the next Windows-box session. NEARLY pure
verification: `caliper.export.v1` shipped platform-neutral by composition
(merge `1796a0f` — service, battery, exemplar affordances, build-time git
stamp), and every piece it composes is ALREADY hardware-proven on the box
(`geom_create_view_ex` / `draw_primitives` / `debug_readback_rgba8` ran there
in the v1_3 and embed passes). What has never run on Windows is the export
veneer itself — and two of its load-bearing claims are genuinely
platform-sensitive (§2). Protocol: D24, same as every prior box pass.
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

- [ ] **1.1** Configure+build green via the box wrappers; the
  `caliper_git_stamp` step runs under vcvars (`cmake -P` + git on PATH —
  the wrappers already provide git; verify the generated header appears and
  carries the true HEAD, and that a dirty tree stamps `-dirty`).
- [ ] **1.2** `caliper_export_tests` builds; if the live cases were
  Apple-gated, land the Vulkan gating edit (test-side only; MSVC REQUIRE
  discipline per the standing gotcha).

## 2. The two platform-sensitive claims (the real point of this pass)

- [ ] **2.1 Atomic replace-existing rename on NTFS.** The filesystem-purity
  design writes `<name>.caliper_tmp` then `std::filesystem::rename` over the
  target. POSIX rename atomically replaces; on Windows, `fs::rename` maps to
  `MoveFileExW(MOVEFILE_REPLACE_EXISTING)` in modern MSVC STLs — VERIFY on
  the box, not from docs: export twice to the same path (file exists,
  replaced, contents = second export); export over a file held open by
  another handle (expect refusal rc 0 with the original intact — record the
  actual behavior; if fs::rename throws/fails non-atomically anywhere, that
  is a STOP-and-diagnose, fix in export_service with the Win32 call, not by
  loosening the contract).
- [ ] **2.2 Truncation/refusal purity on NTFS.** The battery's sentinel case
  (pre-existing file byte-identical after a refused export) must RUN there,
  not skip. Plus the E1-review LOW path: sidecar-write failure rolls the PNG
  back (if the case is Mac-gated, port it).

## 3. Byte-exactness + determinism on Vulkan

- [ ] **3.1** The decoded-quad row: exported PNG pixels == the CPU reference
  on Vulkan (top-down row order — the asymmetric corner checks must pass;
  a flip here is a backend readback bug, not a test to adjust).
- [ ] **3.2** Double-export memcmp byte-identity on Vulkan (same draws →
  identical PNG bytes).
- [ ] **3.3** Full `caliper_export_tests` green on the box, 0 skipped where
  hardware is present; full ctest suite green; gfx unregressed.

## 4. The exemplar run-proof (live, artifacts)

- [ ] **4.1** `CALIPER_EXPORT_SELFTEST=1` autolaunch of twin_scope on
  Vulkan+CUDA: the 3840×2160 figure + sidecar land (sidecar says
  `backend=vulkan platform=windows`, git_commit = the box's HEAD); the
  300-frame sequence completes with a finalized `sequence.json`.
- [ ] **4.2** mesh_scope figure likewise (draw_count=3, MAGMA colormap id in
  the sidecar).
- [ ] **4.3** ffmpeg assembly: if ffmpeg exists on the box, assemble and
  ffprobe (h264, 300 frames); else record the exact command +
  UNVERIFIED-TOOLING per house rule.
- [ ] **4.4** The interrupted-record orphan-guard: kill the app mid-record;
  `sequence.json` must still be finalized with the honest partial
  frame_count (the Mac reviewer verified this accidentally; do it on
  purpose here).

## 5. Closeout (only with artifacts)

- [ ] **5.1** wiki `export-v1.md`: the Windows row flips from
  "compiles/battery pending" to run-proven with this pass's evidence
  (fs::rename semantics finding recorded either way).
- [ ] **5.2** PUBLISHING.md §2 Rung E row + §3 status line: drop the Windows
  qualifier → "run-proven both ecosystems"; name the commits.
- [ ] **5.3** The execution spec + this spec: status headers updated,
  commits named; box scratch ledger records the NTFS findings.
- [ ] **5.4** Commits in house style, Fable trailer; any code fix rides the
  full battery + a live re-proof.

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
