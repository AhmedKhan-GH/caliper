# feed.v1 Windows provider — the hardware pass

**Date:** 2026-07-12
**Status:** EXECUTED 2026-07-12 on the box (`553f7eb` + closeout; run-proof
§4 all ticked, closeout §5 ticked except the whitepaper sentence riding the
next whitepaper touch). Original scope below, unchanged. GREENFIELD on the
provider (nothing Windows-specific exists), but the hard parts are already
frozen and proven: the feed.v1 ABI, the FeedStore, the lifecycle seam, and the
exemplar all shipped platform-neutral on `main` (`e30ac1e`) — this pass writes
ONE new Apple-provider-sibling TU and proves it on the box. Protocol mirrors
the T2 macOS pass (investigation-first, privilege-honest) and the D24
discipline of every prior Windows pass.
**Authority:** the feed.v1 design spec §4/§6
(`2026-07-12-feed-v1-telemetry-design.md`, status EXECUTED; the Windows
provider is its named §6.2 follow-up); the T2 report + review (macOS pass
precedent: probe evidence per signal, fail-closed registration, the
`sys.mem.used` value-honesty rename); D24 (`docs/m2a-windows-verification.md`).
**Checkbox discipline (inherited):** a box is checked only when the suite is
green on the box, the path is run-proven by a logged artifact, and the commit
is named. A channel that cannot read unprivileged DOES NOT EXIST — never a
faked value, never an "run as admin" requirement.

---

## 0. What already exists (all platform-neutral, no new risk)

- `sdk/include/caliper/services/feed_v1.h` — FROZEN. No ABI change is in
  scope for this pass, none is needed.
- `src/host/feed_store.{h,cpp}` — the ring/cursor/gap machinery, fully
  tested in `caliper_tests` (13 cases), compiles on Windows already (it is in
  `caliper_host_lib`, no Apple guards).
- The lifecycle seam: `host_services.cpp` starts/stops the provider under
  `#ifdef __APPLE__` today — this pass adds the `#elif defined(_WIN32)`
  branch calling the new provider, SAME anchors (start at end of
  `services_init` after `g_feed`; stop+join at top of `services_shutdown`
  before `g_jobs.cancel_all_and_join()` — the ordering comments cite the
  crash history; do not re-derive).
- `pulse_scope` — enumerates channels DYNAMICALLY (reviewed property). It
  will display whatever this pass registers, including a smaller set, with
  the honest ladder covering zero. NO applet changes.
- The tests: the provider battery (`test_feed_provider.cpp`) is
  Apple-guarded; this pass adds the Windows twin cases (same three shapes:
  lifecycle, double-cycle, non-fatal live smoke). MSVC doctest discipline:
  operands in locals, no complex REQUIRE expressions.

## 1. Phase 1 — INVESTIGATE (probe evidence per signal, in the report)

Rules from the T2 precedent: one throwaway probe per API, actual values in
the report, fail-closed registration in production. Expected tiers ON THE
BOX (RTX 500 Ada laptop, Windows 11 — expectations, not claims):

- **Guaranteed tier (documented-public, no admin):**
  - `sys.cpu.util` "%" — `GetSystemTimes` deltas (idle/kernel/user), the
    exact analog of the macOS host_processor_info delta math.
  - `sys.mem.used` "%" — `GlobalMemoryStatusEx` (`dwMemoryLoad` is literally
    this; keep the T2-ratified name — it is used-memory, not pressure).
  - `sys.power.battery` "W" — `CallNtPowerInformation(SystemBatteryState)`:
    `SYSTEM_BATTERY_STATE.Rate` (mW, signed; verify the sign convention
    matches the macOS channel: negative = discharge). On a desktop/no-battery
    box the channel honestly does not exist.
- **Probe tier (likely yes, prove it):**
  - `sys.gpu.util` "%" — two candidates, probe both, ship the better:
    (a) **NVML** (`nvml.dll`, runtime-loaded — the cuda_driver.cpp
    runtime-load house pattern, D11: never link a vendor SDK), 
    `nvmlDeviceGetUtilizationRates`; (b) PDH "GPU Engine" counters (the Task
    Manager source — vendor-neutral but fiddly counter-path enumeration).
    NVML is vendor-locked but this box is NVIDIA and NVML also unlocks:
  - `sys.gpu.temp` "degC" — `nvmlDeviceGetTemperature` reads unprivileged
    with the driver present. If NVML proves out, this is the box's ONLY
    honest temperature — Windows has no unprivileged CPU-temp API.
  - `sys.gpu.power` "W" — `nvmlDeviceGetPowerUsage` (mW), same probe.
- **Expected-ABSENT tier (record the refusal, do not fight it):**
  - CPU temperature / fan RPM — `MSAcpi_ThermalZoneTemperature` WMI is
    admin-gated and stale on most laptops; LibreHardwareMonitor-class tools
    need a kernel driver (admin). Per the privilege-honesty invariant these
    channels DO NOT EXIST on Windows v0. Record the probe refusals as
    findings.
  - `sys.thermal.state` — no ProcessInfo.thermalState analog; absent.

Channel-id rule: reuse the macOS id EXACTLY where the semantics match
(`sys.cpu.util`, `sys.mem.used`, `sys.gpu.util`, `sys.power.battery`) so a
future cross-platform applet sees one vocabulary; new NVML-only ids
(`sys.gpu.temp`, `sys.gpu.power`) are honest additions. Expected realistic
outcome: 4–6 channels.

## 2. Phase 2 — implement

- `src/host/feed_provider_win.{h,cpp}` — WIN32-only TU into the same targets
  as the mac provider (mirror its CMake wiring). Same shape as
  `feed_provider_mac.mm`: probe-at-start registration (fail-closed per
  signal), one 10 Hz thread (the T2 cadence; `t_ns` stamped at sample time
  from QueryPerformanceCounter — the steady clock already used elsewhere),
  `kRingCapacity = 4096` at the `add_channel` call site (T1-review carry),
  `warn_once` + skip on mid-run sampler failure, never a fake value.
- NVML: runtime `LoadLibrary("nvml.dll")` + `GetProcAddress` (house
  pattern: `cuda_driver.cpp`); absent/failing NVML → the NVML channels
  simply don't register. `nvmlInit_v2`/`nvmlShutdown` tied to provider
  start/stop.
- Lifecycle: the `#elif defined(_WIN32)` branch at BOTH existing anchors;
  idempotent start, safe double-stop, re-startable across embed
  create/shutdown/create cycles (the embed battery runs on Windows and
  cycles services — it must stay green).

## 3. Tests

Windows twins of the three provider cases (lifecycle, double-cycle,
non-fatal live smoke asserting ≥1 sample per guaranteed-tier channel within
a bounded window, skip-with-message if none). MSVC discipline throughout.
The platform-neutral batteries (feed store 13, pulse_ring 8) must already
pass untouched — they are the canary that no shared code moved.

## 4. Run-proof (the pass's acceptance, all by artifacts)

- [x] **4.1** Full suite green on the box (all ctest suites incl. embed) —
  9/9 at `553f7eb` (embed double-cycle exercises the provider's
  start/stop/start on Windows).
- [x] **4.2** Exe log line landed verbatim: `[feed] Windows provider: 6
  channels live (cpu=1 mem=1 gpu=1 gputemp=1 gpupower=1 power=1) @ 10 Hz`;
  `pulse_scope` via autolaunch: all 6 fresh, 0 gaps (pass report).
- [x] **4.3** THE LOAD PROOF: DuckDB static-lib parallel rebuild pinned
  `sys.cpu.util` at 90–93% across ~26 consecutive summaries (idle 1–5%);
  NVML landed and `sys.gpu.temp` climbed 45→54 °C (util 13→48%) during
  back-to-back gfx-test runs. Artifacts: `caliper-probes/load_capture{,2}.log`
  + pass report.
- [x] **4.4** `embed_host.exe dev.caliper.pulse-scope` (Win32/HWND, Vulkan):
  `pulse-scope: on_init — 6 channel(s) enumerated (caps 0x1)`, summaries
  fresh/0-gaps through the embed path (`CALIPER_EMBED_APPLETS` env).
- [x] **4.5** caps honesty: bit 0 lights on this box (0x1 both paths); the
  zero-channel test re-gated to hosts-without-a-provider (non-Apple,
  non-Windows) — degradation contract exact on Linux/other.

## 5. Closeout (only with artifacts)

- [x] **5.1** wiki `feed-v1.md` platform table: Windows row flipped to
  live/verified with the 6-channel table scoped as THIS box's set (NVML
  channels exist only with the NVIDIA driver; battery absent on desktops),
  same honesty wording as the macOS row.
- [x] **5.2** feed spec §4 DEFERRED bullet + §6.2 follow-up ticked with
  commits (`553f7eb` + this closeout); ROADMAP §7 line gained the Windows
  clause.
- [ ] **5.3** Whitepaper: the "Windows provider pending" sentence rides the
  next whitepaper touch (markdown + tex + rebuilt PDF together — the pass
  is not held on LaTeX, per this spec).
- [x] **5.4** Commits in house style, Fable trailer (`553f7eb` + this
  closeout); box scratch ledger records the probe evidence
  (`.superpowers/sdd/feed-v1-windows-provider-pass-report.md`: NVML rc-999
  RTD3 rate, the 364 W junk read, enforced-limit values, battery Rate sign
  carry-forward).

## Invariants (hold forever)

- A channel that needs admin/kernel-driver access DOES NOT EXIST — the
  provider never asks for elevation and never fakes a value.
- Same ids for same semantics across platforms; new ids only for genuinely
  new signals; names never overclaim (the `mem.used` precedent).
- The frozen feed ABI does not change in this pass. `pulse_scope` does not
  change in this pass. Loss stays visible; absence stays honest.
