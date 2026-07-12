# `caliper.feed.v1` — telemetry ingestion (physical twins, rung one)

**Date:** 2026-07-12
**Status:** EXECUTED — §3 (ABI), §4 (host + macOS provider), §5 (`pulse_scope`
exemplar) all shipped on `feat/feed-v1` (T1 `477f9e5`; T2 `16dd30c` + `b9c1c71`;
T3 this exemplar+closeout series). The macOS provider is live-verified — 7
sudo-free channels @10 Hz, run-proven under CPU load — and `pulse_scope` freezes
the read surface. Follow-ups §6 unchanged (Windows provider, thermal-twin
flagship, applet-registered sources, Compass pane). Originally DESIGN, approved
by the owner (strategy call 2026-07-12: telemetry becomes a platform service, not
a per-applet pattern — closing ROADMAP §7's last open decision). Exemplar
decision: the v0 physical source is THIS
machine's own silicon (option (a)); the exemplar applet is a dashboard
(`pulse_scope`), and the machine-thermal-twin flagship is a NAMED FOLLOW-UP,
not part of this spec (extract-don't-invent: the service freezes against the
dashboard, which exercises every ABI surface; the twin rides later without
holding the ABI hostage).
**Authority:** ROADMAP §7 (the telemetry decision), D3 (only C types cross the
ABI), the honest-degradation register (GEOMETRY.md §1.3 house-wide), the
service-catalog conventions (host_services.cpp, additive revisions,
struct_size discipline). The whitepaper's local-loop claim gains a
restatement when the flagship lands: *local from ingestion onward* — sensor
transport latency is upstream of the loop, never hidden.

---

## 1. One paragraph

Physical twins need live measurements from the world; today every drawn
tensor is computed in-process. `feed.v1` is the pipe: the HOST owns telemetry
providers (v0: the Mac's own sensors), buffers their samples in per-channel
ring buffers with timestamps and sequence numbers, and vends a tiny
pull-based C read surface. Applets copy samples into their own tensors on
their job threads (tensors → pixels stays intact downstream); embedders reach
the same surface through `caliper_core_get_service` (Compass shows machine
telemetry in native panes for free). Sources are enumerable, reads are
any-thread and non-blocking, and data loss is visible (sequence gaps), never
silent.

## 2. The data model

- **Channel** — one named f32 time series. `id` (stable string, e.g.
  `"sys.cpu.util"`), display name, units (`"%"`, `"°C"`, `"W"`), a nominal
  rate hint (Hz). Channels are host-defined in v0 (no applet-registered
  sources — that is a future revision with its own demand).
- **Sample** — `{ uint64 seq; int64 t_ns; float value; float reserved0; }`.
  `seq` is per-channel monotonic from 1; `t_ns` is the host steady clock
  (mach_absolute-derived ns — the same epoch across channels so applets can
  align series). 24 bytes, 8-aligned, static_asserted.
- **Ring buffer** — fixed capacity per channel (4096 samples ≈ 6+ minutes at
  10 Hz). Overflow drops oldest; a reader that fell behind sees `seq` jump —
  the honest-loss contract. No blocking, no backpressure in v0.

## 3. The ABI (four entry points, all any-thread, non-blocking)

`sdk/include/caliper/services/feed_v1.h`, id `"caliper.feed.v1"`:

```c
typedef struct CaliperFeedSample {  /* 24 B, static_asserted */
    uint64_t seq; int64_t t_ns; float value; float reserved0;
} CaliperFeedSample;

typedef struct CaliperFeedChannelInfo {
    uint32_t struct_size;            /* caller sets; host fills the rest */
    char     id[64]; char name[64]; char units[16];
    float    nominal_hz;             /* 0 = irregular */
} CaliperFeedChannelInfo;

typedef struct CaliperFeedV1 {
    uint32_t struct_size;
    uint32_t (*caps)(void);                      /* bit 0: live */
    uint32_t (*channel_count)(void);
    /* 1 on success, 0 on bad index/size (info untouched). */
    uint32_t (*channel_info)(uint32_t index, CaliperFeedChannelInfo* info);
    /* Copy up to max samples with seq > *cursor into buf (oldest-first),
       advance *cursor to the last copied seq, return the count. cursor==0
       starts at the newest sample minus max (tail read). Unknown id -> 0,
       cursor untouched. Non-blocking; a gap in seq relative to the caller's
       previous cursor means the ring overwrote — data was lost, honestly
       observable. */
    uint32_t (*read)(const char* channel_id, CaliperFeedSample* buf,
                     uint32_t max, uint64_t* cursor);
    void*    reserved0;              /* future: applet-registered sources */
} CaliperFeedV1;
```

Rationale for pull + caller cursors: no callbacks across the ABI (no
reentrancy/threading contract to defend), no per-subscriber host state (any
number of readers, including embedder UI threads, at any cadence), and the
Compass/wx-timer case is safe by construction. Push can be layered later
without breaking this surface.

## 4. Host implementation

- `src/host/feed_store.{h,cpp}` — channel registry + ring buffers + one
  mutex per channel (writer = provider thread; readers = anyone). The same
  one-connection-one-mutex simplicity the metrics store proved race-safe.
- **Provider thread** (host-owned, started at services_init when the platform
  provider has channels, stopped at services_shutdown before the stores
  close): samples every 100 ms (10 Hz) into the store.
- **macOS platform provider** (`feed_provider_mac.*`): SUDO-FREE signals
  only. Guaranteed tier (public APIs): CPU utilization (host_processor_info),
  memory USED %-of-total (host_statistics64 — named `sys.mem.used`, NOT
  "pressure": the kernel's true memory-pressure signal is a different metric
  and names must not overclaim; T2-review value-honesty ratification),
  thermal pressure state
  (ProcessInfo.thermalState as 0-3), GPU utilization (IOAccelerator
  PerformanceStatistics — readable unprivileged). Best-effort tier
  (investigate at implementation; include ONLY what reads without privileges
  on this machine, verified: SMC temperatures/fan via a userland SMC read if
  accessible, adapter/battery power via IOPMPowerSource). Whatever is
  unavailable is simply NOT a channel — never a faked value; the exemplar
  displays whatever enumerates. Sampling failures mid-run: the channel goes
  stale (no samples), visible via timestamps; one log line, house style.
- **Deterministic test provider** (`feed_store` API, no thread): tests inject
  exact samples/sequences — the battery never depends on real sensors or
  timing.
- **Windows provider: SHIPPED** (2026-07-12, its own pass —
  `2026-07-12-feed-v1-windows-provider-pass.md`, `553f7eb`): 6 admin-free
  channels live on the box (cpu/mem/battery + NVML gpu util/temp/power,
  runtime-loaded). The LibreHardwareMonitor-class question resolved honestly:
  CPU temp / fan RPM are kernel-driver-gated on Windows, so those channels
  DO NOT EXIST (probe refusals recorded in the pass report).
- Vend: `kFeed` in host_services.cpp, registered in `kIds` + `service()`;
  reachable by applets AND embedders (embed.h thread rules already say
  services documented any-thread are host-thread-safe — feed.v1 documents
  any-thread).

## 5. The exemplar: `pulse_scope` (dashboard, no ML)

New applet `applets/pulse_scope/`: enumerates channels, one job-thread
poller (cursor per channel, ~10 Hz) into per-channel CPU ring copies,
draw_ui renders ImPlot strip charts — INPUT-LOCKED plots (kLockedPlot house
rule), per-channel current value + units + staleness ("last sample 0.3 s
ago") + a gap counter (seq jumps seen). No feed caps → the honest one-liner
("telemetry unavailable on this host"). No tensors/bridge/geometry needed —
this exemplar is deliberately CPU-plot-only; the flagship twin consumes the
same feed THROUGH tensors later.

Acceptance (all by artifacts): channels enumerate on this Mac (guaranteed
tier at minimum); charts move live under load (run a build to heat the CPU —
the plots must show it); kill/relaunch clean; suite green; the test battery
covers: info/read semantics, cursor tail-start, gap-on-overflow, unknown-id
refusals, MSVC-safe assertions.

## 6. Follow-ups (named, not in this spec)

1. **The machine-thermal-twin flagship**: a small model learns utilization →
   temperature response of this Mac live; measured|predicted split like
   TwinScope; assimilation story; whitepaper local-loop restatement lands
   with it.
2. Windows provider pass — **EXECUTED 2026-07-12** (`553f7eb` + closeout;
   spec `2026-07-12-feed-v1-windows-provider-pass.md`): the caps bit flipped
   WITH artifacts — 6 admin-free channels @ 10 Hz live-verified on the
   RTX 500 Ada box, load-proven (cpu 90%+ under a parallel build, gpu.temp
   45→54 °C under gfx runs), embed path enumerating through `embed_host`.
3. Applet-registered/external sources (serial/UDP/MQTT) — needs real
   hardware demand (`reserved0` holds the slot).
4. Compass telemetry pane (native wx strip charts over the same reads).

## Invariants (hold forever)

- Only C types cross the ABI; timestamps are host-clock ns; loss is visible
  (seq gaps), never silent; absent capability → inert entry points + honest
  status, never fake data.
- The feed is an INPUT rung: it feeds tensors→pixels; it never becomes a
  render or control path.
- Providers are host-owned and privilege-honest: a channel that needs sudo
  does not exist rather than half-existing.
