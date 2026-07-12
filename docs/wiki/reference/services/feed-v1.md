# caliper.feed.v1

Service id `caliper.feed.v1` — live telemetry ingestion, the first rung of the
physical-twin ladder (feed spec §3). The **host** owns telemetry providers,
buffers their samples in per-channel ring buffers with timestamps + sequence
numbers, and vends this tiny **pull-based, any-thread, non-blocking** read
surface. Applets copy samples into their own tensors on their job threads
(tensors → pixels stays downstream); embedders reach the same surface through
`caliper_core_get_service` (Compass shows machine telemetry in native panes for
free). This page embeds the header verbatim; the docs build fails if the file
moves.

```c
--8<-- "sdk/include/caliper/services/feed_v1.h"
```

## Semantics

A **channel** is one named f32 time series (`id` like `"sys.cpu.util"`, a
display name, units, a nominal rate hint). A **sample** is
`{ seq, t_ns, value, reserved0 }` — `seq` is per-channel monotonic from 1;
`t_ns` is the host steady clock (mach_absolute-derived ns), the **same epoch
across channels** so a reader can align series on one timeline.

- **`caps()`** — bit 0 (`CALIPER_FEED_CAP_LIVE`) is set **iff** at least one
  channel is registered on this host. No provider (Linux v0) or no channels →
  `0`, and every read yields nothing. Honest degradation, never fake data.
- **`channel_count()` / `channel_info(index, info)`** — enumerate the channels
  this host actually vends. **Enumerate dynamically**; never assume a fixed set
  — other machines vend fewer, and the honest ladder covers zero. The caller
  sets `info->struct_size`; a bad index or too-small size returns `0` and leaves
  `*info` untouched.
- **`read(channel_id, buf, max, cursor)`** — copy up to `max` samples with
  `seq > *cursor` into `buf` (oldest-first), advance `*cursor` to the last
  copied seq, return the count. `cursor == 0` is a **tail read** (start at the
  newest sample minus `max`), so a fresh reader gets at most `max` newest
  samples. Unknown id / null buf / `max == 0` / caught-up → `0`, cursor
  untouched.

### The honest-loss contract

Each channel keeps a **fixed-capacity ring** (4096 samples ≈ 6+ minutes at
10 Hz). Overflow drops the oldest. A reader that falls behind sees its `seq`
**jump**: if `*cursor` points below the oldest still-buffered sample, the copy
resumes at the oldest available one, so the returned seqs skip past
`*cursor + 1`. That jump is exactly the count of samples lost — **visible, never
silent.** A dashboard surfaces it as a gap counter; a stale channel (no new
samples) is visible through its timestamps, never interpolated.

### Thread-callability

Every entry point is **any-thread and non-blocking** — one mutex per channel
(writer = the provider thread; readers = anyone, any cadence). Applet job
threads and embedder UI threads read the same surface concurrently and safely.
No callbacks cross the ABI: no reentrancy or threading contract to defend, and
no per-subscriber host state. Push can layer on later without breaking this
surface (`reserved0` holds the slot).

## Platform status

| Platform | Provider | caps | Channels |
|---|---|---|---|
| **macOS (Apple Silicon)** | **live, verified** | `LIVE` | 7 (below) |
| **Windows** | **live, verified** (feed spec §6.2) | `LIVE` | 6 (below) |
| Linux | none yet | `0` (inert) | none |

The macOS provider is **privilege-honest**: every signal is sudo-free; a channel
that would need root/entitlements does not exist rather than half-existing.
Verified live on this box (M-series, macOS 26), guaranteed + best-effort tiers:

| Channel id | Name | Units | Notes |
|---|---|---|---|
| `sys.cpu.util` | CPU Utilization | `%` | `host_processor_info` deltas |
| `sys.mem.used` | Memory Used | `%` | `host_statistics64` (**used**, not kernel pressure — the name must not overclaim) |
| `sys.thermal.state` | Thermal State | *(unitless)* | `NSProcessInfo` code **0..3** = nominal / fair / serious / critical — a **reader labels the number**, the ABI carries a bare float |
| `sys.gpu.util` | GPU Utilization | `%` | IOAccelerator PerformanceStatistics |
| `sys.fan.rpm` | Fan Speed | `rpm` | userland AppleSMC |
| `sys.temp.battery` | Battery Temperature | `degC` | userland AppleSMC |
| `sys.power.battery` | Battery Power | `W` | IOPMPowerSource |

The Windows provider is privilege-honest the same way: every signal reads
**without elevation**; a channel that would need admin or a kernel driver does
not exist rather than half-existing (so CPU temperature and fan RPM are absent
— Windows has no unprivileged path to them). Verified live on this box
(RTX 500 Ada laptop, Windows 11), guaranteed + probe tiers:

| Channel id | Name | Units | Notes |
|---|---|---|---|
| `sys.cpu.util` | CPU Utilization | `%` | `GetSystemTimes` deltas |
| `sys.mem.used` | Memory Used | `%` | `GlobalMemoryStatusEx` `dwMemoryLoad` (**used**, the same non-overclaiming name) |
| `sys.gpu.util` | GPU Utilization | `%` | NVML, runtime-loaded from the driver's `nvml.dll` — RTD3-parked ticks fail and are skipped, loss visible |
| `sys.gpu.temp` | GPU Temperature | `degC` | NVML — the box's only privilege-free temperature |
| `sys.gpu.power` | GPU Power | `W` | NVML, gated against the device-reported enforced power limit (wake-transition junk reads) |
| `sys.power.battery` | Battery Power | `W` | `CallNtPowerInformation`, negative = discharge (same convention as macOS); absent on desktop boxes |

The NVML channels are this box's set: no NVIDIA driver → no GPU channels
(fail-closed probe at start, never a linked SDK).

Whatever is unavailable is simply **not a channel** — never a faked value; a
consumer displays whatever enumerates. On Linux in v0 every entry point is
inert and consumers degrade honestly (one line, nothing drawn).

## The exemplar

**PulseScope** (`applets/pulse_scope/`, tag *Telemetry*) is the
dashboard that freezes this service: it enumerates the channels dynamically,
polls each on a jobs worker at ~10 Hz into an applet-side ring copy, and draws
one **input-locked** ImPlot strip chart per channel with the current value +
units, a climbing "last sample N.N s ago" staleness label, and a cumulative
"lost M samples" gap counter — the honest-loss contract, made visible. It is
CPU + ImPlot only (no tensors), and it is **host-neutral**: it runs identically
under the `caliper` exe and under an out-of-tree embedder. The machine-thermal
**twin** — a model that learns this Mac's utilization → temperature response and
consumes the same feed *through* tensors — is the named follow-up (feed spec §6).
