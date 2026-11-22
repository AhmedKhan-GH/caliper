# caliper.feed.v1

Service id `caliper.feed.v1`: live telemetry from the machine the host runs on.
The **host** owns the telemetry providers, buffers their samples in per-channel
ring buffers with timestamps and sequence numbers, and vends a small
**pull-based, any-thread, non-blocking** read surface. Applets copy samples into
their own tensors on their job threads, so data still flows from tensors to
pixels. Embedders reach the same surface through `caliper_core_get_service`. One
host service saves every applet from writing its own ingestion, and reading
with cursors that the caller holds needs no callbacks across the ABI and no
state in the host for each reader. This
page embeds the header verbatim; the docs build fails if the file moves.

```c
--8<-- "sdk/include/caliper/services/feed_v1.h"
```

## Semantics

A **channel** is one named f32 time series (`id` like `"sys.cpu.util"`, a
display name, units, a nominal rate hint). A **sample** is
`{ seq, t_ns, value, reserved0 }` — `seq` is per-channel monotonic from 1;
`t_ns` is the host steady clock (nanoseconds from the platform's monotonic clock), the **same epoch
across channels** so a reader can align series on one timeline.

- **`caps()`** — bit 0 (`CALIPER_FEED_CAP_LIVE`) is set **iff** at least one
  channel is registered on this host. No provider (Linux) or no channels →
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

### Honest-Loss Contract

Each channel keeps a **fixed-capacity ring** (4096 samples ≈ 6+ minutes at
10 Hz). Overflow drops the oldest. A reader that falls behind sees its `seq`
**jump**: if `*cursor` points below the oldest still-buffered sample, the copy
resumes at the oldest available one, so the returned seqs skip past
`*cursor + 1`. That jump is exactly the count of samples lost — **visible, never
silent.** A dashboard surfaces it as a gap counter; a stale channel (no new
samples) is visible through its timestamps, never interpolated.

### Thread Callability

Every entry point is **any-thread and non-blocking** — one mutex per channel
(writer = the provider thread; readers = anyone, any cadence). Applet job
threads and embedder UI threads read the same surface concurrently and safely.
No callbacks cross the ABI: no reentrancy or threading contract to defend, and
no per-subscriber host state. The sample struct and the service table each carry a `reserved0` slot, so a
push surface could be added without breaking this one. The service has no
applet-registered or external sources: every channel comes from a host provider.

## Providers

The macOS (Apple Silicon) provider and the Windows provider each register their
channels at host start. Linux has no provider, so `caps()` is `0` and every read
yields nothing. Every provider is **privilege-honest**: each signal is read
without elevated rights, and a channel that would need root, entitlements or a
kernel driver does not exist rather than half-existing. Each channel is sampled
at a nominal 10 Hz into a ring of 4096 samples.

macOS channels:

| Channel id | Name | Units | Source |
|---|---|---|---|
| `sys.cpu.util` | CPU Utilization | `%` | `host_processor_info` deltas |
| `sys.mem.used` | Memory Used | `%` | `host_statistics64`; used memory, not kernel pressure, so the name does not overclaim |
| `sys.thermal.state` | Thermal State | none | `NSProcessInfo` code 0 to 3 for nominal, fair, serious, critical; the ABI carries a bare float and the reader labels it |
| `sys.gpu.util` | GPU Utilization | `%` | IOAccelerator performance statistics |
| `sys.fan.rpm` | Fan Speed | `rpm` | userland AppleSMC |
| `sys.temp.battery` | Battery Temperature | `degC` | userland AppleSMC |
| `sys.power.battery` | Battery Power | `W` | IOPMPowerSource |

The thermal channel always registers. Every other channel registers only when
its source reads successfully on the machine.

Windows channels:

| Channel id | Name | Units | Source |
|---|---|---|---|
| `sys.cpu.util` | CPU Utilization | `%` | `GetSystemTimes` deltas |
| `sys.mem.used` | Memory Used | `%` | `GlobalMemoryStatusEx` `dwMemoryLoad` |
| `sys.gpu.util` | GPU Utilization | `%` | NVML, loaded at runtime from the driver's `nvml.dll`; failed ticks are skipped and the loss is visible |
| `sys.gpu.temp` | GPU Temperature | `degC` | NVML |
| `sys.gpu.power` | GPU Power | `W` | NVML, gated against the device-reported enforced power limit |
| `sys.power.battery` | Battery Power | `W` | `CallNtPowerInformation`; negative means discharging, as on macOS; absent on desktops |

Windows exposes no unprivileged path to CPU temperature or fan speed, so those
channels do not exist there. The NVML channels exist only when an NVIDIA driver
is present; the probe fails closed at start and no SDK is linked.

Whatever is unavailable is simply not a channel, never a faked value. A consumer
displays whatever enumerates, and on a host with no channels it says so in one
line and draws nothing. PulseScope does exactly this (see
[Applets](../in-tree-applets.md)).
