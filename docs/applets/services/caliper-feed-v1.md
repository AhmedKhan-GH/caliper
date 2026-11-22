# caliper.feed.v1

Service id `caliper.feed.v1`: live telemetry from the machine the host runs on.
The host owns the telemetry providers and buffers their samples in a ring buffer for each channel. Each sample has a timestamp and a sequence number. The host offers a pull-based read surface that any thread can call without blocking. Applets copy samples into their own tensors on their job threads, so data still flows from tensors to pixels. Embedders reach the same surface through `caliper_core_get_service`.

The caller holds its own cursor. Thus the service needs no callbacks across the ABI and keeps no state in the host for each reader.

## Types

<!-- c-api types sdk/include/caliper/services/feed_v1.h CaliperFeedSample CaliperFeedChannelInfo level=3 -->

## Calls

<!-- c-api calls sdk/include/caliper/services/feed_v1.h -->

## Ids

<!-- c-api ids sdk/include/caliper/services/feed_v1.h -->

## Semantics

A **channel** is one named f32 time series. It has an `id` such as `"sys.cpu.util"`, a display name, units and a nominal rate hint. A **sample** is `{ seq, t_ns, value, reserved0 }`. All channels share the same epoch, so a reader can align series on one timeline. With no provider (Linux) or no channels, `caps()` returns `0` and every read yields nothing. The service never returns fake data. Enumerate the channels dynamically and never assume a fixed set, because other machines vend fewer channels or none.

### Honest-Loss Contract

Each channel keeps a fixed-capacity ring of 4096 samples, and an overflow drops the oldest sample. If a reader falls behind, its `seq` values jump. When `*cursor` points below the oldest buffered sample, the copy resumes at the oldest available sample, so the returned `seq` values skip past `*cursor + 1`. The size of the jump is exactly the number of lost samples, so a loss is visible and never silent. A dashboard can show the jump as a gap counter. A stale channel with no new samples shows in its timestamps, and the service never interpolates.

### Thread Callability

Every entry point is callable from any thread and never blocks. Applet job threads and embedder UI threads can read the same surface at the same time. No callbacks cross the ABI, so the service has no reentrancy or threading contract and no state for each subscriber. The service has no applet-registered or external sources, because every channel comes from a host provider.

## Providers

The macOS (Apple Silicon) provider and the Windows provider each register their channels at host start. Linux has no provider, so `caps()` is `0` and every read yields nothing. Every provider is **privilege-honest**: it reads each signal without elevated rights. A channel that needs root, entitlements or a kernel driver does not exist. Each channel is sampled at a nominal 10 Hz.

macOS channels:

| Channel id | Name | Units | Source |
|---|---|---|---|
| `sys.cpu.util` | CPU Utilization | `%` | `host_processor_info` deltas |
| `sys.mem.used` | Memory Used | `%` | `host_statistics64`. This is used memory, not kernel pressure. |
| `sys.thermal.state` | Thermal State | none | `NSProcessInfo` code 0 to 3 for nominal, fair, serious, critical. The ABI carries a bare float and the reader labels it. |
| `sys.gpu.util` | GPU Utilization | `%` | IOAccelerator performance statistics |
| `sys.fan.rpm` | Fan Speed | `rpm` | userland AppleSMC |
| `sys.temp.battery` | Battery Temperature | `degC` | userland AppleSMC |
| `sys.power.battery` | Battery Power | `W` | IOPMPowerSource |

The thermal channel always registers. Every other channel registers only when its source reads successfully on the machine.

Windows channels:

| Channel id | Name | Units | Source |
|---|---|---|---|
| `sys.cpu.util` | CPU Utilization | `%` | `GetSystemTimes` deltas |
| `sys.mem.used` | Memory Used | `%` | `GlobalMemoryStatusEx` `dwMemoryLoad` |
| `sys.gpu.util` | GPU Utilization | `%` | NVML, loaded at runtime from the driver `nvml.dll`. The host skips failed ticks and the loss is visible. |
| `sys.gpu.temp` | GPU Temperature | `degC` | NVML |
| `sys.gpu.power` | GPU Power | `W` | NVML, gated against the device-reported enforced power limit |
| `sys.power.battery` | Battery Power | `W` | `CallNtPowerInformation`. Negative means discharging, as on macOS. Absent on desktops. |

Windows has no unprivileged path to CPU temperature or fan speed, so those channels do not exist there. The NVML channels exist only when an NVIDIA driver is present. The probe fails closed at start, and the host links no SDK.

An unavailable signal is not a channel and never has a faked value. A consumer displays whatever enumerates. On a host with no channels, it says so in one line and draws nothing. PulseScope works this way (see [Applets](../in-tree-applets.md)).
