#pragma once
// caliper_host — the macOS system-telemetry provider for caliper.feed.v1 (feed
// spec §4, task T2). APPLE ONLY: it samples the Mac's own SUDO-FREE sensors at
// 10 Hz into a FeedStore. Non-Apple hosts have no provider and keep zero
// channels — the honest-degradation default already landed in T1.
//
// Privilege-honest (spec invariant): only signals that read WITHOUT sudo or
// entitlements on THIS machine are registered as channels. Everything is probed
// once at start(); a signal that does not read is simply not a channel, never a
// faked value. The channel set on this box (Apple M5, macOS 26): the guaranteed
// tier — sys.cpu.util, sys.mem.pressure, sys.thermal.state, sys.gpu.util — plus
// the best-effort tier that verified readable here — sys.fan.rpm and
// sys.temp.battery (userland AppleSMC), sys.power.battery (IOPMPowerSource).
//
// host_services.cpp drives these two calls from services_init / services_shutdown
// (guarded by #ifdef __APPLE__): start AFTER the store exists, stop+JOIN BEFORE
// anything the thread writes is torn down. Both are idempotent and re-entrant
// across services_init/shutdown cycling (the embed create/shutdown/create
// battery): channels are registered exactly once on the persistent store, and a
// fresh sampling thread + sensor handles are spun up each start and released
// each stop.
namespace caliper_host {

class FeedStore;

// Probe the sudo-free sensors, register the readable ones into `store` (once;
// already-registered ids are left as-is), and start the 10 Hz sampling thread.
// Call from services_init AFTER `store` is constructed. No-op-safe to call again
// after a matching feed_provider_stop().
void feed_provider_start(FeedStore& store);

// Signal + JOIN the sampling thread and release the sensor handles. Call from
// services_shutdown BEFORE the store is torn down. Safe if never started.
void feed_provider_stop();

}  // namespace caliper_host
