#pragma once
// caliper_host — the Windows system-telemetry provider for caliper.feed.v1
// (feed spec §6.2, the T2-sibling hardware pass). WIN32 ONLY: it samples this
// box's ADMIN-FREE sensors at 10 Hz into a FeedStore. Hosts with no provider
// (Linux/other) keep zero channels — the honest-degradation default from T1.
//
// Privilege-honest (spec invariant): only signals that read WITHOUT elevation
// on THIS machine are registered as channels. Everything is probed once at
// start(); a signal that does not read is simply not a channel, never a faked
// value. The channel set on this box (RTX 500 Ada laptop, Windows 11):
//   guaranteed — sys.cpu.util (GetSystemTimes), sys.mem.used
//   (GlobalMemoryStatusEx), sys.power.battery (CallNtPowerInformation);
//   probe tier — sys.gpu.util / sys.gpu.temp / sys.gpu.power via NVML,
//   runtime-loaded from the NVIDIA driver's nvml.dll (never a linked SDK, per
//   D11 / the cuda_driver.cpp house pattern). No NVIDIA driver → no GPU
//   channels. CPU temperature and fan RPM are admin-gated on Windows and
//   therefore DO NOT EXIST here.
//
// host_services.cpp drives these two calls from services_init /
// services_shutdown (the #elif defined(_WIN32) branch of the same anchors the
// macOS provider uses): start AFTER the store exists, stop+JOIN BEFORE
// anything the thread writes is torn down. Both are idempotent and re-entrant
// across services_init/shutdown cycling (the embed create/shutdown/create
// battery): channels are registered exactly once on the persistent store, and
// a fresh sampling thread + NVML session are spun up each start and released
// each stop.
namespace caliper_host {

class FeedStore;

// Probe the admin-free sensors, register the readable ones into `store` (once;
// already-registered ids are left as-is), and start the 10 Hz sampling thread.
// Call from services_init AFTER `store` is constructed. No-op-safe to call
// again after a matching feed_provider_stop().
void feed_provider_start(FeedStore& store);

// Signal + JOIN the sampling thread, close the NVML session, and unload
// nvml.dll. Call from services_shutdown BEFORE the store is torn down. Safe if
// never started.
void feed_provider_stop();

}  // namespace caliper_host
