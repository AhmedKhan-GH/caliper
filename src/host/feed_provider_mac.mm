#include "feed_provider_mac.h"
#include "feed_store.h"

#import <Foundation/Foundation.h>
#import <IOKit/IOKitLib.h>

#include <mach/mach.h>
#include <mach/mach_host.h>
#include <mach/mach_time.h>
#include <mach/processor_info.h>
#include <sys/sysctl.h>

#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <thread>

// The macOS telemetry provider (feed spec §4 / T2). Every signal here was probed
// SUDO-FREE on this box before it was written (see the T2 report); a signal that
// does not read is not registered — privilege-honest, never a faked value.
namespace caliper_host {
namespace {

// Spec §2: 4096 samples ≈ 6+ minutes at 10 Hz. This ring capacity is the
// provider's (call-site) responsibility — made explicit here, once.
constexpr uint32_t kRingCapacity = 4096;
constexpr int      kPeriodMs     = 100;    // 10 Hz (spec §4)

// Host steady clock in ns (mach_absolute-derived), the same epoch across
// channels so applets can align series (spec §2). 128-bit intermediate so the
// numer multiply cannot overflow before the denom divide.
int64_t now_ns() {
    static mach_timebase_info_data_t tb = {0, 0};
    if (tb.denom == 0) mach_timebase_info(&tb);
    return (int64_t)((__uint128_t)mach_absolute_time() * tb.numer / tb.denom);
}

// ---------------------------------------------------------------------------
// Minimal userland AppleSMC reader — the canonical 80-byte param struct the
// SMC user client expects (selector 2 = kSMCHandleYPCEvent). Verified to open
// and read WITHOUT privilege on this machine (Apple M5, macOS 26); if IOServiceOpen
// or a key read fails, the fan/temp channels are simply not registered.
// ---------------------------------------------------------------------------
struct SMCParamStruct {
    uint32_t key;
    struct { char major, minor, build, reserved[1]; uint16_t release; } vers;
    struct { uint16_t version, length; uint32_t cpuPLimit, gpuPLimit, memPLimit; } pLimit;
    struct { uint32_t dataSize, dataType; uint8_t dataAttributes; } keyInfo;
    uint8_t  result, status, data8;
    uint32_t data32;
    uint8_t  bytes[32];
};
constexpr uint8_t kSMCReadKeyInfo = 9;
constexpr uint8_t kSMCReadBytes   = 5;

uint32_t smc_fourcc(const char* s) {
    return ((uint32_t)s[0] << 24) | ((uint32_t)s[1] << 16) |
           ((uint32_t)s[2] << 8) | (uint32_t)s[3];
}

// Read a "flt " (IEEE-754 float) SMC key through `conn`. Returns false on any
// failure or a non-float key. Fan RPM (F0Ac) and temperatures (TB0T) are both
// "flt " on Apple Silicon.
bool smc_read_flt(io_connect_t conn, const char* key4, float* out) {
    if (!conn) return false;
    SMCParamStruct in{}, ki{};
    in.key = smc_fourcc(key4);
    in.data8 = kSMCReadKeyInfo;
    size_t sz = sizeof(ki);
    if (IOConnectCallStructMethod(conn, 2, &in, sizeof in, &ki, &sz) != KERN_SUCCESS)
        return false;
    if (ki.keyInfo.dataSize != 4) return false;                 // not a 4-byte scalar
    // dataType is a FourCC packed big-endian in the u32; compare as bytes.
    char t[4] = { (char)(ki.keyInfo.dataType >> 24), (char)(ki.keyInfo.dataType >> 16),
                  (char)(ki.keyInfo.dataType >> 8), (char)ki.keyInfo.dataType };
    if (std::memcmp(t, "flt ", 4) != 0) return false;
    SMCParamStruct vin{}, vout{};
    vin.key = smc_fourcc(key4);
    vin.keyInfo.dataSize = 4;
    vin.data8 = kSMCReadBytes;
    sz = sizeof(vout);
    if (IOConnectCallStructMethod(conn, 2, &vin, sizeof vin, &vout, &sz) != KERN_SUCCESS)
        return false;
    float f;
    std::memcpy(&f, vout.bytes, 4);   // "flt " bytes are native-endian little on ARM64
    *out = f;
    return true;
}

// ---------------------------------------------------------------------------
// The single machine provider (one per process — one host, one sensor set).
// ---------------------------------------------------------------------------
struct Provider {
    FeedStore*        store = nullptr;
    std::thread       th;
    std::atomic<bool> stop{false};
    bool              running = false;   // lifecycle-thread only (no data race)

    // Which signals proved readable at start() → which channels exist.
    bool has_cpu = false, has_mem = false, has_gpu = false;
    bool has_fan = false, has_temp = false, has_power = false;
    // thermal state is always available (public ProcessInfo API).

    // CPU delta baseline (aggregate busy/total ticks across all cores).
    uint64_t prev_busy = 0, prev_total = 0;
    uint64_t total_pages = 0;            // for the memory-pressure denominator

    // Cached IOKit handles (opened at start, released at stop).
    io_registry_entry_t gpu_service = 0;   // IOAccelerator
    io_registry_entry_t pm_service  = 0;   // IOPMPowerSource
    io_connect_t        smc_conn    = 0;   // AppleSMC user client

    // One-shot warn latches (a signal that goes stale mid-run logs ONCE).
    bool warned_cpu = false, warned_mem = false, warned_gpu = false;
    bool warned_fan = false, warned_temp = false, warned_power = false;
};
Provider g_prov;

// --- samplers: each returns false on a read failure (skip the tick) ---

// Aggregate CPU busy/total ticks over all cores.
bool read_cpu_ticks(uint64_t* busy, uint64_t* total) {
    natural_t ncpu = 0;
    processor_info_array_t info = nullptr;
    mach_msg_type_number_t cnt = 0;
    if (host_processor_info(mach_host_self(), PROCESSOR_CPU_LOAD_INFO,
                            &ncpu, &info, &cnt) != KERN_SUCCESS)
        return false;
    auto* load = (processor_cpu_load_info_t)info;
    uint64_t b = 0, tot = 0;
    for (natural_t i = 0; i < ncpu; ++i) {
        uint64_t u = load[i].cpu_ticks[CPU_STATE_USER];
        uint64_t s = load[i].cpu_ticks[CPU_STATE_SYSTEM];
        uint64_t n = load[i].cpu_ticks[CPU_STATE_NICE];
        uint64_t idle = load[i].cpu_ticks[CPU_STATE_IDLE];
        b += u + s + n;
        tot += u + s + n + idle;
    }
    vm_deallocate(mach_task_self(), (vm_address_t)info, cnt * sizeof(int));
    *busy = b;
    *total = tot;
    return true;
}

// CPU utilization % from the delta since the previous tick.
bool sample_cpu(float* out) {
    uint64_t busy, total;
    if (!read_cpu_ticks(&busy, &total)) return false;
    uint64_t db = busy - g_prov.prev_busy;
    uint64_t dt = total - g_prov.prev_total;
    g_prov.prev_busy = busy;
    g_prov.prev_total = total;
    *out = dt ? (float)(100.0 * (double)db / (double)dt) : 0.0f;
    return true;
}

// Memory pressure %: (active + wired + compressed) / physical, a sudo-free
// continuous proxy for the used-memory footprint (spec: host_statistics64).
bool sample_mem(float* out) {
    vm_statistics64_data_t vm;
    mach_msg_type_number_t cnt = HOST_VM_INFO64_COUNT;
    if (host_statistics64(mach_host_self(), HOST_VM_INFO64,
                          (host_info64_t)&vm, &cnt) != KERN_SUCCESS)
        return false;
    if (g_prov.total_pages == 0) return false;
    uint64_t used = (uint64_t)vm.active_count + vm.wire_count + vm.compressor_page_count;
    *out = (float)(100.0 * (double)used / (double)g_prov.total_pages);
    return true;
}

int thermal_state() {
    @autoreleasepool {
        return (int)[[NSProcessInfo processInfo] thermalState];   // 0..3
    }
}

// GPU "Device Utilization %" from the cached IOAccelerator entry.
bool sample_gpu(float* out) {
    if (!g_prov.gpu_service) return false;
    CFMutableDictionaryRef props = nullptr;
    if (IORegistryEntryCreateCFProperties(g_prov.gpu_service, &props,
                                          kCFAllocatorDefault, 0) != KERN_SUCCESS ||
        !props)
        return false;
    bool ok = false;
    if (auto ps = (CFDictionaryRef)CFDictionaryGetValue(props,
                                                        CFSTR("PerformanceStatistics"))) {
        if (auto du = (CFNumberRef)CFDictionaryGetValue(ps,
                                                        CFSTR("Device Utilization %"))) {
            int v = 0;
            CFNumberGetValue(du, kCFNumberIntType, &v);
            *out = (float)v;
            ok = true;
        }
    }
    CFRelease(props);
    return ok;
}

// Net battery power W from the cached IOPMPowerSource entry: InstantAmperage
// (mA, signed; negative = discharge) × Voltage (mV) / 1e6. ~0 W when charged
// on AC — honest, and it swings once running on battery.
bool sample_power(float* out) {
    if (!g_prov.pm_service) return false;
    CFMutableDictionaryRef props = nullptr;
    if (IORegistryEntryCreateCFProperties(g_prov.pm_service, &props,
                                          kCFAllocatorDefault, 0) != KERN_SUCCESS ||
        !props)
        return false;
    int iamp = 0, volt = 0;
    bool hi = false, hv = false;
    if (auto v = (CFNumberRef)CFDictionaryGetValue(props, CFSTR("InstantAmperage"))) {
        CFNumberGetValue(v, kCFNumberIntType, &iamp); hi = true;
    }
    if (auto v = (CFNumberRef)CFDictionaryGetValue(props, CFSTR("Voltage"))) {
        CFNumberGetValue(v, kCFNumberIntType, &volt); hv = true;
    }
    CFRelease(props);
    if (!(hi && hv)) return false;
    *out = (float)((double)iamp * (double)volt / 1.0e6);
    return true;
}

void warn_once(bool* latch, const char* channel) {
    if (*latch) return;
    *latch = true;
    std::fprintf(stderr, "[feed] %s: sensor read failed; channel goes stale "
                         "(no samples) until it recovers\n", channel);
}

// The 10 Hz sampling loop. Each available channel is sampled and pushed with a
// single shared timestamp per tick; a read failure skips only that channel
// (stale = visible via timestamps), never a faked value.
void sample_loop() {
    while (!g_prov.stop.load(std::memory_order_relaxed)) {
        const int64_t t = now_ns();
        @autoreleasepool {
            float v;
            if (g_prov.has_cpu) {
                if (sample_cpu(&v)) g_prov.store->push("sys.cpu.util", t, v);
                else warn_once(&g_prov.warned_cpu, "sys.cpu.util");
            }
            if (g_prov.has_mem) {
                if (sample_mem(&v)) g_prov.store->push("sys.mem.used", t, v);
                else warn_once(&g_prov.warned_mem, "sys.mem.used");
            }
            // thermal state never fails (public API)
            g_prov.store->push("sys.thermal.state", t, (float)thermal_state());
            if (g_prov.has_gpu) {
                if (sample_gpu(&v)) g_prov.store->push("sys.gpu.util", t, v);
                else warn_once(&g_prov.warned_gpu, "sys.gpu.util");
            }
            if (g_prov.has_fan) {
                if (smc_read_flt(g_prov.smc_conn, "F0Ac", &v))
                    g_prov.store->push("sys.fan.rpm", t, v);
                else warn_once(&g_prov.warned_fan, "sys.fan.rpm");
            }
            if (g_prov.has_temp) {
                if (smc_read_flt(g_prov.smc_conn, "TB0T", &v))
                    g_prov.store->push("sys.temp.battery", t, v);
                else warn_once(&g_prov.warned_temp, "sys.temp.battery");
            }
            if (g_prov.has_power) {
                if (sample_power(&v)) g_prov.store->push("sys.power.battery", t, v);
                else warn_once(&g_prov.warned_power, "sys.power.battery");
            }
        }
        // Sleep the period in short slices so stop() joins within ~20 ms.
        for (int i = 0; i < kPeriodMs / 20 &&
                        !g_prov.stop.load(std::memory_order_relaxed); ++i)
            std::this_thread::sleep_for(std::chrono::milliseconds(20));
    }
}

}  // namespace

void feed_provider_start(FeedStore& store) {
    if (g_prov.running) return;   // already live (idempotent)
    g_prov.store = &store;
    g_prov.stop.store(false, std::memory_order_relaxed);

    // --- probe availability + prime baselines (all sudo-free) ---
    // CPU: baseline the tick counters so the first sample is a real delta.
    g_prov.has_cpu = read_cpu_ticks(&g_prov.prev_busy, &g_prov.prev_total);

    // Memory: cache the page-count denominator once.
    {
        vm_size_t page = 0;
        host_page_size(mach_host_self(), &page);
        uint64_t memsize = 0; size_t sz = sizeof(memsize);
        if (sysctlbyname("hw.memsize", &memsize, &sz, nullptr, 0) == 0 && page)
            g_prov.total_pages = memsize / page;
        float tmp;
        g_prov.has_mem = (g_prov.total_pages != 0) && sample_mem(&tmp);
    }

    // GPU: cache the IOAccelerator entry iff Device Utilization reads.
    g_prov.gpu_service = IOServiceGetMatchingService(kIOMainPortDefault,
                                                     IOServiceMatching("IOAccelerator"));
    {
        float tmp;
        g_prov.has_gpu = sample_gpu(&tmp);
        if (!g_prov.has_gpu && g_prov.gpu_service) {
            IOObjectRelease(g_prov.gpu_service);
            g_prov.gpu_service = 0;
        }
    }

    // Power: cache the IOPMPowerSource entry iff Voltage+InstantAmperage read.
    g_prov.pm_service = IOServiceGetMatchingService(kIOMainPortDefault,
                                                    IOServiceMatching("IOPMPowerSource"));
    {
        float tmp;
        g_prov.has_power = sample_power(&tmp);
        if (!g_prov.has_power && g_prov.pm_service) {
            IOObjectRelease(g_prov.pm_service);
            g_prov.pm_service = 0;
        }
    }

    // SMC (best-effort): open the userland client; register fan/temp only if
    // their keys actually read. If the open needs privilege we don't have, the
    // channels simply do not exist (privilege-honest).
    {
        io_service_t svc = IOServiceGetMatchingService(kIOMainPortDefault,
                                                       IOServiceMatching("AppleSMC"));
        if (svc) {
            if (IOServiceOpen(svc, mach_task_self(), 0, &g_prov.smc_conn) != KERN_SUCCESS)
                g_prov.smc_conn = 0;
            IOObjectRelease(svc);
        }
        float tmp;
        g_prov.has_fan  = smc_read_flt(g_prov.smc_conn, "F0Ac", &tmp);
        g_prov.has_temp = smc_read_flt(g_prov.smc_conn, "TB0T", &tmp);
        if (!g_prov.has_fan && !g_prov.has_temp && g_prov.smc_conn) {
            IOServiceClose(g_prov.smc_conn);
            g_prov.smc_conn = 0;
        }
    }

    // Register the readable channels (once; the store refuses duplicates, so a
    // re-start after a stop on the SAME persistent store is a no-op here). The
    // 4096-sample ring capacity is passed explicitly at every call site (§2).
    struct Reg { bool on; const char* id; const char* name; const char* units; };
    const Reg regs[] = {
        {g_prov.has_cpu,   "sys.cpu.util",     "CPU Utilization",     "%"},
        {g_prov.has_mem,   "sys.mem.used", "Memory Used",     "%"},
        {true,             "sys.thermal.state","Thermal State",       ""},
        {g_prov.has_gpu,   "sys.gpu.util",     "GPU Utilization",     "%"},
        {g_prov.has_fan,   "sys.fan.rpm",      "Fan Speed",           "rpm"},
        {g_prov.has_temp,  "sys.temp.battery", "Battery Temperature", "degC"},
        {g_prov.has_power, "sys.power.battery","Battery Power",       "W"},
    };
    for (const auto& r : regs) {
        if (!r.on) continue;
        store.add_channel(r.id, r.name, r.units, 10.0f, kRingCapacity);
    }
    // Report the store's authoritative live count (add_channel refuses dups, so a
    // re-start on the persistent store adds nothing new — the count is unchanged).
    std::fprintf(stderr,
                 "[feed] macOS provider: %u channels live "
                 "(cpu=%d mem=%d thermal=1 gpu=%d fan=%d temp=%d power=%d) @ 10 Hz\n",
                 store.channel_count(), g_prov.has_cpu, g_prov.has_mem, g_prov.has_gpu,
                 g_prov.has_fan, g_prov.has_temp, g_prov.has_power);

    g_prov.running = true;
    g_prov.th = std::thread(sample_loop);
}

void feed_provider_stop() {
    if (!g_prov.running) return;
    g_prov.stop.store(true, std::memory_order_relaxed);
    if (g_prov.th.joinable()) g_prov.th.join();
    g_prov.running = false;

    // Release sensor handles so the next start re-opens cleanly (embed cycling).
    if (g_prov.smc_conn)    { IOServiceClose(g_prov.smc_conn);    g_prov.smc_conn = 0; }
    if (g_prov.gpu_service) { IOObjectRelease(g_prov.gpu_service); g_prov.gpu_service = 0; }
    if (g_prov.pm_service)  { IOObjectRelease(g_prov.pm_service);  g_prov.pm_service = 0; }
    // Re-arm the one-shot warn latches for the next run.
    g_prov.warned_cpu = g_prov.warned_mem = g_prov.warned_gpu = false;
    g_prov.warned_fan = g_prov.warned_temp = g_prov.warned_power = false;
}

}  // namespace caliper_host
