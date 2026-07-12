#include "feed_provider_win.h"
#include "feed_store.h"

#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <powerbase.h>

#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <thread>

// The Windows telemetry provider (feed spec §6.2). Every signal here was probed
// ADMIN-FREE on this box before it was written (see the pass report); a signal
// that does not read is not registered — privilege-honest, never a faked value.
// CPU temperature and fan RPM need a kernel driver on Windows, so per the
// invariant those channels do not exist here.
namespace caliper_host {
namespace {

// Spec §2: 4096 samples ≈ 6+ minutes at 10 Hz. This ring capacity is the
// provider's (call-site) responsibility — made explicit here, once.
constexpr uint32_t kRingCapacity = 4096;
constexpr int      kPeriodMs     = 100;    // 10 Hz (spec §4)

// Host steady clock in ns. MSVC's steady_clock is QueryPerformanceCounter
// underneath — the same epoch across channels so applets can align series.
int64_t now_ns() {
    return (int64_t)std::chrono::duration_cast<std::chrono::nanoseconds>(
               std::chrono::steady_clock::now().time_since_epoch())
        .count();
}

// ---------------------------------------------------------------------------
// Minimal NVML surface, runtime-loaded from the driver's nvml.dll (house
// pattern: cuda_driver.cpp — never link a vendor SDK, D11). No driver, a
// missing export, or a failed init → the GPU channels simply don't register.
// Probed on this box (RTX 500 Ada, driver 596.47): all exports resolve;
// temperature reads 100%; utilization intermittently returns rc 999 while the
// dGPU is RTD3-parked (skip the tick); power once returned a wake-transition
// junk value (364 W on a 40 W-limit part) with rc 0, so samples are gated
// against the device-reported enforced power limit.
// ---------------------------------------------------------------------------
typedef int nvmlReturn_t;                  // 0 == NVML_SUCCESS
typedef struct nvmlDevice_st* nvmlDevice_t;
struct NvmlUtilization { unsigned int gpu; unsigned int memory; };
constexpr int kNvmlTempGpu = 0;            // NVML_TEMPERATURE_GPU

struct Nvml {
    HMODULE      dll = nullptr;
    bool         inited = false;
    nvmlDevice_t dev = nullptr;
    unsigned int junk_bound_mw = 0;        // 0 = no bound proved; accept all

    nvmlReturn_t (*init)(void) = nullptr;
    nvmlReturn_t (*shutdown)(void) = nullptr;
    nvmlReturn_t (*handle_by_index)(unsigned int, nvmlDevice_t*) = nullptr;
    nvmlReturn_t (*utilization)(nvmlDevice_t, NvmlUtilization*) = nullptr;
    nvmlReturn_t (*temperature)(nvmlDevice_t, int, unsigned int*) = nullptr;
    nvmlReturn_t (*power_usage)(nvmlDevice_t, unsigned int*) = nullptr;
    nvmlReturn_t (*enforced_limit)(nvmlDevice_t, unsigned int*) = nullptr;
};

template <typename Fn>
bool nvml_sym(HMODULE dll, const char* name, Fn* out) {
    *out = (Fn)(void*)GetProcAddress(dll, name);
    return *out != nullptr;
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
    bool has_cpu = false, has_mem = false, has_power = false;
    bool has_gpu = false, has_gpu_temp = false, has_gpu_power = false;

    // CPU delta baseline (GetSystemTimes FILETIMEs as 100 ns ticks).
    uint64_t prev_idle = 0, prev_kernel = 0, prev_user = 0;

    Nvml nvml;   // loaded at start, closed at stop (embed cycling re-opens)

    // One-shot warn latches (a signal that fails mid-run logs ONCE).
    bool warned_cpu = false, warned_mem = false, warned_power = false;
    bool warned_gpu = false, warned_gpu_temp = false, warned_gpu_power = false;
};
Provider g_prov;

// --- samplers: each returns false on a read failure (skip the tick) ---

uint64_t ft_ticks(FILETIME f) {
    return ((uint64_t)f.dwHighDateTime << 32) | f.dwLowDateTime;
}

bool read_cpu_times(uint64_t* idle, uint64_t* kernel, uint64_t* user) {
    FILETIME fi, fk, fu;
    if (!GetSystemTimes(&fi, &fk, &fu)) return false;
    *idle = ft_ticks(fi);
    *kernel = ft_ticks(fk);   // kernel time INCLUDES idle
    *user = ft_ticks(fu);
    return true;
}

// CPU utilization % from the delta since the previous tick.
bool sample_cpu(float* out) {
    uint64_t idle, kernel, user;
    if (!read_cpu_times(&idle, &kernel, &user)) return false;
    uint64_t di = idle - g_prov.prev_idle;
    uint64_t dt = (kernel - g_prov.prev_kernel) + (user - g_prov.prev_user);
    g_prov.prev_idle = idle;
    g_prov.prev_kernel = kernel;
    g_prov.prev_user = user;
    *out = dt ? (float)(100.0 * (double)(dt - di) / (double)dt) : 0.0f;
    return true;
}

// Used-memory % straight from the OS (dwMemoryLoad IS this — the T2-ratified
// name: it is used-memory, not pressure; the name never overclaims).
bool sample_mem(float* out) {
    MEMORYSTATUSEX ms;
    ms.dwLength = sizeof(ms);
    if (!GlobalMemoryStatusEx(&ms)) return false;
    *out = (float)ms.dwMemoryLoad;
    return true;
}

// Net battery power W. SYSTEM_BATTERY_STATE.Rate is mW, signed-in-a-ULONG:
// negative = discharging — the SAME convention as the macOS channel
// (InstantAmperage-derived). ~0 W when charged on AC — honest, and it swings
// once running on battery (probed on this box: present, 0 on AC at 96%).
bool sample_power(float* out) {
    SYSTEM_BATTERY_STATE b;
    // NTSTATUS spelled as LONG (0 == STATUS_SUCCESS): winternl.h stays out of
    // this TU under WIN32_LEAN_AND_MEAN.
    LONG st = CallNtPowerInformation(SystemBatteryState, nullptr, 0,
                                     &b, sizeof(b));
    if (st != 0 || !b.BatteryPresent) return false;
    *out = (float)((LONG)b.Rate / 1000.0);
    return true;
}

// --- NVML samplers (all skip-on-fail; see the probe notes on the Nvml block) ---

bool sample_gpu_util(float* out) {
    if (!g_prov.nvml.dev) return false;
    NvmlUtilization u;
    if (g_prov.nvml.utilization(g_prov.nvml.dev, &u) != 0) return false;
    *out = (float)u.gpu;
    return true;
}

bool sample_gpu_temp(float* out) {
    if (!g_prov.nvml.dev) return false;
    unsigned int c = 0;
    if (g_prov.nvml.temperature(g_prov.nvml.dev, kNvmlTempGpu, &c) != 0)
        return false;
    *out = (float)c;
    return true;
}

bool sample_gpu_power(float* out) {
    if (!g_prov.nvml.dev) return false;
    unsigned int mw = 0;
    if (g_prov.nvml.power_usage(g_prov.nvml.dev, &mw) != 0) return false;
    // Wake-transition junk gate: rc 0 with an impossible value (probed: 364 W
    // on a 40 W-limit part). The bound is the DEVICE-reported enforced limit
    // ×1.5 — never a magic constant, and absent a proved limit there is no gate.
    unsigned int bound = g_prov.nvml.junk_bound_mw;
    if (bound && mw > bound) return false;
    *out = (float)(mw / 1000.0);
    return true;
}

// Probe an NVML sampler at start(): RTD3 parking can fail any single call
// (probed: ~24% of ticks), so a channel earns registration on ANY success
// within a few attempts — and still skips failing ticks mid-run.
bool probe_nvml(bool (*sampler)(float*)) {
    for (int i = 0; i < 3; ++i) {
        float tmp;
        if (sampler(&tmp)) return true;
        Sleep(50);
    }
    return false;
}

void nvml_close() {
    Nvml& n = g_prov.nvml;
    if (n.inited && n.shutdown) n.shutdown();
    if (n.dll) FreeLibrary(n.dll);
    n = Nvml{};
}

// Load + init NVML and cache the device-0 handle and power junk bound.
// Any failure → nvml_close() → all GPU channels absent (fail-closed).
bool nvml_open() {
    Nvml& n = g_prov.nvml;
    n.dll = LoadLibraryA("nvml.dll");   // installed to System32 by the driver
    if (!n.dll) return false;
    bool ok = nvml_sym(n.dll, "nvmlInit_v2", &n.init) &&
              nvml_sym(n.dll, "nvmlShutdown", &n.shutdown) &&
              nvml_sym(n.dll, "nvmlDeviceGetHandleByIndex_v2", &n.handle_by_index) &&
              nvml_sym(n.dll, "nvmlDeviceGetUtilizationRates", &n.utilization) &&
              nvml_sym(n.dll, "nvmlDeviceGetTemperature", &n.temperature) &&
              nvml_sym(n.dll, "nvmlDeviceGetPowerUsage", &n.power_usage);
    if (!ok) { nvml_close(); return false; }
    if (n.init() != 0) { nvml_close(); return false; }
    n.inited = true;
    if (n.handle_by_index(0, &n.dev) != 0 || !n.dev) { nvml_close(); return false; }
    // Optional export (older drivers may lack it): without it, no junk gate.
    if (nvml_sym(n.dll, "nvmlDeviceGetEnforcedPowerLimit", &n.enforced_limit)) {
        unsigned int limit_mw = 0;
        if (n.enforced_limit(n.dev, &limit_mw) == 0 && limit_mw)
            n.junk_bound_mw = limit_mw + limit_mw / 2;   // ×1.5 headroom
    }
    return true;
}

void warn_once(bool* latch, const char* channel) {
    if (*latch) return;
    *latch = true;
    std::fprintf(stderr, "[feed] %s: sensor read failed; failing ticks are "
                         "skipped, never a faked value\n", channel);
}

// The 10 Hz sampling loop. Each available channel is sampled and pushed with a
// single shared timestamp per tick; a read failure skips only that channel
// (stale = visible via timestamps), never a faked value.
void sample_loop() {
    while (!g_prov.stop.load(std::memory_order_relaxed)) {
        const int64_t t = now_ns();
        float v;
        if (g_prov.has_cpu) {
            if (sample_cpu(&v)) g_prov.store->push("sys.cpu.util", t, v);
            else warn_once(&g_prov.warned_cpu, "sys.cpu.util");
        }
        if (g_prov.has_mem) {
            if (sample_mem(&v)) g_prov.store->push("sys.mem.used", t, v);
            else warn_once(&g_prov.warned_mem, "sys.mem.used");
        }
        if (g_prov.has_gpu) {
            if (sample_gpu_util(&v)) g_prov.store->push("sys.gpu.util", t, v);
            else warn_once(&g_prov.warned_gpu, "sys.gpu.util");
        }
        if (g_prov.has_gpu_temp) {
            if (sample_gpu_temp(&v)) g_prov.store->push("sys.gpu.temp", t, v);
            else warn_once(&g_prov.warned_gpu_temp, "sys.gpu.temp");
        }
        if (g_prov.has_gpu_power) {
            if (sample_gpu_power(&v)) g_prov.store->push("sys.gpu.power", t, v);
            else warn_once(&g_prov.warned_gpu_power, "sys.gpu.power");
        }
        if (g_prov.has_power) {
            if (sample_power(&v)) g_prov.store->push("sys.power.battery", t, v);
            else warn_once(&g_prov.warned_power, "sys.power.battery");
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

    // --- probe availability + prime baselines (all admin-free) ---
    // CPU: baseline the time counters so the first sample is a real delta.
    g_prov.has_cpu = read_cpu_times(&g_prov.prev_idle, &g_prov.prev_kernel,
                                    &g_prov.prev_user);
    {
        float tmp;
        g_prov.has_mem = sample_mem(&tmp);
        // Battery: absent on a desktop box — the channel honestly does not
        // exist there (BatteryPresent gates inside the sampler).
        g_prov.has_power = sample_power(&tmp);
    }

    // GPU (probe tier): NVML from the installed driver, else no GPU channels.
    if (nvml_open()) {
        g_prov.has_gpu       = probe_nvml(&sample_gpu_util);
        g_prov.has_gpu_temp  = probe_nvml(&sample_gpu_temp);
        g_prov.has_gpu_power = probe_nvml(&sample_gpu_power);
        if (!g_prov.has_gpu && !g_prov.has_gpu_temp && !g_prov.has_gpu_power)
            nvml_close();   // nothing read — don't hold the driver open
    }

    // Register the readable channels (once; the store refuses duplicates, so a
    // re-start after a stop on the SAME persistent store is a no-op here). The
    // 4096-sample ring capacity is passed explicitly at every call site (§2).
    struct Reg { bool on; const char* id; const char* name; const char* units; };
    const Reg regs[] = {
        {g_prov.has_cpu,       "sys.cpu.util",     "CPU Utilization",  "%"},
        {g_prov.has_mem,       "sys.mem.used",     "Memory Used",      "%"},
        {g_prov.has_gpu,       "sys.gpu.util",     "GPU Utilization",  "%"},
        {g_prov.has_gpu_temp,  "sys.gpu.temp",     "GPU Temperature",  "degC"},
        {g_prov.has_gpu_power, "sys.gpu.power",    "GPU Power",        "W"},
        {g_prov.has_power,     "sys.power.battery","Battery Power",    "W"},
    };
    for (const auto& r : regs) {
        if (!r.on) continue;
        store.add_channel(r.id, r.name, r.units, 10.0f, kRingCapacity);
    }
    // Report the store's authoritative live count (add_channel refuses dups, so
    // a re-start on the persistent store adds nothing new — count unchanged).
    std::fprintf(stderr,
                 "[feed] Windows provider: %u channels live "
                 "(cpu=%d mem=%d gpu=%d gputemp=%d gpupower=%d power=%d) @ 10 Hz\n",
                 store.channel_count(), g_prov.has_cpu, g_prov.has_mem,
                 g_prov.has_gpu, g_prov.has_gpu_temp, g_prov.has_gpu_power,
                 g_prov.has_power);

    g_prov.running = true;
    g_prov.th = std::thread(sample_loop);
}

void feed_provider_stop() {
    if (!g_prov.running) return;
    g_prov.stop.store(true, std::memory_order_relaxed);
    if (g_prov.th.joinable()) g_prov.th.join();
    g_prov.running = false;

    // Close the NVML session so the next start re-opens cleanly (embed cycling).
    nvml_close();
    // Re-arm the one-shot warn latches for the next run.
    g_prov.warned_cpu = g_prov.warned_mem = g_prov.warned_power = false;
    g_prov.warned_gpu = g_prov.warned_gpu_temp = g_prov.warned_gpu_power = false;
}

}  // namespace caliper_host
