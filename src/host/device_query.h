#pragma once
#include <caliper/services/device_v1.h>
#include <cstdint>
#include <string>

namespace caliper_host {

struct DeviceInfo {
    CaliperDeviceKind kind = CALIPER_DEV_CPU;
    int32_t index = 0;
    std::string name = "CPU";
    uint64_t free_memory_hint = 0;   // bytes; 0 = unknown
};

// Detect-once, cached for the process lifetime. Never links an ML framework
// (D11): Metal is queried directly on Apple; CUDA detection arrives with
// Phase 4 hardware (until then non-Apple reports CPU).
const DeviceInfo& device_info();

} // namespace caliper_host
