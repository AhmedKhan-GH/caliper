#include "device_query.h"
// Non-Apple fallback until Phase 4 brings CUDA detection (needs hardware/CI).
namespace caliper_host {
const DeviceInfo& device_info() {
    static const DeviceInfo info{};   // CPU defaults
    return info;
}
} // namespace caliper_host
