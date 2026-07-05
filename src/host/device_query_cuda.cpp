#include "device_query.h"
#include "cuda_driver.h"
// Non-Apple device query with real CUDA detection (Phase 4). Uses the driver
// API loaded at runtime (cuda_driver.h) so the host still links no ML
// framework and no CUDA toolkit (D11). No NVIDIA driver -> CPU defaults,
// same behavior as the old stub.
namespace caliper_host {

namespace {
DeviceInfo detect() {
    DeviceInfo info{};   // CPU defaults
    const cudadrv::Api* cu = cudadrv::api();
    if (!cu) return info;

    int count = 0;
    if (cu->cuDeviceGetCount(&count) != cudadrv::CUDA_SUCCESS || count <= 0)
        return info;

    char name[256] = {};
    if (cu->cuDeviceGetName(name, sizeof(name), 0) != cudadrv::CUDA_SUCCESS)
        return info;

    info.kind = CALIPER_DEV_CUDA;
    info.index = 0;
    info.name = name;
    size_t total = 0;
    if (cu->cuDeviceTotalMem(&total, 0) == cudadrv::CUDA_SUCCESS)
        info.free_memory_hint = (uint64_t)total;   // total as the hint; free
                                                   // needs a context (skip)
    return info;
}
}  // namespace

const DeviceInfo& device_info() {
    static const DeviceInfo info = detect();
    return info;
}

}  // namespace caliper_host
