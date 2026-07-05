#include <doctest/doctest.h>
#include "device_query.h"
using namespace caliper_host;

TEST_CASE("device: detection is stable and sane on this machine") {
    const DeviceInfo& a = device_info();
    const DeviceInfo& b = device_info();
    CHECK(&a == &b);                       // detect-once, cached
#ifdef __APPLE__
    // This repo's suite runs on Apple Silicon (environment-dependent by
    // design — the host's job is to detect THIS machine).
    CHECK(a.kind == CALIPER_DEV_METAL);
    CHECK_FALSE(a.name.empty());
    CHECK(a.free_memory_hint > 0);
#else
    // Non-Apple: CUDA when an NVIDIA driver is present (Phase 4 detection via
    // the runtime-loaded driver API), CPU otherwise. Both are sane here.
    CHECK((a.kind == CALIPER_DEV_CPU || a.kind == CALIPER_DEV_CUDA));
    if (a.kind == CALIPER_DEV_CUDA) CHECK_FALSE(a.name.empty());
#endif
    CHECK(a.index == 0);
}
