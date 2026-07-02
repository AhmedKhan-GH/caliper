#include "device_query.h"
#import <Metal/Metal.h>

namespace caliper_host {

const DeviceInfo& device_info() {
    static const DeviceInfo info = [] {
        DeviceInfo d;
        @autoreleasepool {
            id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
            if (dev) {
                d.kind = CALIPER_DEV_METAL;
                d.index = 0;
                d.name = [[dev name] UTF8String];
                d.free_memory_hint = [dev recommendedMaxWorkingSetSize];
            }
        }
        return d;
    }();
    return info;
}

} // namespace caliper_host
