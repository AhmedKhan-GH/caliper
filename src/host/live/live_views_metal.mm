// live_views_metal.mm: the MTLBuffer queries the typed live views need and a
// plain C++ TU cannot make (compiled with ARC, Apple only; live_views.cpp has
// the non-Apple stubs). No GPU work here: length/contents are CPU properties.
#import <Metal/Metal.h>

#include <time.h>

#include "live/live_views_state.h"

namespace caliper_live_views {

BufferInfo metal_buffer_info(void* mtl_buffer) {
    BufferInfo bi;
    if (mtl_buffer == nullptr) return bi;
    id<MTLBuffer> b = (__bridge id<MTLBuffer>)mtl_buffer;
    bi.ok = true;
    bi.length = (uint64_t)b.length;
    // Private storage has no CPU mapping; contents is NULL there by contract.
    bi.contents = b.storageMode == MTLStorageModePrivate ? nullptr : b.contents;
    return bi;
}

void* metal_new_shared_buffer(void* mtl_device, const void* bytes, uint64_t len) {
    if (mtl_device == nullptr || bytes == nullptr || len == 0) return nullptr;
    id<MTLDevice> dev = (__bridge id<MTLDevice>)mtl_device;
    id<MTLBuffer> b = [dev newBufferWithBytes:bytes
                                       length:(NSUInteger)len
                                      options:MTLResourceStorageModeShared];
    return b != nil ? (__bridge_retained void*)b : nullptr;
}

void metal_release(void* obj) {
    if (obj != nullptr) {
        id o = (__bridge_transfer id)obj;
        (void)o;
    }
}

uint64_t now_ns() { return clock_gettime_nsec_np(CLOCK_UPTIME_RAW); }

}  // namespace caliper_live_views
