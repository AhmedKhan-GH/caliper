// live_views_metal.mm: the MTLBuffer query that the heatmap's hover and the
// series and histogram views still make and a plain C++ TU cannot (compiled
// with ARC, Apple only; live_views.cpp has the non-Apple stubs). No GPU work
// here: length and contents are CPU properties. The points view reads its
// buffers through the graphics layer's import and mapping (live_points.h).
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

uint64_t now_ns() { return clock_gettime_nsec_np(CLOCK_UPTIME_RAW); }

}  // namespace caliper_live_views
