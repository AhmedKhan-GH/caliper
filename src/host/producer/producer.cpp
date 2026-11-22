// The producer runtime on a platform without its Metal runtime (producer.mm)
// (caliper_producer.h, design/gpu-programs.md). A Vulkan runtime with CUDA
// tensors is planned in design/producer-runtime.md; until it exists, each call
// that would run on a device answers CALIPER_PRODUCER_UNSUPPORTED.
#include "producer/caliper_producer.h"

#include <cstdlib>
#include <cstring>

namespace {

caliper_producer_status unsupported(caliper_producer_error* error) {
    if (error) {
        error->status = CALIPER_PRODUCER_UNSUPPORTED;
        error->code = "program.unavailable";
        error->line = 0;
        error->column = 0;
        static const char kText[] = "This platform has no producer runtime yet; it runs on Metal only.";
        error->message = static_cast<char*>(std::malloc(sizeof kText));
        if (error->message) std::memcpy(error->message, kText, sizeof kText);
    }
    return CALIPER_PRODUCER_UNSUPPORTED;
}

}  // namespace

extern "C" {

uint32_t caliper_producer_abi_version(void) { return CALIPER_PRODUCER_ABI_VERSION; }

caliper_producer_status caliper_producer_create(void*, caliper_producer** out, caliper_producer_error* error) {
    if (out) *out = nullptr;
    return unsupported(error);
}

caliper_producer_status caliper_producer_import(caliper_producer*, const caliper_producer_tensor*, uint32_t*,
                                                caliper_producer_error* error) {
    return unsupported(error);
}

void caliper_producer_release(caliper_producer*, uint32_t) {}

caliper_producer_status caliper_producer_compile(caliper_producer*, const caliper_producer_program_desc*, uint32_t*,
                                                 caliper_producer_error* error) {
    return unsupported(error);
}

caliper_producer_status caliper_producer_prepare(void*, const char*, size_t, const char*,
                                                 caliper_producer_error* error) {
    return unsupported(error);
}

caliper_producer_status caliper_producer_rebind(caliper_producer*, uint32_t, uint32_t, uint32_t,
                                                caliper_producer_error* error) {
    return unsupported(error);
}

caliper_producer_status caliper_producer_run(caliper_producer*, const uint32_t*, size_t,
                                             const caliper_producer_frame*, double, caliper_producer_pass*,
                                             caliper_producer_error* error) {
    return unsupported(error);
}

caliper_producer_status caliper_producer_read_output(caliper_producer*, uint32_t, const char*, caliper_producer_output*,
                                                caliper_producer_error* error) {
    return unsupported(error);
}

caliper_producer_status caliper_producer_set_params(caliper_producer*, uint32_t, const caliper_producer_param*, size_t,
                                                    caliper_producer_error* error) {
    return unsupported(error);
}

caliper_producer_status caliper_producer_read_status(caliper_producer*, uint32_t,
                                                        caliper_producer_program_status*) {
    return CALIPER_PRODUCER_UNSUPPORTED;
}

void caliper_producer_remove(caliper_producer*, uint32_t) {}

void caliper_producer_destroy(caliper_producer*) {}

size_t caliper_producer_reaping(void) { return 0; }

void caliper_producer_error_free(caliper_producer_error* error) {
    if (!error) return;
    std::free(error->message);
    error->message = nullptr;
}

}  // extern "C"
