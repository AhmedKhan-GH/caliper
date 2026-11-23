// The derive half of a piece on a platform without the Metal runtime
// (caliper_producer_derive.h): as producer.cpp, both functions answer
// CALIPER_PRODUCER_UNSUPPORTED with the code "program.unavailable" until a
// Vulkan runtime exists. On Apple, producer.mm defines them.
#include "producer/caliper_producer_derive.h"

#include <cstdlib>
#include <cstring>

namespace {

caliper_producer_status unavailable(caliper_producer_error* error) {
    if (error) {
        error->status = CALIPER_PRODUCER_UNSUPPORTED;
        error->code = "program.unavailable";
        error->line = 0;
        error->column = 0;
        static const char kText[] = "The derive half of a piece needs the Metal runtime, which this platform lacks.";
        error->message = static_cast<char*>(std::malloc(sizeof kText));
        if (error->message) std::memcpy(error->message, kText, sizeof kText);
    }
    return CALIPER_PRODUCER_UNSUPPORTED;
}

}  // namespace

extern "C" caliper_producer_status caliper_producer_compile_derive(caliper_producer*,
                                                                   const caliper_producer_derive_desc*,
                                                                   uint32_t* program, caliper_producer_error* error) {
    if (program) *program = 0;
    return unavailable(error);
}

extern "C" caliper_producer_status caliper_producer_read_resource(caliper_producer*, uint32_t, const char*,
                                                                  caliper_producer_output* out,
                                                                  caliper_producer_error* error) {
    if (out) std::memset(out, 0, sizeof *out);
    return unavailable(error);
}
