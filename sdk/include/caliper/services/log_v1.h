#pragma once
/* caliper.log.v1 — structured logs into the host console (PLATFORM.md §7.1).
 * IMMUTABLE once published: new capability = log_v2, alongside. */
#include <stdint.h>

#define CALIPER_LOG_V1 "caliper.log.v1"

#ifdef __cplusplus
extern "C" {
#endif

typedef enum CaliperLogLevel {
    CALIPER_LOG_DEBUG = 0,
    CALIPER_LOG_INFO  = 1,
    CALIPER_LOG_WARN  = 2,
    CALIPER_LOG_ERROR = 3
} CaliperLogLevel;

typedef struct CaliperLogV1 {
    uint32_t struct_size;
    void (*log)(CaliperLogLevel level, const char* message_utf8); /* pre-formatted */
} CaliperLogV1;

#ifdef __cplusplus
}
#endif
