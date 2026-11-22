#pragma once
/* caliper.log.v1 — structured logs into the host console (docs/applets/services/caliper-log-v1.md).
 * Immutable once published: a new capability is log_v2, alongside. */
#include <stdint.h>

/** The id of this service, which `get_service` takes. */
#define CALIPER_LOG_V1 "caliper.log.v1"

#ifdef __cplusplus
extern "C" {
#endif

/** The severity of a log line. The host treats a value outside 0 to 3 as `CALIPER_LOG_INFO`. */
typedef enum CaliperLogLevel {
    /** Detail that helps to debug an applet. */
    CALIPER_LOG_DEBUG = 0,
    /** A normal event of interest. */
    CALIPER_LOG_INFO  = 1,
    /** A problem that the applet can continue after. */
    CALIPER_LOG_WARN  = 2,
    /** A failure of an operation. */
    CALIPER_LOG_ERROR = 3
} CaliperLogLevel;

/** The log service, which sends timestamped lines from an applet to the host console. */
typedef struct CaliperLogV1 {
    /** `sizeof(CaliperLogV1)` as the host built it. */
    uint32_t struct_size;
    /**
     * Write one finished UTF-8 line at `level`. The host does not interpret a format string and adds the trailing
     * newline. Any applet thread can call it.
     */
    void (*log)(CaliperLogLevel level, const char* message_utf8);
} CaliperLogV1;

#ifdef __cplusplus
}
#endif
