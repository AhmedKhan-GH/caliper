#pragma once
/* caliper.data.v1 — SQL over the host's embedded analytical store, results
 * out as Arrow C streams (PLATFORM.md §7.7). Datasets become named, shared,
 * queryable resources instead of per-applet private downloads. IMMUTABLE once
 * published. Callable from applet job threads; the host serializes
 * internally.
 *
 * Stream ownership: on success the host fills *out with a live
 * ArrowArrayStream; the CALLER drains it (get_schema/get_next) and MUST call
 * out->release(out) exactly once when done. On failure (false), *out is
 * untouched and last_error() describes why. last_error() returns a host-owned
 * string valid until the next data.v1 call on the same thread. */
#include <stdint.h>
#include <stdbool.h>
#include <caliper/arrow_c.h>

#define CALIPER_DATA_V1 "caliper.data.v1"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct CaliperDataV1 {
    uint32_t struct_size;
    /* Run SQL against the host store; results stream out as Arrow. */
    bool (*query)(const char* sql_utf8, struct ArrowArrayStream* out);
    /* Name a dataset: uri may be a parquet/csv path or an existing table
       name. Re-registering a name replaces it. */
    bool (*register_dataset)(const char* name, const char* uri);
    /* Open a registered dataset as a full-table stream. */
    bool (*open_dataset)(const char* name, struct ArrowArrayStream* out);
    /* Why the last call on this thread returned false; never NULL. */
    const char* (*last_error)(void);
} CaliperDataV1;

#ifdef __cplusplus
}
#endif
