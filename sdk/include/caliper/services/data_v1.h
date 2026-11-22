#pragma once
/* caliper.data.v1 — SQL over the host's embedded analytical store, results out as Arrow C streams
 * (docs/applets/services/caliper-data-v1.md). Immutable once published. Callable from applet job threads. */
#include <stdint.h>
#include <stdbool.h>
#include <caliper/arrow_c.h>

/** The id of this service, which `get_service` takes. */
#define CALIPER_DATA_V1 "caliper.data.v1"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * SQL access to the host's analytical store.
 *
 * Datasets are named, shared resources, and results leave the host as Arrow C streams. An applet can call every
 * entry from a job thread, and the host serializes the calls internally.
 *
 * On success, `query` and `open_dataset` fill `*out` with a live `ArrowArrayStream`. The caller drains the stream
 * with `get_schema` and `get_next`, and must call `out->release(out)` exactly once when it is done. On failure the
 * call returns `false`, leaves `*out` untouched and sets the error that `last_error` returns.
 */
typedef struct CaliperDataV1 {
    /** `sizeof(CaliperDataV1)` as the host built it. */
    uint32_t struct_size;
    /**
     * Run one SQL statement against the host store and stream the result as Arrow. A statement that returns no
     * rows, such as DDL or `INSERT`, succeeds with an empty stream. On failure, return `false`.
     */
    bool (*query)(const char* sql_utf8, struct ArrowArrayStream* out);
    /**
     * Name a dataset. `uri` is a path to a parquet or csv file, or the name of an existing table. The host splices
     * `name` into SQL, so it must be a SQL identifier, and any other name returns `false`. Registering a name
     * again replaces the dataset.
     */
    bool (*register_dataset)(const char* name, const char* uri);
    /** Stream a registered dataset as a full-table Arrow stream, which is the same as `SELECT *` over `query`. */
    bool (*open_dataset)(const char* name, struct ArrowArrayStream* out);
    /**
     * Return why the last call on this thread returned `false`, or an empty string if no call failed. The host
     * owns the string, which is never `NULL` and stays valid until the next call of this service on the same
     * thread.
     */
    const char* (*last_error)(void);
} CaliperDataV1;

#ifdef __cplusplus
}
#endif
