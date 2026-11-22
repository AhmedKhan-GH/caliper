#pragma once
/* The Arrow C Data Interface and C Stream Interface. The definitions are vendored verbatim from the
 * Apache Arrow specification (arrow.apache.org/docs/format/CDataInterface.html). The specification invites
 * projects to copy them, and they are frozen upstream. The doc comments are Caliper's.
 * Tabular results cross caliper.data.v1 with these structs, so no Arrow or DuckDB library type
 * enters the frozen ABI.
 *
 * The include guards below are the guards of the specification and not pragma once. Thus this copy
 * coexists with any other vendored copy, for example inside DuckDB or torch, in the same translation unit. */

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#ifndef ARROW_C_DATA_INTERFACE
#define ARROW_C_DATA_INTERFACE

/** Bit flag of `ArrowSchema.flags`. The values of the dictionary are ordered. */
#define ARROW_FLAG_DICTIONARY_ORDERED 1
/** Bit flag of `ArrowSchema.flags`. The field can hold null values. */
#define ARROW_FLAG_NULLABLE 2
/** Bit flag of `ArrowSchema.flags`. The keys of a map type are sorted. */
#define ARROW_FLAG_MAP_KEYS_SORTED 4

/**
 * The type description of one array, which can have child types.
 *
 * The producer fills the struct and sets `release`. The consumer calls `release` exactly once when it no longer needs
 * the struct. A struct whose `release` is `NULL` is released, and the consumer must not use it.
 */
struct ArrowSchema {
  /** The format string of the type, as a null-terminated UTF-8 string. */
  const char* format;
  /** The name of the field, as a null-terminated UTF-8 string, or `NULL` if the type has no name. */
  const char* name;
  /** The binary-encoded key-value metadata of the field, or `NULL` if there is none. */
  const char* metadata;
  /** The bit flags of the field, a combination of the `ARROW_FLAG_*` values. */
  int64_t flags;
  /** The number of child types in `children`. */
  int64_t n_children;
  /** The array of `n_children` pointers to the child types, or `NULL` if `n_children` is 0. */
  struct ArrowSchema** children;
  /** The type of the dictionary values for a dictionary-encoded type, or `NULL` otherwise. */
  struct ArrowSchema* dictionary;

  /**
   * Release the resources of the struct, including the children and the dictionary, and set `release` to `NULL`.
   * The consumer calls it exactly once.
   */
  void (*release)(struct ArrowSchema*);
  /** Opaque data that belongs to the producer. The consumer must not read or change it. */
  void* private_data;
};

/**
 * The data of one array, which can have child arrays.
 *
 * The producer fills the struct and sets `release`. The consumer calls `release` exactly once when it no longer needs
 * the struct. A struct whose `release` is `NULL` is released, and the consumer must not use it.
 */
struct ArrowArray {
  /** The number of logical elements in the array. */
  int64_t length;
  /** The number of null elements, or -1 if the producer did not compute the count. */
  int64_t null_count;
  /** The logical offset of the first element into the buffers, in elements. It is 0 for most arrays. */
  int64_t offset;
  /** The number of physical buffers in `buffers`. It depends on the type of the array. */
  int64_t n_buffers;
  /** The number of child arrays in `children`. */
  int64_t n_children;
  /** The array of `n_buffers` pointers to the buffers. A pointer can be `NULL` when the spec allows it. */
  const void** buffers;
  /** The array of `n_children` pointers to the child arrays, or `NULL` if `n_children` is 0. */
  struct ArrowArray** children;
  /** The dictionary values for a dictionary-encoded array, or `NULL` otherwise. */
  struct ArrowArray* dictionary;

  /**
   * Release the buffers, the children and the dictionary, and set `release` to `NULL`.
   * The consumer calls it exactly once.
   */
  void (*release)(struct ArrowArray*);
  /** Opaque data that belongs to the producer. The consumer must not read or change it. */
  void* private_data;
};

#endif /* ARROW_C_DATA_INTERFACE */

#ifndef ARROW_C_STREAM_INTERFACE
#define ARROW_C_STREAM_INTERFACE

/**
 * A stream of arrays that all have the same type, which the consumer pulls one array at a time.
 *
 * The consumer calls `release` exactly once when it no longer needs the stream. The arrays that `get_next` returns are
 * independent of the stream, and the consumer must release each one.
 */
struct ArrowArrayStream {
  /**
   * Write the type of the stream to `out`, which is the same for all arrays in the stream. Return 0 on success, or an
   * `errno`-compatible error code otherwise.
   */
  int (*get_schema)(struct ArrowArrayStream*, struct ArrowSchema* out);
  /**
   * Write the next array to `out`. If the call succeeds and the array in `out` is released, the stream has ended.
   * Return 0 on success, or an `errno`-compatible error code otherwise.
   */
  int (*get_next)(struct ArrowArrayStream*, struct ArrowArray* out);
  /**
   * Return a null-terminated description of the last error, or `NULL` if no description is available. Call it only
   * after a stream operation failed with a return code that is not 0.
   */
  const char* (*get_last_error)(struct ArrowArrayStream*);

  /**
   * Release the resources of the stream itself and set `release` to `NULL`. The arrays that `get_next` returned
   * are not affected, and the consumer must release each one.
   */
  void (*release)(struct ArrowArrayStream*);
  /** Opaque data that belongs to the producer. The consumer must not read or change it. */
  void* private_data;
};

#endif /* ARROW_C_STREAM_INTERFACE */

#ifdef __cplusplus
}
#endif
