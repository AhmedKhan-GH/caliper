#pragma once
/* caliper.artifacts.v1 — content-addressed artifact store (docs/instruments/services/caliper-artifacts-v1.md).
 * Immutable once published. Callable from instrument job threads. */
#include <stdint.h>
#include <stdbool.h>

/** The id of this service, which `get_service` takes. */
#define CALIPER_ARTIFACTS_V1 "caliper.artifacts.v1"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * The content-addressed artifact store.
 *
 * The store keeps checkpoints and exports under the sha256 digest of their bytes, deduplicates them, and links each
 * one to the run that produced it. Artifacts are immutable once published. An instrument can call every entry from a
 * job thread, and the host serializes the calls internally.
 */
typedef struct CaliperArtifactsV1 {
    /** `sizeof(CaliperArtifactsV1)` as the host built it. */
    uint32_t struct_size;
    /**
     * Store `len` bytes under their sha256 digest, record `name`, `run`, `len` and a timestamp in the index, and
     * return `true`. Pass a metrics run id as `run`, or 0 for an unlinked artifact. Identical bytes deduplicate to
     * one file. The call writes the digest, 64 hex characters and a terminating NUL, into `out_digest`.
     */
    bool (*put)(const char* name, const void* bytes, uint64_t len,
                uint64_t run, char out_digest[65]);
    /**
     * Resolve a digest or a name to the path of a local file, or return `NULL` if the artifact is unknown. A name
     * resolves to the newest artifact with that name. The host owns the string, and it stays valid until the next
     * call of this service on the same thread.
     */
    const char* (*path_of)(const char* digest_or_name);
    /** Return `true` if the store knows a digest or a name. A name matches its newest artifact. */
    bool (*exists)(const char* digest_or_name);
} CaliperArtifactsV1;

#ifdef __cplusplus
}
#endif
