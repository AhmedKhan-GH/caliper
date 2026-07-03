#pragma once
/* caliper.artifacts.v1 — content-addressed artifact store: the MLflow
 * artifact idea without the server (PLATFORM.md §7.8). Checkpoints/exports
 * are keyed by sha256 (64 hex chars + NUL), deduplicated, and lineage-tracked
 * to the run that produced them. IMMUTABLE once published. Callable from
 * applet job threads; the host serializes internally. */
#include <stdint.h>
#include <stdbool.h>

#define CALIPER_ARTIFACTS_V1 "caliper.artifacts.v1"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct CaliperArtifactsV1 {
    uint32_t struct_size;
    /* Store bytes under a content hash, linked to a run (0 = unlinked).
       Identical bytes dedup to one file. out_digest: 64 hex chars + NUL. */
    bool (*put)(const char* name, const void* bytes, uint64_t len,
                uint64_t run, char out_digest[65]);
    /* Resolve a digest OR name (name -> newest) to a local file path.
       Host-owned string, valid until the next call. */
    const char* (*path_of)(const char* digest_or_name);
    bool (*exists)(const char* digest_or_name);
} CaliperArtifactsV1;

#ifdef __cplusplus
}
#endif
