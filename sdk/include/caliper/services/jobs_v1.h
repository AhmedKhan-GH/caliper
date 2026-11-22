#pragma once
/* caliper.jobs.v1 — background compute with progress and cancel (docs/applets/services/caliper-jobs-v1.md).
 * Immutable once published: a new capability is jobs_v2, alongside.
 *
 * Job functions run on host worker threads as trusted code. The Caliper host runs them under its crash guard,
 * which is best-effort containment like the guard of frame() and not a sandbox. After a fault the process is
 * suspect. Cancellation is cooperative: poll cancelled() in the inner loop and return promptly. */
#include <stdint.h>
#include <stdbool.h>

/** The id of this service, which `get_service` takes. */
#define CALIPER_JOBS_V1 "caliper.jobs.v1"

#ifdef __cplusplus
extern "C" {
#endif

/** The forward declaration of the control handle that the host passes to a running job. */
typedef struct CaliperJobControl CaliperJobControl;
/**
 * The control handle that the host passes to a running job.
 *
 * The job uses it to poll for cancellation and to report progress.
 */
struct CaliperJobControl {
    /** `sizeof(CaliperJobControl)` as the host built it. */
    uint32_t struct_size;
    /** Return `true` after a cancel request. Poll it in loops and return promptly when it is true. */
    bool (*cancelled)(const CaliperJobControl* ctl);
    /**
     * Report the progress of the job as `frac` in the range 0 to 1, with an optional UTF-8 message that can be
     * `NULL`. The host shows both in its jobs tray.
     */
    void (*progress)(const CaliperJobControl* ctl, float frac,
                     const char* msg_utf8);
};

/** The type of a job function. It runs on a host worker thread, and `user` must outlive the job. */
typedef void (*CaliperJobFn)(void* user, const CaliperJobControl* ctl);

/** The job service, which runs functions on host worker threads. */
typedef struct CaliperJobsV1 {
    /** `sizeof(CaliperJobsV1)` as the host built it. */
    uint32_t struct_size;
    /**
     * Start `fn` on a host worker thread with the argument `user`, and show `label_utf8` in the jobs tray. Return
     * the job id, or 0 on error. 0 is never a valid id.
     */
    uint64_t (*submit)(const char* label_utf8, CaliperJobFn fn, void* user);
    /** Ask a job to stop. This only sets a flag, and the job must poll `cancelled` to see it. */
    void     (*request_cancel)(uint64_t job);
    /** Return `true` while a job runs, and `false` for a finished or unknown id. */
    bool     (*is_running)(uint64_t job);
    /** Return the last fraction that the job reported, or 0 if it reported none or the id is unknown. */
    float    (*progress_of)(uint64_t job);
} CaliperJobsV1;

#ifdef __cplusplus
}
#endif
