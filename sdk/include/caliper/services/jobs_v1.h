#pragma once
/* caliper.jobs.v1 — background compute with progress + cancel (PLATFORM.md
 * §7.5). IMMUTABLE once published: new capability = jobs_v2, alongside.
 *
 * THREADING HONESTY (§15): job functions run on HOST WORKER THREADS as
 * trusted code. The Caliper host runs them under its crash guard — the same
 * best-effort containment as frame(), not a sandbox: a fault or an escaping
 * C++ exception fails the job, and a job submitted from inside an applet
 * entry point (or from another of its jobs) quarantines that applet. After a
 * fault the process is suspect; do not rely on the net. Other hosts (a
 * fixture, an embedder's own) need not guard at all. Cancellation is
 * cooperative: poll cancelled() in your inner loop and return promptly. */
#include <stdint.h>
#include <stdbool.h>

#define CALIPER_JOBS_V1 "caliper.jobs.v1"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct CaliperJobControl CaliperJobControl;
struct CaliperJobControl {
    uint32_t struct_size;
    /* Poll in loops; return promptly when true. */
    bool (*cancelled)(const CaliperJobControl* ctl);
    /* frac in [0,1]; msg_utf8 may be NULL. Surfaced in the host jobs tray. */
    void (*progress)(const CaliperJobControl* ctl, float frac,
                     const char* msg_utf8);
};

/* Runs on a host worker thread. user must outlive the job. */
typedef void (*CaliperJobFn)(void* user, const CaliperJobControl* ctl);

typedef struct CaliperJobsV1 {
    uint32_t struct_size;
    /* Returns a job id; 0 = error (never a valid id). */
    uint64_t (*submit)(const char* label_utf8, CaliperJobFn fn, void* user);
    void     (*request_cancel)(uint64_t job);
    bool     (*is_running)(uint64_t job);
    float    (*progress_of)(uint64_t job);  /* last reported frac; 0 if none */
} CaliperJobsV1;

#ifdef __cplusplus
}
#endif
