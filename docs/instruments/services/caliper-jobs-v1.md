# caliper.jobs.v1

Service id `caliper.jobs.v1` provides background compute with progress and cancel.

## Types

<!-- c-api types sdk/include/caliper/services/jobs_v1.h CaliperJobControl CaliperJobFn level=3 -->

## Calls

<!-- c-api calls sdk/include/caliper/services/jobs_v1.h CaliperJobsV1 -->

## Ids

<!-- c-api ids sdk/include/caliper/services/jobs_v1.h -->

## Semantics

**Threading.** Job functions do not run on the frame thread. You call `submit(label, fn, user)` from `on_frame` or from any other place, and the host runs `fn` on a **host worker thread**. Your frame stays fluid while the work runs. Job functions are **trusted code**. The Caliper host runs each one under its crash guard, the same best-effort containment it uses for `frame()` (see [the trust model](../trust-model.md)). The guard is not a sandbox, and other hosts, such as the fixture host or an embedder's own host, can run jobs with no guard at all. Keep job code disciplined.

**A job that faults or throws fails.** Under the Caliper host, a signal (`SIGSEGV`, `SIGBUS`, `SIGFPE` or `SIGILL`, including a stack overflow) or a C++ exception that escapes `fn` ends that job and nothing else. Then `is_running` returns `false` and `progress_of` keeps the last reported value. The jobs tray marks the job `FAILED` with the reason and keeps it there, across Home, until you clear finished jobs. Activity logs a `job failed` row, and the host prints one line on stderr:

```
[jobs] job #3 "train" failed: SIGSEGV (invalid memory access)
[jobs] job #4 "load" failed: C++ exception: file not found
```

`jobs.v1` has no entry that reports failure, so a failed job looks like a finished job to an instrument. To tell them apart, have `fn` set a flag as its last statement and read that flag.

**Quarantine.** When the host can tell which instrument submitted the job, a fault or exception in the job also quarantines that instrument, as a fault in `frame()` does:

| Job submitted from | On a fault or exception |
|---|---|
| an instrument entry point (`create`, `initialize`, `frame`, `cleanup`, `destroy`), or another job submitted from one | The job fails, and the instrument is quarantined before the host calls it again: at its next `frame()`, or at teardown. Its status reads `crashed in background job "{label}": {fault}` (see [Refusal messages](../../reference/refusal-messages.md#quarantined-messages)). A job that threw leaves the instrument restartable with `⌘R Restart`. A job that took a signal needs a Caliper restart. |
| a thread the instrument started itself | The job fails. The instrument keeps running. |
| an embedder, through `caliper_core_get_service` | The job fails. No instrument is quarantined. |

A `frame()` that already runs when its job faults finishes before the quarantine takes effect. Every shipped instrument submits from `initialize()` or `frame()`, so the host attributes all of their jobs.

!!! warning "Containment, not isolation"
    After a signal, the guard leaves the job's stack with `siglongjmp`, so no destructors run. A lock that the job held stays held, and other instrument threads that wait on it block. A host service can hold its own lock while it reads instrument memory (`metrics.v1` `histogram` and `image`, `artifacts.v1` `put`, `feed.v1` reads and `log.v1`). If you pass a bad pointer to such a service, it faults with that lock held, and the next user of the service blocks. After a fault the process is suspect. Quarantine lets you see the failure and restart. Do not rely on it for safety.

**Cancellation is cooperative.** `request_cancel(job)` sets a flag but does not interrupt your function. Your job must poll `ctl->cancelled(ctl)` in its inner loop and return promptly when it reads true. A well-behaved job honors cancel within **100 ms**. This is a tested guarantee, and the exemplar's teardown relies on it. A job that ignores `cancelled()` is a bug in the instrument, not in the host.

**`user` must outlive the job.** `submit` takes a raw `void* user`, and the worker dereferences it for the whole life of the job. If you pass `this`, as the exemplar does, `this` must stay alive while the job runs. The host destroys your instrument object right after `on_cleanup()` returns. Thus an instrument with a live job must, in `on_cleanup`, request cancel and then wait on `is_running(job)` before it returns. The wait ends before `destroy()` frees the object. The 100 ms cancel guarantee keeps the wait bounded, and a short poll loop of up to 300 ms cannot hang teardown.

**Shutdown is bounded.** Going Home and quitting both cancel every job and then wait up to **5 s** for the workers to exit. The host joins each worker as soon as it returns. At the deadline, the host names each worker that still runs on stderr and detaches it:

```
[jobs] 1 job(s) still running 5000 ms after cancel; detaching: "train" (#3)
```

The host then continues to tear down the instrument. A job that ignores `cancelled()` does not hang the host, but it keeps running against `user` after your instrument object is gone. Honor cancel.

**Finished workers.** The host joins the thread of a finished job on the next `submit` or the next frame, so a long session of short jobs does not accumulate threads. The job stays in the full list of the jobs tray (`+N more`) until you clear it. The strip of the tray counts the job as finished while the strip is open.

**Ids.** `submit` returns a job id. The id `0` is never valid and means the submission failed, for example because a headless host does not provide the service. `is_running` and `progress_of` return `false` and `0` for an unknown or finished id.

The MNIST instrument (`examples/mnist/instrument/`) submits a background training job, polls `cancelled()`, publishes state under a mutex, and waits in teardown as described above.
