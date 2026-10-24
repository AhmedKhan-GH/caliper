# caliper.jobs.v1

Service id `caliper.jobs.v1` — background compute with progress + cancel (PLATFORM.md §7.5). This page embeds the header verbatim; the docs build fails if the file moves.

```c
--8<-- "sdk/include/caliper/services/jobs_v1.h"
```

## Semantics

**Threading.** Job functions do not run on the frame thread — that is the whole
point of the service. You call `submit(label, fn, user)` from `on_frame` (or
anywhere), and the host runs `fn` on a **host worker thread**. Your frame stays
fluid while the work grinds. Job functions are **trusted code**. The Caliper host
runs each one under its crash guard, the same best-effort containment it puts
around `frame()` (see [the trust model](../../explanation/trust-model.md)); it is
not a sandbox, and the header lets other hosts (the fixture host, an embedder's
own) run jobs with no guard at all. Keep job code as disciplined as you would
keep any code that runs without a net.

**A job that faults or throws fails.** Under the Caliper host, a signal
(`SIGSEGV`, `SIGBUS`, `SIGFPE`, `SIGILL`, including a stack overflow) or a C++
exception escaping `fn` ends that job and nothing else: `is_running` turns
`false`, `progress_of` keeps the last reported value, the jobs tray marks the
job `FAILED` with the reason (and keeps it there, across Home, until finished
jobs are cleared), Activity logs a `job failed` row, and the host prints one
line on stderr:

```
[jobs] job #3 "train" failed: SIGSEGV (invalid memory access)
[jobs] job #4 "load" failed: C++ exception: file not found
```

`jobs.v1` has no entry that reports failure, so to an applet a failed job looks
like a finished one. If you need to tell them apart, have `fn` record its own
completion (a flag set as its last statement) and read that.

**Quarantine.** When the host can tell which applet submitted the job, a fault
or exception in it also quarantines that applet, exactly as a fault in `frame()`
does:

| Job submitted from | On a fault or exception |
|---|---|
| an applet entry point (`create`, `initialize`, `frame`, `cleanup`, `destroy`), or another job submitted from one | The job fails, and the applet is quarantined before the host calls it again: at its next `frame()`, or at teardown. Its status reads `crashed in background job "{label}": {fault}` (see [Refusal messages](../refusals.md#quarantined-messages)). A job that threw leaves the applet restartable with `⌘R Restart`; a job that took a signal needs a Caliper restart. |
| a thread the applet started itself | The job fails. The applet keeps running. |
| an embedder, through `caliper_core_get_service` | The job fails. No applet is quarantined. |

A `frame()` already running when its job faults finishes before the quarantine
lands. Every shipped applet submits from `initialize()` or `frame()`, so its
jobs are attributed.

!!! warning "Containment, not isolation"
    After a signal, the guard leaves the job's stack with `siglongjmp`, so no
    destructors run: a lock the job held stays held, and other applet threads
    waiting on it block. A bad pointer passed to a host service that holds its
    own lock while it reads applet memory (`metrics.v1` `histogram` and `image`,
    `artifacts.v1` `put`, `feed.v1` reads, `log.v1`) faults with that lock
    held, and the next user of the service blocks. After a fault the process is
    suspect; quarantine is there so you can see the failure and restart, not so
    you can rely on it.

**Cancellation is cooperative.** `request_cancel(job)` sets a flag; it does not
interrupt your function. Your job must poll `ctl->cancelled(ctl)` in its inner
loop and return promptly when it reads true. The framework contract (PLATFORM.md
§16) is that a well-behaved job honours cancel within **≤ 100 ms** — this is a
*tested guarantee*, not advice, and the exemplar's teardown relies on it (below).
A job that ignores `cancelled()` and runs for a minute is a bug in the applet,
not the host.

**`user` must outlive the job.** `submit` takes a raw `void* user` and the
worker dereferences it for the job's entire lifetime. If you pass `this` (the
common case — see the exemplar), then `this` must not be destroyed while the job
is still running. Because the host destroys your applet object right after
`on_cleanup()` returns, an applet with a live job **must, in `on_cleanup`,
request cancel and then bounded-wait on `is_running(job)`** before returning — so
the worker has exited before `destroy()` frees the object out from under it. The
≤ 100 ms cancel contract is what makes that wait bounded: a short poll loop
(e.g. up to 300 ms) cannot hang teardown.

**Shutdown is bounded.** Going Home and quitting both cancel every job and then
wait up to **5 s** for the workers to exit, joining each one as soon as it
returns. A worker still running at the deadline is named on stderr
(`[jobs] 1 job(s) still running 5000 ms after cancel; detaching: "train" (#3)`)
and detached, and the host carries on tearing the applet down. A job that
ignores `cancelled()` therefore no longer hangs the host, but it keeps running
against `user` after your applet object is gone — honour cancel.

**Finished workers are joined promptly.** The host joins a finished job's thread
on the next `submit` or the next frame, so a long session of short jobs does not
accumulate threads. The job stays listed in the tray until
it is cleared.

**Ids.** `submit` returns a job id; `0` is never a valid id — it means the
submission failed (for example, a headless host that does not vend the service).
`is_running`/`progress_of` on an unknown or finished id return `false`/`0`.

The canonical consumer is the [MLScope exemplar](../../tutorials/first-applet.md)
(`examples/ml_scope/`): it submits an MLP training loop, polls `cancelled()` each
epoch, publishes loss under a mutex, and does the bounded-wait teardown described
above.
