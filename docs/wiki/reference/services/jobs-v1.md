# caliper.jobs.v1

Service id `caliper.jobs.v1` — background compute with progress + cancel (PLATFORM.md §7.5). This page embeds the header verbatim; the docs build fails if the file moves.

```c
--8<-- "sdk/include/caliper/services/jobs_v1.h"
```

## Semantics

**Threading.** Job functions do not run on the frame thread — that is the whole
point of the service. You call `submit(label, fn, user)` from `on_frame` (or
anywhere), and the host runs `fn` on a **host worker thread**. Your frame stays
fluid while the work grinds. The corollary, stated in the header and not
softened: job functions run on host worker threads as **trusted code**, and they
are **not crash-guarded** — the signal guard is UI-thread-only by documented
precondition (see [the trust model](../../explanation/trust-model.md)) — so a
fault inside a job takes the whole process down. Keep job code as disciplined as
you would keep any code that runs without a net.

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

**Ids.** `submit` returns a job id; `0` is never a valid id — it means the
submission failed (for example, a headless host that does not vend the service).
`is_running`/`progress_of` on an unknown or finished id return `false`/`0`.

The canonical consumer is the [MLScope exemplar](../../tutorials/first-applet.md)
(`examples/ml_scope/`): it submits an MLP training loop, polls `cancelled()` each
epoch, publishes loss under a mutex, and does the bounded-wait teardown described
above.
