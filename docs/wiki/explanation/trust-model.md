# Trust model

Caliper runs applets **in-process**: an applet is a shared library the host
`dlopen`s and calls directly. That buys raw performance and a zero-ceremony API
(applets program ImGui/ImPlot directly), but it means an applet's code runs with
the host's privileges, in the host's address space. The trust model is therefore
stated plainly: **in-process applets are trusted code.** What follows is how the
host stays alive and honest when that trust is misplaced — not a claim that it is
sandboxed. Real isolation arrives in Phase 6.

## Pre-flight beats post-mortem

The first line of defence runs before any of the applet's code does. Each applet
ships a `caliper.toml` manifest declaring the ABI epoch it was built against, the
minimum host version it needs, and the services it requires. The host checks that
manifest **before the `dlopen`**. A wrong-build applet — compiled for an ABI
epoch this host doesn't speak, or requiring a capability the host can't vend —
never gets loaded at all. It becomes a friendly card on the landing page ("Built
for ABI epoch 1; this host speaks 2 — check for an applet update."), not a loader
crash three frames later. This gate catches the entire class of "wrong build for
this host" failures cheaply and legibly, before they can turn into undefined
behaviour.

## Every applet call is guarded

Manifests catch the predictable failures. They cannot catch a null dereference in
the applet's frame code. So every call from the host into an applet is wrapped in
a crash guard:

- **POSIX (macOS, Linux):** a signal trampoline. Before the call, the host installs
  handlers for `SIGSEGV`, `SIGBUS`, `SIGFPE`, and `SIGILL` and records a jump
  target with `sigsetjmp`. If the applet faults, the handler `siglongjmp`s back
  out of the applet call, and the guard returns a failure naming the signal
  (e.g. `SIGSEGV (invalid memory access)`). The handlers are then restored to
  whatever they were before, so the guard nests and leaves no global state
  behind. A fault that somehow fires outside a guarded region restores the
  default disposition and re-raises, preserving normal crash semantics.
- **Windows:** the same shape via structured exception handling — the applet call
  sits inside a `__try/__except (EXCEPTION_EXECUTE_HANDLER)` frame, and a hardware
  exception is caught and reported by its exception code. *(This path compiles on
  Windows but has not yet been exercised on a Windows host; the POSIX path is the
  one verified today.)*

The guard is the containment primitive the loader wraps around every applet
entry point — construction, `frame()`, teardown.

## A faulting applet is quarantined

When the guard catches a fault, the host does not retry the call. The faulting
applet is torn down and **quarantined**: its card shows the named fault, and it
is not called again for the rest of the session. The host itself survives and
stays interactive; the failure is contained to the one applet that caused it, and
the host offers a restart.

## Containment, not a sandbox

This is the honest part. The crash guard is **best-effort containment, not
isolation.** By the time a signal fires, the applet has already run arbitrary
code in the host's address space. It may have corrupted shared heap state, left a
lock held, or scribbled on memory the host still relies on. `siglongjmp` unwinds
the stack but runs no C++ destructors, so anything the applet owned is leaked or
left half-torn-down. That is exactly why a quarantined applet is never resumed
and why the host treats its own memory as suspect afterward and offers a restart
rather than pretending nothing happened.

Real isolation — where a misbehaving applet genuinely *cannot* corrupt the host —
requires an address-space boundary, i.e. running applets out-of-process. That is
**Phase 6**. Until then, the trust model is: manifest-gated loading, guarded
calls, visible quarantine, and no false promises.

## Frame watchdog

The crash guard catches applets that fault; the watchdog catches applets that
merely misbehave. The host owns the frame clock and calls each applet's `frame()`
once per frame, so a `frame()` that does heavy work blocks the whole UI thread —
the window stops responding until it returns. The watchdog makes that visible: it
times each `frame()`, and when one exceeds the budget (about 250 ms) for three
consecutive frames, it flags the applet's card with a plain note that long work
belongs in `caliper.jobs` rather than the frame loop (PLATFORM.md §15). The flag
**latches** — it stays raised until the applet is relaunched, and a later fast
frame does *not* clear it — so the evidence of a stall doesn't scroll away the
instant the applet recovers. The point is observability, not punishment: the
applet keeps running and nothing is torn down, but the framework's threading rule
("keep `frame()` cheap; push slow work to jobs") becomes something you can see
instead of a convention you have to remember.

## Jobs run unguarded

The watchdog tells you to push slow work to `caliper.jobs.v1` — so it is worth
being just as plain about what that thread does *not* get. The crash guard above
is **UI-thread-only by documented precondition**: the signal trampoline is armed
around the host's calls into applet `frame()`/`init()`/`cleanup()`, all of which
run on the frame thread. A job function submitted to `caliper.jobs.v1` runs on a
**host worker thread**, outside that trampoline, and is therefore **not
crash-guarded at all.** A null dereference or a bad tensor op inside a job does
not `siglongjmp` back into a quarantine — it takes the whole process down, host
and all.

This is not an oversight to be fixed later; it is the same trust boundary stated
from the other side. In-process applets are trusted code, and a worker thread is
simply that trust without even the best-effort net the UI thread gets. The
practical rules that follow are small and non-negotiable: keep job code as
careful as anything that runs without a net; poll `cancelled()` so the host can
stop it within the ≤ 100 ms contract; and make the object you pass as `user`
outlive the job (cancel-and-bounded-wait in `on_cleanup`). Real isolation for
worker code, like everything else here, waits for the out-of-process boundary in
Phase 6.
