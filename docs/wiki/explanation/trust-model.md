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
never gets loaded at all. It is listed on the landing screen as `REFUSED`, with
the reason in the Inspector ("Built for ABI epoch 1; this host speaks 2 — check
for an applet update."), not a loader crash three frames later. This gate
catches the entire class of "wrong build for this host" failures cheaply and
legibly, before they can turn into undefined behaviour.

Two checks back the manifest up. The applet's `id` names its data folder, so the
parser accepts only path-safe ids (`[a-z0-9.-]`, see
[Manifest](../reference/manifest.md)) and the loader checks again before it
builds the path: a crafted id such as `../../x` cannot reach outside the data
root. And after the `dlopen`, the loader checks the binary's own list of
required services against the host, so a manifest that under-declares cannot
start an applet that will call a service the host does not have.

## Every applet call is guarded

Manifests catch the predictable failures. They cannot catch a null dereference in
the applet's frame code. So every call from the host into an applet is wrapped in
a crash guard:

- **POSIX (macOS, Linux):** a signal trampoline. The first guarded call installs
  handlers for `SIGSEGV`, `SIGBUS`, `SIGFPE`, and `SIGILL`, once for the whole
  process. Before each call the guard records a jump target with `sigsetjmp`; if
  the applet faults, the handler `siglongjmp`s back out of the applet call, and
  the guard returns a failure naming the signal (e.g.
  `SIGSEGV (invalid memory access)`). The handler runs on a per-thread alternate
  signal stack, so a **stack overflow** is contained too. The jump state is per
  thread, so guarded calls on different threads do not interfere.
- **Chaining.** A fault outside a guarded region, or on a thread that never
  guards, is not the guard's to contain: it restores the handler that was
  installed before it and re-raises, so the fault reaches the default action or
  an embedder's own crash reporter exactly as if the guard were absent. If
  another library replaces the handlers after the guard installed them (a test
  harness, Python's `faulthandler`), the guard re-arms them around each applet
  call, passes faults outside the call on to that library, and puts its handlers
  back afterwards. It says so once on stderr:
  `[guard] another library replaced the SIGSEGV handler; re-arming the guard around each applet call`.
- **C++ exceptions:** an exception escaping an applet entry point is caught and
  treated like a signal, reported as `C++ exception: <what()>`, or
  `C++ exception (non-std type)` for anything not derived from `std::exception`.
  Applets built with the `CALIPER_APPLET` macro already catch at the ABI
  boundary; this catches what a hand-written applet lets through.
- **Windows:** the same shape via structured exception handling — the applet call
  sits inside a `__try/__except (EXCEPTION_EXECUTE_HANDLER)` frame, and a hardware
  exception is caught and reported by its exception code; a C++ exception is
  caught in its own frame beside it and reported by name. *(This path compiles on
  Windows but has not yet been exercised on a Windows host; the POSIX path is the
  one verified today. Under `/EHsc`, MSVC may report a C++ throw by its SEH code
  `0xE06D7363` instead of by name; it is still quarantined.)*

The guard is the containment primitive the loader wraps around every applet
entry point — construction, `frame()`, teardown — and the host wraps around every
job an applet submits through `caliper.jobs.v1` (see
[Jobs run under the guard](#jobs-run-under-the-guard)).

## A faulting applet is quarantined

When the guard catches a fault, the host does not retry the call. The faulting
applet is torn down and **quarantined**: the host returns to the landing screen,
where the applet's row reads `QUARANTINED` and the Inspector names the fault,
and it is not called again for the rest of the session. The host itself
survives and stays interactive; the failure is contained to the one applet that
caused it, and the host offers a restart.

## Containment, not a sandbox

This is the honest part. The crash guard is **best-effort containment, not
isolation.** By the time a signal fires, the applet has already run arbitrary
code in the host's address space. It may have corrupted shared heap state, left a
lock held, or scribbled on memory the host still relies on. `siglongjmp` unwinds
the stack but runs no C++ destructors, so anything the applet owned is leaked or
left half-torn-down. A C++ exception does run destructors as it unwinds, but it
still leaves the applet's own state wherever the throw interrupted it. That is
exactly why a quarantined applet is never resumed
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
consecutive frames, it puts a plain note in the applet's bar at the top of its
page — `slow: long work belongs in background jobs` — pointing long work to
`caliper.jobs` rather than the frame loop (PLATFORM.md §15). The flag
**latches** — it stays raised until the applet is relaunched, and a later fast
frame does *not* clear it — so the evidence of a stall doesn't scroll away the
instant the applet recovers. The point is observability, not punishment: the
applet keeps running and nothing is torn down, but the framework's threading rule
("keep `frame()` cheap; push slow work to jobs") becomes something you can see
instead of a convention you have to remember.

## Jobs run under the guard

The watchdog tells you to push slow work to `caliper.jobs.v1`, so the worker
threads get the same net as the frame thread. The Caliper host runs every job
function submitted through `caliper.jobs.v1` inside the crash guard, on its
worker thread. A null dereference, a bad tensor op or an escaping exception
inside a job fails that job with the guard's reason
(`failed: SIGSEGV (invalid memory access)` in the jobs tray and on stderr), and
the host carries on.

A failed job also quarantines the applet that submitted it, when the host can
tell which applet that was. The service table is shared by every applet and
`submit` carries no applet identity, so the host tracks the submitter by
thread: around each call into an applet (`create`, `initialize`, `frame`,
`cleanup`, `destroy`) the loader marks that applet as the owner of any job
submitted during the call, and a job passes its owner on to the jobs it submits.
When an owned job faults, the worker only records the fault. The loader
quarantines the applet on its own thread before calling it again — at its next
`frame()`, or at teardown — with the status
`crashed in background job "train": SIGSEGV (invalid memory access)`. A job
submitted from a thread the applet started itself has no owner: it fails, and
the applet keeps running.

The limits of the section above apply with more force on a worker. The jump
out of a job that took a signal runs no destructors, so a lock the job held
stays held, and a bad pointer passed to a host service that reads applet memory
under its own lock
(`metrics.v1` `histogram` and `image`, `artifacts.v1` `put`, `feed.v1` reads,
`log.v1`) leaves that lock held: the next user of the service blocks, and a
crash becomes a hang. This is the Caliper host's behaviour; `jobs_v1.h` lets
other hosts, such as the fixture host or an embedder's own, run jobs with no
guard at all. So the practical rules stay as they were: keep job code as careful
as anything that runs without a net; poll `cancelled()` so the host can stop it
within the ≤ 100 ms contract (Home and quit wait at most 5 s, then detach the
worker); and make the object you pass as `user` outlive the job
(cancel-and-bounded-wait in `on_cleanup`). Real isolation for worker code, like
everything else here, waits for the out-of-process boundary in Phase 6.
