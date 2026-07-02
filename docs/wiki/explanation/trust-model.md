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

*Status: written at Task 9.*
