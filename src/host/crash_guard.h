#pragma once
#include <functional>
#include <string>

namespace caliper_host {

// Best-effort containment, not a sandbox (PLATFORM.md §15): after a fault the
// process memory is suspect; callers must quarantine the applet, not retry it.
// Contained: SIGSEGV/SIGBUS/SIGFPE/SIGILL (SEH on Windows), including a stack
// overflow (the handler runs on a per-thread alternate stack), and any C++
// exception escaping fn — reported as "C++ exception: <what()>" on every OS.

// What a failed guarded call was, which decides whether the applet may be
// restarted in this process (AppletLoader::restart):
//   Exception  a C++ exception escaped fn. The stack unwound normally and
//              every destructor on the way ran, so the applet's objects are
//              as sound as its own exception safety left them, and the
//              process is unharmed.
//   Signal     a memory fault, stack overflow or illegal instruction (any SEH
//              exception on Windows, including a C++ throw MSVC could not
//              unwind through the C ABI). The jump out skipped every
//              destructor and memory may be corrupt: nothing of that applet
//              runs again until the host itself restarts.
enum class FaultKind { None, Exception, Signal };

struct GuardResult {
    bool ok = true;
    std::string fault;   // e.g. "SIGSEGV (invalid memory access)"; "" when ok
    FaultKind kind = FaultKind::None;
};

// The kind of a fault known only by its text: a reason that was passed along
// as a string, such as a background job's (JobOwner::fault(), "background job
// \"<label>\": <guard fault>"). Errs toward Signal: Exception only when the
// text names a C++ exception and does not end in one of the guard's own
// signal or SEH texts, so no label or what() can make a signal read as an
// exception. Prefer GuardResult::kind wherever the result itself is at hand.
FaultKind fault_kind_of(const std::string& fault);

// The signal handlers are installed once, process-wide, by the first call and
// chain to whatever was installed before them for faults outside a guarded
// region (or on a thread that never guards), so unguarded crashes still reach
// the default action or an embedder's own reporter. The jump state is
// thread-local, so guarded regions on different threads do not interfere: the
// frame thread guards applet entry points while job workers guard applet job
// fns (JobSystem::submit_guarded), concurrently.
//
// PRECONDITION: never nest guarded_call on one thread — t_jmp is a single
// buffer, so an inner call silently disarms the outer guard for the rest of
// its body.
GuardResult guarded_call(const std::function<void()>& fn);

} // namespace caliper_host
