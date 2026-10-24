#pragma once
#include <functional>
#include <string>

namespace caliper_host {

// Best-effort containment, not a sandbox (PLATFORM.md §15): after a fault the
// process memory is suspect; callers must quarantine the applet, not retry it.
// Contained: SIGSEGV/SIGBUS/SIGFPE/SIGILL (SEH on Windows), including a stack
// overflow (the handler runs on a per-thread alternate stack), and any C++
// exception escaping fn — reported as "C++ exception: <what()>" on every OS.
struct GuardResult {
    bool ok = true;
    std::string fault;   // e.g. "SIGSEGV (invalid memory access)"; "" when ok
};

// The signal handlers are installed once, process-wide, by the first call and
// chain to whatever was installed before them for faults outside a guarded
// region (or on a thread that never guards), so unguarded crashes still reach
// the default action or an embedder's own reporter. The jump state is
// thread-local, so guarded regions on different threads do not interfere.
//
// PRECONDITION: never nest guarded_call on one thread — t_jmp is a single
// buffer, so an inner call silently disarms the outer guard for the rest of
// its body.
GuardResult guarded_call(const std::function<void()>& fn);

} // namespace caliper_host
