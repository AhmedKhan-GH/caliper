#pragma once
#include <functional>
#include <string>

namespace caliper_host {

// Best-effort containment, not a sandbox (PLATFORM.md §15): after a fault the
// process memory is suspect; callers must quarantine the applet, not retry it.
struct GuardResult {
    bool ok = true;
    std::string fault;   // e.g. "SIGSEGV (invalid memory access)"; "" when ok
};

// PRECONDITIONS:
// - At most one thread may be inside guarded_call at a time: the signal
//   disposition installed via sigaction is process-wide while the jump state
//   is thread-local. The host honors this — applet calls happen only on the
//   UI thread.
// - Never nest guarded_call on one thread: t_jmp is a single buffer, so an
//   inner call silently disarms the outer guard for the rest of its body.
GuardResult guarded_call(const std::function<void()>& fn);

} // namespace caliper_host
