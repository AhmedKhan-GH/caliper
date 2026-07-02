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

GuardResult guarded_call(const std::function<void()>& fn);

} // namespace caliper_host
