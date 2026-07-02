#include "crash_guard.h"

#ifdef _WIN32
#include <windows.h>
#include <cstdio>

namespace caliper_host {
namespace {
// SEH needs a frame without C++ objects requiring unwinding.
int seh_invoke(const std::function<void()>* fn, unsigned long* code) {
    __try {
        (*fn)();
        return 0;
    } __except (EXCEPTION_EXECUTE_HANDLER) {
        *code = GetExceptionCode();
        return 1;
    }
}
} // namespace

GuardResult guarded_call(const std::function<void()>& fn) {
    GuardResult r;
    unsigned long code = 0;
    if (seh_invoke(&fn, &code)) {
        r.ok = false;
        char buf[64];
        snprintf(buf, sizeof buf, "SEH exception 0x%08lX", code);
        r.fault = buf;
    }
    return r;
}
} // namespace caliper_host

#else // POSIX

#include <csetjmp>
#include <csignal>

namespace caliper_host {
namespace {

thread_local sigjmp_buf t_jmp;
thread_local volatile sig_atomic_t t_active = 0;
thread_local volatile int t_signal = 0;

void fault_handler(int sig) {
    if (t_active) {
        t_signal = sig;
        siglongjmp(t_jmp, 1);
    }
    // Fault outside a guarded region: restore default and re-raise.
    std::signal(sig, SIG_DFL);
    std::raise(sig);
}

constexpr int kSignals[] = {SIGSEGV, SIGBUS, SIGFPE, SIGILL};
constexpr int kNumSignals = 4;

const char* describe(int sig) {
    switch (sig) {
        case SIGSEGV: return "SIGSEGV (invalid memory access)";
        case SIGBUS:  return "SIGBUS (bad memory alignment/mapping)";
        case SIGFPE:  return "SIGFPE (arithmetic fault)";
        case SIGILL:  return "SIGILL (illegal instruction)";
        default:      return "signal";
    }
}

} // namespace

GuardResult guarded_call(const std::function<void()>& fn) {
    struct sigaction sa {}, saved[kNumSignals];
    sa.sa_handler = fault_handler;
    sigemptyset(&sa.sa_mask);
    // NODEFER is load-bearing: it keeps the signal unmasked in the handler,
    // so the out-of-guard re-raise path delivers synchronously to SIG_DFL.
    sa.sa_flags = SA_NODEFER;
    for (int i = 0; i < kNumSignals; i++)
        sigaction(kSignals[i], &sa, &saved[i]);

    GuardResult r;
    t_active = 1;
    if (sigsetjmp(t_jmp, 1) == 0) {
        fn();
    } else {
        r.ok = false;
        r.fault = describe(t_signal);
    }
    t_active = 0;

    for (int i = 0; i < kNumSignals; i++)
        sigaction(kSignals[i], &saved[i], nullptr);
    return r;
}

} // namespace caliper_host
#endif
