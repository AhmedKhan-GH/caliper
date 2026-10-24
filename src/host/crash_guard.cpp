#include "crash_guard.h"
#include <exception>

#ifdef _WIN32
#include <windows.h>
#include <cstdio>

namespace caliper_host {
namespace {
// A C++ throw is itself an SEH exception (0xE06D7363); caught in its own
// frame it is reported by name, and the applet quarantines exactly as it
// would for a signal on POSIX. (SEH __try cannot share a function with C++
// try, hence the split.) Returns 0 ok, 1 threw.
int cxx_invoke(const std::function<void()>* fn, std::string* fault) {
    try {
        (*fn)();
        return 0;
    } catch (const std::exception& e) {
        *fault = std::string("C++ exception: ") + e.what();
        return 1;
    } catch (...) {
        *fault = "C++ exception (non-std type)";
        return 1;
    }
}

// SEH needs a frame without C++ objects requiring unwinding. Returns 0 ok,
// 1 C++ exception (fault set), 2 SEH fault (code set).
int seh_invoke(const std::function<void()>* fn, std::string* fault,
               unsigned long* code) {
    __try {
        return cxx_invoke(fn, fault);
    } __except (EXCEPTION_EXECUTE_HANDLER) {
        *code = GetExceptionCode();
        return 2;
    }
}
} // namespace

GuardResult guarded_call(const std::function<void()>& fn) {
    GuardResult r;
    unsigned long code = 0;
    std::string fault;
    const int rc = seh_invoke(&fn, &fault, &code);
    if (rc == 1) {
        r.ok = false;
        r.fault = fault;
    } else if (rc == 2) {
        r.ok = false;
        char buf[64];
        snprintf(buf, sizeof buf, "SEH exception 0x%08lX", code);
        r.fault = buf;
    }
    return r;
}
} // namespace caliper_host

#else // POSIX

#include <algorithm>
#include <csetjmp>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <mutex>

namespace caliper_host {
namespace {

thread_local sigjmp_buf t_jmp;
thread_local volatile sig_atomic_t t_active = 0;
thread_local volatile int t_signal = 0;

constexpr int kSignals[] = {SIGSEGV, SIGBUS, SIGFPE, SIGILL};
constexpr int kNumSignals = 4;
// SIGINFO: the three-argument handler. NODEFER is load-bearing: it keeps the
// signal unmasked in the handler, so the out-of-guard re-raise below delivers
// synchronously to the previous disposition. ONSTACK runs the handler on the
// thread's alternate stack — the only stack left after an overflow.
constexpr int kFlags = SA_SIGINFO | SA_NODEFER | SA_ONSTACK;

// Chain targets for faults we don't own, per kSignals entry: what was there
// when the guard installed itself (base), or — while a call has had to
// re-arm over a later installer — that installer (see Rearm).
struct sigaction g_base_previous[kNumSignals];
struct sigaction g_previous[kNumSignals];

const char* describe(int sig) {
    switch (sig) {
        case SIGSEGV: return "SIGSEGV (invalid memory access)";
        case SIGBUS:  return "SIGBUS (bad memory alignment/mapping)";
        case SIGFPE:  return "SIGFPE (arithmetic fault)";
        case SIGILL:  return "SIGILL (illegal instruction)";
        default:      return "signal";
    }
}

void fault_handler(int sig, siginfo_t*, void*) {
    if (t_active) {
        t_signal = sig;
        siglongjmp(t_jmp, 1);
    }
    // Fault outside a guarded region, or on a thread that never guards: not
    // ours to contain. Put the previous disposition back and re-raise, so it
    // lands there exactly as if the guard had never been installed — the
    // default action (terminate), or an embedder's own crash reporter. A
    // fault can't meaningfully be ignored, so SIG_IGN is treated as default.
    for (int i = 0; i < kNumSignals; i++) {
        if (kSignals[i] != sig) continue;
        const struct sigaction& prev = g_previous[i];
        const bool ignored = !(prev.sa_flags & SA_SIGINFO) &&
                             prev.sa_handler == SIG_IGN;
        if (ignored) std::signal(sig, SIG_DFL);
        else         sigaction(sig, &prev, nullptr);
        break;
    }
    std::raise(sig);
}

bool is_ours(const struct sigaction& sa) {
    return (sa.sa_flags & SA_SIGINFO) && sa.sa_sigaction == fault_handler;
}

struct sigaction ours() {
    struct sigaction sa {};
    sa.sa_sigaction = fault_handler;
    sigemptyset(&sa.sa_mask);
    sa.sa_flags = kFlags;
    return sa;
}

// Installed ONCE for the process (first guarded_call), not swapped per call:
// four sigactions in and four out per applet entry point was eight syscalls
// a frame, and the saved-and-restored dance is what the chaining above
// replaces.
void install_handlers() {
    const struct sigaction sa = ours();
    for (int i = 0; i < kNumSignals; i++) {
        sigaction(kSignals[i], &sa, &g_base_previous[i]);
        g_previous[i] = g_base_previous[i];
    }
}

// A library that installs its own handler AFTER ours would silently switch
// the guard off: a test harness arms one around every case, Python's
// faulthandler around the interpreter. So each call checks (four reads, no
// writes in the common case) and, only if displaced, re-arms — the old
// per-call swap, now the fallback instead of the rule — chaining to the
// displacer meanwhile and restoring it once no guarded call needs ours.
//
// Dispositions are process-wide while guarded calls run on several threads
// at once (the frame thread and applet job workers), so the re-arm is shared:
// whoever finds the guard displaced swaps it back in, and only the LAST
// guarded call out restores the displacer. A per-call restore would pull the
// guard out from under a call still running on another thread. The mutex is
// held only inside these two functions, never across fn, so a siglongjmp out
// of fn cannot strand it.
std::mutex g_rearm_mu;
int  g_rearm_depth = 0;                    // guarded calls in flight
struct sigaction g_rearm_saved[kNumSignals];
bool g_rearm_swapped[kNumSignals] = {};
bool g_rearm_warned = false;

struct Rearm {
    Rearm() {
        std::lock_guard<std::mutex> lk(g_rearm_mu);
        ++g_rearm_depth;
        const struct sigaction sa = ours();
        for (int i = 0; i < kNumSignals; i++) {
            struct sigaction cur {};
            if (sigaction(kSignals[i], nullptr, &cur) != 0 || is_ours(cur))
                continue;
            // Displaced (again, possibly mid-flight): the newest displacer is
            // the one to put back when the last guarded call leaves.
            sigaction(kSignals[i], &sa, &g_rearm_saved[i]);
            g_previous[i] = g_rearm_saved[i];
            g_rearm_swapped[i] = true;
            if (!g_rearm_warned) {
                g_rearm_warned = true;
                std::fprintf(stderr,
                             "[guard] another library replaced the %s handler; "
                             "re-arming the guard around each applet call\n",
                             describe(kSignals[i]));
            }
        }
    }
    ~Rearm() {
        std::lock_guard<std::mutex> lk(g_rearm_mu);
        if (--g_rearm_depth > 0) return;
        for (int i = 0; i < kNumSignals; i++) {
            if (!g_rearm_swapped[i]) continue;
            sigaction(kSignals[i], &g_rearm_saved[i], nullptr);
            g_previous[i] = g_base_previous[i];
            g_rearm_swapped[i] = false;
        }
    }
};

// Per-thread alternate signal stack. A stack overflow faults on the guard
// page with no stack left for the handler itself; the handler runs here
// instead and siglongjmps back onto the (now unwound) normal stack. Set up
// lazily by the first guarded_call on each thread, released with the thread.
// Re-registered if something disabled it in between (the same scoped
// installers that swap handlers also swap alt stacks).
struct AltStack {
    void*  mem = nullptr;
    size_t size = 0;
    AltStack() {
        size = std::max<size_t>(static_cast<size_t>(SIGSTKSZ), 64 * 1024);
        mem = std::malloc(size);
    }
    ~AltStack() {
        if (!mem) return;
        stack_t cur {};
        if (sigaltstack(nullptr, &cur) == 0 && cur.ss_sp == mem) {
            stack_t off {};
            off.ss_flags = SS_DISABLE;
            sigaltstack(&off, nullptr);
        }
        std::free(mem);
    }
    void ensure() {
        if (!mem) return;
        stack_t cur {};
        if (sigaltstack(nullptr, &cur) == 0 && !(cur.ss_flags & SS_DISABLE))
            return;                        // some alt stack is in place: fine
        stack_t ss {};
        ss.ss_sp = mem;
        ss.ss_size = size;
        ss.ss_flags = 0;
        sigaltstack(&ss, nullptr);
    }
};
thread_local AltStack t_altstack;

} // namespace

GuardResult guarded_call(const std::function<void()>& fn) {
    static std::once_flag once;
    std::call_once(once, install_handlers);
    t_altstack.ensure();
    Rearm rearm;

    GuardResult r;
    t_active = 1;
    if (sigsetjmp(t_jmp, 1) == 0) {
        // A C++ exception out of an applet entry point is contained the same
        // way a signal is: the applet quarantines, the host keeps its frame.
        try {
            fn();
        } catch (const std::exception& e) {
            r.ok = false;
            r.fault = std::string("C++ exception: ") + e.what();
        } catch (...) {
            r.ok = false;
            r.fault = "C++ exception (non-std type)";
        }
    } else {
        r.ok = false;
        r.fault = describe(t_signal);
    }
    t_active = 0;
    return r;
}

} // namespace caliper_host
#endif
