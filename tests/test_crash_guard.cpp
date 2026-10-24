#include <doctest/doctest.h>
#include "crash_guard.h"
#include "applet_loader.h"
#include <caliper/fixture_host.h>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>
#ifdef _WIN32
// POSIX setenv/unsetenv don't exist on MSVC; _putenv_s with "" removes.
#define setenv(k, v, overwrite) _putenv_s(k, v)
#define unsetenv(k) _putenv_s(k, "")
#else
#include <csignal>
#include <sys/wait.h>
#include <unistd.h>
#endif
using namespace caliper_host;

static bool names_a_fault(const std::string& fault) {
    // POSIX guard reports "SIGSEGV (...)" / "SIGBUS (...)", Windows "SEH ...".
    return fault.find("SIG") != std::string::npos ||
           fault.find("SEH") != std::string::npos;
}

TEST_CASE("guard: normal call passes through, side effects run") {
    int x = 0;
    auto r = guarded_call([&] { x = 42; });
    CHECK(r.ok);
    CHECK(r.fault.empty());
    CHECK(x == 42);
}

TEST_CASE("guard: null write is contained and named") {
    auto r = guarded_call([] {
        volatile int* p = nullptr;
        *p = 1;
    });
    CHECK_FALSE(r.ok);
    // macOS arm64 reports EXC_BAD_ACCESS as SIGSEGV or SIGBUS — accept either.
    // The Windows guard reports "SEH exception 0x...".
    CHECK(names_a_fault(r.fault));
}

TEST_CASE("guard: handlers restore — ok call after a crash works") {
    (void)guarded_call([] { volatile int* p = nullptr; *p = 1; });
    int x = 0;
    auto r = guarded_call([&] { x = 7; });
    CHECK(r.ok);
    CHECK(x == 7);
}

TEST_CASE("guard: C++ exception is contained and named") {
    auto r = guarded_call([] { throw std::runtime_error("applet said no"); });
    CHECK_FALSE(r.ok);
    CHECK(r.fault.find("exception") != std::string::npos);
    CHECK(r.fault.find("applet said no") != std::string::npos);
    // And the guard is re-armed: a fault right after is still contained.
    auto f = guarded_call([] { volatile int* p = nullptr; *p = 1; });
    CHECK_FALSE(f.ok);
    CHECK(names_a_fault(f.fault));
}

TEST_CASE("guard: non-std exception is contained") {
    auto r = guarded_call([] { throw 42; });
    CHECK_FALSE(r.ok);
    CHECK(r.fault.find("exception") != std::string::npos);
}

namespace {
// Recursion the optimizer cannot flatten: the depth is runtime, the frame is
// written through volatile, and the call goes through a volatile pointer.
int (*volatile g_recurse)(int) = nullptr;
int recurse(int depth) {
    volatile char frame[1024];
    frame[0] = static_cast<char>(depth);
    frame[sizeof frame - 1] = frame[0];
    return g_recurse(depth + 1) + frame[sizeof frame - 1];
}
} // namespace

TEST_CASE("guard: stack overflow is contained") {
    // The fault lands on the guard page with no stack left for a handler —
    // only the per-thread alternate stack makes this catchable at all.
    g_recurse = &recurse;
    auto r = guarded_call([] { (void)g_recurse(0); });
    CHECK_FALSE(r.ok);
    CHECK(names_a_fault(r.fault));
    // The guard must still work on this thread afterwards, for an ordinary
    // fault and for a second overflow.
    auto f = guarded_call([] { volatile int* p = nullptr; *p = 1; });
    CHECK_FALSE(f.ok);
    auto again = guarded_call([] { (void)g_recurse(0); });
    CHECK_FALSE(again.ok);
    CHECK(names_a_fault(again.fault));
    int x = 0;
    CHECK(guarded_call([&] { x = 1; }).ok);
    CHECK(x == 1);
}

static bool names_exception(const std::string& text, const char* what) {
    // POSIX reports "C++ exception: <what>". MSVC under /EHsc may not unwind a
    // C++ throw through an extern "C" function pointer, in which case the SEH
    // frame reports it as "SEH exception 0xE06D7363" — still a quarantine.
    return text.find(what) != std::string::npos ||
           text.find("SEH") != std::string::npos;
}

TEST_CASE("guard: exception thrown by an applet quarantines it, host survives") {
    // Same loader path as the fault-in-frame case in test_loader.cpp, with a
    // C++ throw instead of the null write. The CALIPER_APPLET sugar (hello)
    // catches at the ABI boundary, as it should, so the fixture is a raw
    // descriptor that doesn't: the exception reaches the loader's
    // guarded_call and the applet quarantines exactly as it would for a
    // signal, naming the exception.
    caliper::testing::FixtureHost fx;
    CaliperFrameInfo fi{}; fi.struct_size = sizeof fi;
    const HostCaps caps{2, "0.6.0", {"caliper.ui.v1", "caliper.log.v1"}};
    auto find_throwing = [](AppletLoader& L) {
        for (int k = 0; k < L.count(); k++)
            if (L.at(k).manifest.id == "dev.caliper.throwing") return k;
        return -1;
    };

    SUBCASE("thrown from frame()") {
        unsetenv("CALIPER_THROWING_WHERE");            // fixture default
        AppletLoader L(caps, CALIPER_TEST_DATA_ROOT);
        L.scan(CALIPER_TEST_FIXTURES_DIR);
        int i = find_throwing(L);
        REQUIRE(i >= 0);
        REQUIRE(L.launch(i, *fx.host()));
        CHECK_FALSE(L.frame(i, fi));                   // throw -> quarantined
        CHECK(L.at(i).status == AppletStatus::Quarantined);
        CHECK(L.at(i).status_text.find("frame()") != std::string::npos);
        CHECK(names_exception(L.at(i).status_text, "throwing_applet.frame"));
        CHECK_FALSE(L.frame(i, fi));                   // never called again
    }
    SUBCASE("thrown from initialize()") {
        setenv("CALIPER_THROWING_WHERE", "initialize", 1);
        AppletLoader L(caps, CALIPER_TEST_DATA_ROOT);
        L.scan(CALIPER_TEST_FIXTURES_DIR);
        int i = find_throwing(L);
        REQUIRE(i >= 0);
        CHECK_FALSE(L.launch(i, *fx.host()));
        CHECK(L.at(i).status == AppletStatus::Quarantined);
        CHECK(L.at(i).status_text.find("initialize()") != std::string::npos);
        CHECK(names_exception(L.at(i).status_text, "throwing_applet.initialize"));
        unsetenv("CALIPER_THROWING_WHERE");
    }
    // The host (this test process) is alive to assert all of the above.
}

#ifndef _WIN32
TEST_CASE("guard: a fault outside any guarded region still terminates") {
    // Install-once must not swallow unguarded faults. A forked child arms the
    // guard with an ok call, then faults OUTSIDE guarded_call: whatever
    // disposition is in effect (the guard chaining to the default, or — under
    // doctest, whose own per-case handler is primary — that handler) must end
    // the child BY THE SIGNAL: not an exit code (swallowed), not a timeout
    // (looping). The child's stdio goes to /dev/null so the harness's own
    // crash report from inside it does not masquerade as ours.
    pid_t pid = fork();
    REQUIRE(pid >= 0);
    if (pid == 0) {
        if (FILE* sink = std::fopen("/dev/null", "w")) {
            dup2(fileno(sink), 1);
            dup2(fileno(sink), 2);
        }
        (void)guarded_call([] {});
        volatile int* p = nullptr;
        *p = 1;
        _exit(0);                                      // not reached
    }
    int status = 0;
    REQUIRE(waitpid(pid, &status, 0) == pid);
    CHECK(WIFSIGNALED(status));
    CHECK((WTERMSIG(status) == SIGSEGV || WTERMSIG(status) == SIGBUS));
}
#endif
