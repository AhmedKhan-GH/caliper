#include <doctest/doctest.h>
#include "crash_guard.h"
#include "applet_loader.h"
#include "job_system.h"
#include <caliper/fixture_host.h>
#include <caliper/services/jobs_v1.h>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <stdexcept>
#include <string>
#include <thread>
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

namespace {
bool wait_for(const std::function<bool()>& pred, int ms) {
    const auto until = std::chrono::steady_clock::now() +
                       std::chrono::milliseconds(ms);
    while (std::chrono::steady_clock::now() < until) {
        if (pred()) return true;
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    return pred();
}
bool all_finished(JobSystem& js) {
    auto vs = js.views();
    for (auto& v : vs)
        if (v.running) return false;
    return !vs.empty();
}

// caliper.jobs.v1 over a test-owned JobSystem: host_services.cpp's thunks
// minus the process-wide store. C tables carry no user data, hence the static.
JobSystem* g_test_jobs = nullptr;
uint64_t tj_submit(const char* label, CaliperJobFn fn, void* user) {
    return g_test_jobs->submit_guarded(label ? label : "(job)", fn, user);
}
void  tj_cancel(uint64_t id)   { g_test_jobs->request_cancel(id); }
bool  tj_running(uint64_t id)  { return g_test_jobs->is_running(id); }
float tj_progress(uint64_t id) { return g_test_jobs->progress_of(id); }
const CaliperJobsV1 kTestJobs = {sizeof(CaliperJobsV1), &tj_submit, &tj_cancel,
                                 &tj_running, &tj_progress};
} // namespace

TEST_CASE("guard: a fault in an applet's background job quarantines the applet") {
    // The fixture submits a jobs.v1 job from its first frame(); the JOB fails
    // and frame() itself returns normally. The loader had its JobOwnerScope up
    // around that frame(), so the fault is pinned on the applet, which is
    // quarantined before its next call — exactly like a fault in frame().
    JobSystem js;
    g_test_jobs = &js;
    caliper::testing::FixtureHost fx;
    fx.provide(CALIPER_JOBS_V1, &kTestJobs);
    CaliperFrameInfo fi{}; fi.struct_size = sizeof fi;
    const HostCaps caps{2, "0.6.0",
                        {"caliper.ui.v1", "caliper.log.v1", "caliper.jobs.v1"}};
    auto launch_and_fail_a_job = [&](AppletLoader& L, const char* where) {
        setenv("CALIPER_THROWING_WHERE", where, 1);
        L.scan(CALIPER_TEST_FIXTURES_DIR);
        int i = -1;
        for (int k = 0; k < L.count(); k++)
            if (L.at(k).manifest.id == "dev.caliper.throwing") i = k;
        REQUIRE(i >= 0);
        REQUIRE(L.launch(i, *fx.host()));
        CHECK(L.frame(i, fi));                       // submits; frame is fine
        REQUIRE(wait_for([&] { return all_finished(js); }, 2000));
        CHECK(js.views().front().failed);
        CHECK(L.at(i).status == AppletStatus::Active);   // not yet: next call
        return i;
    };

    SUBCASE("memory fault in the job") {
        AppletLoader L(caps, CALIPER_TEST_DATA_ROOT);
        const int i = launch_and_fail_a_job(L, "job-fault");
        CHECK_FALSE(L.frame(i, fi));                 // quarantined, not called
        CHECK(L.at(i).status == AppletStatus::Quarantined);
        CHECK(L.at(i).status_text.find("crashed in background job") !=
              std::string::npos);
        CHECK(names_a_fault(L.at(i).status_text));
        CHECK(L.at(i).instance == nullptr);          // abandoned, never destroyed
        CHECK_FALSE(L.frame(i, fi));                 // never called again
    }
    SUBCASE("exception thrown by the job") {
        AppletLoader L(caps, CALIPER_TEST_DATA_ROOT);
        const int i = launch_and_fail_a_job(L, "job");
        CHECK_FALSE(L.frame(i, fi));
        CHECK(L.at(i).status == AppletStatus::Quarantined);
        CHECK(names_exception(L.at(i).status_text, "throwing_applet.job"));
    }
    SUBCASE("going home after the job faulted abandons, not cleans up") {
        AppletLoader L(caps, CALIPER_TEST_DATA_ROOT);
        const int i = launch_and_fail_a_job(L, "job-fault");
        L.teardown(i);                               // the Home path
        CHECK(L.at(i).status == AppletStatus::Quarantined);
        CHECK(L.at(i).instance == nullptr);
    }
    unsetenv("CALIPER_THROWING_WHERE");
    // The host (this test process) is alive, and the job system still works.
    std::atomic<bool> ran{false};
    uint64_t id = js.submit("after", [](void* u, const CaliperJobControl*) {
        static_cast<std::atomic<bool>*>(u)->store(true);
    }, &ran);
    CHECK(wait_for([&] { return !js.is_running(id); }, 2000));
    CHECK(ran.load());
    g_test_jobs = nullptr;
}

TEST_CASE("guard: concurrent guarded calls keep the guard armed for each other") {
    // Frame thread and job workers now guard at the same time. Once something
    // has displaced the guard's handlers (doctest around every case, Python's
    // faulthandler for an interpreter's lifetime) the re-arm is shared: the
    // call that finishes first must not restore the displacer while another
    // thread's guarded call is still running.
#ifndef _WIN32
    // Displace it here, so the case does not depend on which ran before it:
    // arm the guard, then install a handler over it. Reaching that handler
    // means a fault went unguarded; it ends the binary with a telling code.
    (void)guarded_call([] {});
    struct sigaction displacer {}, saved_segv {}, saved_bus {};
    displacer.sa_handler = [](int) { _exit(97); };
    sigemptyset(&displacer.sa_mask);
    sigaction(SIGSEGV, &displacer, &saved_segv);
    sigaction(SIGBUS, &displacer, &saved_bus);
#endif
    std::atomic<bool> b_in{false}, a_in{false}, b_done{false};
    std::thread b([&] {
        (void)guarded_call([&] {
            b_in.store(true);
            while (!a_in.load()) std::this_thread::yield();
        });
        b_done.store(true);                          // B's guard has exited
    });
    REQUIRE(wait_for([&] { return b_in.load(); }, 2000));
    auto r = guarded_call([&] {
        a_in.store(true);
        while (!b_done.load()) std::this_thread::yield();
        volatile int* p = nullptr;
        *p = 1;                                      // A faults after B left
    });
    b.join();
#ifndef _WIN32
    sigaction(SIGSEGV, &saved_segv, nullptr);
    sigaction(SIGBUS, &saved_bus, nullptr);
#endif
    CHECK_FALSE(r.ok);
    CHECK(names_a_fault(r.fault));
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

TEST_CASE("guard: a host-internal job that faults still takes the process down") {
    // JobSystem::submit() is the host's own path and stays unguarded on
    // purpose: a host bug must fail loudly, not become a failed job. A forked
    // child arms the guard (so its handlers are in place and chaining), then
    // faults in an unguarded job; it must die BY THE SIGNAL. Exit code 3 would
    // mean the fault was swallowed into a failed job. Bounded wait: a hung
    // child is killed and reported, never waited on forever.
    pid_t pid = fork();
    REQUIRE(pid >= 0);
    if (pid == 0) {
        if (FILE* sink = std::fopen("/dev/null", "w")) {
            dup2(fileno(sink), 1);
            dup2(fileno(sink), 2);
        }
        (void)guarded_call([] {});
        JobSystem js;
        uint64_t id = js.submit("host-internal", [](void*, const CaliperJobControl*) {
            volatile int* p = nullptr;
            *p = 1;
        }, nullptr);
        (void)wait_for([&] { return !js.is_running(id); }, 5000);
        _exit(js.failed(id) ? 3 : 0);                // not reached
    }
    int status = 0;
    pid_t got = 0;
    const bool exited = wait_for([&] {
        got = waitpid(pid, &status, WNOHANG);
        return got == pid;
    }, 20000);
    if (!exited) { kill(pid, SIGKILL); waitpid(pid, &status, 0); }
    REQUIRE(exited);
    CHECK(WIFSIGNALED(status));
    CHECK((WTERMSIG(status) == SIGSEGV || WTERMSIG(status) == SIGBUS));
}
#endif
