#include <doctest/doctest.h>
#include "job_system.h"
#include <atomic>
#include <chrono>
#include <cstdint>
#include <functional>
#include <memory>
#include <stdexcept>
#include <string>
#include <thread>
using namespace caliper_host;
using namespace std::chrono;

namespace {
bool wait_until(const std::function<bool()>& pred, milliseconds timeout) {
    auto deadline = steady_clock::now() + timeout;
    while (steady_clock::now() < deadline) {
        if (pred()) return true;
        std::this_thread::sleep_for(milliseconds(1));
    }
    return pred();
}
} // namespace

TEST_CASE("jobs: runs to completion, progress reported") {
    JobSystem js;
    std::atomic<bool> ran{false};
    auto fn = [](void* user, const CaliperJobControl* ctl) {
        ctl->progress(ctl, 1.0f, "done");
        static_cast<std::atomic<bool>*>(user)->store(true);
    };
    uint64_t id = js.submit("t", fn, &ran);
    REQUIRE(id != 0);
    CHECK(wait_until([&] { return !js.is_running(id); }, milliseconds(2000)));
    CHECK(ran.load());
    CHECK(js.progress_of(id) == doctest::Approx(1.0f));
}

TEST_CASE("jobs: cancel honored within 100ms (§16 contract)") {
    JobSystem js;
    auto fn = [](void*, const CaliperJobControl* ctl) {
        while (!ctl->cancelled(ctl))
            std::this_thread::sleep_for(milliseconds(1));
    };
    uint64_t id = js.submit("spin", fn, nullptr);
    REQUIRE(wait_until([&] { return js.is_running(id); }, milliseconds(2000)));
    auto t0 = steady_clock::now();
    js.request_cancel(id);
    CHECK(wait_until([&] { return !js.is_running(id); }, milliseconds(100)));
    CHECK(duration_cast<milliseconds>(steady_clock::now() - t0).count() <= 100);
}

TEST_CASE("jobs: progress + message visible in views") {
    JobSystem js;
    auto fn = [](void*, const CaliperJobControl* ctl) {
        ctl->progress(ctl, 0.5f, "halfway");
        while (!ctl->cancelled(ctl))
            std::this_thread::sleep_for(milliseconds(1));
    };
    uint64_t id = js.submit("labelled", fn, nullptr);
    REQUIRE(wait_until([&] { return js.progress_of(id) == 0.5f; },
                       milliseconds(2000)));
    auto vs = js.views();
    REQUIRE(!vs.empty());
    bool found = false;
    for (auto& v : vs)
        if (v.id == id) {
            found = true;
            CHECK(v.label == "labelled");
            CHECK(v.message == "halfway");
            CHECK(v.running);
        }
    CHECK(found);
    js.request_cancel(id);
}

TEST_CASE("jobs: unknown id is inert") {
    JobSystem js;
    CHECK_FALSE(js.is_running(424242));
    CHECK(js.progress_of(424242) == 0.0f);
    js.request_cancel(424242);  // no crash
}

TEST_CASE("jobs: destructor cancels and joins") {
    std::atomic<bool> exited{false};
    {
        JobSystem js;
        auto fn = [](void* user, const CaliperJobControl* ctl) {
            while (!ctl->cancelled(ctl))
                std::this_thread::sleep_for(milliseconds(1));
            static_cast<std::atomic<bool>*>(user)->store(true);
        };
        js.submit("forever", fn, &exited);
        std::this_thread::sleep_for(milliseconds(20));
    }   // ~JobSystem must cancel + join, not hang
    CHECK(exited.load());
}

TEST_CASE("jobs: cancel_all_and_join waits for a worker still touching state") {
    // The Home/teardown crash: a worker keeps writing to caller-owned state
    // for a while AFTER seeing cancel. cancel_all_and_join() must not return
    // until that write is finished — otherwise teardown frees the state under
    // a live thread (SIGSEGV). Proves it is a real join, not a timeout.
    struct Shared { std::atomic<bool> writing{false}, done{false}; };
    Shared sh;
    {
        JobSystem js;
        auto fn = [](void* user, const CaliperJobControl* ctl) {
            auto* s = static_cast<Shared*>(user);
            while (!ctl->cancelled(ctl))
                std::this_thread::sleep_for(milliseconds(1));
            s->writing.store(true);                 // still touching state...
            std::this_thread::sleep_for(milliseconds(120));  // ...well past a
            s->done.store(true);                    // short timeout window
            s->writing.store(false);
        };
        js.submit("late-writer", fn, &sh);
        std::this_thread::sleep_for(milliseconds(20));
        js.cancel_all_and_join();                   // must block until done
        CHECK(sh.done.load());                      // worker fully finished
        CHECK_FALSE(sh.writing.load());             // not mid-write on return
    }
}

TEST_CASE("jobs: concurrent jobs get distinct ids and all finish") {
    JobSystem js;
    std::atomic<int> done{0};
    auto fn = [](void* user, const CaliperJobControl*) {
        static_cast<std::atomic<int>*>(user)->fetch_add(1);
    };
    uint64_t a = js.submit("a", fn, &done), b = js.submit("b", fn, &done),
             c = js.submit("c", fn, &done);
    CHECK(a != b);
    CHECK(b != c);
    CHECK(wait_until([&] { return done.load() == 3; }, milliseconds(2000)));
}

TEST_CASE("jobs: a throwing fn fails the job, not the host") {
    // An exception escaping a std::thread is std::terminate — the whole host
    // gone for one bad job. The worker must catch it and report it instead.
    JobSystem js;
    auto fn = [](void*, const CaliperJobControl* ctl) {
        ctl->progress(ctl, 0.25f, "about to throw");
        throw std::runtime_error("boom");
    };
    uint64_t id = js.submit("thrower", fn, nullptr);
    REQUIRE(id != 0);
    REQUIRE(wait_until([&] { return !js.is_running(id); }, milliseconds(2000)));
    CHECK(js.failed(id));
    CHECK(js.progress_of(id) == doctest::Approx(0.25f));   // progress kept
    bool found = false;
    for (auto& v : js.views())
        if (v.id == id) {
            found = true;
            CHECK(v.failed);
            CHECK_FALSE(v.running);
            CHECK(v.message.find("boom") != std::string::npos);
        }
    CHECK(found);
    // The host (this test process) is alive to assert all of the above, and
    // an unrelated job still runs after the failure.
    std::atomic<bool> ran{false};
    auto ok = [](void* user, const CaliperJobControl*) {
        static_cast<std::atomic<bool>*>(user)->store(true);
    };
    uint64_t id2 = js.submit("after", ok, &ran);
    CHECK(wait_until([&] { return !js.is_running(id2); }, milliseconds(2000)));
    CHECK(ran.load());
    CHECK_FALSE(js.failed(id2));
}

TEST_CASE("jobs: a non-std exception is contained with a generic message") {
    JobSystem js;
    auto fn = [](void*, const CaliperJobControl*) { throw 42; };
    uint64_t id = js.submit("thrower-int", fn, nullptr);
    REQUIRE(wait_until([&] { return !js.is_running(id); }, milliseconds(2000)));
    CHECK(js.failed(id));
    bool found = false;
    for (auto& v : js.views())
        if (v.id == id) {
            found = true;
            CHECK(v.failed);
            CHECK(v.message.find("exception") != std::string::npos);
        }
    CHECK(found);
}

TEST_CASE("jobs: cancel_all_and_join gives up on a worker that ignores cancel") {
    // A fn that never polls ctl->cancelled must not hang shutdown forever: the
    // join is bounded by the deadline, the straggler is detached. Its state is
    // static so the detached thread has something valid to touch until it
    // really ends (we wait for that below so the case leaves nothing behind).
    static std::atomic<bool> s_exited{false};
    s_exited.store(false);
    auto fn = [](void* user, const CaliperJobControl*) {
        std::this_thread::sleep_for(milliseconds(600));   // deaf to cancel
        static_cast<std::atomic<bool>*>(user)->store(true);
    };
    auto t0 = steady_clock::now();
    {
        JobSystem js;
        js.submit("deaf", fn, &s_exited);
        std::this_thread::sleep_for(milliseconds(20));
        js.cancel_all_and_join(milliseconds(100));        // must give up ~100ms
        CHECK_FALSE(s_exited.load());                     // it really was deaf
    }   // ~JobSystem: nothing left to wait for
    const auto waited = duration_cast<milliseconds>(steady_clock::now() - t0);
    CHECK(waited.count() < 450);                          // bounded, not 600+
    CHECK(wait_until([&] { return s_exited.load(); }, milliseconds(2000)));
}

TEST_CASE("jobs: finished workers are reaped without clear_finished") {
    // Until now a finished thread stayed unjoined until the panel's "clear".
    // submit()/views() now join on the host's behalf; the entry stays listed.
    JobSystem js;
    auto fn = [](void*, const CaliperJobControl*) {};
    uint64_t id = js.submit("quick", fn, nullptr);
    REQUIRE(wait_until([&] { return !js.is_running(id); }, milliseconds(2000)));
    CHECK(js.reap_finished() == 1);        // the one finished thread, joined
    CHECK(js.reap_finished() == 0);        // and not twice
    CHECK(js.views().size() == 1);         // still listed for the panel

    uint64_t id2 = js.submit("quick-2", fn, nullptr);
    REQUIRE(wait_until([&] { return !js.is_running(id2); }, milliseconds(2000)));
    (void)js.views();                      // the per-frame call reaps too
    CHECK(js.reap_finished() == 0);
    js.clear_finished();                   // still drops them, nothing to join
    CHECK(js.views().empty());
}

// ---------------------------------------------------------------------------
// Guarded jobs (submit_guarded = caliper.jobs.v1's path): applet code under
// the crash guard. A fault or an exception fails the job with the guard's
// reason; the host keeps running and later jobs still run. The fault is pinned
// on the JobOwner in scope at submit time, for the loader to quarantine.
// ---------------------------------------------------------------------------
namespace {
bool names_a_fault(const std::string& text) {
    // POSIX "SIGSEGV (...)" / "SIGBUS (...)" (macOS arm64 reports either for
    // a null write); Windows "SEH exception 0x...".
    return text.find("SIG") != std::string::npos ||
           text.find("SEH") != std::string::npos;
}
std::string message_of(JobSystem& js, uint64_t id) {
    for (auto& v : js.views())
        if (v.id == id) return v.message;
    return {};
}
void null_write(void*, const CaliperJobControl*) {
    volatile int* p = nullptr;
    *p = 1;
}
void mark_ran(void* user, const CaliperJobControl*) {
    static_cast<std::atomic<bool>*>(user)->store(true);
}
} // namespace

TEST_CASE("jobs: an applet job that segfaults fails the job, not the host") {
    JobSystem js;
    uint64_t id = js.submit_guarded("segv", &null_write, nullptr);
    REQUIRE(id != 0);
    REQUIRE(wait_until([&] { return !js.is_running(id); }, milliseconds(2000)));
    CHECK(js.failed(id));
    const std::string msg = message_of(js, id);
    CHECK(msg.rfind("failed: ", 0) == 0);
    CHECK(names_a_fault(msg));
    // The host (this test process) is alive, and later jobs — guarded and
    // host-internal — still run on fresh workers.
    std::atomic<bool> ran_guarded{false}, ran_host{false};
    uint64_t g = js.submit_guarded("after-guarded", &mark_ran, &ran_guarded);
    uint64_t h = js.submit("after-host", &mark_ran, &ran_host);
    CHECK(wait_until([&] { return !js.is_running(g) && !js.is_running(h); },
                     milliseconds(2000)));
    CHECK(ran_guarded.load());
    CHECK(ran_host.load());
    CHECK_FALSE(js.failed(g));
    CHECK_FALSE(js.failed(h));
}

TEST_CASE("jobs: an applet job that throws fails with the exception named") {
    JobSystem js;
    auto fn = [](void*, const CaliperJobControl* ctl) {
        ctl->progress(ctl, 0.5f, "about to throw");
        throw std::runtime_error("applet boom");
    };
    uint64_t id = js.submit_guarded("thrower", fn, nullptr);
    REQUIRE(wait_until([&] { return !js.is_running(id); }, milliseconds(2000)));
    CHECK(js.failed(id));
    CHECK(js.progress_of(id) == doctest::Approx(0.5f));
    const std::string msg = message_of(js, id);
    CHECK(msg.find("exception") != std::string::npos);
    CHECK(msg.find("applet boom") != std::string::npos);
}

TEST_CASE("jobs: a guarded job's fault is recorded on the owner in scope at submit") {
    JobSystem js;
    auto owner = std::make_shared<JobOwner>();
    uint64_t id = 0;
    {
        JobOwnerScope scope(owner);
        CHECK(JobOwnerScope::current() == owner);
        id = js.submit_guarded("owned", &null_write, nullptr);
    }
    CHECK(JobOwnerScope::current() == nullptr);      // scope restored
    REQUIRE(wait_until([&] { return !js.is_running(id); }, milliseconds(2000)));
    CHECK(owner->faulted());
    CHECK(owner->fault().find("background job \"owned\"") != std::string::npos);
    CHECK(names_a_fault(owner->fault()));

    SUBCASE("a job submitted by an owned job inherits the owner") {
        auto parent_owner = std::make_shared<JobOwner>();
        auto parent = [](void* user, const CaliperJobControl*) {
            auto* sys = static_cast<JobSystem*>(user);
            sys->submit_guarded("child", &null_write, nullptr);
        };
        {
            JobOwnerScope scope(parent_owner);
            js.submit_guarded("parent", parent, &js);
        }
        CHECK(wait_until([&] { return parent_owner->faulted(); },
                         milliseconds(2000)));
        CHECK(parent_owner->fault().find("\"child\"") != std::string::npos);
    }
    SUBCASE("outside any scope the job still fails, unattributed") {
        uint64_t loose = js.submit_guarded("loose", &null_write, nullptr);
        REQUIRE(wait_until([&] { return !js.is_running(loose); },
                           milliseconds(2000)));
        CHECK(js.failed(loose));
        CHECK(owner->fault().find("\"owned\"") != std::string::npos);  // first wins
    }
}

TEST_CASE("jobs: a bad progress message faults outside the message lock") {
    // ctl->progress copies the applet's string before taking msg_mutex. Were
    // the copy made under the lock, the guard's siglongjmp would leave the
    // mutex held: the job's own failure report and every later views() (the
    // Jobs panel, each frame) would block on it forever.
    JobSystem js;
    auto fn = [](void*, const CaliperJobControl* ctl) {
        ctl->progress(ctl, 0.75f, reinterpret_cast<const char*>(uintptr_t{8}));
    };
    uint64_t id = js.submit_guarded("bad-msg", fn, nullptr);
    REQUIRE(wait_until([&] { return !js.is_running(id); }, milliseconds(2000)));
    // views() on a helper thread, so a regression shows up as a failed check
    // instead of a hung test binary.
    std::atomic<bool> returned{false};
    std::thread probe([&] { (void)js.views(); returned.store(true); });
    const bool ok = wait_until([&] { return returned.load(); }, milliseconds(2000));
    CHECK(ok);
    if (ok) probe.join(); else probe.detach();
    CHECK(js.failed(id));
    CHECK(names_a_fault(message_of(js, id)));
}
