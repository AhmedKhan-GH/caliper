#include <doctest/doctest.h>
#include "job_system.h"
#include <atomic>
#include <chrono>
#include <functional>
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
