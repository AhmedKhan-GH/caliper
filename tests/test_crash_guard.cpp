#include <doctest/doctest.h>
#include "crash_guard.h"
using namespace caliper_host;

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
    CHECK(r.fault.find("SIG") != std::string::npos);
}

TEST_CASE("guard: handlers restore — ok call after a crash works") {
    (void)guarded_call([] { volatile int* p = nullptr; *p = 1; });
    int x = 0;
    auto r = guarded_call([&] { x = 7; });
    CHECK(r.ok);
    CHECK(x == 7);
}
