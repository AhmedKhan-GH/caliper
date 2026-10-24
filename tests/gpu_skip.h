#pragma once
// Honest GPU skips for the suites that need a device (gfx, export, embed,
// torch). Until now a case that found no GPU printed a MESSAGE and returned,
// and ctest reported the suite green — a GPU-less machine "passed" having
// asserted nothing. Now:
//
//   CALIPER_SKIP_NO_GPU(why)          counts the skip, says why, returns —
//                                      and FAILS when CALIPER_REQUIRE_GPU=1,
//                                      so a lane that must have a GPU cannot
//                                      go green by accident.
//   CALIPER_SKIP_NOT_APPLICABLE(why)  for cases that never apply here (CUDA
//                                      on macOS): counted, never a failure.
//   CALIPER_NOTE_NO_GPU(why)          the counting half alone, for a site that
//                                      must tidy up before it returns.
//
// CALIPER_GPU_TEST_MAIN() replaces DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN: after
// the run it prints "[caliper-tests] skipped N of M test cases ..." and exits
// kSkipExitCode when something was skipped for want of a GPU and nothing
// failed. tests/CMakeLists.txt registers that code as SKIP_RETURN_CODE, so
// ctest shows the suite as "***Skipped", not "Passed". Not-applicable skips
// alone leave the exit code at 0: they recur on every run of that platform
// and would otherwise hide the suite's real passes behind a permanent Skipped.
//
// CALIPER_TEST_FORCE_NO_GPU=1 (test-only) makes the device environments
// report no device, so the skip path itself can be exercised on a machine that
// has one. Plain C++17, no ObjC: two of the including TUs compile as ObjC++.
#include <doctest/doctest.h>

#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace caliper_test {

constexpr int kSkipExitCode = 77;

inline std::atomic<int>& skipped_no_gpu() {
    static std::atomic<int> n{0};
    return n;
}
inline std::atomic<int>& skipped_not_applicable() {
    static std::atomic<int> n{0};
    return n;
}
inline std::atomic<int>& cases_run() {
    static std::atomic<int> n{0};
    return n;
}

inline bool env_is_1(const char* name) {
    const char* v = std::getenv(name);
    return v && std::strcmp(v, "1") == 0;
}
inline bool require_gpu()  { return env_is_1("CALIPER_REQUIRE_GPU"); }
inline bool force_no_gpu() { return env_is_1("CALIPER_TEST_FORCE_NO_GPU"); }

// Counts test cases as they start, so the summary can say "N of M".
struct CaseCounter : doctest::IReporter {
    explicit CaseCounter(const doctest::ContextOptions&) {}
    void report_query(const doctest::QueryData&) override {}
    void test_run_start() override {}
    void test_run_end(const doctest::TestRunStats&) override {}
    void test_case_start(const doctest::TestCaseData&) override { cases_run()++; }
    void test_case_reenter(const doctest::TestCaseData&) override {}
    void test_case_end(const doctest::CurrentTestCaseStats&) override {}
    void test_case_exception(const doctest::TestCaseException&) override {}
    void subcase_start(const doctest::SubcaseSignature&) override {}
    void subcase_end() override {}
    void log_assert(const doctest::AssertData&) override {}
    void log_message(const doctest::MessageData&) override {}
    void test_case_skipped(const doctest::TestCaseData&) override {}
};

// The exit code for a finished run: failures win; otherwise a GPU skip turns
// the run into a skip; otherwise doctest's own result.
inline int finish(int res) {
    const int no_gpu = skipped_no_gpu().load();
    const int na     = skipped_not_applicable().load();
    if (no_gpu == 0 && na == 0) return res;
    std::printf("[caliper-tests] skipped %d of %d test cases: %d no GPU, "
                "%d not applicable on this platform%s\n",
                no_gpu + na, cases_run().load(), no_gpu, na,
                (res == 0 && no_gpu > 0) ? " -> exit 77 (ctest: Skipped)" : "");
    std::fflush(stdout);
    if (res != 0) return res;
    return no_gpu > 0 ? kSkipExitCode : res;
}

} // namespace caliper_test

// `why` is streamed, so "no device: " << reason works.
#define CALIPER_NOTE_NO_GPU(why)                                               \
    do {                                                                       \
        ::caliper_test::skipped_no_gpu()++;                                    \
        if (::caliper_test::require_gpu())                                     \
            FAIL("CALIPER_REQUIRE_GPU=1 but " << why);                         \
        MESSAGE(why << " — skipping");                                         \
    } while (0)

#define CALIPER_SKIP_NO_GPU(why)                                               \
    do { CALIPER_NOTE_NO_GPU(why); return; } while (0)

#define CALIPER_SKIP_NOT_APPLICABLE(why)                                       \
    do {                                                                       \
        ::caliper_test::skipped_not_applicable()++;                            \
        MESSAGE(why << " — not applicable here, skipping");                    \
        return;                                                                \
    } while (0)

#define CALIPER_GPU_TEST_MAIN()                                                \
    REGISTER_LISTENER("caliper_case_counter", 0, ::caliper_test::CaseCounter); \
    int main(int argc, char** argv) {                                          \
        doctest::Context ctx(argc, argv);                                      \
        const int res = ctx.run();                                             \
        if (ctx.shouldExit()) return res;                                      \
        return ::caliper_test::finish(res);                                    \
    }
