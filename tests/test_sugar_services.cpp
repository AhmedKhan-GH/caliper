#include <doctest/doctest.h>
#include <caliper/caliper.hpp>
#include <caliper/fixture_host.h>
#include <string>

// Fake tables: prove the sugar wrappers route through get_service correctly.
namespace {
uint64_t last_submit_seen = 0;
uint64_t fake_submit(const char* label, CaliperJobFn, void*) {
    last_submit_seen = (label && std::string(label) == "train") ? 7u : 1u;
    return last_submit_seen;
}
void fake_cancel(uint64_t) {}
bool fake_running(uint64_t id) { return id == 7; }
float fake_progress(uint64_t) { return 0.25f; }
const CaliperJobsV1 kFakeJobs = {sizeof(CaliperJobsV1), &fake_submit,
                                 &fake_cancel, &fake_running, &fake_progress};

CaliperDeviceKind fake_kind(void) { return CALIPER_DEV_METAL; }
int32_t fake_index(void) { return 0; }
const char* fake_name(void) { return "FakeGPU"; }
uint64_t fake_hint(void) { return 42; }
const CaliperDeviceV1 kFakeDev = {sizeof(CaliperDeviceV1), &fake_kind,
                                  &fake_index, &fake_name, &fake_hint};
} // namespace

TEST_CASE("sugar: Jobs wrapper routes through the service table") {
    caliper::testing::FixtureHost fx;
    fx.provide(CALIPER_JOBS_V1, &kFakeJobs);
    caliper::Host host(fx.host());
    caliper::Jobs jobs(host);
    REQUIRE(static_cast<bool>(jobs));
    CHECK(jobs.submit("train", nullptr, nullptr) == 7);
    CHECK(jobs.is_running(7));
    CHECK(jobs.progress_of(7) == doctest::Approx(0.25f));
}

TEST_CASE("sugar: Jobs wrapper is falsy without the service") {
    caliper::testing::FixtureHost fx;
    caliper::Host host(fx.host());
    caliper::Jobs jobs(host);
    CHECK_FALSE(static_cast<bool>(jobs));
    CHECK(jobs.submit("x", nullptr, nullptr) == 0);   // inert, not UB
}

TEST_CASE("sugar: Device::query snapshots the table") {
    caliper::testing::FixtureHost fx;
    fx.provide(CALIPER_DEVICE_V1, &kFakeDev);
    caliper::Host host(fx.host());
    auto dev = caliper::Device::query(host);
    CHECK(dev.kind == CALIPER_DEV_METAL);
    CHECK(std::string(dev.name) == "FakeGPU");
    CHECK(dev.free_memory_hint == 42);
}

TEST_CASE("sugar: Device::query defaults to CPU without the service") {
    caliper::testing::FixtureHost fx;
    auto dev = caliper::Device::query(caliper::Host(fx.host()));
    CHECK(dev.kind == CALIPER_DEV_CPU);
}
