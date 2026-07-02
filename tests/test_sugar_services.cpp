#include <doctest/doctest.h>
#include <caliper/caliper.hpp>
#include <caliper/fixture_host.h>
#include <caliper/services/metrics_v1.h>
#include <caliper/services/tensor_bridge_v1.h>
#include <string>
#include <vector>

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

// Fake metrics.v1: records the writer calls the sugar routes through it.
struct MetricsCalls {
    std::string last_experiment, last_run_name;
    uint64_t begin_runs = 0;
    uint64_t last_scalar_run = 0;
    std::string last_scalar_tag;
    int64_t last_scalar_step = -1;
    double last_scalar_value = 0.0;
    uint64_t ended = 0;
    std::string last_hparams;
    uint64_t last_hparams_run = 0;
};
MetricsCalls g_metrics;

uint64_t fmet_begin_run(const char* exp, const char* name) {
    g_metrics.last_experiment = exp ? exp : "";
    g_metrics.last_run_name = name ? name : "";
    return ++g_metrics.begin_runs;
}
void fmet_end_run(uint64_t run) { g_metrics.ended = run; }
void fmet_scalar(uint64_t run, const char* tag, int64_t step, double value) {
    g_metrics.last_scalar_run = run;
    g_metrics.last_scalar_tag = tag ? tag : "";
    g_metrics.last_scalar_step = step;
    g_metrics.last_scalar_value = value;
}
void fmet_histogram(uint64_t, const char*, int64_t, const float*, int64_t) {}
void fmet_image(uint64_t, const char*, int64_t, const CaliperTensor*) {}
void fmet_hparams(uint64_t run, const char* json) {
    g_metrics.last_hparams_run = run;
    g_metrics.last_hparams = json ? json : "";
}
const CaliperMetricsV1 kFakeMetrics = {
    sizeof(CaliperMetricsV1), &fmet_begin_run, &fmet_end_run, &fmet_scalar,
    &fmet_histogram, &fmet_image, &fmet_hparams};

// Fake tensor_bridge.v1: records that the Bridge sugar routes through it.
struct BridgeCalls {
    int tex_calls = 0, mapped_calls = 0, update_calls = 0, release_calls = 0;
    int alloc_calls = 0, free_calls = 0;
    uint32_t last_flags = 0;
    int32_t last_colormap = -1;
    float last_vmin = 0.0f, last_vmax = 0.0f;
    CaliperTextureId last_released = 0, last_freed = 0;
};
BridgeCalls g_bridge;

CaliperTextureId fbr_tex(const CaliperTensor*, uint32_t flags) {
    g_bridge.tex_calls++; g_bridge.last_flags = flags; return 100;
}
bool fbr_update(CaliperTextureId, const CaliperTensor*) {
    g_bridge.update_calls++; return true;
}
void fbr_release(CaliperTextureId tex) {
    g_bridge.release_calls++; g_bridge.last_released = tex;
}
CaliperTextureId fbr_mapped(const CaliperTensor*, int32_t cm,
                            float vmin, float vmax, uint32_t) {
    g_bridge.mapped_calls++; g_bridge.last_colormap = cm;
    g_bridge.last_vmin = vmin; g_bridge.last_vmax = vmax; return 200;
}
bool fbr_alloc(CaliperDType, int32_t, const int64_t*, CaliperTensor*,
               CaliperTextureId* out) {
    g_bridge.alloc_calls++; if (out) *out = 300; return true;
}
void fbr_free(CaliperTextureId tex) {
    g_bridge.free_calls++; g_bridge.last_freed = tex;
}
const CaliperTensorBridgeV1 kFakeBridge = {
    sizeof(CaliperTensorBridgeV1), &fbr_tex, &fbr_update, &fbr_release,
    &fbr_mapped, &fbr_alloc, &fbr_free};
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

TEST_CASE("sugar: Metrics wrapper routes writers through the service table") {
    g_metrics = MetricsCalls{};
    caliper::testing::FixtureHost fx;
    fx.provide(CALIPER_METRICS_V1, &kFakeMetrics);
    caliper::Host host(fx.host());
    caliper::Metrics metrics(host);
    REQUIRE(static_cast<bool>(metrics));

    uint64_t run = metrics.begin_run("mnist", "run-A");
    CHECK(run == 1);
    CHECK(g_metrics.last_experiment == "mnist");
    CHECK(g_metrics.last_run_name == "run-A");

    metrics.scalar(run, "loss", 3, 0.5);
    CHECK(g_metrics.last_scalar_run == 1);
    CHECK(g_metrics.last_scalar_tag == "loss");
    CHECK(g_metrics.last_scalar_step == 3);
    CHECK(g_metrics.last_scalar_value == doctest::Approx(0.5));

    metrics.hparams_json(run, "{\"lr\":0.01}");
    CHECK(g_metrics.last_hparams_run == 1);
    CHECK(g_metrics.last_hparams == "{\"lr\":0.01}");

    metrics.end_run(run);
    CHECK(g_metrics.ended == 1);
}

TEST_CASE("sugar: Metrics wrapper is falsy and inert without the service") {
    g_metrics = MetricsCalls{};
    caliper::testing::FixtureHost fx;
    caliper::Host host(fx.host());
    caliper::Metrics metrics(host);
    CHECK_FALSE(static_cast<bool>(metrics));
    CHECK(metrics.begin_run("x", "y") == 0);   // inert, not UB
    metrics.scalar(0, "loss", 0, 0.0);
    metrics.end_run(0);
    metrics.hparams_json(0, "{}");
    CHECK(g_metrics.begin_runs == 0);          // nothing routed through
}

TEST_CASE("sugar: Bridge wrapper routes through the service table") {
    g_bridge = BridgeCalls{};
    caliper::testing::FixtureHost fx;
    fx.provide(CALIPER_TENSOR_BRIDGE_V1, &kFakeBridge);
    caliper::Host host(fx.host());
    caliper::Bridge bridge(host);
    REQUIRE(static_cast<bool>(bridge));

    CHECK(bridge.texture_from_tensor(nullptr, 5u) == 100);
    CHECK(g_bridge.last_flags == 5u);

    CHECK(bridge.texture_from_tensor_mapped(nullptr, CALIPER_CMAP_MAGMA,
                                            -1.0f, 2.0f) == 200);
    CHECK(g_bridge.last_colormap == CALIPER_CMAP_MAGMA);
    CHECK(g_bridge.last_vmin == doctest::Approx(-1.0f));
    CHECK(g_bridge.last_vmax == doctest::Approx(2.0f));

    CHECK(bridge.update_texture(100, nullptr));
    CHECK(g_bridge.update_calls == 1);

    CaliperTextureId shared_tex = 0;
    CHECK(bridge.alloc_shared(CALIPER_DT_F32, 2, nullptr, nullptr, &shared_tex));
    CHECK(shared_tex == 300);

    bridge.release_texture(100);
    CHECK(g_bridge.last_released == 100);
    bridge.free_shared(300);
    CHECK(g_bridge.last_freed == 300);

    // Opaque id -> ImTextureID convenience cast (never a raw GL/Metal handle).
    CHECK(caliper::Bridge::imtex(42) == (ImTextureID)42);
}

TEST_CASE("sugar: Bridge wrapper is falsy and inert without the service") {
    g_bridge = BridgeCalls{};
    caliper::testing::FixtureHost fx;
    caliper::Host host(fx.host());
    caliper::Bridge bridge(host);
    CHECK_FALSE(static_cast<bool>(bridge));
    CHECK(bridge.texture_from_tensor(nullptr, 0) == 0);          // inert, not UB
    CHECK(bridge.texture_from_tensor_mapped(nullptr, 0, 0.0f, 1.0f) == 0);
    CHECK_FALSE(bridge.update_texture(1, nullptr));
    CHECK_FALSE(bridge.alloc_shared(CALIPER_DT_F32, 2, nullptr, nullptr, nullptr));
    bridge.release_texture(1);
    bridge.free_shared(1);
    CHECK(g_bridge.tex_calls == 0);            // nothing routed through
}
