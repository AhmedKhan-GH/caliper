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

// Fake artifacts.v1: records the calls the Artifacts sugar routes through it.
struct ArtifactCalls {
    std::string last_name;
    uint64_t last_len = 0, last_run = 0;
    int puts = 0;
    std::string last_lookup;
};
ArtifactCalls g_artifacts;

bool fart_put(const char* name, const void* bytes, uint64_t len,
              uint64_t run, char out_digest[65]) {
    if (!bytes) return false;
    g_artifacts.puts++;
    g_artifacts.last_name = name ? name : "";
    g_artifacts.last_len = len;
    g_artifacts.last_run = run;
    for (int i = 0; i < 64; i++) out_digest[i] = 'a';
    out_digest[64] = '\0';
    return true;
}
const char* fart_path_of(const char* key) {
    g_artifacts.last_lookup = key ? key : "";
    return "/fake/artifacts/blob";
}
bool fart_exists(const char* key) {
    return key && std::string(key) == std::string(64, 'a');
}
const CaliperArtifactsV1 kFakeArtifacts = {sizeof(CaliperArtifactsV1),
                                           &fart_put, &fart_path_of,
                                           &fart_exists};

// Fake data.v1: serves one static int32+double batch through a real
// ArrowArrayStream, so Data::drain_numeric is exercised end to end.
std::string g_last_sql;
const int32_t kColA[3] = {1, 2, 3};
const double  kColB[3] = {0.5, 1.5, 2.5};

void fdat_schema_release(ArrowSchema* s) { s->release = nullptr; }
void fdat_array_release(ArrowArray* a) { a->release = nullptr; }

int fdat_get_schema(ArrowArrayStream*, ArrowSchema* out) {
    static ArrowSchema children[2];
    static ArrowSchema* child_ptrs[2] = {&children[0], &children[1]};
    children[0] = {};
    children[0].format = "i"; children[0].name = "a";
    children[0].release = &fdat_schema_release;
    children[1] = {};
    children[1].format = "g"; children[1].name = "b";
    children[1].release = &fdat_schema_release;
    *out = {};
    out->format = "+s";
    out->n_children = 2;
    out->children = child_ptrs;
    out->release = &fdat_schema_release;
    return 0;
}
int fdat_get_next(ArrowArrayStream* self, ArrowArray* out) {
    // One batch, then end-of-stream. private_data counts batches served.
    auto* served = static_cast<int*>(self->private_data);
    *out = {};
    if (*served > 0) { out->release = nullptr; return 0; }
    (*served)++;
    static const void* bufs_a[2] = {nullptr, kColA};
    static const void* bufs_b[2] = {nullptr, kColB};
    static ArrowArray children[2];
    static ArrowArray* child_ptrs[2] = {&children[0], &children[1]};
    children[0] = {};
    children[0].length = 3; children[0].n_buffers = 2;
    children[0].buffers = bufs_a; children[0].release = &fdat_array_release;
    children[1] = {};
    children[1].length = 3; children[1].n_buffers = 2;
    children[1].buffers = bufs_b; children[1].release = &fdat_array_release;
    out->length = 3;
    out->n_children = 2;
    out->children = child_ptrs;
    out->release = &fdat_array_release;
    return 0;
}
const char* fdat_stream_error(ArrowArrayStream*) { return nullptr; }
void fdat_stream_release(ArrowArrayStream* self) { self->release = nullptr; }

int g_stream_batches_served = 0;
bool fdat_query(const char* sql, struct ArrowArrayStream* out) {
    g_last_sql = sql ? sql : "";
    if (g_last_sql.find("garbage") != std::string::npos) return false;
    g_stream_batches_served = 0;
    *out = {};
    out->get_schema = &fdat_get_schema;
    out->get_next = &fdat_get_next;
    out->get_last_error = &fdat_stream_error;
    out->release = &fdat_stream_release;
    out->private_data = &g_stream_batches_served;
    return true;
}
bool fdat_register(const char* name, const char* uri) {
    return name && uri && std::string(name) == "points";
}
bool fdat_open(const char* name, struct ArrowArrayStream* out) {
    return fdat_query(name, out);
}
const char* fdat_last_error(void) { return "fake error"; }
const CaliperDataV1 kFakeData = {sizeof(CaliperDataV1), &fdat_query,
                                 &fdat_register, &fdat_open,
                                 &fdat_last_error};

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

TEST_CASE("sugar: Artifacts wrapper routes through the service table") {
    g_artifacts = ArtifactCalls{};
    caliper::testing::FixtureHost fx;
    fx.provide(CALIPER_ARTIFACTS_V1, &kFakeArtifacts);
    caliper::Host host(fx.host());
    caliper::Artifacts artifacts(host);
    REQUIRE(static_cast<bool>(artifacts));

    const char bytes[] = "ckpt";
    std::string digest = artifacts.put("model", bytes, sizeof(bytes), 9);
    CHECK(digest == std::string(64, 'a'));
    CHECK(g_artifacts.puts == 1);
    CHECK(g_artifacts.last_name == "model");
    CHECK(g_artifacts.last_len == sizeof(bytes));
    CHECK(g_artifacts.last_run == 9);

    CHECK(artifacts.exists(digest.c_str()));
    CHECK_FALSE(artifacts.exists("something-else"));
    CHECK(std::string(artifacts.path_of("model")) == "/fake/artifacts/blob");
    CHECK(g_artifacts.last_lookup == "model");
}

TEST_CASE("sugar: Artifacts wrapper is falsy and inert without the service") {
    caliper::testing::FixtureHost fx;
    caliper::Host host(fx.host());
    caliper::Artifacts artifacts(host);
    CHECK_FALSE(static_cast<bool>(artifacts));
    CHECK(artifacts.put("x", "y", 1).empty());
    CHECK(artifacts.path_of("x") == nullptr);
    CHECK_FALSE(artifacts.exists("x"));
}

TEST_CASE("sugar: Data wrapper routes SQL and drains numeric streams") {
    caliper::testing::FixtureHost fx;
    fx.provide(CALIPER_DATA_V1, &kFakeData);
    caliper::Host host(fx.host());
    caliper::Data data(host);
    REQUIRE(static_cast<bool>(data));

    ArrowArrayStream stream = {};
    REQUIRE(data.query("SELECT a, b FROM t", &stream));
    CHECK(g_last_sql == "SELECT a, b FROM t");

    std::vector<std::string> names;
    std::vector<std::vector<double>> cols;
    REQUIRE(caliper::Data::drain_numeric(&stream, &names, &cols));
    CHECK(stream.release == nullptr);  // drained helper released the stream
    REQUIRE(cols.size() == 2);
    CHECK(names == std::vector<std::string>({"a", "b"}));
    CHECK(cols[0] == std::vector<double>({1.0, 2.0, 3.0}));
    CHECK(cols[1] == std::vector<double>({0.5, 1.5, 2.5}));

    CHECK_FALSE(data.query("garbage", &stream));
    CHECK(std::string(data.last_error()) == "fake error");
    CHECK(data.register_dataset("points", "/tmp/p.csv"));
    CHECK_FALSE(data.register_dataset("other", "/tmp/p.csv"));
}

TEST_CASE("sugar: Data wrapper is falsy and inert without the service") {
    caliper::testing::FixtureHost fx;
    caliper::Host host(fx.host());
    caliper::Data data(host);
    CHECK_FALSE(static_cast<bool>(data));
    ArrowArrayStream stream = {};
    CHECK_FALSE(data.query("SELECT 1", &stream));
    CHECK_FALSE(data.register_dataset("x", "y"));
    CHECK(std::string(data.last_error()) == "data.v1 is not available");
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

static uint32_t fake_caps(void) { return CALIPER_BRIDGE_CAP_STREAM_ORDERED; }

TEST_CASE("sugar Bridge::caps(): v1_1 bit surfaces; absent service -> 0") {
    caliper::testing::FixtureHost fx;            // the file's existing fixture type
    // v1-only host: caps() must be 0 (adapters drain).
    fx.provide(CALIPER_TENSOR_BRIDGE_V1, &kFakeBridge);
    {
        caliper::Host host(fx.host());
        caliper::Bridge b(host);
        CHECK(b.caps() == 0u);
    }
    // v1.1 host: the bit crosses.
    static const CaliperTensorBridgeV1_1 kFake11 = {sizeof(CaliperTensorBridgeV1_1),
        kFakeBridge.texture_from_tensor, kFakeBridge.update_texture,
        kFakeBridge.release_texture, kFakeBridge.texture_from_tensor_mapped,
        kFakeBridge.alloc_shared, kFakeBridge.free_shared, &fake_caps};
    fx.provide(CALIPER_TENSOR_BRIDGE_V1_1, &kFake11);
    {
        caliper::Host host(fx.host());
        caliper::Bridge b(host);
        CHECK(b.caps() == CALIPER_BRIDGE_CAP_STREAM_ORDERED);
    }
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
