// caliper_catalog_providers_tests: the host's catalog providers and data
// sources for metrics, feed channels and live views
// (src/host/catalog_providers), headless.
//
// Metrics are written through a real MetricsStore in a temporary folder and
// feed samples through a real FeedStore. Live views are read two ways: through
// LiveCatalog::apply(), the frame-thread step that update() ends with, fed by
// a stand-in for the live core (no GPU needed), and through update() over an
// OFFSCREEN embed core whose typed live views are fed by caliper_core_live_apply
// with CPU payloads, as both transports feed them (the pattern of
// caliper_live_views_tests; skipped without a GPU). Then a Session with the
// providers registered resolves a spec with a live source and the entropy
// transforms, a metric and a feed channel, and the values are checked against
// a hand computation.
#define DOCTEST_CONFIG_IMPLEMENT
#include <doctest/doctest.h>
#include "gpu_skip.h"
#include "test_temp.h"

#include "catalog_providers/catalog_providers.h"
#include "feed_store.h"
#include "metrics_store.h"

#include <caliper/embed.h>
#include <caliper/live.h>
#include <caliper/live_views.h>
#include <caliper_builder/session/session.h>

#include "live/live_views_state.h"   // set_payload_map: a tiled heatmap's payload axes

#include <chrono>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <string>
#include <thread>
#include <vector>

using namespace caliper_builder;
using caliper_host::FeedCatalog;
using caliper_host::FeedStore;
using caliper_host::LiveCatalog;
using caliper_host::MetricsCatalog;
using caliper_host::MetricsStore;
using nlohmann::json;
namespace fs = std::filesystem;

namespace {

struct TempDir {
    fs::path path;
    explicit TempDir(const std::string& name) : path(caliper_test::unique_temp_path(name)) {
        fs::create_directories(path);
    }
    ~TempDir() {
        std::error_code ec;
        fs::remove_all(path, ec);
    }
};

std::vector<std::string> ids_of(const std::vector<CatalogEntry>& entries) {
    std::vector<std::string> out;
    for (const auto& e : entries) out.push_back(e.id);
    return out;
}

CatalogFilter filter(std::optional<CatalogKind> kind = std::nullopt, std::string prefix = {},
                     std::size_t limit = 100) {
    CatalogFilter f;
    f.kind = kind;
    f.prefix = std::move(prefix);
    f.limit = limit;
    return f;
}

CatalogSampleRequest rows(std::size_t limit) {
    CatalogSampleRequest r;
    r.limit = limit;
    return r;
}

std::vector<double> values_of(const TensorData& t) {
    std::vector<double> out;
    for (uint64_t i = 0; i < t.numel(); ++i) out.push_back(t.element(i));
    return out;
}

double entropy(const float* row, int n) {
    double sum = 0;
    for (int i = 0; i < n; ++i) sum += row[i] > 0 ? row[i] : 0;
    double h = 0;
    for (int i = 0; i < n; ++i)
        if (row[i] > 0) h -= (row[i] / sum) * std::log(row[i] / sum);
    return h;
}

// A [layers, heads, tokens, tokens] attention tensor whose rows are softmax
// rows of a fixed pattern; `salt` changes every value.
std::vector<float> attention(int layers, int heads, int tokens, int salt = 0) {
    std::vector<float> a(static_cast<std::size_t>(layers) * heads * tokens * tokens);
    for (int l = 0; l < layers; ++l)
        for (int h = 0; h < heads; ++h)
            for (int q = 0; q < tokens; ++q) {
                float* row = &a[((static_cast<std::size_t>(l) * heads + h) * tokens + q) * tokens];
                double sum = 0;
                for (int k = 0; k < tokens; ++k) {
                    row[k] = std::exp(0.3f * static_cast<float>(((l + 1) * (h + 2) * (q + 3) * (k + 1) + salt) % 7));
                    sum += row[k];
                }
                for (int k = 0; k < tokens; ++k) row[k] = static_cast<float>(row[k] / sum);
            }
    return a;
}

// Entropy over the last axis, then the mean over the query tokens: [layers, heads].
std::vector<double> entropy_per_head(const std::vector<float>& a, int layers, int heads, int tokens) {
    std::vector<double> out;
    for (int lh = 0; lh < layers * heads; ++lh) {
        double m = 0;
        for (int q = 0; q < tokens; ++q) m += entropy(&a[(static_cast<std::size_t>(lh) * tokens + q) * tokens], tokens);
        out.push_back(m / tokens);
    }
    return out;
}

CommandResult run(Session& s, const std::string& name, json args = json::object()) {
    CommandInvocation inv;
    inv.name = name;
    inv.args = std::move(args);
    inv.origin = CommandOrigin::ui;
    return s.invoke(std::move(inv));
}

json ok(Session& s, const std::string& name, json args = json::object()) {
    CommandResult r = run(s, name, std::move(args));
    INFO(name << ": " << r.error.code << " at " << r.error.where << ": " << r.error.message);
    REQUIRE(r.ok);
    return r.result;
}

// A fake clock for the live catalog's copy policy and rates.
struct FakeClock {
    std::shared_ptr<std::uint64_t> t = std::make_shared<std::uint64_t>(1000000000ull);
    std::function<std::uint64_t()> fn() const {
        auto p = t;
        return [p] { return *p; };
    }
    void advance_ms(std::uint64_t ms) { *t += ms * 1000000ull; }
};

LiveCatalog::ViewFrame f32_frame(const std::string& name, std::uint32_t kind, std::uint64_t updates,
                                 std::vector<std::int64_t> shape, const std::vector<float>& values) {
    LiveCatalog::ViewFrame f;
    f.name = name;
    f.kind = kind;
    f.updates = updates;
    f.has_payload = true;
    f.dtype = DType::f32;
    f.shape = std::move(shape);
    f.data = reinterpret_cast<const std::uint8_t*>(values.data());
    f.payload_bytes = values.size() * 4;
    return f;
}

}  // namespace

// ---- metrics ------------------------------------------------------------------

TEST_CASE("catalog/metrics: the store's change counter, series summary and bounded read") {
    TempDir dir("caliper-catalog-metrics-store");
    MetricsStore store;
    REQUIRE(store.open((dir.path / "metrics.duckdb").string()));
    const std::uint64_t r0 = store.revision();
    const std::uint64_t run = store.begin_run("exp", "a");
    CHECK(store.revision() > r0);
    for (int s = 9; s >= 0; --s) store.scalar(run, "loss", s, s * 0.5);   // written out of order
    store.scalar(run, "acc", 4, 0.25);
    const std::uint64_t r1 = store.revision();
    CHECK(r1 > r0 + 10);
    (void)store.scalars(run, "loss");
    (void)store.scalar_series();
    CHECK(store.revision() == r1);   // reads do not count
    const auto series = store.scalar_series();
    REQUIRE(series.size() == 2);
    CHECK(series[0].tag == "acc");
    CHECK(series[0].points == 1);
    CHECK(series[1].tag == "loss");
    CHECK(series[1].points == 10);
    CHECK(series[1].last_step == 9);
    const auto tail = store.scalars(run, "loss", 3);
    REQUIRE(tail.size() == 3);
    CHECK(tail[0].first == 7);
    CHECK(tail[2].first == 9);
    CHECK(tail[2].second == 4.5);
    CHECK(store.scalars(run, "loss", 0).empty());
    store.delete_run(run);
    CHECK(store.revision() > r1);
    CHECK(store.scalar_series().empty());
    store.close();
}

TEST_CASE("catalog/metrics: list, describe and sample runs and tags with filters and limits") {
    TempDir dir("caliper-catalog-metrics");
    MetricsStore store;
    REQUIRE(store.open((dir.path / "metrics.duckdb").string()));
    const std::uint64_t r1 = store.begin_run("exp", "baseline");
    const std::uint64_t r2 = store.begin_run("", "");
    for (int s = 0; s < 10; ++s) store.scalar(r1, "train/loss", s, 10.0 - s);
    store.scalar(r1, "eval/acc", 0, 0.5);
    store.scalar(r1, "eval/acc", 5, std::nan(""));
    store.scalar(r2, "train/loss", 3, 1.0);
    store.scalar(r2, "bad\ntag", 1, 1.0);   // a control character: no valid id, not listed
    MetricsCatalog cat(store, {250, false});
    CHECK(cat.list(filter()).entries.empty());   // nothing until the first refresh
    cat.refresh_now();

    const std::string a = std::to_string(r1), b = std::to_string(r2);
    const CatalogListResult all = cat.list(filter());
    REQUIRE(all.error.ok());
    CHECK(ids_of(all.entries) == std::vector<std::string>{"metric:" + a + "/eval/acc", "metric:" + a + "/train/loss",
                                                          "metric:" + b + "/train/loss"});
    CHECK_FALSE(all.truncated);
    CHECK(cat.list(filter(CatalogKind::feed)).entries.empty());
    CHECK(cat.list(filter(CatalogKind::metric, a + "/")).entries.size() == 2);
    const CatalogListResult one = cat.list(filter(std::nullopt, {}, 1));
    CHECK(one.entries.size() == 1);
    CHECK(one.truncated);

    const CatalogDescribeResult d = cat.describe("metric:" + a + "/train/loss");
    REQUIRE(d.entry);
    const CatalogEntry& e = *d.entry;
    CHECK(validate_catalog_entry(e).ok());
    CHECK(e.title == "baseline train/loss");
    CHECK(e.origin.service == "metrics");
    CHECK(e.origin.experiment == "exp");
    CHECK(e.origin.run == static_cast<std::int64_t>(r1));
    CHECK(e.origin.run_name == "baseline");
    CHECK(e.origin.tag == "train/loss");
    CHECK(e.rows == 10u);
    REQUIRE(e.columns);
    CHECK((*e.columns)[0].name == "step");
    CHECK((*e.columns)[1].type == "DOUBLE");
    CHECK_FALSE(e.rate_hz);
    CHECK_FALSE(e.updated_at);
    CHECK_FALSE(e.wire);
    const CatalogDescribeResult unnamed = cat.describe("metric:" + b + "/train/loss");
    REQUIRE(unnamed.entry);
    CHECK(unnamed.entry->title == "train/loss");
    CHECK_FALSE(unnamed.entry->origin.experiment);
    CHECK_FALSE(unnamed.entry->origin.run_name);
    CHECK_FALSE(cat.describe("metric:" + a + "/nothing").entry);
    CHECK_FALSE(cat.describe("feed:" + a + "/train/loss").entry);

    // The newest points by step, in step order.
    const CatalogSampleResult s = cat.sample("metric:" + a + "/train/loss", rows(3));
    REQUIRE(s.error.ok());
    CHECK(*s.sample.rows == json{{7, 3.0}, {8, 2.0}, {9, 1.0}});
    CHECK(s.sample.total_rows == 10u);
    CHECK(s.sample.truncated);
    const CatalogSampleResult acc = cat.sample("metric:" + a + "/eval/acc", rows(20));
    REQUIRE(acc.error.ok());
    CHECK(*acc.sample.rows == json{{0, 0.5}, {5, "nan"}});
    CHECK_FALSE(acc.sample.truncated);
    CHECK(cat.sample("metric:" + a + "/nothing", rows(5)).error.code == "catalog.not_found");

    // Through a Catalog, every entry passes the strict check.
    Catalog catalog;
    std::shared_ptr<MetricsCatalog> shared(&cat, [](MetricsCatalog*) {});
    REQUIRE(catalog.add_provider("metrics", shared).ok());
    const CatalogMergedList merged = catalog.list(filter());
    CHECK(merged.entries.size() == 3);
    CHECK(merged.warnings.empty());
    CHECK(catalog.sample("metric:" + a + "/train/loss", rows(2)).sample.rows->size() == 2);
    store.close();
}

TEST_CASE("catalog/metrics: generations follow each series, the revision follows the series, and a deleted run goes") {
    TempDir dir("caliper-catalog-metrics-gen");
    MetricsStore store;
    REQUIRE(store.open((dir.path / "metrics.duckdb").string()));
    const std::uint64_t r1 = store.begin_run("exp", "a");
    const std::uint64_t r2 = store.begin_run("exp", "b");
    store.scalar(r1, "loss", 0, 1.0);
    store.scalar(r1, "acc", 0, 0.1);
    store.scalar(r2, "loss", 0, 2.0);
    MetricsCatalog cat(store, {250, false});
    cat.refresh_now();
    const std::string loss = "metric:" + std::to_string(r1) + "/loss";
    const std::string acc = "metric:" + std::to_string(r1) + "/acc";
    const std::string other = "metric:" + std::to_string(r2) + "/loss";
    const std::uint64_t g_loss = cat.generation(loss), g_acc = cat.generation(acc), rev = cat.revision();
    CHECK(g_loss != 0);
    CHECK(cat.generation("metric:999/none") == 0);

    store.scalar(r1, "loss", 1, 0.5);
    CHECK(cat.generation(loss) == g_loss);   // until the summary is refreshed
    cat.refresh_now();
    CHECK(cat.generation(loss) > g_loss);
    CHECK(cat.generation(acc) == g_acc);     // another series did not change
    CHECK(cat.revision() == rev);            // points are data, not schema

    // read(): the newest points within the request, with the summary's generation.
    ServedRequest req;
    req.max_rows = 1;
    ServedData d = cat.read(loss, req);
    REQUIRE(d.error.ok());
    CHECK(d.tabular);
    CHECK(d.rows == json{{1, 0.5}});
    CHECK(d.truncated);
    CHECK(d.generation == cat.generation(loss));
    CHECK(d.columns.size() == 2);

    store.scalar(r1, "lr", 0, 1e-3);
    cat.refresh_now();
    CHECK(cat.revision() > rev);

    // A deleted run: before the refresh the read finds no points; after it the
    // series is gone from the catalog.
    store.delete_run(r2);
    CHECK(cat.read(other, {}).error.code == "resolve.unavailable");
    CHECK(cat.sample(other, rows(5)).error.code == "catalog.no_data");
    const std::uint64_t before = cat.revision();
    cat.refresh_now();
    CHECK_FALSE(cat.describe(other).entry);
    CHECK(cat.generation(other) == 0);
    CHECK(cat.read(other, {}).error.code == "resolve.unavailable");
    CHECK(cat.revision() > before);
    store.close();
}

TEST_CASE("catalog/metrics: the summary's thread picks up new series on its own") {
    TempDir dir("caliper-catalog-metrics-thread");
    MetricsStore store;
    REQUIRE(store.open((dir.path / "metrics.duckdb").string()));
    MetricsCatalog cat(store, {10, true});
    const std::uint64_t run = store.begin_run("exp", "a");
    store.scalar(run, "loss", 0, 1.0);
    const std::string id = "metric:" + std::to_string(run) + "/loss";
    bool seen = false;
    for (int i = 0; i < 300 && !seen; ++i) {
        seen = cat.describe(id).entry.has_value();
        if (!seen) std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    CHECK(seen);
    store.close();   // the summary empties itself once the store is closed
    bool gone = false;
    for (int i = 0; i < 300 && !gone; ++i) {
        gone = !cat.describe(id).entry.has_value();
        if (!gone) std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    CHECK(gone);
}

// ---- feed ---------------------------------------------------------------------

TEST_CASE("catalog/feed: channels with their fields, rate, rows and newest samples") {
    FeedStore store;
    REQUIRE(store.add_channel("sys.cpu.util", "CPU utilisation", "%", 10.0f, 8));
    REQUIRE(store.add_channel("sys.irregular", "", "", 0.0f, 4));
    REQUIRE(store.add_channel(" padded", "Padded", "", 1.0f, 4));   // not a catalog name: not listed
    FeedCatalog cat(store);
    const std::int64_t now = FeedCatalog::steady_now_ns();
    for (int i = 0; i < 10; ++i) store.push("sys.cpu.util", now - (10 - i) * 1000000, static_cast<float>(i));
    store.push("sys.irregular", now, std::nanf(""));

    const CatalogListResult all = cat.list(filter());
    CHECK(ids_of(all.entries) == std::vector<std::string>{"feed:sys.cpu.util", "feed:sys.irregular"});
    CHECK(cat.list(filter(CatalogKind::metric)).entries.empty());
    CHECK(cat.list(filter(CatalogKind::feed, "sys.cpu")).entries.size() == 1);
    CHECK(cat.list(filter(std::nullopt, {}, 1)).truncated);

    const CatalogDescribeResult d = cat.describe("feed:sys.cpu.util");
    REQUIRE(d.entry);
    const CatalogEntry& e = *d.entry;
    CHECK(validate_catalog_entry(e).ok());
    CHECK(e.title == "CPU utilisation");
    CHECK(e.origin.service == "feed");
    CHECK(e.origin.channel == "sys.cpu.util");
    CHECK(e.origin.units == "%");
    CHECK(e.rows == 8u);   // the ring holds 8 of the 10
    REQUIRE(e.rate_hz);
    CHECK(*e.rate_hz == 10.0);
    REQUIRE(e.columns);
    CHECK(e.columns->size() == 3);
    CHECK((*e.columns)[2].type == "FLOAT");
    REQUIRE(e.updated_at);
    CHECK(is_utc_timestamp(*e.updated_at));
    const CatalogDescribeResult irregular = cat.describe("feed:sys.irregular");
    REQUIRE(irregular.entry);
    CHECK(irregular.entry->title == "sys.irregular");
    CHECK_FALSE(irregular.entry->rate_hz);
    CHECK_FALSE(irregular.entry->origin.units);
    CHECK_FALSE(cat.describe("feed:none").entry);
    CHECK_FALSE(cat.describe("feed: padded").entry);

    const CatalogSampleResult s = cat.sample("feed:sys.cpu.util", rows(3));
    REQUIRE(s.error.ok());
    REQUIRE(s.sample.rows->size() == 3);
    CHECK((*s.sample.rows)[0][0] == 8);   // seq, oldest of the newest three
    CHECK((*s.sample.rows)[2][2] == 9.0);
    CHECK(s.sample.total_rows == 8u);
    CHECK(s.sample.truncated);
    CHECK((*cat.sample("feed:sys.irregular", rows(5)).sample.rows)[0][2] == "nan");
    CHECK(cat.sample("feed:none", rows(5)).error.code == "catalog.not_found");

    Catalog catalog;
    std::shared_ptr<FeedCatalog> shared(&cat, [](FeedCatalog*) {});
    REQUIRE(catalog.add_provider("feed", shared).ok());
    CHECK(catalog.list(filter()).warnings.empty());
}

TEST_CASE("catalog/feed: generation, read, the revision and an unknown channel") {
    FeedStore store;
    REQUIRE(store.add_channel("a", "A", "W", 5.0f, 16));
    FeedCatalog cat(store);
    const std::uint64_t rev = cat.revision();
    CHECK(cat.generation("feed:a") == 0);
    CHECK(cat.read("feed:a", {}).error.code == "resolve.unavailable");   // no samples yet
    for (int i = 0; i < 6; ++i) store.push("a", 100 + i, static_cast<float>(i) / 2);
    CHECK(cat.generation("feed:a") == 6);
    ServedRequest req;
    req.max_rows = 4;
    ServedData d = cat.read("feed:a", req);
    REQUIRE(d.error.ok());
    CHECK(d.tabular);
    CHECK(d.rows == json{{3, 102, 1.0}, {4, 103, 1.5}, {5, 104, 2.0}, {6, 105, 2.5}});
    CHECK(d.truncated);
    CHECK(d.generation == 6);
    req.max_cells = 6;   // two rows of three cells
    CHECK(cat.read("feed:a", req).rows.size() == 2);
    store.push("a", 200, 9.0f);
    CHECK(cat.generation("feed:a") == 7);
    REQUIRE(store.add_channel("b", "B", "", 1.0f, 4));
    CHECK(cat.revision() > rev);
    CHECK(cat.generation("feed:none") == 0);
    CHECK(cat.generation("metric:a") == 0);
    CHECK(cat.read("feed:none", {}).error.code == "resolve.unavailable");
    uint32_t capacity = 0;
    uint64_t newest = 0;
    CHECK(store.ring_state("a", &capacity, &newest));
    CHECK(capacity == 16);
    CHECK(newest == 7);
    CHECK_FALSE(store.ring_state("none", &capacity, &newest));
}

// ---- live views, through the frame-thread step ------------------------------------

TEST_CASE("catalog/live: views list with their kinds; a view before its first snapshot has no data") {
    FakeClock clock;
    LiveCatalog::Options options;
    options.clock = clock.fn();
    LiveCatalog cat(options);
    CHECK(cat.list(filter()).entries.empty());
    CHECK(cat.read("live:x", {}).error.message.find("No live session") != std::string::npos);

    const std::vector<float> heat = {1, 2, 3, 4, 5, 6};
    LiveCatalog::ViewFrame waiting;
    waiting.name = "loss";
    waiting.kind = CALIPER_LIVE_KIND_SERIES;
    const std::uint64_t rev0 = cat.revision();
    cat.apply(true, "clp-1", {f32_frame("attn.weights", CALIPER_LIVE_KIND_HEATMAP, 1, {2, 3}, heat), waiting});
    CHECK(cat.revision() > rev0);

    const CatalogListResult all = cat.list(filter());
    CHECK(ids_of(all.entries) == std::vector<std::string>{"live:attn.weights", "live:loss"});
    CHECK(cat.list(filter(CatalogKind::live, "attn")).entries.size() == 1);
    CHECK(cat.list(filter(CatalogKind::feed)).entries.empty());
    CHECK(cat.list(filter(std::nullopt, {}, 1)).truncated);

    const CatalogEntry e = *cat.describe("live:attn.weights").entry;
    CHECK(validate_catalog_entry(e).ok());
    CHECK(e.title == "attn.weights");
    CHECK(e.origin.session == "clp-1");
    CHECK(e.origin.view == "attn.weights");
    CHECK(e.origin.view_type == "heatmap");
    CHECK(e.dtype == CatalogDtype::f32);
    CHECK(e.shape == std::vector<std::int64_t>{2, 3});
    CHECK(e.bytes == 24u);
    CHECK(e.wire);
    REQUIRE(e.updated_at);
    CHECK(is_utc_timestamp(*e.updated_at));
    CHECK_FALSE(e.rate_hz);   // one snapshot so far

    const CatalogEntry w = *cat.describe("live:loss").entry;
    CHECK(validate_catalog_entry(w).ok());
    CHECK_FALSE(w.dtype);
    CHECK_FALSE(w.shape);
    CHECK(w.origin.view_type == "series");
    CHECK(cat.sample("live:loss", {}).error.code == "catalog.no_data");
    CHECK(cat.read("live:loss", {}).error.code == "resolve.unavailable");
    CHECK(cat.sample("live:none", {}).error.code == "catalog.not_found");
    CHECK_FALSE(cat.describe("live:none").entry);

    // The first snapshot is copied at once; the sample reads the copy.
    const CatalogSampleResult s = cat.sample("live:attn.weights", {});
    REQUIRE(s.error.ok());
    CHECK(s.sample.shape == std::vector<std::int64_t>{2, 3});
    CHECK(s.sample.stats->count == 6u);
    CHECK(*s.sample.stats->max == 6.0);
    CHECK(s.sample.slice->values == json{1.0, 2.0, 3.0, 4.0, 5.0, 6.0});
    const ServedData d = cat.read("live:attn.weights", {});
    REQUIRE(d.error.ok());
    CHECK(values_of(d.tensor) == std::vector<double>{1, 2, 3, 4, 5, 6});
    CHECK(d.generation == cat.generation("live:attn.weights"));
    CHECK(d.generation != 0);

    Catalog catalog;
    std::shared_ptr<LiveCatalog> shared(&cat, [](LiveCatalog*) {});
    REQUIRE(catalog.add_provider("live", shared).ok());
    CHECK(catalog.list(filter()).warnings.empty());
    CHECK(catalog.sample("live:loss", {}).error.code == "catalog.no_data");
}

// The names of a view's dims, for the builder's Properties: one per dim of
// its shape, as the stream's payload axes give them, or none.
TEST_CASE("catalog/live: a view's dim names, when its axes give one per dim") {
    FakeClock clock;
    LiveCatalog::Options options;
    options.clock = clock.fn();
    LiveCatalog cat(options);
    const std::vector<float> heat(2 * 3, 1.0f);
    LiveCatalog::ViewFrame named = f32_frame("attn.entropy", CALIPER_LIVE_KIND_HEATMAP, 1, {2, 3}, heat);
    named.dims = {"layer", "head"};
    LiveCatalog::ViewFrame odd = f32_frame("odd", CALIPER_LIVE_KIND_HEATMAP, 1, {2, 3}, heat);
    odd.dims = {"layer"};   // not one per dim
    LiveCatalog::ViewFrame waiting;
    waiting.name = "loss";
    waiting.kind = CALIPER_LIVE_KIND_SERIES;
    waiting.dims = {"step"};   // no payload yet
    cat.apply(true, "clp-1", {named, odd, waiting, f32_frame("plain", CALIPER_LIVE_KIND_HEATMAP, 1, {2, 3}, heat)});
    CHECK(cat.dim_names("live:attn.entropy") == std::vector<std::string>{"layer", "head"});
    CHECK(cat.dim_names("live:odd").empty());
    CHECK(cat.dim_names("live:loss").empty());
    CHECK(cat.dim_names("live:plain").empty());
    CHECK(cat.dim_names("live:none").empty());
    CHECK(cat.dim_names("tensor:attn.entropy").empty());
    cat.apply(false, {}, {});
    CHECK(cat.dim_names("live:attn.entropy").empty());
}

TEST_CASE("catalog/live: a wanted view is copied at every snapshot, an idle one once a second; the rate") {
    FakeClock clock;
    LiveCatalog::Options options;
    options.clock = clock.fn();
    LiveCatalog cat(options);
    std::vector<float> v = {0, 0, 0, 0};
    auto frame = [&](std::uint64_t updates) {
        v[0] = static_cast<float>(updates);
        return f32_frame("w", CALIPER_LIVE_KIND_HEATMAP, updates, {2, 2}, v);
    };
    cat.apply(true, "s", {frame(1)});
    CHECK(cat.copies() == 1);
    const std::uint64_t g1 = cat.generation("live:w");   // this read makes the view wanted
    // Wanted: every new snapshot is copied, one frame later at most.
    for (std::uint64_t u = 2; u <= 6; ++u) {
        clock.advance_ms(100);
        cat.apply(true, "s", {frame(u)});
    }
    CHECK(cat.copies() == 6);
    CHECK(cat.generation("live:w") > g1);
    CHECK(values_of(cat.read("live:w", {}).tensor)[0] == 6.0);
    // Snapshots every 100 ms: 10 per second over the window.
    REQUIRE(cat.describe("live:w").entry->rate_hz);
    CHECK(*cat.describe("live:w").entry->rate_hz == doctest::Approx(10.0));
    // A frame with nothing new copies nothing.
    cat.apply(true, "s", {frame(6)});
    CHECK(cat.copies() == 6);

    // No reader for longer than wanted_ns: new snapshots are copied at most once
    // per idle_copy_ns.
    clock.advance_ms(2500);
    cat.apply(true, "s", {frame(7)});   // the last copy is older than a second: copied
    CHECK(cat.copies() == 7);
    for (std::uint64_t u = 8; u <= 12; ++u) {
        clock.advance_ms(100);
        cat.apply(true, "s", {frame(u)});
    }
    CHECK(cat.copies() == 7);
    clock.advance_ms(600);
    cat.apply(true, "s", {frame(13)});
    CHECK(cat.copies() == 8);
    // The rate keeps measuring every snapshot that arrived, copied or not.
    REQUIRE(cat.describe("live:w").entry->rate_hz);
    // Without snapshots for longer than the window the rate is unknown again.
    clock.advance_ms(6000);
    cat.apply(true, "s", {frame(13)});
    CHECK_FALSE(cat.describe("live:w").entry->rate_hz);
}

TEST_CASE("catalog/live: a tiled heatmap's cells, the copy limit, the revision and a detach") {
    FakeClock clock;
    LiveCatalog::Options options;
    options.clock = clock.fn();
    options.max_view_bytes = 64;
    LiveCatalog cat(options);
    // Three 2 x 2 cells in a 2 x 2 mosaic of 4 x 4, the fourth cell NaN.
    const float nan = std::nanf("");
    const std::vector<float> mosaic = {
        0, 1, 10, 11,
        2, 3, 12, 13,
        20, 21, nan, nan,
        22, 23, nan, nan,
    };
    LiveCatalog::ViewFrame tiled = f32_frame("attn.layers", CALIPER_LIVE_KIND_HEATMAP, 1, {3, 2, 2}, mosaic);
    tiled.cells = 3;
    tiled.tile_rows = 2;
    tiled.tile_cols = 2;
    tiled.cell_h = 2;
    tiled.cell_w = 2;
    const std::vector<float> big(32, 1.0f);   // 128 bytes, past the limit of 64
    cat.apply(true, "s", {tiled, f32_frame("big", CALIPER_LIVE_KIND_HEATMAP, 1, {4, 8}, big)});
    const ServedData d = cat.read("live:attn.layers", {});
    REQUIRE(d.error.ok());
    CHECK(d.tensor.shape() == std::vector<int64_t>{3, 2, 2});
    CHECK(values_of(d.tensor) == std::vector<double>{0, 1, 2, 3, 10, 11, 12, 13, 20, 21, 22, 23});
    CHECK(cat.describe("live:big").entry->shape == std::vector<std::int64_t>{4, 8});
    const ServedData too = cat.read("live:big", {});
    CHECK(too.error.code == "resolve.unavailable");
    CHECK(too.error.message.find("copy limit") != std::string::npos);
    CHECK(cat.sample("live:big", {}).error.code == "catalog.no_data");
    ServedRequest small;
    small.max_bytes = 16;
    CHECK(cat.read("live:attn.layers", small).error.code == "resolve.too_large");

    // A series that grows does not change the revision; a heatmap whose shape
    // changes does; so does a view that goes.
    std::vector<float> s1 = {1, 2}, s2 = {1, 2, 3};
    cat.apply(true, "s", {tiled, f32_frame("loss", CALIPER_LIVE_KIND_SERIES, 1, {2}, s1)});
    const std::uint64_t rev = cat.revision();
    cat.apply(true, "s", {tiled, f32_frame("loss", CALIPER_LIVE_KIND_SERIES, 2, {3}, s2)});
    CHECK(cat.revision() == rev);
    CHECK(cat.describe("live:loss").entry->shape == std::vector<std::int64_t>{3});
    const std::vector<float> wide(8, 2.0f);
    cat.apply(true, "s", {f32_frame("attn.layers", CALIPER_LIVE_KIND_HEATMAP, 2, {2, 4}, wide),
                          f32_frame("loss", CALIPER_LIVE_KIND_SERIES, 2, {3}, s2)});
    CHECK(cat.revision() > rev);
    const std::uint64_t rev2 = cat.revision();
    cat.apply(true, "s", {f32_frame("loss", CALIPER_LIVE_KIND_SERIES, 2, {3}, s2)});
    CHECK(cat.revision() > rev2);
    CHECK_FALSE(cat.describe("live:attn.layers").entry);
    CHECK(cat.read("live:attn.layers", {}).error.message.find("no view") != std::string::npos);

    // A detach: every view goes, and reads say why.
    const std::uint64_t rev3 = cat.revision();
    cat.apply(false, {}, {});
    CHECK(cat.revision() > rev3);
    CHECK(cat.list(filter()).entries.empty());
    CHECK_FALSE(cat.describe("live:loss").entry);
    CHECK(cat.generation("live:loss") == 0);
    const ServedData gone = cat.read("live:loss", {});
    CHECK(gone.error.code == "resolve.unavailable");
    CHECK(gone.error.message.find("No live session is attached") != std::string::npos);
    CHECK(cat.sample("live:loss", {}).error.code == "catalog.not_found");
}

// ---- live views, through a real core -------------------------------------------

namespace {

// One offscreen core per case (one core per process), as caliper_live_views_tests.
struct OffscreenCore {
    CaliperCore* core = nullptr;
    bool gpu = false;
    OffscreenCore() {
        CaliperCoreDesc d{};
        d.struct_size = sizeof d;
        d.renderer = CALIPER_RENDERER_DEFAULT;
        core = caliper_core_create(&d);
        REQUIRE(core != nullptr);
        if (!caliper_test::force_no_gpu()) {
            CaliperCanvasDesc c{};
            c.struct_size = sizeof c;
            c.mode = CALIPER_CANVAS_OFFSCREEN;
            c.width = 320;
            c.height = 240;
            c.content_scale = 1.0f;
            gpu = caliper_core_attach_canvas(core, nullptr, &c) == 1;
        }
    }
    ~OffscreenCore() {
        if (core) caliper_core_shutdown(core);
    }
    bool apply(const char* name, int32_t kind, std::vector<float>& values, std::vector<int64_t> shape,
               uint64_t seq, float stats_lo = 0.0f, float stats_hi = 1.0f) {
        CaliperTensor t{};
        t.struct_size = sizeof t;
        t.data = values.data();
        t.dtype = CALIPER_DT_F32;
        t.ndim = static_cast<int32_t>(shape.size());
        int64_t st = 1;
        for (int i = t.ndim - 1; i >= 0; --i) {
            t.shape[i] = shape[static_cast<std::size_t>(i)];
            t.strides[i] = st;
            st *= shape[static_cast<std::size_t>(i)];
        }
        t.device = CALIPER_DEV_CPU;
        float stats[2] = {stats_lo, stats_hi};
        CaliperLiveSnapshot s{};
        s.struct_size = sizeof s;
        s.name = name;
        s.kind = kind;
        s.data = &t;
        s.stats = stats;
        s.stats_count = 2;
        s.vmin = 0.0f;
        s.vmax = 1.0f;
        s.seq = seq;
        if (kind == CALIPER_LIVE_KIND_SERIES) s.series_total = static_cast<uint64_t>(shape[0]);
        return caliper_core_live_apply(core, &s) == 1;
    }
};

}  // namespace

TEST_CASE("catalog/live: update() reads the typed live views of a core, as the app's attachment feeds them") {
    OffscreenCore c;
    if (!c.gpu) CALIPER_SKIP_NO_GPU("no offscreen canvas for the embed core");
    std::vector<float> heat(4 * 6);
    for (std::size_t i = 0; i < heat.size(); ++i) heat[i] = static_cast<float>(i) / 10.0f;
    std::vector<float> loss = {3.0f, 2.0f, 1.0f};
    std::vector<float> bins = {1.0f, 4.0f, 2.0f};
    REQUIRE_MESSAGE(c.apply("attn", CALIPER_LIVE_KIND_HEATMAP, heat, {4, 6}, 1), caliper_core_last_error(c.core));
    REQUIRE_MESSAGE(c.apply("loss", CALIPER_LIVE_KIND_SERIES, loss, {3}, 1), caliper_core_last_error(c.core));
    REQUIRE_MESSAGE(c.apply("hist", CALIPER_LIVE_KIND_HISTOGRAM, bins, {3}, 1, -1.0f, 1.0f),
                    caliper_core_last_error(c.core));

    LiveCatalog cat;
    cat.update(c.core, "clp-test");
    CHECK(ids_of(cat.list(filter()).entries) == std::vector<std::string>{"live:attn", "live:hist", "live:loss"});
    const CatalogEntry e = *cat.describe("live:attn").entry;
    CHECK(e.shape == std::vector<std::int64_t>{4, 6});
    CHECK(e.dtype == CatalogDtype::f32);
    CHECK(e.origin.session == "clp-test");
    const ServedData d = cat.read("live:attn", {});
    REQUIRE_MESSAGE(d.error.ok(), d.error.message);
    REQUIRE(d.tensor.numel() == heat.size());
    for (std::size_t i = 0; i < heat.size(); ++i) CHECK(d.tensor.element(i) == doctest::Approx(heat[i]));
    CHECK(values_of(cat.read("live:loss", {}).tensor) == std::vector<double>{3, 2, 1});
    CHECK(values_of(cat.read("live:hist", {}).tensor) == std::vector<double>{1, 4, 2});
    CHECK(cat.describe("live:hist").entry->origin.view_type == "histogram");

    // A new snapshot of the wanted view is copied by the next update.
    const std::uint64_t g = cat.generation("live:attn");
    heat[0] = 9.0f;
    REQUIRE(c.apply("attn", CALIPER_LIVE_KIND_HEATMAP, heat, {4, 6}, 2));
    cat.update(c.core, "clp-test");
    CHECK(cat.generation("live:attn") > g);
    CHECK(cat.read("live:attn", {}).tensor.element(0) == doctest::Approx(9.0));

    // A tiled heatmap, as a stream's data view with a tile dim sends it: three
    // 2 x 3 cells in a 2 x 2 mosaic of 4 x 6, read back as [3, 2, 3].
    caliper_live::PayloadMap map;
    map.axes = {{"layer", 0, 1, 3, 3}, {"q", 0, 1, 2, 2}, {"k", 0, 1, 3, 3}};
    map.tile_rows = 2;
    map.tile_cols = 2;
    CHECK(cat.dim_names("live:attn").empty());   // the identity map names nothing
    REQUIRE(caliper_live_views::set_payload_map(c.core, "attn", map));
    cat.update(c.core, "clp-test");
    CHECK(cat.describe("live:attn").entry->shape == std::vector<std::int64_t>{3, 2, 3});
    // Its dims are named by the payload's axes.
    CHECK(cat.dim_names("live:attn") == std::vector<std::string>{"layer", "q", "k"});
    CHECK(cat.dim_names("live:loss").empty());
    heat[1] = 8.0f;   // a new snapshot, so that the cells are copied
    REQUIRE(c.apply("attn", CALIPER_LIVE_KIND_HEATMAP, heat, {4, 6}, 3));
    REQUIRE(caliper_live_views::set_payload_map(c.core, "attn", map));
    cat.update(c.core, "clp-test");
    const ServedData cells = cat.read("live:attn", {});
    REQUIRE(cells.error.ok());
    CHECK(cells.tensor.shape() == std::vector<int64_t>{3, 2, 3});
    // Cell 1 is the mosaic's columns 3 to 5 of rows 0 and 1.
    CHECK(cells.tensor.element(6) == doctest::Approx(heat[3]));
    CHECK(cells.tensor.element(11) == doctest::Approx(heat[6 + 5]));
    // Cell 2 is rows 2 and 3, columns 0 to 2.
    CHECK(cells.tensor.element(12) == doctest::Approx(heat[12]));

    // A view the core removes goes from the catalog.
    REQUIRE(caliper_core_live_view_remove(c.core, "hist"));
    cat.update(c.core, "clp-test");
    CHECK_FALSE(cat.describe("live:hist").entry);
    cat.update(nullptr, {});
    CHECK(cat.list(filter()).entries.empty());
}

// ---- through a Session ------------------------------------------------------------

TEST_CASE("session: a live source with the entropy transforms, a metric and a feed channel resolve through a Session") {
    TempDir dir("caliper-catalog-session");
    MetricsStore metrics;
    REQUIRE(metrics.open((dir.path / "metrics.duckdb").string()));
    FeedStore feed;
    REQUIRE(feed.add_channel("sys.cpu.util", "CPU", "%", 10.0f, 64));
    SessionOptions so;
    so.registry.check_results = true;
    auto opened = Session::open((dir.path / "workspace").string(), so);
    REQUIRE(opened.error.ok());
    Session& s = *opened.session;
    MetricsCatalog::Options mo;
    mo.thread = false;
    caliper_host::CatalogProviders p = caliper_host::register_catalog_providers(s, &metrics, &feed, {}, mo);
    REQUIRE(p.error.ok());
    REQUIRE(p.live);
    REQUIRE(p.feed);
    REQUIRE(p.metrics);

    // The live view: [layers, heads, tokens, tokens] attention.
    const int L = 3, H = 2, T = 5;
    std::vector<float> attn = attention(L, H, T);
    auto frame = [&](std::uint64_t updates) {
        return f32_frame("attn.weights", CALIPER_LIVE_KIND_HEATMAP, updates, {L, H, T, T}, attn);
    };
    p.live->apply(true, "clp-1", {frame(1)});

    // The catalog commands see it, with its shape.
    const json listed = ok(s, "catalog.list", {{"kind", "live"}});
    REQUIRE(listed["entries"].size() == 1);
    CHECK(listed["entries"][0]["shape"] == json{L, H, T, T});
    CHECK(listed["entries"][0]["wire"] == true);
    const json sampled = ok(s, "catalog.sample", {{"id", "live:attn.weights"}});
    CHECK(sampled["stats"]["count"] == L * H * T * T);

    // "Plot attention entropy per layer while it trains."
    ok(s, "spec.new", {{"id", "live"}});
    ok(s, "source.add", {{"id", "attn"}, {"catalog", "live:attn.weights"}});
    ok(s, "source.add", {{"id", "entropy"}, {"from", "attn"},
                         {"transform", {{{"op", "reduce"}, {"axis", -1}, {"fn", "entropy"}},
                                        {{"op", "reduce"}, {"axis", -1}, {"fn", "mean"}}}}});
    ok(s, "panel.create", {{"id", "ent"}, {"primitive", "heatmap"}, {"bind", {{"values", "entropy"}}}});
    const PanelData d = s.resolve("ent");
    REQUIRE_MESSAGE(d.error.ok(), d.error.message);
    REQUIRE(d.slots.size() == 1);
    CHECK(d.slots[0].tensor.shape() == std::vector<int64_t>{L, H});
    const std::vector<double> want = entropy_per_head(attn, L, H, T);
    const std::vector<double> got = values_of(d.slots[0].tensor);
    REQUIRE(got.size() == want.size());
    for (std::size_t i = 0; i < want.size(); ++i) CHECK(got[i] == doctest::Approx(want[i]).epsilon(1e-6));

    // The generation: the probe matches, a frame with nothing new changes
    // nothing, and the next snapshot changes it and the values.
    const spec::Spec spec = s.open_spec()->spec;
    CHECK(d.generation != 0);
    CHECK(panel_generation(spec, "ent", s.data_sources()) == d.generation);
    p.live->apply(true, "clp-1", {frame(1)});
    CHECK(panel_generation(spec, "ent", s.data_sources()) == d.generation);
    const auto computed = s.data_sources().cache_stats().computed;
    CHECK(s.resolve("ent").generation == d.generation);
    CHECK(s.data_sources().cache_stats().computed == computed);   // the transforms did not run again
    attn = attention(L, H, T, 3);
    p.live->apply(true, "clp-1", {frame(2)});
    CHECK(panel_generation(spec, "ent", s.data_sources()) != d.generation);
    const PanelData d2 = s.resolve("ent");
    REQUIRE(d2.error.ok());
    CHECK(d2.generation == panel_generation(spec, "ent", s.data_sources()));
    CHECK(values_of(d2.slots[0].tensor)[0] == doctest::Approx(entropy_per_head(attn, L, H, T)[0]).epsilon(1e-6));

    // A metric and a feed channel as columns.
    const std::uint64_t run = metrics.begin_run("exp", "run");
    for (int i = 0; i < 4; ++i) metrics.scalar(run, "train/loss", i, 1.0 / (i + 1));
    p.metrics->refresh_now();
    const std::int64_t now = FeedCatalog::steady_now_ns();
    for (int i = 0; i < 3; ++i) feed.push("sys.cpu.util", now + i, 10.0f * i);
    const std::string loss_id = "metric:" + std::to_string(run) + "/train/loss";
    ok(s, "source.add", {{"id", "loss"}, {"catalog", loss_id}});
    ok(s, "source.add", {{"id", "cpu"}, {"catalog", "feed:sys.cpu.util"}});
    ok(s, "panel.create", {{"id", "curve"}, {"primitive", "series"}, {"bind", {{"x", "loss.step"}, {"y", {"loss.value"}}}}});
    ok(s, "panel.create", {{"id", "cpu"}, {"primitive", "series"}, {"bind", {{"x", "cpu.t_ns"}, {"y", {"cpu.value"}}}}});
    const PanelData curve = s.resolve("curve");
    REQUIRE_MESSAGE(curve.error.ok(), curve.error.message);
    CHECK(curve.slots[0].values == json{0, 1, 2, 3});
    CHECK(curve.slots[1].values[3] == doctest::Approx(0.25));
    const PanelData cpu = s.resolve("cpu");
    REQUIRE_MESSAGE(cpu.error.ok(), cpu.error.message);
    CHECK(cpu.slots[1].values == json{0.0, 10.0, 20.0});
    const std::uint64_t g_cpu = cpu.generation;
    feed.push("sys.cpu.util", now + 9, 5.0f);
    CHECK(panel_generation(s.open_spec()->spec, "cpu", s.data_sources()) != g_cpu);

    // A detached session: the live panel says so; the others still resolve.
    p.live->apply(false, {}, {});
    const PanelData off = s.resolve("ent");
    CHECK(off.error.code == "resolve.unavailable");
    CHECK(off.error.message.find("No live session is attached") != std::string::npos);
    CHECK(s.resolve("curve").error.ok());

    // Unregistered, every served kind is unavailable again.
    caliper_host::unregister_catalog_providers(s);
    CHECK(s.resolve("curve").error.code == "resolve.unavailable");
    CHECK(ok(s, "catalog.list", json::object())["entries"].empty());
    metrics.close();
}

CALIPER_GPU_TEST_MAIN()
