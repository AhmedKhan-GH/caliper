// Tests for caliper_host::MetricsStore — the DuckDB-backed run/tag/step store
// behind caliper.metrics.v1. TDD: these cases are written before the
// implementation. The §16 contract (10,000 shuffled scalars back strictly
// step-ascending with matching values) is the load-bearing guarantee.
#include <doctest/doctest.h>

#include "metrics_store.h"

#include <algorithm>
#include <atomic>
#include <numeric>
#include <random>
#include <string>
#include <thread>
#include <vector>

using caliper_host::MetricsStore;

TEST_CASE("metrics: begin_run returns nonzero ids and runs() reflects them") {
    MetricsStore store;
    REQUIRE(store.open(":memory:"));

    uint64_t r1 = store.begin_run("exp", "alpha");
    uint64_t r2 = store.begin_run("exp", "beta");
    CHECK(r1 != 0);
    CHECK(r2 != 0);
    CHECK(r1 != r2);

    auto runs = store.runs();
    REQUIRE(runs.size() == 2);
    // Runs report the experiment/name they were opened with.
    bool saw_alpha = false, saw_beta = false;
    for (const auto& r : runs) {
        CHECK(r.experiment == "exp");
        if (r.name == "alpha") saw_alpha = true;
        if (r.name == "beta") saw_beta = true;
        CHECK(r.done == false);  // freshly opened, not ended
    }
    CHECK(saw_alpha);
    CHECK(saw_beta);
}

TEST_CASE("metrics: §16 contract — 10,000 shuffled scalars come back step-ascending with matching values") {
    MetricsStore store;
    REQUIRE(store.open(":memory:"));
    uint64_t run = store.begin_run("contract", "big");
    REQUIRE(run != 0);

    constexpr int N = 10000;
    std::vector<int64_t> steps(N);
    std::iota(steps.begin(), steps.end(), 0);
    std::mt19937 rng(12345);
    std::shuffle(steps.begin(), steps.end(), rng);

    // value(step) = step * 1.5 — a bijection we can verify per-row.
    for (int64_t s : steps) {
        store.scalar(run, "loss", s, static_cast<double>(s) * 1.5);
    }

    auto series = store.scalars(run, "loss");
    REQUIRE(series.size() == static_cast<size_t>(N));

    // Strictly ascending steps AND value fidelity for every row.
    for (int i = 0; i < N; ++i) {
        CHECK(series[i].first == static_cast<int64_t>(i));
        if (i > 0) CHECK(series[i].first > series[i - 1].first);  // strict ascent
        CHECK(series[i].second == doctest::Approx(static_cast<double>(i) * 1.5));
    }
}

TEST_CASE("metrics: two runs isolate — same tag, different values") {
    MetricsStore store;
    REQUIRE(store.open(":memory:"));
    uint64_t a = store.begin_run("exp", "a");
    uint64_t b = store.begin_run("exp", "b");

    for (int64_t s = 0; s < 5; ++s) {
        store.scalar(a, "acc", s, 1.0 + s);
        store.scalar(b, "acc", s, 100.0 + s);
    }

    auto sa = store.scalars(a, "acc");
    auto sb = store.scalars(b, "acc");
    REQUIRE(sa.size() == 5);
    REQUIRE(sb.size() == 5);
    for (int i = 0; i < 5; ++i) {
        CHECK(sa[i].second == doctest::Approx(1.0 + i));
        CHECK(sb[i].second == doctest::Approx(100.0 + i));
    }
}

TEST_CASE("metrics: scalar_tags lists exactly the written tags") {
    MetricsStore store;
    REQUIRE(store.open(":memory:"));
    uint64_t run = store.begin_run("exp", "tags");
    store.scalar(run, "loss", 0, 0.1);
    store.scalar(run, "loss", 1, 0.2);
    store.scalar(run, "acc", 0, 0.9);

    auto tags = store.scalar_tags(run);
    std::sort(tags.begin(), tags.end());
    REQUIRE(tags.size() == 2);
    CHECK(tags[0] == "acc");
    CHECK(tags[1] == "loss");
}

TEST_CASE("metrics: hparams_json round-trips via runs()") {
    MetricsStore store;
    REQUIRE(store.open(":memory:"));
    uint64_t run = store.begin_run("exp", "hp");
    const std::string json = R"({"lr":0.001,"batch":32,"note":"o'brien"})";
    store.hparams_json(run, json.c_str());

    auto runs = store.runs();
    bool found = false;
    for (const auto& r : runs) {
        if (r.id == run) {
            CHECK(r.hparams == json);
            found = true;
        }
    }
    CHECK(found);
}

TEST_CASE("metrics: end_run flips done") {
    MetricsStore store;
    REQUIRE(store.open(":memory:"));
    uint64_t run = store.begin_run("exp", "e");

    auto done_flag = [&](uint64_t id) {
        for (const auto& r : store.runs()) if (r.id == id) return r.done;
        return false;
    };
    CHECK(done_flag(run) == false);
    store.end_run(run);
    CHECK(done_flag(run) == true);
}

TEST_CASE("metrics: unknown-run calls are inert (no throw, empty queries)") {
    MetricsStore store;
    REQUIRE(store.open(":memory:"));
    const uint64_t ghost = 999999;

    // None of these should throw or corrupt state.
    CHECK_NOTHROW(store.scalar(ghost, "loss", 0, 1.0));
    CHECK_NOTHROW(store.end_run(ghost));
    CHECK_NOTHROW(store.hparams_json(ghost, "{}"));
    float f[4] = {1, 2, 3, 4};
    CHECK_NOTHROW(store.histogram(ghost, "h", 0, f, 4));

    // Queries against an unknown run come back empty.
    CHECK(store.scalars(ghost, "loss").empty());
    CHECK(store.scalar_tags(ghost).empty());
    CHECK(store.histograms(ghost, "h").empty());
}

TEST_CASE("metrics: histogram blob survives — write 64 floats, count them back") {
    MetricsStore store;
    REQUIRE(store.open(":memory:"));
    uint64_t run = store.begin_run("exp", "hist");

    std::vector<float> vals(64);
    std::iota(vals.begin(), vals.end(), 0.0f);
    store.histogram(run, "grad", 7, vals.data(), 64);

    auto hs = store.histograms(run, "grad");
    REQUIRE(hs.size() == 1);
    CHECK(hs[0].step == 7);
    CHECK(hs[0].count == 64);
    // The blob is 64 floats — 256 bytes — persisted intact.
    CHECK(hs[0].byte_length == 64 * sizeof(float));
}

TEST_CASE("metrics: threaded smoke — 4 threads x 500 scalars on distinct tags") {
    MetricsStore store;
    REQUIRE(store.open(":memory:"));
    uint64_t run = store.begin_run("exp", "threads");

    constexpr int T = 4;
    constexpr int PER = 500;
    // Start barrier: every writer parks until all T threads are up, so their
    // scalar() calls actually overlap and the test exercises real contention
    // instead of accidentally serializing.
    std::atomic<int> ready{0};
    std::vector<std::thread> threads;
    for (int t = 0; t < T; ++t) {
        threads.emplace_back([&store, &ready, run, t]() {
            std::string tag = "t" + std::to_string(t);
            ready.fetch_add(1);
            while (ready.load() != T) { /* spin until all threads are ready */ }
            for (int64_t s = 0; s < PER; ++s) {
                store.scalar(run, tag.c_str(), s, static_cast<double>(t * 1000 + s));
            }
        });
    }
    for (auto& th : threads) th.join();

    // Every tag has exactly PER step-ordered points; total count is T*PER.
    auto tags = store.scalar_tags(run);
    CHECK(tags.size() == T);
    int total = 0;
    for (const auto& tag : tags) {
        auto s = store.scalars(run, tag);
        CHECK(s.size() == PER);
        for (int i = 0; i < PER; ++i) CHECK(s[i].first == i);  // step-ordered
        total += static_cast<int>(s.size());
    }
    CHECK(total == T * PER);
}
