// Tests for caliper_host::MetricsStore — the DuckDB-backed run/tag/step store
// behind caliper.metrics.v1. TDD: these cases are written before the
// implementation. The §16 contract (10,000 shuffled scalars back strictly
// step-ascending with matching values) is the load-bearing guarantee.
#include <doctest/doctest.h>

#include "metrics_store.h"
#include "duckdb_sizing.h"

#include <caliper/arrow_c.h>

#include <algorithm>
#include <filesystem>
#ifdef _WIN32
#include <process.h>
#define getpid _getpid
// POSIX setenv/unsetenv don't exist on MSVC; _putenv_s with "" removes.
#define setenv(k, v, overwrite) _putenv_s(k, v)
#define unsetenv(k) _putenv_s(k, "")
#else
#include <unistd.h>
#endif
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

TEST_CASE("metrics_store: a throwing DuckDB open degrades to closed, never terminates") {
    // The DuckDB constructor throws on unopenable paths — including the
    // IOException a second Caliper instance gets when the first holds the
    // file lock (cross-process; simulated here with a directory path, which
    // throws through the same constructor). open() must return false and
    // leave an inert store; an uncaught exception here terminated the whole
    // host before this guard.
    auto dir = std::filesystem::temp_directory_path() /
               ("caliper-metrics-dir-" + std::to_string(::getpid()));
    std::filesystem::create_directories(dir);
    MetricsStore store;
    CHECK_FALSE(store.open(dir.string()));    // path is a directory: throws
    CHECK(store.begin_run("e", "r") == 0);    // and inert, not UB
    std::error_code ec;
    std::filesystem::remove_all(dir, ec);
}

TEST_CASE("metrics_store: delete_run removes the run and all its series") {
    MetricsStore store;
    REQUIRE(store.open(":memory:"));
    uint64_t a = store.begin_run("exp", "keep");
    uint64_t b = store.begin_run("exp", "drop");
    store.scalar(a, "loss", 0, 1.0); store.scalar(b, "loss", 0, 2.0);
    store.delete_run(b);
    auto runs = store.runs();
    REQUIRE(runs.size() == 1);
    CHECK(runs[0].id == a);
    CHECK(store.scalars(b, "loss").empty());
    CHECK(store.scalars(a, "loss").size() == 1);
    // deleted run id is inert for future writes, not resurrected
    store.scalar(b, "loss", 1, 3.0);
    CHECK(store.scalars(b, "loss").empty());
}

// ---------------------------------------------------------------------------
// caliper.metrics.v1_1 read surface: query() streams metric rows out as Arrow C
// streams against THIS store's live connection, with READ-ONLY enforced (only a
// lone SELECT is accepted — the connection is the live writer). TDD: written
// before the implementation.
// ---------------------------------------------------------------------------
namespace {

// Drain an ArrowArrayStream of one BIGINT (int64, format "l") column + one
// DOUBLE (format "g") column into vectors — the (step, value) shape a Compass
// scalars query returns.
struct StepValue {
    std::vector<int64_t> step;
    std::vector<double>  value;
    int64_t schema_children = -1;
};

StepValue drain_i64_f64(ArrowArrayStream* stream) {
    StepValue out;
    ArrowSchema schema = {};
    REQUIRE(stream->get_schema(stream, &schema) == 0);
    out.schema_children = schema.n_children;
    if (schema.release) schema.release(&schema);

    for (;;) {
        ArrowArray array = {};
        REQUIRE(stream->get_next(stream, &array) == 0);
        if (!array.release) break;   // end of stream (spec)
        REQUIRE(array.n_children >= 2);
        const ArrowArray* cs = array.children[0];
        const ArrowArray* cv = array.children[1];
        const int64_t* sp = static_cast<const int64_t*>(cs->buffers[1]);
        const double*  vp = static_cast<const double*>(cv->buffers[1]);
        for (int64_t i = 0; i < array.length; i++) {
            out.step.push_back(sp[cs->offset + i]);
            out.value.push_back(vp[cv->offset + i]);
        }
        array.release(&array);
    }
    stream->release(stream);
    CHECK(stream->release == nullptr);   // release must null itself (spec)
    return out;
}

// Count rows of a single-column BIGINT result (e.g. SELECT COUNT(*)).
int64_t drain_single_i64(ArrowArrayStream* stream) {
    ArrowSchema schema = {};
    REQUIRE(stream->get_schema(stream, &schema) == 0);
    if (schema.release) schema.release(&schema);
    int64_t v = -1;
    for (;;) {
        ArrowArray array = {};
        REQUIRE(stream->get_next(stream, &array) == 0);
        if (!array.release) break;
        if (array.length > 0)
            v = static_cast<const int64_t*>(array.children[0]->buffers[1])
                    [array.children[0]->offset];
        array.release(&array);
    }
    stream->release(stream);
    return v;
}

}  // namespace

TEST_CASE("metrics.v1_1: query streams scalar rows out as Arrow, matching writes") {
    MetricsStore store;
    REQUIRE(store.open(":memory:"));
    uint64_t run = store.begin_run("compass", "q");
    REQUIRE(run != 0);
    for (int64_t s = 0; s < 6; ++s) store.scalar(run, "loss", s, static_cast<double>(s) * 0.5);

    ArrowArrayStream stream = {};
    REQUIRE(store.query(
        "SELECT step, value FROM scalars WHERE run = " + std::to_string(run) +
        " AND tag = 'loss' ORDER BY step", &stream));

    StepValue got = drain_i64_f64(&stream);
    CHECK(got.schema_children == 2);
    REQUIRE(got.step.size() == 6);
    for (int i = 0; i < 6; ++i) {
        CHECK(got.step[i] == i);
        CHECK(got.value[i] == doctest::Approx(i * 0.5));
    }
}

TEST_CASE("metrics.v1_1: query lists runs (id round-trips through Arrow)") {
    MetricsStore store;
    REQUIRE(store.open(":memory:"));
    uint64_t a = store.begin_run("exp", "alpha");
    uint64_t b = store.begin_run("exp", "beta");
    (void)a; (void)b;

    ArrowArrayStream stream = {};
    REQUIRE(store.query("SELECT COUNT(*) FROM runs", &stream));
    CHECK(drain_single_i64(&stream) == 2);
}

TEST_CASE("metrics.v1_1: a WITH/CTE SELECT is accepted (read-only)") {
    MetricsStore store;
    REQUIRE(store.open(":memory:"));
    uint64_t run = store.begin_run("exp", "cte");
    for (int64_t s = 0; s < 4; ++s) store.scalar(run, "loss", s, 1.0);

    ArrowArrayStream stream = {};
    REQUIRE(store.query(
        "WITH s AS (SELECT * FROM scalars) SELECT COUNT(*) FROM s", &stream));
    CHECK(drain_single_i64(&stream) == 4);
}

TEST_CASE("metrics.v1_1: non-SELECT is refused and the store is left intact") {
    MetricsStore store;
    REQUIRE(store.open(":memory:"));
    uint64_t run = store.begin_run("exp", "ro");
    store.scalar(run, "loss", 0, 1.0);

    // Each of these must be refused WITHOUT mutating the store (the connection
    // is the live writer — a leak here corrupts metrics). Note: a read-only
    // introspection PRAGMA (e.g. `PRAGMA database_list`) is NOT here — DuckDB's
    // parser rewrites it into a read-only SELECT, so it is legitimately a read;
    // setter pragmas parse to PRAGMA_STATEMENT and ARE rejected. Only a
    // SELECT_STATEMENT is accepted, and no SELECT can mutate data.
    for (const char* bad : {
             "INSERT INTO scalars VALUES (1,'x',0,0)",
             "UPDATE runs SET name = 'hacked'",
             "DELETE FROM scalars",
             "DROP TABLE runs",
             "CREATE TABLE evil(x INT)",
             "ALTER TABLE runs RENAME TO pwned"}) {
        ArrowArrayStream s = {};
        CHECK_FALSE(store.query(bad, &s));
        CHECK(s.release == nullptr);            // stream never filled
        CHECK_FALSE(store.last_error().empty());
    }

    // The run, its scalar, and the schema all survived every refusal.
    auto runs = store.runs();
    REQUIRE(runs.size() == 1);
    CHECK(runs[0].name == "ro");
    CHECK(store.scalars(run, "loss").size() == 1);
}

TEST_CASE("metrics.v1_1: statement chaining cannot smuggle a write past SELECT") {
    MetricsStore store;
    REQUIRE(store.open(":memory:"));
    uint64_t run = store.begin_run("exp", "chain");
    store.scalar(run, "loss", 0, 1.0);

    // DuckDB's Query() runs every statement and returns the last result, so a
    // naive "starts with SELECT" gate would execute the DROP. Parse-based
    // enforcement refuses the whole multi-statement string.
    ArrowArrayStream s = {};
    CHECK_FALSE(store.query("SELECT 1; DROP TABLE runs", &s));
    CHECK(s.release == nullptr);
    CHECK_FALSE(store.last_error().empty());
    CHECK(store.runs().size() == 1);            // runs table still there
}

TEST_CASE("metrics.v1_1: bad SQL fails with a reason, no crash; unopened is inert") {
    MetricsStore store;
    REQUIRE(store.open(":memory:"));
    ArrowArrayStream s = {};
    CHECK_FALSE(store.query("SELECT FROM nonsense syntax", &s));
    CHECK(s.release == nullptr);
    CHECK_FALSE(store.last_error().empty());

    MetricsStore closed;
    ArrowArrayStream s2 = {};
    CHECK_FALSE(closed.query("SELECT 1", &s2));   // never opened
    CHECK(s2.release == nullptr);
    CHECK_FALSE(closed.last_error().empty());
}

TEST_CASE("metrics_store: clear_all empties history but never reuses ids") {
    MetricsStore store;
    REQUIRE(store.open(":memory:"));
    uint64_t a = store.begin_run("exp", "one");
    store.scalar(a, "loss", 0, 1.0);
    store.clear_all();
    CHECK(store.runs().empty());
    CHECK(store.scalars(a, "loss").empty());
    uint64_t c = store.begin_run("exp", "after");
    CHECK(c > a);   // artifacts lineage keys on run ids — never reuse them
}

// ---------------------------------------------------------------------------
// DuckDB resource limits (duckdb_sizing.h / duckdb_config.h): each store opens
// with explicit `threads` / `memory_limit` instead of DuckDB's defaults (every
// core, 80% of RAM — per instance), sized from this machine's RAM and cores
// between a floor and a ceiling, and the environment can override them. The
// rule is pinned on synthetic machines; the stores are checked against what
// the rule computes for THIS machine, never against its actual RAM.
// ---------------------------------------------------------------------------
namespace {
using caliper_host::DuckDBLimits;
using caliper_host::DuckDBSizing;
using caliper_host::MachineResources;
using caliper_host::duckdb_default_limits;
using caliper_host::duckdb_limits_for;
using caliper_host::kArtifactsDuckDBLimits;
using caliper_host::kDataDuckDBLimits;
using caliper_host::kMetricsDuckDBLimits;

constexpr uint64_t kGiB = 1024ull * 1024 * 1024;

// A computed default is always "<n>MB" (DuckDB MB, 10^6 bytes).
int64_t limit_mb(const DuckDBLimits& l) { return std::stoll(l.memory_limit); }

int64_t setting_threads(MetricsStore& s) {
    ArrowArrayStream st = {};
    REQUIRE(s.query("SELECT CAST(current_setting('threads') AS BIGINT)", &st));
    return drain_single_i64(&st);
}
// memory_limit reads back as DuckDB's display string ("15.9 GiB");
// format_bytes renders bytes through the same formatter, so this is exact.
bool memory_limit_is(MetricsStore& s, int64_t bytes) {
    ArrowArrayStream st = {};
    REQUIRE(s.query("SELECT CAST(current_setting('memory_limit') = format_bytes(" +
                        std::to_string(bytes) + ") AS BIGINT)",
                    &st));
    return drain_single_i64(&st) == 1;
}
void clear_duckdb_env() {
    for (const char* k : {"CALIPER_DUCKDB_THREADS", "CALIPER_DUCKDB_MEMORY_LIMIT",
                          "CALIPER_DUCKDB_METRICS_THREADS",
                          "CALIPER_DUCKDB_METRICS_MEMORY_LIMIT",
                          "CALIPER_DUCKDB_DATA_THREADS",
                          "CALIPER_DUCKDB_DATA_MEMORY_LIMIT",
                          "CALIPER_DUCKDB_ARTIFACTS_THREADS",
                          "CALIPER_DUCKDB_ARTIFACTS_MEMORY_LIMIT"})
        unsetenv(k);
}
}  // namespace

TEST_CASE("duckdb sizing: memory and threads follow the machine, within bounds") {
    struct Row { const char* name; DuckDBSizing s; };
    const Row stores[] = {{"metrics", kMetricsDuckDBLimits},
                          {"artifacts", kArtifactsDuckDBLimits},
                          {"data", kDataDuckDBLimits}};
    auto mem = [](const DuckDBSizing& s, const MachineResources& m) {
        return duckdb_default_limits(s, m).memory_limit;
    };
    auto threads = [](const DuckDBSizing& s, const MachineResources& m) {
        return duckdb_default_limits(s, m).threads;
    };

    SUBCASE("an unreadable machine gets the floors") {
        CHECK(mem(kMetricsDuckDBLimits, {}) == "512MB");
        CHECK(mem(kArtifactsDuckDBLimits, {}) == "256MB");
        CHECK(mem(kDataDuckDBLimits, {}) == "2000MB");
        CHECK(threads(kMetricsDuckDBLimits, {}) == 1);
        CHECK(threads(kArtifactsDuckDBLimits, {}) == 1);
        CHECK(threads(kDataDuckDBLimits, {}) == 2);
    }
    SUBCASE("in between, memory is the store's share of RAM") {
        const MachineResources mac64{64 * kGiB, 18};
        CHECK(mem(kDataDuckDBLimits, mac64) == "17179MB");      // 25%
        CHECK(mem(kMetricsDuckDBLimits, mac64) == "1374MB");    // 2%
        CHECK(mem(kArtifactsDuckDBLimits, mac64) == "687MB");   // 1%
    }
    SUBCASE("a huge machine gets the ceilings") {
        const MachineResources big{1024 * kGiB, 256};
        CHECK(mem(kDataDuckDBLimits, big) == "32000MB");
        CHECK(mem(kMetricsDuckDBLimits, big) == "4000MB");
        CHECK(mem(kArtifactsDuckDBLimits, big) == "2000MB");
        CHECK(threads(kDataDuckDBLimits, big) == 16);
        CHECK(threads(kMetricsDuckDBLimits, big) == 4);
        CHECK(threads(kArtifactsDuckDBLimits, big) == 2);
    }
    SUBCASE("eight cores keep the 2/1/4 threads the fixed limits had") {
        const MachineResources eight{16 * kGiB, 8};
        CHECK(threads(kMetricsDuckDBLimits, eight) == 2);
        CHECK(threads(kArtifactsDuckDBLimits, eight) == 1);
        CHECK(threads(kDataDuckDBLimits, eight) == 4);
    }
    SUBCASE("every size stays in bounds and never shrinks as the machine grows") {
        for (const Row& r : stores) {
            CAPTURE(r.name);
            int64_t prev_mb = 0, prev_threads = 0;
            for (uint64_t gib = 1; gib <= 2048; gib *= 2) {
                const unsigned cores = static_cast<unsigned>(gib / 2 + 1);
                const DuckDBLimits l = duckdb_default_limits(r.s, {gib * kGiB, cores});
                CHECK(limit_mb(l) >= r.s.memory_floor_mb);
                CHECK(limit_mb(l) <= r.s.memory_ceiling_mb);
                CHECK(l.threads >= r.s.threads_floor);
                CHECK(l.threads <= r.s.threads_ceiling);
                CHECK(limit_mb(l) >= prev_mb);
                CHECK(l.threads >= prev_threads);
                prev_mb = limit_mb(l);
                prev_threads = l.threads;
            }
        }
    }
}

TEST_CASE("duckdb sizing: this machine's RAM and cores are read") {
#if defined(__APPLE__) || defined(__linux__)
    const MachineResources m = caliper_host::machine_resources();
    CHECK(m.ram_bytes >= 256ull * 1024 * 1024);   // something real, not 0
    CHECK(m.cores >= 1u);
#else
    MESSAGE("RAM query written for this platform but not exercised here");
#endif
}

TEST_CASE("duckdb sizing: overrides win over the computed defaults") {
    clear_duckdb_env();
    const DuckDBLimits defaults = duckdb_default_limits(
        kDataDuckDBLimits, caliper_host::machine_resources());
    SUBCASE("no override: the computed default") {
        const DuckDBLimits l = duckdb_limits_for("DATA", defaults);
        CHECK(l.threads == defaults.threads);
        CHECK(l.memory_limit == defaults.memory_limit);
    }
    SUBCASE("global, then per-store on top") {
        setenv("CALIPER_DUCKDB_THREADS", "3", 1);
        setenv("CALIPER_DUCKDB_MEMORY_LIMIT", "1GB", 1);
        DuckDBLimits l = duckdb_limits_for("DATA", defaults);
        CHECK(l.threads == 3);
        CHECK(l.memory_limit == "1GB");
        setenv("CALIPER_DUCKDB_DATA_THREADS", "5", 1);
        setenv("CALIPER_DUCKDB_DATA_MEMORY_LIMIT", "40GB", 1);   // above the ceiling
        l = duckdb_limits_for("DATA", defaults);
        CHECK(l.threads == 5);
        CHECK(l.memory_limit == "40GB");          // an override is not clamped
        // Another store's variables don't leak across.
        CHECK(duckdb_limits_for("METRICS", defaults).threads == 3);
    }
    SUBCASE("an unparsable thread count keeps the computed default") {
        setenv("CALIPER_DUCKDB_THREADS", "lots", 1);
        CHECK(duckdb_limits_for("DATA", defaults).threads == defaults.threads);
    }
    clear_duckdb_env();
}

TEST_CASE("metrics_store: DuckDB opens with explicit thread and memory limits") {
    clear_duckdb_env();
    const DuckDBLimits want = duckdb_default_limits(
        kMetricsDuckDBLimits, caliper_host::machine_resources());
    MetricsStore store;
    REQUIRE(store.open(":memory:"));
    CHECK(setting_threads(store) == want.threads);
    CHECK(memory_limit_is(store, limit_mb(want) * 1000000));   // not 80% of RAM
}

TEST_CASE("metrics_store: DuckDB limits honor the environment, bad values fall back") {
    clear_duckdb_env();
    const DuckDBLimits want = duckdb_default_limits(
        kMetricsDuckDBLimits, caliper_host::machine_resources());
    SUBCASE("global override") {
        setenv("CALIPER_DUCKDB_THREADS", "1", 1);
        setenv("CALIPER_DUCKDB_MEMORY_LIMIT", "256MB", 1);
        MetricsStore store;
        REQUIRE(store.open(":memory:"));
        CHECK(setting_threads(store) == 1);
        CHECK(memory_limit_is(store, 256000000));
    }
    SUBCASE("per-store override wins over the global one") {
        setenv("CALIPER_DUCKDB_THREADS", "1", 1);
        setenv("CALIPER_DUCKDB_METRICS_THREADS", "3", 1);
        setenv("CALIPER_DUCKDB_MEMORY_LIMIT", "256MB", 1);
        setenv("CALIPER_DUCKDB_METRICS_MEMORY_LIMIT", "300MB", 1);
        MetricsStore store;
        REQUIRE(store.open(":memory:"));
        CHECK(setting_threads(store) == 3);
        CHECK(memory_limit_is(store, 300000000));
    }
    SUBCASE("unparsable values fall back to the defaults, the store still opens") {
        setenv("CALIPER_DUCKDB_THREADS", "lots", 1);
        setenv("CALIPER_DUCKDB_MEMORY_LIMIT", "a bucket", 1);
        MetricsStore store;
        REQUIRE(store.open(":memory:"));
        CHECK(setting_threads(store) == want.threads);
        CHECK(memory_limit_is(store, limit_mb(want) * 1000000));
    }
    clear_duckdb_env();
}
