// metrics_catalog.h: the metrics of caliper.metrics.v1 as sources of the
// builder's catalog (design/catalog.md, the kind `metric`), and their data for
// the session's resolve (caliper_builder::CatalogDataSource). Host-internal;
// built only with CALIPER_WITH_BUILDER.
//
// One source per run and scalar tag of the MetricsStore: the id is
// `metric:<run id>/<tag>`, the title `<run name> <tag>`, the columns
// `step BIGINT` and `value DOUBLE`, `rows` the number of points, and the
// origin {service "metrics", experiment, run, run_name, tag}. A run or tag
// whose id would break the catalog's name rules (a control character, more
// than 256 bytes) is not listed. `rate_hz`, `bytes` and `updated_at` are null:
// points arrive per training step and the store records no time.
//
// THE SUMMARY. A thread of this object keeps a summary of the store's series
// (one GROUP BY and the run list), refreshed when the store's revision()
// changed and at most every `refresh_ms`, so list(), describe(), revision()
// and generation() never touch the store: they read the summary under a short
// mutex and may run on the frame thread. A series' generation is the store
// revision at the refresh that first saw its current point count and last
// step, so it changes when that series gains points, not when another does,
// up to `refresh_ms` after the write. revision() grows when a series appears
// or goes (a deleted run, clear_all), not when points arrive.
//
// THE READS. sample() and read() query the store: the newest `limit + 1`
// points by step, one prepared statement under the store's mutex, so a call
// waits for a write or a query that holds it (a scalar insert is
// microseconds; the summary's GROUP BY is the longest, a few milliseconds for
// a million points). sample() reads at most 1001 points; read() at most the
// request's max_rows (100,000 by default), which costs milliseconds and
// should not run on the frame thread every frame: the spec runtime reads
// again only when generation() changed. A series whose run was deleted is
// not found by describe() and answers resolve.unavailable from read().
//
// Thread-safe; no function throws.
#pragma once

#include <caliper_builder/catalog/catalog.h>
#include <caliper_builder/session/resolve.h>

#include <condition_variable>
#include <cstdint>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <thread>

namespace caliper_host {

class MetricsStore;

class MetricsCatalog : public caliper_builder::CatalogProvider, public caliper_builder::CatalogDataSource {
public:
    struct Options {
        int refresh_ms = 250;   // the summary is refreshed at most this often
        bool thread = true;     // false: no thread; refresh_now() refreshes (tests)
    };
    explicit MetricsCatalog(MetricsStore& store);
    MetricsCatalog(MetricsStore& store, Options options);
    ~MetricsCatalog() override;   // stops and joins the thread
    MetricsCatalog(const MetricsCatalog&) = delete;
    MetricsCatalog& operator=(const MetricsCatalog&) = delete;

    // Refreshes the summary on the calling thread if the store changed since
    // the last refresh.
    void refresh_now();

    // CatalogProvider
    caliper_builder::CatalogListResult list(const caliper_builder::CatalogFilter& filter) override;
    caliper_builder::CatalogDescribeResult describe(const std::string& id) override;
    caliper_builder::CatalogSampleResult sample(const std::string& id,
                                                const caliper_builder::CatalogSampleRequest& request) override;
    std::uint64_t revision() const override;

    // CatalogDataSource
    std::uint64_t generation(const std::string& id) override;
    caliper_builder::ServedData read(const std::string& id, const caliper_builder::ServedRequest& request) override;

private:
    struct Series {
        caliper_builder::CatalogEntry entry;
        std::uint64_t run = 0;
        std::string tag;
        std::int64_t points = 0;
        std::int64_t last_step = 0;
        std::uint64_t generation = 0;
    };
    void run_thread();
    bool find(const std::string& id, Series* out) const;

    MetricsStore& store_;
    Options options_;
    std::mutex refresh_mutex_;              // one refresh at a time
    mutable std::mutex mutex_;              // guards the summary
    std::map<std::string, Series> series_;  // by id
    std::uint64_t seen_ = 0;                // the store revision of the last refresh (0: never)
    std::uint64_t revision_ = 1;
    std::mutex stop_mutex_;
    std::condition_variable stop_cv_;
    bool stop_ = false;
    std::thread thread_;
};

}  // namespace caliper_host
