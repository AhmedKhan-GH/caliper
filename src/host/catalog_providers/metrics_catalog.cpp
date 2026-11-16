#include "catalog_providers/metrics_catalog.h"

#include "catalog_providers/catalog_util.h"
#include "metrics_store.h"

#include <algorithm>
#include <chrono>
#include <exception>
#include <unordered_map>

namespace caliper_host {

using namespace caliper_builder;
using catalog_detail::float_cell;
using catalog_detail::origin_value;

namespace {

std::vector<CatalogColumn> metric_columns() { return {{"step", "BIGINT"}, {"value", "DOUBLE"}}; }

}  // namespace

MetricsCatalog::MetricsCatalog(MetricsStore& store) : MetricsCatalog(store, Options{}) {}

MetricsCatalog::MetricsCatalog(MetricsStore& store, Options options) : store_(store), options_(options) {
    if (options_.thread) thread_ = std::thread([this] { run_thread(); });
}

MetricsCatalog::~MetricsCatalog() {
    {
        std::lock_guard<std::mutex> lock(stop_mutex_);
        stop_ = true;
    }
    stop_cv_.notify_all();
    if (thread_.joinable()) thread_.join();
}

void MetricsCatalog::run_thread() {
    for (;;) {
        refresh_now();
        std::unique_lock<std::mutex> lock(stop_mutex_);
        stop_cv_.wait_for(lock, std::chrono::milliseconds(std::max(1, options_.refresh_ms)), [this] { return stop_; });
        if (stop_) return;
    }
}

void MetricsCatalog::refresh_now() {
    try {
        std::lock_guard<std::mutex> one(refresh_mutex_);
        const std::uint64_t rev = store_.revision();
        {
            std::lock_guard<std::mutex> lock(mutex_);
            if (rev == seen_) return;
        }
        std::unordered_map<std::uint64_t, RunInfo> runs;
        for (RunInfo& r : store_.runs()) runs.emplace(r.id, std::move(r));
        const std::vector<ScalarSeriesInfo> all = store_.scalar_series();

        std::map<std::string, Series> next;
        {
            std::lock_guard<std::mutex> lock(mutex_);
            for (const ScalarSeriesInfo& s : all) {
                const std::string name = std::to_string(s.run) + "/" + s.tag;
                if (!catalog_detail::is_catalog_name(name)) continue;
                Series one;
                one.run = s.run;
                one.tag = s.tag;
                one.points = s.points;
                one.last_step = s.last_step;
                CatalogEntry& e = one.entry;
                e.kind = CatalogKind::metric;
                e.name = name;
                e.id = make_catalog_id(e.kind, name);
                const auto run = runs.find(s.run);
                const std::string run_name = run != runs.end() ? run->second.name : std::string();
                e.title = catalog_detail::title_or(run_name.empty() ? s.tag : run_name + " " + s.tag, name);
                e.origin.service = "metrics";
                if (run != runs.end()) e.origin.experiment = origin_value(run->second.experiment);
                e.origin.run = static_cast<std::int64_t>(s.run);
                e.origin.run_name = origin_value(run_name);
                e.origin.tag = origin_value(s.tag);
                e.columns = metric_columns();
                e.rows = static_cast<std::uint64_t>(std::max<std::int64_t>(0, s.points));
                e.wire = false;
                const auto old = series_.find(e.id);
                one.generation = old != series_.end() && old->second.points == s.points &&
                                         old->second.last_step == s.last_step
                                     ? old->second.generation
                                     : rev;
                next.emplace(e.id, std::move(one));
            }
            bool same_ids = next.size() == series_.size();
            if (same_ids) {
                auto a = next.begin();
                for (auto b = series_.begin(); b != series_.end(); ++a, ++b)
                    if (a->first != b->first) {
                        same_ids = false;
                        break;
                    }
            }
            if (!same_ids) ++revision_;
            series_ = std::move(next);
            seen_ = rev;
        }
    } catch (const std::exception&) {
        // A failed refresh keeps the last summary; the next one tries again.
    }
}

bool MetricsCatalog::find(const std::string& id, Series* out) const {
    std::lock_guard<std::mutex> lock(mutex_);
    const auto it = series_.find(id);
    if (it == series_.end()) return false;
    *out = it->second;
    return true;
}

CatalogListResult MetricsCatalog::list(const CatalogFilter& filter) {
    std::vector<CatalogEntry> all;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        if (filter.kind && *filter.kind != CatalogKind::metric) return {};
        all.reserve(series_.size());
        for (const auto& kv : series_) all.push_back(kv.second.entry);
    }
    return catalog_detail::filter_entries(all, filter);
}

CatalogDescribeResult MetricsCatalog::describe(const std::string& id) {
    CatalogDescribeResult out;
    Series s;
    if (find(id, &s)) out.entry = std::move(s.entry);
    return out;
}

CatalogSampleResult MetricsCatalog::sample(const std::string& id, const CatalogSampleRequest& request) {
    CatalogSampleResult out;
    Series s;
    if (!find(id, &s)) {
        out.error = Error("catalog.not_found", "No metric has the id " + id + ".", "id");
        return out;
    }
    const std::size_t limit = std::max<std::size_t>(1, request.limit);
    std::vector<std::pair<int64_t, double>> points;
    try {
        points = store_.scalars(s.run, s.tag, static_cast<std::uint64_t>(limit) + 1);
    } catch (const std::exception& e) {
        out.error = Error("catalog.provider_failed", std::string("The metrics store failed: ") + e.what(), "id");
        return out;
    }
    if (points.empty()) {
        out.error = Error("catalog.no_data", "The metric " + id + " has no points now; its run may be gone.", "id");
        return out;
    }
    const bool cut = points.size() > limit;
    nlohmann::json rows = nlohmann::json::array();
    for (std::size_t i = cut ? 1 : 0; i < points.size(); ++i)
        rows.push_back(nlohmann::json::array({points[i].first, float_cell(points[i].second)}));
    CatalogSample& sample = out.sample;
    sample.id = id;
    sample.kind = CatalogKind::metric;
    sample.columns = metric_columns();
    sample.total_rows = std::max<std::uint64_t>(static_cast<std::uint64_t>(std::max<std::int64_t>(0, s.points)),
                                                rows.size() + (cut ? 1 : 0));
    sample.rows = std::move(rows);
    sample.truncated = cut;
    return out;
}

std::uint64_t MetricsCatalog::revision() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return revision_;
}

std::uint64_t MetricsCatalog::generation(const std::string& id) {
    std::lock_guard<std::mutex> lock(mutex_);
    const auto it = series_.find(id);
    return it == series_.end() ? 0 : it->second.generation;
}

ServedData MetricsCatalog::read(const std::string& id, const ServedRequest& request) {
    ServedData out;
    Series s;
    if (!find(id, &s)) {
        out.error = Error("resolve.unavailable",
                          "No metric has the id " + id + ": its run or tag is gone, or it has no points yet.");
        return out;
    }
    const std::uint64_t n = std::max<std::uint64_t>(
        1, std::min<std::uint64_t>(request.max_rows, request.max_cells / metric_columns().size()));
    std::vector<std::pair<int64_t, double>> points;
    try {
        points = store_.scalars(s.run, s.tag, n + 1);
    } catch (const std::exception& e) {
        out.error = Error("resolve.unavailable", std::string("The metrics store failed: ") + e.what());
        return out;
    }
    if (points.empty()) {
        out.error = Error("resolve.unavailable", "The metric " + id + " has no points now; its run may be gone.");
        return out;
    }
    const bool cut = points.size() > n;
    out.tabular = true;
    out.generation = s.generation;
    out.columns = {{"step", "BIGINT"}, {"value", "DOUBLE"}};
    for (std::size_t i = cut ? 1 : 0; i < points.size(); ++i)
        out.rows.push_back(nlohmann::json::array({points[i].first, float_cell(points[i].second)}));
    out.truncated = cut;
    return out;
}

}  // namespace caliper_host
