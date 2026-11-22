#pragma once
// The catalog: the providers that each origin implements, and Catalog, which
// merges them (design/catalog.md).
//
// A CatalogProvider lists, describes and samples the sources of one origin
// (loader outputs, live views, feed channels, metrics, or an in-memory set for
// tests). Catalog holds up to 32 providers, each with a unique name and a
// priority, and asks them on every call; it keeps no cache. It checks every
// entry a provider gives with the strict rules of catalog_entry.h and drops a
// faulty one with a warning; it gives a duplicate id to the provider with the
// higher priority, then to the one added first, and reports the duplicate; a
// provider that fails or throws adds a warning and the others still answer.
// Catalog::sample checks the request's bounds before it asks a provider and
// cuts what comes back to the limits.
//
// Catalog is thread-safe. A provider must be safe to call from any thread.
// No function here throws.
#include <caliper_builder/catalog/catalog_entry.h>
#include <caliper_builder/error.h>

#include <nlohmann/json.hpp>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace caliper_builder {

struct CatalogFilter {
    std::optional<CatalogKind> kind;
    std::string prefix;       // matched against the start of the name, case-sensitive
    std::size_t limit = 100;  // 1 to 1000
};

struct CatalogSampleRequest {
    std::size_t limit = 20;            // rows of a tabular source, 1 to 1000
    std::vector<std::int64_t> start;   // per axis of a tensor; empty means zeros
    std::vector<std::int64_t> count;   // per axis of a tensor; empty means the default box
};

// The bounds of samples and lists.
struct CatalogSampleLimits {
    static constexpr std::size_t max_list = 1000;
    static constexpr std::size_t default_list = 100;
    static constexpr std::size_t max_rows = 1000;
    static constexpr std::size_t default_rows = 20;
    static constexpr std::size_t max_cell_bytes = 1024;
    static constexpr std::size_t max_sample_json = 4u << 20;  // 4 MiB
    static constexpr std::int64_t max_slice_axis = 64;
    static constexpr std::int64_t max_slice_values = 4096;
    static constexpr std::uint64_t max_stats_elements = std::uint64_t(1) << 26;
    static constexpr std::size_t max_providers = 32;
};

struct CatalogTensorStats {
    std::uint64_t count = 0;      // elements of the tensor
    std::uint64_t sampled = 0;    // elements read
    bool exact = true;            // sampled == count
    std::uint64_t nonfinite = 0;  // NaN and infinities among those read
    std::optional<double> min, max, mean, std;  // over the finite values read
};

struct CatalogSlice {
    std::vector<std::int64_t> start;
    std::vector<std::int64_t> shape;
    nlohmann::json values = nlohmann::json::array();  // row-major
};

// One sample. A tabular source fills columns, rows, total_rows, truncated and
// cut_cells; a tensor-shaped source fills dtype, shape, stats and slice.
struct CatalogSample {
    std::string id;
    CatalogKind kind = CatalogKind::table;
    std::optional<std::vector<CatalogColumn>> columns;
    std::optional<nlohmann::json> rows;  // a list of rows, each a list of cells
    std::optional<std::uint64_t> total_rows;
    bool truncated = false;
    std::uint64_t cut_cells = 0;
    std::optional<CatalogDtype> dtype;
    std::optional<std::vector<std::int64_t>> shape;
    std::optional<CatalogTensorStats> stats;
    std::optional<CatalogSlice> slice;
};

nlohmann::json catalog_sample_to_json(const CatalogSample& sample);

struct CatalogListResult {
    std::vector<CatalogEntry> entries;
    bool truncated = false;
    Error error;
};

struct CatalogDescribeResult {
    std::optional<CatalogEntry> entry;  // empty with no error: the id is not there
    Error error;
};

struct CatalogSampleResult {
    CatalogSample sample;
    Error error;
};

class CatalogProvider {
public:
    virtual ~CatalogProvider() = default;
    // The matching entries, sorted by id, at most filter.limit of them, with
    // `truncated` set when there are more.
    virtual CatalogListResult list(const CatalogFilter& filter) = 0;
    virtual CatalogDescribeResult describe(const std::string& id) = 0;
    virtual CatalogSampleResult sample(const std::string& id, const CatalogSampleRequest& request) = 0;
    // Increases when an entry is added or removed or its schema changes.
    virtual std::uint64_t revision() const = 0;
};

struct CatalogDuplicate {
    std::string id;
    std::string kept;                 // the provider whose entry is shown
    std::vector<std::string> hidden;  // the providers whose entries are hidden
};

struct CatalogMergedList {
    std::vector<CatalogEntry> entries;
    bool truncated = false;
    std::vector<std::string> warnings;
    std::vector<CatalogDuplicate> duplicates;
    Error error;  // only for a filter out of bounds
};

// Checks a filter: limit 1 to 1000, a prefix of at most 256 bytes of UTF-8.
Error check_catalog_filter(const CatalogFilter& filter);

// Checks a sample request: limit 1 to 1000; start and count empty or one
// entry per axis of `shape` (when the source is tensor-shaped) within bounds.
Error check_catalog_sample_request(const CatalogSampleRequest& request,
                                   const std::vector<std::int64_t>* shape);

class Catalog {
public:
    Catalog() = default;
    Catalog(const Catalog&) = delete;
    Catalog& operator=(const Catalog&) = delete;

    // Adds a provider. Fails with catalog.bad_provider for a null provider, a
    // bad or taken name, or a 33rd provider.
    Error add_provider(std::string name, std::shared_ptr<CatalogProvider> provider, int priority = 0);
    bool remove_provider(std::string_view name);
    std::vector<std::string> provider_names() const;  // in resolution order

    CatalogMergedList list(const CatalogFilter& filter) const;
    // The entry, or catalog.invalid_id, catalog.not_found or
    // catalog.provider_failed.
    CatalogDescribeResult describe(std::string_view id) const;
    CatalogSampleResult sample(std::string_view id, const CatalogSampleRequest& request) const;

    // The sum of the providers' revisions (kept for removed ones) plus the
    // count of providers added and removed; it only grows.
    std::uint64_t revision() const;

private:
    struct Slot {
        std::string name;
        std::shared_ptr<CatalogProvider> provider;
        int priority = 0;
        std::uint64_t order = 0;
    };
    std::vector<Slot> snapshot() const;  // in resolution order

    mutable std::mutex mutex_;
    std::vector<Slot> slots_;
    std::uint64_t next_order_ = 0;
    std::uint64_t changes_ = 0;
    std::uint64_t retired_ = 0;  // the revisions of removed providers
};

}  // namespace caliper_builder
