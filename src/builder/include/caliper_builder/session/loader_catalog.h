#pragma once
// LoaderCatalogProvider: the catalog provider for the outputs of loaders, a
// thin adapter over a LoaderStore (design/builder-session.md, The
// catalog). It lists, describes and samples the `table` and `tensor` sources
// that loader runs recorded in caliper.sources.
//
// - list() asks the store for the matching sources, one more than the limit,
//   so that it can tell whether more exist; other kinds give nothing.
// - describe() turns a LoadedSource into a CatalogEntry: `origin` holds the
//   loader, the file (cut to 1024 bytes at a character boundary) and, for a
//   tensor, its name in the file; at most 16 warnings of at most 1024 bytes are
//   kept; an `updated_at` that is not a UTC timestamp becomes null.
// - sample() reads a table's first rows through the store's bounded read-only
//   query (with NaN and infinities of floating-point columns written nan, inf
//   and -inf) and a tensor's statistics and slice from its mapped bytes with
//   sample_tensor().
// - revision() grows when touch() is called; the session calls it after every
//   command that runs or changes a loader, which is the only way the store's
//   sources change.
//
// Thread-safe as the store is. No function here throws.
#include <caliper_builder/catalog/catalog.h>
#include <caliper_builder/loaders/loaded_source.h>
#include <caliper_builder/loaders/store.h>

#include <atomic>
#include <cstdint>
#include <string>

namespace caliper_builder {

class LoaderCatalogProvider : public CatalogProvider {
public:
    explicit LoaderCatalogProvider(LoaderStore& store);

    CatalogListResult list(const CatalogFilter& filter) override;
    CatalogDescribeResult describe(const std::string& id) override;
    CatalogSampleResult sample(const std::string& id, const CatalogSampleRequest& request) override;
    std::uint64_t revision() const override;

    // Marks that the store's sources may have changed.
    void touch();

    // The catalog entry of a loader output.
    static CatalogEntry entry_of(const LoadedSource& source);

private:
    LoaderStore& store_;
    std::atomic<std::uint64_t> revision_{1};
};

// The text of an SQL identifier in double quotes, with quotes doubled.
std::string quote_sql_identifier(const std::string& name);

}  // namespace caliper_builder
