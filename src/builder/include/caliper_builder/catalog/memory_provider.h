#pragma once
// InMemoryCatalogProvider: a catalog provider that holds its entries, and
// optionally their data, in memory. It serves tests and examples, and shows
// how a provider meets the contract of catalog.h.
//
// put() adds or replaces an entry with no data, so a sample of it answers
// catalog.no_data. put_table() adds a tabular entry with its rows (a list of
// rows, each a list of one cell per column); a table gives its first rows, a
// feed or a metric its last ones. put_tensor() adds a tensor-shaped entry with
// its little-endian bytes, which must be exactly the shape's size. Each put
// checks the entry with the strict rules and refuses a faulty one. The
// revision grows when an entry is added or removed or its schema changes.
// fail_with() makes every later call fail with an error, for tests of a
// failing provider. Thread-safe; no function throws.
#include <caliper_builder/catalog/catalog.h>

#include <map>
#include <mutex>
#include <optional>
#include <string>
#include <vector>

namespace caliper_builder {

class InMemoryCatalogProvider : public CatalogProvider {
public:
    Error put(CatalogEntry entry);
    Error put_table(CatalogEntry entry, nlohmann::json rows);
    Error put_tensor(CatalogEntry entry, std::vector<std::uint8_t> bytes);
    bool remove(const std::string& id);
    void fail_with(Error error);  // an empty Error clears the failure
    std::size_t size() const;

    CatalogListResult list(const CatalogFilter& filter) override;
    CatalogDescribeResult describe(const std::string& id) override;
    CatalogSampleResult sample(const std::string& id, const CatalogSampleRequest& request) override;
    std::uint64_t revision() const override;

private:
    struct Item {
        CatalogEntry entry;
        std::optional<nlohmann::json> rows;
        std::optional<std::vector<std::uint8_t>> bytes;
    };
    Error store(Item item);

    mutable std::mutex mutex_;
    std::map<std::string, Item> items_;
    std::uint64_t revision_ = 0;
    Error failure_;
};

}  // namespace caliper_builder
