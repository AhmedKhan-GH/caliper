#include <caliper_builder/catalog/memory_provider.h>
#include <caliper_builder/catalog/tensor_sample.h>

#include <utility>

namespace caliper_builder {

namespace {

bool same_schema(const CatalogEntry& a, const CatalogEntry& b) {
    auto columns_equal = [](const auto& x, const auto& y) {
        if (x.has_value() != y.has_value()) return false;
        if (!x) return true;
        if (x->size() != y->size()) return false;
        for (std::size_t i = 0; i < x->size(); ++i)
            if ((*x)[i].name != (*y)[i].name || (*x)[i].type != (*y)[i].type) return false;
        return true;
    };
    return a.kind == b.kind && a.dtype == b.dtype && a.shape == b.shape && columns_equal(a.columns, b.columns);
}

}  // namespace

Error InMemoryCatalogProvider::store(Item item) {
    if (Error err = validate_catalog_entry(item.entry)) return err;
    std::lock_guard<std::mutex> lock(mutex_);
    const auto it = items_.find(item.entry.id);
    if (it == items_.end() || !same_schema(it->second.entry, item.entry)) ++revision_;
    const std::string id = item.entry.id;
    items_[id] = std::move(item);
    return {};
}

Error InMemoryCatalogProvider::put(CatalogEntry entry) {
    Item item;
    item.entry = std::move(entry);
    return store(std::move(item));
}

Error InMemoryCatalogProvider::put_table(CatalogEntry entry, nlohmann::json rows) {
    if (!catalog_kind_is_tabular(entry.kind))
        return Error("catalog.bad_request", "Rows belong to a tabular entry.", "kind");
    if (!entry.columns) return Error("catalog.invalid_entry", "A tabular entry must list its columns.", "columns");
    if (!rows.is_array()) return Error("catalog.bad_request", "The rows are a list of rows.", "rows");
    for (std::size_t i = 0; i < rows.size(); ++i)
        if (!rows[i].is_array() || rows[i].size() != entry.columns->size())
            return Error("catalog.bad_request", "Each row is a list of one cell for each column.",
                         "rows[" + std::to_string(i) + "]");
    Item item;
    item.entry = std::move(entry);
    item.rows = std::move(rows);
    return store(std::move(item));
}

Error InMemoryCatalogProvider::put_tensor(CatalogEntry entry, std::vector<std::uint8_t> bytes) {
    if (catalog_kind_is_tabular(entry.kind))
        return Error("catalog.bad_request", "Bytes belong to a tensor-shaped entry.", "kind");
    if (!entry.dtype || !entry.shape)
        return Error("catalog.invalid_entry", "A tensor with data gives its dtype and shape.", "shape");
    if (Error err = validate_catalog_entry(entry)) return err;  // bounds the shape's product
    std::uint64_t elements = 1;
    for (const auto d : *entry.shape) elements *= static_cast<std::uint64_t>(d);
    const std::uint64_t dsize = catalog_dtype_size(*entry.dtype);
    if (elements > UINT64_MAX / dsize || elements * dsize != bytes.size())
        return Error("catalog.bad_tensor", "The bytes do not match the entry's shape and dtype.", "bytes");
    Item item;
    item.entry = std::move(entry);
    item.bytes = std::move(bytes);
    return store(std::move(item));
}

bool InMemoryCatalogProvider::remove(const std::string& id) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (items_.erase(id) == 0) return false;
    ++revision_;
    return true;
}

void InMemoryCatalogProvider::fail_with(Error error) {
    std::lock_guard<std::mutex> lock(mutex_);
    failure_ = std::move(error);
}

std::size_t InMemoryCatalogProvider::size() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return items_.size();
}

CatalogListResult InMemoryCatalogProvider::list(const CatalogFilter& filter) {
    CatalogListResult out;
    std::lock_guard<std::mutex> lock(mutex_);
    if (failure_) {
        out.error = failure_;
        return out;
    }
    for (const auto& kv : items_) {  // a std::map, so sorted by id
        const CatalogEntry& e = kv.second.entry;
        if (filter.kind && e.kind != *filter.kind) continue;
        if (e.name.compare(0, filter.prefix.size(), filter.prefix) != 0) continue;
        if (out.entries.size() >= filter.limit) {
            out.truncated = true;
            break;
        }
        out.entries.push_back(e);
    }
    return out;
}

CatalogDescribeResult InMemoryCatalogProvider::describe(const std::string& id) {
    CatalogDescribeResult out;
    std::lock_guard<std::mutex> lock(mutex_);
    if (failure_) {
        out.error = failure_;
        return out;
    }
    const auto it = items_.find(id);
    if (it != items_.end()) out.entry = it->second.entry;
    return out;
}

CatalogSampleResult InMemoryCatalogProvider::sample(const std::string& id, const CatalogSampleRequest& request) {
    CatalogSampleResult out;
    std::lock_guard<std::mutex> lock(mutex_);
    if (failure_) {
        out.error = failure_;
        return out;
    }
    const auto it = items_.find(id);
    if (it == items_.end()) {
        out.error = Error("catalog.not_found", "No source has the id " + id + ".", "id");
        return out;
    }
    const Item& item = it->second;
    CatalogSample& s = out.sample;
    s.id = id;
    s.kind = item.entry.kind;
    if (catalog_kind_is_tabular(item.entry.kind)) {
        if (!item.rows) {
            out.error = Error("catalog.no_data", "The source " + id + " holds no rows here.", "id");
            return out;
        }
        const nlohmann::json& rows = *item.rows;
        const std::size_t n = rows.size();
        const std::size_t take = n < request.limit ? n : request.limit;
        // A table gives its first rows; a feed or a metric its newest ones.
        const std::size_t first = item.entry.kind == CatalogKind::table ? 0 : n - take;
        nlohmann::json kept = nlohmann::json::array();
        for (std::size_t i = first; i < first + take; ++i) kept.push_back(rows[i]);
        s.columns = item.entry.columns;
        s.rows = std::move(kept);
        s.total_rows = n;
        s.truncated = take < n;
        return out;
    }
    if (!item.bytes || !item.entry.dtype || !item.entry.shape) {
        out.error = Error("catalog.no_data", "The source " + id + " has published nothing yet.", "id");
        return out;
    }
    TensorSampleResult t = sample_tensor(*item.entry.dtype, *item.entry.shape,
                                         item.bytes->empty() ? nullptr : item.bytes->data(), item.bytes->size(),
                                         request.start, request.count);
    if (t.error) {
        out.error = std::move(t.error);
        return out;
    }
    s.dtype = item.entry.dtype;
    s.shape = item.entry.shape;
    s.stats = std::move(t.stats);
    s.slice = std::move(t.slice);
    return out;
}

std::uint64_t InMemoryCatalogProvider::revision() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return revision_;
}

}  // namespace caliper_builder
