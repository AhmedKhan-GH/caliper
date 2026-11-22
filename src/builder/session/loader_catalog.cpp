#include <caliper_builder/session/loader_catalog.h>

#include <caliper_builder/catalog/catalog_entry.h>
#include <caliper_builder/catalog/tensor_sample.h>

#include <exception>

#include "util.h"

namespace caliper_builder {

namespace {

using session_detail::cut_utf8;

CatalogListResult list_impl(LoaderStore& store, const CatalogFilter& filter) {
    CatalogListResult out;
    std::string kind;
    if (filter.kind) {
        if (*filter.kind != CatalogKind::table && *filter.kind != CatalogKind::tensor) return out;
        kind = catalog_kind_name(*filter.kind);
    }
    const std::size_t limit = filter.limit == 0 ? CatalogSampleLimits::default_list : filter.limit;
    const std::vector<LoadedSource> sources = store.sources(kind, filter.prefix, limit + 1);
    for (const LoadedSource& s : sources) {
        if (out.entries.size() >= limit) {
            out.truncated = true;
            break;
        }
        out.entries.push_back(LoaderCatalogProvider::entry_of(s));
    }
    return out;
}

}  // namespace

std::string quote_sql_identifier(const std::string& name) {
    std::string out = "\"";
    for (char c : name) {
        if (c == '"') out += '"';
        out += c;
    }
    return out + "\"";
}

LoaderCatalogProvider::LoaderCatalogProvider(LoaderStore& store) : store_(store) {}

CatalogEntry LoaderCatalogProvider::entry_of(const LoadedSource& s) {
    CatalogEntry e;
    e.id = s.id;
    e.kind = catalog_kind_from_name(s.kind).value_or(CatalogKind::table);
    e.name = s.name;
    e.title = s.title.empty() ? s.name : cut_utf8(s.title, CatalogLimits::max_name);
    if (!s.loader.empty()) e.origin.loader = cut_utf8(s.loader, CatalogLimits::max_origin_value);
    if (!s.file.empty()) e.origin.file = cut_utf8(s.file, CatalogLimits::max_origin_value);
    if (!s.tensor.empty()) e.origin.tensor = cut_utf8(s.tensor, CatalogLimits::max_origin_value);
    if (s.columns) {
        std::vector<CatalogColumn> columns;
        for (const ColumnInfo& c : *s.columns) columns.push_back(CatalogColumn{c.name, c.type});
        e.columns = std::move(columns);
    }
    if (s.rows && *s.rows >= 0) e.rows = static_cast<std::uint64_t>(*s.rows);
    if (s.dtype) e.dtype = catalog_dtype_from_name(dtype_name(*s.dtype));
    e.shape = s.shape;
    e.rate_hz = s.rate_hz;
    if (s.bytes && *s.bytes >= 0) e.bytes = static_cast<std::uint64_t>(*s.bytes);
    e.wire = s.wire;
    for (const std::string& w : s.warnings) {
        if (e.warnings.size() >= CatalogLimits::max_warnings) break;
        if (!w.empty()) e.warnings.push_back(cut_utf8(w, CatalogLimits::max_warning));
    }
    if (is_utc_timestamp(s.updated_at)) e.updated_at = s.updated_at;
    return e;
}

CatalogListResult LoaderCatalogProvider::list(const CatalogFilter& filter) {
    try {
        return list_impl(store_, filter);
    } catch (const std::exception& ex) {
        CatalogListResult out;
        out.error = Error("catalog.provider_failed", ex.what());
        return out;
    }
}

CatalogDescribeResult LoaderCatalogProvider::describe(const std::string& id) {
    CatalogDescribeResult out;
    try {
        LoaderStore::SourceGet got = store_.describe(id);
        if (got.error) {
            if (got.error.code != "catalog.not_found") out.error = got.error;
            return out;
        }
        out.entry = entry_of(got.source);
    } catch (const std::exception& ex) {
        out.error = Error("catalog.provider_failed", ex.what());
    }
    return out;
}

CatalogSampleResult LoaderCatalogProvider::sample(const std::string& id, const CatalogSampleRequest& request) {
    CatalogSampleResult out;
    try {
        LoaderStore::SourceGet got = store_.describe(id);
        if (got.error) {
            out.error = got.error;
            return out;
        }
        const CatalogEntry entry = entry_of(got.source);
        CatalogSample& s = out.sample;
        s.id = id;
        s.kind = entry.kind;
        if (entry.kind == CatalogKind::table) {
            QueryResult q = store_.query("SELECT * FROM " + quote_sql_identifier(got.source.name), request.limit, 10000);
            if (q.error) {
                out.error = Error("catalog.provider_failed", q.error.message, "id");
                return out;
            }
            session_detail::fix_nonfinite(q);
            std::vector<CatalogColumn> columns;
            for (const ColumnInfo& c : q.columns) columns.push_back(CatalogColumn{c.name, c.type});
            s.columns = std::move(columns);
            s.rows = std::move(q.rows);
            s.total_rows = entry.rows;
            s.truncated = q.truncated;
            return out;
        }
        TensorResult t = store_.load_tensor(id);
        if (t.error) {
            out.error = Error("catalog.provider_failed", t.error.message, "id");
            return out;
        }
        const auto dtype = catalog_dtype_from_name(dtype_name(t.tensor.dtype()));
        if (!dtype) {
            out.error = Error("catalog.provider_failed", "The tensor has a dtype the catalog does not know.", "id");
            return out;
        }
        TensorSampleResult ts = sample_tensor(*dtype, t.tensor.shape(), t.tensor.data(),
                                              static_cast<std::size_t>(t.tensor.nbytes()), request.start, request.count);
        if (ts.error) {
            out.error = ts.error;
            return out;
        }
        s.dtype = dtype;
        s.shape = t.tensor.shape();
        s.stats = std::move(ts.stats);
        s.slice = std::move(ts.slice);
    } catch (const std::exception& ex) {
        out.error = Error("catalog.provider_failed", ex.what());
    }
    return out;
}

std::uint64_t LoaderCatalogProvider::revision() const { return revision_.load(); }

void LoaderCatalogProvider::touch() { revision_.fetch_add(1); }

}  // namespace caliper_builder
