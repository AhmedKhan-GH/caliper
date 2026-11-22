#include <caliper_builder/session/resolve.h>

#include <caliper_builder/catalog/catalog_entry.h>
#include <caliper_builder/loaders/tensor_ops.h>
#include <caliper_builder/session/loader_catalog.h>
#include <caliper_builder/spec/format.h>

#include <algorithm>
#include <exception>
#include <map>
#include <set>

#include "util.h"

namespace caliper_builder {

namespace {

using nlohmann::json;

struct Binding {
    std::string source;
    std::string column;  // empty: the whole source
};

Binding split_binding(const std::string& b) {
    const auto dot = b.find('.');
    if (dot == std::string::npos) return {b, {}};
    return {b.substr(0, dot), b.substr(dot + 1)};
}

enum class RootKind { tensor, table, sql, served_tensor, served_table, unavailable };

struct Chain {
    std::vector<const spec::Source*> path;  // the root first, the bound source last
    RootKind kind = RootKind::unavailable;
    CatalogKind catalog_kind = CatalogKind::table;  // of a catalog root
    Error error;
};

bool is_served(RootKind k) { return k == RootKind::served_tensor || k == RootKind::served_table; }

// Follows the chain of `from` from source `id` to its root and classifies the
// root. A served root (live, feed, metric) needs a registered data source.
Chain chain_of(const spec::Spec& spec, const std::string& id, const DataSources* sources) {
    Chain c;
    const spec::Source* cur = spec.find_source(id);
    if (!cur) {
        c.error = Error("resolve.unknown_source", "The spec has no source " + id + ".", id);
        return c;
    }
    std::vector<const spec::Source*> rev{cur};
    while (cur->from) {
        if (rev.size() > spec::limits::kSources) {
            c.error = Error("resolve.cycle", "The chain of from that starts at " + id + " does not end.", id);
            return c;
        }
        const spec::Source* next = spec.find_source(*cur->from);
        if (!next) {
            c.error = Error("resolve.unknown_source", "The spec has no source " + *cur->from + ".", cur->id);
            return c;
        }
        rev.push_back(next);
        cur = next;
    }
    c.path.assign(rev.rbegin(), rev.rend());
    const spec::Source& root = *c.path.front();
    if (root.sql) {
        c.kind = RootKind::sql;
        return c;
    }
    if (!root.catalog) {
        c.error = Error("resolve.unknown_source", "The source " + root.id + " names no data.", root.id);
        return c;
    }
    const CatalogIdParse parsed = parse_catalog_id(*root.catalog);
    if (parsed.error) {
        c.error = Error("resolve.unknown_source", "The catalog id " + *root.catalog + " is not valid.", root.id);
        return c;
    }
    c.catalog_kind = parsed.kind;
    switch (parsed.kind) {
        case CatalogKind::tensor: c.kind = RootKind::tensor; return c;
        case CatalogKind::table: c.kind = RootKind::table; return c;
        case CatalogKind::live:
        case CatalogKind::run: c.kind = RootKind::served_tensor; break;
        case CatalogKind::feed:
        case CatalogKind::metric: c.kind = RootKind::served_table; break;
    }
    if (!sources || !sources->get(parsed.kind)) {
        c.kind = RootKind::unavailable;
        c.error = Error("resolve.unavailable",
                        "The source " + root.id + " is a " + catalog_kind_name(parsed.kind) +
                            " source, which the app serves, and no data source for that kind is registered.",
                        root.id);
    }
    return c;
}

// The text that names a chain of transforms over its root: the root's catalog
// id and the transforms of every source down the chain, in order.
std::string chain_key(const Chain& chain) {
    json parts = json::array();
    for (const spec::Source* s : chain.path) {
        json records = spec::to_json(*s).value("transform", json::array());
        for (auto& r : records)
            if (r.is_object()) r.erase("x");
        parts.push_back(std::move(records));
    }
    return *chain.path.front()->catalog + "\n" + parts.dump();
}

// One step of the mix of generations (splitmix64), so that the panel's
// generation changes when any one of its sources changes.
std::uint64_t mix_generation(std::uint64_t h, std::uint64_t g) {
    std::uint64_t z = h + 0x9E3779B97F4A7C15ull + g;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
    return z ^ (z >> 31);
}

class PanelResolver {
public:
    PanelResolver(const spec::Spec& s, LoaderStore& store, const ResolveLimits& limits, DataSources* sources)
        : spec_(s), store_(store), limits_(limits), sources_(sources) {}

    PanelData run(std::string_view panel_id) {
        PanelData out;
        out.panel = std::string(panel_id);
        const spec::Panel* panel = spec_.find_panel(panel_id);
        if (!panel) {
            out.error = Error("panel.not_found", "The open spec has no panel " + out.panel + ".", out.panel);
            return out;
        }
        out.primitive = panel->primitive;
        // A scene panel draws a live scene of the attached script, which the
        // app's live views hold; it binds no source, so it resolves to no slot.
        if (panel->primitive == spec::Primitive::scene) return out;
        // First, what each tabular root must give, so that each is read once.
        for (const auto& [slot, value] : panel->bind) {
            for (const std::string& b : value.bindings) {
                const Binding parts = split_binding(b);
                Chain chain = chain_of(parts.source);
                if (chain.error) {
                    out.error = chain.error;
                    if (sources_) out.generation = panel_generation(spec_, panel_id, *sources_);
                    return out;
                }
                if (chain.kind != RootKind::table && chain.kind != RootKind::sql && chain.kind != RootKind::served_table)
                    continue;
                const std::string root = chain.path.front()->id;
                if (!parts.column.empty()) {
                    needed_[root].insert(parts.column);
                } else if (slot == "rows" && panel->options.columns) {
                    for (const auto& c : *panel->options.columns) needed_[root].insert(c);
                } else {
                    whole_.insert(root);
                }
            }
        }
        bool served = false;
        std::uint64_t generation = 0;
        for (const auto& [slot, value] : panel->bind) {
            for (std::size_t i = 0; i < value.bindings.size(); ++i) {
                SlotData data;
                data.slot = slot;
                data.index = i;
                data.binding = value.bindings[i];
                const Binding parts = split_binding(data.binding);
                data.source = parts.source;
                data.column = parts.column;
                bool slot_served = false;
                if (Error err = fill(*panel, data, &slot_served)) {
                    out.error = std::move(err);
                    out.slots.clear();
                    if (sources_) out.generation = panel_generation(spec_, panel_id, *sources_);
                    return out;
                }
                if (slot_served) {
                    served = true;
                    generation = mix_generation(generation, data.generation);
                }
                out.slots.push_back(std::move(data));
            }
        }
        out.generation = served ? generation : 0;
        return out;
    }

private:
    Chain chain_of(const std::string& id) const { return ::caliper_builder::chain_of(spec_, id, sources_); }

    Error fill(const spec::Panel& panel, SlotData& data, bool* served) {
        Chain chain = chain_of(data.source);
        if (chain.error) return chain.error;
        *served = is_served(chain.kind);
        if (chain.kind == RootKind::tensor || chain.kind == RootKind::served_tensor) {
            if (!data.column.empty())
                return Error("resolve.not_tabular", "The source " + data.source + " is a tensor and has no columns.",
                             data.binding);
            TensorData t;
            if (Error err = tensor_of(chain, &t, &data.generation)) return err;
            data.kind = SlotData::Kind::tensor;
            data.tensor = std::move(t);
            return {};
        }
        for (const spec::Source* s : chain.path)
            if (!s->transforms.empty())
                return Error("resolve.not_a_tensor", "The source " + s->id + " is tabular, so it takes no transforms.",
                             s->id);
        const QueryResult* q = nullptr;
        if (Error err = table_of(chain, &q)) return err;
        auto column_index = [&](const std::string& name, std::size_t* out) -> Error {
            for (std::size_t c = 0; c < q->columns.size(); ++c)
                if (q->columns[c].name == name) {
                    *out = c;
                    return {};
                }
            return Error("resolve.unknown_column", "The source " + data.source + " has no column " + name + ".",
                         data.binding);
        };
        data.truncated = q->truncated || truncated_.count(chain.path.front()->id) > 0;
        if (*served) data.generation = table_generations_[chain.path.front()->id];
        if (!data.column.empty()) {
            std::size_t c = 0;
            if (Error err = column_index(data.column, &c)) return err;
            data.kind = SlotData::Kind::column;
            data.columns = {q->columns[c]};
            data.values = json::array();
            for (const auto& row : q->rows) data.values.push_back(row.is_array() && c < row.size() ? row[c] : json());
            return {};
        }
        data.kind = SlotData::Kind::table;
        std::vector<std::size_t> picked;
        if (data.slot == "rows" && panel.options.columns) {
            for (const auto& name : *panel.options.columns) {
                std::size_t c = 0;
                if (Error err = column_index(name, &c)) return err;
                picked.push_back(c);
            }
        } else {
            for (std::size_t c = 0; c < q->columns.size(); ++c) picked.push_back(c);
        }
        for (std::size_t c : picked) data.columns.push_back(q->columns[c]);
        data.values = json::array();
        for (const auto& row : q->rows) {
            json kept = json::array();
            for (std::size_t c : picked) kept.push_back(row.is_array() && c < row.size() ? row[c] : json());
            data.values.push_back(std::move(kept));
        }
        return {};
    }

    // The tensor of a served root (a live view), from the cache when the
    // chain was computed from the generation its source has now.
    Error served_tensor_of(const Chain& chain, TensorData* out, std::uint64_t* generation) {
        const spec::Source& root = *chain.path.front();
        std::shared_ptr<CatalogDataSource> source = sources_->get(chain.catalog_kind);
        if (!source)
            return Error("resolve.unavailable", "No data source is registered for the source " + root.id + ".", root.id);
        const std::string key = chain_key(chain);
        const std::uint64_t now = source->generation(*root.catalog);
        if (now != 0 && sources_->find_cached(key, now, out)) {
            *generation = now;
            return {};
        }
        ServedRequest request;
        request.max_rows = limits_.max_rows;
        request.max_cells = limits_.max_cells;
        request.max_bytes = limits_.max_tensor_bytes;
        ServedData data = source->read(*root.catalog, request);
        if (data.error) {
            Error err = data.error;
            if (err.where.empty()) err.where = root.id;
            return err;
        }
        if (data.tabular || !data.tensor.valid())
            return Error("resolve.not_a_tensor", "The served source " + *root.catalog + " gave no tensor.", root.id);
        if (data.tensor.nbytes() > limits_.max_tensor_bytes)
            return Error("resolve.too_large",
                         "The source " + *root.catalog + " gave a tensor of " + std::to_string(data.tensor.nbytes()) +
                             " bytes, past the limit of " + std::to_string(limits_.max_tensor_bytes) + ".",
                         root.id);
        TensorData t = std::move(data.tensor);
        if (Error err = apply_chain(chain, &t)) return err;
        sources_->store_cached(key, data.generation, t);
        *generation = data.generation;
        *out = std::move(t);
        return {};
    }

    Error tensor_of(const Chain& chain, TensorData* out, std::uint64_t* generation) {
        const std::string bound = chain.path.back()->id;
        const auto cached = tensors_.find(bound);
        if (cached != tensors_.end()) {
            *out = cached->second.first;
            *generation = cached->second.second;
            return {};
        }
        TensorData t;
        std::uint64_t g = 0;
        if (chain.kind == RootKind::served_tensor) {
            if (Error err = served_tensor_of(chain, &t, &g)) return err;
        } else {
            const spec::Source& root = *chain.path.front();
            TensorResult loaded = store_.load_tensor(*root.catalog);
            if (loaded.error) return loaded.error;
            t = std::move(loaded.tensor);
            if (Error err = apply_chain(chain, &t)) return err;
        }
        tensors_[bound] = {t, g};
        *generation = g;
        *out = std::move(t);
        return {};
    }

    // Applies the transforms of the root and of each source down the chain, in
    // order, refusing a result past max_tensor_bytes before it is made.
    Error apply_chain(const Chain& chain, TensorData* tensor) {
        TensorData& t = *tensor;
        for (const spec::Source* s : chain.path) {
            if (s->transforms.empty()) continue;
            const json records = spec::to_json(*s).value("transform", json::array());
            for (std::size_t k = 0; k < records.size(); ++k) {
                const std::string where = "source " + s->id + ", transform[" + std::to_string(k) + "]";
                json record = records[k];
                record.erase("x");
                TensorTransform tt;
                if (Error err = transform_from_json(record, &tt)) {
                    err.where = where;
                    return err;
                }
                std::vector<int64_t> shape;
                DType dtype = DType::f32;
                if (Error err = transform_result_shape(t.shape(), t.dtype(), tt, &shape, &dtype)) {
                    err.where = where;
                    return err;
                }
                uint64_t numel = 0;
                uint64_t nbytes = 0;
                if (Error err = check_shape(shape, dtype, &numel, &nbytes)) {
                    err.where = where;
                    return err;
                }
                if (nbytes > limits_.max_tensor_bytes)
                    return Error("resolve.too_large",
                                 "The transform would make a tensor of " + std::to_string(nbytes) +
                                     " bytes, past the limit of " + std::to_string(limits_.max_tensor_bytes) + ".",
                                 where);
                TensorResult r = apply_transform(t, tt);
                if (r.error) {
                    r.error.where = where;
                    return r.error;
                }
                t = std::move(r.tensor);
            }
        }
        return {};
    }

    Error table_of(const Chain& chain, const QueryResult** out) {
        const spec::Source& root = *chain.path.front();
        const auto cached = tables_.find(root.id);
        if (cached != tables_.end()) {
            *out = &cached->second;
            return {};
        }
        QueryResult q;
        const std::uint64_t max_rows = std::max<std::uint64_t>(1, std::min<std::uint64_t>(limits_.max_rows, kMaxQueryRows));
        if (chain.kind == RootKind::served_table) {
            std::shared_ptr<CatalogDataSource> source = sources_->get(chain.catalog_kind);
            if (!source)
                return Error("resolve.unavailable", "No data source is registered for the source " + root.id + ".",
                             root.id);
            ServedRequest request;
            request.max_rows = std::max<std::uint64_t>(1, limits_.max_rows);
            request.max_cells = std::max<std::uint64_t>(1, limits_.max_cells);
            request.max_bytes = limits_.max_tensor_bytes;
            ServedData data = source->read(*root.catalog, request);
            if (data.error) {
                Error err = data.error;
                if (err.where.empty()) err.where = root.id;
                return err;
            }
            if (!data.tabular || !data.rows.is_array())
                return Error("resolve.not_tabular", "The served source " + *root.catalog + " gave no table.", root.id);
            q.columns = std::move(data.columns);
            q.rows = std::move(data.rows);
            q.truncated = data.truncated;
            table_generations_[root.id] = data.generation;
        } else if (chain.kind == RootKind::table) {
            LoaderStore::SourceGet got = store_.describe(*root.catalog);
            if (got.error)
                return Error("resolve.unknown_source",
                             "The catalog has no table " + *root.catalog + "; import the file or run its loader.",
                             root.id);
            if (got.source.kind != "table")
                return Error("resolve.not_tabular", "The catalog source " + *root.catalog + " is not a table.", root.id);
            std::string select;
            std::size_t ncols = got.source.columns ? got.source.columns->size() : 1;
            if (whole_.count(root.id) || !needed_.count(root.id)) {
                select = "*";
            } else {
                for (const auto& c : needed_[root.id]) {
                    if (!select.empty()) select += ", ";
                    select += quote_sql_identifier(c);
                }
                ncols = needed_[root.id].size();
            }
            const std::uint64_t rows = std::max<std::uint64_t>(
                1, std::min<std::uint64_t>(max_rows, limits_.max_cells / std::max<std::size_t>(1, ncols)));
            q = store_.query("SELECT " + select + " FROM " + quote_sql_identifier(got.source.name), rows,
                             limits_.query_timeout_ms);
        } else {
            q = store_.query(*root.sql, max_rows, limits_.query_timeout_ms);
        }
        if (q.error) {
            Error err = q.error;
            if (err.code.rfind("loader.", 0) == 0 || err.code.rfind("query.", 0) == 0) {
                err.message = "The query of the source " + root.id + " failed: " + err.message;
                err.code = err.code == "query.timeout" ? "resolve.timeout" : "resolve.query_failed";
            }
            err.where = root.id;
            return err;
        }
        if (!q.columns.empty()) {
            const std::uint64_t max_rows_by_cells = std::max<std::uint64_t>(1, limits_.max_cells / q.columns.size());
            if (q.rows.size() > max_rows_by_cells) {
                json kept = json::array();
                for (std::size_t r = 0; r < max_rows_by_cells; ++r) kept.push_back(std::move(q.rows[r]));
                q.rows = std::move(kept);
                truncated_.insert(root.id);
            }
        }
        session_detail::fix_nonfinite(q);
        auto [it, inserted] = tables_.emplace(root.id, std::move(q));
        *out = &it->second;
        return {};
    }

    const spec::Spec& spec_;
    LoaderStore& store_;
    const ResolveLimits& limits_;
    DataSources* sources_ = nullptr;
    std::map<std::string, std::set<std::string>> needed_;  // by tabular root
    std::set<std::string> whole_;                          // tabular roots read whole
    std::set<std::string> truncated_;                      // cut at max_cells here
    std::map<std::string, std::pair<TensorData, std::uint64_t>> tensors_;  // by bound source, with its generation
    std::map<std::string, QueryResult> tables_;            // by tabular root
    std::map<std::string, std::uint64_t> table_generations_;  // by served tabular root
};

}  // namespace

// ---- DataSources --------------------------------------------------------------

DataSources::DataSources() = default;
DataSources::DataSources(Options options) : options_(options) {}

bool DataSources::set(CatalogKind kind, std::shared_ptr<CatalogDataSource> source) {
    std::lock_guard<std::mutex> lock(mutex_);
    switch (kind) {
        case CatalogKind::live: live_ = std::move(source); break;
        case CatalogKind::feed: feed_ = std::move(source); break;
        case CatalogKind::metric: metric_ = std::move(source); break;
        case CatalogKind::run: run_ = std::move(source); break;
        default: return false;
    }
    cache_.clear();
    uses_.clear();
    cache_bytes_ = 0;
    return true;
}

std::shared_ptr<CatalogDataSource> DataSources::get(CatalogKind kind) const {
    std::lock_guard<std::mutex> lock(mutex_);
    switch (kind) {
        case CatalogKind::live: return live_;
        case CatalogKind::feed: return feed_;
        case CatalogKind::metric: return metric_;
        case CatalogKind::run: return run_;
        default: return nullptr;
    }
}

std::uint64_t DataSources::generation(std::string_view catalog_id) const {
    const CatalogIdParse parsed = parse_catalog_id(catalog_id);
    if (parsed.error) return 0;
    std::shared_ptr<CatalogDataSource> source = get(parsed.kind);
    if (!source) return 0;
    try {
        return source->generation(std::string(catalog_id));
    } catch (...) {
        return 0;
    }
}

DataSources::CacheStats DataSources::cache_stats() const {
    std::lock_guard<std::mutex> lock(mutex_);
    CacheStats out;
    out.computed = computed_;
    out.reused = reused_;
    out.entries = cache_.size();
    out.bytes = cache_bytes_;
    return out;
}

bool DataSources::find_cached(const std::string& key, std::uint64_t generation, TensorData* out) {
    std::lock_guard<std::mutex> lock(mutex_);
    const auto it = cache_.find(key);
    if (it == cache_.end() || it->second.generation != generation) return false;
    uses_.splice(uses_.begin(), uses_, it->second.use);
    *out = it->second.tensor;
    ++reused_;
    return true;
}

void DataSources::store_cached(const std::string& key, std::uint64_t generation, const TensorData& tensor) {
    std::lock_guard<std::mutex> lock(mutex_);
    ++computed_;
    const auto old = cache_.find(key);
    if (old != cache_.end()) {
        cache_bytes_ -= old->second.tensor.nbytes();
        uses_.erase(old->second.use);
        cache_.erase(old);
    }
    if (generation == 0 || options_.max_cache_entries == 0 || tensor.nbytes() > options_.max_cache_bytes) return;
    while (!uses_.empty() &&
           (cache_.size() >= options_.max_cache_entries || cache_bytes_ + tensor.nbytes() > options_.max_cache_bytes)) {
        const auto victim = cache_.find(uses_.back());
        cache_bytes_ -= victim->second.tensor.nbytes();
        cache_.erase(victim);
        uses_.pop_back();
    }
    uses_.push_front(key);
    cache_[key] = Cached{generation, tensor, uses_.begin()};
    cache_bytes_ += tensor.nbytes();
}

std::uint64_t panel_generation(const spec::Spec& s, std::string_view panel_id, const DataSources& sources) {
    const spec::Panel* panel = s.find_panel(panel_id);
    if (!panel || panel->primitive == spec::Primitive::scene) return 0;
    bool served = false;
    std::uint64_t generation = 0;
    for (const auto& [slot, value] : panel->bind) {
        (void)slot;
        for (const std::string& b : value.bindings) {
            const Chain chain = chain_of(s, split_binding(b).source, &sources);
            if (!is_served(chain.kind) || chain.path.empty()) continue;
            served = true;
            generation = mix_generation(generation, sources.generation(*chain.path.front()->catalog));
        }
    }
    return served ? generation : 0;
}

const char* slot_kind_name(SlotData::Kind kind) {
    switch (kind) {
        case SlotData::Kind::tensor: return "tensor";
        case SlotData::Kind::column: return "column";
        case SlotData::Kind::table: return "table";
    }
    return "tensor";
}

PanelData resolve_panel(const spec::Spec& s, std::string_view panel_id, LoaderStore& store,
                        const ResolveLimits& limits, DataSources* sources) {
    try {
        PanelResolver r(s, store, limits, sources);
        return r.run(panel_id);
    } catch (const std::exception& e) {
        PanelData out;
        out.panel = std::string(panel_id);
        out.error = Error("resolve.failed", std::string("The panel's data could not be resolved: ") + e.what());
        return out;
    }
}

json panel_data_summary(const PanelData& data, std::size_t max_values) {
    json slots = json::array();
    for (const SlotData& s : data.slots) {
        json one{{"slot", s.slot},
                 {"index", s.index},
                 {"binding", s.binding},
                 {"kind", slot_kind_name(s.kind)},
                 {"truncated", s.truncated}};
        json values = json::array();
        if (s.kind == SlotData::Kind::tensor) {
            one["dtype"] = dtype_name(s.tensor.dtype());
            one["shape"] = s.tensor.shape();
            for (uint64_t i = 0; i < s.tensor.numel() && values.size() < max_values; ++i)
                values.push_back(tensor_element_json(s.tensor, i));
        } else {
            json cols = json::array();
            for (const auto& c : s.columns) cols.push_back(c.to_json());
            one["columns"] = std::move(cols);
            one["rows"] = s.values.size();
            for (const auto& v : s.values) {
                if (values.size() >= max_values) break;
                values.push_back(v);
            }
        }
        one["values"] = std::move(values);
        slots.push_back(std::move(one));
    }
    json out{{"panel", data.panel}, {"primitive", spec::to_string(data.primitive)}, {"slots", std::move(slots)}};
    if (data.error) out["error"] = {{"code", data.error.code}, {"message", data.error.message}, {"where", data.error.where}};
    return out;
}

}  // namespace caliper_builder
