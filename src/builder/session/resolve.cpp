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

enum class RootKind { tensor, table, sql, unavailable };

struct Chain {
    std::vector<const spec::Source*> path;  // the root first, the bound source last
    RootKind kind = RootKind::unavailable;
    Error error;
};

class PanelResolver {
public:
    PanelResolver(const spec::Spec& s, LoaderStore& store, const ResolveLimits& limits)
        : spec_(s), store_(store), limits_(limits) {}

    PanelData run(std::string_view panel_id) {
        PanelData out;
        out.panel = std::string(panel_id);
        const spec::Panel* panel = spec_.find_panel(panel_id);
        if (!panel) {
            out.error = Error("panel.not_found", "The open spec has no panel " + out.panel + ".", out.panel);
            return out;
        }
        out.primitive = panel->primitive;
        // First, what each tabular root must give, so that each is read once.
        for (const auto& [slot, value] : panel->bind) {
            for (const std::string& b : value.bindings) {
                const Binding parts = split_binding(b);
                Chain chain = chain_of(parts.source);
                if (chain.error) {
                    out.error = chain.error;
                    return out;
                }
                if (chain.kind != RootKind::table && chain.kind != RootKind::sql) continue;
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
        for (const auto& [slot, value] : panel->bind) {
            for (std::size_t i = 0; i < value.bindings.size(); ++i) {
                SlotData data;
                data.slot = slot;
                data.index = i;
                data.binding = value.bindings[i];
                const Binding parts = split_binding(data.binding);
                data.source = parts.source;
                data.column = parts.column;
                if (Error err = fill(*panel, data)) {
                    out.error = std::move(err);
                    out.slots.clear();
                    return out;
                }
                out.slots.push_back(std::move(data));
            }
        }
        return out;
    }

private:
    Chain chain_of(const std::string& id) {
        Chain c;
        const spec::Source* cur = spec_.find_source(id);
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
            const spec::Source* next = spec_.find_source(*cur->from);
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
        } else if (root.catalog) {
            const CatalogIdParse parsed = parse_catalog_id(*root.catalog);
            if (parsed.error) {
                c.error = Error("resolve.unknown_source", "The catalog id " + *root.catalog + " is not valid.", root.id);
                return c;
            }
            if (parsed.kind == CatalogKind::tensor) c.kind = RootKind::tensor;
            else if (parsed.kind == CatalogKind::table) c.kind = RootKind::table;
            else c.kind = RootKind::unavailable;
        }
        if (c.kind == RootKind::unavailable)
            c.error = Error("resolve.unavailable",
                            "The source " + root.id + " is served by the app (a live view, a feed channel or a "
                            "metric), whose data the session cannot read yet.",
                            root.id);
        return c;
    }

    Error fill(const spec::Panel& panel, SlotData& data) {
        Chain chain = chain_of(data.source);
        if (chain.error) return chain.error;
        if (chain.kind == RootKind::tensor) {
            if (!data.column.empty())
                return Error("resolve.not_tabular", "The source " + data.source + " is a tensor and has no columns.",
                             data.binding);
            TensorData t;
            if (Error err = tensor_of(chain, &t)) return err;
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

    Error tensor_of(const Chain& chain, TensorData* out) {
        const std::string bound = chain.path.back()->id;
        const auto cached = tensors_.find(bound);
        if (cached != tensors_.end()) {
            *out = cached->second;
            return {};
        }
        const spec::Source& root = *chain.path.front();
        TensorResult loaded = store_.load_tensor(*root.catalog);
        if (loaded.error) return loaded.error;
        TensorData t = std::move(loaded.tensor);
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
        tensors_[bound] = t;
        *out = std::move(t);
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
        if (chain.kind == RootKind::table) {
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
    std::map<std::string, std::set<std::string>> needed_;  // by tabular root
    std::set<std::string> whole_;                          // tabular roots read whole
    std::set<std::string> truncated_;                      // cut at max_cells here
    std::map<std::string, TensorData> tensors_;            // by bound source
    std::map<std::string, QueryResult> tables_;            // by tabular root
};

}  // namespace

const char* slot_kind_name(SlotData::Kind kind) {
    switch (kind) {
        case SlotData::Kind::tensor: return "tensor";
        case SlotData::Kind::column: return "column";
        case SlotData::Kind::table: return "table";
    }
    return "tensor";
}

PanelData resolve_panel(const spec::Spec& s, std::string_view panel_id, LoaderStore& store,
                        const ResolveLimits& limits) {
    try {
        PanelResolver r(s, store, limits);
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
