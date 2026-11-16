#include <caliper_builder/catalog/catalog.h>

#include "text_util.h"

#include <algorithm>
#include <cmath>
#include <exception>
#include <map>
#include <utility>

namespace caliper_builder {

namespace {

using nlohmann::json;
using catalog_detail::cut_utf8;
using catalog_detail::has_control;
using catalog_detail::repair_utf8;
using catalog_detail::valid_utf8;

constexpr std::size_t kMaxWarnings = 32;

Error bad_request(std::string message, std::string where) {
    return Error("catalog.bad_request", std::move(message), std::move(where));
}

// An id as it may appear in a message: itself when it is safe to show.
std::string shown_id(const std::string& id) {
    if (id.size() <= 300 && valid_utf8(id) && !has_control(id)) return id;
    return "an entry with a malformed id";
}

std::string error_text(const Error& e) {
    std::string text = repair_utf8(e.message.empty() ? e.code : e.message);
    cut_utf8(text, 512);
    return text;
}

void add_warning(std::vector<std::string>& warnings, std::size_t& dropped, std::string text) {
    if (warnings.size() < kMaxWarnings)
        warnings.push_back(std::move(text));
    else
        ++dropped;
}

bool provider_name_ok(std::string_view name) {
    if (name.empty() || name.size() > 32) return false;
    if (name[0] < 'a' || name[0] > 'z') return false;
    for (const char c : name)
        if (!((c >= 'a' && c <= 'z') || (c >= '0' && c <= '9') || c == '_' || c == '-')) return false;
    return true;
}

json stats_to_json(const CatalogTensorStats& s) {
    auto opt = [](const std::optional<double>& v) { return v ? json(*v) : json(nullptr); };
    return json{{"count", s.count}, {"sampled", s.sampled}, {"exact", s.exact}, {"nonfinite", s.nonfinite},
                {"min", opt(s.min)},  {"max", opt(s.max)},          {"mean", opt(s.mean)},  {"std", opt(s.std)}};
}

// Makes one cell a scalar of at most 1024 bytes, counting the cells it cut.
void fix_cell(json& cell, std::uint64_t& cut_cells) {
    if (cell.is_object() || cell.is_array()) cell = cell.dump(-1, ' ', false, json::error_handler_t::replace);
    if (cell.is_number_float()) {
        const double d = cell.get<double>();
        if (std::isnan(d)) cell = "nan";
        else if (std::isinf(d)) cell = d > 0 ? "inf" : "-inf";
    }
    if (!cell.is_string()) return;
    std::string& s = cell.get_ref<std::string&>();
    if (!valid_utf8(s)) s = repair_utf8(s);
    if (cut_utf8(s, CatalogSampleLimits::max_cell_bytes)) ++cut_cells;
}

}  // namespace

json catalog_sample_to_json(const CatalogSample& s) {
    json out = json::object();
    out["id"] = s.id;
    const char* kind = catalog_kind_name(s.kind);
    out["kind"] = kind ? kind : "";
    if (s.columns) {
        json cols = json::array();
        for (const auto& c : *s.columns) cols.push_back(catalog_column_to_json(c));
        out["columns"] = std::move(cols);
    } else {
        out["columns"] = nullptr;
    }
    out["rows"] = s.rows ? *s.rows : json(nullptr);
    out["total_rows"] = s.total_rows ? json(*s.total_rows) : json(nullptr);
    out["truncated"] = s.truncated;
    out["cut_cells"] = s.cut_cells;
    const char* dtype = s.dtype ? catalog_dtype_name(*s.dtype) : nullptr;
    out["dtype"] = dtype ? json(dtype) : json(nullptr);
    out["shape"] = s.shape ? json(*s.shape) : json(nullptr);
    out["stats"] = s.stats ? stats_to_json(*s.stats) : json(nullptr);
    if (s.slice)
        out["slice"] = json{{"start", s.slice->start}, {"shape", s.slice->shape}, {"values", s.slice->values}};
    else
        out["slice"] = nullptr;
    return out;
}

Error check_catalog_filter(const CatalogFilter& filter) {
    if (filter.limit < 1 || filter.limit > CatalogSampleLimits::max_list)
        return bad_request("The limit of a list is from 1 to 1000.", "limit");
    if (filter.prefix.size() > CatalogLimits::max_name || !valid_utf8(filter.prefix))
        return bad_request("The prefix is at most 256 bytes of UTF-8.", "prefix");
    return {};
}

Error check_catalog_sample_request(const CatalogSampleRequest& r, const std::vector<std::int64_t>* shape) {
    if (r.limit < 1 || r.limit > CatalogSampleLimits::max_rows)
        return bad_request("The row limit of a sample is from 1 to 1000.", "limit");
    if (!shape) {
        if (!r.start.empty() || !r.count.empty())
            return bad_request("start and count apply only to a tensor-shaped source.",
                               r.start.empty() ? "count" : "start");
        return {};
    }
    const std::size_t rank = shape->size();
    if (!r.start.empty() && r.start.size() != rank)
        return bad_request("start gives one position for each of the tensor's " + std::to_string(rank) + " axes.",
                           "start");
    if (!r.count.empty() && r.count.size() != rank)
        return bad_request("count gives one size for each of the tensor's " + std::to_string(rank) + " axes.",
                           "count");
    std::int64_t box = 1;
    for (std::size_t a = 0; a < rank; ++a) {
        const std::int64_t dim = (*shape)[a];
        const std::int64_t start = r.start.empty() ? 0 : r.start[a];
        const std::string at = "[" + std::to_string(a) + "]";
        if (start < 0 || (dim > 0 && start >= dim) || (dim == 0 && start != 0))
            return bad_request("start is outside axis " + std::to_string(a) + ", whose size is " +
                                   std::to_string(dim) + ".",
                               "start" + at);
        if (r.count.empty()) continue;
        const std::int64_t count = r.count[a];
        if (dim == 0) {
            if (count != 0) return bad_request("Axis " + std::to_string(a) + " is empty, so its count is 0.", "count" + at);
            box = 0;
            continue;
        }
        if (count < 1 || count > CatalogSampleLimits::max_slice_axis)
            return bad_request("A count is from 1 to 64.", "count" + at);
        if (count > dim - start)
            return bad_request("start plus count passes the end of axis " + std::to_string(a) + ".", "count" + at);
        if (box > 0) box *= count;
        if (box > CatalogSampleLimits::max_slice_values)
            return bad_request("A slice holds at most 4096 values.", "count");
    }
    return {};
}

Error Catalog::add_provider(std::string name, std::shared_ptr<CatalogProvider> provider, int priority) {
    if (!provider) return Error("catalog.bad_provider", "The provider is null.", name);
    if (!provider_name_ok(name))
        return Error("catalog.bad_provider", "A provider's name matches [a-z][a-z0-9_-]{0,31}.", "name");
    std::lock_guard<std::mutex> lock(mutex_);
    if (slots_.size() >= CatalogSampleLimits::max_providers)
        return Error("catalog.bad_provider", "A catalog holds at most 32 providers.", name);
    for (const auto& s : slots_)
        if (s.name == name) return Error("catalog.bad_provider", "A provider of this name is already added.", name);
    slots_.push_back(Slot{std::move(name), std::move(provider), priority, next_order_++});
    ++changes_;
    return {};
}

bool Catalog::remove_provider(std::string_view name) {
    std::lock_guard<std::mutex> lock(mutex_);
    const auto it = std::find_if(slots_.begin(), slots_.end(), [&](const Slot& s) { return s.name == name; });
    if (it == slots_.end()) return false;
    // Keep the removed provider's revision in the sum, so that it never drops.
    try {
        retired_ += it->provider->revision();
    } catch (...) {
    }
    slots_.erase(it);
    ++changes_;
    return true;
}

std::vector<Catalog::Slot> Catalog::snapshot() const {
    std::vector<Slot> slots;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        slots = slots_;
    }
    std::stable_sort(slots.begin(), slots.end(), [](const Slot& a, const Slot& b) {
        if (a.priority != b.priority) return a.priority > b.priority;
        return a.order < b.order;
    });
    return slots;
}

std::vector<std::string> Catalog::provider_names() const {
    std::vector<std::string> names;
    for (const auto& s : snapshot()) names.push_back(s.name);
    return names;
}

CatalogMergedList Catalog::list(const CatalogFilter& filter) const {
    CatalogMergedList out;
    if (Error err = check_catalog_filter(filter)) {
        out.error = std::move(err);
        return out;
    }
    std::size_t dropped = 0;
    CatalogFilter ask = filter;
    ask.limit = filter.limit + 1;

    std::map<std::string, std::string> owner;      // id -> the provider whose entry is kept
    std::map<std::string, std::size_t> dup_index;  // id -> position in out.duplicates
    for (const Slot& slot : snapshot()) {
        CatalogListResult got;
        try {
            got = slot.provider->list(ask);
        } catch (const std::exception& e) {
            got.error = Error("catalog.provider_failed", e.what());
        } catch (...) {
            got.error = Error("catalog.provider_failed", "The provider threw an exception.");
        }
        if (got.error) {
            add_warning(out.warnings, dropped, "The provider " + slot.name + " failed: " + error_text(got.error));
            continue;
        }
        if (got.truncated) out.truncated = true;
        for (auto& entry : got.entries) {
            if (Error err = validate_catalog_entry(entry)) {
                add_warning(out.warnings, dropped,
                            "The provider " + slot.name + " gave a faulty entry, " + shown_id(entry.id) + ": " +
                                error_text(err) + (err.where.empty() ? "" : " (" + err.where + ")"));
                continue;
            }
            if (filter.kind && entry.kind != *filter.kind) continue;
            if (entry.name.compare(0, filter.prefix.size(), filter.prefix) != 0) continue;
            const auto kept = owner.find(entry.id);
            if (kept != owner.end()) {
                if (kept->second == slot.name) continue;  // the provider repeated itself
                auto d = dup_index.find(entry.id);
                if (d == dup_index.end()) {
                    d = dup_index.emplace(entry.id, out.duplicates.size()).first;
                    out.duplicates.push_back(CatalogDuplicate{entry.id, kept->second, {}});
                }
                auto& hidden = out.duplicates[d->second].hidden;
                if (std::find(hidden.begin(), hidden.end(), slot.name) == hidden.end()) hidden.push_back(slot.name);
                continue;
            }
            owner.emplace(entry.id, slot.name);
            out.entries.push_back(std::move(entry));
        }
    }
    for (const auto& dup : out.duplicates) {
        std::string hidden;
        for (const auto& h : dup.hidden) hidden += (hidden.empty() ? "" : " and ") + h;
        add_warning(out.warnings, dropped,
                    dup.id + " is also offered by " + hidden + "; the entry of " + dup.kept + " is shown.");
    }
    std::sort(out.entries.begin(), out.entries.end(),
              [](const CatalogEntry& a, const CatalogEntry& b) { return a.id < b.id; });
    if (out.entries.size() > filter.limit) {
        out.entries.resize(filter.limit);
        out.truncated = true;
    }
    if (dropped > 0) out.warnings.push_back("And " + std::to_string(dropped) + " more warnings.");
    return out;
}

CatalogDescribeResult Catalog::describe(std::string_view id_view) const {
    CatalogDescribeResult out;
    const CatalogIdParse parsed = parse_catalog_id(id_view);
    if (parsed.error) {
        out.error = parsed.error;
        return out;
    }
    const std::string id(id_view);
    std::string failures;
    for (const Slot& slot : snapshot()) {
        CatalogDescribeResult got;
        try {
            got = slot.provider->describe(id);
        } catch (const std::exception& e) {
            got.error = Error("catalog.provider_failed", e.what());
        } catch (...) {
            got.error = Error("catalog.provider_failed", "The provider threw an exception.");
        }
        if (!got.error && got.entry) {
            if (got.entry->id != id) got.error = Error("catalog.provider_failed", "It gave an entry of another id.");
            else if (Error err = validate_catalog_entry(*got.entry)) got.error = err;
        }
        if (got.error) {
            if (failures.empty()) failures = "The provider " + slot.name + " failed: " + error_text(got.error);
            continue;
        }
        if (got.entry) {
            out.entry = std::move(got.entry);
            return out;
        }
    }
    if (!failures.empty())
        out.error = Error("catalog.provider_failed", failures, "id");
    else
        out.error = Error("catalog.not_found", "No source of the catalog has the id " + id + ".", "id");
    return out;
}

CatalogSampleResult Catalog::sample(std::string_view id_view, const CatalogSampleRequest& request) const {
    CatalogSampleResult out;
    const CatalogIdParse parsed = parse_catalog_id(id_view);
    if (parsed.error) {
        out.error = parsed.error;
        return out;
    }
    if (request.limit < 1 || request.limit > CatalogSampleLimits::max_rows) {
        out.error = bad_request("The row limit of a sample is from 1 to 1000.", "limit");
        return out;
    }
    const std::string id(id_view);
    std::string failures;
    for (const Slot& slot : snapshot()) {
        CatalogDescribeResult found;
        try {
            found = slot.provider->describe(id);
        } catch (...) {
            found.error = Error("catalog.provider_failed", "The provider threw an exception.");
        }
        if (!found.error && found.entry) {
            if (found.entry->id != id) found.error = Error("catalog.provider_failed", "It gave an entry of another id.");
            else if (Error err = validate_catalog_entry(*found.entry)) found.error = err;
        }
        if (found.error) {
            if (failures.empty()) failures = "The provider " + slot.name + " failed: " + error_text(found.error);
            continue;
        }
        if (!found.entry) continue;

        const CatalogEntry& entry = *found.entry;
        const bool tabular = catalog_kind_is_tabular(entry.kind);
        if (!tabular && !entry.shape) {
            out.error = Error("catalog.no_data", "The source " + id + " has published nothing yet.", "id");
            return out;
        }
        if (Error err = check_catalog_sample_request(request, tabular ? nullptr : &*entry.shape)) {
            out.error = std::move(err);
            return out;
        }
        CatalogSampleResult got;
        try {
            got = slot.provider->sample(id, request);
        } catch (const std::exception& e) {
            got.error = Error("catalog.provider_failed", e.what());
        } catch (...) {
            got.error = Error("catalog.provider_failed", "The provider threw an exception.");
        }
        if (got.error) {
            if (got.error.code.rfind("catalog.", 0) != 0) got.error.code = "catalog.provider_failed";
            got.error.message = error_text(got.error);
            out.error = std::move(got.error);
            return out;
        }
        CatalogSample& s = got.sample;
        auto provider_fault = [&](const std::string& what) {
            out.error = Error("catalog.provider_failed", "The provider " + slot.name + " gave a faulty sample: " + what,
                              "id");
            return out;
        };
        if (s.id != id || s.kind != entry.kind) return provider_fault("its id or kind differs from the entry's.");
        if (tabular) {
            if (!s.columns || !s.rows || !s.rows->is_array()) return provider_fault("a tabular sample has columns and rows.");
            json kept = json::array();
            std::size_t bytes = 0;
            for (auto& row : *s.rows) {
                if (kept.size() >= request.limit) {
                    s.truncated = true;
                    break;
                }
                if (!row.is_array() || row.size() != s.columns->size())
                    return provider_fault("a row does not have one cell for each column.");
                for (auto& cell : row) fix_cell(cell, s.cut_cells);
                bytes += row.dump(-1, ' ', false, json::error_handler_t::replace).size() + 1;
                if (bytes > CatalogSampleLimits::max_sample_json) {
                    s.truncated = true;
                    break;
                }
                kept.push_back(std::move(row));
            }
            s.rows = std::move(kept);
            s.dtype.reset();
            s.shape.reset();
            s.stats.reset();
            s.slice.reset();
        } else {
            if (!s.dtype || !s.shape || !s.stats || !s.slice)
                return provider_fault("a tensor sample has a dtype, a shape, statistics and a slice.");
            std::int64_t box = 1;
            for (const auto c : s.slice->shape) {
                if (c < 0 || c > CatalogSampleLimits::max_slice_axis) return provider_fault("the slice is too large.");
                box *= c;
            }
            if (box > CatalogSampleLimits::max_slice_values || !s.slice->values.is_array() ||
                s.slice->values.size() != static_cast<std::size_t>(box))
                return provider_fault("the slice's values do not match its shape or pass 4096.");
            s.columns.reset();
            s.rows.reset();
            s.total_rows.reset();
        }
        out.sample = std::move(s);
        return out;
    }
    if (!failures.empty())
        out.error = Error("catalog.provider_failed", failures, "id");
    else
        out.error = Error("catalog.not_found", "No source of the catalog has the id " + id + ".", "id");
    return out;
}

std::uint64_t Catalog::revision() const {
    std::uint64_t sum = 0;
    std::vector<Slot> slots;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        slots = slots_;
        sum = changes_ + retired_;
    }
    for (const auto& s : slots) {
        try {
            sum += s.provider->revision();
        } catch (...) {
        }
    }
    return sum;
}

}  // namespace caliper_builder
