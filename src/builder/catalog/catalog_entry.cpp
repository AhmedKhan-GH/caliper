#include <caliper_builder/catalog/catalog_entry.h>

#include "text_util.h"

#include <cmath>
#include <utility>

namespace caliper_builder {

namespace {

using nlohmann::json;
using catalog_detail::has_control;
using catalog_detail::valid_utf8;

Error invalid(std::string message, std::string where) {
    return Error("catalog.invalid_entry", std::move(message), std::move(where));
}

struct KindName {
    CatalogKind kind;
    const char* name;
};
constexpr KindName kKinds[] = {
    {CatalogKind::table, "table"}, {CatalogKind::tensor, "tensor"}, {CatalogKind::live, "live"},
    {CatalogKind::feed, "feed"},   {CatalogKind::metric, "metric"},
};

struct DtypeInfo {
    CatalogDtype dtype;
    const char* name;
    std::size_t size;
};
constexpr DtypeInfo kDtypes[] = {
    {CatalogDtype::u8, "u8", 1},   {CatalogDtype::i8, "i8", 1},   {CatalogDtype::u16, "u16", 2},
    {CatalogDtype::i16, "i16", 2}, {CatalogDtype::u32, "u32", 4}, {CatalogDtype::i32, "i32", 4},
    {CatalogDtype::u64, "u64", 8}, {CatalogDtype::i64, "i64", 8}, {CatalogDtype::f16, "f16", 2},
    {CatalogDtype::f32, "f32", 4}, {CatalogDtype::f64, "f64", 8}, {CatalogDtype::boolean, "bool", 1},
};

// The fault of a name or title, or an empty string.
std::string name_fault(std::string_view text) {
    if (text.empty()) return "is empty";
    if (text.size() > CatalogLimits::max_name) return "is longer than 256 bytes";
    if (!valid_utf8(text)) return "is not valid UTF-8";
    if (has_control(text)) return "holds a control character";
    if (text.front() == ' ' || text.back() == ' ') return "begins or ends with a space";
    return {};
}

// The fault of a bounded string value (an origin value, a column, a warning).
std::string text_fault(std::string_view text, std::size_t max_bytes) {
    if (text.empty()) return "is empty";
    if (text.size() > max_bytes) return "is longer than " + std::to_string(max_bytes) + " bytes";
    if (!valid_utf8(text)) return "is not valid UTF-8";
    if (has_control(text)) return "holds a control character";
    return {};
}

bool get_u64(const json& v, std::uint64_t& out) {
    if (v.is_number_unsigned()) {
        out = v.get<std::uint64_t>();
        return true;
    }
    if (v.is_number_integer()) {
        const auto x = v.get<std::int64_t>();
        if (x < 0) return false;
        out = static_cast<std::uint64_t>(x);
        return true;
    }
    if (v.is_number_float()) {
        const double d = v.get<double>();
        if (!std::isfinite(d) || d < 0 || d != std::floor(d) || d >= 18446744073709551616.0) return false;
        out = static_cast<std::uint64_t>(d);
        return true;
    }
    return false;
}

bool get_i64(const json& v, std::int64_t& out) {
    if (v.is_number_unsigned()) {
        const auto x = v.get<std::uint64_t>();
        if (x > static_cast<std::uint64_t>(INT64_MAX)) return false;
        out = static_cast<std::int64_t>(x);
        return true;
    }
    if (v.is_number_integer()) {
        out = v.get<std::int64_t>();
        return true;
    }
    if (v.is_number_float()) {
        const double d = v.get<double>();
        if (!std::isfinite(d) || d != std::floor(d) || d < -9223372036854775808.0 ||
            d >= 9223372036854775808.0)
            return false;
        out = static_cast<std::int64_t>(d);
        return true;
    }
    return false;
}

// The string members of CatalogOrigin, in the order of the JSON form.
struct OriginField {
    const char* key;
    std::optional<std::string> CatalogOrigin::*member;
};
const OriginField kOriginFields[] = {
    {"loader", &CatalogOrigin::loader},       {"file", &CatalogOrigin::file},
    {"output", &CatalogOrigin::output},       {"tensor", &CatalogOrigin::tensor},
    {"database", &CatalogOrigin::database},   {"schema", &CatalogOrigin::schema},
    {"table", &CatalogOrigin::table},         {"session", &CatalogOrigin::session},
    {"view", &CatalogOrigin::view},           {"view_type", &CatalogOrigin::view_type},
    {"service", &CatalogOrigin::service},     {"channel", &CatalogOrigin::channel},
    {"units", &CatalogOrigin::units},         {"experiment", &CatalogOrigin::experiment},
    {"run_name", &CatalogOrigin::run_name},   {"tag", &CatalogOrigin::tag},
};

constexpr const char* kEntryKeys[] = {"id",    "kind",    "name",  "title",   "origin",
                                      "columns", "rows",  "dtype", "shape",   "rate_hz",
                                      "bytes", "wire",    "warnings", "updated_at"};

Error validate_origin(const CatalogOrigin& origin) {
    for (const auto& field : kOriginFields) {
        const auto& value = origin.*(field.member);
        if (!value) continue;
        const std::string fault = text_fault(*value, CatalogLimits::max_origin_value);
        if (!fault.empty()) return invalid("The origin value " + fault + ".", std::string("origin.") + field.key);
    }
    if (origin.run && *origin.run < 0) return invalid("The origin's run id is negative.", "origin.run");
    return {};
}

Error validate_shape(const std::vector<std::int64_t>& shape) {
    if (shape.size() > CatalogLimits::max_rank)
        return invalid("The shape has more than 8 dimensions.", "shape");
    std::int64_t product = 1;
    bool zero = false;
    for (std::size_t i = 0; i < shape.size(); ++i) {
        const std::int64_t d = shape[i];
        const std::string where = "shape[" + std::to_string(i) + "]";
        if (d < 0) return invalid("A size of the shape is negative.", where);
        if (d > CatalogLimits::max_elements) return invalid("A size of the shape is too large.", where);
        if (d == 0) zero = true;
        if (!zero) {
            if (product > CatalogLimits::max_elements / d)
                return invalid("The shape holds more than 2^62 elements.", "shape");
            product *= d;
        }
    }
    return {};
}

}  // namespace

const char* catalog_kind_name(CatalogKind kind) {
    for (const auto& k : kKinds)
        if (k.kind == kind) return k.name;
    return nullptr;
}

std::optional<CatalogKind> catalog_kind_from_name(std::string_view name) {
    for (const auto& k : kKinds)
        if (name == k.name) return k.kind;
    return std::nullopt;
}

bool catalog_kind_is_tabular(CatalogKind kind) {
    return kind == CatalogKind::table || kind == CatalogKind::feed || kind == CatalogKind::metric;
}

const char* catalog_dtype_name(CatalogDtype dtype) {
    for (const auto& d : kDtypes)
        if (d.dtype == dtype) return d.name;
    return nullptr;
}

std::optional<CatalogDtype> catalog_dtype_from_name(std::string_view name) {
    for (const auto& d : kDtypes)
        if (name == d.name) return d.dtype;
    return std::nullopt;
}

std::size_t catalog_dtype_size(CatalogDtype dtype) {
    for (const auto& d : kDtypes)
        if (d.dtype == dtype) return d.size;
    return 0;
}

std::string make_catalog_id(CatalogKind kind, std::string_view name) {
    const char* k = catalog_kind_name(kind);
    std::string id = k ? k : "";
    id += ':';
    id.append(name.data(), name.size());
    return id;
}

CatalogIdParse parse_catalog_id(std::string_view id) {
    CatalogIdParse out;
    const auto colon = id.find(':');
    if (colon == std::string_view::npos) {
        out.error = Error("catalog.invalid_id", "A source id is <kind>:<name>, and this one has no colon.", "id");
        return out;
    }
    const auto kind = catalog_kind_from_name(id.substr(0, colon));
    if (!kind) {
        out.error = Error("catalog.invalid_id",
                          "The kind of a source id is one of table, tensor, live, feed and metric.", "id");
        return out;
    }
    const std::string_view name = id.substr(colon + 1);
    const std::string fault = name_fault(name);
    if (!fault.empty()) {
        out.error = Error("catalog.invalid_id", "The name of the source id " + fault + ".", "id");
        return out;
    }
    out.kind = *kind;
    out.name = std::string(name);
    return out;
}

Error check_catalog_name(std::string_view text, std::string_view where) {
    const std::string fault = name_fault(text);
    if (fault.empty()) return {};
    return invalid("The text " + fault + ".", std::string(where));
}

bool is_utc_timestamp(std::string_view t) {
    // YYYY-MM-DDTHH:MM:SS[.f+]Z
    if (t.size() < 20 || t.size() > 32) return false;
    auto digits = [&](std::size_t pos, std::size_t n, int& value) {
        value = 0;
        for (std::size_t i = pos; i < pos + n; ++i) {
            if (t[i] < '0' || t[i] > '9') return false;
            value = value * 10 + (t[i] - '0');
        }
        return true;
    };
    int year, month, day, hour, minute, second;
    if (!digits(0, 4, year) || t[4] != '-' || !digits(5, 2, month) || t[7] != '-' || !digits(8, 2, day) ||
        t[10] != 'T' || !digits(11, 2, hour) || t[13] != ':' || !digits(14, 2, minute) || t[16] != ':' ||
        !digits(17, 2, second))
        return false;
    if (month < 1 || month > 12 || day < 1 || day > 31 || hour > 23 || minute > 59 || second > 60) return false;
    std::size_t pos = 19;
    if (t[pos] == '.') {
        ++pos;
        const std::size_t start = pos;
        while (pos < t.size() && t[pos] >= '0' && t[pos] <= '9') ++pos;
        if (pos == start || pos - start > 9) return false;
    }
    return pos + 1 == t.size() && t[pos] == 'Z';
}

Error validate_catalog_entry(const CatalogEntry& e) {
    const char* kind = catalog_kind_name(e.kind);
    if (!kind) return invalid("The kind is not one of the five kinds.", "kind");
    if (const std::string fault = name_fault(e.name); !fault.empty())
        return invalid("The name " + fault + ".", "name");
    if (e.id != make_catalog_id(e.kind, e.name))
        return invalid("The id must be the kind and the name joined by a colon, " + make_catalog_id(e.kind, e.name) +
                           ".",
                       "id");
    if (const std::string fault = name_fault(e.title); !fault.empty())
        return invalid("The title " + fault + ".", "title");
    if (Error err = validate_origin(e.origin)) return err;

    const bool tabular = catalog_kind_is_tabular(e.kind);
    if (tabular) {
        if (!e.columns) return invalid(std::string("A ") + kind + " source must list its columns.", "columns");
        if (e.dtype) return invalid(std::string("A ") + kind + " source has no dtype; it must be null.", "dtype");
        if (e.shape) return invalid(std::string("A ") + kind + " source has no shape; it must be null.", "shape");
    } else {
        if (e.columns) return invalid(std::string("A ") + kind + " source has no columns; they must be null.", "columns");
        if (e.rows) return invalid(std::string("A ") + kind + " source has no row count; it must be null.", "rows");
        if (e.kind == CatalogKind::tensor) {
            if (!e.dtype) return invalid("A tensor source must give its dtype.", "dtype");
            if (!e.shape) return invalid("A tensor source must give its shape.", "shape");
        } else if (e.dtype.has_value() != e.shape.has_value()) {
            return invalid("A live source gives both its dtype and its shape, or neither before its first snapshot.",
                           e.dtype ? "shape" : "dtype");
        }
    }
    if (e.columns) {
        if (e.columns->size() > CatalogLimits::max_columns)
            return invalid("A source has at most 4096 columns.", "columns");
        for (std::size_t i = 0; i < e.columns->size(); ++i) {
            const auto& c = (*e.columns)[i];
            const std::string at = "columns[" + std::to_string(i) + "]";
            if (const std::string f = text_fault(c.name, CatalogLimits::max_column_name); !f.empty())
                return invalid("The column name " + f + ".", at + ".name");
            if (const std::string f = text_fault(c.type, CatalogLimits::max_column_type); !f.empty())
                return invalid("The column type " + f + ".", at + ".type");
        }
    }
    if (e.dtype && !catalog_dtype_name(*e.dtype)) return invalid("The dtype is not a known dtype.", "dtype");
    if (e.shape)
        if (Error err = validate_shape(*e.shape)) return err;
    if (e.rate_hz) {
        if (e.kind == CatalogKind::table || e.kind == CatalogKind::metric)
            return invalid(std::string("A ") + kind + " source has no rate; it must be null.", "rate_hz");
        const double r = *e.rate_hz;
        if (!std::isfinite(r) || r <= 0 || r > CatalogLimits::max_rate_hz)
            return invalid("The rate must be a positive number of hertz, at most 1e9.", "rate_hz");
    }
    if (e.warnings.size() > CatalogLimits::max_warnings)
        return invalid("An entry carries at most 16 warnings.", "warnings");
    for (std::size_t i = 0; i < e.warnings.size(); ++i)
        if (const std::string f = text_fault(e.warnings[i], CatalogLimits::max_warning); !f.empty())
            return invalid("The warning " + f + ".", "warnings[" + std::to_string(i) + "]");
    if (e.updated_at && !is_utc_timestamp(*e.updated_at))
        return invalid("updated_at must be a UTC time such as 2026-11-15T12:00:00Z.", "updated_at");
    return {};
}

json catalog_column_to_json(const CatalogColumn& column) {
    return json{{"name", column.name}, {"type", column.type}};
}

json catalog_origin_to_json(const CatalogOrigin& origin) {
    json out = json::object();
    for (const auto& field : kOriginFields) {
        const auto& value = origin.*(field.member);
        if (value) out[field.key] = *value;
    }
    if (origin.run) out["run"] = *origin.run;
    return out;
}

json catalog_entry_to_json(const CatalogEntry& e) {
    json out = json::object();
    out["id"] = e.id;
    const char* kind = catalog_kind_name(e.kind);
    out["kind"] = kind ? kind : "";
    out["name"] = e.name;
    out["title"] = e.title;
    out["origin"] = catalog_origin_to_json(e.origin);
    if (e.columns) {
        json cols = json::array();
        for (const auto& c : *e.columns) cols.push_back(catalog_column_to_json(c));
        out["columns"] = std::move(cols);
    } else {
        out["columns"] = nullptr;
    }
    out["rows"] = e.rows ? json(*e.rows) : json(nullptr);
    const char* dtype = e.dtype ? catalog_dtype_name(*e.dtype) : nullptr;
    out["dtype"] = dtype ? json(dtype) : json(nullptr);
    out["shape"] = e.shape ? json(*e.shape) : json(nullptr);
    out["rate_hz"] = e.rate_hz ? json(*e.rate_hz) : json(nullptr);
    out["bytes"] = e.bytes ? json(*e.bytes) : json(nullptr);
    out["wire"] = e.wire;
    out["warnings"] = e.warnings;
    out["updated_at"] = e.updated_at ? json(*e.updated_at) : json(nullptr);
    return out;
}

CatalogEntryParse catalog_entry_from_json(const json& v) {
    CatalogEntryParse out;
    auto fail = [&](std::string message, std::string where) {
        out.error = invalid(std::move(message), std::move(where));
        return out;
    };
    if (!v.is_object()) return fail("A catalog entry is a JSON object.", "");
    for (const auto& item : v.items()) {
        bool known = false;
        for (const char* k : kEntryKeys) known = known || item.key() == k;
        if (!known) return fail("A catalog entry has no key of this name.", item.key());
    }
    for (const char* k : kEntryKeys)
        if (!v.contains(k)) return fail("A catalog entry must hold every key, with null where it does not apply.", k);

    CatalogEntry& e = out.entry;
    const json& id = v["id"];
    if (!id.is_string()) return fail("The id is a string.", "id");
    e.id = id.get<std::string>();
    const json& kind = v["kind"];
    if (!kind.is_string()) return fail("The kind is a string.", "kind");
    const auto k = catalog_kind_from_name(kind.get_ref<const std::string&>());
    if (!k) return fail("The kind is one of table, tensor, live, feed and metric.", "kind");
    e.kind = *k;
    if (!v["name"].is_string()) return fail("The name is a string.", "name");
    e.name = v["name"].get<std::string>();
    if (!v["title"].is_string()) return fail("The title is a string.", "title");
    e.title = v["title"].get<std::string>();

    const json& origin = v["origin"];
    if (!origin.is_object()) return fail("The origin is an object.", "origin");
    for (const auto& item : origin.items()) {
        const std::string where = "origin." + item.key();
        if (item.key() == "run") {
            std::int64_t run = 0;
            if (!get_i64(item.value(), run)) return fail("The origin's run is an integer.", where);
            e.origin.run = run;
            continue;
        }
        const OriginField* field = nullptr;
        for (const auto& f : kOriginFields)
            if (item.key() == f.key) field = &f;
        if (!field) return fail("The origin has no key of this name.", where);
        if (!item.value().is_string()) return fail("An origin value is a string.", where);
        e.origin.*(field->member) = item.value().get<std::string>();
    }

    const json& columns = v["columns"];
    if (!columns.is_null()) {
        if (!columns.is_array()) return fail("The columns are a list or null.", "columns");
        if (columns.size() > CatalogLimits::max_columns) return fail("A source has at most 4096 columns.", "columns");
        std::vector<CatalogColumn> cols;
        cols.reserve(columns.size());
        for (std::size_t i = 0; i < columns.size(); ++i) {
            const json& c = columns[i];
            const std::string at = "columns[" + std::to_string(i) + "]";
            if (!c.is_object()) return fail("A column is an object with a name and a type.", at);
            for (const auto& item : c.items())
                if (item.key() != "name" && item.key() != "type")
                    return fail("A column has only a name and a type.", at + "." + item.key());
            if (!c.contains("name") || !c["name"].is_string()) return fail("A column's name is a string.", at + ".name");
            if (!c.contains("type") || !c["type"].is_string()) return fail("A column's type is a string.", at + ".type");
            cols.push_back(CatalogColumn{c["name"].get<std::string>(), c["type"].get<std::string>()});
        }
        e.columns = std::move(cols);
    }

    const json& rows = v["rows"];
    if (!rows.is_null()) {
        std::uint64_t n = 0;
        if (!get_u64(rows, n)) return fail("The row count is a non-negative integer or null.", "rows");
        e.rows = n;
    }
    const json& dtype = v["dtype"];
    if (!dtype.is_null()) {
        if (!dtype.is_string()) return fail("The dtype is a string or null.", "dtype");
        const auto d = catalog_dtype_from_name(dtype.get_ref<const std::string&>());
        if (!d) return fail("The dtype is one of u8 i8 u16 i16 u32 i32 u64 i64 f16 f32 f64 bool.", "dtype");
        e.dtype = *d;
    }
    const json& shape = v["shape"];
    if (!shape.is_null()) {
        if (!shape.is_array()) return fail("The shape is a list of sizes or null.", "shape");
        if (shape.size() > CatalogLimits::max_rank) return fail("The shape has more than 8 dimensions.", "shape");
        std::vector<std::int64_t> dims;
        for (std::size_t i = 0; i < shape.size(); ++i) {
            std::int64_t d = 0;
            if (!get_i64(shape[i], d)) return fail("A size of the shape is an integer.", "shape[" + std::to_string(i) + "]");
            dims.push_back(d);
        }
        e.shape = std::move(dims);
    }
    const json& rate = v["rate_hz"];
    if (!rate.is_null()) {
        if (!rate.is_number()) return fail("The rate is a number or null.", "rate_hz");
        e.rate_hz = rate.get<double>();
    }
    const json& bytes = v["bytes"];
    if (!bytes.is_null()) {
        std::uint64_t n = 0;
        if (!get_u64(bytes, n)) return fail("The size in bytes is a non-negative integer or null.", "bytes");
        e.bytes = n;
    }
    if (!v["wire"].is_boolean()) return fail("wire is true or false.", "wire");
    e.wire = v["wire"].get<bool>();
    const json& warnings = v["warnings"];
    if (!warnings.is_array()) return fail("The warnings are a list of strings.", "warnings");
    if (warnings.size() > CatalogLimits::max_warnings) return fail("An entry carries at most 16 warnings.", "warnings");
    for (std::size_t i = 0; i < warnings.size(); ++i) {
        if (!warnings[i].is_string()) return fail("A warning is a string.", "warnings[" + std::to_string(i) + "]");
        e.warnings.push_back(warnings[i].get<std::string>());
    }
    const json& updated = v["updated_at"];
    if (!updated.is_null()) {
        if (!updated.is_string()) return fail("updated_at is a string or null.", "updated_at");
        e.updated_at = updated.get<std::string>();
    }
    out.error = validate_catalog_entry(e);
    return out;
}

}  // namespace caliper_builder
