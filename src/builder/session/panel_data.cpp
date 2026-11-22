// panel_data.cpp: one panel's resolved data, compact and bounded, for the
// browser's spec page (panel_data.h). Each primitive keeps the steps of the
// native spec runtime's prepare_panel() that bound the data
// (src/host/spec_runtime/spec_prepare.cpp), in the same order, so that the
// page draws what the app draws.
#include <caliper_builder/session/panel_data.h>

#include <caliper_builder/commands/sha256.h>
#include <caliper_builder/spec/format.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <vector>

namespace caliper_builder {

namespace {

using nlohmann::json;

constexpr double kNaN = std::numeric_limits<double>::quiet_NaN();

const char kB64[] = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

std::string base64(const unsigned char* p, std::size_t n) {
    std::string out;
    out.reserve((n + 2) / 3 * 4);
    std::size_t i = 0;
    for (; i + 2 < n; i += 3) {
        const std::uint32_t v = (std::uint32_t(p[i]) << 16) | (std::uint32_t(p[i + 1]) << 8) | p[i + 2];
        out += kB64[(v >> 18) & 63];
        out += kB64[(v >> 12) & 63];
        out += kB64[(v >> 6) & 63];
        out += kB64[v & 63];
    }
    if (i + 1 == n) {
        const std::uint32_t v = std::uint32_t(p[i]) << 16;
        out += kB64[(v >> 18) & 63];
        out += kB64[(v >> 12) & 63];
        out += "==";
    } else if (i + 2 == n) {
        const std::uint32_t v = (std::uint32_t(p[i]) << 16) | (std::uint32_t(p[i + 1]) << 8);
        out += kB64[(v >> 18) & 63];
        out += kB64[(v >> 12) & 63];
        out += kB64[(v >> 6) & 63];
        out += '=';
    }
    return out;
}

bool little_endian() {
    const std::uint16_t one = 1;
    unsigned char b = 0;
    std::memcpy(&b, &one, 1);
    return b == 1;
}

// An array of numbers as little-endian bytes in base64.
template <typename T>
json array_json(const std::vector<T>& v, std::vector<std::int64_t> shape) {
    static_assert(sizeof(T) == 4 || sizeof(T) == 8, "f32 or f64");
    std::vector<unsigned char> bytes(v.size() * sizeof(T));
    if (!v.empty()) std::memcpy(bytes.data(), v.data(), bytes.size());
    if (!little_endian())
        for (std::size_t i = 0; i < bytes.size(); i += sizeof(T)) std::reverse(bytes.begin() + i, bytes.begin() + i + sizeof(T));
    return {{"dtype", sizeof(T) == 4 ? "f32" : "f64"}, {"shape", std::move(shape)}, {"base64", base64(bytes.data(), bytes.size())}};
}

template <typename T>
json vector_json(const std::vector<T>& v) {
    return array_json(v, {static_cast<std::int64_t>(v.size())});
}

const SlotData* slot(const PanelData& d, const char* name, std::size_t index = 0) {
    for (const auto& s : d.slots)
        if (s.slot == name && s.index == index) return &s;
    return nullptr;
}

std::vector<const SlotData*> slots(const PanelData& d, const char* name) {
    std::vector<const SlotData*> out;
    for (const auto& s : d.slots)
        if (s.slot == name) out.push_back(&s);
    std::sort(out.begin(), out.end(), [](const SlotData* a, const SlotData* b) { return a->index < b->index; });
    return out;
}

// A cell as a number: numbers, booleans and the catalog's spellings of the
// non-finite values; NaN for anything else (spec_prepare.cpp's cell_number).
double cell_number(const json& c) {
    if (c.is_number()) return c.get<double>();
    if (c.is_boolean()) return c.get<bool>() ? 1.0 : 0.0;
    if (c.is_string()) {
        const std::string& s = c.get_ref<const std::string&>();
        if (s == "nan") return kNaN;
        if (s == "inf") return std::numeric_limits<double>::infinity();
        if (s == "-inf") return -std::numeric_limits<double>::infinity();
        char* end = nullptr;
        const double v = std::strtod(s.c_str(), &end);
        if (end && *end == '\0' && end != s.c_str()) return v;
    }
    return kNaN;
}

// A vector slot (a column, or a rank-1 tensor) as doubles; false otherwise.
bool vector_of(const SlotData& s, std::vector<double>* out) {
    out->clear();
    if (s.kind == SlotData::Kind::column) {
        if (!s.values.is_array()) return false;
        out->reserve(s.values.size());
        for (const auto& c : s.values) out->push_back(cell_number(c));
        return true;
    }
    if (s.kind == SlotData::Kind::tensor) {
        const TensorData& t = s.tensor;
        if (!t.valid() || t.rank() != 1) return false;
        out->reserve(static_cast<std::size_t>(t.numel()));
        for (std::uint64_t i = 0; i < t.numel(); ++i) out->push_back(t.element(i));
        return true;
    }
    return false;
}

// Whether a vector slot holds an integer type: an integer or boolean tensor,
// or a column of an integer or boolean DuckDB type (spec_prepare.cpp's
// integer_type), which the page's class-label rule needs.
bool integer_slot(const SlotData& s) {
    if (s.kind == SlotData::Kind::tensor) return s.tensor.valid() && !dtype_is_float(s.tensor.dtype());
    if (s.kind != SlotData::Kind::column || s.columns.empty()) return false;
    const std::string& t = s.columns[0].type;
    static const char* const kInteger[] = {"TINYINT", "SMALLINT", "INTEGER", "BIGINT", "HUGEINT", "UTINYINT",
                                           "USMALLINT", "UINTEGER", "UBIGINT", "UHUGEINT", "BOOLEAN", "INT"};
    for (const char* n : kInteger) {
        const std::size_t len = std::strlen(n);
        if (t == n || (t.rfind(n, 0) == 0 && t.size() > len && t[len] >= '0' && t[len] <= '9')) return true;
    }
    return false;
}

struct Encoded {
    json state = nullptr;   // {"kind", "detail"}
    json data = nullptr;
    json bounds = json::array();
};

void mismatch(Encoded& e, const std::string& detail) { e.state = {{"kind", "mismatch"}, {"detail", detail}}; }
void empty(Encoded& e, const std::string& detail) { e.state = {{"kind", "empty"}, {"detail", detail}}; }

// ---- the primitives -------------------------------------------------------------

struct Line {
    std::string binding;
    json row = nullptr;          // the row of a rank-2 tensor, or null
    std::vector<double> x, y;
    std::size_t count = 0;       // values before decimation
    double last = 0.0;
    bool has_last = false;
};

// The min/max envelope of y over index order into about 2 * buckets points
// (spec_prepare.cpp's decimate, the live series' rule).
void decimate(Line& l, std::size_t buckets) {
    const std::size_t n = l.y.size();
    if (buckets == 0 || n <= 2 * buckets) return;
    std::vector<double> xs, ys;
    xs.reserve(2 * buckets + 2);
    ys.reserve(2 * buckets + 2);
    const std::size_t per = (n + buckets - 1) / buckets;
    for (std::size_t b0 = 0; b0 < n; b0 += per) {
        const std::size_t b1 = std::min(n, b0 + per);
        std::size_t imin = b0, imax = b0;
        for (std::size_t i = b0; i < b1; ++i) {
            if (l.y[i] < l.y[imin]) imin = i;
            if (l.y[i] > l.y[imax]) imax = i;
        }
        const std::size_t a = std::min(imin, imax), b = std::max(imin, imax);
        xs.push_back(l.x[a]);
        ys.push_back(l.y[a]);
        if (b != a) {
            xs.push_back(l.x[b]);
            ys.push_back(l.y[b]);
        }
    }
    l.x.swap(xs);
    l.y.swap(ys);
}

void encode_series(Encoded& e, const spec::Panel& panel, const PanelData& d, const PanelDataLimits& lim) {
    const bool x_log = panel.options.x_log.value_or(false), y_log = panel.options.y_log.value_or(false);
    std::vector<double> xs;
    const SlotData* xslot = slot(d, "x");
    if (xslot && !vector_of(*xslot, &xs)) return mismatch(e, xslot->binding);
    struct Raw {
        std::string binding;
        json row = nullptr;
        std::vector<double> y;
    };
    std::vector<Raw> raw;
    for (const SlotData* ys : slots(d, "y")) {
        if (ys->kind == SlotData::Kind::tensor && ys->tensor.valid() && ys->tensor.rank() == 2) {
            const TensorData& t = ys->tensor;
            const std::int64_t rows = t.shape()[0], cols = t.shape()[1];
            for (std::int64_t r = 0; r < rows && raw.size() < lim.series_rows; ++r) {
                Raw line{ys->binding, r, {}};
                line.y.reserve(static_cast<std::size_t>(cols));
                for (std::int64_t c = 0; c < cols; ++c) line.y.push_back(t.element(static_cast<std::uint64_t>(r * cols + c)));
                raw.push_back(std::move(line));
            }
            if (rows > static_cast<std::int64_t>(lim.series_rows))
                e.bounds.push_back("lines cut at " + std::to_string(lim.series_rows) + " rows");
            continue;
        }
        Raw line{ys->binding, nullptr, {}};
        if (!vector_of(*ys, &line.y)) return mismatch(e, ys->binding);
        raw.push_back(std::move(line));
    }
    if (raw.empty()) return empty(e, panel.id);
    std::size_t steps = 0;
    bool any = false;
    json lines = json::array();
    for (Raw& r : raw) {
        Line l;
        l.binding = r.binding;
        l.row = r.row;
        const std::size_t n = xslot ? std::min(r.y.size(), xs.size()) : r.y.size();
        l.count = n;
        steps = std::max(steps, n);
        l.x.reserve(n);
        l.y.reserve(n);
        for (std::size_t i = 0; i < n; ++i) {
            const double x = xslot ? xs[i] : static_cast<double>(i), y = r.y[i];
            if (!std::isfinite(x) || !std::isfinite(y)) continue;
            l.last = y;
            l.has_last = true;
            if ((x_log && x <= 0.0) || (y_log && y <= 0.0)) continue;   // no place on a log axis
            l.x.push_back(x);
            l.y.push_back(y);
        }
        if (l.y.size() > lim.series_raw_max) {
            const std::size_t before = l.y.size();
            decimate(l, lim.series_buckets);
            const std::string name = l.binding + (l.row.is_null() ? "" : "[" + l.row.dump() + "]");
            e.bounds.push_back("series " + name + " decimated from " + std::to_string(before) + " to " +
                               std::to_string(l.y.size()) + " points");
        }
        any = any || !l.y.empty();
        lines.push_back({{"binding", l.binding},
                         {"row", l.row},
                         {"count", l.count},
                         {"last", l.has_last ? json(l.last) : json(nullptr)},
                         {"x", vector_json(l.x)},
                         {"y", vector_json(l.y)}});
    }
    if (!any) return empty(e, raw.front().binding);
    e.data = {{"x_index", xslot == nullptr}, {"steps", steps}, {"lines", std::move(lines)}};
}

std::int64_t stride_for(std::int64_t n, std::int64_t max_side) {
    if (max_side <= 0 || n <= max_side) return 1;
    return (n + max_side - 1) / max_side;
}

void encode_heatmap(Encoded& e, const spec::Panel& panel, const PanelData& d, const PanelDataLimits& lim) {
    const SlotData* v = slot(d, "values");
    if (!v) return empty(e, panel.id);
    if (v->kind != SlotData::Kind::tensor || !v->tensor.valid() || v->tensor.rank() != 2) return mismatch(e, v->binding);
    const TensorData& t = v->tensor;
    const std::int64_t R = t.shape()[0], C = t.shape()[1];
    if (R <= 0 || C <= 0) return empty(e, v->binding);
    const std::int64_t sr = stride_for(R, lim.max_texture_side), sc = stride_for(C, lim.max_texture_side);
    const std::int64_t rows = (R + sr - 1) / sr, cols = (C + sc - 1) / sc;
    if (sr > 1 || sc > 1)
        e.bounds.push_back("heatmap sampled from " + std::to_string(R) + "x" + std::to_string(C) + " to " +
                           std::to_string(rows) + "x" + std::to_string(cols));
    std::vector<float> values(static_cast<std::size_t>(rows * cols));
    for (std::int64_t r = 0; r < rows; ++r)
        for (std::int64_t c = 0; c < cols; ++c)
            values[static_cast<std::size_t>(r * cols + c)] =
                static_cast<float>(t.element(static_cast<std::uint64_t>(r * sr * C + c * sc)));
    e.data = {{"source_rows", R}, {"source_cols", C}, {"stride_rows", sr}, {"stride_cols", sc},
              {"values", array_json(values, {rows, cols})}};
}

void encode_image(Encoded& e, const spec::Panel& panel, const PanelData& d, const PanelDataLimits& lim) {
    const SlotData* v = slot(d, "values");
    if (!v) return empty(e, panel.id);
    const TensorData& t = v->tensor;
    if (v->kind != SlotData::Kind::tensor || !t.valid() || (t.rank() != 2 && t.rank() != 3)) return mismatch(e, v->binding);
    const std::int64_t H = t.shape()[0], W = t.shape()[1], K = t.rank() == 3 ? t.shape()[2] : 1;
    if (K != 1 && K != 3 && K != 4) return mismatch(e, v->binding);
    if (H <= 0 || W <= 0) return empty(e, v->binding);
    const std::int64_t s = std::max(stride_for(H, lim.max_texture_side), stride_for(W, lim.max_texture_side));
    const std::int64_t rows = (H + s - 1) / s, cols = (W + s - 1) / s;
    if (s > 1)
        e.bounds.push_back("image sampled from " + std::to_string(H) + "x" + std::to_string(W) + " to " +
                           std::to_string(rows) + "x" + std::to_string(cols));
    std::vector<float> values(static_cast<std::size_t>(rows * cols * K));
    for (std::int64_t r = 0; r < rows; ++r)
        for (std::int64_t c = 0; c < cols; ++c)
            for (std::int64_t k = 0; k < K; ++k)
                values[static_cast<std::size_t>((r * cols + c) * K + k)] =
                    static_cast<float>(t.element(static_cast<std::uint64_t>(((r * s) * W + c * s) * K + k)));
    e.data = {{"source_rows", H}, {"source_cols", W}, {"stride", s}, {"channels", K},
              {"float", dtype_is_float(t.dtype())}, {"values", array_json(values, {rows, cols, K})}};
}

void encode_points(Encoded& e, const spec::Panel& panel, const PanelData& d, const PanelDataLimits& lim) {
    std::vector<double> xs, ys, zs;
    const SlotData* pos = slot(d, "positions");
    bool three_d = false;
    if (pos) {
        const TensorData& t = pos->tensor;
        if (pos->kind != SlotData::Kind::tensor || !t.valid() || t.rank() != 2 || (t.shape()[1] != 2 && t.shape()[1] != 3))
            return mismatch(e, pos->binding);
        const std::int64_t n = t.shape()[0], k = t.shape()[1];
        three_d = k == 3;
        xs.resize(static_cast<std::size_t>(n));
        ys.resize(static_cast<std::size_t>(n));
        if (three_d) zs.resize(static_cast<std::size_t>(n));
        for (std::int64_t i = 0; i < n; ++i) {
            xs[static_cast<std::size_t>(i)] = t.element(static_cast<std::uint64_t>(i * k));
            ys[static_cast<std::size_t>(i)] = t.element(static_cast<std::uint64_t>(i * k + 1));
            if (three_d) zs[static_cast<std::size_t>(i)] = t.element(static_cast<std::uint64_t>(i * k + 2));
        }
    } else {
        const SlotData* sx = slot(d, "x");
        const SlotData* sy = slot(d, "y");
        const SlotData* sz = slot(d, "z");
        if (!sx || !sy) return empty(e, panel.id);
        if (!vector_of(*sx, &xs)) return mismatch(e, sx->binding);
        if (!vector_of(*sy, &ys)) return mismatch(e, sy->binding);
        if (sz && !vector_of(*sz, &zs)) return mismatch(e, sz->binding);
        three_d = !zs.empty();
    }
    std::vector<double> cs, ss;
    bool integer = false;
    if (const SlotData* sc = slot(d, "color")) {
        if (!vector_of(*sc, &cs)) return mismatch(e, sc->binding);
        integer = integer_slot(*sc);
    }
    if (const SlotData* sz = slot(d, "size")) {
        if (!vector_of(*sz, &ss)) return mismatch(e, sz->binding);
    }
    std::size_t n = std::min(xs.size(), ys.size());
    if (three_d) n = std::min(n, zs.size());
    if (!cs.empty()) n = std::min(n, cs.size());
    if (!ss.empty()) n = std::min(n, ss.size());
    if (n == 0) return empty(e, panel.id);
    const std::size_t stride = lim.max_points && n > lim.max_points ? (n + lim.max_points - 1) / lim.max_points : 1;
    if (stride > 1)
        e.bounds.push_back("points sampled from " + std::to_string(n) + " at a stride of " + std::to_string(stride));
    std::vector<float> px, py, pz, pc, ps;
    for (std::size_t i = 0; i < n; i += stride) {
        if (!std::isfinite(xs[i]) || !std::isfinite(ys[i]) || (three_d && !std::isfinite(zs[i]))) continue;
        px.push_back(static_cast<float>(xs[i]));
        py.push_back(static_cast<float>(ys[i]));
        if (three_d) pz.push_back(static_cast<float>(zs[i]));
        if (!cs.empty()) pc.push_back(static_cast<float>(cs[i]));
        if (!ss.empty()) ps.push_back(static_cast<float>(ss[i]));
    }
    if (px.empty()) return empty(e, panel.id);
    json out = {{"three_d", three_d}, {"total", n}, {"stride", stride}, {"x", vector_json(px)}, {"y", vector_json(py)},
                {"z", three_d ? vector_json(pz) : json(nullptr)},
                {"color", cs.empty() ? json(nullptr) : vector_json(pc)},
                {"color_integer", integer},
                {"size", ss.empty() ? json(nullptr) : vector_json(ps)}};
    e.data = std::move(out);
}

void encode_histogram(Encoded& e, const spec::Panel& panel, const PanelData& d) {
    const SlotData* v = slot(d, "values");
    if (!v) return empty(e, panel.id);
    std::vector<double> xs;
    if (v->kind == SlotData::Kind::column) {
        vector_of(*v, &xs);
    } else if (v->kind == SlotData::Kind::tensor && v->tensor.valid()) {
        xs.reserve(static_cast<std::size_t>(v->tensor.numel()));
        for (std::uint64_t i = 0; i < v->tensor.numel(); ++i) xs.push_back(v->tensor.element(i));
    } else {
        return mismatch(e, v->binding);
    }
    double lo = INFINITY, hi = -INFINITY;
    for (double x : xs) {
        if (!std::isfinite(x)) continue;
        lo = std::min(lo, x);
        hi = std::max(hi, x);
    }
    if (!(hi >= lo)) return empty(e, v->binding);
    if (panel.options.range) {
        lo = (*panel.options.range)[0];
        hi = (*panel.options.range)[1];
    }
    if (hi == lo) {   // a flat range widens by 0.5 each way, as the live views do
        lo -= 0.5;
        hi += 0.5;
    }
    const std::int64_t bins = std::clamp<std::int64_t>(panel.options.bins.value_or(64), 1, spec::limits::kMaxBins);
    std::vector<double> counts(static_cast<std::size_t>(bins), 0.0);
    double total = 0.0;
    std::uint64_t outside = 0;
    const double w = (hi - lo) / static_cast<double>(bins);
    for (double x : xs) {
        if (!std::isfinite(x)) continue;
        if (x < lo || x > hi) {
            ++outside;
            continue;
        }
        const std::int64_t b = std::min<std::int64_t>(bins - 1, static_cast<std::int64_t>(std::floor((x - lo) / w)));
        counts[static_cast<std::size_t>(b)] += 1.0;
        total += 1.0;
    }
    if (outside) e.bounds.push_back(std::to_string(outside) + " values outside the range");
    e.data = {{"lo", lo}, {"hi", hi}, {"total", total}, {"outside", outside}, {"counts", counts}};
}

// A cell for the page: numbers, booleans and null as they are, a string cut
// at `max_chars` characters with an ellipsis and its control characters made
// spaces, anything else as its JSON text, cut the same way.
json table_cell(const json& c, std::size_t max_chars) {
    if (c.is_null() || c.is_boolean() || c.is_number()) return c;
    std::string s = c.is_string() ? c.get<std::string>() : c.dump(-1, ' ', false, json::error_handler_t::replace);
    // Cut at `max_chars` code points, on a character boundary.
    std::size_t points = 0, at = 0;
    while (at < s.size() && points < max_chars) {
        ++at;
        while (at < s.size() && (static_cast<unsigned char>(s[at]) & 0xC0) == 0x80) ++at;
        ++points;
    }
    if (at < s.size()) s = s.substr(0, at) + "\xE2\x80\xA6";
    for (char& ch : s)
        if (static_cast<unsigned char>(ch) < 0x20) ch = ' ';
    return s;
}

void encode_table(Encoded& e, const spec::Panel& panel, const PanelData& d, const PanelDataLimits& lim) {
    const SlotData* v = slot(d, "rows");
    if (!v) return empty(e, panel.id);
    if (v->kind != SlotData::Kind::table || !v->values.is_array()) return mismatch(e, v->binding);
    if (v->columns.empty()) return empty(e, v->binding);
    json columns = json::array();
    for (const auto& c : v->columns) columns.push_back(c.to_json());
    const std::size_t total = v->values.size(), shown = std::min(total, lim.max_table_rows);
    if (shown < total)
        e.bounds.push_back("table cut from " + std::to_string(total) + " to " + std::to_string(shown) + " rows");
    json rows = json::array();
    for (std::size_t r = 0; r < shown; ++r) {
        const json& row = v->values[r];
        json cells = json::array();
        for (std::size_t c = 0; c < v->columns.size(); ++c)
            cells.push_back(row.is_array() && c < row.size() ? table_cell(row[c], lim.max_cell_chars) : json(nullptr));
        rows.push_back(std::move(cells));
    }
    e.data = {{"columns", std::move(columns)}, {"rows", std::move(rows)}, {"total_rows", total},
              {"truncated", v->truncated}};
}

void encode_text(Encoded& e, const spec::Panel& panel, const PanelData& d, const PanelDataLimits& lim) {
    const bool has_text = panel.options.text && !panel.options.text->empty();
    json value = nullptr;
    if (const SlotData* v = slot(d, "value")) {
        if (v->kind == SlotData::Kind::column && v->values.is_array()) {
            for (std::size_t i = v->values.size(); i-- > 0;) {
                if (v->values[i].is_null()) continue;
                value = table_cell(v->values[i], lim.max_cell_chars);
                break;
            }
            if (value.is_null() && !has_text) return empty(e, v->binding);
        } else if (v->kind == SlotData::Kind::tensor && v->tensor.valid() && v->tensor.numel() == 1) {
            const double x = v->tensor.element(0);
            // A non-finite value as the native runtime writes it (spec_fmt4).
            value = std::isfinite(x) ? json(x) : json(std::isnan(x) ? "NaN" : x > 0 ? "inf" : "-inf");
        } else {
            return mismatch(e, v->binding);
        }
    }
    if (!has_text && value.is_null()) return empty(e, panel.id);
    e.data = {{"value", value}};
}

}  // namespace

std::string panel_data_version(const spec::Spec& spec, const std::string& panel_id, std::uint64_t catalog_revision,
                               std::uint64_t generation) {
    const std::string text = spec::to_json_text(spec) + "\n" + panel_id + "\n" + std::to_string(catalog_revision) +
                             "\n" + std::to_string(generation);
    return command_sha256_hex(text).substr(0, 16);
}

json panel_data_unchanged(const spec::Spec& spec, const std::string& panel_id, const std::string& version) {
    const spec::Panel* panel = spec.find_panel(panel_id);
    return {{"panel", panel_id},
            {"primitive", panel ? spec::to_string(panel->primitive) : ""},
            {"version", version},
            {"unchanged", true},
            {"error", nullptr},
            {"state", nullptr},
            {"bounds", json::array()},
            {"data", nullptr}};
}

json panel_data_result(const spec::Spec& spec, const PanelData& data, std::uint64_t catalog_revision,
                       const PanelDataLimits& limits) {
    const spec::Panel* panel = spec.find_panel(data.panel);
    json out = {{"panel", data.panel},
                {"primitive", panel ? spec::to_string(panel->primitive) : ""},
                {"version", panel_data_version(spec, data.panel, catalog_revision, data.generation)},
                {"unchanged", false},
                {"error", nullptr},
                {"state", nullptr},
                {"bounds", json::array()},
                {"data", nullptr}};
    if (data.error || !panel) {
        const Error err = data.error ? data.error : Error("panel.not_found", "The spec has no panel " + data.panel + ".", "");
        out["error"] = {{"code", err.code}, {"message", err.message}, {"where", err.where}};
        return out;
    }
    Encoded e;
    try {
        switch (panel->primitive) {
            case spec::Primitive::series: encode_series(e, *panel, data, limits); break;
            case spec::Primitive::heatmap: encode_heatmap(e, *panel, data, limits); break;
            case spec::Primitive::image: encode_image(e, *panel, data, limits); break;
            case spec::Primitive::points: encode_points(e, *panel, data, limits); break;
            case spec::Primitive::histogram: encode_histogram(e, *panel, data); break;
            case spec::Primitive::table: encode_table(e, *panel, data, limits); break;
            case spec::Primitive::text: encode_text(e, *panel, data, limits); break;
            case spec::Primitive::scene: {
                // The app draws the live scene itself; the data names it.
                auto it = panel->bind.find(spec::kSceneSlot);
                const std::optional<std::string> name =
                    it == panel->bind.end() || it->second.bindings.empty()
                        ? std::nullopt
                        : spec::live_scene_name(it->second.bindings.front());
                e.data = {{"scene", name ? json(*name) : json(nullptr)}};
                if (!panel->entities.empty()) encode_scene_entities(*panel, data, limits, &e.data);
                break;
            }
        }
    } catch (const std::exception&) {   // memory: the page shows a data error
        e = {};
        out["error"] = {{"code", "panel.data_failed"}, {"message", "The panel's data could not be encoded."}, {"where", ""}};
        return out;
    }
    out["state"] = std::move(e.state);
    out["bounds"] = std::move(e.bounds);
    if (out["state"].is_null()) out["data"] = std::move(e.data);
    return out;
}

}  // namespace caliper_builder
