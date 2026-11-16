// spec_prepare.cpp: the data of one spec panel made ready to draw
// (spec_prepare.h).
#include "spec_runtime/spec_prepare.h"

#include "tensor_bridge.h"   // colormap_lut, map_f32_to_rgba8: the live views' colours

#include <caliper/services/tensor_bridge_v1.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>

namespace caliper_host {

namespace spec = caliper_builder::spec;
using caliper_builder::DType;
using caliper_builder::PanelData;
using caliper_builder::SlotData;
using caliper_builder::TensorData;
using nlohmann::json;

namespace {

const char* const kTimes = "\xC3\x97";          // U+00D7
const char* const kEllipsis = "\xE2\x80\xA6";   // U+2026
constexpr double kNaN = std::numeric_limits<double>::quiet_NaN();

const SlotData* slot(const PanelData& d, const char* name, size_t index = 0) {
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

// A cell as a number: JSON numbers and booleans, and the strings the catalog
// writes for non-finite values; NaN for anything else.
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

// The name a binding draws under: its column, or its source.
std::string slot_name(const SlotData& s) { return s.kind == SlotData::Kind::column ? s.column : s.source; }

// A vector slot (a column, or a rank-1 tensor) as doubles. False when it is
// neither.
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
        out->reserve((size_t)t.numel());
        for (uint64_t i = 0; i < t.numel(); ++i) out->push_back(t.element(i));
        return true;
    }
    return false;
}

void finite_range(const std::vector<double>& v, double* lo, double* hi, bool positive_only = false) {
    *lo = INFINITY;
    *hi = -INFINITY;
    for (double x : v) {
        if (!std::isfinite(x) || (positive_only && x <= 0.0)) continue;
        *lo = std::min(*lo, x);
        *hi = std::max(*hi, x);
    }
}

// The live views' rule for a flat or missing range: widen a flat one by 0.5
// each way, and use [0, 1] when there are no finite values.
void settle_range(double* lo, double* hi) {
    if (!(*hi >= *lo)) {
        *lo = 0.0;
        *hi = 1.0;
    } else if (*hi == *lo) {
        *lo -= 0.5;
        *hi += 0.5;
    }
}

std::string with_commas(uint64_t n) {
    std::string s = std::to_string(n), out;
    for (size_t i = 0; i < s.size(); ++i) {
        if (i && (s.size() - i) % 3 == 0) out += ',';
        out += s[i];
    }
    return out;
}

// A cell as table text: integers in full, other numbers in six significant
// digits, strings as they are, nothing for null.
std::string cell_text(const json& c, size_t max_chars) {
    std::string s;
    if (c.is_null()) return s;
    if (c.is_boolean()) return c.get<bool>() ? "true" : "false";
    if (c.is_number_integer()) return c.dump();
    if (c.is_number()) {
        const double v = c.get<double>();
        char b[48];
        if (std::isfinite(v) && v == std::floor(v) && std::fabs(v) < 1e15) std::snprintf(b, sizeof b, "%.0f", v);
        else std::snprintf(b, sizeof b, "%.6g", v);
        return b;
    }
    if (c.is_string()) s = c.get<std::string>();
    else s = c.dump();
    if (s.size() > max_chars) {
        size_t cut = max_chars;
        while (cut > 0 && ((unsigned char)s[cut] & 0xC0) == 0x80) --cut;   // a whole UTF-8 character
        s = s.substr(0, cut) + kEllipsis;
    }
    for (char& ch : s)
        if ((unsigned char)ch < 0x20) ch = ' ';
    return s;
}

bool numeric_type(const std::string& t) {
    static const char* const kNumeric[] = {"TINYINT", "SMALLINT", "INTEGER", "BIGINT", "HUGEINT", "UTINYINT",
                                           "USMALLINT", "UINTEGER", "UBIGINT", "UHUGEINT", "FLOAT", "DOUBLE",
                                           "REAL", "DECIMAL", "BOOLEAN", "INT", "FLOAT4", "FLOAT8"};
    for (const char* n : kNumeric)
        if (t.rfind(n, 0) == 0) return true;
    return false;
}

PanelState mismatch(const std::string& detail) { return {PanelStateKind::mismatch, "Mismatched data", detail}; }
PanelState empty_state(const std::string& detail) { return {PanelStateKind::empty, "No data", detail}; }

// ---- the primitives ------------------------------------------------------------

// Min/max envelope of y over index order into about 2 * buckets points, as
// the live series decimates (decimate_series in live_views_draw.cpp).
void decimate(SeriesLine& l, size_t buckets) {
    const size_t n = l.y.size();
    if (buckets == 0 || n <= 2 * buckets) return;
    std::vector<double> xs, ys;
    xs.reserve(2 * buckets + 2);
    ys.reserve(2 * buckets + 2);
    const size_t per = (n + buckets - 1) / buckets;
    for (size_t b0 = 0; b0 < n; b0 += per) {
        const size_t b1 = std::min(n, b0 + per);
        size_t imin = b0, imax = b0;
        for (size_t i = b0; i < b1; ++i) {
            if (l.y[i] < l.y[imin]) imin = i;
            if (l.y[i] > l.y[imax]) imax = i;
        }
        const size_t a = std::min(imin, imax), b = std::max(imin, imax);
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

void prepare_series(PreparedPanel& p, const PanelData& d, const PrepareLimits& lim) {
    PreparedSeries& S = p.series;
    std::vector<double> xs;
    const SlotData* xslot = slot(d, "x");
    if (xslot) {
        if (!vector_of(*xslot, &xs)) {
            p.state = mismatch(xslot->binding);
            return;
        }
        S.x_index = false;
        S.x_name = slot_name(*xslot);
    } else {
        S.x_name = "step";
    }
    // Every y binding: one line, or one per row of a rank-2 tensor.
    struct Raw {
        std::string name;
        std::vector<double> y;
    };
    std::vector<Raw> raw;
    for (const SlotData* ys : slots(d, "y")) {
        if (ys->kind == SlotData::Kind::tensor && ys->tensor.valid() && ys->tensor.rank() == 2) {
            const TensorData& t = ys->tensor;
            const int64_t rows = t.shape()[0], cols = t.shape()[1];
            for (int64_t r = 0; r < rows && raw.size() < 64; ++r) {
                Raw line{ys->source + "[" + std::to_string(r) + "]", {}};
                line.y.reserve((size_t)cols);
                for (int64_t c = 0; c < cols; ++c) line.y.push_back(t.element((uint64_t)(r * cols + c)));
                raw.push_back(std::move(line));
            }
            if (rows > 64) p.bounds.push_back("lines cut at 64 rows");
            continue;
        }
        Raw line{slot_name(*ys), {}};
        if (!vector_of(*ys, &line.y)) {
            p.state = mismatch(ys->binding);
            return;
        }
        raw.push_back(std::move(line));
    }
    if (raw.empty()) {
        p.state = empty_state(p.id);
        return;
    }
    size_t steps = 0;
    for (Raw& r : raw) {
        SeriesLine l;
        l.name = r.name;
        const size_t n = S.x_index ? r.y.size() : std::min(r.y.size(), xs.size());
        l.count = n;
        steps = std::max(steps, n);
        l.x.reserve(n);
        l.y.reserve(n);
        for (size_t i = 0; i < n; ++i) {
            const double x = S.x_index ? (double)i : xs[i], y = r.y[i];
            if (!std::isfinite(x) || !std::isfinite(y)) continue;
            l.last = y;
            l.has_last = true;
            if ((p.x_log && x <= 0.0) || (p.y_log && y <= 0.0)) continue;   // no place on a log axis
            l.x.push_back(x);
            l.y.push_back(y);
        }
        if (l.y.size() > lim.series_raw_max) {
            const size_t before = l.y.size();
            decimate(l, lim.series_buckets);
            p.bounds.push_back("series " + l.name + " decimated from " + std::to_string(before) + " to " +
                               std::to_string(l.y.size()) + " points");
        }
        S.lines.push_back(std::move(l));
    }
    bool any = false;
    for (const auto& l : S.lines) any = any || !l.y.empty();
    if (!any) {
        p.state = empty_state(raw.front().name);
        return;
    }
    if (p.x_label.empty()) p.x_label = S.x_name;
    if (p.y_label.empty() && S.lines.size() == 1) p.y_label = S.lines[0].name;
    char b[160];
    if (S.lines.size() == 1 && S.lines[0].has_last)
        std::snprintf(b, sizeof b, "%s steps  last %s", with_commas(steps).c_str(), spec_fmt4(S.lines[0].last).c_str());
    else
        std::snprintf(b, sizeof b, "%s steps  %zu lines", with_commas(steps).c_str(), S.lines.size());
    p.info = b;
}

// Strides that bring a side under the limit.
int stride_for(int64_t n, int max_side) {
    if (max_side <= 0 || n <= max_side) return 1;
    return (int)((n + max_side - 1) / max_side);
}

void prepare_heatmap(PreparedPanel& p, const spec::Panel& panel, const PanelData& d, const PrepareLimits& lim) {
    const SlotData* v = slot(d, "values");
    if (!v) {
        p.state = empty_state(p.id);
        return;
    }
    if (v->kind != SlotData::Kind::tensor || !v->tensor.valid() || v->tensor.rank() != 2) {
        p.state = mismatch(v->binding);
        return;
    }
    const TensorData& t = v->tensor;
    const int64_t R = t.shape()[0], C = t.shape()[1];
    if (R <= 0 || C <= 0) {
        p.state = empty_state(v->binding);
        return;
    }
    PreparedGrid& G = p.grid;
    G.source_rows = (int)R;
    G.source_cols = (int)C;
    G.stride_r = stride_for(R, lim.max_texture_side);
    G.stride_c = stride_for(C, lim.max_texture_side);
    G.rows = (int)((R + G.stride_r - 1) / G.stride_r);
    G.cols = (int)((C + G.stride_c - 1) / G.stride_c);
    if (G.stride_r > 1 || G.stride_c > 1)
        p.bounds.push_back("heatmap sampled from " + std::to_string(R) + "x" + std::to_string(C) + " to " +
                           std::to_string(G.rows) + "x" + std::to_string(G.cols));
    G.values.resize((size_t)G.rows * (size_t)G.cols);
    double lo = INFINITY, hi = -INFINITY;
    for (int r = 0; r < G.rows; ++r)
        for (int c = 0; c < G.cols; ++c) {
            const double x = t.element((uint64_t)((int64_t)r * G.stride_r * C + (int64_t)c * G.stride_c));
            G.values[(size_t)r * G.cols + c] = (float)x;
            if (std::isfinite(x)) {
                lo = std::min(lo, x);
                hi = std::max(hi, x);
            }
        }
    if (panel.options.range) {
        lo = (*panel.options.range)[0];
        hi = (*panel.options.range)[1];
    }
    settle_range(&lo, &hi);
    G.vmin = (float)lo;
    G.vmax = (float)hi;
    G.colormap = spec_colormap(panel.options.colormap.value_or("viridis"), kCmapViridis);
    G.rgba.resize(G.values.size() * 4);
    map_f32_to_rgba8(G.values.data(), G.cols, G.rows, spec_colormap_lut(G.colormap), G.vmin, G.vmax, G.rgba.data());
    char b[160];
    std::snprintf(b, sizeof b, "%lld%s%lld  %s %s %s", (long long)R, kTimes, (long long)C, spec_fmt4(lo).c_str(),
                  kEllipsis, spec_fmt4(hi).c_str());
    p.info = b;
}

void prepare_image(PreparedPanel& p, const spec::Panel& panel, const PanelData& d, const PrepareLimits& lim) {
    const SlotData* v = slot(d, "values");
    if (!v) {
        p.state = empty_state(p.id);
        return;
    }
    const TensorData& t = v->tensor;
    if (v->kind != SlotData::Kind::tensor || !t.valid() || (t.rank() != 2 && t.rank() != 3)) {
        p.state = mismatch(v->binding);
        return;
    }
    const int64_t H = t.shape()[0], W = t.shape()[1], K = t.rank() == 3 ? t.shape()[2] : 1;
    if (K != 1 && K != 3 && K != 4) {
        p.state = mismatch(v->binding);
        return;
    }
    if (H <= 0 || W <= 0) {
        p.state = empty_state(v->binding);
        return;
    }
    PreparedGrid& G = p.grid;
    G.channels = (int)K;
    G.source_rows = (int)H;
    G.source_cols = (int)W;
    G.stride_r = G.stride_c = std::max(stride_for(H, lim.max_texture_side), stride_for(W, lim.max_texture_side));
    G.rows = (int)((H + G.stride_r - 1) / G.stride_r);
    G.cols = (int)((W + G.stride_c - 1) / G.stride_c);
    if (G.stride_r > 1)
        p.bounds.push_back("image sampled from " + std::to_string(H) + "x" + std::to_string(W) + " to " +
                           std::to_string(G.rows) + "x" + std::to_string(G.cols));
    const bool is_float = caliper_builder::dtype_is_float(t.dtype());
    double lo = 0.0, hi = is_float ? 1.0 : 255.0;
    if (panel.options.range) {
        lo = (*panel.options.range)[0];
        hi = (*panel.options.range)[1];
    }
    G.vmin = (float)lo;
    G.vmax = (float)hi;
    G.rgba.assign((size_t)G.rows * G.cols * 4, 255);
    auto at = [&](int r, int c, int k) {
        return t.element((uint64_t)((((int64_t)r * G.stride_r) * W + (int64_t)c * G.stride_c) * K + k));
    };
    if (K == 1) {
        G.colormap = spec_colormap(panel.options.colormap.value_or("greys"), kCmapGreys);
        G.values.resize((size_t)G.rows * G.cols);
        for (int r = 0; r < G.rows; ++r)
            for (int c = 0; c < G.cols; ++c) G.values[(size_t)r * G.cols + c] = (float)at(r, c, 0);
        map_f32_to_rgba8(G.values.data(), G.cols, G.rows, spec_colormap_lut(G.colormap), G.vmin, G.vmax,
                         G.rgba.data());
    } else {
        const double span = hi - lo;
        for (int r = 0; r < G.rows; ++r)
            for (int c = 0; c < G.cols; ++c)
                for (int k = 0; k < K; ++k) {
                    const double x = at(r, c, k);
                    double u = span != 0.0 && std::isfinite(x) ? (x - lo) / span : 0.0;
                    u = std::clamp(u, 0.0, 1.0);
                    G.rgba[((size_t)r * G.cols + c) * 4 + k] = (uint8_t)(u * 255.0 + 0.5);
                }
    }
    char b[96];
    std::snprintf(b, sizeof b, "%lld%s%lld %s", (long long)H, kTimes, (long long)W,
                  K == 1 ? "grey" : K == 3 ? "RGB" : "RGBA");
    p.info = b;
}

void prepare_points(PreparedPanel& p, const spec::Panel& panel, const PanelData& d, const PrepareLimits& lim) {
    PreparedPoints& P = p.points;
    std::vector<double> xs, ys, zs;
    const SlotData* pos = slot(d, "positions");
    if (pos) {
        const TensorData& t = pos->tensor;
        if (pos->kind != SlotData::Kind::tensor || !t.valid() || t.rank() != 2 ||
            (t.shape()[1] != 2 && t.shape()[1] != 3)) {
            p.state = mismatch(pos->binding);
            return;
        }
        const int64_t n = t.shape()[0], k = t.shape()[1];
        xs.resize((size_t)n);
        ys.resize((size_t)n);
        if (k == 3) zs.resize((size_t)n);
        for (int64_t i = 0; i < n; ++i) {
            xs[(size_t)i] = t.element((uint64_t)(i * k));
            ys[(size_t)i] = t.element((uint64_t)(i * k + 1));
            if (k == 3) zs[(size_t)i] = t.element((uint64_t)(i * k + 2));
        }
        if (p.x_label.empty()) p.x_label = "x";
        if (p.y_label.empty()) p.y_label = "y";
        if (k == 3 && p.z_label.empty()) p.z_label = "z";
    } else {
        const SlotData* sx = slot(d, "x");
        const SlotData* sy = slot(d, "y");
        const SlotData* sz = slot(d, "z");
        if (!sx || !sy) {
            p.state = empty_state(p.id);
            return;
        }
        if (!vector_of(*sx, &xs)) { p.state = mismatch(sx->binding); return; }
        if (!vector_of(*sy, &ys)) { p.state = mismatch(sy->binding); return; }
        if (sz && !vector_of(*sz, &zs)) { p.state = mismatch(sz->binding); return; }
        if (p.x_label.empty()) p.x_label = slot_name(*sx);
        if (p.y_label.empty()) p.y_label = slot_name(*sy);
        if (sz && p.z_label.empty()) p.z_label = slot_name(*sz);
    }
    P.three_d = !zs.empty() || (pos && pos->tensor.shape()[1] == 3);
    std::vector<double> cs, ss;
    if (const SlotData* sc = slot(d, "color")) {
        if (!vector_of(*sc, &cs)) { p.state = mismatch(sc->binding); return; }
        P.color_name = slot_name(*sc);
    }
    if (const SlotData* sz = slot(d, "size")) {
        if (!vector_of(*sz, &ss)) { p.state = mismatch(sz->binding); return; }
        P.size_name = slot_name(*sz);
    }
    size_t n = std::min(xs.size(), ys.size());
    if (P.three_d) n = std::min(n, zs.size());
    if (!cs.empty()) n = std::min(n, cs.size());
    if (!ss.empty()) n = std::min(n, ss.size());
    P.total = n;
    if (n == 0) {
        p.state = empty_state(p.id);
        return;
    }
    P.stride = lim.max_points && n > lim.max_points ? (n + lim.max_points - 1) / lim.max_points : 1;
    if (P.stride > 1)
        p.bounds.push_back("points sampled from " + std::to_string(n) + " at a stride of " + std::to_string(P.stride));
    for (size_t i = 0; i < n; i += P.stride) {
        if (!std::isfinite(xs[i]) || !std::isfinite(ys[i]) || (P.three_d && !std::isfinite(zs[i]))) continue;
        P.x.push_back((float)xs[i]);
        P.y.push_back((float)ys[i]);
        if (P.three_d) P.z.push_back((float)zs[i]);
        if (!cs.empty()) P.color.push_back((float)cs[i]);
        if (!ss.empty()) P.size.push_back((float)ss[i]);
    }
    if (P.x.empty()) {
        p.state = empty_state(p.id);
        return;
    }
    double lo, hi;
    if (!cs.empty()) {
        finite_range(std::vector<double>(P.color.begin(), P.color.end()), &lo, &hi);
        settle_range(&lo, &hi);
        P.cmin = (float)lo;
        P.cmax = (float)hi;
    }
    if (!ss.empty()) {
        finite_range(std::vector<double>(P.size.begin(), P.size.end()), &lo, &hi);
        settle_range(&lo, &hi);
        P.smin = (float)lo;
        P.smax = (float)hi;
    }
    P.colormap = spec_colormap(panel.options.colormap.value_or("viridis"), kCmapViridis);
    if (!P.color.empty()) {
        const uint32_t* lut = spec_colormap_lut(P.colormap);
        const float span = P.cmax - P.cmin;
        P.rgba.reserve(P.color.size());
        for (float c : P.color) {
            const float t = std::isfinite(c) && span != 0.0f ? std::clamp((c - P.cmin) / span, 0.0f, 1.0f) : 0.0f;
            P.rgba.push_back(lut[(int)(t * 255.0f + 0.5f)]);
        }
    }
    if (!P.size.empty()) {
        const float span = P.smax - P.smin;
        P.scale.reserve(P.size.size());
        for (float v : P.size) {
            const float t = std::isfinite(v) && span != 0.0f ? std::clamp((v - P.smin) / span, 0.0f, 1.0f) : 0.0f;
            P.scale.push_back(0.5f + 1.5f * t);
        }
    }
    char b[96];
    std::snprintf(b, sizeof b, "%s points  %d-D", with_commas(n).c_str(), P.three_d ? 3 : 2);
    p.info = b;
}

void prepare_histogram(PreparedPanel& p, const spec::Panel& panel, const PanelData& d) {
    const SlotData* v = slot(d, "values");
    if (!v) {
        p.state = empty_state(p.id);
        return;
    }
    std::vector<double> xs;
    if (v->kind == SlotData::Kind::column) {
        vector_of(*v, &xs);
    } else if (v->kind == SlotData::Kind::tensor && v->tensor.valid()) {
        xs.reserve((size_t)v->tensor.numel());
        for (uint64_t i = 0; i < v->tensor.numel(); ++i) xs.push_back(v->tensor.element(i));
    } else {
        p.state = mismatch(v->binding);
        return;
    }
    double lo, hi;
    finite_range(xs, &lo, &hi);
    if (!(hi >= lo)) {
        p.state = empty_state(v->binding);
        return;
    }
    if (panel.options.range) {
        lo = (*panel.options.range)[0];
        hi = (*panel.options.range)[1];
    }
    settle_range(&lo, &hi);
    PreparedHistogram& H = p.histogram;
    const int64_t bins = std::clamp<int64_t>(panel.options.bins.value_or(64), 1, spec::limits::kMaxBins);
    H.counts.assign((size_t)bins, 0.0);
    H.lo = lo;
    H.hi = hi;
    const double w = (hi - lo) / (double)bins;
    for (double x : xs) {
        if (!std::isfinite(x)) continue;
        if (x < lo || x > hi) {
            ++H.outside;
            continue;
        }
        const int64_t b = std::min<int64_t>(bins - 1, (int64_t)std::floor((x - lo) / w));   // hi joins the last bin
        H.counts[(size_t)b] += 1.0;
        H.total += 1.0;
    }
    if (H.outside) p.bounds.push_back(std::to_string(H.outside) + " values outside the range");
    if (p.x_label.empty()) p.x_label = slot_name(*v);
    if (p.y_label.empty()) p.y_label = "count";
    char b[160];
    std::snprintf(b, sizeof b, "%lld bins  %s %s %s  n %s", (long long)bins, spec_fmt4(lo).c_str(), kEllipsis,
                  spec_fmt4(hi).c_str(), spec_fmt4(H.total).c_str());
    p.info = b;
}

void prepare_table(PreparedPanel& p, const PanelData& d, const PrepareLimits& lim) {
    const SlotData* v = slot(d, "rows");
    if (!v) {
        p.state = empty_state(p.id);
        return;
    }
    if (v->kind != SlotData::Kind::table || !v->values.is_array()) {
        p.state = mismatch(v->binding);
        return;
    }
    PreparedTable& T = p.table;
    for (const auto& c : v->columns) {
        T.columns.push_back(c.name);
        T.numeric.push_back(numeric_type(c.type));
    }
    T.total_rows = v->values.size();
    const size_t shown = std::min(T.total_rows, lim.max_table_rows);
    T.truncated = v->truncated || shown < T.total_rows;
    if (shown < T.total_rows) p.bounds.push_back("table cut from " + std::to_string(T.total_rows) + " to " +
                                                  std::to_string(shown) + " rows");
    T.cells.reserve(shown);
    for (size_t r = 0; r < shown; ++r) {
        const json& row = v->values[r];
        std::vector<std::string> cells;
        cells.reserve(T.columns.size());
        for (size_t c = 0; c < T.columns.size(); ++c)
            cells.push_back(row.is_array() && c < row.size() ? cell_text(row[c], lim.max_cell_chars) : std::string());
        T.cells.push_back(std::move(cells));
    }
    if (T.columns.empty()) {
        p.state = empty_state(v->binding);
        return;
    }
    if (shown < T.total_rows)
        p.info = with_commas(shown) + " of " + with_commas(T.total_rows) + (v->truncated ? "+" : "") + " rows";
    else
        p.info = with_commas(T.total_rows) + (v->truncated ? "+" : "") + " rows";
}

void prepare_text(PreparedPanel& p, const spec::Panel& panel, const PanelData& d, const PrepareLimits& lim) {
    PreparedText& X = p.text;
    X.text = panel.options.text.value_or("");
    if (const SlotData* v = slot(d, "value")) {
        if (v->kind == SlotData::Kind::column && v->values.is_array()) {
            for (size_t i = v->values.size(); i-- > 0;) {
                const json& c = v->values[i];
                if (c.is_null()) continue;
                const double x = cell_number(c);
                X.value = c.is_number() || c.is_boolean() ? spec_fmt4(x) : cell_text(c, lim.max_cell_chars);
                X.has_value = true;
                break;
            }
            if (!X.has_value && X.text.empty()) {
                p.state = empty_state(v->binding);
                return;
            }
        } else if (v->kind == SlotData::Kind::tensor && v->tensor.valid() && v->tensor.numel() == 1) {
            X.value = spec_fmt4(v->tensor.element(0));
            X.has_value = true;
        } else {
            p.state = mismatch(v->binding);
            return;
        }
    }
    if (X.text.empty() && !X.has_value) p.state = empty_state(p.id);
}

}  // namespace

std::string spec_fmt4(double v) {
    if (!std::isfinite(v)) return std::isnan(v) ? "NaN" : v > 0 ? "inf" : "-inf";
    if (v == 0.0) return "0";
    char b[48];
    const double a = std::fabs(v);
    if (a >= 1e5 || a < 1e-3) {
        std::snprintf(b, sizeof b, "%.3e", v);
        std::string t = b;
        const size_t e = t.find('e');
        std::string mant = t.substr(0, e), ex = t.substr(e + 1);
        const bool neg = !ex.empty() && ex[0] == '-';
        if (!ex.empty() && (ex[0] == '-' || ex[0] == '+')) ex.erase(0, 1);
        while (ex.size() > 1 && ex[0] == '0') ex.erase(0, 1);
        if (mant.find('.') != std::string::npos) {
            while (mant.back() == '0') mant.pop_back();
            if (mant.back() == '.') mant.pop_back();
        }
        return mant + "e" + (neg ? "-" : "") + ex;
    }
    std::snprintf(b, sizeof b, "%.4g", v);
    std::snprintf(b, sizeof b, "%.10g", std::strtod(b, nullptr));
    return b;
}

int spec_colormap(const std::string& name, int fallback) {
    if (name == "viridis") return kCmapViridis;
    if (name == "magma") return kCmapMagma;
    if (name == "rdbu") return kCmapRdbu;
    if (name == "greys") return kCmapGreys;
    return fallback;
}

const uint32_t* spec_colormap_lut(int colormap) {
    static const uint32_t* greys = [] {
        static uint32_t t[256];
        for (uint32_t i = 0; i < 256; ++i) t[i] = 0xff000000u | (i << 16) | (i << 8) | i;
        return t;
    }();
    switch (colormap) {
        case kCmapMagma: return colormap_lut(CALIPER_CMAP_MAGMA);
        case kCmapRdbu: return colormap_lut(CALIPER_CMAP_RDBU);
        case kCmapGreys: return greys;
        default: return colormap_lut(CALIPER_CMAP_VIRIDIS);
    }
}

std::string spec_source_origin(const spec::Spec& s, const std::string& source_id) {
    std::string id = source_id;
    for (int hops = 0; hops < 64; ++hops) {
        const spec::Source* src = nullptr;
        for (const auto& x : s.sources)
            if (x.id == id) src = &x;
        if (!src) return source_id;
        if (src->catalog) return *src->catalog;
        if (src->sql) return "sql";
        if (!src->from) return source_id;
        id = *src->from;
    }
    return source_id;
}

PanelState panel_state_for_error(const std::string& code, const std::string& detail) {
    const std::string d = detail.empty() ? code : detail;
    if (code == "resolve.unavailable") return {PanelStateKind::offline, "Source offline", d};
    if (code == "resolve.too_large") return {PanelStateKind::too_large, "Data too large", d};
    if (code == "resolve.query_failed" || code == "resolve.timeout") return {PanelStateKind::query_error, "Query error", d};
    if (code == "resolve.unknown_column") return {PanelStateKind::missing, "Missing column", d};
    if (code == "resolve.unknown_source" || code == "resolve.cycle") return {PanelStateKind::missing, "Missing source", d};
    if (code == "panel.not_found") return {PanelStateKind::no_spec, "Missing panel", d};
    if (code == "spec.no_open_spec") return {PanelStateKind::no_spec, "No spec", d};
    if (code == "resolve.not_tabular" || code == "resolve.not_a_tensor") return mismatch(d);
    if (code.rfind("transform.", 0) == 0) return {PanelStateKind::mismatch, "Transform error", d};
    if (code.rfind("loader.", 0) == 0) return {PanelStateKind::missing, "Missing data", d};
    return {PanelStateKind::error, "Data error", d};
}

PreparedPanel prepare_panel(const spec::Spec& s, const spec::Panel& panel, const PanelData& d,
                            const PrepareLimits& lim) {
    PreparedPanel p;
    p.id = panel.id;
    p.title = panel.title && !panel.title->empty() ? *panel.title : panel.id;
    p.primitive = panel.primitive;
    const spec::Options& o = panel.options;
    p.x_label = o.x_label.value_or("");
    p.y_label = o.y_label.value_or("");
    p.z_label = o.z_label.value_or("");
    p.x_log = o.x_log.value_or(false);
    p.y_log = o.y_log.value_or(false);
    if (o.x_range) {
        p.has_x_range = true;
        p.x_range[0] = (*o.x_range)[0];
        p.x_range[1] = (*o.x_range)[1];
    }
    if (o.y_range) {
        p.has_y_range = true;
        p.y_range[0] = (*o.y_range)[0];
        p.y_range[1] = (*o.y_range)[1];
    }
    if (d.error) {
        const std::string detail = !d.error.where.empty() ? spec_source_origin(s, d.error.where) : d.error.code;
        p.state = panel_state_for_error(d.error.code, detail);
        return p;
    }
    switch (panel.primitive) {
        case spec::Primitive::series: prepare_series(p, d, lim); break;
        case spec::Primitive::heatmap: prepare_heatmap(p, panel, d, lim); break;
        case spec::Primitive::image: prepare_image(p, panel, d, lim); break;
        case spec::Primitive::points: prepare_points(p, panel, d, lim); break;
        case spec::Primitive::histogram: prepare_histogram(p, panel, d); break;
        case spec::Primitive::table: prepare_table(p, d, lim); break;
        case spec::Primitive::text: prepare_text(p, panel, d, lim); break;
    }
    // A state names the source's catalog id rather than the spec's own id.
    if (!p.state.ready() && !p.state.detail.empty()) {
        const std::string& det = p.state.detail;
        const size_t dot = det.find('.');
        const std::string src = det.substr(0, dot);
        for (const auto& x : s.sources)
            if (x.id == src && dot == std::string::npos) p.state.detail = spec_source_origin(s, src);
    }
    return p;
}

}  // namespace caliper_host
