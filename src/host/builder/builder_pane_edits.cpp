// builder_pane_edits.cpp: the pure parts of the builder pane
// (builder_pane_edits.h).
#include "builder/builder_pane_edits.h"

#include <caliper_builder/session/spec_edit.h>

#include <algorithm>
#include <cctype>
#include <cmath>
#include <functional>
#include <set>

namespace caliper_host {

namespace spec = caliper_builder::spec;
using spec::LayoutNode;
using spec::Primitive;
using spec::SlotValue;
using nlohmann::json;

// ---- shapes -----------------------------------------------------------------------------

int64_t PaneShape::elements() const {
    if (tabular || !known) return -1;
    int64_t n = 1;
    for (int64_t d : dims) {
        if (d < 0) return -1;
        n *= d;
    }
    return n;
}

bool pane_type_numeric(const std::string& raw) {
    std::string t;
    for (char c : raw) t += (char)std::toupper((unsigned char)c);
    while (!t.empty() && t.back() == ' ') t.pop_back();
    while (!t.empty() && t.front() == ' ') t.erase(t.begin());
    if (t.empty() || t.back() == ']') return false;
    std::string base = t.substr(0, t.find('('));
    while (!base.empty() && base.back() == ' ') base.pop_back();
    static const std::set<std::string> numeric = {
        "BOOLEAN", "BOOL",      "LOGICAL",  "TINYINT", "INT1",    "SMALLINT", "INT2",   "SHORT",   "INTEGER",
        "INT",     "INT4",      "SIGNED",   "BIGINT",  "INT8",    "LONG",     "HUGEINT", "INT128", "UTINYINT",
        "USMALLINT", "UINTEGER", "UBIGINT", "UHUGEINT", "FLOAT",  "FLOAT4",   "REAL",   "DOUBLE",  "FLOAT8",
        "DECIMAL", "NUMERIC",   "VARINT",   "BIGNUM"};
    return numeric.count(base) != 0;
}

std::string pane_count_text(int64_t n) {
    std::string digits = std::to_string(n < 0 ? -n : n);
    std::string out;
    for (size_t i = 0; i < digits.size(); ++i) {
        if (i && (digits.size() - i) % 3 == 0) out += ',';
        out += digits[i];
    }
    return n < 0 ? "-" + out : out;
}

std::string pane_dims_text(const std::vector<int64_t>& dims) {
    if (dims.empty()) return "scalar";
    std::string s;
    for (size_t i = 0; i < dims.size(); ++i) {
        if (i) s += "\xc3\x97";   // ×
        s += dims[i] < 0 ? std::string("?") : std::to_string(dims[i]);
    }
    return s;
}

std::string pane_shape_text(const PaneShape& shape) {
    if (!shape.known) return {};
    if (shape.tabular) {
        const size_t c = shape.columns.size();
        if (shape.rows) return pane_count_text(*shape.rows) + " \xc3\x97 " + std::to_string(c);
        return std::to_string(c) + (c == 1 ? " column" : " columns");
    }
    std::string s = shape.dtype.empty() ? std::string() : shape.dtype + " ";
    return s + pane_dims_text(shape.dims);
}

PaneShape pane_shape_from(const spec::ResolvedSource& r) {
    PaneShape s;
    s.known = true;
    s.tabular = r.columns.has_value();
    if (r.columns)
        for (const auto& c : *r.columns) s.columns.push_back({c.name, c.type, pane_type_numeric(c.type)});
    s.dtype = r.dtype;
    if (r.shape) s.dims = *r.shape;
    if (!s.tabular && !r.shape) s.known = false;
    return s;
}

PaneShape pane_shape_from(const spec::SourceInfo& info) {
    PaneShape s;
    if (info.kind == spec::DataKind::tabular) {
        s.tabular = true;
        s.known = info.columns.has_value();
        if (info.columns)
            for (const auto& c : *info.columns) s.columns.push_back({c.name, c.type, pane_type_numeric(c.type)});
    } else if (info.kind == spec::DataKind::tensor) {
        s.known = info.shape.has_value();
        s.dtype = info.dtype;
        if (info.shape) s.dims = *info.shape;
    }
    return s;
}

std::map<std::string, PaneShape> pane_source_shapes(const spec::Spec& s, const spec::SourceResolver& resolver) {
    std::map<std::string, PaneShape> out;
    const spec::ValidationReport report = spec::validate(s, resolver);
    for (const auto& info : report.sources) out[info.id] = pane_shape_from(info);
    return out;
}

std::vector<PaneShape> pane_transform_shapes(const spec::Spec& s, const std::string& source_id,
                                             const spec::SourceResolver& resolver) {
    std::vector<PaneShape> out;
    size_t index = s.sources.size();
    for (size_t i = 0; i < s.sources.size(); ++i)
        if (s.sources[i].id == source_id) index = i;
    if (index == s.sources.size()) return out;
    const size_t n = s.sources[index].transforms.size();
    for (size_t k = 0; k <= n; ++k) {
        spec::Spec cut = s;
        cut.sources[index].transforms.resize(k);
        const spec::ValidationReport report = spec::validate(cut, resolver);
        PaneShape shape;
        if (index < report.sources.size()) shape = pane_shape_from(report.sources[index]);
        // A step that the validator refuses leaves the shape unknown, so the
        // inspector shows nothing after it.
        for (const auto& d : report.diagnostics) {
            const std::string prefix = "source[" + std::to_string(index) + "].transform";
            if (d.is_error() && d.path.compare(0, prefix.size(), prefix) == 0) shape.known = false;
        }
        out.push_back(std::move(shape));
    }
    return out;
}

// ---- transform axes ---------------------------------------------------------------------

void pane_name_dims(std::vector<PaneShape>& shapes, const std::vector<spec::Transform>& transforms,
                    const std::vector<std::string>& names) {
    auto tensor = [](const PaneShape& sh) { return sh.known && !sh.tabular; };
    if (shapes.empty() || !tensor(shapes[0]) || names.empty() || shapes[0].dims.size() != names.size()) return;
    std::vector<std::string> now = names;
    shapes[0].names = now;
    for (size_t k = 0; k < transforms.size() && k + 1 < shapes.size(); ++k) {
        const spec::Transform& t = transforms[k];
        const auto rank = (int64_t)now.size();
        auto axis_at = [&](int64_t a) { return a < 0 ? a + rank : a; };
        switch (t.op) {
            case spec::TransformOp::slice: {
                if (t.axis.empty()) return;
                const int64_t a = axis_at(t.axis.front());
                if (a < 0 || a >= rank) return;
                if (t.index) now.erase(now.begin() + a);
                break;
            }
            case spec::TransformOp::reduce: {
                if (t.axis.empty()) return;
                std::vector<bool> gone(now.size(), false);
                for (const int64_t x : t.axis) {
                    const int64_t a = axis_at(x);
                    if (a < 0 || a >= rank) return;
                    gone[(size_t)a] = true;
                }
                if (t.keepdims.value_or(false)) break;
                std::vector<std::string> kept;
                for (size_t d = 0; d < now.size(); ++d)
                    if (!gone[d]) kept.push_back(now[d]);
                now = std::move(kept);
                break;
            }
            case spec::TransformOp::normalise: break;
        }
        PaneShape& next = shapes[k + 1];
        if (!tensor(next) || next.dims.size() != now.size()) return;
        next.names = now;
    }
}

std::string pane_index_range(int64_t size) {
    if (size <= 0) return {};
    if (size == 1) return "0";
    return "0\xe2\x80\x93" + std::to_string(size - 1);   // 0–3
}

std::string pane_axis_text(int64_t axis, int64_t size, const std::string& name) {
    std::string out = "axis " + std::to_string(axis);
    if (!name.empty()) out += " " + name;
    const std::string range = pane_index_range(size);
    if (!range.empty()) out += " \xc2\xb7 " + range;   // axis 1 · 0–3
    return out;
}

int64_t pane_clamp_index(int64_t index, int64_t size, bool* clamped) {
    if (clamped) *clamped = false;
    if (size <= 0) return index;
    if (index >= size || index < -size) {
        if (clamped) *clamped = true;
        return index >= size ? size - 1 : 0;
    }
    return index;
}

// ---- primitives, slots and options ------------------------------------------------------

const std::vector<Primitive>& pane_primitives() {
    static const std::vector<Primitive> all = {Primitive::series, Primitive::heatmap, Primitive::points,
                                               Primitive::histogram, Primitive::table, Primitive::image,
                                               Primitive::text};
    return all;
}

const char* pane_primitive_noun(Primitive p) {
    switch (p) {
        case Primitive::series: return "Series";
        case Primitive::heatmap: return "Heatmap";
        case Primitive::points: return "Points";
        case Primitive::histogram: return "Histogram";
        case Primitive::table: return "Table";
        case Primitive::image: return "Image";
        case Primitive::text: return "Text";
        case Primitive::scene: return "Scene";
    }
    return "";
}

const std::vector<PaneSlot>& pane_slots(Primitive p) {
    static const std::vector<PaneSlot> series = {{"y", true, true}, {"x", false, false}};
    static const std::vector<PaneSlot> values = {{"values", true, false}};
    static const std::vector<PaneSlot> points = {{"positions", false, false}, {"x", false, false},
                                                 {"y", false, false},         {"z", false, false},
                                                 {"color", false, false},     {"size", false, false}};
    static const std::vector<PaneSlot> table = {{"rows", true, false}};
    static const std::vector<PaneSlot> text = {{"value", false, false}};
    static const std::vector<PaneSlot> scene = {{"scene", true, false}};
    switch (p) {
        case Primitive::series: return series;
        case Primitive::heatmap:
        case Primitive::histogram:
        case Primitive::image: return values;
        case Primitive::points: return points;
        case Primitive::table: return table;
        case Primitive::text: return text;
        case Primitive::scene: return scene;   // a scene panel (spec version 1): its live scene
    }
    return text;
}

const std::vector<PaneOption>& pane_options(Primitive p) {
    using K = PaneOptionKind;
    static const std::vector<PaneOption> series = {{"x_label", K::text, "X label"}, {"y_label", K::text, "Y label"},
                                                   {"x_range", K::range, "X range"}, {"y_range", K::range, "Y range"},
                                                   {"x_log", K::flag, "X log"},     {"y_log", K::flag, "Y log"}};
    static const std::vector<PaneOption> heatmap = {{"x_label", K::text, "X label"}, {"y_label", K::text, "Y label"},
                                                    {"colormap", K::colormap, "Colormap"}, {"range", K::range, "Range"}};
    static const std::vector<PaneOption> points = {{"x_label", K::text, "X label"}, {"y_label", K::text, "Y label"},
                                                   {"z_label", K::text, "Z label"}, {"colormap", K::colormap, "Colormap"}};
    static const std::vector<PaneOption> histogram = {{"x_label", K::text, "X label"}, {"y_label", K::text, "Y label"},
                                                      {"bins", K::bins, "Bins"},       {"range", K::range, "Range"},
                                                      {"y_log", K::flag, "Y log"}};
    static const std::vector<PaneOption> table = {{"columns", K::columns, "Columns"}};
    static const std::vector<PaneOption> image = {{"colormap", K::colormap, "Colormap"}, {"range", K::range, "Range"}};
    static const std::vector<PaneOption> text = {{"text", K::longtext, "Text"}};
    static const std::vector<PaneOption> scene = {};   // a scene panel's options are set through panel.patch
    switch (p) {
        case Primitive::series: return series;
        case Primitive::heatmap: return heatmap;
        case Primitive::points: return points;
        case Primitive::histogram: return histogram;
        case Primitive::table: return table;
        case Primitive::image: return image;
        case Primitive::text: return text;
        case Primitive::scene: return scene;
    }
    return text;
}

const std::vector<std::string>& pane_colormaps() {
    static const std::vector<std::string> all = {"viridis", "magma", "rdbu", "greys"};
    return all;
}

namespace {

SlotValue one(const std::string& b) {
    SlotValue v;
    v.bindings = {b};
    return v;
}

std::string col_binding(const std::string& source, const std::string& column) { return source + "." + column; }

const PaneColumn* find_column(const PaneShape& s, const std::string& name) {
    for (const auto& c : s.columns)
        if (c.name == name) return &c;
    return nullptr;
}

std::vector<const PaneColumn*> numeric_columns(const PaneShape& s) {
    std::vector<const PaneColumn*> out;
    for (const auto& c : s.columns)
        if (c.numeric) out.push_back(&c);
    return out;
}

}  // namespace

std::optional<std::map<std::string, SlotValue>> pane_auto_bind(Primitive p, const std::string& source,
                                                               const PaneShape& shape, const std::string& column) {
    std::map<std::string, SlotValue> b;
    if (!shape.known) {
        // Nothing known of the data: the likeliest slot, for the validator to judge.
        switch (p) {
            case Primitive::series: b["y"] = one(column.empty() ? source : col_binding(source, column)); return b;
            case Primitive::table: b["rows"] = one(source); return b;
            case Primitive::text: b["value"] = one(column.empty() ? source : col_binding(source, column)); return b;
            default: b["values"] = one(column.empty() ? source : col_binding(source, column)); return b;
        }
    }
    if (shape.tabular) {
        const PaneColumn* chosen = column.empty() ? nullptr : find_column(shape, column);
        if (!column.empty() && !chosen) return std::nullopt;
        const auto nums = numeric_columns(shape);
        switch (p) {
            case Primitive::series: {
                if (chosen) {
                    if (!chosen->numeric) return std::nullopt;
                    b["y"] = one(col_binding(source, chosen->name));
                    if (!shape.columns.empty() && shape.columns[0].name != chosen->name && shape.columns[0].numeric)
                        b["x"] = one(col_binding(source, shape.columns[0].name));
                    return b;
                }
                if (shape.columns.size() < 2) {
                    if (nums.empty()) return std::nullopt;
                    b["y"] = one(col_binding(source, nums[0]->name));
                    return b;
                }
                SlotValue y;
                for (size_t i = 1; i < shape.columns.size() && y.bindings.size() < 16; ++i)
                    if (shape.columns[i].numeric) y.bindings.push_back(col_binding(source, shape.columns[i].name));
                if (y.bindings.empty()) return std::nullopt;
                y.list = y.bindings.size() > 1;
                b["x"] = one(col_binding(source, shape.columns[0].name));
                b["y"] = std::move(y);
                return b;
            }
            case Primitive::points: {
                if (nums.size() < 2) return std::nullopt;
                b["x"] = one(col_binding(source, nums[0]->name));
                b["y"] = one(col_binding(source, nums[1]->name));
                return b;
            }
            case Primitive::histogram: {
                const PaneColumn* c = chosen ? chosen : nums.empty() ? nullptr : nums[0];
                if (!c || !c->numeric) return std::nullopt;
                b["values"] = one(col_binding(source, c->name));
                return b;
            }
            case Primitive::table:
                b["rows"] = one(source);
                return b;
            case Primitive::text: {
                const PaneColumn* c = chosen ? chosen : shape.columns.empty() ? nullptr : &shape.columns[0];
                if (!c) return std::nullopt;
                b["value"] = one(col_binding(source, c->name));
                return b;
            }
            case Primitive::heatmap:
            case Primitive::image:
            case Primitive::scene:   // a scene binds a live scene, not a source
                return std::nullopt;
        }
        return std::nullopt;
    }
    // A tensor.
    if (!column.empty()) return std::nullopt;
    const size_t rank = shape.dims.size();
    const int64_t last = rank ? shape.dims.back() : 0;
    switch (p) {
        case Primitive::series:
            if (rank != 1 && rank != 2) return std::nullopt;
            b["y"] = one(source);
            return b;
        case Primitive::heatmap:
            if (rank != 2) return std::nullopt;
            b["values"] = one(source);
            return b;
        case Primitive::image:
            if (!(rank == 2 || (rank == 3 && (last == 3 || last == 4)))) return std::nullopt;
            b["values"] = one(source);
            return b;
        case Primitive::points:
            if (!(rank == 2 && (last == 2 || last == 3))) return std::nullopt;
            b["positions"] = one(source);
            return b;
        case Primitive::histogram:
            b["values"] = one(source);
            return b;
        case Primitive::text:
            if (shape.elements() != 1) return std::nullopt;
            b["value"] = one(source);
            return b;
        case Primitive::table:
        case Primitive::scene:
            return std::nullopt;
    }
    return std::nullopt;
}

Primitive pane_default_primitive(const PaneShape& shape, const std::string& column) {
    if (!shape.known) return column.empty() ? Primitive::table : Primitive::series;
    if (shape.tabular) {
        if (!column.empty()) return Primitive::series;
        for (size_t i = 1; i < shape.columns.size(); ++i)
            if (shape.columns[i].numeric) return Primitive::series;
        return Primitive::table;
    }
    const size_t rank = shape.dims.size();
    const int64_t first = rank ? shape.dims.front() : 0, last = rank ? shape.dims.back() : 0;
    if (rank == 1) return Primitive::series;
    if (rank == 2 && (last == 2 || last == 3) && first >= 16) return Primitive::points;
    if (rank == 2) return Primitive::heatmap;
    if (rank == 3 && (last == 3 || last == 4)) return Primitive::image;
    if (shape.elements() == 1) return Primitive::text;
    return Primitive::histogram;
}

json pane_bind_json(const std::map<std::string, SlotValue>& bind) {
    json out = json::object();
    for (const auto& [slot, v] : bind) {
        if (v.list || v.bindings.size() != 1) out[slot] = v.bindings;
        else out[slot] = v.bindings.front();
    }
    return out;
}

std::string pane_fit_hint(Primitive p, const std::string& source_id, const PaneShape& shape) {
    const char* takes = "";
    size_t max_rank = 0;   // the most dimensions it draws, for a hint to slice
    switch (p) {
        case Primitive::series: takes = "a table's columns or a tensor of one or two dimensions"; max_rank = 2; break;
        case Primitive::heatmap: takes = "a tensor of two dimensions"; max_rank = 2; break;
        case Primitive::image: takes = "a tensor [H, W], [H, W, 3] or [H, W, 4]"; max_rank = 3; break;
        case Primitive::points: takes = "a tensor [n, 2] or [n, 3], or two numeric columns"; max_rank = 2; break;
        case Primitive::histogram: takes = "a numeric column or a tensor"; break;
        case Primitive::table: takes = "a table"; break;
        case Primitive::text:
        case Primitive::scene: return {};
    }
    const std::string what = std::string("a") + (p == Primitive::image ? "n " : " ") + spec::to_string(p);
    if (!shape.known) return source_id + "'s data is not known yet, and " + what + " takes " + takes + ".";
    if (shape.tabular) return source_id + " is a table, and " + what + " takes " + takes + ".";
    std::string out = source_id + " is " + pane_shape_text(shape) + ", and " + what + " takes " + takes;
    if (max_rank > 0 && shape.dims.size() > max_rank) out += ": slice or reduce it in Properties";
    return out + ".";
}

// ---- ids --------------------------------------------------------------------------------

bool pane_valid_id(const std::string& id) {
    if (id.empty() || id.size() > 64) return false;
    if (id[0] < 'a' || id[0] > 'z') return false;
    for (char c : id)
        if (!((c >= 'a' && c <= 'z') || (c >= '0' && c <= '9') || c == '_' || c == '-')) return false;
    return true;
}

namespace {

std::string unique_id(std::string base, const std::function<bool(const std::string&)>& taken) {
    if (!taken(base)) return base;
    for (int n = 2; n < 10000; ++n) {
        std::string suffix = "-" + std::to_string(n);
        std::string id = base.substr(0, 64 - suffix.size()) + suffix;
        if (!taken(id)) return id;
    }
    return base;
}

void rename_leaves(LayoutNode& n, const std::string& from, const std::string& to, int depth = 0) {
    if (depth > 64) return;
    if (n.kind == LayoutNode::Kind::panel) {
        if (n.panel == from) n.panel = to;
        return;
    }
    for (auto& c : n.children) rename_leaves(c, from, to, depth + 1);
}

}  // namespace

std::string pane_source_id(const std::string& name, const spec::Spec& s) {
    std::string id;
    for (char c : name) {
        const char l = (char)std::tolower((unsigned char)c);
        id += (l >= 'a' && l <= 'z') || (l >= '0' && l <= '9') || l == '_' || l == '-' ? l : '_';
    }
    while (!id.empty() && id.back() == '_') id.pop_back();
    if (id.empty() || id[0] < 'a' || id[0] > 'z') id = "s" + id;
    if (id.size() > 64) id.resize(64);
    return unique_id(id, [&](const std::string& c) { return s.find_source(c) != nullptr; });
}

std::string pane_new_spec_id(const std::string& title, const std::function<bool(const std::string&)>& taken) {
    std::string id;
    bool dash = false;
    for (char c : title) {
        const char l = (char)std::tolower((unsigned char)c);
        if ((l >= 'a' && l <= 'z') || (l >= '0' && l <= '9')) {
            if (dash && !id.empty()) id += '-';
            dash = false;
            id += l;
        } else {
            dash = true;
        }
    }
    if (id.empty()) id = "dashboard";
    if (id[0] < 'a' || id[0] > 'z') id = "d-" + id;
    if (id.size() > 56) id.resize(56);
    while (!id.empty() && id.back() == '-') id.pop_back();
    return unique_id(id, taken ? taken : [](const std::string&) { return false; });
}

std::string pane_panel_id(Primitive p, const spec::Spec& s) {
    return unique_id(spec::to_string(p), [&](const std::string& c) { return s.find_panel(c) != nullptr; });
}

spec::Spec pane_rename_panel(const spec::Spec& s, const std::string& from, const std::string& to) {
    spec::Spec out = s;
    for (auto& p : out.panels)
        if (p.id == from) p.id = to;
    if (out.layout) rename_leaves(*out.layout, from, to);
    return out;
}

spec::Spec pane_rename_source(const spec::Spec& s, const std::string& from, const std::string& to) {
    spec::Spec out = s;
    for (auto& src : out.sources) {
        if (src.id == from) src.id = to;
        if (src.from && *src.from == from) src.from = to;
    }
    for (auto& p : out.panels)
        for (auto& [slot, v] : p.bind)
            for (auto& b : v.bindings) {
                const size_t dot = b.find('.');
                const std::string head = dot == std::string::npos ? b : b.substr(0, dot);
                if (head == from) b = to + (dot == std::string::npos ? std::string() : b.substr(dot));
            }
    return out;
}

// ---- the spec text ----------------------------------------------------------------------

namespace {

std::vector<std::string> text_lines(const std::string& t) {
    std::vector<std::string> out;
    size_t at = 0;
    while (at < t.size()) {
        size_t end = t.find('\n', at);
        if (end == std::string::npos) end = t.size();
        out.push_back(t.substr(at, end - at));
        at = end + 1;
    }
    return out;
}

}  // namespace

int pane_text_follow_line(const std::string& before, const std::string& after) {
    if (before == after) return -1;
    const std::vector<std::string> a = text_lines(before), b = text_lines(after);
    size_t head = 0;
    while (head < a.size() && head < b.size() && a[head] == b[head]) ++head;
    size_t tail = 0;
    while (tail < a.size() - head && tail < b.size() - head && a[a.size() - 1 - tail] == b[b.size() - 1 - tail]) ++tail;
    if (b.empty()) return -1;
    // The changed lines of `after` are [head, b.size() - tail); with none,
    // lines were only taken out, before line `head`.
    if (b.size() - tail > head) return (int)(b.size() - tail - 1);
    return (int)std::min(head, b.size() - 1);
}

// ---- the layout tree --------------------------------------------------------------------

namespace {

LayoutNode* node_at(LayoutNode& root, const std::string& path) {
    LayoutNode* n = &root;
    size_t at = 0;
    while (at < path.size()) {
        size_t dot = path.find('.', at);
        if (dot == std::string::npos) dot = path.size();
        size_t i = 0;
        for (size_t k = at; k < dot; ++k) {
            if (path[k] < '0' || path[k] > '9') return nullptr;
            i = i * 10 + (size_t)(path[k] - '0');
        }
        if (n->kind == LayoutNode::Kind::panel || i >= n->children.size()) return nullptr;
        n = &n->children[i];
        at = dot + 1;
    }
    return n;
}

// Takes the leaf of `panel` out; a split left with one child becomes that
// child. False when the layout would be empty.
bool take_out(LayoutNode& n, const std::string& panel, bool* gone, int depth = 0) {
    *gone = false;
    if (depth > 64) return false;
    if (n.kind == LayoutNode::Kind::panel) {
        if (n.panel != panel) return false;
        *gone = true;
        return true;
    }
    for (size_t i = 0; i < n.children.size(); ++i) {
        bool drop = false;
        if (!take_out(n.children[i], panel, &drop, depth + 1)) continue;
        if (drop) n.children.erase(n.children.begin() + (std::ptrdiff_t)i);
        if (n.children.empty()) {
            *gone = true;
        } else if (n.kind == LayoutNode::Kind::split && n.children.size() == 1) {
            LayoutNode only = std::move(n.children[0]);
            if (only.kind != LayoutNode::Kind::panel && !only.title && n.title) only.title = n.title;
            n = std::move(only);
        } else if (n.kind == LayoutNode::Kind::split && n.children.size() != 2) {
            n.ratio.reset();
        }
        return true;
    }
    return false;
}

LayoutNode wrap(LayoutNode old, const std::string& panel, const std::string& where) {
    LayoutNode w;
    if (where == "tabbed") {
        if (old.kind == LayoutNode::Kind::tabs) {
            old.children.push_back(LayoutNode::of_panel(panel));
            return old;
        }
        w.kind = LayoutNode::Kind::tabs;
        w.children.push_back(std::move(old));
        w.children.push_back(LayoutNode::of_panel(panel));
        return w;
    }
    w.kind = LayoutNode::Kind::split;
    w.split = where == "beside" || where == "left" ? spec::SplitDirection::horizontal : spec::SplitDirection::vertical;
    if (where == "left" || where == "above") {
        w.children.push_back(LayoutNode::of_panel(panel));
        w.children.push_back(std::move(old));
    } else {
        w.children.push_back(std::move(old));
        w.children.push_back(LayoutNode::of_panel(panel));
    }
    return w;
}

bool place_at(LayoutNode& n, const std::string& target, const std::string& panel, const std::string& where,
              int depth = 0) {
    if (depth > 64) return false;
    if (n.kind == LayoutNode::Kind::panel) {
        if (n.panel != target) return false;
        n = wrap(std::move(n), panel, where);
        return true;
    }
    if (n.kind == LayoutNode::Kind::tabs && where == "tabbed") {
        for (size_t i = 0; i < n.children.size(); ++i)
            if (n.children[i].kind == LayoutNode::Kind::panel && n.children[i].panel == target) {
                n.children.insert(n.children.begin() + (std::ptrdiff_t)(i + 1), LayoutNode::of_panel(panel));
                return true;
            }
    }
    for (auto& c : n.children)
        if (place_at(c, target, panel, where, depth + 1)) return true;
    return false;
}

bool places(const LayoutNode& n, const std::string& panel, int depth = 0) {
    if (depth > 64) return false;
    if (n.kind == LayoutNode::Kind::panel) return n.panel == panel;
    for (const auto& c : n.children)
        if (places(c, panel, depth + 1)) return true;
    return false;
}

}  // namespace

const LayoutNode* pane_layout_node(const LayoutNode& root, const std::string& path) {
    return node_at(const_cast<LayoutNode&>(root), path);
}

LayoutNode pane_place_at_root(const std::optional<LayoutNode>& root, const std::string& panel,
                              const std::string& where) {
    if (!root) return LayoutNode::of_panel(panel);
    LayoutNode r = *root;
    bool gone = false;
    take_out(r, panel, &gone);
    if (gone) return LayoutNode::of_panel(panel);
    return wrap(std::move(r), panel, where);
}

std::optional<LayoutNode> pane_place_next_to(const std::optional<LayoutNode>& root, const std::string& panel,
                                             const std::string& target, const std::string& where) {
    if (!root || panel == target || !places(*root, target)) return std::nullopt;
    LayoutNode r = *root;
    bool gone = false;
    take_out(r, panel, &gone);
    if (gone) return std::nullopt;
    if (!place_at(r, target, panel, where)) return std::nullopt;
    return r;
}

LayoutNode pane_with_ratio(const LayoutNode& root, const std::string& path, double ratio) {
    LayoutNode r = root;
    LayoutNode* n = node_at(r, path);
    if (!n || n->kind != LayoutNode::Kind::split || n->children.size() != 2) return r;
    const double t = std::round(ratio * 1e12) / 1e12;
    if (std::fabs(t - 0.5) < 1e-12) n->ratio.reset();
    else n->ratio = t;
    return r;
}

double pane_snap_ratio(double ratio) {
    if (!std::isfinite(ratio)) return 0.5;
    const double r = std::round(ratio * 100.0) / 100.0;
    return std::clamp(r, 0.05, 0.95);
}

}  // namespace caliper_host
