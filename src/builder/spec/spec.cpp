// The value types of a spec: names of the enumerations, equality and the
// diagnostic helpers.
#include <caliper_builder/spec/spec.h>

#include <cmath>

namespace caliper_builder::spec {

namespace {

struct Name {
    const char* text;
    int value;
};

constexpr Name kPrimitives[] = {
    {"series", static_cast<int>(Primitive::series)},       {"heatmap", static_cast<int>(Primitive::heatmap)},
    {"points", static_cast<int>(Primitive::points)},       {"histogram", static_cast<int>(Primitive::histogram)},
    {"table", static_cast<int>(Primitive::table)},         {"image", static_cast<int>(Primitive::image)},
    {"text", static_cast<int>(Primitive::text)},
};
constexpr Name kOps[] = {
    {"slice", static_cast<int>(TransformOp::slice)},
    {"reduce", static_cast<int>(TransformOp::reduce)},
    {"normalise", static_cast<int>(TransformOp::normalise)},
};
constexpr Name kSplits[] = {
    {"horizontal", static_cast<int>(SplitDirection::horizontal)},
    {"vertical", static_cast<int>(SplitDirection::vertical)},
};

template <std::size_t N>
const char* name_of(const Name (&names)[N], int value) {
    for (const Name& n : names)
        if (n.value == value) return n.text;
    return "";
}

template <std::size_t N>
bool value_of(const Name (&names)[N], std::string_view s, int& out) {
    for (const Name& n : names)
        if (s == n.text) {
            out = n.value;
            return true;
        }
    return false;
}

// Two doubles are equal when both are NaN or they compare equal, so that an
// invalid Spec still equals its own copy.
bool same(double a, double b) { return (std::isnan(a) && std::isnan(b)) || a == b; }

bool same(const std::optional<double>& a, const std::optional<double>& b) {
    if (a.has_value() != b.has_value()) return false;
    return !a || same(*a, *b);
}

bool same(const std::optional<std::array<double, 2>>& a, const std::optional<std::array<double, 2>>& b) {
    if (a.has_value() != b.has_value()) return false;
    return !a || (same((*a)[0], (*b)[0]) && same((*a)[1], (*b)[1]));
}

}  // namespace

const char* to_string(Primitive p) { return name_of(kPrimitives, static_cast<int>(p)); }
const char* to_string(TransformOp op) { return name_of(kOps, static_cast<int>(op)); }
const char* to_string(SplitDirection d) { return name_of(kSplits, static_cast<int>(d)); }

bool primitive_from_string(std::string_view s, Primitive& out) {
    int v = 0;
    if (!value_of(kPrimitives, s, v)) return false;
    out = static_cast<Primitive>(v);
    return true;
}

bool transform_op_from_string(std::string_view s, TransformOp& out) {
    int v = 0;
    if (!value_of(kOps, s, v)) return false;
    out = static_cast<TransformOp>(v);
    return true;
}

bool split_from_string(std::string_view s, SplitDirection& out) {
    int v = 0;
    if (!value_of(kSplits, s, v)) return false;
    out = static_cast<SplitDirection>(v);
    return true;
}

bool Options::empty() const { return set_names().empty() && x.is_null(); }

std::vector<std::string> Options::set_names() const {
    std::vector<std::string> names;
    if (bins) names.emplace_back("bins");
    if (colormap) names.emplace_back("colormap");
    if (columns) names.emplace_back("columns");
    if (range) names.emplace_back("range");
    if (text) names.emplace_back("text");
    if (x_label) names.emplace_back("x_label");
    if (x_log) names.emplace_back("x_log");
    if (x_range) names.emplace_back("x_range");
    if (y_label) names.emplace_back("y_label");
    if (y_log) names.emplace_back("y_log");
    if (y_range) names.emplace_back("y_range");
    if (z_label) names.emplace_back("z_label");
    return names;
}

LayoutNode LayoutNode::of_panel(std::string id) {
    LayoutNode n;
    n.kind = Kind::panel;
    n.panel = std::move(id);
    return n;
}

const Source* Spec::find_source(std::string_view id) const {
    for (const Source& s : sources)
        if (s.id == id) return &s;
    return nullptr;
}

const Panel* Spec::find_panel(std::string_view id) const {
    for (const Panel& p : panels)
        if (p.id == id) return &p;
    return nullptr;
}

const Control* Spec::find_control(std::string_view id) const {
    for (const Control& c : controls)
        if (c.id == id) return &c;
    return nullptr;
}

bool operator==(const Transform& a, const Transform& b) {
    return a.op == b.op && a.axis == b.axis && a.axis_list == b.axis_list && a.index == b.index &&
           a.start == b.start && a.stop == b.stop && a.step == b.step && a.fn == b.fn && a.keepdims == b.keepdims &&
           a.mode == b.mode && a.x == b.x;
}

bool operator==(const Source& a, const Source& b) {
    return a.id == b.id && a.catalog == b.catalog && a.sql == b.sql && a.from == b.from &&
           a.transforms == b.transforms && a.x == b.x;
}

bool operator==(const SlotValue& a, const SlotValue& b) { return a.bindings == b.bindings && a.list == b.list; }

bool operator==(const Options& a, const Options& b) {
    return a.x_label == b.x_label && a.y_label == b.y_label && a.z_label == b.z_label && same(a.x_range, b.x_range) &&
           same(a.y_range, b.y_range) && same(a.range, b.range) && a.x_log == b.x_log && a.y_log == b.y_log &&
           a.colormap == b.colormap && a.bins == b.bins && a.columns == b.columns && a.text == b.text && a.x == b.x;
}

bool operator==(const Panel& a, const Panel& b) {
    return a.id == b.id && a.primitive == b.primitive && a.title == b.title && a.bind == b.bind &&
           a.options == b.options && a.x == b.x;
}

bool operator==(const LayoutNode& a, const LayoutNode& b) {
    return a.kind == b.kind && a.panel == b.panel && a.split == b.split && same(a.ratio, b.ratio) &&
           a.children == b.children && a.title == b.title && a.x == b.x;
}

bool operator==(const Applet& a, const Applet& b) {
    return a.id == b.id && a.title == b.title && a.description == b.description && a.x == b.x;
}

bool operator==(const Spec& a, const Spec& b) {
    return a.version == b.version && a.applet == b.applet && a.sources == b.sources && a.panels == b.panels &&
           a.layout == b.layout && a.controls == b.controls && a.x == b.x;
}

Error Diagnostic::to_error() const {
    std::string where = path;
    if (where.empty() && line > 0) where = std::to_string(line) + ":" + std::to_string(column);
    return Error(code, message, where);
}

bool has_errors(const std::vector<Diagnostic>& diagnostics) {
    for (const Diagnostic& d : diagnostics)
        if (d.is_error()) return true;
    return false;
}

nlohmann::json to_json(const Diagnostic& d) {
    nlohmann::json j = {
        {"severity", d.severity == Severity::error ? "error" : "warning"},
        {"code", d.code},
        {"message", d.message},
        {"path", d.path},
    };
    if (d.line > 0) {
        j["line"] = d.line;
        j["column"] = d.column;
    }
    return j;
}

}  // namespace caliper_builder::spec
