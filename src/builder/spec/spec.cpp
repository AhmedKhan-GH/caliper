// The value types of a spec: names of the enumerations, equality and the
// diagnostic helpers.
#include <caliper_builder/spec/spec.h>

#include <algorithm>
#include <cmath>
#include <set>

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
    {"text", static_cast<int>(Primitive::text)},         {"scene", static_cast<int>(Primitive::scene)},
};
constexpr Name kOps[] = {
    {"slice", static_cast<int>(TransformOp::slice)},
    {"reduce", static_cast<int>(TransformOp::reduce)},
    {"normalise", static_cast<int>(TransformOp::normalise)},
    {"window", static_cast<int>(TransformOp::window)},
    {"surface", static_cast<int>(TransformOp::surface)},
    {"path", static_cast<int>(TransformOp::path)},
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

bool operator==(const SavedView& a, const SavedView& b) {
    return a.name == b.name && a.target == b.target && same(a.distance, b.distance) && same(a.azimuth, b.azimuth) &&
           same(a.elevation, b.elevation);
}

bool operator==(const StateWrite& a, const StateWrite& b) {
    return a.state == b.state && a.field == b.field && a.entity == b.entity;
}

bool Options::empty() const { return set_names().empty() && x.is_null(); }

std::vector<std::string> Options::set_names() const {
    // A set keeps the names in byte order and once each: a key that holds a
    // reference is set, and the literal field behind it is empty.
    std::set<std::string> names;
    if (bins) names.emplace("bins");
    if (colormap) names.emplace("colormap");
    if (columns) names.emplace("columns");
    if (range) names.emplace("range");
    if (text) names.emplace("text");
    if (x_label) names.emplace("x_label");
    if (x_log) names.emplace("x_log");
    if (x_range) names.emplace("x_range");
    if (y_label) names.emplace("y_label");
    if (y_log) names.emplace("y_log");
    if (y_range) names.emplace("y_range");
    if (z_label) names.emplace("z_label");
    // The scene options, merged into byte order with the others.
    if (scene.axes) names.emplace("axes");
    if (!scene.exclude.empty()) names.emplace("exclude");
    if (!scene.include.empty()) names.emplace("include");
    if (scene.labels) names.emplace("labels");
    if (scene.scale_bar) names.emplace("scale_bar");
    if (scene.up) names.emplace("up");
    if (!scene.view_camera.is_null()) names.emplace("view_camera");
    // Version 1.
    if (!legend.is_null()) names.emplace("legend");
    if (!box.is_null()) names.emplace("box");
    if (!views.empty()) names.emplace("views");
    if (select) names.emplace("select");
    if (bound_text) {
        names.emplace("format");
        if (!bound_text->fields.empty()) names.emplace("fields");
    }
    for (const auto& entry : references) names.insert(entry.first);
    return std::vector<std::string>(names.begin(), names.end());
}

LayoutNode LayoutNode::of_panel(std::string id) {
    LayoutNode n;
    n.kind = Kind::panel;
    n.panel = std::move(id);
    return n;
}

const std::vector<IntegerPlace>& integer_places() {
    static const std::vector<IntegerPlace> places = {
        {TransformOp::slice, "index", &Transform::index, -limits::kMaxSafeInteger, limits::kMaxSafeInteger},
        {TransformOp::slice, "start", &Transform::start, -limits::kMaxSafeInteger, limits::kMaxSafeInteger},
        {TransformOp::slice, "stop", &Transform::stop, -limits::kMaxSafeInteger, limits::kMaxSafeInteger},
        {TransformOp::slice, "step", &Transform::step, 1, limits::kMaxSafeInteger},
        {TransformOp::window, "start", &Transform::start, -limits::kMaxSafeInteger, limits::kMaxSafeInteger},
        {TransformOp::window, "stop", &Transform::stop, -limits::kMaxSafeInteger, limits::kMaxSafeInteger},
        {TransformOp::window, "steps", &Transform::steps, 1, limits::kHistorySteps},
    };
    return places;
}

const IntegerPlace* integer_place(TransformOp op, std::string_view key) {
    for (const IntegerPlace& p : integer_places())
        if (p.op == op && key == p.key) return &p;
    return nullptr;
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

const Program* Spec::find_program(std::string_view id) const {
    for (const Program& p : programs)
        if (p.id == id) return &p;
    return nullptr;
}

const StateVar* Spec::find_state(std::string_view id) const {
    for (const StateVar& s : states)
        if (s.id == id) return &s;
    return nullptr;
}

bool operator==(const Transform& a, const Transform& b) {
    return a.op == b.op && a.axis == b.axis && a.axis_list == b.axis_list && a.index == b.index &&
           a.start == b.start && a.stop == b.stop && a.step == b.step && a.fn == b.fn && a.keepdims == b.keepdims &&
           a.mode == b.mode && a.steps == b.steps && a.max_cells == b.max_cells && a.pool == b.pool &&
           a.columns == b.columns && a.max_points == b.max_points && a.references == b.references && a.x == b.x;
}

bool operator==(const Source& a, const Source& b) {
    return a.id == b.id && a.catalog == b.catalog && a.sql == b.sql && a.from == b.from &&
           a.transforms == b.transforms && a.history == b.history && a.x == b.x;
}

bool operator==(const SlotValue& a, const SlotValue& b) {
    return a.bindings == b.bindings && a.list == b.list && a.reference == b.reference;
}

bool operator==(const Options& a, const Options& b) {
    return a.x_label == b.x_label && a.y_label == b.y_label && a.z_label == b.z_label && same(a.x_range, b.x_range) &&
           same(a.y_range, b.y_range) && same(a.range, b.range) && a.x_log == b.x_log && a.y_log == b.y_log &&
           a.colormap == b.colormap && a.bins == b.bins && a.columns == b.columns && a.text == b.text && a.scene == b.scene &&
           a.legend == b.legend && a.box == b.box && a.views == b.views && a.select == b.select &&
           a.bound_text == b.bound_text && a.references == b.references && a.x == b.x;
}

bool operator==(const Panel& a, const Panel& b) {
    return a.id == b.id && a.primitive == b.primitive && a.title == b.title && a.bind == b.bind &&
           a.options == b.options && a.entities == b.entities && a.annotations == b.annotations && a.x == b.x;
}

bool operator==(const LayoutNode& a, const LayoutNode& b) {
    return a.kind == b.kind && a.panel == b.panel && a.split == b.split && same(a.ratio, b.ratio) &&
           a.shares == b.shares && a.min == b.min && a.children == b.children && a.title == b.title && a.x == b.x;
}

bool operator==(const Applet& a, const Applet& b) {
    return a.id == b.id && a.title == b.title && a.description == b.description && a.x == b.x;
}

bool operator==(const Spec& a, const Spec& b) {
    return a.version == b.version && a.applet == b.applet && a.sources == b.sources && a.panels == b.panels &&
           a.layout == b.layout && a.controls == b.controls && a.programs == b.programs && a.stages == b.stages && a.pieces == b.pieces &&
           a.states == b.states && a.provenance == b.provenance && a.x == b.x;
}

std::int64_t primitive_version(Primitive p) { return p == Primitive::scene ? kInstrumentVersion : 0; }

namespace {

bool layout_needs_version_1(const LayoutNode& n) {
    if (!n.shares.empty() || n.min) return true;
    for (const LayoutNode& c : n.children)
        if (layout_needs_version_1(c)) return true;
    return false;
}

bool transform_needs_version_1(const Transform& t) {
    return t.op == TransformOp::window || t.op == TransformOp::surface || t.op == TransformOp::path ||
           !t.references.empty();
}

bool panel_needs_version_1(const Panel& p) {
    if (primitive_version(p.primitive) > 0 || !p.entities.empty() || !p.annotations.empty()) return true;
    for (const auto& slot : p.bind)
        if (slot.second.reference) return true;
    const Options& o = p.options;
    if (!o.legend.is_null() || !o.box.is_null() || !o.views.empty() || o.select || o.bound_text || !o.references.empty())
        return true;
    // The scale bar belongs to version 0 only on a scene panel, which is
    // version 1 itself, so that a set scale bar on another primitive is new.
    return o.scene.scale_bar && p.primitive != Primitive::scene;
}

}  // namespace

std::int64_t required_version(const Spec& spec) {
    if (!spec.controls.empty() || !spec.programs.empty() || !spec.stages.empty() || !spec.provenance.empty() ||
        !spec.states.empty())
        return kInstrumentVersion;
    for (const Source& s : spec.sources) {
        if (s.history) return kInstrumentVersion;
        for (const Transform& t : s.transforms)
            if (transform_needs_version_1(t)) return kInstrumentVersion;
    }
    for (const Panel& p : spec.panels)
        if (panel_needs_version_1(p)) return kInstrumentVersion;
    if (spec.layout && layout_needs_version_1(*spec.layout)) return kInstrumentVersion;
    return 0;
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
