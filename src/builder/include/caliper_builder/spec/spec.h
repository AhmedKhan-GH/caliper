#pragma once
// caliper_builder::spec: the value types of a spec, format version 0
// (design/spec-format.md).
//
// A spec describes an applet built from data: its sources, its panels and their
// layout. `Spec` is the parsed document, a plain value that can be copied,
// compared and edited. Reading and writing it are in format.h, checking it in
// validate.h and merge patches in patch.h.
//
// A `Spec` keeps what the document holds and no more: an optional key that was
// not written is empty, and a key written with its default value keeps it, so
// the canonical writers give back what was read. Tables named `x` are kept as
// opaque JSON objects (null when absent). Nothing here throws.
//
// `Diagnostic` is the one report of the spec library: an error or a warning
// with a stable code ("spec.unknown_key"), one plain sentence, the document path
// of the value it concerns ("panel[2].bind.y") and, for a spec read from TOML,
// the line and column of that value (0 when not known).
#include <array>
#include <cstddef>
#include <cstdint>
#include <map>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include <nlohmann/json.hpp>

#include <caliper_builder/error.h>
#include <caliper_builder/spec/annotation.h>
#include <caliper_builder/spec/entity.h>
#include <caliper_builder/spec/instrument.h>
#include <caliper_builder/spec/program.h>
#include <caliper_builder/spec/reference.h>
#include <caliper_builder/spec/state.h>

namespace caliper_builder::spec {

// The format version of a new Spec (`caliper_spec`). The writers write the
// version a Spec holds and never raise it.
inline constexpr std::int64_t kFormatVersion = 0;
// The newest format version this library reads and writes: version 1 adds
// the scene primitive and the control table (instrument.h). The reader knows
// the versions kFormatVersion to kLatestFormatVersion.
inline constexpr std::int64_t kLatestFormatVersion = kInstrumentVersion;

// The limits of version 0. Each one has the error code the design page lists.
namespace limits {
inline constexpr std::size_t kTextBytes = 512 * 1024;        // spec.too_large
inline constexpr std::size_t kValues = 65536;                 // spec.too_many
inline constexpr int kDepth = 32;                             // spec.too_deep
inline constexpr std::size_t kSources = 64;
inline constexpr std::size_t kPanels = 64;
inline constexpr std::size_t kTransforms = 16;
inline constexpr std::size_t kYBindings = 16;
inline constexpr std::size_t kColumns = 256;
inline constexpr std::size_t kReduceAxes = 8;
inline constexpr int kLayoutDepth = 8;
inline constexpr std::size_t kLayoutNodes = 256;
inline constexpr std::size_t kLayoutChildren = 64;
inline constexpr std::size_t kIdChars = 64;
inline constexpr std::size_t kTitleChars = 256;
inline constexpr std::size_t kDescriptionChars = 4096;
inline constexpr std::size_t kCatalogChars = 256;
inline constexpr std::size_t kColumnChars = 255;
inline constexpr std::size_t kSqlChars = 65536;
inline constexpr std::size_t kTextChars = 16384;
inline constexpr std::size_t kStringChars = 65536;
inline constexpr std::int64_t kMaxSafeInteger = 9007199254740991;  // 2^53 - 1
inline constexpr int kMaxRank = 8;
inline constexpr std::int64_t kMaxBins = 65536;
inline constexpr std::size_t kDiagnostics = 256;  // one report keeps at most this many
// Version 1 (design/instrument-interface.md).
inline constexpr std::int64_t kHistorySteps = 65536;         // a source's `history` and a window's `steps`
inline constexpr std::size_t kSavedViews = 16;
inline constexpr std::int64_t kLeafMinimum = 400;            // a layout leaf's `min`, from 4
}  // namespace limits

// The primitives of a panel. `scene` is part of version 1 only.
enum class Primitive { series, heatmap, points, histogram, table, image, text, scene };
// `window`, `surface` and `path` are part of version 1 only
// (design/instrument-interface.md): a step over the kept steps of a source,
// and the two operators that turn a tensor into geometry.
enum class TransformOp { slice, reduce, normalise, window, surface, path };
enum class SplitDirection { horizontal, vertical };

const char* to_string(Primitive p);
const char* to_string(TransformOp op);
const char* to_string(SplitDirection d);
bool primitive_from_string(std::string_view s, Primitive& out);
bool transform_op_from_string(std::string_view s, TransformOp& out);
bool split_from_string(std::string_view s, SplitDirection& out);

// One tensor transform (contracts, section 4). Which fields apply depends on
// `op`; validation reports a field that does not.
struct Transform {
    TransformOp op = TransformOp::slice;
    std::vector<std::int64_t> axis;          // empty when absent
    bool axis_list = false;                  // reduce: `axis` written as a list
    std::optional<std::int64_t> index;       // slice
    std::optional<std::int64_t> start;       // slice
    std::optional<std::int64_t> stop;        // slice
    std::optional<std::int64_t> step;        // slice
    std::optional<std::string> fn;           // reduce: mean sum min max std var norm argmax argmin entropy
    std::optional<bool> keepdims;            // reduce
    std::optional<std::string> mode;         // normalise: minmax zscore softmax l2
    std::optional<std::int64_t> steps;       // window: the last N kept steps up to the cursor
    std::optional<std::int64_t> max_cells;   // surface: 1 to 262,144, default 32,768
    std::optional<std::string> pool;         // surface: mean max min
    std::vector<std::int64_t> columns;       // path: two or three column indices; empty when absent
    std::optional<std::int64_t> max_points;  // path: 2 to 1,048,576, default 65,536
    // The references at the value places of this step, by key: `index`,
    // `start`, `stop` and `step` of a slice, `start`, `stop` and `steps` of a
    // window. The literal field of a key with a reference is empty.
    std::map<std::string, Reference> references;
    nlohmann::json x;                        // extensions; null when absent
};

// An integer value place of a step: a key that takes a literal or a
// reference, the field of the step that holds it, and the range of its
// values. The validator checks a literal against this range, and the resolver
// checks each value that a reference gives there against the same range, so
// every key in the table is checked at both places, and a key that is not in
// it takes a literal only.
struct IntegerPlace {
    TransformOp op;
    const char* key;
    std::optional<std::int64_t> Transform::*field;
    std::int64_t min;
    std::int64_t max;
};

// The integer places of every step kind.
const std::vector<IntegerPlace>& integer_places();
// The integer place `key` of a step `op`; null when the key takes a literal only.
const IntegerPlace* integer_place(TransformOp op, std::string_view key);

// A source: exactly one of `catalog`, `sql` and `from`, then transforms.
struct Source {
    std::string id;
    std::optional<std::string> catalog;      // "<kind>:<name>"
    std::optional<std::string> sql;          // one read-only SELECT
    std::optional<std::string> from;         // another source's id
    std::vector<Transform> transforms;       // the `transform` array
    std::optional<std::int64_t> history;     // version 1: the steps the app keeps, 0 to 65,536
    nlohmann::json x;
};

// The value of one slot of `bind`: one binding, or a list (series.y only).
struct SlotValue {
    std::vector<std::string> bindings;
    bool list = false;  // written as a list
    // Version 1: a reference to a text state whose choices are source ids,
    // written in place of the binding; `bindings` is then empty.
    std::optional<Reference> reference;
};

// A saved camera view of a scene panel (the option `views`).
struct SavedView {
    std::string name;                        // unique in the panel
    std::array<double, 3> target{0, 0, 0};
    double distance = 1;
    double azimuth = 0;                      // degrees
    double elevation = 0;                    // degrees
};
bool operator==(const SavedView& a, const SavedView& b);

// The option `select`: the field of each pick that goes into a state variable.
struct StateWrite {
    std::string state;
    std::string field;                       // a pick field (design/instrument-interface.md, Pick Fields)
    std::optional<std::string> entity;       // only picks on this entity
};
bool operator==(const StateWrite& a, const StateWrite& b);

// The options of every primitive; each primitive takes only some of them.
struct Options {
    std::optional<std::string> x_label;
    std::optional<std::string> y_label;
    std::optional<std::string> z_label;
    std::optional<std::array<double, 2>> x_range;
    std::optional<std::array<double, 2>> y_range;
    std::optional<std::array<double, 2>> range;
    std::optional<bool> x_log;
    std::optional<bool> y_log;
    std::optional<std::string> colormap;     // viridis magma rdbu greys
    std::optional<std::int64_t> bins;
    std::optional<std::vector<std::string>> columns;
    std::optional<std::string> text;
    SceneOptions scene;                      // scene: up, view_camera, include, exclude, labels, axes, scale_bar
    // Version 1 (design/instrument-interface.md):
    nlohmann::json legend;                   // "auto", true or false; null when absent
    nlohmann::json box;                      // scene: "auto" or three positive numbers; null when absent
    std::vector<SavedView> views;            // scene: at most 16 saved views
    std::optional<StateWrite> select;        // a pick writes one of its fields into a state variable
    std::optional<BoundText> bound_text;     // text: the options `format` and `fields`
    // The references at the value places of the options, by key: `range`,
    // `x_range`, `y_range` and `colormap`. The literal field is then empty.
    std::map<std::string, Reference> references;
    nlohmann::json x;

    bool empty() const;
    // The names of the options that are set, `x` excluded.
    std::vector<std::string> set_names() const;
};

struct Panel {
    std::string id;
    Primitive primitive = Primitive::series;
    std::optional<std::string> title;
    std::map<std::string, SlotValue> bind;   // slot name to its value
    Options options;
    std::vector<Entity> entities;            // scene, version 1: the array `entity`
    std::vector<Annotation> annotations;     // version 1: the array `annotation`
    nlohmann::json x;
};

// A layout node: a panel id, a split or tabs. From spec version 1 a leaf's
// `panel` may also name a control (instrument.h), since panels and controls
// share one set of ids.
struct LayoutNode {
    enum class Kind { panel, split, tabs };
    Kind kind = Kind::panel;
    std::string panel;                         // Kind::panel
    SplitDirection split = SplitDirection::horizontal;  // Kind::split
    std::optional<double> ratio;               // Kind::split, two children only
    std::vector<double> shares;                // Kind::split, version 1: one positive share per child
    // Kind::panel, version 1: the minimum size in columns and lines of the
    // text scale, written as the leaf table { panel, min }.
    std::optional<std::array<std::int64_t, 2>> min;
    std::vector<LayoutNode> children;          // the children of a split, or the tabs
    std::optional<std::string> title;          // Kind::split and Kind::tabs
    nlohmann::json x;

    static LayoutNode of_panel(std::string id);
};

struct Applet {
    std::string id;
    std::optional<std::string> title;
    std::optional<std::string> description;
    nlohmann::json x;
};

struct Spec {
    std::int64_t version = kFormatVersion;   // `caliper_spec`
    Applet applet;
    std::vector<Source> sources;
    std::vector<Panel> panels;
    std::optional<LayoutNode> layout;
    std::vector<Control> controls;   // `control`, spec version 1 (instrument.h); empty in version 0
    std::vector<Program> programs;   // `program`, spec version 1 (program.h); empty in version 0
    std::vector<Stage> stages;       // `stage`, spec version 1 (program.h); empty in version 0
    std::vector<StateVar> states;    // `state`, spec version 1 (state.h); empty in version 0
    Provenance provenance;           // `provenance`, spec version 1 (program.h); empty for the person's own spec
    nlohmann::json x;

    // The first source, panel or control with this id, or null.
    const Source* find_source(std::string_view id) const;
    const Panel* find_panel(std::string_view id) const;
    const Control* find_control(std::string_view id) const;
    const Program* find_program(std::string_view id) const;
    const StateVar* find_state(std::string_view id) const;
};

// The lowest format version that holds `spec`: 1 when it has a scene panel, a
// control, a program, a stage, a provenance table or any key of
// design/instrument-interface.md, else 0. An edit that adds
// one of them to a version 0 spec raises the version to this one in the same
// step.
std::int64_t required_version(const Spec& spec);
// The first version whose documents may use the primitive.
std::int64_t primitive_version(Primitive p);

bool operator==(const Transform& a, const Transform& b);
bool operator==(const Source& a, const Source& b);
bool operator==(const SlotValue& a, const SlotValue& b);
bool operator==(const Options& a, const Options& b);
bool operator==(const Panel& a, const Panel& b);
bool operator==(const LayoutNode& a, const LayoutNode& b);
bool operator==(const Applet& a, const Applet& b);
bool operator==(const Spec& a, const Spec& b);
inline bool operator!=(const Transform& a, const Transform& b) { return !(a == b); }
inline bool operator!=(const Source& a, const Source& b) { return !(a == b); }
inline bool operator!=(const SlotValue& a, const SlotValue& b) { return !(a == b); }
inline bool operator!=(const Options& a, const Options& b) { return !(a == b); }
inline bool operator!=(const Panel& a, const Panel& b) { return !(a == b); }
inline bool operator!=(const LayoutNode& a, const LayoutNode& b) { return !(a == b); }
inline bool operator!=(const Applet& a, const Applet& b) { return !(a == b); }
inline bool operator!=(const Spec& a, const Spec& b) { return !(a == b); }

enum class Severity { error, warning };

struct Diagnostic {
    Severity severity = Severity::error;
    std::string code;
    std::string message;
    std::string path;   // "panel[2].bind.y"; empty for the whole document
    int line = 0;       // 1-based, in the TOML text; 0 when not known
    int column = 0;

    bool is_error() const { return severity == Severity::error; }
    // An Error whose `where` is the path, or "line:column" when there is no path.
    Error to_error() const;
};

bool has_errors(const std::vector<Diagnostic>& diagnostics);
// {"severity": "error", "code": ..., "message": ..., "path": ..., "line": n, "column": n};
// line and column only when known.
nlohmann::json to_json(const Diagnostic& d);

}  // namespace caliper_builder::spec
