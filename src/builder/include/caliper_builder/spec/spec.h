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

namespace caliper_builder::spec {

// The format version this library reads and writes (`caliper_spec`).
inline constexpr std::int64_t kFormatVersion = 0;

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
}  // namespace limits

enum class Primitive { series, heatmap, points, histogram, table, image, text };
enum class TransformOp { slice, reduce, normalise };
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
    nlohmann::json x;                        // extensions; null when absent
};

// A source: exactly one of `catalog`, `sql` and `from`, then transforms.
struct Source {
    std::string id;
    std::optional<std::string> catalog;      // "<kind>:<name>"
    std::optional<std::string> sql;          // one read-only SELECT
    std::optional<std::string> from;         // another source's id
    std::vector<Transform> transforms;       // the `transform` array
    nlohmann::json x;
};

// The value of one slot of `bind`: one binding, or a list (series.y only).
struct SlotValue {
    std::vector<std::string> bindings;
    bool list = false;  // written as a list
};

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
    nlohmann::json x;
};

// A layout node: a panel id, a split or tabs.
struct LayoutNode {
    enum class Kind { panel, split, tabs };
    Kind kind = Kind::panel;
    std::string panel;                         // Kind::panel
    SplitDirection split = SplitDirection::horizontal;  // Kind::split
    std::optional<double> ratio;               // Kind::split, two children only
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
    nlohmann::json x;

    // The first source or panel with this id, or null.
    const Source* find_source(std::string_view id) const;
    const Panel* find_panel(std::string_view id) const;
};

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
