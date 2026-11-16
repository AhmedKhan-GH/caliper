#pragma once
// Internal to caliper_builder_spec: the tables of the format (slots, options,
// transform keys, enumerations), text checks, document paths and the bounded
// sink that collects diagnostics.
#include <cstddef>
#include <string>
#include <string_view>
#include <vector>

#include <nlohmann/json.hpp>

#include <caliper_builder/spec/spec.h>

namespace caliper_builder::spec::detail {

using json = nlohmann::json;

// ---- the format's tables -------------------------------------------------

const std::vector<std::string>& slots_of(Primitive p);
const std::vector<std::string>& options_of(Primitive p);  // `x` excluded
// The keys a transform of this op may hold, `op` and `x` excluded.
const std::vector<std::string>& transform_keys(TransformOp op);
const std::vector<std::string>& reduce_fns();
const std::vector<std::string>& normalise_modes();
const std::vector<std::string>& colormaps();
const std::vector<std::string>& catalog_kinds();
const std::vector<std::string>& all_primitive_names();
bool contains(const std::vector<std::string>& list, std::string_view s);
std::string join(const std::vector<std::string>& list);  // "a, b and c"

// ---- text -----------------------------------------------------------------

bool valid_utf8(std::string_view s);
// `s` with each byte that is not part of valid UTF-8 replaced by U+FFFD.
std::string to_valid_utf8(std::string_view s);
// The number of code points of valid UTF-8 text.
std::size_t code_points(std::string_view s);
// Line text has no control character; long text may hold tab, LF and CR.
enum class TextKind { line, long_text };
bool has_control(std::string_view s, TextKind kind);
bool valid_id(std::string_view s);
// `s` quoted for a message, shortened to at most 64 characters.
std::string quote(std::string_view s);

// ---- paths ----------------------------------------------------------------

std::string key_path(const std::string& base, std::string_view key);
std::string index_path(const std::string& base, std::size_t index);

// ---- diagnostics ----------------------------------------------------------

class Sink {
public:
    explicit Sink(std::vector<Diagnostic>& out) : out_(out) {}
    void error(std::string code, std::string message, std::string path);
    void warning(std::string code, std::string message, std::string path);
    std::size_t errors() const { return errors_; }
    // At limits::kDiagnostics entries the sink is full; an error then replaces
    // the last warning, and anything else is dropped.
    bool full() const { return out_.size() >= limits::kDiagnostics; }

private:
    void add(Severity s, std::string code, std::string message, std::string path);
    std::vector<Diagnostic>& out_;
    std::size_t errors_ = 0;
};

// ---- values ---------------------------------------------------------------

// Checks the rules of "Values" over a JSON tree: no null (unless allowed), finite
// numbers, safe integers, valid UTF-8, string lengths, nesting depth and the
// total count of values, which `count` accumulates. Returns false when it
// reported an error.
bool check_values(const json& v, const std::string& path, int depth, std::size_t& count, Sink& sink,
                  bool allow_null = false);

// The decoders of format.cpp, shared with patch.cpp.
Source decode_source(const json& item, const std::string& path, Sink& sink);
Panel decode_panel(const json& item, const std::string& path, Sink& sink);
LayoutNode decode_layout(const json& item, const std::string& path, Sink& sink);

}  // namespace caliper_builder::spec::detail
