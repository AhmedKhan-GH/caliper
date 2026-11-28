// Internal helpers of caliper_builder_spec: the format's tables, text checks,
// paths, the diagnostic sink and the value rules.
#include "internal.h"

#include <caliper_builder/spec/colormaps.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <iterator>

namespace caliper_builder::spec::detail {

// ---- the format's tables -------------------------------------------------

const std::vector<std::string>& slots_of(Primitive p) {
    static const std::vector<std::string> series = {"x", "y"};
    static const std::vector<std::string> values = {"values"};
    static const std::vector<std::string> points = {"color", "positions", "size", "x", "y", "z"};
    static const std::vector<std::string> table = {"rows"};
    static const std::vector<std::string> text = {"value"};
    static const std::vector<std::string> scene = {"scene"};
    switch (p) {
        case Primitive::series: return series;
        case Primitive::heatmap: return values;
        case Primitive::points: return points;
        case Primitive::histogram: return values;
        case Primitive::table: return table;
        case Primitive::image: return values;
        case Primitive::text: return text;
        case Primitive::scene: return scene;
    }
    return values;
}

const std::vector<std::string>& options_of(Primitive p) {
    // The keys `legend`, `box`, `views`, `select`, `format` and `fields`, and
    // `scale_bar` on image and points, belong to format version 1.
    static const std::vector<std::string> series = {"legend", "select",  "x_label", "x_log",
                                                    "x_range", "y_label", "y_log",   "y_range"};
    static const std::vector<std::string> heatmap = {"colormap", "legend", "range", "select", "x_label", "y_label"};
    static const std::vector<std::string> points = {"colormap", "legend",  "scale_bar", "select",
                                                    "x_label",  "y_label", "z_label"};
    static const std::vector<std::string> histogram = {"bins", "legend", "range", "x_label", "y_label", "y_log"};
    static const std::vector<std::string> table = {"columns"};
    static const std::vector<std::string> image = {"colormap", "legend", "range", "scale_bar", "select"};
    static const std::vector<std::string> text = {"fields", "format", "text"};
    static const std::vector<std::string> scene = {"axes",    "box",       "exclude", "include", "labels", "legend",
                                                   "scale_bar", "select", "up",      "view_camera", "views"};
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

const std::vector<std::string>& transform_keys(TransformOp op) {
    static const std::vector<std::string> slice = {"axis", "index", "start", "stop", "step"};
    static const std::vector<std::string> reduce = {"axis", "fn", "keepdims"};
    static const std::vector<std::string> normalise = {"axis", "mode"};
    static const std::vector<std::string> window = {"start", "stop", "steps"};
    static const std::vector<std::string> surface = {"max_cells", "pool"};
    static const std::vector<std::string> path = {"columns", "max_points"};
    switch (op) {
        case TransformOp::slice: return slice;
        case TransformOp::reduce: return reduce;
        case TransformOp::normalise: return normalise;
        case TransformOp::window: return window;
        case TransformOp::surface: return surface;
        case TransformOp::path: return path;
    }
    return slice;
}

const std::vector<std::string>& reduce_fns() {
    static const std::vector<std::string> v = {"mean", "sum",    "min",    "max",    "std",
                                               "var",  "norm",   "argmax", "argmin", "entropy"};
    return v;
}

const std::vector<std::string>& normalise_modes() {
    static const std::vector<std::string> v = {"minmax", "zscore", "softmax", "l2"};
    return v;
}

const std::vector<std::string>& pool_modes() {
    static const std::vector<std::string> v = {"mean", "max", "min"};
    return v;
}

const std::vector<std::string>& colormaps() { return colormap_names(); }

const std::vector<std::string>& catalog_kinds() {
    static const std::vector<std::string> v = {"table", "tensor", "live", "feed", "metric", "run"};
    return v;
}

const std::vector<std::string>& all_primitive_names() { return primitive_names(kLatestFormatVersion); }

const std::vector<std::string>& primitive_names(std::int64_t /*version*/) {
    static const std::vector<std::string> v = {"series", "heatmap", "points", "histogram",
                                               "table",  "image",   "text",   "scene"};
    return v;
}

const std::vector<std::string>& control_kind_names() {
    static const std::vector<std::string> v = {"slider",  "button",    "range",    "toggle",    "picker",
                                               "text",    "segmented", "readout",  "transport"};
    return v;
}

const std::vector<std::string>& control_option_names() {
    static const std::vector<std::string> v = {"choices",     "choices_from", "fields",      "follow", "format",
                                               "keys",        "log",          "max",         "max_length", "min",
                                               "min_span",    "multiple",     "placeholder", "speed",  "step",
                                               "value"};
    return v;
}

const std::vector<std::string>& scene_up_axes() {
    static const std::vector<std::string> v = {"+x", "-x", "+y", "-y", "+z", "-z"};
    return v;
}

bool contains(const std::vector<std::string>& list, std::string_view s) {
    for (const std::string& e : list)
        if (e == s) return true;
    return false;
}

std::string join(const std::vector<std::string>& list) {
    std::string out;
    for (std::size_t i = 0; i < list.size(); ++i) {
        if (i > 0) out += (i + 1 == list.size()) ? " and " : ", ";
        out += list[i];
    }
    return out;
}

// ---- text -----------------------------------------------------------------

namespace {

// The length of the UTF-8 sequence at s[i], or 0 when it is not valid there.
std::size_t utf8_length(std::string_view s, std::size_t i) {
    const auto b0 = static_cast<unsigned char>(s[i]);
    if (b0 < 0x80) return 1;
    std::size_t n = 0;
    std::uint32_t cp = 0;
    if ((b0 & 0xE0) == 0xC0) {
        n = 2;
        cp = b0 & 0x1F;
    } else if ((b0 & 0xF0) == 0xE0) {
        n = 3;
        cp = b0 & 0x0F;
    } else if ((b0 & 0xF8) == 0xF0) {
        n = 4;
        cp = b0 & 0x07;
    } else {
        return 0;
    }
    if (i + n > s.size()) return 0;
    for (std::size_t k = 1; k < n; ++k) {
        const auto b = static_cast<unsigned char>(s[i + k]);
        if ((b & 0xC0) != 0x80) return 0;
        cp = (cp << 6) | (b & 0x3F);
    }
    // Overlong forms, surrogates and values past U+10FFFF are not UTF-8.
    if ((n == 2 && cp < 0x80) || (n == 3 && cp < 0x800) || (n == 4 && cp < 0x10000)) return 0;
    if (cp >= 0xD800 && cp <= 0xDFFF) return 0;
    if (cp > 0x10FFFF) return 0;
    return n;
}

}  // namespace

bool valid_utf8(std::string_view s) {
    for (std::size_t i = 0; i < s.size();) {
        const std::size_t n = utf8_length(s, i);
        if (n == 0) return false;
        i += n;
    }
    return true;
}

std::string to_valid_utf8(std::string_view s) {
    std::string out;
    out.reserve(s.size());
    for (std::size_t i = 0; i < s.size();) {
        const std::size_t n = utf8_length(s, i);
        if (n == 0) {
            out += "\xEF\xBF\xBD";
            i += 1;
        } else {
            out.append(s.substr(i, n));
            i += n;
        }
    }
    return out;
}

std::size_t code_points(std::string_view s) {
    std::size_t count = 0;
    for (char c : s)
        if ((static_cast<unsigned char>(c) & 0xC0) != 0x80) ++count;
    return count;
}

bool has_control(std::string_view s, TextKind kind) {
    for (char ch : s) {
        const auto c = static_cast<unsigned char>(ch);
        if (c < 0x20 || c == 0x7F) {
            if (kind == TextKind::long_text && (c == '\t' || c == '\n' || c == '\r')) continue;
            return true;
        }
    }
    return false;
}

bool valid_id(std::string_view s) {
    if (s.empty() || s.size() > limits::kIdChars) return false;
    if (s[0] < 'a' || s[0] > 'z') return false;
    for (char c : s)
        if (!((c >= 'a' && c <= 'z') || (c >= '0' && c <= '9') || c == '_' || c == '-')) return false;
    return true;
}

std::string quote(std::string_view s) {
    std::string out = "\"";
    std::size_t shown = 0;
    for (std::size_t i = 0; i < s.size();) {
        if (shown == 64) {
            out += "...";
            break;
        }
        const std::size_t n = utf8_length(s, i);
        const auto c = static_cast<unsigned char>(s[i]);
        if (n == 0) {
            out += "\xEF\xBF\xBD";  // U+FFFD for a byte that is not UTF-8
            i += 1;
        } else if (c < 0x20 || c == 0x7F) {
            char buf[8];
            std::snprintf(buf, sizeof buf, "\\x%02X", c);
            out += buf;
            i += 1;
        } else {
            out.append(s.substr(i, n));
            i += n;
        }
        ++shown;
    }
    out += "\"";
    return out;
}

// ---- paths ----------------------------------------------------------------

std::string key_path(const std::string& base, std::string_view key) {
    if (base.empty()) return std::string(key);
    std::string p = base;
    p += '.';
    p += key;
    return p;
}

std::string index_path(const std::string& base, std::size_t index) {
    return base + "[" + std::to_string(index) + "]";
}

// ---- diagnostics ----------------------------------------------------------

void Sink::add(Severity s, std::string code, std::string message, std::string path) {
    if (s == Severity::error) ++errors_;
    if (full()) {
        // A full report keeps errors before warnings, so that a report with an
        // error never reads as a pass: the last warning gives way.
        if (s != Severity::error) return;
        auto warning = std::find_if(out_.rbegin(), out_.rend(), [](const Diagnostic& d) { return !d.is_error(); });
        if (warning == out_.rend()) return;
        out_.erase(std::next(warning).base());
    }
    Diagnostic d;
    d.severity = s;
    d.code = std::move(code);
    d.message = std::move(message);
    d.path = std::move(path);
    out_.push_back(std::move(d));
}

void Sink::error(std::string code, std::string message, std::string path) {
    add(Severity::error, std::move(code), std::move(message), std::move(path));
}

void Sink::warning(std::string code, std::string message, std::string path) {
    add(Severity::warning, std::move(code), std::move(message), std::move(path));
}

// ---- values ---------------------------------------------------------------

bool check_values(const json& v, const std::string& path, int depth, std::size_t& count, Sink& sink,
                  bool allow_null) {
    const std::string where = path.empty() ? std::string("the document") : path;
    if (++count > limits::kValues) {
        sink.error("spec.too_many", "The document holds more than " + std::to_string(limits::kValues) + " values.",
                   path);
        return false;
    }
    if ((v.is_array() || v.is_object()) && depth > limits::kDepth) {
        sink.error("spec.too_deep",
                   where + " nests more than " + std::to_string(limits::kDepth) + " levels of tables and arrays.",
                   path);
        return false;
    }
    switch (v.type()) {
        case json::value_t::null:
            if (allow_null) return true;
            sink.error("spec.wrong_type", where + " is null, which a spec cannot hold.", path);
            return false;
        case json::value_t::boolean: return true;
        case json::value_t::number_integer: {
            const std::int64_t i = v.get<std::int64_t>();
            if (i > limits::kMaxSafeInteger || i < -limits::kMaxSafeInteger) {
                sink.error("spec.out_of_range", where + " is an integer outside +-(2^53 - 1).", path);
                return false;
            }
            return true;
        }
        case json::value_t::number_unsigned:
            if (v.get<std::uint64_t>() > static_cast<std::uint64_t>(limits::kMaxSafeInteger)) {
                sink.error("spec.out_of_range", where + " is an integer outside +-(2^53 - 1).", path);
                return false;
            }
            return true;
        case json::value_t::number_float:
            if (!std::isfinite(v.get<double>())) {
                sink.error("spec.out_of_range", where + " is not a finite number.", path);
                return false;
            }
            return true;
        case json::value_t::string: {
            const auto& s = v.get_ref<const std::string&>();
            if (!valid_utf8(s)) {
                sink.error("spec.bad_value", where + " is not valid UTF-8.", path);
                return false;
            }
            if (s.size() > limits::kStringChars && code_points(s) > limits::kStringChars) {
                sink.error("spec.string_too_long",
                           where + " is longer than " + std::to_string(limits::kStringChars) + " characters.", path);
                return false;
            }
            return true;
        }
        case json::value_t::binary:
        case json::value_t::discarded:
            sink.error("spec.wrong_type", where + " is not a value a spec can hold.", path);
            return false;
        case json::value_t::array: {
            bool ok = true;
            std::size_t i = 0;
            for (const json& e : v) {
                if (!check_values(e, index_path(path, i++), depth + 1, count, sink, allow_null)) {
                    ok = false;
                    if (count > limits::kValues || sink.full()) return false;
                }
            }
            return ok;
        }
        case json::value_t::object: {
            bool ok = true;
            for (auto it = v.begin(); it != v.end(); ++it) {
                const std::string& key = it.key();
                const std::string child = key_path(path, key);
                if (!valid_utf8(key)) {
                    sink.error("spec.bad_value", "A key in " + where + " is not valid UTF-8.", path);
                    ok = false;
                    continue;
                }
                if (key.size() > limits::kStringChars && code_points(key) > limits::kStringChars) {
                    sink.error("spec.string_too_long", "A key in " + where + " is too long.", path);
                    ok = false;
                    continue;
                }
                if (!check_values(it.value(), child, depth + 1, count, sink, allow_null)) {
                    ok = false;
                    if (count > limits::kValues || sink.full()) return false;
                }
            }
            return ok;
        }
    }
    return true;
}

}  // namespace caliper_builder::spec::detail
