// The format language of bound texts (caliper_builder/text/bound_text.h,
// design/instrument-interface.md, Bound Text).
//
// The language is small on purpose: literal text, doubled braces, and fields
// that name a key of the fields table with an optional number format. It has
// no expressions, no attribute or index access, no nested fields and no
// width or alignment, so that a format cannot run code or grow without bound.
//
// The parser refuses everything outside the grammar with spec.bad_format and
// a message that names the character position. The formatter never fails:
// a null value gives a dash, a value that the format does not fit gives "?",
// and an array of more than one element gives its shape. A value may also be
// the object {"shape": [..]}, which callers use for a tensor that they do not
// copy into JSON; it formats as that shape.
#include <caliper_builder/text/bound_text.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>

namespace caliper_builder::text {

namespace {

using nlohmann::json;

constexpr std::size_t kMaxFormatChars = 1024;   // limits::kFormatChars
constexpr std::size_t kMaxFields = 16;          // limits::kFormatFields
constexpr std::size_t kMaxOutputChars = 1024;
constexpr int kMaxPrecision = 20;
constexpr const char* kDash = "\xE2\x80\x93";        // U+2013, the value is missing
constexpr const char* kEllipsis = "\xE2\x80\xA6";    // U+2026, the output was cut

// The number of code points of a UTF-8 text; continuation bytes do not count.
std::size_t code_points(std::string_view s) {
    std::size_t n = 0;
    for (unsigned char c : s)
        if ((c & 0xC0) != 0x80) ++n;
    return n;
}

Error bad(const std::string& message) { return Error("spec.bad_format", message); }

std::string at(std::size_t pos) { return " at character " + std::to_string(pos + 1); }

// Parses the format part after ':' of one field into `f`.
Error parse_spec(std::string_view spec, std::size_t pos, FieldFormat& f) {
    if (spec.empty()) return bad("The field " + f.name + " has a colon but no format" + at(pos) + ".");
    std::size_t i = 0;
    const auto digits = [&](int& out) -> bool {
        const std::size_t start = i;
        long v = 0;
        while (i < spec.size() && spec[i] >= '0' && spec[i] <= '9' && i - start < 3) v = v * 10 + (spec[i++] - '0');
        if (i == start || (i < spec.size() && spec[i] >= '0' && spec[i] <= '9')) return false;
        out = (int)v;
        return v <= kMaxPrecision;
    };
    if (spec.size() >= 2 && spec[0] == 's' && spec[1] == 'i') {
        f.type = 'k';
        i = 2;
        if (i < spec.size() && spec[i] == '.') {
            ++i;
            if (!digits(f.precision) || f.precision < 1)
                return bad("The format si takes from 1 to 20 significant digits" + at(pos) + ".");
        }
        if (i != spec.size()) return bad("The format " + std::string(spec) + " is not one of the formats" + at(pos) + ".");
        return {};
    }
    if (spec[i] == ',') {
        f.grouping = true;
        ++i;
    }
    if (i < spec.size() && spec[i] == '.') {
        ++i;
        if (!digits(f.precision)) return bad("A precision is from 0 to 20 digits" + at(pos) + ".");
    }
    if (i + 1 != spec.size())
        return bad("The format " + std::string(spec) +
                   " is not one of the formats: d, f, e, g, %, s or si, with an optional comma and precision" + at(pos) +
                   ".");
    const char t = spec[i];
    if (t != 'd' && t != 'f' && t != 'e' && t != 'g' && t != '%' && t != 's')
        return bad(std::string("The format type ") + t + " is not one of d, f, e, g, %, s and si" + at(pos) + ".");
    if (t == 'd' && f.precision >= 0) return bad("The format d takes no precision" + at(pos) + ".");
    if (t == 's' && (f.precision >= 0 || f.grouping)) return bad("The format s takes no comma and no precision" + at(pos) + ".");
    f.type = t;
    return {};
}

// printf into a string, for one number.
std::string printf_number(const char* fmt, int precision, double v) {
    char small[64];
    const int n = std::snprintf(small, sizeof small, fmt, precision, v);
    if (n < 0) return "?";
    if ((std::size_t)n < sizeof small) return std::string(small, (std::size_t)n);
    std::string big((std::size_t)n + 1, '\0');
    std::snprintf(big.data(), big.size(), fmt, precision, v);
    big.resize((std::size_t)n);
    return big;
}

// Commas between the groups of three digits of the integer part.
std::string group_digits(const std::string& s) {
    std::size_t start = (!s.empty() && (s[0] == '-' || s[0] == '+')) ? 1 : 0;
    std::size_t end = start;
    while (end < s.size() && s[end] >= '0' && s[end] <= '9') ++end;
    if (end - start <= 3) return s;
    std::string out = s.substr(0, start);
    const std::size_t len = end - start;
    for (std::size_t k = 0; k < len; ++k) {
        if (k > 0 && (len - k) % 3 == 0) out += ',';
        out += s[start + k];
    }
    out += s.substr(end);
    return out;
}

std::string non_finite(double v) { return std::isnan(v) ? "nan" : (v < 0 ? "-inf" : "inf"); }

// The SI form with `sig` significant digits: 12345.678 with 3 gives "12.3k".
std::string si_number(double v, int sig) {
    if (v == 0.0) return "0";
    static const char* const prefixes[] = {"y", "z", "a", "f", "p", "n", "\xC2\xB5", "m", "",
                                           "k", "M", "G", "T", "P", "E", "Z", "Y"};
    const double a = std::fabs(v);
    int e3 = (int)std::floor(std::log10(a) / 3.0) * 3;
    for (int pass = 0; pass < 2; ++pass) {
        e3 = std::clamp(e3, -24, 24);
        const double scaled = a / std::pow(10.0, e3);
        const int lead = scaled >= 1.0 ? (int)std::floor(std::log10(scaled)) + 1 : 1;
        const int decimals = std::max(0, sig - lead);
        std::string text = printf_number("%.*f", decimals, scaled);
        // Rounding can carry to 1000: write it with the next prefix.
        if (pass == 0 && e3 < 24 && std::strtod(text.c_str(), nullptr) >= 1000.0) {
            e3 += 3;
            continue;
        }
        return (v < 0 ? "-" : "") + text + prefixes[(e3 + 24) / 3];
    }
    return "?";
}

// The plain text of a number for the format s: an integer without a point,
// else up to 6 significant digits.
std::string plain_number(double v) {
    if (!std::isfinite(v)) return non_finite(v);
    if (v == std::floor(v) && std::fabs(v) < 1e15) return printf_number("%.*f", 0, v == 0.0 ? 0.0 : v);
    return printf_number("%.*g", 6, v);
}

std::string format_number(const FieldFormat& f, double v) {
    if (!std::isfinite(v)) return non_finite(v);
    std::string s;
    switch (f.type) {
        case 'd': {
            const double r = std::round(v);
            s = printf_number("%.*f", 0, r == 0.0 ? 0.0 : r);
            break;
        }
        case 'f': s = printf_number("%.*f", f.precision < 0 ? 6 : f.precision, v); break;
        case 'e': s = printf_number("%.*e", f.precision < 0 ? 6 : f.precision, v); break;
        case 'g': s = printf_number("%.*g", f.precision < 0 ? 6 : f.precision, v); break;
        case '%': s = printf_number("%.*f", f.precision < 0 ? 6 : f.precision, v * 100.0) + "%"; break;
        case 'k': return si_number(v, f.precision < 0 ? 3 : f.precision);
        default: return plain_number(v);
    }
    if (f.grouping && s.find('e') == std::string::npos) s = group_digits(s);
    return s;
}

// The shape of a nested JSON array, "[2, 3]", from its first elements.
std::string shape_text(const json& shape) {
    std::string out = "[";
    for (std::size_t i = 0; i < shape.size(); ++i) {
        if (i) out += ", ";
        out += shape[i].is_number_integer() ? std::to_string(shape[i].get<std::int64_t>()) : "?";
    }
    return out + "]";
}

json array_shape(const json& v) {
    json shape = json::array();
    const json* at = &v;
    for (int depth = 0; at->is_array() && depth < 8; ++depth) {
        shape.push_back((std::int64_t)at->size());
        if (at->empty()) break;
        at = &(*at)[0];
    }
    return shape;
}

std::string format_value(const FieldFormat& f, const json& v) {
    if (v.is_null()) return kDash;
    if (v.is_array()) {
        if (v.size() == 1) return format_value(f, v[0]);
        return shape_text(array_shape(v));
    }
    if (v.is_object()) {
        if (auto it = v.find("shape"); it != v.end() && it->is_array()) return shape_text(*it);
        return "?";
    }
    if (v.is_number()) return f.type == 's' ? plain_number(v.get<double>()) : format_number(f, v.get<double>());
    if (f.type != 's') return "?";
    if (v.is_string()) return v.get<std::string>();
    if (v.is_boolean()) return v.get<bool>() ? "true" : "false";
    return "?";
}

// Cuts `s` to at most kMaxOutputChars code points, the last an ellipsis.
std::string limit_output(std::string s) {
    if (code_points(s) <= kMaxOutputChars) return s;
    std::size_t kept = 0, i = 0;
    for (; i < s.size(); ++i) {
        if (((unsigned char)s[i] & 0xC0) != 0x80) {
            if (kept == kMaxOutputChars - 1) break;
            ++kept;
        }
    }
    s.resize(i);
    return s + kEllipsis;
}

}  // namespace

Error parse_bound_text(std::string_view format, const std::vector<std::string>& field_names, ParsedText* out) {
    *out = {};
    if (code_points(format) > kMaxFormatChars)
        return bad("A format has at most 1,024 characters, and this one has " + std::to_string(code_points(format)) + ".");
    ParsedText t;
    std::string literal;
    const auto flush = [&] {
        if (literal.empty()) return;
        t.parts.push_back(std::move(literal));
        t.fields_at.push_back(-1);
        literal.clear();
    };
    for (std::size_t i = 0; i < format.size();) {
        const char c = format[i];
        if (c == '}') {
            if (i + 1 < format.size() && format[i + 1] == '}') {
                literal += '}';
                i += 2;
                continue;
            }
            return bad("A closing brace has no opening brace" + at(i) + "; write }} for a brace in the text.");
        }
        if (c != '{') {
            literal += c;
            ++i;
            continue;
        }
        if (i + 1 < format.size() && format[i + 1] == '{') {
            literal += '{';
            i += 2;
            continue;
        }
        const std::size_t open = i;
        const std::size_t close = format.find_first_of("{}", i + 1);
        if (close == std::string_view::npos || format[close] == '{')
            return bad(close == std::string_view::npos
                           ? "A brace opens a field but does not close" + at(open) + "; write {{ for a brace in the text."
                           : "A field holds another brace" + at(close) + "; fields do not nest.");
        const std::string_view body = format.substr(open + 1, close - open - 1);
        const std::size_t colon = body.find(':');
        FieldFormat f;
        f.name = std::string(body.substr(0, colon));
        if (f.name.empty()) return bad("A field has no name" + at(open) + ".");
        if (std::find(field_names.begin(), field_names.end(), f.name) == field_names.end())
            return bad("The field " + f.name + at(open) + " is not a key of the fields table.");
        if (colon != std::string_view::npos)
            if (Error e = parse_spec(body.substr(colon + 1), open + 1 + colon + 1, f)) return e;
        if (t.fields.size() == kMaxFields) return bad("A format has at most 16 fields" + at(open) + ".");
        flush();
        t.parts.emplace_back();
        t.fields_at.push_back((int)t.fields.size());
        t.fields.push_back(std::move(f));
        i = close + 1;
    }
    flush();
    *out = std::move(t);
    return {};
}

std::string format_bound_text(const ParsedText& text, const std::map<std::string, nlohmann::json>& values) {
    std::string out;
    const json null_value;
    for (std::size_t i = 0; i < text.parts.size() && i < text.fields_at.size(); ++i) {
        const int k = text.fields_at[i];
        if (k < 0 || (std::size_t)k >= text.fields.size()) {
            out += text.parts[i];
        } else {
            const FieldFormat& f = text.fields[(std::size_t)k];
            const auto it = values.find(f.name);
            out += format_value(f, it == values.end() ? null_value : it->second);
        }
        if (out.size() > 8 * kMaxOutputChars) break;   // the cut below keeps 1,024 code points
    }
    return limit_output(std::move(out));
}

}  // namespace caliper_builder::text
