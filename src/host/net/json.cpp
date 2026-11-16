// json.cpp: the strict reader and the canonical writer of json.h.
#include "net/json.h"

#include <algorithm>
#include <charconv>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>

#if defined(_WIN32)
#  include <locale.h>
#elif defined(__APPLE__)
#  include <xlocale.h>
#else
#  include <locale.h>
#endif

namespace caliper_net {
namespace {

// ---- locale-independent number text (a host may have set LC_NUMERIC) ----------
//
// The writer takes its digits from std::to_chars, which is exact on every
// platform and depends on neither the locale nor the C runtime's printf.
// Apple's libc++ has it for double from macOS 13.3, the floor of every build
// that compiles this file (a lower deployment target fails to compile with
// that version named). The reader still parses with strtod in the C locale:
// std::from_chars for double would free it from the C runtime as well, but
// Apple's libc++ offers it only from macOS 26.0, and the macOS builds target
// earlier versions. It is the planned replacement once the macOS floor allows
// it.

#if defined(_WIN32)
_locale_t c_locale() {
    static _locale_t l = _create_locale(LC_NUMERIC, "C");
    return l;
}
double strtod_c(const char* s, char** end) { return _strtod_l(s, end, c_locale()); }
#else
locale_t c_locale() {
    static locale_t l = newlocale(LC_NUMERIC_MASK, "C", (locale_t)0);
    return l;
}
double strtod_c(const char* s, char** end) { return strtod_l(s, end, c_locale()); }
#endif

// The shortest digits that read back to v (finite, non-zero, positive), and
// the decimal exponent of the first digit: std::to_chars's shortest
// scientific form, the shortest digits that round-trip and, of several, the
// nearest to v, which is the choice of Python's repr().
void shortest_digits(double v, std::string* digits, int* exp10) {
    char buf[64];
    const std::to_chars_result r = std::to_chars(buf, buf + sizeof buf, v, std::chars_format::scientific);
    // "d[.ddd]e[+-]XX"
    const char* e = std::find(buf, r.ptr, 'e');
    digits->clear();
    for (const char* q = buf; q < e; ++q)
        if (*q >= '0' && *q <= '9') digits->push_back(*q);
    const char* x = e + 1;
    if (x < r.ptr && *x == '+') ++x;
    *exp10 = 0;
    std::from_chars(x, r.ptr, *exp10);
    while (digits->size() > 1 && digits->back() == '0') digits->pop_back();
}

// ---- the reader ---------------------------------------------------------------------

struct Parser {
    const char* p;
    const char* end;
    unsigned max_depth;
    std::string err;

    bool fail(const std::string& why) {
        if (err.empty()) err = why;
        return false;
    }

    void skip_ws() {
        while (p < end && (*p == ' ' || *p == '\t' || *p == '\n' || *p == '\r')) ++p;
    }

    bool literal(const char* word, size_t n) {
        if ((size_t)(end - p) < n || std::memcmp(p, word, n) != 0) return fail("invalid literal");
        p += n;
        return true;
    }

    static int hex(char c) {
        if (c >= '0' && c <= '9') return c - '0';
        if (c >= 'a' && c <= 'f') return c - 'a' + 10;
        if (c >= 'A' && c <= 'F') return c - 'A' + 10;
        return -1;
    }

    bool hex4(unsigned* out) {
        if (end - p < 4) return fail("short \\u escape");
        unsigned v = 0;
        for (int i = 0; i < 4; ++i) {
            const int h = hex(p[i]);
            if (h < 0) return fail("bad \\u escape");
            v = (v << 4) | (unsigned)h;
        }
        p += 4;
        *out = v;
        return true;
    }

    static void put_utf8(std::string& s, unsigned cp) {
        if (cp < 0x80) {
            s.push_back((char)cp);
        } else if (cp < 0x800) {
            s.push_back((char)(0xc0 | (cp >> 6)));
            s.push_back((char)(0x80 | (cp & 0x3f)));
        } else if (cp < 0x10000) {
            s.push_back((char)(0xe0 | (cp >> 12)));
            s.push_back((char)(0x80 | ((cp >> 6) & 0x3f)));
            s.push_back((char)(0x80 | (cp & 0x3f)));
        } else {
            s.push_back((char)(0xf0 | (cp >> 18)));
            s.push_back((char)(0x80 | ((cp >> 12) & 0x3f)));
            s.push_back((char)(0x80 | ((cp >> 6) & 0x3f)));
            s.push_back((char)(0x80 | (cp & 0x3f)));
        }
    }

    bool string(std::string* out) {
        ++p;   // the opening quote
        out->clear();
        for (;;) {
            if (p >= end) return fail("unterminated string");
            const unsigned char c = (unsigned char)*p;
            if (c == '"') {
                ++p;
                return true;
            }
            if (c < 0x20) return fail("control character in a string");
            if (c != '\\') {
                const char* q = p;
                while (q < end && *q != '"' && *q != '\\' && (unsigned char)*q >= 0x20) ++q;
                out->append(p, q);
                p = q;
                continue;
            }
            if (++p >= end) return fail("unterminated escape");
            const char e = *p++;
            switch (e) {
                case '"': out->push_back('"'); break;
                case '\\': out->push_back('\\'); break;
                case '/': out->push_back('/'); break;
                case 'b': out->push_back('\b'); break;
                case 'f': out->push_back('\f'); break;
                case 'n': out->push_back('\n'); break;
                case 'r': out->push_back('\r'); break;
                case 't': out->push_back('\t'); break;
                case 'u': {
                    unsigned cp;
                    if (!hex4(&cp)) return false;
                    if (cp >= 0xd800 && cp <= 0xdbff && end - p >= 6 && p[0] == '\\' && p[1] == 'u') {
                        const char* save = p;
                        p += 2;
                        unsigned lo;
                        if (!hex4(&lo)) return false;
                        if (lo >= 0xdc00 && lo <= 0xdfff) {
                            cp = 0x10000 + ((cp - 0xd800) << 10) + (lo - 0xdc00);
                        } else {
                            p = save;      // a lone high surrogate, then another escape
                            cp = 0xfffd;
                        }
                    } else if (cp >= 0xd800 && cp <= 0xdfff) {
                        cp = 0xfffd;       // a lone surrogate (accepted, as by Python)
                    }
                    put_utf8(*out, cp);
                    break;
                }
                default:
                    return fail("invalid escape");
            }
        }
    }

    bool number(Json* out) {
        const char* s = p;
        bool neg = false;
        if (*p == '-') {
            neg = true;
            ++p;
        }
        if (p >= end) return fail("bad number");
        if (*p == '0') {
            ++p;
        } else if (*p >= '1' && *p <= '9') {
            while (p < end && *p >= '0' && *p <= '9') ++p;
        } else {
            return fail("bad number");
        }
        bool is_int = true;
        if (p < end && *p == '.') {
            is_int = false;
            ++p;
            if (p >= end || !(*p >= '0' && *p <= '9')) return fail("bad number fraction");
            while (p < end && *p >= '0' && *p <= '9') ++p;
        }
        if (p < end && (*p == 'e' || *p == 'E')) {
            is_int = false;
            ++p;
            if (p < end && (*p == '+' || *p == '-')) ++p;
            if (p >= end || !(*p >= '0' && *p <= '9')) return fail("bad number exponent");
            while (p < end && *p >= '0' && *p <= '9') ++p;
        }
        const std::string text(s, p);
        if (is_int) {
            // Exact when it fits; out of range it becomes a double.
            uint64_t mag = 0;
            bool overflow = false;
            for (size_t i = neg ? 1 : 0; i < text.size(); ++i) {
                const uint64_t d = (uint64_t)(text[i] - '0');
                if (mag > (std::numeric_limits<uint64_t>::max() - d) / 10) {
                    overflow = true;
                    break;
                }
                mag = mag * 10 + d;
            }
            if (!overflow) {
                if (!neg) {
                    *out = Json(mag);
                    return true;
                }
                if (mag <= (uint64_t)std::numeric_limits<int64_t>::max() + 1) {
                    *out = Json(mag == (uint64_t)std::numeric_limits<int64_t>::max() + 1
                                    ? std::numeric_limits<int64_t>::min()
                                    : -(int64_t)mag);
                    return true;
                }
            }
        }
        *out = Json(strtod_c(text.c_str(), nullptr));
        return true;
    }

    bool value(Json* out, unsigned depth) {
        skip_ws();
        if (p >= end) return fail("unexpected end of input");
        switch (*p) {
            case '{': {
                if (depth + 1 > max_depth) return fail("nested deeper than " + std::to_string(max_depth));
                ++p;
                Json o = Json::object();
                skip_ws();
                if (p < end && *p == '}') {
                    ++p;
                    *out = std::move(o);
                    return true;
                }
                for (;;) {
                    skip_ws();
                    if (p >= end || *p != '"') return fail("expected a member name");
                    std::string key;
                    if (!string(&key)) return false;
                    skip_ws();
                    if (p >= end || *p != ':') return fail("expected ':'");
                    ++p;
                    Json v;
                    if (!value(&v, depth + 1)) return false;
                    o.set(key, std::move(v));
                    skip_ws();
                    if (p < end && *p == ',') { ++p; continue; }
                    if (p < end && *p == '}') { ++p; break; }
                    return fail("expected ',' or '}'");
                }
                *out = std::move(o);
                return true;
            }
            case '[': {
                if (depth + 1 > max_depth) return fail("nested deeper than " + std::to_string(max_depth));
                ++p;
                Json a = Json::array();
                skip_ws();
                if (p < end && *p == ']') {
                    ++p;
                    *out = std::move(a);
                    return true;
                }
                for (;;) {
                    Json v;
                    if (!value(&v, depth + 1)) return false;
                    a.push(std::move(v));
                    skip_ws();
                    if (p < end && *p == ',') { ++p; continue; }
                    if (p < end && *p == ']') { ++p; break; }
                    return fail("expected ',' or ']'");
                }
                *out = std::move(a);
                return true;
            }
            case '"': {
                std::string s;
                if (!string(&s)) return false;
                *out = Json(std::move(s));
                return true;
            }
            case 't': if (!literal("true", 4)) return false; *out = Json(true); return true;
            case 'f': if (!literal("false", 5)) return false; *out = Json(false); return true;
            case 'n': if (!literal("null", 4)) return false; *out = Json(); return true;
            default:
                if (*p == '-' || (*p >= '0' && *p <= '9')) return number(out);
                return fail("unexpected character");
        }
    }
};

}  // namespace

// ---- Json -------------------------------------------------------------------------------

void Json::set_unsigned(uint64_t v) {
    if (v <= (uint64_t)std::numeric_limits<int64_t>::max()) {
        type_ = Type::Int;
        i_ = (int64_t)v;
    } else {
        type_ = Type::UInt;
        u_ = v;
    }
}

double Json::as_double() const {
    switch (type_) {
        case Type::Int: return (double)i_;
        case Type::UInt: return (double)u_;
        case Type::Double: return d_;
        default: return 0.0;
    }
}

bool Json::get_i64(int64_t* out) const {
    if (type_ != Type::Int) return false;
    *out = i_;
    return true;
}

bool Json::get_u64(uint64_t* out) const {
    if (type_ == Type::Int && i_ >= 0) {
        *out = (uint64_t)i_;
        return true;
    }
    if (type_ == Type::UInt) {
        *out = u_;
        return true;
    }
    return false;
}

const Json* Json::find(const std::string& key) const {
    if (type_ != Type::Object) return nullptr;
    for (const auto& m : o_)
        if (m.first == key) return &m.second;
    return nullptr;
}

Json& Json::set(const std::string& key, Json v) {
    if (type_ != Type::Object) *this = object();
    for (auto& m : o_)
        if (m.first == key) {
            m.second = std::move(v);
            return m.second;
        }
    o_.emplace_back(key, std::move(v));
    return o_.back().second;
}

void Json::erase(const std::string& key) {
    o_.erase(std::remove_if(o_.begin(), o_.end(), [&](const Member& m) { return m.first == key; }),
             o_.end());
}

bool Json::operator==(const Json& o) const {
    if (type_ != o.type_) return false;
    switch (type_) {
        case Type::Null: return true;
        case Type::Bool: return b_ == o.b_;
        case Type::Int: return i_ == o.i_;
        case Type::UInt: return u_ == o.u_;
        case Type::Double: return d_ == o.d_ || (std::isnan(d_) && std::isnan(o.d_));
        case Type::String: return s_ == o.s_;
        case Type::Array: return a_ == o.a_;
        case Type::Object: {
            if (o_.size() != o.o_.size()) return false;
            for (const auto& m : o_) {
                const Json* v = o.find(m.first);
                if (!v || *v != m.second) return false;
            }
            return true;
        }
    }
    return false;
}

void json_quote(std::string& out, const std::string& s) {
    static const char* hex = "0123456789abcdef";
    out.push_back('"');
    for (unsigned char c : s) {
        switch (c) {
            case '"': out += "\\\""; break;
            case '\\': out += "\\\\"; break;
            case '\b': out += "\\b"; break;
            case '\f': out += "\\f"; break;
            case '\n': out += "\\n"; break;
            case '\r': out += "\\r"; break;
            case '\t': out += "\\t"; break;
            default:
                if (c < 0x20) {
                    out += "\\u00";
                    out.push_back(hex[c >> 4]);
                    out.push_back(hex[c & 15]);
                } else {
                    out.push_back((char)c);
                }
        }
    }
    out.push_back('"');
}

std::string format_double(double v) {
    if (!std::isfinite(v)) return "null";
    if (v == 0.0) return std::signbit(v) ? "-0.0" : "0.0";
    const bool neg = v < 0;
    std::string d;
    int x = 0;
    shortest_digits(std::fabs(v), &d, &x);
    const int decpt = x + 1;   // value = 0.d1d2... * 10^decpt
    std::string out = neg ? "-" : "";
    if (decpt <= -4 || decpt > 16) {
        out += d.substr(0, 1);
        if (d.size() > 1) out += "." + d.substr(1);
        char e[16];
        std::snprintf(e, sizeof e, "e%c%02d", x < 0 ? '-' : '+', x < 0 ? -x : x);
        out += e;
    } else if (decpt <= 0) {
        out += "0." + std::string((size_t)-decpt, '0') + d;
    } else if ((size_t)decpt < d.size()) {
        out += d.substr(0, (size_t)decpt) + "." + d.substr((size_t)decpt);
    } else {
        out += d + std::string((size_t)decpt - d.size(), '0') + ".0";
    }
    return out;
}

void Json::dump_to(std::string& out) const {
    switch (type_) {
        case Type::Null: out += "null"; break;
        case Type::Bool: out += b_ ? "true" : "false"; break;
        case Type::Int: out += std::to_string(i_); break;
        case Type::UInt: out += std::to_string(u_); break;
        case Type::Double: out += format_double(d_); break;
        case Type::String: json_quote(out, s_); break;
        case Type::Array: {
            out.push_back('[');
            for (size_t i = 0; i < a_.size(); ++i) {
                if (i) out.push_back(',');
                a_[i].dump_to(out);
            }
            out.push_back(']');
            break;
        }
        case Type::Object: {
            std::vector<const Member*> m;
            m.reserve(o_.size());
            for (const auto& x : o_) m.push_back(&x);
            std::sort(m.begin(), m.end(), [](const Member* a, const Member* b) { return a->first < b->first; });
            out.push_back('{');
            for (size_t i = 0; i < m.size(); ++i) {
                if (i) out.push_back(',');
                json_quote(out, m[i]->first);
                out.push_back(':');
                m[i]->second.dump_to(out);
            }
            out.push_back('}');
            break;
        }
    }
}

std::string Json::dump() const {
    std::string s;
    dump_to(s);
    return s;
}

bool valid_utf8(const char* s, size_t n) {
    const unsigned char* p = (const unsigned char*)s;
    const unsigned char* e = p + n;
    while (p < e) {
        const unsigned char c = *p;
        if (c < 0x80) { ++p; continue; }
        size_t len;
        unsigned cp;
        if (c >= 0xc2 && c <= 0xdf) { len = 2; cp = c & 0x1f; }
        else if (c >= 0xe0 && c <= 0xef) { len = 3; cp = c & 0x0f; }
        else if (c >= 0xf0 && c <= 0xf4) { len = 4; cp = c & 0x07; }
        else return false;
        if ((size_t)(e - p) < len) return false;
        for (size_t i = 1; i < len; ++i) {
            if ((p[i] & 0xc0) != 0x80) return false;
            cp = (cp << 6) | (p[i] & 0x3f);
        }
        if ((len == 3 && cp < 0x800) || (len == 4 && (cp < 0x10000 || cp > 0x10ffff))) return false;
        if (cp >= 0xd800 && cp <= 0xdfff) return false;
        p += len;
    }
    return true;
}

bool Json::parse(const char* p, size_t n, Json* out, std::string* err, unsigned max_depth) {
    if (!valid_utf8(p, n)) {
        if (err) *err = "not valid UTF-8";
        return false;
    }
    if (n >= 3 && (unsigned char)p[0] == 0xef && (unsigned char)p[1] == 0xbb && (unsigned char)p[2] == 0xbf) {
        if (err) *err = "a byte order mark";
        return false;
    }
    Parser ps{p, p + n, max_depth, {}};
    Json v;
    if (!ps.value(&v, 0)) {
        if (err) *err = ps.err;
        return false;
    }
    ps.skip_ws();
    if (ps.p != ps.end) {
        if (err) *err = "extra data after the JSON value";
        return false;
    }
    *out = std::move(v);
    return true;
}

}  // namespace caliper_net
