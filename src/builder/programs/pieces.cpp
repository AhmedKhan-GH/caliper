// Names, files, imports, the canonical manifest, the hash, the source check
// and the places of composed lines of pieces (programs/pieces.h,
// design/programmable-graphics.md: Piece Files, Imports, Piece Hash and Check
// Order 1, 4 and 5). This file needs only the standard library and
// spec/piece.h, because the browser module compiles it too.
#include <caliper_builder/programs/pieces.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <set>
#include <utility>

namespace caliper_builder::programs {

namespace {

Verdict fail(const char* code, std::string message, std::string where = {}, std::uint32_t line = 0,
             std::uint32_t column = 0) {
    Verdict v;
    v.code = code;
    v.message = std::move(message);
    v.where = std::move(where);
    v.line = line;
    v.column = column;
    return v;
}

std::string quote(std::string_view s) { return "\"" + std::string(s) + "\""; }

bool lower(char c) { return c >= 'a' && c <= 'z'; }
bool digit(char c) { return c >= '0' && c <= '9'; }
bool ident_start(char c) { return (c >= 'A' && c <= 'Z') || lower(c) || c == '_'; }
bool ident_char(char c) { return ident_start(c) || digit(c); }

bool starts_with(std::string_view s, std::string_view prefix) { return s.substr(0, prefix.size()) == prefix; }

// ---- lexing: WGSL without comments

// One identifier or one punctuation character of WGSL, with its place.
struct Token {
    std::string_view text;
    std::uint32_t line = 0;
    std::uint32_t column = 0;
};

// Walks `src` outside comments: calls `at(i, line, column)` for each byte
// that is not in a comment, where the caller may skip ahead by returning the
// count of bytes it consumed (at least 1). A block comment nests, as WGSL
// says.
template <typename F>
void walk(std::string_view src, F at) {
    std::uint32_t line = 1;
    std::uint32_t column = 1;
    std::size_t i = 0;
    auto advance = [&](std::size_t n) {
        for (std::size_t k = 0; k < n && i < src.size(); ++k, ++i) {
            if (src[i] == '\n') {
                ++line;
                column = 1;
            } else {
                ++column;
            }
        }
    };
    while (i < src.size()) {
        if (src.compare(i, 2, "//") == 0) {
            while (i < src.size() && src[i] != '\n') advance(1);
            continue;
        }
        if (src.compare(i, 2, "/*") == 0) {
            int depth = 0;
            do {
                if (src.compare(i, 2, "/*") == 0) {
                    ++depth;
                    advance(2);
                } else if (src.compare(i, 2, "*/") == 0) {
                    --depth;
                    advance(2);
                } else {
                    advance(1);
                }
            } while (depth > 0 && i < src.size());
            continue;
        }
        const std::size_t n = at(i, line, column);
        advance(n == 0 ? 1 : n);
    }
}

// Splits WGSL into identifiers and single punctuation characters, without
// comments and white space. Numbers come out as identifier-like runs. Bytes
// above 0x7F join no token.
std::vector<Token> tokens(std::string_view src) {
    std::vector<Token> out;
    walk(src, [&](std::size_t i, std::uint32_t line, std::uint32_t column) -> std::size_t {
        const char c = src[i];
        if (static_cast<unsigned char>(c) <= 0x20 || static_cast<unsigned char>(c) >= 0x7F) return 1;
        std::size_t end = i + 1;
        if (ident_char(c))
            while (end < src.size() && ident_char(src[end])) ++end;
        out.push_back(Token{src.substr(i, end - i), line, column});
        return end - i;
    });
    return out;
}

// The count of lines of a text that ends with a line end.
std::uint32_t line_count(std::string_view text) {
    std::uint32_t n = 0;
    for (char c : text)
        if (c == '\n') ++n;
    return n;
}

// The text with LF line ends: CR LF and a lone CR become LF.
std::string lf(std::string_view text) {
    std::string out;
    out.reserve(text.size());
    for (std::size_t i = 0; i < text.size(); ++i) {
        if (text[i] == '\r') {
            out += '\n';
            if (i + 1 < text.size() && text[i + 1] == '\n') ++i;
        } else {
            out += text[i];
        }
    }
    return out;
}

// A module of `imports`: "<name>@<version>", the name in the form of a piece
// name and the version from 1 to 999999 without a leading zero.
bool split_import(std::string_view import, std::string* name) {
    const std::size_t at = import.find('@');
    if (at == std::string_view::npos || at == 0 || at > spec::piece_limits::kNameMax) return false;
    const std::string_view n = import.substr(0, at);
    const std::string_view v = import.substr(at + 1);
    if (!lower(n[0])) return false;
    for (char c : n)
        if (!lower(c) && !digit(c) && c != '_') return false;
    if (v.empty() || v.size() > 6 || v[0] == '0') return false;
    for (char c : v)
        if (!digit(c)) return false;
    if (name) *name = std::string(n);
    return true;
}

// The items of a lib module: the names after <module>__ of its identifiers.
std::set<std::string> module_items(std::string_view text, const std::string& module) {
    const std::string prefix = module + "__";
    std::set<std::string> items;
    for (const Token& t : tokens(text))
        if (t.text.size() > prefix.size() && starts_with(t.text, prefix)) items.emplace(t.text.substr(prefix.size()));
    return items;
}

// ---- SHA-256 (FIPS 180-4), as sandbox.cpp computes it, without its dependencies

struct Sha256 {
    std::uint32_t h[8] = {0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
                          0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19};
    unsigned char block[64] = {};
    std::size_t used = 0;
    std::uint64_t bits = 0;

    static std::uint32_t rotr(std::uint32_t x, int n) { return (x >> n) | (x << (32 - n)); }

    void compress() {
        static const std::uint32_t k[64] = {
            0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
            0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
            0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
            0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
            0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
            0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
            0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
            0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2};
        std::uint32_t w[64];
        for (int i = 0; i < 16; ++i)
            w[i] = (std::uint32_t(block[4 * i]) << 24) | (std::uint32_t(block[4 * i + 1]) << 16) |
                   (std::uint32_t(block[4 * i + 2]) << 8) | std::uint32_t(block[4 * i + 3]);
        for (int i = 16; i < 64; ++i) {
            const std::uint32_t s0 = rotr(w[i - 15], 7) ^ rotr(w[i - 15], 18) ^ (w[i - 15] >> 3);
            const std::uint32_t s1 = rotr(w[i - 2], 17) ^ rotr(w[i - 2], 19) ^ (w[i - 2] >> 10);
            w[i] = w[i - 16] + s0 + w[i - 7] + s1;
        }
        std::uint32_t a = h[0], b = h[1], c = h[2], d = h[3], e = h[4], f = h[5], g = h[6], hh = h[7];
        for (int i = 0; i < 64; ++i) {
            const std::uint32_t t1 = hh + (rotr(e, 6) ^ rotr(e, 11) ^ rotr(e, 25)) + ((e & f) ^ (~e & g)) + k[i] + w[i];
            const std::uint32_t t2 = (rotr(a, 2) ^ rotr(a, 13) ^ rotr(a, 22)) + ((a & b) ^ (a & c) ^ (b & c));
            hh = g;
            g = f;
            f = e;
            e = d + t1;
            d = c;
            c = b;
            b = a;
            a = t1 + t2;
        }
        h[0] += a;
        h[1] += b;
        h[2] += c;
        h[3] += d;
        h[4] += e;
        h[5] += f;
        h[6] += g;
        h[7] += hh;
    }

    void update(std::string_view data) {
        for (char ch : data) {
            block[used++] = static_cast<unsigned char>(ch);
            if (used == 64) {
                compress();
                used = 0;
            }
        }
        bits += std::uint64_t(data.size()) * 8;
    }

    std::string hex() {
        const std::uint64_t total = bits;
        update(std::string_view("\x80", 1));
        while (used != 56) update(std::string_view("\0", 1));
        char len[8];
        for (int i = 0; i < 8; ++i) len[i] = static_cast<char>(static_cast<unsigned char>(total >> (56 - 8 * i)));
        update(std::string_view(len, 8));
        static const char* digits = "0123456789abcdef";
        std::string out;
        out.reserve(64);
        for (std::uint32_t v : h)
            for (int s = 28; s >= 0; s -= 4) out += digits[(v >> s) & 0xF];
        return out;
    }
};

// ---- RFC 8785 canonical JSON

// A string as JSON.stringify writes it: '"' and '\' escaped, the control
// characters as \b, \t, \n, \f, \r or \u00xx, every other byte as it is.
std::string json_string(std::string_view s) {
    static const char* hex = "0123456789abcdef";
    std::string out = "\"";
    for (char ch : s) {
        const auto c = static_cast<unsigned char>(ch);
        switch (c) {
            case '"': out += "\\\""; break;
            case '\\': out += "\\\\"; break;
            case '\b': out += "\\b"; break;
            case '\t': out += "\\t"; break;
            case '\n': out += "\\n"; break;
            case '\f': out += "\\f"; break;
            case '\r': out += "\\r"; break;
            default:
                if (c < 0x20) {
                    out += "\\u00";
                    out += hex[c >> 4];
                    out += hex[c & 0xF];
                } else {
                    out += ch;
                }
        }
    }
    return out + "\"";
}

// A number as ECMAScript's Number::toString writes it: the fewest significant
// digits that read back as the same double, in plain notation from 1e-6 to
// below 1e21 and in exponent notation outside. The digits come from printf,
// which rounds correctly, and the decimal separator of the locale is never read.
std::string json_number(double v) {
    if (!std::isfinite(v)) return "null";
    if (v == 0) return "0";
    char buf[48];
    for (int p = 0; p <= 17; ++p) {
        std::snprintf(buf, sizeof buf, "%.*e", p, v);
        if (std::strtod(buf, nullptr) == v) break;
    }
    std::string digits;
    bool negative = false;
    int exponent = 0;
    const char* c = buf;
    if (*c == '-') {
        negative = true;
        ++c;
    }
    for (; *c && *c != 'e' && *c != 'E'; ++c)
        if (digit(*c)) digits += *c;
    if (*c) exponent = std::atoi(c + 1);
    while (digits.size() > 1 && digits.back() == '0') digits.pop_back();
    const int k = static_cast<int>(digits.size());
    const int n = exponent + 1;  // the value is 0.<digits> times 10^n
    std::string out = negative ? "-" : "";
    if (k <= n && n <= 21) {
        out += digits + std::string(static_cast<std::size_t>(n - k), '0');
    } else if (0 < n && n <= 21) {
        out += digits.substr(0, static_cast<std::size_t>(n)) + "." + digits.substr(static_cast<std::size_t>(n));
    } else if (-6 < n && n <= 0) {
        out += "0." + std::string(static_cast<std::size_t>(-n), '0') + digits;
    } else {
        out += digits.substr(0, 1);
        if (k > 1) out += "." + digits.substr(1);
        out += "e";
        out += n - 1 >= 0 ? "+" : "-";
        out += std::to_string(n - 1 >= 0 ? n - 1 : 1 - n);
    }
    return out;
}

// An object of canonical JSON: its members by key in byte order, which is
// the order of UTF-16 code units for the ASCII keys of a manifest.
class Object {
public:
    Object& set(const std::string& key, std::string value) {
        members_[key] = std::move(value);
        return *this;
    }
    Object& str(const std::string& key, std::string_view value) { return set(key, json_string(value)); }
    Object& num(const std::string& key, double value) { return set(key, json_number(value)); }
    Object& uint(const std::string& key, std::uint64_t value) { return set(key, std::to_string(value)); }
    Object& flag(const std::string& key, bool value) { return set(key, value ? "true" : "false"); }
    std::string text() const {
        std::string out = "{";
        bool first = true;
        for (const auto& [k, v] : members_) {
            if (!first) out += ",";
            first = false;
            out += json_string(k) + ":" + v;
        }
        return out + "}";
    }

private:
    std::map<std::string, std::string> members_;
};

std::string array_of(const std::vector<std::string>& values) {
    std::string out = "[";
    for (std::size_t i = 0; i < values.size(); ++i) out += (i ? "," : "") + values[i];
    return out + "]";
}

std::string strings(const std::vector<std::string>& values) {
    std::vector<std::string> out;
    for (const std::string& v : values) out.push_back(json_string(v));
    return array_of(out);
}

std::string counts(const std::array<std::uint32_t, 3>& w) {
    return "[" + std::to_string(w[0]) + "," + std::to_string(w[1]) + "," + std::to_string(w[2]) + "]";
}

void layout_into(Object& o, const spec::Layout& l) {
    if (l.element) o.str("element", *l.element);
    if (l.type_name) o.str("struct", *l.type_name);
    if (!l.fields.empty()) {
        std::vector<std::string> fields;
        for (const spec::Field& f : l.fields) fields.push_back(f.name + ": " + f.type);
        o.set("fields", strings(fields));
    }
}

bool default_workgroups(const std::array<std::uint32_t, 3>& w) { return w[0] == 1 && w[1] == 1 && w[2] == 1; }

std::string pass_json(const spec::DrawPass& p) {
    Object o;
    o.str("kind", spec::pass_kind_name(p.kind));
    if (!p.entry.empty()) o.str("entry", p.entry);
    if (!default_workgroups(p.workgroups)) o.set("workgroups", counts(p.workgroups));
    if (!p.writes.empty()) o.set("writes", strings(p.writes));
    if (p.vertex) o.str("vertex", *p.vertex);
    if (!p.fragment.empty()) o.str("fragment", p.fragment);
    if (p.target != "host") o.str("target", p.target);
    if (!p.reads.empty()) o.set("reads", strings(p.reads));
    if (p.instances_of) {
        o.str("instances", *p.instances_of);
    } else if (p.instances != 1) {
        o.uint("instances", p.instances);
    }
    if (p.vertices) o.uint("vertices", *p.vertices);
    if (p.topology != spec::DrawTopology::triangle_list) o.str("topology", spec::draw_topology_name(p.topology));
    if (p.blend) o.str("blend", spec::draw_blend_name(*p.blend));
    if (p.depth) o.str("depth", spec::draw_depth_name(*p.depth));
    if (p.pick) o.flag("pick", true);
    if (p.indices) o.str("indices", *p.indices);
    return o.text();
}

}  // namespace

// ---- names and paths

Verdict check_piece_name(std::string_view name, std::uint32_t version) {
    bool ok = !name.empty() && name.size() <= spec::piece_limits::kNameMax && lower(name[0]);
    for (char c : name) ok = ok && (lower(c) || digit(c) || c == '_');
    if (!ok)
        return fail(kPieceBadName,
                    "The piece name " + quote(name.substr(0, 64)) +
                        " is not a name of a piece: a lowercase letter, then at most 31 lowercase letters, digits "
                        "and underscores.",
                    "name");
    if (version < 1 || version > spec::piece_limits::kVersionMax)
        return fail(kPieceBadName,
                    "The version " + std::to_string(version) + " of the piece " + quote(name) +
                        " is not from 1 to " + std::to_string(spec::piece_limits::kVersionMax) + ".",
                    "version");
    return {};
}

Verdict check_piece_file(const spec::Piece& piece, std::string_view relative_path) {
    if (relative_path == "piece.toml") return {};
    for (const char* f : spec::kPieceFiles)
        if (relative_path == f) return {};
    constexpr std::string_view kSample = "sample/";
    constexpr std::string_view kBin = ".bin";
    if (relative_path.size() > kSample.size() + kBin.size() && starts_with(relative_path, kSample) &&
        relative_path.substr(relative_path.size() - kBin.size()) == kBin) {
        const std::string input(
            relative_path.substr(kSample.size(), relative_path.size() - kSample.size() - kBin.size()));
        bool is_input = false;
        for (const spec::PieceInput& in : piece.inputs) is_input = is_input || in.name == input;
        if (is_input && piece.sample.count(input) && check_piece_name(input, 1).ok()) return {};
    }
    return fail(kPieceBadPath,
                "A piece holds only piece.toml, common.wgsl, derive.wgsl, draw.wgsl and sample/<input>.bin for an "
                "input that its sample names; " +
                    quote(relative_path.substr(0, 128)) + " is none of them.",
                "files");
}

// ---- imports and hash

Resolved resolve_imports(const spec::Piece& piece) {
    Resolved r;
    std::map<std::string, std::set<std::string>> items;  // by module name
    std::uint32_t line = 1;
    for (std::size_t i = 0; i < piece.imports.size(); ++i) {
        const std::string& import = piece.imports[i];
        const std::string where = "imports[" + std::to_string(i) + "]";
        std::string name;
        const std::string_view text = split_import(import, &name) ? lib_module(import) : std::string_view();
        if (text.empty()) {
            std::string known;
            for (std::string_view m : lib_modules()) known += (known.empty() ? "" : ", ") + std::string(m);
            r.verdict = fail(kPieceUnknownImport,
                             "There is no lib module " + quote(import.substr(0, 64)) + "; the lib modules are " +
                                 known + ".",
                             where);
            return r;
        }
        if (items.count(name)) {
            r.verdict = fail(kPieceBadManifest, "The piece imports the module " + name + " twice; import one version.",
                             where);
            return r;
        }
        items[name] = module_items(text, name);
        r.modules.push_back(import);
        std::string body(text);
        if (body.back() != '\n') body += '\n';
        r.lib_segments.push_back(Segment{"lib/" + import, line, line_count(body)});
        line += line_count(body);
        r.lib_text += body;
    }
    for (const auto& entry : piece.files) {
        const std::string& file = entry.first;
        const std::string& text = entry.second;
        std::string out = text;
        Verdict bad;
        walk(text, [&](std::size_t i, std::uint32_t l, std::uint32_t column) -> std::size_t {
            if (!bad.ok() || text.compare(i, 2, "::") != 0) return 1;
            std::size_t begin = i;
            while (begin > 0 && ident_char(text[begin - 1])) --begin;
            std::size_t end = i + 2;
            while (end < text.size() && ident_char(text[end])) ++end;
            const std::string module = text.substr(begin, i - begin);
            const std::string item = text.substr(i + 2, end - i - 2);
            const std::uint32_t at = column - static_cast<std::uint32_t>(i - begin);
            if (module.empty() || !ident_start(module[0]) || item.empty() || !ident_start(item[0])) {
                bad = fail(kPieceUnknownImport,
                           "A \"::\" joins an imported module and one of its items, as in colormap::magma.", file, l,
                           column);
            } else if (!items.count(module)) {
                bad = fail(kPieceUnknownImport,
                           module + "::" + item + " names the module " + module +
                               ", which the piece does not import; add it to imports, as \"" + module + "@1\".",
                           file, l, at);
            } else if (!items[module].count(item)) {
                std::string module_at;
                for (const std::string& m : r.modules)
                    if (starts_with(m, module + "@")) module_at = m;
                bad = fail(kPieceUnknownImport, "The lib module " + module_at + " has no item " + quote(item) + ".",
                           file, l, at + static_cast<std::uint32_t>(module.size() + 2));
            } else {
                out[i] = '_';
                out[i + 1] = '_';
            }
            return 2;
        });
        if (!bad.ok()) {
            r.verdict = bad;
            r.files.clear();
            return r;
        }
        r.files[file] = std::move(out);
    }
    return r;
}

std::string canonical_manifest(const spec::Piece& piece) {
    Object m;
    m.str("name", piece.name);
    m.uint("version", piece.version);
    if (!piece.description.empty()) m.str("description", piece.description);
    if (!piece.inputs.empty()) {
        std::vector<std::string> list;
        for (const spec::PieceInput& in : piece.inputs) {
            Object o;
            o.str("name", in.name).str("kind", spec::input_kind_name(in.kind));
            if (in.dtype != "f32") o.str("dtype", in.dtype);
            if (in.rank_min != 1) o.uint("rank_min", in.rank_min);
            if (in.rank_max != 8) o.uint("rank_max", in.rank_max);
            layout_into(o, in.layout);
            if (in.title) o.str("title", *in.title);
            if (in.optional) o.flag("optional", true);
            list.push_back(o.text());
        }
        m.set("inputs", array_of(list));
    }
    if (!piece.params.empty()) {
        std::vector<std::string> list;
        for (const spec::PieceParam& p : piece.params) {
            Object o;
            o.str("name", p.name);
            if (p.type != spec::ParamType::f32) o.str("type", spec::param_type_name(p.type));
            if (p.default_value != 0) o.num("default", p.default_value);
            if (p.min) o.num("min", *p.min);
            if (p.max) o.num("max", *p.max);
            if (p.title) o.str("title", *p.title);
            list.push_back(o.text());
        }
        m.set("params", array_of(list));
    }
    if (!piece.resources.empty()) {
        std::vector<std::string> list;
        for (const spec::Resource& r : piece.resources) {
            Object o;
            o.str("id", r.id).str("kind", spec::resource_kind_name(r.kind));
            layout_into(o, r.layout);
            if (r.count) o.uint("count", r.count);
            if (r.counted) o.flag("counted", true);
            if (r.width) o.uint("width", r.width);
            if (r.height) o.uint("height", r.height);
            if (!r.publish) o.flag("publish", false);
            if (r.clear) o.flag("clear", true);
            if (r.keep) o.uint("keep", *r.keep);
            list.push_back(o.text());
        }
        m.set("resources", array_of(list));
    }
    if (!piece.imports.empty()) m.set("imports", strings(piece.imports));
    if (!piece.depends.empty()) {
        std::vector<std::string> list;
        for (const spec::PieceDependency& d : piece.depends)
            list.push_back(Object().str("name", d.name).uint("version", d.version).text());
        m.set("depends", array_of(list));
    }
    if (piece.derive) {
        const spec::Derive& d = *piece.derive;
        std::vector<std::string> passes;
        for (const spec::DerivePass& p : d.passes) {
            Object o;
            o.str("entry", p.entry);
            if (!default_workgroups(p.workgroups)) o.set("workgroups", counts(p.workgroups));
            if (!p.writes.empty()) o.set("writes", strings(p.writes));
            passes.push_back(o.text());
        }
        Object o;
        o.set("passes", array_of(passes));
        if (d.time_ms) o.num("time_ms", *d.time_ms);
        if (d.memory_mb) o.uint("memory_mb", *d.memory_mb);
        if (d.every) o.uint("every", *d.every);
        m.set("derive", o.text());
    }
    if (piece.draw) {
        const spec::Draw& d = *piece.draw;
        Object o;
        if (d.camera != spec::CameraMode::orbit) o.str("camera", spec::camera_mode_name(d.camera));
        if (!d.buffers.empty()) {
            std::vector<std::string> list;
            for (const spec::DrawBuffer& b : d.buffers) {
                Object bo;
                bo.str("id", b.id);
                layout_into(bo, b.layout);
                bo.uint("count", b.count);
                if (b.reset != spec::BufferReset::install) bo.str("reset", spec::buffer_reset_name(b.reset));
                list.push_back(bo.text());
            }
            o.set("buffers", array_of(list));
        }
        if (!d.targets.empty()) {
            std::vector<std::string> list;
            for (const spec::DrawTarget& t : d.targets) {
                Object to;
                to.str("id", t.id);
                if (t.scale != 1.0f) to.num("scale", t.scale);
                if (t.clear != std::array<float, 4>{0, 0, 0, 0}) {
                    std::vector<std::string> c;
                    for (float v : t.clear) c.push_back(json_number(v));
                    to.set("clear", array_of(c));
                }
                list.push_back(to.text());
            }
            o.set("targets", array_of(list));
        }
        std::vector<std::string> passes;
        for (const spec::DrawPass& p : d.passes) passes.push_back(pass_json(p));
        o.set("passes", array_of(passes));
        if (d.pick) o.set("pick", Object().str("resource", d.pick->resource).set("fields", strings(d.pick->fields)).text());
        if (d.text) {
            Object t;
            t.str("anchors", d.text->anchors);
            if (d.text->position != "position") t.str("position", d.text->position);
            if (d.text->string != "string") t.str("string", d.text->string);
            if (d.text->priority) t.str("priority", *d.text->priority);
            t.set("strings", strings(d.text->strings));
            if (d.text->strings_from) t.str("strings_from", *d.text->strings_from);
            if (d.text->align != spec::TextAlign::placed) t.str("align", spec::text_align_name(d.text->align));
            o.set("text", t.text());
        }
        if (d.frame_ms) o.num("frame_ms", *d.frame_ms);
        if (d.memory_mb) o.uint("memory_mb", *d.memory_mb);
        m.set("draw", o.text());
    }
    if (!piece.sample.empty()) {
        std::vector<std::string> list;
        for (const auto& [input, s] : piece.sample) {
            std::vector<std::string> shape;
            for (std::uint32_t v : s.shape) shape.push_back(std::to_string(v));
            list.push_back(Object().str("input", input).set("shape", array_of(shape)).text());
        }
        m.set("sample", array_of(list));
    }
    return m.text();
}

std::string piece_hash(const spec::Piece& piece) {
    Sha256 h;
    h.update("caliper-piece/1\n");
    h.update(canonical_manifest(piece));
    h.update("\n");
    auto part = [&](const std::string& name, std::string_view text) {
        const std::string body = lf(text);
        h.update("--- " + name + " " + std::to_string(body.size()) + "\n");
        h.update(body);
    };
    for (const std::string& import : piece.imports) part("lib/" + import, lib_module(import));
    for (const char* f : spec::kPieceFiles)
        if (auto it = piece.files.find(f); it != piece.files.end()) part(f, it->second);
    return "sha256:" + h.hex();
}

// ---- check 4: one file

Verdict check_piece_source(std::string_view wgsl, std::size_t max_bytes) {
    if (wgsl.size() > max_bytes)
        return fail(kPieceBadPath,
                    "The file has " + std::to_string(wgsl.size()) + " bytes; a WGSL file of a piece holds at most " +
                        std::to_string(max_bytes) + " bytes.");
    const std::vector<Token> toks = tokens(wgsl);
    if (!toks.empty()) {
        const std::string_view first = toks.front().text;
        if (first == "enable" || first == "requires" || first == "diagnostic")
            return fail("program.directive_not_allowed",
                        "The file starts with the directive " + std::string(first) +
                            "; a piece may not enable extensions or change diagnostics.",
                        {}, toks.front().line, toks.front().column);
    }
    for (std::size_t i = 0; i < toks.size(); ++i) {
        const Token& t = toks[i];
        if (t.text == "@" && i + 1 < toks.size() && (toks[i + 1].text == "group" || toks[i + 1].text == "binding"))
            return fail(kProgramBindingNotAllowed,
                        "The file declares @" + std::string(toks[i + 1].text) +
                            "; the host writes every binding of a piece from its manifest.",
                        {}, t.line, t.column);
        const bool declares = t.text == "fn" || t.text == "struct" || t.text == "const" || t.text == "let" ||
                              t.text == "var" || t.text == "override" || t.text == "alias";
        if (!declares) continue;
        std::size_t j = i + 1;
        if (t.text == "var" && j < toks.size() && toks[j].text == "<") {  // var<storage, read> name
            while (j < toks.size() && toks[j].text != ">") ++j;
            ++j;
        }
        if (j >= toks.size() || !ident_start(toks[j].text[0])) continue;
        const std::string_view name = toks[j].text;
        if (starts_with(name, "caliper_") || starts_with(name, "Caliper") || starts_with(name, "CALIPER_") ||
            name == "caliper" || name.find("__") != std::string_view::npos)
            return fail(kProgramReservedName,
                        "The file declares " + std::string(name) +
                            "; the names caliper, caliper_*, Caliper*, CALIPER_* and every name with \"__\" belong "
                            "to the host and the lib modules.",
                        std::string(name), toks[j].line, toks[j].column);
    }
    return {};
}

// ---- places

SourcePlace source_place(const Composed& composed, std::uint32_t line, std::uint32_t column) {
    for (const Segment& s : composed.segments)
        if (line >= s.first_line && line - s.first_line < s.lines) return SourcePlace{s.part, line - s.first_line + 1, column};
    return SourcePlace{"header", line, column};
}

}  // namespace caliper_builder::programs
