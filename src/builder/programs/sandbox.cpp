// The sandbox of GPU programs and stages (sandbox.h, design/gpu-programs.md,
// Sandbox): the declaration checks, the provenance check, the source check,
// the compile and reflection through caliper-wgsl, the binding checks of a
// program and of a stage, and the helpers the hosts share.
//
// The declaration check uses three codes: program.bad_target for a target
// panel or entity path that cannot name one, program.output_mismatch for an
// output whose kind, arrays or keys do not fit together, and
// program.over_limits for every other id, name, count, size or limit outside
// its form or its maximum. The source scan below skips comments and tracks
// lines and columns in bytes from 1, as the compiler reports them, so that a
// directive, a stage's binding or a refused declaration has its place.
#include <caliper_builder/programs/sandbox.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <set>

#if defined(CALIPER_BUILDER_PROGRAMS_HAVE_WGSL)
#include <caliper/wgsl.h>
#endif

namespace caliper_builder::programs {

namespace {

namespace lim = spec::program_limits;

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

// ---- names

// A spec id: a lower-case letter, then at most 63 lower-case letters, digits, _ or -.
bool valid_id(std::string_view s) {
    if (s.empty() || s.size() > 64 || !(s[0] >= 'a' && s[0] <= 'z')) return false;
    for (char c : s)
        if (!((c >= 'a' && c <= 'z') || (c >= '0' && c <= '9') || c == '_' || c == '-')) return false;
    return true;
}

// A program's id: a spec id, or the id "operator.<source>.<step>" that the
// app gives the program of an operator (design/instrument-interface.md,
// Operator Programs), which no spec id can match because it holds dots.
bool valid_program_id(std::string_view s) {
    if (valid_id(s)) return true;
    const std::string_view prefix = "operator.";
    if (s.substr(0, prefix.size()) != prefix) return false;
    const std::string_view rest = s.substr(prefix.size());
    const std::size_t dot = rest.rfind('.');
    if (dot == std::string_view::npos || dot + 1 >= rest.size() || rest.size() - dot - 1 > 5) return false;
    for (char c : rest.substr(dot + 1))
        if (!(c >= '0' && c <= '9')) return false;
    return valid_id(rest.substr(0, dot));
}

// A WGSL identifier of at most 63 bytes, as the command table's pattern allows.
bool valid_identifier(std::string_view s) {
    if (s.empty() || s.size() > 63) return false;
    if (!((s[0] >= 'A' && s[0] <= 'Z') || (s[0] >= 'a' && s[0] <= 'z') || s[0] == '_')) return false;
    for (char c : s)
        if (!((c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z') || (c >= '0' && c <= '9') || c == '_')) return false;
    return true;
}

// An entity path: 1 to 16 segments of 1 to 63 letters, digits, _, . or -,
// joined by /, at most 255 bytes.
bool valid_entity_path(std::string_view s) {
    if (s.empty() || s.size() > 255) return false;
    std::size_t segments = 0;
    std::size_t start = 0;
    while (true) {
        const std::size_t slash = s.find('/', start);
        const std::string_view seg = s.substr(start, slash == std::string_view::npos ? std::string_view::npos : slash - start);
        if (seg.empty() || seg.size() > 63) return false;
        for (char c : seg)
            if (!((c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z') || (c >= '0' && c <= '9') || c == '_' ||
                  c == '.' || c == '-'))
                return false;
        if (++segments > 16) return false;
        if (slash == std::string_view::npos) return true;
        start = slash + 1;
    }
}

// The source of an input: "live:" and a name of 1 to 63 bytes without a
// control character.
bool valid_live_source(std::string_view s) {
    const std::string_view prefix = spec::kLiveInputPrefix;
    if (s.substr(0, prefix.size()) != prefix) return false;
    const std::string_view name = s.substr(prefix.size());
    if (name.empty() || name.size() > 63) return false;
    for (char c : name)
        if (static_cast<unsigned char>(c) < 0x20 || c == 0x7F) return false;
    return true;
}

std::string quote_text(std::string_view s) { return "\"" + std::string(s.substr(0, 64)) + "\""; }

// ---- the source scan

struct Token {
    std::string_view text;
    std::uint32_t line = 0;
    std::uint32_t column = 0;
};

bool ident_start(char c) { return (c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z') || c == '_'; }
bool ident_char(char c) { return ident_start(c) || (c >= '0' && c <= '9'); }

// Splits WGSL into identifiers and single punctuation characters, without
// comments and white space. Numbers come out as identifier-like runs. A block
// comment nests, as WGSL says. Bytes above 0x7F join no token.
std::vector<Token> tokens(std::string_view src) {
    std::vector<Token> out;
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
        const char c = src[i];
        if (c == '/' && i + 1 < src.size() && src[i + 1] == '/') {
            while (i < src.size() && src[i] != '\n') advance(1);
            continue;
        }
        if (c == '/' && i + 1 < src.size() && src[i + 1] == '*') {
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
        if (static_cast<unsigned char>(c) <= 0x20 || static_cast<unsigned char>(c) >= 0x7F) {
            advance(1);
            continue;
        }
        Token t;
        t.line = line;
        t.column = column;
        const std::size_t begin = i;
        if (ident_char(c)) {
            while (i < src.size() && ident_char(src[i])) advance(1);
        } else {
            advance(1);
        }
        t.text = src.substr(begin, i - begin);
        out.push_back(t);
    }
    return out;
}

// The place of the declaration `var<...> name` (or `var name`) of a module
// variable: the place of its name, or nothing.
std::optional<Token> var_declaration(const std::vector<Token>& toks, std::string_view name) {
    for (std::size_t i = 0; i + 1 < toks.size(); ++i) {
        if (toks[i].text != "var") continue;
        std::size_t k = i + 1;
        if (toks[k].text == "<") {
            while (k < toks.size() && toks[k].text != ">") ++k;
            ++k;
        }
        if (k < toks.size() && toks[k].text == name) return toks[k];
    }
    return std::nullopt;
}

Verdict at_declaration(Verdict v, const std::vector<Token>& toks, std::string_view name) {
    if (auto t = var_declaration(toks, name)) {
        v.line = t->line;
        v.column = t->column;
    }
    return v;
}

// ---- the compiler

std::string binding_kind_text(const ReflectedBinding& b) {
    switch (b.kind) {
        case BindingKind::uniform_buffer: return "var<uniform>";
        case BindingKind::storage_buffer: return b.write ? "var<storage, read_write>" : "var<storage, read>";
        case BindingKind::texture:
        case BindingKind::uint_texture: return "a texture";
        case BindingKind::storage_texture: return "a storage texture";
        case BindingKind::sampler: return "a sampler";
    }
    return "a binding";
}

bool is_read_write_storage(const ReflectedBinding& b) {
    return b.kind == BindingKind::storage_buffer && b.read && b.write;
}

// The compute entry point a program runs: the named one, or the one compute
// entry point. Null with a verdict when there is none.
const ReflectedEntry* program_entry(const spec::Program& p, const Reflection& r, Verdict* why) {
    std::vector<const ReflectedEntry*> compute;
    for (const ReflectedEntry& e : r.entries)
        if (e.stage == EntryStage::compute) compute.push_back(&e);
    if (p.entry) {
        for (const ReflectedEntry* e : compute)
            if (e->name == *p.entry) return e;
        if (why)
            *why = fail(kProgramNoEntry, "The source has no compute entry point " + quote_text(*p.entry) + ".", "entry");
        return nullptr;
    }
    if (compute.size() == 1) return compute.front();
    if (why)
        *why = fail(kProgramNoEntry,
                    compute.empty() ? std::string("The source has no compute entry point; add a function with "
                                                  "@compute @workgroup_size(...).")
                                    : "The source has " + std::to_string(compute.size()) +
                                          " compute entry points; name the one to run with entry.",
                    "entry");
    return nullptr;
}

std::uint32_t ceil_div(std::uint64_t a, std::uint64_t b) {
    if (b == 0) return 0;
    const std::uint64_t q = (a + b - 1) / b;
    return q > 0xFFFFFFFFull ? 0xFFFFFFFFu : static_cast<std::uint32_t>(q);
}

// ---- SHA-256 (FIPS 180-4)

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

    void update(const unsigned char* data, std::size_t n) {
        for (std::size_t i = 0; i < n; ++i) {
            block[used++] = data[i];
            if (used == 64) {
                compress();
                used = 0;
            }
        }
        bits += std::uint64_t(n) * 8;
    }

    std::string hex() {
        const std::uint64_t total = bits;
        const unsigned char pad = 0x80;
        update(&pad, 1);
        const unsigned char zero = 0;
        while (used != 56) update(&zero, 1);
        unsigned char len[8];
        for (int i = 0; i < 8; ++i) len[i] = static_cast<unsigned char>(total >> (56 - 8 * i));
        update(len, 8);
        static const char* digits = "0123456789abcdef";
        std::string out;
        out.reserve(64);
        for (std::uint32_t v : h)
            for (int s = 28; s >= 0; s -= 4) out += digits[(v >> s) & 0xF];
        return out;
    }
};

}  // namespace

bool compiler_available() {
#if defined(CALIPER_BUILDER_PROGRAMS_HAVE_WGSL)
    return true;
#else
    return false;
#endif
}

// ---- 1. declarations

Verdict check_declaration(const spec::Program& p) {
    if (!valid_program_id(p.id))
        return fail(kProgramOverLimits,
                    quote_text(p.id) + " is not a program id; an id is a lower-case letter followed by at most 63 "
                                   "lower-case letters, digits, _ or -.",
                    "id");
    if (p.wgsl.empty()) return fail(kProgramOverLimits, "The program's wgsl is empty.", "wgsl");
    if (p.wgsl.size() > lim::kProgramSourceBytes)
        return fail(kProgramTooLarge,
                    "The source has " + std::to_string(p.wgsl.size()) + " bytes; a program's limit is " +
                        std::to_string(lim::kProgramSourceBytes) + " bytes.",
                    "wgsl");
    if (p.entry && !valid_identifier(*p.entry))
        return fail(kProgramOverLimits, quote_text(*p.entry) + " is not a WGSL name of at most 63 bytes.", "entry");
    if (p.inputs.size() > lim::kInputs)
        return fail(kProgramOverLimits,
                    "The program has " + std::to_string(p.inputs.size()) + " inputs; the limit is " +
                        std::to_string(lim::kInputs) + ".",
                    "inputs");

    // The names the output and the frame uniform take.
    const spec::ProgramOutput& o = p.output;
    std::set<std::string> fixed = {spec::kFrameVar, spec::kParamsVar};
    if (o.kind == spec::OutputKind::geometry) {
        fixed.insert(spec::kCountVar);
        for (spec::GeometryArray a : o.arrays) fixed.insert(spec::geometry_array_name(a));
    } else if (o.kind == spec::OutputKind::tensor) {
        fixed.insert(kTensorVar);
    } else {
        fixed.insert(spec::kPixelsVar);
    }
    std::set<std::string> vars;
    for (std::size_t i = 0; i < p.inputs.size(); ++i) {
        const spec::ProgramInput& in = p.inputs[i];
        const std::string where = "inputs[" + std::to_string(i) + "]";
        if (!valid_identifier(in.var))
            return fail(kProgramOverLimits, quote_text(in.var) + " is not a WGSL name of at most 63 bytes.", where + ".var");
        if (!valid_live_source(in.source))
            return fail(kProgramOverLimits,
                        quote_text(in.source) + " is not a tensor of the script; an input reads live:<name>, with a name "
                                            "of 1 to 63 bytes, such as live:layers.0.weight.",
                        where + ".source");
        if (fixed.count(in.var))
            return fail(kProgramOutputMismatch,
                        "The input " + quote_text(in.var) + " has the name of an output or of the frame uniform; choose "
                                                         "another var.",
                        where + ".var");
        if (!vars.insert(in.var).second)
            return fail(kProgramOverLimits, "Two inputs bind the var " + quote_text(in.var) + ".", where + ".var");
    }

    if (p.params.size() > spec::kParams)
        return fail(kProgramOverLimits,
                    "The program has " + std::to_string(p.params.size()) + " params; the limit is " +
                        std::to_string(spec::kParams) + ".",
                    "params");
    for (const auto& [name, value] : p.params) {
        if (!valid_identifier(name) || name.substr(0, 2) == "__")
            return fail(kProgramOverLimits, quote_text(name) + " is not a WGSL field name of at most 63 bytes.",
                        "params." + name);
        if (!spec::is_reference(value) && !std::isfinite(std::get<double>(value)))
            return fail(kProgramOverLimits, "The param " + name + " is not a finite number.", "params." + name);
    }

    if (o.kind == spec::OutputKind::tensor) {
        if (o.shape.empty() || o.shape.size() > 4)
            return fail(kProgramOverLimits,
                        "A tensor output has 1 to 4 axes, not " + std::to_string(o.shape.size()) + ".", "output.shape");
        for (std::int64_t n : o.shape)
            if (n < 1)
                return fail(kProgramOverLimits, "Each axis of a tensor output is at least 1.", "output.shape");
        if (tensor_elements(o) > 1000000)
            return fail(kProgramOverLimits,
                        "A tensor output holds at most 1,000,000 elements, not " + std::to_string(tensor_elements(o)) +
                            ".",
                        "output.shape");
        if (!o.panel.empty() || o.entity || o.max_count != 0 || !o.arrays.empty() || o.width != 0 || o.height != 0 ||
            !(o.style == spec::GeometryStyle{}))
            return fail(kProgramOutputMismatch,
                        "A tensor output takes its shape only; panel, entity, part, max_count, arrays, style, width "
                        "and height are for geometry and pixels.",
                        "output");
    } else if (o.kind == spec::OutputKind::geometry) {
        if (!valid_id(o.panel))
            return fail(kProgramBadTarget,
                        o.panel.empty() ? std::string("A geometry output needs the panel of a scene to draw in.")
                                        : quote_text(o.panel) + " is not a panel id.",
                        "output.panel");
        if (o.entity && !valid_entity_path(*o.entity))
            return fail(kProgramBadTarget,
                        quote_text(*o.entity) + " is not an entity path; a path has 1 to 16 segments of letters, digits, "
                                            "_, . and -, joined by /.",
                        "output.entity");
        if (o.max_count < 1 || o.max_count > lim::kMaxCount)
            return fail(kProgramOverLimits,
                        "max_count must be from 1 to " + std::to_string(lim::kMaxCount) + ", not " +
                            std::to_string(o.max_count) + ".",
                        "output.max_count");
        if (o.width != 0 || o.height != 0)
            return fail(kProgramOutputMismatch, "A geometry output takes no width or height; they are for pixels.",
                        "output.width");
        std::set<spec::GeometryArray> seen;
        for (spec::GeometryArray a : o.arrays)
            if (!seen.insert(a).second)
                return fail(kProgramOutputMismatch,
                            "The output names the array " + std::string(spec::geometry_array_name(a)) + " twice.",
                            "output.arrays");
        if (!seen.count(spec::GeometryArray::positions))
            return fail(kProgramOutputMismatch, "A geometry output needs the array positions.", "output.arrays");
        if (seen.count(spec::GeometryArray::radii) && o.part != spec::GeometryPart::points)
            return fail(kProgramOutputMismatch, "The array radii is for points only; lines take no radii.",
                        "output.arrays");
        if (o.style.vmin && o.style.vmax && !(*o.style.vmin < *o.style.vmax))
            return fail(kProgramOverLimits, "style.vmin must be below style.vmax.", "output.style.vmin");
    } else {
        if (o.width < 1 || o.width > lim::kPixelsSide || o.height < 1 || o.height > lim::kPixelsSide)
            return fail(kProgramOverLimits,
                        "width and height must each be from 1 to " + std::to_string(lim::kPixelsSide) + ", not " +
                            std::to_string(o.width) + " by " + std::to_string(o.height) + ".",
                        o.width < 1 || o.width > lim::kPixelsSide ? "output.width" : "output.height");
        if (!o.panel.empty() || o.entity || o.max_count != 0 || !o.arrays.empty() || !o.shape.empty() ||
            !(o.style == spec::GeometryStyle{}))
            return fail(kProgramOutputMismatch,
                        "A pixels output takes width, height and format only; panel, entity, part, max_count, arrays "
                        "and style are for geometry.",
                        "output");
    }

    const spec::ProgramLimits& l = p.limits;
    if (l.time_ms && !(std::isfinite(*l.time_ms) && *l.time_ms >= 0.01 && *l.time_ms <= lim::kTimeMsMax))
        return fail(kProgramOverLimits, "limits.time_ms must be from 0.01 to 50 milliseconds.", "limits.time_ms");
    if (l.memory_mb && (*l.memory_mb < 1 || *l.memory_mb > lim::kMemoryMbMax))
        return fail(kProgramOverLimits, "limits.memory_mb must be from 1 to 512.", "limits.memory_mb");
    if (l.every && (*l.every < 1 || *l.every > lim::kEveryMax))
        return fail(kProgramOverLimits, "limits.every must be from 1 to 10000.", "limits.every");
    if (p.workgroups)
        for (std::uint32_t n : *p.workgroups)
            if (n < 1 || n > lim::kWorkgroupsPerDimension)
                return fail(kProgramOverLimits, "Each workgroup count must be from 1 to 65535.", "workgroups");
    const std::uint64_t bytes = output_bytes(p);
    if (bytes > l.memory_bytes_or_default())
        return fail(kProgramOverLimits,
                    "The output buffers take " + std::to_string(bytes) + " bytes, above the memory budget of " +
                        std::to_string(l.memory_bytes_or_default()) + " bytes; lower the output's size or raise "
                                                                      "limits.memory_mb.",
                    "limits.memory_mb");
    return {};
}

Verdict check_stage_declaration(const spec::Stage& s) {
    if (!valid_id(s.panel))
        return fail(kStageNotScenePanel, quote_text(s.panel) + " is not a panel id.", "panel");
    if (!valid_entity_path(s.entity))
        return fail(kProgramBadTarget,
                    quote_text(s.entity) + " is not an entity path; a path has 1 to 16 segments of letters, digits, _, . "
                                       "and -, joined by /.",
                    "entity");
    if (s.wgsl.empty()) return fail(kProgramOverLimits, "The stage's wgsl is empty.", "wgsl");
    if (s.wgsl.size() > lim::kStageSourceBytes)
        return fail(kProgramTooLarge,
                    "The source has " + std::to_string(s.wgsl.size()) + " bytes; a stage's limit is " +
                        std::to_string(lim::kStageSourceBytes) + " bytes.",
                    "wgsl");
    return {};
}

// ---- 2. provenance

Verdict check_provenance(const spec::Provenance& provenance, std::string_view target, std::string_view sha256) {
    if (!provenance.foreign) return {};
    for (const spec::ReviewedSource& r : provenance.reviewed)
        if (r.target == target && r.sha256 == sha256) return {};
    std::string origin = provenance.from ? " from " + *provenance.from : std::string();
    return fail(kProgramNeedsReview,
                "This spec came" + origin + " from outside the workspace, and nobody reviewed this source of " +
                    std::string(target) + " yet; it runs after a person reviews it.",
                std::string(target));
}

// ---- 3. source

Verdict check_source(std::string_view wgsl, std::size_t max_bytes) {
    if (wgsl.size() > max_bytes)
        return fail(kProgramTooLarge,
                    "The source has " + std::to_string(wgsl.size()) + " bytes; the limit is " +
                        std::to_string(max_bytes) + " bytes.",
                    "wgsl");
    // Directives come before every declaration, thus only the first
    // statements can be directives; any later one is a syntax error.
    const std::vector<Token> toks = tokens(wgsl);
    if (!toks.empty()) {
        const std::string_view first = toks.front().text;
        if (first == "enable" || first == "requires" || first == "diagnostic")
            return fail(kProgramDirectiveNotAllowed,
                        "The source starts with the directive " + std::string(first) +
                            "; a program or a stage may not enable extensions or change diagnostics.",
                        "wgsl", toks.front().line, toks.front().column);
    }
    return {};
}

// ---- 4. compile

Compiled compile_and_reflect(std::string_view wgsl) { return compile_and_reflect_entry(wgsl, {}); }

Compiled compile_and_reflect_entry(std::string_view wgsl, std::string_view entry) {
    Compiled out;
#if defined(CALIPER_BUILDER_PROGRAMS_HAVE_WGSL)
    if (caliper_wgsl_abi_version() != CALIPER_WGSL_ABI_VERSION) {
        out.verdict = fail(kProgramCompilerUnavailable,
                           "The WGSL compiler of this build has another interface version, so the source cannot be "
                           "checked here.");
        return out;
    }
    caliper_wgsl_module* module = nullptr;
    caliper_wgsl_error error{};
    caliper_wgsl_options options;
    caliper_wgsl_options_init(&options);
    const std::string keep(entry);
    if (!keep.empty()) options.entry_point = keep.c_str();
    const caliper_wgsl_status st = caliper_wgsl_module_create(wgsl.data(), wgsl.size(), &options, &module, &error);
    if (st != CALIPER_WGSL_OK || !module) {
        std::string message = error.message ? error.message : "The compiler refused the source.";
        out.verdict = fail(st == CALIPER_WGSL_ERROR_TOO_LARGE ? kProgramTooLarge : kProgramCompileFailed,
                           std::move(message), "wgsl", error.line, error.column);
        caliper_wgsl_error_free(&error);
        caliper_wgsl_module_destroy(module);
        return out;
    }
    caliper_wgsl_error_free(&error);
    const std::size_t nb = caliper_wgsl_module_binding_count(module);
    for (std::size_t i = 0; i < nb; ++i) {
        caliper_wgsl_binding b{};
        if (caliper_wgsl_module_binding(module, i, &b) != CALIPER_WGSL_OK) continue;
        ReflectedBinding r;
        r.group = b.group;
        r.binding = b.binding;
        r.kind = static_cast<BindingKind>(b.kind);
        r.read = (b.access & CALIPER_WGSL_ACCESS_READ) != 0;
        r.write = (b.access & CALIPER_WGSL_ACCESS_WRITE) != 0;
        r.size = b.size;
        r.name = b.name ? b.name : "";
        r.storage_format = b.storage_format ? b.storage_format : "";
        out.reflection.bindings.push_back(std::move(r));
    }
    const std::size_t ne = caliper_wgsl_module_entry_point_count(module);
    for (std::size_t i = 0; i < ne; ++i) {
        caliper_wgsl_entry_point e{};
        if (caliper_wgsl_module_entry_point(module, i, &e) != CALIPER_WGSL_OK) continue;
        ReflectedEntry r;
        r.name = e.name ? e.name : "";
        r.stage = static_cast<EntryStage>(e.stage);
        r.workgroup_size = {e.workgroup_size[0], e.workgroup_size[1], e.workgroup_size[2]};
        r.workgroup_memory = e.workgroup_memory;
        for (std::size_t k = 0, n = caliper_wgsl_module_entry_io_count(module, i); k < n; ++k) {
            caliper_wgsl_io io{};
            if (caliper_wgsl_module_entry_io(module, i, k, &io) != CALIPER_WGSL_OK) continue;
            ReflectedIo v;
            v.location = io.location;
            v.scalar = static_cast<IoScalar>(io.scalar);
            v.components = io.components;
            v.interpolation = static_cast<IoInterpolation>(io.interpolation);
            v.sampling = static_cast<IoSampling>(io.sampling);
            (io.direction == CALIPER_WGSL_IO_INPUT ? r.inputs : r.outputs).push_back(v);
        }
        out.reflection.entries.push_back(std::move(r));
    }
    caliper_wgsl_module_destroy(module);
#else
    (void)wgsl;
    (void)entry;
    out.verdict = fail(kProgramCompilerUnavailable,
                       "This build has no WGSL compiler, so the source cannot be checked here.");
#endif
    return out;
}

// ---- 5. and 6. bindings

Verdict check_program(const spec::Program& p, const Reflection& r) {
    const std::vector<Token> toks = tokens(p.wgsl);
    Verdict why;
    const ReflectedEntry* entry = program_entry(p, r, &why);
    if (!entry) {
        if (p.entry)
            if (auto t = std::find_if(toks.begin(), toks.end(), [&](const Token& k) { return k.text == *p.entry; });
                t != toks.end()) {
                why.line = t->line;
                why.column = t->column;
            }
        return why;
    }
    // Metal reserves the name main, and the translation to MSL renames such an
    // entry point, which the graphics layer then cannot find by its name.
    if (entry->name == "main") {
        std::uint32_t line = 0, column = 0;
        for (std::size_t i = 0; i + 1 < toks.size(); ++i)
            if (toks[i].text == "fn" && toks[i + 1].text == "main") {
                line = toks[i + 1].line;
                column = toks[i + 1].column;
                break;
            }
        return fail(kProgramCompileFailed,
                    "An entry point cannot be named main, because Metal reserves that name; give it another name, "
                    "such as run.",
                    "entry", line, column);
    }
    const auto& ws = entry->workgroup_size;
    const std::uint64_t invocations = std::uint64_t(ws[0]) * ws[1] * ws[2];
    if (invocations == 0 || invocations > lim::kWorkgroupInvocations)
        return fail(kProgramOverLimits,
                    "The entry point " + quote_text(entry->name) + " has a workgroup of " + std::to_string(invocations) +
                        " invocations; the limit is " + std::to_string(lim::kWorkgroupInvocations) + ".",
                    "entry");
    if (!p.workgroups) {
        const std::array<std::uint32_t, 3> d = dispatch_size(p, r);
        for (std::uint32_t n : d)
            if (n < 1 || n > lim::kWorkgroupsPerDimension)
                return fail(kProgramOverLimits,
                            "The output needs " + std::to_string(n) +
                                " workgroups in one dimension, above 65535; use a larger workgroup size or set "
                                "workgroups.",
                            "workgroups");
    }

    if (Verdict v = check_params(p); !v.ok()) return v;
    const spec::ProgramOutput& o = p.output;
    const bool geometry = o.kind == spec::OutputKind::geometry;
    const bool tensor = o.kind == spec::OutputKind::tensor;
    std::set<std::string> bound;
    for (const ReflectedBinding& b : r.bindings) {
        bound.insert(b.name);
        const auto input = std::find_if(p.inputs.begin(), p.inputs.end(),
                                         [&](const spec::ProgramInput& in) { return in.var == b.name; });
        if (input != p.inputs.end()) {
            if (b.kind != BindingKind::storage_buffer || b.write || !b.read)
                return at_declaration(fail(kProgramWritesInput,
                                           "The input " + b.name + " reads " + input->source +
                                               ", and an input must be var<storage, read>, not " +
                                               binding_kind_text(b) + "; a program never writes a tensor.",
                                           b.name),
                                      toks, b.name);
            continue;
        }
        if (b.name == spec::kFrameVar) {
            if (b.kind != BindingKind::uniform_buffer || b.size != lim::kFrameUniformBytes)
                return at_declaration(fail(kProgramOutputMismatch,
                                           "caliper must be var<uniform> of the 32-byte struct CaliperFrame, not " +
                                               binding_kind_text(b) + " of " + std::to_string(b.size) + " bytes.",
                                           b.name),
                                      toks, b.name);
            continue;
        }
        if (geometry && b.name == spec::kCountVar) {
            if (!is_read_write_storage(b) || b.size != 4)
                return at_declaration(fail(kProgramOutputMismatch,
                                           "count must be var<storage, read_write> count: atomic<u32>.", b.name),
                                      toks, b.name);
            continue;
        }
        if (b.name == spec::kParamsVar && !p.params.empty()) {
            if (b.kind != BindingKind::uniform_buffer)
                return at_declaration(fail(kProgramParamsMismatch,
                                           "params must be var<uniform> params: Params, not " + binding_kind_text(b) +
                                               ".",
                                           b.name),
                                      toks, b.name);
            continue;
        }
        if (tensor && b.name == kTensorVar) {
            if (!is_read_write_storage(b))
                return at_declaration(fail(kProgramOutputMismatch,
                                           "out must be var<storage, read_write> out: array<f32>.", b.name),
                                      toks, b.name);
            continue;
        }
        if (!geometry && !tensor && b.name == spec::kPixelsVar) {
            if (!is_read_write_storage(b))
                return at_declaration(fail(kProgramOutputMismatch,
                                           "pixels must be var<storage, read_write> pixels: array<u32>.", b.name),
                                      toks, b.name);
            continue;
        }
        if (geometry) {
            const auto a = spec::geometry_array_from_name(b.name);
            if (a && std::find(o.arrays.begin(), o.arrays.end(), *a) != o.arrays.end()) {
                if (!is_read_write_storage(b))
                    return at_declaration(fail(kProgramOutputMismatch,
                                               "The output array " + b.name + " must be var<storage, read_write> " +
                                                   b.name + ": array<" +
                                                   (*a == spec::GeometryArray::colors ? "u32" : "f32") + ">.",
                                               b.name),
                                          toks, b.name);
                continue;
            }
        }
        return at_declaration(fail(kProgramBindingUndeclared,
                                   "The binding " + b.name + " is not an input, an output array of the declaration, " +
                                       (geometry ? "count" : tensor ? "out" : "pixels") +
                                       ", params or caliper; declare it as an input or remove it.",
                                   b.name),
                              toks, b.name);
    }
    // The necessary outputs, and every declared array, are bound.
    if (geometry) {
        if (!bound.count(spec::kCountVar))
            return fail(kProgramOutputMismatch,
                        "A geometry program must add its items to count, a var<storage, read_write> atomic<u32>, and "
                        "the entry point does not use it.",
                        spec::kCountVar);
        for (spec::GeometryArray a : o.arrays)
            if (!bound.count(spec::geometry_array_name(a)))
                return fail(kProgramOutputMismatch,
                            "The output array " + std::string(spec::geometry_array_name(a)) +
                                " is declared, and the entry point does not write it; use it or remove it from "
                                "arrays.",
                            spec::geometry_array_name(a));
    } else if (tensor) {
        if (!bound.count(kTensorVar))
            return fail(kProgramOutputMismatch,
                        "A tensor program must write out, a var<storage, read_write> array<f32>, and the entry point "
                        "does not use it.",
                        kTensorVar);
    } else if (!bound.count(spec::kPixelsVar)) {
        return fail(kProgramOutputMismatch,
                    "A pixels program must write pixels, a var<storage, read_write> array<u32>, and the entry point "
                    "does not use it.",
                    spec::kPixelsVar);
    }
    return {};
}

Verdict check_stage(const spec::Stage& s, const Reflection& r) {
    const std::vector<Token> toks = tokens(s.wgsl);
    if (!r.bindings.empty())
        return at_declaration(fail(kStageBindingNotAllowed,
                                   "A stage declares no binding, and the source declares " + r.bindings.front().name +
                                       "; the viewer gives the stage its inputs through MaterialIn and VertexIn.",
                                   r.bindings.front().name),
                              toks, r.bindings.front().name);
    // The compiler drops a binding that no entry point uses, thus the
    // attributes are looked for in the source as well.
    for (const Token& t : toks)
        if (t.text == "@") {
            const auto next = &t + 1;
            if (next < toks.data() + toks.size() && (next->text == "group" || next->text == "binding"))
                return fail(kStageBindingNotAllowed,
                            "A stage declares no binding; the viewer gives the stage its inputs through MaterialIn "
                            "and VertexIn.",
                            "wgsl", t.line, t.column);
        }
    if (!r.entries.empty()) {
        Verdict v = fail(kStageEntryNotAllowed,
                         "A stage declares no entry point, and the source declares " + r.entries.front().name +
                             "; the viewer adds its own entry points, which call fn " + stage_function(s.kind) + ".",
                         r.entries.front().name);
        for (std::size_t i = 0; i + 2 < toks.size(); ++i)
            if (toks[i].text == "fn" && toks[i + 1].text == r.entries.front().name) {
                v.line = toks[i + 1].line;
                v.column = toks[i + 1].column;
            }
        return v;
    }
    for (std::size_t i = 0; i + 1 < toks.size(); ++i)
        if (toks[i].text == "@" &&
            (toks[i + 1].text == "vertex" || toks[i + 1].text == "fragment" || toks[i + 1].text == "compute"))
            return fail(kStageEntryNotAllowed,
                        "A stage declares no entry point; the viewer adds its own entry points, which call fn " +
                            std::string(stage_function(s.kind)) + ".",
                        "wgsl", toks[i].line, toks[i].column);
    return {};
}

// ---- helpers

std::string sha256_hex(std::string_view text) {
    Sha256 h;
    h.update(reinterpret_cast<const unsigned char*>(text.data()), text.size());
    return h.hex();
}

std::array<std::uint32_t, 3> dispatch_size(const spec::Program& p, const Reflection& r) {
    if (p.workgroups) return *p.workgroups;
    const ReflectedEntry* e = program_entry(p, r, nullptr);
    if (!e) return {0, 0, 0};
    const auto& ws = e->workgroup_size;
    if (p.output.kind == spec::OutputKind::geometry) return {ceil_div(p.output.max_count, ws[0]), 1, 1};
    if (p.output.kind == spec::OutputKind::tensor) return {ceil_div(tensor_elements(p.output), ws[0]), 1, 1};
    return {ceil_div(p.output.width, ws[0]), ceil_div(p.output.height, ws[1]), 1};
}

std::uint64_t tensor_elements(const spec::ProgramOutput& o) {
    if (o.kind != spec::OutputKind::tensor || o.shape.empty()) return 0;
    std::uint64_t n = 1;
    for (std::int64_t d : o.shape) {
        if (d < 1) return 0;
        n = n > (std::uint64_t(1) << 40) ? n : n * std::uint64_t(d);
    }
    return n;
}

// ---- params

bool params_fields(std::string_view wgsl, std::vector<ParamField>* fields, Verdict* why) {
    fields->clear();
    const std::vector<Token> toks = tokens(wgsl);
    for (std::size_t i = 0; i + 2 < toks.size(); ++i) {
        if (toks[i].text != "struct" || toks[i + 1].text != "Params" || toks[i + 2].text != "{") continue;
        std::size_t k = i + 3;
        while (k < toks.size() && toks[k].text != "}") {
            if (toks[k].text == ",") {
                ++k;
                continue;
            }
            if (toks[k].text == "@" || k + 2 >= toks.size() || toks[k + 1].text != ":") {
                if (why)
                    *why = fail(kProgramParamsMismatch,
                                "Each field of struct Params is name: f32, i32 or u32, with no attribute.", "params",
                                toks[k].line, toks[k].column);
                return true;
            }
            const std::string_view type = toks[k + 2].text;
            ParamField f;
            f.name = std::string(toks[k].text);
            if (type == "f32") f.type = ParamType::f32;
            else if (type == "i32") f.type = ParamType::i32;
            else if (type == "u32") f.type = ParamType::u32;
            else {
                if (why)
                    *why = fail(kProgramParamsMismatch,
                                "The field " + f.name + " of struct Params is " + std::string(type) +
                                    "; a field is f32, i32 or u32.",
                                "params." + f.name, toks[k + 2].line, toks[k + 2].column);
                return true;
            }
            fields->push_back(std::move(f));
            k += 3;
        }
        return true;
    }
    return false;
}

Verdict check_params(const spec::Program& p) {
    std::vector<ParamField> fields;
    Verdict why;
    const bool has = params_fields(p.wgsl, &fields, &why);
    if (!why.ok()) return why;
    const std::vector<Token> toks = tokens(p.wgsl);
    if (!has) {
        if (p.params.empty()) {
            if (auto t = var_declaration(toks, spec::kParamsVar))
                return fail(kProgramParamsMismatch,
                            "The source declares params, and the program has no params; declare them or remove the "
                            "variable.",
                            "params", t->line, t->column);
            return {};
        }
        return fail(kProgramParamsMismatch,
                    "The program has params, and the source declares no struct Params; declare struct Params with "
                    "one f32, i32 or u32 field for each name and var<uniform> params: Params.",
                    "params");
    }
    std::set<std::string> names;
    for (const auto& kv : p.params) names.insert(kv.first);
    std::set<std::string> declared;
    for (const ParamField& f : fields)
        if (!declared.insert(f.name).second)
            return fail(kProgramParamsMismatch, "struct Params declares the field " + f.name + " twice.", "params");
    if (names != declared) {
        std::string missing, extra;
        for (const std::string& n : names)
            if (!declared.count(n)) missing += (missing.empty() ? "" : ", ") + n;
        for (const std::string& n : declared)
            if (!names.count(n)) extra += (extra.empty() ? "" : ", ") + n;
        std::string message = "The fields of struct Params differ from the names of params:";
        if (!missing.empty()) message += " params names " + missing + ", which the struct lacks";
        if (!missing.empty() && !extra.empty()) message += ", and";
        if (!extra.empty()) message += " the struct declares " + extra + ", which params lacks";
        message += ".";
        Verdict v = fail(kProgramParamsMismatch, message, "params");
        for (std::size_t i = 0; i + 1 < toks.size(); ++i)
            if (toks[i].text == "struct" && toks[i + 1].text == "Params") {
                v.line = toks[i + 1].line;
                v.column = toks[i + 1].column;
                break;
            }
        return v;
    }
    return {};
}

Verdict check_stage_params(const spec::Stage& s) {
    const std::vector<Token> toks = tokens(s.wgsl);
    for (std::size_t i = 0; i + 1 < toks.size(); ++i) {
        const bool own_struct = toks[i].text == "struct" && toks[i + 1].text == "Params";
        if (own_struct)
            return fail(kStageParamsMismatch,
                        "A stage declares no struct Params; the viewer declares it from the stage's params.", "wgsl",
                        toks[i + 1].line, toks[i + 1].column);
    }
    if (auto t = var_declaration(toks, spec::kParamsVar))
        return fail(kStageParamsMismatch,
                    "A stage declares no params variable; the viewer declares params from the stage's params.",
                    "wgsl", t->line, t->column);
    for (std::size_t i = 0; i + 2 < toks.size(); ++i) {
        if (toks[i].text != spec::kParamsVar || toks[i + 1].text != ".") continue;
        if (i > 0 && toks[i - 1].text == ".") continue;   // a field named params of something else
        const std::string name(toks[i + 2].text);
        if (!s.params.count(name))
            return fail(kStageParamsMismatch,
                        s.params.empty() ? "The stage reads params." + name + ", and it has no params."
                                         : "The stage reads params." + name + ", which is not a name of its params.",
                        "params." + name, toks[i + 2].line, toks[i + 2].column);
    }
    return {};
}

bool param_in_range(ParamType type, double value) {
    if (!std::isfinite(value)) return false;
    switch (type) {
        case ParamType::f32: return std::fabs(value) <= static_cast<double>(std::numeric_limits<float>::max());
        case ParamType::i32: {
            const double r = std::round(value);
            return r >= -2147483648.0 && r <= 2147483647.0;
        }
        case ParamType::u32: {
            const double r = std::round(value);
            return r >= 0.0 && r <= 4294967295.0;
        }
    }
    return false;
}

std::vector<std::uint8_t> pack_params(const std::vector<ParamField>& fields,
                                      const std::vector<std::pair<std::string, double>>& values) {
    if (fields.empty()) return {};
    std::vector<std::uint8_t> out(((fields.size() * 4 + 15) / 16) * 16, 0);
    for (std::size_t i = 0; i < fields.size(); ++i) {
        double v = 0;
        for (const auto& kv : values)
            if (kv.first == fields[i].name) v = kv.second;
        if (!std::isfinite(v)) v = 0;
        std::uint32_t bits = 0;
        if (fields[i].type == ParamType::f32) {
            const float f = static_cast<float>(v);
            std::memcpy(&bits, &f, 4);
        } else if (fields[i].type == ParamType::i32) {
            const double r = std::round(std::clamp(v, -2147483648.0, 2147483647.0));
            const std::int32_t n = static_cast<std::int32_t>(r);
            std::memcpy(&bits, &n, 4);
        } else {
            bits = static_cast<std::uint32_t>(std::round(std::clamp(v, 0.0, 4294967295.0)));
        }
        std::memcpy(out.data() + i * 4, &bits, 4);
    }
    return out;
}

std::vector<ParamField> stage_params_fields(const spec::Stage& s) {
    std::vector<ParamField> out;
    for (const auto& kv : s.params) out.push_back(ParamField{kv.first, ParamType::f32});
    return out;
}

std::string stage_params_declaration(const spec::Stage& s, bool uniform) {
    if (s.params.empty()) return {};
    std::string text = "\nstruct Params {";
    bool first = true;
    for (const auto& kv : s.params) {
        text += (first ? " " : ", ") + kv.first + ": f32";
        first = false;
    }
    text += " }\n";
    if (uniform)
        text += "@group(" + std::to_string(kStageParamsGroup) + ") @binding(0) var<uniform> params: Params;\n";
    else
        text += "var<private> params: Params;\n";
    return text;
}

std::uint64_t output_bytes(const spec::Program& p) {
    const spec::ProgramOutput& o = p.output;
    if (o.kind == spec::OutputKind::tensor) return tensor_elements(o) * 4;
    if (o.kind == spec::OutputKind::pixels) return std::uint64_t(o.width) * o.height * 4;
    std::uint64_t bytes = 4;   // count
    for (spec::GeometryArray a : o.arrays) bytes += std::uint64_t(o.max_count) * spec::geometry_array_words(a) * 4;
    return bytes;
}

}  // namespace caliper_builder::programs
