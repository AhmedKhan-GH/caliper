// The composer of a piece's derive half: the generated header of the
// producer runtime's module (programs/pieces.h, Generated Header), then the
// lib modules, common.wgsl and derive.wgsl with their segments.
//
// The header binds, in this order: group 0 the frame uniform `caliper`, the
// uniform `params` when the piece has params, and the uniform `shapes` when it
// has tensor inputs; group 1 each tensor input, read-only, at the input's
// index; group 2 each resource at 2 x its index, read_write, and the counter
// <id>_count of a counted buffer at 2 x its index + 1. A texture resource is a
// packed array<u32> with its size <id>_size: vec2<u32> at 2 x its index + 1, and
// a buffer of the narrow element u8 or u16 is a packed array<u32> too. Every name and type that reaches the text is checked
// here first, because the manifest can come from the wire: a name is a WGSL
// identifier without "__" and without a host prefix, and a type is one of the
// field types of the plan. This file needs only the standard library and
// spec/piece.h, because the browser module compiles it too.
#include <caliper_builder/programs/pieces.h>

#include <algorithm>
#include <cctype>
#include <map>
#include <set>
#include <vector>

namespace caliper_builder::programs {

namespace {

namespace pl = spec::piece_limits;

constexpr std::size_t kIdentifierMax = 63;

Verdict refuse(const char* code, std::string message) {
    Verdict v;
    v.code = code;
    v.message = std::move(message);
    return v;
}

bool identifier_char(char c, bool first) {
    return c == '_' || (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (!first && c >= '0' && c <= '9');
}

// A WGSL identifier of the piece: letters, digits and "_", not "_" alone, no
// "__" and no host prefix, at most 63 bytes.
bool piece_identifier(const std::string& s) {
    if (s.empty() || s.size() > kIdentifierMax || s == "_" || s.find("__") != std::string::npos) return false;
    for (std::size_t i = 0; i < s.size(); ++i)
        if (!identifier_char(s[i], i == 0)) return false;
    return s.rfind("caliper", 0) != 0 && s.rfind("Caliper", 0) != 0 && s.rfind("CALIPER", 0) != 0;
}

bool field_type(const std::string& t) {
    static const char* const kTypes[] = {"f32", "i32", "u32", "vec2<f32>", "vec4<f32>", "vec2<u32>", "vec4<u32>"};
    return std::any_of(std::begin(kTypes), std::end(kTypes), [&](const char* k) { return t == k; });
}

// The WGSL constant of a tensor dtype in the generated header.
const char* dtype_constant(const std::string& t) {
    return t == "f32" ? "CALIPER_F32" : t == "i32" ? "CALIPER_I32" : t == "u32" ? "CALIPER_U32" : t == "u8" ? "CALIPER_U8"
                                                                                                     : "CALIPER_U16";
}

// The body of the checked accessor <name>_at() of a typed tensor input: the element `i` of the bound type
// `bound` from the words of the binding, converted to the read type `read`, and zero at or past the
// element count that the runtime writes into shapes.<name>__len.
std::string typed_accessor(const std::string& name, const std::string& bound, const std::string& read) {
    std::string v;   // the element as a u32, i32 or f32 expression of the bound type
    std::string type = bound;
    if (bound == "u8") {
        v = "(" + name + "[i >> 2u] >> ((i & 3u) * 8u)) & 0xffu";
        type = "u32";
    } else if (bound == "u16") {
        v = "(" + name + "[i >> 1u] >> ((i & 1u) * 16u)) & 0xffffu";
        type = "u32";
    } else if (bound == "f32") {
        v = "bitcast<f32>(" + name + "[i])";
    } else if (bound == "i32") {
        v = "bitcast<i32>(" + name + "[i])";
    } else {
        v = name + "[i]";
    }
    const std::string value = type == read ? v : read + "(" + v + ")";
    return "fn " + name + "_at(i: u32) -> " + read + " {\n    if (i >= shapes." + name + "__len) {\n        return " +
           read + "(0);\n    }\n    return " + value + ";\n}\n";
}

const char* param_type(spec::ParamType t) {
    return t == spec::ParamType::i32 ? "i32" : t == spec::ParamType::u32 ? "u32" : "f32";
}

std::uint32_t count_lines(const std::string& text) {
    return std::uint32_t(std::count(text.begin(), text.end(), '\n'));
}

// The text with one LF at its end, so that the next segment starts on a line of its own.
std::string with_final_newline(std::string text) {
    if (!text.empty() && text.back() != '\n') text += '\n';
    return text;
}

// The element type of a buffer resource, and the struct it needs, if any.
Verdict element_of(const spec::Resource& r, std::string* type, std::string* struct_text) {
    if (r.kind == spec::ResourceKind::texture) {
        *type = "u32";
        return {};
    }
    const spec::Layout& l = r.layout;
    if (l.element && (l.type_name || !l.fields.empty()))
        return refuse(kProgramBadLayout, "The resource " + r.id + " has both `element` and a struct.");
    if (l.element) {
        if (spec::narrow_per_word(*l.element) > 1) {   // u8 or u16, four or two to a word
            *type = "u32";
            return {};
        }
        if (!field_type(*l.element))
            return refuse(kProgramBadLayout, "The resource " + r.id + " has the element type " + *l.element +
                                                 ", which is not f32, i32, u32, vec2<f32>, vec4<f32>, vec2<u32> or "
                                                 "vec4<u32>.");
        *type = *l.element;
        return {};
    }
    if (!l.type_name || l.fields.empty())
        return refuse(kProgramBadLayout, "The resource " + r.id + " needs `element`, or `struct` with `fields`.");
    if (!piece_identifier(*l.type_name))
        return refuse(kProgramBadLayout, "The struct name " + *l.type_name + " of the resource " + r.id +
                                             " is not a WGSL name of a piece.");
    if (l.fields.size() > pl::kFields)
        return refuse(kProgramBadLayout, "The struct " + *l.type_name + " has more than 16 fields.");
    std::set<std::string> names;
    std::string text = "struct " + *l.type_name + " { ";
    for (std::size_t i = 0; i < l.fields.size(); ++i) {
        const spec::Field& f = l.fields[i];
        if (!piece_identifier(f.name) || !names.insert(f.name).second)
            return refuse(kProgramBadLayout, "The field " + f.name + " of the struct " + *l.type_name +
                                                 " is not a WGSL name of a piece, or it repeats.");
        if (!field_type(f.type))
            return refuse(kProgramBadLayout, "The field " + f.name + " of the struct " + *l.type_name +
                                                 " has the type " + f.type + ", which a layout cannot hold.");
        text += (i ? ", " : "") + f.name + ": " + f.type;
    }
    text += " }\n";
    *type = *l.type_name;
    *struct_text = text;
    return {};
}

// Whether a lib module's text names a binding or a struct of the draw prelude:
// an identifier with the prefix caliper_ or Caliper, other than the derive
// header's own CaliperFrame and CaliperShapes. Comments do not count.
bool reads_draw_prelude(std::string_view text) {
    std::size_t i = 0;
    const std::size_t n = text.size();
    auto ident = [](char c) { return std::isalnum(static_cast<unsigned char>(c)) || c == '_'; };
    while (i < n) {
        if (text.compare(i, 2, "//") == 0) {
            while (i < n && text[i] != '\n') ++i;
        } else if (text.compare(i, 2, "/*") == 0) {
            const std::size_t end = text.find("*/", i + 2);
            i = end == std::string_view::npos ? n : end + 2;
        } else if (ident(text[i]) && !std::isdigit(static_cast<unsigned char>(text[i]))) {
            const std::size_t b = i;
            while (i < n && ident(text[i])) ++i;
            const std::string_view w = text.substr(b, i - b);
            if ((w.rfind("caliper_", 0) == 0 || w.rfind("Caliper", 0) == 0) && w != "CaliperFrame" &&
                w != "CaliperShapes")
                return true;
        } else {
            ++i;
        }
    }
    return false;
}

// The same text with every line empty: the line count stays, thus every
// segment after it keeps its lines.
std::string blank_lines(std::string_view text) {
    return std::string(static_cast<std::size_t>(std::count(text.begin(), text.end(), '\n')), '\n');
}

}  // namespace

Resolved split_common(std::string_view common) {
    Resolved r;
    std::size_t at = 0;
    std::uint32_t line = 1;
    for (bool found = true; found;) {
        found = false;
        for (std::string_view name : lib_modules()) {
            std::string body(lib_module(name));
            if (body.empty()) continue;
            if (body.back() != '\n') body += '\n';
            if (common.compare(at, body.size(), body) != 0) continue;
            const auto lines = static_cast<std::uint32_t>(std::count(body.begin(), body.end(), '\n'));
            r.modules.push_back(std::string(name));
            r.lib_segments.push_back(Segment{"lib/" + std::string(name), line, lines});
            r.lib_text += body;
            line += lines;
            at += body.size();
            found = true;
            break;
        }
    }
    if (at < common.size()) r.files["common.wgsl"] = std::string(common.substr(at));
    return r;
}

Composed compose_derive(const spec::Piece& piece, const Resolved& resolved,
                        const std::map<std::string, std::string>& bound) {
    Composed c;
    auto fail = [&](Verdict v) {
        c.verdict = std::move(v);
        c.text.clear();
        c.segments.clear();
        return c;
    };
    if (!resolved.verdict.ok()) return fail(resolved.verdict);
    if (!piece.derive || piece.derive->passes.empty())
        return fail(refuse(kProgramBadPass, "The piece " + piece.name + " has no derive half with a pass."));
    if (piece.derive->passes.size() > pl::kDerivePasses)
        return fail(refuse(kProgramBadPass, "The derive half has more than 8 passes."));
    if (piece.inputs.size() > pl::kInputs || piece.params.size() > pl::kParams ||
        piece.resources.size() > pl::kResources)
        return fail(refuse(kPieceBadManifest, "The piece has more than 8 inputs, 16 params or 8 resources."));

    // Every name of the header, once: inputs, params, resources, counters and structs.
    std::set<std::string> taken{"params", "shapes"};
    auto claim = [&](const std::string& name, const char* code, const std::string& what) -> Verdict {
        if (!piece_identifier(name))
            return refuse(code, "The " + what + " " + name + " is not a WGSL name of a piece: letters, digits and "
                                "_, at most 63 bytes, without __ and without a caliper prefix.");
        if (!taken.insert(name).second) return refuse(code, "The name " + name + " of a " + what + " repeats.");
        return {};
    };
    for (const spec::PieceInput& in : piece.inputs) {
        if (Verdict v = claim(in.name, kPieceBadManifest, "input"); !v.ok()) return fail(v);
        if (in.kind != spec::InputKind::tensor) continue;
        const std::vector<std::string> types = spec::tensor_dtypes(in.dtype);
        if (types.empty())
            return fail(refuse(kPieceBadManifest, "The tensor input " + in.name + " has the dtype " + in.dtype +
                                                      ", which is not f32, i32, u32, u8 or u16, or several of them "
                                                      "joined by |."));
        const auto b = bound.find(in.name);
        if (b != bound.end() && std::find(types.begin(), types.end(), b->second) == types.end())
            return fail(refuse(kPieceBadBinding, "The input " + in.name + " reads " + in.dtype +
                                                     ", and the bound tensor holds " + b->second + "."));
        if (spec::typed_tensor_input(in.dtype))
            for (const char* suffix : {"_at", "_type"})
                if (Verdict v = claim(in.name + suffix, kPieceBadManifest, "accessor of an input"); !v.ok())
                    return fail(v);
    }
    std::set<std::string> param_names;
    for (const spec::PieceParam& p : piece.params)
        if (!piece_identifier(p.name) || !param_names.insert(p.name).second)
            return fail(refuse(kPieceBadManifest, "The param " + p.name + " is not a WGSL name of a piece, or it repeats."));

    std::string structs, resources;
    std::map<std::string, std::string> struct_of;   // name -> its declaration
    for (std::size_t i = 0; i < piece.resources.size(); ++i) {
        const spec::Resource& r = piece.resources[i];
        if (Verdict v = claim(r.id, kProgramBadResource, "resource"); !v.ok()) return fail(v);
        const std::uint64_t most = std::uint64_t(pl::kCountMax) *
                                   (r.layout.element ? spec::narrow_per_word(*r.layout.element) : 1u);
        if (r.kind == spec::ResourceKind::buffer && (r.count == 0 || r.count > most))
            return fail(refuse(kProgramBadResource, "The buffer " + r.id + " has a count outside 1 to " +
                                                        std::to_string(most) + "."));
        if (r.kind == spec::ResourceKind::texture &&
            (r.width == 0 || r.height == 0 || r.width > pl::kTextureSideMax || r.height > pl::kTextureSideMax))
            return fail(refuse(kProgramBadResource, "The texture " + r.id + " has a side outside 1 to 8,192."));
        if (r.kind == spec::ResourceKind::texture && r.format && !spec::texture_format_ok(*r.format))
            return fail(refuse(kProgramBadResource, "The texture " + r.id + " has the format " + *r.format +
                                                        ", which is not rgba8unorm, r8unorm or r32uint."));
        if (r.kind == spec::ResourceKind::texture && r.counted)
            return fail(refuse(kProgramBadResource, "The texture " + r.id + " cannot be counted."));
        std::string type, decl;
        if (Verdict v = element_of(r, &type, &decl); !v.ok()) return fail(v);
        if (!decl.empty()) {
            const auto [it, added] = struct_of.emplace(type, decl);
            if (added) {
                if (!taken.insert(type).second)
                    return fail(refuse(kProgramBadLayout, "The struct name " + type + " repeats another name."));
                structs += decl;
            } else if (it->second != decl) {
                return fail(refuse(kProgramBadLayout, "The struct " + type + " has two different field lists."));
            }
        }
        const std::string b = std::to_string(2 * i);
        resources += "@group(2) @binding(" + b + ") var<storage, read_write> " + r.id + ": array<" + type + ">;\n";
        if (r.counted) {
            if (Verdict v = claim(r.id + "_count", kProgramBadResource, "counter"); !v.ok()) return fail(v);
            resources += "@group(2) @binding(" + std::to_string(2 * i + 1) + ") var<storage, read_write> " + r.id +
                         "_count: atomic<u32>;\n";
        }
        // A texture's size in this version, which the runtime sets to the largest sides before each run.
        if (r.kind == spec::ResourceKind::texture) {
            if (Verdict v = claim(r.id + "_size", kProgramBadResource, "texture size"); !v.ok()) return fail(v);
            resources += "@group(2) @binding(" + std::to_string(2 * i + 1) + ") var<storage, read_write> " + r.id +
                         "_size: vec2<u32>;\n";
        }
    }

    // The header (design/programmable-graphics.md, Generated Header).
    std::string h;
    h += "struct CaliperFrame { step: u32, pass_index: u32, max_count: u32, width: u32, height: u32, time_s: f32, "
         "every: u32, pad: u32 }\n";
    h += "@group(0) @binding(0) var<uniform> caliper: CaliperFrame;\n";
    if (!piece.params.empty()) {
        h += "struct Params { ";
        for (std::size_t i = 0; i < piece.params.size(); ++i)
            h += (i ? ", " : "") + piece.params[i].name + ": " + param_type(piece.params[i].type);
        h += " }\n@group(0) @binding(1) var<uniform> params: Params;\n";
    }
    // A typed input (a narrow type or several types) binds its tensor's bytes as array<u32>, with the
    // constant <name>_type of the bound type, the accessor <name>_at() and its element count as the field
    // <name>__len of `shapes`, after the shapes of all tensor inputs.
    std::string shapes, lens, tensors, accessors;
    for (std::size_t i = 0; i < piece.inputs.size(); ++i) {
        const spec::PieceInput& in = piece.inputs[i];
        if (in.kind != spec::InputKind::tensor) continue;   // the draw half reads a buffer input
        shapes += (shapes.empty() ? "" : ", ") + in.name + ": vec4<u32>";
        if (!spec::typed_tensor_input(in.dtype)) {
            tensors += "@group(1) @binding(" + std::to_string(i) + ") var<storage, read> " + in.name + ": array<" +
                       in.dtype + ">;\n";
            continue;
        }
        const auto b = bound.find(in.name);
        const std::string type = b != bound.end() ? b->second : spec::tensor_dtypes(in.dtype).front();
        lens += ", " + in.name + "__len: u32";
        tensors += "@group(1) @binding(" + std::to_string(i) + ") var<storage, read> " + in.name + ": array<u32>;\n";
        accessors += "const " + in.name + "_type: u32 = " + dtype_constant(type) + ";\n";
        accessors += typed_accessor(in.name, type, spec::tensor_read_type(in.dtype));
    }
    if (!shapes.empty())
        h += "struct CaliperShapes { " + shapes + lens +
             " }\n@group(0) @binding(2) var<uniform> shapes: CaliperShapes;\n";
    h += tensors;
    if (!accessors.empty())
        h += "const CALIPER_F32: u32 = 0u;\nconst CALIPER_I32: u32 = 1u;\nconst CALIPER_U32: u32 = 2u;\n"
             "const CALIPER_U8: u32 = 3u;\nconst CALIPER_U16: u32 = 4u;\n" + accessors;
    h += structs;
    h += resources;

    // The module: the header, the lib modules, common.wgsl and derive.wgsl.
    std::uint32_t line = 1;
    auto add = [&](const std::string& part, const std::string& text) {
        const std::string t = with_final_newline(text);
        const std::uint32_t n = count_lines(t);
        c.segments.push_back({part, line, n});
        c.text += t;
        line += n;
    };
    add("header", h);
    if (!resolved.lib_text.empty()) {
        const std::string text = with_final_newline(resolved.lib_text);
        // A module that reads the draw prelude would not compile here: its lines stay, blank.
        std::vector<std::size_t> starts{0};   // the offset of each line; the last is the end
        for (std::size_t k = 0; k < text.size(); ++k)
            if (text[k] == '\n') starts.push_back(k + 1);
        std::string lib;
        std::size_t done = 0;
        for (const Segment& s : resolved.lib_segments) {
            const std::size_t from = s.first_line ? s.first_line - 1 : 0, to = from + s.lines;
            if (to >= starts.size() || starts[from] < done) continue;
            const std::string_view module = std::string_view(text).substr(starts[from], starts[to] - starts[from]);
            if (!reads_draw_prelude(module)) continue;
            const std::size_t slash = s.part.rfind('/') == std::string::npos ? 0 : s.part.rfind('/') + 1;
            const std::string import = s.part.substr(slash);
            const std::string name = import.substr(0, import.find('@'));
            for (const char* f : {"common.wgsl", "derive.wgsl"})
                if (auto it = resolved.files.find(f);
                    it != resolved.files.end() && it->second.find(name + "__") != std::string::npos)
                    return fail(refuse(kPieceUnknownImport, std::string(f) + " calls " + name + "::, but " + import +
                                                                " reads the draw prelude, which only the draw "
                                                                "half has; call it from draw.wgsl."));
            lib += text.substr(done, starts[from] - done);
            lib += blank_lines(module);
            done = starts[to];
        }
        lib += text.substr(done);
        for (const Segment& s : resolved.lib_segments)
            c.segments.push_back({s.part, line + (s.first_line ? s.first_line - 1 : 0), s.lines});
        c.text += lib;
        line += count_lines(lib);
    }
    const auto common = resolved.files.find("common.wgsl");
    if (common != resolved.files.end() && !common->second.empty()) add("common.wgsl", common->second);
    const auto derive = resolved.files.find("derive.wgsl");
    if (derive == resolved.files.end() || derive->second.empty())
        return fail(refuse(kPieceBadManifest, "The piece " + piece.name + " has a derive half without derive.wgsl."));
    add("derive.wgsl", derive->second);
    return c;
}

}  // namespace caliper_builder::programs
