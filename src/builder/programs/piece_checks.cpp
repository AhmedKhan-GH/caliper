// Layouts and the checks of a piece's manifest, its WebGPU limits and the
// bindings of an instance (programs/pieces.h, design/programmable-graphics.md:
// Piece Manifest, Generated Header, Check Order 2 and Check Codes). This file
// needs only the standard library and spec/piece.h, because the browser
// module compiles it too. Every size is bounded before it is multiplied, and
// every product is computed in 64 bits from factors below 2^32, so that no
// sum or product can wrap.
#include <caliper_builder/programs/pieces.h>

#include <cmath>
#include <optional>
#include <set>
#include <utility>

namespace caliper_builder::programs {

namespace {

namespace pl = spec::piece_limits;

Verdict fail(const char* code, std::string message, std::string where = {}) {
    Verdict v;
    v.code = code;
    v.message = std::move(message);
    v.where = std::move(where);
    return v;
}

std::string quote(std::string_view s) { return "\"" + std::string(s.substr(0, 64)) + "\""; }
std::string at(const std::string& path, std::size_t i) { return path + "[" + std::to_string(i) + "]"; }

bool lower(char c) { return c >= 'a' && c <= 'z'; }
bool upper(char c) { return c >= 'A' && c <= 'Z'; }
bool digit(char c) { return c >= '0' && c <= '9'; }

// [a-z][a-z0-9_]{0,31}: an id of an input, a parameter, a resource, a draw
// buffer or a target, which the header declares as a WGSL name.
bool id_form(std::string_view s) {
    if (s.empty() || s.size() > 32 || !lower(s[0])) return false;
    for (char c : s)
        if (!lower(c) && !digit(c) && c != '_') return false;
    return true;
}
// [a-z_][a-z0-9_]{0,31}: a field of a layout.
bool field_form(std::string_view s) {
    if (s.empty() || s.size() > 32 || !(lower(s[0]) || s[0] == '_')) return false;
    for (char c : s)
        if (!lower(c) && !digit(c) && c != '_') return false;
    return true;
}
// [A-Z][A-Za-z0-9_]{0,31}: the name of a generated struct.
bool struct_form(std::string_view s) {
    if (s.empty() || s.size() > 32 || !upper(s[0])) return false;
    for (char c : s)
        if (!lower(c) && !upper(c) && !digit(c) && c != '_') return false;
    return true;
}
// [A-Za-z_][A-Za-z0-9_]{0,62}: an entry point.
bool entry_form(std::string_view s) {
    if (s.empty() || s.size() > 63 || !(lower(s[0]) || upper(s[0]) || s[0] == '_')) return false;
    for (char c : s)
        if (!lower(c) && !upper(c) && !digit(c) && c != '_') return false;
    return true;
}

// The WGSL keywords and the predeclared names that an id of a manifest may
// not take, because the header declares the id as a module-scope name.
bool wgsl_word(std::string_view s) {
    static const std::set<std::string_view> words = {
        "alias", "array", "atomic", "bitcast", "bool", "break", "case", "const", "const_assert", "continue",
        "continuing", "default", "diagnostic", "discard", "else", "enable", "f16", "f32", "false", "fn", "for",
        "i32", "if", "let", "loop", "mat2x2", "mat2x3", "mat2x4", "mat3x2", "mat3x3", "mat3x4", "mat4x2", "mat4x3",
        "mat4x4", "override", "ptr", "requires", "return", "sampler", "sampler_comparison", "struct", "switch",
        "true", "u32", "var", "vec2", "vec3", "vec4", "while", "self", "super", "this", "unsafe", "in", "out",
        "uniform", "storage", "workgroup", "private", "function", "read", "write", "read_write", "main"};
    return words.count(s) > 0;
}

// The names that the header of a half declares beside the manifest's ids.
bool host_name(std::string_view s) {
    return s == "caliper" || s == "params" || s == "shapes" || s == "counts" || s.substr(0, 8) == "caliper_" ||
           s.find("__") != std::string_view::npos;
}

struct TypeInfo {
    std::uint32_t size = 0;
    std::uint32_t align = 0;
};
// The size and the alignment of a field type by the storage rules of WGSL.
std::optional<TypeInfo> plain_type(std::string_view t) {
    if (t == "f32" || t == "i32" || t == "u32") return TypeInfo{4, 4};
    if (t == "vec2<f32>" || t == "vec2<u32>") return TypeInfo{8, 8};
    if (t == "vec4<f32>" || t == "vec4<u32>") return TypeInfo{16, 16};
    return std::nullopt;
}
bool scalar(std::string_view t) { return t == "f32" || t == "i32" || t == "u32"; }

std::uint32_t round_up(std::uint32_t v, std::uint32_t align) { return (v + align - 1) / align * align; }

bool has_layout(const spec::Layout& l) { return l.element || l.type_name || !l.fields.empty(); }

// The bytes of `count` elements of `stride` bytes; both are below 2^32.
std::uint64_t bytes_of(std::uint32_t count, std::uint32_t stride) { return std::uint64_t(count) * stride; }

const spec::Resource* find_resource(const spec::Piece& p, std::string_view id) {
    for (const spec::Resource& r : p.resources)
        if (r.id == id) return &r;
    return nullptr;
}
const spec::PieceInput* find_input(const spec::Piece& p, std::string_view name) {
    for (const spec::PieceInput& in : p.inputs)
        if (in.name == name) return &in;
    return nullptr;
}

// A layout's field by name, or null.
const spec::Field* find_field(const spec::Layout& l, std::string_view name) {
    for (const spec::Field& f : l.fields)
        if (f.name == name) return &f;
    return nullptr;
}

bool is_import(std::string_view s, std::string* name) {
    const std::size_t a = s.find('@');
    if (a == std::string_view::npos || !id_form(s.substr(0, a))) return false;
    const std::string_view v = s.substr(a + 1);
    if (v.empty() || v.size() > 6 || v[0] == '0') return false;
    for (char c : v)
        if (!digit(c)) return false;
    if (name) *name = std::string(s.substr(0, a));
    return true;
}

class ManifestCheck {
public:
    explicit ManifestCheck(const spec::Piece& p) : p_(p) {}

    Verdict run() {
        Verdict v = check_piece_name(p_.name, p_.version);
        if (!v.ok()) return v;
        for (auto step : {&ManifestCheck::top, &ManifestCheck::names, &ManifestCheck::inputs, &ManifestCheck::params,
                          &ManifestCheck::resources, &ManifestCheck::imports, &ManifestCheck::derive,
                          &ManifestCheck::draw, &ManifestCheck::sample, &ManifestCheck::structs}) {
            v = (this->*step)();
            if (!v.ok()) return v;
        }
        return check_web_limits(p_);
    }

private:
    Verdict top() {
        if (p_.description.size() > pl::kDescriptionMax)
            return fail(kPieceBadManifest,
                        "The description has " + std::to_string(p_.description.size()) + " bytes; the limit is " +
                            std::to_string(pl::kDescriptionMax) + ".",
                        "description");
        for (char c : p_.description)
            if (static_cast<unsigned char>(c) < 0x20)
                return fail(kPieceBadManifest, "The description is one line without control characters.",
                            "description");
        const struct {
            const char* key;
            std::size_t n, limit;
        } lists[] = {{"inputs", p_.inputs.size(), pl::kInputs},       {"params", p_.params.size(), pl::kParams},
                     {"resources", p_.resources.size(), pl::kResources}, {"imports", p_.imports.size(), pl::kImports},
                     {"depends", p_.depends.size(), pl::kDepends}};
        for (const auto& l : lists)
            if (l.n > l.limit)
                return fail(kPieceBadManifest,
                            "The piece has " + std::to_string(l.n) + " " + l.key + "; the limit is " +
                                std::to_string(l.limit) + ".",
                            l.key);
        if (!p_.derive && !p_.draw)
            return fail(kPieceBadManifest, "A piece has a derive half, a draw half or both; this one has neither.");
        for (const auto& [file, text] : p_.files) {
            bool known = false;
            for (const char* f : spec::kPieceFiles) known = known || file == f;
            if (!known)
                return fail(kPieceBadPath,
                            "A piece's files are common.wgsl, derive.wgsl and draw.wgsl; " + quote(file) +
                                " is none of them.",
                            "files");
        }
        auto has = [&](const char* f) {
            auto it = p_.files.find(f);
            return it != p_.files.end() && !it->second.empty();
        };
        if (p_.derive && !has("derive.wgsl"))
            return fail(kPieceBadManifest, "The piece has a derive half but no derive.wgsl.", "files.derive");
        if (p_.draw && !has("draw.wgsl"))
            return fail(kPieceBadManifest, "The piece has a draw half but no draw.wgsl.", "files.draw");
        if (!p_.hash.empty()) {
            bool ok = p_.hash.size() == 71 && p_.hash.compare(0, 7, "sha256:") == 0;
            for (std::size_t i = 7; ok && i < p_.hash.size(); ++i)
                ok = digit(p_.hash[i]) || (p_.hash[i] >= 'a' && p_.hash[i] <= 'f');
            if (!ok)
                return fail(kPieceBadManifest, "A hash is sha256: and 64 lowercase hex digits.", "hash");
        }
        return {};
    }

    // Every name that the headers declare from the manifest is a WGSL name,
    // no host name, and unique among the inputs, the resources with their
    // counters, the draw buffers and the targets.
    Verdict names() {
        std::set<std::string> seen;
        auto one = [&](const std::string& name, const std::string& where, const char* code) -> Verdict {
            if (!id_form(name))
                return fail(code,
                            quote(name) + " is not an id: a lowercase letter, then at most 31 lowercase letters, "
                                          "digits and underscores.",
                            where);
            if (host_name(name) || wgsl_word(name))
                return fail(code,
                            quote(name) + " is a WGSL word or a name of the host (caliper, params, shapes, counts, "
                                          "caliper_*, or a name with \"__\"); choose another id.",
                            where);
            if (!seen.insert(name).second)
                return fail(code, "Two inputs, resources, draw buffers or targets of the piece have the id " +
                                      quote(name) + ".",
                            where);
            return {};
        };
        for (std::size_t i = 0; i < p_.inputs.size(); ++i)
            if (Verdict v = one(p_.inputs[i].name, at("inputs", i) + ".name", kPieceBadManifest); !v.ok()) return v;
        for (std::size_t i = 0; i < p_.resources.size(); ++i) {
            const spec::Resource& r = p_.resources[i];
            if (Verdict v = one(r.id, at("resources", i) + ".id", kProgramBadResource); !v.ok()) return v;
            if (r.counted && !seen.insert(r.id + "_count").second)
                return fail(kProgramBadResource,
                            "The counter " + r.id + "_count of the resource " + quote(r.id) +
                                " has the name of another id of the piece.",
                            at("resources", i) + ".counted");
        }
        if (p_.draw) {
            for (std::size_t i = 0; i < p_.draw->buffers.size(); ++i)
                if (Verdict v = one(p_.draw->buffers[i].id, at("draw.buffers", i) + ".id", kPieceBadManifest); !v.ok())
                    return v;
            for (std::size_t i = 0; i < p_.draw->targets.size(); ++i)
                if (Verdict v = one(p_.draw->targets[i].id, at("draw.targets", i) + ".id", kPieceBadManifest); !v.ok())
                    return v;
            if (seen.count("host"))
                return fail(kPieceBadManifest, "The id host names the host target of a pass; choose another id.");
        }
        return {};
    }

    Verdict inputs() {
        for (std::size_t i = 0; i < p_.inputs.size(); ++i) {
            const spec::PieceInput& in = p_.inputs[i];
            const std::string where = at("inputs", i);
            if (in.kind == spec::InputKind::tensor) {
                if (spec::tensor_dtypes(in.dtype).empty())
                    return fail(kPieceBadManifest,
                                "A tensor input's dtype is f32, i32, u32, u8 or u16, or two to five of them joined by "
                                "|, each once.",
                                where + ".dtype");
                if (in.rank_min > in.rank_max || in.rank_max > 8)
                    return fail(kPieceBadManifest, "A tensor input's ranks satisfy rank_min <= rank_max <= 8.",
                                where + ".rank_max");
                if (has_layout(in.layout))
                    return fail(kPieceBadManifest, "A tensor input has no element, struct or fields.", where);
                if (!p_.derive)
                    return fail(kPieceBadManifest,
                                "A tensor input is read by the derive half, and the piece has none.", where + ".kind");
            } else {
                ElementLayout l;
                if (Verdict v = element_layout(in.layout, &l); !v.ok()) {
                    v.where = where + (v.where.empty() ? "" : "." + v.where);
                    return v;
                }
                if (!p_.draw)
                    return fail(kPieceBadManifest, "A buffer input is read by the draw half, and the piece has none.",
                                where + ".kind");
            }
            if (in.title && (in.title->empty() || in.title->size() > 256))
                return fail(kPieceBadManifest, "A title has 1 to 256 bytes.", where + ".title");
        }
        return {};
    }

    Verdict params() {
        std::set<std::string> seen;
        for (std::size_t i = 0; i < p_.params.size(); ++i) {
            const spec::PieceParam& p = p_.params[i];
            const std::string where = at("params", i);
            if (!id_form(p.name) || wgsl_word(p.name) || p.name.find("__") != std::string::npos)
                return fail(kPieceBadManifest, quote(p.name) + " is not a parameter name.", where + ".name");
            if (!seen.insert(p.name).second)
                return fail(kPieceBadManifest, "Two parameters have the name " + quote(p.name) + ".", where + ".name");
            const std::pair<const char*, std::optional<double>> values[] = {
                {"default", p.default_value}, {"min", p.min}, {"max", p.max}};
            for (const auto& [key, value] : values) {
                if (!value) continue;
                const double x = *value;
                bool ok = std::isfinite(x);
                if (ok && p.type == spec::ParamType::i32) ok = x == std::floor(x) && x >= -2147483648.0 && x <= 2147483647.0;
                if (ok && p.type == spec::ParamType::u32) ok = x == std::floor(x) && x >= 0 && x <= 4294967295.0;
                if (ok && p.type == spec::ParamType::f32) ok = std::fabs(x) <= 3.4028234663852886e38;
                if (!ok)
                    return fail(kPieceBadManifest,
                                std::string("The ") + key + " of the parameter " + quote(p.name) +
                                    " is not a value of its type " + spec::param_type_name(p.type) + ".",
                                where + "." + key);
            }
            if (p.min && p.max && *p.min > *p.max)
                return fail(kPieceBadManifest, "The parameter " + quote(p.name) + " has min above max.", where + ".min");
            if ((p.min && p.default_value < *p.min) || (p.max && p.default_value > *p.max))
                return fail(kPieceBadManifest,
                            "The default of the parameter " + quote(p.name) + " is outside its range.",
                            where + ".default");
            if (p.title && (p.title->empty() || p.title->size() > 256))
                return fail(kPieceBadManifest, "A title has 1 to 256 bytes.", where + ".title");
        }
        return {};
    }

    Verdict resources() {
        std::uint64_t published = 0;   // bytes of all published resources: what one pass sends to a viewer
        for (std::size_t i = 0; i < p_.resources.size(); ++i) {
            const spec::Resource& r = p_.resources[i];
            const std::string where = at("resources", i);
            if (r.keep && (*r.keep < 1 || *r.keep > pl::kKeepMax))
                return fail(kProgramBadResource, "keep is from 1 to " + std::to_string(pl::kKeepMax) + ".",
                            where + ".keep");
            if (r.kind == spec::ResourceKind::buffer) {
                ElementLayout l;
                if (Verdict v = element_layout(r.layout, &l); !v.ok()) {
                    v.where = where + (v.where.empty() ? "" : "." + v.where);
                    return v;
                }
                // kCountMax words: four u8 or two u16 elements to a word.
                const std::uint64_t most = std::uint64_t(pl::kCountMax) * l.per_word;
                if (r.count < 1 || r.count > most)
                    return fail(kProgramBadResource,
                                "The buffer " + quote(r.id) + " has count " + std::to_string(r.count) +
                                    "; a buffer has 1 to " + std::to_string(most) + " elements.",
                                where + ".count");
                // A published buffer goes out with each pass that writes it, and a viewer binds it whole: the
                // largest storage binding of WebGPU's default limits.
                const std::uint64_t max_bytes = r.publish ? pl::kPublishedResourceBytesMax : pl::kResourceBytesMax;
                if (layout_bytes(l, r.count) > max_bytes)
                    return fail(kProgramBadResource,
                                "The buffer " + quote(r.id) + " holds " + std::to_string(layout_bytes(l, r.count)) +
                                    " bytes; a " + (r.publish ? "published" : "private") + " resource holds at most " +
                                    std::to_string(max_bytes) + " bytes.",
                                where + ".count");
                if (r.width || r.height || r.format || r.texels)
                    return fail(kProgramBadResource, "A buffer has no width, height, format or texels.",
                                where + ".width");
                if (r.publish) published += layout_bytes(l, r.count);
            } else {
                if (r.layout.element || r.layout.type_name || r.count || r.counted)
                    return fail(kProgramBadResource,
                                "A texture has a format and a size, and no element, struct, count or counted.", where);
                if (!r.layout.fields.empty()) {
                    const auto& fields = r.layout.fields;
                    if (spec::texture_format(r) != "r32uint" || fields.size() != 1 ||
                        !field_form(fields[0].name) || fields[0].name.find("__") != std::string::npos ||
                        (fields[0].type != "f32" && fields[0].type != "i32" && fields[0].type != "u32"))
                        return fail(kProgramBadResource,
                                    "An r32uint texture can name one scalar field of type f32, i32 or u32.",
                                    where + ".fields");
                }
                if (r.format && !spec::texture_format_ok(*r.format))
                    return fail(kProgramBadResource,
                                "A texture's format is rgba8unorm, r8unorm or r32uint, not " + quote(*r.format) + ".",
                                where + ".format");
                if (r.width < 1 || r.width > pl::kTextureSideMax || r.height < 1 || r.height > pl::kTextureSideMax)
                    return fail(kProgramBadResource,
                                "A texture's width and height are from 1 to " + std::to_string(pl::kTextureSideMax) +
                                    " pixels.",
                                where + ".width");
                if (r.texels && (*r.texels < 1 || *r.texels > std::uint64_t(r.width) * r.height))
                    return fail(kProgramBadResource,
                                "A texture's texels are from 1 to its width times its height.", where + ".texels");
                // At most 2^26 texels of 4 bytes: the product cannot wrap.
                const std::uint64_t bytes = spec::texture_bytes(r);
                const std::uint64_t max_bytes = r.publish ? pl::kPublishedTextureBytesMax : pl::kResourceBytesMax;
                if (bytes > max_bytes)
                    return fail(kProgramBadResource,
                                "The texture " + quote(r.id) + " holds " + std::to_string(bytes) + " bytes; a " +
                                    (r.publish ? "published" : "private") + " resource holds at most " +
                                    std::to_string(max_bytes) + " bytes. Give texels to hold fewer.",
                                where + ".texels");
                if (r.publish) published += bytes;
            }
        }
        if (published > pl::kPublishedBytesMax)
            return fail(kProgramBadResource,
                        "The published resources of the piece hold " + std::to_string(published) +
                            " bytes together; a pass sends at most " + std::to_string(pl::kPublishedBytesMax) +
                            " bytes to a viewer, one version of each in the step history.",
                        "resources");
        return {};
    }

    Verdict imports() {
        std::set<std::string> names;
        for (std::size_t i = 0; i < p_.imports.size(); ++i) {
            std::string name;
            if (!is_import(p_.imports[i], &name) || lib_module(p_.imports[i]).empty()) {
                std::string known;
                for (std::string_view m : lib_modules()) known += (known.empty() ? "" : ", ") + std::string(m);
                return fail(kPieceUnknownImport,
                            "There is no lib module " + quote(p_.imports[i]) + "; the lib modules are " + known + ".",
                            at("imports", i));
            }
            if (!names.insert(name).second)
                return fail(kPieceBadManifest, "The piece imports the module " + name + " twice; import one version.",
                            at("imports", i));
        }
        std::set<std::pair<std::string, std::uint32_t>> deps;
        for (std::size_t i = 0; i < p_.depends.size(); ++i) {
            const spec::PieceDependency& d = p_.depends[i];
            Verdict v = check_piece_name(d.name, d.version);
            if (!v.ok()) return fail(kPieceBadManifest, v.message, at("depends", i) + "." + v.where);
            if (d.name == p_.name)
                return fail(kPieceCycle, "The piece depends on itself.", at("depends", i));
            if (!deps.insert({d.name, d.version}).second)
                return fail(kPieceBadManifest, "The piece names the dependency " + d.name + " twice.", at("depends", i));
        }
        return {};
    }

    static Verdict entry(const std::string& name, const std::string& where) {
        if (!entry_form(name))
            return fail(kProgramBadPass, quote(name) + " is not an entry point name.", where);
        if (name == "main")
            return fail(kProgramBadPass, "An entry point of a piece is never named main.", where);
        return {};
    }

    static Verdict workgroups(const std::array<std::uint32_t, 3>& w, const std::string& where) {
        for (std::uint32_t n : w)
            if (n < 1 || n > pl::kWorkgroupsMax)
                return fail(kProgramBadPass,
                            "Each workgroup count is from 1 to " + std::to_string(pl::kWorkgroupsMax) + ".", where);
        return {};
    }

    Verdict derive() {
        if (!p_.derive) return {};
        const spec::Derive& d = *p_.derive;
        if (d.passes.empty() || d.passes.size() > pl::kDerivePasses)
            return fail(kPieceBadManifest,
                        "The derive half has " + std::to_string(d.passes.size()) + " passes; it has 1 to " +
                            std::to_string(pl::kDerivePasses) + ".",
                        "derive.passes");
        for (std::size_t i = 0; i < d.passes.size(); ++i) {
            const spec::DerivePass& pass = d.passes[i];
            const std::string where = at("derive.passes", i);
            if (Verdict v = entry(pass.entry, where + ".entry"); !v.ok()) return v;
            if (Verdict v = workgroups(pass.workgroups, where + ".workgroups"); !v.ok()) return v;
            std::set<std::string> writes;
            for (std::size_t k = 0; k < pass.writes.size(); ++k) {
                if (!find_resource(p_, pass.writes[k]))
                    return fail(kProgramUnknownResource,
                                "The pass writes " + quote(pass.writes[k]) + ", which is no resource of the piece.",
                                at(where + ".writes", k));
                if (!writes.insert(pass.writes[k]).second)
                    return fail(kProgramBadPass, "The pass names " + quote(pass.writes[k]) + " twice in writes.",
                                at(where + ".writes", k));
            }
        }
        if (d.time_ms && !(*d.time_ms >= 0.01 && *d.time_ms <= 50.0))
            return fail(kPieceBadManifest, "derive.time_ms is from 0.01 to 50.", "derive.time_ms");
        if (d.memory_mb && (*d.memory_mb < 1 || *d.memory_mb > 512))
            return fail(kPieceBadManifest, "derive.memory_mb is from 1 to 512.", "derive.memory_mb");
        if (d.every && (*d.every < 1 || *d.every > 10000))
            return fail(kPieceBadManifest, "derive.every is from 1 to 10000.", "derive.every");
        return {};
    }

    // The buffer that an instanced pass counts: a published buffer resource
    // or a buffer input.
    bool countable(const std::string& id) const {
        if (const spec::Resource* r = find_resource(p_, id))
            return r->kind == spec::ResourceKind::buffer && r->publish;
        const spec::PieceInput* in = find_input(p_, id);
        return in && in->kind == spec::InputKind::buffer;
    }

    Verdict draw() {
        if (!p_.draw) return {};
        const spec::Draw& d = *p_.draw;
        if (d.frame_ms && !(*d.frame_ms >= 0.1 && *d.frame_ms <= pl::kFrameMsMax))
            return fail(kPieceBadManifest, "draw.frame_ms is from 0.1 to 16.", "draw.frame_ms");
        if (d.memory_mb && (*d.memory_mb < 1 || *d.memory_mb > pl::kDrawMemoryMbMax))
            return fail(kPieceBadManifest, "draw.memory_mb is from 1 to 512.", "draw.memory_mb");
        if (d.buffers.size() > pl::kDrawBuffers)
            return fail(kPieceBadManifest, "The draw half keeps at most 4 buffers.", "draw.buffers");
        if (d.targets.size() > pl::kTargets)
            return fail(kPieceBadManifest, "The draw half has at most 4 targets.", "draw.targets");
        std::set<std::string> buffers, targets;
        for (std::size_t i = 0; i < d.buffers.size(); ++i) {
            const spec::DrawBuffer& b = d.buffers[i];
            const std::string where = at("draw.buffers", i);
            ElementLayout l;
            if (Verdict v = element_layout(b.layout, &l); !v.ok()) {
                v.where = where + (v.where.empty() ? "" : "." + v.where);
                return v;
            }
            if (l.per_word > 1)
                return fail(kProgramBadLayout, "A draw buffer has no narrow element u8 or u16.", where + ".element");
            if (b.count < 1 || b.count > pl::kCountMax || bytes_of(b.count, l.stride) > pl::kResourceBytesMax)
                return fail(kPieceBadManifest,
                            "A draw buffer has 1 to " + std::to_string(pl::kCountMax) + " elements and at most " +
                                std::to_string(pl::kResourceBytesMax) + " bytes.",
                            where + ".count");
            buffers.insert(b.id);
        }
        for (std::size_t i = 0; i < d.targets.size(); ++i) {
            const spec::DrawTarget& t = d.targets[i];
            if (t.scale != 1.0f && t.scale != 0.5f && t.scale != 0.25f)
                return fail(kPieceBadManifest, "A target's scale is 1, 0.5 or 0.25.", at("draw.targets", i) + ".scale");
            for (float c : t.clear)
                if (!(c >= 0.0f && c <= 1.0f))
                    return fail(kPieceBadManifest, "A target's clear color has four values from 0 to 1.",
                                at("draw.targets", i) + ".clear");
            targets.insert(t.id);
        }
        if (d.passes.empty() || d.passes.size() > pl::kDrawPasses)
            return fail(kPieceBadManifest,
                        "The draw half has " + std::to_string(d.passes.size()) + " passes; it has 1 to " +
                            std::to_string(pl::kDrawPasses) + ".",
                        "draw.passes");
        std::set<std::string> written;  // targets that an earlier pass wrote
        bool pick_pass = false;
        for (std::size_t i = 0; i < d.passes.size(); ++i) {
            const spec::DrawPass& pass = d.passes[i];
            const std::string where = at("draw.passes", i);
            if (pass.kind == spec::PassKind::compute) {
                if (Verdict v = entry(pass.entry, where + ".entry"); !v.ok()) return v;
                if (Verdict v = workgroups(pass.workgroups, where + ".workgroups"); !v.ok()) return v;
                if (pass.vertex || !pass.fragment.empty() || pass.target != "host" || pass.instances != 1 ||
                    pass.instances_of || pass.vertices || pass.topology != spec::DrawTopology::triangle_list ||
                    pass.blend || pass.depth || pass.pick || pass.indices)
                    return fail(kProgramBadPass,
                                "A compute pass has entry, workgroups, writes and reads, and none of the keys of a "
                                "render pass.",
                                where);
                std::set<std::string> writes;
                for (std::size_t k = 0; k < pass.writes.size(); ++k) {
                    if (!buffers.count(pass.writes[k]))
                        return fail(kProgramUnknownResource,
                                    "The pass writes " + quote(pass.writes[k]) +
                                        ", which is no draw buffer of the piece.",
                                    at(where + ".writes", k));
                    if (!writes.insert(pass.writes[k]).second)
                        return fail(kProgramBadPass, "The pass names " + quote(pass.writes[k]) + " twice in writes.",
                                    at(where + ".writes", k));
                }
            } else {
                if (!pass.entry.empty() || !pass.writes.empty() || pass.workgroups != std::array<std::uint32_t, 3>{1, 1, 1})
                    return fail(kProgramBadPass, "A render pass has no entry, workgroups or writes.", where);
                if (Verdict v = entry(pass.fragment, where + ".fragment"); !v.ok()) return v;
                if (pass.vertex) {
                    if (Verdict v = entry(*pass.vertex, where + ".vertex"); !v.ok()) return v;
                    if (!pass.vertices && !pass.indices)
                        return fail(kProgramBadPass, "A pass with a vertex entry gives vertices or indices.",
                                    where + ".vertices");
                } else {
                    if ((pass.vertices && *pass.vertices != 3) || pass.topology != spec::DrawTopology::triangle_list ||
                        pass.instances != 1 || pass.instances_of)
                        return fail(kProgramBadPass,
                                    "A pass without a vertex entry draws one full-screen triangle: 3 vertices, one "
                                    "instance, triangle_list.",
                                    where);
                }
                if (pass.vertices && (*pass.vertices < 1 || *pass.vertices > pl::kVerticesMax))
                    return fail(kProgramBadPass,
                                "vertices is from 1 to " + std::to_string(pl::kVerticesMax) + ".", where + ".vertices");
                if (pass.indices) {
                    // An index buffer: each index becomes the vertex entry's vertex_index, whose every
                    // read of a buffer is bounds-checked. A strip would read the restart index.
                    if (!pass.vertex)
                        return fail(kProgramBadPass, "A pass with indices has a vertex entry.", where + ".indices");
                    if (pass.topology == spec::DrawTopology::triangle_strip ||
                        pass.topology == spec::DrawTopology::line_strip)
                        return fail(kProgramBadPass,
                                    "A pass with indices draws a list: triangle_list, line_list or point_list.",
                                    where + ".topology");
                    if (!countable(*pass.indices))
                        return fail(kProgramUnknownResource,
                                    "indices names " + quote(*pass.indices) +
                                        ", which is no published buffer resource or buffer input of the piece.",
                                    where + ".indices");
                    const spec::Resource* r = find_resource(p_, *pass.indices);
                    const spec::PieceInput* in = r ? nullptr : find_input(p_, *pass.indices);
                    const spec::Layout& l = r ? r->layout : in->layout;
                    if (!l.element || *l.element != "u32" || !l.fields.empty())
                        return fail(kProgramBadPass, "indices names " + quote(*pass.indices) + ", whose elements are not u32.",
                                    where + ".indices");
                }
                if (pass.instances_of) {
                    const spec::Resource* resource = find_resource(p_, *pass.instances_of);
                    if (!countable(*pass.instances_of) && !(resource && resource->publish))
                        return fail(kProgramUnknownResource,
                                    "instances names " + quote(*pass.instances_of) +
                                        ", which is no published resource or buffer input of the piece.",
                                    where + ".instances");
                } else if (pass.instances < 1 || pass.instances > pl::kInstancesMax) {
                    return fail(kProgramBadPass,
                                "instances is from 1 to " + std::to_string(pl::kInstancesMax) + ", or a buffer's id.",
                                where + ".instances");
                }
                if (pass.target != "host") {
                    if (!targets.count(pass.target))
                        return fail(kProgramBadPass,
                                    "The pass draws into " + quote(pass.target) + ", which is not host or a target of "
                                                                                  "the piece.",
                                    where + ".target");
                    if (pass.depth && *pass.depth != spec::DrawDepth::none)
                        return fail(kProgramBadPass, "An offscreen target has no depth; the pass's depth is none.",
                                    where + ".depth");
                    if (pass.pick)
                        return fail(kProgramBadPass, "Only a pass into the host target writes the pick ID.",
                                    where + ".pick");
                }
                pick_pass = pick_pass || pass.pick;
            }
            std::set<std::string> reads;
            for (std::size_t k = 0; k < pass.reads.size(); ++k) {
                const std::string& r = pass.reads[k];
                if (!targets.count(r))
                    return fail(kProgramBadPass, "The pass reads " + quote(r) + ", which is no target of the piece.",
                                at(where + ".reads", k));
                if (!reads.insert(r).second)
                    return fail(kProgramBadPass, "The pass names " + quote(r) + " twice in reads.", at(where + ".reads", k));
                if (pass.kind == spec::PassKind::render && r == pass.target)
                    return fail(kProgramTargetCycle, "The pass reads its own target " + quote(r) + ".",
                                at(where + ".reads", k));
                if (!written.count(r))
                    return fail(kProgramTargetCycle,
                                "The pass reads the target " + quote(r) + " before a pass draws into it.",
                                at(where + ".reads", k));
            }
            if (pass.kind == spec::PassKind::render && pass.target != "host") written.insert(pass.target);
        }
        if (d.pick) {
            const spec::Resource* r = find_resource(p_, d.pick->resource);
            if (!r || !r->publish)
                return fail(kProgramUnknownResource,
                            "The item map names " + quote(d.pick->resource) +
                                ", which is no published buffer or texture resource of the piece.",
                            "draw.pick.resource");
            if (d.pick->fields.empty() || d.pick->fields.size() > pl::kPickFields)
                return fail(kProgramBadPick, "The item map names 1 to 8 fields.", "draw.pick.fields");
            // A texture's item is one texel in row order. R32Uint can declare its scalar field.
            // Without a declaration, RGBA8 uses rgba and R32Uint uses value.
            if (r->kind == spec::ResourceKind::texture) {
                const std::string format = spec::texture_format(*r);
                const char* field = format == "rgba8unorm" ? "rgba" : format == "r32uint" ? "value" : nullptr;
                if (!r->layout.fields.empty()) field = r->layout.fields[0].name.c_str();
                if (!field || d.pick->fields.size() != 1 || d.pick->fields[0] != field)
                    return fail(kProgramBadPick,
                                "The item map of a texture names its one field: rgba for rgba8unorm, or the declared "
                                "field of r32uint, with value as its default. An r8unorm texture has none.",
                                "draw.pick.fields");
            }
            std::set<std::string> fields;
            for (std::size_t k = 0; r->kind == spec::ResourceKind::buffer && k < d.pick->fields.size(); ++k) {
                const spec::Field* f = find_field(r->layout, d.pick->fields[k]);
                if (!f || !scalar(f->type))
                    return fail(kProgramBadPick,
                                "The item map names " + quote(d.pick->fields[k]) + ", which is no f32, i32 or u32 field "
                                                                                    "of the resource " +
                                    quote(r->id) + ".",
                                at("draw.pick.fields", k));
                if (!fields.insert(f->name).second)
                    return fail(kProgramBadPick, "The item map names " + quote(f->name) + " twice.",
                                at("draw.pick.fields", k));
            }
            if (!pick_pass)
                return fail(kProgramBadPick, "The piece has an item map but no pass with pick = true.", "draw.pick");
        } else if (pick_pass) {
            return fail(kProgramBadPick, "A pass writes the pick ID, and the piece has no item map (draw.pick).",
                        "draw.pick");
        }
        if (d.text) {
            const spec::TextTable& t = *d.text;
            const spec::Resource* r = find_resource(p_, t.anchors);
            if (!r || r->kind != spec::ResourceKind::buffer || !r->publish)
                return fail(kProgramUnknownResource,
                            "The text table's anchors " + quote(t.anchors) +
                                " is no published buffer resource of the piece.",
                            "draw.text.anchors");
            const struct {
                const char* key;
                std::optional<std::string> field;
                const char* type;
            } want[] = {{"position", t.position, "vec4<f32>"}, {"string", t.string, "u32"}, {"priority", t.priority, "f32"}};
            for (const auto& w : want) {
                if (!w.field) continue;
                const spec::Field* f = find_field(r->layout, *w.field);
                if (!f || f->type != w.type)
                    return fail(kProgramBadText,
                                std::string("The anchors need the ") + w.type + " field " + quote(*w.field) + " for " +
                                    w.key + ".",
                                std::string("draw.text.") + w.key);
            }
            if (t.strings.size() > pl::kStrings)
                return fail(kProgramBadText, "The text table has at most 4096 strings.", "draw.text.strings");
            if (t.strings_from) {
                // The strings of a run: each element of a published buffer is one string of at most
                // kStringBytes UTF-8 bytes, which the host checks again when it reads them.
                if (!t.strings.empty())
                    return fail(kProgramBadText, "The text table takes its strings from strings or strings_from, not both.",
                                "draw.text.strings_from");
                if (!countable(*t.strings_from))
                    return fail(kProgramUnknownResource,
                                "strings_from names " + quote(*t.strings_from) +
                                    ", which is no published buffer resource or buffer input of the piece.",
                                "draw.text.strings_from");
                const spec::Resource* sr = find_resource(p_, *t.strings_from);
                const spec::Layout& l = sr ? sr->layout : find_input(p_, *t.strings_from)->layout;
                ElementLayout el;
                bool bytes = element_layout(l, &el).ok() && el.stride > 0 && el.stride <= pl::kStringBytes;
                for (const FieldLayout& f : el.fields)
                    bytes = bytes && (f.type == "u32" || f.type == "vec2<u32>" || f.type == "vec4<u32>");
                if (!bytes)
                    return fail(kProgramBadText,
                                "The strings of " + quote(*t.strings_from) +
                                    " need elements of u32, vec2<u32> or vec4<u32> fields, at most 64 bytes each.",
                                "draw.text.strings_from");
                if (sr && sr->count > pl::kStrings)
                    return fail(kProgramBadText, "The strings of " + quote(*t.strings_from) + " are at most 4096.",
                                "draw.text.strings_from");
            }
            for (std::size_t k = 0; k < t.strings.size(); ++k)
                if (t.strings[k].size() > pl::kStringBytes)
                    return fail(kProgramBadText, "A string of the text table has at most 64 bytes.",
                                at("draw.text.strings", k));
            if (t.selected) {
                const auto param = std::find_if(p_.params.begin(), p_.params.end(), [&](const spec::PieceParam& value) {
                    return value.name == t.selected->param;
                });
                if (param == p_.params.end() || param->type == spec::ParamType::f32)
                    return fail(kProgramBadText, "The selected label requires an i32 or u32 parameter.",
                                "draw.text.selected.param");
                if (t.hover && t.hover->anchor == t.selected->anchor)
                    return fail(kProgramBadText, "The selected and hover labels require distinct anchors.",
                                "draw.text.selected.anchor");
            }
            const auto check_item_label = [&](std::uint32_t anchor, std::uint32_t offset, const std::string& where) -> Verdict {
                if (anchor >= r->count || anchor >= pl::kLabels)
                    return fail(kProgramBadText, "The item label anchor must fit the anchor buffer and the 512-label limit.",
                                where + ".anchor");
                const spec::Resource* items = d.pick ? find_resource(p_, d.pick->resource) : nullptr;
                if (!items || items->kind != spec::ResourceKind::buffer)
                    return fail(kProgramBadText, "An item label requires a buffer item map.", where);
                std::uint64_t strings = t.strings.size();
                if (t.strings_from) {
                    const spec::Resource* source = find_resource(p_, *t.strings_from);
                    if (!source)
                        return fail(kProgramBadText, "Item label strings require a resource with a declared count.",
                                    "draw.text.strings_from");
                    strings = source->count;
                }
                if (std::uint64_t(offset) + items->count > strings)
                    return fail(kProgramBadText, "The item label string offset and item count must fit the string table.",
                                where + ".string_offset");
                return {};
            };
            if (t.hover) {
                const auto checked = check_item_label(t.hover->anchor, t.hover->string_offset, "draw.text.hover");
                if (!checked.ok()) return checked;
            }
            if (t.selected) {
                const auto checked = check_item_label(t.selected->anchor, t.selected->string_offset, "draw.text.selected");
                if (!checked.ok()) return checked;
            }
        }
        return {};
    }

    Verdict sample() {
        std::uint64_t total = 0;
        for (const auto& [input, s] : p_.sample) {
            const std::string where = "sample." + input;
            const spec::PieceInput* in = find_input(p_, input);
            if (!in) return fail(kPieceBadManifest, "The sample names " + quote(input) + ", which is no input.", where);
            if (s.file != "sample/" + input + ".bin")
                return fail(kPieceBadPath, "The sample of " + quote(input) + " is the file sample/" + input + ".bin.",
                            where);
            if (s.shape.empty() || s.shape.size() > 8)
                return fail(kPieceBadManifest, "A sample's shape has 1 to 8 axes.", where + ".shape");
            std::uint32_t element = 4;
            if (in->kind == spec::InputKind::tensor) {
                if (s.shape.size() < in->rank_min || s.shape.size() > in->rank_max)
                    return fail(kPieceBadManifest, "The sample's rank is outside the input's ranks.", where + ".shape");
            } else {
                ElementLayout l;
                if (s.shape.size() != 1 || !element_layout(in->layout, &l).ok() || l.per_word > 1)
                    return fail(kPieceBadManifest,
                                "A buffer input's sample has one axis, its count, and no narrow element.",
                                where + ".shape");
                element = l.stride;
            }
            std::uint64_t bytes = element;
            for (std::uint32_t n : s.shape) {
                if (n < 1) return fail(kPieceBadManifest, "Each axis of a sample has at least 1 element.", where);
                bytes *= n;  // below 2^32 times at most 2^20: no wrap
                if (bytes > pl::kSampleBytes) break;
            }
            total += bytes;
            if (total > pl::kSampleBytes)
                return fail(kPieceBadPath, "The samples of a piece hold at most 1 MiB together.", where);
        }
        return {};
    }

    // A struct name that two layouts share must have the same fields in both,
    // because the composer generates it once.
    Verdict structs() {
        std::map<std::string, const spec::Layout*> seen;
        auto one = [&](const spec::Layout& l, const std::string& where) -> Verdict {
            if (!l.type_name) return {};
            auto [it, fresh] = seen.emplace(*l.type_name, &l);
            if (!fresh && it->second->fields != l.fields)
                return fail(kProgramBadLayout,
                            "Two layouts name the struct " + *l.type_name + " with different fields.", where + ".struct");
            return {};
        };
        for (std::size_t i = 0; i < p_.inputs.size(); ++i)
            if (Verdict v = one(p_.inputs[i].layout, at("inputs", i)); !v.ok()) return v;
        for (std::size_t i = 0; i < p_.resources.size(); ++i)
            if (Verdict v = one(p_.resources[i].layout, at("resources", i)); !v.ok()) return v;
        if (p_.draw)
            for (std::size_t i = 0; i < p_.draw->buffers.size(); ++i)
                if (Verdict v = one(p_.draw->buffers[i].layout, at("draw.buffers", i)); !v.ok()) return v;
        return {};
    }

    const spec::Piece& p_;
};

}  // namespace

Verdict element_layout(const spec::Layout& layout, ElementLayout* out) {
    ElementLayout l;
    if (out) *out = l;
    if (layout.element) {
        if (layout.type_name || !layout.fields.empty())
            return fail(kProgramBadLayout, "A layout gives either element, or struct with fields; not both.",
                        "element");
        if (const std::uint32_t per = spec::narrow_per_word(*layout.element); per > 1) {
            // Narrow elements, packed into 4-byte words from the low bits up.
            l.stride = 4;
            l.align = 4;
            l.per_word = per;
            l.fields.push_back(FieldLayout{std::string(), *layout.element, 0, 4});
            if (out) *out = l;
            return {};
        }
        const auto t = plain_type(*layout.element);
        if (!t)
            return fail(kProgramBadLayout,
                        quote(*layout.element) + " is not an element type: f32, i32, u32, vec2<f32>, vec4<f32>, "
                                                 "vec2<u32>, vec4<u32>, or the narrow u8 or u16.",
                        "element");
        l.stride = t->size;
        l.align = t->align;
        l.fields.push_back(FieldLayout{std::string(), *layout.element, 0, t->size});
        if (out) *out = l;
        return {};
    }
    if (!layout.type_name)
        return fail(kProgramBadLayout, "A buffer gives its element, or struct with fields.", "element");
    const std::string& name = *layout.type_name;
    if (!struct_form(name) || name.rfind("Caliper", 0) == 0 || name == "Params" || name.find("__") != std::string::npos)
        return fail(kProgramBadLayout,
                    quote(name) + " is not a struct name: an uppercase letter, then at most 31 letters, digits and "
                                  "underscores, and not Params, Caliper* or a name with \"__\".",
                    "struct");
    if (layout.fields.empty() || layout.fields.size() > pl::kFields)
        return fail(kProgramBadLayout, "A struct has 1 to 16 fields.", "fields");
    std::set<std::string> names;
    std::uint32_t offset = 0;
    l.align = 4;
    for (std::size_t i = 0; i < layout.fields.size(); ++i) {
        const spec::Field& f = layout.fields[i];
        if (!field_form(f.name) || f.name.find("__") != std::string::npos || !names.insert(f.name).second)
            return fail(kProgramBadLayout,
                        quote(f.name) + " is not a field name, or two fields have it; a field is written name: type.",
                        at("fields", i));
        const auto t = plain_type(f.type);
        if (!t)
            return fail(kProgramBadLayout,
                        "The field " + quote(f.name) + " has the type " + quote(f.type) +
                            "; a field is f32, i32, u32, vec2<f32>, vec4<f32>, vec2<u32> or vec4<u32>.",
                        at("fields", i));
        offset = round_up(offset, t->align);
        l.fields.push_back(FieldLayout{f.name, f.type, offset, t->size});
        offset += t->size;  // at most 16 fields of at most 16 bytes: no wrap
        if (t->align > l.align) l.align = t->align;
    }
    l.stride = round_up(offset, l.align);
    if (l.stride > pl::kStrideMax)
        return fail(kProgramBadLayout,
                    "The struct " + name + " takes " + std::to_string(l.stride) + " bytes; an element holds at most " +
                        std::to_string(pl::kStrideMax) + ".",
                    "fields");
    if (out) *out = l;
    return {};
}

Verdict check_piece_manifest(const spec::Piece& piece) { return ManifestCheck(piece).run(); }

Verdict check_web_limits(const spec::Piece& piece) {
    if (!piece.draw) return {};
    // WebGPU's defaults: 8 storage buffers and 16 sampled textures in one
    // stage, and 65,535 workgroups on each axis.
    std::size_t storage = 1;   // caliper_selection
    std::size_t textures = 1;  // caliper_colormaps
    for (const spec::Resource& r : piece.resources) {
        if (!r.publish) continue;
        if (r.kind == spec::ResourceKind::buffer) ++storage;
        else ++textures;
    }
    for (const spec::PieceInput& in : piece.inputs)
        if (in.kind == spec::InputKind::buffer) ++storage;
    storage += piece.draw->buffers.size();
    if (storage > 8)
        return fail(kProgramOverWebLimits,
                    "The draw half binds " + std::to_string(storage) +
                        " storage buffers in one stage: the published buffers, the buffer inputs, the draw buffers "
                        "and the selection. A browser allows 8.",
                    "draw");
    for (std::size_t i = 0; i < piece.draw->passes.size(); ++i) {
        const spec::DrawPass& pass = piece.draw->passes[i];
        if (textures + pass.reads.size() > 16)
            return fail(kProgramOverWebLimits, "The pass samples more than the 16 textures that a browser allows.",
                        at("draw.passes", i));
        for (std::uint32_t n : pass.workgroups)
            if (n > pl::kWorkgroupsMax)
                return fail(kProgramOverWebLimits, "A browser dispatches at most 65,535 workgroups on each axis.",
                            at("draw.passes", i) + ".workgroups");
    }
    return {};
}

Verdict check_tensor_binding(const spec::PieceInput& input, std::string_view dtype, std::uint32_t rank) {
    if (input.kind != spec::InputKind::tensor)
        return fail(kPieceBadBinding,
                    "The input " + quote(input.name) + " reads a published buffer, as program:<id>/<resource>; it "
                                                       "binds no tensor.",
                    input.name);
    const std::vector<std::string> types = spec::tensor_dtypes(input.dtype);
    if (std::find(types.begin(), types.end(), dtype) == types.end())
        return fail(kPieceBadBinding,
                    "The input " + quote(input.name) + " reads " + input.dtype + ", and the bound tensor is " +
                        std::string(dtype) + ".",
                    input.name);
    if (rank < input.rank_min || rank > input.rank_max)
        return fail(kPieceBadBinding,
                    "The input " + quote(input.name) + " takes " + std::to_string(input.rank_min) + " to " +
                        std::to_string(input.rank_max) + " axes, and the bound tensor has " + std::to_string(rank) + ".",
                    input.name);
    return {};
}

Verdict check_buffer_binding(const spec::PieceInput& input, const spec::Layout& published) {
    if (input.kind != spec::InputKind::buffer)
        return fail(kPieceBadBinding,
                    "The input " + quote(input.name) + " reads a tensor, as live:<tensor>; it binds no buffer.",
                    input.name);
    ElementLayout want, have;
    if (!element_layout(input.layout, &want).ok() || !element_layout(published, &have).ok())
        return fail(kPieceBadBinding, "The input " + quote(input.name) + " or the bound resource has a bad layout.",
                    input.name);
    bool same = want.stride == have.stride && want.per_word == have.per_word && want.fields.size() == have.fields.size();
    for (std::size_t i = 0; same && i < want.fields.size(); ++i)
        same = want.fields[i].name == have.fields[i].name && want.fields[i].type == have.fields[i].type &&
               want.fields[i].offset == have.fields[i].offset;
    if (!same)
        return fail(kPieceBadBinding,
                    "The input " + quote(input.name) +
                        " has another layout than the bound resource: the fields, their types and their order differ.",
                    input.name);
    return {};
}

}  // namespace caliper_builder::programs
