// wire_programs.cpp: protocol 3.2 GPU programs, piece instances, resources,
// and SplitResources reassembly (wire_internal.h).
#include "net/wire_internal.h"

#include <cstring>
#include <set>

namespace caliper_net {
namespace wire {

namespace {

using namespace detail;

constexpr const char* kOutputGeometry = "geometry";
constexpr const char* kOutputPixels = "pixels";
constexpr const char* kOutputTensor = "tensor";
constexpr size_t kParamsMax = 16;          // names in a params object
constexpr size_t kParamNameMax = 63;       // bytes of a params name
constexpr size_t kBuiltinMax = 32;         // bytes of a declaration's "builtin"
constexpr uint64_t kTensorElementsMax = 1000000;

bool lower(char c) { return c >= 'a' && c <= 'z'; }
bool digit(char c) { return c >= '0' && c <= '9'; }

// The operator form "operator.<spec id>.<n>": a spec id (a lower-case letter, then lower-case letters, digits, _ or -),
// a dot, and 1 to 5 decimal digits.
bool operator_id_ok(const std::string& s) {
    static const std::string kPrefix = "operator.";
    if (s.size() <= kPrefix.size() || s.compare(0, kPrefix.size(), kPrefix) != 0) return false;
    const size_t dot = s.rfind('.');
    if (dot <= kPrefix.size()) return false;   // no spec id
    const size_t n = s.size() - dot - 1;
    if (n < 1 || n > 5) return false;
    for (size_t i = dot + 1; i < s.size(); ++i)
        if (!digit(s[i])) return false;
    if (!lower(s[kPrefix.size()])) return false;
    for (size_t i = kPrefix.size(); i < dot; ++i)
        if (!(lower(s[i]) || digit(s[i]) || s[i] == '_' || s[i] == '-')) return false;
    return true;
}

// 1 to 16 names, each a valid name with a finite number.
bool params_ok(const Json& v) {
    if (!v.is_object() || v.size() < 1 || v.size() > kParamsMax) return false;
    for (const auto& kv : v.members())
        if (!param_name_ok(kv.first) || !is_finite(kv.second)) return false;
    return true;
}

bool tensor_shape_ok(const Json& v) {
    if (!v.is_array() || v.size() < 1 || v.size() > 4) return false;
    uint64_t prod = 1;
    for (const Json& e : v.items()) {
        if (!int_in(e, 1, kTensorElementsMax)) return false;
        prod *= e.as_double() > 0 ? (uint64_t)e.as_double() : 1;
        if (prod > kTensorElementsMax) return false;
    }
    return true;
}

bool sha256_hex_ok(const Json* v) {
    if (!v || !v->is_string() || v->as_string().size() != 64) return false;
    for (char c : v->as_string())
        if (!((c >= '0' && c <= '9') || (c >= 'a' && c <= 'f'))) return false;
    return true;
}

bool arrays_ok(const Json& v) {
    static const char* const kArrays[] = {"positions", "colors", "values", "radii"};
    if (!v.is_array() || v.size() < 1 || v.size() > 4) return false;
    std::set<std::string> seen;
    for (const Json& x : v.items()) {
        if (!x.is_string()) return false;
        bool known = false;
        for (const char* a : kArrays) known = known || x.as_string() == a;
        if (!known || !seen.insert(x.as_string()).second) return false;
    }
    return seen.count("positions") == 1;
}

bool input_entry(const Json& v) {
    return v.is_object() && name_ok(v.find("var")) && bounded_text(v.find("source"), CALIPER_NET_TENSOR_NAME_MAX);
}

// _check_declaration: output, then inputs, limits, workgroups and entry.
void check_declaration(const Json& d) {
    const Json& out = req(d, "output", is_obj, "an object");
    const std::string kind = req(out, "kind", [](const Json& v) {
        return one_of(v, {kOutputGeometry, kOutputPixels, kOutputTensor});
    }, "geometry, pixels or tensor").as_string();
    if (kind == kOutputTensor) {
        req(out, "shape", tensor_shape_ok, "1 to 4 integers of at least 1 with at most 1000000 elements");
        for (const auto& kv : out.members())
            if (kv.first != "kind" && kv.first != "shape") bad("a tensor output has only 'kind' and 'shape'");
    } else if (kind == kOutputGeometry) {
        req(out, "part", [](const Json& v) { return one_of(v, {CALIPER_NET_PART_POINTS, CALIPER_NET_PART_LINES}); },
            "points or lines");
        req(out, "max_count", [](const Json& v) { return int_in(v, 1, CALIPER_NET_PROGRAM_MAX_COUNT); },
            "an integer from 1 to 1000000");
        req(out, "arrays", arrays_ok,
            "unique names of positions, colors, values and radii, with positions");
        opt(out, "entity", [](const Json& v) { return v.is_string() && entity_path_ok(v.as_string()); },
            "an entity path");
    } else {
        auto side = [](const Json& v) { return int_in(v, 1, CALIPER_NET_PROGRAM_SIDE_MAX); };
        req(out, "width", side, "an integer from 1 to 2048");
        req(out, "height", side, "an integer from 1 to 2048");
        opt(out, "format", [](const Json& v) { return one_of(v, {"rgba8unorm"}); }, "rgba8unorm");
    }
    opt(d, "inputs", [](const Json& v) {
        if (!v.is_array() || v.size() > CALIPER_NET_PROGRAM_MAX_INPUTS) return false;
        for (const Json& e : v.items())
            if (!input_entry(e)) return false;
        return true;
    }, "at most 8 inputs {var, source}");
    if (const Json* lim = opt(d, "limits", is_obj, "an object")) {
        opt(*lim, "time_ms", [](const Json& v) {
            return is_finite(v) && v.as_double() > 0 && v.as_double() <= CALIPER_NET_PROGRAM_TIME_MS_MAX;
        }, "a number above 0 and at most 50");
        opt(*lim, "memory_mb", [](const Json& v) { return int_in(v, 1, CALIPER_NET_PROGRAM_MEMORY_MB_MAX); },
            "an integer from 1 to 512");
        opt(*lim, "every", [](const Json& v) { return int_in(v, 1, CALIPER_NET_PROGRAM_EVERY_MAX); },
            "an integer from 1 to 10000");
    }
    opt(d, "workgroups", [](const Json& v) {
        if (!v.is_array() || v.size() != 3) return false;
        for (const Json& e : v.items())
            if (!int_in(e, 1, CALIPER_NET_PROGRAM_WORKGROUPS_MAX)) return false;
        return true;
    }, "three integers from 1 to 65535");
    opt(d, "entry", is_name, "an entry point name");
    opt(d, "params", params_ok, "1 to 16 names of finite numbers");
    opt(d, "builtin", [](const Json& v) { return bounded_text(&v, kBuiltinMax); }, "a string of 1 to 32 bytes");
}

// One value of "bind": live:<a tensor name of 1 to 255 bytes>, or program:<program id>/<resource name>.
bool binding_ok(const Json& v) {
    if (!v.is_string()) return false;
    const std::string& s = v.as_string();
    if (s.compare(0, 5, "live:") == 0) return s.size() > 5 && s.size() - 5 <= CALIPER_NET_TENSOR_NAME_MAX;
    if (s.compare(0, 8, "program:") != 0) return false;
    const size_t slash = s.find('/', 8);
    if (slash == std::string::npos) return false;
    const Json id(s.substr(8, slash - 8));
    return program_id_ok(&id) && param_name_ok(s.substr(slash + 1));
}

bool bind_ok(const Json& v) {
    if (!v.is_object() || v.size() > CALIPER_NET_PROGRAM_MAX_INPUTS) return false;
    for (const auto& kv : v.members())
        if (!param_name_ok(kv.first) || !binding_ok(kv.second)) return false;
    return true;
}

bool segments_ok(const Json& v) {
    if (!v.is_array() || v.size() > CALIPER_NET_PIECE_SEGMENTS_MAX) return false;
    for (const Json& x : v.items()) {
        if (!x.is_object() || !name_ok(x.find("part"))) return false;
        const Json* first = x.find("first_line");
        const Json* lines = x.find("lines");
        if (!first || !int_in(*first, 1, 0xffffffffull) || !lines || !is_u32(*lines)) return false;
    }
    return true;
}

// An optional (B,) u8 section of 1 to `most` bytes of UTF-8.
std::optional<std::string> text_section(const Frame& f, const char* name, uint64_t most) {
    const Section* s = f.section(name);
    if (!s) return std::nullopt;
    if (!s->is(CALIPER_NET_DTYPE_U8, 1)) bad(std::string("'") + name + "' must be a (B,) u8 section");
    if (s->shape[0] < 1 || (uint64_t)s->shape[0] > most)
        bad(std::string("'") + name + "' holds 1 to " + std::to_string(most) + " bytes, got " +
            std::to_string(s->shape[0]));
    if (!utf8_ok(s->data, (size_t)s->shape[0])) bad(std::string("'") + name + "' is not UTF-8");
    return std::string(reinterpret_cast<const char*>(s->data), (size_t)s->shape[0]);
}

// The set of an instance of a piece, in _wire.py's order: no key or section of the one-pass form, piece,
// manifest, bind, params, panel, segments, then the sections common, derive and draw.
void program_instance(const Frame& f, Program* p) {
    const Json& m = f.meta;
    if (m.find("declaration") || m.find("scene") || m.find("sha256") || f.section(CALIPER_NET_SECTION_WGSL))
        bad("an instance of a piece carries no 'declaration', 'scene', 'sha256' or 'wgsl'");
    p->piece = req(m, "piece", piece_pin_ok, "a pin {name, version, hash}");
    p->manifest = req(m, "manifest", is_obj, "an object");
    if (const Json* v = opt(m, "bind", bind_ok, "at most 8 inputs bound to live:<tensor> or program:<id>/<resource>"))
        p->bind = *v;
    if (const Json* v = opt(m, "params", params_ok, "1 to 16 names (WGSL field names) of finite numbers"))
        p->params = *v;
    if (const Json* v = opt(m, "panel", is_name, "a panel id")) p->panel = v->as_string();
    if (const Json* v = opt(m, "segments", segments_ok, "at most 16 segments {part, first_line, lines}"))
        p->segments = *v;
    p->common = text_section(f, CALIPER_NET_SECTION_COMMON, CALIPER_NET_PIECE_COMMON_MAX);
    p->derive = text_section(f, CALIPER_NET_SECTION_DERIVE, CALIPER_NET_PROGRAM_SOURCE_MAX);
    p->draw = text_section(f, CALIPER_NET_SECTION_DRAW, CALIPER_NET_PROGRAM_SOURCE_MAX);
    if (!p->derive && !p->draw) bad("an instance of a piece carries the section 'derive' or 'draw'");
}

bool program_error_ok(const Json& v) {
    if (!v.is_object() || !name_ok(v.find("code"))) return false;
    const Json* msg = v.find("message");
    if (!(msg && is_text(*msg))) return false;
    return opt_ok(v, "line", is_u32) && opt_ok(v, "column", is_u32);
}

bool status_item(const Json& v) {
    if (!v.is_object() || !program_id_ok(v.find("id"))) return false;
    const Json* kind = v.find("kind");
    const Json* state = v.find("state");
    // A status item's kind: the output's kind, or "operator" for the program of a built-in operator.
    if (!(kind && one_of(*kind, {kOutputGeometry, kOutputPixels, kOutputTensor, "operator", CALIPER_NET_STATUS_PIECE})))
        return false;
    if (!(state && one_of(*state, {"waiting", "review", "compiled", "running", "refused", "stopped"}))) return false;
    auto every = [](const Json& x) { return int_in(x, 1, CALIPER_NET_PROGRAM_EVERY_MAX); };
    if (!opt_ok(v, "every", every) || !opt_ok(v, "every_min", every) || !opt_ok(v, "last_pass_ms", is_nonneg) ||
        !opt_ok(v, "last_cost_ms", is_nonneg) || !opt_ok(v, "step_ms", is_nonneg) ||
        !opt_ok(v, "overhead_share", is_nonneg) || !opt_ok(v, "passes_per_s", is_nonneg) ||
        !opt_ok(v, "last_step", is_safe_uint) || !opt_ok(v, "runs", is_safe_uint) ||
        !opt_ok(v, "memory_bytes", is_safe_uint) || !opt_ok(v, "memory_limit_bytes", is_safe_uint) ||
        !opt_ok(v, "error", program_error_ok) || !opt_ok(v, "warning", program_error_ok))
        return false;
    const Json* sha = v.find("sha256");
    return !present(sha) || (sha->is_string() && sha->as_string().empty()) || sha256_hex_ok(sha);
}

bool status_items(const Json& v) {
    if (!v.is_array() || v.size() > CALIPER_NET_MAX_PROGRAMS) return false;
    std::set<std::string> ids;
    for (const Json& x : v.items()) {
        if (!status_item(x)) return false;
        ids.insert(x.find("id")->as_string());
    }
    return ids.size() == v.size();
}

}  // namespace

namespace detail {

bool program_id_ok(const Json* v) {
    if (!v || !v->is_string()) return false;
    const std::string& s = v->as_string();
    if (s.empty() || s.size() > CALIPER_NET_PROGRAM_ID_MAX) return false;
    if (s.compare(0, 6, CALIPER_NET_SCENE_PROGRAM_PREFIX) == 0) return scene_program_id(s);
    if (s.compare(0, 9, "operator.") == 0) return operator_id_ok(s);
    if (!lower(s[0])) return false;
    for (char c : s)
        if (!(lower(c) || digit(c) || c == '_' || c == '-')) return false;
    return true;
}

// A params name: a WGSL field name of 1 to 63 bytes, [A-Za-z_][A-Za-z0-9_]*, not starting with "__".
bool param_name_ok(const std::string& s) {
    if (s.empty() || s.size() > kParamNameMax) return false;
    if (!(lower(s[0]) || (s[0] >= 'A' && s[0] <= 'Z') || s[0] == '_')) return false;
    if (s.size() >= 2 && s[0] == '_' && s[1] == '_') return false;
    for (char c : s)
        if (!(lower(c) || (c >= 'A' && c <= 'Z') || digit(c) || c == '_')) return false;
    return true;
}

// The pin of a piece: name [a-z][a-z0-9_]* of 1 to 32 bytes, version 1 to 1,000,000, hash "sha256:" with
// 64 lowercase hex digits, and optional from ("library", "workspace" or "inline").
bool piece_pin_ok(const Json& v) {
    if (!v.is_object()) return false;
    const Json* name = v.find("name");
    if (!name || !name->is_string()) return false;
    const std::string& n = name->as_string();
    if (n.empty() || n.size() > CALIPER_NET_PIECE_NAME_MAX || !lower(n[0])) return false;
    for (char c : n)
        if (!(lower(c) || digit(c) || c == '_')) return false;
    const Json* version = v.find("version");
    if (!version || !int_in(*version, 1, CALIPER_NET_PIECE_VERSION_MAX)) return false;
    const Json* hash = v.find("hash");
    if (!hash || !hash->is_string() || hash->as_string().compare(0, 7, "sha256:") != 0) return false;
    const Json hex(hash->as_string().substr(7));
    if (!sha256_hex_ok(&hex)) return false;
    const Json* from = v.find("from");
    return !present(from) || one_of(*from, {"library", "workspace", "inline"});
}

// RESOURCE, in _wire.py's order: program, resource, step, count, stride, width and height, format,
// element and elements, the keys of "split" (version, parts, part, first), the byte limit on count before any product (a version in parts
// holds at most 128 MiB for buffers or 256 MiB for textures), then the section. Products use bounded
// values, thus none wraps: width and height are at most 8192, count is at most the byte limit
// divided by stride, and a part's first element is below the count before the count less it is taken.
// A texture's and a narrow buffer's count is in 32-bit words, thus a part holds whole words.
Resource decode_resource(const Frame& f) {
    const Json& m = f.meta;
    Resource r;
    r.program = req(m, "program", [](const Json& v) { return program_id_ok(&v); },
                    "a program id of at most 55 bytes").as_string();
    r.resource = req(m, "resource", [](const Json& v) { return v.is_string() && param_name_ok(v.as_string()); },
                     "a WGSL name of 1 to 63 bytes").as_string();
    r.step = u64(req(m, "step", is_safe_uint, "an unsigned integer"));
    r.count = u32(req(m, "count", is_u32, "a uint32"));
    r.stride = u32(req(m, "stride", [](const Json& v) {
                           return int_in(v, 4, CALIPER_NET_RESOURCE_STRIDE_MAX) && u64(v) % 4 == 0;
                       }, "a multiple of 4 from 4 to 256"));
    auto side = [](const Json& v) { return int_in(v, 1, CALIPER_NET_RESOURCE_SIDE_MAX); };
    const Json* width = opt(m, "width", side, "an integer from 1 to 8192");
    const Json* height = opt(m, "height", side, "an integer from 1 to 8192");
    if (!width != !height) bad("a texture holds both 'width' and 'height'");
    const Json* format = opt(m, "format", [](const Json& v) {
        return one_of(v, {CALIPER_NET_TEXTURE_RGBA8, CALIPER_NET_TEXTURE_R8, CALIPER_NET_TEXTURE_R32UINT});
    }, "rgba8unorm, r8unorm or r32uint");
    if (format && !width) bad("'format' belongs to a texture, with 'width' and 'height'");
    if (width) {
        // At most 8192 * 8192 texels of 4 bytes, 2^28 bytes: the product cannot wrap.
        const uint64_t texels = u64(*width) * u64(*height);
        const bool r8 = format && format->as_string() == CALIPER_NET_TEXTURE_R8;
        const uint64_t words = r8 ? (texels + 3) / 4 : texels;
        if (r.stride != 4 || (uint64_t)r.count != words)
            bad("a texture has a stride of 4 and the words of its texels as its count");
        r.width = u32(*width);
        r.height = u32(*height);
        if (format) r.format = format->as_string();
    }
    const Json* element = opt(m, "element", [](const Json& v) {
        return one_of(v, {CALIPER_NET_ELEMENT_U8, CALIPER_NET_ELEMENT_U16});
    }, "u8 or u16");
    const Json* elements = opt(m, "elements", is_u32, "a uint32");
    if (!element != !elements) bad("a buffer of narrow elements holds both 'element' and 'elements'");
    if (element) {
        if (width) bad("a texture has no 'element'");
        // At most 2^32 - 1 elements of 2 bytes in 64 bits: the product cannot wrap.
        const uint64_t bytes = element->as_string() == CALIPER_NET_ELEMENT_U8 ? 1 : 2;
        if (r.stride != 4 || (uint64_t)r.count != (u64(*elements) * bytes + 3) / 4)
            bad("a buffer of narrow elements has a stride of 4 and the words of its elements as its count");
        r.element = element->as_string();
        r.elements = u32(*elements);
    }
    if (const Json* sp = opt(m, "split", is_obj, "an object")) {
        auto need = [&](const char* key, auto ok, const char* what) -> const Json& {
            const Json* v = sp->find(key);
            if (!v || v->is_null() || !ok(*v)) bad(std::string("'split.") + key + "' must be " + what);
            return *v;
        };
        Resource::Split sl;
        sl.version = u64(need("version", is_safe_uint, "an unsigned integer"));
        sl.parts = u32(need("parts", [](const Json& v) { return int_in(v, 2, CALIPER_NET_RESOURCE_MAX_PARTS); },
                            "an integer from 2 to 32"));
        sl.part = u32(need("part", [&](const Json& v) { return int_in(v, 0, sl.parts - 1); },
                           "a part number below 'parts'"));
        sl.first = u32(need("first", is_u32, "a uint32"));
        r.split = sl;
    }
    const uint32_t version_max = r.width ? CALIPER_NET_RESOURCE_TEXTURE_VERSION_MAX_BYTES
                                         : CALIPER_NET_RESOURCE_VERSION_MAX_BYTES;
    const uint32_t max_bytes = r.split ? version_max : CALIPER_NET_RESOURCE_MAX_BYTES;
    if (r.count > max_bytes / r.stride)
        bad(std::string(r.split ? "a resource in parts" : "a resource") + " holds at most " + std::to_string(max_bytes) +
            " bytes");
    const Section* s = f.section(CALIPER_NET_SECTION_RESOURCE);
    if (!s || !s->is(CALIPER_NET_DTYPE_U8, 1)) bad("a resource needs a (B,) u8 'resource' section");
    if (r.split) {
        const Resource::Split& sl = *r.split;
        if (sl.part == 0 && sl.first != 0) bad("part 0 of a resource starts at element 0");
        if (sl.first >= r.count) bad("'split.first' must lie below the count");
        const int64_t b = s->shape[0];
        if (b <= 0 || (uint64_t)b % r.stride != 0 || (uint64_t)b / r.stride > (uint64_t)r.count - sl.first)
            bad("a part of a resource holds whole elements from 'split.first' up to the count");
    } else {
        const uint64_t want = (uint64_t)r.count * r.stride;
        if (s->shape[0] < 0 || (uint64_t)s->shape[0] != want)
            bad("'resource' holds count * stride = " + std::to_string(want) + " bytes, got " +
                std::to_string(s->shape[0]));
    }
    r.data = *s;
    return r;
}

Program decode_program(const Frame& f) {
    const Json& m = f.meta;
    Program p;
    p.request_id = u32(req(m, "request_id", is_u32, "a uint32"));
    p.op = req(m, "op", [](const Json& v) {
                   return one_of(v, {CALIPER_NET_PROGRAM_SET, CALIPER_NET_PROGRAM_REMOVE, CALIPER_NET_PROGRAM_PARAMS});
               }, "set, remove or params").as_string();
    p.id = req(m, "id", [](const Json& v) { return program_id_ok(&v); },
               "a program id of at most 55 bytes").as_string();
    if (p.op == CALIPER_NET_PROGRAM_REMOVE) return p;
    if (p.op == CALIPER_NET_PROGRAM_PARAMS) {
        if (m.find("declaration")) bad("a params op carries no 'declaration'");
        if (!f.sections.empty()) bad("a params op carries no section");
        p.params = req(m, "params", params_ok, "1 to 16 names (WGSL field names) of finite numbers");
        return p;
    }
    if (m.find("piece")) {
        program_instance(f, &p);
        return p;
    }
    const Json& d = req(m, "declaration", is_obj, "an object");
    check_declaration(d);
    const bool geometry = d.find("output")->find("kind")->as_string() == kOutputGeometry;
    const Json* scene = geometry ? &req(m, "scene", is_name, "a scene name") : opt(m, "scene", is_name, "a scene name");
    const Json& sha = req(m, "sha256", [](const Json& v) { return sha256_hex_ok(&v); }, "64 lowercase hex digits");
    const Section* w = f.section(CALIPER_NET_SECTION_WGSL);
    if (!w || !w->is(CALIPER_NET_DTYPE_U8, 1)) bad("a program set needs a (B,) u8 'wgsl' section");
    if (w->shape[0] < 1 || w->shape[0] > (int64_t)CALIPER_NET_PROGRAM_SOURCE_MAX)
        bad("'wgsl' holds 1 to " + std::to_string(CALIPER_NET_PROGRAM_SOURCE_MAX) + " bytes, got " +
            std::to_string(w->shape[0]));
    if (!utf8_ok(w->data, (size_t)w->shape[0])) bad("'wgsl' is not UTF-8");
    p.declaration = d;
    p.wgsl = std::string(reinterpret_cast<const char*>(w->data), (size_t)w->shape[0]);
    p.sha256 = sha.as_string();
    if (scene) p.scene = scene->as_string();
    return p;
}

ProgramState decode_program_state(const Frame& f) {
    const Json& m = f.meta;
    ProgramState s;
    if (const Json* v = opt(m, "request_id", is_u32, "a uint32")) s.request_id = u32(*v);
    if (const Json* v = opt(m, "step", is_safe_uint, "an unsigned integer")) s.step = u64(*v);
    if (const Json* v = opt(m, "step_ms", is_nonneg, "a number of at least 0")) s.step_ms = v->as_double();
    s.programs = req(m, "programs", status_items, "at most 16 status items with unique program ids");
    return s;
}

void resource_to_meta(const Resource& r, Json& m, std::vector<Section>& sections) {
    m.set("program", r.program);
    m.set("resource", r.resource);
    m.set("step", r.step);
    m.set("count", r.count);
    m.set("stride", r.stride);
    if (r.width) m.set("width", *r.width);
    if (r.height) m.set("height", *r.height);
    if (r.format) m.set("format", *r.format);
    if (r.element) m.set("element", *r.element);
    if (r.elements) m.set("elements", *r.elements);
    if (r.split) {
        Json sp = Json::object();
        sp.set("version", r.split->version);
        sp.set("part", r.split->part);
        sp.set("parts", r.split->parts);
        sp.set("first", r.split->first);
        m.set("split", sp);
    }
    sections.push_back(Section::make(CALIPER_NET_SECTION_RESOURCE, CALIPER_NET_DTYPE_U8, {(int64_t)r.data.size},
                                     r.data.data, r.data.size));
}

void program_to_meta(const Program& p, Json& m, std::vector<Section>& sections) {
    m.set("request_id", p.request_id);
    m.set("op", p.op);
    m.set("id", p.id);
    if (!p.declaration.is_null()) m.set("declaration", p.declaration);
    if (!p.params.is_null()) m.set("params", p.params);
    if (p.scene) m.set("scene", *p.scene);
    if (p.sha256) m.set("sha256", *p.sha256);
    if (!p.piece.is_null()) m.set("piece", p.piece);
    if (!p.manifest.is_null()) m.set("manifest", p.manifest);
    if (!p.bind.is_null()) m.set("bind", p.bind);
    if (p.panel) m.set("panel", *p.panel);
    if (!p.segments.is_null()) m.set("segments", p.segments);
    // The sections in _wire.py's order: wgsl, common, derive, draw.
    const std::pair<const char*, const std::optional<std::string>*> texts[] = {
        {CALIPER_NET_SECTION_WGSL, &p.wgsl},
        {CALIPER_NET_SECTION_COMMON, &p.common},
        {CALIPER_NET_SECTION_DERIVE, &p.derive},
        {CALIPER_NET_SECTION_DRAW, &p.draw}};
    for (const auto& [name, text] : texts)
        if (*text)
            sections.push_back(Section::make(name, CALIPER_NET_DTYPE_U8, {(int64_t)(*text)->size()}, (*text)->data(),
                                             (*text)->size()));
}

void program_state_to_meta(const ProgramState& s, Json& m) {
    if (s.request_id) m.set("request_id", *s.request_id);
    if (s.step) m.set("step", *s.step);
    if (s.step_ms) m.set("step_ms", *s.step_ms);
    m.set("programs", s.programs);
}

}  // namespace detail

bool SplitResources::holds(const std::string& program, const std::string& resource) const {
    for (const Assembly& a : open_)
        if (a.head.program == program && a.head.resource == resource) return true;
    return false;
}

uint64_t SplitResources::held_bytes() const {
    uint64_t n = 0;
    for (const Assembly& a : open_) n += a.bytes ? a.bytes->size() : 0;
    return n;
}

SplitResources::Result SplitResources::add(const Resource& r, WholeResource* out, std::string* err) {
    ++frames_;
    for (auto it = open_.begin(); it != open_.end();) {   // the next part did not come in time
        if (frames_ - it->last_frame > CALIPER_NET_RESOURCE_PART_WAIT) {
            ++expired_;
            it = open_.erase(it);
        } else {
            ++it;
        }
    }
    auto held = std::find_if(open_.begin(), open_.end(), [&](const Assembly& a) {
        return a.head.program == r.program && a.head.resource == r.resource;
    });
    if (!r.split || r.split->part == 0) {   // a newer version of the resource
        if (held != open_.end()) {
            ++discarded_;
            open_.erase(held);
        }
    }
    if (!r.split) {
        if (out) {
            out->resource = r;
            out->keep.reset();
        }
        return Result::Whole;
    }
    const Resource::Split& sl = *r.split;
    const std::string name = "'" + r.program + "/" + r.resource + "'";
    // The decoder checked that the part holds whole elements from sl.first up to the count, and that the
    // count times the stride is at most 256 MiB for textures or 128 MiB for buffers: no product or sum wraps.
    const uint64_t n = r.data.size / r.stride;
    if (sl.part == 0) {
        if (open_.size() >= CALIPER_NET_MAX_INCOMPLETE_UPDATES) {
            ++discarded_;
            open_.erase(open_.begin());
        }
        Assembly a;
        a.head = r;
        a.head.data = Section();
        a.next_first = n;
        a.last_frame = frames_;
        if (copy_) {
            a.bytes = std::make_shared<std::vector<uint8_t>>((size_t)r.count * r.stride);
            std::memcpy(a.bytes->data(), r.data.data, r.data.size);
        }
        open_.push_back(std::move(a));
        return Result::Pending;
    }
    if (held == open_.end() || held->head.split->version != sl.version) {
        ++dropped_;   // a part of a version whose part 0 is not held
        return Result::Dropped;
    }
    auto refuse = [&](const std::string& why) {
        ++discarded_;
        open_.erase(held);
        if (err) *err = why;
        return Result::Refused;
    };
    const Resource& h = held->head;
    if (r.step != h.step || r.count != h.count || r.stride != h.stride || r.width != h.width ||
        r.height != h.height || r.format != h.format || r.element != h.element || r.elements != h.elements ||
        sl.parts != h.split->parts)
        return refuse("a part of the resource " + name + " differs from its part 0");
    if (sl.part != held->next_part || sl.first != held->next_first)
        return refuse("a part of the resource " + name + " comes out of order");
    const uint64_t end = held->next_first + n;
    const bool last = sl.part + 1 == sl.parts;
    if (last != (end == h.count))
        return refuse(last ? "the parts of the resource " + name + " end before its count"
                           : "a part of the resource " + name + " before the last reaches its count");
    if (copy_) std::memcpy(held->bytes->data() + held->next_first * h.stride, r.data.data, r.data.size);
    held->next_part += 1;
    held->next_first = end;
    held->last_frame = frames_;
    if (!last) return Result::Pending;
    if (out) {
        out->resource = h;
        out->resource.split.reset();
        out->keep = held->bytes;
        if (held->bytes)
            out->resource.data = Section::make(CALIPER_NET_SECTION_RESOURCE, CALIPER_NET_DTYPE_U8,
                                               {(int64_t)held->bytes->size()}, held->bytes->data(),
                                               held->bytes->size());
    }
    open_.erase(held);
    return Result::Whole;
}

}  // namespace wire
}  // namespace caliper_net
