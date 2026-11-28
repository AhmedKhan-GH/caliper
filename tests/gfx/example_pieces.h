// Two worked piece examples as test fixtures: attention_field (a fragment-only field)
// and ring_arcs (arcs drawn instanced with additive blending, again into a half-size
// target, and a bloom of that target), their files, and one recorded version of each
// published resource, computed on the CPU from a fixed pattern, thus no training runs.
//
// The draw tests reach the render graph through these pieces. Where the checks of pieces
// (programs/pieces.h) answer program.not_built in the tree, the stand-ins below take their
// place: the resolution of imports, the storage layout of WGSL, and a hash of the files.
#pragma once

#include <caliper_builder/programs/pieces.h>
#include <caliper_builder/spec/piece.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

namespace caliper_test_pieces {

namespace cs = caliper_builder::spec;
namespace cp = caliper_builder::programs;

inline constexpr std::uint32_t kLayers = 2, kHeads = 4, kTokens = 32, kRepeat = 11;

inline cs::Layout element(const char* type) {
    cs::Layout l;
    l.element = type;
    return l;
}

inline cs::Layout fields(const char* name, std::vector<cs::Field> f) {
    cs::Layout l;
    l.type_name = name;
    l.fields = std::move(f);
    return l;
}

inline cs::PieceInput tensor_input(const char* name) {
    cs::PieceInput in;
    in.name = name;
    in.kind = cs::InputKind::tensor;
    in.rank_min = 4;
    in.rank_max = 4;
    return in;
}

inline cs::PieceParam param(const char* name, cs::ParamType type, double def, double min, double max) {
    cs::PieceParam p;
    p.name = name;
    p.type = type;
    p.default_value = def;
    p.min = min;
    p.max = max;
    return p;
}

// ---- attention_field -------------------------------------------------------------------------

inline const char* kFieldCommon = R"(// common.wgsl
const H: u32 = 4u;    // heads
const T: u32 = 32u;   // tokens of the fixed sentence
)";

inline const char* kFieldDerive = R"(// derive.wgsl: batch row 0 of each (B, H, T, T) tensor is its first H * T * T values.
@compute @workgroup_size(256)
fn copy_rows(@builtin(global_invocation_id) id: vec3<u32>) {
    let n = H * T * T;
    if (id.x >= n) {
        return;
    }
    pattern[id.x] = attn0[id.x];
    pattern[n + id.x] = attn1[id.x];
}
)";

inline const char* kFieldDraw = R"(// draw.wgsl: rows are the query token, columns the key token.
fn value(layer: u32, head: u32, i: u32, j: u32) -> f32 {
    return pattern[((layer * H + head) * T + i) * T + j];
}

@fragment
fn field(in: CaliperFullscreen) -> @location(0) vec4<f32> {
    let layer = u32(clamp(params.layer, 0, 1));
    let head = u32(clamp(params.head, 0, i32(H) - 1));
    let p = in.uv * f32(T) - 0.5;
    let a = vec2<u32>(clamp(floor(p), vec2<f32>(0.0), vec2<f32>(f32(T - 1u))));
    let b = min(a + 1u, vec2<u32>(T - 1u));
    let f = clamp(p - vec2<f32>(a), vec2<f32>(0.0), vec2<f32>(1.0));
    let v = mix(mix(value(layer, head, a.y, a.x), value(layer, head, a.y, b.x), f.x),
                mix(value(layer, head, b.y, a.x), value(layer, head, b.y, b.x), f.x), f.y);
    let t = clamp(v * params.gain, 0.0, 1.0);
    var color = colormap::magma(t);
    if (abs(fract(t * 8.0) - 0.5) < 0.04) {
        color = mix(color, vec4<f32>(1.0), 0.35);
    }
    return color;
}
)";

inline cs::Piece attention_field() {
    cs::Piece p;
    p.name = "attention_field";
    p.version = 1;
    p.description = "The attention pattern of one head of a two-layer model, as a field with contour lines";
    p.imports = {"colormap@1"};
    p.inputs = {tensor_input("attn0"), tensor_input("attn1")};
    p.params = {param("layer", cs::ParamType::i32, 0, 0, 1), param("head", cs::ParamType::i32, 0, 0, 3),
                param("gain", cs::ParamType::f32, 1.0, 0.1, 10.0)};
    cs::Resource pattern;
    pattern.id = "pattern";
    pattern.layout = element("f32");
    pattern.count = 8192;
    p.resources = {pattern};
    cs::Derive derive;
    cs::DerivePass copy;
    copy.entry = "copy_rows";
    copy.workgroups = {16, 1, 1};
    copy.writes = {"pattern"};
    derive.passes = {copy};
    p.derive = derive;
    cs::Draw draw;
    draw.camera = cs::CameraMode::none;
    cs::DrawPass field;
    field.kind = cs::PassKind::render;
    field.fragment = "field";
    draw.passes = {field};
    p.draw = draw;
    p.files = {{"common.wgsl", kFieldCommon}, {"derive.wgsl", kFieldDerive}, {"draw.wgsl", kFieldDraw}};
    return p;
}

// ---- ring_arcs --------------------------------------------------------------------------

inline const char* kArcsCommon = R"(// common.wgsl
const L: u32 = 2u;     // layers
const H: u32 = 4u;     // heads
const T: u32 = 32u;    // tokens of the fixed sentence
const P: u32 = 12u;    // length of the repeated segment
const S: u32 = 4u;     // first token of its first copy

fn ring_point(layer: u32, token: u32) -> vec3<f32> {
    let a = 6.2831853 * f32(token) / f32(T);
    return vec3<f32>(cos(a), f32(layer) * 0.6, sin(a));
}
)";

inline const char* kArcsDerive = R"(// derive.wgsl
fn attn(layer: u32, head: u32, i: u32, j: u32) -> f32 {
    let k = (head * T + i) * T + j;     // batch row 0
    if (layer == 0u) {
        return attn0[k];
    }
    return attn1[k];
}

@compute @workgroup_size(8)
fn scores(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x >= L * H) {
        return;
    }
    var sum = 0.0;
    for (var t = S + P; t < S + 2u * P; t = t + 1u) {
        sum = sum + attn(id.x / H, id.x % H, t, t - P + 1u);
    }
    heads[id.x] = sum / f32(P);
}

@compute @workgroup_size(64)
fn build(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x >= L * H * T * T) {
        return;
    }
    let j = id.x % T;
    let i = (id.x / T) % T;
    let head = (id.x / (T * T)) % H;
    let layer = id.x / (T * T * H);
    if (j >= i) {
        return;
    }
    let w = attn(layer, head, i, j);
    if (w < params.threshold) {
        return;
    }
    let slot = atomicAdd(&arcs_count, 1u);
    if (slot >= arrayLength(&arcs)) {
        return;
    }
    arcs[slot] = Arc(j, i, layer, head, w, heads[layer * H + head]);
}

@compute @workgroup_size(32)
fn place_labels(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x >= T) {
        return;
    }
    labels[id.x] = Label(vec4<f32>(ring_point(0u, id.x) * 1.15, 1.0), id.x);
}
)";

inline const char* kArcsDraw = R"(// draw.wgsl
const SEG: u32 = 32u;    // segments of one arc: 64 vertices of a strip

struct ArcOut {
    @builtin(position) position: vec4<f32>,
    @location(0) color: vec4<f32>,
    @location(1) @interpolate(flat) item: u32,
}

fn arc_point(arc: Arc, t: f32) -> vec3<f32> {
    let p0 = ring_point(arc.layer, arc.from_token);
    let p2 = ring_point(arc.layer, arc.to_token);
    let p1 = 0.5 * (p0 + p2) + vec3<f32>(0.0, 0.1 + 0.5 * arc.weight, 0.0);
    let u = 1.0 - t;
    return u * u * p0 + 2.0 * u * t * p1 + t * t * p2;
}

@vertex
fn arc_vs(@builtin(instance_index) i: u32, @builtin(vertex_index) v: u32) -> ArcOut {
    let arc = arcs[i];
    let t = f32(v / 2u) / f32(SEG - 1u);
    let e = 0.5 / f32(SEG - 1u);
    let c = camera::clip(arc_point(arc, t));
    let a = camera::clip(arc_point(arc, max(t - e, 0.0)));
    let b = camera::clip(arc_point(arc, min(t + e, 1.0)));
    let d = camera::pixel_direction(a, b);
    let side = f32(v % 2u) - 0.5;
    let width = (1.0 + 5.0 * arc.weight) * caliper_view.time.z;
    var out: ArcOut;
    out.position = camera::offset_pixels(c, vec2<f32>(-d.y, d.x) * side * width);
    let rgb = colormap::viridis(f32(arc.head) / f32(H - 1u)).rgb * (0.3 + 0.7 * arc.weight);
    out.color = vec4<f32>(rgb * arc.weight, arc.weight);
    out.item = i;
    if (pick::selected(i)) {
        out.color = vec4<f32>(1.0, 0.9, 0.6, 1.0);
    }
    return out;
}

@fragment
fn arc_fs(in: ArcOut) -> CaliperPicked {
    var out: CaliperPicked;
    out.color = in.color;
    out.id = pick::id(in.item);
    return out;
}

@fragment
fn arc_glow(in: ArcOut) -> @location(0) vec4<f32> {
    return in.color;
}

@fragment
fn bloom(in: CaliperFullscreen) -> @location(0) vec4<f32> {
    let texel = 1.0 / vec2<f32>(textureDimensions(glow));
    var sum = vec4<f32>(0.0);
    for (var y = -2; y <= 2; y = y + 1) {
        for (var x = -2; x <= 2; x = x + 1) {
            let w = exp(-f32(x * x + y * y) / 4.0);
            let uv = in.uv + vec2<f32>(f32(x), f32(y)) * texel * 2.0;
            sum = sum + w * textureSampleLevel(glow, caliper_linear, uv, 0.0);
        }
    }
    return sum * 0.12 * params.glow;
}
)";

inline cs::Piece ring_arcs() {
    cs::Piece p;
    p.name = "ring_arcs";
    p.version = 1;
    p.description = "Arcs between the tokens of one sentence, one ring for each layer, from the attention weights";
    p.imports = {"colormap@1", "pick@1", "camera@1"};
    p.inputs = {tensor_input("attn0"), tensor_input("attn1")};
    cs::PieceParam threshold = param("threshold", cs::ParamType::f32, 0.05, 0.0, 0.5);
    threshold.title = "Weight Threshold";
    p.params = {threshold, param("glow", cs::ParamType::f32, 1.0, 0.0, 4.0)};
    cs::Resource heads, arcs, labels;
    heads.id = "heads";
    heads.layout = element("f32");
    heads.count = 8;
    arcs.id = "arcs";
    arcs.layout = fields("Arc", {{"from_token", "u32"}, {"to_token", "u32"}, {"layer", "u32"}, {"head", "u32"},
                                 {"weight", "f32"}, {"score", "f32"}});
    arcs.count = 4096;
    arcs.counted = true;
    labels.id = "labels";
    labels.layout = fields("Label", {{"position", "vec4<f32>"}, {"string", "u32"}});
    labels.count = 32;
    p.resources = {heads, arcs, labels};
    cs::Derive derive;
    cs::DerivePass scores, build, place;
    scores.entry = "scores";
    scores.writes = {"heads"};
    build.entry = "build";
    build.workgroups = {128, 1, 1};
    build.writes = {"arcs"};
    place.entry = "place_labels";
    derive.passes = {scores, build, place};
    p.derive = derive;
    cs::Draw draw;
    draw.camera = cs::CameraMode::orbit;
    cs::DrawTarget glow;
    glow.id = "glow";
    glow.scale = 0.5f;
    draw.targets = {glow};
    cs::DrawPass arc, arc_glow, bloom;
    arc.kind = cs::PassKind::render;
    arc.vertex = "arc_vs";
    arc.fragment = "arc_fs";
    arc.instances_of = "arcs";
    arc.vertices = 64;
    arc.topology = cs::DrawTopology::triangle_strip;
    arc.blend = cs::DrawBlend::additive;
    arc.depth = cs::DrawDepth::none;
    arc.pick = true;
    arc_glow = arc;
    arc_glow.fragment = "arc_glow";
    arc_glow.target = "glow";
    arc_glow.blend.reset();
    arc_glow.depth.reset();
    arc_glow.pick = false;
    bloom.kind = cs::PassKind::render;
    bloom.fragment = "bloom";
    bloom.reads = {"glow"};
    bloom.blend = cs::DrawBlend::additive;
    draw.passes = {arc, arc_glow, bloom};
    cs::PickMap pick;
    pick.resource = "arcs";
    pick.fields = {"layer", "head", "from_token", "to_token", "weight", "score"};
    draw.pick = pick;
    cs::TextTable text;
    text.anchors = "labels";
    text.strings = {"the",  "old",  "map",   "shows", "red",  "fox",   "jumps", "over",  "lazy", "dogs", "near",
                    "green", "hills", "under", "blue", "skies", "red", "fox",   "jumps", "over",  "lazy", "dogs",
                    "near", "green", "hills", "under", "blue", "skies", "and",  "then",  "it",    "ends"};
    draw.text = text;
    p.draw = draw;
    p.files = {{"common.wgsl", kArcsCommon}, {"derive.wgsl", kArcsDerive}, {"draw.wgsl", kArcsDraw}};
    return p;
}

// ---- stand-ins for the checks of pieces ---------------------------------------------------------

// The resolution of imports as Imports of the plan states it: the modules' text in the order of
// `imports`, and the author's files with "::" rewritten to "__".
inline cp::Resolved resolve(const cs::Piece& piece) {
    cp::Resolved r;
    std::uint32_t line = 1;
    for (const std::string& m : piece.imports) {
        std::string text(cp::lib_module(m));
        if (text.empty()) {
            r.verdict.code = cp::kPieceUnknownImport;
            r.verdict.message = "no lib module " + m;
            return r;
        }
        if (text.back() != '\n') text += '\n';
        cp::Segment s;
        s.part = "lib/" + m;
        s.first_line = line;
        s.lines = std::uint32_t(std::count(text.begin(), text.end(), '\n'));
        line += s.lines;
        r.lib_segments.push_back(s);
        r.lib_text += text;
        r.modules.push_back(m);
    }
    for (const auto& [name, text] : piece.files) {
        std::string t = text;
        for (std::size_t at = t.find("::"); at != std::string::npos; at = t.find("::", at + 2)) t.replace(at, 2, "__");
        r.files[name] = t;
    }
    return r;
}

// The storage layout of WGSL for the field types of a piece.
inline cp::Verdict layout(const cs::Layout& l, cp::ElementLayout* out) {
    auto size_align = [](const std::string& t, std::uint32_t* size, std::uint32_t* align) {
        if (t == "f32" || t == "i32" || t == "u32") *size = 4, *align = 4;
        else if (t == "vec2<f32>" || t == "vec2<u32>") *size = 8, *align = 8;
        else if (t == "vec4<f32>" || t == "vec4<u32>") *size = 16, *align = 16;
        else return false;
        return true;
    };
    cp::ElementLayout e;
    std::vector<cs::Field> f = l.fields;
    if (l.element) f = {{"value", *l.element}};
    std::uint32_t offset = 0;
    e.align = 4;
    for (const cs::Field& field : f) {
        std::uint32_t size = 0, align = 0;
        if (!size_align(field.type, &size, &align)) {
            cp::Verdict v;
            v.code = cp::kProgramBadLayout;
            v.message = "bad type " + field.type;
            return v;
        }
        offset = (offset + align - 1) / align * align;
        e.fields.push_back({field.name, field.type, offset, size});
        offset += size;
        e.align = std::max(e.align, align);
    }
    e.stride = (offset + e.align - 1) / e.align * e.align;
    *out = e;
    return {};
}

// A hash of the manifest's name and version and of the files, in place of piece_hash().
inline std::string hash(const cs::Piece& piece) {
    std::uint64_t h = 1469598103934665603ull;
    auto mix = [&](const std::string& s) {
        for (unsigned char c : s) h = (h ^ c) * 1099511628211ull;
    };
    mix(piece.name + "@" + std::to_string(piece.version));
    for (const auto& [name, text] : piece.files) mix("--- " + name + "\n" + text);
    char hex[17];
    std::snprintf(hex, sizeof hex, "%016llx", static_cast<unsigned long long>(h));
    std::string out = "sha256:";
    for (int i = 0; i < 4; ++i) out += hex;
    return out;
}

// ---- the recorded resources ---------------------------------------------------------------------

// A fixed attention pattern (layer, head, query i, key j), causal and normalized over j: each
// head attends to recent tokens with its own decay, and head 2 of layer 1 is an induction head
// that attends from token t of the second copy to token t - 11.
inline float attention(std::uint32_t layer, std::uint32_t head, std::uint32_t i, std::uint32_t j) {
    if (j > i) return 0.0f;
    const bool induction = layer == 1 && head == 2 && i >= 16 && i < 28;
    double sum = 0, w = 0;
    for (std::uint32_t k = 0; k <= i; ++k) {
        double v = std::exp(-double(i - k) / (1.0 + head + 2.0 * layer));
        if (induction) v = (k + kRepeat == i) ? 20.0 : 0.05;
        sum += v;
        if (k == j) w = v;
    }
    return float(w / sum);
}

// attention_field's `pattern`: 2 layers, 4 heads, 32 by 32 tokens.
inline std::vector<float> field_pattern() {
    std::vector<float> p(kLayers * kHeads * kTokens * kTokens);
    for (std::uint32_t l = 0; l < kLayers; ++l)
        for (std::uint32_t h = 0; h < kHeads; ++h)
            for (std::uint32_t i = 0; i < kTokens; ++i)
                for (std::uint32_t j = 0; j < kTokens; ++j)
                    p[((l * kHeads + h) * kTokens + i) * kTokens + j] = attention(l, h, i, j);
    return p;
}

struct Arc {
    std::uint32_t from_token, to_token, layer, head;
    float weight, score;
};
static_assert(sizeof(Arc) == 24, "Arc is 24 bytes, as the composer's layout");

struct Label {
    float position[4];
    std::uint32_t string;
    std::uint32_t pad[3];
};
static_assert(sizeof(Label) == 32, "Label is 32 bytes, as the composer's layout");

// ring_arcs' resources as its derive passes compute them, in the order of `build`.
struct ArcResources {
    std::vector<float> heads;
    std::vector<Arc> arcs;
    std::vector<Label> labels;
};

inline ArcResources arc_resources(float threshold = 0.05f) {
    ArcResources r;
    r.heads.resize(kLayers * kHeads);
    for (std::uint32_t x = 0; x < kLayers * kHeads; ++x) {
        float sum = 0;
        for (std::uint32_t t = 16; t < 28; ++t) sum += attention(x / kHeads, x % kHeads, t, t - 12 + 1);
        r.heads[x] = sum / 12.0f;
    }
    for (std::uint32_t l = 0; l < kLayers; ++l)
        for (std::uint32_t h = 0; h < kHeads; ++h)
            for (std::uint32_t i = 0; i < kTokens; ++i)
                for (std::uint32_t j = 0; j < i; ++j) {
                    const float w = attention(l, h, i, j);
                    if (w >= threshold && r.arcs.size() < 4096) r.arcs.push_back({j, i, l, h, w, r.heads[l * kHeads + h]});
                }
    for (std::uint32_t t = 0; t < kTokens; ++t) {
        const float a = 6.2831853f * float(t) / float(kTokens);
        Label label{};
        label.position[0] = std::cos(a) * 1.15f;
        label.position[2] = std::sin(a) * 1.15f;
        label.position[3] = 1.0f;
        label.string = t;
        r.labels.push_back(label);
    }
    return r;
}

// ---- a camera for the arcs ------------------------------------------------------------------------

// Column-major 4 x 4 matrices, as WGSL's mat4x4 reads them.
using Mat4 = std::array<float, 16>;

inline Mat4 multiply(const Mat4& a, const Mat4& b) {
    Mat4 r{};
    for (int c = 0; c < 4; ++c)
        for (int row = 0; row < 4; ++row) {
            float s = 0;
            for (int k = 0; k < 4; ++k) s += a[std::size_t(k * 4 + row)] * b[std::size_t(c * 4 + k)];
            r[std::size_t(c * 4 + row)] = s;
        }
    return r;
}

// A perspective camera at `eye` that looks at `at` with Y up, depth 0 to 1 as in WebGPU.
inline Mat4 view_matrix(const std::array<float, 3>& eye, const std::array<float, 3>& at) {
    auto sub = [](std::array<float, 3> a, std::array<float, 3> b) {
        return std::array<float, 3>{a[0] - b[0], a[1] - b[1], a[2] - b[2]};
    };
    auto norm = [](std::array<float, 3> a) {
        const float n = std::sqrt(a[0] * a[0] + a[1] * a[1] + a[2] * a[2]);
        return std::array<float, 3>{a[0] / n, a[1] / n, a[2] / n};
    };
    auto cross = [](std::array<float, 3> a, std::array<float, 3> b) {
        return std::array<float, 3>{a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]};
    };
    auto dot = [](std::array<float, 3> a, std::array<float, 3> b) { return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]; };
    const auto f = norm(sub(at, eye));
    const auto s = norm(cross(f, {0, 1, 0}));
    const auto u = cross(s, f);
    return {s[0], u[0], -f[0], 0, s[1], u[1], -f[1], 0, s[2], u[2], -f[2], 0, -dot(s, eye), -dot(u, eye), dot(f, eye), 1};
}

inline Mat4 perspective(float fov_y, float aspect, float near_z, float far_z) {
    const float t = 1.0f / std::tan(fov_y / 2);
    Mat4 m{};
    m[0] = t / aspect;
    m[5] = t;
    m[10] = far_z / (near_z - far_z);
    m[11] = -1;
    m[14] = near_z * far_z / (near_z - far_z);
    return m;
}

}  // namespace caliper_test_pieces
