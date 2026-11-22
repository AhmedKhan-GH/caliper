// The viewer's stage entry points (stages.h, design/gpu-programs.md, Stage Declaration).
//
// A stage program is one WGSL module: the prelude of the sandbox, the stage sources, the WGSL of
// the primitive's built-in program (src/host/gfx/shaders, as generated), and the text below. The
// entry points here repeat the built-in vertex function of the primitive with the stage's
// vertex() between the batch's position and the projection, and the built-in fragment functions
// with the stage's material() on the colour. Without a vertex stage, vs_stage computes the same
// position, size and colour as the built-in entry, in the same order of operations, thus a
// material stage changes only the colour.
//
// The stage functions read the time and the step from the frame block's two pad words (the
// compiler fills them only for a frame with stages, CompileOptions::stage_time_s and
// stage_step), thus a stage program binds exactly the built-in program's resources: the frame
// block and the class table of group 0, the draw block and the colormap block of group 1, and
// the images of group 2. The extra varyings of the stage entries are the item's value and the
// entity-space position.
#include "scene/stages.h"

#include "scene/stage_cache.h"

namespace caliper_scene {

namespace {

// The functions both primitives share, and the varyings of the stage entries. The names start
// with caliper_ or Caliper, which a stage must not use.
constexpr const char* kCommon = R"wgsl(
// ---- stage entries (src/host/scene/stages.cpp)
fn caliper_stage_time() -> f32 { return bitcast<f32>(frame.pad0); }
fn caliper_stage_step() -> u32 { return frame.pad1; }
fn caliper_stage_value(src: u32) -> f32 { return select(0.0, bitcast<f32>(src), p.color_mode == 1u); }

struct CaliperStageOut {
  @builtin(position) pos: vec4<f32>,
  @location(0) color: vec4<f32>,
  @location(1) uv: vec2<f32>,
  @location(2) @interpolate(flat, either) pick: vec2<u32>,
  @location(3) obj: vec3<f32>,
  @location(4) value: f32,
};
)wgsl";

// Points: vs_points of scene/points.wgsl with the vertex stage on the position and the radius.
// The radius is in scene units for a batch with radii (radius_mode 1), else half the size in
// pixels.
constexpr const char* kPointsVertexHead = R"wgsl(
@vertex
fn vs_stage(in: PointIn) -> CaliperStageOut {
  var o: CaliperStageOut;
  let q = quad_corner(in.vid);
  let corner = vec2<f32>(q.x * 2.0 - 1.0, q.y);
  o.uv = corner;
  let item = p.item_base + in.iid;
  o.pick = vec2<u32>(p.pick_id, item);
  o.color = item_color(in.color_src);
  o.value = caliper_stage_value(in.color_src);
  var pos = in.pos;
  var radius = select(0.5 * p.size_px, in.radius, p.radius_mode == 1u);
)wgsl";

constexpr const char* kPointsVertexCall = R"wgsl(
  let staged = vertex(VertexIn(pos, o.value, radius, item, caliper_stage_time(), caliper_stage_step()));
  pos = staged.position;
  radius = staged.radius;
)wgsl";

constexpr const char* kPointsVertexTail = R"wgsl(
  o.obj = pos;
  if (any_nan3(pos)) { o.pos = hidden(); return o; }
  let c = p.mvp * vec4<f32>(pos, 1.0);
  var size = select(2.0 * radius, p.size_px, p.radius_mode == 1u);
  if (p.att_ref > 0.0) { size = size * clamp(p.att_ref / max(c.w, 1e-6), 0.5, 2.0); }
  if (p.fade_w1 > p.fade_w0) {
    let f = clamp((c.w - p.fade_w0) / (p.fade_w1 - p.fade_w0), 0.0, 1.0);
    o.color = vec4<f32>(o.color.rgb * (1.0 - 0.5 * f), o.color.a);
  }
  var k = (size / frame.viewport) * c.w;
  if (p.radius_mode == 1u) {
    k = max(radius * p.radius_scale * frame.proj_scale, c.w / frame.viewport);
  }
  o.pos = to_target(vec4<f32>(c.xy + corner * k, c.z, c.w));
  return o;
}
)wgsl";

// The normal of a point: a sphere's over the disc, facing the camera, in view space.
constexpr const char* kPointsFragment = R"wgsl(
fn caliper_stage_point_normal(uv: vec2<f32>) -> vec3<f32> {
  return normalize(vec3<f32>(uv, sqrt(max(1.0 - dot(uv, uv), 0.0))));
}

@fragment
fn fs_stage(in: CaliperStageOut) -> @location(0) vec4<f32> {
  if (p.point_shape == 1u && dot(in.uv, in.uv) > 1.0) { discard; }
  return caliper_stage_color(in.color, in.value, in.pick.y, in.obj, caliper_stage_point_normal(in.uv));
}

@fragment
fn fs_stage_pick(in: CaliperStageOut) -> @location(0) vec2<u32> {
  if (p.point_shape == 1u && dot(in.uv, in.uv) > 1.0) { discard; }
  return in.pick;
}
)wgsl";

// Lines: vs_lines of scene/lines.wgsl for segments (line_mode 0; arrows take no stage), with the
// vertex stage on both ends. The radius is half the width in pixels; the end of the vertex
// gives the width.
constexpr const char* kLinesVertexHead = R"wgsl(
@vertex
fn vs_stage(in: LineIn) -> CaliperStageOut {
  var o: CaliperStageOut;
  let item = p.item_base + in.iid;
  o.pick = vec2<u32>(p.pick_id, item);
  let at_b = quad_corner(in.vid % 6u).x > 0.5;
  let src = select(in.color_a, in.color_b, at_b);
  o.color = item_color(src);
  o.value = caliper_stage_value(src);
  var pa = in.a;
  var pb = in.b;
  var width = p.size_px;
)wgsl";

constexpr const char* kLinesVertexCall = R"wgsl(
  let now = caliper_stage_time();
  let step = caliper_stage_step();
  let staged_a = vertex(VertexIn(pa, caliper_stage_value(in.color_a), 0.5 * p.size_px, item, now, step));
  let staged_b = vertex(VertexIn(pb, caliper_stage_value(in.color_b), 0.5 * p.size_px, item, now, step));
  pa = staged_a.position;
  pb = staged_b.position;
  width = 2.0 * select(staged_a.radius, staged_b.radius, at_b);
)wgsl";

constexpr const char* kLinesVertexTail = R"wgsl(
  o.obj = select(pa, pb, at_b);
  if (in.keep == 0u || any_nan3(pa) || any_nan3(pb)) { o.pos = hidden(); return o; }
  let a = p.mvp * vec4<f32>(pa, 1.0);
  let b = p.mvp * vec4<f32>(pb, 1.0);
  o.pos = segment_corner(a, b, in.vid, width);
  return o;
}
)wgsl";

// The normal of a line faces the camera, in view space.
constexpr const char* kLinesFragment = R"wgsl(
@fragment
fn fs_stage(in: CaliperStageOut) -> @location(0) vec4<f32> {
  return caliper_stage_color(in.color, in.value, in.pick.y, in.obj, vec3<f32>(0.0, 0.0, 1.0));
}

@fragment
fn fs_stage_pick(in: CaliperStageOut) -> @location(0) vec2<u32> { return in.pick; }
)wgsl";

constexpr const char* kColorMaterial = R"wgsl(
fn caliper_stage_color(color: vec4<f32>, value: f32, item: u32, obj: vec3<f32>, normal: vec3<f32>) -> vec4<f32> {
  return material(MaterialIn(color, value, item, obj, normal, caliper_stage_time(), caliper_stage_step()));
}
)wgsl";

constexpr const char* kColorBuiltIn = R"wgsl(
fn caliper_stage_color(color: vec4<f32>, value: f32, item: u32, obj: vec3<f32>, normal: vec3<f32>) -> vec4<f32> {
  return color;
}
)wgsl";

}  // namespace

std::string stage_entries(Primitive primitive, bool material, bool vertex) {
    if (!material && !vertex) return std::string();
    std::string out;
    switch (primitive) {
        case Primitive::Points:
            out = kCommon;
            out += kPointsVertexHead;
            if (vertex) out += kPointsVertexCall;
            out += kPointsVertexTail;
            out += kPointsFragment;
            break;
        case Primitive::Lines:
            out = kCommon;
            out += kLinesVertexHead;
            if (vertex) out += kLinesVertexCall;
            out += kLinesVertexTail;
            out += kLinesFragment;
            break;
        default:
            return std::string();
    }
    out += material ? kColorMaterial : kColorBuiltIn;
    return out;
}

// ---- StageCache (stage_cache.h)

StageCache::StageCache(caliper_gfx::Backend& backend) : backend_(backend) {}

StageCache::~StageCache() {
    for (auto& e : entries_)
        if (e.second.program.program) backend_.release(e.second.program.program);
}

void StageCache::end_frame() {
    for (auto it = entries_.begin(); it != entries_.end();) {
        if (it->second.used == frame_) {
            ++it;
            continue;
        }
        if (it->second.program.program) backend_.release(it->second.program.program);
        it = entries_.erase(it);
    }
}

bool StageCache::animated() const {
    for (const auto& e : entries_)
        if (e.second.ok && e.second.used == frame_ && e.second.build->animated) return true;
    return false;
}

const StageProgram* StageCache::find(std::string_view path, Primitive primitive) const {
    if (!source_ || (primitive != Primitive::Points && primitive != Primitive::Lines)) return nullptr;
    std::shared_ptr<const StageBuild> build = source_->build(path, primitive);
    if (!build || build->msl.empty()) return nullptr;
    std::string key(path);
    key += '\0';
    key += char('0' + int(primitive));
    Entry& e = entries_[key];
    e.used = frame_;
    if (e.build == build) {
        // The params' values of this frame: a new value needs no compile.
        if (e.ok && !build->params.empty()) e.program.params = source_->params(path);
        return e.ok ? &e.program : nullptr;
    }
    if (e.program.program) backend_.release(e.program.program);
    e = Entry{};
    e.used = frame_;
    e.build = build;
    if (!build->params.empty() && !params_bound_) {
        source_->refused(*build, "This scene view does not bind the uniform params of a stage yet, so the entity "
                                 "keeps its built-in program.");
        return nullptr;
    }
    const caliper_gfx::EntryDesc entries[3] = {
        {kStageVertexEntry, uint8_t(caliper_gfx::kStageVertex), {1, 1, 1}},
        {kStageFragmentEntry, uint8_t(caliper_gfx::kStageFragment), {1, 1, 1}},
        {kStagePickEntry, uint8_t(caliper_gfx::kStageFragment), {1, 1, 1}},
    };
    caliper_gfx::ProgramDesc d;
    d.label = build->label.c_str();
    d.entries = entries;
    d.entry_count = 3;
    d.bindings = build->bindings.data();
    d.binding_count = uint8_t(build->bindings.size());
    d.wgsl = build->wgsl.c_str();
    d.msl = build->msl.c_str();
    e.program.program = backend_.create_program(d);
    if (!e.program.program) {
        source_->refused(*build, backend_.last_error());
        return nullptr;
    }
    e.program.vs = kStageVertexEntry;
    e.program.fs = kStageFragmentEntry;
    e.program.fs_pick = kStagePickEntry;
    e.program.material = build->material;
    e.program.vertex = build->vertex;
    e.program.source_hash = build->source_hash;
    if (!build->params.empty()) e.program.params = source_->params(path);
    e.ok = true;
    return &e.program;
}

}  // namespace caliper_scene
