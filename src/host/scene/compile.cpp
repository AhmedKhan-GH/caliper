// compile.cpp: a scene into a draw list (compile.h).
#include "scene/compile.h"

#include <algorithm>
#include <cmath>
#include <cstring>

namespace caliper_scene {

using caliper_gfx::BindEntry;
using caliper_gfx::BindGroup;
using caliper_gfx::Buffer;
using caliper_gfx::DrawList;
using caliper_gfx::DrawRecord;
using caliper_gfx::Format;
using caliper_gfx::Frame;
using caliper_gfx::kArena;
using caliper_gfx::kMaxVertexBuffers;
using caliper_gfx::kNoOffset;
using caliper_gfx::RenderPipelineDesc;
using caliper_gfx::Step;
using caliper_gfx::Texture;
using caliper_gfx::VertexBinding;
using caliper_gfx::VertexFormat;
using caliper_gfx::VertexLayout;
namespace sh = caliper_gfx::shaders;

// ---- Constant streams --------------------------------------------------------------------------------

void StaticGeometry::fill(uint8_t* out) {
    std::memset(out, 0, kBytes);
    const float identity[16] = {1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1};
    std::memcpy(out + kIdentityPose, identity, sizeof identity);
    const uint32_t keep = 0xffffffffu;
    std::memcpy(out + kKeep, &keep, 4);
    const float corner[4][3] = {{0, 0, 1}, {1, 0, 1}, {1, 1, 1}, {0, 1, 1}};
    float frustum[16][3] = {};
    for (int k = 0; k < 4; ++k) {
        std::memcpy(frustum[2 * k + 1], corner[k], 12);                    // centre to corner k
        std::memcpy(frustum[8 + 2 * k], corner[k], 12);                    // side k of the plane
        std::memcpy(frustum[8 + 2 * k + 1], corner[(k + 1) % 4], 12);
    }
    std::memcpy(out + kFrustum, frustum, sizeof frustum);
    const float axes[6][3] = {{0, 0, 0}, {1, 0, 0}, {0, 0, 0}, {0, 1, 0}, {0, 0, 0}, {0, 0, 1}};
    std::memcpy(out + kAxes, axes, sizeof axes);
    const uint32_t colors[6] = {0xff0000ffu, 0xff0000ffu, 0xff00ff00u, 0xff00ff00u, 0xffff0000u, 0xffff0000u};
    std::memcpy(out + kAxesColors, colors, sizeof colors);
}

// ---- Checking layouts against the generated description --------------------------------------------------

namespace {

bool fail(std::string* error, std::string text) {
    if (error) *error = std::move(text);
    return false;
}

bool converts(VertexFormat given, VertexFormat shader) {
    return given == shader || (shader == VertexFormat::F32x3 && given == VertexFormat::F32x2) ||
           (shader == VertexFormat::U32 && given == VertexFormat::U8);
}

const sh::ProgramDescription& describe(ProgramId p) { return sh::kProgramDescriptions[uint32_t(p)]; }

bool has_entry(const sh::ProgramDescription& d, const char* name, uint8_t stage) {
    for (uint8_t i = 0; i < d.entry_count; ++i)
        if (std::strcmp(d.entries[i].name, name) == 0 && d.entries[i].stage == stage) return true;
    return false;
}

bool integer_format(Format f) { return f == Format::R16Uint || f == Format::R32Uint || f == Format::RG32Uint; }

}  // namespace

bool check_vertex_layouts(ProgramId program, const RenderPipelineDesc& desc, std::string* error) {
    const sh::ProgramDescription& d = describe(program);
    if (!desc.vertex_entry || !has_entry(d, desc.vertex_entry, caliper_gfx::kStageVertex))
        return fail(error, std::string(d.name) + " has no vertex entry " + (desc.vertex_entry ? desc.vertex_entry : "?"));
    if (!desc.fragment_entry || !has_entry(d, desc.fragment_entry, caliper_gfx::kStageFragment))
        return fail(error, std::string(d.name) + " has no fragment entry " +
                               (desc.fragment_entry ? desc.fragment_entry : "?"));
    // Every input of the entry is fed once, with a format that converts to it.
    for (uint8_t i = 0; i < d.vertex_input_count; ++i) {
        const sh::VertexInputDesc& in = d.vertex_inputs[i];
        if (std::strcmp(in.entry, desc.vertex_entry) != 0) continue;
        int fed = 0;
        for (uint8_t l = 0; l < desc.vertex_layout_count; ++l)
            for (uint8_t a = 0; a < desc.vertex_layouts[l].attribute_count; ++a) {
                const auto& at = desc.vertex_layouts[l].attributes[a];
                if (at.location != in.location) continue;
                ++fed;
                if (!converts(at.format, in.format))
                    return fail(error, std::string(desc.vertex_entry) + " location " + std::to_string(in.location) +
                                           " gets a format that does not convert to its input");
            }
        if (fed != 1)
            return fail(error, std::string(desc.vertex_entry) + " location " + std::to_string(in.location) +
                                   (fed == 0 ? " is not fed" : " is fed twice"));
    }
    // No attribute feeds a location the entry does not read.
    for (uint8_t l = 0; l < desc.vertex_layout_count; ++l)
        for (uint8_t a = 0; a < desc.vertex_layouts[l].attribute_count; ++a) {
            const uint32_t loc = desc.vertex_layouts[l].attributes[a].location;
            bool read = false;
            for (uint8_t i = 0; i < d.vertex_input_count; ++i)
                read = read || (std::strcmp(d.vertex_inputs[i].entry, desc.vertex_entry) == 0 &&
                                d.vertex_inputs[i].location == loc);
            if (!read) return fail(error, std::string(desc.vertex_entry) + " reads no location " + std::to_string(loc));
        }
    // The fragment output's kind matches the target's.
    for (uint8_t i = 0; i < d.fragment_output_count; ++i) {
        const sh::FragmentOutputDesc& o = d.fragment_outputs[i];
        if (std::strcmp(o.entry, desc.fragment_entry) != 0) continue;
        if (integer_format(o.format) != integer_format(desc.color_format))
            return fail(error, std::string(desc.fragment_entry) + " writes another kind of target");
    }
    return true;
}

// ---- Compiling ---------------------------------------------------------------------------------------------------

uint64_t SceneCompiler::key(EntityId e, Primitive p, uint32_t which) {
    return (uint64_t(e.index) << 32) | (uint64_t(e.generation & 0xffffu) << 16) | (uint64_t(p) << 4) | (which & 15u);
}

namespace {

struct Spec {
    ProgramId program = ProgramId::scene_points;
    const char* vs = nullptr;
    const char* fs = nullptr;
    const char* fs_pick = nullptr;
    uint8_t layout_count = 0;
    VertexLayout layouts[kMaxVertexBuffers];
    VertexBinding vertex[kMaxVertexBuffers];
    Buffer index_buffer;
    uint32_t count = 0, instance_count = 1;
    Texture image;
    uint32_t image_binding = kImageRgba;
    sh::PrimParams params{};
    Mat4 model;                  // world, or the pixel-space local matrix of an overlay
    uint32_t dims = 3;
    bool translucent = false;
    bool overlay = false;
    uint32_t colormap = 0;
    float vmin = 0, vmax = 1;
    bool uses_cmap = false;
    DrawMapEntry entry;
};

void stream(Spec& s, uint32_t location, VertexFormat fmt, Step step, Buffer buf, uint64_t offset, uint32_t stride) {
    VertexLayout& l = s.layouts[s.layout_count];
    l = VertexLayout{};
    l.stride = stride;
    l.step = step;
    l.attribute_count = 1;
    l.attributes[0] = {location, fmt, 0};
    s.vertex[s.layout_count] = VertexBinding{buf, offset};
    ++s.layout_count;
}

uint32_t stride_of(const BatchArray& a) { return a.count == 1 ? 0 : a.stride; }

struct ColorSource { uint32_t mode; const BatchArray* array; };
ColorSource color_source(const Batch& b) {
    if (const BatchArray* a = b.array(Role::Colors)) return {2, a};
    if (const BatchArray* a = b.array(Role::Values)) return {1, a};
    if (const BatchArray* a = b.array(Role::ClassIds)) return {4, a};
    return {0, nullptr};
}

// The largest length of the first n columns of m's linear part: the scale a radius takes.
double column_scale(const Mat4& m, int n) {
    double s = 0;
    for (int c = 0; c < n; ++c) s = std::max(s, length(Vec3{m.at(0, c), m.at(1, c), m.at(2, c)}));
    return s;
}

uint32_t tint(float opacity) {
    const uint32_t a = uint32_t(std::lround(std::clamp(opacity, 0.0f, 1.0f) * 255.0f));
    return (a << 24) | 0x00ffffffu;
}

bool declares_group(ProgramId p, uint32_t group) {
    const sh::ProgramDescription& d = describe(p);
    for (uint8_t i = 0; i < d.binding_count; ++i)
        if (d.bindings[i].group == group) return true;
    return false;
}

}  // namespace

bool SceneCompiler::is_prepared(uint64_t k, const BatchArray& a, uint64_t seq) const {
    const auto it = prepared_.find(k);
    return it != prepared_.end() && it->second.seq == seq && it->second.buffer == a.buffer.id &&
           it->second.offset == a.offset && it->second.count == a.count;
}

void SceneCompiler::prepare(const SceneTree& tree, Resources& res, Frame& frame) {
    fresh_.clear();
    for (uint32_t i = 0; i < tree.entity_count(); ++i) {
        const EntityId e = tree.entity_at(i);
        // The index clamp of a mesh: 3T indices, each clamped to the last vertex.
        if (const Batch* m = tree.batch(e, Primitive::Mesh)) {
            const BatchArray* idx = m->array(Role::Indices);
            const uint64_t k = key(e, Primitive::Mesh, 0);
            if (idx && idx->count > 0 && m->count() > 0 && !is_prepared(k, *idx, m->seq)) {
                const uint64_t n = idx->count * 3, bytes = n * 4;
                const Buffer out = res.owned_buffer(k, bytes, caliper_gfx::kIndex | caliper_gfx::kStorage);
                const sh::ClampParams cp{uint32_t(n), uint32_t(m->count() - 1), 0, 0};
                const BindEntry entries[3] = {{0, kArena, 0, sizeof cp, {}, {}},
                                              {1, idx->buffer, idx->offset, bytes, {}, {}},
                                              {2, out, 0, bytes, {}, {}}};
                const BindGroup g = res.compute_group(ProgramId::index_clamp, caliper_gfx::View<BindEntry>(entries, 3));
                const uint32_t off = frame.uniform(&cp, sizeof cp);
                const BindGroup groups[1] = {g};
                const uint32_t offsets[1] = {off};
                frame.dispatch(res.compute_pipeline(ProgramId::index_clamp, "cs_clamp_indices"), groups, offsets,
                               uint32_t(std::min<uint64_t>((n + 63) / 64, 65535)), 1, 1);
                prepared_[k] = {m->seq, idx->buffer.id, idx->offset, idx->count};
                fresh_.push_back(k);
            }
        }
        // The edge gather of index-pair edges: both ends' positions and per-vertex colour sources.
        if (const Batch* l = tree.batch(e, Primitive::Lines); l && l->line_mode == LineMode::Edges) {
            const BatchArray* pairs = l->array(Role::Indices);
            const BatchArray* pos = l->array(Role::Positions);
            const uint64_t k = key(e, Primitive::Lines, 1);
            if (pairs && pos && pairs->count > 0 && pos->count > 0 && !is_prepared(k, *pairs, l->seq)) {
                const ColorSource cs = color_source(*l);
                const bool per_vertex = cs.array && !l->per_edge && cs.array->count > 1;
                const uint32_t dims = l->dims();
                const uint64_t E = pairs->count;
                const Buffer seg_pos = res.owned_buffer(k, E * 2 * dims * 4, caliper_gfx::kVertex | caliper_gfx::kStorage);
                const Buffer seg_src = res.owned_buffer(key(e, Primitive::Lines, 2), per_vertex ? E * 2 * 4 : 16,
                                                        caliper_gfx::kVertex | caliper_gfx::kStorage);
                const sh::GatherParams gp{uint32_t(E), uint32_t(pos->count - 1), dims, per_vertex ? 1u : 0u};
                const BindEntry entries[6] = {
                    {0, kArena, 0, sizeof gp, {}, {}},
                    {1, pairs->buffer, pairs->offset, E * 8, {}, {}},
                    {2, pos->buffer, pos->offset, pos->count * dims * 4, {}, {}},
                    // vert_src is read only when per_vertex is 1; otherwise the positions stand in.
                    per_vertex ? BindEntry{3, cs.array->buffer, cs.array->offset, cs.array->count * 4, {}, {}}
                               : BindEntry{3, pos->buffer, pos->offset, pos->count * dims * 4, {}, {}},
                    {4, seg_pos, 0, E * 2 * dims * 4, {}, {}},
                    {5, seg_src, 0, per_vertex ? E * 2 * 4 : 16, {}, {}}};
                const BindGroup g = res.compute_group(ProgramId::edge_gather, caliper_gfx::View<BindEntry>(entries, 6));
                const uint32_t off = frame.uniform(&gp, sizeof gp);
                const BindGroup groups[1] = {g};
                const uint32_t offsets[1] = {off};
                frame.dispatch(res.compute_pipeline(ProgramId::edge_gather, "cs_gather_edges"), groups, offsets,
                               uint32_t(std::min<uint64_t>((E + 63) / 64, 65535)), 1, 1);
                prepared_[k] = {l->seq, pairs->buffer.id, pairs->offset, pairs->count};
                fresh_.push_back(k);
            }
        }
    }
    // The owned buffers of batches that are gone (a reset gives every entity a new id) go back to
    // the layer, and so does what was prepared for them.
    std::vector<uint64_t> live;
    for (uint32_t i = 0; i < tree.entity_count(); ++i) {
        const EntityId e = tree.entity_at(i);
        if (const Batch* m = tree.batch(e, Primitive::Mesh); m && m->array(Role::Indices))
            live.push_back(key(e, Primitive::Mesh, 0));
        if (const Batch* l = tree.batch(e, Primitive::Lines);
            l && l->line_mode == LineMode::Edges && l->array(Role::Indices)) {
            live.push_back(key(e, Primitive::Lines, 1));
            live.push_back(key(e, Primitive::Lines, 2));
        }
    }
    res.retain_owned(caliper_gfx::View<uint64_t>(live));
    if (prepared_.size() > live.size()) {
        std::sort(live.begin(), live.end());
        for (auto it = prepared_.begin(); it != prepared_.end();)
            it = std::binary_search(live.begin(), live.end(), it->first) ? std::next(it) : prepared_.erase(it);
    }
}

void SceneCompiler::discard_prepared() {
    for (const uint64_t k : fresh_) prepared_.erase(k);
    fresh_.clear();
}

CompileResult SceneCompiler::compile(const SceneTree& tree, const ViewCamera& camera, const CompileOptions& options,
                                     Resources& res, DrawList& out, DrawMap* map) {
    return run(tree, camera, options, nullptr, res, out, map);
}

CompileResult SceneCompiler::compile_pick(const SceneTree& tree, const ViewCamera& camera,
                                          const CompileOptions& options, const int32_t window[4], Resources& res,
                                          DrawList& out, DrawMap* map) {
    return run(tree, camera, options, window, res, out, map);
}

CompileResult SceneCompiler::run(const SceneTree& tree, const ViewCamera& camera, const CompileOptions& opts,
                                 const int32_t* window, Resources& res, DrawList& out, DrawMap* map) {
    CompileResult result;
    const bool pick = window != nullptr;
    if (opts.id_target && !pick) {
        result.notes.push_back("an ID attachment beside the colour needs fragment entries that write both, "
                               "which the programs do not have; use the pick pass");
        return result;
    }
    if (opts.width == 0 || opts.height == 0) {
        result.notes.push_back("the target has no pixels");
        return result;
    }
    const CameraMatrices cm = camera_matrices(camera, tree, opts.width, opts.height);
    if (!cm.ok) {
        result.notes.push_back(cm.note);
        return result;
    }
    const bool three_d = tree.space() == Space::k3D;
    const bool has_depth = bool(out.target().depth);
    const Buffer geo = res.static_geometry();
    const double radius = tree.scene_bounds().valid ? tree.scene_bounds().radius() : 1.0;

    // The frame block.
    sh::Frame fb{};
    fb.viewport[0] = float(opts.width);
    fb.viewport[1] = float(opts.height);
    fb.proj_scale[0] = float(std::fabs(cm.proj.at(0, 0)));
    fb.proj_scale[1] = float(std::fabs(cm.proj.at(1, 1)));
    fb.ndc_scale[0] = fb.ndc_scale[1] = 1.0f;
    if (pick) {
        const double W = opts.width, H = opts.height, x0 = window[0], y0 = window[1], ww = window[2], wh = window[3];
        fb.ndc_scale[0] = float(W / ww);
        fb.ndc_scale[1] = float(H / wh);
        fb.ndc_offset[0] = float((W - 2 * x0) / ww - 1);
        fb.ndc_offset[1] = float(1 - (H - 2 * y0) / wh);
    }
    fb.class_count = 65536;
    fb.class_default = kDefaultRgba;
    const uint32_t frame_off = out.uniform(&fb, sizeof fb);

    std::vector<Spec> specs;
    auto note = [&](EntityId e, const std::string& text) {
        result.notes.push_back(tree.path(e) + ": " + text);
        ++result.skipped;
    };

    for (uint32_t idx : tree.path_order()) {
        const EntityId e = tree.entity_at(idx);
        if (!tree.drawable(e)) continue;
        const EntityId pcam = tree.pixel_space_of(e);
        const bool overlay = pcam && pcam == cm.through;
        const bool pixel = bool(pcam);
        const Mat4& world = tree.world(e);
        const uint32_t dims = (three_d && !pixel) ? 3 : 2;
        auto base = [&](Primitive p, DrawPart part) {
            Spec s;
            s.model = overlay ? tree.pixel_local(e) : world;
            s.overlay = overlay;
            s.dims = dims;
            s.entry.entity = e;
            s.entry.primitive = p;
            s.entry.part = part;
            s.entry.overlay = overlay;
            return s;
        };
        auto color_streams = [&](Spec& s, const Batch& b, uint32_t loc, Step step) {
            const ColorSource cs = color_source(b);
            s.params.color_mode = cs.mode;
            s.params.flat_rgba = b.flat_rgba;
            if (cs.array) stream(s, loc, VertexFormat::U32, step, cs.array->buffer, cs.array->offset, stride_of(*cs.array));
            else stream(s, loc, VertexFormat::U32, step, geo, StaticGeometry::kZero, 0);
            if (cs.mode == 1) {
                s.uses_cmap = true;
                s.colormap = b.colormap;
                s.vmin = b.vmin;
                s.vmax = b.vmax;
            }
            s.params.vmin = b.vmin;
            s.params.vmax = b.vmax;
            s.translucent = b.translucent || (cs.mode == 0 && (b.flat_rgba >> 24) < 255);
            return cs;
        };

        for (int k = 1; k <= kPrimitiveCount; ++k) {
            const Primitive prim = Primitive(k);
            const Batch* bp = tree.batch(e, prim);
            if (!bp) continue;
            const Batch& b = *bp;
            const uint64_t n = b.count();
            if (n == 0) continue;
            switch (prim) {
                case Primitive::Points: {
                    Spec s = base(prim, DrawPart::Batch);
                    s.program = ProgramId::scene_points;
                    s.vs = "vs_points";
                    s.fs = "fs_points";
                    s.fs_pick = "fs_points_pick";
                    const BatchArray& pos = *b.array(Role::Positions);
                    stream(s, 0, pos.format, Step::Instance, pos.buffer, pos.offset, stride_of(pos));
                    color_streams(s, b, 1, Step::Instance);
                    if (const BatchArray* r = b.array(Role::Radii)) {
                        stream(s, 2, VertexFormat::F32, Step::Instance, r->buffer, r->offset, stride_of(*r));
                        s.params.radius_mode = 1;
                    } else {
                        stream(s, 2, VertexFormat::F32, Step::Instance, geo, StaticGeometry::kZero, 0);
                    }
                    s.params.size_px = (b.size_px > 0 ? b.size_px : default_point_size(n)) * opts.pixel_scale;
                    s.params.point_shape = uint32_t(b.point_shape);
                    s.count = 6;
                    s.instance_count = uint32_t(n);
                    s.entry.items = n;
                    specs.push_back(s);
                    break;
                }
                case Primitive::Lines: {
                    Spec s = base(prim, DrawPart::Batch);
                    s.program = ProgramId::scene_lines;
                    s.vs = "vs_lines";
                    s.fs = "fs_color";
                    s.fs_pick = "fs_pick";
                    const BatchArray& pos = *b.array(Role::Positions);
                    const ColorSource cs = color_source(b);
                    const uint32_t ps = pos.stride;
                    if (b.line_mode == LineMode::Edges) {
                        const BatchArray& pairs = *b.array(Role::Indices);
                        if (!is_prepared(key(e, prim, 1), pairs, b.seq)) {
                            note(e, "lines wait for their edge gather (prepare)");
                            break;
                        }
                        const uint64_t E = pairs.count;
                        const uint32_t seg = 2 * b.dims() * 4;
                        const Buffer sp = res.owned_buffer(key(e, prim, 1), E * seg, caliper_gfx::kVertex | caliper_gfx::kStorage);
                        stream(s, 0, pos.format, Step::Instance, sp, 0, seg);
                        stream(s, 1, pos.format, Step::Instance, sp, seg / 2, seg);
                        const bool per_vertex = cs.array && !b.per_edge && cs.array->count > 1;
                        color_streams(s, b, 2, Step::Instance);   // per edge, one or none: both ends read it
                        if (per_vertex) {
                            const Buffer src = res.owned_buffer(key(e, prim, 2), E * 8, caliper_gfx::kVertex | caliper_gfx::kStorage);
                            s.vertex[2] = VertexBinding{src, 0};
                            s.layouts[2].stride = 8;
                            stream(s, 3, VertexFormat::U32, Step::Instance, src, 4, 8);
                        } else {
                            stream(s, 3, VertexFormat::U32, Step::Instance, s.vertex[2].buffer, s.vertex[2].offset,
                                   s.layouts[2].stride);
                        }
                        stream(s, 4, VertexFormat::U32, Step::Instance, geo, StaticGeometry::kKeep, 0);
                        s.instance_count = uint32_t(E);
                    } else {
                        const bool strips = b.line_mode == LineMode::Strips;
                        const uint32_t step = strips ? ps : 2 * ps;
                        if (step > 2048) {
                            note(e, "lines: a segment's stride above 2,048 bytes cannot bind");
                            break;
                        }
                        const uint64_t inst = strips ? (n >= 2 ? n - 1 : 0) : n / 2;
                        if (inst == 0) break;
                        stream(s, 0, pos.format, Step::Instance, pos.buffer, pos.offset, step);
                        stream(s, 1, pos.format, Step::Instance, pos.buffer, pos.offset + ps, step);
                        color_streams(s, b, 2, Step::Instance);
                        if (cs.array && cs.array->count > 1) {
                            const uint32_t c = cs.array->stride, cstep = strips ? c : 2 * c;
                            s.layouts[2].stride = cstep;
                            stream(s, 3, VertexFormat::U32, Step::Instance, cs.array->buffer, cs.array->offset + c, cstep);
                        } else {
                            stream(s, 3, VertexFormat::U32, Step::Instance, s.vertex[2].buffer, s.vertex[2].offset, 0);
                        }
                        if (strips) {
                            const BatchArray& m = *b.array(Role::BreakMask);
                            stream(s, 4, m.format, Step::Instance, m.buffer, m.offset, m.stride);
                        } else {
                            stream(s, 4, VertexFormat::U32, Step::Instance, geo, StaticGeometry::kKeep, 0);
                        }
                        s.instance_count = uint32_t(inst);
                    }
                    s.params.size_px = b.width_px * opts.pixel_scale;
                    s.params.line_mode = 0;
                    s.count = 6;
                    s.entry.items = s.instance_count;
                    specs.push_back(s);
                    break;
                }
                case Primitive::Arrows: {
                    Spec s = base(prim, DrawPart::Batch);
                    s.program = ProgramId::scene_lines;
                    s.vs = "vs_lines";
                    s.fs = "fs_color";
                    s.fs_pick = "fs_pick";
                    const BatchArray& o = *b.array(Role::Origins);
                    const BatchArray& v = *b.array(Role::Vectors);
                    stream(s, 0, o.format, Step::Instance, o.buffer, o.offset, stride_of(o));
                    stream(s, 1, v.format, Step::Instance, v.buffer, v.offset, stride_of(v));
                    color_streams(s, b, 2, Step::Instance);
                    stream(s, 3, VertexFormat::U32, Step::Instance, s.vertex[2].buffer, s.vertex[2].offset, s.layouts[2].stride);
                    stream(s, 4, VertexFormat::U32, Step::Instance, geo, StaticGeometry::kKeep, 0);
                    s.params.size_px = b.width_px * opts.pixel_scale;
                    s.params.line_mode = 1;
                    s.count = 9;
                    s.instance_count = uint32_t(n);
                    s.entry.items = n;
                    specs.push_back(s);
                    break;
                }
                case Primitive::Boxes: {
                    Spec s = base(prim, DrawPart::Batch);
                    s.program = ProgramId::scene_boxes;
                    s.vs = "vs_box_wire";
                    s.fs = "fs_color";
                    s.fs_pick = "fs_pick";
                    const BatchArray& c = *b.array(Role::Centers);
                    const BatchArray& h = *b.array(Role::HalfSizes);
                    stream(s, 0, c.format, Step::Instance, c.buffer, c.offset, stride_of(c));
                    stream(s, 1, h.format, Step::Instance, h.buffer, h.offset, stride_of(h));
                    if (const BatchArray* r = b.array(Role::Rotations))
                        stream(s, 2, VertexFormat::F32x4, Step::Instance, r->buffer, r->offset, stride_of(*r));
                    else
                        stream(s, 2, VertexFormat::F32x4, Step::Instance, geo, StaticGeometry::kIdentityQuat, 0);
                    color_streams(s, b, 3, Step::Instance);
                    s.params.size_px = b.width_px * opts.pixel_scale;
                    s.count = dims == 3 ? 72 : 24;
                    s.instance_count = uint32_t(n);
                    s.entry.items = n;
                    specs.push_back(s);
                    if (b.fill) {
                        Spec f = s;
                        f.vs = "vs_box_solid";
                        f.count = dims == 3 ? 36 : 6;
                        f.translucent = true;
                        f.entry.part = DrawPart::BoxFaces;
                        specs.push_back(f);
                    }
                    break;
                }
                case Primitive::Mesh: {
                    Spec s = base(prim, DrawPart::Batch);
                    s.program = ProgramId::scene_mesh;
                    s.vs = "vs_mesh";
                    s.fs_pick = "fs_pick";
                    const BatchArray& pos = *b.array(Role::Positions);
                    stream(s, 0, pos.format, Step::Vertex, pos.buffer, pos.offset, pos.stride);
                    const BatchArray* nrm = b.array(Role::Normals);
                    if (nrm) stream(s, 1, VertexFormat::F32x3, Step::Vertex, nrm->buffer, nrm->offset, nrm->stride);
                    else stream(s, 1, VertexFormat::F32x3, Step::Vertex, geo, StaticGeometry::kZero, 0);
                    const BatchArray* uv = b.array(Role::Uvs);
                    if (uv) {
                        stream(s, 2, VertexFormat::U32, Step::Vertex, geo, StaticGeometry::kZero, 0);
                        s.params.color_mode = 3;   // white, times the texture
                        s.params.flat_rgba = b.flat_rgba;
                        s.translucent = b.translucent;
                        stream(s, 3, VertexFormat::F32x2, Step::Vertex, uv->buffer, uv->offset, uv->stride);
                        s.image = b.texture;
                        s.image_binding = kImageRgba;
                        s.params.image_mode = b.image_form == ImageForm::Grey ? 1 : 0;
                        s.fs = "fs_mesh_tex";
                    } else {
                        color_streams(s, b, 2, Step::Vertex);
                        stream(s, 3, VertexFormat::F32x2, Step::Vertex, geo, StaticGeometry::kZero, 0);
                        s.fs = nrm ? "fs_color" : "fs_mesh_flat";
                    }
                    s.params.shade_mode = nrm ? 1 : 0;
                    VertexLayout& pose = s.layouts[s.layout_count];   // the v1.3 pose: identity at stride 0
                    pose = VertexLayout{};
                    pose.stride = 0;
                    pose.step = Step::Instance;
                    pose.attribute_count = 4;
                    for (uint32_t a = 0; a < 4; ++a) pose.attributes[a] = {4 + a, VertexFormat::F32x4, 16 * a};
                    s.vertex[s.layout_count] = VertexBinding{geo, StaticGeometry::kIdentityPose};
                    ++s.layout_count;
                    stream(s, 8, VertexFormat::U32, Step::Instance, geo, StaticGeometry::kZero, 0);
                    s.params.vertex_count = uint32_t(n);
                    if (const BatchArray* idx = b.array(Role::Indices)) {
                        if (!is_prepared(key(e, prim, 0), *idx, b.seq)) {
                            note(e, "mesh waits for its index clamp (prepare)");
                            break;
                        }
                        s.index_buffer = res.owned_buffer(key(e, prim, 0), idx->count * 12,
                                                          caliper_gfx::kIndex | caliper_gfx::kStorage);
                        s.count = uint32_t(idx->count * 3);
                    } else {
                        s.count = uint32_t(n - n % 3);
                    }
                    if (s.count == 0) break;
                    s.entry.items = n;
                    specs.push_back(s);
                    break;
                }
                case Primitive::Image: {
                    if (three_d && !pixel) {
                        note(e, "an image in a 3-D scene is drawn only under a camera");
                        break;
                    }
                    Spec s = base(prim, DrawPart::Batch);
                    s.program = ProgramId::scene_image;
                    s.vs = "vs_image";
                    s.fs_pick = "fs_pick";
                    s.image = b.texture;
                    s.params.flat_rgba = tint(b.opacity);
                    s.translucent = b.opacity < 1.0f || b.translucent;
                    switch (b.image_form) {
                        case ImageForm::Color:
                        case ImageForm::Grey:
                            s.fs = "fs_image_rgba";
                            s.image_binding = kImageRgba;
                            s.params.image_mode = b.image_form == ImageForm::Grey ? 1 : 0;
                            break;
                        case ImageForm::Depth:
                            s.fs = "fs_image_value";
                            s.image_binding = kImageValue;
                            s.uses_cmap = true;
                            s.colormap = b.colormap;
                            s.vmin = s.params.vmin = b.vmin;
                            s.vmax = s.params.vmax = b.vmax;
                            s.params.color_mode = 1;
                            break;
                        case ImageForm::Segmentation:
                            s.fs = "fs_image_class";
                            s.image_binding = kImageClass;
                            s.params.color_mode = 4;
                            break;
                    }
                    s.model = s.model * scaling({double(b.image_width), double(b.image_height), 1});
                    s.count = 6;
                    s.entry.items = 1;
                    s.entry.image_size[0] = b.image_width;
                    s.entry.image_size[1] = b.image_height;
                    specs.push_back(s);
                    // A depth image under a camera: also its back-projected cloud, in 3-D.
                    if (b.image_form == ImageForm::Depth && b.cloud && three_d && pixel) {
                        const Pinhole& pin = *tree.camera(pcam);
                        Spec c = base(prim, DrawPart::DepthCloud);
                        c.overlay = false;
                        c.entry.overlay = false;
                        c.dims = 3;
                        c.program = ProgramId::scene_points;
                        c.vs = "vs_depth_points";
                        c.fs = "fs_points";
                        c.fs_pick = "fs_points_pick";
                        c.image = b.texture;
                        c.image_binding = kImageValue;
                        Mat4 kinv;
                        kinv.at(0, 0) = 1.0 / pin.fx;
                        kinv.at(0, 2) = -pin.cx / pin.fx;
                        kinv.at(1, 1) = 1.0 / pin.fy;
                        kinv.at(1, 2) = -pin.cy / pin.fy;
                        const Mat4& t = tree.pixel_local(e);
                        Mat4 lift;
                        lift.at(0, 0) = t.at(0, 0);
                        lift.at(0, 1) = t.at(0, 1);
                        lift.at(0, 2) = t.at(0, 3);
                        lift.at(1, 0) = t.at(1, 0);
                        lift.at(1, 1) = t.at(1, 1);
                        lift.at(1, 2) = t.at(1, 3);
                        c.model = tree.camera_frame(pcam) * kinv * lift;
                        c.uses_cmap = true;
                        c.colormap = b.colormap;
                        c.vmin = c.params.vmin = b.vmin;
                        c.vmax = c.params.vmax = b.vmax;
                        c.params.color_mode = 1;
                        c.params.flat_rgba = b.flat_rgba;
                        const uint64_t px = uint64_t(b.image_width) * b.image_height;
                        c.params.size_px = default_point_size(px) * opts.pixel_scale;
                        c.params.point_shape = uint32_t(b.point_shape);
                        c.count = 6;
                        c.instance_count = uint32_t(px);
                        c.entry.items = px;
                        specs.push_back(c);
                    }
                    break;
                }
            }
        }

        // A camera entity's frustum, unless the view looks through it.
        const Pinhole* pin = tree.camera(e);
        if (pin && e != cm.through && three_d) {
            Spec s = base(Primitive::Lines, DrawPart::Frustum);
            s.program = ProgramId::scene_lines;
            s.vs = "vs_lines";
            s.fs = "fs_color";
            s.fs_pick = "fs_pick";
            const double d = tree.plane_distance(e);
            Mat4 f;
            f.at(0, 0) = pin->width * d / pin->fx;
            f.at(0, 2) = -pin->cx * d / pin->fx;
            f.at(1, 1) = pin->height * d / pin->fy;
            f.at(1, 2) = -pin->cy * d / pin->fy;
            f.at(2, 2) = d;
            s.model = tree.camera_frame(e) * f;
            stream(s, 0, VertexFormat::F32x3, Step::Instance, geo, StaticGeometry::kFrustum, 24);
            stream(s, 1, VertexFormat::F32x3, Step::Instance, geo, StaticGeometry::kFrustum + 12, 24);
            stream(s, 2, VertexFormat::U32, Step::Instance, geo, StaticGeometry::kZero, 0);
            stream(s, 3, VertexFormat::U32, Step::Instance, geo, StaticGeometry::kZero, 0);
            stream(s, 4, VertexFormat::U32, Step::Instance, geo, StaticGeometry::kKeep, 0);
            s.params.color_mode = 0;
            s.params.flat_rgba = opts.camera_rgba;
            s.params.size_px = opts.camera_width_px * opts.pixel_scale;
            s.count = 6;
            s.instance_count = 8;
            s.entry.items = 8;
            specs.push_back(s);
        }
        // Entity axes, for the entities with a transform or a camera.
        if (opts.show_axes && three_d && !pixel && e != cm.through && (tree.transform(e) || pin)) {
            Spec s = base(Primitive::Lines, DrawPart::Axes);
            s.program = ProgramId::scene_lines;
            s.vs = "vs_lines";
            s.fs = "fs_color";
            s.fs_pick = "fs_pick";
            const double len = opts.axes_length > 0 ? opts.axes_length : std::max(radius, 1e-6) / 10.0;
            s.model = world * scaling({len, len, len});
            stream(s, 0, VertexFormat::F32x3, Step::Instance, geo, StaticGeometry::kAxes, 24);
            stream(s, 1, VertexFormat::F32x3, Step::Instance, geo, StaticGeometry::kAxes + 12, 24);
            stream(s, 2, VertexFormat::U32, Step::Instance, geo, StaticGeometry::kAxesColors, 8);
            stream(s, 3, VertexFormat::U32, Step::Instance, geo, StaticGeometry::kAxesColors + 4, 8);
            stream(s, 4, VertexFormat::U32, Step::Instance, geo, StaticGeometry::kKeep, 0);
            s.params.color_mode = 2;
            s.params.size_px = opts.camera_width_px * opts.pixel_scale;
            s.count = 6;
            s.instance_count = 3;
            s.entry.items = 3;
            specs.push_back(s);
        }
    }

    // Opaque, then translucent, then a looked-through camera's pixel space; path order within each.
    std::stable_sort(specs.begin(), specs.end(), [](const Spec& a, const Spec& b) {
        const int ka = (a.overlay ? 2 : 0) + (a.translucent ? 1 : 0);
        const int kb = (b.overlay ? 2 : 0) + (b.translucent ? 1 : 0);
        return ka < kb;
    });

    // Colormap blocks, one per colormap and range; draws without a colormap share the first.
    struct CmapSlot { uint32_t colormap; uint32_t vmin_bits, vmax_bits; uint32_t offset; };
    std::vector<CmapSlot> cmaps;
    auto cmap_offset = [&](uint32_t colormap, float vmin, float vmax) {
        uint32_t lo, hi;
        std::memcpy(&lo, &vmin, 4);
        std::memcpy(&hi, &vmax, 4);
        for (const CmapSlot& c : cmaps)
            if (c.colormap == colormap && c.vmin_bits == lo && c.vmax_bits == hi) return c.offset;
        sh::Cmap block{};
        res.colormap_block(colormap, vmin, vmax, &block);
        const uint32_t off = out.uniform(&block, sizeof block);
        cmaps.push_back({colormap, lo, hi, off});
        return off;
    };

    if (map) {
        map->entries.clear();
        map->revision = tree.revision();
        map->has_inverse = inverse(cm.proj * cm.view, &map->inverse_view_proj);
    }

    // The scissor: the picture's rectangle when looking through a pinhole, in the target's pixels.
    uint16_t scissor[4] = {0, 0, 0xffff, 0xffff};
    if (cm.through) {
        int32_t x0 = cm.scissor[0], y0 = cm.scissor[1], x1 = x0 + cm.scissor[2], y1 = y0 + cm.scissor[3];
        if (pick) {
            x0 = std::clamp(x0 - window[0], 0, window[2]);
            x1 = std::clamp(x1 - window[0], 0, window[2]);
            y0 = std::clamp(y0 - window[1], 0, window[3]);
            y1 = std::clamp(y1 - window[1], 0, window[3]);
        }
        scissor[0] = uint16_t(x0);
        scissor[1] = uint16_t(y0);
        scissor[2] = uint16_t(std::max(0, x1 - x0));
        scissor[3] = uint16_t(std::max(0, y1 - y0));
    }

    uint32_t pick_id = 0;
    for (Spec& s : specs) {
        RenderPipelineDesc d;
        d.vertex_entry = s.vs;
        d.fragment_entry = pick ? s.fs_pick : s.fs;
        d.vertex_layout_count = s.layout_count;
        for (uint8_t i = 0; i < s.layout_count; ++i) d.vertex_layouts[i] = s.layouts[i];
        d.topology = caliper_gfx::Topology::TriangleList;
        d.color_format = pick ? Format::RG32Uint : opts.color_format;
        d.blend = (!pick && s.translucent) ? caliper_gfx::Blend::Alpha : caliper_gfx::Blend::Opaque;
        d.depth = has_depth;
        d.depth_test = has_depth && three_d && !s.overlay;
        d.depth_write = d.depth_test && (pick || !s.translucent);
        std::string why;
        if (!check_vertex_layouts(s.program, d, &why)) {
            note(s.entry.entity, "internal: " + why);
            continue;
        }
        // Matrices: composed in double, rounded to f32 once.
        const Mat4 vm = cm.view * s.model;
        const Mat4 mvp = s.overlay ? cm.pixel_to_clip * s.model : cm.proj * vm;
        to_f32(mvp, s.params.mvp);
        normal_matrix_columns(s.overlay ? s.model : vm, s.params.nmat0, s.params.nmat1, s.params.nmat2);
        if (s.overlay) {
            const Pinhole* pin = tree.camera(cm.through);
            s.params.radius_scale = float(column_scale(s.model, 2) / pin->fx);
        } else {
            s.params.radius_scale = float(column_scale(vm, int(s.dims)));
        }
        s.params.pick_id = ++pick_id;
        s.params.item_base = 0;

        DrawRecord r;
        r.pipeline = res.render_pipeline(s.program, d);
        for (uint32_t g = 0; g < 3; ++g)
            if (declares_group(s.program, g)) r.groups[g] = res.bind_group(s.program, g, g == 2 ? s.image : Texture{}, s.image_binding);
        const uint32_t params_off = out.uniform(&s.params, sizeof s.params);
        const uint32_t cmap_off = s.uses_cmap ? cmap_offset(s.colormap, s.vmin, s.vmax) : cmap_offset(0, 0.0f, 1.0f);
        // Dynamic offsets in (group, binding) order: the frame block, the draw block, the colormap.
        const sh::ProgramDescription& pd = describe(s.program);
        uint32_t n_dyn = 0;
        for (uint32_t g = 0; g < 4; ++g)
            for (uint32_t bnd = 0; bnd < 8; ++bnd)
                for (uint8_t i = 0; i < pd.binding_count; ++i) {
                    const caliper_gfx::BindingDesc& bd = pd.bindings[i];
                    if (bd.group != g || bd.binding != bnd || !bd.dynamic || n_dyn >= caliper_gfx::kMaxDynamicOffsets) continue;
                    r.dynamic_offsets[n_dyn++] = g == 0 ? frame_off : (bnd == 0 ? params_off : cmap_off);
                }
        for (uint8_t i = 0; i < s.layout_count; ++i) r.vertex[i] = s.vertex[i];
        r.index_buffer = s.index_buffer;
        r.count = s.count;
        r.instance_count = s.instance_count;
        r.pick_id = s.params.pick_id;
        for (int i = 0; i < 4; ++i) r.scissor[i] = scissor[i];
        out.draw(r);
        ++result.draws;
        if (map) {
            s.entry.mvp = mvp;
            s.entry.translucent = s.translucent;
            s.entry.depth_test = d.depth_test;
            s.entry.depth_write = d.depth_test && !s.translucent;   // in the colour pass
            map->entries.push_back(s.entry);
        }
    }
    if (frame_off == kNoOffset) {
        result.notes.push_back("the draw list refused a uniform block");
        return result;
    }
    result.ok = out.error().empty();
    return result;
}

// ---- Labels ---------------------------------------------------------------------------------------------------------

std::vector<LabelAnchor> project_labels(const SceneTree& tree, const ViewCamera& camera, uint32_t width,
                                        uint32_t height) {
    std::vector<LabelAnchor> out;
    const CameraMatrices cm = camera_matrices(camera, tree, width, height);
    if (!cm.ok || width == 0 || height == 0) return out;
    for (uint32_t i = 0; i < tree.entity_count(); ++i) {
        const EntityId e = tree.entity_at(i);
        if (!tree.drawable(e)) continue;
        const EntityId pcam = tree.pixel_space_of(e);
        const bool overlay = pcam && pcam == cm.through;
        for (Primitive p : {Primitive::Points, Primitive::Boxes}) {
            const Batch* b = tree.batch(e, p);
            if (!b) continue;
            const BatchArray* pos = b->array(p == Primitive::Points ? Role::Positions : Role::Centers);
            const BatchArray* ids = b->array(Role::ClassIds);
            if (!pos || !pos->cpu) continue;
            if (b->labels.empty() && !(ids && ids->cpu)) continue;
            const uint64_t n = pos->count;
            const uint64_t step = n > kMaxLabelsPerBatch ? (n + kMaxLabelsPerBatch - 1) / kMaxLabelsPerBatch : 1;
            const Mat4 m = overlay ? cm.pixel_to_clip * tree.pixel_local(e) : cm.proj * cm.view * tree.world(e);
            const uint32_t comps = pos->format == caliper_gfx::VertexFormat::F32x3 ? 3 : 2;
            for (uint64_t k = 0; k < n; k += step) {
                std::string text;
                if (b->labels.size() == n) text = b->labels[k];
                else if (b->labels.size() == 1) text = b->labels[0];
                else if (ids && ids->cpu) {
                    int32_t id;
                    // One row stands for every point, whatever its stride (stride_of).
                    std::memcpy(&id, static_cast<const uint8_t*>(ids->cpu) + k * stride_of(*ids), 4);
                    text = tree.class_label(uint32_t(id));
                }
                if (text.empty()) continue;
                float f[3] = {0, 0, 0};
                std::memcpy(f, static_cast<const uint8_t*>(pos->cpu) + (pos->stride ? k * pos->stride : 0), comps * 4);
                const Vec3 local{f[0], f[1], f[2]};
                double h[4];
                transform_h(m, local, h);
                if (!(h[3] > 0)) continue;
                const double x = h[0] / h[3], y = h[1] / h[3], z = h[2] / h[3];
                if (std::fabs(x) > 1 || std::fabs(y) > 1 || z < 0 || z > 1) continue;
                LabelAnchor a;
                a.entity = e;
                a.primitive = p;
                a.item = k;
                a.x = float((x + 1) * 0.5 * width);
                a.y = float((1 - y) * 0.5 * height);
                a.depth = overlay ? 0.0f : float(length(transform_point(tree.world(e), local) - cm.eye));
                a.text = std::move(text);
                out.push_back(std::move(a));
            }
        }
    }
    std::stable_sort(out.begin(), out.end(), [](const LabelAnchor& a, const LabelAnchor& b) { return a.depth < b.depth; });
    if (out.size() > kMaxLabelsDrawn) out.resize(kMaxLabelsDrawn);
    return out;
}

}  // namespace caliper_scene
