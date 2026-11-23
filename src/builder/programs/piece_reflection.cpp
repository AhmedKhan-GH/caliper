// The checks of a compiled pass of either half against the compiler's
// reflection (programs/pieces.h, design/programmable-graphics.md, Check
// Order 7). This file needs only the standard library and spec/piece.h,
// because the browser module compiles it too.
//
// The reflection of caliper-wgsl gives the entry points with their stages,
// workgroup sizes and user-defined inputs and outputs, and the bindings with
// their access. A render pass's stage interfaces are checked here, before any
// pipeline exists: the vertex entry reads no vertex attribute, because a piece
// has no vertex buffers; it passes at most 16 inter-stage variables, as
// WebGPU's default limit allows; each fragment input is a vertex output of
// the same location, type and interpolation; and the fragment writes exactly
// the attachments of its target, a vec4<f32> color at location 0 and, in a
// pass with pick, the vec2<u32> ID of CaliperPicked at location 1.
#include <caliper_builder/programs/pieces.h>

#include <cstdint>
#include <set>
#include <string>
#include <utility>
#include <vector>

namespace caliper_builder::programs {

namespace {

constexpr std::uint32_t kInvocationsMax = 256;             // WebGPU's maxComputeInvocationsPerWorkgroup
constexpr std::uint32_t kWorkgroupAxisMax[3] = {256, 256, 64};
constexpr std::uint32_t kInterStageMax = 16;               // WebGPU's maxInterStageShaderVariables
constexpr std::uint32_t kStageStorageMax = 8;              // WebGPU's maxStorageBuffersPerShaderStage

Verdict refuse(const char* code, std::string message, std::string where) {
    Verdict v;
    v.code = code;
    v.message = std::move(message);
    v.where = std::move(where);
    return v;
}

const char* stage_name(EntryStage s) {
    return s == EntryStage::vertex ? "vertex" : s == EntryStage::fragment ? "fragment" : "compute";
}

// The entry point `name` with stage `stage`, else a refusal at `where`.
Verdict check_entry(const Reflection& r, const std::string& name, EntryStage stage, const std::string& where) {
    for (const ReflectedEntry& e : r.entries) {
        if (e.name != name) continue;
        if (e.stage != stage)
            return refuse(kProgramEntryStage,
                          "The entry point " + name + " is a " + stage_name(e.stage) + " entry, not a " +
                              stage_name(stage) + " entry.",
                          where);
        if (stage != EntryStage::compute) return {};
        std::uint64_t invocations = 1;
        for (int axis = 0; axis < 3; ++axis) {
            const std::uint32_t n = e.workgroup_size[axis];
            if (n == 0 || n > kWorkgroupAxisMax[axis])
                return refuse(kProgramOverWebLimits,
                              "The workgroup size of " + name + " is " + std::to_string(n) + " on axis " +
                                  std::to_string(axis) + ", outside 1 to " + std::to_string(kWorkgroupAxisMax[axis]) +
                                  ".",
                              where);
            invocations *= n;   // at most 256 * 256 * 64, no wrap in 64 bits
        }
        if (invocations > kInvocationsMax)
            return refuse(kProgramOverWebLimits,
                          "A workgroup of " + name + " has " + std::to_string(invocations) +
                              " invocations, above WebGPU's 256.",
                          where);
        // The Metal hosts give a compute pass no threadgroup memory, thus a var<workgroup> would
        // read zero there and its values in a browser: the piece is refused in every host.
        if (e.workgroup_memory > 0)
            return refuse(kProgramWorkgroupMemory,
                          "The entry point " + name + " uses " + std::to_string(e.workgroup_memory) +
                              " bytes of var<workgroup>, which a pass of a piece cannot have; keep partial results "
                              "in a private resource or a draw buffer.",
                          where);
        return {};
    }
    return refuse(kProgramEntryStage, "The module has no " + std::string(stage_name(stage)) + " entry point " + name + ".",
                  where);
}

const ReflectedEntry* find_entry(const Reflection& r, const std::string& name, EntryStage stage) {
    for (const ReflectedEntry& e : r.entries)
        if (e.name == name && e.stage == stage) return &e;
    return nullptr;
}

const ReflectedIo* at_location(const std::vector<ReflectedIo>& values, std::uint32_t location) {
    for (const ReflectedIo& v : values)
        if (v.location == location) return &v;
    return nullptr;
}

// The WGSL spelling of an interface value's type, such as vec4<f32> or u32.
std::string type_name(const ReflectedIo& v) {
    const char* scalar = v.scalar == IoScalar::f32   ? "f32"
                         : v.scalar == IoScalar::i32 ? "i32"
                         : v.scalar == IoScalar::u32 ? "u32"
                         : v.scalar == IoScalar::f16 ? "f16"
                                                     : "another type";
    if (v.scalar == IoScalar::other || v.components <= 1) return scalar;
    return "vec" + std::to_string(v.components) + "<" + scalar + ">";
}

bool same_type(const ReflectedIo& a, const ReflectedIo& b) {
    return a.scalar == b.scalar && a.components == b.components && a.scalar != IoScalar::other;
}

// Check Order 7 for a render pass: the vertex entry's inputs and outputs, the fragment entry's
// inputs against them, and the fragment entry's outputs against the pass's target and pick.
Verdict check_interfaces(const Reflection& r, const spec::DrawPass& p, const std::string& vertex,
                         const std::string& where) {
    const ReflectedEntry* vs = find_entry(r, vertex, EntryStage::vertex);
    const ReflectedEntry* fs = find_entry(r, p.fragment, EntryStage::fragment);
    if (!vs || !fs) return {};   // check_entry() refused a missing entry already
    if (!vs->inputs.empty())
        return refuse(kProgramInterfaceMismatch,
                      "The vertex entry " + vertex + " reads location " + std::to_string(vs->inputs.front().location) +
                          " as a vertex attribute, and a piece has no vertex buffers: a vertex entry reads "
                          "vertex_index, instance_index and the piece's buffers.",
                      where);
    if (vs->outputs.size() > kInterStageMax)
        return refuse(kProgramOverWebLimits,
                      "The vertex entry " + vertex + " passes " + std::to_string(vs->outputs.size()) +
                          " inter-stage variables, above WebGPU's 16.",
                      where);
    for (const ReflectedIo& out : vs->outputs)
        if (out.location >= kInterStageMax)
            return refuse(kProgramOverWebLimits,
                          "The vertex entry " + vertex + " writes location " + std::to_string(out.location) +
                              ", and WebGPU allows the locations 0 to 15.",
                          where);
    for (const ReflectedIo& in : fs->inputs) {
        const std::string loc = std::to_string(in.location);
        const ReflectedIo* out = at_location(vs->outputs, in.location);
        if (!out)
            return refuse(kProgramInterfaceMismatch,
                          "The fragment entry " + p.fragment + " reads location " + loc + ", which the vertex entry " +
                              vertex + " does not write.",
                          where);
        if (!same_type(*out, in))
            return refuse(kProgramInterfaceMismatch,
                          "The fragment entry " + p.fragment + " reads location " + loc + " as " + type_name(in) +
                              ", and the vertex entry " + vertex + " writes it as " + type_name(*out) + ".",
                          where);
        if (out->interpolation != in.interpolation || out->sampling != in.sampling)
            return refuse(kProgramInterfaceMismatch,
                          "The fragment entry " + p.fragment + " reads location " + loc +
                              " with another interpolation than the vertex entry " + vertex + " writes it with.",
                          where);
    }
    // The attachments of the pass: the color at location 0, and the ID at location 1 with pick.
    ReflectedIo color;
    color.location = 0;
    ReflectedIo id;
    id.location = 1;
    id.scalar = IoScalar::u32;
    id.components = 2;
    std::vector<ReflectedIo> wanted = {color};
    if (p.pick) wanted.push_back(id);
    const std::string target = p.target == "host" ? std::string("the host target") : "the target " + p.target;
    for (const ReflectedIo& w : wanted) {
        const std::string what = w.location == 0 ? "color" : "pick ID";
        const ReflectedIo* out = at_location(fs->outputs, w.location);
        if (!out)
            return refuse(kProgramInterfaceMismatch,
                          "The fragment entry " + p.fragment + " writes no " + what + " at location " +
                              std::to_string(w.location) + ", which " + target +
                              (w.location == 1 ? " of a pass with pick needs: return CaliperPicked." : " needs."),
                          where);
        if (!same_type(*out, w))
            return refuse(kProgramInterfaceMismatch,
                          "The fragment entry " + p.fragment + " writes its " + what + " at location " +
                              std::to_string(w.location) + " as " + type_name(*out) + ", and " + target +
                              " takes " + type_name(w) + ".",
                          where);
    }
    for (const ReflectedIo& out : fs->outputs)
        if (!at_location(wanted, out.location))
            return refuse(kProgramInterfaceMismatch,
                          "The fragment entry " + p.fragment + " writes location " + std::to_string(out.location) +
                              ", and " + target + (p.pick || out.location != 1 ? " has no attachment there."
                                                                              : " of a pass without pick takes no ID."),
                          where);
    return {};
}

}  // namespace

Verdict check_piece_reflection(const spec::Piece& piece, Half half, std::size_t pass, const Reflection& r) {
    const bool draw = half == Half::draw;
    const std::string where = std::string(draw ? "draw" : "derive") + ".passes[" + std::to_string(pass) + "]";
    for (const ReflectedEntry& e : r.entries)
        if (e.name == "main")
            return refuse(kProgramEntryStage, "An entry point may not be named main.", where);

    if (!draw) {
        if (!piece.derive || pass >= piece.derive->passes.size())
            return refuse(kProgramBadPass, "The derive half has no pass " + std::to_string(pass) + ".", where);
        const spec::DerivePass& p = piece.derive->passes[pass];
        if (Verdict v = check_entry(r, p.entry, EntryStage::compute, where); !v.ok()) return v;
        // r is the reflection of the module that keeps only this pass's entry point, which is
        // what the producer binds for the pass. One pass binds at most WebGPU's 8.
        std::uint32_t storage = 0;
        for (const ReflectedBinding& b : r.bindings)
            if (b.kind == BindingKind::storage_buffer) ++storage;
        if (storage > kStageStorageMax)
            return refuse(kProgramOverWebLimits,
                          "The entry point " + p.entry + " binds " + std::to_string(storage) +
                              " storage buffers: its tensor inputs, resources and counters. One derive pass binds at "
                              "most 8; give the work to two passes.",
                          where);
        // The tensor inputs of group 1 are read-only: no program writes a model tensor.
        for (const ReflectedBinding& b : r.bindings)
            if (b.group == kGroupInputs && b.write)
                return refuse(kProgramBindingNotAllowed, "The module writes the tensor input " + b.name + ".", where);
        return {};
    }

    if (!piece.draw || pass >= piece.draw->passes.size())
        return refuse(kProgramBadPass, "The draw half has no pass " + std::to_string(pass) + ".", where);
    const spec::Draw& d = *piece.draw;
    const spec::DrawPass& p = d.passes[pass];
    if (p.kind == spec::PassKind::compute) {
        if (Verdict v = check_entry(r, p.entry, EntryStage::compute, where); !v.ok()) return v;
    } else {
        const std::string vertex = p.vertex ? *p.vertex : std::string("caliper_fullscreen");
        if (Verdict v = check_entry(r, vertex, EntryStage::vertex, where); !v.ok()) return v;
        if (Verdict v = check_entry(r, p.fragment, EntryStage::fragment, where); !v.ok()) return v;
        if (Verdict v = check_interfaces(r, p, vertex, where); !v.ok()) return v;
    }
    // What the pass may write: only the draw buffers that a compute pass names
    // in `writes`. The prelude, the params, the counts, the resources and the
    // buffer inputs are read-only.
    const std::set<std::string> writes(p.writes.begin(), p.writes.end());
    for (const ReflectedBinding& b : r.bindings) {
        if (b.group > kGroupResources)
            return refuse(kProgramBindingNotAllowed,
                          "The module binds " + b.name + " in group " + std::to_string(b.group) + ".", where);
        if (!b.write) continue;
        bool allowed = false;
        if (b.group == kGroupResources && p.kind == spec::PassKind::compute && b.binding < d.buffers.size())
            allowed = writes.count(d.buffers[b.binding].id) != 0;
        if (!allowed)
            return refuse(kProgramBindingNotAllowed, "The pass writes " + b.name + ", which it may only read.", where);
    }
    return {};
}

}  // namespace caliper_builder::programs
