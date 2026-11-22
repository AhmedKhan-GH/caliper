// The checks of a compiled pass of either half against the compiler's
// reflection (programs/pieces.h, design/programmable-graphics.md, Check
// Order 7). This file needs only the standard library and spec/piece.h,
// because the browser module compiles it too.
//
// The reflection of caliper-wgsl gives the entry points with their stages and
// workgroup sizes, and the bindings with their access, but not the inputs and
// outputs of an entry point. The checks of the stage interfaces (vertex
// outputs against fragment inputs, a pick pass's CaliperPicked, the count of
// inter-stage variables) therefore stay with the pipeline creation of the
// graphics layer, as Open Points of the plan states.
#include <caliper_builder/programs/pieces.h>

#include <cstdint>
#include <set>
#include <string>
#include <utility>

namespace caliper_builder::programs {

namespace {

constexpr std::uint32_t kInvocationsMax = 256;             // WebGPU's maxComputeInvocationsPerWorkgroup
constexpr std::uint32_t kWorkgroupAxisMax[3] = {256, 256, 64};

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
        return {};
    }
    return refuse(kProgramEntryStage, "The module has no " + std::string(stage_name(stage)) + " entry point " + name + ".",
                  where);
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
