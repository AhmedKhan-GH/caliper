// fake_resources.h: a Resources for the headless tests. It hands out plain handle values and
// records every request, so that a test reads a compiled draw list as data: which program and
// descriptor each pipeline has, which image each group binds, which buffers the scene core owns.
#pragma once

#include "scene/compile.h"
#include "scene/resources.h"

#include <cstring>
#include <map>
#include <string>
#include <tuple>
#include <vector>

namespace scene_fixtures {

using namespace caliper_gfx;
using caliper_scene::ProgramId;

struct PipelineRecord {
    ProgramId program;
    RenderPipelineDesc desc;
    std::string vs, fs;   // copies of the entry names
};

struct GroupRecord {
    ProgramId program;
    uint32_t group;
    Texture image;
    uint32_t binding;
};

inline bool same_desc(const PipelineRecord& r, ProgramId p, const RenderPipelineDesc& d) {
    if (r.program != p || r.vs != d.vertex_entry || r.fs != d.fragment_entry) return false;
    const RenderPipelineDesc& a = r.desc;
    if (a.vertex_layout_count != d.vertex_layout_count || a.topology != d.topology ||
        a.color_format != d.color_format || a.blend != d.blend || a.depth != d.depth ||
        a.depth_test != d.depth_test || a.depth_write != d.depth_write || a.id_target != d.id_target)
        return false;
    for (int i = 0; i < a.vertex_layout_count; ++i) {
        const VertexLayout &x = a.vertex_layouts[i], &y = d.vertex_layouts[i];
        if (x.stride != y.stride || x.step != y.step || x.attribute_count != y.attribute_count) return false;
        for (int k = 0; k < x.attribute_count; ++k)
            if (x.attributes[k].location != y.attributes[k].location ||
                x.attributes[k].format != y.attributes[k].format || x.attributes[k].offset != y.attributes[k].offset)
                return false;
    }
    return true;
}

class FakeResources : public caliper_scene::Resources {
public:
    static constexpr uint64_t kGeometry = 0x7000;

    Pipeline render_pipeline(ProgramId program, const RenderPipelineDesc& desc) override {
        for (size_t i = 0; i < pipelines.size(); ++i)
            if (same_desc(pipelines[i], program, desc)) return Pipeline{0x100 + i};
        PipelineRecord r{program, desc, desc.vertex_entry, desc.fragment_entry};
        r.desc.vertex_entry = nullptr;
        r.desc.fragment_entry = nullptr;
        pipelines.push_back(r);
        return Pipeline{0x100 + pipelines.size() - 1};
    }
    Pipeline compute_pipeline(ProgramId program, const char* entry) override {
        computes.push_back({program, entry});
        return Pipeline{0x900 + uint64_t(program)};
    }
    BindGroup bind_group(ProgramId program, uint32_t group, Texture image, uint32_t binding) override {
        groups.push_back({program, group, image, binding});
        return BindGroup{0x200 + groups.size() - 1};
    }
    BindGroup compute_group(ProgramId program, View<BindEntry> entries) override {
        compute_groups.push_back({program, std::vector<BindEntry>(entries.begin(), entries.end())});
        return BindGroup{0x500 + compute_groups.size() - 1};
    }
    Buffer static_geometry() override { return Buffer{kGeometry}; }
    Buffer owned_buffer(uint64_t key, uint64_t bytes, uint32_t usage) override {
        auto it = owned.find(key);
        if (it == owned.end() || std::get<1>(it->second) < bytes) {
            const uint64_t id = 0x8000 + owned.size();
            owned[key] = std::make_tuple(id, bytes, usage);
            return Buffer{id};
        }
        return Buffer{std::get<0>(it->second)};
    }
    void colormap_block(uint32_t colormap, float vmin, float vmax, shaders::Cmap* out) override {
        std::memset(out, 0, sizeof *out);
        out->vmin = vmin;
        out->vmax = vmax;
        out->levels[0][1] = 0xC0000000u | colormap;   // recognisable in the arena
        ++cmap_requests;
    }

    const PipelineRecord& pipeline(Pipeline p) const { return pipelines.at(p.id - 0x100); }
    const GroupRecord& group(BindGroup g) const { return groups.at(g.id - 0x200); }

    std::vector<PipelineRecord> pipelines;
    std::vector<std::pair<ProgramId, std::string>> computes;
    std::vector<GroupRecord> groups;
    std::vector<std::pair<ProgramId, std::vector<BindEntry>>> compute_groups;
    std::map<uint64_t, std::tuple<uint64_t, uint64_t, uint32_t>> owned;
    int cmap_requests = 0;
};

// A record's parameter block, read back from the list's arena.
inline shaders::PrimParams params_of(const DrawList& list, const DrawRecord& r) {
    shaders::PrimParams p;
    std::memcpy(&p, list.arena().data + r.dynamic_offsets[1], sizeof p);
    return p;
}
inline shaders::Frame frame_of(const DrawList& list, const DrawRecord& r) {
    shaders::Frame f;
    std::memcpy(&f, list.arena().data + r.dynamic_offsets[0], sizeof f);
    return f;
}
inline shaders::Cmap cmap_of(const DrawList& list, const DrawRecord& r) {
    shaders::Cmap c;
    std::memcpy(&c, list.arena().data + r.dynamic_offsets[2], sizeof c);
    return c;
}

}  // namespace scene_fixtures
