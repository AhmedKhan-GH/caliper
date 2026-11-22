// generated_pipelines.h: a render pipeline descriptor for each pair of entries that the shader
// generator checks (gfx/shaders/generated/programs.h), built from the program's description: the
// vertex entry's inputs in per-vertex layouts of two, and the colour format of the fragment
// entry's output at location 0. The layer's headless tests and its Metal rows share it.
#pragma once

#include "gfx/gfx.h"
#include "gfx/shaders/generated/programs.h"

#include <cstring>

namespace caliper_test {

inline caliper_gfx::RenderPipelineDesc generated_pipeline(const caliper_gfx::shaders::ProgramDescription& pd,
                                                          caliper_gfx::Program program,
                                                          const caliper_gfx::shaders::PairDesc& pair) {
    using namespace caliper_gfx;
    RenderPipelineDesc d;
    d.program = program;
    d.vertex_entry = pair.vertex;
    d.fragment_entry = pair.fragment;
    // Two inputs a buffer, packed, so that a mesh's nine inputs fit the eight vertex buffers.
    uint32_t used = 0;
    for (uint32_t i = 0; i < pd.vertex_input_count; ++i) {
        const shaders::VertexInputDesc& in = pd.vertex_inputs[i];
        if (std::strcmp(in.entry, pair.vertex) != 0) continue;
        if (used % 2 == 0) d.vertex_layout_count++;
        VertexLayout& l = d.vertex_layouts[d.vertex_layout_count - 1];
        l.attributes[l.attribute_count++] = {in.location, in.format, l.stride};
        l.stride += (vertex_format_bytes(in.format) + 3) / 4 * 4;
        ++used;
    }
    for (uint32_t i = 0; i < pd.fragment_output_count; ++i) {
        const shaders::FragmentOutputDesc& out = pd.fragment_outputs[i];
        if (std::strcmp(out.entry, pair.fragment) != 0) continue;
        if (out.location == 0) d.color_format = out.format;
        if (out.location == 1 && out.format == Format::RG32Uint) d.id_target = true;
    }
    return d;
}

}  // namespace caliper_test
