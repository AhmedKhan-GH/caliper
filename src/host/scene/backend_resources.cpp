// backend_resources.cpp: Resources over a backend of the graphics layer (backend_resources.h).
#include "scene/backend_resources.h"

#include "scene/stages.h"

#include <algorithm>
#include <cstring>

namespace caliper_scene {

using namespace caliper_gfx;
namespace sh = caliper_gfx::shaders;

namespace {

void put(std::string& k, uint64_t v) {
    k.append(reinterpret_cast<const char*>(&v), sizeof v);
}

std::string pipeline_key(ProgramId p, const RenderPipelineDesc& d) {
    std::string k = "r";
    put(k, uint64_t(p));
    k += d.vertex_entry ? d.vertex_entry : "";
    k += '|';
    k += d.fragment_entry ? d.fragment_entry : "";
    k += '|';
    put(k, d.vertex_layout_count);
    for (int i = 0; i < d.vertex_layout_count; ++i) {
        const VertexLayout& l = d.vertex_layouts[i];
        put(k, (uint64_t(l.stride) << 32) | (uint64_t(l.step) << 8) | l.attribute_count);
        for (int a = 0; a < l.attribute_count; ++a)
            put(k, (uint64_t(l.attributes[a].location) << 48) | (uint64_t(l.attributes[a].format) << 32) |
                       l.attributes[a].offset);
    }
    put(k, (uint64_t(d.topology) << 40) | (uint64_t(d.color_format) << 32) | (uint64_t(d.blend) << 24) |
               (uint64_t(d.id_target) << 3) | (uint64_t(d.depth) << 2) | (uint64_t(d.depth_test) << 1) |
               uint64_t(d.depth_write));
    return k;
}

}  // namespace

BackendResources::BackendResources(Backend& backend, const ProgramDesc* programs, ColormapSource colormaps)
    : backend_(backend), programs_(programs), colormaps_(std::move(colormaps)) {}

BackendResources::~BackendResources() {
    for (auto& o : stage_groups_) backend_.release(o.second.handle);
    for (auto& o : stage_pipelines_) backend_.release(o.second.handle);
    for (auto& g : groups_) backend_.release(g.second);
    for (auto& c : compute_) backend_.release(c.second.group);
    for (auto& p : pipelines_) backend_.release(p.second);
    for (auto& o : owned_) backend_.release(o.second.buffer);
    for (auto& p : program_) backend_.release(p.second);
    if (geometry_) backend_.release(geometry_);
    if (sampler_) backend_.release(sampler_);
    if (placeholder_rgba_) backend_.release(placeholder_rgba_);
    if (placeholder_uint_) backend_.release(placeholder_uint_);
    if (class_table_) backend_.release(class_table_);
}

bool BackendResources::init(std::string* error) {
    auto refuse = [&](const char* what) {
        error_ = std::string(what) + ": " + backend_.last_error();
        if (error) *error = error_;
        return false;
    };
    BufferDesc gd;
    gd.size = StaticGeometry::kBytes;
    gd.usage = kVertex | kCopyDst;
    gd.label = "scene constant streams";
    geometry_ = backend_.create_buffer(gd);
    if (!geometry_) return refuse("the constant streams");
    uint8_t bytes[StaticGeometry::kBytes];
    StaticGeometry::fill(bytes);
    if (!backend_.write_buffer(geometry_, 0, bytes, sizeof bytes)) return refuse("the constant streams");
    sampler_ = backend_.create_sampler(SamplerDesc{true});
    if (!sampler_) return refuse("the sampler");
    TextureDesc one;
    one.width = one.height = 1;
    one.format = Format::RGBA8Unorm;
    one.usage = kSampled | kCopyDstTex;
    one.label = "scene placeholder image";
    placeholder_rgba_ = backend_.create_texture(one);
    one.format = Format::R32Uint;
    placeholder_uint_ = backend_.create_texture(one);
    if (!placeholder_rgba_ || !placeholder_uint_) return refuse("the placeholder images");
    const uint32_t white = 0xffffffffu, zero = 0;
    backend_.write_texture(placeholder_rgba_, &white, 4);
    backend_.write_texture(placeholder_uint_, &zero, 4);
    TextureDesc ct;
    ct.width = ct.height = kClassTableSide;
    ct.format = Format::RGBA8Unorm;
    ct.usage = kSampled | kCopyDstTex;
    ct.label = "scene class table";
    class_table_ = backend_.create_texture(ct);
    if (!class_table_) return refuse("the class table");
    SceneTree empty;
    class_revision_ = 0;
    update_class_table(empty);
    return true;
}

bool BackendResources::update_class_table(const SceneTree& tree) {
    if (!class_table_ || (class_revision_ != 0 && tree.classes_revision() == class_revision_)) return false;
    std::vector<uint32_t> texels(65536);
    tree.class_texels(texels.data());
    backend_.write_texture(class_table_, texels.data(), kClassTableSide * 4);
    class_revision_ = tree.classes_revision();
    return true;
}

Program BackendResources::program(ProgramId id) {
    auto it = program_.find(uint32_t(id));
    if (it != program_.end()) return it->second;
    const Program p = backend_.create_program(programs_[uint32_t(id)]);
    if (!p) error_ = backend_.last_error();
    program_[uint32_t(id)] = p;
    return p;
}

Pipeline BackendResources::render_pipeline(ProgramId id, const RenderPipelineDesc& desc) {
    const std::string k = pipeline_key(id, desc);
    auto it = pipelines_.find(k);
    if (it != pipelines_.end()) return it->second;
    RenderPipelineDesc d = desc;
    d.program = program(id);
    const Pipeline p = backend_.create_render_pipeline(d);
    if (!p) error_ = backend_.last_error();
    pipelines_[k] = p;
    return p;
}

Pipeline BackendResources::compute_pipeline(ProgramId id, const char* entry) {
    std::string k = "c";
    put(k, uint64_t(id));
    k += entry;
    auto it = pipelines_.find(k);
    if (it != pipelines_.end()) return it->second;
    ComputePipelineDesc d;
    d.program = program(id);
    d.entry = entry;
    const Pipeline p = backend_.create_compute_pipeline(d);
    if (!p) error_ = backend_.last_error();
    pipelines_[k] = p;
    return p;
}

BindGroup BackendResources::bind_group(ProgramId id, uint32_t group, Texture image, uint32_t image_binding) {
    std::string k = "g";
    put(k, (uint64_t(id) << 32) | (uint64_t(group) << 16) | image_binding);
    put(k, group == 2 ? image.id : 0);
    auto it = groups_.find(k);
    if (it != groups_.end()) return it->second;
    const BindGroup g = make_group(program(id), id, group, image, image_binding);
    groups_[k] = g;
    if (group == 2 && image) group_images_[k] = image;
    return g;
}

Pipeline BackendResources::stage_pipeline(const StageProgram& stage, ProgramId builtin, const RenderPipelineDesc& desc) {
    if (!stage.program) return {};
    std::string k = pipeline_key(builtin, desc);
    put(k, stage.program.id);
    auto it = stage_pipelines_.find(k);
    if (it != stage_pipelines_.end()) return Pipeline{it->second.handle};
    RenderPipelineDesc d = desc;
    d.program = stage.program;
    const Pipeline p = backend_.create_render_pipeline(d);
    if (!p) {
        error_ = backend_.last_error();
        return {};
    }
    stage_pipelines_[k] = {stage.program, p.id, {}};
    return p;
}

BindGroup BackendResources::stage_bind_group(const StageProgram& stage, ProgramId builtin, uint32_t group,
                                             Texture image, uint32_t image_binding) {
    if (!stage.program) return {};
    std::string k = "s";
    put(k, stage.program.id);
    put(k, (uint64_t(group) << 16) | image_binding);
    put(k, group == 2 ? image.id : 0);
    auto it = stage_groups_.find(k);
    if (it != stage_groups_.end()) return BindGroup{it->second.handle};
    const BindGroup g = make_group(stage.program, builtin, group, image, image_binding);
    if (!g) return {};
    stage_groups_[k] = {stage.program, g.id, group == 2 ? image : Texture{}};
    return g;
}

BindGroup BackendResources::stage_params_group(const StageProgram& stage) {
    if (!stage.program) return {};
    std::string k = "p";
    put(k, stage.program.id);
    auto it = stage_groups_.find(k);
    if (it != stage_groups_.end()) return BindGroup{it->second.handle};
    BindGroupDesc d;
    d.program = stage.program;
    d.group = 3;   // caliper_builder::programs::kStageParamsGroup
    BindEntry e;
    e.binding = 0;
    e.buffer = kArena;
    d.entries[d.entry_count++] = e;
    const BindGroup g = backend_.create_bind_group(d);
    if (!g) {
        error_ = backend_.last_error();
        return {};
    }
    stage_groups_[k] = {stage.program, g.id, Texture{}};
    return g;
}

BindGroup BackendResources::make_group(Program prog, ProgramId id, uint32_t group, Texture image, uint32_t image_binding) {
    BindGroupDesc d;
    d.program = prog;
    d.group = group;
    const sh::ProgramDescription& pd = sh::kProgramDescriptions[uint32_t(id)];
    for (uint8_t i = 0; i < pd.binding_count && d.entry_count < kMaxBindingsPerGroup; ++i) {
        const BindingDesc& b = pd.bindings[i];
        if (b.group != group) continue;
        BindEntry e;
        e.binding = b.binding;
        switch (b.type) {
            case BindingType::Uniform: e.buffer = kArena; break;
            case BindingType::Texture:
                e.texture = group == 0 ? class_table_ : (b.binding == image_binding && image ? image : placeholder_rgba_);
                break;
            case BindingType::UintTexture:
                e.texture = (b.binding == image_binding && image) ? image : placeholder_uint_;
                break;
            case BindingType::Sampler: e.sampler = sampler_; break;
            default: continue;   // storage bindings belong to compute groups
        }
        d.entries[d.entry_count++] = e;
    }
    const BindGroup g = backend_.create_bind_group(d);
    if (!g) error_ = backend_.last_error();
    return g;
}

BindGroup BackendResources::compute_group(ProgramId id, View<BindEntry> entries) {
    std::string k = "k";
    put(k, uint64_t(id));
    for (const BindEntry& e : entries) {
        put(k, e.binding);
        put(k, e.buffer.id);
        put(k, e.offset);
        put(k, e.size);
    }
    auto it = compute_.find(k);
    if (it != compute_.end()) return it->second.group;
    BindGroupDesc d;
    d.program = program(id);
    d.group = 0;
    Compute c;
    for (const BindEntry& e : entries) {
        if (d.entry_count < kMaxBindingsPerGroup) d.entries[d.entry_count++] = e;
        if (e.buffer && e.buffer != kArena) c.buffers.push_back(e.buffer);
    }
    c.group = backend_.create_bind_group(d);
    if (!c.group) error_ = backend_.last_error();
    compute_[k] = c;
    return c.group;
}

Buffer BackendResources::owned_buffer(uint64_t key, uint64_t bytes, uint32_t usage) {
    const uint64_t size = std::max<uint64_t>(16, (bytes + 3) / 4 * 4);
    auto it = owned_.find(key);
    if (it != owned_.end()) {
        if (it->second.bytes >= size && it->second.usage == usage) return it->second.buffer;
        backend_.release(it->second.buffer);   // deferred by the layer until its last use completed
        owned_.erase(it);
    }
    BufferDesc d;
    d.size = size;
    d.usage = usage | kCopyDst;
    d.label = "scene owned stream";
    const Buffer b = backend_.create_buffer(d);
    if (!b) error_ = backend_.last_error();
    owned_[key] = {b, size, usage};
    return b;
}

void BackendResources::retain_owned(View<uint64_t> keys) {
    // A stage program the view released, or an image a stage group bound that the view released:
    // its pipelines and groups go back to the layer.
    for (auto* objects : {&stage_pipelines_, &stage_groups_})
        for (auto it = objects->begin(); it != objects->end();) {
            if (backend_.alive(it->second.program) && (!it->second.image || backend_.alive(it->second.image))) {
                ++it;
                continue;
            }
            backend_.release(it->second.handle);
            it = objects->erase(it);
        }
    std::vector<uint64_t> live(keys.begin(), keys.end());
    std::sort(live.begin(), live.end());
    for (auto it = owned_.begin(); it != owned_.end();) {
        if (std::binary_search(live.begin(), live.end(), it->first)) {
            ++it;
            continue;
        }
        backend_.release(it->second.buffer);   // deferred by the layer until its last use completed
        it = owned_.erase(it);
    }
    for (auto it = group_images_.begin(); it != group_images_.end();) {
        if (backend_.alive(it->second)) {
            ++it;
            continue;
        }
        const auto g = groups_.find(it->first);
        if (g != groups_.end()) {
            if (g->second) backend_.release(g->second);
            groups_.erase(g);
        }
        it = group_images_.erase(it);
    }
    for (auto it = compute_.begin(); it != compute_.end();) {
        const bool gone = std::any_of(it->second.buffers.begin(), it->second.buffers.end(),
                                      [&](Buffer b) { return !backend_.alive(b); });
        if (!gone) {
            ++it;
            continue;
        }
        if (it->second.group) backend_.release(it->second.group);
        it = compute_.erase(it);
    }
}

void BackendResources::colormap_block(uint32_t colormap, float vmin, float vmax, sh::Cmap* out) {
    if (colormaps_) {
        colormaps_(colormap, vmin, vmax, out);
        return;
    }
    std::memset(out, 0, sizeof *out);
    out->vmin = vmin;
    out->vmax = vmax;
}

}  // namespace caliper_scene
