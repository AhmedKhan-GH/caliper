// backend_resources.h: the scene core's Resources over a backend of the graphics layer.
//
// A view that draws a scene owns one BackendResources per backend. It creates each generated
// program on first use from the backend's table of ProgramDescs (kMetalPrograms, kWebgpuPrograms
// or kVulkanPrograms), keeps one reference to each pipeline and bind group it was handed, uploads
// the scene core's constant streams once, keeps the class table texture in step with the scene,
// and owns the buffers of clamped indices and gathered edges. Colormap blocks come from the
// caller, normally colormap_source over a KeyBlockCache of src/host/gfx/colormap_keys.h.
#pragma once

#include "gfx/gfx.h"
#include "scene/resources.h"
#include "scene/scene.h"

#include <cstdint>
#include <functional>
#include <map>
#include <string>
#include <vector>

namespace caliper_scene {

class BackendResources final : public Resources {
public:
    using ColormapSource = std::function<void(uint32_t colormap, float vmin, float vmax, caliper_gfx::shaders::Cmap*)>;

    // programs: the backend's table, indexed by ProgramId. colormaps may be empty, which leaves
    // every block's levels zero (enough for draws that take no colormap).
    BackendResources(caliper_gfx::Backend& backend, const caliper_gfx::ProgramDesc* programs, ColormapSource colormaps = {});
    ~BackendResources() override;
    BackendResources(const BackendResources&) = delete;
    BackendResources& operator=(const BackendResources&) = delete;

    // Creates the constant streams, the sampler, the placeholder images and the class table.
    bool init(std::string* error);
    // Uploads the scene's class table when it changed since the last call; true when it did
    // (65,536 texels, 256 KiB, one copy from the host).
    bool update_class_table(const SceneTree& tree);
    const std::string& error() const { return error_; }

    caliper_gfx::Pipeline render_pipeline(ProgramId program, const caliper_gfx::RenderPipelineDesc& desc) override;
    caliper_gfx::Pipeline compute_pipeline(ProgramId program, const char* entry) override;
    caliper_gfx::BindGroup bind_group(ProgramId program, uint32_t group, caliper_gfx::Texture image,
                                      uint32_t image_binding) override;
    caliper_gfx::BindGroup compute_group(ProgramId program, caliper_gfx::View<caliper_gfx::BindEntry> entries) override;
    caliper_gfx::Buffer static_geometry() override { return geometry_; }
    caliper_gfx::Buffer owned_buffer(uint64_t key, uint64_t bytes, uint32_t usage) override;
    void retain_owned(caliper_gfx::View<uint64_t> keys) override;
    void colormap_block(uint32_t colormap, float vmin, float vmax, caliper_gfx::shaders::Cmap* out) override;

private:
    caliper_gfx::Program program(ProgramId id);

    caliper_gfx::Backend& backend_;
    const caliper_gfx::ProgramDesc* programs_;
    ColormapSource colormaps_;
    std::string error_;
    std::map<uint32_t, caliper_gfx::Program> program_;
    std::map<std::string, caliper_gfx::Pipeline> pipelines_;
    std::map<std::string, caliper_gfx::BindGroup> groups_;
    // The image of each group 2 by its key: a group whose image the view released goes back to
    // the layer at the next retain_owned, so that images replaced at a new size leave nothing.
    std::map<std::string, caliper_gfx::Texture> group_images_;
    struct Compute { caliper_gfx::BindGroup group; std::vector<caliper_gfx::Buffer> buffers; };
    std::map<std::string, Compute> compute_;   // compute groups, with the buffers they refer to
    struct Owned { caliper_gfx::Buffer buffer; uint64_t bytes; uint32_t usage; };
    std::map<uint64_t, Owned> owned_;
    caliper_gfx::Buffer geometry_;
    caliper_gfx::Sampler sampler_;
    caliper_gfx::Texture placeholder_rgba_, placeholder_uint_, class_table_;
    uint64_t class_revision_ = 0;
};

}  // namespace caliper_scene
