// webgpu_backend.h: the WebGPU backend of the graphics layer (src/host/gfx/gfx.h), on the
// standard webgpu.h. In the browser it runs on Emscripten's emdawnwebgpu port; the same source is
// meant to compile against Dawn natively later (design/renderer-core.md, step 9).
//
// It borrows the device and queue of the renderer that made it (WebGpuRenderer keeps them and
// draws ImGui on them), so the layer never creates a second device. webgpu_backend.cpp opens with
// the mapping of each hook onto webgpu.h.
#pragma once

#include "gfx/gfx.h"

#include <webgpu/webgpu.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace caliper_gfx {

class WebGpuBackend final : public Backend {
public:
    // The device and queue stay the caller's; the backend adds a reference to each and releases it
    // when it is destroyed. adapter is the text Caps::adapter reports. Limits are read from the
    // device and never exceed WebGPU's defaults; on a compatibility-mode device they are the
    // compatibility limits.
    static std::unique_ptr<WebGpuBackend> create(WGPUDevice device, WGPUQueue queue, const char* adapter);
    ~WebGpuBackend() override;

    // The texture views of textures destroyed since the last call. A destroyed texture's view is
    // kept to the end of the frame, because imgui_impl_wgpu keys its per-frame bind groups by the
    // view's address (the renderer calls this after it recorded and submitted ImGui's frame).
    void release_retired();
    size_t retired_view_count() const { return retired_views_.size(); }

    // What the backend holds, for the module's counters: live layer buffers, and released buffers
    // kept for reuse (at most kSparePoolBytes and kSparePoolCount of them).
    size_t live_buffer_count() const;
    size_t spare_buffer_count() const { return spare_.size(); }
    uint64_t spare_bytes() const { return spare_bytes_; }
    uint64_t arena_capacity() const { return arena_cap_; }

    // WebGPU reports the validation of a shader module or pipeline asynchronously, so a hook
    // cannot refuse it at once. Each such error lands here when the browser reports it.
    const std::vector<std::string>& device_errors() const { return *errors_; }
    bool compatibility() const { return compat_; }

    static constexpr uint64_t kSparePoolBytes = 64ull << 20;   // web-geometry.md, "Uploads and reuse"
    static constexpr size_t kSparePoolCount = 16;
    static constexpr uint64_t kMinCapacity = 4096;             // buffers are rounded up to a power of two

protected:
    bool on_create_buffer(uint32_t slot, const BufferDesc& d, std::string* error) override;
    bool on_create_texture(uint32_t slot, const TextureDesc& d, std::string* error) override;
    bool on_create_sampler(uint32_t slot, const SamplerDesc& d, std::string* error) override;
    bool on_create_program(uint32_t slot, const ProgramDesc& d, std::string* error) override;
    bool on_create_render_pipeline(uint32_t slot, const RenderPipelineDesc& d, std::string* error) override;
    bool on_create_compute_pipeline(uint32_t slot, const ComputePipelineDesc& d, std::string* error) override;
    bool on_create_bind_group(uint32_t slot, const BindGroupDesc& d, std::string* error) override;
    void on_destroy(uint64_t handle) override;
    void on_write_buffer(uint32_t slot, uint64_t offset, const void* data, uint64_t size, bool in_use) override;
    void on_write_texture(uint32_t slot, const void* data, uint32_t row_pitch, bool in_use) override;
    void* on_mapped(uint32_t slot) override;
    bool on_import_buffer(uint32_t slot, const ExternalBuffer& ext, ImportResult* r) override;
    bool on_import_texture(uint32_t slot, const ExternalTexture& ext, ImportResult* r) override;
    bool on_submit(Serial s, const Frame& frame, Stats* stats, std::string* error) override;
    bool on_wait(Serial s) override;
    bool on_read_texture(Serial s, uint32_t readback_slot, uint32_t texture_slot, uint32_t x, uint32_t y,
                         uint32_t w, uint32_t h, std::string* error) override;
    void on_take_readback(uint32_t readback_slot, std::vector<uint8_t>* out) override;
    uint64_t on_imgui_texture_id(uint32_t slot) override;

private:
    WebGpuBackend(WGPUDevice device, WGPUQueue queue, const Caps& caps, bool compat);

    struct Buf {
        WGPUBuffer buffer = nullptr;
        uint64_t size = 0;        // what the layer asked for
        uint64_t capacity = 0;    // what the GPU buffer holds
        WGPUBufferUsage usage = 0;
    };
    struct Tex {
        WGPUTexture texture = nullptr;
        WGPUTextureView view = nullptr;
        uint32_t w = 0, h = 0;
        Format format = Format::RGBA8Unorm;
    };
    struct Prog {
        WGPUShaderModule module = nullptr;
        WGPUBindGroupLayout groups[kMaxBindGroups] = {};
        WGPUPipelineLayout layout = nullptr;
        std::vector<BindingDesc> bindings;   // sorted by (group, binding)
        uint8_t group_mask = 0;              // the groups the program declares
        std::vector<EntryDesc> entries;      // names point into the caller's ProgramDesc table
    };
    struct Pipe {
        WGPURenderPipeline render = nullptr;
        WGPUComputePipeline compute = nullptr;
        // The program's groups, copied at creation: the program may be released while its
        // pipelines live, and its slot taken by another program.
        uint8_t group_mask = 0;
        uint8_t vertex_layout_count = 0;
    };
    struct Group {
        WGPUBindGroup group = nullptr;
        WGPUBindGroupLayout layout = nullptr;   // a reference of its own
        uint32_t program_slot = 0;
        uint32_t index = 0;
        // Its dynamic bindings in binding order, and which of them bind the arena: the draw's
        // dynamic offsets of those are rebased onto the pass's slice of the arena buffer.
        uint8_t dynamic_count = 0;
        bool dynamic_arena[kMaxBindingsPerGroup] = {};
        bool uses_arena = false;
        BindGroupDesc desc;       // kept to rebuild it when the arena buffer grows
    };
    struct Pending {              // a readback: its staging buffer, then its tight bytes
        WGPUBuffer staging = nullptr;
        uint32_t w = 0, h = 0, bpp = 0, row_pitch = 0;
        uint32_t x = 0, y = 0;    // the region's corner in the staging rows (a depth texture is copied whole)
        std::vector<uint8_t> bytes;
        bool ready = false;
    };

    template <class T> static T* at(std::vector<T>& v, uint32_t slot);
    template <class T> static T& grow(std::vector<T>& v, uint32_t slot);
    WGPUBuffer acquire_buffer(uint64_t size, WGPUBufferUsage usage, uint64_t* capacity, const char* label);
    void retire_buffer(Buf& b);
    bool build_group(Group& g, std::string* error);
    bool ensure_arena(uint64_t bytes);
    void push_scope();
    void pop_scope(const char* what);

    WGPUDevice device_ = nullptr;
    WGPUQueue queue_ = nullptr;
    bool compat_ = false;
    // The device's compute limits, which Limits does not carry: a workgroup above them is refused.
    uint32_t max_invocations_ = 256, max_group_[3] = {256, 256, 64};

    std::vector<Buf> buffers_;
    std::vector<Tex> textures_;
    std::vector<WGPUSampler> samplers_;
    std::vector<Prog> programs_;
    std::vector<Pipe> pipelines_;
    std::vector<Group> groups_;
    std::vector<std::shared_ptr<Pending>> readbacks_;

    struct Spare { WGPUBuffer buffer; uint64_t capacity; WGPUBufferUsage usage; };
    std::vector<Spare> spare_;
    uint64_t spare_bytes_ = 0;
    std::vector<WGPUTextureView> retired_views_;

    // The uniform arena (decision 6): one GPU buffer that every kArena binding refers to. Each
    // submit writes the arenas of its passes into it once, each at a 256-byte-aligned base.
    WGPUBuffer arena_ = nullptr;
    uint64_t arena_cap_ = 0;
    std::vector<uint8_t> arena_stage_;

    std::shared_ptr<std::vector<std::string>> errors_ = std::make_shared<std::vector<std::string>>();
    // False once the backend is gone: a callback that arrives later must not touch it.
    std::shared_ptr<bool> alive_ = std::make_shared<bool>(true);
};

}  // namespace caliper_gfx
