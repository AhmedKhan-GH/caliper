// recording_backend.h: a backend of the graphics layer with no GPU, for headless tests. It
// records every resource operation and keeps a copy of every frame it is given, holds buffer
// and texture contents in host memory, applies writes, copies and the clears of render passes
// (RGBA8, BGRA8 and R8 colour targets take the clear colour, ID targets 0, depth 1.0; it does
// not rasterise), and answers readbacks from those contents. Completion is immediate by
// default; with auto_complete off a test completes serials itself, to exercise the
// asynchronous paths.
#pragma once

#include "gfx/backend_impl.h"

#include <vector>

namespace caliper_gfx {

class RecordingBackend final : public Backend {
public:
    // WebGPU's default limits, blocking waits, persistent maps, vertex storage and the wrapping
    // of host memory aligned to kHostPage.
    static Caps default_caps();
    static constexpr uint64_t kHostPage = 16384;

    explicit RecordingBackend(const Caps& caps = default_caps());
    ~RecordingBackend() override;

    enum class OpKind : uint32_t { Create, Destroy, WriteBuffer, WriteTexture, Import, Submit, ReadTexture };
    struct Op {
        OpKind kind;
        HandleKind object;   // what the operation is about
        uint32_t slot;
        uint64_t bytes;      // written, imported or copied
        Serial serial;       // Submit and ReadTexture
        Route route;         // Import
    };
    const std::vector<Op>& ops() const { return ops_; }
    size_t count(OpKind k, HandleKind object = HandleKind::None) const;

    struct Submitted { Serial serial; Frame frame; };
    const std::vector<Submitted>& submitted() const { return submitted_; }
    void set_keep_frames(bool keep) { keep_frames_ = keep; }   // off for timing runs
    void clear_log() { ops_.clear(); submitted_.clear(); }

    void set_auto_complete(bool on) { auto_complete_ = on; }
    void finish(Serial s);                  // completes every pending serial up to s
    // The GPU time that each submit reports on completion (< 0, the default, reports none),
    // so that a test can drive the time watchdogs of the hosts without a GPU.
    void set_gpu_ms(double ms) { gpu_ms_ = ms; }

    std::vector<uint8_t> buffer_bytes(Buffer b) const;
    std::vector<uint8_t> texture_bytes(Texture t) const;

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
    struct Buf { std::vector<uint8_t> own; uint8_t* wrapped = nullptr; uint64_t size = 0; };
    struct Tex { std::vector<uint8_t> px; uint32_t w = 0, h = 0, bpp = 0; Format format = Format::RGBA8Unorm; };
    uint8_t* bytes(uint32_t slot) { Buf& b = buffers_[slot]; return b.wrapped ? b.wrapped : b.own.data(); }
    void retire(Serial s);
    void clear_target(Texture t, uint32_t rgba, bool id, bool depth);

    std::vector<Op> ops_;
    std::vector<Submitted> submitted_;
    std::vector<Buf> buffers_;
    std::vector<Tex> textures_;
    std::vector<std::vector<uint8_t>> readbacks_;
    std::vector<Serial> pending_;
    bool keep_frames_ = true;
    bool auto_complete_ = true;
    double gpu_ms_ = -1;
};

}  // namespace caliper_gfx
