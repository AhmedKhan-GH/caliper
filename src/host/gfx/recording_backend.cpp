// recording_backend.cpp: the graphics layer's backend without a GPU (recording_backend.h).
#include "gfx/recording_backend.h"

#include <algorithm>
#include <cstring>

namespace caliper_gfx {

Caps RecordingBackend::default_caps() {
    Caps c;
    c.bits = kCapVertexStorage | kCapPersistentMap | kCapBlockingWait | kCapImportHostMemory;
    c.backend = "recording";
    c.adapter = "none";
    return c;
}

RecordingBackend::RecordingBackend(const Caps& caps) : Backend(caps) {}
RecordingBackend::~RecordingBackend() = default;

size_t RecordingBackend::count(OpKind k, HandleKind object) const {
    return size_t(std::count_if(ops_.begin(), ops_.end(), [&](const Op& o) {
        return o.kind == k && (object == HandleKind::None || o.object == object);
    }));
}

namespace {
template <class V> void grow(V& v, uint32_t slot) {
    if (v.size() <= slot) v.resize(size_t(slot) + 1);
}
}  // namespace

bool RecordingBackend::on_create_buffer(uint32_t slot, const BufferDesc& d, std::string*) {
    grow(buffers_, slot);
    buffers_[slot] = Buf{};
    buffers_[slot].own.assign(size_t(d.size), 0);
    buffers_[slot].size = d.size;
    ops_.push_back({OpKind::Create, HandleKind::Buffer, slot, d.size, 0, Route::InPlace});
    return true;
}

bool RecordingBackend::on_create_texture(uint32_t slot, const TextureDesc& d, std::string*) {
    grow(textures_, slot);
    Tex& t = textures_[slot];
    t.w = d.width;
    t.h = d.height;
    t.bpp = format_bytes(d.format);
    t.format = d.format;
    t.px.assign(size_t(t.w) * t.h * t.bpp, 0);
    ops_.push_back({OpKind::Create, HandleKind::Texture, slot, t.px.size(), 0, Route::InPlace});
    return true;
}

bool RecordingBackend::on_create_sampler(uint32_t slot, const SamplerDesc&, std::string*) {
    ops_.push_back({OpKind::Create, HandleKind::Sampler, slot, 0, 0, Route::InPlace});
    return true;
}

bool RecordingBackend::on_create_program(uint32_t slot, const ProgramDesc&, std::string*) {
    ops_.push_back({OpKind::Create, HandleKind::Program, slot, 0, 0, Route::InPlace});
    return true;
}

bool RecordingBackend::on_create_render_pipeline(uint32_t slot, const RenderPipelineDesc&, std::string*) {
    ops_.push_back({OpKind::Create, HandleKind::Pipeline, slot, 0, 0, Route::InPlace});
    return true;
}

bool RecordingBackend::on_create_compute_pipeline(uint32_t slot, const ComputePipelineDesc&, std::string*) {
    ops_.push_back({OpKind::Create, HandleKind::Pipeline, slot, 0, 0, Route::InPlace});
    return true;
}

bool RecordingBackend::on_create_bind_group(uint32_t slot, const BindGroupDesc&, std::string*) {
    ops_.push_back({OpKind::Create, HandleKind::BindGroup, slot, 0, 0, Route::InPlace});
    return true;
}

void RecordingBackend::on_destroy(uint64_t handle) {
    const uint32_t slot = handle_slot(handle);
    const HandleKind kind = handle_kind(handle);
    if (kind == HandleKind::Buffer && slot < buffers_.size()) buffers_[slot] = Buf{};
    if (kind == HandleKind::Texture && slot < textures_.size()) textures_[slot] = Tex{};
    if (kind == HandleKind::Readback && slot < readbacks_.size()) readbacks_[slot].clear();
    ops_.push_back({OpKind::Destroy, kind, slot, 0, 0, Route::InPlace});
}

void RecordingBackend::on_write_buffer(uint32_t slot, uint64_t offset, const void* data, uint64_t size, bool) {
    std::memcpy(bytes(slot) + offset, data, size_t(size));
    ops_.push_back({OpKind::WriteBuffer, HandleKind::Buffer, slot, size, 0, Route::InPlace});
}

void RecordingBackend::on_write_texture(uint32_t slot, const void* data, uint32_t row_pitch, bool) {
    Tex& t = textures_[slot];
    const size_t row = size_t(t.w) * t.bpp;
    for (uint32_t y = 0; y < t.h; ++y)
        std::memcpy(t.px.data() + y * row, static_cast<const uint8_t*>(data) + size_t(y) * row_pitch, row);
    ops_.push_back({OpKind::WriteTexture, HandleKind::Texture, slot, row * t.h, 0, Route::InPlace});
}

void* RecordingBackend::on_mapped(uint32_t slot) { return bytes(slot); }

bool RecordingBackend::on_import_buffer(uint32_t slot, const ExternalBuffer& ext, ImportResult* r) {
    if (ext.kind != ExternalKind::HostMemory) {
        r->reason = "the recording backend imports host memory only";
        return false;
    }
    grow(buffers_, slot);
    buffers_[slot] = Buf{};
    buffers_[slot].size = ext.size;
    const bool aligned = reinterpret_cast<uintptr_t>(ext.handle) % kHostPage == 0 && ext.size % kHostPage == 0;
    if (caps().has(kCapImportHostMemory) && aligned) {
        buffers_[slot].wrapped = static_cast<uint8_t*>(ext.handle);
        r->route = Route::InPlace;
        r->bytes_copied = 0;
    } else {
        if (!ext.allow_copy) {
            r->reason = "host memory that is not page-aligned can only be copied";
            return false;
        }
        buffers_[slot].own.assign(static_cast<const uint8_t*>(ext.handle),
                                  static_cast<const uint8_t*>(ext.handle) + ext.size);
        r->route = Route::CopiedFromHost;
        r->bytes_copied = ext.size;
        r->reason = aligned ? "this backend does not wrap host memory" : "the memory is not page-aligned";
    }
    ops_.push_back({OpKind::Import, HandleKind::Buffer, slot, r->bytes_copied, 0, r->route});
    return true;
}

bool RecordingBackend::on_import_texture(uint32_t, const ExternalTexture&, ImportResult* r) {
    r->reason = "the recording backend imports no textures";
    return false;
}

void RecordingBackend::clear_target(Texture tex, uint32_t rgba, bool id, bool depth) {
    Tex& t = textures_[handle_slot(tex.id)];
    const size_t n = size_t(t.w) * t.h;
    uint8_t texel[8] = {};
    if (depth) {
        const float one = 1.0f;
        std::memcpy(texel, &one, 4);
    } else if (!id) {
        const uint8_t r = uint8_t(rgba), g = uint8_t(rgba >> 8), b = uint8_t(rgba >> 16), a = uint8_t(rgba >> 24);
        if (t.format == Format::RGBA8Unorm) { texel[0] = r; texel[1] = g; texel[2] = b; texel[3] = a; }
        if (t.format == Format::BGRA8Unorm) { texel[0] = b; texel[1] = g; texel[2] = r; texel[3] = a; }
        if (t.format == Format::R8Unorm) texel[0] = r;
    }
    for (size_t i = 0; i < n; ++i) std::memcpy(t.px.data() + i * t.bpp, texel, t.bpp);
}

bool RecordingBackend::on_submit(Serial s, const Frame& frame, Stats*, std::string*) {
    for (const Pass& pass : frame.passes()) {
        if (const auto* c = std::get_if<CopyPass>(&pass)) {
            Tex& t = textures_[handle_slot(c->dst.id)];
            const uint8_t* src = bytes(handle_slot(c->src.id)) + c->offset;
            const size_t row = size_t(t.w) * t.bpp;
            for (uint32_t y = 0; y < t.h; ++y) std::memcpy(t.px.data() + y * row, src + size_t(y) * c->row_pitch, row);
        } else if (const auto* r = std::get_if<RenderPass>(&pass)) {
            const RenderTarget& tg = r->list.target();
            if (tg.clear) {
                clear_target(tg.color, tg.clear_rgba, false, false);
                if (tg.id) clear_target(tg.id, 0, true, false);
                if (tg.depth) clear_target(tg.depth, 0, false, true);
            }
        }
    }
    if (keep_frames_) submitted_.push_back({s, frame});
    ops_.push_back({OpKind::Submit, HandleKind::None, 0, 0, s, Route::InPlace});
    retire(s);
    return true;
}

void RecordingBackend::retire(Serial s) {
    if (auto_complete_) complete(s);
    else pending_.push_back(s);
}

void RecordingBackend::finish(Serial s) {
    for (size_t i = 0; i < pending_.size();) {
        if (pending_[i] <= s) {
            complete(pending_[i]);
            pending_.erase(pending_.begin() + long(i));
        } else {
            ++i;
        }
    }
    poll();
}

bool RecordingBackend::on_wait(Serial s) {
    finish(s);
    return true;
}

bool RecordingBackend::on_read_texture(Serial s, uint32_t readback_slot, uint32_t texture_slot, uint32_t x,
                                       uint32_t y, uint32_t w, uint32_t h, std::string*) {
    grow(readbacks_, readback_slot);
    const Tex& t = textures_[texture_slot];
    std::vector<uint8_t>& out = readbacks_[readback_slot];
    const size_t row = size_t(w) * t.bpp;
    out.resize(row * h);
    for (uint32_t j = 0; j < h; ++j)
        std::memcpy(out.data() + j * row, t.px.data() + (size_t(y + j) * t.w + x) * t.bpp, row);
    ops_.push_back({OpKind::ReadTexture, HandleKind::Texture, texture_slot, out.size(), s, Route::InPlace});
    retire(s);
    return true;
}

void RecordingBackend::on_take_readback(uint32_t readback_slot, std::vector<uint8_t>* out) {
    *out = std::move(readbacks_[readback_slot]);
    readbacks_[readback_slot].clear();
}

uint64_t RecordingBackend::on_imgui_texture_id(uint32_t slot) { return uint64_t(slot) + 1; }

std::vector<uint8_t> RecordingBackend::buffer_bytes(Buffer b) const {
    if (!alive(b)) return {};
    const Buf& buf = buffers_[handle_slot(b.id)];
    const uint8_t* p = buf.wrapped ? buf.wrapped : buf.own.data();
    return std::vector<uint8_t>(p, p + buf.size);
}

std::vector<uint8_t> RecordingBackend::texture_bytes(Texture t) const {
    if (!alive(t)) return {};
    return textures_[handle_slot(t.id)].px;
}

}  // namespace caliper_gfx
