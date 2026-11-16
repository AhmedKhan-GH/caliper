// ops.cpp: the graphics layer's operations with a portable default (gfx.h, decisions 15 and 19):
// update_texture_from_tensor, the texture path of the tensor bridge, the embed views and the
// heatmap and image live views, and fill_texture_from_buffer. Both are built from the interface's
// own calls and the generated programs colormap_texture and fill_texture, so every backend that
// names its program table (set_programs) runs them unchanged; a backend overrides one only for a
// faster private path with the same result.
//
// update_texture_from_tensor reads its source in place where it can and copies it once where it
// cannot:
//  - an f32 source runs the colormap by keys (colormap_texture): the caller's key block decides
//    every texel's level by integer comparisons, so the texture equals the CPU reference
//    map_f32_to_rgba8 on every backend, whatever the adapter's float arithmetic;
//  - a u8 source of RGBA8 texels runs the fill (fill_texture), which takes any 4-byte offset and
//    row pitch, so no row has to be repacked;
//  - host bytes are copied once: f32 values into a staging buffer that the texture keeps, which the
//    colormap then reads, and RGBA8 texels straight into the texture with write_texture.
// The bind groups, staging buffers and pipelines it makes are cached per texture and buffer and
// released once either is gone; an MTLBuffer imported for one update is released after its submit,
// which the layer defers until the submit completed.
#include "gfx/backend_impl.h"
#include "gfx/gfx.h"
#include "gfx/shaders/generated/programs.h"
#include "gfx/shaders/generated/uniforms.h"

#include <cstdarg>
#include <cstdio>
#include <cstring>
#include <string>
#include <unordered_map>

namespace caliper_gfx {

namespace sh = shaders;

struct OpsState {
    const ProgramDesc* table = nullptr;
    uint32_t count = 0;
    struct Prog {
        Program program;
        Pipeline pipeline;
        BindGroup group1;         // the colormap's group 1: its key block, in the frame's arena
        bool tried = false;
        std::string error;
    };
    Prog cmap, fill;
    struct Staging { Buffer buffer; uint64_t size = 0; };
    std::unordered_map<uint64_t, Staging> staging;   // by destination texture: its host values
    struct Key {
        uint32_t program;
        uint64_t buffer, texture;
        bool operator==(const Key& o) const { return program == o.program && buffer == o.buffer && texture == o.texture; }
    };
    struct KeyHash {
        size_t operator()(const Key& k) const {
            return size_t((k.buffer * 0x9e3779b97f4a7c15ull) ^ (k.texture * 0xc2b2ae3d27d4eb4full) ^ k.program);
        }
    };
    std::unordered_map<Key, BindGroup, KeyHash> groups;   // group 0 of a program, by buffer and texture
};

void Backend::set_programs(const ProgramDesc* table, uint32_t count) {
    if (!ops_) ops_ = std::make_shared<OpsState>();
    ops_->table = table;
    ops_->count = count;
}

namespace {

std::string fmt(const char* f, ...) __attribute__((format(printf, 1, 2)));
std::string fmt(const char* f, ...) {
    char buf[512];
    va_list ap;
    va_start(ap, f);
    std::vsnprintf(buf, sizeof buf, f, ap);
    va_end(ap);
    return buf;
}

// The program, its compute pipeline and (for the colormap) its group 1, made on first use.
OpsState::Prog* ensure(Backend& b, OpsState& o, sh::ProgramId id, std::string* why) {
    OpsState::Prog& p = id == sh::ProgramId::colormap_texture ? o.cmap : o.fill;
    if (p.pipeline) return &p;
    if (p.tried) { *why = p.error; return nullptr; }
    p.tried = true;
    const uint32_t i = uint32_t(id);
    if (o.table == nullptr || i >= o.count) {
        p.error = fmt("the %s backend names no generated programs", b.caps().backend);
        *why = p.error;
        return nullptr;
    }
    const ProgramDesc& d = o.table[i];
    p.program = b.create_program(d);
    if (p.program) {
        ComputePipelineDesc cd;
        cd.program = p.program;
        cd.entry = d.entries[0].name;
        cd.label = d.label;
        p.pipeline = b.create_compute_pipeline(cd);
    }
    if (p.pipeline && id == sh::ProgramId::colormap_texture) {
        BindGroupDesc g;
        g.program = p.program;
        g.group = 1;
        g.entry_count = 1;
        g.entries[0].binding = 1;
        g.entries[0].buffer = kArena;
        g.label = "caliper.texture.cmap";
        p.group1 = b.create_bind_group(g);
        if (!p.group1) {
            b.release(p.pipeline);
            p.pipeline = {};
        }
    }
    if (!p.pipeline) {
        p.error = std::string(d.label ? d.label : "a program") + ": " + b.last_error();
        *why = p.error;
        return nullptr;
    }
    return &p;
}

// Group 0 of a texture program over (buffer, texture): the parameters in the arena, the whole
// buffer and the texture. Cached unless the buffer lives for one update only.
BindGroup group0(Backend& b, OpsState& o, const OpsState::Prog& p, uint32_t id, Buffer buf, Texture dst, bool cache) {
    const OpsState::Key key{id, buf.id, dst.id};
    if (cache) {
        auto it = o.groups.find(key);
        if (it != o.groups.end()) return it->second;
    }
    BindGroupDesc g;
    g.program = p.program;
    g.group = 0;
    g.entry_count = 3;
    g.entries[0].binding = 0;
    g.entries[0].buffer = kArena;
    g.entries[1].binding = 1;
    g.entries[1].buffer = buf;   // size 0: the whole buffer, so any 4-byte offset is a base inside it
    g.entries[2].binding = 2;
    g.entries[2].texture = dst;
    g.label = "caliper.texture.source";
    const BindGroup group = b.create_bind_group(g);
    if (group && cache) o.groups[key] = group;
    return group;
}

// Drops the cached groups and staging buffers of textures and buffers that are gone.
void prune(Backend& b, OpsState& o) {
    for (auto it = o.staging.begin(); it != o.staging.end();) {
        if (b.alive(Texture{it->first})) { ++it; continue; }
        b.release(it->second.buffer);
        it = o.staging.erase(it);
    }
    for (auto it = o.groups.begin(); it != o.groups.end();) {
        if (b.alive(Buffer{it->first.buffer}) && b.alive(Texture{it->first.texture})) { ++it; continue; }
        b.release(it->second);
        it = o.groups.erase(it);
    }
}

uint32_t groups_over(uint32_t n, uint16_t size) { return (n + size - 1) / size; }

}  // namespace

UpdateResult Backend::update_texture_from_tensor(Texture dst, const TensorSource& src, const TextureMapping& m) {
    auto no = [&](const std::string& why) {
        refuse("update_texture_from_tensor: " + why);
        return UpdateResult{};
    };
    if (!ops_) ops_ = std::make_shared<OpsState>();
    OpsState& o = *ops_;
    prune(*this, o);
    if (!alive(dst)) return no("the texture is released or unknown");
    const TextureInfo ti = texture_info(dst);
    if (ti.format != Format::RGBA8Unorm) return no("the texture must be RGBA8Unorm");
    if (src.w != ti.width || src.h != ti.height)
        return no(fmt("the data is %ux%u and the texture %ux%u", src.w, src.h, ti.width, ti.height));
    const bool f32 = src.dtype == CALIPER_DT_F32;
    if (f32 ? src.channels != 1 : !(src.dtype == CALIPER_DT_U8 && src.channels == 4))
        return no("the data must be f32 with one channel or u8 with four (RGBA8 texels)");
    if (src.stride_x != int64_t(src.channels)) return no("a row's texels must be contiguous (stride_x = channels)");
    const uint64_t esize = f32 ? 4 : 1, texel = esize * src.channels;
    const uint64_t row_elems = src.stride_y != 0 ? uint64_t(src.stride_y) : uint64_t(src.w) * src.channels;
    if (src.stride_y < 0 || row_elems < uint64_t(src.w) * src.channels) return no("a row's stride is shorter than a row");
    const uint64_t row_bytes = row_elems * esize;
    if (src.offset % 4 != 0 || row_bytes % 4 != 0) return no("the data's byte offset and row pitch must be multiples of 4");
    const uint64_t extent = row_bytes * (src.h - 1) + uint64_t(src.w) * texel;   // bytes from the offset
    if (!src.buffer && src.tensor == nullptr) return no("the source names neither a buffer nor a tensor");
    if (f32 && (m.cmap == nullptr || m.cmap_bytes != sizeof(sh::Cmap)))
        return no(fmt("an f32 update needs the colormap's key block of %zu bytes", sizeof(sh::Cmap)));
    const bool host = !src.buffer && src.tensor->device == CALIPER_DEV_CPU;
    if (host && !f32) {
        if (!(ti.usage & kCopyDstTex)) return no("an upload of RGBA8 texels needs a kCopyDstTex texture");
    } else if (!(ti.usage & kStorageTex)) {
        return no("the texture must be a kStorageTex texture, which the colormap and the fill write");
    }

    UpdateResult r;
    Buffer buffer = src.buffer;
    Buffer transient;   // an MTLBuffer imported for this update only
    uint64_t offset = src.offset;
    uint64_t row = row_bytes;
    if (host) {
        const uint8_t* bytes = static_cast<const uint8_t*>(src.tensor->data);
        if (bytes == nullptr) return no("the tensor's data is null");
        bytes += src.offset;
        const uint64_t tight = uint64_t(src.w) * texel, size = tight * src.h;
        r.route = Route::CopiedFromHost;
        r.path = "cpu-staged";
        r.bytes_copied = size;
        if (!f32) {   // RGBA8 texels straight into the texture
            if (row_bytes > 0xffffffffull) return no("the row pitch is too large");
            if (!write_texture(dst, bytes, uint32_t(row_bytes))) return UpdateResult{};
            r.ok = true;
            return r;
        }
        OpsState::Staging& st = o.staging[dst.id];
        if (!st.buffer || st.size != size) {
            if (st.buffer) release(st.buffer);
            st.buffer = create_buffer({size, kStorage | kCopyDst, false, "caliper.texture.values"});
            st.size = st.buffer ? size : 0;
            if (!st.buffer) {
                o.staging.erase(dst.id);
                return UpdateResult{};
            }
        }
        if (row_bytes == tight) {
            if (!write_buffer(st.buffer, 0, bytes, size)) return UpdateResult{};
        } else {
            for (uint32_t y = 0; y < src.h; ++y)
                if (!write_buffer(st.buffer, uint64_t(y) * tight, bytes + uint64_t(y) * row_bytes, tight)) return UpdateResult{};
        }
        buffer = st.buffer;
        offset = 0;
        row = tight;
    } else if (buffer) {
        if (!alive(buffer)) return no("the buffer is released or unknown");
        const BufferInfo bi = buffer_info(buffer);
        if (!(bi.usage & kStorage)) return no("the buffer must be readable as storage (kStorage)");
        if (src.offset > bi.size || extent > bi.size - src.offset)
            return no(fmt("the data (%llu bytes from byte %llu) reaches past the end of its buffer of %llu bytes",
                          (unsigned long long)extent, (unsigned long long)src.offset, (unsigned long long)bi.size));
        r.route = bi.imported ? bi.route : Route::InPlace;
        r.path = f32 ? "compute-imported" : "blit-imported";
    } else {
        if (src.tensor->device != CALIPER_DEV_METAL || !caps().has(kCapImportMetalBuffer))
            return no(fmt("the %s backend cannot read a tensor on this device in place", caps().backend));
        if (src.tensor->data == nullptr) return no("the tensor's MTLBuffer is null");
        ExternalBuffer ext;
        ext.kind = ExternalKind::MetalBuffer;
        ext.handle = src.tensor->data;
        ext.size = src.offset + extent;   // a multiple of 4: offset, row pitch and texels are
        ext.usage = kStorage;
        ext.allow_copy = false;
        ext.label = "caliper.texture.tensor";
        const ImportResult imp = import_buffer(ext);
        if (!imp.buffer) return no("the tensor's MTLBuffer cannot be read in place: " + imp.reason);
        transient = buffer = imp.buffer;
        r.route = Route::InPlace;
        r.path = f32 ? "compute" : "blit";
    }

    ExternalSync sync = src.sync;
    if (sync.kind == ExternalSync::Drained && !host && src.tensor != nullptr && src.tensor->stream != nullptr)
        on_sync_after_stream(src.tensor->stream, &sync);   // false: unordered, as the bridge's caps say

    std::string why;
    OpsState::Prog* p = ensure(*this, o, f32 ? sh::ProgramId::colormap_texture : sh::ProgramId::fill_texture, &why);
    const uint32_t pid = uint32_t(f32 ? sh::ProgramId::colormap_texture : sh::ProgramId::fill_texture);
    const BindGroup g0 = p ? group0(*this, o, *p, pid, buffer, dst, !transient) : BindGroup{};
    if (!g0) {
        if (transient) release(transient);
        return no(p ? "the source's bind group: " + last_error() : why);
    }
    Frame f;
    if (sync.kind != ExternalSync::Drained) f.wait(sync);
    const uint16_t* wg = o.table[pid].entries[0].workgroup_size;
    const Texture writes[1] = {dst};
    if (f32) {
        const sh::TexParams tp{src.w, src.h, uint32_t(offset / 4), uint32_t(row / 4)};
        const uint32_t offsets[2] = {f.uniform(&tp, sizeof tp), f.uniform(m.cmap, m.cmap_bytes)};
        const BindGroup groups[2] = {g0, p->group1};
        f.dispatch(p->pipeline, groups, offsets, groups_over(src.w, wg[0]), groups_over(src.h, wg[1]), 1, writes);
    } else {
        const sh::FillParams fp{src.w, src.h, uint32_t(offset / 4), uint32_t(row / 4)};
        const uint32_t offsets[1] = {f.uniform(&fp, sizeof fp)};
        const BindGroup groups[1] = {g0};
        f.dispatch(p->pipeline, groups, offsets, groups_over(src.w, wg[0]), groups_over(src.h, wg[1]), 1, writes);
    }
    r.serial = submit(std::move(f));
    if (transient) {   // destroyed once the submit that reads them completed
        release(g0);
        release(transient);
    }
    if (r.serial == 0) return UpdateResult{};
    r.ok = true;
    return r;
}

UpdateResult Backend::fill_texture_from_buffer(Texture dst, Buffer src, uint64_t offset, uint32_t row_pitch) {
    auto no = [&](const std::string& why) {
        refuse("fill_texture_from_buffer: " + why);
        return UpdateResult{};
    };
    UpdateResult r;
    r.route = Route::CopiedOnGpu;
    if (row_pitch % kCopyRowPitchAlign == 0) {   // a plain copy takes 256-byte rows
        Frame f;
        f.copy_buffer_to_texture(src, offset, row_pitch, dst);
        r.serial = submit(std::move(f));
        if (r.serial == 0) return UpdateResult{};
        r.ok = true;
        r.path = "copy";
        return r;
    }
    if (!ops_) ops_ = std::make_shared<OpsState>();
    OpsState& o = *ops_;
    prune(*this, o);
    if (!alive(dst) || !alive(src)) return no("the texture or the buffer is released or unknown");
    const TextureInfo ti = texture_info(dst);
    const BufferInfo bi = buffer_info(src);
    if (ti.format != Format::RGBA8Unorm || !(ti.usage & kStorageTex))
        return no("a row pitch that is not a multiple of 256 needs the fill program, which writes kStorageTex RGBA8 textures");
    if (!(bi.usage & kStorage)) return no("the fill program reads the buffer as storage (kStorage)");
    if (offset % 4 != 0 || row_pitch % 4 != 0 || row_pitch < ti.width * 4u)
        return no("the offset and row pitch must be multiples of 4, and a row pitch must hold a row");
    const uint64_t need = uint64_t(row_pitch) * (ti.height - 1) + uint64_t(ti.width) * 4;
    if (offset > bi.size || need > bi.size - offset) return no("the fill reads past the end of its buffer");
    std::string why;
    OpsState::Prog* p = ensure(*this, o, sh::ProgramId::fill_texture, &why);
    if (p == nullptr) return no(why);
    const uint32_t pid = uint32_t(sh::ProgramId::fill_texture);
    const BindGroup g0 = group0(*this, o, *p, pid, src, dst, true);
    if (!g0) return no("the source's bind group: " + last_error());
    Frame f;
    const sh::FillParams fp{ti.width, ti.height, uint32_t(offset / 4), row_pitch / 4};
    const uint32_t offsets[1] = {f.uniform(&fp, sizeof fp)};
    const BindGroup groups[1] = {g0};
    const Texture writes[1] = {dst};
    const uint16_t* wg = o.table[pid].entries[0].workgroup_size;
    f.dispatch(p->pipeline, groups, offsets, groups_over(ti.width, wg[0]), groups_over(ti.height, wg[1]), 1, writes);
    r.serial = submit(std::move(f));
    if (r.serial == 0) return UpdateResult{};
    r.ok = true;
    r.path = "fill";
    return r;
}

}  // namespace caliper_gfx
