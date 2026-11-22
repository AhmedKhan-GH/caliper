// core.cpp: the backend-neutral core of the graphics layer (gfx.h). It keeps one record per
// live handle, validates every call and every frame against the backend's limits before a
// backend sees it, defers destruction until the last submit that used a resource completed,
// counts what each submit did, and tracks serials and readbacks. The backends implement only
// the hooks.
#include "gfx/backend_impl.h"

#include <algorithm>
#include <chrono>
#include <cstdarg>
#include <cstdio>
#include <cstring>
#include <unordered_map>

namespace caliper_gfx {

// ---- Small helpers -----------------------------------------------------------------------------

uint32_t format_bytes(Format f) {
    switch (f) {
        case Format::RGBA8Unorm: case Format::BGRA8Unorm: case Format::R32Uint:
        case Format::R32Float: case Format::Depth32Float: return 4;
        case Format::R8Unorm: return 1;
        case Format::R16Uint: return 2;
        case Format::RG32Uint: return 8;
    }
    return 0;
}

uint32_t vertex_format_bytes(VertexFormat f) {
    switch (f) {
        case VertexFormat::F32: case VertexFormat::Unorm8x4: case VertexFormat::U32:
        case VertexFormat::S32: return 4;
        case VertexFormat::F32x2: case VertexFormat::U32x2: return 8;
        case VertexFormat::F32x3: return 12;
        case VertexFormat::F32x4: return 16;
        case VertexFormat::U8: return 1;
    }
    return 0;
}

const char* route_name(Route r) {
    switch (r) {
        case Route::InPlace: return "in-place";
        case Route::CopiedOnGpu: return "copied-on-gpu";
        case Route::CopiedFromHost: return "copied-from-host";
        case Route::Refused: return "refused";
    }
    return "refused";
}

bool is_integer_format(Format f) {
    return f == Format::R16Uint || f == Format::R32Uint || f == Format::RG32Uint;
}
bool is_depth_format(Format f) { return f == Format::Depth32Float; }
bool is_buffer_binding(BindingType t) {
    return t == BindingType::Uniform || t == BindingType::Storage || t == BindingType::StorageReadWrite;
}

namespace {

bool valid_format(Format f) { return uint32_t(f) <= uint32_t(Format::Depth32Float); }
bool blendable(Format f) {
    return f == Format::RGBA8Unorm || f == Format::BGRA8Unorm || f == Format::R8Unorm;
}
bool filterable(Format f) { return blendable(f); }
bool storage_format(Format f) {
    return f == Format::RGBA8Unorm || f == Format::R32Uint || f == Format::R32Float ||
           f == Format::RG32Uint;
}

#if defined(__GNUC__) || defined(__clang__)
std::string fmt(const char* f, ...) __attribute__((format(printf, 1, 2)));
#endif
std::string fmt(const char* f, ...) {
    char buf[512];
    va_list ap;
    va_start(ap, f);
    std::vsnprintf(buf, sizeof buf, f, ap);
    va_end(ap);
    return buf;
}

struct Hasher {
    uint64_t h = 1469598103934665603ull;
    void bytes(const void* p, size_t n) {
        const auto* b = static_cast<const uint8_t*>(p);
        for (size_t i = 0; i < n; ++i) { h ^= b[i]; h *= 1099511628211ull; }
    }
    template <class T> void add(const T& v) { bytes(&v, sizeof v); }
    void str(const char* s) { if (s) bytes(s, std::strlen(s)); add(uint8_t(0)); }
};

uint32_t arena_push(std::vector<uint8_t>& arena, std::string& error, const void* data, uint32_t size) {
    if (data == nullptr || size == 0 || size > 65536) {
        if (error.empty())
            error = fmt("a uniform block of %u bytes is refused (1 to 65,536 bytes)", size);
        return kNoOffset;
    }
    const uint64_t off = align_up(arena.size(), kUniformAlign);
    if (off + align_up(size, kUniformAlign) > 0xffff0000ull) {
        if (error.empty()) error = "the uniform arena is full (4 GiB)";
        return kNoOffset;
    }
    // Each slice is padded to 256 bytes, so a binding of the block's slot size always fits.
    arena.resize(size_t(off + align_up(size, kUniformAlign)), 0);
    std::memcpy(arena.data() + off, data, size);
    return uint32_t(off);
}

}  // namespace

// ---- DrawList and Frame -----------------------------------------------------------------------

uint32_t DrawList::uniform(const void* data, uint32_t size) { return arena_push(arena_, error_, data, size); }

void DrawList::reserve(size_t draws, size_t arena_bytes) {
    records_.reserve(draws);
    arena_.reserve(arena_bytes);
}

void DrawList::reset(const RenderTarget& target) {
    target_ = target;
    records_.clear();
    arena_.clear();
    error_.clear();
}

void Frame::wait(const ExternalSync& sync) { passes_.push_back(WaitPass{sync}); }

void Frame::dispatch(Pipeline p, View<BindGroup> groups, View<uint32_t> offsets, uint32_t x,
                     uint32_t y, uint32_t z, View<Texture> writes) {
    DispatchPass d;
    d.pipeline = p;
    if (groups.size > kMaxBindGroups || offsets.size > kMaxDynamicOffsets) {
        if (error_.empty()) error_ = "a dispatch takes at most 4 bind groups and 4 dynamic offsets";
    }
    for (size_t i = 0; i < groups.size && i < kMaxBindGroups; ++i) d.groups[i] = groups[i];
    for (size_t i = 0; i < offsets.size && i < kMaxDynamicOffsets; ++i) d.dynamic_offsets[i] = offsets[i];
    d.x = x; d.y = y; d.z = z;
    d.writes.assign(writes.begin(), writes.end());
    passes_.push_back(std::move(d));
}

void Frame::copy_buffer_to_texture(Buffer src, uint64_t offset, uint32_t row_pitch, Texture dst) {
    passes_.push_back(CopyPass{src, offset, row_pitch, dst});
}

void Frame::render(DrawList&& list) { passes_.push_back(RenderPass{std::move(list)}); }

uint32_t Frame::uniform(const void* data, uint32_t size) { return arena_push(arena_, error_, data, size); }

// ---- Completion --------------------------------------------------------------------------------

void CompletionSink::complete(Serial s, double gpu_ms) {
    std::lock_guard<std::mutex> lock(mu_);
    if (gpu_ms >= 0) { gpu_ms_ += gpu_ms; ++gpu_n_; gpu_known_ = true; }
    // A wait completes a serial without its time (-1) before or after the backend's handler
    // gives it: only a measured time is kept.
    if (gpu_ms >= 0) {
        times_[s] = gpu_ms;
        while (times_.size() > kSubmitTimes) times_.erase(times_.begin());
    }
    Serial through = through_.load(std::memory_order_relaxed);
    if (s <= through) return;
    if (s != through + 1) { early_.insert(s); return; }
    through = s;
    for (auto it = early_.begin(); it != early_.end() && *it == through + 1; it = early_.erase(it))
        through = *it;
    through_.store(through, std::memory_order_release);
}

double CompletionSink::take_gpu_ms(uint32_t* submits) {
    std::lock_guard<std::mutex> lock(mu_);
    const double r = gpu_known_ ? gpu_ms_ : -1.0;
    if (submits) *submits = gpu_n_;
    gpu_ms_ = 0;
    gpu_n_ = 0;
    gpu_known_ = false;
    return r;
}

bool CompletionSink::submit_gpu_ms(Serial s, double* gpu_ms) const {
    if (s == 0 || s > completed_through()) return false;
    std::lock_guard<std::mutex> lock(mu_);
    const auto it = times_.find(s);
    if (it == times_.end()) return false;
    if (gpu_ms) *gpu_ms = it->second;
    return true;
}

// ---- Program lookups ---------------------------------------------------------------------------

const EntryInfo* ProgramInfo::entry(const char* name, uint8_t stage) const {
    if (name == nullptr) return nullptr;
    for (const EntryInfo& e : entries)
        if (e.stage == stage && e.name == name) return &e;
    return nullptr;
}

const BindingDesc* ProgramInfo::binding(uint32_t group, uint32_t b) const {
    for (const BindingDesc& d : bindings)
        if (d.group == group && d.binding == b) return &d;
    return nullptr;
}

// ---- Handle tables -----------------------------------------------------------------------------

namespace {

constexpr uint64_t kGenerationLimit = 1ull << kHandleGenerationBits;

template <class Info> struct Table {
    struct Slot {
        uint64_t gen = 1;
        bool live = false;      // usable
        bool pending = false;   // released, destroyed once its last use completed
        Serial last_use = 0;
        uint64_t checked = 0;   // the validation epoch that last checked what it refers to
        Info info{};
    };
    HandleKind kind;
    std::vector<Slot> slots;
    std::vector<uint32_t> free_list;

    explicit Table(HandleKind k) : kind(k) {}

    uint64_t id_of(uint32_t i) const {
        return (uint64_t(kind) << 60) | (slots[i].gen << kHandleSlotBits) | i;
    }
    uint64_t alloc() {
        uint32_t i;
        if (!free_list.empty()) {
            i = free_list.back();
            free_list.pop_back();
        } else {
            if (slots.size() >= kHandleSlotMask) return 0;   // the last slot is kArena's
            slots.emplace_back();
            i = uint32_t(slots.size() - 1);
        }
        Slot& s = slots[i];
        s.live = true;
        s.pending = false;
        s.last_use = 0;
        s.checked = 0;
        s.info = Info{};
        return id_of(i);
    }
    Slot* find(uint64_t id, bool include_pending) {
        if (handle_kind(id) != kind) return nullptr;
        const uint32_t i = handle_slot(id);
        if (i >= slots.size()) return nullptr;
        Slot& s = slots[i];
        if (((id >> kHandleSlotBits) & (kGenerationLimit - 1)) != s.gen) return nullptr;
        if (s.live || (include_pending && s.pending)) return &s;
        return nullptr;
    }
    Slot* get(uint64_t id) { return find(id, false); }
    const Slot* get(uint64_t id) const { return const_cast<Table*>(this)->find(id, false); }
    const Slot* any(uint64_t id) const { return const_cast<Table*>(this)->find(id, true); }
    void kill(uint32_t i) { slots[i].live = false; slots[i].pending = true; }
    void recycle(uint32_t i) {
        Slot& s = slots[i];
        s.live = false;
        s.pending = false;
        s.info = Info{};
        if (++s.gen >= kGenerationLimit) return;   // retired: an id is never reused
        free_list.push_back(i);
    }
};

struct SamplerInfo { bool linear = true; };
struct ReadbackInfo { Serial serial = 0; Format format = Format::RGBA8Unorm; uint32_t w = 0, h = 0; };

}  // namespace

class Core {
public:
    explicit Core(const Caps& c) : caps(c) {}

    Caps caps;
    std::string error;
    Table<BufferInfo> buffers{HandleKind::Buffer};
    Table<TextureInfo> textures{HandleKind::Texture};
    Table<SamplerInfo> samplers{HandleKind::Sampler};
    Table<std::shared_ptr<const ProgramInfo>> programs{HandleKind::Program};
    Table<PipelineInfo> pipelines{HandleKind::Pipeline};
    Table<BindGroupInfo> groups{HandleKind::BindGroup};
    Table<ReadbackInfo> readbacks{HandleKind::Readback};
    std::unordered_multimap<uint64_t, uint64_t> pipeline_cache;   // descriptor hash -> id
    struct Deferred { uint64_t id; Serial serial; };
    std::vector<Deferred> deferred;
    std::shared_ptr<CompletionSink> sink = std::make_shared<CompletionSink>();
    Serial next_serial = 1;
    uint64_t epoch = 0;   // one per frame validation
    Stats stats;

    Serial through() const { return sink->completed_through(); }

    bool validate_frame(const Frame& f, Serial s, Stats* st);
    // targets: a render pass's colour, ID and depth textures, which its groups may not bind.
    bool validate_groups(const char* what, size_t index, const ProgramInfo& P, const BindGroup* gs,
                         const uint32_t* offsets, uint64_t arena_size, Serial s,
                         uint64_t verified[kMaxBindGroups], const Texture* targets = nullptr);
    bool fail(std::string msg) { error = std::move(msg); return false; }
};

// ---- Frame validation (decision 13: the whole frame before anything is encoded) ---------------

bool Core::validate_groups(const char* what, size_t index, const ProgramInfo& P, const BindGroup* gs,
                           const uint32_t* offsets, uint64_t arena_size, Serial s,
                           uint64_t verified[kMaxBindGroups], const Texture* targets) {
    uint32_t k = 0;
    for (uint32_t g = 0; g < kMaxBindGroups; ++g) {
        if ((P.group_mask & (1u << g)) == 0) continue;
        auto* gslot = groups.get(gs[g].id);
        if (gslot == nullptr)
            return fail(fmt("%s %zu: bind group %u is missing, released or unknown", what, index, g));
        const BindGroupInfo& bg = gslot->info;
        if (verified[g] != gs[g].id) {
            if (bg.group != g || bg.layout_hash != P.group_hash[g])
                return fail(fmt("%s %zu: bind group %u was made for another group layout", what, index, g));
            if (gslot->checked != epoch) {   // once per frame: what the group refers to is alive
                for (const BindEntry& e : bg.entries) {
                    if (e.buffer && e.buffer != kArena) {
                        auto* b = buffers.get(e.buffer.id);
                        if (b == nullptr)
                            return fail(fmt("%s %zu: bind group %u refers to a released buffer", what, index, g));
                        b->last_use = s;
                    }
                    if (e.texture) {
                        auto* t = textures.get(e.texture.id);
                        if (t == nullptr)
                            return fail(fmt("%s %zu: bind group %u refers to a released texture", what, index, g));
                        t->last_use = s;
                    }
                    if (e.sampler) {
                        auto* sm = samplers.get(e.sampler.id);
                        if (sm == nullptr)
                            return fail(fmt("%s %zu: bind group %u refers to a released sampler", what, index, g));
                        sm->last_use = s;
                    }
                }
                gslot->checked = epoch;
            }
            // A pass may not read a texture it renders into: WebGPU refuses the pass when it is
            // encoded, and on Metal the result is undefined.
            if (targets != nullptr)
                for (const BindEntry& e : bg.entries) {
                    if (!e.texture) continue;
                    static const char* const kTarget[3] = {"colour", "ID", "depth"};
                    for (int t = 0; t < 3; ++t)
                        if (targets[t] && e.texture == targets[t])
                            return fail(fmt("%s %zu: bind group %u binds the pass's own %s target as a texture", what,
                                            index, g, kTarget[t]));
                }
            gslot->last_use = s;
            verified[g] = gs[g].id;
        }
        for (size_t j = 0; j < bg.entries.size(); ++j) {
            if (!bg.layout[j].dynamic) continue;
            const uint32_t off = offsets[k++];
            const BindEntry& e = bg.entries[j];
            if (off % caps.limits.min_uniform_buffer_offset_alignment != 0)
                return fail(fmt("%s %zu: dynamic offset %u is not a multiple of %u", what, index, off,
                                caps.limits.min_uniform_buffer_offset_alignment));
            const uint64_t limit = e.buffer == kArena ? arena_size : buffers.get(e.buffer.id)->info.size;
            const uint64_t start = e.buffer == kArena ? off : e.offset + off;
            if (start + e.size > limit)
                return fail(fmt("%s %zu: dynamic offset %u reads past the end of the %s", what, index, off,
                                e.buffer == kArena ? "uniform arena" : "buffer"));
        }
    }
    return true;
}

bool Core::validate_frame(const Frame& f, Serial s, Stats* st) {
    if (!f.error().empty()) return fail("frame: " + f.error());
    ++epoch;
    size_t pass_index = 0;
    for (const Pass& pass : f.passes()) {
        ++pass_index;
        if (const auto* w = std::get_if<WaitPass>(&pass)) {
            if (w->sync.kind == ExternalSync::MetalEvent &&
                (!caps.has(kCapWaitMetalEvent) || w->sync.object == nullptr))
                return fail(fmt("pass %zu: this backend cannot wait on a Metal event", pass_index));
            if (w->sync.kind == ExternalSync::CudaStream &&
                (!caps.has(kCapWaitCudaStream) || w->sync.object == nullptr))
                return fail(fmt("pass %zu: this backend cannot wait on a CUDA stream", pass_index));
        } else if (const auto* c = std::get_if<CopyPass>(&pass)) {
            auto* b = buffers.get(c->src.id);
            auto* t = textures.get(c->dst.id);
            if (b == nullptr || t == nullptr)
                return fail(fmt("pass %zu: a copy names a released or unknown buffer or texture", pass_index));
            if (!(b->info.usage & kCopySrc) || !(t->info.usage & kCopyDstTex) || is_depth_format(t->info.format))
                return fail(fmt("pass %zu: a copy needs a kCopySrc buffer and a kCopyDstTex colour texture", pass_index));
            const uint64_t bpp = format_bytes(t->info.format), row = uint64_t(t->info.width) * bpp;
            if (c->row_pitch % kCopyRowPitchAlign != 0 || c->row_pitch < row || c->offset % bpp != 0)
                return fail(fmt("pass %zu: a copy's row pitch %u must be a multiple of 256 and hold a row of %llu bytes",
                                pass_index, c->row_pitch, (unsigned long long)row));
            const uint64_t need = c->offset + uint64_t(c->row_pitch) * (t->info.height - 1) + row;
            if (need > b->info.size)
                return fail(fmt("pass %zu: a copy reads past the end of its buffer", pass_index));
            b->last_use = s;
            t->last_use = s;
            st->copies++;
            st->bytes_copied_on_gpu += row * t->info.height;
        } else if (const auto* d = std::get_if<DispatchPass>(&pass)) {
            auto* p = pipelines.get(d->pipeline.id);
            if (p == nullptr || !p->info.compute)
                return fail(fmt("pass %zu: a dispatch names a released, unknown or render pipeline", pass_index));
            p->last_use = s;
            if (d->x < 1 || d->y < 1 || d->z < 1 || d->x > 65535 || d->y > 65535 || d->z > 65535)
                return fail(fmt("pass %zu: a dispatch takes 1 to 65,535 workgroups in each dimension", pass_index));
            uint64_t verified[kMaxBindGroups] = {};
            if (!validate_groups("dispatch", pass_index, *p->info.layout, d->groups, d->dynamic_offsets,
                                 f.arena().size, s, verified))
                return false;
            for (const Texture& t : d->writes) {
                auto* ts = textures.get(t.id);
                if (ts == nullptr || !(ts->info.usage & kStorageTex))
                    return fail(fmt("pass %zu: a dispatch writes a released texture or one without kStorageTex", pass_index));
                ts->last_use = s;
            }
            st->dispatches++;
        } else if (const auto* r = std::get_if<RenderPass>(&pass)) {
            const DrawList& list = r->list;
            if (!list.error().empty()) return fail(fmt("pass %zu: %s", pass_index, list.error().c_str()));
            const RenderTarget& tg = list.target();
            auto* color = textures.get(tg.color.id);
            if (color == nullptr || !(color->info.usage & kRenderTarget) || is_depth_format(color->info.format))
                return fail(fmt("pass %zu: the target's colour texture is missing, released or not a colour render target", pass_index));
            const uint32_t W = color->info.width, H = color->info.height;
            color->last_use = s;
            const Format color_format = color->info.format;
            if (tg.id) {
                auto* t = textures.get(tg.id.id);
                if (t == nullptr || !(t->info.usage & kRenderTarget) || t->info.format != Format::RG32Uint ||
                    t->info.width != W || t->info.height != H)
                    return fail(fmt("pass %zu: the ID target must be a live RG32Uint render target of the colour's size", pass_index));
                t->last_use = s;
            }
            if (tg.depth) {
                auto* t = textures.get(tg.depth.id);
                if (t == nullptr || !(t->info.usage & kRenderTarget) || t->info.format != Format::Depth32Float ||
                    t->info.width != W || t->info.height != H)
                    return fail(fmt("pass %zu: the depth target must be a live Depth32Float render target of the colour's size", pass_index));
                t->last_use = s;
            }
            const uint64_t arena_size = list.arena().size;
            const Texture targets[3] = {tg.color, tg.id, tg.depth};
            uint64_t last_pipeline = 0;
            const PipelineInfo* pi = nullptr;
            uint64_t verified[kMaxBindGroups] = {};
            size_t index = 0;
            for (const DrawRecord& rec : list.records()) {
                if (rec.pipeline.id != last_pipeline) {
                    auto* p = pipelines.get(rec.pipeline.id);
                    if (p == nullptr || p->info.compute)
                        return fail(fmt("draw %zu: the pipeline is released, unknown or a compute pipeline", index));
                    pi = &p->info;
                    if (pi->desc.color_format != color_format || pi->desc.id_target != bool(tg.id) ||
                        pi->desc.depth != bool(tg.depth))
                        return fail(fmt("draw %zu: the pipeline's attachments do not match the target", index));
                    p->last_use = s;
                    last_pipeline = rec.pipeline.id;
                    std::fill(std::begin(verified), std::end(verified), 0);
                }
                if (!validate_groups("draw", index, *pi->layout, rec.groups, rec.dynamic_offsets, arena_size, s, verified,
                                     targets))
                    return false;
                const bool indexed = bool(rec.index_buffer);
                for (uint32_t i = 0; i < pi->desc.vertex_layout_count; ++i) {
                    const VertexBinding& vb = rec.vertex[i];
                    auto* b = buffers.get(vb.buffer.id);
                    if (b == nullptr)
                        return fail(fmt("draw %zu: vertex buffer slot %u is missing, released or unknown", index, i));
                    if (!(b->info.usage & kVertex))
                        return fail(fmt("draw %zu: the buffer in vertex slot %u lacks kVertex", index, i));
                    if (vb.offset % 4 != 0)
                        return fail(fmt("draw %zu: vertex slot %u's offset %llu is not a multiple of 4", index, i,
                                        (unsigned long long)vb.offset));
                    b->last_use = s;
                    const VertexLayout& lay = pi->desc.vertex_layouts[i];
                    const uint64_t end = pi->layout_end[i];
                    if (end == 0) continue;
                    uint64_t need = 0;
                    if (lay.stride == 0) {
                        need = end;
                    } else if (lay.step == Step::Instance) {
                        const uint64_t n = uint64_t(rec.first_instance) + rec.instance_count;
                        need = n ? (n - 1) * lay.stride + end : 0;
                    } else if (!indexed) {
                        const uint64_t n = uint64_t(rec.first_vertex) + rec.count;
                        need = n ? (n - 1) * lay.stride + end : 0;
                    }
                    if (vb.offset > b->info.size || need > b->info.size - vb.offset)
                        return fail(fmt("draw %zu reads past the end of the buffer in vertex slot %u "
                                        "(%llu bytes from offset %llu of %llu)", index, i, (unsigned long long)need,
                                        (unsigned long long)vb.offset, (unsigned long long)b->info.size));
                }
                if (indexed) {
                    auto* b = buffers.get(rec.index_buffer.id);
                    if (b == nullptr || !(b->info.usage & kIndex))
                        return fail(fmt("draw %zu: the index buffer is released, unknown or lacks kIndex", index));
                    if (rec.index_offset % 4 != 0)
                        return fail(fmt("draw %zu: the index offset is not a multiple of 4", index));
                    if (rec.index_offset > b->info.size || uint64_t(rec.count) * 4 > b->info.size - rec.index_offset)
                        return fail(fmt("draw %zu reads past the end of its index buffer", index));
                    b->last_use = s;
                }
                ++index;
            }
            st->draws += uint32_t(list.records().size);
        }
    }
    return true;
}

// ---- Backend ----------------------------------------------------------------------------------

Backend::Backend(const Caps& caps) : core_(std::make_unique<Core>(caps)) {}
Backend::~Backend() = default;

const Caps& Backend::caps() const { return core_->caps; }
const std::string& Backend::last_error() const { return core_->error; }
bool Backend::refuse(std::string reason) { return core_->fail(std::move(reason)); }
void Backend::complete(Serial s, double gpu_ms) { core_->sink->complete(s, gpu_ms); }
std::shared_ptr<CompletionSink> Backend::completion_sink() const { return core_->sink; }

Buffer Backend::create_buffer(const BufferDesc& d) {
    poll();
    Core& c = *core_;
    if (d.size == 0 || d.size % 4 != 0 || d.size > c.caps.limits.max_buffer_size) {
        c.fail(fmt("a buffer of %llu bytes is refused (a multiple of 4, at most %llu)",
                   (unsigned long long)d.size, (unsigned long long)c.caps.limits.max_buffer_size));
        return {};
    }
    if (d.usage == 0 || (d.usage & ~63u) != 0) { c.fail("a buffer needs a known, non-empty usage"); return {}; }
    const uint64_t id = c.buffers.alloc();
    if (id == 0) { c.fail("out of buffer slots"); return {}; }
    auto* s = c.buffers.get(id);
    s->info.size = d.size;
    s->info.usage = d.usage;
    s->info.host_visible = d.host_visible;
    std::string err;
    if (!on_create_buffer(handle_slot(id), d, &err)) {
        c.buffers.recycle(handle_slot(id));
        c.fail("create_buffer: " + err);
        return {};
    }
    return Buffer{id};
}

Texture Backend::create_texture(const TextureDesc& d) {
    poll();
    Core& c = *core_;
    const uint32_t maxd = c.caps.limits.max_texture_dimension_2d;
    if (d.width == 0 || d.height == 0 || d.width > maxd || d.height > maxd) {
        c.fail(fmt("a %ux%u texture is refused (1 to %u on a side)", d.width, d.height, maxd));
        return {};
    }
    if (!valid_format(d.format) || d.usage == 0 || (d.usage & ~31u) != 0) {
        c.fail("a texture needs a known format and a known, non-empty usage");
        return {};
    }
    if (is_depth_format(d.format) && (d.usage & (kStorageTex | kCopyDstTex))) {
        c.fail("a depth texture cannot be a storage texture or a copy destination");
        return {};
    }
    if ((d.usage & kStorageTex) && !storage_format(d.format)) {
        c.fail("this format cannot be a storage texture");
        return {};
    }
    const uint64_t id = c.textures.alloc();
    if (id == 0) { c.fail("out of texture slots"); return {}; }
    auto* s = c.textures.get(id);
    s->info.width = d.width;
    s->info.height = d.height;
    s->info.format = d.format;
    s->info.usage = d.usage;
    std::string err;
    if (!on_create_texture(handle_slot(id), d, &err)) {
        c.textures.recycle(handle_slot(id));
        c.fail("create_texture: " + err);
        return {};
    }
    return Texture{id};
}

Sampler Backend::create_sampler(const SamplerDesc& d) {
    poll();
    Core& c = *core_;
    const uint64_t id = c.samplers.alloc();
    if (id == 0) { c.fail("out of sampler slots"); return {}; }
    c.samplers.get(id)->info.linear = d.linear;
    std::string err;
    if (!on_create_sampler(handle_slot(id), d, &err)) {
        c.samplers.recycle(handle_slot(id));
        c.fail("create_sampler: " + err);
        return {};
    }
    return Sampler{id};
}

Program Backend::create_program(const ProgramDesc& d) {
    poll();
    Core& c = *core_;
    const Limits& L = c.caps.limits;
    auto info = std::make_shared<ProgramInfo>();
    info->label = d.label ? d.label : "";
    if (d.entry_count == 0 || d.entries == nullptr) { c.fail("a program needs at least one entry point"); return {}; }
    for (uint32_t i = 0; i < d.entry_count; ++i) {
        const EntryDesc& e = d.entries[i];
        if (e.name == nullptr || e.name[0] == 0 ||
            (e.stage != kStageVertex && e.stage != kStageFragment && e.stage != kStageCompute)) {
            c.fail("a program's entry needs a name and exactly one stage");
            return {};
        }
        if (info->entry(e.name, e.stage) != nullptr) { c.fail(fmt("entry %s appears twice", e.name)); return {}; }
        EntryInfo ei;
        ei.name = e.name;
        ei.stage = e.stage;
        std::copy(std::begin(e.workgroup_size), std::end(e.workgroup_size), ei.workgroup_size);
        info->entries.push_back(ei);
    }
    if (d.binding_count > 0 && d.bindings == nullptr) { c.fail("a program's bindings are missing"); return {}; }
    uint32_t vertex_storage = 0, frag_storage = 0, comp_storage = 0, per_group[kMaxBindGroups] = {};
    for (uint32_t i = 0; i < d.binding_count; ++i) {
        const BindingDesc& b = d.bindings[i];
        const char* name = info->label.c_str();
        if (b.group >= kMaxBindGroups || b.group >= L.max_bind_groups) {
            c.fail(fmt("program %s: group %u is past the limit of %u bind groups", name, b.group, L.max_bind_groups));
            return {};
        }
        if (++per_group[b.group] > kMaxBindingsPerGroup) {
            c.fail(fmt("program %s: group %u has more than 8 bindings", name, b.group));
            return {};
        }
        if (info->binding(b.group, b.binding) != nullptr) {
            c.fail(fmt("program %s: binding %u of group %u appears twice", name, b.binding, b.group));
            return {};
        }
        if (b.stages == 0 || (b.stages & ~7u) != 0) {
            c.fail(fmt("program %s: binding %u of group %u needs a stage", name, b.binding, b.group));
            return {};
        }
        if (b.dynamic && b.type != BindingType::Uniform) {
            c.fail(fmt("program %s: only a uniform binding takes a dynamic offset", name));
            return {};
        }
        if (b.type == BindingType::Uniform && (b.size == 0 || b.size > L.max_uniform_buffer_binding_size)) {
            c.fail(fmt("program %s: a uniform block of %u bytes is past the limit of %u", name, b.size,
                       L.max_uniform_buffer_binding_size));
            return {};
        }
        if (b.type == BindingType::StorageReadWrite && (b.stages & kStageVertex)) {
            c.fail(fmt("program %s: a read-write storage buffer is not visible to the vertex stage", name));
            return {};
        }
        if (b.type == BindingType::Storage || b.type == BindingType::StorageReadWrite) {
            if (b.size == 0 || b.size % 4 != 0 || b.size > L.max_storage_buffer_binding_size) {
                c.fail(fmt("program %s: a storage block of %u bytes is refused", name, b.size));
                return {};
            }
            if (b.stages & kStageVertex) ++vertex_storage;
            if (b.stages & kStageFragment) ++frag_storage;
            if (b.stages & kStageCompute) ++comp_storage;
        }
        if (b.type == BindingType::StorageTexture && ((b.stages & kStageVertex) || !storage_format(b.storage_format))) {
            c.fail(fmt("program %s: a storage texture is not visible to the vertex stage and needs a storage format", name));
            return {};
        }
        if (is_buffer_binding(b.type) && b.msl_index >= kMetalSizesIndex) {
            c.fail(fmt("program %s: a buffer's MSL index %u collides with the sizes buffer (15) or the vertex "
                       "buffers (16 and up)", name, b.msl_index));
            return {};
        }
        info->bindings.push_back(b);
    }
    if (vertex_storage > L.max_storage_buffers_in_vertex_stage) {
        c.fail(fmt("program %s: %u storage buffers in the vertex stage, the limit is %u", info->label.c_str(),
                   vertex_storage, L.max_storage_buffers_in_vertex_stage));
        return {};
    }
    if (std::max({vertex_storage, frag_storage, comp_storage}) > L.max_storage_buffers_per_stage) {
        c.fail(fmt("program %s: too many storage buffers in one stage", info->label.c_str()));
        return {};
    }
    std::sort(info->bindings.begin(), info->bindings.end(), [](const BindingDesc& a, const BindingDesc& b) {
        return a.group != b.group ? a.group < b.group : a.binding < b.binding;
    });
    Hasher gh[kMaxBindGroups];
    for (const BindingDesc& b : info->bindings) {
        info->group_mask |= uint8_t(1u << b.group);
        Hasher& h = gh[b.group];
        h.add(b.binding); h.add(b.type); h.add(b.stages); h.add(b.dynamic); h.add(b.size);
        h.add(b.storage_format); h.add(b.msl_index);
        if (b.dynamic) info->group_dynamic[b.group]++;
    }
    uint8_t base = 0;
    for (uint32_t g = 0; g < kMaxBindGroups; ++g) {
        info->group_hash[g] = gh[g].h ^ g;
        info->dynamic_base[g] = base;
        base = uint8_t(base + info->group_dynamic[g]);
    }
    info->dynamic_count = base;
    if (base > kMaxDynamicOffsets || base > L.max_dynamic_uniform_buffers) {
        c.fail(fmt("program %s: %u dynamic uniform bindings, at most %u", info->label.c_str(), base, kMaxDynamicOffsets));
        return {};
    }
    const uint64_t id = c.programs.alloc();
    if (id == 0) { c.fail("out of program slots"); return {}; }
    c.programs.get(id)->info = info;
    std::string err;
    if (!on_create_program(handle_slot(id), d, &err)) {
        c.programs.recycle(handle_slot(id));
        c.fail("create_program " + info->label + ": " + err);
        return {};
    }
    return Program{id};
}

namespace {

uint64_t hash_pipeline(const RenderPipelineDesc& d) {
    Hasher h;
    h.add(d.program.id);
    h.str(d.vertex_entry);
    h.str(d.fragment_entry);
    h.add(d.vertex_layout_count);
    for (uint32_t i = 0; i < d.vertex_layout_count; ++i) {
        const VertexLayout& l = d.vertex_layouts[i];
        h.add(l.stride); h.add(l.step); h.add(l.attribute_count);
        for (uint32_t a = 0; a < l.attribute_count; ++a) {
            h.add(l.attributes[a].location); h.add(l.attributes[a].format); h.add(l.attributes[a].offset);
        }
    }
    h.add(d.topology); h.add(d.color_format); h.add(d.blend); h.add(d.id_target);
    h.add(d.depth); h.add(d.depth_test); h.add(d.depth_write); h.add(d.constant_count);
    for (uint32_t i = 0; i < d.constant_count; ++i) { h.add(d.constants[i].id); h.add(d.constants[i].value); }
    return h.h;
}

bool same_pipeline(const RenderPipelineDesc& a, const PipelineInfo& p) {
    const RenderPipelineDesc& b = p.desc;
    if (a.program != b.program || p.vertex_entry != a.vertex_entry || p.fragment_entry != a.fragment_entry ||
        a.vertex_layout_count != b.vertex_layout_count || a.topology != b.topology ||
        a.color_format != b.color_format || a.blend != b.blend || a.id_target != b.id_target ||
        a.depth != b.depth || a.depth_test != b.depth_test || a.depth_write != b.depth_write ||
        a.constant_count != b.constant_count)
        return false;
    for (uint32_t i = 0; i < a.vertex_layout_count; ++i) {
        const VertexLayout& x = a.vertex_layouts[i];
        const VertexLayout& y = b.vertex_layouts[i];
        if (x.stride != y.stride || x.step != y.step || x.attribute_count != y.attribute_count) return false;
        for (uint32_t k = 0; k < x.attribute_count; ++k)
            if (x.attributes[k].location != y.attributes[k].location || x.attributes[k].format != y.attributes[k].format ||
                x.attributes[k].offset != y.attributes[k].offset)
                return false;
    }
    for (uint32_t i = 0; i < a.constant_count; ++i)
        if (a.constants[i].id != b.constants[i].id || a.constants[i].value != b.constants[i].value) return false;
    return true;
}

}  // namespace

Pipeline Backend::create_render_pipeline(const RenderPipelineDesc& d) {
    poll();
    Core& c = *core_;
    const Limits& L = c.caps.limits;
    auto* ps = c.programs.get(d.program.id);
    if (ps == nullptr) { c.fail("create_render_pipeline: the program is released or unknown"); return {}; }
    std::shared_ptr<const ProgramInfo> prog = ps->info;
    if (prog->entry(d.vertex_entry, kStageVertex) == nullptr ||
        prog->entry(d.fragment_entry, kStageFragment) == nullptr) {
        c.fail(fmt("create_render_pipeline: program %s has no vertex entry %s or fragment entry %s",
                   prog->label.c_str(), d.vertex_entry ? d.vertex_entry : "(none)",
                   d.fragment_entry ? d.fragment_entry : "(none)"));
        return {};
    }
    if (d.vertex_layout_count > kMaxVertexBuffers || d.vertex_layout_count > L.max_vertex_buffers) {
        c.fail(fmt("create_render_pipeline: %u vertex buffers, the limit is %u", d.vertex_layout_count,
                   std::min(kMaxVertexBuffers, L.max_vertex_buffers)));
        return {};
    }
    PipelineInfo info;
    uint32_t attributes = 0;
    uint32_t locations_seen[2] = {};   // max_vertex_attributes is at most 64 here
    for (uint32_t i = 0; i < d.vertex_layout_count; ++i) {
        const VertexLayout& lay = d.vertex_layouts[i];
        if (lay.stride % 4 != 0 || lay.stride > L.max_vertex_stride ||
            (lay.step != Step::Vertex && lay.step != Step::Instance)) {
            c.fail(fmt("create_render_pipeline: layout %u's stride %u must be a multiple of 4 up to %u", i,
                       lay.stride, L.max_vertex_stride));
            return {};
        }
        if (lay.attribute_count > kMaxAttributesPerBuffer) {
            c.fail(fmt("create_render_pipeline: layout %u has more than 4 attributes", i));
            return {};
        }
        for (uint32_t a = 0; a < lay.attribute_count; ++a) {
            const VertexAttribute& at = lay.attributes[a];
            const uint32_t sz = vertex_format_bytes(at.format);
            const uint32_t limit = lay.stride ? lay.stride : L.max_vertex_stride;
            if (sz == 0 || at.offset % std::min(4u, sz) != 0 || at.offset + sz > limit) {
                c.fail(fmt("create_render_pipeline: attribute at location %u sits outside its element or is misaligned",
                           at.location));
                return {};
            }
            if (at.location >= L.max_vertex_attributes || at.location >= 64 ||
                (locations_seen[at.location / 32] & (1u << (at.location % 32)))) {
                c.fail(fmt("create_render_pipeline: location %u is past the limit or used twice", at.location));
                return {};
            }
            locations_seen[at.location / 32] |= 1u << (at.location % 32);
            info.layout_end[i] = std::max(info.layout_end[i], at.offset + sz);
            ++attributes;
        }
    }
    if (attributes > L.max_vertex_attributes) {
        c.fail(fmt("create_render_pipeline: %u attributes, the limit is %u", attributes, L.max_vertex_attributes));
        return {};
    }
    if (!valid_format(d.color_format) || is_depth_format(d.color_format) ||
        uint32_t(d.topology) > uint32_t(Topology::PointList) || uint32_t(d.blend) > uint32_t(Blend::Additive)) {
        c.fail("create_render_pipeline: a colour format, topology or blend mode is unknown");
        return {};
    }
    if (d.blend != Blend::Opaque && !blendable(d.color_format)) {
        c.fail("create_render_pipeline: this colour format does not blend; use Blend::Opaque");
        return {};
    }
    if ((d.id_target ? 2u : 1u) > L.max_color_attachments) {
        c.fail("create_render_pipeline: too many colour attachments");
        return {};
    }
    if ((d.depth_test || d.depth_write) && !d.depth) {
        c.fail("create_render_pipeline: a depth test or write needs depth");
        return {};
    }
    if (d.constant_count != 0) {   // decision 16
        c.fail("create_render_pipeline: override constants are refused, because Naga's MSL cannot declare "
               "function constants; a value that varies belongs in a uniform block");
        return {};
    }

    const uint64_t h = hash_pipeline(d);
    auto range = c.pipeline_cache.equal_range(h);
    for (auto it = range.first; it != range.second; ++it) {
        auto* p = c.pipelines.get(it->second);
        if (p != nullptr && same_pipeline(d, p->info)) {
            p->info.refs++;
            return Pipeline{it->second};
        }
    }
    const uint64_t id = c.pipelines.alloc();
    if (id == 0) { c.fail("out of pipeline slots"); return {}; }
    info.program = d.program;
    info.layout = prog;
    info.desc = d;
    info.desc.vertex_entry = nullptr;
    info.desc.fragment_entry = nullptr;
    info.desc.label = nullptr;
    info.vertex_entry = d.vertex_entry;
    info.fragment_entry = d.fragment_entry;
    info.hash = h;
    c.pipelines.get(id)->info = std::move(info);
    std::string err;
    if (!on_create_render_pipeline(handle_slot(id), d, &err)) {
        c.pipelines.recycle(handle_slot(id));
        c.fail("create_render_pipeline: " + err);
        return {};
    }
    c.pipeline_cache.emplace(h, id);
    return Pipeline{id};
}

Pipeline Backend::create_compute_pipeline(const ComputePipelineDesc& d) {
    poll();
    Core& c = *core_;
    auto* ps = c.programs.get(d.program.id);
    if (ps == nullptr) { c.fail("create_compute_pipeline: the program is released or unknown"); return {}; }
    if (ps->info->entry(d.entry, kStageCompute) == nullptr) {
        c.fail("create_compute_pipeline: the program has no such compute entry");
        return {};
    }
    const uint64_t id = c.pipelines.alloc();
    if (id == 0) { c.fail("out of pipeline slots"); return {}; }
    PipelineInfo info;
    info.compute = true;
    info.program = d.program;
    info.layout = ps->info;
    info.compute_entry = d.entry;
    c.pipelines.get(id)->info = std::move(info);
    std::string err;
    if (!on_create_compute_pipeline(handle_slot(id), d, &err)) {
        c.pipelines.recycle(handle_slot(id));
        c.fail("create_compute_pipeline: " + err);
        return {};
    }
    return Pipeline{id};
}

BindGroup Backend::create_bind_group(const BindGroupDesc& d) {
    poll();
    Core& c = *core_;
    const Limits& L = c.caps.limits;
    auto* ps = c.programs.get(d.program.id);
    if (ps == nullptr) { c.fail("create_bind_group: the program is released or unknown"); return {}; }
    const ProgramInfo& P = *ps->info;
    if (d.group >= kMaxBindGroups || (P.group_mask & (1u << d.group)) == 0) {
        c.fail(fmt("create_bind_group: program %s declares no group %u", P.label.c_str(), d.group));
        return {};
    }
    if (d.entry_count > kMaxBindingsPerGroup) { c.fail("create_bind_group: at most 8 entries"); return {}; }
    BindGroupInfo info;
    info.program = d.program;
    info.group = d.group;
    info.layout_hash = P.group_hash[d.group];
    uint32_t matched = 0;
    for (const BindingDesc& b : P.bindings) {
        if (b.group != d.group) continue;
        const BindEntry* e = nullptr;
        for (uint32_t i = 0; i < d.entry_count; ++i)
            if (d.entries[i].binding == b.binding) {
                if (e != nullptr) { c.fail(fmt("create_bind_group: binding %u appears twice", b.binding)); return {}; }
                e = &d.entries[i];
            }
        if (e == nullptr) {
            c.fail(fmt("create_bind_group: binding %u of group %u has no entry", b.binding, d.group));
            return {};
        }
        ++matched;
        BindEntry r = *e;
        if (is_buffer_binding(b.type)) {
            const bool uniform = b.type == BindingType::Uniform;
            if (r.size == 0) {   // a uniform's block, or a storage binding's whole remaining buffer
                auto* bs = r.buffer == kArena ? nullptr : c.buffers.get(r.buffer.id);
                r.size = uniform || bs == nullptr || r.offset > bs->info.size ? b.size : bs->info.size - r.offset;
            }
            if (r.size < b.size) {
                c.fail(fmt("create_bind_group: binding %u's range of %llu bytes is smaller than its block of %u",
                           b.binding, (unsigned long long)r.size, b.size));
                return {};
            }
            const uint64_t max = uniform ? L.max_uniform_buffer_binding_size : L.max_storage_buffer_binding_size;
            if (r.size > max) {
                c.fail(fmt("create_bind_group: binding %u is %llu bytes, past the limit of %llu", b.binding,
                           (unsigned long long)r.size, (unsigned long long)max));
                return {};
            }
            if (r.buffer == kArena) {
                if (!b.dynamic || r.offset != 0) {
                    c.fail(fmt("create_bind_group: binding %u: only a dynamic uniform binding at offset 0 takes the arena",
                               b.binding));
                    return {};
                }
            } else {
                auto* bs = c.buffers.get(r.buffer.id);
                if (bs == nullptr) {
                    c.fail(fmt("create_bind_group: binding %u's buffer is released or unknown", b.binding));
                    return {};
                }
                if (b.type == BindingType::StorageReadWrite && bs->info.read_only) {
                    c.fail(fmt("create_bind_group: binding %u is read_write, and its buffer is a read-only import",
                               b.binding));
                    return {};
                }
                if (!(bs->info.usage & (uniform ? kUniform : kStorage))) {
                    c.fail(fmt("create_bind_group: binding %u's buffer lacks %s", b.binding,
                               uniform ? "kUniform" : "kStorage"));
                    return {};
                }
                const uint32_t align = uniform ? L.min_uniform_buffer_offset_alignment
                                               : L.min_storage_buffer_offset_alignment;
                if (r.offset % align != 0) {
                    c.fail(fmt("create_bind_group: binding %u's offset %llu is not a multiple of %u", b.binding,
                               (unsigned long long)r.offset, align));
                    return {};
                }
                if (r.offset > bs->info.size || r.size > bs->info.size - r.offset) {
                    c.fail(fmt("create_bind_group: binding %u's range is past the end of its buffer", b.binding));
                    return {};
                }
            }
        } else if (b.type == BindingType::Sampler) {
            if (c.samplers.get(r.sampler.id) == nullptr) {
                c.fail(fmt("create_bind_group: binding %u's sampler is released or unknown", b.binding));
                return {};
            }
        } else {
            auto* ts = c.textures.get(r.texture.id);
            if (ts == nullptr) {
                c.fail(fmt("create_bind_group: binding %u's texture is released or unknown", b.binding));
                return {};
            }
            const TextureInfo& t = ts->info;
            bool ok = true;
            if (b.type == BindingType::Texture) ok = (t.usage & kSampled) && filterable(t.format);
            if (b.type == BindingType::UintTexture) ok = (t.usage & kSampled) && is_integer_format(t.format);
            if (b.type == BindingType::StorageTexture) ok = (t.usage & kStorageTex) && t.format == b.storage_format;
            if (!ok) {
                c.fail(fmt("create_bind_group: binding %u's texture has the wrong format or usage", b.binding));
                return {};
            }
        }
        info.entries.push_back(r);
        info.layout.push_back(b);
    }
    if (matched != d.entry_count) {
        c.fail(fmt("create_bind_group: an entry names a binding that group %u does not declare", d.group));
        return {};
    }
    // The hook sees every range resolved, in the program's binding order.
    BindGroupDesc resolved = d;
    resolved.entry_count = uint8_t(info.entries.size());
    for (size_t i = 0; i < info.entries.size(); ++i) resolved.entries[i] = info.entries[i];
    const uint64_t id = c.groups.alloc();
    if (id == 0) { c.fail("out of bind group slots"); return {}; }
    c.groups.get(id)->info = std::move(info);
    std::string err;
    if (!on_create_bind_group(handle_slot(id), resolved, &err)) {
        c.groups.recycle(handle_slot(id));
        c.fail("create_bind_group: " + err);
        return {};
    }
    return BindGroup{id};
}

void Backend::release(uint64_t id) {
    Core& c = *core_;
    const uint32_t slot = handle_slot(id);
    Serial last = 0;
    switch (handle_kind(id)) {
        case HandleKind::Buffer: {
            auto* s = c.buffers.get(id);
            if (s == nullptr) break;
            last = s->last_use; c.buffers.kill(slot);
            c.deferred.push_back({id, last});
            poll();
            return;
        }
        case HandleKind::Texture: {
            auto* s = c.textures.get(id);
            if (s == nullptr) break;
            last = s->last_use; c.textures.kill(slot);
            c.deferred.push_back({id, last});
            poll();
            return;
        }
        case HandleKind::Sampler: {
            auto* s = c.samplers.get(id);
            if (s == nullptr) break;
            last = s->last_use; c.samplers.kill(slot);
            c.deferred.push_back({id, last});
            poll();
            return;
        }
        case HandleKind::Program: {
            auto* s = c.programs.get(id);
            if (s == nullptr) break;
            c.programs.kill(slot);
            c.deferred.push_back({id, 0});   // its pipelines hold what they need
            poll();
            return;
        }
        case HandleKind::Pipeline: {
            auto* s = c.pipelines.get(id);
            if (s == nullptr) break;
            if (--s->info.refs > 0) return;
            if (!s->info.compute) {
                auto range = c.pipeline_cache.equal_range(s->info.hash);
                for (auto it = range.first; it != range.second; ++it)
                    if (it->second == id) { c.pipeline_cache.erase(it); break; }
            }
            last = s->last_use; c.pipelines.kill(slot);
            c.deferred.push_back({id, last});
            poll();
            return;
        }
        case HandleKind::BindGroup: {
            auto* s = c.groups.get(id);
            if (s == nullptr) break;
            last = s->last_use; c.groups.kill(slot);
            c.deferred.push_back({id, last});
            poll();
            return;
        }
        case HandleKind::Readback: {
            auto* s = c.readbacks.get(id);
            if (s == nullptr) break;
            last = s->info.serial; c.readbacks.kill(slot);
            c.deferred.push_back({id, last});
            poll();
            return;
        }
        case HandleKind::None: break;
    }
    c.fail("release: the handle is stale or unknown");
}

void Backend::poll() {
    Core& c = *core_;
    const Serial through = c.through();
    for (size_t i = 0; i < c.deferred.size();) {
        const Core::Deferred d = c.deferred[i];
        if (d.serial > through) { ++i; continue; }
        c.deferred[i] = c.deferred.back();
        c.deferred.pop_back();
        on_destroy(d.id);
        const uint32_t slot = handle_slot(d.id);
        switch (handle_kind(d.id)) {
            case HandleKind::Buffer: c.buffers.recycle(slot); break;
            case HandleKind::Texture: c.textures.recycle(slot); break;
            case HandleKind::Sampler: c.samplers.recycle(slot); break;
            case HandleKind::Program: c.programs.recycle(slot); break;
            case HandleKind::Pipeline: c.pipelines.recycle(slot); break;
            case HandleKind::BindGroup: c.groups.recycle(slot); break;
            case HandleKind::Readback: c.readbacks.recycle(slot); break;
            case HandleKind::None: break;
        }
    }
}

bool Backend::write_buffer(Buffer b, uint64_t offset, const void* data, uint64_t size) {
    Core& c = *core_;
    auto* s = c.buffers.get(b.id);
    if (s == nullptr) return c.fail("write_buffer: the buffer is released or unknown");
    if (!(s->info.usage & kCopyDst)) return c.fail("write_buffer: the buffer lacks kCopyDst");
    if (data == nullptr || offset % 4 != 0 || size % 4 != 0)
        return c.fail("write_buffer: the offset and size must be multiples of 4");
    if (offset > s->info.size || size > s->info.size - offset)
        return c.fail("write_buffer: the write is past the end of the buffer");
    if (size == 0) return true;
    on_write_buffer(handle_slot(b.id), offset, data, size, s->last_use > c.through());
    c.stats.copies++;
    c.stats.bytes_copied_from_host += size;
    return true;
}

bool Backend::write_texture(Texture t, const void* data, uint32_t row_pitch) {
    Core& c = *core_;
    auto* s = c.textures.get(t.id);
    if (s == nullptr) return c.fail("write_texture: the texture is released or unknown");
    if (!(s->info.usage & kCopyDstTex)) return c.fail("write_texture: the texture lacks kCopyDstTex");
    const uint64_t row = uint64_t(s->info.width) * format_bytes(s->info.format);
    if (data == nullptr || row_pitch < row) return c.fail("write_texture: the row pitch is smaller than a row");
    on_write_texture(handle_slot(t.id), data, row_pitch, s->last_use > c.through());
    c.stats.copies++;
    c.stats.bytes_copied_from_host += row * s->info.height;
    return true;
}

void* Backend::mapped(Buffer b) {
    Core& c = *core_;
    auto* s = c.buffers.get(b.id);
    if (s == nullptr || !(s->info.host_visible || s->info.imported) || !c.caps.has(kCapPersistentMap)) return nullptr;
    return on_mapped(handle_slot(b.id));
}

namespace {
ImportResult refused(Core& c, std::string reason) {
    ImportResult r;
    r.reason = std::move(reason);
    c.error = "import: " + r.reason;
    return r;
}
}  // namespace

ImportResult Backend::import_buffer(const ExternalBuffer& ext) {
    poll();
    Core& c = *core_;
    const Limits& L = c.caps.limits;
    if (ext.handle == nullptr) return refused(c, "no handle");
    if (ext.size == 0 || ext.size % 4 != 0 || ext.size > L.max_buffer_size)
        return refused(c, fmt("%llu bytes is above the backend's limits (a multiple of 4, at most %llu)",
                              (unsigned long long)ext.size, (unsigned long long)L.max_buffer_size));
    if (ext.usage == 0 || (ext.usage & ~uint32_t(kVertex | kIndex | kStorage | kCopySrc | kUniform)) != 0)
        return refused(c, "an import's usage is kVertex, kIndex, kStorage, kUniform or kCopySrc");
    if (ext.kind == ExternalKind::MetalBuffer && !c.caps.has(kCapImportMetalBuffer))
        return refused(c, fmt("the %s backend cannot import an MTLBuffer", c.caps.backend));
    if (ext.kind == ExternalKind::Win32Memory && !c.caps.has(kCapImportWin32Memory))
        return refused(c, fmt("the %s backend cannot import Win32 memory", c.caps.backend));
    if (ext.kind == ExternalKind::MetalTexture) return refused(c, "a Metal texture is imported with import_texture");
    const uint64_t id = c.buffers.alloc();
    if (id == 0) return refused(c, "out of buffer slots");
    auto* s = c.buffers.get(id);
    s->info.size = ext.size;
    s->info.usage = ext.usage;
    s->info.imported = true;
    s->info.read_only = ext.read_only;
    ImportResult r;
    const uint32_t slot = handle_slot(id);
    if (!on_import_buffer(slot, ext, &r) || r.route == Route::Refused) {
        c.buffers.recycle(slot);
        return refused(c, r.reason.empty() ? std::string("the backend refused it") : r.reason);
    }
    if (!ext.allow_copy && r.route != Route::InPlace) {
        on_destroy(id);
        c.buffers.recycle(slot);
        return refused(c, "it can only be copied here, and the caller refused a copy" +
                              (r.reason.empty() ? std::string() : " (" + r.reason + ")"));
    }
    s = c.buffers.get(id);
    s->info.route = r.route;
    r.buffer = Buffer{id};
    r.path = route_name(r.route);
    if (r.route == Route::InPlace) {
        c.stats.imports_in_place++;
        r.bytes_copied = 0;
    } else {
        c.stats.imports_copied++;
        c.stats.copies++;
        if (r.route == Route::CopiedFromHost) c.stats.bytes_copied_from_host += r.bytes_copied;
        else c.stats.bytes_copied_on_gpu += r.bytes_copied;
    }
    return r;
}

ImportResult Backend::import_texture(const ExternalTexture& ext) {
    poll();
    Core& c = *core_;
    const uint32_t maxd = c.caps.limits.max_texture_dimension_2d;
    if (ext.handle == nullptr) return refused(c, "no handle");
    if (ext.kind != ExternalKind::MetalTexture || !c.caps.has(kCapImportMetalTexture))
        return refused(c, fmt("the %s backend cannot import this texture", c.caps.backend));
    if (ext.width == 0 || ext.height == 0 || ext.width > maxd || ext.height > maxd || !valid_format(ext.format) ||
        ext.usage == 0 || (ext.usage & ~31u) != 0)
        return refused(c, "the texture's size, format or usage is outside the backend's limits");
    const uint64_t id = c.textures.alloc();
    if (id == 0) return refused(c, "out of texture slots");
    auto* s = c.textures.get(id);
    s->info.width = ext.width;
    s->info.height = ext.height;
    s->info.format = ext.format;
    s->info.usage = ext.usage;
    s->info.imported = true;
    ImportResult r;
    const uint32_t slot = handle_slot(id);
    if (!on_import_texture(slot, ext, &r) || r.route != Route::InPlace) {
        if (r.route != Route::Refused) on_destroy(id);
        c.textures.recycle(slot);
        return refused(c, r.reason.empty() ? std::string("the backend refused it") : r.reason);
    }
    r.texture = Texture{id};
    r.path = route_name(r.route);
    c.stats.imports_in_place++;
    return r;
}

Serial Backend::submit(Frame&& frame) {
    poll();
    Core& c = *core_;
    const auto t0 = std::chrono::steady_clock::now();
    const Serial s = c.next_serial;
    Stats st;
    if (!c.validate_frame(frame, s, &st)) return 0;
    std::string err;
    if (!on_submit(s, frame, &st, &err)) {
        c.fail("submit: " + err);
        return 0;
    }
    c.next_serial++;
    const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    c.stats.submits++;
    c.stats.draws += st.draws;
    c.stats.dispatches += st.dispatches;
    c.stats.copies += st.copies;
    c.stats.bytes_copied_on_gpu += st.bytes_copied_on_gpu;
    c.stats.bytes_copied_from_host += st.bytes_copied_from_host;
    c.stats.encode_cpu_ms += ms;
    return s;
}

bool Backend::completed(Serial s) const { return s <= core_->through(); }
bool Backend::submit_gpu_ms(Serial s, double* gpu_ms) const {
    if (s == 0 || !completed(s)) return false;
    if (core_->sink->submit_gpu_ms(s, gpu_ms)) return true;
    // Without timestamps no time comes; with them, the completion handler may still give it.
    if (core_->caps.has(kCapTimestamps)) return false;
    if (gpu_ms) *gpu_ms = -1;
    return true;
}

bool Backend::wait(Serial s) {
    Core& c = *core_;
    if (completed(s)) return true;
    if (!c.caps.has(kCapBlockingWait) || s >= c.next_serial) return false;
    if (!on_wait(s)) return false;
    poll();
    return completed(s);
}

Serial Backend::last_use(uint64_t id) const {
    const Core& c = *core_;
    switch (handle_kind(id)) {
        case HandleKind::Buffer: { auto* s = c.buffers.any(id); return s ? s->last_use : 0; }
        case HandleKind::Texture: { auto* s = c.textures.any(id); return s ? s->last_use : 0; }
        case HandleKind::Sampler: { auto* s = c.samplers.any(id); return s ? s->last_use : 0; }
        case HandleKind::Pipeline: { auto* s = c.pipelines.any(id); return s ? s->last_use : 0; }
        case HandleKind::BindGroup: { auto* s = c.groups.any(id); return s ? s->last_use : 0; }
        case HandleKind::Readback: { auto* s = c.readbacks.any(id); return s ? s->info.serial : 0; }
        default: return 0;
    }
}

bool Backend::alive(uint64_t id) const {
    const Core& c = *core_;
    switch (handle_kind(id)) {
        case HandleKind::Buffer: return c.buffers.get(id) != nullptr;
        case HandleKind::Texture: return c.textures.get(id) != nullptr;
        case HandleKind::Sampler: return c.samplers.get(id) != nullptr;
        case HandleKind::Program: return c.programs.get(id) != nullptr;
        case HandleKind::Pipeline: return c.pipelines.get(id) != nullptr;
        case HandleKind::BindGroup: return c.groups.get(id) != nullptr;
        case HandleKind::Readback: return c.readbacks.get(id) != nullptr;
        default: return false;
    }
}

Readback Backend::read_texture(Texture t, uint32_t x, uint32_t y, uint32_t w, uint32_t h) {
    poll();
    Core& c = *core_;
    auto* ts = c.textures.get(t.id);
    if (ts == nullptr) { c.fail("read_texture: the texture is released or unknown"); return {}; }
    if (!(ts->info.usage & kCopySrcTex)) { c.fail("read_texture: the texture lacks kCopySrcTex"); return {}; }
    if (w == 0 || h == 0 || uint64_t(x) + w > ts->info.width || uint64_t(y) + h > ts->info.height) {
        c.fail("read_texture: the region is empty or outside the texture");
        return {};
    }
    const uint64_t id = c.readbacks.alloc();
    if (id == 0) { c.fail("out of readback slots"); return {}; }
    const Serial s = c.next_serial;
    auto* rs = c.readbacks.get(id);
    rs->info.serial = s;
    rs->info.format = ts->info.format;
    rs->info.w = w;
    rs->info.h = h;
    std::string err;
    if (!on_read_texture(s, handle_slot(id), handle_slot(t.id), x, y, w, h, &err)) {
        c.readbacks.recycle(handle_slot(id));
        c.fail("read_texture: " + err);
        return {};
    }
    c.next_serial++;
    ts->last_use = s;
    return Readback{id};
}

bool Backend::take_readback(Readback r, std::vector<uint8_t>* out) {
    Core& c = *core_;
    auto* s = c.readbacks.get(r.id);
    if (s == nullptr) return c.fail("take_readback: the readback is taken, released or unknown");
    if (!completed(s->info.serial)) return false;
    if (out) on_take_readback(handle_slot(r.id), out);
    on_destroy(r.id);
    c.readbacks.recycle(handle_slot(r.id));
    return true;
}

uint64_t Backend::imgui_texture_id(Texture t) {
    auto* s = core_->textures.get(t.id);
    return s ? on_imgui_texture_id(handle_slot(t.id)) : 0;
}

Stats Backend::take_stats() {
    Core& c = *core_;
    Stats r = c.stats;
    r.gpu_ms = c.caps.has(kCapTimestamps) ? c.sink->take_gpu_ms(&r.gpu_submits) : -1.0;
    c.stats = Stats{};
    return r;
}

// ---- Accessors for the hooks -------------------------------------------------------------------

namespace {
template <class T> const T& or_empty(const T* p) {
    static const T empty{};
    return p ? *p : empty;
}
}  // namespace

const ProgramInfo& Backend::program_info(Program p) const {
    auto* s = core_->programs.any(p.id);
    return or_empty<ProgramInfo>(s ? s->info.get() : nullptr);
}
const PipelineInfo& Backend::pipeline_info(Pipeline p) const {
    auto* s = core_->pipelines.any(p.id);
    return or_empty<PipelineInfo>(s ? &s->info : nullptr);
}
const BindGroupInfo& Backend::bind_group_info(BindGroup g) const {
    auto* s = core_->groups.any(g.id);
    return or_empty<BindGroupInfo>(s ? &s->info : nullptr);
}
const BufferInfo& Backend::buffer_info(Buffer b) const {
    auto* s = core_->buffers.any(b.id);
    return or_empty<BufferInfo>(s ? &s->info : nullptr);
}
const TextureInfo& Backend::texture_info(Texture t) const {
    auto* s = core_->textures.any(t.id);
    return or_empty<TextureInfo>(s ? &s->info : nullptr);
}

}  // namespace caliper_gfx
