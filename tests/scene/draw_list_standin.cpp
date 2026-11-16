// draw_list_standin.cpp: the out-of-line members of caliper_gfx::DrawList and Frame that the scene core
// calls, for the scene tests while the graphics layer's core (src/host/gfx/core.cpp, the target
// caliper_gfx) is not on this branch. tests/CMakeLists.txt compiles this file only when that
// target is absent; the behaviour follows the header's comments.
#include "gfx/gfx.h"

#include <cstring>

namespace caliper_gfx {

uint32_t DrawList::uniform(const void* data, uint32_t size) {
    if (size == 0 || size > 65536) {
        error_ = "a uniform block is 1 byte to 64 KiB";
        return kNoOffset;
    }
    const size_t offset = (arena_.size() + kUniformAlign - 1) / kUniformAlign * kUniformAlign;
    arena_.resize(offset + size, 0);
    std::memcpy(arena_.data() + offset, data, size);
    return uint32_t(offset);
}

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

namespace {

uint32_t arena_copy(std::vector<uint8_t>& arena, const void* data, uint32_t size) {
    if (size == 0 || size > 65536) return kNoOffset;
    const size_t offset = (arena.size() + kUniformAlign - 1) / kUniformAlign * kUniformAlign;
    arena.resize(offset + size, 0);
    std::memcpy(arena.data() + offset, data, size);
    return uint32_t(offset);
}

}  // namespace

uint32_t Frame::uniform(const void* data, uint32_t size) {
    const uint32_t off = arena_copy(arena_, data, size);
    if (off == kNoOffset) error_ = "a uniform block is 1 byte to 64 KiB";
    return off;
}

void Frame::dispatch(Pipeline p, View<BindGroup> groups, View<uint32_t> offsets, uint32_t x, uint32_t y,
                     uint32_t z, View<Texture> writes) {
    DispatchPass d;
    d.pipeline = p;
    for (size_t i = 0; i < groups.size && i < kMaxBindGroups; ++i) d.groups[i] = groups[i];
    for (size_t i = 0; i < offsets.size && i < kMaxDynamicOffsets; ++i) d.dynamic_offsets[i] = offsets[i];
    d.x = x;
    d.y = y;
    d.z = z;
    d.writes.assign(writes.begin(), writes.end());
    passes_.push_back(std::move(d));
}

void Frame::render(DrawList&& list) { passes_.push_back(RenderPass{std::move(list)}); }

}  // namespace caliper_gfx
