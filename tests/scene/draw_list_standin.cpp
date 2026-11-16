// draw_list_standin.cpp: the out-of-line members of caliper_gfx::DrawList that the scene core
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

}  // namespace caliper_gfx
