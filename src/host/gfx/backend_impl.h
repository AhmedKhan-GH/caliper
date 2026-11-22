// backend_impl.h: what a backend of the graphics layer needs beyond gfx.h. Clients include
// gfx.h only. The core (core.cpp) keeps one record per live handle, validates every call
// against it, and hands the backends these records through Backend's protected accessors.
#pragma once

#include "gfx/gfx.h"

#include <array>
#include <atomic>
#include <memory>
#include <mutex>
#include <set>
#include <string>
#include <vector>

namespace caliper_gfx {

struct EntryInfo {
    std::string name;
    uint8_t stage = 0;
    uint16_t workgroup_size[3] = {1, 1, 1};
};

struct ProgramInfo {
    std::string label;
    std::vector<EntryInfo> entries;
    std::vector<BindingDesc> bindings;              // sorted by (group, binding)
    uint8_t group_mask = 0;                         // the groups the program declares
    uint64_t group_hash[kMaxBindGroups] = {};       // a group's layout, for compatibility
    uint8_t group_dynamic[kMaxBindGroups] = {};     // dynamic bindings in each group
    uint8_t dynamic_base[kMaxBindGroups] = {};      // index of group g's first dynamic offset
    uint8_t dynamic_count = 0;

    const EntryInfo* entry(const char* name, uint8_t stage) const;
    const BindingDesc* binding(uint32_t group, uint32_t binding) const;
};

struct PipelineInfo {
    bool compute = false;
    Program program;
    std::shared_ptr<const ProgramInfo> layout;      // outlives the program's handle
    RenderPipelineDesc desc;                        // entry and label pointers cleared
    std::string vertex_entry, fragment_entry, compute_entry;
    uint32_t layout_end[kMaxVertexBuffers] = {};    // bytes an element of layout i reads
    uint64_t hash = 0;
    uint32_t refs = 1;
};

struct BindGroupInfo {
    Program program;
    uint32_t group = 0;
    uint64_t layout_hash = 0;
    std::vector<BindEntry> entries;                 // sorted by binding, sizes resolved
    std::vector<BindingDesc> layout;                // the program's bindings of this group, same order
};

struct BufferInfo {
    uint64_t size = 0;
    uint32_t usage = 0;
    bool host_visible = false;
    bool imported = false;
    bool read_only = false;   // ExternalBuffer::read_only of an import
    Route route = Route::InPlace;
};

struct TextureInfo {
    uint32_t width = 0, height = 0;
    Format format = Format::RGBA8Unorm;
    uint32_t usage = 0;
    bool imported = false;
};

// Completion, reported by a backend from any thread. A backend's completion handlers hold
// it by shared pointer, so a late handler never reaches a destroyed backend.
class CompletionSink {
public:
    void complete(Serial s, double gpu_ms);
    Serial completed_through() const { return through_.load(std::memory_order_acquire); }
    double take_gpu_ms(uint32_t* submits = nullptr);   // the sum since the last take; < 0 none

private:
    mutable std::mutex mu_;
    std::set<Serial> early_;                        // completed ahead of an earlier serial
    std::atomic<Serial> through_{0};
    double gpu_ms_ = 0;
    uint32_t gpu_n_ = 0;
    bool gpu_known_ = false;
};

// Helpers shared by the backends.
inline uint64_t align_up(uint64_t v, uint64_t a) { return (v + a - 1) / a * a; }
bool is_integer_format(Format f);
bool is_depth_format(Format f);
bool is_buffer_binding(BindingType t);   // Uniform, Storage or StorageReadWrite

}  // namespace caliper_gfx
