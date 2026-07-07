#pragma once
/* caliper/adapters/alloc_registry.hpp — maps a raw device pointer back to the
 * exportable allocation that contains it (base -> {handle, size}), so the
 * adapter can hand the bridge (alloc, offset) instead of a bare pointer.
 * Pure C++, no CUDA dependency: unit-tested on every platform. */
#include <cstdint>
#include <map>
#include <mutex>
#include <optional>

namespace caliper::adapters {

class AllocRegistry {
public:
    struct Hit {
        void*     os_handle;
        uint64_t  size;
        uint64_t  offset;
        uintptr_t base;
    };

    void add(uintptr_t base, uint64_t size, void* os_handle) {
        std::lock_guard<std::mutex> lock(mu_);
        ranges_[base] = Range{size, os_handle};
    }
    void remove(uintptr_t base) {
        std::lock_guard<std::mutex> lock(mu_);
        ranges_.erase(base);
    }
    std::optional<Hit> find(const void* p, uint64_t extent_bytes) const {
        const auto addr = reinterpret_cast<uintptr_t>(p);
        std::lock_guard<std::mutex> lock(mu_);
        auto it = ranges_.upper_bound(addr);   // first base > addr
        if (it == ranges_.begin()) return std::nullopt;
        --it;                                  // candidate: greatest base <= addr
        const uintptr_t base = it->first;
        const Range& r = it->second;
        if (addr < base) return std::nullopt;
        const uint64_t off = addr - base;
        if (off > r.size || r.size - off < extent_bytes) return std::nullopt;
        return Hit{r.os_handle, r.size, off, base};
    }

private:
    struct Range { uint64_t size; void* os_handle; };
    std::map<uintptr_t, Range> ranges_;
    mutable std::mutex mu_;
};

} // namespace caliper::adapters
