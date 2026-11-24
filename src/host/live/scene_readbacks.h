#pragma once
// Frame-thread leases. The view cancels them before its backend can disappear.
#include "gfx/gfx.h"
#include <algorithm>
#include <memory>
#include <vector>

namespace caliper_live_views {
template<class Backend>
class SceneReadbacks {
public:
    class Lease {
        friend class SceneReadbacks;
        Backend* backend_;
        caliper_gfx::Readback handle_;
        explicit Lease(Backend& backend) : backend_(&backend) {}
    public:
        Lease(const Lease&) = delete;
        Lease& operator=(const Lease&) = delete;
        ~Lease() { cancel(); }
        bool pending() const { return bool(handle_); }
        bool take(std::vector<uint8_t>* bytes) {
            if (!backend_ || !handle_ || !backend_->take_readback(handle_, bytes)) return false;
            handle_ = {};
            backend_ = nullptr;
            return true;
        }
        void cancel() {
            if (backend_ && handle_) backend_->release(handle_);
            handle_ = {};
            backend_ = nullptr;
        }
    };
    SceneReadbacks() = default;
    SceneReadbacks(const SceneReadbacks&) = delete;
    SceneReadbacks& operator=(const SceneReadbacks&) = delete;
    ~SceneReadbacks() { cancel(); }
    std::shared_ptr<Lease> read(Backend& backend, caliper_gfx::Texture texture, uint32_t width, uint32_t height) {
        leases_.erase(std::remove_if(leases_.begin(), leases_.end(), [](const auto& p) { return p.expired(); }), leases_.end());
        // Allocate and register the owner before the backend admits a handle.
        auto lease = std::shared_ptr<Lease>(new Lease(backend));
        leases_.push_back(lease);
        lease->handle_ = backend.read_texture(texture, 0, 0, width, height);
        return lease->pending() ? lease : nullptr;
    }
    void cancel() {
        for (auto& weak : leases_) if (auto lease = weak.lock()) lease->cancel();
        leases_.clear();
    }
private:
    std::vector<std::weak_ptr<Lease>> leases_;
};
} // namespace caliper_live_views
