// Bounded CPU ownership checks. This fake backend produces no application evidence.
#include "live/scene_readbacks.h"
#include "cpu_check.h"
#include <set>

struct FakeBackend {
    std::set<uint64_t> pending;
    uint64_t next = 0;
    unsigned released = 0, taken = 0;
    bool refuse = false, ready = false;
    ~FakeBackend() { CIRCUIT_CHECK(pending.empty()); }
    caliper_gfx::Readback read_texture(caliper_gfx::Texture, int, int, uint32_t, uint32_t) {
        if (refuse) return {};
        pending.insert(++next);
        return {next};
    }
    bool take_readback(caliper_gfx::Readback h, std::vector<uint8_t>* bytes) {
        CIRCUIT_CHECK(pending.count(h.id));
        if (!ready) return false;
        const auto erased = pending.erase(h.id);
        CIRCUIT_CHECK(erased == 1);
        *bytes = {1, 2, 3, 4};
        ++taken;
        return true;
    }
    void release(caliper_gfx::Readback h) {
        const auto erased = pending.erase(h.id);
        CIRCUIT_CHECK(erased == 1);
        ++released;
    }
};

int main() {
    using Owner = caliper_live_views::SceneReadbacks<FakeBackend>;
    std::vector<uint8_t> bytes;
    FakeBackend shared;
    Owner survivor;
    auto surviving = survivor.read(shared, {}, 2, 2);
    std::shared_ptr<Owner::Lease> old;
    {
        Owner view;
        old = view.read(shared, {}, 2, 2);
        const bool taken = old->take(&bytes);
        CIRCUIT_CHECK(!taken);
    } // Replacement or page close cancels only this view on the shared backend.
    CIRCUIT_CHECK(shared.released == 1 && shared.pending.size() == 1);
    const bool old_taken = old->take(&bytes);
    CIRCUIT_CHECK(!old->pending() && !old_taken);
    old.reset();
    CIRCUIT_CHECK(shared.released == 1);
    {
        Owner replacement;
        auto capture = replacement.read(shared, {}, 2, 2);
        capture.reset(); // Timeout or capture destruction releases an outstanding handle.
        CIRCUIT_CHECK(shared.released == 2);
        auto first = replacement.read(shared, {}, 2, 2);
        shared.refuse = true;
        const auto refused = replacement.read(shared, {}, 2, 2);
        CIRCUIT_CHECK(!refused);
        first.reset(); // Partial admission releases the first admitted read.
        CIRCUIT_CHECK(shared.released == 3);
        shared.refuse = false;
    }
    shared.ready = true;
    const bool taken = surviving->take(&bytes);
    CIRCUIT_CHECK(taken && bytes == std::vector<uint8_t>({1, 2, 3, 4}));
    surviving.reset();
    CIRCUIT_CHECK(shared.taken == 1 && shared.released == 3 && shared.pending.empty());
    std::shared_ptr<Owner::Lease> late;
    {
        FakeBackend owned;
        Owner view;
        late = view.read(owned, {}, 2, 2);
    } // The view cancels before the owned backend dies.
    const bool late_taken = late->take(&bytes);
    CIRCUIT_CHECK(!late_taken);
    late.reset(); // No stale backend or view access.
}
