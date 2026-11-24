// Bounded CPU ownership checks. This fake backend produces no application evidence.
#include "live/scene_readbacks.h"
#include <cassert>
#include <set>

struct FakeBackend {
    std::set<uint64_t> pending;
    uint64_t next = 0;
    unsigned released = 0, taken = 0;
    bool refuse = false, ready = false;
    ~FakeBackend() { assert(pending.empty()); }
    caliper_gfx::Readback read_texture(caliper_gfx::Texture, int, int, uint32_t, uint32_t) {
        if (refuse) return {};
        pending.insert(++next);
        return {next};
    }
    bool take_readback(caliper_gfx::Readback h, std::vector<uint8_t>* bytes) {
        assert(pending.count(h.id));
        if (!ready) return false;
        assert(pending.erase(h.id) == 1);
        *bytes = {1, 2, 3, 4};
        ++taken;
        return true;
    }
    void release(caliper_gfx::Readback h) {
        assert(pending.erase(h.id) == 1);
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
        assert(!old->take(&bytes));
    } // Replacement or page close cancels only this view on the shared backend.
    assert(shared.released == 1 && shared.pending.size() == 1);
    assert(!old->pending() && !old->take(&bytes));
    old.reset();
    assert(shared.released == 1);
    {
        Owner replacement;
        auto capture = replacement.read(shared, {}, 2, 2);
        capture.reset(); // Timeout or capture destruction releases an outstanding handle.
        assert(shared.released == 2);
        auto first = replacement.read(shared, {}, 2, 2);
        shared.refuse = true;
        assert(!replacement.read(shared, {}, 2, 2));
        first.reset(); // Partial admission releases the first admitted read.
        assert(shared.released == 3);
        shared.refuse = false;
    }
    shared.ready = true;
    assert(surviving->take(&bytes) && bytes.size() == 4);
    surviving.reset();
    assert(shared.taken == 1 && shared.released == 3 && shared.pending.empty());
    std::shared_ptr<Owner::Lease> late;
    {
        FakeBackend owned;
        Owner view;
        late = view.read(owned, {}, 2, 2);
    } // The view cancels before the owned backend dies.
    assert(!late->take(&bytes));
    late.reset(); // No stale backend or view access.
}
