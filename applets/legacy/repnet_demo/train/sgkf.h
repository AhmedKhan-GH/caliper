// Bit-exact C++ port of scikit-learn's StratifiedGroupKFold(shuffle=True).
// Reproduces sklearn StratifiedGroupKFold(n_splits, shuffle=True, random_state=seed)
// ._iter_test_indices, including numpy legacy RandomState (MT19937) row shuffle.
#pragma once

#include <cstdint>
#include <vector>

namespace sgkf {

// Reference MT19937 matching numpy legacy RandomState seeded via init_by_array.
class MT19937 {
public:
    explicit MT19937(uint32_t seed);
    uint32_t genrand_int32();
    // Draw an integer in [0, max_v] using numpy's rk_interval masking-rejection.
    uint32_t rk_interval(uint32_t max_v);
    // Fisher-Yates shuffle of n row indices (numpy legacy order), in place.
    // Applies the same swaps to `rows`, an n-length vector of arbitrary payloads.
    template <typename T>
    void shuffle_rows(std::vector<T>& rows) {
        const size_t n = rows.size();
        if (n < 2) return;
        for (size_t i = n - 1; i >= 1; --i) {
            uint32_t j = rk_interval(static_cast<uint32_t>(i));
            std::swap(rows[i], rows[j]);
            if (i == 1) break;  // size_t underflow guard
        }
    }

private:
    void init_genrand(uint32_t s);
    void init_by_array(const uint32_t* key, int key_length);
    static constexpr int N = 624;
    uint32_t mt_[624];
    int mti_;
};

// Reproduces StratifiedGroupKFold(n_splits, shuffle=True, random_state=seed).split(...,groups).
// y: per-sample class label (e.g. 0/1). groups_inv: per-sample group id, already
// numpy-unique-sorted (values 0..n_groups-1). Returns the test-index list for each of
// the n_splits folds (each sorted ascending), as sample indices into y/groups_inv.
std::vector<std::vector<int>> stratified_group_kfold_test_folds(
    const std::vector<int>& y, const std::vector<int>& groups_inv,
    int n_splits, uint32_t seed);

}  // namespace sgkf
