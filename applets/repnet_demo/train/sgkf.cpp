#include "sgkf.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>

// Disable floating-point contraction (FMA fusion) for the whole translation unit
// so std/variance reductions round identically to numpy. See pop_std for why.
#if defined(__clang__)
#pragma clang fp contract(off)
#endif

namespace sgkf {

// ---------------- MT19937 (reference, numpy legacy RandomState) ----------------

void MT19937::init_genrand(uint32_t s) {
    mt_[0] = s;
    for (mti_ = 1; mti_ < N; ++mti_) {
        mt_[mti_] =
            (1812433253u * (mt_[mti_ - 1] ^ (mt_[mti_ - 1] >> 30)) +
             static_cast<uint32_t>(mti_));
    }
}

void MT19937::init_by_array(const uint32_t* key, int key_length) {
    init_genrand(19650218u);
    int i = 1, j = 0;
    int k = (N > key_length) ? N : key_length;
    for (; k; --k) {
        mt_[i] = (mt_[i] ^ ((mt_[i - 1] ^ (mt_[i - 1] >> 30)) * 1664525u)) +
                 key[j] + static_cast<uint32_t>(j);
        ++i;
        ++j;
        if (i >= N) {
            mt_[0] = mt_[N - 1];
            i = 1;
        }
        if (j >= key_length) j = 0;
    }
    for (k = N - 1; k; --k) {
        mt_[i] = (mt_[i] ^ ((mt_[i - 1] ^ (mt_[i - 1] >> 30)) * 1566083941u)) -
                 static_cast<uint32_t>(i);
        ++i;
        if (i >= N) {
            mt_[0] = mt_[N - 1];
            i = 1;
        }
    }
    mt_[0] = 0x80000000u;
}

MT19937::MT19937(uint32_t seed) : mti_(N + 1) {
    // numpy legacy RandomState(int) seeds via plain init_genrand(seed)
    // (mt19937_seed), NOT init_by_array. Verified against numpy get_state().
    init_genrand(seed);
}

uint32_t MT19937::genrand_int32() {
    static const uint32_t UPPER_MASK = 0x80000000u;
    static const uint32_t LOWER_MASK = 0x7fffffffu;
    static const uint32_t mag01[2] = {0x0u, 0x9908b0dfu};
    uint32_t yv;

    if (mti_ >= N) {
        int kk;
        for (kk = 0; kk < N - 397; ++kk) {
            yv = (mt_[kk] & UPPER_MASK) | (mt_[kk + 1] & LOWER_MASK);
            mt_[kk] = mt_[kk + 397] ^ (yv >> 1) ^ mag01[yv & 0x1u];
        }
        for (; kk < N - 1; ++kk) {
            yv = (mt_[kk] & UPPER_MASK) | (mt_[kk + 1] & LOWER_MASK);
            mt_[kk] = mt_[kk + (397 - N)] ^ (yv >> 1) ^ mag01[yv & 0x1u];
        }
        yv = (mt_[N - 1] & UPPER_MASK) | (mt_[0] & LOWER_MASK);
        mt_[N - 1] = mt_[396] ^ (yv >> 1) ^ mag01[yv & 0x1u];
        mti_ = 0;
    }

    yv = mt_[mti_++];
    yv ^= (yv >> 11);
    yv ^= (yv << 7) & 0x9d2c5680u;
    yv ^= (yv << 15) & 0xefc60000u;
    yv ^= (yv >> 18);
    return yv;
}

uint32_t MT19937::rk_interval(uint32_t max_v) {
    if (max_v == 0) return 0;
    uint32_t mask = max_v;
    mask |= mask >> 1;
    mask |= mask >> 2;
    mask |= mask >> 4;
    mask |= mask >> 8;
    mask |= mask >> 16;
    uint32_t value;
    while ((value = (genrand_int32() & mask)) > max_v) {
    }
    return value;
}

// ---------------- StratifiedGroupKFold ----------------

namespace {

// Population std (ddof=0) of a vector, reproducing numpy's _var reduction.
// CRITICAL: the squared deviation must be rounded to a double BEFORE being added
// to the accumulator. Writing `acc += dv*dv` lets the compiler fuse it into an
// FMA (no intermediate rounding) on platforms like arm64, which diverges from
// numpy by 1 ULP and flips fold-assignment tie-breaks. Keep `sq` as a separate
// volatile-free named temporary AND disable contraction here.
double pop_std(const std::vector<double>& v) {
    const size_t n = v.size();
    if (n == 0) return 0.0;
    double mean = 0.0;
    for (double x : v) mean += x;
    mean /= static_cast<double>(n);
    double acc = 0.0;
    for (double x : v) {
        const double dv = x - mean;
        const double sq = dv * dv;  // force rounding to double (no FMA fusion)
        acc += sq;
    }
    return std::sqrt(acc / static_cast<double>(n));
}

// numpy.isclose default (rtol=1e-5, atol=1e-8), including its infinity handling:
// non-finite values are only close if exactly equal (same-sign inf); a finite
// value is never close to inf (numpy special-cases this, unlike the raw formula).
bool is_close(double a, double b) {
    if (std::isinf(a) || std::isinf(b)) return a == b;
    return std::fabs(a - b) <= (1e-08 + 1e-05 * std::fabs(b));
}

int find_best_fold(std::vector<std::vector<double>>& y_counts_per_fold,
                   const std::vector<double>& y_cnt,
                   const std::vector<double>& group_y_counts,
                   int n_splits, int n_classes) {
    int best_fold = -1;
    double min_eval = std::numeric_limits<double>::infinity();
    double min_samples_in_fold = std::numeric_limits<double>::infinity();
    for (int i = 0; i < n_splits; ++i) {
        for (int c = 0; c < n_classes; ++c) y_counts_per_fold[i][c] += group_y_counts[c];
        // std over folds, per class, of (y_counts_per_fold / y_cnt); then mean over classes.
        double sum_std = 0.0;
        for (int c = 0; c < n_classes; ++c) {
            std::vector<double> col(n_splits);
            for (int f = 0; f < n_splits; ++f) col[f] = y_counts_per_fold[f][c] / y_cnt[c];
            sum_std += pop_std(col);
        }
        double fold_eval = sum_std / static_cast<double>(n_classes);
        for (int c = 0; c < n_classes; ++c) y_counts_per_fold[i][c] -= group_y_counts[c];

        double samples_in_fold = 0.0;
        for (int c = 0; c < n_classes; ++c) samples_in_fold += y_counts_per_fold[i][c];

        bool is_better = (fold_eval < min_eval) ||
                         (is_close(fold_eval, min_eval) &&
                          samples_in_fold < min_samples_in_fold);
        if (is_better) {
            min_eval = fold_eval;
            min_samples_in_fold = samples_in_fold;
            best_fold = i;
        }
    }
    return best_fold;
}

}  // namespace

std::vector<std::vector<int>> stratified_group_kfold_test_folds(
    const std::vector<int>& y, const std::vector<int>& groups_inv,
    int n_splits, uint32_t seed) {
    const int n_samples = static_cast<int>(y.size());

    // unique(y) -> sorted classes, inverse + counts. Map class label -> dense index.
    std::vector<int> classes(y.begin(), y.end());
    std::sort(classes.begin(), classes.end());
    classes.erase(std::unique(classes.begin(), classes.end()), classes.end());
    const int n_classes = static_cast<int>(classes.size());
    std::vector<int> y_class_idx(n_samples);
    {
        // classes is sorted; binary search to dense index.
        for (int s = 0; s < n_samples; ++s) {
            int lo = 0, hi = n_classes - 1, idx = 0;
            while (lo <= hi) {
                int mid = (lo + hi) / 2;
                if (classes[mid] == y[s]) { idx = mid; break; }
                if (classes[mid] < y[s]) lo = mid + 1; else hi = mid - 1;
            }
            y_class_idx[s] = idx;
        }
    }
    std::vector<double> y_cnt(n_classes, 0.0);
    for (int s = 0; s < n_samples; ++s) y_cnt[y_class_idx[s]] += 1.0;

    // n_groups from groups_inv (values already 0..n_groups-1).
    int n_groups = 0;
    for (int g : groups_inv) n_groups = std::max(n_groups, g + 1);

    // y_counts_per_group[group, class].
    std::vector<std::vector<double>> y_counts_per_group(
        n_groups, std::vector<double>(n_classes, 0.0));
    for (int s = 0; s < n_samples; ++s)
        y_counts_per_group[groups_inv[s]][y_class_idx[s]] += 1.0;

    // shuffle ROWS in place (numpy legacy RandomState.shuffle on axis 0).
    {
        MT19937 mt(seed);
        mt.shuffle_rows(y_counts_per_group);
    }

    // argsort(-std(y_counts_per_group, axis=1), kind="mergesort") -> stable, descending std.
    std::vector<int> sorted_groups_idx(n_groups);
    std::iota(sorted_groups_idx.begin(), sorted_groups_idx.end(), 0);
    std::vector<double> row_std(n_groups);
    for (int g = 0; g < n_groups; ++g) row_std[g] = pop_std(y_counts_per_group[g]);
    // Stable sort by descending row_std (ties keep ascending original index).
    std::stable_sort(sorted_groups_idx.begin(), sorted_groups_idx.end(),
                     [&](int a, int b) { return row_std[a] > row_std[b]; });

    std::vector<std::vector<double>> y_counts_per_fold(
        n_splits, std::vector<double>(n_classes, 0.0));
    std::vector<std::vector<char>> in_fold(n_splits, std::vector<char>(n_groups, 0));

    for (int group_idx : sorted_groups_idx) {
        const std::vector<double>& gyc = y_counts_per_group[group_idx];
        int best_fold = find_best_fold(y_counts_per_fold, y_cnt, gyc, n_splits, n_classes);
        for (int c = 0; c < n_classes; ++c) y_counts_per_fold[best_fold][c] += gyc[c];
        in_fold[best_fold][group_idx] = 1;
    }

    // Collect test indices: for fold i, samples whose groups_inv value is in fold i.
    std::vector<std::vector<int>> result(n_splits);
    for (int i = 0; i < n_splits; ++i) {
        for (int s = 0; s < n_samples; ++s) {
            int g = groups_inv[s];
            if (g < n_groups && in_fold[i][g]) result[i].push_back(s);
        }
    }
    return result;
}

}  // namespace sgkf
