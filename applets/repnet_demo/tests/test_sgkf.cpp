// Golden-file test for sgkf: bit-exact StratifiedGroupKFold(shuffle=True) port.
#include "golden_io.h"
#include "sgkf.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <map>
#include <numeric>
#include <vector>

using json = nlohmann::json;

int main() {
    golden::Harness H;

    // ---- 1. MT19937 sanity: shuffle arange(10) under seeds 1119 and 1120 ----
    // Expected captured from numpy RandomState in the repnet venv.
    {
        std::vector<int> a(10);
        std::iota(a.begin(), a.end(), 0);
        sgkf::MT19937 mt(1119);
        mt.shuffle_rows(a);
        std::vector<int> expect1119{0, 7, 8, 2, 9, 3, 6, 5, 1, 4};
        H.check(a == expect1119, "MT shuffle arange(10) seed=1119 matches numpy");
    }
    {
        std::vector<int> a(10);
        std::iota(a.begin(), a.end(), 0);
        sgkf::MT19937 mt(1120);
        mt.shuffle_rows(a);
        std::vector<int> expect1120{1, 9, 2, 5, 0, 7, 8, 6, 4, 3};
        H.check(a == expect1120, "MT shuffle arange(10) seed=1120 matches numpy");
    }

    // ---- Load golden case ----
    json d = json::parse(golden::load_text("sgkf_case.json"));
    std::vector<int> y = d["y"].get<std::vector<int>>();
    std::vector<int> groups_inv = d["groups_inv"].get<std::vector<int>>();

    // ---- 2. Outer split: seed 1119, 5 splits, against full dataset ----
    {
        int n_splits = d["outer"]["n_splits"].get<int>();
        uint32_t seed = d["outer"]["seed"].get<uint32_t>();
        auto folds = sgkf::stratified_group_kfold_test_folds(y, groups_inv, n_splits, seed);
        auto expect = d["outer"]["test_folds"].get<std::vector<std::vector<int>>>();
        H.check(static_cast<int>(folds.size()) == n_splits, "outer: n folds matches");
        for (int i = 0; i < n_splits && i < static_cast<int>(folds.size()); ++i) {
            H.check(folds[i] == expect[i],
                    "outer fold " + std::to_string(i) + " matches golden");
        }
    }

    // ---- 3. Inner split: seed 1120, 8 splits, against dev subset ----
    // dev set = complement of outer fold 0; indices are provided as dev_idx.
    {
        std::vector<int> dev_idx = d["dev_idx"].get<std::vector<int>>();

        // y restricted to dev samples.
        std::vector<int> y_dev;
        y_dev.reserve(dev_idx.size());
        // original group label per dev sample.
        std::vector<int> dev_groups_orig;
        dev_groups_orig.reserve(dev_idx.size());
        for (int s : dev_idx) {
            y_dev.push_back(y[s]);
            dev_groups_orig.push_back(groups_inv[s]);
        }

        // Re-encode dev groups via numpy-unique-sorted order over the dev group labels.
        std::vector<int> uniq(dev_groups_orig.begin(), dev_groups_orig.end());
        std::sort(uniq.begin(), uniq.end());
        uniq.erase(std::unique(uniq.begin(), uniq.end()), uniq.end());
        std::map<int, int> remap;
        for (int i = 0; i < static_cast<int>(uniq.size()); ++i) remap[uniq[i]] = i;
        std::vector<int> groups_inv_dev;
        groups_inv_dev.reserve(dev_groups_orig.size());
        for (int g : dev_groups_orig) groups_inv_dev.push_back(remap[g]);

        int n_splits = d["inner"]["n_splits"].get<int>();
        uint32_t seed = d["inner"]["seed"].get<uint32_t>();
        auto folds = sgkf::stratified_group_kfold_test_folds(y_dev, groups_inv_dev, n_splits, seed);
        auto expect = d["inner"]["test_folds"].get<std::vector<std::vector<int>>>();
        H.check(static_cast<int>(folds.size()) == n_splits, "inner: n folds matches");
        for (int i = 0; i < n_splits && i < static_cast<int>(folds.size()); ++i) {
            H.check(folds[i] == expect[i],
                    "inner fold " + std::to_string(i) + " matches golden");
        }
    }

    return H.report("test_sgkf");
}
