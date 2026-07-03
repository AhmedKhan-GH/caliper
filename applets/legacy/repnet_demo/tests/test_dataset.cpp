// CROWN-JEWEL end-to-end test for the Training Lab data pipeline.
// Load raw ECG CSVs in C++ -> DSP preprocess -> SGKF split -> PerLeadCNN
// forward -> reproduce the recorded split-17 test AUROC = 0.7793.
#include "golden_io.h"
#include "perlead_cnn.h"
#include "train_dataset.h"

#include <nlohmann/json.hpp>
#include <torch/torch.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <numeric>
#include <set>
#include <string>
#include <vector>

using json = nlohmann::json;

// Rank-based AUROC (Mann-Whitney U / trapezoid), tie-aware via average ranks.
static double auroc(const std::vector<double>& score, const std::vector<int>& y) {
    const size_t n = score.size();
    std::vector<size_t> idx(n);
    std::iota(idx.begin(), idx.end(), 0);
    std::sort(idx.begin(), idx.end(),
              [&](size_t a, size_t b) { return score[a] < score[b]; });
    // Assign average ranks (1-based) handling ties.
    std::vector<double> rank(n);
    size_t i = 0;
    while (i < n) {
        size_t j = i;
        while (j + 1 < n && score[idx[j + 1]] == score[idx[i]]) ++j;
        double avg = (static_cast<double>(i + 1) + static_cast<double>(j + 1)) / 2.0;
        for (size_t k = i; k <= j; ++k) rank[idx[k]] = avg;
        i = j + 1;
    }
    double sum_pos = 0.0;
    long n_pos = 0, n_neg = 0;
    for (size_t k = 0; k < n; ++k) {
        if (y[k] == 1) {
            sum_pos += rank[k];
            ++n_pos;
        } else {
            ++n_neg;
        }
    }
    if (n_pos == 0 || n_neg == 0) return 0.0;
    return (sum_pos - n_pos * (n_pos + 1) / 2.0) /
           (static_cast<double>(n_pos) * static_cast<double>(n_neg));
}

static std::string data_dir() {
    if (const char* e = std::getenv("TLAB_DATA_DIR")) return e;
    return "/Users/ahmed/PycharmProjects/repnet/data/seniordesign_upload";
}

int main() {
    golden::Harness H;
    json meta = json::parse(golden::load_text("meta.json"));

    // ---- 1. resample_to_5000 matches resample_5000.bin (<1e-3) ----
    {
        torch::Tensor in = golden::load_bin("resample_2500.bin");   // (12,2500)
        torch::Tensor ref = golden::load_bin("resample_5000.bin");  // (12,5000)
        H.check(in.sizes() == (std::vector<int64_t>{12, 2500}),
                "resample_2500 is (12,2500)");
        torch::Tensor out = tdata::resample_to_5000(in);
        H.check(out.sizes() == (std::vector<int64_t>{12, 5000}),
                "resample output is (12,5000)");
        double max_err = (out - ref).abs().max().item<double>();
        std::fprintf(stderr, "  resample max abs error: %.3e\n", max_err);
        H.check(max_err < 1e-3, "resample matches golden (<1e-3)");
    }

    // ---- 2. load_and_preprocess: N, positives, n_groups ----
    std::fprintf(stderr, "  loading + preprocessing dataset (this may take a while)...\n");
    tdata::Dataset d = tdata::load_and_preprocess(
        data_dir(), [](int done, int total) {
            if (done % 500 == 0 || done == total)
                std::fprintf(stderr, "    %d/%d\n", done, total);
        });
    int want_n = meta["n_samples"].get<int>();
    int want_pos = meta["n_positive"].get<int>();
    int want_groups = meta["n_patients"].get<int>();
    int n = static_cast<int>(d.y.size());
    int pos = std::accumulate(d.y.begin(), d.y.end(), 0);
    std::fprintf(stderr, "  N=%d positives=%d n_groups=%d\n", n, pos, d.n_groups);
    H.check(n == want_n, "N == " + std::to_string(want_n) + " (got " + std::to_string(n) + ")");
    H.check(pos == want_pos,
            "positives == " + std::to_string(want_pos) + " (got " + std::to_string(pos) + ")");
    H.check(d.n_groups == want_groups,
            "n_groups == " + std::to_string(want_groups) + " (got " + std::to_string(d.n_groups) + ")");
    H.check(d.X.sizes() == (std::vector<int64_t>{n, 12, 2500}), "X shape (N,12,2500)");

    // ---- 3. make_split(d,17): test == golden outer.test_folds[0]; integrity ----
    tdata::Split s = tdata::make_split(d, 17);
    {
        json sc = json::parse(golden::load_text("sgkf_case.json"));
        auto want_test = sc["outer"]["test_folds"][0].get<std::vector<int>>();
        std::vector<int> got_test = s.test;
        std::sort(got_test.begin(), got_test.end());
        H.check(got_test == want_test, "split17 test indices == golden outer.test_folds[0]");

        // disjoint + cover all N.
        std::set<int> all;
        bool dup = false;
        for (const auto* v : {&s.train, &s.val, &s.test})
            for (int x : *v)
                if (!all.insert(x).second) dup = true;
        H.check(!dup, "train/val/test are pairwise disjoint");
        H.check(static_cast<int>(all.size()) == n, "train+val+test cover all N samples");

        // group integrity: no patient appears in more than one of train/val/test.
        std::map<int, int> grp_part;  // group -> partition id
        bool leak = false;
        auto assign = [&](const std::vector<int>& part, int pid) {
            for (int idx : part) {
                int g = d.groups_inv[idx];
                auto it = grp_part.find(g);
                if (it == grp_part.end())
                    grp_part[g] = pid;
                else if (it->second != pid)
                    leak = true;
            }
        };
        assign(s.train, 0);
        assign(s.val, 1);
        assign(s.test, 2);
        H.check(!leak, "no patient group spans multiple of train/val/test");
    }

    // ---- 4. CROWN JEWEL: forward PerLeadCNN on X[test], compute AUROC ----
    {
        plcnn::PerLeadCNN model(12, {16, 32, 48}, {31, 21, 11}, 0.15, 2);
        int loaded = plcnn::load_state_dict_bins(model, golden::path("state_dict"));
        H.check(loaded > 0, "loaded state_dict tensors (" + std::to_string(loaded) + ")");
        model->eval();
        torch::NoGradGuard ng;

        std::vector<int> test = s.test;
        std::sort(test.begin(), test.end());
        const int B = static_cast<int>(test.size());
        std::vector<int64_t> sel(test.begin(), test.end());
        torch::Tensor idx = torch::tensor(sel, torch::kLong);
        torch::Tensor Xte = d.X.index_select(0, idx);  // (B,12,2500)

        std::vector<double> probs(B);
        const int chunk = 256;
        for (int start = 0; start < B; start += chunk) {
            int end = std::min(start + chunk, B);
            torch::Tensor xb = Xte.slice(0, start, end);
            torch::Tensor logits = model->forward(xb);          // (b,2)
            torch::Tensor p = torch::softmax(logits, 1).select(1, 1);
            auto acc = p.to(torch::kDouble).contiguous();
            const double* pd = acc.data_ptr<double>();
            for (int i = 0; i < end - start; ++i) probs[start + i] = pd[i];
        }
        std::vector<int> yte(B);
        for (int i = 0; i < B; ++i) yte[i] = d.y[test[i]];

        double au = auroc(probs, yte);
        double want_au = meta["split17_test_auroc"].get<double>();
        std::fprintf(stderr, "  ===> achieved end-to-end test AUROC: %.6f (want %.4f)\n",
                     au, want_au);
        H.check(std::abs(au - 0.7793) < 0.005,
                "end-to-end AUROC within 0.005 of 0.7793");
    }

    return H.report("test_dataset");
}
