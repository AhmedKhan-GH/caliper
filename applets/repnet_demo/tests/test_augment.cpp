// Tests for the on-the-fly ECG augmentation unit (augment::augment_ecg).
// Semantic (not bit-identical) port of repnet train.py augment_ecg.
// Covers: identity (all p=0), shape preservation, determinism, input
// immutability, and lead-drop sanity.

#include "golden_io.h"
#include "train/ecg_augment.h"

#include <torch/torch.h>

#include <cstdio>
#include <random>

int main() {
    golden::Harness H;

    const long T = 2500;

    // Deterministic-ish input waveform (12, T): nonzero everywhere.
    auto make_x = []() {
        torch::manual_seed(7);
        return torch::randn({12, 2500}, torch::kFloat32) + 1.0f;  // mean-shift -> all nonzero-ish
    };

    // 1. Identity: all p_* = 0 -> output equals input exactly, shape (12,T).
    {
        augment::AugCfg cfg;
        cfg.p_noise = cfg.p_amp_scale = cfg.p_time_shift = cfg.p_lead_drop =
            cfg.p_cutout = cfg.p_wander = cfg.p_resample = 0.0;
        torch::Tensor x = make_x();
        std::mt19937_64 rng(42);
        torch::Tensor out = augment::augment_ecg(x, cfg, rng);
        H.check(out.dim() == 2 && out.size(0) == 12 && out.size(1) == T,
                "identity output shape (12,T)");
        double mx = (out.to(torch::kFloat64) - x.to(torch::kFloat64)).abs().max().item<double>();
        H.check(mx == 0.0, "identity output equals input (max abs diff 0)");
    }

    // 2. Shape preserved across many iterations with default cfg.
    {
        augment::AugCfg cfg;  // defaults
        torch::Tensor x = make_x();
        std::mt19937_64 rng(2024);
        bool all_ok = true;
        for (int i = 0; i < 200; ++i) {
            torch::Tensor out = augment::augment_ecg(x, cfg, rng);
            if (!(out.dim() == 2 && out.size(0) == 12 && out.size(1) == T)) {
                all_ok = false;
                break;
            }
        }
        H.check(all_ok, "shape always (12,T) over 200 default-cfg iterations");
    }

    // 3. Determinism: same input + same seeded rng -> identical output.
    {
        augment::AugCfg cfg;  // defaults
        torch::Tensor x = make_x();
        std::mt19937_64 rng1(123), rng2(123);
        torch::Tensor a = augment::augment_ecg(x, cfg, rng1);
        torch::Tensor b = augment::augment_ecg(x, cfg, rng2);
        double mx = (a.to(torch::kFloat64) - b.to(torch::kFloat64)).abs().max().item<double>();
        H.check(mx == 0.0, "determinism: identical output for same seed");
    }

    // 4. Input untouched (clone semantics).
    {
        augment::AugCfg cfg;  // defaults, will mutate copy
        torch::Tensor x = make_x();
        torch::Tensor x_before = x.clone();
        std::mt19937_64 rng(999);
        augment::augment_ecg(x, cfg, rng);
        double mx =
            (x.to(torch::kFloat64) - x_before.to(torch::kFloat64)).abs().max().item<double>();
        H.check(mx == 0.0, "input tensor unchanged after augment_ecg");
    }

    // 5. lead_drop sanity: force only lead_drop; at least sometimes a whole lead
    //    is zeroed while others remain nonzero.
    {
        augment::AugCfg cfg;
        cfg.p_noise = cfg.p_amp_scale = cfg.p_time_shift = cfg.p_cutout =
            cfg.p_wander = cfg.p_resample = 0.0;
        cfg.p_lead_drop = 1.0;  // always attempt
        bool observed = false;
        for (int seed = 0; seed < 200 && !observed; ++seed) {
            torch::Tensor x = make_x();
            std::mt19937_64 rng(seed);
            torch::Tensor out = augment::augment_ecg(x, cfg, rng);
            // per-lead zero check
            auto lead_is_zero = (out == 0).all(/*dim=*/1);  // (12,) bool
            int n_zero = lead_is_zero.to(torch::kInt32).sum().item<int>();
            if (n_zero >= 1 && n_zero < 12) {
                observed = true;
            }
        }
        H.check(observed, "lead_drop sometimes zeros a whole lead while others remain");
    }

    return H.report("test_augment");
}
