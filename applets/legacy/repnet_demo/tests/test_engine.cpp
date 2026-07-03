// Smoke test for the Training Lab live training engine (TrainEngine).
// Builds a tiny SEPARABLE synthetic dataset (class 1 = class 0 + a small fixed
// sinusoidal bump on a couple leads), runs a few fast epochs on a background
// thread, polls thread-safe snapshots, and asserts the model learned the signal
// and that snapshots are well-formed and concurrency-safe.
#include "train_engine.h"
#include "golden_io.h"

#include <torch/torch.h>

#include <chrono>
#include <cmath>
#include <thread>
#include <vector>

namespace {

// Build a separable dataset of shape (N,12,2500). Base is low-amplitude noise;
// class 1 adds a fixed sinusoidal bump on leads 0 and 3 so it is learnable.
// y is balanced (alternating labels).
void make_dataset(int N, uint32_t seed, torch::Tensor& X, std::vector<int>& y) {
    torch::manual_seed(seed);
    const int64_t T = 2500;
    X = torch::randn({N, 12, T}) * 0.1f;
    y.resize(N);

    // Fixed sinusoidal bump.
    torch::Tensor t = torch::arange(T, torch::kFloat32);
    torch::Tensor bump = torch::sin(t * (2.0 * M_PI * 5.0 / static_cast<double>(T))) * 0.6f;

    for (int i = 0; i < N; ++i) {
        int label = i % 2;  // balanced
        y[i] = label;
        if (label == 1) {
            X[i][0] += bump;
            X[i][3] += bump;
        }
    }
}

}  // namespace

int main() {
    golden::Harness H;

    const uint32_t seed = 1121;
    torch::Tensor X_train, X_val;
    std::vector<int> y_train, y_val;
    make_dataset(128, seed, X_train, y_train);
    make_dataset(64, seed + 7, X_val, y_val);

    TrainEngine::Config cfg;
    cfg.max_epochs = 15;
    cfg.patience = 15;
    cfg.batch_size = 32;
    cfg.augment = false;
    cfg.mixup = false;
    cfg.seed = seed;

    TrainEngine engine;
    engine.start(X_train, y_train, X_val, y_val, cfg);

    // Poll snapshot() until done, with a ~120s timeout. Exercise thread-safety
    // by calling snapshot() repeatedly while training runs.
    auto t0 = std::chrono::steady_clock::now();
    TrainSnapshot snap;
    int polls = 0;
    while (true) {
        snap = engine.snapshot();  // must be safe to call repeatedly while running
        ++polls;
        if (snap.done) break;
        auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(
                           std::chrono::steady_clock::now() - t0)
                           .count();
        if (elapsed > 120) break;
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }

    H.check(snap.done, "training reached done within timeout");
    H.check(polls > 1, "snapshot() polled repeatedly while running (no crash)");

    std::fprintf(stderr, "  polls=%d epochs_run=%d\n", polls, snap.epoch);

    // best/final val AUROC must show learning of the separable signal.
    float best_auroc = snap.best_val_auroc;
    std::fprintf(stderr, "  best_val_auroc=%.4f val_auroc=%.4f best_epoch=%d\n",
                 best_auroc, snap.val_auroc, snap.best_epoch);
    H.check(best_auroc > 0.75f,
            "best val AUROC > 0.75 (got " + std::to_string(best_auroc) + ")");

    H.check(snap.best_epoch >= 0,
            "best_epoch >= 0 (got " + std::to_string(snap.best_epoch) + ")");

    // loss_history non-empty and decreasing (mean of last 3 < mean of first 3).
    H.check(!snap.loss_history.empty(), "loss_history non-empty");
    if (snap.loss_history.size() >= 6) {
        const auto& lh = snap.loss_history;
        float first3 = (lh[0] + lh[1] + lh[2]) / 3.0f;
        size_t n = lh.size();
        float last3 = (lh[n - 1] + lh[n - 2] + lh[n - 3]) / 3.0f;
        std::fprintf(stderr, "  first3_loss_mean=%.4f last3_loss_mean=%.4f\n",
                     first3, last3);
        H.check(last3 < first3, "mean(last 3 losses) < mean(first 3 losses)");
    } else {
        H.check(false, "loss_history has >= 6 entries");
    }

    // stage1_kernels must be (16,31).
    H.check(snap.stage1_kernels.defined(), "stage1_kernels defined");
    if (snap.stage1_kernels.defined()) {
        H.check(snap.stage1_kernels.sizes() == (std::vector<int64_t>{16, 31}),
                "stage1_kernels shape == (16,31)");
    }

    return H.report("test_engine");
}
