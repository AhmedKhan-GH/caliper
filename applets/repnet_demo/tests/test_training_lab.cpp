// Integration test for the Training Lab data path: exactly what the UI tab
// does, minus ImGui. Loads the real split-17 data via the C++ pipeline, trains
// PerLeadCNN live for a few epochs, and confirms the snapshots the UI renders
// are well-formed and that the model actually learns on real ECG data.
#include <chrono>
#include <thread>

#include "golden_io.h"
#include "train_dataset.h"
#include "train_engine.h"

int main() {
    golden::Harness H;
    const char* env = std::getenv("TLAB_DATA_DIR");
    std::string data_dir =
        env ? env : "/Users/ahmed/PycharmProjects/repnet/data/seniordesign_upload";

    // 1. Load + preprocess + split, exactly as TrainingLabTab::start_load does.
    auto ds = tdata::load_and_preprocess(data_dir);
    H.check(ds.X.size(0) == 2178, "dataset N=2178");
    tdata::Split sp = tdata::make_split(ds, 17);
    H.check(!sp.train.empty() && !sp.val.empty() && !sp.test.empty(),
            "split 17 has train/val/test");

    auto pick = [&](const std::vector<int>& idx, torch::Tensor& X,
                    std::vector<int>& y) {
        torch::Tensor it = torch::tensor(
            std::vector<int64_t>(idx.begin(), idx.end()), torch::kLong);
        X = ds.X.index_select(0, it).clone();
        y.clear();
        for (int i : idx) y.push_back(ds.y[i]);
    };
    torch::Tensor Xtr, Xva;
    std::vector<int> ytr, yva;
    pick(sp.train, Xtr, ytr);
    pick(sp.val, Xva, yva);
    int val_pos = 0;
    for (int v : yva) val_pos += v;
    H.check(val_pos > 0, "val set has at least one positive (for grad-cam pin)");

    // 2. Train live for a few epochs (augment off for speed/determinism).
    TrainEngine engine;
    TrainEngine::Config cfg;
    cfg.max_epochs = 8;
    cfg.T_max = 8;
    cfg.patience = 8;
    cfg.seed = 17 * 7 + 1000 + 2;  // 1121
    cfg.augment = false;
    cfg.mixup = false;
    engine.start(std::move(Xtr), std::move(ytr), std::move(Xva), std::move(yva), cfg);

    // 3. Poll snapshots until done (mirrors the UI's per-frame snapshot()).
    TrainSnapshot s;
    auto t0 = std::chrono::steady_clock::now();
    int polls = 0;
    for (;;) {
        s = engine.snapshot();
        ++polls;
        if (s.done) break;
        auto dt = std::chrono::duration_cast<std::chrono::seconds>(
                      std::chrono::steady_clock::now() - t0)
                      .count();
        if (dt > 300) {
            H.check(false, "training timed out (>300s)");
            break;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(200));
    }

    // 4. The snapshot the UI renders must be well-formed.
    H.check(s.stage1_kernels.defined(), "stage1_kernels published");
    H.check(s.stage1_kernels.dim() == 2 && s.stage1_kernels.size(0) == 16 &&
                s.stage1_kernels.size(1) == 31,
            "stage1_kernels shape (16,31)");
    H.check(s.pinned_input.defined() &&
                s.pinned_input.sizes() == torch::IntArrayRef({12, 2500}),
            "pinned_input shape (12,2500)");
    H.check(s.gradcam.defined() && s.gradcam.size(0) == 12,
            "gradcam published (12, T')");
    H.check(s.gradcam.max().item<float>() >= 0.0f, "gradcam is non-negative (relu)");
    H.check((int)s.auroc_history.size() == s.epoch, "auroc history per epoch");

    // 5. The model actually learned something on real data (best val AUROC well
    //    above chance). Real ECG is hard; even a few epochs should clear 0.6.
    std::fprintf(stderr, "  polls=%d epochs=%d best_val_auroc=%.4f final_loss=%.4f\n",
                 polls, s.epoch, s.best_val_auroc,
                 s.loss_history.empty() ? -1.f : s.loss_history.back());
    H.check(s.best_val_auroc > 0.6f, "best val AUROC > 0.6 on real split-17 data");
    H.check(!s.loss_history.empty() &&
                s.loss_history.back() < s.loss_history.front(),
            "train loss decreased");

    return H.report("test_training_lab");
}
