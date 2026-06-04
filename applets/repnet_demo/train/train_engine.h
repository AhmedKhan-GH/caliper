// Live training engine for the caliper "Training Lab".
// Trains PerLeadCNN on a background std::thread and publishes thread-safe
// snapshots for real-time visualization. CPU device, deterministic via seed.
#pragma once

#include <torch/torch.h>

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "perlead_cnn.h"

// Thread-safe, copyable view of the latest training state. snapshot() returns a
// copy with cloned tensors so the caller may read it without holding any lock.
struct TrainSnapshot {
    int epoch = 0;
    int best_epoch = -1;
    int max_epochs = 80;
    int patience_left = 20;
    float train_loss = 0;
    float val_auroc = 0;
    float best_val_auroc = 0;
    float lr = 0;
    bool running = false;
    bool done = false;
    std::string device = "CPU";  // "MPS" or "CPU"
    std::vector<float> loss_history;   // per-epoch train loss
    std::vector<float> auroc_history;  // per-epoch val AUROC
    // (16,31) clone of backbone[0].weight squeezed -- for "kernels adapting".
    torch::Tensor stage1_kernels;
    // Currently-selected held-out (val) example (12,2500) and its grad-cam
    // saliency (12,2500, upsampled to align with the waveform) under the
    // current weights -- for "saliency over the actual waveform".
    torch::Tensor pinned_input;
    torch::Tensor gradcam;
    int pinned_label = -1;   // 0/1 true label of the selected example
    int viz_index = -1;      // index into the val set
    float viz_prob = 0.0f;   // model P(positive) for the selected example
    int val_count = 0;       // number of held-out examples available to scrub
};

class TrainEngine {
   public:
    struct Config {
        int max_epochs = 80;
        int patience = 20;
        uint32_t seed = 1121;
        double lr = 1.2e-3;
        double weight_decay = 5e-3;
        double eta_min = 1e-6;
        int T_max = 80;
        int batch_size = 64;
        int accum = 2;
        bool augment = true;
        bool mixup = true;
        bool use_mps = true;  // train on Apple GPU (MPS) when available
    };

    TrainEngine();
    ~TrainEngine();  // sets stop flag + joins thread

    // X_*:(N,12,2500) float32 on CPU; y_*: labels (0/1). Copies what it needs and
    // starts the background training thread.
    void start(torch::Tensor X_train, std::vector<int> y_train,
               torch::Tensor X_val, std::vector<int> y_val, Config cfg);

    // Optional run controls.
    void pause();
    void resume();
    void request_stop();
    void step_once();

    // Select which held-out example the saliency view follows (index into the
    // val set). Recomputed promptly on the training thread, even while paused.
    void set_viz_index(int idx);
    int val_count() const;

    TrainSnapshot snapshot() const;  // thread-safe copy of latest snapshot
    plcnn::PerLeadCNN model() const;  // exposed for grad-cam etc.
    bool is_running() const;

   private:
    void run_loop();
    void publish(const TrainSnapshot& s);

    plcnn::PerLeadCNN model_;
    Config cfg_;

    torch::Tensor X_train_, X_val_;
    std::vector<int> y_train_, y_val_;

    std::thread thread_;
    mutable std::mutex snap_mutex_;
    TrainSnapshot snap_;

    std::atomic<bool> stop_{false};
    std::atomic<bool> running_{false};
    std::atomic<int> viz_index_{-1};   // -1 = auto (first positive val example)

    // pause / step controls.
    std::mutex ctrl_mutex_;
    std::condition_variable ctrl_cv_;
    bool paused_ = false;
    int step_budget_ = -1;  // <0 = unlimited; otherwise epochs allowed before pausing
};
