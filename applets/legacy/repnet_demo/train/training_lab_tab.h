// Training Lab tab: live PerLeadCNN training on the reproducible best split,
// with real-time kernel-adaptation, grad-cam saliency, and metric views.
// Self-contained: owns the dataset load thread, the TrainEngine, reference
// (ghost) kernels, and GL textures. Compiled into the repnet_demo applet.
#pragma once

#include <atomic>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "train_dataset.h"
#include "train_engine.h"

class TrainingLabTab {
   public:
    TrainingLabTab();
    ~TrainingLabTab();

    // Draw the whole tab. Call once per frame from inside a tab item.
    void draw();

   private:
    void draw_controls();
    void draw_example_picker(const TrainSnapshot& s);  // left sidebar scrubber
    void draw_metrics(const TrainSnapshot& s);
    void draw_kernels(const TrainSnapshot& s);
    void draw_saliency(const TrainSnapshot& s);        // saliency over waveform

    void start_load();          // spawn dataset load thread
    void start_training();      // (re)create engine and start on current split
    void load_reference();      // load ghost kernels from reference weights
    void reset();

    // --- dataset (loaded async) ---
    std::string data_dir_;
    std::string ref_dir_;       // dir of reference state_dict bins (ghost)
    std::thread load_thread_;
    std::atomic<bool> loading_{false};
    std::atomic<int> load_done_{0}, load_total_{0};
    std::atomic<bool> loaded_{false};
    std::string load_error_;
    std::mutex ds_mutex_;
    std::shared_ptr<tdata::Dataset> dataset_;  // set when loaded
    tdata::Split split_;

    // --- training ---
    std::unique_ptr<TrainEngine> engine_;
    int split_i_ = 17;          // seed 1119
    int max_epochs_ = 80;
    bool augment_ = true, mixup_ = true;
    bool show_ghost_ = true;

    // --- saliency example scrubber ---
    int viz_index_ = -1;        // selected held-out example (-1 = engine default)
    int viz_lead_ = 1;          // lead shown in the waveform overlay (default II)

    // --- reference (ghost) stage-1 kernels (16,31) ---
    torch::Tensor ref_kernels_;
    bool ref_loaded_ = false;
    float ref_auroc_ = 0.7793f;

    // --- saliency texture cache ---
    unsigned int gradcam_tex_ = 0;
    int gradcam_epoch_cached_ = -1;
};
