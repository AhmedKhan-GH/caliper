// Live training engine for the caliper "Training Lab".
// AdamW + manual CosineAnnealingLR, FocalLoss (gamma=1, label_smoothing=0.05),
// mixup, on-the-fly augmentation, grad-accum, early-stopping on val AUROC. Runs
// on a background thread; publishes thread-safe snapshots after every epoch.
#include "train_engine.h"

#include <algorithm>
#include <cmath>
#include <memory>
#include <numeric>
#include <random>

#include "ecg_augment.h"

namespace {

// FocalLoss: ce = cross_entropy(logits, target, label_smoothing, reduction=none)
// pt = exp(-ce); loss = ((1-pt)^gamma * ce).mean()  (gamma=1.0).
torch::Tensor focal_loss(const torch::Tensor& logits, const torch::Tensor& target,
                         double gamma = 1.0, double label_smoothing = 0.05) {
    auto ce = torch::nn::functional::cross_entropy(
        logits, target,
        torch::nn::functional::CrossEntropyFuncOptions()
            .label_smoothing(label_smoothing)
            .reduction(torch::kNone));
    auto pt = torch::exp(-ce);
    auto loss = (torch::pow(1.0 - pt, gamma) * ce).mean();
    return loss;
}

// Rank-based AUROC (Mann-Whitney U) for binary scores. `scores` are the positive
// class probabilities/logits (higher => more likely positive); labels in {0,1}.
double auroc_mann_whitney(const std::vector<float>& scores,
                          const std::vector<int>& labels) {
    const size_t n = scores.size();
    if (n == 0) return 0.5;
    // Rank scores (average ranks for ties), 1-based.
    std::vector<size_t> idx(n);
    std::iota(idx.begin(), idx.end(), 0);
    std::sort(idx.begin(), idx.end(),
              [&](size_t a, size_t b) { return scores[a] < scores[b]; });
    std::vector<double> rank(n);
    size_t i = 0;
    while (i < n) {
        size_t j = i;
        while (j + 1 < n && scores[idx[j + 1]] == scores[idx[i]]) ++j;
        double avg = (static_cast<double>(i) + static_cast<double>(j)) / 2.0 + 1.0;
        for (size_t k = i; k <= j; ++k) rank[idx[k]] = avg;
        i = j + 1;
    }
    double sum_pos_ranks = 0.0;
    size_t n_pos = 0, n_neg = 0;
    for (size_t k = 0; k < n; ++k) {
        if (labels[k] == 1) {
            sum_pos_ranks += rank[k];
            ++n_pos;
        } else {
            ++n_neg;
        }
    }
    if (n_pos == 0 || n_neg == 0) return 0.5;  // undefined; neutral
    double u = sum_pos_ranks - static_cast<double>(n_pos) * (n_pos + 1) / 2.0;
    return u / (static_cast<double>(n_pos) * static_cast<double>(n_neg));
}

}  // namespace

TrainEngine::TrainEngine() : model_(nullptr) {}

TrainEngine::~TrainEngine() {
    stop_.store(true);
    {
        std::lock_guard<std::mutex> lk(ctrl_mutex_);
        paused_ = false;
    }
    ctrl_cv_.notify_all();
    if (thread_.joinable()) thread_.join();
}

void TrainEngine::start(torch::Tensor X_train, std::vector<int> y_train,
                        torch::Tensor X_val, std::vector<int> y_val, Config cfg) {
    // Join any prior run.
    if (thread_.joinable()) {
        stop_.store(true);
        ctrl_cv_.notify_all();
        thread_.join();
    }
    stop_.store(false);
    cfg_ = cfg;
    X_train_ = X_train.to(torch::kCPU).contiguous().clone();
    X_val_ = X_val.to(torch::kCPU).contiguous().clone();
    y_train_ = std::move(y_train);
    y_val_ = std::move(y_val);

    // Initialize the snapshot so callers polling before the first epoch see sane
    // values (max_epochs, running, etc.).
    {
        std::lock_guard<std::mutex> lk(snap_mutex_);
        snap_ = TrainSnapshot{};
        snap_.max_epochs = cfg_.max_epochs;
        snap_.patience_left = cfg_.patience;
        snap_.running = true;
        snap_.done = false;
        snap_.lr = static_cast<float>(cfg_.lr);
    }

    running_.store(true);
    thread_ = std::thread([this] { run_loop(); });
}

void TrainEngine::pause() {
    std::lock_guard<std::mutex> lk(ctrl_mutex_);
    paused_ = true;
}

void TrainEngine::resume() {
    {
        std::lock_guard<std::mutex> lk(ctrl_mutex_);
        paused_ = false;
        step_budget_ = -1;
    }
    ctrl_cv_.notify_all();
}

void TrainEngine::request_stop() {
    stop_.store(true);
    {
        std::lock_guard<std::mutex> lk(ctrl_mutex_);
        paused_ = false;
    }
    ctrl_cv_.notify_all();
}

void TrainEngine::step_once() {
    {
        std::lock_guard<std::mutex> lk(ctrl_mutex_);
        step_budget_ = 1;
        paused_ = false;
    }
    ctrl_cv_.notify_all();
}

TrainSnapshot TrainEngine::snapshot() const {
    std::lock_guard<std::mutex> lk(snap_mutex_);
    TrainSnapshot copy = snap_;  // copies POD + vectors
    if (snap_.stage1_kernels.defined()) copy.stage1_kernels = snap_.stage1_kernels.clone();
    if (snap_.pinned_input.defined()) copy.pinned_input = snap_.pinned_input.clone();
    if (snap_.gradcam.defined()) copy.gradcam = snap_.gradcam.clone();
    return copy;
}

plcnn::PerLeadCNN TrainEngine::model() const { return model_; }
bool TrainEngine::is_running() const { return running_.load(); }

void TrainEngine::publish(const TrainSnapshot& s) {
    std::lock_guard<std::mutex> lk(snap_mutex_);
    snap_ = s;
    if (s.stage1_kernels.defined()) snap_.stage1_kernels = s.stage1_kernels.clone();
    if (s.pinned_input.defined()) snap_.pinned_input = s.pinned_input.clone();
    if (s.gradcam.defined()) snap_.gradcam = s.gradcam.clone();
}

void TrainEngine::run_loop() {
    torch::manual_seed(cfg_.seed);

    const torch::Device device(torch::kCPU);

    // Fresh model per run.
    model_ = plcnn::PerLeadCNN(12, {16, 32, 48}, {31, 21, 11}, 0.15, 2);
    model_->to(device);

    torch::optim::AdamW optimizer(
        model_->parameters(),
        torch::optim::AdamWOptions(cfg_.lr).weight_decay(cfg_.weight_decay));
    const double base_lr = cfg_.lr;

    const int64_t N = X_train_.size(0);
    const int64_t T = X_train_.size(2);

    // Snapshot working copy that we mutate then publish each epoch.
    TrainSnapshot s;
    s.max_epochs = cfg_.max_epochs;
    s.patience_left = cfg_.patience;
    s.running = true;
    s.best_epoch = -1;
    s.best_val_auroc = 0.0f;

    // RNGs.
    std::mt19937_64 rng(cfg_.seed);
    std::mt19937_64 aug_rng(cfg_.seed ^ 0x9E3779B97F4A7C15ULL);

    // Best checkpoint = CPU clone of best state_dict (by val AUROC).
    std::vector<torch::Tensor> best_params;  // cloned values aligned to parameters()

    int patience_left = cfg_.patience;

    augment::AugCfg aug_cfg;

    auto extract_stage1 = [&]() -> torch::Tensor {
        // backbone[0].weight: (16,1,31) -> squeeze in-channel dim -> (16,31).
        torch::NoGradGuard ng;
        auto seq = model_->backbone;
        auto conv = seq->ptr<torch::nn::Conv1dImpl>(0);
        return conv->weight.detach().squeeze(1).clone();
    };

    // Pin a held-out positive val sample (fallback: first val sample) so the
    // saliency view follows the same waveform from noise -> structure.
    int pinned_idx = 0;
    for (size_t i = 0; i < y_val_.size(); ++i) {
        if (y_val_[i] == 1) { pinned_idx = static_cast<int>(i); break; }
    }
    const torch::Tensor pinned_input =
        (X_val_.size(0) > 0) ? X_val_[pinned_idx].clone() : torch::Tensor();
    const int pinned_label = y_val_.empty() ? -1 : y_val_[pinned_idx];

    // Grad-cam over the last backbone conv activation for the pinned sample,
    // class 1. Returns (12, T'); runs under the current (eval) weights.
    auto compute_gradcam = [&]() -> torch::Tensor {
        if (!pinned_input.defined()) return torch::Tensor();
        const int L = 12;
        auto xr = pinned_input.reshape({L, 1, pinned_input.size(-1)});  // (12,1,T)
        torch::Tensor A = model_->backbone->forward(xr).detach().set_requires_grad(true);
        auto pooled = model_->pool->forward(A).squeeze(-1);     // (12,48)
        auto flat = pooled.reshape({1, L * pooled.size(-1)});   // (1,576)
        auto logits = model_->fc->forward(flat);               // (1,2) (dropout=identity in eval)
        auto score = logits.select(1, 1).sum();
        model_->zero_grad();
        score.backward();
        auto grad = A.grad();                                  // (12,48,T')
        auto weights = grad.mean(2, true);                     // (12,48,1)
        auto cam = torch::relu((weights * A).sum(1));          // (12,T')
        model_->zero_grad();
        return cam.detach().clone();
    };

    if (pinned_input.defined()) {
        s.pinned_input = pinned_input;
        s.pinned_label = pinned_label;
    }

    for (int epoch = 0; epoch < cfg_.max_epochs; ++epoch) {
        if (stop_.load()) break;

        // Honor pause / single-step controls between epochs.
        {
            std::unique_lock<std::mutex> lk(ctrl_mutex_);
            ctrl_cv_.wait(lk, [this] { return !paused_ || stop_.load(); });
            if (step_budget_ == 0) {
                paused_ = true;
                ctrl_cv_.wait(lk, [this] { return !paused_ || stop_.load(); });
            }
            if (step_budget_ > 0) --step_budget_;
        }
        if (stop_.load()) break;

        // ---- TRAIN ----
        model_->train();

        // Per-epoch shuffle.
        std::vector<int64_t> order(N);
        std::iota(order.begin(), order.end(), 0);
        std::shuffle(order.begin(), order.end(), rng);

        std::uniform_real_distribution<double> uni(0.0, 1.0);
        // Beta(a,a) via two Gamma(a) draws.
        std::gamma_distribution<double> gamma_a(0.2, 1.0);

        double loss_sum = 0.0;
        int loss_batches = 0;
        int accum_counter = 0;
        optimizer.zero_grad();

        const int bs = cfg_.batch_size;
        for (int64_t start = 0; start < N; start += bs) {
            if (stop_.load()) break;
            int64_t end = std::min<int64_t>(start + bs, N);
            int64_t b = end - start;

            // Build batch tensor (b,12,T), applying augmentation per-sample.
            torch::Tensor xb = torch::empty({b, 12, T}, torch::kFloat32);
            std::vector<int64_t> yb_vec(b);
            for (int64_t i = 0; i < b; ++i) {
                int64_t gi = order[start + i];
                torch::Tensor xi = X_train_[gi];  // (12,T) view
                if (cfg_.augment) {
                    xb[i] = augment::augment_ecg(xi, aug_cfg, aug_rng);
                } else {
                    xb[i] = xi.clone();
                }
                yb_vec[i] = y_train_[gi];
            }
            torch::Tensor yb = torch::tensor(yb_vec, torch::kLong);

            // Mixup on 50% of batches.
            bool use_mixup = cfg_.mixup && (uni(rng) < 0.5);
            torch::Tensor loss;
            if (use_mixup) {
                double g1 = gamma_a(rng), g2 = gamma_a(rng);
                double lam = (g1 + g2) > 0 ? g1 / (g1 + g2) : 0.5;
                torch::Tensor perm = torch::randperm(b, torch::kLong);
                torch::Tensor xb2 = xb.index_select(0, perm);
                torch::Tensor x_mix = lam * xb + (1.0 - lam) * xb2;
                torch::Tensor yb2 = yb.index_select(0, perm);
                torch::Tensor logits = model_->forward(x_mix);
                loss = lam * focal_loss(logits, yb) +
                       (1.0 - lam) * focal_loss(logits, yb2);
            } else {
                torch::Tensor logits = model_->forward(xb);
                loss = focal_loss(logits, yb);
            }

            // Grad-accum: scale by 1/accum; step every `accum` batches.
            torch::Tensor scaled = loss / static_cast<double>(cfg_.accum);
            scaled.backward();
            loss_sum += loss.item<double>();
            ++loss_batches;
            ++accum_counter;
            if (accum_counter % cfg_.accum == 0) {
                optimizer.step();
                optimizer.zero_grad();
            }

            // Intra-epoch train_loss update.
            {
                std::lock_guard<std::mutex> lk(snap_mutex_);
                snap_.train_loss =
                    static_cast<float>(loss_sum / std::max(1, loss_batches));
            }
        }
        // Flush any leftover accumulated grads.
        if (accum_counter % cfg_.accum != 0) {
            optimizer.step();
            optimizer.zero_grad();
        }
        if (stop_.load()) break;

        float epoch_loss = static_cast<float>(loss_sum / std::max(1, loss_batches));

        // ---- LR schedule (cosine annealing), set after computing for `epoch`.
        double lr = cfg_.eta_min +
                    0.5 * (base_lr - cfg_.eta_min) *
                        (1.0 + std::cos(M_PI * static_cast<double>(epoch) /
                                        static_cast<double>(cfg_.T_max)));
        for (auto& group : optimizer.param_groups()) {
            static_cast<torch::optim::AdamWOptions&>(group.options()).lr(lr);
        }

        // ---- VALIDATE (AUROC) ----
        model_->eval();
        std::vector<float> scores;
        std::vector<int> labels = y_val_;
        {
            torch::NoGradGuard ng;
            const int64_t Nv = X_val_.size(0);
            scores.reserve(Nv);
            for (int64_t start = 0; start < Nv; start += bs) {
                int64_t end = std::min<int64_t>(start + bs, Nv);
                torch::Tensor xb = X_val_.slice(0, start, end);
                torch::Tensor logits = model_->forward(xb);  // (b,2)
                torch::Tensor prob = torch::softmax(logits, 1).select(1, 1);
                for (int64_t i = 0; i < prob.size(0); ++i) {
                    scores.push_back(prob[i].item<float>());
                }
            }
        }
        float val_auroc = static_cast<float>(auroc_mann_whitney(scores, labels));

        // ---- Bookkeeping / early stopping ----
        s.epoch = epoch + 1;
        s.train_loss = epoch_loss;
        s.val_auroc = val_auroc;
        s.lr = static_cast<float>(lr);
        s.loss_history.push_back(epoch_loss);
        s.auroc_history.push_back(val_auroc);

        if (val_auroc > s.best_val_auroc) {
            s.best_val_auroc = val_auroc;
            s.best_epoch = epoch;
            patience_left = cfg_.patience;
            // Save best checkpoint (CPU clone of params).
            torch::NoGradGuard ng;
            best_params.clear();
            for (const auto& p : model_->parameters()) {
                best_params.push_back(p.detach().to(torch::kCPU).clone());
            }
        } else {
            --patience_left;
        }
        s.patience_left = patience_left;
        s.stage1_kernels = extract_stage1();
        s.gradcam = compute_gradcam();

        publish(s);

        if (patience_left <= 0) break;
    }

    // Restore best checkpoint into the live model so model() returns the best.
    if (!best_params.empty()) {
        torch::NoGradGuard ng;
        auto params = model_->parameters();
        for (size_t i = 0; i < params.size() && i < best_params.size(); ++i) {
            params[i].copy_(best_params[i]);
        }
    }

    // Final publish: done.
    s.running = false;
    s.done = true;
    s.stage1_kernels = extract_stage1();
    s.gradcam = compute_gradcam();
    publish(s);
    running_.store(false);
}
