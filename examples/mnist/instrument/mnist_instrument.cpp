// Canonical ML instrument: the PyTorch C++ MNIST CNN trained live with LibTorch.
//
// Model, optimizer, loss, and dataset pipeline follow the official PyTorch C++
// LibTorch MNIST example:
//   https://github.com/pytorch/examples/blob/main/cpp/mnist/mnist.cpp
//
// Sections marked with `// --8<-- [start:<name>]` are embedded verbatim in the
// documentation through pymdownx.snippets.
#include <caliper/caliper.hpp>
#include <caliper/adapters/torch.hpp>
#include <torch/torch.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <filesystem>
#include <memory>
#include <mutex>
#include <random>
#include <string>
#include <system_error>
#include <thread>
#include <vector>

namespace mnist_instrument {

namespace F = torch::nn::functional;

namespace {

// Published hyperparameters from pytorch/examples/cpp/mnist/mnist.cpp.
constexpr int64_t kTrainBatchSize = 64;
constexpr int64_t kTestBatchSize  = 1000;
constexpr int64_t kNumberOfEpochs = 10;
constexpr double  kLearningRate   = 0.01;
constexpr double  kMomentum       = 0.5;
constexpr double  kMnistMean      = 0.1307;
constexpr double  kMnistStd       = 0.3081;

// Live visualization settings matching examples/mnist/train.py.
constexpr int64_t kProbeSize  = 512;
constexpr int64_t kProbeEvery = 10;

// Mosaic dimensions: 10 conv1 filters (5x5 -> 40x40 tiles) and 10 conv1
// post-ReLU feature maps (24x24 -> 48x48 tiles), arranged 2 rows x 5 columns.
constexpr int kTileRows   = 2;
constexpr int kTileCols   = 5;
constexpr int kFilterScale = 8;
constexpr int kActScale    = 2;
constexpr int kTilePad     = 4;

// Tab10 palette for the 10 MNIST digit classes (0..9).
constexpr ImU32 kClassColors[10] = {
    IM_COL32( 31, 119, 180, 255),
    IM_COL32(255, 127,  14, 255),
    IM_COL32( 44, 160,  44, 255),
    IM_COL32(214,  39,  40, 255),
    IM_COL32(148, 103, 189, 255),
    IM_COL32(140,  86,  75, 255),
    IM_COL32(227, 119, 194, 255),
    IM_COL32(127, 127, 127, 255),
    IM_COL32(188, 189,  34, 255),
    IM_COL32( 23, 190, 207, 255),
};

std::string read_env(const char* name) {
#ifdef _WIN32
    char* value = nullptr;
    size_t len = 0;
    if (_dupenv_s(&value, &len, name) != 0 || !value) return {};
    std::string out(value);
    std::free(value);
    return out;
#else
    const char* value = std::getenv(name);
    return value ? std::string(value) : std::string();
#endif
}

bool has_mnist_files(const std::filesystem::path& dir) {
    namespace fs = std::filesystem;
    static constexpr const char* kRequired[4] = {
        "train-images-idx3-ubyte",
        "train-labels-idx1-ubyte",
        "t10k-images-idx3-ubyte",
        "t10k-labels-idx1-ubyte",
    };
    std::error_code ec;
    if (!fs::is_directory(dir, ec)) return false;
    for (const char* file : kRequired) {
        if (!fs::is_regular_file(dir / file, ec)) return false;
    }
    return true;
}

// Look up the cached MNIST IDX files downloaded by examples/mnist/train.py.
std::string find_mnist_dir() {
    namespace fs = std::filesystem;
    std::vector<fs::path> candidates;
    if (std::string env_dir = read_env("CALIPER_MNIST_DATA_DIR"); !env_dir.empty()) {
        fs::path root(env_dir);
        candidates.push_back(root / "MNIST" / "raw");
        candidates.push_back(root);
    }
    if (std::string home = read_env("HOME"); !home.empty()) {
        fs::path root = fs::path(home) / ".cache" / "caliper" / "mnist";
        candidates.push_back(root / "MNIST" / "raw");
        candidates.push_back(root);
    }
#ifdef _WIN32
    if (std::string profile = read_env("USERPROFILE"); !profile.empty()) {
        fs::path root = fs::path(profile) / ".cache" / "caliper" / "mnist";
        candidates.push_back(root / "MNIST" / "raw");
        candidates.push_back(root);
    }
#endif
    for (const fs::path& candidate : candidates) {
        if (has_mnist_files(candidate)) return candidate.string();
    }
    return {};
}

std::string expected_cache_path() {
    namespace fs = std::filesystem;
    if (std::string env_dir = read_env("CALIPER_MNIST_DATA_DIR"); !env_dir.empty())
        return (fs::path(env_dir) / "MNIST" / "raw").string();
    if (std::string home = read_env("HOME"); !home.empty())
        return (fs::path(home) / ".cache" / "caliper" / "mnist" / "MNIST" / "raw").string();
#ifdef _WIN32
    if (std::string profile = read_env("USERPROFILE"); !profile.empty())
        return (fs::path(profile) / ".cache" / "caliper" / "mnist" / "MNIST" / "raw").string();
#endif
    return "~/.cache/caliper/mnist/MNIST/raw";
}

// Fixed seeded orthonormal (in_dim, out_dim) projection built on CPU with
// modified Gram-Schmidt so it does not depend on LAPACK or MPS QR kernels.
torch::Tensor make_orthonormal_projection(int64_t in_dim, int64_t out_dim) {
    std::mt19937 rng(0);
    std::normal_distribution<float> dist(0.0f, 1.0f);
    std::vector<float> mat(static_cast<size_t>(in_dim * out_dim), 0.0f);
    for (int64_t c = 0; c < out_dim; ++c) {
        for (int64_t r = 0; r < in_dim; ++r)
            mat[static_cast<size_t>(r * out_dim + c)] = dist(rng);
        for (int64_t prev = 0; prev < c; ++prev) {
            double dot = 0.0;
            for (int64_t r = 0; r < in_dim; ++r) {
                dot += static_cast<double>(mat[static_cast<size_t>(r * out_dim + c)]) *
                       static_cast<double>(mat[static_cast<size_t>(r * out_dim + prev)]);
            }
            for (int64_t r = 0; r < in_dim; ++r) {
                mat[static_cast<size_t>(r * out_dim + c)] -=
                    static_cast<float>(dot) * mat[static_cast<size_t>(r * out_dim + prev)];
            }
        }
        double norm2 = 0.0;
        for (int64_t r = 0; r < in_dim; ++r) {
            const double v = mat[static_cast<size_t>(r * out_dim + c)];
            norm2 += v * v;
        }
        const float inv = norm2 > 1e-12 ? static_cast<float>(1.0 / std::sqrt(norm2)) : 0.0f;
        for (int64_t r = 0; r < in_dim; ++r)
            mat[static_cast<size_t>(r * out_dim + c)] *= inv;
    }
    return torch::from_blob(mat.data(), {in_dim, out_dim}, torch::kFloat32).clone();
}

// Tile (C, H, W) CPU float32 maps into a 2-D mosaic with integer block scaling
// and a fixed pad value between tiles.
torch::Tensor build_mosaic(const torch::Tensor& maps_cpu, int rows, int cols,
                           int scale, int pad, float pad_val) {
    const int64_t count = maps_cpu.size(0);
    const int64_t h = maps_cpu.size(1);
    const int64_t w = maps_cpu.size(2);
    const int64_t tile_h = h * scale;
    const int64_t tile_w = w * scale;
    const int64_t out_h = rows * tile_h + (rows - 1) * pad;
    const int64_t out_w = cols * tile_w + (cols - 1) * pad;

    auto out = torch::full({out_h, out_w}, pad_val, torch::kFloat32);
    const float* src = maps_cpu.data_ptr<float>();
    float* dst = out.data_ptr<float>();

    for (int r = 0; r < rows; ++r) {
        for (int c = 0; c < cols; ++c) {
            const int64_t idx = static_cast<int64_t>(r * cols + c);
            if (idx >= count) continue;
            const float* tile = src + idx * h * w;
            const int64_t y0 = static_cast<int64_t>(r) * (tile_h + pad);
            const int64_t x0 = static_cast<int64_t>(c) * (tile_w + pad);
            for (int64_t y = 0; y < h; ++y) {
                for (int64_t x = 0; x < w; ++x) {
                    const float val = tile[y * w + x];
                    for (int sy = 0; sy < scale; ++sy) {
                        float* row_ptr = dst + (y0 + y * scale + sy) * out_w + x0 + x * scale;
                        for (int sx = 0; sx < scale; ++sx) row_ptr[sx] = val;
                    }
                }
            }
        }
    }
    return out;
}

} // namespace

// --8<-- [start:model]
// Official PyTorch C++ LibTorch MNIST CNN:
// https://github.com/pytorch/examples/blob/main/cpp/mnist/mnist.cpp
struct NetImpl : torch::nn::Module {
    torch::nn::Conv2d conv1{nullptr};
    torch::nn::Conv2d conv2{nullptr};
    torch::nn::Dropout2d conv2_drop{nullptr};
    torch::nn::Linear fc1{nullptr};
    torch::nn::Linear fc2{nullptr};

    NetImpl() {
        conv1 = register_module(
            "conv1", torch::nn::Conv2d(torch::nn::Conv2dOptions(1, 10, /*kernel_size=*/5)));
        conv2 = register_module(
            "conv2", torch::nn::Conv2d(torch::nn::Conv2dOptions(10, 20, /*kernel_size=*/5)));
        conv2_drop = register_module("conv2_drop", torch::nn::Dropout2d());
        fc1 = register_module("fc1", torch::nn::Linear(320, 50));
        fc2 = register_module("fc2", torch::nn::Linear(50, 10));
    }

    torch::Tensor forward(torch::Tensor x) {
        x = torch::relu(torch::max_pool2d(conv1->forward(x), 2));
        x = torch::relu(
            torch::max_pool2d(conv2_drop->forward(conv2->forward(x)), 2));
        x = x.reshape({-1, 320});
        x = torch::relu(fc1->forward(x));
        x = torch::dropout(x, /*p=*/0.5, /*training=*/is_training());
        x = fc2->forward(x);
        return torch::log_softmax(x, /*dim=*/1);
    }

    // Post-ReLU conv1 feature maps for live activation inspection.
    torch::Tensor conv1_activations(const torch::Tensor& x) {
        return torch::relu(conv1->forward(x));
    }

    // Post-ReLU fc1 hidden representation (50-D) for 3-D embedding projection.
    torch::Tensor hidden(torch::Tensor x) {
        x = torch::relu(torch::max_pool2d(conv1->forward(x), 2));
        x = torch::relu(
            torch::max_pool2d(conv2_drop->forward(conv2->forward(x)), 2));
        x = x.reshape({-1, 320});
        return torch::relu(fc1->forward(x));
    }
};
TORCH_MODULE(Net);
// --8<-- [end:model]

// --8<-- [start:state]
struct MnistState {
    caliper::Host*   host = nullptr;
    caliper::Jobs    jobs;      // required (caliper.jobs.v1)
    caliper::Device  device;    // required (caliper.device.v1)
    caliper::Metrics metrics;   // optional (caliper.metrics.v1)
    caliper::Bridge  bridge;    // optional (caliper.tensor_bridge.v1)

    uint64_t job_id = 0;
    std::string data_dir;
    bool dataset_found = false;
    std::atomic<bool> paused{false};

    // Cross-thread snapshots, guarded by mtx.
    std::mutex mtx;
    std::string device_label = "CPU";
    std::string status = "ready";
    int epoch = 0;
    int64_t step = 0;
    float last_loss = 0.0f;
    float last_acc = 0.0f;
    float test_acc = -1.0f;
    int probe_digit0_label = -1;

    std::vector<float> loss_hist;
    std::vector<float> acc_hist;
    std::vector<float> test_acc_steps;
    std::vector<float> test_acc_hist;

    torch::Tensor filter_mosaic;        // (84, 216) f32 CPU, normalized to [-1, 1]
    torch::Tensor act_mosaic;           // (100, 256) f32 CPU, normalized to [0, 1]
    float filter_abs_max = 0.0f;
    float act_max = 0.0f;
    std::vector<float> fc1_weights;     // flattened fc1.weight (16000 values)
    std::vector<float> emb_x, emb_y, emb_z;
    std::vector<int>   emb_labels;
    uint64_t filter_gen = 0;
    uint64_t probe_gen = 0;

    // Frame-thread-only state.
    uint64_t filter_tex_gen = 0;
    uint64_t act_tex_gen = 0;
    uint64_t emb_plot_gen = 0;
    CaliperTextureId filter_tex = 0;
    CaliperTextureId act_tex = 0;
    std::array<std::vector<float>, 10> cls_x, cls_y, cls_z;
    double emb_min[3] = {-1.0, -1.0, -1.0};
    double emb_max[3] = {1.0, 1.0, 1.0};
    bool emb_refit = false;
    bool emb_autofit = true;
    bool follow = true;
    bool restart_pending = false;
};
// --8<-- [end:state]

namespace {

void set_status(MnistState* st, const std::string& msg) {
    std::lock_guard<std::mutex> lk(st->mtx);
    st->status = msg;
}

// Verify that the installed LibTorch supports every op of Net on `candidate`.
// Falls back cleanly to CPU when an op is unsupported on the device.
torch::Device select_torch_device(const MnistState& st, std::string* label_out) {
    auto d = caliper::Device::query(*st.host);
    torch::Device candidate(torch::kCPU);
    std::string label = "CPU";

    if (d.kind == CALIPER_DEV_CUDA && torch::cuda::is_available()) {
        candidate = torch::Device(torch::kCUDA, static_cast<c10::DeviceIndex>(d.index));
        label = std::string("CUDA (") + (d.name && *d.name ? d.name : "GPU") + ")";
    } else if (d.kind == CALIPER_DEV_METAL && torch::mps::is_available()) {
        candidate = torch::Device(torch::kMPS);
        label = std::string("MPS (") + (d.name && *d.name ? d.name : "Apple GPU") + ")";
    }

    if (!candidate.is_cpu()) {
        try {
            Net probe;
            probe->to(candidate);
            probe->train();
            torch::optim::SGD opt(
                probe->parameters(),
                torch::optim::SGDOptions(kLearningRate).momentum(kMomentum));
            auto x = torch::zeros({2, 1, 28, 28}, torch::TensorOptions(candidate).dtype(torch::kFloat32));
            auto y = torch::zeros({2}, torch::TensorOptions(candidate).dtype(torch::kInt64));
            opt.zero_grad();
            auto out = probe->forward(x);
            auto loss = F::nll_loss(out, y);
            loss.backward();
            opt.step();
            if (candidate.is_mps()) torch::mps::synchronize();
            else if (candidate.is_cuda()) torch::cuda::synchronize();
        } catch (const std::exception& e) {
            if (st.host) {
                std::string msg = std::string("mnist: device ") + label +
                                  " unsupported by installed LibTorch ops, using CPU: " + e.what();
                st.host->log_info(msg.c_str());
            }
            candidate = torch::Device(torch::kCPU);
            label = "CPU (fallback)";
        }
    }
    if (label_out) *label_out = label;
    return candidate;
}

// --8<-- [start:publish]
// Worker side: stage CPU copies outside the lock, then swap them in under mtx.
void publish_step(MnistState* st, Net& model, int epoch, int64_t step,
                  float loss, float batch_acc_pct, uint64_t run_id) {
    torch::NoGradGuard ng;
    auto w = model->conv1->weight.detach().squeeze(1).to(torch::kCPU).to(torch::kFloat32).contiguous();
    const float wmax = std::max(w.abs().max().item<float>(), 1e-6f);
    auto mosaic = build_mosaic(w / wmax, kTileRows, kTileCols, kFilterScale, kTilePad, 0.0f);

    char msg[128];
    std::snprintf(msg, sizeof msg, "epoch %d/%lld  step %lld  loss %.4f  acc %.1f%%",
                  epoch, static_cast<long long>(kNumberOfEpochs),
                  static_cast<long long>(step), loss, batch_acc_pct);
    {
        std::lock_guard<std::mutex> lk(st->mtx);
        st->epoch = epoch;
        st->step = step;
        st->last_loss = loss;
        st->last_acc = batch_acc_pct;
        st->status = msg;
        st->loss_hist.push_back(loss);
        st->acc_hist.push_back(batch_acc_pct);
        st->filter_mosaic = std::move(mosaic);
        st->filter_abs_max = wmax;
        st->filter_gen++;
    }
    if (run_id != 0) {
        st->metrics.scalar(run_id, "train/loss", step, loss);
        st->metrics.scalar(run_id, "train/accuracy", step, batch_acc_pct);
    }
}

void publish_probe(MnistState* st, Net& model, const torch::Tensor& probe_x,
                   const torch::Tensor& probe_y_cpu, const torch::Tensor& proj,
                   int64_t step, uint64_t run_id) {
    const bool was_training = model->is_training();
    model->eval();
    torch::Tensor act_mosaic;
    float act_max = 0.0f;
    std::vector<float> ex, ey, ez, fc1_w;
    std::vector<int> labels;
    int digit0_label = -1;
    {
        torch::NoGradGuard ng;
        // Post-ReLU conv1 activations on fixed test digit 0.
        auto act0 = model->conv1_activations(probe_x.slice(0, 0, 1))[0]
                        .to(torch::kCPU).to(torch::kFloat32).contiguous();
        act_max = std::max(act0.max().item<float>(), 1e-6f);
        act_mosaic = build_mosaic(act0 / act_max, kTileRows, kTileCols, kActScale, kTilePad, 0.0f);

        // 50-D fc1 hidden activations on 512 fixed test digits, projected to 3-D.
        auto emb = torch::matmul(model->hidden(probe_x), proj)
                       .to(torch::kCPU).to(torch::kFloat32).contiguous();
        const int64_t n = emb.size(0);
        const float* ep = emb.data_ptr<float>();
        const int64_t* lp = probe_y_cpu.data_ptr<int64_t>();
        ex.resize(static_cast<size_t>(n));
        ey.resize(static_cast<size_t>(n));
        ez.resize(static_cast<size_t>(n));
        labels.resize(static_cast<size_t>(n));
        for (int64_t i = 0; i < n; ++i) {
            ex[static_cast<size_t>(i)] = ep[i * 3 + 0];
            ey[static_cast<size_t>(i)] = ep[i * 3 + 1];
            ez[static_cast<size_t>(i)] = ep[i * 3 + 2];
            labels[static_cast<size_t>(i)] = static_cast<int>(lp[i]);
        }
        if (n > 0) digit0_label = static_cast<int>(lp[0]);

        // Flattened fc1 weights for the histogram view.
        auto w = model->fc1->weight.detach().to(torch::kCPU).to(torch::kFloat32).contiguous();
        const float* wp = w.data_ptr<float>();
        fc1_w.assign(wp, wp + w.numel());
    }
    if (was_training) model->train();

    if (run_id != 0 && !fc1_w.empty())
        st->metrics.histogram(run_id, "fc1/weight", step, fc1_w.data(),
                              static_cast<int64_t>(fc1_w.size()));

    std::lock_guard<std::mutex> lk(st->mtx);
    st->act_mosaic = std::move(act_mosaic);
    st->act_max = act_max;
    st->emb_x = std::move(ex);
    st->emb_y = std::move(ey);
    st->emb_z = std::move(ez);
    st->emb_labels = std::move(labels);
    st->probe_digit0_label = digit0_label;
    st->fc1_weights = std::move(fc1_w);
    st->probe_gen++;
}
// --8<-- [end:publish]

// --8<-- [start:job]
// Background training job (caliper.jobs.v1). Polls ctl->cancelled every step.
void train_job(void* user, const CaliperJobControl* ctl) {
    auto* st = static_cast<MnistState*>(user);
    if (!ctl || ctl->cancelled(ctl)) return;

    struct RunGuard {
        caliper::Metrics* metrics = nullptr;
        uint64_t run_id = 0;
        ~RunGuard() {
            if (metrics && run_id != 0) metrics->end_run(run_id);
        }
    };

    try {
        std::string dev_label;
        const torch::Device dev = select_torch_device(*st, &dev_label);
        {
            std::lock_guard<std::mutex> lk(st->mtx);
            st->device_label = dev_label;
            st->status = "loading MNIST dataset...";
        }
        if (ctl->cancelled(ctl)) return;

        // Official PyTorch C++ MNIST dataset pipeline:
        // https://github.com/pytorch/examples/blob/main/cpp/mnist/mnist.cpp
        auto train_dataset =
            torch::data::datasets::MNIST(st->data_dir, torch::data::datasets::MNIST::Mode::kTrain)
                .map(torch::data::transforms::Normalize<>(kMnistMean, kMnistStd))
                .map(torch::data::transforms::Stack<>());
        const size_t train_size = train_dataset.size().value();
        auto train_loader =
            torch::data::make_data_loader<torch::data::samplers::RandomSampler>(
                std::move(train_dataset), kTrainBatchSize);

        auto test_dataset =
            torch::data::datasets::MNIST(st->data_dir, torch::data::datasets::MNIST::Mode::kTest)
                .map(torch::data::transforms::Normalize<>(kMnistMean, kMnistStd))
                .map(torch::data::transforms::Stack<>());
        const size_t test_size = test_dataset.size().value();

        // Build the fixed 512-digit test probe (item 0 is the fixed digit for conv1).
        auto probe_dataset =
            torch::data::datasets::MNIST(st->data_dir, torch::data::datasets::MNIST::Mode::kTest)
                .map(torch::data::transforms::Normalize<>(kMnistMean, kMnistStd))
                .map(torch::data::transforms::Stack<>());
        auto probe_loader =
            torch::data::make_data_loader<torch::data::samplers::SequentialSampler>(
                std::move(probe_dataset), static_cast<size_t>(kProbeSize));

        torch::Tensor probe_x, probe_y_cpu;
        for (auto& batch : *probe_loader) {
            probe_x = batch.data.to(dev);
            probe_y_cpu = batch.target.to(torch::kCPU).to(torch::kInt64).contiguous();
            break;
        }
        auto test_loader =
            torch::data::make_data_loader<torch::data::samplers::SequentialSampler>(
                std::move(test_dataset), kTestBatchSize);

        if (ctl->cancelled(ctl)) return;

        torch::manual_seed(1);
        Net model;
        model->to(dev);
        torch::optim::SGD optimizer(
            model->parameters(),
            torch::optim::SGDOptions(kLearningRate).momentum(kMomentum));
        const torch::Tensor proj = make_orthonormal_projection(50, 3).to(dev);

        RunGuard run_guard{&st->metrics, 0};
        if (st->metrics) {
            run_guard.run_id = st->metrics.begin_run("mnist", "libtorch-cnn");
            if (run_guard.run_id != 0) {
                st->metrics.hparams_json(
                    run_guard.run_id,
                    R"({"optimizer":"SGD","lr":0.01,"momentum":0.5,"batch_size":64,"epochs":10})");
            }
        }

        // Initial probe snapshot before step 1 so all views populate immediately.
        publish_probe(st, model, probe_x, probe_y_cpu, proj, 0, run_guard.run_id);
        {
            torch::NoGradGuard ng;
            auto w = model->conv1->weight.detach().squeeze(1).to(torch::kCPU).to(torch::kFloat32).contiguous();
            const float wmax = std::max(w.abs().max().item<float>(), 1e-6f);
            auto mosaic = build_mosaic(w / wmax, kTileRows, kTileCols, kFilterScale, kTilePad, 0.0f);
            std::lock_guard<std::mutex> lk(st->mtx);
            st->filter_mosaic = std::move(mosaic);
            st->filter_abs_max = wmax;
            st->filter_gen++;
        }

        const int64_t batches_per_epoch =
            static_cast<int64_t>((train_size + kTrainBatchSize - 1) / kTrainBatchSize);
        const int64_t total_steps = batches_per_epoch * kNumberOfEpochs;
        int64_t step = 0;

        for (int epoch = 1; epoch <= kNumberOfEpochs; ++epoch) {
            model->train();
            for (auto& batch : *train_loader) {
                while (st->paused.load(std::memory_order_relaxed)) {
                    if (ctl->cancelled(ctl)) return;
                    std::this_thread::sleep_for(std::chrono::milliseconds(20));
                }
                if (ctl->cancelled(ctl)) return;

                auto data = batch.data.to(dev);
                auto targets = batch.target.to(dev);
                optimizer.zero_grad();
                auto output = model->forward(data);
                auto loss = F::nll_loss(output, targets);
                loss.backward();
                optimizer.step();

                const float loss_val = loss.item<float>();
                const float batch_acc_pct =
                    100.0f * output.argmax(1).eq(targets).to(torch::kFloat32).mean().item<float>();
                ++step;

                publish_step(st, model, epoch, step, loss_val, batch_acc_pct, run_guard.run_id);
                if (step == 1 || step % kProbeEvery == 0)
                    publish_probe(st, model, probe_x, probe_y_cpu, proj, step, run_guard.run_id);

                char prog[96];
                std::snprintf(prog, sizeof prog, "epoch %d/%lld  loss %.4f",
                              epoch, static_cast<long long>(kNumberOfEpochs), loss_val);
                ctl->progress(ctl, std::min(1.0f, static_cast<float>(step) / static_cast<float>(total_steps)), prog);
            }

            // Full test split evaluation at the end of each epoch.
            model->eval();
            double test_loss_sum = 0.0;
            int64_t correct = 0;
            {
                torch::NoGradGuard ng;
                for (const auto& batch : *test_loader) {
                    if (ctl->cancelled(ctl)) return;
                    auto data = batch.data.to(dev);
                    auto targets = batch.target.to(dev);
                    auto output = model->forward(data);
                    test_loss_sum +=
                        F::nll_loss(output, targets, F::NLLLossFuncOptions().reduction(torch::kSum))
                            .item<double>();
                    correct += output.argmax(1).eq(targets).sum().item<int64_t>();
                }
            }
            const double test_loss = test_loss_sum / static_cast<double>(std::max<size_t>(test_size, 1));
            const float test_acc_pct =
                100.0f * static_cast<float>(correct) / static_cast<float>(std::max<size_t>(test_size, 1));
            {
                std::lock_guard<std::mutex> lk(st->mtx);
                st->test_acc = test_acc_pct;
                st->test_acc_steps.push_back(static_cast<float>(step));
                st->test_acc_hist.push_back(test_acc_pct);
            }
            if (run_guard.run_id != 0) {
                st->metrics.scalar(run_guard.run_id, "test/loss", step, test_loss);
                st->metrics.scalar(run_guard.run_id, "test/accuracy", step, test_acc_pct);
            }
            publish_probe(st, model, probe_x, probe_y_cpu, proj, step, run_guard.run_id);
        }

        set_status(st, "training complete");
        ctl->progress(ctl, 1.0f, "training complete");
    } catch (const std::exception& e) {
        std::string msg = std::string("mnist training error: ") + e.what();
        set_status(st, msg);
        if (st->host) st->host->log_error(msg.c_str());
    }
}
// --8<-- [end:job]

void update_mapped_texture(caliper::Bridge& bridge, CaliperTextureId& tex,
                           const torch::Tensor& cpu_f32, int32_t cmap,
                           float vmin, float vmax) {
    if (!bridge || !cpu_f32.defined()) return;
    auto ct = caliper::adapters::to_tensor(cpu_f32);
    if (!ct) return;
    if (tex && bridge.update_texture(tex, &*ct)) return;
    if (tex) {
        bridge.release_texture(tex);
        tex = 0;
    }
    tex = bridge.texture_from_tensor_mapped(&*ct, cmap, vmin, vmax);
}

void draw_heatmap_fallback(const char* plot_id, const torch::Tensor& mosaic,
                           ImPlotColormap cmap, double vmin, double vmax, float height) {
    if (!mosaic.defined() || mosaic.dim() != 2) {
        ImGui::TextDisabled("waiting for first step...");
        return;
    }
    ImPlot::PushColormap(cmap);
    if (ImPlot::BeginPlot(plot_id, ImVec2(-1, height),
                          ImPlotFlags_NoLegend | ImPlotFlags_NoMouseText)) {
        ImPlot::SetupAxes(nullptr, nullptr,
                          ImPlotAxisFlags_NoDecorations, ImPlotAxisFlags_NoDecorations);
        ImPlot::PlotHeatmap("##hm", mosaic.data_ptr<float>(),
                            static_cast<int>(mosaic.size(0)),
                            static_cast<int>(mosaic.size(1)),
                            vmin, vmax, nullptr);
        ImPlot::EndPlot();
    }
    ImPlot::PopColormap();
}

} // namespace

class MnistInstrument final : public caliper::Instrument {
public:
    MnistInstrument() : s_(std::make_unique<MnistState>()) {}

    bool on_init(caliper::Host& host) override {
        host_ = &host;
        auto* st = s_.get();
        st->host    = &host;
        st->jobs    = caliper::Jobs(host);
        st->device  = caliper::Device::query(host);
        st->metrics = caliper::Metrics(host);
        st->bridge  = caliper::Bridge(host);

        if (st->device.kind == CALIPER_DEV_CUDA) {
            st->device_label = std::string("CUDA (") +
                               (st->device.name && *st->device.name ? st->device.name : "GPU") + ")";
        } else if (st->device.kind == CALIPER_DEV_METAL) {
            st->device_label = std::string("MPS (") +
                               (st->device.name && *st->device.name ? st->device.name : "Apple GPU") + ")";
        } else {
            st->device_label = "CPU";
        }

        st->data_dir = find_mnist_dir();
        st->dataset_found = !st->data_dir.empty();
        if (st->dataset_found) {
            st->status = "starting MNIST training...";
            st->job_id = st->jobs.submit("mnist: train CNN", &train_job, st);
        } else {
            st->status = "MNIST dataset not found in cache";
        }
        host.log_info("mnist.on_init");
        return true;
    }

    // --8<-- [start:views]
    void on_frame(const caliper::Frame&) override {
        auto* st = s_.get();

        const bool running = st->job_id != 0 && st->jobs.is_running(st->job_id);
        if (st->restart_pending && !running && st->dataset_found) {
            st->restart_pending = false;
            start_fresh_run();
        }

        // Snapshot cross-thread state under mtx once per frame.
        std::string device_label, status;
        int epoch = 0, digit0_label = -1;
        int64_t step = 0;
        float last_loss = 0.0f, last_acc = 0.0f, test_acc = -1.0f;
        float filter_abs_max = 0.0f, act_max = 0.0f;
        std::vector<float> loss_hist, acc_hist, test_acc_steps, test_acc_hist;
        std::vector<float> fc1_weights, emb_x, emb_y, emb_z;
        std::vector<int> emb_labels;
        torch::Tensor filter_mosaic, act_mosaic;
        uint64_t filter_gen = 0, probe_gen = 0;
        {
            std::lock_guard<std::mutex> lk(st->mtx);
            device_label   = st->device_label;
            status         = st->status;
            epoch          = st->epoch;
            step           = st->step;
            last_loss      = st->last_loss;
            last_acc       = st->last_acc;
            test_acc       = st->test_acc;
            digit0_label   = st->probe_digit0_label;
            loss_hist      = st->loss_hist;
            acc_hist       = st->acc_hist;
            test_acc_steps = st->test_acc_steps;
            test_acc_hist  = st->test_acc_hist;
            filter_mosaic  = st->filter_mosaic;
            act_mosaic     = st->act_mosaic;
            filter_abs_max = st->filter_abs_max;
            act_max        = st->act_max;
            fc1_weights    = st->fc1_weights;
            filter_gen     = st->filter_gen;
            probe_gen      = st->probe_gen;
            if (probe_gen != 0 && probe_gen != st->emb_plot_gen) {
                emb_x      = st->emb_x;
                emb_y      = st->emb_y;
                emb_z      = st->emb_z;
                emb_labels = st->emb_labels;
            }
        }

        // Frame-thread texture uploads via caliper.tensor_bridge.v1.
        if (st->bridge && filter_gen != 0 && filter_gen != st->filter_tex_gen &&
            filter_mosaic.defined()) {
            update_mapped_texture(st->bridge, st->filter_tex, filter_mosaic,
                                  CALIPER_CMAP_RDBU, -1.0f, 1.0f);
            st->filter_tex_gen = filter_gen;
        }
        if (st->bridge && probe_gen != 0 && probe_gen != st->act_tex_gen &&
            act_mosaic.defined()) {
            update_mapped_texture(st->bridge, st->act_tex, act_mosaic,
                                  CALIPER_CMAP_VIRIDIS, 0.0f, 1.0f);
            st->act_tex_gen = probe_gen;
        }

        // Partition 3-D embedding points by digit class when a fresh probe lands.
        if (probe_gen != 0 && probe_gen != st->emb_plot_gen && !emb_x.empty()) {
            for (int c = 0; c < 10; ++c) {
                st->cls_x[c].clear();
                st->cls_y[c].clear();
                st->cls_z[c].clear();
            }
            double lo[3] = {1e30, 1e30, 1e30};
            double hi[3] = {-1e30, -1e30, -1e30};
            for (size_t i = 0; i < emb_x.size(); ++i) {
                const int c = emb_labels[i];
                if (c < 0 || c > 9) continue;
                st->cls_x[c].push_back(emb_x[i]);
                st->cls_y[c].push_back(emb_y[i]);
                st->cls_z[c].push_back(emb_z[i]);
                const double pt[3] = {emb_x[i], emb_y[i], emb_z[i]};
                for (int k = 0; k < 3; ++k) {
                    lo[k] = std::min(lo[k], pt[k]);
                    hi[k] = std::max(hi[k], pt[k]);
                }
            }
            for (int k = 0; k < 3; ++k) {
                if (hi[k] <= lo[k]) {
                    lo[k] -= 1.0;
                    hi[k] += 1.0;
                }
                const double pad = 0.08 * (hi[k] - lo[k]);
                st->emb_min[k] = lo[k] - pad;
                st->emb_max[k] = hi[k] + pad;
            }
            st->emb_refit = (st->emb_plot_gen == 0) || st->emb_autofit;
            st->emb_plot_gen = probe_gen;
        }

        // 1. Side-top window: controls, status, loss and accuracy curves.
        ImGui::SetNextWindowPos(ImVec2(8, 8), ImGuiCond_FirstUseEver);
        ImGui::SetNextWindowSize(ImVec2(340, 560), ImGuiCond_FirstUseEver);
        caliper::ui::dock_hint(host_, "MNIST: Training", CALIPER_DOCK_SIDE_TOP);
        ImGui::Begin("MNIST: Training");

        ImGui::TextDisabled("device: %s", device_label.c_str());
        if (!st->dataset_found) {
            ImGui::Separator();
            ImGui::TextWrapped("MNIST dataset files were not found.");
            ImGui::Spacing();
            const std::string cache_path = expected_cache_path();
            ImGui::TextWrapped("Expected cache folder:\n%s", cache_path.c_str());
            ImGui::Spacing();
            ImGui::TextWrapped("Download MNIST with:\npython examples/mnist/train.py");
            ImGui::Spacing();
            if (ImGui::Button("Retry cache lookup")) {
                st->data_dir = find_mnist_dir();
                st->dataset_found = !st->data_dir.empty();
                if (st->dataset_found) start_fresh_run();
            }
        } else {
            ImGui::TextWrapped("%s", status.c_str());
            if (test_acc >= 0.0f) {
                ImGui::Text("test accuracy: %.2f%%  (epoch %d)", test_acc, epoch);
            } else if (step > 0) {
                ImGui::Text("batch loss: %.4f  acc: %.1f%%", last_loss, last_acc);
            }

            if (running) {
                const bool is_paused = st->paused.load(std::memory_order_relaxed);
                if (ImGui::Button(is_paused ? "Play" : "Pause"))
                    st->paused.store(!is_paused, std::memory_order_relaxed);
            } else {
                if (ImGui::Button("Play")) start_fresh_run();
            }
            ImGui::SameLine();
            if (ImGui::Button("Restart")) {
                if (running) {
                    st->paused.store(false, std::memory_order_relaxed);
                    st->jobs.request_cancel(st->job_id);
                    st->restart_pending = true;
                } else {
                    start_fresh_run();
                }
            }
            if (running) {
                ImGui::ProgressBar(st->jobs.progress_of(st->job_id), ImVec2(-1, 0));
            }
        }

        ImGui::Checkbox("follow", &st->follow);
        const ImPlotAxisFlags axis_flags = st->follow ? ImPlotAxisFlags_AutoFit : 0;
        const float plot_h = std::clamp(
            (ImGui::GetContentRegionAvail().y - ImGui::GetStyle().ItemSpacing.y) * 0.5f,
            110.0f, 185.0f);

        if (ImPlot::BeginPlot("loss", ImVec2(-1, plot_h))) {
            ImPlot::SetupAxes("step", "NLL", axis_flags, axis_flags);
            if (!loss_hist.empty())
                ImPlot::PlotLine("loss", loss_hist.data(), static_cast<int>(loss_hist.size()));
            ImPlot::EndPlot();
        }
        if (ImPlot::BeginPlot("accuracy", ImVec2(-1, plot_h))) {
            ImPlot::SetupAxes("step", "acc %", axis_flags, 0);
            if (st->follow)
                ImPlot::SetupAxisLimits(ImAxis_Y1, 0.0, 100.0, ImPlotCond_Always);
            if (!acc_hist.empty())
                ImPlot::PlotLine("batch", acc_hist.data(), static_cast<int>(acc_hist.size()));
            if (!test_acc_hist.empty()) {
                ImPlot::PlotLine("test", test_acc_steps.data(), test_acc_hist.data(),
                                 static_cast<int>(test_acc_hist.size()));
            }
            ImPlot::EndPlot();
        }
        ImGui::End();

        // 2. Side-bottom window: fc1 weight histogram.
        ImGui::SetNextWindowPos(ImVec2(8, 580), ImGuiCond_FirstUseEver);
        ImGui::SetNextWindowSize(ImVec2(340, 320), ImGuiCond_FirstUseEver);
        caliper::ui::dock_hint(host_, "MNIST: Weights", CALIPER_DOCK_SIDE_BOTTOM);
        ImGui::Begin("MNIST: Weights");
        ImGui::TextDisabled("fc1.weight distribution (%zu weights)", fc1_weights.size());
        if (ImPlot::BeginPlot("fc1.weight.hist", ImVec2(-1, -1))) {
            ImPlot::SetupAxes("weight", "count",
                              ImPlotAxisFlags_AutoFit, ImPlotAxisFlags_AutoFit);
            if (!fc1_weights.empty()) {
                ImPlot::PlotHistogram("fc1.weight", fc1_weights.data(),
                                      static_cast<int>(fc1_weights.size()), 40);
            }
            ImPlot::EndPlot();
        }
        ImGui::End();

        // 3. Central window: 3-D fc1 embedding cloud + conv1 filters & activations.
        ImGui::SetNextWindowPos(ImVec2(360, 8), ImGuiCond_FirstUseEver);
        ImGui::SetNextWindowSize(ImVec2(920, 892), ImGuiCond_FirstUseEver);
        caliper::ui::dock_hint(host_, "MNIST: Features", CALIPER_DOCK_CENTRAL);
        ImGui::Begin("MNIST: Features");

        if (ImGui::BeginTable("mnist_features_layout", 2,
                              ImGuiTableFlags_Resizable | ImGuiTableFlags_SizingStretchProp)) {
            ImGui::TableSetupColumn("embedding", ImGuiTableColumnFlags_WidthStretch, 1.35f);
            ImGui::TableSetupColumn("mosaics", ImGuiTableColumnFlags_WidthStretch, 1.0f);
            ImGui::TableNextRow();

            // Left column: 3-D fc1 hidden-layer point cloud on 512 fixed test digits.
            ImGui::TableSetColumnIndex(0);
            ImGui::TextDisabled("fc1 3-D embedding (512 fixed test digits)");
            ImGui::SameLine();
            ImGui::Checkbox("auto-fit", &st->emb_autofit);
            ImGui::SameLine();
            if (ImGui::Button("Refit")) st->emb_refit = true;

            const float plot3d_h = std::max(ImGui::GetContentRegionAvail().y, 180.0f);
            if (ImPlot3D::BeginPlot("##fc1_embedding", ImVec2(-1, plot3d_h))) {
                ImPlot3D::SetupAxes("p0", "p1", "p2");
                ImPlot3D::SetupAxesLimits(
                    st->emb_min[0], st->emb_max[0],
                    st->emb_min[1], st->emb_max[1],
                    st->emb_min[2], st->emb_max[2],
                    st->emb_refit ? ImPlot3DCond_Always : ImPlot3DCond_Once);
                st->emb_refit = false;
                for (int c = 0; c < 10; ++c) {
                    if (st->cls_x[c].empty()) continue;
                    char label[8];
                    std::snprintf(label, sizeof label, "%d", c);
                    ImPlot3D::PlotScatter(
                        label, st->cls_x[c].data(), st->cls_y[c].data(),
                        st->cls_z[c].data(), static_cast<int>(st->cls_x[c].size()),
                        ImPlot3DSpec(ImPlot3DProp_MarkerFillColor, kClassColors[c],
                                     ImPlot3DProp_MarkerLineColor, kClassColors[c],
                                     ImPlot3DProp_MarkerSize, 3.0f));
                }
                ImPlot3D::EndPlot();
            }

            // Right column: conv1 filters (RdBu) and conv1 ReLU activations (Viridis).
            ImGui::TableSetColumnIndex(1);
            const float col_w = std::max(ImGui::GetContentRegionAvail().x, 64.0f);

            ImGui::SeparatorText("conv1 filters (10 x 5x5, RdBu)");
            if (filter_abs_max > 0.0f) {
                ImGui::TextDisabled("range [-%.3f, +%.3f]", filter_abs_max, filter_abs_max);
            }
            if (st->bridge && st->filter_tex && filter_mosaic.defined()) {
                const float aspect =
                    static_cast<float>(filter_mosaic.size(0)) /
                    static_cast<float>(filter_mosaic.size(1));
                ImGui::Image(caliper::Bridge::imtex(st->filter_tex),
                             ImVec2(col_w, std::max(col_w * aspect, 48.0f)));
            } else if (!st->bridge) {
                ImGui::TextDisabled("tensor_bridge.v1 absent — ImPlot heatmap fallback");
                draw_heatmap_fallback("##conv1_filters_fb", filter_mosaic,
                                      ImPlotColormap_RdBu, -1.0, 1.0, 180.0f);
            } else {
                ImGui::TextDisabled(st->dataset_found
                                        ? "conv1 filters appear once training starts"
                                        : "cache MNIST to view live conv1 filters");
            }

            ImGui::Spacing();
            if (digit0_label >= 0) {
                char hdr[96];
                std::snprintf(hdr, sizeof hdr,
                              "conv1 ReLU activations (test #0, digit %d, Viridis)",
                              digit0_label);
                ImGui::SeparatorText(hdr);
            } else {
                ImGui::SeparatorText("conv1 ReLU activations (test #0, Viridis)");
            }
            if (act_max > 0.0f) {
                ImGui::TextDisabled("range [0.000, %.3f]", act_max);
            }
            if (st->bridge && st->act_tex && act_mosaic.defined()) {
                const float aspect =
                    static_cast<float>(act_mosaic.size(0)) /
                    static_cast<float>(act_mosaic.size(1));
                ImGui::Image(caliper::Bridge::imtex(st->act_tex),
                             ImVec2(col_w, std::max(col_w * aspect, 48.0f)));
            } else if (!st->bridge) {
                ImGui::TextDisabled("tensor_bridge.v1 absent — ImPlot heatmap fallback");
                draw_heatmap_fallback("##conv1_act_fb", act_mosaic,
                                      ImPlotColormap_Viridis, 0.0, 1.0, 180.0f);
            } else {
                ImGui::TextDisabled(st->dataset_found
                                        ? "conv1 activations appear once training starts"
                                        : "cache MNIST to view live conv1 activations");
            }

            ImGui::EndTable();
        }
        ImGui::End();
    }
    // --8<-- [end:views]

    // --8<-- [start:cleanup]
    void on_cleanup() override {
        auto* st = s_.get();
        if (st->job_id != 0) {
            st->paused.store(false, std::memory_order_relaxed);
            st->jobs.request_cancel(st->job_id);
            for (int i = 0; i < 5000 && st->jobs.is_running(st->job_id); ++i)
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
            st->job_id = 0;
        }
        if (st->filter_tex) {
            st->bridge.release_texture(st->filter_tex);
            st->filter_tex = 0;
        }
        if (st->act_tex) {
            st->bridge.release_texture(st->act_tex);
            st->act_tex = 0;
        }
        if (host_) host_->log_info("mnist.on_cleanup");
    }
    // --8<-- [end:cleanup]

private:
    void start_fresh_run() {
        auto* st = s_.get();
        if (!st->dataset_found) return;
        st->paused.store(false, std::memory_order_relaxed);
        {
            std::lock_guard<std::mutex> lk(st->mtx);
            st->epoch = 0;
            st->step = 0;
            st->last_loss = 0.0f;
            st->last_acc = 0.0f;
            st->test_acc = -1.0f;
            st->loss_hist.clear();
            st->acc_hist.clear();
            st->test_acc_steps.clear();
            st->test_acc_hist.clear();
            st->status = "starting MNIST training...";
        }
        st->job_id = st->jobs.submit("mnist: train CNN", &train_job, st);
    }

    caliper::Host* host_ = nullptr;
    std::unique_ptr<MnistState> s_;
};

} // namespace mnist_instrument

CALIPER_INSTRUMENT(mnist_instrument::MnistInstrument,
    .id       = "ai.caliperapp.mnist",
    .version  = "0.1.0",
    .name     = "MNIST Example",
    .summary  = "Canonical ML instrument: the PyTorch C++ MNIST CNN trained live with LibTorch.",
    .tag      = "Demo",
    .services = {CALIPER_UI_V1, CALIPER_LOG_V1, CALIPER_JOBS_V1,
                 CALIPER_DEVICE_V1})
