#include "training_lab_tab.h"

#include <imgui.h>
#include <implot.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <fstream>

// Static, non-interactive plots: no pan/zoom/box-select, no context menus.
static constexpr ImPlotFlags kLockedPlot =
    ImPlotFlags_NoInputs | ImPlotFlags_NoMenus | ImPlotFlags_NoBoxSelect |
    ImPlotFlags_NoMouseText;

// ---------------------------------------------------------------------------
// local helpers
// ---------------------------------------------------------------------------
namespace {

// Read a float32 .bin (int32 ndim, int32 dims..., data) as written by the
// golden export. Used to load reference (ghost) kernels.
torch::Tensor load_bin(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) return {};
    int32_t ndim = 0;
    f.read(reinterpret_cast<char*>(&ndim), 4);
    if (ndim <= 0 || ndim > 8) return {};
    std::vector<int64_t> shape(ndim);
    int64_t n = 1;
    for (int i = 0; i < ndim; ++i) {
        int32_t d = 0;
        f.read(reinterpret_cast<char*>(&d), 4);
        shape[i] = d;
        n *= d;
    }
    std::vector<float> data(static_cast<size_t>(n));
    f.read(reinterpret_cast<char*>(data.data()), n * sizeof(float));
    if (!f) return {};
    return torch::from_blob(data.data(), shape, torch::kFloat32).clone();
}

}  // namespace

// ---------------------------------------------------------------------------
TrainingLabTab::TrainingLabTab()
    : data_dir_("/Users/ahmed/PycharmProjects/repnet/data/seniordesign_upload"),
      ref_dir_("applets/repnet_demo/tests/golden/state_dict") {}

TrainingLabTab::~TrainingLabTab() {
    engine_.reset();  // joins training thread
    if (load_thread_.joinable()) load_thread_.join();
}

// ---------------------------------------------------------------------------
void TrainingLabTab::start_load() {
    if (loading_.load()) return;
    if (load_thread_.joinable()) load_thread_.join();
    loaded_.store(false);
    load_error_.clear();
    loading_.store(true);
    load_done_.store(0);
    load_total_.store(0);
    std::string dir = data_dir_;
    int split_i = split_i_;
    load_thread_ = std::thread([this, dir, split_i] {
        try {
            auto ds = std::make_shared<tdata::Dataset>(
                tdata::load_and_preprocess(dir, [this](int d, int t) {
                    load_done_.store(d);
                    load_total_.store(t);
                }));
            tdata::Split sp = tdata::make_split(*ds, split_i);
            {
                std::lock_guard<std::mutex> lk(ds_mutex_);
                dataset_ = ds;
                split_ = sp;
            }
            loaded_.store(true);
        } catch (const std::exception& e) {
            load_error_ = e.what();
        }
        loading_.store(false);
    });
}

void TrainingLabTab::load_reference() {
    torch::Tensor w = load_bin(ref_dir_ + "/backbone__0__weight.bin");  // (16,1,31)
    if (w.defined() && w.dim() == 3) {
        ref_kernels_ = w.squeeze(1).clone();  // (16,31)
        ref_loaded_ = true;
    }
}

void TrainingLabTab::start_training() {
    std::shared_ptr<tdata::Dataset> ds;
    tdata::Split sp;
    {
        std::lock_guard<std::mutex> lk(ds_mutex_);
        ds = dataset_;
        sp = split_;
    }
    if (!ds || sp.train.empty()) return;
    if (!ref_loaded_) load_reference();

    auto pick = [&](const std::vector<int>& idx, torch::Tensor& X,
                    std::vector<int>& y) {
        torch::Tensor it = torch::tensor(
            std::vector<int64_t>(idx.begin(), idx.end()), torch::kLong);
        X = ds->X.index_select(0, it).clone();
        y.clear();
        for (int i : idx) y.push_back(ds->y[i]);
    };
    torch::Tensor Xtr, Xva;
    std::vector<int> ytr, yva;
    pick(sp.train, Xtr, ytr);
    pick(sp.val, Xva, yva);

    engine_ = std::make_unique<TrainEngine>();
    TrainEngine::Config cfg;
    cfg.max_epochs = max_epochs_;
    cfg.T_max = max_epochs_;
    cfg.patience = max_epochs_;  // run full schedule for the demo
    cfg.seed = static_cast<uint32_t>(split_i_ * 7 + 1000 + 2);
    cfg.augment = augment_;
    cfg.mixup = mixup_;
    engine_->start(std::move(Xtr), std::move(ytr), std::move(Xva), std::move(yva), cfg);
    gradcam_epoch_cached_ = -1;
}

void TrainingLabTab::reset() {
    engine_.reset();  // joins
}

// ---------------------------------------------------------------------------
void TrainingLabTab::draw() {
    draw_controls();
    ImGui::Separator();

    if (loading_.load()) {
        int d = load_done_.load(), t = load_total_.load();
        ImGui::Text("Loading dataset + preprocessing (C++ DSP)...  %d / %d", d, t);
        ImGui::ProgressBar(t > 0 ? float(d) / float(t) : 0.0f);
        return;
    }
    if (!load_error_.empty()) {
        ImGui::TextColored(ImVec4(1, 0.4f, 0.4f, 1), "Load error: %s",
                           load_error_.c_str());
    }
    if (!loaded_.load()) {
        ImGui::TextDisabled(
            "Load the split-17 dataset to begin. The full C++ pipeline (DSP -> "
            "StratifiedGroupKFold -> PerLeadCNN) reproduces test AUROC 0.7793.");
        return;
    }

    TrainSnapshot s;
    if (engine_) s = engine_->snapshot();

    // Two-column layout: left = metrics + saliency, right = kernels.
    float avail_w = ImGui::GetContentRegionAvail().x;
    float left_w = avail_w * 0.52f;
    ImGui::BeginChild("##tl_left", ImVec2(left_w, 0), false);
    draw_metrics(s);
    ImGui::Separator();
    draw_saliency(s);
    ImGui::EndChild();
    ImGui::SameLine();
    ImGui::BeginChild("##tl_right", ImVec2(0, 0), false);
    draw_kernels(s);
    ImGui::EndChild();
}

void TrainingLabTab::draw_controls() {
    ImGui::PushItemWidth(420);
    char buf[512];
    std::snprintf(buf, sizeof(buf), "%s", data_dir_.c_str());
    if (ImGui::InputText("data dir", buf, sizeof(buf)))
        data_dir_ = buf;
    ImGui::PopItemWidth();

    ImGui::BeginDisabled(loading_.load());
    if (ImGui::Button(loaded_.load() ? "Reload dataset" : "Load dataset"))
        start_load();
    ImGui::EndDisabled();

    ImGui::SameLine();
    ImGui::BeginDisabled(!loaded_.load());
    bool training = engine_ && engine_->is_running();
    if (!training) {
        if (ImGui::Button("Start training"))
            start_training();
    } else {
        if (ImGui::Button("Pause")) engine_->pause();
        ImGui::SameLine();
        if (ImGui::Button("Resume")) engine_->resume();
    }
    ImGui::SameLine();
    if (ImGui::Button("Reset")) reset();
    ImGui::EndDisabled();

    ImGui::SameLine();
    ImGui::Checkbox("ghost (reference) kernels", &show_ghost_);

    ImGui::SetNextItemWidth(120);
    ImGui::SliderInt("epochs", &max_epochs_, 10, 120);
    ImGui::SameLine();
    ImGui::Checkbox("augment", &augment_);
    ImGui::SameLine();
    ImGui::Checkbox("mixup", &mixup_);
    ImGui::SameLine();
    ImGui::TextDisabled("| split 17 (seed %d)", split_i_ * 7 + 1000);
}

void TrainingLabTab::draw_metrics(const TrainSnapshot& s) {
    ImGui::Text("epoch %d / %d   loss %.4f   val AUROC %.4f   best %.4f (ep %d)   lr %.2e   [%s]",
                s.epoch, s.max_epochs, s.train_loss, s.val_auroc, s.best_val_auroc,
                s.best_epoch + 1, s.lr, s.device.c_str());

    if (ImPlot::BeginPlot("##metrics", ImVec2(-1, 240), kLockedPlot)) {
        ImPlot::SetupAxes("epoch", "train loss", ImPlotAxisFlags_AutoFit,
                          ImPlotAxisFlags_AutoFit);
        ImPlot::SetupAxis(ImAxis_Y2, "val AUROC",
                          ImPlotAxisFlags_AuxDefault | ImPlotAxisFlags_AutoFit);
        ImPlot::SetupAxisLimits(ImAxis_Y2, 0.4, 0.85, ImGuiCond_Always);

        if (!s.loss_history.empty()) {
            std::vector<float> xs(s.loss_history.size());
            for (size_t i = 0; i < xs.size(); ++i) xs[i] = float(i + 1);
            ImPlot::SetAxes(ImAxis_X1, ImAxis_Y1);
            ImPlot::PlotLine("train loss", xs.data(), s.loss_history.data(),
                             (int)xs.size());
            ImPlot::SetAxes(ImAxis_X1, ImAxis_Y2);
            ImPlot::PlotLine("val AUROC", xs.data(), s.auroc_history.data(),
                             (int)s.auroc_history.size());
            // reference target line.
            double tx[2] = {1, (double)xs.size()};
            double ty[2] = {ref_auroc_, ref_auroc_};
            ImPlotSpec ref_spec;
            ref_spec.LineColor = ImVec4(0.6f, 0.6f, 0.6f, 0.8f);
            ImPlot::PlotLine("ref 0.7793", tx, ty, 2, ref_spec);
        }
        ImPlot::EndPlot();
    }
}

void TrainingLabTab::draw_kernels(const TrainSnapshot& s) {
    ImGui::Text("stage-1 kernels (16 filters x 31 taps) adapting");
    ImGui::TextDisabled("solid = live   dashed grey = reference (trained)");
    if (!s.stage1_kernels.defined()) {
        ImGui::TextDisabled("(start training to watch the filters form)");
        return;
    }
    auto K = s.stage1_kernels.to(torch::kCPU).contiguous();  // (16,31)
    int nf = (int)K.size(0), kl = (int)K.size(1);
    auto acc = K.accessor<float, 2>();

    bool have_ref = show_ghost_ && ref_kernels_.defined() &&
                    ref_kernels_.size(0) == nf && ref_kernels_.size(1) == kl;
    torch::Tensor R;
    if (have_ref) R = ref_kernels_.to(torch::kCPU).contiguous();

    float avail = ImGui::GetContentRegionAvail().x;
    int cols = 4;
    float cell = (avail - 8 * cols) / cols;
    if (cell < 60) cell = 60;
    std::vector<float> xs(kl), ys(kl), rys(kl);
    for (int i = 0; i < kl; ++i) xs[i] = (float)i;

    for (int f = 0; f < nf; ++f) {
        for (int i = 0; i < kl; ++i) ys[i] = acc[f][i];
        char id[32];
        std::snprintf(id, sizeof(id), "##k%d", f);
        if (ImPlot::BeginPlot(id, ImVec2(cell, cell * 0.7f),
                              ImPlotFlags_NoLegend | kLockedPlot)) {
            ImPlot::SetupAxes(nullptr, nullptr,
                              ImPlotAxisFlags_NoDecorations,
                              ImPlotAxisFlags_NoDecorations | ImPlotAxisFlags_AutoFit);
            ImPlot::SetupAxisLimits(ImAxis_X1, 0, kl - 1, ImGuiCond_Always);
            if (have_ref) {
                auto racc = R.accessor<float, 2>();
                for (int i = 0; i < kl; ++i) rys[i] = racc[f][i];
                ImPlotSpec gs;
                gs.LineColor = ImVec4(0.6f, 0.6f, 0.6f, 0.6f);
                ImPlot::PlotLine("ref", xs.data(), rys.data(), kl, gs);
            }
            ImPlotSpec ls;
            ls.LineColor = ImVec4(0.25f, 0.8f, 1.0f, 1.0f);
            ls.LineWeight = 1.5f;
            ImPlot::PlotLine("live", xs.data(), ys.data(), kl, ls);
            ImPlot::EndPlot();
        }
        if ((f % cols) != cols - 1) ImGui::SameLine();
    }
}

void TrainingLabTab::draw_saliency(const TrainSnapshot& s) {
    ImGui::Text("grad-cam saliency on a pinned positive ECG (12 leads)");
    ImGui::TextDisabled("flat/noisy early -> concentrates on diagnostic regions");
    if (!s.gradcam.defined()) {
        ImGui::TextDisabled("(start training to watch saliency emerge from noise)");
        return;
    }
    auto cam = s.gradcam.to(torch::kCPU).contiguous();  // (12, T')
    int rows = (int)cam.size(0), colsT = (int)cam.size(1);
    float vmax = cam.max().item<float>();
    if (vmax < 1e-8f) vmax = 1.0f;

    ImPlot::PushColormap(ImPlotColormap_Hot);
    if (ImPlot::BeginPlot("##gradcam", ImVec2(-1, 220),
                          ImPlotFlags_NoLegend | kLockedPlot)) {
        ImPlot::SetupAxes("time", "lead",
                          ImPlotAxisFlags_NoDecorations,
                          ImPlotAxisFlags_NoGridLines);
        ImPlot::PlotHeatmap("cam", cam.data_ptr<float>(), rows, colsT, 0.0, vmax,
                            nullptr, ImPlotPoint(0, 0), ImPlotPoint(1, 1));
        ImPlot::EndPlot();
    }
    ImPlot::PopColormap();
}
