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

    // Three columns: example scrubber sidebar | metrics+saliency | kernels.
    float avail_w = ImGui::GetContentRegionAvail().x;
    float side_w = 230.0f;
    float center_w = (avail_w - side_w) * 0.56f;
    ImGui::BeginChild("##tl_side", ImVec2(side_w, 0), true);
    draw_example_picker(s);
    ImGui::EndChild();
    ImGui::SameLine();
    ImGui::BeginChild("##tl_center", ImVec2(center_w, 0), false);
    draw_metrics(s);
    ImGui::Separator();
    draw_saliency(s);
    ImGui::EndChild();
    ImGui::SameLine();
    ImGui::BeginChild("##tl_right", ImVec2(0, 0), false);
    draw_kernels(s);
    ImGui::EndChild();
}

// 12-lead names for the lead selector / titles.
static const char* kLeadNames[12] = {"I",  "II", "III", "aVR", "aVL", "aVF",
                                      "V1", "V2", "V3",  "V4",  "V5",  "V6"};

void TrainingLabTab::draw_example_picker(const TrainSnapshot& s) {
    ImGui::TextUnformatted("Saliency example");
    ImGui::Separator();

    int n = s.val_count > 0 ? s.val_count : (engine_ ? engine_->val_count() : 0);
    if (n <= 0) {
        ImGui::TextDisabled("start training to\npick an example");
        return;
    }
    // Adopt the engine's chosen example the first time.
    if (viz_index_ < 0) viz_index_ = s.viz_index >= 0 ? s.viz_index : 0;

    ImGui::Text("held-out example");
    ImGui::SetNextItemWidth(-1);
    bool changed = ImGui::SliderInt("##ex", &viz_index_, 0, n - 1);
    if (ImGui::SmallButton("<- prev")) { viz_index_ = std::max(0, viz_index_ - 1); changed = true; }
    ImGui::SameLine();
    if (ImGui::SmallButton("next ->")) { viz_index_ = std::min(n - 1, viz_index_ + 1); changed = true; }
    if (changed && engine_) engine_->set_viz_index(viz_index_);

    ImGui::Spacing();
    // True label + prediction for the shown example.
    bool is_pe = s.pinned_label == 1;
    ImGui::Text("true label:");
    ImGui::SameLine();
    ImGui::TextColored(is_pe ? ImVec4(1.0f, 0.55f, 0.35f, 1) : ImVec4(0.5f, 0.8f, 1.0f, 1),
                       "%s", is_pe ? "Preeclampsia" : "Normal");
    ImGui::Text("P(PE) = %.3f", s.viz_prob);
    bool correct = (s.viz_prob >= 0.5f) == is_pe;
    ImGui::TextColored(correct ? ImVec4(0.4f, 0.9f, 0.4f, 1) : ImVec4(0.95f, 0.4f, 0.4f, 1),
                       "%s", correct ? "correct" : "wrong");

    ImGui::Spacing();
    ImGui::Separator();
    ImGui::TextUnformatted("Lead");
    ImGui::SetNextItemWidth(-1);
    if (ImGui::BeginCombo("##lead", kLeadNames[viz_lead_])) {
        for (int i = 0; i < 12; ++i) {
            if (ImGui::Selectable(kLeadNames[i], viz_lead_ == i)) viz_lead_ = i;
        }
        ImGui::EndCombo();
    }
    ImGui::Spacing();
    ImGui::TextDisabled(
        "Scrub examples and leads.\nThe waveform is the actual\nECG; the heat band behind\nit is grad-cam saliency.\nUpdates live (even paused).");
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
    if (!s.gradcam.defined() || !s.pinned_input.defined()) {
        ImGui::Text("grad-cam saliency over the ECG waveform");
        ImGui::TextDisabled("(start training to watch saliency emerge from noise)");
        return;
    }
    auto cam = s.gradcam.to(torch::kCPU).contiguous();    // (12, T) aligned to wave
    auto wave = s.pinned_input.to(torch::kCPU).contiguous();  // (12, T)
    int rows = (int)cam.size(0);
    int T = (int)cam.size(1);
    float vmax = cam.max().item<float>();
    if (vmax < 1e-8f) vmax = 1.0f;
    const float fs = 250.0f;
    const double duration = T / fs;

    int lead = std::max(0, std::min(rows - 1, viz_lead_));
    ImGui::Text("Lead %s — saliency over the actual waveform   (P(PE)=%.3f)",
                kLeadNames[lead], s.viz_prob);

    // Build the time axis + the selected lead's waveform and saliency row.
    std::vector<float> xs(T), wl(T), sl(T);
    auto wacc = wave.accessor<float, 2>();
    auto cacc = cam.accessor<float, 2>();
    float ylo = 1e30f, yhi = -1e30f;
    for (int t = 0; t < T; ++t) {
        xs[t] = t / fs;
        wl[t] = wacc[lead][t];
        sl[t] = cacc[lead][t];
        ylo = std::min(ylo, wl[t]);
        yhi = std::max(yhi, wl[t]);
    }
    float margin = 0.1f * (yhi - ylo) + 1e-3f;
    ylo -= margin;
    yhi += margin;

    // Main overlay: saliency heat band behind the waveform line.
    ImPlot::PushColormap(ImPlotColormap_Hot);
    if (ImPlot::BeginPlot("##wave_sal", ImVec2(-1, 230),
                          ImPlotFlags_NoLegend | kLockedPlot)) {
        ImPlot::SetupAxes("time (s)", "amplitude (z)", 0,
                          ImPlotAxisFlags_NoGridLines);
        ImPlot::SetupAxisLimits(ImAxis_X1, 0, duration, ImGuiCond_Always);
        ImPlot::SetupAxisLimits(ImAxis_Y1, ylo, yhi, ImGuiCond_Always);
        // Background: 1xT saliency heatmap spanning the full y-range.
        ImPlot::PlotHeatmap("##band", sl.data(), 1, T, 0.0, vmax, nullptr,
                            ImPlotPoint(0, ylo), ImPlotPoint(duration, yhi));
        // Foreground: the ECG trace.
        ImPlotSpec ws;
        ws.LineColor = ImVec4(0.95f, 0.97f, 1.0f, 1.0f);
        ws.LineWeight = 1.3f;
        ImPlot::PlotLine("ecg", xs.data(), wl.data(), T, ws);
        ImPlot::EndPlot();
    }
    ImPlot::PopColormap();

    // Compact 12-lead overview so all leads are visible at once.
    ImGui::TextDisabled("all 12 leads (saliency)");
    ImPlot::PushColormap(ImPlotColormap_Hot);
    if (ImPlot::BeginPlot("##gradcam_all", ImVec2(-1, 150),
                          ImPlotFlags_NoLegend | kLockedPlot)) {
        ImPlot::SetupAxes("time", "lead",
                          ImPlotAxisFlags_NoDecorations,
                          ImPlotAxisFlags_NoGridLines);
        ImPlot::PlotHeatmap("cam", cam.data_ptr<float>(), rows, T, 0.0, vmax,
                            nullptr, ImPlotPoint(0, 0), ImPlotPoint(1, 1));
        ImPlot::EndPlot();
    }
    ImPlot::PopColormap();
}
