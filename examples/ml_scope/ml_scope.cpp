// ============================================================================
// MLScope — the ML exemplar (PLATFORM.md §17 Phase 2, step 1 of the ratified
// sequencing). Shows the idioms of ML on the platform:
//   ML-EXEMPLAR 1 — never train on the frame thread: submit to caliper.jobs.v1
//     and poll cancelled() in the epoch loop (cooperative cancel).
//   ML-EXEMPLAR 2 — the host picks the device (caliper.device.v1); the applet
//     maps the KIND to its framework: METAL -> torch::kMPS here.
//   ML-EXEMPLAR 3 — publish training state to the UI under a mutex; the frame
//     reads a copy. (repnet's snapshot pattern, minimal form.)
//   ML-EXEMPLAR 4 — deliberately NO weight-matrix visualization yet: that is
//     tensor_bridge.v1's job (Plan 2C). A CPU-staged copy here would teach the
//     exact pattern the platform exists to delete.
// ============================================================================
#include <caliper/caliper.hpp>
#include <imgui.h>
#include <implot.h>
#include <torch/torch.h>

#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <mutex>
#include <thread>
#include <vector>

namespace {
constexpr int kEpochs = 300;
constexpr int kN = 512;   // two-moons points

// Synthetic two-moons, generated on CPU (deterministic under manual_seed —
// device RNGs vary), then moved to the training device.
std::pair<torch::Tensor, torch::Tensor> make_moons(torch::Device dev) {
    auto t = torch::rand({kN}) * M_PI;
    auto x0 = torch::stack({torch::cos(t), torch::sin(t)}, 1);
    auto x1 = torch::stack({1.0f - torch::cos(t), 0.5f - torch::sin(t)}, 1);
    auto X = torch::cat({x0, x1}, 0) + torch::randn({2 * kN, 2}) * 0.08f;
    auto y = torch::cat({torch::zeros({kN}), torch::ones({kN})}, 0)
                 .to(torch::kLong);
    return {X.to(dev), y.to(dev)};
}
} // namespace

class MLScope final : public caliper::Applet {
public:
    bool on_init(caliper::Host& host) override {
        host_ = &host;
        jobs_ = caliper::Jobs(host);          // required -> present (manifest)
        device_ = caliper::Device::query(host);
        host.log_info("ml-scope: on_init");
        return true;
    }

    void on_frame(const caliper::Frame&) override {
        ImGui::SetNextWindowPos({60, 80}, ImGuiCond_FirstUseEver);
        ImGui::SetNextWindowSize({560, 420}, ImGuiCond_FirstUseEver);
        ImGui::Begin("MLScope");

        // ML-EXEMPLAR 2 — the negotiated device, and what torch calls it.
        ImGui::TextDisabled("device: %s (%s)  |  free mem hint: %.1f GB",
                            device_.name,
                            device_.kind == CALIPER_DEV_METAL ? "METAL->torch MPS"
                            : device_.kind == CALIPER_DEV_CUDA ? "CUDA"
                                                               : "CPU",
                            device_.free_memory_hint / 1073741824.0);

        const bool running = job_id_ != 0 && jobs_.is_running(job_id_);
        if (!running) {
            if (ImGui::Button("start training")) start_training();
        } else {
            if (ImGui::Button("cancel")) jobs_.request_cancel(job_id_);
            ImGui::SameLine();
            ImGui::ProgressBar(jobs_.progress_of(job_id_), {-1, 0});
        }

        // ML-EXEMPLAR 3 — read a copy of worker-published state.
        std::vector<float> loss;
        {
            std::lock_guard<std::mutex> lk(state_mutex_);
            loss = loss_history_;
        }
        if (ImPlot::BeginPlot("loss", {-1, 260})) {
            ImPlot::SetupAxes("epoch", "NLL");
            if (!loss.empty())
                ImPlot::PlotLine("train", loss.data(), (int)loss.size());
            ImPlot::EndPlot();
        }
        ImGui::TextWrapped("Weight-matrix visualization arrives with "
                           "caliper.tensor_bridge.v1 — GPU-resident, no CPU "
                           "staging. Watch this space (Plan 2C).");
        ImGui::End();
    }

    void on_cleanup() override {
        if (job_id_ != 0) {
            jobs_.request_cancel(job_id_);
            // ML-EXEMPLAR 1b — `user` (this object) must outlive the job
            // (jobs_v1.h contract): wait for the worker to exit BEFORE
            // destroy() frees us. Cancel is honored <=100 ms by tested
            // contract, so this bounded wait cannot hang teardown.
            for (int i = 0; i < 300 && jobs_.is_running(job_id_); i++)
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
        if (host_) host_->log_info("ml-scope: on_cleanup");
    }

private:
    void start_training() {
        {
            std::lock_guard<std::mutex> lk(state_mutex_);
            loss_history_.clear();
        }
        // ML-EXEMPLAR 1 — static trampoline + this: the raw C job contract.
        job_id_ = jobs_.submit("ml_scope: train MLP", &MLScope::train_job, this);
        if (job_id_ == 0 && host_) host_->log_error("ml-scope: submit failed");
    }

    static void train_job(void* user, const CaliperJobControl* ctl) {
        auto* self = static_cast<MLScope*>(user);
        torch::Device dev = self->device_.kind == CALIPER_DEV_METAL &&
                                    torch::hasMPS()
                                ? torch::Device(torch::kMPS)
                                : torch::Device(torch::kCPU);
        torch::manual_seed(7);
        auto [X, y] = make_moons(dev);
        auto model = torch::nn::Sequential(
            torch::nn::Linear(2, 16), torch::nn::ReLU(),
            torch::nn::Linear(16, 16), torch::nn::ReLU(),
            torch::nn::Linear(16, 2));
        model->to(dev);
        torch::optim::Adam opt(model->parameters(),
                               torch::optim::AdamOptions(1e-2));
        for (int epoch = 0; epoch < kEpochs; epoch++) {
            if (ctl->cancelled(ctl)) break;         // ML-EXEMPLAR 1
            opt.zero_grad();
            auto out = torch::log_softmax(model->forward(X), 1);
            auto loss = torch::nll_loss(out, y);
            loss.backward();
            opt.step();
            float l = loss.item<float>();
            {
                std::lock_guard<std::mutex> lk(self->state_mutex_);
                self->loss_history_.push_back(l);
            }
            char msg[64];
            std::snprintf(msg, sizeof msg, "epoch %d/%d  loss %.4f", epoch + 1,
                          kEpochs, l);
            ctl->progress(ctl, (float)(epoch + 1) / kEpochs, msg);
        }
    }

    caliper::Host* host_ = nullptr;
    caliper::Jobs jobs_;
    caliper::Device device_;
    uint64_t job_id_ = 0;
    std::mutex state_mutex_;
    std::vector<float> loss_history_;
};

CALIPER_APPLET(MLScope,
    .id       = "dev.caliper.ml-scope",
    .version  = "0.1.0",
    .name     = "MLScope",
    .summary  = "ML exemplar: trains a tiny MLP off the frame thread via "
                "caliper.jobs.v1, device-negotiated, with live loss. Weight "
                "visualization arrives with tensor_bridge (Phase 2C).",
    .tag      = "ML",
    .services = {CALIPER_UI_V1, CALIPER_LOG_V1, CALIPER_JOBS_V1,
                 CALIPER_DEVICE_V1})
